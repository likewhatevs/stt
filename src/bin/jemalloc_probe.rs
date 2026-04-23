//! Standalone jemalloc per-thread counter probe.
//!
//! Reads the `thread_allocated` / `thread_deallocated` TLS counters
//! out of a running jemalloc-linked process. The counters are
//! maintained unconditionally on jemalloc's alloc/dalloc fast + slow
//! paths (see jemalloc_internal_inlines_c.h:277, 574 and
//! thread_event.h:117-119), so attaching late does not miss prior
//! allocations — the reading is cumulative from thread creation.
//!
//! Two entry points:
//! - `--pid <PID>`: external-pid mode. Attaches to every thread in
//!   the target process via ptrace, reads each thread's TSD counters
//!   through `process_vm_readv`, detaches. DWARF is resolved against
//!   the target's `/proc/<pid>/exe`, so the target ELF must ship
//!   with debuginfo.
//! - `--self-test <BYTES>`: same-process closed-loop mode. Spawns an
//!   allocator thread inside the probe itself, reads its TSD counter
//!   via `process_vm_readv` on the probe's own pid (no ptrace needed
//!   for same-process reads under the same-uid kernel check), and
//!   exits 0 iff the observed counter is at least `BYTES`. DWARF is
//!   resolved against the probe's own ELF (which is never stripped),
//!   so external targets' debuginfo state is irrelevant in this mode.
//!
//! Scope for v1:
//! - Linux x86_64 only.
//! - Static-linked jemalloc only (symbol lives in the main executable's
//!   static TLS image).
//! - External-pid mode requires DWARF debuginfo on the target ELF and
//!   CAP_SYS_PTRACE / root / same-uid-as-target; self-test mode
//!   requires neither (DWARF is on the probe; the target is self).
//!
//! Mechanism (external-pid, per target thread):
//! 1. `ptrace(PTRACE_SEIZE)` + `ptrace(PTRACE_INTERRUPT)` to stop.
//! 2. `ptrace(PTRACE_GETREGSET, NT_PRSTATUS)` to read `fs_base`.
//! 3. `process_vm_readv` 24 bytes at the computed TLS address to read
//!    `thread_allocated` + `thread_allocated_next_event_fast` +
//!    `thread_deallocated` in one syscall while the thread is stopped.
//! 4. `ptrace(PTRACE_DETACH)`.
//!
//! Mechanism (self-test): arch_prctl(ARCH_GET_FS) in the allocator
//! thread instead of PTRACE_GETREGSET; `process_vm_readv` against
//! self_pid; no ptrace attach/detach.
//!
//! Address math (Variant II TLS, x86_64):
//!   addr(tsd_tls) = fs_base - round_up(PT_TLS.p_memsz, PT_TLS.p_align) + st_value
//!   addr(field)   = addr(tsd_tls) + offsetof(tsd_s, field)

// Link jemalloc as the global allocator so the probe binary itself
// carries the `tsd_tls` symbol. Required by the `--self-test` mode
// which resolves tsd_s field offsets against `/proc/self/maps` /
// `/proc/self/exe` — without this, the probe uses glibc malloc and
// has no jemalloc TLS to probe (self-test fails with "ELF has no
// PT_TLS segment"). Matches the global-allocator declaration in
// src/bin/ktstr.rs and src/bin/cargo-ktstr.rs.
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs;
use std::io::IoSliceMut;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use gimli::{AttributeValue, EndianSlice, LittleEndian, Reader, Unit};
use goblin::elf::Elf;
use nix::sys::ptrace;
use nix::sys::ptrace::Options;
use nix::sys::ptrace::regset::NT_PRSTATUS;
use nix::sys::signal::{SigHandler, Signal, signal};
use nix::sys::uio::{RemoteIoVec, process_vm_readv};
use nix::sys::wait::{WaitStatus, waitpid};
use nix::unistd::Pid;
use serde::Serialize;

/// Wire schema version emitted in every [`ProbeOutput`] JSON body.
///
/// **Additive-safe policy**: adding a new always-emitted field or a
/// new optional field (`#[serde(skip_serializing_if = ...)]`) does
/// not require a bump — consumers parsing with serde's default
/// ignore-unknown-fields behavior absorb the new field without
/// semantic change. Only **field removals**, **field renames**, or
/// **semantic changes to existing fields** (value shape, unit,
/// range) trigger a version increment. This keeps the rolling
/// enrichment cadence (per-thread comm, timestamp, error_kind, etc.)
/// from generating spurious version churn.
const SCHEMA_VERSION: u32 = 1;

/// Capture the current wall-clock as Unix epoch seconds. `unwrap_or(0)`
/// handles the impossible pre-epoch-clock case defensively — KVM
/// guests under kvm-clock or NTP always resolve post-1970, so the
/// zero is a never-fires safety net rather than a real fallback.
/// Factored so `run()`, `run_self_test()`, and any future probe-
/// output site reach for the same helper instead of re-typing the
/// `SystemTime::now().duration_since(UNIX_EPOCH)...` chain.
fn now_unix_sec() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Candidate symbol names for jemalloc's per-thread state.
///
/// jemalloc's build may apply a prefix via `--with-jemalloc-prefix`.
/// Observed prefixes:
/// - bare `tsd_tls` (unprefixed builds, older jemalloc versions).
/// - `je_tsd_tls` (default `--with-jemalloc-prefix=je_`).
/// - `_rjem_je_tsd_tls` (what tikv-jemallocator-sys bakes in so the
///   Rust global-allocator's jemalloc cannot collide with system
///   libc malloc symbols at link time).
const TSD_TLS_SYMBOL_NAMES: &[&str] = &["tsd_tls", "je_tsd_tls", "_rjem_je_tsd_tls"];

/// DWARF struct name for jemalloc's per-thread state.
const TSD_STRUCT_NAME: &str = "tsd_s";
/// jemalloc mangles `tsd_s` field names with this fixed prefix via
/// the `TSD_MANGLE` macro (`include/jemalloc/internal/tsd.h`) so
/// that direct field access in C code triggers a compile-time
/// symbol-lookup failure, forcing callers to go through the
/// `tsd_*_get` / `tsd_*_set` accessor macros. The DWARF emitted by
/// the compiler carries the mangled names verbatim — we match on
/// the full prefixed name to avoid accidental false positives on
/// substring overlaps like `thread_allocated_last_event_key` and
/// `thread_allocated_next_event_fast` in the TSD_DATA_SLOW pad.
///
/// Defined as a macro so [`ALLOCATED_FIELD`] / [`DEALLOCATED_FIELD`]
/// can assemble their full constant strings with `concat!` — a
/// `const &str` does not work as an argument to `concat!`. The
/// companion [`TSD_MANGLE_PREFIX`] const re-exposes the same string
/// for runtime use in error messages.
macro_rules! tsd_mangle_prefix {
    () => {
        "cant_access_tsd_items_directly_use_a_getter_or_setter_"
    };
}
/// Runtime-accessible form of [`tsd_mangle_prefix!`]. Used by the
/// `resolve_field_offsets` error message so a future jemalloc that
/// renames the prefix surfaces the drift directly in the diagnostic.
const TSD_MANGLE_PREFIX: &str = tsd_mangle_prefix!();
/// DWARF field name for the cumulative-bytes-allocated counter
/// inside [`TSD_STRUCT_NAME`]. Must be compared as an exact byte
/// match — [`TSD_MANGLE_PREFIX`] is present on every sibling field,
/// so a `contains`/`starts_with` check would collide with other
/// `thread_allocated_*` names.
const ALLOCATED_FIELD: &str = concat!(tsd_mangle_prefix!(), "thread_allocated");
/// DWARF field name for the cumulative-bytes-deallocated counter.
/// Same exact-match rule as [`ALLOCATED_FIELD`].
const DEALLOCATED_FIELD: &str = concat!(tsd_mangle_prefix!(), "thread_deallocated");

/// Probe a running jemalloc-linked process and report per-thread
/// allocated / deallocated byte counters.
#[derive(Parser, Debug)]
#[command(
    name = "ktstr-jemalloc-probe",
    version = env!("CARGO_PKG_VERSION"),
    about = "Read per-thread jemalloc allocated/deallocated byte counters from a running process",
    long_about = "Reads jemalloc's per-thread `thread_allocated` / `thread_deallocated` TLS \
                  counters out of a running process via ptrace + process_vm_readv. Counters are \
                  cumulative from thread creation — attaching late does not miss prior \
                  allocations. Requires CAP_SYS_PTRACE, root, or same-uid-as-target. V1 supports \
                  x86_64 targets with a statically-linked jemalloc and DWARF debuginfo on the \
                  binary carrying the jemalloc TLS symbol.\n\n\
                  The `--enable-stats` jemalloc build flag is NOT required: `thread.allocated` / \
                  `thread.deallocated` use jemalloc's `CTL_RO_NL_GEN` (ungated) and the fast/slow \
                  path writes are unconditional."
)]
struct Cli {
    /// Target process id. Required unless `--self-test` is set.
    #[arg(long, required_unless_present = "self_test")]
    pid: Option<i32>,
    /// Emit JSON on stdout instead of a human-readable table.
    #[arg(long)]
    json: bool,
    /// Self-test mode: spawn a worker thread that allocates the
    /// given number of bytes, read the worker's jemalloc TSD
    /// counters via `/proc/self/mem`, and exit 0 iff the probe
    /// observes at least that many `thread_allocated` bytes on
    /// the worker's thread.
    ///
    /// The probe binary ships with its own DWARF debuginfo so the
    /// TSD offset resolution succeeds even when used against a
    /// stripped external target would fail. Used by
    /// `tests/jemalloc_probe_tests.rs` for the closed-loop VM
    /// validation.
    #[arg(long, conflicts_with = "pid", value_name = "BYTES")]
    self_test: Option<u64>,
}

// ---------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ProbeOutput {
    schema_version: u32,
    pid: i32,
    tool_version: &'static str,
    /// Unix-epoch seconds at the start of the probe run. Intended for
    /// downstream diff tooling that correlates multiple probe
    /// snapshots against a workload timeline — an absolute timestamp
    /// lets callers align probe captures with other sidecar-emitted
    /// events. Unix seconds rather than ISO-8601 because this bin
    /// lives in its own compilation unit with no dependency on the
    /// lib crate's `test_support::timefmt` helper, and a `u64` is
    /// unambiguous and format-free for JSON consumers.
    ///
    /// **Clock source**: guest `CLOCK_REALTIME` (via
    /// `std::time::SystemTime`). Host-guest correlation requires
    /// aligned clocks — kvm-clock on the guest (default for KVM
    /// under ktstr's VMM) or NTP on both sides. Without alignment,
    /// the guest's `CLOCK_REALTIME` drifts against the host's
    /// wall clock over time (NTP slew, stepped corrections, a
    /// VMM-imposed offset at boot) — the skew between the two
    /// timelines is ongoing, not a single fixed offset, so
    /// downstream tools diffing probe captures across host +
    /// guest events must re-anchor against each run's timestamps
    /// rather than applying a constant offset, or the correlation
    /// silently drifts.
    timestamp_unix_sec: u64,
    threads: Vec<ThreadResult>,
}

/// Per-thread probe outcome.
///
/// **Wire format: `#[serde(untagged)]` by deliberate choice.** The
/// two variants have disjoint field sets (`allocated_bytes` /
/// `deallocated_bytes` on `Ok`; `error` / `error_kind` on `Err`),
/// so downstream consumers can discriminate via field presence
/// without a tag. The evaluated alternative was
/// `#[serde(tag = "status")]`, which would add a `"status": "ok"` /
/// `"status": "err"` discriminator to every thread entry.
///
/// Retained untagged on this pass because:
/// * **No present consumer hardship.** The probe's own tests pin
///   the exact shape (see `thread_result_json_shape`), and no
///   external consumer has reported presence-sniffing pain.
/// * **Breaking change cost without a use case.** Flipping to
///   tagged renames every entry on the wire and forces every
///   external parser to update. ktstr is pre-1.0, so the break
///   itself is cheap — but the benefit is speculative until a
///   concrete consumer asks for it.
/// * **Disjoint fields are the natural discriminant.** `error`
///   cannot appear on `Ok`, `allocated_bytes` cannot appear on
///   `Err`. A single field presence check is sufficient
///   (`has("error")` → Err arm, else Ok arm).
///
/// **Re-evaluate** if either (a) a future variant introduces a
/// field that overlaps with the Ok/Err field sets (discriminant
/// collision), or (b) a consumer needs to round-trip the JSON
/// back into a Rust enum — `#[serde(untagged)]` deserialization
/// is order-sensitive and errors less helpfully than tagged.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ThreadResult {
    Ok {
        tid: i32,
        /// Per-thread name from `/proc/{pid}/task/{tid}/comm`, trimmed
        /// of the trailing newline. `None` when the file read fails
        /// — typically the tid exited between enumeration and the
        /// comm read (race) — or when the comm string is empty
        /// after trimming (defense-in-depth; unexpected for live
        /// threads, since the kernel guarantees at least the first
        /// 16 bytes of the task name are populated). Best-effort: a
        /// `None` here never fails the probe.
        #[serde(skip_serializing_if = "Option::is_none")]
        comm: Option<String>,
        allocated_bytes: u64,
        deallocated_bytes: u64,
    },
    Err {
        tid: i32,
        /// Per-thread name from `/proc/{pid}/task/{tid}/comm`, read
        /// with the same semantics as the `Ok` arm. Particularly
        /// useful on failure: knowing WHICH thread exited or refused
        /// attach is harder from the tid alone.
        #[serde(skip_serializing_if = "Option::is_none")]
        comm: Option<String>,
        /// Human-readable error rendering for log / stderr paths.
        error: String,
        /// Structural classification so machine consumers can bucket
        /// failures (race vs. permission vs. arithmetic) without
        /// substring-matching the `error` field. See
        /// [`ThreadErrorKind`] for variant semantics.
        error_kind: ThreadErrorKind,
    },
}

/// Structural classifier for per-thread probe failures. The `error`
/// string is retained for human diagnostics; this enum exists so
/// machine consumers can aggregate (e.g. "n tids exited during
/// probe" vs. "n tids denied ptrace attach") without substring-
/// matching free-form text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, strum::EnumIter)]
#[serde(rename_all = "snake_case")]
enum ThreadErrorKind {
    /// `ptrace(PTRACE_SEIZE)` failed. Typical causes: ESRCH (tid
    /// exited between enumeration and attach — the race is the
    /// common case, not exceptional), EPERM (yama / uid policy /
    /// missing `CAP_SYS_PTRACE`), EBUSY (another tracer already
    /// attached). An operator hitting a persistent EPERM is the
    /// canonical signal to revisit scope / caps / uid — this
    /// variant is distinct from [`Self::PtraceInterrupt`] so
    /// machine consumers can bucket "config problem" vs
    /// "in-flight race" without substring-matching the `error`
    /// field.
    PtraceSeize,
    /// `ptrace(PTRACE_INTERRUPT)` failed after a successful seize.
    /// Separate variant from [`Self::PtraceSeize`] because the
    /// failure mode is narrower: EPERM cannot surface here (the
    /// permission gate already cleared at seize time), so
    /// interrupt failures are effectively race-only — ESRCH if the
    /// tid exited between seize and interrupt. An operator seeing
    /// an elevated `ptrace_interrupt` rate should look at workload
    /// thread churn rather than ptrace configuration.
    PtraceInterrupt,
    /// `waitpid` after interrupt returned an error or an unexpected
    /// status. The tid may have exited between seize and wait; the
    /// kernel reports either `Err(ECHILD)` or a non-Stopped wait
    /// status.
    Waitpid,
    /// `ptrace(PTRACE_GETREGSET, NT_PRSTATUS)` failed — the target
    /// tid exited between attach and register fetch, or the target
    /// is not an x86_64 thread (the probe refuses non-x86_64
    /// upstream of this path, but this variant is held as
    /// belt-and-braces).
    GetRegset,
    /// `process_vm_readv` against the computed TLS address failed or
    /// returned a short read. The address may be unmapped or the tid
    /// may have exited mid-read. Different root cause from
    /// [`Self::PtraceSeize`] / [`Self::PtraceInterrupt`] — we
    /// already have the register set when this fires.
    ProcessVmReadv,
    /// TLS-offset arithmetic overflowed (e.g. `fs_base -
    /// aligned_size + st_value` underflowed in the symbol-pin math).
    /// Should not occur for well-formed jemalloc ELFs; a hit means
    /// the symbol resolution produced a violated invariant.
    TlsArithmetic,
}

impl std::fmt::Display for ThreadErrorKind {
    /// Renders the same snake_case tokens emitted by the
    /// `#[serde(rename_all = "snake_case")]` JSON serialization.
    /// The human stderr path (`print_output`) uses this Display so
    /// operators grepping `warning: tid ... [<kind>]: ...` lines
    /// match against the same tokens that appear in the JSON
    /// `error_kind` field — no second vocabulary. Kept in lock-step
    /// with the serde tokens by a parity test in the tests module.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let token = match self {
            Self::PtraceSeize => "ptrace_seize",
            Self::PtraceInterrupt => "ptrace_interrupt",
            Self::Waitpid => "waitpid",
            Self::GetRegset => "get_regset",
            Self::ProcessVmReadv => "process_vm_readv",
            Self::TlsArithmetic => "tls_arithmetic",
        };
        f.write_str(token)
    }
}

// ---------------------------------------------------------------------
// ELF + DWARF resolution (pure, testable seams)
// ---------------------------------------------------------------------

/// Thread-local symbol lookup result — enough to compute the
/// per-thread address of the TLS image containing jemalloc's
/// `tsd_tls`.
#[derive(Debug, Clone)]
pub(crate) struct TsdTlsSymbol {
    /// Absolute path of the ELF containing the symbol.
    pub elf_path: PathBuf,
    /// `st_value` of the symbol. For a symbol in the main executable's
    /// PT_TLS, this is the offset WITHIN the TLS image (small, positive).
    pub st_value: u64,
    /// Aligned size of the PT_TLS program header:
    /// `round_up(p_memsz, p_align)`. Added as a negative offset to TP
    /// to reach the start of the TLS image (Variant II).
    pub tls_image_aligned_size: u64,
    /// ELF architecture e_machine value — used to refuse non-x86_64
    /// targets with a clear error.
    pub e_machine: u16,
}

/// Locate jemalloc's `tsd_tls` (or `je_tsd_tls`) symbol inside the
/// given ELF. Returns the symbol's `st_value` plus the PT_TLS-aligned
/// image size needed for TP-relative addressing.
///
/// Lookup order (§10 of the accepted design):
/// 1. `.symtab` — `tsd_tls`, then `je_tsd_tls`.
/// 2. `.dynsym` — same two names.
/// 3. TLS-section walk fallback (section flagged `SHF_TLS`,
///    symbol's `st_size` matches the expected `tsd_t` byte size).
///    Implemented as a follow-up path; v1 relies on 1-2 since
///    ktstr's own binaries keep `.symtab`.
pub(crate) fn find_tsd_tls(elf: &Elf<'_>, elf_path: &Path) -> Result<TsdTlsSymbol> {
    let e_machine = elf.header.e_machine;
    let tls_image_aligned_size = extract_pt_tls_size(elf)?;

    // Order-preserving name search across symbol tables.
    let finders: [(&str, &dyn Fn(&str) -> Option<u64>); 2] = [
        (
            ".symtab",
            &|name| find_symbol_by_name(&elf.syms, &elf.strtab, name),
        ),
        (
            ".dynsym",
            &|name| find_symbol_by_name(&elf.dynsyms, &elf.dynstrtab, name),
        ),
    ];
    for (_table_name, finder) in finders {
        for name in TSD_TLS_SYMBOL_NAMES {
            if let Some(st_value) = finder(name) {
                return Ok(TsdTlsSymbol {
                    elf_path: elf_path.to_path_buf(),
                    st_value,
                    tls_image_aligned_size,
                    e_machine,
                });
            }
        }
    }

    Err(anyhow!(
        "jemalloc TLS symbol ({}) not found in .symtab or .dynsym of {}",
        TSD_TLS_SYMBOL_NAMES.join(" / "),
        elf_path.display(),
    ))
}

fn find_symbol_by_name(
    syms: &goblin::elf::Symtab<'_>,
    strs: &goblin::strtab::Strtab<'_>,
    needle: &str,
) -> Option<u64> {
    for sym in syms.iter() {
        if let Some(name) = strs.get_at(sym.st_name)
            && name == needle
        {
            return Some(sym.st_value);
        }
    }
    None
}

fn extract_pt_tls_size(elf: &Elf<'_>) -> Result<u64> {
    let tls_hdr = elf
        .program_headers
        .iter()
        .find(|ph| ph.p_type == goblin::elf::program_header::PT_TLS)
        .ok_or_else(|| anyhow!("ELF has no PT_TLS segment — target does not use static TLS"))?;
    let align = tls_hdr.p_align.max(1);
    let rounded = tls_hdr
        .p_memsz
        .checked_add(align - 1)
        .and_then(|v| Some(v & !(align - 1)))
        .ok_or_else(|| anyhow!("PT_TLS size arithmetic overflow"))?;
    Ok(rounded)
}

/// Offsets of the two counters inside `struct tsd_s`, resolved from
/// DWARF. Computed once per ELF load; shared across every thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CounterOffsets {
    thread_allocated: u64,
    thread_deallocated: u64,
}

impl CounterOffsets {
    /// Construct, enforcing `thread_allocated < thread_deallocated`.
    /// jemalloc's TSD_DATA_FAST block lays them out in that order with
    /// `thread_allocated_next_event_fast` between them
    /// (tsd_internals.h L110-115). A reversed pair means the DWARF walk
    /// picked up a different struct or the layout has drifted; either
    /// way the combined-read math below would underflow and read
    /// garbage, so we fail fast with an actionable error.
    pub fn new(thread_allocated: u64, thread_deallocated: u64) -> Result<Self> {
        if thread_allocated >= thread_deallocated {
            bail!(
                "unexpected tsd_s layout: thread_allocated ({thread_allocated}) \
                 must precede thread_deallocated ({thread_deallocated}) per \
                 jemalloc TSD_DATA_FAST ordering",
            );
        }
        Ok(Self {
            thread_allocated,
            thread_deallocated,
        })
    }

    /// Byte range covering both counters plus the
    /// `thread_allocated_next_event_fast` u64 between them. Used as the
    /// remote iov for a single `process_vm_readv` while the target
    /// thread is stopped.
    pub fn combined_read_span(&self) -> (u64, u64) {
        let start = self.thread_allocated;
        let end = self.thread_deallocated + 8;
        (start, end - start)
    }
}

/// Resolve the byte offsets of `thread_allocated` and
/// `thread_deallocated` inside `struct tsd_s` by walking DWARF on the
/// ELF. Returns `Err` with actionable text when the ELF has no
/// `.debug_info` or the struct/fields are not found.
pub(crate) fn resolve_field_offsets(elf_path: &Path) -> Result<CounterOffsets> {
    let data = fs::read(elf_path)
        .with_context(|| format!("re-read {} for DWARF inspection", elf_path.display()))?;
    let elf = Elf::parse(&data).with_context(|| format!("parse ELF {}", elf_path.display()))?;

    let load_section = |id: gimli::SectionId| -> Result<Cow<'_, [u8]>> {
        let name = id.name();
        for sh in &elf.section_headers {
            if let Some(section_name) = elf.shdr_strtab.get_at(sh.sh_name)
                && section_name == name
            {
                let range = sh.file_range().unwrap_or(0..0);
                return Ok(Cow::Borrowed(&data[range]));
            }
        }
        Ok(Cow::Borrowed(&[]))
    };

    let dwarf_sections = gimli::DwarfSections::load(load_section)?;
    let dwarf = dwarf_sections.borrow(|bytes| EndianSlice::new(bytes, LittleEndian));

    let mut allocated: Option<u64> = None;
    let mut deallocated: Option<u64> = None;

    let mut units = dwarf.units();
    while let Some(header) = units.next()? {
        let unit = dwarf.unit(header)?;
        if let Some((a, d)) = struct_member_offsets_in_unit(&dwarf, &unit)? {
            if let Some(v) = a {
                allocated.get_or_insert(v);
            }
            if let Some(v) = d {
                deallocated.get_or_insert(v);
            }
            if allocated.is_some() && deallocated.is_some() {
                break;
            }
        }
    }

    let allocated = allocated.ok_or_else(|| {
        anyhow!(
            "DWARF walk of {} did not find field '{}' in struct '{}' — \
             target was built without -g, the jemalloc version renamed the field, \
             or the TSD_MANGLE prefix ('{}') drifted",
            elf_path.display(),
            ALLOCATED_FIELD,
            TSD_STRUCT_NAME,
            TSD_MANGLE_PREFIX,
        )
    })?;
    let deallocated = deallocated.ok_or_else(|| {
        anyhow!(
            "DWARF walk of {} did not find field '{}' in struct '{}'",
            elf_path.display(),
            DEALLOCATED_FIELD,
            TSD_STRUCT_NAME,
        )
    })?;
    CounterOffsets::new(allocated, deallocated)
}

#[allow(clippy::type_complexity)]
fn struct_member_offsets_in_unit<R: Reader>(
    dwarf: &gimli::Dwarf<R>,
    unit: &Unit<R>,
) -> Result<Option<(Option<u64>, Option<u64>)>> {
    let mut entries = unit.entries();
    while let Some((_, entry)) = entries.next_dfs()? {
        if entry.tag() != gimli::DW_TAG_structure_type {
            continue;
        }
        let name = match entry.attr_value(gimli::DW_AT_name)? {
            Some(v) => v,
            None => continue,
        };
        let name_str = dwarf.attr_string(unit, name)?;
        if name_str.to_slice()?.as_ref() != TSD_STRUCT_NAME.as_bytes() {
            continue;
        }

        let mut allocated: Option<u64> = None;
        let mut deallocated: Option<u64> = None;
        // depth == 1 is the tsd_s DIE itself; depth == 2 is a DIRECT
        // member; depth > 2 is a nested type's member (e.g. a bitfield
        // of `te_data_t`) — we must not accept those or we'll latch
        // onto a same-named field in a nested DIE.
        let mut depth = 1;
        while let Some((delta, child)) = entries.next_dfs()? {
            depth += delta;
            if depth <= 0 {
                break;
            }
            if depth != 2 {
                continue;
            }
            if child.tag() != gimli::DW_TAG_member {
                continue;
            }
            let child_name = match child.attr_value(gimli::DW_AT_name)? {
                Some(v) => v,
                None => continue,
            };
            let child_name_str = dwarf.attr_string(unit, child_name)?;
            let bytes = child_name_str.to_slice()?;
            let as_str = bytes.as_ref();
            let is_allocated = as_str == ALLOCATED_FIELD.as_bytes();
            let is_deallocated = as_str == DEALLOCATED_FIELD.as_bytes();
            if !is_allocated && !is_deallocated {
                continue;
            }
            let offset = member_offset(child.attr_value(gimli::DW_AT_data_member_location)?)?;
            if is_allocated && allocated.is_none() {
                allocated = offset;
            }
            if is_deallocated && deallocated.is_none() {
                deallocated = offset;
            }
            if allocated.is_some() && deallocated.is_some() {
                return Ok(Some((allocated, deallocated)));
            }
        }
        return Ok(Some((allocated, deallocated)));
    }
    Ok(None)
}

fn member_offset<R: Reader>(attr: Option<AttributeValue<R>>) -> Result<Option<u64>> {
    let Some(attr) = attr else { return Ok(None) };
    match attr {
        AttributeValue::Udata(v) => Ok(Some(v)),
        AttributeValue::Data1(v) => Ok(Some(v as u64)),
        AttributeValue::Data2(v) => Ok(Some(v as u64)),
        AttributeValue::Data4(v) => Ok(Some(v as u64)),
        AttributeValue::Data8(v) => Ok(Some(v)),
        AttributeValue::Sdata(v) if v >= 0 => Ok(Some(v as u64)),
        other => Err(anyhow!(
            "unexpected DW_AT_data_member_location form: {:?} — \
             DWARF expression forms are not supported for field-offset resolution in v1",
            other
        )),
    }
}

/// Compute the absolute address of a TLS variable's field in the target.
///
/// Variant II (x86_64): the thread pointer (`fs_base`) points to the
/// end of the static TLS block; the executable's TLS image sits at
/// `fs_base - tls_image_aligned_size`. The symbol lives at
/// `st_value` bytes within that image; the field lives `field_offset`
/// bytes inside the symbol.
///
/// Returns `Err` on `fs_base < tls_image_aligned_size` — that would
/// indicate the target has not initialized TLS or the ELF layout is
/// malformed; silently wrapping into the top of the address space
/// would produce a read from kernel-space and confuse the error path.
pub(crate) fn compute_tls_address(
    fs_base: u64,
    tls_image_aligned_size: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    let image_base = fs_base.checked_sub(tls_image_aligned_size).ok_or_else(|| {
        anyhow!(
            "fs_base ({fs_base:#x}) is below the aligned TLS image size \
             ({tls_image_aligned_size:#x}); target likely has no static \
             TLS initialized yet"
        )
    })?;
    image_base
        .checked_add(st_value)
        .and_then(|v| v.checked_add(field_offset))
        .ok_or_else(|| anyhow!("TLS address arithmetic overflow"))
}

// ---------------------------------------------------------------------
// /proc/<pid>/{maps,task}
// ---------------------------------------------------------------------

/// Enumerate thread ids for a target pid from `/proc/<pid>/task/`.
///
/// Returns them sorted so output ordering is deterministic across
/// runs and the enumeration is stable to `diff`.
pub(crate) fn iter_task_ids(pid: i32) -> Result<Vec<i32>> {
    let path = format!("/proc/{pid}/task");
    let entries = fs::read_dir(&path).with_context(|| format!("read_dir {path}"))?;
    let mut tids: Vec<i32> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse().ok()))
        .collect();
    tids.sort_unstable();
    Ok(tids)
}

/// Scan `/proc/<pid>/maps` for a mapping whose on-disk ELF contains a
/// jemalloc TLS symbol. Returns the (symbol info, DWARF-derived field
/// offsets) pair for the main executable match.
///
/// v1 is constrained to static-linked jemalloc, so the symbol MUST
/// live in the binary that `/proc/<pid>/exe` points at. If a match
/// turns up in some other ELF (a shared library loaded separately),
/// we bail — the TP math in this tool is only correct for the static
/// TLS image in the main executable; dynamic-TLS DSOs need DTV walks
/// (follow-up #558).
pub(crate) fn find_jemalloc_via_maps(
    pid: i32,
) -> Result<(TsdTlsSymbol, CounterOffsets)> {
    let exe_link = format!("/proc/{pid}/exe");
    let exe_path = fs::read_link(&exe_link)
        .with_context(|| format!("readlink {exe_link} (need it to gate static-TLS match)"))?;

    let maps_path = format!("/proc/{pid}/maps");
    let contents =
        fs::read_to_string(&maps_path).with_context(|| format!("read {maps_path}"))?;

    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    let mut last_symbol_err: Option<anyhow::Error> = None;
    for line in contents.lines() {
        let Some(path) = parse_maps_elf_path(line) else {
            continue;
        };
        if !seen.insert(path.clone()) {
            continue;
        }
        let data = match fs::read(&path) {
            Ok(d) => d,
            // Mapping may reference a path we cannot read (permissions,
            // deleted file). Skip and keep searching.
            Err(_) => continue,
        };
        let elf = match Elf::parse(&data) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let symbol = match find_tsd_tls(&elf, &path) {
            Ok(s) => s,
            Err(e) => {
                last_symbol_err = Some(e);
                continue;
            }
        };
        // Static-TLS guard: the match must be in the main executable.
        // A hit in a DSO is not something v1 can address correctly
        // (no DTV walk).
        if path != exe_path {
            bail!(
                "jemalloc TLS symbol found in {} but static-TLS probe requires \
                 the match be in the main executable ({}); dynamic-TLS lookups \
                 in shared objects are not supported in v1",
                path.display(),
                exe_path.display(),
            );
        }
        // Arch check runs before the (slow) DWARF walk so an aarch64
        // target fails fast with the right message instead of running
        // gimli over unsupported debug info.
        if symbol.e_machine != goblin::elf::header::EM_X86_64 {
            bail!(
                "probe is x86_64-only; target ELF {} is {} (e_machine={:#x})",
                symbol.elf_path.display(),
                e_machine_name(symbol.e_machine),
                symbol.e_machine,
            );
        }
        let offsets = resolve_field_offsets(&path)?;
        return Ok((symbol, offsets));
    }

    let context = last_symbol_err
        .map(|e| format!(" — last per-ELF error: {e}"))
        .unwrap_or_default();
    bail!(
        "jemalloc TLS symbol ({}) not found in any r-x mapping under {}{}",
        TSD_TLS_SYMBOL_NAMES.join(" / "),
        maps_path,
        context
    )
}

/// Human-readable name for an ELF e_machine value. Used in error
/// messages so a user who probed the wrong target gets "aarch64" back
/// instead of the raw hex number. Non-exhaustive; extends as new
/// arches are added to v1's support list.
pub(crate) fn e_machine_name(e_machine: u16) -> &'static str {
    use goblin::elf::header::{EM_386, EM_AARCH64, EM_PPC64, EM_RISCV, EM_S390, EM_X86_64};
    match e_machine {
        EM_X86_64 => "x86_64",
        EM_AARCH64 => "aarch64",
        EM_386 => "i386",
        EM_RISCV => "riscv",
        EM_PPC64 => "ppc64",
        EM_S390 => "s390",
        _ => "unknown",
    }
}

/// Extract the on-disk ELF path from a `/proc/<pid>/maps` line, or
/// `None` if the line is a non-file mapping (anon, [stack], …) or
/// not executable. Returning only `r-x` mappings avoids re-opening
/// the same ELF for each of its segments.
fn parse_maps_elf_path(line: &str) -> Option<PathBuf> {
    let mut iter = line.split_whitespace();
    let _range = iter.next()?;
    let perms = iter.next()?;
    // Skip non-executable mappings (rw-p, r--p, …); we only need the
    // code-bearing mapping once per ELF.
    if !perms.contains('x') {
        return None;
    }
    let _offset = iter.next()?;
    let _dev = iter.next()?;
    let _inode = iter.next()?;
    let path = iter.next()?;
    if !path.starts_with('/') {
        return None;
    }
    Some(PathBuf::from(path))
}

// ---------------------------------------------------------------------
// Per-thread attach / read / detach
// ---------------------------------------------------------------------

/// Single-snapshot counters read from one target thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ThreadCounters {
    pub allocated_bytes: u64,
    pub deallocated_bytes: u64,
}

/// Tracks which tids we've seized so SIGINT cleanup can detach them.
/// A Mutex<BTreeSet> is fine: contention is only between the probe
/// thread and the signal handler, and the handler runs on SIGINT
/// only.
static ATTACHED: OnceLock<Mutex<BTreeSet<i32>>> = OnceLock::new();
static CLEANUP_REQUESTED: AtomicBool = AtomicBool::new(false);

fn attached() -> &'static Mutex<BTreeSet<i32>> {
    ATTACHED.get_or_init(|| Mutex::new(BTreeSet::new()))
}

extern "C" fn on_sigint(_sig: i32) {
    // Async-signal-safe subset: just flip the flag and let the main
    // loop drain. We cannot touch the Mutex from signal context, but
    // the iteration check in probe_single_thread catches it between
    // tids.
    CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
}

/// Install a SIGINT/SIGTERM handler that asks the main loop to detach
/// every still-attached tid and exit. Returns `()` rather than
/// failing — if signal install fails, the probe still works; only the
/// Ctrl-C cleanup guarantee is weakened.
fn install_cleanup_handler() {
    for sig in [Signal::SIGINT, Signal::SIGTERM] {
        // SAFETY: `on_sigint` only touches an `AtomicBool`, which is
        // async-signal-safe.
        unsafe {
            let _ = signal(sig, SigHandler::Handler(on_sigint));
        }
    }
}

/// Per-thread probe error carrying both a human rendering and a
/// structural classifier. Produced by [`probe_single_thread`];
/// consumed at the caller to populate [`ThreadResult::Err`].
struct ThreadProbeError {
    kind: ThreadErrorKind,
    source: anyhow::Error,
}

impl ThreadProbeError {
    fn new(kind: ThreadErrorKind, source: anyhow::Error) -> Self {
        Self { kind, source }
    }

    /// `ptrace(PTRACE_SEIZE)` failure. The `EPERM` branch expands into
    /// a multi-line operator hint enumerating the four common fixes
    /// (root, file capability, uid match, yama scope). Kept out of the
    /// caller so the hint text has a single source of truth.
    fn ptrace_seize(tid: i32, e: nix::errno::Errno) -> Self {
        let source = if e == nix::errno::Errno::EPERM {
            anyhow!(
                "ptrace(PTRACE_SEIZE) on tid {tid}: permission denied (EPERM). \
                 Grant access with one of: (1) run as root, (2) setcap \
                 cap_sys_ptrace+ep ktstr-jemalloc-probe, (3) run under the \
                 same uid as target, (4) set /proc/sys/kernel/yama/ptrace_scope=0 \
                 (requires root; affects system-wide ptrace policy)."
            )
        } else {
            anyhow!("ptrace(PTRACE_SEIZE) on tid {tid}: {e}")
        };
        Self::new(ThreadErrorKind::PtraceSeize, source)
    }

    fn ptrace_interrupt(tid: i32, e: nix::errno::Errno) -> Self {
        Self::new(
            ThreadErrorKind::PtraceInterrupt,
            anyhow!("ptrace(PTRACE_INTERRUPT) on tid {tid}: {e}"),
        )
    }

    fn waitpid_unexpected(tid: i32, status: WaitStatus) -> Self {
        Self::new(
            ThreadErrorKind::Waitpid,
            anyhow!("waitpid on tid {tid} returned unexpected status: {status:?}"),
        )
    }

    fn waitpid_err(tid: i32, e: nix::errno::Errno) -> Self {
        Self::new(
            ThreadErrorKind::Waitpid,
            anyhow!("waitpid on tid {tid}: {e}"),
        )
    }

    fn getregset(tid: i32, e: nix::errno::Errno) -> Self {
        Self::new(
            ThreadErrorKind::GetRegset,
            anyhow!("ptrace(PTRACE_GETREGSET, NT_PRSTATUS) on tid {tid}: {e}"),
        )
    }

    fn tls_arithmetic(source: anyhow::Error) -> Self {
        Self::new(ThreadErrorKind::TlsArithmetic, source)
    }

    fn process_vm_readv_err(tid: i32, addr: u64, e: nix::errno::Errno) -> Self {
        Self::new(
            ThreadErrorKind::ProcessVmReadv,
            anyhow!("process_vm_readv on tid {tid} at {addr:#x}: {e}"),
        )
    }

    fn process_vm_readv_short(tid: i32, n: usize, expected: u64) -> Self {
        Self::new(
            ThreadErrorKind::ProcessVmReadv,
            anyhow!("short process_vm_readv on tid {tid}: got {n} bytes, expected {expected}"),
        )
    }
}

/// Perform the full seize → interrupt → wait → read-regs →
/// read-counters → detach sequence for a single target tid.
fn probe_single_thread(
    tid: i32,
    symbol: &TsdTlsSymbol,
    offsets: &CounterOffsets,
) -> std::result::Result<ThreadCounters, ThreadProbeError> {
    let pid = Pid::from_raw(tid);

    ptrace::seize(pid, Options::empty())
        .map_err(|e| ThreadProbeError::ptrace_seize(tid, e))?;
    // Record the attach BEFORE we try to interrupt, so a failure in
    // interrupt still lets cleanup detach us.
    attached().lock().unwrap().insert(tid);

    let _attached_guard = ScopeDetach(tid);

    ptrace::interrupt(pid).map_err(|e| ThreadProbeError::ptrace_interrupt(tid, e))?;
    match waitpid(pid, None) {
        Ok(WaitStatus::Stopped(_, _) | WaitStatus::PtraceEvent(_, _, _)) => {}
        Ok(other) => return Err(ThreadProbeError::waitpid_unexpected(tid, other)),
        Err(e) => return Err(ThreadProbeError::waitpid_err(tid, e)),
    }

    let regs = ptrace::getregset::<NT_PRSTATUS>(pid)
        .map_err(|e| ThreadProbeError::getregset(tid, e))?;
    let fs_base = regs.fs_base;

    let addr = compute_tls_address(
        fs_base,
        symbol.tls_image_aligned_size,
        symbol.st_value,
        offsets.thread_allocated,
    )
    .map_err(ThreadProbeError::tls_arithmetic)?;

    let (_base, span) = offsets.combined_read_span();
    let mut buf = vec![0u8; span as usize];
    let remote = RemoteIoVec {
        base: addr as usize,
        len: span as usize,
    };
    let mut local = [IoSliceMut::new(&mut buf)];
    let n = process_vm_readv(pid, &mut local, &[remote])
        .map_err(|e| ThreadProbeError::process_vm_readv_err(tid, addr, e))?;
    if n != span as usize {
        return Err(ThreadProbeError::process_vm_readv_short(tid, n, span));
    }

    let allocated = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    // bytes 8..16 are thread_allocated_next_event_fast (discarded).
    let dealloc_offset = (offsets.thread_deallocated - offsets.thread_allocated) as usize;
    let deallocated =
        u64::from_le_bytes(buf[dealloc_offset..dealloc_offset + 8].try_into().unwrap());

    Ok(ThreadCounters {
        allocated_bytes: allocated,
        deallocated_bytes: deallocated,
    })
}

/// Best-effort read of `/proc/{pid}/task/{tid}/comm`. Trims
/// surrounding whitespace, handling the kernel's trailing newline.
/// Returns `None` on any read failure — tid may have exited between
/// enumeration and this read, or the file may be unreadable for
/// permission reasons. The comm string is a diagnostic enrichment;
/// its absence is not a probe failure.
///
/// Captured BEFORE ptrace attach — a thread that renames itself via
/// `prctl(PR_SET_NAME)` mid-probe will appear with its pre-rename
/// name. The race is narrow (single read-modify-write inside the
/// probe loop) and attributing a rename to a specific probe cycle
/// is not a supported diagnostic.
fn read_thread_comm(pid: i32, tid: i32) -> Option<String> {
    let path = format!("/proc/{pid}/task/{tid}/comm");
    let raw = fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Drop guard that detaches the tid on scope exit so a mid-read
/// failure doesn't leave the target thread stopped.
struct ScopeDetach(i32);

impl Drop for ScopeDetach {
    fn drop(&mut self) {
        let pid = Pid::from_raw(self.0);
        let _ = ptrace::detach(pid, None);
        attached().lock().unwrap().remove(&self.0);
    }
}

/// Detach everything still in `ATTACHED`. Called from the main loop
/// when SIGINT arrived between tids.
fn detach_all_attached() {
    let tids: Vec<i32> = attached().lock().unwrap().iter().copied().collect();
    for tid in tids {
        let _ = ptrace::detach(Pid::from_raw(tid), None);
        attached().lock().unwrap().remove(&tid);
    }
}

// ---------------------------------------------------------------------
// Orchestration + output
// ---------------------------------------------------------------------

/// Outcome classification so `main` can decide the exit code without
/// re-inspecting the `threads` vec. `AllFailed` still emits JSON so
/// callers have a machine-parseable explanation; `Fatal` is for
/// pre-probe errors (pid missing, no jemalloc, arch mismatch) where
/// there's no per-thread error to surface.
enum RunOutcome {
    Ok(ProbeOutput),
    AllFailed(ProbeOutput),
    Fatal(anyhow::Error),
}

fn run(cli: &Cli) -> RunOutcome {
    // `required_unless_present = "self_test"` in the Cli derive
    // guarantees `pid` is Some whenever we reach this path.
    let pid = match cli.pid {
        Some(p) => p,
        None => {
            return RunOutcome::Fatal(anyhow!(
                "BUG: run() called without --pid set and without --self-test"
            ));
        }
    };
    // Self-probe reject: PTRACE_SEIZE refuses a tracer's own tgid —
    // ptrace semantics say a process cannot attach to itself. Catching
    // this at the CLI boundary produces an actionable error instead of
    // a per-thread EPERM cascade mid-run that looks like a permissions
    // problem.
    let self_pid = libc::pid_t::try_from(std::process::id())
        .expect("Linux pid_max <= 2^22 so pid fits in pid_t");
    if pid == self_pid {
        return RunOutcome::Fatal(anyhow!(
            "refusing to probe self (pid {pid} == ktstr-jemalloc-probe's own pid). \
             ptrace(PTRACE_SEIZE) rejects self-attach — a process cannot trace \
             itself. Run the probe from a separate process against the target's pid."
        ));
    }
    if !Path::new(&format!("/proc/{pid}")).exists() {
        return RunOutcome::Fatal(anyhow!("pid {pid} does not exist"));
    }

    let (symbol, offsets) = match find_jemalloc_via_maps(pid) {
        Ok(v) => v,
        Err(e) => return RunOutcome::Fatal(e),
    };

    let tids = match iter_task_ids(pid) {
        Ok(v) => v,
        Err(e) => return RunOutcome::Fatal(e),
    };

    // Capture timestamp BEFORE iterating threads so the field
    // represents "start of probe run" as its doc claims — a
    // post-loop capture would tail the variable-length per-thread
    // ptrace work and drift into meaninglessness for long traces.
    let timestamp_unix_sec = now_unix_sec();

    let mut threads: Vec<ThreadResult> = Vec::with_capacity(tids.len());
    let mut interrupted = false;
    for tid in tids {
        if CLEANUP_REQUESTED.load(Ordering::SeqCst) {
            detach_all_attached();
            interrupted = true;
            break;
        }
        // Read comm BEFORE probe: on failure paths the tid may
        // exit mid-probe, and the pre-probe read has the best chance
        // of catching a populated comm. Best-effort either way — a
        // `None` comm never upgrades a per-thread result to Err.
        let comm = read_thread_comm(pid, tid);
        match probe_single_thread(tid, &symbol, &offsets) {
            Ok(c) => threads.push(ThreadResult::Ok {
                tid,
                comm,
                allocated_bytes: c.allocated_bytes,
                deallocated_bytes: c.deallocated_bytes,
            }),
            Err(e) => threads.push(ThreadResult::Err {
                tid,
                comm,
                error: format!("{:#}", e.source),
                error_kind: e.kind,
            }),
        }
    }

    let out = ProbeOutput {
        schema_version: SCHEMA_VERSION,
        pid,
        tool_version: env!("CARGO_PKG_VERSION"),
        timestamp_unix_sec,
        threads,
    };

    if interrupted {
        return RunOutcome::Fatal(anyhow!(
            "probe interrupted by signal — attached threads detached"
        ));
    }

    if out.threads.is_empty() || out.threads.iter().all(|t| matches!(t, ThreadResult::Err { .. }))
    {
        RunOutcome::AllFailed(out)
    } else {
        RunOutcome::Ok(out)
    }
}

fn print_output(cli: &Cli, out: &ProbeOutput) -> Result<()> {
    if cli.json {
        let s = serde_json::to_string_pretty(out)?;
        println!("{s}");
    } else {
        println!("pid={} tool_version={}", out.pid, out.tool_version);
        for t in &out.threads {
            match t {
                ThreadResult::Ok {
                    tid,
                    comm,
                    allocated_bytes,
                    deallocated_bytes,
                } => {
                    let comm_suffix = comm
                        .as_deref()
                        .map(|c| format!(" comm={c}"))
                        .unwrap_or_default();
                    println!(
                        "tid={tid}{comm_suffix} allocated_bytes={allocated_bytes} deallocated_bytes={deallocated_bytes}",
                    );
                }
                ThreadResult::Err {
                    tid,
                    comm,
                    error,
                    error_kind,
                } => {
                    let comm_suffix = comm
                        .as_deref()
                        .map(|c| format!(" comm={c}"))
                        .unwrap_or_default();
                    eprintln!("warning: tid {tid}{comm_suffix} [{error_kind}]: {error}");
                }
            }
        }
    }
    Ok(())
}

/// Result emitted by `--self-test` mode. Probe JSON for external
/// pids goes through [`ProbeOutput`]; self-test surfaces a simpler
/// pass/fail shape because the caller (the VM integration test)
/// only needs the one observation + verdict.
#[derive(Debug, Serialize)]
struct SelfTestOutput {
    schema_version: u32,
    tid: i32,
    known_bytes: u64,
    observed_bytes: u64,
    passed: bool,
}

/// `--self-test <BYTES>`: spawn an allocator worker in this
/// process, wait for it to finish allocating, read its
/// `thread_allocated` TSD counter via `process_vm_readv` on our
/// own pid, compare against `bytes`, and exit accordingly.
///
/// No ptrace: the target thread is in the same process, so
/// `process_vm_readv(self_pid, ...)` succeeds under the same-uid
/// kernel check without CAP_SYS_PTRACE. The worker self-reports
/// its FS base via `arch_prctl(ARCH_GET_FS)` instead of the
/// ptrace GETREGSET path used for external probes, again because
/// we can't ptrace our own threads.
///
/// The probe binary ships with DWARF (not stripped), so
/// `find_jemalloc_via_maps(self_pid)` resolves `tsd_s`'s field
/// layout against the probe's own executable — the external-pid
/// DWARF-requirement does not apply here.
fn run_self_test(bytes: u64) -> RunOutcome {
    use std::sync::atomic::AtomicI32;
    use std::sync::{Arc, mpsc};

    // Load CURRENT thread's arch_prctl(ARCH_GET_FS) to learn its
    // FS base. x86_64-only — aarch64 uses a different TLS model
    // and the probe already bails on non-x86_64 elsewhere.
    fn arch_get_fs() -> Result<u64> {
        const ARCH_GET_FS: libc::c_int = 0x1003;
        let mut fs_base: u64 = 0;
        let r = unsafe {
            libc::syscall(
                libc::SYS_arch_prctl,
                ARCH_GET_FS as libc::c_ulong,
                &mut fs_base as *mut u64,
            )
        };
        if r != 0 {
            return Err(anyhow!(
                "arch_prctl(ARCH_GET_FS) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(fs_base)
    }

    let self_pid: i32 = libc::pid_t::try_from(std::process::id())
        .expect("Linux pid_max <= 2^22 so pid fits in pid_t");
    // Capture timestamp BEFORE spawning the self-test worker so the
    // field represents "start of probe run" per its doc, not the
    // post-join point that would slide with `bytes` or worker
    // scheduling latency.
    let timestamp_unix_sec = now_unix_sec();
    let worker_tid = Arc::new(AtomicI32::new(0));
    let worker_fs_base = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tid_clone = worker_tid.clone();
    let fs_clone = worker_fs_base.clone();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let bytes_usize = bytes as usize;

    let worker = std::thread::spawn(move || {
        let tid = libc::pid_t::try_from(unsafe { libc::syscall(libc::SYS_gettid) })
            .expect("Linux pid_max <= 2^22 so tid fits in pid_t");
        let fs_base = match arch_get_fs() {
            Ok(v) => v,
            Err(e) => {
                let _ = ready_tx.send(Err(format!("arch_get_fs: {e:#}")));
                return;
            }
        };
        tid_clone.store(tid, Ordering::Release);
        fs_clone.store(fs_base, Ordering::Release);
        // Alloc AFTER registering tid/fs so a main-thread read
        // before the alloc completes only misses the counter
        // update, not identity metadata.
        let known: Vec<u8> = vec![0u8; bytes_usize];
        std::hint::black_box(&known);
        let _ = ready_tx.send(Ok(()));
        let _ = stop_rx.recv();
        drop(known);
    });

    let ready = match ready_rx.recv() {
        Ok(r) => r,
        Err(e) => return RunOutcome::Fatal(anyhow!("self-test ready recv: {e}")),
    };
    if let Err(msg) = ready {
        let _ = stop_tx.send(());
        let _ = worker.join();
        return RunOutcome::Fatal(anyhow!("self-test worker: {msg}"));
    }

    let tid = worker_tid.load(Ordering::Acquire);
    let fs_base = worker_fs_base.load(Ordering::Acquire);

    // Resolve jemalloc symbols + offsets against the probe's own
    // ELF (via `/proc/self/maps`). DWARF comes from the probe
    // binary, not the target of an external probe.
    let (symbol, offsets) = match find_jemalloc_via_maps(self_pid) {
        Ok(v) => v,
        Err(e) => {
            let _ = stop_tx.send(());
            let _ = worker.join();
            return RunOutcome::Fatal(e);
        }
    };

    let addr = match compute_tls_address(
        fs_base,
        symbol.tls_image_aligned_size,
        symbol.st_value,
        offsets.thread_allocated,
    ) {
        Ok(a) => a,
        Err(e) => {
            let _ = stop_tx.send(());
            let _ = worker.join();
            return RunOutcome::Fatal(e);
        }
    };

    let mut buf = [0u8; 8];
    let remote = RemoteIoVec {
        base: addr as usize,
        len: 8,
    };
    let mut local = [IoSliceMut::new(&mut buf)];
    let n = match process_vm_readv(Pid::from_raw(self_pid), &mut local, &[remote]) {
        Ok(n) => n,
        Err(e) => {
            let _ = stop_tx.send(());
            let _ = worker.join();
            return RunOutcome::Fatal(anyhow!(
                "process_vm_readv(self_pid={self_pid}, addr={addr:#x}): {e}"
            ));
        }
    };
    let _ = stop_tx.send(());
    let _ = worker.join();
    if n != 8 {
        return RunOutcome::Fatal(anyhow!(
            "short process_vm_readv: got {n} bytes, expected 8"
        ));
    }
    let observed = u64::from_le_bytes(buf);

    let out = SelfTestOutput {
        schema_version: SCHEMA_VERSION,
        tid,
        known_bytes: bytes,
        observed_bytes: observed,
        passed: observed >= bytes,
    };
    match serde_json::to_string_pretty(&out) {
        Ok(s) => println!("{s}"),
        Err(e) => {
            return RunOutcome::Fatal(anyhow!("serialize self-test output: {e}"));
        }
    }
    if out.passed {
        RunOutcome::Ok(ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: self_pid,
            tool_version: env!("CARGO_PKG_VERSION"),
            timestamp_unix_sec,
            threads: Vec::new(),
        })
    } else {
        RunOutcome::AllFailed(ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: self_pid,
            tool_version: env!("CARGO_PKG_VERSION"),
            timestamp_unix_sec,
            threads: Vec::new(),
        })
    }
}

fn main() {
    install_cleanup_handler();
    let cli = Cli::parse();
    // `--self-test` is an alternative entry point; its JSON shape
    // is `SelfTestOutput`, not `ProbeOutput`. Emission happens
    // inside `run_self_test`, so main only needs to translate
    // the `RunOutcome` to an exit code.
    if let Some(bytes) = cli.self_test {
        match run_self_test(bytes) {
            RunOutcome::Ok(_) => return,
            RunOutcome::AllFailed(_) => {
                eprintln!("error: self-test failed (observed < known)");
                std::process::exit(1);
            }
            RunOutcome::Fatal(e) => {
                eprintln!("error: {e:#}");
                std::process::exit(1);
            }
        }
    }
    match run(&cli) {
        RunOutcome::Ok(out) => {
            if let Err(e) = print_output(&cli, &out) {
                eprintln!("error writing output: {e:#}");
                std::process::exit(1);
            }
        }
        RunOutcome::AllFailed(out) => {
            // Emit the structured output anyway so callers have the
            // per-thread error reasons; exit non-zero to signal that
            // nothing succeeded.
            if let Err(e) = print_output(&cli, &out) {
                eprintln!("error writing output: {e:#}");
            }
            eprintln!("error: all threads failed probe");
            detach_all_attached();
            std::process::exit(1);
        }
        RunOutcome::Fatal(e) => {
            // Emit a single structured tag alongside the human
            // rendering so test bodies that want variant-specific
            // pinning (e.g. "probe bailed because the target pid
            // did not exist", as distinct from "target existed but
            // was not jemalloc-linked") can match on a stable
            // substring rather than the free-form `{e:#}` text.
            // The tag shape is intentionally grep-friendly:
            // `ktstr-probe-fatal: <kind>` with `kind` drawn from
            // a short, closed vocabulary. Consumers can filter on
            // the `ktstr-probe-fatal:` prefix to harvest only
            // structured lines even if the underlying human text
            // changes.
            let kind = classify_fatal(&e);
            eprintln!("ktstr-probe-fatal: {kind}");
            eprintln!("error: {e:#}");
            detach_all_attached();
            std::process::exit(1);
        }
    }
}

/// Classify a `RunOutcome::Fatal` error into a short tag for
/// structured stderr emission in [`main`]. Matches the underlying
/// `anyhow::Error` root-cause display against a small set of
/// closed substrings; unknown shapes fall through to `other` so
/// the tag stream stays well-formed even on unexpected errors.
/// The vocabulary is intentionally tiny — adding a new kind is
/// always safe; removing or renaming one is a breaking change to
/// downstream test consumers that match on the substring.
fn classify_fatal(e: &anyhow::Error) -> &'static str {
    let msg = format!("{e:#}");
    if msg.contains("does not exist") {
        "pid-missing"
    } else if msg.contains("jemalloc") && msg.contains("not") {
        "not-jemalloc"
    } else if msg.contains("self-test") {
        "self-test"
    } else {
        "other"
    }
}

// ---------------------------------------------------------------------
// Tests (pure-function seams)
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Variant II TLS TP math: fs_base - aligned_tls_size + st_value +
    /// field_offset. Worked example pins the arithmetic against a
    /// hand-checked case.
    #[test]
    fn compute_tls_address_variant_ii_example() {
        let fs_base = 0x7f12_3456_7000;
        let aligned = 512; // round_up(memsz=500, align=16)
        let st_value = 0x100; // symbol is at byte 256 of the TLS image
        let field = 264; // offsetof(tsd_s, thread_allocated) example
        let addr = compute_tls_address(fs_base, aligned, st_value, field).unwrap();
        // 0x7f1234567000 - 0x200 + 0x100 + 264
        // = 0x7f1234566f00 + 264
        // = 0x7f1234567008
        assert_eq!(addr, 0x7f12_3456_7008);
    }

    /// Thread pointer equal to aligned image size is the minimum
    /// valid configuration — subtraction lands at zero rather than
    /// underflowing. A lower fs_base is an error surface (see next
    /// test).
    #[test]
    fn compute_tls_address_boundary_tp_equals_image_size() {
        let addr = compute_tls_address(/*fs_base*/ 4096, /*aligned*/ 4096, 0, 0).unwrap();
        assert_eq!(addr, 0);
    }

    /// fs_base below the TLS image size is a malformed-state error —
    /// the math must NOT wrap into the top of the u64 address space.
    #[test]
    fn compute_tls_address_underflow_errors() {
        let err = compute_tls_address(4096, 8192, 0, 0).unwrap_err();
        assert!(
            format!("{err}").contains("below the aligned TLS image size"),
            "got: {err}",
        );
    }

    /// `combined_read_span` must cover both counters and the interleaving
    /// `thread_allocated_next_event_fast` — else the single
    /// process_vm_readv would need a second iov.
    #[test]
    fn counter_offsets_combined_span_covers_both() {
        let o = CounterOffsets::new(264, 280).unwrap();
        let (start, span) = o.combined_read_span();
        assert_eq!(start, 264);
        assert_eq!(span, 24, "8 (allocated) + 8 (fast_event) + 8 (deallocated)");
    }

    /// Exact-adjacency: if a future jemalloc drops the fast_event
    /// field and places deallocated immediately after allocated, the
    /// span collapses to 16. Guards against regression that would
    /// truncate the read.
    #[test]
    fn counter_offsets_combined_span_adjacent() {
        let o = CounterOffsets::new(100, 108).unwrap();
        let (_start, span) = o.combined_read_span();
        assert_eq!(span, 16);
    }

    /// Field-order invariant: `thread_allocated` must precede
    /// `thread_deallocated` in the TSD layout. A reversed pair means
    /// DWARF found the wrong struct or the upstream layout drifted;
    /// either way the read math would underflow.
    #[test]
    fn counter_offsets_reject_reversed_order() {
        let err = CounterOffsets::new(280, 264).unwrap_err();
        assert!(
            format!("{err}").contains("unexpected tsd_s layout"),
            "got: {err}",
        );
    }

    /// Equal offsets are also invalid — jemalloc's layout separates
    /// the two counters by `thread_allocated_next_event_fast`.
    #[test]
    fn counter_offsets_reject_equal_offsets() {
        assert!(CounterOffsets::new(100, 100).is_err());
    }

    /// e_machine error-message pretty-printer maps the handful of
    /// common Linux architectures. Guards against regressions like
    /// "probe is x86_64-only; target is 0xb7" — that hex means
    /// aarch64, which is actionable once named.
    #[test]
    fn e_machine_name_common_arches() {
        use goblin::elf::header::{EM_386, EM_AARCH64, EM_X86_64};
        assert_eq!(e_machine_name(EM_X86_64), "x86_64");
        assert_eq!(e_machine_name(EM_AARCH64), "aarch64");
        assert_eq!(e_machine_name(EM_386), "i386");
        assert_eq!(e_machine_name(0xbeef), "unknown");
    }

    /// /proc/<pid>/maps parser: only r-x mappings with on-disk paths
    /// produce a candidate ELF path. Anon / [stack] / non-executable
    /// mappings must be skipped.
    #[test]
    fn parse_maps_elf_path_accepts_rx_only() {
        let line = "5580e0001000-5580e0002000 r-xp 00000000 fd:01 12345 /usr/bin/ktstr";
        assert_eq!(
            parse_maps_elf_path(line),
            Some(PathBuf::from("/usr/bin/ktstr"))
        );
    }

    #[test]
    fn parse_maps_elf_path_rejects_non_executable() {
        let line = "5580e0002000-5580e0003000 rw-p 00001000 fd:01 12345 /usr/bin/ktstr";
        assert!(parse_maps_elf_path(line).is_none());
    }

    #[test]
    fn parse_maps_elf_path_rejects_anon_mapping() {
        let line = "7f1234567000-7f1234568000 rw-p 00000000 00:00 0 ";
        assert!(parse_maps_elf_path(line).is_none());
    }

    #[test]
    fn parse_maps_elf_path_rejects_pseudo_paths() {
        // `[stack]` and friends start with `[` not `/` — not a real
        // file so we skip them.
        let line = "7ffc12345000-7ffc12367000 rw-p 00000000 00:00 0 [stack]";
        assert!(parse_maps_elf_path(line).is_none());
    }

    /// `find_symbol_by_name` negative path: empty strtab must not
    /// panic and returns None.
    #[test]
    fn find_symbol_by_name_nothing_found() {
        let tab: goblin::elf::Symtab<'_> = Default::default();
        let strs = goblin::strtab::Strtab::default();
        assert!(find_symbol_by_name(&tab, &strs, "tsd_tls").is_none());
    }

    /// JSON schema v1: success + error arms round-trip via serde,
    /// with the batch-47 enrichment fields (`comm`, `error_kind`,
    /// `timestamp_unix_sec`) present where expected. Includes a
    /// third entry — an `Ok` with `comm: None` — to pin the
    /// `skip_serializing_if` behavior on the Ok arm as well.
    #[test]
    fn thread_result_json_shape() {
        let ok = ThreadResult::Ok {
            tid: 42,
            comm: Some("worker-0".to_string()),
            allocated_bytes: 1024,
            deallocated_bytes: 512,
        };
        let ok_no_comm = ThreadResult::Ok {
            tid: 44,
            comm: None,
            allocated_bytes: 2048,
            deallocated_bytes: 1024,
        };
        let err = ThreadResult::Err {
            tid: 43,
            comm: None,
            error: "thread exited before probe".to_string(),
            error_kind: ThreadErrorKind::Waitpid,
        };
        let out = ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: 100,
            tool_version: "0.0.0",
            timestamp_unix_sec: 1_700_000_000,
            threads: vec![ok, ok_no_comm, err],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(s.contains("\"schema_version\":1"));
        assert!(s.contains("\"pid\":100"));
        assert!(s.contains("\"tool_version\":\"0.0.0\""));
        assert!(s.contains("\"timestamp_unix_sec\":1700000000"));
        assert!(s.contains("\"allocated_bytes\":1024"));
        assert!(s.contains("\"deallocated_bytes\":512"));
        assert!(s.contains("\"allocated_bytes\":2048"));
        assert!(s.contains("\"deallocated_bytes\":1024"));
        assert!(s.contains("\"comm\":\"worker-0\""));
        assert!(s.contains("\"error\":\"thread exited before probe\""));
        assert!(s.contains("\"error_kind\":\"waitpid\""));
        assert!(s.contains("\"tid\":42"));
        assert!(s.contains("\"tid\":43"));
        assert!(s.contains("\"tid\":44"));
        // `comm: None` on EITHER arm must be omitted (skip_serializing_if).
        // The ok_no_comm and Err entries both have comm: None, so the
        // serialized blob must carry zero `"comm":null` occurrences.
        assert!(!s.contains("\"comm\":null"));
    }

    /// Canonical snake_case token for each `ThreadErrorKind` variant.
    /// Single source of truth consumed by both the serde-serialization
    /// test and the `Display`↔serde parity test. Adding a new variant
    /// triggers a compile error here (missing match arm); combined
    /// with the `strum::EnumIter` derive on the enum and the
    /// `ThreadErrorKind::iter()` loop in each test, no variant can
    /// slip through untested.
    fn expected_error_kind_token(k: ThreadErrorKind) -> &'static str {
        match k {
            ThreadErrorKind::PtraceSeize => "ptrace_seize",
            ThreadErrorKind::PtraceInterrupt => "ptrace_interrupt",
            ThreadErrorKind::Waitpid => "waitpid",
            ThreadErrorKind::GetRegset => "get_regset",
            ThreadErrorKind::ProcessVmReadv => "process_vm_readv",
            ThreadErrorKind::TlsArithmetic => "tls_arithmetic",
        }
    }

    /// `ThreadErrorKind` serializes every variant to its documented
    /// snake_case token. Pins the `#[serde(rename_all = "snake_case")]`
    /// attribute against accidental removal or rename — the error
    /// classification is a wire contract consumed by downstream
    /// tooling, and a silent rename ("get_regset" → "getregset")
    /// would break every consumer that matches on the token.
    /// Iterates via `strum::EnumIter` so a newly-added variant is
    /// covered exhaustively without a parallel array edit.
    #[test]
    fn thread_error_kind_snake_case_serialization() {
        use strum::IntoEnumIterator;
        for k in ThreadErrorKind::iter() {
            let s = serde_json::to_string(&k).unwrap();
            assert_eq!(
                s,
                format!("\"{}\"", expected_error_kind_token(k)),
                "variant {k:?}",
            );
        }
    }

    /// `iter_task_ids` of /proc/self/task must return at least the
    /// current thread. Sorted ascending.
    #[test]
    fn iter_task_ids_self() {
        let pid = libc::pid_t::try_from(std::process::id())
            .expect("Linux pid_max <= 2^22 so pid fits in pid_t");
        let tids = iter_task_ids(pid).expect("self/task must be readable");
        assert!(!tids.is_empty());
        assert!(tids.windows(2).all(|w| w[0] <= w[1]), "tids must be sorted");
    }

    /// `extract_pt_tls_size` rounds PT_TLS.p_memsz up to p_align.
    /// Since we can't easily construct a full goblin::elf::Elf
    /// fixture, test the arithmetic via a small helper that mirrors
    /// the inner logic.
    #[test]
    fn pt_tls_round_up_arithmetic() {
        fn round_up(memsz: u64, align: u64) -> u64 {
            let align = align.max(1);
            (memsz + (align - 1)) & !(align - 1)
        }
        assert_eq!(round_up(500, 16), 512);
        assert_eq!(round_up(512, 16), 512);
        assert_eq!(round_up(513, 16), 528);
        assert_eq!(round_up(0, 1), 0);
    }

    /// `Display` for `ThreadErrorKind` must render the same snake_case
    /// token as the serde JSON serialization AND the canonical
    /// expected-token mapping. The stderr render path (`print_output`)
    /// uses `{error_kind}` so operators matching on
    /// `warning: tid ... [ptrace_seize]: ...` share a pattern with
    /// the JSON `"error_kind": "ptrace_seize"` consumers. A drift
    /// (e.g. Display rendering `PtraceSeize` while serde still emits
    /// `ptrace_seize`) would silently fork the two vocabularies.
    /// Iterates via `strum::EnumIter` so a newly-added variant is
    /// covered without a parallel array edit.
    #[test]
    fn thread_error_kind_display_matches_serde_token() {
        use strum::IntoEnumIterator;
        for k in ThreadErrorKind::iter() {
            let expected = expected_error_kind_token(k);
            let json = serde_json::to_string(&k).unwrap();
            let serde_token = json.trim_matches('"');
            let display_token = format!("{k}");
            assert_eq!(serde_token, expected, "serde token for {k:?}");
            assert_eq!(display_token, expected, "Display token for {k:?}");
        }
    }

    /// `run()` must short-circuit to `RunOutcome::Fatal` when `--pid`
    /// matches the probe's own pid. PTRACE_SEIZE rejects self-attach
    /// at the kernel level, so without this gate every tid would
    /// fail with EPERM mid-loop and the user would see a per-thread
    /// permission cascade instead of an actionable "cannot probe
    /// self" error. Pins the early-return AND the error wording
    /// (`refusing to probe self`) that downstream tests and error-
    /// message consumers match against.
    #[test]
    fn run_rejects_self_probe() {
        let cli = Cli {
            pid: Some(
                libc::pid_t::try_from(std::process::id())
                    .expect("Linux pid_max <= 2^22 so pid fits in pid_t"),
            ),
            json: false,
            self_test: None,
        };
        match run(&cli) {
            RunOutcome::Fatal(err) => {
                let msg = format!("{err:#}");
                assert!(
                    msg.contains("refusing to probe self"),
                    "expected self-probe rejection wording, got: {msg}",
                );
            }
            other => panic!(
                "expected Fatal for pid==self_pid, got variant: {}",
                match other {
                    RunOutcome::Ok(_) => "Ok",
                    RunOutcome::AllFailed(_) => "AllFailed",
                    RunOutcome::Fatal(_) => unreachable!(),
                },
            ),
        }
    }

    /// Acceptance direction for the self-probe gate: a non-self pid
    /// must NOT trigger the `refusing to probe self` short-circuit.
    /// Pairs with `run_rejects_self_probe` to pin the gate's
    /// exactness — a regression that broadened the check (e.g. to
    /// "any pid in the probe's own process group", or a mis-typed
    /// comparison that tripped on unrelated pids) would fire the
    /// self-probe path and be caught here.
    ///
    /// Spawns `sleep 30` as a disposable non-self target; after the
    /// probe call, the child is killed + reaped so nothing leaks.
    /// The spawned process is not jemalloc-linked, so `run()` is
    /// expected to fail later in the pipeline (at
    /// `find_jemalloc_via_maps` or a ptrace step) with a DIFFERENT
    /// error. The assertion is narrow: whatever error surfaces, it
    /// must not be the self-probe message. `Ok` / `AllFailed` are
    /// equally acceptable — all three outcomes prove the self-probe
    /// gate was cleared.
    #[test]
    fn run_accepts_non_self_pid() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep for non-self pid acceptance test");
        let child_pid = libc::pid_t::try_from(child.id())
            .expect("Linux pid_max <= 2^22 so pid fits in pid_t");
        let self_pid = libc::pid_t::try_from(std::process::id())
            .expect("Linux pid_max <= 2^22 so pid fits in pid_t");
        assert_ne!(
            child_pid, self_pid,
            "spawned child pid must differ from parent for this test to be meaningful",
        );
        let cli = Cli {
            pid: Some(child_pid),
            json: false,
            self_test: None,
        };
        let outcome = run(&cli);
        let _ = child.kill();
        let _ = child.wait();
        if let RunOutcome::Fatal(err) = outcome {
            let msg = format!("{err:#}");
            assert!(
                !msg.contains("refusing to probe self"),
                "self-probe gate must NOT fire for non-self pid {child_pid} (self={self_pid}), got: {msg}",
            );
        }
    }

    // -- ThreadProbeError construction helpers --
    //
    // Each per-syscall helper was extracted from open-coded
    // `ThreadProbeError::new(Kind, anyhow!(...))` sites in
    // `probe_single_thread`; these tests pin (1) the `kind` tag each
    // helper emits, (2) the exact message format so operators grepping
    // stderr can keep stable anchors, and (3) the EPERM-branching
    // logic inside `ptrace_seize`.

    #[test]
    fn ptrace_seize_eperm_renders_operator_hint() {
        let err = ThreadProbeError::ptrace_seize(42, nix::errno::Errno::EPERM);
        assert_eq!(err.kind, ThreadErrorKind::PtraceSeize);
        let msg = format!("{}", err.source);
        assert!(msg.contains("tid 42"), "got: {msg}");
        assert!(msg.contains("permission denied"), "got: {msg}");
        // The 4 operator-fix hints must all be enumerated.
        assert!(msg.contains("(1) run as root"), "got: {msg}");
        assert!(msg.contains("(2) setcap"), "got: {msg}");
        assert!(msg.contains("(3) run under the"), "got: {msg}");
        assert!(msg.contains("(4) set /proc/sys/kernel/yama/ptrace_scope=0"), "got: {msg}");
    }

    #[test]
    fn ptrace_seize_non_eperm_uses_generic_rendering() {
        // ESRCH is the common "tid exited before seize" race — must
        // NOT render the EPERM operator hint (that would mislead the
        // operator into chasing a permission issue for a transient
        // exit).
        let err = ThreadProbeError::ptrace_seize(42, nix::errno::Errno::ESRCH);
        assert_eq!(err.kind, ThreadErrorKind::PtraceSeize);
        let msg = format!("{}", err.source);
        assert!(msg.contains("ptrace(PTRACE_SEIZE) on tid 42"), "got: {msg}");
        assert!(!msg.contains("permission denied"), "got: {msg}");
        assert!(!msg.contains("yama"), "got: {msg}");
    }

    #[test]
    fn ptrace_interrupt_formats_tid_and_errno() {
        let err = ThreadProbeError::ptrace_interrupt(17, nix::errno::Errno::ESRCH);
        assert_eq!(err.kind, ThreadErrorKind::PtraceInterrupt);
        let msg = format!("{}", err.source);
        assert!(msg.contains("ptrace(PTRACE_INTERRUPT) on tid 17"), "got: {msg}");
    }

    #[test]
    fn waitpid_unexpected_records_status_debug() {
        let status = WaitStatus::Exited(Pid::from_raw(99), 7);
        let err = ThreadProbeError::waitpid_unexpected(99, status);
        assert_eq!(err.kind, ThreadErrorKind::Waitpid);
        let msg = format!("{}", err.source);
        assert!(msg.contains("waitpid on tid 99"), "got: {msg}");
        assert!(msg.contains("unexpected status"), "got: {msg}");
        // `{status:?}` renders the variant name — pin that the
        // debug-formatted status is carried through.
        assert!(msg.contains("Exited"), "got: {msg}");
    }

    #[test]
    fn waitpid_err_formats_tid_and_errno() {
        let err = ThreadProbeError::waitpid_err(55, nix::errno::Errno::ECHILD);
        assert_eq!(err.kind, ThreadErrorKind::Waitpid);
        let msg = format!("{}", err.source);
        assert!(msg.contains("waitpid on tid 55"), "got: {msg}");
    }

    #[test]
    fn getregset_formats_tid_and_errno() {
        let err = ThreadProbeError::getregset(88, nix::errno::Errno::ESRCH);
        assert_eq!(err.kind, ThreadErrorKind::GetRegset);
        let msg = format!("{}", err.source);
        assert!(msg.contains("PTRACE_GETREGSET"), "got: {msg}");
        assert!(msg.contains("NT_PRSTATUS"), "got: {msg}");
        assert!(msg.contains("tid 88"), "got: {msg}");
    }

    #[test]
    fn tls_arithmetic_passes_through_source() {
        let source = anyhow!("computed TLS address underflowed for fs_base=0x1000");
        let err = ThreadProbeError::tls_arithmetic(source);
        assert_eq!(err.kind, ThreadErrorKind::TlsArithmetic);
        let msg = format!("{}", err.source);
        assert!(msg.contains("underflowed"), "got: {msg}");
    }

    #[test]
    fn process_vm_readv_err_renders_address_hex() {
        let err = ThreadProbeError::process_vm_readv_err(
            123,
            0xdeadbeef,
            nix::errno::Errno::EFAULT,
        );
        assert_eq!(err.kind, ThreadErrorKind::ProcessVmReadv);
        let msg = format!("{}", err.source);
        assert!(msg.contains("tid 123"), "got: {msg}");
        // Address MUST render as hex (format spec `{:#x}`) so the
        // operator can correlate with /proc/<pid>/maps.
        assert!(msg.contains("0xdeadbeef"), "got: {msg}");
    }

    #[test]
    fn process_vm_readv_short_records_got_and_expected() {
        let err = ThreadProbeError::process_vm_readv_short(200, 12, 24);
        assert_eq!(err.kind, ThreadErrorKind::ProcessVmReadv);
        let msg = format!("{}", err.source);
        assert!(msg.contains("short process_vm_readv on tid 200"), "got: {msg}");
        assert!(msg.contains("got 12 bytes"), "got: {msg}");
        assert!(msg.contains("expected 24"), "got: {msg}");
    }
}
