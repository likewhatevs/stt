//! Standalone jemalloc per-thread counter probe.
//!
//! Reads the `thread_allocated` / `thread_deallocated` TLS counters
//! out of a running jemalloc-linked process. The counters are
//! maintained unconditionally on jemalloc's alloc/dalloc fast + slow
//! paths (see jemalloc_internal_inlines_c.h:277, 574 and
//! thread_event.h:117-119), so attaching late does not miss prior
//! allocations — the reading is cumulative from thread creation.
//!
//! Entry point: `--pid <PID>`. Attaches to every thread in the
//! target process via ptrace, reads each thread's TSD counters
//! through `process_vm_readv`, detaches. DWARF is resolved against
//! the target's `/proc/<pid>/exe`, so the target ELF must ship with
//! debuginfo. End-to-end validation runs via the `#[ktstr_test]`
//! integration tests in `tests/jemalloc_probe_tests.rs`, which boot
//! a VM, spawn a jemalloc-linked allocator worker, and run the probe
//! against the worker's live pid.
//!
//! Scope:
//! - Linux, x86_64 and aarch64. Same-arch only (a probe binary built
//!   for x86_64 only handles x86_64 targets; ptrace is same-arch).
//! - Static-linked jemalloc only (symbol lives in the main
//!   executable's static TLS image).
//! - Requires DWARF debuginfo on the target ELF and CAP_SYS_PTRACE /
//!   root / same-uid-as-target.
//!
//! Mechanism (per target thread):
//! 1. `ptrace(PTRACE_SEIZE)` + `ptrace(PTRACE_INTERRUPT)` to stop.
//! 2. Read the thread pointer via `ptrace(PTRACE_GETREGSET, ...)`:
//!    - x86_64 uses `NT_PRSTATUS` to get `user_regs_struct.fs_base`.
//!    - aarch64 uses `NT_ARM_TLS` (regset 0x401) to get `TPIDR_EL0`.
//! 3. `process_vm_readv` 24 bytes at the computed TLS address to read
//!    `thread_allocated` + `thread_allocated_next_event_fast` +
//!    `thread_deallocated` in one syscall while the thread is stopped.
//! 4. `ptrace(PTRACE_DETACH)`.
//!
//! Address math:
//! - Variant II (x86_64): TP points to END of TLS block.
//!     addr(tsd_tls) = fs_base - round_up(PT_TLS.p_memsz, PT_TLS.p_align) + st_value
//!     addr(field)   = addr(tsd_tls) + offsetof(tsd_s, field)
//! - Variant I (aarch64): TP points to start of the 16-byte TCB
//!   header; TLS block starts after the header, aligned up to
//!   PT_TLS.p_align (AArch64 ELF ABI, IHI 0056D §4.1).
//!     addr(tsd_tls) = TPIDR_EL0 + round_up(16, PT_TLS.p_align) + st_value
//!     addr(field)   = addr(tsd_tls) + offsetof(tsd_s, field)

// Link jemalloc as the global allocator for binary-homogeneity
// across ktstr bins — the probe does NOT read its own TSD, so the
// choice here is not a correctness requirement. Matching the
// `#[global_allocator]` declaration in src/bin/ktstr.rs and
// src/bin/cargo-ktstr.rs keeps allocator policy uniform across the
// workspace's shipped binaries: the same allocator runs when a user
// invokes any ktstr tool, and future cross-binary comparisons stay
// apples-to-apples.
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
#[cfg(target_arch = "x86_64")]
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
const SCHEMA_VERSION: u32 = 2;

/// Capture the current wall-clock as Unix epoch seconds. `unwrap_or(0)`
/// handles the impossible pre-epoch-clock case defensively — KVM
/// guests under kvm-clock or NTP always resolve post-1970, so the
/// zero is a never-fires safety net rather than a real fallback.
/// Factored so `run()` and any future probe-output site reach for
/// the same helper instead of re-typing the
/// `SystemTime::now().duration_since(UNIX_EPOCH)...` chain.
fn now_unix_sec() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The probe's own pid as an `i32`. Linux enforces `pid_max <= 2^22`
/// (kernel/pid.c), so the `u32 → i32` conversion is infallible in
/// practice; the `expect` documents that invariant. Used in the
/// self-probe guard and test bodies — centralized so a future
/// platform constraint change lands in one place.
fn self_pid() -> i32 {
    libc::pid_t::try_from(std::process::id())
        .expect("Linux pid_max <= 2^22 so pid fits in pid_t")
}

/// Render the optional per-thread comm string as a trailing
/// `" comm=<name>"` fragment for the human-readable output path, or
/// the empty string when comm is absent. Shared by the Ok and Err
/// arms of [`print_thread_result`] so both lines use identical
/// formatting — a future consumer that greps for ` comm=` catches
/// both. Factored to eliminate the open-coded
/// `.as_deref().map(|c| format!(" comm={c}")).unwrap_or_default()`
/// chain duplicated at every call site.
fn format_comm_suffix(comm: Option<&str>) -> String {
    comm.map(|c| format!(" comm={c}")).unwrap_or_default()
}

/// Per-target-arch primitives: thread pointer read via ptrace on the
/// stopped target, the expected ELF `e_machine`, the regset name for
/// error messages, and a human-readable arch name. Gated on
/// `target_arch` — a probe binary built for x86_64 only handles
/// x86_64 targets (ptrace is same-arch). Both
/// Variants are exposed as pure arithmetic (see
/// [`compute_tls_address_variant_i`] / [`compute_tls_address_variant_ii`])
/// so unit tests for either can run on any host regardless of
/// `target_arch`.
mod arch {
    use super::*;

    /// ELF `e_machine` value the probe is willing to probe. Matches
    /// `target_arch`: a probe built for x86_64 rejects aarch64 targets
    /// and vice versa. The check lives in [`find_jemalloc_via_maps`]
    /// upstream of the DWARF walk so arch mismatches fail fast.
    #[cfg(target_arch = "x86_64")]
    pub const EXPECTED_E_MACHINE: u16 = goblin::elf::header::EM_X86_64;
    #[cfg(target_arch = "aarch64")]
    pub const EXPECTED_E_MACHINE: u16 = goblin::elf::header::EM_AARCH64;

    /// Human-readable name of the arch this probe build targets.
    /// Used only in diagnostic messages — the JSON schema carries the
    /// target ELF's `e_machine` as a hex value elsewhere.
    #[cfg(target_arch = "x86_64")]
    pub const ARCH_NAME: &str = "x86_64";
    #[cfg(target_arch = "aarch64")]
    pub const ARCH_NAME: &str = "aarch64";

    /// Name of the regset this build passes to PTRACE_GETREGSET when
    /// reading the target thread's TP. Surfaced in the
    /// [`ThreadErrorKind::GetRegset`] error message so an operator
    /// grepping `warning: tid X [get_regset]` sees the arch-correct
    /// register name — `NT_PRSTATUS` on x86_64 (for `fs_base` inside
    /// `user_regs_struct`), `NT_ARM_TLS` on aarch64 (for
    /// `tpidr_el0`).
    #[cfg(target_arch = "x86_64")]
    pub const REGSET_NAME: &str = "NT_PRSTATUS";
    #[cfg(target_arch = "aarch64")]
    pub const REGSET_NAME: &str = "NT_ARM_TLS";

    /// `NT_ARM_TLS` regset number, from
    /// `linux/include/uapi/linux/elf.h`. `nix` does not expose this
    /// regset (its `RegisterSetValue` enum is closed and only carries
    /// NT_PRSTATUS / NT_PRFPREG / NT_PRPSINFO / NT_TASKSTRUCT /
    /// NT_AUXV), so the aarch64 read path calls `libc::ptrace`
    /// directly with the raw regset value.
    #[cfg(target_arch = "aarch64")]
    pub const NT_ARM_TLS: libc::c_int = 0x401;

    /// Read the stopped target thread's TP via ptrace.
    ///
    /// - x86_64: `ptrace(PTRACE_GETREGSET, pid, NT_PRSTATUS, ...)`
    ///   returns `user_regs_struct.fs_base`.
    /// - aarch64: `ptrace(PTRACE_GETREGSET, pid, NT_ARM_TLS, ...)`
    ///   returns `[tpidr_el0, tpidr2_el0]` on kernels with TPIDR2
    ///   support, or a single `tpidr_el0` on older kernels. We
    ///   request only the first 8 bytes (tpidr_el0) via the iovec's
    ///   `iov_len`, so the read is version-stable across both.
    #[cfg(target_arch = "x86_64")]
    pub fn read_thread_pointer_ptrace(pid: Pid) -> std::result::Result<u64, nix::errno::Errno> {
        let regs = ptrace::getregset::<NT_PRSTATUS>(pid)?;
        Ok(regs.fs_base)
    }

    #[cfg(target_arch = "aarch64")]
    pub fn read_thread_pointer_ptrace(pid: Pid) -> std::result::Result<u64, nix::errno::Errno> {
        let mut tpidr: u64 = 0;
        let mut iov = libc::iovec {
            iov_base: (&mut tpidr as *mut u64).cast::<libc::c_void>(),
            iov_len: std::mem::size_of::<u64>(),
        };
        // SAFETY: `libc::ptrace` is variadic; the addresses passed
        // must be valid for the duration of the call. `iov.iov_base`
        // points at a stack u64 and `&mut iov` points at a stack
        // iovec — both live for the entire call.
        let res = unsafe {
            libc::ptrace(
                libc::PTRACE_GETREGSET,
                pid.as_raw(),
                NT_ARM_TLS as libc::c_long,
                &mut iov as *mut libc::iovec,
            )
        };
        if res == -1 {
            return Err(nix::errno::Errno::last());
        }
        Ok(tpidr)
    }

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
///
/// # Sampling modes
///
/// - **Single snapshot (default)**: `--snapshots 1` or omitted. The
///   probe emits one entry in the top-level `snapshots` array with
///   `interval_ms` absent.
/// - **Multi-snapshot**: `--snapshots N --interval-ms MS` for
///   `N > 1`. The probe resolves jemalloc symbols + enumerates tids
///   ONCE up-front, then performs N attach/read/detach cycles per
///   tid separated by `interval_ms` of sleep. The repeated work is
///   the per-tid ptrace dance; the setup (ELF/DWARF parse, tid
///   enumeration) is amortized across all N snapshots. The top-level
///   `snapshots` array carries one entry per snapshot. Threads are
///   NOT held stopped between snapshots — each tid is detached
///   before the inter-snapshot sleep so the target workload
///   continues to run.
#[derive(Parser, Debug)]
#[command(
    name = "ktstr-jemalloc-probe",
    version = env!("CARGO_PKG_VERSION"),
    about = "Read per-thread jemalloc allocated/deallocated byte counters from a running process",
    long_about = "Reads jemalloc's per-thread `thread_allocated` / `thread_deallocated` TLS \
                  counters out of a running process via ptrace + process_vm_readv. Counters are \
                  cumulative from thread creation — attaching late does not miss prior \
                  allocations. Requires CAP_SYS_PTRACE, root, or same-uid-as-target. Supports \
                  Linux x86_64 and aarch64 (same-arch only) targets with a statically-linked \
                  jemalloc and DWARF debuginfo on the binary carrying the jemalloc TLS \
                  symbol.\n\n\
                  The `--enable-stats` jemalloc build flag is NOT required: `thread.allocated` / \
                  `thread.deallocated` use jemalloc's `CTL_RO_NL_GEN` (ungated) and the fast/slow \
                  path writes are unconditional.\n\n\
                  Sampling mode: pass `--snapshots N --interval-ms MS` to take N snapshots \
                  separated by MS milliseconds. Symbol resolution and tid enumeration run \
                  once; each snapshot attach/detaches per tid and threads are released during \
                  the inter-snapshot sleep so the workload is not held stopped across the \
                  run. The JSON output always carries a `snapshots` array — single snapshot \
                  is an array of length 1.\n\n\
                  Sidecar enrichment: pass `--sidecar PATH` to append probe metrics into an \
                  existing ktstr sidecar file. The file MUST exist — run the test first to \
                  generate it, then re-invoke with `--sidecar`."
)]
struct Cli {
    /// Target process id. Required. Must be a positive integer; pid
    /// 0 and negative values are rejected at parse time since Linux
    /// tgids start at 1.
    #[arg(long, value_parser = clap::value_parser!(i32).range(1..))]
    pid: i32,
    /// Emit JSON on stdout instead of a human-readable table.
    #[arg(long)]
    json: bool,
    /// Append probe output to an existing ktstr sidecar JSON file
    /// (`target/ktstr/{kernel}-{git}/{test_name}-{hash}.ktstr.json`).
    /// The probe reads the existing [`SidecarResult`], synthesizes a
    /// [`PayloadMetrics`] entry from its own output (walking
    /// numeric JSON leaves into `name: value` [`Metric`] records),
    /// appends it to `sidecar.metrics`, and writes the result back
    /// atomically (tempfile + rename) under an exclusive advisory
    /// lock (`flock(LOCK_EX)`) so concurrent `--sidecar` calls
    /// serialize.
    ///
    /// The sidecar file MUST already exist — the probe will not
    /// synthesize a fresh `SidecarResult`, since most fields
    /// (monitor, stimulus_events, verifier_stats, host context)
    /// cannot be honestly populated from a standalone probe run.
    /// Run the target test first so the harness writes the
    /// sidecar, then invoke the probe with `--sidecar` to enrich
    /// it post-hoc. The path is pre-flight-validated immediately
    /// after `Cli::parse()` so a typo fails fast instead of after
    /// the full ptrace run.
    ///
    /// **Multi-snapshot runs**: produce one `PayloadMetrics` entry
    /// containing all snapshots' flattened leaves —
    /// `snapshots.0.threads.N.allocated_bytes`,
    /// `snapshots.1.threads.N.allocated_bytes`, etc. — not one
    /// entry per snapshot. Downstream stats tooling diffing across
    /// snapshots keys on the snapshot index in the metric name.
    ///
    /// **Fatal errors do NOT modify the sidecar** — a `Fatal`
    /// outcome (pid missing, exe-identity-changed, not-jemalloc)
    /// never produces a usable `ProbeOutput` to flatten.
    /// `AllFailed` DOES append with `exit_code: 1` so consumers
    /// keying on `ExitCodeEq(0)`-equivalents see the failure.
    ///
    /// **Metric namespace**: appended metrics use the
    /// `jemalloc_probe.*` prefix so downstream aggregators walking
    /// the full `Vec<PayloadMetrics>` can discriminate probe-
    /// sourced leaves from the test's primary payload metrics.
    ///
    /// Orthogonal to `--json` — the stdout emission is independent
    /// of the sidecar write, so `--sidecar` invocations remain
    /// debuggable without re-reading the sidecar.
    #[arg(long, value_name = "PATH")]
    sidecar: Option<PathBuf>,
    /// Number of snapshots to take. Defaults to 1 (single-snapshot
    /// mode). Values > 1 engage multi-snapshot mode and require
    /// `--interval-ms`. Range is 1..=100_000; the upper cap bounds
    /// the pre-allocated snapshot vector so a runaway `--snapshots`
    /// cannot request a multi-GiB allocation before any ptrace work
    /// runs.
    #[arg(
        long,
        default_value_t = 1,
        value_parser = clap::value_parser!(u32).range(1..=100_000),
        value_name = "N",
    )]
    snapshots: u32,
    /// Milliseconds to wait between consecutive snapshots. Required
    /// (and only meaningful) when `--snapshots > 1`. Range is
    /// 1..=3_600_000 (1 ms to 1 hour); the upper cap bounds the
    /// max single-run duration and guarantees the `Instant + Duration`
    /// deadline math in [`sleep_with_cancel`] cannot overflow.
    ///
    /// **Delay precision** (`--interval-ms` → actual wait): the
    /// configured delay is honored at the requested millisecond
    /// granularity. `sleep_with_cancel` computes a deadline once at
    /// entry and returns precisely when `Instant::now() >= deadline`;
    /// sub-1ms clock jitter only affects the return instant, not the
    /// accrued delay.
    ///
    /// **SIGINT response latency**: orthogonal to delay precision.
    /// `std::thread::sleep` is not signal-aware, so SIGINT / SIGTERM
    /// cannot shorten an in-flight sleep directly. The loop chunks
    /// the remaining wait by `remaining.min(tick)` with `tick =`
    /// [`CANCEL_POLL_TICK_MS`] (10 ms). For intervals >=
    /// `CANCEL_POLL_TICK_MS` the SIGINT latency is bounded by one
    /// poll tick (~10 ms). For intervals < 10 ms the per-iteration
    /// sleep equals the configured interval, so latency degrades
    /// gracefully. Upper bound is always 10 ms, independent of how
    /// large the configured interval is.
    #[arg(
        long,
        value_parser = clap::value_parser!(u64).range(1..=3_600_000),
        value_name = "MS",
    )]
    interval_ms: Option<u64>,
}

impl Cli {
    /// Validate `--snapshots` / `--interval-ms` combination consistency
    /// beyond what clap's declarative attributes cover. Specifically:
    /// `--snapshots > 1` requires `--interval-ms`, and `--interval-ms`
    /// without `--snapshots > 1` is rejected as a user-intent mismatch.
    ///
    /// Run from [`main`] immediately after `Cli::parse()`; a failure
    /// here aborts the run with a usage-style stderr message and
    /// non-zero exit.
    fn validate_sampling_flags(&self) -> Result<()> {
        if self.snapshots > 1 && self.interval_ms.is_none() {
            bail!(
                "--snapshots {} requires --interval-ms <MS>; multi-snapshot sampling \
                 needs an explicit inter-snapshot wait",
                self.snapshots,
            );
        }
        if self.snapshots == 1 && self.interval_ms.is_some() {
            bail!(
                "--interval-ms is only meaningful with --snapshots > 1; omit --interval-ms \
                 for a single-snapshot run",
            );
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Output schema
// ---------------------------------------------------------------------

/// Unified probe JSON body. Single-snapshot and multi-snapshot runs
/// both emit this shape; single-snapshot produces a `snapshots` array
/// of length 1 with `interval_ms` absent. Consumers distinguish the
/// two modes by `interval_ms` presence or `snapshots.len()`.
///
/// **Tid enumeration is frozen at run start.** `iter_task_ids` runs
/// once before the first snapshot and the same tid list is reused
/// for every subsequent snapshot. Threads spawned AFTER the initial
/// enumeration are invisible to the probe — they will not appear
/// in any snapshot even if they outlive the run. Threads that exit
/// between snapshots produce [`ThreadResult::Err`] entries (typically
/// `PtraceSeize` with ESRCH) on subsequent snapshots rather than
/// disappearing, so index-wise diffing across `snapshots[*].threads`
/// is stable.
#[derive(Debug, Serialize)]
struct ProbeOutput {
    schema_version: u32,
    pid: i32,
    tool_version: &'static str,
    /// Unix-epoch seconds at the start of the probe run (before any
    /// per-tid work, before the first snapshot). Intended for
    /// downstream diff tooling that correlates multiple probe runs
    /// against a workload timeline — an absolute timestamp lets
    /// callers align probe captures with other sidecar-emitted
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
    started_at_unix_sec: u64,
    /// Configured inter-snapshot delay in milliseconds. Present only
    /// on multi-snapshot runs (`--snapshots > 1`); omitted via
    /// `skip_serializing_if` for single-snapshot runs so the wire
    /// shape flags mode explicitly. Useful for downstream tooling
    /// that wants to correlate observed `snapshots[*].timestamp_unix_sec`
    /// deltas against the configured cadence.
    #[serde(skip_serializing_if = "Option::is_none")]
    interval_ms: Option<u64>,
    /// `true` iff the run ended early because a SIGINT / SIGTERM
    /// arrived during the snapshot loop or inter-snapshot sleep.
    /// The `snapshots` array carries every snapshot started before
    /// the signal landed, INCLUDING a partial final snapshot whose
    /// per-tid loop was interrupted mid-iteration: its `threads`
    /// array is truncated to the tids that completed before the
    /// signal. Callers observing `interrupted: true` must expect
    /// the last entry in `snapshots` to potentially cover fewer
    /// tids than earlier entries.
    ///
    /// `false` on a normal completion.
    interrupted: bool,
    snapshots: Vec<Snapshot>,
}

/// One snapshot inside [`ProbeOutput::snapshots`]. Carries the
/// timestamp + per-thread counters observed by a single iteration
/// of the sampling loop. Thread ids come from the tid enumeration
/// captured ONCE at run start (see [`ProbeOutput`] for the frozen-
/// tid-list contract).
#[derive(Debug, Serialize)]
struct Snapshot {
    /// Unix-epoch seconds at the start of this snapshot's per-tid
    /// attach/read/detach loop. Same clock-source semantics as
    /// [`ProbeOutput::started_at_unix_sec`].
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
    /// `ptrace(PTRACE_GETREGSET, <regset>)` failed — the target
    /// tid exited between attach and register fetch, or the target
    /// is not the expected arch for this probe build (the arch
    /// check refuses cross-arch targets upstream of this path, but
    /// this variant is held as belt-and-braces).
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
    /// to reach the start of the TLS image under the Variant II
    /// model (x86_64). Not used by Variant I (aarch64), which only
    /// needs [`TsdTlsSymbol::p_align`].
    pub tls_image_aligned_size: u64,
    /// Raw `PT_TLS.p_align` value. Variant I (aarch64) needs this
    /// to compute `round_up(TCB_SIZE_AARCH64, p_align)` — the offset
    /// from TP to the TLS image base. Retained alongside
    /// `tls_image_aligned_size` for Variant II back-compat rather
    /// than collapsing into the aligned value, because the two
    /// formulas diverge on the arg they need.
    pub p_align: u64,
    /// ELF architecture e_machine value — matched against the probe's
    /// compile-time [`arch::EXPECTED_E_MACHINE`] so a probe built for
    /// x86_64 refuses an aarch64 target (and vice versa) with a clear
    /// error upstream of the ptrace dance.
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
    let (tls_image_aligned_size, p_align) = extract_pt_tls_layout(elf)?;

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
                    p_align,
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

/// Extract both `round_up(p_memsz, p_align)` and the raw `p_align`
/// from the ELF's `PT_TLS` program header. The first is Variant II's
/// TP-to-TLS-image delta (subtracted); the second feeds Variant I's
/// `round_up(TCB_SIZE_AARCH64, p_align)`. Returning both keeps the
/// ELF parse a single pass.
fn extract_pt_tls_layout(elf: &Elf<'_>) -> Result<(u64, u64)> {
    let tls_hdr = elf
        .program_headers
        .iter()
        .find(|ph| ph.p_type == goblin::elf::program_header::PT_TLS)
        .ok_or_else(|| anyhow!("ELF has no PT_TLS segment — target does not use static TLS"))?;
    // PT_TLS.p_align is a power of two (or zero) per the ELF spec
    // (and in practice for every Linux toolchain). The `& !(align - 1)`
    // round-up trick below assumes this invariant; `debug_assert!`
    // surfaces a non-power-of-two in debug builds before the silent
    // miscomputation reaches the probe's address arithmetic. Release
    // builds accept the ELF as-is — a malicious target isn't the
    // threat model.
    debug_assert!(
        tls_hdr.p_align == 0 || tls_hdr.p_align.is_power_of_two(),
        "PT_TLS.p_align must be 0 or a power of two, got {}",
        tls_hdr.p_align,
    );
    let align = tls_hdr.p_align.max(1);
    let rounded = tls_hdr
        .p_memsz
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| anyhow!("PT_TLS size arithmetic overflow"))?;
    Ok((rounded, align))
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

/// Reserved-area size at the low end of AArch64's Variant I thread-
/// control block — 2 words before the TLS image, per the AArch64 ELF
/// Linux ABI (IHI 0056D §4.1). The TLS image base is
/// `TP + round_up(TCB_SIZE_AARCH64, p_align)`; every TLS variable
/// sits at `tls_image_base + st_value + field_offset`.
#[allow(dead_code)] // used by the aarch64 dispatcher + Variant I unit tests
pub(crate) const TCB_SIZE_AARCH64: u64 = 16;

/// Variant II TLS address (x86_64).
///
/// The thread pointer (`fs_base`) points to the END of the static
/// TLS block; the executable's TLS image sits at
/// `fs_base - tls_image_aligned_size`. The symbol lives at
/// `st_value` bytes within that image; the field lives
/// `field_offset` bytes inside the symbol.
///
/// Returns `Err` on `fs_base < tls_image_aligned_size` — that would
/// indicate the target has not initialized TLS or the ELF layout is
/// malformed; silently wrapping into the top of the address space
/// would produce a read from kernel-space and confuse the error path.
pub(crate) fn compute_tls_address_variant_ii(
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

/// Variant I TLS address (aarch64).
///
/// `TPIDR_EL0` (the thread pointer) points to the BEGINNING of the
/// thread-control block; the executable's TLS image sits at
/// `TP + round_up(TCB_SIZE_AARCH64, p_align)`. The symbol lives at
/// `st_value` within that image; the field lives `field_offset`
/// bytes inside the symbol.
///
/// Every `checked_*` guard exists to catch an overflow that would
/// silently wrap into the high part of the address space and confuse
/// the error path — same rationale as
/// [`compute_tls_address_variant_ii`].
#[allow(dead_code)] // used by the aarch64 dispatcher + Variant I unit tests
pub(crate) fn compute_tls_address_variant_i(
    tpidr_el0: u64,
    p_align: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    // Round the TCB reserved area up to the TLS block's alignment.
    // Rust's integer arithmetic traps on underflow in debug builds;
    // the `.max(1)` guards against p_align=0 in a degenerate ELF.
    let align = p_align.max(1);
    let image_offset = TCB_SIZE_AARCH64
        .checked_add(align - 1)
        .map(|v| v & !(align - 1))
        .ok_or_else(|| {
            anyhow!(
                "TLS image offset overflow: tcb={} align={align:#x}",
                TCB_SIZE_AARCH64,
            )
        })?;
    tpidr_el0
        .checked_add(image_offset)
        .and_then(|v| v.checked_add(st_value))
        .and_then(|v| v.checked_add(field_offset))
        .ok_or_else(|| anyhow!("TLS address arithmetic overflow"))
}

/// Arch-dispatched TLS address compute. Routes to Variant II on
/// x86_64 and Variant I on aarch64 via `cfg(target_arch)`. Keeps
/// call site (`probe_single_thread`) arch-neutral.
#[cfg(target_arch = "x86_64")]
pub(crate) fn compute_tls_address(
    tp: u64,
    tls_image_aligned_size: u64,
    _p_align: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    compute_tls_address_variant_ii(tp, tls_image_aligned_size, st_value, field_offset)
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn compute_tls_address(
    tp: u64,
    _tls_image_aligned_size: u64,
    p_align: u64,
    st_value: u64,
    field_offset: u64,
) -> Result<u64> {
    compute_tls_address_variant_i(tp, p_align, st_value, field_offset)
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
/// which v1 does not implement.
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
        // Arch check runs before the (slow) DWARF walk so a
        // cross-arch target fails fast with the right message instead
        // of running gimli over unsupported debug info. The probe is
        // same-arch only: a probe binary built for x86_64 only probes
        // x86_64 targets; aarch64 build only probes aarch64. Cross-
        // arch ptrace is not supported.
        if symbol.e_machine != arch::EXPECTED_E_MACHINE {
            bail!(
                "probe is {}-only; target ELF {} is {} (e_machine={:#x}). \
                 Obtain or build a probe matching the target's architecture \
                 (ptrace is same-arch only — the probe and its target must \
                 share the same machine type).",
                arch::ARCH_NAME,
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

/// Acquire the ATTACHED mutex, recovering from poisoning so a panic
/// in one thread cannot prevent detach cleanup from running in the
/// next. The tracked set is a plain `BTreeSet<i32>` of tids; any
/// panic that poisoned it left the set in a valid state (inserts /
/// removes are transactional), so `into_inner` on the poison error
/// yields the same usable guard. `.unwrap()` was a double-panic
/// hazard — if `ScopeDetach::drop` runs after another site poisoned
/// the mutex, unwrapping would unwind a Drop and abort the process.
fn attached_lock() -> std::sync::MutexGuard<'static, BTreeSet<i32>> {
    attached().lock().unwrap_or_else(|e| e.into_inner())
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
            anyhow!(
                "ptrace(PTRACE_GETREGSET, {}) on tid {tid}: {e}",
                arch::REGSET_NAME,
            ),
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
    // Construct the detach guard IMMEDIATELY after a successful seize
    // — before the `attached` set insert, before interrupt, before any
    // subsequent fallible step. If the following `attached().lock()`
    // panics (poisoned mutex), the guard's Drop still runs and the
    // tid is detached. A reversed order (insert → guard) would leak
    // the attach on that narrow window.
    let _attached_guard = ScopeDetach(tid);
    // Record the attach so the SIGINT handler's `detach_all_attached`
    // sweep sees this tid even if we crash or are interrupted before
    // `interrupt`/`waitpid`.
    attached_lock().insert(tid);

    ptrace::interrupt(pid).map_err(|e| ThreadProbeError::ptrace_interrupt(tid, e))?;
    match waitpid(pid, None) {
        Ok(WaitStatus::Stopped(_, _) | WaitStatus::PtraceEvent(_, _, _)) => {}
        Ok(other) => return Err(ThreadProbeError::waitpid_unexpected(tid, other)),
        Err(e) => return Err(ThreadProbeError::waitpid_err(tid, e)),
    }

    let thread_pointer = arch::read_thread_pointer_ptrace(pid)
        .map_err(|e| ThreadProbeError::getregset(tid, e))?;

    let addr = compute_tls_address(
        thread_pointer,
        symbol.tls_image_aligned_size,
        symbol.p_align,
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
        attached_lock().remove(&self.0);
    }
}

/// Detach everything still in `ATTACHED`. Called from the main loop
/// when SIGINT arrived between tids.
fn detach_all_attached() {
    let tids: Vec<i32> = attached_lock().iter().copied().collect();
    for tid in tids {
        let _ = ptrace::detach(Pid::from_raw(tid), None);
        attached_lock().remove(&tid);
    }
}

// ---------------------------------------------------------------------
// Orchestration + output
// ---------------------------------------------------------------------

/// Outcome classification so `main` can decide the exit code without
/// re-inspecting the `snapshots` vec. `AllFailed` still emits JSON
/// so callers have a machine-parseable explanation; `Fatal` is for
/// pre-probe errors (pid missing, no jemalloc, arch mismatch) where
/// there's no per-thread error to surface.
///
/// The classification criterion for `AllFailed` is "every
/// `ThreadResult` in every snapshot is an `Err`", i.e. the probe
/// never observed a single live counter across the whole run.
/// A multi-snapshot run that was interrupted by SIGINT / SIGTERM
/// but produced at least one successful per-thread observation
/// surfaces as `Ok` with `interrupted: true` on the output — the
/// partial data is still useful to the caller.
enum RunOutcome {
    Ok(ProbeOutput),
    AllFailed(ProbeOutput),
    Fatal(anyhow::Error),
}

/// Granularity (ms) at which [`sleep_with_cancel`] wakes to poll
/// [`CLEANUP_REQUESTED`]. Small enough that SIGINT / SIGTERM during a
/// multi-second interval aborts within one tick, large enough that the
/// polling itself is not measurable load.
const CANCEL_POLL_TICK_MS: u64 = 10;

/// Sleep for `total_ms` milliseconds or until [`CLEANUP_REQUESTED`] is
/// observed, whichever is first. Returns `true` if the sleep was
/// cancelled by the cleanup flag, `false` if it completed normally.
///
/// `std::thread::sleep` is not signal-aware — a signal delivered
/// during a long sleep does not shorten it — so the loop polls at
/// [`CANCEL_POLL_TICK_MS`] granularity. A signal handler that sets
/// the flag therefore unblocks cleanup within one tick regardless of
/// the configured inter-snapshot interval.
///
/// Clap bounds `--interval-ms` to `1..=3_600_000`, so on a normal
/// invocation the `Instant + Duration` deadline math cannot overflow.
/// [`Instant::checked_add`] below is a belt-and-suspenders saturation:
/// an `Instant` near the platform representation's upper bound would
/// otherwise panic on overflow in debug builds. `Instant` has no
/// `saturating_add`, so on `None` we pin the deadline to `now` —
/// the function returns `false` without sleeping, which is the
/// correct degenerate behavior for a deadline that cannot be
/// represented.
fn sleep_with_cancel(total_ms: u64) -> bool {
    let start = std::time::Instant::now();
    let deadline = start
        .checked_add(std::time::Duration::from_millis(total_ms))
        .unwrap_or(start);
    loop {
        if CLEANUP_REQUESTED.load(Ordering::SeqCst) {
            return true;
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return false;
        }
        let remaining = deadline - now;
        let tick = std::time::Duration::from_millis(CANCEL_POLL_TICK_MS);
        std::thread::sleep(remaining.min(tick));
    }
}

/// Take one snapshot: iterate the tids, probe each, return a
/// [`Snapshot`] carrying the timestamp + per-thread results. Shared
/// symbol + offsets are passed in so the expensive ELF/DWARF parse
/// and tid enumeration amortize across all snapshots in a multi-
/// snapshot run.
///
/// On SIGINT / SIGTERM between tids the function detaches every
/// still-attached tid and returns the partial snapshot with
/// `interrupted = true`. The caller is responsible for turning that
/// into a `RunOutcome::Fatal`.
fn take_snapshot(
    pid: i32,
    symbol: &TsdTlsSymbol,
    offsets: &CounterOffsets,
    tids: &[i32],
) -> (Snapshot, bool) {
    // Capture timestamp BEFORE iterating threads so the field
    // represents "start of this snapshot" — a post-loop capture
    // would tail the variable-length per-thread ptrace work and
    // drift as the snapshot progresses.
    let timestamp_unix_sec = now_unix_sec();
    let mut threads: Vec<ThreadResult> = Vec::with_capacity(tids.len());
    let mut interrupted = false;
    for &tid in tids {
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
        match probe_single_thread(tid, symbol, offsets) {
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
    (
        Snapshot {
            timestamp_unix_sec,
            threads,
        },
        interrupted,
    )
}

/// True iff `threads` is empty or every entry is a
/// [`ThreadResult::Err`]. Used to decide between the
/// `Ok` / `AllFailed` run outcomes for single-snapshot runs and
/// (collectively across all snapshots) for multi-snapshot runs.
fn all_failed(threads: &[ThreadResult]) -> bool {
    threads.is_empty() || threads.iter().all(|t| matches!(t, ThreadResult::Err { .. }))
}

/// Stable identity of the target's on-disk executable, captured by
/// `stat(2)` on `/proc/<pid>/exe`. (dev, inode) uniquely identifies
/// the file; re-stating between snapshots lets the probe detect a
/// mid-run `execve` (new inode, same pid) or pid recycling (pid
/// reused for a different executable) and bail with `Fatal` rather
/// than reading stale TLS offsets from a process that no longer
/// matches the ELF/DWARF parse done at run start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExeIdentity {
    dev: u64,
    ino: u64,
}

impl ExeIdentity {
    fn capture(pid: i32) -> Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let path = format!("/proc/{pid}/exe");
        let md = fs::metadata(&path).with_context(|| format!("stat {path}"))?;
        Ok(Self {
            dev: md.dev(),
            ino: md.ino(),
        })
    }
}

fn run(cli: &Cli) -> RunOutcome {
    // Capture run-start timestamp first so every `ProbeOutput` built
    // below — success, all-failed, interrupted — carries the same
    // `started_at_unix_sec`. Taking it inside each arm would drift
    // with the variable pre-probe setup latency.
    let started_at_unix_sec = now_unix_sec();
    let pid = cli.pid;
    // Self-probe reject: PTRACE_SEIZE refuses a tracer's own tgid —
    // ptrace semantics say a process cannot attach to itself. Catching
    // this at the CLI boundary produces an actionable error instead of
    // a per-thread EPERM cascade mid-run that looks like a permissions
    // problem.
    let self_pid = self_pid();
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

    // Capture the target ELF's (dev, inode) BEFORE the ELF/DWARF
    // parse so the parse itself is inside the identity window. A
    // capture taken only AFTER the parse would miss an execve that
    // landed DURING the parse — the symbol + offsets from
    // `find_jemalloc_via_maps` could already be tied to a replaced
    // binary by the time we start sampling.
    let exe_identity = match ExeIdentity::capture(pid) {
        Ok(v) => v,
        Err(e) => return RunOutcome::Fatal(e),
    };

    // Symbol + offset resolution and tid enumeration are run ONCE
    // even when `--snapshots > 1`. The ELF/DWARF parse in
    // `find_jemalloc_via_maps` is the expensive non-per-thread step
    // and was the motivation for sampling mode — repeating it per
    // snapshot would defeat the amortization.
    let (symbol, offsets) = match find_jemalloc_via_maps(pid) {
        Ok(v) => v,
        Err(e) => return RunOutcome::Fatal(e),
    };

    // Re-stat AFTER the parse. If the target execve'd during the
    // parse window the symbol/offsets we cached no longer match
    // /proc/<pid>/exe, and subsequent snapshots would read TLS
    // offsets from a DIFFERENT binary. Bail before any per-tid
    // ptrace work runs.
    match ExeIdentity::capture(pid) {
        Ok(current) if current != exe_identity => {
            return RunOutcome::Fatal(anyhow!(
                "target pid {pid} /proc/<pid>/exe changed during ELF/DWARF parse \
                 (captured dev={:#x} ino={}, now dev={:#x} ino={}); \
                 target execve'd mid-parse or pid recycled, TLS offsets invalid",
                exe_identity.dev,
                exe_identity.ino,
                current.dev,
                current.ino,
            ));
        }
        Ok(_) => {}
        Err(e) => return RunOutcome::Fatal(e),
    }

    let tids = match iter_task_ids(pid) {
        Ok(v) => v,
        Err(e) => return RunOutcome::Fatal(e),
    };

    // `validate_sampling_flags` enforced that `interval_ms` is Some
    // exactly when `snapshots > 1`; unwrap is guarded by that check
    // in the multi-snapshot branch below.
    let snapshot_count = cli.snapshots as usize;
    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(snapshot_count);
    let mut interrupted = false;
    for i in 0..cli.snapshots {
        if CLEANUP_REQUESTED.load(Ordering::SeqCst) {
            interrupted = true;
            break;
        }
        // Re-stat the target's /proc/<pid>/exe between snapshots.
        // A changed (dev, ino) means the target execve'd or pid
        // recycled — either way the TLS offsets we cached are no
        // longer valid and subsequent snapshots would read garbage.
        // Skip on iteration 0 because `exe_identity` was just
        // captured before the loop.
        if i > 0 {
            match ExeIdentity::capture(pid) {
                Ok(current) if current != exe_identity => {
                    return RunOutcome::Fatal(anyhow!(
                        "target pid {pid} /proc/<pid>/exe changed between snapshots \
                         (captured dev={:#x} ino={}, now dev={:#x} ino={}); \
                         target execve'd mid-run or pid recycled, TLS offsets invalid",
                        exe_identity.dev,
                        exe_identity.ino,
                        current.dev,
                        current.ino,
                    ));
                }
                Ok(_) => {}
                Err(e) => return RunOutcome::Fatal(e),
            }
        }
        let (snap, snap_interrupted) = take_snapshot(pid, &symbol, &offsets, &tids);
        snapshots.push(snap);
        if snap_interrupted {
            interrupted = true;
            break;
        }
        // No sleep after the LAST snapshot — the interval separates
        // consecutive snapshots, so N-1 sleeps for N snapshots. The
        // single-snapshot branch threads through here with no sleep
        // (the condition is false on the only iteration). Sleep is
        // cancellable; a SIGINT mid-sleep ends the run with the
        // snapshots taken so far + `interrupted: true` on the output.
        if i + 1 < cli.snapshots {
            let interval_ms = cli.interval_ms.expect(
                "interval_ms guaranteed Some for snapshots > 1 by validate_sampling_flags",
            );
            if sleep_with_cancel(interval_ms) {
                interrupted = true;
                break;
            }
        }
    }

    let out = ProbeOutput {
        schema_version: SCHEMA_VERSION,
        pid,
        tool_version: env!("CARGO_PKG_VERSION"),
        started_at_unix_sec,
        interval_ms: cli.interval_ms,
        interrupted,
        snapshots,
    };
    let all_err = out
        .snapshots
        .iter()
        .all(|s| all_failed(&s.threads));
    if all_err {
        RunOutcome::AllFailed(out)
    } else {
        RunOutcome::Ok(out)
    }
}

/// Render one `ThreadResult` to stdout (Ok path) or stderr (Err
/// path) in the human-readable format shared by single-snapshot
/// and multi-snapshot modes. Extracted so both code paths stay in
/// lock-step for the exact wording every operator greps against.
fn print_thread_result(t: &ThreadResult) {
    match t {
        ThreadResult::Ok {
            tid,
            comm,
            allocated_bytes,
            deallocated_bytes,
        } => {
            let comm_suffix = format_comm_suffix(comm.as_deref());
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
            let comm_suffix = format_comm_suffix(comm.as_deref());
            eprintln!("warning: tid {tid}{comm_suffix} [{error_kind}]: {error}");
        }
    }
}

/// Emit [`ProbeOutput`] in the selected format. JSON wraps the
/// whole structure; human-readable text prefixes each snapshot with
/// a `--- snapshot N/M @ <unix_sec>s ---` banner so a text consumer
/// can `grep '^---'` to find snapshot boundaries. The banner is
/// emitted in BOTH single- and multi-snapshot runs so the text
/// format stays constant — a consumer parsing text does not need to
/// branch on the snapshot count.
fn print_output(cli: &Cli, out: &ProbeOutput) -> Result<()> {
    if cli.json {
        let s = serde_json::to_string_pretty(out)?;
        println!("{s}");
    } else {
        println!("pid={} tool_version={}", out.pid, out.tool_version);
        let total = out.snapshots.len();
        for (i, snap) in out.snapshots.iter().enumerate() {
            println!(
                "--- snapshot {n}/{total} @ {ts}s ---",
                n = i + 1,
                ts = snap.timestamp_unix_sec,
            );
            for t in &snap.threads {
                print_thread_result(t);
            }
        }
    }
    Ok(())
}

/// Payload name recorded as an identifying metric when the probe
/// appends to a sidecar. Not an existing `Payload` fixture — the
/// probe enters the sidecar out-of-band, not through the
/// `ctx.payload()` pipeline, so there is no `Payload::name` to
/// reuse. The prefix on every metric lets downstream stats tooling
/// distinguish probe-sourced metrics from the test's primary
/// payload metrics when iterating `SidecarResult::metrics`.
const SIDECAR_METRIC_PREFIX: &str = "jemalloc_probe";

/// Upgrade a [`Metric`]'s unit + polarity based on its flat-path
/// name. Names ending in `.allocated_bytes` or `.deallocated_bytes`
/// become `(Polarity::LowerBetter, "bytes")` — memory usage is a
/// cost, a regression is an increase. Every other name is left at
/// the walker's default `(Polarity::Unknown, "")`. Stats tooling
/// normally applies hints from [`Payload::metrics`] during in-
/// harness runs; the probe has no `Payload` fixture in the sidecar
/// path, so hints are applied here directly.
fn apply_probe_metric_hints(m: &mut ktstr::test_support::Metric) {
    use ktstr::test_support::Polarity;
    // Match on suffixes to stay robust to the snapshot-index prefix
    // (`snapshots.0.threads.3.allocated_bytes` ends in
    // `allocated_bytes` just like the top-level `allocated_bytes` in
    // a hypothetical future schema).
    if m.name.ends_with(".allocated_bytes") || m.name.ends_with(".deallocated_bytes") {
        m.polarity = Polarity::LowerBetter;
        m.unit = "bytes".to_string();
    }
}

/// Synthesize a [`PayloadMetrics`] from a [`ProbeOutput`] so the
/// result can land in a [`SidecarResult::metrics`] vec. The probe's
/// JSON is passed through [`walk_json_leaves`] with
/// `MetricSource::Json` — same walker the in-harness payload
/// pipeline uses, re-exported from the ktstr lib so the probe and
/// the test harness share a single flattening contract. Every
/// resulting [`Metric::name`] is prefixed with
/// [`SIDECAR_METRIC_PREFIX`] + `.` so downstream consumers can
/// discriminate probe-sourced leaves from the test's primary payload
/// metrics when walking `SidecarResult::metrics` end-to-end.
fn synthesize_payload_metrics(
    out: &ProbeOutput,
    exit_code: i32,
) -> Result<ktstr::test_support::PayloadMetrics> {
    use ktstr::test_support::{MetricSource, PayloadMetrics, walk_json_leaves};
    let value = serde_json::to_value(out)
        .context("serialize ProbeOutput to serde_json::Value for sidecar append")?;
    let mut metrics = walk_json_leaves(&value, MetricSource::Json);
    for m in &mut metrics {
        // Prefix in place to avoid allocating a second Vec; the
        // capacity is exactly `metrics.len()` already.
        m.name = format!("{SIDECAR_METRIC_PREFIX}.{}", m.name);
        apply_probe_metric_hints(m);
    }
    Ok(PayloadMetrics { metrics, exit_code })
}

/// Append a synthesized [`PayloadMetrics`] to the
/// [`SidecarResult::metrics`] vec of the sidecar file at `path`.
/// The file is read, parsed, mutated, and written back atomically
/// via tempfile + rename under an exclusive advisory lock
/// (`flock(LOCK_EX)`) so concurrent `--sidecar` invocations against
/// the same file serialize rather than interleave.
///
/// Missing file is a hard error with an operator-actionable message:
/// the probe will not synthesize a fresh `SidecarResult`, since most
/// fields (monitor, stimulus_events, verifier_stats, host context)
/// cannot be honestly populated from a standalone probe run.
///
/// Malformed JSON is a hard error — the pre-1.0 sidecar policy is
/// "regenerate, don't migrate", so a parse failure points at an
/// out-of-sync schema rather than something the probe should paper
/// over.
fn append_probe_output_to_sidecar(
    path: &Path,
    out: &ProbeOutput,
    exit_code: i32,
) -> Result<()> {
    use ktstr::test_support::SidecarResult;
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, open};

    // Flock on a SIBLING lock file, not on the sidecar itself. The
    // atomic rename() below replaces the sidecar's inode, which
    // would invalidate any lock held on the old inode — a second
    // concurrent invocation would open the new inode and see no
    // lock. A fixed `<sidecar>.lock` path keeps every writer
    // agreeing on the same lock object regardless of how many
    // rename cycles the sidecar has been through.
    let lock_path = path.with_extension({
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext.is_empty() {
            "lock".to_string()
        } else {
            format!("{ext}.lock")
        }
    });
    // `CLOEXEC` so a fork/exec later in the process (e.g. the probe
    // spawning a child for some future reason, or stdlib helpers
    // that internally fork) does not leak the lock to the child —
    // a leaked lock-holding FD would deadlock any subsequent
    // `--sidecar` call that waits on the same path.
    let lock_fd = open(
        &lock_path,
        OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .with_context(|| format!("open lock file {}", lock_path.display()))?;
    // LOCK_EX (blocking). Concurrent `--sidecar` callers queue on
    // the same lock file rather than corrupting each other. The
    // lock is released when `lock_fd` drops at end-of-function.
    flock(&lock_fd, FlockOperation::LockExclusive)
        .with_context(|| format!("flock(LOCK_EX) on {}", lock_path.display()))?;

    // Read INSIDE the flock window — no separate `exists()` call.
    // `fs::read_to_string` itself reports `ErrorKind::NotFound` if
    // the file is absent, so we rewrite that one kind into the
    // operator-actionable message and let every other I/O error
    // propagate with the raw cause. One fewer syscall, no TOCTOU
    // between `exists()` and `open()`.
    let existing = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(
            "sidecar file not found at {}; run the test first to \
             generate it, then re-invoke with --sidecar",
            path.display(),
        ),
        Err(e) => return Err(anyhow::Error::from(e).context(format!("read {}", path.display()))),
    };
    let mut sidecar: SidecarResult = serde_json::from_str(&existing).with_context(|| {
        format!(
            "parse {} as SidecarResult — sidecar may be from an incompatible \
             schema version (pre-1.0 policy: regenerate, do not migrate)",
            path.display(),
        )
    })?;

    let payload_metrics = synthesize_payload_metrics(out, exit_code)?;
    sidecar.metrics.push(payload_metrics);

    let serialized = serde_json::to_string_pretty(&sidecar)
        .context("re-serialize SidecarResult after appending probe metrics")?;

    // Atomic write via `tempfile::NamedTempFile::new_in` in the
    // SAME directory as the target (same filesystem, so
    // `.persist()` is a rename(2), not a copy). NamedTempFile's
    // Drop impl removes the tempfile on panic or early return, so
    // no hand-rolled cleanup needed. Collision-free by construction.
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("sidecar path {} has no parent directory", path.display()))?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("create staging file in {}", dir.display()))?;
    std::io::Write::write_all(tmp.as_file_mut(), serialized.as_bytes())
        .with_context(|| format!("write staging file in {}", dir.display()))?;
    tmp.persist(path)
        .with_context(|| format!("atomic rename staging file into {}", path.display()))?;

    // `lock_fd` drops here; flock is released. Drop order: the
    // rename completed with the lock held, so any concurrent
    // `--sidecar` caller blocked on `flock(LOCK_EX)` will acquire
    // and see the new sidecar contents on its next read.
    drop(lock_fd);
    Ok(())
}

fn main() {
    install_cleanup_handler();
    let cli = Cli::parse();
    if let Err(e) = cli.validate_sampling_flags() {
        eprintln!("error: {e:#}");
        std::process::exit(2);
    }
    // Pre-flight `--sidecar` path validation so a typo (or a
    // user who forgot to run the test first) fails within tens of
    // milliseconds of invocation instead of after a multi-second
    // probe run. This is a UX fast-fail, NOT the correctness gate:
    // the real missing-file check lives inside
    // `append_probe_output_to_sidecar` INSIDE the flock window,
    // where TOCTOU cannot introduce a false positive. A file that
    // exists here and vanishes before the append fires surfaces as
    // the normal inside-flock error.
    if let Some(path) = cli.sidecar.as_deref()
        && !path.exists()
    {
        eprintln!(
            "error: sidecar file not found at {}; run the test \
             first to generate it, then re-invoke with --sidecar",
            path.display(),
        );
        std::process::exit(2);
    }
    match run(&cli) {
        RunOutcome::Ok(out) => {
            if let Err(e) = print_output(&cli, &out) {
                eprintln!("error writing output: {e:#}");
                std::process::exit(1);
            }
            if let Some(path) = cli.sidecar.as_deref()
                && let Err(e) = append_probe_output_to_sidecar(path, &out, 0)
            {
                eprintln!("error appending to sidecar {}: {e:#}", path.display());
                std::process::exit(1);
            }
        }
        RunOutcome::AllFailed(out) => {
            // Emit the structured output anyway so callers have the
            // per-thread error reasons; exit non-zero to signal that
            // nothing succeeded. The `ktstr-probe-all-failed:` tag
            // mirrors the `ktstr-probe-fatal:` convention so test
            // consumers grepping stderr can distinguish "every tid
            // produced an Err" from "pre-probe error" without
            // inspecting the stdout JSON. The trailing marker keys
            // off `cli.snapshots` (the REQUESTED snapshot count),
            // not `out.snapshots.len()` (the observed count): an
            // interrupted multi-snapshot run with one partial
            // snapshot would otherwise be misclassified as
            // `single` purely because it was cancelled early.
            let is_multi = cli.snapshots > 1;
            let marker = if is_multi { "multi" } else { "single" };
            if let Err(e) = print_output(&cli, &out) {
                eprintln!("error writing output: {e:#}");
            }
            // Record the all-failed outcome in the sidecar BEFORE the
            // final exit so downstream stats tooling sees the probe's
            // per-tid error records (via the flattened `metrics`
            // leaves) even when every tid was an Err arm. The probe
            // exits 1 on this branch, so the appended PayloadMetrics
            // carries `exit_code: 1` — consumers keying on
            // `Check::ExitCodeEq(0)`-equivalents see the failure.
            if let Some(path) = cli.sidecar.as_deref()
                && let Err(e) = append_probe_output_to_sidecar(path, &out, 1)
            {
                eprintln!("error appending to sidecar {}: {e:#}", path.display());
            }
            eprintln!("ktstr-probe-all-failed: {marker}");
            eprintln!(
                "error: all threads failed probe{}",
                if is_multi { " in every snapshot" } else { "" },
            );
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
    } else if msg.contains("/proc/<pid>/exe changed") {
        // Both re-stat gates (mid-parse and between-snapshot) share
        // the "/proc/<pid>/exe changed" wording — see run() for the
        // two emission sites. Consumers keying on `exe-identity-changed`
        // see mid-parse and between-snapshot events as a single
        // category; the human line carries the specific phase.
        "exe-identity-changed"
    } else if msg.contains("jemalloc") && msg.contains("not") {
        "not-jemalloc"
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
        let addr = compute_tls_address_variant_ii(fs_base, aligned, st_value, field).unwrap();
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
        let addr =
            compute_tls_address_variant_ii(/*fs_base*/ 4096, /*aligned*/ 4096, 0, 0).unwrap();
        assert_eq!(addr, 0);
    }

    /// fs_base below the TLS image size is a malformed-state error —
    /// the math must NOT wrap into the top of the u64 address space.
    #[test]
    fn compute_tls_address_underflow_errors() {
        let err = compute_tls_address_variant_ii(4096, 8192, 0, 0).unwrap_err();
        assert!(
            format!("{err}").contains("below the aligned TLS image size"),
            "got: {err}",
        );
    }

    /// Variant I (aarch64) worked example pinning the hand-checked
    /// arithmetic: `TP + round_up(TCB_SIZE=16, p_align) + st_value +
    /// field_offset`.
    ///
    /// With `p_align = 16`, `round_up(16, 16) = 16`, so the image
    /// base sits at `TP + 16`. Adding `st_value = 0x100` and
    /// `field = 264` gives `TP + 0x10 + 0x100 + 264`.
    #[test]
    fn compute_tls_address_variant_i_example() {
        let tpidr = 0x7f12_3456_7000;
        let p_align = 16;
        let st_value = 0x100;
        let field = 264;
        let addr = compute_tls_address_variant_i(tpidr, p_align, st_value, field).unwrap();
        // 0x7f1234567000 + 0x10 + 0x100 + 264
        // = 0x7f1234567110 + 264
        // = 0x7f1234567218
        assert_eq!(addr, 0x7f12_3456_7218);
    }

    /// Variant I with `p_align > TCB_SIZE_AARCH64`: the TLS image
    /// base is rounded up to `p_align`, not pinned at 16. Pins the
    /// `round_up(16, p_align)` calculation for a common high-align
    /// case (`p_align = 64`, which jemalloc's tsd_s uses to hit
    /// cache-line alignment).
    #[test]
    fn compute_tls_address_variant_i_high_alignment() {
        // TP + round_up(16, 64) + 0 + 0 = TP + 64
        let addr = compute_tls_address_variant_i(0x1000, 64, 0, 0).unwrap();
        assert_eq!(addr, 0x1040);
    }

    /// Variant I `p_align == TCB_SIZE_AARCH64`: exact fit, no
    /// padding past the reserved TCB words.
    #[test]
    fn compute_tls_address_variant_i_tcb_sized_alignment() {
        let addr = compute_tls_address_variant_i(0x1000, TCB_SIZE_AARCH64, 0, 0).unwrap();
        assert_eq!(addr, 0x1010);
    }

    /// Variant I with `p_align < TCB_SIZE_AARCH64`: `round_up(16, 8)
    /// = 16`. The reserved TCB size is the minimum — sub-TCB
    /// alignments do NOT shrink the image-base offset.
    #[test]
    fn compute_tls_address_variant_i_sub_tcb_alignment() {
        let addr = compute_tls_address_variant_i(0x1000, 8, 0, 0).unwrap();
        assert_eq!(addr, 0x1010);
    }

    /// Variant I degenerate-align fallback: `p_align = 0` in a
    /// malformed ELF must not divide-by-zero. The implementation's
    /// `.max(1)` coerces to `align = 1`, giving
    /// `round_up(16, 1) = 16`.
    #[test]
    fn compute_tls_address_variant_i_zero_align_clamped() {
        let addr = compute_tls_address_variant_i(0x1000, 0, 0, 0).unwrap();
        assert_eq!(addr, 0x1010);
    }

    /// Variant I overflow: `TP + image_offset + st_value +
    /// field_offset` near `u64::MAX` must error rather than wrap
    /// into the low address space.
    #[test]
    fn compute_tls_address_variant_i_overflow_errors() {
        let err = compute_tls_address_variant_i(u64::MAX - 10, 16, 0x100, 0).unwrap_err();
        assert!(
            format!("{err}").contains("TLS address arithmetic overflow"),
            "got: {err}",
        );
    }

    /// Variant I image-offset overflow: a malformed ELF with
    /// `p_align` near `u64::MAX` would make `round_up(TCB_SIZE,
    /// p_align)` overflow the `checked_add` in
    /// `compute_tls_address_variant_i` BEFORE the TP addition runs.
    /// The error must be the image-offset variant, not the address-
    /// arithmetic variant — distinguishing the two helps the
    /// operator know which input is malformed.
    #[test]
    fn compute_tls_address_variant_i_image_offset_overflow_errors() {
        // `p_align = u64::MAX` is non-power-of-two, but the overflow
        // guard fires regardless (release builds don't hit the
        // debug_assert). `TCB_SIZE_AARCH64 + (u64::MAX - 1)`
        // overflows u64, so `checked_add` returns None and the
        // image-offset bail fires.
        let err = compute_tls_address_variant_i(0x1000, u64::MAX, 0, 0).unwrap_err();
        assert!(
            format!("{err}").contains("TLS image offset overflow"),
            "expected image-offset overflow, got: {err}",
        );
    }

    /// Arch dispatcher routes to the right Variant based on
    /// `target_arch`. On x86_64 build the result matches Variant II;
    /// on aarch64 build it matches Variant I. The test picks inputs
    /// that produce distinct answers under each formula so a
    /// cfg-dispatch regression would produce the wrong output.
    #[test]
    fn compute_tls_address_dispatches_by_target_arch() {
        // TP=4096, aligned=4096, p_align=16, st_value=0, field=0.
        // Variant II: 4096 - 4096 + 0 + 0 = 0
        // Variant I:  4096 + round_up(16, 16) + 0 + 0 = 4096 + 16 = 4112
        let got = compute_tls_address(4096, 4096, 16, 0, 0).unwrap();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(got, 0, "x86_64 must dispatch to Variant II");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(got, 4112, "aarch64 must dispatch to Variant I");
    }

    /// Positionally-distinct dispatcher test with non-zero primes
    /// for every argument. A regression that swapped argument
    /// positions (e.g. passed `p_align` where Variant II expects
    /// `aligned_size`, or vice versa) would produce a wrong answer
    /// for ONE variant but the test that uses zeros for most args
    /// cannot detect that class of drift. Each input is a distinct
    /// prime so a position swap shifts the result by an identifiable
    /// amount.
    ///
    /// Inputs: TP=13_000_009 (prime-ish), aligned=1009 (prime),
    /// p_align=64 (power of 2, used only by Variant I),
    /// st_value=307 (prime), field=83 (prime).
    ///
    /// Variant II: 13_000_009 - 1009 + 307 + 83 = 12_999_390
    /// Variant I:  13_000_009 + round_up(16, 64) + 307 + 83
    ///           = 13_000_009 + 64 + 307 + 83
    ///           = 13_000_463
    #[test]
    fn compute_tls_address_dispatches_positionally_distinct() {
        let got = compute_tls_address(13_000_009, 1009, 64, 307, 83).unwrap();
        #[cfg(target_arch = "x86_64")]
        assert_eq!(got, 12_999_390, "x86_64 Variant II formula");
        #[cfg(target_arch = "aarch64")]
        assert_eq!(got, 13_000_463, "aarch64 Variant I formula");
    }

    /// `extract_pt_tls_layout` on the test binary's own ELF (the
    /// bin's `#[cfg(test)]` executable). The probe binary links
    /// `tikv_jemallocator` as the global allocator (see the
    /// `#[global_allocator]` declaration at the top of the file), so
    /// the compiled test binary carries jemalloc's `tsd_tls` in a
    /// real `PT_TLS` segment. Parsing it exercises the ACTUAL
    /// extraction function end-to-end and proves that the tuple
    /// invariants (`p_align` power-of-two, `aligned_size >=
    /// p_align`, `aligned_size % p_align == 0`) hold against a real
    /// toolchain-emitted program header, not a local mirror of the
    /// round-up math.
    #[test]
    fn extract_pt_tls_layout_on_real_elf() {
        let exe = std::env::current_exe().expect("current_exe");
        let data = std::fs::read(&exe).expect("read current_exe");
        let elf = goblin::elf::Elf::parse(&data).expect("parse current_exe");
        let (rounded, align) =
            extract_pt_tls_layout(&elf).expect("probe test binary must carry PT_TLS");
        assert!(
            align.is_power_of_two(),
            "p_align {align} must be a power of two",
        );
        assert!(rounded >= align, "aligned_size {rounded} must be >= align {align}");
        assert!(
            rounded % align == 0,
            "aligned_size {rounded} must be a multiple of align {align}",
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

    /// JSON schema v2: success + error arms round-trip via serde,
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
            started_at_unix_sec: 1_700_000_000,
            interval_ms: None,
            interrupted: false,
            snapshots: vec![Snapshot {
                timestamp_unix_sec: 1_700_000_000,
                threads: vec![ok, ok_no_comm, err],
            }],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(s.contains("\"schema_version\":2"));
        assert!(s.contains("\"pid\":100"));
        assert!(s.contains("\"tool_version\":\"0.0.0\""));
        assert!(s.contains("\"started_at_unix_sec\":1700000000"));
        assert!(s.contains("\"timestamp_unix_sec\":1700000000"));
        assert!(s.contains("\"interrupted\":false"));
        assert!(s.contains("\"snapshots\":["));
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
        // `interval_ms: None` on single-snapshot output must be omitted
        // (skip_serializing_if) so the wire shape discriminates single
        // from multi via field presence, not null sentinel.
        assert!(!s.contains("\"interval_ms\":null"));
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
        let pid = self_pid();
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
            pid: self_pid(),
            json: false,
            snapshots: 1,
            interval_ms: None,
            sidecar: None,
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
        let self_pid = self_pid();
        assert_ne!(
            child_pid, self_pid,
            "spawned child pid must differ from parent for this test to be meaningful",
        );
        let cli = Cli {
            pid: child_pid,
            json: false,
            snapshots: 1,
            interval_ms: None,
            sidecar: None,
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
        // Match the arch-correct regset name — NT_PRSTATUS on x86_64
        // (where fs_base lives in user_regs_struct), NT_ARM_TLS on
        // aarch64 (where tpidr_el0 is reached via regset 0x401).
        assert!(
            msg.contains(arch::REGSET_NAME),
            "expected regset name {}, got: {msg}",
            arch::REGSET_NAME,
        );
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

    // ---- sampling-mode CLI parsing + validation ----
    //
    // `clap::Parser` is already in scope via `use super::*` (the top
    // of `jemalloc_probe.rs` imports it for the `Cli` derive), so
    // `Cli::try_parse_from` resolves without a redundant re-import.

    /// Default invocation (no `--snapshots` / `--interval-ms`): clap
    /// fills `snapshots = 1` and `interval_ms = None`, and
    /// `validate_sampling_flags` accepts the combination.
    #[test]
    fn cli_default_sampling_count_is_one() {
        let cli = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "42"]).unwrap();
        assert_eq!(cli.snapshots, 1);
        assert!(cli.interval_ms.is_none());
        assert!(cli.validate_sampling_flags().is_ok());
    }

    /// Explicit `--snapshots 1` without `--interval-ms` is the same
    /// as the default; validation passes.
    #[test]
    fn cli_explicit_count_one_without_interval_accepted() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "1",
        ])
        .unwrap();
        assert_eq!(cli.snapshots, 1);
        assert!(cli.interval_ms.is_none());
        assert!(cli.validate_sampling_flags().is_ok());
    }

    /// Multi-snapshot invocation with `--snapshots > 1` and a positive
    /// `--interval-ms`: both flags parse, validation passes.
    #[test]
    fn cli_multi_snapshot_accepts_count_and_interval() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "3",
            "--interval-ms",
            "50",
        ])
        .unwrap();
        assert_eq!(cli.snapshots, 3);
        assert_eq!(cli.interval_ms, Some(50));
        assert!(cli.validate_sampling_flags().is_ok());
    }

    /// `--snapshots 0` is rejected at parse time by the
    /// `clap::value_parser!(u32).range(1..=100_000)` attribute —
    /// a zero-count run has no useful output and would only emit
    /// an empty `snapshots` array.
    #[test]
    fn cli_count_zero_rejected() {
        let err = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "0",
        ])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0 is not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    /// `--snapshots 100_001` is rejected at parse time by the upper
    /// bound on the range parser. The cap bounds the pre-allocated
    /// snapshot vector so a runaway `--snapshots` cannot request a
    /// multi-GiB allocation.
    #[test]
    fn cli_snapshots_upper_bound_rejected() {
        let err = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "100001",
        ])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    /// `--interval-ms 0` is rejected at parse time — a zero-ms
    /// interval is semantically back-to-back snapshots with no delay.
    #[test]
    fn cli_interval_zero_rejected() {
        let err = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "2",
            "--interval-ms",
            "0",
        ])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0 is not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    /// `--interval-ms 3_600_001` (>1 hour) is rejected at parse time
    /// by the upper bound on the range parser. The cap bounds the
    /// max single-run duration and guarantees the `Instant + Duration`
    /// deadline math in `sleep_with_cancel` cannot overflow.
    #[test]
    fn cli_interval_upper_bound_rejected() {
        let err = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "2",
            "--interval-ms",
            "3600001",
        ])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    /// `--pid 0` is rejected at parse time: Linux tgids start at 1.
    #[test]
    fn cli_pid_zero_rejected() {
        let err =
            Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "0"]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0 is not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    /// `--pid=-1` is rejected at parse time: negative values are not
    /// valid Linux pids. The `=` form is required because the
    /// standalone `--pid -1` sequence parses `-1` as an unknown flag
    /// rather than the pid value.
    #[test]
    fn cli_pid_negative_rejected() {
        let err = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid=-1"]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    /// `--snapshots > 1` without `--interval-ms` clears clap parsing
    /// but fails `validate_sampling_flags` — multi-snapshot mode must
    /// be explicit about the inter-snapshot wait.
    #[test]
    fn cli_count_greater_than_one_requires_interval() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--snapshots",
            "3",
        ])
        .unwrap();
        let err = cli.validate_sampling_flags().unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("requires --interval-ms"), "got: {msg}");
    }

    /// `--interval-ms` with `--snapshots 1` (default) is a user-intent
    /// mismatch — the interval has nothing to separate. Rejected at
    /// the `validate_sampling_flags` gate.
    #[test]
    fn cli_interval_requires_count_greater_than_one() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--interval-ms",
            "100",
        ])
        .unwrap();
        let err = cli.validate_sampling_flags().unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("only meaningful with --snapshots > 1"),
            "got: {msg}",
        );
    }

    // ---- take_snapshot / sleep_with_cancel helpers ----

    /// `sleep_with_cancel` returns `false` for a normal (uninterrupted)
    /// sleep and consumes roughly the requested duration.
    /// Uses a short wait so the test stays fast; the exact lower bound
    /// guards against a future regression that returns immediately
    /// without sleeping.
    #[test]
    fn sleep_with_cancel_completes_without_flag_set() {
        // Ensure the flag is clear (another test may have set it, but
        // tests run in parallel by default and the atomic is global —
        // safest to reset unconditionally before the observation).
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let cancelled = sleep_with_cancel(25);
        let elapsed = start.elapsed();
        assert!(!cancelled, "sleep should not report cancellation when flag stays clear");
        assert!(
            elapsed >= std::time::Duration::from_millis(20),
            "sleep returned too fast: {elapsed:?}",
        );
    }

    /// `sleep_with_cancel` returns `true` promptly when
    /// `CLEANUP_REQUESTED` is already set on entry.
    #[test]
    fn sleep_with_cancel_observes_pre_set_flag() {
        CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let cancelled = sleep_with_cancel(10_000);
        let elapsed = start.elapsed();
        // Reset for other tests. Multiple tests poke this static so
        // leaving it set would bleed between runs.
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        assert!(cancelled, "pre-set flag must cause immediate cancel");
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "cancel path should return within a poll tick, got: {elapsed:?}",
        );
    }

    /// `all_failed` semantics: empty vec is "all failed" (no live
    /// observations); a vec with only `Err` arms is all-failed; any
    /// `Ok` arm disqualifies.
    #[test]
    fn all_failed_classification() {
        assert!(all_failed(&[]), "empty threads vec is all-failed");
        let only_err = vec![ThreadResult::Err {
            tid: 1,
            comm: None,
            error: "e".into(),
            error_kind: ThreadErrorKind::PtraceSeize,
        }];
        assert!(all_failed(&only_err));
        let mixed = vec![
            ThreadResult::Err {
                tid: 1,
                comm: None,
                error: "e".into(),
                error_kind: ThreadErrorKind::PtraceSeize,
            },
            ThreadResult::Ok {
                tid: 2,
                comm: None,
                allocated_bytes: 10,
                deallocated_bytes: 0,
            },
        ];
        assert!(!all_failed(&mixed));
    }

    /// Multi-snapshot JSON shape: `snapshots` array with per-
    /// snapshot `timestamp_unix_sec` + `threads`; top-level
    /// `pid` / `tool_version` / `schema_version` / `started_at_unix_sec`
    /// / `interval_ms` / `interrupted` carry the run-invariant
    /// metadata. Pins the wire contract consumed by the integration
    /// test's multi-snapshot assertions.
    #[test]
    fn multi_snapshot_output_json_shape() {
        let out = ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: 777,
            tool_version: "0.0.0",
            started_at_unix_sec: 1_699_999_999,
            interval_ms: Some(50),
            interrupted: false,
            snapshots: vec![
                Snapshot {
                    timestamp_unix_sec: 1_700_000_000,
                    threads: vec![ThreadResult::Ok {
                        tid: 42,
                        comm: Some("worker".to_string()),
                        allocated_bytes: 1024,
                        deallocated_bytes: 0,
                    }],
                },
                Snapshot {
                    timestamp_unix_sec: 1_700_000_001,
                    threads: vec![ThreadResult::Ok {
                        tid: 42,
                        comm: Some("worker".to_string()),
                        allocated_bytes: 2048,
                        deallocated_bytes: 0,
                    }],
                },
            ],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(s.contains("\"schema_version\":2"));
        assert!(s.contains("\"pid\":777"));
        assert!(s.contains("\"started_at_unix_sec\":1699999999"));
        assert!(s.contains("\"interval_ms\":50"));
        assert!(s.contains("\"interrupted\":false"));
        assert!(s.contains("\"snapshots\":["));
        assert!(s.contains("\"timestamp_unix_sec\":1700000000"));
        assert!(s.contains("\"timestamp_unix_sec\":1700000001"));
        assert!(s.contains("\"allocated_bytes\":1024"));
        assert!(s.contains("\"allocated_bytes\":2048"));
        // Per-snapshot timestamps move into each snapshot entry; the
        // top-level carries only `started_at_unix_sec`.
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(
            v.get("timestamp_unix_sec").is_none(),
            "top-level timestamp_unix_sec must not appear on ProbeOutput: {s}",
        );
        assert!(
            v.get("threads").is_none(),
            "top-level threads must not appear on ProbeOutput: {s}",
        );
        assert!(v.get("snapshots").is_some(), "snapshots array required: {s}");
        assert!(v.get("started_at_unix_sec").is_some());
        assert!(v.get("interval_ms").is_some());
        assert!(v.get("interrupted").is_some());
    }

    /// Single-snapshot `ProbeOutput` must emit `snapshots` with one
    /// element and omit `interval_ms` via `skip_serializing_if`.
    /// Consumers distinguish single- vs multi-snapshot by
    /// `interval_ms` presence.
    #[test]
    fn single_snapshot_output_omits_interval_ms() {
        let out = ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: 555,
            tool_version: "0.0.0",
            started_at_unix_sec: 1_700_000_000,
            interval_ms: None,
            interrupted: false,
            snapshots: vec![Snapshot {
                timestamp_unix_sec: 1_700_000_000,
                threads: vec![ThreadResult::Ok {
                    tid: 99,
                    comm: None,
                    allocated_bytes: 10,
                    deallocated_bytes: 0,
                }],
            }],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(!s.contains("\"interval_ms\""), "interval_ms must be omitted when None: {s}");
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(v.get("interval_ms").is_none());
        let snaps = v.get("snapshots").and_then(|v| v.as_array()).unwrap();
        assert_eq!(snaps.len(), 1, "single-snapshot must emit snapshots of length 1");
    }

    /// `ExeIdentity::capture` on the probe's own pid round-trips
    /// equal to itself across two back-to-back calls — the probe
    /// binary's on-disk identity is stable within a single run, so
    /// the re-stat gate in `run()` does not false-positive on a
    /// normal invocation.
    #[test]
    fn exe_identity_stable_within_run() {
        let pid = self_pid();
        let a = ExeIdentity::capture(pid).expect("stat /proc/self/exe");
        let b = ExeIdentity::capture(pid).expect("stat /proc/self/exe");
        assert_eq!(a, b, "ExeIdentity must be stable across back-to-back captures");
    }

    /// `interrupted: true` round-trips through serde. Pins the JSON
    /// literal so downstream consumers keying on `"interrupted":true`
    /// to distinguish partial from complete runs see a stable token.
    /// Pairs with the `false` case already exercised by
    /// `thread_result_json_shape` and `multi_snapshot_output_json_shape`.
    #[test]
    fn interrupted_true_serializes_as_json_true() {
        let out = ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: 321,
            tool_version: "0.0.0",
            started_at_unix_sec: 1_700_000_000,
            interval_ms: Some(100),
            interrupted: true,
            snapshots: vec![Snapshot {
                timestamp_unix_sec: 1_700_000_000,
                threads: Vec::new(),
            }],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(
            s.contains("\"interrupted\":true"),
            "expected `\"interrupted\":true` literal, got: {s}",
        );
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v.get("interrupted").and_then(|b| b.as_bool()), Some(true));
    }

    // ---- --sidecar integration ----

    /// Build a minimal [`SidecarResult`] JSON string for sidecar-path
    /// tests. Populates every field required on deserialize — any
    /// schema change that adds a field surfaces as a compile error
    /// at this call site, prompting the test fixture to stay in sync.
    fn minimal_sidecar_json() -> String {
        let sc = ktstr::test_support::SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            payload: None,
            metrics: Vec::new(),
            passed: true,
            skipped: false,
            stats: ktstr::assert::ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
            host: None,
        };
        serde_json::to_string_pretty(&sc).unwrap()
    }

    /// Build a `ProbeOutput` fixture with one Ok thread so
    /// `walk_json_leaves` produces a deterministic set of numeric
    /// leaves. Used across the `append_probe_output_to_sidecar`
    /// tests.
    fn probe_output_fixture() -> ProbeOutput {
        ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: 42,
            tool_version: "0.0.0",
            started_at_unix_sec: 1_700_000_000,
            interval_ms: None,
            interrupted: false,
            snapshots: vec![Snapshot {
                timestamp_unix_sec: 1_700_000_000,
                threads: vec![ThreadResult::Ok {
                    tid: 42,
                    comm: Some("worker".to_string()),
                    allocated_bytes: 1024,
                    deallocated_bytes: 512,
                }],
            }],
        }
    }

    /// Happy path: append a synthesized `PayloadMetrics` to a
    /// pre-existing sidecar JSON file. Verifies (1) the file parses
    /// back to a valid `SidecarResult`, (2) the appended
    /// `PayloadMetrics` is the last entry, (3) its `metrics` contain
    /// the `jemalloc_probe.*`-prefixed leaves walked out of the
    /// probe's output, and (4) the `allocated_bytes` leaf got the
    /// `LowerBetter` polarity + `bytes` unit hint from
    /// `apply_probe_metric_hints`.
    #[test]
    fn sidecar_append_happy_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t-0000000000000000.ktstr.json");
        std::fs::write(&path, minimal_sidecar_json()).unwrap();

        let out = probe_output_fixture();
        append_probe_output_to_sidecar(&path, &out, 0)
            .expect("append happy path");

        let re_read = std::fs::read_to_string(&path).unwrap();
        let sc: ktstr::test_support::SidecarResult =
            serde_json::from_str(&re_read).expect("sidecar re-parse");
        // Pre-existing top-level fields must survive the append
        // unchanged — the probe only touches `metrics`. A regression
        // that rewrote scheduler/topology/etc. would show up here.
        assert_eq!(sc.test_name, "t");
        assert_eq!(sc.topology, "1n1l1c1t");
        assert_eq!(sc.scheduler, "eevdf");
        assert!(sc.passed);
        assert!(!sc.skipped);
        assert_eq!(sc.metrics.len(), 1, "one appended PayloadMetrics");
        let pm = &sc.metrics[0];
        assert_eq!(pm.exit_code, 0);
        // Every metric name carries the probe prefix so downstream
        // aggregators can discriminate probe-sourced leaves.
        for m in &pm.metrics {
            assert!(
                m.name.starts_with(&format!("{SIDECAR_METRIC_PREFIX}.")),
                "metric name {} missing probe prefix",
                m.name,
            );
        }
        let alloc = pm
            .metrics
            .iter()
            .find(|m| m.name.ends_with(".allocated_bytes"))
            .expect("allocated_bytes metric in appended entry");
        assert_eq!(alloc.value, 1024.0);
        assert_eq!(alloc.unit, "bytes");
        assert!(matches!(
            alloc.polarity,
            ktstr::test_support::Polarity::LowerBetter,
        ));
        // Identity leaves (tid, schema_version) retain Unknown
        // polarity — the hints only fire for the named byte counters.
        let tid = pm
            .metrics
            .iter()
            .find(|m| m.name.ends_with(".tid"))
            .expect("tid metric in appended entry");
        assert!(matches!(
            tid.polarity,
            ktstr::test_support::Polarity::Unknown,
        ));
        assert_eq!(tid.unit, "");

        // Lock-file convention: `<sidecar>.<ext>.lock` sits alongside
        // the sidecar. Pins the naming contract so a future refactor
        // that relocates or renames the lock surfaces as a test
        // failure. `.ktstr.json` path extension is `json`, so the
        // lock file is `<...>.json.lock`.
        let lock_path = path.with_extension("json.lock");
        assert!(
            lock_path.exists(),
            "expected lock file at {}",
            lock_path.display(),
        );

        // No orphan staging files must remain after a successful
        // append — `append_probe_output_to_sidecar` renames its
        // `*.tmp` over the sidecar on success, so none should be
        // visible in the parent dir.
        let orphans: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|x| x.to_str()) == Some("tmp")
                    || p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.contains(".tmp"))
            })
            .collect();
        assert!(
            orphans.is_empty(),
            "expected no staging tmp files after successful append, got: {orphans:?}",
        );
    }

    /// Two back-to-back appends stack — the second `PayloadMetrics`
    /// lands after the first. Proves the helper is repeatable and
    /// does not clobber earlier appends (regression guard against a
    /// `vec![new]` overwrite).
    #[test]
    fn sidecar_append_stacks_across_invocations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.ktstr.json");
        std::fs::write(&path, minimal_sidecar_json()).unwrap();

        let out = probe_output_fixture();
        append_probe_output_to_sidecar(&path, &out, 0).unwrap();
        append_probe_output_to_sidecar(&path, &out, 1).unwrap();

        let sc: ktstr::test_support::SidecarResult =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(sc.metrics.len(), 2, "both appends retained");
        assert_eq!(sc.metrics[0].exit_code, 0);
        assert_eq!(sc.metrics[1].exit_code, 1);
        // Both appends must carry the probe prefix on every metric —
        // a regression that prefixed only the first invocation's
        // metrics (e.g. a stale `SIDECAR_METRIC_PREFIX` constant
        // captured once at module init) would be caught here.
        for (i, pm) in sc.metrics.iter().enumerate() {
            for m in &pm.metrics {
                assert!(
                    m.name.starts_with(&format!("{SIDECAR_METRIC_PREFIX}.")),
                    "append {i} metric {} missing probe prefix",
                    m.name,
                );
            }
        }
    }

    /// Starting from a sidecar that already has pre-existing
    /// `metrics` entries (e.g. the test harness recorded its primary
    /// payload invocation), the probe's append must preserve those
    /// entries in order and land its own entry at the end. Guards
    /// against a `sidecar.metrics = vec![new]` regression.
    #[test]
    fn sidecar_append_preserves_prepopulated_metrics() {
        use ktstr::test_support::{Metric, MetricSource, PayloadMetrics, Polarity};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.ktstr.json");

        // Build a sidecar that already carries two PayloadMetrics
        // entries (e.g. the test harness recorded both a primary
        // payload and a secondary workload).
        let mut sc: ktstr::test_support::SidecarResult =
            serde_json::from_str(&minimal_sidecar_json()).unwrap();
        sc.metrics.push(PayloadMetrics {
            metrics: vec![Metric {
                name: "primary.bogo_ops".to_string(),
                value: 12345.0,
                polarity: Polarity::HigherBetter,
                unit: "ops".to_string(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        });
        sc.metrics.push(PayloadMetrics {
            metrics: vec![Metric {
                name: "secondary.latency_us".to_string(),
                value: 42.0,
                polarity: Polarity::LowerBetter,
                unit: "us".to_string(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&sc).unwrap()).unwrap();

        let out = probe_output_fixture();
        append_probe_output_to_sidecar(&path, &out, 0).unwrap();

        let after: ktstr::test_support::SidecarResult =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(after.metrics.len(), 3, "expected 2 pre-existing + 1 appended");
        // Pre-existing entries unchanged in order and content.
        assert_eq!(after.metrics[0].metrics[0].name, "primary.bogo_ops");
        assert_eq!(after.metrics[0].metrics[0].value, 12345.0);
        assert_eq!(after.metrics[1].metrics[0].name, "secondary.latency_us");
        assert_eq!(after.metrics[1].metrics[0].value, 42.0);
        // Probe's append is the LAST entry.
        for m in &after.metrics[2].metrics {
            assert!(
                m.name.starts_with(&format!("{SIDECAR_METRIC_PREFIX}.")),
                "last entry's metric {} missing probe prefix",
                m.name,
            );
        }
    }

    /// Missing file is a hard error with the operator-actionable
    /// "run the test first" wording. Pins the phrasing so a consumer
    /// grepping stderr for `sidecar file not found` keeps working.
    #[test]
    fn sidecar_append_missing_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist.ktstr.json");
        let out = probe_output_fixture();
        let err = append_probe_output_to_sidecar(&missing, &out, 0).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sidecar file not found"),
            "expected missing-file wording, got: {msg}",
        );
        assert!(
            msg.contains("run the test first"),
            "expected operator-actionable hint, got: {msg}",
        );
        // The flag name (`--sidecar`) MUST appear in the hint so
        // operators who `man`-read the error know which invocation
        // produced it — `jemalloc-probe`'s CLI surface has grown
        // several file-path flags and the fix-it hint has to be
        // specific.
        assert!(
            msg.contains("--sidecar"),
            "expected flag name in hint, got: {msg}",
        );
    }

    /// Malformed JSON in the sidecar file is a hard error with a
    /// parse-failure hint pointing at the pre-1.0 regenerate policy.
    /// Covers the "sidecar from an incompatible schema version"
    /// path in `append_probe_output_to_sidecar`.
    #[test]
    fn sidecar_append_malformed_json_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.ktstr.json");
        std::fs::write(&path, "{ this is not valid json }").unwrap();
        let out = probe_output_fixture();
        let err = append_probe_output_to_sidecar(&path, &out, 0).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("parse"),
            "expected parse-failure context, got: {msg}",
        );
        // Pre-1.0 policy hint: the operator's remediation is to
        // regenerate the sidecar, not to patch the JSON by hand.
        // Pinning the substring keeps the hint on-message.
        assert!(
            msg.contains("regenerate"),
            "expected pre-1.0 regenerate-policy hint, got: {msg}",
        );
    }

    /// Probe-specific polarity / unit hints: byte-counter metrics get
    /// `LowerBetter` + `bytes`; everything else keeps the walker's
    /// `Unknown` + empty-unit defaults. Pins the hint surface so a
    /// future rename of `allocated_bytes` in the probe schema forces
    /// a matching update here.
    #[test]
    fn apply_probe_metric_hints_classifies_byte_counters() {
        use ktstr::test_support::{Metric, MetricSource, Polarity};
        let mut alloc = Metric {
            name: "jemalloc_probe.snapshots.0.threads.0.allocated_bytes".to_string(),
            value: 1024.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        };
        apply_probe_metric_hints(&mut alloc);
        assert!(matches!(alloc.polarity, Polarity::LowerBetter));
        assert_eq!(alloc.unit, "bytes");

        let mut dealloc = Metric {
            name: "jemalloc_probe.snapshots.0.threads.0.deallocated_bytes".to_string(),
            value: 512.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        };
        apply_probe_metric_hints(&mut dealloc);
        assert!(matches!(dealloc.polarity, Polarity::LowerBetter));
        assert_eq!(dealloc.unit, "bytes");

        let mut tid = Metric {
            name: "jemalloc_probe.snapshots.0.threads.0.tid".to_string(),
            value: 42.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        };
        apply_probe_metric_hints(&mut tid);
        assert!(matches!(tid.polarity, Polarity::Unknown));
        assert_eq!(tid.unit, "");

        // Negative match: the hint uses `ends_with(".allocated_bytes")`,
        // not `contains`. A metric whose name ends with
        // `allocated_bytes_extra` (or any suffix beyond the exact
        // counter name) must NOT pick up the LowerBetter/bytes hint —
        // substring matching would misclassify arbitrary future
        // metrics. Pins the ends-with contract.
        let mut extra = Metric {
            name: "jemalloc_probe.snapshots.0.threads.0.allocated_bytes_extra".to_string(),
            value: 999.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        };
        apply_probe_metric_hints(&mut extra);
        assert!(
            matches!(extra.polarity, Polarity::Unknown),
            "name ending in _extra must not match the byte-counter hint",
        );
        assert_eq!(extra.unit, "");
        // Same check for deallocated.
        let mut dextra = Metric {
            name: "jemalloc_probe.deallocated_bytes_something".to_string(),
            value: 0.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        };
        apply_probe_metric_hints(&mut dextra);
        assert!(matches!(dextra.polarity, Polarity::Unknown));
        assert_eq!(dextra.unit, "");
    }

    /// Direct [`synthesize_payload_metrics`] test that bypasses the
    /// sidecar file. Constructs a `ProbeOutput` carrying both Ok and
    /// Err per-thread results and asserts (1) every emitted `Metric`
    /// carries the `jemalloc_probe.` prefix, (2) only numeric leaves
    /// surface (Err's `error` string is dropped by `walk_json_leaves`,
    /// `error_kind` is a string-enum so also dropped), (3) the
    /// `exit_code` parameter flows through to the `PayloadMetrics`,
    /// and (4) numeric leaves from both Ok and Err arms (tid from
    /// both) are present.
    #[test]
    fn synthesize_payload_metrics_handles_ok_and_err_threads() {
        let out = ProbeOutput {
            schema_version: SCHEMA_VERSION,
            pid: 1234,
            tool_version: "0.0.0",
            started_at_unix_sec: 1_700_000_000,
            interval_ms: None,
            interrupted: false,
            snapshots: vec![Snapshot {
                timestamp_unix_sec: 1_700_000_000,
                threads: vec![
                    ThreadResult::Ok {
                        tid: 42,
                        comm: Some("ok-worker".to_string()),
                        allocated_bytes: 2048,
                        deallocated_bytes: 128,
                    },
                    ThreadResult::Err {
                        tid: 99,
                        comm: Some("err-worker".to_string()),
                        error: "ptrace(PTRACE_SEIZE): ESRCH".to_string(),
                        error_kind: ThreadErrorKind::PtraceSeize,
                    },
                ],
            }],
        };
        let pm = synthesize_payload_metrics(&out, 7).expect("synthesize");
        assert_eq!(pm.exit_code, 7, "exit_code flows through");

        // All prefixed.
        for m in &pm.metrics {
            assert!(
                m.name.starts_with(&format!("{SIDECAR_METRIC_PREFIX}.")),
                "metric {} missing probe prefix",
                m.name,
            );
        }
        // No string leaves surface — the walker drops non-numeric
        // leaves. `error` and `error_kind` are strings; their
        // metric-ified names must not appear.
        for m in &pm.metrics {
            assert!(
                !m.name.ends_with(".error"),
                "string `error` leaf must not surface, got: {}",
                m.name,
            );
            assert!(
                !m.name.ends_with(".error_kind"),
                "string `error_kind` leaf must not surface, got: {}",
                m.name,
            );
        }
        // Numeric leaves from BOTH Ok and Err arms surface: tid
        // value 42 (Ok) and tid value 99 (Err).
        let tid_values: Vec<f64> = pm
            .metrics
            .iter()
            .filter(|m| m.name.ends_with(".tid"))
            .map(|m| m.value)
            .collect();
        assert!(
            tid_values.contains(&42.0),
            "Ok thread's tid=42 must surface, got: {tid_values:?}",
        );
        assert!(
            tid_values.contains(&99.0),
            "Err thread's tid=99 must surface, got: {tid_values:?}",
        );
        // Only the Ok thread has byte counters — the Err variant
        // has no `allocated_bytes` / `deallocated_bytes` fields, so
        // exactly one of each should appear.
        let alloc_count = pm
            .metrics
            .iter()
            .filter(|m| m.name.ends_with(".allocated_bytes"))
            .count();
        assert_eq!(alloc_count, 1, "one Ok thread emits one allocated_bytes");
    }
}
