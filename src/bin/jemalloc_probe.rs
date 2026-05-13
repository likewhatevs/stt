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
//! the target's `/proc/<pid>/exe` when that ELF carries
//! `.debug_info`; if the target is stripped, the engine walks
//! `.gnu_debuglink` + `NT_GNU_BUILD_ID` to locate external
//! debuginfo (paired `.debug` file or a distro `-dbg` /
//! `-debuginfo` package under `/usr/lib/debug`). The engine itself
//! lives in `src/host_thread_probe.rs` and is source-shared into
//! this binary via `#[path]` (see the `host_thread_probe` mod
//! below for the ctor-avoidance rationale). End-to-end validation
//! runs via the `#[ktstr_test]` integration tests in
//! `tests/jemalloc_probe_tests.rs`, which boot a VM, spawn a
//! jemalloc-linked allocator worker, and run the probe against
//! the worker's live pid.
//!
//! Scope:
//! - Linux, x86_64 and aarch64. Same-arch only (a probe binary built
//!   for x86_64 only handles x86_64 targets; ptrace is same-arch).
//! - Static-linked jemalloc only (symbol lives in the main
//!   executable's static TLS image).
//! - Requires DWARF debuginfo reachable from the target ELF —
//!   either inline `.debug_info` on the target itself OR a paired
//!   external debug file discovered via `.gnu_debuglink` /
//!   `NT_GNU_BUILD_ID` — plus CAP_SYS_PTRACE / root / same-uid or
//!   descendant relationship under YAMA (see "Security posture"
//!   below).
//!
//! # Security posture
//!
//! The probe is distributed as a normal user binary — **no setuid,
//! no setgid, no file capabilities, no suid-helper**. It carries no
//! privileged bits on disk and does not request any at runtime. The
//! only privilege needed is whatever `ptrace(PTRACE_SEIZE)` demands
//! against the target process; everything else (DWARF read via
//! `/proc/<pid>/exe`, `process_vm_readv`) rides on the same access
//! check.
//!
//! The exact privilege story depends on the host's
//! `kernel.yama.ptrace_scope` setting (see
//! `Documentation/admin-guide/LSM/Yama.rst` in the kernel tree):
//!
//! - **`ptrace_scope=0` (no restriction)**: any process sharing the
//!   target's uid can attach. This is the layout the ktstr
//!   integration tests run under — the guest VM boots with the
//!   default kernel sysctls, and both the alloc-worker and the
//!   probe live under the same uid inside that VM. No extra
//!   capability is needed.
//! - **`ptrace_scope=1` (restricted; Debian/Ubuntu default on
//!   bare-metal hosts)**: same-uid alone is NOT sufficient. The
//!   tracer must additionally be an ancestor of the target, OR the
//!   target must have called `prctl(PR_SET_PTRACER, tracer_pid)` /
//!   `PR_SET_PTRACER_ANY` to opt the tracer in, OR the tracer must
//!   carry `CAP_SYS_PTRACE`. For a probe binary running outside
//!   the target's process tree, the practical options on a
//!   scope=1 host are: `sudo setcap cap_sys_ptrace+eip
//!   ktstr-jemalloc-probe` on the release binary (one-time), or
//!   invoke via `sudo -E` so the probe inherits uid 0.
//! - **`ptrace_scope=2` (admin-only)**: only `CAP_SYS_PTRACE` or
//!   uid 0 attaches; user-level opt-in via `PR_SET_PTRACER` is
//!   refused.
//! - **`ptrace_scope=3` (disabled)**: no process may attach to any
//!   other, regardless of capability. The probe cannot function
//!   and `PTRACE_SEIZE` returns `EPERM`.
//!
//! In every scope above, `PTRACE_SEIZE` surfaces a clear `EPERM` /
//! `ESRCH` failure rather than silently degrading — the engine
//! propagates the errno through anyhow context, so operators
//! see the exact syscall that was refused.
//!
//! The probe does not open network sockets, does not write outside
//! its explicit `--sidecar` path (when provided), and does not
//! inspect anything beyond the single target pid named on `--pid`.
//! It cannot escalate to adjacent processes — each invocation names
//! exactly one target and exits when that target is detached.

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

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use nix::sys::ptrace;
use nix::sys::signal::{SaFlags, SigAction, SigHandler, SigSet, Signal, sigaction, signal};
use nix::unistd::Pid;
use serde::Serialize;

// Source-share the engine via `#[path]` rather than `use ktstr::host_thread_probe::*`.
// Linking the ktstr library into this binary would pull in the
// library's early-dispatch ctor (`test_support::dispatch::ktstr_test_early_dispatch`,
// tagged `#[ctor::ctor]`) plus the rest of the crate, bloating the
// initramfs image and adding `.init_array` work that can stall the
// probe's cross-process timing. `#[path]` compiles the same source
// file into this bin crate directly — zero linker cost, same
// single-source-of-truth behavior. Mirrors the idiom used by
// src/bin/jemalloc_alloc_worker.rs:72 to share `worker_ready.rs`.
//
// `#[allow(dead_code)]` blanket on the module: the source-shared
// engine carries items (e.g. the `probe_thread` no-cache wrapper)
// that the lib-crate compilation uses but this bin crate's
// compilation unit does not see used. Without the allow, every
// such item triggers a dead-code warning here even though the lib
// unit consumes them.
#[allow(dead_code)]
#[path = "../host_thread_probe.rs"]
mod host_thread_probe;
use host_thread_probe::{
    AttachError, JemallocProbe, ProbeError, attach_jemalloc, probe_thread_with_cache,
};

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
///
/// Changelog:
/// - v3: `error_kind` tokens migrated from snake_case to kebab-case
///   to align with the library's `ProbeError::tag()` vocabulary.
const SCHEMA_VERSION: u32 = 3;

/// Capture the current wall-clock as Unix epoch seconds. `unwrap_or(0)`
/// handles the impossible pre-epoch-clock case defensively — KVM
/// guests under kvm-clock or NTP always resolve post-1970, so the
/// zero is a never-fires safety net rather than a real fallback.
fn now_unix_sec() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The probe's own pid as an `i32`. Linux enforces `pid_max <= 2^22`
/// (kernel/pid.c), so the `u32 → i32` conversion is infallible in
/// practice; the `expect` documents that invariant.
fn self_pid() -> i32 {
    libc::pid_t::try_from(std::process::id()).expect("Linux pid_max <= 2^22 so pid fits in pid_t")
}

/// Render the optional per-thread comm string as a trailing
/// `" comm=<name>"` fragment for the human-readable output path, or
/// the empty string when comm is absent. Shared by the Ok and Err
/// arms of [`print_thread_result`] so both lines use identical
/// formatting.
fn format_comm_suffix(comm: Option<&str>) -> String {
    comm.map(|c| format!(" comm={c}")).unwrap_or_default()
}

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
///   tid separated by `interval_ms` of sleep. The setup (ELF/DWARF
///   parse) is amortized across all N snapshots. Threads are NOT
///   held stopped between snapshots — each tid is detached before
///   the inter-snapshot sleep so the target workload continues to
///   run.
#[derive(Parser, Debug)]
#[command(
    name = "ktstr-jemalloc-probe",
    version = env!("CARGO_PKG_VERSION"),
    about = "Read per-thread jemalloc allocated/deallocated byte counters from a running process",
    after_help = "\
EXAMPLES:
  Single snapshot against a running pid:
    ktstr-jemalloc-probe --pid 12345 --json

  Multi-snapshot sampling — 5 snapshots at 200ms each (= 1s total):
    ktstr-jemalloc-probe --pid 12345 --snapshots 5 --interval-ms 200 --json

  Time-bounded run — take up to 10 snapshots at 500ms, self-abort after 3s:
    ktstr-jemalloc-probe --pid 12345 --snapshots 10 --interval-ms 500 \\
                         --abort-after-ms 3000 --json

  Enrich an existing ktstr sidecar with probe metrics:
    ktstr-jemalloc-probe --pid 12345 --sidecar \\
      target/ktstr/{kernel}-{project_commit}/{test}-{hash}.ktstr.json\
",
    long_about = "Reads jemalloc's per-thread `thread_allocated` / `thread_deallocated` TLS \
                  counters out of a running process via ptrace + process_vm_readv. Counters are \
                  cumulative from thread creation — attaching late does not miss prior \
                  allocations. Requires CAP_SYS_PTRACE, root, or same-uid / descendant \
                  relationship under YAMA (kernel.yama.ptrace_scope). Supports Linux x86_64 \
                  and aarch64 (same-arch only) targets with a statically-linked jemalloc and \
                  DWARF debuginfo reachable from the target ELF — either inline on the binary \
                  carrying the jemalloc TLS symbol or as a paired external debug file \
                  discovered via .gnu_debuglink / NT_GNU_BUILD_ID (distro -dbg / -debuginfo \
                  packages under /usr/lib/debug).\n\n\
                  Sampling mode: pass `--snapshots N --interval-ms MS` to take N snapshots \
                  separated by MS milliseconds. Symbol resolution runs once; each snapshot \
                  attach/detaches per tid and threads are released during the inter-snapshot \
                  sleep so the workload is not held stopped across the run.\n\n\
                  Sidecar enrichment: pass `--sidecar PATH` to append probe metrics into an \
                  existing ktstr sidecar file. The file MUST exist — run the test first to \
                  generate it, then re-invoke with `--sidecar`.\n\n\
                  CI deadline: pass `--abort-after-ms MS` to self-abort after MS \
                  milliseconds, producing a partial ProbeOutput with interrupted: true \
                  instead of hanging indefinitely on a wedged snapshot loop."
)]
struct Cli {
    /// Target process id. Required.
    #[arg(long, value_parser = clap::value_parser!(i32).range(1..))]
    pid: i32,
    /// Emit JSON on stdout instead of a human-readable table.
    #[arg(long)]
    json: bool,
    /// Append probe output to an existing ktstr sidecar JSON file.
    /// The probe synthesizes a [`PayloadMetrics`] entry from its
    /// own output (walking numeric JSON leaves into `name: value`
    /// records), appends it to `sidecar.metrics`, and writes the
    /// result back atomically (tempfile + rename) under an
    /// exclusive advisory lock.
    ///
    /// Sidecar file MUST already exist. Run the target test first
    /// so the harness writes the sidecar, then invoke the probe
    /// with `--sidecar` to enrich it post-hoc.
    #[arg(long, value_name = "PATH")]
    sidecar: Option<PathBuf>,
    /// Number of snapshots to take. Defaults to 1 (single-snapshot
    /// mode). Values > 1 engage multi-snapshot mode and require
    /// `--interval-ms`. Range 1..=100_000.
    #[arg(
        long,
        default_value_t = 1,
        value_parser = clap::value_parser!(u32).range(1..=100_000),
        value_name = "N",
    )]
    snapshots: u32,
    /// Milliseconds to wait between consecutive snapshots. Required
    /// (and only meaningful) when `--snapshots > 1`. Range
    /// 1..=3_600_000 (1 ms to 1 hour).
    #[arg(
        long,
        value_parser = clap::value_parser!(u64).range(1..=3_600_000),
        value_name = "MS",
    )]
    interval_ms: Option<u64>,
    /// Self-abort deadline in milliseconds. When set, a dedicated
    /// timer thread sleeps the deadline then flips
    /// [`CLEANUP_REQUESTED`], producing a partial `ProbeOutput`
    /// with `interrupted: true` instead of hanging. Range
    /// 1..=3_600_000.
    #[arg(
        long,
        value_parser = clap::value_parser!(u64).range(1..=3_600_000),
        value_name = "MS",
    )]
    abort_after_ms: Option<u64>,
}

impl Cli {
    /// Validate `--snapshots` / `--interval-ms` combination consistency
    /// beyond what clap's declarative attributes cover.
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
#[derive(Debug, Serialize)]
struct ProbeOutput {
    schema_version: u32,
    pid: i32,
    tool_version: &'static str,
    /// Unix-epoch seconds at the start of the probe run (before any
    /// per-tid work, before the first snapshot).
    started_at_unix_sec: u64,
    /// Configured inter-snapshot delay in milliseconds. Present
    /// only on multi-snapshot runs (`--snapshots > 1`); omitted via
    /// `skip_serializing_if` for single-snapshot runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    interval_ms: Option<u64>,
    /// `true` iff the run ended early because a SIGINT / SIGTERM
    /// arrived during the snapshot loop or inter-snapshot sleep,
    /// or a configured `--abort-after-ms` deadline fired.
    interrupted: bool,
    snapshots: Vec<Snapshot>,
}

/// One snapshot inside [`ProbeOutput::snapshots`].
#[derive(Debug, Serialize)]
struct Snapshot {
    /// Unix-epoch seconds at the start of this snapshot's per-tid
    /// attach/read/detach loop.
    timestamp_unix_sec: u64,
    /// Nanoseconds since [`ProbeOutput::started_at_unix_sec`], as
    /// measured by `CLOCK_MONOTONIC` at the start of this
    /// snapshot's per-tid loop. Immune to wall-clock jumps.
    elapsed_since_start_ns: u64,
    threads: Vec<ThreadResult>,
}

/// Per-thread probe outcome.
///
/// Wire format: `#[serde(untagged)]` by deliberate choice. The two
/// variants have disjoint field sets so consumers can discriminate
/// via field presence without a tag.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ThreadResult {
    Ok {
        tid: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        comm: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_time_jiffies: Option<u64>,
        allocated_bytes: u64,
        deallocated_bytes: u64,
    },
    Err {
        tid: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        comm: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        start_time_jiffies: Option<u64>,
        error: String,
        error_kind: ThreadErrorKind,
    },
}

/// Structural classifier for per-thread probe failures. Mirrors the
/// engine's [`host_thread_probe::ProbeError`] taxonomy 1:1 with the
/// kebab-case wire tokens that downstream consumers grep against —
/// the same vocabulary `ProbeError::tag()` already returns, the
/// same vocabulary `FatalKind::tag()` emits, so a single token
/// stream reaches every consumer regardless of which variant
/// produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, strum::EnumIter)]
#[serde(rename_all = "kebab-case")]
enum ThreadErrorKind {
    PtraceSeize,
    PtraceInterrupt,
    Waitpid,
    GetRegset,
    ProcessVmReadv,
    TlsArithmetic,
}

impl std::fmt::Display for ThreadErrorKind {
    /// Renders the same kebab-case tokens emitted by the
    /// `#[serde(rename_all = "kebab-case")]` JSON serialization.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let token = match self {
            Self::PtraceSeize => "ptrace-seize",
            Self::PtraceInterrupt => "ptrace-interrupt",
            Self::Waitpid => "waitpid",
            Self::GetRegset => "get-regset",
            Self::ProcessVmReadv => "process-vm-readv",
            Self::TlsArithmetic => "tls-arithmetic",
        };
        f.write_str(token)
    }
}

impl ThreadErrorKind {
    /// Translate the engine's [`ProbeError`] taxonomy onto the
    /// bin-side classifier via a variant-on-variant match for
    /// compile-time exhaustiveness — `ProbeError` is source-shared
    /// into this binary via `#[path = "../host_thread_probe.rs"]`,
    /// so it counts as IN-CRATE and the `#[non_exhaustive]` marker
    /// does not force a wildcard arm. A new ProbeError variant
    /// fails the match here at compile time, which is exactly the
    /// drift signal we want — the bin grows a matching
    /// `ThreadErrorKind` variant in the same change rather than
    /// silently aliasing the new error onto an existing one.
    /// Tests in `thread_error_kind_*` cover the wire shape and the
    /// variant-count parity invariant.
    fn from_probe_error(err: &ProbeError) -> Self {
        match err {
            ProbeError::PtraceSeize(_) => Self::PtraceSeize,
            ProbeError::PtraceInterrupt(_) => Self::PtraceInterrupt,
            ProbeError::Waitpid(_) => Self::Waitpid,
            ProbeError::GetRegset(_) => Self::GetRegset,
            ProbeError::ProcessVmReadv(_) => Self::ProcessVmReadv,
            ProbeError::TlsArithmetic(_) => Self::TlsArithmetic,
        }
    }
}

// ---------------------------------------------------------------------
// Signal handling
// ---------------------------------------------------------------------

/// Tracks which tids the engine has seized so SIGINT cleanup can
/// detach them. The engine itself maintains a per-call detach guard
/// that runs on Drop; this set tracks attach state at the per-tid
/// level so the SIGINT handler can sweep tids the engine left
/// attached if the bin is killed mid-`probe_thread_with_cache`
/// between the engine's seize and detach.
static ATTACHED: OnceLock<Mutex<BTreeSet<i32>>> = OnceLock::new();

/// Cleanup-requested flag, flipped by the SIGINT / SIGTERM handler
/// (in-band) and by the `--abort-after-ms` deadline timer thread
/// (out-of-band) and polled by the snapshot loop + sleep retry.
///
/// Ordering invariant: every load AND every store uses
/// [`Ordering::SeqCst`].
static CLEANUP_REQUESTED: AtomicBool = AtomicBool::new(false);

fn attached() -> &'static Mutex<BTreeSet<i32>> {
    ATTACHED.get_or_init(|| Mutex::new(BTreeSet::new()))
}

/// Acquire the ATTACHED mutex, recovering from poisoning so a panic
/// in one thread cannot prevent detach cleanup from running in the
/// next.
fn attached_lock() -> std::sync::MutexGuard<'static, BTreeSet<i32>> {
    attached().lock().unwrap_or_else(|e| e.into_inner())
}

extern "C" fn on_sigint(_sig: i32) {
    // Async-signal-safe: just flip the flag and let the snapshot
    // loop drain. Cannot touch the Mutex from signal context, but
    // the iteration check in `take_snapshot` catches it between
    // tids.
    CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
}

/// No-op SIGALRM handler. Sole purpose: interrupt a blocking
/// syscall on the main thread so the in-flight `waitpid` / `flock` /
/// sleep returns `EINTR` and the retry loop observes
/// [`CLEANUP_REQUESTED`] on the next poll boundary.
///
/// CRITICAL: this handler MUST be installed via
/// [`nix::sys::signal::sigaction`] with `SaFlags::empty()` —
/// explicitly clearing `SA_RESTART`. The BSD-compatible
/// `nix::sys::signal::signal` helper sets `SA_RESTART` by default,
/// which silently restarts interrupted syscalls and breaks the
/// `--abort-after-ms` deadline mechanism.
extern "C" fn on_sigalrm(_sig: i32) {
    // Intentionally empty: the syscall interruption IS the work.
}

/// Install a SIGINT / SIGTERM / SIGALRM handler set.
fn install_cleanup_handler() {
    for sig in [Signal::SIGINT, Signal::SIGTERM] {
        // SAFETY: `on_sigint` only touches an `AtomicBool`, which
        // is async-signal-safe.
        unsafe {
            let _ = signal(sig, SigHandler::Handler(on_sigint));
        }
    }
    let action = SigAction::new(
        SigHandler::Handler(on_sigalrm),
        SaFlags::empty(),
        SigSet::empty(),
    );
    // SAFETY: `on_sigalrm` is empty — trivially async-signal-safe.
    unsafe {
        let _ = sigaction(Signal::SIGALRM, &action);
    }
}

/// Detach everything still in `ATTACHED`. Engine-side detach guards
/// already detach on successful return; this sweep covers tids
/// whose engine call was interrupted between seize and Drop by
/// signal-driven exit.
fn detach_all_attached() {
    let tids: Vec<i32> = attached_lock().iter().copied().collect();
    for tid in tids {
        let _ = ptrace::detach(Pid::from_raw(tid), None);
        attached_lock().remove(&tid);
    }
}

// ---------------------------------------------------------------------
// /proc/<pid>/{task,exe} accessors and per-thread enrichment
// ---------------------------------------------------------------------

/// Enumerate thread ids for a target pid from `/proc/<pid>/task/`.
/// Returns them sorted so output ordering is deterministic across
/// runs.
fn iter_task_ids(pid: i32) -> Result<Vec<i32>> {
    let path = format!("/proc/{pid}/task");
    let entries = fs::read_dir(&path).with_context(|| format!("read_dir {path}"))?;
    let mut tids: Vec<i32> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse().ok()))
        .collect();
    tids.sort_unstable();
    Ok(tids)
}

/// Best-effort read of `/proc/{pid}/task/{tid}/comm`. Trims
/// surrounding whitespace, handling the kernel's trailing newline.
/// Returns `None` on any read failure.
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

/// Best-effort read of field 22 (`starttime`) from
/// `/proc/{pid}/task/{tid}/stat`: clock ticks since boot at which
/// this task was started, per `proc(5)`. Paired with `tid` this
/// forms a `(tid, start_time)` composite identity that survives
/// kernel pid reuse.
fn read_thread_start_time(pid: i32, tid: i32) -> Option<u64> {
    let path = format!("/proc/{pid}/task/{tid}/stat");
    let raw = fs::read_to_string(path).ok()?;
    parse_start_time_from_stat(&raw)
}

/// Pure parser for the `starttime` (field 22) extraction. Split
/// from [`read_thread_start_time`] so unit tests exercise the
/// comm-contains-`)` robustness without touching `/proc`.
///
/// `rfind(')')` locates the `comm` field's closing paren, and the
/// whitespace-split tail past it is indexed at offset 19 to reach
/// field 22. `lines().next()` pins the parse to the first line so
/// multi-line input cannot misalign the field index.
fn parse_start_time_from_stat(raw: &str) -> Option<u64> {
    let line = raw.lines().next()?;
    let last_close = line.rfind(')')?;
    let tail = line.get(last_close + 1..)?;
    let mut fields = tail.split_ascii_whitespace();
    // Skip fields 3..=21 (19 tokens) to land on field 22.
    for _ in 0..19 {
        fields.next()?;
    }
    fields.next()?.parse::<u64>().ok()
}

/// Stable identity of the target's on-disk executable, captured by
/// `stat(2)` on `/proc/<pid>/exe`. (dev, inode) uniquely identifies
/// the file; re-stating between snapshots lets the probe detect a
/// mid-run `execve` (new inode, same pid) or pid recycling and
/// bail with `Fatal` rather than reading stale TLS offsets.
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

/// Re-stat the target's `/proc/<pid>/exe` and bail fatal if the
/// identity (dev, inode) differs from the captured baseline.
fn ensure_exe_identity_unchanged(
    pid: i32,
    baseline: &ExeIdentity,
    context: &'static str,
) -> std::result::Result<(), anyhow::Error> {
    match ExeIdentity::capture(pid) {
        Ok(current) if current != *baseline => Err(anyhow!(
            "target pid {pid} /proc/<pid>/exe changed {context} \
             (captured dev={:#x} ino={}, now dev={:#x} ino={}); \
             target execve'd or pid recycled, TLS offsets invalid",
            baseline.dev,
            baseline.ino,
            current.dev,
            current.ino,
        )),
        Ok(_) => Ok(()),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------

/// Outcome classification so `main` can decide the exit code without
/// re-inspecting the `snapshots` vec.
enum RunOutcome {
    Ok(ProbeOutput),
    AllFailed(ProbeOutput),
    Fatal(FatalError),
}

/// Closed vocabulary for `RunOutcome::Fatal` structured stderr tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::EnumIter)]
enum FatalKind {
    PidMissing,
    ExeIdentityChanged,
    JemallocNotFound,
    JemallocInDso,
    ReadlinkFailure,
    MapsReadFailure,
    DwarfParseFailure,
    ArchMismatch,
    SelfProbeRejected,
    TidEnumerationFailure,
    Other,
}

impl FatalKind {
    /// Short token emitted after `ktstr-probe-fatal:` on stderr.
    fn tag(self) -> &'static str {
        match self {
            Self::PidMissing => "pid-missing",
            Self::ExeIdentityChanged => "exe-identity-changed",
            Self::JemallocNotFound => "jemalloc-not-found",
            Self::JemallocInDso => "jemalloc-in-dso",
            Self::ReadlinkFailure => "readlink-failure",
            Self::MapsReadFailure => "maps-read-failure",
            Self::DwarfParseFailure => "dwarf-parse-failure",
            Self::ArchMismatch => "arch-mismatch",
            Self::SelfProbeRejected => "self-probe-rejected",
            Self::TidEnumerationFailure => "tid-enumeration-failure",
            Self::Other => "other",
        }
    }

    /// Translate the engine's [`AttachError`] taxonomy into the
    /// bin-side `FatalKind` via a variant-on-variant match for
    /// compile-time exhaustiveness — `AttachError` is source-shared
    /// into this binary via `#[path = "../host_thread_probe.rs"]`,
    /// so the `#[non_exhaustive]` marker does not force a wildcard
    /// arm. A new AttachError variant fails the match here at
    /// compile time, which is the drift signal we want. Both
    /// vocabularies share kebab-case tokens 1:1 for the seven
    /// AttachError variants. `FatalKind::Other` is reserved as a
    /// forward-compat sink; it currently has no producers —
    /// `from_attach_error` can no longer route into it after the
    /// exhaustive variant match refactor.
    fn from_attach_error(err: &AttachError) -> Self {
        match err {
            AttachError::PidMissing(_) => Self::PidMissing,
            AttachError::ReadlinkFailure(_) => Self::ReadlinkFailure,
            AttachError::MapsReadFailure(_) => Self::MapsReadFailure,
            AttachError::JemallocNotFound(_) => Self::JemallocNotFound,
            AttachError::JemallocInDso(_) => Self::JemallocInDso,
            AttachError::ArchMismatch(_) => Self::ArchMismatch,
            AttachError::DwarfParseFailure(_) => Self::DwarfParseFailure,
        }
    }
}

/// A [`FatalKind`] paired with the underlying error.
struct FatalError {
    kind: FatalKind,
    error: anyhow::Error,
}

impl FatalError {
    fn new(kind: FatalKind, error: anyhow::Error) -> Self {
        Self { kind, error }
    }

    fn pid_missing(error: anyhow::Error) -> Self {
        Self::new(FatalKind::PidMissing, error)
    }

    fn exe_identity_changed(error: anyhow::Error) -> Self {
        Self::new(FatalKind::ExeIdentityChanged, error)
    }

    fn self_probe_rejected(error: anyhow::Error) -> Self {
        Self::new(FatalKind::SelfProbeRejected, error)
    }

    fn tid_enumeration_failure(error: anyhow::Error) -> Self {
        Self::new(FatalKind::TidEnumerationFailure, error)
    }

    /// Construct from the engine's `AttachError`. Maps the engine's
    /// kebab-case tag onto the bin's `FatalKind` and unwraps the
    /// underlying `anyhow::Error` so the operator-facing message
    /// surfaces with full context.
    fn from_attach_error(err: AttachError) -> Self {
        let kind = FatalKind::from_attach_error(&err);
        let error = match err {
            AttachError::PidMissing(e)
            | AttachError::ReadlinkFailure(e)
            | AttachError::MapsReadFailure(e)
            | AttachError::JemallocNotFound(e)
            | AttachError::JemallocInDso(e)
            | AttachError::ArchMismatch(e)
            | AttachError::DwarfParseFailure(e) => e,
        };
        Self::new(kind, error)
    }
}

/// Granularity (ms) at which [`sleep_with_cancel`] wakes to poll
/// [`CLEANUP_REQUESTED`].
const CANCEL_POLL_TICK_MS: u64 = 10;

/// Sleep for `total_ms` milliseconds or until [`CLEANUP_REQUESTED`]
/// is observed, whichever is first. Returns `true` if the sleep was
/// cancelled by the cleanup flag.
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

/// Take one snapshot: iterate the tids, probe each via the engine's
/// [`probe_thread_with_cache`], return a [`Snapshot`] carrying the
/// timestamp + per-thread results.
///
/// Per-tid TP cache: a `HashMap<tid, observed_tp>` is threaded
/// through so snapshots 2..N skip the engine's slow path
/// (seize/interrupt/wait/getregset/detach) for tids observed on a
/// previous snapshot. Cache entries are evicted up-front for tids
/// that fell out of the live enumeration so a kernel-recycled tid
/// cannot hit a stale entry.
fn take_snapshot(
    pid: i32,
    probe: &JemallocProbe,
    tids: &[i32],
    run_start: std::time::Instant,
    tp_cache: &mut std::collections::HashMap<i32, u64>,
) -> (Snapshot, bool) {
    let timestamp_unix_sec = now_unix_sec();
    let elapsed_since_start_ns = run_start.elapsed().as_nanos() as u64;

    // Evict cache entries for tids no longer in the live enumeration
    // BEFORE any lookups this snapshot. An exited tid eventually
    // drops out of `/proc/<pid>/task/`; the kernel may then recycle
    // that tid for a freshly-created thread inside the same tgid.
    // Without this eviction, the recycled tid would hit a stale TP
    // cached against the prior thread.
    let live_tids: BTreeSet<i32> = tids.iter().copied().collect();
    tp_cache.retain(|tid, _| live_tids.contains(tid));

    let mut threads: Vec<ThreadResult> = Vec::with_capacity(tids.len());
    let mut interrupted = false;
    for &tid in tids {
        if CLEANUP_REQUESTED.load(Ordering::SeqCst) {
            detach_all_attached();
            interrupted = true;
            break;
        }
        // Read comm + starttime BEFORE probe: on failure paths the
        // tid may exit mid-probe, and the pre-probe read has the
        // best chance of catching populated diagnostic fields.
        let comm = read_thread_comm(pid, tid);
        let start_time_jiffies = read_thread_start_time(pid, tid);
        let cached_tp = tp_cache.get(&tid).copied();
        // Track the slow-path attach in ATTACHED so the SIGINT
        // sweep covers tids whose engine call was interrupted
        // between seize and Drop. Cache-hit calls do not seize, so
        // they are not registered.
        let registered_for_cleanup = cached_tp.is_none();
        if registered_for_cleanup {
            attached_lock().insert(tid);
        }
        let result = probe_thread_with_cache(probe, tid, cached_tp);
        if registered_for_cleanup {
            attached_lock().remove(&tid);
        }
        match result {
            Ok((c, observed_tp)) => {
                tp_cache.insert(tid, observed_tp);
                threads.push(ThreadResult::Ok {
                    tid,
                    comm,
                    start_time_jiffies,
                    allocated_bytes: c.allocated_bytes,
                    deallocated_bytes: c.deallocated_bytes,
                });
            }
            Err(e) => {
                let kind = ThreadErrorKind::from_probe_error(&e);
                let error = format!("{:#}", e.source());
                threads.push(ThreadResult::Err {
                    tid,
                    comm,
                    start_time_jiffies,
                    error,
                    error_kind: kind,
                });
            }
        }
    }
    (
        Snapshot {
            timestamp_unix_sec,
            elapsed_since_start_ns,
            threads,
        },
        interrupted,
    )
}

/// True iff `threads` is empty or every entry is a
/// [`ThreadResult::Err`].
fn all_failed(threads: &[ThreadResult]) -> bool {
    threads.is_empty()
        || threads
            .iter()
            .all(|t| matches!(t, ThreadResult::Err { .. }))
}

/// True iff every snapshot in `snapshots` is itself all-failed.
/// Empty `snapshots` slice satisfies vacuously — callers MUST
/// handle the empty case explicitly.
fn multi_snapshot_all_failed(snapshots: &[Snapshot]) -> bool {
    snapshots.iter().all(|s| all_failed(&s.threads))
}

fn run(cli: &Cli) -> RunOutcome {
    let started_at_unix_sec = now_unix_sec();
    let run_start = std::time::Instant::now();
    let pid = cli.pid;
    // Self-probe reject: PTRACE_SEIZE refuses a tracer's own tgid.
    let self_pid = self_pid();
    if pid == self_pid {
        return RunOutcome::Fatal(FatalError::self_probe_rejected(anyhow!(
            "refusing to probe self (pid {pid} == ktstr-jemalloc-probe's own pid). \
             ptrace(PTRACE_SEIZE) rejects self-attach — a process cannot trace \
             itself. Run the probe from a separate process against the target's pid."
        )));
    }
    if !Path::new(&format!("/proc/{pid}")).exists() {
        return RunOutcome::Fatal(FatalError::pid_missing(anyhow!("pid {pid} does not exist")));
    }

    // Capture the target ELF's (dev, inode) BEFORE the engine's
    // ELF/DWARF parse so the parse is inside the identity window.
    let exe_identity = match ExeIdentity::capture(pid) {
        Ok(v) => v,
        Err(e) => return RunOutcome::Fatal(FatalError::pid_missing(e)),
    };

    // Engine call: resolve symbols + DWARF offsets ONCE per run.
    let probe = match attach_jemalloc(pid) {
        Ok(p) => p,
        Err(e) => return RunOutcome::Fatal(FatalError::from_attach_error(e)),
    };

    // Re-stat AFTER the parse. If the target execve'd during the
    // parse window the symbol/offsets we cached no longer match
    // /proc/<pid>/exe.
    if let Err(e) = ensure_exe_identity_unchanged(pid, &exe_identity, "during ELF/DWARF parse") {
        return RunOutcome::Fatal(FatalError::exe_identity_changed(e));
    }

    let snapshot_count = cli.snapshots as usize;
    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(snapshot_count);
    let mut interrupted = false;
    let mut tp_cache: std::collections::HashMap<i32, u64> = std::collections::HashMap::new();
    for i in 0..cli.snapshots {
        if CLEANUP_REQUESTED.load(Ordering::SeqCst) {
            interrupted = true;
            break;
        }
        // Re-stat the target's /proc/<pid>/exe between snapshots.
        // Skip on iteration 0 because `exe_identity` was just
        // captured before the loop.
        if i > 0
            && let Err(e) = ensure_exe_identity_unchanged(pid, &exe_identity, "between snapshots")
        {
            return RunOutcome::Fatal(FatalError::exe_identity_changed(e));
        }
        // Re-enumerate /proc/<pid>/task per snapshot so threads
        // spawned AFTER the previous enumeration are visible.
        let tids = match iter_task_ids(pid) {
            Ok(v) => v,
            Err(e) => return RunOutcome::Fatal(FatalError::tid_enumeration_failure(e)),
        };
        let (snap, snap_interrupted) = take_snapshot(pid, &probe, &tids, run_start, &mut tp_cache);
        snapshots.push(snap);
        if snap_interrupted {
            interrupted = true;
            break;
        }
        // No sleep after the LAST snapshot.
        if i + 1 < cli.snapshots {
            let interval_ms = cli
                .interval_ms
                .expect("interval_ms guaranteed Some for snapshots > 1 by validate_sampling_flags");
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
    if out.snapshots.is_empty() {
        RunOutcome::Ok(out)
    } else if multi_snapshot_all_failed(&out.snapshots) {
        RunOutcome::AllFailed(out)
    } else {
        RunOutcome::Ok(out)
    }
}

// ---------------------------------------------------------------------
// Output rendering
// ---------------------------------------------------------------------

/// Render one `ThreadResult` to stdout (Ok path) or stderr (Err
/// path) in the human-readable format shared by single-snapshot
/// and multi-snapshot modes.
fn print_thread_result(t: &ThreadResult) {
    match t {
        ThreadResult::Ok {
            tid,
            comm,
            allocated_bytes,
            deallocated_bytes,
            ..
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
            ..
        } => {
            let comm_suffix = format_comm_suffix(comm.as_deref());
            eprintln!("warning: tid {tid}{comm_suffix} [{error_kind}]: {error}");
        }
    }
}

/// Emit [`ProbeOutput`] in the selected format.
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

// ---------------------------------------------------------------------
// Sidecar enrichment
// ---------------------------------------------------------------------

/// Payload name recorded as an identifying metric when the probe
/// appends to a sidecar.
const SIDECAR_METRIC_PREFIX: &str = "jemalloc_probe";

/// Upgrade a [`Metric`]'s unit + polarity based on its flat-path
/// name. Names ending in `.allocated_bytes` or `.deallocated_bytes`
/// become `(Polarity::LowerBetter, "bytes")`.
fn apply_probe_metric_hints(m: &mut ktstr::test_support::Metric) {
    use ktstr::test_support::Polarity;
    if m.name.ends_with(".allocated_bytes") || m.name.ends_with(".deallocated_bytes") {
        m.polarity = Polarity::LowerBetter;
        m.unit = "bytes".to_string();
    }
}

/// Synthesize a [`PayloadMetrics`] from a [`ProbeOutput`] so the
/// result can land in a [`SidecarResult::metrics`] vec.
fn synthesize_payload_metrics(
    out: &ProbeOutput,
    exit_code: i32,
    payload_index: usize,
) -> Result<ktstr::test_support::PayloadMetrics> {
    use ktstr::test_support::{MetricSource, MetricStream, PayloadMetrics, walk_json_leaves};
    let value = serde_json::to_value(out)
        .context("serialize ProbeOutput to serde_json::Value for sidecar append")?;
    let mut metrics = walk_json_leaves(&value, MetricSource::Json, MetricStream::Stdout);
    for m in &mut metrics {
        m.name = format!("{SIDECAR_METRIC_PREFIX}.{}", m.name);
        apply_probe_metric_hints(m);
    }
    Ok(PayloadMetrics {
        payload_index,
        metrics,
        exit_code,
    })
}

/// Append a synthesized [`PayloadMetrics`] to the
/// [`SidecarResult::metrics`] vec of the sidecar file at `path`.
/// Atomic via tempfile + rename under an exclusive advisory lock.
fn append_probe_output_to_sidecar(path: &Path, out: &ProbeOutput, exit_code: i32) -> Result<()> {
    use ktstr::test_support::SidecarResult;
    use rustix::fs::{FlockOperation, Mode, OFlags, flock, open};

    // Flock on a SIBLING lock file, not on the sidecar itself —
    // the atomic rename below replaces the sidecar's inode, which
    // would invalidate any lock held on the old inode.
    let lock_path = path.with_extension({
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext.is_empty() {
            "lock".to_string()
        } else {
            format!("{ext}.lock")
        }
    });
    let lock_fd = open(
        &lock_path,
        OFlags::CREATE | OFlags::RDWR | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .with_context(|| format!("open lock file {}", lock_path.display()))?;
    const FLOCK_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);
    const FLOCK_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);
    let deadline = std::time::Instant::now() + FLOCK_BUDGET;
    loop {
        // Honor the probe-wide deadline / SIGINT / SIGTERM path
        // BEFORE the flock syscall so an `--abort-after-ms`
        // against a contested lock does not hang for the full
        // 30 s budget.
        if CLEANUP_REQUESTED.load(Ordering::SeqCst) {
            bail!(
                "sidecar append aborted by probe deadline (SIGINT / SIGTERM / --abort-after-ms) \
                 while waiting on flock(LOCK_EX) on {}",
                lock_path.display(),
            );
        }
        match flock(&lock_fd, FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => break,
            Err(rustix::io::Errno::WOULDBLOCK) if std::time::Instant::now() < deadline => {
                std::thread::sleep(FLOCK_RETRY_INTERVAL);
                continue;
            }
            Err(rustix::io::Errno::WOULDBLOCK) => bail!(
                "flock(LOCK_EX) on {} timed out after {:?} — another \
                 --sidecar writer holds the lock. Run `lslocks | grep {}` \
                 or `fuser {}` to identify the flock holder; if it is a \
                 wedged probe, kill it and re-run.",
                lock_path.display(),
                FLOCK_BUDGET,
                lock_path.display(),
                lock_path.display(),
            ),
            Err(e) => {
                return Err(anyhow::Error::from(e).context(format!(
                    "flock(LOCK_EX, non-blocking) on {}",
                    lock_path.display(),
                )));
            }
        }
    }

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

    let payload_metrics = synthesize_payload_metrics(out, exit_code, sidecar.metrics.len())?;
    sidecar.metrics.push(payload_metrics);

    let serialized = serde_json::to_string_pretty(&sidecar)
        .context("re-serialize SidecarResult after appending probe metrics")?;

    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("sidecar path {} has no parent directory", path.display()))?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("create staging file in {}", dir.display()))?;
    std::io::Write::write_all(tmp.as_file_mut(), serialized.as_bytes())
        .with_context(|| format!("write staging file in {}", dir.display()))?;
    tmp.as_file()
        .sync_all()
        .with_context(|| format!("fsync staging file in {}", dir.display()))?;
    tmp.persist(path)
        .with_context(|| format!("atomic rename staging file into {}", path.display()))?;

    let parent_dir = path.parent().unwrap_or(Path::new("."));
    match std::fs::File::open(parent_dir) {
        Ok(parent) => {
            if let Err(e) = parent.sync_all() {
                tracing::warn!(
                    dir = %parent_dir.display(),
                    err = %format!("{e:#}"),
                    "jemalloc_probe: parent-directory fsync failed after \
                     rename; the renamed sidecar is visible in the VFS but a \
                     concurrent crash could drop the directory-entry update \
                     from durable storage",
                );
            }
        }
        Err(e) => tracing::warn!(
            dir = %parent_dir.display(),
            err = %format!("{e:#}"),
            "jemalloc_probe: could not open parent directory for fsync; \
             the rename already committed but the directory entry has no \
             explicit durability guarantee",
        ),
    }

    drop(lock_fd);
    Ok(())
}

fn main() {
    // Restore SIGPIPE so piping the probe's JSON output to `jq |
    // less` or similar doesn't panic inside `print!`.
    ktstr::cli::restore_sigpipe_default();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .try_init();
    install_cleanup_handler();
    let cli = Cli::parse();
    if let Err(e) = cli.validate_sampling_flags() {
        eprintln!("error: {e:#}");
        std::process::exit(2);
    }
    // Pre-flight `--sidecar` path validation so a typo fails fast.
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
    // `--abort-after-ms MS`: spawn a detached timer thread that
    // flips `CLEANUP_REQUESTED` after MS milliseconds, then sends
    // SIGALRM to the main thread via `tgkill` so any in-flight
    // blocking syscall returns `EINTR`.
    if let Some(ms) = cli.abort_after_ms {
        // SAFETY: `gettid(2)` takes no arguments and returns the
        // calling thread's tid; always safe.
        let main_tid = unsafe { libc::syscall(libc::SYS_gettid) } as libc::pid_t;
        let main_pid = std::process::id() as libc::pid_t;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(ms));
            eprintln!("ktstr-probe-deadline: abort after {ms}ms");
            CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
            // SAFETY: `tgkill(2)` is async-signal-safe.
            unsafe {
                libc::syscall(libc::SYS_tgkill, main_pid, main_tid, libc::SIGALRM);
            }
        });
    }
    match run(&cli) {
        RunOutcome::Ok(out) => {
            if let Err(e) = print_output(&cli, &out) {
                eprintln!("error writing output: {e:#}");
                std::process::exit(1);
            }
            // Sidecar-append failure uses a DISTINCT exit code (3)
            // from probe-failure (1). The probe stdout already
            // carries the full `ProbeOutput` successfully on this
            // branch.
            if let Some(path) = cli.sidecar.as_deref()
                && let Err(e) = append_probe_output_to_sidecar(path, &out, 0)
            {
                eprintln!("sidecar append failed (exit 3): {}: {e:#}", path.display());
                std::process::exit(3);
            }
        }
        RunOutcome::AllFailed(out) => {
            let is_multi = cli.snapshots > 1;
            let marker = if is_multi { "multi" } else { "single" };
            if let Err(e) = print_output(&cli, &out) {
                eprintln!("error writing output: {e:#}");
            }
            // Record the all-failed outcome in the sidecar BEFORE
            // the final exit so downstream stats tooling sees the
            // probe's per-tid error records.
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
        RunOutcome::Fatal(fatal) => {
            // Tag shape: `ktstr-probe-fatal: <kind>` with `kind`
            // drawn from [`FatalKind`]'s closed vocabulary. Fatal
            // does NOT append a stub `PayloadMetrics` to the
            // sidecar — the probe never reached the point where
            // `ProbeOutput` gets assembled.
            eprintln!("ktstr-probe-fatal: {}", fatal.kind.tag());
            eprintln!("error: {:#}", fatal.error);
            detach_all_attached();
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------
// Tests (bin-only; engine helper tests live in
// src/host_thread_probe.rs::tests)
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `iter_task_ids` of /proc/self/task must return at least the
    /// current thread.
    #[test]
    fn iter_task_ids_self() {
        let pid = self_pid();
        let tids = iter_task_ids(pid).expect("self/task must be readable");
        assert!(!tids.is_empty());
        assert!(tids.windows(2).all(|w| w[0] <= w[1]), "tids must be sorted");
    }

    /// JSON schema v3: success + error arms round-trip via serde.
    #[test]
    fn thread_result_json_shape() {
        let ok = ThreadResult::Ok {
            tid: 42,
            comm: Some("worker-0".to_string()),
            start_time_jiffies: None,
            allocated_bytes: 1024,
            deallocated_bytes: 512,
        };
        let ok_no_comm = ThreadResult::Ok {
            tid: 44,
            comm: None,
            start_time_jiffies: None,
            allocated_bytes: 2048,
            deallocated_bytes: 1024,
        };
        let err = ThreadResult::Err {
            tid: 43,
            comm: None,
            start_time_jiffies: None,
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
                elapsed_since_start_ns: 0,
                threads: vec![ok, ok_no_comm, err],
            }],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(s.contains("\"schema_version\":3"));
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
        assert!(!s.contains("\"comm\":null"));
        assert!(!s.contains("\"interval_ms\":null"));
    }

    /// Canonical kebab-case token for each `ThreadErrorKind`.
    fn expected_error_kind_token(k: ThreadErrorKind) -> &'static str {
        match k {
            ThreadErrorKind::PtraceSeize => "ptrace-seize",
            ThreadErrorKind::PtraceInterrupt => "ptrace-interrupt",
            ThreadErrorKind::Waitpid => "waitpid",
            ThreadErrorKind::GetRegset => "get-regset",
            ThreadErrorKind::ProcessVmReadv => "process-vm-readv",
            ThreadErrorKind::TlsArithmetic => "tls-arithmetic",
        }
    }

    #[test]
    fn thread_error_kind_kebab_case_serialization() {
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

    /// `ThreadErrorKind::from_probe_error` translates every engine
    /// `ProbeError` variant onto a wire-stable bin-side classifier.
    #[test]
    fn thread_error_kind_from_probe_error_pins_mapping() {
        let cases: Vec<(ProbeError, ThreadErrorKind)> = vec![
            (
                ProbeError::PtraceSeize(anyhow!("x")),
                ThreadErrorKind::PtraceSeize,
            ),
            (
                ProbeError::PtraceInterrupt(anyhow!("x")),
                ThreadErrorKind::PtraceInterrupt,
            ),
            (ProbeError::Waitpid(anyhow!("x")), ThreadErrorKind::Waitpid),
            (
                ProbeError::GetRegset(anyhow!("x")),
                ThreadErrorKind::GetRegset,
            ),
            (
                ProbeError::ProcessVmReadv(anyhow!("x")),
                ThreadErrorKind::ProcessVmReadv,
            ),
            (
                ProbeError::TlsArithmetic(anyhow!("x")),
                ThreadErrorKind::TlsArithmetic,
            ),
        ];
        for (engine_err, expected_bin_kind) in &cases {
            assert_eq!(
                ThreadErrorKind::from_probe_error(engine_err),
                *expected_bin_kind,
                "engine ProbeError tag {} must map to bin {:?}",
                engine_err.tag(),
                expected_bin_kind,
            );
        }
    }

    /// Variant-count parity: `ThreadErrorKind` must carry exactly
    /// the same number of variants as the engine's `ProbeError`.
    /// `from_probe_error` is an exhaustive variant-on-variant match
    /// (no wildcard, see its doc comment), so the compiler already
    /// catches a NEW `ProbeError` variant landing without a matching
    /// arm — but a NEW `ThreadErrorKind` variant added without a
    /// corresponding engine variant would compile silently. This
    /// test pins the inverse direction: a count divergence here
    /// signals "bin grew an unmoored variant" or "engine deleted a
    /// variant the bin still names" and forces the change to land
    /// on both sides simultaneously. The constant `6` mirrors the
    /// current ProbeError variant count (PtraceSeize,
    /// PtraceInterrupt, Waitpid, GetRegset, ProcessVmReadv,
    /// TlsArithmetic); both sides must update together.
    #[test]
    fn thread_error_kind_variant_count_matches_probe_error() {
        use strum::IntoEnumIterator;
        let bin_count = ThreadErrorKind::iter().count();
        // Cases vec in `thread_error_kind_from_probe_error_pins_mapping`
        // already exhausts ProbeError; this test pins the count
        // independently so a new bin-side variant without a
        // corresponding engine arm trips here even if someone
        // forgets to extend the cases vec.
        let engine_count = 6;
        assert_eq!(
            bin_count, engine_count,
            "ThreadErrorKind must mirror ProbeError 1:1 — got {bin_count} \
             bin variants vs {engine_count} engine variants. If \
             ProbeError gained or lost a variant, update both sides \
             in the same change.",
        );
    }

    /// `FatalKind::from_attach_error` translates every engine
    /// `AttachError` variant onto a `FatalKind`.
    #[test]
    fn fatal_kind_from_attach_error_pins_mapping() {
        let cases: Vec<(AttachError, FatalKind)> = vec![
            (AttachError::PidMissing(anyhow!("x")), FatalKind::PidMissing),
            (
                AttachError::ReadlinkFailure(anyhow!("x")),
                FatalKind::ReadlinkFailure,
            ),
            (
                AttachError::MapsReadFailure(anyhow!("x")),
                FatalKind::MapsReadFailure,
            ),
            (
                AttachError::JemallocNotFound(anyhow!("x")),
                FatalKind::JemallocNotFound,
            ),
            (
                AttachError::JemallocInDso(anyhow!("x")),
                FatalKind::JemallocInDso,
            ),
            (
                AttachError::ArchMismatch(anyhow!("x")),
                FatalKind::ArchMismatch,
            ),
            (
                AttachError::DwarfParseFailure(anyhow!("x")),
                FatalKind::DwarfParseFailure,
            ),
        ];
        for (engine_err, expected_kind) in &cases {
            assert_eq!(
                FatalKind::from_attach_error(engine_err),
                *expected_kind,
                "engine AttachError tag {} must map to bin {:?}",
                engine_err.tag(),
                expected_kind,
            );
        }
    }

    /// `run()` short-circuits to `RunOutcome::Fatal` when `--pid`
    /// matches the probe's own pid.
    #[test]
    fn run_rejects_self_probe() {
        let cli = Cli {
            pid: self_pid(),
            json: false,
            snapshots: 1,
            interval_ms: None,
            sidecar: None,
            abort_after_ms: None,
        };
        match run(&cli) {
            RunOutcome::Fatal(fatal) => {
                let msg = format!("{:#}", fatal.error);
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

    /// A non-self pid must NOT trigger the self-probe short-circuit.
    #[test]
    fn run_accepts_non_self_pid() {
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep for non-self pid acceptance test");
        let child_pid =
            libc::pid_t::try_from(child.id()).expect("Linux pid_max <= 2^22 so pid fits in pid_t");
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
            abort_after_ms: None,
        };
        let outcome = run(&cli);
        let _ = child.kill();
        let _ = child.wait();
        if let RunOutcome::Fatal(fatal) = outcome {
            let msg = format!("{:#}", fatal.error);
            assert!(
                !msg.contains("refusing to probe self"),
                "self-probe gate must NOT fire for non-self pid {child_pid} (self={self_pid}), got: {msg}",
            );
        }
    }

    // ---- sampling-mode CLI parsing + validation ----

    #[test]
    fn cli_default_sampling_count_is_one() {
        let cli = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "42"]).unwrap();
        assert_eq!(cli.snapshots, 1);
        assert!(cli.interval_ms.is_none());
        assert!(cli.validate_sampling_flags().is_ok());
    }

    #[test]
    fn cli_explicit_count_one_without_interval_accepted() {
        let cli = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "42", "--snapshots", "1"])
            .unwrap();
        assert_eq!(cli.snapshots, 1);
        assert!(cli.interval_ms.is_none());
        assert!(cli.validate_sampling_flags().is_ok());
    }

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

    #[test]
    fn cli_count_zero_rejected() {
        let err = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "42", "--snapshots", "0"])
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0 is not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

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

    #[test]
    fn cli_abort_after_ms_defaults_none() {
        let cli = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "42"]).unwrap();
        assert!(cli.abort_after_ms.is_none());
    }

    #[test]
    fn cli_abort_after_ms_accepts_positive_value() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--abort-after-ms",
            "500",
        ])
        .unwrap();
        assert_eq!(cli.abort_after_ms, Some(500));
    }

    #[test]
    fn cli_abort_after_ms_lower_boundary_accepted() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--abort-after-ms",
            "1",
        ])
        .unwrap();
        assert_eq!(cli.abort_after_ms, Some(1));
    }

    #[test]
    fn cli_abort_after_ms_upper_boundary_accepted() {
        let cli = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--abort-after-ms",
            "3600000",
        ])
        .unwrap();
        assert_eq!(cli.abort_after_ms, Some(3_600_000));
    }

    #[test]
    fn cli_abort_after_ms_zero_rejected() {
        let err = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--abort-after-ms",
            "0",
        ])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0 is not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    #[test]
    fn cli_abort_after_ms_upper_bound_rejected() {
        let err = Cli::try_parse_from([
            "ktstr-jemalloc-probe",
            "--pid",
            "42",
            "--abort-after-ms",
            "3600001",
        ])
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

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

    #[test]
    fn cli_pid_zero_rejected() {
        let err = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "0"]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("0 is not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    #[test]
    fn cli_pid_negative_rejected() {
        let err = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid=-1"]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not in") || msg.contains("invalid value"),
            "expected clap range-rejection message, got: {msg}",
        );
    }

    #[test]
    fn cli_count_greater_than_one_requires_interval() {
        let cli = Cli::try_parse_from(["ktstr-jemalloc-probe", "--pid", "42", "--snapshots", "3"])
            .unwrap();
        let err = cli.validate_sampling_flags().unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("requires --interval-ms"), "got: {msg}");
    }

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

    // ---- sleep_with_cancel ----

    #[test]
    fn sleep_with_cancel_completes_without_flag_set() {
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let cancelled = sleep_with_cancel(25);
        let elapsed = start.elapsed();
        assert!(
            !cancelled,
            "sleep should not report cancellation when flag stays clear"
        );
        assert!(
            elapsed >= std::time::Duration::from_millis(20),
            "sleep returned too fast: {elapsed:?}",
        );
    }

    #[test]
    fn sleep_with_cancel_observes_pre_set_flag() {
        CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let cancelled = sleep_with_cancel(10_000);
        let elapsed = start.elapsed();
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        assert!(cancelled, "pre-set flag must cause immediate cancel");
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "cancel path should return within a poll tick, got: {elapsed:?}",
        );
    }

    #[test]
    fn sleep_with_cancel_observes_deadline_thread_flip() {
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let flipper = std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
        });
        let cancelled = sleep_with_cancel(10_000);
        let elapsed = start.elapsed();
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        let _ = flipper.join();
        assert!(
            cancelled,
            "deadline thread set the flag at 50ms; sleep must report cancelled",
        );
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "sleep should return within a poll tick of the flag flip; got {elapsed:?}",
        );
        assert!(
            elapsed >= std::time::Duration::from_millis(30),
            "sleep returned before the deadline thread could flip the flag; got {elapsed:?}",
        );
    }

    /// Multi-snapshot all-failed classification.
    #[test]
    fn multi_snapshot_all_failed_classification() {
        let err_thread = || ThreadResult::Err {
            tid: 1,
            comm: None,
            start_time_jiffies: None,
            error: "e".into(),
            error_kind: ThreadErrorKind::PtraceSeize,
        };
        let ok_thread = || ThreadResult::Ok {
            tid: 2,
            comm: None,
            start_time_jiffies: None,
            allocated_bytes: 10,
            deallocated_bytes: 0,
        };
        let snap = |threads: Vec<ThreadResult>| Snapshot {
            timestamp_unix_sec: 1_700_000_000,
            elapsed_since_start_ns: 0,
            threads,
        };
        let all_err = vec![
            snap(vec![err_thread(), err_thread()]),
            snap(vec![err_thread()]),
            snap(vec![err_thread(), err_thread(), err_thread()]),
        ];
        assert!(multi_snapshot_all_failed(&all_err));
        let mixed = vec![
            snap(vec![err_thread()]),
            snap(vec![err_thread(), ok_thread()]),
            snap(vec![err_thread()]),
        ];
        assert!(!multi_snapshot_all_failed(&mixed));
        let empty_threads = vec![snap(vec![]), snap(vec![])];
        assert!(multi_snapshot_all_failed(&empty_threads));
        let empty_snapshots: &[Snapshot] = &[];
        assert!(multi_snapshot_all_failed(empty_snapshots));
    }

    #[test]
    fn all_failed_classification() {
        assert!(all_failed(&[]), "empty threads vec is all-failed");
        let only_err = vec![ThreadResult::Err {
            tid: 1,
            comm: None,
            start_time_jiffies: None,
            error: "e".into(),
            error_kind: ThreadErrorKind::PtraceSeize,
        }];
        assert!(all_failed(&only_err));
        let mixed = vec![
            ThreadResult::Err {
                tid: 1,
                comm: None,
                start_time_jiffies: None,
                error: "e".into(),
                error_kind: ThreadErrorKind::PtraceSeize,
            },
            ThreadResult::Ok {
                tid: 2,
                comm: None,
                start_time_jiffies: None,
                allocated_bytes: 10,
                deallocated_bytes: 0,
            },
        ];
        assert!(!all_failed(&mixed));
    }

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
                    elapsed_since_start_ns: 0,
                    threads: vec![ThreadResult::Ok {
                        tid: 42,
                        comm: Some("worker".to_string()),
                        start_time_jiffies: None,
                        allocated_bytes: 1024,
                        deallocated_bytes: 0,
                    }],
                },
                Snapshot {
                    timestamp_unix_sec: 1_700_000_001,
                    elapsed_since_start_ns: 0,
                    threads: vec![ThreadResult::Ok {
                        tid: 42,
                        comm: Some("worker".to_string()),
                        start_time_jiffies: None,
                        allocated_bytes: 2048,
                        deallocated_bytes: 0,
                    }],
                },
            ],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(s.contains("\"schema_version\":3"));
        assert!(s.contains("\"pid\":777"));
        assert!(s.contains("\"started_at_unix_sec\":1699999999"));
        assert!(s.contains("\"interval_ms\":50"));
        assert!(s.contains("\"interrupted\":false"));
        assert!(s.contains("\"snapshots\":["));
        assert!(s.contains("\"timestamp_unix_sec\":1700000000"));
        assert!(s.contains("\"timestamp_unix_sec\":1700000001"));
        assert!(s.contains("\"allocated_bytes\":1024"));
        assert!(s.contains("\"allocated_bytes\":2048"));
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(
            v.get("timestamp_unix_sec").is_none(),
            "top-level timestamp_unix_sec must not appear on ProbeOutput: {s}",
        );
        assert!(
            v.get("threads").is_none(),
            "top-level threads must not appear on ProbeOutput: {s}",
        );
        assert!(v.get("snapshots").is_some());
        assert!(v.get("started_at_unix_sec").is_some());
        assert!(v.get("interval_ms").is_some());
        assert!(v.get("interrupted").is_some());
    }

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
                elapsed_since_start_ns: 0,
                threads: vec![ThreadResult::Ok {
                    tid: 99,
                    comm: None,
                    start_time_jiffies: None,
                    allocated_bytes: 10,
                    deallocated_bytes: 0,
                }],
            }],
        };
        let s = serde_json::to_string(&out).unwrap();
        assert!(
            !s.contains("\"interval_ms\""),
            "interval_ms must be omitted when None: {s}"
        );
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(v.get("interval_ms").is_none());
        let snaps = v.get("snapshots").and_then(|v| v.as_array()).unwrap();
        assert_eq!(snaps.len(), 1);
    }

    #[test]
    fn exe_identity_stable_within_run() {
        let pid = self_pid();
        let a = ExeIdentity::capture(pid).expect("stat /proc/self/exe");
        let b = ExeIdentity::capture(pid).expect("stat /proc/self/exe");
        assert_eq!(a, b);
    }

    /// `take_snapshot` polls `CLEANUP_REQUESTED` at the TOP of the
    /// per-tid loop; with the flag pre-set the loop breaks
    /// immediately and the returned snapshot carries
    /// `interrupted = true` with an EMPTY `threads` vec.
    ///
    /// Constructs a real `JemallocProbe` against the test binary's
    /// own pid so the engine call would otherwise have a valid
    /// target — the flag pre-set guarantees the loop body never
    /// executes. Skips on environments where attach_jemalloc fails
    /// against self.
    #[test]
    fn take_snapshot_interrupted_flag_truncates_threads_vec() {
        let pid = self_pid();
        let probe = match attach_jemalloc(pid) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "skip: cannot attach to self for take_snapshot interrupt test \
                     (engine: {e})",
                );
                return;
            }
        };

        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        CLEANUP_REQUESTED.store(true, Ordering::SeqCst);

        let tids = vec![1, 2, 3];
        let run_start = std::time::Instant::now();
        let mut tp_cache = std::collections::HashMap::new();
        let (snap, interrupted) = take_snapshot(pid, &probe, &tids, run_start, &mut tp_cache);

        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);

        assert!(interrupted, "pre-set flag must surface interrupted=true");
        assert!(
            snap.threads.is_empty(),
            "truncated snapshot must carry no thread entries; got {} entries",
            snap.threads.len(),
        );
        assert!(
            snap.elapsed_since_start_ns < 1_000_000_000,
            "elapsed_since_start_ns must be populated sub-second; got {} ns",
            snap.elapsed_since_start_ns,
        );
    }

    #[test]
    fn take_snapshot_flag_clear_completes_normally() {
        let pid = self_pid();
        let probe = match attach_jemalloc(pid) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "skip: cannot attach to self for take_snapshot clear-flag test \
                     (engine: {e})",
                );
                return;
            }
        };
        CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
        // Empty tids — loop body never runs; interrupted stays false.
        let tids: Vec<i32> = vec![];
        let run_start = std::time::Instant::now();
        let mut tp_cache = std::collections::HashMap::new();
        let (snap, interrupted) = take_snapshot(pid, &probe, &tids, run_start, &mut tp_cache);
        assert!(
            !interrupted,
            "clear flag + empty tids must not mark interrupted"
        );
        assert!(snap.threads.is_empty());
    }

    #[test]
    fn ensure_exe_identity_unchanged_ok_on_match() {
        let pid = self_pid();
        let baseline = ExeIdentity::capture(pid).expect("stat /proc/self/exe");
        ensure_exe_identity_unchanged(pid, &baseline, "test context")
            .expect("identical baseline must pass");
    }

    #[test]
    fn ensure_exe_identity_unchanged_errs_on_mismatch() {
        let pid = self_pid();
        let baseline = ExeIdentity {
            dev: 0xDEAD_BEEF_DEAD_BEEF,
            ino: 0xCAFE_BABE_CAFE_BABE,
        };
        let err = ensure_exe_identity_unchanged(pid, &baseline, "in unit test")
            .expect_err("synthetic mismatch must produce Err");
        let msg = format!("{err}");
        assert!(
            msg.contains("changed in unit test"),
            "error must carry the context string; got: {msg}",
        );
        assert!(
            msg.contains("dev=0xdeadbeefdeadbeef") || msg.contains("dev=0xdeadbeefDEADBEEF"),
            "error must carry the baseline dev in hex; got: {msg}",
        );
        assert!(
            msg.contains("TLS offsets invalid"),
            "error must carry the downstream consequence; got: {msg}",
        );
    }

    #[test]
    fn ensure_exe_identity_unchanged_error_wraps_into_run_outcome_fatal() {
        let pid = self_pid();
        let baseline = ExeIdentity { dev: 0, ino: 0 };
        let err = ensure_exe_identity_unchanged(pid, &baseline, "between snapshots")
            .expect_err("synthetic mismatch");
        let outcome = RunOutcome::Fatal(FatalError::exe_identity_changed(err));
        match outcome {
            RunOutcome::Fatal(fatal) => {
                assert_eq!(fatal.kind, FatalKind::ExeIdentityChanged);
                let msg = format!("{}", fatal.error);
                assert!(msg.contains("between snapshots"));
            }
            _ => panic!("expected RunOutcome::Fatal"),
        }
    }

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
                elapsed_since_start_ns: 0,
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

    fn minimal_sidecar_json() -> String {
        let sc = ktstr::test_support::SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            scheduler_commit: None,
            project_commit: None,
            payload: None,
            metrics: Vec::new(),
            passed: true,
            skipped: false,
            stats: ktstr::assert::ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: "SpinWait".to_string(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            kernel_commit: None,
            timestamp: String::new(),
            run_id: String::new(),
            host: None,
            cleanup_duration_ms: None,
            run_source: None,
        };
        serde_json::to_string_pretty(&sc).unwrap()
    }

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
                elapsed_since_start_ns: 0,
                threads: vec![ThreadResult::Ok {
                    tid: 42,
                    comm: Some("worker".to_string()),
                    start_time_jiffies: None,
                    allocated_bytes: 1024,
                    deallocated_bytes: 512,
                }],
            }],
        }
    }

    #[test]
    fn sidecar_append_happy_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("t-0000000000000000.ktstr.json");
        std::fs::write(&path, minimal_sidecar_json()).unwrap();

        let out = probe_output_fixture();
        append_probe_output_to_sidecar(&path, &out, 0).expect("append happy path");

        let re_read = std::fs::read_to_string(&path).unwrap();
        let sc: ktstr::test_support::SidecarResult =
            serde_json::from_str(&re_read).expect("sidecar re-parse");
        assert_eq!(sc.test_name, "t");
        assert_eq!(sc.topology, "1n1l1c1t");
        assert_eq!(sc.scheduler, "eevdf");
        assert!(sc.passed);
        assert!(!sc.skipped);
        assert_eq!(sc.metrics.len(), 1, "one appended PayloadMetrics");
        let pm = &sc.metrics[0];
        assert_eq!(pm.exit_code, 0);
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

        let lock_path = path.with_extension("json.lock");
        assert!(
            lock_path.exists(),
            "expected lock file at {}",
            lock_path.display(),
        );

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

    #[test]
    fn sidecar_append_preserves_prepopulated_metrics() {
        use ktstr::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.ktstr.json");

        let mut sc: ktstr::test_support::SidecarResult =
            serde_json::from_str(&minimal_sidecar_json()).unwrap();
        sc.metrics.push(PayloadMetrics {
            payload_index: 0,
            metrics: vec![Metric {
                name: "primary.bogo_ops".to_string(),
                value: 12345.0,
                polarity: Polarity::HigherBetter,
                unit: "ops".to_string(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            }],
            exit_code: 0,
        });
        sc.metrics.push(PayloadMetrics {
            payload_index: 1,
            metrics: vec![Metric {
                name: "secondary.latency_us".to_string(),
                value: 42.0,
                polarity: Polarity::LowerBetter,
                unit: "us".to_string(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            }],
            exit_code: 0,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&sc).unwrap()).unwrap();

        let out = probe_output_fixture();
        append_probe_output_to_sidecar(&path, &out, 0).unwrap();

        let after: ktstr::test_support::SidecarResult =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(after.metrics.len(), 3);
        assert_eq!(after.metrics[0].metrics[0].name, "primary.bogo_ops");
        assert_eq!(after.metrics[0].metrics[0].value, 12345.0);
        assert_eq!(after.metrics[1].metrics[0].name, "secondary.latency_us");
        assert_eq!(after.metrics[1].metrics[0].value, 42.0);
        for m in &after.metrics[2].metrics {
            assert!(
                m.name.starts_with(&format!("{SIDECAR_METRIC_PREFIX}.")),
                "last entry's metric {} missing probe prefix",
                m.name,
            );
        }
    }

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
        assert!(
            msg.contains("--sidecar"),
            "expected flag name in hint, got: {msg}",
        );
    }

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
            "expected parse-failure context, got: {msg}"
        );
        assert!(
            msg.contains("regenerate"),
            "expected pre-1.0 regenerate-policy hint, got: {msg}",
        );
    }

    #[test]
    fn sidecar_append_bails_when_cleanup_requested_preflock() {
        struct FlagGuard;
        impl Drop for FlagGuard {
            fn drop(&mut self) {
                CLEANUP_REQUESTED.store(false, Ordering::SeqCst);
            }
        }
        let _guard = FlagGuard;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pre-flock-bail.ktstr.json");
        std::fs::write(&path, minimal_sidecar_json()).unwrap();
        let out = probe_output_fixture();

        CLEANUP_REQUESTED.store(true, Ordering::SeqCst);
        let err = append_probe_output_to_sidecar(&path, &out, 0)
            .expect_err("flock retry loop must bail when CLEANUP_REQUESTED is set");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("aborted by probe deadline"),
            "expected deadline-abort bail message, got: {msg}",
        );
        assert!(
            msg.contains("flock(LOCK_EX)"),
            "bail message must name the flock phase; got: {msg}",
        );

        let re_read = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            re_read,
            minimal_sidecar_json(),
            "sidecar contents must be unchanged when the flock gate fires",
        );
    }

    #[test]
    fn apply_probe_metric_hints_classifies_byte_counters() {
        use ktstr::test_support::{Metric, MetricSource, MetricStream, Polarity};
        let mut alloc = Metric {
            name: "jemalloc_probe.snapshots.0.threads.0.allocated_bytes".to_string(),
            value: 1024.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
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
            stream: MetricStream::Stdout,
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
            stream: MetricStream::Stdout,
        };
        apply_probe_metric_hints(&mut tid);
        assert!(matches!(tid.polarity, Polarity::Unknown));
        assert_eq!(tid.unit, "");

        let mut extra = Metric {
            name: "jemalloc_probe.snapshots.0.threads.0.allocated_bytes_extra".to_string(),
            value: 999.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
        };
        apply_probe_metric_hints(&mut extra);
        assert!(matches!(extra.polarity, Polarity::Unknown));
        assert_eq!(extra.unit, "");
        let mut dextra = Metric {
            name: "jemalloc_probe.deallocated_bytes_something".to_string(),
            value: 0.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
        };
        apply_probe_metric_hints(&mut dextra);
        assert!(matches!(dextra.polarity, Polarity::Unknown));
        assert_eq!(dextra.unit, "");
    }

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
                elapsed_since_start_ns: 0,
                threads: vec![
                    ThreadResult::Ok {
                        tid: 42,
                        comm: Some("ok-worker".to_string()),
                        start_time_jiffies: None,
                        allocated_bytes: 2048,
                        deallocated_bytes: 128,
                    },
                    ThreadResult::Err {
                        tid: 99,
                        comm: Some("err-worker".to_string()),
                        start_time_jiffies: None,
                        error: "ptrace(PTRACE_SEIZE): ESRCH".to_string(),
                        error_kind: ThreadErrorKind::PtraceSeize,
                    },
                ],
            }],
        };
        let pm = synthesize_payload_metrics(&out, 7, 0).expect("synthesize");
        assert_eq!(pm.exit_code, 7);
        assert_eq!(pm.payload_index, 0);

        for m in &pm.metrics {
            assert!(
                m.name.starts_with(&format!("{SIDECAR_METRIC_PREFIX}.")),
                "metric {} missing probe prefix",
                m.name,
            );
        }
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
        let tid_values: Vec<f64> = pm
            .metrics
            .iter()
            .filter(|m| m.name.ends_with(".tid"))
            .map(|m| m.value)
            .collect();
        assert!(tid_values.contains(&42.0));
        assert!(tid_values.contains(&99.0));
        let alloc_count = pm
            .metrics
            .iter()
            .filter(|m| m.name.ends_with(".allocated_bytes"))
            .count();
        assert_eq!(alloc_count, 1);
    }

    /// Pin the wire-contract strings emitted after
    /// `ktstr-probe-fatal:` on stderr.
    #[test]
    fn fatal_kind_tag_strings_pinned() {
        assert_eq!(FatalKind::PidMissing.tag(), "pid-missing");
        assert_eq!(FatalKind::ExeIdentityChanged.tag(), "exe-identity-changed");
        assert_eq!(FatalKind::JemallocNotFound.tag(), "jemalloc-not-found");
        assert_eq!(FatalKind::JemallocInDso.tag(), "jemalloc-in-dso");
        assert_eq!(FatalKind::ReadlinkFailure.tag(), "readlink-failure");
        assert_eq!(FatalKind::MapsReadFailure.tag(), "maps-read-failure");
        assert_eq!(FatalKind::DwarfParseFailure.tag(), "dwarf-parse-failure");
        assert_eq!(FatalKind::ArchMismatch.tag(), "arch-mismatch");
        assert_eq!(FatalKind::SelfProbeRejected.tag(), "self-probe-rejected");
        assert_eq!(
            FatalKind::TidEnumerationFailure.tag(),
            "tid-enumeration-failure",
        );
        assert_eq!(FatalKind::Other.tag(), "other");
    }

    /// Compile-time exhaustiveness guard for [`FatalKind::tag`].
    #[test]
    fn fatal_kind_exhaustiveness_compile_time_guard() {
        use strum::IntoEnumIterator;

        let mut count = 0;
        for kind in FatalKind::iter() {
            let tag = kind.tag();
            assert!(!tag.is_empty(), "FatalKind::{kind:?}.tag() returned empty");
            assert!(
                tag.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "FatalKind::{kind:?}.tag() = {tag:?} must be lowercase-kebab-case",
            );
            match kind {
                FatalKind::PidMissing
                | FatalKind::ExeIdentityChanged
                | FatalKind::JemallocNotFound
                | FatalKind::JemallocInDso
                | FatalKind::ReadlinkFailure
                | FatalKind::MapsReadFailure
                | FatalKind::DwarfParseFailure
                | FatalKind::ArchMismatch
                | FatalKind::SelfProbeRejected
                | FatalKind::TidEnumerationFailure
                | FatalKind::Other => {}
            }
            count += 1;
        }
        assert_eq!(
            count, 11,
            "FatalKind::iter() must yield exactly the eleven variants pinned in \
             `fatal_kind_tag_strings_pinned`",
        );
    }

    // ---- start_time parser ----

    #[test]
    fn start_time_parser_handles_parens_in_comm() {
        let mut s = String::from("1234 (a)b(c)) S");
        for i in 0..18 {
            s.push(' ');
            s.push_str(&i.to_string());
        }
        s.push_str(" 987654321 rest of line ignored");
        assert_eq!(parse_start_time_from_stat(&s), Some(987654321));
    }

    #[test]
    fn start_time_parser_empty_input_returns_none() {
        assert_eq!(parse_start_time_from_stat(""), None);
    }

    #[test]
    fn start_time_parser_no_close_paren_returns_none() {
        assert_eq!(
            parse_start_time_from_stat("1234 comm_without_parens S 0 0 0 0"),
            None,
        );
    }

    #[test]
    fn start_time_parser_nothing_after_close_paren_returns_none() {
        assert_eq!(parse_start_time_from_stat("1234 (comm)"), None);
    }

    #[test]
    fn start_time_parser_too_few_fields_returns_none() {
        assert_eq!(
            parse_start_time_from_stat("1234 (comm) S 1 2 3 4 5 6 7 8 9"),
            None,
        );
    }

    #[test]
    fn start_time_parser_non_numeric_field_22_returns_none() {
        let mut s = String::from("1234 (comm) S");
        for i in 0..18 {
            s.push(' ');
            s.push_str(&i.to_string());
        }
        s.push_str(" not_a_number trailing garbage");
        assert_eq!(parse_start_time_from_stat(&s), None);
    }
}
