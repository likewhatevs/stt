//! Per-thread host-state profiler data model + capture layer.
//!
//! [`HostStateSnapshot`] is the serialized container for a single
//! host-wide per-thread profile. Capture produces one via the
//! `ktstr host-state -o snapshot.hst.zst` subcommand; comparison
//! reads two and joins them on `(pcomm, comm)`.
//!
//! Every field is cumulative-from-birth so probe timing does not
//! alter the output: the design principle is that a thread sampled
//! twice at different wall-clock instants produces the same numbers
//! so long as its cumulative counters have not rolled over. Metrics
//! that reset on attachment (perf_event_open counters, etc.) are
//! intentionally absent from this capture layer.
//!
//! # Capture model
//!
//! [`capture`] walks `/proc` for every live tgid, enumerates its
//! threads, and populates each [`ThreadState`] from a handful of
//! procfs sources: `stat`, `schedstat`, `status`, `io`, `sched`,
//! `comm`, `cgroup`. Each internal reader returns `Option`
//! (graceful on missing/unreadable — a kernel without
//! `CONFIG_SCHEDSTATS` or `CONFIG_SCHED_DEBUG` yields `None` from
//! the affected reader without failing the rest of the thread).
//! The assembled [`ThreadState`] treats `None` as "absent at
//! capture" via the field type — counters collapse to `0`,
//! identity strings collapse to empty, affinity collapses to an
//! empty vec. A missing reading is therefore indistinguishable
//! from a genuine zero in the serialized output; the capture
//! contract is best-effort, never-fail-the-snapshot. Tests that
//! need stronger guarantees inspect the underlying readers
//! directly (they remain `Option`-shaped, unit-tested in this
//! module).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Top-level serialized artifact produced by `ktstr host-state`.
///
/// The file layout on disk is zstd-compressed JSON of this struct.
/// Extension `.hst.zst` is conventional; nothing in the loader
/// depends on the extension beyond being passed a path that
/// resolves to a readable file.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HostStateSnapshot {
    /// Wall-clock time at capture, nanoseconds since the Unix
    /// epoch. Useful as a tie-breaker when comparing two snapshots
    /// that originate from the same host — the newer one is
    /// candidate by default — but carries no load-bearing role in
    /// the join key.
    pub captured_at_unix_ns: u64,

    /// Host context snapshot (kernel, CPU, memory, tunables).
    /// Optional because older tools or synthetic fixtures may
    /// omit it; comparison degrades to a "host context unavailable"
    /// line rather than failing the whole compare when either
    /// side is missing.
    pub host: Option<crate::host_context::HostContext>,

    /// One entry per observed thread on the host at capture time.
    /// Order is not load-bearing; the comparison pipeline groups
    /// by `(pcomm, comm)` / `cgroup` / `comm` depending on
    /// `--group-by`.
    pub threads: Vec<ThreadState>,

    /// Enrichment metadata for every cgroup that at least one
    /// sampled thread resides in. Keyed by the cgroup path
    /// relative to the v2 mount (e.g.
    /// `/kubepods/burstable/pod-<id>/container`). Populated from
    /// the cgroup filesystem, not the per-thread sample, because
    /// cpu.stat / memory.current describe the cgroup's aggregate
    /// state, not per-thread contribution.
    pub cgroup_stats: BTreeMap<String, CgroupStats>,
}

/// Per-thread cumulative resource profile.
///
/// Populated by the capture layer from `/proc/tid/{sched,status,
/// io,stat,comm,cgroup}`, `sched_getaffinity`, and (for jemalloc-
/// linked processes only) the per-thread-destructor TSD cache.
/// All numeric fields are cumulative since thread birth so the
/// value is insensitive to probe-attach latency.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ThreadState {
    // -- identity --
    /// Kernel task id. Ephemeral across runs; not used for join.
    pub tid: u32,
    /// Thread group id (process id). Ephemeral across runs.
    pub tgid: u32,
    /// Process name, read from `/proc/<tgid>/comm`. Stable across
    /// runs on the same build; part of the comparison join key.
    pub pcomm: String,
    /// Thread name, read from `/proc/<tid>/comm`. Stable when the
    /// runtime assigns deterministic names (worker pools, async
    /// runtimes); part of the comparison join key.
    pub comm: String,
    /// Cgroup v2 path.
    ///
    /// # Namespace semantics
    ///
    /// The path is read verbatim from `/proc/<tid>/cgroup` and
    /// is therefore relative to the CGROUP NAMESPACE ROOT the
    /// capturing process sees — NOT relative to the
    /// system-global v2 mount root. A process outside the
    /// capturing namespace would see the same cgroup under a
    /// different path (prefixed with the namespace-root ancestors
    /// the inner view hides); a process inside a nested cgroup
    /// namespace sees a truncated path. Cross-namespace
    /// comparison requires external canonicalization (e.g.
    /// resolving via `cgroup.procs` inode chains or walking
    /// `/proc/<tid>/ns/cgroup` to the common root) — the
    /// capture layer deliberately does NOT attempt this because
    /// the resolution depends on capture-site privilege and
    /// namespace visibility that varies per caller.
    ///
    /// Kept as `cgroup` (not renamed to `cgroup_ns_relative`)
    /// for consistency with [`GroupBy::Cgroup`],
    /// `cgroup_flatten`, `cgroup_stats`, and every CLI flag
    /// that threads the same concept through the comparison
    /// layer; a rename would cascade through every pinned
    /// string in the compare pipeline without improving the
    /// semantic guarantee. This doc is the canonical
    /// documentation of the namespace-relative contract.
    ///
    /// Enrichment for grouping and filtering views; not a join
    /// key.
    pub cgroup: String,
    /// `/proc/<tid>/stat` field 22 (`start_time`) in USER_HZ
    /// clock ticks since system boot. The kernel exports this
    /// field in USER_HZ units (defined in
    /// `include/asm-generic/param.h` as `USER_HZ == 100` on
    /// every architecture the capture layer targets — x86_64
    /// and aarch64) — NOT raw internal jiffies, which scale
    /// with CONFIG_HZ. The name corrects a prior misnaming:
    /// cross-host comparison between x86_64 and aarch64 IS
    /// meaningful because USER_HZ is the same 100 on both, so
    /// a diff between two hosts on different CONFIG_HZ
    /// settings still compares correctly. Seconds-since-boot
    /// is simply `start_time_clock_ticks / 100` on those
    /// architectures. Other in-tree architectures carry
    /// different USER_HZ (alpha defines 1024, for instance);
    /// a future port must either restate the divisor or
    /// normalise at capture time. `fs/proc/array.c::do_task_stat`
    /// is where the kernel writes the field to procfs.
    pub start_time_clock_ticks: u64,
    /// Scheduling policy (SCHED_OTHER, SCHED_FIFO, SCHED_RR,
    /// SCHED_BATCH, SCHED_IDLE, SCHED_DEADLINE, SCHED_EXT). Stored
    /// as the canonical name string rather than the kernel
    /// integer so comparison output is human-readable without a
    /// reverse-lookup table.
    pub policy: String,
    /// Nice value in the standard [-20, 19] range. Signed i32
    /// because the range includes negative values and
    /// [`parse_stat`] extracts the field via `get_i32` on
    /// procfs's decimal text — the type matches the extraction
    /// path and the kernel-visible range without coercion.
    pub nice: i32,
    /// Allowed CPU set from `sched_getaffinity`. Sorted ascending.
    /// Comparison aggregates via union across the group and
    /// renders as "N cpus (range)" or "mixed" for heterogeneous
    /// sets — see [`crate::host_state_compare::AffinitySummary`].
    pub cpu_affinity: Vec<u32>,

    // -- scheduling (cumulative; /proc/tid/sched, needs CONFIG_SCHED_DEBUG) --
    pub run_time_ns: u64,
    pub wait_time_ns: u64,
    pub timeslices: u64,
    pub voluntary_csw: u64,
    pub nonvoluntary_csw: u64,
    pub nr_wakeups: u64,
    pub nr_wakeups_local: u64,
    pub nr_wakeups_remote: u64,
    pub nr_wakeups_sync: u64,
    pub nr_wakeups_migrate: u64,
    pub nr_wakeups_idle: u64,
    pub nr_migrations: u64,
    pub wait_sum: u64,
    pub wait_count: u64,
    /// Total nanoseconds the task slept (voluntary block in
    /// `schedule()` — sleep syscalls, futex wait, etc.). Populated
    /// from `/proc/<tid>/sched`'s `sum_sleep_runtime` key; earlier
    /// drafts of this field misnamed the kernel key as `sleep_sum`
    /// and therefore never populated. There is no `sleep_count`
    /// counterpart: the kernel does not emit one (the scheduler
    /// records the aggregate runtime but not the sleep-event count
    /// separately from `nr_wakeups`, which already covers the
    /// wake-side tally).
    pub sleep_sum: u64,
    /// Total time blocked in the scheduler — every path that
    /// puts the task into `TASK_UNINTERRUPTIBLE` contributes:
    /// swap-in, page-fault resolution, disk I/O, plus
    /// mutex/rwsem/completion waits inside kernel code that
    /// hold the task off the runqueue. `block_sum - iowait_sum`
    /// is therefore an UPPER BOUND on non-iowait
    /// involuntary-block time — swap/zswap decompression
    /// contributes, but so do the lock-family waits, so the
    /// delta cannot be read as swap latency without further
    /// attribution.
    pub block_sum: u64,
    pub block_count: u64,
    /// Total time in I/O wait specifically (subset of
    /// `block_sum`). Distinguishes disk-backed I/O delay from
    /// the full involuntary-block total — callers that want
    /// disk latency alone read this field, callers that want
    /// every blocked window read `block_sum`.
    pub iowait_sum: u64,
    pub iowait_count: u64,

    // -- memory (jemalloc TSD; /proc/tid/stat fields 10, 12) --
    /// Bytes allocated by this thread's lifetime, summed from
    /// jemalloc's per-thread-destructor cache. Zero for
    /// processes not linked against jemalloc — the capture
    /// layer cannot observe glibc's opaque arena counters, and
    /// a missing value is indistinguishable from a real zero
    /// rather than a capture failure.
    pub allocated_bytes: u64,
    pub deallocated_bytes: u64,
    /// Minor faults (no disk I/O). `/proc/tid/stat` field 10.
    pub minflt: u64,
    /// Major faults (backed by disk). `/proc/tid/stat` field 12.
    pub majflt: u64,

    // -- I/O (/proc/tid/io, CONFIG_TASK_IO_ACCOUNTING) --
    pub rchar: u64,
    pub wchar: u64,
    pub syscr: u64,
    pub syscw: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
}

/// Per-cgroup enrichment counters attached to [`HostStateSnapshot`].
///
/// Populated from the cgroup v2 filesystem at capture time:
/// `cpu.stat` exposes `usage_usec`, `nr_throttled`,
/// `throttled_usec`; `memory.current` is the instantaneous RSS
/// of the cgroup. These are aggregate-over-the-cgroup values —
/// NOT summable from per-thread data — so the capture layer
/// reads them directly from the cgroupfs rather than deriving.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CgroupStats {
    pub cpu_usage_usec: u64,
    pub nr_throttled: u64,
    pub throttled_usec: u64,
    pub memory_current: u64,
}

impl HostStateSnapshot {
    /// Load a snapshot from a zstd-compressed JSON file.
    ///
    /// Errors propagate via [`anyhow`] with the source path in the
    /// context chain so a malformed file surfaces an actionable
    /// message rather than a generic deserialize error. The loader
    /// does not validate that `threads` is non-empty — an empty
    /// snapshot is a legitimate edge case (host idle, capture
    /// filter excluded every thread) and the comparison engine
    /// handles it by emitting an empty diff.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let bytes = std::fs::read(path)
            .with_context(|| format!("read host-state snapshot from {}", path.display()))?;
        let json = zstd::decode_all(bytes.as_slice()).with_context(|| {
            format!("zstd decompress host-state snapshot {}", path.display())
        })?;
        let snap: HostStateSnapshot = serde_json::from_slice(&json).with_context(|| {
            format!(
                "parse host-state snapshot JSON from {} (did the capture format change?)",
                path.display(),
            )
        })?;
        Ok(snap)
    }

    /// Write a snapshot as zstd-compressed JSON.
    ///
    /// Used by the capture layer; exposed from this module so that
    /// both compare-side tests and the capture binary share one
    /// on-disk shape. Compression level `3` mirrors the ktstr
    /// remote-cache convention — adequate ratio at fast speed —
    /// and is not tunable because host-state captures are small
    /// enough that further compression produces diminishing
    /// returns on I/O.
    pub fn write(&self, path: &std::path::Path) -> anyhow::Result<()> {
        use anyhow::Context;
        let json = serde_json::to_vec(self).context("serialize host-state snapshot to JSON")?;
        let compressed = zstd::encode_all(json.as_slice(), 3)
            .context("zstd compress host-state snapshot")?;
        std::fs::write(path, compressed)
            .with_context(|| format!("write host-state snapshot to {}", path.display()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------
// Capture layer: procfs readers + host walk.
// ---------------------------------------------------------------

/// Canonical file extension for a serialized snapshot.
pub const SNAPSHOT_EXTENSION: &str = "hst.zst";

/// Default procfs root on Linux. The `_at` readers accept any
/// `&Path` so tests stage a synthetic tree under a tempdir; the
/// public readers delegate to those with this default.
pub const DEFAULT_PROC_ROOT: &str = "/proc";

/// Default cgroup v2 mount point.
pub const DEFAULT_CGROUP_ROOT: &str = "/sys/fs/cgroup";

fn task_file(proc_root: &Path, tgid: i32, tid: i32, leaf: &str) -> PathBuf {
    proc_root
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join(leaf)
}

fn proc_file(proc_root: &Path, tgid: i32, leaf: &str) -> PathBuf {
    proc_root.join(tgid.to_string()).join(leaf)
}

/// Map a numeric scheduling policy (as it appears in
/// `/proc/<tgid>/task/<tid>/stat` field 41) to the canonical
/// kernel identifier string. Unknown integers render as
/// `"SCHED_UNKNOWN(<n>)"` rather than dropping the value so
/// diff output still surfaces a novel policy from a future
/// kernel.
pub fn policy_name(policy: i32) -> String {
    match policy {
        libc::SCHED_OTHER => "SCHED_OTHER".to_string(),
        libc::SCHED_FIFO => "SCHED_FIFO".to_string(),
        libc::SCHED_RR => "SCHED_RR".to_string(),
        libc::SCHED_BATCH => "SCHED_BATCH".to_string(),
        libc::SCHED_IDLE => "SCHED_IDLE".to_string(),
        // `SCHED_DEADLINE` = 6, `SCHED_EXT` = 7 — neither is
        // exposed by the libc crate as of this writing; use the
        // kernel-canonical numeric codes.
        6 => "SCHED_DEADLINE".to_string(),
        7 => "SCHED_EXT".to_string(),
        other => format!("SCHED_UNKNOWN({other})"),
    }
}

/// Enumerate every numeric directory under the procfs root
/// (live tgids). Returns sorted ids so snapshot ordering is
/// deterministic. Empty vec on read failure.
pub fn iter_tgids_at(proc_root: &Path) -> Vec<i32> {
    let Ok(entries) = fs::read_dir(proc_root) else {
        return Vec::new();
    };
    let mut tgids: Vec<i32> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()))
        .filter(|&p| p > 0)
        .collect();
    tgids.sort_unstable();
    tgids
}

/// Enumerate tids under `<proc_root>/<tgid>/task`. Empty vec on
/// read failure (process exited between enumeration and this
/// call).
pub fn iter_task_ids_at(proc_root: &Path, tgid: i32) -> Vec<i32> {
    let path = proc_root.join(tgid.to_string()).join("task");
    let Ok(entries) = fs::read_dir(&path) else {
        return Vec::new();
    };
    let mut tids: Vec<i32> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()))
        .filter(|&t| t > 0)
        .collect();
    tids.sort_unstable();
    tids
}

pub fn iter_tgids() -> Vec<i32> {
    iter_tgids_at(Path::new(DEFAULT_PROC_ROOT))
}

pub fn iter_task_ids(tgid: i32) -> Vec<i32> {
    iter_task_ids_at(Path::new(DEFAULT_PROC_ROOT), tgid)
}

/// Read `<proc_root>/<tgid>/comm` trimmed. `None` on read
/// failure or empty content.
pub fn read_process_comm_at(proc_root: &Path, tgid: i32) -> Option<String> {
    let raw = fs::read_to_string(proc_file(proc_root, tgid, "comm")).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Read `<proc_root>/<tgid>/task/<tid>/comm` trimmed. `None`
/// on read failure or empty content.
pub fn read_thread_comm_at(proc_root: &Path, tgid: i32, tid: i32) -> Option<String> {
    let raw = fs::read_to_string(task_file(proc_root, tgid, tid, "comm")).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub fn read_process_comm(tgid: i32) -> Option<String> {
    read_process_comm_at(Path::new(DEFAULT_PROC_ROOT), tgid)
}

pub fn read_thread_comm(tgid: i32, tid: i32) -> Option<String> {
    read_thread_comm_at(Path::new(DEFAULT_PROC_ROOT), tgid, tid)
}

/// Selected fields parsed out of `/proc/<tgid>/task/<tid>/stat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StatFields {
    minflt: Option<u64>,
    majflt: Option<u64>,
    nice: Option<i32>,
    start_time_clock_ticks: Option<u64>,
    policy: Option<i32>,
}

/// Pure parser for `/proc/<tgid>/task/<tid>/stat`. Per `proc(5)`,
/// field 2 (`comm`) is wrapped in parens and may contain
/// whitespace or `)`; every later field is indexed relative to
/// the LAST `)` in the line. Tail offsets (0-indexed from the
/// token past the final `)`):
///
/// | field | name      | tail index |
/// |-------|-----------|------------|
/// | 10    | minflt    | 7          |
/// | 12    | majflt    | 9          |
/// | 19    | nice      | 16         |
/// | 22    | starttime | 19         |
/// | 41    | policy    | 38         |
///
/// Missing fields return `None` individually so a short line
/// (tid exited mid-read, stat truncated) degrades gracefully.
fn parse_stat(raw: &str) -> StatFields {
    let Some(line) = raw.lines().next() else {
        return StatFields::default();
    };
    let Some(last_close) = line.rfind(')') else {
        return StatFields::default();
    };
    let Some(tail) = line.get(last_close + 1..) else {
        return StatFields::default();
    };
    let parts: Vec<&str> = tail.split_ascii_whitespace().collect();
    let get_u64 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<u64>().ok());
    let get_i32 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<i32>().ok());
    StatFields {
        minflt: get_u64(7),
        majflt: get_u64(9),
        nice: get_i32(16),
        start_time_clock_ticks: get_u64(19),
        policy: get_i32(38),
    }
}

fn read_stat_at(proc_root: &Path, tgid: i32, tid: i32) -> StatFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "stat")) {
        Ok(raw) => parse_stat(&raw),
        Err(_) => StatFields::default(),
    }
}

/// Parse the three leading u64 fields from a single-line
/// `/proc/<tgid>/task/<tid>/schedstat` — `(run_time_ns,
/// wait_time_ns, timeslices)`. Missing fields drop individually.
fn parse_schedstat(raw: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    let Some(line) = raw.lines().next() else {
        return (None, None, None);
    };
    let mut parts = line.split_ascii_whitespace();
    let run = parts.next().and_then(|s| s.parse::<u64>().ok());
    let wait = parts.next().and_then(|s| s.parse::<u64>().ok());
    let slices = parts.next().and_then(|s| s.parse::<u64>().ok());
    (run, wait, slices)
}

/// Read `<proc_root>/<tgid>/task/<tid>/schedstat`. Three-tuple
/// of `Option<u64>` — kernel without `CONFIG_SCHEDSTATS` yields
/// all-`None`.
pub fn read_schedstat_at(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
) -> (Option<u64>, Option<u64>, Option<u64>) {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "schedstat")) {
        Ok(raw) => parse_schedstat(&raw),
        Err(_) => (None, None, None),
    }
}

pub fn read_schedstat(tgid: i32, tid: i32) -> (Option<u64>, Option<u64>, Option<u64>) {
    read_schedstat_at(Path::new(DEFAULT_PROC_ROOT), tgid, tid)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct IoFields {
    rchar: Option<u64>,
    wchar: Option<u64>,
    syscr: Option<u64>,
    syscw: Option<u64>,
    read_bytes: Option<u64>,
    write_bytes: Option<u64>,
}

/// Parse `/proc/<tgid>/task/<tid>/io` (line-oriented
/// `key: value` format).
fn parse_io(raw: &str) -> IoFields {
    let mut out = IoFields::default();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let parsed = value.trim().parse::<u64>().ok();
        match key.trim() {
            "rchar" => out.rchar = parsed,
            "wchar" => out.wchar = parsed,
            "syscr" => out.syscr = parsed,
            "syscw" => out.syscw = parsed,
            "read_bytes" => out.read_bytes = parsed,
            "write_bytes" => out.write_bytes = parsed,
            _ => {}
        }
    }
    out
}

fn read_io_at(proc_root: &Path, tgid: i32, tid: i32) -> IoFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "io")) {
        Ok(raw) => parse_io(&raw),
        Err(_) => IoFields::default(),
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct StatusFields {
    voluntary_csw: Option<u64>,
    nonvoluntary_csw: Option<u64>,
    /// `Cpus_allowed_list:` as a parsed sorted vec. Kept separate
    /// from the `sched_getaffinity` reader because status-file
    /// reads attribute to the target task without a syscall
    /// round-trip — useful when the caller cannot hold a pid
    /// long enough for the syscall without a race.
    cpus_allowed: Option<Vec<u32>>,
}

fn parse_status(raw: &str) -> StatusFields {
    let mut out = StatusFields::default();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "voluntary_ctxt_switches" => {
                out.voluntary_csw = value.parse::<u64>().ok();
            }
            "nonvoluntary_ctxt_switches" => {
                out.nonvoluntary_csw = value.parse::<u64>().ok();
            }
            "Cpus_allowed_list" => {
                out.cpus_allowed = parse_cpu_list(value);
            }
            _ => {}
        }
    }
    out
}

fn read_status_at(proc_root: &Path, tgid: i32, tid: i32) -> StatusFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "status")) {
        Ok(raw) => parse_status(&raw),
        Err(_) => StatusFields::default(),
    }
}

/// Parse a cpulist string of the form `"0-3,5,7-9"` into a
/// sorted deduped vec of CPU ids. `None` on empty input or any
/// malformed token (partial results are rejected so the caller
/// can distinguish "no data" from "data but garbled").
///
/// # Range expansion cap
///
/// A single `lo-hi` token that would expand to more than
/// [`MAX_CPU_RANGE_EXPANSION`] (65,536) CPUs is rejected as
/// malformed. Without this gate a hostile or corrupted
/// `Cpus_allowed_list:` value like `0-4294967295` would allocate
/// 16 GiB for the expansion vec and crash the capture (or OOM
/// the process). The cap is far above any realistic
/// `CONFIG_NR_CPUS` (current Linux defaults top out at a few
/// thousand; even `NR_CPUS=8192` builds stay inside this
/// bound), so legitimate input is never rejected. See
/// [`parse_cpu_list_rejects_huge_range`] for the regression pin.
pub fn parse_cpu_list(s: &str) -> Option<Vec<u32>> {
    /// Upper bound on the number of CPUs a single `lo-hi` token
    /// can expand to. 64 Ki — orders of magnitude above any
    /// in-production `NR_CPUS` — leaves headroom for future
    /// large-NUMA hosts while capping the worst-case allocation
    /// at 256 KiB (64 Ki × u32 = 256 KiB).
    const MAX_CPU_RANGE_EXPANSION: u64 = 65_536;

    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut out: Vec<u32> = Vec::new();
    for token in s.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = token.split_once('-') {
            let lo: u32 = lo.parse().ok()?;
            let hi: u32 = hi.parse().ok()?;
            if hi < lo {
                return None;
            }
            // Guard against hostile / corrupt range expansions.
            // Use u64 arithmetic so the `hi - lo + 1` compute
            // cannot overflow even at u32::MAX. Reject rather
            // than clamp so the caller's "no data vs data but
            // garbled" distinction stays intact.
            let span = (hi as u64) - (lo as u64) + 1;
            if span > MAX_CPU_RANGE_EXPANSION {
                return None;
            }
            for c in lo..=hi {
                out.push(c);
            }
        } else {
            out.push(token.parse::<u32>().ok()?);
        }
    }
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// Read the effective CPU affinity for a task via the
/// `sched_getaffinity(2)` syscall. Kernel accepts any pid/tid in
/// the caller's namespace; root or same-uid required per the
/// kernel's ptrace-access check. Returns sorted CPU ids.
/// `None` on syscall failure (EPERM, ESRCH) or when the kernel's
/// mask exceeds [`AFFINITY_MAX_BITS`] (hosts beyond 262144 CPUs).
///
/// # Dynamic buffer sizing
///
/// The kernel's `SYSCALL_DEFINE3(sched_getaffinity)`
/// (`kernel/sched/syscalls.c`) rejects a caller buffer shorter
/// than `nr_cpu_ids / BITS_PER_BYTE` with `EINVAL`. The kernel
/// supports `CONFIG_NR_CPUS` values up to 8192 on x86_64 default
/// and higher on custom builds (large NUMA / partitioning
/// hardware). libc's fixed [`libc::cpu_set_t`] is only 1024 bits
/// wide, so calling `sched_getaffinity` with
/// `size_of::<cpu_set_t>()` against a `CONFIG_NR_CPUS > 1024`
/// kernel fails EINVAL even when the caller has legitimate
/// access.
///
/// This helper avoids the cap by allocating a dynamically-sized
/// `Vec<c_ulong>` (an array of kernel `unsigned long` — the
/// wire format the syscall expects, aligned and byte-length a
/// multiple of `sizeof(unsigned long)` per the kernel's second
/// validation). On EINVAL the buffer doubles and the call
/// retries, capped at [`AFFINITY_MAX_BITS`] = 262144 (32 KiB of
/// mask data — covers every real-world `CONFIG_NR_CPUS` setting
/// and bounds the worst-case allocation).
///
/// # Error-class handling
///
/// - `EINVAL` → buffer too small. Double and retry until the
///   ceiling is reached, then surface None.
/// - `EPERM` / `ESRCH` → real access / process-identity failures.
///   Return None so the caller falls back to the procfs
///   `Cpus_allowed_list:` path, which bypasses the permission
///   check (reading `/proc/<tid>/status` only requires directory
///   traversal permission, not `PTRACE_MODE_READ`).
/// - Any other error → return None. The procfs fallback will
///   produce the correct value or its own None.
///
/// Without this split, the previous implementation collapsed
/// every error to None indistinguishably — EINVAL on a
/// >1024-CPU host was treated the same as EPERM, and every
/// caller had to rely on the procfs fallback for correctness,
/// making the syscall path effectively useless on the very
/// hosts where affinity data matters most (1000-plus-CPU NUMA
/// boxes).
pub fn read_affinity(tid: i32) -> Option<Vec<u32>> {
    let mut bits = AFFINITY_INITIAL_BITS;
    loop {
        let mut buffer = affinity_zeroed_buffer(bits);
        let bytes = std::mem::size_of_val(buffer.as_slice());
        // SAFETY: `buffer.as_mut_ptr()` produces a live pointer
        // valid for `bytes` writes; the kernel writes at most
        // `min(bytes, cpumask_size)` and returns the actual byte
        // count. `bits` is always a multiple of
        // `c_ulong::BITS`, so `bytes` satisfies the kernel's
        // alignment validation (`len & (sizeof(unsigned long)-1)
        // == 0`).
        let ret = unsafe {
            libc::syscall(
                libc::SYS_sched_getaffinity,
                tid as libc::pid_t,
                bytes,
                buffer.as_mut_ptr(),
            )
        };
        if ret >= 0 {
            // ret carries the actual byte count the kernel
            // wrote. Bits beyond `ret * 8` were not touched and
            // stay at the zero-init value above — safe to
            // iterate the full buffer, but tightening the bound
            // avoids wasted work on small masks inside a large
            // buffer.
            let written_bytes = ret as usize;
            return extract_cpus_from_mask(&buffer, written_bytes);
        }
        // Error path: classify via errno.
        let errno = std::io::Error::last_os_error().raw_os_error();
        // Only EINVAL warrants a retry — it signals "buffer too
        // small" under the kernel's
        // `(len * BITS_PER_BYTE) < nr_cpu_ids` check. Every other
        // error (EPERM permission denied, ESRCH process gone,
        // EFAULT bad pointer, etc.) is terminal.
        if errno != Some(libc::EINVAL) {
            return None;
        }
        let Some(next) = affinity_next_bits(bits) else {
            // Ceiling reached without success — the host claims
            // more CPUs than the helper is willing to allocate
            // for. Surface None so the caller falls back to the
            // procfs string form, which has no bit-count cap.
            return None;
        };
        bits = next;
    }
}

/// Initial number of CPU bits the affinity buffer starts at.
/// 8192 matches the x86_64 default `CONFIG_NR_CPUS`, so the
/// overwhelming majority of hosts resolve on the first syscall.
pub(crate) const AFFINITY_INITIAL_BITS: usize = 8192;

/// Maximum number of CPU bits [`read_affinity`] is willing to
/// allocate for. 262144 bits = 32 KiB of mask data, well above
/// the largest in-production `CONFIG_NR_CPUS` this project
/// targets. Capping bounds the worst-case allocation and
/// bounds the retry loop's iteration count
/// (`log2(AFFINITY_MAX_BITS / AFFINITY_INITIAL_BITS)` = 5
/// doublings).
pub(crate) const AFFINITY_MAX_BITS: usize = 262144;

/// Given the current buffer size in bits, return the size for
/// the next retry attempt — double the current size, rejecting
/// any step that would exceed [`AFFINITY_MAX_BITS`]. Returns
/// `None` when the ceiling has been reached and no further
/// retry is allowed.
///
/// Extracted from [`read_affinity`] so the loop-termination
/// policy is unit-testable without syscall dispatch.
pub(crate) fn affinity_next_bits(current_bits: usize) -> Option<usize> {
    let doubled = current_bits.checked_mul(2)?;
    if doubled > AFFINITY_MAX_BITS {
        None
    } else {
        Some(doubled)
    }
}

/// Allocate a zeroed buffer of `c_ulong` words sized to hold
/// `bits` CPU-mask bits. The kernel's
/// `sys_sched_getaffinity` rejects any `len & (sizeof(unsigned
/// long)-1) != 0`, so the buffer is allocated in whole-word
/// units.
///
/// Extracted so [`read_affinity`]'s reset-on-retry contract is
/// visible (a fresh zeroed buffer per attempt prevents stale
/// bits from a truncated earlier read leaking into the current
/// attempt's iteration).
fn affinity_zeroed_buffer(bits: usize) -> Vec<libc::c_ulong> {
    let word_bits = libc::c_ulong::BITS as usize;
    let words = bits.div_ceil(word_bits);
    vec![0 as libc::c_ulong; words]
}

/// Walk a successfully-filled cpu-mask buffer and return the
/// sorted list of set CPU ids, or `None` when no bits were set
/// (the kernel writes a mask with at least one bit for any
/// task that was dispatchable at all; an all-zero mask would
/// imply the task has been taken off every CPU, which the
/// kernel does not expose as a valid affinity — surface None
/// rather than `Some(vec![])` so downstream callers can tell
/// "no data" from "legitimately empty mask").
///
/// `written_bytes` is the byte count the syscall reported; we
/// iterate only that range so a small mask inside a large
/// buffer does not scan past what the kernel actually wrote.
fn extract_cpus_from_mask(
    buffer: &[libc::c_ulong],
    written_bytes: usize,
) -> Option<Vec<u32>> {
    let word_bytes = std::mem::size_of::<libc::c_ulong>();
    let word_bits = libc::c_ulong::BITS as usize;
    let written_words = written_bytes / word_bytes;
    let mut cpus: Vec<u32> = Vec::new();
    for (word_idx, &word) in buffer.iter().take(written_words).enumerate() {
        if word == 0 {
            continue;
        }
        for bit in 0..word_bits {
            if word & (1 as libc::c_ulong) << bit != 0 {
                let cpu = word_idx * word_bits + bit;
                cpus.push(cpu as u32);
            }
        }
    }
    if cpus.is_empty() { None } else { Some(cpus) }
}

/// Read the cgroup v2 path from
/// `<proc_root>/<tgid>/task/<tid>/cgroup`. Format per
/// `cgroup(7)`: one line per hierarchy, shape
/// `hid:controllers:path`. The unified (v2) hierarchy is keyed
/// `0::<path>`; mixed-mode hosts expose legacy v1 hierarchies
/// alongside, which this reader skips. `None` on read failure
/// or when no v2 line is present.
pub fn read_cgroup_at(proc_root: &Path, tgid: i32, tid: i32) -> Option<String> {
    let raw = fs::read_to_string(task_file(proc_root, tgid, tid, "cgroup")).ok()?;
    parse_cgroup_v2(&raw)
}

pub fn read_cgroup(tgid: i32, tid: i32) -> Option<String> {
    read_cgroup_at(Path::new(DEFAULT_PROC_ROOT), tgid, tid)
}

fn parse_cgroup_v2(raw: &str) -> Option<String> {
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("0::") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SchedFields {
    nr_wakeups: Option<u64>,
    nr_wakeups_local: Option<u64>,
    nr_wakeups_remote: Option<u64>,
    nr_wakeups_sync: Option<u64>,
    nr_wakeups_migrate: Option<u64>,
    nr_wakeups_idle: Option<u64>,
    nr_migrations: Option<u64>,
    wait_sum: Option<u64>,
    wait_count: Option<u64>,
    sleep_sum: Option<u64>,
    block_sum: Option<u64>,
    block_count: Option<u64>,
    iowait_sum: Option<u64>,
    iowait_count: Option<u64>,
}

/// Parse `/proc/<tgid>/task/<tid>/sched`. Requires
/// `CONFIG_SCHED_DEBUG`. Format is many lines of `key : value`
/// where the key is dot-delimited (`se.statistics.nr_wakeups`);
/// different kernel versions use `se.statistics.`, `stats.`,
/// or bare names. The reader matches on the LAST dot-delimited
/// segment to absorb that variation.
///
/// Some fields (`wait_sum`, `sleep_sum`) are fractional on
/// newer kernels — parsed as `u64` first, falling back to
/// `f64` truncation so a `wait_sum : 123.456` line still
/// contributes the integer part.
fn parse_sched(raw: &str) -> SchedFields {
    let mut out = SchedFields::default();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        let short = key.rsplit('.').next().unwrap_or(key);
        let parsed_u64 = || value.parse::<u64>().ok();
        let parsed_u64_lossy = || {
            value
                .parse::<u64>()
                .ok()
                .or_else(|| value.parse::<f64>().ok().map(|f| f.max(0.0) as u64))
        };
        match short {
            "nr_wakeups" => out.nr_wakeups = parsed_u64(),
            "nr_wakeups_local" => out.nr_wakeups_local = parsed_u64(),
            "nr_wakeups_remote" => out.nr_wakeups_remote = parsed_u64(),
            "nr_wakeups_sync" => out.nr_wakeups_sync = parsed_u64(),
            "nr_wakeups_migrate" => out.nr_wakeups_migrate = parsed_u64(),
            "nr_wakeups_idle" => out.nr_wakeups_idle = parsed_u64(),
            "nr_migrations" => out.nr_migrations = parsed_u64(),
            "wait_sum" => out.wait_sum = parsed_u64_lossy(),
            "wait_count" => out.wait_count = parsed_u64(),
            // Kernel emits `sum_sleep_runtime` (see `kernel/sched/debug.c`
            // -> `proc_sched_show_task`), NOT `sleep_sum`. The old
            // `"sleep_sum"` match arm was a misnaming carried over
            // from an early misread of the procfs dump and never
            // populated for any in-tree kernel. Match on the real
            // kernel key so the field actually carries data.
            "sum_sleep_runtime" => out.sleep_sum = parsed_u64_lossy(),
            // `sleep_count` is NOT emitted anywhere — the
            // counterpart to sum_sleep_runtime is `nr_wakeups` (total
            // wake events), already covered above. The old
            // `sleep_count` field was a ghost parallel to `wait_count`
            // that never had a kernel-side source; removed along
            // with its match arm.
            "block_sum" => out.block_sum = parsed_u64_lossy(),
            "block_count" => out.block_count = parsed_u64(),
            "iowait_sum" => out.iowait_sum = parsed_u64_lossy(),
            "iowait_count" => out.iowait_count = parsed_u64(),
            _ => {}
        }
    }
    out
}

fn read_sched_at(proc_root: &Path, tgid: i32, tid: i32) -> SchedFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "sched")) {
        Ok(raw) => parse_sched(&raw),
        Err(_) => SchedFields::default(),
    }
}

/// Parse cgroup v2 `cpu.stat`. Format is lines of `key value`
/// (space-separated, not `key: value`).
fn parse_cpu_stat(raw: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    let mut usage = None;
    let mut throttled = None;
    let mut throttled_usec = None;
    for line in raw.lines() {
        let mut parts = line.split_ascii_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next() else { continue };
        let parsed = value.parse::<u64>().ok();
        match key {
            "usage_usec" => usage = parsed,
            "nr_throttled" => throttled = parsed,
            "throttled_usec" => throttled_usec = parsed,
            _ => {}
        }
    }
    (usage, throttled, throttled_usec)
}

/// Populate a [`CgroupStats`] by reading the cgroup v2 files
/// for `path` under `cgroup_root`. Missing files collapse to
/// `0` via the struct's `Default`, matching the "absent = 0"
/// contract the struct documents for allocated/deallocated_bytes.
pub fn read_cgroup_stats_at(cgroup_root: &Path, path: &str) -> CgroupStats {
    let relative = path.strip_prefix('/').unwrap_or(path);
    let dir = if relative.is_empty() {
        cgroup_root.to_path_buf()
    } else {
        cgroup_root.join(relative)
    };
    let (usage, throttled, throttled_usec) = fs::read_to_string(dir.join("cpu.stat"))
        .ok()
        .as_deref()
        .map(parse_cpu_stat)
        .unwrap_or((None, None, None));
    let memory_current = fs::read_to_string(dir.join("memory.current"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    CgroupStats {
        cpu_usage_usec: usage.unwrap_or(0),
        nr_throttled: throttled.unwrap_or(0),
        throttled_usec: throttled_usec.unwrap_or(0),
        memory_current: memory_current.unwrap_or(0),
    }
}

/// Heuristic check: does `/proc/<tgid>/maps` mention a jemalloc
/// DSO?
///
/// # Dead-code status and lift-point intent
///
/// Currently UNUSED by the capture layer: [`capture_thread_at`]
/// hardcodes `allocated_bytes: 0` / `deallocated_bytes: 0`
/// because populating those two fields requires DWARF-based TSD
/// offset resolution + ptrace, which lives in the out-of-band
/// `ktstr-jemalloc-probe` binary — NOT in the capture layer.
/// Shipping this detector as a public "scans maps for jemalloc"
/// API that callers cannot actually act on would be a trap; the
/// `#[allow(dead_code)]` attribute below keeps the code in-tree
/// as the lift-point for the future integration without letting
/// it signal "we enumerate jemalloc processes" to external
/// consumers.
///
/// Retained for the follow-up that wires the probe into the
/// capture pass — when that lands, `capture_thread_at` will
/// call this detector to decide whether to spawn the probe for
/// the current tgid, and the `#[allow]` attribute can be
/// removed alongside the `allocated_bytes: 0` hard-coding.
/// Until then, the detector is dead code with a deliberate
/// purpose: existing with the right signature so the wiring
/// change is a drop-in rather than a new file.
#[allow(dead_code)]
pub fn process_linked_against_jemalloc(tgid: i32) -> bool {
    process_linked_against_jemalloc_at(Path::new(DEFAULT_PROC_ROOT), tgid)
}

/// `proc_root`-parameterised variant of
/// [`process_linked_against_jemalloc`]. Lets tests drive the
/// detector against a synthetic `/proc/<tgid>/maps` file under
/// a tempdir without touching the real procfs. Production code
/// should stick with the default-root wrapper above; this
/// variant is kept `pub` so downstream harnesses that stand up
/// alternate `/proc` shapes (containers with mount namespaces,
/// pid-namespaced probes) can reuse the detection without
/// copy-pasting the maps-substring heuristic.
///
/// Shares the dead-code status documented on
/// [`process_linked_against_jemalloc`] — see that function for
/// why the detector exists without a production caller.
#[allow(dead_code)]
pub fn process_linked_against_jemalloc_at(proc_root: &Path, tgid: i32) -> bool {
    let Ok(raw) = fs::read_to_string(proc_root.join(tgid.to_string()).join("maps")) else {
        return false;
    };
    for line in raw.lines() {
        // Maps lines end with an optional path; jemalloc DSOs
        // embed "jemalloc" in the filename, and static-linked
        // jemalloc binaries reference its symbols from the
        // executable's own path. A crude substring match on the
        // path region is sufficient for gating — false positives
        // are absorbed by the downstream TSD probe, which does a
        // full ELF walk.
        if line.contains("jemalloc") {
            return true;
        }
    }
    false
}

/// Capture one thread's profile under an arbitrary procfs root.
/// Each procfs reader returns `Option`; the assembled
/// [`ThreadState`] coerces `None` to the field's default per the
/// module-level capture contract.
///
/// `use_syscall_affinity` gates the `sched_getaffinity(2)` path —
/// tests staging a synthetic `/proc` pass `false` so the syscall
/// does not read the REAL affinity of the test process; production
/// passes `true` and falls back to `Cpus_allowed_list:` when the
/// syscall returns EPERM.
pub fn capture_thread_at(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    pcomm: &str,
    use_syscall_affinity: bool,
) -> ThreadState {
    let comm = read_thread_comm_at(proc_root, tgid, tid).unwrap_or_default();
    let cgroup = read_cgroup_at(proc_root, tgid, tid).unwrap_or_default();
    let stat = read_stat_at(proc_root, tgid, tid);
    let (run_time_ns, wait_time_ns, timeslices) =
        read_schedstat_at(proc_root, tgid, tid);
    let io = read_io_at(proc_root, tgid, tid);
    let status = read_status_at(proc_root, tgid, tid);
    let sched = read_sched_at(proc_root, tgid, tid);
    let cpu_affinity = if use_syscall_affinity {
        read_affinity(tid).or(status.cpus_allowed).unwrap_or_default()
    } else {
        status.cpus_allowed.unwrap_or_default()
    };
    ThreadState {
        tid: tid as u32,
        tgid: tgid as u32,
        pcomm: pcomm.to_string(),
        comm,
        cgroup,
        start_time_clock_ticks: stat.start_time_clock_ticks.unwrap_or(0),
        policy: stat.policy.map(policy_name).unwrap_or_default(),
        nice: stat.nice.unwrap_or(0),
        cpu_affinity,
        run_time_ns: run_time_ns.unwrap_or(0),
        wait_time_ns: wait_time_ns.unwrap_or(0),
        timeslices: timeslices.unwrap_or(0),
        voluntary_csw: status.voluntary_csw.unwrap_or(0),
        nonvoluntary_csw: status.nonvoluntary_csw.unwrap_or(0),
        nr_wakeups: sched.nr_wakeups.unwrap_or(0),
        nr_wakeups_local: sched.nr_wakeups_local.unwrap_or(0),
        nr_wakeups_remote: sched.nr_wakeups_remote.unwrap_or(0),
        nr_wakeups_sync: sched.nr_wakeups_sync.unwrap_or(0),
        nr_wakeups_migrate: sched.nr_wakeups_migrate.unwrap_or(0),
        nr_wakeups_idle: sched.nr_wakeups_idle.unwrap_or(0),
        nr_migrations: sched.nr_migrations.unwrap_or(0),
        wait_sum: sched.wait_sum.unwrap_or(0),
        wait_count: sched.wait_count.unwrap_or(0),
        sleep_sum: sched.sleep_sum.unwrap_or(0),
        block_sum: sched.block_sum.unwrap_or(0),
        block_count: sched.block_count.unwrap_or(0),
        iowait_sum: sched.iowait_sum.unwrap_or(0),
        iowait_count: sched.iowait_count.unwrap_or(0),
        allocated_bytes: 0,
        deallocated_bytes: 0,
        minflt: stat.minflt.unwrap_or(0),
        majflt: stat.majflt.unwrap_or(0),
        rchar: io.rchar.unwrap_or(0),
        wchar: io.wchar.unwrap_or(0),
        syscr: io.syscr.unwrap_or(0),
        syscw: io.syscw.unwrap_or(0),
        read_bytes: io.read_bytes.unwrap_or(0),
        write_bytes: io.write_bytes.unwrap_or(0),
    }
}

#[cfg(test)]
fn capture_thread(tgid: i32, tid: i32, pcomm: &str) -> ThreadState {
    capture_thread_at(Path::new(DEFAULT_PROC_ROOT), tgid, tid, pcomm, true)
}

/// Capture a complete host-wide snapshot under arbitrary procfs
/// and cgroup roots. Walks `<proc_root>` for every live tgid,
/// enumerates its threads, and assembles a [`HostStateSnapshot`]
/// with per-cgroup enrichment populated once per distinct cgroup
/// path (many threads share a cgroup; keep the walk
/// O(cgroups) rather than O(threads)). The default-roots
/// production entry point is [`capture`]; tests pass a tempdir
/// to exercise the walk against a synthetic tree.
pub fn capture_with(
    proc_root: &Path,
    cgroup_root: &Path,
    use_syscall_affinity: bool,
) -> HostStateSnapshot {
    let captured_at_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let host = if use_syscall_affinity {
        Some(crate::host_context::collect_host_context())
    } else {
        None
    };
    let mut threads: Vec<ThreadState> = Vec::new();
    for tgid in iter_tgids_at(proc_root) {
        let pcomm = read_process_comm_at(proc_root, tgid).unwrap_or_default();
        for tid in iter_task_ids_at(proc_root, tgid) {
            let t = capture_thread_at(
                proc_root,
                tgid,
                tid,
                &pcomm,
                use_syscall_affinity,
            );
            // Ghost-thread filter: a tid that exited between the
            // `iter_task_ids_at` readdir and our per-file reads
            // produces an all-Default `ThreadState` — empty comm
            // and zero start_time_clock_ticks, because every
            // procfs file read bailed with ENOENT mid-capture.
            // Including these entries pollutes the comparison: a
            // baseline run might capture 1000 such ghosts and a
            // candidate 500, producing a spurious "500 ghost
            // threads vanished" diff signal in every report. A
            // legitimate thread under a real kernel always
            // carries at least one of these fields — kernel
            // threads have a non-empty comm at creation, user
            // threads inherit one from their parent — so an
            // entry with BOTH empty implies mid-capture exit.
            // The filter preserves the "captures-what-existed"
            // intent without softening the "captures every live
            // thread" invariant.
            if t.comm.is_empty() && t.start_time_clock_ticks == 0 {
                continue;
            }
            threads.push(t);
        }
    }
    let mut cgroup_stats: BTreeMap<String, CgroupStats> = BTreeMap::new();
    for t in &threads {
        if !t.cgroup.is_empty() && !cgroup_stats.contains_key(&t.cgroup) {
            cgroup_stats.insert(t.cgroup.clone(), read_cgroup_stats_at(cgroup_root, &t.cgroup));
        }
    }
    HostStateSnapshot {
        captured_at_unix_ns,
        host,
        threads,
        cgroup_stats,
    }
}

/// Capture a complete host-wide snapshot against the default
/// procfs and cgroup roots (`/proc` and `/sys/fs/cgroup`). Thin
/// shim over [`capture_with`].
pub fn capture() -> HostStateSnapshot {
    capture_with(
        Path::new(DEFAULT_PROC_ROOT),
        Path::new(DEFAULT_CGROUP_ROOT),
        true,
    )
}

/// Capture a snapshot and write it to `path` in the canonical
/// zstd+JSON format. Wrapper over [`capture`] +
/// [`HostStateSnapshot::write`] so CLI code can stay a single
/// call.
pub fn capture_to(path: &Path) -> Result<()> {
    let snap = capture();
    snap.write(path)
        .with_context(|| format!("write host-state snapshot to {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thread(pcomm: &str, comm: &str, run_time_ns: u64) -> ThreadState {
        ThreadState {
            tid: 1,
            tgid: 1,
            pcomm: pcomm.into(),
            comm: comm.into(),
            cgroup: "/".into(),
            start_time_clock_ticks: 0,
            policy: "SCHED_OTHER".into(),
            nice: 0,
            cpu_affinity: vec![0, 1],
            run_time_ns,
            ..ThreadState::default()
        }
    }

    #[test]
    fn snapshot_roundtrip_through_zstd_json() {
        let snap = HostStateSnapshot {
            captured_at_unix_ns: 42,
            host: None,
            threads: vec![
                thread("proc_a", "worker_0", 1_000_000),
                thread("proc_a", "worker_1", 2_000_000),
            ],
            cgroup_stats: BTreeMap::from([(
                "/".into(),
                CgroupStats {
                    cpu_usage_usec: 500,
                    nr_throttled: 0,
                    throttled_usec: 0,
                    memory_current: 1 << 20,
                },
            )]),
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        snap.write(tmp.path()).unwrap();
        let back = HostStateSnapshot::load(tmp.path()).unwrap();
        assert_eq!(back.captured_at_unix_ns, 42);
        assert_eq!(back.threads.len(), 2);
        assert_eq!(back.threads[1].run_time_ns, 2_000_000);
        assert_eq!(back.cgroup_stats["/"].cpu_usage_usec, 500);
    }

    #[test]
    fn load_rejects_non_zstd_payload() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"{\"not\": \"zstd\"}").unwrap();
        let err = HostStateSnapshot::load(tmp.path()).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("zstd"),
            "expected zstd error in context chain, got: {msg}",
        );
    }

    #[test]
    fn load_rejects_zstd_of_garbage_json() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let compressed = zstd::encode_all(&b"not json"[..], 3).unwrap();
        std::fs::write(tmp.path(), compressed).unwrap();
        let err = HostStateSnapshot::load(tmp.path()).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("parse host-state"),
            "expected parse error in context chain, got: {msg}",
        );
    }

    #[test]
    fn parse_stat_robust_against_paren_in_comm() {
        // Field 2 (comm) may contain ')'. The parser must latch on
        // the LAST ')'. Construct a line where comm is
        // `(weird)name)` and fields 3..=22 are 0..=19.
        let mut line = String::from("1234 (weird)name) ");
        for i in 0..20 {
            line.push_str(&format!("{i} "));
        }
        let f = parse_stat(&line);
        assert_eq!(f.start_time_clock_ticks, Some(19));
    }

    #[test]
    fn parse_stat_extracts_min_maj_nice_and_policy() {
        // Fields 3..=41 — tail indices 0..=38.
        // minflt at tail[7] = 7; majflt at tail[9] = 9;
        // nice at tail[16] = 16; starttime at tail[19] = 19;
        // policy at tail[38] = 38.
        let mut line = String::from("1 (n) ");
        for i in 0..=38 {
            line.push_str(&format!("{i} "));
        }
        let f = parse_stat(&line);
        assert_eq!(f.minflt, Some(7));
        assert_eq!(f.majflt, Some(9));
        assert_eq!(f.nice, Some(16));
        assert_eq!(f.start_time_clock_ticks, Some(19));
        assert_eq!(f.policy, Some(38));
    }

    #[test]
    fn parse_stat_short_line_drops_missing_fields() {
        // Only fields 3..=10 present; minflt at 7 landed, majflt at
        // 9 missing, later fields also missing.
        let line = "1 (n) 0 1 2 3 4 5 6 7";
        let f = parse_stat(line);
        assert_eq!(f.minflt, Some(7));
        assert_eq!(f.majflt, None);
        assert_eq!(f.nice, None);
        assert_eq!(f.start_time_clock_ticks, None);
        assert_eq!(f.policy, None);
    }

    #[test]
    fn parse_schedstat_three_fields() {
        let (a, b, c) = parse_schedstat("12345 67890 42\n");
        assert_eq!(a, Some(12345));
        assert_eq!(b, Some(67890));
        assert_eq!(c, Some(42));
    }

    #[test]
    fn parse_schedstat_missing_fields_drop_individually() {
        let (a, b, c) = parse_schedstat("12345\n");
        assert_eq!(a, Some(12345));
        assert_eq!(b, None);
        assert_eq!(c, None);
    }

    #[test]
    fn parse_io_extracts_all_six_fields() {
        let raw = "rchar: 1\n\
                   wchar: 2\n\
                   syscr: 3\n\
                   syscw: 4\n\
                   read_bytes: 5\n\
                   write_bytes: 6\n\
                   cancelled_write_bytes: 7\n";
        let f = parse_io(raw);
        assert_eq!(f.rchar, Some(1));
        assert_eq!(f.wchar, Some(2));
        assert_eq!(f.syscr, Some(3));
        assert_eq!(f.syscw, Some(4));
        assert_eq!(f.read_bytes, Some(5));
        assert_eq!(f.write_bytes, Some(6));
    }

    #[test]
    fn parse_status_extracts_csw_and_affinity() {
        let raw = "Name:\tbash\n\
                   Cpus_allowed_list:\t0-3,5\n\
                   voluntary_ctxt_switches:\t100\n\
                   nonvoluntary_ctxt_switches:\t5\n";
        let f = parse_status(raw);
        assert_eq!(f.voluntary_csw, Some(100));
        assert_eq!(f.nonvoluntary_csw, Some(5));
        assert_eq!(
            f.cpus_allowed.as_deref(),
            Some(&[0u32, 1, 2, 3, 5][..])
        );
    }

    #[test]
    fn parse_cpu_list_accepts_ranges_singletons_and_mixtures() {
        assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("5").unwrap(), vec![5]);
        assert_eq!(parse_cpu_list("0,2,4").unwrap(), vec![0, 2, 4]);
        assert_eq!(
            parse_cpu_list("0-2,4,6-7").unwrap(),
            vec![0, 1, 2, 4, 6, 7]
        );
    }

    #[test]
    fn parse_cpu_list_rejects_malformed_input() {
        assert!(parse_cpu_list("").is_none());
        assert!(parse_cpu_list("5-3").is_none());
        assert!(parse_cpu_list("abc").is_none());
        assert!(parse_cpu_list("0-").is_none());
        assert!(parse_cpu_list("-3").is_none());
    }

    #[test]
    fn parse_cpu_list_dedups_and_sorts() {
        assert_eq!(parse_cpu_list("3,0-2,1,2").unwrap(), vec![0, 1, 2, 3]);
    }

    /// A range whose expansion would exceed 64 Ki CPUs is
    /// rejected as malformed rather than allocating
    /// gigabytes. Without the `span > MAX_CPU_RANGE_EXPANSION`
    /// gate, a hostile or corrupt `Cpus_allowed_list:` value
    /// like `0-4294967295` would try to push 4 billion u32s
    /// into a Vec and either OOM the process or crash the
    /// capture. The cap sits orders of magnitude above any
    /// realistic `CONFIG_NR_CPUS` so legitimate inputs are
    /// never rejected.
    #[test]
    fn parse_cpu_list_rejects_huge_range() {
        // Malicious u32::MAX range — cap bites.
        assert_eq!(parse_cpu_list("0-4294967295"), None);
        // Just above the 64 Ki cap — still rejected.
        assert_eq!(parse_cpu_list("0-65536"), None);
        // At the cap — accepted (65_536 elements, the inclusive
        // `lo..=hi` boundary: 0 through 65_535).
        let at_cap = parse_cpu_list("0-65535").unwrap();
        assert_eq!(at_cap.len(), 65_536);
        // A realistic large-CPU range (e.g. 8192-way host) is
        // well under the cap and passes.
        let realistic = parse_cpu_list("0-8191").unwrap();
        assert_eq!(realistic.len(), 8192);
    }

    #[test]
    fn parse_cgroup_v2_picks_unified_hierarchy() {
        let raw = "12:cpuset:/legacy/cpuset/path\n\
                   0::/unified/path\n\
                   5:freezer:/legacy/freezer\n";
        assert_eq!(parse_cgroup_v2(raw), Some("/unified/path".to_string()));
    }

    #[test]
    fn parse_cgroup_v2_none_when_only_legacy_present() {
        let raw = "12:cpuset:/legacy/path\n";
        assert_eq!(parse_cgroup_v2(raw), None);
    }

    #[test]
    fn parse_sched_accepts_prefixed_and_bare_keys() {
        let raw = "se.statistics.nr_wakeups            :     1000\n\
                   se.nr_migrations                    :     42\n\
                   se.statistics.nr_wakeups_local      :     600\n\
                   se.statistics.wait_sum              :     12345.678\n";
        let f = parse_sched(raw);
        assert_eq!(f.nr_wakeups, Some(1000));
        assert_eq!(f.nr_migrations, Some(42));
        assert_eq!(f.nr_wakeups_local, Some(600));
        assert_eq!(f.wait_sum, Some(12345));
    }

    #[test]
    fn parse_cpu_stat_space_separated_format() {
        let raw = "usage_usec 1234\n\
                   user_usec 1000\n\
                   system_usec 234\n\
                   nr_periods 10\n\
                   nr_throttled 2\n\
                   throttled_usec 500\n";
        let (usage, throttled, throttled_usec) = parse_cpu_stat(raw);
        assert_eq!(usage, Some(1234));
        assert_eq!(throttled, Some(2));
        assert_eq!(throttled_usec, Some(500));
    }

    #[test]
    fn policy_name_known_and_unknown() {
        assert_eq!(policy_name(libc::SCHED_OTHER), "SCHED_OTHER");
        assert_eq!(policy_name(libc::SCHED_FIFO), "SCHED_FIFO");
        assert_eq!(policy_name(libc::SCHED_RR), "SCHED_RR");
        assert_eq!(policy_name(libc::SCHED_BATCH), "SCHED_BATCH");
        assert_eq!(policy_name(libc::SCHED_IDLE), "SCHED_IDLE");
        assert_eq!(policy_name(6), "SCHED_DEADLINE");
        assert_eq!(policy_name(7), "SCHED_EXT");
        assert_eq!(policy_name(99), "SCHED_UNKNOWN(99)");
    }

    #[test]
    fn iter_tgids_includes_self() {
        let tgids = iter_tgids();
        let pid = std::process::id() as i32;
        assert!(tgids.contains(&pid), "self pid {pid} not in /proc walk");
    }

    #[test]
    fn iter_task_ids_self_returns_at_least_main_tid() {
        let pid = std::process::id() as i32;
        let tids = iter_task_ids(pid);
        assert!(tids.contains(&pid), "main tid {pid} absent from /proc/self/task");
    }

    #[test]
    fn read_process_comm_for_self_is_populated() {
        let pid = std::process::id() as i32;
        let comm = read_process_comm(pid).expect("self comm must be readable");
        assert!(!comm.is_empty());
    }

    #[test]
    fn capture_thread_self_populates_identity() {
        let pid = std::process::id() as i32;
        let t = capture_thread(pid, pid, "testproc");
        assert_eq!(t.tid, pid as u32);
        assert_eq!(t.tgid, pid as u32);
        assert_eq!(t.pcomm, "testproc");
        assert!(!t.comm.is_empty());
        // On a real /proc, start_time_clock_ticks populates for live tasks.
        assert!(t.start_time_clock_ticks > 0);
        // Policy at minimum resolves to SCHED_OTHER for a normal process.
        assert!(!t.policy.is_empty());
    }

    #[test]
    fn capture_produces_non_empty_snapshot() {
        let snap = capture();
        assert!(!snap.threads.is_empty());
        // Every captured thread carries non-ephemeral identity.
        let pid = std::process::id() as u32;
        let self_threads: Vec<_> = snap.threads.iter().filter(|t| t.tgid == pid).collect();
        assert!(!self_threads.is_empty(), "own tgid missing from capture");
    }

    #[test]
    fn snapshot_extension_is_stable() {
        // Guard against accidental rename of the canonical extension.
        assert_eq!(SNAPSHOT_EXTENSION, "hst.zst");
    }

    // ------------------------------------------------------------
    // Parser edge-case coverage expansion
    //
    // The existing parse_* tests above cover the documented happy
    // paths plus the most-adversarial documented edge cases
    // (paren-in-comm, huge ranges, fractional fields). The tests
    // below cover MALFORMED, EMPTY, and BOUNDARY inputs that the
    // parsers silently absorb — regressions in this family would
    // land as stray data in the snapshot rather than loud failures,
    // which is exactly the class of drift the capture contract
    // ("absent = 0, best-effort, never-fail-the-snapshot") needs a
    // test gate against.
    // ------------------------------------------------------------

    /// parse_io on empty input produces the default `IoFields`
    /// (every field `None`). Empty input happens when `/proc/<tid>/io`
    /// is present but the kernel was compiled without
    /// `CONFIG_TASK_IO_ACCOUNTING` — the file exists with zero
    /// bytes. Without this gate the parser would silently accept
    /// the no-lines case by producing `IoFields::default()` anyway,
    /// but a regression that inverted an `if`/ early-returned a
    /// partial default would surface here.
    #[test]
    fn parse_io_empty_input_yields_all_none() {
        let f = parse_io("");
        assert_eq!(f, IoFields::default());
    }

    /// parse_io with a non-numeric value for a known key must drop
    /// ONLY the offending field — other lines still populate. Proves
    /// per-field `parse::<u64>().ok()` isolation rather than a
    /// whole-file bail that would zero out unrelated counters.
    #[test]
    fn parse_io_malformed_value_drops_only_that_field() {
        let raw = "rchar: 100\n\
                   wchar: not-a-number\n\
                   syscr: 3\n";
        let f = parse_io(raw);
        assert_eq!(f.rchar, Some(100));
        assert_eq!(f.wchar, None, "malformed value drops to None");
        assert_eq!(f.syscr, Some(3));
    }

    /// parse_cpu_list on a single-CPU range (`"5-5"`) must return
    /// a 1-element vec. `lo == hi` is the boundary of the inclusive
    /// range expansion — a regression that skipped the `lo == hi`
    /// case (e.g. `lo < hi` instead of `lo <= hi` in the loop)
    /// would drop the single element.
    #[test]
    fn parse_cpu_list_single_element_range_lo_equals_hi() {
        assert_eq!(parse_cpu_list("5-5").unwrap(), vec![5]);
        // Also pin at the cap boundary and bottom edge.
        assert_eq!(parse_cpu_list("0-0").unwrap(), vec![0]);
    }

    /// parse_cpu_list with a trailing comma (`"0,1,"`) must succeed
    /// and drop the empty token — the tokenizer has a dedicated
    /// `if token.is_empty() { continue }` arm precisely for this
    /// case. A user-pasted cpulist sometimes carries a stray comma
    /// from copy+paste; rejecting it would be a usability
    /// regression.
    #[test]
    fn parse_cpu_list_trailing_comma_accepted() {
        assert_eq!(parse_cpu_list("0,1,").unwrap(), vec![0, 1]);
        // Also the leading-comma case — same codepath.
        assert_eq!(parse_cpu_list(",0,1").unwrap(), vec![0, 1]);
    }

    /// parse_stat on a line with NO `)` returns `Default` — the
    /// `rfind(')')` guard in parse_stat short-circuits to
    /// `StatFields::default()` without tripping on out-of-bounds.
    /// A procfs file that got truncated mid-comm (impossible under
    /// correct procfs but possible against a fuzzer / synthetic
    /// tree) must not panic.
    #[test]
    fn parse_stat_empty_and_no_paren_return_default() {
        assert_eq!(parse_stat(""), StatFields::default());
        assert_eq!(
            parse_stat("garbage line with no close paren 1 2 3"),
            StatFields::default(),
            "line without `)` must return Default, not panic on \
             out-of-bounds indexing",
        );
        assert_eq!(
            parse_stat("  \n"),
            StatFields::default(),
            "whitespace-only input must also land at Default",
        );
    }

    /// parse_stat on multi-line input reads ONLY the first line.
    /// Production procfs stat is single-line; a synthetic
    /// multi-line file (e.g. a test fixture that appended extra
    /// rows by mistake, or a fuzz input) must not mix field
    /// positions across lines. Pins the `.lines().next()` behavior
    /// so a future refactor that concatenated lines would surface
    /// here.
    #[test]
    fn parse_stat_multi_line_input_uses_only_first_line() {
        let mut first = String::from("1 (proc) ");
        for i in 0..=38 {
            first.push_str(&format!("{i} "));
        }
        // Second line carries clearly-different values — if the
        // parser concatenated or mixed them, `nice` would change.
        let second = "2 (other) 999 999 999 999 999 999 999 999 999 999 \
                      999 999 999 999 999 999 999 999 999 999 999 999 999\n";
        let raw = format!("{first}\n{second}");
        let f = parse_stat(&raw);
        // First-line values untouched.
        assert_eq!(f.nice, Some(16));
        assert_eq!(f.start_time_clock_ticks, Some(19));
        assert_eq!(f.policy, Some(38));
    }

    /// parse_schedstat with more than three leading fields must
    /// accept the first three and ignore the rest. Real procfs
    /// stops at three, but a future kernel could append more or a
    /// synthetic fixture could pad the line — the parser's
    /// three-next-calls design already ignores tail tokens, and
    /// this test pins that invariant.
    ///
    /// Also covers the "invalid u64 token" path — a non-numeric
    /// token routes to None via `.parse::<u64>().ok()`.
    #[test]
    fn parse_schedstat_extra_fields_and_invalid_tokens() {
        // Four fields — fourth ignored.
        let (a, b, c) = parse_schedstat("1 2 3 4\n");
        assert_eq!((a, b, c), (Some(1), Some(2), Some(3)));
        // Invalid middle token drops only that slot.
        let (a, b, c) = parse_schedstat("1 invalid 3\n");
        assert_eq!(a, Some(1));
        assert_eq!(b, None);
        assert_eq!(c, Some(3));
        // Empty input → all None.
        let (a, b, c) = parse_schedstat("");
        assert_eq!((a, b, c), (None, None, None));
    }

    /// policy_name on a NEGATIVE integer must format as
    /// `"SCHED_UNKNOWN(-N)"` rather than panicking or producing an
    /// unsigned-wrapped value. The kernel's `policy` field is
    /// signed i32 (see `parse_stat::get_i32`), so a corrupt or
    /// out-of-band synthetic fixture could carry a negative value;
    /// the fallback branch must handle it cleanly.
    #[test]
    fn policy_name_negative_integer_renders_unknown() {
        assert_eq!(policy_name(-1), "SCHED_UNKNOWN(-1)");
        assert_eq!(policy_name(i32::MIN), format!("SCHED_UNKNOWN({})", i32::MIN));
    }

    /// parse_cpu_stat on empty input produces all-`None`. Same
    /// shape as `parse_io_empty_input_yields_all_none`, but
    /// exercises the space-separated key/value format rather than
    /// the `key: value` colon format — they are distinct parsers.
    #[test]
    fn parse_cpu_stat_empty_and_keyonly_lines_yield_none() {
        let (u, t, tu) = parse_cpu_stat("");
        assert_eq!((u, t, tu), (None, None, None));
        // Line with key but no value — dropped. The `parts.next()`
        // for value returns None → `continue`.
        let (u, t, tu) = parse_cpu_stat("usage_usec\n");
        assert_eq!((u, t, tu), (None, None, None));
    }

    /// parse_status with ONLY `voluntary_ctxt_switches` present
    /// populates only that field — the other two stay `None`. The
    /// production capture path coerces these to `0`; pinning the
    /// `None` at the parser layer proves the "absent vs. zero"
    /// distinction survives through the pure parser even if a
    /// future refactor separates the coercion.
    #[test]
    fn parse_status_partial_and_malformed_fields_isolate_correctly() {
        // Only voluntary_csw → other two None.
        let only_v = "Name:\tfoo\n\
                      voluntary_ctxt_switches:\t9\n";
        let f = parse_status(only_v);
        assert_eq!(f.voluntary_csw, Some(9));
        assert_eq!(f.nonvoluntary_csw, None);
        assert_eq!(f.cpus_allowed, None);

        // Malformed Cpus_allowed_list → cpus_allowed None (parse_cpu_list
        // returns None on bad tokens). Other fields still populate.
        let bad_cpu_list = "Cpus_allowed_list:\t5-3\n\
                            voluntary_ctxt_switches:\t1\n";
        let f = parse_status(bad_cpu_list);
        assert_eq!(f.voluntary_csw, Some(1));
        assert_eq!(
            f.cpus_allowed, None,
            "malformed cpulist must route parse_cpu_list's None \
             into the StatusFields field — not collapse to empty vec",
        );
    }

    /// parse_cgroup_v2 with an empty path (`"0::\n"`) returns None
    /// because the `!trimmed.is_empty()` guard rejects the blank
    /// path. A kernel bug or a synthetic fixture that emitted
    /// `0::` without a path must not land an empty-string cgroup
    /// in the ThreadState (which would then join against other
    /// cgroup-less threads and produce noise).
    ///
    /// Also pins the first-wins behavior when multiple unified
    /// lines appear — real procfs emits ONE v2 line per task, but
    /// a fixture might pad with duplicates; the parser returns on
    /// the first valid match.
    #[test]
    fn parse_cgroup_v2_empty_path_and_multiple_unified_lines() {
        // Empty path after `0::` — the guard rejects.
        assert_eq!(parse_cgroup_v2("0::\n"), None);
        assert_eq!(parse_cgroup_v2("0::   \n"), None);

        // First unified line wins when duplicates exist.
        let raw = "0::/first\n0::/second\n";
        assert_eq!(parse_cgroup_v2(raw), Some("/first".to_string()));
    }

    /// `read_thread_comm_at` returns `None` (not `Some("")`) when
    /// the comm file exists but contains only whitespace. The
    /// trim-then-is-empty guard is load-bearing: a `Some("")` in
    /// ThreadState.comm would both (a) disable the empty-comm ghost
    /// filter and (b) pollute comparison joins that key on comm.
    /// Pins the explicit empty→None routing so a future refactor
    /// that simplified the fn to `.ok().map(|s| s.trim().to_string())`
    /// (losing the empty guard) would break this test.
    #[test]
    fn read_thread_comm_at_whitespace_only_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tgid = 1;
        let tid = 1;
        let task_dir = tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string());
        std::fs::create_dir_all(&task_dir).unwrap();
        std::fs::write(task_dir.join("comm"), "   \n").unwrap();
        assert_eq!(read_thread_comm_at(tmp.path(), tgid, tid), None);

        // Also the missing-file branch (thread exited mid-read).
        assert_eq!(read_thread_comm_at(tmp.path(), tgid, 9999), None);
    }

    // ------------------------------------------------------------
    // read_affinity dynamic-buffer coverage
    // ------------------------------------------------------------

    /// `affinity_next_bits` doubles the buffer until the
    /// [`AFFINITY_MAX_BITS`] ceiling bites, then returns `None`
    /// to signal "give up". Pins the exact sequence 8192 →
    /// 16384 → 32768 → 65536 → 131072 → 262144 → None so a
    /// regression that replaced `checked_mul(2)` with `+= step`
    /// (or otherwise changed the growth curve) surfaces here.
    #[test]
    fn affinity_next_bits_doubles_until_ceiling() {
        assert_eq!(AFFINITY_INITIAL_BITS, 8192);
        assert_eq!(AFFINITY_MAX_BITS, 262144);
        // Full doubling chain from the initial size to the cap.
        let mut cur = AFFINITY_INITIAL_BITS;
        let expected = [16384usize, 32768, 65536, 131072, 262144];
        for &want in &expected {
            let next = affinity_next_bits(cur).expect("doubling must succeed below ceiling");
            assert_eq!(next, want, "expected {want}, got {next}");
            cur = next;
        }
        // At the cap, the next step would be 524288 > 262144 — return None.
        assert_eq!(
            affinity_next_bits(AFFINITY_MAX_BITS),
            None,
            "at the ceiling, no further retry must be allowed",
        );
    }

    /// A single-set-bit mask in the first word must be extracted
    /// to exactly that CPU id. Pins the word_idx*word_bits +
    /// bit offset arithmetic against off-by-one drift.
    #[test]
    fn extract_cpus_from_mask_single_bit_in_first_word() {
        let mut buf = vec![0 as libc::c_ulong; 4];
        // Set CPU 5 in word 0.
        buf[0] = (1 as libc::c_ulong) << 5;
        let bytes = std::mem::size_of_val(buf.as_slice());
        let cpus = extract_cpus_from_mask(&buf, bytes).expect("non-empty mask");
        assert_eq!(cpus, vec![5]);
    }

    /// A bit set in a NON-first word must be offset by
    /// word_bits * word_idx. Guards against a regression that
    /// dropped the `word_idx * word_bits` term and reported the
    /// bit position within the word instead of the absolute CPU
    /// id.
    #[test]
    fn extract_cpus_from_mask_offset_bit_in_later_word() {
        let word_bits = libc::c_ulong::BITS as usize;
        let mut buf = vec![0 as libc::c_ulong; 4];
        // Set CPU (2 * word_bits + 3) in word 2, bit 3.
        buf[2] = (1 as libc::c_ulong) << 3;
        let bytes = std::mem::size_of_val(buf.as_slice());
        let cpus = extract_cpus_from_mask(&buf, bytes).expect("non-empty mask");
        let expected = (2 * word_bits + 3) as u32;
        assert_eq!(cpus, vec![expected]);
    }

    /// `written_bytes` tighter than the buffer size must stop
    /// iteration at that byte count — bits beyond it belong to
    /// caller-zeroed padding and a kernel that returned a
    /// smaller mask than our buffer doesn't promise their shape.
    /// Pins that a stale bit planted past `written_bytes` is
    /// NOT harvested.
    #[test]
    fn extract_cpus_from_mask_respects_written_bytes() {
        let mut buf = vec![0 as libc::c_ulong; 4];
        // Plant CPU bits in word 0 AND word 3; tell the
        // extractor only word 0 was written by the kernel.
        buf[0] = (1 as libc::c_ulong) << 7; // CPU 7
        buf[3] = (1 as libc::c_ulong) << 0; // would-be CPU 3*word_bits
        let one_word_bytes = std::mem::size_of::<libc::c_ulong>();
        let cpus = extract_cpus_from_mask(&buf, one_word_bytes).expect("non-empty mask");
        // Only the bit in the first (kernel-written) word comes back.
        assert_eq!(cpus, vec![7]);
    }

    /// Empty mask (every word zero) → `None`. Pins the
    /// "Some(vec![]) is NOT a valid return" invariant — any
    /// caller that dispatches on `.is_some()` must be able to
    /// trust that a Some carries at least one CPU.
    #[test]
    fn extract_cpus_from_mask_empty_buffer_returns_none() {
        let buf = vec![0 as libc::c_ulong; 4];
        let bytes = std::mem::size_of_val(buf.as_slice());
        assert_eq!(extract_cpus_from_mask(&buf, bytes), None);
    }

    /// `affinity_zeroed_buffer` rounds UP to whole words so the
    /// byte length satisfies the kernel's
    /// `len & (sizeof(unsigned long)-1) == 0` alignment check.
    /// An off-by-one in the `div_ceil` would produce a
    /// non-multiple-of-word-size buffer and the syscall would
    /// reject with EINVAL forever (retry loop would churn but
    /// never succeed).
    #[test]
    fn affinity_zeroed_buffer_rounds_up_and_is_zeroed() {
        let word_bits = libc::c_ulong::BITS as usize;
        // Ask for exactly one word — get exactly one word.
        let exact = affinity_zeroed_buffer(word_bits);
        assert_eq!(exact.len(), 1);
        // Ask for one bit more than a word — get two words.
        let over = affinity_zeroed_buffer(word_bits + 1);
        assert_eq!(over.len(), 2);
        // Initial bits → 8192 / word_bits words.
        let init = affinity_zeroed_buffer(AFFINITY_INITIAL_BITS);
        assert_eq!(init.len(), AFFINITY_INITIAL_BITS / word_bits);
        // Every slot must be zeroed.
        assert!(init.iter().all(|&w| w == 0));
    }

    /// Smoke test against the real syscall for the current
    /// process — `read_affinity(getpid())` must succeed and
    /// return at least one CPU. The test process always has an
    /// affinity set (the kernel never runs a task off all
    /// CPUs), so None here signals a regression in the retry
    /// loop / errno classification.
    ///
    /// Distinct from `capture_thread_self_populates_identity`
    /// which exercises the full capture path — this test
    /// focuses on `read_affinity` in isolation so a failure
    /// localizes to the fn's own logic rather than a
    /// capture-path wiring issue.
    #[test]
    fn read_affinity_for_self_returns_at_least_one_cpu() {
        let pid = std::process::id() as i32;
        let cpus = read_affinity(pid).expect("own affinity must resolve");
        assert!(!cpus.is_empty(), "self affinity must carry at least one CPU");
        // CPUs come out sorted.
        let mut sorted = cpus.clone();
        sorted.sort_unstable();
        assert_eq!(cpus, sorted, "cpus must be returned sorted ascending");
    }

    // ------------------------------------------------------------
    // Synthetic-tree tests (H1-H5)
    //
    // Stage a tempdir shaped like `/proc/<tgid>/{comm,
    // task/<tid>/{stat,schedstat,status,io,sched,comm,cgroup}}`
    // so every capture helper can be driven without touching the
    // real procfs. Mirrors the compare-side pattern in
    // tests/host_state_compare.rs but against the capture side.
    // ------------------------------------------------------------

    /// Build a synthetic `/proc` under `root` carrying exactly one
    /// thread. Writes every file capture walks so every counter
    /// on `ThreadState` round-trips with a known value. `cpus` is
    /// the `Cpus_allowed_list` value (a range string the
    /// `parse_cpu_list` helper decodes).
    fn stage_synthetic_proc(
        root: &Path,
        tgid: i32,
        tid: i32,
        pcomm: &str,
        comm: &str,
    ) {
        use std::fs;
        let tgid_dir = root.join(tgid.to_string());
        let task_dir = tgid_dir.join("task").join(tid.to_string());
        fs::create_dir_all(&task_dir).unwrap();

        // /proc/<tgid>/comm
        fs::write(tgid_dir.join("comm"), format!("{pcomm}\n")).unwrap();
        // /proc/<tgid>/task/<tid>/comm
        fs::write(task_dir.join("comm"), format!("{comm}\n")).unwrap();

        // stat: paren-safe comm, fields 1..41. Comm inserted with
        // parens inside so the rfind(')') anchor has to find the
        // LAST close-paren, not the first. Fields past comm start
        // at index 0 in `tail` (tail[0] is `state`, per procfs
        // field-index-minus-three convention that parse_stat uses).
        // Field indices (post-comm):
        //   [0]=state [1]=ppid [2]=pgrp [3]=session [4]=tty
        //   [5]=tpgid [6]=flags [7]=minflt(field 10)
        //   [8]=cminflt [9]=majflt(field 12) [10]=cmajflt
        //   [11..16]=utime/stime/cutime/cstime/priority
        //   [16]=nice (field 19) [17]=num_threads [18]=itrealvalue
        //   [19]=starttime (field 22) [20..37]=vsize/rss/...
        //   [38]=policy (field 41).
        let stat_line = format!(
            "{tid} (proc (with) parens) R 1 2 3 4 5 6 \
             7777 0 8888 0 10 11 12 13 14 {nice} 1 0 \
             {starttime} 100 200 300 400 500 600 700 800 \
             900 1000 1100 1200 1300 1400 1500 1600 1700 1800 {policy}\n",
            tid = tid,
            nice = -10_i32,
            starttime = 555_555u64,
            policy = 0, // SCHED_OTHER
        );
        fs::write(task_dir.join("stat"), stat_line).unwrap();

        // schedstat: run_time_ns wait_time_ns timeslices
        fs::write(task_dir.join("schedstat"), "1000000 200000 50\n").unwrap();

        // status: voluntary/nonvoluntary csw + Cpus_allowed_list.
        // parse_status matches the lowercase csw keys verbatim;
        // only `Cpus_allowed_list` uses the capitalised leading
        // char of the procfs file.
        let status = "Name:\tfoo\n\
             voluntary_ctxt_switches:\t42\n\
             nonvoluntary_ctxt_switches:\t7\n\
             Cpus_allowed_list:\t0-3\n";
        fs::write(task_dir.join("status"), status).unwrap();

        // io: cumulative byte counters
        let io = "rchar: 100\n\
             wchar: 200\n\
             syscr: 10\n\
             syscw: 20\n\
             read_bytes: 4096\n\
             write_bytes: 8192\n";
        fs::write(task_dir.join("io"), io).unwrap();

        // sched: every parse_sched-matched key, with the
        // `se.statistics.` prefix for the wakeup family to
        // exercise the rsplit('.') short-key logic.
        let sched = "\
             se.statistics.nr_wakeups                       :         11\n\
             se.statistics.nr_wakeups_local                 :          8\n\
             se.statistics.nr_wakeups_remote                :          3\n\
             se.statistics.nr_wakeups_sync                  :          2\n\
             se.statistics.nr_wakeups_migrate               :          1\n\
             se.statistics.nr_wakeups_idle                  :          4\n\
             nr_migrations                                  :          9\n\
             wait_sum                                       :    5000.25\n\
             wait_count                                     :         15\n\
             sum_sleep_runtime                              :    3200.50\n\
             block_sum                                      :    1100.75\n\
             block_count                                    :          2\n\
             iowait_sum                                     :       77.0\n\
             iowait_count                                   :         18\n";
        fs::write(task_dir.join("sched"), sched).unwrap();

        // cgroup: v2-style single entry (0::path). read_cgroup_at
        // parses the `0::` prefix.
        fs::write(
            task_dir.join("cgroup"),
            "0::/ktstr.slice/worker0\n",
        )
        .unwrap();
    }

    /// Ghost-thread filter: a tid whose directory exists but
    /// carries ZERO readable procfs files (classic mid-capture
    /// exit — readdir races the reap) assembles an all-Default
    /// `ThreadState` and must NOT land in the snapshot. Stages
    /// one live thread with real content and one empty-directory
    /// ghost tid under the same tgid, calls `capture_with`, and
    /// asserts the output contains only the live thread.
    ///
    /// Without the filter (the pre-FIX-PhD3 behaviour), the
    /// ghost would land as `{ tid: 202, comm: "", cgroup: "",
    /// start_time_clock_ticks: 0, ...all counters zero }` and
    /// pollute downstream comparisons — a baseline run captures
    /// some number of ghosts, the candidate captures a
    /// different number, and the diff surfaces spurious
    /// "thread vanished" signal on every report.
    #[test]
    fn capture_with_filters_ghost_threads_with_empty_comm_and_zero_start() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 42;
        let live_tid: i32 = 101;
        let ghost_tid: i32 = 202;

        // Stage the live thread in full.
        stage_synthetic_proc(
            proc_tmp.path(),
            tgid,
            live_tid,
            "pcomm-proc",
            "live-thread",
        );

        // Stage a ghost tid directory with NO inner files —
        // simulates the "readdir saw it, per-file reads all
        // ENOENT'd" race window. `iter_task_ids_at` enumerates
        // it (the numeric dir name parses), every capture read
        // returns the default, and the filter rejects the
        // resulting all-zero entry.
        let ghost_dir = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(ghost_tid.to_string());
        std::fs::create_dir_all(&ghost_dir).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), false);

        // Exactly one thread — the live one. The ghost is gone.
        assert_eq!(
            snap.threads.len(),
            1,
            "ghost tid with empty comm + zero start must be filtered; \
             got threads: {:?}",
            snap.threads.iter().map(|t| (t.tid, &t.comm)).collect::<Vec<_>>(),
        );
        assert_eq!(snap.threads[0].tid, live_tid as u32);
        assert_eq!(snap.threads[0].comm, "live-thread");
    }

    /// H1 + H2 — `capture_with` against a synthetic procfs:
    /// staging every file the capture walks and asserting the
    /// assembled `ThreadState` carries the planted values.
    #[test]
    fn capture_with_synthetic_tree_assembles_thread_state() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 42;
        let tid: i32 = 101;

        stage_synthetic_proc(
            proc_tmp.path(),
            tgid,
            tid,
            "pcomm-proc",
            "worker-thread",
        );

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), false);

        // Exactly one thread — the one we planted.
        assert_eq!(snap.threads.len(), 1, "synthetic proc has one tid");
        let t = &snap.threads[0];

        // Identity fields (round-trip from /proc/<tgid>/comm +
        // /proc/<tgid>/task/<tid>/comm).
        assert_eq!(t.tid, tid as u32);
        assert_eq!(t.tgid, tgid as u32);
        assert_eq!(t.pcomm, "pcomm-proc");
        assert_eq!(t.comm, "worker-thread");
        assert_eq!(t.cgroup, "/ktstr.slice/worker0");

        // /proc/<tid>/stat fields parsed out of the paren-comm
        // tail: nice, starttime, policy.
        assert_eq!(t.nice, -10);
        assert_eq!(t.start_time_clock_ticks, 555_555);
        assert_eq!(t.policy, "SCHED_OTHER");
        assert_eq!(t.minflt, 7777);
        assert_eq!(t.majflt, 8888);

        // schedstat — three-tuple of run/wait/slices.
        assert_eq!(t.run_time_ns, 1_000_000);
        assert_eq!(t.wait_time_ns, 200_000);
        assert_eq!(t.timeslices, 50);

        // status — csw + Cpus_allowed_list. With
        // `use_syscall_affinity=false`, the capture path reads
        // cpu_affinity from status only.
        assert_eq!(t.voluntary_csw, 42);
        assert_eq!(t.nonvoluntary_csw, 7);
        assert_eq!(t.cpu_affinity, vec![0, 1, 2, 3]);

        // io — six cumulative counters.
        assert_eq!(t.rchar, 100);
        assert_eq!(t.wchar, 200);
        assert_eq!(t.syscr, 10);
        assert_eq!(t.syscw, 20);
        assert_eq!(t.read_bytes, 4096);
        assert_eq!(t.write_bytes, 8192);

        // sched — every wakeup field, migrations, and the four
        // fractional-parse fields (wait_sum/sleep_sum/block_sum/
        // iowait_sum).
        assert_eq!(t.nr_wakeups, 11);
        assert_eq!(t.nr_wakeups_local, 8);
        assert_eq!(t.nr_wakeups_remote, 3);
        assert_eq!(t.nr_wakeups_sync, 2);
        assert_eq!(t.nr_wakeups_migrate, 1);
        assert_eq!(t.nr_wakeups_idle, 4);
        assert_eq!(t.nr_migrations, 9);
        assert_eq!(t.wait_sum, 5000, "fractional 5000.25 truncates to 5000");
        assert_eq!(t.wait_count, 15);
        assert_eq!(
            t.sleep_sum, 3200,
            "fractional 3200.50 truncates to 3200 — sourced from the \
             kernel `sum_sleep_runtime` key, NOT the misnamed `sleep_sum` \
             of earlier drafts",
        );
        assert_eq!(t.block_sum, 1100, "fractional 1100.75 truncates to 1100");
        assert_eq!(t.block_count, 2);
        assert_eq!(t.iowait_sum, 77, "fractional 77.0 truncates to 77");
        assert_eq!(t.iowait_count, 18);
    }

    // ------------------------------------------------------------
    // H3 — read_cgroup_stats_at synthetic-tree coverage
    // ------------------------------------------------------------

    /// Write a cgroup v2-style `cpu.stat` file at
    /// `<root>/<relative>/cpu.stat`.
    fn write_cpu_stat(root: &Path, relative: &str, contents: &str) {
        let dir = root.join(relative.trim_start_matches('/'));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("cpu.stat"), contents).unwrap();
    }

    fn write_memory_current(root: &Path, relative: &str, contents: &str) {
        let dir = root.join(relative.trim_start_matches('/'));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("memory.current"), contents).unwrap();
    }

    /// Case (a): both `cpu.stat` and `memory.current` present →
    /// every field populated from the file contents.
    #[test]
    fn read_cgroup_stats_at_both_files_populate_all_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_cpu_stat(
            tmp.path(),
            "worker",
            "usage_usec 12345\nnr_throttled 7\nthrottled_usec 8\n",
        );
        write_memory_current(tmp.path(), "worker", "9999\n");
        let stats = read_cgroup_stats_at(tmp.path(), "/worker");
        assert_eq!(stats.cpu_usage_usec, 12345);
        assert_eq!(stats.nr_throttled, 7);
        assert_eq!(stats.throttled_usec, 8);
        assert_eq!(stats.memory_current, 9999);
    }

    /// Case (b): `cpu.stat` only → CPU fields populated,
    /// `memory_current` defaults to 0.
    #[test]
    fn read_cgroup_stats_at_cpu_stat_only_memory_defaults_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_cpu_stat(
            tmp.path(),
            "cpu-only",
            "usage_usec 500\nnr_throttled 0\nthrottled_usec 0\n",
        );
        let stats = read_cgroup_stats_at(tmp.path(), "/cpu-only");
        assert_eq!(stats.cpu_usage_usec, 500);
        assert_eq!(stats.nr_throttled, 0);
        assert_eq!(stats.throttled_usec, 0);
        assert_eq!(
            stats.memory_current, 0,
            "missing memory.current must collapse to 0, not None",
        );
    }

    /// Case (c): `memory.current` only → memory populated, CPU
    /// fields default to 0.
    #[test]
    fn read_cgroup_stats_at_memory_only_cpu_defaults_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        write_memory_current(tmp.path(), "mem-only", "2048\n");
        let stats = read_cgroup_stats_at(tmp.path(), "/mem-only");
        assert_eq!(stats.cpu_usage_usec, 0);
        assert_eq!(stats.nr_throttled, 0);
        assert_eq!(stats.throttled_usec, 0);
        assert_eq!(stats.memory_current, 2048);
    }

    /// Case (d): neither file present → every field zero.
    /// Distinct from "returns None or errors" — the documented
    /// contract is absent = 0.
    #[test]
    fn read_cgroup_stats_at_both_files_missing_all_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("empty-cg")).unwrap();
        let stats = read_cgroup_stats_at(tmp.path(), "/empty-cg");
        assert_eq!(stats.cpu_usage_usec, 0);
        assert_eq!(stats.nr_throttled, 0);
        assert_eq!(stats.throttled_usec, 0);
        assert_eq!(stats.memory_current, 0);
    }

    /// Case (e): `cpu.stat` present but missing `nr_throttled`
    /// key → that field defaults to 0, OTHER known keys still
    /// populate. Proves the parser scans by key rather than
    /// positionally.
    #[test]
    fn read_cgroup_stats_at_cpu_stat_missing_key_defaults_field_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        // Missing `nr_throttled` entirely; other two keys present.
        write_cpu_stat(
            tmp.path(),
            "partial",
            "usage_usec 999\nthrottled_usec 111\n",
        );
        let stats = read_cgroup_stats_at(tmp.path(), "/partial");
        assert_eq!(stats.cpu_usage_usec, 999);
        assert_eq!(stats.nr_throttled, 0, "absent key collapses to 0");
        assert_eq!(stats.throttled_usec, 111);
    }

    // ------------------------------------------------------------
    // H4 — parse_sched every-field coverage + parse fallbacks
    // ------------------------------------------------------------

    /// Populated `/proc/<tid>/sched` with every one of the 14
    /// fields parse_sched recognises. Ordering mixed (sync before
    /// local, migrate before idle) so the test doesn't pin a
    /// single-pass scan order that the helper doesn't actually
    /// promise.
    #[test]
    fn parse_sched_populates_all_fourteen_fields() {
        let raw = "\
             se.statistics.nr_wakeups                       :         11\n\
             se.statistics.nr_wakeups_sync                  :          2\n\
             se.statistics.nr_wakeups_local                 :          8\n\
             se.statistics.nr_wakeups_migrate               :          1\n\
             se.statistics.nr_wakeups_remote                :          3\n\
             se.statistics.nr_wakeups_idle                  :          4\n\
             nr_migrations                                  :          9\n\
             wait_sum                                       :       500\n\
             wait_count                                     :         15\n\
             sum_sleep_runtime                              :       320\n\
             block_sum                                      :       110\n\
             block_count                                    :          2\n\
             iowait_sum                                     :         77\n\
             iowait_count                                   :         18\n";
        let s = parse_sched(raw);
        assert_eq!(s.nr_wakeups, Some(11));
        assert_eq!(s.nr_wakeups_local, Some(8));
        assert_eq!(s.nr_wakeups_remote, Some(3));
        assert_eq!(s.nr_wakeups_sync, Some(2));
        assert_eq!(s.nr_wakeups_migrate, Some(1));
        assert_eq!(s.nr_wakeups_idle, Some(4));
        assert_eq!(s.nr_migrations, Some(9));
        assert_eq!(s.wait_sum, Some(500));
        assert_eq!(s.wait_count, Some(15));
        assert_eq!(
            s.sleep_sum,
            Some(320),
            "kernel key is `sum_sleep_runtime`, not `sleep_sum` — the \
             old match arm was a ghost and never populated",
        );
        assert_eq!(s.block_sum, Some(110));
        assert_eq!(s.block_count, Some(2));
        assert_eq!(s.iowait_sum, Some(77));
        assert_eq!(s.iowait_count, Some(18));
    }

    /// Fractional-parse fallback — newer kernels emit
    /// `wait_sum/sleep_sum/block_sum/iowait_sum` as floats;
    /// parse_sched first tries u64, then falls back to
    /// f64-truncate. The integer part survives.
    #[test]
    fn parse_sched_fractional_fields_truncate_to_integer() {
        let raw = "\
             wait_sum                                       :    1234.5\n\
             sum_sleep_runtime                              :     678.9\n\
             block_sum                                      :      42.1\n\
             iowait_sum                                     :       7.999\n";
        let s = parse_sched(raw);
        assert_eq!(s.wait_sum, Some(1234));
        assert_eq!(s.sleep_sum, Some(678));
        assert_eq!(s.block_sum, Some(42));
        assert_eq!(s.iowait_sum, Some(7));
    }

    /// Fractional fallback must clamp negatives to 0 — f64::as u64
    /// on a negative value is UB-adjacent, so the helper uses
    /// `.max(0.0) as u64`. Pins that clamp.
    #[test]
    fn parse_sched_negative_fractional_clamps_to_zero() {
        let raw = "wait_sum                                       :   -5.0\n";
        let s = parse_sched(raw);
        assert_eq!(s.wait_sum, Some(0));
    }

    /// Bare-key names (no `se.statistics.` prefix) must still
    /// populate — some kernels emit `nr_wakeups : N` at the top
    /// level. The parser's `rsplit('.').next()` treats a no-dot
    /// string as the whole string.
    #[test]
    fn parse_sched_bare_key_names_populate_same_fields() {
        let raw = "\
             nr_wakeups                                     :         11\n\
             nr_wakeups_local                               :          8\n\
             nr_wakeups_remote                              :          3\n\
             nr_wakeups_sync                                :          2\n\
             nr_wakeups_migrate                             :          1\n\
             nr_wakeups_idle                                :          4\n";
        let s = parse_sched(raw);
        assert_eq!(s.nr_wakeups, Some(11));
        assert_eq!(s.nr_wakeups_local, Some(8));
        assert_eq!(s.nr_wakeups_remote, Some(3));
        assert_eq!(s.nr_wakeups_sync, Some(2));
        assert_eq!(s.nr_wakeups_migrate, Some(1));
        assert_eq!(s.nr_wakeups_idle, Some(4));
    }

    /// Future `stats.` or other prefix variants must also
    /// populate — the parser matches on the LAST dot-delimited
    /// segment, so any enclosing prefix is ignored by design.
    #[test]
    fn parse_sched_alternative_prefix_populates_same_fields() {
        let raw = "\
             stats.nr_wakeups                               :         42\n\
             some.other.prefix.nr_migrations                :          9\n";
        let s = parse_sched(raw);
        assert_eq!(s.nr_wakeups, Some(42));
        assert_eq!(s.nr_migrations, Some(9));
    }

    /// Unknown keys don't corrupt populated fields — important
    /// because kernel versions add new lines frequently and the
    /// parser must skip them rather than mis-route.
    #[test]
    fn parse_sched_unknown_keys_are_ignored() {
        let raw = "\
             nr_wakeups                                     :         11\n\
             fictional_new_kernel_stat                      :       9999\n\
             nr_migrations                                  :          9\n";
        let s = parse_sched(raw);
        assert_eq!(s.nr_wakeups, Some(11));
        assert_eq!(s.nr_migrations, Some(9));
    }

    // ------------------------------------------------------------
    // H5 — process_linked_against_jemalloc_at with synthetic maps
    // ------------------------------------------------------------

    /// Maps file containing a `libjemalloc.so.2` DSO path →
    /// detector returns `true`.
    #[test]
    fn process_linked_against_jemalloc_at_detects_dso_in_maps() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tgid = 777;
        let proc_dir = tmp.path().join(tgid.to_string());
        std::fs::create_dir_all(&proc_dir).unwrap();
        let maps = "\
             5583e6f7a000-5583e6f7b000 r-xp 00000000 00:00 0\n\
             7f4567890000-7f4567abc000 r-xp 00000000 fe:00 12345 /usr/lib/x86_64-linux-gnu/libjemalloc.so.2\n\
             7f4567abc000-7f4567def000 r--p 00000000 fe:00 67890 /usr/lib/x86_64-linux-gnu/libc.so.6\n";
        std::fs::write(proc_dir.join("maps"), maps).unwrap();
        assert!(process_linked_against_jemalloc_at(tmp.path(), tgid));
    }

    /// Maps file without a jemalloc reference → detector returns
    /// `false`.
    #[test]
    fn process_linked_against_jemalloc_at_absent_returns_false() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tgid = 888;
        let proc_dir = tmp.path().join(tgid.to_string());
        std::fs::create_dir_all(&proc_dir).unwrap();
        let maps = "\
             5583e6f7a000-5583e6f7b000 r-xp 00000000 00:00 0\n\
             7f4567abc000-7f4567def000 r--p 00000000 fe:00 67890 /usr/lib/x86_64-linux-gnu/libc.so.6\n";
        std::fs::write(proc_dir.join("maps"), maps).unwrap();
        assert!(!process_linked_against_jemalloc_at(tmp.path(), tgid));
    }

    /// Missing `/proc/<tgid>/maps` file (process exited
    /// mid-read) → detector returns `false` gracefully.
    #[test]
    fn process_linked_against_jemalloc_at_missing_file_returns_false() {
        let tmp = tempfile::TempDir::new().unwrap();
        // No /proc/<tgid>/maps staged.
        assert!(!process_linked_against_jemalloc_at(tmp.path(), 999));
    }

    /// Static-linked jemalloc — the "jemalloc" substring appears
    /// in the executable path, not in a libjemalloc.so DSO. The
    /// detector's crude substring match catches both forms; this
    /// pins the static case so a future tightening to
    /// "libjemalloc" prefix doesn't silently drop static
    /// detection without the test surfacing the behaviour change.
    #[test]
    fn process_linked_against_jemalloc_at_detects_static_linked() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tgid = 1234;
        let proc_dir = tmp.path().join(tgid.to_string());
        std::fs::create_dir_all(&proc_dir).unwrap();
        let maps = "\
             5583e6f7a000-5583e6f7b000 r-xp 00000000 fe:00 555 /usr/local/bin/my-jemalloc-linked-app\n\
             7f4567abc000-7f4567def000 r--p 00000000 fe:00 67890 /usr/lib/x86_64-linux-gnu/libc.so.6\n";
        std::fs::write(proc_dir.join("maps"), maps).unwrap();
        assert!(process_linked_against_jemalloc_at(tmp.path(), tgid));
    }
}
