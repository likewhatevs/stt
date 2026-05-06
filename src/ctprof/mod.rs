//! Per-thread ctprof (cgroup/thread profiler) data model + capture layer.
//!
//! [`CtprofSnapshot`] is the serialized container for a single
//! host-wide per-thread profile. Capture produces one via the
//! `ktstr ctprof capture -o snapshot.ctprof.zst` subcommand;
//! comparison reads two and joins them on the selected grouping
//! axis (pcomm, cgroup, or comm).
//!
//! Field families and probe-timing invariance:
//!
//! - **Cumulative counters and totals** (the majority): wakeups,
//!   migrations, csw, run/wait/sleep/block/iowait time, schedstat
//!   counts, page-fault counters, syscall counters, byte counters,
//!   the taskstats per-bucket `*_count` and `*_delay_total_ns`,
//!   the jemalloc per-thread allocated/deallocated TSD counters,
//!   etc. Sampled twice at different instants the value increases
//!   monotonically; probe-attach latency does not alter the
//!   reading.
//! - **Lifetime high-water peaks**: schedstat `*_max` family
//!   (`wait_max`, `sleep_max`, `block_max`, `exec_max`,
//!   `slice_max`), every taskstats `*_delay_max_ns` /
//!   `*_delay_min_ns`, and the memory watermarks
//!   (`hiwater_rss_bytes`, `hiwater_vm_bytes`). These are
//!   non-decreasing-over-time but per-event extrema rather than
//!   sums, so they are non-summable across threads (the registry
//!   reduces them via `MaxPeak` / `MaxPeakBytes`). Same
//!   probe-timing invariance as the cumulative counters.
//! - **Instantaneous gauges** (sensitive to probe timing):
//!   [`ThreadState::nr_threads`] (signal_struct->nr_threads
//!   snapshot), [`ThreadState::fair_slice_ns`] (instantaneous
//!   `p->se.slice`), and [`ThreadState::state`]
//!   (task_state_array letter). Sampled at capture time and can
//!   genuinely differ between two probes of the same thread.
//!   The registry pairs them with `MaxGaugeCount` /
//!   `MaxGaugeNs` / `ModeChar` reductions rather than the
//!   `Sum*` rules used for cumulative counters.
//! - **Categorical / ordinal scalars** (point-in-time
//!   snapshots): `policy`, `nice`, `priority`, `processor`,
//!   `rt_priority`, plus the identity strings (`pcomm`, `comm`,
//!   `cgroup`) and the [`crate::metric_types::CpuSet`]
//!   `cpu_affinity`. These are sampled at capture time and can
//!   change at runtime (e.g. `sched_setaffinity` mid-run flips
//!   `processor` and `cpu_affinity`), so they share the
//!   gauge family's probe-timing sensitivity. The registry
//!   reduces them via `Mode*` / `Range*` / `Affinity` rather
//!   than `Sum*`.
//!
//! The jemalloc per-thread TSD counters
//! (`tsd_s.thread_allocated` / `thread_deallocated`) jemalloc
//! maintains unconditionally on its alloc/dalloc fast and slow
//! paths, so the ptrace-based attach this layer performs does
//! not perturb them; counters previously accumulated remain
//! valid across the brief stop the attach induces. Metrics not
//! derivable from cumulative state (e.g. perf_event_open
//! counters that reset on attachment) are intentionally absent
//! from this capture layer.
//!
//! # Capture model
//!
//! [`capture`] walks `/proc` for every live tgid, enumerates its
//! threads, and populates each [`ThreadState`] from a handful of
//! procfs sources: `stat`, `schedstat`, `status`, `io`, `sched`,
//! `comm`, `cgroup`. The procfs walk runs sequentially per tid in
//! `capture_with` phase 2. Phase 1 attaches the jemalloc TSD
//! probe in parallel across tgids when `use_syscall_affinity` is
//! `true` (the production path); under `use_syscall_affinity =
//! false` (the synthetic-tree test path), phase 1 is skipped
//! entirely — the per-tgid probe map starts and stays empty, and
//! phase 2's per-tid lookup falls through to the absent-counter
//! default of zero. See "Probe wiring" below for the per-tgid
//! mechanics.
//!
//! ## Probe wiring (most-expensive step)
//!
//! For every tgid the walk reaches, the capture pipeline calls
//! the `pub(crate)` `host_thread_probe::attach_jemalloc_at` (or
//! its default-root `attach_jemalloc` wrapper) to resolve the
//! target's jemalloc TLS symbol + per-`tsd_s` field offsets via
//! an ELF parse and DWARF walk; per-thread counter reads then
//! dispatch through `host_thread_probe::probe_thread` for one
//! ptrace cycle: seize → interrupt → waitpid → getregset →
//! `process_vm_readv` → detach (the detach happens automatically
//! via the `ScopeDetach` Drop guard, so any fallible step still
//! leaves the target unstuck). The remote read pulls a
//! contiguous 24-byte counter span — the canonical jemalloc
//! `TSD_DATA_FAST` layout (allocated, fast-event slot,
//! deallocated) — but the byte count is computed dynamically by
//! `combined_read_span` from the DWARF-resolved field offsets, so
//! a future jemalloc layout change is absorbed. This is the
//! dominant wall-clock cost of a snapshot:
//! O(unique-exe-inode tgids) ELF parses + O(jemalloc-linked
//! tgids) DWARF walks + O(threads of jemalloc-linked tgids)
//! ptrace cycles. The first term covers non-jemalloc tgids: each
//! distinct `/proc/<pid>/exe` inode still costs one ELF parse to
//! discover absence (the inode-keyed cache below collapses
//! repeats). `attach_jemalloc_at` is the sole detection gate —
//! tgids that attach successfully populate `allocated_bytes` /
//! `deallocated_bytes`; tgids that fail attach (not jemalloc-
//! linked, stripped binary, ptrace denied, arch mismatch — see
//! `host_thread_probe::AttachError`) land their threads at the
//! absent-counter default of zero.
//!
//! Phase 1 parallelism is gated by host CPU headroom (read from
//! `<proc_root>/loadavg`, clamped to `[1, num_cpus/2 + 1]`) so the
//! capture cannot drown a hot host with concurrent ELF reads.
//! Per-tgid attach results are inode-keyed cached so a fork-bombed
//! tgid family resolves DWARF once. The per-tgid wrapper
//! `try_attach_probe_for_tgid_at` records every outcome in a single
//! `ProbeSummary` tally; `emit_probe_summary` surfaces a single
//! info-level line per snapshot summarising tgids walked, jemalloc
//! detected, probed OK, failed, plus the dominant actionable
//! failure tag and an EPERM remediation hint when ptrace-attach
//! failures dominate.
//!
//! Each internal procfs reader returns `Option` (graceful on
//! missing/unreadable — a kernel without `CONFIG_SCHEDSTATS` or
//! `CONFIG_SCHED_DEBUG` yields `None` from the affected reader
//! without failing the rest of the thread). The assembled
//! [`ThreadState`] treats `None` as "absent at capture" via the
//! field type — counters collapse to `0`, identity strings
//! collapse to empty, affinity collapses to an empty vec. A
//! missing reading is therefore indistinguishable from a genuine
//! zero in the serialized output; the capture contract is
//! best-effort, never-fail-the-snapshot. Tests that need stronger
//! guarantees inspect the underlying readers directly (they remain
//! `Option`-shaped, unit-tested in this module).
//!
//! # Privilege
//!
//! Pulling the jemalloc per-thread TSD counters requires
//! `ptrace(PTRACE_SEIZE)` against the target. Under
//! `kernel.yama.ptrace_scope=0` any same-uid process attaches.
//! Under `=1` (Debian/Ubuntu host default) the tracer must be an
//! ancestor of the target or carry `CAP_SYS_PTRACE`; `=2` and `=3`
//! raise the bar further. When attach fails, the per-thread
//! `allocated_bytes` / `deallocated_bytes` collapse to 0 per the
//! best-effort contract — the rest of the snapshot still
//! populates from procfs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Top-level serialized artifact produced by `ktstr ctprof`.
///
/// The file layout on disk is zstd-compressed JSON of this struct.
/// Extension `.ctprof.zst` is conventional; nothing in the loader
/// depends on the extension beyond being passed a path that
/// resolves to a readable file.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CtprofSnapshot {
    /// Wall-clock time at capture, nanoseconds since the Unix
    /// epoch. Useful as a tie-breaker when comparing two snapshots
    /// that originate from the same host — the newer one is
    /// candidate by default — but carries no load-bearing role in
    /// any grouping axis.
    pub captured_at_unix_ns: u64,

    /// Host context snapshot (kernel, CPU, memory, tunables).
    /// Optional because older tools or synthetic fixtures may
    /// omit it; comparison degrades to a "host context unavailable"
    /// line rather than failing the whole compare when either
    /// side is missing.
    pub host: Option<crate::host_context::HostContext>,

    /// One entry per observed thread on the host at capture time.
    /// Order is not load-bearing; the comparison pipeline groups
    /// by `pcomm` / `cgroup` / `comm` depending on `--group-by`.
    pub threads: Vec<ThreadState>,

    /// Enrichment metadata for every cgroup that at least one
    /// sampled thread resides in. Keyed by the cgroup path
    /// relative to the v2 mount (e.g.
    /// `/kubepods/burstable/pod-<id>/container`). Populated from
    /// the cgroup filesystem, not the per-thread sample, because
    /// cpu.stat / memory.current describe the cgroup's aggregate
    /// state, not per-thread contribution.
    pub cgroup_stats: BTreeMap<String, CgroupStats>,

    /// Probe outcome statistics for the snapshot, when the probe
    /// pass ran. `None` indicates the snapshot was assembled
    /// without the per-tgid jemalloc probe walk (synthetic-tree
    /// tests pass `use_syscall_affinity=false` to skip it).
    /// `Some(_)` carries the per-snapshot tally — see
    /// [`CtprofProbeSummary`] for the curated field set.
    pub probe_summary: Option<CtprofProbeSummary>,

    /// Procfs-read failure statistics for the snapshot, when the
    /// capture pass ran in production mode. Mirrors the
    /// `probe_summary` discipline: `None` indicates synthetic-tree
    /// tests skipped it (`use_syscall_affinity=false`); `Some(_)`
    /// carries the per-snapshot read-level failure tally — see
    /// [`CtprofParseSummary`].
    pub parse_summary: Option<CtprofParseSummary>,

    /// Per-snapshot taskstats genetlink query outcome tally,
    /// populated when the capture pass ran in production mode.
    /// `None` mirrors `probe_summary` / `parse_summary`:
    /// synthetic-tree tests pass `use_syscall_affinity=false`
    /// which skips the netlink path entirely. `Some(_)` carries
    /// the per-snapshot ok/eperm/esrch/other counts so an operator
    /// can distinguish "no taskstats data because every tid raced
    /// exit" (high `esrch_count`) from "no taskstats data because
    /// the kernel was built without `CONFIG_TASKSTATS`" (the
    /// netlink open failed up-front so every counter is zero)
    /// from "no taskstats data because `CAP_NET_ADMIN` is missing"
    /// (high `eperm_count`). See [`crate::taskstats::TaskstatsSummary`]
    /// for the per-counter semantics and remediation guidance.
    pub taskstats_summary: Option<crate::taskstats::TaskstatsSummary>,

    /// Host-level Pressure Stall Information, populated from
    /// `<proc_root>/pressure/{cpu,memory,io,irq}`. Captures
    /// system-wide stall pressure across the four kernel-exposed
    /// resources. Defaults to all-zero when the kernel has
    /// CONFIG_PSI off or when individual resource files are
    /// absent. See [`Psi`] for the per-resource shape and the
    /// system-level cpu.full / irq.some caveats.
    pub psi: Psi,

    /// Global sched_ext sysfs state from `/sys/kernel/sched_ext/`.
    /// `None` when CONFIG_SCHED_CLASS_EXT is not built (no
    /// `sched_ext` sysfs directory exists), or when the
    /// directory itself is unreadable. See [`SchedExtSysfs`]
    /// for the per-field shape and kernel cites. Populated
    /// during the same capture pass as PSI.
    pub sched_ext: Option<SchedExtSysfs>,
}

/// Per-snapshot probe outcome statistics. Curated projection of
/// the capture pipeline's internal probe tally — exposes the
/// counters, the dominant failure tag, and a `privilege_dominant`
/// boolean a downstream consumer needs to decide whether the
/// snapshot's `allocated_bytes` / `deallocated_bytes` fields are
/// trustworthy on a given host without parsing the operator-
/// facing tracing line.
///
/// The internal probe taxonomy (the per-variant
/// `host_thread_probe::AttachError` and `ProbeError` enums) is
/// deliberately NOT mirrored here — it is implementation
/// detail that may change shape without breaking this contract.
/// `dominant_failure` carries the operator-facing tag string
/// (e.g. `"ptrace-seize"`, `"dwarf-parse-failure"`) that the
/// capture pipeline already surfaces in its tracing summary; the
/// stable token format is documented in the `ktstr ctprof
/// capture` CLI help. `privilege_dominant` mirrors the same gate
/// that prints the EPERM remediation hint — true when ≥ 50% of
/// `failed` is `ptrace-seize` or `ptrace-interrupt`.
///
/// The four counters are zero when the probe pass reached zero
/// tgids (e.g. an empty `proc_root`); `dominant_failure` is
/// `None` when no actionable failures landed; `privilege_dominant`
/// is `false` when there are no failures or when ptrace failures
/// are strictly less than half of `failed` (the `>= 50%` gate
/// accepts equality at the boundary).
///
/// # Examples
///
/// ```no_run
/// let snap = ktstr::ctprof::capture();
/// if let Some(ps) = &snap.probe_summary {
///     if let Some(hint) = ps.remediation_hint() {
///         eprintln!("{hint}");
///     }
///     if let Some(tag) = &ps.dominant_failure {
///         eprintln!("dominant failure: {tag}");
///     }
/// }
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CtprofProbeSummary {
    /// Total tgids the probe pass walked. Equals the number of
    /// `/proc/<pid>` directories the capture saw, minus the
    /// calling process's own tgid (which is skipped because
    /// `PTRACE_SEIZE` rejects self-attach).
    pub tgids_walked: u64,
    /// Tgids whose `attach_jemalloc_at` call succeeded — i.e.
    /// the target was identified as jemalloc-linked, the TSD
    /// symbol resolved, and the per-`tsd_s` field offsets came
    /// out of the DWARF walk. A subset of `tgids_walked`.
    pub jemalloc_detected: u64,
    /// Per-thread probe reads that returned a counter pair.
    /// Bounded above by the sum of thread counts across all
    /// `jemalloc_detected` tgids; per-thread failures (target
    /// thread exited mid-attach, EPERM, etc.) reduce this count
    /// below the upper bound.
    pub probed_ok: u64,
    /// Attach-or-probe failures whose tag is classified
    /// ACTIONABLE — see the `ktstr ctprof capture` CLI help
    /// for the full filter rule and tag taxonomy. Routine
    /// non-actionable outcomes (target not jemalloc-linked,
    /// `readlink` race-with-exit) do NOT contribute to this
    /// count.
    pub failed: u64,
    /// Tag string for the most-frequent actionable failure across
    /// all attach-and-probe failures. `None` when `failed == 0`.
    /// Stable single-word identifiers — the wire contract that
    /// downstream consumers match against. The full taxonomy is
    /// documented in the `ktstr ctprof capture` CLI help.
    /// Examples: `"ptrace-seize"`, `"dwarf-parse-failure"`,
    /// `"jemalloc-in-dso"`.
    pub dominant_failure: Option<String>,
    /// `true` when the ptrace failure share crosses the
    /// hint-trigger threshold (≥ 50% of `failed` is `ptrace-seize`
    /// or `ptrace-interrupt`). Mirrors the same gate that prints
    /// the EPERM remediation hint in the operator-facing tracing
    /// summary, so a downstream consumer can reproduce that
    /// signal without parsing the log line. When `true`,
    /// rerunning the capture binary with `CAP_SYS_PTRACE`
    /// (e.g. `sudo setcap cap_sys_ptrace+eip $(which ktstr)`,
    /// or run as root, or `sysctl kernel.yama.ptrace_scope=0`)
    /// resolves most attach failures so jemalloc TSD attach
    /// succeeds across foreign tgids. `false` when
    /// `failed == 0` (no failures to dominate) or when ptrace
    /// failures are strictly less than half of `failed` (the
    /// `>= 50%` gate accepts equality at the boundary).
    ///
    /// Independent of [`Self::dominant_failure`]: ptrace failures
    /// are tallied across both `ptrace-seize` and
    /// `ptrace-interrupt` for the threshold, while
    /// `dominant_failure` reports a single per-tag plurality.
    /// When ptrace counts split across the two tags,
    /// `privilege_dominant` may be `true` while
    /// `dominant_failure` names a non-ptrace tag that won the
    /// single-tag plurality. Conversely, `dominant_failure` may
    /// name a ptrace tag while `privilege_dominant` is `false`
    /// when ptrace failures are below the 50% threshold.
    pub privilege_dominant: bool,
}

impl CtprofProbeSummary {
    /// Operator-facing remediation hint when ptrace failures
    /// dominate the snapshot. Returns `Some(&'static str)` —
    /// the same `PTRACE_EPERM_HINT` constant the capture
    /// pipeline embeds in its tracing summary line (a one-liner
    /// naming `cap_sys_ptrace` — the `setcap`-form spelling of
    /// the capability — and `kernel.yama.ptrace_scope`), or
    /// `None` when [`Self::privilege_dominant`] is false. Lets a
    /// downstream consumer surface the same fix-it message
    /// without parsing the log line or hand-rolling the gate.
    pub fn remediation_hint(&self) -> Option<&'static str> {
        if self.privilege_dominant {
            Some(PTRACE_EPERM_HINT)
        } else {
            None
        }
    }
}

/// Per-snapshot procfs read-failure statistics. Curated projection
/// of the capture pipeline's internal read-tally — exposes per-file
/// counters and a dominant-failure tag a downstream consumer needs
/// to decide whether the snapshot's procfs-derived fields (CSW,
/// schedstats, IO, etc.) are trustworthy on a given host without
/// scanning every thread for default values.
///
/// The read-failure tally ([`Self::read_failures`] /
/// [`Self::read_failures_by_file`]) is read-level only — it
/// counts failures of `fs::read_to_string` against
/// `/proc/<tgid>/task/<tid>/<file>`, not per-field parse failures
/// inside an otherwise-readable file.
/// A present-but-malformed file (e.g. a corrupt `stat` whose
/// `parse_stat` returns all-`None`) does NOT count: the file read
/// succeeded so the tally stays at zero for that category, even
/// though the per-field parsers fold every value to its absent-
/// counter default. Read failures correspond to the kernel never
/// having written the file (ENOENT / kernel without
/// `CONFIG_SCHEDSTATS`), the file disappearing mid-capture (race),
/// or any other I/O-level error from the procfs reader. A snapshot
/// with 1 K schedstat failures across 1 K tids implies a kernel
/// build without `CONFIG_SCHEDSTATS`; 47 stat failures across 1 K
/// tids implies mid-capture races.
///
/// One parse-level signal IS surfaced separately:
/// [`Self::negative_dotted_values`] counts the per-line cases in
/// `/proc/<tid>/sched` where the kernel's PN_SCHEDSTAT format
/// emitted a leading `-` — a rare but observable clock-skew /
/// suspend-resume artifact that the parser otherwise folds
/// silently to zero. Other forms of per-field corruption (
/// non-numeric fractional, malformed key, …) stay outside this
/// summary's scope and surface as zero values on the affected
/// `ThreadState` fields.
///
/// Per-file tokens in [`Self::read_failures_by_file`] are stable
/// kebab-case identifiers downstream consumers match against. The
/// recognized set: `"stat"`, `"schedstat"`, `"io"`, `"status"`,
/// `"sched"`, `"cgroup"`, `"smaps_rollup"`. Adding a new procfs
/// file to the capture adds a new key; the wire shape carries
/// any token the capture emitted, so a consumer that only knows
/// the existing set absorbs new keys without breaking.
///
/// Ghost-filtered tids do NOT contribute to `read_failures` /
/// `read_failures_by_file` — their pending failure bumps are
/// unwound via `discard_pending` when a thread ends up filtered
/// out of `threads` (empty comm + zero start_time), so a busy
/// host with mid-capture exits doesn't inflate the failure tallies
/// with counts that would correspond to threads the snapshot
/// doesn't even contain. `tids_walked` still counts every walk
/// attempt regardless of the ghost filter outcome.
///
/// # Examples
///
/// ```no_run
/// let snap = ktstr::ctprof::capture();
/// if let Some(ps) = &snap.parse_summary
///     && let Some(hint) = ps.kernel_config_hint()
/// {
///     eprintln!("{hint}");
/// }
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CtprofParseSummary {
    /// Total tids the capture pass attempted to read across every
    /// tgid. Non-zero whenever the capture walked any tid; the
    /// denominator a downstream consumer uses to compute "what
    /// fraction of reads failed" without parsing the operator-
    /// facing tracing line.
    pub tids_walked: u64,
    /// Total file-level read failures across all categories. Sum
    /// of [`Self::read_failures_by_file`] values.
    pub read_failures: u64,
    /// Per-file-kind failure tally, keyed by stable kebab tokens
    /// (`"stat"`, `"schedstat"`, `"io"`, `"status"`, `"sched"`,
    /// `"cgroup"`, `"smaps_rollup"`). Empty map when the capture
    /// saw zero failures. Keys present in the map have non-zero
    /// counts; absent keys imply zero failures for that category,
    /// NOT "category unknown".
    pub read_failures_by_file: BTreeMap<String, u64>,
    /// Tag string for the file kind with the most read failures
    /// across the snapshot. `None` when `read_failures == 0`.
    /// Stable kebab tokens (the same vocabulary
    /// [`Self::read_failures_by_file`] keys against). Ties resolve
    /// REVERSE-alphabetically so the output is deterministic — the
    /// alphabetically-earlier tag wins (e.g. `"io"` beats
    /// `"status"` when both count equal).
    pub dominant_read_failure: Option<String>,
    /// `true` when ≥ 50% of `read_failures` are concentrated in
    /// kernel-config-gated files (`"schedstat"`, `"io"`). These
    /// two files are absent on kernels built without
    /// `CONFIG_SCHEDSTATS` / `CONFIG_TASK_IO_ACCOUNTING`
    /// respectively, so a dominance signal here points the
    /// operator at a kernel build/config issue rather than a
    /// transient race or permission problem. `false` when
    /// `read_failures == 0` or when failures are spread across
    /// non-kconfig files.
    pub kernel_config_dominant: bool,
    /// Number of `/proc/<tid>/sched` PN_SCHEDSTAT dotted-ns
    /// values whose integer part read as negative (kernel emitted
    /// a leading `-`, e.g. `-5.000000`). The capture-side parser
    /// (`parsed_ns_from_dotted`) rejects negative integer parts —
    /// a `u64` parse cannot accept the sign — and the call site
    /// then `unwrap_or(0)`s the resulting `None` per the
    /// best-effort capture contract. Without this counter the
    /// silent fold to zero leaves operators with no visibility
    /// into the rate at which schedstat values were silently
    /// truncated.
    ///
    /// Counts per-field-occurrence, NOT per-thread: a single
    /// tid that exposed five negative dotted fields contributes
    /// `5` to this counter (e.g. one tid with negative `wait_sum`,
    /// `sleep_max`, `block_sum`, `iowait_sum`, and `exec_max`
    /// adds 5). The denominator for "fraction of tids affected"
    /// is therefore NOT this field — pair with
    /// [`Self::tids_walked`] only as an upper bound on
    /// affected-tid count.
    ///
    /// Distinct from [`Self::read_failures`]: a negative dotted
    /// value comes from a `sched` file that READ successfully —
    /// it is a parse-level signal, not a read-level signal. The
    /// field stays at zero on a clean host because the kernel
    /// emits non-negative values on every well-behaved schedstat
    /// path; non-zero values are most commonly the result of
    /// clock-skew on suspend/resume where a `delta` calculation
    /// against a stale baseline lands negative.
    ///
    /// Ghost-filter discipline: per-tid bumps are held pending
    /// (alongside the read-failure bumps in
    /// [`crate::ctprof`]'s capture-side `ParseTally`), and
    /// unwound via `discard_pending` when the surrounding tid is
    /// rejected by the empty-comm + zero-start ghost filter so a
    /// busy host with mid-capture exits doesn't inflate this
    /// counter with bumps that correspond to threads the snapshot
    /// doesn't even contain.
    pub negative_dotted_values: u64,
}

impl CtprofParseSummary {
    /// Operator-facing hint when kernel-config-gated file failures
    /// dominate the snapshot. Returns `Some(&'static str)` naming
    /// the two `CONFIG_*` knobs that gate the affected files
    /// (`CONFIG_SCHEDSTATS` for `schedstat`, `CONFIG_TASK_IO_ACCOUNTING`
    /// for `io`), or `None` when [`Self::kernel_config_dominant`]
    /// is `false`. Lets a downstream consumer surface a remediation
    /// pointer without parsing the log line or hand-rolling the
    /// gate, mirroring the [`CtprofProbeSummary::remediation_hint`]
    /// pattern.
    pub fn kernel_config_hint(&self) -> Option<&'static str> {
        if self.kernel_config_dominant {
            Some(PARSE_KCONFIG_HINT)
        } else {
            None
        }
    }
}

/// Stable kernel-config remediation hint for parse summaries.
/// Names the two procfs files that disappear on kernels built
/// without the corresponding `CONFIG_*` knobs.
const PARSE_KCONFIG_HINT: &str = "hint: schedstat / io read failures dominate — \
                                  kernel may be built without CONFIG_SCHEDSTATS \
                                  and/or CONFIG_TASK_IO_ACCOUNTING";

/// Absent-value sentinel for [`ThreadState::state`]. Used by both
/// the manual [`Default`] impl on [`ThreadState`] and the
/// `serde(default = ...)` attribute on the field so the absent
/// state is `'~'` regardless of how a [`ThreadState`] gets
/// constructed (default-built test fixture, partial JSON
/// deserialize, capture-time `unwrap_or` fallback).
///
/// `'~'` (U+007E = 126) is chosen specifically because it sorts
/// strictly AFTER every entry in `fs/proc/array.c::task_state_array`
/// — `R` (82), `S` (83), `D` (68), `T` (84), `t` (116), `X`
/// (88), `Z` (90), `P` (80), `I` (73) all have lower codepoints.
/// [`crate::ctprof_compare::aggregate`] breaks the
/// categorical-mode count-ties (rules
/// [`crate::ctprof_compare::AggRule::Mode`] /
/// [`crate::ctprof_compare::AggRule::ModeChar`] /
/// [`crate::ctprof_compare::AggRule::ModeBool`]) toward the
/// LEX-SMALLEST candidate (the closure
/// `a.1.cmp(&b.1).then(b.0.cmp(&a.0))` inside the
/// `Modeable::mode_across` reduction), so a sentinel smaller
/// than the real letters would HIJACK the tiebreak whenever a
/// default-built thread sat alongside a real one in the same
/// group. `'~'` is larger than all of them, so the real kernel
/// letter always wins the tie.
///
/// `'?'` (U+003F = 63) was the obvious-looking pick but is
/// numerically SMALLER than every state letter the kernel
/// emits, which would make it a tiebreak hijacker rather than
/// a safe sentinel. Avoid.
fn default_state_char() -> char {
    '~'
}

/// Per-thread resource profile.
///
/// Populated by the capture layer from `/proc/<tid>/{sched,status,
/// io,stat,comm,cgroup}`, `sched_getaffinity`, the taskstats
/// genetlink path (delay-accounting + memory-watermark fields),
/// and (for jemalloc-linked processes only, via ptrace +
/// `process_vm_readv`) the per-thread `tsd_s.thread_allocated` /
/// `thread_deallocated` TLS counters.
///
/// Field families (mirrors the module-level breakdown, with
/// the registry-pairing reductions named):
///
/// - **Cumulative counters and totals** (the majority): wakeups,
///   migrations, csw, run/wait/sleep/block/iowait time,
///   schedstat counts, page-fault counters, syscall counters,
///   byte counters, the taskstats per-bucket `*_count` and
///   `*_delay_total_ns`, and the jemalloc per-thread
///   allocated/deallocated TSD counters. Probe-timing invariant
///   modulo monotonic forward progress; reduced via the
///   `Sum*` rules.
/// - **Lifetime high-water peaks**: schedstat `*_max` family,
///   every taskstats `*_delay_max_ns` / `*_delay_min_ns`, and
///   the memory watermarks ([`Self::hiwater_rss_bytes`],
///   [`Self::hiwater_vm_bytes`]). Non-decreasing-over-time but
///   per-event extrema, so non-summable across threads; the
///   registry reduces them via `MaxPeak` / `MaxPeakBytes`.
/// - **Instantaneous gauges** (sensitive to probe timing):
///   [`Self::nr_threads`] (signal_struct->nr_threads snapshot),
///   [`Self::fair_slice_ns`] (instantaneous `p->se.slice`),
///   and [`Self::state`] (task_state_array letter). Two probes
///   of the same thread at different instants can legitimately
///   produce different values. Reduced via `MaxGaugeCount` /
///   `MaxGaugeNs` / `ModeChar`.
/// - **Categorical / ordinal scalars** (point-in-time
///   snapshots): [`Self::policy`], [`Self::nice`],
///   [`Self::priority`], [`Self::processor`],
///   [`Self::rt_priority`], plus the identity strings
///   ([`Self::pcomm`], [`Self::comm`], [`Self::cgroup`]) and
///   the [`crate::metric_types::CpuSet`]
///   [`Self::cpu_affinity`]. Sampled at capture time and can
///   change at runtime (e.g. `sched_setaffinity` mid-run flips
///   `processor` and `cpu_affinity`); reduced via `Mode*` /
///   `Range*` / `Affinity`.
///
/// Same family taxonomy as the module-level block at the top of
/// the file; the per-field docs flag the family on each entry
/// and the registry's [`AggRule`] pairing makes the
/// "category-mismatched aggregation is a compile error"
/// invariant load-bearing.
///
/// [`AggRule`]: crate::ctprof_compare::AggRule
///
/// `Default` is implemented manually rather than derived because
/// the [`Self::state`] field needs `'~'` (the absent-value
/// sentinel) instead of `'\0'` (the `char` Default). See the
/// field doc on [`Self::state`] for why: `'\0'` lex-compares
/// SMALLER than every real kernel state letter, which would
/// poison [`crate::ctprof_compare::AggRule::ModeChar`]
/// tie-breaks toward "absent" whenever a default-constructed
/// thread sat alongside a real one in a group.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ThreadState {
    // -- identity --
    /// Kernel task id. Ephemeral across runs; not used as a
    /// grouping axis.
    pub tid: u32,
    /// Thread group id (process id). Ephemeral across runs.
    pub tgid: u32,
    /// Process name, read from `/proc/<tgid>/comm`. Stable across
    /// runs on the same build. Feeds the grouping key under
    /// `--group-by pcomm` (default), where it flows through the
    /// token-based [`crate::ctprof_compare::pattern_key`]
    /// normalizer so ephemeral worker pools (`worker-0`,
    /// `worker-1`, ...) collapse into a single `worker-{N}`
    /// bucket; pass `--no-thread-normalize` to group by literal
    /// pcomm. Also feeds the smaps_rollup join key (with the same
    /// normalization rules) so per-process memory rows survive
    /// PID churn across snapshots.
    pub pcomm: String,
    /// Thread name, read from `/proc/<tid>/comm`. Stable when the
    /// runtime assigns deterministic names (worker pools, async
    /// runtimes). Feeds the grouping key under `--group-by comm`,
    /// where it flows through the token-based
    /// [`crate::ctprof_compare::pattern_key`] normalizer (same
    /// rules as pcomm). Pass `--no-thread-normalize` to group by
    /// literal comm, or `--group-by comm-exact` for the same
    /// effect on this axis only (smaps still normalizes).
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
    pub cgroup: String,
    /// `/proc/<tid>/stat` field 22 (`start_time`) in USER_HZ
    /// clock ticks since system boot. The kernel exports this
    /// field in USER_HZ units (defined in
    /// `include/asm-generic/param.h` as `USER_HZ == 100` on
    /// every architecture the capture layer targets — x86_64
    /// and aarch64) — NOT raw internal jiffies, which scale
    /// with CONFIG_HZ. Cross-host comparison between x86_64 and
    /// aarch64 is meaningful because USER_HZ is the same 100 on
    /// both, so a diff between two hosts on different CONFIG_HZ
    /// settings still compares correctly. Seconds-since-boot
    /// is simply `start_time_clock_ticks / 100` on those
    /// architectures. Other in-tree architectures carry
    /// different USER_HZ (alpha defines 1024, for instance);
    /// a future port must either restate the divisor or
    /// normalise at capture time. `fs/proc/array.c::do_task_stat`
    /// is where the kernel writes the field to procfs.
    ///
    /// Stored as raw `u64`, NOT wrapped in
    /// [`crate::metric_types::ClockTicks`], because this field
    /// is an identity / ghost-thread sentinel rather than a
    /// metric that flows through the aggregation pipeline. The
    /// ghost-filter in `capture_with` / `capture_pid_with`
    /// keys on `start_time_clock_ticks == 0` (alongside an
    /// empty `comm`) to drop `ThreadState`s assembled from a
    /// tid that exited mid-capture, which is cleaner against a
    /// raw `u64` than against a wrapped sentinel.
    pub start_time_clock_ticks: u64,
    /// Scheduling policy (SCHED_OTHER, SCHED_FIFO, SCHED_RR,
    /// SCHED_BATCH, SCHED_IDLE, SCHED_DEADLINE, SCHED_EXT). Stored
    /// as the canonical name string rather than the kernel
    /// integer so comparison output is human-readable without a
    /// reverse-lookup table. Wrapped in
    /// [`crate::metric_types::CategoricalString`] so the
    /// aggregation pipeline reduces by mode (most-frequent value)
    /// rather than a category-mismatched sum or max.
    pub policy: crate::metric_types::CategoricalString,
    /// Nice value in the standard [-20, 19] range. Signed i32
    /// because the range includes negative values and
    /// [`parse_stat`] extracts the field via `get_i32` on
    /// procfs's decimal text — the inner type matches the
    /// extraction path and the kernel-visible range without
    /// coercion. Wrapped in [`crate::metric_types::OrdinalI32`]
    /// so the aggregation pipeline reduces by `[min, max]` range
    /// rather than sum.
    pub nice: crate::metric_types::OrdinalI32,
    /// Allowed CPU set from `sched_getaffinity`. Sorted ascending.
    /// Comparison aggregates via union across the group and
    /// renders as "N cpus (range)" or "mixed" for heterogeneous
    /// sets — see [`crate::ctprof_compare::AffinitySummary`].
    /// Wrapped in [`crate::metric_types::CpuSet`] so the
    /// aggregation pipeline routes through the dedicated
    /// affinity-summary reduction rather than a numeric path.
    pub cpu_affinity: crate::metric_types::CpuSet,

    // -- task state (last-CPU, run-state) --
    /// Last CPU the thread executed on. `/proc/<tid>/stat` field
    /// 39 (`task_cpu(task)` in `fs/proc/array.c::do_task_stat`,
    /// emitted via `seq_put_decimal_ll`). Signed for symmetry
    /// with [`Self::nice`]; the kernel emits non-negative values
    /// only — `task_cpu` (defined `unsigned int` in
    /// `include/linux/sched.h`) zero-extends through the
    /// `seq_put_decimal_ll` widening to `s64`. `0` is the
    /// absent-value default (collisions with a legitimate CPU 0
    /// are distinguished by inspecting `cpu_affinity`).
    /// Wrapped in [`crate::metric_types::OrdinalI32`] so the
    /// aggregation pipeline reduces by `[min, max]` range across
    /// the group.
    pub processor: crate::metric_types::OrdinalI32,
    /// Single-letter task state from `/proc/<tid>/status` `State:`
    /// line. Real kernel chars are `R`, `S`, `D`, `T`, `t`, `X`,
    /// `Z`, `P`, `I` (see `fs/proc/array.c::task_state_array`,
    /// emitted via `get_task_state`). `'~'` is the absent-value
    /// sentinel — visually distinct from every real kernel char
    /// so a downstream consumer can distinguish "no state read"
    /// from a real value. When `'~'` appears in compare output,
    /// the `/proc/<tid>/status` read failed (thread likely
    /// exited mid-capture).
    ///
    /// `ThreadState::default()`, the capture-time
    /// `unwrap_or_else(default_state_char)` fallback, and
    /// `serde(default)` deserialize of a partial JSON record all
    /// produce `'~'` (NOT `'\0'`, the bare `char` Default). The
    /// manual `Default` impl on `ThreadState`, the
    /// `unwrap_or_else` site in `capture_thread_at_with_tally`,
    /// and the `serde(default = ...)` attribute on this field
    /// are paired specifically so the absent-value sentinel is
    /// the same byte everywhere.
    ///
    /// `'~'` (U+007E = 126) is chosen so it sorts AFTER every
    /// real kernel state letter — `R` (82), `S` (83), `D` (68),
    /// `T` (84), `t` (116), `X` (88), `Z` (90), `P` (80), `I`
    /// (73). [`crate::ctprof_compare::AggRule::ModeChar`]
    /// breaks count-ties toward the LEX-SMALLEST candidate, so
    /// a sentinel smaller than the real letters would silently
    /// elect "absent" whenever a default-built thread sat
    /// alongside a real one in the same group. `'~'` being
    /// larger than all of them lets the real letter win the
    /// tie. The earlier `'?'` (U+003F = 63) sentinel was
    /// numerically smaller than every real state letter — a
    /// tiebreak hijacker; do not return to it.
    #[serde(default = "default_state_char")]
    pub state: char,

    // -- scheduling (cumulative + lifetime peaks; /proc/<tid>/sched, needs CONFIG_SCHED_DEBUG) --
    // -- (sched_ext gate: ext.enabled requires CONFIG_SCHED_CLASS_EXT) --
    /// `true` when the task is currently scheduled by sched_ext —
    /// `/proc/<tid>/sched` `ext.enabled` line. The kernel emits
    /// the literal key `ext.enabled` only when
    /// `CONFIG_SCHED_CLASS_EXT` is enabled; on kernels without it
    /// the field is absent and lands at the default `false`. When
    /// `false` on a task expected under sched_ext, the task may
    /// have been ejected (sched_ext fall-back to CFS on BPF error)
    /// or never enrolled.
    ///
    /// Stays a bare `bool` — not wrapped in a categorical newtype
    /// — because it is the only bool-valued metric in the
    /// registry. The
    /// [`crate::ctprof_compare::AggRule::ModeBool`] dispatch
    /// coerces it to a `String` via `to_string()`/`Display` at
    /// the call site (see the
    /// [`crate::metric_types::CategoricalString`] doc note: if a
    /// second bool-valued metric appears, promote both to a
    /// dedicated `CategoricalBool` wrapper rather than keeping
    /// the ad-hoc coercion).
    pub ext_enabled: bool,
    /// Cumulative on-CPU time, ns; `/proc/<tid>/schedstat`
    /// field 1. `MonotonicNs` per the lifetime-accumulator
    /// contract.
    pub run_time_ns: crate::metric_types::MonotonicNs,
    /// Cumulative time waiting on the runqueue, ns;
    /// `/proc/<tid>/schedstat` field 2. `MonotonicNs`.
    pub wait_time_ns: crate::metric_types::MonotonicNs,
    /// Number of times the task was scheduled onto a CPU;
    /// `/proc/<tid>/schedstat` field 3. `MonotonicCount`.
    pub timeslices: crate::metric_types::MonotonicCount,
    /// Voluntary context switches — task gave up the CPU itself;
    /// `/proc/<tid>/status` `voluntary_ctxt_switches`.
    /// `MonotonicCount`.
    pub voluntary_csw: crate::metric_types::MonotonicCount,
    /// Involuntary context switches — task was preempted;
    /// `/proc/<tid>/status` `nonvoluntary_ctxt_switches`.
    /// `MonotonicCount`.
    pub nonvoluntary_csw: crate::metric_types::MonotonicCount,
    /// Total wakeups via `try_to_wake_up()`; `/proc/<tid>/sched`
    /// `nr_wakeups`. `MonotonicCount`.
    pub nr_wakeups: crate::metric_types::MonotonicCount,
    /// Wakeups landed on the same CPU as the waker;
    /// `/proc/<tid>/sched` `nr_wakeups_local`. `MonotonicCount`.
    pub nr_wakeups_local: crate::metric_types::MonotonicCount,
    /// Wakeups landed on a different CPU than the waker;
    /// `/proc/<tid>/sched` `nr_wakeups_remote`. `MonotonicCount`.
    pub nr_wakeups_remote: crate::metric_types::MonotonicCount,
    /// `WF_SYNC` synchronous-wakeup hint count;
    /// `/proc/<tid>/sched` `nr_wakeups_sync`. `MonotonicCount`.
    pub nr_wakeups_sync: crate::metric_types::MonotonicCount,
    /// Wakeups where the task migrated to a different CPU than
    /// its prior one (`WF_MIGRATED`); `/proc/<tid>/sched`
    /// `nr_wakeups_migrate`. Distinct from `nr_wakeups_remote`
    /// (waker CPU != target CPU). `MonotonicCount`.
    pub nr_wakeups_migrate: crate::metric_types::MonotonicCount,
    /// Wakeups onto this CPU (cache-affine wakeup
    /// fast-path). `/proc/<tid>/sched` `nr_wakeups_affine`,
    /// emitted via `P_SCHEDSTAT`. Plain u64. Zero on kernels
    /// without `CONFIG_SCHEDSTATS`. Zero under sched_ext:
    /// `wake_affine` is a CFS-only path.
    pub nr_wakeups_affine: crate::metric_types::MonotonicCount,
    /// Total invocations of the cache-affine wakeup heuristic
    /// `wake_affine()` — denominator for the affine-wake success
    /// ratio (`nr_wakeups_affine / nr_wakeups_affine_attempts`).
    /// `/proc/<tid>/sched` `nr_wakeups_affine_attempts`, emitted
    /// via `P_SCHEDSTAT` (plain u64). The kernel increments this
    /// counter unconditionally on every `wake_affine()` call in
    /// `kernel/sched/fair.c::wake_affine`, then increments
    /// `nr_wakeups_affine` only when the heuristic chose this
    /// CPU — so the ratio is the success rate of the cache-
    /// affine fast-path. Zero on kernels without
    /// `CONFIG_SCHEDSTATS`. Zero under sched_ext: `wake_affine`
    /// is a CFS-only path and `kernel/sched/ext.c` does not
    /// increment this counter.
    pub nr_wakeups_affine_attempts: crate::metric_types::MonotonicCount,
    /// Total cross-CPU migrations of the task. Incremented
    /// unconditionally at `kernel/sched/core.c` (`p->se.nr_migrations++`)
    /// — no schedstat macro, no class gating. Always populated
    /// regardless of `CONFIG_SCHEDSTATS` or scheduling class.
    /// `MonotonicCount`.
    pub nr_migrations: crate::metric_types::MonotonicCount,
    /// Migrations forced by load balance (the load balancer
    /// migrated the task even though the local heuristic would
    /// have skipped it). `/proc/<tid>/sched` `nr_forced_migrations`,
    /// plain u64 via `P_SCHEDSTAT`. Zero on kernels without
    /// `CONFIG_SCHEDSTATS`.
    pub nr_forced_migrations: crate::metric_types::MonotonicCount,
    /// Failed migrations attributed to affinity mismatch — the
    /// destination CPU was not in `cpus_allowed`. `/proc/<tid>/sched`
    /// `nr_failed_migrations_affine`, plain u64 via `P_SCHEDSTAT`.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`.
    pub nr_failed_migrations_affine: crate::metric_types::MonotonicCount,
    /// Failed migrations attributed to the task being currently
    /// running on the source CPU. `/proc/<tid>/sched`
    /// `nr_failed_migrations_running`, plain u64 via `P_SCHEDSTAT`.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`.
    pub nr_failed_migrations_running: crate::metric_types::MonotonicCount,
    /// Failed migrations attributed to cache-hot heuristic — the
    /// source CPU's cache was too hot to leave. `/proc/<tid>/sched`
    /// `nr_failed_migrations_hot`, plain u64 via `P_SCHEDSTAT`.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`.
    pub nr_failed_migrations_hot: crate::metric_types::MonotonicCount,
    /// Total nanoseconds the task spent on the runqueue waiting
    /// to be picked. Populated from `/proc/<tid>/sched`'s
    /// `wait_sum` key — kernel emits via `PN_SCHEDSTAT` as
    /// `ms.ns_remainder`, reconstructed by the parser to full ns.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`. Zero under
    /// sched_ext: the kernel updates this counter via
    /// `__update_stats_wait_end` (`kernel/sched/stats.c`), called
    /// from CFS/RT/DL paths only — `kernel/sched/ext.c` does not
    /// call that helper.
    pub wait_sum: crate::metric_types::MonotonicNs,
    /// Number of runqueue-wait windows the task accumulated —
    /// the per-event tally that pairs with [`Self::wait_sum`].
    /// Populated from `/proc/<tid>/sched`'s `wait_count` key
    /// (kernel emits as `P_SCHEDSTAT`, plain u64). Zero on
    /// kernels without `CONFIG_SCHEDSTATS`. Same write path as
    /// `wait_sum` (`__update_stats_wait_end` in
    /// `kernel/sched/stats.c`), so the same sched_ext caveat
    /// applies: zero under sched_ext.
    pub wait_count: crate::metric_types::MonotonicCount,
    /// Longest single runqueue-wait window the task ever
    /// experienced, in nanoseconds. `/proc/<tid>/sched` `wait_max`
    /// emitted via `PN_SCHEDSTAT` (`ms.ns_remainder`,
    /// reconstructed to full ns by the parser). Tail-latency
    /// signal that pairs with the `wait_sum` average. Zero on
    /// kernels without `CONFIG_SCHEDSTATS`. Zero under sched_ext:
    /// the kernel sets this counter via
    /// `__update_stats_wait_end` from CFS/RT/DL paths only —
    /// `kernel/sched/ext.c` does not call that helper, so
    /// sched_ext-managed tasks never accumulate wait_max.
    pub wait_max: crate::metric_types::PeakNs,
    /// Pure voluntary sleep time, nanoseconds — `TASK_INTERRUPTIBLE`
    /// off-CPU windows only, with the involuntary-block
    /// component already subtracted at capture.
    ///
    /// Computed at capture as `sum_sleep_runtime - sum_block_runtime`
    /// (saturating; the read-skew window where block briefly
    /// exceeds sleep collapses to zero). The kernel's
    /// `sum_sleep_runtime` key (read via `PN_SCHEDSTAT` in
    /// `/proc/<tid>/sched`) is the FULL off-CPU total because
    /// `__update_stats_enqueue_sleeper` (`kernel/sched/stats.c`)
    /// charges every sleeper window regardless of which sleep
    /// state the task was in — voluntary sleep AND involuntary
    /// block both contribute. Subtracting `sum_block_runtime`
    /// at capture leaves the voluntary-sleep residual, which
    /// is the operationally useful signal for "how much time
    /// did this task spend on a syscall wait that wasn't a
    /// kernel block."
    ///
    /// Capture-side normalization (rather than a derived
    /// metric at compare time) means every consumer sees the
    /// pre-normalized value without re-deriving — and the raw
    /// kernel reading is intentionally NOT preserved in the
    /// snapshot per the project's pre-1.0 disposable-sidecar
    /// policy.
    ///
    /// There is no `voluntary_sleep_count` counterpart: the
    /// kernel does not emit one — the scheduler records the
    /// aggregate runtime but not the sleep-event count
    /// separately from `nr_wakeups`, which already covers the
    /// wake-side tally.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`. Zero under
    /// sched_ext: `__update_stats_enqueue_sleeper` is called
    /// from CFS/RT/DL paths only. Also zero when either
    /// `sum_sleep_runtime` or `sum_block_runtime` fails to parse
    /// from `/proc/<tid>/sched`: the residual is uncomputable
    /// without both halves, and falling back to the unsubtracted
    /// `sum_sleep_runtime` would mislabel involuntary block as
    /// voluntary sleep.
    pub voluntary_sleep_ns: crate::metric_types::MonotonicNs,
    /// Longest single sleep window in nanoseconds.
    /// `/proc/<tid>/sched` `sleep_max` emitted via `PN_SCHEDSTAT`
    /// (`ms.ns_remainder`, reconstructed by the parser). Zero on
    /// kernels without `CONFIG_SCHEDSTATS`. Zero under sched_ext:
    /// the kernel sets this counter via
    /// `__update_stats_enqueue_sleeper` from CFS/RT/DL paths
    /// only.
    pub sleep_max: crate::metric_types::PeakNs,
    /// Total nanoseconds blocked in the scheduler — every path
    /// that puts the task into `TASK_UNINTERRUPTIBLE` contributes:
    /// swap-in, page-fault resolution, disk I/O, plus
    /// mutex/rwsem/completion waits inside kernel code that
    /// hold the task off the runqueue. Populated from
    /// `/proc/<tid>/sched`'s `sum_block_runtime` key (kernel
    /// emits `ms.ns_remainder` via `PN_SCHEDSTAT`; the parser
    /// reconstructs full ns). `block_sum - iowait_sum` is
    /// therefore an UPPER BOUND on non-iowait involuntary-block
    /// time — swap/zswap decompression contributes, but so do
    /// the lock-family waits, so the delta cannot be read as
    /// swap latency without further attribution. There is no
    /// `block_count` counterpart: the kernel does not emit one.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`. Zero under
    /// sched_ext: the kernel updates this counter via
    /// `__update_stats_enqueue_sleeper` (`kernel/sched/stats.c`),
    /// called from CFS/RT/DL paths only.
    pub block_sum: crate::metric_types::MonotonicNs,
    /// Longest single block window in nanoseconds.
    /// `/proc/<tid>/sched` `block_max` emitted via `PN_SCHEDSTAT`
    /// (`ms.ns_remainder`, reconstructed by the parser). Tail-
    /// latency signal that pairs with the `block_sum` average.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`. Zero under
    /// sched_ext: the kernel sets this counter via
    /// `__update_stats_enqueue_sleeper` from CFS/RT/DL paths
    /// only.
    pub block_max: crate::metric_types::PeakNs,
    /// Total nanoseconds in I/O wait specifically (subset of
    /// `block_sum`). Distinguishes disk-backed I/O delay from
    /// the full involuntary-block total — callers that want
    /// disk latency alone read this field, callers that want
    /// every blocked window read `block_sum`. Populated from
    /// `/proc/<tid>/sched`'s `iowait_sum` key (kernel emits
    /// `ms.ns_remainder` via `PN_SCHEDSTAT`; the parser
    /// reconstructs full ns). Zero on kernels without
    /// `CONFIG_SCHEDSTATS`. Zero under sched_ext: the kernel
    /// updates this counter via `__update_stats_enqueue_sleeper`
    /// (`kernel/sched/stats.c`), called from CFS/RT/DL paths
    /// only.
    pub iowait_sum: crate::metric_types::MonotonicNs,
    /// Number of I/O-wait windows the task accumulated — the
    /// per-event tally that pairs with [`Self::iowait_sum`].
    /// Populated from `/proc/<tid>/sched`'s `iowait_count` key
    /// (kernel emits as `P_SCHEDSTAT`, plain u64). Zero on
    /// kernels without `CONFIG_SCHEDSTATS`. Same write path as
    /// `iowait_sum` (`__update_stats_enqueue_sleeper` in
    /// `kernel/sched/stats.c`), so the same sched_ext caveat
    /// applies: zero under sched_ext.
    pub iowait_count: crate::metric_types::MonotonicCount,
    /// Longest single CPU-burst (run-without-preempt window) in
    /// nanoseconds. `/proc/<tid>/sched` `exec_max` emitted via
    /// `PN_SCHEDSTAT` (`ms.ns_remainder`, reconstructed by the
    /// parser). Zero on kernels without `CONFIG_SCHEDSTATS`.
    /// Updated for sched_ext tasks too: the kernel sets it in
    /// `update_se` (`kernel/sched/fair.c`), which sched_ext
    /// reaches via `update_curr_scx` → `update_curr_common`.
    pub exec_max: crate::metric_types::PeakNs,
    /// Longest scheduling slice the task got before being
    /// preempted, in nanoseconds. `/proc/<tid>/sched` `slice_max`
    /// emitted via `PN_SCHEDSTAT` (`ms.ns_remainder`,
    /// reconstructed by the parser). Zero on kernels without
    /// `CONFIG_SCHEDSTATS`. Zero under sched_ext: the kernel sets
    /// this counter only in `set_next_entity`
    /// (`kernel/sched/fair.c`), a CFS-only path —
    /// sched_ext-managed tasks never accumulate slice_max even
    /// when CONFIG_SCHEDSTATS is enabled.
    pub slice_max: crate::metric_types::PeakNs,

    // -- jemalloc per-thread TSD counters (tsd_s.thread_allocated / thread_deallocated, via ptrace) --
    /// Bytes allocated by this thread over its lifetime — read
    /// directly from jemalloc's per-thread TSD u64 counter
    /// (`tsd_s.thread_allocated`) via ptrace + `process_vm_readv`.
    /// Cumulative-from-thread-creation; jemalloc updates the
    /// per-thread TSD counters unconditionally on its alloc fast
    /// and slow paths, so attaching the probe late does not lose
    /// data.
    ///
    /// Distinct from [`crate::host_heap::HostHeapState::allocated_bytes`],
    /// which is the runner process's own
    /// `tikv_jemalloc_ctl::stats::allocated` reading — a global
    /// arena counter for the calling process. This field is the
    /// per-thread TSD counter for an arbitrary target thread the
    /// probe attached to.
    ///
    /// Zero when the capture layer could not pull the counter:
    /// (a) the target process is not linked against jemalloc,
    /// (b) the probe attach failed for any other reason (DWARF
    /// missing, jemalloc in a DSO rather than the main
    /// executable, arch mismatch),
    /// (c) the per-thread ptrace step failed (tid exited
    /// mid-capture, EPERM under YAMA scope=1 without
    /// `CAP_SYS_PTRACE`),
    /// or (d) the thread is in the calling process's own tgid
    /// (PTRACE_SEIZE rejects self-attach). All four collapse to
    /// zero per the best-effort "absent = 0" capture contract.
    /// Snapshot-level diagnosis lives on
    /// [`CtprofProbeSummary::dominant_failure`] (the per-tag
    /// plurality) and
    /// [`CtprofProbeSummary::privilege_dominant`] (the EPERM
    /// remediation gate, true when ptrace tags account for ≥ 50%
    /// of `failed`), reachable via
    /// [`CtprofSnapshot::probe_summary`]; the per-tag taxonomy
    /// is documented in the `ktstr ctprof capture` CLI help.
    pub allocated_bytes: crate::metric_types::Bytes,
    /// Bytes freed by this thread over its lifetime — read from
    /// jemalloc's per-thread TSD u64 counter
    /// (`tsd_s.thread_deallocated`) via the same probe path that
    /// populates [`Self::allocated_bytes`].
    /// `allocated_bytes - deallocated_bytes` is a thread-local
    /// estimate of currently-held bytes; the difference races
    /// any in-flight allocator activity since the two counters
    /// are sampled in one `process_vm_readv` over a 24-byte span
    /// the target may continue to mutate during the read.
    pub deallocated_bytes: crate::metric_types::Bytes,

    // -- procfs /proc/<tid>/stat: page faults + CPU time (fields 10, 12, 14, 15) --
    /// Minor faults (no disk I/O). `/proc/<tid>/stat` field 10.
    pub minflt: crate::metric_types::MonotonicCount,
    /// Major faults (backed by disk). `/proc/<tid>/stat` field 12.
    pub majflt: crate::metric_types::MonotonicCount,
    /// User-mode CPU time in USER_HZ clock ticks since thread
    /// start. `/proc/<tid>/stat` field 14
    /// (`nsec_to_clock_t(utime)` in `fs/proc/array.c::do_task_stat`).
    /// USER_HZ-scaled like [`Self::start_time_clock_ticks`] —
    /// cross-host comparison between x86_64 and aarch64 is
    /// meaningful because USER_HZ is 100 on both, independent of
    /// CONFIG_HZ. Suffix `_clock_ticks` mirrors the existing
    /// `start_time_clock_ticks` precedent.
    pub utime_clock_ticks: crate::metric_types::ClockTicks,
    /// Kernel-mode CPU time in USER_HZ clock ticks since thread
    /// start. `/proc/<tid>/stat` field 15
    /// (`nsec_to_clock_t(stime)` in `fs/proc/array.c::do_task_stat`).
    /// Same USER_HZ scaling and `_clock_ticks` suffix convention as
    /// [`Self::utime_clock_ticks`].
    pub stime_clock_ticks: crate::metric_types::ClockTicks,
    /// Kernel-internal scheduler priority (signed). Distinct
    /// from [`Self::nice`] — `priority` is the post-bias
    /// scheduling priority (`task_prio(task)`) the scheduler
    /// uses for ordering, while `nice` is the
    /// userspace-presentable [-20, 19] preference.
    /// `/proc/<tid>/stat` field 18, emitted via
    /// `seq_put_decimal_ll(m, " ", task_prio(task))` at
    /// `fs/proc/array.c:602`. Range per `task_prio()` at
    /// `kernel/sched/syscalls.c:170`:
    /// CFS / SCHED_OTHER tasks see `[0..39]` (nice [-20..19]
    /// translated by `task_prio()` returning
    /// `p->prio - MAX_RT_PRIO`); SCHED_FIFO / SCHED_RR tasks
    /// see `[-2..-100]`; SCHED_DEADLINE tasks land at `-101`.
    /// Default 0 when the stat read fails — collides with the
    /// CFS nice-0 case, so a CFS task at default nice and an
    /// absent stat line both render 0. Wrapped in
    /// [`crate::metric_types::OrdinalI32`] for the
    /// `[min, max]` range reduction across a group.
    pub priority: crate::metric_types::OrdinalI32,
    /// Real-time scheduler priority. `/proc/<tid>/stat` field
    /// 40, emitted via `seq_put_decimal_ull(m, " ", task->rt_priority)`
    /// at `fs/proc/array.c:637`. Non-zero only when the task
    /// runs SCHED_FIFO or SCHED_RR; CFS / SCHED_OTHER tasks
    /// land at zero. Useful as a post-hoc filter to identify
    /// real-time threads in a snapshot. Wrapped in
    /// [`crate::metric_types::OrdinalU32`] for the
    /// `[min, max]` range reduction across a group; the inner
    /// `u32` matches the kernel's
    /// `unsigned int task_struct::rt_priority` declaration
    /// (`include/linux/sched.h`) exactly. Practical range is
    /// bounded `0..99` regardless of the type width.
    pub rt_priority: crate::metric_types::OrdinalU32,

    // -- /proc/<tid>/sched additions (counters + ordinal + slice gauge) --
    /// Cumulative time this task forced its SMT sibling idle for
    /// core-scheduling, in nanoseconds. `/proc/<tid>/sched`
    /// `core_forceidle_sum`, dotted ms.ns format via
    /// `PN_SCHEDSTAT` (`kernel/sched/debug.c:1335`).
    /// Reconstructed to full ns via the same
    /// `parsed_ns_from_dotted` helper as `wait_sum` /
    /// `block_sum`.
    ///
    /// Increment occurs in `__account_forceidle_time()` at
    /// `kernel/sched/cputime.c:244` (function defined at
    /// `cputime.c:242`), called from
    /// `__sched_core_account_forceidle()` in
    /// `kernel/sched/core_sched.c:287` (function defined at
    /// `core_sched.c:242`). The increment body is a plain
    /// `__schedstat_add(p->stats.core_forceidle_sum, delta)` —
    /// it is CLASS-AGNOSTIC. The caller iterates
    /// `for_each_cpu(i, smt_mask)` and picks
    /// `p = rq_i->core_pick ?: rq_i->curr` on each SMT sibling,
    /// charging whichever task is running there regardless of
    /// scheduling class. So a SCHED_EXT / DEADLINE / RR / FIFO
    /// task on a core-scheduled SMT cohort CAN accrue forceidle
    /// time the same way a CFS task can.
    ///
    /// Real gating is at the rq/build level, not per-task, and
    /// the runtime gates apply IN SERIES rather than equating —
    /// `sched_core_enabled(rq)` and `core_forceidle_count` are
    /// independent conditions that BOTH have to fire:
    ///
    /// - **Build:** `CONFIG_SCHED_CORE` (file-level `#ifdef` in
    ///   `kernel/sched/cputime.c` and
    ///   `kernel/sched/core_sched.c`).
    /// - **Build:** `CONFIG_SCHEDSTATS` (the caller's own
    ///   `#ifdef CONFIG_SCHEDSTATS` at `core_sched.c:239`).
    /// - **Runtime, scheduler-class entry:**
    ///   `sched_core_enabled(rq)` is the FIRST gate — checked
    ///   at `pick_next_task()` entry at `kernel/sched/core.c:6014`
    ///   with an early `__pick_next_task()` return when false.
    ///   No core-wide selection runs without this.
    /// - **Runtime, transient counter:**
    ///   `rq->core->core_forceidle_count > 0` is a SEPARATE
    ///   subsequent gate — `pick_next_task()` only invokes
    ///   `sched_core_account_forceidle(rq)` when this counter is
    ///   non-zero (`kernel/sched/core.c:6059`); the
    ///   `WARN_ON_ONCE(!rq->core->core_forceidle_count)` inside
    ///   `__sched_core_account_forceidle()` at
    ///   `kernel/sched/core_sched.c:252` reasserts the same
    ///   precondition. The early-return at `core_sched.c:254`
    ///   on `core_forceidle_start == 0` is then a third
    ///   transient guard against accounting before
    ///   forceidle has begun.
    /// - **Runtime, occupancy:** non-zero
    ///   `core_forceidle_occupation` (the `WARN_ON_ONCE` at
    ///   `core_sched.c:263`).
    ///
    /// Kernels that fail any build gate, or rqs that fail any
    /// runtime gate, see this counter at zero for every task.
    /// Hosts where no SMT cohort has ever accumulated forceidle
    /// also see zero across the board.
    pub core_forceidle_sum: crate::metric_types::MonotonicNs,
    /// Per-thread `se.slice` in nanoseconds. For fair-class
    /// tasks (SCHED_NORMAL / SCHED_BATCH) this is the
    /// instantaneous slice CFS is currently running the task
    /// with. For SCHED_EXT tasks the line is still emitted but
    /// reflects stale `p->se.slice` state — ext-class
    /// schedulers maintain slice in `p->scx.slice` and do not
    /// update `p->se.slice`. Field name `fair_slice_ns` mirrors
    /// the kernel emission gate `fair_policy(p->policy)`, not a
    /// guarantee about which class actually populated the value.
    ///
    /// `/proc/<tid>/sched` `se.slice`, plain integer via
    /// `P(se.slice)` at `kernel/sched/debug.c:1364`, gated by
    /// `fair_policy(p->policy)` at `kernel/sched/debug.c:1363`.
    /// `fair_policy()` is defined at `kernel/sched/sched.h:203`
    /// as `normal_policy(policy) || policy == SCHED_BATCH`, and
    /// `normal_policy()` at `sched.h:194` returns true for
    /// SCHED_NORMAL AND, when `CONFIG_SCHED_CLASS_EXT` is
    /// built, for SCHED_EXT. So the line IS emitted for
    /// SCHED_EXT tasks on a sched_ext-enabled kernel — but the
    /// value carries the staleness caveat above. The parser
    /// cannot distinguish "ext-class hasn't refreshed
    /// `p->se.slice` since the task left the fair class" from
    /// "CFS task with a current slice that happens to equal the
    /// last value": that ambiguity is the user's to resolve via
    /// `policy` (also captured per-thread). Tasks under
    /// SCHED_DEADLINE / SCHED_RR / SCHED_FIFO / SCHED_IDLE land
    /// at the absent-line default of 0.
    ///
    /// This is a GAUGE (instantaneous current value), not a
    /// counter or high-water mark. Distinct from
    /// [`Self::slice_max`] which IS the schedstat lifetime
    /// high-water — a thread that hasn't run for a long time
    /// can have a stale `fair_slice_ns` value while `slice_max`
    /// continues to reflect the historical worst. Aggregation
    /// across a group uses `Max` so the rendered cell shows the
    /// longest current slice any thread in the group is running
    /// with — Sum would multiply a near-identical instantaneous
    /// value across the group and obscure the signal (and would
    /// also be semantically meaningless: instantaneous gauges
    /// do not add).
    pub fair_slice_ns: crate::metric_types::GaugeNs,

    // -- /proc/<tid>/status (process-wide tgid count) --
    /// Total threads in this task's tgid (process-wide thread
    /// count, the `signal_struct->nr_threads` snapshot). Field
    /// name mirrors the kernel struct member to avoid collision
    /// with [`CtprofSnapshot::threads`] (the snapshot's own
    /// `Vec<ThreadState>`). `/proc/<pid>/status` `Threads:` line
    /// emitted at `fs/proc/array.c:290` via
    /// `seq_put_decimal_ull(m, "Threads:\t", num_threads)`.
    /// Identical for every thread of the same tgid.
    ///
    /// Capture-side dedup: the field is populated ONLY on the
    /// thread leader (tid == tgid) and zero for non-leader
    /// threads of the same process. The registry pairs this with
    /// [`crate::ctprof_compare::AggRule::MaxGaugeCount`] (not
    /// Sum) so the rendered cell surfaces "the largest process
    /// represented in this bucket" regardless of grouping axis.
    /// Sum would be wrong under `--group-by comm` and
    /// `--group-by cgroup` because non-leader buckets get a 0
    /// contribution from every member — a bucket whose leader
    /// thread did NOT match the grouping
    /// would render 0 even though processes are represented.
    /// Wrapped in [`crate::metric_types::GaugeCount`] so the
    /// type system rejects sum-style aggregation: a bucket with
    /// N threads sharing a tgid would over-count the parent
    /// process N-fold under Sum, while Max is well-defined
    /// (largest current count any contributor reported).
    pub nr_threads: crate::metric_types::GaugeCount,

    // -- /proc/<tid>/smaps_rollup (per-MM memory breakdown) --
    /// Per-process memory breakdown from
    /// `/proc/<tid>/smaps_rollup`, parsed as a key-value map
    /// with values in kilobytes (the kernel's native unit on
    /// this file — `__show_smap()` at `fs/proc/task_mmu.c:1330-1368`
    /// emits every line as `Name: NN kB`).
    ///
    /// Stored as a [`BTreeMap`] for forward-compat with the
    /// open key set: rollup mode (gated at task_mmu.c:1336)
    /// emits 22 keys on a recent kernel — Rss, Pss, Pss_Dirty,
    /// Pss_Anon, Pss_File, Pss_Shmem, Shared_Clean,
    /// Shared_Dirty, Private_Clean, Private_Dirty, Referenced,
    /// Anonymous, KSM, LazyFree, AnonHugePages,
    /// ShmemPmdMapped, FilePmdMapped, Shared_Hugetlb,
    /// Private_Hugetlb, Swap, SwapPss, Locked, plus the
    /// `[rollup]` header which the parser elides. The map
    /// preserves any future-kernel keys without a schema bump.
    /// Pss is the most operationally valuable: proportional
    /// share of shared pages — distinguishes "sole owner" from
    /// "one of N sharing".
    ///
    /// Per-MM, not per-thread: every thread of the same tgid
    /// shares one mm_struct, so all threads expose identical
    /// values. Capture-side dedup populates ONLY the thread
    /// leader (tid == tgid) and leaves non-leader threads at
    /// the empty map. Mirrors [`Self::nr_threads`]'s
    /// leader-dedup discipline. The capture cost is one
    /// `read_to_string` per tgid (NOT per-tid) because
    /// non-leaders short-circuit before opening the file.
    ///
    /// Empty when smaps_rollup is absent (older kernels
    /// without `/proc/<pid>/smaps_rollup` support — added
    /// upstream in 4.14) or unreadable (typical
    /// permission-denied for /proc/1/smaps_rollup outside
    /// CAP_SYS_PTRACE).
    pub smaps_rollup_kb: BTreeMap<String, u64>,

    // -- I/O (/proc/<tid>/io) --
    //
    // The whole file is emitted by `do_io_accounting`
    // (`fs/proc/base.c`) under a single `CONFIG_TASK_IO_ACCOUNTING`
    // gate, and `CONFIG_TASK_IO_ACCOUNTING` `depends on`
    // `CONFIG_TASK_XACCT` in `init/Kconfig` — so from the
    // procfs-reader perspective the file either appears with all
    // 7 fields or doesn't appear at all. The XACCT split that
    // sometimes shows up in kernel commentary describes the
    // increment-side path, not the procfs surface; for the
    // capture pipeline the relevant gate is `CONFIG_TASK_IO_ACCOUNTING`
    // for every field below.
    /// Bytes read at the read syscall layer (incl. cached /
    /// pagecache hits). Gated by `CONFIG_TASK_IO_ACCOUNTING`.
    pub rchar: crate::metric_types::Bytes,
    /// Bytes written at the write syscall layer (incl.
    /// pagecache / writeback). Gated by `CONFIG_TASK_IO_ACCOUNTING`.
    pub wchar: crate::metric_types::Bytes,
    /// Number of read syscalls. Gated by `CONFIG_TASK_IO_ACCOUNTING`.
    pub syscr: crate::metric_types::MonotonicCount,
    /// Number of write syscalls. Gated by `CONFIG_TASK_IO_ACCOUNTING`.
    pub syscw: crate::metric_types::MonotonicCount,
    /// Bytes that hit the storage device on read (excludes
    /// pagecache hits). Gated by `CONFIG_TASK_IO_ACCOUNTING`.
    pub read_bytes: crate::metric_types::Bytes,
    /// Bytes that hit the storage device on write
    /// (post-writeback). Gated by `CONFIG_TASK_IO_ACCOUNTING`.
    pub write_bytes: crate::metric_types::Bytes,
    /// Bytes the kernel deaccounted from a prior dirty-write
    /// because the page was reclaimed without writeback (truncate,
    /// inode invalidation). `/proc/<tid>/io` 7th line, gated by
    /// `CONFIG_TASK_IO_ACCOUNTING`.
    ///
    /// `include/linux/task_io_accounting_ops.h:39-42`
    /// (`task_io_account_cancelled_write`) increments
    /// `current->ioac.cancelled_write_bytes` — i.e. the value
    /// records on the task that triggers the deaccount
    /// (the truncating / unmapping task), NOT the original
    /// writer. Sole call site is `folio_account_cleaned`
    /// (`mm/page-writeback.c:2628`), invoked when a dirty folio
    /// is reclaimed without going through writeback.
    ///
    /// Operationally this is a "negative write" signal — bytes
    /// the kernel previously charged to a thread's `wchar`
    /// pipeline that never ended up on disk. Higher values mean
    /// more wasted writeback intent. Per-thread interpretation
    /// is asymmetric vs. [`Self::write_bytes`]: a thread's
    /// `cancelled_write_bytes` does NOT correspond to its own
    /// `write_bytes` — the writer and the canceller may be
    /// distinct tasks. Group-level Sum across a registry-grouped
    /// bucket is therefore meaningful (total bytes the bucket's
    /// threads cancelled), but per-thread `actual_write_bytes
    /// = write_bytes - cancelled_write_bytes` is NOT defined for
    /// that reason — the two counters track different parties.
    pub cancelled_write_bytes: crate::metric_types::Bytes,

    // -- taskstats delay accounting + memory watermarks (genetlink TASKSTATS family) --
    //
    // Per-tid records captured via the kernel's taskstats
    // genetlink interface (NOT exposed in /proc/<tid>/sched or
    // /proc/<tid>/stat). Two field families:
    //
    //   1. Delay accounting — eight categories (cpu/blkio/swapin/
    //      freepages/thrashing/compact/wpcopy/irq), each carrying
    //      count (number of events), delay_total_ns (cumulative
    //      ns of delay), delay_max_ns (longest single window),
    //      delay_min_ns (shortest non-zero window observed;
    //      sentinel 0 means "no events"). Gated on
    //      `CONFIG_TASKSTATS` + `CONFIG_TASK_DELAY_ACCT` plus the
    //      runtime `delayacct=on` toggle (sysctl
    //      `kernel.task_delayacct` or boot param `delayacct`).
    //
    //   2. Memory watermarks — `hiwater_rss_bytes` and
    //      `hiwater_vm_bytes`. Gated on `CONFIG_TASKSTATS` +
    //      `CONFIG_TASK_XACCT` (NOT `CONFIG_TASK_DELAY_ACCT`).
    //      Populated from the shared `mm_struct` so sibling tgid
    //      threads report identical values, and kernel threads
    //      (mm == NULL) leave the field at zero — see the
    //      per-field doc on `hiwater_rss_bytes`.
    //
    // Capture path is the [`crate::taskstats`] module —
    // best-effort, all fields collapse to zero when:
    //   - the kernel was built without `CONFIG_TASKSTATS`,
    //   - the relevant per-family kconfig is off (DELAY_ACCT or
    //     XACCT, depending on the field),
    //   - the runtime `delayacct=on` toggle is off (delay-family
    //     fields only — XACCT does not gate on the toggle),
    //   - the calling process lacks `CAP_NET_ADMIN`,
    //   - the per-tid query races a task exit (ESRCH).
    //
    // CAVEATS:
    //   - cpu_delay is RACY (sched_info path, no lock) — count and
    //     delay_total are not updated atomically.
    //   - swapin and thrashing OVERLAP — a thrashing event is also
    //     a swapin event from the syscall layer; do not sum.
    //   - delay_min == 0 means "no events observed", NOT "saw a
    //     zero-ns event". Compare against the matching count.
    //   - hiwater_* values are per-mm, not per-thread; sibling
    //     tgid threads report identical values, kernel threads
    //     (mm == NULL) report zero. See the per-field doc.
    /// Number of off-CPU windows the task waited for the runqueue
    /// to schedule it. Source: taskstats `cpu_count`, populated at
    /// query time from `tsk->sched_info.pcount` (incremented by
    /// `sched_info_arrive` in `kernel/sched/stats.h`, line 282).
    /// `delayacct_add_tsk` (`kernel/delayacct.c::delayacct_add_tsk`,
    /// line 175) snapshots the value into the reply via
    /// `d->cpu_count += t1` where `t1 = tsk->sched_info.pcount`.
    pub cpu_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns the task spent waiting on the runqueue.
    /// Source: taskstats `cpu_delay_total`. RACY: count and total
    /// are not updated atomically (sched_info path, no lock); a
    /// concurrent reader may observe count or total advance ahead
    /// of the other.
    pub cpu_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single CPU-wait window, ns. Source: taskstats
    /// `cpu_delay_max`. Same lifetime-watermark semantics as
    /// `wait_max` / `block_max` — `MaxPeak` aggregation surfaces
    /// the worst single window any thread in the group ever
    /// experienced.
    pub cpu_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero CPU-wait window, ns. Source: taskstats
    /// `cpu_delay_min`. Sentinel 0 means "no events observed":
    /// the kernel writes the field on every event, so 0 is
    /// distinguishable from a genuine zero-ns event by checking
    /// `cpu_delay_count == 0`. `PeakNs` aggregation surfaces "the
    /// largest minimum any thread reported" across the group.
    pub cpu_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of block-I/O wait windows. Source: taskstats
    /// `blkio_count`. Updates from `delayacct_blkio_start/end` in
    /// `kernel/delayacct.c`.
    pub blkio_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns the task waited on synchronous block I/O.
    /// Source: taskstats `blkio_delay_total`. Distinct from
    /// `iowait_sum` (schedstat) which counts a different bucket;
    /// the delayacct path is the canonical block-I/O delay
    /// accounting.
    pub blkio_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single block-I/O wait window, ns. Source: taskstats
    /// `blkio_delay_max`.
    pub blkio_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero block-I/O wait window, ns. Source:
    /// taskstats `blkio_delay_min`. Sentinel-0 caveat per
    /// `cpu_delay_min_ns`.
    pub blkio_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of swap-in wait windows. Source: taskstats
    /// `swapin_count`. NOTE: overlaps with `thrashing_count` —
    /// every thrashing event is also a swapin event from the
    /// syscall layer; do not sum.
    pub swapin_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns waiting for swap-in to complete. Source:
    /// taskstats `swapin_delay_total`.
    pub swapin_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single swap-in wait, ns. Source: taskstats
    /// `swapin_delay_max`.
    pub swapin_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero swap-in wait, ns. Sentinel-0 caveat per
    /// `cpu_delay_min_ns`.
    pub swapin_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of direct-reclaim (free-pages) wait windows. Source:
    /// taskstats `freepages_count`. Updates from
    /// `delayacct_freepages_start/end` (mm/page_alloc.c).
    pub freepages_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns waiting in direct memory reclaim. Source:
    /// taskstats `freepages_delay_total`.
    pub freepages_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single direct-reclaim wait, ns. Source: taskstats
    /// `freepages_delay_max`.
    pub freepages_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero direct-reclaim wait, ns. Sentinel-0 caveat
    /// per `cpu_delay_min_ns`.
    pub freepages_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of thrashing wait windows. Source: taskstats
    /// `thrashing_count`. OVERLAPS with `swapin_*`: thrashing
    /// detection is a refinement of swapin tracking
    /// (mm/workingset.c).
    pub thrashing_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns waiting under thrashing pressure. Source:
    /// taskstats `thrashing_delay_total`.
    pub thrashing_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single thrashing wait, ns. Source: taskstats
    /// `thrashing_delay_max`.
    pub thrashing_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero thrashing wait, ns. Sentinel-0 caveat per
    /// `cpu_delay_min_ns`.
    pub thrashing_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of memory-compaction wait windows. Source: taskstats
    /// `compact_count`. Updates from `delayacct_compact_start/end`
    /// (mm/compaction.c).
    pub compact_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns waiting on memory compaction. Source:
    /// taskstats `compact_delay_total`.
    pub compact_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single compaction wait, ns. Source: taskstats
    /// `compact_delay_max`.
    pub compact_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero compaction wait, ns. Sentinel-0 caveat
    /// per `cpu_delay_min_ns`.
    pub compact_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of write-protect-copy (CoW) fault wait windows.
    /// Source: taskstats `wpcopy_count`. Updates from
    /// `delayacct_wpcopy_start/end` (mm/memory.c).
    pub wpcopy_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns waiting on write-protect-copy faults. Source:
    /// taskstats `wpcopy_delay_total`.
    pub wpcopy_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single wpcopy wait, ns. Source: taskstats
    /// `wpcopy_delay_max`.
    pub wpcopy_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero wpcopy wait, ns. Sentinel-0 caveat per
    /// `cpu_delay_min_ns`.
    pub wpcopy_delay_min_ns: crate::metric_types::PeakNs,
    /// Number of IRQ-handler windows the task delegated. Source:
    /// taskstats `irq_count`. Updates from `delayacct_irq` in
    /// `kernel/delayacct.c` — counts kernel-IRQ time charged to
    /// the task by the IRQ accounting subsystem.
    pub irq_delay_count: crate::metric_types::MonotonicCount,
    /// Cumulative ns of IRQ handling time charged to the task.
    /// Source: taskstats `irq_delay_total`.
    pub irq_delay_total_ns: crate::metric_types::MonotonicNs,
    /// Longest single IRQ-handler window, ns. Source: taskstats
    /// `irq_delay_max`.
    pub irq_delay_max_ns: crate::metric_types::PeakNs,
    /// Shortest non-zero IRQ-handler window, ns. Sentinel-0 caveat
    /// per `cpu_delay_min_ns`.
    pub irq_delay_min_ns: crate::metric_types::PeakNs,
    /// Lifetime high-watermark of resident-set size, bytes. Source:
    /// taskstats `hiwater_rss` (kB), converted at parse time via
    /// `saturating_mul(1024)`. Updates from `xacct_add_tsk` in
    /// `kernel/tsacct.c::xacct_add_tsk`. Distinct from
    /// `smaps_rollup_kb["Rss"]` which is the CURRENT RSS —
    /// this field is the lifetime peak.
    ///
    /// **Kernel threads read zero**: `xacct_add_tsk` at
    /// `kernel/tsacct.c:99` calls `mm = get_task_mm(p)` and the
    /// hiwater assignments at lines 100-104 are guarded by
    /// `if (mm)`. Kernel threads (`PF_KTHREAD`, `tsk->mm == NULL`)
    /// skip the assignment entirely, so the field stays at the
    /// kernel-side zero default.
    ///
    /// **Sibling threads of the same tgid see the same value**:
    /// `get_mm_hiwater_rss(mm)` reads from the shared
    /// `mm_struct`, so every thread of a process reports the same
    /// hiwater value. The registry's `MaxPeakBytes` aggregation
    /// behaves as a per-process selector when buckets span
    /// multiple tgids: cross-tgid Max picks the largest
    /// per-process watermark in the bucket; intra-tgid Max is a
    /// no-op (every sibling reports the same number).
    pub hiwater_rss_bytes: crate::metric_types::PeakBytes,
    /// Lifetime high-watermark of virtual-memory size, bytes.
    /// Source: taskstats `hiwater_vm` (kB), converted at parse
    /// time. Same kernel write path as `hiwater_rss_bytes` —
    /// inherits the same kernel-thread zero and same sibling-tid
    /// shared-mm caveats; see [`Self::hiwater_rss_bytes`].
    pub hiwater_vm_bytes: crate::metric_types::PeakBytes,
}

impl Default for ThreadState {
    fn default() -> Self {
        Self {
            tid: 0,
            tgid: 0,
            pcomm: String::new(),
            comm: String::new(),
            cgroup: String::new(),
            start_time_clock_ticks: 0,
            policy: Default::default(),
            nice: crate::metric_types::OrdinalI32(0),
            cpu_affinity: Default::default(),
            processor: Default::default(),
            // `'~'` (the absent-value sentinel) instead of the
            // bare `char` Default `'\0'`; see [`Self::state`].
            state: default_state_char(),
            ext_enabled: false,
            run_time_ns: Default::default(),
            wait_time_ns: Default::default(),
            timeslices: Default::default(),
            voluntary_csw: Default::default(),
            nonvoluntary_csw: Default::default(),
            nr_wakeups: Default::default(),
            nr_wakeups_local: Default::default(),
            nr_wakeups_remote: Default::default(),
            nr_wakeups_sync: Default::default(),
            nr_wakeups_migrate: Default::default(),
            nr_wakeups_affine: Default::default(),
            nr_wakeups_affine_attempts: Default::default(),
            nr_migrations: Default::default(),
            nr_forced_migrations: Default::default(),
            nr_failed_migrations_affine: Default::default(),
            nr_failed_migrations_running: Default::default(),
            nr_failed_migrations_hot: Default::default(),
            wait_sum: Default::default(),
            wait_count: Default::default(),
            wait_max: Default::default(),
            voluntary_sleep_ns: Default::default(),
            sleep_max: Default::default(),
            block_sum: Default::default(),
            block_max: Default::default(),
            iowait_sum: Default::default(),
            iowait_count: Default::default(),
            exec_max: Default::default(),
            slice_max: Default::default(),
            allocated_bytes: Default::default(),
            deallocated_bytes: Default::default(),
            minflt: Default::default(),
            majflt: Default::default(),
            utime_clock_ticks: Default::default(),
            stime_clock_ticks: Default::default(),
            priority: Default::default(),
            rt_priority: Default::default(),
            core_forceidle_sum: Default::default(),
            fair_slice_ns: Default::default(),
            nr_threads: Default::default(),
            smaps_rollup_kb: BTreeMap::new(),
            rchar: Default::default(),
            wchar: Default::default(),
            syscr: Default::default(),
            syscw: Default::default(),
            read_bytes: Default::default(),
            write_bytes: Default::default(),
            cancelled_write_bytes: Default::default(),
            cpu_delay_count: Default::default(),
            cpu_delay_total_ns: Default::default(),
            cpu_delay_max_ns: Default::default(),
            cpu_delay_min_ns: Default::default(),
            blkio_delay_count: Default::default(),
            blkio_delay_total_ns: Default::default(),
            blkio_delay_max_ns: Default::default(),
            blkio_delay_min_ns: Default::default(),
            swapin_delay_count: Default::default(),
            swapin_delay_total_ns: Default::default(),
            swapin_delay_max_ns: Default::default(),
            swapin_delay_min_ns: Default::default(),
            freepages_delay_count: Default::default(),
            freepages_delay_total_ns: Default::default(),
            freepages_delay_max_ns: Default::default(),
            freepages_delay_min_ns: Default::default(),
            thrashing_delay_count: Default::default(),
            thrashing_delay_total_ns: Default::default(),
            thrashing_delay_max_ns: Default::default(),
            thrashing_delay_min_ns: Default::default(),
            compact_delay_count: Default::default(),
            compact_delay_total_ns: Default::default(),
            compact_delay_max_ns: Default::default(),
            compact_delay_min_ns: Default::default(),
            wpcopy_delay_count: Default::default(),
            wpcopy_delay_total_ns: Default::default(),
            wpcopy_delay_max_ns: Default::default(),
            wpcopy_delay_min_ns: Default::default(),
            irq_delay_count: Default::default(),
            irq_delay_total_ns: Default::default(),
            irq_delay_max_ns: Default::default(),
            irq_delay_min_ns: Default::default(),
            hiwater_rss_bytes: Default::default(),
            hiwater_vm_bytes: Default::default(),
        }
    }
}

impl ThreadState {
    /// Overwrite the taskstats-sourced delay-accounting fields
    /// from a `DelayStats` payload. Called by `capture_with` /
    /// `capture_pid_with` after a successful per-tid
    /// [`crate::taskstats::TaskstatsClient::query_tid`] call;
    /// query failures leave the fields at the absent-counter
    /// default of zero installed in `capture_thread_at_with_tally`.
    pub(crate) fn apply_delay_stats(&mut self, ds: &crate::taskstats::DelayStats) {
        use crate::metric_types::{MonotonicCount, MonotonicNs, PeakBytes, PeakNs};
        self.cpu_delay_count = MonotonicCount(ds.cpu_count);
        self.cpu_delay_total_ns = MonotonicNs(ds.cpu_delay_total_ns);
        self.cpu_delay_max_ns = PeakNs(ds.cpu_delay_max_ns);
        self.cpu_delay_min_ns = PeakNs(ds.cpu_delay_min_ns);
        self.blkio_delay_count = MonotonicCount(ds.blkio_count);
        self.blkio_delay_total_ns = MonotonicNs(ds.blkio_delay_total_ns);
        self.blkio_delay_max_ns = PeakNs(ds.blkio_delay_max_ns);
        self.blkio_delay_min_ns = PeakNs(ds.blkio_delay_min_ns);
        self.swapin_delay_count = MonotonicCount(ds.swapin_count);
        self.swapin_delay_total_ns = MonotonicNs(ds.swapin_delay_total_ns);
        self.swapin_delay_max_ns = PeakNs(ds.swapin_delay_max_ns);
        self.swapin_delay_min_ns = PeakNs(ds.swapin_delay_min_ns);
        self.freepages_delay_count = MonotonicCount(ds.freepages_count);
        self.freepages_delay_total_ns = MonotonicNs(ds.freepages_delay_total_ns);
        self.freepages_delay_max_ns = PeakNs(ds.freepages_delay_max_ns);
        self.freepages_delay_min_ns = PeakNs(ds.freepages_delay_min_ns);
        self.thrashing_delay_count = MonotonicCount(ds.thrashing_count);
        self.thrashing_delay_total_ns = MonotonicNs(ds.thrashing_delay_total_ns);
        self.thrashing_delay_max_ns = PeakNs(ds.thrashing_delay_max_ns);
        self.thrashing_delay_min_ns = PeakNs(ds.thrashing_delay_min_ns);
        self.compact_delay_count = MonotonicCount(ds.compact_count);
        self.compact_delay_total_ns = MonotonicNs(ds.compact_delay_total_ns);
        self.compact_delay_max_ns = PeakNs(ds.compact_delay_max_ns);
        self.compact_delay_min_ns = PeakNs(ds.compact_delay_min_ns);
        self.wpcopy_delay_count = MonotonicCount(ds.wpcopy_count);
        self.wpcopy_delay_total_ns = MonotonicNs(ds.wpcopy_delay_total_ns);
        self.wpcopy_delay_max_ns = PeakNs(ds.wpcopy_delay_max_ns);
        self.wpcopy_delay_min_ns = PeakNs(ds.wpcopy_delay_min_ns);
        self.irq_delay_count = MonotonicCount(ds.irq_count);
        self.irq_delay_total_ns = MonotonicNs(ds.irq_delay_total_ns);
        self.irq_delay_max_ns = PeakNs(ds.irq_delay_max_ns);
        self.irq_delay_min_ns = PeakNs(ds.irq_delay_min_ns);
        self.hiwater_rss_bytes = PeakBytes(ds.hiwater_rss_bytes);
        self.hiwater_vm_bytes = PeakBytes(ds.hiwater_vm_bytes);
    }

    /// Iterate over [`Self::smaps_rollup_kb`] with values
    /// converted from kilobytes to bytes via `saturating_mul(1024)`.
    /// The kernel emits smaps_rollup values in kB; the
    /// project's display layer auto-scales bytes via the
    /// existing "B" → KiB → MiB → GiB ladder, so a single
    /// helper centralizes the unit conversion at every render
    /// site (write_show + write_diff). Saturating multiply
    /// guards against pathological input from a malformed
    /// snapshot file. Wrapped in
    /// [`crate::metric_types::Bytes`] so the byte-typed value
    /// flows through the same auto-scale path as the rest of
    /// the byte-tagged registry metrics.
    pub fn smaps_rollup_bytes(
        &self,
    ) -> impl Iterator<Item = (&String, crate::metric_types::Bytes)> {
        self.smaps_rollup_kb
            .iter()
            .map(|(k, v)| (k, crate::metric_types::Bytes(v.saturating_mul(1024))))
    }
}

/// Per-cgroup enrichment record attached to [`CtprofSnapshot`].
///
/// Populated from the cgroup v2 filesystem at capture time. The
/// shape mirrors the kernel's per-controller file layout:
/// [`CgroupCpuStats`] holds the `cpu.*` files,
/// [`CgroupMemoryStats`] holds the `memory.*` files,
/// [`CgroupPidsStats`] holds the `pids.*` files, and [`Psi`]
/// holds the `<resource>.pressure` files. These are
/// aggregate-over-the-cgroup values — NOT summable from
/// per-thread data — so the capture layer reads them directly
/// from cgroupfs rather than deriving.
///
/// Nested-struct shape (rather than a flat ~50-field struct)
/// mirrors the kernel's controller-by-controller exposure: a
/// reader who knows the kernel layout can map directly between
/// cgroupfs files and Rust fields, and the merge policy in
/// [`crate::ctprof_compare::flatten_cgroup_stats`] applies
/// per-domain (max for limits, min for floors, saturating_add
/// for counters) without conflating across domains.
///
/// Schema note: the previous flat shape (4 fields:
/// `cpu_usage_usec`, `nr_throttled`, `throttled_usec`,
/// `memory_current`) is gone. Snapshots written by older
/// versions deserialize via serde's defaulting — old fields
/// land on the new nested fields' zero defaults rather than
/// migrating, so a baseline-vs-candidate compare against an
/// old snapshot produces "every counter went from N to 0".
/// Re-capture both sides with the current build to compare
/// faithfully. Per the project's pre-1.0 disposable-sidecar
/// policy this is intentional.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CgroupStats {
    pub cpu: CgroupCpuStats,
    pub memory: CgroupMemoryStats,
    pub pids: CgroupPidsStats,
    /// Pressure Stall Information for this cgroup, per resource.
    /// Populated from `<cgroup>/cpu.pressure`,
    /// `<cgroup>/memory.pressure`, `<cgroup>/io.pressure`, and
    /// `<cgroup>/irq.pressure` (cgroup v2 files declared at
    /// `kernel/cgroup/cgroup.c:5453-5482`). Defaults to all-zero
    /// when the kernel has CONFIG_PSI off, when PSI is disabled
    /// at runtime via the `psi=0` boot param, or when individual
    /// resource files are absent (older kernels missing
    /// irq.pressure).
    pub psi: Psi,
}

/// CPU controller state for one cgroup. Fields mirror the
/// `cpu.*` cgroup v2 files exposed under
/// `<cgroup>/cpu.stat`, `<cgroup>/cpu.max`,
/// `<cgroup>/cpu.weight`, and `<cgroup>/cpu.weight.nice`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CgroupCpuStats {
    /// `usage_usec` from `cpu.stat`. Cumulative CPU time consumed
    /// by tasks in this cgroup, in microseconds.
    pub usage_usec: u64,
    /// `nr_throttled` from `cpu.stat`. Cumulative count of
    /// CFS-bandwidth throttling events that paused this cgroup.
    pub nr_throttled: u64,
    /// `throttled_usec` from `cpu.stat`. Cumulative wall-clock
    /// time the cgroup spent throttled by CFS bandwidth.
    pub throttled_usec: u64,
    /// `cpu.max` quota in microseconds. `None` when the file is
    /// absent (root cgroup) OR when the kernel emits the literal
    /// "max" token (no CFS bandwidth cap configured for this
    /// cgroup).
    pub max_quota_us: Option<u64>,
    /// `cpu.max` period in microseconds. Default 100_000 (100ms)
    /// per the kernel default. Always present alongside the
    /// quota half on a child cgroup; defaults to 100_000 when
    /// the file is absent (root cgroup).
    pub max_period_us: u64,
    /// `cpu.weight` (1..=10_000, default 100). `None` when the
    /// file is absent (root cgroup); the kernel does not allow
    /// 0 as a value, so the absent-vs-zero distinction is
    /// load-bearing.
    pub weight: Option<u64>,
    /// `cpu.weight.nice` (-20..=19, default 0). `None` when the
    /// file is absent. Alias-domain for [`Self::weight`] —
    /// the kernel writes both files in lockstep but they're
    /// captured independently to surface any
    /// kernel-version-specific divergence.
    pub weight_nice: Option<i32>,
}

/// Memory controller state for one cgroup. Fields mirror the
/// `memory.*` cgroup v2 files. `stat` and `events` are
/// captured as flat key-value maps so the data model
/// auto-extends when the kernel adds new keys (memory.stat
/// has 71 keys on a recent kernel; the explicit list is
/// scheduler-correctness-relevant but the map preserves
/// regression-detection on lesser-known counters).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CgroupMemoryStats {
    /// `memory.current`, instantaneous RSS of the cgroup in
    /// bytes.
    pub current: u64,
    /// `memory.max`, hard memory limit in bytes. `None` when
    /// the file is absent (root cgroup) OR when the kernel
    /// emits the literal "max" token (no hard cap).
    pub max: Option<u64>,
    /// `memory.high`, soft pressure limit in bytes. `None` when
    /// absent or unlimited (same "max"-token semantics as
    /// [`Self::max`]).
    pub high: Option<u64>,
    /// `memory.low`, best-effort protection floor in bytes.
    /// `None` when the file is absent (no protection
    /// configured); `Some(u64::MAX)` when the kernel emits the
    /// literal `max` token (request maximum protection — every
    /// byte under the cgroup is protected). Per the kernel's
    /// cgroup v2 docs, memory under `low` is protected from
    /// reclaim unless no unprotected memory remains. Note the
    /// asymmetry vs. limits: `None` means "no floor" (semantic
    /// opposite of "max"-as-no-cap on the limit fields above).
    pub low: Option<u64>,
    /// `memory.min`, hard protection floor in bytes. `None`
    /// when absent (no floor). `Some(u64::MAX)` when the kernel
    /// emits `max` (full protection). Stronger than `low` —
    /// memory under `min` is never reclaimed even under
    /// memory pressure.
    pub min: Option<u64>,
    /// `memory.stat` parsed as a key-value map. Keys mirror the
    /// kernel-emitted strings (e.g. `anon`, `file`,
    /// `workingset_refault_anon`, `pgfault`, `pgmajfault`,
    /// `slab`, the active/inactive variants, etc.). Empty when
    /// the file is absent.
    pub stat: BTreeMap<String, u64>,
    /// `memory.events` parsed as a key-value map. Typical keys:
    /// `low`, `high`, `max`, `oom`, `oom_kill`,
    /// `oom_group_kill`, `sock_throttled` (subset varies by
    /// kernel version). Empty when the file is absent.
    pub events: BTreeMap<String, u64>,
}

/// PIDs controller state for one cgroup. Fields mirror the
/// `pids.*` cgroup v2 files. The pids controller is optional
/// (must be enabled in `cgroup.subtree_control`); on hosts that
/// don't enable it, both fields are `None`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct CgroupPidsStats {
    /// `pids.current`, current task count in this cgroup.
    /// `None` when the file is absent (pids controller not
    /// enabled).
    pub current: Option<u64>,
    /// `pids.max`, hard task-count limit. `None` when the file
    /// is absent OR when the kernel emits the literal "max"
    /// token (no cap).
    pub max: Option<u64>,
}

/// One Pressure Stall Information half-line: either the `some`
/// or `full` row for one resource. Mirrors the kernel emission
/// format `%s avg10=%lu.%02lu avg60=%lu.%02lu avg300=%lu.%02lu total=%llu`
/// at `kernel/sched/psi.c:1284`.
///
/// `avg10/60/300` are stored as **centi-percent** (lossless
/// fixed-point) — the kernel writes `LOAD_INT(avg).LOAD_FRAC(avg)`
/// as a 2-decimal-digit percentage at psi.c:1284. The integer
/// expansion is `int * 100 + frac`, giving a numerical range of
/// `0..=10099`. The upper bound is `100.99` (not `100.00`)
/// because the kernel's EWMA helper at
/// `include/linux/sched/loadavg.h:35` rounds via `newload +=
/// FIXED_1 - 1` before the final `>> FSHIFT`, so a fully-loaded
/// group can land just over `100.0` for one sample. This avoids
/// serde JSON float-roundtrip drift that would manifest as
/// spurious non-zero deltas in compare output.
///
/// `total_usec` is microseconds (kernel
/// `div_u64(total_ns, NSEC_PER_USEC)` at psi.c:1281). Same unit
/// as [`CgroupCpuStats::usage_usec`], so the existing
/// auto_scale "µs" ladder applies.
///
/// "some" semantics: at least one task is stalled on this
/// resource. "full" semantics: every runnable task is stalled.
/// At the SYSTEM level (`/proc/pressure/cpu`), `cpu.full` is
/// always zero by kernel design — the explicit gate
/// `if (!(group == &psi_system && res == PSI_CPU && full))` at
/// `kernel/sched/psi.c:1276-1277` skips the avg/total
/// computation, but the `seq_printf` at psi.c:1284 still emits
/// the structurally-present line. Per-cgroup `cpu.full` (under
/// `<cgroup>/cpu.pressure`) IS meaningful and computed
/// normally. `irq` is full-only (kernel `only_full = res == PSI_IRQ`
/// at psi.c:1268), so [`PsiResource::some`] for irq always reads
/// zero.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PsiHalf {
    /// 10-second running average of pressure %, scaled by 100
    /// (so 0..=10099 covers 0.00..=100.99 — see the EWMA-rounding
    /// note on the struct doc).
    pub avg10: u16,
    /// 60-second running average of pressure %, same scaling.
    pub avg60: u16,
    /// 300-second running average of pressure %, same scaling.
    pub avg300: u16,
    /// Cumulative total stalled time in microseconds.
    pub total_usec: u64,
}

impl PsiHalf {
    /// Convert the centi-percent `avg10` value to a percentage
    /// `f64`. Returns `0.0..=100.99` per the kernel's EWMA
    /// rounding (see struct-level doc).
    pub fn avg10_percent(&self) -> f64 {
        self.avg10 as f64 / 100.0
    }

    /// Convert the centi-percent `avg60` value to a percentage
    /// `f64`. Same range as [`Self::avg10_percent`].
    pub fn avg60_percent(&self) -> f64 {
        self.avg60 as f64 / 100.0
    }

    /// Convert the centi-percent `avg300` value to a percentage
    /// `f64`. Same range as [`Self::avg10_percent`].
    pub fn avg300_percent(&self) -> f64 {
        self.avg300 as f64 / 100.0
    }
}

/// Pressure Stall Information for one resource (cpu / memory /
/// io / irq), bundling the `some` and `full` halves.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PsiResource {
    pub some: PsiHalf,
    pub full: PsiHalf,
}

/// Bundle of [`PsiResource`] for the four kernel-exposed
/// resources. Same shape used at both system level
/// ([`CtprofSnapshot::psi`]) and per-cgroup
/// ([`CgroupStats::psi`]) — the data source differs but the
/// kernel emits the same format and field set in both places.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Psi {
    pub cpu: PsiResource,
    pub memory: PsiResource,
    pub io: PsiResource,
    /// IRQ pressure. Only the `full` half is populated by the
    /// kernel (psi.c:1268 sets `only_full = res == PSI_IRQ`);
    /// `irq.some` is structurally present but always zero.
    /// Requires both `CONFIG_IRQ_TIME_ACCOUNTING` at build AND
    /// `irqtime_enabled()` at runtime (`/proc/pressure/irq` returns
    /// `-EOPNOTSUPP` per `kernel/sched/psi.c:1255` otherwise);
    /// runtime irqtime is gated by the `tsc=...` boot param /
    /// `irqtime_enabled` static branch — when off, the file open
    /// fails and the parser leaves this resource at the default
    /// all-zero value.
    pub irq: PsiResource,
}

/// Global sched_ext sysfs state, captured from
/// `/sys/kernel/sched_ext/`. The kernel registers exactly five
/// global attributes via `scx_global_attrs[]` at
/// `kernel/sched/ext.c:4715-4722`; this struct mirrors them
/// 1-to-1.
///
/// Per-scheduler attrs (`/sys/kernel/sched_ext/root/...`) are
/// out of scope: those are scheduler-specific internals
/// (queued/dispatched/ops-name) that come and go as schedulers
/// load and unload, and answer different questions than the
/// global counters here.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SchedExtSysfs {
    /// `state` — sched_ext class enable state. One of
    /// `enabling`, `enabled`, `disabling`, `disabled` per
    /// `scx_enable_state_str[]` at
    /// `kernel/sched/ext_internal.h:1229-1234`. Emitted by
    /// `scx_attr_state_show()` at
    /// `kernel/sched/ext.c:4680-4684`. Defaults to empty string
    /// when the file is unreadable; `disabled` when no scx
    /// scheduler is currently loaded. The "is sched_ext active
    /// during this capture?" answer.
    pub state: String,

    /// `switch_all` — boolean (rendered as 0/1) indicating
    /// whether ALL scheduling classes have been switched to
    /// scx (vs. only those tasks the BPF scheduler claims via
    /// the per-task selection path). Emitted by
    /// `scx_attr_switch_all_show()` at
    /// `kernel/sched/ext.c:4687-4691` via
    /// `READ_ONCE(scx_switching_all)`.
    pub switch_all: u64,

    /// `nr_rejected` — count of tasks rejected from
    /// SCHED_EXT during init when `ops.init_task()` set
    /// `p->disallow`. Increment at
    /// `kernel/sched/ext.c:3531-3542`: when a task entering
    /// SCHED_EXT has its policy reverted to SCHED_NORMAL
    /// because the BPF scheduler asked the kernel to disallow
    /// it, `atomic_long_inc(&scx_nr_rejected)` fires.
    /// `atomic_long_read(&scx_nr_rejected)` is emitted by
    /// `scx_attr_nr_rejected_show()` at
    /// `kernel/sched/ext.c:4694-4698`.
    ///
    /// Resets to 0 on every scheduler load: `scx_enable()` at
    /// `kernel/sched/ext.c:6646` does
    /// `atomic_long_set(&scx_nr_rejected, 0)` before bringing
    /// the new scheduler online. To detect a reload-driven
    /// reset rather than a genuine cumulative drop, pair the
    /// nr_rejected delta with [`Self::enable_seq`] — any
    /// enable_seq movement across two snapshots invalidates
    /// nr_rejected as a monotonic counter.
    ///
    /// Does NOT count runtime dispatch errors. The "did the
    /// scheduler reject a dispatch operation at runtime?"
    /// question is answered by per-scheduler debug data
    /// (`/sys/kernel/sched_ext/root/...`), out of scope for
    /// this global-attrs struct.
    pub nr_rejected: u64,

    /// `hotplug_seq` — per-CPU-hotplug-event sequence counter.
    /// Atomic long incremented every time the kernel observes a
    /// hotplug transition. Emitted by
    /// `scx_attr_hotplug_seq_show()` at
    /// `kernel/sched/ext.c:4701-4705`. Comparing two snapshots:
    /// any delta indicates that a CPU online/offline event
    /// happened during the interval, which can confound
    /// per-CPU statistics.
    pub hotplug_seq: u64,

    /// `enable_seq` — per-scheduler-load sequence counter.
    /// Atomic long incremented at
    /// `kernel/sched/ext.c:6822` (`atomic_long_inc(&scx_enable_seq)`)
    /// each time a scx scheduler is enabled. Comparing two
    /// snapshots: any delta indicates a scheduler reload
    /// happened during the interval — counter resets on the
    /// scx side will surface here even if the per-thread data
    /// looks continuous.
    pub enable_seq: u64,
}

// Parsers + tallying readers live in `parse.rs` to keep this
// production file under the per-file line budget. The `use
// parse::*;` glob keeps existing call sites unchanged.
mod parse;
use parse::*;

/// Capture one thread's procfs-derived profile under an arbitrary
/// procfs root. Each procfs reader returns `Option`; the assembled
/// [`ThreadState`] coerces `None` to the field's default per the
/// module-level capture contract. The jemalloc per-thread TSD
/// counters (`allocated_bytes` / `deallocated_bytes`) are NOT
/// populated by this function — they require a tgid-scoped probe
/// attach that the caller owns ([`capture_with`] /
/// [`capture_pid_with`] do this and write the counters directly
/// onto the returned `ThreadState`). On the returned struct, both
/// fields therefore land at the absent-counter default of zero
/// unless the caller overwrites them.
///
/// `comm` is the thread name the caller has already read from
/// `<proc_root>/<tgid>/task/<tid>/comm` (typically via
/// [`read_thread_comm_at`]). Passing it in — symmetric with the
/// pre-existing `pcomm` parameter — lets the caller share one
/// procfs read with the per-tid probe-recording path
/// (`probe_thread_recording`), which needs the thread name for
/// tracing on probe failures: a hot loop that re-reads the file
/// inside this fn would double the comm syscalls per tid on hosts
/// with thousands of threads.
///
/// Pass empty string for the absent-comm default; the ghost
/// filter in [`capture_with`] / [`capture_pid_with`] keys on
/// `ThreadState::comm.is_empty()` to drop a tid that exited
/// between `iter_task_ids_at` and this call, so an empty `comm`
/// is the correct shape for that path.
///
/// `use_syscall_affinity` gates the `sched_getaffinity(2)` path —
/// tests staging a synthetic `/proc` pass `false` so the syscall
/// does not read the REAL affinity of the test process; production
/// passes `true` and falls back to `Cpus_allowed_list:` when the
/// syscall returns EPERM.
#[cfg(test)]
fn capture_thread_at(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    pcomm: &str,
    comm: &str,
    use_syscall_affinity: bool,
) -> ThreadState {
    capture_thread_at_with_tally(
        proc_root,
        tgid,
        tid,
        pcomm,
        comm,
        use_syscall_affinity,
        &mut None,
    )
}

/// Per-tid procfs walk. Threads a `&mut ParseTally` through every
/// per-file reader so per-tid read failures land in the
/// per-snapshot [`CtprofParseSummary`] when the capture
/// pipeline runs in production mode (`use_syscall_affinity=true`).
/// Synthetic-tree tests typically pass `&mut None` for the tally,
/// matching the pre-tally shape.
fn capture_thread_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    pcomm: &str,
    comm: &str,
    use_syscall_affinity: bool,
    tally: &mut Option<&mut ParseTally>,
) -> ThreadState {
    let cgroup = read_cgroup_at_with_tally(proc_root, tgid, tid, tally).unwrap_or_default();
    let stat = read_stat_at_with_tally(proc_root, tgid, tid, tally);
    let (run_time_ns, wait_time_ns, timeslices) =
        read_schedstat_at_with_tally(proc_root, tgid, tid, tally);
    let io = read_io_at_with_tally(proc_root, tgid, tid, tally);
    let status = read_status_at_with_tally(proc_root, tgid, tid, tally);
    let sched = read_sched_at_with_tally(proc_root, tgid, tid, tally);
    let smaps_rollup_kb = read_smaps_rollup_at_with_tally(proc_root, tgid, tid, tally);
    let cpu_affinity = if use_syscall_affinity {
        crate::cpu_util::read_affinity(tid)
            .or(status.cpus_allowed)
            .unwrap_or_default()
    } else {
        status.cpus_allowed.unwrap_or_default()
    };
    use crate::metric_types::{
        Bytes, CategoricalString, ClockTicks, CpuSet, GaugeCount, GaugeNs, MonotonicCount,
        MonotonicNs, OrdinalI32, OrdinalU32, PeakNs,
    };
    ThreadState {
        tid: tid as u32,
        tgid: tgid as u32,
        pcomm: pcomm.to_string(),
        comm: comm.to_string(),
        cgroup,
        start_time_clock_ticks: stat.start_time_clock_ticks.unwrap_or(0),
        policy: CategoricalString(stat.policy.map(policy_name).unwrap_or_default()),
        nice: OrdinalI32(stat.nice.unwrap_or(0)),
        cpu_affinity: CpuSet(cpu_affinity),
        processor: OrdinalI32(stat.processor.unwrap_or(0)),
        state: status.state.unwrap_or_else(default_state_char),
        ext_enabled: sched.ext_enabled.unwrap_or(false),
        run_time_ns: MonotonicNs(run_time_ns.unwrap_or(0)),
        wait_time_ns: MonotonicNs(wait_time_ns.unwrap_or(0)),
        timeslices: MonotonicCount(timeslices.unwrap_or(0)),
        voluntary_csw: MonotonicCount(status.voluntary_csw.unwrap_or(0)),
        nonvoluntary_csw: MonotonicCount(status.nonvoluntary_csw.unwrap_or(0)),
        nr_wakeups: MonotonicCount(sched.nr_wakeups.unwrap_or(0)),
        nr_wakeups_local: MonotonicCount(sched.nr_wakeups_local.unwrap_or(0)),
        nr_wakeups_remote: MonotonicCount(sched.nr_wakeups_remote.unwrap_or(0)),
        nr_wakeups_sync: MonotonicCount(sched.nr_wakeups_sync.unwrap_or(0)),
        nr_wakeups_migrate: MonotonicCount(sched.nr_wakeups_migrate.unwrap_or(0)),
        nr_wakeups_affine: MonotonicCount(sched.nr_wakeups_affine.unwrap_or(0)),
        nr_wakeups_affine_attempts: MonotonicCount(sched.nr_wakeups_affine_attempts.unwrap_or(0)),
        nr_migrations: MonotonicCount(sched.nr_migrations.unwrap_or(0)),
        nr_forced_migrations: MonotonicCount(sched.nr_forced_migrations.unwrap_or(0)),
        nr_failed_migrations_affine: MonotonicCount(sched.nr_failed_migrations_affine.unwrap_or(0)),
        nr_failed_migrations_running: MonotonicCount(
            sched.nr_failed_migrations_running.unwrap_or(0),
        ),
        nr_failed_migrations_hot: MonotonicCount(sched.nr_failed_migrations_hot.unwrap_or(0)),
        wait_sum: MonotonicNs(sched.wait_sum.unwrap_or(0)),
        wait_count: MonotonicCount(sched.wait_count.unwrap_or(0)),
        wait_max: PeakNs(sched.wait_max.unwrap_or(0)),
        // Capture-time normalization: kernel's `sum_sleep_runtime`
        // counts BOTH voluntary sleep AND involuntary block (see
        // `__update_stats_enqueue_sleeper` at kernel/sched/stats.c).
        // Subtracting the block component leaves pure voluntary
        // sleep — the operationally useful signal — and avoids the
        // need for a derived metric at compare time.
        //
        // The subtraction is only meaningful when BOTH halves
        // parsed successfully. If `sum_block_runtime` is missing,
        // an `unwrap_or(0)` fallback would yield
        // `sum_sleep_runtime - 0 = full_sleep_total`, mislabelling
        // the involuntary-block component as voluntary sleep and
        // breaking the field-doc contract ("voluntary only"). If
        // `sum_sleep_runtime` is missing, the fallback would yield
        // `0 - block`, which `saturating_sub` collapses to 0 but
        // also discards any real voluntary signal that might have
        // been recorded if the kernel had emitted both. Either
        // half-missing case means the value is uncomputable, so
        // it falls through to 0 — matching the "absent data → 0"
        // convention used by every sibling field at this site
        // (e.g. `wait_sum`, `sleep_max`, `block_sum`) and
        // co-locating with the existing `block_sum: 0` that the
        // same parse miss already produces below.
        //
        // `saturating_sub` remains in the both-Some path as
        // defense against the kernel-ordering edge case:
        // `__update_stats_enqueue_sleeper` adds to
        // `sum_sleep_runtime` BEFORE adding the same delta to
        // `sum_block_runtime`, so a sample read between those
        // writes can transiently yield `block > sleep` even
        // though every in-tree path eventually settles to
        // `block <= sleep`.
        voluntary_sleep_ns: MonotonicNs(
            match (sched.sleep_sum, sched.block_sum) {
                (Some(sleep), Some(block)) => sleep.saturating_sub(block),
                _ => 0,
            },
        ),
        sleep_max: PeakNs(sched.sleep_max.unwrap_or(0)),
        block_sum: MonotonicNs(sched.block_sum.unwrap_or(0)),
        block_max: PeakNs(sched.block_max.unwrap_or(0)),
        iowait_sum: MonotonicNs(sched.iowait_sum.unwrap_or(0)),
        iowait_count: MonotonicCount(sched.iowait_count.unwrap_or(0)),
        exec_max: PeakNs(sched.exec_max.unwrap_or(0)),
        slice_max: PeakNs(sched.slice_max.unwrap_or(0)),
        allocated_bytes: Bytes(0),
        deallocated_bytes: Bytes(0),
        minflt: MonotonicCount(stat.minflt.unwrap_or(0)),
        majflt: MonotonicCount(stat.majflt.unwrap_or(0)),
        utime_clock_ticks: ClockTicks(stat.utime_clock_ticks.unwrap_or(0)),
        stime_clock_ticks: ClockTicks(stat.stime_clock_ticks.unwrap_or(0)),
        priority: OrdinalI32(stat.priority.unwrap_or(0)),
        rt_priority: OrdinalU32(stat.rt_priority.unwrap_or(0)),
        core_forceidle_sum: MonotonicNs(sched.core_forceidle_sum.unwrap_or(0)),
        fair_slice_ns: GaugeNs(sched.fair_slice_ns.unwrap_or(0)),
        // Dedup `nr_threads` to only the thread leader. Every
        // thread of the same tgid sees the same kernel-emitted
        // value; populating it on every thread would let any
        // Sum-style aggregator multiply the count by itself
        // across the group. Leader-only population means the
        // registry's `AggRule::MaxGaugeCount` surfaces the
        // largest process represented in the bucket — reading
        // "the biggest process in this group" rather than "how
        // many threads the kernel believes this group contains"
        // (which is already covered by the row count).
        nr_threads: GaugeCount(if tid == tgid {
            status.nr_threads.unwrap_or(0)
        } else {
            0
        }),
        smaps_rollup_kb,
        rchar: Bytes(io.rchar.unwrap_or(0)),
        wchar: Bytes(io.wchar.unwrap_or(0)),
        syscr: MonotonicCount(io.syscr.unwrap_or(0)),
        syscw: MonotonicCount(io.syscw.unwrap_or(0)),
        read_bytes: Bytes(io.read_bytes.unwrap_or(0)),
        write_bytes: Bytes(io.write_bytes.unwrap_or(0)),
        cancelled_write_bytes: Bytes(io.cancelled_write_bytes.unwrap_or(0)),
        // Taskstats fields land here at zero defaults; the caller
        // (`capture_with` / `capture_pid_with`) overwrites them
        // after the per-tid `TaskstatsClient::query_tid` call,
        // mirroring how `allocated_bytes` / `deallocated_bytes`
        // are placeholdered above and then filled by the jemalloc
        // probe path. Zero defaults are correct for the
        // best-effort contract: a kernel without `CONFIG_TASKSTATS`
        // / `CONFIG_TASK_DELAY_ACCT`, a host with `delayacct=off`
        // at runtime, a process without `CAP_NET_ADMIN`, or a tid
        // that exited before `query_tid` succeeded all collapse
        // to zero per-field.
        cpu_delay_count: MonotonicCount(0),
        cpu_delay_total_ns: MonotonicNs(0),
        cpu_delay_max_ns: PeakNs(0),
        cpu_delay_min_ns: PeakNs(0),
        blkio_delay_count: MonotonicCount(0),
        blkio_delay_total_ns: MonotonicNs(0),
        blkio_delay_max_ns: PeakNs(0),
        blkio_delay_min_ns: PeakNs(0),
        swapin_delay_count: MonotonicCount(0),
        swapin_delay_total_ns: MonotonicNs(0),
        swapin_delay_max_ns: PeakNs(0),
        swapin_delay_min_ns: PeakNs(0),
        freepages_delay_count: MonotonicCount(0),
        freepages_delay_total_ns: MonotonicNs(0),
        freepages_delay_max_ns: PeakNs(0),
        freepages_delay_min_ns: PeakNs(0),
        thrashing_delay_count: MonotonicCount(0),
        thrashing_delay_total_ns: MonotonicNs(0),
        thrashing_delay_max_ns: PeakNs(0),
        thrashing_delay_min_ns: PeakNs(0),
        compact_delay_count: MonotonicCount(0),
        compact_delay_total_ns: MonotonicNs(0),
        compact_delay_max_ns: PeakNs(0),
        compact_delay_min_ns: PeakNs(0),
        wpcopy_delay_count: MonotonicCount(0),
        wpcopy_delay_total_ns: MonotonicNs(0),
        wpcopy_delay_max_ns: PeakNs(0),
        wpcopy_delay_min_ns: PeakNs(0),
        irq_delay_count: MonotonicCount(0),
        irq_delay_total_ns: MonotonicNs(0),
        irq_delay_max_ns: PeakNs(0),
        irq_delay_min_ns: PeakNs(0),
        hiwater_rss_bytes: crate::metric_types::PeakBytes(0),
        hiwater_vm_bytes: crate::metric_types::PeakBytes(0),
    }
}

#[cfg(test)]
fn capture_thread(tgid: i32, tid: i32, pcomm: &str) -> ThreadState {
    let proc_root = Path::new(DEFAULT_PROC_ROOT);
    let comm = read_thread_comm_at(proc_root, tgid, tid).unwrap_or_default();
    capture_thread_at(proc_root, tgid, tid, pcomm, &comm, true)
}

/// Running tally for the per-snapshot jemalloc-probe summary line
/// emitted by [`capture_with`] and [`capture_pid_with`]. The
/// dominant `AttachError` tag and `ProbeError` tag are tracked so
/// the summary can surface a remediation hint when one error class
/// dominates (e.g. EPERM under YAMA).
#[derive(Debug, Default)]
struct ProbeSummary {
    tgids_walked: u64,
    jemalloc_detected: u64,
    probed_ok: u64,
    failed: u64,
    attach_tag_counts: BTreeMap<&'static str, u64>,
    probe_tag_counts: BTreeMap<&'static str, u64>,
}

impl ProbeSummary {
    /// Pick the most frequent ACTIONABLE error tag (across attach
    /// and probe failures) for the summary line. Ties resolve to
    /// REVERSE-alphabetical order so the output is deterministic:
    /// the comparator's secondary key is `b.0.cmp(a.0)` (note the
    /// argument flip), so when two tags share a count, the
    /// alphabetically-EARLIER tag wins (e.g. `dwarf-parse-failure`
    /// beats `ptrace-seize`).
    ///
    /// `jemalloc-not-found` and `readlink-failure` are filtered out
    /// of the attach side: both are the expected outcome on the bulk
    /// of system processes (most tgids are not jemalloc-linked, and
    /// short-lived ones routinely fail readlink mid-walk), so
    /// surfacing them as the operator-facing "dominant failure tag"
    /// would drown the actionable signal (privilege drops, stripped
    /// debuginfo, arch mismatch) under known-benign noise on every
    /// snapshot. The filter is the same matches! arm
    /// `try_attach_probe_for_tgid_at` uses to route those two tags
    /// to debug-level tracing rather than warn-level — the
    /// dominant-tag summary mirrors the same actionable/non-actionable
    /// cut. Probe tags are not filtered: every `ProbeError` variant
    /// is actionable.
    fn dominant_tag(&self) -> Option<&'static str> {
        self.attach_tag_counts
            .iter()
            .filter(|(t, _)| !matches!(**t, "jemalloc-not-found" | "readlink-failure"))
            .chain(self.probe_tag_counts.iter())
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(tag, _)| *tag)
    }

    /// True when `ptrace-seize` (or `ptrace-interrupt`) failures
    /// dominate, signalling a privilege issue. Used to gate the
    /// EPERM remediation hint.
    fn ptrace_dominates(&self) -> bool {
        let total_ptrace: u64 = self
            .probe_tag_counts
            .iter()
            .filter(|(t, _)| matches!(**t, "ptrace-seize" | "ptrace-interrupt"))
            .map(|(_, n)| *n)
            .sum();
        // Half of failures or more attributable to ptrace
        // privilege (ptrace-seize or ptrace-interrupt) — high
        // enough that the hint is useful, low enough that a few
        // EPERMs in an otherwise-clean run don't drown the
        // summary.
        self.failed > 0 && total_ptrace * 2 >= self.failed
    }

    /// Project the internal tally to the curated public surface.
    /// Drops the per-tag `attach_tag_counts` / `probe_tag_counts`
    /// maps (implementation detail) and surfaces only the
    /// counters + dominant tag string + privilege-dominant
    /// signal. Mirrors the actionable/non-actionable cut
    /// [`Self::dominant_tag`] uses, so `dominant_failure` is
    /// `None` exactly when the snapshot has zero actionable
    /// failures. `privilege_dominant` mirrors
    /// [`Self::ptrace_dominates`] so a downstream consumer can
    /// reproduce the EPERM-hint trigger condition without
    /// parsing the operator-facing tracing line.
    fn to_public(&self) -> CtprofProbeSummary {
        CtprofProbeSummary {
            tgids_walked: self.tgids_walked,
            jemalloc_detected: self.jemalloc_detected,
            probed_ok: self.probed_ok,
            failed: self.failed,
            dominant_failure: self.dominant_tag().map(|t| t.to_string()),
            privilege_dominant: self.ptrace_dominates(),
        }
    }
}

/// Internal tally of procfs read-level failures, threaded through
/// [`capture_thread_at_with_tally`] and projected to the public
/// surface via [`Self::to_public`]. Mirrors the [`ProbeSummary`] /
/// [`CtprofProbeSummary`] split: tracks per-tid context plus a
/// per-file-kind failure map, then drops the implementation-detail
/// shape (here the `&'static str` keys vs the public surface's
/// `String` keys, which serde-derive cleanly).
///
/// `tids_walked` is incremented once per tid the capture pass
/// attempts, regardless of whether the tid lands in the snapshot —
/// the bump happens at the call site (before invoking
/// `capture_thread_at_with_tally`), so a ghost-filtered tid still
/// counts as walked. The per-tid `pending_failures` set lets the
/// caller unwind a ghost-filtered tid's read-failure contributions
/// before the summary is finalized — see [`Self::commit_pending`] /
/// [`Self::discard_pending`].
#[derive(Debug, Default)]
struct ParseTally {
    tids_walked: u64,
    failures_by_file: BTreeMap<&'static str, u64>,
    /// Per-tid pending bumps held until the caller commits or
    /// discards based on the ghost filter. Cleared between tids.
    pending_failures: Vec<&'static str>,
    /// Committed total of negative dotted-ns values seen across
    /// the snapshot. The kernel's PN_SCHEDSTAT path (`%Ld.%06ld`
    /// in `kernel/sched/debug.c`) emits a leading `-` when a
    /// schedstat field carries a negative integer part — rare but
    /// observable on clock-skew / suspend-resume hosts. The
    /// capture-side parser previously folded these into the
    /// absent-counter zero silently; this tally surfaces the
    /// rate so an operator can spot a host whose schedstat values
    /// are routinely negative-and-zeroed.
    negative_dotted_values: u64,
    /// Per-tid pending negative-dotted bumps held until
    /// commit / discard, mirroring [`Self::pending_failures`].
    pending_negative_dotted: u64,
}

impl ParseTally {
    /// Record a per-file read failure for the current tid. Held
    /// pending until [`Self::commit_pending`] or
    /// [`Self::discard_pending`] resolves the tid's outcome.
    fn record_failure(&mut self, file_kind: &'static str) {
        self.pending_failures.push(file_kind);
    }

    /// Record a negative dotted-ns value seen during sched parse
    /// for the current tid. Held pending until
    /// [`Self::commit_pending`] / [`Self::discard_pending`].
    fn record_negative_dotted(&mut self) {
        self.pending_negative_dotted = self.pending_negative_dotted.saturating_add(1);
    }

    /// Commit the current tid's pending failures to the per-snapshot
    /// tally. Called when the tid lands in the snapshot.
    fn commit_pending(&mut self) {
        for kind in self.pending_failures.drain(..) {
            *self.failures_by_file.entry(kind).or_insert(0) += 1;
        }
        self.negative_dotted_values = self
            .negative_dotted_values
            .saturating_add(self.pending_negative_dotted);
        self.pending_negative_dotted = 0;
    }

    /// Discard the current tid's pending failures. Called when the
    /// ghost filter rejects the tid — the bumps would correspond to
    /// a thread the snapshot doesn't include, so they must not
    /// inflate the summary.
    fn discard_pending(&mut self) {
        self.pending_failures.clear();
        self.pending_negative_dotted = 0;
    }

    /// Total failures across every file kind. Read-side mirror of
    /// the public surface's `read_failures` field.
    fn total_failures(&self) -> u64 {
        self.failures_by_file.values().sum()
    }

    /// Pick the file kind with the most failures. Ties resolve to
    /// REVERSE-alphabetical order for determinism — the
    /// alphabetically-EARLIER tag wins (mirrors
    /// [`ProbeSummary::dominant_tag`]'s comparator).
    fn dominant_file(&self) -> Option<&'static str> {
        self.failures_by_file
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(tag, _)| *tag)
    }

    /// True when ≥ 50% of failures are in `schedstat` or `io` —
    /// the two procfs files gated by `CONFIG_SCHEDSTATS` /
    /// `CONFIG_TASK_IO_ACCOUNTING`. Mirrors
    /// [`ProbeSummary::ptrace_dominates`]'s shape: dominance gate
    /// at half-or-more, false when total is zero.
    fn kernel_config_dominates(&self) -> bool {
        let total = self.total_failures();
        if total == 0 {
            return false;
        }
        let kconfig: u64 = self
            .failures_by_file
            .iter()
            .filter(|(t, _)| matches!(**t, "schedstat" | "io"))
            .map(|(_, n)| *n)
            .sum();
        kconfig * 2 >= total
    }

    /// Project the internal tally to the curated public surface.
    fn to_public(&self) -> CtprofParseSummary {
        let read_failures = self.total_failures();
        let mut by_file = BTreeMap::new();
        for (k, v) in &self.failures_by_file {
            by_file.insert((*k).to_string(), *v);
        }
        CtprofParseSummary {
            tids_walked: self.tids_walked,
            read_failures,
            read_failures_by_file: by_file,
            dominant_read_failure: self.dominant_file().map(|t| t.to_string()),
            kernel_config_dominant: self.kernel_config_dominates(),
            negative_dotted_values: self.negative_dotted_values,
        }
    }
}

/// Stable EPERM remediation hint for the capture summary. References
/// `$(which ktstr)` rather than a hardcoded path so the suggestion
/// works regardless of where the binary is installed.
const PTRACE_EPERM_HINT: &str = "hint: re-run as root, or sudo setcap cap_sys_ptrace+eip $(which ktstr), or set kernel.yama.ptrace_scope=0";

/// Result of the stateless attach pass for a single tgid:
/// the procfs-derived `pcomm` (for tracing) plus the underlying
/// `attach_jemalloc_at` outcome. Carries no shared state, so it
/// can be assembled by rayon workers in parallel without locking.
struct AttachOutcome {
    pcomm: String,
    result: std::result::Result<
        crate::host_thread_probe::JemallocProbe,
        crate::host_thread_probe::AttachError,
    >,
}

/// Cache value for the per-`(dev, ino)` probe cache in
/// [`capture_with`]'s parallel probe phase. Captures BOTH the
/// `JemallocProbe` (for the success path) and the
/// `AttachError::tag()` string (for the failure path) so a
/// cache hit can re-apply the same `attach_tag_counts` /
/// `failed` bumps that the original miss applied via
/// [`record_attach_outcome`]. Without `failed_tag`, repeat
/// hits on a failed binary would credit only `tgids_walked` —
/// the actionable failure (and its dominant-tag accounting)
/// would be silently undercounted relative to actual attach
/// volume.
#[derive(Clone)]
struct CachedAttachResult {
    probe: Option<crate::host_thread_probe::JemallocProbe>,
    /// `None` for the success path (`probe.is_some()`); `Some`
    /// for every failure path, even non-actionable tags
    /// (`jemalloc-not-found`, `readlink-failure`) — the
    /// dominant-tag filter in [`ProbeSummary::dominant_tag`]
    /// excludes those, but `attach_tag_counts` itself records
    /// every tag for diagnostic completeness.
    failed_tag: Option<&'static str>,
}

/// Stateless half of the per-tgid attach: read `pcomm` and run
/// `attach_jemalloc_at` (the expensive ELF parse + DWARF walk).
/// No summary mutation — the result is paired with `pcomm` and
/// returned to the caller for application via
/// [`record_attach_outcome`]. Splitting attach from the summary
/// update lets the parallel probe phase in [`capture_with`] hold
/// the `summary_mutex` only for the cheap counter+tracing step,
/// rather than serialising every rayon worker on the slowest
/// call in the pipeline.
fn attach_probe_for_tgid_at(proc_root: &Path, tgid: i32) -> AttachOutcome {
    #[cfg(test)]
    {
        // Panic-injection seam: a test sets `PANIC_INJECT_TGID` to
        // a sentinel tgid value before calling `capture_with`. When
        // the rayon worker for that tgid enters this function, we
        // panic to model the failure mode where the ELF parse / DWARF
        // walk panics under fd exhaustion or OOM. The
        // `catch_unwind` wrapper in `capture_with`'s phase 1 must
        // absorb this and surface it through the summary as a
        // `worker-panic` attach tag without crashing the snapshot.
        let injected = PANIC_INJECT_TGID.load(std::sync::atomic::Ordering::Acquire);
        if injected != 0 && injected == tgid {
            // Non-string payload variant: when the bool seam is
            // armed, panic with a typed payload so the
            // `downcast_ref::<&str>` and `downcast_ref::<String>`
            // arms in `capture_with` both miss and the
            // `unwrap_or("<non-string panic payload>")` fallback
            // arm fires. Pinned by
            // `capture_with_rayon_worker_panic_non_string_payload_falls_back`.
            if PANIC_INJECT_NON_STRING.load(std::sync::atomic::Ordering::Acquire) {
                // u64 is `'static + Send`, so it satisfies the
                // `Box<dyn Any + Send>` payload bound but neither
                // downcasts to `&str` nor `String` — exactly the
                // shape the fallback arm guards against.
                std::panic::panic_any(0xDEADBEEFu64);
            }
            panic!("test: injected attach worker panic for tgid {tgid}");
        }
    }
    let pcomm = read_process_comm_at(proc_root, tgid).unwrap_or_default();
    let result = crate::host_thread_probe::attach_jemalloc_at(proc_root, tgid);
    AttachOutcome { pcomm, result }
}

/// Test-only seam for the panic-injection harness consumed by
/// [`attach_probe_for_tgid_at`]. Set to a non-zero tgid to make
/// the next attach call for that tgid panic; reset to 0 to
/// disable. The check fires on the rayon worker thread, so the
/// `catch_unwind` wrapper in [`capture_with`] is the only thing
/// that prevents the panic from propagating out of `pool.install`.
/// `cfg(test)` only — production builds carry no injection
/// surface.
#[cfg(test)]
static PANIC_INJECT_TGID: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Test-only companion seam for [`PANIC_INJECT_TGID`]: when
/// armed (`true`) before calling `capture_with`, the injected
/// panic uses a typed non-string payload (`std::panic::panic_any`
/// over a `u64`) instead of the default formatted-message
/// `panic!`. Lets a test exercise the
/// `unwrap_or("<non-string panic payload>")` fallback arm in
/// `capture_with`'s panic-handling block — the `downcast_ref`
/// chain misses both `&str` and `String` for non-string
/// payloads and must fall back rather than panicking on
/// `unwrap()`.
#[cfg(test)]
static PANIC_INJECT_NON_STRING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Stateful half of the per-tgid attach: apply `outcome` to
/// `summary` and emit one tracing event. Two attach-error tags
/// log at `debug` rather than `warn`: `jemalloc-not-found` (the
/// bulk of system processes are not jemalloc-linked, so this is
/// the dominant non-actionable outcome on a busy host) and
/// `readlink-failure` (a tgid that exited between the procfs
/// walk and `readlink(/proc/<pid>/exe)` is also routine — race-
/// with-exit on short-lived helpers). Every other variant logs
/// at `warn` because a jemalloc-linked target failing to attach
/// is actionable (privilege drop, stripped binary, …). The
/// matches! arm here is the same one [`ProbeSummary::dominant_tag`]
/// uses to filter the operator-facing summary, so the level
/// routing and the dominance ranking surface the same
/// actionable/non-actionable cut. No I/O — safe to call under a
/// short-held mutex from the parallel probe phase.
fn record_attach_outcome(
    tgid: i32,
    outcome: AttachOutcome,
    summary: &mut ProbeSummary,
) -> CachedAttachResult {
    summary.tgids_walked += 1;
    let AttachOutcome { pcomm, result } = outcome;
    match result {
        Ok(probe) => {
            summary.jemalloc_detected += 1;
            tracing::debug!(tgid, %pcomm, "ctprof probe: jemalloc detected");
            CachedAttachResult {
                probe: Some(probe),
                failed_tag: None,
            }
        }
        Err(err) => {
            let tag = err.tag();
            *summary.attach_tag_counts.entry(tag).or_insert(0) += 1;
            if matches!(tag, "jemalloc-not-found" | "readlink-failure") {
                tracing::debug!(tgid, %pcomm, tag, err = %err, "ctprof probe: attach skipped");
            } else {
                summary.failed += 1;
                tracing::warn!(tgid, %pcomm, tag, err = %err, "ctprof probe: attach failed");
            }
            CachedAttachResult {
                probe: None,
                failed_tag: Some(tag),
            }
        }
    }
}

/// Single-call wrapper around [`attach_probe_for_tgid_at`] +
/// [`record_attach_outcome`] for sequential callers (tests + the
/// per-pid `capture_pid_with` path) that don't need the
/// stateless/stateful split. The parallel probe phase in
/// [`capture_with`] calls the two halves separately so the
/// expensive attach runs outside the summary mutex.
fn try_attach_probe_for_tgid_at(
    proc_root: &Path,
    tgid: i32,
    summary: &mut ProbeSummary,
) -> Option<crate::host_thread_probe::JemallocProbe> {
    let outcome = attach_probe_for_tgid_at(proc_root, tgid);
    record_attach_outcome(tgid, outcome, summary).probe
}

/// Pull `(allocated_bytes, deallocated_bytes)` for one tid via the
/// pre-attached probe, recording the outcome in `summary` and
/// emitting a `tracing::warn!` once per failed tgid (the engine
/// shares the same `AttachError`/`ProbeError` taxonomy across every
/// tid of a tgid, so logging each tid would spam the operator).
fn probe_thread_recording(
    probe: &crate::host_thread_probe::JemallocProbe,
    tid: i32,
    tgid: i32,
    pcomm: &str,
    comm: &str,
    summary: &mut ProbeSummary,
    failed_tgids_logged: &mut std::collections::BTreeSet<i32>,
) -> (u64, u64) {
    match crate::host_thread_probe::probe_thread(probe, tid) {
        Ok(c) => {
            summary.probed_ok += 1;
            (c.allocated_bytes, c.deallocated_bytes)
        }
        Err(err) => {
            let tag = err.tag();
            *summary.probe_tag_counts.entry(tag).or_insert(0) += 1;
            summary.failed += 1;
            if failed_tgids_logged.insert(tgid) {
                tracing::warn!(
                    tgid,
                    tid,
                    %pcomm,
                    %comm,
                    tag,
                    err = %err,
                    "ctprof probe: probe_thread failed",
                );
            }
            (0, 0)
        }
    }
}

/// Emit the once-per-snapshot parse-summary line. Mirrors the
/// [`emit_probe_summary`] discipline: one info-level line with the
/// per-snapshot tally counts. Includes the dominant failure file
/// kind when any read failures landed, the kernel-config
/// remediation hint when `schedstat` / `io` dominate, and the
/// negative-dotted-value count when the parser saw any
/// schedstat fields with a leading `-`. The clauses are
/// suppressed when their underlying signal is zero so a clean
/// host emits a single short line.
fn emit_parse_summary(tally: &ParseTally) {
    let tids_walked = tally.tids_walked;
    let read_failures = tally.total_failures();
    let negative_dotted = tally.negative_dotted_values;
    let dominant_clause = tally
        .dominant_file()
        .map(|tag| format!(" (dominant: {tag})"))
        .unwrap_or_default();
    let kconfig_clause = if tally.kernel_config_dominates() {
        format!("; {PARSE_KCONFIG_HINT}")
    } else {
        String::new()
    };
    let negative_clause = if negative_dotted > 0 {
        format!(", {negative_dotted} negative-dotted values")
    } else {
        String::new()
    };
    tracing::info!(
        "ctprof parse: {tids_walked} tids walked, \
         {read_failures} read failures{negative_clause}\
         {dominant_clause}{kconfig_clause}",
    );
}

/// Emit the once-per-snapshot summary line. Includes the dominant
/// failure tag when any failures landed and an EPERM remediation
/// hint when ptrace privilege failures dominate.
fn emit_probe_summary(summary: &ProbeSummary) {
    let tgids_walked = summary.tgids_walked;
    let jemalloc_detected = summary.jemalloc_detected;
    let probed_ok = summary.probed_ok;
    let failed = summary.failed;
    if failed > 0 {
        let dominant = summary.dominant_tag().unwrap_or("?");
        if summary.ptrace_dominates() {
            tracing::info!(
                "ctprof probe: {tgids_walked} tgids walked, \
                 {jemalloc_detected} jemalloc detected, \
                 {probed_ok} probed OK, {failed} failed \
                 (dominant: {dominant}; {})",
                PTRACE_EPERM_HINT,
            );
        } else {
            tracing::info!(
                "ctprof probe: {tgids_walked} tgids walked, \
                 {jemalloc_detected} jemalloc detected, \
                 {probed_ok} probed OK, {failed} failed \
                 (dominant: {dominant})",
            );
        }
    } else {
        tracing::info!(
            "ctprof probe: {tgids_walked} tgids walked, \
             {jemalloc_detected} jemalloc detected, \
             {probed_ok} probed OK, {failed} failed",
        );
    }
}

/// Capture a complete host-wide snapshot under arbitrary procfs
/// and cgroup roots. Walks `<proc_root>` for every live tgid,
/// enumerates its threads, and assembles a [`CtprofSnapshot`]
/// with per-cgroup enrichment populated once per distinct cgroup
/// path (many threads share a cgroup; keep the walk
/// O(cgroups) rather than O(threads)). The default-roots
/// production entry point is [`capture`]; tests pass a tempdir
/// to exercise the walk against a synthetic tree.
///
/// `use_syscall_affinity` gates four real-host touchpoints —
/// (a) the [`crate::host_context::collect_host_context`] sweep
/// (kernel/CPU/memory/tunables read from the live host); (b)
/// phase 1, the parallel jemalloc-probe attach pass that walks
/// every tgid's `/proc/<pid>/exe` for ELF + DWARF metadata; (c)
/// `sched_getaffinity(2)` inside per-thread capture, with
/// fall-back to `Cpus_allowed_list:` on syscall failure;
/// (d) `emit_probe_summary` plus the [`CtprofProbeSummary`]
/// surfaced on the snapshot, both of which are skipped when
/// `use_syscall_affinity` is `false`: `emit_probe_summary` is
/// not called and `probe_summary` is `None`. Synthetic-tree
/// tests pass `false` so the staged procfs is read in isolation
/// (no `sched_getaffinity`, no ELF parses, no `host` block, no
/// `probe_summary`); production passes `true`.
///
/// Self-skip: the caller's own tgid is excluded from the per-tgid
/// probe-attach loop because `PTRACE_SEIZE` rejects self-attach
/// (the rayon `.filter(|&tgid| tgid != self_pid)` drops self
/// before the attach call). Phase 2 still iterates the full tgid
/// list including self_pid, and the per-tid lookup
/// `probe_map.get(&tgid).and_then(|p| p.as_ref())` returns `None`
/// for self_pid because phase 1 never inserted an entry; the
/// closure short-circuits via `.map(...).unwrap_or((0, 0))`,
/// leaving the jemalloc fields at the absent-counter default.
/// Every other procfs-derived
/// field populates normally — `capture_thread_at` runs
/// unconditionally per tid regardless of probe outcome.
fn capture_with(
    proc_root: &Path,
    cgroup_root: &Path,
    sys_root: &Path,
    use_syscall_affinity: bool,
) -> CtprofSnapshot {
    let captured_at_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let host = if use_syscall_affinity {
        Some(crate::host_context::collect_host_context())
    } else {
        None
    };
    // Linux pid_max is bounded above by 2^22 (kernel/pid.c —
    // PID_MAX_LIMIT) on every supported architecture, well
    // inside i32::MAX, so the u32 → i32 cast cannot wrap.
    let self_pid = std::process::id() as i32;
    let mut threads: Vec<ThreadState> = Vec::new();
    let mut failed_tgids_logged: std::collections::BTreeSet<i32> =
        std::collections::BTreeSet::new();

    // Phase 1: resolve probes in parallel via rayon. The expensive
    // ELF parse + DWARF walk runs concurrently across tgids, with
    // an inode cache (Mutex-wrapped) so duplicate binaries are
    // resolved only once. The result is a map of tgid → probe.
    //
    // Cache key shape: `(st_dev, st_ino)` of `/proc/<tgid>/exe`'s
    // metadata. Two tgids whose exes resolve to the same
    // `(dev, ino)` share a cache entry.
    //
    // Overlay-fs / container collision note: the kernel exposes
    // overlayfs files with the OVERLAY MOUNT's superblock device
    // (`dentry->d_sb->s_dev`, the overlayfs `struct super_block`
    // anonymous device number) and a synthetic `st_ino`. The
    // mapping happens in `fs/overlayfs/inode.c::ovl_map_dev_ino`
    // — both fields come from the overlay superblock's view, NOT
    // the underlying upper or lower layer. Two unrelated mounts
    // produce DIFFERENT `s_dev` values (each gets its own
    // anonymous bdev); two containers sharing a single mount of
    // the same lower-layer image-store path see the same `s_dev`
    // and the same hashed `st_ino` for that file. In the
    // shared-mount case a cached jemalloc attach result is
    // reused across containers — BENIGN, because the cached value
    // records "is this binary jemalloc-linked, and at what TSD
    // offset", which is a property of the ELF bytes and identical
    // across container instances of the same image. Mutable-
    // overlay writes (an upper-layer write that copies-up the
    // lower ELF) produce a NEW `(s_dev, st_ino)` pair within the
    // SAME overlay mount — `ovl_map_dev_ino` rehashes against
    // the new upper-layer inode — so the cache misses correctly
    // and re-resolves the rewritten binary.
    let tgids = iter_tgids_at(proc_root);
    let probe_cache: std::sync::Mutex<std::collections::HashMap<(u64, u64), CachedAttachResult>> =
        std::sync::Mutex::new(std::collections::HashMap::new());
    let summary_mutex = std::sync::Mutex::new(ProbeSummary::default());

    let probe_map: std::collections::HashMap<i32, Option<crate::host_thread_probe::JemallocProbe>> =
        if use_syscall_affinity {
            use rayon::prelude::*;
            // Scale parallelism by available CPU headroom: read
            // `<proc_root>/loadavg`, subtract from online CPU count,
            // clamp to [1, num_cpus/2 + 1]. Avoids drowning a hot
            // host. Routing the read through `proc_root` (rather
            // than `/proc` directly) keeps the parameterised-root
            // contract intact so synthetic-tree tests can stage
            // their own loadavg shape.
            let max_threads = {
                let num_cpus = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let load = std::fs::read_to_string(proc_root.join("loadavg"))
                    .ok()
                    .and_then(|s| s.split_whitespace().next()?.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let headroom = (num_cpus as f64 - load).max(1.0) as usize;
                headroom.clamp(1, num_cpus / 2 + 1)
            };
            // ThreadPoolBuilder::build can fail when the OS rejects
            // the per-thread `pthread_create` (RLIMIT_NPROC, kernel
            // task table at PID_MAX). Fall back to the global rayon
            // pool on Err — capture still completes, only loses the
            // bounded-headroom guarantee.
            let pool_result = rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build();
            let work = || {
                tgids
                    .par_iter()
                    .copied()
                    .filter(|&tgid| tgid != self_pid)
                    .map(|tgid| {
                        // Catch panics from the per-tgid attach pipeline so
                        // a single rogue worker (fd exhaustion, OOM during
                        // DWARF parse, or any panic-on-bug under
                        // `attach_jemalloc_at`) cannot tear down
                        // `pool.install` and the surrounding capture call.
                        // Without this guard, `rayon::ThreadPool::install`
                        // re-throws worker panics into the calling thread,
                        // collapsing the entire snapshot into an unwind on
                        // a single tgid's failure. On panic we record a
                        // `worker-panic` attach tag against the summary
                        // (counted under `failed`, surfaced in
                        // `dominant_failure` when it dominates) and return
                        // `(tgid, None)` so phase 2 still walks the tgid's
                        // threads with the absent-counter default. The tag
                        // is treated as actionable — a panicking attach is
                        // a bug or resource-exhaustion signal, distinct
                        // from the benign `jemalloc-not-found` /
                        // `readlink-failure` outcomes the dominant-tag
                        // filter suppresses.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let cache_key =
                                std::fs::metadata(proc_root.join(tgid.to_string()).join("exe"))
                                    .ok()
                                    .map(|m| {
                                        use std::os::unix::fs::MetadataExt;
                                        (m.dev(), m.ino())
                                    });

                            if let Some(key) = cache_key {
                                // `unwrap_or_else(into_inner)` on every
                                // shared-mutex lock so a prior worker
                                // panic that poisoned a lock cannot
                                // cascade-poison every subsequent worker
                                // — the catch_unwind arm below records
                                // the failure as a `worker-panic`
                                // attach-tag bump, and surviving workers
                                // should still make progress on the
                                // partially-mutated state rather than
                                // re-panicking out of `pool.install` and
                                // collapsing the snapshot.
                                let cached = probe_cache
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .get(&key)
                                    .cloned();
                                if let Some(cached_result) = cached {
                                    let mut s =
                                        summary_mutex.lock().unwrap_or_else(|e| e.into_inner());
                                    s.tgids_walked += 1;
                                    match &cached_result.failed_tag {
                                        None => {
                                            // Success path — original miss
                                            // already credited
                                            // `jemalloc_detected`. Re-apply
                                            // here so cache hits stay
                                            // symmetric with cache misses;
                                            // without this, only the first
                                            // sharer of a `(dev, ino)`
                                            // would count toward
                                            // `jemalloc_detected` and
                                            // every subsequent reuse
                                            // would silently undercount.
                                            s.jemalloc_detected += 1;
                                            tracing::debug!(
                                                tgid,
                                                "ctprof probe: cache hit (jemalloc)"
                                            );
                                        }
                                        Some(tag) => {
                                            // Failure path — re-apply the
                                            // SAME bookkeeping
                                            // [`record_attach_outcome`]
                                            // applied on the original
                                            // miss: bump
                                            // `attach_tag_counts[tag]`
                                            // unconditionally, and
                                            // `failed` for actionable
                                            // tags only (matching the
                                            // dominant-tag filter in
                                            // [`ProbeSummary::dominant_tag`]).
                                            // Without this, repeat hits
                                            // on a failed binary would
                                            // credit only `tgids_walked`
                                            // and the dominant-failure
                                            // signal would degrade as
                                            // shared-inode reuse climbs.
                                            // Logging stays at debug level
                                            // — the original miss already
                                            // emitted the warn-level event
                                            // for actionable tags; spamming
                                            // a warn per cache hit would
                                            // drown the operator log.
                                            *s.attach_tag_counts.entry(tag).or_insert(0) += 1;
                                            if !matches!(
                                                *tag,
                                                "jemalloc-not-found" | "readlink-failure"
                                            ) {
                                                s.failed += 1;
                                            }
                                            tracing::debug!(
                                                tgid,
                                                tag,
                                                "ctprof probe: cache hit (prior failure)"
                                            );
                                        }
                                    }
                                    cached_result.probe
                                } else {
                                    // Stateless attach (the expensive ELF parse +
                                    // DWARF walk) runs OUTSIDE the summary mutex
                                    // so rayon workers parallelise it. The lock
                                    // is only held for the cheap counter +
                                    // tracing application via `record_attach_outcome`.
                                    //
                                    // Shared-inode cache misses can produce
                                    // duplicate parses when N workers enter
                                    // simultaneously — all run the attach before
                                    // any inserts. The cache fully amortises
                                    // subsequent lookups; the duplicate work is
                                    // bounded by the rayon pool size.
                                    let outcome = attach_probe_for_tgid_at(proc_root, tgid);
                                    let mut s =
                                        summary_mutex.lock().unwrap_or_else(|e| e.into_inner());
                                    let res = record_attach_outcome(tgid, outcome, &mut s);
                                    drop(s);
                                    let probe = res.probe.clone();
                                    probe_cache
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .insert(key, res);
                                    probe
                                }
                            } else {
                                // No cache key — exe symlink unreadable. Same
                                // attach-outside-lock pattern as the cache-miss
                                // branch above; result is not cached because
                                // there's no key to file it under.
                                let outcome = attach_probe_for_tgid_at(proc_root, tgid);
                                let mut s = summary_mutex.lock().unwrap_or_else(|e| e.into_inner());
                                record_attach_outcome(tgid, outcome, &mut s).probe
                            }
                        }));
                        let probe = match result {
                            Ok(p) => p,
                            Err(panic_payload) => {
                                // Recover the panic message string for
                                // the operator log. The payload is a
                                // `Box<dyn Any + Send>` whose runtime
                                // type is `&'static str` for `panic!("…")`
                                // with a literal and `String` for
                                // `panic!("{…}", …)` with formatted args.
                                // Both of `attach_jemalloc_at`'s likely
                                // panic sites (and the test seam in
                                // `attach_probe_for_tgid_at`) panic with
                                // a formatted message → `String`. Other
                                // panic types (typed values, custom
                                // payloads) collapse to a placeholder so
                                // the log line still surfaces the tgid.
                                let panic_msg = panic_payload
                                    .downcast_ref::<&str>()
                                    .copied()
                                    .or_else(|| {
                                        panic_payload.downcast_ref::<String>().map(|s| s.as_str())
                                    })
                                    .unwrap_or("<non-string panic payload>");
                                // Bump counters to mirror what
                                // `record_attach_outcome` would have done
                                // for an attach error: tgids_walked++,
                                // worker-panic tag++, failed++. The lock
                                // may be poisoned if the inner panic
                                // happened mid-update of the summary, so
                                // recover via `PoisonError::into_inner`
                                // rather than `.unwrap()` — bumping a
                                // counter on partially-mutated state is
                                // strictly less bad than re-panicking out
                                // of the worker and tearing down
                                // `pool.install`.
                                let mut s = summary_mutex.lock().unwrap_or_else(|e| e.into_inner());
                                s.tgids_walked += 1;
                                *s.attach_tag_counts.entry("worker-panic").or_insert(0) += 1;
                                s.failed += 1;
                                tracing::error!(
                                    tgid,
                                    panic_msg,
                                    "ctprof probe: attach worker panicked; tgid skipped",
                                );
                                None
                            }
                        };
                        (tgid, probe)
                    })
                    .collect()
            };
            match pool_result {
                Ok(pool) => pool.install(work),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        max_threads,
                        "rayon ThreadPoolBuilder failed; falling back to global pool"
                    );
                    work()
                }
            }
        } else {
            std::collections::HashMap::new()
        };

    // `mut` is required because phase 2 below threads `&mut
    // summary` into `probe_thread_recording`.
    let mut summary = summary_mutex
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    // Tally for procfs read-level failures, surfaced as
    // `parse_summary` when the production path runs. Tests that
    // pass `use_syscall_affinity=false` skip the assignment so
    // the public field stays `None` — same discipline as
    // `probe_summary`.
    let mut parse_tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = if use_syscall_affinity {
        Some(&mut parse_tally)
    } else {
        None
    };

    // Open a single taskstats genetlink socket for the snapshot.
    // Best-effort: a kernel without `CONFIG_TASKSTATS`, a process
    // without `CAP_NET_ADMIN`, or any other open failure collapses
    // to `None` and every per-tid `query_tid` call short-circuits
    // through the absent-default zeros installed in
    // `capture_thread_at_with_tally`. Synthetic-tree tests pass
    // `use_syscall_affinity=false`, so the socket is never opened
    // — same discipline as the host-context / probe pass.
    let taskstats_client = if use_syscall_affinity {
        match crate::taskstats::TaskstatsClient::open() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "ctprof taskstats: open failed; delay-accounting and memory-watermark \
                     fields will be zero. Ensure the kernel was built with CONFIG_TASKSTATS \
                     (plus CONFIG_TASK_DELAY_ACCT for delay fields and CONFIG_TASK_XACCT for \
                     hiwater fields), the process holds CAP_NET_ADMIN, and the kernel was \
                     booted with `delayacct=on` (or sysctl `kernel.task_delayacct=1`)"
                );
                None
            }
        }
    } else {
        None
    };
    // Per-snapshot tally of `query_tid` outcomes. Allocated only
    // when the production-mode capture path runs (`use_syscall_affinity`
    // is true) — synthetic-tree tests skip it the same way they
    // skip `parse_summary` and `probe_summary`. Counters bump
    // even when `taskstats_client.is_none()` happened (open
    // failed) — the per-tid loop simply never reaches
    // `record_result` in that case, so every counter stays zero
    // and the operator sees a tally of all-zeros pointing at the
    // open-time tracing warning.
    let mut taskstats_tally: Option<crate::taskstats::TaskstatsSummary> = if use_syscall_affinity {
        Some(crate::taskstats::TaskstatsSummary::default())
    } else {
        None
    };

    // Phase 2: sequential per-tid walk + ptrace reads.
    for tgid in &tgids {
        let tgid = *tgid;
        let pcomm = read_process_comm_at(proc_root, tgid).unwrap_or_default();
        let probe: Option<&crate::host_thread_probe::JemallocProbe> = probe_map
            .get(&tgid)
            .and_then(|p: &Option<crate::host_thread_probe::JemallocProbe>| p.as_ref());
        for tid in iter_task_ids_at(proc_root, tgid) {
            if let Some(t) = tally_opt.as_mut() {
                t.tids_walked += 1;
            }
            let comm = read_thread_comm_at(proc_root, tgid, tid).unwrap_or_default();
            let (allocated_bytes, deallocated_bytes) = probe
                .map(|p| {
                    probe_thread_recording(
                        p,
                        tid,
                        tgid,
                        &pcomm,
                        &comm,
                        &mut summary,
                        &mut failed_tgids_logged,
                    )
                })
                .unwrap_or((0, 0));
            let mut t = capture_thread_at_with_tally(
                proc_root,
                tgid,
                tid,
                &pcomm,
                &comm,
                use_syscall_affinity,
                &mut tally_opt,
            );
            t.allocated_bytes = crate::metric_types::Bytes(allocated_bytes);
            t.deallocated_bytes = crate::metric_types::Bytes(deallocated_bytes);
            // Best-effort taskstats query for delay-accounting +
            // hiwater memory watermarks. tid > 0 invariant is
            // guaranteed by `iter_task_ids_at`'s `> 0` filter; the
            // u32 cast is therefore safe. Failures (the kernel
            // doesn't support taskstats, the tid raced exit, the
            // socket was never opened) fall through to the zero
            // defaults already installed in
            // `capture_thread_at_with_tally`. Each query result —
            // success or failure — feeds the per-snapshot tally
            // so the operator can distinguish "every tid raced
            // exit" from "CAP_NET_ADMIN missing" from "kernel
            // built without CONFIG_TASKSTATS" without parsing the
            // tracing log.
            if let Some(client) = taskstats_client.as_ref() {
                let result = client.query_tid(tid as u32);
                if let Some(tally) = taskstats_tally.as_mut() {
                    tally.record_result(&result);
                }
                if let Ok(ds) = result {
                    t.apply_delay_stats(&ds);
                }
            }
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
                if let Some(t) = tally_opt.as_mut() {
                    t.discard_pending();
                }
                continue;
            }
            if let Some(t) = tally_opt.as_mut() {
                t.commit_pending();
            }
            threads.push(t);
        }
    }
    let probe_summary = if use_syscall_affinity {
        emit_probe_summary(&summary);
        Some(summary.to_public())
    } else {
        None
    };
    let parse_summary = if use_syscall_affinity {
        emit_parse_summary(&parse_tally);
        Some(parse_tally.to_public())
    } else {
        None
    };
    let mut cgroup_stats: BTreeMap<String, CgroupStats> = BTreeMap::new();
    for t in &threads {
        if !t.cgroup.is_empty() && !cgroup_stats.contains_key(&t.cgroup) {
            cgroup_stats.insert(
                t.cgroup.clone(),
                read_cgroup_stats_at(cgroup_root, &t.cgroup),
            );
        }
    }
    let psi = read_host_psi_at(proc_root);
    let sched_ext = read_sched_ext_sysfs_at(sys_root);
    CtprofSnapshot {
        captured_at_unix_ns,
        host,
        threads,
        cgroup_stats,
        probe_summary,
        parse_summary,
        taskstats_summary: taskstats_tally,
        psi,
        sched_ext,
    }
}

/// Capture a complete host-wide snapshot against the default
/// procfs and cgroup roots (`/proc` and `/sys/fs/cgroup`).
/// Probes every jemalloc-linked tgid the walk reaches and
/// populates per-thread `allocated_bytes` / `deallocated_bytes`
/// from the jemalloc TSD counters; tgids the probe cannot attach
/// against (ptrace denied, not jemalloc-linked, stripped binary)
/// land their threads at the absent-counter default of 0 per the
/// best-effort capture contract.
///
/// # Cost
///
/// O(threads-on-host) for the procfs walk; additionally one ELF
/// open + DWARF parse for every tgid `attach_jemalloc` resolves
/// successfully, plus a ptrace seize/interrupt/waitpid/detach
/// round-trip per thread of those tgids. On a host with many
/// jemalloc-linked daemons (database / browser / runtime
/// processes) the probe path dominates the wall-clock cost.
/// Callers that need only one tgid's data should use
/// [`capture_pid`] to scope the walk.
pub fn capture() -> CtprofSnapshot {
    capture_with(
        Path::new(DEFAULT_PROC_ROOT),
        Path::new(DEFAULT_CGROUP_ROOT),
        Path::new(DEFAULT_SYS_ROOT),
        true,
    )
}

/// Capture a ctprof snapshot scoped to a single tgid.
///
/// Walks `/proc/<pid>/task` for thread enumeration but skips every
/// other tgid on the host, sidestepping the wall-clock cost (and
/// blast-radius) of the global probe pass that [`capture`] runs.
/// Probes the target tgid's jemalloc TSD counters when it is
/// jemalloc-linked and not the calling process; otherwise the
/// per-thread allocated / deallocated fields land at zero per the
/// best-effort capture contract.
///
/// Useful for tests and tools that already know which process they
/// care about — the resulting snapshot's `threads` vec only carries
/// entries for `pid`'s tgid (one entry per thread of that process).
/// `host` and `cgroup_stats` populate normally so the snapshot
/// stays self-describing.
pub fn capture_pid(pid: i32) -> CtprofSnapshot {
    capture_pid_with(
        Path::new(DEFAULT_PROC_ROOT),
        Path::new(DEFAULT_CGROUP_ROOT),
        Path::new(DEFAULT_SYS_ROOT),
        pid,
        true,
    )
}

/// `proc_root` + `cgroup_root` parameterised variant of
/// [`capture_pid`]. Lets tests stage a synthetic procfs / cgroupfs
/// for the capture walk without touching the real host.
///
/// `use_syscall_affinity` gates the same four real-host
/// touchpoints as [`capture_with`] — host-context collection,
/// the jemalloc probe attach (here scoped to the single target
/// `pid` rather than a phase-1 sweep across every tgid),
/// `sched_getaffinity(2)` inside per-thread capture, and
/// `emit_probe_summary` plus the [`CtprofProbeSummary`] on the
/// snapshot. Synthetic-tree tests pass `false` because the
/// staged procfs has no real ELF behind `/proc/<pid>/exe`;
/// production passes `true`. Self-skip parallels the global path:
/// when `pid == self_pid`, the `probe` binding is `None` (the
/// `&& pid != self_pid` guard skips the attach), and each tid's
/// `probe.as_ref().map(...).unwrap_or((0, 0))` short-circuits to
/// the absent-counter default for the jemalloc fields, with every
/// other procfs-derived field populated normally.
fn capture_pid_with(
    proc_root: &Path,
    cgroup_root: &Path,
    sys_root: &Path,
    pid: i32,
    use_syscall_affinity: bool,
) -> CtprofSnapshot {
    let captured_at_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let host = if use_syscall_affinity {
        Some(crate::host_context::collect_host_context())
    } else {
        None
    };
    // Linux pid_max is bounded above by 2^22 (kernel/pid.c —
    // PID_MAX_LIMIT) on every supported architecture, well
    // inside i32::MAX, so the u32 → i32 cast cannot wrap.
    let self_pid = std::process::id() as i32;
    let pcomm = read_process_comm_at(proc_root, pid).unwrap_or_default();
    let mut summary = ProbeSummary::default();
    let mut failed_tgids_logged: std::collections::BTreeSet<i32> =
        std::collections::BTreeSet::new();
    let probe = if use_syscall_affinity && pid != self_pid {
        try_attach_probe_for_tgid_at(proc_root, pid, &mut summary)
    } else {
        None
    };
    let mut threads: Vec<ThreadState> = Vec::new();
    let mut parse_tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = if use_syscall_affinity {
        Some(&mut parse_tally)
    } else {
        None
    };
    // Best-effort taskstats client — same discipline as `capture_with`.
    let taskstats_client = if use_syscall_affinity {
        match crate::taskstats::TaskstatsClient::open() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "ctprof taskstats: open failed; delay-accounting and memory-watermark \
                     fields will be zero. Ensure the kernel was built with CONFIG_TASKSTATS \
                     (plus CONFIG_TASK_DELAY_ACCT for delay fields and CONFIG_TASK_XACCT for \
                     hiwater fields), the process holds CAP_NET_ADMIN, and the kernel was \
                     booted with `delayacct=on` (or sysctl `kernel.task_delayacct=1`)"
                );
                None
            }
        }
    } else {
        None
    };
    // Per-snapshot tally — mirrors the `capture_with` discipline.
    // Allocated only under `use_syscall_affinity` so the
    // synthetic-tree code path keeps `taskstats_summary: None` on
    // the resulting snapshot, identical to `parse_summary` /
    // `probe_summary`.
    let mut taskstats_tally: Option<crate::taskstats::TaskstatsSummary> = if use_syscall_affinity {
        Some(crate::taskstats::TaskstatsSummary::default())
    } else {
        None
    };
    for tid in iter_task_ids_at(proc_root, pid) {
        if let Some(t) = tally_opt.as_mut() {
            t.tids_walked += 1;
        }
        let comm = read_thread_comm_at(proc_root, pid, tid).unwrap_or_default();
        let (allocated_bytes, deallocated_bytes) = probe
            .as_ref()
            .map(|p| {
                probe_thread_recording(
                    p,
                    tid,
                    pid,
                    &pcomm,
                    &comm,
                    &mut summary,
                    &mut failed_tgids_logged,
                )
            })
            .unwrap_or((0, 0));
        let mut t = capture_thread_at_with_tally(
            proc_root,
            pid,
            tid,
            &pcomm,
            &comm,
            use_syscall_affinity,
            &mut tally_opt,
        );
        t.allocated_bytes = crate::metric_types::Bytes(allocated_bytes);
        t.deallocated_bytes = crate::metric_types::Bytes(deallocated_bytes);
        if let Some(client) = taskstats_client.as_ref() {
            let result = client.query_tid(tid as u32);
            if let Some(tally) = taskstats_tally.as_mut() {
                tally.record_result(&result);
            }
            if let Ok(ds) = result {
                t.apply_delay_stats(&ds);
            }
        }
        if t.comm.is_empty() && t.start_time_clock_ticks == 0 {
            if let Some(t) = tally_opt.as_mut() {
                t.discard_pending();
            }
            continue;
        }
        if let Some(t) = tally_opt.as_mut() {
            t.commit_pending();
        }
        threads.push(t);
    }
    let probe_summary = if use_syscall_affinity {
        emit_probe_summary(&summary);
        Some(summary.to_public())
    } else {
        None
    };
    let parse_summary = if use_syscall_affinity {
        emit_parse_summary(&parse_tally);
        Some(parse_tally.to_public())
    } else {
        None
    };
    let mut cgroup_stats: BTreeMap<String, CgroupStats> = BTreeMap::new();
    for t in &threads {
        if !t.cgroup.is_empty() && !cgroup_stats.contains_key(&t.cgroup) {
            cgroup_stats.insert(
                t.cgroup.clone(),
                read_cgroup_stats_at(cgroup_root, &t.cgroup),
            );
        }
    }
    let psi = read_host_psi_at(proc_root);
    let sched_ext = read_sched_ext_sysfs_at(sys_root);
    CtprofSnapshot {
        captured_at_unix_ns,
        host,
        threads,
        cgroup_stats,
        probe_summary,
        parse_summary,
        taskstats_summary: taskstats_tally,
        psi,
        sched_ext,
    }
}

/// Capture a snapshot and write it to `path` in the canonical
/// zstd+JSON format. Wrapper over [`capture`] +
/// [`CtprofSnapshot::write`] so CLI code can stay a single
/// call.
pub fn capture_to(path: &Path) -> Result<()> {
    let snap = capture();
    snap.write(path)
        .with_context(|| format!("write ctprof snapshot to {}", path.display()))
}

// Test modules — alphabetized.
#[cfg(test)]
mod tests_capture;
#[cfg(test)]
mod tests_cgroup;
#[cfg(test)]
mod tests_helpers;
#[cfg(test)]
mod tests_parse;
#[cfg(test)]
mod tests_parse_summary;
#[cfg(test)]
mod tests_probe;
#[cfg(test)]
mod tests_snapshot;
#[cfg(test)]
mod tests_thread_state;
