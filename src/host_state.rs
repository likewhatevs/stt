//! Per-thread host-state profiler data model + capture layer.
//!
//! [`HostStateSnapshot`] is the serialized container for a single
//! host-wide per-thread profile. Capture produces one via the
//! `ktstr host-state capture -o snapshot.hst.zst` subcommand;
//! comparison reads two and joins them on the selected grouping
//! axis (pcomm, cgroup, or comm).
//!
//! Every field is cumulative-from-birth so probe timing does not
//! alter the output: the design principle is that a thread sampled
//! twice at different wall-clock instants produces the same numbers
//! so long as its cumulative counters have not rolled over. The
//! jemalloc per-thread TSD counters
//! (`tsd_s.thread_allocated` / `thread_deallocated`) jemalloc
//! maintains unconditionally on its alloc/dalloc fast and slow
//! paths, so the ptrace-based attach this layer performs does not
//! perturb them; counters previously accumulated remain valid
//! across the brief stop the attach induces. Metrics not derivable
//! from cumulative state (e.g. perf_event_open counters that reset
//! on attachment) are intentionally absent from this capture layer.
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
    /// [`HostStateProbeSummary`] for the curated field set.
    pub probe_summary: Option<HostStateProbeSummary>,

    /// Procfs-read failure statistics for the snapshot, when the
    /// capture pass ran in production mode. Mirrors the
    /// `probe_summary` discipline: `None` indicates synthetic-tree
    /// tests skipped it (`use_syscall_affinity=false`); `Some(_)`
    /// carries the per-snapshot read-level failure tally — see
    /// [`HostStateParseSummary`].
    pub parse_summary: Option<HostStateParseSummary>,

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
/// stable token format is documented in the `ktstr host-state
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
/// let snap = ktstr::host_state::capture();
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
pub struct HostStateProbeSummary {
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
    /// ACTIONABLE — see the `ktstr host-state capture` CLI help
    /// for the full filter rule and tag taxonomy. Routine
    /// non-actionable outcomes (target not jemalloc-linked,
    /// `readlink` race-with-exit) do NOT contribute to this
    /// count.
    pub failed: u64,
    /// Tag string for the most-frequent actionable failure across
    /// all attach-and-probe failures. `None` when `failed == 0`.
    /// Stable single-word identifiers — the wire contract that
    /// downstream consumers match against. The full taxonomy is
    /// documented in the `ktstr host-state capture` CLI help.
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

impl HostStateProbeSummary {
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
/// let snap = ktstr::host_state::capture();
/// if let Some(ps) = &snap.parse_summary
///     && let Some(hint) = ps.kernel_config_hint()
/// {
///     eprintln!("{hint}");
/// }
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HostStateParseSummary {
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
    /// [`crate::host_state`]'s capture-side `ParseTally`), and
    /// unwound via `discard_pending` when the surrounding tid is
    /// rejected by the empty-comm + zero-start ghost filter so a
    /// busy host with mid-capture exits doesn't inflate this
    /// counter with bumps that correspond to threads the snapshot
    /// doesn't even contain.
    pub negative_dotted_values: u64,
}

impl HostStateParseSummary {
    /// Operator-facing hint when kernel-config-gated file failures
    /// dominate the snapshot. Returns `Some(&'static str)` naming
    /// the two `CONFIG_*` knobs that gate the affected files
    /// (`CONFIG_SCHEDSTATS` for `schedstat`, `CONFIG_TASK_IO_ACCOUNTING`
    /// for `io`), or `None` when [`Self::kernel_config_dominant`]
    /// is `false`. Lets a downstream consumer surface a remediation
    /// pointer without parsing the log line or hand-rolling the
    /// gate, mirroring the [`HostStateProbeSummary::remediation_hint`]
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
/// [`crate::host_state_compare::aggregate`] breaks the
/// categorical-mode count-ties (rules
/// [`crate::host_state_compare::AggRule::Mode`] /
/// [`crate::host_state_compare::AggRule::ModeChar`] /
/// [`crate::host_state_compare::AggRule::ModeBool`]) toward the
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

/// Per-thread cumulative resource profile.
///
/// Populated by the capture layer from `/proc/<tid>/{sched,status,
/// io,stat,comm,cgroup}`, `sched_getaffinity`, and (for jemalloc-
/// linked processes only, via ptrace + `process_vm_readv`) the
/// per-thread `tsd_s.thread_allocated` / `thread_deallocated` TLS
/// counters. All numeric fields are cumulative since thread birth
/// so the value is insensitive to probe-attach latency.
///
/// `Default` is implemented manually rather than derived because
/// the [`Self::state`] field needs `'~'` (the absent-value
/// sentinel) instead of `'\0'` (the `char` Default). See the
/// field doc on [`Self::state`] for why: `'\0'` lex-compares
/// SMALLER than every real kernel state letter, which would
/// poison [`crate::host_state_compare::AggRule::ModeChar`]
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
    /// runs on the same build; the grouping key under
    /// `--group-by pcomm`.
    pub pcomm: String,
    /// Thread name, read from `/proc/<tid>/comm`. Stable when the
    /// runtime assigns deterministic names (worker pools, async
    /// runtimes); the grouping key under `--group-by comm`.
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
    /// sets — see [`crate::host_state_compare::AffinitySummary`].
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
    /// (73). [`crate::host_state_compare::AggRule::ModeChar`]
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

    // -- scheduling (cumulative; /proc/<tid>/sched, needs CONFIG_SCHED_DEBUG) --
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
    /// [`crate::host_state_compare::AggRule::ModeBool`] dispatch
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
    /// Wakeups attributed to the idle path; `/proc/<tid>/sched`
    /// `nr_wakeups_idle`. KNOWN DEAD COUNTER — never incremented
    /// in mainline (registered for parser-completeness).
    /// Currently typed `MonotonicCount` for parser uniformity;
    /// migration to [`crate::metric_types::DeadCounter`] is queued
    /// (it requires a no-op aggregation arm in `AggRule` and a
    /// registry edit, both deliberately out of scope for this
    /// batch). The newtype already exists with no reduction trait
    /// impls so a future field-level flip will fail to compile
    /// against any `Summable`-bound `AggRule` variant.
    pub nr_wakeups_idle: crate::metric_types::MonotonicCount,
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
    /// Migrations skipped under cache-cold heuristic (the source
    /// CPU's cache was deemed too cold to warrant moving).
    /// `/proc/<tid>/sched` `nr_migrations_cold`, plain u64 via
    /// `P_SCHEDSTAT`. KNOWN DEAD COUNTER — no kernel writer
    /// touches `p->stats.nr_migrations_cold` on any current
    /// 6.16 / 7.1 code path; the field is preserved in
    /// `task_struct` for `/proc/<tid>/sched` line-number
    /// stability. Currently typed `MonotonicCount` for parser
    /// uniformity; migration to
    /// [`crate::metric_types::DeadCounter`] is queued (requires
    /// a no-op aggregation arm in `AggRule` and a registry
    /// edit, deliberately out of scope for this batch). Zero on
    /// kernels without `CONFIG_SCHEDSTATS` regardless.
    pub nr_migrations_cold: crate::metric_types::MonotonicCount,
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
    /// Total nanoseconds the task spent off-CPU — voluntary
    /// sleep (`TASK_INTERRUPTIBLE`) PLUS involuntary block
    /// (`TASK_UNINTERRUPTIBLE`). Populated from
    /// `/proc/<tid>/sched`'s `sum_sleep_runtime` key (kernel
    /// emits `ms.ns_remainder` via `PN_SCHEDSTAT`; the parser
    /// reconstructs full ns).
    ///
    /// `__update_stats_enqueue_sleeper` (`kernel/sched/stats.c`)
    /// adds the sleeper's elapsed wall-clock window to this
    /// counter regardless of which sleep state the task was in,
    /// so `sleep_sum` is the full off-CPU total. To recover pure
    /// voluntary sleep, subtract [`Self::block_sum`]:
    /// `voluntary_sleep = sleep_sum - block_sum`.
    ///
    /// There is no `sleep_count` counterpart: the kernel does
    /// not emit one — the scheduler records the aggregate
    /// runtime but not the sleep-event count separately from
    /// `nr_wakeups`, which already covers the wake-side tally.
    /// Zero on kernels without `CONFIG_SCHEDSTATS`. Zero under
    /// sched_ext: `__update_stats_enqueue_sleeper` is called
    /// from CFS/RT/DL paths only.
    pub sleep_sum: crate::metric_types::MonotonicNs,
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
    /// [`HostStateProbeSummary::dominant_failure`] (the per-tag
    /// plurality) and
    /// [`HostStateProbeSummary::privilege_dominant`] (the EPERM
    /// remediation gate, true when ptrace tags account for ≥ 50%
    /// of `failed`), reachable via
    /// [`HostStateSnapshot::probe_summary`]; the per-tag taxonomy
    /// is documented in the `ktstr host-state capture` CLI help.
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
    /// Cumulative blkio delay accrued by this thread, in
    /// USER_HZ clock ticks. `/proc/<tid>/stat` field 42, emitted
    /// via `seq_put_decimal_ull(m, " ", delayacct_blkio_ticks(task))`
    /// at `fs/proc/array.c:639`. Counts wall-clock time the task
    /// was blocked waiting on disk I/O — distinct from
    /// [`Self::iowait_sum`] (which the kernel maintains under
    /// schedstat) because this counter is the delayacct subsystem's
    /// blkio bucket.
    ///
    /// Two-stage gating: (1) build-time
    /// `CONFIG_TASK_DELAY_ACCT` — when not built, the
    /// `static inline delayacct_blkio_ticks()` at
    /// `include/linux/delayacct.h` returns 0 unconditionally;
    /// (2) runtime toggle `delayacct_on` in `kernel/delayacct.c`
    /// driven by either the `delayacct` boot param
    /// (`__setup("delayacct", ...)` at `kernel/delayacct.c:48`)
    /// or the `kernel.task_delayacct` sysctl (declared at
    /// `kernel/delayacct.c:80`). When the runtime toggle is
    /// off, the increment paths
    /// (`delayacct_blkio_start`/`_end`) are gated behind
    /// `static_branch_unlikely(&delayacct_key)` and become
    /// no-ops, so the counter stays at zero even on a kernel
    /// built with `CONFIG_TASK_DELAY_ACCT=y`. Same USER_HZ
    /// scaling as [`Self::utime_clock_ticks`].
    pub delayacct_blkio_ticks: crate::metric_types::ClockTicks,

    // -- /proc/<tid>/sched additions (counters + ordinal + slice gauge) --
    /// Wakeups onto an idle CPU via the passive-wakeup fast path.
    /// `/proc/<tid>/sched` `nr_wakeups_passive`, plain u64 via
    /// the `P_SCHEDSTAT(F)` macro that expands to
    /// `schedstat_val(p->stats.F)` at `kernel/sched/debug.c:1274`
    /// (the line emitting this specific counter is
    /// `kernel/sched/debug.c:1314`). KNOWN DEAD COUNTER on
    /// mainline — the kernel never increments
    /// `task->stats.nr_wakeups_passive` anywhere under `kernel/`.
    /// Captured for parser-completeness and forward
    /// compatibility with downstream kernels that may wire the
    /// increment. Always zero on mainline 6.x. Mirrors the
    /// existing `nr_wakeups_idle` and `nr_migrations_cold`
    /// dead-counter precedent. Currently typed `MonotonicCount`
    /// for parser uniformity; migration to
    /// [`crate::metric_types::DeadCounter`] is queued (requires
    /// a no-op aggregation arm in `AggRule` and a registry
    /// edit, deliberately out of scope for this batch).
    pub nr_wakeups_passive: crate::metric_types::MonotonicCount,
    /// Cumulative time this task forced its SMT sibling idle for
    /// core-scheduling, in nanoseconds. `/proc/<tid>/sched`
    /// `core_forceidle_sum`, dotted ms.ns format via
    /// `PN_SCHEDSTAT` (`kernel/sched/debug.c:1335`).
    /// Reconstructed to full ns via the same
    /// `parsed_ns_from_dotted` helper as `wait_sum` /
    /// `sleep_sum` / `block_sum`.
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
    /// with [`HostStateSnapshot::threads`] (the snapshot's own
    /// `Vec<ThreadState>`). `/proc/<pid>/status` `Threads:` line
    /// emitted at `fs/proc/array.c:290` via
    /// `seq_put_decimal_ull(m, "Threads:\t", num_threads)`.
    /// Identical for every thread of the same tgid.
    ///
    /// Capture-side dedup: the field is populated ONLY on the
    /// thread leader (tid == tgid) and zero for non-leader
    /// threads of the same process. The registry pairs this with
    /// [`crate::host_state_compare::AggRule::MaxGaugeCount`] (not
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
            nice: Default::default(),
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
            nr_wakeups_idle: Default::default(),
            nr_wakeups_affine: Default::default(),
            nr_wakeups_affine_attempts: Default::default(),
            nr_migrations: Default::default(),
            nr_migrations_cold: Default::default(),
            nr_forced_migrations: Default::default(),
            nr_failed_migrations_affine: Default::default(),
            nr_failed_migrations_running: Default::default(),
            nr_failed_migrations_hot: Default::default(),
            wait_sum: Default::default(),
            wait_count: Default::default(),
            wait_max: Default::default(),
            sleep_sum: Default::default(),
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
            delayacct_blkio_ticks: Default::default(),
            nr_wakeups_passive: Default::default(),
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
        }
    }
}

impl ThreadState {
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

/// Per-cgroup enrichment record attached to [`HostStateSnapshot`].
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
/// [`crate::host_state_compare::flatten_cgroup_stats`] applies
/// per-domain (max for limits, min for floors, saturating_add
/// for counters) without conflating across domains.
///
/// **Schema break (#61):** the previous flat shape (4 fields:
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
/// as [`CgroupStats::cpu_usage_usec`], so the existing
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
/// ([`HostStateSnapshot::psi`]) and per-cgroup
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

/// Parse one PSI file's contents. The kernel emits one or two
/// lines (`some` then `full`), each formatted by `seq_printf` at
/// `kernel/sched/psi.c:1284`. Lines are tokenized by whitespace;
/// each token is `key=value`. Unknown keys are ignored so a
/// future kernel that adds a 4th avg or new field doesn't break
/// the parser. Missing fields default to 0 (matching the
/// absent-counter contract used elsewhere in this module).
fn parse_psi(raw: &str) -> PsiResource {
    let mut out = PsiResource::default();
    for line in raw.lines() {
        let mut tokens = line.split_whitespace();
        let Some(prefix) = tokens.next() else {
            continue;
        };
        let half = match prefix {
            "some" => &mut out.some,
            "full" => &mut out.full,
            _ => continue,
        };
        for tok in tokens {
            let Some((key, value)) = tok.split_once('=') else {
                continue;
            };
            match key {
                "avg10" => half.avg10 = parse_centi_percent(value),
                "avg60" => half.avg60 = parse_centi_percent(value),
                "avg300" => half.avg300 = parse_centi_percent(value),
                "total" => half.total_usec = value.parse::<u64>().unwrap_or(0),
                _ => {}
            }
        }
    }
    out
}

/// Convert `"N.NN"` (kernel `%lu.%02lu` format from psi.c:1284)
/// to `N * 100 + NN` (centi-percent integer). On malformed input
/// returns 0, matching the absent-counter default contract.
/// Saturates at u16::MAX to guard against pathological input.
///
/// The kernel always emits a 2-digit zero-padded fraction
/// (`%02lu`), but a robust parser zero-pads its own input to
/// exactly 2 digits before combining: a stray `"1.5"` (one
/// fractional digit) must read as `150` (1.50%), not `105`
/// (1.05%); a stray `"1.501"` (three fractional digits) is
/// truncated to `1.50` rather than producing
/// `1*100 + 501 = 601`. Mirrors the
/// [`parsed_ns_from_dotted`] helper's zero-pad-to-six discipline.
fn parse_centi_percent(s: &str) -> u16 {
    let (int_part, frac_part) = s.split_once('.').unwrap_or((s, ""));
    let Ok(int) = int_part.parse::<u32>() else {
        return 0;
    };
    let frac = if frac_part.is_empty() {
        0
    } else {
        // Zero-pad-to-2 then truncate-to-2: "5" → "50", "501" →
        // "50". Matches the kernel's `%02lu` format width
        // exactly so a parser-side roundtrip can never under- or
        // over-count the fractional weight.
        let padded: String = frac_part
            .chars()
            .chain(std::iter::repeat('0'))
            .take(2)
            .collect();
        padded.parse::<u32>().unwrap_or(0)
    };
    let combined = int.saturating_mul(100).saturating_add(frac);
    combined.try_into().unwrap_or(u16::MAX)
}

/// Read host-level PSI files (`<proc_root>/pressure/{cpu,memory,io,irq}`)
/// and populate a [`Psi`] bundle. Each file is read independently;
/// absent files (older kernels missing irq.pressure, or hosts
/// with CONFIG_PSI off) collapse to the all-zero default per the
/// absent-counter contract.
fn read_host_psi_at(proc_root: &Path) -> Psi {
    let pressure_dir = proc_root.join("pressure");
    Psi {
        cpu: read_psi_file_at(&pressure_dir.join("cpu")),
        memory: read_psi_file_at(&pressure_dir.join("memory")),
        io: read_psi_file_at(&pressure_dir.join("io")),
        irq: read_psi_file_at(&pressure_dir.join("irq")),
    }
}

/// Read global sched_ext sysfs state from
/// `<sys_root>/kernel/sched_ext/`. Returns `None` when the
/// directory itself is absent (CONFIG_SCHED_CLASS_EXT=n
/// kernels never expose it). Per-file misses default the
/// affected field to zero / empty string per the
/// absent-counter contract — a future kernel that adds new
/// global attrs (and that we haven't surfaced as fields yet)
/// won't break the parser; old kernels missing one or more of
/// the existing five collapse cleanly.
fn read_sched_ext_sysfs_at(sys_root: &Path) -> Option<SchedExtSysfs> {
    let dir = sys_root.join("kernel").join("sched_ext");
    // No `tally` arg: directory presence (Option<SchedExtSysfs>)
    // is THE not-built signal; per-attr misses collapse silently
    // per the absent-counter contract.
    if !dir.exists() {
        return None;
    }
    Some(SchedExtSysfs {
        state: fs::read_to_string(dir.join("state"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
        switch_all: read_sysfs_u64(&dir.join("switch_all")),
        nr_rejected: read_sysfs_u64(&dir.join("nr_rejected")),
        hotplug_seq: read_sysfs_u64(&dir.join("hotplug_seq")),
        enable_seq: read_sysfs_u64(&dir.join("enable_seq")),
    })
}

/// Read a single-line u64 sysfs file. Trims trailing newline,
/// parses, defaults to 0 on read or parse failure (matches the
/// absent-counter contract).
fn read_sysfs_u64(path: &Path) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Read per-cgroup PSI files (`<cgroup>/{cpu,memory,io,irq}.pressure`)
/// and populate a [`Psi`] bundle. The four files are exposed by
/// `kernel/cgroup/cgroup.c:5453-5482`; the per-cgroup interface
/// uses the `<resource>.pressure` filename pattern rather than
/// the host-level `pressure/<resource>` directory layout.
fn read_cgroup_psi_at(cgroup_root: &Path, path: &str) -> Psi {
    let relative = path.strip_prefix('/').unwrap_or(path);
    let dir = if relative.is_empty() {
        cgroup_root.to_path_buf()
    } else {
        cgroup_root.join(relative)
    };
    Psi {
        cpu: read_psi_file_at(&dir.join("cpu.pressure")),
        memory: read_psi_file_at(&dir.join("memory.pressure")),
        io: read_psi_file_at(&dir.join("io.pressure")),
        irq: read_psi_file_at(&dir.join("irq.pressure")),
    }
}

/// Read one PSI file by path. Absent files or read errors
/// collapse to a default-zero [`PsiResource`].
fn read_psi_file_at(path: &Path) -> PsiResource {
    fs::read_to_string(path)
        .ok()
        .as_deref()
        .map(parse_psi)
        .unwrap_or_default()
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
    ///
    /// The decompression step is bounded by
    /// [`MAX_DECOMPRESSED_SNAPSHOT_BYTES`] — a payload that
    /// decompresses past that ceiling surfaces an error rather
    /// than allocating unbounded memory, guarding against a
    /// hostile zstd payload (zstd compresses pathologically well
    /// on repeated bytes).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let bytes = std::fs::read(path)
            .with_context(|| format!("read host-state snapshot from {}", path.display()))?;
        let json = decompress_capped(&bytes, MAX_DECOMPRESSED_SNAPSHOT_BYTES)
            .with_context(|| format!("zstd decompress host-state snapshot {}", path.display()))?;
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
        let compressed =
            zstd::encode_all(json.as_slice(), 3).context("zstd compress host-state snapshot")?;
        std::fs::write(path, compressed)
            .with_context(|| format!("write host-state snapshot to {}", path.display()))?;
        Ok(())
    }
}

/// Decompress a zstd payload into a `Vec<u8>` capped at
/// `max_decompressed` bytes — bombing out with an error if the
/// payload would expand past the ceiling. Reads through
/// `Read::take(cap + 1)` so a payload that decompresses to
/// exactly `cap` bytes is accepted while one that produces
/// `cap + 1` bytes (or more) is rejected — the +1 sentinel
/// distinguishes "EOF coincided with the cap" from "more data
/// behind the cap".
fn decompress_capped(bytes: &[u8], max_decompressed: u64) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    let decoder = zstd::stream::read::Decoder::new(bytes)?;
    let mut out = Vec::new();
    decoder
        .take(max_decompressed.saturating_add(1))
        .read_to_end(&mut out)?;
    if out.len() as u64 > max_decompressed {
        anyhow::bail!(
            "zstd-decompressed payload exceeds the {}-byte cap (decompression-bomb guard)",
            max_decompressed,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Capture layer: procfs readers + host walk.
// ---------------------------------------------------------------

/// Canonical file extension for a serialized snapshot.
pub const SNAPSHOT_EXTENSION: &str = "hst.zst";

/// Decompressed-size ceiling for [`HostStateSnapshot::load`].
/// Bounds the allocation a malicious or corrupted zstd payload
/// can force, since zstd compresses pathologically well on
/// repeated bytes (a few-KiB compressed blob can decompress to
/// gigabytes). 256 MiB covers any realistic production snapshot
/// (typical hosts run 1K-100K live threads) while bounding
/// worst-case allocation against hostile zstd payloads.
/// Public so a downstream consumer can size buffers against the
/// same ceiling without hardcoding the value.
pub const MAX_DECOMPRESSED_SNAPSHOT_BYTES: u64 = 256 * 1024 * 1024;

/// Default procfs root on Linux. The `_at` readers accept any
/// `&Path` so tests stage a synthetic tree under a tempdir; the
/// public readers delegate to those with this default.
pub const DEFAULT_PROC_ROOT: &str = "/proc";

/// Default cgroup v2 mount point.
pub const DEFAULT_CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// Default sysfs root. Tests pass a tempdir so they don't read
/// the live `/sys` tree (which would produce nondeterministic
/// `sched_ext` state depending on the host kernel config). The
/// public capture entry points pass this constant to read the
/// real sysfs tree at runtime.
pub const DEFAULT_SYS_ROOT: &str = "/sys";

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
fn policy_name(policy: i32) -> String {
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
fn iter_tgids_at(proc_root: &Path) -> Vec<i32> {
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
fn iter_task_ids_at(proc_root: &Path, tgid: i32) -> Vec<i32> {
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

/// Read `<proc_root>/<tgid>/comm` trimmed. `None` on read
/// failure or empty content.
fn read_process_comm_at(proc_root: &Path, tgid: i32) -> Option<String> {
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
fn read_thread_comm_at(proc_root: &Path, tgid: i32, tid: i32) -> Option<String> {
    let raw = fs::read_to_string(task_file(proc_root, tgid, tid, "comm")).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Selected fields parsed out of `/proc/<tgid>/task/<tid>/stat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StatFields {
    minflt: Option<u64>,
    majflt: Option<u64>,
    utime_clock_ticks: Option<u64>,
    stime_clock_ticks: Option<u64>,
    /// Field 18: kernel-internal priority (signed, distinct
    /// from `nice`). `seq_put_decimal_ll(m, " ", priority)` at
    /// `fs/proc/array.c:602`; the value is the post-bias
    /// scheduler priority (`task_prio(task)`).
    priority: Option<i32>,
    nice: Option<i32>,
    start_time_clock_ticks: Option<u64>,
    processor: Option<i32>,
    /// Field 40: real-time priority. `seq_put_decimal_ull(m,
    /// " ", task->rt_priority)` at `fs/proc/array.c:637`.
    /// Stored as `u32` to match `unsigned int
    /// task_struct::rt_priority` from `include/linux/sched.h`;
    /// non-zero only when the task runs SCHED_FIFO / SCHED_RR.
    rt_priority: Option<u32>,
    policy: Option<i32>,
    /// Field 42: cumulative blkio delay in USER_HZ clock ticks.
    /// `seq_put_decimal_ull(m, " ", delayacct_blkio_ticks(task))`
    /// at `fs/proc/array.c:639`. Counts time the task was blocked
    /// on disk I/O — non-zero only when CONFIG_TASK_DELAY_ACCT
    /// is enabled and the task accrued blkio time.
    delayacct_blkio_ticks: Option<u64>,
}

/// Pure parser for `/proc/<tgid>/task/<tid>/stat`. Per `proc(5)`,
/// field 2 (`comm`) is wrapped in parens and may contain
/// whitespace or `)`; every later field is indexed relative to
/// the LAST `)` in the line. Tail offsets (0-indexed from the
/// token past the final `)`):
///
/// | field | name                  | tail index |
/// |-------|-----------------------|------------|
/// | 10    | minflt                | 7          |
/// | 12    | majflt                | 9          |
/// | 14    | utime                 | 11         |
/// | 15    | stime                 | 12         |
/// | 18    | priority              | 15         |
/// | 19    | nice                  | 16         |
/// | 22    | starttime             | 19         |
/// | 39    | processor             | 36         |
/// | 40    | rt_priority           | 37         |
/// | 41    | policy                | 38         |
/// | 42    | delayacct_blkio_ticks | 39         |
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
    let get_u32 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<u32>().ok());
    let get_i32 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<i32>().ok());
    StatFields {
        minflt: get_u64(7),
        majflt: get_u64(9),
        utime_clock_ticks: get_u64(11),
        stime_clock_ticks: get_u64(12),
        priority: get_i32(15),
        nice: get_i32(16),
        start_time_clock_ticks: get_u64(19),
        processor: get_i32(36),
        rt_priority: get_u32(37),
        policy: get_i32(38),
        delayacct_blkio_ticks: get_u64(39),
    }
}

/// Read `<proc_root>/<tgid>/task/<tid>/stat` and parse fields.
/// Records a `"stat"` failure into `tally` on read error so the
/// per-snapshot [`HostStateParseSummary`] surfaces the dominant
/// procfs read-failure category. `tally: &mut None` skips the
/// recording (the synthetic-tree test pattern).
fn read_stat_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> StatFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "stat")) {
        Ok(raw) => parse_stat(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("stat");
            }
            StatFields::default()
        }
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
/// all-`None`. Records a `"schedstat"` failure on read error
/// when a tally is supplied.
fn read_schedstat_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> (Option<u64>, Option<u64>, Option<u64>) {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "schedstat")) {
        Ok(raw) => parse_schedstat(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("schedstat");
            }
            (None, None, None)
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct IoFields {
    rchar: Option<u64>,
    wchar: Option<u64>,
    syscr: Option<u64>,
    syscw: Option<u64>,
    read_bytes: Option<u64>,
    write_bytes: Option<u64>,
    cancelled_write_bytes: Option<u64>,
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
            "cancelled_write_bytes" => out.cancelled_write_bytes = parsed,
            _ => {}
        }
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/io` and parse fields.
/// Records an `"io"` failure into `tally` on read error (kernel
/// without `CONFIG_TASK_IO_ACCOUNTING` or per-tid race).
fn read_io_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> IoFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "io")) {
        Ok(raw) => parse_io(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("io");
            }
            IoFields::default()
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct StatusFields {
    voluntary_csw: Option<u64>,
    nonvoluntary_csw: Option<u64>,
    /// First non-whitespace character of the `State:` line value.
    /// Real kernel chars are `R` / `S` / `D` / `T` / `t` / `X` /
    /// `Z` / `P` / `I` (see `fs/proc/array.c::task_state_array`).
    /// `None` when the line is absent or blank — the capture site
    /// collapses to `'~'` (via `default_state_char`) which sorts
    /// strictly after every real kernel char in lex order, so
    /// the [`crate::host_state_compare::AggRule::ModeChar`]
    /// lex-smallest-wins tiebreak picks a real letter when one
    /// is present.
    state: Option<char>,
    /// `Cpus_allowed_list:` as a parsed sorted vec. Kept separate
    /// from the `sched_getaffinity` reader because status-file
    /// reads attribute to the target task without a syscall
    /// round-trip — useful when the caller cannot hold a pid
    /// long enough for the syscall without a race.
    cpus_allowed: Option<Vec<u32>>,
    /// `Threads:` value — `signal_struct->nr_threads` snapshot
    /// per `fs/proc/array.c:290`. Identical across every thread
    /// of the same tgid. The capture site dedups by populating
    /// [`ThreadState::nr_threads`] only on tid == tgid threads
    /// (see `capture_thread_at_with_tally`).
    nr_threads: Option<u64>,
}

fn parse_status(raw: &str) -> StatusFields {
    let mut out = StatusFields::default();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            // Kernel emits `State:\t<C> (<long>)` where <C> is the
            // single-letter code from `task_state_array`
            // (R/S/D/T/t/X/Z/P/I — nine codes, including the
            // off-by-default `P` parked state). First non-whitespace
            // char of the trimmed value is the letter;
            // `value.chars().next()` produces `None` only on a truly
            // empty line (which the split_once guards against
            // already).
            "State" => {
                out.state = value.chars().next();
            }
            "voluntary_ctxt_switches" => {
                out.voluntary_csw = value.parse::<u64>().ok();
            }
            "nonvoluntary_ctxt_switches" => {
                out.nonvoluntary_csw = value.parse::<u64>().ok();
            }
            "Cpus_allowed_list" => {
                out.cpus_allowed = crate::cpu_util::parse_cpu_list(value);
            }
            // `Threads:\t<num_threads>\n` per
            // `fs/proc/array.c:290`. Same value across every
            // thread of the same tgid; capture-side dedup picks
            // only the leader thread to avoid double-counting.
            "Threads" => {
                out.nr_threads = value.parse::<u64>().ok();
            }
            _ => {}
        }
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/status` and parse fields.
/// Records a `"status"` failure into `tally` on read error.
fn read_status_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> StatusFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "status")) {
        Ok(raw) => parse_status(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("status");
            }
            StatusFields::default()
        }
    }
}

/// Read the cgroup v2 path from
/// `<proc_root>/<tgid>/task/<tid>/cgroup`. Format per
/// `cgroup(7)`: one line per hierarchy, shape
/// `hid:controllers:path`. The unified (v2) hierarchy is keyed
/// `0::<path>`; mixed-mode hosts expose legacy v1 hierarchies
/// alongside, which this reader skips. `None` on read failure
/// or when no v2 line is present. Test-only — production callers
/// pipe through [`read_cgroup_at_with_tally`] so per-tid failures
/// surface in `parse_summary`.
#[cfg(test)]
fn read_cgroup_at(proc_root: &Path, tgid: i32, tid: i32) -> Option<String> {
    read_cgroup_at_with_tally(proc_root, tgid, tid, &mut None)
}

/// Records a `"cgroup"`
/// failure on read error (file absent — typical when the tid
/// exited mid-capture).
fn read_cgroup_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> Option<String> {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "cgroup")) {
        Ok(raw) => parse_cgroup_v2(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("cgroup");
            }
            None
        }
    }
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
    nr_wakeups_affine: Option<u64>,
    nr_wakeups_affine_attempts: Option<u64>,
    nr_migrations: Option<u64>,
    nr_migrations_cold: Option<u64>,
    nr_forced_migrations: Option<u64>,
    nr_failed_migrations_affine: Option<u64>,
    nr_failed_migrations_running: Option<u64>,
    nr_failed_migrations_hot: Option<u64>,
    wait_sum: Option<u64>,
    wait_count: Option<u64>,
    wait_max: Option<u64>,
    sleep_sum: Option<u64>,
    sleep_max: Option<u64>,
    block_sum: Option<u64>,
    block_max: Option<u64>,
    iowait_sum: Option<u64>,
    iowait_count: Option<u64>,
    exec_max: Option<u64>,
    slice_max: Option<u64>,
    /// `nr_wakeups_passive` from `/proc/<tid>/sched`, emitted via
    /// `P_SCHEDSTAT(nr_wakeups_passive)` at
    /// `kernel/sched/debug.c:1314`. Plain u64. KNOWN DEAD COUNTER:
    /// the kernel never increments this anywhere under
    /// `kernel/` on mainline (audited via tree-wide grep). Captured
    /// for parser-completeness and forward compatibility — a future
    /// kernel that wires the increment will surface immediately.
    /// Emission is gated by the `if (schedstat_enabled())` block
    /// at `kernel/sched/debug.c:1285` (the line lives inside that
    /// `if`), so on a host with schedstat off at runtime
    /// (`/proc/sys/kernel/sched_schedstats == 0`) the line is
    /// absent and the parser arm never fires — leaving the field
    /// at `None`.
    nr_wakeups_passive: Option<u64>,
    /// `core_forceidle_sum` from `/proc/<tid>/sched`, emitted via
    /// `PN_SCHEDSTAT(core_forceidle_sum)` at
    /// `kernel/sched/debug.c:1335`, build-gated on
    /// `CONFIG_SCHED_CORE`. Emission additionally lives inside
    /// the `if (schedstat_enabled())` block at
    /// `kernel/sched/debug.c:1285`, so on a host with schedstat
    /// off at runtime the line is absent and the parser arm
    /// never fires — leaving the field at `None`.
    /// Dotted ms.ns format like the other PN_SCHEDSTAT fields —
    /// reconstructed to full ns via [`parsed_ns_from_dotted`]. Counts
    /// time the task forced its SMT sibling idle for core-scheduling.
    /// `None` on kernels without `CONFIG_SCHED_CORE`, on hosts
    /// with schedstat disabled at runtime, or for tasks whose
    /// SMT cohort never accumulated forceidle.
    core_forceidle_sum: Option<u64>,
    /// `se.slice` from `/proc/<tid>/sched`, emitted via
    /// `P(se.slice)` at `kernel/sched/debug.c:1364`. Plain
    /// `%lld` integer (NOT dotted ns; the `P` macro uses
    /// `%lld`, not `PN`'s `%lld.%06ld`). Per-thread
    /// `p->se.slice` in nanoseconds. For fair-class tasks
    /// (SCHED_NORMAL / SCHED_BATCH) it is the instantaneous
    /// slice CFS is currently running the task with; for
    /// SCHED_EXT tasks it reflects stale `p->se.slice` state
    /// because ext-class schedulers maintain slice in
    /// `p->scx.slice` and do not refresh `p->se.slice`. The
    /// kernel emits this line ONLY when `fair_policy(p->policy)`
    /// holds, which (per `kernel/sched/sched.h:194,203`) is
    /// true for SCHED_NORMAL, SCHED_BATCH, AND — under
    /// `CONFIG_SCHED_CLASS_EXT` — SCHED_EXT. `None` for
    /// SCHED_DEADLINE / SCHED_RR / SCHED_FIFO / SCHED_IDLE.
    fair_slice_ns: Option<u64>,
    ext_enabled: Option<bool>,
}

/// Outcome of [`parsed_ns_from_dotted`]. Distinguishes the two
/// failure modes the caller may want to treat separately:
/// [`Self::Negative`] (kernel emitted a value with a leading
/// `-`, observable on clock-skew / suspend-resume hosts) is
/// counted into [`HostStateParseSummary::negative_dotted_values`]
/// so an operator can see that the snapshot's schedstat values
/// are routinely negative-and-zeroed; [`Self::Malformed`]
/// (non-numeric, empty, overflow) is the every-other failure
/// mode and stays silent (the data source is ill-formed in a way
/// the operator can't act on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseDottedNs {
    /// Trimmed input started with `-` — the kernel's PN_SCHEDSTAT
    /// `%Ld.%06ld` format emitted a negative integer part. The
    /// `parse::<u64>()` rejection is by design (u64 cannot
    /// represent the sign) but the SIGNAL is meaningful: a
    /// negative schedstat field is rare and worth surfacing
    /// rather than silently zeroing.
    ///
    /// Note: `-0.000000` would also route here, but is
    /// unreachable from real kernel output — `SPLIT_NS(0)`
    /// yields `(0, 0)` which `%Ld.%06ld` formats as
    /// `0.000000` with no leading sign. The parser still
    /// classifies the unreachable shape as `Negative` rather
    /// than special-casing it; a fixture that synthesizes
    /// `-0.000000` directly will land in this variant.
    Negative,
    /// Otherwise unparseable: non-numeric integer or fractional
    /// part, empty input, or u64 overflow on the
    /// `ms * 1_000_000 + ns_remainder` reconstruction.
    Malformed,
}

/// Parse a `PN_SCHEDSTAT`-emitted dotted nanosecond value into
/// full ns. The kernel formats schedstat fractional fields via
/// `__PSN(S, F) SEQ_printf(m, "%-45s:%14Ld.%06ld\n", S, SPLIT_NS(F))`
/// where `SPLIT_NS(x) = (x / 1_000_000, x % 1_000_000)` — the
/// integer part is MILLISECONDS, the 6-digit fractional part is
/// the NANOSECOND remainder within a millisecond. Reconstructing
/// the original ns value is `ms * 1_000_000 + ns_remainder`.
///
/// Tolerates fractional widths other than 6 (some test fixtures
/// emit `5000.25` or `7.999`) by zero-padding the right side
/// before parsing — `.25` becomes `.250000` (=250_000 ns), `.999`
/// becomes `.999000` (=999_000 ns). Truncates fractional widths
/// >6 to the first 6 digits.
///
/// Returns [`Err(ParseDottedNs::Negative)`] when EITHER:
/// - the trimmed integer part starts with `-` (kernel emitted
///   `-5.000000` for a magnitude ≥ 1ms negative SPLIT_NS via
///   `%Ld`), OR
/// - the trimmed fractional part starts with `-` (kernel
///   emitted `0.-000500` for a sub-millisecond negative
///   SPLIT_NS — `%Ld` on the `(x / 1_000_000)` integer part
///   yields `0` with no sign for x in `(-1_000_000, 0)`, and
///   `%06ld` on the `(x % 1_000_000)` remainder yields the
///   `-`). The sub-millisecond shape is the COMMON case for
///   clock-skew bugs because most schedstat deltas land
///   sub-millisecond — missing it would defeat the
///   negative-detection contract on the bulk of real
///   negatives.
///
/// The caller records the bump in the per-snapshot
/// [`HostStateParseSummary::negative_dotted_values`] before
/// folding to zero. Returns [`Err(ParseDottedNs::Malformed)`]
/// for any other parse failure (non-numeric, empty, overflow);
/// the caller folds to zero silently per the best-effort capture
/// contract.
///
/// The bare-integer (no dot) branch parses the value as raw ns
/// — used for test fixtures and graceful degradation; the
/// kernel's PN_SCHEDSTAT format always emits the dotted form.
/// Same negative-vs-malformed split applies to the bare-integer
/// branch so a stray bare-integer negative is also tallied.
fn parsed_ns_from_dotted(value: &str) -> Result<u64, ParseDottedNs> {
    if let Some((ms_str, ns_str)) = value.split_once('.') {
        let ms_trimmed = ms_str.trim();
        if ms_trimmed.starts_with('-') {
            return Err(ParseDottedNs::Negative);
        }
        // Sub-millisecond negative: kernel `%06ld` on a negative
        // remainder yields a leading `-` on the fractional side
        // even when the integer side is `0`. `0.-000500` is the
        // canonical shape for SPLIT_NS of a small (>-1ms)
        // negative — the integer-only check above misses it,
        // and it is the COMMON case for clock-skew bugs since
        // most schedstat deltas land sub-millisecond. Check the
        // fractional side BEFORE the chars().take(6) truncation
        // would otherwise swallow a sign-only fractional like
        // `-`.
        if ns_str.trim_start().starts_with('-') {
            return Err(ParseDottedNs::Negative);
        }
        let ms = ms_trimmed
            .parse::<u64>()
            .map_err(|_| ParseDottedNs::Malformed)?;
        let ns_part: String = ns_str.chars().take(6).collect();
        let padded = format!("{:0<6}", ns_part);
        let ns = padded
            .parse::<u64>()
            .map_err(|_| ParseDottedNs::Malformed)?;
        ms.checked_mul(1_000_000)
            .and_then(|x| x.checked_add(ns))
            .ok_or(ParseDottedNs::Malformed)
    } else {
        let trimmed = value.trim();
        if trimmed.starts_with('-') {
            return Err(ParseDottedNs::Negative);
        }
        trimmed.parse::<u64>().map_err(|_| ParseDottedNs::Malformed)
    }
}

/// Parse `/proc/<tgid>/task/<tid>/sched`. Requires
/// `CONFIG_SCHED_DEBUG`. Format is many lines of `key : value`
/// where the key is dot-delimited (`se.statistics.nr_wakeups`);
/// different kernel versions use `se.statistics.`, `stats.`,
/// or bare names. The reader matches on the LAST dot-delimited
/// segment to absorb that variation.
///
/// PN_SCHEDSTAT fields (`wait_sum`, `sum_sleep_runtime`,
/// `sum_block_runtime`, `iowait_sum`) emit a `ms.ns_remainder`
/// dotted format — reconstructed to full ns via
/// [`parsed_ns_from_dotted`]. P_SCHEDSTAT fields
/// (`wait_count`, `iowait_count`, `nr_wakeups*`,
/// `nr_migrations`) emit plain integers — parsed as `u64`.
///
/// `tally`, when supplied, records each negative dotted-ns parse
/// outcome via [`ParseTally::record_negative_dotted`] so the
/// per-snapshot summary surfaces the rate at which schedstat
/// fields were silently zeroed. `&mut None` skips the recording —
/// the synthetic-tree test path that doesn't carry a tally.
fn parse_sched(raw: &str, tally: &mut Option<&mut ParseTally>) -> SchedFields {
    let mut out = SchedFields::default();
    let mut parse_dotted = |value: &str| -> Option<u64> {
        match parsed_ns_from_dotted(value) {
            Ok(v) => Some(v),
            Err(ParseDottedNs::Negative) => {
                if let Some(t) = tally.as_mut() {
                    t.record_negative_dotted();
                }
                None
            }
            Err(ParseDottedNs::Malformed) => None,
        }
    };
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        let parsed_u64 = || value.parse::<u64>().ok();
        // `ext.enabled` is the only key the kernel emits with a
        // literal dot in the variable name (every other dot is a
        // namespace prefix like `se.statistics.`). Match on the full
        // key BEFORE the rsplit-on-dot fallback so a future kernel
        // line ending in `.enabled` cannot collide.
        if key == "ext.enabled" {
            out.ext_enabled = value.parse::<u64>().ok().map(|n| n != 0);
            continue;
        }
        let short = key.rsplit('.').next().unwrap_or(key);
        match short {
            "nr_wakeups" => out.nr_wakeups = parsed_u64(),
            "nr_wakeups_local" => out.nr_wakeups_local = parsed_u64(),
            "nr_wakeups_remote" => out.nr_wakeups_remote = parsed_u64(),
            "nr_wakeups_sync" => out.nr_wakeups_sync = parsed_u64(),
            "nr_wakeups_migrate" => out.nr_wakeups_migrate = parsed_u64(),
            "nr_wakeups_idle" => out.nr_wakeups_idle = parsed_u64(),
            "nr_wakeups_affine" => out.nr_wakeups_affine = parsed_u64(),
            "nr_wakeups_affine_attempts" => out.nr_wakeups_affine_attempts = parsed_u64(),
            "nr_migrations" => out.nr_migrations = parsed_u64(),
            "nr_migrations_cold" => out.nr_migrations_cold = parsed_u64(),
            "nr_forced_migrations" => out.nr_forced_migrations = parsed_u64(),
            "nr_failed_migrations_affine" => out.nr_failed_migrations_affine = parsed_u64(),
            "nr_failed_migrations_running" => out.nr_failed_migrations_running = parsed_u64(),
            "nr_failed_migrations_hot" => out.nr_failed_migrations_hot = parsed_u64(),
            "wait_sum" => out.wait_sum = parse_dotted(value),
            "wait_count" => out.wait_count = parsed_u64(),
            "wait_max" => out.wait_max = parse_dotted(value),
            // Kernel emits `sum_sleep_runtime` (see
            // `kernel/sched/debug.c` -> `proc_sched_show_task`); the
            // matching ThreadState field is named `sleep_sum` for
            // symmetry with `wait_sum` / `block_sum` / `iowait_sum`.
            // The kernel does not emit a `sleep_count` counterpart;
            // `nr_wakeups` (matched above) covers the wake-side
            // event tally.
            "sum_sleep_runtime" => out.sleep_sum = parse_dotted(value),
            "sleep_max" => out.sleep_max = parse_dotted(value),
            // Kernel emits `sum_block_runtime`; the matching
            // ThreadState field is `block_sum` for symmetry with
            // the other `*_sum` fields. There is no `block_count`
            // counterpart from the kernel — the schedstat printout
            // pairs `wait_sum/wait_count` and `iowait_sum/iowait_count`
            // but `sum_block_runtime` has no per-event counter.
            "sum_block_runtime" => out.block_sum = parse_dotted(value),
            "block_max" => out.block_max = parse_dotted(value),
            "iowait_sum" => out.iowait_sum = parse_dotted(value),
            "iowait_count" => out.iowait_count = parsed_u64(),
            "exec_max" => out.exec_max = parse_dotted(value),
            "slice_max" => out.slice_max = parse_dotted(value),
            // P_SCHEDSTAT plain integer; KNOWN dead counter on
            // mainline (ThreadState doc explains).
            "nr_wakeups_passive" => out.nr_wakeups_passive = parsed_u64(),
            // PN_SCHEDSTAT dotted ns; CONFIG_SCHED_CORE-gated. Same
            // ms.ns reconstruction as wait_sum / sleep_sum.
            "core_forceidle_sum" => out.core_forceidle_sum = parse_dotted(value),
            // P plain integer in ns. The kernel emits this only
            // for fair-policy tasks (`fair_policy(p->policy)` at
            // debug.c:1363); for other policies the line is absent
            // and `parsed_u64()` collapses to None.
            "slice" => out.fair_slice_ns = parsed_u64(),
            _ => {}
        }
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/sched` and parse fields.
/// Records a `"sched"` failure into `tally` on read error, plus
/// per-line negative-dotted-value bumps via `parse_sched`.
fn read_sched_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> SchedFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "sched")) {
        Ok(raw) => parse_sched(&raw, tally),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("sched");
            }
            SchedFields::default()
        }
    }
}

/// Parse `/proc/<pid>/smaps_rollup` contents into a key→u64-kB
/// map. Format per `__show_smap()` at
/// `fs/proc/task_mmu.c:1330-1368`: each kv line is
/// `<Name>:<whitespace><u64><whitespace>kB`. The kernel ALSO
/// emits a `<addr_start>-<addr_end> ---p <off> XX:XX <inode> [rollup]`
/// header line (built by `show_vma_header_prefix()` then
/// `seq_puts(m, "[rollup]\n")` at `task_mmu.c:1500-1503`). That
/// header CONTAINS a `:` (the device-major:minor pair `XX:XX`),
/// so a naive `split_once(':')` would mis-extract a junk key
/// (the whitespace-laden address range + flags + offset prefix)
/// with value 0 (the minor-device integer parses as the first
/// whitespace token of the value side). Real smaps_rollup keys
/// are single-word identifiers (Rss, Pss, Pss_Dirty, etc.) that
/// never contain whitespace or `-`; the address-range header
/// always contains both. Reject any line whose pre-`:` segment
/// carries either character.
///
/// Lines whose value field doesn't parse as u64 are silently
/// dropped (best-effort, matching the absent-counter contract).
fn parse_smaps_rollup(raw: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key_trimmed = key.trim();
        // Header-line guard: real smaps_rollup keys never
        // contain whitespace or `-`. Address-range headers
        // (`<addr_start>-<addr_end> ---p <off> XX:XX <inode>
        // [rollup]`) carry both.
        if key_trimmed.contains(char::is_whitespace) || key_trimmed.contains('-') {
            continue;
        }
        // Value field: whitespace-prefixed integer + " kB" suffix
        // (or rarely no suffix on a future kernel addition). The
        // first whitespace-token after trimming IS the integer;
        // dropping the unit suffix happens for free via
        // `split_ascii_whitespace().next()`.
        let Some(n_str) = value.split_ascii_whitespace().next() else {
            continue;
        };
        let Ok(n) = n_str.parse::<u64>() else {
            continue;
        };
        out.insert(key_trimmed.to_string(), n);
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/smaps_rollup` for the
/// thread leader (tid == tgid) and parse it. Non-leader threads
/// short-circuit to an empty map: the underlying mm_struct is
/// shared per-tgid, so reading from any thread yields identical
/// values, and capturing once per tgid avoids redundant
/// per-thread work. Records a `"smaps_rollup"` failure into
/// `tally` on read error.
///
/// Permission semantics: `/proc/<pid>/smaps_rollup` requires
/// `CAP_SYS_PTRACE` for processes the caller doesn't own (PID 1
/// being the canonical example). Read failure is treated as
/// best-effort — empty map, tally bump, no panic. Older kernels
/// (pre-4.14) lack the file entirely; same handling.
fn read_smaps_rollup_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> BTreeMap<String, u64> {
    if tid != tgid {
        // Leader-dedup: non-leader threads share the same
        // mm_struct, so the file would yield identical values.
        // Skip the read entirely.
        return BTreeMap::new();
    }
    match fs::read_to_string(task_file(proc_root, tgid, tid, "smaps_rollup")) {
        Ok(raw) => parse_smaps_rollup(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("smaps_rollup");
            }
            BTreeMap::new()
        }
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

/// Parse a cgroup v2 key-value file (one `<key> <u64>` per
/// line). Used for `memory.stat` and `memory.events`. Lines
/// the parser cannot fully decompose into a key + u64 are
/// silently skipped — a future kernel that introduces non-u64
/// values won't break the parser, just elide the offending key.
fn parse_kv_counters(raw: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let mut parts = line.split_ascii_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next() else { continue };
        let Ok(parsed) = value.parse::<u64>() else {
            continue;
        };
        out.insert(key.to_string(), parsed);
    }
    out
}

/// Parse a single-line LIMIT cgroup file (e.g. `memory.max`,
/// `memory.high`, `pids.max`). The literal token `max` means
/// "no limit" and yields `None`; a numeric value yields
/// `Some(u64)`. Whitespace-only or malformed input also yields
/// `None` (best-effort, matching the absent-counter contract).
///
/// Caller MUST NOT use this for FLOOR files (`memory.low`,
/// `memory.min`) — for floors, the literal token `max` means
/// "maximum protection", not "no floor", which is the semantic
/// opposite. Use [`parse_floor_value`] there instead.
fn parse_max_or_u64(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed == "max" {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

/// Parse a single-line FLOOR cgroup file (`memory.low`,
/// `memory.min`). The literal token `max` means
/// "maximum protection" — yields `Some(u64::MAX)` rather than
/// `None`, because FLOORS use `None` to mean "absent file"
/// only. A numeric value yields `Some(u64)`; whitespace-only or
/// malformed input yields `None` (absent-counter contract).
///
/// The semantic asymmetry vs. [`parse_max_or_u64`] is critical:
/// for limits, "max" is the absence of a cap (collapse to
/// `None`); for floors, "max" is a fully-protected floor (it
/// must NOT collapse to "no floor"). [`merge_min_option`] then
/// correctly picks `min(u64::MAX, 5G) = 5G` instead of None
/// when one contributor has full protection and another has a
/// concrete protection.
fn parse_floor_value(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed == "max" {
        return Some(u64::MAX);
    }
    trimmed.parse::<u64>().ok()
}

/// Parse `cpu.max` (one line, two whitespace-separated tokens:
/// `<quota|max> <period>`). Returns `(quota, period)` where
/// `quota` is `None` for the literal `max` token (no CFS
/// bandwidth cap) and `Some(usec)` otherwise; `period` defaults
/// to the kernel default of 100_000 µs when missing or
/// malformed.
fn parse_cpu_max(raw: &str) -> (Option<u64>, u64) {
    let mut parts = raw.split_ascii_whitespace();
    let quota_token = parts.next();
    let period_token = parts.next();
    let quota = quota_token.and_then(parse_max_or_u64_str);
    let period = period_token
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(CPU_MAX_DEFAULT_PERIOD_US);
    (quota, period)
}

/// Helper for [`parse_cpu_max`]: route a single token through
/// the same `max`-vs-u64 disambiguation as [`parse_max_or_u64`]
/// without committing to a string-trimmed input shape.
fn parse_max_or_u64_str(s: &str) -> Option<u64> {
    if s == "max" {
        return None;
    }
    s.parse::<u64>().ok()
}

/// Default CFS bandwidth period when `cpu.max` is absent or its
/// period token is unreadable. Matches the kernel default
/// returned by `default_bw_period_us()` at
/// `kernel/sched/sched.h:441`; child cgroups inherit this when
/// `cpu.max` is unset.
const CPU_MAX_DEFAULT_PERIOD_US: u64 = 100_000;

/// Populate a [`CgroupStats`] by reading the cgroup v2 files
/// for `path` under `cgroup_root`. Missing files collapse to
/// the struct's `Default` (zero / `None` per field semantics) —
/// the root cgroup is missing most knob files, and child
/// cgroups on hosts without `pids` enabled in
/// `cgroup.subtree_control` are also expected to lack
/// `pids.{current,max}`.
fn read_cgroup_stats_at(cgroup_root: &Path, path: &str) -> CgroupStats {
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
    let (max_quota_us, max_period_us) = fs::read_to_string(dir.join("cpu.max"))
        .ok()
        .as_deref()
        .map(parse_cpu_max)
        .unwrap_or((None, CPU_MAX_DEFAULT_PERIOD_US));
    let weight = fs::read_to_string(dir.join("cpu.weight"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let weight_nice = fs::read_to_string(dir.join("cpu.weight.nice"))
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok());

    let memory_current = fs::read_to_string(dir.join("memory.current"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let memory_max = fs::read_to_string(dir.join("memory.max"))
        .ok()
        .as_deref()
        .and_then(parse_max_or_u64);
    let memory_high = fs::read_to_string(dir.join("memory.high"))
        .ok()
        .as_deref()
        .and_then(parse_max_or_u64);
    let memory_low = fs::read_to_string(dir.join("memory.low"))
        .ok()
        .as_deref()
        .and_then(parse_floor_value);
    let memory_min = fs::read_to_string(dir.join("memory.min"))
        .ok()
        .as_deref()
        .and_then(parse_floor_value);
    let memory_stat = fs::read_to_string(dir.join("memory.stat"))
        .ok()
        .as_deref()
        .map(parse_kv_counters)
        .unwrap_or_default();
    let memory_events = fs::read_to_string(dir.join("memory.events"))
        .ok()
        .as_deref()
        .map(parse_kv_counters)
        .unwrap_or_default();

    let pids_current = fs::read_to_string(dir.join("pids.current"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let pids_max = fs::read_to_string(dir.join("pids.max"))
        .ok()
        .as_deref()
        .and_then(parse_max_or_u64);

    let psi = read_cgroup_psi_at(cgroup_root, path);

    CgroupStats {
        cpu: CgroupCpuStats {
            usage_usec: usage.unwrap_or(0),
            nr_throttled: throttled.unwrap_or(0),
            throttled_usec: throttled_usec.unwrap_or(0),
            max_quota_us,
            max_period_us,
            weight,
            weight_nice,
        },
        memory: CgroupMemoryStats {
            current: memory_current,
            max: memory_max,
            high: memory_high,
            low: memory_low,
            min: memory_min,
            stat: memory_stat,
            events: memory_events,
        },
        pids: CgroupPidsStats {
            current: pids_current,
            max: pids_max,
        },
        psi,
    }
}

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
/// per-snapshot [`HostStateParseSummary`] when the capture
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
        nr_wakeups_idle: MonotonicCount(sched.nr_wakeups_idle.unwrap_or(0)),
        nr_wakeups_affine: MonotonicCount(sched.nr_wakeups_affine.unwrap_or(0)),
        nr_wakeups_affine_attempts: MonotonicCount(sched.nr_wakeups_affine_attempts.unwrap_or(0)),
        nr_migrations: MonotonicCount(sched.nr_migrations.unwrap_or(0)),
        nr_migrations_cold: MonotonicCount(sched.nr_migrations_cold.unwrap_or(0)),
        nr_forced_migrations: MonotonicCount(sched.nr_forced_migrations.unwrap_or(0)),
        nr_failed_migrations_affine: MonotonicCount(sched.nr_failed_migrations_affine.unwrap_or(0)),
        nr_failed_migrations_running: MonotonicCount(
            sched.nr_failed_migrations_running.unwrap_or(0),
        ),
        nr_failed_migrations_hot: MonotonicCount(sched.nr_failed_migrations_hot.unwrap_or(0)),
        wait_sum: MonotonicNs(sched.wait_sum.unwrap_or(0)),
        wait_count: MonotonicCount(sched.wait_count.unwrap_or(0)),
        wait_max: PeakNs(sched.wait_max.unwrap_or(0)),
        sleep_sum: MonotonicNs(sched.sleep_sum.unwrap_or(0)),
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
        delayacct_blkio_ticks: ClockTicks(stat.delayacct_blkio_ticks.unwrap_or(0)),
        nr_wakeups_passive: MonotonicCount(sched.nr_wakeups_passive.unwrap_or(0)),
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
    fn to_public(&self) -> HostStateProbeSummary {
        HostStateProbeSummary {
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
/// [`HostStateProbeSummary`] split: tracks per-tid context plus a
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
    fn to_public(&self) -> HostStateParseSummary {
        let read_failures = self.total_failures();
        let mut by_file = BTreeMap::new();
        for (k, v) in &self.failures_by_file {
            by_file.insert((*k).to_string(), *v);
        }
        HostStateParseSummary {
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
) -> Option<crate::host_thread_probe::JemallocProbe> {
    summary.tgids_walked += 1;
    let AttachOutcome { pcomm, result } = outcome;
    match result {
        Ok(probe) => {
            summary.jemalloc_detected += 1;
            tracing::debug!(tgid, %pcomm, "host-state probe: jemalloc detected");
            Some(probe)
        }
        Err(err) => {
            let tag = err.tag();
            *summary.attach_tag_counts.entry(tag).or_insert(0) += 1;
            if matches!(tag, "jemalloc-not-found" | "readlink-failure") {
                tracing::debug!(tgid, %pcomm, tag, err = %err, "host-state probe: attach skipped");
            } else {
                summary.failed += 1;
                tracing::warn!(tgid, %pcomm, tag, err = %err, "host-state probe: attach failed");
            }
            None
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
    record_attach_outcome(tgid, outcome, summary)
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
                    "host-state probe: probe_thread failed",
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
        "host-state parse: {tids_walked} tids walked, \
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
                "host-state probe: {tgids_walked} tgids walked, \
                 {jemalloc_detected} jemalloc detected, \
                 {probed_ok} probed OK, {failed} failed \
                 (dominant: {dominant}; {})",
                PTRACE_EPERM_HINT,
            );
        } else {
            tracing::info!(
                "host-state probe: {tgids_walked} tgids walked, \
                 {jemalloc_detected} jemalloc detected, \
                 {probed_ok} probed OK, {failed} failed \
                 (dominant: {dominant})",
            );
        }
    } else {
        tracing::info!(
            "host-state probe: {tgids_walked} tgids walked, \
             {jemalloc_detected} jemalloc detected, \
             {probed_ok} probed OK, {failed} failed",
        );
    }
}

/// Capture a complete host-wide snapshot under arbitrary procfs
/// and cgroup roots. Walks `<proc_root>` for every live tgid,
/// enumerates its threads, and assembles a [`HostStateSnapshot`]
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
/// (d) `emit_probe_summary` plus the [`HostStateProbeSummary`]
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
    let tgids = iter_tgids_at(proc_root);
    let probe_cache: std::sync::Mutex<
        std::collections::HashMap<(u64, u64), Option<crate::host_thread_probe::JemallocProbe>>,
    > = std::sync::Mutex::new(std::collections::HashMap::new());
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
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(max_threads)
                .build()
                .unwrap();
            pool.install(|| tgids
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
                        let cache_key = std::fs::metadata(
                            proc_root.join(tgid.to_string()).join("exe"),
                        )
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
                                let mut s = summary_mutex
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                s.tgids_walked += 1;
                                if cached_result.is_some() {
                                    s.jemalloc_detected += 1;
                                    tracing::debug!(tgid, "host-state probe: cache hit (jemalloc)");
                                } else {
                                    tracing::debug!(tgid, "host-state probe: cache hit (not jemalloc or prior failure)");
                                }
                                cached_result
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
                                let mut s = summary_mutex
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                let res = record_attach_outcome(tgid, outcome, &mut s);
                                drop(s);
                                probe_cache
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .insert(key, res.clone());
                                res
                            }
                        } else {
                            // No cache key — exe symlink unreadable. Same
                            // attach-outside-lock pattern as the cache-miss
                            // branch above; result is not cached because
                            // there's no key to file it under.
                            let outcome = attach_probe_for_tgid_at(proc_root, tgid);
                            let mut s = summary_mutex
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            record_attach_outcome(tgid, outcome, &mut s)
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
                                    panic_payload
                                        .downcast_ref::<String>()
                                        .map(|s| s.as_str())
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
                            let mut s = summary_mutex
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            s.tgids_walked += 1;
                            *s.attach_tag_counts.entry("worker-panic").or_insert(0) += 1;
                            s.failed += 1;
                            tracing::error!(
                                tgid,
                                panic_msg,
                                "host-state probe: attach worker panicked; tgid skipped",
                            );
                            None
                        }
                    };
                    (tgid, probe)
                })
                .collect()
            )
        } else {
            std::collections::HashMap::new()
        };

    // `mut` is required because phase 2 below threads `&mut
    // summary` into `probe_thread_recording`.
    let mut summary = summary_mutex.into_inner().unwrap();
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
    HostStateSnapshot {
        captured_at_unix_ns,
        host,
        threads,
        cgroup_stats,
        probe_summary,
        parse_summary,
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
pub fn capture() -> HostStateSnapshot {
    capture_with(
        Path::new(DEFAULT_PROC_ROOT),
        Path::new(DEFAULT_CGROUP_ROOT),
        Path::new(DEFAULT_SYS_ROOT),
        true,
    )
}

/// Capture a host-state snapshot scoped to a single tgid.
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
pub fn capture_pid(pid: i32) -> HostStateSnapshot {
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
/// `emit_probe_summary` plus the [`HostStateProbeSummary`] on the
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
    HostStateSnapshot {
        captured_at_unix_ns,
        host,
        threads,
        cgroup_stats,
        probe_summary,
        parse_summary,
        psi,
        sched_ext,
    }
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
    use crate::metric_types::{
        Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32,
    };
    use tracing_test::traced_test;

    fn thread(pcomm: &str, comm: &str, run_time_ns: u64) -> ThreadState {
        ThreadState {
            tid: 1,
            tgid: 1,
            pcomm: pcomm.into(),
            comm: comm.into(),
            cgroup: "/".into(),
            start_time_clock_ticks: 0,
            policy: CategoricalString("SCHED_OTHER".into()),
            nice: OrdinalI32(0),
            cpu_affinity: CpuSet(vec![0, 1]),
            run_time_ns: MonotonicNs(run_time_ns),
            ..ThreadState::default()
        }
    }

    /// `ThreadState::default()` produces `'~'` (not `'\0'`) for
    /// the `state` char so the absent-value sentinel matches the
    /// capture-time `unwrap_or_else(default_state_char)`
    /// discipline. The bare `char` Default of `'\0'` (U+0000)
    /// lex-compares SMALLER than every real kernel state letter
    /// (`R`/`S`/`D`/`T`/`t`/`X`/`Z`/`P`/`I`); a Mode-tie-break
    /// that picks the lex-smallest would silently elect `'\0'`
    /// whenever a default-built thread sat alongside a real one
    /// in a group, dragging the cell from a meaningful state
    /// letter down to the absent sentinel. The manual
    /// [`Default`] impl on [`ThreadState`] pairs with the
    /// `serde(default = "default_state_char")` attribute on the
    /// field so both construction paths land on `'~'`.
    #[test]
    fn default_threadstate_state_is_sentinel_tilde() {
        let t = ThreadState::default();
        assert_eq!(
            t.state, '~',
            "ThreadState::default().state must be '~' (the \
             absent-value sentinel chosen to lex-sort AFTER \
             every real kernel state letter), not '\\0' (the \
             bare char Default); see field doc on \
             ThreadState::state"
        );
    }

    /// Mode tie-break regression: a default-constructed
    /// `ThreadState` must NOT lex-beat a real kernel state
    /// letter when both contribute to the same Mode aggregation
    /// at equal frequency. The kernel's
    /// [`crate::host_state_compare::aggregate`] closure
    /// `a.1.cmp(&b.1).then(b.0.cmp(&a.0))` selects
    /// LEX-SMALLEST on count-ties, so the sentinel must be
    /// LARGER than every real letter to keep the real letter
    /// winning. `'~'` (U+007E = 126) is larger than every
    /// kernel state letter (`R`=82, `S`=83, `D`=68, `T`=84,
    /// `t`=116, `X`=88, `Z`=90, `P`=80, `I`=73), so the
    /// tiebreak picks `R`. The original `'?'` (U+003F = 63)
    /// sentinel was SMALLER than every real letter, which
    /// would have made this test fail.
    #[test]
    fn mode_tiebreak_against_default_state_picks_real_letter() {
        use crate::host_state_compare::{AggRule, Aggregated, aggregate};
        let default_thread = ThreadState::default();
        let real_thread = ThreadState {
            state: 'R',
            ..ThreadState::default()
        };
        let agg = aggregate(
            AggRule::ModeChar(|t| t.state),
            &[&default_thread, &real_thread],
        );
        match agg {
            Aggregated::Mode { value, .. } => assert_eq!(
                value, "R",
                "Mode tiebreak between '~' (default sentinel) \
                 and 'R' (real kernel state) must elect 'R'; \
                 got {value:?}"
            ),
            other => panic!("expected Mode, got {other:?}"),
        }
    }

    /// Wire-format identity: hand-written JSON with raw
    /// primitive values at every newtype-wrapped field position
    /// must deserialize cleanly into a post-phase-2
    /// `ThreadState` with the wrapper fields holding the
    /// expected values. Covers one representative field per
    /// newtype family — MonotonicCount, MonotonicNs, ClockTicks,
    /// Bytes, PeakNs, GaugeNs, GaugeCount, OrdinalI32,
    /// OrdinalU32, CategoricalString, CpuSet — so a regression
    /// that breaks `serde(transparent)` on any wrapper would
    /// surface here without needing a real .hst.zst file from
    /// pre-phase-2 capture. Pre-phase-2 snapshot files (raw
    /// `u64`/`i32`/`String`/`Vec<u32>` at every position)
    /// continue to deserialize identically.
    #[test]
    fn wire_format_identity_raw_primitives_deserialize_into_wrapped_thread_state() {
        let json = r#"{
            "tid": 1234,
            "tgid": 1234,
            "pcomm": "demo",
            "comm": "demo-w",
            "cgroup": "/app",
            "start_time_clock_ticks": 555000,
            "policy": "SCHED_OTHER",
            "nice": -5,
            "cpu_affinity": [0, 1, 2, 3],
            "processor": 7,
            "state": "R",
            "ext_enabled": false,
            "run_time_ns": 1000000,
            "wait_time_ns": 0,
            "timeslices": 50,
            "voluntary_csw": 100,
            "nonvoluntary_csw": 25,
            "nr_wakeups": 200,
            "nr_wakeups_local": 80,
            "nr_wakeups_remote": 30,
            "nr_wakeups_sync": 10,
            "nr_wakeups_migrate": 5,
            "nr_wakeups_idle": 0,
            "nr_wakeups_affine": 60,
            "nr_wakeups_affine_attempts": 100,
            "nr_migrations": 8,
            "nr_migrations_cold": 0,
            "nr_forced_migrations": 1,
            "nr_failed_migrations_affine": 0,
            "nr_failed_migrations_running": 0,
            "nr_failed_migrations_hot": 0,
            "wait_sum": 5000000,
            "wait_count": 15,
            "wait_max": 250000,
            "sleep_sum": 3200000,
            "sleep_max": 180000,
            "block_sum": 1100000,
            "block_max": 60000,
            "iowait_sum": 77000,
            "iowait_count": 18,
            "exec_max": 90000,
            "slice_max": 400000,
            "allocated_bytes": 16777216,
            "deallocated_bytes": 8388608,
            "minflt": 7777,
            "majflt": 8888,
            "utime_clock_ticks": 10,
            "stime_clock_ticks": 11,
            "priority": 25,
            "rt_priority": 99,
            "delayacct_blkio_ticks": 137,
            "nr_wakeups_passive": 0,
            "core_forceidle_sum": 0,
            "fair_slice_ns": 250000,
            "nr_threads": 4,
            "smaps_rollup_kb": {},
            "rchar": 100,
            "wchar": 200,
            "syscr": 10,
            "syscw": 20,
            "read_bytes": 4096,
            "write_bytes": 8192,
            "cancelled_write_bytes": 1024
        }"#;
        let t: ThreadState = serde_json::from_str(json).expect("deserialize");
        // One representative field per newtype family proves
        // serde(transparent) works post-migration.
        assert_eq!(t.run_time_ns, crate::metric_types::MonotonicNs(1_000_000));
        assert_eq!(t.timeslices, crate::metric_types::MonotonicCount(50));
        assert_eq!(t.utime_clock_ticks, crate::metric_types::ClockTicks(10));
        assert_eq!(t.allocated_bytes, crate::metric_types::Bytes(16_777_216));
        assert_eq!(
            t.cancelled_write_bytes,
            crate::metric_types::Bytes(1024),
            "cancelled_write_bytes round-trips through the JSON \
             wire format alongside the other Bytes-typed fields",
        );
        assert_eq!(t.wait_max, crate::metric_types::PeakNs(250_000));
        assert_eq!(t.fair_slice_ns, crate::metric_types::GaugeNs(250_000));
        assert_eq!(t.nr_threads, crate::metric_types::GaugeCount(4));
        assert_eq!(t.nice, crate::metric_types::OrdinalI32(-5));
        assert_eq!(t.rt_priority, crate::metric_types::OrdinalU32(99));
        assert_eq!(
            t.policy,
            crate::metric_types::CategoricalString::from("SCHED_OTHER")
        );
        assert_eq!(
            t.cpu_affinity,
            crate::metric_types::CpuSet(vec![0, 1, 2, 3])
        );
    }

    /// Type-pin: nr_threads MUST be `GaugeCount`. A future
    /// refactor that flips it to a different newtype (e.g.
    /// `MonotonicCount`, which would silently re-enable Summable
    /// and let `--group-by comm`/`--group-by cgroup` over-count
    /// the parent process N-fold) would break this single
    /// `let _: GaugeCount = ...;` assignment. The test compiles
    /// only when the type is exactly `GaugeCount`.
    #[test]
    fn nr_threads_field_pinned_to_gauge_count() {
        let t = ThreadState::default();
        let _: crate::metric_types::GaugeCount = t.nr_threads;
    }

    /// Type-pin: cancelled_write_bytes MUST be `Bytes`. A future
    /// refactor that flipped it to a non-byte type (e.g. plain
    /// `MonotonicCount`, dropping the IEC-binary auto-scale
    /// ladder and the registry's `unit: "B"` rendering) would
    /// break this single `let _: Bytes = ...;` assignment. The
    /// test compiles only when the type is exactly `Bytes`.
    #[test]
    fn cancelled_write_bytes_field_pinned_to_bytes() {
        let t = ThreadState::default();
        let _: crate::metric_types::Bytes = t.cancelled_write_bytes;
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
            cgroup_stats: BTreeMap::from([("/".into(), {
                let mut cs = CgroupStats::default();
                cs.cpu.usage_usec = 500;
                cs.memory.current = 1 << 20;
                cs
            })]),
            probe_summary: None,
            parse_summary: None,
            psi: Psi::default(),
            sched_ext: None,
        };
        let tmp = tempfile::NamedTempFile::new().unwrap();
        snap.write(tmp.path()).unwrap();
        let back = HostStateSnapshot::load(tmp.path()).unwrap();
        assert_eq!(back.captured_at_unix_ns, 42);
        assert_eq!(back.threads.len(), 2);
        assert_eq!(
            back.threads[1].run_time_ns,
            crate::metric_types::MonotonicNs(2_000_000),
        );
        assert_eq!(back.cgroup_stats["/"].cpu.usage_usec, 500);
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

    /// Decompression-bomb guard: a zstd payload that decompresses
    /// past the configured cap surfaces an error tagged with
    /// "decompression-bomb guard" — the loader must not allocate
    /// past the ceiling. Test uses a small synthetic payload (8
    /// KiB of zeros, which compresses to a tiny blob but
    /// decompresses to 8192 bytes) against a 1024-byte cap so
    /// the test runs in microseconds rather than allocating a
    /// production-sized buffer.
    #[test]
    fn decompress_capped_rejects_decompression_bomb() {
        let payload = vec![0u8; 8192];
        let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
        let cap: u64 = 1024;
        let err = super::decompress_capped(&compressed, cap).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("decompression-bomb guard"),
            "expected decompression-bomb guard error, got: {msg}",
        );
    }

    /// Boundary case: a payload whose decompressed length is
    /// exactly `cap` bytes is accepted (the cap is inclusive).
    /// Pins the `>` (not `>=`) discriminator at the cap boundary
    /// so a future refactor that flips the comparison surfaces
    /// here rather than turning a legal snapshot into a
    /// false-positive bomb rejection.
    #[test]
    fn decompress_capped_accepts_payload_at_cap_boundary() {
        let payload = b"hello world".to_vec();
        let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
        let out = super::decompress_capped(&compressed, payload.len() as u64).unwrap();
        assert_eq!(
            out, payload,
            "payload exactly at the cap must round-trip — \
             cap is inclusive (`>` not `>=`)",
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
    fn parse_stat_extracts_all_known_fields() {
        // Fields 3..=41 — tail indices 0..=38. Token at tail[i] = i.
        // minflt at tail[7] = 7; majflt at tail[9] = 9;
        // utime at tail[11] = 11; stime at tail[12] = 12;
        // nice at tail[16] = 16; starttime at tail[19] = 19;
        // processor at tail[36] = 36; policy at tail[38] = 38.
        let mut line = String::from("1 (n) ");
        for i in 0..=38 {
            line.push_str(&format!("{i} "));
        }
        let f = parse_stat(&line);
        assert_eq!(f.minflt, Some(7));
        assert_eq!(f.majflt, Some(9));
        assert_eq!(f.utime_clock_ticks, Some(11));
        assert_eq!(f.stime_clock_ticks, Some(12));
        assert_eq!(f.nice, Some(16));
        assert_eq!(f.start_time_clock_ticks, Some(19));
        assert_eq!(f.processor, Some(36));
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
        assert_eq!(f.utime_clock_ticks, None);
        assert_eq!(f.stime_clock_ticks, None);
        assert_eq!(f.nice, None);
        assert_eq!(f.start_time_clock_ticks, None);
        assert_eq!(f.processor, None);
        assert_eq!(f.policy, None);
    }

    /// `processor` parses signed values via `get_i32`. The mainline
    /// kernel never emits a negative value (`task_cpu` is
    /// `unsigned int` per `include/linux/sched.h`, zero-extended
    /// through `seq_put_decimal_ll`), but the parser accepts
    /// negatives anyway — pinning that the type choice (`i32`)
    /// does not silently drop a hypothetical out-of-band negative
    /// to `None`. Defends against a regression that swapped
    /// `get_i32` for `get_u64` and made the field reject any
    /// negative token instead of carrying it through.
    #[test]
    fn parse_stat_processor_accepts_negative() {
        // 36 zero-pad tokens, tail[36] = -1, then more padding to
        // reach tail[38] for the policy field.
        let mut line = String::from("1 (n) ");
        for i in 0..36 {
            line.push_str(&format!("{i} "));
        }
        line.push_str("-1 ");
        line.push_str("0 ");
        line.push_str("0 ");
        let f = parse_stat(&line);
        assert_eq!(
            f.processor,
            Some(-1),
            "negative tokens must flow through as Some(-1) — pins \
             the get_i32 vs get_u64 type choice, not kernel emit \
             behavior (which never emits negative)",
        );
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
    fn parse_io_extracts_all_seven_fields() {
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
        assert_eq!(f.cancelled_write_bytes, Some(7));
    }

    #[test]
    fn parse_status_extracts_csw_and_affinity() {
        let raw = "Name:\tbash\n\
                   State:\tS (sleeping)\n\
                   Cpus_allowed_list:\t0-3,5\n\
                   voluntary_ctxt_switches:\t100\n\
                   nonvoluntary_ctxt_switches:\t5\n";
        let f = parse_status(raw);
        assert_eq!(f.voluntary_csw, Some(100));
        assert_eq!(f.nonvoluntary_csw, Some(5));
        assert_eq!(
            f.state,
            Some('S'),
            "first non-whitespace char of `State:` value is the \
             single-letter code (R/S/D/T/t/X/Z/P/I)",
        );
        assert_eq!(f.cpus_allowed.as_deref(), Some(&[0u32, 1, 2, 3, 5][..]));
    }

    /// Every kernel-emitted state code parses correctly. Pins each
    /// entry of `task_state_array` so a regression that lowercased
    /// the match or stripped paren-content would surface. Codes
    /// are from `fs/proc/array.c::task_state_array` — all NINE
    /// entries (R/S/D/T/t/X/Z/P/I), including the off-by-default
    /// `P (parked)` which only appears on kernels that schedule
    /// parked tasks.
    #[test]
    fn parse_status_accepts_every_kernel_state_code() {
        for code in ['R', 'S', 'D', 'T', 't', 'X', 'Z', 'P', 'I'] {
            let raw = format!("State:\t{code} (label)\n");
            assert_eq!(parse_status(&raw).state, Some(code));
        }
    }

    /// Absent `State:` line lands as `None`; capture site collapses
    /// to `'~'`. Pins the absent-default boundary.
    #[test]
    fn parse_status_absent_state_line_yields_none() {
        let raw = "voluntary_ctxt_switches:\t1\n";
        let f = parse_status(raw);
        assert_eq!(f.state, None);
    }

    /// PSI parser pins the kernel emission format
    /// `kernel/sched/psi.c:1284`. Two-line shape (some + full)
    /// is the cpu/memory/io case; cpu's avg/total decomposition
    /// hits both halves so a one-side regression surfaces here.
    #[test]
    fn parse_psi_extracts_some_and_full_halves() {
        let raw = "some avg10=18.59 avg60=24.31 avg300=20.49 total=78097519837\n\
                   full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n";
        let r = parse_psi(raw);
        // some: integer + 2-digit fraction → centi-percent.
        assert_eq!(r.some.avg10, 1859);
        assert_eq!(r.some.avg60, 2431);
        assert_eq!(r.some.avg300, 2049);
        assert_eq!(r.some.total_usec, 78_097_519_837);
        assert_eq!(r.full.avg10, 0);
        assert_eq!(r.full.avg60, 0);
        assert_eq!(r.full.avg300, 0);
        assert_eq!(r.full.total_usec, 0);
    }

    /// IRQ pressure is full-only per
    /// `kernel/sched/psi.c:1268` (`only_full = res == PSI_IRQ`),
    /// so the some-half stays at the absent-line default of zero.
    #[test]
    fn parse_psi_irq_full_only_leaves_some_at_zero() {
        let raw = "full avg10=1.09 avg60=1.08 avg300=1.46 total=80506377366\n";
        let r = parse_psi(raw);
        assert_eq!(r.full.avg10, 109);
        assert_eq!(r.full.avg60, 108);
        assert_eq!(r.full.avg300, 146);
        assert_eq!(r.full.total_usec, 80_506_377_366);
        // `some` half left at default zero — the kernel never
        // emitted a `some` line for irq.
        assert_eq!(r.some.avg10, 0);
        assert_eq!(r.some.avg60, 0);
        assert_eq!(r.some.avg300, 0);
        assert_eq!(r.some.total_usec, 0);
    }

    /// Empty / absent file collapses to all-zero bundle. Pins
    /// the absent-counter contract used elsewhere in this module.
    #[test]
    fn parse_psi_empty_input_yields_default() {
        let r = parse_psi("");
        assert_eq!(r.some.avg10, 0);
        assert_eq!(r.full.total_usec, 0);
    }

    /// Malformed numeric values default to zero rather than
    /// panicking. Mirrors the broader parser's
    /// `value.parse::<u64>().ok()` discipline — best-effort
    /// capture, never a hard error.
    #[test]
    fn parse_psi_malformed_value_defaults_to_zero() {
        let raw = "some avg10=NaN avg60=0.50 avg300=- total=abc\n";
        let r = parse_psi(raw);
        assert_eq!(r.some.avg10, 0, "NaN parses to zero");
        assert_eq!(r.some.avg60, 50, "well-formed neighbor still parses");
        assert_eq!(r.some.avg300, 0, "lone dash parses to zero");
        assert_eq!(r.some.total_usec, 0, "non-numeric total parses to zero");
    }

    /// Centi-percent conversion exhausts the fixed-point range:
    /// `100.00%` maps to 10_000. Pins the upper boundary
    /// against the u16 storage choice.
    #[test]
    fn parse_psi_full_saturation_maps_to_10000() {
        let raw = "some avg10=100.00 avg60=100.00 avg300=100.00 total=42\n";
        let r = parse_psi(raw);
        assert_eq!(r.some.avg10, 10_000);
        assert_eq!(r.some.avg60, 10_000);
        assert_eq!(r.some.avg300, 10_000);
        assert_eq!(r.some.total_usec, 42);
    }

    /// Unknown tokens are silently dropped (forward-compat with
    /// a future kernel that adds a 4th avg or new field).
    #[test]
    fn parse_psi_unknown_keys_ignored() {
        let raw = "some avg10=1.00 avg600=99.99 future_field=42 total=10\n";
        let r = parse_psi(raw);
        assert_eq!(r.some.avg10, 100);
        assert_eq!(r.some.total_usec, 10);
    }

    /// `parse_centi_percent` pads/truncates the fractional part
    /// to exactly 2 digits before combining. The kernel always
    /// emits `%02lu` per `kernel/sched/psi.c:1284`, but a robust
    /// parser must not silently rescale `"1.5"` (one digit) as
    /// `1*100+5 = 105` (1.05%) — that would corrupt the value.
    /// Mirrors `parsed_ns_from_dotted`'s zero-pad-to-six rule.
    #[test]
    fn parse_centi_percent_zero_pads_short_fraction() {
        // No fraction → 0.
        assert_eq!(parse_centi_percent("0"), 0);
        assert_eq!(parse_centi_percent("42"), 4200);
        // One-digit fraction → pad with trailing zero.
        assert_eq!(parse_centi_percent("1.5"), 150, "1.5 must read as 1.50%");
        assert_eq!(parse_centi_percent("0.7"), 70, "0.7 must read as 0.70%");
        // Two-digit fraction → kernel-canonical case.
        assert_eq!(parse_centi_percent("18.59"), 1859);
        // Three+ digit fraction → truncate to 2.
        assert_eq!(
            parse_centi_percent("1.501"),
            150,
            "1.501 truncates to 1.50%"
        );
        // Empty fraction (trailing dot) → 0.
        assert_eq!(parse_centi_percent("3."), 300);
        // EWMA-rounding ceiling per loadavg.h:35.
        assert_eq!(parse_centi_percent("100.99"), 10099);
    }

    /// Stage a synthetic `<proc_root>/pressure/{cpu,memory,io,irq}`
    /// tree and verify [`read_host_psi_at`] returns a fully
    /// populated [`Psi`] bundle. Pins the file naming and the
    /// per-resource bundling — a regression that swapped two
    /// resource sources (e.g. read `pressure/io` into `psi.cpu`)
    /// surfaces here as wrong-resource-wrong-value.
    #[test]
    fn read_host_psi_at_populates_all_four_resources() {
        let tmp = tempfile::TempDir::new().unwrap();
        let pressure = tmp.path().join("pressure");
        std::fs::create_dir_all(&pressure).unwrap();
        std::fs::write(
            pressure.join("cpu"),
            "some avg10=1.00 avg60=2.00 avg300=3.00 total=100\n\
             full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        )
        .unwrap();
        std::fs::write(
            pressure.join("memory"),
            "some avg10=4.50 avg60=5.50 avg300=6.50 total=200\n\
             full avg10=7.50 avg60=8.50 avg300=9.50 total=150\n",
        )
        .unwrap();
        std::fs::write(
            pressure.join("io"),
            "some avg10=10.10 avg60=20.20 avg300=30.30 total=300\n\
             full avg10=40.40 avg60=50.50 avg300=60.60 total=250\n",
        )
        .unwrap();
        std::fs::write(
            pressure.join("irq"),
            "full avg10=0.50 avg60=0.60 avg300=0.70 total=80\n",
        )
        .unwrap();

        let psi = read_host_psi_at(tmp.path());

        // cpu: both halves populated, full all-zero.
        assert_eq!(psi.cpu.some.avg10, 100);
        assert_eq!(psi.cpu.some.avg60, 200);
        assert_eq!(psi.cpu.some.avg300, 300);
        assert_eq!(psi.cpu.some.total_usec, 100);
        assert_eq!(psi.cpu.full.avg10, 0);
        assert_eq!(psi.cpu.full.total_usec, 0);

        // memory: both halves carry distinct nonzero values —
        // catches a regression that returned the same half
        // twice.
        assert_eq!(psi.memory.some.avg10, 450);
        assert_eq!(psi.memory.full.avg10, 750);
        assert_eq!(psi.memory.some.total_usec, 200);
        assert_eq!(psi.memory.full.total_usec, 150);

        // io: largest distinct values; ensures resource-source
        // routing isn't swapped against memory or cpu.
        assert_eq!(psi.io.some.avg10, 1010);
        assert_eq!(psi.io.full.avg300, 6060);
        assert_eq!(psi.io.some.total_usec, 300);

        // irq: full-only; some-half stays at the absent-line
        // default of zero.
        assert_eq!(psi.irq.full.avg10, 50);
        assert_eq!(psi.irq.full.avg60, 60);
        assert_eq!(psi.irq.full.avg300, 70);
        assert_eq!(psi.irq.full.total_usec, 80);
        assert_eq!(psi.irq.some.avg10, 0);
        assert_eq!(psi.irq.some.total_usec, 0);
    }

    /// Absent `pressure/` directory or absent per-resource files
    /// collapse to the all-zero default. Pins the absent-counter
    /// contract so a host with `CONFIG_PSI=n` (or older kernels
    /// missing `irq.pressure`) doesn't error out — capture is
    /// best-effort.
    #[test]
    fn read_host_psi_at_missing_files_yield_default() {
        // tempdir with no `pressure/` subdir at all.
        let tmp = tempfile::TempDir::new().unwrap();
        let psi = read_host_psi_at(tmp.path());
        assert_eq!(psi.cpu.some.avg10, 0);
        assert_eq!(psi.memory.full.total_usec, 0);
        assert_eq!(psi.io.some.avg300, 0);
        assert_eq!(psi.irq.full.avg60, 0);

        // Partial — only `cpu` exists; the other three should
        // still default cleanly.
        let pressure = tmp.path().join("pressure");
        std::fs::create_dir_all(&pressure).unwrap();
        std::fs::write(
            pressure.join("cpu"),
            "some avg10=12.34 avg60=0 avg300=0 total=0\n\
             full avg10=0 avg60=0 avg300=0 total=0\n",
        )
        .unwrap();
        let psi = read_host_psi_at(tmp.path());
        assert_eq!(psi.cpu.some.avg10, 1234);
        assert_eq!(psi.memory.some.avg10, 0);
        assert_eq!(psi.io.full.total_usec, 0);
        assert_eq!(psi.irq.full.avg10, 0);
    }

    /// Stage a synthetic cgroup tree and verify
    /// [`read_cgroup_psi_at`] reads `<cgroup>/<resource>.pressure`
    /// (cgroup v2 file naming, distinct from the host-level
    /// `pressure/<resource>` directory layout). Pins the
    /// path-strip-leading-slash behavior shared with
    /// [`read_cgroup_stats_at`].
    #[test]
    fn read_cgroup_psi_at_uses_resource_dot_pressure_naming() {
        let cgroup_root = tempfile::TempDir::new().unwrap();
        let cg_dir = cgroup_root.path().join("app");
        std::fs::create_dir_all(&cg_dir).unwrap();
        std::fs::write(
            cg_dir.join("cpu.pressure"),
            "some avg10=11.11 avg60=0 avg300=0 total=42\n\
             full avg10=0 avg60=0 avg300=0 total=0\n",
        )
        .unwrap();
        std::fs::write(
            cg_dir.join("memory.pressure"),
            "some avg10=0 avg60=0 avg300=0 total=0\n\
             full avg10=22.22 avg60=0 avg300=0 total=999\n",
        )
        .unwrap();
        // io.pressure absent → default zero. irq.pressure
        // present but full-only.
        std::fs::write(
            cg_dir.join("irq.pressure"),
            "full avg10=33.33 avg60=0 avg300=0 total=7\n",
        )
        .unwrap();

        let psi = read_cgroup_psi_at(cgroup_root.path(), "/app");

        assert_eq!(psi.cpu.some.avg10, 1111);
        assert_eq!(psi.cpu.some.total_usec, 42);
        assert_eq!(psi.memory.full.avg10, 2222);
        assert_eq!(psi.memory.full.total_usec, 999);
        assert_eq!(psi.io.some.avg10, 0, "absent io.pressure → default zero");
        assert_eq!(psi.io.full.total_usec, 0);
        assert_eq!(psi.irq.full.avg10, 3333);
        assert_eq!(psi.irq.some.avg10, 0, "irq is full-only");
    }

    /// `parse_kv_counters` reads cgroup v2 key-value files
    /// (memory.stat, memory.events). Pins:
    /// - well-formed multi-line input populates every key
    /// - malformed lines silently elide the offending key (rest
    ///   of the file still parses)
    /// - empty input yields an empty map
    /// - unknown key prefixes map verbatim (forward-compat with
    ///   future kernel additions to memory.stat).
    #[test]
    fn parse_kv_counters_handles_well_formed_and_malformed_lines() {
        let raw = "anon 12812288\n\
                   file 12623872\n\
                   pgfault 18\n\
                   pgmajfault 4\n\
                   workingset_refault_anon 0\n\
                   workingset_refault_file 27198\n";
        let m = parse_kv_counters(raw);
        assert_eq!(m.get("anon"), Some(&12_812_288));
        assert_eq!(m.get("file"), Some(&12_623_872));
        assert_eq!(m.get("pgfault"), Some(&18));
        assert_eq!(m.get("pgmajfault"), Some(&4));
        assert_eq!(m.get("workingset_refault_anon"), Some(&0));
        assert_eq!(m.get("workingset_refault_file"), Some(&27_198));
        assert_eq!(m.len(), 6);

        // Empty input → empty map.
        assert!(parse_kv_counters("").is_empty());

        // Malformed: missing value, non-u64 value, blank line —
        // each silently dropped; well-formed neighbors persist.
        let raw = "good 42\n\
                   bad_no_value\n\
                   bad_negative -5\n\
                   bad_text foo\n\
                   \n\
                   recover 7\n";
        let m = parse_kv_counters(raw);
        assert_eq!(m.get("good"), Some(&42));
        assert_eq!(m.get("recover"), Some(&7));
        assert_eq!(m.len(), 2, "malformed lines must not pollute the map");
    }

    /// `parse_smaps_rollup` reads cgroup-style `<key>: <u64> kB`
    /// lines and returns a `BTreeMap<String, u64>` of kB
    /// values. Pins:
    /// - well-formed multi-line input populates every key
    /// - the kernel's `<vma_range> [rollup]` header (no `:`)
    ///   is silently skipped
    /// - " kB" suffix is dropped via first-whitespace-token
    ///   extraction (parser doesn't hard-code the unit; a
    ///   future kernel that drops the suffix still parses)
    /// - empty input yields an empty map
    /// - lines whose value field doesn't parse as u64 are
    ///   silently dropped (best-effort, matches the
    ///   absent-counter contract).
    #[test]
    fn parse_smaps_rollup_extracts_kb_values_and_skips_header() {
        let raw = "55796dced000-7ffe1f875000 ---p 00000000 00:00 0                          [rollup]\n\
                   Rss:                2080 kB\n\
                   Pss:                 209 kB\n\
                   Pss_Dirty:           136 kB\n\
                   Pss_Anon:            136 kB\n\
                   Anonymous:           136 kB\n\
                   Swap:                  0 kB\n\
                   SwapPss:               0 kB\n\
                   Locked:                0 kB\n";
        let m = parse_smaps_rollup(raw);
        assert_eq!(m.get("Rss"), Some(&2080), "Rss kB stripped to integer");
        assert_eq!(m.get("Pss"), Some(&209));
        assert_eq!(m.get("Pss_Dirty"), Some(&136));
        assert_eq!(m.get("Pss_Anon"), Some(&136));
        assert_eq!(m.get("Anonymous"), Some(&136));
        assert_eq!(m.get("Swap"), Some(&0));
        assert_eq!(m.get("SwapPss"), Some(&0));
        assert_eq!(m.get("Locked"), Some(&0));
        assert_eq!(
            m.len(),
            8,
            "[rollup] header line is silently elided (no `:` separator)",
        );
    }

    /// Empty file → empty map. Pins the absent-counter contract
    /// for the "kernel pre-4.14 lacks smaps_rollup" path.
    #[test]
    fn parse_smaps_rollup_empty_input_yields_empty_map() {
        assert!(parse_smaps_rollup("").is_empty());
    }

    /// Malformed value fields (non-u64) are silently dropped;
    /// well-formed neighbors still parse. Pins the parser's
    /// best-effort discipline so a future kernel that emits a
    /// new key with an unexpected format doesn't break the
    /// whole capture.
    #[test]
    fn parse_smaps_rollup_malformed_value_silently_dropped() {
        let raw = "Rss:                100 kB\n\
                   BogusKey:        not_a_number kB\n\
                   Pss:                 50 kB\n";
        let m = parse_smaps_rollup(raw);
        assert_eq!(m.get("Rss"), Some(&100));
        assert_eq!(m.get("Pss"), Some(&50), "well-formed neighbor still parses");
        assert!(
            !m.contains_key("BogusKey"),
            "non-u64 value silently dropped"
        );
        assert_eq!(m.len(), 2);
    }

    /// The kernel's smaps_rollup header line carries a `:` in
    /// the device-major:minor pair (`<addr_start>-<addr_end>
    /// ---p <off> XX:XX <inode> [rollup]`). A naive
    /// `split_once(':')` would mis-extract the long
    /// whitespace-laden prefix as a "key" and parse the minor
    /// device integer as the "value", producing a junk
    /// 0-valued entry on every captured process. Pin the
    /// header guard so a regression that drops the
    /// whitespace-or-`-` rejection surfaces here.
    #[test]
    fn parse_smaps_rollup_skips_real_kernel_header_with_device_colon() {
        let raw = "55796dced000-7ffe1f875000 ---p 00000000 00:00 0                          [rollup]\n\
             Rss:                2080 kB\n\
             Pss:                 209 kB\n";
        let m = parse_smaps_rollup(raw);
        // Real keys parsed.
        assert_eq!(m.get("Rss"), Some(&2080));
        assert_eq!(m.get("Pss"), Some(&209));
        // No junk key from the header line — the pre-`:`
        // segment of the header carries whitespace AND `-`,
        // both rejected by the parser's header guard.
        assert_eq!(
            m.len(),
            2,
            "header line with `00:00` device pair must not produce a junk key; got {m:?}",
        );
    }

    /// Stage a synthetic `<sys_root>/kernel/sched_ext/` tree
    /// with all 5 global attrs and verify
    /// [`read_sched_ext_sysfs_at`] returns a fully populated
    /// [`SchedExtSysfs`]. Pins each file's parse routing.
    #[test]
    fn read_sched_ext_sysfs_at_populates_all_five_attrs() {
        let sys_root = tempfile::TempDir::new().unwrap();
        let scx_dir = sys_root.path().join("kernel").join("sched_ext");
        std::fs::create_dir_all(&scx_dir).unwrap();
        std::fs::write(scx_dir.join("state"), "enabled\n").unwrap();
        std::fs::write(scx_dir.join("switch_all"), "1\n").unwrap();
        std::fs::write(scx_dir.join("nr_rejected"), "42\n").unwrap();
        std::fs::write(scx_dir.join("hotplug_seq"), "315\n").unwrap();
        std::fs::write(scx_dir.join("enable_seq"), "7\n").unwrap();
        let scx = read_sched_ext_sysfs_at(sys_root.path())
            .expect("populated sched_ext directory must yield Some");
        assert_eq!(scx.state, "enabled");
        assert_eq!(scx.switch_all, 1);
        assert_eq!(scx.nr_rejected, 42);
        assert_eq!(scx.hotplug_seq, 315);
        assert_eq!(scx.enable_seq, 7);
    }

    /// Absent `<sys_root>/kernel/sched_ext/` directory yields
    /// `None`. Pins the CONFIG_SCHED_CLASS_EXT=n / no-sysfs
    /// path so a kernel without the feature collapses cleanly
    /// into the snapshot's `sched_ext: None`.
    #[test]
    fn read_sched_ext_sysfs_at_absent_directory_yields_none() {
        let sys_root = tempfile::TempDir::new().unwrap();
        // Empty tempdir — no kernel/sched_ext/ subtree.
        assert!(read_sched_ext_sysfs_at(sys_root.path()).is_none());
    }

    /// Per-file misses default to 0 / empty string. Pins the
    /// absent-counter contract for a half-populated sched_ext
    /// directory (older kernel that exposed only a subset of
    /// the 5 attrs).
    #[test]
    fn read_sched_ext_sysfs_at_partial_files_default_zero() {
        let sys_root = tempfile::TempDir::new().unwrap();
        let scx_dir = sys_root.path().join("kernel").join("sched_ext");
        std::fs::create_dir_all(&scx_dir).unwrap();
        // Only state + nr_rejected populated; the other 3 files
        // absent.
        std::fs::write(scx_dir.join("state"), "disabled\n").unwrap();
        std::fs::write(scx_dir.join("nr_rejected"), "100\n").unwrap();
        let scx =
            read_sched_ext_sysfs_at(sys_root.path()).expect("directory exists → returns Some");
        assert_eq!(scx.state, "disabled");
        assert_eq!(scx.nr_rejected, 100);
        assert_eq!(scx.switch_all, 0, "absent file → default 0");
        assert_eq!(scx.hotplug_seq, 0);
        assert_eq!(scx.enable_seq, 0);
    }

    /// Stage a synthetic procfs tree with a leader-thread
    /// (tid==tgid) carrying smaps_rollup, plus a follower
    /// thread (tid != tgid). Verifies:
    ///
    /// - leader thread's read populates the map.
    /// - follower thread's read returns an empty map without
    ///   touching the file (no IO cost on per-tid walks).
    ///
    /// This is the leader-dedup contract that makes per-MM
    /// data cheap to capture across thousands of threads.
    #[test]
    fn read_smaps_rollup_at_with_tally_dedups_to_leader_only() {
        let proc_root = tempfile::TempDir::new().unwrap();
        let tgid = 4242;
        let leader_tid = 4242;
        let follower_tid = 4243;

        // Stage `<tgid>/task/<leader_tid>/smaps_rollup`.
        let leader_dir = proc_root
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(leader_tid.to_string());
        std::fs::create_dir_all(&leader_dir).unwrap();
        std::fs::write(
            leader_dir.join("smaps_rollup"),
            "Rss:                2048 kB\n\
             Pss:                 512 kB\n",
        )
        .unwrap();

        // Stage `<tgid>/task/<follower_tid>/smaps_rollup` with a
        // POISON value — if the reader incorrectly opened it for
        // the follower it would read this and break the
        // assertion below.
        let follower_dir = proc_root
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(follower_tid.to_string());
        std::fs::create_dir_all(&follower_dir).unwrap();
        std::fs::write(
            follower_dir.join("smaps_rollup"),
            "Rss:                9999 kB\nPoison:           1 kB\n",
        )
        .unwrap();

        // Leader: file is read, map is populated.
        let m = read_smaps_rollup_at_with_tally(proc_root.path(), tgid, leader_tid, &mut None);
        assert_eq!(m.get("Rss"), Some(&2048));
        assert_eq!(m.get("Pss"), Some(&512));
        assert_eq!(m.len(), 2);

        // Follower: short-circuits to empty map BEFORE opening
        // the file. Catches a regression that flipped the
        // tid/tgid comparison or removed the dedup.
        let m = read_smaps_rollup_at_with_tally(proc_root.path(), tgid, follower_tid, &mut None);
        assert!(
            m.is_empty(),
            "follower thread must short-circuit to empty map; got {m:?}"
        );
    }

    /// Absent smaps_rollup file yields an empty map (older
    /// kernels pre-4.14 lack this file; CAP_SYS_PTRACE-denied
    /// reads under typical operator runs collapse the same way).
    /// Pins the read-failure path.
    #[test]
    fn read_smaps_rollup_at_with_tally_absent_file_yields_empty_map() {
        let proc_root = tempfile::TempDir::new().unwrap();
        let tgid = 4242;
        let leader_tid = 4242;
        let leader_dir = proc_root
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(leader_tid.to_string());
        std::fs::create_dir_all(&leader_dir).unwrap();
        // No smaps_rollup file written — capture must not error.
        let m = read_smaps_rollup_at_with_tally(proc_root.path(), tgid, leader_tid, &mut None);
        assert!(m.is_empty(), "absent file → empty map; got {m:?}");
    }

    /// `parse_max_or_u64` distinguishes the kernel's literal
    /// `max` token (no limit → `None`) from a concrete u64
    /// (a configured cap). Whitespace-only and malformed input
    /// collapses to `None` per the absent-counter contract.
    #[test]
    fn parse_max_or_u64_distinguishes_max_from_concrete_value() {
        assert_eq!(parse_max_or_u64("max"), None, "literal max → no limit");
        assert_eq!(
            parse_max_or_u64("max\n"),
            None,
            "trailing newline tolerated"
        );
        assert_eq!(
            parse_max_or_u64("9223372036854771712"),
            Some(9_223_372_036_854_771_712)
        );
        assert_eq!(parse_max_or_u64("0"), Some(0));
        assert_eq!(parse_max_or_u64(""), None, "empty input → no limit");
        assert_eq!(parse_max_or_u64("   "), None, "whitespace-only → no limit");
        assert_eq!(parse_max_or_u64("not_a_number"), None);
        // Negative values are not a kernel-emitted shape but the
        // parser tolerates them as malformed input → None.
        assert_eq!(parse_max_or_u64("-1"), None);
    }

    /// `parse_floor_value` is the FLOOR counterpart of
    /// [`parse_max_or_u64`]: literal "max" means "maximum
    /// protection" → `Some(u64::MAX)` (NOT `None`). `None` is
    /// reserved for absent-file / malformed input. The
    /// asymmetry vs. limits is the BLOCKER fix from #61's
    /// fix-round-1: `merge_min_option(Some(u64::MAX), Some(5G))`
    /// then yields 5G instead of None — preserving the lower
    /// concrete floor when one contributor has full protection.
    #[test]
    fn parse_floor_value_treats_max_as_full_protection() {
        assert_eq!(
            parse_floor_value("max"),
            Some(u64::MAX),
            "literal max → maximum protection (NOT no floor)"
        );
        assert_eq!(parse_floor_value("max\n"), Some(u64::MAX));
        assert_eq!(parse_floor_value("0"), Some(0), "zero → no protection");
        assert_eq!(parse_floor_value("1073741824"), Some(1_073_741_824));
        assert_eq!(parse_floor_value(""), None, "empty → absent file");
        assert_eq!(parse_floor_value("not_a_number"), None);
    }

    /// `parse_cpu_max` decodes the two-token `<quota|max> <period>`
    /// format. `period` falls back to the kernel default
    /// 100_000 µs when malformed.
    #[test]
    fn parse_cpu_max_handles_quota_period_pairs() {
        // Concrete cap.
        assert_eq!(parse_cpu_max("50000 100000"), (Some(50_000), 100_000));
        // No cap (`max` token); period preserved.
        assert_eq!(parse_cpu_max("max 100000"), (None, 100_000));
        // Different period (50ms).
        assert_eq!(parse_cpu_max("25000 50000"), (Some(25_000), 50_000));
        // Missing period — period defaults to kernel default.
        assert_eq!(parse_cpu_max("50000"), (Some(50_000), 100_000));
        // Empty input — both default.
        assert_eq!(parse_cpu_max(""), (None, 100_000));
        // Malformed period falls back to the default.
        assert_eq!(parse_cpu_max("50000 garbage"), (Some(50_000), 100_000));
        // Trailing newline tolerated by split_ascii_whitespace.
        assert_eq!(parse_cpu_max("max 100000\n"), (None, 100_000));
    }

    /// Stage a synthetic cgroup tree with every #61 file present
    /// and verify [`read_cgroup_stats_at`] populates the nested
    /// struct end-to-end. Pins file-naming, parse-routing, and
    /// the absent-vs-no-limit distinction.
    #[test]
    fn read_cgroup_stats_at_populates_nested_controllers_end_to_end() {
        let cgroup_root = tempfile::TempDir::new().unwrap();
        let cg_dir = cgroup_root.path().join("app");
        std::fs::create_dir_all(&cg_dir).unwrap();
        std::fs::write(
            cg_dir.join("cpu.stat"),
            "usage_usec 12345\nnr_throttled 7\nthrottled_usec 8\n",
        )
        .unwrap();
        std::fs::write(cg_dir.join("cpu.max"), "50000 100000\n").unwrap();
        std::fs::write(cg_dir.join("cpu.weight"), "200\n").unwrap();
        std::fs::write(cg_dir.join("cpu.weight.nice"), "-5\n").unwrap();
        std::fs::write(cg_dir.join("memory.current"), "9999\n").unwrap();
        std::fs::write(cg_dir.join("memory.max"), "max\n").unwrap();
        std::fs::write(cg_dir.join("memory.high"), "1073741824\n").unwrap();
        std::fs::write(cg_dir.join("memory.low"), "0\n").unwrap();
        std::fs::write(cg_dir.join("memory.min"), "0\n").unwrap();
        std::fs::write(
            cg_dir.join("memory.stat"),
            "anon 100\nfile 200\npgfault 18\nslab 50\n",
        )
        .unwrap();
        std::fs::write(
            cg_dir.join("memory.events"),
            "low 0\nhigh 1\nmax 0\noom 0\noom_kill 0\n",
        )
        .unwrap();
        std::fs::write(cg_dir.join("pids.current"), "42\n").unwrap();
        std::fs::write(cg_dir.join("pids.max"), "1024\n").unwrap();

        let stats = read_cgroup_stats_at(cgroup_root.path(), "/app");

        // CPU domain.
        assert_eq!(stats.cpu.usage_usec, 12_345);
        assert_eq!(stats.cpu.nr_throttled, 7);
        assert_eq!(stats.cpu.throttled_usec, 8);
        assert_eq!(stats.cpu.max_quota_us, Some(50_000));
        assert_eq!(stats.cpu.max_period_us, 100_000);
        assert_eq!(stats.cpu.weight, Some(200));
        assert_eq!(stats.cpu.weight_nice, Some(-5));

        // Memory domain.
        assert_eq!(stats.memory.current, 9999);
        assert_eq!(stats.memory.max, None, "literal max → no limit");
        assert_eq!(stats.memory.high, Some(1_073_741_824));
        assert_eq!(stats.memory.low, Some(0));
        assert_eq!(stats.memory.min, Some(0));
        assert_eq!(stats.memory.stat.get("anon"), Some(&100));
        assert_eq!(stats.memory.stat.get("file"), Some(&200));
        assert_eq!(stats.memory.stat.get("pgfault"), Some(&18));
        assert_eq!(stats.memory.stat.get("slab"), Some(&50));
        assert_eq!(stats.memory.events.get("oom_kill"), Some(&0));
        assert_eq!(stats.memory.events.get("high"), Some(&1));

        // PIDs domain.
        assert_eq!(stats.pids.current, Some(42));
        assert_eq!(stats.pids.max, Some(1024));
    }

    /// Root cgroup typically lacks every knob/limit file. Pins
    /// the absent-vs-no-limit distinction: `Option<u64>` limits
    /// stay `None` (file absent), counters stay 0 (Default),
    /// and `max_period_us` defaults to the kernel default
    /// rather than zero.
    #[test]
    fn read_cgroup_stats_at_root_cgroup_collapses_to_defaults() {
        let cgroup_root = tempfile::TempDir::new().unwrap();
        // No files at all under root — simulating a v2 mount
        // root that only carries `cgroup.*` files (no domain
        // controllers populated).
        let stats = read_cgroup_stats_at(cgroup_root.path(), "/");
        assert_eq!(stats.cpu.usage_usec, 0);
        assert_eq!(stats.cpu.max_quota_us, None);
        assert_eq!(
            stats.cpu.max_period_us, CPU_MAX_DEFAULT_PERIOD_US,
            "absent cpu.max → period defaults to kernel default"
        );
        assert_eq!(stats.cpu.weight, None);
        assert_eq!(stats.memory.current, 0);
        assert_eq!(stats.memory.max, None);
        assert_eq!(stats.memory.high, None);
        assert!(stats.memory.stat.is_empty());
        assert!(stats.memory.events.is_empty());
        assert_eq!(stats.pids.current, None);
        assert_eq!(stats.pids.max, None);
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
        let f = parse_sched(raw, &mut None);
        assert_eq!(f.nr_wakeups, Some(1000));
        assert_eq!(f.nr_migrations, Some(42));
        assert_eq!(f.nr_wakeups_local, Some(600));
        // PN_SCHEDSTAT format: ms.ns_remainder. `12345.678`
        // pads `.678` → `.678000` (= 678_000 ns), then
        // 12345 * 1_000_000 + 678_000 = 12_345_678_000 ns.
        assert_eq!(f.wait_sum, Some(12_345_678_000));
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
        let tgids = iter_tgids_at(Path::new(DEFAULT_PROC_ROOT));
        let pid = std::process::id() as i32;
        assert!(tgids.contains(&pid), "self pid {pid} not in /proc walk");
    }

    #[test]
    fn iter_task_ids_self_returns_at_least_main_tid() {
        let pid = std::process::id() as i32;
        let tids = iter_task_ids_at(Path::new(DEFAULT_PROC_ROOT), pid);
        assert!(
            tids.contains(&pid),
            "main tid {pid} absent from /proc/self/task"
        );
    }

    #[test]
    fn read_process_comm_for_self_is_populated() {
        let pid = std::process::id() as i32;
        let comm = read_process_comm_at(Path::new(DEFAULT_PROC_ROOT), pid)
            .expect("self comm must be readable");
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
        assert!(!t.policy.0.is_empty());
    }

    #[test]
    fn capture_produces_non_empty_snapshot() {
        // Scope to self_pid so the probe-attach pass is skipped (the
        // capture pipeline excludes the calling process from the
        // ptrace path because PTRACE_SEIZE rejects self-attach). The
        // global `capture()` would attempt to probe every jemalloc-
        // linked tgid on the host — orders of magnitude slower in a
        // unit-test context, and not what this test is asserting on.
        // The wiring-end-to-end test path lives in
        // `tests/host_state_capture_jemalloc_wiring.rs`, which spawns
        // a real jemalloc target.
        let pid = std::process::id() as i32;
        let snap = capture_pid(pid);
        assert!(!snap.threads.is_empty());
        let self_threads: Vec<_> = snap
            .threads
            .iter()
            .filter(|t| t.tgid == pid as u32)
            .collect();
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
        assert_eq!(
            policy_name(i32::MIN),
            format!("SCHED_UNKNOWN({})", i32::MIN)
        );
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
    /// in the ThreadState (which would then group with other
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
    /// filter and (b) pollute comparisons grouped by comm.
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
    fn stage_synthetic_proc(root: &Path, tgid: i32, tid: i32, pcomm: &str, comm: &str) {
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

        // status: State + voluntary/nonvoluntary csw + Cpus_allowed_list.
        // parse_status matches the lowercase csw keys verbatim;
        // `State` and `Cpus_allowed_list` use the capitalised
        // leading char of the procfs file.
        let status = "Name:\tfoo\n\
             State:\tR (running)\n\
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
             write_bytes: 8192\n\
             cancelled_write_bytes: 512\n";
        fs::write(task_dir.join("io"), io).unwrap();

        // sched: every parse_sched-matched key, with the
        // `se.statistics.` prefix for the wakeup family to
        // exercise the rsplit('.') short-key logic. `ext.enabled`
        // is unprefixed (literal kernel key) and tests the
        // full-key gate.
        let sched = "\
             se.statistics.nr_wakeups                       :         11\n\
             se.statistics.nr_wakeups_local                 :          8\n\
             se.statistics.nr_wakeups_remote                :          3\n\
             se.statistics.nr_wakeups_sync                  :          2\n\
             se.statistics.nr_wakeups_migrate               :          1\n\
             se.statistics.nr_wakeups_idle                  :          4\n\
             se.statistics.nr_wakeups_affine                :         12\n\
             se.statistics.nr_wakeups_affine_attempts       :         20\n\
             nr_migrations                                  :          9\n\
             se.statistics.nr_migrations_cold               :          5\n\
             se.statistics.nr_forced_migrations             :          7\n\
             se.statistics.nr_failed_migrations_affine      :          1\n\
             se.statistics.nr_failed_migrations_running     :          2\n\
             se.statistics.nr_failed_migrations_hot         :          3\n\
             wait_sum                                       :    5000.25\n\
             wait_count                                     :         15\n\
             se.statistics.wait_max                         :     250.5\n\
             sum_sleep_runtime                              :    3200.50\n\
             se.statistics.sleep_max                        :     180.25\n\
             sum_block_runtime                              :    1100.75\n\
             se.statistics.block_max                        :      60.75\n\
             iowait_sum                                     :       77.0\n\
             iowait_count                                   :         18\n\
             se.statistics.exec_max                         :      90.0\n\
             se.statistics.slice_max                        :     400.5\n\
             ext.enabled                                    :          1\n";
        fs::write(task_dir.join("sched"), sched).unwrap();

        // cgroup: v2-style single entry (0::path). read_cgroup_at
        // parses the `0::` prefix.
        fs::write(task_dir.join("cgroup"), "0::/ktstr.slice/worker0\n").unwrap();
    }

    /// Ghost-thread filter: a tid whose directory exists but
    /// carries ZERO readable procfs files (classic mid-capture
    /// exit — readdir races the reap) assembles an all-Default
    /// `ThreadState` and must NOT land in the snapshot. Stages
    /// one live thread with real content and one empty-directory
    /// ghost tid under the same tgid, calls `capture_with`, and
    /// asserts the output contains only the live thread.
    ///
    /// Without the filter, the ghost would land as `{ tid: 202,
    /// comm: "", cgroup: "", start_time_clock_ticks: 0, ...all
    /// counters zero }` and pollute downstream comparisons — a
    /// baseline run captures some number of ghosts, the candidate
    /// captures a different number, and the diff surfaces spurious
    /// "thread vanished" signal on every report.
    #[test]
    fn capture_with_filters_ghost_threads_with_empty_comm_and_zero_start() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 42;
        let live_tid: i32 = 101;
        let ghost_tid: i32 = 202;

        // Stage the live thread in full.
        stage_synthetic_proc(proc_tmp.path(), tgid, live_tid, "pcomm-proc", "live-thread");

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

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

        // Exactly one thread — the live one. The ghost is gone.
        assert_eq!(
            snap.threads.len(),
            1,
            "ghost tid with empty comm + zero start must be filtered; \
             got threads: {:?}",
            snap.threads
                .iter()
                .map(|t| (t.tid, &t.comm))
                .collect::<Vec<_>>(),
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
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 42;
        let tid: i32 = 101;

        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "pcomm-proc", "worker-thread");

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

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

        use crate::metric_types::{
            Bytes, CategoricalString, ClockTicks, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32,
            PeakNs,
        };

        // /proc/<tid>/stat fields parsed out of the paren-comm
        // tail: nice, utime, stime, starttime, processor, policy,
        // minflt, majflt.
        assert_eq!(t.nice, OrdinalI32(-10));
        assert_eq!(t.start_time_clock_ticks, 555_555);
        assert_eq!(t.policy, CategoricalString::from("SCHED_OTHER"));
        assert_eq!(t.minflt, MonotonicCount(7777));
        assert_eq!(t.majflt, MonotonicCount(8888));
        assert_eq!(
            t.utime_clock_ticks,
            ClockTicks(10),
            "tail[11] of stat fixture lands at utime_clock_ticks",
        );
        assert_eq!(
            t.stime_clock_ticks,
            ClockTicks(11),
            "tail[12] of stat fixture lands at stime_clock_ticks",
        );
        assert_eq!(
            t.processor,
            OrdinalI32(1700),
            "tail[36] of stat fixture (the 17th post-starttime \
             token, value 100*17=1700) lands at processor",
        );

        // schedstat — three-tuple of run/wait/slices.
        assert_eq!(t.run_time_ns, MonotonicNs(1_000_000));
        assert_eq!(t.wait_time_ns, MonotonicNs(200_000));
        assert_eq!(t.timeslices, MonotonicCount(50));

        // status — state + csw + Cpus_allowed_list. With
        // `use_syscall_affinity=false`, the capture path reads
        // cpu_affinity from status only.
        assert_eq!(
            t.state, 'R',
            "first non-whitespace char of `State:\tR (running)` is \
             the single-letter code R",
        );
        assert_eq!(t.voluntary_csw, MonotonicCount(42));
        assert_eq!(t.nonvoluntary_csw, MonotonicCount(7));
        assert_eq!(t.cpu_affinity, CpuSet(vec![0, 1, 2, 3]));

        // io — seven cumulative counters.
        assert_eq!(t.rchar, Bytes(100));
        assert_eq!(t.wchar, Bytes(200));
        assert_eq!(t.syscr, MonotonicCount(10));
        assert_eq!(t.syscw, MonotonicCount(20));
        assert_eq!(t.read_bytes, Bytes(4096));
        assert_eq!(t.write_bytes, Bytes(8192));
        assert_eq!(
            t.cancelled_write_bytes,
            Bytes(512),
            "cancelled_write_bytes round-trips from the 7th line of \
             /proc/<tid>/io",
        );

        // sched — every wakeup field, migrations (including the
        // five new failed/forced/cold/affine variants), the four
        // *_sum fractional-parse fields, the five *_max
        // fractional-parse fields, and the ext.enabled bool.
        assert_eq!(t.nr_wakeups, MonotonicCount(11));
        assert_eq!(t.nr_wakeups_local, MonotonicCount(8));
        assert_eq!(t.nr_wakeups_remote, MonotonicCount(3));
        assert_eq!(t.nr_wakeups_sync, MonotonicCount(2));
        assert_eq!(t.nr_wakeups_migrate, MonotonicCount(1));
        assert_eq!(t.nr_wakeups_idle, MonotonicCount(4));
        assert_eq!(t.nr_wakeups_affine, MonotonicCount(12));
        assert_eq!(
            t.nr_wakeups_affine_attempts,
            MonotonicCount(20),
            "denominator for the affine-wake success ratio \
             (nr_wakeups_affine / nr_wakeups_affine_attempts = 12/20)",
        );
        assert_eq!(t.nr_migrations, MonotonicCount(9));
        assert_eq!(t.nr_migrations_cold, MonotonicCount(5));
        assert_eq!(t.nr_forced_migrations, MonotonicCount(7));
        assert_eq!(t.nr_failed_migrations_affine, MonotonicCount(1));
        assert_eq!(t.nr_failed_migrations_running, MonotonicCount(2));
        assert_eq!(t.nr_failed_migrations_hot, MonotonicCount(3));
        // PN_SCHEDSTAT format is ms.ns_remainder. Reconstructed
        // ns = ms_part * 1_000_000 + zero-right-padded ns_part.
        // `5000.25` → `.25` pads to `.250000` (=250_000 ns) +
        // 5000ms × 1_000_000 = 5_000_250_000 ns total.
        assert_eq!(
            t.wait_sum,
            MonotonicNs(5_000_250_000),
            "PN_SCHEDSTAT 5000.25 reconstructs to 5_000_250_000 ns \
             (5000ms + 250_000ns)",
        );
        assert_eq!(t.wait_count, MonotonicCount(15));
        assert_eq!(
            t.wait_max,
            PeakNs(250_500_000),
            "PN_SCHEDSTAT 250.5 reconstructs to 250_500_000 ns",
        );
        assert_eq!(
            t.sleep_sum,
            MonotonicNs(3_200_500_000),
            "PN_SCHEDSTAT 3200.50 reconstructs to 3_200_500_000 ns; \
             sleep_sum is populated from the kernel's `sum_sleep_runtime` key",
        );
        assert_eq!(
            t.sleep_max,
            PeakNs(180_250_000),
            "PN_SCHEDSTAT 180.25 reconstructs to 180_250_000 ns",
        );
        assert_eq!(
            t.block_sum,
            MonotonicNs(1_100_750_000),
            "PN_SCHEDSTAT 1100.75 reconstructs to 1_100_750_000 ns; \
             block_sum is populated from the kernel's `sum_block_runtime` key",
        );
        assert_eq!(
            t.block_max,
            PeakNs(60_750_000),
            "PN_SCHEDSTAT 60.75 reconstructs to 60_750_000 ns",
        );
        assert_eq!(
            t.iowait_sum,
            MonotonicNs(77_000_000),
            "PN_SCHEDSTAT 77.0 reconstructs to 77_000_000 ns",
        );
        assert_eq!(t.iowait_count, MonotonicCount(18));
        assert_eq!(
            t.exec_max,
            PeakNs(90_000_000),
            "PN_SCHEDSTAT 90.0 reconstructs to 90_000_000 ns",
        );
        assert_eq!(
            t.slice_max,
            PeakNs(400_500_000),
            "PN_SCHEDSTAT 400.5 reconstructs to 400_500_000 ns",
        );
        assert!(
            t.ext_enabled,
            "ext.enabled = 1 round-trips through the full-key gate \
             to ThreadState::ext_enabled true",
        );

        // jemalloc TSD counters: synthetic procfs has no real ELF
        // behind /proc/<tgid>/exe, so the probe attach is gated off
        // (use_syscall_affinity=false). Both fields land at the
        // absent-counter default of 0. Pins this so a future
        // regression that always-probes (ignoring use_syscall_affinity)
        // would either crash on the synthetic /proc or surface garbage
        // here.
        assert_eq!(
            t.allocated_bytes,
            Bytes(0),
            "synthetic-tree capture must not probe — allocated_bytes \
             collapses to absent-counter zero",
        );
        assert_eq!(
            t.deallocated_bytes,
            Bytes(0),
            "synthetic-tree capture must not probe — deallocated_bytes \
             collapses to absent-counter zero",
        );
    }

    /// Capture against an empty `proc_root` (no tgid subdirs at
    /// all) must complete without panic and produce an empty
    /// snapshot. Pins the rayon parallel-probe phase's empty-input
    /// handling: `iter_tgids_at` returns an empty Vec, `par_iter`
    /// over zero elements collects to an empty HashMap, and the
    /// sequential phase 2 loop runs zero iterations. `use_syscall_affinity=true`
    /// is required to enter the rayon block at all (the `false`
    /// branch skips probe-attach entirely and assigns an empty
    /// HashMap directly). Without this gate test, the rayon
    /// par_iter over empty input has zero coverage.
    #[test]
    fn capture_with_empty_proc_root_produces_empty_snapshot() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();

        // Stage `/proc/loadavg` so the parallelism-clamp read at
        // <proc_root>/loadavg succeeds rather than falling back to
        // the 0.0 default. Empty `proc_root` otherwise — no tgid
        // subdirs, so `iter_tgids_at` returns Vec::new().
        std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
        assert!(
            snap.threads.is_empty(),
            "empty proc_root must produce empty snapshot; got {} threads",
            snap.threads.len(),
        );
    }

    /// Exercises the cache-lookup and insert code path in the
    /// rayon probe loop. Two tgids whose `/proc/<tgid>/exe`
    /// symlinks resolve to the same underlying inode trigger
    /// cache interaction: both attach calls fail with
    /// AttachError::MapsReadFailure (the synthetic tree has no
    /// `/proc/<tgid>/maps`), and the absent-counter contract
    /// holds — both threads land in the snapshot with
    /// allocated_bytes==0.
    #[test]
    fn capture_with_inode_cache_collapses_duplicate_binaries() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();

        // Required by the parallelism-clamp logic in capture_with.
        std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

        // One real file, two symlinks pointing at it. Both tgids'
        // exe metadata calls return the same (dev, ino) tuple, so
        // the cache_key matches across them.
        let shared_exe = proc_tmp.path().join("shared-exe");
        std::fs::write(&shared_exe, b"\x7fELFsynthetic\n").unwrap();

        for tgid in [4242, 4243] {
            stage_synthetic_proc(
                proc_tmp.path(),
                tgid,
                tgid + 1,
                "shared-pcomm",
                "shared-comm",
            );
            // `/proc/<tgid>/exe` symlink points at the shared file.
            // `attach_jemalloc_at` will read_link this successfully
            // and then fail on the absent `/proc/<tgid>/maps` →
            // AttachError::MapsReadFailure. The cache stores None
            // keyed by (dev, ino) of the shared file.
            let exe_link = proc_tmp.path().join(tgid.to_string()).join("exe");
            std::os::unix::fs::symlink(&shared_exe, &exe_link).unwrap();
        }

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);

        // Both threads still land in the snapshot — the failed
        // attach just leaves allocated_bytes at the absent-counter
        // default of zero. If the cache-hit branch panicked
        // (poisoned mutex, key collision logic, etc.), the rayon
        // worker would crash and `capture_with` would not return.
        assert_eq!(
            snap.threads.len(),
            2,
            "both staged threads must land in the snapshot",
        );
        for thread in &snap.threads {
            assert_eq!(
                thread.allocated_bytes,
                Bytes(0),
                "synthetic /proc has no maps; attach fails, allocated_bytes \
                 collapses to absent-counter zero — cache-hit branch must not \
                 fabricate a non-zero counter",
            );
        }
    }

    // ------------------------------------------------------------
    // Capture-pipeline error paths (Batch A + B)
    //
    // The synthetic-tree happy path is covered by
    // capture_with_synthetic_tree_assembles_thread_state above.
    // The tests below pin the pipeline's behavior against
    // adversarial inputs:
    // - missing/empty proc_root and tgid dirs (Batch A)
    // - non-numeric junk under proc_root (Batch A)
    // - capture_pid_with against pids that don't exist or are
    //   ghost (Batch A + B)
    // - selectively malformed/corrupted procfs files leaving
    //   the matching ThreadState fields zero-defaulted (Batch B)
    //
    // Each test uses stage_synthetic_proc to lay down a known-
    // good baseline, then mutates one specific axis. Assertions
    // include observed value, expected value, and likely root
    // cause so a regression points the reader at the failure
    // mode without re-derivation.
    // ------------------------------------------------------------

    /// G1 — proc_root pointing at a directory that does NOT
    /// exist must NOT panic. Pipeline collapses to an empty
    /// snapshot via `iter_tgids_at`'s read_dir-fail-→-empty-Vec
    /// guard. Defends against a future change that bubbled the
    /// I/O error to the caller.
    #[test]
    fn capture_with_nonexistent_proc_root_produces_empty_snapshot() {
        let scratch = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        // A path inside a fresh tempdir that we never create —
        // guaranteed to not exist within this test's scope.
        // io::read_dir returns ENOENT, iter_tgids_at returns
        // Vec::new(). Use false for use_syscall_affinity so the
        // parallel probe phase is fully skipped. Reuse the same
        // nonexistent path for sys_root: this test exercises the
        // ENOENT-collapses-cleanly invariant uniformly.
        let nonexistent = scratch.path().join("does-not-exist");
        let snap = capture_with(&nonexistent, cgroup_tmp.path(), &nonexistent, false);
        assert!(
            snap.threads.is_empty(),
            "nonexistent proc_root must produce empty snapshot; got \
             {} threads — iter_tgids_at must collapse ENOENT to empty",
            snap.threads.len(),
        );
    }

    /// G2 — tgid directory present but missing the inner
    /// `task/` subdirectory. `iter_task_ids_at` returns an
    /// empty vec, so the per-tid loop runs zero iterations and
    /// the tgid contributes no threads. Pins that the missing
    /// `task/` does not crash or fabricate a synthetic tid.
    #[test]
    fn capture_with_tgid_missing_task_dir_yields_no_threads_for_that_tgid() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();

        // tgid 4242: has `task/` and one tid (live thread).
        // tgid 4243: numeric directory but NO `task/` subdir.
        let live_tgid: i32 = 4242;
        let live_tid: i32 = 101;
        stage_synthetic_proc(
            proc_tmp.path(),
            live_tgid,
            live_tid,
            "live-pcomm",
            "live-comm",
        );

        let bare_tgid: i32 = 4243;
        std::fs::create_dir_all(proc_tmp.path().join(bare_tgid.to_string())).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

        assert_eq!(
            snap.threads.len(),
            1,
            "tgid 4243 has no `task/` subdir → contributes zero threads; \
             only live tgid 4242's tid should land. got {} threads, expected 1",
            snap.threads.len(),
        );
        assert_eq!(snap.threads[0].tgid, live_tgid as u32);
        assert_eq!(snap.threads[0].tid, live_tid as u32);
    }

    /// G3 — non-numeric directory entries under proc_root
    /// (real procfs has `self`, `thread-self`, `sys`, `kpageflags`,
    /// etc.) MUST be filtered by the parse-as-i32 step in
    /// `iter_tgids_at`. Pins the filter so a future refactor
    /// that loosened it (e.g. accepted any digit-prefix) does
    /// not surface kernel pseudo-files as fake tgids.
    #[test]
    fn capture_with_non_numeric_proc_entries_are_filtered() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();

        // Stage one valid numeric tgid plus several non-numeric
        // names that mimic real procfs entries.
        let live_tgid: i32 = 5151;
        let live_tid: i32 = 5152;
        stage_synthetic_proc(proc_tmp.path(), live_tgid, live_tid, "real", "real-thread");

        for junk in &["self", "thread-self", "sys", "version", "12abc", "abc"] {
            std::fs::create_dir_all(proc_tmp.path().join(junk)).unwrap();
        }
        // Negative or zero are filtered by `> 0` predicate.
        std::fs::create_dir_all(proc_tmp.path().join("0")).unwrap();
        std::fs::create_dir_all(proc_tmp.path().join("-1")).unwrap();

        // Direct check on the parse filter — pins iter_tgids_at
        // independently of the rest of the pipeline. Without this,
        // a loosened parse that accepted "12" from "12abc" would
        // still produce 1 thread downstream (the "12" dir has no
        // task/ subdir → contributes zero threads regardless), so
        // the snap.threads.len()==1 assertion alone wouldn't catch
        // the regression.
        assert_eq!(
            iter_tgids_at(proc_tmp.path()),
            vec![live_tgid],
            "iter_tgids_at must return only the real numeric tgid; \
             non-numeric and `0`/`-1` entries must be filtered by \
             parse::<i32>().ok() + `> 0` predicates",
        );

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

        assert_eq!(
            snap.threads.len(),
            1,
            "non-numeric proc_root entries (`self`, `12abc`, etc.) and \
             `0`/`-1` must be filtered by iter_tgids_at; got {} threads, \
             expected 1 (only the real tgid {live_tgid})",
            snap.threads.len(),
        );
        assert_eq!(snap.threads[0].tgid, live_tgid as u32);
    }

    /// G7 — `capture_pid_with` against a pid whose `/proc/<pid>`
    /// directory does not exist must NOT panic. `iter_task_ids_at`
    /// returns empty, the loop iterates zero times, and the
    /// snapshot's `threads` is empty. Pins that the per-pid
    /// capture path tolerates the same exit-mid-capture race the
    /// global path does.
    #[test]
    fn capture_pid_with_nonexistent_pid_produces_empty_snapshot() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        // pid 99999 is not staged — `proc_tmp/99999` does not exist.
        let snap = capture_pid_with(
            proc_tmp.path(),
            cgroup_tmp.path(),
            sys_tmp.path(),
            99999,
            false,
        );
        assert!(
            snap.threads.is_empty(),
            "capture_pid_with against nonexistent pid must produce empty \
             snapshot; got {} threads — iter_task_ids_at must collapse \
             ENOENT to empty",
            snap.threads.len(),
        );
    }

    /// G4a — corrupt the `stat` file so `parse_stat` returns
    /// all-None defaults (write a single non-paren token, so
    /// `rfind(')')` returns None and `parse_stat`
    /// short-circuits to `StatFields::default()`). With `comm`
    /// intact, the ghost-filter clause does NOT fire, so the
    /// thread lands with stat-derived fields at zero (nice,
    /// start_time, policy, processor, utime, stime) while
    /// comm + status + io still populate from their intact
    /// files.
    #[test]
    fn capture_with_corrupt_stat_file_zeroes_stat_fields_only() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 6161;
        let tid: i32 = 6162;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // Corrupt /proc/<tgid>/task/<tid>/stat — write a single
        // non-paren token so rfind(')') fails.
        let stat_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("stat");
        std::fs::write(&stat_path, "garbage no parens here\n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

        assert_eq!(
            snap.threads.len(),
            1,
            "corrupt stat does not block thread landing — comm + status \
             + io still populate; ghost filter only fires when comm AND \
             start_time are both empty/zero. got {} threads",
            snap.threads.len(),
        );
        let t = &snap.threads[0];
        // stat-derived fields collapse to zero/default.
        assert_eq!(
            t.start_time_clock_ticks, 0,
            "corrupt stat → start_time_clock_ticks default 0; got {}",
            t.start_time_clock_ticks
        );
        use crate::metric_types::{
            Bytes, CategoricalString, ClockTicks, MonotonicCount, OrdinalI32,
        };
        assert_eq!(
            t.nice,
            OrdinalI32(0),
            "corrupt stat → nice default 0; got {}",
            t.nice.0,
        );
        assert_eq!(
            t.policy,
            CategoricalString::from(""),
            "corrupt stat → policy default empty; got {:?}",
            t.policy
        );
        assert_eq!(t.utime_clock_ticks, ClockTicks(0));
        assert_eq!(t.stime_clock_ticks, ClockTicks(0));
        assert_eq!(t.processor, OrdinalI32(0));
        // status-derived fields still populate.
        assert_eq!(
            t.voluntary_csw,
            MonotonicCount(42),
            "status file is intact → voluntary_csw still populates"
        );
        // io-derived fields still populate.
        assert_eq!(
            t.rchar,
            Bytes(100),
            "io file is intact → rchar still populates"
        );
    }

    /// G4b — missing `schedstat` file (kernel without
    /// CONFIG_SCHEDSTATS) leaves run_time_ns / wait_time_ns /
    /// timeslices at zero. The thread still lands because
    /// stat/comm are intact.
    #[test]
    fn capture_with_missing_schedstat_zeroes_schedstat_fields() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7171;
        let tid: i32 = 7172;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // Remove /proc/<tgid>/task/<tid>/schedstat.
        let schedstat_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("schedstat");
        std::fs::remove_file(&schedstat_path).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(
            snap.threads.len(),
            1,
            "thread still lands with schedstat absent"
        );
        let t = &snap.threads[0];
        use crate::metric_types::{MonotonicCount, MonotonicNs};
        assert_eq!(
            t.run_time_ns,
            MonotonicNs(0),
            "missing schedstat → run_time_ns default 0; got {}",
            t.run_time_ns.0
        );
        assert_eq!(t.wait_time_ns, MonotonicNs(0));
        assert_eq!(t.timeslices, MonotonicCount(0));
        // start_time still populates from intact stat.
        assert_eq!(t.start_time_clock_ticks, 555_555);
    }

    /// G4c — malformed `status` file (random text, no recognized
    /// keys) leaves status-derived fields (voluntary_csw,
    /// nonvoluntary_csw, state, cpu_affinity) at default. With
    /// `use_syscall_affinity=false`, cpu_affinity comes from
    /// status only — so this also pins that absent
    /// Cpus_allowed_list defaults to empty Vec, NOT to the
    /// caller process's real affinity.
    #[test]
    fn capture_with_corrupt_status_zeroes_status_fields_and_empty_affinity() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 8181;
        let tid: i32 = 8182;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        let status_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("status");
        // No `:` separators → split_once(':') returns None for
        // every line → no field populates.
        std::fs::write(&status_path, "totally malformed garbage no colons here\n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let t = &snap.threads[0];
        use crate::metric_types::MonotonicCount;
        assert_eq!(
            t.voluntary_csw,
            MonotonicCount(0),
            "corrupt status → voluntary_csw default 0; got {}",
            t.voluntary_csw.0
        );
        assert_eq!(t.nonvoluntary_csw, MonotonicCount(0));
        assert_eq!(
            t.state, '~',
            "corrupt status → state collapses to '~' (capture-time \
             unwrap_or_else(default_state_char)); got {:?}",
            t.state
        );
        assert!(
            t.cpu_affinity.0.is_empty(),
            "use_syscall_affinity=false + corrupt status → cpu_affinity \
             must be empty Vec, NOT inherit caller's real affinity; got {:?}",
            t.cpu_affinity,
        );
    }

    /// G4d — missing `io` file (CONFIG_TASK_IO_ACCOUNTING off
    /// at kernel build) leaves all 6 byte counters at zero.
    /// Pins that the capture continues without io data rather
    /// than failing the whole snapshot.
    #[test]
    fn capture_with_missing_io_zeroes_io_fields() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 9191;
        let tid: i32 = 9192;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        let io_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("io");
        std::fs::remove_file(&io_path).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let t = &snap.threads[0];
        use crate::metric_types::{Bytes, MonotonicCount};
        assert_eq!(
            t.rchar,
            Bytes(0),
            "missing io → rchar default 0; got {}",
            t.rchar.0,
        );
        assert_eq!(t.wchar, Bytes(0));
        assert_eq!(t.syscr, MonotonicCount(0));
        assert_eq!(t.syscw, MonotonicCount(0));
        assert_eq!(t.read_bytes, Bytes(0));
        assert_eq!(t.write_bytes, Bytes(0));
        assert_eq!(t.cancelled_write_bytes, Bytes(0));
        // stat-derived fields still populate.
        assert_eq!(t.start_time_clock_ticks, 555_555);
    }

    /// G4e — missing `sched` file leaves every sched-derived
    /// field at zero (nr_wakeups family, *_sum, *_max,
    /// migrations, ext_enabled). The thread still lands because
    /// stat is intact.
    #[test]
    fn capture_with_missing_sched_zeroes_sched_fields() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 1010;
        let tid: i32 = 1011;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        let sched_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("sched");
        std::fs::remove_file(&sched_path).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let t = &snap.threads[0];
        use crate::metric_types::{MonotonicCount, MonotonicNs, PeakNs};
        assert_eq!(
            t.nr_wakeups,
            MonotonicCount(0),
            "missing sched → nr_wakeups default 0; got {}",
            t.nr_wakeups.0,
        );
        assert_eq!(t.nr_migrations, MonotonicCount(0));
        assert_eq!(t.wait_sum, MonotonicNs(0));
        assert_eq!(t.wait_max, PeakNs(0));
        assert_eq!(t.sleep_sum, MonotonicNs(0));
        assert_eq!(t.block_sum, MonotonicNs(0));
        assert_eq!(t.iowait_sum, MonotonicNs(0));
        assert_eq!(t.exec_max, PeakNs(0));
        assert_eq!(t.slice_max, PeakNs(0));
        assert!(
            !t.ext_enabled,
            "missing sched → ext.enabled key absent → ext_enabled false; \
             got {}",
            t.ext_enabled
        );
    }

    /// G5 — selectively delete EVERY non-comm file under one tid
    /// to simulate a partial mid-capture race (readdir saw the
    /// dir, then the kernel completed exit cleanup before our
    /// per-file reads). With comm intact, the thread still
    /// lands but every counter is zero. Pins the absent-=-zero
    /// contract under the worst plausible mid-capture race.
    #[test]
    fn capture_with_partial_mid_capture_race_lands_zero_thread() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 1212;
        let tid: i32 = 1213;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "racy-pcomm", "racy-comm");
        let task_dir = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string());
        // Remove every per-tid file EXCEPT comm. comm is the
        // ghost filter's anchor — keeping it preserves the
        // thread's identity so the test exercises the
        // counters-zero path rather than the ghost-drop path.
        for f in &["stat", "schedstat", "status", "io", "sched", "cgroup"] {
            std::fs::remove_file(task_dir.join(f)).unwrap();
        }

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1, "comm intact → thread still lands");
        let t = &snap.threads[0];
        use crate::metric_types::{Bytes, MonotonicCount, MonotonicNs};
        assert_eq!(t.comm, "racy-comm", "comm survives the racy partial reads");
        // Every counter zeros.
        assert_eq!(t.start_time_clock_ticks, 0);
        assert_eq!(t.nr_wakeups, MonotonicCount(0));
        assert_eq!(t.run_time_ns, MonotonicNs(0));
        assert_eq!(t.voluntary_csw, MonotonicCount(0));
        assert_eq!(t.rchar, Bytes(0));
        assert_eq!(t.minflt, MonotonicCount(0));
        assert_eq!(t.cgroup, "");
        assert!(
            snap.cgroup_stats.is_empty(),
            "all threads have empty cgroup → enrichment loop skips → \
             cgroup_stats stays empty",
        );
    }

    /// G6 — `capture_pid_with` ghost filter: a tid directory
    /// under the target pid exists but carries zero readable
    /// files (mid-capture exit). `capture_pid_with`'s
    /// terminal ghost-filter check — same shape as the global
    /// `capture_with` path's filter — must drop the
    /// all-Default ThreadState. Pins the per-pid path's filter
    /// independently of the global path.
    #[test]
    fn capture_pid_with_filters_ghost_threads() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 1313;
        let live_tid: i32 = 1314;
        let ghost_tid: i32 = 1315;

        stage_synthetic_proc(proc_tmp.path(), tgid, live_tid, "p", "live");

        // Ghost tid: directory exists but empty (no files).
        let ghost_dir = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(ghost_tid.to_string());
        std::fs::create_dir_all(&ghost_dir).unwrap();

        let snap = capture_pid_with(
            proc_tmp.path(),
            cgroup_tmp.path(),
            sys_tmp.path(),
            tgid,
            false,
        );

        assert_eq!(
            snap.threads.len(),
            1,
            "capture_pid_with must filter ghost tid {ghost_tid}; got {} \
             threads, expected 1 (only live tid {live_tid})",
            snap.threads.len(),
        );
        assert_eq!(snap.threads[0].tid, live_tid as u32);
    }

    /// G8 — malformed `Cpus_allowed_list:` value (a reversed
    /// range like `5-3`) routes through `parse_cpu_list` which
    /// returns `None`. With `use_syscall_affinity=false`, the
    /// capture site has no fallback and `cpu_affinity` stays
    /// at the default empty Vec. Pins that a malformed cpulist
    /// does NOT crash the parse and does NOT silently fabricate
    /// a partial range.
    #[test]
    fn capture_with_malformed_cpus_allowed_list_yields_empty_affinity() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 1414;
        let tid: i32 = 1415;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

        let status_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("status");
        // Reversed range — parse_cpu_list rejects (returns None).
        let status = "Name:\tfoo\n\
             State:\tR (running)\n\
             voluntary_ctxt_switches:\t1\n\
             nonvoluntary_ctxt_switches:\t1\n\
             Cpus_allowed_list:\t5-3\n";
        std::fs::write(&status_path, status).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let t = &snap.threads[0];
        use crate::metric_types::MonotonicCount;
        assert!(
            t.cpu_affinity.0.is_empty(),
            "malformed Cpus_allowed_list `5-3` → parse_cpu_list returns \
             None → cpu_affinity defaults to empty Vec (NOT a partial \
             range, NOT the caller's affinity); got {:?}",
            t.cpu_affinity,
        );
        // Other status fields still populate (the malformed
        // line failed only the cpulist arm of parse_status).
        assert_eq!(
            t.voluntary_csw,
            MonotonicCount(1),
            "malformed cpulist must NOT corrupt csw fields on the same \
             status file — per-arm Option isolation"
        );
    }

    /// G11 — huge `Cpus_allowed_list:` range (above the
    /// MAX_CPU_RANGE_EXPANSION cap at 64 Ki CPUs) routes
    /// through the `parse_cpu_list` cap and returns `None`.
    /// Same observable effect as G8 (empty Vec) but pins a
    /// distinct adversarial input — a hostile /proc with a
    /// `0-4294967295` cpulist must NOT allocate gigabytes.
    #[test]
    fn capture_with_huge_cpu_range_in_status_yields_empty_affinity() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 1515;
        let tid: i32 = 1516;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

        let status_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("status");
        // u32::MAX-spanning range — well above the 64 Ki cap;
        // parse_cpu_list rejects without expansion.
        let status = "Cpus_allowed_list:\t0-4294967295\n\
             voluntary_ctxt_switches:\t1\n\
             nonvoluntary_ctxt_switches:\t1\n";
        std::fs::write(&status_path, status).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let t = &snap.threads[0];
        use crate::metric_types::MonotonicCount;
        assert!(
            t.cpu_affinity.0.is_empty(),
            "huge cpulist range `0-4294967295` exceeds the 64 Ki \
             expansion cap → parse_cpu_list returns None → cpu_affinity \
             empty (NOT a 4-billion-element Vec, NOT a partial range); \
             got {} elements",
            t.cpu_affinity.0.len(),
        );
        // Per-arm isolation: the cap-rejected cpulist must NOT
        // crash the rest of parse_status. csw fields on the same
        // file still populate. Mirrors G8's isolation check.
        assert_eq!(
            t.voluntary_csw,
            MonotonicCount(1),
            "huge cpulist rejection must not break csw parsing on the \
             same status file — per-arm Option isolation"
        );
    }

    /// G9 — non-numeric directory entries under `<proc_root>/<tgid>/task/`
    /// MUST be filtered by the parse-as-i32 step in
    /// `iter_task_ids_at`. Mirrors G3 for the per-tgid `task/` subdir
    /// (G3 covers `<proc_root>` itself). Real procfs has only numeric
    /// task entries, but a hostile or malformed test fixture could
    /// stage non-numeric names; the filter must drop them rather
    /// than surface garbage tids.
    #[test]
    fn capture_with_non_numeric_task_entries_are_filtered() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();

        let live_tgid: i32 = 8181;
        let live_tid: i32 = 8182;
        stage_synthetic_proc(proc_tmp.path(), live_tgid, live_tid, "real", "real-thread");

        // Stage non-numeric entries alongside the real tid under
        // <tgid>/task/. iter_task_ids_at must filter on parse::<i32>().
        let task_dir = proc_tmp.path().join(live_tgid.to_string()).join("task");
        for junk in &["status", "self", "12abc", "abc"] {
            std::fs::create_dir_all(task_dir.join(junk)).unwrap();
        }
        std::fs::create_dir_all(task_dir.join("0")).unwrap();
        std::fs::create_dir_all(task_dir.join("-1")).unwrap();

        // Direct check on the parse filter — pins iter_task_ids_at
        // independently of the rest of the pipeline.
        assert_eq!(
            iter_task_ids_at(proc_tmp.path(), live_tgid),
            vec![live_tid],
            "iter_task_ids_at must return only the real numeric tid; \
             non-numeric and `0`/`-1` entries must be filtered by \
             parse::<i32>().ok() + `> 0` predicates",
        );

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(
            snap.threads.len(),
            1,
            "non-numeric `task/` entries must be filtered by \
             iter_task_ids_at; got {} threads, expected 1",
            snap.threads.len(),
        );
        assert_eq!(snap.threads[0].tid, live_tid as u32);
    }

    /// G10 — a tgid emitting a v1-only `cgroup` file (legacy
    /// hierarchy entries, no `0::` unified line) lands the thread
    /// with `cgroup` defaulting to "". The ghost filter does NOT
    /// fire because comm + start_time are intact. The empty cgroup
    /// is a legitimate observable signal — `capture_with`'s
    /// cgroup_stats enrichment loop skips entries with empty
    /// `cgroup` so no synthetic stats land for the missing path.
    #[test]
    fn capture_with_v1_only_cgroup_yields_empty_cgroup_string() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 9191;
        let tid: i32 = 9192;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

        // Overwrite the cgroup file with only legacy v1 lines —
        // parse_cgroup_v2 returns None, read_cgroup_at returns
        // None, ThreadState.cgroup defaults to "".
        let cgroup_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("cgroup");
        let v1_only = "12:cpuset:/legacy/cpuset/path\n\
             5:freezer:/legacy/freezer\n\
             3:blkio:/\n";
        std::fs::write(&cgroup_path, v1_only).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

        assert_eq!(
            snap.threads.len(),
            1,
            "v1-only cgroup does not block thread landing — comm + \
             start_time are intact, ghost filter does not fire; \
             got {} threads",
            snap.threads.len(),
        );
        let t = &snap.threads[0];
        assert_eq!(
            t.cgroup, "",
            "v1-only cgroup file → parse_cgroup_v2 returns None → \
             ThreadState.cgroup defaults to empty; got {:?}",
            t.cgroup,
        );
        // cgroup_stats enrichment skips empty-cgroup threads. The
        // map must not carry an entry keyed on "" (would otherwise
        // accumulate a meaningless aggregate row in the snapshot).
        assert!(
            !snap.cgroup_stats.contains_key(""),
            "empty-cgroup thread must NOT seed an empty-key entry in \
             cgroup_stats — the enrichment loop's `!is_empty()` guard \
             pins the skip; got keys: {:?}",
            snap.cgroup_stats.keys().collect::<Vec<_>>(),
        );
    }

    /// `capture_to` propagates write errors through anyhow with the
    /// destination path in the context chain so an operator who
    /// passed an unwritable target sees the path in the diagnostic
    /// rather than a bare I/O error. Pins the `with_context` wrapper
    /// at the public-API boundary; without it, the error message
    /// loses the path and operators can't tell which target failed.
    #[test]
    fn capture_to_returns_err_on_unwritable_path() {
        // A path under a directory that does not exist — std::fs::write
        // returns ENOENT for the parent; capture_to's with_context
        // wraps it with the destination path.
        let scratch = tempfile::TempDir::new().unwrap();
        let unwritable = scratch.path().join("missing-dir").join("snap.hst.zst");
        let err = capture_to(&unwritable).unwrap_err();
        let chain = format!("{err:#}");
        assert!(
            chain.contains(unwritable.to_string_lossy().as_ref()),
            "error chain must name the unwritable target path; got: {chain}",
        );
    }

    /// `read_cgroup_stats_at` reads from the path string verbatim;
    /// when the path names a cgroup directory that does not exist
    /// (the thread's cgroup string was captured but the cgroup has
    /// since been rmdir'd, or the cgroup_root differs from the live
    /// host), every cpu.stat / memory.current read fails with
    /// ENOENT and the resulting `CgroupStats` is all-zero. Pins the
    /// "absent = 0" contract for the enrichment loop's stale-string
    /// race.
    #[test]
    fn capture_with_stale_cgroup_path_yields_all_zero_stats() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7373;
        let tid: i32 = 7374;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // stage_synthetic_proc writes "0::/ktstr.slice/worker0" into
        // the cgroup file but does NOT create the matching directory
        // under cgroup_root. The enrichment loop calls
        // read_cgroup_stats_at("/ktstr.slice/worker0"), which
        // resolves to a non-existent dir and returns all-zero stats.

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let stats = snap
            .cgroup_stats
            .get("/ktstr.slice/worker0")
            .expect("non-empty cgroup string must seed the stats map");
        assert_eq!(stats.cpu.usage_usec, 0, "stale cgroup → cpu_usage_usec 0");
        assert_eq!(stats.cpu.nr_throttled, 0, "stale cgroup → nr_throttled 0");
        assert_eq!(
            stats.cpu.throttled_usec, 0,
            "stale cgroup → throttled_usec 0"
        );
        assert_eq!(stats.memory.current, 0, "stale cgroup → memory_current 0");
    }

    /// `read_cgroup_at` returns `None` when the cgroup file is
    /// present but contains only v1 hierarchy lines (no `0::`
    /// unified prefix). Pins the "v1-only → None" path of
    /// `parse_cgroup_v2` from the file-read entry point — distinct
    /// from `parse_cgroup_v2_none_when_only_legacy_present` which
    /// pins the parse function in isolation.
    #[test]
    fn read_cgroup_at_v1_only_cgroup_returns_none() {
        let tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 4242;
        let tid: i32 = 4243;
        let task_dir = tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string());
        std::fs::create_dir_all(&task_dir).unwrap();
        let v1_only = "12:cpuset:/legacy/cpuset/path\n\
             5:freezer:/legacy/freezer\n";
        std::fs::write(task_dir.join("cgroup"), v1_only).unwrap();

        assert_eq!(
            read_cgroup_at(tmp.path(), tgid, tid),
            None,
            "v1-only cgroup file → read_cgroup_at returns None (no 0:: line)",
        );

        // Symmetric missing-file branch: no cgroup file → None.
        assert_eq!(
            read_cgroup_at(tmp.path(), tgid, 9999),
            None,
            "missing cgroup file → read_cgroup_at returns None",
        );
    }

    /// `parse_cgroup_v2` accepts the degenerate "/" root path. A
    /// process cgrouped at the unified root emits "0::/" and the
    /// parser returns Some("/"). Pins the boundary distinct from
    /// `parse_cgroup_v2_empty_path_and_multiple_unified_lines`
    /// (which covers "0::" with empty-string-after-prefix); this
    /// test pins that "/" alone is treated as a valid path, not
    /// folded into the empty-string rejection.
    #[test]
    fn parse_cgroup_v2_root_only_path_returns_slash() {
        // Single "0::/" line — the trim + non-empty guard accepts
        // "/" as a valid path.
        assert_eq!(parse_cgroup_v2("0::/\n"), Some("/".to_string()));
        // Same with trailing whitespace — trim absorbs it but "/"
        // survives as the post-trim value.
        assert_eq!(parse_cgroup_v2("0::/  \n"), Some("/".to_string()));
        // Mixed alongside legacy v1 lines — unified picks "/".
        let raw = "12:cpuset:/legacy/path\n0::/\n5:freezer:/legacy\n";
        assert_eq!(parse_cgroup_v2(raw), Some("/".to_string()));
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
        assert_eq!(stats.cpu.usage_usec, 12345);
        assert_eq!(stats.cpu.nr_throttled, 7);
        assert_eq!(stats.cpu.throttled_usec, 8);
        assert_eq!(stats.memory.current, 9999);
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
        assert_eq!(stats.cpu.usage_usec, 500);
        assert_eq!(stats.cpu.nr_throttled, 0);
        assert_eq!(stats.cpu.throttled_usec, 0);
        assert_eq!(
            stats.memory.current, 0,
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
        assert_eq!(stats.cpu.usage_usec, 0);
        assert_eq!(stats.cpu.nr_throttled, 0);
        assert_eq!(stats.cpu.throttled_usec, 0);
        assert_eq!(stats.memory.current, 2048);
    }

    /// Case (d): neither file present → every field zero.
    /// Distinct from "returns None or errors" — the documented
    /// contract is absent = 0.
    #[test]
    fn read_cgroup_stats_at_both_files_missing_all_zero() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("empty-cg")).unwrap();
        let stats = read_cgroup_stats_at(tmp.path(), "/empty-cg");
        assert_eq!(stats.cpu.usage_usec, 0);
        assert_eq!(stats.cpu.nr_throttled, 0);
        assert_eq!(stats.cpu.throttled_usec, 0);
        assert_eq!(stats.memory.current, 0);
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
        assert_eq!(stats.cpu.usage_usec, 999);
        assert_eq!(stats.cpu.nr_throttled, 0, "absent key collapses to 0");
        assert_eq!(stats.cpu.throttled_usec, 111);
    }

    // ------------------------------------------------------------
    // H4 — parse_sched every-field coverage + parse fallbacks
    // ------------------------------------------------------------

    /// Populated `/proc/<tid>/sched` with every one of the 13
    /// fields parse_sched recognises. Ordering mixed (sync before
    /// local, migrate before idle) so the test doesn't pin a
    /// single-pass scan order that the helper doesn't actually
    /// promise. Integer-only PN_SCHEDSTAT values (no fractional
    /// part) parse via the no-dot branch of `parsed_ns_from_dotted`
    /// — interpreted as plain ns counts — so the values pass
    /// through unchanged.
    #[test]
    fn parse_sched_populates_all_known_fields() {
        let raw = "\
             se.statistics.nr_wakeups                       :         11\n\
             se.statistics.nr_wakeups_sync                  :          2\n\
             se.statistics.nr_wakeups_local                 :          8\n\
             se.statistics.nr_wakeups_migrate               :          1\n\
             se.statistics.nr_wakeups_remote                :          3\n\
             se.statistics.nr_wakeups_idle                  :          4\n\
             se.statistics.nr_wakeups_affine                :         12\n\
             se.statistics.nr_wakeups_affine_attempts       :         20\n\
             nr_migrations                                  :          9\n\
             se.statistics.nr_migrations_cold               :          5\n\
             se.statistics.nr_forced_migrations             :          7\n\
             se.statistics.nr_failed_migrations_affine      :          1\n\
             se.statistics.nr_failed_migrations_running     :          2\n\
             se.statistics.nr_failed_migrations_hot         :          3\n\
             wait_sum                                       :       500\n\
             wait_count                                     :         15\n\
             se.statistics.wait_max                         :       250\n\
             sum_sleep_runtime                              :       320\n\
             se.statistics.sleep_max                        :       180\n\
             sum_block_runtime                              :       110\n\
             se.statistics.block_max                        :        60\n\
             iowait_sum                                     :         77\n\
             iowait_count                                   :         18\n\
             se.statistics.exec_max                         :        90\n\
             se.statistics.slice_max                        :       400\n\
             ext.enabled                                    :          1\n";
        let s = parse_sched(raw, &mut None);
        assert_eq!(s.nr_wakeups, Some(11));
        assert_eq!(s.nr_wakeups_local, Some(8));
        assert_eq!(s.nr_wakeups_remote, Some(3));
        assert_eq!(s.nr_wakeups_sync, Some(2));
        assert_eq!(s.nr_wakeups_migrate, Some(1));
        assert_eq!(s.nr_wakeups_idle, Some(4));
        assert_eq!(s.nr_wakeups_affine, Some(12));
        assert_eq!(s.nr_wakeups_affine_attempts, Some(20));
        assert_eq!(s.nr_migrations, Some(9));
        assert_eq!(s.nr_migrations_cold, Some(5));
        assert_eq!(s.nr_forced_migrations, Some(7));
        assert_eq!(s.nr_failed_migrations_affine, Some(1));
        assert_eq!(s.nr_failed_migrations_running, Some(2));
        assert_eq!(s.nr_failed_migrations_hot, Some(3));
        assert_eq!(s.wait_sum, Some(500));
        assert_eq!(s.wait_count, Some(15));
        assert_eq!(s.wait_max, Some(250));
        assert_eq!(
            s.sleep_sum,
            Some(320),
            "sleep_sum reads the kernel's `sum_sleep_runtime` key",
        );
        assert_eq!(s.sleep_max, Some(180));
        assert_eq!(
            s.block_sum,
            Some(110),
            "block_sum reads the kernel's `sum_block_runtime` key",
        );
        assert_eq!(s.block_max, Some(60));
        assert_eq!(s.iowait_sum, Some(77));
        assert_eq!(s.iowait_count, Some(18));
        assert_eq!(s.exec_max, Some(90));
        assert_eq!(s.slice_max, Some(400));
        assert_eq!(
            s.ext_enabled,
            Some(true),
            "ext.enabled = 1 → Some(true) — full-key match required \
             because rsplit('.') would yield `enabled` and collide \
             with any future field of that name",
        );
    }

    /// `ext.enabled = 0` lands as `Some(false)` (CONFIG_SCHED_CLASS_EXT
    /// kernel where the task is NOT on sched_ext); absent line lands
    /// as `None` and the capture-site `unwrap_or(false)` collapses to
    /// the absent default. Pins the bool round-trip.
    #[test]
    fn parse_sched_ext_enabled_zero_and_absent() {
        let zero = parse_sched("ext.enabled : 0\n", &mut None);
        assert_eq!(zero.ext_enabled, Some(false));
        let absent = parse_sched("nr_wakeups : 1\n", &mut None);
        assert_eq!(absent.ext_enabled, None);
    }

    /// Full-key match on `ext.enabled` MUST take precedence over the
    /// rsplit-on-dot fallback. A line like `foo.enabled : 1` would
    /// otherwise route through rsplit to `enabled`, collide with
    /// `ext.enabled`, and incorrectly populate the bool. Pins the
    /// guard.
    #[test]
    fn parse_sched_ext_enabled_no_collision_via_rsplit() {
        // foo.enabled is not a real kernel key, but proves the
        // full-key gate: rsplit yields `enabled`, but the match
        // arm only fires on the exact key `ext.enabled`.
        let s = parse_sched("foo.enabled : 1\n", &mut None);
        assert_eq!(s.ext_enabled, None);
    }

    /// Dotted PN_SCHEDSTAT fractional values reconstruct full ns
    /// via `ms * 1_000_000 + zero-right-padded ns_remainder`.
    /// Pins the helper for varying fractional widths (1, 2, and
    /// 3 digits past the dot — all zero-pad to 6).
    #[test]
    fn parse_sched_fractional_fields_reconstruct_ns() {
        let raw = "\
             wait_sum                                       :    1234.5\n\
             sum_sleep_runtime                              :     678.9\n\
             sum_block_runtime                              :      42.1\n\
             iowait_sum                                     :       7.999\n";
        let s = parse_sched(raw, &mut None);
        // 1234.5 → .5 pads to .500000 (=500_000) + 1234ms = 1_234_500_000 ns
        assert_eq!(s.wait_sum, Some(1_234_500_000));
        // 678.9 → .9 pads to .900000 (=900_000) + 678ms = 678_900_000 ns
        assert_eq!(s.sleep_sum, Some(678_900_000));
        // 42.1 → .1 pads to .100000 (=100_000) + 42ms = 42_100_000 ns
        assert_eq!(s.block_sum, Some(42_100_000));
        // 7.999 → .999 pads to .999000 (=999_000) + 7ms = 7_999_000 ns
        assert_eq!(s.iowait_sum, Some(7_999_000));
    }

    /// `parsed_ns_from_dotted` rejects negative integer parts —
    /// `u64` parse fails on `-5`. The capture site
    /// `unwrap_or(0)`s these into the absent-counter zero per the
    /// best-effort capture contract, so a kernel that emits a
    /// negative SPLIT_NS (rare; can happen for clock skew on
    /// suspend/resume) does not pollute downstream metrics. The
    /// tally arg is `&mut None` here — the no-tally branch must
    /// still produce None for the negative case so synthetic-tree
    /// tests that don't carry a tally still observe the
    /// pre-tally semantics.
    #[test]
    fn parse_sched_negative_value_returns_none() {
        let raw = "wait_sum                                       :   -5.0\n";
        let s = parse_sched(raw, &mut None);
        assert_eq!(
            s.wait_sum, None,
            "negative ms part fails u64 parse → None; downstream \
             unwrap_or(0) collapses this to absent-counter zero",
        );
    }

    /// Negative dotted-ns value records into the [`ParseTally`]
    /// when one is supplied — pinning the tally-bump path so a
    /// regression that drops the per-line negative detection
    /// surfaces here rather than silently zeroing schedstat
    /// fields. Multiple negative lines bump independently;
    /// non-negative lines on the same parse pass do NOT bump.
    #[test]
    fn parse_sched_negative_value_records_into_tally() {
        let raw = "wait_sum                                       :   -5.0\n\
                   sum_sleep_runtime                              :   12.5\n\
                   sum_block_runtime                              :  -10.0\n";
        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        let s = parse_sched(raw, &mut tally_opt);
        assert_eq!(
            s.wait_sum, None,
            "negative wait_sum still reads None — the tally records \
             but does not change the per-field outcome",
        );
        assert_eq!(
            s.sleep_sum,
            Some(12_500_000),
            "non-negative neighbor still parses normally",
        );
        assert_eq!(s.block_sum, None, "negative block_sum reads None");
        // 2 negative dotted values landed in pending. Commit
        // through the Option-wrapped tally (NLL: while `tally_opt`
        // holds &mut tally, direct access to `tally` would be
        // a borrow-check error).
        tally_opt.as_mut().unwrap().commit_pending();
        // After this point, `tally_opt` is no longer used — NLL
        // releases the inner borrow so `tally` is reborrowable.
        let summary = tally.to_public();
        assert_eq!(
            summary.negative_dotted_values, 2,
            "two negative dotted lines bumped the per-snapshot \
             negative_dotted_values counter; non-negative neighbor \
             did not contribute",
        );
    }

    /// Ghost-filter discipline for the negative-dotted tally: a
    /// tid whose pending bumps are unwound via
    /// [`ParseTally::discard_pending`] must not contribute to
    /// the per-snapshot
    /// [`HostStateParseSummary::negative_dotted_values`]. Mirrors
    /// the read-failure tally's discard semantics so the two
    /// tally families stay symmetric under the ghost-filter
    /// path.
    #[test]
    fn parse_tally_negative_dotted_discard_pending_unwinds_bumps() {
        let raw = "wait_sum :   -5.0\n";
        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        let _ = parse_sched(raw, &mut tally_opt);
        // Pending bump landed; discard_pending must unwind it
        // before commit so the ghost-filtered tid leaves no trace
        // in the public surface. Same NLL-through-Option pattern
        // as `parse_sched_negative_value_records_into_tally`.
        tally_opt.as_mut().unwrap().discard_pending();
        let summary = tally.to_public();
        assert_eq!(
            summary.negative_dotted_values, 0,
            "discard_pending must unwind the negative-dotted \
             pending bump so a ghost-filtered tid does not \
             pollute the per-snapshot tally",
        );
    }

    /// Tally accumulates across multiple commits (multi-tid path
    /// — production captures invoke `parse_sched` once per tid
    /// and `commit_pending` between them). Pin that negative
    /// bumps from a SECOND tid land additively on top of the
    /// first tid's contribution rather than replacing it. Total
    /// after two commits is the sum of pending counts at each
    /// commit.
    #[test]
    fn parse_tally_negative_dotted_accumulates_across_commits() {
        let raw_a = "wait_sum : -1.0\n";
        let raw_b = "wait_sum   : -2.0\n\
                     sleep_max  : -3.0\n";
        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        let _ = parse_sched(raw_a, &mut tally_opt);
        // Commit tid A's 1 pending bump.
        tally_opt.as_mut().unwrap().commit_pending();
        // Now parse tid B's 2 pending bumps.
        let _ = parse_sched(raw_b, &mut tally_opt);
        tally_opt.as_mut().unwrap().commit_pending();
        let summary = tally.to_public();
        assert_eq!(
            summary.negative_dotted_values, 3,
            "1 commit + 2 commit = 3 total — multi-tid commits \
             must add, not overwrite. got {}",
            summary.negative_dotted_values,
        );
    }

    /// All-positive dotted input MUST NOT bump the
    /// `negative_dotted_values` counter. Pins that the negative
    /// detection is gated on the leading `-`, not triggered by
    /// any other parse path. Without this, a regression that
    /// always-bumped (e.g. moving the bump out of the Err arm)
    /// would let a clean host emit a non-zero count.
    #[test]
    fn parse_tally_negative_dotted_zero_for_positive_only_input() {
        let raw = "wait_sum            : 100.5\n\
                   sum_sleep_runtime   : 200\n\
                   sum_block_runtime   : 0.999\n\
                   wait_max            : 0\n\
                   exec_max            : 7\n";
        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        let _ = parse_sched(raw, &mut tally_opt);
        tally_opt.as_mut().unwrap().commit_pending();
        let summary = tally.to_public();
        assert_eq!(
            summary.negative_dotted_values, 0,
            "all-positive dotted input must not bump the \
             negative-dotted tally; got {}",
            summary.negative_dotted_values,
        );
    }

    /// Sub-millisecond negative SPLIT_NS shape: kernel emits
    /// `0.-NNN` when the integer part is `(x / 1_000_000)` for
    /// `x` in `(-1_000_000, 0)` — `%Ld` yields `0` (no sign
    /// because integer division of a negative by 1M lands at
    /// `0` not `-1`) and `%06ld` carries the negative
    /// remainder. Without the fractional-side detection in
    /// [`parsed_ns_from_dotted`] the integer-only check would
    /// miss this shape entirely. Pin both the parser-level
    /// detection and the tally-bump path.
    #[test]
    fn parsed_ns_from_dotted_sub_millisecond_negative_detected() {
        // Direct parser-level shape.
        assert_eq!(
            parsed_ns_from_dotted("0.-000500"),
            Err(ParseDottedNs::Negative),
            "0.-NNN shape (sub-ms negative SPLIT_NS) MUST route \
             through Negative — most schedstat negatives land \
             sub-millisecond and would otherwise slip through",
        );
        assert_eq!(
            parsed_ns_from_dotted("0.-1"),
            Err(ParseDottedNs::Negative),
            "single-digit sub-ms negative shape detected",
        );
        // End-to-end through parse_sched + tally.
        let raw = "wait_sum : 0.-000500\n\
                   sleep_max : 0.-1\n";
        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        let s = parse_sched(raw, &mut tally_opt);
        assert_eq!(
            s.wait_sum, None,
            "sub-ms negative wait_sum collapses to None",
        );
        assert_eq!(
            s.sleep_max, None,
            "sub-ms negative sleep_max collapses to None",
        );
        tally_opt.as_mut().unwrap().commit_pending();
        let summary = tally.to_public();
        assert_eq!(
            summary.negative_dotted_values, 2,
            "two sub-ms negatives both bump the tally — pins \
             that the integer-only detection is NOT enough on \
             its own",
        );
    }

    /// Bare-integer (no-dot) negative value is also recorded —
    /// the kernel's PN_SCHEDSTAT format always emits the dotted
    /// form, but the `parsed_ns_from_dotted` function's bare
    /// branch is exercised by the `slice` (P_SCHEDSTAT, no dot)
    /// arm and by graceful degradation against fixtures that
    /// drop the fractional part. A bare `-5` lands the same
    /// `Negative` arm as `-5.0` so the tally treats both
    /// identically.
    ///
    /// `wait_sum` itself is dotted-only in real kernel output,
    /// but `parsed_ns_from_dotted`'s bare-integer fallback is
    /// reachable via test fixtures that drop the dot — pinning
    /// the bare-branch negative detection ensures the two
    /// branches stay symmetric.
    #[test]
    fn parsed_ns_from_dotted_negative_bare_branch_records() {
        // Direct call into the parser: bare-integer negative.
        assert_eq!(
            parsed_ns_from_dotted("-5"),
            Err(ParseDottedNs::Negative),
            "bare-integer negative routes through Negative",
        );
        // Dotted negative.
        assert_eq!(
            parsed_ns_from_dotted("-5.0"),
            Err(ParseDottedNs::Negative),
            "dotted negative routes through Negative",
        );
        // Non-numeric malformed.
        assert_eq!(
            parsed_ns_from_dotted("garbage"),
            Err(ParseDottedNs::Malformed),
            "non-numeric input routes through Malformed, not \
             Negative — the tally must NOT bump on garbage",
        );
        assert_eq!(
            parsed_ns_from_dotted("garbage.5"),
            Err(ParseDottedNs::Malformed),
            "non-numeric integer part with fractional routes \
             through Malformed",
        );
        assert_eq!(
            parsed_ns_from_dotted(""),
            Err(ParseDottedNs::Malformed),
            "empty input routes through Malformed",
        );
        assert_eq!(
            parsed_ns_from_dotted("5"),
            Ok(5),
            "bare positive integer parses",
        );
        assert_eq!(
            parsed_ns_from_dotted("5.500"),
            Ok(5_500_000),
            "positive dotted parses normally",
        );
    }

    /// Bare-key names (no `se.statistics.` prefix) must still
    /// populate — some kernels emit `nr_wakeups : N` at the top
    /// level. The parser's `rsplit('.').next()` treats a no-dot
    /// string as the whole string. Coverage spans the wakeup
    /// family, one of the new migration counters, and one of the
    /// new *_max ns fields, to prove the bare-key path lights up
    /// every parser arm shape (parsed_u64 + parsed_ns_from_dotted).
    #[test]
    fn parse_sched_bare_key_names_populate_same_fields() {
        let raw = "\
             nr_wakeups                                     :         11\n\
             nr_wakeups_local                               :          8\n\
             nr_wakeups_remote                              :          3\n\
             nr_wakeups_sync                                :          2\n\
             nr_wakeups_migrate                             :          1\n\
             nr_wakeups_idle                                :          4\n\
             nr_migrations_cold                             :         42\n\
             wait_max                                       :     999.5\n";
        let s = parse_sched(raw, &mut None);
        assert_eq!(s.nr_wakeups, Some(11));
        assert_eq!(s.nr_wakeups_local, Some(8));
        assert_eq!(s.nr_wakeups_remote, Some(3));
        assert_eq!(s.nr_wakeups_sync, Some(2));
        assert_eq!(s.nr_wakeups_migrate, Some(1));
        assert_eq!(s.nr_wakeups_idle, Some(4));
        assert_eq!(
            s.nr_migrations_cold,
            Some(42),
            "bare-key `nr_migrations_cold` must populate via \
             rsplit('.').next() returning the whole no-dot string",
        );
        assert_eq!(
            s.wait_max,
            Some(999_500_000),
            "bare-key `wait_max` must populate via the \
             parsed_ns_from_dotted path; 999.5 → 999_500_000 ns",
        );
    }

    /// Future `stats.` or other prefix variants must also
    /// populate — the parser matches on the LAST dot-delimited
    /// segment, so any enclosing prefix is ignored by design.
    #[test]
    fn parse_sched_alternative_prefix_populates_same_fields() {
        let raw = "\
             stats.nr_wakeups                               :         42\n\
             some.other.prefix.nr_migrations                :          9\n";
        let s = parse_sched(raw, &mut None);
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
        let s = parse_sched(raw, &mut None);
        assert_eq!(s.nr_wakeups, Some(11));
        assert_eq!(s.nr_migrations, Some(9));
    }

    // ------------------------------------------------------------
    // H5 — ProbeSummary discipline
    //
    // The capture pipeline tallies every per-tgid attach result and
    // every per-tid probe_thread result into a [`ProbeSummary`]
    // before emitting one info-level line per snapshot. The tests
    // below pin the summary's accounting + EPERM-hint policy
    // independently of any real ptrace dispatch — a regression that
    // mis-categorised a tag, dropped the dominant-tag tiebreak,
    // or flipped the ptrace-dominates threshold lands here loudly.
    // ------------------------------------------------------------

    /// Construct a populated `ProbeSummary` for unit-test cases.
    /// Lifts the otherwise-repetitive default-then-mutate pattern
    /// out of every test (clippy's `field_reassign_with_default`
    /// flags it; using a constructor keeps the tests terse).
    fn make_summary(
        failed: u64,
        attach: &[(&'static str, u64)],
        probe: &[(&'static str, u64)],
    ) -> ProbeSummary {
        ProbeSummary {
            failed,
            attach_tag_counts: attach.iter().copied().collect(),
            probe_tag_counts: probe.iter().copied().collect(),
            ..ProbeSummary::default()
        }
    }

    #[test]
    fn probe_summary_dominant_tag_picks_highest_count() {
        // dwarf-parse-failure is an ACTIONABLE attach tag (it
        // signals a stripped binary worth surfacing), so it
        // survives the `jemalloc-not-found / readlink-failure`
        // filter in `dominant_tag` and competes against the probe
        // side on raw count.
        let s = make_summary(6, &[("dwarf-parse-failure", 5)], &[("ptrace-seize", 1)]);
        assert_eq!(s.dominant_tag(), Some("dwarf-parse-failure"));
    }

    /// `dominant_tag` filters `jemalloc-not-found` and
    /// `readlink-failure` out of the attach side BEFORE the
    /// max-by-count step. Both are the expected outcome on the
    /// bulk of system processes (most tgids are not jemalloc-
    /// linked; short-lived tgids race readlink mid-walk), so
    /// surfacing them as the dominant tag would drown actionable
    /// signal under benign noise. This pin proves the filter
    /// engages even when the filtered tag has the highest raw
    /// count: 100 jemalloc-not-found events lose to a single
    /// ptrace-seize because the former does not enter the
    /// comparison at all.
    ///
    /// Also covers `readlink-failure` symmetrically — both
    /// non-actionable attach tags are filtered, only one is in
    /// the production code's matches! arm but the test doubles
    /// up to keep the contract from quietly degrading to "only
    /// jemalloc-not-found is filtered."
    #[test]
    fn probe_summary_dominant_tag_filters_non_actionable_attach_tags() {
        // jemalloc-not-found dominates by count but is filtered.
        let s = make_summary(101, &[("jemalloc-not-found", 100)], &[("ptrace-seize", 1)]);
        assert_eq!(
            s.dominant_tag(),
            Some("ptrace-seize"),
            "jemalloc-not-found must be filtered out even at \
             100x the count of an actionable tag",
        );
        // readlink-failure dominates by count but is filtered.
        let s = make_summary(101, &[("readlink-failure", 100)], &[("get-regset", 1)]);
        assert_eq!(
            s.dominant_tag(),
            Some("get-regset"),
            "readlink-failure must be filtered out even at \
             100x the count of an actionable tag",
        );
        // Both filtered tags present together: still filtered;
        // the actionable probe tag wins.
        let s = make_summary(
            201,
            &[("jemalloc-not-found", 100), ("readlink-failure", 100)],
            &[("waitpid", 1)],
        );
        assert_eq!(
            s.dominant_tag(),
            Some("waitpid"),
            "both filtered attach tags together must NOT push their \
             aggregate above an actionable probe tag",
        );
        // Only filtered tags, no actionable counterparts: None
        // (the filter removes them, the chain is empty).
        let s = make_summary(5, &[("jemalloc-not-found", 5)], &[]);
        assert_eq!(
            s.dominant_tag(),
            None,
            "only-filtered-tags case must produce None, not the \
             filtered tag itself",
        );
    }

    #[test]
    fn probe_summary_dominant_tag_breaks_ties_reverse_alphabetically() {
        // Two tags tied at count=2 — the tiebreak's secondary key
        // is `b.0.cmp(a.0)` (note the flip), so the alphabetically-
        // EARLIER tag wins. With "ptrace-seize" vs
        // "dwarf-parse-failure", "dwarf-parse-failure" precedes
        // "ptrace-seize" lexicographically, so it wins. This
        // "reverse-alphabetical" framing matches how the
        // `dominant_tag` doc describes the comparator.
        let s = make_summary(4, &[("ptrace-seize", 2)], &[("dwarf-parse-failure", 2)]);
        assert_eq!(s.dominant_tag(), Some("dwarf-parse-failure"));
    }

    #[test]
    fn probe_summary_ptrace_dominates_when_half_of_failures() {
        // 3/6 failures are ptrace-attach — meets the half
        // threshold so the EPERM hint engages.
        let s = make_summary(6, &[], &[("ptrace-seize", 3), ("waitpid", 3)]);
        assert!(s.ptrace_dominates());
    }

    #[test]
    fn probe_summary_ptrace_does_not_dominate_when_below_half() {
        let s = make_summary(6, &[], &[("ptrace-seize", 2), ("waitpid", 4)]);
        assert!(!s.ptrace_dominates());
    }

    #[test]
    fn probe_summary_no_failures_no_dominant_tag() {
        let s = ProbeSummary::default();
        assert!(!s.ptrace_dominates());
        assert_eq!(s.dominant_tag(), None);
    }

    /// EPERM remediation hint references `$(which ktstr)` rather
    /// than a hardcoded path — pins the wording so a future drift
    /// to a fixed install path lands here loudly.
    #[test]
    fn ptrace_eperm_hint_uses_which_ktstr() {
        assert!(
            PTRACE_EPERM_HINT.contains("$(which ktstr)"),
            "EPERM hint must use $(which ktstr) for portability, got: {PTRACE_EPERM_HINT}",
        );
        assert!(PTRACE_EPERM_HINT.contains("cap_sys_ptrace"));
        assert!(PTRACE_EPERM_HINT.contains("yama.ptrace_scope"));
    }

    /// `to_public()` carries every counter through verbatim and
    /// projects `dominant_tag` to `dominant_failure` as the owned
    /// tag string. Pins the public surface contract so a refactor
    /// that drops a counter or rewires the projection lands here.
    #[test]
    fn to_public_carries_counters_and_dominant_tag() {
        let mut s = make_summary(3, &[("dwarf-parse-failure", 2)], &[("ptrace-seize", 1)]);
        s.tgids_walked = 10;
        s.jemalloc_detected = 5;
        s.probed_ok = 4;

        let public = s.to_public();
        assert_eq!(public.tgids_walked, 10);
        assert_eq!(public.jemalloc_detected, 5);
        assert_eq!(public.probed_ok, 4);
        assert_eq!(public.failed, 3);
        assert_eq!(
            public.dominant_failure.as_deref(),
            Some("dwarf-parse-failure"),
            "dominant_tag picks the highest-count actionable tag, \
             projected as an owned String",
        );
        // 1 ptrace-seize out of 3 failed (33%) is below the 50%
        // hint-trigger threshold → privilege_dominant is false.
        assert!(
            !public.privilege_dominant,
            "ptrace 1/3 < 50% → privilege_dominant false",
        );
    }

    /// Zero-failure summary projects to `dominant_failure: None` —
    /// the absence-of-failure case must surface as None, not an
    /// empty string. Mirrors the internal `dominant_tag` returning
    /// None when no actionable tags remain after the
    /// non-actionable filter (the fixture seeds
    /// `jemalloc-not-found`, which `dominant_tag` filters out).
    /// `privilege_dominant` must also be false (no failures to
    /// dominate).
    #[test]
    fn to_public_dominant_failure_is_none_when_no_failures() {
        let s = make_summary(0, &[("jemalloc-not-found", 12)], &[]);
        let public = s.to_public();
        assert_eq!(public.failed, 0);
        assert!(
            public.dominant_failure.is_none(),
            "no actionable failures means dominant_failure is None; \
             got {:?}",
            public.dominant_failure,
        );
        assert!(
            !public.privilege_dominant,
            "no failures means privilege_dominant is false",
        );
    }

    /// Privilege-dominated snapshot projects
    /// `privilege_dominant: true` so a downstream consumer can
    /// reproduce the EPERM-hint trigger condition without parsing
    /// the tracing summary. Mirrors the
    /// `summary_emits_privilege_hint_when_ptrace_dominates`
    /// emission test below.
    #[test]
    fn to_public_privilege_dominant_when_ptrace_crosses_threshold() {
        // 4 failed total, all ptrace-seize → 100% ≥ 50% → true.
        let s = make_summary(4, &[], &[("ptrace-seize", 4)]);
        let public = s.to_public();
        assert_eq!(public.failed, 4);
        assert!(
            public.privilege_dominant,
            "ptrace 4/4 ≥ 50% → privilege_dominant true",
        );

        // 2 ptrace + 2 dwarf = 50% / 50% → boundary
        // (`total_ptrace * 2 >= self.failed` accepts equality).
        let s = make_summary(4, &[("dwarf-parse-failure", 2)], &[("ptrace-seize", 2)]);
        let public = s.to_public();
        assert!(
            public.privilege_dominant,
            "ptrace 2/4 = 50% boundary → privilege_dominant true (>= threshold)",
        );

        // 1 ptrace + 3 dwarf = 25% < 50% → false.
        let s = make_summary(4, &[("dwarf-parse-failure", 3)], &[("ptrace-seize", 1)]);
        let public = s.to_public();
        assert!(
            !public.privilege_dominant,
            "ptrace 1/4 < 50% → privilege_dominant false",
        );
    }

    /// `privilege_dominant` covers the full ptrace tag set, the
    /// smallest-`failed` corners of the threshold, and the default
    /// shape of the public surface. Pins:
    ///
    /// 1. `ptrace-interrupt` alone trips the threshold — proves the
    ///    `matches!` arm in `ptrace_dominates` covers both tags, not
    ///    just `ptrace-seize`.
    /// 2. `dwarf-parse-failure` (2) plus split ptrace tags
    ///    (`ptrace-seize` 1 + `ptrace-interrupt` 1) out of 4 failed —
    ///    proves `privilege_dominant` and `dominant_failure` are
    ///    independent reductions and can DIVERGE: summed ptrace
    ///    crosses the 50% gate (`privilege_dominant: true`) while
    ///    `dominant_failure` names the non-ptrace tag that won the
    ///    single-tag plurality (`dwarf-parse-failure`).
    /// 3. `failed == 1` with one ptrace tag is the smallest input
    ///    that flips the gate true (1*2 >= 1).
    /// 4. `failed == 1` with one non-ptrace tag is the smallest
    ///    input that keeps the gate false (0*2 < 1) — pins that
    ///    `total_ptrace == 0` keeps the gate false even when
    ///    `failed > 0`.
    /// 5. `HostStateProbeSummary::default()` has
    ///    `privilege_dominant: false` — pins
    ///    `HostStateProbeSummary::default()` for callers that may
    ///    use struct-update syntax.
    /// 6. ptrace wins the single-tag plurality but stays below the
    ///    50% threshold — the converse of bullet 2: `dominant_failure`
    ///    names a ptrace tag while `privilege_dominant` is `false`.
    ///    Pins the converse direction of the independence claim.
    #[test]
    fn to_public_privilege_dominant_ptrace_interrupt_and_edge_cases() {
        // 1. ptrace-interrupt alone: 2/2 = 100% ≥ 50% → true.
        let s = make_summary(2, &[], &[("ptrace-interrupt", 2)]);
        let public = s.to_public();
        assert!(
            public.privilege_dominant,
            "ptrace-interrupt 2/2 ≥ 50% → privilege_dominant true \
             (matches! arm covers ptrace-interrupt as well as ptrace-seize)",
        );

        // 2. divergence: summed ptrace tags trip the privilege gate
        //    while a non-ptrace tag wins the single-tag plurality.
        //    dwarf-parse-failure (2) + ptrace-seize (1) + ptrace-interrupt (1)
        //    out of 4 failed: total_ptrace = 2, 2*2 = 4 >= 4 →
        //    privilege_dominant true; dominant_tag picks
        //    dwarf-parse-failure as the highest single-tag count (2).
        //    Pins that the two fields reduce independently.
        let s = make_summary(
            4,
            &[("dwarf-parse-failure", 2)],
            &[("ptrace-seize", 1), ("ptrace-interrupt", 1)],
        );
        let public = s.to_public();
        assert!(
            public.privilege_dominant,
            "summed ptrace 2/4 ≥ 50% → privilege_dominant true",
        );
        assert_eq!(
            public.dominant_failure.as_deref(),
            Some("dwarf-parse-failure"),
            "dominant_failure names the non-ptrace tag that won the \
             single-tag plurality while privilege_dominant is true — \
             proves the two fields are independent",
        );

        // 3. smallest true: failed == 1 with one ptrace tag.
        let s = make_summary(1, &[], &[("ptrace-seize", 1)]);
        let public = s.to_public();
        assert!(
            public.privilege_dominant,
            "ptrace 1/1 ≥ 50% → privilege_dominant true at the \
             smallest-failed boundary",
        );

        // 4. smallest false: failed == 1 with no ptrace tag. Guards
        //    that `total_ptrace == 0` keeps the gate false even when
        //    `failed > 0`.
        let s = make_summary(1, &[("dwarf-parse-failure", 1)], &[]);
        let public = s.to_public();
        assert!(
            !public.privilege_dominant,
            "no ptrace tags with failed == 1 → privilege_dominant \
             false (total_ptrace == 0 keeps the gate closed)",
        );

        // 5. default invariant: a freshly-defaulted summary must
        //    not claim privilege dominance.
        assert!(
            !HostStateProbeSummary::default().privilege_dominant,
            "HostStateProbeSummary::default().privilege_dominant \
             must be false",
        );

        // 6. converse: ptrace wins the per-tag plurality but stays
        //    below the 50% threshold → privilege_dominant false while
        //    dominant_failure names the ptrace tag.
        let s = make_summary(
            10,
            &[("dwarf-parse-failure", 3), ("jemalloc-in-dso", 3)],
            &[("ptrace-seize", 4)],
        );
        let public = s.to_public();
        assert!(
            !public.privilege_dominant,
            "ptrace 4/10 < 50% → privilege_dominant false",
        );
        assert_eq!(
            public.dominant_failure.as_deref(),
            Some("ptrace-seize"),
            "dominant_failure names a ptrace tag while privilege_dominant \
             is false — converse of the independence claim",
        );
    }

    /// `remediation_hint()` returns `Some` exactly when
    /// `privilege_dominant` is true, and the returned text matches
    /// the same `PTRACE_EPERM_HINT` constant the emission path
    /// prints — so a downstream consumer surfaces the same fix-it
    /// message the operator-facing tracing summary does. Pins both
    /// the gate semantics and the text-equality contract.
    #[test]
    fn remediation_hint_returns_some_iff_privilege_dominant() {
        // privilege_dominant=true → Some(PTRACE_EPERM_HINT).
        let ps = HostStateProbeSummary {
            privilege_dominant: true,
            ..Default::default()
        };
        assert_eq!(
            ps.remediation_hint(),
            Some(PTRACE_EPERM_HINT),
            "privilege_dominant=true must surface the same hint text \
             the tracing summary prints",
        );

        // privilege_dominant=false → None.
        let ps = HostStateProbeSummary::default();
        assert!(
            !ps.privilege_dominant,
            "default privilege_dominant must be false (sanity)",
        );
        assert_eq!(
            ps.remediation_hint(),
            None,
            "privilege_dominant=false → remediation_hint returns None",
        );
    }

    // ------------------------------------------------------------
    // Summary-line emission discipline (tracing assertions)
    //
    // emit_probe_summary is the single source of truth for the
    // operator-facing per-snapshot summary. The tests below run
    // under `#[traced_test]` so the emitted `tracing::info!` /
    // `tracing::warn!` events are captured into an in-memory
    // buffer queryable via `logs_contain`. Without these, a
    // refactor that silently dropped the dominant-tag clause or
    // the EPERM hint would be invisible — the structural unit
    // tests above pin the helpers that feed the summary, but
    // only an emission test pins what the operator actually
    // reads.
    // ------------------------------------------------------------

    /// Zero-failure snapshot emits a clean summary line — no
    /// failure-class clause, no privilege hint. Pins the "happy
    /// path" shape so a future refactor that always-appended a
    /// hint would surface here.
    ///
    /// Test fn names deliberately avoid the substrings asserted
    /// against (e.g. "dominant", "hint") because
    /// `tracing-test`'s `logs_contain` matches across the entire
    /// captured frame INCLUDING the span (which is the test fn
    /// name). The terse `summary_emits_*` naming keeps the span
    /// text disjoint from the assertions.
    #[traced_test]
    #[test]
    fn summary_emits_clean_line_when_no_failures() {
        let summary = make_summary(0, &[("jemalloc-not-found", 12)], &[]);
        emit_probe_summary(&summary);
        assert!(logs_contain("host-state probe:"));
        assert!(logs_contain("0 tgids walked"));
        assert!(logs_contain("0 failed"));
        assert!(
            !logs_contain("(dominant:"),
            "no failures means the dominant-tag clause is omitted",
        );
        assert!(
            !logs_contain("hint:"),
            "no failures means the EPERM hint is omitted",
        );
    }

    /// Privilege-dominated snapshot emits the hint with the
    /// `$(which ktstr)` substring intact. Catches a regression
    /// that drops the hint when the ptrace-dominates threshold
    /// fires.
    #[traced_test]
    #[test]
    fn summary_emits_privilege_hint_when_ptrace_dominates() {
        let summary = ProbeSummary {
            tgids_walked: 4,
            jemalloc_detected: 2,
            probed_ok: 0,
            failed: 4,
            attach_tag_counts: BTreeMap::new(),
            probe_tag_counts: [("ptrace-seize", 4u64)].into_iter().collect(),
        };
        emit_probe_summary(&summary);
        assert!(logs_contain("(dominant: ptrace-seize"));
        assert!(logs_contain("hint:"));
        assert!(logs_contain("$(which ktstr)"));
        assert!(logs_contain("cap_sys_ptrace"));
        assert!(logs_contain("yama.ptrace_scope"));
    }

    /// `ptrace-interrupt`-dominated snapshot also emits the
    /// privilege hint. Pins the `matches!` arm in
    /// `ProbeSummary::ptrace_dominates` covering both ptrace
    /// tags, not just `ptrace-seize` — a regression that
    /// narrowed the gate to `ptrace-seize` only would silently
    /// drop the hint on hosts where the per-thread interrupt
    /// step (rather than the initial seize) is the failure
    /// mode (for example: yama scope=1 lets the seize succeed
    /// against an opted-in target but blocks the per-tid
    /// `PTRACE_INTERRUPT` step against threads created after
    /// the opt-in window).
    #[traced_test]
    #[test]
    fn summary_emits_privilege_hint_when_ptrace_interrupt_dominates() {
        let summary = ProbeSummary {
            tgids_walked: 4,
            jemalloc_detected: 2,
            probed_ok: 0,
            failed: 4,
            attach_tag_counts: BTreeMap::new(),
            probe_tag_counts: [("ptrace-interrupt", 4u64)].into_iter().collect(),
        };
        emit_probe_summary(&summary);
        assert!(logs_contain("(dominant: ptrace-interrupt"));
        assert!(logs_contain("hint:"));
        assert!(logs_contain("$(which ktstr)"));
        assert!(logs_contain("cap_sys_ptrace"));
        assert!(logs_contain("yama.ptrace_scope"));
    }

    /// Mixed-failure snapshot (DWARF + ptrace) where ptrace
    /// stays below the half threshold emits the dominant tag
    /// but NOT the privilege hint — a stripped-binary host
    /// doesn't need the privilege fix, it needs debuginfo.
    #[traced_test]
    #[test]
    fn summary_omits_privilege_hint_when_debuginfo_failures_lead() {
        let summary = ProbeSummary {
            tgids_walked: 5,
            jemalloc_detected: 3,
            probed_ok: 0,
            failed: 5,
            attach_tag_counts: [("dwarf-parse-failure", 4u64)].into_iter().collect(),
            probe_tag_counts: [("ptrace-seize", 1u64)].into_iter().collect(),
        };
        emit_probe_summary(&summary);
        assert!(logs_contain("(dominant: dwarf-parse-failure"));
        assert!(
            !logs_contain("hint:"),
            "DWARF-dominated failures must NOT trigger the privilege \
             hint — only privilege failures earn the privilege remediation",
        );
    }

    /// Clean parse-summary emission: zero failures, zero negative
    /// dotted values. Pins that no dominant-tag clause, no kconfig
    /// hint, and no negative-clause render when the underlying
    /// signals are zero. Mirrors the
    /// `summary_emits_clean_line_when_no_failures` discipline for
    /// the probe summary side.
    ///
    /// Test fn name uses `parse_summary_emits_*` rather than
    /// `summary_emits_*` to keep the captured span text disjoint
    /// from the asserted substrings (`tracing-test`'s
    /// `logs_contain` matches the entire captured frame including
    /// the span — same caveat the probe-summary emit tests
    /// document).
    #[traced_test]
    #[test]
    fn parse_summary_emits_clean_line_when_no_failures() {
        let tally = ParseTally::default();
        emit_parse_summary(&tally);
        assert!(logs_contain("host-state parse:"));
        assert!(logs_contain("0 tids walked"));
        assert!(logs_contain("0 read failures"));
        assert!(
            !logs_contain("(dominant:"),
            "no failures means the dominant clause is omitted",
        );
        assert!(
            !logs_contain("hint:"),
            "no failures means the kconfig hint is omitted",
        );
        assert!(
            !logs_contain("negative-dotted"),
            "zero negative-dotted values means the negative \
             clause is omitted",
        );
    }

    /// Negative-dotted clause renders when the tally carries any
    /// negative bumps. Pins the `, N negative-dotted values`
    /// substring so a regression that drops the clause when read
    /// failures are zero (the emit's failure path) surfaces
    /// here.
    #[traced_test]
    #[test]
    fn parse_summary_emits_negative_dotted_clause_when_present() {
        let mut tally = ParseTally {
            tids_walked: 5,
            ..ParseTally::default()
        };
        // Drive the negative-dotted counter through the public
        // path: pending bumps + commit, mirroring the production
        // capture pipeline.
        tally.record_negative_dotted();
        tally.record_negative_dotted();
        tally.record_negative_dotted();
        tally.commit_pending();
        emit_parse_summary(&tally);
        assert!(
            logs_contain("3 negative-dotted values"),
            "negative-dotted clause must surface the count when \
             the tally is non-zero — the operator-visibility \
             motivation depends on this rendering",
        );
        assert!(logs_contain("0 read failures"));
    }

    /// Kconfig hint renders alongside the dominant clause when
    /// schedstat / io failures dominate. Pins both clauses
    /// firing together so a refactor that conditioned them
    /// independently surfaces here.
    #[traced_test]
    #[test]
    fn parse_summary_emits_kconfig_hint_when_dominant() {
        let mut tally = ParseTally {
            tids_walked: 100,
            ..ParseTally::default()
        };
        // 60 schedstat + 40 io = 100% kconfig share, well above
        // the 50% gate.
        for _ in 0..60 {
            tally.record_failure("schedstat");
        }
        for _ in 0..40 {
            tally.record_failure("io");
        }
        tally.commit_pending();
        emit_parse_summary(&tally);
        assert!(logs_contain("(dominant: schedstat)"));
        assert!(logs_contain("hint:"));
        assert!(logs_contain("CONFIG_SCHEDSTATS"));
        assert!(logs_contain("CONFIG_TASK_IO_ACCOUNTING"));
    }

    /// `try_attach_probe_for_tgid_at` against a known-bad pid (0,
    /// reserved by the kernel) emits a `tracing::warn!` event
    /// (not debug) because PidMissing is NOT the
    /// jemalloc-not-found case — it's a hard error worth
    /// surfacing. Pins the level-routing rule from the helper's
    /// doc.
    #[traced_test]
    #[test]
    fn try_attach_probe_for_tgid_at_warns_on_pid_missing() {
        let mut summary = ProbeSummary::default();
        let probe = try_attach_probe_for_tgid_at(Path::new(DEFAULT_PROC_ROOT), 0, &mut summary);
        assert!(probe.is_none(), "pid 0 must not produce a probe");
        // PidMissing → tag "pid-missing", logged at warn, counted as failed.
        assert!(logs_contain("attach failed"));
        assert!(logs_contain("pid-missing"));
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.jemalloc_detected, 0);
        assert_eq!(summary.tgids_walked, 1);
        assert_eq!(
            summary.attach_tag_counts.get("pid-missing").copied(),
            Some(1),
            "PidMissing tag must increment its bucket",
        );
    }

    /// `try_attach_probe_for_tgid_at` against a real process that
    /// is NOT jemalloc-linked (`/bin/sleep` spawned for the
    /// duration of the test) returns `None` AND logs at debug,
    /// not warn — the JemallocNotFound case is the expected
    /// outcome for the bulk of system processes and must not
    /// flood the operator's log. Pins the
    /// `jemalloc-not-found → debug` routing rule.
    #[traced_test]
    #[test]
    fn try_attach_probe_for_tgid_at_debugs_on_non_jemalloc_target() {
        // /bin/sleep is a coreutils binary not linked against
        // jemalloc; attach_jemalloc walks its /proc/<pid>/maps,
        // finds no TSD symbol, and returns JemallocNotFound.
        let mut child = match std::process::Command::new("sleep")
            .arg("3")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                eprintln!("skipping — /bin/sleep unavailable");
                return;
            }
        };
        // Poll for `/proc/<pid>/exe` to become readable rather than
        // burning a hardcoded settle window. On a fast host the
        // exe symlink resolves within microseconds of fork+exec; on
        // a contended CI runner it can lag a few ms. A 1 s deadline
        // with 1 ms backoff bounds the worst case while keeping the
        // common case nearly instantaneous, and deterministically
        // gates the test on the actual readiness signal rather than
        // a guess. `read_link` is the same syscall the probe attach
        // exercises, so once it succeeds the downstream
        // `try_attach_probe_for_tgid_at` call is guaranteed to find
        // an exe symlink it can resolve.
        let pid = child.id() as i32;
        let exe_link = std::path::PathBuf::from(format!("/proc/{pid}/exe"));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        while std::fs::read_link(&exe_link).is_err() {
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "/proc/{pid}/exe did not become readable within 1s — \
                     kernel did not surface the freshly-forked child's exe \
                     symlink in time, the test cannot proceed"
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let mut summary = ProbeSummary::default();
        let probe = try_attach_probe_for_tgid_at(Path::new(DEFAULT_PROC_ROOT), pid, &mut summary);

        let _ = child.kill();
        let _ = child.wait();

        assert!(probe.is_none(), "sleep is not jemalloc-linked");
        assert_eq!(summary.tgids_walked, 1);
        assert_eq!(summary.jemalloc_detected, 0);
        assert_eq!(
            summary.failed, 0,
            "jemalloc-not-found must NOT count as failure — it's the \
             expected outcome for the bulk of system processes",
        );
        assert_eq!(
            summary.attach_tag_counts.get("jemalloc-not-found").copied(),
            Some(1),
        );
        // The debug event carries the "attach skipped" message;
        // tracing-test's logs_contain looks across all captured
        // events including debug.
        assert!(
            logs_contain("attach skipped"),
            "JemallocNotFound must emit the debug 'attach skipped' \
             event so log filters can route it separately from \
             actionable warnings",
        );
        assert!(
            !logs_contain("attach failed"),
            "jemalloc-not-found must NOT emit the warn 'attach failed' \
             event — that level is reserved for actionable failures",
        );
    }

    // ------------------------------------------------------------
    // T28 — HostStateParseSummary: per-file read-failure tally
    // ------------------------------------------------------------

    /// Stage a synthetic procfs tree for parse-summary tests:
    /// a single live tgid + tid with `comm` and `stat` populated
    /// so the ghost filter does NOT fire (start_time is parseable
    /// from `stat`). The caller then deletes the specific
    /// per-file targets they want to fail. `cgroup` and other
    /// non-asserted files are populated so the surrounding reads
    /// succeed and the tally only counts the targeted failures.
    fn stage_minimal_proc_for_parse(root: &Path, tgid: i32, tid: i32) {
        use std::fs;
        let tgid_dir = root.join(tgid.to_string());
        let task_dir = tgid_dir.join("task").join(tid.to_string());
        fs::create_dir_all(&task_dir).unwrap();
        fs::write(tgid_dir.join("comm"), "p\n").unwrap();
        fs::write(task_dir.join("comm"), "live\n").unwrap();
        // Non-zero start_time keeps the ghost filter from firing
        // even when other files vanish.
        let stat_line = format!(
            "{tid} (live) R 1 2 3 4 5 6 7 0 8 0 10 11 12 13 14 0 1 0 \
             555555 100 200 300 400 500 600 700 800 900 1000 1100 \
             1200 1300 1400 1500 1600 1700 1800 0\n"
        );
        fs::write(task_dir.join("stat"), stat_line).unwrap();
        fs::write(task_dir.join("schedstat"), "0 0 0\n").unwrap();
        fs::write(
            task_dir.join("status"),
            "voluntary_ctxt_switches:\t0\n\
             nonvoluntary_ctxt_switches:\t0\n",
        )
        .unwrap();
        fs::write(task_dir.join("io"), "rchar: 0\n").unwrap();
        fs::write(task_dir.join("sched"), "").unwrap();
        fs::write(task_dir.join("cgroup"), "0::/\n").unwrap();
    }

    /// Per-file-kind tally: deleting `schedstat` lands a single
    /// `"schedstat"` failure in the summary's per-file map. Other
    /// categories stay at zero (key absent from the map).
    #[test]
    fn parse_summary_records_schedstat_failure() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 5050;
        let tid: i32 = 5051;
        stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
        // Delete schedstat so the read fails.
        std::fs::remove_file(
            proc_tmp
                .path()
                .join(tgid.to_string())
                .join("task")
                .join(tid.to_string())
                .join("schedstat"),
        )
        .unwrap();

        // capture_with(_, _, false) skips the production gate so
        // parse_summary is None; use true and stage a /proc tree
        // that the host_context probe absorbs without panicking.
        // For the synthetic-tree pattern, stage a tally directly.
        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let _ = capture_thread_at_with_tally(
            proc_tmp.path(),
            tgid,
            tid,
            "p",
            "live",
            false,
            &mut tally_opt,
        );
        tally_opt.as_mut().unwrap().commit_pending();

        let summary = tally.to_public();
        assert_eq!(summary.tids_walked, 1);
        assert_eq!(summary.read_failures, 1);
        assert_eq!(summary.read_failures_by_file.get("schedstat"), Some(&1));
        assert!(!summary.read_failures_by_file.contains_key("stat"));
        assert!(!summary.read_failures_by_file.contains_key("io"));
    }

    /// Per-file-kind tally: deleting `io` lands an `"io"` failure.
    #[test]
    fn parse_summary_records_io_failure() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 5060;
        let tid: i32 = 5061;
        stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
        std::fs::remove_file(
            proc_tmp
                .path()
                .join(tgid.to_string())
                .join("task")
                .join(tid.to_string())
                .join("io"),
        )
        .unwrap();

        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let _ = capture_thread_at_with_tally(
            proc_tmp.path(),
            tgid,
            tid,
            "p",
            "live",
            false,
            &mut tally_opt,
        );
        tally_opt.as_mut().unwrap().commit_pending();

        let summary = tally.to_public();
        assert_eq!(summary.read_failures_by_file.get("io"), Some(&1));
    }

    /// Per-file-kind tally: a fully populated synthetic /proc
    /// (every reader succeeds) lands an empty map and zero
    /// `read_failures`. Pins the "absent key = zero" contract.
    #[test]
    fn parse_summary_clean_proc_yields_empty_map() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 5070;
        let tid: i32 = 5071;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let _ = capture_thread_at_with_tally(
            proc_tmp.path(),
            tgid,
            tid,
            "p",
            "live",
            false,
            &mut tally_opt,
        );
        tally_opt.as_mut().unwrap().commit_pending();

        let summary = tally.to_public();
        assert_eq!(summary.tids_walked, 1);
        assert_eq!(summary.read_failures, 0);
        assert!(
            summary.read_failures_by_file.is_empty(),
            "clean procfs must yield an empty map, got {:?}",
            summary.read_failures_by_file,
        );
        assert!(summary.dominant_read_failure.is_none());
        assert!(!summary.kernel_config_dominant);
    }

    /// Ghost filter discipline (T28.2): a tid that exits between
    /// readdir and the per-file reads (every read fails with
    /// ENOENT, comm is empty, ghost filter rejects the tid) must
    /// NOT contribute to the parse-summary tally. Otherwise a
    /// busy host with mid-capture exits would inflate
    /// `read_failures` with bumps that correspond to threads the
    /// snapshot doesn't even contain.
    #[test]
    fn parse_summary_excludes_ghost_filtered_tids() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 5080;
        let tid: i32 = 5081;
        // Stage only the empty task directory (no comm, no stat,
        // no other files) so every read fails AND the ghost filter
        // fires (empty comm + zero start_time).
        let task_dir = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string());
        std::fs::create_dir_all(&task_dir).unwrap();

        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let t =
            capture_thread_at_with_tally(proc_tmp.path(), tgid, tid, "", "", false, &mut tally_opt);
        // Ghost filter: empty comm + zero start_time → discard.
        if t.comm.is_empty() && t.start_time_clock_ticks == 0 {
            tally_opt.as_mut().unwrap().discard_pending();
        } else {
            tally_opt.as_mut().unwrap().commit_pending();
        }

        let summary = tally.to_public();
        assert_eq!(
            summary.read_failures, 0,
            "ghost-filtered tid must NOT contribute to read_failures; \
             got {} failures (the discard_pending unwind is broken)",
            summary.read_failures,
        );
        assert!(summary.read_failures_by_file.is_empty());
        // tids_walked still incremented — the tid was attempted.
        assert_eq!(summary.tids_walked, 1);
    }

    /// Serde round-trip: a populated `HostStateParseSummary`
    /// preserves every field through JSON.
    #[test]
    fn parse_summary_serde_round_trip() {
        let mut by_file = BTreeMap::new();
        by_file.insert("schedstat".to_string(), 100);
        by_file.insert("io".to_string(), 50);
        let summary = HostStateParseSummary {
            tids_walked: 1000,
            read_failures: 150,
            read_failures_by_file: by_file,
            dominant_read_failure: Some("schedstat".to_string()),
            kernel_config_dominant: true,
            negative_dotted_values: 7,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: HostStateParseSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tids_walked, 1000);
        assert_eq!(back.read_failures, 150);
        assert_eq!(back.read_failures_by_file.get("schedstat"), Some(&100));
        assert_eq!(back.read_failures_by_file.get("io"), Some(&50));
        assert_eq!(back.dominant_read_failure.as_deref(), Some("schedstat"));
        assert!(back.kernel_config_dominant);
        assert_eq!(
            back.negative_dotted_values, 7,
            "negative_dotted_values surfaces in the public surface \
             and round-trips through JSON",
        );
    }

    /// `dominant_read_failure` picks the file kind with the most
    /// failures. Ties resolve REVERSE-alphabetically (mirrors the
    /// probe-summary comparator) — alphabetically-EARLIER tag
    /// wins.
    #[test]
    fn parse_summary_dominant_picks_max_file_kind() {
        let mut tally = ParseTally::default();
        // schedstat: 10 failures, io: 5, status: 5. schedstat wins.
        for _ in 0..10 {
            tally.record_failure("schedstat");
        }
        for _ in 0..5 {
            tally.record_failure("io");
        }
        for _ in 0..5 {
            tally.record_failure("status");
        }
        tally.commit_pending();
        let summary = tally.to_public();
        assert_eq!(summary.dominant_read_failure.as_deref(), Some("schedstat"));

        // Tie between io and status (same count) — io wins (earlier
        // alphabetical, matches the reverse-alphabetical comparator).
        let mut tally2 = ParseTally::default();
        for _ in 0..3 {
            tally2.record_failure("io");
        }
        for _ in 0..3 {
            tally2.record_failure("status");
        }
        tally2.commit_pending();
        let summary2 = tally2.to_public();
        assert_eq!(
            summary2.dominant_read_failure.as_deref(),
            Some("io"),
            "tie must resolve to alphabetically-earlier tag — \
             `io` beats `status`",
        );
    }

    /// `kernel_config_hint` returns Some(_) when ≥ 50% of failures
    /// land in `schedstat`/`io`. Pins the gate equality at the
    /// boundary.
    #[test]
    fn parse_summary_kernel_config_hint_gate() {
        // 50/50 split: 5 schedstat + 5 status. Kconfig share = 50%.
        let mut tally = ParseTally::default();
        for _ in 0..5 {
            tally.record_failure("schedstat");
        }
        for _ in 0..5 {
            tally.record_failure("status");
        }
        tally.commit_pending();
        let summary = tally.to_public();
        assert!(
            summary.kernel_config_dominant,
            "50% kconfig share must hit the gate (>= 50% boundary inclusive)",
        );
        assert!(summary.kernel_config_hint().is_some());

        // Below threshold: 1 schedstat, 9 status. Kconfig share 10%.
        let mut tally2 = ParseTally::default();
        tally2.record_failure("schedstat");
        for _ in 0..9 {
            tally2.record_failure("status");
        }
        tally2.commit_pending();
        let summary2 = tally2.to_public();
        assert!(!summary2.kernel_config_dominant);
        assert!(summary2.kernel_config_hint().is_none());

        // Zero failures: kconfig_dominant must be false (no failures
        // to dominate), hint is None.
        let summary3 = ParseTally::default().to_public();
        assert!(!summary3.kernel_config_dominant);
        assert!(summary3.kernel_config_hint().is_none());
    }

    /// `dominant_read_failure` is None when zero failures landed,
    /// even though the tally was constructed.
    #[test]
    fn parse_summary_dominant_none_when_zero_failures() {
        let summary = ParseTally::default().to_public();
        assert_eq!(summary.read_failures, 0);
        assert!(summary.dominant_read_failure.is_none());
    }

    /// `capture_with(_, _, false)` skips the production gate so
    /// `parse_summary` stays `None` on the assembled snapshot —
    /// mirrors the `probe_summary` discipline. Synthetic-tree
    /// tests must not see a populated parse summary.
    #[test]
    fn capture_with_synthetic_tree_yields_no_parse_summary() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 5090;
        let tid: i32 = 5091;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert!(
            snap.parse_summary.is_none(),
            "use_syscall_affinity=false must skip parse_summary; \
             got Some — production-gate discipline is broken",
        );
    }

    // ------------------------------------------------------------
    // T43 — Additional capture-pipeline error-path tests
    // ------------------------------------------------------------

    /// Phase-1 loadavg missing: capture_with must not panic when
    /// the parallelism-clamp `proc_root/loadavg` read fails. The
    /// reader's `.ok().and_then(...).unwrap_or(0.0)` chain folds
    /// the missing-file branch into the 0.0 default, so the
    /// headroom calculation continues to clamp at
    /// `[1, num_cpus/2 + 1]`. Pins the missing-loadavg branch.
    #[test]
    fn capture_with_phase1_loadavg_missing_does_not_panic() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        // No loadavg file. iter_tgids_at returns Vec::new() so the
        // probe-attach loop iterates zero times — but the clamp
        // computation runs unconditionally inside the
        // use_syscall_affinity=true branch, exercising the
        // missing-file path.
        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
        assert!(
            snap.threads.is_empty(),
            "missing loadavg + empty proc_root → empty snapshot, \
             got {} threads",
            snap.threads.len(),
        );
    }

    /// Phase-1 loadavg malformed: a non-float first token must
    /// fold into the 0.0 default via the `.parse::<f64>().ok()`
    /// step. Pins that a hostile `proc_root/loadavg` cannot crash
    /// the parallelism-clamp computation.
    #[test]
    fn capture_with_phase1_loadavg_malformed_does_not_panic() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(proc_tmp.path().join("loadavg"), "not_a_number\n").unwrap();
        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
        assert!(
            snap.threads.is_empty(),
            "malformed loadavg → 0.0 default, empty proc_root → empty \
             snapshot; got {} threads",
            snap.threads.len(),
        );
    }

    /// Non-UTF-8 bytes in `comm`: `fs::read_to_string` returns Err
    /// on invalid UTF-8, so [`read_thread_comm_at`] yields None
    /// and the caller defaults to "". With `start_time` non-zero
    /// (intact `stat`), the ghost filter does NOT fire and the
    /// thread lands with empty comm.
    #[test]
    fn capture_with_non_utf8_comm_treated_as_absent() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 6161;
        let tid: i32 = 6162;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // Overwrite tid/comm with non-UTF-8 bytes (lone 0xFF, then
        // 0xFE — never valid UTF-8 lead bytes).
        let comm_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("comm");
        std::fs::write(&comm_path, [0xFF, 0xFE]).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(
            snap.threads.len(),
            1,
            "non-UTF-8 comm folds to empty; ghost filter does NOT \
             fire because start_time is intact; thread still lands. \
             got {} threads",
            snap.threads.len(),
        );
        assert_eq!(
            snap.threads[0].comm, "",
            "non-UTF-8 comm must collapse to empty (read_to_string \
             returns Err on invalid UTF-8)",
        );
        assert_ne!(
            snap.threads[0].start_time_clock_ticks, 0,
            "start_time must be intact for the ghost filter NOT to fire",
        );
    }

    /// Cgroup path traversal: a `0::/../escape` payload in the
    /// per-tid cgroup file lands in `ThreadState.cgroup` verbatim
    /// (no sanitization at parse time), and the cgroup_stats
    /// enrichment loop calls `read_cgroup_stats_at` with the same
    /// string. The current behaviour bounds the read inside the
    /// configured `cgroup_root` via `Path::join` — which DOES NOT
    /// reject `..` components. Pin that the path-traversal string
    /// round-trips through the snapshot but does not surface
    /// out-of-tree cgroup data: the stats land at the all-zero
    /// default because no matching cgroup directory exists under
    /// `cgroup_root`.
    #[test]
    fn capture_with_cgroup_path_traversal_yields_zero_stats() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 6262;
        let tid: i32 = 6263;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // Overwrite cgroup with a traversal string.
        let cgroup_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("cgroup");
        std::fs::write(&cgroup_path, "0::/../escape\n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        assert_eq!(
            snap.threads[0].cgroup, "/../escape",
            "traversal string round-trips verbatim through ThreadState.cgroup",
        );
        let stats = snap
            .cgroup_stats
            .get("/../escape")
            .expect("non-empty cgroup string must seed the stats map");
        assert_eq!(
            stats.cpu.usage_usec, 0,
            "no matching cgroup dir under cgroup_root → all-zero stats; \
             a traversal that escaped the cgroup_root would have \
             non-zero values from the parent directory",
        );
    }

    /// Empty `Cpus_allowed_list:` value: `parse_cpu_list("")`
    /// returns None at the empty-input guard, so `cpu_affinity`
    /// lands as the empty Vec. Same observable effect as a
    /// malformed range (G8) but pins the empty-string branch
    /// distinctly.
    #[test]
    fn capture_with_empty_cpus_allowed_yields_empty_affinity() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 6363;
        let tid: i32 = 6364;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        let status_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("status");
        let status = "Cpus_allowed_list:\t\n\
             voluntary_ctxt_switches:\t1\n\
             nonvoluntary_ctxt_switches:\t1\n";
        std::fs::write(&status_path, status).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(snap.threads.len(), 1);
        let t = &snap.threads[0];
        assert!(
            t.cpu_affinity.0.is_empty(),
            "empty Cpus_allowed_list value → parse_cpu_list returns \
             None at the empty-input guard → cpu_affinity empty; \
             got {} elements",
            t.cpu_affinity.0.len(),
        );
        assert_eq!(
            t.voluntary_csw,
            MonotonicCount(1),
            "empty cpulist must not break csw parsing on the same \
             status file",
        );
    }

    /// Ghost filter AND-semantics: an empty `comm` paired with a
    /// NON-zero `start_time_clock_ticks` does NOT fire the filter.
    /// The clause requires BOTH conditions (see
    /// `t.comm.is_empty() && t.start_time_clock_ticks == 0`). Pins
    /// the AND so a future refactor that flipped to OR would
    /// surface here rather than hiding legitimate threads with
    /// transient empty comms.
    #[test]
    fn capture_with_empty_comm_nonzero_start_time_keeps_thread() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 6464;
        let tid: i32 = 6465;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // Overwrite comm with whitespace so read_thread_comm_at
        // returns None → comm defaults to "". start_time stays
        // intact at 555_555 (the value stage_synthetic_proc writes).
        let comm_path = proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("comm");
        std::fs::write(&comm_path, "   \n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(
            snap.threads.len(),
            1,
            "empty comm + nonzero start_time MUST NOT fire ghost filter \
             (AND-semantics requires both empty); got {} threads",
            snap.threads.len(),
        );
        let t = &snap.threads[0];
        assert_eq!(t.comm, "", "empty-comm thread surfaces with empty comm");
        assert_ne!(
            t.start_time_clock_ticks, 0,
            "start_time must be non-zero so the AND-clause has a `false` half",
        );
    }

    // ------------------------------------------------------------
    // T45 — Additional parse_summary + capture-pipeline coverage
    // ------------------------------------------------------------

    /// W2: every tid is ghost-filtered. With N empty task dirs the
    /// ghost filter rejects every tid, so each tid's pending failure
    /// bumps unwind via `discard_pending`. `tids_walked` is bumped
    /// at the call site BEFORE the discard, so it still reads N.
    /// `read_failures` lands at zero (every bump unwound), the per-
    /// file map is empty, and `dominant_read_failure` is None. Pins
    /// the "tids_walked counts attempts; failure tallies count only
    /// committed bumps" split end-to-end through `capture_with`.
    #[test]
    fn parse_summary_all_ghosts_yields_nonzero_tids_walked_zero_failures() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7070;
        let n: u64 = 4;
        // Stage one tgid with N empty task dirs (no comm, no stat,
        // no other files). Every read fails; ghost filter fires for
        // every tid; every pending tally is unwound.
        let tgid_dir = proc_tmp.path().join(tgid.to_string());
        for k in 0..n {
            let tid = (tgid as u64 + 1 + k) as i32;
            std::fs::create_dir_all(tgid_dir.join("task").join(tid.to_string())).unwrap();
        }
        // Stage `loadavg` so the parallelism-clamp read in phase 1
        // resolves cleanly (the missing-file fallback is exercised
        // by capture_with_phase1_loadavg_missing_does_not_panic).
        std::fs::write(proc_tmp.path().join("loadavg"), "0.10 0.05 0.01 1/1 1\n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
        assert!(
            snap.threads.is_empty(),
            "every tid is ghost-filtered → threads must be empty, got {}",
            snap.threads.len(),
        );
        let summary = snap
            .parse_summary
            .expect("use_syscall_affinity=true must populate parse_summary");
        assert_eq!(
            summary.tids_walked, n,
            "tids_walked counts every walk attempt, not committed reads — \
             got {}, want {n}",
            summary.tids_walked,
        );
        assert_eq!(
            summary.read_failures, 0,
            "ghost-filtered tids' failures unwind via discard_pending — \
             got {} failures, want 0",
            summary.read_failures,
        );
        assert!(
            summary.read_failures_by_file.is_empty(),
            "no failure bucket survives the ghost-filter unwind, got {:?}",
            summary.read_failures_by_file,
        );
        assert!(
            summary.dominant_read_failure.is_none(),
            "zero failures → dominant_read_failure is None, got {:?}",
            summary.dominant_read_failure,
        );
        assert!(
            !summary.kernel_config_dominant,
            "zero failures → kernel_config_dominant is false, got true",
        );
    }

    /// W3: pin which file-kind tokens count as kernel-config-gated.
    /// `kernel_config_dominates` filters on `matches!(t, "schedstat"
    /// | "io")`. Iterate every recognised kebab token solo (one
    /// failure of that kind, no others) and assert the gate flips
    /// the way the implementation says it should — schedstat/io
    /// land 100% kconfig and the gate fires; stat/status/sched/cgroup
    /// land 0% kconfig and the gate stays false. A future refactor
    /// that added or removed a token from the kconfig set without
    /// updating the docs would surface here.
    #[test]
    fn parse_summary_kernel_config_token_list_pinned() {
        let kconfig_tokens: &[&'static str] = &["schedstat", "io"];
        for tag in kconfig_tokens {
            let mut tally = ParseTally::default();
            tally.record_failure(tag);
            tally.commit_pending();
            let summary = tally.to_public();
            assert!(
                summary.kernel_config_dominant,
                "solo `{tag}` failure must flip kernel_config_dominant true \
                 (kconfig share = 100%); got false — token dropped from the \
                 kconfig set",
            );
        }

        let non_kconfig_tokens: &[&'static str] = &["stat", "status", "sched", "cgroup"];
        for tag in non_kconfig_tokens {
            let mut tally = ParseTally::default();
            tally.record_failure(tag);
            tally.commit_pending();
            let summary = tally.to_public();
            assert!(
                !summary.kernel_config_dominant,
                "solo `{tag}` failure must keep kernel_config_dominant false \
                 (kconfig share = 0%); got true — token incorrectly added to \
                 the kconfig set",
            );
        }
    }

    /// W5: tally aggregates across multiple tids. Stage 2 tids
    /// where each fails a different file (one missing io, one
    /// missing schedstat). Both bumps must commit (neither tid is
    /// ghost-filtered) and the per-file map carries one entry per
    /// failure kind with count 1, total `read_failures` = 2.
    #[test]
    fn parse_summary_aggregates_across_multiple_tids() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7080;
        let tid_a: i32 = 7081;
        let tid_b: i32 = 7082;
        stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid_a);
        // Second tid under the same tgid: write a fresh task dir.
        let tgid_dir = proc_tmp.path().join(tgid.to_string());
        let task_b = tgid_dir.join("task").join(tid_b.to_string());
        std::fs::create_dir_all(&task_b).unwrap();
        std::fs::write(task_b.join("comm"), "live\n").unwrap();
        let stat_line = format!(
            "{tid_b} (live) R 1 2 3 4 5 6 7 0 8 0 10 11 12 13 14 0 1 0 \
             555555 100 200 300 400 500 600 700 800 900 1000 1100 \
             1200 1300 1400 1500 1600 1700 1800 0\n"
        );
        std::fs::write(task_b.join("stat"), stat_line).unwrap();
        std::fs::write(task_b.join("schedstat"), "0 0 0\n").unwrap();
        std::fs::write(
            task_b.join("status"),
            "voluntary_ctxt_switches:\t0\n\
             nonvoluntary_ctxt_switches:\t0\n",
        )
        .unwrap();
        std::fs::write(task_b.join("io"), "rchar: 0\n").unwrap();
        std::fs::write(task_b.join("sched"), "").unwrap();
        std::fs::write(task_b.join("cgroup"), "0::/\n").unwrap();

        // tid_a: delete io. tid_b: delete schedstat.
        std::fs::remove_file(tgid_dir.join("task").join(tid_a.to_string()).join("io")).unwrap();
        std::fs::remove_file(task_b.join("schedstat")).unwrap();

        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        for tid in [tid_a, tid_b] {
            tally_opt.as_mut().unwrap().tids_walked += 1;
            let _ = capture_thread_at_with_tally(
                proc_tmp.path(),
                tgid,
                tid,
                "p",
                "live",
                false,
                &mut tally_opt,
            );
            tally_opt.as_mut().unwrap().commit_pending();
        }
        let summary = tally.to_public();
        assert_eq!(summary.tids_walked, 2);
        assert_eq!(
            summary.read_failures, 2,
            "two tids, one failure each → 2 total; got {}",
            summary.read_failures,
        );
        assert_eq!(
            summary.read_failures_by_file.get("io"),
            Some(&1),
            "tid_a missing io → io bucket = 1; got {:?}",
            summary.read_failures_by_file.get("io"),
        );
        assert_eq!(
            summary.read_failures_by_file.get("schedstat"),
            Some(&1),
            "tid_b missing schedstat → schedstat bucket = 1; got {:?}",
            summary.read_failures_by_file.get("schedstat"),
        );
    }

    /// W7: deleting cgroup lands a `"cgroup"` failure. Mirrors the
    /// schedstat/io single-failure tests so the cgroup-read tally
    /// path is exercised explicitly — `read_cgroup_at_with_tally`
    /// is the only producer of the `"cgroup"` tag and a future
    /// refactor that bypassed the tally would surface here.
    #[test]
    fn parse_summary_records_cgroup_failure() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7090;
        let tid: i32 = 7091;
        stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
        std::fs::remove_file(
            proc_tmp
                .path()
                .join(tgid.to_string())
                .join("task")
                .join(tid.to_string())
                .join("cgroup"),
        )
        .unwrap();

        let mut tally = ParseTally::default();
        let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let _ = capture_thread_at_with_tally(
            proc_tmp.path(),
            tgid,
            tid,
            "p",
            "live",
            false,
            &mut tally_opt,
        );
        tally_opt.as_mut().unwrap().commit_pending();

        let summary = tally.to_public();
        assert_eq!(
            summary.read_failures_by_file.get("cgroup"),
            Some(&1),
            "missing cgroup file → cgroup bucket = 1; got {:?}",
            summary.read_failures_by_file.get("cgroup"),
        );
    }

    /// W6: the production gate (`use_syscall_affinity=true`)
    /// populates `parse_summary` end-to-end. Mirror of
    /// `capture_with_synthetic_tree_yields_no_parse_summary` but
    /// with the gate flipped — pins that the production-path
    /// assignment is wired through.
    #[test]
    fn capture_with_production_gate_populates_parse_summary() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7100;
        let tid: i32 = 7101;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // loadavg lets the parallelism-clamp read resolve cleanly.
        std::fs::write(proc_tmp.path().join("loadavg"), "0.10 0.05 0.01 1/1 1\n").unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
        assert!(
            snap.parse_summary.is_some(),
            "use_syscall_affinity=true must populate parse_summary on \
             the assembled snapshot — production-gate wiring is broken",
        );
    }

    /// X2: non-UTF-8 bytes in `<tgid>/comm` (the pcomm path).
    /// `read_process_comm_at` calls `fs::read_to_string`, which
    /// returns Err on invalid UTF-8; `.ok()?` propagates None and
    /// the caller defaults `pcomm` to "" via `.unwrap_or_default()`.
    /// Pin that capture does not panic and the per-thread `pcomm`
    /// surfaces empty. Mirror of
    /// `capture_with_non_utf8_comm_treated_as_absent` but for the
    /// process-level (`<tgid>/comm`) read rather than the per-tid
    /// (`<tgid>/task/<tid>/comm`) read.
    #[test]
    fn capture_with_non_utf8_pcomm_treated_as_absent() {
        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        let tgid: i32 = 7110;
        let tid: i32 = 7111;
        stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
        // Overwrite the pcomm path (`<tgid>/comm`) with non-UTF-8
        // lead bytes (0xFF and 0xFE — never valid UTF-8 starts).
        let pcomm_path = proc_tmp.path().join(tgid.to_string()).join("comm");
        std::fs::write(&pcomm_path, [0xFF, 0xFE]).unwrap();

        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
        assert_eq!(
            snap.threads.len(),
            1,
            "non-UTF-8 pcomm must not break the capture — the thread still \
             lands; got {} threads",
            snap.threads.len(),
        );
        assert_eq!(
            snap.threads[0].pcomm, "",
            "non-UTF-8 pcomm collapses to empty (read_to_string returns Err \
             on invalid UTF-8 and unwrap_or_default → \"\")",
        );
    }

    /// Y1: panic-injection harness for rayon worker panics.
    ///
    /// `attach_jemalloc_at` reads `/proc/<pid>/exe`, opens the ELF
    /// file, and walks DWARF — every step can panic under fd
    /// exhaustion or OOM. Without the `catch_unwind` guard in
    /// `capture_with`'s phase-1 worker closure, a single panicking
    /// tgid would propagate through `pool.install` and tear down
    /// the whole snapshot. No realistic synthetic input can force
    /// the underlying readers to panic, so this test installs an
    /// explicit injection seam (`PANIC_INJECT_TGID`) that fires
    /// inside `attach_probe_for_tgid_at` for the matching tgid and
    /// drives the rayon worker into a panic. The capture pipeline
    /// must absorb it, surface it as a `worker-panic` attach tag,
    /// and still walk the surviving tgid's threads.
    ///
    /// Asserts:
    ///   - `capture_with(.., true)` returns rather than unwinding,
    ///   - the surviving tgid's thread lands in the snapshot,
    ///   - `probe_summary.failed >= 1` (the panic is counted),
    ///   - `dominant_failure == Some("worker-panic")` (the new tag
    ///     surfaces in the curated public surface).
    #[test]
    fn capture_with_rayon_worker_panic_is_caught_and_surfaced() {
        // Serialize panic-hook test against any future test that
        // also installs a custom hook, so the silenced hook below
        // is not clobbered. `Mutex<()>` is enough — the lock is
        // only held for the duration of the capture call.
        static PANIC_INJECT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = PANIC_INJECT_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let proc_tmp = tempfile::TempDir::new().unwrap();
        let cgroup_tmp = tempfile::TempDir::new().unwrap();
        let sys_tmp = tempfile::TempDir::new().unwrap();
        // Required by the parallelism-clamp logic in capture_with.
        std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

        // Two tgids: the survivor (clean attach attempt → fails
        // benignly with `readlink-failure` because the synthetic
        // /proc has no `<tgid>/exe` symlink — the dominant-tag
        // filter suppresses this, leaving worker-panic as the
        // sole dominant candidate) and the panic target (the
        // sentinel tgid the seam matches against). Sentinel value
        // 99001 is intentionally outside any other test's range so
        // a parallel run cannot cross-fire.
        let survivor_tgid: i32 = 99000;
        let survivor_tid: i32 = 99002;
        let panic_tgid: i32 = 99001;
        let panic_tid: i32 = 99003;
        stage_synthetic_proc(
            proc_tmp.path(),
            survivor_tgid,
            survivor_tid,
            "ok-pcomm",
            "ok-comm",
        );
        stage_synthetic_proc(
            proc_tmp.path(),
            panic_tgid,
            panic_tid,
            "panic-pcomm",
            "panic-comm",
        );

        // Silence the default panic hook: rayon's worker panic
        // would otherwise dump a stack trace to stderr and pollute
        // the test output. Restore the hook before the lock
        // releases so subsequent tests see the real hook again.
        let saved_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_info| {}));

        // Arm the seam, run capture, then disarm BEFORE restoring
        // the hook so a panic during disarm (none expected) still
        // hits the silenced hook rather than the real one.
        PANIC_INJECT_TGID.store(panic_tgid, std::sync::atomic::Ordering::Release);
        let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
        PANIC_INJECT_TGID.store(0, std::sync::atomic::Ordering::Release);

        std::panic::set_hook(saved_hook);

        // Survivor thread must land. The panicking tgid's threads
        // are walked too (phase 2 still iterates every tgid in
        // `tgids`), so total threads is 2.
        assert_eq!(
            snap.threads.len(),
            2,
            "rayon worker panic must not block phase 2 — both staged tgids \
             walk their threads; got {} threads",
            snap.threads.len(),
        );

        let summary = snap
            .probe_summary
            .expect("use_syscall_affinity=true must populate probe_summary");
        assert!(
            summary.failed >= 1,
            "worker-panic must count as a failure; got failed={}",
            summary.failed,
        );
        assert_eq!(
            summary.dominant_failure.as_deref(),
            Some("worker-panic"),
            "worker-panic is the only ACTIONABLE failure tag in this \
             scenario. The survivor's synthetic /proc has no `exe` \
             symlink, so attach short-circuits with `readlink-failure` \
             — the dominant-tag comparator filters that benign tag out \
             (same `matches!` arm `record_attach_outcome` uses to log it \
             at debug rather than warn), leaving worker-panic as the \
             sole candidate. A regression that demoted worker-panic \
             out of the dominant set, or that miscounted the panic, \
             would fail here. Got {:?}",
            summary.dominant_failure,
        );
    }
}
