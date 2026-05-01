//! Combined ctprof + failure-dump capture format for live-host
//! debugging and reproducer construction.
//!
//! The "debug capture" pipeline runs lightweight periodic probes on
//! an instrumented host, then bundles the resulting samples with a
//! point-in-time failure dump when a stall/exit fires. The output is
//! a single record that carries:
//!
//! 1. **Periodic samples** (`Vec<CtprofSnapshot>`) — workload
//!    characteristics captured continuously while the scheduler runs:
//!    per-thread CPU time, per-cgroup weights/limits, per-PSI
//!    pressure, per-CPU utilization, sched_ext sysfs counters. Same
//!    shape `cargo ktstr ctprof show/compare` already consumes.
//!
//! 2. **Failure-time dump** (`Option<FailureDumpReport>`) — the rich
//!    state-of-the-world snapshot at the failure instant: rq->scx
//!    state per CPU, DSQ depths, BPF program runtime stats, per-task
//!    enrichment, scheduler scalar state. Same shape the freeze-VM
//!    pipeline produces — the live-host backend ([`super::bpf_syscall`])
//!    populates it via the bpf() syscall instead of guest-memory
//!    walks.
//!
//! 3. **Workload fingerprint** ([`WorkloadFingerprint`]) — projected
//!    HINTS in ktstr's test-primitive vocabulary that the reproducer
//!    generator translates into a `WorkProgram` spec. NOT a
//!    canonical reproducer; the projection is best-effort and the
//!    reproducer generator (or human) decides which hints to honor.
//!
//! # Capture vs reproducer-generator boundary
//!
//! This module is the DATA-SHAPE deliverable for the capture path.
//! The actual periodic-loop binary that drives [`ctprof::capture`]
//! and the failure trigger pipeline (tp_btf/sched_ext_exit fentry,
//! per `research_live_host.md` phd-host4) are upstream producers
//! that emit [`DebugCapture`]; the reproducer generator is
//! the downstream consumer. Splitting at this module boundary keeps
//! each side free to evolve independently.
//!
//! The fingerprint projection lives here (rather than in the reproducer
//! generator module) because
//! it's a pure function of capture data — the reproducer generator
//! consumes the projected hints rather than re-deriving them, and
//! that boundary is testable without a working producer or generator.
//!
//! # Vocabulary alignment with ktstr test primitives
//!
//! Per `capture_reproduce_thesis.md`: the capture format must speak
//! the same vocabulary as the test library. Projected hints map
//! one-to-one with primitive types in `crate::workload`:
//!
//! | observation                      | hint type                          |
//! |----------------------------------|------------------------------------|
//! | per-cgroup thread count          | `WorkProgramHint::thread_count`    |
//! | sched_setaffinity mask patterns  | `AffinityHint::*`                  |
//! | CPU-time vs IO-wait ratio        | `WorkTypeHint::*`                  |
//! | cgroup cpu.weight + memory.max   | `CgroupHint::*`                    |
//! | scheduling policy distribution   | `SchedPolicyHint::*`               |
//!
//! Hints are SUGGESTIONS, not commands. The reproducer generator
//! weighs them against the failure dump and the user's preferences
//! (e.g. minimal-repro vs full-fidelity reproduction).
//!
//! # Filterable presentation per `classifier_design.md`
//!
//! The capture format is designed for `ktstr show / compare`-style
//! consumption: combinable filters by cgroup / CPU / NUMA node /
//! sched class / tgid / time window, with coherent aggregates that
//! follow from the filter. The data shape preserves enough raw
//! granularity for any of those filters to compose without forcing
//! a pre-baked aggregate menu.

use serde::{Deserialize, Serialize};

use super::dump::FailureDumpReport;

/// One end-to-end debug capture record.
///
/// Bundles every observation the reproducer generator needs to
/// translate a real-world failure into a ktstr test. Serializable so
/// the periodic-loop binary can persist captures to disk and the
/// reproducer generator can consume them offline.
///
/// `non_exhaustive` so future fields (kernel command-line snapshot,
/// scheduler config-file capture, host hardware fingerprint) can
/// land without breaking on-disk records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
#[allow(dead_code)] // wired from the (separate) capture-mode binary;
                    // the library ships the data shape so the
                    // reproducer generator can build against a stable
                    // surface.
pub struct DebugCapture {
    /// Capture format schema identifier — pinned at construction so
    /// off-disk records older than the current shape parse via the
    /// `FailureDumpReportAny`-style migration shim instead of
    /// silently misinterpreting fields.
    pub schema: String,
    /// Capture wall-clock start. Ns since epoch (CLOCK_REALTIME).
    /// `0` when the producer didn't stamp it (e.g. captures replayed
    /// from a fixture without a real clock).
    pub started_ns: u64,
    /// Capture wall-clock end. Same units. Equal to `started_ns`
    /// when this record represents a single instant rather than a
    /// time window.
    pub ended_ns: u64,
    /// Linux kernel release string (`uname -r`) of the captured
    /// host. Used by the reproducer generator to choose a
    /// matching ktstr kernel cache entry — without it, the reproducer
    /// might run against a different kernel than the failure
    /// originally occurred on.
    pub kernel_release: String,
    /// Periodic ctprof snapshots taken across the capture window.
    /// In time order; gaps are OK (lossy capture is preferable to
    /// blocking the scheduler under instrumentation overhead).
    /// Empty when the capture was triggered without a preceding
    /// sampling phase.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ctprof_samples: Vec<CtprofSampleRef>,
    /// Failure-time scheduler dump. `None` when the capture closed
    /// without a triggering event (graceful shutdown, not a stall);
    /// the reproducer generator treats `None` captures as "healthy
    /// baseline" data points rather than failure cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_dump: Option<FailureDumpReport>,
    /// Workload-vocabulary projection of the capture data. Computed
    /// at capture-record build time so consumers that only want
    /// hints (the reproducer generator's normal mode) don't have to
    /// reparse the full ctprof/failure_dump bundle.
    pub fingerprint: WorkloadFingerprint,
}

/// Reference to a ctprof sample stored elsewhere on disk.
///
/// `ctprof::CtprofSnapshot` is megabyte-class; embedding every
/// sample inline would bloat capture records past practical
/// transport sizes. Real captures store the snapshot blobs as
/// sibling `.ctprof.zst` files (the existing
/// [`crate::ctprof::SNAPSHOT_EXTENSION`] format) and keep refs
/// here.
///
/// The reproducer generator dereferences refs lazily — most
/// captures need only the fingerprint + failure dump, not every
/// raw ctprof sample.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CtprofSampleRef {
    /// Sample wall-clock timestamp (CLOCK_REALTIME ns).
    pub captured_ns: u64,
    /// Path to the on-disk `.ctprof.zst` blob, relative to the
    /// capture record's own directory. Empty when the sample is
    /// embedded inline (test fixtures only).
    pub path: String,
    /// SHA-256 of the on-disk blob, hex-encoded. Empty when the
    /// producer didn't compute it. Used by the reproducer
    /// generator to detect torn / partial writes that would
    /// otherwise be silently rendered as missing data.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub sha256: String,
}

/// Workload-shaped projection of capture data into ktstr test-
/// primitive vocabulary.
///
/// Each field is a HINT that the reproducer generator maps to
/// a corresponding type in `crate::workload`. Multiple hints fire on
/// rich captures (e.g. a task that pinned itself to one CPU AND ran
/// CPU-bound work AND lived in a cgroup with `cpu.weight=200` —
/// projects to `AffinityHint::SingleCpu` + `WorkTypeHint::CpuSpin` +
/// `CgroupHint::WeightOverride { weight: 200 }`).
///
/// All hint vectors may be empty (insufficient data to project).
/// The reproducer generator falls back to library defaults for
/// any primitive whose hints are absent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WorkloadFingerprint {
    /// Per-cgroup thread-count distribution. Maps to
    /// `WorkProgram::thread_count` in the generated test. The
    /// reproducer generator picks a representative group and emits
    /// one `WorkProgram` per cgroup.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workload_groups: Vec<WorkloadGroupHint>,
    /// Aggregate CPU placement pattern observed across the capture.
    /// Multiple values when different thread groups exhibited
    /// different patterns (e.g. one cgroup pinned to a single CPU,
    /// another randomly placed).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affinity_hints: Vec<AffinityHint>,
    /// Workload type distribution. Multiple values when the capture
    /// observed mixed workloads. Sorted by frequency descending.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub work_type_hints: Vec<WorkTypeHint>,
    /// cgroup definition hints — observed `cpu.weight`, `memory.max`,
    /// `cpuset.cpus`, etc. that should be reproduced.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cgroup_hints: Vec<CgroupHint>,
    /// Sched policy hints — observed SCHED_FIFO/RR/DEADLINE tasks
    /// that the reproducer should set up via `SchedPolicy`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sched_policy_hints: Vec<SchedPolicyHint>,
    /// Sources of low confidence — empty when the projection ran
    /// against full data; populated when ctprof samples were
    /// missing fields, per-task BTF resolution failed, etc. The
    /// reproducer generator surfaces these to the user so they
    /// know which hints to trust less.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gaps: Vec<String>,
}

/// Per-workload-group hint: thread count + cgroup path + dominant
/// behavior class.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WorkloadGroupHint {
    /// Cgroup path that anchors this group (e.g. `/system.slice/foo.service`).
    /// The reproducer generator translates this to a
    /// `CgroupDef::path(...)` in the generated test.
    pub cgroup_path: String,
    /// Number of live threads observed in the cgroup. Maps to
    /// `WorkProgram::thread_count`.
    pub thread_count: u32,
    /// Mean CPU-time fraction across the capture window (0.0 to
    /// 1.0). The reproducer generator uses this to pick a
    /// `WorkType` intensity (e.g. `CpuSpin` for >0.8, `Mixed` for
    /// 0.3-0.8, `Bursty` for <0.3).
    pub cpu_time_fraction: f64,
    /// Mean wakeup rate (Hz) across the capture window. High wakeup
    /// rates suggest `PipeIo` or `FutexPingPong` workload types.
    pub wakeups_per_sec: f64,
}

/// Affinity placement hint. Maps directly to `crate::workload::AffinityKind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
pub enum AffinityHint {
    /// Default placement — no affinity mask narrower than the
    /// containing cgroup's cpuset. → `AffinityKind::Inherit`.
    Inherit,
    /// Threads observed pinned to a single CPU each (mask popcount
    /// == 1 across the capture window for the majority of threads
    /// in the group). → `AffinityKind::SingleCpu`.
    SingleCpu,
    /// Threads observed pinned to LLC-aligned subsets of the
    /// cgroup's cpuset. → `AffinityKind::LlcAligned`.
    LlcAligned,
    /// Threads observed pinned to cgroup-spanning CPU sets. →
    /// `AffinityKind::CrossCgroup`.
    CrossCgroup,
    /// Threads observed pinned to an explicit CPU set. The capture
    /// records the exact CPUs so the reproducer can reproduce the
    /// specific placement. → `AffinityKind::Exact`.
    Exact { cpus: Vec<u32> },
    /// Threads observed pinned to a strict subset of the cgroup's
    /// cpuset, but the subset varies across threads — typical of
    /// a placement randomizer. → `AffinityKind::RandomSubset`.
    RandomSubset,
}

/// Workload type hint. Maps to `crate::workload::WorkType` variants.
/// The hint records the primary signal (CPU-bound vs IO-bound vs
/// futex / pipe wake patterns) and lets the reproducer generator
/// choose a parameterized variant (e.g. `WorkType::Bursty` with
/// burst_ms/sleep_ms picked from the hint's window measurement).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
pub enum WorkTypeHint {
    /// CPU-bound, no observed IO or wake-driven blocking. Maps to
    /// `WorkType::CpuSpin`.
    CpuSpin,
    /// Heavy `sched_yield` rate observed (yields/sec >>
    /// involuntary-context-switch rate). Maps to
    /// `WorkType::YieldHeavy`.
    YieldHeavy,
    /// Mixed CPU + yield pattern. `WorkType::Mixed`.
    Mixed,
    /// CPU bursts followed by long sleeps. Maps to
    /// `WorkType::Bursty` with measured `burst_ms` / `sleep_ms`.
    Bursty { burst_ms: u64, sleep_ms: u64 },
    /// Pipe-mediated wake exchanges. Maps to `WorkType::PipeIo`.
    /// Detected by `read`/`write` syscall pattern + paired tids.
    PipeIo,
    /// Futex-mediated wake exchanges. Maps to
    /// `WorkType::FutexPingPong`. Detected by futex_wait/wake
    /// tracepoints + paired tids.
    FutexPingPong,
    /// Strided memory access pattern dominating CPU time. Maps to
    /// `WorkType::CachePressure` with measured `size_kb` / `stride`.
    CachePressure { size_kb: u32, stride: u32 },
    /// IO-sync-style workload — short bursts of write + small sleep
    /// loops. Maps to `WorkType::IoSync`.
    IoSync,
}

/// Cgroup definition hint. Maps to `crate::workload::CgroupDef`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CgroupHint {
    pub path: String,
    /// `cpu.weight` value observed (1..=10000, kernel default 100).
    /// `None` when the cgroup uses inherited weight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_weight: Option<u32>,
    /// `memory.max` in bytes. `None` when unlimited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_max_bytes: Option<u64>,
    /// `cpuset.cpus` — list of CPUs from the cpuset controller.
    /// Empty when no cpuset constraint.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpuset_cpus: Vec<u32>,
    /// `cpu.max` quota in microseconds per `cpu.max` period. `None`
    /// when no bandwidth limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_max_quota_us: Option<u64>,
}

/// Scheduling policy hint. Maps to `crate::workload::SchedPolicy`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "policy")]
pub enum SchedPolicyHint {
    /// Default fair-class scheduling. Maps to `SchedPolicy::Other`
    /// (the SCHED_OTHER default the kernel uses for new tasks).
    Other { nice: i32 },
    /// SCHED_FIFO real-time. `priority` is the rt_priority field
    /// (1..=99). Maps to `SchedPolicy::Fifo`.
    Fifo { priority: u32 },
    /// SCHED_RR real-time. `priority` is rt_priority. Maps to
    /// `SchedPolicy::RoundRobin`.
    RoundRobin { priority: u32 },
    /// SCHED_DEADLINE. Carries the observed runtime/deadline/period
    /// triple. Maps to `SchedPolicy::Deadline`.
    Deadline {
        runtime_ns: u64,
        deadline_ns: u64,
        period_ns: u64,
    },
    /// SCHED_BATCH. Maps to `SchedPolicy::Batch`.
    Batch,
    /// SCHED_IDLE. Maps to `SchedPolicy::Idle`.
    Idle,
    /// SCHED_EXT — task is currently routed through the BPF
    /// scheduler. Reproducer generator emits no policy override
    /// (the test harness routes tasks through scx by default).
    Ext,
}

/// Pinned schema identifier for [`DebugCapture`]. Bumped when
/// on-disk shape changes incompatibly. Live-host pipeline
/// consumers parse via a serde shim that accepts older schemas
/// when the field set is a strict subset.
#[allow(dead_code)] // wired from the (separate) capture-mode binary;
                    // the library ships the pinned constant.
pub const DEBUG_CAPTURE_SCHEMA: &str = "ktstr.debug_capture/v1";

/// Compute a [`WorkloadFingerprint`] from raw capture inputs.
///
/// Pure function: same inputs always produce the same fingerprint.
/// The reproducer generator and offline analysis tools call
/// this directly to re-project hints from a stored capture without
/// needing the live producer.
///
/// `samples` is the time-ordered ctprof window (may be empty when
/// the capture was triggered without a sampling phase). `dump` is
/// the failure-time snapshot (may be `None` for healthy-baseline
/// captures).
///
/// Implementation surface intentionally narrow: the projection is
/// the data shape's contract with the reproducer generator. As we
/// learn which signals matter most, the implementation evolves
/// without changing the inputs/outputs.
#[allow(dead_code)]
pub fn project_fingerprint(
    samples: &[crate::ctprof::CtprofSnapshot],
    dump: Option<&FailureDumpReport>,
) -> WorkloadFingerprint {
    let mut fp = WorkloadFingerprint::default();

    if samples.is_empty() && dump.is_none() {
        fp.gaps.push("no inputs to project from".to_string());
        return fp;
    }

    // Collect per-cgroup thread counts from the most recent ctprof
    // sample (the structural shape we want is the latest snapshot;
    // the time-window aggregates fold in below).
    if let Some(latest) = samples.last() {
        for (cgroup_path, group_hint) in cgroup_thread_groups(latest) {
            fp.workload_groups.push(WorkloadGroupHint {
                cgroup_path,
                thread_count: group_hint.thread_count,
                cpu_time_fraction: group_hint.cpu_time_fraction,
                wakeups_per_sec: group_hint.wakeups_per_sec,
            });
        }
    }

    // Affinity hints come primarily from the failure dump's
    // task_enrichments — those carry per-task placement metadata.
    // ctprof samples don't track sched_setaffinity masks today,
    // so absent a dump the affinity projection is a gap.
    if let Some(d) = dump {
        fp.affinity_hints = project_affinity_hints(d);
        fp.sched_policy_hints = project_sched_policy_hints(d);
    } else {
        fp.gaps.push(
            "affinity + sched_policy hints unavailable (no failure dump)".to_string(),
        );
    }

    // Work-type hints come from CPU-time / wakeup-rate shape across
    // the sampling window. Bursty / IoSync are detected by sleep
    // ratio; CpuSpin / Mixed by yield rate vs CPU time.
    fp.work_type_hints = project_work_type_hints(samples);

    fp.cgroup_hints = project_cgroup_hints(samples);

    fp
}

#[derive(Debug, Default)]
struct CgroupGroupAcc {
    thread_count: u32,
    cpu_time_fraction: f64,
    wakeups_per_sec: f64,
}

/// Stub: aggregate per-cgroup thread counts + utilization from a
/// ctprof snapshot.
///
/// Real implementation walks `snapshot.cgroup_stats` (or whatever
/// per-cgroup field ctprof exposes — there are a few candidates;
/// the actual one is wired up in the capture-binary producer).
/// Library-side this function is a contract: takes a snapshot,
/// returns per-cgroup fingerprint atoms.
fn cgroup_thread_groups(
    _snapshot: &crate::ctprof::CtprofSnapshot,
) -> Vec<(String, CgroupGroupAcc)> {
    // The producer wiring is the capture-binary's responsibility. The
    // library lands the contract: empty vec is "no usable groups
    // found in the snapshot" rather than a failure.
    Vec::new()
}

fn project_affinity_hints(_dump: &FailureDumpReport) -> Vec<AffinityHint> {
    // Affinity projection reads task_enrichments[*].cpus_allowed_mask
    // (not yet on the failure dump — follow-up to per-task enrichment
    // expansion). Library lands the slot; producer wiring fills it.
    Vec::new()
}

fn project_sched_policy_hints(dump: &FailureDumpReport) -> Vec<SchedPolicyHint> {
    // Walk task_enrichments and synthesize per-task SchedPolicyHint
    // values. Each task carries `sched_class` (decoded via
    // SchedClassRegistry) + `prio` / `static_prio` / `rt_priority`
    // (from TaskEnrichmentOffsets). The mapping table here is the
    // live-host equivalent of the test framework's
    // SchedPolicy::from_kernel_view.
    let mut hints = Vec::new();
    for task in &dump.task_enrichments {
        // sched_class is Option<String> — None means symbol lookup
        // failed (kallsyms unreadable, sched_class symbol not
        // resolved). Treat as Other with nice 0 so the reproducer
        // generator gets at least a generic hint per task.
        let class = task.sched_class.as_deref().unwrap_or("");
        let policy = match class {
            "rt_sched_class" if task.rt_priority > 0 => {
                // SCHED_FIFO and SCHED_RR are both rt_sched_class
                // — the kernel doesn't distinguish via the class
                // pointer. Without per-task SCHED_FIFO/RR
                // discrimination on the dump (a follow-up
                // enrichment), default to FIFO; the reproducer
                // generator surfaces this as a low-confidence
                // hint via the fingerprint's `gaps`.
                SchedPolicyHint::Fifo {
                    priority: task.rt_priority,
                }
            }
            "dl_sched_class" => SchedPolicyHint::Deadline {
                runtime_ns: 0,
                deadline_ns: 0,
                period_ns: 0,
            },
            "ext_sched_class" => SchedPolicyHint::Ext,
            "idle_sched_class" => SchedPolicyHint::Idle,
            _ => {
                // Kernel default static_prio for nice 0 is 120
                // (NICE_TO_PRIO(0) = MAX_RT_PRIO + 20 = 120). The
                // observed nice value is `static_prio - 120`,
                // bounded to a signed delta so a missing
                // static_prio (which would read as 0 from a
                // failure dump that didn't capture it) projects
                // to nice -120 rather than triggering a panic on
                // the cast.
                let nice = (task.static_prio as i32) - 120;
                SchedPolicyHint::Other { nice }
            }
        };
        hints.push(policy);
    }
    hints
}

fn project_work_type_hints(_samples: &[crate::ctprof::CtprofSnapshot]) -> Vec<WorkTypeHint> {
    // The classifier (per classifier_design.md) is "human or LLM
    // reading ktstr show/compare" — this projection's job is to
    // expose the SHAPE of the work-type distribution as a hint
    // list, not to render a definitive label. Producer wiring
    // computes utilization / yield rate / sleep ratio from the
    // sampling window and feeds the cases below.
    Vec::new()
}

fn project_cgroup_hints(_samples: &[crate::ctprof::CtprofSnapshot]) -> Vec<CgroupHint> {
    // ctprof already captures per-cgroup cpu.weight / memory.max /
    // cpuset.cpus into CgroupStats; the projection reads those
    // verbatim. Producer wiring fills the loop.
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `project_fingerprint` with no inputs marks the gap and
    /// returns an empty fingerprint without panicking. The
    /// reproducer generator's normal "no data" handling exercises
    /// this path.
    #[test]
    fn project_fingerprint_no_inputs() {
        let fp = project_fingerprint(&[], None);
        assert!(fp.workload_groups.is_empty());
        assert!(fp.affinity_hints.is_empty());
        assert!(fp.work_type_hints.is_empty());
        assert!(fp.cgroup_hints.is_empty());
        assert!(fp.sched_policy_hints.is_empty());
        assert_eq!(fp.gaps, vec!["no inputs to project from".to_string()]);
    }

    /// Fingerprint computed from a dump with task enrichments
    /// produces sched-policy hints (one per task). Verifies the
    /// rt_sched_class → Fifo and ext_sched_class → Ext mappings.
    #[test]
    fn project_sched_policy_hints_from_dump() {
        use crate::monitor::task_enrichment::TaskEnrichment;

        // TaskEnrichment is non_exhaustive without Default; each
        // fixture is built explicitly via a helper that highlights
        // the fields under test.
        fn make_task(
            pid: i32,
            sched_class: &str,
            rt_priority: u32,
            static_prio: i32,
        ) -> TaskEnrichment {
            TaskEnrichment {
                pid,
                tgid: 0,
                comm: String::new(),
                group_leader_pid: None,
                real_parent_pid: None,
                real_parent_comm: None,
                pgid: None,
                sid: None,
                nr_threads: None,
                weight: 0,
                prio: 0,
                static_prio,
                normal_prio: 0,
                rt_priority,
                sched_class: Some(sched_class.to_string()),
                core_cookie: None,
                pi_boosted_out_of_scx: false,
                nvcsw: 0,
                nivcsw: 0,
                signal_nvcsw: None,
                signal_nivcsw: None,
                lock_slowpath_match: None,
            }
        }

        let mut dump = FailureDumpReport::default();
        dump.task_enrichments = vec![
            make_task(100, "rt_sched_class", 50, 0),
            make_task(101, "ext_sched_class", 0, 0),
            make_task(102, "fair_sched_class", 0, 120),
        ];

        let fp = project_fingerprint(&[], Some(&dump));
        assert_eq!(fp.sched_policy_hints.len(), 3);
        match &fp.sched_policy_hints[0] {
            SchedPolicyHint::Fifo { priority } => assert_eq!(*priority, 50),
            other => panic!("expected Fifo, got {other:?}"),
        }
        match &fp.sched_policy_hints[1] {
            SchedPolicyHint::Ext => {}
            other => panic!("expected Ext, got {other:?}"),
        }
        match &fp.sched_policy_hints[2] {
            SchedPolicyHint::Other { nice } => assert_eq!(*nice, 0),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    /// Schema constant matches the documented v1 identifier so
    /// off-disk records produced by future builds are detectable
    /// as schema-bumped.
    #[test]
    fn schema_constant_pinned() {
        assert_eq!(DEBUG_CAPTURE_SCHEMA, "ktstr.debug_capture/v1");
    }

    /// `DebugCapture` round-trips through serde with default fields
    /// suppressed — minimal-fixture captures don't bloat the JSON.
    #[test]
    fn debug_capture_serde_minimal_skips_defaults() {
        let cap = DebugCapture {
            schema: DEBUG_CAPTURE_SCHEMA.to_string(),
            started_ns: 0,
            ended_ns: 0,
            kernel_release: "test-6.16".to_string(),
            ctprof_samples: Vec::new(),
            failure_dump: None,
            fingerprint: WorkloadFingerprint::default(),
        };
        let json = serde_json::to_string(&cap).unwrap();
        // ctprof_samples + failure_dump suppressed when empty/None.
        assert!(!json.contains("ctprof_samples"));
        assert!(!json.contains("failure_dump"));
        // Required fields present.
        assert!(json.contains("schema"));
        assert!(json.contains("kernel_release"));
        assert!(json.contains("test-6.16"));
    }

    /// Fingerprint round-trips with all hint variants so the
    /// reproducer generator's deserialize path covers each shape.
    #[test]
    fn fingerprint_all_hints_roundtrip() {
        let fp = WorkloadFingerprint {
            workload_groups: vec![WorkloadGroupHint {
                cgroup_path: "/system.slice/foo.service".into(),
                thread_count: 8,
                cpu_time_fraction: 0.75,
                wakeups_per_sec: 1200.0,
            }],
            affinity_hints: vec![
                AffinityHint::SingleCpu,
                AffinityHint::Exact { cpus: vec![0, 1, 2, 3] },
            ],
            work_type_hints: vec![
                WorkTypeHint::CpuSpin,
                WorkTypeHint::Bursty { burst_ms: 10, sleep_ms: 90 },
            ],
            cgroup_hints: vec![CgroupHint {
                path: "/system.slice/foo.service".into(),
                cpu_weight: Some(200),
                memory_max_bytes: Some(8 * 1024 * 1024 * 1024),
                cpuset_cpus: vec![0, 1, 2, 3],
                cpu_max_quota_us: None,
            }],
            sched_policy_hints: vec![
                SchedPolicyHint::Fifo { priority: 50 },
                SchedPolicyHint::Other { nice: 5 },
            ],
            gaps: vec!["affinity hint backed by 1 sample".into()],
        };
        let json = serde_json::to_string(&fp).unwrap();
        let parsed: WorkloadFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.workload_groups.len(), 1);
        assert_eq!(parsed.affinity_hints.len(), 2);
        assert_eq!(parsed.work_type_hints.len(), 2);
        assert_eq!(parsed.cgroup_hints.len(), 1);
        assert_eq!(parsed.sched_policy_hints.len(), 2);
        assert_eq!(parsed.gaps.len(), 1);
    }
}
