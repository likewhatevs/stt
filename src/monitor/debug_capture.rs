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
//!    generator translates into a [`crate::workload::WorkloadConfig`]
//!    spec. NOT a canonical reproducer; the projection is
//!    best-effort and the reproducer generator (or human) decides
//!    which hints to honor.
//!
//! # Capture vs reproducer-generator boundary
//!
//! This module is the DATA-SHAPE deliverable for the capture path:
//! it defines [`DebugCapture`] and [`WorkloadFingerprint`] but does
//! not produce records itself. Records are populated by external
//! producers (any caller that fills [`DebugCapture`] fields and
//! emits the value); the reproducer generator
//! ([`super::reproducer_gen`]) is the downstream consumer.
//!
//! The fingerprint projection lives here rather than in the
//! reproducer generator module because it's a pure function of
//! capture data — the reproducer generator consumes the projected
//! hints rather than re-deriving them, and that boundary is testable
//! without a producer or a generator.
//!
//! # Vocabulary alignment with ktstr test primitives
//!
//! The capture format must speak the same vocabulary as the test
//! library. Projected hints map one-to-one with primitive types in
//! `crate::workload`:
//!
//! | observation                      | hint type                          |
//! |----------------------------------|------------------------------------|
//! | per-cgroup thread count          | `WorkloadGroupHint::thread_count`  |
//! | sched_setaffinity mask patterns  | `AffinityHint::*`                  |
//! | CPU-time vs IO-wait ratio        | `WorkTypeHint::*`                  |
//! | cgroup cpu.weight + memory.max   | `CgroupHint::*`                    |
//! | scheduling policy distribution   | `SchedPolicyHint::*`               |
//!
//! Hints are SUGGESTIONS, not commands. The reproducer generator
//! weighs them against the failure dump and the user's preferences
//! (e.g. minimal-repro vs full-fidelity reproduction).
//!
//! # Filterable presentation
//!
//! The capture format is designed for `ktstr show / compare`-style
//! consumption: combinable filters by cgroup / CPU / NUMA node /
//! sched class / tgid / time window, with coherent aggregates that
//! follow from the filter. The data shape preserves enough raw
//! granularity for any of those filters to compose without forcing
//! a pre-baked aggregate menu.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::dump::FailureDumpReport;
use crate::workload::humantime_serde_helper;

/// One end-to-end debug capture record.
///
/// Bundles every observation the reproducer generator needs to
/// translate a real-world failure into a ktstr test. Serializable so
/// producers can persist captures to disk and the reproducer
/// generator can consume them offline.
///
/// `non_exhaustive` so future fields (kernel command-line snapshot,
/// scheduler config-file capture, host hardware fingerprint) can
/// land without breaking on-disk records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
#[allow(dead_code)] // the library ships the data shape; producers
// populate it externally and the reproducer
// generator consumes it.
pub struct DebugCapture {
    /// Capture format schema identifier — pinned at construction
    /// to the current [`DEBUG_CAPTURE_SCHEMA`] string. Consumers
    /// inspect the field to detect off-disk records emitted by an
    /// incompatible build before deserialising the rest of the
    /// record. No automatic migration runs today: a mismatched
    /// schema is a hard error at the consumer.
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
/// projects to `AffinityHint::SingleCpu` + `WorkTypeHint::SpinWait` +
/// `CgroupHint::WeightOverride { weight: 200 }`).
///
/// All hint vectors may be empty (insufficient data to project).
/// The reproducer generator falls back to library defaults for
/// any primitive whose hints are absent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct WorkloadFingerprint {
    /// Per-cgroup thread-count distribution. Maps to
    /// [`crate::workload::WorkloadConfig::num_workers`] in the
    /// generated test. The reproducer generator picks a
    /// representative group and emits one
    /// [`crate::workload::WorkloadConfig`] per cgroup.
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
    /// [`crate::workload::WorkloadConfig::num_workers`].
    pub thread_count: u32,
    /// Mean CPU-time fraction across the capture window (0.0 to
    /// 1.0). The reproducer generator uses this to pick a
    /// `WorkType` intensity (e.g. `SpinWait` for >0.8, `Mixed` for
    /// 0.3-0.8, `Bursty` for <0.3).
    pub cpu_time_fraction: f64,
    /// Mean wakeup rate (Hz) across the capture window. High wakeup
    /// rates suggest `PipeIo` or `FutexPingPong` workload types.
    pub wakeups_per_sec: f64,
}

/// Affinity placement hint. Maps directly to `crate::workload::AffinityIntent`.
///
/// Each topology-aware variant (`SingleCpu`, `LlcAligned`,
/// `CrossCgroup`, `SmtSiblingPair`, `RandomSubset`) carries an
/// optional `cpus` payload recording the actual CPUs the producer
/// observed at capture time.
/// Empty `cpus` means the producer classified the pattern but did not
/// record concrete CPUs (legacy producers, or projection from a
/// failure dump that lacked per-task `cpus_allowed_mask` data); the
/// reproducer generator falls back to emitting the matching
/// topology-aware [`crate::workload::AffinityIntent`] variant plus a
/// hand-edit note in that case. Non-empty `cpus` lets the reproducer
/// generator emit a concrete [`crate::workload::AffinityIntent::Exact`]
/// (or `RandomSubset` with a populated pool) directly, producing a
/// runnable spec without scenario-engine resolution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
pub enum AffinityHint {
    /// Default placement — no affinity mask narrower than the
    /// containing cgroup's cpuset. → `AffinityIntent::Inherit`.
    /// Also the [`Default`] for this enum, mirroring
    /// [`crate::workload::AffinityIntent::Inherit`] which is the
    /// default `AffinityIntent`.
    #[default]
    Inherit,
    /// Threads observed pinned to a single CPU each (mask popcount
    /// == 1 across the capture window for the majority of threads
    /// in the group). → `AffinityIntent::SingleCpu`.
    ///
    /// `cpus` records the specific CPU(s) observed across the
    /// thread group. Typically a single element when every thread
    /// pinned to the same CPU; multiple elements when different
    /// threads in the group each pinned to a distinct CPU. Empty
    /// when the producer did not record concrete CPUs.
    SingleCpu {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cpus: Vec<u32>,
    },
    /// Threads observed pinned to LLC-aligned subsets of the
    /// cgroup's cpuset. → `AffinityIntent::LlcAligned`.
    ///
    /// `cpus` records the observed LLC's CPU set when the producer
    /// resolved it. Empty when the producer classified the pattern
    /// without recording the resolved LLC mask.
    LlcAligned {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cpus: Vec<u32>,
    },
    /// Threads observed pinned to cgroup-spanning CPU sets. →
    /// `AffinityIntent::CrossCgroup`.
    ///
    /// `cpus` records the observed cross-cgroup CPU span when the
    /// producer resolved it. Empty when the producer classified the
    /// pattern without recording the resolved span.
    CrossCgroup {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cpus: Vec<u32>,
    },
    /// Threads observed pinned to the SMT siblings of one physical
    /// core (mask popcount equal to `threads_per_core` (2 on x86_64;
    /// potentially higher on SMT-N architectures such as POWER9
    /// SMT-4, POWER10 SMT-4/SMT-8, ThunderX2, or Xeon Phi) across
    /// the capture window for the majority of threads in the group,
    /// and all siblings share a `thread_siblings_list` entry per
    /// `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`).
    /// → `AffinityIntent::SmtSiblingPair`.
    ///
    /// `cpus` records the observed full sibling set when the
    /// producer resolved it — preserved as captured so SMT-N hosts
    /// retain every observed sibling. The resolver picks the lowest
    /// 2 siblings from the N-way sibling set (the scenario engine
    /// always resolves [`crate::workload::AffinityIntent::SmtSiblingPair`]
    /// to a 2-CPU pair regardless of `threads_per_core`). The
    /// variant name "Pair" reflects the resolved downstream
    /// contract, not a capture-time constraint. Empty `cpus` when
    /// the producer classified the pattern without recording the
    /// concrete siblings.
    ///
    /// Detection: appears when capture observes popcount equal to
    /// `threads_per_core` AND the CPUs share a
    /// `thread_siblings_list` entry. Popcount == 1 →
    /// [`Self::SingleCpu`]. Popcount == 2 but the CPUs are NOT
    /// siblings → [`Self::Exact`]. Partial sibling sets (popcount
    /// `> 1` but `< threads_per_core` on SMT-N>2 hosts) project to
    /// [`Self::Exact`], not `SmtSiblingPair`.
    SmtSiblingPair {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        cpus: Vec<u32>,
    },
    /// Threads observed pinned to an explicit CPU set. The capture
    /// records the exact CPUs so the reproducer can reproduce the
    /// specific placement. → `AffinityIntent::Exact`.
    Exact { cpus: Vec<u32> },
    /// Threads observed pinned to a strict subset of the cgroup's
    /// cpuset, but the subset varies across threads — typical of
    /// a placement randomizer. → `AffinityIntent::RandomSubset`.
    ///
    /// `from` records the observed source pool (the union of CPUs
    /// any thread was pinned to across the capture window) when the
    /// producer resolved it. `count` records the typical mask
    /// popcount per thread. Empty `from` or zero `count` means the
    /// producer classified the pattern without recording the
    /// resolved pool / sample size. Field name matches
    /// [`crate::workload::AffinityIntent::RandomSubset::from`] so the
    /// hint and the resolved intent share the same vocabulary.
    RandomSubset {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        from: Vec<u32>,
        #[serde(default, skip_serializing_if = "is_zero_u32")]
        count: u32,
    },
}

impl AffinityHint {
    /// Construct a `SingleCpu` hint with the producer-observed CPU
    /// set. Mirrors [`crate::workload::AffinityIntent::exact`]'s
    /// iterator flexibility — accepts arrays, ranges, `Vec`, or any
    /// `IntoIterator<Item = u32>`.
    ///
    /// Pass an empty iterator (e.g. `[] as [u32; 0]`,
    /// `Vec::<u32>::new()`) to construct the unresolved form: the
    /// producer classified the pattern as single-CPU pinning but did
    /// not record the concrete CPU. Downstream the reproducer
    /// generator falls back to
    /// [`crate::workload::AffinityIntent::SingleCpu`] plus a
    /// hand-edit note. Same convention applies to
    /// [`Self::llc_aligned`], [`Self::cross_cgroup`],
    /// [`Self::smt_sibling_pair`].
    pub fn single_cpu(cpus: impl IntoIterator<Item = u32>) -> Self {
        Self::SingleCpu {
            cpus: cpus.into_iter().collect(),
        }
    }

    /// Construct an `LlcAligned` hint with the producer-observed LLC
    /// CPU set. Empty iterator → unresolved form (see
    /// [`Self::single_cpu`] for the unresolved-classification
    /// semantics).
    pub fn llc_aligned(cpus: impl IntoIterator<Item = u32>) -> Self {
        Self::LlcAligned {
            cpus: cpus.into_iter().collect(),
        }
    }

    /// Construct a `CrossCgroup` hint with the producer-observed
    /// cross-cgroup CPU span. Empty iterator → unresolved form (see
    /// [`Self::single_cpu`]).
    pub fn cross_cgroup(cpus: impl IntoIterator<Item = u32>) -> Self {
        Self::CrossCgroup {
            cpus: cpus.into_iter().collect(),
        }
    }

    /// Construct an `SmtSiblingPair` hint with the producer-observed
    /// sibling-set CPU IDs. The `IntoIterator` parameter accepts the
    /// full sibling set so SMT-N>2 hosts (POWER9 SMT-4, POWER10
    /// SMT-4/SMT-8, ThunderX2, Xeon Phi) preserve every observed
    /// sibling — see the variant doc for the resolver-side narrowing
    /// to 2 CPUs. Empty iterator → unresolved form (see
    /// [`Self::single_cpu`]).
    pub fn smt_sibling_pair(cpus: impl IntoIterator<Item = u32>) -> Self {
        Self::SmtSiblingPair {
            cpus: cpus.into_iter().collect(),
        }
    }

    /// Construct an `Exact` hint with the producer-observed CPU set.
    /// Maps to [`crate::workload::AffinityIntent::Exact`] in the
    /// reproducer generator.
    ///
    /// No `exact_unresolved()` companion exists. `Exact` is the
    /// terminal resolved state — the variant carries concrete CPUs
    /// and has no semantic interpretation without them. The 4
    /// topology-aware variants ([`Self::SingleCpu`],
    /// [`Self::LlcAligned`], [`Self::CrossCgroup`],
    /// [`Self::SmtSiblingPair`]) retain their meaning when the
    /// payload is empty (the resolver determines the CPUs from
    /// topology); `Exact` does not, so `Self::exact([])` is
    /// permitted for type symmetry but produces a placeholder that
    /// the spawn-time affinity gate rejects as malformed.
    pub fn exact(cpus: impl IntoIterator<Item = u32>) -> Self {
        Self::Exact {
            cpus: cpus.into_iter().collect(),
        }
    }

    /// Construct a `RandomSubset` hint with the producer-observed
    /// pool and per-thread popcount. Mirrors
    /// [`crate::workload::AffinityIntent::random_subset`]. The
    /// reproducer generator emits an empty placeholder when `from`
    /// is empty or `count == 0`.
    pub fn random_subset(from: impl IntoIterator<Item = u32>, count: u32) -> Self {
        Self::RandomSubset {
            from: from.into_iter().collect(),
            count,
        }
    }

    /// Construct an unresolved `RandomSubset` hint — the producer
    /// classified the pattern but did not record the resolved pool /
    /// sample size. The reproducer generator emits an empty
    /// placeholder that the spawn-time gate rejects, prompting
    /// hand-edit before the spec runs.
    pub fn random_subset_unresolved() -> Self {
        Self::RandomSubset {
            from: Vec::new(),
            count: 0,
        }
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// Workload type hint. Maps to `crate::workload::WorkType` variants.
/// The hint records the primary signal (CPU-bound vs IO-bound vs
/// futex / pipe wake patterns) and lets the reproducer generator
/// choose a parameterized variant (e.g. `WorkType::Bursty` with
/// `burst_duration` / `sleep_duration` picked from the hint's window
/// measurement).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
pub enum WorkTypeHint {
    /// CPU-bound, no observed IO or wake-driven blocking. Maps to
    /// `WorkType::SpinWait`.
    SpinWait,
    /// Heavy `sched_yield` rate observed (yields/sec >>
    /// involuntary-context-switch rate). Maps to
    /// `WorkType::YieldHeavy`.
    YieldHeavy,
    /// Mixed CPU + yield pattern. `WorkType::Mixed`.
    Mixed,
    /// CPU bursts followed by long sleeps. Maps to
    /// `WorkType::Bursty` with measured `burst_duration` /
    /// `sleep_duration`.
    Bursty {
        #[serde(with = "humantime_serde_helper")]
        burst_duration: Duration,
        #[serde(with = "humantime_serde_helper")]
        sleep_duration: Duration,
    },
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
    /// Synchronous-write workload — short bursts of `pwrite` followed
    /// by `fdatasync`, opened with `O_SYNC`. Detection signal (when
    /// capture pipeline wired): `O_SYNC` open flag plus a sequential
    /// pwrite pattern. Maps to `WorkType::IoSyncWrite`.
    IoSyncWrite,
    /// Random-read direct-IO workload — single-block `pread` at
    /// random offsets, opened with `O_DIRECT`. Detection signal
    /// (when capture pipeline wired): `O_DIRECT` open flag plus a
    /// single-block read-only pattern at scattered offsets. Maps
    /// to `WorkType::IoRandRead`.
    IoRandRead,
    /// Interleaved sequential `pwrite` and random `pread` with
    /// periodic `fdatasync` via `O_DIRECT`. Detection signal (when
    /// capture pipeline wired): `O_DIRECT` open flag plus mixed
    /// read/write traffic at the same fd. Maps to
    /// `WorkType::IoConvoy`.
    IoConvoy,
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
/// on-disk shape changes incompatibly. Consumers compare the
/// stamped [`DebugCapture::schema`] against this constant and reject
/// mismatches before deserialising; no automatic migration runs.
#[allow(dead_code)] // the library ships the pinned constant;
// producers stamp it at capture time.
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
        fp.gaps
            .push("affinity + sched_policy hints unavailable (no failure dump)".to_string());
    }

    // WorkSpec-type hints come from CPU-time / wakeup-rate shape across
    // the sampling window. Bursty / IoSyncWrite are detected by sleep
    // ratio; SpinWait / Mixed by yield rate vs CPU time.
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
        // `SchedClassRegistry::decode` returns the short class name
        // ("fair", "rt", "dl", "idle", "stop", "ext"); match against
        // those exact strings, NOT the kernel symbol names
        // ("rt_sched_class" etc.). A previous regression matched
        // against the long form and silently routed every rt/dl/ext
        // task to the `_` arm, projecting them as
        // `SchedPolicyHint::Other` with a clamped nice value.
        let class = task.sched_class.as_deref().unwrap_or("");
        let policy = match class {
            "rt" if task.rt_priority > 0 => {
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
            "dl" => SchedPolicyHint::Deadline {
                runtime_ns: 0,
                deadline_ns: 0,
                period_ns: 0,
            },
            "ext" => SchedPolicyHint::Ext,
            "idle" => SchedPolicyHint::Idle,
            _ => {
                // Kernel default static_prio for nice 0 is 120
                // (NICE_TO_PRIO(0) = MAX_RT_PRIO + 20 = 120), so the
                // raw observed nice value is `static_prio - 120`.
                // The cast to i32 keeps the subtraction in signed
                // arithmetic so a zero-initialised static_prio (from
                // a failure dump that didn't capture the field, or a
                // synthetic test fixture) doesn't panic.
                //
                // Linux exposes nice in `[-20, 19]`
                // (`include/linux/sched/prio.h`'s `MIN_NICE = -20`,
                // `MAX_NICE = 19`). The `raw` value can fall outside
                // that range when the dump's static_prio is
                // missing/zero-init. Clamp to the legal range so the
                // projected `SchedPolicyHint::Other { nice }` is
                // spawnable by the reproducer generator without
                // further sanitisation.
                let raw = task.static_prio - 120;
                let nice = raw.clamp(-20, 19);
                SchedPolicyHint::Other { nice }
            }
        };
        hints.push(policy);
    }
    hints
}

fn project_work_type_hints(_samples: &[crate::ctprof::CtprofSnapshot]) -> Vec<WorkTypeHint> {
    // Returns empty; classification is human/LLM reading
    // `ktstr show`/`compare` output.
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
    /// `rt` → Fifo and `ext` → Ext mappings — class names match
    /// `SchedClassRegistry::decode`'s short-name return values
    /// ("fair", "rt", "dl", "idle", "stop", "ext"), NOT the kernel
    /// symbol strings ("rt_sched_class" etc.).
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

        let dump = FailureDumpReport {
            task_enrichments: vec![
                make_task(100, "rt", 50, 0),
                make_task(101, "ext", 0, 0),
                make_task(102, "fair", 0, 120),
                // Zero-init static_prio + fair-class entry: a failure
                // dump captured before the kernel populated the field.
                // Pre-clamp this would project to nice=-120
                // (`(0 as i32) - 120`) and produce an unspawnable
                // SchedPolicyHint::Other. The clamp(-20, 19) bound (per
                // `MIN_NICE`/`MAX_NICE` in
                // `include/linux/sched/prio.h`) keeps the projected
                // nice value in the kernel's legal range.
                make_task(103, "fair", 0, 0),
            ],
            ..FailureDumpReport::default()
        };

        let fp = project_fingerprint(&[], Some(&dump));
        assert_eq!(fp.sched_policy_hints.len(), 4);
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
        // Zero-init static_prio with a fair-class entry must clamp
        // to MIN_NICE = -20 instead of producing nice=-120.
        match &fp.sched_policy_hints[3] {
            SchedPolicyHint::Other { nice } => assert_eq!(
                *nice, -20,
                "static_prio=0 (zero-init) must clamp to MIN_NICE=-20, got nice={nice}",
            ),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    /// Regression: every short class name returned by
    /// `SchedClassRegistry::decode` ("fair", "rt", "dl", "idle",
    /// "stop", "ext") must match the projection arms; long kernel
    /// symbol names ("rt_sched_class", "dl_sched_class",
    /// "ext_sched_class", "idle_sched_class") must NOT match and
    /// must fall through to the `_` arm (projecting as `Other`).
    /// Pins the symptom of the previous regression where the
    /// projection matched against the long form and silently
    /// misclassified every rt/dl/ext task.
    #[test]
    fn project_sched_policy_short_names_match_long_names_fall_through() {
        use crate::monitor::task_enrichment::TaskEnrichment;

        fn make_task(class: &str, rt_priority: u32, static_prio: i32) -> TaskEnrichment {
            TaskEnrichment {
                pid: 0,
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
                sched_class: Some(class.to_string()),
                core_cookie: None,
                pi_boosted_out_of_scx: false,
                nvcsw: 0,
                nivcsw: 0,
                signal_nvcsw: None,
                signal_nivcsw: None,
                lock_slowpath_match: None,
            }
        }

        // Short names: each maps to its dedicated SchedPolicyHint
        // variant (the wired-up cases).
        let dump = FailureDumpReport {
            task_enrichments: vec![
                make_task("rt", 75, 0),
                make_task("dl", 0, 0),
                make_task("ext", 0, 0),
                make_task("idle", 0, 0),
            ],
            ..FailureDumpReport::default()
        };
        let fp = project_fingerprint(&[], Some(&dump));
        assert_eq!(fp.sched_policy_hints.len(), 4);
        assert!(
            matches!(fp.sched_policy_hints[0], SchedPolicyHint::Fifo { priority: 75 }),
            "short name 'rt' must project to Fifo, got {:?}",
            fp.sched_policy_hints[0],
        );
        assert!(
            matches!(fp.sched_policy_hints[1], SchedPolicyHint::Deadline { .. }),
            "short name 'dl' must project to Deadline, got {:?}",
            fp.sched_policy_hints[1],
        );
        assert!(
            matches!(fp.sched_policy_hints[2], SchedPolicyHint::Ext),
            "short name 'ext' must project to Ext, got {:?}",
            fp.sched_policy_hints[2],
        );
        assert!(
            matches!(fp.sched_policy_hints[3], SchedPolicyHint::Idle),
            "short name 'idle' must project to Idle, got {:?}",
            fp.sched_policy_hints[3],
        );

        // Long kernel symbol names: every one must fall through to
        // the `_` arm. A regression that re-introduces the long-name
        // match would surface here as a non-Other variant.
        let long_names_dump = FailureDumpReport {
            task_enrichments: vec![
                make_task("rt_sched_class", 75, 120),
                make_task("dl_sched_class", 0, 120),
                make_task("ext_sched_class", 0, 120),
                make_task("idle_sched_class", 0, 120),
            ],
            ..FailureDumpReport::default()
        };
        let long_fp = project_fingerprint(&[], Some(&long_names_dump));
        assert_eq!(long_fp.sched_policy_hints.len(), 4);
        for (i, hint) in long_fp.sched_policy_hints.iter().enumerate() {
            assert!(
                matches!(hint, SchedPolicyHint::Other { .. }),
                "long kernel symbol name at index {i} must NOT match \
                 any specialised arm (regression guard); got {hint:?}",
            );
        }
    }

    /// Schema constant matches the documented `v1` identifier.
    /// Consumers compare the stamped [`DebugCapture::schema`] against
    /// this exact string; a future bump that flips the constant
    /// without coordinating consumers must surface here first.
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
                AffinityHint::SingleCpu { cpus: Vec::new() },
                AffinityHint::Exact {
                    cpus: vec![0, 1, 2, 3],
                },
            ],
            work_type_hints: vec![
                WorkTypeHint::SpinWait,
                WorkTypeHint::Bursty {
                    burst_duration: Duration::from_millis(10),
                    sleep_duration: Duration::from_millis(90),
                },
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

    /// Topology-aware [`AffinityHint`] variants round-trip both
    /// shapes — empty `cpus` (unresolved, the producer classified
    /// the pattern without recording concrete CPUs) and non-empty
    /// `cpus` (resolved, the producer recorded the observed CPU
    /// set). `RandomSubset` adds the `from`/`count` pair. Pins the
    /// `#[serde(default, skip_serializing_if = "Vec::is_empty")]`
    /// wire shape so a regression that drops the resolved-payload
    /// fields silently surfaces here rather than at
    /// reproducer-generation time.
    #[test]
    fn affinity_hint_resolved_payload_roundtrips() {
        let hints = [
            AffinityHint::Inherit,
            AffinityHint::SingleCpu { cpus: Vec::new() },
            AffinityHint::SingleCpu { cpus: vec![3] },
            AffinityHint::LlcAligned { cpus: Vec::new() },
            AffinityHint::LlcAligned {
                cpus: vec![0, 1, 2, 3],
            },
            AffinityHint::CrossCgroup { cpus: Vec::new() },
            AffinityHint::CrossCgroup {
                cpus: vec![4, 5, 6, 7],
            },
            AffinityHint::SmtSiblingPair { cpus: Vec::new() },
            AffinityHint::SmtSiblingPair { cpus: vec![2, 3] },
            AffinityHint::Exact {
                cpus: vec![0, 1, 2, 3],
            },
            AffinityHint::RandomSubset {
                from: Vec::new(),
                count: 0,
            },
            AffinityHint::RandomSubset {
                from: vec![0, 1, 2, 3, 4, 5],
                count: 3,
            },
        ];
        for hint in &hints {
            let json = serde_json::to_string(hint).expect("AffinityHint must serialize");
            let back: AffinityHint =
                serde_json::from_str(&json).expect("AffinityHint must deserialize");
            match (hint, &back) {
                (AffinityHint::Inherit, AffinityHint::Inherit) => {}
                (AffinityHint::SingleCpu { cpus: a }, AffinityHint::SingleCpu { cpus: b }) => {
                    assert_eq!(a, b, "SingleCpu cpus must round-trip")
                }
                (AffinityHint::LlcAligned { cpus: a }, AffinityHint::LlcAligned { cpus: b }) => {
                    assert_eq!(a, b, "LlcAligned cpus must round-trip")
                }
                (AffinityHint::CrossCgroup { cpus: a }, AffinityHint::CrossCgroup { cpus: b }) => {
                    assert_eq!(a, b, "CrossCgroup cpus must round-trip")
                }
                (
                    AffinityHint::SmtSiblingPair { cpus: a },
                    AffinityHint::SmtSiblingPair { cpus: b },
                ) => assert_eq!(a, b, "SmtSiblingPair cpus must round-trip"),
                (AffinityHint::Exact { cpus: a }, AffinityHint::Exact { cpus: b }) => {
                    assert_eq!(a, b, "Exact cpus must round-trip")
                }
                (
                    AffinityHint::RandomSubset {
                        from: pa,
                        count: ca,
                    },
                    AffinityHint::RandomSubset {
                        from: pb,
                        count: cb,
                    },
                ) => {
                    assert_eq!(pa, pb, "RandomSubset from must round-trip");
                    assert_eq!(ca, cb, "RandomSubset count must round-trip");
                }
                _ => panic!("AffinityHint round-trip mismatch: sent {hint:?}, got {back:?}",),
            }
        }
    }

    /// `WorkTypeHint::IoRandRead` and `WorkTypeHint::IoConvoy`
    /// round-trip through the `#[serde(tag = "kind")]` wire format
    /// without losing the variant. Pins both new IO-mode hints
    /// alongside the existing `IoSyncWrite` so a regression that
    /// drops one of them silently from the fingerprint serializer
    /// surfaces here rather than at reproducer-generation time.
    #[test]
    fn work_type_hint_io_variants_roundtrip() {
        for hint in [
            WorkTypeHint::IoSyncWrite,
            WorkTypeHint::IoRandRead,
            WorkTypeHint::IoConvoy,
        ] {
            let json = serde_json::to_string(&hint).expect("WorkTypeHint must serialize");
            let back: WorkTypeHint =
                serde_json::from_str(&json).expect("WorkTypeHint must deserialize");
            // Match-arm equality so the test fails on a wrong
            // variant rather than a generic mismatch.
            match (&hint, &back) {
                (WorkTypeHint::IoSyncWrite, WorkTypeHint::IoSyncWrite) => {}
                (WorkTypeHint::IoRandRead, WorkTypeHint::IoRandRead) => {}
                (WorkTypeHint::IoConvoy, WorkTypeHint::IoConvoy) => {}
                _ => panic!("IO hint roundtrip mismatch: sent {hint:?}, got {back:?}",),
            }
        }
    }

    /// Constructors produce the same enum variants as struct-literal
    /// construction. Pins the convention that
    /// `AffinityHint::single_cpu(cpus)` and
    /// `AffinityHint::SingleCpu { cpus }` are interchangeable for
    /// both resolved (non-empty `cpus`) and unresolved (empty `cpus`)
    /// shapes. Each topology-aware ctor drives both paths via an
    /// empty / non-empty iterator. `Exact` and `RandomSubset` are
    /// also covered.
    #[test]
    fn affinity_hint_constructors_match_struct_literal() {
        // Resolved + unresolved single_cpu via the same ctor.
        assert!(matches!(
            AffinityHint::single_cpu([3u32]),
            AffinityHint::SingleCpu { cpus } if cpus == vec![3]
        ));
        assert!(matches!(
            AffinityHint::single_cpu([] as [u32; 0]),
            AffinityHint::SingleCpu { cpus } if cpus.is_empty()
        ));
        // Resolved + unresolved llc_aligned.
        assert!(matches!(
            AffinityHint::llc_aligned(0u32..4),
            AffinityHint::LlcAligned { cpus } if cpus == vec![0, 1, 2, 3]
        ));
        assert!(matches!(
            AffinityHint::llc_aligned(Vec::<u32>::new()),
            AffinityHint::LlcAligned { cpus } if cpus.is_empty()
        ));
        // Resolved + unresolved cross_cgroup.
        assert!(matches!(
            AffinityHint::cross_cgroup(vec![4u32, 5, 6, 7]),
            AffinityHint::CrossCgroup { cpus } if cpus == vec![4, 5, 6, 7]
        ));
        assert!(matches!(
            AffinityHint::cross_cgroup(Vec::<u32>::new()),
            AffinityHint::CrossCgroup { cpus } if cpus.is_empty()
        ));
        // Resolved + unresolved smt_sibling_pair.
        assert!(matches!(
            AffinityHint::smt_sibling_pair([2u32, 3]),
            AffinityHint::SmtSiblingPair { cpus } if cpus == vec![2, 3]
        ));
        assert!(matches!(
            AffinityHint::smt_sibling_pair(Vec::<u32>::new()),
            AffinityHint::SmtSiblingPair { cpus } if cpus.is_empty()
        ));
        // Exact (no semantic unresolved form — see `Self::exact`
        // doc); RandomSubset has its own resolved/unresolved pair.
        assert!(matches!(
            AffinityHint::exact([0u32, 1, 2]),
            AffinityHint::Exact { cpus } if cpus == vec![0, 1, 2]
        ));
        assert!(matches!(
            AffinityHint::random_subset([0u32, 1, 2, 3, 4, 5], 3),
            AffinityHint::RandomSubset { from, count }
                if from == vec![0, 1, 2, 3, 4, 5] && count == 3
        ));
        assert!(matches!(
            AffinityHint::random_subset_unresolved(),
            AffinityHint::RandomSubset { from, count }
                if from.is_empty() && count == 0
        ));
    }
}
