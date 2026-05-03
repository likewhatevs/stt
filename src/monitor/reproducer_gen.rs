//! Translate [`super::debug_capture::WorkloadFingerprint`] hints into
//! ktstr test specs (`WorkloadConfig` values) and generated source
//! code.
//!
//! Pipeline position: this module sits at the END of the live-host
//! pipeline.
//!
//! ```text
//! BpfSyscallAccessor    ──┐
//! LiveHostKernelEnv      ─┤   ┌───────────────────┐    ┌──────────────────┐
//! KallsymsTable          ─┼─→ │ DebugCapture      │ →  │ ReproducerSpec   │
//! dmesg_scx parser       ─┤   │ + Fingerprint     │    │ (this module)    │
//! ctprof::CtprofSnapshot ──┘   └───────────────────┘    └──────────────────┘
//!                                (capture pipeline)        (this module)
//! ```
//!
//! # The translation contract
//!
//! The generator's job is to take fingerprint HINTS (best-effort
//! projections from
//! capture data) and emit a test spec the framework will execute.
//! The generator is NOT a classifier — it does not decide "this
//! workload is locking-bound" or "this is a cache pressure
//! pathology"; it just maps observed shapes to primitive types.
//! Pathology classification is a separate, downstream concern (the
//! reader / LLM consuming `ktstr show / compare`).
//!
//! # Mapping table
//!
//! | fingerprint hint              | ktstr type                              |
//! |-------------------------------|-----------------------------------------|
//! | WorkloadGroupHint.thread_count| WorkloadConfig::num_workers             |
//! | AffinityHint::SingleCpu       | ResolvedAffinity::SingleCpu(0)              |
//! | AffinityHint::Exact{cpus}     | ResolvedAffinity::Fixed(set)                |
//! | AffinityHint::Inherit         | ResolvedAffinity::None                      |
//! | AffinityHint::RandomSubset    | ResolvedAffinity::Random { from, count }    |
//! | WorkTypeHint::SpinWait         | WorkType::SpinWait                       |
//! | WorkTypeHint::YieldHeavy      | WorkType::YieldHeavy                    |
//! | WorkTypeHint::Mixed           | WorkType::Mixed                         |
//! | WorkTypeHint::Bursty{b,s}     | WorkType::Bursty { burst_ms, sleep_ms } |
//! | WorkTypeHint::PipeIo          | WorkType::PipeIo { burst_iters: 1024 }  |
//! | WorkTypeHint::FutexPingPong   | WorkType::FutexPingPong { spin_iters: 1024 } |
//! | WorkTypeHint::CachePressure   | WorkType::CachePressure { size_kb, stride } |
//! | WorkTypeHint::IoSyncWrite     | WorkType::IoSyncWrite                   |
//! | WorkTypeHint::IoRandRead      | WorkType::IoRandRead                    |
//! | WorkTypeHint::IoConvoy        | WorkType::IoConvoy                      |
//! | SchedPolicyHint::Other{nice}  | SchedPolicy::Normal + nice              |
//! | SchedPolicyHint::Fifo{prio}   | SchedPolicy::Fifo(prio)                 |
//! | SchedPolicyHint::RoundRobin   | SchedPolicy::RoundRobin(prio)           |
//! | SchedPolicyHint::Deadline     | SchedPolicy::Deadline(...)              |
//! | SchedPolicyHint::Batch        | SchedPolicy::Batch                      |
//! | SchedPolicyHint::Idle         | SchedPolicy::Idle                       |
//! | SchedPolicyHint::Ext          | (no explicit policy — scx default)      |
//!
//! `IoRandRead` and `IoConvoy` hints are accepted by the projection
//! layer but not yet emitted by the capture pipeline; all real-disk-
//! IO captures currently project to `IoSyncWrite`. The mapping is
//! ready for the day the pipeline learns to discriminate IO modes
//! from the captured open-flag + IO-shape signals documented on
//! [`super::debug_capture::WorkTypeHint`].
//!
//! Hints that don't fire produce framework defaults. Hints that
//! fire ambiguously (multiple variants in one fingerprint) pick
//! the first observed-frequency-ranked entry; the rest are emitted
//! as `notes` on [`ReproducerSpec`] so the human / generator
//! consumer can choose to override.
//!
//! # Output formats
//!
//! Two surfaces:
//!
//! - [`ReproducerSpec`] — a programmatic value that the framework
//!   can execute directly. Used by tests and tooling that
//!   construct workloads in-process.
//! - [`render_run_file_source`] / [`render_ktstr_test_source`] —
//!   generated Rust source string that recreates the spec via
//!   library APIs. Used by the `cargo ktstr export` /
//!   `cargo ktstr capture-reproduce` flow that emits a self-
//!   contained test file.

use std::collections::BTreeSet;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::debug_capture::{
    AffinityHint, CgroupHint, DebugCapture, SchedPolicyHint, WorkTypeHint, WorkloadFingerprint,
    WorkloadGroupHint,
};
use crate::workload::{ResolvedAffinity, MemPolicy, MpolFlags, SchedPolicy, WorkType, WorkloadConfig, CloneMode};

/// One reproducer spec — a `WorkloadConfig` value plus diagnostic
/// notes about confidence / ambiguity.
///
/// The framework can execute the `config` directly. The `notes` and
/// `cgroup_hints` fields surface low-confidence projections so the
/// caller (or generated source) can flag them for the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)] // wired by the (separate) cargo ktstr capture-
                    // reproduce subcommand; library lands the type.
pub struct ReproducerSpec {
    /// The mappable WorkloadConfig — what the framework executes.
    /// Skipped from serde because WorkloadConfig isn't (and doesn't
    /// need to be) Serialize. The serialized form is a pair of
    /// the source-rendered spec text + the `notes`.
    #[serde(skip)]
    pub config: WorkloadConfig,
    /// Cgroup-shaped hints that don't fit on `WorkloadConfig`
    /// directly — the framework's [`crate::workload`] surface
    /// doesn't yet have a `cgroup` field on `WorkloadConfig`, so
    /// the generator emits cgroup hints alongside for the test
    /// harness or `.run` shar to apply at setup time. Maps to
    /// `cgroup_def!` macro in generated source.
    pub cgroup_hints: Vec<CgroupHint>,
    /// Notes about projection quality — ambiguous mappings,
    /// unmapped variants, low-confidence hints (sample size = 1,
    /// fingerprint gaps cited in input).
    pub notes: Vec<String>,
    /// The scheduler name the capture was running, when known.
    /// Lets the generated test pick the right scheduler binary
    /// to attach.
    pub scheduler_name: String,
}

impl Default for ReproducerSpec {
    fn default() -> Self {
        Self {
            config: WorkloadConfig::default(),
            cgroup_hints: Vec::new(),
            notes: Vec::new(),
            scheduler_name: String::new(),
        }
    }
}

/// Produce a [`ReproducerSpec`] from a [`DebugCapture`].
///
/// Pure function: same capture always produces the same spec. The
/// projection is deterministic and dependency-free — only the
/// fingerprint hints matter. Fingerprint gaps propagate into
/// `spec.notes` so the caller can see why a particular field
/// fell back to default.
///
/// Picks the FIRST hint of each kind when multiple are present
/// (fingerprint atoms are documented to be sorted by
/// frequency-descending). Subsequent hints are recorded in `notes`
/// as alternative observations the human / LLM can choose to
/// override.
#[allow(dead_code)]
pub fn generate_spec(capture: &DebugCapture) -> ReproducerSpec {
    let mut spec = ReproducerSpec {
        scheduler_name: failure_scheduler_name(capture),
        ..Default::default()
    };

    map_workload_groups(&capture.fingerprint, &mut spec);
    map_affinity(&capture.fingerprint, &mut spec);
    map_work_type(&capture.fingerprint, &mut spec);
    map_sched_policy(&capture.fingerprint, &mut spec);
    spec.cgroup_hints = capture.fingerprint.cgroup_hints.clone();

    // Carry fingerprint gaps forward — the user sees them so they
    // know where to refine the capture or hand-edit the spec.
    for gap in &capture.fingerprint.gaps {
        spec.notes.push(format!("fingerprint gap: {gap}"));
    }

    spec
}

fn failure_scheduler_name(capture: &DebugCapture) -> String {
    // Prefer the failure-dump's render of the scheduler if present;
    // FailureDumpReport doesn't carry the scheduler name field
    // today (a likely follow-up enrichment), so derive from the
    // capture's metadata where possible. For now, leave empty —
    // generated source will note the scheduler-name omission and
    // expect the user to fill it in.
    let _ = capture;
    String::new()
}

fn map_workload_groups(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.workload_groups.first() else {
        spec.notes.push(
            "no workload groups in fingerprint — defaulting num_workers=1".into(),
        );
        return;
    };
    spec.config.num_workers = primary.thread_count.max(1) as usize;
    if fp.workload_groups.len() > 1 {
        let alts: Vec<String> = fp
            .workload_groups
            .iter()
            .skip(1)
            .map(|g: &WorkloadGroupHint| {
                format!("{} ({} threads)", g.cgroup_path, g.thread_count)
            })
            .collect();
        spec.notes.push(format!(
            "additional workload groups not modeled in primary spec: {}",
            alts.join(", ")
        ));
    }
}

fn map_affinity(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.affinity_hints.first() else {
        return;
    };
    spec.config.affinity = match primary {
        AffinityHint::Inherit => ResolvedAffinity::None,
        AffinityHint::SingleCpu => ResolvedAffinity::SingleCpu(0),
        AffinityHint::LlcAligned => {
            // ResolvedAffinity doesn't carry an LlcAligned variant —
            // the topology resolver handles that at framework
            // level (`AffinityIntent::LlcAligned`). Fall back to
            // `None` and surface the hint as a note.
            spec.notes.push(
                "AffinityHint::LlcAligned observed; \
                 ResolvedAffinity lacks an LlcAligned variant — \
                 framework runs with ResolvedAffinity::None and the \
                 test harness should use AffinityIntent::LlcAligned \
                 from the higher-level workload builder"
                    .into(),
            );
            ResolvedAffinity::None
        }
        AffinityHint::CrossCgroup => {
            spec.notes.push(
                "AffinityHint::CrossCgroup observed; framework runs \
                 with ResolvedAffinity::None — the cgroup-spanning placement \
                 is the harness's responsibility to set up via \
                 AffinityIntent::CrossCgroup at the test-builder level"
                    .into(),
            );
            ResolvedAffinity::None
        }
        AffinityHint::Exact { cpus } => {
            let set: BTreeSet<usize> = cpus.iter().map(|&c| c as usize).collect();
            ResolvedAffinity::Fixed(set)
        }
        AffinityHint::RandomSubset => {
            // Without the source pool, default to single-CPU
            // sampling. The note tells the user to refine the
            // pool when they hand-edit.
            spec.notes.push(
                "AffinityHint::RandomSubset observed — no source \
                 pool inferred; emitting ResolvedAffinity::SingleCpu(0). \
                 Hand-edit to ResolvedAffinity::Random { from, count } \
                 with the actual cpuset."
                    .into(),
            );
            ResolvedAffinity::SingleCpu(0)
        }
    };

    if fp.affinity_hints.len() > 1 {
        let alts: Vec<String> = fp
            .affinity_hints
            .iter()
            .skip(1)
            .map(|a| format!("{a:?}"))
            .collect();
        spec.notes.push(format!(
            "additional affinity hints not modeled: {}",
            alts.join(", ")
        ));
    }
}

fn map_work_type(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.work_type_hints.first() else {
        spec.notes.push(
            "no work-type hint in fingerprint — defaulting to \
             WorkType::SpinWait"
                .into(),
        );
        return;
    };
    spec.config.work_type = match primary {
        WorkTypeHint::SpinWait => WorkType::SpinWait,
        WorkTypeHint::YieldHeavy => WorkType::YieldHeavy,
        WorkTypeHint::Mixed => WorkType::Mixed,
        WorkTypeHint::Bursty { burst_ms, sleep_ms } => WorkType::Bursty {
            burst_ms: *burst_ms,
            sleep_ms: *sleep_ms,
        },
        WorkTypeHint::PipeIo => WorkType::PipeIo { burst_iters: 1024 },
        WorkTypeHint::FutexPingPong => WorkType::FutexPingPong { spin_iters: 1024 },
        WorkTypeHint::CachePressure { size_kb, stride } => WorkType::CachePressure {
            size_kb: *size_kb as usize,
            stride: *stride as usize,
        },
        WorkTypeHint::IoSyncWrite => WorkType::IoSyncWrite,
        WorkTypeHint::IoRandRead => WorkType::IoRandRead,
        WorkTypeHint::IoConvoy => WorkType::IoConvoy,
    };

    if fp.work_type_hints.len() > 1 {
        let alts: Vec<String> = fp
            .work_type_hints
            .iter()
            .skip(1)
            .map(|w| format!("{w:?}"))
            .collect();
        spec.notes.push(format!(
            "additional work-type hints observed: {}",
            alts.join(", ")
        ));
    }
}

fn map_sched_policy(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.sched_policy_hints.first() else {
        return;
    };
    match primary {
        SchedPolicyHint::Other { nice } => {
            spec.config.sched_policy = SchedPolicy::Normal;
            spec.config.nice = *nice;
        }
        SchedPolicyHint::Fifo { priority } => {
            spec.config.sched_policy = SchedPolicy::Fifo(*priority);
        }
        SchedPolicyHint::RoundRobin { priority } => {
            spec.config.sched_policy = SchedPolicy::RoundRobin(*priority);
        }
        SchedPolicyHint::Deadline {
            runtime_ns,
            deadline_ns,
            period_ns,
        } => {
            spec.config.sched_policy = SchedPolicy::deadline(
                Duration::from_nanos(*runtime_ns),
                Duration::from_nanos(*deadline_ns),
                Duration::from_nanos(*period_ns),
            );
        }
        SchedPolicyHint::Batch => spec.config.sched_policy = SchedPolicy::Batch,
        SchedPolicyHint::Idle => spec.config.sched_policy = SchedPolicy::Idle,
        SchedPolicyHint::Ext => {
            // No SchedPolicy mapping for SCHED_EXT — the harness
            // routes tasks through scx by default. Note for the
            // generated source.
            spec.notes.push(
                "SchedPolicyHint::Ext observed; framework defaults to \
                 scx routing — no policy override emitted"
                    .into(),
            );
        }
    }
    // Ensure framework defaults for unset MemPolicy/MpolFlags/CloneMode.
    spec.config.mem_policy = MemPolicy::Default;
    spec.config.mpol_flags = MpolFlags::NONE;
    spec.config.clone_mode = CloneMode::Fork;

    if fp.sched_policy_hints.len() > 1 {
        let alts: Vec<String> = fp
            .sched_policy_hints
            .iter()
            .skip(1)
            .map(|s| format!("{s:?}"))
            .collect();
        spec.notes.push(format!(
            "additional sched-policy hints observed: {}",
            alts.join(", ")
        ));
    }
}

/// Render a `.run` shar-style entry-point source string from a
/// [`ReproducerSpec`].
///
/// The output is a small Rust program that constructs the
/// `WorkloadConfig` via builder calls and runs it through the
/// framework's standalone harness. Designed to drop into the
/// `.run` archive shape that `cargo ktstr export` already
/// produces — a self-contained reproducer the user can run
/// independently of their original test environment.
///
/// `template_name` becomes the generated function name. Use a
/// stable, descriptive name that survives a re-run with the same
/// capture (e.g. derived from `capture.scheduler_name` +
/// `capture.started_ns`).
#[allow(dead_code)]
pub fn render_run_file_source(spec: &ReproducerSpec, template_name: &str) -> String {
    let mut s = String::new();
    s.push_str("// Auto-generated reproducer from a debug capture.\n");
    s.push_str("// Edit the WorkloadConfig builder calls to refine\n");
    s.push_str("// the projection.\n\n");

    if !spec.scheduler_name.is_empty() {
        s.push_str(&format!("// Scheduler: {}\n", spec.scheduler_name));
    }
    if !spec.notes.is_empty() {
        s.push_str("//\n// Generator notes:\n");
        for note in &spec.notes {
            s.push_str(&format!("// - {note}\n"));
        }
        s.push('\n');
    }

    s.push_str("use ktstr::workload::*;\n");
    s.push_str("use std::collections::BTreeSet;\n");
    s.push_str("use std::time::Duration;\n\n");

    s.push_str(&format!("pub fn {template_name}() -> WorkloadConfig {{\n"));
    s.push_str("    WorkloadConfig::default()\n");
    s.push_str(&format!(
        "        .workers({})\n",
        spec.config.num_workers
    ));
    s.push_str(&format!(
        "        .affinity({})\n",
        render_affinity(&spec.config.affinity)
    ));
    s.push_str(&format!(
        "        .work_type({})\n",
        render_work_type(&spec.config.work_type)
    ));
    s.push_str(&format!(
        "        .sched_policy({})\n",
        render_sched_policy(&spec.config.sched_policy)
    ));
    if spec.config.nice != 0 {
        s.push_str(&format!("        .nice({})\n", spec.config.nice));
    }
    s.push_str("}\n");

    if !spec.cgroup_hints.is_empty() {
        s.push_str("\n// Cgroup hints — apply at harness setup:\n");
        for h in &spec.cgroup_hints {
            s.push_str(&format!(
                "// {} (weight={:?}, mem_max={:?}, cpuset={:?})\n",
                h.path, h.cpu_weight, h.memory_max_bytes, h.cpuset_cpus
            ));
        }
    }

    s
}

/// Render a `#[ktstr_test]`-decorated function from a
/// [`ReproducerSpec`].
///
/// Wraps [`render_run_file_source`]'s body with the proc-macro
/// attribute and a `#[allow(unused)]` import block. Output is
/// drop-in to a Rust file under `tests/`.
#[allow(dead_code)]
pub fn render_ktstr_test_source(spec: &ReproducerSpec, template_name: &str) -> String {
    let body = render_run_file_source(spec, template_name);
    // Prefix the generated function with the ktstr_test attribute.
    // The attribute applies to functions returning `WorkloadConfig`;
    // body's `pub fn` already matches that shape.
    body.replace(
        &format!("pub fn {template_name}"),
        &format!("#[ktstr::ktstr_test]\npub fn {template_name}"),
    )
}

fn render_affinity(a: &ResolvedAffinity) -> String {
    match a {
        ResolvedAffinity::None => "ResolvedAffinity::None".into(),
        ResolvedAffinity::SingleCpu(c) => format!("ResolvedAffinity::SingleCpu({c})"),
        ResolvedAffinity::Fixed(set) => {
            let cpus: Vec<String> = set.iter().map(|c| c.to_string()).collect();
            format!(
                "ResolvedAffinity::Fixed(BTreeSet::from([{}]))",
                cpus.join(", ")
            )
        }
        ResolvedAffinity::Random { from, count } => {
            let cpus: Vec<String> = from.iter().map(|c| c.to_string()).collect();
            format!(
                "ResolvedAffinity::Random {{ from: BTreeSet::from([{}]), count: {} }}",
                cpus.join(", "),
                count
            )
        }
    }
}

fn render_work_type(w: &WorkType) -> String {
    match w {
        WorkType::SpinWait => "WorkType::SpinWait".into(),
        WorkType::YieldHeavy => "WorkType::YieldHeavy".into(),
        WorkType::Mixed => "WorkType::Mixed".into(),
        WorkType::IoSyncWrite => "WorkType::IoSyncWrite".into(),
        WorkType::IoRandRead => "WorkType::IoRandRead".into(),
        WorkType::IoConvoy => "WorkType::IoConvoy".into(),
        WorkType::Bursty { burst_ms, sleep_ms } => format!(
            "WorkType::Bursty {{ burst_ms: {burst_ms}, sleep_ms: {sleep_ms} }}"
        ),
        WorkType::PipeIo { burst_iters } => {
            format!("WorkType::PipeIo {{ burst_iters: {burst_iters} }}")
        }
        WorkType::FutexPingPong { spin_iters } => {
            format!("WorkType::FutexPingPong {{ spin_iters: {spin_iters} }}")
        }
        WorkType::CachePressure { size_kb, stride } => format!(
            "WorkType::CachePressure {{ size_kb: {size_kb}, stride: {stride} }}"
        ),
        // Other variants we don't currently project from fingerprint
        // hints — emit a placeholder that compiles with a TODO so
        // hand-editing is obvious.
        _ => "WorkType::SpinWait /* TODO: refine from capture */".into(),
    }
}

fn render_sched_policy(p: &SchedPolicy) -> String {
    match p {
        SchedPolicy::Normal => "SchedPolicy::Normal".into(),
        SchedPolicy::Batch => "SchedPolicy::Batch".into(),
        SchedPolicy::Idle => "SchedPolicy::Idle".into(),
        SchedPolicy::Fifo(prio) => format!("SchedPolicy::Fifo({prio})"),
        SchedPolicy::RoundRobin(prio) => format!("SchedPolicy::RoundRobin({prio})"),
        SchedPolicy::Deadline {
            runtime,
            deadline,
            period,
        } => format!(
            "SchedPolicy::Deadline {{ runtime: Duration::from_nanos({}), \
             deadline: Duration::from_nanos({}), period: Duration::from_nanos({}) }}",
            runtime.as_nanos(),
            deadline.as_nanos(),
            period.as_nanos(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::debug_capture::WorkloadGroupHint;

    /// Empty fingerprint → default WorkloadConfig + notes about
    /// every projection that fell back.
    #[test]
    fn generate_spec_empty_fingerprint() {
        let cap = DebugCapture::default();
        let spec = generate_spec(&cap);
        assert_eq!(spec.config.num_workers, 1);
        assert!(matches!(spec.config.affinity, ResolvedAffinity::None));
        assert!(matches!(spec.config.work_type, WorkType::SpinWait));
        assert!(spec.notes.iter().any(|n| n.contains("no workload groups")));
        assert!(spec.notes.iter().any(|n| n.contains("no work-type hint")));
    }

    /// Workload-group hint with thread_count=8 → num_workers=8.
    #[test]
    fn generate_spec_thread_count_to_workers() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.workload_groups = vec![WorkloadGroupHint {
            cgroup_path: "/test".into(),
            thread_count: 8,
            cpu_time_fraction: 0.5,
            wakeups_per_sec: 100.0,
        }];
        let spec = generate_spec(&cap);
        assert_eq!(spec.config.num_workers, 8);
    }

    /// AffinityHint::Exact{cpus} → ResolvedAffinity::Fixed(set).
    #[test]
    fn generate_spec_exact_affinity() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::Exact {
            cpus: vec![0, 1, 4, 5],
        }];
        let spec = generate_spec(&cap);
        match spec.config.affinity {
            ResolvedAffinity::Fixed(set) => {
                let v: Vec<usize> = set.into_iter().collect();
                assert_eq!(v, vec![0, 1, 4, 5]);
            }
            other => panic!("expected Fixed, got {other:?}"),
        }
    }

    /// WorkTypeHint::Bursty{b,s} → WorkType::Bursty{b,s} pass-through.
    #[test]
    fn generate_spec_bursty_passthrough() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::Bursty {
            burst_ms: 5,
            sleep_ms: 95,
        }];
        let spec = generate_spec(&cap);
        match spec.config.work_type {
            WorkType::Bursty { burst_ms, sleep_ms } => {
                assert_eq!(burst_ms, 5);
                assert_eq!(sleep_ms, 95);
            }
            other => panic!("expected Bursty, got {other:?}"),
        }
    }

    /// SchedPolicyHint::Fifo{prio} → SchedPolicy::Fifo(prio).
    #[test]
    fn generate_spec_fifo_priority() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.sched_policy_hints =
            vec![SchedPolicyHint::Fifo { priority: 50 }];
        let spec = generate_spec(&cap);
        match spec.config.sched_policy {
            SchedPolicy::Fifo(prio) => assert_eq!(prio, 50),
            other => panic!("expected Fifo, got {other:?}"),
        }
    }

    /// SchedPolicyHint::Other{nice} → Normal + nice value applied.
    #[test]
    fn generate_spec_nice_applied() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.sched_policy_hints =
            vec![SchedPolicyHint::Other { nice: 5 }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.sched_policy, SchedPolicy::Normal));
        assert_eq!(spec.config.nice, 5);
    }

    /// Multiple work-type hints → first wins, rest in notes.
    #[test]
    fn generate_spec_multiple_hints_first_wins() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints =
            vec![WorkTypeHint::SpinWait, WorkTypeHint::IoSyncWrite];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.work_type, WorkType::SpinWait));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("additional work-type hints"))
        );
    }

    /// `WorkTypeHint::IoRandRead` and `WorkTypeHint::IoConvoy`
    /// each project to the matching `WorkType::IoRandRead` /
    /// `WorkType::IoConvoy`. Pins the dedicated mapping the
    /// generator gained when the capture pipeline learned to
    /// distinguish IO modes — a regression that silently
    /// collapses either hint back to `IoSyncWrite` (the previous
    /// "absent by design" fallback) would surface here.
    #[test]
    fn generate_spec_maps_each_io_hint_directly() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::IoRandRead];
        let spec = generate_spec(&cap);
        assert!(
            matches!(spec.config.work_type, WorkType::IoRandRead),
            "IoRandRead hint must map to WorkType::IoRandRead, got {:?}",
            spec.config.work_type,
        );

        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::IoConvoy];
        let spec = generate_spec(&cap);
        assert!(
            matches!(spec.config.work_type, WorkType::IoConvoy),
            "IoConvoy hint must map to WorkType::IoConvoy, got {:?}",
            spec.config.work_type,
        );

        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::IoSyncWrite];
        let spec = generate_spec(&cap);
        assert!(
            matches!(spec.config.work_type, WorkType::IoSyncWrite),
            "IoSyncWrite hint must map to WorkType::IoSyncWrite, got {:?}",
            spec.config.work_type,
        );
    }

    /// Fingerprint gaps propagate to notes.
    #[test]
    fn generate_spec_propagates_gaps() {
        let mut cap = DebugCapture::default();
        cap.fingerprint
            .gaps
            .push("affinity hint backed by 1 sample".into());
        let spec = generate_spec(&cap);
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("affinity hint backed by 1 sample"))
        );
    }

    /// LlcAligned hint surfaces a note (no ResolvedAffinity mapping).
    #[test]
    fn generate_spec_llc_aligned_note() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::LlcAligned];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, ResolvedAffinity::None));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::LlcAligned"))
        );
    }

    /// `render_run_file_source` produces compilable-shape output
    /// containing the expected builder calls + import lines. Sets up
    /// a fingerprint that produces zero generator notes (single
    /// workload group + single work-type hint, no gaps or other
    /// hint vectors) so the test pins the unconditional skeleton —
    /// note rendering is conditional on `!spec.notes.is_empty()`
    /// and is covered by [`render_run_file_source_renders_notes`].
    #[test]
    fn render_run_file_source_basic_shape() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.workload_groups = vec![WorkloadGroupHint {
            cgroup_path: "/test".into(),
            thread_count: 4,
            cpu_time_fraction: 0.0,
            wakeups_per_sec: 0.0,
        }];
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::SpinWait];
        let spec = generate_spec(&cap);
        // Sanity-pin the no-notes precondition so a future change
        // that starts emitting notes for this shape lands here
        // first rather than in a flake.
        assert!(
            spec.notes.is_empty(),
            "basic-shape fingerprint must produce no notes; got {:?}",
            spec.notes,
        );

        let src = render_run_file_source(&spec, "regression_repro");

        assert!(src.contains("use ktstr::workload::*;"));
        assert!(src.contains("pub fn regression_repro"));
        assert!(src.contains(".workers(4)"));
        assert!(src.contains(".work_type(WorkType::SpinWait)"));
        // Notes are conditionally rendered — no notes here means
        // no "Generator notes:" comment block (verified by the
        // dedicated test).
        assert!(!src.contains("Generator notes:"));
    }

    /// When the generator emits any notes (e.g. from fingerprint
    /// gaps), `render_run_file_source` surfaces them under a
    /// `// Generator notes:` comment block prefixed by `// - `.
    /// Pins the conditional-rendering branch at L395-401 so a
    /// regression that drops the comment block surfaces here.
    #[test]
    fn render_run_file_source_renders_notes() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.gaps = vec!["test gap from fingerprint".into()];
        let spec = generate_spec(&cap);
        assert!(
            !spec.notes.is_empty(),
            "fingerprint gap must propagate to spec.notes",
        );

        let src = render_run_file_source(&spec, "with_notes");
        assert!(src.contains("Generator notes:"));
        assert!(src.contains("// - fingerprint gap: test gap from fingerprint"));
    }

    /// `render_ktstr_test_source` decorates the generated function
    /// with the proc-macro attribute.
    #[test]
    fn render_ktstr_test_source_has_attribute() {
        let cap = DebugCapture::default();
        let spec = generate_spec(&cap);
        let src = render_ktstr_test_source(&spec, "auto_repro");
        assert!(src.contains("#[ktstr::ktstr_test]"));
        assert!(src.contains("pub fn auto_repro"));
    }

    /// Cgroup hints render as comments at the bottom of generated
    /// source (the harness applies them at setup time, not via
    /// WorkloadConfig builder).
    #[test]
    fn render_run_file_source_includes_cgroup_hints_as_comments() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.cgroup_hints = vec![CgroupHint {
            path: "/system.slice/foo.service".into(),
            cpu_weight: Some(200),
            memory_max_bytes: Some(8 * 1024 * 1024 * 1024),
            cpuset_cpus: vec![0, 1, 2, 3],
            cpu_max_quota_us: None,
        }];
        let spec = generate_spec(&cap);
        let src = render_run_file_source(&spec, "with_cgroup");
        assert!(src.contains("Cgroup hints"));
        assert!(src.contains("/system.slice/foo.service"));
        assert!(src.contains("weight=Some(200)"));
    }

    /// Affinity render handles every ResolvedAffinity variant.
    #[test]
    fn render_affinity_all_variants() {
        assert_eq!(render_affinity(&ResolvedAffinity::None), "ResolvedAffinity::None");
        assert_eq!(
            render_affinity(&ResolvedAffinity::SingleCpu(3)),
            "ResolvedAffinity::SingleCpu(3)"
        );
        let fixed = ResolvedAffinity::Fixed(BTreeSet::from([0usize, 1, 2]));
        assert_eq!(
            render_affinity(&fixed),
            "ResolvedAffinity::Fixed(BTreeSet::from([0, 1, 2]))"
        );
        let random = ResolvedAffinity::Random {
            from: BTreeSet::from([0usize, 1]),
            count: 1,
        };
        assert_eq!(
            render_affinity(&random),
            "ResolvedAffinity::Random { from: BTreeSet::from([0, 1]), count: 1 }"
        );
    }
}
