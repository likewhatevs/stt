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
//! projections from capture data) and emit a test spec the framework
//! will execute. The generator is NOT a classifier — it does not
//! decide "this workload is locking-bound" or "this is a cache
//! pressure pathology"; it just maps observed shapes to primitive
//! types. Pathology classification is a separate, downstream
//! concern (the reader / LLM consuming `ktstr show / compare`).
//!
//! # Mapping table
//!
//! | fingerprint hint              | ktstr type                              |
//! |-------------------------------|-----------------------------------------|
//! | WorkloadGroupHint.thread_count| WorkloadConfig::num_workers             |
//! | AffinityHint::SingleCpu{cpus} | AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::SingleCpu when empty |
//! | AffinityHint::Exact{cpus}     | AffinityIntent::Exact(set)               |
//! | AffinityHint::Inherit         | AffinityIntent::Inherit                  |
//! | AffinityHint::LlcAligned{cpus}| AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::LlcAligned when empty |
//! | AffinityHint::CrossCgroup{cpus}| AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::CrossCgroup when empty |
//! | AffinityHint::SmtSiblingPair{cpus}| AffinityIntent::Exact(set) when cpus non-empty; AffinityIntent::SmtSiblingPair when empty |
//! | AffinityHint::RandomSubset { from, count: popcount-per-thread } | AffinityIntent::RandomSubset { from, count } when both non-empty; placeholder otherwise [†] |
//! | WorkTypeHint::SpinWait         | WorkType::SpinWait                       |
//! | WorkTypeHint::YieldHeavy      | WorkType::YieldHeavy                    |
//! | WorkTypeHint::Mixed           | WorkType::Mixed                         |
//! | WorkTypeHint::Bursty{b,s}     | WorkType::Bursty { burst_duration: b, sleep_duration: s } |
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
//! [†] The `RandomSubset` row emits an empty-pool / zero-count
//! placeholder that the spawn-time affinity gate REJECTS when the
//! producer did not record a resolved pool. The resulting spec is
//! not runnable as-is — hand-edit `from` to the actual CPU pool and
//! `count` to the desired sample size before running, or change to
//! `AffinityIntent::Inherit`. When the producer DID record a pool,
//! the generator emits a fully-populated `AffinityIntent::RandomSubset`
//! that the spawn-time gate accepts directly.
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
use crate::workload::{AffinityIntent, MemPolicy, MpolFlags, SchedPolicy, WorkType, WorkloadConfig, CloneMode};

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

/// Build the user-facing note attached when a topology-aware
/// [`AffinityHint`] is projected without a resolved CPU set. The 4
/// topology-aware variants (`SingleCpu`, `LlcAligned`, `CrossCgroup`,
/// `SmtSiblingPair`) share the same structure: name the variant,
/// describe what the scenario engine resolves at apply time, then
/// point the user at the concrete `AffinityIntent::Exact(...)` they
/// should hand-edit to if they want to spawn directly via
/// [`crate::workload::WorkloadHandle::spawn`] (which rejects the
/// topology-aware variants — they require scenario context the
/// spawn-time gate doesn't have).
fn topology_aware_note(
    variant: &str,
    engine_action: &str,
    hand_edit_target: &str,
) -> String {
    format!(
        "AffinityHint::{variant} observed without resolved CPUs; \
         emitting AffinityIntent::{variant} — the scenario engine \
         {engine_action} at apply time. Direct \
         WorkloadHandle::spawn rejects this variant (no topology \
         context); use the scenario engine or hand-edit to \
         AffinityIntent::Exact({hand_edit_target})"
    )
}

/// Build the note attached when a topology-aware [`AffinityHint`]
/// carried resolved CPUs and the generator collapsed it to
/// [`AffinityIntent::Exact`]. The note preserves the original
/// pattern classification (SingleCpu / LlcAligned / CrossCgroup /
/// SmtSiblingPair) so the consumer can see what the producer
/// observed before resolution, even though the emitted spec is a
/// flat `Exact`.
fn topology_resolved_note(variant: &str, cpus: &[u32]) -> String {
    format!(
        "AffinityHint::{variant} observed with resolved CPUs {cpus:?}; \
         emitting AffinityIntent::Exact directly so the spec runs \
         without scenario-engine resolution",
    )
}

/// Build a `BTreeSet<usize>` from a slice of `u32` CPU IDs. Centralises
/// the `u32 → usize` widening the `AffinityHint` payload requires
/// before it can populate an [`AffinityIntent::Exact`] / `RandomSubset`
/// pool.
fn cpus_to_set(cpus: &[u32]) -> BTreeSet<usize> {
    cpus.iter().map(|&c| c as usize).collect()
}

/// Resolve a topology-aware [`AffinityHint`] payload to an
/// [`AffinityIntent`] and append the matching note to `spec.notes`.
///
/// The 4 topology-aware variants (`SingleCpu`, `LlcAligned`,
/// `CrossCgroup`, `SmtSiblingPair`) share the same shape: an empty
/// `cpus` payload falls back to the matching topology-aware
/// `AffinityIntent` variant (resolved by the scenario engine at apply
/// time) plus a hand-edit note, while a non-empty payload collapses
/// to [`AffinityIntent::Exact`] containing the observed CPUs.
///
/// `variant` names the [`AffinityHint`] variant for the note text,
/// `topology_intent` is the matching topology-aware intent emitted on
/// the empty path, `engine_action` and `hand_edit_target` describe
/// the scenario engine's resolution and the user's hand-edit target
/// respectively (the strings appear in [`topology_aware_note`]).
fn map_topology_aware_affinity(
    cpus: &[u32],
    variant: &str,
    topology_intent: AffinityIntent,
    engine_action: &str,
    hand_edit_target: &str,
    spec: &mut ReproducerSpec,
) -> AffinityIntent {
    if cpus.is_empty() {
        spec.notes.push(topology_aware_note(
            variant,
            engine_action,
            hand_edit_target,
        ));
        topology_intent
    } else {
        spec.notes.push(topology_resolved_note(variant, cpus));
        AffinityIntent::Exact(cpus_to_set(cpus))
    }
}

fn map_affinity(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.affinity_hints.first() else {
        return;
    };
    spec.config.affinity = match primary {
        AffinityHint::Inherit => AffinityIntent::Inherit,
        AffinityHint::SingleCpu { cpus } => map_topology_aware_affinity(
            cpus,
            "SingleCpu",
            AffinityIntent::SingleCpu,
            "picks the concrete CPU from the cgroup's cpuset",
            "[cpu]",
            spec,
        ),
        AffinityHint::LlcAligned { cpus } => map_topology_aware_affinity(
            cpus,
            "LlcAligned",
            AffinityIntent::LlcAligned,
            "resolves the LLC mask from the cgroup's cpuset",
            "<llc cpus>",
            spec,
        ),
        AffinityHint::CrossCgroup { cpus } => map_topology_aware_affinity(
            cpus,
            "CrossCgroup",
            AffinityIntent::CrossCgroup,
            "expands to the full topology",
            "<all cpus>",
            spec,
        ),
        AffinityHint::SmtSiblingPair { cpus } => map_topology_aware_affinity(
            cpus,
            "SmtSiblingPair",
            AffinityIntent::SmtSiblingPair,
            "picks an SMT-sibling pair from the cgroup's effective cpuset, \
             or the full topology when no cpuset is active",
            "[sibling_a, sibling_b]",
            spec,
        ),
        AffinityHint::Exact { cpus } => AffinityIntent::Exact(cpus_to_set(cpus)),
        AffinityHint::RandomSubset { from, count } => {
            if from.is_empty() || *count == 0 {
                spec.notes.push(
                    "AffinityHint::RandomSubset observed without a \
                     resolved pool / count; emitting \
                     AffinityIntent::RandomSubset { from: empty, count: 0 } \
                     as a placeholder — the spawn-time affinity gate \
                     rejects empty-pool / zero-count RandomSubset, so \
                     this spec is NOT runnable as-is. Hand-edit `from` \
                     to the actual CPU pool and `count` to the desired \
                     sample size before running, or change to \
                     AffinityIntent::Inherit."
                        .into(),
                );
                AffinityIntent::RandomSubset {
                    from: BTreeSet::new(),
                    count: 0,
                }
            } else {
                spec.notes.push(format!(
                    "AffinityHint::RandomSubset observed with resolved \
                     pool {from:?} count={count}; emitting \
                     AffinityIntent::RandomSubset directly so the \
                     spawn-time affinity gate accepts it without \
                     hand-editing",
                ));
                AffinityIntent::RandomSubset {
                    from: cpus_to_set(from),
                    count: *count as usize,
                }
            }
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
        WorkTypeHint::Bursty {
            burst_duration,
            sleep_duration,
        } => WorkType::Bursty {
            burst_duration: *burst_duration,
            sleep_duration: *sleep_duration,
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

fn render_affinity(a: &AffinityIntent) -> String {
    match a {
        AffinityIntent::Inherit => "AffinityIntent::Inherit".into(),
        AffinityIntent::SingleCpu => "AffinityIntent::SingleCpu".into(),
        AffinityIntent::LlcAligned => "AffinityIntent::LlcAligned".into(),
        AffinityIntent::CrossCgroup => "AffinityIntent::CrossCgroup".into(),
        AffinityIntent::SmtSiblingPair => "AffinityIntent::SmtSiblingPair".into(),
        AffinityIntent::RandomSubset { from, count } => {
            let cpus: Vec<String> = from.iter().map(|c| c.to_string()).collect();
            format!(
                "AffinityIntent::RandomSubset {{ from: BTreeSet::from([{}]), count: {} }}",
                cpus.join(", "),
                count
            )
        }
        AffinityIntent::Exact(set) => {
            let cpus: Vec<String> = set.iter().map(|c| c.to_string()).collect();
            format!(
                "AffinityIntent::Exact(BTreeSet::from([{}]))",
                cpus.join(", ")
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
        WorkType::Bursty {
            burst_duration,
            sleep_duration,
        } => format!(
            "WorkType::Bursty {{ \
             burst_duration: Duration::from_millis({}), \
             sleep_duration: Duration::from_millis({}) \
             }}",
            burst_duration.as_millis(),
            sleep_duration.as_millis(),
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
        assert!(matches!(spec.config.affinity, AffinityIntent::Inherit));
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

    /// AffinityHint::Exact{cpus} → AffinityIntent::Exact(set).
    #[test]
    fn generate_spec_exact_affinity() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::Exact {
            cpus: vec![0, 1, 4, 5],
        }];
        let spec = generate_spec(&cap);
        match spec.config.affinity {
            AffinityIntent::Exact(set) => {
                let v: Vec<usize> = set.into_iter().collect();
                assert_eq!(v, vec![0, 1, 4, 5]);
            }
            other => panic!("expected Exact, got {other:?}"),
        }
    }

    /// `WorkTypeHint::Bursty {burst_duration, sleep_duration}`
    /// passes its `Duration` fields straight through to
    /// `WorkType::Bursty` in the hint→work-type mapping.
    #[test]
    fn generate_spec_bursty_passthrough() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::Bursty {
            burst_duration: Duration::from_millis(5),
            sleep_duration: Duration::from_millis(95),
        }];
        let spec = generate_spec(&cap);
        match spec.config.work_type {
            WorkType::Bursty {
                burst_duration,
                sleep_duration,
            } => {
                assert_eq!(burst_duration, Duration::from_millis(5));
                assert_eq!(sleep_duration, Duration::from_millis(95));
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

    /// Unresolved `LlcAligned` hint (empty `cpus`) emits
    /// [`AffinityIntent::LlcAligned`] and surfaces a note reminding
    /// the consumer that direct [`crate::workload::WorkloadHandle::spawn`]
    /// rejects this variant (the scenario engine resolves it from
    /// cgroup cpuset context). Pins the unresolved-fallback path of
    /// the topology-aware projection.
    #[test]
    fn generate_spec_llc_aligned_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::LlcAligned { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, AffinityIntent::LlcAligned));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::LlcAligned")
                    && n.contains("without resolved CPUs")),
            "unresolved LlcAligned must surface a topology-aware-fallback note: {:?}",
            spec.notes,
        );
    }

    /// Resolved `LlcAligned` hint (non-empty `cpus`) collapses to
    /// [`AffinityIntent::Exact`] containing those CPUs and surfaces
    /// a note that preserves the original pattern classification.
    /// The emitted spec is runnable directly via
    /// [`crate::workload::WorkloadHandle::spawn`] — no scenario-engine
    /// resolution required. Pins the resolved-data path that #401
    /// added.
    #[test]
    fn generate_spec_llc_aligned_resolved_emits_exact() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::LlcAligned {
            cpus: vec![0, 1, 2, 3],
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::Exact(set) => {
                let v: Vec<usize> = set.iter().copied().collect();
                assert_eq!(
                    v,
                    vec![0, 1, 2, 3],
                    "resolved LlcAligned must collapse to Exact with the observed CPUs: got {v:?}",
                );
            }
            other => panic!("expected Exact, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::LlcAligned")
                    && n.contains("with resolved CPUs")),
            "resolved LlcAligned must surface a resolved-collapse note: {:?}",
            spec.notes,
        );
    }

    /// Resolved `SingleCpu` hint (non-empty `cpus`) collapses to
    /// [`AffinityIntent::Exact`]. Mirrors the LlcAligned resolved
    /// case for the SingleCpu pattern — the producer recorded the
    /// concrete CPU(s) and the generator emits a runnable spec.
    #[test]
    fn generate_spec_single_cpu_resolved_emits_exact() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::SingleCpu { cpus: vec![7] }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::Exact(set) => {
                let v: Vec<usize> = set.iter().copied().collect();
                assert_eq!(v, vec![7]);
            }
            other => panic!("expected Exact, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::SingleCpu")
                    && n.contains("with resolved CPUs")),
            "resolved SingleCpu must surface a resolved-collapse note: {:?}",
            spec.notes,
        );
    }

    /// Unresolved `SingleCpu` hint falls back to
    /// [`AffinityIntent::SingleCpu`] with a hand-edit note. Pins the
    /// fallback path so a regression that drops the unresolved
    /// branch surfaces here.
    #[test]
    fn generate_spec_single_cpu_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::SingleCpu { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, AffinityIntent::SingleCpu));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::SingleCpu")
                    && n.contains("without resolved CPUs")),
        );
    }

    /// Resolved `CrossCgroup` hint collapses to
    /// [`AffinityIntent::Exact`]. The producer recorded the
    /// observed cross-cgroup span and the generator emits it
    /// directly.
    #[test]
    fn generate_spec_cross_cgroup_resolved_emits_exact() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::CrossCgroup {
            cpus: vec![2, 4, 6, 8],
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::Exact(set) => {
                let v: Vec<usize> = set.iter().copied().collect();
                assert_eq!(v, vec![2, 4, 6, 8]);
            }
            other => panic!("expected Exact, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::CrossCgroup")
                    && n.contains("with resolved CPUs")),
        );
    }

    /// Unresolved `CrossCgroup` hint (empty `cpus`) emits
    /// [`AffinityIntent::CrossCgroup`] and surfaces a note reminding
    /// the consumer that direct [`crate::workload::WorkloadHandle::spawn`]
    /// rejects this variant (the scenario engine expands it to the
    /// full topology). Pins the unresolved-fallback path of the
    /// CrossCgroup topology-aware projection.
    #[test]
    fn generate_spec_cross_cgroup_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints =
            vec![AffinityHint::CrossCgroup { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, AffinityIntent::CrossCgroup));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::CrossCgroup")
                    && n.contains("without resolved CPUs")),
            "unresolved CrossCgroup must surface a topology-aware-fallback note: {:?}",
            spec.notes,
        );
    }

    /// Resolved `RandomSubset` hint (non-empty `from`, non-zero
    /// `count`) emits [`AffinityIntent::RandomSubset`] with the
    /// resolved pool and count. The spawn-time gate accepts this
    /// shape directly — no hand-editing required.
    #[test]
    fn generate_spec_random_subset_resolved_emits_populated() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: vec![0, 1, 2, 3, 4, 5],
            count: 3,
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::RandomSubset { from, count } => {
                let v: Vec<usize> = from.iter().copied().collect();
                assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
                assert_eq!(*count, 3);
            }
            other => panic!("expected populated RandomSubset, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::RandomSubset")
                    && n.contains("with resolved pool")),
        );
    }

    /// Unresolved `RandomSubset` hint (empty `from` or zero `count`)
    /// emits the empty placeholder and surfaces the hand-edit note.
    /// Pins the legacy fallback for producers that classify the
    /// pattern without recording the pool.
    #[test]
    fn generate_spec_random_subset_unresolved_emits_placeholder() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: Vec::new(),
            count: 0,
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::RandomSubset { from, count } => {
                assert!(from.is_empty(), "unresolved RandomSubset must emit empty pool");
                assert_eq!(*count, 0);
            }
            other => panic!("expected placeholder RandomSubset, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::RandomSubset")
                    && n.contains("without a resolved pool")),
        );
    }

    /// Resolved `SmtSiblingPair` hint (non-empty `cpus`) collapses to
    /// [`AffinityIntent::Exact`]. The producer recorded the observed
    /// SMT sibling pair and the generator emits a runnable spec.
    #[test]
    fn generate_spec_smt_sibling_pair_resolved_emits_exact() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints =
            vec![AffinityHint::SmtSiblingPair { cpus: vec![2, 3] }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::Exact(set) => {
                let v: Vec<usize> = set.iter().copied().collect();
                assert_eq!(v, vec![2, 3]);
            }
            other => panic!("expected Exact, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::SmtSiblingPair")
                    && n.contains("with resolved CPUs")),
            "resolved SmtSiblingPair must surface a resolved-collapse note: {:?}",
            spec.notes,
        );
    }

    /// Unresolved `SmtSiblingPair` hint (empty `cpus`) emits
    /// [`AffinityIntent::SmtSiblingPair`] and surfaces a note
    /// reminding the consumer that direct
    /// [`crate::workload::WorkloadHandle::spawn`] rejects this
    /// variant (the scenario engine resolves it from the cgroup's
    /// cpuset). Pins the unresolved-fallback path of the
    /// SmtSiblingPair topology-aware projection.
    #[test]
    fn generate_spec_smt_sibling_pair_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints =
            vec![AffinityHint::SmtSiblingPair { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(
            spec.config.affinity,
            AffinityIntent::SmtSiblingPair
        ));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.contains("AffinityHint::SmtSiblingPair")
                    && n.contains("without resolved CPUs")),
            "unresolved SmtSiblingPair must surface a topology-aware-fallback note: {:?}",
            spec.notes,
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

    /// Affinity render handles every AffinityIntent variant.
    #[test]
    fn render_affinity_all_variants() {
        assert_eq!(
            render_affinity(&AffinityIntent::Inherit),
            "AffinityIntent::Inherit"
        );
        assert_eq!(
            render_affinity(&AffinityIntent::SingleCpu),
            "AffinityIntent::SingleCpu"
        );
        assert_eq!(
            render_affinity(&AffinityIntent::LlcAligned),
            "AffinityIntent::LlcAligned"
        );
        assert_eq!(
            render_affinity(&AffinityIntent::CrossCgroup),
            "AffinityIntent::CrossCgroup"
        );
        assert_eq!(
            render_affinity(&AffinityIntent::SmtSiblingPair),
            "AffinityIntent::SmtSiblingPair"
        );
        let random = AffinityIntent::RandomSubset {
            from: BTreeSet::from([0usize, 1, 2, 3]),
            count: 2,
        };
        assert_eq!(
            render_affinity(&random),
            "AffinityIntent::RandomSubset { from: BTreeSet::from([0, 1, 2, 3]), count: 2 }"
        );
        let exact = AffinityIntent::Exact(BTreeSet::from([0usize, 1, 2]));
        assert_eq!(
            render_affinity(&exact),
            "AffinityIntent::Exact(BTreeSet::from([0, 1, 2]))"
        );
    }
}
