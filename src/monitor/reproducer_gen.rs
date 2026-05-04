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
//! | AffinityHint::Exact{cpus}     | AffinityIntent::Exact(set) [‡]           |
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
//! [‡] Empty `cpus` emits a hand-edit-required note alongside the
//! placeholder `AffinityIntent::Exact(empty)` — the spawn-time
//! affinity gate rejects an empty Exact set, so the rendered spec
//! is NOT runnable as-is until the user pastes in the observed CPUs
//! (or switches to `AffinityIntent::Inherit`). Non-empty `cpus`
//! emits a resolved-collapse note and the runnable
//! `AffinityIntent::Exact` directly.
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
use crate::workload::{AffinityIntent, SchedPolicy, WorkType, WorkloadConfig};

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
    ///
    /// Each entry is a typed [`ReproducerNote`] whose kind
    /// (`Informational` / `Resolved` / `UnresolvedAffinity` /
    /// `UnmappedWorkType`) drives [`Self::is_runnable`] and
    /// [`Self::unresolved_count`] without substring matching.
    pub notes: Vec<ReproducerNote>,
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

/// Diagnostic note attached to a [`ReproducerSpec`].
///
/// The variant classifies the note's effect on runnability:
/// `UnresolvedAffinity` and `UnmappedWorkType` are the only kinds
/// that block [`ReproducerSpec::is_runnable`]; `Informational` and
/// `Resolved` are zero-cost surface for the generated source's
/// comment block. Replaced an earlier substring-marker scheme that
/// probed the rendered text — see [`Self::is_unresolved`] for the
/// runnability filter.
///
/// Each variant carries the rendered text the generator emits — the
/// text format is unchanged from the prior `Vec<String>` shape, so
/// tests that match on note wording continue to work via
/// [`Self::message`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ReproducerNote {
    /// Default-projection or fingerprint-gap context. Does NOT
    /// affect runnability. Examples: "no workload groups in
    /// fingerprint — defaulting num_workers=1", "additional
    /// affinity hints not modeled: ...", "fingerprint gap: ...",
    /// "SchedPolicyHint::Ext observed; framework defaults to scx
    /// routing — no policy override emitted".
    Informational(String),
    /// A topology-aware [`AffinityHint`] carried resolved CPUs and
    /// the generator collapsed it to [`AffinityIntent::Exact`], OR
    /// a populated `RandomSubset` whose pool the spawn-time gate
    /// accepts directly. Does NOT block runnability — the rendered
    /// spec runs without scenario-engine resolution.
    Resolved(String),
    /// An affinity hint whose projection produces a placeholder the
    /// spawn-time affinity gate REJECTS. Spec is NOT runnable until
    /// the user hand-edits the rendered source (or switches to
    /// `AffinityIntent::Inherit`). Counted by
    /// [`ReproducerSpec::unresolved_count`].
    UnresolvedAffinity(String),
    /// The projected [`WorkType`] is one [`render_work_type`]
    /// dispatches to [`render_work_type_todo`] (renders as
    /// `WorkType::SpinWait /* TODO: ... */`). Spec is NOT runnable
    /// until the user replaces the placeholder with a real builder
    /// call. Counted by [`ReproducerSpec::unresolved_count`].
    UnmappedWorkType(String),
}

impl ReproducerNote {
    /// The rendered note text — the same string the prior
    /// `Vec<String>` field carried directly. Drives the comment
    /// block in [`render_run_file_source`] and lets test assertions
    /// match wording through `note.message().contains("...")`.
    #[allow(dead_code)]
    pub fn message(&self) -> &str {
        match self {
            ReproducerNote::Informational(s)
            | ReproducerNote::Resolved(s)
            | ReproducerNote::UnresolvedAffinity(s)
            | ReproducerNote::UnmappedWorkType(s) => s,
        }
    }

    /// `true` for the kinds [`ReproducerSpec::unresolved_count`]
    /// counts — `UnresolvedAffinity` and `UnmappedWorkType`. Both
    /// signal "the rendered spec is NOT runnable until the user
    /// hand-edits".
    fn is_unresolved(&self) -> bool {
        matches!(
            self,
            ReproducerNote::UnresolvedAffinity(_) | ReproducerNote::UnmappedWorkType(_)
        )
    }
}

impl std::fmt::Display for ReproducerNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl ReproducerSpec {
    /// Returns `true` when the spec is runnable as-is (the spawn-time
    /// affinity gate accepts it without hand-editing AND the
    /// rendered source carries no `WorkType::SpinWait /* TODO: ... */`
    /// placeholder). `false` means at least one hand-edit-required
    /// note is present, or [`Self::config`]'s `work_type` is one the
    /// generator does not know how to render as a runnable builder
    /// call yet.
    ///
    /// The check is the union of two signals:
    ///
    /// 1. Hand-edit-required notes — counted via
    ///    [`Self::unresolved_count`], which sums every
    ///    [`ReproducerNote::UnresolvedAffinity`] and
    ///    [`ReproducerNote::UnmappedWorkType`] entry by enum kind
    ///    (no substring matching).
    /// 2. Direct check on [`Self::config`]'s `work_type` —
    ///    [`is_unmapped_work_type`] returns `true` for variants that
    ///    [`render_work_type`] dispatches to [`render_work_type_todo`].
    ///    This catches specs constructed without going through
    ///    [`generate_spec`] / [`map_work_type`] (e.g. callers that set
    ///    `config.work_type` directly), so a manually-built spec with
    ///    `WorkType::CacheYield { .. }` is correctly classified as
    ///    NOT runnable even though no unresolved note was pushed.
    ///
    /// `Informational` and `Resolved` notes do not affect the
    /// outcome — they are surfaced in the rendered comment block but
    /// carry no runnability-blocking semantic.
    #[allow(dead_code)]
    pub fn is_runnable(&self) -> bool {
        self.unresolved_count() == 0 && !is_unmapped_work_type(&self.config.work_type)
    }

    /// Number of hand-edit-required notes in [`Self::notes`]. Counts
    /// every [`ReproducerNote::UnresolvedAffinity`] (affinity
    /// hand-edit prompts) and [`ReproducerNote::UnmappedWorkType`]
    /// (work-type TODO prompts); useful for surfacing "this
    /// reproducer needs N edits" messaging in `cargo ktstr` tooling.
    ///
    /// Does NOT include the `is_unmapped_work_type` direct-config
    /// signal that [`Self::is_runnable`] folds in — that path
    /// catches manually-constructed specs without notes and is
    /// outside the "count of edit prompts" semantic.
    #[allow(dead_code)]
    pub fn unresolved_count(&self) -> usize {
        self.notes.iter().filter(|n| n.is_unresolved()).count()
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
        spec.notes
            .push(ReproducerNote::Informational(format!(
                "fingerprint gap: {gap}"
            )));
    }

    spec
}

fn failure_scheduler_name(capture: &DebugCapture) -> String {
    // Always returns the empty string. The current
    // [`FailureDumpReport`] shape does not carry a scheduler-name
    // field, and [`DebugCapture`] itself does not duplicate it. The
    // returned empty string signals to [`render_run_file_source`]
    // that the rendered source should not emit a `// Scheduler:`
    // comment line — the consumer fills in the scheduler name when
    // pasting the generated reproducer into a test file.
    let _ = capture;
    String::new()
}

fn map_workload_groups(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.workload_groups.first() else {
        spec.notes.push(ReproducerNote::Informational(
            "no workload groups in fingerprint — defaulting num_workers=1".into(),
        ));
        return;
    };
    spec.config.num_workers = primary.thread_count.max(1) as usize;
    push_extras_note(
        &mut spec.notes,
        "additional workload groups not modeled in primary spec",
        fp.workload_groups
            .iter()
            .skip(1)
            .map(|g: &WorkloadGroupHint| format!("{} ({} threads)", g.cgroup_path, g.thread_count)),
    );
}

/// Emit an "additional X observed: ..." note when the fingerprint
/// carries more than one hint of a kind. Centralises the pattern
/// shared by [`map_workload_groups`], [`map_affinity`],
/// [`map_work_type`], and [`map_sched_policy`]: skip-first-then-render
/// the secondary entries, comma-join them, and only push when the
/// iterator yields anything.
///
/// `header` is the lead phrase before the colon (e.g.
/// `"additional affinity hints not modeled"`). `entries` is an
/// iterator of pre-rendered `String` descriptions for each
/// secondary entry. The pushed note is classified
/// [`ReproducerNote::Informational`] — secondary-hint enumeration
/// is descriptive context, never a runnability blocker.
fn push_extras_note(
    notes: &mut Vec<ReproducerNote>,
    header: &str,
    entries: impl Iterator<Item = String>,
) {
    let alts: Vec<String> = entries.collect();
    if !alts.is_empty() {
        notes.push(ReproducerNote::Informational(format!(
            "{header}: {}",
            alts.join(", ")
        )));
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
///
/// `hand_edit_target` carries paste-ready Rust (an
/// `AffinityIntent::exact(...)` call with angle-bracket placeholders)
/// so the generated note can be copied into a test file with minimal
/// editing.
fn topology_aware_note(variant: &str, engine_action: &str, hand_edit_target: &str) -> String {
    // The pushed note is classified
    // [`ReproducerNote::UnresolvedAffinity`] at the call site
    // ([`map_topology_aware_affinity`]); the typed variant — not
    // the wording — is what
    // [`ReproducerSpec::unresolved_count`] uses to count
    // hand-edit-required notes. The "spawn-time affinity gate
    // rejects" wording survives so existing reproducer-output
    // consumers see the same human-facing diagnostic.
    format!(
        "AffinityHint::{variant} observed without resolved CPUs; \
         emitting AffinityIntent::{variant} — the scenario engine \
         {engine_action} at apply time. The spawn-time affinity gate \
         rejects this variant (no topology context); use the \
         scenario engine or hand-edit to {hand_edit_target}"
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
        spec.notes.push(ReproducerNote::UnresolvedAffinity(
            topology_aware_note(variant, engine_action, hand_edit_target),
        ));
        topology_intent
    } else {
        spec.notes
            .push(ReproducerNote::Resolved(topology_resolved_note(
                variant, cpus,
            )));
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
            "AffinityIntent::exact([<cpu>])",
            spec,
        ),
        AffinityHint::LlcAligned { cpus } => map_topology_aware_affinity(
            cpus,
            "LlcAligned",
            AffinityIntent::LlcAligned,
            "resolves the LLC mask from the cgroup's cpuset",
            "AffinityIntent::exact([<llc_cpu_0>, <llc_cpu_1>, ...])",
            spec,
        ),
        AffinityHint::CrossCgroup { cpus } => map_topology_aware_affinity(
            cpus,
            "CrossCgroup",
            AffinityIntent::CrossCgroup,
            "expands to the full topology",
            "AffinityIntent::exact([<cpu_0>, <cpu_1>, ...])",
            spec,
        ),
        AffinityHint::SmtSiblingPair { cpus } => map_topology_aware_affinity(
            cpus,
            "SmtSiblingPair",
            AffinityIntent::SmtSiblingPair,
            "picks an SMT-sibling pair from the cgroup's effective cpuset, \
             or the full topology when no cpuset is active",
            "AffinityIntent::exact([<sibling_a>, <sibling_b>])",
            spec,
        ),
        AffinityHint::Exact { cpus } => {
            // Empty `cpus` produces an empty Exact set that the
            // spawn-time affinity gate rejects. Emit a note so the
            // reproducer surface is consistent — every other arm
            // pushes a note when it lands a placeholder, and the
            // resolved arms push a `topology_resolved_note`. An
            // empty Exact is the malformed shape; a populated Exact
            // is the runnable shape.
            if cpus.is_empty() {
                spec.notes.push(ReproducerNote::UnresolvedAffinity(
                    "AffinityHint::Exact observed with no CPUs; emitting \
                     AffinityIntent::Exact(empty) — the spawn-time \
                     affinity gate rejects an empty Exact set, so this \
                     spec is NOT runnable as-is. Hand-edit to \
                     AffinityIntent::exact([<cpu_0>, <cpu_1>, ...]) \
                     with the observed CPUs, or change to \
                     AffinityIntent::Inherit."
                        .into(),
                ));
            } else {
                spec.notes
                    .push(ReproducerNote::Resolved(topology_resolved_note(
                        "Exact", cpus,
                    )));
            }
            AffinityIntent::Exact(cpus_to_set(cpus))
        }
        AffinityHint::RandomSubset { from, count } => {
            if from.is_empty() || *count == 0 {
                spec.notes.push(ReproducerNote::UnresolvedAffinity(
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
                ));
                AffinityIntent::RandomSubset {
                    from: BTreeSet::new(),
                    count: 0,
                }
            } else {
                spec.notes.push(ReproducerNote::Resolved(format!(
                    "AffinityHint::RandomSubset observed with resolved \
                     pool {from:?} count={count}; emitting \
                     AffinityIntent::RandomSubset directly so the \
                     spawn-time affinity gate accepts it without \
                     hand-editing",
                )));
                AffinityIntent::RandomSubset {
                    from: cpus_to_set(from),
                    count: *count as usize,
                }
            }
        }
    };

    push_extras_note(
        &mut spec.notes,
        "additional affinity hints not modeled",
        fp.affinity_hints.iter().skip(1).map(|a| format!("{a:?}")),
    );
}

fn map_work_type(fp: &WorkloadFingerprint, spec: &mut ReproducerSpec) {
    let Some(primary) = fp.work_type_hints.first() else {
        spec.notes.push(ReproducerNote::Informational(
            "no work-type hint in fingerprint — defaulting to \
             WorkType::SpinWait"
                .into(),
        ));
        return;
    };
    let work_type = match primary {
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
    record_work_type(work_type, spec);

    push_extras_note(
        &mut spec.notes,
        "additional work-type hints observed",
        fp.work_type_hints.iter().skip(1).map(|w| format!("{w:?}")),
    );
}

/// Assign `work_type` to `spec.config.work_type` and push a
/// [`ReproducerNote::UnmappedWorkType`] note when the assigned variant
/// is one [`render_work_type`] dispatches to [`render_work_type_todo`].
///
/// Extracted from [`map_work_type`] so the unmapped-projection branch
/// has a production code path that tests can drive directly. Calling
/// this with `WorkType::ForkExit` (or any other variant
/// [`is_unmapped_work_type`] returns `true` for) exercises the
/// safety-net branch end-to-end — the test is no longer a manual
/// re-construction of the same logic.
///
/// No current [`WorkTypeHint`] variant projects to an unmapped
/// [`WorkType`], so the unmapped branch is reachable today only via
/// this helper. When a future hint variant lands on a TODO arm the
/// existing call site in [`map_work_type`] starts firing the branch
/// automatically — no duplicated logic to keep in sync.
fn record_work_type(work_type: WorkType, spec: &mut ReproducerSpec) {
    spec.config.work_type = work_type;
    if is_unmapped_work_type(&spec.config.work_type) {
        spec.notes.push(ReproducerNote::UnmappedWorkType(format!(
            "no fingerprint mapping for WorkType::{:?} — \
             render_run_file_source emits a TODO-decorated \
             SpinWait placeholder; hand-edit the rendered source to \
             a real builder call before running",
            spec.config.work_type,
        )));
    }
}

/// Return `true` when `w` is a [`WorkType`] variant that
/// [`render_work_type`] dispatches to [`render_work_type_todo`] (i.e.
/// renders as `WorkType::SpinWait /* TODO: ... */` because no
/// fingerprint mapping exists yet). Mirrors the runnable / TODO split
/// in [`render_work_type`] one-for-one — a new variant added there
/// must be classified here in the same pass.
///
/// Used by [`ReproducerSpec::is_runnable`] and [`map_work_type`] to
/// flag specs whose `work_type` cannot be rendered as a real builder
/// call. The runnable arm returns `false`; every TODO arm returns
/// `true`.
fn is_unmapped_work_type(w: &WorkType) -> bool {
    match w {
        // Variants the projection layer maps from a fingerprint hint
        // — render as runnable builder calls in [`render_work_type`].
        WorkType::SpinWait
        | WorkType::YieldHeavy
        | WorkType::Mixed
        | WorkType::IoSyncWrite
        | WorkType::IoRandRead
        | WorkType::IoConvoy
        | WorkType::Bursty { .. }
        | WorkType::PipeIo { .. }
        | WorkType::FutexPingPong { .. }
        | WorkType::CachePressure { .. } => false,
        // Variants no fingerprint hint currently projects to —
        // [`render_work_type`] dispatches each of these to
        // [`render_work_type_todo`]. Adding one to the runnable arm
        // above MUST also flip this match in lock-step.
        WorkType::CacheYield { .. }
        | WorkType::CachePipe { .. }
        | WorkType::FutexFanOut { .. }
        | WorkType::Sequence { .. }
        | WorkType::ForkExit
        | WorkType::NiceSweep
        | WorkType::AffinityChurn { .. }
        | WorkType::PolicyChurn { .. }
        | WorkType::FanOutCompute { .. }
        | WorkType::PageFaultChurn { .. }
        | WorkType::MutexContention { .. }
        | WorkType::Custom { .. }
        | WorkType::ThunderingHerd { .. }
        | WorkType::PriorityInversion { .. }
        | WorkType::ProducerConsumerImbalance { .. }
        | WorkType::RtStarvation { .. }
        | WorkType::AsymmetricWaker { .. }
        | WorkType::WakeChain { .. }
        | WorkType::NumaWorkingSetSweep { .. }
        | WorkType::CgroupChurn { .. }
        | WorkType::SignalStorm { .. }
        | WorkType::PreemptStorm { .. }
        | WorkType::EpollStorm { .. }
        | WorkType::NumaMigrationChurn { .. }
        | WorkType::IdleChurn { .. }
        | WorkType::AluHot { .. }
        | WorkType::SmtSiblingSpin
        | WorkType::IpcVariance { .. } => true,
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
            spec.notes.push(ReproducerNote::Informational(
                "SchedPolicyHint::Ext observed; framework defaults to \
                 scx routing — no policy override emitted"
                    .into(),
            ));
        }
    }

    push_extras_note(
        &mut spec.notes,
        "additional sched-policy hints observed",
        fp.sched_policy_hints
            .iter()
            .skip(1)
            .map(|s| format!("{s:?}")),
    );
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
    // Build from `WorkloadConfig::default()` and overlay every
    // captured value explicitly. Each `.workers/.affinity/...` call
    // pins the captured value directly, so the rendered spec runs
    // identically even if the upstream default changes — the
    // reproducer is independent of the host's `WorkloadConfig`
    // defaults at render time.
    s.push_str("    WorkloadConfig::default()\n");
    s.push_str(&format!("        .workers({})\n", spec.config.num_workers));
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
    // Always emit `.nice(...)` so the rendered config does not
    // silently inherit a future-changed `WorkloadConfig::nice`
    // default.
    s.push_str(&format!("        .nice({})\n", spec.config.nice));
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

/// Render a [`WorkType`] back to Rust source. Exhaustive match: a
/// new variant added to [`WorkType`] in `crate::workload` is a
/// compile error here, forcing the reproducer generator to make a
/// deliberate decision about how to render it (either a real
/// builder call or a `TODO` placeholder via
/// [`render_work_type_todo`]). The exhaustive form prevents the
/// "silent collapse to SpinWait" failure mode of the previous
/// wildcard arm.
fn render_work_type(w: &WorkType) -> String {
    match w {
        // Variants the projection layer maps from a fingerprint hint
        // — render them as runnable builder calls.
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
        WorkType::CachePressure { size_kb, stride } => {
            format!("WorkType::CachePressure {{ size_kb: {size_kb}, stride: {stride} }}")
        }
        // Variants that no fingerprint hint currently projects to.
        // Each produces an explicit TODO placeholder so the rendered
        // source compiles, surfaces the hand-edit requirement, and
        // names the unmapped variant. When a future
        // `WorkTypeHint::*` variant projects to one of these, move
        // the corresponding arm above this comment with a real
        // builder call.
        WorkType::CacheYield { .. } => render_work_type_todo("CacheYield"),
        WorkType::CachePipe { .. } => render_work_type_todo("CachePipe"),
        WorkType::FutexFanOut { .. } => render_work_type_todo("FutexFanOut"),
        WorkType::Sequence { .. } => render_work_type_todo("Sequence"),
        WorkType::ForkExit => render_work_type_todo("ForkExit"),
        WorkType::NiceSweep => render_work_type_todo("NiceSweep"),
        WorkType::AffinityChurn { .. } => render_work_type_todo("AffinityChurn"),
        WorkType::PolicyChurn { .. } => render_work_type_todo("PolicyChurn"),
        WorkType::FanOutCompute { .. } => render_work_type_todo("FanOutCompute"),
        WorkType::PageFaultChurn { .. } => render_work_type_todo("PageFaultChurn"),
        WorkType::MutexContention { .. } => render_work_type_todo("MutexContention"),
        WorkType::Custom { .. } => render_work_type_todo("Custom"),
        WorkType::ThunderingHerd { .. } => render_work_type_todo("ThunderingHerd"),
        WorkType::PriorityInversion { .. } => render_work_type_todo("PriorityInversion"),
        WorkType::ProducerConsumerImbalance { .. } => {
            render_work_type_todo("ProducerConsumerImbalance")
        }
        WorkType::RtStarvation { .. } => render_work_type_todo("RtStarvation"),
        WorkType::AsymmetricWaker { .. } => render_work_type_todo("AsymmetricWaker"),
        WorkType::WakeChain { .. } => render_work_type_todo("WakeChain"),
        WorkType::NumaWorkingSetSweep { .. } => render_work_type_todo("NumaWorkingSetSweep"),
        WorkType::CgroupChurn { .. } => render_work_type_todo("CgroupChurn"),
        WorkType::SignalStorm { .. } => render_work_type_todo("SignalStorm"),
        WorkType::PreemptStorm { .. } => render_work_type_todo("PreemptStorm"),
        WorkType::EpollStorm { .. } => render_work_type_todo("EpollStorm"),
        WorkType::NumaMigrationChurn { .. } => render_work_type_todo("NumaMigrationChurn"),
        WorkType::IdleChurn { .. } => render_work_type_todo("IdleChurn"),
        WorkType::AluHot { .. } => render_work_type_todo("AluHot"),
        WorkType::SmtSiblingSpin => render_work_type_todo("SmtSiblingSpin"),
        WorkType::IpcVariance { .. } => render_work_type_todo("IpcVariance"),
    }
}

/// Render a placeholder `WorkType` builder call for a variant the
/// projection layer doesn't yet know how to translate from a
/// fingerprint hint. Output names the variant explicitly so the
/// hand-edit prompt is unambiguous, avoiding the silent
/// collapse-to-SpinWait failure mode the previous-generation
/// wildcard arm produced — every unmapped variant now resolves
/// through this fn with its own variant-named TODO placeholder.
fn render_work_type_todo(variant: &str) -> String {
    format!(
        "WorkType::SpinWait /* TODO: no fingerprint mapping for \
         WorkType::{variant} — refine from capture */"
    )
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
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("no workload groups"))
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("no work-type hint"))
        );
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

    /// AffinityHint::Exact{cpus} → AffinityIntent::Exact(set), and
    /// the populated path now emits a resolved-collapse note (the
    /// Exact branch is the only one that previously left
    /// `spec.notes` empty — fixed alongside #465 to match every
    /// other arm's note-emission behavior).
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
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::Exact") && n.message().contains("with resolved CPUs")),
            "populated Exact must emit a resolved-collapse note for surface consistency: {:?}",
            spec.notes,
        );
    }

    /// Empty `AffinityHint::Exact { cpus: vec![] }` produces an
    /// empty `AffinityIntent::Exact` set that the spawn-time
    /// affinity gate rejects. The mapper must surface a
    /// hand-edit-required note pointing at paste-ready Rust
    /// (`AffinityIntent::exact([<cpu_0>, <cpu_1>, ...])`) so the
    /// reproducer surface mirrors the topology-aware variants'
    /// unresolved branch instead of silently producing a malformed
    /// spec. Pins the asymmetry doc on
    /// [`crate::monitor::debug_capture::AffinityHint::exact`].
    #[test]
    fn generate_spec_exact_empty_emits_unresolved_note() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::Exact { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::Exact(set) => {
                assert!(
                    set.is_empty(),
                    "empty Exact must propagate through to AffinityIntent: {set:?}"
                );
            }
            other => panic!("expected empty Exact, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::Exact") && n.message().contains("no CPUs")),
            "empty Exact must surface a hand-edit-required note: {:?}",
            spec.notes,
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityIntent::exact([<cpu_0>, <cpu_1>, ...])")),
            "empty Exact note must include paste-ready Rust hand-edit target: {:?}",
            spec.notes,
        );
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
        cap.fingerprint.sched_policy_hints = vec![SchedPolicyHint::Fifo { priority: 50 }];
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
        cap.fingerprint.sched_policy_hints = vec![SchedPolicyHint::Other { nice: 5 }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.sched_policy, SchedPolicy::Normal));
        assert_eq!(spec.config.nice, 5);
    }

    /// Multiple work-type hints → first wins, rest in notes.
    #[test]
    fn generate_spec_multiple_hints_first_wins() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::SpinWait, WorkTypeHint::IoSyncWrite];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.work_type, WorkType::SpinWait));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("additional work-type hints"))
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
                .any(|n| n.message().contains("affinity hint backed by 1 sample"))
        );
    }

    /// Unresolved `LlcAligned` hint (empty `cpus`) emits
    /// [`AffinityIntent::LlcAligned`] and surfaces a note reminding
    /// the consumer that direct [`crate::workload::WorkloadHandle::spawn`]
    /// rejects this variant (the scenario engine resolves it from
    /// cgroup cpuset context). Pins the unresolved-fallback path of
    /// the topology-aware projection. The hand-edit target must be
    /// paste-ready Rust (`AffinityIntent::exact(...)`).
    #[test]
    fn generate_spec_llc_aligned_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::LlcAligned { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, AffinityIntent::LlcAligned));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::LlcAligned")
                    && n.message().contains("without resolved CPUs")),
            "unresolved LlcAligned must surface a topology-aware-fallback note: {:?}",
            spec.notes,
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityIntent::exact([<llc_cpu_0>, <llc_cpu_1>, ...])")),
            "unresolved LlcAligned note must include paste-ready Rust hand-edit target: {:?}",
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
                .any(|n| n.message().contains("AffinityHint::LlcAligned")
                    && n.message().contains("with resolved CPUs")),
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
                .any(|n| n.message().contains("AffinityHint::SingleCpu") && n.message().contains("with resolved CPUs")),
            "resolved SingleCpu must surface a resolved-collapse note: {:?}",
            spec.notes,
        );
    }

    /// Unresolved `SingleCpu` hint falls back to
    /// [`AffinityIntent::SingleCpu`] with a hand-edit note. Pins the
    /// fallback path so a regression that drops the unresolved
    /// branch surfaces here. The hand-edit target must be paste-ready
    /// Rust (`AffinityIntent::exact(...)`).
    #[test]
    fn generate_spec_single_cpu_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::SingleCpu { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, AffinityIntent::SingleCpu));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::SingleCpu")
                    && n.message().contains("without resolved CPUs")),
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityIntent::exact([<cpu>])")),
            "unresolved SingleCpu note must include paste-ready Rust hand-edit target: {:?}",
            spec.notes,
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
            spec.notes.iter().any(
                |n| n.message().contains("AffinityHint::CrossCgroup") && n.message().contains("with resolved CPUs")
            ),
        );
    }

    /// Unresolved `CrossCgroup` hint (empty `cpus`) emits
    /// [`AffinityIntent::CrossCgroup`] and surfaces a note reminding
    /// the consumer that direct [`crate::workload::WorkloadHandle::spawn`]
    /// rejects this variant (the scenario engine expands it to the
    /// full topology). Pins the unresolved-fallback path of the
    /// CrossCgroup topology-aware projection. The hand-edit target
    /// must be paste-ready Rust (`AffinityIntent::exact(...)`).
    #[test]
    fn generate_spec_cross_cgroup_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::CrossCgroup { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(spec.config.affinity, AffinityIntent::CrossCgroup));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::CrossCgroup")
                    && n.message().contains("without resolved CPUs")),
            "unresolved CrossCgroup must surface a topology-aware-fallback note: {:?}",
            spec.notes,
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityIntent::exact([<cpu_0>, <cpu_1>, ...])")),
            "unresolved CrossCgroup note must include paste-ready Rust hand-edit target: {:?}",
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
                .any(|n| n.message().contains("AffinityHint::RandomSubset")
                    && n.message().contains("with resolved pool")),
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
                assert!(
                    from.is_empty(),
                    "unresolved RandomSubset must emit empty pool"
                );
                assert_eq!(*count, 0);
            }
            other => panic!("expected placeholder RandomSubset, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::RandomSubset")
                    && n.message().contains("without a resolved pool")),
        );
    }

    /// Mixed-input `RandomSubset`: non-empty `from` with `count == 0`
    /// projects to the unresolved placeholder. The mapper rejects on
    /// EITHER `from.is_empty()` OR `*count == 0`, so a producer that
    /// records the pool but loses the popcount value (or vice versa)
    /// must surface as the hand-edit-required placeholder rather
    /// than a half-populated `AffinityIntent::RandomSubset` that the
    /// spawn-time gate rejects with a less actionable error.
    #[test]
    fn generate_spec_random_subset_pool_without_count_is_placeholder() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: vec![0, 1, 2],
            count: 0,
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::RandomSubset { from, count } => {
                assert!(
                    from.is_empty(),
                    "(non_empty, 0) must drop pool to placeholder: got {from:?}",
                );
                assert_eq!(*count, 0);
            }
            other => panic!("expected placeholder RandomSubset, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::RandomSubset")
                    && n.message().contains("without a resolved pool")),
            "(non_empty, 0) must surface unresolved-pool note: {:?}",
            spec.notes,
        );
    }

    /// Mixed-input `RandomSubset`: empty `from` with non-zero `count`
    /// projects to the unresolved placeholder for the same reason
    /// (either side missing → placeholder). The popcount alone is
    /// insufficient to spawn — the spawn-time gate needs a real CPU
    /// pool, so we surface the hand-edit prompt up-front.
    #[test]
    fn generate_spec_random_subset_count_without_pool_is_placeholder() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: Vec::new(),
            count: 3,
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::RandomSubset { from, count } => {
                assert!(from.is_empty());
                assert_eq!(
                    *count, 0,
                    "([], non_zero) must drop count to placeholder: got {count}",
                );
            }
            other => panic!("expected placeholder RandomSubset, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::RandomSubset")
                    && n.message().contains("without a resolved pool")),
            "([], non_zero) must surface unresolved-pool note: {:?}",
            spec.notes,
        );
    }

    /// `count` > `from.len()` is accepted as resolved — the mapper
    /// gates only on emptiness, not on whether `count` exceeds the
    /// pool size. The spawn-time affinity resolver enforces the
    /// `count <= from.len()` invariant; the projection layer trusts
    /// the producer-observed values verbatim and lets the downstream
    /// gate surface the constraint violation.
    #[test]
    fn generate_spec_random_subset_count_exceeds_pool_is_populated() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: vec![0, 1],
            count: 10,
        }];
        let spec = generate_spec(&cap);
        match &spec.config.affinity {
            AffinityIntent::RandomSubset { from, count } => {
                let v: Vec<usize> = from.iter().copied().collect();
                assert_eq!(v, vec![0, 1]);
                assert_eq!(
                    *count, 10,
                    "count > pool.len() must passthrough verbatim: got {count}",
                );
            }
            other => panic!("expected populated RandomSubset, got {other:?}"),
        }
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::RandomSubset")
                    && n.message().contains("with resolved pool")),
            "count > pool.len() must take the resolved path: {:?}",
            spec.notes,
        );
    }

    /// Resolved `SmtSiblingPair` hint (non-empty `cpus`) collapses to
    /// [`AffinityIntent::Exact`]. The producer recorded the observed
    /// SMT sibling pair and the generator emits a runnable spec.
    #[test]
    fn generate_spec_smt_sibling_pair_resolved_emits_exact() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::SmtSiblingPair { cpus: vec![2, 3] }];
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
                .any(|n| n.message().contains("AffinityHint::SmtSiblingPair")
                    && n.message().contains("with resolved CPUs")),
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
    /// SmtSiblingPair topology-aware projection. The hand-edit
    /// target string is paste-ready Rust
    /// (`AffinityIntent::exact([<sibling_a>, <sibling_b>])`), so the
    /// note must contain that exact substring.
    #[test]
    fn generate_spec_smt_sibling_pair_unresolved_emits_topology_aware() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::SmtSiblingPair { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(matches!(
            spec.config.affinity,
            AffinityIntent::SmtSiblingPair
        ));
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityHint::SmtSiblingPair")
                    && n.message().contains("without resolved CPUs")),
            "unresolved SmtSiblingPair must surface a topology-aware-fallback note: {:?}",
            spec.notes,
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| n.message().contains("AffinityIntent::exact([<sibling_a>, <sibling_b>])")),
            "unresolved SmtSiblingPair note must include paste-ready Rust hand-edit target: {:?}",
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
    /// Pins the conditional-rendering branch in
    /// [`render_run_file_source`] so a regression that drops the
    /// comment block surfaces here.
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

    /// `render_run_file_source` always emits `.nice(N)` even when
    /// `N == 0`. Pins the explicit-render guarantee from #521: the
    /// rendered spec must NOT silently inherit a future-changed
    /// upstream `WorkloadConfig::nice` default. A regression that
    /// suppresses the call when nice is 0 surfaces here.
    #[test]
    fn render_run_file_source_emits_nice_zero_explicitly() {
        let cap = DebugCapture::default();
        let spec = generate_spec(&cap);
        // Default fingerprint → no sched-policy hints → nice stays 0.
        assert_eq!(spec.config.nice, 0);

        let src = render_run_file_source(&spec, "explicit_nice");
        assert!(
            src.contains(".nice(0)"),
            "rendered source must always emit `.nice(0)` (no silent \
             inheritance of upstream defaults): {src}",
        );
    }

    /// `render_ktstr_test_source` rewrites the `pub fn` line by
    /// substring-replacing `format!("pub fn {template_name}")`. If a
    /// caller passes a template_name that also appears as a substring
    /// elsewhere in the rendered body (e.g. as part of a comment
    /// path or an `AffinityIntent::Exact` field name), the
    /// `String::replace` could in theory rewrite an unintended
    /// occurrence. Verify the actual behaviour: only the `pub fn ...`
    /// site is rewritten because the search pattern includes the
    /// `pub fn ` prefix, so unrelated substrings are not matched.
    #[test]
    fn render_ktstr_test_source_template_name_substring_in_body() {
        // Choose a template_name that also appears in the body — the
        // path "auto" appears nowhere else, but "default" does (in
        // `WorkloadConfig::default()`). Pick "default" as the test
        // name to verify that only the `pub fn default(` line gets
        // the attribute prefix.
        let cap = DebugCapture::default();
        let spec = generate_spec(&cap);
        let src = render_ktstr_test_source(&spec, "default");

        // The attribute must appear exactly once, attached to the
        // `pub fn default(` declaration.
        let attribute = "#[ktstr::ktstr_test]";
        let attribute_count = src.matches(attribute).count();
        assert_eq!(
            attribute_count, 1,
            "attribute must be inserted exactly once, got {attribute_count} \
             occurrences in: {src}",
        );

        // The `WorkloadConfig::default()` call in the body must NOT
        // have been mangled into `WorkloadConfig::#[...] default()`.
        assert!(
            src.contains("WorkloadConfig::default()"),
            "WorkloadConfig::default() must remain intact (substring \
             replace must not match the `default()` body call): {src}",
        );

        // The rewritten function declaration must be present.
        assert!(
            src.contains("#[ktstr::ktstr_test]\npub fn default"),
            "rewritten `pub fn default` must carry the attribute: {src}",
        );
    }

    /// [`ReproducerSpec::is_runnable`] returns `true` for a
    /// fully-resolved `RandomSubset` (non-empty `from`, non-zero
    /// `count`). The spawn-time affinity gate accepts that shape, so
    /// the resolved note ("...accepts it without hand-editing") must
    /// be classified [`ReproducerNote::Resolved`] — NOT
    /// [`ReproducerNote::UnresolvedAffinity`] — even though both
    /// notes share the "spawn-time affinity gate" prefix. This test
    /// pins the typed-classification contract: a regression that
    /// pushed the resolved-collapse note as
    /// `ReproducerNote::UnresolvedAffinity` (e.g. wrong variant at
    /// the call site) would surface here as
    /// `is_runnable() == false`.
    #[test]
    fn is_runnable_resolved_random_subset() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: vec![0, 1, 2],
            count: 2,
        }];
        let spec = generate_spec(&cap);
        assert!(
            spec.is_runnable(),
            "resolved RandomSubset must be runnable; notes: {:?}",
            spec.notes,
        );
        assert_eq!(spec.unresolved_count(), 0);
    }

    /// [`ReproducerSpec::is_runnable`] returns `false` for an
    /// unresolved `SingleCpu` (empty `cpus`). Pins that the
    /// topology-aware unresolved-fallback note is classified
    /// [`ReproducerNote::UnresolvedAffinity`] so callers can detect
    /// the runnability gap without re-parsing the rendered source.
    #[test]
    fn is_runnable_unresolved_single_cpu() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::SingleCpu { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(
            !spec.is_runnable(),
            "unresolved SingleCpu must NOT be runnable; notes: {:?}",
            spec.notes,
        );
        assert_eq!(spec.unresolved_count(), 1);
    }

    /// [`ReproducerSpec::is_runnable`] returns `false` for an
    /// empty-`Exact` projection. Pins that the empty-`Exact` note
    /// is classified [`ReproducerNote::UnresolvedAffinity`].
    #[test]
    fn is_runnable_empty_exact() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::Exact { cpus: Vec::new() }];
        let spec = generate_spec(&cap);
        assert!(
            !spec.is_runnable(),
            "empty Exact must NOT be runnable; notes: {:?}",
            spec.notes,
        );
        assert_eq!(spec.unresolved_count(), 1);
    }

    /// [`ReproducerSpec::is_runnable`] returns `false` for an
    /// unresolved `RandomSubset` (empty pool, zero count). Pins that
    /// the placeholder note is classified
    /// [`ReproducerNote::UnresolvedAffinity`].
    #[test]
    fn is_runnable_unresolved_random_subset() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::RandomSubset {
            from: Vec::new(),
            count: 0,
        }];
        let spec = generate_spec(&cap);
        assert!(
            !spec.is_runnable(),
            "unresolved RandomSubset must NOT be runnable; notes: {:?}",
            spec.notes,
        );
        assert_eq!(spec.unresolved_count(), 1);
    }

    /// [`ReproducerSpec::is_runnable`] returns `true` for a default
    /// (empty-fingerprint) capture. Default `WorkloadConfig` has
    /// `AffinityIntent::Inherit` which the spawn-time gate accepts;
    /// no hand-edit-required notes are generated, so `is_runnable`
    /// must return true even though the spec carries informational
    /// notes (e.g. "no workload groups in fingerprint").
    #[test]
    fn is_runnable_empty_fingerprint() {
        let cap = DebugCapture::default();
        let spec = generate_spec(&cap);
        assert!(
            spec.is_runnable(),
            "empty fingerprint must be runnable (default Inherit); notes: {:?}",
            spec.notes,
        );
        assert_eq!(spec.unresolved_count(), 0);
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

    /// End-to-end smoke test: build a fingerprint with one hint of
    /// every kind (workload group, a resolved topology-aware affinity
    /// variant, a parameterized work-type, a cgroup hint, a
    /// sched-policy hint, and a fingerprint gap), run
    /// [`generate_spec`] → [`render_run_file_source`], and assert the
    /// output is deterministic (same input → byte-identical output)
    /// and carries every expected fragment. The test pins the full
    /// pipeline so a regression in any single stage (projection,
    /// mapping, render) surfaces here even when the per-variant unit
    /// tests pass.
    ///
    /// Picks `SmtSiblingPair { cpus: vec![4, 5] }` as the primary
    /// affinity hint to exercise the resolved-collapse path; a
    /// secondary `Exact` hint demonstrates the
    /// "additional affinity hints not modeled" fallback.
    /// Unresolved-payload branches for the topology-aware variants
    /// have dedicated tests above.
    #[test]
    fn render_run_file_source_e2e_smoke() {
        let mut cap = DebugCapture::default();
        cap.fingerprint.workload_groups = vec![WorkloadGroupHint {
            cgroup_path: "/system.slice/foo.service".into(),
            thread_count: 16,
            cpu_time_fraction: 0.65,
            wakeups_per_sec: 850.0,
        }];
        // Pick one resolved affinity hint — the first wins; rest fold
        // into notes via the existing "additional affinity hints"
        // formatter.
        cap.fingerprint.affinity_hints = vec![
            AffinityHint::SmtSiblingPair { cpus: vec![4, 5] },
            AffinityHint::Exact {
                cpus: vec![0, 1, 2, 3],
            },
        ];
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::Bursty {
            burst_duration: Duration::from_millis(7),
            sleep_duration: Duration::from_millis(43),
        }];
        cap.fingerprint.cgroup_hints = vec![CgroupHint {
            path: "/system.slice/foo.service".into(),
            cpu_weight: Some(150),
            memory_max_bytes: Some(4 * 1024 * 1024 * 1024),
            cpuset_cpus: vec![4, 5],
            cpu_max_quota_us: Some(50_000),
        }];
        cap.fingerprint.sched_policy_hints = vec![SchedPolicyHint::Fifo { priority: 60 }];
        cap.fingerprint.gaps = vec!["sample window had 2 dropouts".into()];

        let spec1 = generate_spec(&cap);
        let src1 = render_run_file_source(&spec1, "e2e_repro");

        // Determinism: same capture → byte-identical render.
        let spec2 = generate_spec(&cap);
        let src2 = render_run_file_source(&spec2, "e2e_repro");
        assert_eq!(
            src1, src2,
            "render_run_file_source must be deterministic for the same capture",
        );

        // Skeleton + import lines always present.
        assert!(src1.contains("use ktstr::workload::*;"));
        assert!(src1.contains("use std::collections::BTreeSet;"));
        assert!(src1.contains("use std::time::Duration;"));
        assert!(src1.contains("pub fn e2e_repro"));

        // Workload-group projection wired into builder.
        assert!(
            src1.contains(".workers(16)"),
            "thread_count=16 must surface as .workers(16): {src1}",
        );

        // The resolved SmtSiblingPair hint collapses to Exact —
        // the captured siblings land in the rendered builder call.
        assert!(
            src1.contains(".affinity(AffinityIntent::Exact"),
            "first affinity hint (SmtSiblingPair) must collapse to Exact: {src1}",
        );
        assert!(
            src1.contains("BTreeSet::from([4, 5])"),
            "Exact pool must contain the SmtSiblingPair CPUs: {src1}",
        );

        // Bursty parameters come through.
        assert!(
            src1.contains(".work_type(WorkType::Bursty"),
            "Bursty work-type hint must reach the builder: {src1}",
        );
        assert!(
            src1.contains("Duration::from_millis(7)") && src1.contains("Duration::from_millis(43)"),
            "Bursty durations must surface in the rendered call: {src1}",
        );

        // Sched-policy + cgroup hints in the rendered surface.
        assert!(
            src1.contains(".sched_policy(SchedPolicy::Fifo(60))"),
            "Fifo priority must surface: {src1}",
        );
        assert!(
            src1.contains("Cgroup hints"),
            "cgroup hints must render as comments: {src1}",
        );
        assert!(
            src1.contains("/system.slice/foo.service"),
            "cgroup path must appear in the rendered comments: {src1}",
        );

        // Notes block contains the propagated fingerprint gap +
        // resolved-collapse note for SmtSiblingPair + the
        // additional-hints fallback for the second affinity entry.
        assert!(
            src1.contains("Generator notes:"),
            "non-empty notes must trigger the comment block: {src1}",
        );
        assert!(
            src1.contains("fingerprint gap: sample window had 2 dropouts"),
            "fingerprint gap must propagate verbatim: {src1}",
        );
        assert!(
            src1.contains("AffinityHint::SmtSiblingPair"),
            "resolved-collapse note must cite the original variant: {src1}",
        );
        assert!(
            src1.contains("additional affinity hints not modeled"),
            "second affinity hint must surface as an additional-hints note: {src1}",
        );

        // No `/* TODO:` placeholder for any implemented work-type
        // — Bursty is fully mapped, so `render_work_type_todo` must
        // not fire for it. The `/* TODO:` substring is the lead-in
        // every `render_work_type_todo` output emits regardless of
        // the variant name; matching it catches any TODO placeholder
        // even after the variant-specific wording in the body
        // changes.
        assert!(
            !src1.contains("/* TODO:"),
            "implemented work-type variants must not render any TODO placeholder: {src1}",
        );
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

    /// [`is_unmapped_work_type`] returns `false` for every variant
    /// [`render_work_type`] dispatches to a runnable builder-call arm,
    /// and `true` for every variant it dispatches to
    /// [`render_work_type_todo`]. Pins the runnable / TODO split
    /// in lock-step with [`render_work_type`] — adding a new variant
    /// to one match arm without updating the other shows up here as
    /// a failed expectation.
    #[test]
    fn is_unmapped_work_type_split_matches_render() {
        // Sample one of each runnable variant — both nullary
        // (`SpinWait`) and parameterized (`Bursty` / `CachePressure`)
        // forms must classify as runnable.
        let runnable_samples: Vec<WorkType> = vec![
            WorkType::SpinWait,
            WorkType::YieldHeavy,
            WorkType::Mixed,
            WorkType::IoSyncWrite,
            WorkType::IoRandRead,
            WorkType::IoConvoy,
            WorkType::Bursty {
                burst_duration: Duration::from_millis(5),
                sleep_duration: Duration::from_millis(95),
            },
            WorkType::PipeIo { burst_iters: 1024 },
            WorkType::FutexPingPong { spin_iters: 1024 },
            WorkType::CachePressure {
                size_kb: 256,
                stride: 64,
            },
        ];
        for w in &runnable_samples {
            assert!(
                !is_unmapped_work_type(w),
                "{w:?} renders as a runnable builder call — \
                 is_unmapped_work_type must return false",
            );
            // Each runnable variant must NOT pass through
            // render_work_type_todo, so the rendered output must NOT
            // carry the TODO marker substring.
            let rendered = render_work_type(w);
            assert!(
                !rendered.contains("/* TODO:"),
                "{w:?} must render without a TODO placeholder: {rendered}",
            );
        }

        // Sample several TODO variants — the TODO arms must classify
        // as unmapped and render through render_work_type_todo
        // (visible via the `/* TODO:` substring).
        let unmapped_samples: Vec<WorkType> = vec![
            WorkType::CacheYield {
                size_kb: 256,
                stride: 64,
            },
            WorkType::ForkExit,
            WorkType::SmtSiblingSpin,
            WorkType::NiceSweep,
        ];
        for w in &unmapped_samples {
            assert!(
                is_unmapped_work_type(w),
                "{w:?} renders as a TODO placeholder — \
                 is_unmapped_work_type must return true",
            );
            let rendered = render_work_type(w);
            assert!(
                rendered.contains("/* TODO:"),
                "{w:?} must render through render_work_type_todo: {rendered}",
            );
        }
    }

    /// Manually-constructed [`ReproducerSpec`] with an unmapped
    /// [`WorkType`] variant in `config.work_type` is correctly
    /// classified as not runnable by [`ReproducerSpec::is_runnable`],
    /// even though no projection-time note was pushed (the spec was
    /// not built via [`generate_spec`] / [`map_work_type`]). Pins the
    /// direct-config check in `is_runnable` that catches the
    /// "renders as TODO but no note in spec.notes" failure mode the
    /// projection-only signal would miss.
    #[test]
    fn is_runnable_unmapped_work_type_via_direct_config() {
        let mut spec = ReproducerSpec::default();
        spec.config.work_type = WorkType::CacheYield {
            size_kb: 256,
            stride: 64,
        };
        // No notes pushed — direct construction bypasses the
        // projection layer.
        assert!(
            spec.notes.is_empty(),
            "fixture must not add notes; got {:?}",
            spec.notes,
        );
        assert!(
            !spec.is_runnable(),
            "spec with unmapped work_type must NOT be runnable, even \
             without projection notes; config.work_type: {:?}",
            spec.config.work_type,
        );
        // unresolved_count counts notes only — direct-config path
        // doesn't push notes, so the count stays at zero.
        assert_eq!(
            spec.unresolved_count(),
            0,
            "unresolved_count covers typed unresolved notes only; got {}",
            spec.unresolved_count(),
        );
    }

    /// [`record_work_type`] is the production path that
    /// [`map_work_type`] funnels every projected [`WorkType`] through.
    /// When the assigned variant is one [`render_work_type`]
    /// dispatches to [`render_work_type_todo`], the helper must push a
    /// [`ReproducerNote::UnmappedWorkType`] entry that
    /// [`ReproducerSpec::unresolved_count`] picks up. Today no
    /// [`WorkTypeHint`] variant projects to a TODO variant, so the
    /// branch is reachable from production code only via this helper —
    /// the test drives [`record_work_type`] directly with
    /// [`WorkType::ForkExit`] to exercise the live branch (rather than
    /// a hand-rolled re-implementation of the same logic).
    #[test]
    fn record_work_type_emits_note_for_unmapped_projection() {
        let mut spec = ReproducerSpec::default();
        record_work_type(WorkType::ForkExit, &mut spec);

        assert!(
            matches!(spec.config.work_type, WorkType::ForkExit),
            "record_work_type must assign the variant to spec.config.work_type",
        );
        assert!(
            spec.notes
                .iter()
                .any(|n| matches!(n, ReproducerNote::UnmappedWorkType(_))),
            "unmapped projection must push a ReproducerNote::UnmappedWorkType: {:?}",
            spec.notes,
        );
        assert_eq!(
            spec.unresolved_count(),
            1,
            "unresolved_count must include the UnmappedWorkType note",
        );
        assert!(
            !spec.is_runnable(),
            "spec with TODO note + unmapped work_type must be NOT runnable",
        );
    }

    /// [`record_work_type`] does NOT push an
    /// [`ReproducerNote::UnmappedWorkType`] note when the assigned
    /// variant is one [`render_work_type`] renders as a runnable
    /// builder call. Pins the runnable / TODO split through the
    /// production code path — a regression that flips a runnable
    /// variant onto the TODO arm of [`is_unmapped_work_type`] would
    /// surface here as a spurious unmapped note.
    #[test]
    fn record_work_type_does_not_emit_note_for_runnable_projection() {
        let mut spec = ReproducerSpec::default();
        record_work_type(WorkType::SpinWait, &mut spec);

        assert!(
            spec.notes.is_empty(),
            "runnable projection must NOT push an unmapped note: {:?}",
            spec.notes,
        );
        assert_eq!(spec.unresolved_count(), 0);
    }

    /// `is_runnable()` for a spec with an unmapped work_type AND a
    /// pre-existing affinity hand-edit note returns `false` and the
    /// counts compose correctly: each note contributes one unit to
    /// `unresolved_count`. Drives the unmapped-projection branch via
    /// [`record_work_type`] (the production code path) rather than a
    /// hand-rolled note insertion — composes the two unresolved-note
    /// kinds end-to-end.
    #[test]
    fn is_runnable_combines_work_type_and_affinity_signals() {
        // Empty Exact — produces an affinity hand-edit note typed
        // ReproducerNote::UnresolvedAffinity.
        let mut cap = DebugCapture::default();
        cap.fingerprint.affinity_hints = vec![AffinityHint::Exact { cpus: Vec::new() }];
        let mut spec = generate_spec(&cap);
        // Drive the unmapped-work-type branch via the production
        // helper so the composed test exercises both production paths.
        record_work_type(WorkType::ForkExit, &mut spec);

        assert!(!spec.is_runnable());
        assert_eq!(
            spec.unresolved_count(),
            2,
            "expected 2 unresolved notes (1 affinity + 1 work-type), got {}: {:?}",
            spec.unresolved_count(),
            spec.notes,
        );
    }

    /// [`ReproducerNote`] serializes to a snake_case wire format —
    /// `kind` field values are `informational` / `resolved` /
    /// `unresolved_affinity` / `unmapped_work_type`. Pins the
    /// `#[serde(rename_all = "snake_case")]` attribute on the enum
    /// so a regression that drops the attribute (which would silently
    /// revert to PascalCase variant names like `UnresolvedAffinity`)
    /// surfaces here as a wire-format mismatch.
    ///
    /// Asserts BOTH directions: the serialized text contains the
    /// snake_case `kind` literal, AND deserialization round-trips
    /// back to the same variant. The roundtrip path catches the
    /// asymmetric-rename failure mode (where serialization and
    /// deserialization disagree on the wire format), which a
    /// one-direction assertion would miss.
    #[test]
    fn reproducer_note_wire_format_is_snake_case() {
        let cases: &[(ReproducerNote, &str)] = &[
            (
                ReproducerNote::Informational("info".into()),
                "informational",
            ),
            (ReproducerNote::Resolved("res".into()), "resolved"),
            (
                ReproducerNote::UnresolvedAffinity("ua".into()),
                "unresolved_affinity",
            ),
            (
                ReproducerNote::UnmappedWorkType("uwt".into()),
                "unmapped_work_type",
            ),
        ];
        for (note, expected_kind) in cases {
            let json = serde_json::to_string(note)
                .expect("ReproducerNote must serialize via the derive impl");
            let kind_pin = format!(r#""kind":"{expected_kind}""#);
            assert!(
                json.contains(&kind_pin),
                "wire format must encode kind={expected_kind:?} (snake_case) — \
                 a regression that drops `#[serde(rename_all = \"snake_case\")]` \
                 from the enum would revert to PascalCase. note={note:?}, json={json}",
            );
            let round_tripped: ReproducerNote = serde_json::from_str(&json)
                .expect("ReproducerNote must deserialize from its own serialized form");
            assert_eq!(
                std::mem::discriminant(note),
                std::mem::discriminant(&round_tripped),
                "roundtrip must preserve the variant — asymmetric rename \
                 between Serialize and Deserialize would surface here. \
                 sent={note:?}, got={round_tripped:?}",
            );
            assert_eq!(
                note.message(),
                round_tripped.message(),
                "roundtrip must preserve the message payload",
            );
        }
    }

    /// `render_run_file_source` output compiles as valid Rust.
    ///
    /// Builds a representative [`ReproducerSpec`] that exercises every
    /// rendered surface (parameterized [`WorkType::Bursty`], populated
    /// [`AffinityIntent::Exact`], [`SchedPolicy::Deadline`] with
    /// `Duration` fields, cgroup hint comments, generator notes), then
    /// invokes `rustc --edition 2021 --crate-type lib` on the rendered
    /// output to confirm:
    ///
    /// - format strings produce parseable Rust (no stray commas /
    ///   missing braces)
    /// - field names match the API (a typo like
    ///   `WorkType::Bursty { burst_dur: ... }` would surface here as
    ///   a compile error against the stubbed `WorkType::Bursty`)
    /// - builder method names match
    ///   ([`render_run_file_source`] emits `.workers/.affinity/
    ///   .work_type/.sched_policy/.nice` — a typo would not resolve
    ///   against the stub)
    ///
    /// The test prepends a `mod ktstr { pub mod workload { ... } }`
    /// stub before the rendered source so the rendered
    /// `use ktstr::workload::*;` resolves to the stub. This isolates
    /// the test from the surrounding crate build (no `--extern`
    /// gymnastics) while still exercising every type and variant the
    /// rendered source mentions. A regression in the renderer (typo,
    /// extra brace, drifted field name) surfaces as a `rustc` failure
    /// with the rendered source attached for diagnostic.
    ///
    /// # Precondition
    ///
    /// Requires `rustc` to be invocable. The test resolves the
    /// compiler via `$RUSTC` first, falling back to `rustc` on
    /// `$PATH`. Cargo always exports `$RUSTC` when running tests, so
    /// the standard `cargo nextest run` / `cargo ktstr test`
    /// invocation always satisfies the precondition. A missing
    /// `rustc` produces a panic with a precondition-explicit message
    /// pointing the operator at the likely cause (running tests
    /// outside of cargo without `$PATH` covering `rustc`); the test
    /// does NOT silently no-op when the compiler is absent —
    /// silent-skip would give the false signal of green when the
    /// rendered-source compile-check was never exercised.
    #[test]
    fn render_run_file_source_compiles_via_rustc() {
        // Spec covering: parameterized WorkType, populated Exact
        // affinity, Deadline policy with Duration fields, cgroup
        // hints, generator notes (fingerprint gap), and a non-zero
        // nice value.
        let mut cap = DebugCapture::default();
        cap.fingerprint.workload_groups = vec![WorkloadGroupHint {
            cgroup_path: "/system.slice/foo.service".into(),
            thread_count: 16,
            cpu_time_fraction: 0.65,
            wakeups_per_sec: 850.0,
        }];
        cap.fingerprint.affinity_hints = vec![AffinityHint::Exact {
            cpus: vec![0, 1, 4, 5],
        }];
        cap.fingerprint.work_type_hints = vec![WorkTypeHint::Bursty {
            burst_duration: Duration::from_millis(7),
            sleep_duration: Duration::from_millis(43),
        }];
        cap.fingerprint.cgroup_hints = vec![CgroupHint {
            path: "/system.slice/foo.service".into(),
            cpu_weight: Some(150),
            memory_max_bytes: Some(4 * 1024 * 1024 * 1024),
            cpuset_cpus: vec![0, 1, 4, 5],
            cpu_max_quota_us: Some(50_000),
        }];
        cap.fingerprint.sched_policy_hints = vec![SchedPolicyHint::Deadline {
            runtime_ns: 1_000_000,
            deadline_ns: 5_000_000,
            period_ns: 10_000_000,
        }];
        cap.fingerprint.gaps = vec!["sample window had 2 dropouts".into()];
        let spec = generate_spec(&cap);
        let rendered = render_run_file_source(&spec, "compile_check_repro");

        // Stub module mirroring the API surface the rendered source
        // uses. `pub use std::collections::BTreeSet` and
        // `std::time::Duration` re-exports are NOT needed — the
        // rendered source imports them directly from `std`. Stub
        // types are unit / payload-bearing variants that match the
        // names the renderer emits one-for-one, so the rendered
        // builder calls type-check against this surface.
        let stub = r#"
#[allow(dead_code, unused_variables, unused_imports)]
mod ktstr { pub mod workload {
    use std::collections::BTreeSet;
    use std::time::Duration;
    pub struct WorkloadConfig;
    impl WorkloadConfig {
        pub fn default() -> Self { Self }
        pub fn workers(self, _: usize) -> Self { self }
        pub fn affinity(self, _: AffinityIntent) -> Self { self }
        pub fn work_type(self, _: WorkType) -> Self { self }
        pub fn sched_policy(self, _: SchedPolicy) -> Self { self }
        pub fn nice(self, _: i32) -> Self { self }
    }
    pub enum AffinityIntent {
        Inherit,
        SingleCpu,
        LlcAligned,
        CrossCgroup,
        SmtSiblingPair,
        RandomSubset { from: BTreeSet<usize>, count: usize },
        Exact(BTreeSet<usize>),
    }
    pub enum WorkType {
        SpinWait,
        YieldHeavy,
        Mixed,
        IoSyncWrite,
        IoRandRead,
        IoConvoy,
        Bursty { burst_duration: Duration, sleep_duration: Duration },
        PipeIo { burst_iters: u64 },
        FutexPingPong { spin_iters: u64 },
        CachePressure { size_kb: usize, stride: usize },
    }
    pub enum SchedPolicy {
        Normal,
        Batch,
        Idle,
        Fifo(u32),
        RoundRobin(u32),
        Deadline { runtime: Duration, deadline: Duration, period: Duration },
    }
}}
"#;

        // The rendered source begins with header comments, then
        // `use ktstr::workload::*;`. Splice the stub in BEFORE the
        // rendered output so the `use` resolves to the stub module.
        // Splicing before the rendered output (rather than replacing
        // the use line) keeps the rendered source byte-identical to
        // what callers actually emit, so the test exercises the
        // production surface verbatim.
        let combined = format!("{stub}\n{rendered}");

        // Write to a tempfile — prefer NamedTempFile so the file is
        // cleaned up automatically even if rustc panics. The `.rs`
        // suffix is required for rustc to accept the input path
        // without a `--crate-name` override.
        use std::io::Write as _;
        let mut tmp = tempfile::Builder::new()
            .prefix("ktstr_reproducer_compile_check_")
            .suffix(".rs")
            .tempfile()
            .expect("create tempfile for rendered source");
        tmp.write_all(combined.as_bytes())
            .expect("write rendered source");
        tmp.flush().expect("flush rendered source");

        // rustc invocation: `--edition 2021 --crate-type lib`
        // matches the task spec and the rendered source's idiom (no
        // `fn main`, library shape). `--out-dir <tempdir>` keeps
        // build artifacts out of the workspace; the tempdir drops at
        // end of scope so artifacts don't leak.
        let out_dir = tempfile::TempDir::new().expect("rustc out tempdir");
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let output = std::process::Command::new(&rustc)
            .arg("--edition")
            .arg("2021")
            .arg("--crate-type")
            .arg("lib")
            .arg("--out-dir")
            .arg(out_dir.path())
            .arg(tmp.path())
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "render_run_file_source_compiles_via_rustc requires rustc \
                     (resolved via $RUSTC, then $PATH) — failed to spawn {rustc:?}: {e}. \
                     Cargo sets $RUSTC for cargo-test / cargo-nextest invocations; if \
                     you are running this test outside of cargo, ensure rustc is on \
                     $PATH or set $RUSTC explicitly. The test does NOT silently skip \
                     when rustc is missing — silent-skip would falsely report green \
                     when the rendered-source compile-check never ran.",
                )
            });

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "rustc rejected rendered source\n\
                 ---- rustc stderr ----\n\
                 {stderr}\n\
                 ---- combined source ----\n\
                 {combined}",
            );
        }
    }
}
