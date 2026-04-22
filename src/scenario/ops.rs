//! Composable ops/steps system for dynamic cgroup topology changes.
//!
//! [`Op`] is an atomic cgroup operation. [`Step`] sequences ops with a
//! hold period. [`CgroupDef`] bundles create + cpuset + spawn into a
//! single declaration. [`execute_steps()`] runs a step sequence with
//! scheduler liveness checks and stimulus event recording.
//!
//! See the [Ops and Steps](https://likewhatevs.github.io/ktstr/guide/concepts/ops.html)
//! chapter for a guide.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::assert::AssertResult;
use crate::vmm::shm_ring::{self, StimulusPayload};
use crate::workload::{AffinityKind, MemPolicy, Work, WorkType, WorkloadConfig, WorkloadHandle};

use super::{CgroupGroup, Ctx, process_alive};

// ---------------------------------------------------------------------------
// Op / CpusetSpec
// ---------------------------------------------------------------------------

/// Atomic operation on the cgroup topology.
///
/// Names use `Cow<'static, str>` so ops can reference compile-time
/// literals (zero-cost) or runtime-generated strings (owned).
#[derive(Clone, Debug)]
pub enum Op {
    /// Create a new cgroup under the managed cgroup parent.
    AddCgroup { name: Cow<'static, str> },
    /// Remove a cgroup (stops its workers first).
    RemoveCgroup { cgroup: Cow<'static, str> },
    /// Set a cgroup's cpuset to the resolved CPU set.
    SetCpuset {
        cgroup: Cow<'static, str>,
        cpus: CpusetSpec,
    },
    /// Clear a cgroup's cpuset (allow all CPUs).
    ClearCpuset { cgroup: Cow<'static, str> },
    /// Read both cgroups' cpusets and swap them.
    SwapCpusets {
        a: Cow<'static, str>,
        b: Cow<'static, str>,
    },
    /// Spawn workers and move them into the target cgroup.
    ///
    /// The work type is used as-is; gauntlet `work_type_override` does
    /// not apply. Use [`CgroupDef`] with `swappable(true)` when the
    /// work type should be overridable.
    Spawn {
        cgroup: Cow<'static, str>,
        work: Work,
    },
    /// Stop all workers in a cgroup (does not remove the cgroup).
    StopCgroup { cgroup: Cow<'static, str> },
    /// Set worker affinity in a cgroup. Resolved at apply time via
    /// [`resolve_affinity_for_cgroup()`](super::resolve_affinity_for_cgroup).
    SetAffinity {
        cgroup: Cow<'static, str>,
        affinity: AffinityKind,
    },
    /// Spawn workers in the parent cgroup (not in a managed cgroup).
    ///
    /// `Work` is resolved to a `WorkloadConfig` at apply time, matching
    /// the resolution pattern used by `Op::Spawn`.
    SpawnHost { work: Work },
    /// Move all tasks from one cgroup to another.
    ///
    /// Each task is moved via `cgroup.procs`. If any move fails, the
    /// error propagates and handle name keys are left unchanged (workers
    /// remain addressed under `from`). On success, handle name keys are
    /// updated to `to` so subsequent ops address the moved workers.
    MoveAllTasks {
        from: Cow<'static, str>,
        to: Cow<'static, str>,
    },
    /// Spawn a userspace [`Payload`](crate::test_support::Payload)
    /// binary in the background and track its
    /// [`PayloadHandle`](crate::scenario::payload_run::PayloadHandle)
    /// under the step's payload-handle set.
    ///
    /// Subsequent [`Op::WaitPayload`] / [`Op::KillPayload`] address
    /// the running child by the composite
    /// (`Payload::name`, `cgroup`) key — the same payload can run
    /// concurrently in two different cgroups without a dedup
    /// collision, but the lookup from the waiting op must match
    /// the pair the run op recorded. See [`Op::WaitPayload`] /
    /// [`Op::KillPayload`] for the ambiguity rules when the
    /// waiting op supplies only the name.
    ///
    /// Only [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are spawnable; scheduler-kind payloads are rejected at
    /// apply time with an actionable error.
    ///
    /// `args` is appended to `payload.default_args`. `cgroup`, when
    /// set, places the child in the named cgroup (resolved relative
    /// to the scenario's parent cgroup) via
    /// [`PayloadRun::in_cgroup`](crate::scenario::payload_run::PayloadRun::in_cgroup);
    /// unset inherits the spawning process's cgroup.
    ///
    /// Handles not explicitly consumed by `WaitPayload` / `KillPayload`
    /// are drained at step-teardown by `collect_step` (step-local) or
    /// at scenario end by `collect_backdrop` (when the handle lives on
    /// the Backdrop), matching the [`CgroupDef::workload`] semantics.
    ///
    /// # Scheduler-kind rejection across surfaces
    ///
    /// Three surfaces accept a `&Payload` and each rejects a
    /// scheduler-kind Payload differently — deliberately, to match
    /// the lifecycle of the caller:
    ///
    /// | Surface                                                                                   | Rejection             | When          |
    /// |-------------------------------------------------------------------------------------------|-----------------------|---------------|
    /// | [`PayloadRun::run`](crate::scenario::payload_run::PayloadRun::run) (`ctx.payload(&X)...`) | `Err(anyhow::Error)`  | scenario-time |
    /// | [`CgroupDef::workload`]                                                                   | `panic!`              | declaration-time |
    /// | `Op::RunPayload` (this variant)                                                           | `Err(anyhow::Error)`  | apply-ops-time |
    ///
    /// Rationale: `CgroupDef::workload` is a builder invoked during
    /// test construction (nextest `--list` phase) — a panic there
    /// surfaces the misuse before any VM boot, with a full
    /// backtrace pointing at the offending call. `ctx.payload()`
    /// and `Op::RunPayload` both run inside an executing scenario
    /// where one bad misuse should not crash the whole test run;
    /// they `bail!` with an actionable message and let the
    /// surrounding step-sequence skip to teardown. The three
    /// paths are symmetric in *what* they reject (scheduler-kind
    /// Payloads in non-scheduler slots); they differ only in
    /// *how* the misuse is surfaced, matched to caller context.
    RunPayload {
        payload: &'static crate::test_support::Payload,
        args: Vec<String>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// Block until the payload named `name` exits naturally, then
    /// evaluate its checks and record metrics to the per-test sidecar.
    ///
    /// The target is looked up by composite key (`name`, `cgroup`).
    /// `cgroup: None` matches the unique live copy (whatever its
    /// placement); if two or more copies of the same payload are
    /// live in different cgroups, the lookup bails with an
    /// "ambiguous — specify cgroup" error so the test doesn't
    /// silently wait on the wrong one. Use
    /// [`Op::wait_payload_in_cgroup`] to disambiguate.
    ///
    /// A consumed or unknown `(name, cgroup)` pair returns `Err`
    /// with an actionable message — test authors must not silently
    /// wait for payloads that were never started or have already
    /// been consumed by a prior `WaitPayload`/`KillPayload`.
    ///
    /// **No timeout.** `WaitPayload` waits indefinitely for the
    /// child to exit. A binary that never terminates (e.g. a
    /// benchmark configured without `--runtime=N`, or a stress-ng
    /// run without `--timeout`) will hang the step until the
    /// outer test watchdog fires. For time-boxed long-running
    /// payloads, prefer [`KillPayload`](Self::KillPayload) paired
    /// with a [`HoldSpec::Fixed`] / [`HoldSpec::Frac`] step
    /// boundary that guarantees forward progress; the payload's
    /// own CLI (`--runtime`, `--timeout`) is the reliable way to
    /// cap a single invocation's runtime.
    ///
    /// Check failures from the payload are recorded to the sidecar
    /// for regression analysis but do NOT fail the step or the test
    /// in-process. Use
    /// [`ctx.payload(&X).run()`](crate::scenario::payload_run::PayloadRun::run)
    /// directly if the test body needs to gate on check results.
    WaitPayload {
        name: Cow<'static, str>,
        cgroup: Option<Cow<'static, str>>,
    },
    /// SIGKILL the payload named `name`, reap the child, evaluate
    /// checks, and record metrics. Mirrors the behavior of
    /// step-teardown drain for an explicitly-targeted payload.
    ///
    /// The target is looked up by composite key (`name`, `cgroup`)
    /// — see [`Op::WaitPayload`] for the ambiguity rules.
    ///
    /// A consumed or unknown `(name, cgroup)` pair returns `Err`
    /// with an actionable message, identical to [`Op::WaitPayload`]'s
    /// lookup semantics.
    ///
    /// Check failures from the payload are recorded to the sidecar
    /// for regression analysis but do NOT fail the step or the test
    /// in-process. Use
    /// [`ctx.payload(&X).run()`](crate::scenario::payload_run::PayloadRun::run)
    /// directly if the test body needs to gate on check results.
    KillPayload {
        name: Cow<'static, str>,
        cgroup: Option<Cow<'static, str>>,
    },
}

/// How to compute a cpuset from topology.
#[derive(Clone, Debug)]
pub enum CpusetSpec {
    /// All CPUs in a given LLC index.
    Llc(usize),
    /// All CPUs in a given NUMA node index.
    Numa(usize),
    /// Fractional range of usable CPUs [start_frac..end_frac).
    Range { start_frac: f64, end_frac: f64 },
    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    Disjoint { index: usize, of: usize },
    /// Like Disjoint but each set overlaps neighbors by `frac` of its size.
    Overlap { index: usize, of: usize, frac: f64 },
    /// Exact CPU set (no topology resolution).
    Exact(BTreeSet<usize>),
}

impl CpusetSpec {
    /// Construct an `Exact` cpuset from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        CpusetSpec::Exact(cpus.into_iter().collect())
    }

    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    pub fn disjoint(index: usize, of: usize) -> Self {
        CpusetSpec::Disjoint { index, of }
    }

    /// Like [`disjoint`](Self::disjoint) but each set overlaps neighbors by `frac` of its size.
    pub fn overlap(index: usize, of: usize, frac: f64) -> Self {
        CpusetSpec::Overlap { index, of, frac }
    }

    /// Fractional range of usable CPUs `[start_frac..end_frac)`.
    pub fn range(start_frac: f64, end_frac: f64) -> Self {
        CpusetSpec::Range {
            start_frac,
            end_frac,
        }
    }

    /// All CPUs in a given LLC index.
    pub fn llc(index: usize) -> Self {
        CpusetSpec::Llc(index)
    }

    /// All CPUs in a given NUMA node index.
    pub fn numa(index: usize) -> Self {
        CpusetSpec::Numa(index)
    }
}

// ---------------------------------------------------------------------------
// CgroupDef
// ---------------------------------------------------------------------------

/// Declarative cgroup definition: name + cpuset + synthetic
/// [`Work`] groups + optional userspace [`Payload`](crate::test_support::Payload).
///
/// Bundles the ops that always go together (AddCgroup + SetCpuset +
/// Spawn) into a single value. The executor creates the cgroup, optionally
/// sets its cpuset, spawns workers for each [`Work`] entry, and moves
/// them into the cgroup.
///
/// Multiple [`Work`] entries run in parallel within the cgroup. Each
/// entry spawns its own set of worker processes. The optional
/// [`Self::payload`] slot is a *single* userspace binary that runs
/// alongside those synthetic [`Work`] groups (hence "plural works,
/// singular payload" — the pluralization in the legacy "workload(s)"
/// prose elided this distinction).
///
/// Use `CgroupDef` in `Step::with_defs` for scenarios where cgroups are
/// created once and run for the step duration. Use `Op::AddCgroup` +
/// `Op::Spawn` directly when you need mid-step cgroup creation, removal,
/// or other dynamic operations between spawn and collect.
///
/// ```
/// # use ktstr::scenario::ops::{CgroupDef, CpusetSpec};
/// # use ktstr::workload::{Work, WorkType};
/// // Single work group via convenience methods.
/// let def = CgroupDef::named("workers")
///     .with_cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(4)
///     .work_type(WorkType::CpuSpin);
///
/// assert_eq!(def.name, "workers");
/// assert_eq!(def.works[0].num_workers, Some(4));
///
/// // Multiple concurrent work groups via .work().
/// let def = CgroupDef::named("mixed")
///     .work(Work::default().workers(4).work_type(WorkType::CpuSpin))
///     .work(Work::default().workers(2).work_type(WorkType::YieldHeavy));
///
/// assert_eq!(def.works.len(), 2);
///
/// // Synthetic work + userspace binary side-by-side via .workload(&X).
/// // The binary runs inside the same cgroup as the Work handles;
/// // both spawn in apply_setup, the Work groups first, then the
/// // Payload after the cpuset settles.
/// # use ktstr::test_support::{OutputFormat, Payload, PayloadKind};
/// # const BENCH: Payload = Payload {
/// #     name: "bench",
/// #     kind: PayloadKind::Binary("bench"),
/// #     output: OutputFormat::ExitCode,
/// #     default_args: &[],
/// #     default_checks: &[],
/// #     metrics: &[],
/// # };
/// let def = CgroupDef::named("io_and_spin")
///     .with_cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(2)
///     .work_type(WorkType::CpuSpin)
///     .workload(&BENCH);
///
/// assert!(def.payload.is_some());
/// assert_eq!(def.works[0].num_workers, Some(2));
/// ```
#[derive(Clone, Debug)]
pub struct CgroupDef {
    /// Cgroup name relative to the scenario's parent cgroup. Must be a
    /// valid cgroupfs filename.
    pub name: Cow<'static, str>,
    /// Optional cpuset assignment. `None` inherits the parent cgroup's
    /// cpuset (typically the scenario's usable CPU set).
    pub cpuset: Option<CpusetSpec>,
    /// Work groups to spawn. Empty means use a single default Work
    /// (CpuSpin, Normal, ctx.workers_per_cgroup workers).
    pub works: Vec<Work>,
    /// When true, the gauntlet work_type override replaces each Work's
    /// work_type (applied per-Work via resolve_work_type).
    pub swappable: bool,
    /// Optional userspace [`Payload`](crate::test_support::Payload) to
    /// launch inside this cgroup.
    ///
    /// **Spawn order within `apply_setup`**: the cgroup is created
    /// (`add_cgroup_no_cpuset`), its cpuset is resolved + set, then
    /// each `Work` entry is spawned and moved into the cgroup in
    /// declaration order, and finally — after every synthetic
    /// `Work` handle has started — the `Payload` is spawned via
    /// `PayloadRun::new(ctx, p).in_cgroup(name).spawn()`. This
    /// fixed order lets the cgroup cpuset and mempolicy settle on
    /// the `Work` handles before the binary inherits placement, so
    /// the binary sees a stable topology. Once spawned, all three
    /// (cgroup, works, payload) run concurrently until teardown.
    ///
    /// Only
    /// [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are accepted — scheduler-kind payloads are rejected
    /// at construction time via [`Self::workload`]. The payload is
    /// killed at step-teardown (before cgroup removal) so the cgroup
    /// removal does not fail with EBUSY.
    pub payload: Option<&'static crate::test_support::Payload>,
}

impl CgroupDef {
    /// Create a CgroupDef with defaults (empty works, no cpuset).
    ///
    /// **Worker-spawning default:** `CgroupDef::named("cg_0")` alone
    /// still spawns workers at execution time — `apply_setup` fills
    /// an empty `works` slice with one default [`Work`] (CpuSpin,
    /// SCHED_NORMAL, `ctx.workers_per_cgroup` workers). To express
    /// an empty move-target cgroup with NO workers, declare it via
    /// [`Op::AddCgroup`] at step or Backdrop level instead of using
    /// a `CgroupDef`.
    pub fn named(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the cpuset for this cgroup. Use when defining cgroups in step
    /// setup (initial topology). For mid-run cpuset changes, use [`Op::SetCpuset`].
    pub fn with_cpuset(mut self, cpus: CpusetSpec) -> Self {
        self.cpuset = Some(cpus);
        self
    }

    /// Add a work group. Can be called multiple times for concurrent
    /// work groups within this cgroup.
    pub fn work(mut self, w: Work) -> Self {
        self.works.push(w);
        self
    }

    /// Ensure works[0] exists for single-Work builder methods.
    fn ensure_default_work(&mut self) {
        if self.works.is_empty() {
            self.works.push(Work::default());
        }
    }

    /// Set the number of workers (convenience for single Work).
    pub fn workers(mut self, n: usize) -> Self {
        self.ensure_default_work();
        self.works[0].num_workers = Some(n);
        self
    }

    /// Set the work type (convenience for single Work).
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.ensure_default_work();
        self.works[0].work_type = wt;
        self
    }

    /// Set the scheduling policy (convenience for single Work).
    pub fn sched_policy(mut self, p: crate::workload::SchedPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].sched_policy = p;
        self
    }

    /// Set the per-worker affinity (convenience for single Work).
    pub fn affinity(mut self, a: crate::workload::AffinityKind) -> Self {
        self.ensure_default_work();
        self.works[0].affinity = a;
        self
    }

    /// Set the NUMA memory placement policy (convenience for single Work).
    pub fn mem_policy(mut self, p: crate::workload::MemPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags (convenience for single Work).
    pub fn mpol_flags(mut self, f: crate::workload::MpolFlags) -> Self {
        self.ensure_default_work();
        self.works[0].mpol_flags = f;
        self
    }

    /// When true, the gauntlet work_type override replaces each Work's work type.
    pub fn swappable(mut self, swappable: bool) -> Self {
        self.swappable = swappable;
        self
    }

    /// Attach a userspace payload binary that runs inside this cgroup
    /// alongside any synthetic [`Work`] groups. The payload spawns
    /// when the step enters `apply_setup` and is killed during
    /// step-teardown so the cgroup can be removed cleanly.
    ///
    /// Only
    /// [`PayloadKind::Binary`](crate::test_support::PayloadKind::Binary)
    /// payloads are accepted; passing a scheduler-kind
    /// [`Payload`](crate::test_support::Payload) panics with an
    /// actionable message.
    ///
    /// **Why panic at declaration time, not at spawn time?** Three
    /// reasons, all of which favor failing fast:
    /// 1. **Discovery-time surfacing.** `CgroupDef` builders run
    ///    during test construction, which nextest's `--list`
    ///    invocation reaches BEFORE any VM boot. A panic here
    ///    emits a full backtrace inside the test binary and
    ///    surfaces the offending call site immediately; a deferred
    ///    runtime error would require a KVM-capable host + a
    ///    kernel image + an initramfs build to observe — a 30+
    ///    second feedback loop for what is purely a
    ///    typed-API misuse.
    /// 2. **No side effects.** The panic happens before
    ///    `CgroupDef.payload = Some(p)` assignment runs, so the
    ///    in-progress builder is left in its prior (no-payload)
    ///    state. A caller that catches the panic via
    ///    `catch_unwind` sees a valid CgroupDef either way.
    /// 3. **Scheduler-kind is always a programming error here.**
    ///    `Payload::KERNEL_DEFAULT` in `CgroupDef::workload` is never a
    ///    legitimate use case — it means the author confused the
    ///    `scheduler` slot (test-level) with the `workload` slot
    ///    (cgroup-level). There is no recovery path; the only
    ///    resolution is editing the source.
    ///
    /// Scheduler-kind payloads in the step-level `Op::RunPayload`
    /// path bail with an `anyhow::Error` instead of panicking —
    /// that path runs during scenario execution where one bad op
    /// should not crash a whole test run.
    pub fn workload(mut self, p: &'static crate::test_support::Payload) -> Self {
        assert!(
            !p.is_scheduler(),
            "CgroupDef::workload called with a scheduler-kind Payload ({}); \
             CgroupDef.workload is for userspace binary payloads only. \
             Use #[ktstr_test(scheduler = ...)] for scheduler placement.",
            p.name,
        );
        self.payload = Some(p);
        self
    }
}

impl Default for CgroupDef {
    fn default() -> Self {
        Self {
            name: Cow::Borrowed("cg_0"),
            cpuset: None,
            works: vec![],
            swappable: false,
            payload: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Step / HoldSpec
// ---------------------------------------------------------------------------

/// How to produce the CgroupDefs for a step's setup phase.
pub enum Setup {
    /// Static list of cgroup definitions.
    Defs(Vec<CgroupDef>),
    /// Factory that generates definitions from the runtime context.
    Factory(fn(&Ctx) -> Vec<CgroupDef>),
}

impl Clone for Setup {
    fn clone(&self) -> Self {
        match self {
            Setup::Defs(defs) => Setup::Defs(defs.clone()),
            Setup::Factory(f) => Setup::Factory(*f),
        }
    }
}

impl std::fmt::Debug for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Setup::Defs(defs) => f.debug_tuple("Defs").field(defs).finish(),
            Setup::Factory(_) => f
                .debug_tuple("Factory")
                .field(&"fn(&Ctx) -> Vec<CgroupDef>")
                .finish(),
        }
    }
}

impl Setup {
    fn resolve(&self, ctx: &Ctx) -> Vec<CgroupDef> {
        match self {
            Setup::Defs(defs) => defs.clone(),
            Setup::Factory(f) => f(ctx),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Setup::Defs(defs) => defs.is_empty(),
            Setup::Factory(_) => false,
        }
    }
}

impl From<Vec<CgroupDef>> for Setup {
    fn from(defs: Vec<CgroupDef>) -> Self {
        Setup::Defs(defs)
    }
}

/// A sequence of ops followed by a hold period.
///
/// For non-`Loop` steps, `ops` are applied first, then `setup` cgroups
/// are created, configured, and populated. For `Loop` steps, `setup`
/// runs once before the ops loop. Use `Step::new` to create a step
/// with only ops (no setup).
#[derive(Clone, Debug)]
pub struct Step {
    /// Cgroup setup applied before (non-`Loop`) or once above (`Loop`)
    /// the ops list. Runtime cgroups are spawned from this spec.
    pub setup: Setup,
    /// Ordered operations applied each time the step body runs:
    /// cpuset edits, task moves, spawn/despawn, etc.
    pub ops: Vec<Op>,
    /// How long, and whether to loop, after the ops finish one pass.
    pub hold: HoldSpec,
}

impl Step {
    /// Create a step with ops only (no CgroupDef setup).
    pub fn new(ops: Vec<Op>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(vec![]),
            ops,
            hold,
        }
    }

    /// Create a step with CgroupDef setup and a hold period.
    ///
    /// Most steps only need cgroup definitions and a hold duration.
    /// Use [`set_ops`](Step::set_ops) to chain ops onto the step.
    pub fn with_defs(defs: Vec<CgroupDef>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(defs),
            ops: vec![],
            hold,
        }
    }

    /// Replace the ops for a step, consuming and returning it.
    ///
    /// Named `set_ops` rather than `with_ops` because the semantics
    /// are REPLACE, not EXTEND — contrast
    /// [`Backdrop::with_ops`](crate::scenario::backdrop::Backdrop::with_ops),
    /// which appends. A chained `Step::new(ops).set_ops(more)`
    /// drops `ops` and keeps only `more`.
    pub fn set_ops(mut self, ops: Vec<Op>) -> Self {
        self.ops = ops;
        self
    }

    /// Create a step that spawns a single userspace
    /// [`Payload`](crate::test_support::Payload) binary in the
    /// background and holds for the given duration before teardown.
    ///
    /// Shorthand for `Step::new(vec![Op::run_payload(payload,
    /// vec![])], hold)`. The returned step is chainable — add
    /// `.set_ops(...)` to replace the ops vec (note the
    /// REPLACE-not-EXTEND semantics), or use
    /// `Op::wait_payload(name)` / `Op::kill_payload(name)` on later
    /// steps to control the spawned child.
    ///
    /// Test authors who want the payload placed in a named cgroup
    /// should use `Op::run_payload_in_cgroup` directly; this
    /// convenience targets the common "one payload, whole step"
    /// shape.
    pub fn with_payload(payload: &'static crate::test_support::Payload, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(vec![]),
            ops: vec![Op::run_payload(payload, vec![])],
            hold,
        }
    }
}

/// How a step advances after its ops are applied. `Frac` and `Fixed`
/// hold for a duration; `Loop` repeatedly re-applies `Step::ops` at a
/// fixed interval instead of holding.
#[derive(Clone, Debug)]
pub enum HoldSpec {
    /// Fraction of the total scenario duration.
    Frac(f64),
    /// Fixed duration.
    Fixed(Duration),
    /// Repeat the step's ops in a loop at the given interval until the
    /// remaining scenario time is exhausted.
    Loop { interval: Duration },
}

impl HoldSpec {
    /// Hold for the full scenario duration (`Frac(1.0)`).
    pub const FULL: HoldSpec = HoldSpec::Frac(1.0);

    /// Reject hold values that are vacuous (no-op step) or would
    /// panic downstream.
    ///
    /// Rules:
    /// - `Fixed(Duration::ZERO)` — the step applies ops and then
    ///   immediately advances; workers get no run time before the
    ///   next step. Almost always a typo; reject.
    /// - `Frac(f)` with `!f.is_finite()` (NaN/Inf) — propagates into
    ///   `Duration::from_secs_f64(f)` which panics.
    /// - `Frac(f)` with `f <= 0.0` — zero is vacuous, negative
    ///   panics in `Duration::from_secs_f64`.
    /// - `Loop { interval: Duration::ZERO }` — busy-polls the
    ///   deadline loop without yielding; almost always a typo.
    pub fn validate(&self) -> std::result::Result<(), String> {
        match self {
            HoldSpec::Fixed(d) if d.is_zero() => {
                Err("HoldSpec::Fixed(Duration::ZERO) is vacuous — workers \
                     get no run time before the next step; use at least a \
                     few ms or drop the step entirely"
                    .into())
            }
            HoldSpec::Frac(f) if !f.is_finite() => Err(format!(
                "HoldSpec::Frac({f}) is not finite (NaN/Inf) — would \
                     panic in Duration::from_secs_f64"
            )),
            HoldSpec::Frac(f) if *f <= 0.0 => Err(format!(
                "HoldSpec::Frac({f}) must be > 0.0; negative values \
                     panic in Duration::from_secs_f64 and zero is vacuous"
            )),
            HoldSpec::Loop { interval } if interval.is_zero() => {
                Err("HoldSpec::Loop { interval: Duration::ZERO } would \
                     busy-spin the deadline check without yielding; use a \
                     non-zero interval"
                    .into())
            }
            _ => Ok(()),
        }
    }
}

impl Op {
    /// Return a unique bit index for each Op variant (for op_kinds bitmask).
    fn discriminant(&self) -> u32 {
        match self {
            Op::AddCgroup { .. } => 0,
            Op::RemoveCgroup { .. } => 1,
            Op::SetCpuset { .. } => 2,
            Op::ClearCpuset { .. } => 3,
            Op::SwapCpusets { .. } => 4,
            Op::Spawn { .. } => 5,
            Op::StopCgroup { .. } => 6,
            Op::SetAffinity { .. } => 7,
            Op::SpawnHost { .. } => 8,
            Op::MoveAllTasks { .. } => 9,
            Op::RunPayload { .. } => 10,
            Op::WaitPayload { .. } => 11,
            Op::KillPayload { .. } => 12,
        }
    }

    /// Create a new cgroup.
    pub fn add_cgroup(name: impl Into<Cow<'static, str>>) -> Self {
        Op::AddCgroup { name: name.into() }
    }

    /// Remove a cgroup (stops its workers first).
    pub fn remove_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::RemoveCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set a cgroup's cpuset.
    pub fn set_cpuset(cgroup: impl Into<Cow<'static, str>>, cpus: CpusetSpec) -> Self {
        Op::SetCpuset {
            cgroup: cgroup.into(),
            cpus,
        }
    }

    /// Clear a cgroup's cpuset (allow all CPUs).
    pub fn clear_cpuset(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::ClearCpuset {
            cgroup: cgroup.into(),
        }
    }

    /// Swap cpusets between two cgroups.
    pub fn swap_cpusets(a: impl Into<Cow<'static, str>>, b: impl Into<Cow<'static, str>>) -> Self {
        Op::SwapCpusets {
            a: a.into(),
            b: b.into(),
        }
    }

    /// Spawn workers in a cgroup.
    pub fn spawn(cgroup: impl Into<Cow<'static, str>>, work: Work) -> Self {
        Op::Spawn {
            cgroup: cgroup.into(),
            work,
        }
    }

    /// Stop all workers in a cgroup.
    pub fn stop_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::StopCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set worker affinity in a cgroup.
    pub fn set_affinity(cgroup: impl Into<Cow<'static, str>>, affinity: AffinityKind) -> Self {
        Op::SetAffinity {
            cgroup: cgroup.into(),
            affinity,
        }
    }

    /// Spawn workers in the parent cgroup.
    pub fn spawn_host(work: Work) -> Self {
        Op::SpawnHost { work }
    }

    /// Move all tasks from one cgroup to another.
    pub fn move_all_tasks(
        from: impl Into<Cow<'static, str>>,
        to: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::MoveAllTasks {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Spawn a [`Payload`](crate::test_support::Payload) binary in the
    /// background. `args` is appended to `payload.default_args`.
    /// Placement is inherited from the caller; use
    /// [`run_payload_in_cgroup`](Self::run_payload_in_cgroup) to put
    /// the child into a named cgroup.
    pub fn run_payload(payload: &'static crate::test_support::Payload, args: Vec<String>) -> Self {
        Op::RunPayload {
            payload,
            args,
            cgroup: None,
        }
    }

    /// Spawn a [`Payload`](crate::test_support::Payload) in the
    /// background and place the child in a cgroup (relative to the
    /// scenario's parent cgroup).
    pub fn run_payload_in_cgroup(
        payload: &'static crate::test_support::Payload,
        args: Vec<String>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::RunPayload {
            payload,
            args,
            cgroup: Some(cgroup.into()),
        }
    }

    /// Block until the payload named `name` exits, evaluate checks,
    /// and record metrics. Matches whichever cgroup the payload is
    /// in when exactly one copy of the name is live; bails when two
    /// or more copies are live (use
    /// [`wait_payload_in_cgroup`](Self::wait_payload_in_cgroup) to
    /// disambiguate).
    pub fn wait_payload(name: impl Into<Cow<'static, str>>) -> Self {
        Op::WaitPayload {
            name: name.into(),
            cgroup: None,
        }
    }

    /// Block until the payload named `name` that's running inside
    /// the given `cgroup` exits. Use this form when two or more
    /// copies of the same payload are live in different cgroups
    /// and a cgroup-less `wait_payload` would be ambiguous. An
    /// empty-string `cgroup` matches payloads that inherited their
    /// parent's placement (spawned via `Op::run_payload(..., cgroup:
    /// None)`); explicit names match payloads placed via
    /// [`Op::run_payload_in_cgroup`] or
    /// [`CgroupDef::workload`](crate::scenario::ops::CgroupDef::workload).
    pub fn wait_payload_in_cgroup(
        name: impl Into<Cow<'static, str>>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::WaitPayload {
            name: name.into(),
            cgroup: Some(cgroup.into()),
        }
    }

    /// SIGKILL the payload named `name`, evaluate checks, and record
    /// metrics. Matches the unique live copy by name; bails on
    /// ambiguity. See [`wait_payload`](Self::wait_payload) for the
    /// full ambiguity rules and
    /// [`kill_payload_in_cgroup`](Self::kill_payload_in_cgroup)
    /// for the disambiguating form.
    pub fn kill_payload(name: impl Into<Cow<'static, str>>) -> Self {
        Op::KillPayload {
            name: name.into(),
            cgroup: None,
        }
    }

    /// SIGKILL the payload named `name` that's running inside the
    /// given `cgroup`. See
    /// [`wait_payload_in_cgroup`](Self::wait_payload_in_cgroup) for
    /// the placement-matching contract.
    pub fn kill_payload_in_cgroup(
        name: impl Into<Cow<'static, str>>,
        cgroup: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::KillPayload {
            name: name.into(),
            cgroup: Some(cgroup.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// SHM writer for stimulus events
// ---------------------------------------------------------------------------

/// SHM ring writer for guest-to-host data transfer.
///
/// Prefers mmap of /dev/mem for zero-copy access. Falls back to
/// pread/pwrite when mmap of the E820 gap fails (common on kernels
/// that restrict mmap of non-RAM physical ranges).
enum ShmWriter {
    /// mmap succeeded — direct pointer access.
    Mapped {
        ptr: *mut u8,
        map_base: *mut libc::c_void,
        map_size: usize,
        shm_size: usize,
    },
    /// mmap failed — use pread/pwrite on the /dev/mem fd.
    Fd {
        fd: std::fs::File,
        shm_base: u64,
        shm_size: usize,
    },
}

impl ShmWriter {
    /// Try to open the SHM region. Returns None if SHM params are absent
    /// from /proc/cmdline or /dev/mem cannot be opened.
    fn try_open() -> Option<Self> {
        let cmdline = std::fs::read_to_string("/proc/cmdline").ok()?;
        let (shm_base, shm_size) = shm_ring::parse_shm_params_from_str(&cmdline)?;

        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;

        let fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_SYNC)
            .open("/dev/mem")
            .ok()?;

        match shm_ring::mmap_devmem(
            std::os::unix::io::AsRawFd::as_raw_fd(&fd),
            shm_base,
            shm_size,
        ) {
            Some(m) => Some(ShmWriter::Mapped {
                ptr: m.ptr,
                map_base: m.map_base,
                map_size: m.map_size,
                shm_size: shm_size as usize,
            }),
            None => {
                eprintln!(
                    "ktstr: SHM mmap failed ({}), using pread/pwrite fallback",
                    std::io::Error::last_os_error(),
                );
                Some(ShmWriter::Fd {
                    fd,
                    shm_base,
                    shm_size: shm_size as usize,
                })
            }
        }
    }

    /// Write a TLV message to the SHM ring.
    ///
    /// Acquires `SHM_WRITE_LOCK` to serialize against concurrent writers
    /// (sched-exit-mon thread via `write_msg`).
    fn write(&self, msg_type: u32, payload: &[u8]) {
        // Recover from poisoning: the ring is fully overwritten on
        // each write, so a panicking writer does not leave shared
        // invariants in a bad state.
        let _guard = shm_ring::SHM_WRITE_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match self {
            ShmWriter::Mapped { ptr, shm_size, .. } => {
                let buf = unsafe { std::slice::from_raw_parts_mut(*ptr, *shm_size) };
                shm_ring::shm_write(buf, 0, msg_type, payload);
            }
            ShmWriter::Fd {
                fd,
                shm_base,
                shm_size,
            } => {
                use std::os::unix::io::AsRawFd;

                // Read current SHM state, apply the ring write, write back.
                let mut buf = vec![0u8; *shm_size];
                let n = unsafe {
                    libc::pread(
                        fd.as_raw_fd(),
                        buf.as_mut_ptr() as *mut libc::c_void,
                        buf.len(),
                        *shm_base as libc::off_t,
                    )
                };
                if n < 0 {
                    return;
                }

                shm_ring::shm_write(&mut buf, 0, msg_type, payload);

                unsafe {
                    libc::pwrite(
                        fd.as_raw_fd(),
                        buf.as_ptr() as *const libc::c_void,
                        buf.len(),
                        *shm_base as libc::off_t,
                    );
                }
            }
        }
    }
}

impl Drop for ShmWriter {
    fn drop(&mut self) {
        if let ShmWriter::Mapped {
            map_base, map_size, ..
        } = self
        {
            unsafe {
                libc::munmap(*map_base, *map_size);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CpusetSpec resolution
// ---------------------------------------------------------------------------

impl CpusetSpec {
    /// Check whether this spec can produce a non-empty cpuset for the
    /// given topology. Returns `Err` with a human-readable reason on
    /// failure.
    pub fn validate(&self, ctx: &Ctx) -> std::result::Result<(), String> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) if *idx >= ctx.topo.num_llcs() => Err(format!(
                "Llc({idx}) out of range: topology has {} LLCs",
                ctx.topo.num_llcs()
            )),
            CpusetSpec::Numa(node) if *node >= ctx.topo.num_numa_nodes() => Err(format!(
                "Numa({node}) out of range: topology has {} NUMA nodes",
                ctx.topo.num_numa_nodes()
            )),
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. } if *of == 0 => {
                Err("partition count (of) must be > 0".into())
            }
            CpusetSpec::Disjoint { index, of, .. } | CpusetSpec::Overlap { index, of, .. }
                if *index >= *of =>
            {
                Err(format!("index {index} >= partition count {of}"))
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if !start_frac.is_finite() || !end_frac.is_finite() => Err(format!(
                "Range start_frac ({start_frac}) or end_frac ({end_frac}) is not finite"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if *start_frac < 0.0 || *end_frac > 1.0 => Err(format!(
                "Range fracs must lie in [0.0, 1.0]: start_frac={start_frac}, end_frac={end_frac}"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if start_frac >= end_frac => Err(format!(
                "Range start_frac ({start_frac}) >= end_frac ({end_frac})"
            )),
            CpusetSpec::Overlap { frac, .. } if !frac.is_finite() => {
                Err(format!("Overlap frac ({frac}) is not finite"))
            }
            CpusetSpec::Overlap { frac, .. } if *frac < 0.0 || *frac > 1.0 => {
                Err(format!("Overlap frac ({frac}) must lie in [0.0, 1.0]"))
            }
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. }
                if usable.len() < *of =>
            {
                Err(format!(
                    "not enough usable CPUs ({}) for {} partitions",
                    usable.len(),
                    of
                ))
            }
            CpusetSpec::Exact(cpus) if cpus.is_empty() => {
                Err("CpusetSpec::Exact(empty) would assign no CPUs to the \
                 cgroup; cpuset.cpus rejects an empty mask and the \
                 cgroup would become unschedulable"
                    .into())
            }
            CpusetSpec::Exact(cpus) => {
                // Reject only CPUs the topology doesn't physically have
                // (`all_cpuset`), not the ones outside `usable_cpuset`.
                // A scheduler author may intentionally pin to an
                // isolated CPU (e.g. the root-reserved one) for
                // testing; writing it to cpuset.cpus is a legitimate
                // operation and the kernel is the final authority on
                // whether the write succeeds. Only truly-nonexistent
                // CPU indices are guaranteed to produce EINVAL.
                let all = ctx.topo.all_cpuset();
                let missing: Vec<usize> =
                    cpus.iter().copied().filter(|c| !all.contains(c)).collect();
                if !missing.is_empty() {
                    return Err(format!(
                        "CpusetSpec::Exact contains CPU(s) {missing:?} \
                         outside the topology's physical CPU set (max \
                         CPU index: {}); writing them to cpuset.cpus \
                         would fail with EINVAL",
                        all.iter().next_back().copied().unwrap_or(0),
                    ));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Resolve to a concrete CPU set given the topology.
    ///
    /// **Callers MUST run [`Self::validate`] first and propagate its error.**
    /// `apply_setup` and `apply_ops::SetCpuset` do so via `anyhow::bail!`.
    ///
    /// Defense-in-depth: every malformed input that `validate`
    /// rejects (out-of-range `Llc`/`Numa`, partition `of == 0`,
    /// `index >= of`, inverted or non-finite `Range.start_frac` /
    /// `end_frac`, out-of-bounds `Overlap.frac`) also has a
    /// panic-free fallback here — out-of-range indices clamp to the
    /// last valid index with a `tracing::warn!`, `of == 0` returns
    /// an empty set with a warn, and inverted/non-finite fracs clamp
    /// to `[0, len]` so the resulting slice never inverts. Skipping
    /// `validate` therefore degrades into a usable (possibly empty)
    /// cpuset rather than crashing the caller.
    pub fn resolve(&self, ctx: &Ctx) -> BTreeSet<usize> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) => {
                if *idx >= ctx.topo.num_llcs() {
                    // Graceful fallback: clamp to last LLC instead of panicking.
                    let clamped = ctx.topo.num_llcs().saturating_sub(1);
                    tracing::warn!(
                        llc_idx = idx,
                        num_llcs = ctx.topo.num_llcs(),
                        clamped,
                        "CpusetSpec::Llc index out of range, clamping",
                    );
                    ctx.topo.llc_aligned_cpuset(clamped)
                } else {
                    ctx.topo.llc_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Numa(idx) => {
                if *idx >= ctx.topo.num_numa_nodes() {
                    let clamped = ctx.topo.num_numa_nodes().saturating_sub(1);
                    tracing::warn!(
                        numa_node = idx,
                        num_numa_nodes = ctx.topo.num_numa_nodes(),
                        clamped,
                        "CpusetSpec::Numa index out of range, clamping",
                    );
                    ctx.topo.numa_aligned_cpuset(clamped)
                } else {
                    ctx.topo.numa_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } => {
                let len = usable.len();
                // Defense-in-depth: clamp non-finite fracs to 0 (NaN
                // would saturate to 0 via `as usize` anyway; explicit
                // check matches validate's rejection reason).
                let sf = if start_frac.is_finite() {
                    *start_frac
                } else {
                    0.0
                };
                let ef = if end_frac.is_finite() { *end_frac } else { 0.0 };
                let start = (len as f64 * sf) as usize;
                let end = (len as f64 * ef) as usize;
                // Guard against inverted Range (start_frac > end_frac)
                // — `&usable[start..end]` panics when start > end even
                // if both are clamped to `len`. `start.min(end)` keeps
                // the slice empty in that case instead of panicking.
                let s = start.min(len);
                let e = end.min(len).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Disjoint { index, of } => {
                if *of == 0 {
                    // Defense-in-depth: `validate` rejects of==0 with a
                    // clear error. If a caller reaches `resolve` with
                    // of==0 anyway (skipped validate, or used a
                    // malformed programmatic spec), returning an empty
                    // set is safer than the div-by-zero panic.
                    tracing::warn!("CpusetSpec::Disjoint with of=0 — returning empty cpuset");
                    return BTreeSet::new();
                }
                let chunk = usable.len() / of;
                let start = index * chunk;
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    (index + 1) * chunk
                };
                let s = start.min(usable.len());
                let e = end.min(usable.len()).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Overlap { index, of, frac } => {
                if *of == 0 {
                    tracing::warn!("CpusetSpec::Overlap with of=0 — returning empty cpuset");
                    return BTreeSet::new();
                }
                let chunk = usable.len() / of;
                // Clamp non-finite / out-of-range frac to 0 so the
                // overlap computation stays bounded.
                let frac = if frac.is_finite() {
                    frac.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let overlap = (chunk as f64 * frac) as usize;
                let start = if *index == 0 {
                    0
                } else {
                    (index * chunk).saturating_sub(overlap)
                };
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    ((index + 1) * chunk + overlap).min(usable.len())
                };
                let s = start.min(usable.len());
                let e = end.min(usable.len()).max(s);
                usable[s..e].iter().copied().collect()
            }
            CpusetSpec::Exact(cpus) => cpus.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Step executor
// ---------------------------------------------------------------------------

/// Persistent scenario-wide state owned by
/// [`execute_scenario_with`]. Lives for the entire step sequence;
/// cgroups, workload handles, and payload handles declared by the
/// [`Backdrop`](super::backdrop::Backdrop) go here and only tear
/// down at scenario end (success or Err). See [`StepState`] for
/// the step-local counterpart.
struct BackdropState<'a> {
    /// RAII cgroup guard for persistent cgroups — removes them on drop.
    cgroups: CgroupGroup<'a>,
    /// Active workload handles in persistent cgroups, keyed by cgroup name.
    handles: Vec<(String, WorkloadHandle)>,
    /// Resolved cpusets per persistent cgroup name.
    cpusets: std::collections::HashMap<String, BTreeSet<usize>>,
    /// Active payload-binary handles owned by the backdrop. Drained
    /// via `.kill()` at scenario teardown so the metric-emission
    /// pipeline still fires.
    payload_handles: Vec<PayloadEntry>,
}

impl<'a> BackdropState<'a> {
    /// Empty backdrop state (no persistent entities), scoped to `ctx.cgroups`.
    fn empty(ctx: &'a Ctx) -> Self {
        Self {
            cgroups: CgroupGroup::new(ctx.cgroups),
            handles: Vec::new(),
            cpusets: std::collections::HashMap::new(),
            payload_handles: Vec::new(),
        }
    }
}

/// Step-local execution state. Fresh per step, torn down at step
/// boundary: cgroups removed (via RAII drop), workload handles
/// collected, payload handles killed with metric emission. Any ops
/// in the step that reference a cgroup name look here first before
/// falling through to [`BackdropState`].
struct StepState<'a> {
    /// RAII cgroup guard — removes step-local cgroups on drop.
    cgroups: CgroupGroup<'a>,
    /// Active workload handles keyed by step-local cgroup name.
    handles: Vec<(String, WorkloadHandle)>,
    /// Resolved cpusets per step-local cgroup name, for isolation checks.
    cpusets: std::collections::HashMap<String, BTreeSet<usize>>,
    /// Active payload-binary handles keyed by cgroup name. Each entry
    /// came from either a [`CgroupDef::workload`] spawn in
    /// `apply_setup` or an explicit [`Op::RunPayload`] invocation;
    /// `source` tags which path spawned it so the duplicate-name
    /// dedup in `Op::RunPayload` can point at the original site. All
    /// are killed during step-teardown / cgroup removal so cgroupfs
    /// cleanup never trips EBUSY on a live process.
    payload_handles: Vec<PayloadEntry>,
}

impl<'a> StepState<'a> {
    /// Empty step state scoped to `ctx.cgroups`.
    fn empty(ctx: &'a Ctx) -> Self {
        Self {
            cgroups: CgroupGroup::new(ctx.cgroups),
            handles: Vec::new(),
            cpusets: std::collections::HashMap::new(),
            payload_handles: Vec::new(),
        }
    }
}

/// Combined mutable view over step-local and backdrop state.
///
/// Every function that touches execution state (apply_setup,
/// apply_ops, the drain helpers) receives a
/// `ScenarioState`; lookups prefer step-local, falling through to
/// backdrop. New state created via ops/setup inside a step writes
/// to step-local by default — that is the primary mechanism
/// enforcing per-step bounded lifetime. Setup for the Backdrop
/// itself (run once before the step loop) writes straight to the
/// backdrop side via [`ScenarioState::with_target_backdrop`].
struct ScenarioState<'a, 'b> {
    step: &'b mut StepState<'a>,
    backdrop: &'b mut BackdropState<'a>,
    /// When true, all mutations route to [`Self::backdrop`] instead
    /// of [`Self::step`]. Set by [`Self::with_target_backdrop`] when
    /// running the Backdrop's initial `apply_setup` / `apply_ops`
    /// before the first step.
    target_backdrop: bool,
}

impl<'a, 'b> ScenarioState<'a, 'b> {
    /// Build a combined scenario view. Starts with the step-local
    /// slot as the mutation target — call [`Self::with_target_backdrop`]
    /// to flip into backdrop-setup mode for Backdrop's own
    /// apply_setup / apply_ops pass.
    fn new(step: &'b mut StepState<'a>, backdrop: &'b mut BackdropState<'a>) -> Self {
        Self {
            step,
            backdrop,
            target_backdrop: false,
        }
    }

    /// Run `f` with writes routed to the backdrop side.
    fn with_target_backdrop<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.target_backdrop;
        self.target_backdrop = true;
        let r = f(self);
        self.target_backdrop = prev;
        r
    }

    /// `cgroups` group that receives newly-created cgroups. Step-local
    /// by default; backdrop when [`Self::with_target_backdrop`] is active.
    fn target_cgroups(&mut self) -> &mut CgroupGroup<'a> {
        if self.target_backdrop {
            &mut self.backdrop.cgroups
        } else {
            &mut self.step.cgroups
        }
    }

    /// `handles` vec that receives newly-spawned workload handles.
    fn target_handles(&mut self) -> &mut Vec<(String, WorkloadHandle)> {
        if self.target_backdrop {
            &mut self.backdrop.handles
        } else {
            &mut self.step.handles
        }
    }

    /// `cpusets` map that receives resolved cpusets for new cgroups.
    fn target_cpusets(&mut self) -> &mut std::collections::HashMap<String, BTreeSet<usize>> {
        if self.target_backdrop {
            &mut self.backdrop.cpusets
        } else {
            &mut self.step.cpusets
        }
    }

    /// `payload_handles` vec that receives newly-spawned payload handles.
    fn target_payload_handles(&mut self) -> &mut Vec<PayloadEntry> {
        if self.target_backdrop {
            &mut self.backdrop.payload_handles
        } else {
            &mut self.step.payload_handles
        }
    }

    /// Resolved cpuset for a cgroup name, looked up step-first then backdrop.
    fn lookup_cpuset(&self, name: &str) -> Option<&BTreeSet<usize>> {
        self.step
            .cpusets
            .get(name)
            .or_else(|| self.backdrop.cpusets.get(name))
    }

    /// Returns the live payload handle matching the composite key
    /// (`payload_name`, `cgroup_key`) from either step-local or
    /// backdrop state, or `None` when no entry matches. Used for
    /// the `Op::RunPayload` duplicate guard, which now treats
    /// "same payload in a different cgroup" as legitimate rather
    /// than a name collision.
    fn find_live_payload_with_cgroup(
        &self,
        payload_name: &str,
        cgroup_key: &str,
    ) -> Option<&PayloadEntry> {
        let matches =
            |e: &&PayloadEntry| e.handle.payload_name() == payload_name && e.cgroup == cgroup_key;
        self.step
            .payload_handles
            .iter()
            .find(matches)
            .or_else(|| self.backdrop.payload_handles.iter().find(matches))
    }

    /// Drop a payload handle by composite key (`name`, optional
    /// `cgroup`). Checks step-local first, then backdrop.
    ///
    /// - `cgroup = Some(c)`: exact match on both name and cgroup.
    /// - `cgroup = None`: if exactly one entry matches `name` across
    ///   both slots, consume it (backward-compat for
    ///   `Op::wait_payload(name)` / `Op::kill_payload(name)` when
    ///   only one copy is live). If two or more match, returns
    ///   `Err(ambiguous_cgroups)` where `ambiguous_cgroups` is the
    ///   list of cgroup keys for the candidates so the caller can
    ///   produce an actionable error.
    ///
    /// Returns `Ok(None)` when no entry matches.
    fn take_payload_by_name(
        &mut self,
        name: &str,
        cgroup: Option<&str>,
    ) -> std::result::Result<Option<PayloadEntry>, Vec<String>> {
        if let Some(c) = cgroup {
            // Composite-key path: exact match on both.
            if let Some(idx) = self
                .step
                .payload_handles
                .iter()
                .position(|e| e.handle.payload_name() == name && e.cgroup == c)
            {
                return Ok(Some(self.step.payload_handles.swap_remove(idx)));
            }
            if let Some(idx) = self
                .backdrop
                .payload_handles
                .iter()
                .position(|e| e.handle.payload_name() == name && e.cgroup == c)
            {
                return Ok(Some(self.backdrop.payload_handles.swap_remove(idx)));
            }
            return Ok(None);
        }
        // Name-only path: disambiguate across both slots before
        // consuming, so a mid-test wait on an ambiguous name
        // surfaces the caller's bug rather than silently waiting
        // on the first match.
        let mut step_idx: Option<usize> = None;
        let mut backdrop_idx: Option<usize> = None;
        let mut cgroups: Vec<String> = Vec::new();
        for (i, e) in self.step.payload_handles.iter().enumerate() {
            if e.handle.payload_name() == name {
                if step_idx.is_none() {
                    step_idx = Some(i);
                }
                cgroups.push(e.cgroup.clone());
            }
        }
        for (i, e) in self.backdrop.payload_handles.iter().enumerate() {
            if e.handle.payload_name() == name {
                if backdrop_idx.is_none() && step_idx.is_none() {
                    backdrop_idx = Some(i);
                }
                cgroups.push(e.cgroup.clone());
            }
        }
        if cgroups.len() > 1 {
            return Err(cgroups);
        }
        if let Some(i) = step_idx {
            return Ok(Some(self.step.payload_handles.swap_remove(i)));
        }
        if let Some(i) = backdrop_idx {
            return Ok(Some(self.backdrop.payload_handles.swap_remove(i)));
        }
        Ok(None)
    }

    /// Drain every live payload handle in step + backdrop state by
    /// calling `.kill()` so the metric-emission pipeline fires. Used
    /// on error paths in the step loop so mid-scenario failure still
    /// leaves a usable sidecar.
    fn drain_all_payloads(&mut self) {
        drain_all_payload_handles(&mut self.step.payload_handles);
        drain_all_payload_handles(&mut self.backdrop.payload_handles);
    }

    /// Kill every payload handle (step-first, then backdrop) whose
    /// cgroup matches `cgroup`. Called before a cgroup removal so
    /// cgroupfs cleanup does not trip EBUSY on a live process.
    fn drain_payloads_for_cgroup(&mut self, cgroup: &str) {
        drain_payload_handles_for_cgroup(&mut self.step.payload_handles, cgroup);
        drain_payload_handles_for_cgroup(&mut self.backdrop.payload_handles, cgroup);
    }

    /// Remove every workload handle whose key matches `cgroup`. The
    /// handles themselves drop (which SIGKILLs the workers) — this is
    /// appropriate for `Op::StopCgroup` and `Op::RemoveCgroup`.
    fn drop_handles_for_cgroup(&mut self, cgroup: &str) {
        self.step.handles.retain(|(n, _)| n.as_str() != cgroup);
        self.backdrop.handles.retain(|(n, _)| n.as_str() != cgroup);
    }

    /// Forget a tracked cpuset (step-first, then backdrop) for a cgroup.
    fn forget_cpuset(&mut self, cgroup: &str) {
        self.step.cpusets.remove(cgroup);
        self.backdrop.cpusets.remove(cgroup);
    }

    /// Record / overwrite the resolved cpuset for a cgroup. If the
    /// cgroup is known to step-local state, the step-local entry
    /// updates; if it's known to backdrop, the backdrop entry
    /// updates; otherwise the entry goes into the currently-active
    /// target (step-local, or backdrop inside `with_target_backdrop`).
    fn record_cpuset(&mut self, cgroup: &str, cpuset: BTreeSet<usize>) {
        if self.step.cpusets.contains_key(cgroup) {
            self.step.cpusets.insert(cgroup.to_string(), cpuset);
        } else if self.backdrop.cpusets.contains_key(cgroup) {
            self.backdrop.cpusets.insert(cgroup.to_string(), cpuset);
        } else {
            self.target_cpusets().insert(cgroup.to_string(), cpuset);
        }
    }

    /// Re-key every workload handle from `from` to `to`. When `to`
    /// names a Backdrop-owned cgroup, step-local handles are also
    /// transferred into [`Self::backdrop`] so their lifetime extends
    /// to scenario end instead of dying at step teardown. Backdrop
    /// handles stay in the backdrop slot regardless of `to`.
    ///
    /// Called by `Op::MoveAllTasks` after the kernel-side
    /// `cgroup.procs` writes succeed so subsequent ops that address
    /// the moved workers by cgroup name find them under the new key
    /// and in the correct state slot.
    fn rename_handles(&mut self, from: &str, to: &str) {
        let to_is_backdrop = self.cgroup_name_is_backdrop(to);
        if to_is_backdrop {
            // Move step-local handles keyed under `from` into the
            // backdrop slot, re-keyed to `to`. Iterate in reverse so
            // swap_remove indices stay stable.
            let mut i = self.step.handles.len();
            while i > 0 {
                i -= 1;
                if self.step.handles[i].0.as_str() == from {
                    let (_, handle) = self.step.handles.swap_remove(i);
                    self.backdrop.handles.push((to.to_string(), handle));
                }
            }
        } else {
            // Step-local destination: keep ownership, just rename.
            for (name, _) in &mut self.step.handles {
                if name.as_str() == from {
                    *name = to.to_string();
                }
            }
        }
        // Backdrop handles are never demoted to step-local ownership
        // regardless of destination — a backdrop worker is declared
        // persistent and stays persistent for the scenario. Rename
        // in place so subsequent ops still find it under the new key.
        for (name, _) in &mut self.backdrop.handles {
            if name.as_str() == from {
                *name = to.to_string();
            }
        }
    }

    /// Iterate every live workload handle across step + backdrop.
    /// Used by `Op::MoveAllTasks` / `Op::SetAffinity` which act on
    /// whichever cgroup owns the handle without caring about which
    /// state slot it's in.
    fn all_handles(&self) -> impl Iterator<Item = &(String, WorkloadHandle)> {
        self.step.handles.iter().chain(self.backdrop.handles.iter())
    }

    /// True iff a cgroup with the given name is already tracked by
    /// either step-local or backdrop state. Used to reject duplicate
    /// names at `apply_setup` time so a user can't accidentally
    /// shadow a Backdrop cgroup with a step-local `CgroupDef`.
    fn cgroup_name_is_tracked(&self, name: &str) -> bool {
        self.step.cgroups.names().iter().any(|n| n == name)
            || self.backdrop.cgroups.names().iter().any(|n| n == name)
    }

    /// True iff a cgroup with the given name is tracked by backdrop
    /// (persistent) state. Used by `Op::RemoveCgroup` to reject a
    /// step-local op that would remove a Backdrop-owned cgroup out
    /// from under later Steps.
    fn cgroup_name_is_backdrop(&self, name: &str) -> bool {
        self.backdrop.cgroups.names().iter().any(|n| n == name)
    }
}

/// Whether a live payload handle was spawned by an explicit
/// [`Op::RunPayload`] inside the step or by a
/// [`CgroupDef::workload`] attachment at `apply_setup`. Held by
/// every [`PayloadEntry`] so the dedup path in `Op::RunPayload`
/// can name the original source when rejecting a second spawn of
/// the same name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PayloadSource {
    /// Spawned by `CgroupDef::workload(&payload)` during `apply_setup`.
    CgroupDefWorkload,
    /// Spawned by `Op::RunPayload { payload, .. }` inside the step's ops.
    OpRunPayload,
}

impl PayloadSource {
    /// Human-readable tag for error output. Describes the API surface
    /// that originated the spawn, not the internal dispatch site.
    fn describe(self) -> &'static str {
        match self {
            PayloadSource::CgroupDefWorkload => "CgroupDef::workload",
            PayloadSource::OpRunPayload => "Op::RunPayload",
        }
    }
}

/// One live payload handle plus the cgroup it runs inside and the
/// API surface that spawned it. `cgroup` is empty iff
/// `source == PayloadSource::OpRunPayload` was invoked without a
/// `cgroup = Some(...)` argument — in which case the payload runs
/// in whatever cgroup its parent process inherited (no explicit
/// placement).
struct PayloadEntry {
    cgroup: String,
    source: PayloadSource,
    handle: crate::scenario::payload_run::PayloadHandle,
}

/// Execute a single step with CgroupDefs that hold for the full duration.
///
/// Convenience wrapper around [`execute_steps`] for the common pattern
/// of creating cgroups and running them for [`HoldSpec::FULL`].
pub fn execute_defs(ctx: &Ctx, defs: Vec<CgroupDef>) -> Result<AssertResult> {
    execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])
}

/// Execute a sequence of steps against the given context.
///
/// Convenience wrapper around [`execute_steps_with`] that passes
/// `None` for checks, falling back to `ctx.assert`. Use
/// [`execute_steps_with`] when you need to override `ctx.assert`.
pub fn execute_steps(ctx: &Ctx, steps: Vec<Step>) -> Result<AssertResult> {
    execute_steps_with(ctx, steps, None)
}

/// Execute a [`Backdrop`](super::backdrop::Backdrop) + Steps sequence
/// against the given context.
///
/// The Backdrop declares persistent scenario-wide state
/// (long-running payloads, cgroups referenced by many Steps) while
/// Steps express bounded per-phase behavior. The runtime sets up
/// the Backdrop before the first Step, runs the Step sequence
/// with per-Step teardown (cgroups removed, workload handles
/// collected, payload handles killed at step boundary), and tears
/// the Backdrop down at the end.
pub fn execute_scenario(
    ctx: &Ctx,
    backdrop: super::backdrop::Backdrop,
    steps: Vec<Step>,
) -> Result<AssertResult> {
    execute_scenario_with(ctx, backdrop, steps, None)
}

/// [`execute_scenario`] with an explicit
/// [`Assert`](crate::assert::Assert) override — the Backdrop
/// equivalent of [`execute_steps_with`].
pub fn execute_scenario_with(
    ctx: &Ctx,
    backdrop: super::backdrop::Backdrop,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    run_scenario(ctx, backdrop, steps, checks)
}

/// Execute steps with an explicit [`Assert`](crate::assert::Assert) for
/// worker checks. When `checks` is `Some`, it overrides `ctx.assert`.
/// When `None`, uses `ctx.assert` (the merged three-layer config).
///
/// Thin wrapper around [`execute_scenario_with`] with an empty
/// [`Backdrop`](super::backdrop::Backdrop) — every Step's effects
/// (cgroups, workloads, payloads) tear down at the step boundary.
pub fn execute_steps_with(
    ctx: &Ctx,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    execute_scenario_with(ctx, super::backdrop::Backdrop::EMPTY, steps, checks)
}

/// Internal driver: runs Backdrop setup, the Step loop with
/// per-Step teardown, and final Backdrop teardown.
fn run_scenario(
    ctx: &Ctx,
    backdrop: super::backdrop::Backdrop,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    // Validate every step's hold spec up front so a typo doesn't
    // reach `Duration::from_secs_f64(NaN)` / `thread::sleep(ZERO)` /
    // a no-yield Loop busy-wait after ops have already been applied.
    for (i, step) in steps.iter().enumerate() {
        if let Err(reason) = step.hold.validate() {
            anyhow::bail!("step {i} hold validation: {reason}");
        }
    }
    // Validate Backdrop payloads before creating any runtime state.
    // Only binary payloads can be spawned by Op::RunPayload, which
    // is what the Backdrop setup uses under the hood. Reject
    // scheduler-kind payloads here so the failure surface is the
    // Backdrop declaration, not a mid-scenario spawn error after
    // cgroups have already been created.
    for p in &backdrop.payloads {
        if p.is_scheduler() {
            anyhow::bail!(
                "Backdrop::with_payload received scheduler-kind Payload '{}' — \
                 only PayloadKind::Binary payloads run in the Backdrop; \
                 place scheduler-kind payloads on the #[ktstr_test(scheduler = ...)] \
                 attribute instead",
                p.name,
            );
        }
    }
    // Scheduler-kind payloads smuggled via Backdrop::with_op(Op::RunPayload { ... })
    // would otherwise bypass the check above and only bail deep inside
    // apply_ops. Reject them here with a Backdrop-specific error so
    // the failure surface matches the declaration surface.
    for op in &backdrop.ops {
        if let Op::RunPayload { payload, .. } = op
            && payload.is_scheduler()
        {
            anyhow::bail!(
                "Backdrop::with_op(Op::RunPayload) received scheduler-kind Payload '{}' — \
                 only PayloadKind::Binary payloads run in the Backdrop; \
                 place scheduler-kind payloads on the #[ktstr_test(scheduler = ...)] \
                 attribute instead",
                payload.name,
            );
        }
    }
    let effective_checks = checks.unwrap_or(&ctx.assert);

    let mut backdrop_state = BackdropState::empty(ctx);
    let mut result = AssertResult::pass();

    // Open SHM once for the entire step sequence. No-op outside a VM.
    let shm = ShmWriter::try_open();

    let scenario_start = std::time::Instant::now();

    // ScenarioStart marker.
    if let Some(ref w) = shm {
        w.write(shm_ring::MSG_TYPE_SCENARIO_START, &[]);
    }

    // When a host-side BPF map write is configured, signal the host that
    // probes are attached and the scenario is starting, then wait for the
    // host to complete the write before starting the workload.
    if ctx.wait_for_map_write {
        shm_ring::signal_value(1, shm_ring::SIGNAL_PROBES_READY);
        match shm_ring::wait_for(0, std::time::Duration::from_secs(10)) {
            Ok(()) => {
                // Brief delay for the crash trigger to propagate.
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(e) => {
                eprintln!("ktstr: signal slot 0 wait failed: {e} — proceeding without sync");
            }
        }
    }

    // --- Backdrop setup (persistent) ---
    // Run before the first Step. Cgroups + payloads declared on
    // `backdrop` land in `backdrop_state` so they survive every
    // Step's teardown. On error, drain Backdrop payload handles
    // (metric emission) and propagate.
    if !backdrop.is_empty() {
        let mut step_staging = StepState::empty(ctx);
        let mut scratch = ScenarioState::new(&mut step_staging, &mut backdrop_state);
        let setup_res = scratch.with_target_backdrop(|s| {
            // Order: cgroups → ops → payloads. CgroupDefs go first so
            // a later `Op::add_cgroup` / `Op::run_payload_in_cgroup`
            // can target cgroups that `apply_setup` just created.
            // Payloads spawn last so `run_payload` resolving a cgroup
            // placement lands inside a cgroup that either apply pass
            // already built.
            if !backdrop.cgroups.is_empty() {
                apply_setup(ctx, s, &backdrop.cgroups)?;
            }
            // Raw ops: typically `Op::AddCgroup` for empty move-target
            // cgroups (can't be expressed via CgroupDef because
            // apply_setup forces a worker spawn), or placement-aware
            // `Op::RunPayload` targeting a just-created backdrop
            // cgroup.
            if !backdrop.ops.is_empty() {
                apply_ops(ctx, s, &backdrop.ops)?;
            }
            // Shorthand payloads: one Op::RunPayload per entry,
            // inherited cgroup placement.
            if !backdrop.payloads.is_empty() {
                let ops: Vec<Op> = backdrop
                    .payloads
                    .iter()
                    .map(|p| Op::run_payload(p, Vec::<String>::new()))
                    .collect();
                apply_ops(ctx, s, &ops)?;
            }
            Ok::<(), anyhow::Error>(())
        });
        if let Err(err) = setup_res {
            // Collect any workers that DID spawn before the failure
            // so their stats reach the final result instead of being
            // discarded by `WorkloadHandle::drop` (which SIGKILLs
            // without gathering scheduler-side data). `collect_*`
            // drain `payload_handles` internally, so the backdrop-
            // and step-side payloads still get `.kill()` (SHM metric
            // emission) on the error path.
            //
            // `with_target_backdrop` routes every target writer to
            // the backdrop slot, so `step_staging` normally holds
            // nothing — but collect defensively so a partial-failure
            // path that leaks a non-backdrop write surfaces here
            // rather than disappearing into `StepState::drop`.
            let mut r = collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo);
            let staging_result = collect_step(&mut step_staging, effective_checks, ctx.topo);
            r.merge(staging_result);
            r.merge(result);
            // step_staging's CgroupGroup RAII still drops here,
            // removing any cgroups the failed Backdrop setup routed
            // into step-local state.
            r.passed = false;
            r.details.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("Backdrop setup failed: {err:#}"),
            ));
            return Ok(r);
        }
        // `step_staging` should not have accumulated anything
        // because `with_target_backdrop` routed every target writer
        // to the backdrop side. Collect any stray handles defensively
        // before dropping so a future refactor that leaks a non-
        // backdrop write here surfaces as a missed teardown rather
        // than silently discarded state.
        drain_all_payload_handles(&mut step_staging.payload_handles);
    }

    // --- Step loop with per-Step teardown ---
    for (step_idx, step) in steps.iter().enumerate() {
        // Check scheduler liveness between steps (skip before first).
        if step_idx > 0 && !process_alive(ctx.sched_pid) {
            // Collect backdrop-owned workload handles into the
            // result before reporting the crash so whatever the
            // persistent workers produced is still assertable.
            let mut r = collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo);
            r.merge(result);
            r.passed = false;
            r.details.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Monitor,
                format!(
                    "scheduler crashed after completing step {} of {} ({:.1}s into test)",
                    step_idx,
                    steps.len(),
                    scenario_start.elapsed().as_secs_f64(),
                ),
            ));
            return Ok(r);
        }

        let mut step_state = StepState::empty(ctx);
        let step_res = run_step(
            ctx,
            step,
            step_idx,
            &mut step_state,
            &mut backdrop_state,
            shm.as_ref(),
            scenario_start,
            effective_checks,
        );

        // Per-Step teardown ALWAYS runs — on success and on error.
        // This is the core of the "Step is fully bounded" invariant:
        // cgroups created this step go away, workload handles are
        // collected into the result, payload handles are killed
        // with metric emission. Backdrop state is untouched.
        let step_result = collect_step(&mut step_state, effective_checks, ctx.topo);
        result.merge(step_result);

        // A step-level error is converted into a failure on the
        // accumulated result after teardown has run so every step
        // boundary leaves clean state behind even on failure. The
        // caller keeps the prior-steps' merged AssertResult plus
        // the error context as a detail, instead of an opaque Err
        // that discards everything.
        if let Err(err) = step_res {
            // Collect Backdrop-owned workload handles into a fresh
            // result first, then merge the accumulated step result
            // on top. `collect_backdrop` drains
            // `backdrop_state.payload_handles` internally, so the
            // backdrop-side payloads still get `.kill()` (metric
            // emission) on the error path. Ordering mirrors the
            // scheduler-crash path above so detail order is
            // consistent across both Ok(failed) returns.
            let mut r = collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo);
            r.merge(result);
            r.passed = false;
            r.details.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Other,
                format!("step {step_idx} failed: {err:#}"),
            ));
            return Ok(r);
        }
    }

    // ScenarioEnd marker.
    if let Some(ref w) = shm {
        let elapsed = scenario_start.elapsed().as_millis() as u32;
        w.write(shm_ring::MSG_TYPE_SCENARIO_END, &elapsed.to_ne_bytes());
    }

    // Final liveness check.
    let sched_dead = !process_alive(ctx.sched_pid);

    // --- Backdrop teardown ---
    let backdrop_result = collect_backdrop(&mut backdrop_state, effective_checks, ctx.topo);
    result.merge(backdrop_result);

    if sched_dead {
        result.passed = false;
        result.details.push(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Monitor,
            format!(
                "scheduler crashed during test (detected after all {} steps completed, {:.1}s elapsed)",
                steps.len(),
                scenario_start.elapsed().as_secs_f64(),
            ),
        ));
    }

    Ok(result)
}

/// Run a single step's setup + ops + hold against step-local state.
///
/// On error, the caller is expected to invoke `collect_step` for
/// per-step teardown (which runs regardless) and then propagate.
#[allow(clippy::too_many_arguments)]
fn run_step<'a>(
    ctx: &Ctx,
    step: &Step,
    step_idx: usize,
    step_state: &mut StepState<'a>,
    backdrop_state: &mut BackdropState<'a>,
    shm: Option<&ShmWriter>,
    scenario_start: std::time::Instant,
    _effective_checks: &crate::assert::Assert,
) -> Result<()> {
    let mut scenario = ScenarioState::new(step_state, backdrop_state);

    // Any `?` out of apply_ops / apply_setup would bypass the
    // per-step teardown ordering; `drain_on_err!` kills payload
    // handles across step + backdrop (metric-emitting) before
    // propagating so a mid-scenario spawn failure still leaves a
    // usable sidecar.
    macro_rules! drain_on_err {
        ($scenario:expr, $e:expr) => {
            match $e {
                Ok(v) => v,
                Err(err) => {
                    $scenario.drain_all_payloads();
                    return Err(err);
                }
            }
        };
    }

    match &step.hold {
        HoldSpec::Loop { interval } => {
            // Setup runs once before the loop.
            if !step.setup.is_empty() {
                let defs = step.setup.resolve(ctx);
                drain_on_err!(scenario, apply_setup(ctx, &mut scenario, &defs));
            }
            // Loop mode: apply ops repeatedly at interval until
            // the remaining scenario time is exhausted.
            let deadline = scenario_start + ctx.duration;
            while std::time::Instant::now() < deadline {
                drain_on_err!(scenario, apply_ops(ctx, &mut scenario, &step.ops));
                let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                thread::sleep(remaining.min(*interval));
            }
        }
        _ => {
            // Ops first (e.g. parent cgroup creation), then
            // CgroupDef setup (children with workers).
            drain_on_err!(scenario, apply_ops(ctx, &mut scenario, &step.ops));
            if !step.setup.is_empty() {
                let defs = step.setup.resolve(ctx);
                drain_on_err!(scenario, apply_setup(ctx, &mut scenario, &defs));
            }

            // Write stimulus event after applying ops.
            if let Some(w) = shm {
                let payload = build_stimulus(&scenario_start, step_idx, &step.ops, &scenario);
                w.write(
                    shm_ring::MSG_TYPE_STIMULUS,
                    zerocopy::IntoBytes::as_bytes(&payload),
                );
            }

            let hold_dur = match &step.hold {
                HoldSpec::Frac(f) => Duration::from_secs_f64(ctx.duration.as_secs_f64() * f),
                HoldSpec::Fixed(d) => *d,
                HoldSpec::Loop { .. } => unreachable!(),
            };
            thread::sleep(hold_dur);
        }
    }

    Ok(())
}

/// Build a StimulusPayload from the current scenario state (step + backdrop).
fn build_stimulus(
    scenario_start: &std::time::Instant,
    step_idx: usize,
    ops: &[Op],
    state: &ScenarioState<'_, '_>,
) -> StimulusPayload {
    let mut op_kinds: u32 = 0;
    for op in ops {
        op_kinds |= 1 << op.discriminant();
    }

    let total_iterations: u64 = state
        .all_handles()
        .flat_map(|(_, h)| h.snapshot_iterations())
        .sum();

    let cgroup_count = state.step.cgroups.names().len() + state.backdrop.cgroups.names().len();
    let worker_count = state.step.handles.len() + state.backdrop.handles.len();

    StimulusPayload {
        elapsed_ms: scenario_start.elapsed().as_millis() as u32,
        step_index: step_idx as u16,
        op_count: ops.len() as u16,
        op_kinds,
        cgroup_count: cgroup_count as u16,
        worker_count: worker_count as u16,
        total_iterations,
    }
}

/// Create cgroups, set cpusets, and spawn workers from CgroupDefs.
///
/// Validate that a MemPolicy's nodes are covered by the NUMA nodes
/// reachable from the resolved cpuset. Returns `Err` with a description
/// when the policy requests nodes outside the cpuset's NUMA coverage.
fn validate_mempolicy_cpuset(
    policy: &MemPolicy,
    cpuset: &BTreeSet<usize>,
    ctx: &Ctx,
    cgroup_name: &str,
) -> Result<()> {
    let policy_nodes = policy.node_set();
    if policy_nodes.is_empty() {
        return Ok(());
    }
    let cpuset_numa = ctx.topo.numa_nodes_for_cpuset(cpuset);
    let uncovered: Vec<usize> = policy_nodes
        .iter()
        .copied()
        .filter(|n| !cpuset_numa.contains(n))
        .collect();
    if !uncovered.is_empty() {
        anyhow::bail!(
            "cgroup '{}': MemPolicy references NUMA node(s) {:?} \
             but cpuset covers only node(s) {:?}",
            cgroup_name,
            uncovered,
            cpuset_numa,
        );
    }
    Ok(())
}

/// Each CgroupDef's `works` vec is iterated, spawning one WorkloadHandle
/// per Work entry. Multiple Works for the same cgroup produce multiple
/// handle entries with the same name key; Ops that filter by cgroup name
/// (StopCgroup, SetAffinity, etc.) naturally apply to all of them.
///
/// When `works` is empty, a single default Work is used (CpuSpin, Normal,
/// ctx.workers_per_cgroup workers).
///
/// Cgroups created here route into step-local or backdrop state per
/// `state.target_backdrop`. A duplicate name (already tracked by
/// either state) bails — a `CgroupDef` must not silently shadow a
/// cgroup that another state slot has already created.
fn apply_setup(ctx: &Ctx, state: &mut ScenarioState<'_, '_>, defs: &[CgroupDef]) -> Result<()> {
    let default_work = [Work::default()];
    for def in defs {
        if state.cgroup_name_is_tracked(&def.name) {
            anyhow::bail!(
                "cgroup '{}' is already tracked (by a prior Backdrop or step-local CgroupDef) — \
                 declare it in exactly one place; use a fresh name for the step-local cgroup",
                def.name,
            );
        }
        state.target_cgroups().add_cgroup_no_cpuset(&def.name)?;
        if let Some(ref cpuset_spec) = def.cpuset {
            if let Err(reason) = cpuset_spec.validate(ctx) {
                anyhow::bail!(
                    "cgroup '{}': CpusetSpec validation failed: {}",
                    def.name,
                    reason
                );
            }
            let resolved = cpuset_spec.resolve(ctx);
            ctx.cgroups.set_cpuset(&def.name, &resolved)?;
            state.record_cpuset(&def.name, resolved);
        }
        let effective_works: &[Work] = if def.works.is_empty() {
            &default_work
        } else {
            &def.works
        };
        for work in effective_works {
            if let Err(reason) = work.mem_policy.validate() {
                anyhow::bail!("cgroup '{}': {}", def.name, reason);
            }
        }
        // Clone the cpuset out so we don't keep a borrow into
        // `state` across the mutable spawn calls below.
        let cgroup_cpuset: Option<BTreeSet<usize>> = state.lookup_cpuset(&def.name).cloned();
        if let Some(ref resolved) = cgroup_cpuset {
            for work in effective_works {
                validate_mempolicy_cpuset(&work.mem_policy, resolved, ctx, &def.name)?;
            }
        }
        for work in effective_works {
            let n = super::resolve_num_workers(work, ctx.workers_per_cgroup, &def.name)?;
            let effective_work_type = crate::workload::resolve_work_type(
                &work.work_type,
                ctx.work_type_override.as_ref(),
                def.swappable,
                n,
            );
            let affinity = super::resolve_affinity_for_cgroup(
                &work.affinity,
                cgroup_cpuset.as_ref(),
                ctx.topo,
            );
            let wl = WorkloadConfig {
                num_workers: n,
                affinity,
                work_type: effective_work_type,
                sched_policy: work.sched_policy,
                mem_policy: work.mem_policy.clone(),
                mpol_flags: work.mpol_flags,
            };
            let mut h = WorkloadHandle::spawn(&wl)?;
            ctx.cgroups.move_tasks(&def.name, &h.tids())?;
            h.start();
            state.target_handles().push((def.name.to_string(), h));
        }
        // After synthetic workers are in place, spawn the optional
        // userspace payload inside the same cgroup. The payload runs
        // concurrently with the Work groups; its metrics are recorded
        // to the sidecar via the guest-to-host SHM ring when the
        // handle is killed at step-teardown. Spawning after the Work
        // handles lets the cgroup cpuset + mempolicy settle first so
        // the binary inherits a stable placement.
        if let Some(payload) = def.payload {
            // Composite-key dedup: the same payload CAN live in a
            // different cgroup, but two copies in THIS cgroup would
            // collide on teardown (one handle masks the other in
            // the sidecar). Reject upfront with the same error
            // shape as the Op::RunPayload path.
            if let Some(existing) =
                state.find_live_payload_with_cgroup(payload.name, def.name.as_ref())
            {
                anyhow::bail!(
                    "CgroupDef::workload: payload '{}' already running in cgroup '{}' (spawned by {}) — \
                     declare it in exactly one place per cgroup",
                    payload.name,
                    def.name,
                    existing.source.describe(),
                );
            }
            let handle = crate::scenario::payload_run::PayloadRun::new(ctx, payload)
                .in_cgroup(def.name.clone())
                .spawn()
                .map_err(|e| {
                    anyhow::anyhow!(
                        "cgroup '{}': spawn payload '{}': {:#}",
                        def.name,
                        payload.name,
                        e,
                    )
                })?;
            state.target_payload_handles().push(PayloadEntry {
                cgroup: def.name.to_string(),
                source: PayloadSource::CgroupDefWorkload,
                handle,
            });
        }
    }
    Ok(())
}

/// Apply a slice of Ops to the running state.
///
/// Ops that create new entities (`AddCgroup`, `Spawn`, `SpawnHost`,
/// `RunPayload`) route into step-local state by default, or into
/// backdrop when the Backdrop's initial setup phase is active.
/// Ops that read or mutate existing entities (`SetCpuset`,
/// `ClearCpuset`, `SwapCpusets`, `SetAffinity`, `MoveAllTasks`,
/// `RemoveCgroup`, `StopCgroup`, `WaitPayload`, `KillPayload`)
/// resolve the target name against step-local first, then backdrop
/// — so a Step's ops can reach into Backdrop-declared cgroups by
/// name without the Backdrop leaking implementation details.
fn apply_ops(ctx: &Ctx, state: &mut ScenarioState<'_, '_>, ops: &[Op]) -> Result<()> {
    for op in ops {
        match op {
            Op::AddCgroup { name } => {
                state.target_cgroups().add_cgroup_no_cpuset(name)?;
            }
            Op::RemoveCgroup { cgroup } => {
                // A Step's ops must not remove a Backdrop-owned
                // cgroup — later Steps expect the Backdrop's cgroups
                // to survive every per-step teardown. Reject
                // explicitly so a typo in the cgroup name does not
                // silently dismantle a persistent cgroup and let
                // subsequent Steps fail with confusing
                // "cgroup missing" errors. Ops running during the
                // Backdrop's own setup pass (`target_backdrop`)
                // are exempt: the Backdrop is allowed to structure
                // its own state however it needs before the Step
                // loop starts.
                if !state.target_backdrop && state.cgroup_name_is_backdrop(cgroup) {
                    anyhow::bail!(
                        "Op::RemoveCgroup targets Backdrop-owned cgroup '{}' — \
                         Backdrop cgroups live for the full scenario and must \
                         not be removed from a Step; drop the op or move the \
                         cgroup declaration out of the Backdrop",
                        cgroup,
                    );
                }
                // Stop workers + payload binaries in this cgroup
                // before the cgroupfs removal. A live process in the
                // cgroup makes `rmdir` fail with EBUSY; kill the
                // payload handles first so the cgroup frees up.
                state.drain_payloads_for_cgroup(cgroup);
                state.drop_handles_for_cgroup(cgroup);
                state.forget_cpuset(cgroup);
                // ENOENT is expected here only as a TOCTOU outcome:
                // `CgroupManager::remove_cgroup` first checks
                // `p.exists()` and returns `Ok(())` when the dir is
                // already gone, so a clean "already removed by a
                // prior op" case never reaches this error arm. The
                // remaining ENOENT path is the narrow race where the
                // dir is unlinked by another process between
                // `exists()` and `fs::remove_dir(&p)`, which is
                // benign — the post-condition we want (no dir) still
                // holds. Every other error — EBUSY from a surviving
                // task, EACCES from a permissions regression, I/O
                // errors from a broken cgroupfs mount — gets logged
                // so the failure surfaces in test output instead of
                // being swallowed by `let _ = `.
                if let Err(err) = ctx.cgroups.remove_cgroup(cgroup) {
                    let is_enoent = err
                        .root_cause()
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound);
                    if !is_enoent {
                        tracing::warn!(
                            cgroup = %cgroup,
                            err = %format!("{err:#}"),
                            "Op::RemoveCgroup: rmdir returned non-ENOENT error",
                        );
                    }
                }
            }
            Op::SetCpuset { cgroup, cpus } => {
                if let Err(reason) = cpus.validate(ctx) {
                    anyhow::bail!(
                        "cgroup '{}': CpusetSpec validation failed: {}",
                        cgroup,
                        reason
                    );
                }
                let resolved = cpus.resolve(ctx);
                ctx.cgroups.set_cpuset(cgroup, &resolved)?;
                state.record_cpuset(cgroup, resolved);
            }
            Op::ClearCpuset { cgroup } => {
                ctx.cgroups.clear_cpuset(cgroup)?;
                state.forget_cpuset(cgroup);
            }
            Op::SwapCpusets { a, b } => {
                // Read current cpusets from the cgroup filesystem, swap them.
                let cpus_a = read_cpuset(ctx, a);
                let cpus_b = read_cpuset(ctx, b);
                if let Some(ca) = cpus_a {
                    ctx.cgroups.set_cpuset(b, &ca)?;
                    state.record_cpuset(b, ca);
                }
                if let Some(cb) = cpus_b {
                    ctx.cgroups.set_cpuset(a, &cb)?;
                    state.record_cpuset(a, cb);
                }
            }
            Op::Spawn { cgroup, work } => {
                if let Err(reason) = work.mem_policy.validate() {
                    anyhow::bail!("cgroup '{}': {}", cgroup, reason);
                }
                let n = super::resolve_num_workers(work, ctx.workers_per_cgroup, cgroup)?;
                let cgroup_cpuset: Option<BTreeSet<usize>> = state.lookup_cpuset(cgroup).cloned();
                if let Some(ref resolved) = cgroup_cpuset {
                    validate_mempolicy_cpuset(&work.mem_policy, resolved, ctx, cgroup)?;
                }
                let affinity = super::resolve_affinity_for_cgroup(
                    &work.affinity,
                    cgroup_cpuset.as_ref(),
                    ctx.topo,
                );
                let wl = WorkloadConfig {
                    num_workers: n,
                    affinity,
                    work_type: work.work_type.clone(),
                    sched_policy: work.sched_policy,
                    mem_policy: work.mem_policy.clone(),
                    mpol_flags: work.mpol_flags,
                };
                let mut h = WorkloadHandle::spawn(&wl)?;
                ctx.cgroups.move_tasks(cgroup, &h.tids())?;
                h.start();
                state.target_handles().push((cgroup.to_string(), h));
            }
            Op::StopCgroup { cgroup } => {
                // Same invariant as Op::RemoveCgroup: a Step's ops
                // must not stop a Backdrop-owned cgroup's workers.
                // `drain_payloads_for_cgroup` and
                // `drop_handles_for_cgroup` both touch step + backdrop
                // state by name, so a silent step-local stop would
                // kill persistent workers. Ops running inside the
                // Backdrop's own setup pass (`target_backdrop`)
                // stay exempt.
                if !state.target_backdrop && state.cgroup_name_is_backdrop(cgroup) {
                    anyhow::bail!(
                        "Op::StopCgroup targets Backdrop-owned cgroup '{}' — \
                         Backdrop workers live for the full scenario and must \
                         not be stopped from a Step; drop the op or move the \
                         cgroup declaration out of the Backdrop",
                        cgroup,
                    );
                }
                state.drain_payloads_for_cgroup(cgroup);
                state.drop_handles_for_cgroup(cgroup);
            }
            Op::SetAffinity { cgroup, affinity } => {
                let cgroup_cpuset: Option<BTreeSet<usize>> = state.lookup_cpuset(cgroup).cloned();
                let resolved =
                    super::resolve_affinity_for_cgroup(affinity, cgroup_cpuset.as_ref(), ctx.topo);
                for (name, handle) in state.all_handles() {
                    if name.as_str() == *cgroup {
                        match &resolved {
                            crate::workload::AffinityMode::None => {}
                            crate::workload::AffinityMode::Fixed(cpus) => {
                                for idx in 0..handle.tids().len() {
                                    let _ = handle.set_affinity(idx, cpus);
                                }
                            }
                            crate::workload::AffinityMode::Random { from, count }
                                if !from.is_empty() && *count > 0 =>
                            {
                                use rand::seq::IndexedRandom;
                                let v: Vec<usize> = from.iter().copied().collect();
                                for idx in 0..handle.tids().len() {
                                    let chosen: BTreeSet<usize> =
                                        v.sample(&mut rand::rng(), *count).copied().collect();
                                    let _ = handle.set_affinity(idx, &chosen);
                                }
                            }
                            // Empty pool OR count == 0: divergence from
                            // `workload::resolve_affinity`, which BAILS
                            // on count == 0 because it runs during
                            // workload setup where a zero-sample is a
                            // caller-config bug. Here at the
                            // step-execution layer the affinity was
                            // already applied at spawn and this op is
                            // re-applying; a zero-sample is a harmless
                            // skip, not a caller error. No-op rather
                            // than bail so a mid-run SetAffinity with
                            // a degenerate Random spec doesn't abort
                            // the whole scenario.
                            crate::workload::AffinityMode::Random { .. } => {}
                            crate::workload::AffinityMode::SingleCpu(cpu) => {
                                let cpus: BTreeSet<usize> = [*cpu].into_iter().collect();
                                for idx in 0..handle.tids().len() {
                                    let _ = handle.set_affinity(idx, &cpus);
                                }
                            }
                        }
                    }
                }
            }
            Op::SpawnHost { work } => {
                if let Err(reason) = work.mem_policy.validate() {
                    anyhow::bail!("SpawnHost: {}", reason);
                }
                let n = super::resolve_num_workers(work, ctx.workers_per_cgroup, "<host>")?;
                let affinity = super::resolve_affinity_for_cgroup(&work.affinity, None, ctx.topo);
                let wl = WorkloadConfig {
                    num_workers: n,
                    affinity,
                    work_type: work.work_type.clone(),
                    sched_policy: work.sched_policy,
                    mem_policy: work.mem_policy.clone(),
                    mpol_flags: work.mpol_flags,
                };
                let mut h = WorkloadHandle::spawn(&wl)?;
                h.start();
                // Empty string key: workers in parent cgroup, not a managed cgroup.
                state.target_handles().push((String::new(), h));
            }
            Op::MoveAllTasks { from, to } => {
                // A step-local MoveAllTasks that pulls from a
                // Backdrop-owned cgroup into a step-local cgroup
                // would strand persistent workers inside a cgroup
                // that gets rmdir'd at step boundary. Reject
                // explicitly. Ops running inside the Backdrop's
                // own setup pass (`target_backdrop`) stay exempt.
                if !state.target_backdrop
                    && state.cgroup_name_is_backdrop(from)
                    && !state.cgroup_name_is_backdrop(to)
                {
                    anyhow::bail!(
                        "Op::MoveAllTasks from Backdrop-owned '{}' to step-local '{}' \
                         would leave persistent workers in a cgroup that disappears \
                         at step boundary; declare `{}` in the Backdrop too, or \
                         move the workers back into a Backdrop-owned cgroup",
                        from,
                        to,
                        to,
                    );
                }
                // Clear subtree_control on the destination before moving
                // tasks. The kernel's no-internal-process constraint
                // (cgroup_migrate_vet_dst) returns EBUSY when writing to
                // cgroup.procs of a cgroup with subtree_control set.
                if let Err(e) = ctx.cgroups.clear_subtree_control(to) {
                    tracing::warn!(
                        cgroup = to.as_ref(),
                        err = %e,
                        "failed to clear subtree_control before task move"
                    );
                }
                // Perform the cgroup.procs writes for every handle
                // currently keyed under `from` (step-local and backdrop).
                for (name, handle) in state.all_handles() {
                    if name.as_str() == *from {
                        ctx.cgroups.move_tasks(to, &handle.tids())?;
                    }
                }
                // Re-key handles under `to` and transfer ownership
                // when required. A step-local handle whose `to`
                // names a Backdrop cgroup moves into the backdrop
                // slot so its lifetime extends with the destination
                // cgroup — without the transfer, the step's
                // teardown would SIGKILL the worker even though the
                // user moved it into a persistent cgroup. Backdrop
                // handles always stay in the backdrop slot
                // regardless of `to`; "Backdrop is persistent" does
                // not degrade to step-local ownership because a
                // later MoveAllTasks targets a step-local cgroup.
                state.rename_handles(from, to);
            }
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                if payload.is_scheduler() {
                    anyhow::bail!(
                        "Op::RunPayload called with scheduler-kind Payload ('{}'); \
                         only PayloadKind::Binary payloads can be spawned by step ops",
                        payload.name,
                    );
                }
                // Compute the cgroup key now so the composite-key
                // dedup sees the same `(name, cgroup)` pair the
                // spawn is about to record.
                let cgroup_key = cgroup.as_ref().map(|c| c.to_string()).unwrap_or_default();
                if let Some(existing) =
                    state.find_live_payload_with_cgroup(payload.name, &cgroup_key)
                {
                    // Same payload in the same cgroup is still a
                    // collision: two concurrent runs would write
                    // overlapping metrics to the sidecar and there's
                    // no way for a subsequent WaitPayload / KillPayload
                    // to tell them apart. Same payload in a DIFFERENT
                    // cgroup is now legitimate (placement-disambiguated).
                    // Name the surface that spawned the live handle
                    // so the user can find the original site without
                    // guessing.
                    anyhow::bail!(
                        "Op::RunPayload: payload '{}' already running in cgroup {} (spawned by {}) — \
                         WaitPayload/KillPayload it before spawning another with the same name in the same cgroup",
                        payload.name,
                        render_cgroup_key(&existing.cgroup),
                        existing.source.describe(),
                    );
                }
                let mut run = crate::scenario::payload_run::PayloadRun::new(ctx, payload);
                if !args.is_empty() {
                    run = run.args(args.iter().cloned());
                }
                if let Some(c) = cgroup {
                    run = run.in_cgroup(c.clone());
                }
                let handle = run.spawn().with_context(|| {
                    format!(
                        "Op::RunPayload: spawn payload '{}' in cgroup {}",
                        payload.name,
                        render_cgroup_key(&cgroup_key),
                    )
                })?;
                state.target_payload_handles().push(PayloadEntry {
                    cgroup: cgroup_key,
                    source: PayloadSource::OpRunPayload,
                    handle,
                });
            }
            Op::WaitPayload { name, cgroup } => {
                let entry = take_payload_for_op(
                    state,
                    "Op::WaitPayload",
                    "waiting",
                    "Op::wait_payload_in_cgroup",
                    name,
                    cgroup.as_deref(),
                )?;
                // Check verdicts + metrics are recorded to the sidecar
                // via the SHM ring inside `handle.wait()`; the returned
                // tuple is discarded here because step-ops surface per-
                // payload results through the sidecar, not the ops API.
                let _result = entry
                    .handle
                    .wait()
                    .with_context(|| format!("Op::WaitPayload: wait payload '{name}'"))?;
            }
            Op::KillPayload { name, cgroup } => {
                let entry = take_payload_for_op(
                    state,
                    "Op::KillPayload",
                    "killing",
                    "Op::kill_payload_in_cgroup",
                    name,
                    cgroup.as_deref(),
                )?;
                let _result = entry
                    .handle
                    .kill()
                    .with_context(|| format!("Op::KillPayload: kill payload '{name}'"))?;
            }
        }
    }
    Ok(())
}

/// Shared lookup for `Op::WaitPayload` / `Op::KillPayload`.
///
/// Consumes the payload handle matching the composite key
/// (`name`, `cgroup`). Produces the op-specific not-found /
/// ambiguous errors so the match arms stay short.
///
/// Callers pass the static trio that shapes the error text:
///
/// - `op_tag` — the user-facing op name (e.g. `"Op::WaitPayload"`).
/// - `verb_ing` — the `-ing` form of the action for "before
///   waiting" / "before killing" prose (no trailing
///   `to_lowercase` munging so two-word op names don't collide
///   into one word).
/// - `ctor_path` — the fully-qualified constructor the user
///   should switch to on ambiguity, e.g.
///   `"Op::wait_payload_in_cgroup"`. Copying this hint into
///   source must produce a callable path.
fn take_payload_for_op(
    state: &mut ScenarioState<'_, '_>,
    op_tag: &str,
    verb_ing: &str,
    ctor_path: &str,
    name: &str,
    cgroup: Option<&str>,
) -> Result<PayloadEntry> {
    match state.take_payload_by_name(name, cgroup) {
        Ok(Some(entry)) => Ok(entry),
        Ok(None) => match cgroup {
            Some(c) => anyhow::bail!(
                "{op_tag}: no running payload named '{name}' in cgroup {} \
                 (spawn it via Op::RunPayload or CgroupDef::workload before {verb_ing})",
                render_cgroup_key(c),
            ),
            None => anyhow::bail!(
                "{op_tag}: no running payload named '{name}' \
                 (spawn it via Op::RunPayload or CgroupDef::workload before {verb_ing})",
            ),
        },
        Err(cgroups) => {
            // Name-only lookup matched >1 live payload. Enumerate
            // the candidate cgroups so the caller knows which
            // qualified form they need.
            let rendered: Vec<String> = cgroups.iter().map(|c| render_cgroup_key(c)).collect();
            anyhow::bail!(
                "{op_tag}: payload '{name}' is ambiguous — {} live copies in cgroups {} — \
                 use {ctor_path}(name, cgroup) to disambiguate",
                rendered.len(),
                rendered.join(", "),
            )
        }
    }
}

/// Read the effective cpuset for a cgroup by reading cpuset.cpus.
fn read_cpuset(ctx: &Ctx, name: &str) -> Option<BTreeSet<usize>> {
    let path = ctx.cgroups.parent_path().join(name).join("cpuset.cpus");
    let content = std::fs::read_to_string(&path).ok()?;
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    let cpus: BTreeSet<usize> = crate::topology::parse_cpu_list_lenient(content)
        .into_iter()
        .collect();
    Some(cpus)
}

/// Collect step-local worker results and produce an AssertResult.
///
/// Drains step-local handles + payload handles; backdrop state is
/// untouched. Called at every step boundary (success AND error
/// paths) as the "Step is fully bounded" teardown. The
/// `step_state` goes out of scope at the end of this step's
/// iteration, so its `CgroupGroup` drop removes every step-local
/// cgroup immediately after `run_scenario` propagates the result
/// of this call.
fn collect_step(
    step_state: &mut StepState<'_>,
    checks: &crate::assert::Assert,
    topo: &crate::topology::TestTopology,
) -> AssertResult {
    // Kill any CgroupDef::workload / Op::RunPayload payload binaries
    // still live at step teardown so cgroupfs cleanup does not trip
    // EBUSY. Metrics are emitted to the SHM ring by PayloadHandle::kill
    // via the `evaluate()` pipeline.
    drain_all_payload_handles(&mut step_state.payload_handles);
    let handles = std::mem::take(&mut step_state.handles);
    super::collect_handles(
        handles
            .into_iter()
            .map(|(name, h)| (h, step_state.cpusets.get(&name))),
        checks,
        Some(topo),
    )
}

/// Collect backdrop (persistent) worker results. Called once at
/// scenario end after every Step has torn down. The
/// `backdrop_state.cgroups` RAII guard drops persistent cgroups
/// when `backdrop_state` itself drops.
fn collect_backdrop(
    backdrop_state: &mut BackdropState<'_>,
    checks: &crate::assert::Assert,
    topo: &crate::topology::TestTopology,
) -> AssertResult {
    drain_all_payload_handles(&mut backdrop_state.payload_handles);
    let handles = std::mem::take(&mut backdrop_state.handles);
    super::collect_handles(
        handles
            .into_iter()
            .map(|(name, h)| (h, backdrop_state.cpusets.get(&name))),
        checks,
        Some(topo),
    )
}

/// Kill every payload handle whose cgroup matches `cgroup` and drop
/// the matched entries from `handles`. Runs before the cgroup is
/// removed or stopped; failures are logged to stderr but do not
/// propagate — the cgroup removal is best-effort already, and the
/// payload-kill failure is never the primary error.
///
/// **Metric emission depends on the explicit `.kill()` call** —
/// if a future refactor replaces the `.kill()` below with plain
/// `drop(handle)`, the `PayloadHandle::drop` SIGKILLs the child
/// but skips the evaluate-and-emit pipeline that records metrics
/// to the SHM ring. Test helpers that drain payload handles
/// likewise route through `drain_all_payload_handles` for the
/// same reason. Preserve `.kill()` on every path that claims to
/// drain handles for metric capture.
fn drain_payload_handles_for_cgroup(handles: &mut Vec<PayloadEntry>, cgroup: &str) {
    let mut i = 0;
    while i < handles.len() {
        if handles[i].cgroup.as_str() == cgroup {
            let entry = handles.swap_remove(i);
            if let Err(e) = entry.handle.kill() {
                eprintln!("ktstr: kill payload in cgroup '{cgroup}': {e:#}");
            }
        } else {
            i += 1;
        }
    }
}

/// Kill every payload handle regardless of cgroup and clear the
/// vector. Called at step-sequence teardown so every handle gets a
/// terminal `.kill()` (and therefore a sidecar metric emission) even
/// when no explicit `RemoveCgroup`/`StopCgroup` op targeted it.
fn drain_all_payload_handles(handles: &mut Vec<PayloadEntry>) {
    for entry in handles.drain(..) {
        if let Err(e) = entry.handle.kill() {
            eprintln!(
                "ktstr: teardown kill payload in cgroup {}: {e:#}",
                render_cgroup_key(&entry.cgroup),
            );
        }
    }
}

/// Render a cgroup key for inclusion in user-facing error text.
/// An empty string is replaced with `(no cgroup)` so
/// `Op::RunPayload { cgroup: None }` failures don't produce messages
/// like `cgroup ''` that look like a corrupt log line. Non-empty
/// keys are quoted so they read clearly next to surrounding prose.
fn render_cgroup_key(cgroup: &str) -> String {
    if cgroup.is_empty() {
        "(no cgroup)".to_string()
    } else {
        format!("'{cgroup}'")
    }
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::*;
    use crate::vmm::shm_ring::parse_shm_params_from_str;

    // -- Traverse combinator (test-only) --

    /// Layout strategy for Traverse phases.
    #[derive(Debug)]
    enum Layout {
        Disjoint,
        /// Overlapping cpusets. (min_frac, max_frac) — PRNG picks a value in range.
        Overlap(f64, f64),
    }

    /// Generates a random walk of cgroup topology changes across phases.
    ///
    /// Each phase picks a random (cgroup_count, layout) pair, generates SetCpuset
    /// ops, spawns workers in new cgroups, and holds for phase_duration.
    ///
    /// `persistent_cgroups` cgroups are created in phase 0 and never removed.
    /// Only cgroups at index >= `persistent_cgroups` are added/removed by the
    /// random walk. The `cgroup_count` range applies to the total cgroup count
    /// (persistent + ephemeral).
    ///
    /// `cgroup_workloads` controls the workload for each cgroup index. If the
    /// vec has fewer entries than the cgroup index, the last entry repeats.
    #[derive(Debug)]
    struct Traverse {
        seed: Option<u64>,
        cgroup_count: RangeInclusive<usize>,
        layouts: Vec<Layout>,
        phases: usize,
        phase_duration: Duration,
        settle: Duration,
        /// Cgroups [0..persistent_cgroups) are created once and never removed.
        persistent_cgroups: usize,
        /// Work definition per cgroup index. Last entry repeats for higher indices.
        cgroup_workloads: Vec<Work>,
    }

    impl Traverse {
        /// Generate a `Vec<Step>` from the Traverse configuration.
        fn generate(&self, ctx: &Ctx) -> Vec<Step> {
            use rand::RngExt;

            let seed = self.seed.unwrap_or_else(|| std::process::id() as u64);
            let mut rng = seeded_rng(seed);

            let usable_len = ctx.topo.usable_cpus().len();
            let max_cgroups = (*self.cgroup_count.end()).min(usable_len / 2).max(1);
            let min_cgroups = (*self.cgroup_count.start()).max(1).min(max_cgroups);

            let mut steps = Vec::with_capacity(self.phases + 1);
            let mut live_cgroups: Vec<Cow<'static, str>> = Vec::new();

            let names: Vec<Cow<'static, str>> = (0..max_cgroups)
                .map(|i| Cow::Owned(format!("cg_{i}")))
                .collect();

            for phase in 0..self.phases {
                let range = max_cgroups - min_cgroups + 1;
                let target_count = min_cgroups + rng.random_range(0..range);
                let layout_idx = rng.random_range(0..self.layouts.len());
                let layout = &self.layouts[layout_idx];

                let mut ops = Vec::new();

                // Add cgroups if needed.
                while live_cgroups.len() < target_count {
                    let idx = live_cgroups.len();
                    let name = names[idx].clone();
                    let w = self
                        .cgroup_workloads
                        .get(idx)
                        .or(self.cgroup_workloads.last())
                        .cloned()
                        .unwrap_or_default();
                    ops.push(Op::AddCgroup { name: name.clone() });
                    ops.push(Op::Spawn {
                        cgroup: name.clone(),
                        work: w,
                    });
                    live_cgroups.push(name);
                }

                // Remove cgroups if needed (never remove persistent cgroups).
                while live_cgroups.len() > target_count
                    && live_cgroups.len() > self.persistent_cgroups
                {
                    if let Some(name) = live_cgroups.pop() {
                        ops.push(Op::StopCgroup {
                            cgroup: name.clone(),
                        });
                        ops.push(Op::RemoveCgroup { cgroup: name });
                    }
                }

                // Apply cpuset layout.
                for (i, name) in live_cgroups.iter().enumerate() {
                    let spec = match layout {
                        Layout::Disjoint => CpusetSpec::Disjoint {
                            index: i,
                            of: live_cgroups.len(),
                        },
                        Layout::Overlap(min_frac, max_frac) => {
                            let frac = min_frac
                                + rng.random_range(0..100) as f64 / 100.0 * (max_frac - min_frac);
                            CpusetSpec::Overlap {
                                index: i,
                                of: live_cgroups.len(),
                                frac,
                            }
                        }
                    };
                    ops.push(Op::SetCpuset {
                        cgroup: name.clone(),
                        cpus: spec,
                    });
                }

                let hold = if phase == 0 {
                    // First phase includes settle time.
                    HoldSpec::Fixed(self.settle + self.phase_duration)
                } else {
                    HoldSpec::Fixed(self.phase_duration)
                };

                steps.push(Step {
                    setup: vec![].into(),
                    ops,
                    hold,
                });
            }

            steps
        }
    }

    /// Seeded PRNG for deterministic topology generation.
    fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    // -- Op discriminant tests --

    #[test]
    fn op_discriminant_unique() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static TRUE_BIN: Payload = Payload {
            name: "true_bin",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let ops: Vec<Op> = vec![
            Op::AddCgroup { name: "a".into() },
            Op::RemoveCgroup { cgroup: "a".into() },
            Op::SetCpuset {
                cgroup: "a".into(),
                cpus: CpusetSpec::exact([]),
            },
            Op::ClearCpuset { cgroup: "a".into() },
            Op::SwapCpusets {
                a: "a".into(),
                b: "b".into(),
            },
            Op::Spawn {
                cgroup: "a".into(),
                work: Default::default(),
            },
            Op::StopCgroup { cgroup: "a".into() },
            Op::SetAffinity {
                cgroup: "a".into(),
                affinity: Default::default(),
            },
            Op::SpawnHost {
                work: Default::default(),
            },
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into(),
            },
            Op::RunPayload {
                payload: &TRUE_BIN,
                args: vec![],
                cgroup: None,
            },
            Op::WaitPayload {
                name: "p".into(),
                cgroup: None,
            },
            Op::KillPayload {
                name: "p".into(),
                cgroup: None,
            },
        ];
        let mut seen = std::collections::BTreeSet::new();
        for op in &ops {
            assert!(seen.insert(op.discriminant()), "duplicate discriminant");
        }
    }

    #[test]
    fn op_discriminant_values() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static TRUE_BIN: Payload = Payload {
            name: "true_bin",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        assert_eq!(Op::AddCgroup { name: "a".into() }.discriminant(), 0);
        assert_eq!(Op::RemoveCgroup { cgroup: "a".into() }.discriminant(), 1);
        assert_eq!(
            Op::SpawnHost {
                work: Default::default()
            }
            .discriminant(),
            8
        );
        assert_eq!(
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into()
            }
            .discriminant(),
            9
        );
        assert_eq!(
            Op::RunPayload {
                payload: &TRUE_BIN,
                args: vec![],
                cgroup: None,
            }
            .discriminant(),
            10,
        );
        assert_eq!(
            Op::WaitPayload {
                name: "p".into(),
                cgroup: None,
            }
            .discriminant(),
            11,
        );
        assert_eq!(
            Op::KillPayload {
                name: "p".into(),
                cgroup: None,
            }
            .discriminant(),
            12,
        );
    }

    // -- seeded_rng tests --

    #[test]
    fn seeded_rng_deterministic() {
        use rand::RngExt;
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);
        for _ in 0..100 {
            assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());
        }
    }

    #[test]
    fn seeded_rng_different_seeds_differ() {
        use rand::RngExt;
        let mut rng1 = seeded_rng(1);
        let mut rng2 = seeded_rng(2);
        let same = (0..10).all(|_| rng1.random::<u64>() == rng2.random::<u64>());
        assert!(!same);
    }

    // -- HoldSpec validate --

    #[test]
    fn holdspec_validate_accepts_valid() {
        HoldSpec::Frac(0.5).validate().unwrap();
        HoldSpec::Frac(1.0).validate().unwrap();
        HoldSpec::Fixed(Duration::from_millis(1))
            .validate()
            .unwrap();
        HoldSpec::Loop {
            interval: Duration::from_millis(100),
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn holdspec_validate_rejects_fixed_zero() {
        let err = HoldSpec::Fixed(Duration::ZERO).validate().unwrap_err();
        assert!(
            err.contains("Fixed") && err.contains("vacuous"),
            "error must name the variant and reason: {err}"
        );
    }

    #[test]
    fn holdspec_validate_rejects_frac_zero() {
        let err = HoldSpec::Frac(0.0).validate().unwrap_err();
        assert!(err.contains("Frac") && err.contains("> 0"), "got: {err}");
    }

    #[test]
    fn holdspec_validate_rejects_frac_negative() {
        let err = HoldSpec::Frac(-0.5).validate().unwrap_err();
        assert!(err.contains("Frac") && err.contains("> 0"), "got: {err}");
    }

    #[test]
    fn holdspec_validate_rejects_frac_nan() {
        let err = HoldSpec::Frac(f64::NAN).validate().unwrap_err();
        assert!(
            err.contains("not finite") || err.contains("NaN"),
            "got: {err}"
        );
    }

    #[test]
    fn holdspec_validate_rejects_frac_inf() {
        let err = HoldSpec::Frac(f64::INFINITY).validate().unwrap_err();
        assert!(
            err.contains("not finite") || err.contains("Inf"),
            "got: {err}"
        );
    }

    #[test]
    fn holdspec_validate_rejects_loop_zero_interval() {
        let err = HoldSpec::Loop {
            interval: Duration::ZERO,
        }
        .validate()
        .unwrap_err();
        assert!(err.contains("Loop") && err.contains("busy"), "got: {err}");
    }

    // -- HoldSpec variants --

    #[test]
    fn holdspec_frac() {
        let step = Step::new(vec![], HoldSpec::Frac(0.5));
        match step.hold {
            HoldSpec::Frac(f) => assert!((f - 0.5).abs() < f64::EPSILON),
            _ => panic!("expected Frac"),
        }
    }

    #[test]
    fn holdspec_fixed() {
        let step = Step::new(vec![], HoldSpec::Fixed(Duration::from_secs(3)));
        match step.hold {
            HoldSpec::Fixed(d) => assert_eq!(d, Duration::from_secs(3)),
            _ => panic!("expected Fixed"),
        }
    }

    #[test]
    fn holdspec_loop() {
        let step = Step::new(
            vec![],
            HoldSpec::Loop {
                interval: Duration::from_millis(100),
            },
        );
        match step.hold {
            HoldSpec::Loop { interval } => assert_eq!(interval, Duration::from_millis(100)),
            _ => panic!("expected Loop"),
        }
    }

    // -- CpusetSpec::Exact --

    #[test]
    fn cpusetspec_exact_is_passthrough() {
        let cpus: BTreeSet<usize> = [0, 2, 4].iter().copied().collect();
        let spec = CpusetSpec::Exact(cpus.clone());
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(1, 1, 4, 1),
        );
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &cgroups,
            topo: &topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: 0,
            settle: Duration::from_millis(1000),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let resolved = spec.resolve(&ctx);
        assert_eq!(resolved, cpus);
    }

    // -- Defense-in-depth: resolve must not panic on spec shapes that
    // -- validate rejects. Each test exercises a concrete panic the
    // -- resolver's hardening guards against.

    #[test]
    fn resolve_disjoint_of_zero_returns_empty_instead_of_panicking() {
        // `usable.len() / of` with of=0 would panic without hardening.
        // Current behavior: returns an empty BTreeSet with a
        // tracing::warn.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_overlap_of_zero_returns_empty_instead_of_panicking() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 0,
            frac: 0.5,
        };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_range_inverted_fracs_returns_empty_instead_of_panicking() {
        // Without hardening, `usable[start.min(len)..end.min(len)]`
        // with start_frac > end_frac produced start > end after
        // clamping and panicked the slice operation. Current
        // behavior: the slice is clamped to length-zero instead.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: 0.2,
        };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_range_nan_fracs_clamps_to_zero_instead_of_panicking() {
        // NaN as usize saturates to 0 on stable Rust, but inverted
        // start/end after both saturate is still fine post-fix.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: f64::NAN,
            end_frac: f64::NAN,
        };
        assert!(spec.resolve(&ctx).is_empty());
    }

    #[test]
    fn resolve_overlap_nonfinite_frac_clamps_to_zero() {
        // NaN frac pre-fix flowed through `(chunk as f64 * frac) as
        // usize` and could produce an out-of-range overlap. Post-fix
        // clamps NaN to 0, yielding the same partition boundaries as
        // Disjoint.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::NAN,
        };
        // No panic; result must be non-empty because index/of are valid.
        let result = spec.resolve(&ctx);
        assert!(!result.is_empty());
    }

    // -- parse_shm_params_from_str (from shm_ring) --

    #[test]
    fn ops_parse_shm_params_valid() {
        let cmdline = "console=ttyS0 KTSTR_SHM_BASE=0xfc000000 KTSTR_SHM_SIZE=0x10000 quiet";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x10000);
    }

    #[test]
    fn ops_parse_shm_params_missing() {
        assert!(parse_shm_params_from_str("console=ttyS0 quiet").is_none());
    }

    // -- CpusetSpec resolution helpers --

    fn make_ctx(
        llcs: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(1, llcs, cores, threads),
        );
        (cgroups, topo)
    }

    fn ctx_from<'a>(
        cgroups: &'a crate::cgroup::CgroupManager,
        topo: &'a crate::topology::TestTopology,
    ) -> Ctx<'a> {
        Ctx {
            cgroups,
            topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: 0,
            settle: Duration::ZERO,
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    // -- CpusetSpec::Disjoint --

    #[test]
    fn cpusetspec_disjoint_two_partitions() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Disjoint { index: 0, of: 2 }.resolve(&ctx);
        let b = CpusetSpec::Disjoint { index: 1, of: 2 }.resolve(&ctx);
        // Partitions must be disjoint.
        assert!(a.is_disjoint(&b), "partitions overlap: {:?} vs {:?}", a, b);
        // Together they cover all usable CPUs.
        let usable = ctx.topo.usable_cpuset();
        let union: BTreeSet<usize> = a.union(&b).copied().collect();
        assert_eq!(union, usable);
    }

    #[test]
    fn cpusetspec_disjoint_remainder_to_last() {
        // 7 usable CPUs / 3 partitions = chunk=2, so partition 0=[0,1], 1=[2,3], 2=[4,5,6].
        // Last partition gets the remainder.
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let usable_len = ctx.topo.usable_cpus().len();
        let c = CpusetSpec::Disjoint { index: 2, of: 3 }.resolve(&ctx);
        let chunk = usable_len / 3;
        // Last partition should be >= chunk size (gets remainder).
        assert!(
            c.len() >= chunk,
            "last partition {}: expected >= {}",
            c.len(),
            chunk
        );
    }

    #[test]
    fn cpusetspec_disjoint_single_partition() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let all = CpusetSpec::Disjoint { index: 0, of: 1 }.resolve(&ctx);
        let usable = ctx.topo.usable_cpuset();
        assert_eq!(all, usable);
    }

    #[test]
    fn cpusetspec_disjoint_index_beyond_of_returns_empty() {
        // Defense-in-depth: `validate` rejects index >= of with a clear
        // error, but callers that skip validation (e.g. programmatic
        // spec construction) must not hit the div-by-zero or panic in
        // `resolve`. With index = 5 and of = 3 on 3 usable CPUs
        // (4 total, 1 reserved by `usable_cpus`), chunk = 1 and
        // start = 5 clamps past `usable.len()` to yield an empty set
        // — a safe fallback, not a panic.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Disjoint { index: 5, of: 3 }.resolve(&ctx);
        assert!(
            cpus.is_empty(),
            "Disjoint with index beyond `of` must return an empty \
             cpuset rather than panicking, got: {cpus:?}",
        );
    }

    // -- CpusetSpec::Range --

    #[test]
    fn cpusetspec_range_first_half() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        let expected_len = usable.len() / 2;
        assert_eq!(cpus.len(), expected_len);
        // Should contain the first usable CPUs.
        for &cpu in &cpus {
            assert!(usable.contains(&cpu));
        }
    }

    #[test]
    fn cpusetspec_range_second_half() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 1.0,
        }
        .resolve(&ctx);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn cpusetspec_range_full() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 1.0,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpuset();
        assert_eq!(cpus, usable);
    }

    #[test]
    fn cpusetspec_range_empty() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        assert!(cpus.is_empty());
    }

    #[test]
    fn cpusetspec_range_clamps_to_bounds() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        // end_frac > 1.0 should be clamped to usable.len().
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 2.0,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpuset();
        assert_eq!(cpus, usable);
    }

    // -- CpusetSpec::Overlap --

    #[test]
    fn cpusetspec_overlap_neighbors_share_cpus() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.5,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Overlap {
            index: 1,
            of: 2,
            frac: 0.5,
        }
        .resolve(&ctx);
        let shared: BTreeSet<usize> = a.intersection(&b).copied().collect();
        assert!(!shared.is_empty(), "overlap=0.5 should produce shared CPUs");
    }

    #[test]
    fn cpusetspec_overlap_zero_frac_is_disjoint() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.0,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Overlap {
            index: 1,
            of: 2,
            frac: 0.0,
        }
        .resolve(&ctx);
        assert!(a.is_disjoint(&b), "frac=0 should be disjoint");
    }

    #[test]
    fn cpusetspec_overlap_last_partition_covers_tail() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let last = CpusetSpec::Overlap {
            index: 2,
            of: 3,
            frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        // Last partition should include the last usable CPU.
        assert!(last.contains(usable.last().unwrap()));
    }

    #[test]
    fn cpusetspec_overlap_first_partition_starts_at_zero() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let first = CpusetSpec::Overlap {
            index: 0,
            of: 3,
            frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        assert!(first.contains(&usable[0]));
    }

    // -- CpusetSpec::Llc --

    #[test]
    fn cpusetspec_llc_index_zero() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Llc(0).resolve(&ctx);
        assert!(!cpus.is_empty());
        // All CPUs in the set should belong to LLC 0.
        let llc0 = ctx.topo.llc_aligned_cpuset(0);
        assert_eq!(cpus, llc0);
    }

    #[test]
    fn cpusetspec_llc_two_llcs_disjoint() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let llc0 = CpusetSpec::Llc(0).resolve(&ctx);
        let llc1 = CpusetSpec::Llc(1).resolve(&ctx);
        assert!(llc0.is_disjoint(&llc1), "LLCs should be disjoint");
    }

    // -- CpusetSpec::Numa --

    fn make_numa_ctx(
        numa_nodes: u32,
        llcs: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(numa_nodes, llcs, cores, threads),
        );
        (cgroups, topo)
    }

    #[test]
    fn cpusetspec_numa_node_zero() {
        // 2 NUMA nodes, 4 LLCs (2 per NUMA), 4 cores, 1 thread
        // LLCs 0,1 -> NUMA 0 (CPUs 0-7), LLCs 2,3 -> NUMA 1 (CPUs 8-15)
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(0).resolve(&ctx);
        let expected: BTreeSet<usize> = (0..8).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn cpusetspec_numa_node_one() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(1).resolve(&ctx);
        let expected: BTreeSet<usize> = (8..16).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn cpusetspec_numa_disjoint() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let node0 = CpusetSpec::Numa(0).resolve(&ctx);
        let node1 = CpusetSpec::Numa(1).resolve(&ctx);
        assert!(
            node0.is_disjoint(&node1),
            "NUMA nodes should be disjoint: {:?} vs {:?}",
            node0,
            node1
        );
        let union: BTreeSet<usize> = node0.union(&node1).copied().collect();
        assert_eq!(union, ctx.topo.all_cpuset());
    }

    #[test]
    fn cpusetspec_numa_single_node_returns_all() {
        let (cg, topo) = make_numa_ctx(1, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(0).resolve(&ctx);
        assert_eq!(cpus, ctx.topo.all_cpuset());
    }

    #[test]
    fn cpusetspec_numa_validate_out_of_range() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Numa(5);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn cpusetspec_numa_validate_valid() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        assert!(CpusetSpec::Numa(0).validate(&ctx).is_ok());
        assert!(CpusetSpec::Numa(1).validate(&ctx).is_ok());
    }

    #[test]
    fn cpusetspec_numa_convenience_constructor() {
        let spec = CpusetSpec::numa(0);
        assert!(matches!(spec, CpusetSpec::Numa(0)));
    }

    // -- Traverse::generate --

    #[test]
    fn traverse_generate_produces_steps() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(42),
            cgroup_count: 2..=4,
            layouts: vec![Layout::Disjoint],
            phases: 3,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 0,
            cgroup_workloads: vec![Work::default()],
        };
        let steps = t.generate(&ctx);
        assert_eq!(steps.len(), 3);
        for step in &steps {
            assert!(!step.ops.is_empty(), "each phase should have ops");
        }
    }

    #[test]
    fn traverse_generate_deterministic() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(99),
            cgroup_count: 2..=4,
            layouts: vec![Layout::Disjoint, Layout::Overlap(0.2, 0.5)],
            phases: 5,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 1,
            cgroup_workloads: vec![Work::default()],
        };
        let steps1 = t.generate(&ctx);
        let steps2 = t.generate(&ctx);
        assert_eq!(steps1.len(), steps2.len());
        for (s1, s2) in steps1.iter().zip(steps2.iter()) {
            assert_eq!(
                s1.ops.len(),
                s2.ops.len(),
                "deterministic seed should produce same ops"
            );
        }
    }

    #[test]
    fn traverse_generate_persistent_cgroups_preserved() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(42),
            cgroup_count: 1..=4,
            layouts: vec![Layout::Disjoint],
            phases: 5,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 2,
            cgroup_workloads: vec![Work::default()],
        };
        let steps = t.generate(&ctx);
        // Every phase should have at least persistent_cgroups worth of SetCpuset ops
        // (cg_0, cg_1 are never removed).
        for step in &steps {
            let remove_ops: Vec<&Op> = step.ops.iter()
                .filter(|op| matches!(op, Op::RemoveCgroup { cgroup } if cgroup == "cg_0" || cgroup == "cg_1"))
                .collect();
            assert!(
                remove_ops.is_empty(),
                "persistent cgroups should never be removed"
            );
        }
    }

    // -- CgroupDef builder --

    #[test]
    fn cgroup_def_builder_chain() {
        let d = CgroupDef::named("test")
            .with_cpuset(CpusetSpec::llc(0))
            .workers(8)
            .work_type(WorkType::bursty(50, 100))
            .sched_policy(crate::workload::SchedPolicy::Batch)
            .swappable(true);
        assert_eq!(d.name, "test");
        assert!(d.cpuset.is_some());
        assert_eq!(d.works.len(), 1);
        assert_eq!(d.works[0].num_workers, Some(8));
        assert!(d.swappable);
    }

    #[test]
    fn cgroup_def_default() {
        let d = CgroupDef::default();
        assert_eq!(d.name, "cg_0");
        assert!(d.cpuset.is_none());
        assert!(d.works.is_empty());
        assert!(!d.swappable);
    }

    #[test]
    fn cgroup_def_multi_work() {
        let d = CgroupDef::named("multi")
            .work(Work::default().workers(4).work_type(WorkType::CpuSpin))
            .work(Work::default().workers(2).work_type(WorkType::YieldHeavy));
        assert_eq!(d.works.len(), 2);
        assert_eq!(d.works[0].num_workers, Some(4));
        assert_eq!(d.works[1].num_workers, Some(2));
    }

    #[test]
    fn cgroup_def_old_api_then_work() {
        let d = CgroupDef::named("mixed")
            .workers(4)
            .work(Work::default().workers(2));
        assert_eq!(d.works.len(), 2);
        assert_eq!(d.works[0].num_workers, Some(4));
        assert_eq!(d.works[1].num_workers, Some(2));
    }

    #[test]
    fn cgroup_def_work_only_no_phantom() {
        let d = CgroupDef::named("explicit").work(Work::default().workers(3));
        assert_eq!(d.works.len(), 1);
        assert_eq!(d.works[0].num_workers, Some(3));
    }

    // -- Setup --

    #[test]
    fn setup_defs_resolves() {
        let defs = vec![CgroupDef::named("a"), CgroupDef::named("b")];
        let setup = Setup::Defs(defs);
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let resolved = setup.resolve(&ctx);
        assert_eq!(resolved.len(), 2);
        assert!(!setup.is_empty());
    }

    #[test]
    fn setup_defs_empty() {
        let setup = Setup::Defs(vec![]);
        assert!(setup.is_empty());
    }

    #[test]
    fn setup_factory_not_empty() {
        let setup = Setup::Factory(|_| vec![CgroupDef::named("generated")]);
        assert!(!setup.is_empty());
    }

    // -- Step::with_defs / with_ops --

    #[test]
    fn step_with_defs_empty() {
        let step = Step::with_defs(vec![], HoldSpec::Frac(0.5));
        assert!(step.setup.is_empty());
        assert!(step.ops.is_empty());
    }

    #[test]
    fn step_with_defs_populated() {
        let step = Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(Duration::from_secs(5)),
        );
        assert!(!step.setup.is_empty());
        assert!(step.ops.is_empty());
    }

    #[test]
    fn step_with_defs_then_ops() {
        let step = Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::FULL).set_ops(vec![
            Op::AddCgroup {
                name: "cg_1".into(),
            },
        ]);
        assert!(!step.setup.is_empty());
        assert_eq!(step.ops.len(), 1);
    }

    #[test]
    fn step_set_ops_replaces() {
        let step = Step::new(
            vec![Op::AddCgroup { name: "a".into() }],
            HoldSpec::Frac(0.5),
        )
        .set_ops(vec![
            Op::AddCgroup { name: "b".into() },
            Op::RemoveCgroup { cgroup: "c".into() },
        ]);
        assert_eq!(step.ops.len(), 2);
    }

    // -- CpusetSpec::validate --

    #[test]
    fn cpusetspec_validate_disjoint_of_zero() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_disjoint_index_ge_of() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 3, of: 3 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("index 3 >= partition count 3"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_of_zero() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 0,
            frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_index_ge_of() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 2,
            of: 2,
            frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("index 2 >= partition count 2"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_start_ge_end() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: 0.2,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("start_frac"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_nan() {
        // Regression: IEEE 754 comparisons with NaN always return false, so
        // `start_frac >= end_frac` failed to reject it. validate() now
        // rejects non-finite fracs explicitly.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: f64::NAN,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_infinity() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: f64::INFINITY,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_negative() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: -0.5,
            end_frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_above_one() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 1.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_nan_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::NAN,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_infinity_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::INFINITY,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_out_of_range_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 1.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_too_few_cpus_for_partitions() {
        // 1 LLC, 2 cores, 1 thread => 2 total cpus, 2 usable
        let (cg, topo) = make_ctx(1, 2, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 5 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not enough usable CPUs"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_exact_in_range_ok() {
        // 1 LLC * 4 cores * 1 thread = CPUs 0..=3 physically present.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::exact([0, 2]);
        assert!(spec.validate(&ctx).is_ok());
    }

    #[test]
    fn cpusetspec_validate_exact_empty_rejected() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Exact(BTreeSet::new());
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("Exact") && err.contains("empty"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_exact_out_of_range_rejected() {
        // Topology has CPUs 0..=3; 99 is not physically present.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::exact([99]);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(
            err.contains("99") && err.contains("physical CPU set"),
            "error must name the offending CPU and call it physical: {err}"
        );
    }

    /// Regression: the reserved last CPU (when `total_cpus > 2`,
    /// `usable_cpus` drops the last one to leave the root cgroup a
    /// home) is still PHYSICALLY present. A scheduler author pinning
    /// a cgroup to that CPU for testing is legitimate — validate
    /// must NOT reject on `usable_cpuset` membership. Accepting it
    /// here is the contract that lets isolated-CPU tests compile.
    #[test]
    fn cpusetspec_validate_exact_accepts_reserved_last_cpu() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let total = ctx.topo.all_cpus().len();
        assert!(total > 2, "test requires a topology that reserves a CPU");
        let reserved_cpu = total - 1;
        assert!(
            !ctx.topo.usable_cpuset().contains(&reserved_cpu),
            "precondition: reserved CPU {reserved_cpu} must sit outside usable_cpuset",
        );
        assert!(
            ctx.topo.all_cpuset().contains(&reserved_cpu),
            "precondition: reserved CPU {reserved_cpu} must be physically present",
        );
        let spec = CpusetSpec::exact([reserved_cpu]);
        assert!(
            spec.validate(&ctx).is_ok(),
            "validate must accept the reserved CPU — physical presence, not \
             usable-set membership, is the bar",
        );
    }

    /// Regression guard for the HoldSpec pre-loop validation:
    /// execute_steps_with must bail on a vacuous hold BEFORE running
    /// any op. Failure mode without the pre-loop check: ops mutate
    /// cgroup state, then `Duration::from_secs_f64` / `thread::sleep`
    /// hit the downstream panic, leaving orphan cgroups on disk.
    #[test]
    fn execute_steps_with_bails_on_invalid_hold_before_ops() {
        let parent =
            std::env::temp_dir().join(format!("ktstr-hold-validate-{}", std::process::id()));
        // Pre-clean in case a prior failing test left a directory.
        let _ = std::fs::remove_dir_all(&parent);
        std::fs::create_dir_all(&parent).unwrap();
        let cgroups = crate::cgroup::CgroupManager::new(parent.to_str().unwrap());
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(1, 1, 4, 1),
        );
        let ctx = ctx_from(&cgroups, &topo);
        let cg_name = "should_never_exist";
        let step = Step::new(
            vec![Op::add_cgroup(cg_name)],
            HoldSpec::Fixed(Duration::ZERO),
        );
        let err = execute_steps_with(&ctx, vec![step], None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("hold validation") && msg.contains("Fixed"),
            "error must cite hold validation + variant: {msg}"
        );
        assert!(
            !parent.join(cg_name).exists(),
            "AddCgroup op ran before hold validation — cgroup dir '{}' exists",
            parent.join(cg_name).display()
        );
        let _ = std::fs::remove_dir_all(&parent);
    }

    /// The SetAffinity dispatcher's `AffinityMode::Random` arm is
    /// guarded by `!from.is_empty() && *count > 0` (see the
    /// `AffinityMode::Random` arm with that same guard in
    /// `apply_ops`). This test mirrors that classification to lock
    /// the contract in place: future refactors that drop either
    /// side of the AND must update this test alongside the dispatch.
    /// The live dispatcher path is partially covered by the
    /// `apply_setup_*` tests via `MockCgroupOps`, but the SetAffinity
    /// arm specifically still requires a running workload handle to
    /// exercise end-to-end and is therefore only covered by its
    /// classification guard here.
    #[test]
    fn set_affinity_random_no_op_conditions() {
        fn should_apply(from: &BTreeSet<usize>, count: usize) -> bool {
            !from.is_empty() && count > 0
        }
        let pool: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        let empty: BTreeSet<usize> = BTreeSet::new();
        assert!(should_apply(&pool, 2));
        assert!(!should_apply(&pool, 0), "count=0 → no-op");
        assert!(!should_apply(&empty, 1), "empty pool → no-op");
        assert!(!should_apply(&empty, 0), "both zero → no-op");
    }

    #[test]
    fn cpusetspec_validate_llc_out_of_range() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Llc(5);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_valid_disjoint_ok() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 1, of: 2 };
        assert!(spec.validate(&ctx).is_ok());
    }

    // -- MemPolicy + cpuset validation tests --

    #[test]
    fn validate_mempolicy_default_always_ok() {
        // 2 NUMA nodes, 2 LLCs (1 per node), 4 cores, 1 thread = 8 CPUs
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        assert!(validate_mempolicy_cpuset(&MemPolicy::Default, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_local_always_ok() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        assert!(validate_mempolicy_cpuset(&MemPolicy::Local, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_bind_covered() {
        // 2 NUMA nodes, 2 LLCs, 4 cores each = 8 CPUs total
        // LLC 0 (CPUs 0-3) = NUMA 0, LLC 1 (CPUs 4-7) = NUMA 1
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..8).collect(); // covers both nodes
        let policy = MemPolicy::Bind([0, 1].into_iter().collect());
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_bind_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Bind([1].into_iter().collect()); // node 1 not in cpuset
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_err());
    }

    #[test]
    fn validate_mempolicy_preferred_covered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (4..8).collect(); // NUMA node 1
        let policy = MemPolicy::Preferred(1);
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_preferred_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Preferred(1);
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_err());
    }

    #[test]
    fn validate_mempolicy_interleave_partial_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Interleave([0, 1].into_iter().collect());
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_err());
    }

    #[test]
    fn cgroupdef_mem_policy_builder() {
        let def = CgroupDef::named("test").mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(def.works[0].mem_policy, MemPolicy::Bind(_)));
    }

    // ---------------------------------------------------------------
    // apply_setup tests via MockCgroupOps
    // ---------------------------------------------------------------
    //
    // MockCgroupOps is a recording implementor of crate::cgroup::CgroupOps
    // that stores every call it receives in an internal Vec and can be
    // primed to return an error from the next call. This lets
    // apply_setup tests assert on the sequence of cgroup operations
    // without touching /sys/fs/cgroup, so they run as regular userspace
    // unit tests.
    //
    // apply_setup still calls WorkloadHandle::spawn, which forks real
    // worker processes. That's intentional: fork does not require root,
    // and the cgroup.procs write (which would require root in the real
    // kernel) is abstracted behind the mock. The test subject is the
    // orchestration logic — "for each def, call create_cgroup, then
    // set_cpuset if spec.is_some(), then move_tasks after spawn".
    //
    // Parallel-nextest behavior: verified non-flaky over repeated
    // `cargo nextest run --lib -E 'test(apply_setup)' --test-threads 8`
    // invocations and back-to-back full-suite runs. Each `MockCgroupOps`
    // owns its own `Mutex<Vec<CgroupCall>>`, so cross-test recording
    // cannot contend. `apply_setup` does call `WorkloadHandle::start`
    // (see top of this file) — workers wake, run briefly, and are then
    // SIGKILL'd when the owning `WorkloadHandle` drops via
    // `cleanup_state(&mut state)` / `state.handles.clear()` at the tail
    // of each test. No test assertion depends on worker output, only
    // on mock-recorded cgroup calls, so worker timing is not
    // observable. Fd footprint is 4 pipes × `workers()` per test — 8
    // fds for the 2-worker tests, well inside any RLIMIT_NOFILE the
    // harness sets.

    use crate::cgroup::CgroupOps;
    use std::path::Path;
    use std::sync::Mutex;

    /// A call captured by MockCgroupOps during apply_setup execution.
    /// Equality-comparable so tests can assert on the exact sequence.
    /// `MoveTasks` stores the tid count rather than the full `tids` Vec
    /// because PIDs are unpredictable between runs.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum CgroupCall {
        Setup(bool),
        CreateCgroup(String),
        RemoveCgroup(String),
        SetCpuset(String, BTreeSet<usize>),
        ClearCpuset(String),
        MoveTask(String, libc::pid_t),
        MoveTasks(String, usize), // (cgroup name, number of tids)
        ClearSubtreeControl(String),
        DrainTasks(String),
        CleanupAll,
    }

    struct MockCgroupOps {
        parent: std::path::PathBuf,
        calls: Mutex<Vec<CgroupCall>>,
        // When Some, the Nth call (indexed from 0 at insertion time)
        // returns an error and decrements; otherwise all calls return Ok.
        fail_at: Mutex<Option<(usize, String)>>,
    }

    impl MockCgroupOps {
        fn new() -> Self {
            Self {
                parent: std::path::PathBuf::from("/mock/cgroup"),
                calls: Mutex::new(Vec::new()),
                fail_at: Mutex::new(None),
            }
        }

        /// Return an error from the Nth call (0-indexed from now) with
        /// the given message. Used by tests that check error
        /// propagation through apply_setup.
        fn fail_call_at(&self, index: usize, message: &str) {
            *self.fail_at.lock().unwrap() = Some((index, message.to_string()));
        }

        fn calls(&self) -> Vec<CgroupCall> {
            self.calls.lock().unwrap().clone()
        }

        /// Record a call and decide whether to return Ok or inject an
        /// error. Centralizes the fail_at logic so every trait method
        /// gets it for free.
        fn record(&self, call: CgroupCall) -> Result<()> {
            let mut calls = self.calls.lock().unwrap();
            let current_index = calls.len();
            calls.push(call);
            drop(calls);
            let mut fail = self.fail_at.lock().unwrap();
            if let Some((index, ref message)) = *fail
                && current_index == index
            {
                let err_msg = message.clone();
                *fail = None;
                return Err(anyhow::anyhow!(err_msg));
            }
            Ok(())
        }
    }

    impl CgroupOps for MockCgroupOps {
        fn parent_path(&self) -> &Path {
            &self.parent
        }
        fn setup(&self, enable_cpu_controller: bool) -> Result<()> {
            self.record(CgroupCall::Setup(enable_cpu_controller))
        }
        fn create_cgroup(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::CreateCgroup(name.to_string()))
        }
        fn remove_cgroup(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::RemoveCgroup(name.to_string()))
        }
        fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
            self.record(CgroupCall::SetCpuset(name.to_string(), cpus.clone()))
        }
        fn clear_cpuset(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::ClearCpuset(name.to_string()))
        }
        fn move_task(&self, name: &str, tid: libc::pid_t) -> Result<()> {
            self.record(CgroupCall::MoveTask(name.to_string(), tid))
        }
        fn move_tasks(&self, name: &str, tids: &[libc::pid_t]) -> Result<()> {
            self.record(CgroupCall::MoveTasks(name.to_string(), tids.len()))
        }
        fn clear_subtree_control(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::ClearSubtreeControl(name.to_string()))
        }
        fn drain_tasks(&self, name: &str) -> Result<()> {
            self.record(CgroupCall::DrainTasks(name.to_string()))
        }
        fn cleanup_all(&self) -> Result<()> {
            self.record(CgroupCall::CleanupAll)
        }
    }

    /// Build a Ctx backed by MockCgroupOps so apply_setup can be driven
    /// without cgroup filesystem access. Topology fixed at 1 NUMA /
    /// 1 LLC / 4 cores / 1 thread = 4 CPUs — enough range to cover
    /// per-cpu cpuset assertions without making the mock brittle.
    fn mock_ctx<'a>(mock: &'a MockCgroupOps, topo: &'a crate::topology::TestTopology) -> Ctx<'a> {
        Ctx {
            cgroups: mock,
            topo,
            duration: Duration::from_secs(1),
            workers_per_cgroup: 1,
            sched_pid: 0,
            settle: Duration::ZERO,
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    fn mock_topo() -> crate::topology::TestTopology {
        crate::topology::TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(
            1, 1, 4, 1,
        ))
    }

    /// Drop workload + payload handles inside state so apply_setup
    /// tests don't leak worker or payload processes. Synthetic
    /// `WorkloadHandle`s SIGKILL their workers on Drop, so a
    /// `handles.clear()` is enough; `PayloadHandle` likewise
    /// SIGKILLs its child on Drop (with an eprintln warning about
    /// metrics not being recorded — acceptable in the test path
    /// where metrics aren't what's under test). Calling
    /// `drain_all_payload_handles` routes through `.kill()` so the
    /// metric-emission branch runs and the test doesn't trigger
    /// the Drop-warning banner on stderr.
    fn cleanup_state(state: &mut StepState<'_>) {
        state.handles.clear();
        drain_all_payload_handles(&mut state.payload_handles);
    }

    /// Test helper: call `apply_setup` against a step-local-only
    /// [`ScenarioState`]. Constructs a throwaway backdrop state
    /// pointing at the same mock-cgroups handle `state` uses so
    /// tests that only exercise step-local semantics stay terse.
    fn apply_setup_test<'a>(
        ctx: &'a Ctx<'a>,
        state: &mut StepState<'a>,
        defs: &[CgroupDef],
    ) -> Result<()> {
        let mut backdrop = BackdropState::empty(ctx);
        let mut scenario = ScenarioState::new(state, &mut backdrop);
        apply_setup(ctx, &mut scenario, defs)
    }

    /// Test helper: call `apply_ops` against a step-local-only
    /// [`ScenarioState`]. Mirrors [`apply_setup_test`] for ops.
    fn apply_ops_test<'a>(ctx: &'a Ctx<'a>, state: &mut StepState<'a>, ops: &[Op]) -> Result<()> {
        let mut backdrop = BackdropState::empty(ctx);
        let mut scenario = ScenarioState::new(state, &mut backdrop);
        apply_ops(ctx, &mut scenario, ops)
    }

    #[test]
    fn apply_setup_empty_defs_is_noop() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_setup_test(&ctx, &mut state, &[]).unwrap();
        assert!(
            mock.calls().is_empty(),
            "apply_setup on zero defs must not call any cgroup op, got: {:?}",
            mock.calls()
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_creates_cgroup_per_def() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let defs = vec![
            CgroupDef::named("cg_a").workers(1),
            CgroupDef::named("cg_b").workers(1),
        ];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        let creates: Vec<&CgroupCall> = calls
            .iter()
            .filter(|c| matches!(c, CgroupCall::CreateCgroup(_)))
            .collect();
        assert_eq!(
            creates,
            vec![
                &CgroupCall::CreateCgroup("cg_a".to_string()),
                &CgroupCall::CreateCgroup("cg_b".to_string()),
            ],
            "one create_cgroup call per def, in order"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_sets_cpuset_when_spec_present() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::Exact(cpus.clone()))
                .workers(1),
        ];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::SetCpuset("cg_0".to_string(), cpus.clone())),
            "set_cpuset must be called with exactly the resolved cpu set, got: {calls:?}"
        );
        // state.cpusets should mirror the set so later SetAffinity /
        // MemPolicy checks see the resolved cpuset.
        assert_eq!(
            state.cpusets.get("cg_0"),
            Some(&cpus),
            "state.cpusets must record the resolved set"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_skips_cpuset_when_none() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // cpuset: None → inherit parent's set, apply_setup must not
        // emit a set_cpuset call.
        let defs = vec![CgroupDef::named("cg_inherit").workers(1)];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        let has_set_cpuset = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::SetCpuset(_, _)));
        assert!(
            !has_set_cpuset,
            "no set_cpuset should be emitted when CgroupDef.cpuset is None, got: {calls:?}"
        );
        assert!(
            state.cpusets.is_empty(),
            "state.cpusets should stay empty when no CpusetSpec was resolved"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_moves_spawned_tasks_into_cgroup() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // workers(2): after spawn, apply_setup must call move_tasks
        // with 2 pids.
        let defs = vec![CgroupDef::named("cg_move").workers(2)];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        assert!(
            calls.contains(&CgroupCall::MoveTasks("cg_move".to_string(), 2)),
            "move_tasks must be called with the 2 spawned worker pids, got: {calls:?}"
        );
        // Ordering invariant: move_tasks follows create_cgroup, and
        // set_cpuset (when present) follows create_cgroup but precedes
        // move_tasks. Here with no cpuset, just assert create precedes
        // move.
        let create_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::CreateCgroup(n) if n == "cg_move"))
            .expect("create_cgroup for cg_move");
        let move_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_move"))
            .expect("move_tasks for cg_move");
        assert!(
            create_idx < move_idx,
            "create_cgroup must precede move_tasks for the same cgroup: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_sets_cpuset_before_move_tasks() {
        // Ordering invariant: for a cgroup with both a cpuset spec and
        // workers, `set_cpuset` MUST precede `move_tasks` so the
        // kernel enforces the cpu mask on the first scheduling
        // decision after the task enters the cgroup. Moving first
        // would let tasks briefly run on cpus outside the intended
        // set.
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_ordered")
                .with_cpuset(CpusetSpec::Exact(cpus.clone()))
                .workers(2),
        ];
        apply_setup_test(&ctx, &mut state, &defs).unwrap();
        let calls = mock.calls();
        let set_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::SetCpuset(n, _) if n == "cg_ordered"))
            .expect("set_cpuset for cg_ordered");
        let move_idx = calls
            .iter()
            .position(|c| matches!(c, CgroupCall::MoveTasks(n, _) if n == "cg_ordered"))
            .expect("move_tasks for cg_ordered");
        assert!(
            set_idx < move_idx,
            "set_cpuset must precede move_tasks for the same cgroup: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_bails_on_invalid_cpuset_spec() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Llc(99) on a 1-LLC topology is out of range; CpusetSpec::validate
        // bails after create_cgroup runs but before set_cpuset / move_tasks
        // fire.
        let defs = vec![CgroupDef::named("cg_bad").with_cpuset(CpusetSpec::Llc(99))];
        let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CpusetSpec validation failed"),
            "expected validation error, got: {msg}"
        );
        // create_cgroup runs before cpuset validation — record that
        // here so future refactors notice if the order flips.
        let calls = mock.calls();
        assert_eq!(
            calls,
            vec![CgroupCall::CreateCgroup("cg_bad".to_string())],
            "current ordering: create_cgroup first, then cpuset validation"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_propagates_set_cpuset_error() {
        let mock = MockCgroupOps::new();
        // Inject failure at call index 1. Index 0 is the create_cgroup
        // emitted before the cpuset write; index 1 is the set_cpuset
        // itself.
        mock.fail_call_at(1, "set_cpuset kernel EBUSY");
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let cpus: BTreeSet<usize> = [0, 1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_setfail")
                .with_cpuset(CpusetSpec::Exact(cpus))
                .workers(1),
        ];
        let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("set_cpuset kernel EBUSY"),
            "set_cpuset error must propagate, got: {msg}"
        );
        // Check the failure halted apply_setup before reaching spawn:
        // no MoveTasks call should have been recorded.
        let calls = mock.calls();
        let has_move = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _)));
        assert!(
            !has_move,
            "no move_tasks call should follow a failed set_cpuset, got: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    #[test]
    fn apply_setup_validates_mempolicy_against_cpuset() {
        let mock = MockCgroupOps::new();
        // 2 NUMA / 2 LLCs (1 per node) / 4 cores / 1 thread = 8 CPUs
        let topo = crate::topology::TestTopology::from_vm_topology(
            &crate::vmm::topology::Topology::new(2, 2, 4, 1),
        );
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // cpuset = NUMA node 0 only (CPUs 0-3); mem_policy binds to
        // node 1 — must bail, no downstream spawn.
        let cpus: BTreeSet<usize> = (0..4).collect();
        let bind: BTreeSet<usize> = [1].into_iter().collect();
        let defs = vec![
            CgroupDef::named("cg_memfail")
                .with_cpuset(CpusetSpec::Exact(cpus))
                .mem_policy(MemPolicy::Bind(bind))
                .workers(1),
        ];
        let err = apply_setup_test(&ctx, &mut state, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cg_memfail"),
            "error must name the bad cgroup, got: {msg}"
        );
        // set_cpuset was called before the mempolicy check (order
        // documented by apply_setup). Assert move_tasks did not run —
        // that would mean the pre-validation guard failed.
        let calls = mock.calls();
        let has_move = calls
            .iter()
            .any(|c| matches!(c, CgroupCall::MoveTasks(_, _)));
        assert!(
            !has_move,
            "mempolicy validation must bail before spawn, got: {calls:?}"
        );
        cleanup_state(&mut state);
    }

    // -- CgroupDef::workload --

    /// Default CgroupDef has no payload attached — every test that
    /// doesn't opt in stays Payload-free so the synthetic-workload
    /// path is unaffected.
    #[test]
    fn cgroup_def_default_payload_is_none() {
        let def = CgroupDef::named("cg_0");
        assert!(def.payload.is_none());
    }

    /// The `.workload(&FIO)` builder stores the reference on the
    /// CgroupDef so apply_setup can spawn it. Because `Payload` is
    /// `Copy`, the builder preserves identity through pointer
    /// equality after conversion to `&'static` refs.
    #[test]
    fn cgroup_def_workload_stores_payload() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let def = CgroupDef::named("cg_0").workload(&FIO);
        let p = def.payload.expect("workload was attached");
        assert_eq!(p.name, "fio");
        assert!(!p.is_scheduler());
    }

    /// Scheduler-kind payloads are rejected at builder time — the
    /// `workload` slot is exclusively for userspace binaries that
    /// run *under* a scheduler, not for scheduler placement itself.
    #[test]
    #[should_panic(expected = "CgroupDef::workload called with a scheduler-kind Payload")]
    fn cgroup_def_workload_rejects_scheduler_kind_payload() {
        use crate::test_support::Payload;
        let _ = CgroupDef::named("cg_0").workload(&Payload::KERNEL_DEFAULT);
    }

    /// The drain helper kills + removes entries whose cgroup name
    /// matches the target. Non-matching entries stay in the vector
    /// so subsequent step teardown (via `collect_step`) or scenario
    /// end (via `collect_backdrop`) kills them in turn.
    #[test]
    fn drain_payload_handles_for_cgroup_removes_matching_only() {
        use crate::cgroup::CgroupManager;
        use crate::scenario::payload_run::PayloadRun;
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        use crate::topology::TestTopology;

        static TRUE_BIN: Payload = Payload {
            name: "true_bin",
            kind: PayloadKind::Binary("/bin/true"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };

        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();

        let h_a = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true for cg_a");
        let h_b = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true for cg_b");

        let mut handles = vec![
            PayloadEntry {
                cgroup: "cg_a".to_string(),
                source: PayloadSource::CgroupDefWorkload,
                handle: h_a,
            },
            PayloadEntry {
                cgroup: "cg_b".to_string(),
                source: PayloadSource::CgroupDefWorkload,
                handle: h_b,
            },
        ];
        drain_payload_handles_for_cgroup(&mut handles, "cg_a");

        assert_eq!(handles.len(), 1);
        assert_eq!(handles[0].cgroup, "cg_b");

        drain_all_payload_handles(&mut handles);
        assert!(handles.is_empty());
    }

    // -- Step::with_payload + Op::RunPayload/WaitPayload/KillPayload --

    /// Step::with_payload emits a step whose ops consist of a single
    /// Op::RunPayload carrying the supplied payload. Hold passes
    /// through unchanged.
    #[test]
    fn step_with_payload_emits_runpayload_op() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let step = Step::with_payload(&FIO, HoldSpec::Fixed(Duration::from_millis(50)));
        assert_eq!(step.ops.len(), 1);
        match &step.ops[0] {
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                assert_eq!(payload.name, "fio");
                assert!(args.is_empty());
                assert!(cgroup.is_none());
            }
            other => panic!("expected RunPayload, got {other:?}"),
        }
        assert!(matches!(step.hold, HoldSpec::Fixed(d) if d == Duration::from_millis(50)));
        assert!(matches!(&step.setup, Setup::Defs(d) if d.is_empty()));
    }

    /// Op convenience constructors — `run_payload`, `wait_payload`,
    /// `kill_payload`, `run_payload_in_cgroup` — build the expected
    /// enum shapes with the right field contents.
    #[test]
    fn op_payload_constructors_produce_expected_variants() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };

        let op = Op::run_payload(&FIO, vec!["--warmup".into()]);
        match op {
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                assert_eq!(payload.name, "fio");
                assert_eq!(args, vec!["--warmup".to_string()]);
                assert!(cgroup.is_none());
            }
            other => panic!("expected RunPayload, got {other:?}"),
        }

        let op = Op::run_payload_in_cgroup(&FIO, vec![], "cg_0");
        match op {
            Op::RunPayload {
                payload,
                args,
                cgroup,
            } => {
                assert_eq!(payload.name, "fio");
                assert!(args.is_empty());
                assert_eq!(cgroup.as_deref(), Some("cg_0"));
            }
            other => panic!("expected RunPayload, got {other:?}"),
        }

        let op = Op::wait_payload("fio");
        assert!(matches!(
            op,
            Op::WaitPayload { ref name, ref cgroup } if name.as_ref() == "fio" && cgroup.is_none(),
        ));

        let op = Op::kill_payload("fio");
        assert!(matches!(
            op,
            Op::KillPayload { ref name, ref cgroup } if name.as_ref() == "fio" && cgroup.is_none(),
        ));

        let op = Op::wait_payload_in_cgroup("fio", "cg_0");
        assert!(matches!(
            op,
            Op::WaitPayload { ref name, cgroup: Some(ref c) } if name.as_ref() == "fio" && c.as_ref() == "cg_0",
        ));

        let op = Op::kill_payload_in_cgroup("fio", "cg_0");
        assert!(matches!(
            op,
            Op::KillPayload { ref name, cgroup: Some(ref c) } if name.as_ref() == "fio" && c.as_ref() == "cg_0",
        ));
    }

    /// Op::RunPayload rejects scheduler-kind payloads at apply time
    /// with an actionable error message. The existing CgroupDef
    /// path panics at builder time; the Op path runs at scenario
    /// time and must bail instead of panicking so one bad step in
    /// a sequence doesn't crash the harness.
    #[test]
    fn apply_ops_runpayload_rejects_scheduler_kind() {
        use crate::test_support::Payload;
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let ops = vec![Op::RunPayload {
            payload: &Payload::KERNEL_DEFAULT,
            args: vec![],
            cgroup: None,
        }];
        let err = apply_ops_test(&ctx, &mut state, &ops).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scheduler-kind Payload") && msg.contains("kernel_default"),
            "error must name the scheduler-kind reason AND the payload name, got: {msg}"
        );
        assert!(
            state.payload_handles.is_empty(),
            "no handle should be stored when RunPayload rejects the kind"
        );
    }

    /// Op::WaitPayload with no matching handle surfaces a descriptive
    /// error rather than silently no-op'ing. Ditto KillPayload. A
    /// silent no-op would let test authors wait for ghosts and pass
    /// scenarios that never ran what they claim.
    #[test]
    fn apply_ops_wait_unknown_payload_bails() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::WaitPayload {
                name: "ghost".into(),
                cgroup: None,
            }],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no running payload named 'ghost'"),
            "error must name the missing payload, got: {msg}"
        );
    }

    #[test]
    fn apply_ops_kill_unknown_payload_bails() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::KillPayload {
                name: "ghost".into(),
                cgroup: None,
            }],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no running payload named 'ghost'"),
            "error must name the missing payload, got: {msg}"
        );
    }

    /// End-to-end on a real payload binary: Op::RunPayload spawns
    /// a long-running `/bin/sleep`, Op::KillPayload matches by
    /// payload.name and consumes the handle. The handle should
    /// disappear from state.payload_handles so later teardown
    /// drains don't double-consume.
    #[test]
    fn apply_ops_run_then_kill_consumes_handle() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            // Name distinct from binary so the payload_name lookup
            // path is exercised against a non-basename key.
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .expect("spawn /bin/sleep");
        assert_eq!(state.payload_handles.len(), 1, "one payload is live");
        assert_eq!(state.payload_handles[0].handle.payload_name(), "sleeper");

        apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")])
            .expect("kill the live payload");
        assert!(
            state.payload_handles.is_empty(),
            "handle must be consumed by KillPayload"
        );
    }

    /// Spawning a second payload with the same name while the first
    /// is still live is a caller bug — the `WaitPayload`/
    /// `KillPayload` lookup would hit the first match and leave the
    /// second leaked. Reject at RunPayload time.
    #[test]
    fn apply_ops_run_duplicate_payload_name_bails() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .expect("first spawn");

        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("payload 'sleeper' already running"),
            "error must flag the duplicate, got: {msg}"
        );
        // The dup error must identify the surface that spawned the
        // live handle so the user knows where to go to fix it. The
        // first spawn was via Op::RunPayload, not CgroupDef::workload.
        assert!(
            msg.contains("Op::RunPayload"),
            "dup error must name the originating surface, got: {msg}"
        );
        // The Op::RunPayload in this test ran without a
        // `cgroup = Some(..)`, so the rendered cgroup key must be
        // `(no cgroup)`, not an empty-quoted `''`.
        assert!(
            msg.contains("(no cgroup)"),
            "empty-cgroup key must render as '(no cgroup)', got: {msg}"
        );
        assert!(
            !msg.contains("cgroup ''"),
            "empty-cgroup key must not render as quoted empty, got: {msg}"
        );
        assert_eq!(
            state.payload_handles.len(),
            1,
            "second spawn must not add a handle on failure"
        );

        // Clean up the live handle so the test process doesn't leak
        // a /bin/sleep.
        apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")]).expect("teardown kill");
    }

    /// When the first spawn came from `CgroupDef::workload` in
    /// `cg_def` and a subsequent `Op::run_payload_in_cgroup` targets
    /// the same `cg_def` with the same payload name, the composite-
    /// key dup check fires and names `CgroupDef::workload` as the
    /// originating surface. A cross-cgroup duplicate (same name,
    /// different cgroup) is legitimate and tested separately.
    #[test]
    fn apply_ops_run_rejects_payload_already_owned_by_cgroup_def() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        // Simulate the def-owned handle directly — apply_setup pushes
        // entries with PayloadSource::CgroupDefWorkload, so construct
        // the equivalent here without invoking the real spawn path
        // (apply_setup needs workers(N) and cgroupfs ops which MockCgroupOps
        // does not implement for this test shape).
        let h = crate::scenario::payload_run::PayloadRun::new(&ctx, &SLEEP)
            .args(["3600".to_string()])
            .spawn()
            .expect("manual def-source spawn");
        state.payload_handles.push(PayloadEntry {
            cgroup: "def_cg".to_string(),
            source: PayloadSource::CgroupDefWorkload,
            handle: h,
        });

        // Targeting the SAME cgroup as the pre-existing entry: dup.
        let err = apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload_in_cgroup(
                &SLEEP,
                vec!["1".into()],
                "def_cg",
            )],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("CgroupDef::workload"),
            "dup error must name the def-source surface, got: {msg}"
        );
        assert!(
            msg.contains("'def_cg'"),
            "dup error must name the cgroup the live handle is in, got: {msg}"
        );
        // Only the original handle remains — op branch bailed pre-spawn.
        assert_eq!(state.payload_handles.len(), 1);

        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::kill_payload_in_cgroup("sleeper", "def_cg")],
        )
        .expect("teardown kill");
    }

    /// [`render_cgroup_key`] renders an empty string as
    /// `(no cgroup)` and a populated name as single-quoted prose.
    /// Pins the formatting so every error path that echoes the
    /// cgroup key through this helper stays consistent.
    #[test]
    fn render_cgroup_key_handles_empty_and_populated() {
        assert_eq!(render_cgroup_key(""), "(no cgroup)");
        assert_eq!(render_cgroup_key("cg_a"), "'cg_a'");
    }

    // -- payload_handles drain on error paths in execute_steps_with --

    /// An Err return from `execute_steps_with` (here: a vacuous
    /// `HoldSpec::Fixed(ZERO)` caught by up-front validation)
    /// leaves no live payload_handles because no setup/ops ran.
    /// Pins the invariant that the pre-ops validation path does
    /// not spawn anything that could then leak.
    #[test]
    fn execute_steps_with_early_validation_err_has_nothing_to_drain() {
        use crate::cgroup::CgroupManager;
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = mock_topo();
        let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();
        let step = Step::new(vec![], HoldSpec::Fixed(Duration::ZERO));
        let err = execute_steps_with(&ctx, vec![step], None).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("hold validation") && msg.contains("vacuous"),
            "expected pre-ops validation err, got: {msg}"
        );
    }

    /// When a live payload has been spawned and a later op returns
    /// Err, the drain-on-err path consumes the payload handles via
    /// `.kill()` (which emits metrics) rather than leaking them to
    /// `PayloadHandle::drop` (which SIGKILLs without recording).
    ///
    /// This test exercises the drain path directly by spawning a
    /// /bin/sleep, then calling `apply_ops` with an op that forces
    /// an error (unknown-name `WaitPayload`). After the Err, the
    /// state's payload_handles must still be consulted by the
    /// drain — verified by checking the live count before +
    /// explicit teardown after.
    #[test]
    fn apply_ops_error_does_not_lose_live_payload_handles() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper_drain",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::run_payload(&SLEEP, vec!["3600".into()])],
        )
        .expect("spawn");
        assert_eq!(state.payload_handles.len(), 1);
        // Trigger an Err via WaitPayload on an unknown name. Before
        // the fix, execute_steps_with would propagate the Err via
        // `?` and leave the SLEEP handle to be SIGKILLed by Drop
        // (losing the metric emission).
        let err =
            apply_ops_test(&ctx, &mut state, &[Op::wait_payload("never_spawned")]).unwrap_err();
        assert!(
            format!("{err:#}").contains("no running payload named 'never_spawned'"),
            "expected wait-unknown-name err",
        );
        // The live handle is still in state — apply_ops itself does
        // not drain on Err (that's execute_steps_with's
        // responsibility). Manually drain via the helper to
        // terminate the child cleanly.
        drain_all_payload_handles(&mut state.payload_handles);
        assert!(state.payload_handles.is_empty());
    }

    // ---------------------------------------------------------------
    // Step/Backdrop ruling invariants
    // ---------------------------------------------------------------

    /// A step-local `Op::RemoveCgroup` that targets a Backdrop-owned
    /// cgroup must bail before any cgroupfs write. Ops running inside
    /// the Backdrop's own setup pass (i.e. `target_backdrop == true`)
    /// stay exempt.
    #[test]
    fn remove_cgroup_rejects_backdrop_target() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        // Populate a backdrop cgroup and leave the step state empty.
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg")
            .expect("add backdrop cgroup");

        // Step-local RemoveCgroup must reject.
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            let err = apply_ops(&ctx, &mut scenario, &[Op::remove_cgroup("bd_cg")]).unwrap_err();
            let msg = format!("{err:#}");
            assert!(
                msg.contains("Backdrop-owned") && msg.contains("bd_cg"),
                "error must name the backdrop cgroup and explain why, got: {msg}"
            );
        }
        // The backdrop cgroup is still tracked.
        assert_eq!(backdrop_state.cgroups.names(), &["bd_cg".to_string()]);
        // No remove_cgroup call was issued to the mock — the bail
        // happened before any cgroupfs write.
        let calls = mock.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, CgroupCall::RemoveCgroup(_))),
            "pre-bail path must not invoke remove_cgroup, got: {calls:?}"
        );

        // Backdrop-pass RemoveCgroup (target_backdrop = true) is
        // allowed and routes through to the mock.
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            scenario
                .with_target_backdrop(|s| apply_ops(&ctx, s, &[Op::remove_cgroup("bd_cg")]))
                .expect("backdrop-pass remove is permitted");
        }
        let calls = mock.calls();
        assert!(
            calls
                .iter()
                .any(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "bd_cg")),
            "backdrop-pass remove must reach the cgroup ops, got: {calls:?}"
        );

        cleanup_state(&mut step_state);
    }

    /// `Op::MoveAllTasks` from a step-local cgroup to a Backdrop
    /// cgroup must transfer the handle from step-local slot to
    /// backdrop slot so the worker survives the step boundary. A
    /// step-to-step move keeps ownership step-local. A backdrop-to-
    /// step move keeps the handle in the backdrop slot (persistent
    /// does not degrade).
    #[test]
    fn move_all_tasks_transfers_handle_ownership_step_to_backdrop() {
        use crate::workload::{Work, WorkloadConfig, WorkloadHandle};

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        // Backdrop owns "bd_cg"; the step owns "step_cg" and a
        // handle keyed under it.
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("bd_cg")
            .unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("step_cg").unwrap();
        let w = Work::default();
        let wl = WorkloadConfig {
            num_workers: 1,
            affinity: crate::workload::AffinityMode::None,
            work_type: w.work_type,
            sched_policy: w.sched_policy,
            mem_policy: w.mem_policy,
            mpol_flags: w.mpol_flags,
        };
        let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
        step_state.handles.push(("step_cg".to_string(), h));
        assert_eq!(step_state.handles.len(), 1);
        assert_eq!(backdrop_state.handles.len(), 0);

        // Move tasks from step_cg to bd_cg: ownership transfers.
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(
                &ctx,
                &mut scenario,
                &[Op::move_all_tasks("step_cg", "bd_cg")],
            )
            .expect("move into backdrop");
        }
        assert_eq!(
            step_state.handles.len(),
            0,
            "step-local handle must leave the step slot after transfer",
        );
        assert_eq!(
            backdrop_state.handles.len(),
            1,
            "backdrop slot must receive the transferred handle",
        );
        assert_eq!(
            backdrop_state.handles[0].0, "bd_cg",
            "transferred handle must be re-keyed to `to`",
        );

        // Clear the handles before the test drops (handles SIGKILL on
        // drop — avoid leaking the worker process).
        backdrop_state.handles.clear();
        step_state.handles.clear();
    }

    /// Step→step move does NOT cross state slots (companion to the
    /// step→backdrop transfer test above).
    #[test]
    fn move_all_tasks_step_to_step_keeps_step_ownership() {
        use crate::workload::{Work, WorkloadConfig, WorkloadHandle};

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();
        let w = Work::default();
        let wl = WorkloadConfig {
            num_workers: 1,
            affinity: crate::workload::AffinityMode::None,
            work_type: w.work_type,
            sched_policy: w.sched_policy,
            mem_policy: w.mem_policy,
            mpol_flags: w.mpol_flags,
        };
        let h = WorkloadHandle::spawn(&wl).expect("spawn");
        step_state.handles.push(("src".to_string(), h));
        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("src", "dst")])
                .expect("step-to-step move");
        }
        assert_eq!(step_state.handles.len(), 1);
        assert_eq!(step_state.handles[0].0, "dst");
        assert_eq!(backdrop_state.handles.len(), 0);
        step_state.handles.clear();
    }

    /// A step-local `Op::MoveAllTasks` that
    /// pulls from a Backdrop-owned cgroup into a step-local cgroup
    /// must bail before touching cgroupfs. The persistent worker
    /// would otherwise be stranded in a cgroup that gets rmdir'd at
    /// the step boundary. Backdrop-setup ops (`target_backdrop`)
    /// stay exempt.
    #[test]
    fn move_all_tasks_backdrop_to_step_rejected() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state.cgroups.add_cgroup_no_cpuset("bd").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("step").unwrap();

        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("bd", "step")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Backdrop-owned 'bd'") && msg.contains("step-local 'step'"),
            "error must name both cgroups and the direction, got: {msg}"
        );
        // The mock must not have seen a cgroup.procs write — the
        // guard bails before any kernel-side work.
        let calls = mock.calls();
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, CgroupCall::MoveTasks(_, _))),
            "pre-bail path must not invoke move_tasks, got: {calls:?}"
        );
    }

    /// `run_scenario` rejects a scheduler-kind payload in
    /// `Backdrop::payloads` before running any setup.
    #[test]
    fn run_scenario_rejects_scheduler_kind_backdrop_payload() {
        use crate::cgroup::CgroupManager;
        use crate::test_support::Payload;
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = mock_topo();
        let ctx = crate::scenario::Ctx::builder(&cgroups, &topo).build();
        let backdrop = super::super::backdrop::Backdrop::new().with_payload(&Payload::KERNEL_DEFAULT);
        let err = execute_scenario_with(
            &ctx,
            backdrop,
            vec![Step::new(vec![], HoldSpec::Fixed(Duration::from_millis(1)))],
            None,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("scheduler-kind") && msg.contains("Backdrop"),
            "error must name the kind mismatch and the Backdrop surface, got: {msg}"
        );
    }

    /// `apply_setup` rejects a step-local CgroupDef whose name
    /// collides with a Backdrop-tracked cgroup.
    #[test]
    fn apply_setup_rejects_name_collision_with_backdrop() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("shared")
            .unwrap();
        let defs = vec![CgroupDef::named("shared").workers(1)];
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        let err = apply_setup(&ctx, &mut scenario, &defs).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already tracked") && msg.contains("shared"),
            "error must cite the collision and the offending name, got: {msg}"
        );
        cleanup_state(&mut step_state);
    }

    // ---------------------------------------------------------------
    // composite-key (name, cgroup) dedup for Op::RunPayload
    // ---------------------------------------------------------------

    /// Push a synthetic live PayloadEntry into `state`'s step slot
    /// so tests can exercise dedup / lookup paths without paying
    /// the cost of a real cgroupfs-backed spawn (which fails inside
    /// the MockCgroupOps test harness because `/mock/cgroup/...`
    /// doesn't exist on disk).
    fn push_fake_payload_entry<'a>(
        ctx: &'a Ctx<'a>,
        state: &mut StepState<'a>,
        payload: &'static crate::test_support::Payload,
        cgroup: &str,
        source: PayloadSource,
    ) {
        let h = crate::scenario::payload_run::PayloadRun::new(ctx, payload)
            .args(["3600".to_string()])
            .spawn()
            .expect("manual spawn (no cgroup placement)");
        state.payload_handles.push(PayloadEntry {
            cgroup: cgroup.to_string(),
            source,
            handle: h,
        });
    }

    /// Same payload live in `cg_a` AND `cg_b`; a third
    /// `Op::RunPayload` targeting a brand-new `cg_c` must NOT trip
    /// the composite-key dedup because the (name, cgroup) pair is
    /// fresh. Simulated via direct state injection so the test
    /// doesn't depend on cgroupfs.
    #[test]
    fn apply_ops_run_duplicate_name_different_cgroups_allowed() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let scenario = ScenarioState::new(&mut state, &mut backdrop);
        // The `find_live_payload_with_cgroup` lookup for ("sleeper", "cg_c")
        // returns None because no live entry matches that pair — so
        // the dup check passes and run_scenario would let the spawn
        // proceed. We check the lookup directly (spawning against
        // MockCgroupOps would fail on the pre_exec cgroup write).
        assert!(
            scenario
                .find_live_payload_with_cgroup("sleeper", "cg_c")
                .is_none(),
            "fresh (name, cgroup) pair must not collide with live entries in other cgroups",
        );
        // And the existing same-cgroup entry still collides.
        assert!(
            scenario
                .find_live_payload_with_cgroup("sleeper", "cg_a")
                .is_some(),
            "same (name, cgroup) still matches — only the pair matters",
        );

        cleanup_state(&mut state);
    }

    /// `take_payload_by_name` in composite mode matches only the
    /// exact `(name, cgroup)` pair and leaves sibling copies alone.
    #[test]
    fn take_payload_by_composite_key_matches_exact_cgroup() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
        let taken = scenario
            .take_payload_by_name("sleeper", Some("cg_a"))
            .expect("composite lookup does not bail on ambiguity")
            .expect("one entry matches");
        assert_eq!(taken.cgroup, "cg_a");
        // The cg_b entry survives.
        assert_eq!(state.payload_handles.len(), 1);
        assert_eq!(state.payload_handles[0].cgroup, "cg_b");
        // Drain to avoid leaking the live child.
        drain_all_payload_handles(&mut state.payload_handles);
        let _ = taken.handle.kill();
    }

    /// Bare `take_payload_by_name(name, None)` returns
    /// `Err(ambiguous_cgroups)` when two or more copies are live,
    /// surfacing both cgroup keys so the caller can disambiguate.
    #[test]
    fn take_payload_by_bare_name_reports_ambiguous_cgroups() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
        let err = match scenario.take_payload_by_name("sleeper", None) {
            Err(cgroups) => cgroups,
            Ok(_) => panic!("bare lookup over multi-copy must surface ambiguity"),
        };
        assert_eq!(err.len(), 2);
        assert!(err.contains(&"cg_a".to_string()) && err.contains(&"cg_b".to_string()));
        // No handle consumed — both still live.
        assert_eq!(state.payload_handles.len(), 2);
        drain_all_payload_handles(&mut state.payload_handles);
    }

    /// Bare `take_payload_by_name(name, None)` succeeds when
    /// exactly one copy is live, so `Op::wait_payload(name)` and
    /// `Op::kill_payload(name)` don't need to carry a cgroup
    /// argument in the single-copy case.
    #[test]
    fn take_payload_by_bare_name_succeeds_on_single_copy() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );

        let mut backdrop = BackdropState::empty(&ctx);
        let mut scenario = ScenarioState::new(&mut state, &mut backdrop);
        let taken = scenario
            .take_payload_by_name("sleeper", None)
            .expect("single-copy bare lookup returns Ok")
            .expect("one entry matches");
        assert_eq!(taken.cgroup, "cg_a");
        assert!(state.payload_handles.is_empty());
        let _ = taken.handle.kill();
    }

    /// The apply_ops ambiguity hint must spell the full snake_case
    /// constructor path so a user copying the hint into source
    /// writes something that actually compiles. Covers both
    /// `Op::wait_payload` and `Op::kill_payload` entry points
    /// because they route through the same helper.
    #[test]
    fn apply_ops_bare_wait_and_kill_ambiguity_hint_names_full_constructor() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};
        static SLEEP: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);

        // WaitPayload path.
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );
        let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("sleeper")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("ambiguous"),
            "wait ambiguity message must flag ambiguity, got: {msg}"
        );
        assert!(
            msg.contains("Op::wait_payload_in_cgroup(name, cgroup)"),
            "wait ambiguity hint must name the full snake_case constructor \
             so a copy-paste into source compiles, got: {msg}"
        );
        drain_all_payload_handles(&mut state.payload_handles);

        // KillPayload path.
        let mut state = StepState::empty(&ctx);
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_a",
            PayloadSource::OpRunPayload,
        );
        push_fake_payload_entry(
            &ctx,
            &mut state,
            &SLEEP,
            "cg_b",
            PayloadSource::OpRunPayload,
        );
        let err = apply_ops_test(&ctx, &mut state, &[Op::kill_payload("sleeper")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("Op::kill_payload_in_cgroup(name, cgroup)"),
            "kill ambiguity hint must name the full snake_case constructor, got: {msg}"
        );
        drain_all_payload_handles(&mut state.payload_handles);
    }

    /// The not-found arm uses `-ing` verb form ("before waiting" /
    /// "before killing"), not the collapsed single-word lowercase
    /// a previous implementation emitted.
    #[test]
    fn apply_ops_not_found_message_uses_gerund_verb() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        let err = apply_ops_test(&ctx, &mut state, &[Op::wait_payload("ghost")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("before waiting"),
            "wait not-found message must say 'before waiting', got: {msg}"
        );
        assert!(
            !msg.contains("before waitpayload"),
            "must not collapse 'wait payload' into 'waitpayload', got: {msg}"
        );

        let err = apply_ops_test(&ctx, &mut state, &[Op::kill_payload("ghost")]).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("before killing"),
            "kill not-found message must say 'before killing', got: {msg}"
        );
    }

    // ---------------------------------------------------------------
    // Step-local vs Backdrop state invariants
    // ---------------------------------------------------------------

    /// Op::RemoveCgroup dispatches `ctx.cgroups.remove_cgroup`
    /// directly but does NOT forget the name from CgroupGroup's
    /// tracked `names` vec. Later, CgroupGroup's Drop iterates every
    /// tracked name and calls remove_cgroup again. The second call
    /// is swallowed by `let _ = ` in Drop, so the desync is
    /// observable (two remove_cgroup calls for the same name) but
    /// harmless. Pin the behavior so a future refactor that prunes
    /// names out from under a live Drop (or that stops swallowing
    /// the second error) surfaces here.
    #[test]
    fn remove_cgroup_does_not_forget_name_in_cgroupgroup_but_drop_is_safe() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup("cg_keep"), Op::add_cgroup("cg_drop")],
        )
        .unwrap();
        // Op::RemoveCgroup records on the mock but does NOT prune
        // `cg_drop` from the tracked names — both names stay.
        apply_ops_test(&ctx, &mut state, &[Op::remove_cgroup("cg_drop")]).unwrap();
        assert_eq!(
            state.cgroups.names(),
            &["cg_keep".to_string(), "cg_drop".to_string()],
            "Op::RemoveCgroup must not mutate CgroupGroup::names (current \
             invariant); Drop is the single rmdir dispatcher",
        );
        // The Drop call is safe because CgroupManager::remove_cgroup
        // is idempotent and CgroupGroup::drop swallows the second
        // error. Proved by dropping the state and asserting the mock
        // observed two RemoveCgroup calls for cg_drop (one from the
        // op, one from Drop), in that order, and did not panic.
        drop(state);
        let calls = mock.calls();
        let drops: Vec<&CgroupCall> = calls
            .iter()
            .filter(|c| matches!(c, CgroupCall::RemoveCgroup(n) if n == "cg_drop"))
            .collect();
        assert_eq!(
            drops.len(),
            2,
            "expected Op::RemoveCgroup + Drop to both hit the mock for cg_drop: {calls:?}",
        );
    }

    /// Step-local `Op::AddCgroup` with a name that already lives
    /// in the Backdrop must route through the same
    /// `cgroup_name_is_tracked` collision guard as `apply_setup`,
    /// rather than letting the CgroupGroup push a shadow entry that
    /// later steps could address. Currently
    /// `apply_ops`/`Op::AddCgroup` calls
    /// `target_cgroups().add_cgroup_no_cpuset(name)` with no
    /// collision check — this test documents the current behavior
    /// so a future refactor that adds the guard at the op level
    /// (mirroring apply_setup) flips the assertion.
    #[test]
    fn op_add_cgroup_step_local_allows_collision_with_backdrop_today() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        backdrop_state
            .cgroups
            .add_cgroup_no_cpuset("shared")
            .expect("add backdrop cgroup");
        let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
        // Current behavior: no collision guard at apply_ops level
        // — the op succeeds and the CgroupGroup's name list gains
        // a step-local copy of the same name. If a future change
        // lifts the apply_setup-style guard into apply_ops, this
        // assertion becomes an unwrap_err instead.
        let result = apply_ops(&ctx, &mut scenario, &[Op::add_cgroup("shared")]);
        assert!(
            result.is_ok(),
            "current apply_ops does not collision-check Op::AddCgroup; got: {result:?}",
        );
        assert!(
            step_state.cgroups.names().iter().any(|n| n == "shared"),
            "step-local names must include the new copy",
        );
        assert!(
            backdrop_state.cgroups.names().iter().any(|n| n == "shared"),
            "backdrop copy must survive the op",
        );
    }

    /// `Op::AddCgroup` applied twice in one step pushes
    /// two entries into the same CgroupGroup's `names` vec. Drop then
    /// calls `remove_cgroup` for each — the second hits an
    /// already-removed cgroup (safe via `let _ = `).
    #[test]
    fn op_add_cgroup_duplicate_in_same_step_pushes_twice() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[Op::add_cgroup("cg_dup"), Op::add_cgroup("cg_dup")],
        )
        .unwrap();
        let names = state.cgroups.names();
        assert_eq!(
            names.iter().filter(|n| n.as_str() == "cg_dup").count(),
            2,
            "current apply_ops pushes a name entry per Op::AddCgroup, even \
             on duplicate names; got: {names:?}",
        );
    }

    /// `MoveAllTasks` must re-key EVERY workload handle whose
    /// current name matches `from`, not just the first. Multiple
    /// handles on the same cgroup arise when a scenario issues two
    /// `Op::Spawn` ops on the same cgroup name.
    #[test]
    fn move_all_tasks_renames_every_handle_keyed_under_from() {
        use crate::workload::{AffinityMode, WorkType, WorkloadConfig, WorkloadHandle};

        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut step_state = StepState::empty(&ctx);
        let mut backdrop_state = BackdropState::empty(&ctx);
        step_state.cgroups.add_cgroup_no_cpuset("src").unwrap();
        step_state.cgroups.add_cgroup_no_cpuset("dst").unwrap();

        // Push THREE handles all keyed under "src" — simulates two
        // Op::Spawn ops in the same cgroup + one from CgroupDef.
        for _ in 0..3 {
            let wl = WorkloadConfig {
                num_workers: 1,
                affinity: AffinityMode::None,
                work_type: WorkType::CpuSpin,
                ..Default::default()
            };
            let h = WorkloadHandle::spawn(&wl).expect("spawn worker");
            step_state.handles.push(("src".to_string(), h));
        }
        assert_eq!(step_state.handles.len(), 3);

        {
            let mut scenario = ScenarioState::new(&mut step_state, &mut backdrop_state);
            apply_ops(&ctx, &mut scenario, &[Op::move_all_tasks("src", "dst")]).expect("move");
        }

        assert_eq!(step_state.handles.len(), 3, "no handles lost");
        assert!(
            step_state.handles.iter().all(|(name, _)| name == "dst"),
            "every handle must be re-keyed to 'dst': {:?}",
            step_state
                .handles
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
        );
        // SIGKILL before drop so the synthetic workers don't leak.
        step_state.handles.clear();
    }

    /// Per-step teardown is observable via the mock's call log.
    /// `execute_scenario` runs Step::cgroups Drop at step boundary;
    /// with MockCgroupOps we can pin that the rmdir calls happen
    /// (a) only on step-local cgroups, (b) in REVERSE order of
    /// addition — nested-cgroup-safe teardown.
    #[test]
    fn per_step_teardown_removes_step_local_cgroups_in_reverse_order() {
        let mock = MockCgroupOps::new();
        let topo = mock_topo();
        let ctx = mock_ctx(&mock, &topo);
        let mut state = StepState::empty(&ctx);
        apply_ops_test(
            &ctx,
            &mut state,
            &[
                Op::add_cgroup("cg_a"),
                Op::add_cgroup("cg_a/sub"),
                Op::add_cgroup("cg_b"),
            ],
        )
        .unwrap();
        // Simulate step boundary: drop the state to run CgroupGroup::Drop.
        drop(state);
        let calls = mock.calls();
        let removes: Vec<&str> = calls
            .iter()
            .filter_map(|c| match c {
                CgroupCall::RemoveCgroup(n) => Some(n.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            removes,
            vec!["cg_b", "cg_a/sub", "cg_a"],
            "per-step teardown must rmdir in reverse addition order so a \
             child cgroup's directory is gone before its parent's rmdir \
             runs",
        );
    }
}
