//! Scenario definitions, flag system, and test execution.
//!
//! Key types:
//! - [`Scenario`] -- data-driven test case (cgroups, cpusets, workloads)
//! - [`Ctx`] -- runtime context passed to scenario functions
//! - [`CpusetMode`] -- how to partition CPUs across cgroups
//! - [`Action`] -- steady-state or custom scenario logic
//! - [`CgroupWork`] -- per-cgroup workload definition
//! - [`AffinityKind`] -- scenario-level affinity intent
//! - [`CgroupGroup`] -- RAII guard that removes cgroups on drop
//! - [`FlagProfile`] -- a set of active flags for a run
//! - [`flags::FlagDecl`] -- typed flag declaration with dependencies
//!
//! The [`ops`] submodule provides composable cgroup topology operations.
//!
//! See the [Scenarios](https://likewhatevs.github.io/stt/guide/concepts/scenarios.html)
//! and [Writing Tests](https://likewhatevs.github.io/stt/guide/writing-tests.html)
//! chapters of the guide.

pub mod affinity;
pub mod basic;
mod catalog;
pub mod cpuset;
pub mod dynamic;
pub mod interaction;
pub mod nested;
pub mod ops;
pub mod performance;
pub mod scenarios;
pub mod stress;

pub use catalog::all_scenarios;

use std::collections::BTreeSet;
use std::thread;
use std::time::Duration;

use anyhow::Result;

use nix::sys::signal::kill;
use nix::unistd::Pid;

use crate::assert::{self, AssertResult};
use crate::cgroup::CgroupManager;
use crate::topology::TestTopology;
use crate::workload::*;

/// Check if a process is alive via kill(pid, 0).
fn process_alive(pid: u32) -> bool {
    kill(Pid::from_raw(pid as i32), None).is_ok()
}

pub(crate) use crate::read_kmsg;

// ---------------------------------------------------------------------------
// Flag system
// ---------------------------------------------------------------------------

/// Flag declarations and name constants.
///
/// The built-in `*_DECL` constants (`LLC_DECL`, `BORROW_DECL`, etc.)
/// have empty `args` fields. They define flag names and dependencies
/// for stt's internal scenario catalog. External consumers define
/// their own `FlagDecl` statics with populated `args` fields
/// containing their scheduler's actual CLI arguments.
///
/// String name constants (`LLC`, `BORROW`, etc.) are used in
/// [`FlagProfile::flags`] and [`generate_profiles()`](Scheduler::generate_profiles).
pub mod flags {
    /// A scheduler feature flag with CLI arguments and dependencies.
    ///
    /// Each scheduler defines its own `FlagDecl` statics. The `args`
    /// field contains CLI arguments passed to the scheduler binary when
    /// the flag is active. The `requires` field expresses dependencies
    /// between flags (e.g. work-stealing requires LLC awareness).
    ///
    /// `FlagDecl` is re-exported in the [prelude](crate::prelude).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use stt::prelude::*;
    ///
    /// static MY_LLC: FlagDecl = FlagDecl {
    ///     name: "llc",
    ///     args: &["--enable-llc-awareness"],
    ///     requires: &[],
    /// };
    ///
    /// static MY_STEAL: FlagDecl = FlagDecl {
    ///     name: "steal",
    ///     args: &["--enable-work-stealing"],
    ///     requires: &[&MY_LLC],
    /// };
    /// ```
    pub struct FlagDecl {
        /// Flag name used in profiles and constraint matching
        /// (e.g. `"llc"`, `"borrow"`, `"steal"`).
        pub name: &'static str,
        /// CLI arguments passed to the scheduler binary when this flag
        /// is active. Empty for the built-in `*_DECL` constants;
        /// external consumers populate this with their scheduler's
        /// actual flags (e.g. `&["--enable-llc-awareness"]`).
        pub args: &'static [&'static str],
        /// Flags that must also be active when this flag is active.
        /// `generate_profiles()` rejects combinations where a
        /// required flag is missing.
        pub requires: &'static [&'static FlagDecl],
    }

    impl std::fmt::Debug for FlagDecl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let req_names: Vec<&str> = self.requires.iter().map(|d| d.name).collect();
            f.debug_struct("FlagDecl")
                .field("name", &self.name)
                .field("args", &self.args)
                .field("requires", &req_names)
                .finish()
        }
    }

    pub static LLC_DECL: FlagDecl = FlagDecl {
        name: "llc",
        args: &[],
        requires: &[],
    };
    pub static BORROW_DECL: FlagDecl = FlagDecl {
        name: "borrow",
        args: &[],
        requires: &[],
    };
    pub static STEAL_DECL: FlagDecl = FlagDecl {
        name: "steal",
        args: &[],
        requires: &[&LLC_DECL],
    };
    pub static REBAL_DECL: FlagDecl = FlagDecl {
        name: "rebal",
        args: &[],
        requires: &[],
    };
    pub static REJECT_PIN_DECL: FlagDecl = FlagDecl {
        name: "reject-pin",
        args: &[],
        requires: &[],
    };
    pub static NO_CTRL_DECL: FlagDecl = FlagDecl {
        name: "no-ctrl",
        args: &[],
        requires: &[],
    };

    /// All flag declarations in canonical order.
    pub static ALL_DECLS: &[&FlagDecl] = &[
        &LLC_DECL,
        &BORROW_DECL,
        &STEAL_DECL,
        &REBAL_DECL,
        &REJECT_PIN_DECL,
        &NO_CTRL_DECL,
    ];

    // String name constants for FlagProfile.flags.
    pub const LLC: &str = "llc";
    pub const BORROW: &str = "borrow";
    pub const STEAL: &str = "steal";
    pub const REBAL: &str = "rebal";
    pub const REJECT_PIN: &str = "reject-pin";
    pub const NO_CTRL: &str = "no-ctrl";

    pub const ALL: &[&str] = &[LLC, BORROW, STEAL, REBAL, REJECT_PIN, NO_CTRL];

    pub fn from_short_name(s: &str) -> Option<&'static str> {
        ALL.iter().find(|&&f| f == s).copied()
    }

    /// Look up a FlagDecl by name.
    pub fn decl_by_name(name: &str) -> Option<&'static FlagDecl> {
        ALL_DECLS.iter().find(|d| d.name == name).copied()
    }
}

/// A set of active flags for a scenario run.
///
/// Display name is the flags joined with `+` (e.g. `"llc+borrow"`),
/// or `"default"` when empty.
#[derive(Debug, Clone)]
pub struct FlagProfile {
    /// Active flags, sorted in canonical order.
    pub flags: Vec<&'static str>,
}

impl FlagProfile {
    /// Display name: flags joined with `+`, or `"default"` when empty.
    pub fn name(&self) -> String {
        if self.flags.is_empty() {
            "default".into()
        } else {
            self.flags.join("+")
        }
    }
}

/// Look up flag dependencies via FlagDecl.requires.
fn flag_requires(flag: &str) -> Vec<&'static str> {
    flags::decl_by_name(flag)
        .map(|d| d.requires.iter().map(|r| r.name).collect())
        .unwrap_or_default()
}

fn generate_profiles(required: &[&'static str], excluded: &[&'static str]) -> Vec<FlagProfile> {
    let optional: Vec<&'static str> = flags::ALL
        .iter()
        .copied()
        .filter(|f| !required.contains(f) && !excluded.contains(f))
        .collect();
    let mut out = Vec::new();
    for mask in 0..(1u32 << optional.len()) {
        let mut fl: Vec<&'static str> = required.to_vec();
        for (i, &f) in optional.iter().enumerate() {
            if mask & (1 << i) != 0 {
                fl.push(f);
            }
        }
        let valid = fl
            .iter()
            .all(|f| flag_requires(f).iter().all(|r| fl.contains(r)));
        if valid {
            fl.sort_by_key(|f| flags::ALL.iter().position(|a| a == f).unwrap_or(usize::MAX));
            out.push(FlagProfile { flags: fl });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scenario definition (data-driven)
// ---------------------------------------------------------------------------

/// Scenario-level CPU partitioning strategy.
///
/// Determines how CPUs are split across all cgroups in a data-driven
/// [`Scenario`]. Each variant produces a different cpuset assignment
/// based on the VM's [`TestTopology`] and the scenario's `num_cgroups`.
///
/// This is distinct from [`ops::CpusetSpec`], which computes a single
/// cpuset for one cgroup in the ops/steps system. `CpusetMode`
/// partitions at the scenario level; `CpusetSpec` specifies per-cgroup.
#[derive(Clone, Debug)]
pub enum CpusetMode {
    None,
    LlcAligned,
    SplitHalf,
    SplitMisaligned,
    Overlap(f64),
    Uneven(f64),   // fraction for cgroup 0
    Holdback(f64), // fraction of CPUs held back, rest split evenly
}

/// What happens during a scenario's workload phase.
///
/// `Steady` runs workers for the configured duration with no dynamic
/// operations. `Custom` allows arbitrary test logic via a function
/// that receives the [`Ctx`].
#[derive(Clone)]
pub enum Action {
    /// Run workers for the configured duration with no dynamic operations.
    Steady,
    /// Execute custom scenario logic with access to the full test context.
    Custom(fn(&Ctx) -> Result<AssertResult>),
}

impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Steady => write!(f, "Steady"),
            Action::Custom(_) => write!(f, "Custom(fn)"),
        }
    }
}

/// Per-cgroup workload definition.
///
/// Specifies the number of workers, their [`WorkType`], scheduling
/// policy, and affinity for a single cgroup in a [`Scenario`].
///
/// When a scenario has fewer `CgroupWork` entries than `num_cgroups`,
/// the first entry is reused for remaining cgroups.
#[derive(Clone, Debug)]
pub struct CgroupWork {
    /// Number of workers. `None` means use `Ctx::workers_per_cgroup`.
    pub num_workers: Option<usize>,
    /// What each worker process does.
    pub work_type: WorkType,
    /// Linux scheduling policy for workers.
    pub policy: SchedPolicy,
    /// How to set worker CPU affinity.
    pub affinity: AffinityKind,
}

/// Scenario-level affinity intent.
///
/// Resolved to a concrete [`AffinityMode`] at runtime based on the
/// topology and cpuset assignments.
#[derive(Clone, Debug)]
pub enum AffinityKind {
    /// No affinity constraint -- inherit from parent cgroup.
    Inherit,
    /// Pin to a random subset of the cgroup's cpuset.
    RandomSubset,
    /// Pin to the CPUs in the worker's LLC.
    LlcAligned,
    /// Pin to all CPUs (crosses cgroup boundaries).
    CrossCgroup,
    /// Pin to a single CPU.
    SingleCpu,
}

impl Default for CgroupWork {
    fn default() -> Self {
        Self {
            num_workers: None,
            work_type: WorkType::CpuSpin,
            policy: SchedPolicy::Normal,
            affinity: AffinityKind::Inherit,
        }
    }
}

/// A data-driven test case.
///
/// Declares the cgroup topology, CPU partitioning, workloads, and
/// execution mode for a single test scenario. All scenarios are
/// registered in [`all_scenarios()`].
///
/// # Flag constraints
///
/// `required_flags` and `excluded_flags` use typed [`FlagDecl`](flags::FlagDecl)
/// references. [`profiles()`](Scenario::profiles) generates all valid
/// flag combinations that satisfy these constraints.
///
/// ```
/// # use stt::scenario::all_scenarios;
/// let scenarios = all_scenarios();
/// assert!(!scenarios.is_empty());
///
/// let first = &scenarios[0];
/// assert!(!first.name.is_empty());
/// assert!(!first.category.is_empty());
///
/// let profiles = first.profiles();
/// assert!(!profiles.is_empty());
/// ```
#[derive(Clone, Debug)]
pub struct Scenario {
    /// Unique identifier (e.g. `"cgroup_steady"`).
    pub name: &'static str,
    /// Category: basic, cpuset, affinity, sched_class, dynamic, stress,
    /// stall, advanced, nested, interaction, or performance.
    pub category: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// Flags that must be present in every run.
    pub required_flags: &'static [&'static flags::FlagDecl],
    /// Flags that must not be present in any run.
    pub excluded_flags: &'static [&'static flags::FlagDecl],
    /// Number of cgroups to create.
    pub num_cgroups: usize,
    /// How to partition CPUs across cgroups.
    pub cpuset_mode: CpusetMode,
    /// Per-cgroup workload definitions.
    pub cgroup_works: Vec<CgroupWork>,
    /// Execution mode: steady-state or custom logic.
    pub action: Action,
}

impl Scenario {
    /// Generate all valid flag profiles for this scenario.
    ///
    /// Enumerates all flag combinations that satisfy `required_flags`,
    /// `excluded_flags`, and per-flag `requires` dependencies.
    pub fn profiles(&self) -> Vec<FlagProfile> {
        let req: Vec<&'static str> = self.required_flags.iter().map(|d| d.name).collect();
        let excl: Vec<&'static str> = self.excluded_flags.iter().map(|d| d.name).collect();
        generate_profiles(&req, &excl)
    }
    #[allow(dead_code)]
    pub fn profiles_with(&self, active: &[&str]) -> Vec<FlagProfile> {
        let req: Vec<&'static str> = self.required_flags.iter().map(|d| d.name).collect();
        let mut excl: Vec<&'static str> = self.excluded_flags.iter().map(|d| d.name).collect();
        for &f in flags::ALL {
            if !active.contains(&f) && !req.contains(&f) {
                excl.push(f);
            }
        }
        generate_profiles(&req, &excl)
    }
    /// Returns `"{name}/{profile_name}"` (e.g. `"cgroup_steady/llc+borrow"`).
    pub fn qualified_name(&self, p: &FlagProfile) -> String {
        format!("{}/{}", self.name, p.name())
    }
}

// ---------------------------------------------------------------------------
// RAII cgroup group
// ---------------------------------------------------------------------------

/// RAII guard that removes cgroups on drop.
///
/// Prevents cgroup leaks when workload spawning or other operations fail
/// between cgroup creation and cleanup.
#[derive(Debug)]
#[must_use = "dropping a CgroupGroup immediately destroys the cgroups it manages"]
pub struct CgroupGroup<'a> {
    cgroups: &'a CgroupManager,
    names: Vec<String>,
}

impl<'a> CgroupGroup<'a> {
    /// Create an empty group. Cgroups added via `add_cgroup` or
    /// `add_cgroup_no_cpuset` are removed when the group is dropped.
    pub fn new(cgroups: &'a CgroupManager) -> Self {
        Self {
            cgroups,
            names: Vec::new(),
        }
    }

    /// Create a cgroup and set its cpuset. The cgroup is tracked for cleanup on drop.
    pub fn add_cgroup(&mut self, name: &str, cpuset: &BTreeSet<usize>) -> Result<()> {
        self.cgroups.create_cgroup(name)?;
        self.cgroups.set_cpuset(name, cpuset)?;
        self.names.push(name.to_string());
        Ok(())
    }

    /// Create a cgroup without a cpuset. The cgroup is tracked for cleanup on drop.
    pub fn add_cgroup_no_cpuset(&mut self, name: &str) -> Result<()> {
        self.cgroups.create_cgroup(name)?;
        self.names.push(name.to_string());
        Ok(())
    }

    /// Names of all tracked cgroups.
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

impl Drop for CgroupGroup<'_> {
    fn drop(&mut self) {
        for name in &self.names {
            let _ = self.cgroups.remove_cgroup(name);
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime context and interpreter
// ---------------------------------------------------------------------------

/// Runtime context passed to scenario functions.
///
/// Provides access to cgroup management, topology information, and
/// test configuration. Custom scenarios (`Action::Custom`) receive
/// this as their sole parameter.
#[derive(Debug)]
pub struct Ctx<'a> {
    /// Cgroup filesystem manager for creating/removing cgroups.
    pub cgroups: &'a CgroupManager,
    /// VM CPU topology.
    pub topo: &'a TestTopology,
    /// How long to run the workload.
    pub duration: Duration,
    /// Default number of workers per cgroup.
    pub workers_per_cgroup: usize,
    /// PID of the running scheduler (for liveness checks).
    pub sched_pid: u32,
    /// Time to wait after cgroup creation for scheduler stabilization.
    pub settle: Duration,
    /// Override work type for scenarios that use `CpuSpin` by default.
    pub work_type_override: Option<WorkType>,
    /// Merged assertion config (default_checks + scheduler + per-test).
    /// Used by `run_scenario` for data-driven scenarios and by
    /// `execute_steps` as the default when no explicit checks are
    /// passed to `execute_steps_with`.
    pub assert: crate::assert::Assert,
    /// When true, `execute_steps` polls SHM signal slot 0 after writing
    /// the scenario start marker, blocking until the host confirms its
    /// BPF map write is complete.
    pub wait_for_map_write: bool,
}

/// Run a scenario. Returns assertion result.
pub fn run_scenario(scenario: &Scenario, ctx: &Ctx) -> Result<AssertResult> {
    tracing::info!(scenario = scenario.name, "running");
    if let Action::Custom(f) = &scenario.action {
        return f(ctx);
    }

    let cpusets = resolve_cpusets(&scenario.cpuset_mode, scenario.num_cgroups, ctx.topo);

    // Skip if topology doesn't support the test
    if let Some(ref cs) = cpusets
        && cs.iter().any(|s| s.is_empty())
    {
        return Ok(AssertResult::skip("skipped: not enough CPUs/LLCs"));
    }

    let names: Vec<String> = (0..scenario.num_cgroups)
        .map(|i| format!("cg_{i}"))
        .collect();
    let mut cgroup_guard = CgroupGroup::new(ctx.cgroups);
    for (i, name) in names.iter().enumerate() {
        cgroup_guard.add_cgroup_no_cpuset(name)?;
        if let Some(ref cs) = cpusets {
            ctx.cgroups.set_cpuset(name, &cs[i])?;
        }
    }
    tracing::debug!(cgroups = scenario.num_cgroups, "cgroups created, settling");
    thread::sleep(ctx.settle);

    // Bail early if the scheduler died after cgroup creation
    if !process_alive(ctx.sched_pid) {
        anyhow::bail!("scheduler died after cgroup creation");
    }

    let mut handles = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let cw = scenario
            .cgroup_works
            .get(i)
            .or(scenario.cgroup_works.first())
            .cloned()
            .unwrap_or_default();
        let n = cw.num_workers.unwrap_or(ctx.workers_per_cgroup);
        let affinity = resolve_affinity_kind(&cw.affinity, cpusets.as_deref(), i, ctx.topo);
        let effective_work_type = crate::workload::resolve_work_type(
            &cw.work_type,
            ctx.work_type_override.as_ref(),
            matches!(cw.work_type, WorkType::CpuSpin),
            n,
        );
        let wl = WorkloadConfig {
            num_workers: n,
            affinity,
            work_type: effective_work_type,
            sched_policy: cw.policy,
        };
        let h = WorkloadHandle::spawn(&wl)?;
        tracing::debug!(cgroup = %name, workers = n, tids = h.tids().len(), "spawned workers");
        for tid in h.tids() {
            ctx.cgroups.move_task(name, tid)?;
        }
        handles.push(h);
    }

    // Start all workers now that they're in their cgroups
    for h in &mut handles {
        h.start();
    }

    tracing::debug!(duration_s = ctx.duration.as_secs(), "running workload");

    // Poll scheduler liveness during the workload phase instead of a
    // single sleep. Detects scheduler death within 500ms rather than
    // waiting the full duration and collecting misleading results.
    let deadline = std::time::Instant::now() + ctx.duration;
    let mut sched_dead = false;
    while std::time::Instant::now() < deadline {
        if !process_alive(ctx.sched_pid) {
            sched_dead = true;
            tracing::warn!("scheduler died during workload phase");
            break;
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        thread::sleep(remaining.min(Duration::from_millis(500)));
    }

    if !sched_dead {
        sched_dead = !process_alive(ctx.sched_pid);
    }

    let mut result = AssertResult::pass();
    for (i, h) in handles.into_iter().enumerate() {
        let reports = h.stop_and_collect();
        let cs = cpusets.as_ref().map(|v| &v[i]);
        result.merge(ctx.assert.assert_cgroup(&reports, cs));
    }

    // Capture kernel log on failure
    if !result.passed {
        for line in read_kmsg().lines() {
            result.details.push(line.to_string());
        }
    }

    if sched_dead {
        result.passed = false;
        result.details.push("scheduler died".into());
    }

    Ok(result)
}

fn resolve_cpusets(
    mode: &CpusetMode,
    n: usize,
    topo: &TestTopology,
) -> Option<Vec<BTreeSet<usize>>> {
    let all = topo.all_cpus();
    let usable = topo.usable_cpus();
    match mode {
        CpusetMode::None => None,
        CpusetMode::LlcAligned => {
            let llcs = topo.split_by_llc();
            if llcs.len() < 2 {
                return Some(vec![BTreeSet::new()]);
            }
            // Remove last CPU from last LLC to reserve for cgroup 0
            let mut sets: Vec<BTreeSet<usize>> = llcs[..n.min(llcs.len())].to_vec();
            if let Some(last) = sets.last_mut()
                && last.len() > 1
            {
                last.remove(&all[all.len() - 1]);
            }
            Some(sets)
        }
        CpusetMode::SplitHalf => {
            let mid = usable.len() / 2;
            Some(vec![
                usable[..mid].iter().copied().collect(),
                usable[mid..].iter().copied().collect(),
            ])
        }
        CpusetMode::SplitMisaligned => {
            let split = if topo.num_llcs() > 1 {
                topo.cpus_in_llc(0).len() / 2
            } else {
                usable.len() / 2
            };
            Some(vec![
                usable[..split].iter().copied().collect(),
                usable[split..].iter().copied().collect(),
            ])
        }
        CpusetMode::Overlap(frac) => Some(topo.overlapping_cpusets(n, *frac)),
        CpusetMode::Uneven(frac) => {
            let split = (usable.len() as f64 * frac) as usize;
            Some(vec![
                usable[..split.max(1)].iter().copied().collect(),
                usable[split.max(1)..].iter().copied().collect(),
            ])
        }
        CpusetMode::Holdback(frac) => {
            let keep = all.len() - (all.len() as f64 * frac) as usize;
            let mid = keep / 2;
            Some(vec![
                all[..mid.max(1)].iter().copied().collect(),
                all[mid.max(1)..keep].iter().copied().collect(),
            ])
        }
    }
}

fn resolve_affinity_kind(
    kind: &AffinityKind,
    cpusets: Option<&[BTreeSet<usize>]>,
    cgroup_idx: usize,
    topo: &TestTopology,
) -> AffinityMode {
    match kind {
        AffinityKind::Inherit => AffinityMode::None,
        AffinityKind::RandomSubset => {
            let pool = cpusets
                .map(|cs| cs[cgroup_idx].clone())
                .unwrap_or_else(|| topo.all_cpuset());
            let count = (pool.len() / 2).max(1);
            AffinityMode::Random { from: pool, count }
        }
        AffinityKind::LlcAligned => {
            let idx = cgroup_idx % topo.num_llcs();
            AffinityMode::Fixed(topo.llc_aligned_cpuset(idx))
        }
        AffinityKind::CrossCgroup => AffinityMode::Fixed(topo.all_cpuset()),
        AffinityKind::SingleCpu => {
            let cpu = topo.all_cpus()[cgroup_idx % topo.total_cpus()];
            AffinityMode::SingleCpu(cpu)
        }
    }
}

// ---------------------------------------------------------------------------
// Custom scenario helpers
// ---------------------------------------------------------------------------

/// Create N cgroups, spawn workers in each, and start them.
///
/// Returns the worker handles and an RAII [`CgroupGroup`] that removes
/// the cgroups on drop. Workers are moved into their target cgroups
/// before being signaled to start.
pub fn setup_cgroups<'a>(
    ctx: &'a Ctx,
    n: usize,
    wl: &WorkloadConfig,
) -> Result<(Vec<WorkloadHandle>, CgroupGroup<'a>)> {
    let mut guard = CgroupGroup::new(ctx.cgroups);
    for i in 0..n {
        guard.add_cgroup_no_cpuset(&format!("cg_{i}"))?;
    }
    thread::sleep(ctx.settle);
    if !process_alive(ctx.sched_pid) {
        anyhow::bail!("scheduler died after cgroup creation");
    }
    let handles: Result<Vec<_>> = (0..n)
        .map(|i| {
            let h = WorkloadHandle::spawn(wl)?;
            ctx.cgroups.move_tasks(&format!("cg_{i}"), &h.tids())?;
            Ok(h)
        })
        .collect();
    let mut handles = handles?;
    for h in &mut handles {
        h.start();
    }
    Ok((handles, guard))
}

/// Stop all workers, collect reports, and run assertion checks.
///
/// Uses `checks` for worker evaluation. When the Assert has no
/// worker-level checks configured (all fields None), falls back
/// to `assert_not_starved`. Returns a merged [`AssertResult`]
/// across all workers.
pub fn collect_all(handles: Vec<WorkloadHandle>, checks: &crate::assert::Assert) -> AssertResult {
    let mut r = AssertResult::pass();
    for h in handles {
        let reports = h.stop_and_collect();
        if checks.has_worker_checks() {
            r.merge(checks.assert_cgroup(&reports, None));
        } else {
            r.merge(assert::assert_not_starved(&reports));
        }
    }
    r
}

/// Default [`WorkloadConfig`] with `ctx.workers_per_cgroup` workers.
pub fn dfl_wl(ctx: &Ctx) -> WorkloadConfig {
    WorkloadConfig {
        num_workers: ctx.workers_per_cgroup,
        ..Default::default()
    }
}

#[cfg(test)]
pub fn split_half(ctx: &Ctx) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let usable = ctx.topo.usable_cpus();
    let mid = usable.len() / 2;
    (
        usable[..mid].iter().copied().collect(),
        usable[mid..].iter().copied().collect(),
    )
}

/// Spawn diverse workloads across N cgroups: CpuSpin, Bursty, IoSync, Mixed, YieldHeavy.
pub fn spawn_diverse(ctx: &Ctx, cgroup_names: &[&str]) -> Result<Vec<WorkloadHandle>> {
    let types = [
        WorkType::CpuSpin,
        WorkType::Bursty {
            burst_ms: 50,
            sleep_ms: 100,
        },
        WorkType::IoSync,
        WorkType::Mixed,
        WorkType::YieldHeavy,
    ];
    let mut handles = Vec::new();
    for (i, name) in cgroup_names.iter().enumerate() {
        let wt = types[i % types.len()].clone();
        let n = if matches!(wt, WorkType::IoSync) {
            2
        } else {
            ctx.workers_per_cgroup
        };
        let mut h = WorkloadHandle::spawn(&WorkloadConfig {
            num_workers: n,
            work_type: wt,
            ..Default::default()
        })?;
        ctx.cgroups.move_tasks(name, &h.tids())?;
        h.start();
        handles.push(h);
    }
    Ok(handles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_short_name_roundtrip() {
        for &f in flags::ALL {
            assert_eq!(flags::from_short_name(f), Some(f));
        }
    }

    #[test]
    fn flag_all_unique_short_names() {
        let unique: std::collections::HashSet<&&str> = flags::ALL.iter().collect();
        assert_eq!(flags::ALL.len(), unique.len());
    }

    #[test]
    fn flag_from_short_name_unknown() {
        assert_eq!(flags::from_short_name("nonexistent"), None);
    }

    #[test]
    fn profile_name_default() {
        assert_eq!(FlagProfile { flags: vec![] }.name(), "default");
    }

    #[test]
    fn profile_name_with_flags() {
        let p = FlagProfile {
            flags: vec![flags::LLC, flags::BORROW],
        };
        assert_eq!(p.name(), "llc+borrow");
    }

    #[test]
    fn generate_profiles_no_constraints() {
        // 2^6=64 minus 16 invalid (steal without llc) = 48
        assert_eq!(generate_profiles(&[], &[]).len(), 48);
    }

    #[test]
    fn generate_profiles_work_stealing_requires_llc() {
        let profiles = generate_profiles(&[flags::STEAL], &[]);
        for p in &profiles {
            assert!(
                p.flags.contains(&flags::LLC),
                "steal without llc: {:?}",
                p.flags
            );
        }
    }

    #[test]
    fn generate_profiles_excluded_never_present() {
        let profiles = generate_profiles(&[], &[flags::NO_CTRL]);
        for p in &profiles {
            assert!(!p.flags.contains(&flags::NO_CTRL));
        }
    }

    #[test]
    fn generate_profiles_required_always_present() {
        let profiles = generate_profiles(&[flags::BORROW], &[]);
        for p in &profiles {
            assert!(p.flags.contains(&flags::BORROW));
        }
    }

    #[test]
    fn generate_profiles_required_and_excluded() {
        let profiles = generate_profiles(&[flags::BORROW], &[flags::REBAL]);
        for p in &profiles {
            assert!(p.flags.contains(&flags::BORROW));
            assert!(!p.flags.contains(&flags::REBAL));
        }
    }

    #[test]
    fn all_scenarios_non_empty() {
        assert!(!all_scenarios().is_empty());
    }

    #[test]
    fn all_scenarios_unique_names() {
        let scenarios = all_scenarios();
        let names: Vec<&str> = scenarios.iter().map(|s| s.name).collect();
        let unique: std::collections::HashSet<&&str> = names.iter().collect();
        assert_eq!(names.len(), unique.len(), "duplicate scenario names");
    }

    #[test]
    fn all_scenarios_have_profiles() {
        for s in &all_scenarios() {
            assert!(!s.profiles().is_empty(), "{} has no valid profiles", s.name);
        }
    }

    #[test]
    fn resolve_cpusets_none_returns_none() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        assert!(resolve_cpusets(&CpusetMode::None, 2, &t).is_none());
    }

    #[test]
    fn resolve_cpusets_split_half_covers_usable() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::SplitHalf, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        // Last CPU reserved for cgroup 0 → 7 usable
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn resolve_cpusets_llc_aligned() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::LlcAligned, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        // Both sets non-empty
        assert!(!r[0].is_empty());
        assert!(!r[1].is_empty());
    }

    #[test]
    fn resolve_cpusets_uneven() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::Uneven(0.75), 2, &t).unwrap();
        assert!(r[0].len() > r[1].len(), "75/25 split should be uneven");
    }

    #[test]
    fn resolve_cpusets_holdback() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::Holdback(0.5), 2, &t).unwrap();
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert!(total < 8, "holdback should use fewer CPUs");
    }

    #[test]
    fn resolve_cpusets_overlap() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::Overlap(0.5), 3, &t).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn resolve_affinity_inherit() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        assert!(matches!(
            resolve_affinity_kind(&AffinityKind::Inherit, None, 0, &t),
            AffinityMode::None
        ));
    }

    #[test]
    fn resolve_affinity_single_cpu() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_kind(&AffinityKind::SingleCpu, None, 0, &t) {
            AffinityMode::SingleCpu(c) => assert_eq!(c, 0),
            other => panic!("expected SingleCpu, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_cross_cgroup() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_kind(&AffinityKind::CrossCgroup, None, 0, &t) {
            AffinityMode::Fixed(cpus) => assert_eq!(cpus.len(), 8),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_kind(&AffinityKind::LlcAligned, None, 1, &t) {
            AffinityMode::Fixed(cpus) => assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect()),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_random_subset() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1, 2, 3].into_iter().collect()];
        match resolve_affinity_kind(&AffinityKind::RandomSubset, Some(&cpusets), 0, &t) {
            AffinityMode::Random { from, count } => {
                assert_eq!(from, cpusets[0]);
                assert_eq!(count, 2); // half of 4
            }
            other => panic!("expected Random, got {:?}", other),
        }
    }

    #[test]
    fn profiles_with_filters_correctly() {
        let s = &all_scenarios()[0]; // proportional, no required/excluded
        let profiles = s.profiles_with(&[flags::BORROW]);
        for p in &profiles {
            // Only borrow (and its dependencies) should be possible
            for f in &p.flags {
                assert!(
                    *f == flags::BORROW || flag_requires(flags::BORROW).contains(f),
                    "unexpected flag {:?}",
                    f
                );
            }
        }
    }

    #[test]
    fn resolve_cpusets_split_misaligned() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::SplitMisaligned, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert!(total > 0);
        // Misaligned means split within an LLC, not at LLC boundary
        assert_ne!(r[0].len(), 4, "misaligned should NOT split at LLC boundary");
    }

    #[test]
    fn resolve_cpusets_llc_aligned_single_llc() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let r = resolve_cpusets(&CpusetMode::LlcAligned, 2, &t).unwrap();
        // With 1 LLC, can only make 1 set -> returns empty for missing
        assert!(
            r.iter().any(|s| s.is_empty()),
            "should signal skip with empty set"
        );
    }

    #[test]
    fn resolve_cpusets_small_topology() {
        let t = crate::topology::TestTopology::synthetic(2, 1);
        let r = resolve_cpusets(&CpusetMode::SplitHalf, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        // 2 CPUs, no reserve (too small), each gets 1
        assert_eq!(r[0].len(), 1);
        assert_eq!(r[1].len(), 1);
    }

    #[test]
    fn cgroup_work_default() {
        let cw = CgroupWork::default();
        assert_eq!(cw.num_workers, None);
        assert!(matches!(cw.work_type, WorkType::CpuSpin));
        assert!(matches!(cw.policy, SchedPolicy::Normal));
        assert!(matches!(cw.affinity, AffinityKind::Inherit));
    }

    #[test]
    fn scenario_qualified_name() {
        let s = &all_scenarios()[0];
        let p = FlagProfile { flags: vec![] };
        assert_eq!(s.qualified_name(&p), format!("{}/default", s.name));
    }

    #[test]
    fn scenario_qualified_name_with_flags() {
        let s = &all_scenarios()[0];
        let p = FlagProfile {
            flags: vec![flags::LLC, flags::BORROW],
        };
        assert_eq!(s.qualified_name(&p), format!("{}/llc+borrow", s.name));
    }

    #[test]
    fn all_scenarios_count() {
        let scenarios = all_scenarios();
        assert!(
            scenarios.len() >= 30,
            "expected >=30 scenarios, got {}",
            scenarios.len()
        );
    }

    #[test]
    fn scenario_categories_valid() {
        let valid = [
            "basic",
            "cpuset",
            "affinity",
            "sched_class",
            "dynamic",
            "stress",
            "stall",
            "advanced",
            "nested",
            "interaction",
            "performance",
        ];
        for s in &all_scenarios() {
            assert!(
                valid.contains(&s.category),
                "unknown category '{}' in {}",
                s.category,
                s.name
            );
        }
    }

    #[test]
    fn generate_profiles_single_required_count() {
        // Required=[borrow], 5 optional, steal needs llc
        // 2^5=32 minus 8 invalid = 24
        assert_eq!(generate_profiles(&[flags::BORROW], &[]).len(), 24);
    }

    #[test]
    fn profiles_sorted_by_flag_order() {
        for p in &generate_profiles(&[], &[]) {
            for w in p.flags.windows(2) {
                let pos0 = flags::ALL.iter().position(|a| a == &w[0]).unwrap();
                let pos1 = flags::ALL.iter().position(|a| a == &w[1]).unwrap();
                assert!(pos0 < pos1, "flags not sorted: {:?}", p.flags);
            }
        }
    }

    #[test]
    fn resolve_cpusets_holdback_reserves_cpus() {
        let t = crate::topology::TestTopology::synthetic(12, 3);
        let r = resolve_cpusets(&CpusetMode::Holdback(0.33), 2, &t).unwrap();
        let total: usize = r.iter().map(|s| s.len()).sum();
        // 12 CPUs, holdback 33%: keep = 12 - floor(12*0.33) = 12 - 3 = 9
        assert_eq!(total, 9, "holdback 33% of 12 should keep 9");
        assert!(total < 12, "holdback should use fewer CPUs than total");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn resolve_cpusets_overlap_sets_overlap() {
        let t = crate::topology::TestTopology::synthetic(12, 1);
        let r = resolve_cpusets(&CpusetMode::Overlap(0.5), 2, &t).unwrap();
        let overlap: BTreeSet<usize> = r[0].intersection(&r[1]).copied().collect();
        assert!(
            !overlap.is_empty(),
            "50% overlap should have overlapping CPUs"
        );
    }

    #[test]
    fn resolve_affinity_random_no_cpusets() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_kind(&AffinityKind::RandomSubset, None, 0, &t) {
            AffinityMode::Random { from, count } => {
                assert_eq!(from.len(), 8); // all CPUs
                assert_eq!(count, 4); // half
            }
            other => panic!("expected Random, got {:?}", other),
        }
    }

    #[test]
    fn split_half_even() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &ctx_cg,
            topo: &t,
            duration: std::time::Duration::from_secs(1),
            workers_per_cgroup: 4,
            sched_pid: 0,
            settle: Duration::from_millis(3000),
            work_type_override: None,
            assert: assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let (a, b) = split_half(&ctx);
        // Last CPU reserved for cgroup 0 → 7 usable, split 3/4
        assert_eq!(a.len() + b.len(), 7);
        assert!(a.intersection(&b).count() == 0, "halves should not overlap");
    }

    #[test]
    fn split_half_small() {
        let t = crate::topology::TestTopology::synthetic(2, 1);
        let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &ctx_cg,
            topo: &t,
            duration: std::time::Duration::from_secs(1),
            workers_per_cgroup: 1,
            sched_pid: 0,
            settle: Duration::from_millis(3000),
            work_type_override: None,
            assert: assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let (a, b) = split_half(&ctx);
        assert_eq!(a.len() + b.len(), 2);
    }

    #[test]
    fn dfl_wl_propagates_workers() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let ctx_cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &ctx_cg,
            topo: &t,
            duration: std::time::Duration::from_secs(1),
            workers_per_cgroup: 7,
            sched_pid: 0,
            settle: Duration::from_millis(3000),
            work_type_override: None,
            assert: assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let wl = dfl_wl(&ctx);
        assert_eq!(wl.num_workers, 7);
        assert!(matches!(wl.work_type, WorkType::CpuSpin));
    }

    #[test]
    fn process_alive_current_pid() {
        let pid = std::process::id();
        assert!(process_alive(pid));
    }

    #[test]
    fn process_alive_nonexistent_pid() {
        // Read /proc/sys/kernel/pid_max and use pid_max - 1, which is
        // valid as a PID value but extremely unlikely to be in use.
        let pid_max: u32 = std::fs::read_to_string("/proc/sys/kernel/pid_max")
            .unwrap_or_else(|_| "4194304".into())
            .trim()
            .parse()
            .unwrap_or(4194304);
        // pid_max - 1 is a valid PID but should not be alive.
        // In the astronomically unlikely case it IS alive, skip the test.
        if process_alive(pid_max - 1) {
            return;
        }
        assert!(!process_alive(pid_max - 1));
    }

    #[test]
    fn cgroup_group_new_empty() {
        let cg = crate::cgroup::CgroupManager::new("/nonexistent");
        let group = CgroupGroup::new(&cg);
        assert!(group.names().is_empty());
    }

    // -- flag decl_by_name tests --

    #[test]
    fn decl_by_name_valid() {
        for &name in flags::ALL {
            assert!(flags::decl_by_name(name).is_some(), "should find {name}");
        }
    }

    #[test]
    fn decl_by_name_unknown() {
        assert!(flags::decl_by_name("nonexistent").is_none());
    }

    #[test]
    fn decl_by_name_steal_requires_llc() {
        let steal = flags::decl_by_name("steal").unwrap();
        assert_eq!(steal.requires.len(), 1);
        assert_eq!(steal.requires[0].name, "llc");
    }

    #[test]
    fn decl_by_name_borrow_no_requires() {
        let borrow = flags::decl_by_name("borrow").unwrap();
        assert!(borrow.requires.is_empty());
    }

    // -- flag_requires tests --

    #[test]
    fn flag_requires_steal_returns_llc() {
        let req = flag_requires("steal");
        assert_eq!(req, vec!["llc"]);
    }

    #[test]
    fn flag_requires_borrow_returns_empty() {
        assert!(flag_requires("borrow").is_empty());
    }

    #[test]
    fn flag_requires_unknown_returns_empty() {
        assert!(flag_requires("nonexistent").is_empty());
    }

    // -- FlagProfile name tests --

    #[test]
    fn profile_name_three_flags() {
        let p = FlagProfile {
            flags: vec![flags::LLC, flags::BORROW, flags::REBAL],
        };
        assert_eq!(p.name(), "llc+borrow+rebal");
    }

    // -- resolve_cpusets edge cases --

    #[test]
    fn resolve_cpusets_split_misaligned_single_llc() {
        let t = crate::topology::TestTopology::synthetic(8, 1);
        let r = resolve_cpusets(&CpusetMode::SplitMisaligned, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert!(total > 0);
    }

    #[test]
    fn resolve_cpusets_uneven_small_frac() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::Uneven(0.1), 2, &t).unwrap();
        assert!(
            r[0].len() < r[1].len(),
            "0.1 fraction should give smaller first set"
        );
    }

    // -- scenario profiles edge cases --

    #[test]
    fn scenario_profiles_count_bounded() {
        for s in &all_scenarios() {
            let n = s.profiles().len();
            // Each scenario should have at least 1 profile and at most 48 (all flag combos)
            assert!(n >= 1, "{} has {} profiles", s.name, n);
            assert!(n <= 48, "{} has {} profiles (>48)", s.name, n);
        }
    }

    // -- ALL_DECLS tests --

    #[test]
    fn all_decls_matches_all_strings() {
        assert_eq!(flags::ALL_DECLS.len(), flags::ALL.len());
        for (decl, &name) in flags::ALL_DECLS.iter().zip(flags::ALL.iter()) {
            assert_eq!(decl.name, name);
        }
    }

    // -- resolve_affinity_kind edge cases --

    #[test]
    fn resolve_affinity_single_cpu_wraps() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        // cgroup_idx=5 with 4 CPUs should wrap via modulo
        match resolve_affinity_kind(&AffinityKind::SingleCpu, None, 5, &t) {
            AffinityMode::SingleCpu(c) => assert_eq!(c, 1), // 5 % 4 = 1
            other => panic!("expected SingleCpu, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned_wraps() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // cgroup_idx=3 with 2 LLCs should wrap via modulo
        match resolve_affinity_kind(&AffinityKind::LlcAligned, None, 3, &t) {
            AffinityMode::Fixed(cpus) => {
                // 3 % 2 = 1 -> LLC 1
                assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect());
            }
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    // -- qualified_name computed values --

    #[test]
    fn qualified_name_all_scenarios_with_default() {
        let p = FlagProfile { flags: vec![] };
        for s in &all_scenarios() {
            // Every scenario produces "{name}/default" with the default profile.
            assert_eq!(
                s.qualified_name(&p),
                format!("{}/default", s.name),
                "qualified_name mismatch for {}",
                s.name
            );
        }
    }

    #[test]
    fn qualified_name_single_flag() {
        let s = &all_scenarios()[0];
        let p = FlagProfile {
            flags: vec![flags::REBAL],
        };
        assert_eq!(s.qualified_name(&p), format!("{}/rebal", s.name));
    }

    #[test]
    fn qualified_name_three_flags_joined() {
        let s = &all_scenarios()[0];
        let p = FlagProfile {
            flags: vec![flags::LLC, flags::STEAL, flags::BORROW],
        };
        // FlagProfile.name() joins with "+".
        assert_eq!(s.qualified_name(&p), format!("{}/llc+steal+borrow", s.name));
    }

    // -- generate_profiles computed counts --

    #[test]
    fn generate_profiles_steal_only_forces_llc() {
        // steal requires llc. Profiles with steal must also contain llc.
        let profiles = generate_profiles(&[flags::STEAL], &[]);
        assert!(!profiles.is_empty());
        for p in &profiles {
            assert!(
                p.flags.contains(&flags::STEAL),
                "steal missing: {:?}",
                p.flags
            );
            assert!(
                p.flags.contains(&flags::LLC),
                "llc missing when steal present: {:?}",
                p.flags
            );
        }
    }

    #[test]
    fn generate_profiles_all_excluded_returns_single_empty() {
        // Exclude everything -> only the empty profile (no flags) remains.
        let profiles = generate_profiles(&[], flags::ALL);
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0].flags.is_empty());
    }

    #[test]
    fn generate_profiles_required_and_excluded_all_others() {
        // Require borrow, exclude everything else -> exactly 1 profile: [borrow].
        let excluded: Vec<&str> = flags::ALL
            .iter()
            .copied()
            .filter(|f| *f != flags::BORROW)
            .collect();
        let profiles = generate_profiles(&[flags::BORROW], &excluded);
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].flags, vec![flags::BORROW]);
    }

    // -- resolve_cpusets computed values --

    #[test]
    fn resolve_cpusets_split_half_exact_cpu_assignment() {
        // synthetic(8, 2) -> all_cpus=[0..7], usable=[0..6] (7 reserved).
        // SplitHalf: mid=3, first=[0,1,2], second=[3,4,5,6].
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::SplitHalf, 2, &t).unwrap();
        assert_eq!(r[0], [0, 1, 2].into_iter().collect());
        assert_eq!(r[1], [3, 4, 5, 6].into_iter().collect());
    }

    #[test]
    fn resolve_cpusets_uneven_75_exact_split() {
        // synthetic(8, 2) -> usable=[0..6] (7 CPUs).
        // Uneven(0.75): split = floor(7 * 0.75) = 5.
        // first=[0,1,2,3,4], second=[5,6].
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::Uneven(0.75), 2, &t).unwrap();
        assert_eq!(r[0].len(), 5);
        assert_eq!(r[1].len(), 2);
        assert_eq!(r[0], [0, 1, 2, 3, 4].into_iter().collect());
        assert_eq!(r[1], [5, 6].into_iter().collect());
    }

    #[test]
    fn resolve_cpusets_holdback_50_exact() {
        // synthetic(8, 2) -> all_cpus=[0..7] (8 CPUs).
        // Holdback(0.5): keep = 8 - floor(8*0.5) = 4. mid=2.
        // first=[0,1], second=[2,3].
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetMode::Holdback(0.5), 2, &t).unwrap();
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert_eq!(total, 4, "holdback 50% of 8 should keep 4");
        assert_eq!(r[0], [0, 1].into_iter().collect());
        assert_eq!(r[1], [2, 3].into_iter().collect());
    }
}
