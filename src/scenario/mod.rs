//! Scenario definitions, flag system, and test execution.
//!
//! Most tests use the declarative ops API from the [`ops`] submodule:
//! - [`ops::CgroupDef`] -- declarative cgroup definition (name + cpuset + workload)
//! - [`ops::Step`] -- a sequence of ops followed by a hold period
//! - [`ops::Op`] -- atomic cgroup topology operation
//! - [`ops::CpusetSpec`] -- how to compute a cpuset from topology
//! - [`ops::HoldSpec`] -- how long to hold after a step
//! - [`ops::execute_defs`] -- run cgroup definitions for the full duration
//! - [`ops::execute_steps`] -- run a multi-step sequence
//!
//! Types defined in this module:
//! - [`Ctx`] -- runtime context passed to scenario functions
//! - [`CgroupGroup`] -- RAII guard that removes cgroups on drop
//! - [`flags::FlagDecl`] -- typed flag declaration with dependencies
//!
//! The [`scenarios`] submodule provides curated canned scenarios.
//!
//! For data-driven test cases (used by the internal catalog), see
//! [`Scenario`], [`CpusetPartition`], and [`Action`].
//!
//! See the [Scenarios](https://likewhatevs.github.io/ktstr/guide/concepts/scenarios.html)
//! and [Writing Tests](https://likewhatevs.github.io/ktstr/guide/writing-tests.html)
//! chapters of the guide.

pub mod affinity;
pub mod backdrop;
pub mod basic;
mod catalog;
pub mod cpuset;
pub mod dynamic;
pub mod interaction;
pub mod nested;
pub mod ops;
pub mod payload_run;
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
use crate::topology::TestTopology;
use crate::workload::*;

/// Check if a process is alive via kill(pid, 0).
///
/// Returns `false` for pid 0: `kill(0, ...)` targets the caller's
/// process group rather than a single process, so the syscall would
/// always report success and falsely mark "no process" as alive.
///
/// Returns `false` for `pid <= 0`. Non-positive pid_t values are
/// invalid targets — `kill(0, ...)` signals the caller's process
/// group and `kill(-1, ...)` signals every process the caller is
/// permitted to signal. Neither matches "is this specific process
/// alive?", so we refuse rather than probe.
fn process_alive(pid: libc::pid_t) -> bool {
    if pid <= 0 {
        return false;
    }
    kill(Pid::from_raw(pid), None).is_ok()
}

pub(crate) use crate::read_kmsg;

// ---------------------------------------------------------------------------
// Flag system
// ---------------------------------------------------------------------------

/// Flag declarations and name constants.
///
/// The built-in `*_DECL` constants (`LLC_DECL`, `BORROW_DECL`, etc.)
/// have empty `args` fields. They define flag names and dependencies
/// for ktstr's internal scenario catalog. External consumers define
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
    /// use ktstr::prelude::*;
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

    /// All flag declarations in canonical order. Single source of truth
    /// for the flag set -- adding a [`FlagDecl`] here automatically
    /// extends every consumer (`--flags` parsing, profile generation,
    /// drift tests, docs) because [`ALL`] and the short-name constants
    /// project `ALL_DECLS[i].name`.
    pub static ALL_DECLS: &[&FlagDecl] = &[
        &LLC_DECL,
        &BORROW_DECL,
        &STEAL_DECL,
        &REBAL_DECL,
        &REJECT_PIN_DECL,
        &NO_CTRL_DECL,
    ];

    /// Number of declared flags. Compile-time bound on [`ALL`].
    pub const N_FLAGS: usize = 6;
    const _: () = assert!(
        ALL_DECLS.len() == N_FLAGS,
        "N_FLAGS must equal ALL_DECLS.len(); update both together",
    );

    // Short-name constants projected from `ALL_DECLS`. Keeping these as
    // named `const`s preserves ergonomic `flags::LLC` references without
    // re-spelling the literal -- any drift vs. `ALL_DECLS` fails to compile.
    pub const LLC: &str = ALL_DECLS[0].name;
    pub const BORROW: &str = ALL_DECLS[1].name;
    pub const STEAL: &str = ALL_DECLS[2].name;
    pub const REBAL: &str = ALL_DECLS[3].name;
    pub const REJECT_PIN: &str = ALL_DECLS[4].name;
    pub const NO_CTRL: &str = ALL_DECLS[5].name;

    // Enforce positional `ALL_DECLS[i].name` invariant at build time. Any
    // reorder, insertion, or rename of the decls that does not also update
    // the corresponding short-name constant above triggers a compile error
    // here, preventing silent downstream breakage.
    const _: () = {
        // `const fn` cannot call `str::eq` directly, so compare via byte
        // slices with a small helper.
        const fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
            if a.len() != b.len() {
                return false;
            }
            let mut i = 0;
            while i < a.len() {
                if a[i] != b[i] {
                    return false;
                }
                i += 1;
            }
            true
        }
        assert!(
            bytes_eq(LLC.as_bytes(), b"llc"),
            "ALL_DECLS[0] must be `llc`"
        );
        assert!(
            bytes_eq(BORROW.as_bytes(), b"borrow"),
            "ALL_DECLS[1] must be `borrow`"
        );
        assert!(
            bytes_eq(STEAL.as_bytes(), b"steal"),
            "ALL_DECLS[2] must be `steal`"
        );
        assert!(
            bytes_eq(REBAL.as_bytes(), b"rebal"),
            "ALL_DECLS[3] must be `rebal`"
        );
        assert!(
            bytes_eq(REJECT_PIN.as_bytes(), b"reject-pin"),
            "ALL_DECLS[4] must be `reject-pin`"
        );
        assert!(
            bytes_eq(NO_CTRL.as_bytes(), b"no-ctrl"),
            "ALL_DECLS[5] must be `no-ctrl`"
        );
    };

    const fn build_all() -> [&'static str; N_FLAGS] {
        let mut out = [""; N_FLAGS];
        let mut i = 0;
        while i < N_FLAGS {
            out[i] = ALL_DECLS[i].name;
            i += 1;
        }
        out
    }

    /// Canonical flag-name list, projected from [`ALL_DECLS`] at compile
    /// time. Consumers iterate `ALL` for name-only use; `ALL_DECLS`
    /// for dependency-aware use.
    pub static ALL: &[&str] = &build_all();

    /// Look up a canonical flag name by its short name string.
    pub fn from_short_name(s: &str) -> Option<&'static str> {
        ALL.iter().find(|&&f| f == s).copied()
    }

    /// Look up a FlagDecl by name.
    pub fn decl_by_name(name: &str) -> Option<&'static FlagDecl> {
        ALL_DECLS.iter().find(|d| d.name == name).copied()
    }

    /// JSON-serializable representation of a [`FlagDecl`].
    ///
    /// Flattens `requires` from `&[&FlagDecl]` to `Vec<String>` (flag
    /// names) so it can be serialized/deserialized without static
    /// references. Used by `--ktstr-list-flags` output and the
    /// `cargo ktstr verifier --all-profiles` parser.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct FlagDeclJson {
        pub name: String,
        pub args: Vec<String>,
        pub requires: Vec<String>,
    }

    impl FlagDeclJson {
        /// Convert a static [`FlagDecl`] to its JSON-serializable form.
        pub fn from_decl(decl: &FlagDecl) -> Self {
            Self {
                name: decl.name.to_string(),
                args: decl.args.iter().map(|s| s.to_string()).collect(),
                requires: decl.requires.iter().map(|r| r.name.to_string()).collect(),
            }
        }
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

// Re-export AffinityKind from workload so existing `use super::*` in
// submodules (catalog.rs, affinity.rs, etc.) can find it.
pub use crate::workload::AffinityKind;

/// Look up flag dependencies via FlagDecl.requires.
fn flag_requires(flag: &str) -> Vec<&'static str> {
    flags::decl_by_name(flag)
        .map(|d| d.requires.iter().map(|r| r.name).collect())
        .unwrap_or_default()
}

/// Enumerate flag profiles from a flag-name universe, a
/// requires-edge resolver, and required/excluded constraints.
///
/// Shared with [`Scheduler::generate_profiles`](crate::test_support::Scheduler::generate_profiles)
/// and the `cargo ktstr verifier --all-profiles` generator — each
/// previously hand-rolled the same power-set + requires-filter +
/// sort logic against a different flag-source type. `all_names` is
/// the canonical flag ordering (profiles are sorted in this order);
/// `requires_fn(f)` returns the flags that must also be present
/// when `f` is. `required` and `excluded` filter the optional set.
///
/// Caller maps the result back to its own profile type — the return
/// `Vec<Vec<T>>` keeps the generic free of either `FlagProfile` or
/// `(name, Vec<String>)` plumbing.
pub fn compute_flag_profiles<T, F>(
    all_names: &[T],
    requires_fn: F,
    required: &[T],
    excluded: &[T],
) -> Vec<Vec<T>>
where
    T: Clone + PartialEq,
    F: Fn(&T) -> Vec<T>,
{
    let optional: Vec<T> = all_names
        .iter()
        .filter(|f| !required.contains(f) && !excluded.contains(f))
        .cloned()
        .collect();
    debug_assert!(
        optional.len() < 32,
        "compute_flag_profiles: {} optional flags would overflow u32 power-set mask",
        optional.len(),
    );
    let mut out = Vec::new();
    for mask in 0..(1u32 << optional.len()) {
        let mut fl: Vec<T> = required.to_vec();
        for (i, f) in optional.iter().enumerate() {
            if mask & (1 << i) != 0 {
                fl.push(f.clone());
            }
        }
        let valid = fl
            .iter()
            .all(|f| requires_fn(f).iter().all(|r| fl.contains(r)));
        if valid {
            fl.sort_by_key(|f| all_names.iter().position(|a| a == f).unwrap_or(usize::MAX));
            out.push(fl);
        }
    }
    out
}

fn generate_profiles(required: &[&'static str], excluded: &[&'static str]) -> Vec<FlagProfile> {
    compute_flag_profiles(flags::ALL, |&f| flag_requires(f), required, excluded)
        .into_iter()
        .map(|flags| FlagProfile { flags })
        .collect()
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
/// cpuset for one cgroup in the ops/steps system. `CpusetPartition`
/// partitions at the scenario level; `CpusetSpec` specifies per-cgroup.
#[derive(Clone, Debug)]
pub enum CpusetPartition {
    /// Each cgroup sees the full usable CPU set; no partitioning.
    None,
    /// Each cgroup is pinned to the CPUs of one full LLC, assigned in
    /// order. Requires at least as many LLCs as cgroups.
    LlcAligned,
    /// Partition `usable_cpus()` into contiguous even halves (or
    /// as-even-as-possible groups) per cgroup.
    SplitHalf,
    /// Like `SplitHalf` but chooses boundaries that cross LLC lines
    /// to stress cross-LLC placement.
    SplitMisaligned,
    /// Each cgroup gets the full set overlapped by the given fraction
    /// with its neighbour; `0.0` behaves like `SplitHalf`, `1.0` like
    /// `None`.
    Overlap(f64),
    /// Asymmetric split where cgroup 0 receives this fraction of the
    /// usable CPUs and the remaining cgroups share the rest.
    Uneven(f64),
    /// Reserve this fraction of CPUs outside the scenario, then split
    /// the rest evenly across cgroups.
    Holdback(f64),
}

/// Callable body for [`Action::Custom`].
///
/// Wrapped in [`std::sync::Arc`] so [`Action`] (and the enclosing
/// [`Scenario`]) stays cheaply [`Clone`] while allowing closures that
/// capture state. Bare `fn` pointers forced callers to hoist state
/// into static or thread-local slots; `Arc<dyn Fn>` keeps captures
/// where the scenario is built. The `Send + Sync` bounds mirror
/// [`Scenario`]'s — catalog entries are shared across the worker
/// dispatch pool.
pub type CustomFn = std::sync::Arc<dyn Fn(&Ctx) -> Result<AssertResult> + Send + Sync>;

/// What happens during a scenario's workload phase.
///
/// `Steady` runs workers for the configured duration with no dynamic
/// operations. `Custom` wraps a caller-supplied closure (see
/// [`Action::custom`] for an example with captured state).
#[derive(Clone)]
pub enum Action {
    /// Run workers for the configured duration with no dynamic operations.
    Steady,
    /// Execute custom scenario logic with access to the full test context.
    Custom(CustomFn),
}

impl Action {
    /// Build a [`Custom`](Self::Custom) from any closure. Wraps the
    /// body in the shared [`CustomFn`] `Arc` so callers avoid spelling
    /// out `Arc::new(...)` at every catalog entry.
    ///
    /// The closure may capture state. Captured values must be
    /// `Send + Sync + 'static` so the closure can be shared across
    /// threads. The resulting [`Action`] is cheaply [`Clone`] via
    /// `Arc` reference counting.
    ///
    /// # Example
    /// ```
    /// use ktstr::scenario::Action;
    /// use ktstr::assert::AssertResult;
    /// use std::sync::atomic::{AtomicU32, Ordering};
    /// use std::sync::Arc;
    ///
    /// // Capture a shared counter that the scenario increments per run.
    /// let counter = Arc::new(AtomicU32::new(0));
    /// let counter_ref = counter.clone();
    /// let _action = Action::custom(move |_ctx| {
    ///     counter_ref.fetch_add(1, Ordering::Relaxed);
    ///     Ok(AssertResult::pass())
    /// });
    /// ```
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(&Ctx) -> Result<AssertResult> + Send + Sync + 'static,
    {
        Action::Custom(std::sync::Arc::new(f))
    }
}

impl std::fmt::Debug for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Steady => write!(f, "Steady"),
            Action::Custom(_) => write!(f, "Custom(<closure>)"),
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
/// # use ktstr::scenario::all_scenarios;
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
    pub cpuset_partition: CpusetPartition,
    /// Per-cgroup workload definitions.
    pub cgroup_works: Vec<Work>,
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
#[must_use = "dropping a CgroupGroup immediately destroys the cgroups it manages"]
pub struct CgroupGroup<'a> {
    cgroups: &'a dyn crate::cgroup::CgroupOps,
    names: Vec<String>,
}

impl std::fmt::Debug for CgroupGroup<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CgroupGroup")
            .field("cgroups", &self.cgroups.parent_path())
            .field("names", &self.names)
            .finish()
    }
}

impl<'a> CgroupGroup<'a> {
    /// Create an empty group. Cgroups added via `add_cgroup` or
    /// `add_cgroup_no_cpuset` are removed when the group is dropped.
    pub fn new(cgroups: &'a dyn crate::cgroup::CgroupOps) -> Self {
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
        // Reverse-iterate so nested cgroups (children created AFTER
        // their parents) are removed before their parents. Removing a
        // cgroup directory that still has child cgroup directories
        // under it fails with ENOTEMPTY.
        for name in self.names.iter().rev() {
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
pub struct Ctx<'a> {
    /// Cgroup filesystem operations. `&dyn CgroupOps` (not `&CgroupManager`)
    /// so scenario code can be driven by an in-memory test double without
    /// touching `/sys/fs/cgroup`. Production callers pass
    /// `&CgroupManager` and the auto-coercion is transparent at the call
    /// site — `ctx.cgroups.set_cpuset(...)` works unchanged.
    pub cgroups: &'a dyn crate::cgroup::CgroupOps,
    /// VM CPU topology.
    pub topo: &'a TestTopology,
    /// How long to run the workload.
    pub duration: Duration,
    /// Default number of workers per cgroup.
    pub workers_per_cgroup: usize,
    /// PID of the running scheduler (for liveness checks). Stored as
    /// `pid_t` to match the kernel's native type — avoids u32→i32
    /// sign-cast wraparound at the `kill`/`process_alive` boundary.
    pub sched_pid: libc::pid_t,
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
    /// BPF map write is complete. Set automatically by the framework
    /// when a `KtstrTestEntry` declares `bpf_map_write`; custom
    /// scenarios typically do not flip this manually.
    pub wait_for_map_write: bool,
}

impl std::fmt::Debug for Ctx<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `&dyn CgroupOps` is not Debug (dropped the supertrait to
        // avoid bloating the test-double surface); render the parent
        // path instead so debug prints are still informative.
        f.debug_struct("Ctx")
            .field("cgroups", &self.cgroups.parent_path())
            .field("topo", &self.topo)
            .field("duration", &self.duration)
            .field("workers_per_cgroup", &self.workers_per_cgroup)
            .field("sched_pid", &self.sched_pid)
            .field("settle", &self.settle)
            .field("work_type_override", &self.work_type_override)
            .field("assert", &self.assert)
            .field("wait_for_map_write", &self.wait_for_map_write)
            .finish()
    }
}

/// Fluent builder for [`Ctx`].
///
/// Scenario unit tests reach for a [`Ctx`] with sane defaults so they
/// can exercise scenario logic without booting a VM. The direct
/// struct-literal construction at ~14 call sites forces every test to
/// repeat the full 9-field init and keeps diverging defaults in sync
/// by hand; this builder centralises those defaults and keeps required
/// fields (borrowed `cgroups`/`topo`) in their types.
///
/// Defaults:
/// - `duration`: 1 s — matches the `scenario::basic` test helper
///   (`scenario::stress` uses 2 s and sets it explicitly)
/// - `workers_per_cgroup`: 1
/// - `sched_pid`: 0 — matches `Ctx` consumers that treat 0 as
///   "no scheduler attached"; [`run_scenario`] uses this to short-circuit
///   liveness checks via [`crate::workload::set_sched_pid`].
/// - `settle`: 0 ms — tests do not need to wait for scheduler stabilisation
/// - `work_type_override`: `None`
/// - `assert`: [`crate::assert::Assert::default_checks()`] —
///   the same policy production paths merge through
/// - `wait_for_map_write`: `false`
///
/// Override any default via the corresponding method, then materialise
/// the context with [`CtxBuilder::build`].
///
/// # Example
/// ```ignore
/// let cgroups = CgroupManager::new("/nonexistent");
/// let topo = TestTopology::synthetic(4, 1);
/// let ctx = Ctx::builder(&cgroups, &topo)
///     .workers_per_cgroup(3)
///     .duration(Duration::from_secs(2))
///     .build();
/// ```
pub struct CtxBuilder<'a> {
    cgroups: &'a dyn crate::cgroup::CgroupOps,
    topo: &'a TestTopology,
    duration: Duration,
    workers_per_cgroup: usize,
    sched_pid: libc::pid_t,
    settle: Duration,
    work_type_override: Option<WorkType>,
    assert: crate::assert::Assert,
    wait_for_map_write: bool,
}

impl<'a> CtxBuilder<'a> {
    /// Wall-clock budget for the workload phase of the scenario.
    pub fn duration(mut self, d: Duration) -> Self {
        self.duration = d;
        self
    }

    /// Number of worker threads started per cgroup by the default workload.
    pub fn workers_per_cgroup(mut self, n: usize) -> Self {
        self.workers_per_cgroup = n;
        self
    }

    /// PID of the scheduler process; `0` means "no scheduler attached"
    /// and disables the liveness checks in [`run_scenario`].
    pub fn sched_pid(mut self, pid: libc::pid_t) -> Self {
        self.sched_pid = pid;
        self
    }

    /// Time to wait after cgroup creation for scheduler stabilisation.
    pub fn settle(mut self, s: Duration) -> Self {
        self.settle = s;
        self
    }

    /// Override the default work type for scenarios that would
    /// otherwise use `CpuSpin`.
    pub fn work_type_override(mut self, wt: Option<WorkType>) -> Self {
        self.work_type_override = wt;
        self
    }

    /// Merged assertion config. Callers that want the production
    /// layering should pass `Assert::default_checks().merge(&...)`;
    /// tests that pin a specific policy can pass
    /// [`crate::assert::Assert::NO_OVERRIDES`] directly.
    pub fn assert(mut self, a: crate::assert::Assert) -> Self {
        self.assert = a;
        self
    }

    /// When true, `execute_steps` polls the SHM signal slot after
    /// writing the scenario start marker. See the field doc on
    /// [`Ctx::wait_for_map_write`].
    pub fn wait_for_map_write(mut self, v: bool) -> Self {
        self.wait_for_map_write = v;
        self
    }

    /// Materialise the configured [`Ctx`].
    pub fn build(self) -> Ctx<'a> {
        Ctx {
            cgroups: self.cgroups,
            topo: self.topo,
            duration: self.duration,
            workers_per_cgroup: self.workers_per_cgroup,
            sched_pid: self.sched_pid,
            settle: self.settle,
            work_type_override: self.work_type_override,
            assert: self.assert,
            wait_for_map_write: self.wait_for_map_write,
        }
    }
}

impl<'a> Ctx<'a> {
    /// Start a new [`CtxBuilder`] with required `cgroups` and `topo`
    /// borrows and sane defaults for every other field. See
    /// [`CtxBuilder`] for the full default set.
    pub fn builder(
        cgroups: &'a dyn crate::cgroup::CgroupOps,
        topo: &'a TestTopology,
    ) -> CtxBuilder<'a> {
        CtxBuilder {
            cgroups,
            topo,
            duration: Duration::from_secs(1),
            workers_per_cgroup: 1,
            sched_pid: 0,
            settle: Duration::from_millis(0),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    /// Start a [`PayloadRun`](crate::scenario::payload_run::PayloadRun)
    /// builder for the given [`Payload`](crate::test_support::Payload).
    ///
    /// The builder inherits `payload.default_args` and
    /// `payload.default_checks`; chained `.arg(...)` / `.check(...)`
    /// calls extend them; `.clear_args()` / `.clear_checks()` wipe
    /// both defaults and prior appends. Terminal `.run()` blocks and
    /// returns `Result<(AssertResult, PayloadMetrics)>`.
    ///
    /// Only `PayloadKind::Binary` payloads are runnable here;
    /// `.run()` on a `PayloadKind::Scheduler` payload returns `Err`.
    pub fn payload(
        &'a self,
        p: &'static crate::test_support::Payload,
    ) -> crate::scenario::payload_run::PayloadRun<'a> {
        crate::scenario::payload_run::PayloadRun::new(self, p)
    }
}

/// Run a scenario and return its assertion result.
///
/// Skips early (returning `AssertResult::skip`) when the scenario's
/// requested cpuset partitioning produces an empty cpuset for any
/// cgroup under the current topology. Polls scheduler liveness at
/// 500ms intervals. If the scheduler exits after cgroup creation but
/// before the workload starts, returns an `Err` so callers can treat
/// the run as a setup failure. A scheduler death mid-workload is
/// reported as a completed-but-failed `AssertResult` with
/// "scheduler crashed during workload" in `details`, not an error.
/// On workload failure, captures the guest kernel console via
/// `read_kmsg` so diagnostics include the stall or crash context.
pub fn run_scenario(scenario: &Scenario, ctx: &Ctx) -> Result<AssertResult> {
    tracing::info!(scenario = scenario.name, "running");
    if let Action::Custom(f) = &scenario.action {
        return f(ctx);
    }

    let cpusets = resolve_cpusets(&scenario.cpuset_partition, scenario.num_cgroups, ctx.topo);

    // Skip if topology doesn't support the test
    if let Some(ref cs) = cpusets
        && cs.iter().any(|s| s.is_empty())
    {
        return Ok(AssertResult::skip("skipped: not enough CPUs/LLCs"));
    }

    let scenario_start = std::time::Instant::now();

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
        if let Err(reason) = cw.mem_policy.validate() {
            anyhow::bail!("cgroup '{}': {}", name, reason);
        }
        let n = resolve_num_workers(&cw, ctx.workers_per_cgroup, name)?;
        let cpuset = cpusets.as_deref().and_then(|cs| cs.get(i));
        if let Some(cs) = cpusets.as_deref()
            && i >= cs.len()
        {
            // Panic in debug builds to surface caller bugs early;
            // release builds fall through to the warn + fallback below.
            debug_assert!(
                i < cs.len(),
                "cgroup_idx {i} out of range for cpusets of len {}",
                cs.len(),
            );
            tracing::warn!(
                cgroup_idx = i,
                cpusets_len = cs.len(),
                "cgroup index out of range for cpusets array; falling back to unrestricted pool"
            );
        }
        let affinity = resolve_affinity_for_cgroup(&cw.affinity, cpuset, ctx.topo);
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
            sched_policy: cw.sched_policy,
            mem_policy: cw.mem_policy.clone(),
            mpol_flags: cw.mpol_flags,
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
        let numa_nodes = cs.map(|c| ctx.topo.numa_nodes_for_cpuset(c));
        result.merge(
            ctx.assert
                .assert_cgroup_with_numa(&reports, cs, numa_nodes.as_ref()),
        );
    }

    // Capture kernel log on failure
    if !result.passed {
        for line in read_kmsg().lines() {
            result.details.push(line.to_string().into());
        }
    }

    if sched_dead {
        result.passed = false;
        result.details.push(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Monitor,
            format!(
                "scheduler crashed during workload ({:.1}s into test)",
                scenario_start.elapsed().as_secs_f64(),
            ),
        ));
    }

    Ok(result)
}

fn resolve_cpusets(
    mode: &CpusetPartition,
    n: usize,
    topo: &TestTopology,
) -> Option<Vec<BTreeSet<usize>>> {
    let all = topo.all_cpus();
    let usable = topo.usable_cpus();
    match mode {
        CpusetPartition::None => None,
        CpusetPartition::LlcAligned => {
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
        CpusetPartition::SplitHalf => {
            let mid = usable.len() / 2;
            Some(vec![
                usable[..mid].iter().copied().collect(),
                usable[mid..].iter().copied().collect(),
            ])
        }
        CpusetPartition::SplitMisaligned => {
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
        CpusetPartition::Overlap(frac) => Some(topo.overlapping_cpusets(n, *frac)),
        CpusetPartition::Uneven(frac) => {
            let split = (usable.len() as f64 * frac) as usize;
            Some(vec![
                usable[..split.max(1)].iter().copied().collect(),
                usable[split.max(1)..].iter().copied().collect(),
            ])
        }
        CpusetPartition::Holdback(frac) => {
            let keep = all.len() - (all.len() as f64 * frac) as usize;
            let mid = keep / 2;
            Some(vec![
                all[..mid.max(1)].iter().copied().collect(),
                all[mid.max(1)..keep].iter().copied().collect(),
            ])
        }
    }
}

/// Resolve an [`AffinityKind`] to a concrete [`AffinityMode`] for workers
/// in a cgroup with the given effective cpuset.
///
/// When a cpuset is active, affinity masks are intersected with it so the
/// effective `sched_setaffinity` mask matches what the kernel will enforce.
/// Without a cpuset, the full topology is used.
/// Resolve a [`Work`]'s `num_workers`, falling back to `default_n` when unset,
/// and reject `num_workers=0`.
///
/// A cgroup with no workers emits no `WorkerReport`s, so every downstream
/// assertion vacuously passes. Callers that want "no load" on a cgroup
/// should either drop the `Work` entry entirely (letting the default apply)
/// or use a single sentinel worker so assertions have something to check.
pub(crate) fn resolve_num_workers(work: &Work, default_n: usize, label: &str) -> Result<usize> {
    let n = work.num_workers.unwrap_or(default_n);
    if n == 0 {
        anyhow::bail!(
            "cgroup '{}': num_workers=0 is not allowed — assertions would \
             vacuously pass with no WorkerReports; use at least 1 worker or \
             drop this Work entry",
            label,
        );
    }
    Ok(n)
}

pub fn resolve_affinity_for_cgroup(
    kind: &AffinityKind,
    cpuset: Option<&BTreeSet<usize>>,
    topo: &TestTopology,
) -> AffinityMode {
    match kind {
        AffinityKind::Inherit => AffinityMode::None,
        AffinityKind::RandomSubset => {
            let pool = cpuset.cloned().unwrap_or_else(|| topo.all_cpuset());
            if pool.is_empty() {
                tracing::debug!(
                    "RandomSubset: empty cpuset and empty topology pool, \
                     falling back to AffinityMode::None"
                );
                AffinityMode::None
            } else {
                let count = (pool.len() / 2).max(1);
                AffinityMode::Random { from: pool, count }
            }
        }
        AffinityKind::LlcAligned => {
            let pool = cpuset.cloned().unwrap_or_else(|| topo.all_cpuset());
            // Find the LLC that has the most overlap with the cpuset.
            let mut best_llc = topo.llc_aligned_cpuset(0);
            let mut best_overlap = best_llc.intersection(&pool).count();
            for idx in 1..topo.num_llcs() {
                let llc = topo.llc_aligned_cpuset(idx);
                let overlap = llc.intersection(&pool).count();
                if overlap > best_overlap {
                    best_llc = llc;
                    best_overlap = overlap;
                }
            }
            // Intersect with cpuset so effective affinity matches kernel behavior.
            let effective: BTreeSet<usize> = best_llc.intersection(&pool).copied().collect();
            if effective.is_empty() {
                // All LLC CPUs outside cpuset -- fall back to inheriting cpuset.
                AffinityMode::None
            } else {
                AffinityMode::Fixed(effective)
            }
        }
        AffinityKind::CrossCgroup => {
            // When a cpuset is active, crossing cgroup boundaries is the intent,
            // but the kernel will intersect. Use all CPUs -- the kernel enforces
            // the cpuset constraint.
            AffinityMode::Fixed(topo.all_cpuset())
        }
        AffinityKind::SingleCpu => {
            let pool = cpuset.cloned().unwrap_or_else(|| topo.all_cpuset());
            if let Some(&cpu) = pool.iter().next() {
                AffinityMode::SingleCpu(cpu)
            } else {
                AffinityMode::None
            }
        }
        AffinityKind::Exact(cpus) => {
            if let Some(cs) = cpuset {
                let effective: BTreeSet<usize> = cpus.intersection(cs).copied().collect();
                if effective.is_empty() {
                    AffinityMode::None
                } else {
                    AffinityMode::Fixed(effective)
                }
            } else {
                AffinityMode::Fixed(cpus.clone())
            }
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

/// Stop workers, collect reports, and merge assertion results.
///
/// Each item is a `(WorkloadHandle, Option<&BTreeSet<usize>>)` pair
/// where the optional cpuset is passed through to
/// [`Assert::assert_cgroup`](crate::assert::Assert::assert_cgroup)
/// for isolation checks. When `checks` has no worker-level checks,
/// falls back to [`assert_not_starved`](assert::assert_not_starved).
pub(crate) fn collect_handles<'a>(
    handles: impl IntoIterator<Item = (WorkloadHandle, Option<&'a BTreeSet<usize>>)>,
    checks: &crate::assert::Assert,
    topo: Option<&crate::topology::TestTopology>,
) -> AssertResult {
    let mut r = AssertResult::pass();
    for (h, cpuset) in handles {
        let reports = h.stop_and_collect();
        if checks.has_worker_checks() {
            let numa_nodes = cpuset.and_then(|cs| topo.map(|t| t.numa_nodes_for_cpuset(cs)));
            r.merge(checks.assert_cgroup_with_numa(&reports, cpuset, numa_nodes.as_ref()));
        } else {
            r.merge(assert::assert_not_starved(&reports));
        }
    }
    r
}

/// Stop all workers, collect reports, and run assertion checks.
///
/// Uses `checks` for worker evaluation. When the Assert has no
/// worker-level checks configured (all fields None), falls back
/// to `assert_not_starved`. Returns a merged [`AssertResult`]
/// across all workers.
pub fn collect_all(handles: Vec<WorkloadHandle>, checks: &crate::assert::Assert) -> AssertResult {
    collect_handles(handles.into_iter().map(|h| (h, None)), checks, None)
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

/// Spawn diverse workloads across N cgroups: CpuSpin, Bursty, IoSync,
/// Mixed, YieldHeavy. Each cgroup uses `ctx.workers_per_cgroup`
/// workers except IoSync cgroups, which always use 2 workers to
/// avoid drowning the scenario in blocking IO.
pub fn spawn_diverse(ctx: &Ctx, cgroup_names: &[&str]) -> Result<Vec<WorkloadHandle>> {
    let types = [
        WorkType::CpuSpin,
        WorkType::bursty(50, 100),
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
        assert!(resolve_cpusets(&CpusetPartition::None, 2, &t).is_none());
    }

    #[test]
    fn resolve_cpusets_split_half_covers_usable() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::SplitHalf, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        // Last CPU reserved for cgroup 0 → 7 usable
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn resolve_cpusets_llc_aligned() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::LlcAligned, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        // Both sets non-empty
        assert!(!r[0].is_empty());
        assert!(!r[1].is_empty());
    }

    #[test]
    fn resolve_cpusets_uneven() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::Uneven(0.75), 2, &t).unwrap();
        assert!(r[0].len() > r[1].len(), "75/25 split should be uneven");
    }

    #[test]
    fn resolve_cpusets_holdback() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::Holdback(0.5), 2, &t).unwrap();
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert!(total < 8, "holdback should use fewer CPUs");
    }

    #[test]
    fn resolve_cpusets_overlap() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::Overlap(0.5), 3, &t).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn resolve_affinity_inherit() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        assert!(matches!(
            resolve_affinity_for_cgroup(&AffinityKind::Inherit, None, &t),
            AffinityMode::None
        ));
    }

    #[test]
    fn resolve_affinity_single_cpu() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_for_cgroup(&AffinityKind::SingleCpu, None, &t) {
            AffinityMode::SingleCpu(c) => assert_eq!(c, 0),
            other => panic!("expected SingleCpu, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_cross_cgroup() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_for_cgroup(&AffinityKind::CrossCgroup, None, &t) {
            AffinityMode::Fixed(cpus) => assert_eq!(cpus.len(), 8),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // No cpuset: both LLCs cover the full pool equally. LLC 0
        // is found first with max overlap, so result is LLC 0 CPUs.
        match resolve_affinity_for_cgroup(&AffinityKind::LlcAligned, None, &t) {
            AffinityMode::Fixed(cpus) => assert_eq!(cpus, [0, 1, 2, 3].into_iter().collect()),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned_with_cpuset() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // Cpuset restricted to LLC 1 CPUs: LlcAligned picks LLC 1.
        let cpusets: Vec<BTreeSet<usize>> = vec![
            [0, 1, 2, 3].into_iter().collect(),
            [4, 5, 6, 7].into_iter().collect(),
        ];
        match resolve_affinity_for_cgroup(&AffinityKind::LlcAligned, cpusets.get(1), &t) {
            AffinityMode::Fixed(cpus) => assert_eq!(cpus, [4, 5, 6, 7].into_iter().collect()),
            other => panic!("expected Fixed, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_random_subset() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1, 2, 3].into_iter().collect()];
        match resolve_affinity_for_cgroup(&AffinityKind::RandomSubset, cpusets.first(), &t) {
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
        let r = resolve_cpusets(&CpusetPartition::SplitMisaligned, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert!(total > 0);
        // Misaligned means split within an LLC, not at LLC boundary
        assert_ne!(r[0].len(), 4, "misaligned should NOT split at LLC boundary");
    }

    #[test]
    fn resolve_cpusets_llc_aligned_single_llc() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let r = resolve_cpusets(&CpusetPartition::LlcAligned, 2, &t).unwrap();
        // With 1 LLC, can only make 1 set -> returns empty for missing
        assert!(
            r.iter().any(|s| s.is_empty()),
            "should signal skip with empty set"
        );
    }

    #[test]
    fn resolve_cpusets_small_topology() {
        let t = crate::topology::TestTopology::synthetic(2, 1);
        let r = resolve_cpusets(&CpusetPartition::SplitHalf, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        // 2 CPUs, no reserve (too small), each gets 1
        assert_eq!(r[0].len(), 1);
        assert_eq!(r[1].len(), 1);
    }

    #[test]
    fn cgroup_work_default() {
        let cw = Work::default();
        assert_eq!(cw.num_workers, None);
        assert!(matches!(cw.work_type, WorkType::CpuSpin));
        assert!(matches!(cw.sched_policy, SchedPolicy::Normal));
        assert!(matches!(cw.affinity, AffinityKind::Inherit));
        assert!(matches!(cw.mem_policy, MemPolicy::Default));
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
        let r = resolve_cpusets(&CpusetPartition::Holdback(0.33), 2, &t).unwrap();
        let total: usize = r.iter().map(|s| s.len()).sum();
        // 12 CPUs, holdback 33%: keep = 12 - floor(12*0.33) = 12 - 3 = 9
        assert_eq!(total, 9, "holdback 33% of 12 should keep 9");
        assert!(total < 12, "holdback should use fewer CPUs than total");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn resolve_cpusets_overlap_sets_overlap() {
        let t = crate::topology::TestTopology::synthetic(12, 1);
        let r = resolve_cpusets(&CpusetPartition::Overlap(0.5), 2, &t).unwrap();
        let overlap: BTreeSet<usize> = r[0].intersection(&r[1]).copied().collect();
        assert!(
            !overlap.is_empty(),
            "50% overlap should have overlapping CPUs"
        );
    }

    #[test]
    fn resolve_affinity_random_no_cpusets() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        match resolve_affinity_for_cgroup(&AffinityKind::RandomSubset, None, &t) {
            AffinityMode::Random { from, count } => {
                assert_eq!(from.len(), 8); // all CPUs
                assert_eq!(count, 4); // half
            }
            other => panic!("expected Random, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_random_subset_empty_pool_is_none() {
        // Regression: empty cpuset produced AffinityMode::Random { from: empty,
        // count: 1 }, which previously produced an empty affinity mask
        // rejected by sched_setaffinity with EINVAL. Must short-circuit
        // to AffinityMode::None here.
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let empty: BTreeSet<usize> = BTreeSet::new();
        match resolve_affinity_for_cgroup(&AffinityKind::RandomSubset, Some(&empty), &t) {
            AffinityMode::None => {}
            other => panic!("expected None for empty cpuset, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_oob_cgroup_idx_falls_back_to_unrestricted() {
        // The wrapper that bounds-checked cgroup_idx against cpusets
        // was inlined into run_scenario (see the tracing::warn in
        // run_scenario's affinity-resolve block). The underlying
        // `cpusets.get(idx)` returns None on OOB, and
        // resolve_affinity_for_cgroup's None arm delivers the
        // unrestricted fallback. This test pins the fallback contract.
        let t = crate::topology::TestTopology::synthetic(4, 1);
        let cpusets: Vec<BTreeSet<usize>> = vec![[0, 1].into_iter().collect()];
        let oob_idx = 5;
        // Mirror the inlined expression in run_scenario:
        let cpuset = cpusets.get(oob_idx);
        assert!(cpuset.is_none(), "OOB index must yield None cpuset");
        match resolve_affinity_for_cgroup(&AffinityKind::RandomSubset, cpuset, &t) {
            AffinityMode::Random { from, count } => {
                assert_eq!(from.len(), 4, "OOB idx falls back to full topology");
                assert_eq!(count, 2);
            }
            other => panic!("expected Random with full pool, got {:?}", other),
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
    fn process_alive_self_is_true() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        assert!(process_alive(pid));
    }

    #[test]
    fn process_alive_zero_is_false() {
        // kill(0, sig) targets the caller's process group, so without
        // an explicit guard kill(0, 0) succeeds and would falsely
        // report "process 0" as alive.
        assert!(!process_alive(0));
    }

    #[test]
    fn process_alive_negative_is_false() {
        // kill(negative, sig) targets a process group (or, for -1,
        // every process the caller can signal). A pid_t <= 0 is
        // never a live-process query and must return false.
        assert!(!process_alive(-1));
        assert!(!process_alive(libc::pid_t::MIN));
    }

    #[test]
    fn process_alive_nonexistent_pid() {
        // Fork a child that exits immediately, then waitpid to reap it.
        // After reap the PID is returned to the kernel and process_alive
        // must report false. Picking an arbitrary PID leaves a race
        // where that PID could be in use by another process on the host;
        // using a PID we just freed ourselves is deterministic.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed: {}", std::io::Error::last_os_error());
        if pid == 0 {
            unsafe { libc::_exit(0) };
        }
        let mut status: libc::c_int = 0;
        let waited = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(
            waited,
            pid,
            "waitpid failed: {}",
            std::io::Error::last_os_error()
        );
        assert!(!process_alive(pid));
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
        let r = resolve_cpusets(&CpusetPartition::SplitMisaligned, 2, &t).unwrap();
        assert_eq!(r.len(), 2);
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert!(total > 0);
    }

    #[test]
    fn resolve_cpusets_uneven_small_frac() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::Uneven(0.1), 2, &t).unwrap();
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

    // -- resolve_affinity_for_cgroup edge cases --

    #[test]
    fn resolve_affinity_single_cpu_with_cpuset() {
        let t = crate::topology::TestTopology::synthetic(4, 1);
        // Cpuset restricts to CPUs {2,3}: SingleCpu picks first in cpuset.
        let cpusets: Vec<BTreeSet<usize>> = vec![[2, 3].into_iter().collect()];
        match resolve_affinity_for_cgroup(&AffinityKind::SingleCpu, cpusets.first(), &t) {
            AffinityMode::SingleCpu(c) => assert_eq!(c, 2),
            other => panic!("expected SingleCpu, got {:?}", other),
        }
    }

    #[test]
    fn resolve_affinity_llc_aligned_picks_best_overlap() {
        let t = crate::topology::TestTopology::synthetic(8, 2);
        // Cpuset spans both LLCs but has more CPUs in LLC 1.
        // LLC 0 = {0,1,2,3}, LLC 1 = {4,5,6,7}.
        // Cpuset = {3, 4, 5, 6, 7}: LLC 1 has 4 CPUs in cpuset, LLC 0 has 1.
        let cpusets: Vec<BTreeSet<usize>> = vec![[3, 4, 5, 6, 7].into_iter().collect()];
        match resolve_affinity_for_cgroup(&AffinityKind::LlcAligned, cpusets.first(), &t) {
            AffinityMode::Fixed(cpus) => {
                // LLC 1 has best overlap; result is intersection {4,5,6,7}.
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
        let r = resolve_cpusets(&CpusetPartition::SplitHalf, 2, &t).unwrap();
        assert_eq!(r[0], [0, 1, 2].into_iter().collect());
        assert_eq!(r[1], [3, 4, 5, 6].into_iter().collect());
    }

    #[test]
    fn resolve_cpusets_uneven_75_exact_split() {
        // synthetic(8, 2) -> usable=[0..6] (7 CPUs).
        // Uneven(0.75): split = floor(7 * 0.75) = 5.
        // first=[0,1,2,3,4], second=[5,6].
        let t = crate::topology::TestTopology::synthetic(8, 2);
        let r = resolve_cpusets(&CpusetPartition::Uneven(0.75), 2, &t).unwrap();
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
        let r = resolve_cpusets(&CpusetPartition::Holdback(0.5), 2, &t).unwrap();
        let total: usize = r.iter().map(|s| s.len()).sum();
        assert_eq!(total, 4, "holdback 50% of 8 should keep 4");
        assert_eq!(r[0], [0, 1].into_iter().collect());
        assert_eq!(r[1], [2, 3].into_iter().collect());
    }

    #[test]
    fn resolve_num_workers_zero_rejected_with_label() {
        let w = Work {
            num_workers: Some(0),
            ..Default::default()
        };
        let err = resolve_num_workers(&w, 4, "victim").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cgroup 'victim'"),
            "label must appear in error: {msg}"
        );
        assert!(
            msg.contains("num_workers=0"),
            "error must name the offending field: {msg}"
        );
    }

    #[test]
    fn resolve_num_workers_zero_default_also_rejected() {
        // When num_workers is unset AND the default is 0, still reject.
        let w = Work {
            num_workers: None,
            ..Default::default()
        };
        assert!(resolve_num_workers(&w, 0, "cg").is_err());
    }

    #[test]
    fn resolve_num_workers_falls_back_to_default() {
        let w = Work {
            num_workers: None,
            ..Default::default()
        };
        assert_eq!(resolve_num_workers(&w, 3, "cg").unwrap(), 3);
    }

    #[test]
    fn resolve_num_workers_explicit_wins_over_default() {
        let w = Work {
            num_workers: Some(7),
            ..Default::default()
        };
        assert_eq!(resolve_num_workers(&w, 3, "cg").unwrap(), 7);
    }

    /// Catalog sweep: every bundled [`Scenario`] must have
    /// `num_workers != Some(0)` in each of its `cgroup_works` entries.
    /// A zero-worker entry would vacuously pass every assertion at runtime.
    #[test]
    fn catalog_has_no_zero_worker_cgroup_works() {
        for scenario in crate::scenario::all_scenarios() {
            for (idx, cw) in scenario.cgroup_works.iter().enumerate() {
                assert_ne!(
                    cw.num_workers,
                    Some(0),
                    "scenario {:?} cgroup_works[{}] declares num_workers=Some(0)",
                    scenario.name,
                    idx,
                );
            }
        }
    }
}
