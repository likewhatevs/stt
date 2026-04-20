//! Persistent scenario state that lives across every Step in a
//! `#[ktstr_test]` run.
//!
//! Tests usually express "a scheduler is under load for N seconds"
//! as a Step sequence. Some tests also want entities that persist
//! for the WHOLE run — a long-running binary payload, a synthetic
//! workload that spans the whole scenario, a cgroup whose identity
//! is referenced by multiple Steps. Those go in a [`Backdrop`].
//!
//! # Step vs Backdrop
//!
//! - A [`Step`](super::ops::Step) is bounded: everything it creates
//!   cleans up when the step ends.
//! - A [`Backdrop`] is persistent: what it sets up lives until the
//!   end of the Step sequence, and RAII-teardown happens there.
//!
//! In the "bursty load + scheduler stress test" pattern:
//!
//! - The bursty payload (a persistent fio, a running stress-ng) is
//!   a `Backdrop::with_payload(...)` entry — it runs THROUGHOUT the
//!   test, irrespective of which Step is currently applying ops.
//! - Each Step handles a discrete phase ("settle", "inject
//!   contention", "measure") with its own CgroupDefs that come and
//!   go.
//!
//! # Stage 1 scope note
//!
//! This commit introduces the [`Backdrop`] type and the
//! [`execute_scenario`](super::ops::execute_scenario) entry point
//! WITHOUT changing per-Step lifecycle semantics yet. Step effects
//! still carry over across steps in the current runtime — that
//! auto-cleanup lands in subsequent stages. The primitive is in
//! place so downstream scenarios and tests can already move
//! persistent entities into a Backdrop, and the runtime plumbing
//! catches up around them.

use super::ops::CgroupDef;
use crate::test_support::Payload;

/// Persistent state for a Step sequence.
///
/// Hold long-running entities here instead of re-declaring them in
/// every Step. [`execute_scenario`](super::ops::execute_scenario)
/// owns the Backdrop for the duration of the run, sets up every
/// declared entity once before the first Step, and tears them down
/// at the end (success or Err).
///
/// # Empty default
///
/// Scenarios with no persistent state pass
/// [`Backdrop::EMPTY`](Self::EMPTY), which is also what the
/// legacy [`execute_steps`](super::ops::execute_steps) /
/// [`execute_defs`](super::ops::execute_defs) wrappers forward to
/// internally. There is no cost to using the empty default — the
/// runtime skips the Backdrop setup/teardown phases entirely when
/// every vec is empty.
///
/// # Example
///
/// ```no_run
/// use ktstr::scenario::backdrop::Backdrop;
/// use ktstr::scenario::ops::{CgroupDef, CpusetSpec};
/// use ktstr::test_support::{OutputFormat, Payload, PayloadKind};
///
/// const BG_LOAD: Payload = Payload {
///     name: "bg_load",
///     kind: PayloadKind::Binary("stress-ng"),
///     output: OutputFormat::ExitCode,
///     default_args: &["--cpu", "2"],
///     default_checks: &[],
///     metrics: &[],
/// };
///
/// let backdrop = Backdrop::new()
///     .with_cgroup(CgroupDef::named("bg_cell").with_cpuset(CpusetSpec::disjoint(0, 2)))
///     .with_payload(&BG_LOAD);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Backdrop {
    /// Long-lived cgroups created once and removed at scenario end.
    /// Any Step can reference them by name via `Op::MoveAllTasks`,
    /// `Op::SetCpuset`, etc.
    pub cgroups: Vec<CgroupDef>,
    /// Long-lived binary payloads spawned once before the first
    /// Step. Handles live in [`BackdropState`] and are drained via
    /// `.kill()` (preserving metric emission) at scenario teardown.
    pub payloads: Vec<&'static Payload>,
}

impl Backdrop {
    /// Empty Backdrop — no persistent state. Used by
    /// [`execute_steps`](super::ops::execute_steps) and
    /// [`execute_defs`](super::ops::execute_defs) as the default
    /// passed to [`execute_scenario`](super::ops::execute_scenario).
    pub const EMPTY: Backdrop = Backdrop {
        cgroups: Vec::new(),
        payloads: Vec::new(),
    };

    /// Fresh Backdrop builder. Equivalent to
    /// [`Backdrop::EMPTY.clone()`](Self::EMPTY) but reads more
    /// naturally in chain position:
    /// `Backdrop::new().with_cgroup(...).with_payload(...)`.
    pub fn new() -> Self {
        Backdrop::EMPTY
    }

    /// Add a persistent cgroup to the Backdrop. The cgroup is
    /// created before the first Step runs and removed after the
    /// last Step tears down. Steps reference it by name via
    /// `Op::MoveAllTasks` / `Op::SetCpuset` / etc.
    pub fn with_cgroup(mut self, def: CgroupDef) -> Self {
        self.cgroups.push(def);
        self
    }

    /// Add several persistent cgroups at once.
    pub fn with_cgroups<I: IntoIterator<Item = CgroupDef>>(mut self, defs: I) -> Self {
        self.cgroups.extend(defs);
        self
    }

    /// Add a persistent binary payload. The payload spawns before
    /// the first Step runs and is killed + metric-drained after the
    /// last Step. Scheduler-kind payloads are rejected at
    /// `execute_scenario` entry; this builder does not check the
    /// kind so the check stays in one place.
    pub fn with_payload(mut self, payload: &'static Payload) -> Self {
        self.payloads.push(payload);
        self
    }

    /// True when the Backdrop has no persistent entities declared.
    /// `execute_scenario` checks this to skip the Backdrop setup /
    /// teardown path entirely — zero overhead for scenarios that
    /// do not use persistent state.
    pub fn is_empty(&self) -> bool {
        self.cgroups.is_empty() && self.payloads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{OutputFormat, PayloadKind};

    const TEST_PAYLOAD: Payload = Payload {
        name: "test_bin",
        kind: PayloadKind::Binary("/bin/true"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
    };

    #[test]
    fn empty_backdrop_has_no_entities() {
        let b = Backdrop::EMPTY;
        assert!(b.cgroups.is_empty());
        assert!(b.payloads.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn new_returns_empty() {
        let b = Backdrop::new();
        assert!(b.is_empty());
    }

    #[test]
    fn with_cgroup_appends_and_loses_empty() {
        let b = Backdrop::new().with_cgroup(CgroupDef::named("cg0"));
        assert_eq!(b.cgroups.len(), 1);
        assert_eq!(b.cgroups[0].name.as_ref(), "cg0");
        assert!(!b.is_empty());
    }

    #[test]
    fn with_cgroups_appends_several() {
        let b = Backdrop::new().with_cgroups([
            CgroupDef::named("cg0"),
            CgroupDef::named("cg1"),
            CgroupDef::named("cg2"),
        ]);
        assert_eq!(b.cgroups.len(), 3);
        assert_eq!(b.cgroups[0].name.as_ref(), "cg0");
        assert_eq!(b.cgroups[2].name.as_ref(), "cg2");
    }

    #[test]
    fn with_payload_appends() {
        let b = Backdrop::new().with_payload(&TEST_PAYLOAD);
        assert_eq!(b.payloads.len(), 1);
        assert_eq!(b.payloads[0].name, "test_bin");
        assert!(!b.is_empty());
    }

    #[test]
    fn chain_builds_in_order() {
        let b = Backdrop::new()
            .with_cgroup(CgroupDef::named("cg_a"))
            .with_payload(&TEST_PAYLOAD)
            .with_cgroup(CgroupDef::named("cg_b"));
        assert_eq!(b.cgroups.len(), 2);
        assert_eq!(b.cgroups[0].name.as_ref(), "cg_a");
        assert_eq!(b.cgroups[1].name.as_ref(), "cg_b");
        assert_eq!(b.payloads.len(), 1);
        assert!(!b.is_empty());
    }

    #[test]
    fn default_impl_matches_empty() {
        let d: Backdrop = Default::default();
        assert!(d.is_empty());
        assert_eq!(d.cgroups.len(), Backdrop::EMPTY.cgroups.len());
        assert_eq!(d.payloads.len(), Backdrop::EMPTY.payloads.len());
    }
}
