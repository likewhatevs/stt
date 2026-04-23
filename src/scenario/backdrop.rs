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
//!   (cgroups, workload handles, payload handles) is torn down when
//!   the step finishes. The runtime enforces this automatically —
//!   no explicit teardown op is required.
//! - A [`Backdrop`] is persistent: what it sets up lives for the
//!   entire Step sequence. Its cgroups are created once before the
//!   first Step and RAII-removed at scenario end; its payloads
//!   spawn once and are killed (with metric emission) after the
//!   last Step tears down.
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
//! Steps may reference Backdrop-owned cgroups by name through
//! cgroup-addressing ops (`Op::SetCpuset`, `Op::MoveAllTasks`, etc.)
//! — name lookups resolve step-local first, then fall through to
//! the Backdrop. Step-local cgroups must not shadow a Backdrop
//! cgroup name, and step-local `Op::RemoveCgroup` targeting a
//! Backdrop cgroup is rejected so later Steps cannot find it
//! missing.

use super::ops::{CgroupDef, Op};
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
/// shorthand [`execute_steps`](super::ops::execute_steps) /
/// [`execute_defs`](super::ops::execute_defs) wrappers forward to
/// internally. There is no cost to using the empty default — the
/// runtime skips the Backdrop setup phase entirely when every vec
/// is empty.
///
/// # Example
///
/// ```no_run
/// use ktstr::prelude::*;
///
/// #[derive(Payload)]
/// #[payload(binary = "stress-ng")]
/// #[default_args("--cpu", "2")]
/// struct BgLoadPayload;
///
/// // Worker-bearing cgroup + empty move target + long-running payload,
/// // all persistent for the scenario.
/// let backdrop = Backdrop::new()
///     .with_cgroup(CgroupDef::named("bg_cell").with_cpuset(CpusetSpec::disjoint(0, 2)))
///     .with_op(Op::add_cgroup("bg_overflow"))
///     .with_payload(&BG_LOAD);
/// ```
#[derive(Debug, Default)]
pub struct Backdrop {
    /// Long-lived cgroups created once and removed at scenario end.
    /// Any Step can reference them by name via `Op::MoveAllTasks`,
    /// `Op::SetCpuset`, etc. Every [`CgroupDef`] here spawns at
    /// least one worker (declared [`Work`](crate::workload::Work)
    /// entries, or a single default Work when `works` is empty).
    /// Declare empty move-target cgroups via [`Self::ops`] /
    /// [`Self::with_op`] using [`Op::AddCgroup`] instead.
    pub cgroups: Vec<CgroupDef>,
    /// Long-lived binary payloads spawned once before the first
    /// Step. The runtime holds the live handles for the duration of
    /// the Step sequence and drains them via `.kill()` (preserving
    /// metric emission) at scenario teardown.
    pub payloads: Vec<&'static Payload>,
    /// Raw [`Op`]s applied during Backdrop setup, before any Step
    /// runs. Run AFTER [`Self::cgroups`] apply_setup and BEFORE
    /// [`Self::payloads`] spawn, in declaration order. Backdrop
    /// ops run with full authority — they can target Backdrop
    /// cgroups with [`Op::RemoveCgroup`] / [`Op::StopCgroup`] /
    /// [`Op::MoveAllTasks`] where step-local ops would be
    /// rejected, since the Backdrop owns the cgroups it's setting
    /// up. Any cgroup / handle / payload these ops create is
    /// tracked by the Backdrop slot and tears down at scenario
    /// end. The typical use is [`Op::AddCgroup`] for empty
    /// move-target cgroups (a [`CgroupDef`] can't express the
    /// zero-worker case because apply_setup forces a worker
    /// spawn).
    pub ops: Vec<Op>,
}

impl Backdrop {
    /// Empty Backdrop — no persistent state. Used by
    /// [`execute_steps`](super::ops::execute_steps) and
    /// [`execute_defs`](super::ops::execute_defs) as the default
    /// passed to [`execute_scenario`](super::ops::execute_scenario).
    pub const EMPTY: Backdrop = Backdrop {
        cgroups: Vec::new(),
        payloads: Vec::new(),
        ops: Vec::new(),
    };

    /// Fresh Backdrop builder. Reads naturally in chain position:
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

    /// Add a persistent binary payload with no extra args. The
    /// payload spawns before the first Step runs and is killed +
    /// metric-drained after the last Step. Scheduler-kind payloads
    /// are rejected at `execute_scenario` entry; this builder does
    /// not check the kind so the check stays in one place.
    ///
    /// **Need custom args or a cgroup placement?** Use
    /// [`Self::with_op`] instead:
    /// `.with_op(Op::run_payload(&BG, vec!["--cpu".into(), "4".into()]))`
    /// or `Op::run_payload_in_cgroup(...)` — both spawn through the
    /// same pipeline as this shorthand but expose the full argument
    /// and placement surface that [`Op::RunPayload`] carries.
    pub fn with_payload(mut self, payload: &'static Payload) -> Self {
        self.payloads.push(payload);
        self
    }

    /// Append several persistent binary payloads at once. See
    /// [`Self::with_payload`] for the spawn-order and argument
    /// contract — every element follows the same no-custom-args
    /// rule, so pass `with_op(Op::run_payload(..))` entries via
    /// [`Self::with_ops`] when per-payload args are required.
    pub fn with_payloads<I: IntoIterator<Item = &'static Payload>>(mut self, payloads: I) -> Self {
        self.payloads.extend(payloads);
        self
    }

    /// Append a raw [`Op`] to run during Backdrop setup. Typical
    /// use: `Op::AddCgroup { .. }` to create empty move-target
    /// cgroups that persist for the scenario but never spawn
    /// workers (a [`CgroupDef`] always spawns at least one Work
    /// entry, so empty cgroups are only expressible via ops).
    ///
    /// Setup order: CgroupDefs apply first, then ops run, then
    /// payloads spawn last. Backdrop ops execute with the backdrop
    /// target slot active so any cgroup / handle / payload they
    /// create is tracked by the Backdrop and survives every Step's
    /// teardown.
    pub fn with_op(mut self, op: Op) -> Self {
        self.ops.push(op);
        self
    }

    /// Append several raw [`Op`]s at once. See [`Self::with_op`]
    /// for the ordering and routing contract.
    pub fn with_ops<I: IntoIterator<Item = Op>>(mut self, ops: I) -> Self {
        self.ops.extend(ops);
        self
    }

    /// True when the Backdrop has no persistent entities declared.
    /// `execute_scenario` checks this to skip the Backdrop setup
    /// phase entirely — zero overhead for scenarios that do not
    /// use persistent state.
    pub fn is_empty(&self) -> bool {
        self.cgroups.is_empty() && self.payloads.is_empty() && self.ops.is_empty()
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
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
    };

    #[test]
    fn empty_backdrop_has_no_entities() {
        let b = Backdrop::EMPTY;
        assert!(b.cgroups.is_empty());
        assert!(b.payloads.is_empty());
        assert!(b.ops.is_empty());
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
    fn with_payloads_extends_in_order() {
        let b = Backdrop::new().with_payloads([&TEST_PAYLOAD, &TEST_PAYLOAD]);
        assert_eq!(b.payloads.len(), 2);
        assert_eq!(b.payloads[0].name, "test_bin");
        assert_eq!(b.payloads[1].name, "test_bin");
    }

    #[test]
    fn with_payloads_appends_after_with_payload() {
        let b = Backdrop::new()
            .with_payload(&TEST_PAYLOAD)
            .with_payloads([&TEST_PAYLOAD]);
        assert_eq!(b.payloads.len(), 2);
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
        assert_eq!(d.ops.len(), Backdrop::EMPTY.ops.len());
    }

    #[test]
    fn with_op_appends_and_loses_empty() {
        let b = Backdrop::new().with_op(Op::add_cgroup("empty_target"));
        assert_eq!(b.ops.len(), 1);
        assert!(matches!(&b.ops[0], Op::AddCgroup { name } if name.as_ref() == "empty_target"));
        assert!(!b.is_empty());
    }

    #[test]
    fn with_ops_appends_several_in_order() {
        let b = Backdrop::new().with_ops(vec![Op::add_cgroup("cg_1"), Op::add_cgroup("cg_1/sub")]);
        assert_eq!(b.ops.len(), 2);
        assert!(matches!(&b.ops[0], Op::AddCgroup { name } if name.as_ref() == "cg_1"));
        assert!(matches!(&b.ops[1], Op::AddCgroup { name } if name.as_ref() == "cg_1/sub"));
    }

    #[test]
    fn chain_with_op_interleaves_with_other_builders() {
        let b = Backdrop::new()
            .with_cgroup(CgroupDef::named("cg_workers"))
            .with_op(Op::add_cgroup("cg_empty"))
            .with_payload(&TEST_PAYLOAD);
        assert_eq!(b.cgroups.len(), 1);
        assert_eq!(b.ops.len(), 1);
        assert_eq!(b.payloads.len(), 1);
        assert!(!b.is_empty());
    }
}
