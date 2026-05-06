//! Nested cgroup hierarchy scenario implementations.

use super::backdrop::Backdrop;
use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_scenario, execute_steps};
use super::{CgroupGroup, Ctx, collect_all, dfl_wl, setup_cgroups};
use crate::assert::{self, AssertResult};
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

/// Four nested sub-cgroups up to three levels deep with steady workload.
pub fn custom_nested_cgroup_steady(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0/sub_a"),
                CgroupDef::named("cg_0/sub_b"),
                CgroupDef::named("cg_1/sub_b"),
                CgroupDef::named("cg_1/sub_a/deep"),
            ],
            HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration),
        )
        .set_ops(vec![
            Op::add_cgroup("cg_0"),
            Op::add_cgroup("cg_1"),
            Op::add_cgroup("cg_1/sub_a"),
        ]),
    ];

    execute_steps(ctx, steps)
}

/// Move workers through nested hierarchy: sub -> parent ->
/// cross-hierarchy sub -> parent.
///
/// The four cgroups (`cg_0/sub` with workers; `cg_0`, `cg_1`, and
/// `cg_1/sub` as empty move targets) all persist for the full
/// scenario — declaring them on the Backdrop is what lets every
/// Step's `MoveAllTasks` resolve the target cgroup without the
/// previous Step's teardown rmdir'ing it. Workers spawn inside
/// `cg_0/sub` via [`CgroupDef`]; the empty peer cgroups go through
/// [`Backdrop::with_ops`] so no implicit worker spawn happens
/// there.
pub fn custom_nested_cgroup_task_move(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0/sub"))
        .with_ops(vec![
            Op::add_cgroup("cg_0"),
            Op::add_cgroup("cg_1"),
            Op::add_cgroup("cg_1/sub"),
        ]);
    let steps = vec![
        // Settle: hold once so workers run inside cg_0/sub before
        // the first MoveAllTasks. Matches the legacy 2s + duration/4
        // budget the pre-refactor single-Step version used.
        Step::new(
            vec![],
            HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration / 4),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_0/sub", "cg_0")],
            HoldSpec::Frac(0.25),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_0", "cg_1/sub")],
            HoldSpec::Frac(0.25),
        ),
        Step::new(
            vec![Op::move_all_tasks("cg_1/sub", "cg_1")],
            HoldSpec::Frac(0.25),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Rapid nested cgroup create/destroy with dynamic names. Custom logic
/// for dynamic naming.
pub fn custom_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    let (handles, mut guard) = setup_cgroups(ctx, 2, &dfl_wl(ctx))?;
    let deadline = Instant::now() + ctx.duration;
    let mut i = 0usize;
    // Cap on the number of distinct ephemeral cgroup names. The
    // parent and 'deep' child remove paths are both best-effort
    // (see comments below); without a cap a long scenario with
    // persistent EBUSY/ENOENT churn would accumulate one cgroup
    // per iteration in the cgroupfs tree until the
    // `setup_cgroups` guard's Drop reaps them at scenario
    // teardown. Reusing the same 100 names via `i % 100` bounds
    // the peak resident leaked-cgroup count to at most 100
    // parents (plus their `deep` children on the every-3rd
    // iterations) while still exercising the rapid
    // create→remove churn the test is designed to drive.
    // `create_cgroup` is idempotent on a name whose dir already
    // exists (`if !p.exists()` in `CgroupManager::create_cgroup`),
    // so a cycle that lapped a still-resident sibling is a no-op
    // re-create rather than an error. Mirrors the cap in the
    // single-level sibling `custom_cgroup_rapid_churn` in
    // `scenario/dynamic.rs`.
    //
    // Each parent (and 'deep' child on every-3rd iterations) is
    // registered in the `setup_cgroups` guard via
    // `add_cgroup_no_cpuset` so its Drop reaps any cgroup whose
    // best-effort remove_cgroup below failed. The reverse-iterate
    // contract in `CgroupGroup::drop` removes children before
    // parents (matters here: a `deep` push always happens after
    // its parent's push within the same iteration, so reverse
    // iteration tears down the child first — preventing the
    // ENOTEMPTY that an already-leaked deep would otherwise
    // produce when the guard tries to remove its parent).
    const MAX_EPHEMERAL_NAMES: usize = 100;
    while Instant::now() < deadline {
        let path = format!("cg_0/churn_{}", i % MAX_EPHEMERAL_NAMES);
        guard.add_cgroup_no_cpuset(&path)?;
        if i.is_multiple_of(3) {
            let deep = format!("{path}/deep");
            guard.add_cgroup_no_cpuset(&deep)?;
            thread::sleep(Duration::from_millis(50));
            // Best-effort teardown of the nested 'deep' child
            // before its parent: a transient EBUSY from the
            // kernel's drain path or ENOENT if the parent's
            // removal below races and reaps it leaves the path
            // for the parent to clean up. The setup_cgroups
            // guard reaps any leaked cgroups at scenario
            // teardown.
            if let Err(e) = ctx.cgroups.remove_cgroup(&deep) {
                tracing::warn!(cgroup = %deep, err = %format!("{e:#}"), "nested churn: remove_cgroup(deep) failed; parent removal or guard Drop will reap");
            }
        }
        thread::sleep(Duration::from_millis(50));
        // Parent removal in the same churn loop. EBUSY (a child
        // 'deep' cgroup is still being torn down by its own
        // remove_cgroup above) or ENOENT (already gone) here
        // leaves the cgroup for the guard's Drop to reap on
        // scenario teardown. Bailing would truncate the churn
        // workload mid-run and mask hierarchy races.
        if let Err(e) = ctx.cgroups.remove_cgroup(&path) {
            tracing::warn!(cgroup = %path, err = %format!("{e:#}"), "nested churn: remove_cgroup(path) failed; guard Drop will reap on scenario teardown");
        }
        i = i.wrapping_add(1);
    }
    Ok(collect_all(handles, &ctx.assert))
}

/// Nested cgroups with cpusets. `create_cgroup` auto-enables
/// controllers on intermediate cgroup `subtree_control` for
/// nested paths.
pub fn custom_nested_cgroup_cpuset(ctx: &Ctx) -> Result<AssertResult> {
    let all = ctx.topo.all_cpus();
    if all.len() < 4 {
        return Ok(AssertResult::skip("need >=4 CPUs"));
    }
    let mid = all.len() / 2;
    let set_a: BTreeSet<usize> = all[..mid].iter().copied().collect();

    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cg_0", &set_a)?;
    thread::sleep(Duration::from_secs(2));

    let sub_set: BTreeSet<usize> = all[..mid / 2].iter().copied().collect();
    _guard.add_cgroup("cg_0/narrow", &sub_set)?;

    let wl = WorkloadConfig {
        num_workers: ctx.workers_per_cgroup,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&wl)?;
    ctx.cgroups
        .move_tasks("cg_0/narrow", &h.worker_pids_for_cgroup_procs()?)?;
    h.start();

    thread::sleep(ctx.duration);
    let reports = h.stop_and_collect();
    let mut r = AssertResult::pass();
    r.merge(assert::assert_not_starved(&reports));
    r.merge(assert::assert_isolation(&reports, &sub_set));
    Ok(r)
}

/// Nested sub-cgroups with heavy SpinWait vs light Bursty load imbalance.
pub fn custom_nested_cgroup_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0/sub_a").workers(8),
                CgroupDef::named("cg_1/sub_b")
                    .workers(2)
                    .work_type(WorkType::bursty(
                        Duration::from_millis(50),
                        Duration::from_millis(100),
                    )),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration),
        )
        .set_ops(vec![Op::add_cgroup("cg_0"), Op::add_cgroup("cg_1")]),
    ];

    execute_steps(ctx, steps)
}

/// Three-level nested hierarchy with workers at leaf cgroups.
pub fn custom_nested_cgroup_no_ctrl(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0/sub_a/deep"),
                CgroupDef::named("cg_1/sub_b"),
            ],
            HoldSpec::Fixed(ctx.settle + ctx.duration),
        )
        .set_ops(vec![
            Op::add_cgroup("cg_0"),
            Op::add_cgroup("cg_0/sub_a"),
            Op::add_cgroup("cg_1"),
        ]),
    ];

    execute_steps(ctx, steps)
}
