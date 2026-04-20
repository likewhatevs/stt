//! Cross-cgroup interaction scenario implementations.

use super::Ctx;
use super::backdrop::Backdrop;
use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_scenario, execute_steps};
use crate::assert::AssertResult;
use crate::workload::*;
use anyhow::Result;
use std::time::Duration;

/// Add a heavy 16-worker cgroup mid-run alongside two light YieldHeavy
/// cgroups.
///
/// `cg_0` and `cg_1` are the steady YieldHeavy residents — they live
/// for the whole scenario on the Backdrop. `cg_2` joins mid-run as a
/// step-local CgroupDef and tears down at the step boundary.
pub fn custom_cgroup_add_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new()
        .with_cgroup(
            CgroupDef::named("cg_0")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        )
        .with_cgroup(
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        );
    let steps = vec![
        // Phase 1: settle with the two steady YieldHeavy cgroups.
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        // Phase 2: add the heavy cg_2 mid-run.
        Step::with_defs(
            vec![CgroupDef::named("cg_2").workers(16)],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Three cgroups with CpuSpin, Bursty, and IoSync workloads.
pub fn custom_cgroup_imbalance_mixed_workload(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(8),
            CgroupDef::named("cg_1")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::bursty(100, 50)),
            CgroupDef::named("cg_2")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSync),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Oscillate load between two cgroups across four phases.
///
/// The two cgroups (`cg_0`, `cg_1`) persist for the whole scenario
/// and live in the [`Backdrop`]; each Step's `Op::stop_cgroup` +
/// `Op::spawn` cycle would otherwise fight with per-Step cgroup
/// teardown because later Steps reference the cgroups by name.
/// Step 0 holds the Backdrop's initial workload assignment; Steps
/// 1-3 swap the heavy/light assignment between `cg_0` and `cg_1`
/// by stopping the currently-running workers (both Backdrop and
/// step-local) and spawning replacements.
pub fn custom_cgroup_load_oscillation(ctx: &Ctx) -> Result<AssertResult> {
    let heavy = Work::default().workers(ctx.workers_per_cgroup * 2);
    let light = Work::default().workers(1).work_type(WorkType::YieldHeavy);

    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup * 2))
        .with_cgroup(
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        );

    // Step 0: hold the Backdrop's initial assignment for the first
    // quarter of the run.
    let mut steps = vec![Step::new(
        vec![],
        HoldSpec::Fixed(ctx.settle + ctx.duration / 4),
    )];

    // Phases 1-3: swap load by stopping and respawning. Workers
    // spawned here are step-local and die at step teardown — which
    // is exactly what the next iteration's `Op::stop_cgroup` was
    // doing explicitly before the Backdrop split landed.
    for i in 1..4 {
        let (heavy_cgroup, light_cgroup): (&str, &str) = if i % 2 == 0 {
            ("cg_0", "cg_1")
        } else {
            ("cg_1", "cg_0")
        };
        steps.push(Step::new(
            vec![
                Op::stop_cgroup("cg_0"),
                Op::stop_cgroup("cg_1"),
                Op::spawn(heavy_cgroup, heavy.clone()),
                Op::spawn(light_cgroup, light.clone()),
            ],
            HoldSpec::Frac(0.25),
        ));
    }

    execute_scenario(ctx, backdrop, steps)
}

/// Four cgroups with 16/1/8/4 workers testing multi-cell rebalancing.
pub fn custom_cgroup_4way_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.all_cpus().len() < 5 {
        return Ok(AssertResult::skip("skipped: need >=5 CPUs for 4 cgroups"));
    }

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(16),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
            CgroupDef::named("cg_2").workers(8),
            CgroupDef::named("cg_3").workers(4),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Disjoint cpusets with oversubscribed CpuSpin vs light Bursty workers.
pub fn custom_cgroup_cpuset_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    let mid = ctx.topo.usable_cpus().len() / 2;

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::disjoint(0, 2))
                .workers(mid * 2),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::disjoint(1, 2))
                .workers(2)
                .work_type(WorkType::bursty(50, 150)),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Three overlapping cpusets with heavy, bursty, and yield-heavy workers.
pub fn custom_cgroup_cpuset_overlap_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    let sets = ctx.topo.overlapping_cpusets(3, 0.5);
    if sets.iter().any(|s| s.is_empty()) {
        return Ok(AssertResult::skip("skipped: not enough CPUs"));
    }

    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .with_cpuset(CpusetSpec::Exact(sets[0].clone()))
                .workers(12),
            CgroupDef::named("cg_1")
                .with_cpuset(CpusetSpec::Exact(sets[1].clone()))
                .workers(2)
                .work_type(WorkType::bursty(50, 100)),
            CgroupDef::named("cg_2")
                .with_cpuset(CpusetSpec::Exact(sets[2].clone()))
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// Workers ping-pong between cg_mobile and cg_1 across 9 MoveAllTasks
/// phases.
///
/// All three cgroups persist for the scenario (the `MoveAllTasks`
/// ops reference them by name across every Step), so they live in
/// the [`Backdrop`]. `cg_1` is declared without `.workers(...)` but
/// still receives a default `Work` under the current
/// `apply_setup` semantics — the `MoveAllTasks` body targets the
/// handle whose key matches `from`, so an extra default worker in
/// `cg_1` participates in the ping-pong. Workers that
/// [`Op::MoveAllTasks`] transfers into a Backdrop cgroup retain
/// their Backdrop ownership so the persistent workers survive
/// every step teardown.
pub fn custom_cgroup_noctrl_task_migration(ctx: &Ctx) -> Result<AssertResult> {
    // cg_0: permanent residents. cg_mobile: workers that ping-pong to cg_1.
    // Each cgroup has exactly one handle so MoveAllTasks tracks correctly.
    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup))
        .with_cgroup(CgroupDef::named("cg_mobile").workers(ctx.workers_per_cgroup))
        .with_cgroup(CgroupDef::named("cg_1"));

    // Settle: let the Backdrop-spawned workers stabilize before the
    // first move.
    let mut steps = vec![Step::new(vec![], HoldSpec::Fixed(Duration::from_secs(2)))];

    // 9 ping-pong phases.
    let mut move_steps: Vec<Step> = (0..9)
        .map(|i| {
            let (from, to) = if i % 2 == 0 {
                ("cg_mobile", "cg_1")
            } else {
                ("cg_1", "cg_mobile")
            };
            Step::new(vec![Op::move_all_tasks(from, to)], HoldSpec::Frac(0.1))
        })
        .collect();
    steps.append(&mut move_steps);
    // Final hold so workers have residency after the last move.
    steps.push(Step::new(vec![], HoldSpec::Frac(0.1)));

    execute_scenario(ctx, backdrop, steps)
}

/// Heavy, light, and mobile workers with tasks ping-ponging to overflow
/// cgroup.
///
/// All four cgroups persist — the `MoveAllTasks` ops in every Step
/// reference `cg_mobile` / `cg_overflow` by name and the permanent
/// `cg_heavy` / `cg_light` workers run across the whole scenario.
/// Declaring them in the [`Backdrop`] makes that persistence
/// explicit and keeps per-step teardown from removing any of them
/// at a step boundary.
pub fn custom_cgroup_noctrl_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    // cg_heavy: 6 permanent CPU-spin workers.
    // cg_light: 2 permanent bursty workers.
    // cg_mobile: 2 workers that ping-pong to cg_overflow.
    // cg_overflow: empty move target (gets a default Work under
    // apply_setup semantics; the move_all_tasks body moves whichever
    // handle is keyed under `from` into `to`).
    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_heavy").workers(6))
        .with_cgroup(CgroupDef::named("cg_mobile").workers(2))
        .with_cgroup(
            CgroupDef::named("cg_light")
                .workers(2)
                .work_type(WorkType::bursty(50, 100)),
        )
        .with_cgroup(CgroupDef::named("cg_overflow"));

    let mut steps = vec![Step::new(vec![], HoldSpec::Fixed(ctx.settle))];

    let mut move_steps: Vec<Step> = (0..5)
        .map(|i| {
            let (from, to) = if i % 2 == 0 {
                ("cg_mobile", "cg_overflow")
            } else {
                ("cg_overflow", "cg_mobile")
            };
            Step::new(
                vec![Op::move_all_tasks(from, to)],
                HoldSpec::Frac(1.0 / 6.0),
            )
        })
        .collect();
    steps.append(&mut move_steps);
    steps.push(Step::new(vec![], HoldSpec::Frac(1.0 / 6.0)));

    execute_scenario(ctx, backdrop, steps)
}

/// Disjoint cpusets cleared mid-run with cpu-controller disabled.
///
/// Two cgroups (`cg_0`, `cg_1`) persist across both Steps — the
/// second Step's `Op::clear_cpuset` targets them by name. Declaring
/// them in the [`Backdrop`] keeps their cpuset assignment alive for
/// the first phase and lets the second Step reach the same cgroups
/// without per-step teardown removing them.
pub fn custom_cgroup_noctrl_cpuset_change(ctx: &Ctx) -> Result<AssertResult> {
    let backdrop = Backdrop::new()
        .with_cgroup(CgroupDef::named("cg_0").with_cpuset(CpusetSpec::disjoint(0, 2)))
        .with_cgroup(CgroupDef::named("cg_1").with_cpuset(CpusetSpec::disjoint(1, 2)));

    let steps = vec![
        // Phase 1: hold the Backdrop's initial disjoint cpusets.
        Step::new(vec![], HoldSpec::Fixed(ctx.settle + ctx.duration / 2)),
        // Phase 2: clear cpusets, hold remaining half.
        Step::new(
            vec![Op::clear_cpuset("cg_0"), Op::clear_cpuset("cg_1")],
            HoldSpec::Frac(0.5),
        ),
    ];

    execute_scenario(ctx, backdrop, steps)
}

/// Heavy CpuSpin vs light YieldHeavy cgroups with cpu-controller disabled.
pub fn custom_cgroup_noctrl_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0").workers(16),
            CgroupDef::named("cg_1")
                .workers(1)
                .work_type(WorkType::YieldHeavy),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}

/// IoSync cgroup vs fully-subscribed CpuSpin cgroup.
pub fn custom_cgroup_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step::with_defs(
        vec![
            CgroupDef::named("cg_0")
                .workers(ctx.workers_per_cgroup)
                .work_type(WorkType::IoSync),
            CgroupDef::named("cg_1").workers(ctx.topo.total_cpus()),
        ],
        HoldSpec::Fixed(ctx.settle + ctx.duration),
    )];

    execute_steps(ctx, steps)
}
