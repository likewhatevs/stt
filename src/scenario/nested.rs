//! Nested cgroup hierarchy scenario implementations.

use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use super::{CgroupGroup, Ctx, collect_all, dfl_wl, setup_cgroups};
use crate::assert::{self, AssertResult};
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

pub fn custom_nested_cgroup_steady(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0/sub_a"),
            CgroupDef::named("cg_0/sub_b"),
            CgroupDef::named("cg_1/sub_a"),
            CgroupDef::named("cg_1/sub_a/deep"),
        ]
        .into(),
        ops: vec![
            Op::AddCgroup {
                name: "cg_0".into(),
            },
            Op::AddCgroup {
                name: "cg_1".into(),
            },
        ],
        hold: HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_nested_cgroup_task_move(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cg_0/sub")].into(),
            ops: vec![
                // Create parents and empty targets for MoveAllTasks.
                Op::AddCgroup {
                    name: "cg_0".into(),
                },
                Op::AddCgroup {
                    name: "cg_1".into(),
                },
                Op::AddCgroup {
                    name: "cg_1/sub".into(),
                },
            ],
            hold: HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration / 4),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::MoveAllTasks {
                from: "cg_0/sub".into(),
                to: "cg_0".into(),
            }],
            hold: HoldSpec::Frac(0.25),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::MoveAllTasks {
                from: "cg_0".into(),
                to: "cg_1/sub".into(),
            }],
            hold: HoldSpec::Frac(0.25),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::MoveAllTasks {
                from: "cg_1/sub".into(),
                to: "cg_1".into(),
            }],
            hold: HoldSpec::Frac(0.25),
        },
    ];

    execute_steps(ctx, steps)
}

/// Rapid nested cgroup create/destroy with dynamic names. Custom logic
/// for dynamic naming.
pub fn custom_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    let (handles, _guard) = setup_cgroups(ctx, 2, &dfl_wl(ctx))?;
    let deadline = Instant::now() + ctx.duration;
    let mut i = 0;
    while Instant::now() < deadline {
        let path = format!("cg_0/churn_{i}");
        ctx.cgroups.create_cgroup(&path)?;
        if i % 3 == 0 {
            let deep = format!("{path}/deep");
            ctx.cgroups.create_cgroup(&deep)?;
            thread::sleep(Duration::from_millis(50));
            let _ = ctx.cgroups.remove_cgroup(&deep);
        }
        thread::sleep(Duration::from_millis(50));
        let _ = ctx.cgroups.remove_cgroup(&path);
        i += 1;
    }
    Ok(collect_all(handles, &ctx.assert))
}

/// Nested cgroups with cpusets. `create_cgroup` auto-enables
/// controllers on intermediate cgroup `subtree_control` for
/// nested paths.
pub fn custom_nested_cgroup_cpuset(ctx: &Ctx) -> Result<AssertResult> {
    let all = ctx.topo.all_cpus();
    if all.len() < 4 {
        return Ok(AssertResult::skip("skipped: need >=4 CPUs"));
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
    ctx.cgroups.move_tasks("cg_0/narrow", &h.tids())?;
    h.start();

    thread::sleep(ctx.duration);
    let reports = h.stop_and_collect();
    let mut r = AssertResult::pass();
    r.merge(assert::assert_not_starved(&reports));
    r.merge(assert::assert_isolation(&reports, &sub_set));
    Ok(r)
}

pub fn custom_nested_cgroup_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0/sub_a").workers(8),
            CgroupDef::named("cg_1/sub_b")
                .workers(2)
                .work_type(WorkType::Bursty {
                    burst_ms: 50,
                    sleep_ms: 100,
                }),
        ]
        .into(),
        ops: vec![
            Op::AddCgroup {
                name: "cg_0".into(),
            },
            Op::AddCgroup {
                name: "cg_1".into(),
            },
        ],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_nested_cgroup_noctrl(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cg_0/sub_a/deep"),
            CgroupDef::named("cg_1/sub_b"),
        ]
        .into(),
        ops: vec![
            Op::AddCgroup {
                name: "cg_0".into(),
            },
            Op::AddCgroup {
                name: "cg_0/sub_a".into(),
            },
            Op::AddCgroup {
                name: "cg_1".into(),
            },
        ],
        hold: HoldSpec::Fixed(ctx.settle + ctx.duration),
    }];

    execute_steps(ctx, steps)
}
