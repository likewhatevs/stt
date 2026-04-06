use super::ops::{CgroupDef, HoldSpec, Op, Step, execute_steps};
use super::{CgroupGroup, Ctx, collect_all, dfl_wl, setup_cells};
use crate::verify::{self, VerifyResult};
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

pub fn custom_nested_cgroup_steady(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cell_0/sub_a"),
            CgroupDef::named("cell_0/sub_b"),
            CgroupDef::named("cell_1/sub_a"),
            CgroupDef::named("cell_1/sub_a/deep"),
        ]
        .into(),
        ops: vec![
            Op::AddCgroup {
                name: "cell_0".into(),
            },
            Op::AddCgroup {
                name: "cell_1".into(),
            },
        ],
        hold: HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_nested_cgroup_task_move(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![
        Step {
            setup: vec![CgroupDef::named("cell_0/sub")].into(),
            ops: vec![
                // Create parents and empty targets for MoveAllTasks.
                Op::AddCgroup {
                    name: "cell_0".into(),
                },
                Op::AddCgroup {
                    name: "cell_1".into(),
                },
                Op::AddCgroup {
                    name: "cell_1/sub".into(),
                },
            ],
            hold: HoldSpec::Fixed(Duration::from_secs(2) + ctx.duration / 4),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::MoveAllTasks {
                from: "cell_0/sub".into(),
                to: "cell_0".into(),
            }],
            hold: HoldSpec::Frac(0.25),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::MoveAllTasks {
                from: "cell_0".into(),
                to: "cell_1/sub".into(),
            }],
            hold: HoldSpec::Frac(0.25),
        },
        Step {
            setup: vec![].into(),
            ops: vec![Op::MoveAllTasks {
                from: "cell_1/sub".into(),
                to: "cell_1".into(),
            }],
            hold: HoldSpec::Frac(0.25),
        },
    ];

    execute_steps(ctx, steps)
}

/// Rapid nested cgroup create/destroy with dynamic names. Custom logic
/// for dynamic naming.
pub fn custom_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<VerifyResult> {
    let (handles, _guard) = setup_cells(ctx, 2, &dfl_wl(ctx))?;
    let deadline = Instant::now() + ctx.duration;
    let mut i = 0;
    while Instant::now() < deadline {
        let path = format!("cell_0/churn_{i}");
        ctx.cgroups.create_cell(&path)?;
        if i % 3 == 0 {
            let deep = format!("{path}/deep");
            ctx.cgroups.create_cell(&deep)?;
            thread::sleep(Duration::from_millis(50));
            let _ = ctx.cgroups.remove_cell(&deep);
        }
        thread::sleep(Duration::from_millis(50));
        let _ = ctx.cgroups.remove_cell(&path);
        i += 1;
    }
    Ok(collect_all(handles))
}

/// Nested cgroups with cpusets + subtree_control filesystem writes.
/// Custom filesystem ops not representable via Op, stays custom.
pub fn custom_nested_cgroup_cpuset(ctx: &Ctx) -> Result<VerifyResult> {
    let all = ctx.topo.all_cpus();
    if all.len() < 4 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=4 CPUs".into()],
            stats: Default::default(),
        });
    }
    let mid = all.len() / 2;
    let set_a: BTreeSet<usize> = all[..mid].iter().copied().collect();

    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cell_0", &set_a)?;
    thread::sleep(Duration::from_secs(2));

    let sc = std::path::Path::new(&ctx.cgroups.parent_path()).join("cell_0/cgroup.subtree_control");
    let _ = std::fs::write(&sc, "+cpuset");

    let sub_set: BTreeSet<usize> = all[..mid / 2].iter().copied().collect();
    _guard.add_cgroup("cell_0/narrow", &sub_set)?;

    let wl = WorkloadConfig {
        num_workers: ctx.workers_per_cell,
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&wl)?;
    ctx.cgroups.move_tasks("cell_0/narrow", &h.tids())?;
    h.start();

    thread::sleep(ctx.duration);
    let reports = h.stop_and_collect();
    let mut r = VerifyResult::pass();
    r.merge(verify::verify_not_starved(&reports));
    r.merge(verify::verify_isolation(&reports, &sub_set));
    Ok(r)
}

pub fn custom_nested_cgroup_imbalance(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cell_0/sub_a").workers(8),
            CgroupDef::named("cell_1/sub_b")
                .workers(2)
                .work_type(WorkType::Bursty {
                    burst_ms: 50,
                    sleep_ms: 100,
                }),
        ]
        .into(),
        ops: vec![
            Op::AddCgroup {
                name: "cell_0".into(),
            },
            Op::AddCgroup {
                name: "cell_1".into(),
            },
        ],
        hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_nested_cgroup_noctrl(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("cell_0/sub_a/deep"),
            CgroupDef::named("cell_1/sub_b"),
        ]
        .into(),
        ops: vec![
            Op::AddCgroup {
                name: "cell_0".into(),
            },
            Op::AddCgroup {
                name: "cell_0/sub_a".into(),
            },
            Op::AddCgroup {
                name: "cell_1".into(),
            },
        ],
        hold: HoldSpec::Fixed(Duration::from_secs(3) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}
