use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Setup, Step, execute_steps};
use super::{CgroupGroup, Ctx, collect_all, spawn_diverse};
use crate::verify::{self, VerifyResult};
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

pub fn custom_cgroup_per_cpu(ctx: &Ctx) -> Result<VerifyResult> {
    fn per_cpu_defs(ctx: &super::Ctx) -> Vec<CgroupDef> {
        let all = ctx.topo.all_cpus();
        let n = (all.len() - 1).min(64);
        (0..n)
            .map(|i| {
                CgroupDef::named(format!("many_{i}"))
                    .with_cpuset(CpusetSpec::Exact([all[i]].into_iter().collect()))
                    .workers(1)
            })
            .collect()
    }

    let steps = vec![Step {
        setup: Setup::Factory(per_cpu_defs),
        ops: vec![],
        hold: HoldSpec::Fixed(Duration::from_secs(1) + ctx.duration),
    }];

    execute_steps(ctx, steps)
}

pub fn custom_cgroup_exhaust_reuse(ctx: &Ctx) -> Result<VerifyResult> {
    fn reuse_defs(ctx: &super::Ctx) -> Vec<CgroupDef> {
        let all = ctx.topo.all_cpus();
        let n = (all.len() - 1).min(15);
        let half = n / 2;
        (0..half)
            .map(|i| {
                CgroupDef::named(format!("reuse_{i}"))
                    .with_cpuset(CpusetSpec::Exact(
                        [all[i % all.len()]].into_iter().collect(),
                    ))
                    .workers(1)
            })
            .collect()
    }

    let all = ctx.topo.all_cpus();
    let n = (all.len() - 1).min(15);
    let half = n / 2;

    // Phase 1 ops: create empty cgroups with cpusets but no workers.
    // Uses raw AddCgroup + SetCpuset ops (not CgroupDefs) because
    // CgroupDef always spawns workers via apply_setup.
    let mut exhaust_ops = Vec::new();
    for i in 0..n {
        let name = format!("exhaust_{i}");
        let cpus: BTreeSet<usize> = [all[i % all.len()]].into_iter().collect();
        exhaust_ops.push(super::ops::Op::AddCgroup {
            name: name.clone().into(),
        });
        exhaust_ops.push(super::ops::Op::SetCpuset {
            cgroup: name.into(),
            cpus: CpusetSpec::Exact(cpus),
        });
    }

    let mut remove_ops = Vec::new();
    for i in 0..half {
        let name = format!("exhaust_{i}");
        remove_ops.push(super::ops::Op::RemoveCgroup { name: name.into() });
    }

    let steps = vec![
        // Phase 1: create N exhaust cgroups (no workers — they just occupy slots).
        Step {
            setup: vec![].into(),
            ops: exhaust_ops,
            hold: HoldSpec::Fixed(Duration::from_secs(1)),
        },
        // Phase 2: remove first half.
        Step {
            setup: vec![].into(),
            ops: remove_ops,
            hold: HoldSpec::Fixed(Duration::from_secs(1)),
        },
        // Phase 3: create replacement cgroups with workers.
        Step {
            setup: Setup::Factory(reuse_defs),
            ops: vec![],
            hold: HoldSpec::Fixed(ctx.duration),
        },
    ];

    execute_steps(ctx, steps)
}

/// Per-CPU pinned workers + custom gap verification (max_gap_ms > 1500).
/// Not expressible via Op/Step's standard verify_not_starved.
pub fn custom_cgroup_dsq_contention(ctx: &Ctx) -> Result<VerifyResult> {
    // Multiple CPUs sharing a DSQ under bursty wake patterns. Lockless
    // peek can miss tasks when store visibility ordering delays the
    // first_task pointer update. Without a fallback to the locked
    // consume path, CPUs go idle and never retry.
    let all = ctx.topo.all_cpus();
    if all.len() < 4 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=4 CPUs".into()],
            stats: Default::default(),
        });
    }
    let last = all.len() - 1;

    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cell_0", &all[..last].iter().copied().collect())?;
    thread::sleep(Duration::from_secs(3));

    let n_unpinned = (last * 3).max(8);
    let mut h_cell = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: n_unpinned,
        work_type: WorkType::Bursty {
            burst_ms: 10,
            sleep_ms: 5,
        },
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cell_0", &h_cell.tids())?;

    let n_pinned = last.min(4);
    let mut pinned_handles = Vec::new();
    for &cpu in all.iter().take(n_pinned) {
        let h = WorkloadHandle::spawn(&WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::SingleCpu(cpu),
            work_type: WorkType::Bursty {
                burst_ms: 10,
                sleep_ms: 5,
            },
            ..Default::default()
        })?;
        ctx.cgroups.move_tasks("cell_0", &h.tids())?;
        pinned_handles.push(h);
    }

    h_cell.start();
    for h in &mut pinned_handles {
        h.start();
    }
    thread::sleep(ctx.duration);

    let mut r = VerifyResult::pass();
    r.merge(verify::verify_not_starved(&h_cell.stop_and_collect()));
    for h in pinned_handles {
        let reports = h.stop_and_collect();
        for w in &reports {
            if w.max_gap_ms > 1500 {
                r.passed = false;
                r.details.push(format!(
                    "pinned worker {} on CPU {} had {}ms gap (dispatch contention stall)",
                    w.tid,
                    w.cpus_used.iter().next().unwrap_or(&0),
                    w.max_gap_ms
                ));
            }
        }
        r.merge(verify::verify_not_starved(&reports));
    }
    Ok(r)
}

/// Uses spawn_diverse helper for 5 different workload types across cells.
/// Dynamic cell count and workload rotation logic is not Op/Step compatible.
pub fn custom_cgroup_workload_variety(ctx: &Ctx) -> Result<VerifyResult> {
    // All workload types across 5 cells, no flags. Exercises base dispatch with every work pattern.
    if ctx.topo.all_cpus().len() < 6 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=6 CPUs for 5 cells".into()],
            stats: Default::default(),
        });
    }
    let names: Vec<String> = (0..5).map(|i| format!("cell_{i}")).collect();
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    for n in &names {
        _guard.add_cgroup_no_cpuset(n)?;
    }
    thread::sleep(Duration::from_secs(3));
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let handles = spawn_diverse(ctx, &name_refs)?;
    thread::sleep(ctx.duration);
    Ok(collect_all(handles))
}

/// Uses spawn_diverse for workload variety + manual cpuset partitioning.
pub fn custom_cgroup_cpuset_workload_variety(ctx: &Ctx) -> Result<VerifyResult> {
    // All workload types with cpusets.
    let all = ctx.topo.all_cpus();
    if all.len() < 6 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=6 CPUs".into()],
            stats: Default::default(),
        });
    }
    let last = all.len() - 1;
    let chunk = last / 3;
    let names = ["cell_0", "cell_1", "cell_2"];
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    for (i, n) in names.iter().enumerate() {
        let start = i * chunk;
        let end = if i == 2 { last } else { (i + 1) * chunk };
        _guard.add_cgroup(n, &all[start..end].iter().copied().collect())?;
    }
    thread::sleep(Duration::from_secs(3));
    let handles = spawn_diverse(ctx, &names)?;
    thread::sleep(ctx.duration);
    Ok(collect_all(handles))
}

/// spawn_diverse + dynamic cell add/remove mid-run.
pub fn custom_cgroup_dynamic_workload_variety(ctx: &Ctx) -> Result<VerifyResult> {
    // Dynamic cell ops with diverse workloads.
    if ctx.topo.all_cpus().len() < 5 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=5 CPUs for dynamic cell add".into()],
            stats: Default::default(),
        });
    }
    let names: Vec<String> = (0..3).map(|i| format!("cell_{i}")).collect();
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    for n in &names {
        _guard.add_cgroup_no_cpuset(n)?;
    }
    thread::sleep(Duration::from_secs(3));
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut handles = spawn_diverse(ctx, &name_refs)?;
    thread::sleep(ctx.duration / 3);
    // Add cells with more workload types
    _guard.add_cgroup_no_cpuset("cell_3")?;
    let mut h = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: 4,
        work_type: WorkType::Bursty {
            burst_ms: 100,
            sleep_ms: 50,
        },
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cell_3", &h.tids())?;
    h.start();
    handles.push(h);
    thread::sleep(ctx.duration / 3);
    // Remove cell_3 — guard still tracks it, but explicit removal
    // during the scenario is fine; guard's drop will skip missing cells.
    if let Some(h) = handles.pop() {
        h.stop_and_collect();
    }
    let _ = ctx.cgroups.remove_cell("cell_3");
    thread::sleep(ctx.duration / 3);
    Ok(collect_all(handles))
}

/// LLC-specific cpusets + tight flip loop. Uses Instant::now() deadline
/// loop and LLC-aligned BTreeSets computed at runtime. Not Op/Step compatible.
pub fn custom_cgroup_cpuset_crossllc_race(ctx: &Ctx) -> Result<VerifyResult> {
    // Need at least 2 LLCs to flip cpusets across LLC boundaries.
    if ctx.topo.num_llcs() < 2 {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: need >=2 LLCs".into()],
            stats: Default::default(),
        });
    }
    let llc0_full: BTreeSet<usize> = ctx.topo.llc_aligned_cpuset(0);
    let llc1_full: BTreeSet<usize> = ctx.topo.llc_aligned_cpuset(1);
    if llc0_full.is_empty() {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: LLC0 has no CPUs".into()],
            stats: Default::default(),
        });
    }

    // Reserve one CPU from LLC0 for cell 0 to avoid cell-0-starvation.
    let reserved = *llc0_full.iter().next().unwrap();
    let llc0: BTreeSet<usize> = llc0_full
        .iter()
        .copied()
        .filter(|c| *c != reserved)
        .collect();
    let llc1: BTreeSet<usize> = llc1_full.clone();
    if llc0.is_empty() {
        return Ok(VerifyResult {
            passed: true,
            details: vec!["skipped: LLC0 too small after reserving for cell 0".into()],
            stats: Default::default(),
        });
    }

    // Two cells, initially each on its own LLC.
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cell_0", &llc0)?;
    _guard.add_cgroup("cell_1", &llc1)?;
    thread::sleep(Duration::from_secs(2));

    // Oversubscribe both cells — lots of enqueue pressure.
    let n = llc0.len().max(4) * 8;
    let mut h0 = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: n,
        work_type: WorkType::Mixed,
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cell_0", &h0.tids())?;
    let mut h1 = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: n,
        work_type: WorkType::Mixed,
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cell_1", &h1.tids())?;
    h0.start();
    h1.start();

    // Rapidly flip cpusets across LLC boundaries to race with LLC assignment.
    // Build cross-LLC sets (excluding the reserved CPU).
    let cross0: BTreeSet<usize> = llc1.iter().copied().collect();
    let cross1: BTreeSet<usize> = llc0.iter().copied().collect();
    let deadline = Instant::now() + ctx.duration;
    let mut flip = false;
    while Instant::now() < deadline {
        if flip {
            // cell_0 on LLC1 CPUs, cell_1 on LLC0 CPUs — cross-LLC
            let _ = ctx.cgroups.set_cpuset("cell_0", &cross0);
            let _ = ctx.cgroups.set_cpuset("cell_1", &cross1);
        } else {
            // cell_0 on LLC0 CPUs, cell_1 on LLC1 CPUs — aligned
            let _ = ctx.cgroups.set_cpuset("cell_0", &llc0);
            let _ = ctx.cgroups.set_cpuset("cell_1", &llc1);
        }
        flip = !flip;
        // Short sleep to let rebalancing/reconfiguration run between flips.
        thread::sleep(Duration::from_millis(200));
    }

    let mut r = VerifyResult::pass();
    r.merge(verify::verify_not_starved(&h0.stop_and_collect()));
    r.merge(verify::verify_not_starved(&h1.stop_and_collect()));
    Ok(r)
}
