//! Stress and edge-case scenario implementations.

use super::ops::{CgroupDef, CpusetSpec, HoldSpec, Op, Setup, Step, execute_steps};
use super::{CgroupGroup, Ctx, collect_all, spawn_diverse};
use crate::assert::{self, AssertResult};
use crate::workload::*;
use anyhow::Result;
use std::collections::BTreeSet;
use std::thread;
use std::time::{Duration, Instant};

fn per_cpu_defs(ctx: &super::Ctx) -> Vec<CgroupDef> {
    let all = ctx.topo.all_cpus();
    let n = (all.len() - 1).min(64);
    (0..n)
        .map(|i| {
            CgroupDef::named(format!("many_{i}"))
                .with_cpuset(CpusetSpec::exact([all[i]]))
                .workers(1)
        })
        .collect()
}

fn cgroup_per_cpu_steps(ctx: &Ctx) -> Vec<Step> {
    vec![Step {
        setup: Setup::Factory(per_cpu_defs),
        ops: vec![],
        hold: HoldSpec::Fixed(Duration::from_secs(1) + ctx.duration),
    }]
}

/// One cgroup per CPU, each with a single pinned worker. Stresses
/// the scheduler with up to 64 cgroups on disjoint single-CPU cpusets.
pub fn custom_cgroup_per_cpu(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, cgroup_per_cpu_steps(ctx))
}

fn reuse_defs(ctx: &super::Ctx) -> Vec<CgroupDef> {
    let all = ctx.topo.all_cpus();
    let n = (all.len() - 1).min(15);
    let half = n / 2;
    (0..half)
        .map(|i| {
            CgroupDef::named(format!("reuse_{i}"))
                .with_cpuset(CpusetSpec::exact([all[i % all.len()]]))
                .workers(1)
        })
        .collect()
}

fn cgroup_exhaust_reuse_steps(ctx: &Ctx) -> Vec<Step> {
    let all = ctx.topo.all_cpus();
    let n = (all.len() - 1).min(15);
    let half = n / 2;

    // Phase 1 ops: create empty cgroups with cpusets but no workers.
    // Uses raw AddCgroup + SetCpuset ops (not CgroupDefs) because
    // CgroupDef always spawns workers via apply_setup.
    let mut exhaust_ops = Vec::new();
    for i in 0..n {
        let name = format!("exhaust_{i}");
        exhaust_ops.push(Op::add_cgroup(name.clone()));
        exhaust_ops.push(Op::set_cpuset(
            name,
            CpusetSpec::exact([all[i % all.len()]]),
        ));
    }

    let mut remove_ops = Vec::new();
    for i in 0..half {
        remove_ops.push(Op::remove_cgroup(format!("exhaust_{i}")));
    }

    vec![
        // Phase 1: create N exhaust cgroups (no workers — they just occupy slots).
        Step::new(exhaust_ops, HoldSpec::Fixed(Duration::from_secs(1))),
        // Phase 2: remove first half.
        Step::new(remove_ops, HoldSpec::Fixed(Duration::from_secs(1))),
        // Phase 3: create replacement cgroups with workers.
        Step {
            setup: Setup::Factory(reuse_defs),
            ops: vec![],
            hold: HoldSpec::Fixed(ctx.duration),
        },
    ]
}

/// Exhaust cgroup slots with empty cpuset-pinned cgroups, remove half,
/// then create replacement cgroups with workers to test slot reuse.
pub fn custom_cgroup_exhaust_reuse(ctx: &Ctx) -> Result<AssertResult> {
    execute_steps(ctx, cgroup_exhaust_reuse_steps(ctx))
}

/// Per-CPU pinned workers + custom gap assertion (max_gap_ms > 1500).
/// Not expressible via Op/Step's standard assert pipeline.
pub fn custom_cgroup_dsq_contention(ctx: &Ctx) -> Result<AssertResult> {
    // Multiple CPUs sharing a DSQ under bursty wake patterns. Lockless
    // peek can miss tasks when store visibility ordering delays the
    // first_task pointer update. Without a fallback to the locked
    // consume path, CPUs go idle and never retry.
    let all = ctx.topo.all_cpus();
    if all.len() < 4 {
        return Ok(AssertResult::skip("need >=4 CPUs"));
    }
    let last = all.len() - 1;

    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cg_0", &all[..last].iter().copied().collect())?;
    thread::sleep(ctx.settle);

    let n_unpinned = (last * 3).max(8);
    let mut h_cgroup = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: n_unpinned,
        work_type: WorkType::bursty(10, 5),
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cg_0", &h_cgroup.worker_pids())?;

    let n_pinned = last.min(4);
    let mut pinned_handles = Vec::new();
    for &cpu in all.iter().take(n_pinned) {
        let h = WorkloadHandle::spawn(&WorkloadConfig {
            num_workers: 1,
            affinity: AffinityMode::SingleCpu(cpu),
            work_type: WorkType::bursty(10, 5),
            ..Default::default()
        })?;
        ctx.cgroups.move_tasks("cg_0", &h.worker_pids())?;
        pinned_handles.push(h);
    }

    h_cgroup.start();
    for h in &mut pinned_handles {
        h.start();
    }
    thread::sleep(ctx.duration);

    let mut r = AssertResult::pass();
    r.merge(assert::assert_not_starved(&h_cgroup.stop_and_collect()));
    for h in pinned_handles {
        let reports = h.stop_and_collect();
        for w in &reports {
            if w.max_gap_ms > 1500 {
                r.passed = false;
                r.details.push(crate::assert::AssertDetail::new(
                    crate::assert::DetailKind::Stuck,
                    format!(
                        "pinned worker {} on CPU {} had {}ms gap (dispatch contention stall)",
                        w.tid,
                        w.cpus_used.iter().next().unwrap_or(&0),
                        w.max_gap_ms
                    ),
                ));
            }
        }
        r.merge(assert::assert_not_starved(&reports));
    }
    Ok(r)
}

/// Uses spawn_diverse helper for 5 different workload types across cgroups.
/// Dynamic cgroup count and workload rotation logic is not Op/Step compatible.
pub fn custom_cgroup_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    // All workload types across 5 cgroups, no flags. Exercises base dispatch with every work pattern.
    if ctx.topo.all_cpus().len() < 6 {
        return Ok(AssertResult::skip("need >=6 CPUs for 5 cgroups"));
    }
    let names: Vec<String> = (0..5).map(|i| format!("cg_{i}")).collect();
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    for n in &names {
        _guard.add_cgroup_no_cpuset(n)?;
    }
    thread::sleep(ctx.settle);
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let handles = spawn_diverse(ctx, &name_refs)?;
    thread::sleep(ctx.duration);
    Ok(collect_all(handles, &ctx.assert))
}

/// Uses spawn_diverse for workload variety + manual cpuset partitioning.
pub fn custom_cgroup_cpuset_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    // All workload types with cpusets.
    let all = ctx.topo.all_cpus();
    if all.len() < 6 {
        return Ok(AssertResult::skip("need >=6 CPUs"));
    }
    let last = all.len() - 1;
    let chunk = last / 3;
    let names = ["cg_0", "cg_1", "cg_2"];
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    for (i, n) in names.iter().enumerate() {
        let start = i * chunk;
        let end = if i == 2 { last } else { (i + 1) * chunk };
        _guard.add_cgroup(n, &all[start..end].iter().copied().collect())?;
    }
    thread::sleep(ctx.settle);
    let handles = spawn_diverse(ctx, &names)?;
    thread::sleep(ctx.duration);
    Ok(collect_all(handles, &ctx.assert))
}

/// spawn_diverse + dynamic cgroup add/remove mid-run.
pub fn custom_cgroup_dynamic_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    // Dynamic cgroup ops with diverse workloads.
    if ctx.topo.all_cpus().len() < 5 {
        return Ok(AssertResult::skip(
            "need >=5 CPUs for dynamic cgroup add",
        ));
    }
    let names: Vec<String> = (0..3).map(|i| format!("cg_{i}")).collect();
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    for n in &names {
        _guard.add_cgroup_no_cpuset(n)?;
    }
    thread::sleep(ctx.settle);
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let mut handles = spawn_diverse(ctx, &name_refs)?;
    thread::sleep(ctx.duration / 3);
    // Add cgroups with more workload types
    _guard.add_cgroup_no_cpuset("cg_3")?;
    let mut h = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: 4,
        work_type: WorkType::bursty(100, 50),
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cg_3", &h.worker_pids())?;
    h.start();
    handles.push(h);
    thread::sleep(ctx.duration / 3);
    // Remove cg_3 — guard still tracks it, but explicit removal
    // during the scenario is fine; guard's drop will skip missing cgroups.
    if let Some(h) = handles.pop() {
        h.stop_and_collect();
    }
    let _ = ctx.cgroups.remove_cgroup("cg_3");
    thread::sleep(ctx.duration / 3);
    Ok(collect_all(handles, &ctx.assert))
}

/// LLC-specific cpusets + tight flip loop. Uses Instant::now() deadline
/// loop and LLC-aligned BTreeSets computed at runtime. Not Op/Step compatible.
pub fn custom_cgroup_cpuset_cross_llc_race(ctx: &Ctx) -> Result<AssertResult> {
    // Need at least 2 LLCs to flip cpusets across LLC boundaries.
    if ctx.topo.num_llcs() < 2 {
        return Ok(AssertResult::skip("need >=2 LLCs"));
    }
    let llc0_full: BTreeSet<usize> = ctx.topo.llc_aligned_cpuset(0);
    let llc1_full: BTreeSet<usize> = ctx.topo.llc_aligned_cpuset(1);
    if llc0_full.is_empty() {
        return Ok(AssertResult::skip("LLC0 has no CPUs"));
    }

    // Reserve one CPU from LLC0 for cg_0 to avoid cg_0-starvation.
    let reserved = *llc0_full.iter().next().unwrap();
    let llc0: BTreeSet<usize> = llc0_full
        .iter()
        .copied()
        .filter(|c| *c != reserved)
        .collect();
    let llc1: BTreeSet<usize> = llc1_full.clone();
    if llc0.is_empty() {
        return Ok(AssertResult::skip(
            "LLC0 too small after reserving for cg_0",
        ));
    }

    // Two cgroups, initially each on its own LLC.
    let mut _guard = CgroupGroup::new(ctx.cgroups);
    _guard.add_cgroup("cg_0", &llc0)?;
    _guard.add_cgroup("cg_1", &llc1)?;
    thread::sleep(Duration::from_secs(2));

    // Oversubscribe both cgroups — lots of enqueue pressure.
    let n = llc0.len().max(4) * 8;
    let mut h0 = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: n,
        work_type: WorkType::Mixed,
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cg_0", &h0.worker_pids())?;
    let mut h1 = WorkloadHandle::spawn(&WorkloadConfig {
        num_workers: n,
        work_type: WorkType::Mixed,
        ..Default::default()
    })?;
    ctx.cgroups.move_tasks("cg_1", &h1.worker_pids())?;
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
            // cg_0 on LLC1 CPUs, cg_1 on LLC0 CPUs — cross-LLC
            let _ = ctx.cgroups.set_cpuset("cg_0", &cross0);
            let _ = ctx.cgroups.set_cpuset("cg_1", &cross1);
        } else {
            // cg_0 on LLC0 CPUs, cg_1 on LLC1 CPUs — aligned
            let _ = ctx.cgroups.set_cpuset("cg_0", &llc0);
            let _ = ctx.cgroups.set_cpuset("cg_1", &llc1);
        }
        flip = !flip;
        // Short sleep to let rebalancing/reconfiguration run between flips.
        thread::sleep(Duration::from_millis(200));
    }

    let mut r = AssertResult::pass();
    r.merge(assert::assert_not_starved(&h0.stop_and_collect()));
    r.merge(assert::assert_not_starved(&h1.stop_and_collect()));
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgroup::CgroupManager;
    use crate::topology::TestTopology;

    fn ctx_for_test<'a>(cgroups: &'a CgroupManager, topo: &'a TestTopology) -> Ctx<'a> {
        Ctx {
            cgroups,
            topo,
            duration: Duration::from_secs(2),
            workers_per_cgroup: 1,
            sched_pid: Some(1),
            settle: Duration::from_millis(100),
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    #[test]
    fn per_cpu_factory_produces_cgroup_per_cpu_capped_at_64() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(1, 1, 4, 1));
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = cgroup_per_cpu_steps(&ctx);
        assert_eq!(steps.len(), 1);
        assert!(steps[0].ops.is_empty());
        let factory = match steps[0].setup {
            Setup::Factory(f) => f,
            Setup::Defs(_) => panic!("per_cpu should use Factory setup"),
        };
        let defs = factory(&ctx);
        // n = (all_cpus - 1).min(64) with 4 CPUs → 3 defs.
        assert_eq!(defs.len(), 3);
        for (i, d) in defs.iter().enumerate() {
            assert_eq!(d.name, format!("many_{i}"));
            assert!(d.cpuset.is_some());
            assert_eq!(d.works[0].num_workers, Some(1));
        }
    }

    #[test]
    fn exhaust_reuse_builds_three_phases_with_matching_add_remove_counts() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::from_vm_topology(&crate::vmm::topology::Topology::new(1, 1, 8, 1));
        let ctx = ctx_for_test(&cgroups, &topo);

        let steps = cgroup_exhaust_reuse_steps(&ctx);
        assert_eq!(steps.len(), 3);

        // Phase 1: n = (8-1).min(15) = 7 cgroups, each paired AddCgroup+SetCpuset.
        let adds = steps[0]
            .ops
            .iter()
            .filter(|o| matches!(o, Op::AddCgroup { .. }))
            .count();
        let sets = steps[0]
            .ops
            .iter()
            .filter(|o| matches!(o, Op::SetCpuset { .. }))
            .count();
        assert_eq!(adds, 7);
        assert_eq!(sets, 7);
        assert_eq!(steps[0].ops.len(), 14);

        // Phase 2: remove first half (7/2 = 3).
        let removes = steps[1]
            .ops
            .iter()
            .filter(|o| matches!(o, Op::RemoveCgroup { .. }))
            .count();
        assert_eq!(removes, 3);
        assert_eq!(steps[1].ops.len(), 3);

        // Phase 3: factory-built defs, no ops.
        assert!(steps[2].ops.is_empty());
        let factory = match steps[2].setup {
            Setup::Factory(f) => f,
            Setup::Defs(_) => panic!("phase 3 should use Factory setup"),
        };
        let defs = factory(&ctx);
        // half = 7/2 = 3 replacement cgroups.
        assert_eq!(defs.len(), 3);
        for d in &defs {
            assert_eq!(d.works[0].num_workers, Some(1));
        }
    }
}
