use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::stt_test;
use stt::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};

fn main() {
    let args = libtest_mimic::Arguments::from_args();
    let presets = stt::vm::gauntlet_presets();
    let host_llcs = stt::test_support::host_llc_count();
    let host_topo = stt::vmm::host_topology::HostTopology::from_sysfs().ok();
    let mut trials = Vec::new();

    for entry in stt::test_support::STT_TESTS.iter() {
        let profiles = entry
            .scheduler
            .generate_profiles(entry.required_flags, entry.excluded_flags);

        // Base trial: runs with the entry's own topology, not ignored
        // unless it's a demo_ test.
        let expect_err = entry.expect_err;
        trials.push(
            libtest_mimic::Trial::test(entry.name.to_string(), move || {
                match stt::test_support::run_stt_test(entry) {
                    Ok(_) if expect_err => Err("expected error but test passed".into()),
                    Ok(_) => Ok(()),
                    Err(_) if expect_err => Ok(()),
                    Err(e) => Err(format!("{e:#}").into()),
                }
            })
            .with_ignored_flag(entry.name.starts_with("demo_")),
        );

        // Gauntlet variants: topology x flags, always ignored by default.
        for preset in &presets {
            if !stt::test_support::preset_matches(preset, entry, host_llcs) {
                continue;
            }
            if let Some(ref host) = host_topo
                && !stt::test_support::host_preset_compatible(preset, host)
            {
                continue;
            }
            let t = &preset.topology;
            let topo_str = format!(
                "{}s{}c{}t",
                t.sockets, t.cores_per_socket, t.threads_per_core,
            );
            let cpus = t.total_cpus();
            let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
            let preset_name = preset.name;

            for profile in &profiles {
                let pname = profile.name();
                let name = format!("gauntlet/{}/{}/{}", entry.name, preset_name, pname);
                let topo_str = topo_str.clone();
                let flags: Vec<String> = profile.flags.iter().map(|s| s.to_string()).collect();

                let expect_err = entry.expect_err;
                trials.push(
                    libtest_mimic::Trial::test(name, move || {
                        let (sockets, cores, threads) =
                            stt::test_support::parse_topo_string(&topo_str)
                                .expect("invalid topo string");
                        let topo = stt::test_support::TopoOverride {
                            sockets,
                            cores,
                            threads,
                            memory_mb,
                        };
                        match stt::test_support::run_stt_test_with_topo_and_flags(
                            entry, &topo, &flags,
                        ) {
                            Ok(_) if expect_err => Err("expected error but test passed".into()),
                            Ok(_) => Ok(()),
                            Err(_) if expect_err => Ok(()),
                            Err(e) => Err(format!("{e:#}").into()),
                        }
                    })
                    .with_ignored_flag(true),
                );
            }
        }
    }

    let conclusion = libtest_mimic::run(&args, trials);
    if let Ok(dir) = std::env::var("STT_SIDECAR_DIR") {
        let sidecars = stt::test_support::collect_sidecars(std::path::Path::new(&dir));
        let rows: Vec<_> = sidecars.iter().map(stt::stats::sidecar_to_row).collect();
        if !rows.is_empty() {
            eprintln!("{}", stt::stats::analyze_rows(&rows));
        }
    }
    conclusion.exit();
}

const STT_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

// -- basic --

#[stt_test(sockets = 1, cores = 2, threads = 1, memory_mb = 2048)]
fn cover_cgroup_pipe_io(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::basic::custom_cgroup_pipe_io(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048, sustained_samples = 25)]
fn cover_sched_mixed(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::basic::custom_sched_mixed(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_host_cgroup_contention(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::basic::custom_host_cgroup_contention(ctx)
}

// -- affinity --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_affinity_change(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::affinity::custom_cgroup_affinity_change(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::affinity::custom_cgroup_multicpu_pin(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_multicpu_pin(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::affinity::custom_cgroup_cpuset_multicpu_pin(ctx)
}

// -- cpuset --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_apply_midrun(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_apply_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_clear_midrun(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_clear_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_resize(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_resize(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_cpuset_swap_disjoint(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_swap_disjoint(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_workload_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_workload_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_change_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_change_imbalance(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048, max_imbalance_ratio = 20.0, sustained_samples = 15)]
fn cover_cgroup_cpuset_load_shift(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::cpuset::custom_cgroup_cpuset_load_shift(ctx)
}

// -- dynamic --

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048, sustained_samples = 25)]
fn cover_cgroup_add_midrun(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::dynamic::custom_cgroup_add_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_remove_midrun(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::dynamic::custom_cgroup_remove_midrun(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::dynamic::custom_cgroup_rapid_churn(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_add_remove(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::dynamic::custom_cgroup_cpuset_add_remove(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_add_during_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::dynamic::custom_cgroup_add_during_imbalance(ctx)
}

// -- interaction --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_imbalance_mixed_workload(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_imbalance_mixed_workload(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_add_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_add_load_imbalance(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048, sustained_samples = 25)]
fn cover_cgroup_load_oscillation(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_load_oscillation(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_4way_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_4way_load_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_cpuset_imbalance_combined(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_overlap_imbalance_combined(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_cpuset_overlap_imbalance_combined(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_task_migration(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_task_migration(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_cpuset_change(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_cpuset_change(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_noctrl_load_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_noctrl_load_imbalance(ctx)
}

#[stt_test(scheduler = STT_SCHED, sockets = 1, cores = 4, threads = 1, memory_mb = 2048, sustained_samples = 25)]
fn cover_cgroup_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::interaction::custom_cgroup_io_compute_imbalance(ctx)
}

// -- nested --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_steady(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::nested::custom_nested_cgroup_steady(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_task_move(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::nested::custom_nested_cgroup_task_move(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_rapid_churn(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::nested::custom_nested_cgroup_rapid_churn(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_cpuset(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::nested::custom_nested_cgroup_cpuset(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::nested::custom_nested_cgroup_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_nested_cgroup_noctrl(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::nested::custom_nested_cgroup_noctrl(ctx)
}

// -- stress --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_per_cpu(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_per_cpu(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_exhaust_reuse(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_exhaust_reuse(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_dsq_contention(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_dsq_contention(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_workload_variety(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_cpuset_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_cpuset_workload_variety(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 2, memory_mb = 2048)]
fn cover_cgroup_dynamic_workload_variety(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_dynamic_workload_variety(ctx)
}

#[stt_test(sockets = 2, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cgroup_cpuset_crossllc_race(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::stress::custom_cgroup_cpuset_crossllc_race(ctx)
}

// -- performance --

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cache_pressure_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::performance::custom_cache_pressure_imbalance(ctx)
}

#[stt_test(sockets = 2, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cache_yield_wake_affine(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::performance::custom_cache_yield_wake_affine(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_cache_pipe_io_compute_imbalance(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::performance::custom_cache_pipe_io_compute_imbalance(ctx)
}

#[stt_test(sockets = 1, cores = 4, threads = 1, memory_mb = 2048)]
fn cover_fanout_wake(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::performance::custom_fanout_wake(ctx)
}

// -- watchdog timeout overwrite --

#[stt_test(
    scheduler = STT_SCHED,
    sockets = 1, cores = 4, threads = 1, memory_mb = 2048,
    watchdog_timeout_s = 60,
    max_imbalance_ratio = 10.0,
    fail_on_stall = false,
)]
fn cover_watchdog_long_timeout_survives(ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::basic::custom_sched_mixed(ctx)
}

// -- watchdog forced stall (expects scheduler death) --

fn scenario_sched_mixed(_ctx: &Ctx) -> Result<AssertResult> {
    stt::scenario::basic::custom_sched_mixed(_ctx)
}

fn scenario_forced_failure(_ctx: &Ctx) -> Result<AssertResult> {
    let mut r = stt::scenario::basic::custom_sched_mixed(_ctx)?;
    r.passed = false;
    r.details.push("forced failure for auto-repro test".into());
    Ok(r)
}

fn scenario_yield_heavy(ctx: &Ctx) -> Result<AssertResult> {
    use std::time::Duration;
    use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
    use stt::workload::WorkType;
    let steps = vec![Step {
        setup: vec![
            CgroupDef::named("demo_workers")
                .work_type(WorkType::YieldHeavy)
                .workers(4),
        ]
        .into(),
        ops: vec![],
        hold: HoldSpec::Fixed(Duration::from_secs(8)),
    }];
    execute_steps(ctx, steps)
}

static BPF_CRASH: BpfMapWrite = BpfMapWrite {
    map_name_suffix: ".bss",
    offset: 4,
    value: 1,
};

use stt::test_support::SttTestEntry;

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_FORCED_STALL: SttTestEntry = SttTestEntry {
    name: "cover_watchdog_forced_stall",
    func: scenario_sched_mixed,
    cores: 4,
    scheduler: &STT_SCHED,
    extra_sched_args: &["--stall-after", "1"],
    watchdog_timeout_s: 2,
    performance_mode: true,
    expect_err: true,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_STALL_DETECT: SttTestEntry = SttTestEntry {
    name: "neg_stall_detection_scx_exit",
    func: scenario_sched_mixed,
    cores: 4,
    scheduler: &STT_SCHED,
    auto_repro: false,
    extra_sched_args: &["--stall-after", "1"],
    watchdog_timeout_s: 3,
    expect_err: true,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_SCHED_DEATH: SttTestEntry = SttTestEntry {
    name: "neg_sched_death_no_verify_result",
    func: scenario_sched_mixed,
    cores: 4,
    scheduler: &STT_SCHED,
    extra_sched_args: &["--stall-after", "1"],
    watchdog_timeout_s: 3,
    duration_s: 10,
    expect_err: true,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_AUTO_REPRO_VERIFY: SttTestEntry = SttTestEntry {
    name: "neg_auto_repro_on_verify_failure",
    func: scenario_forced_failure,
    cores: 4,
    scheduler: &STT_SCHED,
    expect_err: true,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_CRASH_AFTER: SttTestEntry = SttTestEntry {
    name: "neg_crash_after_auto_repro",
    func: scenario_sched_mixed,
    cores: 4,
    scheduler: &STT_SCHED,
    bpf_map_write: Some(&BPF_CRASH),
    expect_err: true,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_DEMO_BPF_CRASH: SttTestEntry = SttTestEntry {
    name: "demo_bpf_crash_auto_repro",
    func: scenario_sched_mixed,
    cores: 4,
    scheduler: &STT_SCHED,
    bpf_map_write: Some(&BPF_CRASH),
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_HOST_CRASH: SttTestEntry = SttTestEntry {
    name: "neg_host_crash_auto_repro",
    func: scenario_yield_heavy,
    cores: 4,
    scheduler: &STT_SCHED,
    bpf_map_write: Some(&BPF_CRASH),
    expect_err: true,
    ..SttTestEntry::DEFAULT
};

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_DEMO_HOST_CRASH: SttTestEntry = SttTestEntry {
    name: "demo_host_crash_auto_repro",
    func: scenario_yield_heavy,
    cores: 4,
    scheduler: &STT_SCHED,
    bpf_map_write: Some(&BPF_CRASH),
    ..SttTestEntry::DEFAULT
};

// -- monitor evaluation path with default thresholds --

#[stt_test(
    scheduler = STT_SCHED,
    sockets = 1, cores = 4, threads = 1, memory_mb = 2048,
    watchdog_timeout_s = 60,
    max_imbalance_ratio = 20.0,
    sustained_samples = 15,
)]
fn cover_monitor_evaluation(ctx: &Ctx) -> Result<AssertResult> {
    // Exercises the host-side monitor threshold evaluation path in
    // run_stt_test_inner (lines 550-571). The scenario passes, then
    // the host evaluates monitor samples against merged thresholds.
    stt::scenario::basic::custom_sched_mixed(ctx)
}
