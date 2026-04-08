use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps_with};
use stt::test_support::{Scheduler, SchedulerSpec, SttTestEntry};

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

    // Verifier tests: build scheduler once, call library with paths.
    trials.push(
        libtest_mimic::Trial::test("demo_verifier_brief", || {
            let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
            let result = stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])
                .map_err(|e| format!("{e:#}"))?;
            let output = stt::verifier::format_verifier_output("stt-sched", &result, false);
            if !output.contains("stt_enqueue") {
                return Err("output should list stt_enqueue".into());
            }
            if !output.contains("stt_dispatch") {
                return Err("output should list stt_dispatch".into());
            }
            if !output.contains("insns=") {
                return Err("output should contain insns=".into());
            }
            if !output.contains("processed=") {
                return Err("output should contain processed=".into());
            }
            Ok(())
        })
        .with_ignored_flag(true),
    );

    trials.push(
        libtest_mimic::Trial::test("demo_verifier_diff", || {
            let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
            let result_a =
                stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])
                    .map_err(|e| format!("{e:#}"))?;
            let result_b =
                stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &[])
                    .map_err(|e| format!("{e:#}"))?;
            let output = stt::verifier::format_verifier_diff(
                "stt-sched",
                &result_a.stats,
                "stt-sched",
                &result_b.stats,
            );
            if !output.contains("delta") {
                return Err("diff output should contain 'delta' header".into());
            }
            if !output.contains("program") {
                return Err("diff output should contain 'program' column".into());
            }
            if !output.contains("+0") {
                return Err("self-comparison deltas should be 0".into());
            }
            Ok(())
        })
        .with_ignored_flag(true),
    );

    trials.push(
        libtest_mimic::Trial::test("demo_verifier_cycle_collapse", || {
            let (sched_bin, stt_bin, kernel) = resolve_verifier_paths("stt-sched")?;
            let sched_args = vec!["--verify-loop".to_string()];
            let result =
                stt::verifier::collect_verifier_output(&sched_bin, &stt_bin, &kernel, &sched_args)
                    .map_err(|e| format!("{e:#}"))?;
            let output = stt::verifier::format_verifier_output("stt-sched", &result, false);
            if !output.contains("stt_dispatch") {
                return Err("output should contain stt_dispatch program".into());
            }
            if !output.contains("scheduler log") {
                return Err("output should contain scheduler log section".into());
            }
            if !output.contains("identical iterations omitted") {
                return Err("cycle collapse should compress verifier loop traces".into());
            }
            Ok(())
        })
        .with_ignored_flag(true),
    );

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

/// Build a scheduler package and resolve paths for verifier tests.
fn resolve_verifier_paths(
    package: &str,
) -> std::result::Result<
    (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf),
    libtest_mimic::Failed,
> {
    let sched_bin =
        stt::build_and_find_binary(package).map_err(|e| format!("build {package}: {e:#}"))?;
    let stt_bin = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let kernel =
        stt::find_kernel().ok_or_else(|| libtest_mimic::Failed::from("no kernel found"))?;
    Ok((sched_bin, stt_bin, kernel))
}

// -- demo_verifier_fail_verify: BPF load rejection via --fail-verify --

const FAIL_SCHED: Scheduler = Scheduler::new("stt_sched").binary(SchedulerSpec::Name("stt-sched"));

fn scenario_fail_verify(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(2)].into(),
        ops: vec![],
        hold: HoldSpec::Frac(1.0),
    }];
    execute_steps_with(ctx, steps, None)
}

#[linkme::distributed_slice(stt::test_support::STT_TESTS)]
#[linkme(crate = linkme)]
static __STT_ENTRY_FAIL_VERIFY: SttTestEntry = SttTestEntry {
    name: "demo_verifier_fail_verify",
    func: scenario_fail_verify,
    scheduler: &FAIL_SCHED,
    extra_sched_args: &["--fail-verify"],
    duration_s: 5,
    workers_per_cgroup: 2,
    ..SttTestEntry::DEFAULT
};
