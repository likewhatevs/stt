//! Scenario execution engine.
//!
//! See the [Running Tests](https://likewhatevs.github.io/ktstr/guide/running-tests.html)
//! chapter of the guide.

use anyhow::{Context, Result};
use std::time::{Duration, Instant};

use crate::assert::ScenarioStats;
use crate::cgroup::CgroupManager;
use crate::probe::btf::discover_bpf_symbols;
use crate::probe::stack::{expand_bpf_to_kernel_callers, filter_traceable, load_probe_stack};
use crate::scenario::{self, Ctx, FlagProfile, Scenario, flags};
use crate::topology::TestTopology;

/// Full configuration for a scenario run session.
///
/// Controls flag selection, durations, and verification behavior.
/// The scheduler is managed externally -- `ktstr run` does not
/// start or stop schedulers.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub parent_cgroup: String,
    pub duration: Duration,
    pub workers_per_cgroup: usize,
    pub active_flags: Option<Vec<&'static str>>,
    pub repro: bool,
    /// Crash stack for auto-probe (file path or comma-separated function names).
    pub probe_stack: Option<String>,
    /// Auto-repro: crash -> extract stack -> rerun with probe-stack.
    pub auto_repro: bool,
    pub kernel_dir: Option<String>,
    /// Time to wait after cgroup creation for scheduler stabilization.
    pub settle: Duration,
    /// Sleep after cgroup cleanup before next scenario.
    pub cleanup: Duration,
    /// Override work_type for all swappable CgroupDefs and steady-state cgroups.
    pub work_type_override: Option<crate::workload::WorkType>,
    /// Caller-level assertion overrides merged onto `Assert::default_checks()`.
    pub assert: crate::assert::Assert,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            parent_cgroup: "/sys/fs/cgroup/ktstr".into(),
            duration: Duration::from_secs(20),
            workers_per_cgroup: 4,
            active_flags: None,
            repro: false,
            probe_stack: None,
            auto_repro: false,
            kernel_dir: None,
            settle: Duration::from_millis(500),
            cleanup: Duration::from_millis(200),
            work_type_override: None,
            assert: crate::assert::Assert::NONE,
        }
    }
}

/// Result of running a single scenario with a specific flag profile.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub passed: bool,
    pub duration_s: f64,
    pub details: Vec<String>,
    #[serde(default)]
    pub stats: ScenarioStats,
}

/// Expand scenarios into (scenario, profile) runs based on active_flags config.
///
/// - `None` active_flags: all profiles for each scenario
/// - `Some([])` (empty): single default profile
/// - `Some([f1, f2])`: single profile with those flags
///
/// Results are sorted by profile name.
pub fn expand_scenario_runs<'a>(
    scenarios: &[&'a Scenario],
    active_flags: &Option<Vec<&'static str>>,
) -> Vec<(&'a Scenario, FlagProfile)> {
    let mut runs = Vec::new();
    for s in scenarios {
        let profiles = match active_flags {
            None => s.profiles(),
            Some(flags) if flags.is_empty() => vec![FlagProfile { flags: vec![] }],
            Some(flags) => vec![FlagProfile {
                flags: flags.clone(),
            }],
        };
        for p in profiles {
            runs.push((*s, p));
        }
    }
    runs.sort_by_key(|a| a.1.name());
    runs
}

/// Orchestrates scenario execution.
///
/// Runs each scenario with the appropriate configuration. The
/// scheduler is managed externally -- the runner does not start
/// or stop scheduler processes.
pub struct Runner {
    /// Run configuration.
    pub config: RunConfig,
    /// Host CPU topology.
    pub topo: TestTopology,
}

impl Runner {
    /// Create a runner with the given configuration and topology.
    pub fn new(config: RunConfig, topo: TestTopology) -> Result<Self> {
        if config.repro {
            crate::workload::set_repro_mode(true);
        }
        Ok(Self { config, topo })
    }

    /// Run all scenarios with their flag profiles.
    pub fn run_scenarios(&self, scenarios: &[&Scenario]) -> Result<Vec<ScenarioResult>> {
        let runs = expand_scenario_runs(scenarios, &self.config.active_flags);

        let mut results = Vec::new();

        for (s, profile) in &runs {
            let qname = s.qualified_name(profile);

            let start = Instant::now();
            let cgroups = CgroupManager::new(&self.config.parent_cgroup);
            let needs_cpu_ctrl = !profile.flags.contains(&flags::NO_CTRL);
            cgroups.setup(needs_cpu_ctrl).context("cgroup setup")?;

            crate::workload::set_sched_pid(0);
            let ctx = Ctx {
                cgroups: &cgroups,
                topo: &self.topo,
                duration: self.config.duration,
                workers_per_cgroup: self.config.workers_per_cgroup,
                sched_pid: 0,
                settle: self.config.settle,
                work_type_override: self.config.work_type_override.clone(),
                assert: crate::assert::Assert::default_checks().merge(&self.config.assert),
                wait_for_map_write: false,
            };

            // Start BPF skeleton probes for auto-probe.
            let probe_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            type SkeletonHandle = (
                std::thread::JoinHandle<(
                    Option<Vec<crate::probe::process::ProbeEvent>>,
                    crate::probe::process::ProbeDiagnostics,
                )>,
                Vec<(u32, String)>,
                std::collections::HashMap<String, String>,
                std::collections::HashMap<String, Vec<(String, String)>>,
            );
            let mut skeleton_handle: Option<SkeletonHandle> = if self.config.repro
                && let Some(ref stack_input) = self.config.probe_stack
            {
                let mut functions = filter_traceable(load_probe_stack(stack_input));
                let stack_display_names: Vec<&str> = functions
                    .iter()
                    .filter(|f| f.is_bpf)
                    .map(|f| f.display_name.as_str())
                    .collect();
                let bpf_syms = discover_bpf_symbols(&stack_display_names);
                if !bpf_syms.is_empty() {
                    tracing::debug!(n = bpf_syms.len(), "auto-probe: BPF symbols discovered");
                    functions.extend(bpf_syms);
                }
                // Expand BPF functions to their kernel-side callers
                // so they can be probed via kprobe.
                let functions = expand_bpf_to_kernel_callers(functions);
                if functions.is_empty() {
                    tracing::warn!("auto-probe: no functions in stack input");
                    None
                } else {
                    let kernel_names: Vec<&str> = functions
                        .iter()
                        .filter(|f| !f.is_bpf)
                        .map(|f| f.raw_name.as_str())
                        .collect();
                    let btf_path =
                        crate::probe::btf::resolve_btf_path(self.config.kernel_dir.as_deref());
                    let mut btf_funcs = crate::probe::btf::parse_btf_functions(
                        &kernel_names,
                        btf_path.as_ref().and_then(|p| p.to_str()),
                    );
                    // Parse BPF function signatures from BPF program BTF.
                    let bpf_btf_args: Vec<(&str, u32)> = functions
                        .iter()
                        .filter(|f| f.is_bpf)
                        .filter_map(|f| Some((f.display_name.as_str(), f.bpf_prog_id?)))
                        .collect();
                    if !bpf_btf_args.is_empty() {
                        let bpf_btf = crate::probe::btf::parse_bpf_btf_functions(&bpf_btf_args);
                        btf_funcs.extend(bpf_btf);
                    }
                    let func_names: Vec<(u32, String)> = functions
                        .iter()
                        .enumerate()
                        .map(|(i, f)| (i as u32, f.display_name.clone()))
                        .collect();
                    // Resolve BPF source locations from program BTF.
                    let bpf_prog_ids: Vec<u32> =
                        functions.iter().filter_map(|f| f.bpf_prog_id).collect();
                    let bpf_locs = crate::probe::btf::resolve_bpf_source_locs(&bpf_prog_ids);
                    let bpf_fds = crate::probe::process::open_bpf_prog_fds(&functions);
                    let param_names = crate::probe::output::build_param_names(&btf_funcs);
                    let stop_clone = probe_stop.clone();
                    let probes_ready =
                        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let probes_ready_thread = probes_ready.clone();
                    let handle = std::thread::spawn(move || {
                        crate::probe::process::run_probe_skeleton(
                            &functions,
                            &btf_funcs,
                            &stop_clone,
                            &bpf_fds,
                            &probes_ready_thread,
                        )
                    });
                    while !probes_ready.load(std::sync::atomic::Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Some((handle, func_names, bpf_locs, param_names))
                }
            } else {
                None
            };

            tracing::info!(qname, "starting scenario");
            let res = scenario::run_scenario(s, &ctx);
            tracing::info!(qname, elapsed = ?start.elapsed(), "scenario complete");

            // Stop probes and collect results.
            let probe_output =
                if let Some((handle, func_names, bpf_locs, param_names)) = skeleton_handle.take() {
                    probe_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    handle.join().ok().and_then(|(events, diag)| {
                        let mut out = String::new();
                        // Format diagnostics summary.
                        let pipeline = crate::test_support::PipelineDiagnostics::default();
                        out.push_str(&crate::test_support::format_probe_diagnostics(
                            &pipeline, &diag,
                        ));
                        if let Some(events) = events {
                            out.push_str(&crate::probe::output::format_probe_events_with_bpf_locs(
                                &events,
                                &func_names,
                                self.config.kernel_dir.as_deref(),
                                &bpf_locs,
                                Some(self.topo.total_cpus() as u32),
                                &param_names,
                            ));
                        }
                        if out.is_empty() { None } else { Some(out) }
                    })
                } else {
                    None
                };

            let _ = cgroups.cleanup_all();
            std::thread::sleep(self.config.cleanup);

            let r = match res {
                Ok(mut v) => {
                    if let Some(output) = probe_output {
                        v.passed = false;
                        for line in output.lines() {
                            if !line.trim().is_empty() {
                                v.details.push(line.to_string());
                            }
                        }
                    }
                    ScenarioResult {
                        scenario_name: qname,
                        passed: v.passed,
                        duration_s: start.elapsed().as_secs_f64(),
                        details: v.details,
                        stats: v.stats,
                    }
                }
                Err(e) => ScenarioResult {
                    scenario_name: qname,
                    passed: false,
                    duration_s: start.elapsed().as_secs_f64(),
                    details: vec![format!("{e:#}")],
                    stats: Default::default(),
                },
            };
            results.push(r);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_result_serde_roundtrip() {
        let r = ScenarioResult {
            scenario_name: "test/default".into(),
            passed: false,
            duration_s: 15.5,
            details: vec!["unfair".into(), "stuck 3000ms".into()],
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 4,
                total_cpus: 8,
                total_migrations: 12,
                worst_spread: 25.0,
                worst_gap_ms: 3000,
                worst_gap_cpu: 5,
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ScenarioResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r.scenario_name, r2.scenario_name);
        assert_eq!(r.passed, r2.passed);
        assert_eq!(r.details, r2.details);
        assert_eq!(r.stats.worst_gap_ms, r2.stats.worst_gap_ms);
        assert_eq!(r.stats.total_workers, r2.stats.total_workers);
    }

    #[test]
    fn scenario_result_default_stats() {
        let json = r#"{"scenario_name":"t","passed":true,"duration_s":1.0,"details":[]}"#;
        let r: ScenarioResult = serde_json::from_str(json).unwrap();
        assert!(r.passed);
        assert_eq!(r.stats.total_workers, 0);
        assert_eq!(r.stats.cgroups.len(), 0);
    }

    #[test]
    fn scenario_result_with_cgroups() {
        let r = ScenarioResult {
            scenario_name: "proportional/default".into(),
            passed: true,
            duration_s: 20.0,
            details: vec![],
            stats: ScenarioStats {
                cgroups: vec![
                    crate::assert::CgroupStats {
                        num_workers: 4,
                        num_cpus: 4,
                        avg_off_cpu_pct: 75.0,
                        min_off_cpu_pct: 70.0,
                        max_off_cpu_pct: 80.0,
                        spread: 10.0,
                        max_gap_ms: 50,
                        max_gap_cpu: 0,
                        total_migrations: 3,
                        ..Default::default()
                    },
                    crate::assert::CgroupStats {
                        num_workers: 4,
                        num_cpus: 4,
                        avg_off_cpu_pct: 72.0,
                        min_off_cpu_pct: 68.0,
                        max_off_cpu_pct: 76.0,
                        spread: 8.0,
                        max_gap_ms: 30,
                        max_gap_cpu: 4,
                        total_migrations: 2,
                        ..Default::default()
                    },
                ],
                total_workers: 8,
                total_cpus: 8,
                total_migrations: 5,
                worst_spread: 10.0,
                worst_gap_ms: 50,
                worst_gap_cpu: 0,
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ScenarioResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.stats.cgroups.len(), 2);
        assert_eq!(r2.stats.cgroups[0].num_workers, 4);
        assert_eq!(r2.stats.cgroups[1].max_gap_cpu, 4);
    }

    #[test]
    fn run_config_cpu_controller_flag() {
        let profile_no_ctrl = FlagProfile {
            flags: vec![flags::NO_CTRL],
        };
        assert!(profile_no_ctrl.flags.contains(&flags::NO_CTRL));
        let needs_cpu_ctrl = !profile_no_ctrl.flags.contains(&flags::NO_CTRL);
        assert!(!needs_cpu_ctrl);

        let profile_default = FlagProfile { flags: vec![] };
        let needs_cpu_ctrl = !profile_default.flags.contains(&flags::NO_CTRL);
        assert!(needs_cpu_ctrl);
    }

    #[test]
    fn scenario_result_serde_special_chars() {
        let r = ScenarioResult {
            scenario_name: "test/with\"quotes".into(),
            passed: false,
            duration_s: 1.0,
            details: vec!["line with\nnewline".into(), "tab\there".into()],
            stats: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ScenarioResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.scenario_name, "test/with\"quotes");
        assert_eq!(r2.details.len(), 2);
    }

    #[test]
    fn scenario_result_serde_large_duration() {
        let r = ScenarioResult {
            scenario_name: "long_running".into(),
            passed: true,
            duration_s: 86400.123456,
            details: vec![],
            stats: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ScenarioResult = serde_json::from_str(&json).unwrap();
        assert!((r2.duration_s - 86400.123456).abs() < 0.001);
    }

    #[test]
    fn flag_profile_with_other_flags_still_needs_ctrl() {
        let profile = FlagProfile {
            flags: vec![flags::LLC, flags::BORROW],
        };
        let needs_cpu_ctrl = !profile.flags.contains(&flags::NO_CTRL);
        assert!(needs_cpu_ctrl);
    }

    #[test]
    fn scenario_result_serde_with_stats() {
        let r = ScenarioResult {
            scenario_name: "test/borrow".into(),
            passed: false,
            duration_s: 12.5,
            details: vec!["stuck 3000ms on cpu2".into(), "unfair cgroup".into()],
            stats: crate::assert::ScenarioStats {
                cgroups: vec![crate::assert::CgroupStats {
                    num_workers: 4,
                    num_cpus: 2,
                    avg_off_cpu_pct: 50.0,
                    min_off_cpu_pct: 40.0,
                    max_off_cpu_pct: 60.0,
                    spread: 20.0,
                    max_gap_ms: 3000,
                    max_gap_cpu: 2,
                    total_migrations: 7,
                    ..Default::default()
                }],
                total_workers: 4,
                total_cpus: 2,
                total_migrations: 7,
                worst_spread: 20.0,
                worst_gap_ms: 3000,
                worst_gap_cpu: 2,
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ScenarioResult = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.scenario_name, "test/borrow");
        assert!(!r2.passed);
        assert_eq!(r2.duration_s, 12.5);
        assert_eq!(r2.details.len(), 2);
        assert_eq!(r2.stats.total_workers, 4);
        assert_eq!(r2.stats.worst_gap_ms, 3000);
    }

    #[test]
    fn scenario_result_serde_empty_details_preserves_duration() {
        let r = ScenarioResult {
            scenario_name: "empty/default".into(),
            passed: true,
            duration_s: 0.0,
            details: vec![],
            stats: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let r2: ScenarioResult = serde_json::from_str(&json).unwrap();
        assert!(r2.details.is_empty());
        assert!(r2.passed);
        // Verify zero duration survives roundtrip (f64 edge case).
        assert_eq!(r2.duration_s, 0.0);
        // Verify stats default to zero, not garbage.
        assert_eq!(r2.stats.worst_spread, 0.0);
        assert_eq!(r2.stats.worst_gap_ms, 0);
    }

    #[test]
    fn scenario_result_serde_missing_stats_uses_default_values() {
        // JSON without "stats" field — serde #[serde(default)] should fill defaults.
        let json =
            r#"{"scenario_name":"missing_stats","passed":true,"duration_s":1.0,"details":["ok"]}"#;
        let r: ScenarioResult = serde_json::from_str(json).unwrap();
        assert_eq!(r.scenario_name, "missing_stats");
        assert!(r.passed);
        assert_eq!(r.details, vec!["ok"]);
        // Verify every stats field gets its Default value, not just total_workers.
        assert_eq!(r.stats.total_workers, 0);
        assert_eq!(r.stats.total_cpus, 0);
        assert_eq!(r.stats.total_migrations, 0);
        assert_eq!(r.stats.worst_spread, 0.0);
        assert_eq!(r.stats.worst_gap_ms, 0);
        assert_eq!(r.stats.worst_gap_cpu, 0);
        assert!(r.stats.cgroups.is_empty());
    }

    #[test]
    fn flag_profile_empty_disables_no_ctrl() {
        // Empty profile means cpu controller IS needed.
        let profile = FlagProfile { flags: vec![] };
        assert!(!profile.flags.contains(&flags::NO_CTRL));
        // Verify it also doesn't contain any other flag.
        assert!(profile.flags.is_empty());
        assert_eq!(profile.name(), "default");
    }

    #[test]
    fn runner_new_preserves_config() {
        let topo = TestTopology::from_spec(2, 4, 2);
        let config = RunConfig {
            duration: Duration::from_secs(30),
            workers_per_cgroup: 8,
            active_flags: Some(vec![flags::BORROW, flags::LLC]),
            cleanup: Duration::from_millis(300),
            ..Default::default()
        };
        let runner = Runner::new(config, topo).unwrap();
        assert_eq!(runner.topo.total_cpus(), 16);
        assert_eq!(runner.config.duration, Duration::from_secs(30));
        assert_eq!(runner.config.workers_per_cgroup, 8);
        assert_eq!(runner.config.settle, Duration::from_millis(500));
        assert_eq!(runner.config.cleanup, Duration::from_millis(300));
        assert_eq!(runner.config.active_flags.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn run_config_fields_carry_probe_and_repro() {
        let config = RunConfig {
            duration: Duration::from_secs(5),
            workers_per_cgroup: 2,
            repro: true,
            probe_stack: Some("do_enqueue_task,balance_one".into()),
            auto_repro: true,
            kernel_dir: Some("/usr/src/linux".into()),
            work_type_override: Some(crate::workload::WorkType::Mixed),
            ..Default::default()
        };
        assert!(config.repro);
        assert!(config.auto_repro);
        assert_eq!(
            config.probe_stack.as_deref(),
            Some("do_enqueue_task,balance_one")
        );
        assert_eq!(config.kernel_dir.as_deref(), Some("/usr/src/linux"));
        assert!(matches!(
            config.work_type_override,
            Some(crate::workload::WorkType::Mixed)
        ));
    }

    #[test]
    fn scenario_result_debug_shows_field_values() {
        let r = ScenarioResult {
            scenario_name: "proportional/borrow".into(),
            passed: false,
            duration_s: 15.5,
            details: vec!["stuck 3000ms".into()],
            stats: Default::default(),
        };
        let s = format!("{:?}", r);
        assert!(s.contains("proportional/borrow"), "must show scenario_name");
        assert!(s.contains("15.5"), "must show duration_s value");
        assert!(s.contains("stuck 3000ms"), "must show detail contents");
        assert!(s.contains("false"), "must show passed=false");
    }

    #[test]
    fn run_config_debug_shows_field_values() {
        let config = RunConfig {
            duration: Duration::from_secs(30),
            ..Default::default()
        };
        let s = format!("{:?}", config);
        assert!(s.contains("30"), "must show duration value");
        assert!(s.contains("4"), "must show workers_per_cgroup value");
    }

    #[test]
    fn run_config_clone_preserves_all_fields() {
        let config = RunConfig {
            parent_cgroup: "/sys/fs/cgroup/ktstr".into(),
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            active_flags: Some(vec![flags::LLC]),
            repro: true,
            probe_stack: Some("func1".into()),
            auto_repro: true,
            kernel_dir: Some("/path".into()),
            settle: Duration::from_millis(500),
            cleanup: Duration::from_millis(100),
            work_type_override: Some(crate::workload::WorkType::CpuSpin),
            assert: crate::assert::Assert::NONE.max_gap_ms(5000),
        };
        let c2 = config.clone();
        assert_eq!(c2.duration, config.duration);
        assert_eq!(c2.workers_per_cgroup, config.workers_per_cgroup);
        assert_eq!(c2.repro, config.repro);
        assert_eq!(c2.auto_repro, config.auto_repro);
        assert_eq!(c2.probe_stack, config.probe_stack);
        assert_eq!(c2.kernel_dir, config.kernel_dir);
        assert_eq!(c2.settle, config.settle);
    }

    #[test]
    fn scenario_result_clone_preserves_all_fields() {
        let r = ScenarioResult {
            scenario_name: "test/borrow".into(),
            passed: false,
            duration_s: 12.5,
            details: vec!["err1".into(), "err2".into()],
            stats: ScenarioStats {
                cgroups: vec![],
                total_workers: 4,
                total_cpus: 8,
                total_migrations: 12,
                worst_spread: 25.0,
                worst_gap_ms: 3000,
                worst_gap_cpu: 5,
                ..Default::default()
            },
        };
        let r2 = r.clone();
        assert_eq!(r2.scenario_name, "test/borrow");
        assert!(!r2.passed);
        assert_eq!(r2.duration_s, 12.5);
        assert_eq!(r2.details, vec!["err1", "err2"]);
        assert_eq!(r2.stats.total_workers, 4);
        assert_eq!(r2.stats.worst_gap_ms, 3000);
        assert_eq!(r2.stats.worst_gap_cpu, 5);
    }

    // -- expand_scenario_runs tests --

    #[test]
    fn expand_runs_none_flags_uses_all_profiles() {
        let scenarios = scenario::all_scenarios();
        let first = &scenarios[0];
        let refs: Vec<&Scenario> = vec![first];
        let runs = expand_scenario_runs(&refs, &None);
        // None = all profiles
        assert_eq!(runs.len(), first.profiles().len());
    }

    #[test]
    fn expand_runs_empty_flags_single_default() {
        let scenarios = scenario::all_scenarios();
        let first = &scenarios[0];
        let refs: Vec<&Scenario> = vec![first];
        let runs = expand_scenario_runs(&refs, &Some(vec![]));
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1.name(), "default");
    }

    #[test]
    fn expand_runs_specific_flags() {
        let scenarios = scenario::all_scenarios();
        let first = &scenarios[0];
        let refs: Vec<&Scenario> = vec![first];
        let runs = expand_scenario_runs(&refs, &Some(vec![flags::BORROW]));
        assert_eq!(runs.len(), 1);
        assert!(runs[0].1.flags.contains(&flags::BORROW));
    }

    #[test]
    fn expand_runs_multiple_scenarios() {
        let scenarios = scenario::all_scenarios();
        let refs: Vec<&Scenario> = scenarios.iter().take(3).collect();
        let runs = expand_scenario_runs(&refs, &Some(vec![]));
        // Each scenario with default profile -> 3 runs
        assert_eq!(runs.len(), 3);
    }

    #[test]
    fn expand_runs_sorted_by_profile_name() {
        let scenarios = scenario::all_scenarios();
        let first = &scenarios[0];
        let refs: Vec<&Scenario> = vec![first];
        let runs = expand_scenario_runs(&refs, &None);
        for w in runs.windows(2) {
            assert!(w[0].1.name() <= w[1].1.name());
        }
    }

    #[test]
    fn expand_runs_empty_scenarios() {
        let refs: Vec<&Scenario> = vec![];
        let runs = expand_scenario_runs(&refs, &None);
        assert!(runs.is_empty());
    }

    #[test]
    fn expand_runs_two_flags() {
        let scenarios = scenario::all_scenarios();
        let first = &scenarios[0];
        let refs: Vec<&Scenario> = vec![first];
        let runs = expand_scenario_runs(&refs, &Some(vec![flags::LLC, flags::BORROW]));
        assert_eq!(runs.len(), 1);
        assert!(runs[0].1.flags.contains(&flags::LLC));
        assert!(runs[0].1.flags.contains(&flags::BORROW));
        assert_eq!(runs[0].1.name(), "llc+borrow");
    }

    #[test]
    fn expand_runs_scenario_name_preserved() {
        let scenarios = scenario::all_scenarios();
        let first = &scenarios[0];
        let refs: Vec<&Scenario> = vec![first];
        let runs = expand_scenario_runs(&refs, &Some(vec![]));
        assert_eq!(runs[0].0.name, first.name);
    }
}
