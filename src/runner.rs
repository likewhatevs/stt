//! Scenario execution engine with scheduler lifecycle management.
//!
//! See the [Running Tests](https://likewhatevs.github.io/ktstr/guide/running-tests.html)
//! chapter of the guide.

use anyhow::{Context, Result, bail};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::assert::ScenarioStats;
use crate::cgroup::CgroupManager;
use crate::probe::btf::discover_bpf_symbols;
use crate::probe::stack::{
    expand_bpf_to_kernel_callers, extract_stack_functions_all, filter_traceable, load_probe_stack,
};
use crate::scenario::{self, Ctx, FlagProfile, Scenario, flags};
use crate::topology::TestTopology;

/// Full configuration for a scenario run session.
///
/// Controls scheduler binary, flag selection, durations, and
/// verification behavior.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub scheduler_bin: Option<String>,
    pub scheduler_args: Vec<String>,
    pub parent_cgroup: String,
    pub duration: Duration,
    pub workers_per_cgroup: usize,
    pub json: bool,
    pub verbose: bool,
    pub active_flags: Option<Vec<&'static str>>,
    pub repro: bool,
    /// Crash stack for auto-probe (file path or comma-separated function names).
    pub probe_stack: Option<String>,
    /// Auto-repro: crash -> extract stack -> rerun with probe-stack.
    pub auto_repro: bool,
    pub kernel_dir: Option<String>,
    /// Time to wait after cgroup creation for scheduler stabilization.
    pub settle: Duration,
    /// Sleep after scheduler process start to let it initialize.
    pub scheduler_startup: Duration,
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
            scheduler_bin: None,
            scheduler_args: Vec::new(),
            parent_cgroup: "/sys/fs/cgroup/ktstr".into(),
            duration: Duration::from_secs(20),
            workers_per_cgroup: 4,
            json: false,
            verbose: false,
            active_flags: None,
            repro: false,
            probe_stack: None,
            auto_repro: false,
            kernel_dir: None,
            settle: Duration::from_millis(500),
            scheduler_startup: Duration::from_millis(2000),
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

/// Extract auto-repro function names from scenario result details.
///
/// Looks for a "functions:" line first, then falls back to stack extraction.
/// Returns `None` if no function names are found.
pub fn extract_auto_repro_functions(results: &[ScenarioResult]) -> Option<String> {
    let all_text: String = results
        .iter()
        .flat_map(|r| r.details.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    all_text
        .lines()
        .find(|l| l.contains("functions:"))
        .map(|l| {
            l.split("functions:")
                .nth(1)
                .unwrap_or("")
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(",")
        })
        .or_else(|| {
            let fns = crate::probe::stack::extract_stack_function_names(&all_text);
            if fns.is_empty() {
                None
            } else {
                Some(fns.join(","))
            }
        })
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

/// Orchestrates scenario execution with scheduler lifecycle management.
///
/// Starts/stops the scheduler process as needed when flag profiles
/// change, and runs each scenario with the appropriate configuration.
pub struct Runner {
    /// Run configuration.
    pub config: RunConfig,
    /// VM CPU topology.
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

    /// Run all scenarios with their flag profiles. Manages scheduler
    /// process lifecycle between profile changes.
    pub fn run_scenarios(&self, scenarios: &[&Scenario]) -> Result<Vec<ScenarioResult>> {
        let runs = expand_scenario_runs(scenarios, &self.config.active_flags);

        let mut results = Vec::new();
        let mut cur_profile = String::new();
        let mut sched: Option<SchedulerProcess> = None;

        for (s, profile) in &runs {
            let qname = s.qualified_name(profile);
            let pname = profile.name();

            let start = Instant::now();
            let cgroups = CgroupManager::new(&self.config.parent_cgroup);
            let needs_cpu_ctrl = !profile.flags.contains(&flags::NO_CTRL);
            cgroups.setup(needs_cpu_ctrl).context("cgroup setup")?;

            if pname != cur_profile {
                if let Some(mut p) = sched.take() {
                    p.stop();
                }
                if let Some(ref bin) = self.config.scheduler_bin {
                    let args = self.config.scheduler_args.clone();
                    tracing::info!(bin = %bin, ?args, "starting scheduler");
                    let mut p = SchedulerProcess::start(bin, &args)?;
                    std::thread::sleep(self.config.scheduler_startup);
                    if p.is_dead() {
                        let _ = cgroups.cleanup_all();
                        bail!("scheduler exited immediately");
                    }
                    tracing::info!("scheduler running");
                    sched = Some(p);
                }
                cur_profile = pname;
            }

            let sched_pid = sched.as_ref().map(|s| s.pid()).unwrap_or(0);
            crate::workload::set_sched_pid(sched_pid as i32);
            let ctx = Ctx {
                cgroups: &cgroups,
                topo: &self.topo,
                duration: self.config.duration,
                workers_per_cgroup: self.config.workers_per_cgroup,
                sched_pid,
                settle: self.config.settle,
                work_type_override: self.config.work_type_override.clone(),
                assert: crate::assert::Assert::default_checks().merge(&self.config.assert),
                wait_for_map_write: false,
            };

            // Start BPF skeleton probes for auto-probe.
            let probe_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            type SkeletonHandle = (
                std::thread::JoinHandle<Option<Vec<crate::probe::process::ProbeEvent>>>,
                Vec<(u32, String)>,
                std::collections::HashMap<String, String>,
            );
            let mut skeleton_handle: Option<SkeletonHandle> = if self.config.repro
                && let Some(ref stack_input) = self.config.probe_stack
            {
                let mut functions = filter_traceable(load_probe_stack(stack_input));
                let bpf_syms = discover_bpf_symbols();
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
                    let stop_clone = probe_stop.clone();
                    let handle = std::thread::spawn(move || {
                        crate::probe::process::run_probe_skeleton(
                            &functions,
                            &btf_funcs,
                            "scx_disable_workfn",
                            &stop_clone,
                        )
                    });
                    std::thread::sleep(Duration::from_secs(2));
                    Some((handle, func_names, bpf_locs))
                }
            } else {
                None
            };

            tracing::info!(qname, "starting scenario");
            let res = scenario::run_scenario(s, &ctx);
            tracing::info!(qname, elapsed = ?start.elapsed(), "scenario complete");

            // Stop probes and collect results.
            let probe_output = if let Some((handle, func_names, bpf_locs)) = skeleton_handle.take()
            {
                probe_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                handle.join().ok().flatten().map(|events| {
                    crate::probe::output::format_probe_events_with_bpf_locs(
                        &events,
                        &func_names,
                        self.config.kernel_dir.as_deref(),
                        &bpf_locs,
                    )
                })
            } else {
                None
            };

            let sched_dead = sched.as_mut().map(|s| s.is_dead()).unwrap_or(false);
            let elapsed_secs = start.elapsed().as_secs_f64();
            if sched_dead {
                tracing::warn!(
                    qname,
                    elapsed_s = format!("{elapsed_secs:.1}"),
                    "scheduler crashed"
                );
            }

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
                    if sched_dead {
                        v.passed = false;
                        v.details.push(format!(
                            "scheduler crashed ({:.1}s into test)",
                            elapsed_secs,
                        ));
                    }
                    // On failure: kill scheduler so it writes exit dump, then read it
                    if !v.passed {
                        if let Some(mut s) = sched.take() {
                            s.stop();
                            std::thread::sleep(Duration::from_millis(100));
                            let dump = s.read_stderr();
                            if !dump.is_empty() {
                                let is_autoprobe =
                                    self.config.probe_stack.is_some() || self.config.auto_repro;
                                for line in dump.lines() {
                                    if line.trim().is_empty() {
                                        continue;
                                    }
                                    if is_autoprobe && !self.config.verbose {
                                        if line.contains("runtime error")
                                            || line.contains("EXIT:")
                                            || line.contains("Error:")
                                            || line.starts_with("CELL[")
                                            || line.starts_with("  CELL[")
                                            || line.starts_with("CPU[")
                                            || line.starts_with("  CPU[")
                                        {
                                            v.details.push(line.to_string());
                                        }
                                    } else {
                                        v.details.push(line.to_string());
                                    }
                                }
                                // Extract stack from the FULL dump (not the
                                // filtered details) for auto-probe rerun
                                if self.config.repro && self.config.probe_stack.is_none() {
                                    let stack_fns = extract_stack_functions_all(&dump);
                                    if !stack_fns.is_empty() {
                                        let stack_path = std::env::temp_dir().join(format!(
                                            "ktstr-crash-stack-{}.txt",
                                            std::process::id()
                                        ));
                                        let stack_text: String = stack_fns
                                            .iter()
                                            .map(|f| f.raw_name.as_str())
                                            .collect::<Vec<_>>()
                                            .join("\n");
                                        let _ = std::fs::write(&stack_path, &stack_text);
                                        let names: Vec<&str> = stack_fns
                                            .iter()
                                            .map(|f| f.display_name.as_str())
                                            .collect();
                                        v.details.push(format!(
                                            "auto-probe: rerun with --probe-stack {}",
                                            stack_path.display()
                                        ));
                                        v.details
                                            .push(format!("  functions: {}", names.join(", ")));
                                    }
                                }
                            }
                        }
                        cur_profile.clear();
                    } else if sched_dead {
                        sched.take();
                        cur_profile.clear();
                    }
                    ScenarioResult {
                        scenario_name: qname,
                        passed: v.passed,
                        duration_s: start.elapsed().as_secs_f64(),
                        details: v.details,
                        stats: v.stats,
                    }
                }
                Err(e) => {
                    let mut details = vec![format!("{e:#}")];
                    if let Some(mut s) = sched.take() {
                        s.stop();
                        std::thread::sleep(Duration::from_millis(100));
                        let dump = s.read_stderr();
                        for line in dump.lines() {
                            if !line.trim().is_empty() {
                                details.push(line.to_string());
                            }
                        }
                    }
                    cur_profile.clear();
                    ScenarioResult {
                        scenario_name: qname,
                        passed: false,
                        duration_s: start.elapsed().as_secs_f64(),
                        details,
                        stats: Default::default(),
                    }
                }
            };
            results.push(r);
        }

        if let Some(mut p) = sched.take() {
            p.stop();
        }
        Ok(results)
    }
}

/// RAII handle to a running scheduler process.
pub struct SchedulerProcess {
    child: Child,
    stderr_path: std::path::PathBuf,
}

impl SchedulerProcess {
    fn start(bin: &str, args: &[String]) -> Result<Self> {
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-sched-{}.log", std::process::id()));
        let stderr_file = std::fs::File::create(&stderr_path)?;
        let child = Command::new(bin)
            .args(args)
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .with_context(|| format!("spawn {bin}"))?;
        Ok(Self { child, stderr_path })
    }
    /// PID of the scheduler process.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    /// Read scheduler output (includes watchdog dumps on stall exit).
    pub fn read_stderr(&self) -> String {
        std::fs::read_to_string(&self.stderr_path).unwrap_or_default()
    }
    /// Check if the scheduler has exited.
    pub fn is_dead(&mut self) -> bool {
        self.child.try_wait().ok().flatten().is_some()
    }
    fn stop(&mut self) {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        let _ = kill(Pid::from_raw(self.child.id() as i32), Signal::SIGTERM);
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            if Instant::now() > deadline {
                let _ = self.child.kill();
                let _ = self.child.wait();
                return;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for SchedulerProcess {
    fn drop(&mut self) {
        self.stop();
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
    fn scheduler_process_stop_terminates() {
        // Spawn a long-running process, wrap in SchedulerProcess, call stop(),
        // verify it terminates within a reasonable time (SIGTERM -> poll -> SIGKILL).
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-stop-{}.log", std::process::id()));
        let stderr_file = std::fs::File::create(&stderr_path).unwrap();
        let child = Command::new("sleep")
            .arg("999")
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .unwrap();
        let mut proc = SchedulerProcess {
            child,
            stderr_path: stderr_path.clone(),
        };

        // Process should be alive before stop.
        assert!(!proc.is_dead(), "process should be alive before stop");

        let before = Instant::now();
        proc.stop();
        let elapsed = before.elapsed();

        // stop() sends SIGTERM then polls for 3s then SIGKILL.
        // sleep handles SIGTERM by exiting, so it should complete quickly.
        assert!(
            elapsed < Duration::from_secs(5),
            "stop() took too long: {elapsed:?}"
        );

        // Process must be dead after stop.
        assert!(proc.is_dead(), "process should be dead after stop");

        // Clean up.
        let _ = std::fs::remove_file(&stderr_path);
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
    fn scheduler_process_read_stderr_empty() {
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-stderr-{}.log", std::process::id()));
        std::fs::write(&stderr_path, "").unwrap();
        let child = Command::new("true")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let proc = SchedulerProcess {
            child,
            stderr_path: stderr_path.clone(),
        };
        let stderr = proc.read_stderr();
        assert!(stderr.is_empty());
        let _ = std::fs::remove_file(&stderr_path);
    }

    #[test]
    fn scheduler_process_read_stderr_content() {
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-stderr2-{}.log", std::process::id()));
        std::fs::write(&stderr_path, "error: scheduler died").unwrap();
        let child = Command::new("true")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let proc = SchedulerProcess {
            child,
            stderr_path: stderr_path.clone(),
        };
        let stderr = proc.read_stderr();
        assert_eq!(stderr, "error: scheduler died");
        let _ = std::fs::remove_file(&stderr_path);
    }

    #[test]
    fn scheduler_process_pid() {
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-pid-{}.log", std::process::id()));
        let stderr_file = std::fs::File::create(&stderr_path).unwrap();
        let child = Command::new("sleep")
            .arg("999")
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .unwrap();
        let expected_pid = child.id();
        let mut proc = SchedulerProcess {
            child,
            stderr_path: stderr_path.clone(),
        };
        assert_eq!(proc.pid(), expected_pid);
        assert!(!proc.is_dead());
        proc.stop();
        let _ = std::fs::remove_file(&stderr_path);
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
    fn scheduler_process_start_nonexistent_fails() {
        let result = SchedulerProcess::start("__nonexistent_scheduler_binary_xyz__", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn scheduler_process_start_and_immediate_death() {
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-death-{}.log", std::process::id()));
        let stderr_file = std::fs::File::create(&stderr_path).unwrap();
        let child = Command::new("false") // exits immediately with code 1
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .unwrap();
        let mut proc = SchedulerProcess {
            child,
            stderr_path: stderr_path.clone(),
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(proc.is_dead(), "'false' should exit immediately");
        let _ = std::fs::remove_file(&stderr_path);
    }

    #[test]
    fn scheduler_process_drop_stops_child() {
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-drop-{}.log", std::process::id()));
        let stderr_file = std::fs::File::create(&stderr_path).unwrap();
        let child = Command::new("sleep")
            .arg("999")
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .unwrap();
        let pid = child.id();
        {
            let _proc = SchedulerProcess {
                child,
                stderr_path: stderr_path.clone(),
            };
            // proc drops here, calling stop()
        }
        // Verify process is gone by trying to send signal 0.
        let kill_result = unsafe { libc::kill(pid as i32, 0) };
        assert_eq!(kill_result, -1, "process should be gone after drop");
        let _ = std::fs::remove_file(&stderr_path);
    }

    #[test]
    fn scheduler_process_read_stderr_nonexistent_path() {
        let child = Command::new("true")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let proc = SchedulerProcess {
            child,
            stderr_path: std::path::PathBuf::from("/nonexistent/path.log"),
        };
        assert!(proc.read_stderr().is_empty());
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
            scheduler_bin: Some("scx_mitosis".into()),
            scheduler_args: vec!["--verbose".into()],
            duration: Duration::from_secs(30),
            workers_per_cgroup: 8,
            json: true,
            verbose: true,
            active_flags: Some(vec![flags::BORROW, flags::LLC]),
            cleanup: Duration::from_millis(300),
            ..Default::default()
        };
        let runner = Runner::new(config, topo).unwrap();
        // Verify topology was correctly propagated (2*4*2=16 CPUs).
        assert_eq!(runner.topo.total_cpus(), 16);
        // Verify config fields survived construction.
        assert_eq!(runner.config.scheduler_bin.as_deref(), Some("scx_mitosis"));
        assert_eq!(runner.config.scheduler_args, vec!["--verbose"]);
        assert_eq!(runner.config.duration, Duration::from_secs(30));
        assert_eq!(runner.config.workers_per_cgroup, 8);
        assert!(runner.config.json);
        assert!(runner.config.verbose);
        assert_eq!(runner.config.settle, Duration::from_millis(500));
        assert_eq!(runner.config.scheduler_startup, Duration::from_millis(2000));
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
            scheduler_bin: Some("scx_mitosis".into()),
            duration: Duration::from_secs(30),
            verbose: true,
            ..Default::default()
        };
        let s = format!("{:?}", config);
        assert!(s.contains("scx_mitosis"), "must show scheduler_bin value");
        assert!(s.contains("30"), "must show duration value");
        assert!(s.contains("4"), "must show workers_per_cgroup value");
    }

    #[test]
    fn run_config_clone_preserves_all_fields() {
        let config = RunConfig {
            scheduler_bin: Some("scx_mitosis".into()),
            scheduler_args: vec!["--arg".into()],
            parent_cgroup: "/sys/fs/cgroup/ktstr".into(),
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            json: true,
            verbose: true,
            active_flags: Some(vec![flags::LLC]),
            repro: true,
            probe_stack: Some("func1".into()),
            auto_repro: true,
            kernel_dir: Some("/path".into()),
            settle: Duration::from_millis(500),
            scheduler_startup: Duration::from_millis(2000),
            cleanup: Duration::from_millis(100),
            work_type_override: Some(crate::workload::WorkType::CpuSpin),
            assert: crate::assert::Assert::NONE.max_gap_ms(5000),
        };
        let c2 = config.clone();
        assert_eq!(c2.scheduler_bin, config.scheduler_bin);
        assert_eq!(c2.scheduler_args, config.scheduler_args);
        assert_eq!(c2.duration, config.duration);
        assert_eq!(c2.workers_per_cgroup, config.workers_per_cgroup);
        assert_eq!(c2.json, config.json);
        assert_eq!(c2.verbose, config.verbose);
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

    #[test]
    fn scheduler_process_stop_then_read_stderr_empty() {
        // Exercises the combined stop+read path: after stopping a process
        // that wrote nothing to stderr, read_stderr returns empty.
        let stderr_path =
            std::env::temp_dir().join(format!("ktstr-test-stop-read-{}.log", std::process::id()));
        let stderr_file = std::fs::File::create(&stderr_path).unwrap();
        let child = Command::new("sleep")
            .arg("999")
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .unwrap();
        let pid = child.id();
        let mut proc = SchedulerProcess {
            child,
            stderr_path: stderr_path.clone(),
        };
        assert_eq!(proc.pid(), pid);
        assert!(!proc.is_dead());
        proc.stop();
        assert!(proc.is_dead(), "must be dead after stop");
        let stderr = proc.read_stderr();
        assert!(stderr.is_empty(), "sleep writes nothing to stderr");
        let _ = std::fs::remove_file(&stderr_path);
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
