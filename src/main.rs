use std::io::Write;

use anyhow::Result;
use clap::Parser;
use console::style;

use runner::{RunConfig, Runner};
use stats::{GauntletMonitorData, VmRunResult};
#[cfg(test)]
use stt::monitor;
use stt::{
    cgroup, probe, runner, scenario, stats, test_support, topology, verify, vm, vmm, workload,
};
use topology::TestTopology;

#[derive(Debug, Parser)]
#[clap(name = "stt", about = "stt - scheduler test tools")]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Debug, Parser)]
enum Command {
    /// Run test scenarios (guest-side dispatch, not for direct use)
    #[clap(hide = true)]
    Run(RunArgs),
    /// Launch VM(s) and run tests inside
    Vm(VmArgs),
    /// Probe kernel functions from a crash stack
    Probe(ProbeArgs),
    /// List available scenarios
    List,
    /// Show CPU topology
    Topo,
    /// Clean up test cgroups
    Cleanup(CleanupArgs),
    /// Run integration tests via nextest with sidecar collection
    Test(TestArgs),
}

#[derive(Debug, Parser)]
struct ProbeArgs {
    /// Crash stack source: file path, or "-" for stdin
    input: Option<String>,
    /// Grab latest sched_ext crash from dmesg
    #[clap(long)]
    dmesg: bool,
    /// Probe specific functions (comma-separated)
    #[clap(long)]
    functions: Option<String>,
    /// Path to linux source tree (for symbolization)
    #[clap(long)]
    kernel_dir: Option<String>,
    /// Include bootlin URLs in output
    #[clap(long)]
    bootlin: bool,
    /// Trigger function (default: scx_exit for sched_ext, or specify custom)
    #[clap(long)]
    trigger: Option<String>,
}

#[derive(Debug, Parser)]
struct RunArgs {
    scenarios: Vec<String>,
    #[clap(long)]
    all: bool,
    /// Scheduler binary path (omit to run without a scheduler)
    #[clap(long)]
    scheduler_bin: Option<String>,
    /// Extra scheduler arguments (repeatable)
    #[clap(long)]
    scheduler_arg: Vec<String>,
    #[clap(long, hide = true)]
    mitosis_bin: Option<String>,
    #[clap(long, default_value = "/sys/fs/cgroup/stt")]
    parent_cgroup: String,
    #[clap(long, default_value = "15")]
    duration_s: u64,
    #[clap(long, default_value = "4")]
    workers: usize,
    #[clap(long)]
    json: bool,
    #[clap(long)]
    verbose: bool,
    #[clap(long, conflicts_with = "flags")]
    all_flags: bool,
    #[clap(long, value_delimiter = ',')]
    flags: Vec<String>,
    /// Log unfairness but don't fail on it
    #[clap(long)]
    warn_unfair: bool,
    /// Reproducer mode: extend watchdog, disable dump trigger, run
    /// BPF assertion probes to catch invariant violations.
    #[clap(long)]
    repro: bool,
    /// Auto-probe: crash stack trace (file path or comma-separated function
    /// names). Attaches BPF kprobes that capture arguments at each
    /// function in the crash chain. Implies --repro.
    #[clap(long)]
    probe_stack: Option<String>,
    /// Auto-repro: crash once to get the stack, then automatically rerun
    /// with --probe-stack to capture arguments at each function. Implies --repro.
    #[clap(long, conflicts_with = "probe_stack")]
    auto_repro: bool,
    /// Include bootlin URLs in source line output
    #[clap(long)]
    bootlin: bool,
    /// Path to linux source tree (for kernel boot and symbolization)
    #[clap(long)]
    kernel_dir: Option<String>,
    /// Override work_type for all swappable cgroups (e.g. CpuSpin, Bursty, PipeIo)
    #[clap(long)]
    work_type: Option<String>,
}

#[derive(Debug, Parser)]
struct VmArgs {
    #[clap(long)]
    kernel: Option<String>,
    #[clap(long, default_value = "2")]
    sockets: usize,
    #[clap(long, default_value = "2")]
    cores: usize,
    #[clap(long, default_value = "2")]
    threads: usize,
    #[clap(long, default_value = "4096")]
    memory_mb: usize,
    #[clap(long)]
    gauntlet: bool,
    #[clap(long)]
    parallel: Option<usize>,
    /// Max infra-failure retries per VM (default: 3)
    #[clap(long, default_value = "3")]
    retries: usize,
    /// Flags to enable for gauntlet runs (comma-separated short names).
    /// Without this, gauntlet uses each scenario's default profiles.
    #[clap(long, value_delimiter = ',')]
    flags: Vec<String>,
    /// Scheduler binary path for gauntlet (required for gauntlet mode)
    #[clap(long)]
    scheduler_bin: Option<String>,
    /// Replica multiplier for each (topology, scenario, flags) cell.
    #[clap(long, default_value = "1")]
    replicas: usize,
    /// Linux source tree with built kernel (boots this instead of host kernel)
    #[clap(long)]
    kernel_dir: Option<String>,
    /// Save gauntlet results as a baseline JSON file for later comparison.
    #[clap(long)]
    save_baseline: Option<String>,
    /// Compare current gauntlet run against a saved baseline JSON file.
    #[clap(long)]
    compare: Option<String>,
    /// Work types for gauntlet dimension (comma-separated, e.g. CpuSpin,Bursty,PipeIo).
    /// Without this, gauntlet uses each scenario's default work type (no override).
    #[clap(long, value_delimiter = ',')]
    work_types: Vec<String>,
    #[clap(last = true)]
    run_args: Vec<String>,
}

#[derive(Debug, Parser)]
struct CleanupArgs {
    #[clap(long, default_value = "/sys/fs/cgroup/stt")]
    parent_cgroup: String,
}

#[derive(Debug, Parser)]
struct TestArgs {
    /// Filter expression passed to nextest (or cargo test)
    #[clap(long)]
    filter: Option<String>,
    /// Save results as a baseline JSON file
    #[clap(long)]
    save_baseline: Option<String>,
    /// Compare results against a saved baseline JSON file
    #[clap(long)]
    compare: Option<String>,
    /// Nextest profile (e.g. ci, quick)
    #[clap(long, default_value = "default")]
    nextest_profile: String,
    /// Path to a kernel image for VM-based tests
    #[clap(long)]
    kernel: Option<String>,
    /// Override scheduler binary path
    #[clap(long)]
    scheduler_bin: Option<String>,
    /// Skip post-run sidecar analysis
    #[clap(long)]
    no_analysis: bool,
    /// Extra arguments passed through to nextest (or cargo test)
    #[clap(last = true)]
    nextest_args: Vec<String>,
}

/// Split run_args into scenario names and extra CLI options.
///
/// Bare words are scenario names; tokens starting with `-` are options.
/// For `--key value` (no `=`), the next non-`-` token is consumed as
/// the option's value.
fn split_run_args(run_args: &[String]) -> (Vec<String>, Vec<String>) {
    let mut scenario_names = Vec::new();
    let mut extra_args = Vec::new();
    let mut iter = run_args.iter().peekable();
    while let Some(a) = iter.next() {
        if a.starts_with('-') {
            extra_args.push(a.clone());
            if !a.contains('=')
                && let Some(v) = iter.peek()
                && !v.starts_with('-')
                && let Some(val) = iter.next()
            {
                extra_args.push(val.clone());
            }
        } else {
            scenario_names.push(a.clone());
        }
    }
    (scenario_names, extra_args)
}

/// Extract auto-repro function names from scenario result details.
///
/// Looks for a "functions:" line first, then falls back to stack extraction.
/// Returns `None` if no function names are found.
fn extract_auto_repro_functions(results: &[runner::ScenarioResult]) -> Option<String> {
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
            let fns = runner::extract_stack_functions_all_pub(&all_text);
            if fns.is_empty() {
                None
            } else {
                Some(fns.join(","))
            }
        })
}

/// Build the gauntlet job matrix from presets, scenarios, flags, and work types.
#[allow(clippy::too_many_arguments)]
fn build_gauntlet_jobs(
    presets: &[vm::TopoPreset],
    scenarios: &[scenario::Scenario],
    fixed_profile: Option<&scenario::FlagProfile>,
    work_type_names: &[&str],
    replicas: usize,
    kernel: &Option<String>,
    kernel_dir: &Option<String>,
    scheduler_bin: &Option<String>,
    retries: usize,
) -> Vec<VmJob> {
    let mut jobs = Vec::new();
    for p in presets {
        for s in scenarios {
            let profiles = if let Some(fp) = fixed_profile {
                vec![fp.clone()]
            } else {
                s.profiles()
            };
            for prof in profiles {
                let mut base_stt_args = vec![
                    "run".to_string(),
                    "--json".to_string(),
                    "--duration-s".to_string(),
                    "20".to_string(),
                    s.name.to_string(),
                ];
                if let Some(bin) = scheduler_bin {
                    base_stt_args.push("--scheduler-bin".to_string());
                    base_stt_args.push(bin.clone());
                }
                if !prof.flags.is_empty() {
                    base_stt_args.push(format!("--flags={}", prof.flags.join(",")));
                }

                let wt_variants: Vec<Option<&str>> = if work_type_names.is_empty() {
                    vec![None]
                } else {
                    work_type_names.iter().map(|n| Some(*n)).collect()
                };

                for wt in &wt_variants {
                    let mut stt_args = base_stt_args.clone();
                    let cell_label = if let Some(wt_name) = wt {
                        stt_args.push(format!("--work-type={wt_name}"));
                        format!("{}/{}/{}/{}", p.name, s.name, prof.name(), wt_name)
                    } else {
                        format!("{}/{}/{}", p.name, s.name, prof.name())
                    };
                    for replica in 0..replicas {
                        let label = if replicas > 1 {
                            format!("{}#{}", cell_label, replica + 1)
                        } else {
                            cell_label.clone()
                        };
                        jobs.push(VmJob {
                            label,
                            topo: p.topology,
                            mem: p.memory_mb,
                            stt_args: stt_args.clone(),
                            kernel: kernel.clone(),
                            kernel_dir: kernel_dir.clone(),
                            retries,
                        });
                    }
                }
            }
        }
    }
    jobs
}

/// Truncate a string to at most `max_bytes` bytes without splitting a
/// multi-byte UTF-8 character. Returns the longest prefix that fits.
fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn main() -> Result<()> {
    // Tracing to stderr - inner stt (in VM) uses stdout for JSON/table output
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("stt=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Command::Run(a) => cmd_run(a),
        Command::Vm(a) => {
            if a.gauntlet {
                cmd_gauntlet(&a)
            } else {
                cmd_vm(a)
            }
        }
        Command::Probe(a) => cmd_probe(a),
        Command::List => cmd_list(),
        Command::Topo => cmd_topo(),
        Command::Cleanup(a) => {
            cgroup::CgroupManager::new(&a.parent_cgroup).cleanup_all()?;
            Ok(())
        }
        Command::Test(a) => cmd_test(a),
    }
}

/// Resolve flag short names to static flag constants.
///
/// - `all_flags=true`: returns `None` (all combinations).
/// - Empty flags: returns `Some(vec![])` (default profile only).
/// - Named flags: resolves each via `from_short_name`, errors on unknown.
fn resolve_flags(flags: &[String], all_flags: bool) -> Result<Option<Vec<&'static str>>> {
    if all_flags {
        return Ok(None);
    }
    if flags.is_empty() {
        return Ok(Some(vec![]));
    }
    let mut out = Vec::new();
    for s in flags {
        match scenario::flags::from_short_name(s) {
            Some(f) => out.push(f),
            None => anyhow::bail!(
                "unknown flag: {s}\navailable: {}",
                scenario::flags::ALL.join(", ")
            ),
        }
    }
    Ok(Some(out))
}

/// Select scenarios by name from the full catalog.
///
/// If `names` is empty or `all` is true, returns all scenarios.
/// Errors on unknown scenario names.
fn select_scenarios<'a>(
    scenarios: &'a [scenario::Scenario],
    names: &[String],
    all: bool,
) -> Result<Vec<&'a scenario::Scenario>> {
    if all || names.is_empty() {
        return Ok(scenarios.iter().collect());
    }
    let mut selected = Vec::new();
    for name in names {
        match scenarios.iter().find(|s| s.name == name.as_str()) {
            Some(s) => selected.push(s),
            None => {
                let available: Vec<&str> = scenarios.iter().map(|s| s.name).collect();
                anyhow::bail!(
                    "unknown scenario: {name}\navailable: {}",
                    available.join(", ")
                );
            }
        }
    }
    Ok(selected)
}

/// Resolve a single work type name to a `WorkType`.
fn resolve_work_type(name: &str) -> Result<workload::WorkType> {
    workload::WorkType::from_name(name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown work type: {name}\navailable: {}",
            workload::WorkType::ALL_NAMES.join(", ")
        )
    })
}

/// Validate work type names against the known set.
///
/// Returns the validated names on success, or errors on the first unknown name.
fn validate_work_types(names: &[String]) -> Result<Vec<&str>> {
    for s in names {
        if workload::WorkType::from_name(s).is_none() {
            anyhow::bail!(
                "unknown work type: {s}\navailable: {}",
                workload::WorkType::ALL_NAMES.join(", ")
            );
        }
    }
    Ok(names.iter().map(|s| s.as_str()).collect())
}

/// Format the scenario list table as a string.
fn format_scenario_list(scenarios: &[scenario::Scenario]) -> String {
    let mut out = String::new();
    let mut total = 0;
    for s in scenarios {
        let n = s.profiles().len();
        total += n;
        out.push_str(&format!(
            "{:<25} {:<12} {:>3}  {}\n",
            s.name, s.category, n, s.description
        ));
    }
    out.push_str(&format!(
        "\n{} scenarios, {} total runs with --all-flags\n",
        scenarios.len(),
        total
    ));
    out
}

fn parse_flags(args: &RunArgs) -> Result<Option<Vec<&'static str>>> {
    resolve_flags(&args.flags, args.all_flags)
}

fn cmd_run(args: RunArgs) -> Result<()> {
    let topo = TestTopology::from_system()?;
    let scenarios = scenario::all_scenarios();
    let selected = select_scenarios(
        &scenarios,
        &args.scenarios,
        args.all || args.scenarios.is_empty(),
    )?;
    let active_flags = parse_flags(&args)?;
    let scheduler_bin = args.scheduler_bin.or(args.mitosis_bin);
    if args.warn_unfair {
        verify::set_warn_unfair(true);
    }
    let repro = args.repro || args.probe_stack.is_some() || args.auto_repro;
    if repro {
        workload::set_repro_mode(true);
    }
    let work_type_override = match args.work_type.as_deref() {
        Some(s) => Some(resolve_work_type(s)?),
        None => None,
    };
    let config = RunConfig {
        scheduler_bin,
        scheduler_args: args.scheduler_arg,
        parent_cgroup: args.parent_cgroup,
        duration_s: args.duration_s,
        workers_per_cell: args.workers,
        json: args.json,
        verbose: args.verbose,
        active_flags,
        repro,
        probe_stack: args.probe_stack,
        auto_repro: args.auto_repro,
        bootlin: args.bootlin,
        kernel_dir: args.kernel_dir,
        settle_ms: 3000,
        scheduler_startup_ms: 500,
        cleanup_ms: 200,
        work_type_override,
    };
    let mut results = Runner::new(config.clone(), topo.clone())?.run_scenarios(&selected)?;
    let failed = results.iter().filter(|r| !r.passed).count();

    // Auto-repro: if run 1 crashed, extract function names and rerun with --probe-stack
    if config.auto_repro && failed > 0 && config.probe_stack.is_none() {
        let names = extract_auto_repro_functions(&results);
        if let Some(ref names) = names {
            let fn_count = names.split(',').count();
            println!(
                "\n{} auto-repro: rerunning with --probe-stack ({fn_count} functions)\n",
                style(">>>").cyan().bold(),
            );
            let mut config2 = config;
            config2.probe_stack = Some(names.clone());
            results = Runner::new(config2, topo.clone())?.run_scenarios(&selected)?;
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        print_results(&results);
    }
    let failed = results.iter().filter(|r| !r.passed).count();
    if failed > 0 {
        anyhow::bail!("{failed} scenario(s) failed");
    }
    Ok(())
}

fn cmd_vm(args: VmArgs) -> Result<()> {
    let topo = vmm::Topology {
        sockets: args.sockets as u32,
        cores_per_socket: args.cores as u32,
        threads_per_core: args.threads as u32,
    };

    // Parse scenario names and options from run_args.
    // Options (--foo bar or --foo=bar) go to extra_args, bare words are scenarios.
    let run_all = args.run_args.is_empty();
    let (scenario_names, extra_args) = split_run_args(&args.run_args);

    let max_par = args.parallel.unwrap_or(1);
    if max_par > 1 && !scenario_names.is_empty() {
        return cmd_vm_parallel(&args, &topo, &scenario_names, &extra_args, max_par);
    }

    // Single-VM mode (original behavior)
    let cfg = vm::VmConfig {
        kernel: args.kernel,
        memory_mb: args.memory_mb,
        topology: topo,
        timeout: None,
        kernel_dir: args.kernel_dir,
    };
    let t = &cfg.topology;
    println!(
        "{} VM: {} CPUs, {} LLCs",
        style("launching").cyan().bold(),
        t.total_cpus(),
        t.num_llcs()
    );
    let mut stt_args = vec!["run".into()];
    if let Some(ref bin) = args.scheduler_bin {
        stt_args.push("--scheduler-bin".into());
        stt_args.push(bin.clone());
    }
    if let Some(ref kd) = cfg.kernel_dir {
        stt_args.push("--kernel-dir".into());
        stt_args.push(kd.clone());
    }
    if run_all {
        stt_args.push("--all".into());
    } else {
        stt_args.extend(args.run_args);
    }
    let r = vm::run_in_vm(&cfg, &stt_args)?;
    if !r.output.is_empty() {
        print!("{}", r.output);
    }
    if !r.stderr.is_empty() {
        eprint!("{}", r.stderr);
    }
    if r.timed_out {
        anyhow::bail!("VM timed out");
    }
    if !r.success {
        anyhow::bail!("VM failed with exit code {}", r.exit_code);
    }
    println!(
        "{} ({:.1}s)",
        style("PASS").green().bold(),
        r.duration.as_secs_f64()
    );
    Ok(())
}

/// A single VM job: label, topology, memory, stt args.
struct VmJob {
    label: String,
    topo: vmm::Topology,
    mem: usize,
    stt_args: Vec<String>,
    kernel: Option<String>,
    kernel_dir: Option<String>,
    retries: usize,
}

/// Run VM jobs in parallel using rayon, print progress, return results.
fn run_parallel_vms(jobs: Vec<VmJob>, max_par: usize, banner: &str) -> Result<Vec<VmRunResult>> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let total = jobs.len();
    println!("{banner}");

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_par)
        .build()
        .map_err(|e| anyhow::anyhow!("create thread pool: {e}"))?;

    let completed = AtomicUsize::new(0);
    let fail_count = AtomicUsize::new(0);

    let results: Vec<VmRunResult> = pool.install(|| {
        jobs.par_iter()
            .map(|job| {
                let timeout = vm::compute_timeout(1, 20, job.topo.total_cpus() as usize);
                let cfg = vm::VmConfig {
                    kernel: job.kernel.clone(),
                    topology: job.topo,
                    memory_mb: job.mem,
                    timeout: Some(timeout),
                    kernel_dir: job.kernel_dir.clone(),
                };
                let (ok, dur, detail, inner_results, mon_data) =
                    run_vm_with_retries(&cfg, &job.stt_args, job.retries);

                let n = completed.fetch_add(1, Ordering::Relaxed) + 1;
                let is_skip = inner_results
                    .iter()
                    .any(|r| r.details.iter().any(|d| d.contains("skipped")));
                let status = if is_skip {
                    "SKIP"
                } else if ok {
                    "PASS"
                } else {
                    "FAIL"
                };

                let stats_str = inner_results
                    .first()
                    .map(format_gauntlet_stats)
                    .unwrap_or_default();
                let detail_str = format_gauntlet_detail(ok, &detail, &inner_results);
                let label = &job.label;

                println!("[{n}/{total}] {status} {label} ({dur:.0}s){stats_str}{detail_str}");
                if !ok {
                    fail_count.fetch_add(1, Ordering::Relaxed);
                    for r in inner_results.iter().filter(|r| !r.passed) {
                        for d in &r.details {
                            println!("  {d}");
                        }
                    }
                }

                (job.label.clone(), ok, dur, detail, inner_results, mon_data)
            })
            .collect()
    });

    Ok(results)
}

/// Format pass/fail summary. Returns the summary string and whether
/// all tests passed.
fn format_vm_summary(results: &[VmRunResult]) -> (String, bool) {
    let passed = results.iter().filter(|r| r.1).count();
    let failed: Vec<_> = results.iter().filter(|r| !r.1).collect();

    let mut out = format!("\n=== {}/{} passed ===", passed, results.len());

    if !failed.is_empty() {
        out.push_str("\n\nFailed:");
        for (name, _, _, d, inner, _) in &failed {
            out.push_str(&format!("\n\n  {name}:"));
            if !d.is_empty() {
                out.push_str(&format!("\n    {d}"));
            }
            for r in inner.iter().filter(|r| !r.passed) {
                for detail in &r.details {
                    out.push_str(&format!("\n    {detail}"));
                }
            }
        }
    }
    (out, failed.is_empty())
}

/// Print pass/fail summary and return error on failures.
fn print_vm_summary(results: &[VmRunResult]) -> Result<()> {
    let (summary, all_passed) = format_vm_summary(results);
    println!("{summary}");
    if !all_passed {
        let failed = results.iter().filter(|r| !r.1).count();
        anyhow::bail!("{failed} VM(s) failed");
    }
    Ok(())
}

fn cmd_vm_parallel(
    args: &VmArgs,
    topo: &vmm::Topology,
    scenarios: &[String],
    extra_args: &[String],
    max_par: usize,
) -> Result<()> {
    let jobs: Vec<VmJob> = scenarios
        .iter()
        .map(|sname| {
            let mut stt_args = vec!["run".into(), "--json".into(), sname.clone()];
            if let Some(ref bin) = args.scheduler_bin {
                stt_args.push("--scheduler-bin".into());
                stt_args.push(bin.clone());
            }
            stt_args.extend_from_slice(extra_args);
            VmJob {
                label: sname.clone(),
                topo: *topo,
                mem: args.memory_mb,
                stt_args,
                kernel: args.kernel.clone(),
                kernel_dir: args.kernel_dir.clone(),
                retries: args.retries,
            }
        })
        .collect();

    let banner = format!(
        "{} {} VMs, {} parallel, {} CPUs, {} LLCs",
        style("launching").cyan().bold(),
        jobs.len(),
        max_par,
        topo.total_cpus(),
        topo.num_llcs()
    );
    let results = run_parallel_vms(jobs, max_par, &banner)?;
    print_vm_summary(&results)
}

fn cmd_gauntlet(args: &VmArgs) -> Result<()> {
    let presets = vm::gauntlet_presets();
    let scenarios = scenario::all_scenarios();
    let max_par = args.parallel.unwrap_or_else(|| (num_cpus() / 8).max(1));

    // resolve_flags with all_flags=false: gauntlet only accepts explicit flags
    let gauntlet_flags = resolve_flags(&args.flags, false)?.unwrap_or_default();
    let fixed_profile = if gauntlet_flags.is_empty() {
        None
    } else {
        Some(scenario::FlagProfile {
            flags: gauntlet_flags,
        })
    };

    let replicas = args.replicas.max(1);
    let work_type_names = validate_work_types(&args.work_types)?;

    let jobs = build_gauntlet_jobs(
        &presets,
        &scenarios,
        fixed_profile.as_ref(),
        &work_type_names,
        replicas,
        &args.kernel,
        &args.kernel_dir,
        &args.scheduler_bin,
        args.retries,
    );

    let sched_label = args.scheduler_bin.as_deref().unwrap_or("eevdf");
    let banner = format!(
        "gauntlet [{}]: {} VMs, {} parallel",
        sched_label,
        jobs.len(),
        max_par,
    );
    let results = run_parallel_vms(jobs, max_par, &banner)?;
    print!("{}", stats::analyze_gauntlet(&results));

    let current_rows = stats::extract_rows(&results);

    // Save baseline if requested.
    if let Some(ref path) = args.save_baseline {
        let now = chrono_timestamp();
        let baseline = stats::GauntletBaseline {
            scheduler: sched_label.to_string(),
            timestamp: now,
            git_commit: None,
            replicas: replicas as u32,
            rows: current_rows.clone(),
        };
        baseline.save(path)?;
        println!(
            "\n{} baseline saved to {path} ({} rows)",
            style("saved").green().bold(),
            baseline.rows.len()
        );
    }

    // Compare against baseline if requested.
    if let Some(ref path) = args.compare {
        let baseline = stats::GauntletBaseline::load(path)?;
        println!(
            "\n{} comparing against {path} ({}, {})",
            style("compare").cyan().bold(),
            baseline.scheduler,
            baseline.timestamp,
        );
        let report = stats::compare_baselines(&baseline.rows, &current_rows);
        print!("{report}");
    }

    print_vm_summary(&results)
}

fn cmd_list() -> Result<()> {
    let scenarios = scenario::all_scenarios();
    print!("{}", format_scenario_list(&scenarios));
    Ok(())
}

fn cmd_topo() -> Result<()> {
    topology::print_topo()
}

fn format_results(results: &[runner::ScenarioResult]) -> String {
    let mut out = String::new();
    for r in results {
        let tag = if r.passed { "PASS" } else { "FAIL" };
        out.push_str(&format!(
            "{tag} {} ({:.1}s)\n",
            r.scenario_name, r.duration_s
        ));
        if !r.passed {
            for d in &r.details {
                out.push_str(&format!("  {d}\n"));
            }
        }
    }
    let (p, f) = (
        results.iter().filter(|r| r.passed).count(),
        results.iter().filter(|r| !r.passed).count(),
    );
    out.push('\n');
    if f > 0 {
        out.push_str(&format!("{f} failed, {p} passed\n"));
    } else {
        out.push_str(&format!("{p} passed\n"));
    }
    out
}

fn print_results(results: &[runner::ScenarioResult]) {
    let text = format_results(results);
    // Re-apply color for terminal output
    let mut out = std::io::stdout();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("PASS ") {
            let _ = writeln!(out, "{} {rest}", style("PASS").green().bold());
        } else if let Some(rest) = line.strip_prefix("FAIL ") {
            let _ = writeln!(out, "{} {rest}", style("FAIL").red().bold());
        } else if line.ends_with("passed") && line.contains("failed") {
            let _ = writeln!(out, "{}", style(line).red().bold());
        } else if line.ends_with("passed") {
            let _ = writeln!(out, "{}", style(line).green().bold());
        } else {
            let _ = writeln!(out, "{line}");
        }
    }
}

fn format_gauntlet_stats(r: &runner::ScenarioResult) -> String {
    let s = &r.stats;
    let cells: Vec<String> = s
        .cgroups
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "c{}:{}w/{}c={:.0}-{:.0}%",
                i, c.num_workers, c.num_cpus, c.min_runnable_pct, c.max_runnable_pct
            )
        })
        .collect();
    let mut extra = String::new();
    if s.worst_spread > 15.0 {
        extra += &format!(" UNFAIR={:.0}%", s.worst_spread);
    }
    if s.worst_gap_ms > 100 {
        extra += &format!(" STUCK={}ms@cpu{}", s.worst_gap_ms, s.worst_gap_cpu);
    }
    format!(" {} mig={}{}", cells.join(" "), s.total_migrations, extra)
}

fn format_gauntlet_detail(
    ok: bool,
    detail: &str,
    inner_results: &[runner::ScenarioResult],
) -> String {
    if !ok && !detail.is_empty() {
        format!(" | {}", truncate_str(detail, 120))
    } else if !ok && inner_results.is_empty() {
        " | VM failed (no results)".to_string()
    } else if !ok {
        let fail_d: Vec<String> = inner_results
            .iter()
            .filter(|r| !r.passed)
            .flat_map(|r| r.details.first().cloned())
            .collect();
        if !fail_d.is_empty() {
            format!(" | {}", truncate_str(&fail_d[0], 120))
        } else {
            String::new()
        }
    } else {
        String::new()
    }
}

/// Run a single VM with retry logic for infra failures.
/// Returns (passed, duration_secs, detail, inner_results, monitor_summary).
fn run_vm_with_retries(
    cfg: &vm::VmConfig,
    stt_args: &[String],
    retries: usize,
) -> (
    bool,
    f64,
    String,
    Vec<runner::ScenarioResult>,
    Option<GauntletMonitorData>,
) {
    let mut ok = false;
    let mut dur = 0.0;
    let mut detail = String::new();
    let mut inner_results = vec![];
    let mut monitor_data = None;
    for attempt in 0..retries {
        let (a_ok, a_dur, a_detail, a_inner, a_mon) = match vm::run_in_vm(cfg, stt_args) {
            Ok(r) if r.timed_out => {
                let stim = r.stimulus_events;
                let mon = r.monitor.map(|m| GauntletMonitorData {
                    summary: m.summary,
                    samples: m.samples,
                    stimulus_events: stim,
                });
                (
                    false,
                    r.duration.as_secs_f64(),
                    "timed out".into(),
                    vec![],
                    mon,
                )
            }
            Ok(r) => {
                let parsed: Vec<runner::ScenarioResult> = extract_json(&r.output);
                let d = if parsed.is_empty() {
                    let last_err = r
                        .stderr
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("no output");
                    format!("VM failed: {}", truncate_str(last_err, 120))
                } else {
                    String::new()
                };
                let stim = r.stimulus_events;
                let mon = r.monitor.map(|m| GauntletMonitorData {
                    summary: m.summary,
                    samples: m.samples,
                    stimulus_events: stim,
                });
                (
                    r.success && !parsed.iter().any(|r| !r.passed),
                    r.duration.as_secs_f64(),
                    d,
                    parsed,
                    mon,
                )
            }
            Err(e) => (false, 0.0, format!("{e:#}"), vec![], None),
        };
        ok = a_ok;
        dur = a_dur;
        detail = a_detail.clone();
        inner_results = a_inner;
        monitor_data = a_mon;
        if ok || !is_infra_failure(&inner_results, &a_detail) {
            break;
        }
        if attempt + 1 < retries {
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
    (ok, dur, detail, inner_results, monitor_data)
}

fn is_infra_failure(inner_results: &[runner::ScenarioResult], detail: &str) -> bool {
    let all_details: String = inner_results
        .iter()
        .flat_map(|r| r.details.iter())
        .chain(std::iter::once(&detail.to_string()))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    all_details.contains("fork failed")
        || all_details.contains("timed out")
        || all_details.contains("no JSON")
        || all_details.contains("spawn")
        || all_details.contains("scheduler died")
        || all_details.contains("VM failed")
}

/// Format current time as an ISO 8601 UTC timestamp using libc.
fn chrono_timestamp() -> String {
    let mut tv = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    unsafe { libc::gettimeofday(&mut tv, std::ptr::null_mut()) };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe { libc::gmtime_r(&tv.tv_sec, &mut tm) };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
    )
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn extract_json(output: &str) -> Vec<runner::ScenarioResult> {
    // Prefer delimited extraction: content between JSON markers emitted by
    // the init script.  Falls back to bracket matching for compatibility
    // with older initramfs images.
    let payload = if let Some(s) = output.find("===STT_JSON_START===")
        && let Some(e) = output.find("===STT_JSON_END===")
        && s < e
    {
        let after_marker = s + "===STT_JSON_START===".len();
        output[after_marker..e].trim()
    } else {
        output
    };

    if let Some(start) = payload.find('[')
        && let Some(end) = payload.rfind(']')
        && let Ok(parsed) = serde_json::from_str(&payload[start..=end])
    {
        return parsed;
    }
    vec![]
}

fn cmd_probe(args: ProbeArgs) -> Result<()> {
    use probe::{
        btf::{discover_bpf_symbols, parse_btf_functions},
        output::format_probe_events,
        process::run_probe_skeleton,
        stack::{filter_traceable, load_probe_stack},
    };

    let input = if args.dmesg {
        let log = stt::read_kmsg();
        let lines: Vec<&str> = log.lines().collect();
        let start = lines
            .iter()
            .rposition(|l| l.contains("sched_ext:") && l.contains("BPF scheduler"))
            .unwrap_or(0);
        lines[start..].join("\n")
    } else if let Some(ref funcs) = args.functions {
        funcs.clone()
    } else if let Some(ref path) = args.input {
        if path == "-" {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        } else {
            std::fs::read_to_string(path)?
        }
    } else {
        anyhow::bail!("provide a stack file, --dmesg, --functions, or - for stdin");
    };

    let mut functions = filter_traceable(load_probe_stack(&input));
    let bpf_syms = discover_bpf_symbols();
    if !bpf_syms.is_empty() {
        println!(
            "{} discovered {} BPF scheduler functions",
            style("probe").cyan().bold(),
            bpf_syms.len()
        );
        functions.extend(bpf_syms);
    }

    if functions.is_empty() {
        println!(
            "{} no traceable functions found",
            style("error").red().bold()
        );
        return Ok(());
    }

    println!(
        "{} probing {} functions, waiting for trigger...",
        style("probe").cyan().bold(),
        functions.len()
    );

    // Resolve BTF for kernel functions
    let kernel_names: Vec<&str> = functions
        .iter()
        .filter(|f| !f.is_bpf)
        .map(|f| f.raw_name.as_str())
        .collect();
    let btf_path = probe::btf::resolve_btf_path(args.kernel_dir.as_deref());
    let btf_funcs = parse_btf_functions(&kernel_names, btf_path.as_ref().and_then(|p| p.to_str()));

    let trigger = args.trigger.as_deref().unwrap_or("scx_exit");

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_clone = stop.clone();

    let _ = ctrlc::set_handler(move || {
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    let func_names: Vec<(u32, String)> = functions
        .iter()
        .enumerate()
        .map(|(i, f)| (i as u32, f.display_name.clone()))
        .collect();

    let events = run_probe_skeleton(&functions, &btf_funcs, trigger, &stop_clone);

    if let Some(events) = events {
        let report = format_probe_events(
            &events,
            &func_names,
            args.kernel_dir.as_deref(),
            args.bootlin,
        );
        println!("{report}");
    } else {
        println!("{} no violation captured", style("probe").yellow().bold());
    }

    Ok(())
}

fn cmd_test(args: TestArgs) -> Result<()> {
    let sidecar_dir = std::env::temp_dir().join(format!("stt-sidecar-{}", std::process::id()));
    std::fs::create_dir_all(&sidecar_dir)?;

    // Determine whether to use nextest or cargo test.
    let has_nextest = std::process::Command::new("cargo")
        .args(["nextest", "--version"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());

    let mut cmd = std::process::Command::new("cargo");
    if has_nextest {
        cmd.args(["nextest", "run", "-p", "stt"]);
        cmd.args(["--profile", &args.nextest_profile]);
    } else {
        cmd.args(["test", "-p", "stt"]);
    }

    if let Some(ref filter) = args.filter {
        if has_nextest {
            cmd.args(["-E", filter]);
        } else {
            cmd.arg(filter);
        }
    }

    if !args.nextest_args.is_empty() {
        cmd.args(&args.nextest_args);
    }

    cmd.env("STT_SIDECAR_DIR", &sidecar_dir);
    if let Some(ref kernel) = args.kernel {
        cmd.env("STT_TEST_KERNEL", kernel);
    }
    if let Some(ref bin) = args.scheduler_bin {
        cmd.env("STT_SCHEDULER", bin);
    }

    // Stream output to terminal.
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());

    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("spawn test runner: {e}"))?;

    // Scan sidecar directory for results.
    let sidecars = test_support::collect_sidecars(&sidecar_dir);

    if !sidecars.is_empty() && !args.no_analysis {
        let passed = sidecars.iter().filter(|s| s.passed).count();
        let failed = sidecars.iter().filter(|s| !s.passed).count();
        println!(
            "\n{} {} tests collected, {} passed, {} failed",
            style("sidecar").cyan().bold(),
            sidecars.len(),
            passed,
            failed,
        );

        // Convert to GauntletRow for baseline save/compare.
        let rows: Vec<stats::GauntletRow> = sidecars.iter().map(stats::sidecar_to_row).collect();

        if let Some(ref path) = args.save_baseline {
            let sched_label = args.scheduler_bin.as_deref().unwrap_or("eevdf");
            let baseline = stats::GauntletBaseline {
                scheduler: sched_label.to_string(),
                timestamp: chrono_timestamp(),
                git_commit: None,
                replicas: 1,
                rows: rows.clone(),
            };
            baseline.save(path)?;
            println!(
                "{} baseline saved to {path} ({} rows)",
                style("saved").green().bold(),
                baseline.rows.len()
            );
        }

        if let Some(ref path) = args.compare {
            let baseline = stats::GauntletBaseline::load(path)?;
            println!(
                "\n{} comparing against {path} ({}, {})",
                style("compare").cyan().bold(),
                baseline.scheduler,
                baseline.timestamp,
            );
            let report = stats::compare_baselines(&baseline.rows, &rows);
            print!("{report}");
        }
    }

    // Clean up sidecar directory.
    let _ = std::fs::remove_dir_all(&sidecar_dir);

    // Propagate the test runner's exit code.
    let code = status.code().unwrap_or(1);
    if code != 0 {
        anyhow::bail!("test runner exited with code {code}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_valid() {
        let input = r#"[{"scenario_name":"test","passed":true,"duration_s":1.0,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}]"#;
        let r = extract_json(input);
        assert_eq!(r.len(), 1);
        assert!(r[0].passed);
    }

    #[test]
    fn extract_json_with_prefix() {
        let input = "boot noise\n[{\"scenario_name\":\"t\",\"passed\":false,\"duration_s\":2.0,\"details\":[\"failed\"],\"stats\":{\"cgroups\":[],\"total_workers\":0,\"total_cpus\":0,\"total_migrations\":0,\"worst_spread\":0.0,\"worst_gap_ms\":0,\"worst_gap_cpu\":0}}]\nmore";
        let r = extract_json(input);
        assert_eq!(r.len(), 1);
        assert!(!r[0].passed);
    }

    #[test]
    fn extract_json_empty() {
        assert!(extract_json("").is_empty());
    }

    #[test]
    fn extract_json_invalid() {
        assert!(extract_json("[not json]").is_empty());
    }

    #[test]
    fn extract_json_no_brackets() {
        assert!(extract_json("no json here").is_empty());
    }

    #[test]
    fn extract_json_multiple_results() {
        let input = r#"[{"scenario_name":"a","passed":true,"duration_s":1.0,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}},{"scenario_name":"b","passed":false,"duration_s":2.0,"details":["err"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}]"#;
        let r = extract_json(input);
        assert_eq!(r.len(), 2);
        assert!(r[0].passed);
        assert!(!r[1].passed);
        assert_eq!(r[1].details, vec!["err"]);
    }

    #[test]
    fn extract_json_empty_array() {
        assert!(extract_json("[]").is_empty());
    }

    #[test]
    fn extract_json_with_delimiters() {
        let input = "boot noise\n===STT_JSON_START===\n[{\"scenario_name\":\"t\",\"passed\":true,\"duration_s\":1.0,\"details\":[],\"stats\":{\"cgroups\":[],\"total_workers\":0,\"total_cpus\":0,\"total_migrations\":0,\"worst_spread\":0.0,\"worst_gap_ms\":0,\"worst_gap_cpu\":0}}]\n===STT_JSON_END===\nSTT_EXIT=0\n";
        let r = extract_json(input);
        assert_eq!(r.len(), 1);
        assert!(r[0].passed);
    }

    #[test]
    fn extract_json_delimiters_ignore_noise() {
        // Brackets in boot noise before the start marker should be ignored.
        let input = "[garbage]\n===STT_JSON_START===\n[{\"scenario_name\":\"t\",\"passed\":false,\"duration_s\":2.0,\"details\":[\"err\"],\"stats\":{\"cgroups\":[],\"total_workers\":0,\"total_cpus\":0,\"total_migrations\":0,\"worst_spread\":0.0,\"worst_gap_ms\":0,\"worst_gap_cpu\":0}}]\n===STT_JSON_END===\n";
        let r = extract_json(input);
        assert_eq!(r.len(), 1);
        assert!(!r[0].passed);
    }

    #[test]
    fn extract_json_preserves_stats() {
        let input = r#"[{"scenario_name":"t","passed":true,"duration_s":1.0,"details":[],"stats":{"cgroups":[{"num_workers":4,"num_cpus":2,"avg_runnable_pct":50.0,"min_runnable_pct":40.0,"max_runnable_pct":60.0,"spread":20.0,"max_gap_ms":100,"max_gap_cpu":1,"total_migrations":5}],"total_workers":4,"total_cpus":2,"total_migrations":5,"worst_spread":20.0,"worst_gap_ms":100,"worst_gap_cpu":1}}]"#;
        let r = extract_json(input);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].stats.total_workers, 4);
        assert_eq!(r[0].stats.cgroups.len(), 1);
        assert_eq!(r[0].stats.cgroups[0].num_workers, 4);
    }

    fn sr(name: &str, passed: bool, dur: f64, details: Vec<&str>) -> runner::ScenarioResult {
        runner::ScenarioResult {
            scenario_name: name.into(),
            passed,
            duration_s: dur,
            details: details.into_iter().map(|s| s.into()).collect(),
            stats: Default::default(),
        }
    }

    #[test]
    fn format_results_pass_one_line() {
        let out = format_results(&[sr("proportional/default", true, 5.2, vec![])]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "PASS proportional/default (5.2s)");
        // No detail lines for pass
        assert!(!out.contains("  "));
    }

    #[test]
    fn format_results_fail_shows_details() {
        let out = format_results(&[sr(
            "proportional/default",
            false,
            6.7,
            vec![
                "unfair cell: spread=85%",
                "stuck 2448ms on cpu4",
                "sched_ext: mitosis disabled (stall)",
            ],
        )]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "FAIL proportional/default (6.7s)");
        assert_eq!(lines[1], "  unfair cell: spread=85%");
        assert_eq!(lines[2], "  stuck 2448ms on cpu4");
        assert_eq!(lines[3], "  sched_ext: mitosis disabled (stall)");
    }

    #[test]
    fn format_results_no_details_hidden_for_pass() {
        let out = format_results(&[
            sr("a/default", true, 3.0, vec![]),
            sr("b/default", false, 5.0, vec!["broken"]),
        ]);
        assert!(out.contains("PASS a/default (3.0s)"));
        assert!(out.contains("FAIL b/default (5.0s)"));
        assert!(out.contains("  broken"));
        assert!(out.contains("1 failed, 1 passed"));
    }

    #[test]
    fn format_results_all_pass_summary() {
        let out = format_results(&[
            sr("a/default", true, 1.0, vec![]),
            sr("b/default", true, 2.0, vec![]),
        ]);
        assert!(out.contains("2 passed"));
        assert!(!out.contains("failed"));
    }

    #[test]
    fn format_results_dump_lines_raw() {
        let out = format_results(&[sr(
            "test/default",
            false,
            1.0,
            vec![
                "scheduler died",
                "EXIT dump:",
                "cell[0] cpus=0-3 vtime=12345",
                "cell[1] cpus=4-7 vtime=67890",
            ],
        )]);
        // Each detail on its own line, indented, no | joining
        assert!(out.contains("  scheduler died\n"));
        assert!(out.contains("  EXIT dump:\n"));
        assert!(out.contains("  cell[0] cpus=0-3 vtime=12345\n"));
        assert!(out.contains("  cell[1] cpus=4-7 vtime=67890\n"));
        assert!(!out.contains(" | "));
    }

    #[test]
    fn format_results_dmesg_lines_raw() {
        let out = format_results(&[sr(
            "test/default",
            false,
            1.0,
            vec![
                "stuck 3000ms on cpu2",
                "sched_ext: BPF scheduler disabled (runnable task stall)",
                "sched_ext: mitosis: (worker)[42] failed to run for 3.0s",
            ],
        )]);
        assert!(out.contains("  sched_ext: BPF scheduler disabled (runnable task stall)\n"));
        assert!(out.contains("  sched_ext: mitosis: (worker)[42] failed to run for 3.0s\n"));
    }

    // -- gauntlet formatting tests --

    fn sr_with_stats(
        name: &str,
        passed: bool,
        cgroups: Vec<verify::CgroupStats>,
        spread: f64,
        gap_ms: u64,
        gap_cpu: usize,
        mig: u64,
    ) -> runner::ScenarioResult {
        runner::ScenarioResult {
            scenario_name: name.into(),
            passed,
            duration_s: 20.0,
            details: if passed { vec![] } else { vec!["stuck".into()] },
            stats: verify::ScenarioStats {
                cgroups,
                total_workers: 4,
                total_cpus: 4,
                total_migrations: mig,
                worst_spread: spread,
                worst_gap_ms: gap_ms,
                worst_gap_cpu: gap_cpu,
            },
        }
    }

    fn cg(workers: usize, cpus: usize, min: f64, max: f64) -> verify::CgroupStats {
        verify::CgroupStats {
            num_workers: workers,
            num_cpus: cpus,
            avg_runnable_pct: (min + max) / 2.0,
            min_runnable_pct: min,
            max_runnable_pct: max,
            spread: max - min,
            max_gap_ms: 0,
            max_gap_cpu: 0,
            total_migrations: 0,
        }
    }

    #[test]
    fn gauntlet_stats_basic() {
        let r = sr_with_stats(
            "test",
            true,
            vec![cg(4, 2, 50.0, 60.0), cg(4, 2, 55.0, 65.0)],
            10.0,
            50,
            0,
            7,
        );
        let s = format_gauntlet_stats(&r);
        assert!(s.contains("c0:4w/2c=50-60%"));
        assert!(s.contains("c1:4w/2c=55-65%"));
        assert!(s.contains("mig=7"));
        assert!(!s.contains("UNFAIR"));
        assert!(!s.contains("STUCK"));
    }

    #[test]
    fn gauntlet_stats_unfair() {
        let r = sr_with_stats("test", false, vec![cg(4, 2, 10.0, 80.0)], 70.0, 50, 0, 3);
        let s = format_gauntlet_stats(&r);
        assert!(s.contains("UNFAIR=70%"));
    }

    #[test]
    fn gauntlet_stats_stuck() {
        let r = sr_with_stats("test", false, vec![cg(4, 2, 50.0, 60.0)], 10.0, 2500, 3, 5);
        let s = format_gauntlet_stats(&r);
        assert!(s.contains("STUCK=2500ms@cpu3"));
    }

    #[test]
    fn gauntlet_stats_unfair_and_stuck() {
        let r = sr_with_stats("test", false, vec![cg(4, 2, 10.0, 90.0)], 80.0, 3000, 1, 0);
        let s = format_gauntlet_stats(&r);
        assert!(s.contains("UNFAIR=80%"));
        assert!(s.contains("STUCK=3000ms@cpu1"));
    }

    #[test]
    fn gauntlet_detail_vm_error() {
        let d = format_gauntlet_detail(false, "spawn failed", &[]);
        assert_eq!(d, " | spawn failed");
    }

    #[test]
    fn gauntlet_detail_no_results() {
        let d = format_gauntlet_detail(false, "", &[]);
        assert_eq!(d, " | VM failed (no results)");
    }

    #[test]
    fn gauntlet_detail_first_failure() {
        let d = format_gauntlet_detail(
            false,
            "",
            &[sr(
                "test",
                false,
                1.0,
                vec!["unfair cell: spread=85%", "stuck 2000ms"],
            )],
        );
        assert_eq!(d, " | unfair cell: spread=85%");
    }

    #[test]
    fn gauntlet_detail_pass_empty() {
        let d = format_gauntlet_detail(true, "", &[sr("test", true, 1.0, vec![])]);
        assert_eq!(d, "");
    }

    #[test]
    fn gauntlet_detail_truncates_long() {
        let long = "x".repeat(200);
        let d = format_gauntlet_detail(false, &long, &[]);
        assert!(d.len() <= 124); // " | " + 120 chars
    }

    #[test]
    fn is_infra_fork_failed() {
        assert!(is_infra_failure(&[], "fork failed: resource unavailable"));
    }

    #[test]
    fn is_infra_timed_out() {
        assert!(is_infra_failure(&[], "timed out"));
    }

    #[test]
    fn is_infra_vm_failed() {
        assert!(is_infra_failure(&[], "VM failed: no output"));
    }

    #[test]
    fn is_infra_scheduler_died() {
        assert!(is_infra_failure(
            &[sr("test", false, 1.0, vec!["scheduler died"])],
            ""
        ));
    }

    #[test]
    fn not_infra_real_failure() {
        assert!(!is_infra_failure(
            &[sr("test", false, 1.0, vec!["unfair cell: spread=85%"])],
            ""
        ));
    }

    #[test]
    fn not_infra_stuck() {
        assert!(!is_infra_failure(
            &[sr("test", false, 1.0, vec!["stuck 3000ms on cpu2"])],
            ""
        ));
    }

    // -- truncate_str tests --

    #[test]
    fn truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn truncate_str_cuts() {
        assert_eq!(truncate_str("hello world", 5), "hello");
    }

    #[test]
    fn truncate_str_empty() {
        assert_eq!(truncate_str("", 5), "");
    }

    #[test]
    fn truncate_str_multibyte() {
        // "café" is 5 bytes (é = 2 bytes). Truncating at 4 bytes must not
        // split the é — should return "caf" (3 bytes).
        assert_eq!(truncate_str("café", 4), "caf");
    }

    #[test]
    fn truncate_str_zero() {
        assert_eq!(truncate_str("hello", 0), "");
    }

    // -- sidecar_to_row tests --

    #[test]
    fn sidecar_to_row_basic() {
        let sc = test_support::SidecarResult {
            test_name: "my_test".to_string(),
            topology: "2s4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: true,
            stats: verify::ScenarioStats {
                cgroups: vec![],
                total_workers: 4,
                total_cpus: 8,
                total_migrations: 12,
                worst_spread: 15.0,
                worst_gap_ms: 200,
                worst_gap_cpu: 3,
            },
            monitor: Some(monitor::MonitorSummary {
                total_samples: 10,
                max_imbalance_ratio: 2.5,
                max_local_dsq_depth: 4,
                stall_detected: true,
                event_deltas: Some(monitor::ScxEventDeltas {
                    total_fallback: 7,
                    fallback_rate: 0.5,
                    max_fallback_burst: 2,
                    total_dispatch_offline: 0,
                    total_dispatch_keep_last: 3,
                    keep_last_rate: 0.2,
                    total_enq_skip_exiting: 0,
                    total_enq_skip_migration_disabled: 0,
                }),
            }),
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
        };
        let row = stats::sidecar_to_row(&sc);
        assert_eq!(row.scenario, "my_test");
        assert_eq!(row.topology, "2s4c2t");
        assert!(row.passed);
        assert_eq!(row.spread, 15.0);
        assert_eq!(row.gap_ms, 200);
        assert_eq!(row.migrations, 12);
        assert_eq!(row.imbalance_ratio, 2.5);
        assert_eq!(row.max_dsq_depth, 4);
        assert_eq!(row.stall_count, 1);
        assert_eq!(row.fallback_count, 7);
        assert_eq!(row.keep_last_count, 3);
        assert!(row.flags.is_empty());
        assert_eq!(row.replica, 1);
    }

    #[test]
    fn sidecar_to_row_no_monitor() {
        let sc = test_support::SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1s2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
        };
        let row = stats::sidecar_to_row(&sc);
        assert_eq!(row.scenario, "eevdf_test");
        assert!(!row.passed);
        assert_eq!(row.imbalance_ratio, 0.0);
        assert_eq!(row.max_dsq_depth, 0);
        assert_eq!(row.stall_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    #[test]
    fn truncate_str_multibyte_boundary() {
        let s = "é"; // 2 bytes
        assert_eq!(truncate_str(s, 1), "");
        assert_eq!(truncate_str(s, 2), "é");
    }

    #[test]
    fn truncate_str_multibyte_mixed() {
        let s = "aé"; // 'a' = 1 byte, 'é' = 2 bytes = 3 total
        assert_eq!(truncate_str(s, 1), "a");
        assert_eq!(truncate_str(s, 2), "a");
        assert_eq!(truncate_str(s, 3), "aé");
    }

    #[test]
    fn chrono_timestamp_iso8601_format() {
        let ts = chrono_timestamp();
        // ISO 8601: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20);
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert!(ts.contains('-'));
        assert!(ts.contains(':'));
    }

    #[test]
    fn num_cpus_nonzero() {
        assert!(num_cpus() > 0);
    }

    #[test]
    fn is_infra_spawn_failure() {
        assert!(is_infra_failure(&[], "spawn error: resource unavailable"));
    }

    #[test]
    fn is_infra_no_json() {
        assert!(is_infra_failure(&[], "no JSON output"));
    }

    #[test]
    fn gauntlet_detail_fail_no_details_empty() {
        // Fail with results but all pass — unusual but possible after retry.
        let d = format_gauntlet_detail(false, "", &[sr("test", true, 1.0, vec![])]);
        assert_eq!(d, "");
    }

    #[test]
    fn format_results_single_all_fail() {
        let out = format_results(&[sr("x/default", false, 1.0, vec!["err"])]);
        assert!(out.contains("1 failed, 0 passed"));
    }

    #[test]
    fn sidecar_to_row_no_stall() {
        let sc = test_support::SidecarResult {
            test_name: "t".to_string(),
            topology: "1s1c1t".to_string(),
            scheduler: "test".to_string(),
            passed: true,
            stats: Default::default(),
            monitor: Some(monitor::MonitorSummary {
                total_samples: 5,
                max_imbalance_ratio: 1.0,
                max_local_dsq_depth: 0,
                stall_detected: false,
                event_deltas: None,
            }),
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
        };
        let row = stats::sidecar_to_row(&sc);
        assert_eq!(row.stall_count, 0);
        assert_eq!(row.fallback_count, 0);
        assert_eq!(row.keep_last_count, 0);
    }

    // -- format_results edge cases --

    #[test]
    fn format_results_empty() {
        let out = format_results(&[]);
        assert!(out.contains("0 passed"));
    }

    #[test]
    fn format_results_many_failures() {
        let results: Vec<runner::ScenarioResult> = (0..5)
            .map(|i| sr(&format!("s{i}/default"), false, 1.0, vec!["broken"]))
            .collect();
        let out = format_results(&results);
        assert!(out.contains("5 failed, 0 passed"));
    }

    // -- gauntlet stats edge cases --

    #[test]
    fn gauntlet_stats_no_cgroups() {
        let r = runner::ScenarioResult {
            scenario_name: "test".into(),
            passed: true,
            duration_s: 1.0,
            details: vec![],
            stats: verify::ScenarioStats {
                cgroups: vec![],
                total_workers: 0,
                total_cpus: 0,
                total_migrations: 0,
                worst_spread: 0.0,
                worst_gap_ms: 0,
                worst_gap_cpu: 0,
            },
        };
        let s = format_gauntlet_stats(&r);
        assert!(s.contains("mig=0"));
    }

    // -- is_infra_failure edge cases --

    #[test]
    fn is_infra_empty_inputs() {
        assert!(!is_infra_failure(&[], ""));
    }

    #[test]
    fn is_infra_detail_in_inner_results() {
        let results = vec![sr("t", false, 1.0, vec!["fork failed"])];
        assert!(is_infra_failure(&results, ""));
    }

    // -- extract_json edge cases --

    #[test]
    fn extract_json_nested_brackets() {
        // Noise brackets before the real JSON array cause find('[') to
        // match the noise. Without STT_JSON delimiters the substring
        // spans noise..last-bracket which is invalid JSON.
        let input = r#"some [noise] before [{"scenario_name":"t","passed":true,"duration_s":1.0,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}]"#;
        let r = extract_json(input);
        assert!(r.is_empty());
    }

    // -- truncate_str additional --

    #[test]
    fn truncate_str_3byte_utf8() {
        // Chinese character is 3 bytes
        let s = "\u{4e16}"; // 世
        assert_eq!(truncate_str(s, 1), "");
        assert_eq!(truncate_str(s, 2), "");
        assert_eq!(truncate_str(s, 3), "\u{4e16}");
    }

    #[test]
    fn truncate_str_4byte_utf8() {
        let s = "\u{1f600}"; // emoji, 4 bytes
        assert_eq!(truncate_str(s, 1), "");
        assert_eq!(truncate_str(s, 2), "");
        assert_eq!(truncate_str(s, 3), "");
        assert_eq!(truncate_str(s, 4), "\u{1f600}");
    }

    // -- chrono_timestamp --

    #[test]
    fn chrono_timestamp_parses_as_date() {
        let ts = chrono_timestamp();
        let parts: Vec<&str> = ts.split('T').collect();
        assert_eq!(parts.len(), 2);
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        assert_eq!(date_parts.len(), 3);
        let year: u32 = date_parts[0].parse().unwrap();
        assert!((2024..=2100).contains(&year));
    }

    // -- sidecar_to_row_labeled tests --

    #[test]
    fn sidecar_to_row_labeled_basic() {
        let sc = test_support::SidecarResult {
            test_name: "fallback".to_string(),
            topology: "1s2c1t".to_string(),
            scheduler: "test".to_string(),
            passed: true,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
        };
        let row = stats::sidecar_to_row_labeled(&sc, "tiny-1llc/proportional/borrow");
        assert_eq!(row.topology, "tiny-1llc");
        assert_eq!(row.scenario, "proportional");
        assert_eq!(row.flags, "borrow");
    }

    // -- format_gauntlet_detail edge cases --

    #[test]
    fn gauntlet_detail_fail_with_multiple_details() {
        let d = format_gauntlet_detail(
            false,
            "",
            &[
                sr("a", false, 1.0, vec!["first error"]),
                sr("b", false, 1.0, vec!["second error"]),
            ],
        );
        // Should show the first failure's first detail
        assert!(d.contains("first error"));
    }

    // -- split_run_args tests --

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn split_run_args_empty() {
        let (names, extra) = split_run_args(&[]);
        assert!(names.is_empty());
        assert!(extra.is_empty());
    }

    #[test]
    fn split_run_args_scenarios_only() {
        let (names, extra) = split_run_args(&s(&["proportional", "stall_detect"]));
        assert_eq!(names, ["proportional", "stall_detect"]);
        assert!(extra.is_empty());
    }

    #[test]
    fn split_run_args_options_only() {
        let (names, extra) = split_run_args(&s(&["--flags=borrow,rebal", "--duration-s", "30"]));
        assert!(names.is_empty());
        assert_eq!(extra, ["--flags=borrow,rebal", "--duration-s", "30"]);
    }

    #[test]
    fn split_run_args_mixed() {
        let (names, extra) = split_run_args(&s(&[
            "proportional",
            "--flags=borrow",
            "stall_detect",
            "--duration-s",
            "30",
        ]));
        assert_eq!(names, ["proportional", "stall_detect"]);
        assert_eq!(extra, ["--flags=borrow", "--duration-s", "30"]);
    }

    #[test]
    fn split_run_args_key_equals_value() {
        let (names, extra) = split_run_args(&s(&["--key=value", "scenario"]));
        assert_eq!(names, ["scenario"]);
        assert_eq!(extra, ["--key=value"]);
    }

    #[test]
    fn split_run_args_key_space_value_with_dash() {
        // --key followed by --another should NOT consume --another as a value
        let (names, extra) = split_run_args(&s(&["--key", "--another"]));
        assert!(names.is_empty());
        assert_eq!(extra, ["--key", "--another"]);
    }

    #[test]
    fn split_run_args_short_flag_consumes_value() {
        // -v followed by non-dash token: treated as -v <value>
        let (names, extra) = split_run_args(&s(&["-v", "proportional"]));
        assert!(names.is_empty());
        assert_eq!(extra, ["-v", "proportional"]);
    }

    #[test]
    fn split_run_args_boolean_short_flag() {
        // -v followed by another dash token: -v is standalone
        let (names, extra) = split_run_args(&s(&["-v", "--json"]));
        assert!(names.is_empty());
        assert_eq!(extra, ["-v", "--json"]);
    }

    // -- extract_auto_repro_functions tests --

    #[test]
    fn extract_auto_repro_functions_from_functions_line() {
        let results = vec![sr(
            "test",
            false,
            1.0,
            vec!["  functions: scx_exit, do_exit, panic"],
        )];
        let names = extract_auto_repro_functions(&results);
        assert_eq!(names, Some("scx_exit,do_exit,panic".to_string()));
    }

    #[test]
    fn extract_auto_repro_functions_no_match() {
        let results = vec![sr("test", false, 1.0, vec!["unfair cell: spread=85%"])];
        let names = extract_auto_repro_functions(&results);
        assert!(names.is_none());
    }

    #[test]
    fn extract_auto_repro_functions_empty_results() {
        let names = extract_auto_repro_functions(&[]);
        assert!(names.is_none());
    }

    #[test]
    fn extract_auto_repro_functions_empty_functions_line() {
        let results = vec![sr("test", false, 1.0, vec!["  functions:"])];
        let names = extract_auto_repro_functions(&results);
        // "functions:" with nothing after -> all filtered out -> empty join -> Some("")
        // but the caller checks for non-empty, this is the raw extraction.
        assert_eq!(names, Some("".to_string()));
    }

    #[test]
    fn extract_auto_repro_functions_pass_results_no_details() {
        let results = vec![sr("test", true, 5.0, vec![])];
        let names = extract_auto_repro_functions(&results);
        assert!(names.is_none());
    }

    #[test]
    fn extract_auto_repro_functions_multiple_results() {
        let results = vec![
            sr("a", false, 1.0, vec!["stuck 3000ms"]),
            sr("b", false, 1.0, vec!["  functions: func_a, func_b"]),
        ];
        let names = extract_auto_repro_functions(&results);
        assert_eq!(names, Some("func_a,func_b".to_string()));
    }

    // -- build_gauntlet_jobs tests --

    #[test]
    fn build_gauntlet_jobs_empty_presets() {
        let scenarios = scenario::all_scenarios();
        let jobs = build_gauntlet_jobs(&[], &scenarios, None, &[], 1, &None, &None, &None, 3);
        assert!(jobs.is_empty());
    }

    #[test]
    fn build_gauntlet_jobs_empty_scenarios() {
        let presets = vm::gauntlet_presets();
        let jobs = build_gauntlet_jobs(&presets, &[], None, &[], 1, &None, &None, &None, 3);
        assert!(jobs.is_empty());
    }

    #[test]
    fn build_gauntlet_jobs_single_preset_single_scenario() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            None,
            &[],
            1,
            &None,
            &None,
            &None,
            3,
        );
        // 1 preset x 1 scenario x N profiles
        let expected = first_scenario[0].profiles().len();
        assert_eq!(jobs.len(), expected);
        // Label format: preset/scenario/profile
        assert!(jobs[0].label.starts_with(&format!(
            "{}/{}/",
            first_preset[0].name, first_scenario[0].name
        )));
    }

    #[test]
    fn build_gauntlet_jobs_fixed_profile() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            1,
            &None,
            &None,
            &None,
            3,
        );
        // Fixed profile -> 1 job per preset x scenario
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].label.ends_with("/default"));
    }

    #[test]
    fn build_gauntlet_jobs_replicas() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            3,
            &None,
            &None,
            &None,
            1,
        );
        assert_eq!(jobs.len(), 3);
        assert!(jobs[0].label.ends_with("#1"));
        assert!(jobs[1].label.ends_with("#2"));
        assert!(jobs[2].label.ends_with("#3"));
    }

    #[test]
    fn build_gauntlet_jobs_work_types() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &["CpuSpin", "Bursty"],
            1,
            &None,
            &None,
            &None,
            3,
        );
        assert_eq!(jobs.len(), 2);
        assert!(jobs[0].label.contains("/CpuSpin"));
        assert!(jobs[1].label.contains("/Bursty"));
        assert!(
            jobs[0]
                .stt_args
                .contains(&"--work-type=CpuSpin".to_string())
        );
        assert!(jobs[1].stt_args.contains(&"--work-type=Bursty".to_string()));
    }

    #[test]
    fn build_gauntlet_jobs_scheduler_bin() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            1,
            &None,
            &None,
            &Some("/path/to/scheduler".to_string()),
            3,
        );
        assert_eq!(jobs.len(), 1);
        assert!(jobs[0].stt_args.contains(&"--scheduler-bin".to_string()));
        assert!(jobs[0].stt_args.contains(&"/path/to/scheduler".to_string()));
    }

    #[test]
    fn build_gauntlet_jobs_kernel_and_kernel_dir() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let kernel = Some("bzImage".to_string());
        let kernel_dir = Some("/path/to/linux".to_string());
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            1,
            &kernel,
            &kernel_dir,
            &None,
            5,
        );
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].kernel, kernel);
        assert_eq!(jobs[0].kernel_dir, kernel_dir);
        assert_eq!(jobs[0].retries, 5);
    }

    #[test]
    fn build_gauntlet_jobs_with_flags() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile {
            flags: vec![scenario::flags::BORROW, scenario::flags::REBAL],
        };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            1,
            &None,
            &None,
            &None,
            3,
        );
        assert_eq!(jobs.len(), 1);
        assert!(
            jobs[0]
                .stt_args
                .contains(&"--flags=borrow,rebal".to_string())
        );
    }

    #[test]
    fn build_gauntlet_jobs_stt_args_structure() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            1,
            &None,
            &None,
            &None,
            3,
        );
        let args = &jobs[0].stt_args;
        assert_eq!(args[0], "run");
        assert_eq!(args[1], "--json");
        assert_eq!(args[2], "--duration-s");
        assert_eq!(args[3], "20");
        assert_eq!(args[4], first_scenario[0].name);
    }

    #[test]
    fn build_gauntlet_jobs_topology_propagated() {
        let presets = vm::gauntlet_presets();
        let scenarios = scenario::all_scenarios();
        let first_preset = &presets[..1];
        let first_scenario = &scenarios[..1];
        let fp = scenario::FlagProfile { flags: vec![] };
        let jobs = build_gauntlet_jobs(
            first_preset,
            first_scenario,
            Some(&fp),
            &[],
            1,
            &None,
            &None,
            &None,
            3,
        );
        assert_eq!(jobs[0].topo.sockets, first_preset[0].topology.sockets);
        assert_eq!(
            jobs[0].topo.cores_per_socket,
            first_preset[0].topology.cores_per_socket
        );
        assert_eq!(
            jobs[0].topo.threads_per_core,
            first_preset[0].topology.threads_per_core
        );
        assert_eq!(jobs[0].mem, first_preset[0].memory_mb);
    }

    // -- resolve_flags tests --

    #[test]
    fn resolve_flags_all_flags() {
        let r = resolve_flags(&[], true).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn resolve_flags_empty() {
        let r = resolve_flags(&[], false).unwrap();
        assert_eq!(r, Some(vec![]));
    }

    #[test]
    fn resolve_flags_valid() {
        let flags = s(&["borrow", "rebal"]);
        let r = resolve_flags(&flags, false).unwrap().unwrap();
        assert_eq!(r, vec!["borrow", "rebal"]);
    }

    #[test]
    fn resolve_flags_unknown() {
        let flags = s(&["borrow", "nonexistent"]);
        let r = resolve_flags(&flags, false);
        assert!(r.is_err());
        let err = r.unwrap_err().to_string();
        assert!(
            err.contains("nonexistent"),
            "error should name the bad flag: {err}"
        );
        assert!(
            err.contains("available:"),
            "error should list available flags: {err}"
        );
    }

    #[test]
    fn resolve_flags_single() {
        let flags = s(&["llc"]);
        let r = resolve_flags(&flags, false).unwrap().unwrap();
        assert_eq!(r, vec!["llc"]);
    }

    #[test]
    fn resolve_flags_all_known() {
        let flags: Vec<String> = scenario::flags::ALL.iter().map(|f| f.to_string()).collect();
        let r = resolve_flags(&flags, false).unwrap().unwrap();
        assert_eq!(r.len(), scenario::flags::ALL.len());
    }

    // -- select_scenarios tests --

    #[test]
    fn select_scenarios_all() {
        let scenarios = scenario::all_scenarios();
        let selected = select_scenarios(&scenarios, &[], true).unwrap();
        assert_eq!(selected.len(), scenarios.len());
    }

    #[test]
    fn select_scenarios_empty_names_returns_all() {
        let scenarios = scenario::all_scenarios();
        let selected = select_scenarios(&scenarios, &[], false).unwrap();
        assert_eq!(selected.len(), scenarios.len());
    }

    #[test]
    fn select_scenarios_by_name() {
        let scenarios = scenario::all_scenarios();
        let names = s(&[scenarios[0].name, scenarios[1].name]);
        let selected = select_scenarios(&scenarios, &names, false).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].name, scenarios[0].name);
        assert_eq!(selected[1].name, scenarios[1].name);
    }

    #[test]
    fn select_scenarios_unknown_name() {
        let scenarios = scenario::all_scenarios();
        let names = s(&["nonexistent_scenario"]);
        let r = select_scenarios(&scenarios, &names, false);
        let err = r.err().expect("should fail");
        let msg = err.to_string();
        assert!(msg.contains("nonexistent_scenario"));
        assert!(msg.contains("available:"));
    }

    #[test]
    fn select_scenarios_partial_unknown() {
        let scenarios = scenario::all_scenarios();
        let names = s(&[scenarios[0].name, "bad_name"]);
        let r = select_scenarios(&scenarios, &names, false);
        assert!(r.is_err());
    }

    // -- format_scenario_list tests --

    #[test]
    fn format_scenario_list_contains_all() {
        let scenarios = scenario::all_scenarios();
        let out = format_scenario_list(&scenarios);
        for s in &scenarios {
            assert!(out.contains(s.name), "missing scenario: {}", s.name);
        }
    }

    #[test]
    fn format_scenario_list_has_summary() {
        let scenarios = scenario::all_scenarios();
        let out = format_scenario_list(&scenarios);
        assert!(out.contains(&format!("{} scenarios", scenarios.len())));
        assert!(out.contains("total runs with --all-flags"));
    }

    #[test]
    fn format_scenario_list_empty() {
        let out = format_scenario_list(&[]);
        assert!(out.contains("0 scenarios"));
    }

    #[test]
    fn format_scenario_list_categories() {
        let scenarios = scenario::all_scenarios();
        let out = format_scenario_list(&scenarios);
        for s in &scenarios {
            assert!(out.contains(s.category));
        }
    }

    // -- resolve_work_type tests --

    #[test]
    fn resolve_work_type_valid() {
        let wt = resolve_work_type("CpuSpin").unwrap();
        assert!(matches!(wt, workload::WorkType::CpuSpin));
    }

    #[test]
    fn resolve_work_type_unknown() {
        match resolve_work_type("NonexistentType") {
            Err(e) => {
                let msg = e.to_string();
                assert!(msg.contains("NonexistentType"));
                assert!(msg.contains("available:"));
            }
            Ok(_) => panic!("should fail on unknown work type"),
        }
    }

    // -- validate_work_types tests --

    #[test]
    fn validate_work_types_empty() {
        let r = validate_work_types(&[]).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn validate_work_types_valid() {
        let names = s(&["CpuSpin"]);
        let r = validate_work_types(&names).unwrap();
        assert_eq!(r, vec!["CpuSpin"]);
    }

    #[test]
    fn validate_work_types_unknown() {
        let names = s(&["CpuSpin", "BadType"]);
        let r = validate_work_types(&names);
        assert!(r.is_err());
    }

    #[test]
    fn validate_work_types_multiple_valid() {
        let names = s(&["CpuSpin", "Bursty"]);
        let r = validate_work_types(&names).unwrap();
        assert_eq!(r.len(), 2);
    }

    // -- parse_flags tests --

    #[test]
    fn parse_flags_with_valid_flags() {
        let args = RunArgs {
            scenarios: vec![],
            all: false,
            scheduler_bin: None,
            scheduler_arg: vec![],
            mitosis_bin: None,
            parent_cgroup: "/sys/fs/cgroup/stt".into(),
            duration_s: 15,
            workers: 4,
            json: false,
            verbose: false,
            all_flags: false,
            flags: s(&["borrow", "rebal"]),
            warn_unfair: false,
            repro: false,
            probe_stack: None,
            auto_repro: false,
            bootlin: false,
            kernel_dir: None,
            work_type: None,
        };
        let r = parse_flags(&args).unwrap();
        assert_eq!(r, Some(vec!["borrow", "rebal"]));
    }

    #[test]
    fn parse_flags_all_flags_true() {
        let args = RunArgs {
            scenarios: vec![],
            all: false,
            scheduler_bin: None,
            scheduler_arg: vec![],
            mitosis_bin: None,
            parent_cgroup: "/sys/fs/cgroup/stt".into(),
            duration_s: 15,
            workers: 4,
            json: false,
            verbose: false,
            all_flags: true,
            flags: vec![],
            warn_unfair: false,
            repro: false,
            probe_stack: None,
            auto_repro: false,
            bootlin: false,
            kernel_dir: None,
            work_type: None,
        };
        let r = parse_flags(&args).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn parse_flags_unknown_flag_returns_error() {
        let args = RunArgs {
            scenarios: vec![],
            all: false,
            scheduler_bin: None,
            scheduler_arg: vec![],
            mitosis_bin: None,
            parent_cgroup: "/sys/fs/cgroup/stt".into(),
            duration_s: 15,
            workers: 4,
            json: false,
            verbose: false,
            all_flags: false,
            flags: s(&["nonexistent"]),
            warn_unfair: false,
            repro: false,
            probe_stack: None,
            auto_repro: false,
            bootlin: false,
            kernel_dir: None,
            work_type: None,
        };
        let r = parse_flags(&args);
        assert!(r.is_err());
    }

    // -- format_vm_summary tests --

    #[test]
    fn format_vm_summary_all_pass() {
        let results: Vec<VmRunResult> = vec![
            ("job1".into(), true, 10.0, String::new(), vec![], None),
            ("job2".into(), true, 20.0, String::new(), vec![], None),
        ];
        let (summary, all_passed) = format_vm_summary(&results);
        assert!(all_passed);
        assert!(summary.contains("2/2 passed"));
        assert!(!summary.contains("Failed:"));
    }

    #[test]
    fn format_vm_summary_with_failures() {
        let results: Vec<VmRunResult> = vec![
            ("job1".into(), true, 10.0, String::new(), vec![], None),
            ("job2".into(), false, 20.0, "timed out".into(), vec![], None),
        ];
        let (summary, all_passed) = format_vm_summary(&results);
        assert!(!all_passed);
        assert!(summary.contains("1/2 passed"));
        assert!(summary.contains("Failed:"));
        assert!(summary.contains("job2"));
        assert!(summary.contains("timed out"));
    }

    #[test]
    fn format_vm_summary_empty() {
        let results: Vec<VmRunResult> = vec![];
        let (summary, all_passed) = format_vm_summary(&results);
        assert!(all_passed);
        assert!(summary.contains("0/0 passed"));
    }

    #[test]
    fn format_vm_summary_failure_with_inner_results() {
        let inner = vec![sr("scenario_a", false, 1.0, vec!["unfair cell"])];
        let results: Vec<VmRunResult> =
            vec![("job1".into(), false, 15.0, String::new(), inner, None)];
        let (summary, all_passed) = format_vm_summary(&results);
        assert!(!all_passed);
        assert!(summary.contains("job1"));
        assert!(summary.contains("unfair cell"));
    }

    #[test]
    fn format_vm_summary_all_fail() {
        let results: Vec<VmRunResult> = vec![
            ("j1".into(), false, 1.0, "err1".into(), vec![], None),
            ("j2".into(), false, 2.0, "err2".into(), vec![], None),
        ];
        let (summary, all_passed) = format_vm_summary(&results);
        assert!(!all_passed);
        assert!(summary.contains("0/2 passed"));
        assert!(summary.contains("j1"));
        assert!(summary.contains("j2"));
    }
}
