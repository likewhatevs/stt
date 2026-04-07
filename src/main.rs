use std::io::Write;

use anyhow::Result;
use clap::Parser;
use console::style;

use runner::{RunConfig, Runner};
use stt::{probe, runner, scenario, topology, verify, workload};
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
    /// Probe kernel functions from a crash stack
    Probe(ProbeArgs),
    /// Show CPU topology
    Topo,
    /// Kernel build, clean, and config management
    Kernel(KernelArgs),
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
struct KernelArgs {
    #[clap(subcommand)]
    action: KernelAction,
}

#[derive(Debug, Parser)]
enum KernelAction {
    /// Build a kernel with stt's config fragment
    Build(KernelPathArg),
    /// Clean a kernel source tree (make mrproper)
    Clean(KernelPathArg),
    /// Print stt's kernel config fragment to stdout
    Kconfig,
}

#[derive(Debug, Parser)]
struct KernelPathArg {
    /// Path to linux source tree (default: current directory)
    #[clap(default_value = ".")]
    path: String,
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
        Command::Probe(a) => cmd_probe(a),
        Command::Topo => cmd_topo(),
        Command::Kernel(k) => match k.action {
            KernelAction::Build(a) => cmd_build_kernel(a),
            KernelAction::Clean(a) => cmd_clean_kernel(a),
            KernelAction::Kconfig => {
                print!("{KERNEL_CONFIG}");
                Ok(())
            }
        },
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
        workers_per_cgroup: args.workers,
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

    let trigger = args.trigger.as_deref().unwrap_or("scx_disable_workfn");

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

/// Embedded kernel config fragment for stt kernel builds.
const KERNEL_CONFIG: &str = include_str!("../kernel.config");

/// Validate that a path looks like a kernel source tree.
fn validate_kernel_tree(path: &std::path::Path) -> Result<()> {
    if !path.join("Makefile").exists() {
        anyhow::bail!("{}: not a kernel tree (no Makefile)", path.display());
    }
    if !path.join("kernel").exists() {
        anyhow::bail!("{}: not a kernel tree (no kernel/)", path.display());
    }
    Ok(())
}

fn run_step(label: &str, cmd: &str, args: &[&str]) -> Result<()> {
    println!("{} {label}", style(">>>").cyan().bold());
    let status = std::process::Command::new(cmd)
        .args(args)
        .status()
        .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    if !status.success() {
        anyhow::bail!("{label} failed (exit {})", status.code().unwrap_or(-1));
    }
    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn cmd_build_kernel(args: KernelPathArg) -> Result<()> {
    let path = std::path::Path::new(&args.path)
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("{}: {e}", args.path))?;
    validate_kernel_tree(&path)?;

    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("path is not valid UTF-8"))?;

    let dot_config = path.join(".config");
    let jobs = format!("-j{}", num_cpus());

    run_step("defconfig", "make", &["-C", path_str, "defconfig"])?;

    // Append stt config fragment to .config. Using cat-append + olddefconfig
    // instead of merge_config.sh because merge_config.sh runs its own
    // olddefconfig internally which can drop options whose dependencies
    // weren't yet resolved in the base config.
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&dot_config)
            .map_err(|e| anyhow::anyhow!("append to .config: {e}"))?;
        f.write_all(b"\n")
            .and_then(|_| f.write_all(KERNEL_CONFIG.as_bytes()))
            .map_err(|e| anyhow::anyhow!("write config fragment: {e}"))?;
    }
    println!(">>> {}", console::style("append stt config").cyan());

    run_step("olddefconfig", "make", &["-C", path_str, "olddefconfig"])?;
    run_step(&format!("build ({jobs})"), "make", &["-C", path_str, &jobs])?;
    println!("{} kernel built", style("done").green().bold());
    Ok(())
}

fn cmd_clean_kernel(args: KernelPathArg) -> Result<()> {
    let path = std::path::Path::new(&args.path)
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("{}: {e}", args.path))?;
    validate_kernel_tree(&path)?;

    println!("{} make mrproper", style(">>>").cyan().bold());
    let status = std::process::Command::new("make")
        .args(["-C", path.to_str().unwrap(), "mrproper"])
        .status()
        .map_err(|e| anyhow::anyhow!("mrproper: {e}"))?;
    if !status.success() {
        anyhow::bail!("mrproper failed (exit {})", status.code().unwrap_or(-1));
    }
    println!("{} clean", style("done").green().bold());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use stt::{monitor, stats, test_support, verify};

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
                "unfair cgroup: spread=85%",
                "stuck 2448ms on cpu4",
                "sched_ext: mitosis disabled (stall)",
            ],
        )]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "FAIL proportional/default (6.7s)");
        assert_eq!(lines[1], "  unfair cgroup: spread=85%");
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

    #[test]
    fn format_results_single_all_fail() {
        let out = format_results(&[sr("x/default", false, 1.0, vec!["err"])]);
        assert!(out.contains("1 failed, 0 passed"));
    }

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

    // -- resolve_flags tests --

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

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

    // -- kernel config tests --

    #[test]
    fn kernel_config_contains_sched_ext() {
        assert!(KERNEL_CONFIG.contains("CONFIG_SCHED_CLASS_EXT=y"));
    }

    #[test]
    fn kernel_config_contains_bpf() {
        assert!(KERNEL_CONFIG.contains("CONFIG_BPF_SYSCALL=y"));
        assert!(KERNEL_CONFIG.contains("CONFIG_BPF_JIT=y"));
    }

    #[test]
    fn kernel_config_contains_btf() {
        assert!(KERNEL_CONFIG.contains("CONFIG_DEBUG_INFO_BTF=y"));
    }

    #[test]
    fn kernel_config_contains_kprobes() {
        assert!(KERNEL_CONFIG.contains("CONFIG_KPROBES=y"));
        assert!(KERNEL_CONFIG.contains("CONFIG_KPROBE_EVENTS=y"));
    }

    #[test]
    fn kernel_config_contains_cgroups() {
        assert!(KERNEL_CONFIG.contains("CONFIG_CGROUPS=y"));
        assert!(KERNEL_CONFIG.contains("CONFIG_CPUSETS=y"));
    }

    #[test]
    fn kernel_config_contains_serial() {
        assert!(KERNEL_CONFIG.contains("CONFIG_SERIAL_8250=y"));
        assert!(KERNEL_CONFIG.contains("CONFIG_SERIAL_8250_CONSOLE=y"));
    }

    #[test]
    fn kernel_config_contains_numa() {
        assert!(KERNEL_CONFIG.contains("CONFIG_SMP=y"));
        assert!(KERNEL_CONFIG.contains("CONFIG_NUMA=y"));
    }

    #[test]
    fn kernel_config_disables_lockdep() {
        assert!(KERNEL_CONFIG.contains("# CONFIG_PROVE_LOCKING is not set"));
    }

    #[test]
    fn kernel_config_disables_psi() {
        assert!(KERNEL_CONFIG.contains("# CONFIG_PSI is not set"));
    }

    #[test]
    fn validate_kernel_tree_missing_makefile() {
        let dir = std::env::temp_dir().join("stt-test-no-makefile");
        let _ = std::fs::create_dir_all(&dir);
        let r = validate_kernel_tree(&dir);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("no Makefile"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_kernel_tree_missing_arch() {
        let dir = std::env::temp_dir().join("stt-test-no-arch");
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(dir.join("Makefile"), "");
        let r = validate_kernel_tree(&dir);
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("no kernel/"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_kernel_tree_valid() {
        let dir = std::env::temp_dir().join("stt-test-valid-tree");
        let _ = std::fs::create_dir_all(dir.join("kernel"));
        let _ = std::fs::write(dir.join("Makefile"), "");
        let r = validate_kernel_tree(&dir);
        assert!(r.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn num_cpus_nonzero() {
        assert!(num_cpus() > 0);
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
        let results = vec![sr("test", false, 1.0, vec!["unfair cgroup: spread=85%"])];
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
                ..Default::default()
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
}
