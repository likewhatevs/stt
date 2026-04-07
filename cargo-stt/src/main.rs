use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use clap::Parser;
use console::style;

use stt::test_support::SttTestInfo;

#[derive(Debug, Parser)]
#[clap(
    name = "cargo-stt",
    bin_name = "cargo stt",
    about = "cargo plugin for stt - scheduler test tools"
)]
struct Cli {
    /// Cargo passes "stt" as the first arg when invoked as `cargo stt`.
    /// Accept and discard it.
    #[clap(hide = true)]
    _subcommand: Option<String>,
    #[clap(subcommand)]
    command: Cmd,
}

#[derive(Debug, Parser)]
enum Cmd {
    /// Run scenarios inside a VM
    Vm(VmArgs),
    /// Run integration tests via nextest with sidecar collection
    Test(TestArgs),
    /// Run tests across topology presets in parallel VMs
    Gauntlet(GauntletArgs),
    /// List registered `#[stt_test]` entries
    List(ListArgs),
    /// Show CPU topology
    Topo,
    /// Probe kernel functions from a crash stack
    Probe(ProbeArgs),
}

#[derive(Debug, Parser)]
struct VmArgs {
    /// Path to kernel image
    #[clap(long)]
    kernel: Option<String>,
    /// Number of sockets
    #[clap(long, default_value = "2")]
    sockets: usize,
    /// Cores per socket
    #[clap(long, default_value = "2")]
    cores: usize,
    /// Threads per core
    #[clap(long, default_value = "2")]
    threads: usize,
    /// Memory in MB
    #[clap(long, default_value = "4096")]
    memory_mb: usize,
    /// Run gauntlet (all scenarios x topology presets)
    #[clap(long)]
    gauntlet: bool,
    /// Max parallel VMs
    #[clap(long)]
    parallel: Option<usize>,
    /// Max infra-failure retries per VM
    #[clap(long, default_value = "3")]
    retries: usize,
    /// Flags to enable (comma-separated short names)
    #[clap(long, value_delimiter = ',')]
    flags: Vec<String>,
    /// Enable all flag combinations
    #[clap(long, conflicts_with = "flags")]
    all_flags: bool,
    /// Scheduler binary path (direct override, skips build)
    #[clap(long, conflicts_with = "package")]
    scheduler_bin: Option<String>,
    /// Build and use scheduler from this cargo package
    #[clap(long, short)]
    package: Option<String>,
    /// Replica multiplier for gauntlet cells
    #[clap(long, default_value = "1")]
    replicas: usize,
    /// Linux source tree with built kernel
    #[clap(long)]
    kernel_dir: Option<String>,
    /// Save results as baseline JSON
    #[clap(long)]
    save_baseline: Option<String>,
    /// Compare against baseline JSON
    #[clap(long)]
    compare: Option<String>,
    /// Work types for gauntlet (comma-separated)
    #[clap(long, value_delimiter = ',')]
    work_types: Vec<String>,
    /// Auto-repro mode
    #[clap(long)]
    auto_repro: bool,
    /// Scenario duration in seconds
    #[clap(long)]
    duration_s: Option<u64>,
    /// Extra arguments passed to stt run
    #[clap(last = true)]
    run_args: Vec<String>,
}

#[derive(Debug, Parser)]
struct TestArgs {
    /// Filter expression passed to nextest
    #[clap(long)]
    filter: Option<String>,
    /// Save results as a baseline JSON file
    #[clap(long)]
    save_baseline: Option<String>,
    /// Compare results against a saved baseline JSON file
    #[clap(long)]
    compare: Option<String>,
    /// Nextest profile
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
    /// Package to test (default: stt)
    #[clap(long, short, default_value = "stt")]
    package: String,
    /// Extra arguments passed through to nextest
    #[clap(last = true)]
    nextest_args: Vec<String>,
}

#[derive(Debug, Parser)]
struct GauntletArgs {
    /// Max parallel VMs
    #[clap(long)]
    parallel: Option<usize>,
    /// Package containing `#[stt_test]` entries
    #[clap(long, short, default_value = "stt")]
    package: String,
    /// Save gauntlet results as a baseline JSON file
    #[clap(long)]
    save_baseline: Option<String>,
    /// Compare against a saved baseline JSON file
    #[clap(long)]
    compare: Option<String>,
    /// Filter tests by name (substring match)
    #[clap(long)]
    filter: Option<String>,
    /// Flags to enable (comma-separated). Constrains which flag profiles
    /// are generated — only profiles containing exactly these flags are run.
    #[clap(long, value_delimiter = ',')]
    flags: Vec<String>,
    /// Work types for cross dimension (comma-separated, e.g. CpuSpin,Bursty).
    #[clap(long, value_delimiter = ',')]
    work_types: Vec<String>,
}

#[derive(Debug, Parser)]
struct ListArgs {
    /// Package containing `#[stt_test]` entries
    #[clap(long, short, default_value = "stt")]
    package: String,
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
    /// Trigger function
    #[clap(long)]
    trigger: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Vm(a) => cmd_vm(a),
        Cmd::Test(a) => cmd_test(a),
        Cmd::Gauntlet(a) => cmd_gauntlet(a),
        Cmd::List(a) => cmd_list(a),
        Cmd::Topo => cmd_topo(),
        Cmd::Probe(a) => cmd_probe(a),
    }
}

// ---------------------------------------------------------------------------
// Test binary discovery
// ---------------------------------------------------------------------------

/// A test entry paired with the binary that contains it.
struct DiscoveredTest {
    info: SttTestInfo,
    binary: PathBuf,
}

/// Discover test binary paths by running `cargo test --no-run`.
fn discover_test_binaries(package: &str) -> Result<Vec<PathBuf>> {
    let output = Command::new("cargo")
        .args(["test", "--no-run", "--message-format=json", "-p", package])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("run cargo test --no-run")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("cargo test --no-run failed:\n{stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut binaries = Vec::new();
    for line in stdout.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line)
            && msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-artifact")
            && msg
                .get("profile")
                .and_then(|p| p.get("test"))
                .and_then(|t| t.as_bool())
                == Some(true)
            && let Some(filenames) = msg.get("filenames").and_then(|f| f.as_array())
        {
            for f in filenames {
                if let Some(path) = f.as_str() {
                    binaries.push(PathBuf::from(path));
                }
            }
        }
    }
    Ok(binaries)
}

/// Query a test binary for registered `#[stt_test]` entries via --stt-list.
fn query_test_entries(binary: &Path) -> Result<Vec<SttTestInfo>> {
    let output = Command::new(binary)
        .arg("--stt-list")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("run --stt-list on {}", binary.display()))?;

    if !output.status.success() {
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let entries: Vec<SttTestInfo> = serde_json::from_str(&stdout).unwrap_or_default();
    Ok(entries)
}

/// Discover all `#[stt_test]` entries with their binary paths.
fn discover_tests_with_binaries(package: &str) -> Result<Vec<DiscoveredTest>> {
    let binaries = discover_test_binaries(package)?;
    let mut all = Vec::new();
    for bin in &binaries {
        let entries = query_test_entries(bin)?;
        for info in entries {
            all.push(DiscoveredTest {
                info,
                binary: bin.clone(),
            });
        }
    }
    Ok(all)
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

/// Build a cargo binary package and return its output path.
///
/// Runs `cargo build -p <package> --message-format=json` and parses
/// compiler-artifact messages to find the binary. Filters for non-test
/// artifacts with `target.kind` containing `"bin"`.
fn build_and_find_binary(package: &str) -> Result<PathBuf> {
    let output = Command::new("cargo")
        .args(["build", "-p", package, "--message-format=json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("run cargo build")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("cargo build -p {package} failed:\n{stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_binary_artifact(&stdout, package)
}

/// Parse cargo JSON output for a binary artifact path.
fn parse_binary_artifact(json_output: &str, package: &str) -> Result<PathBuf> {
    for line in json_output.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line)
            && msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-artifact")
            && msg
                .get("profile")
                .and_then(|p| p.get("test"))
                .and_then(|t| t.as_bool())
                == Some(false)
            && msg
                .get("target")
                .and_then(|t| t.get("kind"))
                .and_then(|k| k.as_array())
                .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("bin")))
            && let Some(filenames) = msg.get("filenames").and_then(|f| f.as_array())
            && let Some(path) = filenames.first().and_then(|f| f.as_str())
        {
            return Ok(PathBuf::from(path));
        }
    }
    anyhow::bail!("no binary artifact found for package '{package}'")
}

fn cmd_vm(args: VmArgs) -> Result<()> {
    // Resolve scheduler binary: -p builds from source, --scheduler-bin is direct.
    let scheduler_bin = if let Some(ref pkg) = args.package {
        println!(
            "{} building scheduler package '{pkg}'",
            style("build").cyan().bold()
        );
        Some(
            build_and_find_binary(pkg)
                .with_context(|| format!("build scheduler package '{pkg}'"))?,
        )
    } else {
        args.scheduler_bin.map(PathBuf::from)
    };

    let stt_bin = find_stt_binary()?;
    let mut cmd = Command::new(&stt_bin);
    cmd.arg("vm");

    if let Some(ref kernel) = args.kernel {
        cmd.args(["--kernel", kernel]);
    }
    cmd.args(["--sockets", &args.sockets.to_string()]);
    cmd.args(["--cores", &args.cores.to_string()]);
    cmd.args(["--threads", &args.threads.to_string()]);
    cmd.args(["--memory-mb", &args.memory_mb.to_string()]);
    cmd.args(["--retries", &args.retries.to_string()]);
    cmd.args(["--replicas", &args.replicas.to_string()]);

    if args.gauntlet {
        cmd.arg("--gauntlet");
    }
    if let Some(par) = args.parallel {
        cmd.args(["--parallel", &par.to_string()]);
    }
    // --flags goes before -- only in gauntlet mode (stt's VmArgs uses
    // flags only for gauntlet). In single-VM mode, flags must go after
    // -- so they reach stt run.
    if args.gauntlet && !args.flags.is_empty() {
        cmd.args(["--flags", &args.flags.join(",")]);
    }
    if let Some(ref bin) = scheduler_bin {
        cmd.args(["--scheduler-bin", &bin.display().to_string()]);
    }
    if let Some(ref kd) = args.kernel_dir {
        cmd.args(["--kernel-dir", kd]);
    }
    if let Some(ref path) = args.save_baseline {
        cmd.args(["--save-baseline", path]);
    }
    if let Some(ref path) = args.compare {
        cmd.args(["--compare", path]);
    }
    if !args.work_types.is_empty() {
        cmd.args(["--work-types", &args.work_types.join(",")]);
    }

    // Build run args: flags forwarded to stt run (after --),
    // plus any trailing args the user passed.
    let mut run_args = Vec::new();
    if args.auto_repro {
        run_args.push("--auto-repro".to_string());
    }
    if let Some(dur) = args.duration_s {
        run_args.push(format!("--duration-s={dur}"));
    }
    if args.all_flags {
        run_args.push("--all-flags".to_string());
    }
    if !args.gauntlet && !args.flags.is_empty() {
        run_args.push(format!("--flags={}", args.flags.join(",")));
    }
    run_args.extend(args.run_args);
    if !run_args.is_empty() {
        cmd.arg("--");
        cmd.args(&run_args);
    }

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    let status = cmd.status().context("run stt vm")?;
    std::process::exit(status.code().unwrap_or(1));
}

fn cmd_test(args: TestArgs) -> Result<()> {
    let sidecar_dir = std::env::temp_dir().join(format!("stt-sidecar-{}", std::process::id()));
    std::fs::create_dir_all(&sidecar_dir)?;

    let has_nextest = Command::new("cargo")
        .args(["nextest", "--version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());

    let mut cmd = Command::new("cargo");
    if has_nextest {
        cmd.args(["nextest", "run", "-p", &args.package]);
        cmd.args(["--profile", &args.nextest_profile]);
    } else {
        cmd.args(["test", "-p", &args.package]);
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

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("spawn test runner: {e}"))?;

    // Scan sidecar directory for results.
    let sidecars = stt::test_support::collect_sidecars(&sidecar_dir);

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
    }

    // Build rows for baseline save/compare.
    let rows: Vec<stt::stats::GauntletRow> =
        sidecars.iter().map(stt::stats::sidecar_to_row).collect();

    if let Some(ref path) = args.save_baseline {
        let baseline = stt::stats::GauntletBaseline {
            scheduler: String::new(),
            timestamp: {
                let d = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                format!("{}", d.as_secs())
            },
            git_commit: None,
            replicas: 1,
            rows: rows.clone(),
        };
        baseline.save(path)?;
        println!(
            "\n{} baseline saved to {path}",
            style("saved").green().bold()
        );
    }

    if let Some(ref path) = args.compare {
        let baseline = stt::stats::GauntletBaseline::load(path)?;
        let report = stt::stats::compare_baselines(&baseline.rows, &rows);
        print!("{report}");
    }

    let _ = std::fs::remove_dir_all(&sidecar_dir);
    std::process::exit(status.code().unwrap_or(1));
}

/// A gauntlet job: test name, binary path, topology, flags.
struct GauntletJob {
    label: String,
    test_name: String,
    binary: PathBuf,
    topo_str: String,
    /// Active flags for this job (empty = default profile).
    flags: Vec<String>,
}

/// Result from a single gauntlet VM run.
struct GauntletResult {
    label: String,
    passed: bool,
    duration_s: f64,
    detail: String,
}

/// Check whether a topology preset satisfies a test's constraints.
fn preset_matches_constraints(preset: &stt::vm::TopoPreset, info: &SttTestInfo) -> bool {
    let t = &preset.topology;
    if t.sockets < info.min_sockets {
        return false;
    }
    if t.num_llcs() < info.min_llcs {
        return false;
    }
    if info.requires_smt && t.threads_per_core < 2 {
        return false;
    }
    if t.total_cpus() < info.min_cpus {
        return false;
    }
    true
}

/// Compute flag profiles for a test. When `--flags` override is set,
/// produce a single profile with exactly those flags. Otherwise
/// enumerate valid profiles from the scheduler's flag declarations
/// constrained by the test's required/excluded flags.
fn compute_profiles(info: &SttTestInfo, cli_flags: &[String]) -> Vec<Vec<String>> {
    if !cli_flags.is_empty() {
        return vec![cli_flags.to_vec()];
    }
    // No scheduler flags => single default (empty) profile.
    if info.scheduler_flags.is_empty() {
        return vec![vec![]];
    }
    let required: Vec<&str> = info.required_flags.iter().map(|s| s.as_str()).collect();
    let excluded: Vec<&str> = info.excluded_flags.iter().map(|s| s.as_str()).collect();
    let optional: Vec<&str> = info
        .scheduler_flags
        .iter()
        .map(|s| s.as_str())
        .filter(|f| !required.contains(f) && !excluded.contains(f))
        .collect();
    // Enumerate power set of optional flags, keeping only profiles
    // where all dependency constraints are satisfied.
    let mut out = Vec::new();
    for mask in 0..(1u32 << optional.len()) {
        let mut fl: Vec<String> = required.iter().map(|s| s.to_string()).collect();
        for (i, &f) in optional.iter().enumerate() {
            if mask & (1 << i) != 0 {
                fl.push(f.to_string());
            }
        }
        // Dependency check: each flag's requires must also be present.
        // Use flag_requires from the FlagDecl system.
        let valid = fl.iter().all(|f| {
            stt::scenario::flags::decl_by_name(f)
                .map(|d| d.requires.iter().all(|r| fl.iter().any(|ff| ff == r.name)))
                .unwrap_or(true)
        });
        if valid {
            out.push(fl);
        }
    }
    if out.is_empty() {
        out.push(vec![]);
    }
    out
}

/// Format a flag profile name: "default" when empty, flags joined by "+".
fn profile_name(flags: &[String]) -> String {
    if flags.is_empty() {
        "default".into()
    } else {
        flags.join("+")
    }
}

fn cmd_gauntlet(args: GauntletArgs) -> Result<()> {
    let discovered = discover_tests_with_binaries(&args.package)?;
    if discovered.is_empty() {
        println!(
            "{} no `#[stt_test]` entries found",
            style("error").red().bold()
        );
        return Ok(());
    }

    let discovered: Vec<&DiscoveredTest> = if let Some(ref filter) = args.filter {
        discovered
            .iter()
            .filter(|t| t.info.name.contains(filter.as_str()))
            .collect()
    } else {
        discovered.iter().collect()
    };

    if discovered.is_empty() {
        println!("{} no tests match filter", style("error").red().bold());
        return Ok(());
    }

    let presets = stt::vm::gauntlet_presets();
    let max_par = args.parallel.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| (n.get() / 8).max(1))
            .unwrap_or(1)
    });

    // Build job matrix: test x matching_topology x flag_profile.
    let mut jobs = Vec::new();
    for test in &discovered {
        let profiles = compute_profiles(&test.info, &args.flags);
        for preset in &presets {
            if !preset_matches_constraints(preset, &test.info) {
                continue;
            }
            let t = &preset.topology;
            let topo_str = format!(
                "{}s{}c{}t",
                t.sockets, t.cores_per_socket, t.threads_per_core
            );
            for profile in &profiles {
                let pname = profile_name(profile);
                let label = format!("{}/{}/{}", preset.name, test.info.name, pname);
                jobs.push(GauntletJob {
                    label,
                    test_name: test.info.name.clone(),
                    binary: test.binary.clone(),
                    topo_str: topo_str.clone(),
                    flags: profile.clone(),
                });
            }
        }
    }

    let total = jobs.len();
    println!(
        "{} gauntlet: {} tests x topologies x profiles = {} VMs, {} parallel",
        style("launching").cyan().bold(),
        discovered.len(),
        total,
        max_par,
    );

    // Sidecar directory for result collection.
    let sidecar_dir = std::env::temp_dir().join(format!("stt-gauntlet-{}", std::process::id()));
    std::fs::create_dir_all(&sidecar_dir)?;

    // Run jobs in parallel.
    let results = run_gauntlet_jobs(jobs, max_par, &sidecar_dir)?;

    // Print summary.
    let passed = results.iter().filter(|r| r.passed).count();
    let failed: Vec<&GauntletResult> = results.iter().filter(|r| !r.passed).collect();

    let total_dur: f64 = results.iter().map(|r| r.duration_s).sum();
    println!(
        "\n=== {}/{} passed ({:.0}s total) ===",
        passed, total, total_dur
    );

    if !failed.is_empty() {
        println!("\nFailed:");
        for r in &failed {
            println!("  {}: {}", r.label, r.detail);
        }
    }

    // Build GauntletRows from per-job sidecar results.
    let mut rows: Vec<stt::stats::GauntletRow> = Vec::new();
    for r in &results {
        let job_dir = sidecar_dir.join(r.label.replace('/', "_"));
        let sidecars = stt::test_support::collect_sidecars(&job_dir);
        for sc in &sidecars {
            rows.push(stt::stats::sidecar_to_row_labeled(sc, &r.label));
        }
    }

    if !rows.is_empty() {
        let report = stt::stats::analyze_rows(&rows);
        print!("{report}");
    }

    // Save baseline if requested.
    if let Some(ref path) = args.save_baseline {
        let baseline = stt::stats::GauntletBaseline {
            scheduler: String::new(),
            timestamp: {
                let d = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                format!("{}", d.as_secs())
            },
            git_commit: None,
            replicas: 1,
            rows: rows.clone(),
        };
        baseline.save(path)?;
        println!(
            "\n{} baseline saved to {path}",
            style("saved").green().bold()
        );
    }

    // Compare against baseline if requested.
    if let Some(ref path) = args.compare {
        let baseline = stt::stats::GauntletBaseline::load(path)?;
        let report = stt::stats::compare_baselines(&baseline.rows, &rows);
        print!("{report}");
    }

    let _ = std::fs::remove_dir_all(&sidecar_dir);

    if !failed.is_empty() {
        std::process::exit(1);
    }
    Ok(())
}

fn run_gauntlet_jobs(
    jobs: Vec<GauntletJob>,
    max_par: usize,
    sidecar_dir: &Path,
) -> Result<Vec<GauntletResult>> {
    use rayon::prelude::*;

    let total = jobs.len();
    let completed = AtomicUsize::new(0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_par)
        .build()
        .map_err(|e| anyhow::anyhow!("create thread pool: {e}"))?;

    let sidecar_dir = sidecar_dir.to_path_buf();
    let results: Vec<GauntletResult> = pool.install(|| {
        jobs.par_iter()
            .map(|job| {
                let start = std::time::Instant::now();

                // Per-job sidecar dir so files don't collide.
                let job_sidecar = sidecar_dir.join(job.label.replace('/', "_"));
                let _ = std::fs::create_dir_all(&job_sidecar);

                let topo_arg = format!("--stt-topo={}", job.topo_str);
                let mut cmd = Command::new(&job.binary);
                cmd.args(["--stt-test-fn", &job.test_name, &topo_arg]);
                if !job.flags.is_empty() {
                    cmd.arg(format!("--stt-flags={}", job.flags.join(",")));
                }
                let output = cmd
                    .env("STT_SIDECAR_DIR", &job_sidecar)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .output();

                let dur = start.elapsed().as_secs_f64();
                let n = completed.fetch_add(1, Ordering::Relaxed) + 1;

                let (ok, detail) = match output {
                    Ok(out) => {
                        let success = out.status.success();
                        let detail = if success {
                            String::new()
                        } else {
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            let stdout = String::from_utf8_lossy(&out.stdout);
                            extract_failure_detail(&stdout, &stderr)
                        };
                        (success, detail)
                    }
                    Err(e) => (false, format!("spawn failed: {e}")),
                };

                let status = if ok { "PASS" } else { "FAIL" };
                let detail_preview = if detail.is_empty() {
                    String::new()
                } else {
                    format!(" | {}", &detail[..detail.len().min(120)])
                };
                println!(
                    "[{n}/{total}] {status} {} ({dur:.0}s){detail_preview}",
                    job.label
                );

                GauntletResult {
                    label: job.label.clone(),
                    passed: ok,
                    duration_s: dur,
                    detail,
                }
            })
            .collect()
    });

    Ok(results)
}

/// Extract a concise failure detail from test binary output.
fn extract_failure_detail(stdout: &str, stderr: &str) -> String {
    // Look for anyhow error messages in stderr first.
    for line in stderr.lines().rev() {
        let trimmed = line.trim();
        if !trimmed.is_empty()
            && !trimmed.starts_with("thread")
            && !trimmed.starts_with("note:")
            && !trimmed.starts_with("stack backtrace")
        {
            return trimmed.to_string();
        }
    }
    // Fall back to last non-empty stdout line.
    for line in stdout.lines().rev() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "unknown failure".to_string()
}

fn cmd_list(args: ListArgs) -> Result<()> {
    let tests = discover_tests_with_binaries(&args.package)?;
    if tests.is_empty() {
        println!(
            "No `#[stt_test]` entries found in package '{}'",
            args.package
        );
        return Ok(());
    }

    println!(
        "{} {} registered test(s):\n",
        style("stt").cyan().bold(),
        tests.len()
    );
    for t in &tests {
        let mut extras = Vec::new();
        if !t.info.required_flags.is_empty() {
            extras.push(format!("req={}", t.info.required_flags.join("+")));
        }
        if !t.info.excluded_flags.is_empty() {
            extras.push(format!("excl={}", t.info.excluded_flags.join("+")));
        }
        let extra_str = if extras.is_empty() {
            String::new()
        } else {
            format!("  {}", extras.join(" "))
        };
        println!(
            "  {:<35} {}s{}c{}t {}MB  sched={} replicas={}{}",
            t.info.name,
            t.info.sockets,
            t.info.cores,
            t.info.threads,
            t.info.memory_mb,
            t.info.scheduler,
            t.info.replicas,
            extra_str,
        );
    }
    Ok(())
}

fn cmd_topo() -> Result<()> {
    stt::topology::print_topo()
}

fn cmd_probe(args: ProbeArgs) -> Result<()> {
    let stt_bin = find_stt_binary()?;
    let mut cmd = Command::new(&stt_bin);
    cmd.arg("probe");
    if args.dmesg {
        cmd.arg("--dmesg");
    }
    if let Some(ref funcs) = args.functions {
        cmd.args(["--functions", funcs]);
    }
    if let Some(ref kd) = args.kernel_dir {
        cmd.args(["--kernel-dir", kd]);
    }
    if args.bootlin {
        cmd.arg("--bootlin");
    }
    if let Some(ref trigger) = args.trigger {
        cmd.args(["--trigger", trigger]);
    }
    if let Some(ref input) = args.input {
        cmd.arg(input);
    }

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let status = cmd.status().context("run stt probe")?;
    std::process::exit(status.code().unwrap_or(1));
}

/// Find the stt binary for delegation.
fn find_stt_binary() -> Result<PathBuf> {
    for dir in &["target/debug", "target/release"] {
        let candidate = PathBuf::from(dir).join("stt");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    let status = Command::new("cargo")
        .args(["build", "-p", "stt"])
        .status()
        .context("build stt binary")?;
    if status.success() {
        let candidate = PathBuf::from("target/debug/stt");
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!("stt binary not found; run `cargo build -p stt` first")
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // extract_failure_detail
    // -----------------------------------------------------------------------

    #[test]
    fn extract_failure_detail_anyhow_error() {
        let stderr = "\
thread 'main' panicked at 'assertion failed'
note: run with RUST_BACKTRACE=1
stack backtrace:
   0: std::backtrace
Error: scheduler exited with signal 11";
        let detail = extract_failure_detail("", stderr);
        assert_eq!(detail, "Error: scheduler exited with signal 11");
    }

    #[test]
    fn extract_failure_detail_stdout_fallback() {
        let stdout = "running test\nFAIL cpuset_disjoint (12.3s)";
        let detail = extract_failure_detail(stdout, "");
        assert_eq!(detail, "FAIL cpuset_disjoint (12.3s)");
    }

    #[test]
    fn extract_failure_detail_empty() {
        let detail = extract_failure_detail("", "");
        assert_eq!(detail, "unknown failure");
    }

    #[test]
    fn extract_failure_detail_skips_noise() {
        let stderr = "\
thread 'main' panicked at 'boom'
note: run with backtrace
stack backtrace:";
        // All stderr lines are noise — falls through to stdout.
        let detail = extract_failure_detail("some output", stderr);
        assert_eq!(detail, "some output");
    }

    // -----------------------------------------------------------------------
    // parse_binary_artifact
    // -----------------------------------------------------------------------

    #[test]
    fn parse_binary_artifact_finds_bin() {
        let json = r#"{"reason":"compiler-message","message":"compiling..."}
{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"scx_mitosis"},"profile":{"test":false},"filenames":["/path/to/target/debug/scx_mitosis"]}
{"reason":"build-finished","success":true}"#;
        let path = parse_binary_artifact(json, "scx_mitosis").unwrap();
        assert_eq!(path, PathBuf::from("/path/to/target/debug/scx_mitosis"));
    }

    #[test]
    fn parse_binary_artifact_skips_test_artifacts() {
        let json = r#"{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"scx_mitosis"},"profile":{"test":true},"filenames":["/path/to/target/debug/scx_mitosis-abc123"]}"#;
        let err = parse_binary_artifact(json, "scx_mitosis").unwrap_err();
        assert!(err.to_string().contains("no binary artifact found"));
    }

    #[test]
    fn parse_binary_artifact_skips_lib() {
        let json = r#"{"reason":"compiler-artifact","target":{"kind":["lib"],"name":"stt"},"profile":{"test":false},"filenames":["/path/to/target/debug/libstt.rlib"]}"#;
        let err = parse_binary_artifact(json, "stt").unwrap_err();
        assert!(err.to_string().contains("no binary artifact found"));
    }

    #[test]
    fn parse_binary_artifact_no_artifacts() {
        let json = r#"{"reason":"build-finished","success":true}"#;
        let err = parse_binary_artifact(json, "foo").unwrap_err();
        assert!(err.to_string().contains("no binary artifact found"));
    }

    #[test]
    fn parse_binary_artifact_empty_filenames() {
        let json = r#"{"reason":"compiler-artifact","target":{"kind":["bin"],"name":"foo"},"profile":{"test":false},"filenames":[]}"#;
        let err = parse_binary_artifact(json, "foo").unwrap_err();
        assert!(err.to_string().contains("no binary artifact found"));
    }

    // -----------------------------------------------------------------------
    // CLI parsing
    // -----------------------------------------------------------------------

    #[test]
    fn cli_vm_package_flag() {
        let cli = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "vm",
            "-p",
            "scx_mitosis",
            "--sockets",
            "1",
            "--cores",
            "2",
        ])
        .unwrap();
        match cli.command {
            Cmd::Vm(args) => {
                assert_eq!(args.package.as_deref(), Some("scx_mitosis"));
                assert_eq!(args.sockets, 1);
                assert_eq!(args.cores, 2);
                assert!(args.scheduler_bin.is_none());
            }
            _ => panic!("expected Vm"),
        }
    }

    #[test]
    fn cli_vm_scheduler_bin_flag() {
        let cli =
            Cli::try_parse_from(["cargo-stt", "stt", "vm", "--scheduler-bin", "/path/to/bin"])
                .unwrap();
        match cli.command {
            Cmd::Vm(args) => {
                assert_eq!(args.scheduler_bin.as_deref(), Some("/path/to/bin"));
                assert!(args.package.is_none());
            }
            _ => panic!("expected Vm"),
        }
    }

    #[test]
    fn cli_vm_package_and_scheduler_bin_conflict() {
        let result = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "vm",
            "-p",
            "scx_mitosis",
            "--scheduler-bin",
            "/path/to/bin",
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_vm_defaults() {
        let cli = Cli::try_parse_from(["cargo-stt", "stt", "vm"]).unwrap();
        match cli.command {
            Cmd::Vm(args) => {
                assert_eq!(args.sockets, 2);
                assert_eq!(args.cores, 2);
                assert_eq!(args.threads, 2);
                assert_eq!(args.memory_mb, 4096);
                assert_eq!(args.retries, 3);
                assert_eq!(args.replicas, 1);
                assert!(!args.gauntlet);
                assert!(!args.all_flags);
                assert!(args.flags.is_empty());
                assert!(args.package.is_none());
                assert!(args.scheduler_bin.is_none());
            }
            _ => panic!("expected Vm"),
        }
    }

    #[test]
    fn cli_vm_run_args() {
        let cli = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "vm",
            "--",
            "proportional",
            "--flags=borrow,rebal",
        ])
        .unwrap();
        match cli.command {
            Cmd::Vm(args) => {
                assert_eq!(args.run_args, vec!["proportional", "--flags=borrow,rebal"]);
            }
            _ => panic!("expected Vm"),
        }
    }

    #[test]
    fn cli_vm_gauntlet_with_flags() {
        let cli = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "vm",
            "--gauntlet",
            "--flags",
            "borrow,rebal",
            "--parallel",
            "4",
        ])
        .unwrap();
        match cli.command {
            Cmd::Vm(args) => {
                assert!(args.gauntlet);
                assert_eq!(args.flags, vec!["borrow", "rebal"]);
                assert_eq!(args.parallel, Some(4));
            }
            _ => panic!("expected Vm"),
        }
    }

    // -----------------------------------------------------------------------
    // find_stt_binary
    // -----------------------------------------------------------------------

    /// Guard to serialize tests that call set_current_dir, which is
    /// process-global and races with parallel test execution.
    static CWD_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn find_stt_binary_in_temp_dir() {
        let _lock = CWD_MUTEX.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("stt-test-{}", std::process::id()));
        let debug_dir = dir.join("target/debug");
        std::fs::create_dir_all(&debug_dir).unwrap();
        let stt_path = debug_dir.join("stt");
        std::fs::write(&stt_path, "fake").unwrap();

        let orig = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let result = find_stt_binary();
        std::env::set_current_dir(&orig).unwrap();
        let _ = std::fs::remove_dir_all(&dir);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), PathBuf::from("target/debug/stt"));
    }

    #[test]
    fn cli_vm_auto_repro_and_duration() {
        let cli = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "vm",
            "--auto-repro",
            "--duration-s",
            "60",
        ])
        .unwrap();
        match cli.command {
            Cmd::Vm(args) => {
                assert!(args.auto_repro);
                assert_eq!(args.duration_s, Some(60));
            }
            _ => panic!("expected Vm"),
        }
    }

    #[test]
    fn cli_vm_all_flags_and_flags_conflict() {
        let result =
            Cli::try_parse_from(["cargo-stt", "stt", "vm", "--all-flags", "--flags", "borrow"]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // profile_name
    // -----------------------------------------------------------------------

    #[test]
    fn profile_name_empty() {
        assert_eq!(profile_name(&[]), "default");
    }

    #[test]
    fn profile_name_single() {
        assert_eq!(profile_name(&["borrow".into()]), "borrow");
    }

    #[test]
    fn profile_name_multiple() {
        assert_eq!(
            profile_name(&["borrow".into(), "rebal".into()]),
            "borrow+rebal"
        );
    }

    // -----------------------------------------------------------------------
    // preset_matches_constraints
    // -----------------------------------------------------------------------

    fn test_info(
        min_sockets: u32,
        min_llcs: u32,
        requires_smt: bool,
        min_cpus: u32,
    ) -> SttTestInfo {
        SttTestInfo {
            name: "test".into(),
            sockets: 1,
            cores: 2,
            threads: 1,
            memory_mb: 2048,
            scheduler: "eevdf".into(),
            replicas: 1,
            required_flags: vec![],
            excluded_flags: vec![],
            min_sockets,
            min_llcs,
            requires_smt,
            min_cpus,
            scheduler_flags: vec![],
        }
    }

    #[test]
    fn preset_matches_defaults() {
        let preset = &stt::vm::gauntlet_presets()[0];
        let info = test_info(1, 1, false, 1);
        assert!(preset_matches_constraints(preset, &info));
    }

    #[test]
    fn preset_rejects_insufficient_sockets() {
        let preset = &stt::vm::gauntlet_presets()[0]; // tiny presets are 1 socket
        let info = test_info(4, 1, false, 1);
        assert!(!preset_matches_constraints(preset, &info));
    }

    #[test]
    fn preset_rejects_insufficient_cpus() {
        let preset = &stt::vm::gauntlet_presets()[0]; // tiny presets have few CPUs
        let info = test_info(1, 1, false, 999);
        assert!(!preset_matches_constraints(preset, &info));
    }

    // -----------------------------------------------------------------------
    // compute_profiles
    // -----------------------------------------------------------------------

    #[test]
    fn compute_profiles_no_scheduler_flags() {
        let info = test_info(1, 1, false, 1);
        let profiles = compute_profiles(&info, &[]);
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0].is_empty());
    }

    #[test]
    fn compute_profiles_cli_override() {
        let info = test_info(1, 1, false, 1);
        let flags = vec!["borrow".to_string(), "rebal".to_string()];
        let profiles = compute_profiles(&info, &flags);
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0], flags);
    }

    // -----------------------------------------------------------------------
    // gauntlet CLI parsing
    // -----------------------------------------------------------------------

    #[test]
    fn cli_gauntlet_with_flags() {
        let cli = Cli::try_parse_from(["cargo-stt", "stt", "gauntlet", "--flags", "borrow,rebal"])
            .unwrap();
        match cli.command {
            Cmd::Gauntlet(args) => {
                assert_eq!(args.flags, vec!["borrow", "rebal"]);
            }
            _ => panic!("expected Gauntlet"),
        }
    }

    #[test]
    fn cli_gauntlet_with_work_types() {
        let cli = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "gauntlet",
            "--work-types",
            "CpuSpin,Bursty",
        ])
        .unwrap();
        match cli.command {
            Cmd::Gauntlet(args) => {
                assert_eq!(args.work_types, vec!["CpuSpin", "Bursty"]);
            }
            _ => panic!("expected Gauntlet"),
        }
    }

    #[test]
    fn cli_gauntlet_defaults() {
        let cli = Cli::try_parse_from(["cargo-stt", "stt", "gauntlet"]).unwrap();
        match cli.command {
            Cmd::Gauntlet(args) => {
                assert!(args.flags.is_empty());
                assert!(args.work_types.is_empty());
                assert!(args.filter.is_none());
                assert_eq!(args.package, "stt");
            }
            _ => panic!("expected Gauntlet"),
        }
    }
}
