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
    /// Load scheduler BPF programs and report verifier statistics
    Verifier(VerifierArgs),
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
    /// Replica multiplier for gauntlet cgroups
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

#[derive(Debug, Parser)]
struct VerifierArgs {
    /// Cargo package containing the scheduler
    #[clap(long, short, default_value = "stt-sched")]
    package: String,
    /// Second package for A/B instruction count delta
    #[clap(long)]
    diff: Option<String>,
    /// Full verifier log output
    #[clap(long, short)]
    verbose: bool,
    /// Path to kernel image for the VM
    #[clap(long)]
    kernel: Option<String>,
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
        Cmd::Verifier(a) => cmd_verifier(a),
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

/// Generate a nextest tool config that assigns `threads-required` to
/// performance_mode tests based on host LLC topology. This prevents
/// nextest from scheduling more perf tests in parallel than the host
/// can support without CPU contention.
///
/// The pinning plan maps each virtual socket to a physical LLC group,
/// so threads-required = sum(llc_groups[i].cpus.len() for i in 0..sockets) + 1.
/// The per-group sum handles asymmetric LLCs correctly. Uses cpus.len()
/// (logical CPUs / hardware threads), not physical core count.
///
/// Returns the path to the generated temp file.
/// Result of generating a perf tool config. Holds the path and, when
/// backed by memfd, the open file descriptor that keeps the memfd alive.
struct PerfToolConfig {
    path: PathBuf,
    /// Kept open so `/proc/self/fd/N` remains valid until nextest exits.
    _memfd: Option<std::fs::File>,
}

fn generate_perf_tool_config(package: &str) -> Result<PerfToolConfig> {
    let tests = discover_tests_with_binaries(package)?;
    let perf_tests: Vec<&DiscoveredTest> = tests
        .iter()
        .filter(|t| t.info.performance_mode && t.info.total_vcpus > 0)
        .collect();

    let host_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut config = String::new();
    config.push_str(&format!(
        "[test-groups.\"@tool:stt:perf\"]\nmax-threads = {host_cpus}\n"
    ));

    // Read host topology for LLC-aware thread reservation.
    let host_topo = stt::vmm::host_topology::HostTopology::from_sysfs().ok();

    // Group perf tests by threads-required.
    // Pinning maps each virtual socket to a physical LLC group.
    // threads-required = sum of cpus.len() for the first N LLC groups + 1
    // (service CPU). Falls back to vcpus + 1 if host topology is unavailable.
    // Note: threads-required prevents total CPU overcommit but does not
    // enforce per-LLC exclusion. Two tests using 1 socket each could be
    // scheduled on the same LLC group by nextest. super_perf_mode's
    // exclusivity is enforced at VM build time, not at scheduling time.
    let mut by_threads: std::collections::BTreeMap<u32, Vec<&str>> =
        std::collections::BTreeMap::new();
    for t in &perf_tests {
        let threads_required = if let Some(ref topo) = host_topo {
            let llcs_used = t.info.sockets as usize;
            let reserved: usize = topo
                .llc_groups
                .iter()
                .take(llcs_used)
                .map(|g| g.cpus.len())
                .sum();
            reserved as u32 + 1
        } else {
            t.info.total_vcpus + 1
        };
        by_threads
            .entry(threads_required)
            .or_default()
            .push(&t.info.name);
    }
    for (threads_required, names) in &by_threads {
        let filter: String = names
            .iter()
            .map(|n| format!("test(={n})"))
            .collect::<Vec<_>>()
            .join(" | ");
        config.push_str(&format!(
            "\n[[profile.default.overrides]]\nfilter = \"{filter}\"\n\
             threads-required = {threads_required}\n\
             test-group = \"@tool:stt:perf\"\n"
        ));
    }

    write_perf_tool_config(&config)
}

/// Write tool config to memfd (in-memory, auto-cleaned on exit).
/// Falls back to a temp file if memfd_create is unavailable.
fn write_perf_tool_config(config: &str) -> Result<PerfToolConfig> {
    use std::io::Write;
    use std::os::unix::io::FromRawFd;

    let fd = unsafe { libc::memfd_create(c"stt-nextest".as_ptr(), 0) };
    if fd >= 0 {
        let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
        file.write_all(config.as_bytes())
            .context("write memfd tool config")?;
        let path = PathBuf::from(format!("/proc/self/fd/{fd}"));
        return Ok(PerfToolConfig {
            path,
            _memfd: Some(file),
        });
    }

    // Fallback: temp file on disk.
    let path = std::env::temp_dir().join(format!("stt-nextest-{}.toml", std::process::id()));
    std::fs::write(&path, config)?;
    Ok(PerfToolConfig { path, _memfd: None })
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

    // Generate nextest tool config for performance_mode tests so they
    // reserve LLC group CPUs (+1 for service) via threads-required.
    // _tool_config must outlive cmd.status() so the memfd stays open.
    let _tool_config = if has_nextest {
        let tc = generate_perf_tool_config(&args.package).ok();
        if let Some(ref tc) = tc {
            cmd.arg("--tool-config-file");
            cmd.arg(format!("stt:{}", tc.path.display()));
        }
        tc
    } else {
        None
    };

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
            policies: vec![],
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
    // Clean up temp file fallback (memfd is auto-cleaned on drop).
    if let Some(ref tc) = _tool_config
        && tc._memfd.is_none()
    {
        let _ = std::fs::remove_file(&tc.path);
    }
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
            policies: vec![],
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

// ---------------------------------------------------------------------------
// Verifier subcommand
// ---------------------------------------------------------------------------

/// Parsed verifier stats from the kernel log line:
/// `processed N insns (limit M) max_states_per_insn X total_states Y peak_states Z mark_read W`
struct VerifierStats {
    processed_insns: u64,
    total_states: u64,
    peak_states: u64,
    time_usec: Option<u64>,
    stack_depth: Option<String>,
}

/// Parse verifier stats from the log output.
///
/// The kernel always emits a "processed N insns ..." line. When
/// BPF_LOG_STATS is set, it also emits "verification time" and
/// "stack depth" lines.
fn parse_verifier_stats(log: &str) -> VerifierStats {
    let mut stats = VerifierStats {
        processed_insns: 0,
        total_states: 0,
        peak_states: 0,
        time_usec: None,
        stack_depth: None,
    };

    let mut found_insns = false;
    let mut found_time = false;
    let mut found_stack = false;

    for line in log.lines().rev() {
        if !found_insns && line.starts_with("processed ") {
            found_insns = true;
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.len() >= 2 {
                stats.processed_insns = words[1].parse().unwrap_or(0);
            }
            for (i, &w) in words.iter().enumerate() {
                if w == "total_states"
                    && let Some(v) = words.get(i + 1)
                {
                    stats.total_states = v.parse().unwrap_or(0);
                }
                if w == "peak_states"
                    && let Some(v) = words.get(i + 1)
                {
                    stats.peak_states = v.parse().unwrap_or(0);
                }
            }
        }
        if !found_time && line.contains("verification time") {
            found_time = true;
            for word in line.split_whitespace() {
                if let Ok(n) = word.parse::<u64>() {
                    stats.time_usec = Some(n);
                    break;
                }
            }
        }
        if !found_stack && line.contains("stack depth") {
            found_stack = true;
            if let Some(pos) = line.find("stack depth") {
                let after = &line[pos + "stack depth".len()..];
                let depth_str = after.trim();
                if !depth_str.is_empty() {
                    stats.stack_depth = Some(depth_str.to_string());
                }
            }
        }
        if found_insns && found_time && found_stack {
            break;
        }
    }

    stats
}

/// Normalize a BPF verifier log line by stripping variable register-state
/// annotations so that lines from different loop iterations compare equal.
///
/// Handles:
/// - Instruction with `; frame` annotation: `3006: (07) r9 += 1  ; frame1: R9_w=2`
/// - Instruction with `; R` + digit annotation: `9: (15) if r7 == 0x0 goto pc+1  ; R7=scalar(...)`
/// - Branch with inline target state: `3026: (b5) if r6 <= 0x11dc0 goto pc+2 3029: frame1: R0=1`
/// - Standalone register dump with frame: `3041: frame1: R0_w=scalar()`
/// - Standalone register dump without frame: `3029: R0=1 R6=scalar()`
///
/// Preserves source comments (`; for (int j = 0; ...)`) and non-annotation
/// semicolons (`; Return value`) -- these serve as cycle anchors.
fn normalize_verifier_line(line: &str) -> &str {
    let trimmed = line.trim();
    if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
        return trimmed;
    }
    // "3041: frame1: ..." or "3041: R0_w=scalar()" — standalone register dump.
    // State-only lines; keep just the instruction index.
    if let Some(colon) = trimmed.find(": ") {
        let after = &trimmed[colon + 2..];
        if after.starts_with("frame")
            || (after.starts_with('R')
                && after.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit()))
        {
            return &trimmed[..colon + 1];
        }
    }
    // "; frame" annotation on instruction line
    if let Some(pos) = trimmed.find("; frame") {
        return trimmed[..pos].trim_end();
    }
    // "; R" followed by digit — register annotation without frame prefix
    if let Some(pos) = trimmed.find("; R")
        && trimmed
            .as_bytes()
            .get(pos + 3)
            .is_some_and(|b| b.is_ascii_digit())
    {
        return trimmed[..pos].trim_end();
    }
    // Inline branch-target state: "goto pc+2 3029: frame1: ..."
    if let Some(goto_pos) = trimmed.find("goto pc") {
        let after_goto = &trimmed[goto_pos + 7..];
        let end = after_goto
            .find(|c: char| c != '+' && c != '-' && !c.is_ascii_digit())
            .unwrap_or(after_goto.len());
        let insn_end = goto_pos + 7 + end;
        if insn_end < trimmed.len() {
            return trimmed[..insn_end].trim_end();
        }
    }
    trimmed
}

/// Detect a single repeating cycle in a slice of lines.
///
/// Returns `Some((start, period, count))` where the cycle begins at
/// `start`, each iteration is `period` lines, and it repeats `count` times.
fn detect_cycle(lines: &[&str]) -> Option<(usize, usize, usize)> {
    const MIN_PERIOD: usize = 5;
    const MIN_REPS: usize = 6;

    if lines.len() < MIN_PERIOD * MIN_REPS {
        return None;
    }

    let normalized: Vec<&str> = lines.iter().map(|l| normalize_verifier_line(l)).collect();

    // Find most frequent non-trivial normalized line (the "anchor").
    let mut sorted_norms: Vec<&str> = normalized
        .iter()
        .filter(|l| l.len() >= 10)
        .copied()
        .collect();
    sorted_norms.sort_unstable();

    let mut best_anchor: Option<(&str, usize)> = None;
    let mut i = 0;
    while i < sorted_norms.len() {
        let mut j = i + 1;
        while j < sorted_norms.len() && sorted_norms[j] == sorted_norms[i] {
            j += 1;
        }
        let count = j - i;
        if count >= MIN_REPS && best_anchor.is_none_or(|(_, best)| count > best) {
            best_anchor = Some((sorted_norms[i], count));
        }
        i = j;
    }

    let (anchor, _) = best_anchor?;

    let positions: Vec<usize> = normalized
        .iter()
        .enumerate()
        .filter(|(_, l)| **l == anchor)
        .map(|(i, _)| i)
        .collect();

    // Try strides 1..3 to handle anchors appearing K times per cycle.
    for stride in 1..=3usize {
        if positions.len() <= stride {
            continue;
        }

        let mut gaps: Vec<usize> = positions
            .windows(stride + 1)
            .map(|w| w[stride] - w[0])
            .filter(|g| *g >= MIN_PERIOD)
            .collect();
        gaps.sort_unstable();

        let mut best_period = 0;
        let mut best_gap_count = 0;
        let mut gi = 0;
        while gi < gaps.len() {
            let mut gj = gi + 1;
            while gj < gaps.len() && gaps[gj] == gaps[gi] {
                gj += 1;
            }
            let count = gj - gi;
            if count > best_gap_count {
                best_gap_count = count;
                best_period = gaps[gi];
            }
            gi = gj;
        }
        if best_period == 0 || best_gap_count < MIN_REPS - 1 {
            continue;
        }
        let period = best_period;

        for &pos in &positions {
            if pos + 2 * period > lines.len() {
                break;
            }
            if normalized[pos..pos + period] == normalized[pos + period..pos + 2 * period] {
                let first_block = &normalized[pos..pos + period];
                let mut count = 1;
                while pos + (count + 1) * period <= lines.len() {
                    if normalized[pos + count * period..pos + (count + 1) * period] != *first_block
                    {
                        break;
                    }
                    count += 1;
                }
                // Try earlier starts to find best alignment.
                let mut best_start = pos;
                let mut best_count = count;
                for offset in 1..period {
                    let Some(cand) = pos.checked_sub(offset) else {
                        break;
                    };
                    if cand + 2 * period > lines.len() {
                        continue;
                    }
                    if normalized[cand..cand + period]
                        != normalized[cand + period..cand + 2 * period]
                    {
                        continue;
                    }
                    let mut c = 2;
                    while cand + (c + 1) * period <= lines.len()
                        && normalized[cand + c * period..cand + (c + 1) * period]
                            == normalized[cand..cand + period]
                    {
                        c += 1;
                    }
                    if c > best_count {
                        best_start = cand;
                        best_count = c;
                    }
                }
                if best_count >= MIN_REPS {
                    return Some((best_start, period, best_count));
                }
            }
        }
    }

    None
}

/// Collapse repeating cycles in a verifier log.
///
/// Runs cycle detection iteratively (up to 5 passes for nested loops).
/// Each cycle is replaced with a header (`--- Nx of the following M lines ---`),
/// the first iteration, an omission marker (`--- N identical iterations omitted ---`),
/// the last iteration, and an end marker (`--- end repeat ---`).
fn collapse_cycles(log: &str) -> String {
    const MAX_PASSES: usize = 5;
    let mut text = log.to_string();

    for _ in 0..MAX_PASSES {
        let lines: Vec<&str> = text.lines().collect();
        let (start, period, count) = match detect_cycle(&lines) {
            Some(c) => c,
            None => break,
        };

        let mut out = String::new();
        for line in &lines[..start] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {}x of the following {} lines ---\n",
            count, period
        ));
        for line in &lines[start..start + period] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&format!(
            "--- {} identical iterations omitted ---\n",
            count - 2
        ));
        let last_start = start + (count - 1) * period;
        for line in &lines[last_start..last_start + period] {
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("--- end repeat ---\n");
        let suffix_start = start + count * period;
        for line in &lines[suffix_start..] {
            out.push_str(line);
            out.push('\n');
        }
        text = out;
    }

    text
}

/// Format a single program's brief output line (without ANSI color).
fn format_brief_line(name: &str, insn_cnt: usize, vs: &VerifierStats) -> String {
    let mut extra = String::new();
    if vs.total_states > 0 {
        extra.push_str(&format!("  states={}/{}", vs.peak_states, vs.total_states));
    }
    if let Some(t) = vs.time_usec {
        extra.push_str(&format!("  time={t}us"));
    }
    if let Some(ref s) = vs.stack_depth {
        extra.push_str(&format!("  stack={s}"));
    }
    format!(
        "  {:<40} insns={:<6} processed={:<6}{}",
        name, insn_cnt, vs.processed_insns, extra
    )
}

/// Boot a scheduler in a VM and capture its verifier output from dmesg.
///
/// The scheduler loads BPF programs during startup; the kernel writes
/// verifier stats to dmesg on COM1. Returns the captured output.
/// Per-program verifier statistics parsed from VM output.
struct ProgStats {
    name: String,
    /// Pre-verification program size (BPF insns).
    insn_cnt: usize,
    /// Verifier log (stats-only or full, depending on log level).
    log: String,
}

/// Parse structured verifier output from a VM run.
///
/// The scheduler binary emits lines when invoked with `--dump-verifier`:
///   STT_VERIFIER_PROG <name> insn_cnt=<N>
///   STT_VERIFIER_LOG <name> <log line>
///   STT_VERIFIER_DONE
fn parse_vm_verifier_output(output: &str) -> Vec<ProgStats> {
    let mut stats: Vec<ProgStats> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_insn_cnt = 0usize;
    let mut current_log = String::new();

    for line in output.lines() {
        if let Some(rest) = line.strip_prefix("STT_VERIFIER_PROG ") {
            if let Some(name) = current_name.take() {
                stats.push(ProgStats {
                    name,
                    insn_cnt: current_insn_cnt,
                    log: std::mem::take(&mut current_log),
                });
            }
            let parts: Vec<&str> = rest.splitn(2, ' ').collect();
            current_name = Some(parts[0].to_string());
            current_insn_cnt = parts
                .get(1)
                .and_then(|s| s.strip_prefix("insn_cnt="))
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            current_log.clear();
        } else if let Some(rest) = line.strip_prefix("STT_VERIFIER_LOG ") {
            if let Some((_, log_line)) = rest.split_once(' ') {
                if !current_log.is_empty() {
                    current_log.push('\n');
                }
                current_log.push_str(log_line);
            }
        } else if line.starts_with("STT_VERIFIER_DONE")
            && let Some(name) = current_name.take()
        {
            stats.push(ProgStats {
                name,
                insn_cnt: current_insn_cnt,
                log: std::mem::take(&mut current_log),
            });
        }
    }
    if let Some(name) = current_name {
        stats.push(ProgStats {
            name,
            insn_cnt: current_insn_cnt,
            log: current_log,
        });
    }
    stats
}

/// A single row in the A/B diff output.
struct DiffRow {
    name: String,
    a: u64,
    b: u64,
    delta: i64,
}

/// Build diff rows from A stats and B lookup map.
fn build_diff_rows(
    stats_a: &[ProgStats],
    b_map: &std::collections::HashMap<String, u64>,
) -> Vec<DiffRow> {
    let mut rows = Vec::new();
    for ps in stats_a {
        let a = parse_verifier_stats(&ps.log).processed_insns;
        let b = b_map.get(&ps.name).copied().unwrap_or(0);
        rows.push(DiffRow {
            name: ps.name.clone(),
            a,
            b,
            delta: a as i64 - b as i64,
        });
    }
    rows
}

/// Build the B-side lookup map from collected stats.
fn build_b_map(stats_b: &[ProgStats]) -> std::collections::HashMap<String, u64> {
    stats_b
        .iter()
        .map(|ps| {
            let vs = parse_verifier_stats(&ps.log);
            (ps.name.clone(), vs.processed_insns)
        })
        .collect()
}

/// Build a scheduler, boot it in a VM with `--dump-verifier`, and
/// parse the structured verifier output.
fn collect_verifier_via_vm(
    package: &str,
    verbose: bool,
    kernel: Option<&str>,
) -> Result<Vec<ProgStats>> {
    println!(
        "{} building scheduler package '{package}'",
        style("build").cyan().bold(),
    );
    let sched_bin = build_and_find_binary(package)
        .with_context(|| format!("build scheduler package '{package}'"))?;

    let kernel_path = if let Some(k) = kernel {
        PathBuf::from(k)
    } else {
        stt::find_kernel()
            .ok_or_else(|| anyhow::anyhow!("no kernel found; use --kernel to specify one"))?
    };

    let mut sched_args = vec!["--dump-verifier".to_string()];
    if verbose {
        sched_args.push("--dump-verifier-verbose".to_string());
    }

    let vm = stt::vmm::SttVm::builder()
        .kernel(&kernel_path)
        .scheduler_binary(&sched_bin)
        .sched_args(&sched_args)
        .topology(1, 1, 1)
        .memory_mb(2048)
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("build verifier VM")?;

    println!(
        "{} booting VM with scheduler from '{package}'",
        style("verifier").cyan().bold(),
    );
    let result = vm.run().context("run verifier VM")?;

    if !result.success && !result.output.contains("STT_VERIFIER_DONE") {
        anyhow::bail!(
            "verifier VM exited with code {} (timed_out={})\n{}",
            result.exit_code,
            result.timed_out,
            result.output,
        );
    }

    Ok(parse_vm_verifier_output(&result.output))
}

fn cmd_verifier(args: VerifierArgs) -> Result<()> {
    let verbose = args.verbose;

    let stats_a = collect_verifier_via_vm(&args.package, verbose, args.kernel.as_deref())?;

    println!("\n{}", style(&args.package).bold());
    for ps in &stats_a {
        let vs = parse_verifier_stats(&ps.log);
        println!("{}", format_brief_line(&ps.name, ps.insn_cnt, &vs));
    }

    for ps in &stats_a {
        if !ps.log.is_empty() {
            println!(
                "\n{}  {}",
                style(&args.package).bold(),
                style(&ps.name).cyan()
            );
            if verbose {
                print!("{}", ps.log);
            } else {
                print!("{}", collapse_cycles(&ps.log));
            }
        }
    }

    // A/B comparison mode.
    if let Some(ref diff_pkg) = args.diff {
        let stats_b = collect_verifier_via_vm(diff_pkg, verbose, args.kernel.as_deref())?;

        println!("\n{}", style(diff_pkg).bold());
        for ps in &stats_b {
            let vs = parse_verifier_stats(&ps.log);
            println!("{}", format_brief_line(&ps.name, ps.insn_cnt, &vs));
        }

        let b_map = build_b_map(&stats_b);
        let diff_rows = build_diff_rows(&stats_a, &b_map);

        println!(
            "\n{} A/B diff: {} vs {diff_pkg}",
            style("delta").cyan().bold(),
            args.package,
        );
        println!(
            "  {:<40} {:>10} {:>10} {:>10}",
            "program", "A", "B", "delta"
        );
        println!("  {}", "-".repeat(72));

        for row in &diff_rows {
            println!(
                "  {:<40} {:>10} {:>10} {:>+10}",
                row.name, row.a, row.b, row.delta
            );
        }
    }

    Ok(())
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
            performance_mode: false,
            super_perf_mode: false,
            total_vcpus: 2,
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

    // -----------------------------------------------------------------------
    // verifier subcommand
    // -----------------------------------------------------------------------

    #[test]
    fn cli_verifier_defaults() {
        let cli = Cli::try_parse_from(["cargo-stt", "stt", "verifier"]).unwrap();
        match cli.command {
            Cmd::Verifier(args) => {
                assert_eq!(args.package, "stt-sched");
                assert!(args.diff.is_none());
                assert!(!args.verbose);
            }
            _ => panic!("expected Verifier"),
        }
    }

    #[test]
    fn cli_verifier_with_flags() {
        let cli = Cli::try_parse_from([
            "cargo-stt",
            "stt",
            "verifier",
            "-p",
            "my_sched",
            "--diff",
            "other_sched",
            "-v",
        ])
        .unwrap();
        match cli.command {
            Cmd::Verifier(args) => {
                assert_eq!(args.package, "my_sched");
                assert_eq!(args.diff.as_deref(), Some("other_sched"));
                assert!(args.verbose);
            }
            _ => panic!("expected Verifier"),
        }
    }

    #[test]
    fn parse_verifier_stats_full_line() {
        let log = "processed 1234 insns (limit 1000000) max_states_per_insn 5 total_states 200 peak_states 50 mark_read 10\nverification time 42 usec\nstack depth 32+0\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 1234);
        assert_eq!(vs.total_states, 200);
        assert_eq!(vs.peak_states, 50);
        assert_eq!(vs.time_usec, Some(42));
        assert_eq!(vs.stack_depth.as_deref(), Some("32+0"));
    }

    #[test]
    fn parse_verifier_stats_insns_only() {
        let log = "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states 3 mark_read 0\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 500);
        assert_eq!(vs.total_states, 10);
        assert_eq!(vs.peak_states, 3);
        assert!(vs.time_usec.is_none());
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_empty() {
        let vs = super::parse_verifier_stats("");
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
        assert!(vs.time_usec.is_none());
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_garbage_lines() {
        let log = "some random output\nnot a stats line\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert!(vs.time_usec.is_none());
    }

    #[test]
    fn parse_verifier_stats_time_without_insns() {
        // Time line present but no processed insns line.
        let log = "verification time 100 usec\nstack depth 64\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.time_usec, Some(100));
        assert_eq!(vs.stack_depth.as_deref(), Some("64"));
    }

    #[test]
    fn parse_verifier_stats_multi_subprogram_stack() {
        let log = "processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\nstack depth 32+16+8\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 42);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+16+8"));
    }

    #[test]
    fn parse_verifier_stats_noise_between_lines() {
        let log = "\
libbpf: loading something
processed 999 insns (limit 1000000) max_states_per_insn 3 total_states 77 peak_states 20 mark_read 5
libbpf: prog 'dispatch': attached
verification time 7 usec
stack depth 48+0
";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 999);
        assert_eq!(vs.total_states, 77);
        assert_eq!(vs.peak_states, 20);
        assert_eq!(vs.time_usec, Some(7));
        assert_eq!(vs.stack_depth.as_deref(), Some("48+0"));
    }

    #[test]
    fn parse_verifier_stats_partial_insns_line() {
        // Truncated: only "processed N" without the rest.
        let log = "processed 123\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 123);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_only_stack_depth() {
        let log = "stack depth 128\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("128"));
        assert_eq!(vs.processed_insns, 0);
    }

    #[test]
    fn parse_verifier_stats_zero_insns() {
        let log = "processed 0 insns (limit 1000000) max_states_per_insn 0 total_states 0 peak_states 0 mark_read 0\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_large_values() {
        let log = "processed 999999 insns (limit 1000000) max_states_per_insn 100 total_states 50000 peak_states 12345 mark_read 9999\nverification time 123456 usec\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 999999);
        assert_eq!(vs.total_states, 50000);
        assert_eq!(vs.peak_states, 12345);
        assert_eq!(vs.time_usec, Some(123456));
    }

    #[test]
    fn parse_verifier_stats_stack_depth_single() {
        let log = "stack depth 64\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("64"));
    }

    #[test]
    fn parse_verifier_stats_stack_depth_many_subprograms() {
        let log = "stack depth 32+16+8+0+0\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.stack_depth.as_deref(), Some("32+16+8+0+0"));
    }

    #[test]
    fn parse_verifier_stats_multiple_processed_lines_takes_last() {
        // Kernel only emits one, but test that rev() scan takes the first
        // match from the end.
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\nprocessed 200 insns (limit 1000000) max_states_per_insn 2 total_states 10 peak_states 4 mark_read 0\n";
        let vs = super::parse_verifier_stats(log);
        // rev() scan hits the second line first.
        assert_eq!(vs.processed_insns, 200);
        assert_eq!(vs.total_states, 10);
    }

    #[test]
    fn parse_verifier_stats_complexity_error_with_stats() {
        // Verifier rejects a program but still emits the stats line.
        let log = "\
func#0 @0
0: R1=ctx() R10=fp0
1: (bf) r6 = r1                       ; R1=ctx() R6_w=ctx()
back-edge from insn 42 to 10
BPF program is too complex
processed 131071 insns (limit 131072) max_states_per_insn 12 total_states 9999 peak_states 5000 mark_read 800
verification time 250000 usec
stack depth 96+32
";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 131071);
        assert_eq!(vs.total_states, 9999);
        assert_eq!(vs.peak_states, 5000);
        assert_eq!(vs.time_usec, Some(250000));
        assert_eq!(vs.stack_depth.as_deref(), Some("96+32"));
    }

    #[test]
    fn parse_verifier_stats_complexity_error_no_stats() {
        // Verifier rejects before emitting stats (e.g. immediate type error).
        let log = "\
func#0 @0
0: R1=ctx() R10=fp0
R1 type=ctx expected=fp
";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
        assert_eq!(vs.total_states, 0);
        assert!(vs.time_usec.is_none());
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_loop_warning_with_stats() {
        // Loop detection warnings interspersed with normal output.
        let log = "\
infinite loop detected at insn 15
back-edge from insn 30 to 15
processed 500 insns (limit 1000000) max_states_per_insn 3 total_states 40 peak_states 15 mark_read 5
verification time 100 usec
";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 500);
        assert_eq!(vs.total_states, 40);
        assert_eq!(vs.peak_states, 15);
        assert_eq!(vs.time_usec, Some(100));
    }

    // -- adversary edge cases: malformed/truncated stats lines --

    #[test]
    fn parse_verifier_stats_processed_no_number() {
        // Just "processed" with nothing after it.
        let log = "processed\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 0);
    }

    #[test]
    fn parse_verifier_stats_keyword_at_end_no_value() {
        // "total_states" is the last word with no value after it.
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 100);
        assert_eq!(vs.total_states, 0);
    }

    #[test]
    fn parse_verifier_stats_non_numeric_values() {
        // Numeric processed count but non-numeric state values.
        let log = "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states abc peak_states xyz mark_read 0\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 100);
        assert_eq!(vs.total_states, 0);
        assert_eq!(vs.peak_states, 0);
    }

    #[test]
    fn parse_verifier_stats_verification_time_no_number() {
        // "unknown" is not a valid u64.
        let log = "verification time unknown usec\n";
        let vs = super::parse_verifier_stats(log);
        assert!(vs.time_usec.is_none());
    }

    #[test]
    fn parse_verifier_stats_stack_depth_empty() {
        // "stack depth" followed by only whitespace.
        let log = "stack depth   \n";
        let vs = super::parse_verifier_stats(log);
        assert!(vs.stack_depth.is_none());
    }

    #[test]
    fn parse_verifier_stats_peak_states_at_end() {
        // "peak_states" is the last keyword with no value.
        let log = "processed 50 insns (limit 1000000) max_states_per_insn 1 total_states 10 peak_states\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 50);
        assert_eq!(vs.total_states, 10);
        assert_eq!(vs.peak_states, 0);
    }

    // -----------------------------------------------------------------------
    // format_brief_line
    // -----------------------------------------------------------------------

    #[test]
    fn format_brief_line_full_stats() {
        let vs = super::VerifierStats {
            processed_insns: 1234,
            total_states: 200,
            peak_states: 50,
            time_usec: Some(42),
            stack_depth: Some("32+0".into()),
        };
        let line = super::format_brief_line("dispatch", 100, &vs);
        assert!(line.contains("insns=100"), "insns: {line}");
        assert!(line.contains("processed=1234"), "processed: {line}");
        assert!(line.contains("states=50/200"), "states: {line}");
        assert!(line.contains("time=42us"), "time: {line}");
        assert!(line.contains("stack=32+0"), "stack: {line}");
    }

    #[test]
    fn format_brief_line_insns_only() {
        let vs = super::VerifierStats {
            processed_insns: 500,
            total_states: 0,
            peak_states: 0,
            time_usec: None,
            stack_depth: None,
        };
        let line = super::format_brief_line("init", 20, &vs);
        assert!(line.contains("insns=20"), "insns: {line}");
        assert!(line.contains("processed=500"), "processed: {line}");
        assert!(!line.contains("states="), "no states: {line}");
        assert!(!line.contains("time="), "no time: {line}");
        assert!(!line.contains("stack="), "no stack: {line}");
    }

    #[test]
    fn format_brief_line_zero_processed() {
        let vs = super::VerifierStats {
            processed_insns: 0,
            total_states: 0,
            peak_states: 0,
            time_usec: None,
            stack_depth: None,
        };
        let line = super::format_brief_line("broken", 0, &vs);
        assert!(line.contains("insns=0"), "insns: {line}");
        assert!(line.contains("processed=0"), "processed: {line}");
    }

    #[test]
    fn format_brief_line_states_without_time() {
        let vs = super::VerifierStats {
            processed_insns: 100,
            total_states: 10,
            peak_states: 5,
            time_usec: None,
            stack_depth: None,
        };
        let line = super::format_brief_line("prog", 50, &vs);
        assert!(line.contains("states=5/10"), "states: {line}");
        assert!(!line.contains("time="), "no time: {line}");
    }

    #[test]
    fn format_brief_line_long_name_alignment() {
        let vs = super::VerifierStats {
            processed_insns: 42,
            total_states: 0,
            peak_states: 0,
            time_usec: None,
            stack_depth: None,
        };
        let short = super::format_brief_line("x", 1, &vs);
        let long = super::format_brief_line("a_very_long_program_name_here", 1, &vs);
        // Both should contain the same data.
        assert!(short.contains("processed=42"));
        assert!(long.contains("processed=42"));
    }

    // -----------------------------------------------------------------------
    // normalize_verifier_line
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_plain_instruction() {
        assert_eq!(
            super::normalize_verifier_line("100: (07) r1 += 8"),
            "100: (07) r1 += 8"
        );
    }

    #[test]
    fn normalize_strips_frame_annotation() {
        assert_eq!(
            super::normalize_verifier_line("3006: (07) r9 += 1  ; frame1: R9_w=2"),
            "3006: (07) r9 += 1"
        );
    }

    #[test]
    fn normalize_strips_register_annotation() {
        assert_eq!(
            super::normalize_verifier_line("42: (bf) r6 = r1 ; R1=ctx() R6_w=ctx()"),
            "42: (bf) r6 = r1"
        );
    }

    #[test]
    fn normalize_standalone_register_dump() {
        assert_eq!(
            super::normalize_verifier_line("3041: frame1: R0_w=scalar()"),
            "3041:"
        );
    }

    #[test]
    fn normalize_goto_inline_state() {
        assert_eq!(
            super::normalize_verifier_line(
                "3026: (b5) if r6 <= 0x11dc0 goto pc+2 3029: frame1: R0=1 R6=scalar()"
            ),
            "3026: (b5) if r6 <= 0x11dc0 goto pc+2"
        );
    }

    #[test]
    fn normalize_goto_no_inline_state() {
        assert_eq!(
            super::normalize_verifier_line("50: (05) goto pc+10"),
            "50: (05) goto pc+10"
        );
    }

    #[test]
    fn normalize_non_instruction_line() {
        assert_eq!(super::normalize_verifier_line("func#0 @0"), "func#0 @0");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(super::normalize_verifier_line(""), "");
    }

    // -----------------------------------------------------------------------
    // detect_cycle / collapse_cycles
    // -----------------------------------------------------------------------

    /// Build a log with a repeating block.
    ///
    /// Mimics real verifier output: the loop body revisits the same
    /// BPF instruction numbers each iteration, with different register
    /// state annotations that normalization strips.
    fn repeating_log(prefix: usize, period: usize, reps: usize, suffix: usize) -> String {
        let mut lines = Vec::new();
        for i in 0..prefix {
            lines.push(format!("{}: (07) r1 += {i}", 1000 + i));
        }
        // Each iteration visits the same instruction range (100..100+period)
        // with varying register annotations.
        for rep in 0..reps {
            for j in 0..period {
                let insn = 100 + j;
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w={}",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 100 + j
                ));
            }
        }
        for i in 0..suffix {
            lines.push(format!("{}: (95) exit_{i}", 2000 + i));
        }
        lines.join("\n")
    }

    #[test]
    fn detect_cycle_basic() {
        let log = repeating_log(0, 10, 8, 0);
        let lines: Vec<&str> = log.lines().collect();
        let result = super::detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle");
        let (start, period, count) = result.unwrap();
        assert_eq!(period, 10);
        assert!(count >= 6, "count={count}");
        assert_eq!(start, 0);
    }

    #[test]
    fn detect_cycle_with_prefix_suffix() {
        let log = repeating_log(5, 10, 8, 5);
        let lines: Vec<&str> = log.lines().collect();
        let result = super::detect_cycle(&lines);
        assert!(result.is_some(), "should detect cycle with prefix/suffix");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 10);
        assert!(count >= 6);
    }

    #[test]
    fn detect_cycle_too_few_reps() {
        // Only 3 reps, MIN_REPS is 6.
        let log = repeating_log(0, 10, 3, 0);
        let lines: Vec<&str> = log.lines().collect();
        assert!(super::detect_cycle(&lines).is_none());
    }

    #[test]
    fn detect_cycle_too_few_lines() {
        // Too few total lines for any cycle to meet MIN_PERIOD * MIN_REPS.
        let lines: Vec<String> = (0..20)
            .map(|i| format!("{}: (07) r1 += {i}", 100 + i % 3))
            .collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        assert!(super::detect_cycle(&refs).is_none());
    }

    #[test]
    fn detect_cycle_no_cycle() {
        let lines: Vec<String> = (0..100).map(|i| format!("{i}: unique_insn_{i}")).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        assert!(super::detect_cycle(&refs).is_none());
    }

    #[test]
    fn detect_cycle_empty() {
        let empty: Vec<&str> = vec![];
        assert!(super::detect_cycle(&empty).is_none());
    }

    #[test]
    fn normalize_goto_negative_offset() {
        assert_eq!(
            super::normalize_verifier_line("50: (05) goto pc-10 60: frame1: R0=1"),
            "50: (05) goto pc-10"
        );
    }

    #[test]
    fn normalize_semicolon_source_comment() {
        // Source comments are cycle anchors — must NOT be stripped.
        let line = "100: (07) r1 += 8 ; for (int j = 0; j < n; j++)";
        assert_eq!(super::normalize_verifier_line(line), line);
    }

    #[test]
    fn normalize_semicolon_return_value_comment() {
        // "; Return value" — R is followed by 'e', not a digit.
        let line = "200: (b7) r0 = 0 ; Return value";
        assert_eq!(super::normalize_verifier_line(line), line);
    }

    #[test]
    fn normalize_standalone_bare_register_dump() {
        // Pattern 7: "NNNN: R0=..." without frame prefix.
        assert_eq!(
            super::normalize_verifier_line("3029: R0=1 R6=scalar(id=1)"),
            "3029:"
        );
    }

    #[test]
    fn normalize_standalone_r10_dump() {
        assert_eq!(super::normalize_verifier_line("42: R10=fp0"), "42:");
    }

    #[test]
    fn detect_cycle_exact_boundary() {
        // Exactly MIN_PERIOD(5) * MIN_REPS(6) = 30 lines, all identical after normalization.
        let log = repeating_log(0, 5, 6, 0);
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 30);
        let result = super::detect_cycle(&lines);
        assert!(result.is_some(), "boundary case should detect cycle");
        let (_start, period, count) = result.unwrap();
        assert_eq!(period, 5);
        assert_eq!(count, 6);
    }

    #[test]
    fn collapse_cycles_empty_string() {
        assert_eq!(super::collapse_cycles(""), "");
    }

    #[test]
    fn parse_verifier_stats_windows_line_endings() {
        let log = "processed 42 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\r\nverification time 10 usec\r\nstack depth 16\r\n";
        let vs = super::parse_verifier_stats(log);
        assert_eq!(vs.processed_insns, 42);
        assert_eq!(vs.time_usec, Some(10));
        // Stack depth may contain trailing \r from the line.
        assert!(vs.stack_depth.is_some());
    }

    #[test]
    fn collapse_cycles_basic() {
        let log = repeating_log(2, 10, 8, 2);
        let collapsed = super::collapse_cycles(&log);
        assert!(
            collapsed.contains("identical iterations omitted"),
            "should contain omission marker: {collapsed}"
        );
        assert!(
            collapsed.contains("8x of the following 10 lines"),
            "should state count and period: {collapsed}"
        );
        assert!(
            collapsed.contains("end repeat"),
            "should contain end marker: {collapsed}"
        );
        // Collapsed output should be shorter.
        assert!(
            collapsed.lines().count() < log.lines().count(),
            "collapsed ({}) should be shorter than original ({})",
            collapsed.lines().count(),
            log.lines().count()
        );
    }

    #[test]
    fn collapse_cycles_no_cycle() {
        let log = "line 1\nline 2\nline 3\n";
        let collapsed = super::collapse_cycles(log);
        assert_eq!(collapsed, log);
    }

    #[test]
    fn collapse_cycles_preserves_stats() {
        // Stats at the end should survive collapse.
        let mut log = repeating_log(0, 10, 8, 0);
        log.push_str("\nprocessed 1000 insns (limit 1000000) max_states_per_insn 5 total_states 100 peak_states 30 mark_read 10\n");
        let collapsed = super::collapse_cycles(&log);
        assert!(
            collapsed.contains("processed 1000 insns"),
            "stats must survive: {collapsed}"
        );
    }

    #[test]
    fn collapse_cycles_with_register_annotations() {
        // Lines differ only in register state — normalization makes them equal.
        // Same insn numbers each iteration (realistic verifier output).
        let mut lines = Vec::new();
        lines.push("0: (07) r1 += 1".to_string());
        for rep in 0..8 {
            for j in 0..6 {
                let insn = 100 + j; // same insn each iteration
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w={}",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 100 + j
                ));
            }
        }
        lines.push("200: (95) exit".to_string());
        let log = lines.join("\n");
        let collapsed = super::collapse_cycles(&log);
        assert!(
            collapsed.contains("identical iterations omitted"),
            "should collapse despite different register state: {collapsed}"
        );
    }

    // -----------------------------------------------------------------------
    // parse_vm_verifier_output
    // -----------------------------------------------------------------------

    #[test]
    fn parse_vm_verifier_output_basic() {
        let output = "\
some boot noise
STT_VERIFIER_PROG stt_init insn_cnt=42
STT_VERIFIER_LOG stt_init processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0
STT_VERIFIER_LOG stt_init verification time 10 usec
STT_VERIFIER_PROG stt_dispatch insn_cnt=200
STT_VERIFIER_LOG stt_dispatch processed 500 insns (limit 1000000) max_states_per_insn 3 total_states 50 peak_states 20 mark_read 5
STT_VERIFIER_DONE
more noise
";
        let stats = super::parse_vm_verifier_output(output);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].name, "stt_init");
        assert_eq!(stats[0].insn_cnt, 42);
        assert!(stats[0].log.contains("processed 100 insns"));
        assert_eq!(stats[1].name, "stt_dispatch");
        assert_eq!(stats[1].insn_cnt, 200);
        assert!(stats[1].log.contains("processed 500 insns"));
    }

    #[test]
    fn parse_vm_verifier_output_empty() {
        let stats = super::parse_vm_verifier_output("no markers here\n");
        assert!(stats.is_empty());
    }

    #[test]
    fn parse_vm_verifier_output_no_done_marker() {
        let output = "\
STT_VERIFIER_PROG stt_init insn_cnt=10
STT_VERIFIER_LOG stt_init processed 50 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0
";
        let stats = super::parse_vm_verifier_output(output);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "stt_init");
    }

    #[test]
    fn parse_vm_verifier_output_fail_program() {
        let output = "\
STT_VERIFIER_PROG broken_prog insn_cnt=0
STT_VERIFIER_LOG broken_prog FAIL: verification failed
STT_VERIFIER_DONE
";
        let stats = super::parse_vm_verifier_output(output);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].name, "broken_prog");
        assert!(stats[0].log.contains("FAIL"));
    }

    // -----------------------------------------------------------------------
    // build_b_map / build_diff_rows
    // -----------------------------------------------------------------------

    fn prog(name: &str, insn_cnt: usize, log: &str) -> super::ProgStats {
        super::ProgStats {
            name: name.to_string(),
            insn_cnt,
            log: log.to_string(),
        }
    }

    #[test]
    fn build_b_map_basic() {
        let stats_b = vec![prog(
            "dispatch",
            100,
            "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\n",
        )];
        let map = super::build_b_map(&stats_b);
        assert_eq!(map.get("dispatch"), Some(&500));
    }

    #[test]
    fn build_b_map_empty() {
        let map = super::build_b_map(&[]);
        assert!(map.is_empty());
    }

    #[test]
    fn build_diff_rows_matching_programs() {
        let stats_a = vec![prog(
            "dispatch",
            100,
            "processed 500 insns (limit 1000000) max_states_per_insn 1 total_states 5 peak_states 2 mark_read 0\n",
        )];
        let mut b_map = std::collections::HashMap::new();
        b_map.insert("dispatch".to_string(), 300u64);

        let rows = super::build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "dispatch");
        assert_eq!(rows[0].a, 500);
        assert_eq!(rows[0].b, 300);
        assert_eq!(rows[0].delta, 200);
    }

    #[test]
    fn build_diff_rows_program_missing_from_b() {
        let stats_a = vec![prog(
            "new_prog",
            50,
            "processed 100 insns (limit 1000000) max_states_per_insn 1 total_states 2 peak_states 1 mark_read 0\n",
        )];
        let b_map = std::collections::HashMap::new();

        let rows = super::build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].a, 100);
        assert_eq!(rows[0].b, 0);
        assert_eq!(rows[0].delta, 100);
    }

    #[test]
    fn build_diff_rows_negative_delta() {
        let stats_a = vec![prog(
            "dispatch",
            100,
            "processed 200 insns (limit 1000000) max_states_per_insn 1 total_states 3 peak_states 1 mark_read 0\n",
        )];
        let mut b_map = std::collections::HashMap::new();
        b_map.insert("dispatch".to_string(), 500u64);

        let rows = super::build_diff_rows(&stats_a, &b_map);
        assert_eq!(rows[0].delta, -300);
    }

    #[test]
    fn build_diff_rows_empty_a() {
        let b_map = std::collections::HashMap::new();
        let rows = super::build_diff_rows(&[], &b_map);
        assert!(rows.is_empty());
    }

    // -----------------------------------------------------------------------
    // verifier integration tests (require KVM, run with --ignored)
    // -----------------------------------------------------------------------

    /// Boot scheduler in VM, capture per-program verifier stats.
    ///   cargo test -p cargo-stt -- --ignored verifier_subcommand_brief
    #[test]
    #[ignore]
    fn verifier_subcommand_brief() {
        let result = super::cmd_verifier(super::VerifierArgs {
            package: "stt-sched".into(),
            diff: None,
            verbose: false,
            kernel: None,
        });
        assert!(result.is_ok(), "verifier failed: {}", result.unwrap_err());
    }

    /// Boot scheduler in VM with verbose output (full dmesg).
    ///   cargo test -p cargo-stt -- --ignored --nocapture verifier_subcommand_verbose
    #[test]
    #[ignore]
    fn verifier_subcommand_verbose() {
        let result = super::cmd_verifier(super::VerifierArgs {
            package: "stt-sched".into(),
            diff: None,
            verbose: true,
            kernel: None,
        });
        assert!(
            result.is_ok(),
            "verbose verifier failed: {}",
            result.unwrap_err()
        );
    }

    /// A/B diff mode: compare stt-sched against itself (delta = 0).
    ///   cargo test -p cargo-stt -- --ignored --nocapture verifier_subcommand_diff
    #[test]
    #[ignore]
    fn verifier_subcommand_diff() {
        let result = super::cmd_verifier(super::VerifierArgs {
            package: "stt-sched".into(),
            diff: Some("stt-sched".into()),
            verbose: false,
            kernel: None,
        });
        assert!(
            result.is_ok(),
            "diff verifier failed: {}",
            result.unwrap_err()
        );
    }

    /// Demonstrate cycle collapse on synthetic verifier output.
    ///
    /// Generates a realistic verifier log with loop unrolling, then shows
    /// the raw vs collapsed output. Run with `--nocapture` to see the
    /// difference:
    ///   cargo test -p cargo-stt -- --ignored --nocapture verifier_cycle_collapse_demo
    #[test]
    #[ignore]
    fn verifier_cycle_collapse_demo() {
        let mut lines = Vec::new();
        lines.push("func#0 @0".to_string());
        lines.push("0: R1=ctx() R10=fp0".to_string());
        lines.push("1: (bf) r6 = r1".to_string());

        // 10 iterations of an 8-instruction loop body.
        for rep in 0..10 {
            for j in 0..8 {
                let insn = 100 + j;
                lines.push(format!(
                    "{insn}: (bf) r{} = r{} ; frame1: R{}_w=scalar(id={},umin={})",
                    j % 10,
                    (j + 1) % 10,
                    j % 10,
                    rep * 10 + j,
                    rep * 100 + j
                ));
            }
        }

        lines.push("200: (95) exit".to_string());
        lines.push("processed 500 insns (limit 1000000) max_states_per_insn 5 total_states 100 peak_states 30 mark_read 10".to_string());
        lines.push("verification time 42 usec".to_string());
        lines.push("stack depth 32+0".to_string());
        let raw_log = lines.join("\n");

        let collapsed = super::collapse_cycles(&raw_log);

        let raw_lines = raw_log.lines().count();
        let collapsed_lines = collapsed.lines().count();

        println!("\n=== CYCLE COLLAPSE DEMO ===\n");
        println!("Raw verifier log: {} lines", raw_lines);
        println!("Collapsed output: {} lines", collapsed_lines);
        println!(
            "Compression: {:.0}% reduction\n",
            (1.0 - collapsed_lines as f64 / raw_lines as f64) * 100.0
        );
        println!("--- COLLAPSED OUTPUT ---");
        print!("{collapsed}");
        println!("--- END ---\n");

        assert!(
            collapsed_lines < raw_lines,
            "collapsed ({collapsed_lines}) should be shorter than raw ({raw_lines})"
        );
        assert!(collapsed.contains("identical iterations omitted"));
        assert!(collapsed.contains("end repeat"));
        assert!(collapsed.contains("processed 500 insns"));
    }
}
