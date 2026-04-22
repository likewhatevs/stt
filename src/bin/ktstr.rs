#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use ktstr::cgroup::CgroupManager;
use ktstr::cli;
use ktstr::cli::KernelCommand;
use ktstr::runner::Runner;
use ktstr::scenario;
use ktstr::topology::TestTopology;

#[derive(Parser)]
#[command(
    name = "ktstr",
    about = "Run ktstr scheduler test scenarios on the host",
    after_help = "See also: `cargo ktstr` for cargo-integrated workflows \
                  (test, coverage, verifier, stats)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run test scenarios on the host under whatever scheduler is already active.
    ///
    /// Requires cgroup v2 mounted at `/sys/fs/cgroup` and write
    /// permission on it (typically root, or a delegated subtree).
    /// Each invocation creates `/sys/fs/cgroup/ktstr-<pid>/` and
    /// removes it on exit; `ktstr cleanup` reaps directories left
    /// behind by crashed runs.
    ///
    /// Probe modes interact as follows:
    /// - `--repro` alone: no-op (probe attachment requires a stack).
    /// - `--repro` + `--probe-stack`: attach BPF kprobes on the
    ///   listed functions for the duration of the scenario; emit a
    ///   probe-event report at exit and force the scenario to fail
    ///   so the report is preserved in CI output.
    /// - `--auto-repro`: applies only to VM-based runs (the
    ///   `#[ktstr_test]` harness invoked via `cargo nextest run` /
    ///   `cargo ktstr test`). Has no effect here -- this command runs
    ///   on the host without spawning a VM, so there is no second
    ///   boot to perform.
    Run {
        /// Scenario duration in seconds.
        #[arg(long, default_value = "20")]
        duration: u64,

        /// Workers per cgroup.
        #[arg(long, default_value = "4")]
        workers: usize,

        // Hardcoded list of valid flags below MUST mirror
        // `scenario::flags::ALL`. The drift test
        // `run_help_flags_lists_match_flags_all` (in tests/ktstr_cli.rs)
        // fails the build if the two diverge.
        /// Active flags (comma-separated). Omit for all profiles.
        /// Valid: llc, borrow, steal, rebal, reject-pin, no-ctrl.
        #[arg(long, value_delimiter = ',')]
        flags: Option<Vec<String>>,

        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Attach BPF probes during the scenario. Has no effect
        /// without `--probe-stack`; see the command help above for
        /// the full probe-mode interaction.
        #[arg(long)]
        repro: bool,

        /// Crash stack for auto-probe (file path or comma-separated function names).
        #[arg(long)]
        probe_stack: Option<String>,

        /// VM-based test re-run on scheduler crash. Applies only when
        /// the scenario is invoked through the `#[ktstr_test]`
        /// harness (`cargo ktstr test` / `cargo nextest run`); has no
        /// effect on this `ktstr run` command, which executes on the
        /// host. Documented here for parity with the test harness's
        /// flag set.
        #[arg(long)]
        auto_repro: bool,

        /// Kernel build directory (for DWARF source locations).
        #[arg(long)]
        kernel_dir: Option<String>,

        // Hardcoded list of valid work types below MUST mirror
        // `WorkType::ALL_NAMES` minus `Sequence` (requires explicit
        // phases) and `Custom` (requires a function pointer). The
        // drift test `run_help_work_type_lists_match_all_names` (in
        // tests/ktstr_cli.rs) fails the build if they diverge.
        /// Override work type for all cgroups. Case-sensitive.
        /// Valid: CpuSpin, YieldHeavy, Mixed, IoSync, Bursty, PipeIo,
        /// FutexPingPong, CachePressure, CacheYield, CachePipe,
        /// FutexFanOut, ForkExit, NiceSweep, AffinityChurn, PolicyChurn,
        /// FanOutCompute, PageFaultChurn, MutexContention.
        #[arg(long)]
        work_type: Option<String>,

        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
    },
    /// List available scenarios.
    List {
        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output in JSON format for CI scripting.
        #[arg(long)]
        json: bool,
    },
    /// Show host CPU topology.
    Topo,
    /// Clean up leftover cgroups.
    ///
    /// Without `--parent-cgroup`, scans `/sys/fs/cgroup` for the
    /// default ktstr parents (`ktstr` and `ktstr-<pid>`, the paths
    /// `ktstr run` and the in-process test harness create) and
    /// rmdirs each. `ktstr-<pid>` directories whose pid is still a
    /// running ktstr or cargo-ktstr process are skipped, so a
    /// concurrent cleanup run doesn't yank an active run's cgroup.
    Cleanup {
        /// Parent cgroup path. When set, cleans only this path and
        /// leaves the parent directory in place; when omitted, scans
        /// `/sys/fs/cgroup` for the default ktstr parents
        /// (`ktstr/` and `ktstr-<pid>/`) and rmdirs each.
        #[arg(long)]
        parent_cgroup: Option<String>,
    },
    /// Manage cached kernel images.
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    /// Boot an interactive shell in a KVM virtual machine.
    ///
    /// Launches a VM with busybox and drops into a shell. Files and
    /// directories passed via -i are available at /include-files/<name>
    /// inside the guest. Directories are walked recursively, preserving
    /// structure. Dynamically-linked ELF binaries get automatic shared
    /// library resolution via ELF DT_NEEDED parsing.
    Shell {
        #[arg(long, help = ktstr::cli::KERNEL_HELP_NO_RAW)]
        kernel: Option<String>,
        /// Virtual topology as "numa_nodes,llcs,cores,threads".
        #[arg(long, default_value = "1,1,1,1")]
        topology: String,
        /// Files or directories to include in the guest. Repeatable.
        #[arg(short = 'i', long = "include-files", action = ArgAction::Append)]
        include_files: Vec<PathBuf>,
        /// Guest memory in MB (minimum 128). When absent, estimated
        /// from payload and include file sizes.
        #[arg(long = "memory-mb", value_parser = clap::value_parser!(u32).range(128..))]
        memory_mb: Option<u32>,
        /// Forward kernel console (COM1/dmesg) to stderr in real-time.
        /// Sets loglevel=7 for verbose kernel output.
        #[arg(long)]
        dmesg: bool,
        /// Run a command in the VM instead of an interactive shell.
        /// The VM exits after the command completes.
        #[arg(long)]
        exec: Option<String>,
    },
    /// Generate shell completions for ktstr.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
        /// Binary name to register the completion under. Override
        /// when invoking ktstr through a symlink with a different
        /// name (the shell looks up completions by argv[0]).
        #[arg(long, default_value = "ktstr")]
        binary: String,
    },
}

/// RAII guard that cleans up an auto-generated cgroup directory on drop.
struct CgroupGuard {
    path: String,
}

impl Drop for CgroupGuard {
    fn drop(&mut self) {
        let cgroups = CgroupManager::new(&self.path);
        let _ = cgroups.cleanup_all();
        let _ = std::fs::remove_dir(&self.path);
    }
}

/// Acquire source, configure, build, and cache a kernel image.
fn kernel_build(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
) -> Result<()> {
    use ktstr::cache::CacheDir;
    use ktstr::fetch;

    let cache = CacheDir::new()?;

    // Temporary directory for tarball/git source extraction.
    let tmp_dir = tempfile::TempDir::new()?;

    // Acquire source.
    let client = reqwest::blocking::Client::new();
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path, "ktstr")?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path(), "ktstr")?
    } else {
        // Tarball download: explicit version, prefix, or latest stable.
        let ver = match version {
            Some(v) if fetch::is_major_minor_prefix(&v) => {
                // Major.minor prefix (e.g., "6.12") — resolve latest patch.
                fetch::fetch_version_for_prefix(&client, &v, "ktstr")?
            }
            Some(v) => v,
            None => fetch::fetch_latest_stable_version(&client, "ktstr")?,
        };
        // Check cache before downloading.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!("{ver}-tarball-{arch}-kc{}", ktstr::cache_key_suffix());
        if !force && let Some(entry) = cli::cache_lookup(&cache, &cache_key, "ktstr") {
            eprintln!("ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("ktstr: use --force to rebuild");
            return Ok(());
        }
        let sp = cli::Spinner::start("Downloading kernel...");
        let result = fetch::download_tarball(&client, &ver, tmp_dir.path(), "ktstr");
        drop(sp);
        result?
    };

    // Check cache for --source and --git (tarball already checked above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cli::cache_lookup(&cache, &acquired.cache_key, "ktstr")
    {
        eprintln!("ktstr: cached kernel found: {}", entry.path.display());
        eprintln!("ktstr: use --force to rebuild");
        return Ok(());
    }

    cli::kernel_build_pipeline(&acquired, &cache, "ktstr", clean, source.is_some())?;

    Ok(())
}

fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Cli::parse();

    match args.command {
        Command::Run {
            duration,
            workers,
            flags: flag_arg,
            filter,
            json,
            repro,
            probe_stack,
            auto_repro,
            kernel_dir,
            work_type,
            no_perf_mode,
        } => {
            if no_perf_mode {
                unsafe { std::env::set_var("KTSTR_NO_PERF_MODE", "1") };
            }

            let parent_cgroup = format!("/sys/fs/cgroup/ktstr-{}", std::process::id());

            // Guard cleans up auto-generated cgroups on exit (pass or fail).
            let _guard = CgroupGuard {
                path: parent_cgroup.clone(),
            };

            let active_flags = cli::resolve_flags(flag_arg)?;
            let work_type_override = cli::parse_work_type(work_type.as_deref())?;

            let config = cli::build_run_config(
                parent_cgroup,
                duration,
                workers,
                active_flags,
                repro,
                probe_stack,
                auto_repro,
                kernel_dir,
                work_type_override,
            );

            let topo = TestTopology::from_system()?;
            let runner = Runner::new(config, topo)?;

            let scenarios = scenario::all_scenarios();
            let refs = cli::filter_scenarios(&scenarios, filter.as_deref())?;

            let results = runner.run_scenarios(&refs)?;

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                for r in &results {
                    let status = if r.skipped {
                        "SKIP"
                    } else if r.passed {
                        "PASS"
                    } else {
                        "FAIL"
                    };
                    println!("[{status}] {} ({:.1}s)", r.scenario_name, r.duration_s);
                    for d in &r.details {
                        println!("  {d}");
                    }
                }
                let passed = results.iter().filter(|r| r.passed && !r.skipped).count();
                let skipped = results.iter().filter(|r| r.skipped).count();
                let total = results.len();
                let failed = total - passed - skipped;
                if skipped > 0 {
                    println!("\n{passed}/{total} passed ({skipped} skipped, {failed} failed)");
                } else {
                    println!("\n{passed}/{total} passed");
                }
            }
        }

        Command::List { filter, json } => {
            let scenarios = scenario::all_scenarios();
            let filtered: Vec<&scenario::Scenario> = scenarios
                .iter()
                .filter(|s| filter.as_ref().is_none_or(|f| s.name.contains(f.as_str())))
                .collect();

            if json {
                let entries: Vec<serde_json::Value> = filtered
                    .iter()
                    .map(|s| {
                        let profiles: Vec<String> = s.profiles().iter().map(|p| p.name()).collect();
                        serde_json::json!({
                            "name": s.name,
                            "category": s.category,
                            "description": s.description,
                            "profiles": profiles,
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else {
                for s in &filtered {
                    let profiles: Vec<String> = s.profiles().iter().map(|p| p.name()).collect();
                    println!(
                        "{:<30} [{:<12}] {} (profiles: {})",
                        s.name,
                        s.category,
                        s.description,
                        profiles.join(", "),
                    );
                }
                println!("\n{} scenarios", filtered.len());
            }
        }

        Command::Topo => {
            let topo = TestTopology::from_system()?;
            println!("CPUs:       {}", topo.total_cpus());
            println!("LLCs:       {}", topo.num_llcs());
            println!("NUMA nodes: {}", topo.num_numa_nodes());
            for (i, llc) in topo.llcs().iter().enumerate() {
                println!("  LLC {} (node {}): {:?}", i, llc.numa_node(), llc.cpus(),);
            }
        }

        Command::Cleanup { parent_cgroup } => cli::cleanup(parent_cgroup)?,

        Command::Kernel { command } => match command {
            KernelCommand::List { json } => cli::kernel_list(json)?,
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
            } => kernel_build(version, source, git, git_ref, force, clean)?,
            KernelCommand::Clean { keep, force } => cli::kernel_clean(keep, force)?,
        },

        Command::Shell {
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
        } => {
            cli::check_kvm()?;
            let kernel_path = cli::resolve_kernel_image(
                kernel.as_deref(),
                &cli::KernelResolvePolicy {
                    accept_raw_image: false,
                    cli_label: "ktstr",
                },
            )?;

            // Parse topology "N,L,C,T" (numa_nodes,llcs,cores,threads).
            let parts: Vec<&str> = topology.split(',').collect();
            anyhow::ensure!(
                parts.len() == 4,
                "invalid topology '{topology}': expected 'numa_nodes,llcs,cores,threads' (e.g. '1,2,4,1')"
            );
            let numa_nodes: u32 = parts[0]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid numa_nodes value: '{}'", parts[0]))?;
            let llcs: u32 = parts[1]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid llcs value: '{}'", parts[1]))?;
            let cores: u32 = parts[2]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid cores value: '{}'", parts[2]))?;
            let threads: u32 = parts[3]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid threads value: '{}'", parts[3]))?;
            anyhow::ensure!(
                numa_nodes > 0 && llcs > 0 && cores > 0 && threads > 0,
                "invalid topology '{topology}': all values must be >= 1"
            );

            let resolved_includes = cli::resolve_include_files(&include_files)?;

            let include_refs: Vec<(&str, &Path)> = resolved_includes
                .iter()
                .map(|(a, p)| (a.as_str(), p.as_path()))
                .collect();

            ktstr::run_shell(
                kernel_path,
                numa_nodes,
                llcs,
                cores,
                threads,
                &include_refs,
                memory_mb,
                dmesg,
                exec.as_deref(),
            )?;
        }

        Command::Completions { shell, binary } => {
            run_completions(shell, &binary);
        }
    }

    Ok(())
}
