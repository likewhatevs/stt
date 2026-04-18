use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use ktstr::cgroup::CgroupManager;
use ktstr::cli;
use ktstr::runner::Runner;
use ktstr::scenario;
use ktstr::topology::TestTopology;

#[derive(Parser)]
#[command(
    name = "ktstr",
    about = "Run ktstr scheduler test scenarios on the host",
    after_help = "See also: `cargo ktstr` for cargo-integrated workflows \
                  (test, coverage, verifier, test-stats)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run test scenarios on the host under whatever scheduler is already active.
    Run {
        /// Scenario duration in seconds.
        #[arg(long, default_value = "20")]
        duration: u64,

        /// Workers per cgroup.
        #[arg(long, default_value = "4")]
        workers: usize,

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

        /// Enable repro mode (attach BPF probes).
        #[arg(long)]
        repro: bool,

        /// Crash stack for auto-probe (file path or comma-separated function names).
        #[arg(long)]
        probe_stack: Option<String>,

        /// On scheduler crash, rerun the scenario in a second VM with
        /// BPF probes attached on the crash call chain. Requires a
        /// kernel with the `sched_ext_exit` tracepoint; falls back to
        /// dynamic stack discovery when no --probe-stack is supplied.
        #[arg(long)]
        auto_repro: bool,

        /// Kernel build directory (for DWARF source locations).
        #[arg(long)]
        kernel_dir: Option<String>,

        /// Override work type for all cgroups. Case-sensitive.
        /// Valid: CpuSpin, YieldHeavy, Mixed, IoSync, Bursty, PipeIo,
        /// FutexPingPong, CachePressure, CacheYield, CachePipe,
        /// FutexFanOut, ForkExit, NiceSweep, AffinityChurn, PolicyChurn,
        /// SchBench.
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
    /// Without `--parent-cgroup`, removes all cgroups matching
    /// `/sys/fs/cgroup/ktstr` and `/sys/fs/cgroup/ktstr-<pid>`
    /// (the paths `ktstr run` and the in-process test harness create).
    /// The directories themselves are removed too. `ktstr-<pid>`
    /// directories whose pid is still a running ktstr or cargo-ktstr
    /// process are skipped, so a concurrent cleanup run doesn't
    /// yank an active run's cgroup.
    Cleanup {
        /// Parent cgroup path. When set, cleans only this path and
        /// leaves the parent directory in place; when omitted, globs
        /// the default ktstr parents and rmdirs each.
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
    },
}

#[derive(Subcommand)]
enum KernelCommand {
    /// List cached kernel images.
    List {
        /// Output in JSON format for CI scripting. Each entry includes
        /// a computed `eol` boolean derived by fetching kernel.org's
        /// `releases.json`; this requires network access on the host.
        #[arg(long)]
        json: bool,
    },
    /// Download, build, and cache a kernel image.
    Build {
        /// Kernel version to download (e.g. 6.14.2, 6.15-rc3). A
        /// major.minor prefix (e.g. 6.12) resolves to the highest
        /// patch release in that series, falling back to probing
        /// cdn.kernel.org for EOL series no longer in releases.json.
        #[arg(conflicts_with_all = ["source", "git"])]
        version: Option<String>,
        /// Path to existing kernel source directory.
        #[arg(long, conflicts_with_all = ["version", "git"])]
        source: Option<PathBuf>,
        /// Git URL to clone kernel source from. Cloned shallow (depth 1)
        /// at the ref supplied via --ref.
        #[arg(long, requires = "git_ref", conflicts_with_all = ["version", "source"])]
        git: Option<String>,
        /// Git ref to checkout (branch, tag, commit). Required with --git.
        #[arg(long = "ref", requires = "git")]
        git_ref: Option<String>,
        /// Rebuild even if a cached image exists.
        #[arg(long)]
        force: bool,
        /// Run make mrproper before configuring (local source only).
        #[arg(long)]
        clean: bool,
    },
    /// Remove cached kernel images.
    Clean {
        /// Keep the N most recent cached kernels. When absent, removes
        /// all cached entries (subject to the confirmation prompt
        /// unless --force is also set).
        #[arg(long)]
        keep: Option<usize>,
        /// Skip confirmation prompt. Required in non-interactive contexts.
        #[arg(long)]
        force: bool,
    },
}

/// List cgroup directories that `ktstr cleanup` targets by default:
/// `/sys/fs/cgroup/ktstr` (test-harness parent) and any
/// `/sys/fs/cgroup/ktstr-<pid>` left behind by a `ktstr run` that
/// crashed or was SIGKILLed.
///
/// Returns only entries that exist and are directories. Silently
/// returns empty when `/sys/fs/cgroup` isn't a cgroup v2 mount.
/// Skips `ktstr-<pid>` directories whose pid still owns a live
/// ktstr (or cargo-ktstr) process, so a concurrent `cleanup` run
/// doesn't rmdir an active run's cgroup out from under it.
fn default_cleanup_parents() -> Vec<PathBuf> {
    let root = Path::new("/sys/fs/cgroup");
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let Ok(ty) = entry.file_type() else { continue };
        if !ty.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name == "ktstr" {
            out.push(entry.path());
            continue;
        }
        if let Some(pid_str) = name.strip_prefix("ktstr-")
            && !pid_str.is_empty()
            && pid_str.bytes().all(|b| b.is_ascii_digit())
        {
            if is_ktstr_pid_alive(pid_str) {
                eprintln!("ktstr: skipping {} (live process)", entry.path().display());
                continue;
            }
            out.push(entry.path());
        }
    }
    out.sort();
    out
}

/// Return true when `/proc/{pid}/comm` identifies a live ktstr or
/// cargo-ktstr process. Returns false on any read error (pid exited,
/// non-Linux host, /proc not mounted) so the caller treats the cgroup
/// as cleanable.
fn is_ktstr_pid_alive(pid: &str) -> bool {
    let comm_path = format!("/proc/{pid}/comm");
    let Ok(comm) = std::fs::read_to_string(&comm_path) else {
        return false;
    };
    let comm = comm.trim();
    comm == "ktstr" || comm == "cargo-ktstr"
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
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path, "ktstr").map_err(|e| anyhow::anyhow!("{e}"))?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path(), "ktstr")
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        // Tarball download: explicit version, prefix, or latest stable.
        let ver = match version {
            Some(v) if fetch::is_major_minor_prefix(&v) => {
                // Major.minor prefix (e.g., "6.12") — resolve latest patch.
                fetch::fetch_version_for_prefix(&v, "ktstr").map_err(|e| anyhow::anyhow!("{e}"))?
            }
            Some(v) => v,
            None => {
                fetch::fetch_latest_stable_version("ktstr").map_err(|e| anyhow::anyhow!("{e}"))?
            }
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
        let result = fetch::download_tarball(&ver, tmp_dir.path(), "ktstr")
            .map_err(|e| anyhow::anyhow!("{e}"));
        sp.clear();
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

fn run_completions(shell: clap_complete::Shell) {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, "ktstr", &mut std::io::stdout());
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
                    let status = if r.passed { "PASS" } else { "FAIL" };
                    println!("[{status}] {} ({:.1}s)", r.scenario_name, r.duration_s);
                    for d in &r.details {
                        println!("  {d}");
                    }
                }
                let passed = results.iter().filter(|r| r.passed).count();
                let total = results.len();
                println!("\n{passed}/{total} passed");
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

        Command::Cleanup { parent_cgroup } => match parent_cgroup {
            Some(path) => {
                if !Path::new(&path).exists() {
                    anyhow::bail!("cgroup path not found: {path}");
                }
                let cgroups = CgroupManager::new(&path);
                cgroups.cleanup_all()?;
                println!("cleaned up {path}");
            }
            None => {
                let parents = default_cleanup_parents();
                if parents.is_empty() {
                    println!("no leftover cgroups found");
                } else {
                    for path in parents {
                        let cgroups = CgroupManager::new(path.to_str().unwrap_or_default());
                        if let Err(e) = cgroups.cleanup_all() {
                            eprintln!("ktstr: cleanup_all failed on {}: {e}", path.display());
                            continue;
                        }
                        match std::fs::remove_dir(&path) {
                            Ok(()) => println!("cleaned up {}", path.display()),
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                                println!("cleaned up {}", path.display());
                            }
                            Err(e) => {
                                eprintln!("ktstr: failed to remove {}: {e}", path.display());
                            }
                        }
                    }
                }
            }
        },

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

        Command::Completions { shell } => {
            run_completions(shell);
        }
    }

    Ok(())
}
