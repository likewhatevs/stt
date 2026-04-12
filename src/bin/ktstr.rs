use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use ktstr::cache::KernelMetadata;
use ktstr::cgroup::CgroupManager;
use ktstr::cli;
use ktstr::runner::Runner;
use ktstr::scenario;
use ktstr::topology::TestTopology;

#[derive(Parser)]
#[command(
    name = "ktstr",
    about = "Run ktstr scheduler test scenarios on the host"
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

        /// Enable auto-repro on crash.
        #[arg(long)]
        auto_repro: bool,

        /// Kernel build directory (for DWARF source locations).
        #[arg(long)]
        kernel_dir: Option<String>,

        /// Override work type for all cgroups.
        /// Valid: CpuSpin, YieldHeavy, Mixed, IoSync, Bursty, PipeIo,
        /// FutexPingPong, CachePressure, CacheYield, CachePipe, FutexFanOut.
        #[arg(long)]
        work_type: Option<String>,
    },
    /// List available scenarios.
    List {
        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Show host CPU topology.
    Topo,
    /// Clean up leftover cgroups.
    Cleanup {
        /// Parent cgroup path.
        #[arg(long, default_value = "/sys/fs/cgroup/ktstr")]
        parent_cgroup: String,
    },
    /// Manage cached kernel images.
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    /// Boot an interactive shell in a KVM virtual machine.
    ///
    /// Launches a VM with busybox and drops into a shell. Files passed
    /// via -i are available at /include-files/<name> inside the guest.
    /// Dynamically-linked ELF binaries get automatic shared library
    /// resolution via ELF DT_NEEDED parsing.
    Shell {
        /// Kernel identifier: path (`../linux`), version (`6.14.2`),
        /// or cache key (`6.14.2-tarball-x86_64`, see `ktstr kernel list`).
        /// When absent, resolves automatically via cache then filesystem.
        #[arg(long)]
        kernel: Option<String>,
        /// Virtual topology as "sockets,cores,threads" (default: "1,1,1").
        #[arg(long, default_value = "1,1,1")]
        topology: String,
        /// Files to include in the guest at /include-files/<name>.
        /// Dynamically-linked ELF binaries get shared library resolution.
        #[arg(short = 'i', long = "include-files", action = ArgAction::Append)]
        include_files: Vec<PathBuf>,
        /// Guest memory in MB (minimum 128). When absent, estimated
        /// from payload and include file sizes.
        #[arg(long, value_parser = clap::value_parser!(u32).range(128..))]
        memory: Option<u32>,
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
        /// Output in JSON format for CI scripting.
        #[arg(long)]
        json: bool,
    },
    /// Download, build, and cache a kernel image.
    Build {
        /// Kernel version to download (e.g. 6.14.2, 6.15-rc3).
        #[arg(conflicts_with_all = ["source", "git"])]
        version: Option<String>,
        /// Path to existing kernel source directory.
        #[arg(long, conflicts_with_all = ["version", "git"])]
        source: Option<PathBuf>,
        /// Git URL to clone kernel source from.
        #[arg(long, requires = "git_ref", conflicts_with_all = ["version", "source"])]
        git: Option<String>,
        /// Git ref to checkout (branch, tag, commit).
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
        /// Keep the N most recent cached kernels.
        #[arg(long)]
        keep: Option<usize>,
        /// Skip confirmation prompt.
        #[arg(long)]
        force: bool,
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
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path).map_err(|e| anyhow::anyhow!("{e}"))?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path()).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        // Tarball download: explicit version or latest stable.
        let ver = match version {
            Some(v) => v,
            None => fetch::fetch_latest_stable_version().map_err(|e| anyhow::anyhow!("{e}"))?,
        };
        // Check cache before downloading.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!("{ver}-tarball-{arch}");
        if !force && let Some(entry) = cache.lookup(&cache_key) {
            if entry.has_stale_kconfig(&cli::embedded_kconfig_hash()) {
                eprintln!("ktstr: cached kernel has stale kconfig, rebuilding");
            } else {
                eprintln!("ktstr: cached kernel found: {}", entry.path.display());
                eprintln!("ktstr: use --force to rebuild");
                return Ok(());
            }
        }
        fetch::download_tarball(&ver, tmp_dir.path()).map_err(|e| anyhow::anyhow!("{e}"))?
    };

    // Check cache for --source and --git (tarball already checked above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cache.lookup(&acquired.cache_key)
    {
        if entry.has_stale_kconfig(&cli::embedded_kconfig_hash()) {
            eprintln!("ktstr: cached kernel has stale kconfig, rebuilding");
        } else {
            eprintln!("ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("ktstr: use --force to rebuild");
            return Ok(());
        }
    }

    let source_dir = &acquired.source_dir;

    // Clean step (local source only).
    if clean {
        if source.is_none() {
            eprintln!(
                "ktstr: --clean is only meaningful with --source (downloaded sources start clean)"
            );
        } else {
            eprintln!("ktstr: make mrproper");
            cli::run_make(source_dir, &["mrproper"])?;
        }
    }

    // Configure.
    if !cli::has_sched_ext(source_dir) {
        eprintln!("ktstr: configuring kernel (sched_ext not found in .config)");
        cli::configure_kernel(source_dir, cli::EMBEDDED_KCONFIG)?;
    }

    // Build.
    eprintln!("ktstr: building kernel in {}", source_dir.display());
    cli::make_kernel(source_dir)?;

    // Generate compile_commands.json for local trees (LSP support).
    if !acquired.is_temp {
        eprintln!("ktstr: generating compile_commands.json");
        cli::run_make(source_dir, &["compile_commands.json"])?;
    }

    // Find the built kernel image and vmlinux.
    let image_path = ktstr::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| anyhow::anyhow!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let vmlinux_ref = if vmlinux_path.exists() {
        if let Ok(file_meta) = std::fs::metadata(&vmlinux_path) {
            let mb = file_meta.len() as f64 / (1024.0 * 1024.0);
            eprintln!("ktstr: caching vmlinux ({mb:.0} MB)");
        }
        Some(vmlinux_path.as_path())
    } else {
        eprintln!("ktstr: warning: vmlinux not found, BTF will not be cached");
        None
    };

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("ktstr: kernel built at {}", image_path.display());
        eprintln!("ktstr: skipping cache (dirty tree)");
        return Ok(());
    }

    // Compute config hash.
    let config_path = source_dir.join(".config");
    let config_hash = if config_path.exists() {
        let data = std::fs::read(&config_path)?;
        Some(format!("{:08x}", crc32fast::hash(&data)))
    } else {
        None
    };

    let (arch, image_name) = fetch::arch_info();
    let kconfig_hash = cli::embedded_kconfig_hash();

    let metadata = KernelMetadata::new(
        acquired.source_type.clone(),
        arch.to_string(),
        image_name.to_string(),
        cli::now_iso8601(),
    )
    .with_version(acquired.version.clone())
    .with_config_hash(config_hash)
    .with_ktstr_kconfig_hash(Some(kconfig_hash))
    .with_git_hash(acquired.git_hash.clone())
    .with_git_ref(acquired.git_ref.clone())
    .with_source_tree_path(if source.is_some() {
        Some(acquired.source_dir.clone())
    } else {
        None
    });

    let entry = cache.store(&acquired.cache_key, &image_path, vmlinux_ref, &metadata)?;

    eprintln!("ktstr: kernel cached as {}", acquired.cache_key);
    eprintln!("ktstr: image: {}", entry.path.join(image_name).display());

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
        } => {
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

        Command::Cleanup { parent_cgroup } => {
            let cgroups = CgroupManager::new(&parent_cgroup);
            cgroups.cleanup_all()?;
            println!("cleaned up {parent_cgroup}");
        }

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
            memory,
        } => {
            cli::check_kvm()?;
            let kernel_path = cli::resolve_kernel_image(kernel.as_deref())?;

            // Parse topology "S,C,T".
            let parts: Vec<&str> = topology.split(',').collect();
            anyhow::ensure!(
                parts.len() == 3,
                "invalid topology '{topology}': expected 'sockets,cores,threads' (e.g. '2,4,1')"
            );
            let sockets: u32 = parts[0]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid sockets value: '{}'", parts[0]))?;
            let cores: u32 = parts[1]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid cores value: '{}'", parts[1]))?;
            let threads: u32 = parts[2]
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid threads value: '{}'", parts[2]))?;
            anyhow::ensure!(
                sockets > 0 && cores > 0 && threads > 0,
                "invalid topology '{topology}': sockets, cores, and threads must all be >= 1"
            );

            // Build include_files vec: resolve each path, construct archive pairs.
            let mut resolved_includes: Vec<(String, PathBuf)> = Vec::new();
            for path in &include_files {
                let is_explicit_path = {
                    use std::path::Component;
                    matches!(
                        path.components().next(),
                        Some(Component::RootDir | Component::CurDir | Component::ParentDir)
                    ) || path.components().count() > 1
                };
                let resolved = if is_explicit_path {
                    anyhow::ensure!(
                        path.exists(),
                        "--include-files path not found: {}",
                        path.display()
                    );
                    path.clone()
                } else {
                    // Bare name: search PATH.
                    if path.exists() {
                        path.clone()
                    } else {
                        cli::resolve_in_path(path).ok_or_else(|| {
                            anyhow::anyhow!(
                                "-i {}: not found in filesystem or PATH",
                                path.display()
                            )
                        })?
                    }
                };
                anyhow::ensure!(
                    !resolved.is_dir(),
                    "-i {}: is a directory. --include-files does not support directories, \
                     pass individual files",
                    resolved.display()
                );
                let file_name = resolved
                    .file_name()
                    .ok_or_else(|| {
                        anyhow::anyhow!("include file has no filename: {}", resolved.display())
                    })?
                    .to_string_lossy();
                let archive_path = format!("include-files/{file_name}");
                resolved_includes.push((archive_path, resolved));
            }

            let include_refs: Vec<(&str, &Path)> = resolved_includes
                .iter()
                .map(|(a, p)| (a.as_str(), p.as_path()))
                .collect();

            ktstr::run_shell(kernel_path, sockets, cores, threads, &include_refs, memory)?;
        }

        Command::Completions { shell } => {
            run_completions(shell);
        }
    }

    Ok(())
}
