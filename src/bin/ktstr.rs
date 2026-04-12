use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::Command as ProcessCommand;

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

use ktstr::cache::{CacheDir, CacheEntry, KernelMetadata};
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

/// ktstr.kconfig embedded at compile time.
const EMBEDDED_KCONFIG: &str = include_str!("../../ktstr.kconfig");

/// Compute CRC32 of the embedded ktstr.kconfig fragment.
fn embedded_kconfig_hash() -> String {
    let hash = crc32fast::hash(EMBEDDED_KCONFIG.as_bytes());
    format!("{hash:08x}")
}

/// Format a human-readable table row for a cache entry.
fn format_entry_row(entry: &CacheEntry, kconfig_hash: &str) -> String {
    match &entry.metadata {
        Some(meta) => {
            let version = meta.version.as_deref().unwrap_or("-");
            let source = meta.source.to_string();
            let stale = match &meta.ktstr_kconfig_hash {
                Some(h) if h != kconfig_hash => " (stale kconfig)",
                _ => "",
            };
            format!(
                "  {:<36} {:<12} {:<8} {:<7} {}{}",
                entry.key, version, source, meta.arch, meta.built_at, stale,
            )
        }
        None => {
            format!("  {:<36} (corrupt metadata)", entry.key)
        }
    }
}

fn kernel_list(json: bool) -> Result<()> {
    let cache = CacheDir::new()?;
    let entries = cache.list()?;
    let kconfig_hash = embedded_kconfig_hash();

    if json {
        let json_entries: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| match &e.metadata {
                Some(meta) => serde_json::json!({
                    "key": e.key,
                    "path": e.path.display().to_string(),
                    "version": meta.version,
                    "source": meta.source,
                    "arch": meta.arch,
                    "built_at": meta.built_at,
                    "ktstr_kconfig_hash": meta.ktstr_kconfig_hash,
                    "stale_kconfig": e.has_stale_kconfig(&kconfig_hash),
                    "config_hash": meta.config_hash,
                    "image_name": meta.image_name,
                    "image_path": e.path.join(&meta.image_name).display().to_string(),
                    "vmlinux_name": meta.vmlinux_name,
                    "git_hash": meta.git_hash,
                    "git_ref": meta.git_ref,
                    "source_tree_path": meta.source_tree_path,
                }),
                None => serde_json::json!({
                    "key": e.key,
                    "path": e.path.display().to_string(),
                    "error": "corrupt metadata",
                }),
            })
            .collect();
        let wrapper = serde_json::json!({
            "current_ktstr_kconfig_hash": kconfig_hash,
            "entries": json_entries,
        });
        println!("{}", serde_json::to_string_pretty(&wrapper)?);
        return Ok(());
    }

    eprintln!("cache: {}", cache.root().display());

    if entries.is_empty() {
        println!(
            "no cached kernels. Run `ktstr kernel build --source PATH` to build and cache a kernel."
        );
        return Ok(());
    }

    println!(
        "  {:<36} {:<12} {:<8} {:<7} BUILT",
        "KEY", "VERSION", "SOURCE", "ARCH"
    );
    let mut has_stale = false;
    for entry in &entries {
        if entry.has_stale_kconfig(&kconfig_hash) {
            has_stale = true;
        }
        println!("{}", format_entry_row(entry, &kconfig_hash));
    }
    if has_stale {
        eprintln!(
            "warning: entries marked (stale kconfig) were built with a different ktstr.kconfig. \
             Rebuild with: ktstr kernel build --force VERSION"
        );
    }
    Ok(())
}

use ktstr::cli::has_sched_ext;

/// Run make in a kernel directory.
fn run_make(kernel_dir: &std::path::Path, args: &[&str]) -> Result<()> {
    let status = ProcessCommand::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .status()?;
    anyhow::ensure!(status.success(), "make {} failed", args.join(" "));
    Ok(())
}

/// Configure the kernel with sched_ext support.
fn configure_kernel(kernel_dir: &std::path::Path, fragment: &str) -> Result<()> {
    eprintln!("ktstr: configuring kernel (sched_ext not found in .config)");

    let config_path = kernel_dir.join(".config");
    if !config_path.exists() {
        run_make(kernel_dir, &["defconfig"])?;
    }

    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(&config_path)?;
    std::io::Write::write_all(&mut config, fragment.as_bytes())?;

    run_make(kernel_dir, &["olddefconfig"])?;
    Ok(())
}

/// Build the kernel (parallel make).
fn make_kernel(kernel_dir: &std::path::Path) -> Result<()> {
    eprintln!("ktstr: building kernel in {}", kernel_dir.display());
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let args = ktstr::cli::build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make(kernel_dir, &arg_refs)
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
            if entry.has_stale_kconfig(&embedded_kconfig_hash()) {
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
        if entry.has_stale_kconfig(&embedded_kconfig_hash()) {
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
            run_make(source_dir, &["mrproper"])?;
        }
    }

    // Configure.
    if !has_sched_ext(source_dir) {
        configure_kernel(source_dir, EMBEDDED_KCONFIG)?;
    }

    // Build.
    make_kernel(source_dir)?;

    // Generate compile_commands.json for local trees (LSP support).
    if !acquired.is_temp {
        eprintln!("ktstr: generating compile_commands.json");
        run_make(source_dir, &["compile_commands.json"])?;
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
    let kconfig_hash = embedded_kconfig_hash();

    let metadata = KernelMetadata::new(
        acquired.source_type.clone(),
        arch.to_string(),
        image_name.to_string(),
        now_iso8601(),
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

/// Current time as ISO 8601 string (UTC, second precision).
fn now_iso8601() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Remove cached kernels with optional keep-N and confirmation prompt.
fn kernel_clean(keep: Option<usize>, force: bool) -> Result<()> {
    let cache = CacheDir::new()?;
    let entries = cache.list()?;

    if entries.is_empty() {
        println!("nothing to clean");
        return Ok(());
    }

    let kconfig_hash = embedded_kconfig_hash();
    let skip = keep.unwrap_or(0);
    let to_remove: Vec<&CacheEntry> = entries.iter().skip(skip).collect();

    if to_remove.is_empty() {
        println!("nothing to clean");
        return Ok(());
    }

    if !force {
        // SAFETY: isatty is always safe to call with a valid fd.
        if unsafe { libc::isatty(libc::STDIN_FILENO) } == 0 {
            anyhow::bail!("confirmation requires a terminal. Use --force to skip.");
        }
        println!("the following entries will be removed:");
        for entry in &to_remove {
            println!("{}", format_entry_row(entry, &kconfig_hash));
        }
        eprint!("remove {} entries? [y/N] ", to_remove.len());
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().lock().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y") {
            println!("aborted");
            return Ok(());
        }
    }

    let total = to_remove.len();
    let mut removed = 0usize;
    let mut last_err: Option<String> = None;
    for entry in &to_remove {
        match std::fs::remove_dir_all(&entry.path) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                removed += 1;
            }
            Err(e) => {
                last_err = Some(format!("remove {}: {e}", entry.key));
            }
        }
    }

    println!("removed {removed} cached kernel(s).");
    if let Some(err) = last_err {
        anyhow::bail!("removed {removed} of {total} entries; {err}");
    }
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
            KernelCommand::List { json } => kernel_list(json)?,
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
            } => kernel_build(version, source, git, git_ref, force, clean)?,
            KernelCommand::Clean { keep, force } => kernel_clean(keep, force)?,
        },

        Command::Completions { shell } => {
            run_completions(shell);
        }
    }

    Ok(())
}
