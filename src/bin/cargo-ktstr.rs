use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{ArgAction, CommandFactory, Parser, Subcommand};
use ktstr::cache::{CacheDir, CacheEntry};
use ktstr::cli;

use ktstr::fetch;
use ktstr::remote_cache;

#[derive(Parser)]
#[command(name = "cargo-ktstr", bin_name = "cargo")]
struct Cargo {
    #[command(subcommand)]
    command: CargoSub,
}

#[derive(Subcommand)]
enum CargoSub {
    /// ktstr dev workflow: build kernel + run tests.
    Ktstr(Ktstr),
}

#[derive(Parser)]
struct Ktstr {
    #[command(subcommand)]
    command: KtstrCommand,
}

#[derive(Subcommand)]
enum KtstrCommand {
    /// Deprecated: use `cargo ktstr kernel build` instead.
    #[command(hide = true)]
    BuildKernel {
        /// Path to the kernel source directory.
        #[arg(long)]
        kernel: PathBuf,
        /// Run `make mrproper` first for a full reconfigure + rebuild.
        #[arg(long)]
        clean: bool,
    },
    /// Build the kernel (if needed) and run tests via cargo nextest.
    Test {
        /// Kernel identifier: path (`../linux`), version (`6.14.2`),
        /// or cache key (see `cargo ktstr kernel list`).
        /// When absent, resolves automatically via cache then filesystem.
        #[arg(long)]
        kernel: Option<String>,
        /// Arguments passed through to cargo nextest run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build the kernel (if needed) and run tests with coverage via cargo llvm-cov nextest.
    Coverage {
        /// Kernel identifier: path (`../linux`), version (`6.14.2`),
        /// or cache key (see `cargo ktstr kernel list`).
        /// When absent, resolves automatically via cache then filesystem.
        #[arg(long)]
        kernel: Option<String>,
        /// Arguments passed through to cargo llvm-cov nextest.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Print gauntlet analysis from sidecar JSON files.
    TestStats {
        /// Path to the sidecar directory. Defaults to KTSTR_SIDECAR_DIR
        /// or target/ktstr/{branch}-{hash}/.
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Manage cached kernel images.
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    /// Collect BPF verifier statistics for a scheduler.
    ///
    /// Builds the scheduler (or uses a pre-built binary), boots a VM,
    /// loads the scheduler's BPF programs, and reports per-program
    /// verified instruction counts from host-side memory introspection.
    Verifier {
        /// Scheduler package name to build and analyze.
        #[arg(long)]
        scheduler: Option<String>,
        /// Path to pre-built scheduler binary (alternative to --scheduler).
        #[arg(long, conflicts_with = "scheduler")]
        scheduler_bin: Option<PathBuf>,
        /// Kernel identifier: path (`../linux`), version (`6.14.2`),
        /// or cache key (see `cargo ktstr kernel list`).
        /// When absent, resolves automatically via cache then filesystem.
        #[arg(long)]
        kernel: Option<String>,
        /// Print raw verifier output without formatting.
        #[arg(long)]
        raw: bool,
        /// Run verifier for all flag profiles. Discovers flags via
        /// `--ktstr-list-flags`, constructs profiles (power set
        /// respecting requires dependencies), and collects verifier
        /// stats per profile.
        #[arg(long)]
        all_profiles: bool,
        /// Run verifier for specific profiles only (comma-separated
        /// names, e.g. `default,llc,llc+steal`). Implies --all-profiles
        /// for flag discovery.
        #[arg(long, value_delimiter = ',')]
        profiles: Vec<String>,
    },
    /// Generate shell completions for cargo-ktstr.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
        /// Binary name for completions (default: "cargo").
        #[arg(long, default_value = "cargo")]
        binary: String,
    },
    /// Boot an interactive shell in a KVM virtual machine.
    ///
    /// Launches a VM with busybox and drops into a shell. Files and
    /// directories passed via -i are available at /include-files/<name>
    /// inside the guest. Directories are walked recursively, preserving
    /// structure. Dynamically-linked ELF binaries get automatic shared
    /// library resolution via ELF DT_NEEDED parsing.
    Shell {
        /// Kernel identifier: path (`../linux`), version (`6.14.2`),
        /// or cache key (see `cargo ktstr kernel list`).
        /// When absent, resolves automatically via cache then filesystem.
        #[arg(long)]
        kernel: Option<String>,
        /// Virtual topology as "numa_nodes,llcs,cores,threads" (default: "1,1,1,1").
        #[arg(long, default_value = "1,1,1,1")]
        topology: String,
        /// Files or directories to include in the guest at /include-files/<name>.
        /// Directories are walked recursively, preserving structure.
        /// Dynamically-linked ELF binaries get shared library resolution.
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

/// Configure if needed and build the kernel.
fn build_kernel(kernel_dir: &Path, clean: bool) -> Result<(), String> {
    if !kernel_dir.is_dir() {
        return Err(format!("{}: not a directory", kernel_dir.display()));
    }

    if clean {
        eprintln!("cargo-ktstr: make mrproper");
        cli::run_make(kernel_dir, &["mrproper"]).map_err(|e| format!("{e:#}"))?;
    }

    if !cli::has_sched_ext(kernel_dir) {
        let sp = cli::Spinner::start("Configuring kernel...");
        let result =
            cli::configure_kernel(kernel_dir, cli::EMBEDDED_KCONFIG).map_err(|e| format!("{e:#}"));
        if result.is_err() {
            sp.clear();
        } else {
            sp.finish("Kernel configured");
        }
        result?;
    }

    let sp = cli::Spinner::start("Building kernel...");
    let result = cli::make_kernel_with_output(kernel_dir, Some(&sp)).map_err(|e| format!("{e:#}"));
    if result.is_err() {
        sp.clear();
    } else {
        sp.finish("Kernel built");
    }
    result?;

    cli::validate_kernel_config(kernel_dir).map_err(|e| format!("{e:#}"))?;

    let sp = cli::Spinner::start("Generating compile_commands.json...");
    let result = cli::run_make_with_output(kernel_dir, &["compile_commands.json"], Some(&sp))
        .map_err(|e| format!("{e:#}"));
    if result.is_err() {
        sp.clear();
    } else {
        sp.finish("Done");
    }
    result?;
    Ok(())
}

fn run_test(kernel: Option<String>, args: Vec<String>) -> Result<(), String> {
    use ktstr::kernel_path::KernelId;

    let mut cmd = Command::new("cargo");
    cmd.args(["nextest", "run"]).args(&args);

    if let Some(ref val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let dir = std::fs::canonicalize(&p).unwrap_or_else(|_| PathBuf::from(&p));
                build_kernel(&dir, false)?;
                cmd.env("KTSTR_KERNEL", &dir);
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel_with_remote(&id)?;
                eprintln!("cargo-ktstr: using cached kernel {}", cache_dir.display());
                cmd.env("KTSTR_KERNEL", &cache_dir);
            }
        }
    }
    // When kernel is None, the test framework discovers a kernel via
    // resolve_test_kernel() (KTSTR_TEST_KERNEL, then find_kernel() for
    // cache and filesystem fallbacks).

    eprintln!("cargo-ktstr: running tests");
    let err = cmd.exec();
    Err(format!("exec cargo nextest run: {err}"))
}

fn run_coverage(kernel: Option<String>, args: Vec<String>) -> Result<(), String> {
    use ktstr::kernel_path::KernelId;

    let mut cmd = Command::new("cargo");
    cmd.args(["llvm-cov", "nextest"]).args(&args);

    if let Some(ref val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let dir = std::fs::canonicalize(&p).unwrap_or_else(|_| PathBuf::from(&p));
                build_kernel(&dir, false)?;
                cmd.env("KTSTR_KERNEL", &dir);
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel_with_remote(&id)?;
                eprintln!("cargo-ktstr: using cached kernel {}", cache_dir.display());
                cmd.env("KTSTR_KERNEL", &cache_dir);
            }
        }
    }

    eprintln!("cargo-ktstr: running coverage");
    let err = cmd.exec();
    Err(format!("exec cargo llvm-cov nextest: {err}"))
}

fn test_stats(dir: &Option<PathBuf>) -> Result<(), String> {
    let output = cli::run_test_stats(dir.as_deref());
    if !output.is_empty() {
        print!("{output}");
    }
    Ok(())
}

/// Acquire source, configure, build, and cache a kernel image.
fn kernel_build(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
) -> Result<(), String> {
    let cache = CacheDir::new().map_err(|e| format!("open cache: {e:#}"))?;

    // Temporary directory for tarball/git source extraction.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e:#}"))?;

    // Acquire source.
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path)?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path())?
    } else {
        // Tarball download: explicit version, prefix, or latest stable.
        let ver = match version {
            Some(v) if v.matches('.').count() < 2 && !v.contains("-rc") => {
                // Major.minor prefix (e.g., "6.12") — resolve latest patch.
                fetch::fetch_version_for_prefix(&v)?
            }
            Some(v) => v,
            None => fetch::fetch_latest_stable_version()?,
        };
        // Check cache before downloading.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!("{ver}-tarball-{arch}-kc{}", ktstr::cache_key_suffix());
        if !force && let Some(entry) = cache_lookup(&cache, &cache_key) {
            eprintln!("cargo-ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("cargo-ktstr: use --force to rebuild");
            return Ok(());
        }
        let sp = cli::Spinner::start("Downloading kernel...");
        let result = fetch::download_tarball(&ver, tmp_dir.path());
        sp.clear();
        result?
    };

    // Check cache for --source and --git (tarball already checked
    // pre-download above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cache_lookup(&cache, &acquired.cache_key)
    {
        eprintln!("cargo-ktstr: cached kernel found: {}", entry.path.display());
        eprintln!("cargo-ktstr: use --force to rebuild");
        return Ok(());
    }

    let result =
        cli::kernel_build_pipeline(&acquired, &cache, "cargo-ktstr", clean, source.is_some())
            .map_err(|e| format!("{e:#}"))?;

    // Store to remote cache when enabled.
    if let Some(ref entry) = result.entry
        && remote_cache::is_enabled()
    {
        remote_cache::remote_store(entry);
    }

    Ok(())
}

/// Look up a cache key, checking local first, then remote (if enabled).
fn cache_lookup(cache: &CacheDir, cache_key: &str) -> Option<CacheEntry> {
    if let Some(entry) = cache.lookup(cache_key) {
        return Some(entry);
    }

    if remote_cache::is_enabled() {
        return remote_cache::remote_lookup(cache, cache_key);
    }

    None
}

/// Resolve a kernel identifier to a bootable image path.
///
/// Uses `resolve_cached_kernel_with_remote` for Version/CacheKey
/// lookups so GHA remote cache is checked when enabled.
fn resolve_kernel_image(kernel: Option<&str>) -> Result<PathBuf, String> {
    use ktstr::kernel_path::KernelId;

    if let Some(val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let path = PathBuf::from(&p);
                if path.is_file() {
                    Ok(path)
                } else if path.is_dir() {
                    ktstr::kernel_path::find_image_in_dir(&path)
                        .ok_or_else(|| format!("no kernel image found in {}", path.display()))
                } else {
                    Err(format!("kernel path not found: {}", path.display()))
                }
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel_with_remote(&id)?;
                ktstr::kernel_path::find_image_in_dir(&cache_dir)
                    .ok_or_else(|| format!("no kernel image found in {}", cache_dir.display()))
            }
        }
    } else {
        ktstr::find_kernel()
            .map_err(|e| format!("{e:#}"))?
            .ok_or_else(|| {
                "no kernel found. Provide --kernel or run \
                 `cargo ktstr kernel build` to download and cache one."
                    .to_string()
            })
    }
}

/// Resolve cached kernel with remote cache fallback.
fn resolve_cached_kernel_with_remote(id: &ktstr::kernel_path::KernelId) -> Result<PathBuf, String> {
    use ktstr::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            let cache = CacheDir::new().map_err(|e| format!("open cache: {e:#}"))?;
            let (arch, _) = fetch::arch_info();
            let cache_key = format!("{ver}-tarball-{arch}-kc{}", ktstr::cache_key_suffix());
            match cache_lookup(&cache, &cache_key) {
                Some(entry) => {
                    entry
                        .metadata
                        .as_ref()
                        .ok_or_else(|| format!("cached entry {cache_key} has corrupt metadata"))?;
                    Ok(entry.path)
                }
                None => Err(format!(
                    "kernel version {ver} not found in cache. \
                     Run `cargo ktstr kernel build {ver}` first."
                )),
            }
        }
        KernelId::CacheKey(key) => {
            let cache = CacheDir::new().map_err(|e| format!("open cache: {e:#}"))?;
            match cache_lookup(&cache, key) {
                Some(entry) => {
                    entry
                        .metadata
                        .as_ref()
                        .ok_or_else(|| format!("cached entry {key} has corrupt metadata"))?;
                    Ok(entry.path)
                }
                None => Err(format!(
                    "cache key {key} not found. \
                     Run `cargo ktstr kernel list` to see available entries."
                )),
            }
        }
        KernelId::Path(_) => {
            Err("resolve_cached_kernel_with_remote called with Path variant".to_string())
        }
    }
}

fn run_shell(
    kernel: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
    memory_mb: Option<u32>,
    dmesg: bool,
    exec: Option<String>,
) -> Result<(), String> {
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    // Parse topology "N,L,C,T" (numa_nodes,llcs,cores,threads).
    let parts: Vec<&str> = topology.split(',').collect();
    if parts.len() != 4 {
        return Err(format!(
            "invalid topology '{topology}': expected 'numa_nodes,llcs,cores,threads' (e.g. '1,2,4,1')"
        ));
    }
    let numa_nodes: u32 = parts[0]
        .parse()
        .map_err(|_| format!("invalid numa_nodes value: '{}'", parts[0]))?;
    let llcs: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid llcs value: '{}'", parts[1]))?;
    let cores: u32 = parts[2]
        .parse()
        .map_err(|_| format!("invalid cores value: '{}'", parts[2]))?;
    let threads: u32 = parts[3]
        .parse()
        .map_err(|_| format!("invalid threads value: '{}'", parts[3]))?;
    if numa_nodes == 0 || llcs == 0 || cores == 0 || threads == 0 {
        return Err(format!(
            "invalid topology '{topology}': all values must be >= 1"
        ));
    }

    let resolved_includes =
        cli::resolve_include_files(&include_files).map_err(|e| format!("{e:#}"))?;

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
    )
    .map_err(|e| format!("{e:#}"))
}

/// Query a scheduler binary's flag declarations via `--ktstr-list-flags`.
///
/// Runs the binary with `--ktstr-list-flags` and parses its stdout as
/// JSON. Returns an empty vec if the binary doesn't support the flag
/// (exits non-zero or produces no output).
fn query_scheduler_flags(
    sched_bin: &Path,
) -> Result<Vec<ktstr::scenario::flags::FlagDeclJson>, String> {
    let output = Command::new(sched_bin)
        .arg("--ktstr-list-flags")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("run scheduler --ktstr-list-flags: {e:#}"))?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    serde_json::from_str(trimmed).map_err(|e| format!("parse --ktstr-list-flags output: {e:#}"))
}

/// Generate flag profiles from flag declarations.
///
/// Produces the power set of flags, filtered by requires constraints.
/// Each profile's flags are sorted in declaration order (matching the
/// library's `Scheduler::generate_profiles`).
fn generate_flag_profiles(
    flags: &[ktstr::scenario::flags::FlagDeclJson],
) -> Vec<(String, Vec<String>)> {
    let n = flags.len();
    let mut profiles = Vec::new();

    if n > 31 {
        eprintln!(
            "cargo-ktstr: error: scheduler has {n} flags, power set too large (2^{n}). \
             Use --profiles to select specific profiles."
        );
        return profiles;
    }

    for mask in 0..(1u32 << n) {
        let active: Vec<&ktstr::scenario::flags::FlagDeclJson> = (0..n)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| &flags[i])
            .collect();
        let active_names: Vec<&str> = active.iter().map(|f| f.name.as_str()).collect();

        // Check requires constraints: every active flag's requires
        // must also be in the active set.
        let valid = active.iter().all(|f| {
            f.requires
                .iter()
                .all(|r| active_names.contains(&r.as_str()))
        });
        if !valid {
            continue;
        }

        // Sort by declaration order (position in the input slice).
        let mut flag_names: Vec<String> = active.iter().map(|f| f.name.clone()).collect();
        flag_names.sort_by_key(|name| {
            flags
                .iter()
                .position(|f| f.name == *name)
                .unwrap_or(usize::MAX)
        });
        let name = if flag_names.is_empty() {
            "default".to_string()
        } else {
            flag_names.join("+")
        };
        profiles.push((name, flag_names));
    }

    profiles
}

/// Collect the extra scheduler args for a set of active flags.
fn profile_sched_args(
    active_flags: &[String],
    all_flags: &[ktstr::scenario::flags::FlagDeclJson],
) -> Vec<String> {
    let mut args = Vec::new();
    for flag_name in active_flags {
        if let Some(decl) = all_flags.iter().find(|f| f.name == *flag_name) {
            args.extend(decl.args.iter().cloned());
        }
    }
    args
}

fn run_verifier(
    scheduler: Option<String>,
    scheduler_bin: Option<PathBuf>,
    kernel: Option<String>,
    raw: bool,
    all_profiles: bool,
    profiles_filter: Vec<String>,
) -> Result<(), String> {
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;

    // Resolve scheduler binary.
    let sched_bin = match (scheduler, scheduler_bin) {
        (Some(package), None) => {
            ktstr::build_and_find_binary(&package).map_err(|e| format!("build scheduler: {e:#}"))?
        }
        (None, Some(path)) => {
            if !path.exists() {
                return Err(format!("scheduler binary not found: {}", path.display()));
            }
            path
        }
        (None, None) => {
            return Err("either --scheduler or --scheduler-bin is required".to_string());
        }
        // clap conflicts_with prevents this.
        (Some(_), Some(_)) => unreachable!(),
    };

    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    // Build the ktstr init binary.
    let ktstr_bin =
        ktstr::build_and_find_binary("ktstr").map_err(|e| format!("build ktstr: {e:#}"))?;

    if all_profiles || !profiles_filter.is_empty() {
        return run_verifier_all_profiles(
            &sched_bin,
            &ktstr_bin,
            &kernel_path,
            raw,
            &profiles_filter,
        );
    }

    eprintln!("cargo-ktstr: collecting verifier stats");
    let result =
        ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, &kernel_path, &[])
            .map_err(|e| format!("collect verifier output: {e:#}"))?;

    let output = ktstr::verifier::format_verifier_output("verifier", &result, raw);
    print!("{output}");

    Ok(())
}

fn run_verifier_all_profiles(
    sched_bin: &Path,
    ktstr_bin: &Path,
    kernel_path: &Path,
    raw: bool,
    profiles_filter: &[String],
) -> Result<(), String> {
    let flags = query_scheduler_flags(sched_bin)?;
    if flags.is_empty() {
        eprintln!(
            "cargo-ktstr: scheduler does not support --ktstr-list-flags, \
             running with default profile only"
        );
        let result =
            ktstr::verifier::collect_verifier_output(sched_bin, ktstr_bin, kernel_path, &[])
                .map_err(|e| format!("collect verifier output: {e:#}"))?;
        let output = ktstr::verifier::format_verifier_output("default", &result, raw);
        print!("{output}");
        return Ok(());
    }

    let all_profiles = generate_flag_profiles(&flags);

    // Filter profiles if --profiles was specified.
    let profiles: Vec<&(String, Vec<String>)> = if profiles_filter.is_empty() {
        all_profiles.iter().collect()
    } else {
        let filtered: Vec<_> = all_profiles
            .iter()
            .filter(|(name, _)| profiles_filter.iter().any(|f| f == name))
            .collect();
        if filtered.is_empty() {
            return Err(format!(
                "no matching profiles found. Available: {}",
                all_profiles
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        filtered
    };

    let total = profiles.len();
    if total > 32 {
        eprintln!(
            "cargo-ktstr: warning: {total} profiles to verify (>32). \
             Use --profiles to select a subset."
        );
    }

    eprintln!(
        "cargo-ktstr: verifying {total} profile{}",
        if total == 1 { "" } else { "s" }
    );

    // Per-profile summary table: (profile_name, Vec<(prog_name, verified_insns)>).
    let mut summary: Vec<(String, Vec<(String, u32)>)> = Vec::new();

    for (i, (profile_name, active_flags)) in profiles.iter().enumerate() {
        eprintln!(
            "cargo-ktstr: [{}/{}] profile: {}",
            i + 1,
            total,
            profile_name
        );

        let extra_args = profile_sched_args(active_flags, &flags);
        let result = ktstr::verifier::collect_verifier_output(
            sched_bin,
            ktstr_bin,
            kernel_path,
            &extra_args,
        )
        .map_err(|e| format!("profile {profile_name}: {e:#}"))?;

        let output = ktstr::verifier::format_verifier_output(profile_name, &result, raw);
        print!("{output}");

        let prog_stats: Vec<(String, u32)> = result
            .stats
            .iter()
            .map(|ps| (ps.name.clone(), ps.verified_insns))
            .collect();
        summary.push((profile_name.clone(), prog_stats));
    }

    // Print per-profile summary table.
    if summary.len() > 1 {
        print_profile_summary(&summary);
    }

    Ok(())
}

/// Print a summary table comparing verified_insns across profiles.
fn print_profile_summary(summary: &[(String, Vec<(String, u32)>)]) {
    // Collect all unique program names in insertion order.
    let mut prog_names: Vec<String> = Vec::new();
    for (_, progs) in summary {
        for (name, _) in progs {
            if !prog_names.contains(name) {
                prog_names.push(name.clone());
            }
        }
    }

    println!("\n--- profile summary ---");

    // Header: program name, then one column per profile.
    let profile_names: Vec<&str> = summary.iter().map(|(n, _)| n.as_str()).collect();
    print!("  {:<40}", "program");
    for pn in &profile_names {
        print!(" {:>12}", pn);
    }
    println!();
    print!("  {}", "-".repeat(40));
    for _ in &profile_names {
        print!(" {}", "-".repeat(12));
    }
    println!();

    // Rows: one per program.
    for prog in &prog_names {
        print!("  {:<40}", prog);
        for (_, progs) in summary {
            let insns = progs
                .iter()
                .find(|(n, _)| n == prog)
                .map(|(_, v)| *v)
                .unwrap_or(0);
            print!(" {:>12}", insns);
        }
        println!();
    }
}

fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cargo::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}

fn main() {
    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    let result = match ktstr.command {
        KtstrCommand::BuildKernel { kernel, clean } => {
            eprintln!(
                "cargo-ktstr: warning: build-kernel is deprecated, use `cargo ktstr kernel build --source {}` instead",
                kernel.display()
            );
            build_kernel(&kernel, clean)
        }
        KtstrCommand::Completions { shell, binary } => {
            run_completions(shell, &binary);
            Ok(())
        }
        KtstrCommand::Test { kernel, args } => run_test(kernel, args),
        KtstrCommand::Coverage { kernel, args } => run_coverage(kernel, args),
        KtstrCommand::Verifier {
            scheduler,
            scheduler_bin,
            kernel,
            raw,
            all_profiles,
            profiles,
        } => run_verifier(
            scheduler,
            scheduler_bin,
            kernel,
            raw,
            all_profiles,
            profiles,
        ),
        KtstrCommand::TestStats { ref dir } => test_stats(dir),
        KtstrCommand::Shell {
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
        } => run_shell(kernel, topology, include_files, memory_mb, dmesg, exec),
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json } => cli::kernel_list(json).map_err(|e| format!("{e:#}")),
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
            } => kernel_build(version, source, git, git_ref, force, clean),
            KernelCommand::Clean { keep, force } => {
                cli::kernel_clean(keep, force).map_err(|e| format!("{e:#}"))
            }
        },
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use ktstr::cache::KernelMetadata;

    // -- structural validation --

    #[test]
    fn cli_debug_assert() {
        Cargo::command().debug_assert();
    }

    // -- try_get_matches_from: test subcommand --

    #[test]
    fn parse_test_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "test"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_test_with_kernel() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "test", "--kernel", "6.14.2"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_test_with_passthrough_args() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "test",
            "--",
            "-p",
            "ktstr",
            "--no-capture",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Test { args, .. } => {
                assert_eq!(args, vec!["-p", "ktstr", "--no-capture"]);
            }
            _ => panic!("expected Test"),
        }
    }

    // -- try_get_matches_from: coverage subcommand --

    #[test]
    fn parse_coverage_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_coverage_with_kernel() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--kernel", "6.14.2"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_coverage_with_passthrough_args() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "coverage",
            "--",
            "--workspace",
            "--lcov",
            "--output-path",
            "lcov.info",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Coverage { args, .. } => {
                assert_eq!(
                    args,
                    vec!["--workspace", "--lcov", "--output-path", "lcov.info"]
                );
            }
            _ => panic!("expected Coverage"),
        }
    }

    // -- try_get_matches_from: shell subcommand --

    #[test]
    fn parse_shell_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "shell"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_shell_with_topology() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--topology", "1,2,4,1"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { topology, .. } => {
                assert_eq!(topology, "1,2,4,1");
            }
            _ => panic!("expected Shell"),
        }
    }

    #[test]
    fn parse_shell_default_topology() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { topology, .. } => {
                assert_eq!(topology, "1,1,1,1");
            }
            _ => panic!("expected Shell"),
        }
    }

    #[test]
    fn parse_shell_include_files() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "-i", "/tmp/a", "-i", "/tmp/b"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { include_files, .. } => {
                assert_eq!(include_files.len(), 2);
            }
            _ => panic!("expected Shell"),
        }
    }

    // -- try_get_matches_from: kernel list --

    #[test]
    fn parse_kernel_list() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_list_json() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--json"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    // -- try_get_matches_from: kernel build --

    #[test]
    fn parse_kernel_build_version() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "6.14.2"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_build_source() {
        let m =
            Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--source", "../linux"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_build_source_conflicts_with_version() {
        let m = Cargo::try_parse_from([
            "cargo", "ktstr", "kernel", "build", "--source", "../linux", "6.14.2",
        ]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_kernel_build_git_requires_ref() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "build",
            "--git",
            "https://example.com/linux.git",
        ]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_kernel_build_git_with_ref() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "build",
            "--git",
            "https://example.com/linux.git",
            "--ref",
            "v6.14",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_build_git_conflicts_with_source() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "build",
            "--git",
            "https://example.com/linux.git",
            "--ref",
            "v6.14",
            "--source",
            "../linux",
        ]);
        assert!(m.is_err());
    }

    // -- try_get_matches_from: kernel clean --

    #[test]
    fn parse_kernel_clean() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_clean_keep() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean", "--keep", "3"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Kernel {
                command: KernelCommand::Clean { keep, .. },
            } => {
                assert_eq!(keep, Some(3));
            }
            _ => panic!("expected Kernel Clean"),
        }
    }

    // -- try_get_matches_from: verifier --

    #[test]
    fn parse_verifier_with_scheduler() {
        let m =
            Cargo::try_parse_from(["cargo", "ktstr", "verifier", "--scheduler", "scx_rustland"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_verifier_with_scheduler_bin() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler-bin",
            "/tmp/sched",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_verifier_scheduler_conflicts_with_scheduler_bin() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler",
            "scx_rustland",
            "--scheduler-bin",
            "/tmp/sched",
        ]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_verifier_all_profiles() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler",
            "scx_rustland",
            "--all-profiles",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_verifier_profiles_filter() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler",
            "scx_rustland",
            "--profiles",
            "default,llc,llc+steal",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Verifier { profiles, .. } => {
                assert_eq!(profiles, vec!["default", "llc", "llc+steal"]);
            }
            _ => panic!("expected Verifier"),
        }
    }

    // -- try_get_matches_from: completions --

    #[test]
    fn parse_completions_bash() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "bash"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_completions_invalid_shell() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "noshell"]);
        assert!(m.is_err());
    }

    // -- error cases --

    #[test]
    fn parse_missing_subcommand() {
        let m = Cargo::try_parse_from(["cargo", "ktstr"]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_unknown_subcommand() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "nonexistent"]);
        assert!(m.is_err());
    }

    // -- topology parsing --

    #[test]
    fn topology_valid() {
        let parts: Vec<&str> = "1,2,4,1".split(',').collect();
        assert_eq!(parts.len(), 4);
        assert!(parts[0].parse::<u32>().is_ok());
        assert!(parts[1].parse::<u32>().is_ok());
        assert!(parts[2].parse::<u32>().is_ok());
        assert!(parts[3].parse::<u32>().is_ok());
    }

    #[test]
    fn topology_invalid_one_component() {
        let parts: Vec<&str> = "abc".split(',').collect();
        assert_ne!(parts.len(), 4);
    }

    #[test]
    fn topology_invalid_non_numeric() {
        let parts: Vec<&str> = "a,b,c,d".split(',').collect();
        assert_eq!(parts.len(), 4);
        assert!(parts[0].parse::<u32>().is_err());
    }

    #[test]
    fn topology_invalid_three_components() {
        let parts: Vec<&str> = "1,2,1".split(',').collect();
        assert_ne!(parts.len(), 4);
    }

    #[test]
    fn topology_invalid_zero_component() {
        // run_shell rejects zero values.
        let parts: Vec<&str> = "0,1,1,1".split(',').collect();
        assert_eq!(parts.len(), 4);
        let val: u32 = parts[0].parse().unwrap();
        assert_eq!(val, 0);
    }

    // -- completions --

    #[test]
    fn completions_bash_non_empty() {
        let mut buf = Vec::new();
        let mut cmd = Cargo::command();
        clap_complete::generate(clap_complete::Shell::Bash, &mut cmd, "cargo", &mut buf);
        assert!(!buf.is_empty());
    }

    #[test]
    fn completions_zsh_contains_subcommands() {
        let mut buf = Vec::new();
        let mut cmd = Cargo::command();
        clap_complete::generate(clap_complete::Shell::Zsh, &mut cmd, "cargo", &mut buf);
        let output = String::from_utf8(buf).expect("completions should be valid UTF-8");
        assert!(output.contains("test"), "zsh completions missing 'test'");
        assert!(
            output.contains("coverage"),
            "zsh completions missing 'coverage'"
        );
        assert!(output.contains("shell"), "zsh completions missing 'shell'");
        assert!(
            output.contains("kernel"),
            "zsh completions missing 'kernel'"
        );
    }

    // -- generate_flag_profiles --

    #[test]
    fn generate_flag_profiles_empty() {
        let profiles = generate_flag_profiles(&[]);
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].0, "default");
        assert!(profiles[0].1.is_empty());
    }

    #[test]
    fn generate_flag_profiles_single_flag() {
        let flags = vec![ktstr::scenario::flags::FlagDeclJson {
            name: "llc".to_string(),
            args: vec!["--llc".to_string()],
            requires: vec![],
        }];
        let profiles = generate_flag_profiles(&flags);
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].0, "default");
        assert_eq!(profiles[1].0, "llc");
    }

    #[test]
    fn generate_flag_profiles_requires_constraint() {
        let flags = vec![
            ktstr::scenario::flags::FlagDeclJson {
                name: "llc".to_string(),
                args: vec!["--llc".to_string()],
                requires: vec![],
            },
            ktstr::scenario::flags::FlagDeclJson {
                name: "steal".to_string(),
                args: vec!["--steal".to_string()],
                requires: vec!["llc".to_string()],
            },
        ];
        let profiles = generate_flag_profiles(&flags);
        // Valid: default, llc, llc+steal. Invalid: steal alone.
        let names: Vec<&str> = profiles.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(profiles.len(), 3);
        assert!(names.contains(&"default"));
        assert!(names.contains(&"llc"));
        assert!(names.contains(&"llc+steal"));
        assert!(!names.contains(&"steal"));
    }

    // -- profile_sched_args --

    #[test]
    fn profile_sched_args_collects_args() {
        let flags = vec![
            ktstr::scenario::flags::FlagDeclJson {
                name: "llc".to_string(),
                args: vec!["--llc".to_string()],
                requires: vec![],
            },
            ktstr::scenario::flags::FlagDeclJson {
                name: "steal".to_string(),
                args: vec!["--steal".to_string(), "--aggressive".to_string()],
                requires: vec![],
            },
        ];
        let active = vec!["llc".to_string(), "steal".to_string()];
        let args = profile_sched_args(&active, &flags);
        assert_eq!(args, vec!["--llc", "--steal", "--aggressive"]);
    }

    #[test]
    fn profile_sched_args_empty() {
        let flags = vec![ktstr::scenario::flags::FlagDeclJson {
            name: "llc".to_string(),
            args: vec!["--llc".to_string()],
            requires: vec![],
        }];
        let active: Vec<String> = vec![];
        let args = profile_sched_args(&active, &flags);
        assert!(args.is_empty());
    }

    // -- format_entry_row helpers --

    fn test_metadata() -> KernelMetadata {
        KernelMetadata::new(
            ktstr::cache::SourceType::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
    }

    /// Store a fake kernel image and return the CacheEntry.
    fn store_test_entry(cache: &CacheDir, key: &str, meta: &KernelMetadata) -> CacheEntry {
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join(&meta.image_name);
        std::fs::write(&image, b"fake kernel").unwrap();
        cache.store(key, &image, None, None, meta).unwrap()
    }

    /// Create a corrupt entry (directory exists but no valid metadata).
    fn store_corrupt_entry(cache: &CacheDir, key: &str) -> CacheEntry {
        let dir = cache.root().join(key);
        std::fs::create_dir_all(&dir).unwrap();
        // list() returns entries with metadata: None for corrupt dirs.
        cache
            .list()
            .unwrap()
            .into_iter()
            .find(|e| e.key == key)
            .unwrap()
    }

    // -- format_entry_row --

    #[test]
    fn format_entry_row_with_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata();
        let entry = store_test_entry(&cache, "6.14.2-tarball-x86_64", &meta);
        let row = cli::format_entry_row(&entry, "abc123");
        assert!(row.contains("6.14.2-tarball-x86_64"));
        assert!(row.contains("6.14.2"));
        assert!(row.contains("tarball"));
        assert!(row.contains("x86_64"));
        assert!(row.contains("2026-04-12T10:00:00Z"));
    }

    #[test]
    fn format_entry_row_stale_kconfig() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata().with_ktstr_kconfig_hash(Some("old_hash".to_string()));
        let entry = store_test_entry(&cache, "stale-key", &meta);
        let row = cli::format_entry_row(&entry, "new_hash");
        assert!(
            row.contains("stale kconfig"),
            "should show stale kconfig marker"
        );
    }

    #[test]
    fn format_entry_row_matching_kconfig() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata().with_ktstr_kconfig_hash(Some("same".to_string()));
        let entry = store_test_entry(&cache, "match-key", &meta);
        let row = cli::format_entry_row(&entry, "same");
        assert!(
            !row.contains("stale kconfig"),
            "should not show stale marker when hashes match"
        );
    }

    #[test]
    fn format_entry_row_no_kconfig_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata();
        let entry = store_test_entry(&cache, "no-hash-key", &meta);
        let row = cli::format_entry_row(&entry, "anything");
        assert!(
            !row.contains("stale kconfig"),
            "should not show stale marker when entry has no hash"
        );
    }

    #[test]
    fn format_entry_row_no_version() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = KernelMetadata::new(
            ktstr::cache::SourceType::Local,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        let entry = store_test_entry(&cache, "local-key", &meta);
        let row = cli::format_entry_row(&entry, "hash");
        assert!(row.contains("-"), "missing version should show dash");
    }

    #[test]
    fn format_entry_row_corrupt_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let entry = store_corrupt_entry(&cache, "corrupt-key");
        let row = cli::format_entry_row(&entry, "hash");
        assert!(row.contains("corrupt-key"));
        assert!(row.contains("corrupt metadata"));
    }

    // -- has_stale_kconfig (via CacheEntry method) --

    #[test]
    fn has_stale_kconfig_different_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata().with_ktstr_kconfig_hash(Some("old".to_string()));
        let entry = store_test_entry(&cache, "stale", &meta);
        assert!(entry.has_stale_kconfig("new"));
    }

    #[test]
    fn has_stale_kconfig_same_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata().with_ktstr_kconfig_hash(Some("same".to_string()));
        let entry = store_test_entry(&cache, "fresh", &meta);
        assert!(!entry.has_stale_kconfig("same"));
    }

    #[test]
    fn has_stale_kconfig_no_hash_in_entry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let meta = test_metadata();
        let entry = store_test_entry(&cache, "no-hash", &meta);
        assert!(!entry.has_stale_kconfig("anything"));
    }

    #[test]
    fn has_stale_kconfig_no_metadata() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache")).unwrap();
        let entry = store_corrupt_entry(&cache, "corrupt");
        assert!(!entry.has_stale_kconfig("anything"));
    }

    // -- embedded_kconfig_hash --

    #[test]
    fn embedded_kconfig_hash_deterministic() {
        let h1 = cli::embedded_kconfig_hash();
        let h2 = cli::embedded_kconfig_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn embedded_kconfig_hash_is_hex() {
        let h = cli::embedded_kconfig_hash();
        assert_eq!(h.len(), 8, "CRC32 hex should be 8 chars");
        assert!(
            h.chars().all(|c| c.is_ascii_hexdigit()),
            "should be hex digits: {h}"
        );
    }

    #[test]
    fn embedded_kconfig_hash_matches_manual_crc32() {
        let expected = format!("{:08x}", crc32fast::hash(cli::EMBEDDED_KCONFIG.as_bytes()));
        assert_eq!(cli::embedded_kconfig_hash(), expected);
    }
}
