use std::io::{BufRead, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_ktstr::{build_make_args, has_sched_ext, run_test_stats};
use clap::{ArgAction, CommandFactory, Parser, Subcommand};
use ktstr::cache::{CacheDir, CacheEntry, KernelMetadata};

mod fetch;
#[cfg(feature = "gha-cache")]
mod remote_cache;

/// ktstr.kconfig embedded at compile time so `cargo install` works.
const EMBEDDED_KCONFIG: &str = include_str!("../../ktstr.kconfig");

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
        /// or cache key (`6.14.2-tarball-x86_64`, see `cargo ktstr kernel list`).
        /// When absent, resolves automatically via cache then filesystem.
        #[arg(long)]
        kernel: Option<String>,
        /// Arguments passed through to cargo nextest run.
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
        /// or cache key (`6.14.2-tarball-x86_64`, see `cargo ktstr kernel list`).
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
    /// Launches a VM with busybox and drops into a shell. Files passed
    /// via -i are available at /include-files/<name> inside the guest.
    /// Dynamically-linked ELF binaries get automatic shared library
    /// resolution via ldd. Note: ldd executes the binary's ELF
    /// interpreter, so only include trusted binaries.
    Shell {
        /// Kernel identifier: path (`../linux`), version (`6.14.2`),
        /// or cache key (`6.14.2-tarball-x86_64`, see `cargo ktstr kernel list`).
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

/// Configure the kernel with sched_ext support.
///
/// Appends the kconfig fragment to .config and runs `make olddefconfig`
/// to resolve dependencies. kconfig uses last-value-wins semantics for
/// duplicate symbols (confdata.c:conf_read_simple), so appending is
/// equivalent to merge_config.sh's delete-then-append approach — both
/// produce the same resolved config after olddefconfig.
fn configure_kernel(kernel_dir: &Path, fragment: &str) -> Result<(), String> {
    eprintln!("cargo-ktstr: configuring kernel (sched_ext not found in .config)");

    let config_path = kernel_dir.join(".config");
    if !config_path.exists() {
        run_make(kernel_dir, &["defconfig"])?;
    }

    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(&config_path)
        .map_err(|e| format!("open .config: {e}"))?;
    std::io::Write::write_all(&mut config, fragment.as_bytes())
        .map_err(|e| format!("append kconfig: {e}"))?;

    run_make(kernel_dir, &["olddefconfig"])?;
    Ok(())
}

/// Run make in the kernel directory with the given args.
fn run_make(kernel_dir: &Path, args: &[&str]) -> Result<(), String> {
    let status = Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .status()
        .map_err(|e| format!("make {}: {e}", args.join(" ")))?;
    if !status.success() {
        return Err(format!("make {} failed", args.join(" ")));
    }
    Ok(())
}

/// Run make in the kernel directory for a parallel build.
fn make_kernel(kernel_dir: &Path) -> Result<(), String> {
    eprintln!("cargo-ktstr: building kernel in {}", kernel_dir.display());
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let args = build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make(kernel_dir, &arg_refs)
}

/// Configure if needed and build the kernel.
fn build_kernel(kernel_dir: &Path, clean: bool) -> Result<(), String> {
    if !kernel_dir.is_dir() {
        return Err(format!("{}: not a directory", kernel_dir.display()));
    }

    if clean {
        eprintln!("cargo-ktstr: make mrproper");
        run_make(kernel_dir, &["mrproper"])?;
    }

    if !has_sched_ext(kernel_dir) {
        configure_kernel(kernel_dir, EMBEDDED_KCONFIG)?;
    }

    make_kernel(kernel_dir)?;
    gen_compile_commands(kernel_dir)?;
    Ok(())
}

/// Generate compile_commands.json for LSP support.
///
/// Runs `make compile_commands.json` which invokes
/// `scripts/clang-tools/gen_compile_commands.py` over the build
/// artifacts. Requires a completed kernel build (depends on vmlinux.a).
fn gen_compile_commands(kernel_dir: &Path) -> Result<(), String> {
    eprintln!("cargo-ktstr: generating compile_commands.json");
    run_make(kernel_dir, &["compile_commands.json"])
}

fn run_test(kernel: Option<String>, args: Vec<String>) -> Result<(), String> {
    use ktstr::kernel_path::KernelId;

    let mut cmd = Command::new("cargo");
    cmd.args(["nextest", "run"]).args(&args);

    if let Some(ref val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let dir = PathBuf::from(&p);
                build_kernel(&dir, false)?;
                cmd.env("KTSTR_KERNEL", &dir);
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel(&id)?;
                eprintln!("cargo-ktstr: using cached kernel {}", cache_dir.display());
                cmd.env("KTSTR_KERNEL", &cache_dir);
            }
        }
    }
    // When kernel is None, the test framework discovers a kernel via
    // find_kernel() (which now includes cache lookup).

    eprintln!("cargo-ktstr: running tests");
    let err = cmd.exec();
    Err(format!("exec cargo nextest run: {err}"))
}

fn test_stats(dir: &Option<PathBuf>) -> Result<(), String> {
    let output = run_test_stats(dir.as_deref());
    if !output.is_empty() {
        print!("{output}");
    }
    Ok(())
}

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

/// Check if a cache entry has a stale kconfig hash.
fn is_stale_kconfig(entry: &CacheEntry, kconfig_hash: &str) -> bool {
    entry
        .metadata
        .as_ref()
        .and_then(|m| m.ktstr_kconfig_hash.as_deref())
        .is_some_and(|h| h != kconfig_hash)
}

fn kernel_list(json: bool) -> Result<(), String> {
    let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;
    let entries = cache.list().map_err(|e| format!("list cache: {e}"))?;
    let kconfig_hash = embedded_kconfig_hash();

    if json {
        let json_entries: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| match &e.metadata {
                Some(meta) => {
                    // source_tree_path contains local filesystem paths.
                    // Included for completeness since JSON output is local
                    // tool output, not a published artifact.
                    serde_json::json!({
                        "key": e.key,
                        "path": e.path.display().to_string(),
                        "version": meta.version,
                        "source": meta.source,
                        "arch": meta.arch,
                        "built_at": meta.built_at,
                        "ktstr_kconfig_hash": meta.ktstr_kconfig_hash,
                        "stale_kconfig": is_stale_kconfig(e, &kconfig_hash),
                        "config_hash": meta.config_hash,
                        "image_name": meta.image_name,
                        "image_path": e.path.join(&meta.image_name).display().to_string(),
                        "vmlinux_name": meta.vmlinux_name,
                        "git_hash": meta.git_hash,
                        "git_ref": meta.git_ref,
                        "source_tree_path": meta.source_tree_path,
                    })
                }
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
        let output =
            serde_json::to_string_pretty(&wrapper).map_err(|e| format!("serialize json: {e}"))?;
        println!("{output}");
        return Ok(());
    }

    eprintln!("cache: {}", cache.root().display());

    if entries.is_empty() {
        println!(
            "no cached kernels. Run `cargo ktstr kernel build` to download and build a kernel."
        );
        return Ok(());
    }

    println!(
        "  {:<36} {:<12} {:<8} {:<7} BUILT",
        "KEY", "VERSION", "SOURCE", "ARCH"
    );
    let mut has_stale = false;
    for entry in &entries {
        if is_stale_kconfig(entry, &kconfig_hash) {
            has_stale = true;
        }
        println!("{}", format_entry_row(entry, &kconfig_hash));
    }
    if has_stale {
        eprintln!(
            "warning: entries marked (stale kconfig) were built with a different ktstr.kconfig. \
             Rebuild with: cargo ktstr kernel build --force VERSION"
        );
    }
    Ok(())
}

/// Remove cached kernels with optional keep-N and confirmation prompt.
fn kernel_clean(keep: Option<usize>, force: bool) -> Result<(), String> {
    let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;
    let entries = cache.list().map_err(|e| format!("list cache: {e}"))?;

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
            return Err("confirmation requires a terminal. Use --force to skip.".to_string());
        }
        println!("the following entries will be removed:");
        for entry in &to_remove {
            println!("{}", format_entry_row(entry, &kconfig_hash));
        }
        eprint!("remove {} entries? [y/N] ", to_remove.len());
        std::io::stderr()
            .flush()
            .map_err(|e| format!("flush stderr: {e}"))?;
        let mut answer = String::new();
        std::io::stdin()
            .lock()
            .read_line(&mut answer)
            .map_err(|e| format!("read stdin: {e}"))?;
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
        return Err(format!("removed {removed} of {total} entries; {err}"));
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
    let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;

    // Temporary directory for tarball/git source extraction.
    // Lives until end of function so the source tree persists through build.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e}"))?;

    // Acquire source.
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path)?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path())?
    } else {
        // Tarball download: explicit version or latest stable.
        let ver = match version {
            Some(v) => v,
            None => fetch::fetch_latest_stable_version()?,
        };
        // Check cache before downloading.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!("{ver}-tarball-{arch}");
        if !force && let Some(entry) = cache_lookup(&cache, &cache_key) {
            if is_stale_kconfig(&entry, &embedded_kconfig_hash()) {
                eprintln!("cargo-ktstr: cached kernel has stale kconfig, rebuilding");
            } else {
                eprintln!("cargo-ktstr: cached kernel found: {}", entry.path.display());
                eprintln!("cargo-ktstr: use --force to rebuild");
                return Ok(());
            }
        }
        fetch::download_tarball(&ver, tmp_dir.path())?
    };

    // Check cache for --source and --git (tarball already checked
    // pre-download above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cache_lookup(&cache, &acquired.cache_key)
    {
        if is_stale_kconfig(&entry, &embedded_kconfig_hash()) {
            eprintln!("cargo-ktstr: cached kernel has stale kconfig, rebuilding");
        } else {
            eprintln!("cargo-ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("cargo-ktstr: use --force to rebuild");
            return Ok(());
        }
    }

    let source_dir = &acquired.source_dir;

    // Clean step (local source only).
    if clean {
        if source.is_none() {
            eprintln!(
                "cargo-ktstr: --clean is only meaningful with --source (downloaded sources start clean)"
            );
        } else {
            eprintln!("cargo-ktstr: make mrproper");
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
        gen_compile_commands(source_dir)?;
    }

    // Find the built kernel image and vmlinux.
    let image_path = ktstr::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| format!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let vmlinux_ref = if vmlinux_path.exists() {
        if let Ok(file_meta) = std::fs::metadata(&vmlinux_path) {
            let mb = file_meta.len() as f64 / (1024.0 * 1024.0);
            eprintln!("cargo-ktstr: caching vmlinux ({mb:.0} MB)");
        }
        Some(vmlinux_path.as_path())
    } else {
        eprintln!("cargo-ktstr: warning: vmlinux not found, BTF will not be cached");
        None
    };

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("cargo-ktstr: kernel built at {}", image_path.display());
        eprintln!("cargo-ktstr: skipping cache (dirty tree)");
        return Ok(());
    }

    // Compute config hash.
    let config_path = source_dir.join(".config");
    let config_hash = if config_path.exists() {
        let data = std::fs::read(&config_path).map_err(|e| format!("read .config: {e}"))?;
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

    let entry = cache
        .store(&acquired.cache_key, &image_path, vmlinux_ref, &metadata)
        .map_err(|e| format!("cache store: {e}"))?;

    // Store to remote cache when enabled.
    #[cfg(feature = "gha-cache")]
    if remote_cache::is_enabled() {
        remote_cache::remote_store(&entry);
    }

    eprintln!("cargo-ktstr: kernel cached as {}", acquired.cache_key);
    eprintln!(
        "cargo-ktstr: image: {}",
        entry.path.join(image_name).display()
    );

    Ok(())
}

/// Current time as ISO 8601 string (UTC, second precision).
fn now_iso8601() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Manual UTC formatting to avoid adding chrono as a dependency.
    // Good enough for cache metadata timestamps.
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since Unix epoch to Y-M-D (simplified Gregorian).
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from https://howardhinnant.github.io/date_algorithms.html
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

/// Look up a cache key, checking local first, then remote (if enabled).
fn cache_lookup(cache: &CacheDir, cache_key: &str) -> Option<CacheEntry> {
    // Local first.
    if let Some(entry) = cache.lookup(cache_key) {
        return Some(entry);
    }

    // Remote fallback.
    #[cfg(feature = "gha-cache")]
    if remote_cache::is_enabled() {
        return remote_cache::remote_lookup(cache, cache_key);
    }

    None
}

/// Resolve the cache entry directory from a Version or CacheKey identifier.
///
/// Returns the cache entry directory path, which contains the boot
/// image and optionally vmlinux. Callers set KTSTR_KERNEL to this directory
/// so build.rs (BTF/vmlinux.h) and nextest (boot image) both resolve
/// from the same location.
fn resolve_cached_kernel(id: &ktstr::kernel_path::KernelId) -> Result<PathBuf, String> {
    use ktstr::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;
            let (arch, _) = fetch::arch_info();
            let cache_key = format!("{ver}-tarball-{arch}");
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
            let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;
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
        KernelId::Path(_) => Err("resolve_cached_kernel called with Path variant".to_string()),
    }
}

/// Pre-flight check for /dev/kvm availability and permissions.
fn check_kvm() -> Result<(), String> {
    if !Path::new("/dev/kvm").exists() {
        return Err("/dev/kvm not found. KVM requires:\n  \
             - Linux kernel with KVM support (CONFIG_KVM)\n  \
             - Access to /dev/kvm (check permissions or add user to 'kvm' group)\n  \
             - Hardware virtualization enabled in BIOS (VT-x/AMD-V)"
            .to_string());
    }
    if let Err(e) = std::fs::File::open("/dev/kvm") {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            return Err(
                "/dev/kvm: permission denied. Add your user to the 'kvm' group:\n  \
                 sudo usermod -aG kvm $USER\n  \
                 then log out and back in."
                    .to_string(),
            );
        }
        return Err(format!("/dev/kvm: {e}"));
    }
    Ok(())
}

/// Resolve a kernel identifier to a bootable image path.
///
/// Handles all `KernelId` variants (path to file or directory,
/// version string, cache key) and the `None` case (automatic
/// resolution via cache then filesystem).
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
                let cache_dir = resolve_cached_kernel(&id)?;
                ktstr::kernel_path::find_image_in_dir(&cache_dir)
                    .ok_or_else(|| format!("no kernel image found in {}", cache_dir.display()))
            }
        }
    } else {
        ktstr::find_kernel()
            .map_err(|e| format!("kernel resolution: {e}"))?
            .ok_or_else(|| {
                "no kernel found. Provide --kernel or run \
                 `cargo ktstr kernel build` to download and cache one."
                    .to_string()
            })
    }
}

/// Search PATH for a bare executable name.
fn resolve_in_path(name: &Path) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if let Ok(meta) = std::fs::metadata(&candidate)
            && meta.is_file()
            && meta.permissions().mode() & 0o111 != 0
        {
            return Some(candidate);
        }
    }
    None
}

fn run_shell(
    kernel: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
) -> Result<(), String> {
    check_kvm()?;
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    // Parse topology "S,C,T".
    let parts: Vec<&str> = topology.split(',').collect();
    if parts.len() != 3 {
        return Err(format!(
            "invalid topology '{topology}': expected 'sockets,cores,threads' (e.g. '2,4,1')"
        ));
    }
    let sockets: u32 = parts[0]
        .parse()
        .map_err(|_| format!("invalid sockets value: '{}'", parts[0]))?;
    let cores: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid cores value: '{}'", parts[1]))?;
    let threads: u32 = parts[2]
        .parse()
        .map_err(|_| format!("invalid threads value: '{}'", parts[2]))?;
    if sockets == 0 || cores == 0 || threads == 0 {
        return Err(format!(
            "invalid topology '{topology}': sockets, cores, and threads must all be >= 1"
        ));
    }

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
            // Explicit path (contains / or starts with . or ..): must exist.
            if !path.exists() {
                return Err(format!(
                    "--include-files path not found: {}",
                    path.display()
                ));
            }
            path.clone()
        } else {
            // Bare name: search PATH.
            if path.exists() {
                path.clone()
            } else {
                resolve_in_path(path).ok_or_else(|| {
                    format!("-i {}: not found in filesystem or PATH", path.display())
                })?
            }
        };
        if resolved.is_dir() {
            return Err(format!(
                "-i {}: is a directory. --include-files does not support directories, \
                 pass individual files",
                resolved.display()
            ));
        }
        let file_name = resolved
            .file_name()
            .ok_or_else(|| format!("include file has no filename: {}", resolved.display()))?
            .to_string_lossy();
        let archive_path = format!("include-files/{file_name}");
        resolved_includes.push((archive_path, resolved));
    }

    let include_refs: Vec<(&str, &Path)> = resolved_includes
        .iter()
        .map(|(a, p)| (a.as_str(), p.as_path()))
        .collect();

    ktstr::run_shell(kernel_path, sockets, cores, threads, &include_refs)
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
        .map_err(|e| format!("run scheduler --ktstr-list-flags: {e}"))?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    serde_json::from_str(trimmed).map_err(|e| format!("parse --ktstr-list-flags output: {e}"))
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
    check_kvm()?;

    // Resolve scheduler binary.
    let sched_bin = match (scheduler, scheduler_bin) {
        (Some(package), None) => {
            ktstr::build_and_find_binary(&package).map_err(|e| format!("build scheduler: {e}"))?
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
        ktstr::build_and_find_binary("ktstr").map_err(|e| format!("build ktstr: {e}"))?;

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
            .map_err(|e| format!("collect verifier output: {e}"))?;

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
                .map_err(|e| format!("collect verifier output: {e}"))?;
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
        .map_err(|e| format!("profile {profile_name}: {e}"))?;

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
    #[cfg(not(feature = "gha-cache"))]
    if std::env::var("KTSTR_GHA_CACHE")
        .ok()
        .is_some_and(|v| v == "1")
    {
        eprintln!(
            "cargo-ktstr: warning: KTSTR_GHA_CACHE=1 but binary compiled without gha-cache feature. \
             Rebuild with: cargo install cargo-ktstr --features gha-cache"
        );
    }

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
        } => run_shell(kernel, topology, include_files),
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json } => kernel_list(json),
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
            } => kernel_build(version, source, git, git_ref, force, clean),
            KernelCommand::Clean { keep, force } => kernel_clean(keep, force),
        },
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

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
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--topology", "2,4,1"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { topology, .. } => {
                assert_eq!(topology, "2,4,1");
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
                assert_eq!(topology, "1,1,1");
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
        let parts: Vec<&str> = "2,4,1".split(',').collect();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].parse::<u32>().is_ok());
        assert!(parts[1].parse::<u32>().is_ok());
        assert!(parts[2].parse::<u32>().is_ok());
    }

    #[test]
    fn topology_invalid_one_component() {
        let parts: Vec<&str> = "abc".split(',').collect();
        assert_ne!(parts.len(), 3);
    }

    #[test]
    fn topology_invalid_non_numeric() {
        let parts: Vec<&str> = "a,b,c".split(',').collect();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].parse::<u32>().is_err());
    }

    #[test]
    fn topology_invalid_two_components() {
        let parts: Vec<&str> = "1,2".split(',').collect();
        assert_ne!(parts.len(), 3);
    }

    #[test]
    fn topology_invalid_zero_component() {
        // run_shell rejects zero values.
        let parts: Vec<&str> = "0,1,1".split(',').collect();
        assert_eq!(parts.len(), 3);
        let val: u32 = parts[0].parse().unwrap();
        assert_eq!(val, 0);
    }

    // -- days_to_ymd --

    #[test]
    fn days_to_ymd_epoch() {
        // Day 0 = 1970-01-01.
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        // 2026-04-12 = 20555 days since epoch.
        assert_eq!(days_to_ymd(20555), (2026, 4, 12));
    }

    #[test]
    fn days_to_ymd_leap_year_feb29() {
        // 2024-02-29 = 19782 days since epoch.
        assert_eq!(days_to_ymd(19782), (2024, 2, 29));
    }

    #[test]
    fn days_to_ymd_end_of_year() {
        // 2023-12-31 = 19722 days since epoch.
        assert_eq!(days_to_ymd(19722), (2023, 12, 31));
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
}
