use std::io::{BufRead, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_ktstr::{build_make_args, has_sched_ext, run_test_stats};
use clap::{Parser, Subcommand};
use ktstr::cache::{CacheDir, CacheEntry, KernelMetadata};

mod fetch;

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
    /// Build the kernel with sched_ext support.
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
            KernelId::Version(ver) => {
                let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;
                let (arch, _) = fetch::arch_info();
                let cache_key = format!("{ver}-tarball-{arch}");
                match cache.lookup(&cache_key) {
                    Some(entry) => {
                        let image = entry
                            .metadata
                            .as_ref()
                            .map(|m| entry.path.join(&m.image_name))
                            .ok_or_else(|| {
                                format!("cached entry {cache_key} has corrupt metadata")
                            })?;
                        eprintln!("cargo-ktstr: using cached kernel {}", image.display());
                        cmd.env("KTSTR_TEST_KERNEL", &image);
                    }
                    None => {
                        return Err(format!(
                            "kernel version {ver} not found in cache. \
                             Run `cargo ktstr kernel build {ver}` first."
                        ));
                    }
                }
            }
            KernelId::CacheKey(key) => {
                let cache = CacheDir::new().map_err(|e| format!("open cache: {e}"))?;
                match cache.lookup(&key) {
                    Some(entry) => {
                        let image = entry
                            .metadata
                            .as_ref()
                            .map(|m| entry.path.join(&m.image_name))
                            .ok_or_else(|| format!("cached entry {key} has corrupt metadata"))?;
                        eprintln!("cargo-ktstr: using cached kernel {}", image.display());
                        cmd.env("KTSTR_TEST_KERNEL", &image);
                    }
                    None => {
                        return Err(format!(
                            "cache key {key} not found. \
                             Run `cargo ktstr kernel list` to see available entries."
                        ));
                    }
                }
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
        if !force && let Some(entry) = cache.lookup(&cache_key) {
            eprintln!("cargo-ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("cargo-ktstr: use --force to rebuild");
            return Ok(());
        }
        fetch::download_tarball(&ver, tmp_dir.path())?
    };

    // Check cache for --source and --git (tarball already checked
    // pre-download above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cache.lookup(&acquired.cache_key)
    {
        eprintln!("cargo-ktstr: cached kernel found: {}", entry.path.display());
        eprintln!("cargo-ktstr: use --force to rebuild");
        return Ok(());
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

    // Find the built kernel image.
    let image_path = ktstr::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| format!("no kernel image found in {}", source_dir.display()))?;

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
        .store(&acquired.cache_key, &image_path, &metadata)
        .map_err(|e| format!("cache store: {e}"))?;

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

fn main() {
    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    let result = match ktstr.command {
        KtstrCommand::BuildKernel { kernel, clean } => build_kernel(&kernel, clean),
        KtstrCommand::Test { kernel, args } => run_test(kernel, args),
        KtstrCommand::TestStats { ref dir } => test_stats(dir),
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
}
