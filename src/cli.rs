//! CLI support functions shared between `ktstr` and `cargo-ktstr`.
//!
//! Validation, configuration, and kernel/KVM resolution logic used
//! by both binaries.

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::cache::{CacheDir, CacheEntry};
use crate::runner::RunConfig;
use crate::scenario::{Scenario, flags};
use crate::workload::WorkType;

/// Help text for `--kernel` in contexts that reject raw image files
/// (test, coverage, `ktstr shell`). Matches `KernelResolvePolicy {
/// accept_raw_image: false, .. }`.
pub const KERNEL_HELP_NO_RAW: &str = "Kernel identifier: a source directory \
     path (e.g. `../linux`), a version (`6.14.2`, or major.minor prefix \
     `6.14` for latest patch), or a cache key (see `kernel list`). Raw \
     image files are rejected. Source directories auto-build (can be slow \
     on a fresh tree); versions auto-download from kernel.org on cache \
     miss.";

/// Help text for `--kernel` in contexts that accept raw image files
/// (verifier, `cargo ktstr shell`). Matches `KernelResolvePolicy {
/// accept_raw_image: true, .. }`.
pub const KERNEL_HELP_RAW_OK: &str = "Kernel identifier: a source directory \
     path (e.g. `../linux`), a raw image file (`bzImage` / `Image`), a \
     version (`6.14.2`, or major.minor prefix `6.14` for latest patch), \
     or a cache key (see `kernel list`). Source directories auto-build \
     (can be slow on a fresh tree); versions auto-download from kernel.org \
     on cache miss. When absent, resolves via cache then filesystem, \
     falling back to downloading the latest stable kernel.";

/// ktstr.kconfig embedded at compile time.
pub const EMBEDDED_KCONFIG: &str = crate::EMBEDDED_KCONFIG;

/// Compute CRC32 of the embedded ktstr.kconfig fragment.
pub fn embedded_kconfig_hash() -> String {
    crate::kconfig_hash()
}

/// Extract major.minor prefix from a version string.
/// "6.12.81" → "6.12", "7.0" → "7.0", "abc" → None.
fn version_prefix(version: &str) -> Option<String> {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 2 {
        Some(format!("{}.{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// Check if a version's series is in the active releases list.
fn is_eol(version: &str, active_prefixes: &[String]) -> bool {
    let Some(prefix) = version_prefix(version) else {
        return false;
    };
    !active_prefixes.iter().any(|p| p == &prefix)
}

/// Fetch active kernel series prefixes from releases.json.
/// Returns major.minor prefixes for all stable/longterm/mainline entries.
pub fn fetch_active_prefixes() -> Vec<String> {
    let releases = match crate::fetch::fetch_releases() {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut prefixes = Vec::new();
    for (moniker, version) in &releases {
        if moniker == "linux-next" {
            continue;
        }
        if let Some(prefix) = version_prefix(version)
            && !prefixes.contains(&prefix)
        {
            prefixes.push(prefix);
        }
    }
    prefixes
}

/// Format a human-readable table row for a cache entry.
pub fn format_entry_row(
    entry: &CacheEntry,
    kconfig_hash: &str,
    active_prefixes: &[String],
) -> String {
    match &entry.metadata {
        Some(meta) => {
            let version = meta.version.as_deref().unwrap_or("-");
            let source = meta.source.to_string();
            let mut tags = String::new();
            if entry.has_stale_kconfig(kconfig_hash) {
                tags.push_str(" (stale kconfig)");
            }
            if version != "-" && is_eol(version, active_prefixes) {
                tags.push_str(" (EOL)");
            }
            format!(
                "  {:<48} {:<12} {:<8} {:<7} {}{}",
                entry.key, version, source, meta.arch, meta.built_at, tags,
            )
        }
        None => {
            format!("  {:<48} (corrupt metadata)", entry.key)
        }
    }
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

/// List cached kernel images.
pub fn kernel_list(json: bool) -> Result<()> {
    let cache = CacheDir::new()?;
    let entries = cache.list()?;
    let kconfig_hash = embedded_kconfig_hash();

    let active_prefixes = fetch_active_prefixes();

    if json {
        let json_entries: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| match &e.metadata {
                Some(meta) => {
                    let v = meta.version.as_deref().unwrap_or("-");
                    let eol = v != "-" && is_eol(v, &active_prefixes);
                    serde_json::json!({
                        "key": e.key,
                        "path": e.path.display().to_string(),
                        "version": meta.version,
                        "source": meta.source,
                        "arch": meta.arch,
                        "built_at": meta.built_at,
                        "ktstr_kconfig_hash": meta.ktstr_kconfig_hash,
                        "stale_kconfig": e.has_stale_kconfig(&kconfig_hash),
                        "eol": eol,
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
        println!("{}", serde_json::to_string_pretty(&wrapper)?);
        return Ok(());
    }

    eprintln!("cache: {}", cache.root().display());

    if entries.is_empty() {
        println!("no cached kernels. Run `kernel build` to download and build a kernel.");
        return Ok(());
    }

    println!(
        "  {:<48} {:<12} {:<8} {:<7} BUILT",
        "KEY", "VERSION", "SOURCE", "ARCH"
    );
    let mut has_stale_kconfig = false;
    for entry in &entries {
        if entry.has_stale_kconfig(&kconfig_hash) {
            has_stale_kconfig = true;
        }
        println!(
            "{}",
            format_entry_row(entry, &kconfig_hash, &active_prefixes)
        );
    }
    if has_stale_kconfig {
        eprintln!(
            "warning: entries marked (stale kconfig) were built with a different ktstr.kconfig. \
             Rebuild with: kernel build --force VERSION"
        );
    }
    Ok(())
}

/// Remove cached kernels with optional keep-N and confirmation prompt.
pub fn kernel_clean(keep: Option<usize>, force: bool) -> Result<()> {
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
            bail!("confirmation requires a terminal. Use --force to skip.");
        }
        println!("the following entries will be removed:");
        for entry in &to_remove {
            println!("{}", format_entry_row(entry, &kconfig_hash, &[]));
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
        bail!("removed {removed} of {total} entries; {err}");
    }
    Ok(())
}

/// Run make in a kernel directory.
pub fn run_make(kernel_dir: &Path, args: &[&str]) -> Result<()> {
    let status = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .status()?;
    anyhow::ensure!(status.success(), "make {} failed", args.join(" "));
    Ok(())
}

/// Ensure the kconfig fragment is applied to the kernel's .config.
///
/// Creates a default .config via `make defconfig` if none exists.
/// Checks each non-comment CONFIG line from the fragment against
/// the current .config. If all are present, .config is not touched
/// (preserving mtime for make's dependency tracking). If any are
/// missing, appends the full fragment and runs `make olddefconfig`
/// to resolve new options with defaults — without this, the
/// subsequent `make` launches interactive `conf` prompts that hang
/// when stdout/stderr are piped.
pub fn configure_kernel(kernel_dir: &Path, fragment: &str) -> Result<()> {
    let config_path = kernel_dir.join(".config");
    if !config_path.exists() {
        run_make(kernel_dir, &["defconfig"])?;
    }

    let config_content = std::fs::read_to_string(&config_path)?;
    let all_present = fragment
        .lines()
        .filter(|l| {
            let trimmed = l.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .all(|l| config_content.contains(l.trim()));

    if all_present {
        return Ok(());
    }

    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(&config_path)?;
    std::io::Write::write_all(&mut config, fragment.as_bytes())?;

    run_make(kernel_dir, &["olddefconfig"])?;

    Ok(())
}

/// Run make with merged stdout+stderr piped through a spinner.
/// Uses `sh -c "make ... 2>&1"` for a single pipe — one reader,
/// no threads, no channel, no pipe-buffer deadlock.
///
/// When a spinner is active, each line is printed via `println()`
/// so the spinner redraws below the output. When no spinner,
/// output is captured and shown only on failure.
pub fn run_make_with_output(
    kernel_dir: &Path,
    args: &[&str],
    spinner: Option<&Spinner>,
) -> Result<()> {
    let make_cmd = format!("make {} 2>&1", args.join(" "));
    let mut child = std::process::Command::new("sh")
        .args(["-c", &make_cmd])
        .current_dir(kernel_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    let stdout = child.stdout.take().expect("piped stdout");
    let mut captured = Vec::new();
    for line in std::io::BufReader::new(stdout)
        .lines()
        .map_while(Result::ok)
    {
        if let Some(sp) = spinner {
            sp.println(&line);
        }
        captured.push(line);
    }

    let status = child.wait()?;
    if !status.success() {
        // Always show captured output on failure so CI logs contain
        // the actual compiler errors, not just "make failed".
        for line in &captured {
            eprintln!("{line}");
        }
        bail!("make {} failed", args.join(" "));
    }
    Ok(())
}

/// Build the kernel with output piped through a spinner.
pub fn make_kernel_with_output(kernel_dir: &Path, spinner: Option<&Spinner>) -> Result<()> {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let args = build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make_with_output(kernel_dir, &arg_refs, spinner)
}

/// Resolve flag names, erroring on unknown flags.
pub fn resolve_flags(flag_arg: Option<Vec<String>>) -> Result<Option<Vec<&'static str>>> {
    match flag_arg {
        Some(fs) => {
            let mut resolved = Vec::new();
            for f in &fs {
                match flags::from_short_name(f) {
                    Some(name) => resolved.push(name),
                    None => bail!(
                        "unknown flag: '{f}'. valid flags: {}",
                        flags::ALL.join(", "),
                    ),
                }
            }
            Ok(Some(resolved))
        }
        None => Ok(None),
    }
}

/// Parse and validate a work type name.
pub fn parse_work_type(name: Option<&str>) -> Result<Option<WorkType>> {
    match name {
        Some(name) => match WorkType::from_name(name) {
            Some(wt) => Ok(Some(wt)),
            None => bail!(
                "unknown work type: '{name}'. valid types: {}",
                WorkType::ALL_NAMES.join(", "),
            ),
        },
        None => Ok(None),
    }
}

/// Filter scenarios by name substring.
pub fn filter_scenarios<'a>(
    scenarios: &'a [Scenario],
    filter: Option<&str>,
) -> Result<Vec<&'a Scenario>> {
    let refs: Vec<&Scenario> = scenarios
        .iter()
        .filter(|s| filter.is_none_or(|f| s.name.contains(f)))
        .collect();
    if refs.is_empty() {
        bail!("no scenarios matched filter. run 'ktstr list' to see available scenarios");
    }
    Ok(refs)
}

/// Build a RunConfig from parsed CLI arguments.
#[allow(clippy::too_many_arguments)]
pub fn build_run_config(
    parent_cgroup: String,
    duration: u64,
    workers: usize,
    active_flags: Option<Vec<&'static str>>,
    repro: bool,
    probe_stack: Option<String>,
    auto_repro: bool,
    kernel_dir: Option<String>,
    work_type_override: Option<WorkType>,
) -> RunConfig {
    RunConfig {
        parent_cgroup,
        duration: Duration::from_secs(duration),
        workers_per_cgroup: workers,
        active_flags,
        repro,
        probe_stack,
        auto_repro,
        kernel_dir,
        work_type_override,
        ..Default::default()
    }
}

/// Check if a kernel .config contains CONFIG_SCHED_CLASS_EXT=y.
pub fn has_sched_ext(kernel_dir: &std::path::Path) -> bool {
    let config = kernel_dir.join(".config");
    std::fs::read_to_string(config)
        .map(|s| s.lines().any(|l| l == "CONFIG_SCHED_CLASS_EXT=y"))
        .unwrap_or(false)
}

/// Validate the output .config for critical options that the kconfig
/// fragment requested but the kernel build system may have silently
/// disabled (e.g. CONFIG_DEBUG_INFO_BTF requires pahole).
///
/// Call after `make` succeeds. Returns `Err` with a diagnostic
/// message listing missing options and likely causes.
pub fn validate_kernel_config(kernel_dir: &std::path::Path) -> Result<()> {
    let config_path = kernel_dir.join(".config");
    let config = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;

    let required: &[(&str, &str)] = &[
        (
            "CONFIG_SCHED_CLASS_EXT",
            "depends on CONFIG_DEBUG_INFO_BTF — ensure pahole >= 1.16 is installed (dwarves package)",
        ),
        (
            "CONFIG_DEBUG_INFO_BTF",
            "requires pahole >= 1.16 (dwarves package)",
        ),
        ("CONFIG_BPF_SYSCALL", "required for BPF program loading"),
        (
            "CONFIG_FTRACE",
            "gate for all tracing infrastructure — arm64 defconfig disables it, \
             silently dropping KPROBE_EVENTS and BPF_EVENTS",
        ),
        (
            "CONFIG_KPROBE_EVENTS",
            "required for ktstr probe pipeline (depends on FTRACE + KPROBES)",
        ),
        (
            "CONFIG_BPF_EVENTS",
            "required for BPF kprobe/tracepoint attachment (depends on KPROBE_EVENTS + PERF_EVENTS)",
        ),
    ];

    let mut missing = Vec::new();
    for &(option, hint) in required {
        let enabled = format!("{option}=y");
        if !config.lines().any(|l| l == enabled) {
            missing.push((option, hint));
        }
    }

    if !missing.is_empty() {
        let mut msg =
            String::from("kernel build completed but critical config options are missing:\n");
        for (option, hint) in &missing {
            msg.push_str(&format!("  {option} not set — {hint}\n"));
        }
        msg.push_str(
            "\nThe kernel build system silently disables options whose dependencies \
             are not met. Install missing tools and rebuild with --force.",
        );
        bail!("{msg}");
    }
    Ok(())
}

/// Result of the post-acquisition kernel build pipeline.
///
/// Returned by [`kernel_build_pipeline`] so callers can inspect
/// the cache entry and built image path.
#[non_exhaustive]
pub struct KernelBuildResult {
    /// Cache entry, if the build was cached. `None` for dirty trees
    /// or when cache store fails.
    pub entry: Option<crate::cache::CacheEntry>,
    /// Path to the built kernel image.
    pub image_path: std::path::PathBuf,
}

/// Post-acquisition kernel build pipeline.
///
/// Handles: clean, configure, build, validate config, generate
/// compile_commands.json for local trees, find image, strip vmlinux,
/// compute metadata, cache store, and remote cache store (when
/// enabled). Callers handle source acquisition.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `is_local_source` should be true when the user passed `--source`.
/// It controls the mrproper warning and `source_tree_path` in metadata.
pub fn kernel_build_pipeline(
    acquired: &crate::fetch::AcquiredSource,
    cache: &crate::cache::CacheDir,
    cli_label: &str,
    clean: bool,
    is_local_source: bool,
) -> Result<KernelBuildResult> {
    let source_dir = &acquired.source_dir;
    let (arch, image_name) = crate::fetch::arch_info();

    if clean {
        if !is_local_source {
            eprintln!(
                "{cli_label}: --clean is only meaningful with --source (downloaded sources start clean)"
            );
        } else {
            eprintln!("{cli_label}: make mrproper");
            run_make(source_dir, &["mrproper"])?;
        }
    }

    if !has_sched_ext(source_dir) {
        let sp = Spinner::start("Configuring kernel...");
        let result = configure_kernel(source_dir, EMBEDDED_KCONFIG);
        if result.is_err() {
            sp.clear();
        } else {
            sp.finish("Kernel configured");
        }
        result?;
    }

    let sp = Spinner::start("Building kernel...");
    let result = make_kernel_with_output(source_dir, Some(&sp));
    if result.is_err() {
        sp.clear();
    } else {
        sp.finish("Kernel built");
    }
    result?;

    // Validate critical config options were not silently disabled.
    validate_kernel_config(source_dir)?;

    // Generate compile_commands.json for local trees (LSP support).
    if !acquired.is_temp {
        let sp = Spinner::start("Generating compile_commands.json...");
        let result = run_make_with_output(source_dir, &["compile_commands.json"], Some(&sp));
        if result.is_err() {
            sp.clear();
        } else {
            sp.finish("compile_commands.json generated");
        }
        result?;
    }

    // Find the built kernel image and vmlinux.
    let image_path = crate::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| anyhow::anyhow!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let _stripped_dir;
    let vmlinux_ref = if vmlinux_path.exists() {
        let orig_mb = std::fs::metadata(&vmlinux_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        match crate::cache::strip_vmlinux_debug(&vmlinux_path) {
            Ok((dir, stripped_path)) => {
                let stripped_mb = std::fs::metadata(&stripped_path)
                    .map(|m| m.len() as f64 / (1024.0 * 1024.0))
                    .unwrap_or(0.0);
                eprintln!(
                    "{cli_label}: caching vmlinux ({orig_mb:.0} MB -> {stripped_mb:.0} MB, debug stripped)"
                );
                _stripped_dir = Some(dir);
                Some(stripped_path)
            }
            Err(e) => {
                eprintln!(
                    "{cli_label}: warning: vmlinux strip failed ({e:#}), caching unstripped ({orig_mb:.0} MB)"
                );
                _stripped_dir = None;
                Some(vmlinux_path.clone())
            }
        }
    } else {
        eprintln!("{cli_label}: warning: vmlinux not found, BTF will not be cached");
        _stripped_dir = None;
        None
    };
    let vmlinux_ref = vmlinux_ref.as_deref();

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("{cli_label}: kernel built at {}", image_path.display());
        eprintln!("{cli_label}: skipping cache (dirty tree)");
        return Ok(KernelBuildResult {
            entry: None,
            image_path,
        });
    }

    let config_path = source_dir.join(".config");
    let config_hash = if config_path.exists() {
        let data = std::fs::read(&config_path)?;
        Some(format!("{:08x}", crc32fast::hash(&data)))
    } else {
        None
    };

    let kconfig_hash = embedded_kconfig_hash();

    let metadata = crate::cache::KernelMetadata::new(
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
    .with_source_tree_path(if is_local_source {
        Some(acquired.source_dir.clone())
    } else {
        None
    });

    let config_ref = config_path.exists().then_some(config_path.as_path());
    let entry = match cache.store(
        &acquired.cache_key,
        &image_path,
        vmlinux_ref,
        config_ref,
        &metadata,
    ) {
        Ok(entry) => {
            success(&format!("\u{2713} Kernel cached: {}", acquired.cache_key));
            eprintln!(
                "{cli_label}: image: {}",
                entry.path.join(image_name).display()
            );
            if crate::remote_cache::is_enabled() {
                crate::remote_cache::remote_store(&entry, cli_label);
            }
            Some(entry)
        }
        Err(e) => {
            warn(&format!("{cli_label}: cache store failed: {e:#}"));
            None
        }
    };

    Ok(KernelBuildResult { entry, image_path })
}

/// Build the make arguments for a kernel build.
///
/// Returns the argument list that would be passed to `make` for a
/// parallel kernel build: `["-jN", "KCFLAGS=-Wno-error"]`.
fn build_make_args(nproc: usize) -> Vec<String> {
    vec![format!("-j{nproc}"), "KCFLAGS=-Wno-error".into()]
}

/// Read sidecar JSON files and return the gauntlet analysis report.
///
/// Source directory:
/// - `KTSTR_SIDECAR_DIR` if set, else
/// - the most recently modified subdirectory under
///   `{CARGO_TARGET_DIR or "target"}/ktstr/`.
///
/// `cargo ktstr stats` doesn't itself run a kernel, so it can't
/// reconstruct the `{kernel}-{git_short}` key the test process used; the
/// mtime fallback mirrors "show me the report from my last test run."
///
/// Returns an empty report with a warning on stderr when no sidecars
/// are found. This is not an error -- regular test runs that skip
/// gauntlet tests produce no sidecar files.
pub fn print_stats_report() -> String {
    let dir = match std::env::var("KTSTR_SIDECAR_DIR") {
        Ok(d) if !d.is_empty() => Some(std::path::PathBuf::from(d)),
        _ => crate::test_support::newest_run_dir(),
    };
    let report = match dir.as_deref() {
        Some(d) => crate::test_support::analyze_sidecars(Some(d)),
        None => String::new(),
    };
    if report.is_empty() {
        eprintln!("cargo ktstr: no sidecar data found (skipped)");
        return String::new();
    }
    report
}

/// List test runs under `{CARGO_TARGET_DIR or "target"}/ktstr/`.
pub fn list_runs() -> Result<()> {
    crate::stats::list_runs()
}

/// Compare two test runs and report regressions.
pub fn compare_runs(a: &str, b: &str, filter: Option<&str>, threshold: Option<f64>) -> Result<i32> {
    crate::stats::compare_runs(a, b, filter, threshold)
}

/// Pre-flight check for /dev/kvm availability and permissions.
pub fn check_kvm() -> Result<()> {
    use std::path::Path;
    if !Path::new("/dev/kvm").exists() {
        bail!(
            "/dev/kvm not found. KVM requires:\n  \
             - Linux kernel with KVM support (CONFIG_KVM)\n  \
             - Access to /dev/kvm (check permissions or add user to 'kvm' group)\n  \
             - Hardware virtualization enabled in BIOS (VT-x/AMD-V)"
        );
    }
    if let Err(e) = std::fs::File::open("/dev/kvm") {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            bail!(
                "/dev/kvm: permission denied. Add your user to the 'kvm' group:\n  \
                 sudo usermod -aG kvm $USER\n  \
                 then log out and back in."
            );
        }
        bail!("/dev/kvm: {e}");
    }
    Ok(())
}

/// Search PATH for a bare executable name.
fn resolve_in_path(name: &std::path::Path) -> Option<std::path::PathBuf> {
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

/// Resolve `--include-files` arguments into `(archive_path, host_path)` pairs.
///
/// Each path is resolved as follows:
/// - Explicit paths (starting with `/`, `.`, `..`, or containing `/`): must exist.
/// - Bare names: searched in PATH.
/// - Directories: walked recursively via `walkdir`, following symlinks.
///   The directory's basename becomes the root under `include-files/`.
///   Non-regular files (sockets, pipes, device nodes) are skipped.
///   Empty directories produce a warning to stderr.
/// - Regular files: included directly as `include-files/<filename>`.
pub fn resolve_include_files(
    paths: &[std::path::PathBuf],
) -> Result<Vec<(String, std::path::PathBuf)>> {
    use std::path::{Component, PathBuf};

    let mut resolved_includes: Vec<(String, PathBuf)> = Vec::new();
    for path in paths {
        let is_explicit_path = {
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
                resolve_in_path(path).ok_or_else(|| {
                    anyhow::anyhow!("-i {}: not found in filesystem or PATH", path.display())
                })?
            }
        };
        if resolved.is_dir() {
            let dir_name = resolved
                .file_name()
                .ok_or_else(|| {
                    anyhow::anyhow!("include directory has no name: {}", resolved.display())
                })?
                .to_string_lossy()
                .to_string();
            let prefix = format!("include-files/{dir_name}");
            let mut count = 0usize;
            for entry in walkdir::WalkDir::new(&resolved).follow_links(true) {
                let entry = entry.map_err(|e| anyhow::anyhow!("-i {}: {e}", resolved.display()))?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let rel = entry
                    .path()
                    .strip_prefix(&resolved)
                    .expect("walkdir entry is under root");
                let archive_path = format!("{prefix}/{}", rel.display());
                resolved_includes.push((archive_path, entry.into_path()));
                count += 1;
            }
            if count == 0 {
                eprintln!(
                    "warning: -i {}: directory contains no regular files",
                    resolved.display()
                );
            }
        } else {
            let file_name = resolved
                .file_name()
                .ok_or_else(|| {
                    anyhow::anyhow!("include file has no filename: {}", resolved.display())
                })?
                .to_string_lossy();
            let archive_path = format!("include-files/{file_name}");
            resolved_includes.push((archive_path, resolved));
        }
    }

    // Detect duplicate archive paths (e.g. `-i ./a/dir -i ./b/dir` both
    // containing the same relative file). The cpio format silently
    // overwrites earlier entries, so duplicates must be caught here.
    let mut seen = std::collections::HashMap::<&str, &std::path::Path>::new();
    for (archive_path, host_path) in &resolved_includes {
        if let Some(prev) = seen.insert(archive_path.as_str(), host_path.as_path()) {
            anyhow::bail!(
                "duplicate include path '{}': provided by both {} and {}",
                archive_path,
                prev.display(),
                host_path.display(),
            );
        }
    }

    Ok(resolved_includes)
}

/// Look up a cache key, checking local first, then remote (if enabled).
///
/// `cli_label` prefixes diagnostic output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn cache_lookup(
    cache: &crate::cache::CacheDir,
    cache_key: &str,
    cli_label: &str,
) -> Option<crate::cache::CacheEntry> {
    if let Some(entry) = cache.lookup(cache_key) {
        return Some(entry);
    }

    if crate::remote_cache::is_enabled() {
        return crate::remote_cache::remote_lookup(cache, cache_key, cli_label);
    }

    None
}

/// Resolve a Version or CacheKey identifier to a cache entry directory.
///
/// Lookup order: local cache, then the remote GHA cache when
/// `remote_cache::is_enabled()` returns true. Miss behavior differs
/// by variant:
/// - **Version**: major.minor prefixes (e.g. `"6.14"`) resolve to
///   the latest patch via [`crate::fetch::fetch_version_for_prefix`]
///   first. On full miss, downloads the kernel from kernel.org,
///   builds it, and stores it in the cache via
///   [`download_and_cache_version`].
/// - **CacheKey**: errors on miss — cache keys are content-hashes
///   and not downloadable. The error hint suggests running
///   `{cli_label} kernel list`.
///
/// `cli_label` is the human-facing command name (`"ktstr"` or
/// `"cargo ktstr"`) threaded into status output and error messages.
pub fn resolve_cached_kernel(
    id: &crate::kernel_path::KernelId,
    cli_label: &str,
) -> Result<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            // Major.minor prefix (e.g. "6.14") → resolve to latest patch.
            let resolved = if crate::fetch::is_major_minor_prefix(ver) {
                crate::fetch::fetch_version_for_prefix(ver, cli_label)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            } else {
                ver.clone()
            };
            let cache = crate::cache::CacheDir::new()?;
            let (arch, _) = crate::fetch::arch_info();
            let cache_key = format!("{resolved}-tarball-{arch}-kc{}", crate::cache_key_suffix());
            if let Some(entry) = cache_lookup(&cache, &cache_key, cli_label) {
                entry.metadata.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("cached entry {cache_key} has corrupt metadata")
                })?;
                return Ok(entry.path);
            }
            // Cache miss: download and build the requested version.
            download_and_cache_version(&resolved, cli_label)
        }
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new()?;
            if let Some(entry) = cache_lookup(&cache, key, cli_label) {
                entry
                    .metadata
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("cached entry {key} has corrupt metadata"))?;
                return Ok(entry.path);
            }
            bail!(
                "cache key {key} not found. \
                 Run `{cli_label} kernel list` to see available entries."
            )
        }
        KernelId::Path(_) => bail!("resolve_cached_kernel called with Path variant"),
    }
}

/// Policy controlling `resolve_kernel_image` behavior across binaries.
///
/// The resolution pipeline — directory auto-build, version
/// auto-download, cache lookup — is shared. `KernelResolvePolicy`
/// carries the per-binary knobs documented on each field.
pub struct KernelResolvePolicy<'a> {
    /// Accept raw kernel image files (e.g. `bzImage`, `Image`) passed
    /// as `--kernel`. `ktstr` uses `false` (rejects); `cargo ktstr`
    /// uses `true` (accepts).
    pub accept_raw_image: bool,
    /// CLI label for diagnostic status messages (e.g. `"ktstr"`,
    /// `"cargo ktstr"`), threaded into auto-build and auto-download
    /// status output.
    pub cli_label: &'a str,
}

/// Resolve a kernel identifier to a bootable image path.
///
/// Handles `KernelId` variants: directory (auto-build), version
/// string, and cache key. Raw image file acceptance is controlled by
/// `policy.accept_raw_image`. The `None` case resolves automatically
/// via cache then filesystem, falling back to auto-download.
pub fn resolve_kernel_image(
    kernel: Option<&str>,
    policy: &KernelResolvePolicy<'_>,
) -> Result<std::path::PathBuf> {
    use crate::kernel_path::KernelId;

    if let Some(val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let path = std::path::PathBuf::from(&p);
                if path.is_dir() {
                    resolve_kernel_dir(&path, policy.cli_label)
                } else if path.is_file() {
                    if policy.accept_raw_image {
                        Ok(path)
                    } else {
                        // Raw kernel image file — reject. Use a source
                        // directory or version string so kconfig validation
                        // and caching work correctly.
                        bail!(
                            "--kernel {}: raw image files are not supported. \
                             Pass a source directory, version, or cache key.",
                            path.display()
                        )
                    }
                } else {
                    bail!("kernel path not found: {}", path.display())
                }
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel(&id, policy.cli_label)?;
                crate::kernel_path::find_image_in_dir(&cache_dir).ok_or_else(|| {
                    anyhow::anyhow!("no kernel image found in {}", cache_dir.display())
                })
            }
        }
    } else {
        match crate::find_kernel()? {
            Some(image) => Ok(image),
            None => auto_download_kernel(policy.cli_label),
        }
    }
}

/// Auto-download, build, and cache the latest stable kernel.
///
/// Called when no --kernel is specified and no kernel is found via
/// cache or filesystem. Resolves the latest stable version and
/// delegates to [`download_and_cache_version`]. `cli_label` prefixes
/// status output (e.g. `"ktstr"`, `"cargo ktstr"`).
pub fn auto_download_kernel(cli_label: &str) -> Result<std::path::PathBuf> {
    status(&format!(
        "{cli_label}: no kernel found, downloading latest stable"
    ));

    let sp = Spinner::start("Fetching latest kernel version...");
    let ver =
        crate::fetch::fetch_latest_stable_version(cli_label).map_err(|e| anyhow::anyhow!("{e}"))?;
    sp.finish(format!("Latest stable: {ver}"));

    let cache_dir = download_and_cache_version(&ver, cli_label)?;
    let (_, image_name) = crate::fetch::arch_info();
    Ok(cache_dir.join(image_name))
}

/// Download a specific kernel version, build it, and store in the
/// cache. Returns the cache entry directory path (NOT the image path).
///
/// Checks the cache one more time with the resolved version to cover
/// races and prefix-resolved entries. Delegates to
/// [`kernel_build_pipeline`] for configure/build/validate/cache.
fn download_and_cache_version(version: &str, cli_label: &str) -> Result<std::path::PathBuf> {
    let (arch, _) = crate::fetch::arch_info();
    let cache_key = format!("{version}-tarball-{arch}-kc{}", crate::cache_key_suffix());

    // Check cache one more time with the resolved version.
    if let Ok(cache) = crate::cache::CacheDir::new()
        && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label)
        && entry.metadata.is_some()
    {
        return Ok(entry.path);
    }

    let tmp_dir = tempfile::TempDir::new()?;

    let sp = Spinner::start("Downloading kernel...");
    let acquired = crate::fetch::download_tarball(version, tmp_dir.path(), cli_label)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    sp.finish("Downloaded");

    let cache = crate::cache::CacheDir::new()?;
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, false)?;

    match result.entry {
        Some(entry) => Ok(entry.path),
        None => bail!(
            "kernel built but cache store failed — cannot return image from temporary directory"
        ),
    }
}

/// Resolve a kernel directory: auto-build from source tree.
///
/// Requires Makefile + Kconfig. Checks cache for clean trees,
/// delegates to [`kernel_build_pipeline`] on miss. `cli_label`
/// prefixes status output and is passed through to
/// [`kernel_build_pipeline`] as the diagnostic label.
pub fn resolve_kernel_dir(path: &std::path::Path, cli_label: &str) -> Result<std::path::PathBuf> {
    let is_source_tree = path.join("Makefile").exists() && path.join("Kconfig").exists();
    if !is_source_tree {
        bail!(
            "no kernel image found in {} (not a kernel source tree — \
             missing Makefile or Kconfig)",
            path.display()
        );
    }

    let acquired =
        crate::fetch::local_source(path, cli_label).map_err(|e| anyhow::anyhow!("{e}"))?;
    let (_, image_name) = crate::fetch::arch_info();
    let cache_key = acquired.cache_key.clone();

    // Clean trees: cache lookup before build.
    // Dirty trees: skip cache, always build.
    if !acquired.is_dirty
        && let Ok(cache) = crate::cache::CacheDir::new()
        && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label)
        && let Some(ref meta) = entry.metadata
    {
        let image = entry.path.join(&meta.image_name);
        if image.exists() {
            success(&format!("{cli_label}: using cached kernel {cache_key}"));
            return Ok(image);
        }
    }

    let cache = crate::cache::CacheDir::new()?;
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true)?;

    // Prefer the cached image path (stable across rebuilds).
    match result.entry {
        Some(entry) => Ok(entry.path.join(image_name)),
        None => Ok(result.image_path),
    }
}

/// Whether stderr supports color (cached per process).
pub fn stderr_color() -> bool {
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| unsafe { libc::isatty(libc::STDERR_FILENO) } != 0)
}

/// Print a styled status message to stderr.
fn status(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[1m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Print a green success message to stderr.
fn success(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[32m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Print a blue warning to stderr.
fn warn(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[34m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Progress spinner for long-running CLI operations.
///
/// When stderr is a TTY, draws an animated spinner via indicatif,
/// ticks in the background, and disables stdin echo to prevent
/// keypress jank. When stderr is not a TTY, skips all indicatif
/// machinery and falls back to plain stderr writes.
/// Call `finish` with a completion message or `clear` to remove.
#[derive(Clone)]
pub struct Spinner {
    /// None when stderr is not a TTY — no indicatif overhead.
    pb: Option<indicatif::ProgressBar>,
    /// Saved termios for echo restore. None when stdin is not a tty
    /// or when the spinner is inactive (non-TTY stderr).
    saved_termios: Option<std::sync::Arc<std::sync::Mutex<libc::termios>>>,
}

impl Spinner {
    /// Start a spinner with the given message (e.g. "Building kernel...").
    ///
    /// When stderr is not a TTY, no ProgressBar or ticker thread is
    /// created — all output methods fall back to plain `eprintln!`.
    pub fn start(msg: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        if !stderr_color() {
            return Spinner {
                pb: None,
                saved_termios: None,
            };
        }

        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(
            indicatif::ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .expect("valid template"),
        );
        pb.set_message(msg);
        pb.enable_steady_tick(Duration::from_millis(80));

        // indicatif hides the bar when NO_COLOR is set or TERM is
        // dumb, even on a real TTY. Downgrade to the non-TTY path
        // so println/finish output is not silently dropped.
        if pb.is_hidden() {
            return Spinner {
                pb: None,
                saved_termios: None,
            };
        }

        let saved_termios = Self::disable_echo();

        Spinner {
            pb: Some(pb),
            saved_termios,
        }
    }

    fn disable_echo() -> Option<std::sync::Arc<std::sync::Mutex<libc::termios>>> {
        unsafe {
            let fd = libc::STDIN_FILENO;
            if libc::isatty(fd) == 0 {
                return None;
            }
            let mut termios: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut termios) != 0 {
                return None;
            }
            let saved = termios;
            termios.c_lflag &= !libc::ECHO;
            libc::tcsetattr(fd, libc::TCSANOW, &termios);
            Some(std::sync::Arc::new(std::sync::Mutex::new(saved)))
        }
    }

    fn restore_echo(&self) {
        if let Some(ref saved) = self.saved_termios
            && let Ok(termios) = saved.lock()
        {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &*termios);
            }
        }
    }

    /// Update the spinner message.
    pub fn set_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        if let Some(ref pb) = self.pb {
            pb.set_message(msg);
        }
    }

    /// Finish the spinner, replacing it with a completion message.
    ///
    /// In non-TTY mode, prints the message to stderr directly.
    pub fn finish(self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.restore_echo();
        match self.pb {
            Some(pb) => pb.finish_with_message(msg),
            None => eprintln!("{}", msg.into()),
        }
    }

    /// Print a line above the spinner. The spinner redraws below.
    ///
    /// In non-TTY mode, prints directly to stderr.
    pub fn println(&self, msg: impl AsRef<str>) {
        match self.pb {
            Some(ref pb) => pb.println(msg),
            None => eprintln!("{}", msg.as_ref()),
        }
    }

    /// Suspend the spinner tick, execute a closure, then resume.
    /// Use for terminal output that must not race with the spinner.
    ///
    /// In non-TTY mode, calls `f` directly (no spinner to suspend).
    pub fn suspend<F: FnOnce() -> R, R>(&self, f: F) -> R {
        match self.pb {
            Some(ref pb) => pb.suspend(f),
            None => f(),
        }
    }

    /// Clear the spinner from the terminal.
    ///
    /// In non-TTY mode, this is a no-op.
    pub fn clear(self) {
        self.restore_echo();
        if let Some(pb) = self.pb {
            pb.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario;

    // -- resolve_flags --

    #[test]
    fn cli_resolve_flags_none_returns_none() {
        assert!(resolve_flags(None).unwrap().is_none());
    }

    #[test]
    fn cli_resolve_flags_valid_single() {
        let result = resolve_flags(Some(vec!["llc".into()])).unwrap().unwrap();
        assert_eq!(result, vec!["llc"]);
    }

    #[test]
    fn cli_resolve_flags_valid_multiple() {
        let result = resolve_flags(Some(vec!["llc".into(), "borrow".into()]))
            .unwrap()
            .unwrap();
        assert_eq!(result, vec!["llc", "borrow"]);
    }

    #[test]
    fn cli_resolve_flags_all_valid() {
        let all: Vec<String> = flags::ALL.iter().map(|s| s.to_string()).collect();
        let result = resolve_flags(Some(all)).unwrap().unwrap();
        assert_eq!(result.len(), flags::ALL.len());
    }

    #[test]
    fn cli_resolve_flags_unknown_errors() {
        let err = resolve_flags(Some(vec!["nonexistent".into()])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown flag: 'nonexistent'"), "{msg}");
        assert!(msg.contains("valid flags:"), "{msg}");
    }

    #[test]
    fn cli_resolve_flags_mixed_valid_and_unknown_errors() {
        let err = resolve_flags(Some(vec!["llc".into(), "bogus".into()])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown flag: 'bogus'"), "{msg}");
    }

    // -- parse_work_type --

    #[test]
    fn cli_parse_work_type_none_returns_none() {
        assert!(parse_work_type(None).unwrap().is_none());
    }

    #[test]
    fn cli_parse_work_type_cpu_spin() {
        let wt = parse_work_type(Some("CpuSpin")).unwrap().unwrap();
        assert_eq!(wt.name(), "CpuSpin");
    }

    #[test]
    fn cli_parse_work_type_yield_heavy() {
        let wt = parse_work_type(Some("YieldHeavy")).unwrap().unwrap();
        assert_eq!(wt.name(), "YieldHeavy");
    }

    #[test]
    fn cli_parse_work_type_all_valid() {
        for &name in WorkType::ALL_NAMES {
            if name == "Sequence" || name == "Custom" {
                continue;
            }
            let wt = parse_work_type(Some(name)).unwrap().unwrap();
            assert_eq!(wt.name(), name);
        }
    }

    #[test]
    fn cli_parse_work_type_unknown_errors() {
        let err = parse_work_type(Some("Nonexistent")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown work type: 'Nonexistent'"), "{msg}");
        assert!(msg.contains("valid types:"), "{msg}");
    }

    #[test]
    fn cli_parse_work_type_sequence_errors() {
        let err = parse_work_type(Some("Sequence")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown work type: 'Sequence'"), "{msg}");
    }

    #[test]
    fn cli_parse_work_type_case_sensitive() {
        let err = parse_work_type(Some("cpuspin")).unwrap_err();
        assert!(format!("{err}").contains("unknown work type:"));
    }

    // -- filter_scenarios --

    #[test]
    fn cli_filter_scenarios_no_filter_returns_all() {
        let scenarios = scenario::all_scenarios();
        let result = filter_scenarios(&scenarios, None).unwrap();
        assert_eq!(result.len(), scenarios.len());
    }

    #[test]
    fn cli_filter_scenarios_matching_filter() {
        let scenarios = scenario::all_scenarios();
        let first_name = scenarios[0].name;
        let result = filter_scenarios(&scenarios, Some(first_name)).unwrap();
        assert!(!result.is_empty());
        for s in &result {
            assert!(s.name.contains(first_name));
        }
    }

    #[test]
    fn cli_filter_scenarios_no_match_errors() {
        let scenarios = scenario::all_scenarios();
        let err = filter_scenarios(&scenarios, Some("__nonexistent_scenario_xyz__")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no scenarios matched"), "{msg}");
        assert!(msg.contains("ktstr list"), "{msg}");
    }

    #[test]
    fn cli_filter_scenarios_partial_match() {
        let scenarios = scenario::all_scenarios();
        let result = filter_scenarios(&scenarios, Some("steady")).unwrap();
        assert!(!result.is_empty());
    }

    // -- build_run_config --

    #[test]
    fn cli_build_run_config_defaults() {
        let config = build_run_config(
            "/sys/fs/cgroup/ktstr".into(),
            20,
            4,
            None,
            false,
            None,
            false,
            None,
            None,
        );
        assert_eq!(config.parent_cgroup, "/sys/fs/cgroup/ktstr");
        assert_eq!(config.duration, Duration::from_secs(20));
        assert_eq!(config.workers_per_cgroup, 4);
        assert!(config.active_flags.is_none());
        assert!(!config.repro);
        assert!(config.probe_stack.is_none());
        assert!(!config.auto_repro);
        assert!(config.kernel_dir.is_none());
        assert!(config.work_type_override.is_none());
    }

    #[test]
    fn cli_build_run_config_all_fields() {
        let config = build_run_config(
            "/sys/fs/cgroup/test".into(),
            30,
            8,
            Some(vec!["llc", "borrow"]),
            true,
            Some("do_enqueue_task".into()),
            true,
            Some("/usr/src/linux".into()),
            Some(WorkType::Mixed),
        );
        assert_eq!(config.parent_cgroup, "/sys/fs/cgroup/test");
        assert_eq!(config.duration, Duration::from_secs(30));
        assert_eq!(config.workers_per_cgroup, 8);
        let af = config.active_flags.unwrap();
        assert_eq!(af, vec!["llc", "borrow"]);
        assert!(config.repro);
        assert_eq!(config.probe_stack.as_deref(), Some("do_enqueue_task"));
        assert!(config.auto_repro);
        assert_eq!(config.kernel_dir.as_deref(), Some("/usr/src/linux"));
        assert!(config.work_type_override.is_some());
    }

    #[test]
    fn cli_build_run_config_duration_converts() {
        let config = build_run_config("cg".into(), 60, 1, None, false, None, false, None, None);
        assert_eq!(config.duration, Duration::from_secs(60));
    }

    // -- scenario catalog --

    #[test]
    fn cli_all_scenarios_non_empty() {
        let scenarios = scenario::all_scenarios();
        assert!(!scenarios.is_empty());
    }

    #[test]
    fn cli_all_scenarios_have_names() {
        for s in &scenario::all_scenarios() {
            assert!(!s.name.is_empty());
            assert!(!s.category.is_empty());
        }
    }

    // -- has_sched_ext --

    #[test]
    fn cli_has_sched_ext_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_SCHED_CLASS_EXT=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_absent() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_module_not_builtin() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "CONFIG_SCHED_CLASS_EXT=m\n").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_commented_out() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "# CONFIG_SCHED_CLASS_EXT is not set\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_no_config_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_empty_config() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    // -- build_make_args --

    #[test]
    fn cli_build_make_args_single_core() {
        let args = build_make_args(1);
        assert_eq!(args, vec!["-j1", "KCFLAGS=-Wno-error"]);
    }

    #[test]
    fn cli_build_make_args_multi_core() {
        let args = build_make_args(16);
        assert_eq!(args, vec!["-j16", "KCFLAGS=-Wno-error"]);
    }

    // -- analyze_sidecars (library API used by print_stats_report) --

    #[test]
    fn cli_analyze_sidecars_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = crate::test_support::analyze_sidecars(Some(tmp.path()));
        assert!(result.is_empty());
    }

    #[test]
    fn cli_analyze_sidecars_nonexistent_dir() {
        let result =
            crate::test_support::analyze_sidecars(Some(std::path::Path::new("/nonexistent/path")));
        assert!(result.is_empty());
    }

    // -- days_to_ymd --

    #[test]
    fn days_to_ymd_epoch() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        // 2024-01-01 = 19723 days since epoch
        assert_eq!(days_to_ymd(19723), (2024, 1, 1));
    }

    #[test]
    fn days_to_ymd_leap_day() {
        // 2024-02-29 = 19723 + 31 (jan) + 28 (feb 1-28) = 19782
        assert_eq!(days_to_ymd(19782), (2024, 2, 29));
    }

    #[test]
    fn days_to_ymd_end_of_year() {
        // 2023-12-31 = 19722
        assert_eq!(days_to_ymd(19722), (2023, 12, 31));
    }

    // -- validate_kernel_config --

    #[test]
    fn validate_kernel_config_all_present() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".config"),
            "CONFIG_SCHED_CLASS_EXT=y\n\
             CONFIG_DEBUG_INFO_BTF=y\n\
             CONFIG_BPF_SYSCALL=y\n\
             CONFIG_FTRACE=y\n\
             CONFIG_KPROBE_EVENTS=y\n\
             CONFIG_BPF_EVENTS=y\n",
        )
        .unwrap();
        assert!(validate_kernel_config(dir.path()).is_ok());
    }

    #[test]
    fn validate_kernel_config_missing_btf() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".config"),
            "CONFIG_SCHED_CLASS_EXT=y\n\
             CONFIG_BPF_SYSCALL=y\n\
             CONFIG_FTRACE=y\n\
             CONFIG_KPROBE_EVENTS=y\n\
             CONFIG_BPF_EVENTS=y\n",
        )
        .unwrap();
        let err = validate_kernel_config(dir.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CONFIG_DEBUG_INFO_BTF"), "got: {msg}");
    }

    #[test]
    fn validate_kernel_config_missing_multiple() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF_SYSCALL=y\n").unwrap();
        let err = validate_kernel_config(dir.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CONFIG_SCHED_CLASS_EXT"), "got: {msg}");
        assert!(msg.contains("CONFIG_DEBUG_INFO_BTF"), "got: {msg}");
    }

    #[test]
    fn validate_kernel_config_no_config_file() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(validate_kernel_config(dir.path()).is_err());
    }

    // -- configure_kernel --

    #[test]
    fn configure_kernel_appends_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF=y\n").unwrap();
        // configure_kernel runs `make olddefconfig` after appending.
        // Provide a stub Makefile so `make` succeeds without a real
        // kernel tree.
        std::fs::write(dir.path().join("Makefile"), "olddefconfig:\n\t@true\n").unwrap();
        let fragment = "CONFIG_EXTRA=y\n";
        configure_kernel(dir.path(), fragment).unwrap();
        let config = std::fs::read_to_string(dir.path().join(".config")).unwrap();
        assert!(config.contains("CONFIG_EXTRA=y"));
        assert!(config.contains("CONFIG_BPF=y"));
    }

    #[test]
    fn configure_kernel_skips_when_present() {
        let dir = tempfile::TempDir::new().unwrap();
        let initial = "CONFIG_BPF=y\nCONFIG_EXTRA=y\n";
        std::fs::write(dir.path().join(".config"), initial).unwrap();
        let fragment = "CONFIG_EXTRA=y\n";
        configure_kernel(dir.path(), fragment).unwrap();
        let config = std::fs::read_to_string(dir.path().join(".config")).unwrap();
        // Should not have appended (mtime preserved behavior).
        assert_eq!(config, initial);
    }

    // -- resolve_in_path --

    #[test]
    fn resolve_in_path_finds_sh() {
        let result = resolve_in_path(std::path::Path::new("sh"));
        assert!(result.is_some(), "sh should be in PATH");
        assert!(result.unwrap().exists());
    }

    #[test]
    fn resolve_in_path_nonexistent() {
        let result = resolve_in_path(std::path::Path::new("nonexistent_binary_xyz_12345"));
        assert!(result.is_none());
    }

    // -- resolve_include_files --

    #[test]
    fn resolve_include_files_single_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = resolve_include_files(&[file]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].0.contains("test.txt"));
    }

    #[test]
    fn resolve_include_files_nonexistent() {
        let result = resolve_include_files(&[std::path::PathBuf::from("/nonexistent/file.txt")]);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_include_files_bare_name_in_path() {
        // "sh" is in PATH on all systems.
        let result = resolve_include_files(&[std::path::PathBuf::from("sh")]);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].0.contains("sh"));
    }
}
