//! CLI support functions shared between `ktstr` and `cargo-ktstr`.
//!
//! Validation, configuration, and kernel/KVM resolution logic used
//! by both binaries.

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::cache::{CacheDir, CacheEntry};
use crate::runner::RunConfig;
use crate::scenario::{Scenario, flags};
use crate::workload::WorkType;

/// ktstr.kconfig embedded at compile time.
pub const EMBEDDED_KCONFIG: &str = include_str!("../ktstr.kconfig");

/// Compute CRC32 of the embedded ktstr.kconfig fragment.
pub fn embedded_kconfig_hash() -> String {
    let hash = crc32fast::hash(EMBEDDED_KCONFIG.as_bytes());
    format!("{hash:08x}")
}

/// Format a human-readable table row for a cache entry.
pub fn format_entry_row(entry: &CacheEntry, kconfig_hash: &str) -> String {
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

/// Current time as ISO 8601 string (UTC, second precision).
pub fn now_iso8601() -> String {
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
pub fn days_to_ymd(days: u64) -> (u64, u64, u64) {
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
        println!("no cached kernels. Run `kernel build` to download and build a kernel.");
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

/// Configure the kernel with sched_ext support.
///
/// Appends the kconfig fragment to .config and runs `make olddefconfig`
/// to resolve dependencies. kconfig uses last-value-wins semantics for
/// duplicate symbols (confdata.c:conf_read_simple), so appending is
/// equivalent to merge_config.sh's delete-then-append approach -- both
/// produce the same resolved config after olddefconfig.
pub fn configure_kernel(kernel_dir: &Path, fragment: &str) -> Result<()> {
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
pub fn make_kernel(kernel_dir: &Path) -> Result<()> {
    let nproc = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let args = build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make(kernel_dir, &arg_refs)
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

/// Build the make arguments for a kernel build.
///
/// Returns the argument list that would be passed to `make` for a
/// parallel kernel build: `["-jN", "KCFLAGS=-Wno-error"]`.
pub fn build_make_args(nproc: usize) -> Vec<String> {
    vec![format!("-j{nproc}"), "KCFLAGS=-Wno-error".into()]
}

/// Read sidecar JSON files and return the gauntlet analysis report.
///
/// When `dir` is `Some`, reads from that directory. Otherwise uses
/// the default sidecar directory (KTSTR_SIDECAR_DIR or
/// `target/ktstr/{branch}-{hash}/`).
///
/// Returns an empty report with a warning on stderr when no sidecars
/// are found. This is not an error -- regular test runs that skip
/// gauntlet tests produce no sidecar files.
pub fn run_test_stats(dir: Option<&std::path::Path>) -> String {
    let report = crate::test_support::analyze_sidecars(dir);
    if report.is_empty() {
        eprintln!("cargo-ktstr: no sidecar data found (skipped)");
        return String::new();
    }
    report
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
pub fn resolve_in_path(name: &std::path::Path) -> Option<std::path::PathBuf> {
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

/// Resolve the cache entry directory from a Version or CacheKey identifier.
///
/// Checks local cache first. When the `gha-cache` feature is enabled
/// and `remote_cache::is_enabled()` returns true, falls back to the
/// remote GHA cache on local miss.
pub fn resolve_cached_kernel(id: &crate::kernel_path::KernelId) -> Result<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            let cache = crate::cache::CacheDir::new()?;
            let (arch, _) = crate::fetch::arch_info();
            let cache_key = format!("{ver}-tarball-{arch}");
            if let Some(entry) = cache.lookup(&cache_key) {
                entry.metadata.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("cached entry {cache_key} has corrupt metadata")
                })?;
                return Ok(entry.path);
            }
            #[cfg(feature = "gha-cache")]
            if crate::remote_cache::is_enabled() {
                if let Some(entry) = crate::remote_cache::remote_lookup(&cache, &cache_key) {
                    entry.metadata.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("cached entry {cache_key} has corrupt metadata")
                    })?;
                    return Ok(entry.path);
                }
            }
            bail!(
                "kernel version {ver} not found in cache. \
                 Build with `kernel build {ver}` first."
            )
        }
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new()?;
            if let Some(entry) = cache.lookup(key) {
                entry
                    .metadata
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("cached entry {key} has corrupt metadata"))?;
                return Ok(entry.path);
            }
            #[cfg(feature = "gha-cache")]
            if crate::remote_cache::is_enabled() {
                if let Some(entry) = crate::remote_cache::remote_lookup(&cache, key) {
                    entry.metadata.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("cached entry {key} has corrupt metadata")
                    })?;
                    return Ok(entry.path);
                }
            }
            bail!(
                "cache key {key} not found. \
                 Run `kernel list` to see available entries."
            )
        }
        KernelId::Path(_) => bail!("resolve_cached_kernel called with Path variant"),
    }
}

/// Resolve a kernel identifier to a bootable image path.
///
/// Handles all `KernelId` variants (path to file or directory,
/// version string, cache key) and the `None` case (automatic
/// resolution via cache then filesystem).
pub fn resolve_kernel_image(kernel: Option<&str>) -> Result<std::path::PathBuf> {
    use crate::kernel_path::KernelId;

    if let Some(val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let path = std::path::PathBuf::from(&p);
                if path.is_file() {
                    Ok(path)
                } else if path.is_dir() {
                    crate::kernel_path::find_image_in_dir(&path).ok_or_else(|| {
                        anyhow::anyhow!("no kernel image found in {}", path.display())
                    })
                } else {
                    bail!("kernel path not found: {}", path.display())
                }
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel(&id)?;
                crate::kernel_path::find_image_in_dir(&cache_dir).ok_or_else(|| {
                    anyhow::anyhow!("no kernel image found in {}", cache_dir.display())
                })
            }
        }
    } else {
        crate::find_kernel()?.ok_or_else(|| {
            anyhow::anyhow!(
                "no kernel found. Provide --kernel or run \
                 `kernel build` to download and cache one."
            )
        })
    }
}

/// Progress spinner for long-running CLI operations.
///
/// Draws to stderr via indicatif. Automatically ticks in the background.
/// Call `finish` with a completion message or `clear` to remove.
pub struct Spinner(indicatif::ProgressBar);

impl Spinner {
    /// Start a spinner with the given message (e.g. "Building kernel...").
    pub fn start(msg: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(
            indicatif::ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .expect("valid template"),
        );
        pb.set_message(msg);
        pb.enable_steady_tick(Duration::from_millis(80));
        Spinner(pb)
    }

    /// Update the spinner message.
    pub fn set_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.0.set_message(msg);
    }

    /// Finish the spinner, replacing it with a completion message.
    pub fn finish(self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.0.finish_with_message(msg);
    }

    /// Clear the spinner from the terminal.
    pub fn clear(self) {
        self.0.finish_and_clear();
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
            if name == "Sequence" {
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

    // -- run_test_stats --

    #[test]
    fn cli_test_stats_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = run_test_stats(Some(tmp.path()));
        assert!(result.is_empty());
    }

    #[test]
    fn cli_test_stats_nonexistent_dir() {
        let result = run_test_stats(Some(std::path::Path::new("/nonexistent/path")));
        assert!(result.is_empty());
    }
}
