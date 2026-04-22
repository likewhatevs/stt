//! CLI support functions shared between `ktstr` and `cargo-ktstr`.
//!
//! Validation, configuration, and kernel/KVM resolution logic used
//! by both binaries.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Subcommand;

use crate::cache::{CacheDir, CacheEntry, KconfigStatus};
use crate::runner::RunConfig;
use crate::scenario::{Scenario, flags};
use crate::workload::WorkType;

/// Shared `kernel` subcommand tree used by both `ktstr` and
/// `cargo ktstr`. The two binaries embed this as
/// `ktstr kernel <subcmd>` / `cargo ktstr kernel <subcmd>` and
/// dispatch identically; defining the variants once means a new
/// `kernel` subcommand (or a flag change) lands in both surfaces by
/// construction.
#[derive(Subcommand)]
pub enum KernelCommand {
    /// List cached kernel images.
    List {
        /// Output in JSON format for CI scripting. Each entry's
        /// `eol` boolean is derived by fetching kernel.org's
        /// `releases.json` to learn the active series prefixes; on
        /// fetch failure (offline, kernel.org unreachable, response
        /// malformed) the active list is empty and no entry is
        /// flagged EOL. The cache listing itself is local and
        /// always succeeds; only the EOL annotation degrades.
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
        /// Run `make mrproper` before configuring. Only meaningful
        /// with `--source`: downloaded tarball and freshly cloned
        /// git sources start clean, so this flag prints a notice
        /// and is ignored in those modes.
        #[arg(long)]
        clean: bool,
    },
    /// Remove cached kernel images.
    Clean {
        /// Keep the N most recent cached kernels. When absent, removes
        /// every cached entry.
        #[arg(long)]
        keep: Option<usize>,
        /// Skip the y/N confirmation prompt before deleting. Always
        /// required in non-interactive contexts: without `--force`
        /// the command bails on a non-tty stdin rather than hang
        /// waiting for input. In an interactive shell, omit
        /// `--force` to be prompted.
        #[arg(long)]
        force: bool,
    },
}

/// Help text for `--kernel` in contexts that reject raw image files:
/// `cargo ktstr test`, `cargo ktstr coverage`, and `ktstr shell`.
/// Matches `KernelResolvePolicy { accept_raw_image: false, .. }`.
///
/// Raw images are rejected here because these commands depend on a
/// matching `vmlinux` and the cached kconfig fragment alongside the
/// image (test/coverage need BTF, `ktstr shell` reuses the cache
/// entry for kconfig discovery). A bare `bzImage`/`Image` passed
/// directly carries neither, so silently accepting it would produce
/// hard-to-diagnose mid-run failures. The verifier and
/// `cargo ktstr shell` accept raw images because their flows do not
/// need that companion metadata; see [`KERNEL_HELP_RAW_OK`].
pub const KERNEL_HELP_NO_RAW: &str = "Kernel identifier: a source directory \
     path (e.g. `../linux`), a version (`6.14.2`, or major.minor prefix \
     `6.14` for latest patch), or a cache key (see `kernel list`). Raw \
     image files are rejected. Source directories auto-build (can be slow \
     on a fresh tree); versions auto-download from kernel.org on cache \
     miss.";

/// Help text for `--kernel` in contexts that accept raw image files:
/// `cargo ktstr verifier` and `cargo ktstr shell`. Matches
/// `KernelResolvePolicy { accept_raw_image: true, .. }`. See
/// [`KERNEL_HELP_NO_RAW`] for the converse and the rationale for
/// the asymmetry.
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

/// Return `true` when `version`'s major.minor series is absent
/// from a non-empty `active_prefixes` list — i.e. the version is
/// end-of-life relative to the kernel.org releases snapshot the
/// caller supplied.
///
/// Returns `false` in three cases:
/// - `active_prefixes` is empty. Callers pass an empty slice to
///   signal "active list unknown" (fetch failure, or skipped
///   lookup), per the `kernel list --json` doc contract that
///   fetch failure must not flag any entry EOL. Without the
///   explicit empty-slice guard, `!any(..)` on an empty iterator
///   is `true` and every entry would be tagged EOL — the exact
///   opposite of the contract.
/// - `version` has no parseable major.minor prefix (e.g. a cache
///   key or freeform string).
/// - `version`'s major.minor prefix appears in `active_prefixes`.
fn is_eol(version: &str, active_prefixes: &[String]) -> bool {
    if active_prefixes.is_empty() {
        return false;
    }
    let Some(prefix) = version_prefix(version) else {
        return false;
    };
    !active_prefixes.iter().any(|p| p == &prefix)
}

/// Fetch active kernel series prefixes from releases.json.
///
/// Returns major.minor prefixes for every stable/longterm/mainline
/// entry on success. Propagates the underlying
/// [`crate::fetch::fetch_releases`] error on failure (network error,
/// HTTP status, JSON parse failure, missing releases array) so
/// callers can distinguish "fetched and empty" (kernel.org shipped
/// no active series — a violated assumption) from "fetch failed"
/// (transient outage where EOL annotation must degrade, not flip).
///
/// See [`is_eol`]'s empty-slice guard for the recommended fallback pattern.
pub(crate) fn fetch_active_prefixes() -> Result<Vec<String>, String> {
    let releases = crate::fetch::fetch_releases()?;
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
    Ok(prefixes)
}

/// Format a human-readable table row for a cache entry.
pub fn format_entry_row(
    entry: &CacheEntry,
    kconfig_hash: &str,
    active_prefixes: &[String],
) -> String {
    let meta = &entry.metadata;
    let version = meta.version.as_deref().unwrap_or("-");
    let source = meta.source.to_string();
    let mut tags = String::new();
    // Compose the kconfig tag from `KconfigStatus`'s `Display` impl
    // so the tag word ("stale" / "untracked") and the JSON
    // `kconfig_status` field both flow through one source of truth.
    // `Matches` emits no tag — `kernel list` only annotates entries
    // that deviate from the current kconfig.
    let status = entry.kconfig_status(kconfig_hash);
    if !matches!(status, KconfigStatus::Matches) {
        tags.push_str(&format!(" ({status} kconfig)"));
    }
    if version != "-" && is_eol(version, active_prefixes) {
        tags.push_str(" (EOL)");
    }
    format!(
        "  {:<48} {:<12} {:<8} {:<7} {}{}",
        entry.key, version, source, meta.arch, meta.built_at, tags,
    )
}

/// List cached kernel images.
pub fn kernel_list(json: bool) -> Result<()> {
    let cache = CacheDir::new()?;
    let entries = cache.list()?;
    let kconfig_hash = embedded_kconfig_hash();

    let active_prefixes = match fetch_active_prefixes() {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "kernel list: failed to fetch active kernel series ({e}); \
                 EOL annotation disabled for this run",
            );
            Vec::new()
        }
    };

    if json {
        let json_entries: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| match e {
                crate::cache::ListedEntry::Valid(entry) => {
                    let meta = &entry.metadata;
                    let v = meta.version.as_deref().unwrap_or("-");
                    let eol = v != "-" && is_eol(v, &active_prefixes);
                    let kconfig_status = entry.kconfig_status(&kconfig_hash).to_string();
                    serde_json::json!({
                        "key": entry.key,
                        "path": entry.path.display().to_string(),
                        "version": meta.version,
                        "source": meta.source,
                        "arch": meta.arch,
                        "built_at": meta.built_at,
                        "ktstr_kconfig_hash": meta.ktstr_kconfig_hash,
                        "kconfig_status": kconfig_status,
                        "eol": eol,
                        "config_hash": meta.config_hash,
                        "image_name": meta.image_name,
                        "image_path": entry.image_path().display().to_string(),
                        "has_vmlinux": meta.has_vmlinux,
                    })
                }
                crate::cache::ListedEntry::Corrupt { key, path, reason } => serde_json::json!({
                    "key": key,
                    "path": path.display().to_string(),
                    "error": reason,
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
    let mut any_stale = false;
    for listed in &entries {
        match listed {
            crate::cache::ListedEntry::Valid(entry) => {
                if entry.kconfig_status(&kconfig_hash).is_stale() {
                    any_stale = true;
                }
                println!(
                    "{}",
                    format_entry_row(entry, &kconfig_hash, &active_prefixes)
                );
            }
            crate::cache::ListedEntry::Corrupt { key, reason, .. } => {
                println!("  {key:<48} (corrupt: {reason})");
            }
        }
    }
    if any_stale {
        eprintln!(
            "warning: entries marked (stale kconfig) were built against a different ktstr.kconfig. \
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
    let to_remove: Vec<&crate::cache::ListedEntry> = entries.iter().skip(skip).collect();

    if to_remove.is_empty() {
        println!("nothing to clean");
        return Ok(());
    }

    if !force {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            bail!("confirmation requires a terminal. Use --force to skip.");
        }
        println!("the following entries will be removed:");
        for listed in &to_remove {
            match listed {
                crate::cache::ListedEntry::Valid(entry) => {
                    println!("{}", format_entry_row(entry, &kconfig_hash, &[]));
                }
                crate::cache::ListedEntry::Corrupt { key, reason, .. } => {
                    println!("  {key:<48} (corrupt: {reason})");
                }
            }
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
    for listed in &to_remove {
        match std::fs::remove_dir_all(listed.path()) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                removed += 1;
            }
            Err(e) => {
                last_err = Some(format!("remove {}: {e}", listed.key()));
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
/// Pure check used by [`configure_kernel`]: every non-empty line of
/// `fragment` (including disable directives like
/// `# CONFIG_X is not set`) must appear as an exact line of `config`.
///
/// Exact-line matching avoids the prefix-aliasing hazard of the prior
/// `config.contains(fragment_line)` formulation, where a fragment line
/// false-matches when it appears as a substring of an unrelated
/// `.config` line — e.g. fragment `CONFIG_NR_CPUS=1` appearing inside
/// `CONFIG_NR_CPUS=128`, or any numeric-tail option where the
/// requested value is a prefix of the existing value.
///
/// `# CONFIG_X is not set` comments ARE kconfig semantics (the
/// canonical way to disable an option), so they participate in the
/// check; the only lines skipped are genuinely empty ones.
fn all_fragment_lines_present(fragment: &str, config: &str) -> bool {
    let existing: std::collections::HashSet<&str> = config.lines().map(str::trim).collect();
    fragment
        .lines()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .all(|t| existing.contains(t))
}

/// Checks each non-empty line of the fragment against the current
/// `.config` via [`all_fragment_lines_present`]. If every fragment
/// line already appears in `.config`, the file is not touched
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
    if all_fragment_lines_present(fragment, &config_content) {
        return Ok(());
    }

    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(&config_path)?;
    std::io::Write::write_all(&mut config, fragment.as_bytes())?;

    run_make(kernel_dir, &["olddefconfig"])?;

    Ok(())
}

/// Drain a reader into a `Vec<String>`, one entry per newline-delimited
/// chunk, with a final partial chunk (no trailing newline) emitted
/// with the same lossy-UTF-8 conversion. Byte-oriented so non-UTF-8
/// input survives via `from_utf8_lossy` (U+FFFD replacement) instead
/// of being dropped at the line boundary. Strips the trailing `\n`
/// and an optional preceding `\r` so CRLF input matches LF semantics.
/// Calls `on_line` for each line before appending to the returned
/// `Vec`.
///
/// Extracted from [`run_make_with_output`] so the read logic is
/// testable with in-memory readers (the caller still owns child
/// kill+wait).
fn drain_lines_lossy(
    mut reader: impl BufRead,
    mut on_line: impl FnMut(&str),
) -> std::io::Result<Vec<String>> {
    let mut captured = Vec::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        let mut slice: &[u8] = &buf;
        if let Some(rest) = slice.strip_suffix(b"\n") {
            slice = rest;
            if let Some(rest) = slice.strip_suffix(b"\r") {
                slice = rest;
            }
        }
        let line = String::from_utf8_lossy(slice).into_owned();
        on_line(&line);
        captured.push(line);
    }
    Ok(captured)
}

/// Run make with merged stdout+stderr piped through a spinner.
///
/// Creates a single pipe via `nix::unistd::pipe2(O_CLOEXEC)`, hands
/// the write end to the child's stdout AND stderr (a clone), and
/// reads from the read end. `O_CLOEXEC` prevents the raw pipe fds
/// from leaking into any concurrently-spawned children on other
/// threads — without the flag, a race between `pipe()` and the
/// `Stdio::from()` consumption could let an unrelated `fork+exec`
/// inherit the write end and hold the reader open indefinitely.
/// One pipe, one reader — no threads, no channel, no chance of a
/// deadlock where reading stdout blocks while stderr fills its
/// buffer. Same merged-stream semantics that `sh -c "make … 2>&1"`
/// gives, without the shell-out.
///
/// When a spinner is active, each line is printed via `println()`
/// so the spinner redraws below the output. When no spinner,
/// output is captured and shown only on failure.
///
/// Pipe-read I/O errors propagate via `Err` rather than silently
/// ending the read loop. The prior line-iterator formulation
/// (`.lines()` + `Result::ok`) dropped every error-tagged item —
/// a mid-stream read failure just looked like EOF and the child's
/// tail output disappeared without a diagnostic. The byte-oriented
/// [`drain_lines_lossy`] now surfaces such failures with `anyhow`
/// context naming the merged-stream read, so a broken-pipe or EIO
/// during make's output is caught at the call site.
pub fn run_make_with_output(
    kernel_dir: &Path,
    args: &[&str],
    spinner: Option<&Spinner>,
) -> Result<()> {
    let (read_fd, write_fd) = nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC)
        .context("create pipe for merged make stdout+stderr")?;
    let write_fd_err = write_fd
        .try_clone()
        .context("clone pipe write end for stderr")?;

    let mut child = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .stdout(std::process::Stdio::from(write_fd))
        .stderr(std::process::Stdio::from(write_fd_err))
        .spawn()
        .with_context(|| format!("spawn make {}", args.join(" ")))?;

    // Parent has no remaining writer handles. `Stdio::from(OwnedFd)`
    // consumed `write_fd` and `write_fd_err` into the Command
    // builder; during `.spawn()` the builder installs them as the
    // child's stdout/stderr via `dup2`, then drops its own OwnedFd
    // copies. The child therefore holds the only live write ends
    // (its dup2'd stdout/stderr, fd 1/2). When `make` exits, those
    // fds are closed and the reader here sees EOF naturally.
    //
    // Read as bytes and convert each line via `from_utf8_lossy` at
    // the boundary. Compiler output can include non-UTF-8 bytes —
    // source paths on exotic filesystems, embedded binary fragments
    // from diagnostic tools, locale-encoded text — and a pure-String
    // reader would drop those lines via the `Result::ok` filter,
    // hiding real compiler errors in CI logs. Lossy conversion keeps
    // every line visible with U+FFFD where the bytes were not valid
    // UTF-8.
    let reader = std::io::BufReader::new(std::fs::File::from(read_fd));
    let captured = match drain_lines_lossy(reader, |line| {
        if let Some(sp) = spinner {
            sp.println(line);
        }
    }) {
        Ok(v) => v,
        Err(e) => {
            // On pipe-read I/O failure, kill and reap the child
            // before propagating so `make` doesn't linger as a
            // zombie — stdlib's Child does not auto-wait on drop.
            // Both ops use `.ok()` because the read-side error is
            // the actionable diagnostic; a secondary wait/kill
            // failure should not mask it.
            child.kill().ok();
            child.wait().ok();
            return Err(e).context("read merged make stdout+stderr");
        }
    };

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
/// Critical `.config` options checked by [`validate_kernel_config`].
///
/// Each entry pairs a `CONFIG_X` name with a diagnostic hint —
/// human-readable context on the dependency that typically causes the
/// option to be silently dropped during `make`. The list is curated:
/// every entry here is an option whose absence at the post-build
/// check has historically surfaced as a specific tool-install or
/// arch-default-override. The companion test
/// `critical_options_are_in_embedded_kconfig` proves every name is
/// present in [`EMBEDDED_KCONFIG`] as `=y`, so a kconfig edit that
/// removes a critical entry fails the test immediately instead of
/// surfacing later as a build that passes validation but behaves
/// differently.
const VALIDATE_CONFIG_CRITICAL: &[(&str, &str)] = &[
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

pub fn validate_kernel_config(kernel_dir: &std::path::Path) -> Result<()> {
    let config_path = kernel_dir.join(".config");
    let config = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;

    let mut missing = Vec::new();
    for &(option, hint) in VALIDATE_CONFIG_CRITICAL {
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
        Spinner::with_progress("Configuring kernel...", "Kernel configured", |_| {
            configure_kernel(source_dir, EMBEDDED_KCONFIG)
        })?;
    }

    Spinner::with_progress("Building kernel...", "Kernel built", |sp| {
        make_kernel_with_output(source_dir, Some(sp))
    })?;

    // Validate critical config options were not silently disabled.
    validate_kernel_config(source_dir)?;

    // Generate compile_commands.json for local trees (LSP support).
    if !acquired.is_temp {
        Spinner::with_progress(
            "Generating compile_commands.json...",
            "compile_commands.json generated",
            |sp| run_make_with_output(source_dir, &["compile_commands.json"], Some(sp)),
        )?;
    }

    // Find the built kernel image and vmlinux.
    let image_path = crate::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| anyhow::anyhow!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let vmlinux_ref = if vmlinux_path.exists() {
        let orig_mb = std::fs::metadata(&vmlinux_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        eprintln!("{cli_label}: caching vmlinux ({orig_mb:.0} MB, will be stripped)");
        Some(vmlinux_path.as_path())
    } else {
        eprintln!("{cli_label}: warning: vmlinux not found, BTF will not be cached");
        None
    };

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("{cli_label}: kernel built at {}", image_path.display());
        eprintln!(
            "{cli_label}: skipping cache — working tree has uncommitted changes; \
             commit or stash to enable caching"
        );
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
        acquired.kernel_source.clone(),
        arch.to_string(),
        image_name.to_string(),
        crate::test_support::now_iso8601(),
    )
    .with_version(acquired.version.clone())
    .with_config_hash(config_hash)
    .with_ktstr_kconfig_hash(Some(kconfig_hash));

    let mut artifacts = crate::cache::CacheArtifacts::new(&image_path);
    if let Some(v) = vmlinux_ref {
        artifacts = artifacts.with_vmlinux(v);
    }
    let entry = match cache.store(&acquired.cache_key, &artifacts, &metadata) {
        Ok(entry) => {
            success(&format!("\u{2713} Kernel cached: {}", acquired.cache_key));
            eprintln!("{cli_label}: image: {}", entry.image_path().display());
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
/// Returns `None` with a warning on stderr when no sidecars are found.
/// This is not an error -- regular test runs that skip gauntlet tests
/// produce no sidecar files.
pub fn print_stats_report() -> Option<String> {
    let dir = match std::env::var("KTSTR_SIDECAR_DIR") {
        Ok(d) if !d.is_empty() => Some(std::path::PathBuf::from(d)),
        _ => crate::test_support::newest_run_dir(),
    };
    let report = dir
        .as_deref()
        .map(|d| crate::test_support::analyze_sidecars(Some(d)))
        .filter(|r| !r.is_empty());
    if report.is_none() {
        eprintln!("cargo ktstr: no sidecar data found (skipped)");
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

/// Collect the current host context via
/// [`crate::host_context::collect_host_context`] and render it as
/// a human-readable multi-line report via
/// [`crate::host_context::HostContext::format_human`]. The output
/// ends with a newline; callers print it verbatim.
pub fn show_host() -> String {
    crate::host_context::collect_host_context().format_human()
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

/// List cgroup directories that `ktstr cleanup` / `cargo ktstr cleanup`
/// target by default: `/sys/fs/cgroup/ktstr` (test-harness parent) and
/// any `/sys/fs/cgroup/ktstr-<pid>` left behind by a `ktstr run` that
/// crashed or was SIGKILLed.
///
/// Returns only entries that exist and are directories. Silently
/// returns empty when `/sys/fs/cgroup` isn't a cgroup v2 mount. Skips
/// `ktstr-<pid>` directories whose pid still owns a live ktstr (or
/// cargo-ktstr) process, so a concurrent cleanup run doesn't rmdir an
/// active run's cgroup out from under it.
pub fn default_cleanup_parents() -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new("/sys/fs/cgroup");
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
pub fn is_ktstr_pid_alive(pid: &str) -> bool {
    let comm_path = format!("/proc/{pid}/comm");
    let Ok(comm) = std::fs::read_to_string(&comm_path) else {
        return false;
    };
    let comm = comm.trim();
    comm == "ktstr" || comm == "cargo-ktstr"
}

/// Reap leftover ktstr cgroup directories.
///
/// With `parent_cgroup` set, cleans only that path and leaves the
/// directory itself in place (matches `CgroupManager::cleanup_all`
/// semantics: purge children, keep parent). With `parent_cgroup` as
/// `None`, scans `/sys/fs/cgroup` for the default ktstr parents
/// reported by [`default_cleanup_parents`] and rmdirs each after
/// cleaning. Per-directory failures print to stderr and do not halt
/// the remaining sweep.
pub fn cleanup(parent_cgroup: Option<String>) -> Result<()> {
    use crate::cgroup::CgroupManager;

    match parent_cgroup {
        Some(path) => {
            if !std::path::Path::new(&path).exists() {
                bail!("cgroup path not found: {path}");
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
                // lookup() returns Some only for valid-metadata entries.
                return Ok(entry.path);
            }
            // Cache miss: download and build the requested version.
            download_and_cache_version(&resolved, cli_label)
        }
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new()?;
            if let Some(entry) = cache_lookup(&cache, key, cli_label) {
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
    let cache_key = acquired.cache_key.clone();

    // Clean trees: cache lookup before build.
    // Dirty trees: skip cache, always build.
    if !acquired.is_dirty
        && let Ok(cache) = crate::cache::CacheDir::new()
        && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label)
    {
        let image = entry.image_path();
        if image.exists() {
            success(&format!("{cli_label}: using cached kernel {cache_key}"));
            return Ok(image);
        }
    }

    let cache = crate::cache::CacheDir::new()?;
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true)?;

    // Prefer the cached image path (stable across rebuilds).
    match result.entry {
        Some(entry) => Ok(entry.image_path()),
        None => Ok(result.image_path),
    }
}

/// Whether stderr supports color (cached per process).
pub fn stderr_color() -> bool {
    use std::io::IsTerminal;
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| std::io::stderr().is_terminal())
}

/// Whether stdout supports color (cached per process). Distinct from
/// [`stderr_color`] because `cargo ktstr stats compare > report.txt`
/// pipes stdout to a file while leaving stderr on the TTY — gating
/// stdout tables on the stderr TTY state would leave ANSI escapes
/// in the file. Table-rendering code paths gate on this reading;
/// diagnostic/status prints use [`stderr_color`].
pub fn stdout_color() -> bool {
    use std::io::IsTerminal;
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| std::io::stdout().is_terminal())
}

/// Build a borderless comfy-table with styling gated on
/// [`stdout_color`]. When stdout is not a TTY (CI, piped-to-file),
/// `force_no_tty` suppresses cell color escapes so a log or grep
/// capture does not land raw `\x1b[...` sequences. The NOTHING preset
/// skips box-drawing characters and keeps whitespace-padded columns,
/// matching the previous hand-rolled `format!("{:<30}…")` look while
/// auto-measuring each column from actual cell contents.
pub fn new_table() -> comfy_table::Table {
    use comfy_table::{ContentArrangement, Table, presets::NOTHING};
    let mut t = Table::new();
    t.load_preset(NOTHING);
    t.set_content_arrangement(ContentArrangement::Disabled);
    if !stdout_color() {
        t.force_no_tty();
    }
    t
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
/// Call `finish` with a completion message to replace it with a
/// final line, or let it drop to remove it silently; [`Drop`] also
/// restores echo and clears the bar so a panic or early `?`
/// propagation leaves the terminal in a usable state. Note: Drop
/// does NOT run on SIGINT/SIGTERM kill; if the spinner is
/// interrupted mid-operation, run `stty sane` to restore echo.
pub struct Spinner {
    /// None when stderr is not a TTY — no indicatif overhead.
    pb: Option<indicatif::ProgressBar>,
    /// Saved termios for echo restore. None when stdin is not a tty
    /// or when the spinner is inactive (non-TTY stderr). Owned directly
    /// (not Arc<Mutex>) because Spinner is not Clone.
    saved_termios: Option<libc::termios>,
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

    fn disable_echo() -> Option<libc::termios> {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            return None;
        }
        unsafe {
            let fd = libc::STDIN_FILENO;
            let mut termios: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut termios) != 0 {
                return None;
            }
            let saved = termios;
            termios.c_lflag &= !libc::ECHO;
            libc::tcsetattr(fd, libc::TCSANOW, &termios);
            Some(saved)
        }
    }

    /// Restore stdin echo if we disabled it, consuming `saved_termios`
    /// via [`Option::take`]. Idempotent — `finish` and the `Drop`
    /// impl both call this; only the first call has any effect. The
    /// old standalone `clear` method was consolidated into `Drop`
    /// (calling `drop(spinner)` produces the same effect).
    fn teardown(&mut self) {
        if let Some(termios) = self.saved_termios.take() {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios);
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
    pub fn finish(mut self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.teardown();
        match self.pb.take() {
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

    /// Run `f` under a spinner that starts with `start_msg`, replaces
    /// itself with `success_msg` on `Ok`, and drops silently on `Err`
    /// so the error propagates without a stale progress bar obscuring
    /// the caller's diagnostics. The closure receives the live
    /// `&Spinner` so it can call [`Self::println`] / [`Self::suspend`]
    /// / [`Self::set_message`] during the operation.
    pub fn with_progress<T, E, F>(
        start_msg: impl Into<std::borrow::Cow<'static, str>>,
        success_msg: impl Into<std::borrow::Cow<'static, str>>,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(&Spinner) -> Result<T, E>,
    {
        let sp = Spinner::start(start_msg);
        let result = f(&sp);
        match result {
            Ok(v) => {
                sp.finish(success_msg);
                Ok(v)
            }
            Err(e) => {
                drop(sp);
                Err(e)
            }
        }
    }
}

impl Drop for Spinner {
    /// Restore terminal echo and clear any live progress bar on drop.
    ///
    /// [`finish`](Self::finish) calls [`Self::teardown`] and takes
    /// `self.pb` via [`Option::take`], so this impl is a no-op after
    /// an explicit end. When the spinner is dropped implicitly
    /// (panic, `?` propagation, `drop(sp)`, or scope exit), this
    /// restores the termios saved in [`Self::disable_echo`] and
    /// clears the live bar so stdin is usable afterwards.
    fn drop(&mut self) {
        self.teardown();
        if let Some(pb) = self.pb.take() {
            pb.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario;

    // -- show_host smoke --

    /// `show_host` must return a non-empty, newline-terminated
    /// human-readable report. On Linux, `uname_sysname` is
    /// populated from the `uname()` syscall regardless of `/proc`
    /// or `/sys` availability, so the output is guaranteed to
    /// contain that key. This also guards the
    /// [`print!("{}", cli::show_host())`] dispatch in
    /// `cargo-ktstr` — a future change that returns a
    /// trailing-newline-less string would drop the final line
    /// on the terminal.
    #[test]
    fn show_host_returns_populated_report() {
        let out = show_host();
        assert!(!out.is_empty(), "show_host must return non-empty output");
        assert!(
            out.ends_with('\n'),
            "show_host output must end with a newline for print! use: {out:?}",
        );
        assert!(
            out.contains("uname_sysname"),
            "show_host must surface the uname_sysname field: {out}",
        );
    }

    // -- Spinner Drop --

    #[test]
    fn spinner_drop_without_finish_does_not_panic_in_non_tty() {
        // Regression: Spinner previously had no Drop impl so early return
        // or panic leaked the disabled-ECHO termios. The added Drop must
        // run cleanly even on the non-TTY path (pb is None, saved_termios
        // is None) that nextest exercises under stderr capture.
        let sp = Spinner::start("test");
        drop(sp);
    }

    #[test]
    fn spinner_finish_then_drop_is_idempotent() {
        // finish() takes pb via Option::take so Drop's pb.take() sees None
        // and is a no-op on the progress bar side. teardown() is
        // idempotent because it consumes saved_termios via Option::take;
        // the second call finds None and does nothing. This test
        // exercises that lifecycle end-to-end.
        let sp = Spinner::start("test");
        sp.finish("done");
    }

    // -- drain_lines_lossy --

    #[test]
    fn drain_lines_lossy_eof_terminated_happy_path() {
        let input: &[u8] = b"alpha\nbeta\ngamma\n";
        let mut seen = Vec::new();
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |line| {
            seen.push(line.to_string())
        })
        .unwrap();
        assert_eq!(captured, vec!["alpha", "beta", "gamma"]);
        assert_eq!(seen, captured);
    }

    #[test]
    fn drain_lines_lossy_strips_crlf() {
        let input: &[u8] = b"one\r\ntwo\r\nthree\r\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["one", "two", "three"]);
    }

    #[test]
    fn drain_lines_lossy_non_utf8_bytes_survive_via_replacement() {
        // 0xFF is not valid UTF-8 in any position. `from_utf8_lossy`
        // replaces it with U+FFFD instead of dropping the line.
        let input: &[u8] = b"valid\n\xffbroken\ntail\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["valid", "\u{FFFD}broken", "tail"]);
    }

    #[test]
    fn drain_lines_lossy_empty_stream_yields_empty_vec() {
        let input: &[u8] = b"";
        let mut calls = 0usize;
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| calls += 1).unwrap();
        assert!(captured.is_empty());
        assert_eq!(calls, 0);
    }

    #[test]
    fn drain_lines_lossy_single_line_without_trailing_newline() {
        // Final chunk without a trailing newline should still be
        // emitted; BufRead::read_until returns the partial buffer
        // on EOF.
        let input: &[u8] = b"no-newline";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["no-newline"]);
    }

    #[test]
    fn drain_lines_lossy_lone_cr_at_eof_is_preserved() {
        // Bare CR without a following LF is NOT stripped — the CR
        // strip is nested inside the LF strip, so only `\r\n` is
        // normalized. A final chunk ending in `\r` keeps it.
        let input: &[u8] = b"foo\r";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["foo\r"]);
    }

    #[test]
    fn drain_lines_lossy_interior_cr_is_preserved() {
        // Only the trailing `\r` before `\n` is stripped; an
        // interior CR in the line body passes through verbatim.
        let input: &[u8] = b"ab\rcd\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["ab\rcd"]);
    }

    #[test]
    fn drain_lines_lossy_propagates_io_error_after_first_read() {
        use std::io::{BufReader, ErrorKind, Read};

        // Reader returns "line1\n" on the first read, then BrokenPipe
        // on the second. `drain_lines_lossy` must surface the Err —
        // the pre-refactor `.lines()` + `Result::ok` formulation
        // silently dropped errors and this test guards against that
        // regression.
        struct FlakyReader {
            calls: usize,
        }
        impl Read for FlakyReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                match self.calls {
                    1 => {
                        let data = b"line1\n";
                        let n = data.len().min(buf.len());
                        buf[..n].copy_from_slice(&data[..n]);
                        Ok(n)
                    }
                    _ => Err(std::io::Error::new(ErrorKind::BrokenPipe, "pipe closed")),
                }
            }
        }

        let err = drain_lines_lossy(BufReader::new(FlakyReader { calls: 0 }), |_| {})
            .expect_err("flaky reader must surface Err");
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    }

    #[test]
    fn drain_lines_lossy_mixed_lf_and_crlf() {
        // A single stream with both LF-only and CRLF line endings.
        // Each line is stripped independently: the CR strip is nested
        // inside the LF strip, so an LF-only line passes through
        // without CR stripping while a CRLF line loses the CR.
        let input: &[u8] = b"lf-line\ncrlf-line\r\nlf-again\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["lf-line", "crlf-line", "lf-again"]);
    }

    #[test]
    fn drain_lines_lossy_empty_lines_lf() {
        // A bare `\n` between two non-empty lines produces an empty
        // string in the captured Vec — after strip_suffix(b"\n")
        // the remaining slice is empty and from_utf8_lossy("") == "".
        let input: &[u8] = b"a\n\nb\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["a", "", "b"]);
    }

    #[test]
    fn drain_lines_lossy_empty_lines_crlf() {
        // A bare `\r\n` produces an empty string after both the LF
        // and the preceding CR are stripped.
        let input: &[u8] = b"\r\n\r\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["", ""]);
    }

    #[test]
    fn drain_lines_lossy_callback_fires_once_per_line_in_order() {
        // Pin the externally-observable callback contract: `on_line`
        // is invoked exactly once per emitted line, in the same order
        // the lines appear in the returned Vec. Each invocation
        // records the count of prior invocations, yielding [0, 1, 2]
        // across three lines — proving once-per-line invocation and
        // stable ordering.
        let input: &[u8] = b"a\nb\nc\n";
        let lens = std::cell::RefCell::new(Vec::<usize>::new());
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_line| {
            let mut v = lens.borrow_mut();
            let current = v.len();
            v.push(current);
        })
        .unwrap();
        assert_eq!(captured, vec!["a", "b", "c"]);
        assert_eq!(lens.into_inner(), vec![0, 1, 2]);
    }

    // -- run_make_with_output --

    /// `Command::current_dir` on a non-existent path causes
    /// `Command::spawn` to fail before exec, with an underlying
    /// `io::Error` of kind `NotFound`. `run_make_with_output` wraps
    /// that error via `.with_context(|| format!("spawn make {}", ...))`,
    /// so the rendered anyhow chain must surface BOTH the
    /// `"spawn make <args>"` annotation (proving the wrapping landed
    /// on the failing operation) AND the underlying
    /// `"No such file or directory"` message (proving the io::Error
    /// chain was preserved through context). A regression that
    /// dropped either layer — bare `?` losing the context, or
    /// `Error::msg` losing the source — would surface here. Pipe2 +
    /// try_clone run before spawn, so reaching this error proves
    /// they succeeded too.
    #[test]
    fn run_make_with_output_surfaces_actionable_error_when_kernel_dir_missing() {
        let missing = std::path::Path::new("/this/path/should/not/exist/ktstr_test");
        let err = run_make_with_output(missing, &["foo"], None)
            .expect_err("nonexistent kernel_dir must surface a spawn failure");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("spawn make foo"),
            "expected `spawn make foo` context layer, got: {rendered}"
        );
        assert!(
            rendered.contains("No such file or directory"),
            "expected underlying io::Error chain, got: {rendered}"
        );
    }

    /// End-to-end exercise of the merged-pipe path against a real
    /// `make` invocation that emits to BOTH stdout and stderr in
    /// large enough volume to fill a 64 KiB pipe buffer (Linux
    /// default), then exits non-zero. Two invariants:
    ///
    /// 1. **No-deadlock.** The production code creates ONE pipe
    ///    shared between stdout and stderr and reads it with a
    ///    single BufReader (no threads, no select). If the merge
    ///    were broken — e.g. if stderr were left attached to the
    ///    inherited fd 2 instead of `try_clone`'d onto the pipe
    ///    write end — high-volume stderr writes would still complete
    ///    via the inherited fd, but a regression that wired stderr
    ///    to a SECOND independent pipe with no reader would hang
    ///    the child after the first ~64 KiB of stderr writes. The
    ///    pipe-buffer-overflow lines below force that scenario; if
    ///    the test completes, no-deadlock holds.
    ///
    /// 2. **Failure-path Err.** A non-zero exit must surface as
    ///    `Err` with the `"make ... failed"` wording from the
    ///    `bail!` at the end of `run_make_with_output`. A regression
    ///    that swallowed the exit status or routed it through `Ok`
    ///    would hide compiler errors in CI logs.
    ///
    /// Skipped (passes silently) when `make` is not on PATH so the
    /// test suite stays runnable on minimal CI containers without
    /// build tools. The companion
    /// [`run_make_with_output_surfaces_actionable_error_when_kernel_dir_missing`]
    /// test exercises the spawn-failure path without needing make.
    #[test]
    fn run_make_with_output_drains_high_volume_failing_make_without_deadlock() {
        if resolve_in_path(std::path::Path::new("make")).is_none() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        // Default target prints 1 KiB to stdout and 1 KiB to stderr
        // for 100 iterations (~200 KiB total — well past the Linux
        // 64 KiB pipe buffer), then `false` exits 1. Recipe lines
        // MUST start with a TAB, not spaces — make rejects spaces.
        // `printf` with a 1 KiB byte string per call avoids relying
        // on a specific shell loop construct; the Makefile uses
        // make's own iteration via repetition.
        let stdout_chunk: String = "S".repeat(1024);
        let stderr_chunk: String = "E".repeat(1024);
        let mut recipe = String::new();
        for _ in 0..100 {
            recipe.push_str(&format!("\t@printf '%s\\n' '{stdout_chunk}'\n"));
            recipe.push_str(&format!("\t@printf '%s\\n' '{stderr_chunk}' >&2\n"));
        }
        let makefile = format!("default:\n{recipe}\t@false\n");
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();
        // Passing `default` explicitly anchors the error message
        // wording to `"make default failed"` rather than the
        // double-space `"make  failed"` form that `&[]` produces.
        let err = run_make_with_output(dir.path(), &["default"], None)
            .expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording from bail!, got: {rendered}"
        );
    }

    /// Redirect the test process's `stderr` (fd 2) to a tempfile for
    /// the duration of `f`, then restore. Returns the bytes written
    /// to fd 2 during the call. Used by tests that need to observe
    /// what the production code emitted via `eprintln!` — there is
    /// no in-band way to capture that without process-level fd
    /// manipulation because `eprintln!` writes straight through to
    /// fd 2.
    ///
    /// Uses [`nix::unistd::dup`] (returns an `OwnedFd` for the
    /// saved-stderr handle) and [`nix::unistd::dup2_stderr`] (the
    /// nix 0.31 purpose-built wrapper for redirecting fd 2); the
    /// latter sidesteps the generic `dup2`'s `&mut OwnedFd` newfd
    /// requirement — which would be awkward here, because fd 2 is
    /// not an `OwnedFd` we can lawfully construct. Dropping `saved`
    /// at scope exit closes it; fd 2 retains its own kernel-level
    /// reference to the original stderr open file description.
    ///
    /// Test-only utility — no production caller. Lives in this test
    /// module so its scope is bounded.
    fn capture_test_stderr<R>(f: impl FnOnce() -> R) -> (R, Vec<u8>) {
        use nix::unistd::{dup, dup2_stderr};
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut sink = tempfile::tempfile().expect("create stderr-capture tempfile");
        // Flush any pending stderr buffering before redirect. eprintln!
        // is line-buffered to a Stderr lock; flush ensures pre-call
        // output reaches the original fd 2 instead of leaking into
        // the captured tempfile.
        std::io::stderr().flush().ok();
        let saved = dup(std::io::stderr()).expect("dup(stderr) failed");
        dup2_stderr(&sink).expect("dup2_stderr(sink) failed");
        let result = f();
        // Flush before restore so all of f's writes land in the sink.
        std::io::stderr().flush().ok();
        dup2_stderr(&saved).expect("dup2_stderr(saved) failed");
        sink.seek(SeekFrom::Start(0)).expect("rewind sink");
        let mut bytes = Vec::new();
        sink.read_to_end(&mut bytes).expect("read sink");
        (result, bytes)
    }

    /// Direct proof of the merge: emit a unique marker on stdout and
    /// a different unique marker on stderr from a child process,
    /// then exit non-zero so `run_make_with_output` enters the
    /// failure branch and `eprintln!`s every captured line. The
    /// captured Vec is internal to the function, but the production
    /// code's failure-path `eprintln!` loop (lines 521-525) dumps
    /// every captured line to fd 2 before the `bail!` fires. Capture
    /// fd 2 around the call via [`capture_test_stderr`] and assert
    /// BOTH markers appear in the output.
    ///
    /// If the merge were broken (e.g. stderr installed on a
    /// separate pipe with no reader, or left attached to the
    /// parent's fd 2), the stderr marker would either deadlock the
    /// child or end up on the test process's original stderr — NOT
    /// in the captured Vec, NOT re-emitted by the failure-branch
    /// eprintln loop, NOT in the bytes this test reads from the
    /// sink. Asserting both markers appear is the empirical merge
    /// proof that complements the no-deadlock invariant pinned by
    /// `run_make_with_output_drains_high_volume_failing_make_without_deadlock`.
    ///
    /// Skipped when `make` is not on PATH for the same reason the
    /// high-volume test is.
    #[test]
    fn run_make_with_output_merges_stderr_into_captured_output() {
        if resolve_in_path(std::path::Path::new("make")).is_none() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        // Two distinguishable markers so the assertion can detect a
        // half-merge regression where only one stream made it through.
        let stdout_marker = "KTSTR_STDOUT_MARKER_e7f9";
        let stderr_marker = "KTSTR_STDERR_MARKER_a1b2";
        let makefile = format!(
            "default:\n\
             \t@printf '%s\\n' '{stdout_marker}'\n\
             \t@printf '%s\\n' '{stderr_marker}' >&2\n\
             \t@false\n"
        );
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();

        let (result, captured_bytes) =
            capture_test_stderr(|| run_make_with_output(dir.path(), &["default"], None));
        let err = result.expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording, got: {rendered}"
        );
        let captured = String::from_utf8_lossy(&captured_bytes);
        assert!(
            captured.contains(stdout_marker),
            "stdout marker missing from captured output (eprintln'd via failure path) — \
             expected `{stdout_marker}` in: {captured:?}"
        );
        assert!(
            captured.contains(stderr_marker),
            "stderr marker missing from captured output — proves the merge is BROKEN: \
             stderr did not reach the captured Vec. expected `{stderr_marker}` in: {captured:?}"
        );
    }

    /// Stderr-only high-volume burst: emit ~128 KiB to stderr alone
    /// (no interleaved stdout writes), then exit non-zero. This
    /// isolates the stderr-merge invariant from the stdout path.
    /// A regression that wired stderr to a separate unread pipe
    /// would deadlock the child after the first ~64 KiB (Linux
    /// default pipe buffer) since no reader exists on the broken
    /// stderr pipe. 128 KiB is double the buffer — definitely
    /// triggers the deadlock condition. Distinct from the
    /// interleaved high-volume test in
    /// [`run_make_with_output_drains_high_volume_failing_make_without_deadlock`]:
    /// that test interleaves so partial-merge regressions could
    /// "look" like they work because alternating stdout drains the
    /// pipe between stderr writes; this test forces stderr to drain
    /// alone. Test completion = no-deadlock pass.
    ///
    /// Skipped when `make` is not on PATH.
    #[test]
    fn run_make_with_output_drains_stderr_only_high_volume_without_deadlock() {
        if resolve_in_path(std::path::Path::new("make")).is_none() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        // 128 iterations * 1 KiB = 128 KiB of stderr — 2x the default
        // 64 KiB pipe buffer. No stdout writes at all so the buffer
        // can only drain via the merged-pipe reader.
        let chunk: String = "X".repeat(1024);
        let mut recipe = String::new();
        for _ in 0..128 {
            recipe.push_str(&format!("\t@printf '%s\\n' '{chunk}' >&2\n"));
        }
        let makefile = format!("default:\n{recipe}\t@false\n");
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();
        let err = run_make_with_output(dir.path(), &["default"], None)
            .expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording, got: {rendered}"
        );
    }

    /// Spawn-failure path must not leak the pipe2 OwnedFds. Three
    /// fds are allocated before spawn: `read_fd` (pipe2 read end),
    /// `write_fd` (pipe2 write end, consumed by `Stdio::from`), and
    /// `write_fd_err` (`try_clone` of write_fd). When spawn fails:
    /// - `write_fd` is owned by the Command builder, which is
    ///   dropped on the early-return path.
    /// - `write_fd_err` is still in scope as an `OwnedFd` and drops
    ///   when the function returns.
    /// - `read_fd` is still in scope and drops on return.
    ///
    /// All three should release via OwnedFd's Drop (which calls
    /// `close()` on the inner fd). Count `/proc/self/fd` entries
    /// before and after a guaranteed-spawn-failure call; the count
    /// must not increase. A regression that switched to raw fd
    /// integers (no Drop) or that consumed the write_fd via a path
    /// other than Stdio::from (leaving it dangling on early return)
    /// would surface here as a leak of 1-3 fds per call.
    ///
    /// Linux-only: relies on `/proc/self/fd` enumeration. Skipped
    /// silently when /proc isn't a procfs mount.
    #[test]
    fn run_make_with_output_releases_fds_on_spawn_failure() {
        let proc_fd = std::path::Path::new("/proc/self/fd");
        if !proc_fd.is_dir() {
            eprintln!(
                "skipping run_make_with_output_releases_fds_on_spawn_failure: \
                 /proc/self/fd not available"
            );
            return;
        }
        let count_fds = || -> usize {
            std::fs::read_dir(proc_fd)
                .expect("read /proc/self/fd")
                .filter_map(|e| e.ok())
                .count()
        };
        // Warm-up pass: the first call may allocate process-wide
        // resources (thread-local buffers, lazy_static-like state in
        // the std/anyhow paths) that look like fd growth on a single
        // before/after measurement. Run once outside the measurement
        // window so steady-state fd usage is what we sample.
        let missing = std::path::Path::new("/this/path/should/not/exist/ktstr_test_fdcheck");
        let _ = run_make_with_output(missing, &["foo"], None);
        let before = count_fds();
        // Run the failing path several times — a per-call leak of
        // even one fd would compound here and surface as a clear
        // delta. A single-call measurement could miss small leaks
        // masked by transient fd churn from the test runtime.
        for _ in 0..16 {
            let _ = run_make_with_output(missing, &["foo"], None);
        }
        let after = count_fds();
        assert!(
            after <= before,
            "fd leak on spawn failure: {before} -> {after} (16 calls, expected no growth)"
        );
    }

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

    // days_to_ymd tests moved to test_support::timefmt tests
    // (days_to_ymd_2024_jan_1, _2024_leap_day, _2023_end_of_year)
    // since the single implementation now lives there.

    // -- validate_kernel_config --

    /// Every entry in `VALIDATE_CONFIG_CRITICAL` must appear as `=y`
    /// in the embedded kconfig fragment. If a critical option is
    /// dropped from the fragment, builds would skip the option but
    /// validation would keep flagging it as missing — the user sees
    /// a build failure that no amount of tool installation fixes.
    /// This test catches the drift at compile-test time.
    #[test]
    fn critical_options_are_in_embedded_kconfig() {
        let fragment = crate::EMBEDDED_KCONFIG;
        for &(option, _) in VALIDATE_CONFIG_CRITICAL {
            let enabled = format!("{option}=y");
            assert!(
                fragment.lines().any(|l| l.trim() == enabled),
                "VALIDATE_CONFIG_CRITICAL lists {option:?} but ktstr.kconfig does not \
                 enable it; either add `{option}=y` to the fragment or drop the entry \
                 from VALIDATE_CONFIG_CRITICAL",
            );
        }
    }

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

    #[test]
    fn configure_kernel_rejects_numeric_prefix_false_match() {
        // Fragment asks `CONFIG_NR_CPUS=1`, .config has
        // `CONFIG_NR_CPUS=128`. A plain
        // `config_content.contains(fragment_line)` would treat the
        // substring "CONFIG_NR_CPUS=1" as present inside
        // "CONFIG_NR_CPUS=128" (numeric prefix) and skip the append.
        // Exact-line matching via the HashSet helper correctly
        // distinguishes the two and appends.
        let dir = tempfile::TempDir::new().unwrap();
        let initial = "CONFIG_NR_CPUS=128\n";
        std::fs::write(dir.path().join(".config"), initial).unwrap();
        std::fs::write(dir.path().join("Makefile"), "olddefconfig:\n\t@true\n").unwrap();
        let fragment = "CONFIG_NR_CPUS=1\n";
        configure_kernel(dir.path(), fragment).unwrap();
        let config = std::fs::read_to_string(dir.path().join(".config")).unwrap();
        assert!(
            config.lines().any(|l| l.trim() == "CONFIG_NR_CPUS=1"),
            "CONFIG_NR_CPUS=1 must be appended as its own line: {config:?}"
        );
        assert!(
            config.lines().any(|l| l.trim() == "CONFIG_NR_CPUS=128"),
            "original CONFIG_NR_CPUS=128 must be preserved: {config:?}"
        );
    }

    // -- all_fragment_lines_present pure helper --

    #[test]
    fn all_fragment_lines_present_exact_match() {
        let config = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        assert!(all_fragment_lines_present("CONFIG_FOO=y\n", config));
        assert!(all_fragment_lines_present("CONFIG_BAR=m\n", config));
        assert!(all_fragment_lines_present(
            "CONFIG_FOO=y\nCONFIG_BAR=m\n",
            config
        ));
    }

    #[test]
    fn all_fragment_lines_present_numeric_prefix_not_present() {
        // The bug case. Substring match would incorrectly report present.
        let config = "CONFIG_NR_CPUS=128\n";
        assert!(!all_fragment_lines_present("CONFIG_NR_CPUS=1\n", config));
        assert!(!all_fragment_lines_present("CONFIG_NR_CPUS=12\n", config));
    }

    #[test]
    fn all_fragment_lines_present_disable_directive_participates() {
        // `# CONFIG_X is not set` is a real kconfig semantic (disable),
        // not a comment to be skipped. It must participate in the check.
        let config = "CONFIG_BPF=y\n";
        // Fragment disables CONFIG_BPF via the standard kconfig comment
        // syntax. Since .config has it enabled, the disable line is
        // NOT present and the helper must return false.
        assert!(!all_fragment_lines_present(
            "# CONFIG_BPF is not set\n",
            config
        ));
    }

    #[test]
    fn all_fragment_lines_present_empty_lines_skipped() {
        // Truly empty lines in the fragment carry no kconfig state.
        let config = "CONFIG_FOO=y\n";
        assert!(all_fragment_lines_present("\n\nCONFIG_FOO=y\n\n", config));
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

    // -- kernel_list JSON/human parity --

    /// Pin the [`format_entry_row`] staleness mapping: the human
    /// `(stale kconfig)` tag appears iff `CacheEntry::kconfig_status`
    /// returns `KconfigStatus::Stale`. The test exercises every
    /// `KconfigStatus` variant (Matches, Stale, Untracked), so a
    /// regression that tightened the variant-to-tag mapping — e.g.
    /// surfacing Untracked as stale or dropping the Stale branch —
    /// surfaces as a tag/status disagreement.
    ///
    /// `kernel list --json` emits `kconfig_status` as a 3-value
    /// string (`"matches"` / `"stale"` / `"untracked"`) via
    /// `CacheEntry::kconfig_status(...).to_string()` at the
    /// JSON-branch call site — NOT a `stale_kconfig` boolean. The
    /// "JSON/human parity" phrasing in the test name refers to the
    /// shared `kconfig_status` gate both branches key off.
    ///
    /// The current body only evaluates the human branch against the
    /// `kconfig_status` method return; it does not exercise the
    /// JSON emission path, so a regression that broke the
    /// JSON-branch string serialization would slip through this
    /// test.
    #[test]
    fn kernel_list_stale_kconfig_json_human_parity() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        fn metadata_with_hash(hash: Option<&str>) -> KernelMetadata {
            KernelMetadata::new(
                KernelSource::Tarball,
                "x86_64".to_string(),
                "bzImage".to_string(),
                "2026-04-12T10:00:00Z".to_string(),
            )
            .with_version(Some("6.14.2".to_string()))
            .with_ktstr_kconfig_hash(hash.map(str::to_string))
        }

        // (case label, entry's recorded ktstr_kconfig_hash, caller's
        // current hash). These cover every KconfigStatus variant:
        // Matches, Stale, Untracked.
        let cases: &[(&str, Option<&str>, &str)] = &[
            ("matches", Some("same"), "same"),
            ("stale", Some("old"), "new"),
            ("untracked", None, "anything"),
        ];

        for &(label, entry_hash, current_hash) in cases {
            let tmp = tempfile::TempDir::new().unwrap();
            let cache = CacheDir::with_root(tmp.path().join("cache"));
            let src = tempfile::TempDir::new().unwrap();
            let image = src.path().join("bzImage");
            std::fs::write(&image, b"fake kernel").unwrap();
            let meta = metadata_with_hash(entry_hash);
            let entry = cache
                .store(label, &CacheArtifacts::new(&image), &meta)
                .unwrap();

            let json_stale = entry.kconfig_status(current_hash).is_stale();

            // Human branch: format_entry_row emits "(stale kconfig)"
            // iff kconfig_status returns Stale.
            let human_row = format_entry_row(&entry, current_hash, &[]);
            let human_stale = human_row.contains("stale kconfig");

            assert_eq!(
                json_stale, human_stale,
                "kernel_list JSON/human stale-kconfig disagreement on `{label}` \
                 (entry_hash={entry_hash:?}, current_hash={current_hash:?}); \
                 json_stale={json_stale}, human_row={human_row:?}"
            );
        }
    }

    // -- is_eol predicate --
    //
    // Pure function, no env / fixtures. Pins every return branch
    // documented on `is_eol`: the empty-slice guard, the
    // prefix-in-list branch, the prefix-absent branch, and the
    // unparseable-prefix branch.

    /// Empty `active_prefixes` is the "active list unknown" signal
    /// (fetch failure, skipped lookup). The empty-slice guard must
    /// return false so the `kernel list --json` contract holds:
    /// releases.json failure means no entry is tagged EOL. Without
    /// the guard, `!any(..)` on an empty iterator is `true` and the
    /// predicate would flip to tagging every entry EOL — the exact
    /// opposite of the contract.
    #[test]
    fn is_eol_empty_active_prefixes_returns_false() {
        assert!(!is_eol("6.14.2", &[]));
    }

    /// Happy path for an active series: the major.minor prefix
    /// (`6.14`) appears in the supplied `active_prefixes` list, so
    /// the any-match arm fires and the overall predicate returns
    /// false (not EOL).
    #[test]
    fn is_eol_prefix_in_active_list_returns_false() {
        assert!(!is_eol("6.14.2", &["6.14".to_string()]));
    }

    /// The failure path the predicate exists to detect: the
    /// version's `5.10` prefix is absent from a non-empty active
    /// list, so `!any(..)` fires and the predicate returns true.
    /// Sanity-checks the only code path that produces `true` in the
    /// current implementation.
    #[test]
    fn is_eol_prefix_absent_from_active_list_returns_true() {
        assert!(is_eol(
            "5.10.200",
            &["6.14".to_string(), "6.12".to_string()],
        ));
    }

    /// A version string with no parseable major.minor prefix (e.g.
    /// a cache key or freeform identifier) short-circuits via
    /// `version_prefix` and returns false. Distinct from the
    /// empty-slice branch above: the active list is non-empty here,
    /// so reaching false requires the prefix-absent short-circuit to
    /// fire.
    #[test]
    fn is_eol_unparseable_version_returns_false() {
        assert!(!is_eol("abc", &["6.14".to_string()]));
    }

    /// Snapshot pin for `format_entry_row` across the 6-case outcome
    /// matrix over (EOL, not-EOL) × (Matches, Stale, Untracked);
    /// empty and unparseable `active_prefixes` branches are pinned by
    /// sibling `is_eol_` tests. A 7th case fixes the `version == "-"`
    /// short-circuit at cli.rs where a missing version skips the EOL
    /// tag even under a non-empty active list.
    ///
    /// Inline snapshot captures exact padding and tag ordering so any
    /// drift — column width change, tag reorder, `(EOL)` string
    /// rename, Display-impl tweak on `KconfigStatus` — fails this one
    /// test. Uses `KernelSource::Tarball` because it is the simplest
    /// variant to construct; `Display` on `KernelSource` strips
    /// payload fields for every variant, so source choice only
    /// affects the rendered column when the Display impl changes.
    /// Fixed `built_at` timestamp keeps the snapshot date-stable.
    /// Key lengths vary (12-20 chars) but all fit within the 48-char
    /// column, so padding drift surfaces at multiple pad counts
    /// across c1-c7.
    #[test]
    fn format_entry_row_renders_eol_kconfig_matrix() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = tmp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let image = src_dir.join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();

        let current_hash = "a1b2c3d4";
        let active_prefixes = ["6.14".to_string()];

        // version "6.14.2" is in active list → not EOL; version
        // "2.6.32" is long-EOL; version None → rendered as "-" and
        // short-circuits the EOL guard regardless of active list.
        // entry_hash == current → Matches; entry_hash != current →
        // Stale; None → Untracked.
        let build_row =
            |key: &str, version: Option<&str>, entry_hash: Option<&str>| -> String {
                let meta = KernelMetadata::new(
                    KernelSource::Tarball,
                    "x86_64".to_string(),
                    "bzImage".to_string(),
                    "2026-04-12T10:00:00Z".to_string(),
                )
                .with_version(version.map(str::to_string))
                .with_ktstr_kconfig_hash(entry_hash.map(str::to_string));
                let entry = cache
                    .store(key, &CacheArtifacts::new(&image), &meta)
                    .unwrap();
                format_entry_row(&entry, current_hash, &active_prefixes)
            };

        let rows = [
            build_row("c1-active-matches", Some("6.14.2"), Some(current_hash)),
            build_row("c2-active-stale", Some("6.14.2"), Some("deadbeef")),
            build_row("c3-active-untracked", Some("6.14.2"), None),
            build_row("c4-eol-matches", Some("2.6.32"), Some(current_hash)),
            build_row("c5-eol-stale", Some("2.6.32"), Some("deadbeef")),
            build_row("c6-eol-untracked", Some("2.6.32"), None),
            build_row("c7-active-no-version", None, Some(current_hash)),
        ];
        let joined = rows.join("\n");
        insta::assert_snapshot!(joined, @r"
          c1-active-matches                                6.14.2       tarball  x86_64  2026-04-12T10:00:00Z
          c2-active-stale                                  6.14.2       tarball  x86_64  2026-04-12T10:00:00Z (stale kconfig)
          c3-active-untracked                              6.14.2       tarball  x86_64  2026-04-12T10:00:00Z (untracked kconfig)
          c4-eol-matches                                   2.6.32       tarball  x86_64  2026-04-12T10:00:00Z (EOL)
          c5-eol-stale                                     2.6.32       tarball  x86_64  2026-04-12T10:00:00Z (stale kconfig) (EOL)
          c6-eol-untracked                                 2.6.32       tarball  x86_64  2026-04-12T10:00:00Z (untracked kconfig) (EOL)
          c7-active-no-version                             -            tarball  x86_64  2026-04-12T10:00:00Z
        ");
    }

    /// Regression pin for `format_entry_row` with empty
    /// `active_prefixes` — the fallback path `kernel_list` enters
    /// when [`fetch_active_prefixes`] returns `Err`. The `(EOL)` tag
    /// must not appear on the rendered row regardless of how old the
    /// entry's version is, since "fetch failed" is an
    /// unknown-active-list signal, not a universal-EOL signal.
    /// Cross-checked against the non-empty branch so the suppression
    /// is owned by the empty-slice fallback, not by some other code
    /// path that happens to be quiet on this fixture.
    #[test]
    fn format_entry_row_empty_active_prefixes_does_not_tag_eol() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();
        // Ancient long-EOL version: if the active list contained only
        // modern series, is_eol would return true. The empty slice
        // forces the guard branch instead.
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("2.6.32".to_string()));
        let entry = cache
            .store(
                "fetch-failed-fallback",
                &CacheArtifacts::new(&image),
                &meta,
            )
            .unwrap();

        let row_fallback = format_entry_row(&entry, "kconfig_hash", &[]);
        assert!(
            !row_fallback.contains("(EOL)"),
            "empty active_prefixes (fetch-failed fallback) must not tag any entry EOL, \
             got row: {row_fallback:?}",
        );

        // Sanity: the same entry IS tagged EOL when a non-empty active
        // list excludes its prefix. Confirms the suppression above
        // flows through the empty-slice guard, not some unrelated
        // short-circuit.
        let row_with_active = format_entry_row(&entry, "kconfig_hash", &["6.14".to_string()]);
        assert!(
            row_with_active.contains("(EOL)"),
            "non-empty active_prefixes excluding entry's prefix must tag EOL, \
             got row: {row_with_active:?}",
        );
    }
}
