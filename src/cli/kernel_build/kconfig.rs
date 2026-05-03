//! kconfig fragment merging, validation, and post-build probing.
//!
//! Holds the helpers that compose, parse, and validate the kernel
//! configuration:
//! - [`configure_kernel`] runs `make defconfig` (when no `.config`
//!   exists), checks the merged fragment is present, appends and
//!   re-runs `olddefconfig` only when needed.
//! - [`all_fragment_lines_present`] / [`is_kconfig_semantic_line`]
//!   gate the no-op short-circuit so a clean `.config` doesn't
//!   churn the configure pass.
//! - [`read_extra_kconfig`] parses the user's `--extra-kconfig PATH`
//!   into the fragment string with actionable error wording per
//!   I/O failure mode.
//! - [`append_extra_kconfig_suffix`] composes the cache-key suffix
//!   so the extras-aware build lands at a distinct cache slot.
//! - [`warn_extra_kconfig_overrides_baked_in`] /
//!   [`warn_dropped_extra_kconfig_lines`] surface user/baked-in
//!   conflicts and silent olddefconfig drops respectively.
//! - [`parse_kconfig_symbol`] / [`render_kconfig_value`] are the
//!   shared primitives behind those warning passes.
//! - [`validate_kernel_config`] is the post-build critical-option
//!   pass; [`has_sched_ext`] is the short-circuit probe driving
//!   `kernel_build_pipeline`'s configure-skip.

use std::path::Path;

use anyhow::{Context, Result, bail};

use super::super::kernel_cmd::EMBEDDED_KCONFIG;
use super::make::run_make;

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
/// check.
///
/// Free-text comments (`#` lines that are NOT the
/// `# CONFIG_X is not set` form, e.g. `# Build for testing scx`)
/// are dropped before the probe: kbuild emits its own header into
/// `.config` and strips user-added free-text comments, so a fragment
/// containing decorative `#` lines would always fail this check and
/// trigger a redundant `make olddefconfig` re-resolve on every
/// configure pass.
///
/// Genuinely empty lines are also skipped.
pub(super) fn all_fragment_lines_present(fragment: &str, config: &str) -> bool {
    let existing: std::collections::HashSet<&str> = config.lines().map(str::trim).collect();
    fragment
        .lines()
        .map(str::trim)
        .filter(|t| is_kconfig_semantic_line(t))
        .all(|t| existing.contains(t))
}

/// True for lines that participate in kconfig semantics — i.e.
/// `CONFIG_X=...` assignments and the `# CONFIG_X is not set`
/// disable-directive form. Empty lines and free-text `#` comments
/// return false.
///
/// Drives [`all_fragment_lines_present`]'s filter so a fragment with
/// decorative comments doesn't churn the configure pass on every
/// rebuild. The disable-directive form is a kconfig sentinel kbuild
/// emits into `.config` as the canonical way to record a disabled
/// `tristate`/`bool` symbol; it survives `make olddefconfig` and
/// must participate in the present-in-config check.
pub(super) fn is_kconfig_semantic_line(trimmed: &str) -> bool {
    if trimmed.is_empty() {
        return false;
    }
    if let Some(rest) = trimmed.strip_prefix('#') {
        // `# CONFIG_X is not set` — the disable-directive sentinel
        // kbuild writes verbatim into .config. Tolerant of internal
        // whitespace variation by trimming the rest.
        let rest = rest.trim_start();
        return rest.starts_with("CONFIG_") && rest.ends_with(" is not set");
    }
    // Non-comment, non-empty line: a CONFIG_X=... assignment or
    // similar. The probe defers semantic validity to the existing
    // hash-set match against the on-disk `.config`.
    true
}

/// Read a `--extra-kconfig PATH` file. Returns `Ok(content)` on
/// success or `Err(message)` with an actionable diagnostic naming
/// `--extra-kconfig` and the user's literal input path verbatim so
/// a typo names the exact string they passed.
///
/// Four distinguishing arms each produce an actionable message:
/// - `ENOENT` (not-found) → tells the operator to verify the path
///   spelling and that the file exists
/// - `EISDIR` (is-a-directory) → tells the operator to pass a regular
///   file rather than a directory
/// - `EACCES` (permission-denied) → tells the operator to check file
///   ownership and mode
/// - empty file (zero-byte read success) → emits a `tracing::warn!`
///   explaining the cache-slot consequence and pointing at the likely
///   operator intent (a non-empty fragment), then proceeds
///
/// Other I/O errors fall through with the OS error rendered verbatim
/// (`--extra-kconfig {path}: {os error}`). A non-UTF-8 file errors
/// with a message identifying the constraint (kconfig fragments are
/// ASCII text).
///
/// Symlinks resolve transparently (`std::fs::read` opens through
/// `open(2)` which follows symlinks per kernel default).
pub fn read_extra_kconfig(path: &Path, cli_label: &str) -> std::result::Result<String, String> {
    let display = path.display();
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            let msg = match e.kind() {
                std::io::ErrorKind::NotFound => {
                    format!(
                        "--extra-kconfig {display}: file not found; check the \
                         path spelling and that the file exists"
                    )
                }
                std::io::ErrorKind::IsADirectory => {
                    format!(
                        "--extra-kconfig {display}: is a directory; pass a \
                         regular file containing kconfig fragment lines"
                    )
                }
                std::io::ErrorKind::PermissionDenied => {
                    format!(
                        "--extra-kconfig {display}: permission denied; check \
                         file ownership and mode (the kconfig fragment must \
                         be readable by the current user)"
                    )
                }
                _ => format!("--extra-kconfig {display}: {e}"),
            };
            return Err(msg);
        }
    };

    // Empty file: warn but proceed. The build still lands at a
    // distinct cache slot via the `-xkc{hash_of_empty}` segment, but
    // no user symbols merge — the operator likely meant to populate
    // the file with `CONFIG_X=...` lines.
    if bytes.is_empty() {
        let path_str = display.to_string();
        tracing::warn!(
            cli_label = cli_label,
            path = %path_str,
            "--extra-kconfig file is empty; the build will land at a \
             distinct cache slot but no user symbols will merge into the \
             configuration. Did you mean to populate {path_str} with \
             CONFIG_X=... lines?",
        );
    }

    String::from_utf8(bytes).map_err(|_| {
        format!(
            "--extra-kconfig {display}: file is not valid UTF-8; kconfig \
             fragments must be ASCII text"
        )
    })
}

/// Append the `-xkc{extra_hash}` segment to a cache key built around
/// the bare baked-in suffix (`...-kc{baked_hash}`), bringing it to the
/// two-segment shape produced by [`crate::cache_key_suffix_with_extra`].
///
/// `local_source` and `git_clone` populate `acquired.cache_key` against
/// the bare [`crate::cache_key_suffix`] which carries only the baked-in
/// hash. With `--extra-kconfig` set, the cache lookup and the
/// post-build store must target the extras-aware slot — this helper
/// performs the plain string append so both binaries share one merge
/// path. Plain append (vs rewriting the suffix) preserves the upstream
/// key prefix exactly and is robust to any future shape change in the
/// head segments. No-op when `extra` is `None`.
pub fn append_extra_kconfig_suffix(cache_key: &mut String, extra: Option<&str>) {
    if let Some(content) = extra {
        cache_key.push_str("-xkc");
        cache_key.push_str(&crate::extra_kconfig_hash(content));
    }
}

/// Pre-configure pass that warns when a user `--extra-kconfig` line
/// overrides a baked-in symbol from `EMBEDDED_KCONFIG`. The build
/// proceeds with the user value winning (per kbuild's last-wins
/// rule and the design intent of `--extra-kconfig`); this helper
/// just surfaces the override so the operator sees that their
/// fragment is shadowing a baked-in setting.
///
/// Output shape:
/// `--extra-kconfig overrides baked-in CONFIG_FOO (was =y, now =n)`
/// where the "was"/"now" values are extracted from the matching
/// lines on each side. `# CONFIG_X is not set` (kbuild's disable
/// directive) is normalized to "is not set" in the rendered
/// before/after for readability.
///
/// Free-text `#`-comments and blank lines in the user fragment are
/// skipped — only `CONFIG_X=...` assignments and `# CONFIG_X is
/// not set` directives count as overrides.
pub(super) fn warn_extra_kconfig_overrides_baked_in(extra: &str, cli_label: &str) {
    // Build a per-symbol map of the baked-in declarations once.
    // `EMBEDDED_KCONFIG` is small (<200 lines per ktstr.kconfig)
    // so a single pass is cheap.
    let mut baked: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for raw in EMBEDDED_KCONFIG.lines() {
        let line = raw.trim();
        if let Some(sym) = parse_kconfig_symbol(line) {
            baked.insert(sym, line);
        }
    }

    for raw in extra.lines() {
        let line = raw.trim();
        let Some(user_sym) = parse_kconfig_symbol(line) else {
            continue;
        };
        let Some(baked_line) = baked.get(user_sym) else {
            continue;
        };
        if *baked_line == line {
            continue;
        }
        tracing::warn!(
            cli_label = cli_label,
            symbol = user_sym,
            was = *baked_line,
            now = line,
            "--extra-kconfig overrides baked-in {user_sym} (was {}, now {})",
            render_kconfig_value(baked_line, user_sym),
            render_kconfig_value(line, user_sym),
        );
    }
}

/// Extract the symbol name from a kbuild `.config`-shaped line.
///
/// Returns `Some("CONFIG_FOO")` for `CONFIG_FOO=...` or
/// `# CONFIG_FOO is not set`, `None` for anything else (free-text
/// comments, blank lines, malformed input).
pub(super) fn parse_kconfig_symbol(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("# ")
        && let Some(sym) = rest.strip_suffix(" is not set")
        && sym.starts_with("CONFIG_")
    {
        return Some(sym);
    }
    if line.starts_with("CONFIG_")
        && let Some((sym, _)) = line.split_once('=')
    {
        return Some(sym);
    }
    None
}

/// Render the value half of a `.config` line for the override
/// warning's `(was =y, now =n)` formatting. For an assignment
/// line `CONFIG_FOO=y`, returns `=y`. For a disable directive
/// `# CONFIG_FOO is not set`, returns `is not set`. Falls back to
/// the full line when the shape is unrecognized so the operator
/// still gets information.
pub(super) fn render_kconfig_value<'a>(line: &'a str, sym: &str) -> &'a str {
    if let Some(value) = line.strip_prefix(sym)
        && value.starts_with('=')
    {
        return value;
    }
    if line == format!("# {sym} is not set") {
        return "is not set";
    }
    line
}

/// Post-`olddefconfig` validation pass for `--extra-kconfig` lines.
///
/// Scan the user's `extra` fragment line-by-line and verify each
/// non-empty, non-comment line either appears verbatim in the final
/// `.config` or matches the is-not-set sentinel for a
/// missing-from-output symbol (which kbuild renders as
/// `# CONFIG_X is not set` or simply omits). When a user line was
/// silently dropped by olddefconfig (typically because of an unmet
/// dependency), emit a `tracing::warn!` naming the requested setting
/// and the actual `.config` state — operator gets the diagnostic
/// without the build failing.
///
/// Best-effort: a missing or unreadable `.config` collapses to
/// silent return, since the surrounding pipeline failure would be
/// the actionable signal in those cases.
///
/// Lines starting with `#` that are NOT the kbuild
/// `# CONFIG_X is not set` form are treated as free-text comments
/// and skipped — they exist in user fragments but have no
/// `.config` counterpart.
pub(super) fn warn_dropped_extra_kconfig_lines(kernel_dir: &Path, extra: &str, cli_label: &str) {
    let config_path = kernel_dir.join(".config");
    let Ok(final_config) = std::fs::read_to_string(&config_path) else {
        return;
    };
    let final_lines: std::collections::HashSet<&str> =
        final_config.lines().map(str::trim).collect();

    for raw_line in extra.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        // Skip free-text comments — `# foo bar baz`. The kbuild
        // disable form `# CONFIG_X is not set` is a special-case
        // comment that participates in `.config` and IS checked.
        let is_disable_directive = line.starts_with("# CONFIG_") && line.ends_with(" is not set");
        let is_assignment = line.starts_with("CONFIG_") && line.contains('=');
        if !is_disable_directive && !is_assignment {
            continue;
        }
        if final_lines.contains(line) {
            continue;
        }
        // Line missing — either dropped by olddefconfig or rewritten
        // (e.g. `=y` → `is not set` because dep didn't resolve).
        // Look up the symbol's actual final value to enrich the
        // warning. Symbol name is everything up to `=` (assignment)
        // or between `# ` and ` is not set` (disable directive).
        let sym_name = if is_assignment {
            line.split('=').next().unwrap_or(line)
        } else {
            line.trim_start_matches("# ")
                .trim_end_matches(" is not set")
        };
        let final_state = final_config
            .lines()
            .find(|l| {
                let t = l.trim();
                t.starts_with(&format!("{sym_name}=")) || t == format!("# {sym_name} is not set")
            })
            .map(str::trim)
            .unwrap_or(
                "(absent — symbol not present in .config; likely \
                        disabled or unrecognized by kconfig)",
            );
        tracing::warn!(
            cli_label = cli_label,
            requested = line,
            final_state = final_state,
            "--extra-kconfig line did not survive `make olddefconfig` (likely an \
             unmet dependency or unrecognized symbol)"
        );
    }
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

/// Check if a kernel .config contains CONFIG_SCHED_CLASS_EXT=y.
pub fn has_sched_ext(kernel_dir: &std::path::Path) -> bool {
    let config = kernel_dir.join(".config");
    std::fs::read_to_string(config)
        .map(|s| s.lines().any(|l| l == "CONFIG_SCHED_CLASS_EXT=y"))
        .unwrap_or(false)
}

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
    (
        "CONFIG_VIRTIO_BLK",
        "required for ktstr DiskConfig — backs /dev/vd* in the guest. Depends on VIRTIO + BLOCK; \
         a user --extra-kconfig that strips BLOCK would silently disable this and disk-IO WorkTypes \
         would fail with a confusing 'no /dev/vda' inside the guest instead of a clear build error",
    ),
];

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

    // Build a HashSet of trimmed `.config` lines once, then probe
    // each critical option in O(1). The previous formulation
    // walked `config.lines()` once per critical option (O(N×M)
    // with M≈6 critical options and N≈5000 .config lines), which
    // turned every kernel-build pipeline into a 30K-line scan.
    // Trimming each line matches `all_fragment_lines_present`'s
    // configure-time behavior so the same `.config` parses
    // identically across both checks — without trim, a
    // configure-time write that produced trailing whitespace (or
    // a `.config` edited by hand on a Windows host with `\r\n`
    // line endings) would silently flag every critical option as
    // missing here while passing the configure-time check.
    let existing: std::collections::HashSet<&str> = config.lines().map(str::trim).collect();

    let mut missing = Vec::new();
    for &(option, hint) in VALIDATE_CONFIG_CRITICAL {
        let enabled = format!("{option}=y");
        if !existing.contains(enabled.as_str()) {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    // -- validate_kernel_config --

    /// Every entry in `VALIDATE_CONFIG_CRITICAL` must appear as `=y`
    /// in the embedded kconfig fragment. If a critical option is
    /// dropped from the fragment, builds skip it but validation
    /// keeps flagging it as missing — the user sees a build failure
    /// no tool install fixes. This test catches the drift at
    /// compile-test time.
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
             CONFIG_BPF_EVENTS=y\n\
             CONFIG_VIRTIO_BLK=y\n",
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

    /// Pin the per-line `str::trim` semantics in
    /// [`validate_kernel_config`]. CRLF line endings (`\r\n` from a
    /// Windows-edited file) and trailing-space lines must STILL
    /// validate when the option-after-trim equals the expected
    /// `CONFIG_X=y` form. A regression that dropped `.map(str::trim)`
    /// would surface every option as missing.
    #[test]
    fn validate_kernel_config_trim_handles_crlf_and_trailing_whitespace() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".config"),
            "CONFIG_SCHED_CLASS_EXT=y\r\n\
             CONFIG_DEBUG_INFO_BTF=y \n\
             CONFIG_BPF_SYSCALL=y\r\n\
             CONFIG_FTRACE=y \n\
             CONFIG_KPROBE_EVENTS=y\r\n\
             CONFIG_BPF_EVENTS=y \n\
             CONFIG_VIRTIO_BLK=y\r\n",
        )
        .unwrap();
        let result = validate_kernel_config(dir.path());
        assert!(
            result.is_ok(),
            "validate_kernel_config must trim per-line whitespace \
             before the HashSet probe; got: {result:?}",
        );
    }

    // -- parse_kconfig_symbol + render_kconfig_value --

    #[test]
    fn parse_kconfig_symbol_assignment() {
        assert_eq!(parse_kconfig_symbol("CONFIG_FOO=y"), Some("CONFIG_FOO"));
        assert_eq!(parse_kconfig_symbol("CONFIG_FOO=m"), Some("CONFIG_FOO"));
        assert_eq!(parse_kconfig_symbol("CONFIG_FOO=n"), Some("CONFIG_FOO"));
        assert_eq!(
            parse_kconfig_symbol("CONFIG_BAR=\"value\""),
            Some("CONFIG_BAR")
        );
    }

    #[test]
    fn parse_kconfig_symbol_disable_directive() {
        assert_eq!(
            parse_kconfig_symbol("# CONFIG_FOO is not set"),
            Some("CONFIG_FOO")
        );
    }

    #[test]
    fn parse_kconfig_symbol_rejects_free_text_comment() {
        assert!(parse_kconfig_symbol("# user note about foo").is_none());
        assert!(parse_kconfig_symbol("#").is_none());
        assert!(parse_kconfig_symbol("# this is a doc line").is_none());
    }

    #[test]
    fn parse_kconfig_symbol_rejects_blank_and_non_config() {
        assert!(parse_kconfig_symbol("").is_none());
        assert!(parse_kconfig_symbol("not a kconfig line").is_none());
        assert!(parse_kconfig_symbol("FOO=y").is_none());
    }

    #[test]
    fn render_kconfig_value_assignment_returns_value_with_equals() {
        assert_eq!(render_kconfig_value("CONFIG_FOO=y", "CONFIG_FOO"), "=y");
        assert_eq!(render_kconfig_value("CONFIG_FOO=n", "CONFIG_FOO"), "=n");
        assert_eq!(
            render_kconfig_value("CONFIG_BAR=\"value\"", "CONFIG_BAR"),
            "=\"value\""
        );
    }

    #[test]
    fn render_kconfig_value_disable_returns_is_not_set() {
        assert_eq!(
            render_kconfig_value("# CONFIG_FOO is not set", "CONFIG_FOO"),
            "is not set"
        );
    }

    #[test]
    fn render_kconfig_value_falls_back_to_full_line_on_unknown_shape() {
        let s = "CONFIG_FOO without equals";
        assert_eq!(render_kconfig_value(s, "CONFIG_FOO"), s);
    }

    // -- warn_extra_kconfig_overrides_baked_in --

    #[test]
    fn warn_extra_kconfig_overrides_does_not_panic_on_empty_fragment() {
        warn_extra_kconfig_overrides_baked_in("", "test");
    }

    #[test]
    fn warn_extra_kconfig_overrides_does_not_panic_on_no_overrides() {
        let novel = "CONFIG_KTSTR_TEST_NOVEL_SYMBOL_OVERRIDE_TEST=y\n";
        assert!(
            !EMBEDDED_KCONFIG.contains("CONFIG_KTSTR_TEST_NOVEL_SYMBOL_OVERRIDE_TEST"),
            "test fixture must use a symbol absent from EMBEDDED_KCONFIG"
        );
        warn_extra_kconfig_overrides_baked_in(novel, "test");
    }

    #[test]
    fn warn_extra_kconfig_overrides_does_not_panic_on_actual_override() {
        let user = "# CONFIG_BPF is not set\n";
        warn_extra_kconfig_overrides_baked_in(user, "test");
    }

    #[test]
    fn warn_extra_kconfig_overrides_skips_matching_assignments() {
        let user = "CONFIG_BPF=y\n";
        warn_extra_kconfig_overrides_baked_in(user, "test");
    }

    #[test]
    fn warn_extra_kconfig_overrides_skips_free_text_comments() {
        let user = "# this is a comment about something\n# another comment\n";
        warn_extra_kconfig_overrides_baked_in(user, "test");
    }

    // -- read_extra_kconfig --

    #[test]
    fn read_extra_kconfig_returns_content_for_valid_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("frag.kconfig");
        let content = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        std::fs::write(&path, content).unwrap();
        let got = read_extra_kconfig(&path, "test").unwrap();
        assert_eq!(got, content, "content must round-trip byte-for-byte");
    }

    #[test]
    fn read_extra_kconfig_not_found_arm_names_path_and_intent() {
        let dir = tempfile::TempDir::new().unwrap();
        let missing = dir.path().join("does-not-exist.kconfig");
        let display = missing.display().to_string();
        let err = read_extra_kconfig(&missing, "cargo ktstr").expect_err("missing file must Err");
        assert!(
            err.contains(&display),
            "ENOENT message must name the literal path: {err}"
        );
        assert!(
            err.contains("--extra-kconfig"),
            "ENOENT message must name the flag: {err}"
        );
        assert!(
            err.contains("file not found"),
            "ENOENT arm must surface a `file not found` token: {err}"
        );
    }

    #[test]
    fn read_extra_kconfig_directory_arm_distinguishes_from_not_found() {
        let dir = tempfile::TempDir::new().unwrap();
        let display = dir.path().display().to_string();
        let err =
            read_extra_kconfig(dir.path(), "cargo ktstr").expect_err("directory path must Err");
        assert!(
            err.contains(&display),
            "EISDIR message must name the path: {err}"
        );
        assert!(
            err.contains("is a directory"),
            "EISDIR arm must surface its specific token: {err}"
        );
    }

    #[test]
    fn read_extra_kconfig_invalid_utf8_arm_names_constraint() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("binary.bin");
        std::fs::write(&path, [0xFFu8, 0xFE, 0x00, 0x42]).unwrap();
        let err = read_extra_kconfig(&path, "cargo ktstr").expect_err("non-UTF-8 file must Err");
        assert!(
            err.contains("not valid UTF-8"),
            "UTF-8 arm must surface the constraint name: {err}"
        );
        assert!(
            err.contains("ASCII text"),
            "UTF-8 arm must mention the kconfig content constraint: {err}"
        );
    }

    #[test]
    fn read_extra_kconfig_empty_file_returns_empty_string() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("empty.kconfig");
        std::fs::write(&path, "").unwrap();
        let got = read_extra_kconfig(&path, "cargo ktstr").unwrap();
        assert_eq!(
            got, "",
            "empty file must round-trip as empty String, not Err"
        );
    }

    #[test]
    #[cfg(unix)]
    fn read_extra_kconfig_follows_symlink_chain() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::TempDir::new().unwrap();
        let target = dir.path().join("real.kconfig");
        let link = dir.path().join("link.kconfig");
        std::fs::write(&target, "CONFIG_BPF=y\n").unwrap();
        symlink(&target, &link).unwrap();
        let got = read_extra_kconfig(&link, "test").unwrap();
        assert_eq!(
            got, "CONFIG_BPF=y\n",
            "symlink must resolve to target content"
        );
    }

    // -- warn_dropped_extra_kconfig_lines --

    #[test]
    fn warn_dropped_extra_kconfig_lines_silent_when_config_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        let extra = "CONFIG_FOO=y\n";
        warn_dropped_extra_kconfig_lines(dir.path(), extra, "test");
    }

    #[test]
    fn warn_dropped_extra_kconfig_lines_silent_when_all_present() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_FOO=y\nCONFIG_BAR=m\n").unwrap();
        let extra = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        warn_dropped_extra_kconfig_lines(dir.path(), extra, "test");
    }

    #[test]
    fn warn_dropped_extra_kconfig_lines_does_not_panic_on_dropped_line() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF=y\n").unwrap();
        let extra = "CONFIG_KTSTR_DROPPED_TEST_NOVEL=y\n";
        warn_dropped_extra_kconfig_lines(dir.path(), extra, "test");
    }

    #[test]
    fn warn_dropped_extra_kconfig_lines_does_not_panic_on_rewritten_line() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".config"),
            "# CONFIG_KTSTR_REWRITE_TEST is not set\n",
        )
        .unwrap();
        let extra = "CONFIG_KTSTR_REWRITE_TEST=y\n";
        warn_dropped_extra_kconfig_lines(dir.path(), extra, "test");
    }

    #[test]
    fn warn_dropped_extra_kconfig_lines_skips_free_text_comments() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF=y\n").unwrap();
        let extra = "# decorative header\nCONFIG_BPF=y\n";
        warn_dropped_extra_kconfig_lines(dir.path(), extra, "test");
    }

    // -- configure_kernel --

    #[test]
    fn configure_kernel_appends_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF=y\n").unwrap();
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
        assert_eq!(config, initial);
    }

    /// Fragment asks `CONFIG_NR_CPUS=1`, .config has
    /// `CONFIG_NR_CPUS=128`. A plain `contains(fragment_line)` would
    /// false-match the substring "CONFIG_NR_CPUS=1" inside
    /// "CONFIG_NR_CPUS=128" and skip the append. Exact-line matching
    /// via the HashSet helper distinguishes the two and appends.
    #[test]
    fn configure_kernel_rejects_numeric_prefix_false_match() {
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
        let config = "CONFIG_NR_CPUS=128\n";
        assert!(!all_fragment_lines_present("CONFIG_NR_CPUS=1\n", config));
        assert!(!all_fragment_lines_present("CONFIG_NR_CPUS=12\n", config));
    }

    #[test]
    fn all_fragment_lines_present_disable_directive_participates() {
        let config = "CONFIG_BPF=y\n";
        assert!(!all_fragment_lines_present(
            "# CONFIG_BPF is not set\n",
            config
        ));
    }

    #[test]
    fn all_fragment_lines_present_empty_lines_skipped() {
        let config = "CONFIG_FOO=y\n";
        assert!(all_fragment_lines_present("\n\nCONFIG_FOO=y\n\n", config));
    }

    #[test]
    fn all_fragment_lines_present_free_text_comment_stripped() {
        let config = "CONFIG_FOO=y\n";
        let fragment = "# Build for testing scx schedulers\nCONFIG_FOO=y\n";
        assert!(
            all_fragment_lines_present(fragment, config),
            "free-text comment must not block the present-in-config check"
        );
    }

    #[test]
    fn all_fragment_lines_present_disable_directive_still_participates() {
        let config = "CONFIG_FOO=y\n# CONFIG_BAR is not set\n";
        let fragment_present = "# CONFIG_BAR is not set\n";
        assert!(
            all_fragment_lines_present(fragment_present, config),
            "disable directive present in config must satisfy probe"
        );
        let config_missing = "CONFIG_FOO=y\n";
        assert!(
            !all_fragment_lines_present(fragment_present, config_missing),
            "disable directive missing from config must fail probe"
        );
    }

    #[test]
    fn all_fragment_lines_present_section_header_comment_stripped() {
        let config = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        let fragment = "\
# == BPF support ==\n\
CONFIG_FOO=y\n\
# == Tracing ==\n\
CONFIG_BAR=m\n";
        assert!(all_fragment_lines_present(fragment, config));
    }

    // -- is_kconfig_semantic_line predicate --

    #[test]
    fn is_kconfig_semantic_line_classifies_assignment_disable_and_comment() {
        assert!(is_kconfig_semantic_line("CONFIG_FOO=y"));
        assert!(is_kconfig_semantic_line("CONFIG_NR_CPUS=128"));
        assert!(is_kconfig_semantic_line("# CONFIG_BPF is not set"));
        assert!(is_kconfig_semantic_line("#  CONFIG_BPF is not set"));
        assert!(!is_kconfig_semantic_line("# Build for testing"));
        assert!(!is_kconfig_semantic_line("# == Section header =="));
        assert!(!is_kconfig_semantic_line(""));
        assert!(!is_kconfig_semantic_line("# CONFIG_FOO is enabled"));
        assert!(!is_kconfig_semantic_line("# CONFIG_FOO"));
    }
}
