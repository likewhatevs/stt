//! `cargo ktstr verifier` subcommand dispatch helpers.
//!
//! Houses the [`run_verifier`] dispatcher that builds (or finds)
//! a scheduler binary, resolves `--kernel` to one or more
//! kernel images, and feeds each pair into the
//! [`ktstr::verifier::collect_verifier_output`] BPF-stats
//! collector. Multi-kernel inputs iterate sequentially with
//! per-kernel header lines; single-kernel inputs preserve the
//! historical bare-output shape.
//!
//! Profile expansion (the `--all-profiles` / `--profiles A,B,C`
//! shapes) lives here too: [`generate_flag_profiles`] computes the
//! power set of declared flags filtered by `requires`
//! constraints, and [`profile_sched_args`] resolves each profile
//! name back to the scheduler-binary CLI args declared in
//! `--ktstr-list-flags`.

use std::path::{Path, PathBuf};
use std::process::Command;

use ktstr::cli;

use crate::kernel::{resolve_kernel_image, resolve_kernel_set};

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
/// Produces the power set of flags, filtered by requires constraints,
/// via the shared [`ktstr::scenario::compute_flag_profiles`] generator.
/// Each profile's flags are sorted in declaration order. The profile
/// name is the flags joined with `+`, or `"default"` when empty.
///
/// The `n > 31` cap protects against unbounded power-set growth: 2^31
/// = ~2.1 billion profiles would exhaust memory on any host long
/// before the verifier sees the first profile. The bound matches the
/// `i32` upper half (signed-safe shift base) and gives operators a
/// clear error to act on (`Use --profiles ...`) rather than an OOM
/// panic. Real schedulers we ship today expose ~5-10 flags;
/// hand-picked invocations with `--profiles A,B,C` do not flow through
/// this generator at all.
fn generate_flag_profiles(
    flags: &[ktstr::scenario::flags::FlagDeclJson],
) -> Vec<(String, Vec<String>)> {
    let n = flags.len();
    if n > 31 {
        eprintln!(
            "cargo ktstr: error: scheduler has {n} flags, power set too large (2^{n}). \
             Use --profiles to select specific profiles."
        );
        return Vec::new();
    }

    let all: Vec<String> = flags.iter().map(|f| f.name.clone()).collect();
    let requires_fn = |name: &String| -> Vec<String> {
        flags
            .iter()
            .find(|f| f.name == *name)
            .map(|f| f.requires.clone())
            .unwrap_or_default()
    };

    ktstr::scenario::compute_flag_profiles(&all, requires_fn, &[], &[])
        .into_iter()
        .map(|flag_names| {
            let name = if flag_names.is_empty() {
                "default".to_string()
            } else {
                flag_names.join("+")
            };
            (name, flag_names)
        })
        .collect()
}

/// Collect the extra scheduler args for a set of active flags.
///
/// Returns `Err` if any flag in `active_flags` is not declared in
/// `all_flags`. Silently dropping unknown flags masked typos in
/// CLI `--profiles` lists and version-drift in cached nextest args —
/// the caller would see "flag applied" in the profile name but the
/// scheduler was actually invoked without the corresponding CLI arg.
fn profile_sched_args(
    active_flags: &[String],
    all_flags: &[ktstr::scenario::flags::FlagDeclJson],
) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    for flag_name in active_flags {
        match all_flags.iter().find(|f| f.name == *flag_name) {
            Some(decl) => args.extend(decl.args.iter().cloned()),
            None => {
                let known: Vec<&str> = all_flags.iter().map(|f| f.name.as_str()).collect();
                return Err(format!(
                    "unknown flag {flag_name:?} (known: {})",
                    known.join(", ")
                ));
            }
        }
    }
    Ok(args)
}

/// Dispatch the `cargo ktstr verifier` subcommand.
///
/// Profile selection is governed at clap parse time: `--profiles
/// A,B,C` and `--all-profiles` are NOT mutually exclusive — the
/// CLI definition lets the operator pass both, in which case the
/// non-empty `profiles_filter` wins. The dispatch arm below is:
///   - `all_profiles == false && profiles_filter.is_empty()` →
///     single-profile mode: build with no extra flags, run the
///     verifier exactly once. This is the historical behaviour
///     for schedulers that don't expose `--ktstr-list-flags`.
///   - `all_profiles == true || !profiles_filter.is_empty()` →
///     multi-profile mode: discover flags via
///     `--ktstr-list-flags`, expand the power set (or the
///     filter), and run the verifier once per resulting profile
///     with per-profile sched args.
///
/// `profiles_filter` non-empty implicitly enables multi-profile
/// mode without `--all-profiles`. The clap layer documents this
/// "implies --all-profiles for flag discovery" wording in the
/// `--profiles` help text so operators do not pass both together
/// out of caution. Pinning these flags as non-conflicting is
/// deliberate: `--profiles foo,bar --all-profiles` is treated as
/// "filter the all-profiles power set down to foo and bar" (the
/// current implementation drops the unfiltered expansion when
/// the filter is non-empty), and a strict `conflicts_with`
/// would reject the operator's intent.
pub(crate) fn run_verifier(
    scheduler: Option<String>,
    scheduler_bin: Option<PathBuf>,
    kernel: Vec<String>,
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

    // Resolve --kernel into a flat (label, kernel_dir) list. Empty
    // input falls through to the single-kernel auto-discovery path
    // below (`resolve_kernel_image(None)` → `find_kernel`'s
    // fallback chain), preserving the no-flag behaviour. A
    // single-entry list is treated identically to the historical
    // single-kernel path: one verifier run, no kernel-prefixed
    // output. Two or more entries (multiple `--kernel` flags, OR
    // a single `--kernel` Range that expanded to multiple
    // releases) iterate sequentially with per-kernel header lines.
    let kernel_paths: Vec<(String, PathBuf)> = if kernel.is_empty() {
        // Auto-discovery: route through `resolve_kernel_image(None)`
        // so the existing `find_kernel` cascade applies, then label
        // the result `"auto"` for diagnostic visibility on the rare
        // path where the user neither passed `--kernel` nor exported
        // `KTSTR_KERNEL`.
        let path = resolve_kernel_image(None)?;
        vec![("auto".to_string(), path)]
    } else {
        // Multi-kernel resolution shares its plumbing with the test
        // path (`run_cargo_sub`'s `resolve_kernel_set` call),
        // including Range expansion and Git fetch. Each resolved
        // entry is a built / cached kernel directory; convert it to
        // a bootable image via `find_image_in_dir` since the
        // verifier collects stats from a loaded image rather than
        // a directory.
        let resolved = resolve_kernel_set(&kernel)?;
        if resolved.is_empty() {
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        let mut out: Vec<(String, PathBuf)> = Vec::with_capacity(resolved.len());
        for (label, dir) in resolved {
            let image = ktstr::kernel_path::find_image_in_dir(&dir).ok_or_else(|| {
                format!(
                    "no kernel image found in {} (resolved from --kernel {label})",
                    dir.display()
                )
            })?;
            out.push((label, image));
        }
        out
    };

    // Build the ktstr init binary.
    let ktstr_bin =
        ktstr::build_and_find_binary("ktstr").map_err(|e| format!("build ktstr: {e:#}"))?;

    let multi_kernel = kernel_paths.len() > 1;
    for (i, (label, kernel_path)) in kernel_paths.iter().enumerate() {
        if multi_kernel {
            eprintln!(
                "cargo ktstr: [kernel {}/{}] {label}",
                i + 1,
                kernel_paths.len(),
            );
            // Header on stdout so a redirected `>` capture
            // separates kernels even when stderr isn't pulled.
            println!("\n=== kernel: {label} ===");
        }

        if all_profiles || !profiles_filter.is_empty() {
            run_verifier_all_profiles(&sched_bin, &ktstr_bin, kernel_path, raw, &profiles_filter)?;
            continue;
        }

        eprintln!("cargo ktstr: collecting verifier stats");
        let result =
            ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, kernel_path, &[])
                .map_err(|e| format!("collect verifier output: {e:#}"))?;

        let output = ktstr::verifier::format_verifier_output("verifier", &result, raw);
        print!("{output}");
    }

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
            "cargo ktstr: scheduler does not support --ktstr-list-flags, \
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
    if total == 0 {
        // Differentiate the empty-profile cases so the user gets an
        // actionable error rather than a generic "0 profiles" message.
        return Err(if flags.len() > 31 {
            format!(
                "no profiles to verify: power-set generation is capped at \
                 31 flags (found {}); use --profiles to select a subset",
                flags.len(),
            )
        } else {
            format!(
                "no profiles to verify: {} flag(s) advertised but profile \
                 generation produced 0 profiles — check `requires` \
                 dependencies and exclusions for cycles or conflicts",
                flags.len(),
            )
        });
    }
    if total > 32 {
        eprintln!(
            "cargo ktstr: warning: {total} profiles to verify (>32). \
             Use --profiles to select a subset."
        );
    }

    eprintln!(
        "cargo ktstr: verifying {total} profile{}",
        if total == 1 { "" } else { "s" }
    );

    // Per-profile summary table: (profile_name, Vec<(prog_name, verified_insns)>).
    let mut summary: Vec<(String, Vec<(String, u32)>)> = Vec::new();

    for (i, (profile_name, active_flags)) in profiles.iter().enumerate() {
        eprintln!(
            "cargo ktstr: [{}/{}] profile: {}",
            i + 1,
            total,
            profile_name
        );

        let extra_args = profile_sched_args(active_flags, &flags)
            .map_err(|e| format!("profile {profile_name}: {e}"))?;
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

    let profile_names: Vec<&str> = summary.iter().map(|(n, _)| n.as_str()).collect();
    let mut table = ktstr::cli::new_table();
    let mut header: Vec<&str> = Vec::with_capacity(1 + profile_names.len());
    header.push("program");
    header.extend(profile_names.iter().copied());
    table.set_header(header);

    for prog in &prog_names {
        let mut row: Vec<String> = Vec::with_capacity(1 + profile_names.len());
        row.push(prog.clone());
        for (_, progs) in summary {
            let insns = progs
                .iter()
                .find(|(n, _)| n == prog)
                .map(|(_, v)| *v)
                .unwrap_or(0);
            row.push(insns.to_string());
        }
        table.add_row(row);
    }

    println!("{table}");
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let args = profile_sched_args(&active, &flags).unwrap();
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
        let args = profile_sched_args(&active, &flags).unwrap();
        assert!(args.is_empty());
    }

    #[test]
    fn profile_sched_args_unknown_flag_errors() {
        // Silently dropping unknown flag names would mask typos in
        // --profiles CLI lists and version-drift in cached nextest
        // args — the profile NAME would still say "foo+bar" while
        // the scheduler was invoked without the corresponding CLI
        // switches.
        let flags = vec![ktstr::scenario::flags::FlagDeclJson {
            name: "llc".to_string(),
            args: vec!["--llc".to_string()],
            requires: vec![],
        }];
        let active = vec!["llc".to_string(), "unknown_flag".to_string()];
        let err = profile_sched_args(&active, &flags).unwrap_err();
        assert!(
            err.contains("unknown_flag"),
            "error should cite flag: {err}"
        );
        assert!(err.contains("llc"), "error should list known flags: {err}");
    }
}
