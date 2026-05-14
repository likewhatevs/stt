//! `cargo ktstr verifier` subcommand dispatch helpers.
//!
//! Houses the [`run_verifier`] dispatcher that builds (or finds)
//! a scheduler binary, resolves `--kernel` to one or more
//! kernel images, and feeds each pair into the
//! [`ktstr::verifier::collect_verifier_output`] BPF-stats
//! collector. Multi-kernel inputs iterate sequentially with
//! per-kernel header lines; single-kernel inputs preserve the
//! historical bare-output shape. Each invocation runs the
//! scheduler exactly once per kernel with no extra flags.

use std::path::PathBuf;

use ktstr::cli;

use crate::kernel::{resolve_kernel_image, resolve_kernel_set};

/// Dispatch the `cargo ktstr verifier` subcommand.
pub(crate) fn run_verifier(
    scheduler: Option<String>,
    scheduler_bin: Option<PathBuf>,
    kernel: Vec<String>,
    raw: bool,
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

        eprintln!("cargo ktstr: collecting verifier stats");
        // Single-CPU 1,1,1,1 topology — one verifier run per kernel
        // with no scheduler args.
        let result = ktstr::verifier::collect_verifier_output(
            &sched_bin,
            &ktstr_bin,
            kernel_path,
            &[],
            ktstr::test_support::TopologyJson::SINGLE_CPU,
        )
        .map_err(|e| format!("collect verifier output: {e:#}"))?;

        let output = ktstr::verifier::format_verifier_output("verifier", &result, raw);
        print!("{output}");
    }

    Ok(())
}
