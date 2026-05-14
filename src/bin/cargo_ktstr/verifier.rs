//! `cargo ktstr verifier` subcommand dispatch helpers.
//!
//! Houses the [`run_verifier`] dispatcher that builds (or finds) a
//! scheduler binary, queries it for declared schedulers via the
//! `--ktstr-list-schedulers` JSON protocol, resolves each scheduler's
//! declared kernels (or the `--kernel` override), filters the gauntlet
//! topology presets through each scheduler's
//! [`TopologyConstraints`], and feeds every accepted
//! (scheduler × kernel × topology) cell into
//! [`ktstr::verifier::collect_verifier_output`]. Binaries that produce
//! no declared schedulers (legacy non-ktstr schedulers, or ones where
//! the `--ktstr-list-schedulers` constructor isn't linked) fall back
//! to a single-cell run with [`TopologyJson::SINGLE_CPU`].
//!
//! [`TopologyConstraints`]: ktstr::test_support::TopologyConstraints
//! [`TopologyJson::SINGLE_CPU`]: ktstr::test_support::TopologyJson::SINGLE_CPU

use std::path::{Path, PathBuf};
use std::process::Command;

use ktstr::cli;
use ktstr::test_support::{
    SchedulerJson, TopologyConstraints, TopologyJson, host_capacity,
};

use crate::kernel::{resolve_kernel_image, resolve_kernel_set};

/// Spawn `bin --ktstr-list-schedulers` and parse the emitted JSON
/// array of [`SchedulerJson`]. ktstr-based scheduler binaries
/// intercept the flag via the `ctor::ctor` constructor attribute and
/// exit 0 with the JSON on stdout. Non-ktstr binaries fail one of
/// three ways:
///
/// 1. Spawn fails (binary not executable) — returns `Vec::new()`.
/// 2. Binary exits non-zero (the flag is unknown) — returns `Vec::new()`.
/// 3. Binary exits 0 but stdout isn't a `SchedulerJson` array — emits
///    a stderr warning and returns `Vec::new()`. This last case
///    surfaces schema drift or partial output that would otherwise
///    silently lose every declared scheduler.
fn list_schedulers_from_binary(bin: &Path) -> Vec<SchedulerJson> {
    let Ok(output) = Command::new(bin).arg("--ktstr-list-schedulers").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    match serde_json::from_slice(&output.stdout) {
        Ok(list) => list,
        Err(e) => {
            let preview: String =
                String::from_utf8_lossy(&output.stdout).chars().take(200).collect();
            eprintln!(
                "cargo ktstr: warning: {} exited 0 but --ktstr-list-schedulers stdout is not a SchedulerJson array ({e}); falling back to single-cell dispatch. stdout preview: {preview:?}",
                bin.display(),
            );
            Vec::new()
        }
    }
}

/// Whether the verifier should emulate topology via KVM without
/// pinning host CPUs (`KTSTR_NO_PERF_MODE` set) or pin host CPUs
/// (perf mode — the default). Plumbs into the gauntlet preset
/// filter to select between
/// [`TopologyConstraints::accepts_no_perf_mode`] (KVM-emulated,
/// host-LLC checks skipped) and
/// [`TopologyConstraints::accepts`] (perf-mode, full host-capacity
/// checks).
fn no_perf_mode_active() -> bool {
    std::env::var("KTSTR_NO_PERF_MODE").is_ok()
}

/// Iterate (label, kernel-image-path) pairs honoring the
/// `--kernel`-override / declared-list / auto-discovery cascade.
/// Empty `kernel_override` AND empty `declared` falls through to the
/// historical `resolve_kernel_image(None)` auto-discovery path; either
/// non-empty input is resolved via the same `resolve_kernel_set` +
/// `find_image_in_dir` plumbing the test path uses.
fn resolve_kernels(
    kernel_override: &[String],
    declared: &[String],
) -> Result<Vec<(String, PathBuf)>, String> {
    let specs = if !kernel_override.is_empty() {
        kernel_override
    } else {
        declared
    };
    if specs.is_empty() {
        let path = resolve_kernel_image(None)?;
        return Ok(vec![("auto".to_string(), path)]);
    }
    let resolved = resolve_kernel_set(specs)?;
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
    Ok(out)
}

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

    // Build the ktstr init binary.
    let ktstr_bin =
        ktstr::build_and_find_binary("ktstr").map_err(|e| format!("build ktstr: {e:#}"))?;

    // Discover declared schedulers in the binary. Empty result means
    // either: (a) a non-ktstr scheduler binary that doesn't recognize
    // the --ktstr-list-schedulers flag (legitimate legacy path), or
    // (b) a ktstr binary that ran the constructor but had no
    // declare_scheduler! registrations. Both cases fall through to
    // single-cell dispatch since neither has scheduler-side metadata
    // to drive a sweep.
    let scheduler_jsons = list_schedulers_from_binary(&sched_bin);

    if scheduler_jsons.is_empty() {
        return run_single_cell(&sched_bin, &ktstr_bin, &kernel, raw);
    }

    run_sweep(&sched_bin, &ktstr_bin, &kernel, &scheduler_jsons, raw)
}

/// Legacy single-cell verifier run for binaries with no declared
/// schedulers. Resolves `--kernel` (or auto-discovers) and runs
/// `collect_verifier_output` once per kernel with `SINGLE_CPU`. Uses
/// the same `=== fallback | kernel <klabel> | topology single_cpu ===`
/// header schema as the sweep path so downstream parsers see a
/// uniform cell-banner regardless of which dispatch mode ran.
fn run_single_cell(
    sched_bin: &Path,
    ktstr_bin: &Path,
    kernel: &[String],
    raw: bool,
) -> Result<(), String> {
    let kernel_paths = resolve_kernels(kernel, &[])?;
    for (klabel, kernel_path) in &kernel_paths {
        eprintln!(
            "cargo ktstr: [fallback | kernel {klabel} | topology single_cpu] collecting verifier stats",
        );
        println!(
            "\n=== fallback | kernel {klabel} | topology single_cpu ==="
        );
        let result = ktstr::verifier::collect_verifier_output(
            sched_bin,
            ktstr_bin,
            kernel_path,
            &[],
            TopologyJson::SINGLE_CPU,
        )
        .map_err(|e| format!("collect verifier output for fallback × {klabel}: {e:#}"))?;
        let output = ktstr::verifier::format_verifier_output("verifier", &result, raw);
        print!("{output}");
    }
    Ok(())
}

/// Sweep dispatch: iterate every (scheduler × kernel × accepted
/// gauntlet preset) cell and run the verifier per cell. Per-scheduler
/// kernels come from `SchedulerJson.kernels` (overridden by `--kernel`
/// when set). Per-cell topology comes from `gauntlet_presets()`
/// filtered through [`TopologyConstraints::accepts`] (when perf-mode
/// pinning is on) or [`TopologyConstraints::accepts_no_perf_mode`]
/// (when `KTSTR_NO_PERF_MODE` is set). Per-cell verifier failures emit
/// the error to stderr and continue to the next cell; infrastructure
/// failures (kernel resolution, host capacity) abort the sweep. The
/// sweep returns `Err` if zero cells executed so CI can distinguish
/// "verifier passed all cells" from "no cells ran."
fn run_sweep(
    sched_bin: &Path,
    ktstr_bin: &Path,
    kernel_override: &[String],
    scheduler_jsons: &[SchedulerJson],
    raw: bool,
) -> Result<(), String> {
    let presets = ktstr::vm::gauntlet_presets();
    let (host_cpus, host_llcs, host_max_cpus_per_llc) = host_capacity();
    let no_perf_mode = no_perf_mode_active();

    eprintln!(
        "cargo ktstr: sweep over {} declared scheduler(s) × per-scheduler kernels × {} presets (no_perf_mode={no_perf_mode}, host_cpus={host_cpus}, host_llcs={host_llcs})",
        scheduler_jsons.len(),
        presets.len(),
    );

    let mut cells_attempted: u32 = 0;
    let mut cells_failed: u32 = 0;

    for sched_json in scheduler_jsons {
        let constraints: TopologyConstraints = sched_json.constraints.into();
        let kernel_paths = resolve_kernels(kernel_override, &sched_json.kernels)?;

        // Filter presets via the same accept-logic gauntlet dispatch
        // uses (dispatch::for_each_gauntlet_variant). Perf-mode honors
        // host LLC + per-LLC width; no-perf-mode skips those since
        // KVM emulates the topology.
        let accepted: Vec<&ktstr::vm::TopoPreset> = presets
            .iter()
            .filter(|p| {
                if no_perf_mode {
                    constraints.accepts_no_perf_mode(&p.topology, host_cpus)
                } else {
                    constraints.accepts(
                        &p.topology,
                        host_cpus,
                        host_llcs,
                        host_max_cpus_per_llc,
                    )
                }
            })
            .collect();

        if accepted.is_empty() {
            eprintln!(
                "cargo ktstr: scheduler {} has no gauntlet preset accepted (constraints: NUMA {min_nn}..{max_nn:?} / LLCs {min_l}..{max_l:?} / CPUs {min_c}..{max_c:?} / SMT={smt}; host has {host_cpus} CPUs / {host_llcs} LLCs / {host_max_cpus_per_llc} max-cpus-per-llc) — skipping",
                sched_json.name,
                min_nn = constraints.min_numa_nodes,
                max_nn = constraints.max_numa_nodes,
                min_l = constraints.min_llcs,
                max_l = constraints.max_llcs,
                min_c = constraints.min_cpus,
                max_c = constraints.max_cpus,
                smt = constraints.requires_smt,
            );
            continue;
        }

        let sched_args: Vec<String> = sched_json.sched_args.clone();

        for (klabel, kernel_path) in &kernel_paths {
            for preset in &accepted {
                cells_attempted += 1;
                eprintln!(
                    "cargo ktstr: [{name} | kernel {klabel} | topology {preset}] collecting verifier stats",
                    name = sched_json.name,
                    preset = preset.name,
                );
                println!(
                    "\n=== {name} | kernel {klabel} | topology {preset} ===",
                    name = sched_json.name,
                    preset = preset.name,
                );

                let topology = TopologyJson::from(preset.topology);
                match ktstr::verifier::collect_verifier_output(
                    sched_bin,
                    ktstr_bin,
                    kernel_path,
                    &sched_args,
                    topology,
                ) {
                    Ok(result) => {
                        let output =
                            ktstr::verifier::format_verifier_output("verifier", &result, raw);
                        print!("{output}");
                    }
                    Err(e) => {
                        // Per-cell failure: log and continue to the next cell.
                        // Aggregate counter at end decides whether the sweep as
                        // a whole failed.
                        cells_failed += 1;
                        eprintln!(
                            "cargo ktstr: cell {name} × {klabel} × {preset_name} FAILED: {e:#}",
                            name = sched_json.name,
                            preset_name = preset.name,
                        );
                    }
                }
            }
        }
    }

    if cells_attempted == 0 {
        return Err(format!(
            "verifier sweep executed 0 cells across {} declared scheduler(s) — every scheduler's gauntlet-preset filter rejected every preset (no_perf_mode={no_perf_mode}, host_cpus={host_cpus}). Relax scheduler max_cpus / max_llcs, run on a larger host, or (if the bottleneck is per-LLC width rather than total host CPUs) set KTSTR_NO_PERF_MODE=1.",
            scheduler_jsons.len(),
        ));
    }
    if cells_failed > 0 {
        return Err(format!(
            "verifier sweep: {cells_failed} of {cells_attempted} cell(s) failed",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_kernels_override_wins_over_declared() {
        // override and declared both non-empty: override wins. The
        // precedence branch is observable through the specific
        // empty-spec error string that ONLY the override-with-
        // whitespace path produces (line 96 of resolve_kernels). If a
        // regression reversed the precedence, declared="6.14" would
        // route through resolve_kernel_set which produces a different
        // error.
        let override_specs = vec![" ".to_string()];
        let declared = vec!["6.14".to_string()];
        let err = resolve_kernels(&override_specs, &declared)
            .expect_err("whitespace-only override must reject");
        assert!(
            err.contains("every supplied value parsed to empty"),
            "error should be the override-empty-parse error: {err}"
        );
    }

    #[test]
    fn resolve_kernels_empty_inputs_falls_through_to_auto_discovery() {
        // With both override and declared empty, resolve_kernels
        // hits resolve_kernel_image(None) which either auto-discovers
        // (returns Ok with the "auto" label per line 88) or fails
        // with the cascade's last error. Both outcomes pin the
        // empty-empty branch: Ok-path asserts the "auto" label
        // signature; Err-path asserts the absence of the empty-parse
        // error (which only the non-empty branch produces).
        match resolve_kernels(&[], &[]) {
            Ok(result) => {
                assert_eq!(
                    result.len(),
                    1,
                    "auto-discovery returns a single (label, path) pair"
                );
                assert_eq!(
                    result[0].0, "auto",
                    "auto-discovery label is 'auto' per the empty-empty branch"
                );
            }
            Err(err) => {
                assert!(
                    !err.contains("every supplied value parsed to empty"),
                    "auto-discovery error should not be the empty-parse error: {err}",
                );
            }
        }
    }
}
