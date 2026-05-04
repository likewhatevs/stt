//! `cargo ktstr shell` — boot a KVM VM and drop into a busybox shell.
//!
//! Houses [`run_shell`], the dispatcher behind the `Shell` enum
//! variant in [`crate::cli::KtstrCommand`]. Kernel-spec resolution
//! delegates to [`crate::kernel::resolve_kernel_image`]; topology
//! parsing, include-file resolution, and KVM probe go through
//! [`ktstr::cli`]; the actual VM boot is [`ktstr::run_shell`].

use std::path::{Path, PathBuf};

use ktstr::cli;

use crate::kernel::resolve_kernel_image;

/// Dispatch the `cargo ktstr shell` subcommand: launch a KVM VM and
/// drop into a busybox shell inside the guest.
///
/// `--cpu-cap` requiring `--no-perf-mode` is enforced at clap parse
/// time via `requires = "no_perf_mode"` on the Shell variant in
/// [`crate::cli`]; this dispatcher trusts the parser's gate and only
/// re-validates the env-level conflict (`KTSTR_BYPASS_LLC_LOCKS=1`)
/// that clap cannot see.
///
/// Both `unsafe std::env::set_var` calls below run on the call chain
/// `main` → match arm → `run_shell` with no thread spawn anywhere —
/// the binary's `tokio` features (`rt` only, no `rt-multi-thread`)
/// rule out runtime-spawned worker threads, and no helper here calls
/// `thread::spawn` before the env write. The single-threaded
/// invariant the SAFETY comments rely on holds for the lifetime of
/// these writes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_shell(
    kernel: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
    memory_mb: Option<u32>,
    dmesg: bool,
    exec: Option<String>,
    no_perf_mode: bool,
    cpu_cap: Option<usize>,
    disk: Option<String>,
) -> Result<(), String> {
    if no_perf_mode {
        // SAFETY: single-threaded at this point — main → dispatch →
        // run_shell, no prior thread spawn (ktstr's tokio feature set
        // is `rt` only, no `rt-multi-thread`; no helper above this
        // point calls `thread::spawn`). No concurrent env readers
        // exist, so set_var is sound.
        unsafe { std::env::set_var("KTSTR_NO_PERF_MODE", "1") };
    }
    if let Some(cap) = cpu_cap {
        // Env-level conflict with KTSTR_BYPASS_LLC_LOCKS=1. The
        // CLI-level `--cpu-cap` requires `--no-perf-mode` rule is
        // enforced at clap parse time via `requires = "no_perf_mode"`
        // on the Shell variant in crate::cli, not here.
        if std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_some_and(|v| !v.is_empty())
        {
            return Err(
                "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                 --cpu-cap is a resource contract; bypass disables the contract entirely."
                    .to_string(),
            );
        }
        // Validate early so a bad cap surfaces at CLI-parse time.
        cli::CpuCap::new(cap).map_err(|e| format!("{e:#}"))?;
        // SAFETY: single-threaded at this point per the chain
        // documented on the function — no concurrent env readers.
        unsafe { std::env::set_var("KTSTR_CPU_CAP", cap.to_string()) };
    }
    // Parse the human-readable disk size into a DiskConfig before the
    // KVM probe so a bad string surfaces at CLI-argument time, not
    // mid-VM-setup. `parse_disk_arg` returns `Ok(None)` when the
    // attribute is absent and applies `DiskConfig::default()` for
    // every knob except `capacity_mb` when present.
    let disk_cfg = cli::parse_disk_arg(disk.as_deref()).map_err(|e| format!("{e:#}"))?;
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    let (numa_nodes, llcs, cores, threads) =
        cli::parse_topology_string(&topology).map_err(|e| format!("{e:#}"))?;

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
        disk_cfg,
    )
    .map_err(|e| format!("{e:#}"))
}
