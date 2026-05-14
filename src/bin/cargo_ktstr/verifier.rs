//! `cargo ktstr verifier` subcommand: thin wrapper around
//! `cargo nextest run` filtered to the `verifier/` test-name prefix.
//!
//! Each test binary that links ktstr-test-support and has at least
//! one `declare_scheduler!` declaration emits one nextest test per
//! (declared scheduler × declared kernel × accepted gauntlet preset)
//! cell. The lister + cell handler live in
//! `src/test_support/dispatch.rs::list_verifier_cells_all` and
//! `run_verifier_cell`. Nextest provides per-cell parallelism,
//! retries, and failure isolation; this dispatcher only resolves the
//! `--kernel` argument into the existing `KTSTR_KERNEL` /
//! `KTSTR_KERNEL_LIST` env-var protocol the test binary already
//! consumes, then exec's nextest with a verifier-prefix filter
//! expression.

use std::process::Command;

use crate::kernel::{encode_kernel_list, resolve_kernel_set};

/// Dispatch the `cargo ktstr verifier` subcommand.
pub(crate) fn run_verifier(kernel: Vec<String>, _raw: bool) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["nextest", "run", "-E", "test(/^verifier/)"]);

    if !kernel.is_empty() {
        let resolved = resolve_kernel_set(&kernel)?;
        if resolved.is_empty() {
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        let first_dir = &resolved[0].1;
        cmd.env(ktstr::KTSTR_KERNEL_ENV, first_dir);
        if resolved.len() > 1 {
            let encoded = encode_kernel_list(&resolved)?;
            eprintln!(
                "cargo ktstr verifier: fanning across {n} kernels",
                n = resolved.len(),
            );
            cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
        }
    }

    let status = cmd
        .status()
        .map_err(|e| format!("spawn cargo nextest run: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "cargo nextest run exited with {}",
            status
                .code()
                .map_or("signal".to_string(), |c| c.to_string()),
        ))
    }
}
