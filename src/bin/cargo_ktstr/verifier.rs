//! `cargo ktstr verifier` subcommand: thin wrapper around
//! `cargo nextest run` filtered to the `verifier/` test-name prefix.
//!
//! Each test binary that links ktstr-test-support and has at least
//! one `declare_scheduler!` declaration emits one nextest test per
//! (declared scheduler × kernel-list entry × accepted gauntlet
//! preset) cell. The lister + cell handler live in
//! `src/test_support/dispatch.rs::list_verifier_cells_all` and
//! `run_verifier_cell`. Nextest provides per-cell parallelism,
//! retries, and failure isolation; this dispatcher resolves the
//! `--kernel` argument into the `KTSTR_KERNEL_LIST` env-var matrix
//! dimension the test binary's lister walks, plumbs `--raw` via
//! `KTSTR_VERIFIER_RAW`, and spawns nextest with a verifier-prefix
//! filter expression.
//!
//! `KTSTR_KERNEL_LIST` is ALWAYS populated by this dispatcher — even
//! with no `--kernel` flag the dispatcher auto-discovers one kernel
//! and synthesizes a single-entry list with a path-derived label.
//! That keeps the test-binary cell handler's lookup path unified
//! (always look up by label in the list, never fall through to a
//! resolve_test_kernel single-kernel fallback that would silently
//! run a cell against an unrelated kernel).

use std::path::PathBuf;
use std::process::Command;

use crate::kernel::{
    encode_kernel_list, path_kernel_label, resolve_kernel_image, resolve_kernel_set,
};

/// Dispatch the `cargo ktstr verifier` subcommand.
pub(crate) fn run_verifier(kernel: Vec<String>, raw: bool) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(["nextest", "run", "-E", "test(/^verifier/)"]);

    if raw {
        cmd.env(ktstr::KTSTR_VERIFIER_RAW_ENV, "1");
    }

    // Always produce a non-empty kernel list. When --kernel is
    // omitted, auto-discover one kernel and synthesize a single
    // entry with a path-basename label. The test-binary cell
    // handler keys on this list as its single source of truth.
    let resolved: Vec<(String, PathBuf)> = if !kernel.is_empty() {
        let r = resolve_kernel_set(&kernel)?;
        if r.is_empty() {
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        r
    } else {
        let path = resolve_kernel_image(None)?;
        let label = path_kernel_label(&path);
        vec![(label, path)]
    };

    cmd.env(ktstr::KTSTR_KERNEL_ENV, &resolved[0].1);
    let encoded = encode_kernel_list(&resolved)?;
    cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
    let kernel_count = resolved.len();

    eprintln!(
        "cargo ktstr verifier: dispatching to nextest with filter test(/^verifier/) \
         on {kernel_count} resolved kernel(s){raw}",
        raw = if raw { " (raw output)" } else { "" },
    );

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
