//! `cargo ktstr model {fetch,status,clean}` — LLM model cache
//! management.
//!
//! Wraps the three model-cache primitives in
//! [`ktstr::test_support`]:
//! - [`run_model_fetch`] = `ensure(&DEFAULT_MODEL)` with operator-
//!   readable progress + final cache-path line.
//! - [`run_model_status`] = `status(&DEFAULT_MODEL)` rendered into
//!   four `ShaVerdict` arms (NotCached, CheckFailed, Mismatches,
//!   Matches), each with a remediation hint where applicable.
//! - [`run_model_clean`] = `clean(&DEFAULT_MODEL)` rendered as a
//!   per-file deletion log + total-freed summary.
//!
//! All three share `DEFAULT_MODEL` as the cache key — the binary
//! today only manages the single pinned LLM artifact backing
//! `OutputFormat::LlmExtract`. Future per-model variants would
//! lift the spec selection into the CLI surface.

/// `cargo ktstr model fetch` — download + SHA-check the default model
/// into the user's cache. Wraps `ktstr::test_support::ensure` with a
/// human-readable progress line; the status is printed after so
/// users can see the final cache path regardless of whether the
/// fetch did any work.
pub(crate) fn run_model_fetch() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    match ktstr::test_support::ensure(&spec) {
        Ok(path) => {
            println!(
                "ktstr: model '{}' ready at {}",
                spec.file_name,
                path.display()
            );
            Ok(())
        }
        Err(e) => Err(format!("fetch model '{}': {e:#}", spec.file_name)),
    }
}

/// `cargo ktstr model status` — report the cache path and whether a
/// SHA-checked copy of the default model is already present.
pub(crate) fn run_model_status() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    let status = ktstr::test_support::status(&spec).map_err(|e| format!("{e:#}"))?;
    println!("model:    {}", status.spec.file_name);
    println!("path:     {}", status.path.display());
    println!("cached:   {}", status.sha_verdict.is_cached());
    println!("checked:  {}", status.sha_verdict.is_match());
    // Distinguish the four verdict variants so each gets a
    // remediation-specific line: absent cache, I/O failure during
    // the SHA check, successful hash that didn't match, and the
    // all-clear case (no annotation needed). An I/O failure points
    // at the filesystem entry (permissions, truncation); a mismatch
    // points at the bytes themselves.
    // "Re-fetch to replace it" is the shared remediation tail for
    // every non-Matches cached-file branch (both CheckFailed and
    // Mismatches land on the same operator action — the cause
    // differs but the fix does not). Factoring the tail into one
    // string keeps the two arms in lock-step so a wording change
    // lands in both places by construction.
    const RE_FETCH_TAIL: &str = "re-fetch to replace it";
    match &status.sha_verdict {
        ktstr::test_support::ShaVerdict::NotCached => println!(
            "(no cached copy — run `cargo ktstr model fetch` to download {} MiB)",
            status.spec.size_bytes / 1024 / 1024,
        ),
        ktstr::test_support::ShaVerdict::CheckFailed(err) => {
            // Defensively collapse any embedded `\n` into `; `
            // before placing `err` inside the "(single
            // parenthesized note)" wrapper. The alternate
            // anyhow format (`{e:#}`) that produced `err` joins
            // causes with `: ` and is single-line in practice;
            // this replace is a guard against a future error
            // source whose Display impl injects its own
            // newlines (std::io errors wrapping multi-line OS
            // messages, third-party crates formatting
            // call-chain trees). Keeping the output on one
            // line preserves the visual grouping the other
            // match arms use.
            let single_line = err.replace('\n', "; ");
            println!(
                "(cached file could not be checked: {single_line}; \
                 inspect the cache entry or {RE_FETCH_TAIL})",
            );
        }
        ktstr::test_support::ShaVerdict::Mismatches => {
            println!("(cached file failed SHA-256 check; {RE_FETCH_TAIL})",);
        }
        ktstr::test_support::ShaVerdict::Matches => {}
    }
    Ok(())
}

/// `cargo ktstr model clean` — remove the cached GGUF artifact and
/// its `.mtime-size` sidecar. Wraps
/// `ktstr::test_support::clean` and renders the resulting
/// [`ktstr::test_support::CleanReport`] as a per-file deletion
/// log plus a total-freed summary line. Sizes pass through
/// `indicatif::HumanBytes` for IEC-prefixed rendering ("2.34 GiB",
/// "52 B") consistent with the rest of the model surface.
///
/// Empty-cache case (neither file present) prints a single
/// "no cached model found" line so an idempotent re-run after a
/// successful clean produces a clear "nothing to do" outcome
/// rather than two "(absent)" lines.
pub(crate) fn run_model_clean() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    let report =
        ktstr::test_support::clean(&spec).map_err(|e| format!("clean model cache: {e:#}"))?;
    if report.is_empty() {
        println!(
            "no cached model found at {}",
            report.artifact_path.display()
        );
        return Ok(());
    }
    if let Some(bytes) = report.artifact_freed_bytes {
        println!(
            "removed {} ({})",
            report.artifact_path.display(),
            indicatif::HumanBytes(bytes),
        );
    }
    if let Some(bytes) = report.sidecar_freed_bytes {
        println!(
            "removed {} ({})",
            report.sidecar_path.display(),
            indicatif::HumanBytes(bytes),
        );
    }
    println!(
        "freed {} total",
        indicatif::HumanBytes(report.total_freed_bytes()),
    );
    Ok(())
}
