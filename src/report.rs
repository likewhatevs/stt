//! Centralized test-reporting helpers.
//!
//! Single source of truth for the `ktstr: SKIP: <reason>` format
//! consumed by stats tooling and nextest harness output parsers. A
//! prefix change is a one-file edit here; the alternative — 15+
//! open-coded `eprintln!` sites across the crate — drifts into
//! inconsistent casings (SKIP vs skipping vs skip) that break every
//! grep-based test-summary tool.

use std::fmt;

/// Emit a canonical test-skip line to stderr: `ktstr: SKIP: <reason>`.
///
/// Prints only — callers drive their own control flow. Some sites
/// early-return `()`, some return `Err` / `Ok(...)` / `None`, some
/// `continue` out of a label-driven loop; a single return-forcing
/// macro cannot serve them all. The [`skip!`](crate::skip) macro in
/// `src/test_macros.rs` wraps this call with a trailing `return;`
/// for `fn() -> ()` test helpers.
pub(crate) fn test_skip(reason: impl fmt::Display) {
    eprintln!("ktstr: SKIP: {reason}");
}
