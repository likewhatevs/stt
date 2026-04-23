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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_helpers::capture_stderr;

    /// `test_skip` must emit a line matching `ktstr: SKIP: <reason>`
    /// to stderr. Pins the prefix, separator, and trailing newline.
    ///
    /// The task motivating this test: nextest shows `PASS` for a
    /// test body that returns `AssertResult::skip` WITHOUT going
    /// through an `eprintln!` path — the skip verdict is structural
    /// (the AssertResult variant) and surfaces in the sidecar JSON,
    /// but stats-tooling and operators grepping stderr for the
    /// `ktstr: SKIP:` banner only see the line when
    /// `test_skip` / the `skip!` macro fires. This test pins the
    /// banner format so consumer tools have a stable anchor.
    #[test]
    fn test_skip_emits_canonical_stderr_banner() {
        let (_, bytes) = capture_stderr(|| {
            test_skip("topology mismatch: requires 4 cores, got 2");
        });
        let text = std::str::from_utf8(&bytes).expect("stderr is UTF-8");
        assert_eq!(
            text,
            "ktstr: SKIP: topology mismatch: requires 4 cores, got 2\n",
            "expected canonical banner with trailing newline",
        );
    }

    /// A skip with an empty reason still emits the prefix + colon +
    /// separator. Pins the format against a future regression that
    /// short-circuits on empty input (e.g. `if reason.is_empty()
    /// { return; }`) — silent skips are the anti-pattern this
    /// helper exists to prevent.
    #[test]
    fn test_skip_with_empty_reason_still_emits_banner() {
        let (_, bytes) = capture_stderr(|| {
            test_skip("");
        });
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text, "ktstr: SKIP: \n");
    }

    /// The banner must START with `ktstr: SKIP:` so grep-based
    /// consumers can anchor the line with `^ktstr: SKIP:`. Pins the
    /// left-anchor explicitly so a future refactor that wraps the
    /// emission in a tracing layer or a prefix-prepending wrapper
    /// would fail this test before silently breaking downstream
    /// parsers.
    #[test]
    fn test_skip_stderr_line_is_left_anchored() {
        let (_, bytes) = capture_stderr(|| {
            test_skip("anchor-check reason");
        });
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(
            text.starts_with("ktstr: SKIP:"),
            "banner must be left-anchored; got: {text:?}",
        );
    }
}
