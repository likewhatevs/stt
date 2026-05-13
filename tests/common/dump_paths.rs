//! Shared path helpers for failure-dump E2E tests.
//!
//! `test_support::eval::run_ktstr_test_inner` attaches a fixed
//! per-test failure-dump path on every VM builder it constructs
//! (`{sidecar_dir()}/{test_name}.failure-dump.json`). Tests that
//! read back the JSON need to recompute the same path. Centralising
//! the format string here keeps every reader site in sync with the
//! attachment site — if the framework's naming convention ever
//! shifts, only this helper updates.

use ktstr::test_support::sidecar_dir;

/// Compute the per-test failure-dump path.
///
/// Mirrors `test_support::eval::run_ktstr_test_inner`'s attachment
/// path. The framework attaches this path on every VM builder it
/// constructs; tests that read the dump back must use the same
/// path. Format: `{sidecar_dir()}/{test_name}.failure-dump.json`.
///
/// `#[allow(dead_code)]` because not every integration-test binary
/// imports this helper; Rust compiles the `common/` tree once per
/// integration target and flags unused fns per target.
#[allow(dead_code)]
pub fn failure_dump_path(test_name: &str) -> std::path::PathBuf {
    sidecar_dir().join(format!("{test_name}.failure-dump.json"))
}
