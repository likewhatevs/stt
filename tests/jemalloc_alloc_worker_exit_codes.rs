//! Host-side exit-code contract tests for
//! `ktstr-jemalloc-alloc-worker`.
//!
//! These tests spawn the alloc-worker binary directly via
//! `Command::new` and assert the exit-code contract spelled out in
//! the worker's module doc. No VM, no probe — pure host-side
//! exercise of the fail-fast branches. Runs in well under a second
//! per test.
//!
//! Lives in its own integration-test file (rather than alongside the
//! VM-based `jemalloc_probe_tests.rs`) because the ktstr early-
//! dispatch ctor at `test_support::dispatch::ktstr_test_early_dispatch`
//! intercepts `--list` / `--exact` in nextest protocol mode when the
//! linked binary contains any real `#[ktstr_test]` entries. That
//! intercept emits only the ktstr variants and hides plain `#[test]`
//! functions. This file carries no `#[ktstr_test]` entries so
//! `KTSTR_TESTS.iter().any(e => e.name != "__unit_test_dummy__")`
//! returns false, the intercept is skipped, and the standard
//! rustc test harness picks up the `#[test]` functions below.

/// bytes=0 must exit with code 2. The worker's module doc pins
/// `2: bytes == 0`; this test catches any refactor that silently
/// re-routes the zero-size alloc guard to a different code or drops
/// it entirely.
#[test]
fn worker_exits_2_on_bytes_zero() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    let output = std::process::Command::new(worker)
        .arg("0")
        .output()
        .expect("spawn worker");
    assert_eq!(
        output.status.code(),
        Some(2),
        "bytes=0 must exit with code 2; got {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Missing positional `<BYTES>` must exit with code 5 (argument
/// parse failure). Covers the argv-absent branch of the
/// `expect() → exit(5)` refactor.
#[test]
fn worker_exits_5_on_missing_bytes_arg() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    let output = std::process::Command::new(worker)
        .output()
        .expect("spawn worker");
    assert_eq!(
        output.status.code(),
        Some(5),
        "missing BYTES must exit with code 5; got {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Non-numeric `<BYTES>` must exit with code 5. Covers the
/// parse-error branch.
#[test]
fn worker_exits_5_on_non_numeric_bytes_arg() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    let output = std::process::Command::new(worker)
        .arg("not-a-number")
        .output()
        .expect("spawn worker");
    assert_eq!(
        output.status.code(),
        Some(5),
        "non-numeric BYTES must exit with code 5; got {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Ready-marker write failure must exit with code 4. Uses the
/// `KTSTR_WORKER_READY_MARKER_OVERRIDE` test-only env hook to point
/// the write at a path under a non-existent parent directory, which
/// `std::fs::write`'s internal `open(..., O_CREAT)` can't create →
/// ENOENT → exit 4. Bypasses the race-prone alternative of
/// pre-creating a directory at the pid-scoped default path. Passes
/// `1024` as BYTES so the self-check + allocation succeed; the
/// ready-marker write is the first failure the worker hits.
#[test]
fn worker_exits_4_on_ready_marker_write_fail() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    let output = std::process::Command::new(worker)
        .arg("1024")
        .env(
            "KTSTR_WORKER_READY_MARKER_OVERRIDE",
            "/nonexistent-ktstr-test-dir/marker",
        )
        .output()
        .expect("spawn worker");
    assert_eq!(
        output.status.code(),
        Some(4),
        "ready-marker write failure must exit with code 4; got {:?}; stderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to write ready marker"),
        "stderr must name the failure; got: {stderr}",
    );
}
