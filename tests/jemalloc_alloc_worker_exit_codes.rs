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

use ktstr::worker_ready::{WORKER_READY_MARKER_OVERRIDE_ENV, WORKER_STDERR_PREFIX};

/// Render a `std::process::ExitStatus` as a human-actionable string
/// for assertion-failure diagnostics.
///
/// The default `Debug` / `{:?}` for `status.code()` collapses every
/// signal-kill to a bare `None`, which strips the single most
/// important fact a failing test needs: whether the worker was
/// terminated by a signal at all and, if so, which one. A reader
/// staring at `got None; stderr: ""` in CI output cannot
/// distinguish SIGSEGV from SIGKILL from a genuinely-missing exit
/// code, and must cross-reference the binary's behavior to decide
/// whether the failure is a crash or an orderly signal-kill.
///
/// This helper produces one of:
/// - `"exit code N"` when `status.code()` is `Some(N)` — the
///   normal setup-failure path documented in the worker's "Exit
///   codes" legend.
/// - `"signal-killed (signal N)"` when `status.code()` is `None`
///   and `ExitStatusExt::signal()` yields `Some(N)` on unix.
/// - `"signal-killed"` when both are `None` (the non-unix fallback
///   / defense-in-depth — unreachable on the Linux test platform
///   but kept so the helper compiles everywhere and never panics).
fn format_exit_status(status: std::process::ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("exit code {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return format!("signal-killed (signal {sig})");
        }
    }
    "signal-killed".to_string()
}

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
        "bytes=0 must exit with code 2; got {}; stderr: {}",
        format_exit_status(output.status),
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
        "missing BYTES must exit with code 5; got {}; stderr: {}",
        format_exit_status(output.status),
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
        "non-numeric BYTES must exit with code 5; got {}; stderr: {}",
        format_exit_status(output.status),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// Ready-marker write failure must exit with code 4. Uses the
/// [`ktstr::worker_ready::WORKER_READY_MARKER_OVERRIDE_ENV`]
/// test-only env hook to point the write at a path under a
/// non-existent parent directory, which `std::fs::write`'s internal
/// `open(..., O_CREAT)` can't create → ENOENT → exit 4. Bypasses
/// the race-prone alternative of pre-creating a directory at the
/// pid-scoped default path. Passes `1024` as BYTES so the
/// self-check + allocation succeed; the ready-marker write is the
/// first failure the worker hits.
#[test]
fn worker_exits_4_on_ready_marker_write_fail() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    let output = std::process::Command::new(worker)
        .arg("1024")
        .env(
            WORKER_READY_MARKER_OVERRIDE_ENV,
            "/nonexistent-ktstr-test-dir/marker",
        )
        // Pin MALLOC_CONF to background_thread:false so that an
        // operator with the opposite setting in their shell (or a
        // sibling test that set it on its own invocation and had the
        // state leak through an inheritance path we haven't caught)
        // cannot race the worker into exiting 3 (thread count != 1)
        // before it reaches the ready-marker branch we're trying to
        // assert. Without this pin, a stray MALLOC_CONF would make
        // this test flaky in exactly the conditions that
        // `worker_exits_3_on_thread_count_not_one` deliberately
        // exercises. Setting both the generic and tikv-jemallocator
        // prefixed forms mirrors worker_exits_3's rationale (the
        // `_rjem_` symbol prefix gates which variant the in-process
        // allocator reads).
        .env("MALLOC_CONF", "background_thread:false")
        .env("_RJEM_MALLOC_CONF", "background_thread:false")
        .output()
        .expect("spawn worker");
    assert_eq!(
        output.status.code(),
        Some(4),
        "ready-marker write failure must exit with code 4; got {}; stderr: {}",
        format_exit_status(output.status),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to write ready marker"),
        "stderr must name the failure; got: {stderr}",
    );
}

/// `/proc/self/task` thread count != 1 must exit with code 3. The
/// worker's default mode rejects any silent extra thread (background
/// allocator threads, a runtime pulled in by a new dep, etc.) via
/// the single-thread self-check in `main` before the allocation is
/// materialized. Forcing that branch from the host side requires
/// the worker to start with a helper thread already alive at the
/// self-check; the cleanest way without patching the binary is to
/// opt into jemalloc's background-thread worker via
/// `background_thread:true`, which spawns the helper during
/// allocator init (before `main` reads `/proc/self/task`).
///
/// The env var is set under both the generic `MALLOC_CONF` name and
/// the tikv-jemallocator runtime-prefix alias `_RJEM_MALLOC_CONF`.
/// tikv-jemallocator's default build prefixes the symbol table with
/// `_rjem_` (the `unprefixed_malloc_on_supported_platforms` Cargo
/// feature is NOT enabled in this workspace — see `Cargo.toml`'s
/// `tikv-jemallocator = { version = "0.6", features = ["stats"] }`
/// stanza), so the generic `MALLOC_CONF` is not read by the
/// in-process jemalloc copy. Setting both variants keeps the test
/// robust against a future feature flip that unprefixes the symbols.
///
/// # Dependency on jemalloc-init-via-`std::env::args()`
///
/// This test's correctness rests on an implicit invariant in the
/// worker binary's `main`: the FIRST call into any jemalloc-backed
/// code path must occur AFTER the process's environment is
/// readable. In the current worker, that first call is implicit —
/// `std::env::args().skip(1).collect::<Vec<String>>()` at the top
/// of `main` allocates a `Vec<String>`, which goes through
/// `tikv_jemallocator::Jemalloc` (the `#[global_allocator]`) and
/// forces jemalloc to initialize on the spot. The initializer
/// reads `MALLOC_CONF` / `_RJEM_MALLOC_CONF` via
/// `getenv()` / `__environ` exactly once during that first
/// allocation, sees `background_thread:true`, and spawns the
/// helper thread as part of init. By the time the worker reaches
/// the `/proc/self/task` self-check, the helper is live, the
/// thread count is ≥ 2, and the exit-3 branch fires.
///
/// A future refactor that (a) marks the env read as pre-main via
/// a `ctor::ctor` constructor, (b) moves argv parsing into a
/// no-alloc path (e.g. `argv.iter()` on a raw `&[&str]` provided
/// by a shim), or (c) adds an `unsafe extern "C" fn main` that
/// bypasses the Rust runtime's env initialization would BREAK this
/// test in a subtle way: jemalloc would still initialize on some
/// later allocation, but by then the env read could race the
/// `/proc/self/task` scan and produce a flaky exit 3 ↔ exit 0
/// result depending on thread scheduling. If you are the author of
/// such a refactor, update this test to force the first allocation
/// explicitly (e.g. via a `let _ = Vec::<u8>::with_capacity(1)` at
/// the top of `main` under a `// jemalloc-init probe` comment) or
/// switch to a more robust forcing mechanism.
#[test]
fn worker_exits_3_on_thread_count_not_one() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    // MALLOC_CONF is LOAD-BEARING here: the `background_thread:true`
    // setting is the only reason the worker ever reaches the exit-3
    // branch. Without this env, jemalloc starts in single-thread
    // mode, `/proc/self/task` has exactly one entry, and the
    // self-check passes — the test would then fail because the
    // worker proceeded past the guard we are trying to exercise.
    // Contrast with `worker_exits_4_on_ready_marker_write_fail` and
    // `worker_stderr_lines_share_centralized_prefix`, which set
    // `background_thread:false` as BELT-AND-SUSPENDERS — defensive
    // pins against a leaking env var from an operator's shell or a
    // sibling test, not a prerequisite for the branch under test.
    let output = std::process::Command::new(worker)
        .arg("1024")
        .env("MALLOC_CONF", "background_thread:true")
        .env("_RJEM_MALLOC_CONF", "background_thread:true")
        .output()
        .expect("spawn worker");
    assert_eq!(
        output.status.code(),
        Some(3),
        "background_thread:true must spawn a helper thread before \
         the /proc/self/task self-check and exit 3; got {}; stderr: {}",
        format_exit_status(output.status),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("/proc/self/task has"),
        "stderr must name the self-check that fired; got: {stderr}",
    );
}

/// Every fail-fast stderr line the worker emits must start with the
/// shared [`WORKER_STDERR_PREFIX`]. Pins the "one source of truth
/// for the worker's stderr prefix" contract: a literal-vs-const
/// drift — someone retypes `"jemalloc-alloc-worker:"` with a typo,
/// or omits it on a new eprintln! — would have this assertion
/// trip on the specific failure path. Drives one failure mode per
/// exit code the binary can produce from the host side (missing
/// argv → 5, bytes=0 → 2, bad marker path → 4) so every stderr-
/// emitting branch is sampled at least once.
#[test]
fn worker_stderr_lines_share_centralized_prefix() {
    let worker = env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker");
    // Exit 5: missing BYTES.
    let output = std::process::Command::new(worker)
        .output()
        .expect("spawn worker (missing-bytes case)");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.starts_with(WORKER_STDERR_PREFIX),
        "missing-BYTES stderr must start with WORKER_STDERR_PREFIX ({WORKER_STDERR_PREFIX:?}); \
         got: {stderr}",
    );
    // Exit 2: bytes=0.
    let output = std::process::Command::new(worker)
        .arg("0")
        .output()
        .expect("spawn worker (bytes=0 case)");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.starts_with(WORKER_STDERR_PREFIX),
        "bytes=0 stderr must start with WORKER_STDERR_PREFIX ({WORKER_STDERR_PREFIX:?}); \
         got: {stderr}",
    );
    // Exit 4: marker-write failure via the override env var.
    // Pin MALLOC_CONF to background_thread:false for the same
    // reason `worker_exits_4_on_ready_marker_write_fail` does — a
    // leaking background_thread:true setting would race this case
    // into exit 3 before the marker write is attempted.
    let output = std::process::Command::new(worker)
        .arg("1024")
        .env(
            WORKER_READY_MARKER_OVERRIDE_ENV,
            "/nonexistent-ktstr-test-dir/marker",
        )
        .env("MALLOC_CONF", "background_thread:false")
        .env("_RJEM_MALLOC_CONF", "background_thread:false")
        .output()
        .expect("spawn worker (marker-write case)");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.starts_with(WORKER_STDERR_PREFIX),
        "marker-write stderr must start with WORKER_STDERR_PREFIX ({WORKER_STDERR_PREFIX:?}); \
         got: {stderr}",
    );
}
