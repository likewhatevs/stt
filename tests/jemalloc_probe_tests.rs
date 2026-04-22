//! Closed-loop validation of the jemalloc TLS probe
//! (`src/bin/jemalloc_probe.rs`) inside a ktstr VM.
//!
//! The probe's `--self-test <BYTES>` mode spawns an allocator
//! thread inside the probe process, reads the worker's
//! `thread_allocated` counter via `process_vm_readv` on its own
//! pid (no ptrace needed for same-process reads), and exits 0 iff
//! the observed counter is at least the known allocation size.
//! This approach keeps DWARF inside the probe binary — which is
//! never stripped — and avoids needing DWARF on the much larger
//! ktstr-init binary that runs as PID 1 inside the VM.
//!
//! The probe binary reaches the guest via the initramfs wiring
//! activated by the `KTSTR_PROBE_BINARY` env var, set by
//! [`set_probe_binary_env_var`] at static init time. The init
//! binary stays stripped; the probe carries its own DWARF and
//! self-probes.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Check, OutputFormat, Payload, PayloadKind};

// ---------------------------------------------------------------------------
// Probe-binary env var setup
// ---------------------------------------------------------------------------

/// Run at static init before `#[ktstr_test]` macros register
/// their entries. Sets `KTSTR_PROBE_BINARY` to the absolute host
/// path of `ktstr-jemalloc-probe` so the ktstr test harness
/// (`build_vm_builder_base`) packs the probe into every VM's
/// initramfs as `/bin/ktstr-jemalloc-probe` (on the guest `PATH`).
///
/// `env!` resolves at compile time against cargo's integration-test
/// env, so the path is pinned to whichever profile compiled this
/// file (dev or release). `std::env::set_var` is marked `unsafe`
/// under edition 2024 because it races with concurrent env reads;
/// ctors run before any thread spawns, so the call is race-free in
/// practice.
#[::ktstr::__private::ctor::ctor(crate_path = ::ktstr::__private::ctor)]
fn set_probe_binary_env_var() {
    unsafe {
        std::env::set_var(
            "KTSTR_PROBE_BINARY",
            env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
        );
    }
}

// ---------------------------------------------------------------------------
// Payload fixtures
// ---------------------------------------------------------------------------

/// Self-test probe invocation. `Check::ExitCodeEq(0)` gates a
/// non-zero probe exit as a failing AssertResult; the probe's
/// `--self-test` mode exits 0 iff the observed `thread_allocated`
/// counter is at least the known allocation size.
static JEMALLOC_PROBE_SELFTEST: Payload = Payload {
    name: "jemalloc_probe_selftest",
    kind: PayloadKind::Binary("ktstr-jemalloc-probe"),
    output: OutputFormat::Json,
    default_args: &[],
    default_checks: &[Check::ExitCodeEq(0)],
    metrics: &[],
};

/// External-pid probe invocation without exit-code gating. Used by
/// the error-path test that deliberately probes a non-jemalloc
/// target and reads `metrics.exit_code` directly.
static JEMALLOC_PROBE_NO_EXIT_CHECK: Payload = Payload {
    name: "jemalloc_probe_no_exit_check",
    kind: PayloadKind::Binary("ktstr-jemalloc-probe"),
    output: OutputFormat::Json,
    default_args: &[],
    default_checks: &[],
    metrics: &[],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a failing `AssertResult` from a message. Kept local to
/// this file because `AssertResult` exposes `pass` / `skip`
/// constructors but not a generic fail constructor — tests that
/// want to return a diagnostic-only failure assemble the struct by
/// hand.
fn fail_result(msg: impl Into<String>) -> AssertResult {
    AssertResult {
        passed: false,
        skipped: false,
        details: vec![AssertDetail::new(DetailKind::Other, msg)],
        stats: Default::default(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Single-worker closed loop via the probe's `--self-test` mode.
/// 16 MiB is large enough that jemalloc routes the allocation
/// through the huge path, which unconditionally updates
/// `thread_allocated` per the #479 design review. The probe
/// observes its own process memory (no ptrace required for
/// same-process `process_vm_readv`) and exits 0 iff the observed
/// counter is at least 16 MiB.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_single_worker_observes_known_allocation(ctx: &Ctx) -> Result<AssertResult> {
    const KNOWN_BYTES: u64 = 16 * 1024 * 1024;
    let (assert_result, _metrics) = ctx
        .payload(&JEMALLOC_PROBE_SELFTEST)
        .arg("--self-test")
        .arg(KNOWN_BYTES.to_string())
        .run()?;
    Ok(assert_result)
}

/// Error path — probe a pid that does not exist. The probe must
/// exit non-zero; the test reads `exit_code` directly rather
/// than gating via `Check::ExitCodeEq(0)` because the expected
/// outcome is failure. Uses [`JEMALLOC_PROBE_NO_EXIT_CHECK`] so
/// the framework does not mark the `AssertResult` as failed when
/// the probe returns non-zero.
///
/// Targets a deliberately-large pid (999_999_999) that cannot
/// realistically be allocated by the kernel's pid allocator. A
/// `#[ktstr_test]` VM carries only the test binary at `/init`
/// plus the probe at `/bin/ktstr-jemalloc-probe` — no busybox
/// applets, no scheduler other than whatever the test spec
/// declares — so spawning a non-jemalloc workload would require
/// more initramfs wiring than the error-path test is worth.
/// Probing a nonexistent pid exercises the probe's `"pid X does
/// not exist"` fatal path, which is the same `RunOutcome::Fatal`
/// branch a probe-against-busybox run would hit after
/// `find_jemalloc_via_maps` fails.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_rejects_non_jemalloc_target(ctx: &Ctx) -> Result<AssertResult> {
    let fake_pid: i32 = 999_999_999;
    let (_assert, metrics) = ctx
        .payload(&JEMALLOC_PROBE_NO_EXIT_CHECK)
        .arg("--pid")
        .arg(fake_pid.to_string())
        .arg("--json")
        .run()?;

    if metrics.exit_code == 0 {
        return Ok(fail_result(format!(
            "probe exit_code=0 against nonexistent pid {fake_pid}; \
             probe must fail with non-zero exit on an invalid target"
        )));
    }
    Ok(AssertResult::pass())
}
