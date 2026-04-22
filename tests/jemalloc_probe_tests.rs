//! Closed-loop validation of the jemalloc TLS probe
//! (`src/bin/jemalloc_probe.rs`) inside a ktstr VM.
//!
//! Every test in this file:
//! 1. Runs inside the guest as part of the ktstr-init process,
//!    which links `tikv_jemallocator::Jemalloc` as its global
//!    allocator (see `src/bin/ktstr.rs:2`).
//! 2. Spawns one or more worker threads that allocate a known
//!    number of bytes and signal ready AFTER the allocation
//!    completes.
//! 3. Launches `ktstr-jemalloc-probe --pid <OWN_PID> --json` as a
//!    foreground `Payload::Binary`, then parses the JSON output
//!    (flattened via `walk_json_leaves` into `threads.N.tid` /
//!    `threads.N.allocated_bytes`) and asserts on the worker's
//!    entry by tid match.
//!
//! The probe binary reaches the guest via the initramfs wiring
//! activated by the `KTSTR_PROBE_BINARY` env var, set by
//! [`set_probe_binary_env_var`] at static init time. The init
//! binary ships with DWARF preserved (see
//! `KtstrVmBuilder::preserve_init_dwarf`) so the probe can
//! resolve `tsd_s.thread_allocated` / `tsd_s.thread_deallocated`
//! offsets via gimli.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use anyhow::{Result, anyhow};
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::scenario::payload_run::PayloadHandle;
use ktstr::test_support::{Check, OutputFormat, Payload, PayloadKind, PayloadMetrics};

// ---------------------------------------------------------------------------
// Probe-binary env var setup
// ---------------------------------------------------------------------------

/// Run at static init before `#[ktstr_test]` macros register
/// their entries. Sets `KTSTR_PROBE_BINARY` to the absolute host
/// path of `ktstr-jemalloc-probe` so the ktstr test harness
/// (`build_vm_builder_base`) packs the probe into every VM's
/// initramfs and preserves the init binary's DWARF.
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

/// Primary probe payload. `Check::ExitCodeEq(0)` gates non-zero
/// exits as failures so tests that expect probe success surface
/// the probe's own error message on the failing path.
static JEMALLOC_PROBE: Payload = Payload {
    name: "jemalloc_probe",
    kind: PayloadKind::Binary("ktstr-jemalloc-probe"),
    output: OutputFormat::Json,
    default_args: &[],
    default_checks: &[Check::ExitCodeEq(0)],
    metrics: &[],
};

/// Variant that does NOT gate on exit code; used by the error-path
/// test that deliberately probes a non-jemalloc target and
/// inspects the exit code directly.
static JEMALLOC_PROBE_NO_EXIT_CHECK: Payload = Payload {
    name: "jemalloc_probe_no_exit_check",
    kind: PayloadKind::Binary("ktstr-jemalloc-probe"),
    output: OutputFormat::Json,
    default_args: &[],
    default_checks: &[],
    metrics: &[],
};

/// Background workload for the error-path test — busybox `sleep`
/// with no jemalloc. Invoked via the absolute path `/bin/busybox`
/// rather than a bare `sleep` because the test-dispatch init
/// (src/vmm/rust_init.rs:183-188) installs busybox applet
/// symlinks ONLY in shell mode, not for `#[ktstr_test]` VMs — so
/// `/bin/sleep` does not exist in this VM. `busybox` dispatches
/// to its `sleep` applet when invoked with `sleep` as argv[1].
/// The `60` arg is longer than any realistic probe run; the test
/// kills it explicitly after probing.
static BUSYBOX_SLEEP: Payload = Payload {
    name: "busybox_sleep",
    kind: PayloadKind::Binary("/bin/busybox"),
    output: OutputFormat::ExitCode,
    default_args: &["sleep", "60"],
    default_checks: &[],
    metrics: &[],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Spawn a worker thread that allocates exactly `bytes` on itself
/// and blocks until a stop signal arrives. Returns the
/// ready-receiver, the stop-sender, the worker's join handle, and
/// the worker's TID (populated after `ready_rx.recv()` returns).
///
/// The allocation is held live under `std::hint::black_box` so the
/// optimizer cannot elide it before the probe reads the counters.
/// The TID is recorded into an `AtomicI32` with `Release` ordering
/// BEFORE the ready signal fires; the main thread reads it with
/// `Acquire` ordering AFTER the ready recv, which matches the
/// channel's happens-before guarantee.
fn spawn_allocator_worker(
    bytes: usize,
) -> (
    std::sync::mpsc::Receiver<()>,
    std::sync::mpsc::Sender<()>,
    std::thread::JoinHandle<()>,
    Arc<AtomicI32>,
) {
    let tid = Arc::new(AtomicI32::new(0));
    let tid_clone = tid.clone();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<()>();
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let handle = std::thread::spawn(move || {
        let self_tid = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
        tid_clone.store(self_tid, Ordering::Release);
        let known: Vec<u8> = vec![0u8; bytes];
        std::hint::black_box(&known);
        // Ready is signaled AFTER the allocation completes so the
        // probe always reads stable state. If the main thread
        // drops `ready_rx` or `stop_tx` without completing the
        // handshake (e.g. it returned Err from `.run()?` or
        // panicked), both `send` and `recv` become `Err` cases —
        // swallow them silently so the worker exits cleanly and
        // its drop does not cascade a second panic into the
        // ktstr panic hook (which reboots the VM on first panic).
        let _ = ready_tx.send(());
        let _ = stop_rx.recv();
        drop(known);
    });
    (ready_rx, stop_tx, handle, tid)
}

/// Extract the `allocated_bytes` value for `worker_tid` from the
/// flat metric list produced by `walk_json_leaves` over the probe's
/// JSON output.
///
/// The probe emits
/// `{"pid": N, "threads": [{"tid": T, "allocated_bytes": A, "deallocated_bytes": D}, ...]}`
/// which flattens to `threads.N.tid` / `threads.N.allocated_bytes`
/// per index. This scans for the array entry whose `tid` equals
/// `worker_tid` and returns the corresponding `allocated_bytes`.
/// Returns `None` when the worker tid is not present in the probe
/// output.
///
/// The 1024-entry upper bound is a safety cap — realistic ktstr
/// tests run a single worker plus the test-body main thread and
/// whatever threads ktstr-init spawns internally, well under 100.
fn observed_allocated(metrics: &PayloadMetrics, worker_tid: i32) -> Option<u64> {
    let worker_tid_f64 = worker_tid as f64;
    for i in 0..1024 {
        let tid_key = format!("threads.{i}.tid");
        let tid_m = metrics.metrics.iter().find(|m| m.name == tid_key)?;
        if tid_m.value == worker_tid_f64 {
            let alloc_key = format!("threads.{i}.allocated_bytes");
            return metrics
                .metrics
                .iter()
                .find(|m| m.name == alloc_key)
                .map(|m| m.value as u64);
        }
    }
    None
}

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

/// Single-worker closed loop. A 16 MiB allocation is large enough
/// that jemalloc routes it through the huge-allocation path, which
/// unconditionally updates `thread_allocated` per #479 a-impl's
/// source read. The probe must observe at least that many bytes
/// on the worker's thread.
///
/// Lower-bound assertion only. Jemalloc's internal bookkeeping
/// (arena promotion, per-thread tcache fills, allocator
/// scratchpad) adds variable over-count that makes an upper-bound
/// check brittle. The probe-correctness failure mode is
/// under-counting; over-counting from jemalloc's own overhead is
/// not a probe bug.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_single_worker_observes_known_allocation(ctx: &Ctx) -> Result<AssertResult> {
    const KNOWN_BYTES: usize = 16 * 1024 * 1024;

    let (ready_rx, stop_tx, worker, tid) = spawn_allocator_worker(KNOWN_BYTES);
    // Inner closure: on any early return (Err), the outer body
    // still reaches the `stop_tx.send` + `worker.join` cleanup.
    // Without this pattern, a `.run()?` error would drop `stop_tx`
    // via the early return and the worker thread would panic on
    // `stop_rx.recv`; ktstr's panic hook reboots on first panic,
    // swallowing the real error before it reaches the test
    // harness as a structured AssertResult.
    let run_result: Result<AssertResult> = (|| {
        ready_rx.recv().map_err(|e| anyhow!("worker ready: {e}"))?;
        let worker_tid = tid.load(Ordering::Acquire);
        if worker_tid == 0 {
            return Err(anyhow!("worker tid not populated after ready signal"));
        }
        let pid = std::process::id();
        let (assert_result, metrics) = ctx
            .payload(&JEMALLOC_PROBE)
            .arg("--pid")
            .arg(pid.to_string())
            .arg("--json")
            .run()?;
        let observed = observed_allocated(&metrics, worker_tid).ok_or_else(|| {
            anyhow!(
                "probe JSON missing threads entry for worker tid {worker_tid}; \
                 flat metrics list: {:?}",
                metrics
                    .metrics
                    .iter()
                    .map(|m| (m.name.as_str(), m.value))
                    .collect::<Vec<_>>(),
            )
        })?;
        let expected = KNOWN_BYTES as u64;
        if observed < expected {
            return Ok(fail_result(format!(
                "probe allocated_bytes={observed} for tid={worker_tid}, \
                 expected >= {expected}"
            )));
        }
        Ok(assert_result)
    })();

    let _ = stop_tx.send(());
    let _ = worker.join();
    run_result
}

/// Two workers on distinct threads with distinct known
/// allocations. Beyond the per-worker lower bound, this test
/// cross-checks that the delta between the two observations
/// matches the delta between the two known allocations — a probe
/// bug where every thread reports the same (process-total) value
/// would pass both lower bounds but fail the delta check.
///
/// `cores = 2` lets both workers run concurrently so the
/// allocation sequence is not serialized through a single CPU,
/// which would bias the ordering of jemalloc's internal arena
/// promotions.
#[ktstr_test(llcs = 1, cores = 2, threads = 1)]
fn jemalloc_probe_multi_worker_per_thread_attribution(ctx: &Ctx) -> Result<AssertResult> {
    const SMALL_BYTES: usize = 8 * 1024 * 1024;
    const LARGE_BYTES: usize = 24 * 1024 * 1024;

    let (ready_a, stop_a, handle_a, tid_a_atom) = spawn_allocator_worker(SMALL_BYTES);
    let (ready_b, stop_b, handle_b, tid_b_atom) = spawn_allocator_worker(LARGE_BYTES);
    // Same cleanup-after-early-return pattern as the single-worker
    // test: see that test's comment block for the panic-hook
    // rationale.
    let run_result: Result<AssertResult> = (|| {
        ready_a.recv().map_err(|e| anyhow!("worker A ready: {e}"))?;
        ready_b.recv().map_err(|e| anyhow!("worker B ready: {e}"))?;
        let tid_a = tid_a_atom.load(Ordering::Acquire);
        let tid_b = tid_b_atom.load(Ordering::Acquire);
        if tid_a == 0 || tid_b == 0 || tid_a == tid_b {
            return Err(anyhow!(
                "worker tids not distinct + populated: a={tid_a} b={tid_b}"
            ));
        }
        let pid = std::process::id();
        let (_assert, metrics) = ctx
            .payload(&JEMALLOC_PROBE)
            .arg("--pid")
            .arg(pid.to_string())
            .arg("--json")
            .run()?;
        let obs_a = observed_allocated(&metrics, tid_a)
            .ok_or_else(|| anyhow!("probe output missing worker A tid {tid_a}"))?;
        let obs_b = observed_allocated(&metrics, tid_b)
            .ok_or_else(|| anyhow!("probe output missing worker B tid {tid_b}"))?;
        if obs_a < SMALL_BYTES as u64 {
            return Ok(fail_result(format!(
                "worker A (tid={tid_a}) observed {obs_a} < expected {SMALL_BYTES}"
            )));
        }
        if obs_b < LARGE_BYTES as u64 {
            return Ok(fail_result(format!(
                "worker B (tid={tid_b}) observed {obs_b} < expected {LARGE_BYTES}"
            )));
        }
        let delta_required = (LARGE_BYTES - SMALL_BYTES) as u64;
        if obs_b < obs_a + delta_required {
            return Ok(fail_result(format!(
                "per-thread attribution failed: obs_a={obs_a}, obs_b={obs_b}; \
                 expected obs_b - obs_a >= {delta_required}"
            )));
        }
        Ok(AssertResult::pass())
    })();

    let _ = stop_a.send(());
    let _ = stop_b.send(());
    let _ = handle_a.join();
    let _ = handle_b.join();
    run_result
}

/// Error path — probe a process that has no jemalloc (busybox
/// `sleep`). The probe must exit non-zero; the test reads
/// `exit_code` directly rather than gating via
/// `Check::ExitCodeEq(0)` because the expected outcome is
/// failure. Uses [`JEMALLOC_PROBE_NO_EXIT_CHECK`] so the
/// framework doesn't mark the AssertResult as failed when the
/// probe returns non-zero.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_rejects_non_jemalloc_target(ctx: &Ctx) -> Result<AssertResult> {
    let sleep_handle: PayloadHandle = ctx.payload(&BUSYBOX_SLEEP).spawn()?;
    let target_pid = sleep_handle
        .pid()
        .ok_or_else(|| anyhow!("busybox sleep handle has no pid (child already consumed)"))?;

    let (_assert, metrics) = ctx
        .payload(&JEMALLOC_PROBE_NO_EXIT_CHECK)
        .arg("--pid")
        .arg(target_pid.to_string())
        .arg("--json")
        .run()?;

    // Release the background sleep regardless of probe outcome.
    let _ = sleep_handle.kill();

    if metrics.exit_code == 0 {
        return Ok(fail_result(format!(
            "probe exit_code=0 against busybox sleep (pid={target_pid}); \
             probe must fail with non-zero exit on a target without jemalloc TLS"
        )));
    }
    Ok(AssertResult::pass())
}
