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
//! activated by the `KTSTR_JEMALLOC_PROBE_BINARY` env var, set by
//! [`set_probe_binary_env_var`] at static init time. The init
//! binary stays stripped; the probe carries its own DWARF and
//! self-probes.

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
/// their entries. Sets `KTSTR_JEMALLOC_PROBE_BINARY` to the absolute
/// host path of `ktstr-jemalloc-probe` so the ktstr test harness
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
            "KTSTR_JEMALLOC_PROBE_BINARY",
            env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
        );
        std::env::set_var(
            "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
            env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker"),
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

/// External-pid probe invocation. Used by the cross-process
/// closed-loop test which reads the JSON output directly to find
/// a specific thread's `allocated_bytes`.
static JEMALLOC_PROBE_EXTERNAL: Payload = Payload {
    name: "jemalloc_probe_external",
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

/// Allocator worker target. Spawned as a background payload; the
/// test body reads its pid from the `PayloadHandle`, then probes
/// externally via `--pid=<worker_pid>`. The worker is
/// single-threaded (`tid == pid`) so the test can match on
/// `threads[N].tid == worker_pid` in the probe's flat metric
/// output without an extra TID handshake.
static JEMALLOC_ALLOC_WORKER: Payload = Payload {
    name: "jemalloc_alloc_worker",
    kind: PayloadKind::Binary("ktstr-jemalloc-alloc-worker"),
    output: OutputFormat::ExitCode,
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

/// Outcome of scanning the flat metric list for a tid-keyed thread
/// entry. Distinguishes "tid not present" from "tid present but
/// `allocated_bytes` missing" so the caller can issue a precise
/// diagnostic instead of a blanket "not found".
enum ThreadLookup {
    /// `threads.N.tid == worker_tid` and `threads.N.allocated_bytes`
    /// are both present. Returns the observed counter plus the
    /// companion `deallocated_bytes` (if emitted).
    Found {
        allocated_bytes: u64,
        deallocated_bytes: Option<u64>,
    },
    /// Probe emitted a `threads.N.tid` matching `worker_tid`, but
    /// no `threads.N.allocated_bytes` sibling. The probe hit an
    /// error on that thread — typically an `error` entry replaces
    /// the counter fields.
    MissingAllocatedBytes,
    /// No `threads.N.tid == worker_tid` entry in the flat metric
    /// list. Probe did not visit the worker at all.
    TidAbsent,
}

/// Extract the `allocated_bytes` / `deallocated_bytes` values for
/// `worker_tid` from the flat metric list produced by
/// `walk_json_leaves` over the probe's JSON output.
///
/// The probe emits
/// `{"pid":P,"threads":[{"tid":T,"allocated_bytes":A,"deallocated_bytes":D,...}, ...]}`
/// which `walk_json_leaves` flattens per array index into contiguous
/// keys `threads.0.tid`, `threads.1.tid`, … with no gaps. The scan
/// below stops at the first `threads.N.tid` miss, which is the
/// natural array terminator. The 1024 cap is a belt-and-suspenders
/// safety bound — realistic probe runs see at most a few dozen
/// threads in a single-allocator worker process.
fn lookup_thread(metrics: &PayloadMetrics, worker_tid: i32) -> ThreadLookup {
    let worker_tid_f64 = worker_tid as f64;
    for i in 0..1024 {
        let tid_key = format!("threads.{i}.tid");
        let tid_m = match metrics.metrics.iter().find(|m| m.name == tid_key) {
            Some(m) => m,
            None => return ThreadLookup::TidAbsent,
        };
        if tid_m.value == worker_tid_f64 {
            let alloc_key = format!("threads.{i}.allocated_bytes");
            let dealloc_key = format!("threads.{i}.deallocated_bytes");
            let allocated_bytes = match metrics
                .metrics
                .iter()
                .find(|m| m.name == alloc_key)
                .map(|m| m.value as u64)
            {
                Some(v) => v,
                None => return ThreadLookup::MissingAllocatedBytes,
            };
            let deallocated_bytes = metrics
                .metrics
                .iter()
                .find(|m| m.name == dealloc_key)
                .map(|m| m.value as u64);
            return ThreadLookup::Found {
                allocated_bytes,
                deallocated_bytes,
            };
        }
    }
    ThreadLookup::TidAbsent
}

/// Count the number of `threads.N.tid` entries in the flat metric
/// list. Walk uses the same contiguous-index property documented on
/// [`lookup_thread`]: array flattening yields indices 0..N without
/// gaps, so scanning stops at the first miss.
fn thread_count(metrics: &PayloadMetrics) -> usize {
    let mut n = 0;
    for i in 0..1024 {
        let tid_key = format!("threads.{i}.tid");
        if metrics.metrics.iter().any(|m| m.name == tid_key) {
            n += 1;
        } else {
            break;
        }
    }
    n
}

/// Look up a flat metric by exact key. Returns `None` if absent.
fn metric_u64(metrics: &PayloadMetrics, key: &str) -> Option<u64> {
    metrics
        .metrics
        .iter()
        .find(|m| m.name == key)
        .map(|m| m.value as u64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Single-worker closed loop via the probe's `--self-test` mode.
/// 16 MiB is large enough that jemalloc routes the allocation
/// through the huge-size path, which unconditionally updates
/// `thread_allocated` on every alloc regardless of tcache state.
/// The probe observes its own process memory (no ptrace required
/// for same-process `process_vm_readv`) and exits 0 iff the
/// observed counter is at least 16 MiB.
///
/// The assertion here does not rely solely on the probe's exit
/// code (already gated by `Check::ExitCodeEq(0)` on
/// `JEMALLOC_PROBE_SELFTEST`). It re-reads the JSON output via
/// the framework's flat-metric pipeline so a future probe change
/// that leaks a false-positive exit (e.g. `passed: true` with
/// `observed_bytes: 0`) still fails the test.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_single_worker_observes_known_allocation(ctx: &Ctx) -> Result<AssertResult> {
    const KNOWN_BYTES: u64 = 16 * 1024 * 1024;
    // Upper bound on post-allocation slop: kernel + jemalloc
    // startup noise (metadata arenas, tsd init, small bookkeeping
    // vectors) on the worker thread. 4 MiB covers observed
    // overhead with slack; a much larger observed value would
    // indicate either a test leak or a probe reading the wrong
    // address.
    const MAX_SLOP: u64 = 4 * 1024 * 1024;
    let (assert_result, metrics) = ctx
        .payload(&JEMALLOC_PROBE_SELFTEST)
        .arg("--self-test")
        .arg(KNOWN_BYTES.to_string())
        .run()?;
    if !assert_result.passed {
        return Ok(assert_result);
    }
    let observed = metric_u64(&metrics, "observed_bytes").ok_or_else(|| {
        anyhow!(
            "self-test metrics missing observed_bytes; flat metrics: {:?}",
            metrics
                .metrics
                .iter()
                .map(|m| (m.name.as_str(), m.value))
                .collect::<Vec<_>>(),
        )
    })?;
    if observed < KNOWN_BYTES {
        return Ok(fail_result(format!(
            "self-test observed_bytes={observed}, expected >= {KNOWN_BYTES}"
        )));
    }
    if observed > KNOWN_BYTES + MAX_SLOP {
        return Ok(fail_result(format!(
            "self-test observed_bytes={observed} exceeds known={KNOWN_BYTES} + slop={MAX_SLOP}; \
             probe may be reading the wrong address or a shared counter"
        )));
    }
    Ok(AssertResult::pass())
}

/// Cross-process closed loop. Spawns the jemalloc-alloc-worker
/// as a background payload with a known allocation size, runs
/// the probe in external-pid mode against the worker's pid,
/// parses the probe's JSON output, and asserts that the worker's
/// thread reports `allocated_bytes` inside a tight band around
/// the known size with `deallocated_bytes` near zero, and that
/// the probe sees exactly one thread in the single-threaded
/// worker process.
///
/// The worker is single-threaded so its `tid == pid` — the test
/// body uses `PayloadHandle::pid()` as the match key against the
/// probe's `threads[N].tid` entries. Readiness is communicated
/// through a pid-scoped marker file the worker writes AFTER its
/// allocation + black_box triple (see jemalloc_alloc_worker.rs);
/// the test polls for that file's existence before launching the
/// probe. The marker replaces the prior fixed-500ms settle — each
/// `#[ktstr_test]` boots a fresh VM with a clean `/tmp`, so there
/// is no stale-file race to defend against.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_external_target_observes_known_allocation(ctx: &Ctx) -> Result<AssertResult> {
    const KNOWN_BYTES: u64 = 16 * 1024 * 1024;
    // See `jemalloc_probe_single_worker_observes_known_allocation`
    // for the rationale behind this bound.
    const MAX_SLOP: u64 = 4 * 1024 * 1024;
    // Worker allocates once and parks — `deallocated_bytes`
    // should be dominated by jemalloc's own bookkeeping + Rust's
    // static-init churn, well below this cap. A larger value
    // indicates either a test-side leak (the worker is freeing
    // its Vec) or the probe reading the wrong offset.
    const DEALLOC_CAP: u64 = 1024 * 1024;
    let worker: PayloadHandle = ctx
        .payload(&JEMALLOC_ALLOC_WORKER)
        .arg(KNOWN_BYTES.to_string())
        .spawn()?;
    let worker_pid = worker
        .pid()
        .ok_or_else(|| anyhow!("worker PayloadHandle has no pid (child already consumed)"))?;
    // Wait for the worker's pid-scoped ready marker. The worker
    // writes this file after its allocation + black_box triple
    // completes, so a successful poll implies the probe will see a
    // materialized heap buffer and a stable `thread_allocated`.
    // 5s is generous vs the worker's expected sub-50ms dispatch +
    // 16 MiB allocation time on a warm guest; a timeout implies
    // the worker died during startup or the VM is heavily stalled.
    let ready_path = format!("/tmp/ktstr-worker-ready-{worker_pid}");
    let ready_deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !std::path::Path::new(&ready_path).exists() {
        if std::time::Instant::now() >= ready_deadline {
            let _ = worker.kill();
            return Err(anyhow!(
                "worker pid={worker_pid} did not create ready marker {ready_path} within 5s"
            ));
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let run_outcome = ctx
        .payload(&JEMALLOC_PROBE_EXTERNAL)
        .arg("--pid")
        .arg(worker_pid.to_string())
        .arg("--json")
        .run();

    // Release the background worker regardless of probe outcome,
    // then propagate the probe's Err or unpack its result.
    let _ = worker.kill();
    let (_assert, metrics) = run_outcome?;

    // Worker is single-threaded → its main thread's tid equals
    // its pid, and probing it should surface exactly one thread
    // entry. A larger thread count would mean the worker forked
    // or spawned a helper thread (breaking the
    // `worker_pid == worker_tid` identity).
    let n_threads = thread_count(&metrics);
    if n_threads != 1 {
        return Ok(fail_result(format!(
            "probe saw {n_threads} thread entries for single-threaded worker pid={worker_pid}; \
             expected exactly 1"
        )));
    }

    let worker_tid = worker_pid as i32;
    let (allocated, deallocated) = match lookup_thread(&metrics, worker_tid) {
        ThreadLookup::Found {
            allocated_bytes,
            deallocated_bytes,
        } => (allocated_bytes, deallocated_bytes),
        ThreadLookup::MissingAllocatedBytes => {
            return Err(anyhow!(
                "probe JSON has threads entry for tid={worker_tid} but no allocated_bytes; \
                 probe likely emitted an error record in place of the counter fields"
            ));
        }
        ThreadLookup::TidAbsent => {
            return Err(anyhow!(
                "probe JSON has no threads.N.tid == {worker_tid} entry; \
                 flat metrics: {:?}",
                metrics
                    .metrics
                    .iter()
                    .map(|m| (m.name.as_str(), m.value))
                    .collect::<Vec<_>>(),
            ));
        }
    };
    if allocated < KNOWN_BYTES {
        return Ok(fail_result(format!(
            "worker (tid={worker_tid}) allocated_bytes={allocated}, expected >= {KNOWN_BYTES}"
        )));
    }
    if allocated > KNOWN_BYTES + MAX_SLOP {
        return Ok(fail_result(format!(
            "worker (tid={worker_tid}) allocated_bytes={allocated} exceeds known={KNOWN_BYTES} \
             + slop={MAX_SLOP}; probe may be reading the wrong address"
        )));
    }
    match deallocated {
        Some(d) if d >= DEALLOC_CAP => {
            return Ok(fail_result(format!(
                "worker (tid={worker_tid}) deallocated_bytes={d} exceeds cap={DEALLOC_CAP}; \
                 worker should hold its Vec until kill — unexpected free implied"
            )));
        }
        _ => {}
    }
    Ok(AssertResult::pass())
}

/// Error path — probe a pid that does not exist. The probe must
/// exit non-zero; the test reads `exit_code` directly rather
/// than gating via `Check::ExitCodeEq(0)` because the expected
/// outcome is failure. Uses [`JEMALLOC_PROBE_NO_EXIT_CHECK`] so
/// the framework does not mark the `AssertResult` as failed when
/// the probe returns non-zero.
///
/// Coverage scope: this test only exercises the pid-not-found
/// branch (`find_jemalloc_via_maps` fails on a missing
/// `/proc/<pid>`). It does NOT cover the complementary
/// "target exists but is not jemalloc-linked" branch — that
/// path is reachable only by spawning a live non-jemalloc process
/// (e.g. busybox), which a `#[ktstr_test]` VM cannot do without
/// extra initramfs wiring (no busybox applets are packed in the
/// test-only base image). Both branches funnel into the same
/// `RunOutcome::Fatal` emission, so the guarantee tested here is
/// "probe reports fatal and exits non-zero on an invalid pid".
/// The not-jemalloc branch is tracked as a follow-up task and
/// exercised by unit tests in the probe crate.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn jemalloc_probe_fatal_on_nonexistent_pid(ctx: &Ctx) -> Result<AssertResult> {
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
