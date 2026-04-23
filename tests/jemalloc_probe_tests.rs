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
use ktstr::worker_ready_wait::wait_for_worker_ready;

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
    include_files: &[],
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
    include_files: &[],
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
    include_files: &[],
};

/// Allocator worker target. Spawned as a background payload; the
/// test body reads its pid from the `PayloadHandle`, then probes
/// externally via `--pid=<worker_pid>`. The worker is
/// single-threaded (`tid == pid`) so the test can match on
/// `threads[N].tid == worker_pid` in the probe's flat metric
/// output without an extra TID handshake.
///
/// See [`JEMALLOC_ALLOC_WORKER_CHURN`] for the thread-churn variant
/// used by the ESRCH stress test — same binary, `--churn` flag,
/// disables the single-thread self-check.
static JEMALLOC_ALLOC_WORKER: Payload = Payload {
    name: "jemalloc_alloc_worker",
    kind: PayloadKind::Binary("ktstr-jemalloc-alloc-worker"),
    output: OutputFormat::ExitCode,
    default_args: &[],
    default_checks: &[],
    metrics: &[],
    include_files: &[],
};

/// Churn-mode allocator worker. Same binary as
/// [`JEMALLOC_ALLOC_WORKER`] but invoked with `--churn`, which
/// disables the single-thread self-check and enters a tight
/// spawn+join loop after the main-thread allocation completes.
/// Used by `jemalloc_probe_survives_thread_churn` to stress the
/// probe's ESRCH handling: the probe races rapidly-exiting helper
/// tids and every seized tid that dies before PTRACE_INTERRUPT
/// surfaces as a `ThreadResult::Err` rather than a crash.
static JEMALLOC_ALLOC_WORKER_CHURN: Payload = Payload {
    name: "jemalloc_alloc_worker_churn",
    kind: PayloadKind::Binary("ktstr-jemalloc-alloc-worker"),
    output: OutputFormat::ExitCode,
    default_args: &["--churn"],
    default_checks: &[],
    metrics: &[],
    include_files: &[],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
        return Ok(AssertResult::fail_other(format!("self-test observed_bytes={observed}, expected >= {KNOWN_BYTES}")));
    }
    if observed > KNOWN_BYTES + MAX_SLOP {
        return Ok(AssertResult::fail_other(format!(
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
    let mut worker: PayloadHandle = ctx
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
    // `try_wait` each tick so a worker that exits BEFORE creating
    // the marker (e.g. the single-thread self-check failed, or
    // allocation OOM'd) is detected immediately instead of waiting
    // the full 5s deadline with stale pid-scoped path polling.
    wait_for_worker_ready(
        &mut worker,
        worker_pid,
        std::time::Duration::from_secs(5),
        "worker",
        "2=bytes==0, 3=/proc/self/task thread count != 1, \
         4=ready-marker write failed, 5=argument parse failed, \
         6=/proc/self/task unreadable, 101=Rust panic, \
         negative=killed by signal",
    )?;

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

    // The worker is declared single-threaded (its /proc/self/task
    // self-check enforces it), so the strong invariant is that
    // `worker_pid` appears AS a tid in the probe's per-thread list —
    // tid-identity. Pin both a lower bound on `n_threads` and the
    // tid-identity check:
    //
    // - Lower bound (`n_threads >= 1`): the probe must reach per-
    //   thread iteration and emit at least one entry. Zero means the
    //   probe bailed early.
    // - Tid-identity: `lookup_thread(metrics, worker_pid)` must
    //   resolve to `ThreadLookup::Found`. `TidAbsent` means the
    //   probe emitted some tids but none of them was the worker.
    //
    // `n_threads` is attached to the passing `AssertResult` at the
    // end of this test as a `DetailKind::Other` diagnostic so
    // future jemalloc versions that lazily spawn a background thread
    // (decay / bg_thd) surface visibly in CI output: `n_threads > 1`
    // would be a heads-up that the worker is no longer strictly
    // single-threaded, without breaking the test's tid-identity
    // contract. A strict `!= 1` assertion here would regress — the
    // probe can still locate the worker's counter correctly even
    // when jemalloc runs its own helper thread.
    let n_threads = thread_count(&metrics);
    if n_threads < 1 {
        return Ok(AssertResult::fail_other(format!(
                "probe saw n_threads={n_threads} for worker pid={worker_pid}; \
                 probe must emit at least one thread entry — bailed before \
                 per-thread iteration or filtered out every tid"
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
                "probe JSON has threads entry for tid={worker_tid} but no \
                 allocated_bytes (n_threads={n_threads}); probe likely emitted \
                 an error record in place of the counter fields"
            ));
        }
        ThreadLookup::TidAbsent => {
            return Err(anyhow!(
                "probe JSON has no threads.N.tid == {worker_tid} entry despite \
                 n_threads={n_threads} — the probe emitted some tids but none \
                 matched worker_pid, tid-identity is broken. Flat metrics: {:?}",
                metrics
                    .metrics
                    .iter()
                    .map(|m| (m.name.as_str(), m.value))
                    .collect::<Vec<_>>(),
            ));
        }
    };
    if allocated < KNOWN_BYTES {
        return Ok(AssertResult::fail_other(format!(
                "worker (tid={worker_tid}) allocated_bytes={allocated}, expected >= {KNOWN_BYTES}"
            )));
    }
    if allocated > KNOWN_BYTES + MAX_SLOP {
        return Ok(AssertResult::fail_other(format!(
                "worker (tid={worker_tid}) allocated_bytes={allocated} exceeds known={KNOWN_BYTES} \
                 + slop={MAX_SLOP}; probe may be reading the wrong address"
            )));
    }
    match deallocated {
        Some(d) if d >= DEALLOC_CAP => {
            return Ok(AssertResult::fail_other(format!(
                    "worker (tid={worker_tid}) deallocated_bytes={d} exceeds cap={DEALLOC_CAP}; \
                     worker should hold its Vec until kill — unexpected free implied"
                )));
        }
        _ => {}
    }
    // Attach the observed n_threads to the passing result so CI
    // output surfaces the count. If a future jemalloc version grows
    // a lazily-spawned background thread the test still passes (via
    // tid-identity), but `n_threads=2` in the detail makes the
    // regression to "worker is no longer strictly single-threaded"
    // visible for human review without a silent capability loss.
    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "jemalloc_probe_external_target: n_threads={n_threads} for \
             worker pid={worker_pid} (expected 1 for single-threaded worker; \
             >1 indicates jemalloc or a future dep spawned a helper thread)"
        ),
    ));
    Ok(result)
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

    // The probe's `run()` pipeline returns `RunOutcome::Fatal` on a
    // missing `/proc/<pid>`, which `main` maps to a stderr
    // "error: {e}" line and `std::process::exit(1)`. No ProbeOutput
    // JSON is emitted on the Fatal arm — only the `Ok` / `AllFailed`
    // arms go through `print_output()`. The assertions here pin
    // that exact shape:
    //   1. `exit_code == 1` — matches the Fatal exit code but is
    //      NOT exclusive to Fatal: the AllFailed arm also exits 1
    //      (see `src/bin/jemalloc_probe.rs` main). This assertion
    //      alone does not distinguish Fatal from AllFailed; it
    //      rules out signal-kill (negative, probe crashed, which is
    //      the regression) and unexpected success (`0`). The
    //      Fatal-vs-AllFailed distinction is carried by assertion 2.
    //   2. The flat metric list is empty. This IS Fatal-exclusive:
    //      AllFailed still emits ProbeOutput via `print_output()`
    //      (populating `pid` and `schema_version` numerics even with
    //      an empty or all-Err `threads` array), so the metric list
    //      would carry at least `pid` and `schema_version` under
    //      AllFailed. Empty-metrics proves the probe took the Fatal
    //      arm and exited via stderr before reaching
    //      `print_output()`, which is what a nonexistent pid must
    //      trigger. Checking emptiness (rather than maintaining an
    //      allowlist of ProbeOutput field names) keeps the test
    //      agnostic to future ProbeOutput field additions.
    if metrics.exit_code != 1 {
        return Ok(AssertResult::fail_other(format!(
                "probe exit_code={} against nonexistent pid {fake_pid}; \
                 expected 1 (RunOutcome::Fatal arm). Negative = signal-kill \
                 crash; 0 = unexpected success; other = unknown failure mode",
                metrics.exit_code,
            )));
    }
    if !metrics.metrics.is_empty() {
        let names: Vec<&str> = metrics.metrics.iter().map(|m| m.name.as_str()).collect();
        return Ok(AssertResult::fail_other(format!(
                "probe against nonexistent pid {fake_pid} emitted {} metric(s) \
                 {names:?}; Fatal arm should exit via stderr before \
                 print_output() populates ProbeOutput, leaving the metric \
                 list empty",
                metrics.metrics.len(),
            )));
    }
    Ok(AssertResult::pass())
}

/// ESRCH-handling stress test. The churn-mode worker spawns and
/// joins short-lived helper threads in a tight loop. The probe is
/// invoked against the worker's pid multiple times; its
/// `readdir(/proc/<pid>/task)` enumerates tids that may die before
/// the subsequent `PTRACE_SEIZE` / `PTRACE_INTERRUPT` lands,
/// returning `ESRCH`. The probe must survive every invocation
/// without panicking or exiting by signal — ESRCH errors must
/// surface as `ThreadResult::Err { kind: PtraceSeize |
/// PtraceInterrupt }` entries in the JSON output, not crashes.
///
/// The assertion is deliberately coarse: `probe exit_code == 0`
/// and at least one invocation saw more than one thread in the
/// probe JSON (confirms churn is actually producing tids for the
/// probe to race). We do NOT assert a specific count of
/// `ThreadResult::Err` entries: whether any given invocation wins
/// every race or loses every race is inherently timing-dependent
/// and would produce a flaky test if pinned.
///
/// N=10 invocations: the churn loop is strictly sequential
/// (spawn → join → respawn), so 1-2 tids visible per readdir;
/// main continuously re-spawns to maximize the number of seize-race
/// opportunities across probe iterations. 10 invocations is
/// empirically enough to land at least one PTRACE_SEIZE /
/// PTRACE_INTERRUPT against a tid that dies mid-probe on an idle
/// guest. Keeping N low bounds the test's wall-time ceiling — each
/// probe invocation costs ~20-40ms.
// Topology: `llcs = 1, cores = 2, threads = 2` — ≥2 CPUs ensure the
// probe process and the churn worker's main thread run concurrently,
// maximizing the window where a just-spawned helper tid is visible
// to readdir before join completes.
#[ktstr_test(llcs = 1, cores = 2, threads = 2)]
fn jemalloc_probe_survives_thread_churn(ctx: &Ctx) -> Result<AssertResult> {
    const KNOWN_BYTES: u64 = 1024 * 1024;
    const INVOCATIONS: usize = 10;
    let mut worker: PayloadHandle = ctx
        .payload(&JEMALLOC_ALLOC_WORKER_CHURN)
        .arg(KNOWN_BYTES.to_string())
        .spawn()?;
    let worker_pid = worker
        .pid()
        .ok_or_else(|| anyhow!("churn worker handle has no pid"))?;
    // Wait for the same pid-scoped ready marker the non-churn path
    // writes — identical handshake shape, simpler reuse than a
    // separate /tmp path for the churn variant. Churn mode skips
    // the single-thread self-check (exit code 3), so it is omitted
    // from the legend here.
    wait_for_worker_ready(
        &mut worker,
        worker_pid,
        std::time::Duration::from_secs(5),
        "churn worker",
        "2=bytes==0, 4=ready-marker write failed, \
         5=argument parse failed, 101=Rust panic, negative=killed by signal",
    )?;

    let mut any_multi_thread_seen = false;
    // A `threads.N.tid` entry without the sibling
    // `threads.N.allocated_bytes` numeric is the probe emitting an
    // Err arm — `walk_json_leaves` drops the string-valued `error`
    // field, so the absence of `allocated_bytes` is the only signal
    // for an ESRCH / PtraceSeize error surfaced through the flat
    // metric layout. Counted per-invocation so the returned
    // AssertResult can carry the observed count as a diagnostic
    // detail even on the pass paths.
    let mut error_invocations: u32 = 0;
    for i in 0..INVOCATIONS {
        // Every iteration spawns a fresh probe subprocess against
        // the live churn worker. A signal-death exit (negative
        // exit_code per PayloadMetrics convention) means the probe
        // panicked or SIGABORT'd on an ESRCH race — the regression
        // this test exists to prevent.
        let (_assert, metrics) = ctx
            .payload(&JEMALLOC_PROBE_NO_EXIT_CHECK)
            .arg("--pid")
            .arg(worker_pid.to_string())
            .arg("--json")
            .run()?;
        if metrics.exit_code < 0 {
            let _ = worker.kill();
            return Ok(AssertResult::fail_other(format!(
                    "invocation {i}: probe died by signal (exit_code={}); \
                     ESRCH race should surface as ThreadResult::Err, not crash",
                    metrics.exit_code
                )));
        }
        // Non-zero (non-signal) exit would mean a fatal probe-side
        // error OUTSIDE the per-thread loop (e.g. find_jemalloc_via_maps
        // failure). That's not what this test exercises — it should
        // reach the per-thread path and at least attempt some tids.
        if metrics.exit_code != 0 {
            let _ = worker.kill();
            return Ok(AssertResult::fail_other(format!(
                    "invocation {i}: probe exit_code={} — fatal error before per-thread loop; \
                     ESRCH stress test requires the probe to enter the tid iteration",
                    metrics.exit_code,
                )));
        }
        if thread_count(&metrics) > 1 {
            any_multi_thread_seen = true;
        }
        // Scan for an error entry on this invocation: the probe's Err
        // arm emits `threads.N.tid` without an `allocated_bytes`
        // sibling (the `error` string is flattened-away by
        // walk_json_leaves). Any such pair is evidence the ESRCH race
        // actually fired on this invocation.
        for j in 0..1024 {
            let tid_key = format!("threads.{j}.tid");
            if !metrics.metrics.iter().any(|m| m.name == tid_key) {
                break;
            }
            let alloc_key = format!("threads.{j}.allocated_bytes");
            if !metrics.metrics.iter().any(|m| m.name == alloc_key) {
                error_invocations += 1;
                break;
            }
        }
    }
    let _ = worker.kill();

    if !any_multi_thread_seen {
        return Ok(AssertResult::fail_other(format!(
                "none of {INVOCATIONS} probe invocations saw more than one thread — \
                 churn worker may not be producing tids fast enough to race the probe, \
                 or readdir(/proc/<pid>/task) is not observing the churn"
            )));
    }
    // Both pass paths attach a DetailKind::Other diagnostic so
    // `error_invocations` is observable in the test report (JSON /
    // stdout). No dedicated `Info` variant exists in DetailKind — the
    // kind is advisory here since the result passes either way; the
    // message is the payload. Hard pass = saw the race at least once;
    // soft pass = race window present (multi-thread view) but never
    // lost. Whether a given invocation wins or loses every race is
    // inherently timing-dependent, so this test does not pin a
    // specific error count.
    let mut result = AssertResult::pass();
    let message = if error_invocations > 0 {
        format!(
            "{error_invocations} of {INVOCATIONS} probe invocations observed \
             ThreadResult::Err entries — ESRCH race window confirmed exercised"
        )
    } else {
        format!(
            "0 of {INVOCATIONS} invocations observed ThreadResult::Err entries — \
             race window may not have been exercised (multi-thread view was \
             visible, but no tid died mid-probe)"
        )
    };
    result
        .details
        .push(AssertDetail::new(DetailKind::Other, message));
    Ok(result)
}

// Host-side worker exit-code tests (exit codes 2/4/5) live in their
// own integration-test file at `tests/jemalloc_alloc_worker_exit_codes.rs`
// to avoid the ktstr early-dispatch ctor's `--list` intercept (which
// hides plain `#[test]` functions in any binary that also carries
// `#[ktstr_test]` entries like this one).
