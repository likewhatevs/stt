//! VM-backed end-to-end coverage of the ctprof capture
//! pipeline's per-thread jemalloc TSD counter wiring.
//!
//! The host-side wiring tests in
//! `tests/ctprof_capture_jemalloc_wiring.rs` exercise the
//! probe attach + `process_vm_readv` path against a real spawned
//! `ktstr-jemalloc-alloc-worker` on the OUTER host's /proc. This
//! file flips the perspective: the alloc-worker (and its
//! companion churn variant) run INSIDE a ktstr VM, and
//! [`ktstr::ctprof::capture`] walks the GUEST's /proc + cgroup
//! v2 mount, runs `attach_jemalloc` against the worker's tgid, and
//! pulls counters via ptrace from inside the guest kernel. The two
//! files together prove the wiring lands correctly on both sides
//! of the host/guest boundary.
//!
//! Why VM-backed: the host-side test races against arbitrary
//! concurrent activity on the CI worker (every other process the
//! capture walks). Inside the guest the only jemalloc-linked
//! process is the alloc-worker, so the capture's per-tgid behavior
//! is deterministic — no transient probe failures from unrelated
//! daemons polluting the assertion.
//!
//! The alloc-worker binary reaches the guest via the initramfs
//! wiring activated by the `KTSTR_JEMALLOC_ALLOC_WORKER_BINARY`
//! env var, set at static init time alongside the probe binary
//! the sibling `tests/jemalloc_probe_tests.rs` already exercises.

use anyhow::{Result, anyhow};
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::metric_types::Bytes;
use ktstr::scenario::Ctx;
use ktstr::scenario::payload_run::PayloadHandle;
use ktstr::test_support::{OutputFormat, Payload, PayloadKind};
use ktstr::worker_ready_wait::wait_for_worker_ready;

// ---------------------------------------------------------------------------
// Initramfs wiring — set the env var consumed by ktstr's VM builder so the
// alloc-worker binary lands at /bin/ktstr-jemalloc-alloc-worker on the guest
// PATH. Mirrors the ctor pattern in tests/jemalloc_probe_tests.rs; both
// integration-test files declare their own ctor because each compiles to
// a distinct integration-test binary with its own static init list.
// ---------------------------------------------------------------------------

#[::ktstr::__private::ctor::ctor(crate_path = ::ktstr::__private::ctor)]
fn set_alloc_worker_binary_env_var() {
    unsafe {
        std::env::set_var(
            "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
            env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker"),
        );
    }
}

// ---------------------------------------------------------------------------
// Payload fixtures — duplicated from tests/jemalloc_probe_tests.rs because
// integration-test crates do not share statics across binaries. The fixture
// fields mirror the originals exactly so a refactor in either file lands
// loudly via a behavioral diff rather than a silent drift.
// ---------------------------------------------------------------------------

static JEMALLOC_ALLOC_WORKER: Payload = Payload::new(
    "jemalloc_alloc_worker",
    PayloadKind::Binary("ktstr-jemalloc-alloc-worker"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
    None,
);

static JEMALLOC_ALLOC_WORKER_CHURN: Payload = Payload::new(
    "jemalloc_alloc_worker_churn",
    PayloadKind::Binary("ktstr-jemalloc-alloc-worker"),
    OutputFormat::ExitCode,
    &["--churn"],
    &[],
    &[],
    &[],
    false,
    None,
    None,
);

// ---------------------------------------------------------------------------
// Tunables shared across the e2e tests.
// ---------------------------------------------------------------------------

/// Allocation size the alloc-worker is asked to plant. Picked well
/// above jemalloc's tcache threshold so the allocation lands on the
/// slow path and `tsd_s.thread_allocated` is updated synchronously
/// (no per-thread cache deferral). Mirrors the value used by the
/// host-side wiring test and the probe tests.
const KNOWN_BYTES: u64 = 16 * 1024 * 1024;

/// Upper bound on jemalloc + Rust runtime overhead added on top of
/// [`KNOWN_BYTES`]. Mirrors the slop the in-VM probe tests use; a
/// larger observed value implies either a worker leak or the probe
/// reading the wrong address.
const MAX_SLOP: u64 = 4 * 1024 * 1024;

/// Smaller allocation for the churn-worker invocation. The churn
/// test cares about probe survival across rapidly-exiting helper
/// tids, not the allocation magnitude — keep it small to bound the
/// test's per-snapshot wall time.
const CHURN_KNOWN_BYTES: u64 = 1024 * 1024;

/// Worker-ready handshake timeout. 5 s is generous vs the
/// alloc-worker's expected sub-50 ms dispatch + the planted
/// allocation; a timeout implies the worker died during startup
/// or the VM is heavily stalled.
const READY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// `kthreadd`, the kernel thread daemon, is always tgid=2 on
/// Linux (init=1, kthreadd=2). Pinned by the kernel since v2.6;
/// no architecture or distro variation. Used as the canonical
/// "kernel thread present in every guest" anchor for the
/// bare-guest test below.
const KTHREADD_TGID: u32 = 2;

// ---------------------------------------------------------------------------
// T1 — capture pulls populated allocated_bytes for the alloc-worker
// ---------------------------------------------------------------------------

/// Spawn the alloc-worker with a known size inside the guest, wait
/// for its ready marker, then run `ctprof::capture()` against
/// the guest's /proc. Find the worker's tgid in the snapshot and
/// assert `allocated_bytes >= KNOWN_BYTES` and within slop. The
/// worker is single-threaded (default mode enforces it), so any
/// thread under the worker's tgid carries the planted allocation.
///
/// Topology mirrors the probe tests: 1 LLC / 1 core / 1 thread. The
/// test cares about the capture wiring, not scheduler behavior — a
/// larger topology adds wall-clock time without raising signal.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn ctprof_capture_records_allocated_bytes_for_jemalloc_alloc_worker(
    ctx: &Ctx,
) -> Result<AssertResult> {
    let mut worker: PayloadHandle = ctx
        .payload(&JEMALLOC_ALLOC_WORKER)
        .arg(KNOWN_BYTES.to_string())
        .spawn()?;
    let worker_pid = worker
        .pid()
        .ok_or_else(|| anyhow!("alloc-worker handle has no pid (child already consumed)"))?;
    wait_for_worker_ready(
        &mut worker,
        worker_pid,
        READY_TIMEOUT,
        "alloc-worker",
        "2=bytes==0, 3=/proc/self/task thread count != 1, \
         4=ready-marker write failed, 5=argument parse failed, \
         6=/proc/self/task unreadable, 101=Rust panic, \
         negative=killed by signal",
    )?;

    // Capture the guest's ctprof. `capture()` walks
    // /proc and /sys/fs/cgroup, runs `attach_jemalloc` against
    // every non-self tgid (the alloc-worker is non-self, the
    // calling test process is self), and pulls per-thread
    // counters from the worker via ptrace + process_vm_readv.
    let snap = ktstr::ctprof::capture();

    // Release the worker before any assertion can short-circuit
    // the function — keeps a test failure from orphaning the
    // process inside the guest.
    let _ = worker.kill();

    // Locate the worker's threads in the snapshot. The capture
    // walks every live tgid; we filter on the worker's tgid to
    // avoid mis-counting other guest processes.
    let worker_threads: Vec<&ktstr::ctprof::ThreadState> = snap
        .threads
        .iter()
        .filter(|t| t.tgid == worker_pid)
        .collect();
    if worker_threads.is_empty() {
        return Ok(AssertResult::fail_msg(format!(
            "ctprof::capture() did not see worker tgid={worker_pid} in \
             its /proc walk; total threads in snapshot: {}",
            snap.threads.len(),
        )));
    }

    // The worker's main thread carries the planted allocation;
    // pick the maximum across the worker's threads to be robust
    // against any future jemalloc helper thread. Default-mode
    // worker enforces single-threaded via /proc/self/task self-
    // check, so this is normally just one entry.
    let allocated: u64 = worker_threads
        .iter()
        .map(|t| t.allocated_bytes.0)
        .max()
        .expect("worker_threads non-empty per the gate above");
    let deallocated: u64 = worker_threads
        .iter()
        .map(|t| t.deallocated_bytes.0)
        .max()
        .expect("worker_threads non-empty per the gate above");

    if allocated < KNOWN_BYTES {
        return Ok(AssertResult::fail_msg(format!(
            "worker (tgid={worker_pid}) allocated_bytes={allocated}, \
             expected >= {KNOWN_BYTES}; threads in worker tgid: {}. \
             Capture's attach_jemalloc either failed against the worker's \
             ELF (DWARF missing, jemalloc-not-found) or the per-thread \
             ptrace step failed (check ptrace_scope inside the guest).",
            worker_threads.len(),
        )));
    }
    if allocated > KNOWN_BYTES + MAX_SLOP {
        return Ok(AssertResult::fail_msg(format!(
            "worker allocated_bytes={allocated} exceeds known {KNOWN_BYTES} \
             + slop {MAX_SLOP}; capture may be reading the wrong address \
             or the worker leaked extra allocations beyond the planted Vec",
        )));
    }
    // The worker holds its Vec until kill, so deallocations stay
    // bounded to jemalloc startup + Rust runtime churn — well
    // below the planted size.
    if deallocated >= KNOWN_BYTES {
        return Ok(AssertResult::fail_msg(format!(
            "worker deallocated_bytes={deallocated} >= KNOWN_BYTES \
             ({KNOWN_BYTES}); worker should not free its planted Vec \
             before kill",
        )));
    }

    // Pass — annotate the result with the observed allocation so
    // CI output surfaces the actual reading. Useful for
    // distinguishing slop variations across kernel versions
    // without breaking the assertion contract.
    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "ctprof_capture_records_allocated_bytes: tgid={worker_pid}, \
             threads_in_tgid={}, allocated={allocated}, deallocated={deallocated}",
            worker_threads.len(),
        ),
    ));
    Ok(result)
}

// ---------------------------------------------------------------------------
// T2 — capture against a bare guest leaves kthreadd's counters at zero
// ---------------------------------------------------------------------------

/// Boot a minimal guest with NO payload, run
/// `ctprof::capture()`, and assert that kthreadd (tgid=2)
/// carries `allocated_bytes==0` AND `deallocated_bytes==0`.
/// Kthreadd is a kernel thread; it has no userspace ELF behind
/// `/proc/2/exe` and therefore `attach_jemalloc` returns a
/// readlink failure (or jemalloc-not-found, depending on what the
/// kernel exposes for kernel threads on this kernel version).
/// Either way the per-thread counters land at the absent-counter
/// default.
///
/// This is the negative complement to T1: T1 proves a real
/// jemalloc target populates non-zero counters; this test proves
/// non-jemalloc targets stay at zero — together they pin the
/// "absent = 0" capture contract on both sides of the boundary.
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn ctprof_capture_completes_against_bare_guest(_ctx: &Ctx) -> Result<AssertResult> {
    // No payload — capture runs against whatever processes the
    // guest has after boot. Init (pid 1), kthreadd (pid 2), and
    // any other kernel threads are guaranteed; userspace activity
    // is bounded to the test binary itself plus whatever ktstr's
    // init brings up.
    let snap = ktstr::ctprof::capture();

    // First-level sanity: the walk visited SOMETHING. An empty
    // snapshot here would mean iter_tgids_at(/proc) returned no
    // entries, i.e. /proc is unreadable from inside the guest.
    if snap.threads.is_empty() {
        return Ok(AssertResult::fail_msg(
            "ctprof::capture() returned zero threads on a bare guest — \
             /proc walk produced no entries, indicating the capture layer \
             is not reading the guest's procfs successfully",
        ));
    }

    // Find kthreadd. It's tgid 2 on every Linux kernel; if it's
    // not there, either the guest kernel is wedged or the
    // capture-layer tgid filter rejected it. Distinct error
    // path from "found but counters non-zero".
    let kthreadd_threads: Vec<&ktstr::ctprof::ThreadState> = snap
        .threads
        .iter()
        .filter(|t| t.tgid == KTHREADD_TGID)
        .collect();
    if kthreadd_threads.is_empty() {
        return Ok(AssertResult::fail_msg(format!(
            "kthreadd (tgid={KTHREADD_TGID}) absent from snapshot; \
             total threads: {}, observed tgids preview: {}. \
             Either the guest kernel skipped tgid=2 or the capture \
             /proc walk filtered it out.",
            snap.threads.len(),
            tgids_dump(&snap),
        )));
    }

    // Kthreadd and other kernel threads have no userspace ELF;
    // attach_jemalloc fails on every kernel thread and the
    // counters collapse to the absent-default of 0. A non-zero
    // value here means either the capture pipeline mistakenly
    // populated kernel-thread counters from some other source,
    // or the absent-counter contract regressed.
    for t in &kthreadd_threads {
        if t.allocated_bytes != Bytes(0) {
            return Ok(AssertResult::fail_msg(format!(
                "kthreadd tid={} carries allocated_bytes={}; kernel \
                 threads have no userspace heap, the absent-counter \
                 contract requires this to be 0",
                t.tid, t.allocated_bytes,
            )));
        }
        if t.deallocated_bytes != Bytes(0) {
            return Ok(AssertResult::fail_msg(format!(
                "kthreadd tid={} carries deallocated_bytes={}; expected 0",
                t.tid, t.deallocated_bytes,
            )));
        }
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "ctprof_capture_completes_against_bare_guest: \
             kthreadd_threads={}, total_threads={}",
            kthreadd_threads.len(),
            snap.threads.len(),
        ),
    ));
    Ok(result)
}

/// Helper that builds a small diagnostic of the observed tgids in
/// a snapshot — used by the kthreadd-absent failure path so a
/// reviewer chasing a regression can see whether tgid 2 was
/// genuinely missing or just filtered out by the capture-layer
/// ghost-thread logic. Caps the preview at 16 entries so a guest
/// with many tgids does not blow up the failure message.
fn tgids_dump(snap: &ktstr::ctprof::CtprofSnapshot) -> String {
    let tgids: std::collections::BTreeSet<u32> = snap.threads.iter().map(|t| t.tgid).collect();
    let total = tgids.len();
    let preview: Vec<u32> = tgids.into_iter().take(16).collect();
    format!("{preview:?} (of {total} distinct tgids)")
}

// ---------------------------------------------------------------------------
// T4 — capture against a churn worker survives the ESRCH race window
// ---------------------------------------------------------------------------

/// Boot a guest, spawn the alloc-worker in `--churn` mode (tight
/// spawn+join loop on helper threads after the main allocation),
/// then run `ctprof::capture()` against the guest's /proc.
/// The capture's per-tgid attach + per-tid probe must survive
/// every helper tid that exits between `iter_task_ids_at`
/// enumeration and the per-tid ptrace step (the dominant
/// production case documented on
/// `host_thread_probe::ProbeError::PtraceSeize`). Assert the
/// worker's main tid populates AND the snapshot is otherwise
/// non-empty — the race-with-thread-death case must NOT crash
/// the capture.
#[ktstr_test(llcs = 1, cores = 2, threads = 2)]
fn ctprof_capture_against_churn_worker_does_not_panic(ctx: &Ctx) -> Result<AssertResult> {
    let mut worker: PayloadHandle = ctx
        .payload(&JEMALLOC_ALLOC_WORKER_CHURN)
        .arg(CHURN_KNOWN_BYTES.to_string())
        .spawn()?;
    let worker_pid = worker
        .pid()
        .ok_or_else(|| anyhow!("churn worker handle has no pid"))?;
    wait_for_worker_ready(
        &mut worker,
        worker_pid,
        READY_TIMEOUT,
        "churn alloc-worker",
        "2=bytes==0, 4=ready-marker write failed, 5=argument parse failed, \
         101=Rust panic, negative=killed by signal",
    )?;

    // The churn worker is busy spawning + joining helper threads
    // when capture fires. Each helper tid that exits between
    // iter_task_ids_at and the per-tid attach surfaces as a
    // ProbeError::PtraceSeize / Waitpid in the capture pipeline;
    // the absent-counter contract absorbs these into 0 without
    // failing the snapshot. The strong assertion is "capture
    // completes without crashing", which a successful return
    // of `capture()` proves implicitly — a panic in the probe
    // engine would propagate out and the test would never
    // reach the post-capture code.
    let snap = ktstr::ctprof::capture();

    // Release the churn worker before assertions short-circuit.
    let _ = worker.kill();

    // The worker tgid must appear in the walk — proves the
    // per-tgid attach + per-tid pull made it through the ESRCH
    // race window without aborting the snapshot. The worker's
    // MAIN tid (tid == pid for default mode; in churn mode the
    // main tid still matches pid because Linux assigns tid=pid
    // to the leader thread) must be present specifically:
    // helper tids are racy by design, but the main tid is
    // long-lived for the duration of the test.
    let main_tid_present = snap
        .threads
        .iter()
        .any(|t| t.tgid == worker_pid && t.tid == worker_pid);
    if !main_tid_present {
        let worker_thread_count = snap.threads.iter().filter(|t| t.tgid == worker_pid).count();
        return Ok(AssertResult::fail_msg(format!(
            "capture saw {worker_thread_count} threads under tgid={worker_pid} \
             but none with tid={worker_pid} — the leader (main) thread \
             is missing from the snapshot. The leader is long-lived, so \
             its absence implies the capture pipeline filtered it out \
             during the per-tid walk (likely an ESRCH race between \
             iter_task_ids_at and the per-tid procfs reads, mis-\
             classifying the leader as a ghost thread)",
        )));
    }

    // Sanity: the snapshot is non-empty overall — if capture
    // returned an empty `threads` vec the helper-thread race
    // somehow wedged the entire walk.
    if snap.threads.is_empty() {
        return Ok(AssertResult::fail_msg(
            "capture against churn worker returned zero threads — \
             the ESRCH race window appears to have aborted the \
             entire /proc walk rather than collapsing per-tid",
        ));
    }

    let mut result = AssertResult::pass();
    let main_alloc: u64 = snap
        .threads
        .iter()
        .find(|t| t.tgid == worker_pid && t.tid == worker_pid)
        .map(|t| t.allocated_bytes.0)
        .unwrap_or(0);
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "ctprof_capture_against_churn_worker: tgid={worker_pid}, \
             total_threads={}, main_allocated_bytes={main_alloc}",
            snap.threads.len(),
        ),
    ));
    Ok(result)
}
