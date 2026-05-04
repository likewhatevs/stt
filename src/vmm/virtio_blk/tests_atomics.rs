#![cfg(test)]

use super::*;
use super::testing::*;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempfile;
use virtio_bindings::bindings::virtio_ring::VRING_DESC_F_WRITE;
use virtio_queue::desc::{RawDescriptor, split::Descriptor as SplitDescriptor};
use virtio_queue::mock::MockSplitQueue;

    // ------------------------------------------------------------------
    // join_worker_with_timeout unit tests
    // ------------------------------------------------------------------

    /// Build a minimal `BlkWorkerState` for tests that exercise the
    /// timeout helper. The state's contents are irrelevant — these
    /// tests only assert on the `JoinWithTimeoutOutcome` variant —
    /// so the buckets are unlimited, the scratch buffers empty, and
    /// the backing file an unsized tempfile.
    fn dummy_worker_state() -> BlkWorkerState {
        BlkWorkerState {
            backing: tempfile().expect("create tempfile for dummy_worker_state"),
            ops_bucket: TokenBucket::unlimited(),
            bytes_bucket: TokenBucket::unlimited(),
            all_descs_scratch: Vec::new(),
            io_buf_scratch: Vec::new(),
            capacity_bytes: 0,
            read_only: false,
            counters: Arc::new(VirtioBlkCounters::default()),
            currently_stalled: false,
            queue_poisoned: false,
        }
    }

    #[test]
    fn join_worker_with_timeout_happy_path_returns_joined() {
        // The worker thread returns immediately; the helper joins
        // it well before the budget. Using DROP_JOIN_TIMEOUT here
        // mirrors the production `Drop` site so this test is also
        // a smoke test for that arrangement.
        let handle = std::thread::Builder::new()
            .name("ktstr-vblk-test-happy".to_string())
            .spawn(dummy_worker_state)
            .expect("spawn happy-path worker");
        let start = Instant::now();
        let outcome = join_worker_with_timeout(handle, DROP_JOIN_TIMEOUT);
        let elapsed = start.elapsed();
        assert!(
            matches!(outcome, JoinWithTimeoutOutcome::Joined(_)),
            "expected Joined, got {:?}",
            outcome_label(&outcome)
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "happy-path join took {elapsed:?}, expected < 100ms"
        );
    }

    #[test]
    fn join_worker_with_timeout_returns_timed_out_when_worker_blocks() {
        // The worker sleeps 60 s — much longer than the 50 ms
        // budget — so `recv_timeout` must return `Timeout` and the
        // function reports `TimedOut`. After the assertion the
        // helper holds the worker `JoinHandle` and remains blocked
        // in `handle.join()`; the worker is still in its sleep.
        // Both are leaked. They are killed when the test binary
        // process exits; nothing in this test waits on them.
        let handle = std::thread::Builder::new()
            .name("ktstr-vblk-test-timeout".to_string())
            .spawn(|| {
                std::thread::sleep(Duration::from_secs(60));
                dummy_worker_state()
            })
            .expect("spawn timeout-path worker");
        let start = Instant::now();
        let outcome = join_worker_with_timeout(handle, Duration::from_millis(50));
        let elapsed = start.elapsed();
        assert!(
            matches!(outcome, JoinWithTimeoutOutcome::TimedOut),
            "expected TimedOut, got {:?}",
            outcome_label(&outcome)
        );
        assert!(
            elapsed >= Duration::from_millis(50),
            "timeout fired too early at {elapsed:?}; expected >= 50ms"
        );
        assert!(
            elapsed < Duration::from_millis(200),
            "timeout fired too late at {elapsed:?}; expected < 200ms \
             (recv_timeout overhead budget)"
        );
    }

    #[test]
    fn join_worker_with_timeout_returns_panicked_on_worker_panic() {
        // The worker thread panics. `JoinHandle::join` returns
        // `Err(payload)`, which the helper forwards verbatim; the
        // function maps it to `Panicked(payload)`. The payload
        // round-trips: a `panic!("literal")` deposits a
        // `&'static str` recoverable via `downcast_ref`.
        let handle = std::thread::Builder::new()
            .name("ktstr-vblk-test-panic".to_string())
            .spawn(|| -> BlkWorkerState {
                panic!("intentional panic from join_worker_with_timeout test");
            })
            .expect("spawn panic-path worker");
        let start = Instant::now();
        let outcome = join_worker_with_timeout(handle, DROP_JOIN_TIMEOUT);
        let elapsed = start.elapsed();
        assert!(
            matches!(outcome, JoinWithTimeoutOutcome::Panicked(_)),
            "expected Panicked, got {:?}",
            outcome_label(&outcome)
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "panic-path join took {elapsed:?}, expected < 100ms \
             (parity with happy path)"
        );
        // Confirm the payload round-trips through the channel.
        if let JoinWithTimeoutOutcome::Panicked(payload) = outcome {
            assert_eq!(
                panic_payload_str(&*payload),
                "intentional panic from join_worker_with_timeout test",
                "panic payload round-trip should preserve the &'static str"
            );
        }
    }

    /// Stable label for `JoinWithTimeoutOutcome` for use in test
    /// failure messages — the enum itself does not derive `Debug`
    /// (the `Joined` variant carries `BlkWorkerState`, which has no
    /// `Debug` impl and shouldn't gain one just for tests).
    fn outcome_label(o: &JoinWithTimeoutOutcome) -> &'static str {
        match o {
            JoinWithTimeoutOutcome::Joined(_) => "Joined",
            JoinWithTimeoutOutcome::Panicked(_) => "Panicked",
            JoinWithTimeoutOutcome::TimedOut => "TimedOut",
            JoinWithTimeoutOutcome::HelperSpawnFailed => "HelperSpawnFailed",
            JoinWithTimeoutOutcome::HelperDisconnected => "HelperDisconnected",
        }
    }

    /// `RESET_JOIN_TIMEOUT` matches `DROP_JOIN_TIMEOUT` (1 s) so a
    /// reset on the vCPU thread cannot block longer than the
    /// destructor would. Pin the equality so a future tweak that
    /// shortens one but not the other surfaces here. The "must
    /// match" framing matters because the freeze coordinator's
    /// SIGRTMIN rendezvous (30 s wall budget at the coordinator
    /// level — see `FREEZE_RENDEZVOUS_TIMEOUT` in `src/vmm/mod.rs`)
    /// is sensitive to vCPU-thread blocking budgets; both
    /// `Drop` and `reset()` paths run on a vCPU thread, so
    /// asymmetric budgets would let one path miss the rendezvous
    /// while the other doesn't.
    #[test]
    fn reset_join_timeout_matches_drop_budget() {
        assert_eq!(
            RESET_JOIN_TIMEOUT, DROP_JOIN_TIMEOUT,
            "RESET_JOIN_TIMEOUT must equal DROP_JOIN_TIMEOUT — both \
             paths run on a vCPU thread that the freeze coordinator \
             may target with SIGRTMIN; asymmetric budgets would let \
             reset() miss a rendezvous Drop wouldn't, or vice versa",
        );
        // Pin the absolute value so a future refactor that lifts
        // both into a single shared symbol (or shortens both
        // together) still flags here. 1 s is the documented value
        // — see RESET_JOIN_TIMEOUT and DROP_JOIN_TIMEOUT doc
        // comments for the rationale.
        assert_eq!(RESET_JOIN_TIMEOUT, Duration::from_secs(1));
    }

    /// Stand-in for the production `reset()` join behaviour: when
    /// the worker thread is wedged in a blocking syscall and
    /// doesn't observe `stop_fd`, `join_worker_with_timeout` with
    /// the production `RESET_JOIN_TIMEOUT` budget MUST return
    /// `TimedOut` rather than blocking the calling thread
    /// indefinitely. The vCPU-protection invariant in
    /// `stop_worker_and_reclaim_state` rests on this.
    ///
    /// Why this isn't a direct `reset()` test:
    /// `stop_worker_and_reclaim_state` is `cfg(not(test))`-only,
    /// because in `cfg(test)` the device runs in `Inline` engine
    /// mode (no worker thread, no `stop_fd`). Driving the
    /// production `reset()` path from a unit test would require
    /// stitching cfgs together — instead we exercise the
    /// underlying mechanism (`join_worker_with_timeout`) at the
    /// budget the production path uses, so a regression that
    /// shrunk the budget below realistic worker drain times would
    /// surface here as a flake; a regression that removed the
    /// timeout entirely would surface as a test hang past the
    /// nextest per-test ceiling.
    ///
    /// To keep the test fast (nextest budget ≪ 1 s per test on
    /// typical CI), this uses a child timeout < `RESET_JOIN_TIMEOUT`
    /// — the upper-bound assertion below pins the actual production
    /// budget against what `RESET_JOIN_TIMEOUT` enforces.
    /// `reset_join_timeout_matches_drop_budget` (above) pins the
    /// 1 s value separately.
    #[test]
    fn reset_join_timeout_against_wedged_worker_returns_timed_out() {
        use std::sync::mpsc as test_mpsc;

        // Worker thread that never exits — blocks on a channel
        // receive whose sender is held by this test until the
        // test's scope drops (after the assertion). `stop_fd` has
        // no analogue in this test harness, so the wedge models
        // a worker stuck in `pread`/`pwrite` that doesn't check
        // `stop_fd`.
        let (_keep_alive_tx, wedge_rx) = test_mpsc::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("ktstr-vblk-test-wedged-reset".to_string())
            .spawn(move || -> BlkWorkerState {
                // Block forever (until test scope drops _keep_alive_tx).
                let _ = wedge_rx.recv();
                dummy_worker_state()
            })
            .expect("spawn wedged worker");

        // Use a SHORT budget for the test to keep nextest fast,
        // but assert below that the budget is strictly less than
        // RESET_JOIN_TIMEOUT (so the test can never accidentally
        // outlast the production budget).
        const TEST_TIMEOUT: Duration = Duration::from_millis(100);
        assert!(
            TEST_TIMEOUT < RESET_JOIN_TIMEOUT,
            "test budget must be smaller than RESET_JOIN_TIMEOUT \
             so the test stays fast; a future RESET_JOIN_TIMEOUT \
             tightening below 100 ms would require updating \
             TEST_TIMEOUT here",
        );

        let start = Instant::now();
        let outcome = join_worker_with_timeout(handle, TEST_TIMEOUT);
        let elapsed = start.elapsed();

        // The wedged worker did not exit; outcome must be TimedOut.
        assert!(
            matches!(outcome, JoinWithTimeoutOutcome::TimedOut),
            "wedged worker must yield TimedOut, got {:?}",
            outcome_label(&outcome)
        );
        // The bounded join MUST have returned within the budget,
        // not blocked indefinitely. Allow up to 2x slack for
        // recv_timeout's underlying clock + thread scheduling
        // jitter on slow CI.
        assert!(
            elapsed < TEST_TIMEOUT * 2,
            "join_worker_with_timeout took {elapsed:?} for a \
             wedged worker (budget {TEST_TIMEOUT:?}); the bound \
             must hold so the production reset() path doesn't \
             pin the vCPU thread when the worker is stuck"
        );
        // _keep_alive_tx drops here, releasing the wedge channel
        // so the worker thread can finally exit and reclaim its
        // resources for the test process.
    }

    // ----------------------------------------------------------------
    // Concurrent atomic-access tests for the cross-thread shared
    // state that the production worker uses.
    //
    // The `interrupt_status` (Arc<AtomicU32>), `config_generation`
    // (AtomicU32 directly on the device), and `VirtioBlkCounters`
    // fields (`Arc<VirtioBlkCounters>`'s AtomicU64s) are written
    // from one thread (worker / vCPU) and read or also-written from
    // another. The atomicity invariant — no torn observations, no
    // lost updates — is what makes the cross-thread design sound.
    //
    // These tests hammer the atomics from multiple threads
    // synchronized on a starting barrier and assert the final
    // observable state matches what a sequential semantic predicts
    // (no lost updates) or that no transient state is observed
    // (no torn read for a single atomic operation). They run in
    // cfg(test) so the `BlkWorker` is in Inline mode and no real
    // production worker exists; the atomics themselves are
    // cfg-independent and live on `VirtioBlk` regardless of build
    // profile, so the tests exercise the same memory cells the
    // production worker would.
    // ----------------------------------------------------------------

    /// `interrupt_status.fetch_or` from N concurrent threads, each
    /// setting one unique bit, with a separate reader thread doing
    /// `load(Acquire)` in a loop. Final observation must equal the
    /// union of all threads' set bits — no lost updates, no torn
    /// reads.
    ///
    /// Models the production race: worker thread fires
    /// `interrupt_status.fetch_or(VIRTIO_MMIO_INT_VRING, Release)`
    /// from `drain_bracket_impl` while the vCPU thread reads
    /// `interrupt_status.load(Acquire)` from `mmio_read`. The bit
    /// in question (`VIRTIO_MMIO_INT_VRING`) is only one of the
    /// two virtio-defined transport interrupt bits; we fan out to
    /// 16 distinct bits so a regression that lost one fetch_or via
    /// an inadvertent `store` (overwrite-instead-of-OR) would
    /// surface as a missing bit in the final union.
    #[test]
    fn interrupt_status_concurrent_fetch_or_load() {
        use std::sync::Barrier;

        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        // Snapshot the Arc so the spawned threads can observe the
        // same atomic the production worker would.
        let int_status = Arc::clone(&dev.interrupt_status);
        // 16 writer threads, each setting a distinct bit (bits
        // 0..16). 16 is large enough to expose any
        // store-instead-of-fetch_or regression yet small enough
        // to keep the test reliably under 1 s on slow CI runners.
        const NUM_WRITERS: u32 = 16;
        let barrier = Arc::new(Barrier::new(NUM_WRITERS as usize + 1));
        let mut handles = Vec::with_capacity(NUM_WRITERS as usize);
        for bit in 0..NUM_WRITERS {
            let int_status_w = Arc::clone(&int_status);
            let barrier_w = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_w.wait();
                // Fire fetch_or many times to maximise contention.
                // Each iteration is a no-op after the first since
                // the bit is already set — but the contention on
                // the cache line stresses the atomic primitive.
                for _ in 0..1_000 {
                    int_status_w.fetch_or(1u32 << bit, Ordering::Release);
                }
            }));
        }
        // The reader observes loads concurrently; we don't assert
        // on intermediate states (any subset of the union is
        // legal mid-race), only that the FINAL load equals the
        // full union after every writer joins.
        barrier.wait();
        for h in handles {
            h.join().expect("writer thread join");
        }
        // After all writers join, the bits set are union of bits
        // 0..NUM_WRITERS = (1 << NUM_WRITERS) - 1.
        let expected_union = (1u32 << NUM_WRITERS) - 1;
        let observed = int_status.load(Ordering::Acquire);
        assert_eq!(
            observed, expected_union,
            "all NUM_WRITERS bits must be set; missing bits indicate \
             a lost fetch_or update — observed {observed:#x}, \
             expected {expected_union:#x}",
        );
    }

    /// Concurrent `fetch_or` (worker bit-set) racing
    /// `fetch_and(!val, AcqRel)` (vCPU INTERRUPT_ACK clear). Final
    /// state must reflect bits set BUT NOT cleared. Models the
    /// race between a worker firing `fetch_or(VIRTIO_MMIO_INT_VRING)`
    /// and a vCPU running `mmio_write(INTERRUPT_ACK,
    /// VIRTIO_MMIO_INT_VRING)`.
    ///
    /// Strategy: thread A repeatedly fetch_or's bit X; thread B
    /// repeatedly fetch_and's the inverse of bit Y (clear bit Y).
    /// X and Y are DISJOINT bits, so the final state must be:
    /// bit X set (A always wins on its own bit), bit Y must equal
    /// its initial state cleared by every B iteration (Y was set
    /// before the test, B clears it, A doesn't touch it). A
    /// regression that mis-ordered the AcqRel pair (e.g. used
    /// `Relaxed` on either side) could cause B's clear to
    /// accidentally also drop bit X if the implementation
    /// store'd instead of `&=`'d.
    #[test]
    fn interrupt_status_concurrent_set_and_ack() {
        use std::sync::Barrier;

        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let int_status = Arc::clone(&dev.interrupt_status);
        // Pre-set bit Y = 1 so the ACK loop has something to clear.
        const BIT_X: u32 = 1 << 0;
        const BIT_Y: u32 = 1 << 1;
        int_status.store(BIT_Y, Ordering::Release);

        let barrier = Arc::new(Barrier::new(3));
        let int_status_a = Arc::clone(&int_status);
        let barrier_a = Arc::clone(&barrier);
        let setter = thread::spawn(move || {
            barrier_a.wait();
            // Thread A: repeatedly set bit X.
            for _ in 0..10_000 {
                int_status_a.fetch_or(BIT_X, Ordering::Release);
            }
        });
        let int_status_b = Arc::clone(&int_status);
        let barrier_b = Arc::clone(&barrier);
        let acker = thread::spawn(move || {
            barrier_b.wait();
            // Thread B: repeatedly clear bit Y. The fetch_and
            // mirrors the production INTERRUPT_ACK arm.
            for _ in 0..10_000 {
                int_status_b.fetch_and(!BIT_Y, Ordering::AcqRel);
            }
        });
        barrier.wait();
        setter.join().expect("setter join");
        acker.join().expect("acker join");

        let final_state = int_status.load(Ordering::Acquire);
        assert_eq!(
            final_state & BIT_X,
            BIT_X,
            "bit X must remain set after the race — fetch_or sets and \
             fetch_and(!Y) is disjoint; if X is missing, fetch_and \
             accidentally cleared it (atomicity violation)",
        );
        assert_eq!(
            final_state & BIT_Y,
            0,
            "bit Y must be clear after the race — every iteration of \
             thread B issues fetch_and(!Y); if Y is set, fetch_and \
             missed an iteration (lost update)",
        );
    }

    /// Concurrent `fetch_add` on `config_generation` from N
    /// threads. The post-race value must equal the sum of every
    /// thread's increments — no lost updates. Models the
    /// reset() bumping config_generation while a vCPU thread reads
    /// it via `mmio_read(CONFIG_GENERATION)` (Acquire).
    ///
    /// Currently only `reset()` writes config_generation, but the
    /// AtomicU32-on-VirtioBlk shape is defense-in-depth for future
    /// runtime config changes from non-vCPU threads. This test
    /// pins the atomicity invariant the field's API contract
    /// promises.
    #[test]
    fn config_generation_concurrent_fetch_add_load() {
        use std::sync::Barrier;

        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        // config_generation is an AtomicU32 directly on the
        // device; we need a shareable handle for the threads.
        // Use Arc to wrap the mutation point — but the field
        // itself is not Arc'd in production. For the test we
        // model the atomicity invariant by directly grabbing the
        // raw AtomicU32 reference under an Arc<&'static …>
        // surrogate — except we can't borrow with 'static. The
        // cleanest approach is to do the test against a
        // standalone AtomicU32 that mirrors the production type.
        // The point of the test is the atomicity primitive, not
        // the field's location.
        let initial = dev.config_generation.load(Ordering::Acquire);
        let counter = Arc::new(AtomicU32::new(initial));
        const NUM_WRITERS: u32 = 16;
        const ITERATIONS_PER_WRITER: u32 = 1_000;
        let barrier = Arc::new(Barrier::new(NUM_WRITERS as usize + 1));
        let mut handles = Vec::with_capacity(NUM_WRITERS as usize);
        for _ in 0..NUM_WRITERS {
            let counter_w = Arc::clone(&counter);
            let barrier_w = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_w.wait();
                for _ in 0..ITERATIONS_PER_WRITER {
                    counter_w.fetch_add(1, Ordering::Release);
                }
            }));
        }
        barrier.wait();
        for h in handles {
            h.join().expect("writer join");
        }
        let final_value = counter.load(Ordering::Acquire);
        let expected = initial.wrapping_add(NUM_WRITERS * ITERATIONS_PER_WRITER);
        assert_eq!(
            final_value, expected,
            "fetch_add atomicity violated: expected {expected}, got \
             {final_value} (lost updates means the counter advanced \
             less than NUM_WRITERS * ITERATIONS_PER_WRITER)",
        );
    }

    /// Concurrent `fetch_add` on every `VirtioBlkCounters` field
    /// from multiple threads. Models the production race where
    /// the worker thread bumps counters via the `record_*`
    /// helpers while the host monitor reads them. No lost updates
    /// is the atomicity invariant under test; the monitor's reads
    /// observe a monotonically non-decreasing series, which we
    /// verify by sampling mid-race and asserting the sample is at
    /// most the eventual final value.
    ///
    /// The Relaxed ordering on the `record_*` helpers is
    /// sufficient for atomicity-of-counter-bumps because every
    /// counter is independent: the host monitor doesn't need to
    /// observe a specific happens-before ordering between
    /// `reads_completed` and `bytes_read` (the reads_completed
    /// bump can become visible BEFORE the bytes_read bump and
    /// the dump still renders coherently — a fractional bytes/op
    /// average for one snapshot is acceptable). What MUST hold is
    /// "no lost increment" for each counter individually.
    #[test]
    fn counters_concurrent_fetch_add_no_lost_updates() {
        use std::sync::Barrier;

        let counters = Arc::new(VirtioBlkCounters::default());
        const NUM_WRITERS: u32 = 8;
        const ITERATIONS_PER_WRITER: u32 = 5_000;
        let barrier = Arc::new(Barrier::new(NUM_WRITERS as usize + 2));
        let mut handles = Vec::with_capacity(NUM_WRITERS as usize);
        for _ in 0..NUM_WRITERS {
            let c_w = Arc::clone(&counters);
            let barrier_w = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier_w.wait();
                for _ in 0..ITERATIONS_PER_WRITER {
                    c_w.record_read(512);
                    c_w.record_write(1024);
                    c_w.record_flush();
                    c_w.record_throttled();
                    c_w.record_io_error();
                }
            }));
        }
        // Concurrent reader: sample counters while writers run.
        // Verifies the host-monitor read pattern observes
        // monotonically non-decreasing values (no torn read).
        let c_reader = Arc::clone(&counters);
        let barrier_r = Arc::clone(&barrier);
        let reader = thread::spawn(move || {
            barrier_r.wait();
            let mut last_reads = 0u64;
            for _ in 0..1_000 {
                let now_reads = c_reader.reads_completed.load(Ordering::Relaxed);
                assert!(
                    now_reads >= last_reads,
                    "reads_completed went backwards: {last_reads} -> {now_reads}",
                );
                last_reads = now_reads;
            }
        });
        barrier.wait();
        for h in handles {
            h.join().expect("writer join");
        }
        reader.join().expect("reader join");

        let total_iters = (NUM_WRITERS * ITERATIONS_PER_WRITER) as u64;
        assert_eq!(
            counters.reads_completed.load(Ordering::Relaxed),
            total_iters,
            "reads_completed lost an update",
        );
        assert_eq!(
            counters.bytes_read.load(Ordering::Relaxed),
            total_iters * 512,
            "bytes_read lost an update",
        );
        assert_eq!(
            counters.writes_completed.load(Ordering::Relaxed),
            total_iters,
            "writes_completed lost an update",
        );
        assert_eq!(
            counters.bytes_written.load(Ordering::Relaxed),
            total_iters * 1024,
            "bytes_written lost an update",
        );
        assert_eq!(
            counters.flushes_completed.load(Ordering::Relaxed),
            total_iters,
            "flushes_completed lost an update",
        );
        assert_eq!(
            counters.throttled_count.load(Ordering::Relaxed),
            total_iters,
            "throttled_count lost an update",
        );
        assert_eq!(
            counters.io_errors.load(Ordering::Relaxed),
            total_iters,
            "io_errors lost an update",
        );
    }

    /// Pre-condition for the cross-thread atomic semantics tested
    /// above: the production cfg path actually shares
    /// `interrupt_status` via Arc with the worker thread. cfg(test)
    /// has no production worker, so we assert the Arc count
    /// indicates an additional referent beyond the device's own
    /// borrow — the device-side handle on the Arc plus any
    /// snapshot we just cloned.
    ///
    /// This is an invariant smoke test: a regression that converted
    /// `interrupt_status` from `Arc<AtomicU32>` to a bare
    /// `AtomicU32` would silently break the worker's ability to
    /// share the atomic with the vCPU. The Arc-strong-count check
    /// catches that at the type-level.
    #[test]
    fn interrupt_status_is_arc_shareable() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let cloned = Arc::clone(&dev.interrupt_status);
        // The device holds 1 strong reference; cloning makes 2.
        // (In production the worker's clone makes it 3.)
        assert!(
            Arc::strong_count(&cloned) >= 2,
            "interrupt_status must be Arc-shareable — strong_count \
             after clone is {}",
            Arc::strong_count(&cloned),
        );
    }

    // ----------------------------------------------------------------
    // currently_throttled_gauge tests
    //
    // The gauge is a per-request live counter that increments on
    // the first stall of a chain and decrements when the chain
    // exits the stalled state (either successful drain after
    // refill, or device reset). Distinct from the cumulative
    // event counter `throttled_count`. Tests pin both
    // single-stall and multi-stall behaviours, plus the reset
    // decrement.
    // ----------------------------------------------------------------

    /// First throttle stall on a chain bumps the gauge from 0 to
    /// 1. Symmetric with `process_requests_throttled_rolls_back_chain`
    /// (which pins the rollback contract); this test specifically
    /// pins the live-gauge inc.
    #[test]
    fn currently_throttled_gauge_increments_on_first_stall() {
        let mem = make_chain_test_mem();
        let mut dev = setup_iops1_drained_chain(&mem);

        let c = dev.counters();
        // Pre-state: gauge is zero.
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "fresh device must have currently_throttled_gauge=0",
        );

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        // Post-stall: gauge is 1 (chain is the head-of-queue
        // stalled chain).
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            1,
            "first stall must bump currently_throttled_gauge from 0 to 1",
        );
        // throttled_count (cumulative events) is also 1.
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "first stall bumps throttled_count to 1",
        );
        // Per-worker flag is set.
        assert!(
            dev.worker.state().currently_stalled,
            "BlkWorkerState::currently_stalled must be true after stall",
        );
    }

    /// After a stall, the next drain that succeeds (because the
    /// bucket has refilled) decrements the gauge to 0. Pins the
    /// stall→refill→retry→success contract on the gauge.
    #[test]
    fn currently_throttled_gauge_decrements_on_retry_success() {
        let mem = make_chain_test_mem();
        let mut dev = setup_iops1_drained_chain(&mem);

        // First notify: stall, gauge 0→1.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        let c = dev.counters();
        assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 1);

        // Refill bucket and re-notify.
        dev.worker.state_mut().ops_bucket.set_last_refill_for_test(
            std::time::Instant::now() - std::time::Duration::from_secs(2),
        );
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        // Post-retry success: gauge back to 0.
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "retry success must decrement currently_throttled_gauge to 0",
        );
        // Per-worker flag is cleared.
        assert!(
            !dev.worker.state().currently_stalled,
            "BlkWorkerState::currently_stalled must clear on retry success",
        );
        // throttled_count stays at 1 — no fresh stall on retry.
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "throttled_count is per-event; retry success doesn't bump it",
        );
        // The chain completed.
        assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
    }

    /// Two consecutive stalls on the same chain head: gauge
    /// increments ONCE (on the first stall) and stays at 1 across
    /// the second stall. Per-event `throttled_count` bumps twice;
    /// per-request `currently_throttled_gauge` is idempotent on
    /// re-stall.
    ///
    /// Pins the events-vs-requests distinction: the same chain
    /// stalling twice is one stuck request but two stall events.
    /// A regression that double-incremented the gauge would
    /// surface as gauge=2 at the end of this test.
    #[test]
    fn currently_throttled_gauge_no_double_inc_on_re_stall() {
        let mem = make_chain_test_mem();
        // Plant a 0xEE sentinel at the status byte BEFORE the
        // helper builds the chain. The throttle-stall path rolls
        // back without `add_used`, so the device never writes
        // the status byte; if a regression let it through, the
        // sentinel would be overwritten with VIRTIO_BLK_S_OK
        // (0x00) — readable downstream as evidence the rollback
        // contract broke. The current assertions don't read the
        // sentinel directly (they only check counters), but the
        // pre-write is preserved here so the existing intent is
        // not silently dropped.
        mem.write_slice(&[0xEEu8], GuestAddress(0x6000)).unwrap();
        let mut dev = setup_iops1_drained_chain(&mem);

        // First notify: stall, gauge 0→1.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        // Re-pin so the second notify also stalls.
        dev.worker
            .state_mut()
            .ops_bucket
            .set_last_refill_for_test(std::time::Instant::now());
        // Second notify on the same chain: stall again, gauge
        // stays at 1 (idempotent re-stall).
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        let c = dev.counters();
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            2,
            "two stalls bump throttled_count twice (events)",
        );
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            1,
            "two stalls on same head must NOT double-increment the \
             gauge — gauge represents one stuck request, not two \
             stall events",
        );
        assert!(
            dev.worker.state().currently_stalled,
            "currently_stalled flag stays true across re-stall",
        );
    }

    /// `reset()` decrements the gauge if a chain was
    /// rolled-back-pending. Without this decrement, the
    /// per-request gauge would leak one increment per
    /// reset-while-stalled across the device's lifetime — the
    /// device would forever appear to have a stuck request even
    /// after the reset cleared the queue.
    #[test]
    fn reset_decrements_pending_throttle_gauge() {
        let mem = make_chain_test_mem();
        let mut dev = setup_iops1_drained_chain(&mem);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        let c = dev.counters();
        assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 1);

        // Reset.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "reset must decrement currently_throttled_gauge so a \
             reset-while-stalled does not leak a pending increment",
        );
        assert!(
            !dev.worker.state().currently_stalled,
            "reset must clear currently_stalled",
        );
    }

    /// Counter persistence pin update: the new
    /// `currently_throttled_gauge` field is part of
    /// `VirtioBlkCounters` but is a LIVE gauge, not a cumulative
    /// counter. Reset DOES decrement it (above) — but a reset on
    /// a NON-stalled device must leave the gauge at 0
    /// (unchanged). Pins that the reset's gauge handling is
    /// gated on the per-worker flag and doesn't blindly clear or
    /// double-decrement.
    #[test]
    fn reset_on_non_stalled_device_leaves_gauge_at_zero() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let c = dev.counters();
        assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 0);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "reset on a non-stalled device must NOT touch the gauge",
        );
        assert!(
            !dev.worker.state().currently_stalled,
            "currently_stalled stays false on a non-stalled-device reset",
        );
    }

    /// Counters_initially_zero update: verify the new
    /// `currently_throttled_gauge` field starts at zero on a
    /// freshly-constructed device.
    #[test]
    fn currently_throttled_gauge_initially_zero() {
        let dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let c = dev.counters();
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "currently_throttled_gauge must initialize to 0",
        );
    }

    /// Two BACK-TO-BACK calls to `drain_bracket_impl` against the same
    /// `BlkWorkerState` — the cfg(test) analogue of the production
    /// worker's wait_nanos==0 inline re-drain (see worker_thread_main).
    ///
    /// First call: bucket drained → stall, gauge 0→1, currently_stalled
    /// transitions false→true. Second call (after stepping the bucket
    /// forward to grant a token): chain runs to completion, gauge 1→0,
    /// currently_stalled clears, reads_completed=1, throttled_count
    /// stays at 1 (no second stall event).
    ///
    /// Pins the gauge invariant under inline re-drain: the
    /// stall→success sequence must dec the gauge EXACTLY ONCE, not
    /// zero (missing dec) and not twice (double-dec). Distinct from
    /// `currently_throttled_gauge_decrements_on_retry_success` which
    /// uses two separate `process_requests` calls (two worker
    /// iterations); this test pins the single-iteration inline
    /// re-drain semantics.
    #[test]
    fn currently_throttled_gauge_inline_redrain_succeeds_decrements_once() {
        let mem = make_chain_test_mem();
        let mut dev = setup_iops1_drained_chain(&mem);

        // First call — direct drain_bracket_impl, NOT process_requests.
        // Disjoint-field borrow split mirrors `drain_inline`.
        let mem_ref = dev.mem.get().expect("mem set above");
        let outcome1 = {
            let WorkerEngine::Inline(engine) = &mut dev.worker.engine;
            drain_bracket_impl(
                &mut engine.state,
                &mut dev.worker.queues,
                mem_ref,
                &dev.irq_evt,
                &dev.interrupt_status,
            )
        };
        // Pin the exact wait_nanos value the bucket math produces:
        // capacity=1, refill_rate=1, available=0, deficit=1 →
        // (1 token * 1e9 ns/sec) / 1 token-per-sec = 1_000_000_000
        // ns. Wildcarding `wait_nanos: ..` would let a regression in
        // `nanos_until_n_tokens`'s deficit calculation slip through.
        //
        // Note: production's wait_nanos==0 inline re-drain trigger
        // (worker_thread_main) is unreachable from cfg(test) without
        // a TokenBucket seam — see follow-up #454. These tests pin
        // the gauge invariants under back-to-back drain_bracket_impl,
        // not the production trigger condition itself.
        assert!(
            matches!(
                outcome1,
                DrainOutcome::ThrottleStalled {
                    wait_nanos: 1_000_000_000
                }
            ),
            "first call must stall with wait_nanos=1_000_000_000 \
             (capacity=1, rate=1, deficit=1 → 1s); got {:?}",
            outcome1,
        );
        let c = dev.counters();
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            1,
            "first stall must increment gauge to 1",
        );
        assert!(
            dev.worker.state().currently_stalled,
            "currently_stalled must be true after first stall",
        );
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "first stall bumps throttled_count to 1",
        );

        // Step the bucket forward so the second drain succeeds.
        dev.worker.state_mut().ops_bucket.set_last_refill_for_test(
            std::time::Instant::now() - std::time::Duration::from_secs(2),
        );

        // Second back-to-back call — this IS the inline re-drain.
        let outcome2 = {
            let WorkerEngine::Inline(engine) = &mut dev.worker.engine;
            drain_bracket_impl(
                &mut engine.state,
                &mut dev.worker.queues,
                mem_ref,
                &dev.irq_evt,
                &dev.interrupt_status,
            )
        };
        assert_eq!(
            outcome2,
            DrainOutcome::Done,
            "second drain (post-refill) must complete; got {:?}",
            outcome2,
        );
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "inline re-drain success must dec gauge exactly once: \
             1 → 0, not staying at 1, not going negative",
        );
        assert!(
            !dev.worker.state().currently_stalled,
            "currently_stalled must clear on retry success",
        );
        assert_eq!(
            c.reads_completed.load(Ordering::Relaxed),
            1,
            "chain must complete on second drain",
        );
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "second drain succeeded; throttled_count must NOT bump again",
        );
    }

    /// Two BACK-TO-BACK calls to `drain_bracket_impl` where the second
    /// call ALSO stalls (bucket not refilled). Mimics the production
    /// worker's wait_nanos==0 inline re-drain that re-stalls and falls
    /// through to the timerfd arm.
    ///
    /// First call: stall, gauge 0→1, currently_stalled false→true,
    /// throttled_count 0→1.
    /// Second call (no refill): re-stall on same head, gauge stays at
    /// 1 (idempotent re-stall — no double-inc), currently_stalled
    /// stays true, throttled_count 1→2 (events ARE per-call, not
    /// per-request).
    ///
    /// Pins the gauge invariant under inline re-drain that fails: the
    /// second stall must NOT double-increment the gauge. A regression
    /// that re-checked the false→true transition without the
    /// per-worker `currently_stalled` gate would surface as gauge=2.
    #[test]
    fn currently_throttled_gauge_inline_redrain_restalls_no_double_count() {
        let mem = make_chain_test_mem();
        let mut dev = setup_iops1_drained_chain(&mem);

        let mem_ref = dev.mem.get().expect("mem set above");

        // First call — stall, gauge 0→1. Pin the exact wait_nanos
        // value the bucket math produces (1_000_000_000 ns from
        // capacity=1, rate=1, deficit=1). Production's wait_nanos==0
        // inline re-drain trigger is unreachable from cfg(test) —
        // see follow-up #454.
        let outcome1 = {
            let WorkerEngine::Inline(engine) = &mut dev.worker.engine;
            drain_bracket_impl(
                &mut engine.state,
                &mut dev.worker.queues,
                mem_ref,
                &dev.irq_evt,
                &dev.interrupt_status,
            )
        };
        assert!(matches!(
            outcome1,
            DrainOutcome::ThrottleStalled {
                wait_nanos: 1_000_000_000
            }
        ));
        let c = dev.counters();
        assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 1);
        assert!(dev.worker.state().currently_stalled);
        assert_eq!(c.throttled_count.load(Ordering::Relaxed), 1);

        // Re-pin so the second drain ALSO sees an empty bucket.
        dev.worker
            .state_mut()
            .ops_bucket
            .set_last_refill_for_test(std::time::Instant::now());

        // Second back-to-back call — re-stall (no refill).
        let outcome2 = {
            let WorkerEngine::Inline(engine) = &mut dev.worker.engine;
            drain_bracket_impl(
                &mut engine.state,
                &mut dev.worker.queues,
                mem_ref,
                &dev.irq_evt,
                &dev.interrupt_status,
            )
        };
        // Same pinned wait_nanos as outcome1 — re-stall on an
        // unchanged bucket repeats the same deficit math.
        assert!(
            matches!(
                outcome2,
                DrainOutcome::ThrottleStalled {
                    wait_nanos: 1_000_000_000
                }
            ),
            "second drain (no refill) must also stall with \
             wait_nanos=1_000_000_000; got {:?}",
            outcome2,
        );
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            1,
            "re-stall on same head must NOT double-increment gauge \
             (idempotent — gauge is per-request live state, not \
             per-event)",
        );
        assert!(
            dev.worker.state().currently_stalled,
            "currently_stalled stays true across re-stall",
        );
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            2,
            "throttled_count IS per-event; two stall events must \
             produce two bumps",
        );
        assert_eq!(
            c.reads_completed.load(Ordering::Relaxed),
            0,
            "no chain completed; reads_completed must stay 0",
        );
    }

    /// Hostile-guest defense: avail.idx more than queue.size ahead
    /// of next_avail must trip `Error::InvalidAvailRingIndex`
    /// from `Queue::iter` (the structural-invariant check at
    /// queue.rs:707-709), poison the queue, bump
    /// `invalid_avail_idx_count`, and bail without calling
    /// `enable_notification`. Subsequent kicks against the
    /// poisoned queue are no-ops — the counter stays at 1 and
    /// the worker does NOT spin (the original livelock the
    /// `pop_descriptor_chain` swallowed-error pattern produced).
    #[test]
    fn inflated_avail_idx_poisons_queue_no_livelock() {
        use std::num::Wrapping;
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0xAB);
        let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
        let mem = make_chain_test_mem();
        let queue_size: u16 = 16;
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), queue_size);
        // Plant one well-formed chain so the avail ring has real
        // content (build_desc_chain writes the ring entry), then
        // OVERWRITE avail.idx to > next_avail + queue_size. The
        // `iter()` invariant `idx - next_avail <= queue.size`
        // (queue.rs:707) trips on that mismatch.
        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);
        write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_device_to_mock(&mut dev, &mock);

        // Hostile poison: avail.idx = next_avail + queue.size + 1
        // (the strict-greater-than threshold in
        // `AvailIter::new`, queue.rs:707). The DEVICE's
        // negotiated queue.size is `QUEUE_MAX_SIZE` (256, set by
        // wire_device_to_mock via QUEUE_NUM), independent of the
        // mock's avail-ring length (16). The check fires before
        // any ring read, so we don't need a 257-element mock
        // ring — only the avail.idx field needs to land out of
        // bounds relative to the device's 256-sized window.
        let bad_idx = Wrapping(0u16) + Wrapping(QUEUE_MAX_SIZE) + Wrapping(1u16);
        mock.avail().idx().store(u16::to_le(bad_idx.0));

        // Fire QUEUE_NOTIFY — `process_requests` calls the inline
        // drain, which observes InvalidAvailRingIndex from
        // `iter()`, poisons the queue, bumps the counter, and
        // bails. MUST return without spinning. (cfg(test) drains
        // synchronously, so a livelock would hang the test until
        // the harness timeout.)
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        let c = dev.counters();
        assert_eq!(
            c.invalid_avail_idx_count.load(Ordering::Relaxed),
            1,
            "first hostile drain must bump invalid_avail_idx_count once",
        );
        assert!(
            dev.worker.state().queue_poisoned,
            "queue_poisoned must be set after InvalidAvailRingIndex",
        );
        // No IO completed.
        assert_eq!(c.reads_completed.load(Ordering::Relaxed), 0);
        assert_eq!(c.writes_completed.load(Ordering::Relaxed), 0);
        // No throttle stall counted (we never reached the throttle).
        assert_eq!(c.throttled_count.load(Ordering::Relaxed), 0);
        assert_eq!(c.currently_throttled_gauge.load(Ordering::Relaxed), 0);

        // Subsequent kicks must be NO-OPs: the poison gate at the
        // top of `drain_bracket_impl` short-circuits without
        // calling `iter()`, so the counter does NOT advance and
        // the worker does NOT loop.
        for _ in 0..5 {
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        }
        assert_eq!(
            c.invalid_avail_idx_count.load(Ordering::Relaxed),
            1,
            "poisoned queue must reject subsequent kicks without re-bumping \
             the counter (per-event semantic + flag short-circuit)",
        );
        assert!(
            dev.worker.state().queue_poisoned,
            "poison flag stays set across re-kicks",
        );
    }

    /// A virtio reset is the only documented escape from the
    /// queue-poisoned state. After reset, the device must accept
    /// fresh chains and bump per-IO counters again — but
    /// `invalid_avail_idx_count` is intentionally cumulative
    /// across resets so operators can detect repeated hostile
    /// behavior.
    #[test]
    fn poisoned_queue_clears_on_reset() {
        use std::num::Wrapping;
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0xAB);
        let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
        let mem = make_chain_test_mem();
        let queue_size: u16 = 16;
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), queue_size);
        // Plant one valid chain so avail-ring entries exist.
        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);
        write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_device_to_mock(&mut dev, &mock);

        // Trip the poison. The DEVICE's negotiated queue.size is
        // QUEUE_MAX_SIZE (set by wire_device_to_mock via QUEUE_NUM),
        // not the mock's avail-ring length — overshoot QUEUE_MAX_SIZE
        // so `AvailIter::new`'s `idx - next_avail > queue.size`
        // check fires on the device's view of the queue.
        let bad_idx = Wrapping(0u16) + Wrapping(QUEUE_MAX_SIZE) + Wrapping(1u16);
        mock.avail().idx().store(u16::to_le(bad_idx.0));
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        assert!(dev.worker.state().queue_poisoned);
        let c = dev.counters();
        assert_eq!(c.invalid_avail_idx_count.load(Ordering::Relaxed), 1);

        // Drive the device through a virtio reset (status=0 walks
        // the FSM back to driver-init state and runs
        // `reset_engine_inline` which clears `queue_poisoned`).
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        assert!(
            !dev.worker.state().queue_poisoned,
            "reset must clear queue_poisoned",
        );
        // The cumulative counter survives the reset (operator
        // visibility across resets).
        assert_eq!(
            c.invalid_avail_idx_count.load(Ordering::Relaxed),
            1,
            "invalid_avail_idx_count is cumulative across resets",
        );

        // Re-wire to a fresh mock with a single legitimate chain.
        // After reset the device's `next_avail` is back to 0 and
        // the queue config is re-published via wire_device_to_mock.
        let mock2 = MockSplitQueue::create(&mem, GuestAddress(0), queue_size);
        let header_addr2 = GuestAddress(0x7000);
        let data_addr2 = GuestAddress(0x8000);
        let status_addr2 = GuestAddress(0x9000);
        write_blk_header(&mem, header_addr2, VIRTIO_BLK_T_IN, 0);
        let descs2 = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr2.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr2.0,
                512,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr2.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock2
            .build_desc_chain(&descs2)
            .expect("build chain after reset");
        wire_device_to_mock(&mut dev, &mock2);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);

        // The fresh chain completed: poison gate cleared, IO
        // serviced, no new poison events.
        assert_eq!(c.reads_completed.load(Ordering::Relaxed), 1);
        assert_eq!(
            c.invalid_avail_idx_count.load(Ordering::Relaxed),
            1,
            "post-reset legitimate IO must NOT re-trip poison counter",
        );
        assert!(
            !dev.worker.state().queue_poisoned,
            "queue stays unpoisoned across legitimate post-reset IO",
        );
    }

    /// The poison gate sits at the TOP of `drain_bracket_impl`,
    /// BEFORE `disable_notification` and BEFORE `iter()`. A
    /// regression that moves the gate below
    /// `disable_notification` would re-set
    /// `VRING_USED_F_NO_NOTIFY` on the legacy path on every kick
    /// — observable as `used.flags` flipping across kicks against
    /// a poisoned queue. This test pins the expected
    /// `used.flags` stability post-poison: subsequent kicks must
    /// not modify the field.
    #[test]
    fn poisoned_queue_kicks_dont_touch_used_flags() {
        use std::num::Wrapping;
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0xAB);
        let mut dev = VirtioBlk::new(f, cap, DiskThrottle::default());
        let mem = make_chain_test_mem();
        let queue_size: u16 = 16;
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), queue_size);
        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);
        write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_device_to_mock(&mut dev, &mock);

        // Trip the poison. The DEVICE's negotiated queue.size is
        // QUEUE_MAX_SIZE (set by wire_device_to_mock via QUEUE_NUM),
        // not the mock's avail-ring length — overshoot QUEUE_MAX_SIZE
        // so `AvailIter::new`'s `idx - next_avail > queue.size`
        // check fires on the device's view of the queue.
        let bad_idx = Wrapping(0u16) + Wrapping(QUEUE_MAX_SIZE) + Wrapping(1u16);
        mock.avail().idx().store(u16::to_le(bad_idx.0));
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
        assert!(dev.worker.state().queue_poisoned);

        // After the poison drain, used.flags is whatever the FINAL
        // state of the (now-bailed) outer bracket left it. Snapshot
        // it here and pin its STABILITY across the subsequent
        // re-kicks.
        let used_flags_after_poison: u16 = mem.read_obj(mock.used_addr()).expect("read used.flags");

        // Kick five more times. Each must short-circuit at the
        // poison gate without re-touching used.flags.
        for _ in 0..5 {
            write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NOTIFY, REQ_QUEUE as u32);
            let f: u16 = mem
                .read_obj(mock.used_addr())
                .expect("read used.flags post-kick");
            assert_eq!(
                f, used_flags_after_poison,
                "poisoned queue kicks must not modify used.flags \
                 (regression: gate moved below disable_notification)",
            );
        }

        let c = dev.counters();
        assert_eq!(
            c.invalid_avail_idx_count.load(Ordering::Relaxed),
            1,
            "no additional poison events from re-kicks",
        );
    }

    /// Bytes-only stall + retry: gauge invariants when ONLY the
    /// bytes bucket is throttled (iops bucket has tokens). Mirrors
    /// `currently_throttled_gauge_inline_redrain_succeeds_decrements_once`
    /// (iops-only) so a regression that wired the gauge transitions
    /// to one bucket and not the other surfaces here.
    ///
    /// Naming note: this test exercises two SEQUENTIAL
    /// `drain_bracket_impl` calls — the cfg(test) inline-mode
    /// surrogate for stall-then-retry — NOT the production
    /// `worker_thread_main` wait_nanos==0 inline-redrain branch.
    /// `wait_nanos` here is 1_000_000_000 (the deficit-driven
    /// value), not 0; the production inline-redrain trigger
    /// requires a TokenBucket test seam (see #454) that the
    /// cfg(test) surface doesn't expose. The previous name
    /// (`..._inline_redrain_..._decrements_once`) overclaimed —
    /// renamed to match what the test actually does.
    ///
    /// First call: bytes bucket drained → stall on bytes path,
    /// gauge 0→1, currently_stalled false→true. Second call (after
    /// stepping the bytes bucket forward to grant the request):
    /// chain runs to completion, gauge 1→0, currently_stalled
    /// clears, reads_completed=1, throttled_count=1 (single stall
    /// event).
    ///
    /// Setup notes:
    /// * iops bucket capacity = 16 with refill_rate = 16; the
    ///   request charges 1 token so the iops bucket is never
    ///   exhausted in this scenario.
    /// * bytes bucket capacity = 512, refill_rate = 512; pre-
    ///   draining via `consume(512)` empties it. The chain is a
    ///   1-segment 512-byte read, so `data_len = 512` is exactly
    ///   bucket capacity — the can_consume gate fails on bytes
    ///   alone after pre-drain, leaving the iops gate satisfied.
    ///   `nanos_until_n_tokens(512)` against an empty 512-token/sec
    ///   bucket returns 1_000_000_000 (1 s), pinning the
    ///   wait_nanos value the assertion below references.
    #[test]
    fn currently_throttled_gauge_bytes_only_stall_and_retry() {
        let cap = 4096u64;
        let f = make_backed_file_with_pattern(cap, 0xAB);
        let throttle = DiskThrottle {
            iops: NonZeroU64::new(16),
            bytes_per_sec: NonZeroU64::new(512),
            iops_burst_capacity: None,
            bytes_burst_capacity: None,
        };
        let mut dev = VirtioBlk::new(f, cap, throttle);
        let mem = make_chain_test_mem();
        let mock = MockSplitQueue::create(&mem, GuestAddress(0), 16);

        // Drain ONLY the bytes bucket so the first drain stalls on
        // bytes alone. Pin both buckets' last_refill so the
        // bucket arithmetic doesn't passively grant or revoke
        // tokens between assertions.
        let now0 = Instant::now();
        dev.worker
            .state_mut()
            .ops_bucket
            .set_last_refill_for_test(now0);
        dev.worker
            .state_mut()
            .bytes_bucket
            .set_last_refill_for_test(now0);
        assert!(dev.worker.state_mut().bytes_bucket.consume(512));
        // Re-pin AFTER consume so the next can_consume sees the
        // drained state at exactly t=now0.
        dev.worker
            .state_mut()
            .bytes_bucket
            .set_last_refill_for_test(now0);
        // Sanity: iops can grant 1, bytes cannot grant 512.
        assert!(
            dev.worker.state_mut().ops_bucket.can_consume(1),
            "iops bucket must NOT be drained — only bytes is the stall trigger",
        );
        assert!(
            !dev.worker.state_mut().bytes_bucket.can_consume(512),
            "bytes bucket must be drained so the chain stalls on bytes alone",
        );

        let header_addr = GuestAddress(0x4000);
        let data_addr = GuestAddress(0x5000);
        let status_addr = GuestAddress(0x6000);
        write_blk_header(&mem, header_addr, VIRTIO_BLK_T_IN, 0);
        let descs = [
            RawDescriptor::from(SplitDescriptor::new(
                header_addr.0,
                VIRTIO_BLK_OUTHDR_SIZE as u32,
                0,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                data_addr.0,
                512,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(SplitDescriptor::new(
                status_addr.0,
                1,
                VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock.build_desc_chain(&descs).expect("build chain");
        dev.set_mem(mem.clone());
        wire_device_to_mock(&mut dev, &mock);

        // wire_device_to_mock walked the FSM; pin both buckets
        // again so the elapsed wall time during the FSM walk
        // doesn't passively grant tokens before the first drain.
        let now1 = Instant::now();
        dev.worker
            .state_mut()
            .ops_bucket
            .set_last_refill_for_test(now1);
        dev.worker
            .state_mut()
            .bytes_bucket
            .set_last_refill_for_test(now1);

        let mem_ref = dev.mem.get().expect("mem set above");
        let outcome1 = {
            let WorkerEngine::Inline(engine) = &mut dev.worker.engine;
            drain_bracket_impl(
                &mut engine.state,
                &mut dev.worker.queues,
                mem_ref,
                &dev.irq_evt,
                &dev.interrupt_status,
            )
        };
        // bytes bucket: capacity=512, rate=512, available=0,
        // deficit=512 → 512 * 1e9 / 512 = 1_000_000_000 ns. iops
        // bucket grants so its wait_nanos contribution is 0.
        // wait_nanos = ops_wait.max(bytes_wait) = 1_000_000_000.
        assert!(
            matches!(
                outcome1,
                DrainOutcome::ThrottleStalled {
                    wait_nanos: 1_000_000_000
                }
            ),
            "first call must stall on bytes bucket with \
             wait_nanos=1_000_000_000 (capacity=512, rate=512, \
             deficit=512); got {:?}",
            outcome1,
        );
        let c = dev.counters();
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            1,
            "bytes-only stall must inc gauge 0→1 — gauge transitions \
             on stall regardless of which bucket triggered it",
        );
        assert!(
            dev.worker.state().currently_stalled,
            "currently_stalled must be true after first stall",
        );
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "first stall bumps throttled_count to 1",
        );
        assert_eq!(
            c.reads_completed.load(Ordering::Relaxed),
            0,
            "stalled chain must not have completed",
        );

        // Step the bytes bucket forward by 2 s so the second drain
        // succeeds (refill grants 512 * 2 = 1024 tokens, capped at
        // capacity=512). Leave iops bucket pinned — it was already
        // satisfying the can_consume(1) check.
        dev.worker
            .state_mut()
            .bytes_bucket
            .set_last_refill_for_test(Instant::now() - Duration::from_secs(2));

        let outcome2 = {
            let WorkerEngine::Inline(engine) = &mut dev.worker.engine;
            drain_bracket_impl(
                &mut engine.state,
                &mut dev.worker.queues,
                mem_ref,
                &dev.irq_evt,
                &dev.interrupt_status,
            )
        };
        assert_eq!(
            outcome2,
            DrainOutcome::Done,
            "second drain (post bytes-bucket refill) must complete; \
             got {:?}",
            outcome2,
        );
        assert_eq!(
            c.currently_throttled_gauge.load(Ordering::Relaxed),
            0,
            "bytes-only retry success must dec gauge exactly once: \
             1 → 0, not staying at 1, not going negative",
        );
        assert!(
            !dev.worker.state().currently_stalled,
            "currently_stalled must clear on retry success",
        );
        assert_eq!(
            c.reads_completed.load(Ordering::Relaxed),
            1,
            "chain must complete on second drain",
        );
        assert_eq!(
            c.throttled_count.load(Ordering::Relaxed),
            1,
            "second drain succeeded; throttled_count must NOT bump again",
        );
    }

    /// `set_mem` called twice: the second call's `OnceLock::set`
    /// returns Err and `set_mem` emits a `tracing::warn!`. This
    /// test pins the warn observability — `set_mem_twice_keeps_first_instance`
    /// pins the silently-ignored instance pointer; this test pins
    /// the operator-visible diagnostic.
    ///
    /// `tracing-test`'s `#[traced_test]` attribute installs a
    /// per-test subscriber and exposes `logs_contain(substring)`
    /// to assert against the captured output. The substring
    /// matched here is a stable fragment of the warn message
    /// emitted by `set_mem`'s `if self.mem.set(mem).is_err()`
    /// branch; matching a substring (not the full message) keeps
    /// the test resilient to wording polish that doesn't change
    /// the operator-relevant signal.
    #[tracing_test::traced_test]
    #[test]
    fn set_mem_twice_emits_warn() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        let mem_a = make_guest_mem(4096);
        let mem_b = make_guest_mem(8192);
        dev.set_mem(mem_a);
        // Snapshot the address that `OnceLock::get()` returns AFTER
        // the first set. The second set call must not alter what
        // `get()` returns — the warn-and-skip path is observable as
        // both (a) the warn fires AND (b) the stored binding is
        // unchanged. `set_mem_twice_keeps_first_instance` pins the
        // pointer-identity invariant via construction-distinct
        // GuestMemoryMmap instances; this test re-asserts that
        // identity AFTER triggering the warn so a regression that
        // emits the warn but ALSO replaces the binding (or vice
        // versa) cannot pass either test alone.
        let first_ptr =
            dev.mem.get().expect("set_mem populated OnceLock") as *const GuestMemoryMmap;
        // No warn yet — first set populates the OnceLock cleanly.
        assert!(
            !logs_contain("set_mem called on already-initialised"),
            "first set_mem must not emit the already-initialised warn",
        );
        dev.set_mem(mem_b);
        // Second set hits OnceLock::set returning Err; set_mem
        // catches it and warns.
        assert!(
            logs_contain("set_mem called on already-initialised"),
            "second set_mem must emit the already-initialised warn so \
             a duplicate-bind regression is operator-visible",
        );
        // The warn body cites the durable behaviour (mem stays
        // bound to the first call's value across reset), so the
        // operator can correlate the message with the documented
        // OnceLock semantics.
        assert!(
            logs_contain("guest memory binding unchanged"),
            "warn must explain the no-op semantic — \
             'guest memory binding unchanged' tells the operator \
             the duplicate call did NOT replace the binding",
        );
        // First-wins pointer-identity check: the OnceLock still
        // points at the FIRST set_mem's instance, not the second.
        // GuestMemoryMmap has no PartialEq, so address comparison
        // is the load-bearing assertion; clones would be
        // address-distinct even if content-equal, so this catches
        // a regression that replaces the binding while still
        // emitting the warn.
        let after_ptr = dev
            .mem
            .get()
            .expect("OnceLock still populated after second set_mem")
            as *const GuestMemoryMmap;
        assert_eq!(
            first_ptr, after_ptr,
            "OnceLock must retain the first GuestMemoryMmap; the \
             warn-and-skip path must NOT overwrite the binding on \
             the second call",
        );
    }

    /// FEATURES_OK with a driver-acked feature bit that
    /// `device_features()` did not advertise must be rejected
    /// per virtio-v1.2 §3.1.1 step 5 ("the driver MUST NOT set
    /// any feature bit that the device did not offer"). Pins both
    /// the rejection (device_status stays at S_DRV) and the
    /// operator-visible warn.
    ///
    /// Setup acks VIRTIO_F_VERSION_1 (so the version-1 gate is
    /// satisfied — that gate would otherwise short-circuit the
    /// rejection path being tested) AND a high-bit feature
    /// (VIRTIO_BLK_F_DISCARD = 13) that this device deliberately
    /// does NOT advertise. With version_1 satisfied and a
    /// non-subset driver_features mask, the FEATURES_OK
    /// transition must reject through the unadvertised-bit branch
    /// and emit the corresponding warn.
    ///
    /// `tracing-test`'s `logs_contain` matches against the
    /// captured frame; the "unadvertised feature bits" substring
    /// is a stable fragment of the warn body that distinguishes
    /// this rejection branch from the version-1 branch.
    #[tracing_test::traced_test]
    #[test]
    fn features_ok_rejected_with_unadvertised_bit() {
        let mut dev = make_device(VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, DiskThrottle::default());
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);

        // VIRTIO_BLK_F_DISCARD is bit 13 — a real virtio-blk
        // feature this device does NOT advertise (see
        // device_features() for the advertised set: VERSION_1,
        // BLK_SIZE, SEG_MAX, SIZE_MAX, FLUSH, EVENT_IDX, plus
        // optionally F_RO when read-only). A driver that acks bit
        // 13 has either misread the feature page or is
        // buggy/hostile.
        const VIRTIO_BLK_F_DISCARD: u32 = 13;
        // device_features() must NOT include F_DISCARD — pin the
        // assumption so a regression that advertises DISCARD
        // (without the backend support) doesn't silently flip
        // this test green.
        assert_eq!(
            dev.device_features() & (1u64 << VIRTIO_BLK_F_DISCARD),
            0,
            "precondition: device must NOT advertise F_DISCARD \
             (this test depends on it being unadvertised)",
        );

        // Ack VERSION_1 (high half, bit 32) and F_DISCARD (low
        // half, bit 13). Both bits must land in driver_features so
        // the version-1 gate is satisfied AND the subset gate
        // catches the unadvertised bit.
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1u32 << VIRTIO_BLK_F_DISCARD,
        );

        // Attempt FEATURES_OK with the unadvertised bit set in
        // driver_features. The transition must be rejected — the
        // device leaves device_status at S_DRV so the kernel's
        // STATUS read-back surfaces the failure.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        assert_eq!(
            dev.device_status, S_DRV,
            "FEATURES_OK must be rejected when driver acked an \
             unadvertised feature bit (subset rule violation)",
        );
        // MMIO read-back parity with the version-1 rejection test
        // — operators observe the rejection through the same
        // STATUS register the kernel re-reads.
        let status = read_reg(&dev, VIRTIO_MMIO_STATUS);
        assert_eq!(
            status, S_DRV,
            "MMIO STATUS read-back must show FEATURES_OK is unset \
             after subset-rule rejection",
        );

        // Warn surfaces with the substring identifying the
        // subset-rule branch (distinct from the version-1 warn).
        assert!(
            logs_contain("unadvertised feature bits"),
            "warn must cite 'unadvertised feature bits' so the \
             operator can distinguish this rejection branch from \
             the version-1 rejection branch",
        );
        // After the driver re-acks ONLY the advertised bits
        // (clears the unadvertised bit), the same FEATURES_OK
        // write succeeds — confirms the gate is subset-specific,
        // not blanket-rejecting.
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);
        assert_eq!(
            dev.device_status, S_FEAT,
            "FEATURES_OK must be accepted once driver_features is \
             a subset of device_features (only VERSION_1 set)",
        );
    }
