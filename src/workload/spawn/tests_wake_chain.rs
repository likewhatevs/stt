//! Spawn-pipeline tests — wake_chain group.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::types::*;
use super::super::worker::*;
use super::testing::*;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

#[test]
fn worker_group_size_wake_chain() {
    // WakeChain group_size == depth (per-chain). Spawn-side
    // allocates one futex region per chain; the chain count
    // derives from `num_workers / depth` so multi-chain
    // configurations need num_workers ≥ depth + 1 multiple.
    let wc = WorkType::wake_chain(8, WakeMechanism::Futex, Duration::from_micros(100));
    assert_eq!(wc.worker_group_size(), Some(8));
    let wc1 = WorkType::wake_chain(3, WakeMechanism::Pipe, Duration::from_micros(50));
    assert_eq!(wc1.worker_group_size(), Some(3));
}
/// `WakeChain { wake: WakeMechanism::Pipe }` must dispatch
/// the bootstrap byte ONLY at stage 0. The dispatch site
/// (`workload.rs::worker_main` pipe-mode arm) gates the
/// first-iteration `libc::write` behind both `iterations == 0`
/// AND `pos == 0`. A regression that drops `pos == 0` would
/// have every stage fire its bootstrap byte simultaneously at
/// iteration 0, putting `depth` bytes in flight on the ring
/// instead of one. The chain still runs (each stage's
/// predecessor wrote its bootstrap), but throughput rises by
/// factor `depth` because each per-stage poll succeeds
/// immediately on a pre-queued byte instead of waiting
/// `work_per_hop` for its predecessor's CPU burst + write to
/// arrive.
///
/// Test signature: pin the total iteration count across all
/// stages to the wake-bubble throughput ceiling. With
/// `depth=4` and `work_per_hop=50ms` over a 1-second window:
///
/// - Correct: ~20 iters total across all 4 stages (one wake
///   per `work_per_hop`, summed regardless of which stage
///   produced it). Per-stage rate = 1 / (depth × work_per_hop).
/// - Buggy: ~80 iters total (≈ depth × correct). Per-stage
///   rate ≈ 1 / work_per_hop because every stage's poll picks
///   up an already-queued byte from its predecessor's
///   bootstrap write.
///
/// Threshold `total ≤ 40` catches any ratio > 2× while
/// retaining margin against scheduling jitter on noisy hosts.
/// `work_per_hop=50ms` is intentionally an order of magnitude
/// above typical scheduling-noise floors so the per-iter rate
/// stays bounded by the spin loop, not the kernel scheduler.
/// Fork mode is canonical (default) — under Thread mode the
/// shared fd table makes the bug behavior identical, and
/// Fork covers the production path.
///
/// This test is intentionally narrow — it asserts the
/// bootstrap-once invariant by failing catastrophically when
/// the guard is dropped, and does not exercise normal
/// operation. Normal-operation coverage lives in
/// [`spawn_thread_with_wake_chain_pipe`] and similar.
#[test]
fn wake_chain_pipe_bootstrap_once_invariant() {
    // DEPTH=4 is load-bearing: with depth=2 the buggy total
    // (40) equals the threshold, eliminating discrimination
    // margin. depth=4 gives correct=20, buggy=80,
    // threshold=40 with 2x margin on each side.
    const DEPTH: usize = 4;
    const WORK_PER_HOP_MS: u64 = 50;
    const TEST_WINDOW_MS: u64 = 1000;
    // Threshold 40 is the geometric midpoint of [20, 80]
    // (sqrt(20*80)=40), placing the boundary equidistant in
    // log-space from the correct upper bound (20) and the
    // buggy expectation (80). The arithmetic midpoint would
    // be 50.
    const TOTAL_ITER_THRESHOLD: u64 = 40;

    if require_isolated_cpus(DEPTH, "wake_chain_pipe_bootstrap_once_invariant") {
        return;
    }

    let config = WorkloadConfig {
        num_workers: DEPTH,
        work_type: WorkType::WakeChain {
            depth: DEPTH,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_millis(WORK_PER_HOP_MS),
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&config).expect("WakeChain wake=Pipe spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(TEST_WINDOW_MS));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        DEPTH,
        "WakeChain wake=Pipe collects one report per worker"
    );
    let total_iters: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(
        total_iters <= TOTAL_ITER_THRESHOLD,
        "WakeChain wake=Pipe total iterations across {DEPTH} stages \
         exceeded {TOTAL_ITER_THRESHOLD} over {TEST_WINDOW_MS}ms with \
         work_per_hop={WORK_PER_HOP_MS}ms (got {total_iters}). The \
         bootstrap-once invariant requires only stage 0 to fire the \
         initial pipe write; if every stage fires a bootstrap byte at \
         iteration 0, the ring carries {DEPTH} simultaneous bytes and \
         per-stage throughput rises by factor {DEPTH}. Expected \
         correct total ~{}; expected buggy total ~{}. Per-worker \
         reports: {:?}",
        TEST_WINDOW_MS / WORK_PER_HOP_MS,
        (TEST_WINDOW_MS / WORK_PER_HOP_MS) * (DEPTH as u64),
        reports,
    );
    // Lower bound: the correct chain MUST make at least one
    // full ring round-trip over a 1s window with 50ms hops.
    // One round-trip = DEPTH stages = 4 iters (the byte
    // visits each stage exactly once). A total below that
    // indicates the chain deadlocked or stalled at the
    // bootstrap site (a different bug than the one this test
    // primarily guards, but worth surfacing here too).
    // `>= 4` is tighter than `> 0` without producing false
    // positives — the correct expectation is ~20 iters.
    assert!(
        total_iters >= 4,
        "WakeChain wake=Pipe made fewer than one ring round-trip \
         over {TEST_WINDOW_MS}ms (got {total_iters}, expected ≥ 4) — \
         the bootstrap byte never completed a full lap. Per-worker \
         reports: {:?}",
        reports,
    );
}
/// `WakeChain { wake: WakeMechanism::Pipe }` must dispatch
/// the bootstrap byte at iteration 0 ONLY. The dispatch site
/// gates the bootstrap `libc::write` behind both
/// `iterations == 0` AND `pos == 0`. A regression that drops
/// `iterations == 0` (keeping only `pos == 0`) has stage 0
/// fire its bootstrap byte every iteration into pipe[0],
/// queueing extra bytes that stage 1 reads without blocking.
/// Stage 0's poll on pipe[depth-1] then resolves immediately
/// against stage 1's prior-iteration write, so stage 0's
/// throughput rises beyond the `work_per_hop`-bounded
/// per-stage ceiling — the symptom is "stage 0 running ahead
/// of peers", distinct from the `pos == 0` regression covered
/// by `wake_chain_pipe_bootstrap_once_invariant` (which has
/// every stage running ahead at iteration 0 simultaneously).
///
/// Parameters: depth=2, num_workers=2, work_per_hop=50ms,
/// 1000ms window. Correct steady-state: ~10 iters per stage,
/// ~20 total. The `[4, 25]` window catches both shoulders
/// without flaking on jitter — 4 ensures progress (matches the
/// `>= 4` lower bound from the bootstrap-once test); 25 is the
/// geometric upper bound that excludes the regression's
/// stage-0-runs-ahead pattern (with the bug, even a 50% boost
/// to stage 0 pushes total above 25). Per-stage iteration
/// count comparison adds a sharper signal: with the bug,
/// stage 0's iters grow disproportionately while stage 1
/// stays bounded by its CPU burst; the cross-stage ratio
/// constraint flags the asymmetry directly.
#[test]
fn wake_chain_pipe_no_repeat_bootstrap_invariant() {
    const DEPTH: usize = 2;
    const NUM_WORKERS: usize = 2;
    const WORK_PER_HOP_MS: u64 = 50;
    const TEST_WINDOW_MS: u64 = 1000;
    const TOTAL_ITER_LOWER: u64 = 4;
    const TOTAL_ITER_UPPER: u64 = 25;

    if require_isolated_cpus(DEPTH, "wake_chain_pipe_no_repeat_bootstrap_invariant") {
        return;
    }

    let config = WorkloadConfig {
        num_workers: NUM_WORKERS,
        work_type: WorkType::WakeChain {
            depth: DEPTH,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_millis(WORK_PER_HOP_MS),
        },
        ..Default::default()
    };
    let mut h =
        WorkloadHandle::spawn(&config).expect("WakeChain wake=Pipe depth=2 spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(TEST_WINDOW_MS));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        NUM_WORKERS,
        "WakeChain wake=Pipe depth=2 collects one report per worker"
    );
    let total_iters: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(
        (TOTAL_ITER_LOWER..=TOTAL_ITER_UPPER).contains(&total_iters),
        "WakeChain wake=Pipe depth=2 total iterations over \
         {TEST_WINDOW_MS}ms with work_per_hop={WORK_PER_HOP_MS}ms must \
         land in [{TOTAL_ITER_LOWER}, {TOTAL_ITER_UPPER}] (got \
         {total_iters}). Correct steady-state is ~20 (one wake per \
         work_per_hop, summed across stages); a regression that drops \
         the iterations==0 guard has stage 0 fire its bootstrap byte \
         every iteration, queueing extra bytes in pipe[0] and \
         unblocking stage 1's poll instantly — stage 0's poll on \
         pipe[1] then resolves on stage 1's prior-iteration write, \
         pushing stage 0 above the work_per_hop-bounded ceiling. \
         Per-worker reports: {:?}",
        reports,
    );
    // Per-stage comparison: with the regression, stage 0 runs
    // ahead of its peer because its repeat-bootstrap writes
    // collapse the wait on pipe[depth-1]. Bound the cross-stage
    // ratio at 2× — a generous threshold that absorbs scheduler
    // jitter while still flagging the systematic skew the
    // regression produces. Stage 0 = reports[0], stage 1 =
    // reports[1] (worker-index ordering preserved by
    // `stop_and_collect`'s children iteration). Use saturating
    // arithmetic so a degenerate `iterations == 0` peer doesn't
    // panic the comparison before the lower-bound assertion
    // surfaces it.
    let stage0_iters = reports[0].iterations;
    let stage1_iters = reports[1].iterations;
    let max_iters = stage0_iters.max(stage1_iters);
    let min_iters = stage0_iters.min(stage1_iters);
    assert!(
        min_iters > 0 && max_iters <= min_iters.saturating_mul(2),
        "WakeChain wake=Pipe depth=2 per-stage iteration counts must \
         stay within 2× of each other (stage0={stage0_iters}, \
         stage1={stage1_iters}). A regression that drops the \
         iterations==0 guard has stage 0 fire its bootstrap byte every \
         iteration, queueing bytes that bypass stage 1's wait and \
         letting stage 0's poll resolve instantly on stage 1's prior \
         write — symptom: stage 0 runs ahead of stage 1. Per-worker \
         reports: {:?}",
        reports,
    );
}
/// `WakeChain { wake: WakeMechanism::Pipe }` multi-chain
/// bootstrap independence. Each chain owns its own pipe ring
/// and its own bootstrap stage; chains do not share fds and
/// must run independently — a regression that crosses chain
/// indices (e.g. uses a global "stage 0" rather than the
/// per-chain `pos == 0`) would have one chain's bootstrap
/// activate the other's pipes, or have only one chain make
/// progress while the other stalls.
///
/// Parameters: depth=4, num_workers=8 (2 chains of 4),
/// work_per_hop=50ms, 1000ms window. Workers `[0..4)` are
/// chain 0, workers `[4..8)` are chain 1 — `chain_idx = i /
/// depth` matches the spawn-side derivation at
/// `workload.rs:4600` (`chain_pipes_base + i / depth`).
/// Each chain's correct steady-state is ~20 iters total
/// (one wake per work_per_hop, depth stages share that rate);
/// the `[4, 30]` per-chain window catches stalls (lower
/// bound: < 4 means the chain didn't complete a ring
/// round-trip) and per-chain runaway throughput from a
/// chain-mixing bug (upper bound: > 30 means chain rate
/// exceeds work_per_hop ceiling). Cross-chain ratio ≤ 2×
/// flags asymmetric progress — both chains receive the same
/// scheduler attention modulo jitter, so a 2× ratio is well
/// outside expected variance and well within the symptom
/// envelope of a chain-mixing or chain-starvation regression.
/// Per-chain assertions (not aggregate) avoid the case where
/// one chain at 40 iters and the other at 0 totals 40 (just
/// above the aggregate threshold's ceiling) but represents a
/// catastrophic isolation failure.
#[test]
fn wake_chain_pipe_multi_chain_bootstrap_independence() {
    const DEPTH: usize = 4;
    const NUM_WORKERS: usize = 8;
    const NUM_CHAINS: usize = NUM_WORKERS / DEPTH;
    const WORK_PER_HOP_MS: u64 = 50;
    const TEST_WINDOW_MS: u64 = 1000;
    const PER_CHAIN_LOWER: u64 = 4;
    const PER_CHAIN_UPPER: u64 = 30;

    if require_isolated_cpus(
        NUM_WORKERS,
        "wake_chain_pipe_multi_chain_bootstrap_independence",
    ) {
        return;
    }

    let config = WorkloadConfig {
        num_workers: NUM_WORKERS,
        work_type: WorkType::WakeChain {
            depth: DEPTH,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_millis(WORK_PER_HOP_MS),
        },
        ..Default::default()
    };
    let mut h =
        WorkloadHandle::spawn(&config).expect("WakeChain wake=Pipe multi-chain spawn must succeed");
    h.start();
    std::thread::sleep(Duration::from_millis(TEST_WINDOW_MS));
    let reports = h.stop_and_collect();
    assert_eq!(
        reports.len(),
        NUM_WORKERS,
        "WakeChain wake=Pipe multi-chain collects one report per worker"
    );

    // Per-chain totals: chain_idx = i / depth matches the
    // spawn-side allocator's `chain_pipes_base + i / depth`.
    let mut per_chain_totals: [u64; NUM_CHAINS] = [0; NUM_CHAINS];
    for (i, r) in reports.iter().enumerate() {
        let chain_idx = i / DEPTH;
        per_chain_totals[chain_idx] += r.iterations;
    }

    for (chain_idx, &chain_total) in per_chain_totals.iter().enumerate() {
        assert!(
            (PER_CHAIN_LOWER..=PER_CHAIN_UPPER).contains(&chain_total),
            "WakeChain wake=Pipe multi-chain: chain {chain_idx} total \
             iterations over {TEST_WINDOW_MS}ms with \
             work_per_hop={WORK_PER_HOP_MS}ms must land in \
             [{PER_CHAIN_LOWER}, {PER_CHAIN_UPPER}] (got \
             {chain_total}). Correct steady-state is ~20 per chain \
             (one wake per work_per_hop across {DEPTH} stages); a \
             chain-mixing regression has one chain stall while the \
             other absorbs both bootstraps, or has the wrong stage \
             fire its bootstrap. Per-chain totals: {:?}. Per-worker \
             reports: {:?}",
            per_chain_totals,
            reports,
        );
    }

    let max_chain = *per_chain_totals.iter().max().unwrap();
    let min_chain = *per_chain_totals.iter().min().unwrap();
    assert!(
        min_chain > 0 && max_chain <= min_chain.saturating_mul(2),
        "WakeChain wake=Pipe multi-chain: cross-chain iteration \
         ratio must stay within 2× (max={max_chain}, min={min_chain}). \
         Both chains receive the same scheduler attention modulo \
         jitter; a > 2× spread indicates one chain is starving the \
         other, which under independent fd ownership cannot happen \
         unless a regression crosses chain indices. Per-chain totals: \
         {:?}. Per-worker reports: {:?}",
        per_chain_totals,
        reports,
    );
}
/// `WorkType::WakeChain` smoke test. depth=2, num_workers=2 →
/// 1 chain of 2 workers. Single linear chain.
#[test]
fn pathology_wake_chain_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::WakeChain {
            depth: 2,
            wake: WakeMechanism::Futex,
            work_per_hop: Duration::from_micros(50),
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("WakeChain must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    let total: u64 = reports.iter().map(|r| r.iterations).sum();
    assert!(total > 0, "WakeChain ring must iterate: {reports:?}");
}
/// `WorkType::WakeChain { wake: WakeMechanism::Pipe }` smoke test. Drives the
/// anon-pipe ring path so the kernel `wake_up_interruptible_sync_poll`
/// → `__wake_up_sync_key` → `WF_SYNC` chain runs end-to-end.
/// Asserts every worker iterates at least once; the rigorous
/// WF_SYNC-fired assertion lives in #294.
#[test]
fn pathology_wake_chain_sync_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::WakeChain {
            depth: 2,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_micros(50),
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("WakeChain wake=Pipe must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "WakeChain wake=Pipe worker must iterate: {r:?}"
        );
    }
}
/// `WakeChain { wake: WakeMechanism::Pipe }` deeper chain.
/// depth=4, num_workers=4 → 1 chain of 4 workers. Verifies the
/// ring closes (stage 3 wakes stage 0) by requiring every
/// stage iterates.
#[test]
fn pathology_wake_chain_sync_deeper_chain() {
    let cfg = WorkloadConfig {
        num_workers: 4,
        work_type: WorkType::WakeChain {
            depth: 4,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_micros(20),
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("WakeChain wake=Pipe depth=4 must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(300));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "WakeChain wake=Pipe depth=4 worker must iterate: {r:?}"
        );
    }
}
/// `WakeChain { wake: WakeMechanism::Pipe }` multi-chain.
/// depth=2, num_workers=4 → 2 stages × 2 parallel chains
/// (chains derived from `num_workers / depth`). Each chain
/// owns its own pipe ring; pipes are not shared across
/// chains. All workers must iterate independently.
#[test]
fn pathology_wake_chain_sync_multi_chain() {
    let cfg = WorkloadConfig {
        num_workers: 4,
        work_type: WorkType::WakeChain {
            depth: 2,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_micros(50),
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("WakeChain wake=Pipe multi-chain must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 4);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "WakeChain wake=Pipe multi-chain worker must iterate: {r:?}"
        );
    }
}
/// `WakeChain` rejects `num_workers % depth != 0` at spawn so
/// the chain-pipe allocator never produces a partial chain.
/// The framework derives `chains = num_workers / depth`, which
/// silently truncates if the ratio is not exact — leaving
/// workers unaccounted for. Pins the spawn-side check that
/// catches the bug at construction with an actionable
/// diagnostic naming the divisor (worker_group_size = depth).
/// Covered cases: depth=2 with num_workers in {1, 3, 5, 7},
/// depth=4 with num_workers in {1, 2, 3, 5, 6, 7}. Each
/// `spawn` must return `Err`; the diagnostic must mention
/// `divisible by` and the offending depth.
#[test]
fn wake_chain_spawn_rejects_non_multiple_num_workers() {
    for &(num_workers, depth) in &[
        (1usize, 2usize),
        (3, 2),
        (5, 2),
        (7, 2),
        (1, 4),
        (2, 4),
        (3, 4),
        (5, 4),
        (6, 4),
        (7, 4),
    ] {
        let cfg = WorkloadConfig {
            num_workers,
            work_type: WorkType::WakeChain {
                depth,
                wake: WakeMechanism::Pipe,
                work_per_hop: Duration::from_micros(50),
            },
            ..Default::default()
        };
        let err = WorkloadHandle::spawn(&cfg).err().unwrap_or_else(|| {
            panic!(
                "WakeChain spawn must reject num_workers={num_workers} \
                 with depth={depth} (not a positive multiple)",
            )
        });
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("divisible by"),
            "WakeChain rejection diagnostic must mention `divisible by`; \
             num_workers={num_workers}, depth={depth}, got: {rendered}",
        );
        assert!(
            rendered.contains(&depth.to_string()),
            "WakeChain rejection diagnostic must name the offending depth \
             ({depth}); num_workers={num_workers}, got: {rendered}",
        );
        let typed = err
            .downcast_ref::<WorkTypeValidationError>()
            .unwrap_or_else(|| {
                panic!(
                    "error must downcast to WorkTypeValidationError; \
                     num_workers={num_workers}, depth={depth}, err: {rendered}"
                )
            });
        match typed {
            WorkTypeValidationError::NonDivisibleWorkerCount {
                name,
                group_idx,
                group_size,
                num_workers: nw,
            } => {
                assert_eq!(
                    name, "WakeChain",
                    "name field must be WakeChain; got: {name}",
                );
                assert_eq!(
                    *group_idx, 0,
                    "primary group has group_idx == 0; got: {group_idx}",
                );
                assert_eq!(
                    *group_size, depth,
                    "group_size must equal depth; got: {group_size}",
                );
                assert_eq!(
                    *nw, num_workers,
                    "num_workers field must echo input; got: {nw}",
                );
            }
            other => panic!(
                "expected NonDivisibleWorkerCount; got: {other:?}; \
                 num_workers={num_workers}, depth={depth}"
            ),
        }
    }
}
/// `WakeChain` with `depth == 1` and
/// `wake: WakeMechanism::Pipe` is rejected at spawn — a
/// 1-stage chain has no successor to wake AND the post-fork
/// fd-close logic would close the worker's own write end
/// (deadlock). The typed-error variant must be
/// [`WorkTypeValidationError::InsufficientWakeChainDepth`]
/// so callers can program against the depth precondition
/// without parsing the diagnostic. The depth-rejection check
/// only fires for `wake: Pipe` because
/// [`chain_pipe_depth`](WorkType::chain_pipe_depth) returns
/// `None` for `wake: Futex`.
#[test]
fn wake_chain_spawn_rejects_depth_one_pipe() {
    let cfg = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::WakeChain {
            depth: 1,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_micros(50),
        },
        ..Default::default()
    };
    let err = WorkloadHandle::spawn(&cfg)
        .err()
        .expect("WakeChain wake=Pipe with depth=1 must be rejected at spawn");
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("depth must be >= 2"),
        "diagnostic must mention `depth must be >= 2`; got: {rendered}",
    );
    let typed = err
        .downcast_ref::<WorkTypeValidationError>()
        .expect("error must downcast to WorkTypeValidationError");
    assert!(
        matches!(
            typed,
            WorkTypeValidationError::InsufficientWakeChainDepth {
                depth: 1,
                group_idx: 0,
            }
        ),
        "expected InsufficientWakeChainDepth {{ depth: 1, group_idx: 0 }}, got: {typed:?}",
    );
}
/// `WakeChain` accepts every positive multiple of depth at
/// spawn — pinned alongside the negative test above so a
/// regression that broadened the rejection (e.g. requiring
/// num_workers == depth) trips here. Spawned configurations
/// are immediately stopped to avoid running real work.
#[test]
fn wake_chain_spawn_accepts_positive_multiples_of_depth() {
    for &(num_workers, depth) in &[(2usize, 2usize), (4, 2), (6, 2), (4, 4), (8, 4), (12, 4)] {
        let cfg = WorkloadConfig {
            num_workers,
            work_type: WorkType::WakeChain {
                depth,
                wake: WakeMechanism::Pipe,
                work_per_hop: Duration::from_micros(20),
            },
            ..Default::default()
        };
        let mut h = WorkloadHandle::spawn(&cfg).unwrap_or_else(|e| {
            panic!(
                "WakeChain spawn must accept num_workers={num_workers} \
                 with depth={depth} (positive multiple); err: {e:#}"
            )
        });
        // Run briefly and stop — this verifies spawn accepted
        // the configuration without actually exercising the
        // long-running chain.
        h.start();
        std::thread::sleep(Duration::from_millis(20));
        let _ = h.stop_and_collect();
    }
}
/// `WakeChain { wake: WakeMechanism::Pipe }` stop responsiveness. Pins the
/// FIX 1 contract: workers blocked in the pipe `read` must
/// re-check `stop_requested` via the `poll(POLLIN, 100ms)`
/// loop and exit cleanly. `stop_and_collect` must complete
/// within 500 ms (well under the SIGUSR1 escalation deadline)
/// and every worker must report `completed == true`.
#[test]
fn pathology_wake_chain_sync_stop_responsive() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::WakeChain {
            depth: 2,
            wake: WakeMechanism::Pipe,
            work_per_hop: Duration::from_micros(50),
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("WakeChain wake=Pipe must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let stop_start = Instant::now();
    let reports = h.stop_and_collect();
    let stop_elapsed = stop_start.elapsed();
    assert!(
        stop_elapsed < Duration::from_millis(500),
        "stop_and_collect took {stop_elapsed:?}, expected < 500ms"
    );
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.completed,
            "WakeChain wake=Pipe worker must complete on stop: {r:?}"
        );
    }
}
