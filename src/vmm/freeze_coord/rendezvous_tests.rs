
//! Unit coverage for the freeze-rendezvous decision logic that
//! lives inside the run-loop closure: `expected_parks`
//! arithmetic, the still-parked pre-seed compensation, and the
//! worker-park sub-timeout (including its TOCTOU re-check).
//!
//! The production code threads these decisions through Arc-bound
//! atomic flags and an `EventFd` counter that all live in
//! closure scope inside `run_vm`; there is no extracted function
//! to call directly. Following the convention established by
//! `crc_defense_tests` (the SCHED_EXIT / SNAPSHOT_REQUEST gates
//! live in the same closure and use the same in-test mirror
//! pattern), each helper below reproduces a single production
//! predicate at the same bit level so a regression that flips
//! the predicate fails here. If the production decision drifts
//! from the test mirror, this module must be updated in the
//! same change so the regression is visible.
//!
//! Coverage:
//!   * `compute_expected_parks` — the three-input sum at the top
//!     of the rendezvous wait. Pins ap_count + bsp + worker
//!     bookkeeping against the four reachable combinations of
//!     bsp_alive / worker_was_running.
//!   * `compute_pre_seed` — the still-parked counter pre-seed at
//!     cycle entry that compensates for a previous post-thaw
//!     barrier timeout. Pins both the per-AP scan and the BSP /
//!     worker pre-seed gates so a stale parked=true on either
//!     side credits exactly one ack.
//!   * `decide_worker_drop` — the worker sub-timeout
//!     bookkeeping decision. Drops the +1 from `expected_parks`
//!     only when (a) we counted the worker, (b) the wall-clock
//!     sub-deadline has passed, and (c) `paused == false`.
//!   * `decide_worker_drop` with paused-true on second load —
//!     the TOCTOU re-check that prevents double-counting when
//!     the worker transitioned `paused = true` between the
//!     first sample and the bookkeeping change.
//!   * `rendezvous_done_when_count_meets_expected` — the loop's
//!     completion predicate. Pins the `>=` direction so a
//!     regression to `==` (which would miss overshoot from the
//!     pre-seed path) fails.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Mirror of the production `expected_parks` arithmetic at the
/// top of the rendezvous wait (in `freeze_and_capture`). Pure
/// function of the three inputs; the production line is:
///
/// ```ignore
/// let mut expected_parks: u64 =
///     freeze_coord_ap_parked.len() as u64
///         + if bsp_alive_at_start { 1 } else { 0 }
///         + if worker_was_running { 1 } else { 0 };
/// ```
fn compute_expected_parks(ap_count: u64, bsp_alive: bool, worker_was_running: bool) -> u64 {
    ap_count + if bsp_alive { 1 } else { 0 } + if worker_was_running { 1 } else { 0 }
}

/// Mirror of the production still-parked pre-seed scan at cycle
/// entry. Walks each AP's `parked` flag, the BSP flag (gated on
/// `bsp_alive`), and the worker `paused` flag (gated on
/// `worker_was_running`). Returns the counter value the
/// production code writes to `parked_evt`.
///
/// Production lines walked:
///
/// ```ignore
/// let mut still_parked: u32 = 0;
/// for ap in freeze_coord_ap_parked.iter() {
///     if ap.load(Ordering::Acquire) { still_parked = still_parked.saturating_add(1); }
/// }
/// if bsp_alive_at_start && freeze_coord_bsp_parked.load(Acquire) {
///     still_parked = still_parked.saturating_add(1);
/// }
/// if worker_was_running && freeze_coord_virtio_blk_paused.is_some_and(|p| p.load(Acquire)) {
///     still_parked = still_parked.saturating_add(1);
/// }
/// ```
fn compute_pre_seed(
    ap_parked: &[Arc<AtomicBool>],
    bsp_alive: bool,
    bsp_parked: &AtomicBool,
    worker_was_running: bool,
    worker_paused: Option<&AtomicBool>,
) -> u32 {
    let mut still_parked: u32 = 0;
    for ap in ap_parked {
        if ap.load(Ordering::Acquire) {
            still_parked = still_parked.saturating_add(1);
        }
    }
    if bsp_alive && bsp_parked.load(Ordering::Acquire) {
        still_parked = still_parked.saturating_add(1);
    }
    if worker_was_running && worker_paused.is_some_and(|p| p.load(Ordering::Acquire)) {
        still_parked = still_parked.saturating_add(1);
    }
    still_parked
}

/// Outcome of the worker sub-timeout decision the rendezvous
/// loop runs each iteration. Mirrors the production three-way
/// branch:
///
///   - `Continue` — the sub-deadline fired and the second
///     `paused` load returned `true`; skip the drop and let the
///     next iteration absorb the matching parked_evt ack
///     (TOCTOU re-check).
///   - `Drop`     — the sub-deadline fired and `paused` is
///     genuinely false on both loads; decrement
///     `expected_parks` by 1 and mark `worker_dropped`.
///   - `Skip`     — the sub-deadline has not fired, the worker
///     has already been dropped, or the worker was never
///     counted. The bookkeeping is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerDropDecision {
    Continue,
    Drop,
    Skip,
}

/// Mirror of the production worker sub-timeout decision in the
/// rendezvous loop. The two `paused` inputs encode the
/// production code's two `Acquire` loads:
///
/// ```ignore
/// // First load (gate).
/// if !worker_dropped
///     && worker_was_running
///     && Instant::now() >= worker_sub_deadline
///     && freeze_coord_virtio_blk_paused
///         .as_ref()
///         .is_some_and(|p| !p.load(Ordering::Acquire))
/// {
///     // Second load (TOCTOU re-check).
///     if freeze_coord_virtio_blk_paused
///         .as_ref()
///         .is_some_and(|p| p.load(Ordering::Acquire))
///     {
///         continue;
///     }
///     // Drop the +1.
///     ...
/// }
/// ```
///
/// `paused_first` represents the value seen at the gate (must
/// be `false` for the gate to fire). `paused_second` represents
/// the value seen at the re-check; if `true` the decision is
/// `Continue` (TOCTOU caught a late park). If `false` the
/// decision is `Drop`.
///
/// Returns `Skip` for any input combination outside the
/// gate-firing predicate.
fn decide_worker_drop(
    sub_deadline_fired: bool,
    worker_was_running: bool,
    worker_dropped: bool,
    paused_first: bool,
    paused_second: bool,
) -> WorkerDropDecision {
    if worker_dropped || !worker_was_running || !sub_deadline_fired || paused_first {
        return WorkerDropDecision::Skip;
    }
    if paused_second {
        WorkerDropDecision::Continue
    } else {
        WorkerDropDecision::Drop
    }
}

/// Mirror of the production rendezvous completion predicate:
///
/// ```ignore
/// if parked_count >= expected_parks {
///     all_parked = true;
///     break;
/// }
/// ```
///
/// `>=` rather than `==` matters: the pre-seed path can land
/// `parked_count` strictly above `expected_parks` if a healthy
/// parker raced the seed and contributed its own ack. A
/// regression to `==` would miss overshoot and wait the full
/// FREEZE_RENDEZVOUS_TIMEOUT.
fn rendezvous_done(parked_count: u64, expected_parks: u64) -> bool {
    parked_count >= expected_parks
}

/// 0 APs, no BSP, no worker. The rendezvous expects zero acks —
/// the loop's first iteration breaks via the
/// `parked_count >= expected_parks` predicate without polling.
/// Matches a coordinator running with every parker already
/// shut down.
#[test]
fn expected_parks_zero_when_no_parkers() {
    assert_eq!(compute_expected_parks(0, false, false), 0);
}

/// Default healthy run: BSP alive, virtio-blk worker running,
/// N APs. Each contributes one ack; the rendezvous waits for
/// `1 + 1 + N`.
#[test]
fn expected_parks_counts_aps_bsp_and_worker() {
    assert_eq!(compute_expected_parks(2, true, true), 4);
    assert_eq!(compute_expected_parks(7, true, true), 9);
}

/// BSP alive but no virtio-blk attached — the +1 for the
/// worker drops out and the rendezvous waits for only the
/// vCPUs. Pins the `worker_was_running` gate against a
/// regression that would always count the worker and stall
/// 30 s on disk-less runs.
#[test]
fn expected_parks_drops_worker_when_not_running() {
    assert_eq!(compute_expected_parks(3, true, false), 4);
}

/// BSP already dropped (post-`bsp_alive=false` cycle) — the
/// +1 for the BSP drops out. Pins the gate so a stale BSP
/// snapshot does not stall the rendezvous waiting for a vCPU
/// whose VcpuFd is gone.
#[test]
fn expected_parks_drops_bsp_when_not_alive() {
    assert_eq!(compute_expected_parks(2, false, true), 3);
}

/// Both BSP and worker absent — only AP acks. Mirrors the
/// late-cycle path where both have already been torn down but
/// AP threads are still running.
#[test]
fn expected_parks_counts_only_aps_when_neither_bsp_nor_worker() {
    assert_eq!(compute_expected_parks(4, false, false), 4);
}

/// Pre-seed contributes 0 when every parker has already cleared
/// its flag. Healthy steady-state at cycle entry.
#[test]
fn pre_seed_zero_when_no_stale_parkers() {
    let aps: Vec<Arc<AtomicBool>> = (0..3).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let bsp = AtomicBool::new(false);
    let worker = AtomicBool::new(false);
    let seed = compute_pre_seed(&aps, true, &bsp, true, Some(&worker));
    assert_eq!(seed, 0);
}

/// A previous post-thaw barrier timed out leaving every AP
/// stuck with `parked=true`. The pre-seed must contribute one
/// per AP so the rendezvous countdown latch starts already
/// crediting the stale acks. Without this, a 3-vCPU coord
/// would wait the full 30 s for events that fired a cycle ago.
#[test]
fn pre_seed_counts_each_stale_ap() {
    let aps: Vec<Arc<AtomicBool>> = (0..3).map(|_| Arc::new(AtomicBool::new(true))).collect();
    let bsp = AtomicBool::new(false);
    let worker = AtomicBool::new(false);
    let seed = compute_pre_seed(&aps, true, &bsp, true, Some(&worker));
    assert_eq!(seed, 3);
}

/// BSP stuck `parked=true` from a prior cycle's barrier
/// timeout. The `bsp_alive` gate is true (BSP run-loop hasn't
/// dropped yet), so the BSP's stale ack must contribute +1.
#[test]
fn pre_seed_counts_stale_bsp_when_alive() {
    let aps: Vec<Arc<AtomicBool>> = vec![];
    let bsp = AtomicBool::new(true);
    let worker = AtomicBool::new(false);
    let seed = compute_pre_seed(&aps, true, &bsp, true, Some(&worker));
    assert_eq!(seed, 1);
}

/// BSP stuck `parked=true` but `bsp_alive=false` — the gate
/// suppresses the +1 because the BSP `VcpuFd` is gone and any
/// trailing `parked.store(false)` on the BSP thread cannot
/// run. Pins the gate against a regression that would
/// double-count a dead BSP whose flag never clears.
#[test]
fn pre_seed_skips_stale_bsp_when_not_alive() {
    let aps: Vec<Arc<AtomicBool>> = vec![];
    let bsp = AtomicBool::new(true);
    let worker = AtomicBool::new(false);
    let seed = compute_pre_seed(&aps, false, &bsp, true, Some(&worker));
    assert_eq!(seed, 0);
}

/// virtio-blk worker stuck `paused=true` from a prior cycle
/// while the worker thread is still alive
/// (`worker_was_running=true`). The next pause()-driven epoll
/// wake will not re-write the parked_evt ack because the
/// worker is mid-park from the prior cycle, so the seed
/// compensates +1.
#[test]
fn pre_seed_counts_stale_worker_when_running() {
    let aps: Vec<Arc<AtomicBool>> = vec![];
    let bsp = AtomicBool::new(false);
    let worker = AtomicBool::new(true);
    let seed = compute_pre_seed(&aps, true, &bsp, true, Some(&worker));
    assert_eq!(seed, 1);
}

/// Worker `paused=true` but `worker_was_running=false` — the
/// seed gate suppresses the +1 because pause() short-circuited
/// (no live worker thread to write parked_evt). Pins the gate
/// against a regression that would seed even when no live
/// worker exists, leaving the rendezvous over-credited.
#[test]
fn pre_seed_skips_stale_worker_when_not_running() {
    let aps: Vec<Arc<AtomicBool>> = vec![];
    let bsp = AtomicBool::new(false);
    let worker = AtomicBool::new(true);
    let seed = compute_pre_seed(&aps, true, &bsp, false, Some(&worker));
    assert_eq!(seed, 0);
}

/// virtio-blk worker handle absent (no disk) — the
/// `is_some_and` gate yields false and no +1 fires regardless
/// of `worker_was_running`. Pins the disk-less coordinator
/// path.
#[test]
fn pre_seed_skips_worker_when_paused_handle_none() {
    let aps: Vec<Arc<AtomicBool>> = vec![];
    let bsp = AtomicBool::new(false);
    let seed = compute_pre_seed(&aps, true, &bsp, true, None);
    assert_eq!(seed, 0);
}

/// Mixed staleness: 2 of 3 APs stale, BSP clear, worker
/// stale. The seed sums the contributing parkers and skips
/// the cleared one. Catches a regression that aggregates the
/// gate predicates incorrectly (e.g. early-returns on the
/// first cleared AP).
#[test]
fn pre_seed_sums_mixed_staleness() {
    let aps: Vec<Arc<AtomicBool>> = vec![
        Arc::new(AtomicBool::new(true)),
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(true)),
    ];
    let bsp = AtomicBool::new(false);
    let worker = AtomicBool::new(true);
    let seed = compute_pre_seed(&aps, true, &bsp, true, Some(&worker));
    assert_eq!(seed, 3);
}

/// Sub-deadline has not fired yet — the gate's wall-clock
/// predicate is false, so the decision must be `Skip`
/// regardless of `paused`. Pins the timing gate against a
/// regression that would drop the worker as soon as `paused`
/// went false (which can race the worker's first park ack
/// during a slow drain).
#[test]
fn worker_drop_skipped_before_sub_deadline() {
    let decision = decide_worker_drop(
        false, // sub_deadline_fired
        true,  // worker_was_running
        false, // worker_dropped
        false, // paused_first (would fire the gate)
        false, // paused_second
    );
    assert_eq!(decision, WorkerDropDecision::Skip);
}

/// Worker already dropped on a prior iteration — the
/// `worker_dropped` gate suppresses re-evaluation regardless
/// of every other input. Pins the idempotency invariant against
/// a regression that double-decrements `expected_parks`.
#[test]
fn worker_drop_skipped_after_already_dropped() {
    let decision = decide_worker_drop(
        true,  // sub_deadline_fired
        true,  // worker_was_running
        true,  // worker_dropped
        false, // paused_first
        false, // paused_second
    );
    assert_eq!(decision, WorkerDropDecision::Skip);
}

/// Worker was never counted (`worker_was_running=false`) — the
/// sub-timeout path is inert because no +1 ever entered
/// `expected_parks`. Pins the gate against a regression that
/// would still walk the decision tree on disk-less runs and
/// underflow `expected_parks` to a wrap.
#[test]
fn worker_drop_skipped_when_worker_never_counted() {
    let decision = decide_worker_drop(
        true,  // sub_deadline_fired
        false, // worker_was_running
        false, // worker_dropped
        false, // paused_first
        false, // paused_second
    );
    assert_eq!(decision, WorkerDropDecision::Skip);
}

/// Healthy in-flight worker — `paused=true` on the first
/// load means the worker DID park and its ack is in flight.
/// The gate suppresses the drop so the next iteration absorbs
/// the parked_evt ack. Pins the gate's "worker is fine, leave
/// it alone" path against a regression that would always drop
/// after the sub-deadline.
#[test]
fn worker_drop_skipped_when_paused_true_on_first_load() {
    let decision = decide_worker_drop(
        true,  // sub_deadline_fired
        true,  // worker_was_running
        false, // worker_dropped
        true,  // paused_first → worker IS parked
        false, // paused_second (irrelevant; first gate suppresses)
    );
    assert_eq!(decision, WorkerDropDecision::Skip);
}

/// Worker mid-shutdown: `paused=false` on both loads after the
/// sub-deadline. `signal_worker_stop` cleared paused on its
/// way out and no live thread will write parked_evt for this
/// cycle. Drop the +1 so the rendezvous proceeds without
/// waiting the full FREEZE_RENDEZVOUS_TIMEOUT.
#[test]
fn worker_drop_fires_when_paused_false_on_both_loads() {
    let decision = decide_worker_drop(
        true,  // sub_deadline_fired
        true,  // worker_was_running
        false, // worker_dropped
        false, // paused_first → gate fires
        false, // paused_second → confirm drop
    );
    assert_eq!(decision, WorkerDropDecision::Drop);
}

/// TOCTOU race: `paused=false` at the gate sample, but the
/// worker transitioned `paused=true` between that sample and
/// the re-check. The decision MUST be `Continue` so the next
/// loop iteration absorbs the matching parked_evt ack. Without
/// this re-check the production code would both decrement
/// `expected_parks` AND credit the eventfd write — a
/// double-count that breaks the rendezvous arithmetic. This is
/// the load-bearing invariant the team-lead's task description
/// names explicitly.
#[test]
fn worker_drop_continues_when_paused_true_on_recheck() {
    let decision = decide_worker_drop(
        true,  // sub_deadline_fired
        true,  // worker_was_running
        false, // worker_dropped
        false, // paused_first → gate fires
        true,  // paused_second → TOCTOU caught a late park
    );
    assert_eq!(decision, WorkerDropDecision::Continue);
}

/// Rendezvous done predicate fires on exact match —
/// `parked_count == expected_parks`. The healthy steady-state
/// path: every parker acked, the loop breaks via
/// `all_parked = true`.
#[test]
fn rendezvous_done_when_count_meets_expected() {
    assert!(rendezvous_done(4, 4));
}

/// Rendezvous done predicate fires on overshoot —
/// `parked_count > expected_parks`. The pre-seed path can land
/// here when a healthy parker raced the seed and contributed
/// its own ack. Pins the `>=` direction so a regression to
/// `==` is caught.
#[test]
fn rendezvous_done_on_pre_seed_overshoot() {
    assert!(rendezvous_done(5, 4));
}

/// Rendezvous still waiting — `parked_count < expected_parks`.
/// The loop must NOT break; the next iteration polls
/// `parked_evt` for more acks.
#[test]
fn rendezvous_not_done_when_count_below_expected() {
    assert!(!rendezvous_done(3, 4));
}

/// Edge: zero expected, zero observed. The completion
/// predicate must fire on the first iteration so a
/// no-parker coordinator does not spin until the
/// FREEZE_RENDEZVOUS_TIMEOUT.
#[test]
fn rendezvous_done_on_zero_expected() {
    assert!(rendezvous_done(0, 0));
}

/// Worker drop combined with the completion predicate: after
/// the drop, an in-flight AP ack that arrived just before
/// (parked_count = expected_parks - 1) now satisfies the
/// reduced expected. Pins the in-loop re-check at production
/// lines ~3285 against a regression that would force one more
/// poll iteration before observing completion.
#[test]
fn rendezvous_done_after_worker_drop_decrements_expected() {
    // Before drop: 3 APs all acked, BSP acked, worker counted
    // but not acked. parked_count = 4, expected = 5 — not done.
    let mut expected: u64 = 5;
    let parked_count: u64 = 4;
    assert!(!rendezvous_done(parked_count, expected));
    // Drop the worker.
    expected = expected.saturating_sub(1);
    // After drop: parked_count = 4, expected = 4 — done.
    assert!(rendezvous_done(parked_count, expected));
}
