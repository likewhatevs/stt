//! Lightweight state types and constants shared across the freeze
//! coordinator's call sites.
//!
//! Pure types and constants only — no behaviour. Behaviour lives in
//! the modules that consume these (the run-loop closure in
//! [`super`], the snapshot request handlers in [`super::snapshot`]).
//! Splitting the types out here lets the closure body shrink and
//! lets each consumer use the same vocabulary without re-deriving it
//! locally.
//!
//! Three groups live here:
//!
//! * [`FREEZE_RENDEZVOUS_TIMEOUT`] — wall-clock budget for parked-
//!   vCPU rendezvous and the matching post-thaw barrier.
//! * [`BspExitReason`] — diagnostic enum logged when the BSP run
//!   loop breaks.
//! * [`SnapshotRequest`] — typed view of a guest-side
//!   `MSG_TYPE_SNAPSHOT_REQUEST` TLV.
//! * [`FreezeState`] — the dump state machine the run-loop closure
//!   advances on each freeze cycle.
//!
//! All four were previously defined inline at the top of
//! `freeze_coord.rs` (or, for `FreezeState`, inside the run-loop
//! closure body); the public surface is unchanged.

use std::time::Duration;

/// Maximum wall-clock duration the freeze coordinator will wait for
/// every vCPU to acknowledge parked state before logging a timeout
/// and giving up on the dump. Well above the worst-case drain-dance
/// and single-iteration park latency on healthy guests; a real
/// timeout indicates a vCPU stuck in KVM_RUN that the
/// `immediate_exit` kick failed to interrupt.
pub(super) const FREEZE_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// Why [`super::KtstrVm::run_bsp_loop`] exited. Logged at break time
/// so an operator reading stderr (`BSP: loop exit reason=...`) can
/// diagnose a `code=-1` exit without correlating to peer-vCPU
/// stderr or `tracing` output.
///
/// Mapping to the BSP loop's exit_code:
///   - [`Shutdown`](Self::Shutdown) → exit_code = 0 (the only path
///     that overwrites the local `-1` sentinel).
///   - Every other variant → exit_code = -1, but
///     [`super::super::KtstrVm::collect_results`] re-derives the
///     final [`super::super::result::VmResult::exit_code`] from the
///     bulk-port `MSG_TYPE_EXIT` payload (or COM2 `KTSTR_EXIT:`
///     sentinel) when either is present, so a `-1` from the BSP
///     run-loop is not authoritative for caller-visible test
///     outcome.
#[derive(Debug, Clone, Copy)]
pub(super) enum BspExitReason {
    /// `kill.load(Acquire)` returned `true` at the top of the loop —
    /// some peer (an AP that observed [`super::exit_dispatch::ExitAction::Shutdown`] or
    /// [`super::exit_dispatch::ExitAction::Fatal`], the panic hook, the monitor thread on
    /// `MSG_TYPE_SCHED_EXIT`, or `collect_results`) flipped the flag.
    /// In particular, on a clean test exit where the kernel's i8042
    /// reset OUT is dispatched to a non-BSP vCPU, the AP path sets
    /// `kill` and the BSP exits via this branch. The default value
    /// for the local — every break path that does not explicitly
    /// reassign falls into this case.
    ExternalKill,
    /// BSP itself observed [`super::exit_dispatch::ExitAction::Shutdown`] from
    /// `classify_exit` (i8042 reset on x86_64, PSCI SystemEvent /
    /// `VcpuExit::Shutdown` on aarch64). The only path that sets
    /// exit_code to 0.
    Shutdown,
    /// BSP itself observed [`super::exit_dispatch::ExitAction::Fatal`] from `classify_exit`
    /// (`VcpuExit::FailEntry` or `VcpuExit::InternalError`). Kill
    /// flag is propagated to peers before break.
    Fatal,
    /// `bsp.run()` returned a non-EINTR/EAGAIN errno. Indicates a
    /// permanent KVM_RUN failure on the BSP vCPU fd.
    RunError,
}

/// Decoded contents of a guest-side `MSG_TYPE_SNAPSHOT_REQUEST` TLV
/// frame consumed from the virtio-console port-1 TX stream by the
/// coordinator's TOKEN_TX handler. The request id is echoed in the
/// matching `MSG_TYPE_SNAPSHOT_REPLY` payload so the guest's blocking
/// reader can pair the reply against its outstanding request; `kind`
/// selects the CAPTURE / WATCH dispatch path and `tag` carries the
/// snapshot name (CAPTURE) or symbol path (WATCH).
pub(super) struct SnapshotRequest {
    pub(super) request_id: u32,
    pub(super) kind: u32,
    pub(super) tag: String,
}

/// Dual-snapshot state machine the freeze coordinator's run-loop
/// advances on each capture cycle. Only the `TookEarly` variant is
/// reachable when `freeze_coord_dual_snapshot` is true; the single-
/// snapshot path drives the same transitions but skips the early
/// branch entirely.
///
/// * [`Idle`](Self::Idle) — no dump captured yet.
/// * [`TookEarly`](Self::TookEarly) — early snapshot captured
///   (dual-snapshot mode only); waiting for the err_exit latch to
///   fire.
/// * [`Done`](Self::Done) — no further late-trigger captures
///   attempted. Reach paths: (1) late Captured → full single or dual
///   SCHEMA JSON on the main dump path; (2) late Degraded → degraded
///   SCHEMA JSON on the main dump path (and, if dual-mode held a
///   Captured early, the early reaches disk at the
///   `early-pre-late-degraded` tagged sibling per
///   [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED`]);
///   (3) late Suppressed → no main-path emit (clean exit, no late
///   failure to dump), but if dual-mode held a Captured early, that
///   early reaches disk at the `early-only-late-suppressed` tagged
///   sibling per
///   [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED`].
///   Every captured snapshot reaches disk regardless of how the late
///   path resolves. Coord idles until kill / bsp_done. The shared
///   terminal semantic is "stop probing the trigger" — what differs
///   across reach paths is which file(s) the operator finds in the
///   dump directory.
///
///   Reach path (4) does NOT pass through `Done` via a late-trigger
///   arm at all: the late trigger never fires (no `err_exit_detected`
///   BPF latch flip for the run), `freeze_state` stays at `Idle` or
///   `TookEarly`, and the coordinator's normal-exit cleanup at the
///   `'coord:` loop tail drains a still-held `early_snapshot` to
///   `early-only-late-never-fired` (per
///   [`crate::monitor::dump::SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED`]).
///   This case is distinct from path (3) — both share the "no main
///   dump, early at tagged sibling" shape, but signal differs: (3)
///   means the late trigger fired and the gate decided clean; (4)
///   means the late trigger never reached terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FreezeState {
    Idle,
    TookEarly,
    Done,
}

/// Format the bridge tag for periodic boundary `idx`. The wire shape
/// — zero-padded 3-digit `"periodic_NNN"` — is documented on
/// [`crate::test_support::entry::KtstrTestEntry::num_snapshots`] and
/// pinned by the unit tests in this module. Pulled out of the inline
/// `format!` site so the format string lives at one location and a
/// future width change does not silently shift between fire path and
/// downstream string-matching tests.
pub(super) fn periodic_tag(idx: u32) -> String {
    format!("periodic_{:03}", idx)
}

/// Compute the absolute boundary timestamps (nanoseconds since
/// `run_start`) at which the freeze coordinator's periodic-capture
/// loop fires `freeze_and_capture(false)`. The window is the
/// 10 %–90 % slice of the workload duration: a 10 % pre-buffer at
/// the start (workload ramp-up) and a 10 % post-buffer at the end
/// (workload ramp-down) keep periodic samples off transient state.
/// The remaining 80 % is divided into `num_snapshots + 1` equal
/// intervals, yielding `num_snapshots` interior boundary points.
///
/// Anchor offset: every boundary is biased by `scenario_anchor_ns`
/// so the returned timestamps are absolute (comparable against
/// `run_start.elapsed().as_nanos()` in the run-loop). Compute in
/// `u128` to avoid intermediate overflow on long durations: a
/// 10-minute workload is ~6e11 ns; with `i = N − 1 = 63`, the
/// product `window · (i + 1)` would saturate `u64`. The final
/// boundary fits in `u64` (saturating cast) for callers that read
/// against the run-loop's `u64` clock.
///
/// # Edge cases
///
/// * `num_snapshots == 0` → empty `Vec` (the disabled case; the
///   run-loop's check skips the entire branch when the boundary
///   list is empty / unset).
/// * `num_snapshots == 1` → single boundary at the workload
///   midpoint (`anchor + 0.5 · d`).
/// * `num_snapshots == N` for `N ≥ 2` → `N` boundaries at
///   `anchor + 0.1 · d + (i + 1) · 0.8 · d / (N + 1)` for
///   `i ∈ 0..N`.
///
/// Boundaries are strictly monotonically increasing — a property
/// the run-loop relies on (linear scan via `next_periodic_idx`).
pub(super) fn compute_periodic_boundaries_ns(
    scenario_anchor_ns: u64,
    workload_duration: Duration,
    num_snapshots: u32,
) -> Vec<u64> {
    if num_snapshots == 0 {
        return Vec::new();
    }
    let n = num_snapshots as u128;
    let total_ns = workload_duration.as_nanos();
    let pre_buffer = total_ns / 10;
    let window = total_ns.saturating_sub(2u128.saturating_mul(pre_buffer));
    let mut boundaries: Vec<u64> = Vec::with_capacity(num_snapshots as usize);
    for i in 0..n {
        let offset = pre_buffer.saturating_add(window.saturating_mul(i + 1) / (n + 1));
        let absolute = (scenario_anchor_ns as u128).saturating_add(offset);
        boundaries.push(u64::try_from(absolute).unwrap_or(u64::MAX));
    }
    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dispatched into the periodic-capture loop's tag-formatting
    /// step. The tag wire format is documented in
    /// [`crate::test_support::entry::KtstrTestEntry::num_snapshots`]:
    /// each interior boundary is stored on the bridge under
    /// `"periodic_NNN"` (zero-padded 3-digit index). Tests assert
    /// the exact string shape so a regression to a different padding
    /// width or separator surfaces here, not in downstream test
    /// authors' string-matching code.
    #[test]
    fn periodic_tag_name_format_low_index() {
        // Index 0 — leftmost slot.
        assert_eq!(periodic_tag(0), "periodic_000");
        // Single-digit, single-digit — pad to 3 chars.
        assert_eq!(periodic_tag(1), "periodic_001");
        assert_eq!(periodic_tag(7), "periodic_007");
    }

    /// Two- and three-digit indices must keep the zero-padding
    /// width at exactly 3 characters. A naive `format!("periodic_{i}")`
    /// without `:03` would surface here as `"periodic_10"` /
    /// `"periodic_64"` — failing the comparison.
    #[test]
    fn periodic_tag_name_format_high_index() {
        assert_eq!(periodic_tag(10), "periodic_010");
        assert_eq!(periodic_tag(63), "periodic_063");
        // MAX_STORED_SNAPSHOTS == 64; the cap rejects N > 64 but
        // index 64 itself is unreachable in practice (boundaries
        // are 0 .. N-1). Pin the format for index 64 anyway so a
        // future cap raise that produced index 64 doesn't surface
        // as a tag-format regression.
        assert_eq!(periodic_tag(64), "periodic_064");
    }

    /// `compute_periodic_boundaries_ns` divides the workload
    /// duration into `num_snapshots + 1` equal intervals after
    /// reserving a 10 % pre-buffer and 10 % post-buffer. For
    /// `N == 1` the lone interior boundary lands at the workload
    /// midpoint — `scenario_anchor + 0.5 · d`. Pinning the midpoint
    /// case proves the formula's symmetry: a shift of the
    /// pre/post buffer balance would surface here as a non-mid
    /// landing point.
    #[test]
    fn compute_periodic_boundaries_ns_n1_lands_at_midpoint() {
        // 10 s workload anchored at 0 → midpoint = 5 s = 5 · 1e9 ns.
        let d = std::time::Duration::from_secs(10);
        let boundaries = compute_periodic_boundaries_ns(0, d, 1);
        assert_eq!(
            boundaries.len(),
            1,
            "N=1 must produce exactly one interior boundary",
        );
        assert_eq!(
            boundaries[0], 5_000_000_000,
            "N=1 lands at start + 0.5·d (midpoint of 10 s = 5 s)",
        );
    }

    /// `N == 3` lands captures at 0.3·d, 0.5·d, 0.7·d when the
    /// 10 % pre-buffer + 80 % usable span + 10 % post-buffer
    /// formula is honoured. The doc on
    /// [`crate::test_support::entry::KtstrTestEntry::num_snapshots`]
    /// pins these exact landing points; regressing the formula
    /// (e.g. dropping the buffer, splitting at fence-post boundaries
    /// instead of equal-interval interior points) would surface
    /// here.
    #[test]
    fn compute_periodic_boundaries_ns_n3_quartile_landings() {
        // 10 s workload anchored at 0:
        //   pre-buffer = 1 s, window = 8 s, splits = 4 intervals
        //   boundary[0] = 1 s + 1/4 · 8 s = 3 s
        //   boundary[1] = 1 s + 2/4 · 8 s = 5 s
        //   boundary[2] = 1 s + 3/4 · 8 s = 7 s
        let d = std::time::Duration::from_secs(10);
        let boundaries = compute_periodic_boundaries_ns(0, d, 3);
        assert_eq!(boundaries.len(), 3, "N=3 must produce 3 boundaries");
        assert_eq!(boundaries[0], 3_000_000_000, "N=3 boundary[0] at 0.3·d");
        assert_eq!(boundaries[1], 5_000_000_000, "N=3 boundary[1] at 0.5·d");
        assert_eq!(boundaries[2], 7_000_000_000, "N=3 boundary[2] at 0.7·d");
    }

    /// Boundaries must be biased by the scenario anchor — a non-zero
    /// anchor must shift every boundary by the same offset. Pins
    /// the absolute-vs-relative contract: the formula is
    /// `anchor + pre_buffer + (i+1) · window / (N+1)`, so a
    /// regression that dropped the anchor add (returning relative
    /// offsets) would surface here as `boundaries[0] != anchor + 5e9`.
    #[test]
    fn compute_periodic_boundaries_ns_respects_anchor_offset() {
        // Anchor at 2 s, 10 s duration, N=1 → boundary at
        // anchor + 5 s = 7 s.
        let d = std::time::Duration::from_secs(10);
        let anchor_ns: u64 = 2_000_000_000;
        let boundaries = compute_periodic_boundaries_ns(anchor_ns, d, 1);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(
            boundaries[0], 7_000_000_000,
            "anchor(2 s) + midpoint(5 s) = 7 s",
        );
    }

    /// `N == 0` is the disabled-periodic-capture case. The function
    /// must return an empty `Vec` — NOT a panic, NOT a single
    /// boundary at the midpoint. The freeze coordinator's
    /// per-iteration check uses the boundary list as the gate; an
    /// empty vec means no periodic samples fire.
    #[test]
    fn compute_periodic_boundaries_ns_n0_yields_empty() {
        let d = std::time::Duration::from_secs(10);
        let boundaries = compute_periodic_boundaries_ns(0, d, 0);
        assert!(
            boundaries.is_empty(),
            "N=0 must yield no boundaries, got {boundaries:?}",
        );
    }

    /// Boundaries must be strictly monotonically increasing.
    /// Dispatch loop relies on this invariant: the run-loop
    /// scans boundaries linearly via `next_periodic_idx` and
    /// expects `boundaries[i] < boundaries[i+1]`. A regression
    /// that produced duplicate or descending boundaries would
    /// silently fire the same idx twice or stall.
    #[test]
    fn compute_periodic_boundaries_ns_strictly_monotonic() {
        let d = std::time::Duration::from_secs(60);
        for &n in &[1u32, 2, 3, 7, 16, 32, 64] {
            let boundaries = compute_periodic_boundaries_ns(0, d, n);
            assert_eq!(
                boundaries.len(),
                n as usize,
                "N={n} must produce exactly {n} boundaries",
            );
            for w in boundaries.windows(2) {
                assert!(
                    w[0] < w[1],
                    "boundaries must be strictly monotonic; got {w:?} for N={n}",
                );
            }
        }
    }

    /// Every boundary must land inside the 10 %–90 % window of
    /// `[anchor, anchor + duration]`. The pre-buffer and
    /// post-buffer give the workload ramp-up / ramp-down room
    /// without periodic samples landing on transient state. Pins
    /// the buffer invariant against a regression that removed or
    /// shifted the buffers (which would surface here as a
    /// boundary at < 0.1·d or > 0.9·d).
    #[test]
    fn compute_periodic_boundaries_ns_within_buffer_window() {
        let d = std::time::Duration::from_secs(10);
        let total_ns = d.as_nanos() as u64;
        // 10 % pre = 1 s; 90 % post = 9 s. Use inclusive bounds —
        // boundaries land STRICTLY inside the window, never AT
        // the buffer edge (the formula is `+ (i+1) · window /
        // (N+1)` for i ∈ 0..N, so the lowest landing is at
        // `pre + 1·window/(N+1) > pre`).
        for &n in &[1u32, 4, 8] {
            let boundaries = compute_periodic_boundaries_ns(0, d, n);
            for (i, &b) in boundaries.iter().enumerate() {
                assert!(
                    b > total_ns / 10,
                    "boundary {i} ({b}) must be strictly above 10% pre-buffer ({})",
                    total_ns / 10,
                );
                assert!(
                    b < total_ns - total_ns / 10,
                    "boundary {i} ({b}) must be strictly below 90% post-buffer ({})",
                    total_ns - total_ns / 10,
                );
            }
        }
    }

    /// Pin the rounding behaviour for non-round-number durations.
    /// `compute_periodic_boundaries_ns` performs all arithmetic in
    /// `u128` and rounds toward zero on the integer division
    /// `window * (i + 1) / (N + 1)`. A future arithmetic refactor
    /// (rounding to nearest, switching to f64, swapping operand
    /// order) would silently shift periodic-sample landings.
    /// These exact-byte assertions surface that drift here, before
    /// it ships.
    ///
    /// The expected values are computed by hand from the formula
    ///   boundary[i] = anchor + (total / 10)
    ///                 + (total - 2 · (total / 10)) · (i + 1) / (N + 1)
    /// with all operations in u128 truncating-divide. Each case is
    /// derived in the test body comment so a future reader can
    /// re-verify without re-running the math.
    #[test]
    fn compute_periodic_boundaries_ns_odd_ns_rounding() {
        // Case 1: 1_000_000_001 ns, N = 3.
        //   pre = 1_000_000_001 / 10 = 100_000_000
        //   window = 1_000_000_001 − 200_000_000 = 800_000_001
        //   i=0: pre + 800_000_001 · 1 / 4 = 100_000_000 + 200_000_000 = 300_000_000
        //   i=1: pre + 800_000_001 · 2 / 4 = 100_000_000 + 400_000_000 = 500_000_000
        //   i=2: pre + 800_000_001 · 3 / 4 = 100_000_000 + 600_000_000 = 700_000_000
        //
        // (`800_000_001 · k / 4` truncates the `+ 1` for k = 1..3
        // because 4 · 200_000_000 = 800_000_000 < 800_000_001.)
        let d = std::time::Duration::from_nanos(1_000_000_001);
        let boundaries = compute_periodic_boundaries_ns(0, d, 3);
        assert_eq!(
            boundaries,
            vec![300_000_000, 500_000_000, 700_000_000],
            "1_000_000_001 ns, N=3 must round each boundary down to \
             the multiple-of-100_000_000 nearest the truncating-divide \
             result"
        );

        // Case 2: 1_000_000_007 ns, N = 2.
        //   pre = 100_000_000  (since 1_000_000_007 / 10 = 100_000_000)
        //   window = 1_000_000_007 − 200_000_000 = 800_000_007
        //   i=0: 100_000_000 + 800_000_007 · 1 / 3 = 100_000_000 + 266_666_669 = 366_666_669
        //   i=1: 100_000_000 + 800_000_007 · 2 / 3 = 100_000_000 + 533_333_338 = 633_333_338
        //
        // 3 · 266_666_669 = 800_000_007 exactly, so i=0 lands on the
        // exact divide; 3 · 533_333_338 = 1_600_000_014 exactly so
        // i=1 also exact. Pin the values so a future change
        // (e.g. operand-order swap that produces the same exact
        // result vs a `f64` round-to-nearest path that would shift
        // by 1 ns) surfaces here.
        let d2 = std::time::Duration::from_nanos(1_000_000_007);
        let boundaries2 = compute_periodic_boundaries_ns(0, d2, 2);
        assert_eq!(boundaries2, vec![366_666_669, 633_333_338]);

        // Case 3: 1_000_000_007 ns, N = 4 — exercises the same
        // odd-ns total against a denser boundary grid.
        //   window = 800_000_007, n+1 = 5
        //   i=0: 100_000_000 + 800_000_007 / 5 = 100_000_000 + 160_000_001 = 260_000_001
        //         (5 · 160_000_001 = 800_000_005, remainder 2)
        //   i=1: 100_000_000 + 1_600_000_014 / 5 = 100_000_000 + 320_000_002 = 420_000_002
        //         (5 · 320_000_002 = 1_600_000_010, remainder 4)
        //   i=2: 100_000_000 + 2_400_000_021 / 5 = 100_000_000 + 480_000_004 = 580_000_004
        //         (5 · 480_000_004 = 2_400_000_020, remainder 1)
        //   i=3: 100_000_000 + 3_200_000_028 / 5 = 100_000_000 + 640_000_005 = 740_000_005
        //         (5 · 640_000_005 = 3_200_000_025, remainder 3)
        let boundaries3 = compute_periodic_boundaries_ns(0, d2, 4);
        assert_eq!(
            boundaries3,
            vec![260_000_001, 420_000_002, 580_000_004, 740_000_005]
        );

        // Case 4: anchor offset survives the odd-ns rounding —
        // anchor + computed offset, no extra rounding step.
        //   anchor = 12_345; everything else is Case 1's setup
        //   each boundary should be Case 1's boundary + 12_345.
        let boundaries4 = compute_periodic_boundaries_ns(12_345, d, 3);
        assert_eq!(
            boundaries4,
            vec![300_012_345, 500_012_345, 700_012_345],
            "anchor offset must be added to the truncated boundary, \
             not folded into the truncating-divide"
        );
    }
}
