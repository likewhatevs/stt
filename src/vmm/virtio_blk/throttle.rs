//! Token-bucket throttle primitives for the virtio-blk device.
//!
//! Pure arithmetic + monotonic-clock timing — no MMIO, no epoll, no
//! guest memory access. Extracted from the device/worker for module
//! locality so the throttle logic (which has the most adversarial
//! testing surface in the device) lives next to its tests rather
//! than scattered through 16 KLoC of MMIO/FSM/IO code.
//!
//! # Public surface
//!
//! `TokenBucket` is the single throttle primitive; `VirtioBlk` and
//! its worker each hold one for IOPS and one for bandwidth, both
//! materialised from a [`DiskThrottle`] config via
//! `buckets_from_throttle`. The full type-level rationale —
//! overconsumption policy, non-blocking invariants, why throttle
//! exhaustion is hidden from the guest — lives on `TokenBucket`'s
//! own doc comment below.

use std::num::NonZeroU64;
use std::time::Instant;

use crate::vmm::disk_config::DiskThrottle;

// ----------------------------------------------------------------------------
// Token bucket throttle
// ----------------------------------------------------------------------------

/// Single token-bucket. `capacity` tokens accumulate at `refill_rate`
/// per second. `consume(n)` succeeds when `>= n` tokens are available
/// AND drains them; otherwise leaves the bucket untouched and returns
/// false, signalling the caller to backoff.
///
/// "Leak rate" is implicit: every `consume` call first refills based
/// on elapsed wall time since the last refill, capped at `capacity`.
/// No periodic timer needed — the refill is on-demand per request.
///
/// # Overconsumption (request size > capacity)
///
/// `available` is `i64` (not `u64`) so a single oversized request —
/// `n > capacity` — can be granted by driving `available` negative
/// instead of stalling forever. Without this allowance, a guest that
/// submits a chain whose `data_len` exceeds the bytes-bucket
/// capacity would never make progress: `refill()` caps `available`
/// at `capacity`, so the `available >= n` gate is permanently
/// unsatisfiable, the worker arms its retry timerfd at every
/// `RETRY_TIMER_MAX_NANOS` boundary, and the chain re-stalls
/// indefinitely (livelock — the guest's hung-task watchdog is the
/// only escape, default 120 s).
///
/// `consume(n)` policy:
///
/// - `n <= capacity` (normal path): grant when `available >= n`,
///   `available -= n`. Otherwise return false; caller stalls.
/// - `n > capacity` (overconsume): grant when `available >= 0`,
///   `available -= n` (drives negative). Otherwise return false;
///   caller stalls.
///
/// Followers (any subsequent request, regardless of size) wait
/// proportional to the accumulated debt: while `available < 0`,
/// every `consume(m)` for `m > 0` returns false, and
/// `nanos_until_n_tokens` reports the time required for `refill()`
/// to bring `available` back to either `m` (normal-sized followers)
/// or `0` (oversized followers). The negative-balance design
/// converts a stall-forever livelock into a finite-but-proportional
/// wait: the oversized request runs immediately at the cost of
/// debt that subsequent requests amortise over.
///
/// `consume(0)` and `can_consume(0)` always succeed — even when the
/// bucket is in debt — because a zero-cost request has no token
/// charge. T_FLUSH chains issue `bytes_bucket.consume(0)` and must
/// not stall on a sibling oversized-T_OUT debt.
///
/// # Non-blocking invariant
///
/// `consume` runs inside `drain_bracket_impl`, which executes on the
/// thread that owns the `BlkWorkerState` — the worker thread in
/// production (`cfg(not(test))`), the vCPU/test thread inline
/// (`cfg(test)`). Both modes require the bucket to be non-blocking:
///
/// - **Production worker thread:** A sleeping worker cannot
///   service the STOP_TOKEN or KICK_TOKEN epoll events, so a
///   blocked bucket would defer worker shutdown and queue-notify
///   delivery for the duration of the sleep. Liveness for the
///   worker means each `consume` / `can_consume` must return
///   immediately whether the bucket is satisfied or empty; the
///   caller (drain_bracket_impl) handles exhaustion by
///   undoing the pop with
///   `set_next_avail(prev.wrapping_sub(1))` and returning
///   `DrainOutcome::ThrottleStalled { wait_nanos }`, after which
///   the worker arms a CLOCK_MONOTONIC timerfd
///   (THROTTLE_TOKEN) so the chain is re-drained when the bucket
///   refills.
///
/// - **`cfg(test)` inline path:** Tests call `process_requests`
///   synchronously and assert on the post-call queue + counter
///   state; a sleeping bucket would deadlock the test thread on
///   the throttle clock. Tests exercise the post-stall retry by
///   stepping the bucket forward via `set_last_refill_for_test`
///   and re-issuing `QUEUE_NOTIFY`, since the inline path
///   discards `DrainOutcome` and has no worker thread to arm a
///   timerfd. The synchronous test surface depends on `consume`
///   always returning promptly.
///
/// **Critical invariant: this bucket NEVER calls `thread::sleep` or
/// any blocking syscall.** `std::thread::sleep` in particular
/// retries on EINTR per the Rust std source, so even a
/// signal-targeted thread would not wake until the sleep duration
/// elapsed.
///
/// Throttle exhaustion is NOT surfaced to the guest as `S_IOERR`.
/// `drain_bracket_impl` rewinds the queue cursor and arms the
/// retry timerfd; `throttled_count` ticks for host-side
/// observability while the guest sees only the deferred latency.
/// Realism of disk latency is NOT a goal of the test fixture;
/// worker-thread liveness (production) and synchronous test
/// progress (`cfg(test)`) are.
///
/// `unlimited` (capacity == 0) is a fast path that always returns
/// true. `DiskConfig` materialises this when neither IOPS nor bytes
/// throttle is set; the cold path here would otherwise charge a
/// monotonic-clock read per request unconditionally.
#[derive(Debug)]
pub(crate) struct TokenBucket {
    pub(crate) capacity: u64,
    pub(crate) refill_rate: u64, // tokens per second
    /// Current token balance. `i64` (not `u64`) so an oversized
    /// `consume(n > capacity)` can drive the balance negative
    /// rather than re-stall forever. Range invariant:
    /// `i64::MIN + 1 <= available <= i64::try_from(capacity).unwrap_or(i64::MAX)`.
    /// The reachable lower bound is `i64::MIN + 1` (driven by
    /// `0.saturating_sub(i64::MAX)` when an `available == 0`
    /// overconsume hits the largest `n_signed` that the consume
    /// gate accepts). `i64::MIN` itself is unreachable from the
    /// consume gate but is handled defensively by `saturating_sub`.
    /// Negative values represent debt accumulated by a prior
    /// overconsume; `refill()` monotonically pays it down at
    /// `refill_rate`. See the type-level "Overconsumption" doc for
    /// the full policy.
    pub(crate) available: i64,
    pub(crate) last_refill: Instant,
    pub(crate) unlimited: bool,
    /// Test-only override for `nanos_until_n_tokens`. When `Some(n)`,
    /// `nanos_until_n_tokens` returns `n` directly, bypassing
    /// `refill()` and the deficit math. Lets tests pin the
    /// `wait_nanos == 0` inline-re-drain branch in
    /// `worker_thread_main` without relying on real-time behaviour
    /// of `Instant::now()` / `Duration::from_nanos`. Production
    /// code never sets this — it remains `None` outside tests.
    #[cfg(test)]
    pub(crate) forced_nanos_until_n_tokens: Option<u64>,
}

impl TokenBucket {
    pub(crate) fn unlimited() -> Self {
        Self {
            capacity: 0,
            refill_rate: 0,
            available: 0,
            last_refill: Instant::now(),
            unlimited: true,
            #[cfg(test)]
            forced_nanos_until_n_tokens: None,
        }
    }

    /// Build a bucket with the given capacity that refills at the
    /// rate per second. `capacity == 0` becomes `unlimited()` (no
    /// throttle).
    ///
    /// `capacity > i64::MAX` is clamped at `i64::MAX` for the
    /// initial `available` value. The bucket's `> capacity`
    /// overconsume gate still uses the original `u64` capacity, so
    /// the only observable effect is that the seed balance can
    /// hold at most `i64::MAX` tokens — an immaterial bound for
    /// realistic ktstr throttle settings (IOPS in the millions,
    /// bytes/sec in the GB/s, both << 2^63).
    pub(crate) fn new(capacity: u64, refill_rate_per_sec: u64) -> Self {
        if capacity == 0 || refill_rate_per_sec == 0 {
            return Self::unlimited();
        }
        Self {
            capacity,
            refill_rate: refill_rate_per_sec,
            available: i64::try_from(capacity).unwrap_or(i64::MAX),
            last_refill: Instant::now(),
            unlimited: false,
            #[cfg(test)]
            forced_nanos_until_n_tokens: None,
        }
    }

    pub(crate) fn refill(&mut self) {
        if self.unlimited {
            return;
        }
        let now = Instant::now();
        let elapsed_ns = now.duration_since(self.last_refill).as_nanos();
        if elapsed_ns == 0 {
            return;
        }
        // tokens = refill_rate * elapsed_seconds; do the math in u128
        // to avoid overflow on long stalls. Refill rate is small
        // enough (typically <= a few million per second) that the
        // multiplication fits in u128 trivially.
        let new_tokens = (self.refill_rate as u128 * elapsed_ns) / 1_000_000_000;
        let new_tokens_u64 = u64::try_from(new_tokens).unwrap_or(u64::MAX);
        // Only advance `last_refill` when at least one token was
        // granted. At low rates (e.g. 100 IOPS = one token every
        // 10 ms) sub-10ms calls produce `new_tokens_u64 == 0`; if
        // we updated `last_refill` anyway, the elapsed window
        // would reset on every call and the bucket would never
        // refill in steady state. Preserving the old `last_refill`
        // on a 0-token computation lets elapsed time accumulate
        // across calls until enough has passed for at least one
        // whole token to be granted.
        if new_tokens_u64 == 0 {
            return;
        }
        // Pay down accumulated debt (negative `available`) and cap
        // the positive side at `capacity`. `saturating_add` with an
        // i64 addend pinned to `i64::MAX` keeps the addition safe
        // for pathological elapsed-time values; `min(cap_i64)` then
        // enforces the upper bound. `i64::try_from(self.capacity)`
        // matches the seed clamp in `new()`.
        let add = i64::try_from(new_tokens_u64).unwrap_or(i64::MAX);
        let cap_i64 = i64::try_from(self.capacity).unwrap_or(i64::MAX);
        self.available = self.available.saturating_add(add).min(cap_i64);
        self.last_refill = now;
    }

    /// Drain `n` tokens. Returns `true` on success, `false` when the
    /// bucket cannot satisfy the request (caller stalls).
    ///
    /// Three branches:
    /// - `n == 0`: always succeeds with no mutation. Zero-cost
    ///   requests (T_FLUSH on the bytes bucket) must not stall on a
    ///   sibling oversized-T_OUT debt.
    /// - `n > capacity` (overconsume): grant when `available >= 0`,
    ///   set `available -= n` (drives negative). The negative
    ///   balance is paid down by subsequent `refill()` calls;
    ///   followers stall via the normal-path branch until the debt
    ///   clears.
    /// - `n <= capacity` (normal): grant when `available >= n`, set
    ///   `available -= n`. Followers with `available < n` stall.
    ///
    /// `n > i64::MAX` is rejected (`false`). The drain caller caps
    /// `data_len` at `SEG_MAX × SIZE_MAX = 128 MiB << 2^63` so the
    /// rejection branch is unreachable from production callers,
    /// but the guard prevents silent wraparound in `n as i64`
    /// against a future caller bypassing the caps.
    pub(crate) fn consume(&mut self, n: u64) -> bool {
        if self.unlimited {
            return true;
        }
        if n == 0 {
            // Skips refill — a stream of zero-cost requests does not advance last_refill.
            return true;
        }
        self.refill();
        let Ok(n_signed) = i64::try_from(n) else {
            return false;
        };
        let granted = if n > self.capacity {
            self.available >= 0
        } else {
            self.available >= n_signed
        };
        if !granted {
            return false;
        }
        // saturating_sub keeps `available >= i64::MIN` in the
        // pathological-`n` case (n_signed near i64::MAX, available
        // small-positive). The realistic range stays well above
        // i64::MIN; the saturate is defense-in-depth.
        self.available = self.available.saturating_sub(n_signed);
        true
    }

    /// Check whether `n` tokens are currently available without
    /// consuming them. Used by the per-request "both buckets must
    /// pass" gate so a request that fails the bytes check doesn't
    /// silently drain the ops bucket (or vice versa). Refills
    /// on-demand so the answer reflects up-to-the-instant state.
    ///
    /// Returns the same predicate `consume(n)` would: zero-cost
    /// requests always pass, oversized requests pass when
    /// `available >= 0`, normal requests pass when
    /// `available >= n`. `n > i64::MAX` returns false.
    pub(crate) fn can_consume(&mut self, n: u64) -> bool {
        if self.unlimited {
            return true;
        }
        if n == 0 {
            return true;
        }
        self.refill();
        let Ok(n_signed) = i64::try_from(n) else {
            return false;
        };
        if n > self.capacity {
            self.available >= 0
        } else {
            self.available >= n_signed
        }
    }

    /// Refill-deficit estimate: nanoseconds required for the bucket
    /// to permit `consume(need)` (post-refill). Returns `0` for the
    /// unlimited fast path, the zero-cost case, and when the
    /// post-refill state already satisfies `can_consume(need)`. Used
    /// to size the worker thread's retry-timer when a request
    /// stalls on throttle exhaustion.
    ///
    /// The deficit calculation matches `consume`'s grant predicate:
    ///
    /// - `need > capacity` (overconsume retry): the gate is
    ///   `available >= 0`. With `available < 0`, the caller waits
    ///   `-available` tokens worth of refill time.
    /// - `need <= capacity` (normal retry): the gate is
    ///   `available >= need`. With `available < need` (possibly
    ///   negative from a prior overconsume), the caller waits
    ///   `need - available` tokens — `available`'s sign is
    ///   handled directly (subtracting a negative `available`
    ///   from `need` widens the deficit, which is exactly the
    ///   "wait proportional to accumulated debt" property the
    ///   overconsume policy promises).
    ///
    /// All math in `i128`/`u128` to keep deficits accurate even
    /// when `available` approaches `i64::MIN` (the most-negative
    /// post-overconsume balance, pinned by `consume`'s
    /// `saturating_sub`) and `need` approaches `u64::MAX`.
    /// Capping at `u64::MAX` nanoseconds saturates if `need` is
    /// pathologically large relative to `refill_rate`; the caller
    /// (worker_thread_main) further clamps the timer arm to
    /// `RETRY_TIMER_MAX_NANOS` (1 s) so a pathological refill
    /// rate can't push the retry past the guest's hung-task
    /// watchdog (`kernel.hung_task_timeout_secs`, default 120 s —
    /// virtio_blk has no `mq_ops->timeout`, so an unpublished
    /// request hangs until the watchdog fires or a higher layer
    /// retries).
    ///
    /// Caller has already failed `can_consume(need)` so the
    /// non-zero return is the dominant path; the post-refill
    /// `0` shortcut covers the race where the bucket refilled
    /// between the upstream `can_consume` and this call.
    pub(crate) fn nanos_until_n_tokens(&mut self, need: u64) -> u64 {
        // Test seam: when `forced_nanos_until_n_tokens` is set,
        // return it directly without touching refill state. Lets
        // tests pin the worker loop's `wait_nanos == 0` inline
        // re-drain branch (and any other deficit-driven decision)
        // without depending on real wall-clock behaviour. The
        // override is taken even on the unlimited fast path so a
        // test can simulate "throttled bucket reports zero deficit"
        // against an unlimited bucket without rebuilding.
        #[cfg(test)]
        if let Some(forced) = self.forced_nanos_until_n_tokens {
            return forced;
        }
        if self.unlimited || need == 0 {
            return 0;
        }
        self.refill();
        // Deficit in i128 to avoid overflow: `available` ranges
        // down to `i64::MIN` post-overconsume (pinned by
        // `consume`'s `saturating_sub`); subtracting from a u64
        // `need` near `u64::MAX` would otherwise overflow i64.
        let deficit_i128: i128 = if need > self.capacity {
            if self.available >= 0 {
                return 0;
            }
            -(self.available as i128)
        } else {
            let avail_i128 = self.available as i128;
            let need_i128 = need as i128;
            if avail_i128 >= need_i128 {
                return 0;
            }
            need_i128 - avail_i128
        };
        debug_assert!(
            deficit_i128 > 0,
            "deficit must be positive after the early-return \
             arms above (need={need}, available={})",
            self.available,
        );
        // tokens / (tokens/sec) = sec. Want nanos: deficit * 1e9 /
        // refill_rate, rounded up. ceil-div via div_ceil; in u128
        // to fit the post-multiply numerator for large deficits.
        let deficit_u128 = deficit_i128 as u128;
        let numerator = deficit_u128 * 1_000_000_000;
        let denom = self.refill_rate as u128;
        let nanos_u128 = numerator.div_ceil(denom);
        u64::try_from(nanos_u128).unwrap_or(u64::MAX)
    }

    /// Test-only knob: rewind `last_refill` so the next `refill()`
    /// computes "as if X ago". Lets tests pin throttle behaviour
    /// without burning real wall time. Production code uses
    /// `Instant::now()` exclusively — no trait injection, because
    /// per-request overhead matters and the bucket's correctness is
    /// independent of clock source (the formula is a per-second
    /// rate that any monotonic clock produces correctly).
    #[cfg(test)]
    pub(crate) fn set_last_refill_for_test(&mut self, t: Instant) {
        self.last_refill = t;
    }

    /// Test-only knob: pin the next (and every subsequent)
    /// `nanos_until_n_tokens` return value to `forced` until
    /// cleared via `clear_forced_nanos_until_n_tokens_for_test`.
    /// The override short-circuits before `refill()`, so the
    /// reported deficit is strictly deterministic regardless of
    /// `Instant::now()`. Pairs with the worker-loop test seam for
    /// the `wait_nanos == 0` inline-re-drain branch — a test can
    /// force one bucket to report zero deficit and assert the
    /// worker's `StallAction` decision without timing tolerances.
    #[cfg(test)]
    pub(crate) fn set_forced_nanos_until_n_tokens_for_test(&mut self, forced: u64) {
        self.forced_nanos_until_n_tokens = Some(forced);
    }

    /// Test-only knob: drop a previously-installed
    /// `forced_nanos_until_n_tokens` override so subsequent calls
    /// fall through to the refill+deficit math.
    #[cfg(test)]
    pub(crate) fn clear_forced_nanos_until_n_tokens_for_test(&mut self) {
        self.forced_nanos_until_n_tokens = None;
    }
}

/// Materialise a [`DiskThrottle`] into a pair of token buckets.
/// `None` on the rate field becomes the unlimited fast path.
/// `Option<NonZeroU64>` is unwrapped via `NonZeroU64::get` so the
/// bucket sees a plain `u64`; the type-level invariant (the value
/// can't be 0) means the `if rate == 0` branch in
/// `TokenBucket::new` is unreachable from this caller — kept there
/// for defense-in-depth against direct construction.
///
/// # Bucket capacity
///
/// When `*_burst_capacity` is set, the bucket capacity equals the
/// burst value (peak instantaneous burst the device absorbs).
/// When `*_burst_capacity` is `None`, the capacity falls back to
/// the refill rate — the historical 1-second-burst default.
/// [`DiskThrottle::validate`] enforces `burst >= rate` and rejects
/// burst-without-rate at VM build time, so this materialisation
/// step trusts the input and never down-clamps the burst below the
/// rate (such a bucket would discard refilled tokens immediately
/// and silently reduce the steady-state rate).
pub(crate) fn buckets_from_throttle(throttle: DiskThrottle) -> (TokenBucket, TokenBucket) {
    let ops_bucket = throttle.iops.map_or_else(TokenBucket::unlimited, |nz| {
        let r = nz.get();
        let cap = throttle.iops_burst_capacity.map_or(r, NonZeroU64::get);
        TokenBucket::new(cap, r)
    });
    let bytes_bucket = throttle
        .bytes_per_sec
        .map_or_else(TokenBucket::unlimited, |nz| {
            let r = nz.get();
            let cap = throttle.bytes_burst_capacity.map_or(r, NonZeroU64::get);
            TokenBucket::new(cap, r)
        });
    (ops_bucket, bytes_bucket)
}

#[cfg(test)]
mod tests {
    //! Tier 2 of the test co-location split: the TokenBucket /
    //! buckets_from_throttle / nanos_until_n_tokens / DiskThrottle
    //! primitive tests live here next to their type definitions.
    //! All ~25 tests are self-contained (no fixture from
    //! `super::testing` is required) — the throttle primitive
    //! is exercised at the value-API level without constructing a
    //! `VirtioBlk`.
    use super::super::DiskThrottle;
    use super::*;
    use std::num::NonZeroU64;

    #[test]
    fn token_bucket_unlimited_always_grants() {
        let mut tb = TokenBucket::unlimited();
        for _ in 0..1_000_000 {
            assert!(tb.consume(1));
        }
    }

    #[test]
    fn token_bucket_consumes_capacity() {
        let mut tb = TokenBucket::new(100, 1); // 100 capacity, refills 1/sec
        for _ in 0..100 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
    }

    #[test]
    fn token_bucket_refills_over_time() {
        // Slow refill (10/sec) so the consume loop's wall-time
        // overhead doesn't refill enough to mask the bucket
        // exhaustion. At 10 tokens/sec, ~100ms must elapse before
        // even a single token refills.
        let mut tb = TokenBucket::new(100, 10);
        for _ in 0..100 {
            assert!(tb.consume(1));
        }
        assert!(
            !tb.consume(1),
            "bucket exhausted; refill too slow to top up in microseconds",
        );
        // Sleep enough to refill at least 1 token (>=100ms at
        // 10/sec). Use 200ms for slack.
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert!(
            tb.consume(1),
            "after 200ms at 10 tokens/sec, at least 1 should be available",
        );
    }

    #[test]
    fn throttle_zero_rate_becomes_unlimited() {
        // The DiskThrottle public surface uses Option<NonZeroU64>, so
        // a zero rate is unrepresentable at construction. This test
        // pins TokenBucket's defense-in-depth fallback at the
        // primitive layer: if a future caller (or a reflective
        // construction path that bypasses NonZeroU64) hands
        // TokenBucket::new a 0 rate, the bucket must become the
        // unlimited fast path rather than infinitely-failing
        // consume(1) calls.
        let mut tb = TokenBucket::new(0, 100);
        for _ in 0..10_000 {
            assert!(tb.consume(1));
        }
        let mut tb = TokenBucket::new(100, 0);
        for _ in 0..10_000 {
            assert!(tb.consume(1));
        }
    }

    #[test]
    fn token_bucket_refill_uses_elapsed_wall_time() {
        // Drain to empty, sleep 1 second, observe a full refill.
        // Use small absolute numbers (<=10) so the test is fast and
        // any timing slop in the test harness produces a rounding
        // difference of <= 1 token rather than a flake.
        let mut tb = TokenBucket::new(10, 10);
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
        std::thread::sleep(std::time::Duration::from_millis(1100));
        // After ~1.1s at 10/sec, capacity caps at 10. Verify we get
        // back the full bucket.
        for _ in 0..10 {
            assert!(
                tb.consume(1),
                "bucket should have refilled to capacity after sleep"
            );
        }
    }

    #[test]
    fn token_bucket_consume_zero_is_free() {
        // A zero-byte data transfer (e.g. T_FLUSH) should not consume
        // any bytes-bucket tokens. Pin that consume(0) is a no-op
        // success.
        let mut tb = TokenBucket::new(10, 10);
        for _ in 0..1_000 {
            assert!(tb.consume(0));
        }
        // Bucket still full.
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
    }

    /// `set_forced_nanos_until_n_tokens_for_test` pins
    /// `nanos_until_n_tokens` to the injected value regardless of
    /// the bucket's actual deficit, refill timing, or `unlimited`
    /// fast path. Pins the contract the worker-loop tests depend
    /// on for the `wait_nanos == 0` inline-re-drain branch:
    /// without the override, exercising that branch would require
    /// hitting a microscopic refill window between `can_consume`
    /// and `nanos_until_n_tokens` — flaky under any real-world
    /// scheduling. With the override, the branch is reachable
    /// deterministically.
    #[test]
    fn token_bucket_forced_nanos_until_n_tokens_overrides_deficit() {
        // Throttled bucket with a real deficit: capacity 10, rate
        // 10/sec, balance == 0 (drain it). Real
        // `nanos_until_n_tokens(5)` would compute a positive
        // deficit (≥ 500 ms). The override forces 0.
        let mut tb = TokenBucket::new(10, 10);
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        assert!(!tb.consume(1));
        // Without the override, the deficit must be > 0 (we drained
        // the whole bucket; refill rate is 10/sec).
        let real_nanos = tb.nanos_until_n_tokens(5);
        assert!(
            real_nanos > 0,
            "real deficit must be positive after drain; got {real_nanos}",
        );
        // Install override and confirm the value is returned.
        tb.set_forced_nanos_until_n_tokens_for_test(0);
        assert_eq!(
            tb.nanos_until_n_tokens(5),
            0,
            "override must force 0 regardless of deficit",
        );
        assert_eq!(
            tb.nanos_until_n_tokens(u64::MAX),
            0,
            "override is need-independent; u64::MAX still returns the forced value",
        );
        // Override a non-zero value too.
        tb.set_forced_nanos_until_n_tokens_for_test(123_456);
        assert_eq!(tb.nanos_until_n_tokens(1), 123_456);
        // Clear and verify fall-through to the real deficit math.
        tb.clear_forced_nanos_until_n_tokens_for_test();
        let post_clear = tb.nanos_until_n_tokens(5);
        assert!(
            post_clear > 0,
            "clearing override must restore real deficit math; got {post_clear}",
        );
    }

    /// `set_forced_nanos_until_n_tokens_for_test` also overrides
    /// the `unlimited` fast path so a test can simulate "throttled
    /// bucket reports zero deficit" against an unlimited bucket
    /// without rebuilding it. This matches the seam doc on
    /// `nanos_until_n_tokens` (the override is checked before
    /// `unlimited`).
    #[test]
    fn token_bucket_forced_nanos_until_n_tokens_overrides_unlimited() {
        let mut tb = TokenBucket::unlimited();
        // Unlimited buckets normally return 0 already, so prove the
        // override by pinning a non-zero value.
        assert_eq!(tb.nanos_until_n_tokens(1_000), 0);
        tb.set_forced_nanos_until_n_tokens_for_test(7);
        assert_eq!(
            tb.nanos_until_n_tokens(1_000),
            7,
            "override must take precedence over the unlimited fast path",
        );
        tb.clear_forced_nanos_until_n_tokens_for_test();
        assert_eq!(tb.nanos_until_n_tokens(1_000), 0);
    }

    /// `nanos_until_n_tokens` saturates at `u64::MAX` when the
    /// deficit is pathologically large relative to refill_rate.
    /// Path: `numerator = deficit * 1e9` in u128 → divide by
    /// `refill_rate` (also in u128) → `try_from` to u64 returns
    /// `u64::MAX` on overflow via `unwrap_or(u64::MAX)`.
    ///
    /// To reach the saturation path under the overconsume policy,
    /// the bucket must be in debt before the call: with `available`
    /// non-negative the `need > capacity` branch returns 0
    /// immediately. Drive `available` deeply negative via an
    /// oversized consume, pin `last_refill` so refill yields zero,
    /// then ask for the wait — the deficit is effectively u64-scale
    /// and the post-multiply numerator (~u64::MAX * 1e9) exceeds
    /// u64::MAX in the final try_from cast, hitting the saturate arm.
    #[test]
    fn nanos_until_n_tokens_saturates_at_u64_max() {
        // Capacity = 1, refill_rate = 1/sec. Overconsume(i64::MAX)
        // pushes available from 1 to 1 - i64::MAX = -(i64::MAX - 1)
        // = i64::MIN + 2 (well below zero, near i64::MIN).
        let mut tb = TokenBucket::new(1, 1);
        // Pin last_refill at construction so the consume() call
        // below cannot pick up stray wall-clock refill between
        // `new()`'s `Instant::now()` and our subsequent calls.
        // Without this pin a slow test runner could trickle a
        // refill-rate=1 token in, perturb `available`, and shift
        // the post-overconsume balance off `i64::MIN + 2`.
        tb.set_last_refill_for_test(std::time::Instant::now());
        let huge = i64::MAX as u64;
        assert!(tb.consume(huge), "overconsume succeeds when available >= 0");
        // Re-pin so the in-place refill inside `can_consume` and
        // `nanos_until_n_tokens` yields 0 tokens — keeps the deficit
        // math deterministic.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert!(
            !tb.can_consume(1),
            "post-overconsume balance must be negative — \
             any positive consume rejected by the gate",
        );
        // Re-pin again before the final `nanos_until_n_tokens` call
        // (`can_consume` above also issues a refill).
        tb.set_last_refill_for_test(std::time::Instant::now());
        // need (u64::MAX) > capacity (1); blocker is available < 0.
        // deficit_i128 = -(available as i128); with available near
        // i64::MIN, deficit is ~i64::MAX. nanos = deficit * 1e9 / 1
        // overflows u64 → saturates.
        assert_eq!(
            tb.nanos_until_n_tokens(u64::MAX),
            u64::MAX,
            "u64-scale deficit at rate=1 must saturate at u64::MAX",
        );
    }

    /// `nanos_until_n_tokens` ceil-divs deficit/rate to nanoseconds.
    /// With capacity=10, rate=10/sec, drained, deficit=5 → required
    /// time = 5 / 10 = 0.5 s = 500_000_000 ns. The ceil-div formula
    /// `(deficit * 1e9 + rate - 1) / rate` matches `div_ceil` and
    /// produces the exact value for evenly-divisible deficits.
    #[test]
    fn nanos_until_n_tokens_ceil_div_exact() {
        let mut tb = TokenBucket::new(10, 10);
        // Drain.
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        // Pin last_refill so refill is a no-op (elapsed_ns matters,
        // but at this rate <1 token would refill within microseconds
        // anyway — pinning makes the math deterministic).
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(5),
            500_000_000,
            "deficit=5 with rate=10/sec must equal 0.5s = 500_000_000 ns",
        );
    }

    /// `nanos_until_n_tokens` short-circuits on the unlimited fast
    /// path — even `u64::MAX` returns 0. Without this, callers
    /// would compute a fake wait for a bucket that always grants.
    #[test]
    fn nanos_until_n_tokens_unlimited_returns_zero() {
        let mut tb = TokenBucket::unlimited();
        assert_eq!(
            tb.nanos_until_n_tokens(u64::MAX),
            0,
            "unlimited bucket must return 0 regardless of need",
        );
    }

    /// `nanos_until_n_tokens` returns 0 when the in-place refill
    /// brings `available >= need`. Path: drain bucket, set
    /// last_refill 2s into the past → refill grants enough tokens
    /// to satisfy `need=1`, and the early-return arm fires. This is
    /// the case `clamp_retry_nanos(0) → 1` exists for: the bucket
    /// already refilled between the upstream `can_consume` check
    /// and this nanosecond computation.
    #[test]
    fn nanos_until_n_tokens_post_refill_returns_zero() {
        let mut tb = TokenBucket::new(10, 10);
        // Drain.
        for _ in 0..10 {
            assert!(tb.consume(1));
        }
        // Step last_refill 2s into the past — refill grants 20
        // tokens, capped at capacity=10. available=10 >= need=1
        // post-refill → return 0.
        tb.set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(2));
        assert_eq!(
            tb.nanos_until_n_tokens(1),
            0,
            "post-refill `available >= need` must return 0",
        );
    }

    /// A single oversized request (`n > capacity`) is granted
    /// immediately when the bucket is non-negative, driving
    /// `available` negative. Without this allowance the chain
    /// would stall forever — `refill()` caps `available` at
    /// `capacity`, so `available >= n` is permanently
    /// unsatisfiable for `n > capacity`. Pins the negative-balance
    /// overconsume semantic (see `TokenBucket` type-level
    /// "Overconsumption" doc).
    #[test]
    fn token_bucket_oversized_grants_and_drives_negative() {
        let mut tb = TokenBucket::new(100, 100);
        // 150 > capacity (100) and available (100) >= 0 → grant.
        assert!(
            tb.consume(150),
            "oversized consume must grant when available >= 0",
        );
        // Pin last_refill so the in-place refill inside
        // `nanos_until_n_tokens` yields 0 tokens — keeps the deficit
        // math deterministic (rate=100/sec, sub-millisecond elapsed
        // from `consume(150)` to here floor-divides to 0 anyway).
        tb.set_last_refill_for_test(std::time::Instant::now());
        // Probe the post-debt balance via the public deficit API
        // instead of reading the private `available` field. need=101
        // is oversized (>capacity=100); the gate is `available >= 0`
        // and the deficit is `-available`. Balance of -50 → deficit
        // 50 → wait at rate=100 = 50 / 100 = 0.5 s = 500_000_000 ns.
        // A regression that drove `available` to a different value
        // would surface here as a wrong nanos number.
        assert_eq!(
            tb.nanos_until_n_tokens(101),
            500_000_000,
            "post-overconsume debt of 50 (capacity=100, n=150 → 100-150) \
             produces 500ms at rate=100/sec; deficit math: \
             -available * 1e9 / refill_rate = 50 * 1e9 / 100",
        );
        // can_consume mirrors consume; a follower observation also
        // sees the post-debt state.
        assert!(
            !tb.can_consume(1),
            "follower (any size) stalls while bucket is in debt",
        );
    }

    /// Two oversized requests back-to-back: the first grants
    /// (driving available negative), the second stalls because
    /// `available < 0` fails the overconsume gate. The follower
    /// must wait for `refill()` to climb back to >= 0 — that's
    /// the "wait proportional to accumulated debt" property of
    /// the overconsume policy.
    #[test]
    fn token_bucket_oversized_back_to_back_second_stalls() {
        let mut tb = TokenBucket::new(100, 100);
        // First oversized grants. Probe debt via deficit API:
        // need=101 is oversized, deficit = -available. Pin
        // last_refill before the probe so the in-place refill
        // contributes 0 tokens.
        assert!(tb.consume(150));
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(101),
            500_000_000,
            "post-first-overconsume debt of 50 → 500ms at rate=100",
        );
        // Pin last_refill again so the second consume's refill
        // grants no tokens; otherwise the test would race
        // wall-clock refills.
        tb.set_last_refill_for_test(std::time::Instant::now());
        // Second oversized must stall: available (-50) < 0 fails
        // the overconsume gate.
        assert!(
            !tb.consume(150),
            "second oversized must stall while bucket is in debt",
        );
        // Balance unchanged after the failed consume (consume
        // returned false → no decrement). Re-probe the deficit:
        // same value as before proves the debt was not deepened.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(101),
            500_000_000,
            "failed consume must NOT deepen the debt — deficit \
             unchanged at 50",
        );
        // can_consume mirrors consume.
        assert!(!tb.can_consume(150));
    }

    /// `nanos_until_n_tokens` reports the time-to-zero for an
    /// oversized follower (need > capacity, available < 0): wait
    /// = -available / refill_rate. With available=-50 and
    /// rate=100/sec, the wait is 50/100 = 0.5 s = 500 ms ns.
    #[test]
    fn nanos_until_n_tokens_oversized_follower_waits_for_zero() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(tb.consume(150));
        // Pin last_refill so the in-place refill yields 0 tokens
        // and the deficit math is deterministic. The post-overconsume
        // balance of -50 is exercised by the deficit assertion below
        // — a regression that drove the balance to a different value
        // would surface as a wrong nanos number.
        tb.set_last_refill_for_test(std::time::Instant::now());
        // need (200) > capacity (100); blocker is available < 0.
        // deficit = -(-50) = 50; nanos = 50 * 1e9 / 100 = 500_000_000.
        assert_eq!(
            tb.nanos_until_n_tokens(200),
            500_000_000,
            "oversized follower waits for available to climb to 0",
        );
    }

    /// `nanos_until_n_tokens` reports the wider deficit for a
    /// normal-sized follower behind an overconsume debt: wait
    /// = (need + |available|) / refill_rate. With need=10,
    /// available=-50, rate=100/sec, wait = 60 / 100 = 0.6 s.
    /// Verifies the negative-available case in the i128 deficit
    /// math.
    #[test]
    fn nanos_until_n_tokens_normal_follower_after_debt() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(tb.consume(150));
        // Pin last_refill so the in-place refill yields 0 tokens.
        // Post-overconsume balance of -50 is implicit in the
        // deficit assertion below.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(10),
            600_000_000,
            "normal-sized follower waits for available to climb \
             from -50 to need=10",
        );
    }

    /// `consume(n)` rejects `n > i64::MAX` to prevent silent
    /// wraparound when casting `n as i64`. The drain caller caps
    /// `data_len` well below i64::MAX (SEG_MAX × SIZE_MAX =
    /// 128 MiB), so this branch is unreachable from production
    /// callers — defense-in-depth against a future caller that
    /// bypasses the gate. `can_consume(n)` mirrors the rejection.
    #[test]
    fn token_bucket_consume_rejects_n_above_i64_max() {
        let mut tb = TokenBucket::new(100, 100);
        let pathological = (i64::MAX as u64) + 1;
        assert!(
            !tb.can_consume(pathological),
            "n > i64::MAX must fail can_consume — i64 cast guard",
        );
        assert!(
            !tb.consume(pathological),
            "n > i64::MAX must fail consume — i64 cast guard",
        );
        // Balance unchanged after rejection — the full seed of
        // capacity tokens must still be grantable. Probing via
        // `can_consume(100)` is a tight check: capacity caps
        // `available` at 100, so this passes if and only if
        // `available == 100`.
        assert!(
            tb.can_consume(100),
            "rejection must NOT decrement balance — full seed of \
             100 tokens must still be grantable",
        );
        // u64::MAX also rejected.
        assert!(!tb.consume(u64::MAX));
        assert!(!tb.can_consume(u64::MAX));
        assert!(
            tb.can_consume(100),
            "second rejection round must also leave balance \
             unchanged at the seed value",
        );
    }

    /// `consume(0)` and `can_consume(0)` always succeed — even
    /// when the bucket is in debt. T_FLUSH chains issue
    /// `bytes_bucket.consume(0)` (data_len == 0 for flushes) and
    /// must not stall on a sibling oversized-T_OUT debt.
    /// Distinct from the existing `token_bucket_consume_zero_is_free`
    /// test which checks the happy-path (full bucket); this test
    /// pins the in-debt case.
    #[test]
    fn token_bucket_zero_consume_succeeds_in_debt() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(tb.consume(150));
        // Pin so refills inside `can_consume` / `nanos_until_n_tokens`
        // contribute 0 tokens to the deficit math.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert!(
            !tb.can_consume(1),
            "bucket must be in debt — any positive consume rejected",
        );
        // Zero-cost requests pass regardless of debt.
        assert!(tb.consume(0));
        assert!(tb.can_consume(0));
        // Balance unchanged at -50: re-probe the deficit. need=101
        // is oversized, deficit = -available = 50, nanos =
        // 50 * 1e9 / 100 = 500_000_000.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(101),
            500_000_000,
            "consume(0) / can_consume(0) must NOT touch balance — \
             debt of 50 unchanged after zero-cost requests",
        );
    }

    /// After enough refill, an in-debt bucket recovers and admits
    /// followers normally. Pin the recovery semantic: with
    /// available=-50 at rate=100/sec, ≥0.5 s of wall-clock refill
    /// brings available back to >= 0; subsequent `consume(50)`
    /// succeeds.
    #[test]
    fn token_bucket_debt_clears_with_refill() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(tb.consume(150));
        // Post-overconsume balance is -50 — verified implicitly by
        // the refill-recovery deficit assertion below (a different
        // post-consume balance would shift the post-refill deficit
        // off the asserted 10_000_000 ns).
        // Step last_refill back 1 s — refill grants 100 tokens,
        // pays the -50 debt, brings available to +50, capped at
        // capacity=100. Then consume(50) succeeds.
        tb.set_last_refill_for_test(std::time::Instant::now() - std::time::Duration::from_secs(1));
        assert!(
            tb.consume(50),
            "consume must succeed after refill clears the debt",
        );
        // Post-consume balance is 0: re-pin and probe the deficit
        // for need=1. With available=0 and rate=100/sec, deficit=1
        // → 1 * 1e9 / 100 = 10_000_000 ns. A regression that left
        // the balance non-zero (e.g. failed-to-clear debt → -50, or
        // residual capacity → +50) would surface as a different
        // nanos value.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(1),
            10_000_000,
            "post-consume(50) balance must be 0 — refill paid -50 \
             debt and consume(50) drained the recovered +50; \
             nanos for need=1 at rate=100 = 10_000_000",
        );
    }

    /// An unlimited bucket grants every consume regardless of `n`,
    /// including `n > i64::MAX`. The `unlimited` short-circuit
    /// runs before the `i64::try_from` guard so a hostile guest
    /// against an unconfigured-throttle disk still gets serviced.
    #[test]
    fn token_bucket_unlimited_grants_oversized() {
        let mut tb = TokenBucket::unlimited();
        assert!(tb.consume(u64::MAX));
        assert!(tb.can_consume(u64::MAX));
        assert_eq!(
            tb.nanos_until_n_tokens(u64::MAX),
            0,
            "unlimited bucket reports zero wait for any need",
        );
    }

    /// `consume(n)` with `n == capacity` takes the normal-path
    /// branch (`available >= n_signed`), NOT the overconsume
    /// branch (`available >= 0` for `n > capacity`). Pins the
    /// strict-greater boundary in `consume`'s grant predicate:
    /// changing `n > self.capacity` to `n >= self.capacity` would
    /// re-route exact-capacity drains through the overconsume gate
    /// and let a follower drain to debt without first earning
    /// the full balance back.
    ///
    /// Construction: `new(100, 100)`, `consume(100)` succeeds
    /// (available 100 >= 100), available drops to 0. Pin
    /// `last_refill` so the second call's refill yields no
    /// tokens. Second `consume(100)` must FAIL (normal path:
    /// available 0 < 100; overconsume path also rejects because
    /// 100 is NOT > capacity 100). Available remains 0 — proving
    /// the overconsume branch was not entered (which would have
    /// driven it to -100).
    #[test]
    fn token_bucket_consume_at_capacity_takes_normal_branch() {
        let mut tb = TokenBucket::new(100, 100);
        assert!(
            tb.consume(100),
            "n == capacity must succeed via normal-path \
             available >= n_signed gate",
        );
        // Probe the post-drain balance via the deficit API. need=1
        // is normal-path (gate `available >= 1`), deficit =
        // 1 - available. With available=0, deficit=1, nanos =
        // 1 * 1e9 / 100 = 10_000_000. A regression that drove the
        // balance to e.g. -100 (the overconsume-branch value) would
        // produce deficit=101 → 1_010_000_000 ns — a 100x mismatch
        // that flags the boundary check failure.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(1),
            10_000_000,
            "post-drain balance must be 0 (not negative); deficit=1 \
             at rate=100 = 10_000_000 ns. Overconsume branch entered \
             would drive balance to -100, deficit=101 → 1_010_000_000",
        );
        // Pin last_refill so the next consume's refill grants no
        // tokens; otherwise wall-clock drift could top up the
        // bucket and mask the failure mode the test pins.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert!(
            !tb.consume(100),
            "n == capacity (not > capacity) must fail when \
             available < n_signed; overconsume branch is \
             strictly `n > capacity`, not `n >= capacity`",
        );
        // Re-probe the deficit: balance must STILL be 0 (the failed
        // consume cannot have entered the overconsume branch which
        // would have driven it to -100). Same 10_000_000 ns asserts
        // the balance is unchanged.
        tb.set_last_refill_for_test(std::time::Instant::now());
        assert_eq!(
            tb.nanos_until_n_tokens(1),
            10_000_000,
            "available unchanged at 0 — overconsume branch did \
             NOT drive it to -100 (which would yield 1_010_000_000), \
             proving the boundary check is `>` not `>=`",
        );
    }

    /// `buckets_from_throttle` falls back to capacity = refill_rate
    /// (1-second burst) when `*_burst_capacity` is `None`. Mirrors
    /// the historical default before burst-capacity was a
    /// configurable knob — every existing test that constructs a
    /// throttle without burst fields must continue to observe the
    /// old behaviour.
    #[test]
    fn buckets_from_throttle_default_burst_equals_rate() {
        let throttle = DiskThrottle {
            iops: NonZeroU64::new(1_000),
            bytes_per_sec: NonZeroU64::new(50_000),
            iops_burst_capacity: None,
            bytes_burst_capacity: None,
        };
        let (mut ops, mut bytes) = buckets_from_throttle(throttle);
        assert_eq!(ops.capacity, 1_000);
        assert_eq!(ops.refill_rate, 1_000);
        // `can_consume(capacity)` is a tight equality check on the
        // seed: the gate requires `available >= capacity`, and
        // `refill()` caps `available` at `capacity`, so the predicate
        // passes iff the bucket was seeded full.
        assert!(
            ops.can_consume(1_000),
            "1-second-burst seed equals rate — bucket admits a \
             capacity-sized request immediately",
        );
        assert_eq!(bytes.capacity, 50_000);
        assert_eq!(bytes.refill_rate, 50_000);
        assert!(
            bytes.can_consume(50_000),
            "bytes bucket also seeded full at capacity",
        );
    }

    /// `buckets_from_throttle` honours `*_burst_capacity` when
    /// set: bucket capacity equals the burst value, refill rate
    /// stays at the configured rate. A 5-second burst (capacity
    /// = 5×rate) lets the bucket absorb a 5-second-equivalent
    /// transient before throttling kicks in.
    #[test]
    fn buckets_from_throttle_burst_capacity_overrides_rate() {
        let throttle = DiskThrottle {
            iops: NonZeroU64::new(1_000),
            bytes_per_sec: NonZeroU64::new(50_000),
            iops_burst_capacity: NonZeroU64::new(5_000),
            bytes_burst_capacity: NonZeroU64::new(250_000),
        };
        let (mut ops, mut bytes) = buckets_from_throttle(throttle);
        assert_eq!(ops.capacity, 5_000);
        assert_eq!(ops.refill_rate, 1_000);
        // `can_consume(burst_capacity)` is a tight check that the
        // seed equals the burst — `refill()` caps `available` at
        // `capacity`, so the predicate passes iff seeded full.
        assert!(
            ops.can_consume(5_000),
            "ops bucket seeded equal to burst capacity (5_000)",
        );
        assert_eq!(bytes.capacity, 250_000);
        assert_eq!(bytes.refill_rate, 50_000);
        assert!(
            bytes.can_consume(250_000),
            "bytes bucket seeded equal to burst capacity (250_000)",
        );
    }

    /// `buckets_from_throttle` ignores `*_burst_capacity` when
    /// the matching rate is `None`. The validate() step at the
    /// API boundary rejects this combination, but materialisation
    /// must be safe for any input: a `None`-rate field produces an
    /// unlimited bucket regardless of any orphaned burst value.
    #[test]
    fn buckets_from_throttle_burst_without_rate_is_unlimited() {
        let throttle = DiskThrottle {
            iops: None,
            bytes_per_sec: None,
            iops_burst_capacity: NonZeroU64::new(5_000),
            bytes_burst_capacity: NonZeroU64::new(250_000),
        };
        let (ops, bytes) = buckets_from_throttle(throttle);
        assert!(ops.unlimited);
        assert!(bytes.unlimited);
    }

    /// Mixed configuration: IOPS rate-only, bandwidth rate+burst.
    /// Pins per-dimension independence — setting bandwidth burst
    /// does not affect the IOPS bucket, and vice versa.
    #[test]
    fn buckets_from_throttle_per_dimension_independence() {
        let throttle = DiskThrottle {
            iops: NonZeroU64::new(1_000),
            bytes_per_sec: NonZeroU64::new(50_000),
            iops_burst_capacity: None,
            bytes_burst_capacity: NonZeroU64::new(200_000),
        };
        let (ops, bytes) = buckets_from_throttle(throttle);
        assert_eq!(ops.capacity, 1_000, "iops bucket falls back to rate");
        assert_eq!(bytes.capacity, 200_000, "bytes bucket honours burst");
    }
}
