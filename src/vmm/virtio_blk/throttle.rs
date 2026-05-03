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
}

impl TokenBucket {
    pub(crate) fn unlimited() -> Self {
        Self {
            capacity: 0,
            refill_rate: 0,
            available: 0,
            last_refill: Instant::now(),
            unlimited: true,
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
    let ops_bucket = throttle
        .iops
        .map_or_else(TokenBucket::unlimited, |nz| {
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
