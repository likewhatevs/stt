//! Type-safe wrappers for per-thread metric values.
//!
//! Each registered metric in [`crate::host_state_compare::HOST_STATE_METRICS`]
//! has a kernel-source-grounded semantic category — counter,
//! cumulative-time, peak high-water, instantaneous gauge, byte
//! count, ordinal scalar, categorical, or cpuset. The aggregation
//! pipeline reduces values per category: counters sum, peaks take
//! max, gauges take max, ordinals carry a [min, max] range,
//! categoricals carry the mode (most-frequent value), and cpusets
//! carry an affinity summary.
//!
//! # Temporal window
//!
//! Every counter / cumulative-time / peak / byte-count newtype
//! defined here represents a value that the kernel accumulates
//! across the THREAD LIFETIME — from thread birth to the
//! moment of the procfs read. All of these fields share the
//! same window because they live in the same `task_struct` and
//! tick along with the same task. That shared window is what
//! makes ratios across fields well-defined (e.g.
//! `cpu_efficiency = run_time_ns / (run_time_ns + wait_time_ns)`
//! is a meaningful fraction because both numerator and
//! denominator measure the same task's same lifetime).
//!
//! Cross-file read skew during one capture pass (the
//! capture pipeline reads `/proc/<tid>/stat`, then `/sched`,
//! then `/io`, etc. with a few hundred microseconds of drift
//! between them) is negligible against cumulative-from-birth
//! totals that grow over hours or days of thread runtime —
//! the small in-flight delta during the read is rounding noise
//! relative to the lifetime accumulator. The qualifier holds
//! relative to a lifetime accumulator that has had time to
//! integrate; threads captured very early in their lifetime
//! carry larger relative read-skew error, but their absolute
//! contribution to any group aggregate is correspondingly
//! small (a thread alive for 500 µs cannot meaningfully drag
//! a group total even if its individual reads are skewed by
//! 100 µs).
//!
//! [`crate::host_state_compare`] runs in two modes that both
//! preserve the shared-window property: SHOW renders one
//! snapshot's lifetime totals; COMPARE subtracts two snapshots
//! captured at different wall-clock instants to scope the values
//! to the (capture-A, capture-B) interval. In both modes every
//! field carries the same temporal window, so cross-field ratios
//! and per-thread totals stay well-defined.
//!
//! Two newtypes break this convention deliberately: [`GaugeNs`]
//! (a current-instantaneous reading like the scheduler's current
//! slice) and [`GaugeCount`] (a current count like
//! `signal_struct->nr_threads`) — the per-newtype docs call out
//! the gauge family separately.
//!
//! # Type-system enforcement
//!
//! Encoding the category into the type system surfaces
//! category-mismatched aggregation as a compile error. The
//! [`crate::host_state_compare::AggRule`] dispatch routes each
//! variant through the typed newtype's reduction trait — `Sum*`
//! through [`Summable::sum_across`], `Max*` through
//! [`Maxable::max_across`], `Range*` through
//! [`Rangeable::range_across`], and `Mode*` through
//! [`Modeable::mode_across`] — so a registry entry that pairs a
//! peak field with a sum reduction (e.g. `t.wait_max`
//! ([`PeakNs`]) bound to a `Sum*` rule whose accessor returns a
//! [`Summable`] value) fails to compile rather than producing a
//! meaningless `1×1s ⊕ 1000×1ms` aggregate. This module defines
//! the newtypes and traits the dispatch consumes.
//!
//! # The newtypes
//!
//! - [`MonotonicCount`] — pure counter (only ever goes up across a
//!   thread's lifetime). Examples: `nr_wakeups`, `nr_migrations`,
//!   `voluntary_csw`.
//! - [`DeadCounter`] — same wire shape as [`MonotonicCount`] but
//!   tagged for kernel counters whose update path is permanently
//!   dead (the field exists in `task_struct` but no kernel writer
//!   touches it on any current code path — `nr_wakeups_idle`,
//!   `nr_migrations_cold`, `nr_wakeups_passive` all match this
//!   shape today). Captured for parity with `/proc/<tid>/sched`
//!   line numbers but does NOT implement any reduction trait
//!   ([`Summable`] / [`Maxable`] / [`Rangeable`] / [`Modeable`])
//!   — the value is structurally zero, so every reduction is
//!   trivially zero and rendering it through any of the live
//!   reductions implies "we measured a thing" when in fact we
//!   measured a kernel-side dead pointer. The registry-level
//!   accommodation (a no-op aggregation arm or registry removal)
//!   is the migration batch's problem; this newtype's job is to
//!   make the dead-counter status visible at the field
//!   declaration so the migration can't accidentally pair it
//!   with a [`Summable`]-bound `AggRule` variant.
//! - [`MonotonicNs`] — cumulative-time counter, ns. Examples:
//!   `run_time_ns`, `wait_sum`, `voluntary_sleep_ns`,
//!   `block_sum`, `iowait_sum`, `core_forceidle_sum`.
//! - [`PeakNs`] — lifetime high-water mark, ns. The kernel
//!   updates these via `if (delta > stat->max) stat->max = delta`
//!   inside `update_stats_*` wrappers (kernel/sched/stats.c) and
//!   inline schedstat updates in `kernel/sched/fair.c` (e.g.
//!   `slice_max` in `set_next_entity`, `exec_max` in
//!   `update_se`). Summing peaks is a category error —
//!   `1 thread × 1s peak` carries different meaning than
//!   `1000 threads × 1ms peak`. Examples: `wait_max`,
//!   `sleep_max`, `block_max`, `exec_max`, `slice_max`.
//! - [`GaugeNs`] — instantaneous gauge sampled at capture time, ns.
//!   `fair_slice_ns` is the canonical example. Summing gauges is a
//!   category error — N nearly-identical instantaneous samples
//!   sum to N×gauge with no physical meaning.
//! - [`GaugeCount`] — gauge-family unitless count (u64) that can
//!   go up AND down at runtime. Carries the same Maxable-only
//!   contract as [`GaugeNs`] but renders as a plain count rather
//!   than a nanosecond ladder. `nr_threads` (the process-wide
//!   thread count from `signal_struct->nr_threads`) is the
//!   canonical example — threads spawn and exit so the value is
//!   not monotonic, and the registry reduces it by Max across a
//!   group rather than Sum. Distinct from [`GaugeNs`] because
//!   "thread count" and "current slice in nanoseconds" do not
//!   share a unit; routing nr_threads through GaugeNs would
//!   render it on the ns auto-scale ladder, which is a unit lie.
//! - [`ClockTicks`] — USER_HZ-scaled time. Examples:
//!   `utime_clock_ticks`, `stime_clock_ticks`,
//!   `delayacct_blkio_ticks`. Auto-scale ladder is
//!   `ticks → Kticks → Mticks` (decimal SI), distinct from ns
//!   (also decimal SI, different unit) and bytes (IEC binary).
//! - [`Bytes`] — byte counts. Examples: `allocated_bytes`,
//!   `read_bytes`, `wchar`. Auto-scale ladder is IEC binary
//!   (`B → KiB → MiB → GiB → TiB`).
//! - [`OrdinalI32`] / [`OrdinalU32`] / [`OrdinalU64`] — bounded
//!   scalar, range-aggregated (no sum). [`OrdinalI32`] examples:
//!   `nice` ([-20, 19]), `priority`
//!   (CFS=[0, 39], RT=[-2, -100], DL=-101), `processor` (last
//!   CPU the task ran on; signed for symmetry with `nice` — the
//!   kernel's `task_cpu()` returns `unsigned int`
//!   (`include/linux/sched.h`), but ktstr stores i32 to share
//!   the [`OrdinalI32`] wrapper with the genuinely-signed nice
//!   and priority fields). [`OrdinalU32`] is for u32-backed
//!   ordinal fields like `rt_priority` (real-time priority,
//!   0..99 in practice for SCHED_FIFO / SCHED_RR; the kernel
//!   declares `unsigned int task_struct::rt_priority` in
//!   `include/linux/sched.h`, so a `u32` matches the kernel
//!   field width exactly). [`OrdinalU64`] is reserved for
//!   future ordinal metrics whose kernel-side type genuinely
//!   exceeds `u32::MAX`; no field uses it today.
//! - [`CategoricalString`] — string-valued, mode-aggregated.
//!   `policy` is the only example after phase 2. The `state` char
//!   and `ext_enabled` bool fields stay unwrapped on
//!   [`crate::host_state::ThreadState`]; the
//!   [`crate::host_state_compare::AggRule::Mode`] accessor coerces
//!   them through `String` via `to_string()`/`Display` at the call
//!   site. If a second bool field appears, promote both to a
//!   dedicated `CategoricalBool` wrapper rather than continuing the
//!   ad-hoc coercion.
//! - [`CpuSet`] — `Vec<u32>` of CPU IDs, affinity-aggregated.
//!   `cpu_affinity` is the only example.
//!
//! # The marker traits
//!
//! - [`Summable`] — sum across a group. Implemented by the four
//!   counter newtypes ([`MonotonicCount`], [`MonotonicNs`],
//!   [`ClockTicks`], [`Bytes`]). NOT implemented by [`PeakNs`] /
//!   [`GaugeNs`] / [`GaugeCount`] / [`OrdinalI32`] /
//!   [`OrdinalU32`] / [`OrdinalU64`] / [`CategoricalString`] /
//!   [`CpuSet`]. The trait is sealed via
//!   [`sealed::SummableSealed`] so a downstream crate cannot add
//!   `impl Summable for PeakNs` to bypass the category invariant.
//! - [`Maxable`] — reduce by max. Implemented by [`PeakNs`]
//!   (max-of-peak is "worst peak any contributor saw across its
//!   lifetime"), [`GaugeNs`] (max-of-gauge is "longest current
//!   slice in the bucket"), and [`GaugeCount`] (max-of-count is
//!   "biggest current count any contributor carried"). NOT
//!   implemented by [`Summable`] cumulative counters
//!   ([`MonotonicCount`] / [`MonotonicNs`] / [`ClockTicks`] /
//!   [`Bytes`]) — max-across-snapshots on a lifetime accumulator
//!   reduces to "the last snapshot's value", which is mostly
//!   noise relative to the lifetime-integrated quantity it
//!   reports. NOT implemented by ordinals (those carry a
//!   `[min, max]` range, not a single max), nor by
//!   [`CategoricalString`] (string max has no useful semantic),
//!   nor by [`CpuSet`] (the affinity reduction is a custom
//!   summary, not a bare max). Sealed via
//!   [`sealed::MaxableSealed`].
//!
//!   `max_across` returns `Option<Self>`: `None` for an empty
//!   iterator (so callers can distinguish "no contributors" from
//!   "all contributors had zero"), `Some(largest)` otherwise.
//!   The parallel `Summable::try_sum_across` returns
//!   `Option<Self>` with the same empty-iterator semantics. The
//!   `try_` prefix (rather than `checked_`) avoids colliding
//!   with the stdlib's overflow-detection naming convention —
//!   this is an empty-iterator check, not an arithmetic check.
//! - [`Modeable`] — reduce by mode (most-frequent value).
//!   Implemented by [`CategoricalString`] only. Sealed via
//!   [`sealed::ModeableSealed`].
//! - [`Rangeable`] — reduce by `[min, max]`. Implemented by
//!   [`OrdinalI32`], [`OrdinalU32`], and [`OrdinalU64`]. Sealed
//!   via [`sealed::RangeableSealed`]. `range_across` returns
//!   `Option<Range<Self>>` — the [`Range`] newtype enforces
//!   `min ≤ max` at construction so a downstream consumer cannot
//!   observe a swapped pair.
//!
//! Reductions are exposed as **trait methods** on
//! [`Summable`] / [`Maxable`] / [`Rangeable`] / [`Modeable`].
//! Callers must import the relevant trait (or `use
//! ktstr::metric_types::*;`) to call `T::sum_across(...)` /
//! `T::max_across(...)` / `T::range_across(...)` /
//! `T::mode_across(...)`. The traits double as compile-time
//! markers — a generic site that wants "any summable type" can
//! take `T: Summable` and statically reject `PeakNs`.
//!
//! # Wire-format compatibility
//!
//! Every wrapper carries `#[serde(transparent)]` so the JSON
//! representation matches the unwrapped primitive. The
//! [`crate::host_state::ThreadState`] migration to these
//! newtypes (phase 2) preserves wire format — existing
//! snapshot files (`.hst.zst`) deserialize unchanged.
//!
//! # What this module is NOT
//!
//! - It is NOT a unit-of-measure system. There is no
//!   `MonotonicNs * MonotonicNs = MonotonicNs²` — these wrappers
//!   carry semantic category, not algebraic dimensionality.
//! - It is NOT a runtime-typed value enum (that lives next to
//!   the [`crate::host_state_compare::AggRule`] dispatch). This
//!   module only defines the building-block newtypes.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Newtype wrappers
// ---------------------------------------------------------------------------

/// Pure monotonic counter — only ever goes up over a thread's
/// lifetime, accumulated by the kernel from thread birth to the
/// moment of the procfs read. Sum across a group; delta across
/// snapshots scopes the value to the inter-capture interval.
///
/// Examples in [`crate::host_state_compare::HOST_STATE_METRICS`]:
/// `nr_wakeups`, `nr_migrations`, `voluntary_csw`,
/// `nonvoluntary_csw`, `wait_count`, `iowait_count`,
/// `timeslices`, `minflt`, `majflt`, `syscr`, `syscw`.
///
/// `nr_threads` is NOT in this category — it is a structural
/// gauge that goes up AND down at runtime (threads spawn and
/// exit), so it reduces by max across a group, not sum. See
/// [`GaugeCount`].
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonotonicCount(pub u64);

/// Cumulative-time counter, nanoseconds, accumulated by the
/// kernel from thread birth. Same temporal-window shape as
/// [`MonotonicCount`] but tagged for the ns auto-scale ladder
/// (ns → µs → ms → s).
///
/// Examples: `run_time_ns`, `wait_time_ns`, `wait_sum`,
/// `voluntary_sleep_ns`, `block_sum`, `iowait_sum`,
/// `core_forceidle_sum`.
/// Cross-field ratios (e.g.
/// `run_time_ns / (run_time_ns + wait_time_ns)`) are valid
/// because every [`MonotonicNs`] field on
/// [`crate::host_state::ThreadState`] is integrated over the
/// same thread-lifetime window.
///
/// # u64 backing vs kernel s64
///
/// Some kernel sources for these values are typed `s64` —
/// `sum_sleep_runtime` and `sum_block_runtime` live in
/// `struct sched_statistics` (`include/linux/sched.h`) as
/// `s64`. The capture pipeline parses these via
/// `parsed_ns_from_dotted` in [`crate::host_state`], which
/// returns `None` on negative dotted values; the capture-site
/// `unwrap_or(0)` then collapses `None` to zero before the
/// wrapper is constructed. The `u64` backing here is therefore
/// safe because the parser path guarantees non-negative input
/// — NOT because the kernel field type promises non-negative.
/// Any new writer that bypasses `parsed_ns_from_dotted` must
/// replicate its non-negative guard. A future capture-side
/// change that exposes raw kernel s64 directly would need a
/// sentinel-aware wrapper or a dedicated `SignedNs` newtype.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonotonicNs(pub u64);

/// USER_HZ-scaled tick counter, accumulated by the kernel from
/// thread birth. The kernel exposes user-mode and kernel-mode
/// CPU time, plus delayacct blkio delay, in ticks of the
/// userspace-visible `USER_HZ` frequency. Auto-scale ladder is
/// `ticks → Kticks → Mticks` (decimal SI), kept distinct from
/// ns and bytes so the rendered cell carries the correct unit
/// suffix.
///
/// Examples: `utime_clock_ticks`, `stime_clock_ticks`,
/// `delayacct_blkio_ticks`. Same lifetime-window contract as
/// [`MonotonicNs`]; sum across a group, delta across snapshots.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ClockTicks(pub u64);

/// Byte count, IEC-binary auto-scaled
/// (`B → KiB → MiB → GiB → TiB`). Accumulated by the kernel
/// (or jemalloc, for the per-thread TSD allocator counters)
/// from thread birth.
///
/// Examples: `allocated_bytes`, `deallocated_bytes`, `rchar`,
/// `wchar`, `read_bytes`, `write_bytes`, `cancelled_write_bytes`.
/// Same lifetime-window contract as [`MonotonicNs`]; sum across
/// a group, delta across snapshots.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Bytes(pub u64);

/// Kernel counter whose update path is permanently dead. The
/// field exists in `task_struct` (and is exposed via
/// `/proc/<tid>/sched`) but no kernel writer touches it on any
/// current code path.
///
/// Examples that historically motivated this newtype:
/// `nr_wakeups_idle`, `nr_migrations_cold`, `nr_wakeups_passive`
/// — these fields were removed from
/// [`crate::host_state::ThreadState`] because no kernel code
/// path increments them on 6.16 or 7.1 (no
/// `schedstat_inc(p->stats.nr_wakeups_idle)` /
/// `nr_migrations_cold` / `nr_wakeups_passive` call site exists
/// anywhere under `kernel/`). The newtype remains as
/// infrastructure for future dead counters that get exposed in
/// `/proc` before (or instead of) being wired up as live
/// counters.
///
/// Wire format matches [`MonotonicCount`] (`u64`,
/// `serde(transparent)`); the capture pipeline parses the same
/// procfs lines and stores the same bits. The type-system
/// difference is in the trait list: a [`MonotonicCount`] is
/// [`Summable`] / [`Maxable`], while [`DeadCounter`] is neither.
/// A registry entry that pairs a `DeadCounter` field with a
/// [`Summable`]-bound `AggRule` variant fails to compile,
/// flagging the dead status at the type level rather than
/// surfacing as a "0 + 0 + 0" rendered cell.
///
/// # Migration affordance
///
/// A field can be flipped from [`MonotonicCount`] to
/// [`DeadCounter`] without regenerating any `.hst.zst` snapshot
/// files: the `repr(transparent)` + `serde(transparent)` wire
/// format is structurally identical (a bare `u64`). Existing
/// snapshots deserialize unchanged. The flip changes only the
/// in-memory trait surface, which the registry consumes through
/// `AggRule` accessors — adjusting those (or removing the
/// field's registry entry entirely) is the only edit beyond the
/// field type itself.
///
/// Defaults to zero. The reduction-trait omission is
/// deliberate: all four reductions ([`Summable::sum_across`],
/// [`Maxable::max_across`], [`Rangeable::range_across`],
/// [`Modeable::mode_across`]) on a column of structural zeros
/// trivially produce zero, but rendering that "zero" through a
/// live reduction implies "we measured zero events" when the
/// truth is "we measured a kernel-side dead pointer." Either
/// add a no-op `AggRule` variant in the migration batch, or
/// drop these fields from the registry entirely — both are the
/// migration batch's call.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct DeadCounter(pub u64);

/// Lifetime high-water mark, nanoseconds. The kernel updates
/// these as a max-against-prior in `update_stats_*` /
/// `update_se` / `set_next_entity` paths
/// (`kernel/sched/stats.c`, `kernel/sched/fair.c`); the value
/// at any procfs read is the largest single window the thread
/// has accumulated since its birth. Group reduction takes max
/// across contributors so the rendered cell surfaces the worst
/// single window any thread experienced over its lifetime.
///
/// # Cross-thread vs cross-snapshot semantics
///
/// The Max reduction over a bucket of threads produces the
/// worst single window observed across DIFFERENT tasks — task
/// A's `wait_max` and task B's `wait_max` measure two distinct
/// scheduling histories, and the bucket-level max picks
/// whichever task experienced the worst case. The result
/// belongs to that one worst task, not to the bucket as a
/// whole; downstream consumers should read the rendered cell
/// as "this bucket contained at least one task that saw N ns
/// of wait" rather than "all tasks in this bucket saw at most
/// N ns of wait" (which is the same shape, but a much weaker
/// statement).
///
/// In COMPARE mode the per-thread `PeakNs` delta between two
/// snapshots is `peak_after - peak_before` — the kernel only
/// ever raises the field, so the delta is non-negative and
/// represents the AMOUNT BY WHICH THE LIFETIME HIGH-WATER LINE
/// ROSE during the (capture-A, capture-B) interval, NOT the
/// magnitude of the worst event in that interval. A new
/// scheduling window inside the interval only moves the
/// high-water line if its own magnitude exceeds every prior
/// window the task had ever experienced; if every interval
/// event was strictly smaller than `peak_before`, the delta is
/// zero even though events did occur. The delta is therefore
/// not itself a PeakNs in the same sense as the lifetime
/// reading — it is a difference of high-water marks. The
/// bucket reduction takes max over those deltas, surfacing the
/// worst rise across contributors during the interval; this
/// can dramatically under-report transient bad windows that
/// happened earlier in any contributor's lifetime.
///
/// Summing peaks across threads is a category error — does not
/// implement [`Summable`]. Implements [`Maxable`].
///
/// Examples: `wait_max`, `sleep_max`, `block_max`, `exec_max`,
/// `slice_max`.
///
/// # u64 backing vs kernel s64
///
/// Of the `*_max` schedstat fields, only `exec_max` is typed
/// `s64` in `struct sched_statistics`
/// (`include/linux/sched.h`); `wait_max`, `sleep_max`,
/// `block_max`, and `slice_max` are `u64`. The capture pipeline
/// parses every dotted-ms.ns value via `parsed_ns_from_dotted`
/// in [`crate::host_state`], which returns `None` on negative
/// dotted values; the capture-site `unwrap_or(0)` then collapses
/// `None` to zero before the wrapper is constructed. The `u64`
/// backing here is therefore safe even for `exec_max` because
/// the parser path guarantees non-negative input — NOT because
/// every kernel-side field promises non-negative. Any new
/// writer that bypasses `parsed_ns_from_dotted` must replicate
/// its non-negative guard.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PeakNs(pub u64);

/// Instantaneous gauge sampled at capture time, nanoseconds.
/// Distinct from [`PeakNs`]: a gauge is a snapshot of the
/// CURRENT value of a kernel field, not a lifetime maximum.
/// `fair_slice_ns` reads the per-thread `slice` line from
/// `/proc/<tid>/sched`, which carries the scheduler's current
/// timeslice for the task — a point-in-time reading, not a
/// thread-lifetime accumulator. Cross-field ratios with
/// [`MonotonicNs`] / [`MonotonicCount`] / etc. produce a
/// quantity with mixed temporal interpretation (numerator
/// integrates from thread birth, denominator samples the
/// present), so callers should treat such ratios as a
/// rough hint rather than a well-defined fraction.
///
/// Group reduction takes max across contributors. Sum across
/// threads is a category error — does not implement [`Summable`].
/// Implements [`Maxable`].
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct GaugeNs(pub u64);

/// Gauge-family unitless count (u64). Distinct from
/// [`MonotonicCount`]: a [`MonotonicCount`] only ever goes UP
/// over a thread's lifetime (integrated from birth), while a
/// [`GaugeCount`] is sampled at capture time and can go up AND
/// down at runtime as the underlying state changes. Distinct
/// from [`GaugeNs`]: same Maxable-only contract, but renders as
/// a unitless count rather than a nanosecond ladder.
///
/// `nr_threads` (the process-wide thread count from
/// `signal_struct->nr_threads`) is the canonical example —
/// threads spawn and exit, so the value is not monotonic.
/// Summing thread counts across a group is meaningless (a bucket
/// of N threads sharing a tgid would over-count their parent
/// process N-fold); the registry reduces by Max so the rendered
/// cell shows "the largest process represented in this bucket."
///
/// Routing this kind of field through [`GaugeNs`] would render
/// it on the ns auto-scale ladder — a unit lie. The dedicated
/// type makes the intent explicit at the field declaration and
/// lets the format dispatch in phase 4 pick the unitless ladder
/// instead.
///
/// Implements [`Maxable`]. Does NOT implement [`Summable`].
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct GaugeCount(pub u64);

/// Bounded ordinal scalar (i32). Range-aggregated across a
/// group: the cell carries the observed `[min, max]` interval,
/// not a sum. Sum is meaningless for ordinals — adding two `nice`
/// values doesn't produce a third nice value.
///
/// Examples: `nice` ([-20, 19]), `priority`
/// (CFS=[0, 39], RT=[-2, -100], DL=-101), `processor` (last CPU
/// the task ran on; signed for symmetry with `nice` — the
/// kernel's `task_cpu()` returns `unsigned int`
/// (`include/linux/sched.h`), but ktstr stores i32 to share the
/// [`OrdinalI32`] wrapper with the genuinely-signed nice and
/// priority fields).
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct OrdinalI32(pub i32);

/// Bounded ordinal scalar (u32). Same range-aggregation contract
/// as [`OrdinalI32`] but for unsigned 32-bit fields.
///
/// Example: `rt_priority` (real-time priority, bounded 0..99 in
/// practice for SCHED_FIFO / SCHED_RR). The kernel declares
/// `unsigned int task_struct::rt_priority` at
/// `include/linux/sched.h`; emitted by procfs via
/// `seq_put_decimal_ull(m, " ", task->rt_priority)` at
/// `fs/proc/array.c:637`. A `u32` matches the kernel field width
/// exactly — narrower than the historical `u64` parse path
/// because no plausible kernel-side rt_priority value exceeds
/// `u32::MAX`.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct OrdinalU32(pub u32);

/// Bounded ordinal scalar (u64). Same range-aggregation contract
/// as [`OrdinalI32`] but for unsigned 64-bit fields. No registry
/// metric uses this width today; reserved for future ordinal
/// metrics whose kernel-side type genuinely exceeds `u32::MAX`.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct OrdinalU64(pub u64);

/// Categorical string-valued field. Group reduction takes the
/// mode (most-frequent value); ties break alphabetically per the
/// existing `aggregate(AggRule::Mode, ...)` rule.
///
/// `policy` (SCHED_OTHER, SCHED_FIFO, SCHED_RR, SCHED_BATCH,
/// SCHED_IDLE, SCHED_DEADLINE, SCHED_EXT) is the only
/// [`CategoricalString`] field on
/// [`crate::host_state::ThreadState`] after phase 2. The
/// `state: char` and `ext_enabled: bool` fields stay unwrapped
/// — the `AggRule::Mode` accessor coerces them through `String`
/// via `to_string()`/`Display` at the call site. If a second
/// bool-valued metric appears, promote both to a dedicated
/// `CategoricalBool` wrapper rather than continuing the ad-hoc
/// coercion.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CategoricalString(pub String);

/// CPU affinity set. Group reduction produces an
/// [`crate::host_state_compare::AffinitySummary`] carrying the
/// num_cpus range plus a uniform-cpuset flag.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CpuSet(pub Vec<u32>);

// ---------------------------------------------------------------------------
// `From<primitive>` and `From<newtype>` round-trip impls
// ---------------------------------------------------------------------------
//
// Each newtype gets a pair of From impls so the capture layer can
// keep parsing primitives and convert at the boundary, and so
// downstream callers reading the .0 field can pull the primitive
// back out. Note that From between *different* newtypes is
// deliberately NOT implemented — the whole point of the type
// system is to reject cross-category mixing.

macro_rules! impl_u64_newtype_from {
    ($t:ident) => {
        impl From<u64> for $t {
            fn from(v: u64) -> Self {
                Self(v)
            }
        }
        impl From<$t> for u64 {
            fn from(v: $t) -> Self {
                v.0
            }
        }
    };
}

impl_u64_newtype_from!(MonotonicCount);
impl_u64_newtype_from!(MonotonicNs);
impl_u64_newtype_from!(ClockTicks);
impl_u64_newtype_from!(Bytes);
impl_u64_newtype_from!(DeadCounter);
impl_u64_newtype_from!(PeakNs);
impl_u64_newtype_from!(GaugeNs);
impl_u64_newtype_from!(GaugeCount);

impl From<i32> for OrdinalI32 {
    fn from(v: i32) -> Self {
        Self(v)
    }
}
impl From<OrdinalI32> for i32 {
    fn from(v: OrdinalI32) -> Self {
        v.0
    }
}

impl From<u32> for OrdinalU32 {
    fn from(v: u32) -> Self {
        Self(v)
    }
}
impl From<OrdinalU32> for u32 {
    fn from(v: OrdinalU32) -> Self {
        v.0
    }
}

impl From<u64> for OrdinalU64 {
    fn from(v: u64) -> Self {
        Self(v)
    }
}
impl From<OrdinalU64> for u64 {
    fn from(v: OrdinalU64) -> Self {
        v.0
    }
}

impl From<String> for CategoricalString {
    fn from(v: String) -> Self {
        Self(v)
    }
}
impl From<CategoricalString> for String {
    fn from(v: CategoricalString) -> Self {
        v.0
    }
}
impl From<&str> for CategoricalString {
    fn from(v: &str) -> Self {
        Self(v.to_string())
    }
}

impl From<Vec<u32>> for CpuSet {
    fn from(v: Vec<u32>) -> Self {
        Self(v)
    }
}
impl From<CpuSet> for Vec<u32> {
    fn from(v: CpuSet) -> Self {
        v.0
    }
}

// ---------------------------------------------------------------------------
// `Display` — delegate to the underlying primitive so the
// auto-scale ladder (phase 4) and ad-hoc `format!("{}", ...)`
// callers can render a wrapped value as a bare integer / string
// without unwrapping `.0`. Wrappers carry semantic category, not
// formatting policy: a unit-aware render path is the format
// dispatch in phase 4, which consults the registry's `unit` tag
// rather than the wrapper type. `Display` here is the
// minimal pass-through.
// ---------------------------------------------------------------------------

macro_rules! impl_display_passthrough {
    ($t:ident) => {
        impl std::fmt::Display for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

impl_display_passthrough!(MonotonicCount);
impl_display_passthrough!(MonotonicNs);
impl_display_passthrough!(ClockTicks);
impl_display_passthrough!(Bytes);
impl_display_passthrough!(DeadCounter);
impl_display_passthrough!(PeakNs);
impl_display_passthrough!(GaugeNs);
impl_display_passthrough!(GaugeCount);
impl_display_passthrough!(OrdinalI32);
impl_display_passthrough!(OrdinalU32);
impl_display_passthrough!(OrdinalU64);
impl_display_passthrough!(CategoricalString);

// CpuSet has no canonical Display — the rendered form depends
// on whether the call site wants `format_cpu_range` ("0-3,5"
// collapsed runs) or a verbatim debug list. Callers reach for
// `cpuset.0` and feed it through the appropriate renderer.

// ---------------------------------------------------------------------------
// Marker traits + reductions
// ---------------------------------------------------------------------------

/// Private sealing module: the supertraits live here so a
/// downstream crate cannot bypass the category invariant by
/// writing `impl Summable for PeakNs`. Adding a new Summable
/// (or Maxable, Modeable, Rangeable) requires editing this
/// module — the choke point the type system creates.
mod sealed {
    /// Sealed supertrait of [`super::Summable`].
    pub trait SummableSealed {}
    /// Sealed supertrait of [`super::Maxable`].
    pub trait MaxableSealed {}
    /// Sealed supertrait of [`super::Modeable`].
    pub trait ModeableSealed {}
    /// Sealed supertrait of [`super::Rangeable`].
    pub trait RangeableSealed {}
}

/// Marker for newtypes that can be summed across a group.
///
/// Implemented by [`MonotonicCount`], [`MonotonicNs`],
/// [`ClockTicks`], and [`Bytes`] — every newtype whose value is
/// a thread-lifetime accumulator. Summing two such accumulators
/// across a group is well-defined because both contributors
/// carry the same temporal window (each thread's lifetime),
/// and the group total represents the same window union.
///
/// Deliberately NOT implemented by [`PeakNs`] / [`GaugeNs`] /
/// [`GaugeCount`] / [`OrdinalI32`] / [`OrdinalU32`] /
/// [`OrdinalU64`] / [`CategoricalString`] / [`CpuSet`] — those
/// reductions are category errors and a generic site bound on
/// `T: Summable` will reject them at compile time.
///
/// Sealed via [`sealed::SummableSealed`]: a downstream crate
/// cannot write `impl Summable for PeakNs` because the sealed
/// supertrait is private to this module.
///
/// `sum_across` uses `saturating_add` to mirror the existing
/// [`crate::host_state_compare::aggregate`] contract: per-thread
/// counters are non-negative u64s, the group total cannot exceed
/// `u64::MAX`, and a hostile or corrupt reading that would push
/// the sum past `u64::MAX` saturates rather than wrapping.
///
/// `sum_across` collapses an empty iterator to the additive
/// identity (zero, via `Self::default()` shape — the four
/// counter newtypes default to `Self(0)`). Callers that need
/// to distinguish "no contributors" from "all contributors had
/// zero" — for example, to suppress a derived ratio whose
/// denominator bucket was empty rather than zero-valued — use
/// [`try_sum_across`](Self::try_sum_across), which returns
/// `None` for an empty iterator and `Some(total)` otherwise.
/// The two methods report the same value on every non-empty
/// input. The `try_` prefix (rather than `checked_`) avoids
/// colliding with the stdlib's `checked_*` numeric methods,
/// which detect arithmetic overflow — this method only flags
/// an empty iterator (saturation happens unconditionally in
/// `sum_across`).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not Summable — summing it would conflate semantic categories \
               or temporal windows",
    label = "this metric type cannot be summed across a group",
    note = "PeakNs (lifetime high-water): use Maxable::max_across; \
            GaugeNs/GaugeCount (instantaneous samples — different temporal window than the \
            lifetime accumulators): use Maxable::max_across; \
            OrdinalI32/OrdinalU32/OrdinalU64 (bounded scalars): use Rangeable::range_across; \
            CategoricalString: use Modeable::mode_across; CpuSet: use the \
            AffinitySummary reduction in host_state_compare; \
            DeadCounter: kernel-side dead pointer — value is structurally zero; the \
            registry must use a no-op aggregation arm (or omit the field) rather than \
            sum across structural zeros"
)]
pub trait Summable: sealed::SummableSealed + Sized + Copy {
    /// Sum across the iterator, saturating at `u64::MAX`.
    /// Empty input collapses to the additive identity (zero).
    fn sum_across(items: impl IntoIterator<Item = Self>) -> Self;

    /// Same total as [`sum_across`](Self::sum_across) on every
    /// non-empty input; returns `None` for an empty iterator so
    /// callers can distinguish "no contributors" from "all
    /// contributors summed to zero." Useful when a downstream
    /// derived metric (e.g. a ratio) needs to suppress the
    /// row entirely rather than render `0 / 0`.
    ///
    /// The `try_` prefix (rather than `checked_`) avoids
    /// colliding with the stdlib's `checked_*` numeric methods,
    /// which detect arithmetic overflow. This method only
    /// flags an empty iterator — overflow handling is identical
    /// to `sum_across` (saturating, unconditional).
    fn try_sum_across(items: impl IntoIterator<Item = Self>) -> Option<Self> {
        let mut it = items.into_iter();
        // Relies on Self: Copy (Summable trait bound) so the
        // next-and-chain pattern works without duplicating the
        // first element — `it.next()?` consumes the first item
        // for the empty check, and `iter::once(first)` re-emits
        // the same value into the chain that feeds sum_across.
        let first = it.next()?;
        Some(Self::sum_across(std::iter::once(first).chain(it)))
    }
}

/// Marker for newtypes that can be reduced by max across a
/// group.
///
/// Implemented by [`PeakNs`] (max-of-peak is the worst
/// high-water mark any contributor saw across its lifetime),
/// [`GaugeNs`] (max-of-gauge is the longest current value in
/// the bucket — distinct temporal window: each gauge is a
/// fresh sample at capture time, not a lifetime accumulator),
/// and [`GaugeCount`] (max-of-count is the biggest current
/// value in the bucket — same gauge-window caveat).
///
/// Deliberately NOT implemented by [`Summable`] cumulative
/// counters ([`MonotonicCount`] / [`MonotonicNs`] /
/// [`ClockTicks`] / [`Bytes`]): max-across-snapshots on a
/// thread-lifetime accumulator reduces to "the value of the
/// last snapshot," because each snapshot's reading dominates
/// every prior reading by construction (the kernel only ever
/// raises a lifetime counter). That gives a reduction whose
/// "maximum" is the most-recent reading rather than a worst
/// single window — useful as a sanity bound, but rendering it
/// alongside per-thread peaks invites confusion. If a future
/// metric truly needs the lifetime-integrated max of a
/// cumulative counter, introduce a dedicated peak-of-counter
/// newtype rather than re-adding `Maxable` to a Summable type.
/// Deliberately NOT implemented by ordinals (those carry a
/// `[min, max]` range, not a single max), nor by
/// [`CategoricalString`] (string max has no useful semantic),
/// nor by [`CpuSet`] (the affinity reduction is a custom
/// summary, not a bare max).
///
/// Sealed via [`sealed::MaxableSealed`]: a downstream crate
/// cannot write `impl Maxable for CategoricalString` because the
/// sealed supertrait is private to this module.
///
/// `max_across` returns `Option<Self>`: `None` for an empty
/// iterator (so callers can distinguish "no contributors" from
/// "max was zero — the worst reading any contributor reported
/// happened to be the additive identity"), `Some(largest)`
/// otherwise. Aggregation callers that want to preserve the
/// pre-Option contract collapse `None` to the type's
/// `default()` value at the call site.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not Maxable — `max` is undefined for this category",
    label = "this metric type does not support max-across",
    note = "MonotonicCount/MonotonicNs/ClockTicks/Bytes (Summable cumulative counters): \
            use Summable::sum_across; max-across-snapshots on a lifetime accumulator \
            reduces to the most-recent reading, not a worst window; \
            OrdinalI32/OrdinalU32/OrdinalU64: use Rangeable::range_across; \
            CategoricalString: use Modeable::mode_across; CpuSet: use the \
            AffinitySummary reduction in host_state_compare; \
            DeadCounter: kernel-side dead pointer — value is structurally zero; the \
            registry must use a no-op aggregation arm (or omit the field) rather than \
            max across structural zeros"
)]
pub trait Maxable: sealed::MaxableSealed + Sized + Copy + Ord {
    fn max_across(items: impl IntoIterator<Item = Self>) -> Option<Self>;
}

/// Marker for newtypes reduced by mode (most-frequent value).
/// Implemented by [`CategoricalString`].
///
/// `mode_across` returns `None` when the input iterator is
/// empty. Ties break by ascending sort order on the value type
/// to match the existing
/// [`crate::host_state_compare::aggregate`]
/// [`crate::host_state_compare::AggRule::Mode`] contract:
/// "lexicographically smaller wins" for equal-frequency strings.
///
/// Sealed via [`sealed::ModeableSealed`]: a downstream crate
/// cannot write `impl Modeable for u64` because the sealed
/// supertrait is private to this module.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not Modeable — Modeable is reserved for CategoricalString in this codebase",
    label = "this metric type does not support mode-across",
    note = "CategoricalString is the only Modeable type today. Numeric types use Summable / \
            Maxable / Rangeable depending on category. If a new categorical newtype \
            needs mode-aggregation, add the impl in metric_types.rs."
)]
pub trait Modeable: sealed::ModeableSealed + Sized + Clone + Eq + Ord {
    /// Returns `(mode_value, count, total)` over the input
    /// iterator, or `None` when the iterator is empty.
    fn mode_across(items: impl IntoIterator<Item = Self>) -> Option<(Self, usize, usize)> {
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<Self, usize> = BTreeMap::new();
        let mut total = 0usize;
        for item in items {
            *counts.entry(item).or_default() += 1;
            total += 1;
        }
        if total == 0 {
            return None;
        }
        // BTreeMap iterates in ascending key order, so the
        // sequence of (key, count) pairs walks the candidate
        // values lex-ascending. The closure `a.1.cmp(&b.1)
        // .then(b.0.cmp(&a.0))` ranks first by count (higher
        // wins), then by key (smaller wins). Each key appears
        // at most once in the map by construction (BTreeMap
        // dedups on key), so the closure never compares two
        // entries with identical primary AND secondary keys —
        // meaning the std-library "max_by keeps the LAST equally-
        // maximum element" tiebreak is unreachable here. The
        // closure is a strict total order over the unique keys
        // and produces the lex-smallest mode at the highest
        // count.
        let (value, count) = counts
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
            .expect("non-empty inputs produce a non-empty count map");
        Some((value, count, total))
    }
}

/// Inclusive `[min, max]` interval over a [`Rangeable`] type.
///
/// The constructor enforces `min ≤ max`, so a `Range<T>` value
/// in hand is a proof that the contained pair is well-ordered;
/// downstream consumers can read [`min`](Self::min) /
/// [`max`](Self::max) without re-checking. The invariant is
/// checked at runtime in debug builds via `debug_assert!`.
///
/// Construction sites in this crate (the [`Rangeable::range_across`]
/// reduction) walk the input iterator and produce a `Range`
/// directly; misuse — calling `Range::new(b, a)` with `a < b` —
/// is a programmer error and panics in debug, sneaks through in
/// release (the wrapped pair is then `[max, min]`, so any caller
/// reading [`min`](Self::min) gets the larger value). External
/// callers constructing a `Range` from external bounds should
/// pre-sort.
///
/// `Range<T>` deliberately omits `Ord` / `PartialOrd` /
/// `Serialize` / `Deserialize`:
/// - It is an in-memory aggregation result, not a wire-format
///   boundary; the `aggregate()` dispatch destructures `Range`
///   into the existing
///   [`crate::host_state_compare::Aggregated::OrdinalRange`]
///   variant (which carries `min: i64, max: i64`), so the typed
///   invariant is enforced at the reduction boundary and the
///   untyped tuple shape continues to cross every serialized
///   boundary downstream.
/// - Comparing two `Range` values to each other has no defined
///   semantic — there is no obvious ordering on intervals — and
///   adding `derive(Ord)` would bring [`std::cmp::Ord::min`] /
///   [`std::cmp::Ord::max`] into scope on `Range<T>` and shadow
///   the inherent accessors at every call site.
///
/// **Heads-up for future contributors**: if Ord/PartialOrd
/// derives are ever added, expect breakage at every existing
/// `.min()` / `.max()` call site — those resolve to the
/// inherent methods today and will start resolving to the trait
/// methods (different signature, different return type) the
/// moment the derives are added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Range<T: PartialOrd> {
    min: T,
    max: T,
}

impl<T: PartialOrd> Range<T> {
    /// Construct a `Range` from a `(min, max)` pair.
    ///
    /// `debug_assert!`s that `min ≤ max` — the [`Rangeable`]
    /// reduction guarantees this by walking the input and
    /// tracking min and max separately, so the assertion never
    /// fires on internal call sites. External callers must
    /// pre-sort.
    pub fn new(min: T, max: T) -> Self {
        debug_assert!(
            min.partial_cmp(&max) != Some(std::cmp::Ordering::Greater),
            "Range::new requires min <= max — got a min that compares strictly greater"
        );
        Self { min, max }
    }

    /// The lower bound of the interval.
    pub fn min(&self) -> &T {
        &self.min
    }

    /// The upper bound of the interval.
    pub fn max(&self) -> &T {
        &self.max
    }

    /// Consume the range and return the `(min, max)` tuple.
    /// Useful at boundaries where the caller has its own
    /// pair-shaped representation (e.g. the
    /// [`crate::host_state_compare::Aggregated::OrdinalRange`]
    /// variant).
    pub fn into_tuple(self) -> (T, T) {
        (self.min, self.max)
    }
}

/// Marker for newtypes reduced by `[min, max]` range.
/// Implemented by [`OrdinalI32`], [`OrdinalU32`], and
/// [`OrdinalU64`].
///
/// `range_across` returns `Option<Range<Self>>` — `None` for
/// an empty iterator, `Some(Range)` otherwise. The wrapped
/// `Range` value carries `min ≤ max` as a type-system invariant
/// so downstream consumers (the format dispatch, derived
/// metrics, the `Aggregated::OrdinalRange` boundary) cannot
/// observe a swapped pair. The reduction tracks min and max
/// separately while walking the input, so the constructor
/// invariant is satisfied by construction.
///
/// Sealed via [`sealed::RangeableSealed`]: a downstream crate
/// cannot write `impl Rangeable for u64` because the sealed
/// supertrait is private to this module.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not Rangeable — Rangeable is reserved for bounded ordinals in this codebase",
    label = "this metric type does not support range-across",
    note = "Counters/peaks/gauges: use Summable::sum_across or Maxable::max_across; \
            CategoricalString: use Modeable::mode_across; CpuSet: use the AffinitySummary \
            reduction in host_state_compare"
)]
pub trait Rangeable: sealed::RangeableSealed + Sized + Copy + Ord {
    fn range_across(items: impl IntoIterator<Item = Self>) -> Option<Range<Self>> {
        let mut it = items.into_iter();
        let first = it.next()?;
        let mut min = first;
        let mut max = first;
        for v in it {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        Some(Range::new(min, max))
    }
}

// Macro for the four cumulative-counter shapes (Summable only,
// NOT Maxable — see the Maxable trait doc for the
// last-snapshot-dominates-everything rationale). The sealed
// supertrait impl gates Summable so external crates can't extend
// the trait list outside this module.
//
// "only" in `impl_summable_only_u64` = Summable-only (i.e. NOT
// also Maxable, the natural sibling) — see the Maxable trait
// doc for why. Renaming to `impl_summable_not_maxable_u64`
// would be more explicit but verbose; the macro body below
// shows the trait surface in 3 lines.
macro_rules! impl_summable_only_u64 {
    ($t:ident) => {
        impl sealed::SummableSealed for $t {}
        impl Summable for $t {
            fn sum_across(items: impl IntoIterator<Item = Self>) -> Self {
                let mut total: u64 = 0;
                for v in items {
                    total = total.saturating_add(v.0);
                }
                Self(total)
            }
        }
    };
}

impl_summable_only_u64!(MonotonicCount);
impl_summable_only_u64!(MonotonicNs);
impl_summable_only_u64!(ClockTicks);
impl_summable_only_u64!(Bytes);

// Peak / Gauge are Maxable, NOT Summable. `max_across` walks
// the input and returns Option<Self> so the empty-iterator case
// is distinguishable from "all contributors had zero".
macro_rules! impl_maxable_only_u64 {
    ($t:ident) => {
        impl sealed::MaxableSealed for $t {}
        impl Maxable for $t {
            fn max_across(items: impl IntoIterator<Item = Self>) -> Option<Self> {
                let mut it = items.into_iter();
                let first = it.next()?;
                let mut out = first;
                for v in it {
                    if v > out {
                        out = v;
                    }
                }
                Some(out)
            }
        }
    };
}

impl_maxable_only_u64!(PeakNs);
impl_maxable_only_u64!(GaugeNs);
impl_maxable_only_u64!(GaugeCount);

impl sealed::RangeableSealed for OrdinalI32 {}
impl sealed::RangeableSealed for OrdinalU32 {}
impl sealed::RangeableSealed for OrdinalU64 {}
impl Rangeable for OrdinalI32 {}
impl Rangeable for OrdinalU32 {}
impl Rangeable for OrdinalU64 {}

impl sealed::ModeableSealed for CategoricalString {}
impl Modeable for CategoricalString {}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Round-trip From impls -----------------------------------------------

    #[test]
    fn monotonic_count_from_u64_roundtrips() {
        let v: MonotonicCount = 42u64.into();
        assert_eq!(v.0, 42);
        let back: u64 = v.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn monotonic_ns_from_u64_roundtrips() {
        let v: MonotonicNs = 1_000_000u64.into();
        assert_eq!(v.0, 1_000_000);
        let back: u64 = v.into();
        assert_eq!(back, 1_000_000);
    }

    #[test]
    fn clock_ticks_from_u64_roundtrips() {
        let v: ClockTicks = 1234u64.into();
        assert_eq!(v.0, 1234);
        let back: u64 = v.into();
        assert_eq!(back, 1234);
    }

    #[test]
    fn bytes_from_u64_roundtrips() {
        let v: Bytes = (1024 * 1024).into();
        assert_eq!(v.0, 1024 * 1024);
        let back: u64 = v.into();
        assert_eq!(back, 1024 * 1024);
    }

    #[test]
    fn peak_ns_from_u64_roundtrips() {
        let v: PeakNs = 999u64.into();
        assert_eq!(v.0, 999);
        let back: u64 = v.into();
        assert_eq!(back, 999);
    }

    #[test]
    fn gauge_ns_from_u64_roundtrips() {
        let v: GaugeNs = 7u64.into();
        assert_eq!(v.0, 7);
        let back: u64 = v.into();
        assert_eq!(back, 7);
    }

    #[test]
    fn ordinal_i32_from_i32_roundtrips() {
        let v: OrdinalI32 = (-20).into();
        assert_eq!(v.0, -20);
        let back: i32 = v.into();
        assert_eq!(back, -20);
    }

    #[test]
    fn gauge_count_from_u64_roundtrips() {
        let v: GaugeCount = 16u64.into();
        assert_eq!(v.0, 16);
        let back: u64 = v.into();
        assert_eq!(back, 16);
    }

    #[test]
    fn ordinal_u32_from_u32_roundtrips() {
        let v: OrdinalU32 = 99u32.into();
        assert_eq!(v.0, 99);
        let back: u32 = v.into();
        assert_eq!(back, 99);
    }

    #[test]
    fn ordinal_u64_from_u64_roundtrips() {
        let v: OrdinalU64 = 99u64.into();
        assert_eq!(v.0, 99);
        let back: u64 = v.into();
        assert_eq!(back, 99);
    }

    #[test]
    fn categorical_string_from_string_roundtrips() {
        let v: CategoricalString = "SCHED_FIFO".to_string().into();
        assert_eq!(v.0, "SCHED_FIFO");
        let back: String = v.into();
        assert_eq!(back, "SCHED_FIFO");
    }

    #[test]
    fn categorical_string_from_str_works() {
        let v: CategoricalString = "SCHED_OTHER".into();
        assert_eq!(v.0, "SCHED_OTHER");
    }

    #[test]
    fn cpuset_from_vec_roundtrips() {
        let cpus = vec![0u32, 1, 2, 3];
        let v: CpuSet = cpus.clone().into();
        assert_eq!(v.0, cpus);
        let back: Vec<u32> = v.into();
        assert_eq!(back, cpus);
    }

    // -- Summable -------------------------------------------------------------

    #[test]
    fn summable_monotonic_count_sums_to_total() {
        let xs = [MonotonicCount(10), MonotonicCount(20), MonotonicCount(30)];
        let s = MonotonicCount::sum_across(xs);
        assert_eq!(s, MonotonicCount(60));
    }

    #[test]
    fn summable_monotonic_ns_saturates_on_overflow() {
        let xs = [MonotonicNs(u64::MAX), MonotonicNs(5)];
        let s = MonotonicNs::sum_across(xs);
        assert_eq!(s, MonotonicNs(u64::MAX));
    }

    #[test]
    fn summable_clock_ticks_sums() {
        let xs = [ClockTicks(100), ClockTicks(50)];
        let s = ClockTicks::sum_across(xs);
        assert_eq!(s, ClockTicks(150));
    }

    #[test]
    fn summable_bytes_sums() {
        let xs = [Bytes(1024), Bytes(2048), Bytes(4096)];
        let s = Bytes::sum_across(xs);
        assert_eq!(s, Bytes(7168));
    }

    #[test]
    fn summable_empty_iterator_returns_zero() {
        let s = MonotonicCount::sum_across(std::iter::empty());
        assert_eq!(s, MonotonicCount(0));
    }

    /// Compile-time gate: counters implement Summable; PeakNs /
    /// GaugeNs / GaugeCount / ordinals / categoricals do NOT.
    /// The static `assert_summable<T>()` helper compiles only
    /// when the type satisfies `T: Summable`, so this test pins
    /// the four counter newtypes by exercising the bound. The
    /// negative assertion — that `assert_summable::<PeakNs>()`,
    /// `assert_summable::<GaugeNs>()`,
    /// `assert_summable::<GaugeCount>()` etc. fail to compile —
    /// is enforced by the [`sealed::SummableSealed`] supertrait
    /// and the omission of those `impl SummableSealed` lines.
    /// Adding any one would require an explicit edit to this
    /// module's `impl_summable_only_u64!` invocations.
    #[test]
    fn summable_only_implemented_for_counters() {
        fn assert_summable<T: Summable>() {}
        assert_summable::<MonotonicCount>();
        assert_summable::<MonotonicNs>();
        assert_summable::<ClockTicks>();
        assert_summable::<Bytes>();
    }

    #[test]
    fn try_sum_across_empty_returns_none() {
        let s = MonotonicCount::try_sum_across(std::iter::empty());
        assert!(s.is_none());
    }

    #[test]
    fn try_sum_across_non_empty_matches_sum_across() {
        let xs = [MonotonicCount(10), MonotonicCount(20), MonotonicCount(30)];
        let unchecked = MonotonicCount::sum_across(xs);
        let tried = MonotonicCount::try_sum_across(xs).expect("non-empty");
        assert_eq!(unchecked, tried);
        assert_eq!(tried, MonotonicCount(60));
    }

    #[test]
    fn try_sum_across_saturates_on_overflow() {
        let xs = [MonotonicNs(u64::MAX), MonotonicNs(5)];
        let s = MonotonicNs::try_sum_across(xs).expect("non-empty");
        assert_eq!(s, MonotonicNs(u64::MAX));
    }

    /// Singleton input still produces `Some(value)` — proves
    /// `try_sum_across` does not "consume the first element to
    /// test for emptiness" in a way that would lose data.
    #[test]
    fn try_sum_across_singleton_returns_that_value() {
        let s = MonotonicCount::try_sum_across([MonotonicCount(42)]).expect("non-empty");
        assert_eq!(s, MonotonicCount(42));
    }

    /// Compile-time gate: `try_sum_across` is part of the
    /// `Summable` trait surface, so every Summable type carries
    /// the empty-aware variant for free.
    #[test]
    fn try_sum_across_available_on_every_summable() {
        fn assert_try_sum<T: Summable>() {
            let _ = T::try_sum_across(std::iter::empty());
        }
        assert_try_sum::<MonotonicCount>();
        assert_try_sum::<MonotonicNs>();
        assert_try_sum::<ClockTicks>();
        assert_try_sum::<Bytes>();
    }

    // -- Maxable --------------------------------------------------------------

    #[test]
    fn maxable_peak_ns_picks_largest() {
        let xs = [PeakNs(100), PeakNs(500), PeakNs(200)];
        let m = PeakNs::max_across(xs).expect("non-empty");
        assert_eq!(m, PeakNs(500));
    }

    #[test]
    fn maxable_gauge_ns_picks_largest() {
        let xs = [GaugeNs(7), GaugeNs(99), GaugeNs(50)];
        let m = GaugeNs::max_across(xs).expect("non-empty");
        assert_eq!(m, GaugeNs(99));
    }

    #[test]
    fn maxable_gauge_count_picks_largest() {
        let xs = [GaugeCount(3), GaugeCount(11), GaugeCount(7)];
        let m = GaugeCount::max_across(xs).expect("non-empty");
        assert_eq!(m, GaugeCount(11));
    }

    #[test]
    fn maxable_empty_iterator_returns_none() {
        let m = PeakNs::max_across(std::iter::empty());
        assert!(m.is_none());
    }

    #[test]
    fn maxable_singleton_returns_that_value() {
        let m = PeakNs::max_across([PeakNs(42)]).expect("non-empty");
        assert_eq!(m, PeakNs(42));
    }

    /// Singleton with the additive-identity value still produces
    /// `Some(zero)` rather than `None` — pins the contract that
    /// `None` exclusively signals "empty input," not "max happens
    /// to be zero."
    #[test]
    fn maxable_singleton_zero_returns_some_zero() {
        let m = PeakNs::max_across([PeakNs(0)]).expect("non-empty");
        assert_eq!(m, PeakNs(0));
    }

    /// Compile-time gate: Maxable is implemented by exactly the
    /// peak / gauge family — `PeakNs`, `GaugeNs`, `GaugeCount`.
    /// The four Summable cumulative counter newtypes are
    /// deliberately NOT Maxable: a static `assert_maxable<T>()`
    /// helper would refuse to compile against `MonotonicCount` /
    /// `MonotonicNs` / `ClockTicks` / `Bytes`. The
    /// `compile_fail_*` tests under `tests/compile_fail/` pin the
    /// negative side empirically; this test pins the positive
    /// side.
    #[test]
    fn maxable_implemented_for_peaks_and_gauges() {
        fn assert_maxable<T: Maxable>() {}
        assert_maxable::<PeakNs>();
        assert_maxable::<GaugeNs>();
        assert_maxable::<GaugeCount>();
    }

    // -- Rangeable ------------------------------------------------------------

    #[test]
    fn rangeable_ordinal_i32_finds_min_max() {
        let xs = [
            OrdinalI32(-5),
            OrdinalI32(10),
            OrdinalI32(0),
            OrdinalI32(-20),
        ];
        let r = OrdinalI32::range_across(xs).expect("non-empty");
        assert_eq!(*r.min(), OrdinalI32(-20));
        assert_eq!(*r.max(), OrdinalI32(10));
    }

    #[test]
    fn rangeable_ordinal_u32_finds_min_max() {
        let xs = [OrdinalU32(7), OrdinalU32(3), OrdinalU32(15)];
        let r = OrdinalU32::range_across(xs).expect("non-empty");
        assert_eq!(*r.min(), OrdinalU32(3));
        assert_eq!(*r.max(), OrdinalU32(15));
    }

    #[test]
    fn rangeable_ordinal_u64_finds_min_max() {
        let xs = [
            OrdinalU64(50),
            OrdinalU64(99),
            OrdinalU64(0),
            OrdinalU64(25),
        ];
        let r = OrdinalU64::range_across(xs).expect("non-empty");
        assert_eq!(*r.min(), OrdinalU64(0));
        assert_eq!(*r.max(), OrdinalU64(99));
    }

    #[test]
    fn rangeable_singleton_min_eq_max() {
        let r = OrdinalI32::range_across([OrdinalI32(42)]).expect("non-empty");
        assert_eq!(*r.min(), OrdinalI32(42));
        assert_eq!(*r.max(), OrdinalI32(42));
    }

    #[test]
    fn rangeable_empty_iterator_returns_none() {
        let r = OrdinalI32::range_across(std::iter::empty());
        assert!(r.is_none());
    }

    /// `Range::new` enforces `min ≤ max` via `debug_assert!`, so
    /// the type-system invariant matches the runtime check in
    /// debug builds. Pins the constructor's debug-build behavior.
    #[test]
    #[should_panic(expected = "min <= max")]
    fn range_new_debug_asserts_min_le_max_when_swapped() {
        let _ = Range::new(OrdinalI32(10), OrdinalI32(5));
    }

    #[test]
    fn range_new_min_eq_max_is_allowed() {
        let r = Range::new(OrdinalI32(42), OrdinalI32(42));
        assert_eq!(*r.min(), OrdinalI32(42));
        assert_eq!(*r.max(), OrdinalI32(42));
    }

    #[test]
    fn range_into_tuple_preserves_pair() {
        let r = Range::new(OrdinalU32(3), OrdinalU32(15));
        let (min, max) = r.into_tuple();
        assert_eq!(min, OrdinalU32(3));
        assert_eq!(max, OrdinalU32(15));
    }

    /// `range_across` always satisfies `min ≤ max` because it
    /// tracks `min` and `max` separately while walking the input
    /// — the constructor's `debug_assert!` never fires on the
    /// reduction path. Pin this by exercising a worst-case
    /// reverse-sorted input.
    #[test]
    fn range_across_preserves_min_le_max_on_reversed_input() {
        let xs = [
            OrdinalI32(99),
            OrdinalI32(10),
            OrdinalI32(0),
            OrdinalI32(-20),
        ];
        let r = OrdinalI32::range_across(xs).expect("non-empty");
        assert!(r.min() <= r.max());
        assert_eq!(*r.min(), OrdinalI32(-20));
        assert_eq!(*r.max(), OrdinalI32(99));
    }

    // -- DeadCounter ----------------------------------------------------------

    #[test]
    fn dead_counter_from_u64_roundtrips() {
        let v: DeadCounter = 0u64.into();
        assert_eq!(v.0, 0);
        let back: u64 = v.into();
        assert_eq!(back, 0);
    }

    /// Wire format for [`DeadCounter`] matches a bare `u64`,
    /// identical to [`MonotonicCount`]. The type-system
    /// difference is in the trait list (no Summable / Maxable /
    /// Rangeable / Modeable impl), not in the wire bytes.
    #[test]
    fn dead_counter_serde_transparent() {
        let v = DeadCounter(0);
        let json = serde_json::to_string(&v).expect("serialize");
        assert_eq!(json, "0");
        let back: DeadCounter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(v, back);

        // Non-zero round-trip even though the kernel write path is
        // dead — the wire format must be identical to MonotonicCount
        // so the future migration can flip a field's wrapper without
        // regenerating snapshot files.
        let nonzero = DeadCounter(42);
        let nonzero_json = serde_json::to_string(&nonzero).expect("serialize");
        assert_eq!(nonzero_json, "42");
        let nonzero_back: DeadCounter = serde_json::from_str(&nonzero_json).expect("deserialize");
        assert_eq!(nonzero, nonzero_back);
    }

    #[test]
    fn dead_counter_default_is_zero() {
        assert_eq!(DeadCounter::default(), DeadCounter(0));
    }

    #[test]
    fn dead_counter_repr_transparent_size() {
        use std::mem::size_of;
        assert_eq!(size_of::<DeadCounter>(), size_of::<u64>());
    }

    #[test]
    fn dead_counter_display_passthrough() {
        assert_eq!(format!("{}", DeadCounter(0)), "0");
        assert_eq!(format!("{}", DeadCounter(42)), "42");
    }

    // -- Modeable -------------------------------------------------------------

    #[test]
    fn modeable_categorical_string_picks_most_frequent() {
        let xs = [
            CategoricalString::from("SCHED_OTHER"),
            CategoricalString::from("SCHED_OTHER"),
            CategoricalString::from("SCHED_FIFO"),
        ];
        let (value, count, total) = CategoricalString::mode_across(xs).expect("non-empty");
        assert_eq!(value, CategoricalString::from("SCHED_OTHER"));
        assert_eq!(count, 2);
        assert_eq!(total, 3);
    }

    /// Tie-break: equal counts → lex-smallest wins. Mirrors the
    /// host_state_compare::aggregate(AggRule::Mode) rule (see
    /// `mode_rule_tie_break_is_lexicographic` over there).
    #[test]
    fn modeable_tie_break_is_lex_smallest() {
        let xs = [
            CategoricalString::from("SCHED_OTHER"),
            CategoricalString::from("SCHED_FIFO"),
        ];
        let (value, count, total) = CategoricalString::mode_across(xs).expect("non-empty");
        assert_eq!(value, CategoricalString::from("SCHED_FIFO"));
        assert_eq!(count, 1);
        assert_eq!(total, 2);
    }

    #[test]
    fn modeable_empty_iterator_returns_none() {
        let r = CategoricalString::mode_across(std::iter::empty());
        assert!(r.is_none());
    }

    #[test]
    fn modeable_unanimous_returns_total() {
        let xs = [
            CategoricalString::from("R"),
            CategoricalString::from("R"),
            CategoricalString::from("R"),
        ];
        let (value, count, total) = CategoricalString::mode_across(xs).expect("non-empty");
        assert_eq!(value, CategoricalString::from("R"));
        assert_eq!(count, 3);
        assert_eq!(total, 3);
    }

    // -- repr(transparent) wire compatibility --------------------------------

    /// Serde transparent: every newtype must serialize identically
    /// to its primitive. Pin the JSON shape so the
    /// `ThreadState` migration (phase 2) preserves existing
    /// snapshot files.
    #[test]
    fn serde_transparent_matches_primitive() {
        let raw_count = MonotonicCount(123);
        let raw_count_json = serde_json::to_string(&raw_count).expect("serialize");
        assert_eq!(raw_count_json, "123");

        let raw_ns = MonotonicNs(456_789);
        let raw_ns_json = serde_json::to_string(&raw_ns).expect("serialize");
        assert_eq!(raw_ns_json, "456789");

        let raw_ticks = ClockTicks(2048);
        let raw_ticks_json = serde_json::to_string(&raw_ticks).expect("serialize");
        assert_eq!(raw_ticks_json, "2048");

        let raw_bytes = Bytes(1024 * 1024);
        let raw_bytes_json = serde_json::to_string(&raw_bytes).expect("serialize");
        assert_eq!(raw_bytes_json, "1048576");

        let raw_dead = DeadCounter(7);
        let raw_dead_json = serde_json::to_string(&raw_dead).expect("serialize");
        assert_eq!(raw_dead_json, "7");

        let raw_peak = PeakNs(99);
        let raw_peak_json = serde_json::to_string(&raw_peak).expect("serialize");
        assert_eq!(raw_peak_json, "99");

        let raw_gauge = GaugeNs(7_500_000);
        let raw_gauge_json = serde_json::to_string(&raw_gauge).expect("serialize");
        assert_eq!(raw_gauge_json, "7500000");

        let raw_gauge_count = GaugeCount(16);
        let raw_gauge_count_json = serde_json::to_string(&raw_gauge_count).expect("serialize");
        assert_eq!(raw_gauge_count_json, "16");

        let raw_ordi = OrdinalI32(-5);
        let raw_ordi_json = serde_json::to_string(&raw_ordi).expect("serialize");
        assert_eq!(raw_ordi_json, "-5");

        let raw_ordu32 = OrdinalU32(99);
        let raw_ordu32_json = serde_json::to_string(&raw_ordu32).expect("serialize");
        assert_eq!(raw_ordu32_json, "99");

        let raw_ordu64 = OrdinalU64(99);
        let raw_ordu64_json = serde_json::to_string(&raw_ordu64).expect("serialize");
        assert_eq!(raw_ordu64_json, "99");

        let raw_str = CategoricalString::from("R");
        let raw_str_json = serde_json::to_string(&raw_str).expect("serialize");
        assert_eq!(raw_str_json, "\"R\"");

        let raw_cpus = CpuSet(vec![0, 2, 4]);
        let raw_cpus_json = serde_json::to_string(&raw_cpus).expect("serialize");
        assert_eq!(raw_cpus_json, "[0,2,4]");
    }

    /// Round-trip via JSON: serialize then deserialize must
    /// produce an equal value. Defends against an asymmetric
    /// transparent attribute (e.g. only on Serialize) that
    /// would silently produce different wire formats.
    #[test]
    fn serde_round_trip_through_json() {
        let v = MonotonicNs(987_654_321);
        let json = serde_json::to_string(&v).expect("serialize");
        let back: MonotonicNs = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(v, back);

        let s = CategoricalString::from("SCHED_DEADLINE");
        let json = serde_json::to_string(&s).expect("serialize");
        let back: CategoricalString = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s, back);
    }

    /// repr(transparent) means each u64-backed newtype occupies
    /// exactly one u64 of memory. Pin this so a future derive
    /// addition that displaced the layout (e.g. forgetting
    /// repr(transparent) when adding fields) fails the test.
    #[test]
    fn repr_transparent_matches_primitive_size() {
        use std::mem::size_of;
        assert_eq!(size_of::<MonotonicCount>(), size_of::<u64>());
        assert_eq!(size_of::<MonotonicNs>(), size_of::<u64>());
        assert_eq!(size_of::<ClockTicks>(), size_of::<u64>());
        assert_eq!(size_of::<Bytes>(), size_of::<u64>());
        assert_eq!(size_of::<DeadCounter>(), size_of::<u64>());
        assert_eq!(size_of::<PeakNs>(), size_of::<u64>());
        assert_eq!(size_of::<GaugeNs>(), size_of::<u64>());
        assert_eq!(size_of::<GaugeCount>(), size_of::<u64>());
        assert_eq!(size_of::<OrdinalI32>(), size_of::<i32>());
        assert_eq!(size_of::<OrdinalU32>(), size_of::<u32>());
        assert_eq!(size_of::<OrdinalU64>(), size_of::<u64>());
    }

    /// Default values are zero / empty — pin so a future change
    /// that shifts the default (e.g. signaling "no data" with a
    /// sentinel) doesn't slip in unnoticed.
    #[test]
    fn defaults_are_zero_or_empty() {
        assert_eq!(MonotonicCount::default(), MonotonicCount(0));
        assert_eq!(MonotonicNs::default(), MonotonicNs(0));
        assert_eq!(ClockTicks::default(), ClockTicks(0));
        assert_eq!(Bytes::default(), Bytes(0));
        assert_eq!(DeadCounter::default(), DeadCounter(0));
        assert_eq!(PeakNs::default(), PeakNs(0));
        assert_eq!(GaugeNs::default(), GaugeNs(0));
        assert_eq!(GaugeCount::default(), GaugeCount(0));
        assert_eq!(OrdinalI32::default(), OrdinalI32(0));
        assert_eq!(OrdinalU32::default(), OrdinalU32(0));
        assert_eq!(OrdinalU64::default(), OrdinalU64(0));
        assert_eq!(CategoricalString::default(), CategoricalString::from(""));
        assert_eq!(CpuSet::default(), CpuSet(vec![]));
    }
}
