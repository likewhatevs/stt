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
//! Encoding the category into the type system surfaces
//! category-mismatched aggregation as a compile error. Once the
//! [`crate::host_state_compare::AggRule`] dispatch migrates to
//! typed reductions in phase 3, a future registry entry that
//! pairs a peak field with a sum reduction — e.g. `t.wait_max`
//! (`PeakNs`) bound to a `Sum(...)` rule whose accessor returns
//! a `Summable` value — will fail to compile rather than produce
//! a meaningless `1×1s ⊕ 1000×1ms` aggregate. This module
//! defines the newtypes and traits the migration consumes;
//! AggRule itself still carries the legacy `fn(&ThreadState)
//! -> u64` shape until phase 3 lands.
//!
//! # The newtypes
//!
//! - [`MonotonicCount`] — pure counter (only ever goes up across a
//!   thread's lifetime). Examples: `nr_wakeups`, `nr_migrations`,
//!   `voluntary_csw`.
//! - [`MonotonicNs`] — cumulative-time counter, ns. Examples:
//!   `run_time_ns`, `wait_sum`, `sleep_sum`, `block_sum`,
//!   `iowait_sum`, `core_forceidle_sum`.
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
//!   sum to N×gauge with no physical meaning. Structural counts
//!   that go up AND down at runtime (e.g. `nr_threads`, the
//!   process-wide thread count from `signal_struct->nr_threads`)
//!   share the gauge family conceptually — they are not
//!   monotonic — and reduce by max across a group rather than
//!   sum, even though the underlying quantity is integer rather
//!   than time. Phase 2 may either route them through
//!   [`GaugeNs`] (with a unit-tag override on the rendered cell)
//!   or grow a dedicated `GaugeCount` newtype.
//! - [`ClockTicks`] — USER_HZ-scaled time. Examples:
//!   `utime_clock_ticks`, `stime_clock_ticks`,
//!   `delayacct_blkio_ticks`. Auto-scale ladder is
//!   `ticks → Kticks → Mticks` (decimal SI), distinct from ns
//!   (also decimal SI, different unit) and bytes (IEC binary).
//! - [`Bytes`] — byte counts. Examples: `allocated_bytes`,
//!   `read_bytes`, `wchar`. Auto-scale ladder is IEC binary
//!   (`B → KiB → MiB → GiB`).
//! - [`OrdinalI32`] / [`OrdinalU32`] / [`OrdinalU64`] — bounded
//!   scalar, range-aggregated (no sum). [`OrdinalI32`] examples:
//!   `nice` ([-20, 19]), `priority`
//!   (CFS=[0, 39], RT=[-2, -100], DL=-101), `processor` (last
//!   CPU the task ran on; signed for symmetry with `nice` — the
//!   kernel's `task_cpu()` returns `unsigned int`
//!   (`include/linux/sched.h`), but ktstr stores i32 to share
//!   the [`OrdinalI32`] wrapper with the genuinely-signed nice
//!   and priority fields). [`OrdinalU64`] is for u64-backed
//!   ordinal fields like `rt_priority` (real-time priority,
//!   0..99). [`OrdinalU32`] currently has no example in the
//!   registry — it is reserved for phase-2 migration targets
//!   that need an unsigned 32-bit ordinal (e.g. a future capture
//!   field whose value is bounded by `u32::MAX` rather than
//!   `i32::MAX`).
//! - [`CategoricalString`] — string-valued, mode-aggregated.
//!   Examples: `policy`, `state`. Note: `ext_enabled` is currently
//!   coerced through `String` via `Display`; if a second bool
//!   field appears, promote both to a dedicated `CategoricalBool`
//!   wrapper.
//! - [`CpuSet`] — `Vec<u32>` of CPU IDs, affinity-aggregated.
//!   `cpu_affinity` is the only example.
//!
//! # The marker traits
//!
//! - [`Summable`] — sum across a group. Implemented by the four
//!   counter newtypes ([`MonotonicCount`], [`MonotonicNs`],
//!   [`ClockTicks`], [`Bytes`]). NOT implemented by [`PeakNs`] /
//!   [`GaugeNs`] / [`OrdinalI32`] / [`OrdinalU32`] /
//!   [`OrdinalU64`] / [`CategoricalString`] / [`CpuSet`]. The
//!   trait is sealed via [`sealed::SummableSealed`] so a
//!   downstream crate cannot add `impl Summable for PeakNs` to
//!   bypass the category invariant.
//! - [`Maxable`] — reduce by max. Implemented by every newtype
//!   that has a meaningful "worst observed" reading: every
//!   `Summable` (max-of-counter is "biggest single contributor")
//!   plus [`PeakNs`] (max-of-peak is "worst peak any contributor
//!   saw") plus [`GaugeNs`] (max-of-gauge is "longest current
//!   slice in the bucket"). NOT implemented by ordinals (those
//!   carry a `[min, max]` range, not a single max), nor by
//!   [`CategoricalString`] (string max has no useful semantic),
//!   nor by [`CpuSet`] (the affinity reduction is a custom
//!   summary, not a bare max). Sealed via
//!   [`sealed::MaxableSealed`].
//! - [`Modeable`] — reduce by mode (most-frequent value).
//!   Implemented by [`CategoricalString`] only.
//! - [`Rangeable`] — reduce by `[min, max]`. Implemented by
//!   [`OrdinalI32`], [`OrdinalU32`], and [`OrdinalU64`].
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
//! representation matches the unwrapped primitive. Existing
//! snapshot files (`.hst.zst`) keep deserializing once
//! [`crate::host_state::ThreadState`] migrates from raw `u64` to
//! these newtypes (phase 2 of the migration; see the task
//! sequence on issue #52).
//!
//! # What this module is NOT
//!
//! - It is NOT a unit-of-measure system. There is no
//!   `MonotonicNs * MonotonicNs = MonotonicNs²` — these wrappers
//!   carry semantic category, not algebraic dimensionality.
//! - It is NOT a runtime-typed value enum (that lives next to the
//!   [`crate::host_state_compare::AggRule`] dispatch in phase 3).
//!   This module only defines the building-block newtypes.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Newtype wrappers
// ---------------------------------------------------------------------------

/// Pure monotonic counter — only ever goes up over a thread's
/// lifetime. Sum across a group, delta across snapshots.
///
/// Examples in [`crate::host_state_compare::HOST_STATE_METRICS`]:
/// `nr_wakeups`, `nr_migrations`, `voluntary_csw`,
/// `nonvoluntary_csw`, `wait_count`, `iowait_count`,
/// `timeslices`, `minflt`, `majflt`, `syscr`, `syscw`.
///
/// `nr_threads` is NOT in this category — it is a structural
/// gauge that goes up AND down at runtime (threads spawn and
/// exit), so it reduces by max across a group, not sum. See
/// the gauge family note on [`GaugeNs`].
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonotonicCount(pub u64);

/// Cumulative-time counter, nanoseconds. Same shape as
/// [`MonotonicCount`] but tagged for the ns auto-scale ladder
/// (ns → µs → ms → s).
///
/// Examples: `run_time_ns`, `wait_time_ns`, `wait_sum`,
/// `sleep_sum`, `block_sum`, `iowait_sum`, `core_forceidle_sum`.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonotonicNs(pub u64);

/// USER_HZ-scaled tick counter. The kernel exposes user-mode and
/// kernel-mode CPU time, plus delayacct blkio delay, in ticks of
/// the userspace-visible `USER_HZ` frequency. Auto-scale ladder
/// is `ticks → Kticks → Mticks` (decimal SI), kept distinct from
/// ns and bytes so the rendered cell carries the correct unit
/// suffix.
///
/// Examples: `utime_clock_ticks`, `stime_clock_ticks`,
/// `delayacct_blkio_ticks`.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ClockTicks(pub u64);

/// Byte count, IEC-binary auto-scaled
/// (`B → KiB → MiB → GiB → TiB`).
///
/// Examples: `allocated_bytes`, `deallocated_bytes`, `rchar`,
/// `wchar`, `read_bytes`, `write_bytes`.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Bytes(pub u64);

/// Lifetime high-water mark, nanoseconds. The kernel updates
/// these as a max-against-prior in `update_stats_*` /
/// `update_se` / `set_next_entity` paths
/// (`kernel/sched/stats.c`, `kernel/sched/fair.c`). Group
/// reduction takes max across contributors so the rendered cell
/// surfaces the worst single window any thread experienced.
///
/// Summing peaks across threads is a category error — does not
/// implement [`Summable`]. Implements [`Maxable`].
///
/// Examples: `wait_max`, `sleep_max`, `block_max`, `exec_max`,
/// `slice_max`.
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
/// high-water mark.
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
/// as [`OrdinalI32`] but for unsigned 32-bit fields. No registry
/// metric uses this width today; reserved for phase-2 migration
/// targets that need an unsigned 32-bit ordinal.
#[repr(transparent)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct OrdinalU32(pub u32);

/// Bounded ordinal scalar (u64). Same range-aggregation contract
/// as [`OrdinalI32`] but for unsigned 64-bit fields.
///
/// Example: `rt_priority` (real-time priority, bounded 0..99 in
/// practice for SCHED_FIFO / SCHED_RR; stored as `u64` in
/// [`crate::host_state::ThreadState`] because
/// `/proc/<tid>/stat` field 40 is parsed via the same
/// `parse::<u64>()` path as the other unsigned schedstat
/// fields).
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
/// Examples: `policy` (SCHED_OTHER, SCHED_FIFO, SCHED_RR,
/// SCHED_BATCH, SCHED_IDLE, SCHED_DEADLINE, SCHED_EXT), `state`
/// (R, S, D, Z, T, etc.). The `ext_enabled` bool field is
/// coerced through `String` via `Display` at the
/// `AggRule::Mode` accessor — this wrapper does not currently
/// hold a dedicated `bool` form because the codebase has only
/// one bool-valued metric.
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
impl_u64_newtype_from!(PeakNs);
impl_u64_newtype_from!(GaugeNs);

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
// Marker traits + reductions
// ---------------------------------------------------------------------------

/// Private sealing module: the supertraits live here so a
/// downstream crate cannot bypass the category invariant by
/// writing `impl Summable for PeakNs`. Adding a new Summable
/// (or Maxable) requires editing this module — the choke point
/// the type system creates.
mod sealed {
    /// Sealed supertrait of [`super::Summable`].
    pub trait SummableSealed {}
    /// Sealed supertrait of [`super::Maxable`].
    pub trait MaxableSealed {}
}

/// Marker for newtypes that can be summed across a group.
///
/// Implemented by [`MonotonicCount`], [`MonotonicNs`],
/// [`ClockTicks`], and [`Bytes`]. Deliberately NOT implemented by
/// [`PeakNs`] / [`GaugeNs`] / [`OrdinalI32`] / [`OrdinalU32`] /
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
pub trait Summable: sealed::SummableSealed + Sized + Copy {
    fn sum_across(items: impl IntoIterator<Item = Self>) -> Self;
}

/// Marker for newtypes that can be reduced by max across a
/// group.
///
/// Implemented by every [`Summable`] (max-of-counter answers
/// "what's the biggest single contributor's value" — well-defined
/// even for cumulative counters), plus [`PeakNs`] (max-of-peak is
/// the worst high-water mark any contributor saw) and
/// [`GaugeNs`] (max-of-gauge is the longest current value in
/// the bucket).
///
/// Deliberately NOT implemented by ordinals (those carry a
/// `[min, max]` range, not a single max), nor by
/// [`CategoricalString`] (string max has no useful semantic),
/// nor by [`CpuSet`] (the affinity reduction is a custom
/// summary, not a bare max).
///
/// Sealed via [`sealed::MaxableSealed`]: a downstream crate
/// cannot write `impl Maxable for CategoricalString` because the
/// sealed supertrait is private to this module.
pub trait Maxable: sealed::MaxableSealed + Sized + Copy + Ord {
    fn max_across(items: impl IntoIterator<Item = Self>) -> Self;
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
pub trait Modeable: Sized + Clone + Eq + Ord {
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

/// Marker for newtypes reduced by `[min, max]` range.
/// Implemented by [`OrdinalI32`], [`OrdinalU32`], and
/// [`OrdinalU64`].
///
/// `range_across` returns `None` on an empty iterator.
pub trait Rangeable: Sized + Copy + Ord {
    fn range_across(items: impl IntoIterator<Item = Self>) -> Option<(Self, Self)> {
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
        Some((min, max))
    }
}

// Macro for the four counter shapes — sum_across uses
// saturating_add and max_across uses Ord-based max. Both share
// the underlying u64 representation. The sealed supertrait impls
// gate Summable / Maxable so external crates can't extend the
// trait list outside this module.
macro_rules! impl_summable_maxable_u64 {
    ($t:ident) => {
        impl sealed::SummableSealed for $t {}
        impl sealed::MaxableSealed for $t {}
        impl Summable for $t {
            fn sum_across(items: impl IntoIterator<Item = Self>) -> Self {
                let mut total: u64 = 0;
                for v in items {
                    total = total.saturating_add(v.0);
                }
                Self(total)
            }
        }
        impl Maxable for $t {
            fn max_across(items: impl IntoIterator<Item = Self>) -> Self {
                let mut out = Self::default();
                for v in items {
                    if v > out {
                        out = v;
                    }
                }
                out
            }
        }
    };
}

impl_summable_maxable_u64!(MonotonicCount);
impl_summable_maxable_u64!(MonotonicNs);
impl_summable_maxable_u64!(ClockTicks);
impl_summable_maxable_u64!(Bytes);

// Peak / Gauge are Maxable but explicitly NOT Summable.
macro_rules! impl_maxable_only_u64 {
    ($t:ident) => {
        impl sealed::MaxableSealed for $t {}
        impl Maxable for $t {
            fn max_across(items: impl IntoIterator<Item = Self>) -> Self {
                let mut out = Self::default();
                for v in items {
                    if v > out {
                        out = v;
                    }
                }
                out
            }
        }
    };
}

impl_maxable_only_u64!(PeakNs);
impl_maxable_only_u64!(GaugeNs);

impl Rangeable for OrdinalI32 {}
impl Rangeable for OrdinalU32 {}
impl Rangeable for OrdinalU64 {}

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
    fn ordinal_u32_from_u32_roundtrips() {
        let v: OrdinalU32 = 256u32.into();
        assert_eq!(v.0, 256);
        let back: u32 = v.into();
        assert_eq!(back, 256);
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
    /// GaugeNs / ordinals / categoricals do NOT. The static
    /// `assert_summable<T>()` helper compiles only when the type
    /// satisfies `T: Summable`, so this test pins the four
    /// counter newtypes by exercising the bound. The negative
    /// assertion — that `assert_summable::<PeakNs>()` fails to
    /// compile — is enforced by the [`sealed::SummableSealed`]
    /// supertrait and the omission of `impl SummableSealed for
    /// PeakNs`. Adding it would require an explicit edit to this
    /// module's `impl_summable_maxable_u64!` invocations.
    #[test]
    fn summable_only_implemented_for_counters() {
        fn assert_summable<T: Summable>() {}
        assert_summable::<MonotonicCount>();
        assert_summable::<MonotonicNs>();
        assert_summable::<ClockTicks>();
        assert_summable::<Bytes>();
    }

    // -- Maxable --------------------------------------------------------------

    #[test]
    fn maxable_peak_ns_picks_largest() {
        let xs = [PeakNs(100), PeakNs(500), PeakNs(200)];
        let m = PeakNs::max_across(xs);
        assert_eq!(m, PeakNs(500));
    }

    #[test]
    fn maxable_gauge_ns_picks_largest() {
        let xs = [GaugeNs(7), GaugeNs(99), GaugeNs(50)];
        let m = GaugeNs::max_across(xs);
        assert_eq!(m, GaugeNs(99));
    }

    #[test]
    fn maxable_summable_counter_picks_largest() {
        let xs = [MonotonicCount(3), MonotonicCount(8), MonotonicCount(5)];
        let m = MonotonicCount::max_across(xs);
        assert_eq!(m, MonotonicCount(8));
    }

    #[test]
    fn maxable_empty_iterator_returns_zero() {
        let m = PeakNs::max_across(std::iter::empty());
        assert_eq!(m, PeakNs(0));
    }

    #[test]
    fn maxable_singleton_returns_that_value() {
        let m = PeakNs::max_across([PeakNs(42)]);
        assert_eq!(m, PeakNs(42));
    }

    /// Compile-time gate: Maxable is implemented by every
    /// newtype that has a "worst observed" reading. Static
    /// assertions for the implementing types.
    #[test]
    fn maxable_implemented_for_all_counter_and_peak_gauge() {
        fn assert_maxable<T: Maxable>() {}
        assert_maxable::<MonotonicCount>();
        assert_maxable::<MonotonicNs>();
        assert_maxable::<ClockTicks>();
        assert_maxable::<Bytes>();
        assert_maxable::<PeakNs>();
        assert_maxable::<GaugeNs>();
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
        let (min, max) = OrdinalI32::range_across(xs).expect("non-empty");
        assert_eq!(min, OrdinalI32(-20));
        assert_eq!(max, OrdinalI32(10));
    }

    #[test]
    fn rangeable_ordinal_u32_finds_min_max() {
        let xs = [OrdinalU32(7), OrdinalU32(3), OrdinalU32(15)];
        let (min, max) = OrdinalU32::range_across(xs).expect("non-empty");
        assert_eq!(min, OrdinalU32(3));
        assert_eq!(max, OrdinalU32(15));
    }

    #[test]
    fn rangeable_ordinal_u64_finds_min_max() {
        let xs = [
            OrdinalU64(50),
            OrdinalU64(99),
            OrdinalU64(0),
            OrdinalU64(25),
        ];
        let (min, max) = OrdinalU64::range_across(xs).expect("non-empty");
        assert_eq!(min, OrdinalU64(0));
        assert_eq!(max, OrdinalU64(99));
    }

    #[test]
    fn rangeable_singleton_min_eq_max() {
        let (min, max) = OrdinalI32::range_across([OrdinalI32(42)]).expect("non-empty");
        assert_eq!(min, OrdinalI32(42));
        assert_eq!(max, OrdinalI32(42));
    }

    #[test]
    fn rangeable_empty_iterator_returns_none() {
        let r = OrdinalI32::range_across(std::iter::empty());
        assert!(r.is_none());
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
    /// to its primitive. Pin the JSON shape so the future
    /// ThreadState migration (phase 2) doesn't break existing
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

        let raw_peak = PeakNs(99);
        let raw_peak_json = serde_json::to_string(&raw_peak).expect("serialize");
        assert_eq!(raw_peak_json, "99");

        let raw_gauge = GaugeNs(7_500_000);
        let raw_gauge_json = serde_json::to_string(&raw_gauge).expect("serialize");
        assert_eq!(raw_gauge_json, "7500000");

        let raw_ordi = OrdinalI32(-5);
        let raw_ordi_json = serde_json::to_string(&raw_ordi).expect("serialize");
        assert_eq!(raw_ordi_json, "-5");

        let raw_ordu32 = OrdinalU32(256);
        let raw_ordu32_json = serde_json::to_string(&raw_ordu32).expect("serialize");
        assert_eq!(raw_ordu32_json, "256");

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
        assert_eq!(size_of::<PeakNs>(), size_of::<u64>());
        assert_eq!(size_of::<GaugeNs>(), size_of::<u64>());
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
        assert_eq!(PeakNs::default(), PeakNs(0));
        assert_eq!(GaugeNs::default(), GaugeNs(0));
        assert_eq!(OrdinalI32::default(), OrdinalI32(0));
        assert_eq!(OrdinalU32::default(), OrdinalU32(0));
        assert_eq!(OrdinalU64::default(), OrdinalU64(0));
        assert_eq!(CategoricalString::default(), CategoricalString::from(""));
        assert_eq!(CpuSet::default(), CpuSet(vec![]));
    }
}
