//! Per-controller merge primitives for the cgroup-flatten path.
//!
//! Two layers:
//!
//! 1. Per-controller merge fns — [`merge_cgroup_cpu`],
//!    [`merge_cgroup_memory`], [`merge_cgroup_pids`] — encode the
//!    field-class policy: counters use `saturating_add`, gauges
//!    take max, limits use the max-for-limits / max-for-weights
//!    rule via [`merge_max_option`], floors use min-for-floors via
//!    [`merge_min_option`]. [`merge_psi`] / [`merge_psi_resource`] /
//!    [`merge_psi_half`] mirror the same policy split for PSI:
//!    avg fields take max (200% has no meaning), `total_usec`
//!    counters add. [`merge_memory_stat`] dispatches per-key on
//!    [`MEMORY_STAT_GAUGE_KEYS`] so cumulative event counts add
//!    while pool sizes max.
//!
//! 2. Generic kv helpers — [`merge_max_option`] /
//!    [`merge_min_option`] / [`merge_kv_counters`]. The two
//!    Option-shaped helpers encode opposing kernel semantics with
//!    surface-symmetric `None`-poisoning: limits propagate `None`
//!    when ANY contributor is unbounded (the merged bucket is as
//!    permissive as its weakest contributor's limit), floors
//!    propagate `None` when ANY contributor has no floor (the
//!    merged bucket is as unprotected as its weakest
//!    contributor's floor). [`merge_kv_counters`] is the plain
//!    per-key sum used by `memory.events` and any other purely-
//!    counter-shaped key/value map.
//!
//! All entry points are `pub(super)` — every consumer is the
//! [`super::flatten_cgroup_stats`] orchestrator in mod.rs. Per-
//! field policy rationale lives on the individual fns.

use std::collections::BTreeMap;

use crate::ctprof::{
    CgroupCpuStats, CgroupMemoryStats, CgroupPidsStats, Psi, PsiHalf, PsiResource,
};

/// Merge two [`CgroupCpuStats`]: counters use `saturating_add`,
/// limits/knobs use the max-for-limits / max-for-weights rule.
/// Floors don't apply here (none in this domain). `period`
/// takes the larger value as a stable fallback when
/// contributors set different periods.
pub(super) fn merge_cgroup_cpu(agg: &mut CgroupCpuStats, src: &CgroupCpuStats) {
    agg.usage_usec = agg.usage_usec.saturating_add(src.usage_usec);
    agg.nr_throttled = agg.nr_throttled.saturating_add(src.nr_throttled);
    agg.throttled_usec = agg.throttled_usec.saturating_add(src.throttled_usec);
    agg.max_quota_us = merge_max_option(agg.max_quota_us, src.max_quota_us);
    agg.max_period_us = agg.max_period_us.max(src.max_period_us);
    // `weight` and `weight_nice` are aliases of the same kernel
    // knob (`kernel/sched/core.c::sched_weight_to_nice` /
    // `nice_to_weight`). Apply the SAME merge policy to both —
    // asymmetric merging would render a `weight=10, weight_nice=None`
    // bucket as if its contributors disagreed when they cannot
    // (the kernel writes both atomically). Use `merge_max_option`
    // (None-poisons) for both: the merged bucket is "no weight
    // configured" if any contributor is unconfigured.
    agg.weight = merge_max_option(agg.weight, src.weight);
    agg.weight_nice = match (agg.weight_nice, src.weight_nice) {
        (Some(a), Some(b)) => Some(a.max(b)),
        // Mirror merge_max_option's None-poisoning policy:
        // None ∨ Some = None. Treats "absent file" as
        // "unconfigured" — merged bucket inherits the
        // unconfigured state.
        (Some(_), None) | (None, Some(_)) | (None, None) => None,
    };
}

/// Merge two [`CgroupMemoryStats`]. `current` is instantaneous
/// RSS — `max` matches the existing memory_current policy.
/// Limits (`max`, `high`) use max-for-limits, floors (`low`,
/// `min`) use min-for-floors per Q4. `stat` is a heterogeneous
/// map (counters + gauges) — see [`merge_memory_stat`] for the
/// per-key policy. `events` is purely counter-shaped — sum
/// per-key via [`merge_kv_counters`].
pub(super) fn merge_cgroup_memory(agg: &mut CgroupMemoryStats, src: &CgroupMemoryStats) {
    agg.current = agg.current.max(src.current);
    agg.max = merge_max_option(agg.max, src.max);
    agg.high = merge_max_option(agg.high, src.high);
    agg.low = merge_min_option(agg.low, src.low);
    agg.min = merge_min_option(agg.min, src.min);
    merge_memory_stat(&mut agg.stat, &src.stat);
    merge_kv_counters(&mut agg.events, &src.events);
}

/// `memory.stat` keys whose values are INSTANTANEOUS GAUGES,
/// not cumulative counters. The kernel emits these as the
/// current (point-in-time) byte count for that pool — summing
/// across cgroups overstates the merged-bucket gauge, so the
/// merge takes max instead. Keys NOT in this list are
/// counter-shaped (pgfault, pgmajfault, workingset_*,
/// pgsteal_*, pgscan_*, pgrefill, etc.) and merge via
/// `saturating_add`.
///
/// List sourced from inspecting the v2 `memory.stat` emission
/// path in `mm/memcontrol.c` and the cgroup v2 documentation:
/// these names denote pools (active resident bytes), not
/// occurrences. Conservative — if a key is unknown, the merge
/// defaults to sum (the existing kv-counter policy).
const MEMORY_STAT_GAUGE_KEYS: &[&str] = &[
    "anon",
    "file",
    "kernel",
    "kernel_stack",
    "pagetables",
    "sec_pagetables",
    "percpu",
    "sock",
    "vmalloc",
    "shmem",
    "zswap",
    "zswapped",
    "file_mapped",
    "file_dirty",
    "file_writeback",
    "swapcached",
    "anon_thp",
    "file_thp",
    "shmem_thp",
    "inactive_anon",
    "active_anon",
    "inactive_file",
    "active_file",
    "unevictable",
    "slab_reclaimable",
    "slab_unreclaimable",
    "slab",
    "hugetlb",
];

/// Merge `memory.stat` maps with per-key policy: gauge keys
/// (per [`MEMORY_STAT_GAUGE_KEYS`]) take max; counter keys
/// take saturating_add. Gauges are point-in-time pool sizes
/// (`anon`, `file`, `slab`, etc.) — summing across cgroups
/// overstates the merged-bucket pool. Counter keys
/// (workingset_refault_*, pgfault, pgmajfault, pgsteal_*,
/// etc.) are cumulative event counts — additive across the
/// merged bucket.
pub(super) fn merge_memory_stat(agg: &mut BTreeMap<String, u64>, src: &BTreeMap<String, u64>) {
    for (key, value) in src {
        let is_gauge = MEMORY_STAT_GAUGE_KEYS.contains(&key.as_str());
        agg.entry(key.clone())
            .and_modify(|v| {
                *v = if is_gauge {
                    (*v).max(*value)
                } else {
                    v.saturating_add(*value)
                };
            })
            .or_insert(*value);
    }
}

/// Merge two [`CgroupPidsStats`]. `current` is a point-in-time
/// task count — the merged bucket's count is the sum across
/// contributors at the moment of capture (each contributor's
/// processes are disjoint by construction, so the sum is the
/// total count). `max` is a limit (max-for-limits).
pub(super) fn merge_cgroup_pids(agg: &mut CgroupPidsStats, src: &CgroupPidsStats) {
    agg.current = match (agg.current, src.current) {
        (Some(a), Some(b)) => Some(a.saturating_add(b)),
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    };
    agg.max = merge_max_option(agg.max, src.max);
}

/// Merge policy for `Option<u64>` LIMITS: take the max across
/// contributors. `None` means "no limit" — propagating `None`
/// when EITHER side is unbounded matches the kernel's actual
/// behavior (the merged bucket is unbounded if any contributor
/// is). When both sides have concrete values, max gives "the
/// largest cap any contributor enforces".
///
/// Surface-symmetric with [`merge_min_option`] but the kernel
/// semantics are OPPOSITE: limits use `None` to mean
/// "unbounded" (any contributor unbounded ⇒ merged unbounded);
/// floors use `None` to mean "no protection" (any contributor
/// unprotected ⇒ merged unprotected). They share the same
/// None-poisoning shape because both interpret missing as
/// "the weakest contributor wins" in their respective
/// directions.
pub(super) fn merge_max_option(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        // `None` = "no limit"; merged bucket is unbounded if
        // either contributor is. Drop the concrete value rather
        // than synthesize a bound that doesn't reflect reality.
        (Some(_), None) | (None, Some(_)) => None,
        (None, None) => None,
    }
}

/// Merge policy for `Option<u64>` FLOORS (memory.low,
/// memory.min): take the min across contributors. `None` means
/// "no floor" (no protection); propagate `None` when either
/// side has no floor — the merged bucket is only as protected
/// as its weakest contributor.
pub(super) fn merge_min_option(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(_), None) | (None, Some(_)) => None,
        (None, None) => None,
    }
}

/// Per-key sum of two key-value counter maps. Keys present only
/// on one side are copied verbatim; keys on both sides sum with
/// saturating_add.
pub(super) fn merge_kv_counters(agg: &mut BTreeMap<String, u64>, src: &BTreeMap<String, u64>) {
    for (key, value) in src {
        agg.entry(key.clone())
            .and_modify(|v| *v = v.saturating_add(*value))
            .or_insert(*value);
    }
}

/// Merge two [`Psi`] bundles for the cgroup-flatten path. PSI
/// avg fields (`avg10/60/300`) are percentages, so summing
/// across cgroups overstates the merged-bucket pressure
/// (200% has no meaning); max gives "worst-pressured cgroup
/// in the merged bucket" which is the actionable signal for
/// regression detection. `total_usec` is cumulative microseconds
/// of stall time, additive across the merged cgroups —
/// `saturating_add` matches the existing `throttled_usec`
/// flatten policy directly above.
pub(super) fn merge_psi(a: Psi, b: Psi) -> Psi {
    Psi {
        cpu: merge_psi_resource(a.cpu, b.cpu),
        memory: merge_psi_resource(a.memory, b.memory),
        io: merge_psi_resource(a.io, b.io),
        irq: merge_psi_resource(a.irq, b.irq),
    }
}

fn merge_psi_resource(a: PsiResource, b: PsiResource) -> PsiResource {
    PsiResource {
        some: merge_psi_half(a.some, b.some),
        full: merge_psi_half(a.full, b.full),
    }
}

fn merge_psi_half(a: PsiHalf, b: PsiHalf) -> PsiHalf {
    PsiHalf {
        avg10: a.avg10.max(b.avg10),
        avg60: a.avg60.max(b.avg60),
        avg300: a.avg300.max(b.avg300),
        total_usec: a.total_usec.saturating_add(b.total_usec),
    }
}
