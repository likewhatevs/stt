//! Tests for `super::cgroup_merge` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::*;
use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters,
    merge_max_option, merge_memory_stat, merge_min_option, merge_psi,
};
use super::columns::{compare_columns_for, format_cgroup_only_section_warning};
use super::compare::sort_diff_rows_by_keys;
use super::groups::build_row;
use super::pattern::{
    Segment, apply_systemd_template, cgroup_normalize_skeleton, cgroup_skeleton_tokens,
    classify_token, is_token_separator, pattern_counts_union, pattern_key, split_into_segments,
    tighten_group,
};
use super::render::psi_pair_has_data;
use super::scale::{auto_scale, format_delta_cell};
use super::tests_fixtures::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// `merge_max_option` policy: take the max across
/// contributors when both have a concrete cap; propagate
/// `None` when either side is unbounded (matches kernel
/// "no limit" semantics — the merged bucket is unbounded if
/// any contributor is).
#[test]
fn merge_max_option_propagates_no_limit() {
    assert_eq!(merge_max_option(Some(100), Some(200)), Some(200));
    assert_eq!(merge_max_option(Some(200), Some(100)), Some(200));
    assert_eq!(merge_max_option(Some(50), Some(50)), Some(50));
    // None ∨ Some = None (an unbounded contributor makes
    // the merged bucket unbounded).
    assert_eq!(merge_max_option(None, Some(100)), None);
    assert_eq!(merge_max_option(Some(100), None), None);
    assert_eq!(merge_max_option(None, None), None);
}

/// `merge_min_option` policy: take the min across
/// contributors when both have a concrete floor; propagate
/// `None` when either side has no floor (matches the floor
/// equivalent of the limit policy — merged bucket is only
/// as protected as its weakest contributor).
#[test]
fn merge_min_option_propagates_no_floor() {
    assert_eq!(merge_min_option(Some(100), Some(200)), Some(100));
    assert_eq!(merge_min_option(Some(200), Some(100)), Some(100));
    assert_eq!(merge_min_option(None, Some(100)), None);
    assert_eq!(merge_min_option(Some(100), None), None);
    assert_eq!(merge_min_option(None, None), None);
}

/// `merge_kv_counters` per-key sum: keys present on both
/// sides sum; one-sided keys copy verbatim. Pure
/// counter-shaped policy — used for `memory.events` where
/// every key is a counter.
#[test]
fn merge_kv_counters_per_key_sum() {
    let mut agg: BTreeMap<String, u64> = BTreeMap::new();
    agg.insert("oom_kill".into(), 10);
    agg.insert("high".into(), 20);
    let mut src: BTreeMap<String, u64> = BTreeMap::new();
    src.insert("oom_kill".into(), 5);
    src.insert("low".into(), 7);
    merge_kv_counters(&mut agg, &src);
    assert_eq!(agg.get("oom_kill"), Some(&15), "common key sums");
    assert_eq!(agg.get("high"), Some(&20), "agg-only key preserved");
    assert_eq!(agg.get("low"), Some(&7), "src-only key copied");
}

/// `merge_memory_stat` per-key dispatch: gauge keys (per
/// [`MEMORY_STAT_GAUGE_KEYS`]) take max; counter keys take
/// saturating_add. Summing instantaneous pool sizes
/// (anon, file, slab) overstates the merged-bucket gauge,
/// so the gauge keys take max instead.
#[test]
fn merge_memory_stat_dispatches_gauge_vs_counter() {
    let mut agg: BTreeMap<String, u64> = BTreeMap::new();
    agg.insert("anon".into(), 1_000_000);
    agg.insert("file".into(), 500_000);
    agg.insert("slab".into(), 800_000);
    agg.insert("pgfault".into(), 100);
    agg.insert("workingset_refault_anon".into(), 50);
    let mut src: BTreeMap<String, u64> = BTreeMap::new();
    src.insert("anon".into(), 2_000_000);
    src.insert("file".into(), 100_000);
    src.insert("slab".into(), 300_000);
    src.insert("pgfault".into(), 25);
    src.insert("workingset_refault_anon".into(), 10);
    merge_memory_stat(&mut agg, &src);
    // Gauges: max wins (NOT sum).
    assert_eq!(agg.get("anon"), Some(&2_000_000), "anon is gauge → max");
    assert_eq!(agg.get("file"), Some(&500_000), "file is gauge → max");
    assert_eq!(agg.get("slab"), Some(&800_000), "slab is gauge → max");
    // Counters: sum.
    assert_eq!(agg.get("pgfault"), Some(&125), "pgfault is counter → sum");
    assert_eq!(
        agg.get("workingset_refault_anon"),
        Some(&60),
        "workingset_refault_anon is counter → sum"
    );
}

