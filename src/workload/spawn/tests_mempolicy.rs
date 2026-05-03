//! Spawn-pipeline tests — mempolicy group.

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
fn build_nodemask_empty() {
    let (mask, maxnode) = build_nodemask(&BTreeSet::new());
    assert!(mask.is_empty());
    assert_eq!(maxnode, 0);
}
#[test]
fn build_nodemask_single() {
    let (mask, maxnode) = build_nodemask(&[0].into_iter().collect());
    // kernel get_nodes() does --maxnode, so maxnode = max_node + 2
    assert_eq!(maxnode, 2);
    assert_eq!(mask.len(), 1);
    assert_eq!(mask[0], 1);
}
#[test]
fn build_nodemask_multiple() {
    let (mask, maxnode) = build_nodemask(&[0, 2].into_iter().collect());
    assert_eq!(maxnode, 4); // max_node=2, +2 = 4
    assert_eq!(mask[0] & 1, 1); // node 0
    assert_eq!(mask[0] & 4, 4); // node 2
    assert_eq!(mask[0] & 2, 0); // node 1 not set
}
#[test]
fn build_nodemask_high_node() {
    let bits_per_word = std::mem::size_of::<libc::c_ulong>() * 8;
    let high = bits_per_word + 3;
    let (mask, maxnode) = build_nodemask(&[high].into_iter().collect());
    assert_eq!(maxnode, (high + 2) as libc::c_ulong);
    assert_eq!(mask.len(), 2);
    assert_eq!(mask[0], 0);
    assert_eq!(mask[1], 1 << 3);
}
#[test]
fn apply_mempolicy_default_is_noop() {
    apply_mempolicy_with_flags(&MemPolicy::Default, MpolFlags::NONE);
}
#[test]
fn apply_mempolicy_empty_bind_skipped() {
    apply_mempolicy_with_flags(&MemPolicy::Bind(BTreeSet::new()), MpolFlags::NONE);
}
#[test]
fn apply_mempolicy_empty_interleave_skipped() {
    apply_mempolicy_with_flags(&MemPolicy::Interleave(BTreeSet::new()), MpolFlags::NONE);
}
/// `WorkType::NumaWorkingSetSweep` smoke test. Empty
/// `target_nodes` disables binding (per the variant's doc:
/// "Empty list disables binding ... no migration is
/// triggered"); the worker still touches the region every
/// iteration. Sufficient for the pathology smoke check —
/// real multi-node migration tests live under
/// `tests/numa_tests.rs` (see #143/#146).
#[test]
fn pathology_numa_working_set_sweep_iterates() {
    let cfg = WorkloadConfig {
        num_workers: 2,
        work_type: WorkType::NumaWorkingSetSweep {
            region_kb: 256,
            sweep_period_ms: 100,
            target_nodes: vec![],
        },
        ..Default::default()
    };
    let mut h = WorkloadHandle::spawn(&cfg).expect("NumaWorkingSetSweep must spawn");
    h.start();
    std::thread::sleep(Duration::from_millis(200));
    let reports = h.stop_and_collect();
    assert_eq!(reports.len(), 2);
    for r in &reports {
        assert!(
            r.iterations > 0,
            "NumaWorkingSetSweep worker must iterate: {r:?}"
        );
    }
}
