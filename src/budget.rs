//! Budget-based test selection via greedy coverage maximization.
//!
//! When `KTSTR_BUDGET_SECS` is set during `--list`, selects the subset of
//! tests that maximizes feature coverage within the time budget. Each
//! test is encoded as a bitset feature vector capturing scheduler,
//! topology, and workload properties. The greedy algorithm picks tests
//! with the highest marginal-coverage-per-second ratio.

use crate::test_support::KtstrTestEntry;
use crate::vmm::topology::Topology;

/// A test candidate for budget selection.
pub(crate) struct TestCandidate {
    /// Full test name for `--list` output (e.g. `"gauntlet/basic/tiny-1llc: test"`).
    pub name: String,
    /// Bitset encoding test properties for coverage measurement.
    pub features: u64,
    /// Estimated wall-clock seconds to run this test.
    pub estimated_secs: f64,
}

// Bit layout (non-overlapping, one-hot for multi-value fields):
//
// All multi-value fields use one-hot encoding so that distinct values
// set disjoint bits. This preserves the submodularity property that
// the greedy algorithm relies on: covering one bucket value never
// partially covers a different bucket value.
//
//   Bits  0..3:  scheduler name hash (4 one-hot bits)
//   Bits  4..8:  CPU count bucket (5 one-hot bits)
//   Bits  9..13: LLC count bucket (5 one-hot bits)
//   Bit  14:     SMT (threads_per_core > 1)
//   Bit  15:     performance_mode
//   Bit  16:     host_only
//   Bit  17:     expect_err
//   Bits 18..20: duration bucket (3 one-hot bits)
//   Bit  21:     workers_per_cgroup bucket (1 bit)
//   Bit  22:     is gauntlet variant
//   Bits 23..26: test name hash (4 one-hot bits)
//   Bits 27..30: NUMA node count bucket (4 one-hot bits)

const SCHED_SHIFT: u32 = 0;
const CPU_BUCKET_SHIFT: u32 = 4;
const LLC_BUCKET_SHIFT: u32 = 9;
const SMT_SHIFT: u32 = 14;
const PERF_MODE_SHIFT: u32 = 15;
const HOST_ONLY_SHIFT: u32 = 16;
const EXPECT_ERR_SHIFT: u32 = 17;
const DURATION_SHIFT: u32 = 18;
const WORKERS_SHIFT: u32 = 21;
const GAUNTLET_SHIFT: u32 = 22;
const NAME_HASH_SHIFT: u32 = 23;
const NUMA_BUCKET_SHIFT: u32 = 27;

/// DJB2 string hash.
fn djb2_hash(name: &str) -> u32 {
    let mut h: u32 = 5381;
    for b in name.bytes() {
        h = h.wrapping_mul(33).wrapping_add(b as u32);
    }
    h
}

/// Classify total CPU count into a one-hot bit (5 classes).
fn cpu_bucket(total_cpus: u32) -> u64 {
    match total_cpus {
        0..=8 => 1 << 0,
        9..=16 => 1 << 1,
        17..=64 => 1 << 2,
        65..=128 => 1 << 3,
        _ => 1 << 4,
    }
}

/// Classify LLC count into a one-hot bit (5 classes).
fn llc_bucket(num_llcs: u32) -> u64 {
    match num_llcs {
        0..=1 => 1 << 0,
        2 => 1 << 1,
        3..=4 => 1 << 2,
        5..=8 => 1 << 3,
        _ => 1 << 4,
    }
}

/// Classify duration (seconds) into a one-hot bit (3 classes).
fn duration_bucket(duration_secs: u64) -> u64 {
    match duration_secs {
        0..=2 => 1 << 0,
        3..=10 => 1 << 1,
        _ => 1 << 2,
    }
}

/// Classify workers_per_cgroup (1 bit: low or high).
fn workers_bucket(workers: u32) -> u64 {
    if workers <= 2 { 0 } else { 1 }
}

/// Classify NUMA node count into a one-hot bit (4 classes).
fn numa_bucket(numa_nodes: u32) -> u64 {
    match numa_nodes {
        0..=1 => 1 << 0,
        2 => 1 << 1,
        3..=4 => 1 << 2,
        _ => 1 << 3,
    }
}

/// Extract feature bitset from a KtstrTestEntry and topology.
///
/// `is_gauntlet` distinguishes base from gauntlet variants.
pub(crate) fn extract_features(
    entry: &KtstrTestEntry,
    topo: &Topology,
    is_gauntlet: bool,
    test_name: &str,
) -> u64 {
    let mut bits = 0u64;

    // Scheduler name hash (4 one-hot bits)
    bits |= (1u64 << (djb2_hash(entry.scheduler.name) % 4)) << SCHED_SHIFT;

    // Topology
    bits |= cpu_bucket(topo.total_cpus()) << CPU_BUCKET_SHIFT;
    bits |= llc_bucket(topo.num_llcs()) << LLC_BUCKET_SHIFT;
    if topo.threads_per_core > 1 {
        bits |= 1 << SMT_SHIFT;
    }
    bits |= numa_bucket(topo.numa_nodes) << NUMA_BUCKET_SHIFT;

    // Test properties
    if entry.performance_mode {
        bits |= 1 << PERF_MODE_SHIFT;
    }
    if entry.host_only {
        bits |= 1 << HOST_ONLY_SHIFT;
    }
    if entry.expect_err {
        bits |= 1 << EXPECT_ERR_SHIFT;
    }

    // Duration bucket
    bits |= duration_bucket(entry.duration.as_secs()) << DURATION_SHIFT;

    // Workers bucket
    bits |= workers_bucket(entry.workers_per_cgroup) << WORKERS_SHIFT;

    // Gauntlet flag
    if is_gauntlet {
        bits |= 1 << GAUNTLET_SHIFT;
    }

    // Test name hash (4 one-hot bits)
    bits |= (1u64 << (djb2_hash(test_name) % 4)) << NAME_HASH_SHIFT;

    bits
}

/// Estimate wall-clock seconds for a test.
///
/// VM tests: `10 + max(0, (total_cpus - 16) / 10) + duration + 2`.
/// Adds 3s for `performance_mode` (hugepage setup, vCPU pinning).
/// The 10s baseline covers kernel boot for a 16-CPU VM; beyond that
/// baseline, one additional second is added per 10 CPUs of VM size
/// to cover the linear vCPU + memory setup overhead.
///
/// Host-only tests run without a VM: `duration + 2`.
pub(crate) fn estimate_duration(entry: &KtstrTestEntry, topo: &Topology) -> f64 {
    let duration_secs = entry.duration.as_secs();
    if entry.host_only {
        return (duration_secs + 2) as f64;
    }
    let cpus = topo.total_cpus() as u64;
    let boot_overhead = 10 + cpus.saturating_sub(16) / 10;
    let perf_overhead: u64 = if entry.performance_mode { 3 } else { 0 };
    (boot_overhead + duration_secs + 2 + perf_overhead) as f64
}

/// Select tests that maximize feature coverage within a time budget.
///
/// Pure cost-effective greedy: repeatedly pick the test with the highest
/// `marginal_coverage / estimated_duration` ratio. Ties broken by test
/// name (lexicographic) for determinism. Stops when adding the next
/// test would exceed the budget or no candidate adds new coverage.
///
/// Returns indices into the input `candidates` slice, sorted.
pub(crate) fn select(candidates: &[TestCandidate], budget_secs: f64) -> Vec<usize> {
    if candidates.is_empty() || budget_secs <= 0.0 {
        return Vec::new();
    }

    let n = candidates.len();
    let mut selected = Vec::new();
    let mut used = vec![false; n];
    let mut covered: u64 = 0;
    let mut remaining_budget = budget_secs;

    loop {
        let mut best_idx: Option<usize> = None;
        let mut best_ratio: f64 = 0.0;
        let mut best_name: &str = "";

        for (i, c) in candidates.iter().enumerate() {
            if used[i] || c.estimated_secs > remaining_budget {
                continue;
            }
            let marginal = (c.features & !covered).count_ones() as f64;
            if marginal == 0.0 {
                continue;
            }
            let ratio = if c.estimated_secs > 0.0 {
                marginal / c.estimated_secs
            } else {
                marginal * 1e6 // zero-cost test: effectively infinite ratio
            };

            let better = ratio > best_ratio || (ratio == best_ratio && c.name.as_str() < best_name);
            if better {
                best_ratio = ratio;
                best_idx = Some(i);
                best_name = &c.name;
            }
        }

        match best_idx {
            Some(i) => {
                selected.push(i);
                used[i] = true;
                covered |= candidates[i].features;
                remaining_budget -= candidates[i].estimated_secs;
            }
            None => break,
        }
    }

    selected.sort_unstable();
    selected
}

/// Coverage statistics for the stderr summary.
pub(crate) struct SelectionStats {
    pub selected: usize,
    pub total: usize,
    pub budget_used: f64,
    pub budget_total: f64,
    pub bits_covered: u32,
    pub bits_possible: u32,
}

/// Compute coverage statistics for a selection.
pub(crate) fn selection_stats(
    candidates: &[TestCandidate],
    selected: &[usize],
    budget_secs: f64,
) -> SelectionStats {
    let mut covered: u64 = 0;
    let mut budget_used = 0.0;
    for &i in selected {
        covered |= candidates[i].features;
        budget_used += candidates[i].estimated_secs;
    }
    let mut all_features: u64 = 0;
    for c in candidates {
        all_features |= c.features;
    }
    SelectionStats {
        selected: selected.len(),
        total: candidates.len(),
        budget_used,
        budget_total: budget_secs,
        bits_covered: covered.count_ones(),
        bits_possible: all_features.count_ones(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_bucket_one_hot() {
        assert_eq!(cpu_bucket(1), 1 << 0);
        assert_eq!(cpu_bucket(8), 1 << 0);
        assert_eq!(cpu_bucket(9), 1 << 1);
        assert_eq!(cpu_bucket(16), 1 << 1);
        assert_eq!(cpu_bucket(17), 1 << 2);
        assert_eq!(cpu_bucket(64), 1 << 2);
        assert_eq!(cpu_bucket(65), 1 << 3);
        assert_eq!(cpu_bucket(128), 1 << 3);
        assert_eq!(cpu_bucket(129), 1 << 4);
        assert_eq!(cpu_bucket(252), 1 << 4);
    }

    #[test]
    fn cpu_bucket_no_shared_bits() {
        // Each bucket value must be a distinct power of 2.
        let vals = [
            cpu_bucket(4),
            cpu_bucket(12),
            cpu_bucket(32),
            cpu_bucket(96),
            cpu_bucket(200),
        ];
        for (i, &a) in vals.iter().enumerate() {
            assert_eq!(a.count_ones(), 1, "bucket {i} not one-hot: {a:#b}");
            for &b in &vals[i + 1..] {
                assert_eq!(a & b, 0, "buckets share bits: {a:#b} & {b:#b}");
            }
        }
    }

    #[test]
    fn llc_bucket_one_hot() {
        assert_eq!(llc_bucket(1), 1 << 0);
        assert_eq!(llc_bucket(2), 1 << 1);
        assert_eq!(llc_bucket(3), 1 << 2);
        assert_eq!(llc_bucket(4), 1 << 2);
        assert_eq!(llc_bucket(5), 1 << 3);
        assert_eq!(llc_bucket(8), 1 << 3);
        assert_eq!(llc_bucket(9), 1 << 4);
        assert_eq!(llc_bucket(15), 1 << 4);
    }

    #[test]
    fn llc_bucket_no_shared_bits() {
        let vals = [
            llc_bucket(1),
            llc_bucket(2),
            llc_bucket(3),
            llc_bucket(7),
            llc_bucket(10),
        ];
        for (i, &a) in vals.iter().enumerate() {
            assert_eq!(a.count_ones(), 1);
            for &b in &vals[i + 1..] {
                assert_eq!(a & b, 0);
            }
        }
    }

    #[test]
    fn duration_bucket_one_hot() {
        assert_eq!(duration_bucket(0), 1 << 0);
        assert_eq!(duration_bucket(2), 1 << 0);
        assert_eq!(duration_bucket(3), 1 << 1);
        assert_eq!(duration_bucket(10), 1 << 1);
        assert_eq!(duration_bucket(11), 1 << 2);
    }

    #[test]
    fn duration_bucket_no_shared_bits() {
        let vals = [duration_bucket(1), duration_bucket(5), duration_bucket(20)];
        for (i, &a) in vals.iter().enumerate() {
            assert_eq!(a.count_ones(), 1);
            for &b in &vals[i + 1..] {
                assert_eq!(a & b, 0);
            }
        }
    }

    #[test]
    fn workers_bucket_values() {
        assert_eq!(workers_bucket(1), 0);
        assert_eq!(workers_bucket(2), 0);
        assert_eq!(workers_bucket(3), 1);
        assert_eq!(workers_bucket(32), 1);
    }

    #[test]
    fn numa_bucket_one_hot() {
        assert_eq!(numa_bucket(0), 1 << 0);
        assert_eq!(numa_bucket(1), 1 << 0);
        assert_eq!(numa_bucket(2), 1 << 1);
        assert_eq!(numa_bucket(3), 1 << 2);
        assert_eq!(numa_bucket(4), 1 << 2);
        assert_eq!(numa_bucket(5), 1 << 3);
        assert_eq!(numa_bucket(8), 1 << 3);
    }

    #[test]
    fn numa_bucket_no_shared_bits() {
        let vals = [
            numa_bucket(1),
            numa_bucket(2),
            numa_bucket(3),
            numa_bucket(5),
        ];
        for (i, &a) in vals.iter().enumerate() {
            assert_eq!(a.count_ones(), 1, "bucket {i} not one-hot: {a:#b}");
            for &b in &vals[i + 1..] {
                assert_eq!(a & b, 0, "buckets share bits: {a:#b} & {b:#b}");
            }
        }
    }

    #[test]
    fn extract_features_numa_differentiation() {
        let entry = KtstrTestEntry::DEFAULT;
        let topo1 = Topology {
            llcs: 4,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let topo2 = Topology {
            llcs: 4,
            cores_per_llc: 4,
            threads_per_core: 1,
            numa_nodes: 2,
            nodes: None,
            distances: None,
        };
        let f1 = extract_features(&entry, &topo1, false, "numa_test");
        let f2 = extract_features(&entry, &topo2, false, "numa_test");
        // Same CPU/LLC counts, different NUMA => different features.
        assert_ne!(f1, f2);
        // NUMA bits differ.
        let numa_mask = 0xFu64 << NUMA_BUCKET_SHIFT;
        assert_ne!(f1 & numa_mask, f2 & numa_mask);
    }

    #[test]
    fn extract_features_base_test() {
        use crate::test_support::KtstrTestEntry;

        let entry = KtstrTestEntry::DEFAULT;
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let features = extract_features(&entry, &topo, false, "basic_test");
        // Should have bits set for scheduler hash, cpu bucket 0, llc bucket 0,
        // duration bucket 0, workers bucket 0, no gauntlet flag.
        assert_eq!(features & (1 << GAUNTLET_SHIFT), 0);
        // SMT off
        assert_eq!(features & (1 << SMT_SHIFT), 0);
        // performance_mode off
        assert_eq!(features & (1 << PERF_MODE_SHIFT), 0);
    }

    #[test]
    fn extract_features_smt_set() {
        let entry = KtstrTestEntry::DEFAULT;
        let topo = Topology {
            llcs: 2,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let features = extract_features(&entry, &topo, false, "smt_test");
        assert_ne!(features & (1 << SMT_SHIFT), 0);
    }

    #[test]
    fn extract_features_gauntlet() {
        let entry = KtstrTestEntry::DEFAULT;
        let topo = Topology {
            llcs: 4,
            cores_per_llc: 4,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let features = extract_features(&entry, &topo, true, "gauntlet_test");
        assert_ne!(features & (1 << GAUNTLET_SHIFT), 0);
    }

    #[test]
    fn estimate_duration_small_topo() {
        let entry = KtstrTestEntry {
            duration: std::time::Duration::from_secs(2),
            ..KtstrTestEntry::DEFAULT
        };
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        // boot_overhead = 10 + 0 = 10, duration = 2, settle = 2
        assert_eq!(estimate_duration(&entry, &topo), 14.0);
    }

    #[test]
    fn estimate_duration_large_topo() {
        let entry = KtstrTestEntry {
            duration: std::time::Duration::from_secs(5),
            ..KtstrTestEntry::DEFAULT
        };
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        // 252 CPUs: boot_overhead = 10 + (252-16)/10 = 10 + 23 = 33
        // duration = 5, settle = 2 -> 40.0
        assert_eq!(estimate_duration(&entry, &topo), 40.0);
    }

    #[test]
    fn estimate_duration_performance_mode() {
        let entry = KtstrTestEntry {
            duration: std::time::Duration::from_secs(2),
            performance_mode: true,
            ..KtstrTestEntry::DEFAULT
        };
        let topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        // boot_overhead = 10, duration = 2, settle = 2, perf = 3
        assert_eq!(estimate_duration(&entry, &topo), 17.0);
    }

    #[test]
    fn select_empty() {
        assert!(select(&[], 100.0).is_empty());
    }

    #[test]
    fn select_zero_budget() {
        let candidates = vec![TestCandidate {
            name: "t1".into(),
            features: 0xFF,
            estimated_secs: 10.0,
        }];
        assert!(select(&candidates, 0.0).is_empty());
    }

    #[test]
    fn select_single_fits() {
        let candidates = vec![TestCandidate {
            name: "t1".into(),
            features: 0xFF,
            estimated_secs: 10.0,
        }];
        let sel = select(&candidates, 20.0);
        assert_eq!(sel, vec![0]);
    }

    #[test]
    fn select_single_too_expensive() {
        let candidates = vec![TestCandidate {
            name: "t1".into(),
            features: 0xFF,
            estimated_secs: 30.0,
        }];
        let sel = select(&candidates, 20.0);
        assert!(sel.is_empty());
    }

    #[test]
    fn select_prefers_coverage_per_second() {
        // t1: 4 features, 20s -> ratio 0.2
        // t2: 2 features, 5s  -> ratio 0.4 (better)
        // With budget=25s, should pick t2 first, then t1 if it fits.
        let candidates = vec![
            TestCandidate {
                name: "t1".into(),
                features: 0b1111,
                estimated_secs: 20.0,
            },
            TestCandidate {
                name: "t2".into(),
                features: 0b110000,
                estimated_secs: 5.0,
            },
        ];
        let sel = select(&candidates, 25.0);
        assert_eq!(sel, vec![0, 1]); // both selected (sorted by index)
    }

    #[test]
    fn select_budget_constraint() {
        // Budget only fits one test. Both have 4 features at 15s each.
        // Equal ratio -> lexicographic tiebreak picks "t1".
        let candidates = vec![
            TestCandidate {
                name: "t1".into(),
                features: 0b1111,
                estimated_secs: 15.0,
            },
            TestCandidate {
                name: "t2".into(),
                features: 0b110000,
                estimated_secs: 15.0,
            },
        ];
        let sel = select(&candidates, 20.0);
        assert_eq!(sel, vec![0]);
    }

    #[test]
    fn select_marginal_coverage_decreases() {
        // t1 and t2 have identical features. After selecting t1,
        // t2 adds zero marginal coverage and should be skipped.
        let candidates = vec![
            TestCandidate {
                name: "t1".into(),
                features: 0b1111,
                estimated_secs: 5.0,
            },
            TestCandidate {
                name: "t2".into(),
                features: 0b1111,
                estimated_secs: 5.0,
            },
            TestCandidate {
                name: "t3".into(),
                features: 0b110000,
                estimated_secs: 5.0,
            },
        ];
        let sel = select(&candidates, 100.0);
        // Should select t1 and t3 (or t2 and t3), but not both t1 and t2
        assert_eq!(sel.len(), 2);
        assert!(sel.contains(&2)); // t3 always selected (unique features)
    }

    #[test]
    fn select_sorted_output() {
        let candidates = vec![
            TestCandidate {
                name: "t1".into(),
                features: 0b01,
                estimated_secs: 10.0,
            },
            TestCandidate {
                name: "t2".into(),
                features: 0b10,
                estimated_secs: 5.0,
            },
        ];
        let sel = select(&candidates, 100.0);
        // Output should be sorted by index
        assert_eq!(sel, vec![0, 1]);
    }

    #[test]
    fn djb2_hash_one_hot_4() {
        // 1 << (hash % 4) is always a single bit in the low 4 positions.
        for name in &["eevdf", "scx_mitosis", "scx_rusty", "", "x"] {
            let one_hot = 1u64 << (djb2_hash(name) % 4);
            assert_eq!(one_hot.count_ones(), 1);
            assert!(one_hot < 16);
        }
    }

    #[test]
    fn djb2_hash_different_names_differ() {
        let h1 = djb2_hash("eevdf");
        let h2 = djb2_hash("scx_mitosis");
        assert_ne!(h1, h2);
    }

    #[test]
    fn select_different_features_both_selected() {
        // Two tests with disjoint features both get selected.
        let candidates = vec![
            TestCandidate {
                name: "sched_a_test1".into(),
                features: 0b0001,
                estimated_secs: 5.0,
            },
            TestCandidate {
                name: "sched_b_test1".into(),
                features: 0b0010 | (1 << 3),
                estimated_secs: 5.0,
            },
        ];
        let sel = select(&candidates, 100.0);
        assert_eq!(sel, vec![0, 1]);
    }

    #[test]
    fn select_tie_broken_by_name() {
        // Two tests with identical features and cost. Lexicographic
        // tiebreak should pick "aaa" over "zzz".
        let candidates = vec![
            TestCandidate {
                name: "zzz".into(),
                features: 0b1111,
                estimated_secs: 10.0,
            },
            TestCandidate {
                name: "aaa".into(),
                features: 0b1111,
                estimated_secs: 10.0,
            },
        ];
        let sel = select(&candidates, 15.0);
        assert_eq!(sel, vec![1]); // "aaa" wins the tiebreak
    }

    #[test]
    fn selection_stats_basic() {
        let candidates = vec![
            TestCandidate {
                name: "t1".into(),
                features: 0b0011,
                estimated_secs: 5.0,
            },
            TestCandidate {
                name: "t2".into(),
                features: 0b1100,
                estimated_secs: 5.0,
            },
            TestCandidate {
                name: "t3".into(),
                features: 0b1111,
                estimated_secs: 5.0,
            },
        ];
        let sel = vec![0, 1];
        let stats = selection_stats(&candidates, &sel, 100.0);
        assert_eq!(stats.selected, 2);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.budget_used, 10.0);
        assert_eq!(stats.budget_total, 100.0);
        assert_eq!(stats.bits_covered, 4);
        assert_eq!(stats.bits_possible, 4);
    }

    #[test]
    fn estimate_duration_host_only() {
        let entry = KtstrTestEntry {
            duration: std::time::Duration::from_secs(5),
            host_only: true,
            ..KtstrTestEntry::DEFAULT
        };
        let topo = Topology {
            llcs: 14,
            cores_per_llc: 9,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        // host_only: no VM boot overhead, just duration + 2
        assert_eq!(estimate_duration(&entry, &topo), 7.0);
    }

    #[test]
    fn select_zero_cost_selected_first() {
        // A zero-cost test with unique features should always be selected.
        let candidates = vec![
            TestCandidate {
                name: "free".into(),
                features: 0b0001,
                estimated_secs: 0.0,
            },
            TestCandidate {
                name: "expensive".into(),
                features: 0b0010,
                estimated_secs: 100.0,
            },
        ];
        let sel = select(&candidates, 50.0);
        // "free" selected (zero cost), "expensive" doesn't fit
        assert_eq!(sel, vec![0]);
    }
}
