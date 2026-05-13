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
//   Bit  21:     reserved (formerly workers_per_cgroup bucket)
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

/// Bit-width of a `1u64 << (djb2_hash(name) % 4)` one-hot value:
/// 4 distinct bit positions (0..=3), so 4 bits. cfg(test)-only
/// because the only consumer is the overlap test that derives
/// multi_bit_ranges; production extract_features uses the SHIFT
/// constants directly.
#[cfg(test)]
const fn sched_hash_bits() -> u32 {
    4
}

/// Bit-width of `name_hash` one-hot value. Same `% 4` modulus as
/// `sched_hash_bits` — kept as a separate fn so the SHIFT
/// enumeration cross-reference at the overlap test reads each
/// width from its own field-named source.
#[cfg(test)]
const fn name_hash_bits() -> u32 {
    4
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

/// Bit-width of `cpu_bucket`'s output: 5 one-hot positions (0..=4).
#[cfg(test)]
const fn cpu_bucket_bits() -> u32 {
    5
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

/// Bit-width of `llc_bucket`'s output: 5 one-hot positions (0..=4).
#[cfg(test)]
const fn llc_bucket_bits() -> u32 {
    5
}

/// Classify duration (seconds) into a one-hot bit (3 classes).
fn duration_bucket(duration_secs: u64) -> u64 {
    match duration_secs {
        0..=2 => 1 << 0,
        3..=10 => 1 << 1,
        _ => 1 << 2,
    }
}

/// Bit-width of `duration_bucket`'s output: 3 one-hot positions (0..=2).
#[cfg(test)]
const fn duration_bucket_bits() -> u32 {
    3
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

/// Bit-width of `numa_bucket`'s output: 4 one-hot positions (0..=3).
#[cfg(test)]
const fn numa_bucket_bits() -> u32 {
    4
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

    // Build-script-generated authoritative list of every
    // `const *_SHIFT: u32 = N;` declaration in this file. Drives
    // the `all_shifts_classified_in_exactly_one_enumeration` test
    // so a new SHIFT cannot be added without being classified.
    include!(concat!(env!("OUT_DIR"), "/shift_registry.rs"));

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

    /// Bit-shift orthogonality + ground-truth pin: each one-bit
    /// flag must land at its canonical bit position AND must set
    /// ONLY that bit (no overlap with any other one-bit flag).
    ///
    /// Today's sibling tests (`extract_features_smt_set`,
    /// `extract_features_gauntlet`) verify each bit fires when its
    /// trigger is active but read both sides through the same
    /// `*_SHIFT` constants — a coherent swap (SMT_SHIFT ↔
    /// PERF_MODE_SHIFT swapped) passes both because writer + reader
    /// move together. An overlap (one shift constant coincidentally
    /// equal to another) silently degrades the feature vector's
    /// one-hot guarantee.
    ///
    /// This test pins each one-bit shift against a literal ground
    /// truth (catches swap + reassignment) AND asserts the
    /// triggered features bit fires while every OTHER one-bit
    /// flag stays clear by NAME-keyed comparison (catches overlap;
    /// a value-keyed skip would silently exempt the colliding
    /// neighbor along with the trigger itself).
    #[test]
    fn extract_features_bit_shifts_orthogonal() {
        // Ground-truth pin: literal bit positions for every
        // one-bit flag. If any *_SHIFT constant changes value, the
        // change must land here too — making the renumber an
        // explicit decision rather than a silent drift.
        assert_eq!(SMT_SHIFT, 14, "SMT_SHIFT pinned");
        assert_eq!(PERF_MODE_SHIFT, 15, "PERF_MODE_SHIFT pinned");
        assert_eq!(HOST_ONLY_SHIFT, 16, "HOST_ONLY_SHIFT pinned");
        assert_eq!(EXPECT_ERR_SHIFT, 17, "EXPECT_ERR_SHIFT pinned");
        assert_eq!(GAUNTLET_SHIFT, 22, "GAUNTLET_SHIFT pinned");

        // Plain non-SMT topo so the SMT bit is naturally clear in
        // every "trigger isolated" path. SMT requires its own
        // topo (threads_per_core > 1).
        let plain_topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let smt_topo = Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 2,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };

        // List every one-bit feature flag. Each row: (shift,
        // human name). DURATION/CPU/LLC/NUMA/SCHED/NAME_HASH are
        // multi-bit one-hot fields out of scope here.
        let one_bit_shifts: &[(u32, &str)] = &[
            (SMT_SHIFT, "SMT"),
            (PERF_MODE_SHIFT, "PERF_MODE"),
            (HOST_ONLY_SHIFT, "HOST_ONLY"),
            (EXPECT_ERR_SHIFT, "EXPECT_ERR"),
            (GAUNTLET_SHIFT, "GAUNTLET"),
        ];

        // Helper: assert ONLY the named one-bit flag fires; every
        // other one-bit flag stays clear. Skip by NAME (identity),
        // not by shift VALUE — a value-keyed skip would exempt any
        // neighbor that coincidentally shares the trigger's shift,
        // silently masking the overlap regression this test exists
        // to catch.
        let assert_only = |features: u64, trigger_label: &str| {
            let trigger_shift = one_bit_shifts
                .iter()
                .find(|(_, name)| *name == trigger_label)
                .map(|(shift, _)| *shift)
                .expect("trigger_label must be present in one_bit_shifts");
            assert_ne!(
                features & (1 << trigger_shift),
                0,
                "{trigger_label}: triggered bit must be set",
            );
            for &(other_shift, other_name) in one_bit_shifts {
                if other_name == trigger_label {
                    continue;
                }
                assert_eq!(
                    features & (1 << other_shift),
                    0,
                    "{trigger_label} trigger must leave {other_name} bit \
                     (shift {other_shift}) clear — collision indicates \
                     a shift-overlap regression",
                );
            }
        };

        // SMT-only: plain entry, SMT topo. Only SMT bit fires
        // among one-bit flags.
        let smt_f = extract_features(&KtstrTestEntry::DEFAULT, &smt_topo, false, "smt_test");
        assert_only(smt_f, "SMT");

        // PERF_MODE-only: performance_mode entry, plain (non-SMT)
        // topo, is_gauntlet=false.
        let perf_entry = KtstrTestEntry {
            performance_mode: true,
            ..KtstrTestEntry::DEFAULT
        };
        let perf_f = extract_features(&perf_entry, &plain_topo, false, "perf_test");
        assert_only(perf_f, "PERF_MODE");

        // HOST_ONLY-only: host_only entry, plain topo.
        let host_only_entry = KtstrTestEntry {
            host_only: true,
            ..KtstrTestEntry::DEFAULT
        };
        let host_only_f = extract_features(&host_only_entry, &plain_topo, false, "host_test");
        assert_only(host_only_f, "HOST_ONLY");

        // EXPECT_ERR-only: expect_err entry, plain topo.
        let expect_err_entry = KtstrTestEntry {
            expect_err: true,
            ..KtstrTestEntry::DEFAULT
        };
        let expect_err_f =
            extract_features(&expect_err_entry, &plain_topo, false, "expect_err_test");
        assert_only(expect_err_f, "EXPECT_ERR");

        // GAUNTLET-only: plain entry, plain topo, is_gauntlet=true.
        let gauntlet_f =
            extract_features(&KtstrTestEntry::DEFAULT, &plain_topo, true, "gauntlet_test");
        assert_only(gauntlet_f, "GAUNTLET");
    }

    /// One-bit-vs-multi-bit shift overlap guard.
    ///
    /// `extract_features_bit_shifts_orthogonal` catches collisions
    /// between two one-bit shifts but reads multi-bit fields through
    /// the same `*_SHIFT` constants the writer uses, so a one-bit
    /// shift that accidentally lands INSIDE a multi-bit field's
    /// range (e.g. `PERF_MODE_SHIFT = 18` collides with the
    /// 3-bit DURATION field at `[18..=20]`) would fire on every path
    /// — making the orthogonality test pass vacuously while the
    /// feature vector silently drops a coverage axis.
    ///
    /// Widths match the bit layout doc at the top of this module and
    /// the bucket helpers in this file:
    /// - SCHED 4 bits (1u64 << (hash % 4))
    /// - CPU_BUCKET 5 bits (cpu_bucket: 1<<0..=1<<4)
    /// - LLC_BUCKET 5 bits (llc_bucket: 1<<0..=1<<4)
    /// - DURATION 3 bits (duration_bucket: 1<<0..=1<<2)
    /// - NAME_HASH 4 bits (1u64 << (hash % 4))
    /// - NUMA_BUCKET 4 bits (numa_bucket: 1<<0..=1<<3)
    #[test]
    fn extract_features_one_bit_shifts_outside_multi_bit_ranges() {
        // Each (shift_start, shift_end_inclusive, name) entry is a
        // multi-bit field's bit range. One-bit shifts must NOT fall
        // inside any of these ranges. Widths are read from each
        // bucket fn's sibling _bits() const so growing a bucket's
        // match arms forces an explicit width update at one site
        // rather than diverging silently from this test's
        // expectations.
        let multi_bit_ranges: &[(u32, u32, &str)] = &[
            (SCHED_SHIFT, SCHED_SHIFT + sched_hash_bits() - 1, "SCHED"),
            (
                CPU_BUCKET_SHIFT,
                CPU_BUCKET_SHIFT + cpu_bucket_bits() - 1,
                "CPU_BUCKET",
            ),
            (
                LLC_BUCKET_SHIFT,
                LLC_BUCKET_SHIFT + llc_bucket_bits() - 1,
                "LLC_BUCKET",
            ),
            (
                DURATION_SHIFT,
                DURATION_SHIFT + duration_bucket_bits() - 1,
                "DURATION",
            ),
            (
                NAME_HASH_SHIFT,
                NAME_HASH_SHIFT + name_hash_bits() - 1,
                "NAME_HASH",
            ),
            (
                NUMA_BUCKET_SHIFT,
                NUMA_BUCKET_SHIFT + numa_bucket_bits() - 1,
                "NUMA_BUCKET",
            ),
        ];
        let one_bit_shifts: &[(u32, &str)] = &[
            (SMT_SHIFT, "SMT"),
            (PERF_MODE_SHIFT, "PERF_MODE"),
            (HOST_ONLY_SHIFT, "HOST_ONLY"),
            (EXPECT_ERR_SHIFT, "EXPECT_ERR"),
            (GAUNTLET_SHIFT, "GAUNTLET"),
        ];
        for &(shift, name) in one_bit_shifts {
            for &(start, end, range_name) in multi_bit_ranges {
                assert!(
                    shift < start || shift > end,
                    "{name} shift={shift} falls inside multi-bit field {range_name} \
                     range [{start}..={end}] — overlap would let {range_name} \
                     silently set the {name} bit"
                );
            }
        }
    }

    /// Exhaustive-classification guard for `*_SHIFT` constants.
    ///
    /// `extract_features_bit_shifts_orthogonal` enumerates the
    /// one-bit shifts and `extract_features_one_bit_shifts_outside_multi_bit_ranges`
    /// enumerates the multi-bit shifts, but both are hand-maintained
    /// — adding a new `*_SHIFT: u32 = N;` constant without updating
    /// either list would leave the new shift unguarded by the
    /// orthogonality / overlap tests.
    ///
    /// `ALL_SHIFTS` is generated by build.rs from a text scan of
    /// `src/budget.rs` and lists every matching constant. This test
    /// takes the value-union of the two hand-enumerations and asserts
    /// it equals the build-script-generated set. A new constant fails
    /// loudly here until it is classified into one of the two lists.
    #[test]
    fn all_shifts_classified_in_exactly_one_enumeration() {
        use std::collections::HashSet;

        let one_bit_values: HashSet<u32> = [
            SMT_SHIFT,
            PERF_MODE_SHIFT,
            HOST_ONLY_SHIFT,
            EXPECT_ERR_SHIFT,
            GAUNTLET_SHIFT,
        ]
        .into_iter()
        .collect();
        let multi_bit_values: HashSet<u32> = [
            SCHED_SHIFT,
            CPU_BUCKET_SHIFT,
            LLC_BUCKET_SHIFT,
            DURATION_SHIFT,
            NAME_HASH_SHIFT,
            NUMA_BUCKET_SHIFT,
        ]
        .into_iter()
        .collect();

        let classified: HashSet<u32> = one_bit_values
            .union(&multi_bit_values)
            .copied()
            .collect();
        let registry: HashSet<u32> = ALL_SHIFTS.iter().map(|(v, _)| *v).collect();

        let unclassified: Vec<(u32, &str)> = ALL_SHIFTS
            .iter()
            .filter(|(v, _)| !classified.contains(v))
            .copied()
            .collect();
        // Split phantom check by source enumeration so the failure
        // message tells the contributor WHICH list carries the stale
        // value — saves a grep across both extract_features_*
        // tests.
        let phantom_one_bit: Vec<u32> = one_bit_values.difference(&registry).copied().collect();
        let phantom_multi_bit: Vec<u32> = multi_bit_values.difference(&registry).copied().collect();
        let overlap: Vec<u32> = one_bit_values
            .intersection(&multi_bit_values)
            .copied()
            .collect();

        assert!(
            unclassified.is_empty(),
            "unclassified *_SHIFT constants (present in build-script \
             scan, absent from both test enumerations): {unclassified:?}. \
             Add to extract_features_bit_shifts_orthogonal's one_bit_shifts \
             (if a single-bit flag) or extract_features_one_bit_shifts_outside_multi_bit_ranges's \
             multi_bit_ranges (if a multi-bit field).",
        );
        assert!(
            phantom_one_bit.is_empty() && phantom_multi_bit.is_empty(),
            "test enumerations reference shift values that no \
             `const *_SHIFT: u32 = N;` in src/budget.rs declares — stale \
             enumeration entries. one_bit_shifts has phantom values \
             {phantom_one_bit:?} (remove from \
             extract_features_bit_shifts_orthogonal); multi_bit_ranges \
             has phantom values {phantom_multi_bit:?} (remove from \
             extract_features_one_bit_shifts_outside_multi_bit_ranges).",
        );
        assert!(
            overlap.is_empty(),
            "*_SHIFT constants classified as BOTH one-bit and multi-bit: \
             {overlap:?} — each shift must belong to exactly one \
             enumeration. Remove from either \
             extract_features_bit_shifts_orthogonal's one_bit_shifts \
             (if it is a multi-bit field) or \
             extract_features_one_bit_shifts_outside_multi_bit_ranges's \
             multi_bit_ranges (if it is a single-bit flag).",
        );
    }

    /// Each bucket fn's max output value must fit in the bit-width
    /// it advertises via its sibling `_bits()` const. Catches the
    /// silent-overlap regression where a bucket fn grows a match
    /// arm (e.g. `129..=256 => 1 << 5`) without bumping the
    /// `_bits()` value — the overlap test
    /// (extract_features_one_bit_shifts_outside_multi_bit_ranges)
    /// would otherwise compute a stale range and let the new
    /// high-order bit bleed into the neighbouring field.
    ///
    /// The probe values cover every match-arm boundary in the
    /// bucket fn (one per case) so the max is sampled accurately
    /// rather than approximated.
    #[test]
    fn cpu_bucket_fits_advertised_width() {
        // PROBE-SET INVARIANT: every cpu_bucket match arm has ≥1
        // probe value in its range. When adding a new arm, ADD a
        // probe value that lands inside it — otherwise the new
        // arm's bit position may not be sampled and this assertion
        // stays silent on width regressions.
        let max = [0u32, 8, 9, 16, 17, 64, 65, 128, 129, u32::MAX]
            .iter()
            .map(|&x| cpu_bucket(x))
            .max()
            .expect("non-empty");
        let bits_used = 64 - max.leading_zeros();
        assert!(
            bits_used <= cpu_bucket_bits(),
            "cpu_bucket max output {max:#b} uses {bits_used} bits but \
             advertised {} bits — grow cpu_bucket_bits() or shrink the bucket",
            cpu_bucket_bits(),
        );
    }

    #[test]
    fn llc_bucket_fits_advertised_width() {
        // PROBE-SET INVARIANT: every llc_bucket match arm has ≥1
        // probe value in its range. When adding a new arm, ADD a
        // probe value that lands inside it.
        let max = [0u32, 1, 2, 3, 4, 5, 8, 9, u32::MAX]
            .iter()
            .map(|&x| llc_bucket(x))
            .max()
            .expect("non-empty");
        let bits_used = 64 - max.leading_zeros();
        assert!(
            bits_used <= llc_bucket_bits(),
            "llc_bucket max output {max:#b} uses {bits_used} bits but \
             advertised {} bits — grow llc_bucket_bits() or shrink the bucket",
            llc_bucket_bits(),
        );
    }

    #[test]
    fn duration_bucket_fits_advertised_width() {
        // PROBE-SET INVARIANT: every duration_bucket match arm has
        // ≥1 probe value in its range. When adding a new arm, ADD
        // a probe value that lands inside it.
        let max = [0u64, 2, 3, 10, 11, u64::MAX]
            .iter()
            .map(|&x| duration_bucket(x))
            .max()
            .expect("non-empty");
        let bits_used = 64 - max.leading_zeros();
        assert!(
            bits_used <= duration_bucket_bits(),
            "duration_bucket max output {max:#b} uses {bits_used} bits but \
             advertised {} bits — grow duration_bucket_bits() or shrink the bucket",
            duration_bucket_bits(),
        );
    }

    #[test]
    fn numa_bucket_fits_advertised_width() {
        // PROBE-SET INVARIANT: every numa_bucket match arm has ≥1
        // probe value in its range. When adding a new arm, ADD a
        // probe value that lands inside it.
        let max = [0u32, 1, 2, 3, 4, 5, u32::MAX]
            .iter()
            .map(|&x| numa_bucket(x))
            .max()
            .expect("non-empty");
        let bits_used = 64 - max.leading_zeros();
        assert!(
            bits_used <= numa_bucket_bits(),
            "numa_bucket max output {max:#b} uses {bits_used} bits but \
             advertised {} bits — grow numa_bucket_bits() or shrink the bucket",
            numa_bucket_bits(),
        );
    }

    /// Hash one-hot widths advertised by `sched_hash_bits()` and
    /// `name_hash_bits()` must match the actual `1u64 << (hash % 4)`
    /// shape used at both call sites in `extract_features`
    /// (scheduler-name hash shifted by SCHED_SHIFT, test-name hash
    /// shifted by NAME_HASH_SHIFT). Probe every hash % 4 value
    /// (0..=3) — the max bit position is 3 → 4 bits used.
    #[test]
    fn hash_bits_match_one_hot_shape() {
        let max_hash_value = (0u32..=3).map(|h| 1u64 << h).max().expect("non-empty");
        let bits_used = 64 - max_hash_value.leading_zeros();
        assert!(
            bits_used <= sched_hash_bits(),
            "sched_hash max output {max_hash_value:#b} uses {bits_used} bits but \
             advertised {} bits",
            sched_hash_bits(),
        );
        assert!(
            bits_used <= name_hash_bits(),
            "name_hash max output {max_hash_value:#b} uses {bits_used} bits but \
             advertised {} bits",
            name_hash_bits(),
        );
    }

    /// Two `*_SHIFT` constants sharing the same numeric value would
    /// collapse to one entry in the HashSet<u32> registry used by
    /// `all_shifts_classified_in_exactly_one_enumeration`,
    /// silently masking the duplicate. Compare slice-cardinality
    /// (which counts duplicates) against HashSet-cardinality
    /// (which dedups) to surface the collision.
    #[test]
    fn all_shift_values_unique() {
        use std::collections::HashSet;
        let distinct: HashSet<u32> = ALL_SHIFTS.iter().map(|(v, _)| *v).collect();
        assert_eq!(
            distinct.len(),
            ALL_SHIFTS.len(),
            "duplicate SHIFT values detected — two `const *_SHIFT: u32 = N;` \
             declarations in src/budget.rs share the same numeric value. \
             ALL_SHIFTS entries: {:?}",
            ALL_SHIFTS,
        );
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
