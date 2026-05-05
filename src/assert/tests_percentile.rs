//! Nearest-rank `percentile` helper: empty / single-element /
//! n=100 boundary / saturating-bounds / p50 odd-count cases plus
//! the `debug_assert!`-only sorted-input precondition pin.

use super::*;

// -- percentile: nearest-rank without off-by-one --

#[test]
fn percentile_empty_slice_is_zero() {
    assert_eq!(percentile(&[], 0.99), 0);
}

#[test]
fn percentile_single_element() {
    assert_eq!(percentile(&[42], 0.99), 42);
}

#[test]
fn percentile_p99_of_100_samples_is_element_98() {
    // Regression: previous formulation `ceil(n * 0.99)` returned
    // index 99 (the max) for n=100. The correct nearest-rank p99
    // of [0, 1, 2, ..., 99] is 98 — the 99th element 1-indexed.
    let sorted: Vec<u64> = (0..100).collect();
    assert_eq!(percentile(&sorted, 0.99), 98);
}

#[test]
fn percentile_p99_of_1000_samples_is_element_989() {
    let sorted: Vec<u64> = (0..1000).collect();
    assert_eq!(percentile(&sorted, 0.99), 989);
}

#[test]
fn percentile_saturates_into_bounds_for_small_n() {
    // For very small n, ceil(n * 0.99) may equal n, so the helper
    // must saturating_sub(1) and clamp to n-1 to stay in bounds.
    for n in 1u64..=10 {
        let sorted: Vec<u64> = (0..n).collect();
        let v = percentile(&sorted, 0.99);
        assert!(v < n, "percentile({sorted:?}, 0.99)={v} must be < n ({n})");
    }
}

#[test]
fn percentile_p50_on_odd_count_is_middle() {
    // p50 of [0..9] at nearest-rank: ceil(9 * 0.5) - 1 = 4.
    let sorted: Vec<u64> = (0..9).collect();
    assert_eq!(percentile(&sorted, 0.50), 4);
}

/// Pins the debug-build precondition that callers pass sorted input.
/// Skipped under release because the contract is enforced via
/// `debug_assert!` only — production paths sort upstream and the
/// release build deliberately omits the linear-scan check.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "percentile() requires sorted input")]
fn percentile_unsorted_input_panics_in_debug() {
    let unsorted: Vec<u64> = vec![5, 1, 3];
    let _ = percentile(&unsorted, 0.99);
}

/// Two-element non-decreasing slices (equal pair, increasing pair)
/// must NOT trip the debug_assert — the precondition is "sorted",
/// not "strictly increasing." Pins the correct comparator (`<=`)
/// so a regression switching to `<` would fail every cgroup with
/// repeated wake-latency samples.
#[test]
fn percentile_equal_consecutive_values_do_not_panic() {
    let with_dups: Vec<u64> = vec![1, 1, 1, 2, 2, 3];
    let _ = percentile(&with_dups, 0.99);
}
