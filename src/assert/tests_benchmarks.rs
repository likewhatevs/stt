//! `assert_benchmarks` and `AssertPlan` benchmarking-path tests:
//! p99 / CV / iteration-rate thresholds, the ns-vs-µs unit
//! invariant, wake-latency populate paths in `assert_not_starved`,
//! schedstat run-delay aggregation, and the `assert_cgroup`
//! migration-ratio gate.

use super::tests_common::{rpt, rpt_with_latencies};
use super::*;

#[test]
fn assert_benchmarks_empty_reports() {
    // Empty reports → skip (passed stays true for gate-compat, but
    // `skipped` is set and a detail with DetailKind::Skip carries
    // the reason). The thresholds supplied here cannot be evaluated
    // against zero signal, so a silent pass would mask a broken run.
    let r = assert_benchmarks(&[], Some(1000), Some(0.5), Some(100.0));
    assert!(r.passed, "skip keeps passed=true for gate-compat");
    assert!(r.skipped, "no reports must surface as skipped");
    assert!(
        r.details
            .iter()
            .any(|d| matches!(d.kind, DetailKind::Skip) && d.message.contains("no worker reports")),
        "skip detail must carry the 'no worker reports' reason: {:?}",
        r.details,
    );
}

#[test]
fn assert_benchmarks_no_thresholds() {
    let reports = [rpt_with_latencies(
        1,
        vec![1000, 2000, 3000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, None, None, None);
    assert!(r.passed);
}

#[test]
fn assert_benchmarks_p99_pass() {
    let reports = [rpt_with_latencies(
        1,
        vec![100, 200, 300, 400, 500],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, Some(1000), None, None);
    assert!(r.passed, "p99 500ns < 1000ns limit: {:?}", r.details);
}

#[test]
fn assert_benchmarks_p99_n100_at_limit_passes() {
    // With samples [0..100], the nearest-rank p99 is 98
    // (sorted[ceil(100*0.99) - 1] = sorted[98]). Setting the
    // limit to 99 must pass (98 <= 99). An off-by-one that
    // returns sorted[99] = 99 would pass the same limit for
    // the wrong reason — the paired _fail test below pins
    // down the correct index.
    let latencies: Vec<u64> = (0..100).collect();
    let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
    let r = assert_benchmarks(&reports, Some(99), None, None);
    assert!(
        r.passed,
        "p99 should be 98, under limit 99: {:?}",
        r.details
    );
}

#[test]
fn assert_benchmarks_p99_n100_below_old_p100_passes() {
    // Tighter regression: with samples [0..100], set the limit to
    // 98. Correct p99 (98) equals the limit and passes (strict
    // `p99 > p99_limit` comparison). The old off-by-one returned
    // 99, which would have FAILED (99 > 98). This test therefore
    // only passes with the corrected index.
    let latencies: Vec<u64> = (0..100).collect();
    let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
    let r = assert_benchmarks(&reports, Some(98), None, None);
    assert!(
        r.passed,
        "corrected p99 (98) must equal limit 98 and pass: {:?}",
        r.details
    );
}

#[test]
fn assert_not_starved_p99_n100_is_99_microseconds() {
    // assert_not_starved exposes p99 as microseconds via
    // ScenarioStats. Samples = [1000, 2000, ..., 100_000] ns
    // (100 values at kilo-ns spacing) so the reported p99 is
    // exactly 99.0us with the correct index
    // (sorted[ceil(100*0.99) - 1] = sorted[98] = 99_000ns = 99us).
    // An off-by-one that returns sorted[99] would yield 100us.
    let latencies: Vec<u64> = (1..=100).map(|v: u64| v * 1000).collect();
    let reports = [rpt_with_latencies(1, latencies, 100, 5_000_000_000)];
    let r = assert_not_starved(&reports);
    assert_eq!(
        r.stats.worst_p99_wake_latency_us, 99.0,
        "p99 must equal 99.0us (sorted[98] = 99_000ns), got {}us",
        r.stats.worst_p99_wake_latency_us
    );
}

#[test]
fn assert_benchmarks_p99_fail() {
    let reports = [rpt_with_latencies(
        1,
        vec![100, 200, 300, 400, 2000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, Some(1000), None, None);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("p99 wake latency")));
}

/// Unit-boundary pin: the `max_p99_wake_latency_ns` threshold
/// MUST be compared against `WorkerReport::resume_latencies_ns`
/// (nanoseconds) — never against the microsecond-valued
/// `CgroupStats::p99_wake_latency_us` field. A regression that
/// divided either side by 1000 (or multiplied by 1000) would
/// make the threshold fire 1000× too often or 1000× too rarely,
/// silently corrupting every regression gate that uses this
/// field.
///
/// Construction: plant `resume_latencies_ns` values that are
/// clearly in the NS scale (e.g. 5000 ns = 5 µs) and set a
/// threshold of 4999 ns. The assertion must FAIL at 4999 ns and
/// PASS at 5001 ns. If the comparison were accidentally
/// converting the threshold to µs (dividing by 1000), 4999
/// would behave like "4.999 µs threshold against a 5 µs p99"
/// — technically still a fail but for the wrong reason. The
/// bracket here (5000-1 vs 5000+1) sits inside the 1000× slop
/// so a unit-swap regression would flip the verdict on one of
/// the two cases.
#[test]
fn assert_p99_ns_threshold_compares_against_ns_latencies() {
    // Single-sample latency set: p99 == the sample value.
    let reports = [rpt_with_latencies(1, vec![5000], 10, 5_000_000_000)];

    // Threshold just below the 5000 ns sample -> FAIL.
    let fail = assert_benchmarks(&reports, Some(4999), None, None);
    assert!(
        !fail.passed,
        "threshold 4999 ns against 5000 ns p99 must fail — if this \
         passes, the comparison may be converting to µs and eating \
         3 digits of resolution",
    );

    // Threshold just above the 5000 ns sample -> PASS.
    let pass = assert_benchmarks(&reports, Some(5001), None, None);
    assert!(
        pass.passed,
        "threshold 5001 ns against 5000 ns p99 must pass — if this \
         fails, the comparison may be multiplying the threshold by \
         1000 (treating it as µs)",
    );

    // Cross-check the reporting path: `assert_not_starved`
    // populates `worst_p99_wake_latency_us` in MICROSECONDS
    // (ns / 1000). A regression that conflated the reporting
    // field with the threshold input would surface as either
    // `us == ns` (forgot to divide) or `us == ns/1_000_000`
    // (double-converted).
    let stats = assert_not_starved(&reports);
    assert_eq!(
        stats.stats.worst_p99_wake_latency_us, 5.0,
        "5000 ns / 1000 = 5.0 µs — if this renders as 5000 (forgot /1000) \
         or 0.005 (extra /1000), the reporting-path unit conversion drifted",
    );
}

#[test]
fn assert_benchmarks_cv_pass() {
    // All same latency -> CV = 0.
    let reports = [rpt_with_latencies(
        1,
        vec![1000, 1000, 1000, 1000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, None, Some(0.5), None);
    assert!(r.passed, "uniform latencies CV=0: {:?}", r.details);
}

#[test]
fn assert_benchmarks_cv_fail() {
    // High variance latencies.
    let reports = [rpt_with_latencies(
        1,
        vec![100, 100, 100, 100000],
        10,
        5_000_000_000,
    )];
    let r = assert_benchmarks(&reports, None, Some(0.5), None);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("wake latency CV")));
}

#[test]
fn assert_benchmarks_iteration_rate_pass() {
    // 1000 iterations in 5 seconds = 200/s, above 100/s floor.
    let reports = [rpt_with_latencies(1, vec![], 1000, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(r.passed, "200/s > 100/s floor: {:?}", r.details);
}

#[test]
fn assert_benchmarks_iteration_rate_fail() {
    // 10 iterations in 5 seconds = 2/s, below 100/s floor.
    let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("iteration rate")));
}

#[test]
fn assert_benchmarks_zero_wall_time_skips_rate() {
    let reports = [rpt_with_latencies(1, vec![], 10, 0)];
    let r = assert_benchmarks(&reports, None, None, Some(100.0));
    assert!(r.passed, "zero wall_time should skip rate check");
}

#[test]
fn assert_benchmarks_no_latencies_skips_p99() {
    let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, Some(1000), None, None);
    assert!(r.passed, "empty latencies should skip p99 check");
}

#[test]
fn assert_benchmarks_single_latency_cv_skipped() {
    // Single sample -> len < 2, CV check skipped.
    let reports = [rpt_with_latencies(1, vec![1000], 10, 5_000_000_000)];
    let r = assert_benchmarks(&reports, None, Some(0.1), None);
    assert!(r.passed, "single sample should skip CV check");
}

// -- wake latency stats in assert_not_starved --

#[test]
fn not_starved_wake_latency_stats() {
    let reports = [
        rpt_with_latencies(1, vec![1000, 2000, 3000, 4000, 5000], 100, 5_000_000_000),
        rpt_with_latencies(2, vec![6000, 7000, 8000, 9000, 10000], 200, 5_000_000_000),
    ];
    let r = assert_not_starved(&reports);
    assert!(r.passed, "{:?}", r.details);
    let s = &r.stats;
    // p99 of [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000] in us:
    // sorted, percentile index = ceil(10*0.99) - 1 = 9 -> sorted[9] = 10000ns = 10.0us
    assert!(
        s.worst_p99_wake_latency_us > 9.0,
        "p99: {}",
        s.worst_p99_wake_latency_us
    );
    // median of 10 samples via `percentile(sorted, 0.5)`:
    // nearest-rank index = ceil(10 * 0.5) - 1 = 4 →
    // sorted[4] = 5000ns = 5.0us. The lower-of-two-middles
    // bound matches the convention documented on
    // `CgroupStats::median_wake_latency_us`.
    assert!(
        (s.worst_median_wake_latency_us - 5.0).abs() < 0.1,
        "median: {}",
        s.worst_median_wake_latency_us
    );
    assert!(
        s.worst_wake_latency_cv > 0.0,
        "cv: {}",
        s.worst_wake_latency_cv
    );
    assert_eq!(s.total_iterations, 300);
}

#[test]
fn not_starved_empty_latencies_zero_stats() {
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
    let r = assert_not_starved(&reports);
    assert!(r.passed);
    assert_eq!(r.stats.worst_p99_wake_latency_us, 0.0);
    assert_eq!(r.stats.worst_median_wake_latency_us, 0.0);
    assert_eq!(r.stats.worst_wake_latency_cv, 0.0);
}

#[test]
fn not_starved_run_delay_stats() {
    let mut w1 = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50);
    w1.schedstat_run_delay_ns = 100_000; // 100us
    let mut w2 = rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50);
    w2.schedstat_run_delay_ns = 300_000; // 300us
    let r = assert_not_starved(&[w1, w2]);
    assert!(r.passed, "{:?}", r.details);
    // mean_run_delay = (100 + 300) / 2 = 200us
    assert!(
        (r.stats.worst_mean_run_delay_us - 200.0).abs() < 0.1,
        "mean: {}",
        r.stats.worst_mean_run_delay_us
    );
    // worst_run_delay = 300us
    assert!(
        (r.stats.worst_run_delay_us - 300.0).abs() < 0.1,
        "worst: {}",
        r.stats.worst_run_delay_us
    );
}

// -- AssertPlan benchmarking integration --

#[test]
fn plan_benchmarks_p99_via_assert_cgroup() {
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: Some(500),
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let reports = [rpt_with_latencies(
        1,
        vec![100, 200, 300, 400, 1000],
        10,
        5_000_000_000,
    )];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.passed, "p99 1000ns > 500ns limit");
    assert!(r.details.iter().any(|d| d.contains("p99 wake latency")));
}

#[test]
fn plan_migration_ratio_gate() {
    let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
    w.migration_count = 10;
    w.iterations = 100;
    // ratio = 10/100 = 0.10, threshold 0.05 → fail
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: Some(0.05),
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[w], None, None);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("migration ratio")));
}

#[test]
fn plan_migration_ratio_gate_pass() {
    let mut w = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50);
    w.migration_count = 2;
    w.iterations = 100;
    // ratio = 2/100 = 0.02, threshold 0.05 → pass
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: None,
        max_migration_ratio: Some(0.05),
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let r = plan.assert_cgroup(&[w], None, None);
    assert!(r.passed, "{:?}", r.details);
}

#[test]
fn plan_benchmarks_iteration_rate_via_assert_cgroup() {
    let plan = AssertPlan {
        not_starved: false,
        isolation: false,
        max_gap_ms: None,
        max_spread_pct: None,
        max_throughput_cv: None,
        min_work_rate: None,
        max_p99_wake_latency_ns: None,
        max_wake_latency_cv: None,
        min_iteration_rate: Some(1000.0),
        max_migration_ratio: None,
        min_page_locality: None,
        max_cross_node_migration_ratio: None,
        max_slow_tier_ratio: None,
    };
    let reports = [rpt_with_latencies(1, vec![], 10, 5_000_000_000)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.passed, "2/s < 1000/s floor");
    assert!(r.details.iter().any(|d| d.contains("iteration rate")));
}

#[test]
fn assert_throughput_parity_all_zero_cpu_time_fails_when_cv_set() {
    // When every worker recorded zero cpu_time the per-worker rate
    // is zero, the mean is zero, and CV is mathematically
    // undefined. The previous gate (`mean > 0.0`) silently skipped
    // the check and reported a pass — masking a workload that
    // never accumulated any CPU time. The fix surfaces it as a
    // failure so the operator sees the broken run.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 0;
    let r = assert_throughput_parity(&[a, b], Some(0.5), None);
    assert!(!r.passed, "all-zero cpu_time must fail when max_cv set");
    assert!(
        r.details.iter().any(|d| d.contains("CV undefined")),
        "diagnostic must surface the undefined-CV root cause: {:?}",
        r.details
    );
}

#[test]
fn assert_throughput_parity_all_zero_cpu_time_passes_without_cv() {
    // No CV check requested → no failure. The min_rate floor is
    // also unset, so the function has nothing to evaluate and
    // passes. This pins the gate scope: the new failure is
    // specific to the configured-CV-with-zero-mean case.
    let mut a = rpt(1, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    let mut b = rpt(2, 0, 5_000_000_000, 5_000_000_000, &[0], 0);
    a.cpu_time_ns = 0;
    b.cpu_time_ns = 0;
    let r = assert_throughput_parity(&[a, b], None, None);
    assert!(r.passed, "no CV configured → no failure: {:?}", r.details);
}
