//! Worker-level checks: `assert_not_starved`, `assert_isolation`,
//! gap / spread / stuck classification, single-worker passthroughs,
//! and negative-diagnostic-message tests that pin the
//! human-readable strings every consumer greps for.

use super::tests_common::rpt;
use super::*;

#[test]
fn healthy_pass() {
    let r = assert_not_starved(&[
        rpt(1, 1000, 5_000_000_000, 500_000_000, &[0, 1], 50),
        rpt(2, 1000, 5_000_000_000, 600_000_000, &[0, 1], 60),
        rpt(3, 1000, 5_000_000_000, 550_000_000, &[0, 1], 45),
    ]);
    assert!(r.passed, "{:?}", r.details);
}

#[test]
fn starved_fail() {
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 0, 5e9 as u64, 5e9 as u64, &[0], 50),
    ]);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("starved")));
}

#[test]
fn unfair_spread_fail() {
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50), // 10%
        rpt(2, 500, 5e9 as u64, 4e9 as u64, &[0, 1], 50),  // 80%
        rpt(3, 800, 5e9 as u64, 2e9 as u64, &[0, 1], 50),  // 40%
    ]);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("unfair")));
}

#[test]
fn fair_oversubscribed_pass() {
    let r = assert_not_starved(&[
        rpt(1, 100, 5e9 as u64, (3.75e9) as u64, &[0], 50),
        rpt(2, 100, 5e9 as u64, (3.70e9) as u64, &[0], 50),
        rpt(3, 100, 5e9 as u64, (3.80e9) as u64, &[0], 50),
        rpt(4, 100, 5e9 as u64, (3.75e9) as u64, &[0], 50),
    ]);
    assert!(r.passed, "{:?}", r.details);
}

#[test]
fn stuck_fail() {
    let threshold = gap_threshold_ms();
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
    ]);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("stuck")));
}

#[test]
fn isolation_pass() {
    let expected: BTreeSet<usize> = [0, 1, 2, 3].into_iter().collect();
    let r = assert_isolation(
        &[
            rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
            rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[2, 3], 50),
        ],
        &expected,
    );
    assert!(r.passed);
}

#[test]
fn isolation_fail() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_isolation(
        &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)],
        &expected,
    );
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("unexpected")));
}

#[test]
fn spread_boundary() {
    let threshold = spread_threshold_pct();
    // At threshold exactly - pass
    // Worker 1: 10% off-CPU, Worker 2: 10%+threshold off-CPU
    let at_threshold_ns = ((10.0 + threshold) / 100.0 * 5e9) as u64;
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
        rpt(2, 1000, 5e9 as u64, at_threshold_ns, &[0], 50), // 10% + threshold
    ]);
    assert!(
        r.passed,
        "{threshold}% spread at threshold: {:?}",
        r.details
    );
    // Above threshold - fail
    let above_ns = ((15.0 + threshold) / 100.0 * 5e9) as u64;
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50), // 10%
        rpt(2, 1000, 5e9 as u64, above_ns, &[0], 50),   // 10% + threshold + 5%
    ]);
    assert!(!r.passed, "spread above {threshold}% should fail");
}

#[test]
fn empty_pass() {
    assert!(assert_not_starved(&[]).passed);
}

#[test]
fn zero_wall_time() {
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 0, 0, 0, &[], 0),
    ]);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("starved")));
}

#[test]
fn single_worker_always_pass() {
    let r = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)]);
    assert!(r.passed);
    assert_eq!(r.stats.total_workers, 1);
    assert_eq!(r.stats.cgroups.len(), 1);
}

#[test]
fn stats_accuracy() {
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),  // 20%
        rpt(2, 1000, 5e9 as u64, 15e8 as u64, &[1], 60), // 30%
    ]);
    assert!(r.passed); // spread = 10% < 15%
    let c = &r.stats.cgroups[0];
    assert_eq!(c.num_workers, 2);
    assert_eq!(c.num_cpus, 2);
    assert!((c.min_off_cpu_pct - 20.0).abs() < 0.1);
    assert!((c.max_off_cpu_pct - 30.0).abs() < 0.1);
    assert!((c.spread - 10.0).abs() < 0.1);
    assert!((c.avg_off_cpu_pct - 25.0).abs() < 0.1);
}

#[test]
fn isolation_empty_reports() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    assert!(assert_isolation(&[], &expected).passed);
}

#[test]
fn gap_boundary_at_threshold_pass() {
    let threshold = gap_threshold_ms();
    let r = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold)]);
    assert!(r.passed, "gap at threshold should pass: {:?}", r.details);
}

#[test]
fn gap_boundary_above_threshold_fail() {
    let threshold = gap_threshold_ms();
    let r = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 1)]);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("stuck")));
}

#[test]
fn multiple_stuck_workers() {
    let threshold = gap_threshold_ms();
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], threshold + 500),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], threshold + 1500),
    ]);
    assert!(!r.passed);
    let stuck_count = r.details.iter().filter(|d| d.contains("stuck")).count();
    assert_eq!(stuck_count, 2, "both workers should be flagged stuck");
}

#[test]
fn migration_tracking() {
    let mut report = rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50);
    report.migration_count = 5;
    let r = assert_not_starved(&[report]);
    assert_eq!(r.stats.total_migrations, 5);
}

#[test]
fn single_worker_spread_zero() {
    let r = assert_not_starved(&[rpt(1, 500, 5e9 as u64, 25e8 as u64, &[0, 1], 50)]);
    assert!(r.passed);
    let c = &r.stats.cgroups[0];
    assert!((c.spread - 0.0).abs() < f64::EPSILON);
}

#[test]
fn zero_wall_time_nonzero_work() {
    // wall_time=0 but work_units>0: the worker did work but the timer
    // didn't advance. Should not produce a starved failure since work was done.
    // The off_cpu_pct computation skips this worker (no pcts entry).
    let r = assert_not_starved(&[rpt(1, 100, 0, 0, &[0], 0)]);
    assert!(
        r.passed,
        "nonzero work with zero wall_time: {:?}",
        r.details
    );
}

#[test]
fn isolation_empty_expected_set() {
    // Empty expected set means no CPUs are "expected", so any CPU
    // used by the worker is unexpected. difference(empty) == worker's set.
    let expected: BTreeSet<usize> = BTreeSet::new();
    let r = assert_isolation(
        &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50)],
        &expected,
    );
    // Worker used CPUs {0,1}, expected is empty, so all are unexpected.
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("unexpected")));
}

#[test]
fn isolation_worker_used_no_cpus() {
    // Worker used no CPUs -- difference with expected is empty, so passes.
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_isolation(&[rpt(1, 0, 0, 0, &[], 0)], &expected);
    assert!(r.passed);
}

#[test]
fn isolation_all_unexpected_cpus() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let r = assert_isolation(
        &[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[4, 5, 6], 50)],
        &expected,
    );
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("unexpected")));
}

// ---------------------------------------------------------------
// Negative tests: check that diagnostics catch controlled failures
// ---------------------------------------------------------------

#[test]
fn neg_starvation_zero_work_detected() {
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 0, 5e9 as u64, 0, &[0], 0), // starved
        rpt(3, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
    ]);
    assert!(!r.passed, "starvation must be caught");
    let starved = r.details.iter().filter(|d| d.contains("starved")).count();
    assert_eq!(starved, 1, "exactly one starved worker expected");
    // Format: "tid 2 starved (0 work units)"
    let detail = r.details.iter().find(|d| d.contains("starved")).unwrap();
    assert!(
        detail.contains("tid 2"),
        "must name the starved tid: {detail}"
    );
    assert!(
        detail.contains("0 work units"),
        "must state zero work: {detail}"
    );
}

#[test]
fn neg_isolation_violation_outside_cpuset() {
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2, 3], 50),
    ];
    let r = assert_isolation(&reports, &expected);
    assert!(!r.passed, "isolation violation must be caught");
    // Format: "tid 2 ran on unexpected CPUs {2, 3}"
    let detail = r
        .details
        .iter()
        .find(|d| d.contains("unexpected CPUs"))
        .unwrap();
    assert!(
        detail.contains("tid 2"),
        "must name violating tid: {detail}"
    );
    assert!(detail.contains("2"), "must list out-of-set CPU 2: {detail}");
    assert!(detail.contains("3"), "must list out-of-set CPU 3: {detail}");
    // Worker 1 ran only on {0,1} which is within expected — no violation.
    assert_eq!(r.details.len(), 1, "only tid 2 should violate");
}

#[test]
fn neg_unfairness_extreme_spread_detected() {
    let r = assert_not_starved(&[
        rpt(1, 100, 5e9 as u64, 25e7 as u64, &[0, 1], 50), // 5%
        rpt(2, 5000, 5e9 as u64, 475e7 as u64, &[0, 1], 50), // 95%
    ]);
    assert!(!r.passed, "extreme unfairness must be caught");
    // Format: "unfair cgroup: spread=90% (5-95%) 2 workers on 2 cpus (threshold 15%)"
    let detail = r.details.iter().find(|d| d.contains("unfair")).unwrap();
    assert!(
        detail.contains("spread="),
        "must include spread value: {detail}"
    );
    assert!(
        detail.contains("workers"),
        "must include worker count: {detail}"
    );
    assert!(detail.contains("cpus"), "must include cpu count: {detail}");
    // Threshold must appear so a regression dropping the bound surfaces here.
    // The literal value comes from `spread_threshold_pct()` which differs
    // between debug and release builds; pin only the textual prefix.
    assert!(
        detail.contains("threshold "),
        "must include threshold bound: {detail}"
    );
    let c = &r.stats.cgroups[0];
    assert!(
        c.spread > 80.0,
        "spread should be >80%, got {:.1}",
        c.spread
    );
    assert_eq!(c.num_workers, 2);
    assert_eq!(c.num_cpus, 2);
    assert!(
        c.min_off_cpu_pct < 10.0,
        "min pct should be ~5%: {:.1}",
        c.min_off_cpu_pct
    );
    assert!(
        c.max_off_cpu_pct > 90.0,
        "max pct should be ~95%: {:.1}",
        c.max_off_cpu_pct
    );
}

#[test]
fn neg_scheduling_gap_exceeds_threshold() {
    let threshold = gap_threshold_ms();
    let gap = threshold + 2000;
    let r = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], gap),
    ]);
    assert!(!r.passed, "scheduling gap must be caught");
    // Format: "tid 2 stuck {gap}ms on cpu1 at +1000ms (threshold 2000ms)"
    let detail = r.details.iter().find(|d| d.contains("stuck")).unwrap();
    assert!(
        detail.contains(&format!("{}ms", gap)),
        "must include gap duration: {detail}"
    );
    assert!(
        detail.contains("on cpu"),
        "must include CPU number: {detail}"
    );
    assert!(
        detail.contains("at +"),
        "must include timing offset: {detail}"
    );
    assert!(detail.contains("cpu1"), "gap is on cpu1: {detail}");
    // tid must be named so an operator triaging a multi-worker cgroup can
    // identify the offender without reverse-mapping CPU placement.
    assert!(
        detail.contains("tid 2"),
        "must name violating tid (2): {detail}"
    );
    // Threshold appears for parity with the AssertPlan custom-threshold path
    // and so a regression dropping the bound from the default-path message
    // surfaces here.
    assert!(
        detail.contains(&format!("threshold {}ms", threshold)),
        "must include default-path threshold: {detail}"
    );
    // Stats must reflect the gap.
    assert_eq!(r.stats.worst_gap_ms, gap);
    assert_eq!(r.stats.worst_gap_cpu, 1);
}

#[test]
fn neg_plan_custom_gap_catches_lower_threshold() {
    let plan = AssertPlan::new().check_not_starved().max_gap_ms(500);
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
    ];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.passed, "custom 500ms threshold must catch 1000ms gap");
    // Format: "tid 2 stuck 1000ms on cpu1 at +1000ms (threshold 500ms)"
    let detail = r.details.iter().find(|d| d.contains("stuck")).unwrap();
    assert!(
        detail.contains("1000ms"),
        "must include gap duration: {detail}"
    );
    assert!(detail.contains("cpu1"), "must include CPU: {detail}");
    assert!(
        detail.contains("threshold 500ms"),
        "must include custom threshold: {detail}"
    );
    // tid must be named; pins parity with the bare-path message.
    assert!(
        detail.contains("tid 2"),
        "must name violating tid (2): {detail}"
    );
}

#[test]
fn neg_isolation_plus_starvation_both_reported() {
    let plan = AssertPlan::new().check_not_starved().check_isolation();
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let reports = [
        rpt(1, 0, 5e9 as u64, 0, &[0], 0),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[4, 5], 50),
    ];
    let r = plan.assert_cgroup(&reports, Some(&expected), None);
    assert!(!r.passed);
    // Starvation detail must name tid 1 with "0 work units".
    let starved_detail = r.details.iter().find(|d| d.contains("starved")).unwrap();
    assert!(
        starved_detail.contains("tid 1"),
        "starved tid: {starved_detail}"
    );
    assert!(
        starved_detail.contains("0 work units"),
        "format: {starved_detail}"
    );
    // Isolation detail must name tid 2 with CPUs {4, 5}.
    let iso_detail = r.details.iter().find(|d| d.contains("unexpected")).unwrap();
    assert!(iso_detail.contains("tid 2"), "isolation tid: {iso_detail}");
    assert!(iso_detail.contains("4"), "must list CPU 4: {iso_detail}");
    assert!(iso_detail.contains("5"), "must list CPU 5: {iso_detail}");
}

#[test]
fn neg_assert_cgroup_via_assert_struct() {
    let v = Assert::NO_OVERRIDES.check_not_starved().check_isolation();
    let expected: BTreeSet<usize> = [0].into_iter().collect();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 2], 50)];
    let r = v.assert_cgroup(&reports, Some(&expected));
    assert!(
        !r.passed,
        "Assert.assert_cgroup must catch isolation failure"
    );
    let detail = r.details.iter().find(|d| d.contains("unexpected")).unwrap();
    assert!(detail.contains("tid 1"), "must name tid: {detail}");
    assert!(detail.contains("1"), "must list CPU 1: {detail}");
    assert!(detail.contains("2"), "must list CPU 2: {detail}");
}

#[test]
fn neg_plan_custom_gap_passes_below_threshold() {
    let plan = AssertPlan::new().check_not_starved().max_gap_ms(5000);
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 1000),
    ];
    let r = plan.assert_cgroup(&reports, None, None);
    // 1000ms gap < 5000ms threshold, so it passes.
    let has_stuck = r.details.iter().any(|d| d.contains("stuck"));
    assert!(!has_stuck, "1000ms gap should pass 5000ms threshold");
}
