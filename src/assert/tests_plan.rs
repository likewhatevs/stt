//! `AssertPlan` builder + dispatch tests. Pins per-check toggles,
//! custom-threshold paths (gap, spread), and the
//! permissive-overrides-strip-but-keep-starved invariant that
//! lets a per-test plan loosen specific bounds without
//! suppressing genuine starvation findings.

use super::tests_common::rpt;
use super::*;

#[test]
fn plan_default_empty() {
    let plan = AssertPlan::new();
    assert!(!plan.not_starved);
    assert!(!plan.isolation);
    assert!(plan.max_gap_ms.is_none());
    assert!(plan.max_spread_pct.is_none());
}

#[test]
fn plan_check_not_starved() {
    let plan = AssertPlan::new().check_not_starved();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(r.passed);
    assert_eq!(r.stats.total_workers, 1);
}

#[test]
fn plan_check_isolation_with_cpuset() {
    let plan = AssertPlan::new().check_not_starved().check_isolation();
    let expected: BTreeSet<usize> = [0, 1].into_iter().collect();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
    let r = plan.assert_cgroup(&reports, Some(&expected), None);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("unexpected")));
}

#[test]
fn plan_isolation_skipped_without_cpuset() {
    let plan = AssertPlan::new().check_isolation();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1, 4], 50)];
    // No cpuset provided -- isolation check is skipped.
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(r.passed);
}

#[test]
fn plan_custom_gap_threshold_pass() {
    let plan = AssertPlan::new().check_not_starved().max_gap_ms(3000);
    // 2500ms gap: passes with 3000ms threshold.
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2500)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(r.passed, "2500ms < 3000ms threshold: {:?}", r.details);
}

#[test]
fn plan_custom_gap_threshold_fail() {
    let plan = AssertPlan::new().check_not_starved().max_gap_ms(1500);
    // 2000ms gap: fails with 1500ms threshold.
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2000)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.passed);
    assert!(r.details.iter().any(|d| d.contains("stuck")));
    assert!(r.details.iter().any(|d| d.contains("threshold 1500ms")));
}

#[test]
fn plan_custom_gap_threshold_produces_stuck_kind() {
    // AssertPlan's custom-threshold stuck re-emission must tag
    // DetailKind::Stuck so downstream kind filters (and any test
    // expecting structural categorization) see it.
    let plan = AssertPlan::new().check_not_starved().max_gap_ms(1500);
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 2000)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(!r.passed);
    assert!(
        r.details.iter().any(|d| d.kind == DetailKind::Stuck),
        "custom gap override must produce a Stuck-kind detail: {:?}",
        r.details
    );
}

#[test]
fn plan_permissive_overrides_clear_unfair_and_stuck_preserve_starved() {
    // When custom spread + gap thresholds are permissive enough
    // to absorb the default-threshold failures, AssertPlan must
    // strip the Unfair/Stuck details it generated but keep the
    // Starved detail (kind-based filtering, not substring match).
    //
    // Worker 1: 10% off-CPU, 500ms gap — fair, not stuck.
    // Worker 2: work=0 — starved (kind=Starved).
    // Worker 3: 80% off-CPU — would trigger default Unfair; absorbed
    //                         by permissive max_spread_pct.
    // Worker 4: 4000ms gap — would trigger default Stuck; absorbed
    //                        by permissive max_gap_ms.
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 500),
        rpt(2, 0, 5e9 as u64, 0, &[0], 500),
        rpt(3, 500, 5e9 as u64, 4e9 as u64, &[0], 500),
        rpt(4, 1000, 5e9 as u64, 5e8 as u64, &[0], 4000),
    ];
    let mut plan = AssertPlan::new();
    plan.not_starved = true;
    plan.max_spread_pct = Some(100.0);
    plan.max_gap_ms = Some(5000);
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(
        r.details.iter().any(|d| d.kind == DetailKind::Starved),
        "starved detail must survive permissive overrides: {:?}",
        r.details
    );
    assert!(
        !r.details.iter().any(|d| d.kind == DetailKind::Unfair),
        "unfair detail must be cleared by permissive spread: {:?}",
        r.details
    );
    assert!(
        !r.details.iter().any(|d| d.kind == DetailKind::Stuck),
        "stuck detail must be cleared by permissive gap: {:?}",
        r.details
    );
    assert!(!r.passed, "starved alone is still a failure");
}

#[test]
fn plan_no_checks_always_passes() {
    let plan = AssertPlan::new();
    let reports = [rpt(1, 0, 0, 0, &[], 5000)]; // starved + stuck
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(r.passed, "no checks enabled should pass");
}

#[test]
fn plan_default_all_checks_disabled() {
    // Default::default() must produce the same state as new() —
    // all checks disabled, no gap override.
    let plan = AssertPlan::default();
    assert!(!plan.not_starved, "default must not enable not_starved");
    assert!(!plan.isolation, "default must not enable isolation");
    assert!(
        plan.max_gap_ms.is_none(),
        "default must not set gap override"
    );
    assert!(
        plan.max_spread_pct.is_none(),
        "default must not set spread override"
    );
    // A plan with all checks disabled must pass even pathological input.
    let reports = [rpt(1, 0, 0, 0, &[], 99999)];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(r.passed, "all-disabled plan must pass any input");
}

#[test]
fn assert_plan_default_equals_new() {
    // Default impl calls new(). Check field-by-field equivalence
    // and that both produce identical assert_cgroup results.
    let d = AssertPlan::default();
    let n = AssertPlan::new();
    assert_eq!(d.not_starved, n.not_starved);
    assert_eq!(d.isolation, n.isolation);
    assert_eq!(d.max_gap_ms, n.max_gap_ms);
    assert_eq!(d.max_spread_pct, n.max_spread_pct);
    // Both should produce identical pass/fail on the same input.
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
    let rd = d.assert_cgroup(&reports, None, None);
    let rn = n.assert_cgroup(&reports, None, None);
    assert_eq!(rd.passed, rn.passed);
}

#[test]
fn plan_starved_still_fails_with_custom_gap() {
    // A starved worker (work_units=0) must still cause failure even
    // when the custom max_gap_ms threshold is high enough that the
    // gap check passes.
    let plan = AssertPlan::new().check_not_starved().max_gap_ms(5000);
    let reports = [
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100), // healthy
        rpt(2, 0, 5e9 as u64, 0, &[1], 1500),            // starved, gap < threshold
    ];
    let r = plan.assert_cgroup(&reports, None, None);
    assert!(
        !r.passed,
        "starved worker must fail even with relaxed gap threshold"
    );
    assert!(r.details.iter().any(|d| d.contains("starved")));
    // The gap (1500ms) is below the 5000ms threshold, so no "stuck" detail.
    assert!(!r.details.iter().any(|d| d.contains("stuck")));
}
