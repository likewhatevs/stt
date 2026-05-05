//! `Assert` struct tests: `NO_OVERRIDES` / `default_checks`,
//! per-field merge precedence, builder setters, the
//! `worker_plan` / `monitor_thresholds` extractors,
//! `has_worker_checks` discriminator, and the
//! `gap_threshold_ms` debug-vs-release helper.

use super::tests_common::rpt;
use super::*;

#[test]
fn assert_no_overrides_has_no_checks() {
    let v = Assert::NO_OVERRIDES;
    assert!(v.not_starved.is_none());
    assert!(v.isolation.is_none());
    assert!(v.max_gap_ms.is_none());
    assert!(v.max_spread_pct.is_none());
    assert!(v.max_imbalance_ratio.is_none());
}

#[test]
fn assert_default_checks_enables_not_starved() {
    let v = Assert::default_checks();
    assert_eq!(v.not_starved, Some(true));
    assert!(v.isolation.is_none());
    assert!(v.max_imbalance_ratio.is_some());
    assert!(v.max_local_dsq_depth.is_some());
    assert!(v.fail_on_stall.is_some());
    assert!(v.sustained_samples.is_some());
    assert!(v.max_fallback_rate.is_some());
    assert!(v.max_keep_last_rate.is_some());
}

#[test]
fn assert_merge_other_overrides_self() {
    let base = Assert::NO_OVERRIDES;
    let other = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_gap_ms(5000)
        .max_imbalance_ratio(2.0);
    let merged = base.merge(&other);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_gap_ms, Some(5000));
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
}

#[test]
fn assert_merge_preserves_self_when_other_is_none() {
    let base = Assert::default_checks();
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.not_starved, Some(true));
    assert!(merged.max_imbalance_ratio.is_some());
    assert!(merged.max_local_dsq_depth.is_some());
}

#[test]
fn assert_merge_other_takes_precedence() {
    let base = Assert::NO_OVERRIDES.max_imbalance_ratio(4.0);
    let other = Assert::NO_OVERRIDES.max_imbalance_ratio(2.0);
    let merged = base.merge(&other);
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
}

#[test]
fn assert_merge_last_some_wins() {
    let base = Assert::NO_OVERRIDES.check_not_starved();
    let other = Assert::NO_OVERRIDES.check_isolation();
    let merged = base.merge(&other);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.isolation, Some(true));
}

#[test]
fn assert_merge_child_disables_not_starved() {
    let base = Assert::default_checks(); // not_starved = Some(true)
    let other = Assert {
        not_starved: Some(false),
        ..Assert::NO_OVERRIDES
    };
    let merged = base.merge(&other);
    assert_eq!(merged.not_starved, Some(false));
    assert!(!merged.worker_plan().not_starved);
}

#[test]
fn assert_merge_child_disables_isolation() {
    let base = Assert::NO_OVERRIDES.check_isolation(); // isolation = Some(true)
    let other = Assert {
        isolation: Some(false),
        ..Assert::NO_OVERRIDES
    };
    let merged = base.merge(&other);
    assert_eq!(merged.isolation, Some(false));
    assert!(!merged.worker_plan().isolation);
}

#[test]
fn assert_worker_plan_extraction() {
    let v = Assert::NO_OVERRIDES
        .check_not_starved()
        .check_isolation()
        .max_gap_ms(3000)
        .max_spread_pct(25.0);
    assert_eq!(v.not_starved, Some(true));
    assert_eq!(v.isolation, Some(true));
    let plan = v.worker_plan();
    assert!(plan.not_starved);
    assert!(plan.isolation);
    assert_eq!(plan.max_gap_ms, Some(3000));
    assert_eq!(plan.max_spread_pct, Some(25.0));
}

#[test]
fn assert_cgroup_delegates_to_plan() {
    let v = Assert::NO_OVERRIDES.check_not_starved();
    let reports = [rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)];
    let r = v.assert_cgroup(&reports, None);
    assert!(r.passed);
    assert_eq!(r.stats.total_workers, 1);
}

#[test]
fn assert_monitor_thresholds_extraction() {
    let v = Assert::NO_OVERRIDES
        .max_imbalance_ratio(2.5)
        .max_local_dsq_depth(100)
        .fail_on_stall(false)
        .sustained_samples(10)
        .max_fallback_rate(50.0)
        .max_keep_last_rate(25.0);
    let t = v.monitor_thresholds();
    assert!((t.max_imbalance_ratio - 2.5).abs() < f64::EPSILON);
    assert_eq!(t.max_local_dsq_depth, 100);
    assert!(!t.fail_on_stall);
    assert_eq!(t.sustained_samples, 10);
    assert!((t.max_fallback_rate - 50.0).abs() < f64::EPSILON);
    assert!((t.max_keep_last_rate - 25.0).abs() < f64::EPSILON);
}

#[test]
fn assert_monitor_thresholds_defaults_when_none() {
    let v = Assert::NO_OVERRIDES;
    let t = v.monitor_thresholds();
    let d = crate::monitor::MonitorThresholds::DEFAULT;
    assert!((t.max_imbalance_ratio - d.max_imbalance_ratio).abs() < f64::EPSILON);
    assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
}

#[test]
fn assert_chain_all_setters() {
    let v = Assert::NO_OVERRIDES
        .check_not_starved()
        .check_isolation()
        .max_gap_ms(1000)
        .max_spread_pct(5.0)
        .max_imbalance_ratio(3.0)
        .max_local_dsq_depth(20)
        .fail_on_stall(true)
        .sustained_samples(3)
        .max_fallback_rate(100.0)
        .max_keep_last_rate(50.0);
    assert_eq!(v.not_starved, Some(true));
    assert_eq!(v.isolation, Some(true));
    assert_eq!(v.max_gap_ms, Some(1000));
    assert_eq!(v.max_spread_pct, Some(5.0));
    assert_eq!(v.max_imbalance_ratio, Some(3.0));
    assert_eq!(v.max_local_dsq_depth, Some(20));
    assert_eq!(v.fail_on_stall, Some(true));
    assert_eq!(v.sustained_samples, Some(3));
    assert_eq!(v.max_fallback_rate, Some(100.0));
    assert_eq!(v.max_keep_last_rate, Some(50.0));
}

// -- gap_threshold_ms tests --

#[test]
fn gap_threshold_default() {
    let t = gap_threshold_ms();
    if cfg!(debug_assertions) {
        assert_eq!(t, 3000);
    } else {
        assert_eq!(t, 2000);
    }
}

// -- Assert::merge per-field tests --

#[test]
fn assert_merge_max_spread_pct() {
    let base = Assert::NO_OVERRIDES.max_spread_pct(10.0);
    let other = Assert::NO_OVERRIDES.max_spread_pct(5.0);
    assert_eq!(base.merge(&other).max_spread_pct, Some(5.0));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).max_spread_pct, Some(10.0));
}

#[test]
fn assert_merge_fail_on_stall() {
    let base = Assert::NO_OVERRIDES.fail_on_stall(true);
    let other = Assert::NO_OVERRIDES.fail_on_stall(false);
    assert_eq!(base.merge(&other).fail_on_stall, Some(false));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).fail_on_stall, Some(true));
}

#[test]
fn assert_merge_sustained_samples() {
    let base = Assert::NO_OVERRIDES.sustained_samples(5);
    let other = Assert::NO_OVERRIDES.sustained_samples(10);
    assert_eq!(base.merge(&other).sustained_samples, Some(10));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).sustained_samples, Some(5));
}

#[test]
fn assert_merge_max_fallback_rate() {
    let base = Assert::NO_OVERRIDES.max_fallback_rate(200.0);
    let other = Assert::NO_OVERRIDES.max_fallback_rate(50.0);
    assert_eq!(base.merge(&other).max_fallback_rate, Some(50.0));
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES).max_fallback_rate,
        Some(200.0)
    );
}

#[test]
fn assert_merge_max_keep_last_rate() {
    let base = Assert::NO_OVERRIDES.max_keep_last_rate(100.0);
    let other = Assert::NO_OVERRIDES.max_keep_last_rate(25.0);
    assert_eq!(base.merge(&other).max_keep_last_rate, Some(25.0));
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES).max_keep_last_rate,
        Some(100.0)
    );
}

#[test]
fn assert_merge_max_local_dsq_depth() {
    let base = Assert::NO_OVERRIDES.max_local_dsq_depth(50);
    let other = Assert::NO_OVERRIDES.max_local_dsq_depth(100);
    assert_eq!(base.merge(&other).max_local_dsq_depth, Some(100));
    assert_eq!(
        base.merge(&Assert::NO_OVERRIDES).max_local_dsq_depth,
        Some(50)
    );
}

#[test]
fn assert_merge_max_gap_ms() {
    let base = Assert::NO_OVERRIDES.max_gap_ms(2000);
    let other = Assert::NO_OVERRIDES.max_gap_ms(5000);
    assert_eq!(base.merge(&other).max_gap_ms, Some(5000));
    assert_eq!(base.merge(&Assert::NO_OVERRIDES).max_gap_ms, Some(2000));
}

#[test]
fn assert_merge_three_layers() {
    let defaults = Assert::default_checks();
    let sched = Assert::NO_OVERRIDES
        .max_imbalance_ratio(2.0)
        .max_fallback_rate(50.0);
    let test = Assert::NO_OVERRIDES.max_gap_ms(5000);
    let merged = defaults.merge(&sched).merge(&test);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    assert_eq!(merged.max_fallback_rate, Some(50.0));
    assert_eq!(merged.max_gap_ms, Some(5000));
    assert_eq!(merged.sustained_samples, Some(5));
}

#[test]
fn assert_merge_no_overrides_preserves_base() {
    let base = Assert::default_checks();
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.not_starved, Some(true));
    assert!(merged.max_imbalance_ratio.is_some());
    assert!(merged.fail_on_stall.is_some());
}

/// `Assert::NO_OVERRIDES` is the two-sided identity for `merge`. The
/// right-identity case is covered above; this locks the
/// left-identity case so a `NO_OVERRIDES.merge(&default_checks())`
/// at either order in the runtime chain produces the same defaults.
#[test]
fn assert_merge_no_overrides_is_left_identity() {
    let merged = Assert::NO_OVERRIDES.merge(&Assert::default_checks());
    let baseline = Assert::default_checks();
    assert_eq!(merged.not_starved, baseline.not_starved);
    assert_eq!(merged.max_imbalance_ratio, baseline.max_imbalance_ratio);
    assert_eq!(merged.max_local_dsq_depth, baseline.max_local_dsq_depth);
    assert_eq!(merged.fail_on_stall, baseline.fail_on_stall);
    assert_eq!(merged.sustained_samples, baseline.sustained_samples);
    assert_eq!(merged.max_fallback_rate, baseline.max_fallback_rate);
    assert_eq!(merged.max_keep_last_rate, baseline.max_keep_last_rate);
    // Fields that default_checks leaves None remain None.
    assert!(merged.max_gap_ms.is_none());
    assert!(merged.isolation.is_none());
}

/// The runtime three-layer chain
/// `default_checks -> scheduler -> test` collapses to
/// `default_checks` when both override layers are `NO_OVERRIDES`.
/// This proves the documented "no override, not no checks"
/// invariant end-to-end.
#[test]
fn assert_merge_runtime_chain_with_no_overrides_yields_defaults() {
    let scheduler_assert = Assert::NO_OVERRIDES;
    let test_assert = Assert::NO_OVERRIDES;
    let merged = Assert::default_checks()
        .merge(&scheduler_assert)
        .merge(&test_assert);
    let baseline = Assert::default_checks();
    assert_eq!(merged.not_starved, baseline.not_starved);
    assert_eq!(merged.max_imbalance_ratio, baseline.max_imbalance_ratio);
    assert_eq!(merged.max_local_dsq_depth, baseline.max_local_dsq_depth);
    assert_eq!(merged.fail_on_stall, baseline.fail_on_stall);
    assert_eq!(merged.sustained_samples, baseline.sustained_samples);
    assert_eq!(merged.max_fallback_rate, baseline.max_fallback_rate);
    assert_eq!(merged.max_keep_last_rate, baseline.max_keep_last_rate);
}

#[test]
fn assert_merge_overrides_fields() {
    let base = Assert::NO_OVERRIDES;
    let overrides = Assert::NO_OVERRIDES
        .max_imbalance_ratio(5.0)
        .max_gap_ms(1000)
        .check_not_starved();
    let merged = base.merge(&overrides);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(5.0));
    assert_eq!(merged.max_gap_ms, Some(1000));
}

#[test]
fn assert_merge_later_overrides_earlier() {
    let a = Assert::NO_OVERRIDES.max_imbalance_ratio(2.0);
    let b = Assert::NO_OVERRIDES.max_imbalance_ratio(10.0);
    let merged = a.merge(&b);
    assert_eq!(merged.max_imbalance_ratio, Some(10.0));
}

#[test]
fn assert_worker_plan_extracts_fields() {
    let v = Assert::NO_OVERRIDES
        .check_not_starved()
        .check_isolation()
        .max_gap_ms(500)
        .max_spread_pct(10.0);
    assert_eq!(v.not_starved, Some(true));
    assert_eq!(v.isolation, Some(true));
    let plan = v.worker_plan();
    assert!(plan.not_starved);
    assert!(plan.isolation);
    assert_eq!(plan.max_gap_ms, Some(500));
    assert_eq!(plan.max_spread_pct, Some(10.0));
}

#[test]
fn assert_monitor_thresholds_defaults() {
    let v = Assert::NO_OVERRIDES;
    let t = v.monitor_thresholds();
    // Should use MonitorThresholds::DEFAULT values.
    let d = crate::monitor::MonitorThresholds::DEFAULT;
    assert_eq!(t.max_imbalance_ratio, d.max_imbalance_ratio);
    assert_eq!(t.max_local_dsq_depth, d.max_local_dsq_depth);
}

#[test]
fn assert_monitor_thresholds_overridden() {
    let v = Assert::NO_OVERRIDES
        .max_imbalance_ratio(99.0)
        .max_local_dsq_depth(42)
        .fail_on_stall(false)
        .sustained_samples(10)
        .max_fallback_rate(0.5)
        .max_keep_last_rate(0.3);
    let t = v.monitor_thresholds();
    assert_eq!(t.max_imbalance_ratio, 99.0);
    assert_eq!(t.max_local_dsq_depth, 42);
    assert!(!t.fail_on_stall);
    assert_eq!(t.sustained_samples, 10);
    assert_eq!(t.max_fallback_rate, 0.5);
    assert_eq!(t.max_keep_last_rate, 0.3);
}

#[test]
fn assert_max_spread_pct() {
    let v = Assert::NO_OVERRIDES.max_spread_pct(25.0);
    assert_eq!(v.max_spread_pct, Some(25.0));
}

#[test]
fn gap_threshold_debug_vs_release() {
    let t = gap_threshold_ms();
    // In test builds (debug_assertions=true), threshold is 3000.
    assert!(t >= 2000, "threshold should be at least 2000ms: {t}");
}

// -- Assert::has_worker_checks --

#[test]
fn assert_no_overrides_has_no_worker_checks() {
    assert!(!Assert::NO_OVERRIDES.has_worker_checks());
}

#[test]
fn assert_default_checks_has_worker_checks() {
    assert!(Assert::default_checks().has_worker_checks());
}

#[test]
fn assert_single_field_has_worker_checks() {
    assert!(Assert::NO_OVERRIDES.max_gap_ms(5000).has_worker_checks());
    assert!(Assert::NO_OVERRIDES.check_isolation().has_worker_checks());
    assert!(
        Assert::NO_OVERRIDES
            .max_spread_pct(10.0)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_throughput_cv(0.5)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .min_work_rate(100.0)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_p99_wake_latency_ns(1000)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_wake_latency_cv(0.5)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .min_iteration_rate(10.0)
            .has_worker_checks()
    );
    assert!(
        Assert::NO_OVERRIDES
            .max_migration_ratio(0.5)
            .has_worker_checks()
    );
}

#[test]
fn assert_monitor_only_no_worker_checks() {
    let a = Assert::NO_OVERRIDES
        .max_imbalance_ratio(5.0)
        .fail_on_stall(true);
    assert!(!a.has_worker_checks());
}

// -- Assert::merge worker + benchmark + monitor fields --

#[test]
fn assert_merge_all_field_categories() {
    // Layer 1: defaults (worker + monitor fields).
    let defaults = Assert::default_checks();

    // Layer 2: scheduler sets worker and benchmark fields.
    let sched = Assert::NO_OVERRIDES
        .max_spread_pct(50.0)
        .max_p99_wake_latency_ns(100_000)
        .max_migration_ratio(0.5);

    // Layer 3: test overrides a worker field and sets isolation.
    let test = Assert::NO_OVERRIDES.check_isolation().max_spread_pct(80.0);

    let merged = defaults.merge(&sched).merge(&test);

    // test overrides sched's spread.
    assert_eq!(merged.max_spread_pct, Some(80.0));
    // sched's benchmark fields survive (test didn't set them).
    assert_eq!(merged.max_p99_wake_latency_ns, Some(100_000));
    assert_eq!(merged.max_migration_ratio, Some(0.5));
    // test sets isolation.
    assert_eq!(merged.isolation, Some(true));
    // defaults: monitor fields survive all layers.
    assert_eq!(merged.fail_on_stall, Some(true));
}
