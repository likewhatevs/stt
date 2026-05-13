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
fn assert_default_checks_is_no_overrides() {
    let v = Assert::default_checks();
    assert!(v.not_starved.is_none());
    assert!(v.isolation.is_none());
    assert!(v.max_imbalance_ratio.is_none());
    assert!(v.max_local_dsq_depth.is_none());
    assert!(v.fail_on_stall.is_none());
    assert!(v.sustained_samples.is_none());
    assert!(v.max_fallback_rate.is_none());
    assert!(v.max_keep_last_rate.is_none());
}

/// Wire-format pin: `Assert::with_monitor_defaults()` MUST set
/// `enforce_monitor_thresholds = true`. The propagation chain is:
/// builder.with_monitor_defaults() → enforce_monitor_thresholds=true
/// → monitor_thresholds().enforce=true → MonitorThresholds.enforce=true
/// → evaluate() respects enforce.
///
/// Per `Assert::with_monitor_defaults` doc: opt into pass/fail
/// enforcement for monitor thresholds. Without this call, monitor
/// violations are reported in the verdict's details but do not
/// fail the test. With it, any monitor threshold violation fails
/// the test.
///
/// A regression that breaks the propagation (e.g. forgets to set
/// enforce_monitor_thresholds in the auto-fill helper, OR changes
/// the field name without updating the threshold extractor) would
/// silently disable enforcement for every test using the builder
/// pattern.
#[test]
fn with_monitor_defaults_propagates_enforce_true_to_thresholds() {
    let assert = Assert::NO_OVERRIDES.with_monitor_defaults();
    let t = assert.monitor_thresholds();
    assert!(
        t.enforce,
        "with_monitor_defaults() must propagate enforce=true to MonitorThresholds"
    );
}

/// Wire-format pin: without `.with_monitor_defaults()` the builder
/// produces enforce=false. This is the post-enforce-default-flip
/// report-only default that broke 19 tests when the default flipped
/// — pin both polarities so a flip in either direction is caught.
#[test]
fn assert_default_without_monitor_defaults_yields_enforce_false() {
    let assert = Assert::NO_OVERRIDES;
    let t = assert.monitor_thresholds();
    assert!(
        !t.enforce,
        "default Assert (no .with_monitor_defaults()) must yield enforce=false"
    );
}

/// Direct-field pin: `Assert::NO_OVERRIDES.enforce_monitor_thresholds`
/// MUST be `false`. Parallel to the `monitor_thresholds()` canary above —
/// catches a regression where the helper is preserved but the underlying
/// field default flips. NO_OVERRIDES is the runtime composition base
/// (see the runtime composition documented on `Assert::merge`), so a
/// flipped default here would silently enable enforcement for every
/// scheduler that doesn't explicitly opt out.
#[test]
fn no_overrides_enforce_monitor_thresholds_field_is_false() {
    let v = Assert::NO_OVERRIDES;
    assert!(
        !v.enforce_monitor_thresholds,
        "NO_OVERRIDES.enforce_monitor_thresholds must be false (post-enforce-default-flip default)"
    );
}

/// Merge semantics pin: `enforce_monitor_thresholds` uses sticky-OR
/// (see the OR expression in `Assert::merge`:
/// `self.enforce_monitor_thresholds || other.enforce_monitor_thresholds`).
/// Distinct from every other field which uses last-Some-wins. The
/// OR rule means a child Assert that opted into enforcement cannot
/// be silently downgraded by merging over (or under) a non-enforcing
/// peer.
///
/// This canary covers the false||false=false case — the baseline that
/// guarantees a merge of two NO_OVERRIDES (the runtime no-op composition)
/// stays in report-only mode.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_false_false() {
    let merged = Assert::NO_OVERRIDES.merge(&Assert::NO_OVERRIDES);
    assert!(
        !merged.enforce_monitor_thresholds,
        "false || false must yield false"
    );
}

/// Merge semantics pin: true (self) || false (other) → true.
/// A test-level `Assert::NO_OVERRIDES.with_monitor_defaults()` on the
/// left merged with a NO_OVERRIDES override on the right keeps
/// enforce=true. Without OR-semantics, a regression could silently
/// flip back to last-Some-wins and drop enforcement on every test
/// that uses a non-enforcing override layer.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_true_left() {
    let left = Assert::NO_OVERRIDES.with_monitor_defaults();
    let right = Assert::NO_OVERRIDES;
    let merged = left.merge(&right);
    assert!(
        merged.enforce_monitor_thresholds,
        "true (self) || false (other) must yield true (sticky OR)"
    );
}

/// Merge semantics pin: false (self) || true (other) → true.
/// Symmetric to the previous canary — a non-enforcing base layer
/// merged with an enforcing override layer must yield enforce=true.
/// Both directions matter because the runtime composes via
/// `default_checks().merge(&scheduler.assert).merge(&test.assert)`
/// (see `Assert::merge`), so the enforcing assert can appear on either
/// side depending on which layer opted in.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_false_true() {
    let left = Assert::NO_OVERRIDES;
    let right = Assert::NO_OVERRIDES.with_monitor_defaults();
    let merged = left.merge(&right);
    assert!(
        merged.enforce_monitor_thresholds,
        "false (self) || true (other) must yield true (sticky OR)"
    );
}

/// Behavior pin: `with_monitor_defaults()` fills ALL six unset
/// monitor-threshold Option fields with `MonitorThresholds::DEFAULT`
/// values, not just `enforce_monitor_thresholds`. Documented on
/// `Assert::with_monitor_defaults` ("Also populates any unset
/// monitor-threshold field with the canonical default"). Without
/// this canary, a regression that drops the per-field auto-fill
/// loop in `with_monitor_defaults` would silently let enforced
/// tests run against `None` fields that
/// downstream `monitor_thresholds()` would still default — same end
/// behavior today, but a future API consumer reading the raw
/// `Assert` fields would see misleading `None`s.
#[test]
fn with_monitor_defaults_fills_all_unset_threshold_fields() {
    use crate::monitor::MonitorThresholds;
    let assert = Assert::NO_OVERRIDES.with_monitor_defaults();
    let d = MonitorThresholds::DEFAULT;
    assert!(
        (assert.max_imbalance_ratio.unwrap() - d.max_imbalance_ratio).abs() < f64::EPSILON,
        "max_imbalance_ratio must be auto-filled with DEFAULT"
    );
    assert_eq!(
        assert.max_local_dsq_depth,
        Some(d.max_local_dsq_depth),
        "max_local_dsq_depth must be auto-filled with DEFAULT"
    );
    assert_eq!(
        assert.fail_on_stall,
        Some(d.fail_on_stall),
        "fail_on_stall must be auto-filled with DEFAULT"
    );
    assert_eq!(
        assert.sustained_samples,
        Some(d.sustained_samples),
        "sustained_samples must be auto-filled with DEFAULT"
    );
    assert!(
        (assert.max_fallback_rate.unwrap() - d.max_fallback_rate).abs() < f64::EPSILON,
        "max_fallback_rate must be auto-filled with DEFAULT"
    );
    assert!(
        (assert.max_keep_last_rate.unwrap() - d.max_keep_last_rate).abs() < f64::EPSILON,
        "max_keep_last_rate must be auto-filled with DEFAULT"
    );
    assert!(
        assert.enforce_monitor_thresholds,
        "enforce_monitor_thresholds must be set to true"
    );
}

/// Merge semantics pin: true (self) || true (other) → true.
/// Completes the OR truth-table coverage that the false_false /
/// true_left / false_true canaries above pin three of four cells
/// for. The TT cell is the case a future XOR-vs-OR regression would
/// catch only via this canary — false||false=F, false||true=T, and
/// true||false=T pass under XOR semantics too. true||true=T does
/// NOT pass under XOR (XOR would yield F here), so a regression
/// that swapped OR for XOR escapes the existing 3 canaries unless
/// this 4th cell pins it.
#[test]
fn merge_enforce_monitor_thresholds_or_semantics_true_true() {
    let left = Assert::NO_OVERRIDES.with_monitor_defaults();
    let right = Assert::NO_OVERRIDES.with_monitor_defaults();
    let merged = left.merge(&right);
    assert!(
        merged.enforce_monitor_thresholds,
        "true || true must yield true (OR, not XOR)"
    );
}

/// Behavior pin: `with_monitor_defaults()` MUST NOT clobber
/// user-set values for the 6 unset-Option monitor-threshold fields.
/// The auto-fill loop in `with_monitor_defaults` uses `if X.is_none()`
/// guards; a regression that drops the guard and unconditionally
/// stores `Some(default)` would silently overwrite a user's custom
/// thresholds (e.g. test author sets max_imbalance_ratio=2.0 to
/// catch a known schedule-balance issue, with_monitor_defaults
/// silently restores 4.0). The companion canary
/// `with_monitor_defaults_fills_all_unset_threshold_fields` above
/// tests the auto-fill SIDE of the contract; this canary tests
/// the inverse SIDE: user-set values survive.
#[test]
fn with_monitor_defaults_preserves_user_set_values() {
    use crate::monitor::MonitorThresholds;
    let d = MonitorThresholds::DEFAULT;
    // Pick values DIFFERENT from each DEFAULT so the test fails if
    // the auto-fill loop drops its `is_none()` guard and silently
    // overwrites. Differs from DEFAULT by ~50% on each axis where
    // the type allows; bool / counter axes flip / shift respectively.
    let custom_imbalance = d.max_imbalance_ratio * 0.5;
    let custom_local_dsq = d.max_local_dsq_depth * 2;
    let custom_fail_on_stall = !d.fail_on_stall;
    let custom_sustained = d.sustained_samples * 2;
    let custom_fallback = d.max_fallback_rate * 0.5;
    let custom_keep_last = d.max_keep_last_rate * 0.5;
    let assert = Assert::NO_OVERRIDES
        .max_imbalance_ratio(custom_imbalance)
        .max_local_dsq_depth(custom_local_dsq)
        .fail_on_stall(custom_fail_on_stall)
        .sustained_samples(custom_sustained)
        .max_fallback_rate(custom_fallback)
        .max_keep_last_rate(custom_keep_last)
        .with_monitor_defaults();
    assert!(
        (assert.max_imbalance_ratio.unwrap() - custom_imbalance).abs() < f64::EPSILON,
        "max_imbalance_ratio user-set value must survive with_monitor_defaults"
    );
    assert_eq!(
        assert.max_local_dsq_depth,
        Some(custom_local_dsq),
        "max_local_dsq_depth user-set value must survive"
    );
    assert_eq!(
        assert.fail_on_stall,
        Some(custom_fail_on_stall),
        "fail_on_stall user-set value must survive"
    );
    assert_eq!(
        assert.sustained_samples,
        Some(custom_sustained),
        "sustained_samples user-set value must survive"
    );
    assert!(
        (assert.max_fallback_rate.unwrap() - custom_fallback).abs() < f64::EPSILON,
        "max_fallback_rate user-set value must survive"
    );
    assert!(
        (assert.max_keep_last_rate.unwrap() - custom_keep_last).abs() < f64::EPSILON,
        "max_keep_last_rate user-set value must survive"
    );
    assert!(
        assert.enforce_monitor_thresholds,
        "enforce_monitor_thresholds still set to true (the only field with_monitor_defaults unconditionally writes)"
    );
}

/// OR-chain coverage: `has_monitor_thresholds()` MUST return false
/// when every monitor-threshold Option is None. NO_OVERRIDES is
/// the canonical all-None starting point; this canary pins the
/// false-side of the OR-chain in `has_monitor_thresholds`.
#[test]
fn has_monitor_thresholds_false_when_all_none() {
    let v = Assert::NO_OVERRIDES;
    assert!(
        !v.has_monitor_thresholds(),
        "NO_OVERRIDES must report has_monitor_thresholds() == false"
    );
}

/// OR-chain coverage: `has_monitor_thresholds()` MUST return true
/// when ANY single monitor-threshold Option is Some. Parametric
/// over the 6 fields — each iteration sets exactly one field to a
/// non-None value via the builder, asserts has_monitor_thresholds
/// returns true. Catches a regression that drops a field from the
/// `has_monitor_thresholds` OR-chain (the same field-drift class
/// the FailureDumpReport round-trip test catches on the dump
/// side). A dropped field would silently skip monitor enforcement
/// for tests that set only the missed field.
#[test]
fn has_monitor_thresholds_true_when_any_set() {
    // Each closure returns an Assert with exactly one threshold
    // field set. Adding a new MonitorThresholds field requires
    // adding a closure here — that's the maintenance contract.
    let setters: &[(&str, fn() -> Assert)] = &[
        ("max_imbalance_ratio", || {
            Assert::NO_OVERRIDES.max_imbalance_ratio(3.0)
        }),
        ("max_local_dsq_depth", || {
            Assert::NO_OVERRIDES.max_local_dsq_depth(64)
        }),
        ("fail_on_stall", || Assert::NO_OVERRIDES.fail_on_stall(true)),
        ("sustained_samples", || {
            Assert::NO_OVERRIDES.sustained_samples(7)
        }),
        ("max_fallback_rate", || {
            Assert::NO_OVERRIDES.max_fallback_rate(150.0)
        }),
        ("max_keep_last_rate", || {
            Assert::NO_OVERRIDES.max_keep_last_rate(75.0)
        }),
    ];
    for (field, build) in setters {
        let v = build();
        assert!(
            v.has_monitor_thresholds(),
            "has_monitor_thresholds() must return true when only `{field}` is set"
        );
    }
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
    let base = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_imbalance_ratio(4.0);
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(4.0));
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
    let base = Assert::NO_OVERRIDES.check_not_starved();
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
    assert_eq!(merged.not_starved, None);
    assert_eq!(merged.max_imbalance_ratio, Some(2.0));
    assert_eq!(merged.max_fallback_rate, Some(50.0));
    assert_eq!(merged.max_gap_ms, Some(5000));
    assert_eq!(merged.sustained_samples, None);
}

#[test]
fn assert_merge_no_overrides_preserves_base() {
    let base = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_imbalance_ratio(4.0)
        .fail_on_stall(true);
    let merged = base.merge(&Assert::NO_OVERRIDES);
    assert_eq!(merged.not_starved, Some(true));
    assert_eq!(merged.max_imbalance_ratio, Some(4.0));
    assert_eq!(merged.fail_on_stall, Some(true));
}

/// `default_checks()` is `NO_OVERRIDES`, so merging in either
/// direction is the identity.
#[test]
fn assert_merge_no_overrides_is_left_identity() {
    let merged = Assert::NO_OVERRIDES.merge(&Assert::default_checks());
    assert!(merged.not_starved.is_none());
    assert!(merged.max_imbalance_ratio.is_none());
    assert!(merged.max_gap_ms.is_none());
    assert!(merged.isolation.is_none());
}

/// The runtime three-layer chain
/// `default_checks -> scheduler -> test` collapses to
/// `NO_OVERRIDES` when both override layers are also `NO_OVERRIDES`.
#[test]
fn assert_merge_runtime_chain_with_no_overrides_yields_defaults() {
    let scheduler_assert = Assert::NO_OVERRIDES;
    let test_assert = Assert::NO_OVERRIDES;
    let merged = Assert::default_checks()
        .merge(&scheduler_assert)
        .merge(&test_assert);
    assert!(merged.not_starved.is_none());
    assert!(merged.max_imbalance_ratio.is_none());
    assert!(merged.max_local_dsq_depth.is_none());
    assert!(merged.fail_on_stall.is_none());
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
fn assert_default_checks_has_no_worker_checks() {
    assert!(!Assert::default_checks().has_worker_checks());
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
    // Layer 1: defaults (all None now).
    let defaults = Assert::default_checks();

    // Layer 2: scheduler sets worker and benchmark fields.
    let sched = Assert::NO_OVERRIDES
        .max_spread_pct(50.0)
        .max_p99_wake_latency_ns(100_000)
        .max_migration_ratio(0.5)
        .fail_on_stall(true);

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
    // sched's monitor fields survive (test didn't set them).
    assert_eq!(merged.fail_on_stall, Some(true));
}
