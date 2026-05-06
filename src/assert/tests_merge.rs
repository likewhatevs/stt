//! `AssertResult::merge` and the per-field worst-wins / lowest-non-zero
//! / sum aggregation rules for `ScenarioStats`. Every polarity is
//! exercised in both directions so a sign-flip regression surfaces
//! regardless of which side carries the worse value.

use super::tests_common::rpt;
use super::*;

#[test]
fn merge_cgroups() {
    let r1 = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0, 1], 50),
        rpt(2, 1000, 5e9 as u64, 6e8 as u64, &[0, 1], 60),
    ]);
    let r2 = assert_not_starved(&[
        rpt(3, 1000, 5e9 as u64, 25e8 as u64, &[2, 3], 50),
        rpt(4, 1000, 5e9 as u64, 26e8 as u64, &[2, 3], 50),
    ]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(m.stats.cgroups.len(), 2);
    assert_eq!(m.stats.total_workers, 4);
    assert!(m.passed, "diff cgroups diff off_cpu should pass");
}

#[test]
fn merge_takes_worst_gap() {
    let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 100)]);
    let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 500)]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(m.stats.worst_gap_ms, 500);
    assert_eq!(m.stats.worst_gap_cpu, 1);
}

/// Reverse direction of [`merge_takes_worst_gap`]: the forward
/// case picks `other`'s larger gap and must re-couple to
/// `other`'s CPU. This test pins the self-retains branch — when
/// `self.worst_gap_ms > other.worst_gap_ms`, `worst_gap_cpu`
/// must stay on `self`'s CPU and NOT leak over to `other`'s.
///
/// Without both directions pinned, a regression that always
/// overwrote `worst_gap_cpu` from `other` (regardless of which
/// gap won) would pass the forward test — the forward case
/// already asks for `other`'s cpu anyway — and land silently.
/// Pairing the two directions is what actually guards the
/// "coupled fields stay coupled" invariant from the merge doc.
#[test]
fn merge_takes_worst_gap_reverse_self_retains() {
    // r1 has the larger gap (700ms on cpu 0); r2 has the smaller
    // gap (200ms on cpu 1). After merge, self must keep both
    // its 700ms AND its cpu 0 — not adopt cpu 1 from the
    // loser's report.
    let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 700)]);
    let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 200)]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(
        m.stats.worst_gap_ms, 700,
        "self's larger gap must be retained",
    );
    assert_eq!(
        m.stats.worst_gap_cpu, 0,
        "worst_gap_cpu must stay coupled to self's worst_gap_ms — \
         a regression overwriting cpu from other would set this to 1",
    );
}

#[test]
fn merge_takes_worst_spread() {
    let r1 = assert_not_starved(&[
        rpt(1, 1000, 5e9 as u64, 1e9 as u64, &[0], 50),
        rpt(2, 1000, 5e9 as u64, 12e8 as u64, &[0], 50),
    ]); // spread = 4%
    let r2 = assert_not_starved(&[
        rpt(3, 1000, 5e9 as u64, 1e9 as u64, &[1], 50),
        rpt(4, 1000, 5e9 as u64, 15e8 as u64, &[1], 50),
    ]); // spread = 10%
    let mut m = r1;
    m.merge(r2);
    assert!((m.stats.worst_spread - 10.0).abs() < 0.1);
}

#[test]
fn merge_skip_plus_pass_demotes_skip() {
    let mut a = AssertResult::skip("optional");
    let b = AssertResult::pass();
    a.merge(b);
    assert!(!a.skipped);
    assert!(a.passed);
}

#[test]
fn merge_skip_plus_fail_is_fail_not_skip() {
    let mut a = AssertResult::skip("topo missing");
    let mut b = AssertResult::pass();
    b.passed = false;
    a.merge(b);
    assert!(!a.passed);
    assert!(!a.skipped);
}

#[test]
fn merge_accumulates_totals() {
    let r1 = assert_not_starved(&[rpt(1, 1000, 5e9 as u64, 5e8 as u64, &[0], 50)]);
    let r2 = assert_not_starved(&[rpt(2, 1000, 5e9 as u64, 5e8 as u64, &[1], 50)]);
    let mut m = r1;
    m.merge(r2);
    assert_eq!(m.stats.total_workers, 2);
    assert_eq!(m.stats.total_cpus, 2);
}

/// Multi-cgroup merge-aggregation contract: merging `N > 2`
/// `AssertResult`s (each carrying one populated `CgroupStats`
/// plus `ScenarioStats` headline fields) must:
///   - append every per-cgroup entry into `stats.cgroups` in
///     merge order, preserving cardinality;
///   - pick the worst value of every higher-is-worse
///     `worst_*` field across all merged cgroups;
///   - pick the lowest-non-zero value of `worst_page_locality`
///     and `worst_iterations_per_worker` (0.0 is the unreported
///     sentinel for both fields, matching the accumulator-pass
///     convention in `AssertResult::pass().merge(real)`);
///   - SUM `total_iterations` across all cgroups, not max it.
///
/// Sibling `merge_scenario_stats_worst_wins_and_iterations_sum`
/// already covers the 2-cgroup case with headline fields only;
/// this test exercises 3 cgroups AND the per-cgroup accumulator
/// (`stats.cgroups.extend`) so a regression that dropped
/// cgroups, clobbered the per-cgroup vector, or flipped one of
/// the polarity folds surfaces in the stronger form.
#[test]
fn merge_three_cgroups_worst_wins_and_iterations_sum() {
    fn mk(
        worst_spread: f64,
        worst_mig: f64,
        worst_p99_us: f64,
        total_iters: u64,
        page_locality: f64,
        iters_per_worker: f64,
        cg_total_iters: u64,
    ) -> AssertResult {
        let cg = CgroupStats {
            total_iterations: cg_total_iters,
            page_locality,
            ..CgroupStats::default()
        };
        // `iters_per_worker` flows into the ScenarioStats roll-up
        // below; the per-cgroup [`CgroupStats::iterations_per_worker`]
        // is now method-only and recomputed on read from
        // `total_iterations / num_workers`.
        AssertResult {
            passed: true,
            skipped: false,
            details: vec![],
            stats: ScenarioStats {
                total_iterations: total_iters,
                worst_spread,
                worst_migration_ratio: worst_mig,
                worst_p99_wake_latency_us: worst_p99_us,
                worst_page_locality: page_locality,
                worst_iterations_per_worker: iters_per_worker,
                cgroups: vec![cg],
                ..ScenarioStats::default()
            },
            measurements: std::collections::BTreeMap::new(),
        }
    }

    // Three cgroups with deliberately heterogeneous values so
    // each `worst_*` aggregation is sourced from a DIFFERENT
    // cgroup — a regression that folded only within-cgroup
    // would still produce a plausible-looking aggregate on a
    // 2-cgroup test but would fail here.
    let mut acc = mk(10.0, 0.1, 50.0, 100, 0.8, 300.0, 100);
    acc.merge(mk(5.0, 0.3, 20.0, 200, 0.5, 150.0, 200));
    acc.merge(mk(20.0, 0.2, 70.0, 400, 0.9, 500.0, 400));

    let s = &acc.stats;
    assert_eq!(
        s.cgroups.len(),
        3,
        "3 cgroups must accumulate; a missing entry means stats.cgroups.extend dropped a merge",
    );
    // Per-cgroup order is preserved (merge calls, in order):
    assert_eq!(s.cgroups[0].total_iterations, 100);
    assert_eq!(s.cgroups[1].total_iterations, 200);
    assert_eq!(s.cgroups[2].total_iterations, 400);

    // Worst-wins across 3 cgroups (higher-is-worse):
    assert_eq!(s.worst_spread, 20.0, "third cgroup's 20.0 is worst");
    assert_eq!(s.worst_migration_ratio, 0.3, "second cgroup's 0.3 is worst");
    assert_eq!(
        s.worst_p99_wake_latency_us, 70.0,
        "third cgroup's 70.0us p99 is worst",
    );
    // Lower-is-worse rollups across 3 cgroups (every value is
    // strictly positive so the sentinel branch is never taken;
    // both fields use `fold_lowest_nonzero`):
    assert_eq!(
        s.worst_page_locality, 0.5,
        "second cgroup's 0.5 is the lowest-non-zero — 0 sentinel never wins",
    );
    assert_eq!(
        s.worst_iterations_per_worker, 150.0,
        "second cgroup's 150 is the lowest-non-zero per-worker throughput",
    );
    // total_iterations SUMS across cgroups, not maxes:
    assert_eq!(
        s.total_iterations,
        100 + 200 + 400,
        "total_iterations must sum (not max) across all merged cgroups",
    );
}

#[test]
fn merge_scenario_stats_worst_wins_and_iterations_sum() {
    // Aggregates-across-cgroups contract: every `worst_*` field on
    // ScenarioStats takes the larger value between the two cgroups,
    // and `total_iterations` sums. Exercises fields that are not
    // covered by the narrower merge_takes_worst_* tests: the wake-
    // latency trio, the run-delay pair, the migration ratio, and
    // the cross-node migration ratio.
    let mut a = AssertResult::pass();
    a.stats.total_iterations = 100;
    a.stats.worst_spread = 5.0;
    a.stats.worst_migration_ratio = 0.1;
    a.stats.worst_p99_wake_latency_us = 20.0;
    a.stats.worst_median_wake_latency_us = 10.0;
    a.stats.worst_wake_latency_cv = 0.2;
    a.stats.worst_run_delay_us = 50.0;
    a.stats.worst_mean_run_delay_us = 30.0;
    a.stats.worst_cross_node_migration_ratio = 0.05;

    let mut b = AssertResult::pass();
    b.stats.total_iterations = 400;
    b.stats.worst_spread = 15.0;
    b.stats.worst_migration_ratio = 0.4;
    b.stats.worst_p99_wake_latency_us = 80.0;
    b.stats.worst_median_wake_latency_us = 40.0;
    b.stats.worst_wake_latency_cv = 0.5;
    b.stats.worst_run_delay_us = 120.0;
    b.stats.worst_mean_run_delay_us = 90.0;
    b.stats.worst_cross_node_migration_ratio = 0.25;

    a.merge(b);

    assert_eq!(a.stats.total_iterations, 500);
    assert_eq!(a.stats.worst_spread, 15.0);
    assert_eq!(a.stats.worst_migration_ratio, 0.4);
    assert_eq!(a.stats.worst_p99_wake_latency_us, 80.0);
    assert_eq!(a.stats.worst_median_wake_latency_us, 40.0);
    assert_eq!(a.stats.worst_wake_latency_cv, 0.5);
    assert_eq!(a.stats.worst_run_delay_us, 120.0);
    assert_eq!(a.stats.worst_mean_run_delay_us, 90.0);
    assert_eq!(a.stats.worst_cross_node_migration_ratio, 0.25);
}

/// `ScenarioStats::merge` rolls up the new derived-ratio fields
/// across cgroups with opposite polarities: `worst_wake_latency_tail_ratio`
/// is higher-is-worse (max), `worst_iterations_per_worker` is
/// lower-is-worse (`fold_lowest_nonzero` — 0.0 is the unreported
/// sentinel matching the accumulator-pass convention; the
/// `AssertResult::pass().merge(real)` pattern relies on a
/// positive `other` overriding `self`'s default-zero rather
/// than being masked by it).  A regression that merged either
/// with the wrong polarity would surface a regression as an
/// improvement or vice versa — exactly the kind of sign-flip
/// that would silently break `stats compare`.
#[test]
fn merge_derived_ratios_use_correct_polarities() {
    let mut a = AssertResult::pass();
    a.stats.worst_wake_latency_tail_ratio = 2.0;
    a.stats.worst_iterations_per_worker = 500.0;

    let mut b = AssertResult::pass();
    b.stats.worst_wake_latency_tail_ratio = 8.0;
    b.stats.worst_iterations_per_worker = 100.0;

    a.merge(b);

    assert_eq!(
        a.stats.worst_wake_latency_tail_ratio, 8.0,
        "tail ratio uses max — 8.0 is worse than 2.0 (more \
         amplification); got {}",
        a.stats.worst_wake_latency_tail_ratio,
    );
    assert_eq!(
        a.stats.worst_iterations_per_worker, 100.0,
        "iterations_per_worker uses lowest-non-zero — 100.0 is \
         worse than 500.0 (less throughput per worker); got {}",
        a.stats.worst_iterations_per_worker,
    );

    // Sentinel-zero convention, direction 1: a 0.0 reading on
    // `other` is the unreported sentinel and MUST NOT clobber
    // self's positive measurement. `fold_lowest_nonzero` keeps
    // self=300 when other=0.
    let mut c = AssertResult::pass();
    c.stats.worst_iterations_per_worker = 300.0;
    let mut empty = AssertResult::pass();
    empty.stats.worst_iterations_per_worker = 0.0;
    c.merge(empty);
    assert_eq!(
        c.stats.worst_iterations_per_worker, 300.0,
        "self=300 must be retained when other=0 (unreported \
         sentinel) — a plain min would let the sentinel \
         clobber the real reading; got {}",
        c.stats.worst_iterations_per_worker,
    );

    // Sentinel-zero convention, direction 2: the symmetric
    // case where `self` starts at 0.0 (the accumulator-default
    // sentinel from `AssertResult::pass()`) and `other`
    // reports a positive reading. self must adopt other's
    // measurement; this is the load-bearing case for
    // `AssertResult::pass().merge(real)`.
    let mut d = AssertResult::pass();
    d.stats.worst_iterations_per_worker = 0.0;
    let mut real = AssertResult::pass();
    real.stats.worst_iterations_per_worker = 300.0;
    d.merge(real);
    assert_eq!(
        d.stats.worst_iterations_per_worker, 300.0,
        "self=0 (accumulator sentinel) must adopt other=300 \
         — the `AssertResult::pass().merge(real)` pattern \
         depends on this; got {}",
        d.stats.worst_iterations_per_worker,
    );

    // Both-zero: no positive reading on either side, the
    // sentinel-fold keeps the field at 0.0.
    let mut e = AssertResult::pass();
    e.stats.worst_iterations_per_worker = 0.0;
    let mut f = AssertResult::pass();
    f.stats.worst_iterations_per_worker = 0.0;
    e.merge(f);
    assert_eq!(
        e.stats.worst_iterations_per_worker, 0.0,
        "both-zero must stay zero; got {}",
        e.stats.worst_iterations_per_worker,
    );

    // Tail-ratio polarity, reverse direction: when `self`
    // starts at the higher value and `other` is smaller,
    // `self` must retain its larger worst. Pair with the
    // forward direction above (self=2, other=8 → 8) so both
    // branches of the `.max()` are pinned — otherwise a
    // regression that silently flipped to `.min()` would
    // pass the forward-direction assertion and surface
    // only here.
    let mut g = AssertResult::pass();
    g.stats.worst_wake_latency_tail_ratio = 8.0;
    let mut h = AssertResult::pass();
    h.stats.worst_wake_latency_tail_ratio = 2.0;
    g.merge(h);
    assert_eq!(
        g.stats.worst_wake_latency_tail_ratio, 8.0,
        "tail_ratio uses max: self=8.0, other=2.0 → self \
         retains 8.0 (higher is worse); got {}",
        g.stats.worst_wake_latency_tail_ratio,
    );
}

#[test]
fn merge_scenario_stats_worst_wins_when_other_is_smaller() {
    // Symmetric case: when `other` reports smaller values, `self`
    // retains its larger worst. Covers the "self wins" branch of
    // every scalar worst-comparison in merge (9 fields total:
    // 8 `.max()` calls + the coupled `worst_gap_ms` guard).
    let mut a = AssertResult::pass();
    a.stats.worst_spread = 30.0;
    a.stats.worst_gap_ms = 500;
    a.stats.worst_gap_cpu = 7;
    a.stats.worst_migration_ratio = 0.9;
    a.stats.worst_p99_wake_latency_us = 100.0;
    a.stats.worst_median_wake_latency_us = 60.0;
    a.stats.worst_wake_latency_cv = 0.7;
    a.stats.worst_run_delay_us = 300.0;
    a.stats.worst_mean_run_delay_us = 200.0;
    a.stats.worst_cross_node_migration_ratio = 0.35;
    a.stats.total_iterations = 500;

    let mut b = AssertResult::pass();
    b.stats.worst_spread = 5.0;
    b.stats.worst_gap_ms = 100;
    b.stats.worst_gap_cpu = 3;
    b.stats.worst_migration_ratio = 0.1;
    b.stats.worst_p99_wake_latency_us = 10.0;
    b.stats.worst_median_wake_latency_us = 5.0;
    b.stats.worst_wake_latency_cv = 0.1;
    b.stats.worst_run_delay_us = 40.0;
    b.stats.worst_mean_run_delay_us = 20.0;
    b.stats.worst_cross_node_migration_ratio = 0.05;
    b.stats.total_iterations = 50;

    a.merge(b);

    assert_eq!(a.stats.worst_spread, 30.0);
    assert_eq!(a.stats.worst_gap_ms, 500);
    // `worst_gap_cpu` stays 7: coupling means it retains `self`'s
    // index when `self` wins on `worst_gap_ms`.
    assert_eq!(a.stats.worst_gap_cpu, 7);
    assert_eq!(a.stats.worst_migration_ratio, 0.9);
    assert_eq!(a.stats.worst_p99_wake_latency_us, 100.0);
    assert_eq!(a.stats.worst_median_wake_latency_us, 60.0);
    assert_eq!(a.stats.worst_wake_latency_cv, 0.7);
    assert_eq!(a.stats.worst_run_delay_us, 300.0);
    assert_eq!(a.stats.worst_mean_run_delay_us, 200.0);
    assert_eq!(a.stats.worst_cross_node_migration_ratio, 0.35);
    // Totals always sum, independent of worst-wins direction.
    assert_eq!(a.stats.total_iterations, 550);
}

#[test]
fn merge_worst_page_locality_lowest_non_zero() {
    // `worst_page_locality` can't use plain `.min()` because 0.0
    // is the "unreported" sentinel — a fresh cgroup with no NUMA
    // readings would otherwise clobber a real reading from a
    // reporting cgroup. The merge instead takes the lowest
    // non-zero value.

    // (a) self=0.0 (unreported) + other=0.8 (reported) → 0.8.
    let mut a = AssertResult::pass();
    a.stats.worst_page_locality = 0.0;
    let mut b = AssertResult::pass();
    b.stats.worst_page_locality = 0.8;
    a.merge(b);
    assert_eq!(
        a.stats.worst_page_locality, 0.8,
        "unreported self must adopt other's reading"
    );

    // (b) self=0.6 + other=0.8 → 0.6 (self's lower reading wins).
    let mut a = AssertResult::pass();
    a.stats.worst_page_locality = 0.6;
    let mut b = AssertResult::pass();
    b.stats.worst_page_locality = 0.8;
    a.merge(b);
    assert_eq!(
        a.stats.worst_page_locality, 0.6,
        "lower non-zero reading wins across cgroups"
    );

    // (c) self=0.8 (reported) + other=0.0 (unreported) → 0.8.
    // Plain `.min()` would select 0.0 here — the guard rejects
    // other's sentinel instead of overwriting self.
    let mut a = AssertResult::pass();
    a.stats.worst_page_locality = 0.8;
    let mut b = AssertResult::pass();
    b.stats.worst_page_locality = 0.0;
    a.merge(b);
    assert_eq!(
        a.stats.worst_page_locality, 0.8,
        "unreported other must not clobber self's reading"
    );
}

#[test]
fn merge_ext_metrics_higher_is_worse_takes_max() {
    // "worst_spread" is registered with higher_is_worse=true → merge max.
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("worst_spread".into(), 10.0);
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("worst_spread".into(), 42.0);
    a.merge(b);
    assert_eq!(a.stats.ext_metrics["worst_spread"], 42.0);
}

#[test]
fn merge_ext_metrics_higher_is_better_takes_min() {
    // Regression: "total_iterations" is registered with
    // higher_is_worse=false. Merge must take min (worst case)
    // rather than max (best case). Previously returned 42.0.
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("total_iterations".into(), 10.0);
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("total_iterations".into(), 42.0);
    a.merge(b);
    assert_eq!(
        a.stats.ext_metrics["total_iterations"], 10.0,
        "higher_is_worse=false must take min on merge"
    );
}

#[test]
fn merge_ext_metrics_unknown_metric_defaults_to_max() {
    // Unregistered metric names fall back to max (conservative —
    // treat as higher-is-worse until a MetricDef is registered).
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("unknown_metric".into(), 10.0);
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("unknown_metric".into(), 42.0);
    a.merge(b);
    assert_eq!(a.stats.ext_metrics["unknown_metric"], 42.0);
}

#[test]
fn merge_ext_metrics_first_insert_uses_other_value() {
    // When the key is absent on self, insert other's value verbatim
    // regardless of polarity (no prior value to compare against).
    let mut a = AssertResult::pass();
    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("total_iterations".into(), 77.0);
    a.merge(b);
    assert_eq!(a.stats.ext_metrics["total_iterations"], 77.0);
}

#[test]
fn merge_pass_and_fail() {
    let pass = AssertResult::pass();
    let mut fail = AssertResult::pass();
    fail.passed = false;
    fail.details.push("something failed".into());

    let mut merged = pass;
    merged.merge(fail);
    assert!(!merged.passed, "merging pass+fail must produce fail");
    assert!(
        merged
            .details
            .iter()
            .any(|d| d.contains("something failed"))
    );
}

#[test]
fn merge_fail_and_pass() {
    let mut fail = AssertResult::pass();
    fail.passed = false;
    fail.details.push("first failed".into());
    let pass = AssertResult::pass();

    let mut merged = fail;
    merged.merge(pass);
    assert!(!merged.passed, "merging fail+pass must produce fail");
}

#[test]
fn assert_result_merge_combines_stats() {
    let mut a = AssertResult {
        passed: true,
        skipped: false,
        details: vec!["a".into()],
        stats: ScenarioStats {
            cgroups: vec![],
            total_workers: 2,
            total_cpus: 4,
            total_migrations: 10,
            worst_spread: 5.0,
            worst_gap_ms: 100,
            worst_gap_cpu: 0,
            ..Default::default()
        },
        measurements: std::collections::BTreeMap::new(),
    };
    let b = AssertResult {
        passed: false,
        skipped: false,
        details: vec!["b".into()],
        stats: ScenarioStats {
            cgroups: vec![],
            total_workers: 3,
            total_cpus: 6,
            total_migrations: 20,
            worst_spread: 15.0,
            worst_gap_ms: 500,
            worst_gap_cpu: 2,
            ..Default::default()
        },
        measurements: std::collections::BTreeMap::new(),
    };
    a.merge(b);
    assert!(!a.passed);
    assert_eq!(a.details, vec!["a", "b"]);
    assert_eq!(a.stats.total_workers, 5);
    assert_eq!(a.stats.total_cpus, 10);
    assert_eq!(a.stats.total_migrations, 30);
    assert_eq!(a.stats.worst_spread, 15.0);
    assert_eq!(a.stats.worst_gap_ms, 500);
    assert_eq!(a.stats.worst_gap_cpu, 2);
}

// -- AssertResult::merge ext_metrics --

#[test]
fn assert_result_merge_ext_metrics_max_value() {
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("latency".into(), 10.0);
    a.stats.ext_metrics.insert("throughput".into(), 100.0);

    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("latency".into(), 20.0);
    b.stats.ext_metrics.insert("jitter".into(), 5.0);

    a.merge(b);
    assert_eq!(a.stats.ext_metrics["latency"], 20.0);
    assert_eq!(a.stats.ext_metrics["throughput"], 100.0);
    assert_eq!(a.stats.ext_metrics["jitter"], 5.0);
}

#[test]
fn assert_result_merge_ext_metrics_keeps_larger() {
    let mut a = AssertResult::pass();
    a.stats.ext_metrics.insert("x".into(), 50.0);

    let mut b = AssertResult::pass();
    b.stats.ext_metrics.insert("x".into(), 30.0);

    a.merge(b);
    assert_eq!(a.stats.ext_metrics["x"], 50.0);
}
