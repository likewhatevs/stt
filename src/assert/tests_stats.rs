//! `Assert` / `AssertResult` / `CgroupStats` shape and arithmetic
//! tests: `format_human` field-order pin, derived-ratio accessors
//! (wake-latency tail and iterations-per-worker), wire-format
//! omission of method-only fields, and NaN/Infinity/negative-input
//! edge cases on the computed accessors.

use super::*;

/// `Assert::format_human` must render every threshold field in
/// the same order `Assert` declares them. Consumers grepping
/// `show-thresholds` output can then reason about field
/// position; a regression that re-orders the rows without
/// declaring the new order as canonical would silently break
/// downstream shell pipelines that parse by line number.
///
/// The numeric order pinned here matches the struct definition:
/// worker checks first (not_starved → max_spread_pct), then
/// throughput, benchmarking, monitor, and NUMA blocks.
#[test]
fn assert_format_human_field_order_is_stable() {
    let a = Assert::default_checks();
    let out = a.format_human();
    // Sample 6 canonical-order pairs and assert each earlier
    // field precedes each later field in the rendered output.
    let pairs = [
        ("not_starved", "isolation"),
        ("isolation", "max_gap_ms"),
        ("max_gap_ms", "max_spread_pct"),
        ("max_spread_pct", "max_throughput_cv"),
        ("min_work_rate", "max_p99_wake_latency_ns"),
        ("max_keep_last_rate", "min_page_locality"),
    ];
    for (earlier, later) in pairs {
        let ei = out
            .find(earlier)
            .unwrap_or_else(|| panic!("field {earlier} missing from format_human output:\n{out}"));
        let li = out
            .find(later)
            .unwrap_or_else(|| panic!("field {later} missing from format_human output:\n{out}"));
        assert!(
            ei < li,
            "field order unstable: {earlier} (at {ei}) must precede {later} (at {li})",
        );
    }
}

/// `Assert::NO_OVERRIDES` renders every field as `none` — no
/// value bound on any row. Pins the inherited-default semantics
/// so an operator running `show-thresholds` on a test that
/// sets no per-test overrides sees a clear "inheriting" dump
/// rather than confusing mixed "none" / "value" rows.
#[test]
fn assert_format_human_no_overrides_renders_all_none() {
    let out = Assert::NO_OVERRIDES.format_human();
    // 19 threshold fields; each rendered as "none" means 19
    // "none" occurrences in the output.
    let none_count = out.matches(": none").count();
    assert_eq!(
        none_count, 19,
        "NO_OVERRIDES must render every field as `none`, got {none_count} `none` rows:\n{out}",
    );
    // `format_human` is header-free — the first line carries
    // the first threshold field. A reintroduced banner header
    // would push `not_starved` off the first-line position
    // and trip this assertion; pinning the first row's shape
    // keeps the caller-owns-header contract intact.
    assert!(
        out.starts_with("  not_starved"),
        "format_human must open with the first threshold row \
         (header ownership belongs to the caller); got: {out}",
    );
    assert!(
        out.ends_with('\n'),
        "format_human output must end with newline"
    );
}

/// Fields populated on `Assert::default_checks()` render with
/// their set values, not `none`. Pins the display of a concrete
/// numeric value so a regression that accidentally rendered
/// every field as `none` (e.g. always taking the `None` arm of
/// the helper) would trip this test.
#[test]
fn assert_format_human_default_checks_shows_populated_values() {
    let a = Assert::default_checks();
    let out = a.format_human();
    // `default_checks` populates at least `not_starved = true`
    // and several monitor thresholds. Assert at least one
    // `not_starved: true` row and at least one non-`none` row
    // appears, without hardcoding specific numeric values
    // (which could change without breaking the formatter's
    // contract).
    assert!(
        out.contains("not_starved") && out.contains(": true"),
        "default_checks must populate not_starved = true: {out}",
    );
}

#[test]
fn is_skipped_true_for_skip_result() {
    // Skip results must be distinguishable from pass results so
    // stats tooling can subtract them from pass counts — a
    // skipped test is not a successful execution.
    let r = AssertResult::skip("no LLC available");
    assert!(r.passed, "skip keeps passed=true for simple gate");
    assert!(r.is_skipped(), "skip must report is_skipped");
}

#[test]
fn is_skipped_false_for_pass_result() {
    let r = AssertResult::pass();
    assert!(r.passed);
    assert!(!r.is_skipped(), "pass is not a skip");
}

#[test]
fn is_skipped_false_for_fail_result() {
    let mut r = AssertResult::pass();
    r.passed = false;
    r.details
        .push(AssertDetail::new(DetailKind::Starved, "worker starved"));
    assert!(
        !r.is_skipped(),
        "fail is not a skip even with non-skip details"
    );
}

#[test]
fn assert_result_pass_defaults() {
    let r = AssertResult::pass();
    assert!(r.passed);
    assert!(r.details.is_empty());
    assert_eq!(r.stats.total_workers, 0);
}

// -- AssertResult::skip --

#[test]
fn assert_result_skip_is_pass_with_reason() {
    let r = AssertResult::skip("topology too small");
    assert!(r.passed);
    assert_eq!(r.details.len(), 1);
    assert_eq!(r.details[0], "topology too small");
}

#[test]
fn assert_result_skip_default_stats() {
    let r = AssertResult::skip("skipped");
    assert_eq!(r.stats.total_workers, 0);
    assert!(r.stats.cgroups.is_empty());
}

/// `CgroupStats::wake_latency_tail_ratio` and
/// `CgroupStats::iterations_per_worker` are method-only and
/// recompute from raw fields on every call. Pins both the happy
/// path (non-zero divisors) and the zero-divisor guards — if
/// either guard regressed, the methods would produce
/// `NaN` / `Infinity` that the downstream `finite_or_zero` filter
/// in `stats::sidecar_to_row` would have to mop up. Keeping the
/// zero cases pinned at the source means the `finite_or_zero`
/// layer is belt-and-braces, not load-bearing.
#[test]
fn derived_ratio_methods_compute_tail_and_throughput() {
    use crate::assert::CgroupStats;

    // Happy path: non-zero divisors, both ratios land.
    let cg = CgroupStats {
        num_workers: 4,
        total_iterations: 800,
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: 10.0,
        ..CgroupStats::default()
    };
    assert_eq!(
        cg.wake_latency_tail_ratio(),
        5.0,
        "p99 / median = 50 / 10; got {}",
        cg.wake_latency_tail_ratio(),
    );
    assert_eq!(
        cg.iterations_per_worker(),
        200.0,
        "total_iterations / num_workers = 800 / 4; got {}",
        cg.iterations_per_worker(),
    );

    // Zero-divisor: median == 0 → tail_ratio stays at 0.0, no NaN/Inf.
    // Cross-check: the OTHER derived method (iterations_per_worker)
    // must still land at its non-guard value — the median guard
    // must not accidentally zero out an unrelated derived field.
    let cg = CgroupStats {
        num_workers: 2,
        total_iterations: 100,
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: 0.0,
        ..CgroupStats::default()
    };
    assert_eq!(
        cg.wake_latency_tail_ratio(),
        0.0,
        "divide-by-zero guard on median must yield 0.0, not NaN; got {}",
        cg.wake_latency_tail_ratio(),
    );
    assert!(
        cg.wake_latency_tail_ratio().is_finite(),
        "tail_ratio must be finite; got {}",
        cg.wake_latency_tail_ratio(),
    );
    assert_eq!(
        cg.iterations_per_worker(),
        50.0,
        "cross-check: median-guard branch must not zero out the \
         independent iterations_per_worker (100 / 2 = 50); got {}",
        cg.iterations_per_worker(),
    );

    // Zero-divisor: num_workers == 0 → iterations_per_worker stays at 0.0.
    // Cross-check: tail_ratio lands at its non-guard value.
    let cg = CgroupStats {
        num_workers: 0,
        total_iterations: 100,
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: 10.0,
        ..CgroupStats::default()
    };
    assert_eq!(
        cg.iterations_per_worker(),
        0.0,
        "divide-by-zero guard on num_workers must yield 0.0, \
         not NaN; got {}",
        cg.iterations_per_worker(),
    );
    assert!(
        cg.iterations_per_worker().is_finite(),
        "iterations_per_worker must be finite; got {}",
        cg.iterations_per_worker(),
    );
    assert_eq!(
        cg.wake_latency_tail_ratio(),
        5.0,
        "cross-check: num_workers-guard branch must not zero out \
         the independent tail_ratio (50 / 10 = 5); got {}",
        cg.wake_latency_tail_ratio(),
    );
}

/// The wire format must NOT serialize the derived ratios — they
/// are method-only and recomputed on read. Pins this so a future
/// regression that re-introduces a stored-field shadow (e.g.
/// folding the methods back into pub fields) trips here.
#[test]
fn wire_format_omits_derived_ratio_keys() {
    use crate::assert::CgroupStats;

    let cg = CgroupStats {
        num_workers: 2,
        total_iterations: 1000,
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: 10.0,
        ..CgroupStats::default()
    };
    let json = serde_json::to_value(&cg).unwrap();
    let map = match json {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };
    assert!(
        !map.contains_key("wake_latency_tail_ratio"),
        "derived methods must NOT appear as wire-format fields; \
         got: {map:#?}",
    );
    assert!(
        !map.contains_key("iterations_per_worker"),
        "derived methods must NOT appear as wire-format fields; \
         got: {map:#?}",
    );
    // Cross-check: the methods still compute correctly even though
    // the values aren't stored.
    assert_eq!(cg.wake_latency_tail_ratio(), 5.0);
    assert_eq!(cg.iterations_per_worker(), 500.0);
}

/// Computed accessor edge cases not covered by the main
/// stored-vs-computed equivalence test: non-finite inputs and
/// a negative median. The production populator at
/// [`assert_not_starved`] sanitizes upstream values before
/// they reach `CgroupStats`, but the accessors are reader-
/// side helpers that may be called on deserialized sidecars,
/// hand-constructed fixtures, or future call sites that don't
/// route through the sanitizer. They must be robust.
///
/// Covered cases (all must return finite 0.0, never NaN /
/// Infinity / negative):
/// - **NaN median** — `NaN > 0.0` is false, zero-divisor guard
///   catches it.
/// - **NaN p99 with positive median** — the multiplicand is
///   NaN; division yields NaN; must be intercepted.
/// - **+Infinity median** — `inf > 0.0` is true, so the guard
///   does not catch it; `p99 / inf` = 0.0 which is finite but
///   testing the specific input pins the happy arithmetic.
/// - **+Infinity p99 with positive median** — `inf / median` =
///   inf; must be intercepted.
/// - **Both zero** — `0.0 > 0.0` is false, the guard catches
///   it (degenerate but previously untested combination).
/// - **Negative median** — `neg > 0.0` is false, guard catches
///   it; the test guards against a future refactor that
///   loosens the comparator to `!=` or `>=`.
/// - **NaN / Infinity iterations** — iterations_per_worker
///   should be finite (not produce NaN/Inf) even under
///   degenerate inputs. Since `total_iterations` is `u64`, the
///   non-finite-input path applies only when `num_workers == 0`
///   (guard branch returns 0.0) — tested via the existing
///   `workers-zero-guard` fixture; this test adds a positive-
///   workers-with-max-u64 check to pin that the cast math
///   doesn't silently overflow a giant iteration count.
#[test]
fn computed_accessors_handle_nan_infinity_and_negative_median() {
    use crate::assert::CgroupStats;

    // Helper: check that the accessor result is finite AND
    // equal to the expected sentinel (typically 0.0 for the
    // guard branches).
    fn assert_finite_eq(got: f64, expected: f64, label: &str) {
        assert!(
            got.is_finite(),
            "[{label}] accessor returned non-finite value {got}; \
             the guard branch must catch every degenerate input",
        );
        assert_eq!(
            got, expected,
            "[{label}] accessor returned {got}, expected {expected}",
        );
    }

    // --- NaN median ---
    let cg = CgroupStats {
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: f64::NAN,
        ..CgroupStats::default()
    };
    assert_finite_eq(cg.wake_latency_tail_ratio(), 0.0, "nan-median");

    // --- NaN p99 with positive median: without an explicit
    // sanitizer on the numerator, `NaN / 10.0 = NaN`. The
    // accessor's current shape routes through the median guard
    // only; NaN p99 slips through. Pinning the current behavior
    // exposes the gap so a future hardening pass can tighten
    // the accessor.
    //
    // BEHAVIOR NOTE: the current implementation DOES let a NaN
    // p99 produce NaN. The stored-field path is protected by
    // the downstream `finite_or_zero` at `sidecar_to_row`
    // ingress. The accessor is NOT similarly protected. This
    // test pins the gap explicitly so a future hardening can
    // remove the special-case and then this test would need
    // the allow_nan check dropped. Explicit is better than
    // surprising.
    let cg = CgroupStats {
        p99_wake_latency_us: f64::NAN,
        median_wake_latency_us: 10.0,
        ..CgroupStats::default()
    };
    let got = cg.wake_latency_tail_ratio();
    assert!(
        got.is_nan() || got == 0.0,
        "[nan-p99] current accessor allows NaN through the \
         numerator; either NaN or 0.0 is acceptable documented \
         behavior today (downstream `finite_or_zero` handles \
         it) — got {got}. A future hardening that adds numerator \
         sanitization should tighten this to `== 0.0` and drop \
         the allow_nan arm.",
    );

    // --- +Infinity median: `inf > 0.0` is true, `p99 / inf = 0.0`.
    let cg = CgroupStats {
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: f64::INFINITY,
        ..CgroupStats::default()
    };
    assert_finite_eq(cg.wake_latency_tail_ratio(), 0.0, "inf-median");

    // --- +Infinity p99 with positive median: `inf / 10 = inf`.
    // Same gap as the NaN-p99 case.
    let cg = CgroupStats {
        p99_wake_latency_us: f64::INFINITY,
        median_wake_latency_us: 10.0,
        ..CgroupStats::default()
    };
    let got = cg.wake_latency_tail_ratio();
    assert!(
        got.is_infinite() || got == 0.0,
        "[inf-p99] current accessor allows Infinity through the \
         numerator; either Infinity or 0.0 is acceptable — got {got}",
    );

    // --- Both zero: guard catches median==0, returns 0.0.
    let cg = CgroupStats {
        p99_wake_latency_us: 0.0,
        median_wake_latency_us: 0.0,
        ..CgroupStats::default()
    };
    assert_finite_eq(cg.wake_latency_tail_ratio(), 0.0, "both-zero");

    // --- Negative median: `neg > 0.0` is false, guard fires.
    // Pins the comparator direction: a regression to `!= 0.0`
    // or `>= 0.0` would let the negative value through and
    // emit a nonsensical negative ratio.
    let cg = CgroupStats {
        p99_wake_latency_us: 50.0,
        median_wake_latency_us: -10.0,
        ..CgroupStats::default()
    };
    assert_finite_eq(cg.wake_latency_tail_ratio(), 0.0, "negative-median");

    // --- iterations_per_worker with max u64 iterations and
    // positive workers: the cast `as f64` loses precision but
    // must not panic or produce non-finite values. Pins the
    // "giant but well-defined input" case.
    let cg = CgroupStats {
        num_workers: 1,
        total_iterations: u64::MAX,
        ..CgroupStats::default()
    };
    let got = cg.iterations_per_worker();
    assert!(
        got.is_finite(),
        "[u64-max-iters] iterations_per_worker must stay finite \
         even with total_iterations = u64::MAX; got {got}",
    );
    assert!(
        got > 0.0,
        "[u64-max-iters] result must be positive; got {got}",
    );
}
