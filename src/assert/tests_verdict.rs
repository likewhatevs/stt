//! `Verdict` / `ClaimBuilder` / `SetClaim` / `SeqClaim` and the
//! `claim!` macro. Coverage mirrors what the previous
//! `Expect` / `Checks` tests pinned, expressed in the new
//! claim-based shape: every comparator on every supported type
//! bound (PartialEq, PartialOrd, f64-special, container).

use super::*;
use crate::claim;
use crate::workload::{WorkerReport, WorkerReportClaim};

/// `AssertResult::note` records a `DetailKind::Note` detail
/// without flipping `passed` or `skipped`. Tests downstream of
/// note() rely on this — a sidecar consumer that filters notes
/// from genuine failures depends on the verdict bits being
/// untouched while the detail is appended.
#[test]
fn assert_result_note_does_not_flip_passed_or_skipped() {
    let mut r = AssertResult::pass();
    let was_passed = r.passed;
    let was_skipped = r.skipped;
    r.note("observed worker.iterations=12345");
    assert_eq!(r.passed, was_passed);
    assert_eq!(r.skipped, was_skipped);
    assert_eq!(r.details.len(), 1);
    assert_eq!(r.details[0].kind, DetailKind::Note);
    assert!(r.details[0].message.contains("worker.iterations"));

    // Same on a skip-only result.
    let mut r = AssertResult::skip("topo missing");
    r.note("topo had 2 LLCs, test wants 4");
    assert!(r.passed);
    assert!(r.skipped);
    // skip pushed a Skip detail; note pushed a Note detail.
    // Two details, one each kind.
    assert_eq!(r.details.len(), 2);
    assert_eq!(r.details[0].kind, DetailKind::Skip);
    assert_eq!(r.details[1].kind, DetailKind::Note);
}

/// `AssertResult::with_note` is the builder-style sibling of
/// `note`. Same invariant: never flips verdict, appends Note
/// detail, returns the same owned shape.
#[test]
fn assert_result_with_note_preserves_verdict() {
    let r = AssertResult::pass().with_note("max_wchar=6543");
    assert!(r.passed);
    assert!(!r.skipped);
    assert_eq!(r.details.len(), 1);
    assert_eq!(r.details[0].kind, DetailKind::Note);
    assert!(r.details[0].message.contains("max_wchar=6543"));
}

/// Note kind survives serde round-trip — sidecar consumers
/// match on DetailKind::Note structurally, so the wire format
/// must preserve the variant.
#[test]
fn note_kind_survives_serde_roundtrip() {
    let r = AssertResult::pass().with_note("snapshot=disabled");
    let json = serde_json::to_string(&r).unwrap();
    let r2: AssertResult = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.details.len(), 1);
    assert_eq!(r2.details[0].kind, DetailKind::Note);
    assert!(r2.details[0].message.contains("snapshot=disabled"));
}

// -- Verdict pointwise-claim API -------------------------------------

#[test]
fn verdict_empty_is_passing() {
    let r = Verdict::new().into_result();
    assert!(r.passed);
    assert!(r.details.is_empty());
    assert_eq!(r.stats.total_workers, 0);
}

#[test]
fn verdict_default_matches_new() {
    let d = Verdict::default();
    let n = Verdict::new();
    assert_eq!(d.passed(), n.passed());
    assert_eq!(d.detail_count(), n.detail_count());
}

#[test]
fn verdict_assert_verdict_attaches_threshold_config() {
    let v = Assert::defaults().verdict();
    assert!(v.passed());
    assert!(v.assert().is_some());

    let v = Verdict::new();
    assert!(v.assert().is_none());
}

#[test]
fn verdict_passing_into_result_matches_assert_result_pass() {
    let r1 = Verdict::new().into_result();
    let r2 = AssertResult::pass();
    assert_eq!(r1.passed, r2.passed);
    assert_eq!(r1.skipped, r2.skipped);
    assert_eq!(r1.details, r2.details);
}

#[test]
fn claim_eq_pass_returns_passing_verdict() {
    let mut v = Verdict::new();
    let answer = 42u64;
    claim!(v, answer).eq(42);
    let r = v.into_result();
    assert!(r.passed);
    assert!(
        r.details.is_empty(),
        "passing comparator must not push a detail",
    );
}

#[test]
fn claim_eq_fail_names_subject_and_values() {
    let mut v = Verdict::new();
    let answer = 42u64;
    claim!(v, answer).eq(7);
    let r = v.into_result();
    assert!(!r.passed);
    assert_eq!(r.details.len(), 1);
    let d = &r.details[0];
    assert_eq!(d.kind, DetailKind::Other);
    assert!(d.message.contains("answer"), "msg: {}", d.message);
    assert!(d.message.contains("expected"), "msg: {}", d.message);
    assert!(d.message.contains("was"), "msg: {}", d.message);
    assert!(d.message.contains("42"), "msg: {}", d.message);
    assert!(d.message.contains('7'), "msg: {}", d.message);
}

#[test]
fn claim_ne_pass_and_fail() {
    let mut v = Verdict::new();
    let flag_pass = 0u64;
    claim!(v, flag_pass).ne(1);
    assert!(v.passed());

    let mut v = Verdict::new();
    let flag_fail = 1u64;
    claim!(v, flag_fail).ne(1);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("flag_fail"));
    assert!(r.details[0].message.contains("!="));
}

#[test]
fn claim_at_least_boundary_is_inclusive() {
    let mut v = Verdict::new();
    let counter = 100u64;
    claim!(v, counter).at_least(100);
    assert!(v.passed());

    let mut v = Verdict::new();
    let counter = 99u64;
    claim!(v, counter).at_least(100);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("at least 100"));
    assert!(r.details[0].message.contains("counter"));
}

#[test]
fn claim_at_most_boundary_is_inclusive() {
    let mut v = Verdict::new();
    let counter = 100u64;
    claim!(v, counter).at_most(100);
    assert!(v.passed());

    let mut v = Verdict::new();
    let counter = 101u64;
    claim!(v, counter).at_most(100);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("at most 100"));
}

#[test]
fn claim_lt_strict_upper_bound() {
    let mut v = Verdict::new();
    let x = 99u64;
    claim!(v, x).lt(100);
    assert!(v.passed());

    let mut v = Verdict::new();
    let x = 100u64;
    claim!(v, x).lt(100);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("less than 100"));
}

#[test]
fn claim_gt_strict_lower_bound() {
    let mut v = Verdict::new();
    let x = 101u64;
    claim!(v, x).gt(100);
    assert!(v.passed());

    let mut v = Verdict::new();
    let x = 100u64;
    claim!(v, x).gt(100);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("greater than 100"));
}

#[test]
fn claim_between_inclusive_on_both_ends() {
    let mut v = Verdict::new();
    let lo = 10u64;
    let hi = 20u64;
    let mid = 15u64;
    claim!(v, lo).between(10, 20);
    claim!(v, hi).between(10, 20);
    claim!(v, mid).between(10, 20);
    assert!(v.passed());

    let mut v = Verdict::new();
    let below = 9u64;
    claim!(v, below).between(10, 20);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("[10, 20]"));
}

#[test]
fn claim_between_inverted_interval_fails_with_visible_typo() {
    let mut v = Verdict::new();
    let x = 15u64;
    claim!(v, x).between(20, 10);
    let r = v.into_result();
    assert!(!r.passed);
    let msg = &r.details[0].message;
    assert!(msg.contains("caller error"), "msg: {msg}");
    assert!(msg.contains("interval inverted"), "msg: {msg}");
    assert!(msg.contains("lo=20"), "msg: {msg}");
    assert!(msg.contains("hi=10"), "msg: {msg}");
}

#[test]
fn claim_kind_override_is_persisted_to_detail() {
    let mut v = Verdict::new();
    let p99 = 5000u64;
    claim!(v, p99).kind(DetailKind::Benchmark).at_most(1000);
    let r = v.into_result();
    assert!(!r.passed);
    assert_eq!(r.details[0].kind, DetailKind::Benchmark);
}

#[test]
fn claim_default_kind_is_other() {
    let mut v = Verdict::new();
    let anything = 1u64;
    claim!(v, anything).eq(2);
    let r = v.into_result();
    assert!(!r.passed);
    assert_eq!(r.details[0].kind, DetailKind::Other);
}

#[test]
fn claim_works_across_concrete_types() {
    let mut v = Verdict::new();
    let counter = 1u64;
    let pid = 42i32;
    let len = 3usize;
    claim!(v, counter).eq(1);
    claim!(v, pid).at_least(0);
    claim!(v, len).between(1, 5);
    assert!(v.passed());
}

#[test]
fn claim_is_finite_passes_for_normal_values() {
    for v_val in [0.0_f64, 1.0, -1.0, 1e308, -1e308] {
        let mut v = Verdict::new();
        let x = v_val;
        claim!(v, x).is_finite();
        assert!(v.passed(), "{v_val} should be finite");
    }
}

#[test]
fn claim_is_finite_fails_for_nan_and_infinities() {
    for v_val in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut v = Verdict::new();
        let x = v_val;
        claim!(v, x).is_finite();
        let r = v.into_result();
        assert!(!r.passed, "{v_val} must fail is_finite");
        assert!(r.details[0].message.contains("expected finite"));
    }
}

#[test]
fn claim_near_inclusive_at_tolerance_boundary() {
    let mut v = Verdict::new();
    let exact = 1.0_f64;
    let on_edge = 1.001_f64;
    claim!(v, exact).near(1.0, 0.001);
    claim!(v, on_edge).near(1.0, 0.001);
    assert!(v.passed());

    let mut v = Verdict::new();
    let outside = 1.002_f64;
    claim!(v, outside).near(1.0, 0.001);
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("near 1"));
}

#[test]
fn claim_near_nan_input_fails() {
    let mut v = Verdict::new();
    let nan = f64::NAN;
    claim!(v, nan).near(1.0, 0.5);
    assert!(!v.passed());
}

#[test]
fn claim_near_negative_tolerance_is_caller_error() {
    let mut v = Verdict::new();
    let exact = 1.0_f64;
    claim!(v, exact).near(1.0, -0.001);
    let r = v.into_result();
    assert!(
        !r.passed,
        "negative tolerance must surface as a caller-error fail",
    );
    let msg = &r.details[0].message;
    assert!(msg.contains("caller error"), "msg: {msg}");
    assert!(msg.contains("tolerance negative"), "msg: {msg}");
    assert!(msg.contains("-0.001"), "msg: {msg}");
}

#[test]
fn claim_near_handles_infinity_equality() {
    let mut v = Verdict::new();
    let pos_inf = f64::INFINITY;
    let neg_inf = f64::NEG_INFINITY;
    claim!(v, pos_inf).near(f64::INFINITY, 0.001);
    claim!(v, neg_inf).near(f64::NEG_INFINITY, 0.001);
    assert!(
        v.passed(),
        "infinity == infinity must pass near() despite NaN diff",
    );

    let mut v = Verdict::new();
    let pos_inf = f64::INFINITY;
    claim!(v, pos_inf).near(f64::NEG_INFINITY, 0.001);
    assert!(!v.passed());
}

#[test]
fn verdict_continues_past_failure_and_accumulates_all_details() {
    let mut v = Verdict::new();
    let a = 5u64;
    let b = 200u64;
    let c = 42i32;
    let d = 7u64;
    claim!(v, a).at_least(50); // fail
    claim!(v, b).at_most(100); // fail
    claim!(v, c).eq(42); // pass
    claim!(v, d).between(10, 20); // fail
    let r = v.into_result();
    assert!(!r.passed);
    assert_eq!(
        r.details.len(),
        3,
        "exactly the 3 failing claims must record details: {:?}",
        r.details,
    );
    assert!(r.details.iter().any(|d| d.message.contains("a:")));
    assert!(r.details.iter().any(|d| d.message.contains("b:")));
    assert!(r.details.iter().any(|d| d.message.contains("d:")));
    assert!(
        !r.details.iter().any(|d| d.message.contains("c:")),
        "passing claim must not push a detail: {:?}",
        r.details,
    );
}

#[test]
fn verdict_per_claim_kind_override_routes_to_detail() {
    let mut v = Verdict::new();
    let p99 = 5000u64;
    let locality = 0.5_f64;
    claim!(v, p99).kind(DetailKind::Benchmark).at_most(1000);
    claim!(v, locality)
        .kind(DetailKind::PageLocality)
        .at_least(0.9);
    let r = v.into_result();
    assert!(!r.passed);
    let bench = r
        .details
        .iter()
        .find(|d| d.kind == DetailKind::Benchmark)
        .expect("Benchmark kind must propagate");
    assert!(bench.message.contains("p99"));
    let loc = r
        .details
        .iter()
        .find(|d| d.kind == DetailKind::PageLocality)
        .expect("PageLocality kind must propagate");
    assert!(loc.message.contains("locality"));
}

#[test]
fn verdict_merge_folds_in_external_assert_result() {
    let mut v = Verdict::new();
    let a = 100u64;
    claim!(v, a).at_least(50); // pass

    let mut external = AssertResult::pass();
    external.passed = false;
    external
        .details
        .push(AssertDetail::new(DetailKind::Starved, "tid 7 starved"));
    v.merge(external);

    let b = 5u64;
    claim!(v, b).at_least(50); // fail

    let r = v.into_result();
    assert!(!r.passed);
    assert_eq!(r.details.len(), 2);
    assert!(r.details.iter().any(|d| d.kind == DetailKind::Starved));
    assert!(r.details.iter().any(|d| d.kind == DetailKind::Other));
}

#[test]
fn verdict_passed_and_detail_count_are_non_consuming_reads() {
    let mut v = Verdict::new();
    assert!(v.passed());
    assert_eq!(v.detail_count(), 0);

    let x = 100u64;
    claim!(v, x).at_least(50);
    assert!(v.passed());
    assert_eq!(v.detail_count(), 0);

    let y = 5u64;
    claim!(v, y).at_least(50);
    assert!(!v.passed());
    assert_eq!(v.detail_count(), 1);

    // Re-read to confirm peeks are non-consuming.
    assert!(!v.passed());
    assert_eq!(v.detail_count(), 1);

    let z = 200u64;
    claim!(v, z).at_most(100);
    assert_eq!(v.detail_count(), 2);
}

#[test]
fn claim_against_cgroup_stats_via_derived_accessors() {
    let cg = CgroupStats {
        num_workers: 2,
        num_cpus: 2,
        max_gap_ms: 50,
        total_iterations: 1000,
        ..Default::default()
    };

    let mut v = Verdict::new();
    cg.claim_max_gap_ms(&mut v).at_most(100);
    cg.claim_num_workers(&mut v).between(1, 10);
    cg.claim_total_iterations(&mut v).at_least(100);
    let r = v.into_result();
    assert!(r.passed, "details: {:?}", r.details);

    // Failing claim still names the field.
    let mut v = Verdict::new();
    cg.claim_max_gap_ms(&mut v).at_most(10); // 50 > 10 → fail
    let r = v.into_result();
    assert!(!r.passed);
    assert!(r.details[0].message.contains("max_gap_ms"));
    assert!(r.details[0].message.contains("at most 10"));
}

#[test]
fn claim_against_worker_report_via_derived_accessors() {
    let report = WorkerReport {
        tid: 4242,
        work_units: 1_000_000,
        cpu_time_ns: 2_500_000_000,
        wall_time_ns: 5_000_000_000,
        off_cpu_ns: 2_500_000_000,
        migration_count: 3,
        cpus_used: [0, 1].into_iter().collect(),
        migrations: vec![],
        max_gap_ms: 50,
        max_gap_cpu: 0,
        max_gap_at_ms: 1000,
        resume_latencies_ns: vec![100, 200, 300, 400, 500],
        wake_sample_total: 5,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 1000,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    };

    let mut v = Verdict::new();
    report.claim_tid(&mut v).eq(4242);
    report.claim_iterations(&mut v).at_least(100);
    report.claim_migration_count(&mut v).at_most(10);
    report.claim_completed(&mut v).eq(true);
    report.claim_cpus_used(&mut v).len_at_most(4);
    report.claim_resume_latencies_ns(&mut v).len_eq(5);
    let r = v.into_result();
    assert!(r.passed, "details: {:?}", r.details);
}

#[test]
fn claim_set_comparators_cover_membership_and_size() {
    let s: BTreeSet<usize> = [1, 2, 3].into_iter().collect();
    let mut v = Verdict::new();
    v.claim_set("s", &s).contains(&1);
    v.claim_set("s", &s).len_eq(3);
    v.claim_set("s", &s).len_at_most(5);
    v.claim_set("s", &s).len_at_least(1);
    v.claim_set("s", &s).nonempty();
    let allowed: BTreeSet<usize> = [1, 2, 3, 4].into_iter().collect();
    v.claim_set("s", &s).subset_of(&allowed);
    let forbidden: BTreeSet<usize> = [10, 11].into_iter().collect();
    v.claim_set("s", &s).disjoint_from(&forbidden);
    assert!(v.passed());

    let empty: BTreeSet<usize> = BTreeSet::new();
    let mut v = Verdict::new();
    v.claim_set("s", &empty).empty();
    assert!(v.passed());
}

#[test]
fn claim_seq_comparators_cover_membership_and_size() {
    let v_seq: Vec<u64> = vec![10, 20, 30];
    let mut verdict = Verdict::new();
    verdict.claim_seq("seq", &v_seq).contains(&20);
    verdict.claim_seq("seq", &v_seq).len_eq(3);
    verdict.claim_seq("seq", &v_seq).len_at_most(10);
    verdict.claim_seq("seq", &v_seq).len_at_least(1);
    verdict.claim_seq("seq", &v_seq).nonempty();
    assert!(verdict.passed());

    let empty: Vec<u64> = vec![];
    let mut verdict = Verdict::new();
    verdict.claim_seq("seq", &empty).empty();
    assert!(verdict.passed());
}

#[test]
fn verdict_skip_marks_skipped_without_failing() {
    let mut v = Verdict::new();
    v.skip("topology missing");
    let r = v.into_result();
    assert!(r.passed);
    assert!(r.skipped);
    assert!(
        r.details
            .iter()
            .any(|d| { d.kind == DetailKind::Skip && d.message.contains("topology missing") })
    );
}

/// `Verdict::skip` MUST NOT mask a prior failed claim. A claim that
/// failed produced real evidence — a later skip cannot retroactively
/// erase it. The verdict must surface as failed-and-skipped so gate
/// callers see both the failure and the reason the scenario stopped.
///
/// A prior implementation forced `passed = true` on skip, which let
/// a missing-precondition skip mask a real failure that had already
/// been recorded. That is silent data loss: the test would render as
/// a clean skip and the failure would never surface.
#[test]
fn verdict_skip_preserves_prior_failure() {
    let mut v = Verdict::new();
    claim!(v, 5u64).at_most(3);
    assert!(!v.passed(), "prior claim should fail (5 > 3)");
    v.skip("precondition missing");
    let r = v.into_result();
    assert!(
        !r.passed,
        "prior failure must NOT be masked by a later skip — got passed={}",
        r.passed,
    );
    assert!(
        r.skipped,
        "skip must still mark `skipped=true` so callers see the skip reason",
    );
    // The skip reason and the prior failure must both appear in details.
    assert!(
        r.details
            .iter()
            .any(|d| d.kind == DetailKind::Skip && d.message.contains("precondition missing")),
        "skip reason must be recorded: {:?}",
        r.details,
    );
    assert!(
        r.details.iter().any(|d| d.message.contains("at most 3")),
        "prior claim failure must be retained: {:?}",
        r.details,
    );
}

#[test]
fn verdict_skip_if_is_conditional() {
    let mut v = Verdict::new();
    v.skip_if(false, "nope");
    assert!(!v.into_result().skipped);

    let mut v = Verdict::new();
    v.skip_if(true, "yep");
    assert!(v.into_result().skipped);
}

#[test]
fn verdict_note_does_not_affect_verdict() {
    let mut v = Verdict::new();
    v.note("observed counter=12345");
    let r = v.into_result();
    assert!(r.passed);
    assert!(!r.skipped);
    assert_eq!(r.details.len(), 1);
    assert_eq!(r.details[0].kind, DetailKind::Note);
}

#[test]
fn claim_eq_against_nan_follows_ieee_754() {
    let mut v = Verdict::new();
    let nan = f64::NAN;
    claim!(v, nan).eq(f64::NAN);
    assert!(
        !v.passed(),
        "NaN == NaN is false per IEEE 754; eq(NaN) must FAIL",
    );

    let mut v = Verdict::new();
    let nan = f64::NAN;
    claim!(v, nan).ne(f64::NAN);
    assert!(
        v.passed(),
        "NaN != NaN is true per IEEE 754; ne(NaN) must PASS",
    );

    // The recommended NaN test routes through bool-of-`is_nan`:
    let value = f64::NAN;
    let mut v = Verdict::new();
    let is_nan = value.is_nan();
    claim!(v, is_nan).eq(true);
    assert!(v.passed());
}

#[test]
fn claim_because_reason_appears_in_failure_message() {
    let mut v = Verdict::new();
    let counter = 5u64;
    claim!(v, counter)
        .because("scheduler should have produced more events")
        .at_least(50);
    let r = v.into_result();
    assert!(!r.passed);
    let msg = &r.details[0].message;
    assert!(
        msg.contains("scheduler should have produced more events"),
        "msg: {msg}"
    );
    assert!(msg.contains("counter"), "msg: {msg}");
    assert!(msg.contains("at least 50"), "msg: {msg}");
}

#[test]
fn verdict_clone_carries_state() {
    let mut original = Verdict::new();
    let counter = 5u64;
    claim!(original, counter).at_least(50); // fail → push detail
    let copy = original.clone();
    assert_eq!(original.passed(), copy.passed());
    assert_eq!(original.detail_count(), copy.detail_count());

    // Mutating one must not affect the other.
    let mut copy = copy;
    let more = 1u64;
    claim!(copy, more).eq(1); // pass — no detail
    assert_eq!(original.detail_count(), 1);
    assert_eq!(copy.detail_count(), 1);

    let yet = 0u64;
    claim!(copy, yet).eq(1); // fail
    assert_eq!(original.detail_count(), 1);
    assert_eq!(copy.detail_count(), 2);
}

#[test]
fn verdict_merge_skipped_does_not_fail_accumulator() {
    let mut v = Verdict::new();
    let counter = 100u64;
    claim!(v, counter).at_least(50); // pass
    v.merge(AssertResult::skip("optional probe"));
    assert!(
        v.passed(),
        "merging a skip must not flip the accumulator to failing",
    );
    let r = v.into_result();
    assert!(r.passed);
    assert!(
        r.details
            .iter()
            .any(|d| d.message.contains("optional probe")),
        "skip rationale must reach merged details: {:?}",
        r.details
    );
}

/// `AssertDetail::display_with_kind` renders `[<variant>] <message>`
/// without altering the bare `Display` path. Pins both surfaces so a
/// regression that conflates the two (e.g. injecting the kind prefix
/// into the default formatter and breaking every consumer that
/// expected `format!("{}", d)` to produce just the message) trips
/// here.
#[test]
fn assert_detail_display_with_kind_prefixes_variant_token() {
    let d = AssertDetail::new(DetailKind::Stuck, "tid 7 stuck 1500ms on cpu3");
    assert_eq!(
        d.to_string(),
        "tid 7 stuck 1500ms on cpu3",
        "bare Display must remain message-only",
    );
    assert_eq!(
        d.display_with_kind().to_string(),
        "[Stuck] tid 7 stuck 1500ms on cpu3",
        "display_with_kind must prepend [<variant>]",
    );
}

/// `display_with_kind` rendering uses the Debug form of the
/// variant (e.g. `SchedulerDied`), not the snake_case rename
/// from any future `serde::Serialize` impl. Pinning the spelling
/// for a multi-word variant catches a swap from `{:?}` to
/// `{:#?}` (which would line-break) or to a serde-driven
/// renderer (which could rename the token).
#[test]
fn assert_detail_display_with_kind_uses_debug_token_for_multiword_variant() {
    let d = AssertDetail::new(DetailKind::SchedulerDied, "scheduler process died");
    assert_eq!(
        d.display_with_kind().to_string(),
        "[SchedulerDied] scheduler process died",
    );
}
