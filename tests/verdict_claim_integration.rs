//! Integration test for the [`Verdict`] + [`claim!`] surface using
//! real [`WorkerReport`] data.
//!
//! `Verdict` is the primary user-facing pointwise-assertion surface,
//! built via [`Verdict::new`] / [`Assert::verdict`] and finished via
//! [`Verdict::into_result`]. The `#[derive(Claim)]` macro on
//! [`WorkerReport`] generates one `claim_<field>` accessor per public
//! field; the [`claim!`](ktstr::claim) macro covers local bindings and
//! arbitrary expressions. Both routes label the recorded claim via
//! `stringify!` over their input — the label cannot drift independent
//! of the source token a regression in the codegen would surface here.
//!
//! This test runs a real Thread-mode workload host-side (no VM
//! required), then exercises both label sources against the collected
//! `WorkerReport`s. It pins three properties:
//!
//! 1. The typed `claim_<field>` accessors exist for every WorkerReport
//!    field type the user is likely to assert against (scalar, set,
//!    sequence) and route through the Verdict accumulator without
//!    type-check ergonomics issues.
//! 2. The `claim!` macro accepts both local-binding labels (`claim!(v,
//!    iter_total)`) and expression labels
//!    (`claim!(v, total_iters as f64 / wall_secs)`), and the labels
//!    rendered into [`AssertDetail::message`] match `stringify!` of
//!    the input tokens.
//! 3. `Verdict::into_result` produces an `AssertResult` whose
//!    `passed`/`details` correctly reflect the claim outcomes — a
//!    failed verdict surfaces a detail naming the field via the
//!    derive-generated `stringify!(field)` label, not a hand-typed
//!    string.

use ktstr::assert::{Assert, AssertResult, DetailKind, Verdict};
use ktstr::claim;
use ktstr::workload::{
    CloneMode, ResolvedAffinity, SchedPolicy, WorkType, WorkerReport, WorkerReportClaim,
    WorkloadConfig, WorkloadHandle,
};

/// Spawn 2 Thread-mode CpuSpin workers, let them run for 200ms, then
/// collect their `WorkerReport`s. Thread mode runs in-process via
/// `std::thread::spawn`, so this is a pure host-side test — no VM,
/// no `/dev/kvm`, no kernel build. Returns at least one report
/// (panics if the workload returns none, since downstream claims
/// would be vacuous).
fn collect_real_reports() -> Vec<WorkerReport> {
    let config = WorkloadConfig {
        num_workers: 2,
        clone_mode: CloneMode::Thread,
        work_type: WorkType::CpuSpin,
        affinity: ResolvedAffinity::None,
        sched_policy: SchedPolicy::Normal,
        ..Default::default()
    };
    let mut handle = WorkloadHandle::spawn(&config).expect("Thread-mode CpuSpin must spawn");
    handle.start();
    // 200ms is well above the wake-cadence floor for a CpuSpin worker
    // (1024-iter checkpoints clear in microseconds), so every report
    // records non-zero wall_time_ns and a non-zero work_units count
    // — the at_least(1) bounds below depend on real progress.
    std::thread::sleep(std::time::Duration::from_millis(200));
    let reports = handle.stop_and_collect();
    assert!(
        !reports.is_empty(),
        "Thread-mode CpuSpin must produce at least one WorkerReport",
    );
    reports
}

/// Real WorkerReport data + passing claims via both the typed
/// accessor and the macro routes. Verifies `Verdict::into_result`
/// returns `passed = true` with no details when every comparator
/// passes.
#[test]
fn verdict_passing_claims_against_real_worker_report() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();

    // Typed scalar accessor. wall_time_ns is u64; `at_least(1)` is the
    // weakest non-trivial floor (any worker that ran at all has a
    // non-zero wall clock).
    report.claim_wall_time_ns(&mut v).at_least(1);
    // Typed scalar accessor with `eq` -- `completed` is bool (PartialEq
    // + Display). Real reports from a graceful stop_and_collect have
    // `completed = true`; sentinel reports have `false`. We assert the
    // graceful-stop path executed.
    report.claim_completed(&mut v).eq(true);
    // Typed set accessor. `cpus_used` is BTreeSet<usize>; SetClaim's
    // `nonempty` covers the "worker actually ran on at least one CPU"
    // baseline.
    report.claim_cpus_used(&mut v).nonempty();

    // Macro on a local binding. The label should be `"work_units"`
    // (the binding name, not the field name on the report).
    let work_units = report.work_units;
    claim!(v, work_units).at_least(0);

    // Macro on an expression. The label is the full token tree
    // serialized by stringify!.
    claim!(v, report.iterations + report.work_units).at_least(0);

    let r = v.into_result();
    assert!(
        r.passed,
        "every claim should pass against a real Thread-mode WorkerReport: {:?}",
        r.details,
    );
    assert!(
        r.details.is_empty(),
        "passing claims must not push details: {:?}",
        r.details,
    );
}

/// Failing claims must label each detail with the `stringify!` token
/// from the call site -- the field name for typed accessors, the
/// source token for the macro. A regression that hard-codes a label
/// or drifts independent of the source surfaces here.
#[test]
fn verdict_failing_claims_label_with_stringify_tokens() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();

    // Force a failure on the typed accessor route: ceiling far below
    // any real wall_time_ns (a 150ms run is at least 100µs even on a
    // pathologically slow host). Detail message must contain
    // "wall_time_ns" -- the field name from `stringify!(wall_time_ns)`
    // in the derive-generated method body.
    report.claim_wall_time_ns(&mut v).at_most(0);

    // Force a failure on the macro local-binding route. Label must be
    // "iter_count" -- the binding name, not "iterations" (the field
    // name) -- because the macro stringifies the expression token,
    // not the path it traverses.
    let iter_count = report.iterations;
    claim!(v, iter_count).eq(u64::MAX);

    // Force a failure on the macro expression route. Label must be
    // the entire token tree "report.cpus_used.len()" verbatim, since
    // stringify! preserves all source tokens including dot
    // separators and parentheses.
    claim!(v, report.cpus_used.len()).at_least(usize::MAX);

    let r = v.into_result();
    assert!(
        !r.passed,
        "deliberately-failing claims must produce a failed verdict",
    );
    assert_eq!(
        r.details.len(),
        3,
        "one detail per failed claim; got: {:?}",
        r.details,
    );

    // Every detail records DetailKind::Other (the ClaimBuilder default
    // when no `.kind(...)` was set) and surfaces the comparator's
    // formatted message.
    for d in &r.details {
        assert_eq!(d.kind, DetailKind::Other);
    }

    // Detail 0: typed accessor on the `wall_time_ns` field. The
    // derive-generated method body calls
    // `verdict.claim(stringify!(wall_time_ns), ...)`, so the label
    // is the field name verbatim.
    assert!(
        r.details[0].message.contains("wall_time_ns"),
        "typed accessor failure must label with the field name from \
         stringify!(field) in the derive expansion: {}",
        r.details[0].message,
    );
    assert!(
        r.details[0].message.contains("at most 0"),
        "at_most failure must render the bound: {}",
        r.details[0].message,
    );

    // Detail 1: macro on local binding. Label is the binding ident
    // verbatim -- proves the macro reads the expression token, not
    // the variable's underlying source.
    assert!(
        r.details[1].message.contains("iter_count"),
        "claim! macro failure on a local binding must label with the \
         binding name from stringify!($value): {}",
        r.details[1].message,
    );
    // Negative: the field name `iterations` (which the binding was
    // initialized from) MUST NOT appear -- the label tracks the
    // expression token, not the value's provenance.
    assert!(
        !r.details[1].message.contains("iterations"),
        "claim! must NOT leak the underlying field name when the \
         caller bound it to a different ident: {}",
        r.details[1].message,
    );

    // Detail 2: macro on a multi-token expression. Label preserves
    // every token from the input, including method-call parens.
    assert!(
        r.details[2].message.contains("report.cpus_used.len()"),
        "claim! on an expression must stringify the full token tree: {}",
        r.details[2].message,
    );
}

/// `Verdict` mixes upstream `AssertResult` values via [`Verdict::merge`]
/// alongside pointwise claims. Real production tests fold an
/// `assert_not_starved` (or similar) result into a verdict that also
/// carries pointwise claims; this test pins that the merge path
/// preserves both passing pointwise records and a failing merged
/// upstream.
#[test]
fn verdict_merges_external_assert_result_into_pointwise_claims() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();
    // Passing pointwise claim against a real value.
    report.claim_completed(&mut v).eq(true);

    // Synthesize a failing upstream AssertResult (simulating an
    // `assert_*` returning a failure).
    let mut upstream = AssertResult::pass();
    upstream.passed = false;
    upstream.details.push(ktstr::assert::AssertDetail::new(
        DetailKind::Other,
        "synthetic upstream failure".to_string(),
    ));

    v.merge(upstream);

    let r = v.into_result();
    assert!(
        !r.passed,
        "merging a failing upstream must conjoin into the verdict",
    );
    let messages: Vec<&str> = r.details.iter().map(|d| d.message.as_str()).collect();
    assert!(
        messages.iter().any(|m| m.contains("synthetic upstream failure")),
        "merged upstream details must survive into the final result: {:?}",
        messages,
    );
}

/// Skip path. `Verdict::skip` records a skip reason and leaves
/// `passed` true (skips are not failures). Real tests use this when a
/// precondition (kernel feature, hardware) is missing; pin that the
/// skip detail kind and reason survive `into_result`.
#[test]
fn verdict_skip_records_skip_kind_with_reason() {
    let reports = collect_real_reports();
    let report = &reports[0];

    let mut v = Verdict::new();
    // Real claim against the report so the verdict has at least one
    // pointwise record alongside the skip.
    report.claim_completed(&mut v).eq(true);
    v.skip("integration test demonstrates skip path");

    let r = v.into_result();
    assert!(
        r.passed,
        "skip must NOT mark the verdict failed: {:?}",
        r.details,
    );
    assert!(r.skipped, "skip flag must be set on the result");
    let skip_detail = r
        .details
        .iter()
        .find(|d| d.kind == DetailKind::Skip)
        .expect("at least one Skip-kind detail must be present");
    assert!(
        skip_detail
            .message
            .contains("integration test demonstrates skip path"),
        "skip detail must carry the supplied reason verbatim: {}",
        skip_detail.message,
    );
}
