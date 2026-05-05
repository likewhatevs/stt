//! `format_sched_died_*` template pins. The three helpers are
//! the single source of truth for emitter-side message
//! formatting; every production site goes through them. These
//! tests pin the exact message templates so operators grepping
//! stderr can keep stable anchors, and the numeric formatting
//! (step N of M, `{:.1}s`). The structural detection path —
//! matching on `DetailKind::SchedulerDied` — is exercised in
//! `eval.rs` tests directly; the message format is kept stable
//! here purely as a human-readable contract.

use super::*;

#[test]
fn format_sched_died_after_step_has_expected_template() {
    // 4.23 → "4.2" under `{:.1}`; avoid half-boundary inputs like
    // x.x5 whose rounding depends on round-half-to-even.
    let msg = format_sched_died_after_step(3, 10, 4.23);
    assert_eq!(
        msg,
        "scheduler process died unexpectedly after completing step 3 of 10 (4.2s into test)",
    );
}

#[test]
fn format_sched_died_after_all_steps_has_expected_template() {
    let msg = format_sched_died_after_all_steps(7, 12.08);
    assert_eq!(
        msg,
        "scheduler process died unexpectedly (detected after all 7 steps completed, 12.1s elapsed)",
    );
}

#[test]
fn format_sched_died_during_workload_has_expected_template() {
    let msg = format_sched_died_during_workload(2.04);
    assert_eq!(
        msg,
        "scheduler process died unexpectedly during workload (2.0s into test)",
    );
}

/// Every `format_sched_died_*` helper output begins with
/// [`SCHED_DIED_PREFIX`]. Operators grepping stderr for the
/// prefix rely on this invariant; pin it against a regression
/// in any one helper's template that accidentally drops the
/// prefix string.
#[test]
fn format_sched_died_helpers_start_with_prefix() {
    for msg in [
        format_sched_died_after_step(1, 1, 0.0),
        format_sched_died_after_all_steps(1, 0.0),
        format_sched_died_during_workload(0.0),
    ] {
        assert!(
            msg.starts_with(SCHED_DIED_PREFIX),
            "every sched-died helper output must start with SCHED_DIED_PREFIX: {msg}",
        );
    }
}
