//! Tests for the `_PAYLOAD` const emitted by `#[derive(Scheduler)]`.
//!
//! Lives in its own crate for the same reason as
//! `derive_payload_tests.rs`: the ctor-based dispatcher in
//! `ktstr_test_macro.rs` hides plain `#[test]` functions from
//! nextest's `--list`.

use ktstr::test_support::{OutputFormat, PayloadKind};

/// A scheduler with a binary and non-trivial topology — exercises
/// the full builder chain before the `_PAYLOAD` const emission.
#[derive(ktstr::Scheduler)]
#[scheduler(name = "mitosis", binary = "scx_mitosis", topology(1, 2, 4, 1))]
#[allow(dead_code)]
enum MitosisFlag {
    #[flag(args = ["--enable-llc"])]
    Llc,
    #[flag(args = ["--enable-steal"])]
    Steal,
}

#[test]
fn derive_scheduler_emits_payload_wrapper_const() {
    // The scheduler const exists (regression guard for the original
    // emission) and the wrapper const sits next to it.
    assert_eq!(MITOSIS.name, "mitosis");
    assert_eq!(MITOSIS_PAYLOAD.name, "mitosis");
}

#[test]
fn derive_scheduler_payload_wrapper_kind_points_at_scheduler() {
    match MITOSIS_PAYLOAD.kind {
        PayloadKind::Scheduler(s) => {
            // The Payload's Scheduler variant carries a &'static ref
            // to the same Scheduler const — compare identity via the
            // stable `name` field rather than pointer equality (which
            // would require a Scheduler: PartialEq the struct doesn't
            // implement).
            assert_eq!(s.name, MITOSIS.name);
        }
        PayloadKind::Binary(_) => {
            panic!("derived scheduler payload must wrap the scheduler")
        }
    }
}

#[test]
fn derive_scheduler_payload_wrapper_defaults() {
    // The wrapper does not carry binary-specific metric surface —
    // schedulers are framework-launched, not body-invoked, so the
    // binary-oriented fields are empty.
    assert!(matches!(MITOSIS_PAYLOAD.output, OutputFormat::ExitCode));
    assert!(MITOSIS_PAYLOAD.default_args.is_empty());
    assert!(MITOSIS_PAYLOAD.default_checks.is_empty());
    assert!(MITOSIS_PAYLOAD.metrics.is_empty());
}

/// Confirm the wrapper const is const-usable as a static initializer,
/// matching the shape `#[ktstr_test(workloads = [..])]` consumes.
const _WRAPPER_REF: &ktstr::test_support::Payload = &MITOSIS_PAYLOAD;

#[test]
fn derive_scheduler_payload_wrapper_is_const_ref() {
    assert_eq!(_WRAPPER_REF.name, "mitosis");
    assert!(_WRAPPER_REF.is_scheduler());
}

/// Empty-flags scheduler: the suffix-strip ("Flag"/"Flags") + naming
/// still produce a usable `_PAYLOAD` const.
#[derive(ktstr::Scheduler)]
#[scheduler(name = "empty_sched")]
#[allow(dead_code)]
enum EmptySchedFlag {}

#[test]
fn derive_scheduler_empty_flags_emits_payload_wrapper() {
    assert_eq!(EMPTY_SCHED.name, "empty_sched");
    assert_eq!(EMPTY_SCHED_PAYLOAD.name, "empty_sched");
    assert!(EMPTY_SCHED_PAYLOAD.is_scheduler());
}

/// Enum without "Flag"/"Flags" suffix: the base const keeps its full
/// uppercase name and the wrapper adds `_PAYLOAD`.
#[derive(ktstr::Scheduler)]
#[scheduler(name = "plain")]
#[allow(dead_code)]
enum PlainSched {}

#[test]
fn derive_scheduler_no_suffix_emits_payload_wrapper() {
    assert_eq!(PLAIN_SCHED.name, "plain");
    assert_eq!(PLAIN_SCHED_PAYLOAD.name, "plain");
    assert!(PLAIN_SCHED_PAYLOAD.is_scheduler());
}
