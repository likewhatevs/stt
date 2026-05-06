//! `AssertResult::note_value` / `NoteValue` plus the
//! `any_of` / `all_of` short-circuit constructors. The note_value
//! tests pin the From-impl routing per scalar type, the
//! independent-buffer invariant against `details`, the merge
//! union with last-write-wins on key collision, and the wire
//! format's `skip_serializing_if = "is_empty"` softness.

use super::*;

/// Each `From` impl on [`NoteValue`] routes to the matching enum
/// variant. Pin every variant so a regression that swapped two
/// `From` arms (e.g. `i64` mistakenly producing `NoteValue::Uint`)
/// trips here, not at the consumer's run-time mismatch.
#[test]
fn note_value_from_impls_route_to_correct_variant() {
    assert_eq!(NoteValue::from(42i64), NoteValue::Int(42));
    assert_eq!(NoteValue::from(42u64), NoteValue::Uint(42));
    assert_eq!(NoteValue::from(0.5_f64), NoteValue::Float(0.5));
    assert_eq!(NoteValue::from(true), NoteValue::Bool(true));
    assert_eq!(
        NoteValue::from("hello".to_string()),
        NoteValue::Text("hello".to_string()),
    );
    assert_eq!(
        NoteValue::from("borrowed"),
        NoteValue::Text("borrowed".to_string()),
    );
}

/// `note_value` writes into [`AssertResult::measurements`] without
/// altering the verdict. Distinct from [`Self::note`] (which
/// writes a `String` to `details`) — a producer commonly calls
/// BOTH and they occupy independent buffers.
#[test]
fn note_value_records_without_altering_verdict() {
    let mut r = AssertResult::pass();
    let was_passed = r.passed;
    let was_skipped = r.skipped;
    let was_details = r.details.len();
    r.note_value("max_wchar", 12345i64);
    r.note_value("psi_available", true);
    assert_eq!(r.passed, was_passed);
    assert_eq!(r.skipped, was_skipped);
    assert_eq!(r.details.len(), was_details);
    assert_eq!(r.measurements.len(), 2);
    assert_eq!(r.measurements["max_wchar"], NoteValue::Int(12345));
    assert_eq!(r.measurements["psi_available"], NoteValue::Bool(true));
}

/// Duplicate-key write overwrites: producers that re-record under
/// the same key (a producer bug, but well-defined) get the latest
/// value. Pin this so a future "first write wins" refactor surfaces
/// here.
#[test]
fn note_value_overwrites_on_duplicate_key() {
    let mut r = AssertResult::pass();
    r.note_value("counter", 1i64);
    r.note_value("counter", 2i64);
    assert_eq!(r.measurements["counter"], NoteValue::Int(2));
    assert_eq!(r.measurements.len(), 1);
}

/// `merge` folds `other.measurements` into `self.measurements`
/// with last-write-wins on key collision (matching `note_value`).
/// Pins the shape so a regression that union-with-keep-first
/// (e.g. `entry.or_insert(v)`) trips here.
#[test]
fn merge_unions_measurements_last_write_wins() {
    let mut a = AssertResult::pass();
    a.note_value("a_only", 1i64);
    a.note_value("shared", 100i64);

    let mut b = AssertResult::pass();
    b.note_value("b_only", 2i64);
    b.note_value("shared", 200i64);

    a.merge(b);
    assert_eq!(a.measurements.len(), 3);
    assert_eq!(a.measurements["a_only"], NoteValue::Int(1));
    assert_eq!(a.measurements["b_only"], NoteValue::Int(2));
    assert_eq!(
        a.measurements["shared"],
        NoteValue::Int(200),
        "merge must adopt other's value on key collision (last write wins)",
    );
}

/// `measurements` survives serde round-trip with the `untagged`
/// representation flowing into the right variant on deserialize.
/// Pins the wire-format invariant so a regression that switched
/// to a tagged enum representation (and broke existing parsers)
/// trips here.
#[test]
fn note_value_survives_serde_roundtrip() {
    let mut r = AssertResult::pass();
    r.note_value("answer", 42i64);
    r.note_value("ratio", 0.5_f64);
    r.note_value("name", "fio");
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        json.contains("\"measurements\""),
        "measurements key must appear in JSON when populated: {json}",
    );
    let r2: AssertResult = serde_json::from_str(&json).unwrap();
    assert_eq!(r2.measurements["answer"], NoteValue::Int(42));
    assert_eq!(r2.measurements["ratio"], NoteValue::Float(0.5));
    assert_eq!(r2.measurements["name"], NoteValue::Text("fio".to_string()));
}

/// Empty `measurements` is present in the wire format as `{}`.
/// `skip_serializing_if` was removed because AssertResult is
/// serialized with bincode (positional) — skipping a field on
/// serialize misaligns the deserializer. `#[serde(default)]`
/// handles old sidecars that lack the key.
#[test]
fn empty_measurements_present_in_wire_format() {
    let r = AssertResult::pass();
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        json.contains("\"measurements\":{}"),
        "empty measurements must be present in JSON for bincode compat: {json}",
    );
}

// -- AssertResult::any_of / AssertResult::all_of --------------------

/// `any_of` with at least one passing branch returns a passing
/// result and annotates which branch was chosen. Pin: failed-branch
/// details are dropped (they would only confuse the operator with
/// messages from not-taken paths).
#[test]
fn any_of_chooses_passing_branch() {
    let r = AssertResult::any_of([
        {
            let mut a = AssertResult::pass();
            a.passed = false;
            a.details.push(AssertDetail::new(DetailKind::Other, "boom"));
            a
        },
        AssertResult::pass(),
    ]);
    assert!(r.passed);
    // The "boom" detail from the failed branch must NOT appear —
    // the chosen branch's details prevail.
    assert!(
        !r.details.iter().any(|d| d.message.contains("boom")),
        "failed-branch details must be dropped: {:?}",
        r.details,
    );
    // The chosen-branch annotation MUST appear with branch index 1.
    assert!(
        r.details
            .iter()
            .any(|d| d.kind == DetailKind::Note
                && d.message.contains("any_of: branch 1 satisfied")),
        "chosen-branch annotation missing: {:?}",
        r.details,
    );
}

/// `any_of` with all branches failing returns a failing result and
/// concatenates every branch's details under a `any_of[<idx>]:`
/// prefix so the operator can identify which branch produced
/// which failure.
#[test]
fn any_of_concatenates_branch_failures_with_index_prefixes() {
    let r = AssertResult::any_of([
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "first boom")),
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "second boom")),
    ]);
    assert!(!r.passed);
    assert!(
        r.details
            .iter()
            .any(|d| d.message == "any_of[0]: first boom"),
        "branch 0 detail must carry index prefix: {:?}",
        r.details,
    );
    assert!(
        r.details
            .iter()
            .any(|d| d.message == "any_of[1]: second boom"),
        "branch 1 detail must carry index prefix: {:?}",
        r.details,
    );
    // A summary line names how many branches failed.
    assert!(
        r.details
            .iter()
            .any(|d| d.message.contains("all 2 branches failed")),
        "summary line missing: {:?}",
        r.details,
    );
}

/// `any_of` with empty input fails — an empty disjunction is
/// logically false. Pinned to surface a producer bug as a
/// nameable failure rather than a vacuous pass.
#[test]
fn any_of_empty_input_fails() {
    let r = AssertResult::any_of(std::iter::empty());
    assert!(!r.passed);
    assert!(
        r.details
            .iter()
            .any(|d| d.message.contains("empty branch list")),
        "empty disjunction must surface as named failure: {:?}",
        r.details,
    );
}

/// `all_of` is conjunction: passes iff every branch passes.
/// Empty input yields the passing identity (matches
/// `Iterator::all` semantics).
#[test]
fn all_of_passes_when_every_branch_passes() {
    let r = AssertResult::all_of([AssertResult::pass(), AssertResult::pass()]);
    assert!(r.passed);

    // One failing branch flips the verdict.
    let r = AssertResult::all_of([
        AssertResult::pass(),
        AssertResult::fail(AssertDetail::new(DetailKind::Other, "boom")),
    ]);
    assert!(!r.passed);

    // Empty input is the passing identity.
    let r = AssertResult::all_of(std::iter::empty());
    assert!(r.passed);
    assert!(r.details.is_empty());
}

/// `Verdict::note_value` mirrors [`AssertResult::note_value`] —
/// records under `measurements` without altering the verdict.
#[test]
fn verdict_note_value_records_into_underlying_result() {
    let mut v = Verdict::new();
    v.note_value("max_wchar", 12345i64);
    v.note_value("psi_available", false);
    let r = v.into_result();
    assert!(r.passed);
    assert_eq!(r.measurements.len(), 2);
    assert_eq!(r.measurements["max_wchar"], NoteValue::Int(12345));
    assert_eq!(r.measurements["psi_available"], NoteValue::Bool(false));
}
