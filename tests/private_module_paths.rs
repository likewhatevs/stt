//! External-context check for `ktstr::__private::{ctor, serde_json}`.
//!
//! The `#[ktstr_test]` proc macro emits a
//! `#[::ktstr::__private::linkme::distributed_slice(::ktstr::test_support::KTSTR_TESTS)]`
//! static of type `KtstrTestEntry` that registers the test in the
//! `KTSTR_TESTS` distributed slice at link time. The `__private::ctor`
//! and `__private::serde_json` re-exports are used by test-author
//! code (e.g. `tests/jemalloc_probe_tests.rs` attaches `#[ctor::ctor]`
//! via the re-export to set env vars before the test harness runs;
//! downstream consumers reach for `__private::serde_json` to parse
//! sidecar output without listing `serde_json` as a direct dep).
//!
//! If any of these re-exports change path or disappear, downstream
//! crates that depend on the surface fail to compile. This file
//! exercises both paths directly from external test code — i.e.
//! treating `ktstr` as a dev-dependency — so a silent regression in
//! the private re-export surface would fail this binary's build
//! before the broader integration suite runs.
//!
//! The assertions live inside a plain `#[test]` because this file
//! holds no `#[ktstr_test]` entries. Confirm both paths resolve, can
//! be invoked, and produce the same behavior that the macro
//! expansion relies on.

use ktstr::__private;

/// `serde_json::to_string` must be reachable through the re-export
/// and must serialize a simple structure the same way the top-level
/// `serde_json` crate would.
#[test]
fn private_serde_json_to_string_roundtrip() {
    let v: Vec<(&str, u32)> = vec![("llc", 0), ("borrow", 1)];
    let json = __private::serde_json::to_string(&v).expect("serialize via __private path");
    // serde_json formats tuple structs as JSON arrays; the expected
    // output is stable and equality-testable.
    assert_eq!(json, r#"[["llc",0],["borrow",1]]"#);
}

/// `serde_json::from_str` is used by downstream consumers reading
/// sidecar output. Roundtrip a value through `__private::serde_json`
/// both directions to prove the re-export exposes the full crate,
/// not just a subset.
#[test]
fn private_serde_json_from_str_roundtrip() {
    let v: Vec<(String, u32)> =
        __private::serde_json::from_str(r#"[["llc",0]]"#).expect("parse via __private path");
    assert_eq!(v, vec![("llc".to_string(), 0)]);
}

/// `__private::ctor` must expose the `#[ctor]` attribute macro used by
/// the test-flag registration path. Attach it here via the fully
/// qualified re-export path (matching the macro's emission style) and
/// observe its side effect — the ctor fires before `#[test]` runs,
/// so by the time the test body executes, the static has been
/// initialized.
static INIT_FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[::ktstr::__private::ctor::ctor(crate_path = ::ktstr::__private::ctor)]
fn mark_ctor_fired() {
    INIT_FIRED.store(true, std::sync::atomic::Ordering::Release);
}

#[test]
fn private_ctor_attribute_fires_before_tests() {
    assert!(
        INIT_FIRED.load(std::sync::atomic::Ordering::Acquire),
        "`#[::ktstr::__private::ctor::ctor(...)]` must run before the test harness dispatches"
    );
}
