use anyhow::Result;
use stt::scenario::Ctx;
use stt::stt_test;
use stt::verify::VerifyResult;

/// Minimal stt_test that verifies the macro compiles and the generated
/// linkme registration + test wrapper resolve correctly from an
/// integration test.
///
/// The test body skips at runtime if no kernel is available, since
/// actually booting a VM requires a bzImage.
#[stt_test(sockets = 1, cores = 2, threads = 1, memory_mb = 2048)]
fn basic_topology_check(ctx: &Ctx) -> Result<VerifyResult> {
    let total = ctx.topo.total_cpus();
    if total == 0 {
        return Ok(VerifyResult {
            passed: false,
            details: vec!["no CPUs detected".into()],
            stats: Default::default(),
        });
    }
    Ok(VerifyResult::pass())
}

/// Second stt_test with default attributes to verify defaults work.
#[stt_test]
fn default_attrs_compile(ctx: &Ctx) -> Result<VerifyResult> {
    let _ = ctx;
    Ok(VerifyResult::pass())
}

/// Verify that find_test can locate registered entries.
#[test]
fn find_registered_tests() {
    assert!(
        stt::test_support::find_test("basic_topology_check").is_some(),
        "basic_topology_check should be registered in STT_TESTS"
    );
    assert!(
        stt::test_support::find_test("default_attrs_compile").is_some(),
        "default_attrs_compile should be registered in STT_TESTS"
    );
}

/// Verify entry field values match the macro attributes.
#[test]
fn entry_fields_match_attrs() {
    let entry = stt::test_support::find_test("basic_topology_check").unwrap();
    assert_eq!(entry.sockets, 1);
    assert_eq!(entry.cores, 2);
    assert_eq!(entry.threads, 1);
    assert_eq!(entry.memory_mb, 2048);
}

/// Verify default attribute values.
#[test]
fn entry_default_fields() {
    let entry = stt::test_support::find_test("default_attrs_compile").unwrap();
    assert_eq!(entry.sockets, 1);
    assert_eq!(entry.cores, 2);
    assert_eq!(entry.threads, 1);
    assert_eq!(entry.memory_mb, 2048);
}
