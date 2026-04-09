use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::stt_test;

/// Minimal stt_test that verifies the macro compiles and the generated
/// linkme registration + test wrapper resolve correctly from an
/// integration test.
///
/// The test body skips at runtime if no kernel is available, since
/// actually booting a VM requires a bzImage.
#[stt_test(sockets = 1, cores = 2, threads = 1, memory_mb = 2048)]
fn basic_topology_check(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    if total == 0 {
        return Ok(AssertResult {
            passed: false,
            details: vec!["no CPUs detected".into()],
            stats: Default::default(),
        });
    }
    Ok(AssertResult::pass())
}

/// Second stt_test with default attributes to verify defaults work.
#[stt_test]
fn default_attrs_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify resolve_func_ip returns a real nonzero address inside the VM.
/// On the host, kptr_restrict or kernel lockdown hides addresses.
#[cfg(feature = "integration")]
#[stt_test(sockets = 1, cores = 1, threads = 1)]
fn resolve_func_ip_known_symbol(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let ip = stt::resolve_func_ip("schedule");
    if let Some(addr) = ip
        && addr > 0
    {
        return Ok(AssertResult::pass());
    }
    Ok(AssertResult {
        passed: false,
        details: vec![format!("schedule address: {ip:?}")],
        stats: Default::default(),
    })
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
    assert_eq!(entry.topology.sockets, 1);
    assert_eq!(entry.topology.cores_per_socket, 2);
    assert_eq!(entry.topology.threads_per_core, 1);
    assert_eq!(entry.memory_mb, 2048);
}

/// Verify default attribute values.
#[test]
fn entry_default_fields() {
    let entry = stt::test_support::find_test("default_attrs_compile").unwrap();
    assert_eq!(entry.topology.sockets, 1);
    assert_eq!(entry.topology.cores_per_socket, 2);
    assert_eq!(entry.topology.threads_per_core, 1);
    assert_eq!(entry.memory_mb, 2048);
    assert!(entry.required_flags.is_empty());
    assert!(entry.excluded_flags.is_empty());
    assert_eq!(entry.constraints.min_sockets, 1);
    assert_eq!(entry.constraints.min_llcs, 1);
    assert!(!entry.constraints.requires_smt);
    assert_eq!(entry.constraints.min_cpus, 1);
}

/// Test with required_flags and excluded_flags attributes.
#[stt_test(
    sockets = 1,
    cores = 2,
    threads = 1,
    required_flags = ["borrow", "rebal"],
    excluded_flags = ["steal"]
)]
fn flags_attrs_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify required_flags and excluded_flags propagate to the entry.
#[test]
fn entry_flags_match_attrs() {
    let entry = stt::test_support::find_test("flags_attrs_compile").unwrap();
    assert_eq!(entry.required_flags, &["borrow", "rebal"]);
    assert_eq!(entry.excluded_flags, &["steal"]);
}

/// Test with topology constraint attributes.
#[stt_test(
    sockets = 2,
    cores = 4,
    threads = 2,
    min_sockets = 2,
    min_llcs = 4,
    requires_smt = true,
    min_cpus = 8
)]
fn topo_constraints_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify topology constraints propagate to the entry.
#[test]
fn entry_topo_constraints_match_attrs() {
    let entry = stt::test_support::find_test("topo_constraints_compile").unwrap();
    assert_eq!(entry.constraints.min_sockets, 2);
    assert_eq!(entry.constraints.min_llcs, 4);
    assert!(entry.constraints.requires_smt);
    assert_eq!(entry.constraints.min_cpus, 8);
}

/// Test with performance_mode — verifies macro sets the field.
#[stt_test(sockets = 1, cores = 2, threads = 1, performance_mode = true)]
fn performance_mode_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify performance_mode is set in generated entry.
#[test]
fn entry_performance_mode_set() {
    let entry = stt::test_support::find_test("performance_mode_compile").unwrap();
    assert!(
        entry.performance_mode,
        "performance_mode = true must be set in generated entry",
    );
}
