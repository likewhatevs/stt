use anyhow::Result;
use stt::assert::AssertResult;
use stt::scenario::Ctx;
use stt::stt_test;

/// Minimal stt_test that verifies the macro compiles and the generated
/// linkme registration + test wrapper resolve correctly from an
/// integration test.
///
/// The generated `#[test]` wrapper calls `run_stt_test`, which requires
/// KVM and a kernel image — it errors if either is unavailable.
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

/// Scheduler with the flags referenced by flags_attrs_compile.
#[derive(stt::Scheduler)]
#[scheduler(name = "flag_attrs_test", topology(1, 2, 1))]
#[allow(dead_code)]
enum FlagAttrsTestFlag {
    Borrow,
    Rebal,
    Steal,
}

/// Test with required_flags and excluded_flags attributes.
#[stt_test(
    scheduler = FLAG_ATTRS_TEST,
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

/// Scheduler with a distinctive topology for inheritance tests.
const TOPO_SCHED: stt::test_support::Scheduler =
    stt::test_support::Scheduler::new("topo_test").topology(3, 6, 1);

/// Full topology inheritance: all three dimensions from TOPO_SCHED.
#[stt_test(scheduler = TOPO_SCHED)]
fn topo_inherit_full(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Partial topology inheritance: threads overridden, sockets and cores
/// inherited from TOPO_SCHED.
#[stt_test(scheduler = TOPO_SCHED, threads = 2)]
fn topo_inherit_partial(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify full topology inheritance from scheduler.
#[test]
fn entry_topo_inherit_full() {
    let entry = stt::test_support::find_test("topo_inherit_full").unwrap();
    assert_eq!(entry.topology.sockets, 3);
    assert_eq!(entry.topology.cores_per_socket, 6);
    assert_eq!(entry.topology.threads_per_core, 1);
}

/// Verify partial topology inheritance: threads overridden, rest inherited.
#[test]
fn entry_topo_inherit_partial() {
    let entry = stt::test_support::find_test("topo_inherit_partial").unwrap();
    assert_eq!(entry.topology.sockets, 3);
    assert_eq!(entry.topology.cores_per_socket, 6);
    assert_eq!(entry.topology.threads_per_core, 2);
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

// ---------------------------------------------------------------------------
// Scheduler derive macro tests
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(
    name = "test_derive",
    binary = "test-binary",
    topology(2, 4, 1),
    cgroup_parent = "/test",
    sched_args = ["--arg1", "--arg2"]
)]
#[allow(dead_code)]
enum TestDeriveFlag {
    #[flag(args = ["--enable-alpha"])]
    Alpha,
    #[flag(args = ["--enable-beta"], requires = [Alpha])]
    Beta,
    #[flag(args = ["--enable-gamma-delta"])]
    GammaDelta,
}

/// Verify the derive generates a const Scheduler with the correct name.
#[test]
fn derive_scheduler_const_name() {
    let _ = &TEST_DERIVE;
    assert_eq!(TEST_DERIVE.name, "test_derive");
}

/// Verify scheduler binary spec.
#[test]
fn derive_scheduler_binary() {
    assert!(matches!(
        TEST_DERIVE.binary,
        stt::test_support::SchedulerSpec::Name("test-binary")
    ));
}

/// Verify scheduler topology.
#[test]
fn derive_scheduler_topology() {
    assert_eq!(TEST_DERIVE.topology.sockets, 2);
    assert_eq!(TEST_DERIVE.topology.cores_per_socket, 4);
    assert_eq!(TEST_DERIVE.topology.threads_per_core, 1);
}

/// Verify scheduler cgroup_parent.
#[test]
fn derive_scheduler_cgroup_parent() {
    assert_eq!(TEST_DERIVE.cgroup_parent, Some("/test"));
}

/// Verify scheduler sched_args.
#[test]
fn derive_scheduler_sched_args() {
    assert_eq!(TEST_DERIVE.sched_args, &["--arg1", "--arg2"]);
}

/// Verify the derive generates the correct number of flags.
#[test]
fn derive_scheduler_flag_count() {
    assert_eq!(TEST_DERIVE.flags.len(), 3);
}

/// Verify flag names are kebab-cased from variant names.
#[test]
fn derive_flag_names() {
    assert_eq!(TEST_DERIVE.flags[0].name, "alpha");
    assert_eq!(TEST_DERIVE.flags[1].name, "beta");
    assert_eq!(TEST_DERIVE.flags[2].name, "gamma-delta");
}

/// Verify flag args.
#[test]
fn derive_flag_args() {
    assert_eq!(TEST_DERIVE.flags[0].args, &["--enable-alpha"]);
    assert_eq!(TEST_DERIVE.flags[1].args, &["--enable-beta"]);
    assert_eq!(TEST_DERIVE.flags[2].args, &["--enable-gamma-delta"]);
}

/// Verify flag requires dependencies.
#[test]
fn derive_flag_requires() {
    assert!(TEST_DERIVE.flags[0].requires.is_empty());
    assert_eq!(TEST_DERIVE.flags[1].requires.len(), 1);
    assert_eq!(TEST_DERIVE.flags[1].requires[0].name, "alpha");
    assert!(TEST_DERIVE.flags[2].requires.is_empty());
}

/// Verify associated name constants.
#[test]
fn derive_name_constants() {
    assert_eq!(TestDeriveFlag::ALPHA, "alpha");
    assert_eq!(TestDeriveFlag::BETA, "beta");
    assert_eq!(TestDeriveFlag::GAMMA_DELTA, "gamma-delta");
}

/// Verify profile generation respects requires dependencies.
#[test]
fn derive_profiles_respect_requires() {
    let profiles = TEST_DERIVE.generate_profiles(&[TestDeriveFlag::BETA], &[]);
    for p in &profiles {
        assert!(
            p.flags.contains(&TestDeriveFlag::ALPHA),
            "beta requires alpha: {:?}",
            p.flags
        );
    }
}

/// Verify typed flag refs work in #[stt_test] required_flags.
#[stt_test(
    scheduler = TEST_DERIVE,
    required_flags = [TestDeriveFlag::ALPHA, TestDeriveFlag::BETA],
    excluded_flags = [TestDeriveFlag::GAMMA_DELTA]
)]
fn typed_flags_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify typed flag refs propagate correctly to the entry.
#[test]
fn entry_typed_flags_match() {
    let entry = stt::test_support::find_test("typed_flags_compile").unwrap();
    assert_eq!(entry.required_flags, &["alpha", "beta"]);
    assert_eq!(entry.excluded_flags, &["gamma-delta"]);
}

/// Verify mixed string/path flag refs work.
#[stt_test(
    scheduler = TEST_DERIVE,
    required_flags = ["alpha", TestDeriveFlag::BETA]
)]
fn mixed_flags_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify mixed flag refs propagate correctly.
#[test]
fn entry_mixed_flags_match() {
    let entry = stt::test_support::find_test("mixed_flags_compile").unwrap();
    assert_eq!(entry.required_flags, &["alpha", "beta"]);
}

/// Verify topology inheritance from derived scheduler.
#[stt_test(scheduler = TEST_DERIVE)]
fn derive_topo_inherit(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Verify topology inheritance from derived scheduler.
#[test]
fn entry_derive_topo_inherit() {
    let entry = stt::test_support::find_test("derive_topo_inherit").unwrap();
    assert_eq!(entry.topology.sockets, 2);
    assert_eq!(entry.topology.cores_per_socket, 4);
    assert_eq!(entry.topology.threads_per_core, 1);
}

// ---------------------------------------------------------------------------
// Empty enum edge case
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(name = "empty_sched", binary = "empty-binary", topology(1, 2, 1))]
#[allow(dead_code)]
enum EmptySchedFlag {}

/// Verify the const name is derived correctly for an empty enum.
#[test]
fn derive_empty_enum_const_name() {
    assert_eq!(EMPTY_SCHED.name, "empty_sched");
}

/// Verify an empty enum produces an empty flags slice.
#[test]
fn derive_empty_enum_no_flags() {
    assert!(EMPTY_SCHED.flags.is_empty());
}

/// Verify binary is set even with no flags.
#[test]
fn derive_empty_enum_binary() {
    assert!(matches!(
        EMPTY_SCHED.binary,
        stt::test_support::SchedulerSpec::Name("empty-binary")
    ));
}

/// Verify profile generation works with zero flags: exactly one profile
/// (the empty "default" profile).
#[test]
fn derive_empty_enum_profiles() {
    let profiles = EMPTY_SCHED.generate_profiles(&[], &[]);
    assert_eq!(profiles.len(), 1);
    assert!(profiles[0].flags.is_empty());
    assert_eq!(profiles[0].name(), "default");
}

// ---------------------------------------------------------------------------
// "Flags" (plural) suffix stripping
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(name = "test_flags", topology(1, 2, 1))]
#[allow(dead_code)]
enum TestFlags {
    #[flag(args = ["--x"])]
    Xray,
}

/// Verify "Flags" suffix is stripped: TestFlags -> TEST.
#[test]
fn derive_flags_suffix_stripping() {
    assert_eq!(TEST.name, "test_flags");
    assert_eq!(TEST.flags.len(), 1);
    assert_eq!(TEST.flags[0].name, "xray");
    assert_eq!(TestFlags::XRAY, "xray");
}

// ---------------------------------------------------------------------------
// No-suffix enum (unwrap_or fallback)
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(name = "plain", topology(1, 2, 1))]
#[allow(dead_code)]
enum PlainSched {
    #[flag(args = ["--y"])]
    Yankee,
}

/// Verify enum without "Flag"/"Flags" suffix uses full name: PlainSched -> PLAIN_SCHED.
#[test]
fn derive_no_suffix_const_name() {
    assert_eq!(PLAIN_SCHED.name, "plain");
    assert_eq!(PLAIN_SCHED.flags[0].name, "yankee");
    assert_eq!(PlainSched::YANKEE, "yankee");
}

// ---------------------------------------------------------------------------
// Variant without #[flag] attribute
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(name = "bare_variant", topology(1, 2, 1))]
#[allow(dead_code)]
enum BareVariantFlag {
    NakedVariant,
    #[flag(args = ["--with-args"])]
    WithArgs,
}

/// Verify a variant without #[flag(...)] produces a FlagDecl with empty
/// args and empty requires.
#[test]
fn derive_bare_variant_empty_args() {
    let naked = BARE_VARIANT.flags[0];
    assert_eq!(naked.name, "naked-variant");
    assert!(naked.args.is_empty());
    assert!(naked.requires.is_empty());
}

/// Verify the other variant still has its args.
#[test]
fn derive_bare_variant_other_has_args() {
    let with_args = BARE_VARIANT.flags[1];
    assert_eq!(with_args.name, "with-args");
    assert_eq!(with_args.args, &["--with-args"]);
}

// ---------------------------------------------------------------------------
// All-caps acronym variants
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(name = "acronym_test", topology(1, 2, 1))]
#[allow(dead_code, clippy::upper_case_acronyms)]
enum AcronymFlag {
    #[flag(args = ["--llc"])]
    LLC,
    #[flag(args = ["--io-heavy"])]
    IOHeavy,
}

/// Verify all-caps "LLC" produces kebab name "llc".
/// Note: AcronymFlag::LLC resolves as the enum variant (not the &str
/// constant) because the variant and constant share the same identifier.
/// Verify via the flags array instead.
#[test]
fn derive_acronym_llc() {
    assert_eq!(ACRONYM.flags[0].name, "llc");
    assert_eq!(ACRONYM.flags[0].args, &["--llc"]);
}

/// Verify "IOHeavy" produces kebab name "io-heavy" and constant IO_HEAVY.
#[test]
fn derive_acronym_io_heavy() {
    assert_eq!(ACRONYM.flags[1].name, "io-heavy");
    assert_eq!(AcronymFlag::IO_HEAVY, "io-heavy");
}

// ---------------------------------------------------------------------------
// Minimal derive (name only, all other attributes use defaults)
// ---------------------------------------------------------------------------

#[derive(stt::Scheduler)]
#[scheduler(name = "minimal")]
#[allow(dead_code)]
enum MinimalFlag {}

/// Verify a minimal derive with only name produces correct defaults:
/// no binary, default topology, no flags, no sched_args, no cgroup_parent.
#[test]
fn derive_minimal_defaults() {
    assert_eq!(MINIMAL.name, "minimal");
    assert!(!MINIMAL.binary.has_active_scheduling());
    assert!(matches!(
        MINIMAL.binary,
        stt::test_support::SchedulerSpec::None
    ));
    assert_eq!(MINIMAL.topology.sockets, 1);
    assert_eq!(MINIMAL.topology.cores_per_socket, 2);
    assert_eq!(MINIMAL.topology.threads_per_core, 1);
    assert!(MINIMAL.flags.is_empty());
    assert!(MINIMAL.sched_args.is_empty());
    assert!(MINIMAL.cgroup_parent.is_none());
}
