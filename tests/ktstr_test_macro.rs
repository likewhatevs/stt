use anyhow::Result;
use ktstr::assert::AssertResult;
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;

/// Minimal ktstr_test that checks the macro compiles and the generated
/// linkme registration + test wrapper resolve correctly from an
/// integration test.
///
/// The generated `#[test]` wrapper calls `run_ktstr_test`, which requires
/// KVM and a kernel image — it errors if either is unavailable.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, memory_mb = 2048)]
fn basic_topology_check(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    if total == 0 {
        return Ok(AssertResult {
            passed: false,
            skipped: false,
            details: vec!["no CPUs detected".into()],
            stats: Default::default(),
            measurements: std::collections::BTreeMap::new(),
        });
    }
    Ok(AssertResult::pass())
}

/// Second ktstr_test with default attributes to check defaults work.
#[ktstr_test]
fn default_attrs_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// ktstr_test with host_only = true to check the macro accepts the
/// attribute and wires it into KtstrTestEntry.host_only.
#[ktstr_test(host_only = true)]
fn host_only_attr_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// `workloads = []` must compile cleanly. The macro accepts an
/// empty array as "no additional binary workloads composed under
/// the scheduler" — the same semantic as omitting the attribute.
/// The emitted `KtstrTestEntry.workloads` is `&[]`, which entry
/// validation accepts (the Scheduler-kind rejection loop iterates
/// zero elements and passes). Host-only so nextest discovery does
/// not drag the test through a VM boot.
#[ktstr_test(host_only = true, workloads = [])]
fn empty_workloads_compiles(_ctx: &Ctx) -> Result<AssertResult> {
    Ok(AssertResult::pass())
}

/// Check resolve_func_ip returns a real nonzero address inside the VM.
/// On the host, kptr_restrict or kernel lockdown hides addresses.
#[cfg(feature = "integration")]
#[ktstr_test(llcs = 1, cores = 1, threads = 1)]
fn resolve_func_ip_known_symbol(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let ip = ktstr::resolve_func_ip("schedule");
    if let Some(addr) = ip
        && addr > 0
    {
        return Ok(AssertResult::pass());
    }
    Ok(AssertResult {
        passed: false,
        skipped: false,
        details: vec![format!("schedule address: {ip:?}").into()],
        stats: Default::default(),
        measurements: Default::default(),
    })
}

/// Check that find_test can locate registered entries.
#[test]
fn find_registered_tests() {
    assert!(
        ktstr::test_support::find_test("basic_topology_check").is_some(),
        "basic_topology_check should be registered in KTSTR_TESTS"
    );
    assert!(
        ktstr::test_support::find_test("default_attrs_compile").is_some(),
        "default_attrs_compile should be registered in KTSTR_TESTS"
    );
}

/// Check entry field values match the macro attributes.
#[test]
fn entry_fields_match_attrs() {
    let entry = ktstr::test_support::find_test("basic_topology_check").unwrap();
    assert_eq!(entry.topology.llcs, 1);
    assert_eq!(entry.topology.cores_per_llc, 2);
    assert_eq!(entry.topology.threads_per_core, 1);
    assert_eq!(entry.memory_mb, 2048);
}

/// Check default attribute values.
#[test]
fn entry_default_fields() {
    let entry = ktstr::test_support::find_test("default_attrs_compile").unwrap();
    assert_eq!(entry.topology.llcs, 1);
    assert_eq!(entry.topology.cores_per_llc, 2);
    assert_eq!(entry.topology.threads_per_core, 1);
    assert_eq!(entry.memory_mb, 2048);
    assert_eq!(entry.constraints.min_numa_nodes, 1);
    assert_eq!(entry.constraints.min_llcs, 1);
    assert!(!entry.constraints.requires_smt);
    assert_eq!(entry.constraints.min_cpus, 1);
    assert_eq!(entry.constraints.max_llcs, Some(12));
    assert_eq!(entry.constraints.max_numa_nodes, Some(1));
    assert_eq!(entry.constraints.max_cpus, Some(192));
    assert!(!entry.host_only);
}

/// Check that host_only = true is threaded into KtstrTestEntry.
#[test]
fn entry_host_only_attr() {
    let entry = ktstr::test_support::find_test("host_only_attr_compile").unwrap();
    assert!(entry.host_only);
}

/// Stub `post_vm` callback used by the
/// `post_vm_attr_compile` macro test below. The macro must
/// accept a path-valued `post_vm = NAME` attribute and route
/// it onto `KtstrTestEntry.post_vm`. The callback never runs in
/// the host-only attribute test (host_only short-circuits the
/// VM-boot path), so the body is a trivial Ok — this stub exists
/// only to give the macro a function path to wire.
fn macro_post_vm_stub(_result: &ktstr::prelude::VmResult) -> Result<()> {
    Ok(())
}

/// `ktstr_test` with `post_vm = NAME` to check the macro
/// accepts the attribute and wires it into
/// `KtstrTestEntry.post_vm`. Host-only so nextest discovery does
/// not drag the test through a VM boot — the post_vm field
/// itself is the unit under test, not the VM-side dispatch.
#[ktstr_test(host_only = true, post_vm = macro_post_vm_stub)]
fn post_vm_attr_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check that `post_vm = NAME` is threaded into
/// `KtstrTestEntry.post_vm` as `Some(NAME)` and the default
/// (no `post_vm = ...` attribute) leaves the field at `None`.
#[test]
fn entry_post_vm_attr() {
    let with_post = ktstr::test_support::find_test("post_vm_attr_compile").unwrap();
    assert!(
        with_post.post_vm.is_some(),
        "post_vm = NAME must wire onto KtstrTestEntry.post_vm as Some(_)",
    );
    let without_post = ktstr::test_support::find_test("default_attrs_compile").unwrap();
    assert!(
        without_post.post_vm.is_none(),
        "post_vm omitted from #[ktstr_test] must leave KtstrTestEntry.post_vm = None",
    );
}

// Scheduler that exercises the sysctls + kargs attributes.
ktstr::declare_scheduler!(SYS_KARGS_TEST, {
    name = "sys_kargs_test",
    binary = "scx-ktstr",
    sysctls = [
        ktstr::test_support::Sysctl::new("kernel.sched_cfs_bandwidth_slice_us", "1000"),
        ktstr::test_support::Sysctl::new("kernel.sched_rr_timeslice_ms", "25"),
    ],
    kargs = ["nosmt", "iomem=relaxed"],
});

/// Check the macro threads sysctls + kargs into the Scheduler const.
#[test]
fn declare_scheduler_sysctls_kargs() {
    assert_eq!(SYS_KARGS_TEST.sysctls.len(), 2);
    assert_eq!(
        SYS_KARGS_TEST.sysctls[0].key,
        "kernel.sched_cfs_bandwidth_slice_us"
    );
    assert_eq!(SYS_KARGS_TEST.sysctls[0].value, "1000");
    assert_eq!(
        SYS_KARGS_TEST.sysctls[1].key,
        "kernel.sched_rr_timeslice_ms"
    );
    assert_eq!(SYS_KARGS_TEST.sysctls[1].value, "25");
    assert_eq!(SYS_KARGS_TEST.kargs, &["nosmt", "iomem=relaxed"]);
}

// (flag-attribute propagation tests deleted with the flag system —
// required_flags / excluded_flags fields no longer exist on
// KtstrTestEntry, and the gauntlet flag-profile expansion they fed
// is removed.)

/// Test with topology constraint attributes.
#[ktstr_test(
    llcs = 2,
    cores = 4,
    threads = 2,
    min_numa_nodes = 2,
    max_numa_nodes = 4,
    min_llcs = 4,
    requires_smt = true,
    min_cpus = 8
)]
fn topo_constraints_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check topology constraints propagate to the entry.
#[test]
fn entry_topo_constraints_match_attrs() {
    let entry = ktstr::test_support::find_test("topo_constraints_compile").unwrap();
    assert_eq!(entry.constraints.min_numa_nodes, 2);
    assert_eq!(entry.constraints.min_llcs, 4);
    assert!(entry.constraints.requires_smt);
    assert_eq!(entry.constraints.min_cpus, 8);
    assert_eq!(entry.constraints.max_numa_nodes, Some(4));
    assert_eq!(entry.constraints.max_llcs, Some(12));
    assert_eq!(entry.constraints.max_cpus, Some(192));
}

/// Test with max constraint attributes.
#[ktstr_test(max_llcs = 4, max_numa_nodes = 2, max_cpus = 32)]
fn max_constraints_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check max constraint attributes propagate to the entry.
#[test]
fn entry_max_constraints_match_attrs() {
    let entry = ktstr::test_support::find_test("max_constraints_compile").unwrap();
    assert_eq!(entry.constraints.max_llcs, Some(4));
    assert_eq!(entry.constraints.max_numa_nodes, Some(2));
    assert_eq!(entry.constraints.max_cpus, Some(32));
}

// ---------------------------------------------------------------------------
// Scheduler-level constraint inheritance
// ---------------------------------------------------------------------------

// Scheduler with constraint attributes for inheritance tests.
ktstr::declare_scheduler!(CONSTRAINED_SCHED, {
    name = "constrained_sched",
    binary = "scx-ktstr",
    topology = (1, 2, 4, 1),
    constraints = ktstr::test_support::TopologyConstraints {
        max_llcs: Some(8),
        max_cpus: Some(64),
        ..ktstr::test_support::TopologyConstraints::DEFAULT
    },
});

/// Inherits constraints from CONSTRAINED_SCHED without overriding.
#[ktstr_test(scheduler = CONSTRAINED_SCHED)]
fn inherit_sched_constraints(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check inherited constraint values match the scheduler definition.
#[test]
fn entry_inherit_sched_constraints() {
    let entry = ktstr::test_support::find_test("inherit_sched_constraints").unwrap();
    assert_eq!(entry.constraints.max_llcs, Some(8));
    assert_eq!(entry.constraints.max_cpus, Some(64));
    // Not set on scheduler — inherits from TopologyConstraints::DEFAULT.
    assert_eq!(entry.constraints.max_numa_nodes, Some(1));
    assert_eq!(entry.constraints.min_llcs, 1);
    assert_eq!(entry.constraints.min_cpus, 1);
    assert!(!entry.constraints.requires_smt);
}

/// Overrides max_llcs from the scheduler while inheriting everything else.
#[ktstr_test(scheduler = CONSTRAINED_SCHED, max_llcs = 16)]
fn override_sched_constraint(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check the override applies while other fields are still inherited.
#[test]
fn entry_override_sched_constraint() {
    let entry = ktstr::test_support::find_test("override_sched_constraint").unwrap();
    assert_eq!(entry.constraints.max_llcs, Some(16)); // overridden
    assert_eq!(entry.constraints.max_cpus, Some(64)); // inherited from scheduler
    assert_eq!(entry.constraints.max_numa_nodes, Some(1)); // inherited from DEFAULT
}

/// Scheduler with a distinctive topology for inheritance tests.
/// Uses EEVDF (no binary) — the test validates topology inheritance,
/// not scheduler behavior.
const TOPO_SCHED: ktstr::test_support::Scheduler =
    ktstr::test_support::Scheduler::new("topo_test").topology(1, 2, 3, 1);

/// Full topology inheritance: all three dimensions from TOPO_SCHED.
#[ktstr_test(scheduler = TOPO_SCHED)]
fn topo_inherit_full(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Partial topology inheritance: threads overridden, LLCs and cores
/// inherited from TOPO_SCHED.
#[ktstr_test(scheduler = TOPO_SCHED, threads = 2)]
fn topo_inherit_partial(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check full topology inheritance from scheduler.
#[test]
fn entry_topo_inherit_full() {
    let entry = ktstr::test_support::find_test("topo_inherit_full").unwrap();
    assert_eq!(entry.topology.llcs, 2);
    assert_eq!(entry.topology.cores_per_llc, 3);
    assert_eq!(entry.topology.threads_per_core, 1);
}

/// Check partial topology inheritance: threads overridden, rest inherited.
#[test]
fn entry_topo_inherit_partial() {
    let entry = ktstr::test_support::find_test("topo_inherit_partial").unwrap();
    assert_eq!(entry.topology.llcs, 2);
    assert_eq!(entry.topology.cores_per_llc, 3);
    assert_eq!(entry.topology.threads_per_core, 2);
}

/// Test with performance_mode — checks macro sets the field.
#[ktstr_test(llcs = 1, cores = 2, threads = 1, performance_mode = true)]
fn performance_mode_compile(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check performance_mode is set in generated entry.
#[test]
fn entry_performance_mode_set() {
    let entry = ktstr::test_support::find_test("performance_mode_compile").unwrap();
    assert!(
        entry.performance_mode,
        "performance_mode = true must be set in generated entry",
    );
}

// ---------------------------------------------------------------------------
// declare_scheduler! macro tests
// ---------------------------------------------------------------------------

ktstr::declare_scheduler!(TEST_DECLARE, {
    name = "test_derive",
    binary = "scx-ktstr",
    topology = (1, 2, 4, 1),
    cgroup_parent = "/test",
    sched_args = ["--arg1", "--arg2"],
    config_file = "test-config.toml",
});

/// Check the macro generates a const Scheduler with the correct name.
#[test]
fn declare_scheduler_const_name() {
    let _ = &TEST_DECLARE;
    assert_eq!(TEST_DECLARE.name, "test_derive");
}

/// Check scheduler binary spec.
#[test]
fn declare_scheduler_binary() {
    assert!(matches!(
        TEST_DECLARE.binary,
        ktstr::test_support::SchedulerSpec::Discover("scx-ktstr")
    ));
}

/// Check scheduler topology.
#[test]
fn declare_scheduler_topology() {
    assert_eq!(TEST_DECLARE.topology.llcs, 2);
    assert_eq!(TEST_DECLARE.topology.cores_per_llc, 4);
    assert_eq!(TEST_DECLARE.topology.threads_per_core, 1);
}

/// Check scheduler cgroup_parent.
#[test]
fn declare_scheduler_cgroup_parent() {
    assert_eq!(
        TEST_DECLARE.cgroup_parent,
        Some(ktstr::test_support::CgroupPath::new("/test"))
    );
}

/// Check scheduler sched_args.
#[test]
fn declare_scheduler_sched_args() {
    assert_eq!(TEST_DECLARE.sched_args, &["--arg1", "--arg2"]);
}

/// Check scheduler config_file.
#[test]
fn declare_scheduler_config_file() {
    assert_eq!(TEST_DECLARE.config_file, Some("test-config.toml"));
}

/// Check topology inheritance from declared scheduler.
#[ktstr_test(scheduler = TEST_DECLARE)]
fn declare_topo_inherit(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Check topology inheritance from derived scheduler.
#[test]
fn entry_declare_topo_inherit() {
    let entry = ktstr::test_support::find_test("declare_topo_inherit").unwrap();
    assert_eq!(entry.topology.llcs, 2);
    assert_eq!(entry.topology.cores_per_llc, 4);
    assert_eq!(entry.topology.threads_per_core, 1);
}

// ---------------------------------------------------------------------------
// Empty enum edge case
// ---------------------------------------------------------------------------

ktstr::declare_scheduler!(EMPTY_SCHED, {
    name = "empty_sched",
    binary = "empty-binary",
    topology = (1, 1, 2, 1),
});

/// Check the const name is correct.
#[test]
fn declare_scheduler_empty_const_name() {
    assert_eq!(EMPTY_SCHED.name, "empty_sched");
}

/// Check binary spec is wired.
#[test]
fn declare_scheduler_empty_binary() {
    assert!(matches!(
        EMPTY_SCHED.binary,
        ktstr::test_support::SchedulerSpec::Discover("empty-binary")
    ));
}

/// Topology validation: boot a multi-LLC VM and check the guest sees
/// more than the 2-CPU default. The base test boots 2l2c1t (4 CPUs, 2
/// LLCs); gauntlet variants boot larger topologies. Catches regressions
/// where guest-side topology discovery falls back to incorrect defaults.
#[ktstr_test(
    llcs = 2,
    cores = 2,
    threads = 1,
    memory_mb = 2048,
    min_numa_nodes = 2,
    max_numa_nodes = 4,
    min_llcs = 2,
    min_cpus = 4
)]
fn topology_matches_vm_spec(ctx: &Ctx) -> Result<AssertResult> {
    let total = ctx.topo.total_cpus();
    let llcs = ctx.topo.num_llcs();
    let mut details = Vec::new();
    let mut passed = true;

    // The VM must have more than the 2-CPU / 1-LLC default. Any
    // regression that replaces sysfs with the entry default will fail.
    if total < 4 {
        details.push(ktstr::assert::AssertDetail::from(format!(
            "expected >= 4 CPUs, got {total}"
        )));
        passed = false;
    }
    if llcs < 2 {
        details.push(ktstr::assert::AssertDetail::from(format!(
            "expected >= 2 LLCs, got {llcs}"
        )));
        passed = false;
    }
    // LLCs cannot exceed CPU count.
    if llcs > total {
        details.push(ktstr::assert::AssertDetail::from(format!(
            "LLCs ({llcs}) > CPUs ({total})"
        )));
        passed = false;
    }
    if passed {
        Ok(AssertResult::pass())
    } else {
        Ok(AssertResult {
            passed: false,
            skipped: false,
            details,
            stats: Default::default(),
            measurements: std::collections::BTreeMap::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// #[ktstr_test(payload = ..., workloads = [...])] macro surface
// ---------------------------------------------------------------------------

// Payload fixtures live in this test crate so the #[ktstr_test]
// attribute can reference them by local path, exactly the shape a
// real test author will write.
use ktstr::test_support::{OutputFormat, Payload, PayloadKind};

#[allow(dead_code)]
const PAYLOAD_A: Payload = Payload::new(
    "payload_a",
    PayloadKind::Binary("/bin/true"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
    None,
);

#[allow(dead_code)]
const PAYLOAD_B: Payload = Payload::new(
    "payload_b",
    PayloadKind::Binary("/bin/true"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
    None,
);

#[allow(dead_code)]
const PAYLOAD_C: Payload = Payload::new(
    "payload_c",
    PayloadKind::Binary("/bin/true"),
    OutputFormat::ExitCode,
    &[],
    &[],
    &[],
    &[],
    false,
    None,
    None,
);

/// `payload = PATH` wires `entry.payload = Some(&PATH)`. Assertions
/// are inline so the nextest-visible wrapper is the assertion site —
/// plain `#[test]` fns sharing a file with `#[ktstr_test]` are
/// filtered out by the crate's ctor-based nextest dispatcher.
#[ktstr_test(payload = PAYLOAD_A, host_only = true)]
fn macro_payload_attr_wires_entry(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let entry = ktstr::test_support::find_test("macro_payload_attr_wires_entry")
        .expect("self-registration via #[ktstr_test]");
    let payload = entry.payload.expect("payload= must populate entry.payload");
    assert_eq!(payload.name, "payload_a");
    assert!(
        entry.workloads.is_empty(),
        "primary-only: workloads stays empty",
    );
    Ok(AssertResult::pass())
}

/// `workloads = [A, B]` wires `entry.workloads = &[&A, &B]`.
#[ktstr_test(workloads = [PAYLOAD_A, PAYLOAD_B], host_only = true)]
fn macro_workloads_attr_wires_entry(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let entry = ktstr::test_support::find_test("macro_workloads_attr_wires_entry")
        .expect("self-registration via #[ktstr_test]");
    assert!(entry.payload.is_none());
    assert_eq!(entry.workloads.len(), 2);
    assert_eq!(entry.workloads[0].name, "payload_a");
    assert_eq!(entry.workloads[1].name, "payload_b");
    Ok(AssertResult::pass())
}

/// `payload = ...` and `workloads = [...]` coexist — the primary
/// binary runs UNDER the scheduler and composes with any workloads.
#[ktstr_test(payload = PAYLOAD_A, workloads = [PAYLOAD_B, PAYLOAD_C], host_only = true)]
fn macro_payload_and_workloads_coexist(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let entry = ktstr::test_support::find_test("macro_payload_and_workloads_coexist")
        .expect("self-registration via #[ktstr_test]");
    let primary = entry
        .payload
        .expect("payload= must populate entry.payload even when workloads= is set");
    assert_eq!(primary.name, "payload_a");
    assert_eq!(entry.workloads.len(), 2);
    assert_eq!(entry.workloads[0].name, "payload_b");
    assert_eq!(entry.workloads[1].name, "payload_c");
    Ok(AssertResult::pass())
}

/// Defaults: neither attribute set ⇒ `None` / `&[]`. The pre-existing
/// `default_attrs_compile` entry is the subject; this wrapper reads
/// it back via `find_test` to assert the defaults are actually what
/// the macro emitted, not what the field type happens to make
/// visible.
#[ktstr_test(host_only = true)]
fn macro_defaults_leave_payload_none_workloads_empty(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let entry = ktstr::test_support::find_test("default_attrs_compile")
        .expect("default_attrs_compile must be registered");
    assert!(
        entry.payload.is_none(),
        "default macro invocation must leave entry.payload = None",
    );
    assert!(
        entry.workloads.is_empty(),
        "default macro invocation must leave entry.workloads empty",
    );
    Ok(AssertResult::pass())
}

// ---------------------------------------------------------------------------
// `config = EXPR` macro attribute paired with scheduler.config_file_def
// ---------------------------------------------------------------------------

/// Scheduler that declares `config_file_def`, so a paired `#[ktstr_test]`
/// must supply `config = ...`. The macro emits a const assertion that
/// checks the pairing at compile time; this fixture proves the happy
/// path (def + content → registers cleanly with both fields set).
const CFG_PAIRING_SCHED: ktstr::test_support::Scheduler =
    ktstr::test_support::Scheduler::new("cfg_pairing_test")
        .config_file_def("--config {file}", "/include-files/cfg.json");

/// Inline-literal form: `config = "..."` lands as `Some("...")` in the
/// emitted entry's `config_content` field, paired with a scheduler that
/// declares `config_file_def`.
#[ktstr_test(
    scheduler = CFG_PAIRING_SCHED,
    host_only = true,
    config = "{\"layers\":[]}",
)]
fn config_literal_compiles(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// Path form: `config = SOME_CONST` resolves to a `const &'static str`
/// the entry references directly. Same emission shape as the literal
/// form, just with named indirection — proves both expression kinds
/// flow through the parser arm.
const PATH_CONFIG: &str = "{\"path\":true}";

#[ktstr_test(
    scheduler = CFG_PAIRING_SCHED,
    host_only = true,
    config = PATH_CONFIG,
)]
fn config_path_compiles(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    Ok(AssertResult::pass())
}

/// `config = "..."` lands on the entry's `config_content` field as
/// `Some("...")`. Pins the literal-form happy path.
#[test]
fn entry_config_literal_propagates() {
    let entry = ktstr::test_support::find_test("config_literal_compiles")
        .expect("config_literal_compiles must be registered");
    assert_eq!(
        entry.config_content,
        Some("{\"layers\":[]}"),
        "config = \"...\" must wire onto KtstrTestEntry.config_content as Some(...)",
    );
    assert!(
        entry.scheduler.config_file_def.is_some(),
        "fixture scheduler must declare config_file_def",
    );
}

/// `config = SOME_CONST` lands on the entry's `config_content` field
/// as `Some(SOME_CONST)`. Pins the path-form happy path.
#[test]
fn entry_config_path_propagates() {
    let entry = ktstr::test_support::find_test("config_path_compiles")
        .expect("config_path_compiles must be registered");
    assert_eq!(
        entry.config_content,
        Some(PATH_CONFIG),
        "config = SOME_CONST must wire onto KtstrTestEntry.config_content as Some(SOME_CONST)",
    );
}

/// Default macro invocation (no `config = ...`) leaves the entry's
/// `config_content` field at `None`. Pins that the new attribute is
/// strictly opt-in and does not regress existing tests.
#[test]
fn entry_config_default_none() {
    let entry = ktstr::test_support::find_test("default_attrs_compile")
        .expect("default_attrs_compile must be registered");
    assert!(
        entry.config_content.is_none(),
        "omitted config attribute must leave KtstrTestEntry.config_content = None",
    );
}

// ---------------------------------------------------------------------------
// #[derive(Payload)] integration with #[ktstr_test(workloads = [...])]
// ---------------------------------------------------------------------------

/// Derived const is usable as a `&'static Payload` on
/// `#[ktstr_test(workloads = [...])]` — proves the emitted const
/// and PayloadKind::Binary value are indeed static + const-
/// constructible. The remaining structural derive(Payload) tests
/// live in `tests/derive_payload_tests.rs` so they are reachable
/// by the standard `#[test]` harness (the ctor dispatcher in this
/// file intercepts nextest's `--list` for ktstr-registered tests
/// and hides plain `#[test]` functions).
#[derive(ktstr::Payload)]
#[payload(binary = "/bin/true", name = "true")]
#[allow(dead_code)]
struct TruePayload;

#[ktstr_test(workloads = [TRUE], host_only = true)]
fn derive_payload_workloads_accepts_derived_const(ctx: &Ctx) -> Result<AssertResult> {
    let _ = ctx;
    let entry = ktstr::test_support::find_test("derive_payload_workloads_accepts_derived_const")
        .expect("self-registration via #[ktstr_test]");
    assert_eq!(entry.workloads.len(), 1);
    assert_eq!(entry.workloads[0].name, "true");
    Ok(AssertResult::pass())
}

// -- json! macro tests --

#[test]
fn json_macro_simple_object() {
    let s = ktstr::json!({"key": "value", "num": 42});
    assert_eq!(s, r#"{"key":"value","num":42}"#);
}

#[test]
fn json_macro_nested() {
    let s = ktstr::json!({
        "layers": [{
            "name": "batch",
            "kind": { "Grouped": { "cpus_range": [0, 4] } },
        }],
    });
    assert_eq!(
        s,
        r#"{"layers":[{"name":"batch","kind":{"Grouped":{"cpus_range":[0,4]}}}]}"#
    );
}

#[test]
fn json_macro_booleans_and_null() {
    let s = ktstr::json!({"a": true, "b": false, "c": null});
    assert_eq!(s, r#"{"a":true,"b":false,"c":null}"#);
}

#[test]
fn json_macro_negative_number() {
    let s = ktstr::json!({"n": -1});
    assert_eq!(s, r#"{"n":-1}"#);
}

#[test]
fn json_macro_trailing_commas_stripped() {
    let s = ktstr::json!({
        "a": 1,
        "b": 2,
    });
    assert_eq!(s, r#"{"a":1,"b":2}"#);
}

#[test]
fn json_macro_array_trailing_comma() {
    let s = ktstr::json!([1, 2, 3,]);
    assert_eq!(s, "[1,2,3]");
}

#[test]
fn json_macro_const_context() {
    const CFG: &str = ktstr::json!({"hello": "world"});
    assert_eq!(CFG, r#"{"hello":"world"}"#);
}
