//! Coverage for [`declare_scheduler!`]: macro expansion shape,
//! `KTSTR_SCHEDULERS` distributed-slice registration,
//! `find_scheduler` lookup, and `SchedulerJson` serde roundtrip.
//!
//! The trybuild compile-error fixtures for invalid inputs
//! (lowercase name, reserved name, missing required field,
//! type-mismatch on value-expecting fields, topology validation)
//! live alongside the existing `derive_*` fixtures under
//! `tests/compile_fail/` and are exercised by the `compile_fail`
//! integration test target.

use ktstr::declare_scheduler;
use ktstr::test_support::{
    KTSTR_SCHEDULERS, Scheduler, SchedulerJson, SchedulerSpec, Sysctl, TopologyConstraints,
    find_scheduler,
};

// -- minimal expansion --

declare_scheduler!(DECLARE_SCHEDULER_MINIMAL, {
    name = "declare_scheduler_minimal",
    binary = "scx-ktstr",
});

#[test]
fn minimal_expansion_emits_scheduler() {
    assert_eq!(DECLARE_SCHEDULER_MINIMAL.name, "declare_scheduler_minimal");
    assert!(matches!(
        DECLARE_SCHEDULER_MINIMAL.binary,
        SchedulerSpec::Discover("scx-ktstr")
    ));
    assert!(DECLARE_SCHEDULER_MINIMAL.kernels.is_empty());
    assert!(DECLARE_SCHEDULER_MINIMAL.sched_args.is_empty());
    assert!(DECLARE_SCHEDULER_MINIMAL.sysctls.is_empty());
    assert!(DECLARE_SCHEDULER_MINIMAL.kargs.is_empty());
    assert!(DECLARE_SCHEDULER_MINIMAL.cgroup_parent.is_none());
}

// -- full field set --

declare_scheduler!(DECLARE_SCHEDULER_FULL, {
    name = "declare_scheduler_full",
    binary = "scx-full",
    sched_args = ["--a", "--b"],
    kernels = ["6.14", "7.0..7.2"],
    cgroup_parent = "/declare_scheduler_full",
    kargs = ["nosmt"],
    sysctls = [Sysctl::new("k", "v")],
    topology = (1, 2, 4, 1),
    config_file = "cfg.toml",
    constraints = TopologyConstraints {
        min_llcs: 1,
        max_llcs: Some(8),
        max_cpus: Some(64),
        ..TopologyConstraints::DEFAULT
    },
});

#[test]
fn full_field_set_roundtrips() {
    assert_eq!(DECLARE_SCHEDULER_FULL.name, "declare_scheduler_full");
    assert_eq!(DECLARE_SCHEDULER_FULL.sched_args, &["--a", "--b"]);
    assert_eq!(DECLARE_SCHEDULER_FULL.kernels, &["6.14", "7.0..7.2"]);
    assert_eq!(DECLARE_SCHEDULER_FULL.kargs, &["nosmt"]);
    assert_eq!(DECLARE_SCHEDULER_FULL.sysctls.len(), 1);
    assert_eq!(DECLARE_SCHEDULER_FULL.sysctls[0].key, "k");
    assert_eq!(DECLARE_SCHEDULER_FULL.sysctls[0].value, "v");
    assert_eq!(DECLARE_SCHEDULER_FULL.topology.numa_nodes, 1);
    assert_eq!(DECLARE_SCHEDULER_FULL.topology.llcs, 2);
    assert_eq!(DECLARE_SCHEDULER_FULL.topology.cores_per_llc, 4);
    assert_eq!(DECLARE_SCHEDULER_FULL.topology.threads_per_core, 1);
    assert_eq!(DECLARE_SCHEDULER_FULL.constraints.min_llcs, 1);
    assert_eq!(DECLARE_SCHEDULER_FULL.constraints.max_llcs, Some(8));
    assert_eq!(DECLARE_SCHEDULER_FULL.constraints.max_cpus, Some(64));
    assert_eq!(DECLARE_SCHEDULER_FULL.config_file, Some("cfg.toml"));
    assert!(DECLARE_SCHEDULER_FULL.cgroup_parent.is_some());
}

// -- explicit-empty kernels --

declare_scheduler!(DECLARE_SCHEDULER_EXPLICIT_EMPTY, {
    name = "declare_scheduler_explicit_empty",
    binary = "scx-ee",
    kernels = [],
});

#[test]
fn explicit_empty_kernels_equals_default() {
    assert!(DECLARE_SCHEDULER_MINIMAL.kernels.is_empty());
    assert!(DECLARE_SCHEDULER_EXPLICIT_EMPTY.kernels.is_empty());
}

// -- KTSTR_SCHEDULERS registration --

#[test]
fn registers_in_distributed_slice() {
    // Confirm both the macro-emitted const and the registry static
    // are reachable: the lookup returns the same pointer as the
    // exported const itself.
    let found = find_scheduler("declare_scheduler_minimal")
        .expect("declare_scheduler! must register in KTSTR_SCHEDULERS");
    assert!(std::ptr::eq(found, &DECLARE_SCHEDULER_MINIMAL));

    let found_full = find_scheduler("declare_scheduler_full")
        .expect("declare_scheduler! must register every declared scheduler");
    assert!(std::ptr::eq(found_full, &DECLARE_SCHEDULER_FULL));
}

#[test]
fn slice_contains_every_declared_scheduler() {
    let names: Vec<&'static str> = KTSTR_SCHEDULERS.iter().map(|s| s.name).collect();
    assert!(names.contains(&"declare_scheduler_minimal"));
    assert!(names.contains(&"declare_scheduler_full"));
    assert!(names.contains(&"declare_scheduler_explicit_empty"));
}

// -- SchedulerJson roundtrip --

#[test]
fn scheduler_json_serde_roundtrip() {
    let j = SchedulerJson::from_scheduler(&DECLARE_SCHEDULER_FULL);
    let s = serde_json::to_string(&j).expect("serialize");
    let back: SchedulerJson = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back.name, "declare_scheduler_full");
    assert_eq!(back.binary, Some("scx-full".to_string()));
    assert_eq!(back.sched_args, vec!["--a", "--b"]);
    assert_eq!(back.kernels, vec!["6.14", "7.0..7.2"]);
    assert_eq!(back.constraints.min_llcs, 1);
    assert_eq!(back.constraints.max_llcs, Some(8));
    assert_eq!(back.constraints.max_cpus, Some(64));
}

#[test]
fn scheduler_json_eevdf_has_no_binary() {
    let j = SchedulerJson::from_scheduler(&Scheduler::EEVDF);
    assert_eq!(j.name, "eevdf");
    assert!(
        j.binary.is_none(),
        "EEVDF is SchedulerSpec::Eevdf — no binary artifact"
    );
}

#[test]
fn scheduler_json_discover_carries_binary() {
    let j = SchedulerJson::from_scheduler(&DECLARE_SCHEDULER_MINIMAL);
    assert_eq!(j.binary, Some("scx-ktstr".to_string()));
}
