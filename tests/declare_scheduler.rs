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
//!
//! The crate-level `#![deny(missing_docs)]` guard below pins the
//! `#[allow(missing_docs)]` attribute that `declare_scheduler!`
//! emits on the generated `pub static` — every macro invocation in
//! this file is a `pub` item, so a regression that drops the
//! suppression would refuse to compile here.

#![deny(missing_docs)]

use ktstr::assert::Assert;
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

// -- missing-docs suppression --
//
// With `#![deny(missing_docs)]` at the crate root above, every
// `pub static` emitted by `declare_scheduler!` is a `pub` item that
// would trip the lint without the macro-emitted
// `#[allow(missing_docs)]` attribute. This declaration plus the
// `allow_missing_docs_attribute_lets_pub_static_compile` test below
// pin that the attribute is in place; if the macro drops it, this
// file fails to compile.

declare_scheduler!(DECLARE_SCHEDULER_NO_DOCS, {
    name = "declare_scheduler_no_docs",
    binary = "scx_no_docs",
});

#[test]
fn allow_missing_docs_attribute_lets_pub_static_compile() {
    assert_eq!(DECLARE_SCHEDULER_NO_DOCS.name, "declare_scheduler_no_docs");
    assert!(matches!(
        DECLARE_SCHEDULER_NO_DOCS.binary,
        SchedulerSpec::Discover("scx_no_docs")
    ));
}

// -- KernelId variant coverage --
//
// Exercises every non-Version `KernelId` shape the macro's
// kernels-validator must accept: range via `..=`, Path,
// Path containing `..` (the macro's dot-dot heuristic is
// CacheKey-only and must not reject paths), Git, and CacheKey.
// A regression that broke the macro's routing for any
// non-Version/non-`..`-Range variant would fail this declaration
// at expand time.

declare_scheduler!(DECLARE_SCHEDULER_KERNEL_VARIANTS, {
    name = "declare_scheduler_kernel_variants",
    binary = "scx-variants",
    kernels = [
        "6.14",
        "6.14..=6.20",
        "/tmp/linux-custom",
        "foo/../bar/linux",
        "git+https://example.com/linux.git#main",
        "my-cache-key-x86-64",
    ],
});

#[test]
fn kernel_variant_strings_accepted_by_macro() {
    assert_eq!(DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels.len(), 6);
    assert_eq!(DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels[0], "6.14");
    assert_eq!(DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels[1], "6.14..=6.20");
    assert_eq!(DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels[2], "/tmp/linux-custom");
    assert_eq!(DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels[3], "foo/../bar/linux");
    assert_eq!(
        DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels[4],
        "git+https://example.com/linux.git#main"
    );
    assert_eq!(DECLARE_SCHEDULER_KERNEL_VARIANTS.kernels[5], "my-cache-key-x86-64");
}

// -- assert + config_file_def fields --
//
// Pins the new fields' threading from the macro through the
// emitted Scheduler. The assert validator accepts method chains
// rooted at a const path (canonical Assert pattern); the
// config_file_def validator accepts a 2-tuple of non-empty
// string literals where arg_template contains `{file}` and
// guest_path is absolute.

declare_scheduler!(DECLARE_SCHEDULER_WITH_ASSERT, {
    name = "declare_scheduler_with_assert",
    binary = "scx-assert",
    assert = Assert::NO_OVERRIDES
        .check_not_starved()
        .max_gap_ms(5000)
        .max_imbalance_ratio(2.5)
        .fail_on_stall(true)
        .sustained_samples(15),
});

#[test]
fn assert_field_threads_to_scheduler() {
    assert_eq!(DECLARE_SCHEDULER_WITH_ASSERT.assert.not_starved, Some(true));
    assert_eq!(DECLARE_SCHEDULER_WITH_ASSERT.assert.max_gap_ms, Some(5000));
    assert_eq!(
        DECLARE_SCHEDULER_WITH_ASSERT.assert.max_imbalance_ratio,
        Some(2.5)
    );
    assert_eq!(DECLARE_SCHEDULER_WITH_ASSERT.assert.fail_on_stall, Some(true));
    assert_eq!(
        DECLARE_SCHEDULER_WITH_ASSERT.assert.sustained_samples,
        Some(15)
    );
    // Unset fields stay None.
    assert_eq!(DECLARE_SCHEDULER_WITH_ASSERT.assert.max_spread_pct, None);
}

declare_scheduler!(DECLARE_SCHEDULER_DEFAULT_CHECKS, {
    name = "declare_scheduler_default_checks",
    binary = "scx-defaults",
    assert = Assert::default_checks(),
});

#[test]
fn assert_accepts_const_fn_call() {
    // `Assert::default_checks()` is a snake_case const fn that the
    // constraints validator would reject; the assert validator
    // accepts it (and any other const-eligible expression).
    let _ = &DECLARE_SCHEDULER_DEFAULT_CHECKS.assert;
}

declare_scheduler!(DECLARE_SCHEDULER_BARE_NO_OVERRIDES, {
    name = "declare_scheduler_bare_no_overrides",
    binary = "scx-bare",
    assert = Assert::NO_OVERRIDES,
});

#[test]
fn assert_accepts_bare_const_path() {
    // Pins the simplest valid assert shape: a bare const path with
    // no method-chain or call. Validator must accept Expr::Path
    // alongside the MethodCall and Call shapes the other
    // declarations exercise.
    assert_eq!(
        DECLARE_SCHEDULER_BARE_NO_OVERRIDES.assert.not_starved,
        None
    );
    assert_eq!(
        DECLARE_SCHEDULER_BARE_NO_OVERRIDES.assert.max_gap_ms,
        None
    );
}

#[test]
fn omitted_assert_defaults_to_no_overrides() {
    // When the macro omits `assert = ...`, Scheduler::new's
    // default (`Assert::NO_OVERRIDES`, all-None) flows through.
    // Verified via DECLARE_SCHEDULER_MINIMAL which omits assert.
    assert_eq!(DECLARE_SCHEDULER_MINIMAL.assert.not_starved, None);
    assert_eq!(DECLARE_SCHEDULER_MINIMAL.assert.max_gap_ms, None);
    assert_eq!(DECLARE_SCHEDULER_MINIMAL.assert.fail_on_stall, None);
}

declare_scheduler!(DECLARE_SCHEDULER_CFG_DEF, {
    name = "declare_scheduler_cfg_def",
    binary = "scx-cfg-def",
    config_file_def = ("--config {file}", "/include-files/cfg.json"),
});

#[test]
fn config_file_def_threads_tuple_to_scheduler() {
    assert_eq!(
        DECLARE_SCHEDULER_CFG_DEF.config_file_def,
        Some(("--config {file}", "/include-files/cfg.json")),
    );
}

declare_scheduler!(DECLARE_SCHEDULER_CFG_DEF_ALT, {
    name = "declare_scheduler_cfg_def_alt",
    binary = "scx-cfg-def-alt",
    config_file_def = ("f:{file}", "/include-files/layered.json"),
});

#[test]
fn config_file_def_supports_compact_arg() {
    assert_eq!(
        DECLARE_SCHEDULER_CFG_DEF_ALT.config_file_def,
        Some(("f:{file}", "/include-files/layered.json")),
    );
}

declare_scheduler!(DECLARE_SCHEDULER_FULL_NEW, {
    name = "declare_scheduler_full_new",
    binary = "scx-full-new",
    assert = Assert::NO_OVERRIDES.max_gap_ms(3000),
    config_file_def = ("--cfg {file}", "/inc/x.json"),
    kernels = ["6.14"],
    sched_args = ["--exit-dump-len", "1024"],
});

#[test]
fn assert_and_config_file_def_coexist_with_other_fields() {
    assert_eq!(DECLARE_SCHEDULER_FULL_NEW.assert.max_gap_ms, Some(3000));
    assert!(DECLARE_SCHEDULER_FULL_NEW.config_file_def.is_some());
    assert_eq!(DECLARE_SCHEDULER_FULL_NEW.kernels, &["6.14"]);
    assert_eq!(
        DECLARE_SCHEDULER_FULL_NEW.sched_args,
        &["--exit-dump-len", "1024"]
    );
}

declare_scheduler!(BOTH_CONFIGS, {
    name = "both_configs",
    binary = "scx-both",
    config_file = "host-path.toml",
    config_file_def = ("--cfg {file}", "/inc/c.json"),
});

#[test]
fn both_config_fields_coexist() {
    assert_eq!(BOTH_CONFIGS.config_file, Some("host-path.toml"));
    assert_eq!(
        BOTH_CONFIGS.config_file_def,
        Some(("--cfg {file}", "/inc/c.json"))
    );
}

declare_scheduler!(ONLY_CONFIG_FILE, {
    name = "only_cf",
    binary = "scx-cf",
    config_file = "host.toml",
});

declare_scheduler!(ONLY_CONFIG_FILE_DEF, {
    name = "only_cfd",
    binary = "scx-cfd",
    config_file_def = ("--c {file}", "/g.json"),
});

#[test]
fn config_fields_independent_defaults() {
    assert_eq!(ONLY_CONFIG_FILE.config_file_def, None);
    assert_eq!(ONLY_CONFIG_FILE_DEF.config_file, None);
}

// -- visibility prefix --
//
// The macro accepts an optional Rust visibility prefix before the
// const name. Omitted defaults to `pub`. Explicit prefixes (`pub`,
// `pub(crate)`, `pub(super)`, `pub(in path)`) flow through verbatim.
// The registry static stays plain `static` regardless of the
// user-facing const's visibility.

declare_scheduler!(pub VIS_EXPLICIT_PUB, {
    name = "vis_explicit_pub",
    binary = "scx-vis-pub",
});

declare_scheduler!(pub(crate) VIS_PUB_CRATE, {
    name = "vis_pub_crate",
    binary = "scx-vis-crate",
});

mod vis_inner {
    use ktstr::declare_scheduler;
    declare_scheduler!(pub(super) VIS_PUB_SUPER, {
        name = "vis_pub_super",
        binary = "scx-vis-super",
    });

    // `pub(in path)` visibility — pinned via a nested module that
    // exposes the const to a specific ancestor path.
    pub mod vis_inner_inner {
        use ktstr::declare_scheduler;
        declare_scheduler!(pub(in super::super::vis_inner) VIS_PUB_IN_PATH, {
            name = "vis_pub_in_path",
            binary = "scx-vis-in-path",
        });
    }

    // Re-export through a function that proves the in-path const is
    // accessible from within `vis_inner`'s scope.
    pub fn pub_in_path_name() -> &'static str {
        vis_inner_inner::VIS_PUB_IN_PATH.name
    }
}

#[test]
fn visibility_prefixes_flow_through_to_emit() {
    // `pub VIS_EXPLICIT_PUB` is reachable from this crate-internal
    // test module — same accessibility as the default-pub case.
    assert_eq!(VIS_EXPLICIT_PUB.name, "vis_explicit_pub");
    // `pub(crate) VIS_PUB_CRATE` is reachable here (we ARE the crate).
    assert_eq!(VIS_PUB_CRATE.name, "vis_pub_crate");
    // `pub(super) VIS_PUB_SUPER` declared inside `vis_inner`; visible
    // here as `vis_inner::VIS_PUB_SUPER` was promoted to the parent
    // (this file) via pub(super). Access through the inner module
    // path proves the visibility chain works.
    assert_eq!(vis_inner::VIS_PUB_SUPER.name, "vis_pub_super");
}

#[test]
fn visibility_prefixed_schedulers_still_register_in_slice() {
    // All four visibility-prefixed declarations register in
    // KTSTR_SCHEDULERS via linkme regardless of Rust visibility,
    // because linkme gathers via link-section walking, not name
    // resolution.
    let names: Vec<&'static str> = KTSTR_SCHEDULERS.iter().map(|s| s.name).collect();
    assert!(names.contains(&"vis_explicit_pub"));
    assert!(names.contains(&"vis_pub_crate"));
    assert!(names.contains(&"vis_pub_super"));
    assert!(names.contains(&"vis_pub_in_path"));
}

#[test]
fn pub_in_path_visibility_accessible_within_path_scope() {
    // The `pub(in super::super::vis_inner)` declaration is reachable
    // from inside `vis_inner` (via the `pub_in_path_name` helper).
    // Confirms `pub(in path)` syntax flows through the macro parser
    // + emit correctly.
    assert_eq!(vis_inner::pub_in_path_name(), "vis_pub_in_path");
}

// -- binary_path (SchedulerSpec::Path) --

declare_scheduler!(DECLARE_SCHEDULER_BINARY_PATH, {
    name = "declare_scheduler_binary_path",
    binary_path = "/usr/local/bin/scx_custom_binary",
});

#[test]
fn binary_path_emits_path_variant() {
    assert!(matches!(
        DECLARE_SCHEDULER_BINARY_PATH.binary,
        SchedulerSpec::Path("/usr/local/bin/scx_custom_binary"),
    ));
    assert_eq!(
        DECLARE_SCHEDULER_BINARY_PATH.binary.display_name(),
        "/usr/local/bin/scx_custom_binary",
    );
    assert!(DECLARE_SCHEDULER_BINARY_PATH.binary.has_active_scheduling());
}

// -- kernel_builtin_enable + kernel_builtin_disable (SchedulerSpec::KernelBuiltin) --

declare_scheduler!(DECLARE_SCHEDULER_KERNEL_BUILTIN, {
    name = "declare_scheduler_kernel_builtin",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    kernel_builtin_disable = ["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
});

#[test]
fn kernel_builtin_pair_emits_kernel_builtin_variant() {
    let SchedulerSpec::KernelBuiltin { enable, disable } =
        DECLARE_SCHEDULER_KERNEL_BUILTIN.binary
    else {
        panic!(
            "expected KernelBuiltin, got display_name {}",
            DECLARE_SCHEDULER_KERNEL_BUILTIN.binary.display_name(),
        );
    };
    assert_eq!(
        enable,
        &["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    );
    assert_eq!(
        disable,
        &["echo 0 > /proc/sys/kernel/sched_autogroup_enabled"],
    );
    assert_eq!(
        DECLARE_SCHEDULER_KERNEL_BUILTIN.binary.display_name(),
        "kernel",
    );
    assert!(DECLARE_SCHEDULER_KERNEL_BUILTIN.binary.has_active_scheduling());
}

declare_scheduler!(DECLARE_SCHEDULER_KERNEL_MULTILINE, {
    name = "declare_scheduler_kernel_multiline",
    kernel_builtin_enable = [
        "# enable both autogroup and schedstats",
        "echo 1 > /proc/sys/kernel/sched_autogroup_enabled",
        "echo 1 > /proc/sys/kernel/sched_schedstats",
    ],
    kernel_builtin_disable = [
        "echo 0 > /proc/sys/kernel/sched_schedstats",
        "echo 0 > /proc/sys/kernel/sched_autogroup_enabled",
    ],
});

#[test]
fn kernel_builtin_accepts_comments_and_multiple_commands() {
    let SchedulerSpec::KernelBuiltin { enable, disable } =
        DECLARE_SCHEDULER_KERNEL_MULTILINE.binary
    else {
        panic!("expected KernelBuiltin");
    };
    assert_eq!(enable.len(), 3);
    assert_eq!(enable[0], "# enable both autogroup and schedstats");
    assert_eq!(disable.len(), 2);
}

declare_scheduler!(DECLARE_SCHEDULER_KERNEL_ENABLE_ONLY, {
    name = "declare_scheduler_kernel_enable_only",
    kernel_builtin_enable = ["echo 1 > /proc/sys/kernel/sched_autogroup_enabled"],
    // The disable list may be empty when the enable side has commands.
    // The macro rejects only the both-empty case (functionally identical
    // to the EEVDF baseline); a one-sided pair expresses "set this kernel
    // policy and leave it set."
    kernel_builtin_disable = [],
});

#[test]
fn kernel_builtin_allows_empty_disable_slot() {
    let SchedulerSpec::KernelBuiltin { enable: _, disable } =
        DECLARE_SCHEDULER_KERNEL_ENABLE_ONLY.binary
    else {
        panic!("expected KernelBuiltin");
    };
    assert!(disable.is_empty());
}

declare_scheduler!(DECLARE_SCHEDULER_NAME_KERNEL_WITH_DISCOVER, {
    name = "kernel",
    binary = "scx-name-kernel-discover",
});

#[test]
fn name_kernel_accepted_when_source_is_discover() {
    // The `name = "kernel"` reservation only fires when the
    // `kernel_builtin_*` pair is set (the variant whose display_name
    // is `"kernel"`). With `binary = ...` the name carries no
    // collision risk and the macro accepts it.
    assert_eq!(DECLARE_SCHEDULER_NAME_KERNEL_WITH_DISCOVER.name, "kernel");
    assert!(matches!(
        DECLARE_SCHEDULER_NAME_KERNEL_WITH_DISCOVER.binary,
        SchedulerSpec::Discover("scx-name-kernel-discover"),
    ));
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
    assert!(names.contains(&"declare_scheduler_no_docs"));
    assert!(names.contains(&"declare_scheduler_kernel_variants"));
    assert!(names.contains(&"declare_scheduler_binary_path"));
    assert!(names.contains(&"declare_scheduler_kernel_builtin"));
    assert!(names.contains(&"declare_scheduler_kernel_multiline"));
    assert!(names.contains(&"declare_scheduler_kernel_enable_only"));
    assert!(names.contains(&"kernel"));
    assert!(names.contains(&"declare_scheduler_with_assert"));
    assert!(names.contains(&"declare_scheduler_default_checks"));
    assert!(names.contains(&"declare_scheduler_bare_no_overrides"));
    assert!(names.contains(&"declare_scheduler_cfg_def"));
    assert!(names.contains(&"declare_scheduler_cfg_def_alt"));
    assert!(names.contains(&"declare_scheduler_full_new"));
    assert!(names.contains(&"both_configs"));
    assert!(names.contains(&"only_cf"));
    assert!(names.contains(&"only_cfd"));
}

#[test]
fn find_scheduler_returns_none_for_unknown_name() {
    // Pin the negative path: a name not in `KTSTR_SCHEDULERS`
    // returns `None`, not `Some(arbitrary)`. A regression that
    // returned the first slice entry on miss would silently
    // produce wrong-scheduler attribution in sidecars.
    assert!(find_scheduler("__definitely_not_a_real_scheduler__").is_none());
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
