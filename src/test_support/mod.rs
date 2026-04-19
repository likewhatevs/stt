//! Runtime support for `#[ktstr_test]` integration tests.
//!
//! Provides the registration type, distributed slice, VM launcher,
//! and result evaluation. Includes guest-side profraw flush for
//! coverage-instrumented builds.
//!
//! See the [Writing Tests](https://likewhatevs.github.io/ktstr/guide/writing-tests.html)
//! and [`#[ktstr_test]` Macro](https://likewhatevs.github.io/ktstr/guide/writing-tests/ktstr-test-macro.html)
//! chapters of the guide.
//!
//! # Consumer API
//!
//! Test authors interact primarily with the `#[ktstr_test]` proc
//! macro; programmatic test generation can instead populate
//! [`KtstrTestEntry`] values into the [`KTSTR_TESTS`]
//! `linkme` distributed slice. The remaining items in this module
//! are runtime glue invoked by the macro-generated code and the
//! `ktstr` / `cargo-ktstr` binaries.
//!
//! # Module layout
//!
//! Implementation is split across 10 private submodules re-exported
//! at `test_support::*` for a flat public API: `entry` (scheduler +
//! test-entry types), `eval` (host-side VM result evaluation), `probe`
//! (auto-repro and BPF probe pipeline), `dispatch` (ktstr / cargo-ktstr
//! CLI entry points), `sidecar` (per-run JSON records), `output`
//! (guest-output and console parsing), `args` (CLI argument extraction),
//! `profraw` (coverage flush), `topo` (topology override parsing), and
//! `timefmt` (ISO-8601 + run-id helpers).

#[cfg(test)]
use crate::monitor::MonitorSummary;

#[cfg(test)]
use crate::assert::AssertResult;
#[cfg(test)]
use crate::scenario::Ctx;
#[cfg(test)]
use anyhow::Result;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::time::Duration;

pub use crate::scenario::flags::FlagDecl;

mod args;
mod dispatch;
mod entry;
mod eval;
mod output;
mod probe;
mod profraw;
mod sidecar;
mod timefmt;
mod topo;

// extract_probe_stack_arg and extract_work_type_arg are reached in
// production via `super::args::` (probe.rs, eval.rs); the re-export here
// preserves the flat-namespace invariant so `test_support::X` resolves
// uniformly across all CLI arg extractors.
#[allow(unused_imports)]
pub(crate) use args::{
    extract_flags_arg, extract_probe_stack_arg, extract_test_fn_arg, extract_topo_arg,
    extract_work_type_arg,
};
#[cfg(test)]
pub(crate) use sidecar::write_sidecar;
pub use sidecar::{SidecarResult, newest_run_dir, runs_root};
pub(crate) use sidecar::{
    collect_sidecars, format_callback_profile, format_kvm_stats, format_verifier_stats,
    sidecar_dir, write_skip_sidecar,
};

pub use dispatch::{analyze_sidecars, ktstr_main, ktstr_test_early_dispatch, run_ktstr_test};
pub(crate) use entry::validate_entry_flags;
pub use entry::{
    BpfMapWrite, CgroupPath, KTSTR_TESTS, KtstrTestEntry, MemSideCache, NumaDistance, NumaNode,
    Scheduler, SchedulerSpec, Sysctl, Topology, TopologyConstraints, find_test,
};
pub(crate) use eval::run_ktstr_test_inner;
#[cfg(test)]
pub(crate) use eval::{config_file_parts, evaluate_vm_result, scheduler_label};
pub use eval::{nextest_setup, resolve_scheduler, resolve_test_kernel};
#[cfg(test)]
pub(crate) use output::{
    RESULT_END, RESULT_START, SCHED_OUTPUT_END, SCHED_OUTPUT_START, classify_init_stage,
    ensure_kvm, extract_kernel_version, extract_panic_message, extract_sched_ext_dump,
    format_console_diagnostics, parse_assert_result, parse_sched_output, sched_log_fingerprint,
};
pub use probe::maybe_dispatch_vm_test;
#[cfg(test)]
pub(crate) use probe::{PROBE_OUTPUT_END, PROBE_OUTPUT_START, ProbePayload, extract_probe_output};
pub(crate) use probe::{
    PipelineDiagnostics, format_probe_diagnostics, maybe_dispatch_vm_test_with_args,
    maybe_dispatch_vm_test_with_phase_a, start_probe_phase_a,
};
#[cfg(test)]
pub(crate) use profraw::MSG_TYPE_PROFRAW;
pub(crate) use profraw::try_flush_profraw;

// Re-exports used only by the #[cfg(test)] module. `use super::*` in
// the tests brings these into scope; suppress the non-test unused-
// import warning until tests co-locate with their production modules.
#[cfg(test)]
pub(crate) use profraw::{find_symbol_vaddrs, parse_shm_params, target_dir};
#[cfg(test)]
pub(crate) use timefmt::{days_to_ymd, generate_run_id, is_leap, now_iso8601};
pub(crate) use topo::{TopoOverride, parse_topo_string};

// ---------------------------------------------------------------------------
// Test infrastructure requirements
// ---------------------------------------------------------------------------
//
// `require_*` helpers turn missing test infrastructure into a panic with
// an actionable message instead of a silent skip. Use them when a test
// is meaningless without the resource -- a missing kernel, vmlinux,
// scheduler binary, or kernel-symbol resolution means the harness is
// misconfigured, not that the test should pass quietly. CI silently
// passing 100 "tests" that all early-returned because no kernel was
// findable is the failure mode these helpers exist to prevent.
//
// For genuine skips (raw BTF at /sys/kernel/btf/vmlinux, host without
// the architectural dependency the test exercises), call the crate's
// `skip!("reason: {detail}")` macro (see `src/test_macros.rs`). It
// emits the canonical `ktstr: SKIP: ...` line and returns from the
// test.

/// Resolve a kernel image path or panic with an actionable message.
///
/// Wraps [`crate::find_kernel`]: an `Err` (KTSTR_KERNEL points at a
/// path with no kernel image, cache lookup failed) and a successful
/// `Ok(None)` (no kernel discoverable) both panic. Tests that boot a
/// VM cannot proceed without a kernel; silently skipping turns CI
/// breakage into a green run.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn require_kernel() -> std::path::PathBuf {
    match crate::find_kernel() {
        Ok(Some(p)) => p,
        Ok(None) => panic!(
            "ktstr: test requires a kernel but none was found. \
             Set KTSTR_KERNEL to a kernel source dir / version / cache key, \
             place a built kernel under ./linux or ../linux, or run \
             `cargo ktstr kernel build` to populate the cache."
        ),
        Err(e) => panic!("ktstr: kernel resolution failed: {e:#}"),
    }
}

/// Resolve a vmlinux path next to a kernel image or panic.
///
/// `kernel_path` is the value returned by [`require_kernel`]. The
/// vmlinux is required for symbol address lookup, BTF, and probe
/// source resolution -- a kernel image without vmlinux means the
/// cache entry is corrupt or the build was incomplete, which is an
/// infrastructure failure rather than a legitimate skip.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn require_vmlinux(kernel_path: &std::path::Path) -> std::path::PathBuf {
    crate::vmm::find_vmlinux(kernel_path).unwrap_or_else(|| {
        panic!(
            "ktstr: no vmlinux found alongside {}. The cache entry or \
             kernel build is incomplete. Rebuild with `cargo ktstr kernel \
             build --force` or point KTSTR_KERNEL at a directory that \
             contains both the kernel image and `vmlinux`.",
            kernel_path.display(),
        )
    })
}

/// Build a workspace package and return its binary path, or panic.
///
/// Wraps [`crate::build_and_find_binary`]. A failed build or missing
/// artifact for a required scheduler binary (e.g. `scx-ktstr`) is an
/// infrastructure failure -- the workspace is broken, not the test.
#[cfg(test)]
pub(crate) fn require_binary(package: &str) -> std::path::PathBuf {
    crate::build_and_find_binary(package).unwrap_or_else(|e| {
        panic!(
            "ktstr: build of `{package}` failed: {e:#}. \
             Run `cargo build -p {package}` to reproduce and diagnose."
        )
    })
}

/// Resolve [`crate::monitor::symbols::KernelSymbols`] from a vmlinux
/// or panic. The symbol table is required for any host-side memory
/// introspection; an unparseable vmlinux is an infrastructure failure.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn require_kernel_symbols(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::symbols::KernelSymbols {
    crate::monitor::symbols::KernelSymbols::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr: kernel symbol resolution from {} failed: {e:#}",
            vmlinux_path.display(),
        )
    })
}

/// Resolve [`crate::monitor::btf_offsets::KernelOffsets`] from a vmlinux
/// or panic. BTF resolution is required for any host-side kernel
/// struct introspection; a vmlinux whose BTF fails to parse is an
/// infrastructure failure, not a test-skip condition.
#[cfg(test)]
pub(crate) fn require_kernel_offsets(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::btf_offsets::KernelOffsets {
    crate::monitor::btf_offsets::KernelOffsets::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr: kernel BTF resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

/// Resolve [`crate::monitor::btf_offsets::BpfMapOffsets`] from a vmlinux
/// or panic. A vmlinux whose BTF fails to yield BPF map offsets is an
/// infrastructure failure, not a test-skip condition.
#[cfg(test)]
pub(crate) fn require_bpf_map_offsets(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::btf_offsets::BpfMapOffsets {
    crate::monitor::btf_offsets::BpfMapOffsets::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr: BpfMapOffsets resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

/// Resolve [`crate::monitor::btf_offsets::BpfProgOffsets`] from a vmlinux
/// or panic. A vmlinux whose BTF fails to yield BPF program offsets is
/// an infrastructure failure, not a test-skip condition.
#[cfg(test)]
pub(crate) fn require_bpf_prog_offsets(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::btf_offsets::BpfProgOffsets {
    crate::monitor::btf_offsets::BpfProgOffsets::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr: BpfProgOffsets resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

/// Serializes tests that mutate env vars. Shared across every `#[cfg(test)]`
/// module in the crate: nextest runs tests in parallel within a binary, and
/// `std::env::set_var` is process-wide, so any test that mutates an env var
/// must hold this mutex for its full save/mutate/restore window.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::shm_ring::parse_shm_params_from_str;
    use linkme::distributed_slice;

    const EVAL_TOPO: Topology = Topology::new(1, 1, 2, 1);

    // Register a test entry in the distributed slice for unit testing find_test.
    fn __ktstr_inner_unit_test_dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }

    #[distributed_slice(KTSTR_TESTS)]
    static __KTSTR_ENTRY_UNIT_TEST_DUMMY: KtstrTestEntry = KtstrTestEntry {
        name: "__unit_test_dummy__",
        func: __ktstr_inner_unit_test_dummy,
        ..KtstrTestEntry::DEFAULT
    };

    #[test]
    fn find_test_registered_entry() {
        let entry = find_test("__unit_test_dummy__");
        assert!(entry.is_some(), "registered entry should be found");
        let entry = entry.unwrap();
        assert_eq!(entry.name, "__unit_test_dummy__");
        assert_eq!(entry.topology.llcs, 1);
        assert_eq!(entry.topology.cores_per_llc, 2);
    }

    #[test]
    fn find_test_nonexistent() {
        assert!(find_test("__nonexistent_test_xyz__").is_none());
    }

    #[test]
    fn extract_test_fn_arg_equals() {
        let args = vec![
            "ktstr".into(),
            "run".into(),
            "--ktstr-test-fn=my_test".into(),
        ];
        assert_eq!(extract_test_fn_arg(&args), Some("my_test"));
    }

    #[test]
    fn extract_test_fn_arg_space() {
        let args = vec![
            "ktstr".into(),
            "run".into(),
            "--ktstr-test-fn".into(),
            "my_test".into(),
        ];
        assert_eq!(extract_test_fn_arg(&args), Some("my_test"));
    }

    #[test]
    fn extract_test_fn_arg_missing() {
        let args = vec!["ktstr".into(), "run".into()];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    #[test]
    fn extract_test_fn_arg_trailing() {
        let args = vec!["ktstr".into(), "run".into(), "--ktstr-test-fn".into()];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    #[test]
    fn parse_assert_result_valid() {
        let json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("noise\n{RESULT_START}\n{json}\n{RESULT_END}\nmore");
        let r = parse_assert_result(&output).unwrap();
        assert!(r.passed);
    }

    #[test]
    fn parse_assert_result_missing_start() {
        let output = format!("no start\n{RESULT_END}\n");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_missing_end() {
        let output = format!("{RESULT_START}\n{{}}");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_failed() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck 3000ms"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let r = parse_assert_result(&output).unwrap();
        assert!(!r.passed);
        assert_eq!(r.details, vec!["stuck 3000ms"]);
    }

    #[test]
    fn parse_shm_params_absent() {
        // Host /proc/cmdline does not contain KTSTR_SHM_BASE/KTSTR_SHM_SIZE.
        let result = parse_shm_params();
        assert!(
            result.is_none(),
            "host should not have KTSTR_SHM_BASE in /proc/cmdline"
        );
    }

    // -- parse_shm_params_from_str tests --

    #[test]
    fn parse_shm_params_from_str_lowercase_hex() {
        let cmdline = "console=ttyS0 KTSTR_SHM_BASE=0xfc000000 KTSTR_SHM_SIZE=0x400000 quiet";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_uppercase_hex() {
        let cmdline = "KTSTR_SHM_BASE=0XFC000000 KTSTR_SHM_SIZE=0X400000";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xFC000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_no_prefix() {
        let cmdline = "KTSTR_SHM_BASE=fc000000 KTSTR_SHM_SIZE=400000";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_missing_base() {
        let cmdline = "console=ttyS0 KTSTR_SHM_SIZE=0x400000";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_missing_size() {
        let cmdline = "KTSTR_SHM_BASE=0xfc000000 quiet";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_missing_both() {
        let cmdline = "console=ttyS0 quiet";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_empty() {
        assert!(parse_shm_params_from_str("").is_none());
    }

    #[test]
    fn parse_shm_params_from_str_invalid_hex() {
        let cmdline = "KTSTR_SHM_BASE=0xZZZZ KTSTR_SHM_SIZE=0x400000";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    // -- extract_test_fn_arg additional tests --

    #[test]
    fn extract_test_fn_arg_empty_value() {
        let args = vec!["ktstr".into(), "run".into(), "--ktstr-test-fn=".into()];
        assert_eq!(extract_test_fn_arg(&args), Some(""));
    }

    #[test]
    fn extract_test_fn_arg_space_form_empty_args() {
        let args: Vec<String> = vec![];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    // -- parse_assert_result additional tests --

    #[test]
    fn parse_assert_result_malformed_json() {
        let output = format!("{RESULT_START}\nnot valid json\n{RESULT_END}");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_empty_json_between_delimiters() {
        let output = format!("{RESULT_START}\n\n{RESULT_END}");
        assert!(parse_assert_result(&output).is_err());
    }

    #[test]
    fn parse_assert_result_with_details() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Other","message":"err1"},{"kind":"Other","message":"err2"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let r = parse_assert_result(&output).unwrap();
        assert!(!r.passed);
        assert_eq!(r.details.len(), 2);
        assert_eq!(r.details[0], "err1");
        assert_eq!(r.details[1], "err2");
    }

    // -- target_dir tests --

    #[test]
    fn target_dir_with_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "LLVM_COV_TARGET_DIR";
        let prev = std::env::var(key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, "/tmp/my-cov-dir") };
        let dir = target_dir();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert_eq!(dir, PathBuf::from("/tmp/my-cov-dir"));
    }

    #[test]
    fn target_dir_from_llvm_profile_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key_cov = "LLVM_COV_TARGET_DIR";
        let key_prof = "LLVM_PROFILE_FILE";
        let prev_cov = std::env::var(key_cov).ok();
        let prev_prof = std::env::var(key_prof).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe {
            std::env::remove_var(key_cov);
            std::env::set_var(key_prof, "/tmp/cov-target/ktstr-%p-%m.profraw");
        }
        let dir = target_dir();
        unsafe {
            match prev_cov {
                Some(v) => std::env::set_var(key_cov, v),
                None => std::env::remove_var(key_cov),
            }
            match prev_prof {
                Some(v) => std::env::set_var(key_prof, v),
                None => std::env::remove_var(key_prof),
            }
        }
        assert_eq!(dir, PathBuf::from("/tmp/cov-target"));
    }

    #[test]
    fn target_dir_without_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key_cov = "LLVM_COV_TARGET_DIR";
        let key_prof = "LLVM_PROFILE_FILE";
        let prev_cov = std::env::var(key_cov).ok();
        let prev_prof = std::env::var(key_prof).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe {
            std::env::remove_var(key_cov);
            std::env::remove_var(key_prof);
        }
        let dir = target_dir();
        unsafe {
            match prev_cov {
                Some(v) => std::env::set_var(key_cov, v),
                None => std::env::remove_var(key_cov),
            }
            match prev_prof {
                Some(v) => std::env::set_var(key_prof, v),
                None => std::env::remove_var(key_prof),
            }
        }
        // Falls back to current_exe parent + "llvm-cov-target".
        assert!(
            dir.ends_with("llvm-cov-target"),
            "expected path ending in llvm-cov-target, got: {}",
            dir.display()
        );
    }

    // -- shm_write return value on full ring --

    #[test]
    fn shm_write_returns_zero_on_full_ring() {
        use crate::vmm::shm_ring::{HEADER_SIZE, MSG_HEADER_SIZE, shm_init, shm_write};

        // Small ring: header + 32 bytes data.
        let shm_size = HEADER_SIZE + 32;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);

        // Fill the ring: 16-byte header + 16-byte payload = 32 bytes.
        let payload = vec![0xAA; 16];
        let written = shm_write(&mut buf, 0, MSG_TYPE_PROFRAW, &payload);
        assert_eq!(written, MSG_HEADER_SIZE + 16);

        // Ring is full — next write returns 0.
        let written = shm_write(&mut buf, 0, MSG_TYPE_PROFRAW, b"overflow");
        assert_eq!(written, 0);
    }

    // -- resolve_test_kernel tests --

    #[test]
    fn resolve_test_kernel_with_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_TEST_KERNEL";
        let prev = std::env::var(key).ok();
        let exe = crate::resolve_current_exe().unwrap();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, exe.to_str().unwrap()) };
        let result = resolve_test_kernel();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), exe);
    }

    #[test]
    fn resolve_test_kernel_with_nonexistent_env_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_TEST_KERNEL";
        let prev = std::env::var(key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, "/nonexistent/kernel/path") };
        let result = resolve_test_kernel();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert!(result.is_err());
    }

    // -- MSG_TYPE_PROFRAW encoding --

    #[test]
    fn msg_type_profraw_ascii() {
        // 0x50524157 == "PRAW" in ASCII.
        let bytes = MSG_TYPE_PROFRAW.to_be_bytes();
        assert_eq!(&bytes, b"PRAW");
    }

    // -- KVM check --

    #[test]
    fn kvm_accessible_on_test_host() {
        // Verifies /dev/kvm is accessible with read+write permissions.
        ensure_kvm().expect("/dev/kvm not accessible");
    }

    // -- resolve_scheduler tests --

    #[test]
    fn resolve_scheduler_none() {
        let result = resolve_scheduler(&SchedulerSpec::None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn resolve_scheduler_path_exists() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = resolve_scheduler(&SchedulerSpec::Path(Box::leak(
            exe.to_str().unwrap().to_string().into_boxed_str(),
        )))
        .unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn resolve_scheduler_path_missing() {
        let result = resolve_scheduler(&SchedulerSpec::Path("/nonexistent/scheduler"));
        assert!(result.is_err());
    }

    #[test]
    fn resolve_scheduler_name_missing() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SCHEDULER";
        let prev = std::env::var(key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::remove_var(key) };
        let result = resolve_scheduler(&SchedulerSpec::Name("__nonexistent_scheduler_xyz__"));
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert!(result.is_err());
    }

    #[test]
    fn resolve_scheduler_name_via_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SCHEDULER";
        let prev = std::env::var(key).ok();
        let exe = crate::resolve_current_exe().unwrap();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, exe.to_str().unwrap()) };
        let result = resolve_scheduler(&SchedulerSpec::Name("anything"));
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap().unwrap(), exe);
    }

    // -- scheduler_label tests --

    #[test]
    fn scheduler_label_none_empty() {
        assert_eq!(scheduler_label(&SchedulerSpec::None), "");
    }

    #[test]
    fn scheduler_label_name() {
        assert_eq!(
            scheduler_label(&SchedulerSpec::Name("scx_mitosis")),
            " [sched=scx_mitosis]"
        );
    }

    #[test]
    fn scheduler_label_path() {
        assert_eq!(
            scheduler_label(&SchedulerSpec::Path("/usr/bin/sched")),
            " [sched=/usr/bin/sched]"
        );
    }

    // -- nextest_setup --

    #[test]
    fn nextest_setup_writes_kernel_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_TEST_KERNEL";
        let prev = std::env::var(key).ok();
        let exe = crate::resolve_current_exe().unwrap();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, exe.to_str().unwrap()) };

        let mut buf = Vec::new();
        let result = nextest_setup(&[exe.as_path()], &mut buf);

        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }

        assert!(result.is_ok(), "nextest_setup failed: {result:?}");
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.starts_with("KTSTR_TEST_KERNEL="),
            "expected KTSTR_TEST_KERNEL=..., got: {output}"
        );
    }

    // -- parse_sched_output tests --

    #[test]
    fn parse_sched_output_valid() {
        let output = format!(
            "noise\n{SCHED_OUTPUT_START}\nscheduler log line 1\nline 2\n{SCHED_OUTPUT_END}\nmore"
        );
        let parsed = parse_sched_output(&output);
        assert!(parsed.is_some());
        let content = parsed.unwrap();
        assert!(content.contains("scheduler log line 1"));
        assert!(content.contains("line 2"));
    }

    #[test]
    fn parse_sched_output_missing_start() {
        let output = format!("no start\n{SCHED_OUTPUT_END}\n");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_missing_end() {
        let output = format!("{SCHED_OUTPUT_START}\nsome content");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_empty_content() {
        let output = format!("{SCHED_OUTPUT_START}\n\n{SCHED_OUTPUT_END}");
        assert!(parse_sched_output(&output).is_none());
    }

    #[test]
    fn parse_sched_output_with_stack_traces() {
        let stack = "do_enqueue_task+0x1a0/0x380\nbalance_one+0x50/0x100\n";
        let output = format!("{SCHED_OUTPUT_START}\n{stack}\n{SCHED_OUTPUT_END}");
        let parsed = parse_sched_output(&output).unwrap();
        assert!(parsed.contains("do_enqueue_task"));
        assert!(parsed.contains("balance_one"));
    }

    #[test]
    fn parse_sched_output_rfind_survives_end_marker_in_content() {
        // Regression for #25: if the scheduler log echoes the END
        // marker inside its own content (e.g. a shell heredoc, a
        // diagnostic that quotes the sentinel), `find` truncated the
        // section at the first occurrence — which was inside the
        // content, not at the terminator. `rfind` anchors on the last
        // occurrence, which is the real terminator.
        let content = format!("line1\nfake {SCHED_OUTPUT_END} inside\nline3");
        let output = format!("{SCHED_OUTPUT_START}\n{content}\n{SCHED_OUTPUT_END}\n");
        let parsed = parse_sched_output(&output).unwrap();
        assert!(
            parsed.contains("line3"),
            "rfind must keep content after an embedded END marker: {parsed:?}"
        );
        assert!(
            parsed.contains("fake"),
            "content before the embedded marker must also survive: {parsed:?}"
        );
    }

    // -- sched_log_fingerprint tests --

    #[test]
    fn sched_log_fingerprint_last_line() {
        let output = format!(
            "{SCHED_OUTPUT_START}\nstarting scheduler\nError: apply_cell_config BPF program returned error -2\n{SCHED_OUTPUT_END}",
        );
        assert_eq!(
            sched_log_fingerprint(&output),
            Some("Error: apply_cell_config BPF program returned error -2"),
        );
    }

    #[test]
    fn sched_log_fingerprint_skips_trailing_blanks() {
        let output = format!("{SCHED_OUTPUT_START}\nfatal error here\n\n\n{SCHED_OUTPUT_END}",);
        assert_eq!(sched_log_fingerprint(&output), Some("fatal error here"));
    }

    #[test]
    fn sched_log_fingerprint_none_without_markers() {
        assert!(sched_log_fingerprint("no markers").is_none());
    }

    #[test]
    fn sched_log_fingerprint_none_empty_content() {
        let output = format!("{SCHED_OUTPUT_START}\n\n{SCHED_OUTPUT_END}");
        assert!(sched_log_fingerprint(&output).is_none());
    }

    // -- extract_probe_stack_arg tests --

    #[test]
    fn extract_probe_stack_arg_equals() {
        let args = vec![
            "ktstr".into(),
            "run".into(),
            "--ktstr-probe-stack=func_a,func_b".into(),
        ];
        assert_eq!(
            extract_probe_stack_arg(&args),
            Some("func_a,func_b".to_string())
        );
    }

    #[test]
    fn extract_probe_stack_arg_missing() {
        let args = vec!["ktstr".into(), "run".into()];
        assert!(extract_probe_stack_arg(&args).is_none());
    }

    #[test]
    fn extract_probe_stack_arg_empty_value() {
        let args = vec!["ktstr".into(), "--ktstr-probe-stack=".into()];
        assert!(extract_probe_stack_arg(&args).is_none());
    }

    // -- extract_probe_output tests --

    #[test]
    fn extract_probe_output_valid_json() {
        use crate::probe::process::ProbeEvent;
        let payload = ProbePayload {
            events: vec![ProbeEvent {
                func_idx: 0,
                task_ptr: 1,
                ts: 100,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 42)],
                kstack: vec![],
                str_val: None,
                ..Default::default()
            }],
            func_names: vec![(0, "schedule".to_string())],
            bpf_source_locs: Default::default(),
            diagnostics: None,
            nr_cpus: None,
            param_names: Default::default(),
            render_hints: Default::default(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let output = format!("noise\n{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}\nmore");
        let parsed = extract_probe_output(&output, None);
        assert!(parsed.is_some());
        let formatted = parsed.unwrap();
        assert!(
            formatted.contains("schedule"),
            "should contain func name: {formatted}"
        );
        assert!(
            formatted.contains("pid"),
            "should contain field name: {formatted}"
        );
    }

    #[test]
    fn extract_probe_output_missing() {
        assert!(extract_probe_output("no markers", None).is_none());
    }

    #[test]
    fn extract_probe_output_empty() {
        let output = format!("{PROBE_OUTPUT_START}\n\n{PROBE_OUTPUT_END}");
        assert!(extract_probe_output(&output, None).is_none());
    }

    #[test]
    fn extract_probe_output_invalid_json() {
        let output = format!("{PROBE_OUTPUT_START}\nnot valid json\n{PROBE_OUTPUT_END}");
        assert!(extract_probe_output(&output, None).is_none());
    }

    #[test]
    fn extract_probe_output_enriched_fields() {
        use crate::probe::process::ProbeEvent;
        let payload = ProbePayload {
            events: vec![
                ProbeEvent {
                    func_idx: 0,
                    task_ptr: 1,
                    ts: 100,
                    args: [0xDEAD, 0, 0, 0, 0, 0],
                    fields: vec![
                        ("prev:task_struct.pid".to_string(), 42),
                        ("prev:task_struct.scx_flags".to_string(), 0x1c),
                    ],
                    kstack: vec![],
                    str_val: None,
                    ..Default::default()
                },
                ProbeEvent {
                    func_idx: 1,
                    task_ptr: 1,
                    ts: 200,
                    args: [0; 6],
                    fields: vec![("rq:rq.cpu".to_string(), 3)],
                    kstack: vec![],
                    str_val: None,
                    ..Default::default()
                },
            ],
            func_names: vec![
                (0, "schedule".to_string()),
                (1, "pick_task_scx".to_string()),
            ],
            bpf_source_locs: Default::default(),
            diagnostics: None,
            nr_cpus: None,
            param_names: Default::default(),
            render_hints: Default::default(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        let output = format!("{PROBE_OUTPUT_START}\n{json}\n{PROBE_OUTPUT_END}");
        let formatted = extract_probe_output(&output, None).unwrap();

        // Decoded fields present (not raw args).
        assert!(formatted.contains("pid"), "pid field: {formatted}");
        assert!(formatted.contains("42"), "pid value: {formatted}");
        assert!(
            formatted.contains("scx_flags"),
            "scx_flags field: {formatted}"
        );
        assert!(formatted.contains("cpu"), "cpu field: {formatted}");
        assert!(formatted.contains("3"), "cpu value: {formatted}");

        // Type header grouping for struct params.
        assert!(
            formatted.contains("task_struct *prev"),
            "type header for task_struct: {formatted}"
        );
        assert!(
            formatted.contains("rq *rq"),
            "type header for rq: {formatted}"
        );

        // Raw args suppressed when fields present.
        assert!(
            !formatted.contains("arg0"),
            "raw args should not appear when fields exist: {formatted}"
        );

        // Function names present.
        assert!(formatted.contains("schedule"), "func schedule: {formatted}");
        assert!(
            formatted.contains("pick_task_scx"),
            "func pick_task_scx: {formatted}"
        );
    }

    // -- extract_sched_ext_dump tests --

    #[test]
    fn extract_sched_ext_dump_present() {
        let output = "noise\n  ktstr-0  [001]  0.500: sched_ext_dump: Debug dump\n  ktstr-0  [001]  0.501: sched_ext_dump: scheduler state\nmore";
        let parsed = extract_sched_ext_dump(output);
        assert!(parsed.is_some());
        let dump = parsed.unwrap();
        assert!(dump.contains("sched_ext_dump: Debug dump"));
        assert!(dump.contains("sched_ext_dump: scheduler state"));
    }

    #[test]
    fn extract_sched_ext_dump_absent() {
        assert!(extract_sched_ext_dump("no dump lines here").is_none());
    }

    #[test]
    fn extract_sched_ext_dump_empty_output() {
        assert!(extract_sched_ext_dump("").is_none());
    }

    // -- Scheduler method tests --

    #[test]
    fn scheduler_eevdf_defaults() {
        let s = &Scheduler::EEVDF;
        assert_eq!(s.name, "eevdf");
        assert!(s.flags.is_empty());
        assert!(s.sysctls.is_empty());
        assert!(s.kargs.is_empty());
        assert!(s.assert.not_starved.is_none());
        assert!(s.assert.max_imbalance_ratio.is_none());
    }

    static FLAG_A: FlagDecl = FlagDecl {
        name: "flag_a",
        args: &["--flag-a"],
        requires: &[],
    };
    static BORROW: FlagDecl = FlagDecl {
        name: "borrow",
        args: &["--borrow"],
        requires: &[],
    };
    static REBAL: FlagDecl = FlagDecl {
        name: "rebal",
        args: &["--rebal"],
        requires: &[],
    };
    static TEST_LLC: FlagDecl = FlagDecl {
        name: "llc",
        args: &["--llc"],
        requires: &[],
    };
    static TEST_STEAL: FlagDecl = FlagDecl {
        name: "steal",
        args: &["--steal"],
        requires: &[&TEST_LLC],
    };
    static BORROW_LONG: FlagDecl = FlagDecl {
        name: "borrow",
        args: &["--enable-borrow"],
        requires: &[],
    };
    static TEST_A: FlagDecl = FlagDecl {
        name: "a",
        args: &["-a"],
        requires: &[],
    };
    static TEST_B: FlagDecl = FlagDecl {
        name: "b",
        args: &["-b"],
        requires: &[],
    };

    // Static flag slices for tests (Scheduler.flags needs &'static).
    static FLAGS_A: &[&FlagDecl] = &[&FLAG_A];
    static FLAGS_BORROW_REBAL: &[&FlagDecl] = &[&BORROW, &REBAL];
    static FLAGS_STEAL_LLC: &[&FlagDecl] = &[&TEST_STEAL, &TEST_LLC];
    static FLAGS_BORROW_LONG: &[&FlagDecl] = &[&BORROW_LONG];
    static FLAGS_AB: &[&FlagDecl] = &[&TEST_A, &TEST_B];
    static FLAGS_LLC_STEAL: &[&FlagDecl] = &[&TEST_LLC, &TEST_STEAL];

    #[test]
    fn scheduler_new_builder() {
        static TEST_SYSCTLS: &[Sysctl] =
            &[Sysctl::new("kernel.sched_cfs_bandwidth_slice_us", "1000")];
        let s = Scheduler::new("test_sched")
            .binary(SchedulerSpec::Name("test_bin"))
            .flags(FLAGS_A)
            .sysctls(TEST_SYSCTLS)
            .kargs(&["nosmt"]);
        assert_eq!(s.name, "test_sched");
        assert_eq!(s.flags.len(), 1);
        assert_eq!(s.sysctls.len(), 1);
        assert_eq!(s.kargs.len(), 1);
    }

    #[test]
    fn scheduler_supported_flag_names() {
        let s = Scheduler::new("sched").flags(FLAGS_BORROW_REBAL);
        let names = s.supported_flag_names();
        assert_eq!(names, vec!["borrow", "rebal"]);
    }

    #[test]
    fn scheduler_flag_requires_found() {
        let s = Scheduler::new("sched").flags(FLAGS_STEAL_LLC);
        assert_eq!(s.flag_requires("steal"), vec!["llc"]);
        assert!(s.flag_requires("llc").is_empty());
    }

    #[test]
    fn scheduler_flag_requires_not_found() {
        let s = Scheduler::new("sched").flags(&[]);
        assert!(s.flag_requires("nonexistent").is_empty());
    }

    #[test]
    fn scheduler_flag_args_found() {
        let s = Scheduler::new("sched").flags(FLAGS_BORROW_LONG);
        assert_eq!(s.flag_args("borrow"), Some(["--enable-borrow"].as_slice()));
    }

    #[test]
    fn scheduler_flag_args_not_found() {
        let s = Scheduler::new("sched").flags(&[]);
        assert!(s.flag_args("nonexistent").is_none());
    }

    #[test]
    fn scheduler_generate_profiles_no_flags() {
        let s = Scheduler::new("sched");
        let profiles = s.generate_profiles(&[], &[]);
        assert_eq!(profiles.len(), 1);
        assert!(profiles[0].flags.is_empty());
    }

    #[test]
    fn scheduler_generate_profiles_all_optional() {
        let s = Scheduler::new("sched").flags(FLAGS_AB);
        let profiles = s.generate_profiles(&[], &[]);
        assert_eq!(profiles.len(), 4);
    }

    #[test]
    fn scheduler_generate_profiles_with_required() {
        let s = Scheduler::new("sched").flags(FLAGS_AB);
        let profiles = s.generate_profiles(&["a"], &[]);
        assert_eq!(profiles.len(), 2);
        for p in &profiles {
            assert!(p.flags.contains(&"a"));
        }
    }

    #[test]
    fn scheduler_generate_profiles_with_excluded() {
        let s = Scheduler::new("sched").flags(FLAGS_AB);
        let profiles = s.generate_profiles(&[], &["a"]);
        assert_eq!(profiles.len(), 2);
        for p in &profiles {
            assert!(!p.flags.contains(&"a"));
        }
    }

    #[test]
    fn scheduler_generate_profiles_dependency_filter() {
        let s = Scheduler::new("sched").flags(FLAGS_LLC_STEAL);
        let profiles = s.generate_profiles(&[], &[]);
        assert_eq!(profiles.len(), 3);
        let steal_alone = profiles
            .iter()
            .any(|p| p.flags.contains(&"steal") && !p.flags.contains(&"llc"));
        assert!(!steal_alone);
    }

    #[test]
    fn scheduler_with_verify() {
        let v = crate::assert::Assert::NONE
            .check_not_starved()
            .max_imbalance_ratio(3.0);
        let s = Scheduler::new("sched").assert(v);
        assert_eq!(s.assert.not_starved, Some(true));
        assert_eq!(s.assert.max_imbalance_ratio, Some(3.0));
    }

    #[test]
    fn sidecar_result_roundtrip() {
        let sc = SidecarResult {
            test_name: "my_test".to_string(),
            topology: "2s4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: true,
            skipped: false,
            stats: crate::assert::ScenarioStats {
                cgroups: vec![crate::assert::CgroupStats {
                    num_workers: 4,
                    num_cpus: 2,
                    avg_off_cpu_pct: 50.0,
                    min_off_cpu_pct: 40.0,
                    max_off_cpu_pct: 60.0,
                    spread: 20.0,
                    max_gap_ms: 100,
                    max_gap_cpu: 1,
                    total_migrations: 5,
                    ..Default::default()
                }],
                total_workers: 4,
                total_cpus: 2,
                total_migrations: 5,
                worst_spread: 20.0,
                worst_gap_ms: 100,
                worst_gap_cpu: 1,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                prog_stats_deltas: None,
                total_samples: 10,
                max_imbalance_ratio: 1.5,
                max_local_dsq_depth: 3,
                stall_detected: false,
                event_deltas: Some(crate::monitor::ScxEventDeltas {
                    total_fallback: 7,
                    fallback_rate: 0.5,
                    max_fallback_burst: 2,
                    total_dispatch_offline: 0,
                    total_dispatch_keep_last: 3,
                    keep_last_rate: 0.2,
                    total_enq_skip_exiting: 0,
                    total_enq_skip_migration_disabled: 0,
                    ..Default::default()
                }),
                schedstat_deltas: None,
                ..Default::default()
            }),
            stimulus_events: vec![crate::timeline::StimulusEvent {
                elapsed_ms: 500,
                label: "StepStart[0]".to_string(),
                op_kind: Some("SetCpuset".to_string()),
                detail: Some("4 cpus".to_string()),
                total_iterations: None,
            }],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string_pretty(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.test_name, "my_test");
        assert_eq!(loaded.topology, "2s4c2t");
        assert_eq!(loaded.scheduler, "scx_mitosis");
        assert!(loaded.passed);
        assert_eq!(loaded.stats.total_workers, 4);
        assert_eq!(loaded.stats.cgroups.len(), 1);
        assert_eq!(loaded.stats.cgroups[0].num_workers, 4);
        assert_eq!(loaded.stats.worst_spread, 20.0);
        let mon = loaded.monitor.unwrap();
        assert_eq!(mon.total_samples, 10);
        assert_eq!(mon.max_imbalance_ratio, 1.5);
        assert_eq!(mon.max_local_dsq_depth, 3);
        assert!(!mon.stall_detected);
        let deltas = mon.event_deltas.unwrap();
        assert_eq!(deltas.total_fallback, 7);
        assert_eq!(deltas.total_dispatch_keep_last, 3);
        assert_eq!(loaded.stimulus_events.len(), 1);
        assert_eq!(loaded.stimulus_events[0].label, "StepStart[0]");
    }

    #[test]
    fn sidecar_result_roundtrip_no_monitor() {
        let sc = SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1s2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: false,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.test_name, "eevdf_test");
        assert!(!loaded.passed);
        assert!(loaded.monitor.is_none());
        assert!(loaded.stimulus_events.is_empty());
        // monitor field should be absent from JSON when None
        assert!(!json.contains("\"monitor\""));
    }

    // -- extract_topo_arg tests --

    #[test]
    fn extract_topo_arg_equals() {
        let args = vec!["bin".into(), "--ktstr-topo=2s4c2t".into()];
        assert_eq!(extract_topo_arg(&args), Some("2s4c2t".to_string()));
    }

    #[test]
    fn extract_topo_arg_missing() {
        let args = vec!["bin".into(), "--ktstr-test-fn=test".into()];
        assert!(extract_topo_arg(&args).is_none());
    }

    #[test]
    fn extract_topo_arg_empty_value() {
        let args = vec!["bin".into(), "--ktstr-topo=".into()];
        assert!(extract_topo_arg(&args).is_none());
    }

    #[test]
    fn extract_topo_arg_with_other_args() {
        let args = vec![
            "bin".into(),
            "--ktstr-test-fn=my_test".into(),
            "--ktstr-topo=1s2c1t".into(),
        ];
        assert_eq!(extract_topo_arg(&args), Some("1s2c1t".to_string()));
    }

    #[test]
    fn extract_kernel_version_from_boot() {
        let console = "[    0.000000] Linux version 6.14.0-rc3+ (user@host) (gcc) #1 SMP\n\
                        [    0.001000] Command line: console=ttyS0";
        assert_eq!(
            extract_kernel_version(console),
            Some("6.14.0-rc3+".to_string()),
        );
    }

    #[test]
    fn extract_kernel_version_none() {
        assert_eq!(extract_kernel_version("no kernel here"), None);
    }

    #[test]
    fn extract_kernel_version_bare() {
        let console = "Linux version 6.12.0";
        assert_eq!(extract_kernel_version(console), Some("6.12.0".to_string()),);
    }

    // -- format_console_diagnostics tests --

    #[test]
    fn format_console_diagnostics_empty_ok() {
        assert_eq!(format_console_diagnostics("", 0, "test stage"), "");
    }

    #[test]
    fn format_console_diagnostics_empty_nonzero_exit() {
        let s = format_console_diagnostics("", 1, "test stage");
        assert!(s.contains("exit_code=1"));
        assert!(s.contains("--- diagnostics ---"));
        assert!(s.contains("stage: test stage"));
        assert!(!s.contains("console ("));
    }

    #[test]
    fn format_console_diagnostics_with_console() {
        let console = "line1\nline2\nKernel panic - not syncing\n";
        let s = format_console_diagnostics(console, -1, "payload started");
        assert!(s.contains("exit_code=-1"));
        assert!(s.contains("console (3 lines)"));
        assert!(s.contains("Kernel panic"));
        assert!(s.contains("stage: payload started"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn format_console_diagnostics_truncates_long() {
        let lines: Vec<String> = (0..50).map(|i| format!("boot line {i}")).collect();
        let console = format!("{}\n", lines.join("\n"));
        let s = format_console_diagnostics(&console, 0, "test");
        assert!(s.contains("console (20 lines)"));
        assert!(s.contains("boot line 49"));
        assert!(!s.contains("boot line 29"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn format_console_diagnostics_short_console() {
        let console = "Linux version 6.14.0\nbooted ok\n";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (2 lines)"));
        assert!(s.contains("Linux version 6.14.0"));
        assert!(s.contains("booted ok"));
        assert!(!s.contains("truncated"));
    }

    #[test]
    fn format_console_diagnostics_no_truncation_with_trailing_newline() {
        let console = "line1\nline2\nline3\n";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("console (3 lines)"));
        assert!(!s.contains("truncated"));
        assert!(!s.contains("[truncated]"));
    }

    #[test]
    fn format_console_diagnostics_truncation_without_trailing_newline() {
        let console = "line1\nline2\npartial li";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains(", truncated)"));
        assert!(s.contains("partial li [truncated]"));
    }

    // -- extract_work_type_arg tests --

    #[test]
    fn extract_work_type_arg_equals() {
        let args = vec!["ktstr".into(), "--ktstr-work-type=CpuSpin".into()];
        assert_eq!(extract_work_type_arg(&args), Some("CpuSpin".to_string()));
    }

    #[test]
    fn extract_work_type_arg_missing() {
        let args = vec!["ktstr".into(), "run".into()];
        assert!(extract_work_type_arg(&args).is_none());
    }

    #[test]
    fn extract_work_type_arg_empty_value() {
        let args = vec!["ktstr".into(), "--ktstr-work-type=".into()];
        assert!(extract_work_type_arg(&args).is_none());
    }

    // -- collect_sidecars tests --

    #[test]
    fn collect_sidecars_empty_dir() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-empty-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_nonexistent_dir() {
        let results = collect_sidecars(std::path::Path::new("/nonexistent/path"));
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_reads_json() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-json-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let sc = SidecarResult {
            test_name: "test_x".to_string(),
            topology: "1s2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(tmp.join("test_x.ktstr.json"), &json).unwrap();
        // Non-ktstr JSON should be ignored.
        std::fs::write(tmp.join("other.json"), r#"{"key":"val"}"#).unwrap();
        let results = collect_sidecars(&tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "test_x");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_recurses_one_level() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-recurse-test");
        let sub = tmp.join("job-0");
        std::fs::create_dir_all(&sub).unwrap();
        let sc = SidecarResult {
            test_name: "nested_test".to_string(),
            topology: "2s4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: false,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(sub.join("nested_test.ktstr.json"), &json).unwrap();
        let results = collect_sidecars(&tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "nested_test");
        assert!(!results[0].passed);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_skips_invalid_json() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-invalid-test");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("bad.ktstr.json"), "not json").unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_skips_non_ktstr_json() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-notktstr-test");
        std::fs::create_dir_all(&tmp).unwrap();
        // File ends in .json but does NOT contain ".ktstr." in the name
        std::fs::write(tmp.join("other.json"), r#"{"test":"val"}"#).unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn sidecar_result_work_type_field() {
        let sc = SidecarResult {
            test_name: "t".to_string(),
            topology: "1s1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "Bursty".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.work_type, "Bursty");
    }

    #[test]
    fn write_sidecar_defaults_to_target_dir_without_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let kernel_key = "KTSTR_KERNEL";
        let target_key = "CARGO_TARGET_DIR";
        let prev = std::env::var(key).ok();
        let prev_kernel = std::env::var(kernel_key).ok();
        let prev_target = std::env::var(target_key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe {
            std::env::remove_var(key);
            std::env::remove_var(kernel_key);
            std::env::remove_var(target_key);
        };

        let dir = sidecar_dir();
        let expected = format!("target/ktstr/unknown-{}", crate::GIT_HASH);
        assert_eq!(dir, PathBuf::from(&expected));

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_default_dir__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let verify_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin", &[]);

        // Clean up written file.
        let path = dir.join("__sidecar_default_dir__.ktstr.json");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);

        unsafe {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
            match prev_kernel {
                Some(v) => std::env::set_var(kernel_key, v),
                None => std::env::remove_var(kernel_key),
            }
            match prev_target {
                Some(v) => std::env::set_var(target_key, v),
                None => std::env::remove_var(target_key),
            }
        }
    }

    #[test]
    fn write_sidecar_writes_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-write-test");
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_write_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let verify_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin", &[]);

        // Sidecar filename now includes a variant hash suffix so
        // gauntlet variants don't clobber each other. Find the file
        // by prefix match rather than exact path.
        let path: std::path::PathBuf = std::fs::read_dir(&tmp)
            .expect("sidecar dir was created")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("__sidecar_write_test__-") && n.ends_with(".ktstr.json"))
                    .unwrap_or(false)
            })
            .expect("sidecar file with variant suffix should be written");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__sidecar_write_test__");
        assert!(loaded.passed);
        assert!(!loaded.skipped, "pass result is not a skip");

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_active_flags() {
        // Regression for #34: two gauntlet variants differing ONLY in
        // active_flags must produce distinct sidecar filenames so
        // neither clobbers the other. This is the scenario the prior
        // fix (based on work_type/sysctls/kargs alone) missed.
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-flagvariant-test");
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__flagvariant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let ok = AssertResult::pass();
        let flags_a = vec!["llc".to_string()];
        let flags_b = vec!["llc".to_string(), "steal".to_string()];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_a);
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_b);

        let names: Vec<String> = std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("__flagvariant_test__-"))
            .collect();
        assert_eq!(
            names.len(),
            2,
            "two active_flags variants must produce two distinct files, got {names:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_work_types() {
        // Regression for #34: two gauntlet variants differing only in
        // work_type must produce distinct sidecar filenames so neither
        // clobbers the other.
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-variant-test");
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__variant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[]);
        write_sidecar(&entry, &vm_result, &[], &ok, "YieldHeavy", &[]);

        let names: Vec<String> = std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("__variant_test__-"))
            .collect();
        assert_eq!(
            names.len(),
            2,
            "two work_type variants must produce two distinct files, got {names:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn find_test_from_distributed_slice() {
        // KTSTR_TESTS should contain at least the __unit_test_dummy__ entry.
        assert!(!KTSTR_TESTS.is_empty());
    }

    // -- KtstrTestEntry::validate coverage --

    fn validate_entry(
        name: &'static str,
        memory_mb: u32,
        replicas: u32,
        duration: Duration,
        workers_per_cgroup: u32,
    ) -> KtstrTestEntry {
        KtstrTestEntry {
            name,
            memory_mb,
            replicas,
            duration,
            workers_per_cgroup,
            ..KtstrTestEntry::DEFAULT
        }
    }

    #[test]
    fn ktstr_test_entry_validate_accepts_defaults() {
        let e = validate_entry("ok", 512, 1, Duration::from_secs(2), 2);
        e.validate().unwrap();
    }

    #[test]
    fn ktstr_test_entry_validate_rejects_empty_name() {
        let e = validate_entry("", 512, 1, Duration::from_secs(2), 2);
        let err = e.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("name") && msg.contains("non-empty"),
            "got: {msg}"
        );
    }

    #[test]
    fn ktstr_test_entry_validate_rejects_zero_memory() {
        let e = validate_entry("t", 0, 1, Duration::from_secs(2), 2);
        let err = e.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("memory_mb") && msg.contains("> 0") && msg.contains("'t'"),
            "got: {msg}"
        );
    }

    #[test]
    fn ktstr_test_entry_validate_rejects_zero_replicas() {
        let e = validate_entry("t", 512, 0, Duration::from_secs(2), 2);
        let err = e.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("replicas") && msg.contains("> 0"),
            "got: {msg}"
        );
    }

    #[test]
    fn ktstr_test_entry_validate_rejects_zero_duration() {
        let e = validate_entry("t", 512, 1, Duration::ZERO, 2);
        let err = e.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("duration") && msg.contains("> 0"),
            "got: {msg}"
        );
    }

    #[test]
    fn ktstr_test_entry_validate_rejects_zero_workers() {
        let e = validate_entry("t", 512, 1, Duration::from_secs(2), 0);
        let err = e.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("workers_per_cgroup") && msg.contains("> 0"),
            "got: {msg}"
        );
    }

    // -- evaluate_vm_result error path tests --

    fn dummy_test_fn(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }

    fn eevdf_entry(name: &'static str) -> KtstrTestEntry {
        KtstrTestEntry {
            name,
            func: dummy_test_fn,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        }
    }

    static SCHED_TEST: Scheduler = Scheduler {
        name: "test_sched",
        binary: SchedulerSpec::Name("test_sched_bin"),
        flags: &[],
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NONE,
        cgroup_parent: None,
        sched_args: &[],
        topology: crate::vmm::topology::Topology {
            llcs: 1,
            cores_per_llc: 2,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        },
        constraints: TopologyConstraints::DEFAULT,
        config_file: None,
    };

    fn sched_entry(name: &'static str) -> KtstrTestEntry {
        KtstrTestEntry {
            name,
            func: dummy_test_fn,
            scheduler: &SCHED_TEST,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        }
    }

    fn no_repro(_output: &str) -> Option<String> {
        None
    }

    fn make_vm_result(
        output: &str,
        stderr: &str,
        exit_code: i32,
        timed_out: bool,
    ) -> crate::vmm::VmResult {
        crate::vmm::VmResult {
            success: !timed_out && exit_code == 0,
            exit_code,
            duration: std::time::Duration::from_secs(1),
            timed_out,
            output: output.to_string(),
            stderr: stderr.to_string(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        }
    }

    #[test]
    fn eval_eevdf_no_com2_output() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_eevdf_no_out__");
        let result = make_vm_result("", "boot log line\nKernel panic", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("test function produced no output"),
            "EEVDF with no COM2 output should say 'test function produced no output', got: {msg}",
        );
        assert!(
            !msg.contains("no test result received from guest"),
            "EEVDF error should not use the scheduler-path wording, got: {msg}",
        );
        assert!(
            msg.contains("exit_code=1"),
            "should include exit code, got: {msg}"
        );
        assert!(
            msg.contains("Kernel panic"),
            "should include console output, got: {msg}"
        );
    }

    #[test]
    fn eval_sched_dies_no_com2_output() {
        let entry = sched_entry("__eval_sched_dies__");
        let result = make_vm_result("", "boot ok", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no test result received from guest"),
            "scheduler present with no output should take the scheduler-path fallback, got: {msg}",
        );
        assert!(
            !msg.contains("test function produced no output"),
            "should not say 'test function produced no output' when scheduler is set, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_dies_with_sched_log() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let sched_log = format!(
            "noise\n{SCHED_OUTPUT_START}\ndo_enqueue_task+0x1a0\nbalance_one+0x50\n{SCHED_OUTPUT_END}\nmore",
        );
        let entry = sched_entry("__eval_sched_log__");
        let result = make_vm_result(&sched_log, "", -1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("no test result received from guest"),
            "should take the scheduler-path fallback, got: {msg}",
        );
        assert!(
            msg.contains("--- scheduler log ---"),
            "should include scheduler log section, got: {msg}",
        );
        assert!(
            msg.contains("do_enqueue_task"),
            "should include scheduler log content, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_mid_test_death_triggers_repro() {
        // Scheduler dies mid-test: sched_exit_monitor dumps log to COM2
        // but does NOT write "SCHEDULER_DIED". Auto-repro should still
        // trigger because has_active_scheduling() is true and no
        // AssertResult was produced.
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nError: BPF program error\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_mid_death_repro__");
        let result = make_vm_result(&sched_log, "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let repro_called = std::sync::atomic::AtomicBool::new(false);
        let repro_fn = |_output: &str| -> Option<String> {
            repro_called.store(true, std::sync::atomic::Ordering::Relaxed);
            Some("repro data".to_string())
        };
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &repro_fn,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            repro_called.load(std::sync::atomic::Ordering::Relaxed),
            "repro_fn should be called for mid-test scheduler death without SCHEDULER_DIED marker",
        );
        assert!(
            msg.contains("--- auto-repro ---"),
            "error should include auto-repro section, got: {msg}",
        );
        assert!(
            msg.contains("repro data"),
            "error should include repro output, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_repro_no_data_shows_diagnostic() {
        // When repro_fn returns the fallback diagnostic, the error
        // output should include it so the user knows auto-repro was
        // tried and why it produced nothing.
        let entry = sched_entry("__eval_repro_no_data__");
        let result = make_vm_result("", "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let repro_fn = |_output: &str| -> Option<String> {
            Some(
                "auto-repro: no probe data — scheduler may have exited before \
                 probes could attach. Check the sched_ext dump and scheduler \
                 log sections above for crash details."
                    .to_string(),
            )
        };
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &repro_fn,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--- auto-repro ---"),
            "should include auto-repro section, got: {msg}",
        );
        assert!(
            msg.contains("no probe data"),
            "should include diagnostic message, got: {msg}",
        );
        assert!(
            msg.contains("sched_ext dump"),
            "should direct user to dump section, got: {msg}",
        );
    }

    #[test]
    fn eval_timeout_no_result() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_timeout__");
        let result = make_vm_result("", "booting...\nstill booting...", 0, true);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("timed out"),
            "should say timed out, got: {msg}",
        );
        assert!(
            msg.contains("no result in SHM or COM2"),
            "should mention SHM or COM2, got: {msg}",
        );
        assert!(
            msg.contains("booting"),
            "should include console output, got: {msg}",
        );
        assert!(
            msg.contains("[topo="),
            "error should include topology, got: {msg}",
        );
    }

    #[test]
    fn eval_payload_exits_no_verify_result() {
        // Payload wrote something to COM2 but not a valid AssertResult.
        let entry = eevdf_entry("__eval_no_verify__");
        let result = make_vm_result(
            "some output but no delimiters",
            "Linux version 6.14.0\nboot complete",
            0,
            false,
        );
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("test function produced no output"),
            "non-parseable COM2 with EEVDF should say 'test function produced no output', got: {msg}",
        );
        assert!(
            !msg.contains("no test result received from guest"),
            "EEVDF should not use the scheduler-path wording, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_ext_dump_included() {
        let dump_line = "ktstr-0 [001] 0.5: sched_ext_dump: Debug dump line";
        let entry = sched_entry("__eval_dump__");
        let result = make_vm_result("", dump_line, -1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("--- sched_ext dump ---"),
            "should include dump section, got: {msg}",
        );
        assert!(
            msg.contains("sched_ext_dump: Debug dump"),
            "should include dump content, got: {msg}",
        );
    }

    #[test]
    fn eval_verify_result_passed_returns_ok() {
        let json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_pass__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        assert!(
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro,
            )
            .is_ok(),
            "passing AssertResult should return Ok",
        );
    }

    #[test]
    fn eval_verify_result_failed_includes_details() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck 3000ms"},{"kind":"Unfair","message":"spread 45%"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_fail_details__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("failed:"), "got: {msg}");
        assert!(msg.contains("stuck 3000ms"), "got: {msg}");
        assert!(msg.contains("spread 45%"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_includes_sched_log() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"worker 0 stuck 5000ms"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nscheduler noise line\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fail_sched_log__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("worker 0 stuck 5000ms"), "got: {msg}");
        assert!(msg.contains("scheduler noise"), "got: {msg}");
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_has_fingerprint() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck 3000ms"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let error_line = "Error: apply_cell_config BPF program returned error -2";
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fingerprint__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains(error_line), "got: {msg}");
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(fp_pos < name_pos, "got: {msg}");
    }

    #[test]
    fn eval_timeout_has_fingerprint() {
        let error_line = "Error: scheduler panicked";
        let output = format!("{SCHED_OUTPUT_START}\n{error_line}\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_timeout_fp__");
        let result = make_vm_result(&output, "", 0, true);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(error_line),
            "timeout should contain fingerprint, got: {msg}",
        );
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(
            fp_pos < name_pos,
            "fingerprint should appear before ktstr_test line, got: {msg}",
        );
    }

    #[test]
    fn eval_no_result_has_fingerprint() {
        let error_line = "Error: fatal scheduler crash";
        let output =
            format!("{SCHED_OUTPUT_START}\nstartup log\n{error_line}\n{SCHED_OUTPUT_END}",);
        let entry = sched_entry("__eval_no_result_fp__");
        let result = make_vm_result(&output, "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains(error_line),
            "no-result failure should contain fingerprint, got: {msg}",
        );
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(
            fp_pos < name_pos,
            "fingerprint should appear before ktstr_test line, got: {msg}",
        );
    }

    #[test]
    fn eval_no_sched_output_no_fingerprint() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Stuck","message":"stuck"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_no_fp__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.starts_with("ktstr_test"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_has_fingerprint() {
        let pass_json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let error_line = "Error: imbalance detected internally";
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",);
        let output = format!("{RESULT_START}\n{pass_json}\n{RESULT_END}\n{sched_log}");
        let entry = sched_entry("__eval_monitor_fp__");
        let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
            .map(|i| {
                crate::monitor::MonitorSample::new(
                    (i * 100) as u64,
                    vec![
                        crate::monitor::CpuSnapshot {
                            nr_running: 10,
                            scx_nr_running: 10,
                            local_dsq_depth: 0,
                            rq_clock: 1000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                        crate::monitor::CpuSnapshot {
                            nr_running: 1,
                            scx_nr_running: 1,
                            local_dsq_depth: 0,
                            rq_clock: 2000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                    ],
                )
            })
            .collect();
        let summary =
            crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
        let result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output,
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: imbalance_samples,
                summary,
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let assertions = crate::assert::Assert::default_checks();
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(
            msg.contains("passed scenario but monitor failed"),
            "got: {msg}"
        );
        assert!(msg.contains(error_line), "got: {msg}");
        let fp_pos = msg.find(error_line).unwrap();
        let name_pos = msg.find("ktstr_test").unwrap();
        assert!(fp_pos < name_pos, "got: {msg}");
    }

    #[test]
    fn eval_timeout_with_sched_includes_diagnostics() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = sched_entry("__eval_timeout_sched__");
        let result = make_vm_result("", "Linux version 6.14.0\nkernel panic here", -1, true);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("timed out"),
            "should say timed out, got: {msg}"
        );
        assert!(
            msg.contains("[sched=test_sched_bin]"),
            "should include scheduler label, got: {msg}"
        );
        assert!(
            msg.contains("--- diagnostics ---"),
            "should include diagnostics, got: {msg}"
        );
        assert!(
            msg.contains("kernel panic here"),
            "should include console tail, got: {msg}"
        );
    }

    // -- classify_init_stage tests --

    #[test]
    fn classify_no_sentinels() {
        assert_eq!(
            classify_init_stage(""),
            "init script never started (kernel or mount failure)",
        );
    }

    #[test]
    fn classify_init_started_only() {
        assert_eq!(
            classify_init_stage("KTSTR_INIT_STARTED\nsome noise"),
            "init started but payload never ran (cgroup/scheduler setup failed)",
        );
    }

    #[test]
    fn classify_payload_starting() {
        let output = "KTSTR_INIT_STARTED\nKTSTR_PAYLOAD_STARTING\nsome output";
        assert_eq!(
            classify_init_stage(output),
            "payload started but produced no test result",
        );
    }

    #[test]
    fn classify_payload_starting_without_init() {
        // Edge case: payload sentinel present but init sentinel missing.
        // payload_starting implies init ran, so classify as payload started.
        assert_eq!(
            classify_init_stage("KTSTR_PAYLOAD_STARTING"),
            "payload started but produced no test result",
        );
    }

    // -- sentinel integration in evaluate_vm_result --

    #[test]
    fn eval_no_sentinels_shows_initramfs_failure() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_no_sentinel__");
        let result = make_vm_result("", "Kernel panic", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("init script never started"),
            "no sentinels should indicate kernel/mount failure, got: {msg}",
        );
    }

    #[test]
    fn eval_init_started_but_no_payload() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_init_only__");
        let result = make_vm_result("KTSTR_INIT_STARTED\n", "boot log", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("init started but payload never ran"),
            "init sentinel only should indicate cgroup/scheduler setup failure, got: {msg}",
        );
    }

    #[test]
    fn eval_payload_started_no_result() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_payload_start__");
        let output = "KTSTR_INIT_STARTED\nKTSTR_PAYLOAD_STARTING\ngarbage";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("payload started but produced no test result"),
            "both sentinels should indicate payload ran but failed, got: {msg}",
        );
    }

    // -- guest panic detection tests --

    #[test]
    fn eval_crash_in_output_says_guest_crashed() {
        let entry = sched_entry("__eval_crash_detect__");
        let output = "KTSTR_INIT_STARTED\nPANIC: panicked at src/foo.rs:42: assertion failed";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("guest crashed:"), "got: {msg}");
        assert!(msg.contains("assertion failed"), "got: {msg}");
    }

    #[test]
    fn eval_crash_eevdf_says_guest_crashed() {
        let entry = eevdf_entry("__eval_crash_eevdf__");
        let output = "PANIC: panicked at src/bar.rs:10: index out of bounds";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("guest crashed:"), "got: {msg}");
        assert!(msg.contains("index out of bounds"), "got: {msg}");
    }

    #[test]
    fn eval_crash_message_from_shm() {
        let entry = sched_entry("__eval_crash_shm__");
        let shm_crash = "PANIC: panicked at src/test.rs:42: assertion failed\n   \
                          0: ktstr::vmm::rust_init::ktstr_guest_init\n";
        // COM2 also has a PANIC: line (serial fallback). SHM must take priority.
        let output = "PANIC: panicked at src/test.rs:42: assertion failed";
        let mut result = make_vm_result(output, "", 1, false);
        result.crash_message = Some(shm_crash.to_string());
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(
            &entry,
            &result,
            &assertions,
            &[],
            &EVAL_TOPO,
            &[],
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("guest crashed:"),
            "should say 'guest crashed:', got: {msg}",
        );
        assert!(
            msg.contains("ktstr_guest_init"),
            "SHM backtrace content should be present, got: {msg}",
        );
        // SHM path uses "guest crashed:\n{shm_crash}" (multiline),
        // COM2 path uses "guest crashed: {msg}" (single line).
        // The backtrace frame proves SHM was used, not COM2.
        assert!(
            msg.contains("0: ktstr::vmm::rust_init::ktstr_guest_init"),
            "full backtrace from SHM should appear, got: {msg}",
        );
    }

    #[test]
    fn extract_panic_message_found() {
        let output = "noise\nPANIC: panicked at src/main.rs:5: oh no\nmore";
        assert_eq!(
            extract_panic_message(output),
            Some("panicked at src/main.rs:5: oh no"),
        );
    }

    #[test]
    fn extract_panic_message_absent() {
        assert!(extract_panic_message("no panic here").is_none());
    }

    #[test]
    fn extract_panic_message_empty() {
        assert!(extract_panic_message("").is_none());
    }

    // -- format_verifier_stats tests --

    fn make_sidecar_with_vstats(
        vstats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    ) -> SidecarResult {
        SidecarResult {
            test_name: "t".to_string(),
            topology: "1s1c1t".to_string(),
            scheduler: "test".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vstats,
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        }
    }

    #[test]
    fn format_verifier_stats_empty() {
        assert!(format_verifier_stats(&[]).is_empty());
    }

    #[test]
    fn format_verifier_stats_no_data() {
        let sc = make_sidecar_with_vstats(vec![]);
        assert!(format_verifier_stats(&[sc]).is_empty());
    }

    #[test]
    fn format_verifier_stats_table() {
        let sc = make_sidecar_with_vstats(vec![
            crate::monitor::bpf_prog::ProgVerifierStats {
                name: "dispatch".to_string(),
                verified_insns: 50000,
            },
            crate::monitor::bpf_prog::ProgVerifierStats {
                name: "enqueue".to_string(),
                verified_insns: 30000,
            },
        ]);
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("BPF VERIFIER STATS"));
        assert!(result.contains("dispatch"));
        assert!(result.contains("enqueue"));
        assert!(result.contains("50000"));
        assert!(result.contains("30000"));
        assert!(result.contains("total verified insns: 80000"));
        assert!(!result.contains("WARNING"));
    }

    #[test]
    fn format_verifier_stats_warning() {
        let sc = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "heavy".to_string(),
            verified_insns: 800000,
        }]);
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("WARNING"));
        assert!(result.contains("heavy"));
        assert!(result.contains("80.0%"));
    }

    #[test]
    fn sidecar_verifier_stats_serde_roundtrip() {
        let sc = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "init".to_string(),
            verified_insns: 5000,
        }]);
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("verifier_stats"));
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verifier_stats.len(), 1);
        assert_eq!(loaded.verifier_stats[0].name, "init");
        assert_eq!(loaded.verifier_stats[0].verified_insns, 5000);
    }

    #[test]
    fn sidecar_verifier_stats_empty_omitted() {
        let sc = make_sidecar_with_vstats(vec![]);
        let json = serde_json::to_string(&sc).unwrap();
        assert!(!json.contains("verifier_stats"));
    }

    #[test]
    fn format_verifier_stats_deduplicates() {
        let sc1 = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 50000,
        }]);
        let sc2 = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 50000,
        }]);
        let result = format_verifier_stats(&[sc1, sc2]);
        // Deduplicated: total should be 50000, not 100000.
        assert!(result.contains("total verified insns: 50000"));
    }

    // -- diagnostic section tests --

    #[test]
    fn eval_sched_died_includes_console() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Monitor","message":"scheduler crashed after completing step 1 of 2 (0.5s into test)"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_died_console__");
        let result = make_vm_result(&output, "kernel panic\nsched_ext: disabled", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("--- diagnostics ---"), "got: {msg}");
        assert!(msg.contains("kernel panic"), "got: {msg}");
    }

    #[test]
    fn eval_sched_died_includes_monitor() {
        let json = r#"{"passed":false,"skipped":false,"details":[{"kind":"Monitor","message":"scheduler crashed during workload (2.0s into test)"}],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_died_monitor__");
        let result = crate::vmm::VmResult {
            success: false,
            exit_code: 1,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: output.to_string(),
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: vec![],
                summary: crate::monitor::MonitorSummary {
                    total_samples: 5,
                    max_imbalance_ratio: 3.0,
                    max_local_dsq_depth: 2,
                    stall_detected: false,
                    event_deltas: None,
                    schedstat_deltas: None,
                    prog_stats_deltas: None,
                    ..Default::default()
                },
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(msg.contains("--- monitor ---"), "got: {msg}");
        assert!(msg.contains("max_imbalance"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_includes_sched_log() {
        let pass_json = r#"{"passed":true,"skipped":false,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let sched_log =
            format!("{SCHED_OUTPUT_START}\nscheduler debug output here\n{SCHED_OUTPUT_END}",);
        let output = format!("{RESULT_START}\n{pass_json}\n{RESULT_END}\n{sched_log}");
        let entry = sched_entry("__eval_monitor_fail_sched__");
        // Imbalance ratio 10.0 exceeds default threshold of 4.0,
        // sustained for 5+ samples past the 20-sample warmup window.
        let imbalance_samples: Vec<crate::monitor::MonitorSample> = (0..30)
            .map(|i| {
                crate::monitor::MonitorSample::new(
                    (i * 100) as u64,
                    vec![
                        crate::monitor::CpuSnapshot {
                            nr_running: 10,
                            scx_nr_running: 10,
                            local_dsq_depth: 0,
                            rq_clock: 1000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                        crate::monitor::CpuSnapshot {
                            nr_running: 1,
                            scx_nr_running: 1,
                            local_dsq_depth: 0,
                            rq_clock: 2000 + (i as u64 * 100),
                            scx_flags: 0,
                            event_counters: None,
                            schedstat: None,
                            vcpu_cpu_time_ns: None,
                            sched_domains: None,
                        },
                    ],
                )
            })
            .collect();
        let summary =
            crate::monitor::MonitorSummary::from_samples_with_threshold(&imbalance_samples, 0);
        let result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output,
            stderr: String::new(),
            monitor: Some(crate::monitor::MonitorReport {
                samples: imbalance_samples,
                summary,
                preemption_threshold_ns: 0,
                watchdog_observation: None,
            }),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let assertions = crate::assert::Assert::default_checks();
        let msg = format!(
            "{}",
            evaluate_vm_result(
                &entry,
                &result,
                &assertions,
                &[],
                &EVAL_TOPO,
                &[],
                &no_repro
            )
            .unwrap_err()
        );
        assert!(
            msg.contains("passed scenario but monitor failed"),
            "got: {msg}"
        );
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }

    // -- find_symbol_vaddrs --

    #[test]
    fn find_symbol_vaddrs_resolves_known_symbol() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        // "main" is present in the symtab of any Rust test binary.
        let results = find_symbol_vaddrs(&data, &["main"]);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_some(),
            "main symbol should be resolved in test binary"
        );
        assert_ne!(results[0].unwrap(), 0, "main address should be nonzero");
    }

    #[test]
    fn find_symbol_vaddrs_missing_symbol_returns_none() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let results = find_symbol_vaddrs(&data, &["__nonexistent_symbol_xyz__"]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_none());
    }

    #[test]
    fn find_symbol_vaddrs_mixed_results() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let results = find_symbol_vaddrs(&data, &["main", "__nonexistent_symbol_xyz__"]);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some(), "main should resolve");
        assert!(results[1].is_none(), "nonexistent should not resolve");
    }

    // -- TopologyConstraints tests --

    #[test]
    fn topology_constraints_default_has_max_values() {
        let c = TopologyConstraints::DEFAULT;
        assert_eq!(c.max_llcs, Some(12));
        assert_eq!(c.max_numa_nodes, Some(1));
        assert_eq!(c.max_cpus, Some(192));
    }

    #[test]
    fn topology_constraints_max_fields_set() {
        let c = TopologyConstraints {
            max_llcs: Some(16),
            max_numa_nodes: Some(4),
            max_cpus: Some(128),
            ..TopologyConstraints::DEFAULT
        };
        assert_eq!(c.max_llcs, Some(16));
        assert_eq!(c.max_numa_nodes, Some(4));
        assert_eq!(c.max_cpus, Some(128));
        assert_eq!(c.min_numa_nodes, 1);
        assert_eq!(c.min_llcs, 1);
        assert_eq!(c.min_cpus, 1);
    }

    #[test]
    fn topology_constraints_equality() {
        let a = TopologyConstraints::DEFAULT;
        let b = TopologyConstraints::DEFAULT;
        assert_eq!(a, b);

        let c = TopologyConstraints {
            max_llcs: Some(8),
            ..TopologyConstraints::DEFAULT
        };
        assert_ne!(a, c);
    }

    #[test]
    fn accepts_default_allows_within_limits() {
        let c = TopologyConstraints::DEFAULT;
        // 1 NUMA, 8 LLCs, 4 cores, 2 threads = 64 CPUs
        let t = Topology::new(1, 8, 4, 2);
        assert!(c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_default_rejects_multi_numa() {
        let c = TopologyConstraints::DEFAULT;
        // 2 NUMA, 8 LLCs, 4 cores, 2 threads = 64 CPUs
        let t = Topology::new(2, 8, 4, 2);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_default_rejects_too_many_llcs() {
        let c = TopologyConstraints::DEFAULT;
        // 16 LLCs exceeds max_llcs=12
        let t = Topology::new(1, 16, 2, 1);
        assert!(!c.accepts(&t, 128, 32, 32));
    }

    #[test]
    fn accepts_none_means_no_limit() {
        let c = TopologyConstraints {
            max_llcs: None,
            max_numa_nodes: None,
            max_cpus: None,
            ..TopologyConstraints::DEFAULT
        };
        // 4 NUMA, 16 LLCs, 8 cores, 2 threads = 256 CPUs
        let t = Topology::new(4, 16, 8, 2);
        assert!(c.accepts(&t, 512, 32, 32));
    }

    #[test]
    fn accepts_rejects_too_many_llcs() {
        let c = TopologyConstraints {
            max_llcs: Some(4),
            ..TopologyConstraints::DEFAULT
        };
        let t = Topology::new(1, 8, 2, 1);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_allows_llcs_at_max() {
        let c = TopologyConstraints {
            max_llcs: Some(4),
            ..TopologyConstraints::DEFAULT
        };
        let t = Topology::new(1, 4, 2, 1);
        assert!(c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_too_many_numa_nodes() {
        let c = TopologyConstraints {
            max_numa_nodes: Some(2),
            ..TopologyConstraints::DEFAULT
        };
        let t = Topology::new(4, 4, 2, 1);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_allows_numa_at_max() {
        let c = TopologyConstraints {
            max_numa_nodes: Some(2),
            ..TopologyConstraints::DEFAULT
        };
        let t = Topology::new(2, 4, 2, 1);
        assert!(c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_too_many_cpus() {
        let c = TopologyConstraints {
            max_cpus: Some(16),
            ..TopologyConstraints::DEFAULT
        };
        // 4 LLCs * 4 cores * 2 threads = 32 CPUs
        let t = Topology::new(1, 4, 4, 2);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_allows_cpus_at_max() {
        let c = TopologyConstraints {
            max_cpus: Some(16),
            ..TopologyConstraints::DEFAULT
        };
        // 2 LLCs * 4 cores * 2 threads = 16 CPUs
        let t = Topology::new(1, 2, 4, 2);
        assert!(c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_too_few_llcs() {
        let c = TopologyConstraints {
            min_llcs: 4,
            ..TopologyConstraints::DEFAULT
        };
        let t = Topology::new(1, 2, 4, 1);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_exceeding_host_cpus() {
        let c = TopologyConstraints::DEFAULT;
        let t = Topology::new(1, 4, 4, 2); // 32 CPUs
        assert!(!c.accepts(&t, 16, 16, 32)); // host has only 16
    }

    #[test]
    fn accepts_rejects_exceeding_host_llcs() {
        let c = TopologyConstraints::DEFAULT;
        let t = Topology::new(1, 8, 2, 1);
        assert!(!c.accepts(&t, 128, 4, 32)); // host has only 4 LLCs
    }

    #[test]
    fn accepts_combined_min_and_max() {
        let c = TopologyConstraints {
            min_llcs: 2,
            max_llcs: Some(8),
            min_cpus: 4,
            max_cpus: Some(32),
            ..TopologyConstraints::DEFAULT
        };
        // 1 LLC, 4 CPUs -- rejected (min_llcs=2)
        assert!(!c.accepts(&Topology::new(1, 1, 4, 1), 128, 16, 32));
        // 2 LLCs, 4 CPUs -- accepted
        assert!(c.accepts(&Topology::new(1, 2, 2, 1), 128, 16, 32));
        // 16 LLCs, 32 CPUs -- rejected (max_llcs=8)
        assert!(!c.accepts(&Topology::new(1, 16, 2, 1), 128, 16, 32));
        // 8 LLCs, 16 CPUs -- accepted
        assert!(c.accepts(&Topology::new(1, 8, 2, 1), 128, 16, 32));
    }

    #[test]
    fn accepts_requires_smt() {
        let c = TopologyConstraints {
            requires_smt: true,
            ..TopologyConstraints::DEFAULT
        };
        let no_smt = Topology::new(1, 2, 4, 1);
        let with_smt = Topology::new(1, 2, 4, 2);
        assert!(!c.accepts(&no_smt, 128, 16, 32));
        assert!(c.accepts(&with_smt, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_too_few_numa_nodes() {
        let c = TopologyConstraints {
            min_numa_nodes: 2,
            max_numa_nodes: None,
            ..TopologyConstraints::DEFAULT
        };
        let t = Topology::new(1, 4, 4, 1);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_too_few_cpus() {
        let c = TopologyConstraints {
            min_cpus: 32,
            ..TopologyConstraints::DEFAULT
        };
        // 2 LLCs * 4 cores * 2 threads = 16 CPUs
        let t = Topology::new(1, 2, 4, 2);
        assert!(!c.accepts(&t, 128, 16, 32));
    }

    #[test]
    fn accepts_rejects_exceeding_host_cpus_per_llc() {
        let c = TopologyConstraints::DEFAULT;
        // cores_per_llc=8, threads_per_core=2 → 16 CPUs/LLC
        let t = Topology::new(1, 2, 8, 2);
        assert!(!c.accepts(&t, 128, 16, 8));
    }

    // -- validate_entry_flags panic paths --

    #[test]
    #[should_panic(expected = "unknown required_flag")]
    fn validate_entry_flags_unknown_required() {
        static SCHED: Scheduler = Scheduler::new("sched").flags(FLAGS_AB);
        let entry = KtstrTestEntry {
            name: "bad_required",
            scheduler: &SCHED,
            required_flags: &["nonexistent"],
            ..KtstrTestEntry::DEFAULT
        };
        validate_entry_flags(&entry);
    }

    #[test]
    #[should_panic(expected = "in both required_flags and excluded_flags")]
    fn validate_entry_flags_both_required_and_excluded() {
        static SCHED: Scheduler = Scheduler::new("sched").flags(FLAGS_AB);
        let entry = KtstrTestEntry {
            name: "bad_both",
            scheduler: &SCHED,
            required_flags: &["a"],
            excluded_flags: &["a"],
            ..KtstrTestEntry::DEFAULT
        };
        validate_entry_flags(&entry);
    }

    #[test]
    fn config_file_parts_nested_path() {
        static SCHED: Scheduler = Scheduler::new("cfg").config_file("configs/my_sched.toml");
        let entry = KtstrTestEntry {
            name: "cfg_test",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (archive, host, guest) = config_file_parts(&entry).unwrap();
        assert_eq!(archive, "include-files/my_sched.toml");
        assert_eq!(host, PathBuf::from("configs/my_sched.toml"));
        assert_eq!(guest, "/include-files/my_sched.toml");
    }

    #[test]
    fn config_file_parts_bare_filename() {
        static SCHED: Scheduler = Scheduler::new("cfg").config_file("config.toml");
        let entry = KtstrTestEntry {
            name: "cfg_bare",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (archive, host, guest) = config_file_parts(&entry).unwrap();
        assert_eq!(archive, "include-files/config.toml");
        assert_eq!(host, PathBuf::from("config.toml"));
        assert_eq!(guest, "/include-files/config.toml");
    }

    #[test]
    fn config_file_parts_none_when_unset() {
        let entry = KtstrTestEntry {
            name: "no_cfg",
            ..KtstrTestEntry::DEFAULT
        };
        assert!(config_file_parts(&entry).is_none());
    }

    // -- now_iso8601 / days_to_ymd / is_leap tests --

    #[test]
    fn days_to_ymd_epoch() {
        let (y, m, d) = days_to_ymd(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_date() {
        let (y, m, d) = days_to_ymd(18628);
        assert_eq!((y, m, d), (2021, 1, 1));
    }

    #[test]
    fn days_to_ymd_leap_day() {
        let (y, m, d) = days_to_ymd(11016);
        assert_eq!((y, m, d), (2000, 2, 29));
    }

    #[test]
    fn is_leap_years() {
        assert!(is_leap(2000));
        assert!(is_leap(2024));
        assert!(!is_leap(1900));
        assert!(!is_leap(2023));
    }

    #[test]
    fn now_iso8601_format() {
        let ts = now_iso8601();
        assert!(ts.ends_with('Z'));
        assert!(ts.contains('T'));
        assert_eq!(ts.len(), 20);
    }

    // -- generate_run_id tests --

    #[test]
    fn generate_run_id_contains_hash() {
        let id = generate_run_id();
        assert!(id.contains(crate::GIT_HASH));
    }

    #[test]
    fn generate_run_id_monotonic() {
        let id1 = generate_run_id();
        let id2 = generate_run_id();
        assert_ne!(id1, id2);
    }
}
