//! Runtime support for `#[ktstr_test]` integration tests.
//!
//! Provides the registration type, distributed slice, VM launcher,
//! and result evaluation. Includes guest-side profraw flush for
//! coverage-instrumented builds.
//!
//! The entry point for test authors is the [`macro@crate::ktstr_test`]
//! attribute macro; see the user-facing Writing Tests guide shipped
//! with the crate's mdbook for end-to-end examples and the full
//! attribute grammar.
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
//! Implementation is split across 14 private submodules re-exported
//! at `test_support::*` for a flat public API: `args` (CLI argument
//! extraction), `dispatch` (ktstr / cargo-ktstr CLI entry points),
//! `entry` (scheduler + test-entry types), `eval` (host-side VM
//! result evaluation), `metrics` (payload stdout → `Metric` list),
//! `model` (LLM backend + model cache), `output` (guest-output and
//! console parsing), `payload` (`Payload` / `Check` / `Metric` /
//! `Polarity`), `probe` (auto-repro and BPF probe pipeline),
//! `profraw` (coverage flush), `runtime` (neutral home for
//! verbose/shm-size/config-file-parts shared by eval and probe so
//! they don't circularly depend on each other), `sidecar` (per-run
//! JSON records), `timefmt` (ISO-8601 + run-id helpers), and `topo`
//! (topology override parsing).

#[cfg(test)]
use crate::assert::AssertResult;
#[cfg(test)]
use crate::scenario::Ctx;
#[cfg(test)]
use anyhow::Result;

pub use crate::scenario::flags::FlagDecl;

mod args;
mod dispatch;
mod entry;
mod eval;
mod metrics;
mod model;
mod output;
mod payload;
mod probe;
mod probe_metrics;
mod profraw;
mod runtime;
mod sidecar;
#[cfg(test)]
pub(crate) mod test_helpers;
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
pub use sidecar::{SidecarResult, newest_run_dir, runs_root};
pub(crate) use sidecar::{
    collect_sidecars, format_callback_profile, format_kvm_stats, format_verifier_stats, sidecar_dir,
};

pub use dispatch::{analyze_sidecars, ktstr_main, ktstr_test_early_dispatch, run_ktstr_test};
pub(crate) use entry::validate_entry_flags;
pub use entry::{
    BpfMapWrite, CgroupPath, KTSTR_TESTS, KtstrTestEntry, MemSideCache, NumaDistance, NumaNode,
    Scheduler, SchedulerSpec, Sysctl, Topology, TopologyConstraints, find_test,
};
pub use eval::{ResolveSource, resolve_scheduler, resolve_test_kernel};
pub(crate) use eval::{record_skip_sidecar, run_ktstr_test_inner};
pub use metrics::{
    MAX_WALK_DEPTH, WALK_TRUNCATION_SENTINEL_NAME, extract_metrics, is_truncation_sentinel_name,
    walk_json_leaves,
};
pub use model::{
    DEFAULT_MODEL, LLM_DEBUG_RESPONSES_ENV, ModelSpec, ModelStatus, OFFLINE_ENV, ShaVerdict,
    ensure, status,
};
pub(crate) use output::{
    SENTINEL_EXEC_EXIT_PREFIX, SENTINEL_EXIT_PREFIX, SENTINEL_INIT_STARTED,
    SENTINEL_PAYLOAD_STARTING, SENTINEL_SCHEDULER_DIED, SENTINEL_SCHEDULER_NOT_ATTACHED,
};
pub use payload::{
    Check, Metric, MetricBounds, MetricHint, MetricSource, MetricStream, OutputFormat, Payload,
    PayloadKind, PayloadMetrics, Polarity,
};
pub(crate) use payload::{RawPayloadOutput, WireMetricHint};
pub(crate) use probe::maybe_dispatch_vm_test;
pub(crate) use probe::{
    PipelineDiagnostics, format_probe_diagnostics, maybe_dispatch_vm_test_with_args,
    maybe_dispatch_vm_test_with_phase_a, propagate_rust_env_from_cmdline, start_probe_phase_a,
};
pub use probe_metrics::{
    MAX_SCAN_INDEX, ThreadLookup, count_indexed_metrics, find_metric, find_metric_u64,
    flat_metrics_dump, has_metric, lookup_thread, snapshot_count, snapshot_worker_allocated,
    thread_count,
};
pub(crate) use profraw::try_flush_profraw;
pub(crate) use timefmt::now_iso8601;
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
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn require_kernel() -> std::path::PathBuf {
    match crate::find_kernel() {
        Ok(Some(p)) => p,
        Ok(None) => panic!(
            "ktstr_test: test requires a kernel but none was found. {}",
            crate::KTSTR_KERNEL_HINT
        ),
        Err(e) => panic!("ktstr_test: kernel resolution failed: {e:#}"),
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
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn require_vmlinux(kernel_path: &std::path::Path) -> std::path::PathBuf {
    crate::vmm::find_vmlinux(kernel_path).unwrap_or_else(|| {
        panic!(
            "ktstr_test: no vmlinux found alongside {}. The cache entry or \
             kernel build is incomplete. Rebuild with `cargo ktstr kernel \
             build --force`; the specified kernel must include `vmlinux` \
             alongside the boot image. {}",
            kernel_path.display(),
            crate::KTSTR_KERNEL_HINT,
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
            "ktstr_test: build of `{package}` failed: {e:#}. \
             Run `cargo build -p {package}` to reproduce and diagnose."
        )
    })
}

/// Resolve [`crate::monitor::symbols::KernelSymbols`] from a vmlinux
/// or panic. The symbol table is required for any host-side memory
/// introspection; an unparseable vmlinux is an infrastructure failure.
#[cfg(test)]
#[allow(dead_code)] // called from x86_64-only tests in vmm/mod.rs
pub(crate) fn require_kernel_symbols(
    vmlinux_path: &std::path::Path,
) -> crate::monitor::symbols::KernelSymbols {
    crate::monitor::symbols::KernelSymbols::from_vmlinux(vmlinux_path).unwrap_or_else(|e| {
        panic!(
            "ktstr_test: kernel symbol resolution from {} failed: {e:#}",
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
            "ktstr_test: kernel BTF resolution from {} failed: {e:#}. \
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
            "ktstr_test: BpfMapOffsets resolution from {} failed: {e:#}. \
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
            "ktstr_test: BpfProgOffsets resolution from {} failed: {e:#}. \
             The kernel must be built with CONFIG_DEBUG_INFO_BTF=y; \
             rebuild with `cargo ktstr kernel build --force` if the \
             cache entry was produced without BTF.",
            vmlinux_path.display(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use linkme::distributed_slice;

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
    fn find_test_from_distributed_slice() {
        // KTSTR_TESTS should contain at least the __unit_test_dummy__ entry.
        assert!(!KTSTR_TESTS.is_empty());
    }
}
