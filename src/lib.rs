//! VM-based test framework for Linux kernel subsystems, with a focus on sched_ext.
//!
//! ktstr boots lightweight KVM virtual machines with controlled CPU topologies,
//! runs scheduler test scenarios inside them, and evaluates results from the
//! host via guest memory introspection. Each test creates cgroups, spawns
//! worker processes, and checks that the scheduler handled the workload
//! correctly. Also tests under the kernel's default EEVDF scheduler.
//!
//! # Quick start
//!
//! Declare cgroups and workloads as data, let the framework handle
//! lifecycle and checking:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
//! fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
//!     execute_defs(ctx, vec![
//!         CgroupDef::named("cg_0").workers(2),
//!         CgroupDef::named("cg_1").workers(2),
//!     ])
//! }
//! ```
//!
//! Requires a kernel image; see [`find_kernel()`] for the resolution chain.
//!
//! For multi-phase scenarios with dynamic topology changes:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
//! fn my_dynamic_test(ctx: &Ctx) -> Result<AssertResult> {
//!     let steps = vec![
//!         Step::with_defs(
//!             vec![CgroupDef::named("cg_0").workers(4)],
//!             HoldSpec::Frac(0.5),
//!         ),
//!         Step::new(
//!             vec![Op::stop_cgroup("cg_0"), Op::remove_cgroup("cg_0")],
//!             HoldSpec::Frac(0.5),
//!         ),
//!     ];
//!     execute_steps(ctx, steps)
//! }
//! ```
//!
//! # Scheduler definition
//!
//! Tests work with just topology parameters (as above). When multiple
//! tests share a scheduler, use `#[derive(Scheduler)]` to declare it
//! once with typed flags and a default topology. Tests reference the
//! generated const and inherit its configuration:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[derive(Scheduler)]
//! #[scheduler(name = "my_sched", binary = "scx_my_sched", topology(1, 2, 4, 1))]
//! #[allow(dead_code)]
//! enum MySchedFlag {
//!     #[flag(args = ["--enable-llc"])]
//!     Llc,
//!     #[flag(args = ["--enable-stealing"], requires = [Llc])]
//!     Steal,
//! }
//!
//! #[ktstr_test(scheduler = MY_SCHED_PAYLOAD)]
//! fn basic(ctx: &Ctx) -> Result<AssertResult> {
//!     execute_defs(ctx, vec![
//!         CgroupDef::named("cg_0").workers(2),
//!         CgroupDef::named("cg_1").workers(2),
//!     ])
//! }
//! ```
//!
//! For full control over cgroup setup, worker spawning, and assertion
//! you can use the low-level API directly:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
//! fn my_low_level_test(ctx: &Ctx) -> Result<AssertResult> {
//!     let mut group = CgroupGroup::new(ctx.cgroups);
//!     group.add_cgroup_no_cpuset("workers")?;
//!     let cpus = ctx.topo.all_cpuset();
//!     ctx.cgroups.set_cpuset("workers", &cpus)?;
//!
//!     let cfg = WorkloadConfig {
//!         num_workers: 2,
//!         work_type: WorkType::SpinWait,
//!         ..Default::default()
//!     };
//!     let mut handle = WorkloadHandle::spawn(&cfg)?;
//!     ctx.cgroups.move_tasks("workers", &handle.worker_pids_for_cgroup_procs()?)?;
//!     handle.start();
//!
//!     std::thread::sleep(ctx.duration);
//!     let reports = handle.stop_and_collect();
//!
//!     let a = Assert::default_checks();
//!     Ok(a.assert_cgroup(&reports, None))
//! }
//! ```
//!
//! For pointwise assertions against captured stats — the most direct
//! way to express "this counter is at least N", "this rate is between
//! A and B", "this metric is finite" — use [`Verdict`] +
//! `#[derive(Claim)]` accessors and the [`claim!`] macro:
//!
//! ```rust
//! use ktstr::prelude::*;
//! use ktstr::workload::WorkerReport;
//! use std::collections::{BTreeMap, BTreeSet};
//!
//! // A test author would obtain `cg` and `report` from `ctx`-driven
//! // execution; the literal here just illustrates the assertion shape.
//! let cg = CgroupStats {
//!     num_workers: 2,
//!     num_cpus: 2,
//!     max_gap_ms: 50,
//!     p99_wake_latency_us: 25.0,
//!     median_wake_latency_us: 10.0,
//!     total_iterations: 5_000,
//!     ..Default::default()
//! };
//! let work_units = 10_000u64;
//! let throughput = work_units as f64 / 5.0;
//!
//! let mut v = Assert::defaults().verdict();
//! cg.claim_max_gap_ms(&mut v).at_most(100);          // typed CgroupStats accessor
//! cg.claim_p99_wake_latency_us(&mut v).at_most(50.0);
//! cg.claim_total_iterations(&mut v).at_least(1_000);
//! claim!(v, work_units).at_least(5_000);             // local-binding label
//! claim!(v, throughput).is_finite();                  // expression label
//! claim!(v, cg.wake_latency_tail_ratio()).between(1.0, 5.0);
//! let r = v.into_result();
//! assert!(r.passed);
//! ```
//!
//! Every claim is labeled by `stringify!` on either a struct field name
//! (via the derive) or an identifier/expression (via the macro), so a
//! rename or refactor updates the failure-message label automatically
//! and a stale call site fails to compile. There is no manual-string
//! escape hatch — by design, every label is source-text-grounded.
//!
//! Run with `cargo nextest run` (requires `/dev/kvm`).
//!
//! See the [`prelude`] module for the full set of re-exports.
//!
//! # Library usage
//!
//! ```toml
//! [dev-dependencies]
//! ktstr = { version = "0.4" }
//! ```
//!
//! The only feature flag is `integration`, which gates
//! `resolve_func_ip` visibility for integration tests.
//!
//! # Crate organization
//!
//! - [`cache`] -- kernel image cache (XDG directories, metadata, atomic writes)
//! - [`cgroup`] -- cgroup v2 filesystem operations
//! - [`cli`] -- shared helpers backing the `ktstr` and `cargo-ktstr` binaries
//! - [`fetch`] -- kernel tarball and git source acquisition
//! - [`flock`] -- advisory file-locking primitives used by cache + LLC reservations
//! - [`kernel_path`] -- kernel ID parsing and filesystem image discovery
//! - [`remote_cache`] -- GitHub Actions cache integration
//! - [`scenario`] -- declarative ops API (`CgroupDef`, `Step`, `Op`, `Backdrop`, `execute_defs`, `execute_steps`, `execute_scenario`)
//! - [`scenario::scenarios`] -- curated canned scenarios for common patterns
//! - [`mod@assert`] -- pass/fail assertions (starvation, isolation, fairness)
//! - [`test_support`] -- `#[ktstr_test]` runtime and registration
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`verifier`] -- BPF verifier log parsing, cycle detection, and output formatting
//! - [`worker_ready`] / [`worker_ready_wait`] -- pid-scoped marker file the alloc/test workers write before the parent samples them
//! - [`workload`] -- worker process types and telemetry collection
//!
//! ## ctprof subsystem
//!
//! Per-thread + per-process runtime profile, captured via
//! `ktstr ctprof capture` and compared via
//! `ktstr ctprof compare`:
//!
//! - [`host_context`] -- one-shot host snapshot (kernel, CPU, memory, tunables)
//! - [`host_heap`] -- jemalloc global heap counters (mallctl)
//! - [`ctprof`] -- per-thread procfs walk + cumulative scheduling, I/O, page-fault, jemalloc TSD counters
//! - [`ctprof_compare`] -- two-snapshot diff engine (group-by + delta tables)
//!
//! `host_thread_probe` (the ELF/DWARF + ptrace + `process_vm_readv`
//! engine that pulls per-thread jemalloc TSD counters) is
//! `pub(crate)`-only and consumed exclusively by `ctprof` plus
//! the source-shared standalone `ktstr-jemalloc-probe` binary.
//! Direct probe access from downstream is intentionally not part
//! of the surface — scheduler authors get the captured counters
//! through `ctprof::ThreadState`.
//!
//! Internal modules (not re-exported): `host_thread_probe` reads
//! per-thread jemalloc TSD counters via ptrace, `monitor` reads
//! live guest state, `probe` attaches BPF probes to traced
//! functions, `vmm` owns the KVM VM lifecycle, and `timeline`
//! correlates stimulus events with monitor samples for
//! phase-aligned reporting.

// `#[derive(Payload)]` and `#[derive(Scheduler)]` expand into
// `::ktstr::test_support::...` paths so downstream crates can use
// them without a `use` import. This alias lets the same derives be
// used inside the ktstr crate itself — for example by doctests and
// by integration-test modules under `tests/common/` that pull the
// derive through the same public path downstream authors take. No
// runtime cost: `extern crate self as ktstr` is a pure name-binding.
extern crate self as ktstr;

#[allow(
    clippy::all,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod bpf_skel;

#[cfg(test)]
#[macro_use]
mod test_macros;

/// Shared guidance for every `#[non_exhaustive]` type in this
/// crate. Individual types link here instead of repeating the
/// same migration rules in every doc block.
///
/// # `#[non_exhaustive]` conventions in ktstr
///
/// Most of ktstr's public structs and enums carry `#[non_exhaustive]`
/// so that adding a field or variant is not a breaking change for
/// downstream crates. The attribute has two consequences downstream
/// consumers must account for:
///
/// ## Pattern matching
///
/// Matches on a `#[non_exhaustive]` struct or enum from outside this
/// crate must end with a wildcard `..` (for structs) or `_ =>` arm
/// (for enums). Without it, a future addition to the type forces
/// every matcher into a compile break even when the new field or
/// variant is irrelevant to the caller.
///
/// ```ignore
/// // Good: `..` absorbs future fields.
/// if let MyStruct { name, .. } = value { /* ... */ }
/// match my_enum {
///     MyEnum::A => {}
///     MyEnum::B => {}
///     _ => {}          // absorbs future variants
/// }
/// ```
///
/// ## Construction
///
/// Cross-crate consumers **cannot** use any struct-expression form
/// for a `#[non_exhaustive]` struct — bare literals
/// (`MyStruct { name: "x", .. }`) and functional-update spreads
/// (`MyStruct { name: "x", ..Default::default() }`) are both
/// rejected by the compiler (E0639). Construction must go through
/// one of:
///
/// 1. A dedicated constructor (`MyStruct::new(...)`,
///    `MyStruct::from_*(...)`) exposed by the defining crate.
/// 2. A [`Default`] instance followed by field mutation, when the
///    type derives `Default`.
/// 3. A named `test_fixture` or equivalent associated function for
///    types that expose a populated baseline instead of the
///    all-default minimum.
///
/// The per-type doc picks whichever of these the type actually
/// supports; see [`host_context::HostContext`],
/// [`host_heap::HostHeapState`], and the Op/CpusetSpec docs in
/// [`scenario::ops`] for worked examples across the different
/// shapes.
///
/// ## Pattern matching inside this crate
///
/// `#[non_exhaustive]` is enforced only across crate boundaries.
/// In-crate matchers can remain exhaustive (and should, so the
/// compiler flags forgotten variants at the definition site), and
/// in-crate struct-literal construction still works for the tests
/// and fixtures that live alongside the type.
#[doc(hidden)]
pub mod non_exhaustive {}

pub mod cache;
pub mod cgroup;
pub mod flock;

/// Map a raw errno value to its C constant name.
///
/// Returns `None` for unrecognized values. [`nix::errno::Errno`] has
/// `#[derive(Debug)]`, but `format!("{:?}", e)` allocates a fresh
/// `String` on every call; the hand-rolled match below returns a
/// `&'static str` pointing at a literal instead. [`nix::errno::Errno`]
/// is used here to gate unknown errnos via
/// `matches!(e, UnknownErrno)`. Adding a new errno means extending
/// both nix's port-constants table (for the UnknownErrno gate) and
/// this match; the test suite pins a representative subset so a
/// stale arm surfaces at build time.
pub(crate) fn errno_name(errno: i32) -> Option<&'static str> {
    let e = nix::errno::Errno::from_raw(errno);
    if matches!(e, nix::errno::Errno::UnknownErrno) {
        return None;
    }
    // Hand-rolled match: returns a `&'static str` pointing at a
    // literal, avoiding the allocation that `format!("{:?}", e)` would
    // incur. Callers that compare these against string literals in
    // error formatting paths rely on the stable symbolic names below.
    Some(match e {
        nix::errno::Errno::EPERM => "EPERM",
        nix::errno::Errno::ENOENT => "ENOENT",
        nix::errno::Errno::ESRCH => "ESRCH",
        nix::errno::Errno::EINTR => "EINTR",
        nix::errno::Errno::EIO => "EIO",
        nix::errno::Errno::ENXIO => "ENXIO",
        nix::errno::Errno::E2BIG => "E2BIG",
        nix::errno::Errno::ENOEXEC => "ENOEXEC",
        nix::errno::Errno::EBADF => "EBADF",
        nix::errno::Errno::ECHILD => "ECHILD",
        nix::errno::Errno::EAGAIN => "EAGAIN",
        nix::errno::Errno::ENOMEM => "ENOMEM",
        nix::errno::Errno::EACCES => "EACCES",
        nix::errno::Errno::EFAULT => "EFAULT",
        nix::errno::Errno::EBUSY => "EBUSY",
        nix::errno::Errno::EEXIST => "EEXIST",
        nix::errno::Errno::ENODEV => "ENODEV",
        nix::errno::Errno::ENOTDIR => "ENOTDIR",
        nix::errno::Errno::EISDIR => "EISDIR",
        nix::errno::Errno::EINVAL => "EINVAL",
        nix::errno::Errno::ENFILE => "ENFILE",
        nix::errno::Errno::EMFILE => "EMFILE",
        nix::errno::Errno::ENOSPC => "ENOSPC",
        nix::errno::Errno::ESPIPE => "ESPIPE",
        nix::errno::Errno::EROFS => "EROFS",
        nix::errno::Errno::EPIPE => "EPIPE",
        nix::errno::Errno::EDOM => "EDOM",
        nix::errno::Errno::ERANGE => "ERANGE",
        nix::errno::Errno::EDEADLK => "EDEADLK",
        nix::errno::Errno::ENAMETOOLONG => "ENAMETOOLONG",
        nix::errno::Errno::ENOSYS => "ENOSYS",
        nix::errno::Errno::ENOTEMPTY => "ENOTEMPTY",
        nix::errno::Errno::ELOOP => "ELOOP",
        nix::errno::Errno::ENOTSUP => "ENOTSUP",
        nix::errno::Errno::EADDRINUSE => "EADDRINUSE",
        nix::errno::Errno::ECONNREFUSED => "ECONNREFUSED",
        nix::errno::Errno::ETIMEDOUT => "ETIMEDOUT",
        // Other well-defined constants exist on nix::errno::Errno
        // but were not in the previous curated list. Return None for
        // them to preserve the prior contract — callers that want
        // more coverage can extend this match explicitly.
        _ => return None,
    })
}

/// Read the kernel ring buffer (equivalent to `dmesg --notime`).
/// Exposed as `pub` so scenario tests that need to assert on
/// kernel-log content (e.g. the sched_ext stall duration emitted
/// by `scx_exit(SCX_EXIT_ERROR_STALL)` in `kernel/sched/ext.c`)
/// can read the same buffer the framework captures into
/// `AssertResult::details` on scheduler-died failures.
pub fn read_kmsg() -> String {
    match rmesg::log_entries(rmesg::Backend::Default, false) {
        Ok(entries) => entries
            .iter()
            .map(|e| e.message.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        Err(_) => String::new(),
    }
}

pub mod assert;
pub(crate) mod budget;
pub mod cli;
pub mod cpu_util;
pub mod ctprof;
pub mod ctprof_compare;
pub(crate) mod elf_strip;
pub mod export;
pub mod fetch;
pub mod fun;
pub mod host_context;
pub mod host_heap;
pub(crate) mod host_thread_probe;
pub mod kernel_path;
pub mod metric_types;
pub(crate) mod monitor;
pub(crate) mod probe;
pub(crate) mod report;
pub mod scenario;
pub(crate) mod stats;
pub(crate) mod taskstats;
pub mod test_support;
pub(crate) mod timeline;
pub mod topology;

/// Public surface for the live-host introspection pipeline.
///
/// Re-exports from the otherwise-internal `monitor` module so the
/// live-host capture binary, integration tests, and downstream
/// consumers can invoke the bpf()-syscall data path, kernel
/// auto-discovery, kallsyms parser, dmesg-scx parser, and the
/// reproducer-generator translation layer without the `monitor`
/// module's frozen-VM internals leaking into the public API.
///
/// This module is the entry point for binaries and tests that
/// consume the live-host capture pipeline.
pub mod live_host {
    pub use crate::monitor::bpf_map::{
        BpfMapAccessor, BpfMapInfo, BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH,
        BPF_MAP_TYPE_PERCPU_ARRAY,
    };
    pub use crate::monitor::bpf_syscall::BpfSyscallAccessor;
    pub use crate::monitor::debug_capture::{
        AffinityHint, CgroupHint, CtprofSampleRef, DebugCapture, SchedPolicyHint,
        WorkTypeHint, WorkloadFingerprint, WorkloadGroupHint, project_fingerprint,
        DEBUG_CAPTURE_SCHEMA,
    };
    pub use crate::monitor::dmesg_scx::{
        ScxExitEvent, ScxExitKind, StackSymbol, extract_stack_symbols, parse_kmsg_window,
    };
    pub use crate::monitor::live_host_kernel::{
        KallsymsTable, LiveHostKernelEnv, uname_release,
    };
    pub use crate::monitor::reproducer_gen::{
        ReproducerSpec, generate_spec, render_ktstr_test_source, render_run_file_source,
    };
    pub use crate::monitor::timeline::{
        DEFAULT_SNAPSHOT_RING_DEPTH, IncrementalCapture, IncrementalSnapshot, SnapshotRing,
        TimelineCapture, TimelineEvent, TimelineEventRaw, parse_timeline_buf,
        parse_timeline_record, tl_evt,
    };
}

pub mod remote_cache;
pub(crate) mod sync;
pub(crate) mod tar_util;
pub mod verifier;
pub(crate) mod vm;
pub(crate) mod vmm;
pub mod worker_ready;
pub mod worker_ready_wait;
pub mod workload;

/// Static busybox binary compiled in build.rs for guest shell mode.
pub(crate) const BUSYBOX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

/// Contents of `ktstr.kconfig` (the kernel-config fragment that
/// enables sched_ext, BPF, kprobes, cgroups, and the other options
/// ktstr requires) baked into the binary at build time via
/// `include_str!`. Consumed by the kernel build pipeline to
/// `olddefconfig` a kernel source tree, and used to derive the
/// cache key suffix so a kconfig change produces a fresh cache
/// entry.
pub const EMBEDDED_KCONFIG: &str = include_str!("../ktstr.kconfig");

/// CRC32 hash of the embedded kconfig fragment (8 hex chars).
pub fn kconfig_hash() -> String {
    format!("{:08x}", crc32fast::hash(EMBEDDED_KCONFIG.as_bytes()))
}

/// CRC32 hash (8 hex chars) of a user-supplied `--extra-kconfig`
/// fragment, hashed verbatim.
///
/// Hashes raw bytes — no comment stripping, no CRLF
/// canonicalization. Two semantically-equivalent inputs with
/// different comments or line endings produce different hashes and
/// therefore land at distinct cache entries — accept the disk waste
/// in exchange for byte-deterministic cache discrimination.
pub fn extra_kconfig_hash(extra: &str) -> String {
    format!("{:08x}", crc32fast::hash(extra.as_bytes()))
}

/// Cache key suffix derived from the embedded kconfig fragment.
/// Used in kernel cache keys so a kconfig change produces a distinct
/// cache entry. The kernel binary is independent of ktstr userspace
/// source, so no ktstr or consumer build identity feeds this suffix.
pub fn cache_key_suffix() -> String {
    kconfig_hash()
}

/// Two-segment cache key suffix accounting for an optional
/// `--extra-kconfig` fragment.
///
/// The suffix uses TWO segments instead of folding both inputs into
/// one hash:
///
/// - `extra = None` → `kconfig_hash()` only — byte-identical to
///   [`cache_key_suffix`], so paths that don't expose
///   `--extra-kconfig` (test / coverage / shell / verifier) keep
///   resolving the existing keyspace and pre-1.0 cached kernels are
///   not orphaned.
/// - `extra = Some(content)` → `{kconfig_hash()}-xkc{extra_hash}`,
///   making `kernel list` self-describing: a reader can see at a
///   glance which entries carry user extras and which are pure
///   baked-in builds. Different extra content yields different
///   `xkc{...}` segments, so cache discrimination across distinct
///   `--extra-kconfig` invocations is structural rather than
///   collapsed into a single opaque hash.
pub fn cache_key_suffix_with_extra(extra: Option<&str>) -> String {
    match extra {
        None => kconfig_hash(),
        Some(content) => format!("{}-xkc{}", kconfig_hash(), extra_kconfig_hash(content)),
    }
}

/// Merge the user-supplied `--extra-kconfig` fragment on top of
/// [`EMBEDDED_KCONFIG`] for the configure pass. Returns a
/// [`std::borrow::Cow`] so the no-extras branch borrows `baked`
/// without allocating; only the `Some` branch heaps the merged
/// String.
///
/// The user fragment is appended AFTER the baked-in fragment so
/// kbuild's last-wins rule
/// (`scripts/kconfig/confdata.c::conf_read_simple` —
/// "If conflicting CONFIG options are given from an input file,
/// the last one wins.") makes user values override baked-in ones
/// on conflict.
///
/// A single `\n` separator is interleaved between the two
/// fragments. EMBEDDED_KCONFIG ends in a newline today, so the
/// interleaved `\n` produces a blank line between the segments —
/// kbuild's `.config` parser ignores blank lines (every
/// `if (!line[0])` short-circuit in `conf_read_simple`), so the
/// blank line is harmless. The separator is mandatory for the
/// adversarial case where the operator hand-crafts an
/// EMBEDDED_KCONFIG without a trailing newline AND a user
/// fragment that starts with `CONFIG_X` — without the
/// interleaved `\n`, the two would concatenate into a single
/// malformed line. Always emit the separator so the merge is
/// safe regardless of either side's terminator.
///
/// The production configure path in
/// [`crate::cli::kernel_build_pipeline`] calls this helper to build
/// the bytes handed to `configure_kernel`. Tests that assert
/// merge-ordering invariants call it directly so the production
/// byte sequence is what kbuild's last-wins rule operates on.
/// (Note: [`cache_key_suffix_with_extra`] hashes `extra` ALONE for
/// its `xkc{...}` segment — it doesn't pass through this helper —
/// so the cache-key suffix and the merged-fragment content evolve
/// independently. The cache-key segment exists to discriminate
/// extras-vs-no-extras at the cache layer; the merge ordering
/// exists to give kbuild the right final value.)
pub fn merge_kconfig_fragments<'a>(
    baked: &'a str,
    extra: Option<&str>,
) -> std::borrow::Cow<'a, str> {
    match extra {
        None => std::borrow::Cow::Borrowed(baked),
        Some(content) => std::borrow::Cow::Owned(format!("{baked}\n{content}")),
    }
}

pub use ktstr_macros::Claim;
pub use ktstr_macros::Payload;
pub use ktstr_macros::Scheduler;
pub use ktstr_macros::ktstr_test;

/// Internal re-exports for proc-macro-generated code. Not public API.
///
/// Grouped into a single hidden module so that `use ktstr::*;` pulls
/// in one module name instead of three leading-underscore items.
/// Consumers of `#[ktstr_test]` should not reference anything under
/// this path — the macro expansion names these crates via
/// `::ktstr::__private::ctor` / `linkme` / `serde_json` and the set
/// may change without notice.
#[doc(hidden)]
pub mod __private {
    pub use ctor;
    pub use linkme;
    pub use serde_json;
}

#[cfg(feature = "integration")]
pub use crate::probe::process::resolve_func_ip;

/// Re-exports for writing `#[ktstr_test]` functions.
///
/// ```rust
/// use ktstr::prelude::*;
///
/// #[ktstr_test(llcs = 1, cores = 2, threads = 1)]
/// fn my_test(ctx: &Ctx) -> Result<AssertResult> {
///     Ok(AssertResult::pass())
/// }
/// ```
///
/// For curated canned scenarios, see [`scenario::scenarios`].
pub mod prelude {
    pub use anyhow::Result;

    // `Scheduler` exists in both the type namespace (the
    // `test_support::Scheduler` *struct* — the scheduler-definition
    // record test authors build) and the macro namespace (the
    // `#[derive(Scheduler)]` *derive macro* from `ktstr-macros`).
    // Rust separates these, so `let s: Scheduler = ...` and
    // `#[derive(Scheduler)]` both resolve unambiguously. The double
    // spelling is intentional: `#[derive(Scheduler)]` generates a
    // `static` of type `Scheduler`, and matching the derive-macro
    // name to its emitted type reads naturally at the call site.
    pub use crate::assert::{
        Assert, AssertDetail, AssertResult, ClaimBuilder, DetailKind, NoteValue, SeqClaim,
        SetClaim, Verdict,
    };
    pub use crate::claim;
    pub use crate::cgroup::CgroupManager;
    pub use crate::host_context::HostContext;
    pub use crate::host_heap::HostHeapState;
    pub use crate::ktstr_test;
    pub use crate::scenario::backdrop::Backdrop;
    pub use crate::scenario::flags::FlagDecl;
    pub use crate::scenario::ops::{
        CgroupDef, CpusetSpec, HoldSpec, Op, Setup, Step, execute_defs, execute_scenario,
        execute_scenario_with, execute_steps, execute_steps_with,
    };
    pub use crate::scenario::payload_run::{PayloadHandle, PayloadRun};
    pub use crate::scenario::scenarios;
    pub use crate::scenario::{CgroupGroup, Ctx, collect_all, spawn_diverse};
    pub use crate::test_support::{
        BpfMapWrite, CgroupPath, MetricCheck, MemSideCache, Metric, MetricBounds, MetricHint,
        MetricSource, NumaDistance, NumaNode, OutputFormat, Payload, PayloadKind, PayloadMetrics,
        Polarity, Scheduler, SchedulerSpec, SidecarResult, Sysctl, Topology, extract_metrics,
    };
    pub use crate::{Payload, Scheduler};
    // `FlagDecl` is already re-exported via `crate::scenario::flags::FlagDecl`
    // above; omitted here because `test_support::FlagDecl` is the same item
    // (`pub use crate::scenario::flags::FlagDecl;` in test_support/mod.rs),
    // and a second prelude line would shadow harmlessly but add noise.
    //
    // The following items are intentionally NOT in the prelude. They
    // are binary-entry helpers (the `ktstr` / `cargo-ktstr` bins) or
    // macro-generated glue the `#[ktstr_test]` expansion consumes —
    // audiences distinct from the test-author surface this module
    // provides. Import directly from `ktstr::test_support::<item>`
    // when needed:
    // `newest_run_dir`, `runs_root`, `analyze_sidecars`, `ktstr_main`,
    // `ktstr_test_early_dispatch`, `run_ktstr_test`,
    // `resolve_scheduler`, `resolve_test_kernel`.
    pub use crate::topology::{LlcInfo, NodeMemInfo, TestTopology};
    pub use crate::vmm::disk_config::{DiskConfig, DiskThrottle, Filesystem};
    pub use crate::vmm::VirtioBlkCounters;
    pub use crate::workload::{
        AffinityIntent, ResolvedAffinity, CloneMode, MemPolicy, Migration, MpolFlags, Phase,
        SchedPolicy, WorkSpec, WorkType, WorkerReport, WorkloadConfig, WorkloadHandle, build_nodemask,
    };
}

/// Name of the environment variable that selects a kernel for every
/// ktstr entry point (`ktstr run`, `ktstr shell`, `cargo ktstr test`,
/// in-process tests, post-run analysis). Single source of truth so
/// the name is not spelled by hand at each reader; if the name ever
/// changes, the change lands in one place instead of fanning out to
/// every call site.
pub const KTSTR_KERNEL_ENV: &str = "KTSTR_KERNEL";

/// Name of the environment variable that carries the multi-kernel
/// fan-out list across the `cargo ktstr` → `cargo nextest` → test-
/// binary boundary. Format: `label1=path1;label2=path2;…` (semicolon
/// entry separator, `=` separates label from absolute kernel-dir
/// path). Empty / unset means "single-kernel mode" — the test binary
/// honours `KTSTR_KERNEL_ENV` directly.
///
/// Set by `cargo ktstr test --kernel A --kernel B` (or any
/// `--kernel` value that expands to ≥ 2 entries — repeated
/// `--kernel` flags, or a single `--kernel START..END` range that
/// expands to multiple stable releases via
/// [`crate::kernel_path::KernelId::Range`]) before the `exec` into
/// `cargo nextest`. Read by the test binary's `--list` /
/// `--exact` handlers in [`crate::test_support::dispatch`] to fan
/// the gauntlet across kernels: each (test × scenario × topology ×
/// flags × kernel) tuple becomes a distinct nextest test case so
/// nextest's parallelism, retries, and `-E` filtering work
/// natively. Per-variant subprocesses re-export `KTSTR_KERNEL` to
/// the kernel directory selected by the test name's `kernel_…`
/// suffix.
///
/// `KTSTR_KERNEL_ENV` is always set in tandem (to the first entry's
/// path) so downstream code that reads `KTSTR_KERNEL` directly —
/// budget-listing's vmlinux probe in `dispatch.rs` for example —
/// still observes a valid kernel even when running under multi-
/// kernel mode.
///
/// Single source of truth so the name is not spelled by hand at
/// each reader; if the name ever changes, the change lands in one
/// place instead of fanning out to every call site.
pub const KTSTR_KERNEL_LIST_ENV: &str = "KTSTR_KERNEL_LIST";

/// Name of the environment variable that overrides the rayon
/// pool width used by `cargo ktstr`'s `resolve_kernel_set` to
/// fan out per-spec kernel resolves (download / git-clone /
/// build) in parallel. Default cap is `available_parallelism()`
/// — the host's logical CPU count — chosen so download streams
/// do not outnumber threads the host can drive without
/// thrashing a contended local network (kernel.org CDN
/// per-IP throttle, developer ISP, CI shared NIC).
///
/// Operators override when the default is wrong for their
/// environment: a fast NIC + slow CPU benefits from raising
/// the cap above logical-CPU count to keep more downloads
/// in flight; a contended CI runner with concurrent jobs
/// benefits from lowering it to 1 or 2 to leave bandwidth
/// for siblings; a multi-version `--kernel A..Z` resolve on
/// a workstation may want a hand-tuned middle value to
/// balance throughput against background load.
///
/// Parsed as `usize`; 0 and unparseable values fall through
/// to the default cap so a typoed export does not silently
/// disable parallelism. Leading/trailing whitespace is trimmed
/// before parsing so a shell-quoted `=" 8 "` behaves the same
/// as the unquoted form. Read by
/// [`crate::cli::resolve_kernel_parallelism`] (the helper
/// that combines this env value with the
/// `available_parallelism()` fallback) so the parsing rules
/// live in one place.
///
/// Single source of truth so the name is not spelled by hand at
/// each reader; if the name ever changes, the change lands in one
/// place instead of fanning out to every call site.
pub const KTSTR_KERNEL_PARALLELISM_ENV: &str = "KTSTR_KERNEL_PARALLELISM";

/// Shared skip / error hint for call sites that cannot proceed
/// without a resolvable kernel. Phrased so the user sees the same
/// wording regardless of which layer surfaced the failure — tests,
/// CLI, monitor probes, and sidecar writers all point the operator
/// at the same remediation. Referenced by the non-VM-boot skip
/// paths in `cache.rs`, `probe/btf.rs`, `monitor/mod.rs`,
/// `test_support/eval.rs`, and `test_support/mod.rs`.
///
/// Format: caller prefixes the actionable first clause (e.g.
/// "no vmlinux found") and appends this constant as the
/// remediation tail. Keeping the prefix per-caller lets each site
/// name the specific artifact it needs while the `KTSTR_KERNEL`
/// wording stays consistent.
pub const KTSTR_KERNEL_HINT: &str = "set KTSTR_KERNEL to a kernel source directory, \
    a version (e.g. `6.14.2`), or a cache key (see `cargo ktstr kernel list`), or run \
    `cargo ktstr kernel build` to populate the cache";

/// Read [`KTSTR_KERNEL_ENV`] once, normalizing the raw value:
/// missing / empty / whitespace-only reads collapse to `None`, and
/// a surrounding-whitespace trim is applied so a shell-quoted
/// `KTSTR_KERNEL=" ../linux"` behaves the same as the unquoted
/// form. Every caller that reads the env var should route through
/// this helper so the normalization rules live in one place; a
/// future change to the rules (e.g. accepting a trailing slash)
/// propagates to every site automatically.
///
/// Returns the raw string; callers that need a structured
/// identifier parse with [`kernel_path::KernelId::parse`].
pub fn ktstr_kernel_env() -> Option<String> {
    std::env::var(KTSTR_KERNEL_ENV)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Find a bootable kernel image on the host.
///
/// Resolution chain:
/// 1. `KTSTR_KERNEL` env var, parsed via `KernelId`:
///    - Path: search that directory for an arch-specific image
///    - Version/CacheKey: require cache access (error if cache
///      directory cannot be opened); on cache miss, skip the
///      general cache scan (step 2) and fall to filesystem
/// 2. XDG cache: most recent cached image (newest first)
/// 3. Local build trees (`./linux`, `../linux`,
///    `/lib/modules/{release}/build`)
/// 4. Host paths (`/lib/modules/{release}/vmlinuz`,
///    `/boot/vmlinuz-{release}`, `/boot/vmlinuz`)
///
/// Returns `Err` when `KTSTR_KERNEL` is a path that does not contain
/// a kernel image, or when it is a version/cache key and the cache
/// directory cannot be opened. Returns `Ok(None)` when no kernel is
/// found.
pub fn find_kernel() -> anyhow::Result<Option<std::path::PathBuf>> {
    use kernel_path::KernelId;

    let release = rustix::system::uname()
        .release()
        .to_str()
        .ok()
        .map(str::to_owned);
    let release_ref = release.as_deref();

    // Track whether KTSTR_KERNEL was set with a non-path value.
    // When the user explicitly requests a version or cache key that
    // misses cache, the general cache scan (step 2) must be skipped
    // to avoid silently returning a different kernel.
    let mut skip_cache_scan = false;

    // 1. KTSTR_KERNEL env var with KernelId parsing. Route through
    // `ktstr_kernel_env()` so the empty/whitespace normalization
    // matches every other reader in the crate.
    if let Some(val) = ktstr_kernel_env() {
        match KernelId::parse(&val) {
            KernelId::Path(_) => match kernel_path::find_image(Some(&val), release_ref) {
                Some(p) => return Ok(Some(p)),
                None => anyhow::bail!(
                    "KTSTR_KERNEL={val} does not contain a kernel image. {KTSTR_KERNEL_HINT}"
                ),
            },
            KernelId::Version(ref ver) => {
                // Only tarball keys use the {ver}-tarball-{arch}-kc{suffix} pattern.
                // Git keys are {ref}-git-{hash}-{arch}-kc{suffix} and local keys
                // are local-{hash}-{arch}-kc{suffix} — neither contains the
                // version as a prefix, so only tarball lookup is valid here.
                let cache = cache::CacheDir::new().map_err(|e| {
                    anyhow::anyhow!(
                        "KTSTR_KERNEL={val} requires cache access, \
                         but cache directory could not be opened: {e}"
                    )
                })?;
                let arch = std::env::consts::ARCH;
                let key = format!("{ver}-tarball-{arch}-kc{}", cache_key_suffix());
                if let Some(entry) = cache.lookup(&key) {
                    return Ok(Some(entry.image_path()));
                }
                // Version not in cache — skip general cache scan to
                // avoid returning a different kernel version.
                skip_cache_scan = true;
            }
            KernelId::CacheKey(ref key) => {
                let cache = cache::CacheDir::new().map_err(|e| {
                    anyhow::anyhow!(
                        "KTSTR_KERNEL={val} requires cache access, \
                         but cache directory could not be opened: {e}"
                    )
                })?;
                if let Some(entry) = cache.lookup(key) {
                    return Ok(Some(entry.image_path()));
                }
                // Explicit cache key not found — skip general cache scan.
                skip_cache_scan = true;
            }
            // Multi-kernel specs (`A..B` ranges, `git+URL#REF` sources)
            // are only meaningful at the test/coverage/verifier
            // subcommand entry points where the runner can fan out
            // across kernels. The KTSTR_KERNEL env reader resolves a
            // single kernel image for in-process use (BTF lookup,
            // direct boot path) and has no dispatch loop, so a range
            // or git spec here cannot be expanded.
            //
            // Run `validate()` first so an inverted range surfaces
            // the specific "swap the endpoints" diagnostic instead
            // of getting masked by the generic "not supported in
            // env-var form" bail below — operators with a typo see
            // the actionable fix; valid-but-unsupported specs get
            // the generic redirect.
            id @ (KernelId::Range { .. } | KernelId::Git { .. }) => {
                if let Err(e) = id.validate() {
                    anyhow::bail!("KTSTR_KERNEL={val}: {e}");
                }
                anyhow::bail!(
                    "KTSTR_KERNEL={val}: multi-kernel specs (ranges, \
                     git sources) are not supported in env-var form. \
                     Use --kernel on the test/coverage/verifier \
                     subcommands, or set KTSTR_KERNEL to a single \
                     version, cache key, or path."
                );
            }
        }
    }

    // 2. XDG cache: most recent cached image.
    // Skipped when KTSTR_KERNEL was an explicit version or cache key
    // that missed — returning a different kernel would be surprising.
    if !skip_cache_scan
        && let Ok(cache) = cache::CacheDir::new()
        && let Ok(entries) = cache.list()
    {
        let kc_hash = kconfig_hash();
        for listed in &entries {
            let cache::ListedEntry::Valid(entry) = listed else {
                continue;
            };
            // Skip entries built with a different kconfig. Untracked
            // (pre-kconfig-tracking) entries are reused — their image
            // could still boot correctly, and skipping them would
            // permanently orphan legacy cache entries.
            if entry.kconfig_status(&kc_hash).is_stale() {
                continue;
            }
            let image = entry.image_path();
            // TOCTOU guard: list() guarantees image existence at scan time,
            // but a concurrent cache-clean could delete between scan and use.
            if !image.exists() {
                continue;
            }
            // Guard: if a cached vmlinux is present but is missing
            // the symbols monitor code requires, skip the entry so
            // the caller falls through to a source tree. Older
            // caches built by a strip pipeline that dropped data
            // sections would pass the image-exists check but fail
            // downstream when the monitor initializes.
            if let Some(vmlinux) = entry.vmlinux_path()
                && let Err(e) = monitor::symbols::KernelSymbols::from_vmlinux(&vmlinux)
            {
                tracing::warn!(
                    entry = %entry.path.display(),
                    error = %e,
                    "skipping cached kernel with unusable vmlinux"
                );
                continue;
            }
            return Ok(Some(image));
        }
    }

    // 3-4. Filesystem fallbacks (local build trees, host paths).
    Ok(kernel_path::find_image(None, release_ref))
}

/// Build a cargo binary package and return its output path.
///
/// Runs from the ktstr crate's manifest directory (which is also the
/// workspace root in this repo) so that workspace-level feature
/// unification (e.g. vendored libbpf-sys) is always in effect,
/// regardless of the calling process's working directory.
pub fn build_and_find_binary(package: &str) -> anyhow::Result<std::path::PathBuf> {
    let output = std::process::Command::new("cargo")
        .args(["build", "-p", package, "--message-format=json"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| anyhow::anyhow!("cargo build: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("cargo build -p {package} failed:\n{stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if let Ok(msg) = serde_json::from_str::<serde_json::Value>(line)
            && msg.get("reason").and_then(|r| r.as_str()) == Some("compiler-artifact")
            && msg
                .get("profile")
                .and_then(|p| p.get("test"))
                .and_then(|t| t.as_bool())
                == Some(false)
            && msg
                .get("target")
                .and_then(|t| t.get("kind"))
                .and_then(|k| k.as_array())
                .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("bin")))
            && let Some(filenames) = msg.get("filenames").and_then(|f| f.as_array())
            && let Some(path) = filenames.first().and_then(|f| f.as_str())
        {
            return Ok(std::path::PathBuf::from(path));
        }
    }
    anyhow::bail!("no binary artifact found for package '{package}'")
}

/// Resolve the current executable path, falling back to `/proc/self/exe`
/// when the binary has been deleted (e.g. by `cargo llvm-cov`).
///
/// On Linux, `std::env::current_exe()` reads `/proc/self/exe`.  When the
/// binary is unlinked while running, the kernel appends ` (deleted)` to
/// the readlink target, producing a path that does not exist on disk.
/// `/proc/self/exe` itself remains usable as a file path because the
/// kernel keeps the inode alive, so we fall back to it.
pub(crate) fn resolve_current_exe() -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context;
    let exe = std::env::current_exe().context("resolve current exe")?;
    if exe.exists() {
        return Ok(exe);
    }
    let proc_exe = std::path::PathBuf::from("/proc/self/exe");
    anyhow::ensure!(
        proc_exe.exists(),
        "current exe not found: {}",
        exe.display()
    );
    Ok(proc_exe)
}

/// Boot a KVM VM in interactive shell mode.
///
/// Builds an initramfs with busybox and optional include files, then
/// launches a VM with bidirectional stdin/stdout forwarding. The guest
/// runs a shell via busybox; user-provided files are available at
/// `/include-files/<name>`.
///
/// `kernel`: path to the kernel image (bzImage/Image).
/// `numa_nodes`, `llcs`, `cores`, `threads`: guest CPU topology.
/// `include_files`: `(archive_path, host_path)` pairs for files to
///   include in the guest.
/// `memory_mb`: explicit guest memory override in MB. When `None`,
///   memory is computed from actual initramfs size after build.
/// `disk`: optional virtio-blk device backing for `/dev/vda`. When
///   `Some`, the framework calls
///   [`vmm::KtstrVm::builder`]'s `.disk(..)` so the guest probes a
///   raw block device sized per `disk.capacity_mb`.
#[allow(clippy::too_many_arguments)]
pub fn run_shell(
    kernel: std::path::PathBuf,
    numa_nodes: u32,
    llcs: u32,
    cores: u32,
    threads: u32,
    include_files: &[(&str, &std::path::Path)],
    memory_mb: Option<u32>,
    dmesg: bool,
    exec: Option<&str>,
    disk: Option<vmm::disk_config::DiskConfig>,
) -> anyhow::Result<()> {
    let payload = resolve_current_exe()?;

    let owned_includes: Vec<(String, std::path::PathBuf)> = include_files
        .iter()
        .map(|(a, p)| (a.to_string(), p.to_path_buf()))
        .collect();

    let mut cmdline = format!("KTSTR_MODE=shell KTSTR_TOPO={numa_nodes},{llcs},{cores},{threads}");
    if dmesg {
        cmdline.push_str(" loglevel=7");
    }
    if let Ok(val) = std::env::var("RUST_LOG") {
        cmdline.push_str(&format!(" RUST_LOG={val}"));
    }

    // Pass host terminal environment to guest.
    if let Ok(term) = std::env::var("TERM") {
        cmdline.push_str(&format!(" KTSTR_TERM={term}"));
    }
    if let Ok(ct) = std::env::var("COLORTERM") {
        cmdline.push_str(&format!(" KTSTR_COLORTERM={ct}"));
    }

    // Pass host terminal dimensions to guest for correct line wrapping.
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(libc::STDIN_FILENO, libc::TIOCGWINSZ, &mut ws) == 0
            && ws.ws_col > 0
            && ws.ws_row > 0
        {
            cmdline.push_str(&format!(
                " KTSTR_COLS={} KTSTR_ROWS={}",
                ws.ws_col, ws.ws_row
            ));
        }
    }

    let no_perf_mode = std::env::var("KTSTR_NO_PERF_MODE").is_ok();
    let mut builder = vmm::KtstrVm::builder()
        .kernel(&kernel)
        .init_binary(&payload)
        .topology(numa_nodes, llcs, cores, threads)
        .cmdline(&cmdline)
        .include_files(owned_includes)
        .busybox(true)
        .dmesg(dmesg)
        .no_perf_mode(no_perf_mode);

    if let Some(cmd) = exec {
        builder = builder.exec_cmd(cmd.to_string());
    }

    if let Some(d) = disk {
        builder = builder.disk(d);
    }

    builder = match memory_mb {
        Some(mb) => builder.memory_mb(mb),
        None => builder.memory_deferred(),
    };

    let vm = builder.build()?;

    vm.run_interactive()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_current_exe_happy_path() {
        let exe = resolve_current_exe().unwrap();
        // The test binary is running, so current_exe() returns a path that
        // exists on disk. resolve_current_exe should return that same path
        // (the exe.exists() early-return branch).
        let std_exe = std::env::current_exe().unwrap();
        if std_exe.exists() {
            // Happy path: binary not deleted, should return std::env::current_exe().
            assert_eq!(exe, std_exe);
        } else {
            // Fallback: binary deleted (llvm-cov), should return /proc/self/exe.
            assert_eq!(exe, std::path::PathBuf::from("/proc/self/exe"));
        }
    }

    // -- errno_name --

    #[test]
    fn errno_name_known_values() {
        assert_eq!(errno_name(libc::EPERM), Some("EPERM"));
        assert_eq!(errno_name(libc::ENOENT), Some("ENOENT"));
        assert_eq!(errno_name(libc::EINVAL), Some("EINVAL"));
        assert_eq!(errno_name(libc::ENOMEM), Some("ENOMEM"));
        assert_eq!(errno_name(libc::EBUSY), Some("EBUSY"));
        assert_eq!(errno_name(libc::EACCES), Some("EACCES"));
        assert_eq!(errno_name(libc::EAGAIN), Some("EAGAIN"));
        assert_eq!(errno_name(libc::ENOSYS), Some("ENOSYS"));
        assert_eq!(errno_name(libc::ETIMEDOUT), Some("ETIMEDOUT"));
    }

    #[test]
    fn errno_name_unknown() {
        assert_eq!(errno_name(9999), None);
        assert_eq!(errno_name(0), None);
        assert_eq!(errno_name(-1), None);
    }

    // -- find_kernel cache filter --

    /// `find_kernel`'s cache-scan step (step 2) must keep cache
    /// entries whose `ktstr_kconfig_hash` is `None` (pre-tracking
    /// format → [`cache::KconfigStatus::Untracked`]). The filter
    /// checks for `KconfigStatus::Stale` specifically; `Untracked`
    /// falls through to the return.
    ///
    /// A regression that tightened the filter to "anything not
    /// Matches" (e.g. `kconfig_status(&kc) != Matches`) would quietly
    /// drop every legacy cache entry built before ktstr tracked the
    /// kconfig fingerprint, forcing users to rebuild kernels whose
    /// only defect is the absence of a recorded hash. This test
    /// materializes exactly that shape — one valid image, no
    /// recorded hash — and asserts `find_kernel` returns it.
    #[test]
    fn find_kernel_preserves_untracked_cache_entries() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        // `find_kernel` reads KTSTR_KERNEL and KTSTR_CACHE_DIR. Both
        // are process-wide, so other tests in this binary must not be
        // mutating them in parallel while this test owns them.
        let _env_lock = lock_env();
        // KTSTR_KERNEL: unset so find_kernel skips step 1 and falls
        // straight into the cache scan (step 2), which is the branch
        // this test targets.
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");

        // KTSTR_CACHE_DIR: point at an isolated temp dir so the
        // test sees only the Untracked entry we stage below — and
        // the host's real cache (if any) does not influence the
        // result.
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        // Stage one valid image with `ktstr_kconfig_hash = None`
        // (Untracked). `has_vmlinux` stays false so find_kernel's
        // vmlinux-symbol guard at lib.rs (only reached when
        // vmlinux_path() is Some) does not fire.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"fake kernel image").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()));
        // `ktstr_kconfig_hash` defaults to None in `KernelMetadata::new`,
        // which is exactly the Untracked shape this test needs.
        assert!(
            meta.ktstr_kconfig_hash.is_none(),
            "test fixture must have no recorded kconfig hash to exercise the \
             Untracked branch of kconfig_status"
        );
        let entry = cache
            .store("untracked-entry", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        let expected_image = entry.image_path();
        assert!(
            expected_image.exists(),
            "fixture image must exist on disk so find_kernel's image.exists() \
             check passes — got {expected_image:?}"
        );

        // find_kernel must return the Untracked entry's image.
        let resolved = find_kernel().unwrap();
        assert_eq!(
            resolved,
            Some(expected_image),
            "find_kernel dropped an Untracked cache entry — the kconfig-hash \
             filter at lib.rs must treat `Untracked` as keep, not stale"
        );
    }

    /// [`find_kernel`]'s cache scan (step 2) must skip entries where
    /// `matches!(entry.kconfig_status(&hash), KconfigStatus::Stale { .. })`
    /// and fall through to the next viable candidate.
    ///
    /// Two valid entries are seeded:
    ///
    /// * a current-hash entry built at `2026-04-01T00:00:00Z`, and
    /// * a stale-hash entry built at `2026-04-20T00:00:00Z`.
    ///
    /// [`cache::CacheDir::list`] sorts by `built_at` descending, so
    /// the stale entry is yielded first. If the `KconfigStatus::Stale`
    /// skip branch at the top of [`find_kernel`]'s cache-scan loop
    /// regressed to a no-op, `find_kernel` would return the stale
    /// entry's image. Asserting it returns the current entry's image
    /// proves the filter actually engages — strictly stronger than a
    /// single-entry `assert_ne!` on the stale path.
    #[test]
    fn find_kernel_skips_stale_cache_entry() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");

        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let current_hash = crate::kconfig_hash();
        let stale_hash = format!("{current_hash}-stale");

        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();

        // Current-hash entry: older built_at. cache.list() sorts
        // newest-first, so this entry lands AFTER the stale entry — if
        // find_kernel failed to skip stale, it would return the stale
        // image instead of this one.
        let current_image = src_dir.path().join("current.bzImage");
        std::fs::write(&current_image, b"current kernel image").unwrap();
        let current_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "current.bzImage".to_string(),
            "2026-04-01T00:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
        .with_ktstr_kconfig_hash(Some(current_hash.clone()));
        let current_entry = cache
            .store(
                "current-entry",
                &CacheArtifacts::new(&current_image),
                &current_meta,
            )
            .unwrap();

        // Stale entry: newer built_at, so list() yields it first.
        let stale_image = src_dir.path().join("stale.bzImage");
        std::fs::write(&stale_image, b"stale kernel image").unwrap();
        let stale_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "stale.bzImage".to_string(),
            "2026-04-20T00:00:00Z".to_string(),
        )
        .with_version(Some("6.14.3".to_string()))
        .with_ktstr_kconfig_hash(Some(stale_hash));
        cache
            .store(
                "stale-entry",
                &CacheArtifacts::new(&stale_image),
                &stale_meta,
            )
            .unwrap();

        let resolved = find_kernel().unwrap();
        assert_eq!(
            resolved,
            Some(current_entry.image_path()),
            "find_kernel must skip the newer stale entry and return the \
             current-hash entry — regression of the KconfigStatus::Stale \
             skip branch in find_kernel's cache-scan loop",
        );
    }

    // -- worker_ready marker path format --

    /// Pin the path format produced by
    /// [`crate::worker_ready::worker_ready_marker_path`].
    /// Downstream callers never spell the marker path as a literal —
    /// the worker writes it via `worker_ready_marker_path(pid)` and
    /// the test side polls via the same function — so a rename of
    /// the prefix does not surface at any caller's build time. This
    /// test's literal assertions are where the drift lands: changing
    /// the prefix breaks these equalities, catching the serialized-
    /// form change that would otherwise go silent.
    ///
    /// Lives in lib.rs (not in `worker_ready.rs`) because that file
    /// is dual-compiled via `#[path]` into the worker bin; a
    /// `#[cfg(test)] mod tests` inside it would duplicate the test
    /// into both the lib test binary and the bin test binary.
    #[test]
    fn worker_ready_marker_path_format_is_stable() {
        use crate::worker_ready::{WORKER_READY_MARKER_PREFIX, worker_ready_marker_path};
        assert_eq!(WORKER_READY_MARKER_PREFIX, "/tmp/ktstr-worker-ready-");
        assert_eq!(worker_ready_marker_path(0), "/tmp/ktstr-worker-ready-0");
        assert_eq!(
            worker_ready_marker_path(12345),
            "/tmp/ktstr-worker-ready-12345"
        );
        assert_eq!(
            worker_ready_marker_path(u32::MAX),
            "/tmp/ktstr-worker-ready-4294967295"
        );
    }

    // -- ktstr_kernel_env + KTSTR_KERNEL round-trip --
    //
    // KTSTR_KERNEL is the canonical cross-process hand-off for kernel
    // selection: `cargo ktstr test` resolves `--kernel` in the parent,
    // writes the resolved path back through `KTSTR_KERNEL_ENV`, and
    // every reader in the child (`find_kernel`, `detect_kernel_version`,
    // `find_test_vmlinux`) pulls it via `ktstr_kernel_env`. A
    // normalization drift between writer and reader would produce
    // silent resolution errors on the child side. Pin the round-trip
    // so such a drift is caught at test time.

    /// `ktstr_kernel_env` must return exactly the unmodified interior
    /// string when the env holds a plain absolute path. This is the
    /// happy-path channel between the parent's
    /// `std::fs::canonicalize(&p)` → `cmd.env(KTSTR_KERNEL_ENV, dir)`
    /// and the child's `ktstr_kernel_env` read. A regression that
    /// changed the normalization (adding a trailing slash, resolving
    /// symlinks, URL-encoding, etc.) would break the hand-off.
    #[test]
    fn ktstr_kernel_env_round_trips_absolute_path() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let canonical = std::fs::canonicalize(tmp.path()).unwrap();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, &canonical);
        let read_back = ktstr_kernel_env().expect("env is set");
        assert_eq!(
            std::path::PathBuf::from(&read_back),
            canonical,
            "writer-reader round-trip must preserve the exact path; \
             drift between parent's canonicalize output and child's \
             ktstr_kernel_env read breaks every downstream resolver",
        );
    }

    /// Unset env reads as `None` — the default-resolution branch
    /// that `find_kernel`'s cache scan and `find_test_vmlinux`'s
    /// local-tree fallback both depend on.
    #[test]
    fn ktstr_kernel_env_unset_is_none() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::remove(KTSTR_KERNEL_ENV);
        assert!(
            ktstr_kernel_env().is_none(),
            "unset KTSTR_KERNEL must read as None so fallback resolvers activate",
        );
    }

    /// Empty-string env reads as `None`. Every reader should treat
    /// `KTSTR_KERNEL=""` the same as "unset" rather than erroring
    /// on an empty path — shells and Makefiles routinely emit empty
    /// values for unused variables, and failing on them would break
    /// unrelated CI flows.
    #[test]
    fn ktstr_kernel_env_empty_is_none() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "");
        assert!(
            ktstr_kernel_env().is_none(),
            "empty KTSTR_KERNEL must collapse to None; CI flows routinely \
             pass empty strings for unused variables",
        );
    }

    /// Whitespace-only env reads as `None`. A shell-quoted
    /// `KTSTR_KERNEL="   "` is semantically the empty case even
    /// though `std::env::var` sees a non-empty string — the reader
    /// must trim before the empty check.
    #[test]
    fn ktstr_kernel_env_whitespace_is_none() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "   \t\n  ");
        assert!(
            ktstr_kernel_env().is_none(),
            "whitespace-only KTSTR_KERNEL must collapse to None via trim \
             + empty-filter; no caller parses a whitespace-only value \
             meaningfully",
        );
    }

    /// A valid path with surrounding whitespace is trimmed and
    /// returned as the interior token. Pins the contract that the
    /// reader tolerates shell-quoting quirks without distorting the
    /// underlying value.
    #[test]
    fn ktstr_kernel_env_trims_surrounding_whitespace() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "  ../linux  ");
        let read_back = ktstr_kernel_env().expect("env is set");
        assert_eq!(
            read_back, "../linux",
            "surrounding whitespace must be trimmed but the interior \
             preserved verbatim",
        );
    }

    /// `find_kernel` must call `KernelId::validate()` BEFORE the
    /// generic "multi-kernel specs are not supported in env-var form"
    /// bail when the env value parses as a Range or Git spec, so an
    /// inverted range like `KTSTR_KERNEL=6.16..6.12` surfaces the
    /// actionable "swap the endpoints" diagnostic. A future
    /// regression that drops the validate() call (or reorders it
    /// after the generic bail) would flip the error from the
    /// specific form to the generic redirect, landing here.
    ///
    /// `KTSTR_CACHE_DIR` is pointed at a fresh tempdir so the cache
    /// scan can't preempt the env-reader (the reader runs first
    /// regardless, but pinning the cache state prevents host noise
    /// from changing the assertion shape).
    #[test]
    fn find_kernel_inverted_range_env_surfaces_swap_diagnostic() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _env_lock = lock_env();
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);
        let _kernel_guard = EnvVarGuard::set(KTSTR_KERNEL_ENV, "6.16..6.12");

        let err = find_kernel().expect_err("inverted range must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("inverted kernel range"),
            "validate() diagnostic must surface ahead of the generic \
             env-form bail; got: {msg}",
        );
        assert!(
            msg.contains("6.12..6.16"),
            "swap suggestion must appear in the error; got: {msg}",
        );
        assert!(
            !msg.contains("not supported in env-var form"),
            "validate() must short-circuit before the generic bail; got: {msg}",
        );
    }

    /// `KTSTR_KERNEL_ENV` must match the literal spelling `"KTSTR_KERNEL"`.
    /// Trivial pin, but load-bearing: readers that bypass the helper
    /// (e.g. hand-rolled `std::env::var("KTSTR_KERNEL")` in an ad-hoc
    /// script) match this string. A typo here would silently divorce
    /// the crate's canonical reader from every external tool.
    #[test]
    fn ktstr_kernel_env_constant_is_literal() {
        assert_eq!(KTSTR_KERNEL_ENV, "KTSTR_KERNEL");
    }

    // -- extra_kconfig_hash + cache_key_suffix_with_extra --
    //
    // The two-segment cache-key suffix underpins cargo-ktstr's
    // `--extra-kconfig` behavior: an extra-kconfig build must land
    // at a distinct cache slot from a vanilla build (different
    // content = miss), the same extra content must hit the same
    // slot on re-run (same content = hit), and `None` (no
    // `--extra-kconfig`) must produce the byte-identical suffix to
    // `cache_key_suffix()` so paths that don't expose the flag
    // continue resolving the existing keyspace.
    //
    // Suffix shape is `kc{baked_hash}` (no extra) or
    // `kc{baked_hash}-xkc{extra_hash}` (with extra). The two-segment
    // form makes `kernel list` self-describing — a reader can see at
    // a glance which entries carry user extras.

    /// `cache_key_suffix_with_extra(None)` must equal
    /// `cache_key_suffix()` byte-for-byte. Pins the
    /// no-`--extra-kconfig` path's compatibility contract: the
    /// test/coverage/shell/verifier resolution paths (which never
    /// pass `Some(extra)`) keep producing the pre-flag cache keys so
    /// existing cache entries remain addressable across the addition
    /// of this feature.
    #[test]
    fn cache_key_suffix_with_extra_none_matches_bare_suffix() {
        assert_eq!(cache_key_suffix_with_extra(None), cache_key_suffix());
    }

    /// `cache_key_suffix_with_extra(Some(content))` must contain
    /// the bare baked-in hash AS A PREFIX followed by `-xkc{...}`.
    /// Pins the two-segment shape:
    /// the leading segment is the existing `kconfig_hash()` so
    /// `kernel list` can decompose the suffix into baked-in vs
    /// user-extras components.
    #[test]
    fn cache_key_suffix_with_extra_some_has_two_segment_shape() {
        let suffix = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        let baked = kconfig_hash();
        assert!(
            suffix.starts_with(&baked),
            "Some suffix must start with bare baked-in hash {baked:?}, got {suffix:?}"
        );
        let after = &suffix[baked.len()..];
        assert!(
            after.starts_with("-xkc"),
            "after the baked-in segment, the next bytes must be `-xkc`, got {after:?}"
        );
        let extra_segment = &after["-xkc".len()..];
        assert_eq!(
            extra_segment.len(),
            8,
            "extra-hash segment must be 8 hex chars, got {extra_segment:?}"
        );
        assert!(
            extra_segment.chars().all(|c| c.is_ascii_hexdigit()),
            "extra-hash segment must be lowercase hex, got {extra_segment:?}"
        );
    }

    /// `cache_key_suffix_with_extra(Some(...))` must DIFFER from
    /// `cache_key_suffix()` for any non-empty user fragment. Pins
    /// the cache-discrimination contract: a build with a user
    /// fragment lands at a different cache key from a vanilla
    /// build. Also asserts the 8-hex-char shape of the appended
    /// xkc segment so a regression that changed the hash width
    /// (e.g. switched to a longer/shorter hex digest) doesn't
    /// silently slip past this test.
    #[test]
    fn cache_key_suffix_with_extra_some_differs_from_bare_suffix() {
        let suffix = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        assert_ne!(suffix, cache_key_suffix());
        // Width pin: the suffix-shape contract embeds the same
        // 8-hex-char width on both segments.
        let baked = kconfig_hash();
        let after = &suffix[baked.len()..];
        assert_eq!(
            after.len(),
            "-xkc".len() + 8,
            "suffix tail must be `-xkc{{8 hex chars}}`, got {after:?}"
        );
    }

    /// Production format-string shape assertion: the suffix
    /// `cache_key_suffix_with_extra(Some(...))` produces must be
    /// exactly the literal `format!("{baked}-xkc{extra_hash}")`
    /// cargo-ktstr.rs uses to build its tarball cache key. Pins
    /// the structural mirror so a refactor that changed the
    /// helper's format would surface here as a divergence from
    /// the production call site.
    #[test]
    fn cache_key_suffix_with_extra_matches_production_format_string() {
        let extra = "CONFIG_FOO=y\n";
        let baked = kconfig_hash();
        let extra_h = extra_kconfig_hash(extra);
        let helper = cache_key_suffix_with_extra(Some(extra));
        let expected = format!("{baked}-xkc{extra_h}");
        assert_eq!(
            helper, expected,
            "helper output must match production format `{{baked}}-xkc{{extra}}` \
             (cargo-ktstr.rs builds the tarball cache key with the same shape \
             via `{{ver}}-tarball-{{arch}}-kc{{cache_key_suffix_with_extra(...)}}`)"
        );
    }

    /// An empty user fragment ALSO differs from `None`: the
    /// `Some("")` branch always emits `-xkc{empty_hash}` as a
    /// distinct second segment, while `None` produces only the
    /// bare baked-in hash. Pins that `--extra-kconfig /empty/file`
    /// is a deliberate signal — even when no symbols are added,
    /// the build lands in a distinct cache slot from a no-flag
    /// build.
    #[test]
    fn cache_key_suffix_with_extra_empty_differs_from_none() {
        let with_empty = cache_key_suffix_with_extra(Some(""));
        let without = cache_key_suffix_with_extra(None);
        assert_ne!(with_empty, without);
    }

    /// Same user fragment must produce the SAME suffix across
    /// invocations. Pins cache-hit determinism for the
    /// `--extra-kconfig` repeat-invocation case.
    #[test]
    fn cache_key_suffix_with_extra_same_content_same_suffix() {
        let extra = "CONFIG_FOO=y\nCONFIG_BAR=n\n";
        let a = cache_key_suffix_with_extra(Some(extra));
        let b = cache_key_suffix_with_extra(Some(extra));
        assert_eq!(a, b, "same fragment must produce same suffix");
    }

    /// Different user fragments must produce DIFFERENT suffixes.
    /// Pins cache-miss discrimination across distinct extra files.
    /// The `xkc` segment carries the discriminator while the
    /// baked-in `kc` segment stays constant.
    #[test]
    fn cache_key_suffix_with_extra_different_content_different_suffix() {
        let a = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        let b = cache_key_suffix_with_extra(Some("CONFIG_FOO=n\n"));
        assert_ne!(a, b, "distinct fragments must produce distinct suffixes");
        // The baked-in prefix must match across both — only the
        // `-xkc{...}` tail differs.
        let baked = kconfig_hash();
        assert!(a.starts_with(&baked) && b.starts_with(&baked));
    }

    /// `extra_kconfig_hash` is 8 hex chars (CRC32 in lowercase
    /// hex), matching the existing `kconfig_hash` shape so the
    /// `xkc{...}` segment width is consistent with the baked-in
    /// `kc{...}` segment.
    #[test]
    fn extra_kconfig_hash_is_8_hex_chars() {
        for content in ["", "CONFIG_X=y\n", "# CONFIG_BPF is not set\n"] {
            let h = extra_kconfig_hash(content);
            assert_eq!(h.len(), 8, "expected 8 hex chars, got {h}");
            assert!(
                h.chars().all(|c| c.is_ascii_hexdigit()),
                "expected lowercase hex, got {h}",
            );
        }
    }

    /// `extra_kconfig_hash` hashes raw bytes — no comment stripping,
    /// no CRLF canonicalization. Two semantically-equivalent inputs
    /// with different comments or line endings produce different
    /// hashes and therefore land at distinct cache slots.
    #[test]
    fn extra_kconfig_hash_is_byte_sensitive() {
        let lf = "CONFIG_FOO=y\n";
        let crlf = "CONFIG_FOO=y\r\n";
        assert_ne!(
            extra_kconfig_hash(lf),
            extra_kconfig_hash(crlf),
            "CRLF and LF must hash differently per ruling D2"
        );

        let with_comment = "# user note\nCONFIG_FOO=y\n";
        let without_comment = "CONFIG_FOO=y\n";
        assert_ne!(
            extra_kconfig_hash(with_comment),
            extra_kconfig_hash(without_comment),
            "comments must affect the hash per ruling D1"
        );
    }

    /// Coordinator item 18: CRLF cache-key discrimination. A user
    /// who edits their fragment on a Windows host and saves with
    /// CRLF line endings must land at a different cache slot from
    /// a Unix-LF fragment with otherwise-identical content. Per
    /// ruling D2 (no canonicalization), this is by design — the
    /// disk waste is the price of byte-deterministic discrimination.
    #[test]
    fn cache_key_suffix_with_extra_crlf_differs_from_lf() {
        let lf = "CONFIG_FOO=y\n";
        let crlf = "CONFIG_FOO=y\r\n";
        let lf_suffix = cache_key_suffix_with_extra(Some(lf));
        let crlf_suffix = cache_key_suffix_with_extra(Some(crlf));
        assert_ne!(
            lf_suffix, crlf_suffix,
            "LF and CRLF user fragments must produce distinct cache \
             keys per ruling D2 (no CRLF canonicalization). A Windows \
             operator and a Unix operator who supplied 'the same' \
             fragment land at distinct cache slots; this is the \
             documented byte-deterministic cache contract."
        );
    }

    /// Coordinator item 22: legacy cache entry must NOT be served
    /// when `--extra-kconfig` is passed. The cache key shape
    /// `kc{baked}` (legacy, no extras) and `kc{baked}-xkc{...}`
    /// (with extras) are STRUCTURALLY distinct strings, so any
    /// cache lookup that builds its key via
    /// `cache_key_suffix_with_extra(Some(...))` cannot collide with
    /// an entry stored under `cache_key_suffix()` only.
    ///
    /// Pins this invariant at the suffix level: an extras-aware
    /// suffix is strictly LONGER than the bare suffix, and the
    /// bare suffix is a proper prefix of the extras suffix. A
    /// substring lookup (cache lookup is exact-match by key, not
    /// substring) therefore cannot serve the legacy slot when the
    /// caller asks for an extras key. End-to-end roundtrip is
    /// covered at the integration level; this unit test pins the
    /// structural property the integration relies on.
    #[test]
    fn legacy_bare_suffix_is_proper_prefix_of_extras_suffix() {
        let bare = cache_key_suffix();
        let extras = cache_key_suffix_with_extra(Some("CONFIG_FOO=y\n"));
        assert!(
            extras.starts_with(&bare),
            "extras suffix must extend bare suffix — bare={bare:?} extras={extras:?}",
        );
        assert!(
            extras.len() > bare.len(),
            "extras suffix must be strictly longer than bare suffix",
        );
        assert_ne!(
            bare, extras,
            "structural distinction: legacy entries (key ending in `kc{{bare}}`) cannot \
             collide with extras entries (key ending in `kc{{bare}}-xkc{{...}}`) on \
             exact-match cache lookup."
        );
    }

    /// Coordinator item 135 / item 29: pin that
    /// `extra_kconfig_hash("")` is the legitimate CRC32 of zero
    /// bytes (`00000000`), NOT a sentinel meaning "no extras".
    /// `None` is the no-extras signal — `Some("")` means "operator
    /// supplied an empty fragment file". An empty Some still
    /// produces a distinct cache key from None, so cache-key
    /// readers must distinguish "extras absent" (None →
    /// `kc{baked}` only) from "extras empty" (Some("") →
    /// `kc{baked}-xkc00000000`) by the presence/absence of the
    /// `-xkc` segment, NOT by hash content.
    #[test]
    fn extra_kconfig_hash_empty_is_crc32_zero_not_sentinel() {
        let empty = extra_kconfig_hash("");
        assert_eq!(
            empty, "00000000",
            "CRC32 of zero bytes is 0x00000000 by spec. This value is \
             a legitimate hash output, not a sentinel — readers that \
             want to detect 'no extras' must check the metadata's \
             `extra_kconfig_hash: Option<String>` for None, not for \
             this string."
        );
    }

    /// MERGE-SEMANTICS pin: when the user fragment overrides a
    /// baked-in symbol, the merged content fed to
    /// `make olddefconfig` must place the user line LAST so
    /// kbuild's last-wins rule
    /// (`scripts/kconfig/confdata.c::conf_read_simple` — "If
    /// conflicting CONFIG options are given from an input file,
    /// the last one wins.") makes the user value take precedence.
    ///
    /// Calls [`merge_kconfig_fragments`] DIRECTLY — the same
    /// helper [`crate::cli::kernel_build_pipeline`] uses to build
    /// the configure-pass content. Pinning the production helper
    /// (rather than reproducing its `format!` shape inline) means
    /// a regression that swapped the merge order would break this
    /// test in lock-step with the production path it represents.
    ///
    /// `EMBEDDED_KCONFIG` carries `CONFIG_BPF=y` (see
    /// `ktstr.kconfig`). A user fragment that flips it to
    /// `# CONFIG_BPF is not set` must appear AFTER the baked-in
    /// `CONFIG_BPF=y` line in the merged content. We can't drive
    /// `make olddefconfig` from a pure unit test (no in-tree
    /// kbuild fixture), so the test directly inspects the merged-
    /// fragment construction the pipeline uses and verifies the
    /// ordering invariant kbuild's last-wins rule depends on.
    #[test]
    fn merge_user_extra_appears_after_baked_in_for_conflict_resolution() {
        let user = "# CONFIG_BPF is not set\n";
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, Some(user));
        let baked_pos = merged
            .find("CONFIG_BPF=y")
            .expect("baked-in CONFIG_BPF=y must be present in merged fragment");
        let user_pos = merged
            .find("# CONFIG_BPF is not set")
            .expect("user override must be present in merged fragment");
        assert!(
            baked_pos < user_pos,
            "baked-in line must appear BEFORE user override so kbuild's \
             last-wins rule (confdata.c::conf_read_simple) keeps the user \
             value (baked_pos={baked_pos}, user_pos={user_pos})",
        );
        // Item 20: pin that the LAST occurrence of the symbol in
        // the merged content is the user's override. Walks every
        // line, tracks the last hit, and asserts it is the user's
        // disable directive — the kbuild parser walks lines top to
        // bottom and the last assignment wins, so the LAST
        // occurrence determines the final value.
        let mut last = None;
        for line in merged.lines() {
            let trimmed = line.trim();
            if trimmed == "CONFIG_BPF=y" || trimmed == "# CONFIG_BPF is not set" {
                last = Some(trimmed.to_string());
            }
        }
        assert_eq!(
            last.as_deref(),
            Some("# CONFIG_BPF is not set"),
            "the LAST occurrence of CONFIG_BPF in the merged content must \
             be the user override; kbuild's `conf_read_simple` walks lines \
             top-to-bottom and keeps the last assignment, so the user line \
             determines the final config value",
        );
    }

    /// NON-CONFLICT pin: when the user fragment adds a symbol the
    /// baked-in fragment doesn't
    /// mention, the merged content carries BOTH lines verbatim.
    /// `make olddefconfig` then sees two distinct symbols (one
    /// from each origin) and produces a `.config` containing
    /// both. Calls [`merge_kconfig_fragments`] directly so the
    /// production path's combine semantics are pinned.
    #[test]
    fn merge_user_extra_combines_with_baked_in_for_disjoint_symbols() {
        // Pick a CONFIG_X name not present in EMBEDDED_KCONFIG.
        let novel = "CONFIG_KTSTR_TEST_NOVEL_SYMBOL_FOR_MERGE_TEST=y\n";
        assert!(
            !EMBEDDED_KCONFIG.contains("CONFIG_KTSTR_TEST_NOVEL_SYMBOL_FOR_MERGE_TEST"),
            "test fixture must use a symbol absent from EMBEDDED_KCONFIG"
        );
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, Some(novel));
        // Both the user-novel line and at least one canonical
        // baked-in line must appear in the merged content.
        assert!(
            merged.contains("CONFIG_KTSTR_TEST_NOVEL_SYMBOL_FOR_MERGE_TEST=y"),
            "user-novel line must appear in merged fragment",
        );
        assert!(
            merged.contains("CONFIG_BPF=y"),
            "baked-in CONFIG_BPF=y must still appear in merged fragment",
        );
    }

    /// `merge_kconfig_fragments(baked, None)` returns the baked
    /// string unchanged — pinning the no-extras short-circuit so
    /// callers that pass `None` always observe the original
    /// fragment byte-for-byte.
    #[test]
    fn merge_kconfig_fragments_none_returns_baked_unchanged() {
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, None);
        assert_eq!(
            merged, EMBEDDED_KCONFIG,
            "merge with None must return the baked fragment unchanged"
        );
    }

    /// `merge_kconfig_fragments(baked, Some(""))` returns
    /// `{baked}\n` — the empty string still triggers the
    /// `format!` branch which appends a separator newline. Pins
    /// that the helper's branch boundary is `Option::is_some`,
    /// not `Option::is_some && !is_empty`. Operators reading the
    /// merged content under an empty fragment see the baked-in
    /// content followed by a single trailing newline (which kbuild
    /// ignores).
    #[test]
    fn merge_kconfig_fragments_some_empty_appends_separator_newline() {
        let merged = merge_kconfig_fragments(EMBEDDED_KCONFIG, Some(""));
        let expected = format!("{EMBEDDED_KCONFIG}\n");
        assert_eq!(merged, expected);
    }

    /// Last-wins ordering invariant: when the user fragment overrides
    /// a baked-in symbol, the user line MUST appear AFTER the baked-in
    /// line in the merged content. kbuild's
    /// `scripts/kconfig/confdata.c::conf_read_simple` keeps the
    /// last-occurring assignment per symbol, so a regression that
    /// flipped the order would silently make user values lose. Pinning
    /// at the merge-helper level catches the byte sequence kbuild
    /// operates on without spinning up a kernel build.
    #[test]
    fn merge_kconfig_fragments_user_line_appears_after_baked_for_overrides() {
        let baked = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        let user = "CONFIG_FOO=n\n";
        let merged = merge_kconfig_fragments(baked, Some(user)).into_owned();
        let baked_idx = merged
            .find("CONFIG_FOO=y")
            .expect("baked CONFIG_FOO=y must be present");
        let user_idx = merged
            .find("CONFIG_FOO=n")
            .expect("user CONFIG_FOO=n override must be present");
        assert!(
            baked_idx < user_idx,
            "baked-in CONFIG_FOO=y must precede user override CONFIG_FOO=n so \
             kbuild's last-wins rule picks the user value: {merged}"
        );
    }

    /// Disjoint fragments (user adds a symbol the baked-in fragment
    /// doesn't mention) combine verbatim — both lines reach kbuild
    /// untouched. Pins the non-conflict path so a refactor that
    /// dedupes or reorders user lines doesn't drop additive
    /// configuration.
    #[test]
    fn merge_kconfig_fragments_disjoint_symbols_both_present() {
        let baked = "CONFIG_FOO=y\n";
        let user = "CONFIG_DISJOINT_TEST_SYMBOL=m\n";
        let merged = merge_kconfig_fragments(baked, Some(user)).into_owned();
        assert!(
            merged.contains("CONFIG_FOO=y"),
            "baked symbol must survive merge: {merged}"
        );
        assert!(
            merged.contains("CONFIG_DISJOINT_TEST_SYMBOL=m"),
            "user-added disjoint symbol must survive merge: {merged}"
        );
    }

    // -- --extra-kconfig cache roundtrip / discrimination --
    //
    // These integration tests pin the cache-key behavior the
    // `--extra-kconfig` plumbing depends on. They use
    // `CacheDir::with_root` to plant fixture entries against an
    // isolated tempdir so the host's real cache (if any) does not
    // interfere, and exercise the production `cache.lookup` (the
    // same call site `cli::cache_lookup` consumes) directly. No
    // network, no kernel build — the bug surface is the cache-key
    // suffix machinery, and the tests target that surface
    // directly.

    /// Two consecutive lookups with the SAME `--extra-kconfig`
    /// content must hit the same cache slot. Pins the cache-hit
    /// branch of the roundtrip: identical extras content →
    /// identical `cache_key_suffix_with_extra` → identical cache
    /// key → planted entry retrieved.
    #[test]
    fn cache_lookup_same_extras_hits_planted_entry() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let extra = "CONFIG_KTSTR_CACHE_ROUNDTRIP_TEST_A=y\n";
        let extra_hash = extra_kconfig_hash(extra);
        let cache_key = format!("test-roundtrip-{}-xkc{}", kconfig_hash(), extra_hash);

        // Plant a fixture entry whose key matches what a build
        // with this extras content would produce.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"fake kernel image").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_extra_kconfig_hash(Some(extra_hash.clone()));
        cache
            .store(&cache_key, &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Look up the same key — must hit.
        let hit = cache.lookup(&cache_key);
        assert!(
            hit.is_some(),
            "cache lookup with same extras must return planted entry; \
             cache_key={cache_key}"
        );
        assert_eq!(
            hit.as_ref().unwrap().metadata.extra_kconfig_hash.as_deref(),
            Some(extra_hash.as_str()),
            "retrieved entry must carry the planted extra_kconfig_hash"
        );
    }

    /// Different `--extra-kconfig` contents must land at distinct
    /// cache slots — a build with extras=A must NOT serve a cached
    /// entry produced with extras=B. Pins the cache-discrimination
    /// branch: distinct extras → distinct hashes → distinct keys.
    #[test]
    fn cache_lookup_different_extras_misses_planted_entry() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let extra_a = "CONFIG_KTSTR_CACHE_DISCRIMINATE_A=y\n";
        let extra_b = "CONFIG_KTSTR_CACHE_DISCRIMINATE_B=y\n";
        let key_a = format!("test-disc-{}-xkc{}", kconfig_hash(), extra_kconfig_hash(extra_a));
        let key_b = format!("test-disc-{}-xkc{}", kconfig_hash(), extra_kconfig_hash(extra_b));
        assert_ne!(
            key_a, key_b,
            "extras A and B must produce distinct cache keys (precondition)"
        );

        // Plant entry under key_a only.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"fake kernel image A").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_extra_kconfig_hash(Some(extra_kconfig_hash(extra_a)));
        cache
            .store(&key_a, &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Lookup with key_b must MISS — extras=B's slot is
        // different from extras=A's slot.
        let hit_b = cache.lookup(&key_b);
        assert!(
            hit_b.is_none(),
            "lookup with extras=B's key must miss when only extras=A is planted; \
             key_a={key_a} key_b={key_b}"
        );

        // Sanity: the planted entry IS reachable via key_a.
        assert!(
            cache.lookup(&key_a).is_some(),
            "planted entry must be reachable via its own key"
        );
    }

    /// A bare-suffix entry (built without `--extra-kconfig`) must
    /// NOT be served when an extras lookup runs — and conversely,
    /// an extras-suffix entry must NOT be served to a bare lookup.
    /// Both halves of the segregation are pinned because either
    /// regression silently mis-serves a kernel built against a
    /// different configuration to a build that asked for a
    /// distinct one.
    #[test]
    fn cache_lookup_bare_and_extras_keys_segregated() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let baked = kconfig_hash();
        let extra = "CONFIG_KTSTR_CACHE_SEGREGATE=y\n";
        let bare_key = format!("test-seg-{baked}");
        let extras_key = format!("test-seg-{baked}-xkc{}", extra_kconfig_hash(extra));
        assert_ne!(
            bare_key, extras_key,
            "bare and extras-suffix keys must be distinct (precondition)"
        );

        // Plant only the bare entry.
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"bare kernel").unwrap();
        let bare_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        // bare_meta has extra_kconfig_hash=None by default.
        assert!(
            bare_meta.extra_kconfig_hash.is_none(),
            "bare entry fixture must not carry extras hash"
        );
        cache
            .store(&bare_key, &CacheArtifacts::new(&image), &bare_meta)
            .unwrap();

        // An extras lookup must not reach the bare entry.
        assert!(
            cache.lookup(&extras_key).is_none(),
            "extras lookup must NOT serve the bare entry — operator built with \
             --extra-kconfig and would silently get a kernel without their \
             user symbols if this regressed"
        );

        // Now plant an extras entry.
        let extras_image = src_dir.path().join("bzImage-extras");
        std::fs::write(&extras_image, b"extras kernel").unwrap();
        let extras_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-13T10:00:00Z".to_string(),
        )
        .with_extra_kconfig_hash(Some(extra_kconfig_hash(extra)));
        cache
            .store(&extras_key, &CacheArtifacts::new(&extras_image), &extras_meta)
            .unwrap();

        // Both keys must now hit their own slot independently.
        let bare_hit = cache.lookup(&bare_key).expect("bare entry");
        let extras_hit = cache.lookup(&extras_key).expect("extras entry");
        assert!(
            bare_hit.metadata.extra_kconfig_hash.is_none(),
            "bare entry must report None extras hash"
        );
        assert!(
            extras_hit.metadata.extra_kconfig_hash.is_some(),
            "extras entry must report Some(hash)"
        );
    }

    /// `CacheEntry::has_extra_kconfig()` must return true for an
    /// entry built with `--extra-kconfig` and false for a bare
    /// entry. Pins the metadata-readback contract that drives the
    /// `(extra kconfig)` tag in `kernel list` output — operators
    /// inspecting their cache need to distinguish at-a-glance
    /// which entries carry user modifications.
    #[test]
    fn cache_entry_has_extra_kconfig_reflects_metadata() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

        let _env_lock = lock_env();
        let _kernel_guard = EnvVarGuard::remove("KTSTR_KERNEL");
        let tmp = tempfile::TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &cache_root);

        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = tempfile::TempDir::new().unwrap();
        let image = src_dir.path().join("bzImage");
        std::fs::write(&image, b"img").unwrap();

        // Bare entry: extra_kconfig_hash = None.
        let bare_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        let bare = cache
            .store("test-has-bare", &CacheArtifacts::new(&image), &bare_meta)
            .unwrap();
        assert!(
            !bare.has_extra_kconfig(),
            "bare entry (extra_kconfig_hash = None) must report has_extra_kconfig() = false"
        );

        // Extras entry: extra_kconfig_hash = Some(hash).
        let extras_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-13T10:00:00Z".to_string(),
        )
        .with_extra_kconfig_hash(Some("deadbeef".to_string()));
        let extras = cache
            .store("test-has-extras", &CacheArtifacts::new(&image), &extras_meta)
            .unwrap();
        assert!(
            extras.has_extra_kconfig(),
            "entry with extra_kconfig_hash = Some(...) must report has_extra_kconfig() = true"
        );
    }
}
