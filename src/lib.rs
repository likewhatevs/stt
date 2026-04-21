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
//!         work_type: WorkType::CpuSpin,
//!         ..Default::default()
//!     };
//!     let mut handle = WorkloadHandle::spawn(&cfg)?;
//!     ctx.cgroups.move_tasks("workers", &handle.tids())?;
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
//! - [`runner`] -- host-side scenario runner used by `ktstr run`
//! - [`scenario`] -- declarative ops API (`CgroupDef`, `Step`, `Op`, `Backdrop`, `execute_defs`, `execute_steps`, `execute_scenario`)
//! - [`scenario::scenarios`] -- curated canned scenarios for common patterns
//! - [`mod@assert`] -- pass/fail assertions (starvation, isolation, fairness)
//! - [`workload`] -- worker process types and telemetry collection
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`kernel_path`] -- kernel ID parsing and filesystem image discovery
//! - [`verifier`] -- BPF verifier log parsing, cycle detection, and output formatting
//! - [`test_support`] -- `#[ktstr_test]` runtime and registration
//! - [`fetch`] -- kernel tarball and git source acquisition
//! - [`remote_cache`] -- GitHub Actions cache integration
//!
//! Internal modules (not re-exported): `monitor` reads live guest
//! state, `probe` attaches BPF probes to traced functions, `vmm`
//! owns the KVM VM lifecycle, and `timeline` correlates stimulus
//! events with monitor samples for phase-aligned reporting.

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

pub mod cache;
pub mod cgroup;

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
pub(crate) fn read_kmsg() -> String {
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
pub(crate) mod elf_strip;
pub mod fetch;
pub mod kernel_path;
pub(crate) mod monitor;
pub(crate) mod probe;
pub mod runner;
pub mod scenario;
pub(crate) mod stats;
pub mod test_support;
pub(crate) mod timeline;
pub mod topology;

pub mod remote_cache;
pub mod verifier;
pub(crate) mod vm;
pub(crate) mod vmm;
pub mod workload;

/// Static busybox binary compiled in build.rs for guest shell mode.
pub(crate) const BUSYBOX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

/// Short git commit hash at ktstr build time, or "unknown" when
/// built without `.git` in the manifest dir (see build.rs).
pub const GIT_HASH: &str = env!("KTSTR_GIT_HASH");
/// Full git commit hash at ktstr build time, or "unknown" when
/// built without `.git` in the manifest dir (see build.rs).
pub const GIT_FULL_HASH: &str = env!("KTSTR_GIT_FULL_HASH");

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

/// Cache key suffix derived from the embedded kconfig fragment.
/// Used in kernel cache keys so a kconfig change produces a distinct
/// cache entry. The kernel binary is independent of ktstr userspace
/// source, so no ktstr or consumer build identity feeds this suffix.
pub fn cache_key_suffix() -> String {
    kconfig_hash()
}

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
    pub use crate::assert::{Assert, AssertResult};
    pub use crate::cgroup::CgroupManager;
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
        BpfMapWrite, CgroupPath, Check, MemSideCache, Metric, MetricHint, MetricSource,
        NumaDistance, NumaNode, OutputFormat, Payload, PayloadKind, PayloadMetrics, Polarity,
        Scheduler, SchedulerSpec, SidecarResult, Sysctl, Topology, extract_metrics,
    };
    pub use crate::{Payload, Scheduler};
    // `FlagDecl` is already re-exported via `crate::scenario::flags::FlagDecl`
    // above; omitted here because `test_support::FlagDecl` is the same item
    // (`pub use crate::scenario::flags::FlagDecl;` in test_support/mod.rs),
    // and a second prelude line would shadow harmlessly but add noise.
    //
    // The following items are intentionally NOT in the prelude. They
    // are binary-entry helpers (the `ktstr` / `cargo-ktstr` bins),
    // macro-generated glue the `#[ktstr_test]` expansion consumes, or
    // nextest-setup-script entry points — audiences distinct from the
    // test-author surface this module provides. Import directly from
    // `ktstr::test_support::<item>` when needed:
    // `newest_run_dir`, `runs_root`, `analyze_sidecars`, `ktstr_main`,
    // `ktstr_test_early_dispatch`, `run_ktstr_test`, `nextest_setup`,
    // `resolve_scheduler`, `resolve_test_kernel`.
    pub use crate::topology::{LlcInfo, NodeMemInfo, TestTopology};
    pub use crate::workload::{
        AffinityKind, AffinityMode, MemPolicy, MpolFlags, Phase, SchedPolicy, Work, WorkType,
        WorkerReport, WorkloadConfig, WorkloadHandle, build_nodemask,
    };
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

    // 1. KTSTR_KERNEL env var with KernelId parsing.
    if let Some(val) = std::env::var("KTSTR_KERNEL")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
    {
        match KernelId::parse(&val) {
            KernelId::Path(_) => match kernel_path::find_image(Some(&val), release_ref) {
                Some(p) => return Ok(Some(p)),
                None => anyhow::bail!("KTSTR_KERNEL={val} does not contain a kernel image"),
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
            if let cache::KconfigStatus::Stale { .. } = entry.kconfig_status(&kc_hash) {
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
        let _cache_guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.to_str().unwrap());

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
}
