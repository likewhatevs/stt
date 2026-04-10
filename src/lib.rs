//! Test harness for Linux process schedulers, with a focus on sched_ext.
//!
//! ktstr runs scheduler test scenarios inside lightweight KVM virtual machines
//! with controlled CPU topologies. Each test creates cgroups, spawns worker
//! processes, and verifies that the scheduler handled the workload correctly.
//! Also tests under the kernel's default EEVDF scheduler.
//!
//! # Quick start
//!
//! Declare cgroups and workloads as data, let the framework handle
//! lifecycle and verification:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(sockets = 1, cores = 2, threads = 1)]
//! fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
//!     execute_defs(ctx, vec![
//!         CgroupDef::named("cg_0").workers(2),
//!         CgroupDef::named("cg_1").workers(2),
//!     ])
//! }
//! ```
//!
//! For multi-phase scenarios with dynamic topology changes:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(sockets = 1, cores = 2, threads = 1)]
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
//! Use `#[derive(Scheduler)]` to declare a scheduler with typed flags
//! and a default topology. Tests reference the generated const and
//! inherit its configuration:
//!
//! ```rust
//! use ktstr::prelude::*;
//!
//! #[derive(Scheduler)]
//! #[scheduler(name = "my_sched", binary = "scx_my_sched", topology(2, 4, 1))]
//! #[allow(dead_code)]
//! enum MySchedFlag {
//!     #[flag(args = ["--enable-llc"])]
//!     Llc,
//!     #[flag(args = ["--enable-stealing"], requires = [Llc])]
//!     Steal,
//! }
//!
//! #[ktstr_test(scheduler = MY_SCHED)]
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
//! #[ktstr_test(sockets = 1, cores = 2, threads = 1)]
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
//! # Crate organization
//!
//! - [`cgroup`] -- cgroup v2 filesystem operations
//! - [`scenario`] -- test case definitions, flag system, cgroup helpers
//! - [`scenario::scenarios`] -- curated canned scenarios for common patterns
//! - [`mod@assert`] -- pass/fail assertions (starvation, isolation, fairness)
//! - [`workload`] -- worker process types and telemetry collection
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`verifier`] -- BPF verifier complexity analysis
//! - [`test_support`] -- `#[ktstr_test]` runtime and registration

#[allow(
    clippy::all,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod bpf_skel;

pub mod cgroup;

/// Map a raw errno value to its C constant name.
///
/// Covers the errno values most commonly seen in cgroup, KVM, and
/// scheduler error paths. Returns `None` for unrecognized values.
pub(crate) fn errno_name(errno: i32) -> Option<&'static str> {
    match errno {
        libc::EPERM => Some("EPERM"),
        libc::ENOENT => Some("ENOENT"),
        libc::ESRCH => Some("ESRCH"),
        libc::EINTR => Some("EINTR"),
        libc::EIO => Some("EIO"),
        libc::ENXIO => Some("ENXIO"),
        libc::E2BIG => Some("E2BIG"),
        libc::ENOEXEC => Some("ENOEXEC"),
        libc::EBADF => Some("EBADF"),
        libc::ECHILD => Some("ECHILD"),
        libc::EAGAIN => Some("EAGAIN"),
        libc::ENOMEM => Some("ENOMEM"),
        libc::EACCES => Some("EACCES"),
        libc::EFAULT => Some("EFAULT"),
        libc::EBUSY => Some("EBUSY"),
        libc::EEXIST => Some("EEXIST"),
        libc::ENODEV => Some("ENODEV"),
        libc::ENOTDIR => Some("ENOTDIR"),
        libc::EISDIR => Some("EISDIR"),
        libc::EINVAL => Some("EINVAL"),
        libc::ENFILE => Some("ENFILE"),
        libc::EMFILE => Some("EMFILE"),
        libc::ENOSPC => Some("ENOSPC"),
        libc::ESPIPE => Some("ESPIPE"),
        libc::EROFS => Some("EROFS"),
        libc::EPIPE => Some("EPIPE"),
        libc::EDOM => Some("EDOM"),
        libc::ERANGE => Some("ERANGE"),
        libc::EDEADLK => Some("EDEADLK"),
        libc::ENAMETOOLONG => Some("ENAMETOOLONG"),
        libc::ENOSYS => Some("ENOSYS"),
        libc::ENOTEMPTY => Some("ENOTEMPTY"),
        libc::ELOOP => Some("ELOOP"),
        // EWOULDBLOCK == EAGAIN on Linux, covered above
        libc::ENOTSUP => Some("ENOTSUP"),
        libc::EADDRINUSE => Some("EADDRINUSE"),
        libc::ECONNREFUSED => Some("ECONNREFUSED"),
        libc::ETIMEDOUT => Some("ETIMEDOUT"),
        _ => None,
    }
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
#[allow(dead_code)]
pub(crate) mod monitor;
#[allow(dead_code)]
pub(crate) mod probe;
#[allow(dead_code)]
pub(crate) mod runner;
pub mod scenario;
#[allow(dead_code)]
pub(crate) mod stats;
pub mod test_support;
#[allow(dead_code)]
pub(crate) mod timeline;
pub mod topology;

pub mod verifier;
#[allow(dead_code)]
pub(crate) mod vm;
#[allow(dead_code)]
pub(crate) mod vmm;
pub mod workload;

#[doc(hidden)]
pub use linkme as __linkme;

pub use ktstr_macros::Scheduler;
pub use ktstr_macros::ktstr_test;

#[cfg(feature = "integration")]
pub use crate::probe::process::resolve_func_ip;

/// Re-exports for writing `#[ktstr_test]` functions.
///
/// ```rust
/// use ktstr::prelude::*;
///
/// #[ktstr_test(sockets = 1, cores = 2, threads = 1)]
/// fn my_test(ctx: &Ctx) -> Result<AssertResult> {
///     Ok(AssertResult::pass())
/// }
/// ```
///
/// For curated canned scenarios, see [`scenario::scenarios`].
pub mod prelude {
    pub use anyhow::Result;

    pub use crate::Scheduler;
    pub use crate::assert::{Assert, AssertResult};
    pub use crate::cgroup::CgroupManager;
    pub use crate::ktstr_test;
    pub use crate::scenario::flags::FlagDecl;
    pub use crate::scenario::ops::{
        CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_defs, execute_steps, execute_steps_with,
    };
    pub use crate::scenario::scenarios;
    pub use crate::scenario::{CgroupGroup, Ctx, collect_all, spawn_diverse};
    pub use crate::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};
    pub use crate::topology::{LlcInfo, TestTopology};
    pub use crate::workload::{
        AffinityKind, AffinityMode, Phase, SchedPolicy, Work, WorkType, WorkerReport,
        WorkloadConfig, WorkloadHandle,
    };
}

/// Find a bootable kernel image on the host.
///
/// Checks build-tree paths first, then versioned paths
/// (`/lib/modules/$(uname -r)/vmlinuz`, `/boot/vmlinuz-$(uname -r)`),
/// then falls back to the unversioned `/boot/vmlinuz` symlink.
pub fn find_kernel() -> Option<std::path::PathBuf> {
    let rel = {
        let mut buf: libc::utsname = unsafe { std::mem::zeroed() };
        if unsafe { libc::uname(&mut buf) } == 0 {
            let cstr = unsafe { std::ffi::CStr::from_ptr(buf.release.as_ptr()) };
            Some(cstr.to_string_lossy().into_owned())
        } else {
            None
        }
    };

    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // Build-tree kernels (for development)
    #[cfg(target_arch = "x86_64")]
    {
        candidates.push("./linux/arch/x86/boot/bzImage".into());
        candidates.push("../linux/arch/x86/boot/bzImage".into());
    }
    #[cfg(target_arch = "aarch64")]
    {
        candidates.push("./linux/arch/arm64/boot/Image".into());
        candidates.push("../linux/arch/arm64/boot/Image".into());
    }

    // Versioned paths (usually world-readable)
    if let Some(ref r) = rel {
        candidates.push(format!("/lib/modules/{r}/vmlinuz").into());
        candidates.push(format!("/boot/vmlinuz-{r}").into());
    }

    // Unversioned fallback
    candidates.push("/boot/vmlinuz".into());

    candidates
        .into_iter()
        .find(|p| std::fs::File::open(p).is_ok())
}

/// Build a cargo binary package and return its output path.
pub fn build_and_find_binary(package: &str) -> anyhow::Result<std::path::PathBuf> {
    let output = std::process::Command::new("cargo")
        .args(["build", "-p", package, "--message-format=json"])
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
}
