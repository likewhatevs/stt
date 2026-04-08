//! Test harness for Linux process schedulers, with a focus on sched_ext.
//!
//! stt runs scheduler test scenarios inside lightweight KVM virtual machines
//! with controlled CPU topologies. Each test creates cgroups, spawns worker
//! processes, and verifies that the scheduler handled the workload correctly.
//! Also tests under the kernel's default EEVDF scheduler.
//!
//! # Quick start
//!
//! Write a test that boots a VM, creates a cgroup, runs a workload, and
//! checks the result:
//!
//! ```rust
//! use stt::prelude::*;
//! use std::collections::BTreeSet;
//!
//! #[stt_test(sockets = 1, cores = 2, threads = 1)]
//! fn my_scheduler_test(ctx: &Ctx) -> Result<AssertResult> {
//!     // Create a cgroup and assign all CPUs.
//!     let mut group = CgroupGroup::new(ctx.cgroups);
//!     group.add_cgroup_no_cpuset("workers")?;
//!     let cpus: BTreeSet<usize> = ctx.topo.all_cpus().iter().copied().collect();
//!     ctx.cgroups.set_cpuset("workers", &cpus)?;
//!
//!     // Spawn workers into the cgroup.
//!     let cfg = WorkloadConfig {
//!         num_workers: 2,
//!         work_type: WorkType::CpuSpin,
//!         ..Default::default()
//!     };
//!     let mut handle = WorkloadHandle::spawn(&cfg)?;
//!     for tid in handle.tids() {
//!         ctx.cgroups.move_task("workers", tid)?;
//!     }
//!     handle.start();
//!
//!     // Let workers run, then collect results.
//!     std::thread::sleep(ctx.duration);
//!     let reports = handle.stop_and_collect();
//!
//!     // Assert: no worker was starved.
//!     let plan = AssertPlan::new().check_not_starved();
//!     Ok(plan.assert_cgroup(&reports, None))
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
//! - [`assert`] -- pass/fail assertions (starvation, isolation, fairness)
//! - [`workload`] -- worker process types and telemetry collection
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`verifier`] -- BPF verifier complexity analysis
//! - [`test_support`] -- `#[stt_test]` runtime and registration

#[allow(
    clippy::all,
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
mod bpf_skel;

pub mod cgroup;

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

pub use stt_macros::stt_test;

#[cfg(feature = "integration")]
pub use crate::probe::process::resolve_func_ip;

/// Re-exports for writing `#[stt_test]` functions.
///
/// ```rust
/// use stt::prelude::*;
///
/// #[stt_test(sockets = 1, cores = 2, threads = 1)]
/// fn my_test(ctx: &Ctx) -> Result<AssertResult> {
///     Ok(AssertResult::pass())
/// }
/// ```
///
/// For curated canned scenarios, see [`scenario::scenarios`].
pub mod prelude {
    pub use anyhow::Result;

    pub use crate::assert::{Assert, AssertPlan, AssertResult};
    pub use crate::cgroup::CgroupManager;
    pub use crate::scenario::ops::{
        CgroupDef, CpusetSpec, HoldSpec, Op, Step, execute_steps, execute_steps_with,
    };
    pub use crate::scenario::scenarios;
    pub use crate::scenario::{CgroupGroup, Ctx};
    pub use crate::stt_test;
    pub use crate::test_support::{BpfMapWrite, Scheduler, SchedulerSpec};
    pub use crate::workload::{
        AffinityMode, SchedPolicy, WorkProgram, WorkType, WorkerReport, WorkloadConfig,
        WorkloadHandle,
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
