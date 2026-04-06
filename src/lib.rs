//! Test harness for Linux process schedulers, with a focus on sched_ext.
//!
//! stt runs scheduler test scenarios inside lightweight KVM virtual machines
//! with controlled CPU topologies. Each test creates cgroups, spawns worker
//! processes, and verifies that the scheduler handled the workload correctly.
//! Also tests under the kernel's default EEVDF scheduler.
//!
//! # Crate organization
//!
//! - [`cgroup`] -- cgroup v2 filesystem operations
//! - [`scenario`] -- test case definitions, flag system, cgroup helpers
//! - [`runner`] -- scenario execution engine with scheduler lifecycle
//! - [`verify`] -- pass/fail evaluation (starvation, isolation, fairness)
//! - [`workload`] -- worker process types and telemetry collection
//! - [`monitor`] -- host-side guest memory observation via BTF
//! - [`topology`] -- CPU topology abstraction (LLCs, NUMA nodes)
//! - [`vm`] -- VM launch configuration and gauntlet presets
//! - [`vmm`] -- KVM virtual machine monitor implementation
//! - [`test_support`] -- `#[stt_test]` runtime and registration
//! - [`probe`] -- crash investigation via BPF kprobes
//! - [`stats`] -- gauntlet analysis and baseline comparison
//! - [`timeline`] -- stimulus/phase correlation

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
pub mod monitor;
pub mod probe;
pub mod runner;
pub mod scenario;
pub mod stats;
pub mod test_support;
pub mod timeline;
pub mod topology;
pub mod verify;
pub mod vm;
pub mod vmm;
pub mod workload;

pub use stt_macros::stt_test;

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
    candidates.push("./linux/arch/x86/boot/bzImage".into());
    candidates.push("../linux/arch/x86/boot/bzImage".into());

    // Versioned paths (usually world-readable)
    if let Some(ref r) = rel {
        candidates.push(format!("/lib/modules/{r}/vmlinuz").into());
        candidates.push(format!("/boot/vmlinuz-{r}").into());
    }

    // Unversioned fallback
    candidates.push("/boot/vmlinuz".into());

    candidates.into_iter().find(|p| p.exists())
}

/// Resolve the current executable path, falling back to `/proc/self/exe`
/// when the binary has been deleted (e.g. by `cargo llvm-cov`).
///
/// On Linux, `std::env::current_exe()` reads `/proc/self/exe`.  When the
/// binary is unlinked while running, the kernel appends ` (deleted)` to
/// the readlink target, producing a path that does not exist on disk.
/// `/proc/self/exe` itself remains usable as a file path because the
/// kernel keeps the inode alive, so we fall back to it.
pub fn resolve_current_exe() -> anyhow::Result<std::path::PathBuf> {
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
