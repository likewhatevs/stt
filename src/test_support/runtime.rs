//! Runtime configuration primitives shared by `eval` and `probe`.
//!
//! `eval` calls `probe::attempt_auto_repro` from its failure path,
//! so items shared between the two siblings live here to avoid a
//! circular import chain. All items are `pub(crate)` and remain
//! internal to `test_support`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::entry::KtstrTestEntry;

/// True when `RUST_BACKTRACE` is set to `"1"` or `"full"`.
///
/// Controls whether the full guest kernel console is appended to the
/// `--- diagnostics ---` section of a failed test, and whether
/// auto-repro forwards the repro VM's COM1/COM2 output to the host
/// terminal in real time. The scheduler-log and sched_ext-dump
/// sections of a failure are always emitted regardless of this flag.
pub(crate) fn verbose() -> bool {
    std::env::var("RUST_BACKTRACE")
        .map(|v| v == "1" || v == "full")
        .unwrap_or(false)
}

/// SHM size for ktstr_test VMs: 16 MB. Sized for profraw (1-2 MB),
/// stimulus events, exit code, and test results with mid-flight drain
/// headroom.
pub(crate) const KTSTR_TEST_SHM_SIZE: u64 = 16 * 1024 * 1024;

/// Derive initramfs archive path, host path, and guest path from a
/// scheduler's `config_file`. Returns `None` when no config file is set.
pub(crate) fn config_file_parts(entry: &KtstrTestEntry) -> Option<(String, PathBuf, String)> {
    let config_path = entry.scheduler.config_file()?;
    let file_name = Path::new(config_path)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("config_file must have a valid filename");
    let archive_path = format!("include-files/{file_name}");
    let guest_path = format!("/include-files/{file_name}");
    Some((archive_path, PathBuf::from(config_path), guest_path))
}

/// Build the shared `cmdline=` string appended to every ktstr_test
/// guest boot. `iomem=relaxed`, per-scheduler sysctls, per-scheduler
/// kargs, and `RUST_BACKTRACE` / `RUST_LOG` propagation — anything
/// the two VM-launch sites (`run_ktstr_test_inner` and
/// `attempt_auto_repro`) previously re-implemented side-by-side.
pub(crate) fn build_cmdline_extra(entry: &KtstrTestEntry) -> String {
    let mut parts = vec!["iomem=relaxed".to_string()];
    for s in entry.scheduler.sysctls() {
        parts.push(format!("sysctl.{}={}", s.key, s.value));
    }
    for &karg in entry.scheduler.kargs() {
        parts.push(karg.to_string());
    }
    if let Ok(bt) = std::env::var("RUST_BACKTRACE") {
        parts.push(format!("RUST_BACKTRACE={bt}"));
    }
    if let Ok(log) = std::env::var("RUST_LOG") {
        parts.push(format!("RUST_LOG={log}"));
    }
    parts.join(" ")
}

/// Resolve the VM topology and memory size from an optional
/// TopoOverride.
///
/// Returns `(topology, memory_mb)` where `topology` is the
/// `vmm::topology::Topology` passed to the VM builder and `memory_mb`
/// is the memory allocation in megabytes. When `topo` is `Some`, both
/// come from the override. When `topo` is `None`, the topology comes
/// from `entry.topology` and memory is `max(total_cpus * 64, 256,
/// entry.memory_mb)`. Shared with `attempt_auto_repro` so the repro
/// VM always sizes memory the same way as the first VM.
pub(crate) fn resolve_vm_topology(
    entry: &KtstrTestEntry,
    topo: Option<&super::topo::TopoOverride>,
) -> (crate::vmm::topology::Topology, u32) {
    match topo {
        Some(t) => (crate::vmm::topology::Topology::from(t), t.memory_mb),
        None => {
            let cpus = entry.topology.total_cpus();
            let mem = (cpus * 64).max(256).max(entry.memory_mb);
            (entry.topology, mem)
        }
    }
}

/// Append per-scheduler `sched_args` entries shared by both VM-launch
/// paths: `--config <guest_path>` if the scheduler declared one, the
/// cgroup-parent switch, the scheduler's own fixed args, and
/// per-entry extra args. Active-flag dispatch and probe-specific args
/// remain at the call site because they differ between the paths.
///
/// The caller owns the `include_files` binding on the builder;
/// `config_file_parts` and the guest-path push are returned separately
/// so the caller decides whether to attach include files (production
/// does, probe-only repro pipelines that already pass `include_files`
/// can skip it).
pub(crate) fn append_base_sched_args(entry: &KtstrTestEntry, args: &mut Vec<String>) {
    if let Some(cgroup_path) = entry.scheduler.cgroup_parent() {
        args.push("--cell-parent-cgroup".to_string());
        args.push(cgroup_path.to_string());
    }
    args.extend(entry.scheduler.sched_args().iter().map(|s| s.to_string()));
    args.extend(entry.extra_sched_args.iter().map(|s| s.to_string()));
}

/// Host-side watchdog applied to every ktstr_test VM. Kept as a
/// single const so a bump in one path cannot drift against the
/// auto-repro path.
pub(crate) const KTSTR_VM_TIMEOUT: Duration = Duration::from_secs(60);

/// Configure the ktstr_test VM builder prefix shared by the main
/// test path ([`super::eval::run_ktstr_test_inner`]) and the
/// auto-repro path ([`super::probe::attempt_auto_repro`]).
///
/// Applies, in order: kernel, init binary, topology, memory floor,
/// guest cmdline, SHM size, guest argv, host-side timeout, perf-mode
/// disable flag, optional scheduler binary, every queued BPF map
/// write, and the scheduler watchdog timeout.
///
/// The caller owns the divergent tail. `run_ktstr_test_inner`
/// additionally wires `performance_mode`,
/// `sched_enable_cmds`/`sched_disable_cmds` for kernel-built
/// schedulers, `monitor_thresholds`, and `sched_args` extended with
/// active-flag mappings from `entry.scheduler.flag_args`.
/// `attempt_auto_repro` additionally wires `include_files` plus
/// base `sched_args` (no active-flag extension).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_vm_builder_base(
    entry: &KtstrTestEntry,
    kernel: &Path,
    ktstr_bin: &Path,
    scheduler: Option<&Path>,
    vm_topology: crate::vmm::topology::Topology,
    memory_mb: u32,
    cmdline_extra: &str,
    guest_args: &[String],
    no_perf_mode: bool,
) -> crate::vmm::KtstrVmBuilder {
    let mut builder = crate::vmm::KtstrVm::builder()
        .kernel(kernel)
        .init_binary(ktstr_bin)
        .with_topology(vm_topology)
        .memory_deferred_min(memory_mb)
        .cmdline(cmdline_extra)
        .shm_size(KTSTR_TEST_SHM_SIZE)
        .run_args(guest_args)
        .timeout(KTSTR_VM_TIMEOUT)
        .no_perf_mode(no_perf_mode);

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }

    for bpf_write in entry.bpf_map_write {
        builder =
            builder.bpf_map_write(bpf_write.map_name_suffix, bpf_write.offset, bpf_write.value);
    }

    builder.watchdog_timeout(entry.watchdog_timeout)
}

#[cfg(test)]
mod tests {
    use super::super::entry::Scheduler;
    use super::super::payload::Payload;
    use super::*;

    #[test]
    fn config_file_parts_nested_path() {
        static SCHED: Scheduler = Scheduler::new("cfg").config_file("configs/my_sched.toml");
        static SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "cfg_test",
            scheduler: &SCHED_PAYLOAD,
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
        static SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "cfg_bare",
            scheduler: &SCHED_PAYLOAD,
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

    // -- build_cmdline_extra --

    use super::super::entry::{KtstrTestEntry, Sysctl};
    use super::super::test_helpers::ENV_LOCK;

    #[test]
    fn build_cmdline_extra_includes_iomem_relaxed_by_default() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Make sure the env does not inject spurious RUST_BACKTRACE /
        // RUST_LOG entries that would break the default assertion.
        let prev_bt = std::env::var("RUST_BACKTRACE").ok();
        let prev_log = std::env::var("RUST_LOG").ok();
        // SAFETY: test holds ENV_LOCK — no other thread may mutate env concurrently.
        unsafe {
            std::env::remove_var("RUST_BACKTRACE");
            std::env::remove_var("RUST_LOG");
        }

        let entry = KtstrTestEntry {
            name: "cmdline_test",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(out, "iomem=relaxed");

        // SAFETY: single-threaded test restoration under ENV_LOCK.
        unsafe {
            match prev_bt {
                Some(v) => std::env::set_var("RUST_BACKTRACE", v),
                None => std::env::remove_var("RUST_BACKTRACE"),
            }
            match prev_log {
                Some(v) => std::env::set_var("RUST_LOG", v),
                None => std::env::remove_var("RUST_LOG"),
            }
        }
    }

    #[test]
    fn build_cmdline_extra_appends_sysctls_kargs() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_bt = std::env::var("RUST_BACKTRACE").ok();
        let prev_log = std::env::var("RUST_LOG").ok();
        // SAFETY: test holds ENV_LOCK.
        unsafe {
            std::env::remove_var("RUST_BACKTRACE");
            std::env::remove_var("RUST_LOG");
        }

        static SYSCTLS: &[Sysctl] = &[Sysctl::new("kernel.foo", "1")];
        static SCHED: Scheduler = Scheduler::new("s").sysctls(SYSCTLS).kargs(&["quiet"]);
        static SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "cmd",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(out, "iomem=relaxed sysctl.kernel.foo=1 quiet");

        // SAFETY: single-threaded test restoration under ENV_LOCK.
        unsafe {
            match prev_bt {
                Some(v) => std::env::set_var("RUST_BACKTRACE", v),
                None => std::env::remove_var("RUST_BACKTRACE"),
            }
            match prev_log {
                Some(v) => std::env::set_var("RUST_LOG", v),
                None => std::env::remove_var("RUST_LOG"),
            }
        }
    }

    #[test]
    fn build_cmdline_extra_propagates_rust_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let prev_bt = std::env::var("RUST_BACKTRACE").ok();
        let prev_log = std::env::var("RUST_LOG").ok();
        // SAFETY: test holds ENV_LOCK.
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "1");
            std::env::set_var("RUST_LOG", "debug");
        }

        let entry = KtstrTestEntry {
            name: "cmd",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert!(
            out.contains("RUST_BACKTRACE=1"),
            "expected RUST_BACKTRACE propagation: {out}"
        );
        assert!(
            out.contains("RUST_LOG=debug"),
            "expected RUST_LOG propagation: {out}"
        );

        // SAFETY: single-threaded test restoration under ENV_LOCK.
        unsafe {
            match prev_bt {
                Some(v) => std::env::set_var("RUST_BACKTRACE", v),
                None => std::env::remove_var("RUST_BACKTRACE"),
            }
            match prev_log {
                Some(v) => std::env::set_var("RUST_LOG", v),
                None => std::env::remove_var("RUST_LOG"),
            }
        }
    }

    // -- resolve_vm_topology --

    #[test]
    fn resolve_vm_topology_override_is_verbatim() {
        let entry = KtstrTestEntry {
            name: "topo_test",
            ..KtstrTestEntry::DEFAULT
        };
        let over = super::super::topo::TopoOverride {
            numa_nodes: 2,
            llcs: 4,
            cores: 8,
            threads: 2,
            memory_mb: 4096,
        };
        let (topo, mem) = resolve_vm_topology(&entry, Some(&over));
        assert_eq!(mem, 4096);
        assert_eq!(topo.llcs, 4);
        assert_eq!(topo.cores_per_llc, 8);
        assert_eq!(topo.threads_per_core, 2);
        assert_eq!(topo.numa_nodes, 2);
    }

    #[test]
    fn resolve_vm_topology_none_floors_memory_at_256() {
        // Tiny topology: 1*1*1=1 cpu -> 64 MB raw, entry.memory_mb=0,
        // floor = max(64, 256, 0) = 256.
        //
        // Override memory_mb explicitly to 0 — KtstrTestEntry::DEFAULT
        // sets memory_mb=2048, which would bypass the floor entirely
        // and leave this test vacuously passing regardless of the
        // max(…, 256, …) branch. Setting memory_mb=0 makes the 256
        // floor the exact lower bound the assertion verifies.
        let entry = KtstrTestEntry {
            name: "tiny",
            memory_mb: 0,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(mem, 256, "memory floor = 256 MB, got {mem}");
    }

    #[test]
    fn resolve_vm_topology_none_honors_entry_memory_mb() {
        // Entry with explicit memory_mb above the cpu*64 and 256 floors.
        let entry = KtstrTestEntry {
            name: "mem",
            memory_mb: 8192,
            ..KtstrTestEntry::DEFAULT
        };
        let (_topo, mem) = resolve_vm_topology(&entry, None);
        assert_eq!(mem, 8192);
    }

    // -- append_base_sched_args --

    #[test]
    fn append_base_sched_args_empty_when_none_set() {
        let entry = KtstrTestEntry {
            name: "nosched",
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert!(args.is_empty(), "no sched args expected: {args:?}");
    }

    #[test]
    fn append_base_sched_args_includes_cgroup_parent_and_sched_args() {
        use super::super::entry::CgroupPath;
        static CG: CgroupPath = CgroupPath::new("/sys/fs/cgroup/ktstr");
        static SCHED: Scheduler = Scheduler::new("s")
            .cgroup_parent("/sys/fs/cgroup/ktstr")
            .sched_args(&["-v", "--flag"]);
        static SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCHED);
        // Touch the static so the compiler doesn't drop it; verifies
        // the path we store matches what cgroup_parent produces.
        let _ = &CG;
        let entry = KtstrTestEntry {
            name: "sched",
            scheduler: &SCHED_PAYLOAD,
            extra_sched_args: &["--extra"],
            ..KtstrTestEntry::DEFAULT
        };
        let mut args = Vec::new();
        append_base_sched_args(&entry, &mut args);
        assert_eq!(
            args,
            vec![
                "--cell-parent-cgroup".to_string(),
                "/sys/fs/cgroup/ktstr".to_string(),
                "-v".to_string(),
                "--flag".to_string(),
                "--extra".to_string(),
            ],
        );
    }

    // -- build_vm_builder_base --

    /// Kernel-path surfaces in the builder's "kernel not found" error.
    /// Proves the `kernel()` setter is wired through the helper.
    #[test]
    fn build_vm_builder_base_propagates_kernel_path() {
        let entry = KtstrTestEntry {
            name: "vmb_kernel_path",
            ..KtstrTestEntry::DEFAULT
        };
        let exe = crate::resolve_current_exe().unwrap();
        let missing_kernel =
            PathBuf::from("/nonexistent/build_vm_builder_base_test_kernel.bzImage");
        let result = build_vm_builder_base(
            &entry,
            &missing_kernel,
            &exe,
            None,
            crate::vmm::topology::Topology::new(1, 1, 1, 1),
            256,
            "",
            &["run".to_string()],
            true,
        )
        .build();
        // `KtstrVm` does not implement Debug, so `.unwrap_err()` is not
        // available — collapse Ok into a panic to extract the error by hand.
        let err = match result {
            Ok(_) => panic!("builder.build() unexpectedly succeeded for missing kernel"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("kernel not found"),
            "expected kernel not found error, got: {msg}",
        );
        assert!(
            msg.contains("build_vm_builder_base_test_kernel"),
            "expected the fake kernel path to appear in the error, got: {msg}",
        );
    }

    /// A zero-`llcs` topology is forwarded to the builder and surfaces
    /// as a validation error. Proves `with_topology()` is wired through.
    #[test]
    fn build_vm_builder_base_propagates_topology_validation() {
        let entry = KtstrTestEntry {
            name: "vmb_topology",
            ..KtstrTestEntry::DEFAULT
        };
        let exe = crate::resolve_current_exe().unwrap();
        let bad_topology = crate::vmm::topology::Topology {
            llcs: 0,
            cores_per_llc: 1,
            threads_per_core: 1,
            numa_nodes: 1,
            nodes: None,
            distances: None,
        };
        let result = build_vm_builder_base(
            &entry,
            &exe,
            &exe,
            None,
            bad_topology,
            256,
            "",
            &["run".to_string()],
            true,
        )
        .build();
        let err = match result {
            Ok(_) => panic!("builder.build() unexpectedly succeeded for zero-llcs topology"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("llcs must be > 0"),
            "expected topology validation error, got: {msg}",
        );
    }

    /// An optional scheduler binary is attached when `Some(path)`
    /// is supplied, surfacing as a "scheduler binary not found"
    /// error when the path is missing.
    #[test]
    fn build_vm_builder_base_propagates_scheduler_binary() {
        let entry = KtstrTestEntry {
            name: "vmb_scheduler",
            ..KtstrTestEntry::DEFAULT
        };
        let exe = crate::resolve_current_exe().unwrap();
        let missing_scheduler =
            PathBuf::from("/nonexistent/build_vm_builder_base_test_scheduler");
        let result = build_vm_builder_base(
            &entry,
            &exe,
            &exe,
            Some(&missing_scheduler),
            crate::vmm::topology::Topology::new(1, 1, 1, 1),
            256,
            "",
            &["run".to_string()],
            true,
        )
        .build();
        let err = match result {
            Ok(_) => panic!("builder.build() unexpectedly succeeded for missing scheduler"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler binary not found"),
            "expected scheduler binary error, got: {msg}",
        );
        assert!(
            msg.contains("build_vm_builder_base_test_scheduler"),
            "expected the fake scheduler path to appear, got: {msg}",
        );
    }
}
