//! Runtime configuration primitives shared by `eval` and `probe`.
//!
//! These items live here instead of either `eval` or `probe` so the
//! two siblings do not form a circular import chain. `eval` calls
//! `probe::attempt_auto_repro` from its failure path; `probe`
//! previously borrowed `verbose`, `KTSTR_TEST_SHM_SIZE`, and
//! `config_file_parts` back from `eval`. Hoisting those three into a
//! neutral module breaks the cycle without introducing any new
//! concept — the items remain internal to `test_support`.

use std::path::{Path, PathBuf};

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
    let config_path = entry.scheduler.config_file?;
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
    for s in entry.scheduler.sysctls {
        parts.push(format!("sysctl.{}={}", s.key, s.value));
    }
    for &karg in entry.scheduler.kargs {
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

/// Resolve `(vm_topology, memory_mb)` from an optional TopoOverride.
/// When absent, derives memory from the entry's declared topology
/// using the 64 MB-per-CPU floor (clamped to at least 256 and at
/// least `entry.memory_mb`). Shared with `attempt_auto_repro` so the
/// repro VM always sizes memory the same way as the first VM.
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
    if let Some(ref cgroup_path) = entry.scheduler.cgroup_parent {
        args.push("--cell-parent-cgroup".to_string());
        args.push(cgroup_path.to_string());
    }
    args.extend(entry.scheduler.sched_args.iter().map(|s| s.to_string()));
    args.extend(entry.extra_sched_args.iter().map(|s| s.to_string()));
}

#[cfg(test)]
mod tests {
    use super::super::entry::Scheduler;
    use super::*;

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
}
