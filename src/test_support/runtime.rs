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
/// guest boot. Per-scheduler sysctls, per-scheduler kargs,
/// `RUST_BACKTRACE` / `RUST_LOG` propagation, and the host-resolved
/// `KTSTR_SIDECAR_DIR` so the guest's `sidecar_dir()` returns the
/// SAME path the host's freeze coordinator writes to. Without that
/// propagation, host and guest each compute the run directory
/// independently — the host walks `gix::discover` from a real
/// workspace cwd and produces `{kernel}-{commit}` whereas the
/// guest's cwd is `/` (no git repo, no kernel env), yielding the
/// `unknown-unknown` fallback. Anything the two VM-launch sites
/// (`run_ktstr_test_inner` and `attempt_auto_repro`) previously
/// re-implemented side-by-side lives here.
pub(crate) fn build_cmdline_extra(entry: &KtstrTestEntry) -> String {
    let mut parts: Vec<String> = Vec::new();
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
    // Propagate the host-resolved sidecar dir so the guest scenario
    // computes the same path the host's freeze coordinator wrote to
    // (e.g. when a test reads `sidecar_dir().join("foo.json")` from
    // inside the guest, the path matches the host's writer site).
    // The host resolves via the OnceLock-cached project commit walk
    // from the workspace cwd; the guest's cwd is `/` and would
    // otherwise fall back to `unknown-unknown`. Sidecar dir paths
    // are filesystem-safe ASCII (kernel version + 7-char hex
    // commit, optional `-dirty` suffix), so the cmdline-as-token
    // shape is sound — no escaping needed for whitespace.
    //
    // Absolutize via `current_dir().join()` when the resolved path
    // is relative (the default-branch shape:
    // `target/ktstr/{kernel}-{commit}` against the host cwd). The
    // guest's cwd is `/`, so a relative token would resolve there
    // instead of at the host's workspace root — the propagation
    // must carry the FULL absolute path so the guest's
    // `sidecar_dir()` reports the same string the host's writer
    // site used. Falls back to the raw resolved path when the cwd
    // probe fails (extremely rare; happens only when the process's
    // cwd was rmdir'd while alive — a metadata probe has no
    // recourse, leave the path as-is).
    let resolved = super::sidecar::sidecar_dir();
    let absolute = if resolved.is_absolute() {
        resolved
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&resolved))
            .unwrap_or(resolved)
    };
    if let Some(s) = absolute.to_str() {
        parts.push(format!("KTSTR_SIDECAR_DIR={s}"));
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

/// Headroom added to the test's base duration to derive the
/// host-side VM kill timer. Must exceed the freeze-coordinator
/// rendezvous timeout (30s) plus dump render + probe collection
/// so the host doesn't kill the VM mid-dump.
const VM_TIMEOUT_HEADROOM: Duration = Duration::from_secs(45);

/// Derive the host-side VM timeout from the test entry's watchdog
/// and duration. The VM should die shortly after the stall fires
/// and the dump completes — not linger for a hardcoded 60s.
pub(crate) fn vm_timeout_from_entry(entry: &super::entry::KtstrTestEntry) -> Duration {
    let base = entry
        .watchdog_timeout
        .max(entry.duration)
        .max(Duration::from_secs(1));
    base + VM_TIMEOUT_HEADROOM
}

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
    // The base builder deliberately does NOT set
    // `failure_dump_path` — the per-VM target is caller-specific
    // (primary vs auto-repro). Stale-file pre-clear lives at the
    // dispatch sites (`test_support::eval` for primary;
    // `test_support::probe::attempt_auto_repro` for repro), not
    // inside the setter or this base call. The setter is pure
    // (no FS side effects); placing the pre-clear in the dispatch
    // layer prevents the auto-repro path's reuse of this base
    // builder from accidentally erasing the primary dump that
    // just landed.
    let mut builder = crate::vmm::KtstrVm::builder()
        .kernel(kernel)
        .init_binary(ktstr_bin)
        .with_topology(vm_topology)
        .memory_deferred_min(memory_mb)
        .cmdline(cmdline_extra)
        .run_args(guest_args)
        .timeout(vm_timeout_from_entry(entry))
        .no_perf_mode(no_perf_mode);

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }

    // Opt-in jemalloc-probe wiring. An integration test that needs
    // the probe (see `tests/jemalloc_probe_tests.rs`) sets
    // `KTSTR_JEMALLOC_PROBE_BINARY` to the absolute host path of
    // `ktstr-jemalloc-probe` via `#[ctor]` before the test harness
    // dispatches. When set, the probe is packed into every VM's
    // base initramfs; the init binary stays stripped because the
    // paired alloc-worker carries DWARF. Absent env var = existing
    // behavior (no probe).
    //
    // Required ctor shape in a new test file that needs the probe
    // in the guest — paste verbatim, adjust the two binary names:
    //
    // ```ignore
    // #[::ktstr::__private::ctor::ctor(crate_path = ::ktstr::__private::ctor)]
    // fn set_probe_binary_env_var() {
    //     // SAFETY: ctor runs before any `#[ktstr_test]` thread or
    //     // probe thread spawns; glibc's `__environ` mutation is
    //     // single-threaded here.
    //     unsafe {
    //         std::env::set_var(
    //             "KTSTR_JEMALLOC_PROBE_BINARY",
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-probe"),
    //         );
    //         std::env::set_var(
    //             "KTSTR_JEMALLOC_ALLOC_WORKER_BINARY",
    //             env!("CARGO_BIN_EXE_ktstr-jemalloc-alloc-worker"),
    //         );
    //     }
    // }
    // ```
    //
    // The `crate_path = ::ktstr::__private::ctor` argument is
    // non-negotiable: `#[ctor::ctor]` without the re-export path
    // panics at compile time because the `ctor` crate is not
    // listed in the test crate's direct deps. ktstr re-exports
    // `ctor` under `__private::ctor` exactly so test authors do
    // not need to add it themselves.
    if let Ok(probe_path) = std::env::var("KTSTR_JEMALLOC_PROBE_BINARY")
        && !probe_path.is_empty()
    {
        // Pack the probe binary into the guest initramfs at
        // `/bin/ktstr-jemalloc-probe`. Closed-loop probe tests run
        // the probe via `--pid <alloc_worker_pid>` against the
        // paired `ktstr-jemalloc-alloc-worker` target; DWARF comes
        // from the worker's own ELF, not the init's.
        builder = builder.jemalloc_probe_binary(std::path::PathBuf::from(probe_path));
    }
    if let Ok(worker_path) = std::env::var("KTSTR_JEMALLOC_ALLOC_WORKER_BINARY")
        && !worker_path.is_empty()
    {
        // Pack the jemalloc-alloc-worker binary alongside the
        // probe. Only the cross-process closed-loop test sets
        // this; scheduler-only tests leave the env var unset and
        // skip the wiring.
        builder = builder.jemalloc_alloc_worker_binary(std::path::PathBuf::from(worker_path));
    }

    for bpf_write in entry.bpf_map_write {
        builder =
            builder.bpf_map_write(bpf_write.map_name_suffix, bpf_write.offset, bpf_write.value);
    }

    if let Some(disk_cfg) = entry.disk.clone() {
        builder = builder.disk(disk_cfg);
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
    use super::super::test_helpers::{EnvVarGuard, lock_env};

    #[test]
    fn build_cmdline_extra_default_is_sidecar_only() {
        let _lock = lock_env();
        // Make sure the env does not inject spurious RUST_BACKTRACE /
        // RUST_LOG entries that would break the default assertion.
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        // Pin KTSTR_SIDECAR_DIR so the propagation token shape is
        // stable across tests; without the override, the call falls
        // through to the `{kernel}-{commit}` resolver whose output
        // depends on the test process's git state.
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/tmp/ktstr-test");

        let entry = KtstrTestEntry {
            name: "cmdline_test",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(out, "KTSTR_SIDECAR_DIR=/tmp/ktstr-test");
    }

    #[test]
    fn build_cmdline_extra_appends_sysctls_kargs() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/tmp/ktstr-test");

        static SYSCTLS: &[Sysctl] = &[Sysctl::new("kernel.foo", "1")];
        static SCHED: Scheduler = Scheduler::new("s").sysctls(SYSCTLS).kargs(&["quiet"]);
        static SCHED_PAYLOAD: Payload = Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "cmd",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(
            out,
            "sysctl.kernel.foo=1 quiet KTSTR_SIDECAR_DIR=/tmp/ktstr-test"
        );
    }

    #[test]
    fn build_cmdline_extra_propagates_rust_env() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::set("RUST_BACKTRACE", "1");
        let _env_log = EnvVarGuard::set("RUST_LOG", "debug");
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/tmp/ktstr-test");

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
        assert!(
            out.contains("KTSTR_SIDECAR_DIR=/tmp/ktstr-test"),
            "expected KTSTR_SIDECAR_DIR propagation: {out}"
        );
    }

    #[test]
    fn build_cmdline_extra_propagates_sidecar_dir() {
        let _lock = lock_env();
        let _env_bt = EnvVarGuard::remove("RUST_BACKTRACE");
        let _env_log = EnvVarGuard::remove("RUST_LOG");
        // Explicit override path proves the token shape is exactly
        // `KTSTR_SIDECAR_DIR=<path>` and uses the override verbatim
        // (host's `sidecar_dir()` honours the env var as the
        // operator-chosen override slot).
        let _env_sd = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "/explicit/sidecar/dir");

        let entry = KtstrTestEntry {
            name: "cmd",
            ..KtstrTestEntry::DEFAULT
        };
        let out = build_cmdline_extra(&entry);
        assert_eq!(out, "KTSTR_SIDECAR_DIR=/explicit/sidecar/dir");
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
        let missing_scheduler = PathBuf::from("/nonexistent/build_vm_builder_base_test_scheduler");
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

    // -- vm_timeout_from_entry tests --

    /// VM timeout = max(watchdog, duration, 1s) + VM_TIMEOUT_HEADROOM.
    /// Watchdog dominates when it's the largest.
    #[test]
    fn vm_timeout_from_entry_uses_watchdog_when_largest() {
        let entry = KtstrTestEntry {
            name: "wdog",
            watchdog_timeout: Duration::from_secs(60),
            duration: Duration::from_secs(30),
            ..KtstrTestEntry::DEFAULT
        };
        // base = max(60, 30, 1) = 60s, plus 45s headroom = 105s.
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(105));
    }

    /// Duration dominates when it's larger than the watchdog.
    #[test]
    fn vm_timeout_from_entry_uses_duration_when_largest() {
        let entry = KtstrTestEntry {
            name: "dur",
            watchdog_timeout: Duration::from_secs(5),
            duration: Duration::from_secs(120),
            ..KtstrTestEntry::DEFAULT
        };
        // base = max(5, 120, 1) = 120s, plus 45s headroom = 165s.
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(165));
    }

    /// Both watchdog and duration well under 1s → 1s floor + headroom.
    /// The .max(Duration::from_secs(1)) clause prevents tests with
    /// micro-second values from producing a sub-headroom timeout.
    #[test]
    fn vm_timeout_from_entry_floor_when_both_small() {
        let entry = KtstrTestEntry {
            name: "tiny",
            watchdog_timeout: Duration::from_millis(10),
            duration: Duration::from_millis(50),
            ..KtstrTestEntry::DEFAULT
        };
        // base = max(10ms, 50ms, 1s) = 1s, plus 45s headroom = 46s.
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(46));
    }

    /// VM_TIMEOUT_HEADROOM constant is exactly 45s — pin the value
    /// because the doc comment refers to it as covering the
    /// freeze-coordinator rendezvous (30s) plus dump render +
    /// probe collection. A drift would silently undercut the
    /// dump-completion budget.
    #[test]
    fn vm_timeout_headroom_is_45_seconds() {
        assert_eq!(VM_TIMEOUT_HEADROOM, Duration::from_secs(45));
    }

    /// Default entry: watchdog_timeout=4s, duration=2s → base=4s,
    /// timeout = 4 + 45 = 49s.
    #[test]
    fn vm_timeout_from_default_entry() {
        let entry = KtstrTestEntry {
            name: "default",
            ..KtstrTestEntry::DEFAULT
        };
        assert_eq!(vm_timeout_from_entry(&entry), Duration::from_secs(49));
    }
}
