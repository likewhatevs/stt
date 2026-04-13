//! Runtime support for `#[ktstr_test]` integration tests.
//!
//! Provides the registration type, distributed slice, VM launcher,
//! and result evaluation. Includes guest-side profraw flush for
//! coverage-instrumented builds.
//!
//! See the [Writing Tests](https://likewhatevs.github.io/ktstr/guide/writing-tests.html)
//! and [`#[ktstr_test]` Macro](https://likewhatevs.github.io/ktstr/guide/writing-tests/ktstr-test-macro.html)
//! chapters of the guide.

use anyhow::{Context, Result};
use linkme::distributed_slice;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::assert::{AssertResult, ScenarioStats};
use crate::monitor::MonitorSummary;
use crate::scenario::Ctx;
use crate::timeline::StimulusEvent;
use crate::vmm;

/// True when RUST_BACKTRACE is set to "1" or "full".
/// Gates verbose diagnostic output (dmesg, scheduler log, COM1/COM2 dumps).
fn verbose() -> bool {
    std::env::var("RUST_BACKTRACE")
        .map(|v| v == "1" || v == "full")
        .unwrap_or(false)
}

/// Test result sidecar written to KTSTR_SIDECAR_DIR for post-run analysis.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SidecarResult {
    pub test_name: String,
    pub topology: String,
    pub scheduler: String,
    pub passed: bool,
    pub stats: ScenarioStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor: Option<MonitorSummary>,
    pub stimulus_events: Vec<StimulusEvent>,
    #[serde(default = "crate::stats::default_work_type")]
    pub work_type: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verifier_stats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kvm_stats: Option<crate::vmm::KvmStatsTotals>,
}

/// Scan a directory for ktstr sidecar JSON files. Recurses one level
/// into subdirectories to handle per-job gauntlet layouts.
pub(crate) fn collect_sidecars(dir: &std::path::Path) -> Vec<SidecarResult> {
    let mut sidecars = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return sidecars,
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("json")
            && path.to_str().is_some_and(|s| s.contains(".ktstr."))
            && let Ok(data) = std::fs::read_to_string(&path)
            && let Ok(sc) = serde_json::from_str::<SidecarResult>(&data)
        {
            sidecars.push(sc);
        }
    }
    // Recurse one level for gauntlet per-job subdirectories.
    for sub in subdirs {
        if let Ok(entries) = std::fs::read_dir(&sub) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json")
                    && path.to_str().is_some_and(|s| s.contains(".ktstr."))
                    && let Ok(data) = std::fs::read_to_string(&path)
                    && let Ok(sc) = serde_json::from_str::<SidecarResult>(&data)
                {
                    sidecars.push(sc);
                }
            }
        }
    }
    sidecars
}

/// Early dispatch for `#[ktstr_test]` test execution.
///
/// Runs before `main()` in any binary that links against ktstr.
///
/// When running as PID 1 (the binary is `/init` in the VM), calls
/// `ktstr_guest_init()` which handles the full init lifecycle and never
/// returns.
///
/// - `--ktstr-test-fn=NAME --ktstr-topo=NsNcNt`: host-side dispatch —
///   boots a VM with the specified topology and runs the test inside it.
/// - `--ktstr-test-fn=NAME` (without `--ktstr-topo`): guest-side dispatch —
///   runs the test function directly (inside a VM that was already booted).
/// - nextest protocol (`--list`/`--exact`): intercepted when running
///   under nextest (`NEXTEST` env var set), delegates to [`ktstr_main`].
/// - Otherwise: no-op (falls through to the standard test harness).
#[doc(hidden)]
#[ctor::ctor]
pub fn ktstr_test_early_dispatch() {
    // PID 1: the binary is /init in the VM. Perform full init lifecycle
    // (mounts, scheduler, test dispatch, reboot). Never returns.
    if unsafe { libc::getpid() } == 1 {
        ktstr_guest_init();
    }

    if let Some(code) = maybe_dispatch_host_test() {
        std::process::exit(code);
    }
    if let Some(code) = maybe_dispatch_vm_test() {
        // The LLVM profiling runtime registers its atexit handler via a
        // .init_array entry (C++ global initializer). Our ctor also lives
        // in .init_array, and the execution order between them is
        // non-deterministic. If our ctor runs first, the atexit handler
        // was never registered, so std::process::exit() won't write the
        // profraw. Serialize profraw to a buffer and write it to the SHM
        // ring for host-side extraction.
        try_flush_profraw();
        std::process::exit(code);
    }

    // nextest protocol: intercept --list and --exact when running under
    // nextest. Under cargo test, fall through to the standard harness
    // which runs the #[test] wrappers generated by #[ktstr_test].
    //
    // Binaries with real #[ktstr_test] entries need the ctor to handle
    // listing (gauntlet expansion) and dispatch (VM booting). The lib
    // test binary has only the dummy entry and no gauntlet variants —
    // skip interception so the standard harness discovers #[cfg(test)]
    // module #[test] functions (unit tests).
    if std::env::var_os("NEXTEST").is_some() {
        let has_real_tests = KTSTR_TESTS.iter().any(|e| e.name != "__unit_test_dummy__");
        if has_real_tests {
            let args: Vec<String> = std::env::args().collect();
            if args.iter().any(|a| a == "--list" || a == "--exact") {
                ktstr_main();
            }
        }
    }
}

/// Guest init entry point. Called when running as PID 1 (the binary is
/// `/init` in the VM). Handles the full init lifecycle: mounts,
/// scheduler start, test dispatch, cleanup, and reboot. Never returns.
pub(crate) fn ktstr_guest_init() -> ! {
    vmm::rust_init::ktstr_guest_init()
}

/// Host-side dispatch: if both `--ktstr-test-fn` and `--ktstr-topo` are
/// present, boot a VM with the specified topology and run the test
/// inside it. Returns `Some(exit_code)` if dispatched, `None` otherwise.
fn maybe_dispatch_host_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_test_fn_arg(&args)?;
    let topo_str = extract_topo_arg(&args)?;

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    let (sockets, cores, threads) = match parse_topo_string(&topo_str) {
        Some(t) => t,
        None => {
            eprintln!("ktstr_test: invalid --ktstr-topo format '{topo_str}' (expected NsNcNt)");
            return Some(1);
        }
    };

    let cpus = sockets * cores * threads;
    let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
    let topo = TopoOverride {
        sockets,
        cores,
        threads,
        memory_mb,
    };

    let active_flags = extract_flags_arg(&args).unwrap_or_default();
    match run_ktstr_test_with_topo_and_flags(entry, &topo, &active_flags) {
        Ok(_) => Some(0),
        Err(e) => {
            eprintln!("ktstr_test: {e:#}");
            Some(1)
        }
    }
}

/// SHM ring message type for profraw data.
pub(crate) const MSG_TYPE_PROFRAW: u32 = 0x50524157; // "PRAW"

/// SHM size for ktstr_test VMs: 16 MB.
/// Sized for profraw (1-2 MB), stimulus events, exit code, and test
/// results with mid-flight drain headroom.
const KTSTR_TEST_SHM_SIZE: u64 = 16 * 1024 * 1024;

/// How to specify the scheduler binary for an `#[ktstr_test]`.
pub enum SchedulerSpec {
    /// No scheduler binary — use EEVDF (kernel default).
    None,
    /// Auto-discover a scheduler binary by name.
    Name(&'static str),
    /// Explicit path to a scheduler binary.
    Path(&'static str),
    /// Kernel-built scheduler (e.g. BPF-less sched_ext or debugfs-tuned).
    /// Activated/deactivated via shell commands rather than a binary.
    KernelBuiltin {
        enable: &'static [&'static str],
        disable: &'static [&'static str],
    },
}

impl SchedulerSpec {
    /// Whether this spec represents an active scheduling policy
    /// (anything other than the kernel default EEVDF).
    pub const fn has_active_scheduling(&self) -> bool {
        !matches!(self, SchedulerSpec::None)
    }
}

pub use crate::scenario::flags::FlagDecl;

/// Definition of a scheduler for the test framework.
///
/// Captures everything the framework needs to know about a scheduler:
/// its binary, flag declarations, sysctls, kernel args, cgroup parent,
/// scheduler args, and monitor thresholds.
pub struct Scheduler {
    pub name: &'static str,
    pub binary: SchedulerSpec,
    pub flags: &'static [&'static FlagDecl],
    pub sysctls: &'static [(&'static str, &'static str)],
    pub kargs: &'static [&'static str],
    pub assert: crate::assert::Assert,
    /// Cgroup parent path. When set, the init creates
    /// `/sys/fs/cgroup/{path}` before starting the scheduler, and
    /// `--cell-parent-cgroup {path}` is injected into scheduler args.
    pub cgroup_parent: Option<&'static str>,
    /// Scheduler CLI args, prepended before per-test `extra_sched_args`.
    pub sched_args: &'static [&'static str],
    /// Default VM topology for tests using this scheduler. Tests inherit
    /// this topology unless they override `sockets`, `cores`, or
    /// `threads` explicitly in `#[ktstr_test]`.
    pub topology: Topology,
}

impl Scheduler {
    pub const EEVDF: Scheduler = Scheduler {
        name: "eevdf",
        binary: SchedulerSpec::None,
        flags: &[],
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NONE,
        cgroup_parent: None,
        sched_args: &[],
        topology: Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        },
    };

    /// Const constructor for defining schedulers in static context.
    pub const fn new(name: &'static str) -> Scheduler {
        Scheduler {
            name,
            binary: SchedulerSpec::None,
            flags: &[],
            sysctls: &[],
            kargs: &[],
            assert: crate::assert::Assert::NONE,
            cgroup_parent: None,
            sched_args: &[],
            topology: Topology {
                sockets: 1,
                cores_per_socket: 2,
                threads_per_core: 1,
            },
        }
    }

    /// Set the binary spec. Returns self for const chaining.
    pub const fn binary(mut self, binary: SchedulerSpec) -> Self {
        self.binary = binary;
        self
    }

    /// Set flag declarations. Returns self for const chaining.
    pub const fn flags(mut self, flags: &'static [&'static FlagDecl]) -> Self {
        self.flags = flags;
        self
    }

    /// Set sysctls. Returns self for const chaining.
    pub const fn sysctls(mut self, sysctls: &'static [(&'static str, &'static str)]) -> Self {
        self.sysctls = sysctls;
        self
    }

    /// Set kernel args. Returns self for const chaining.
    pub const fn kargs(mut self, kargs: &'static [&'static str]) -> Self {
        self.kargs = kargs;
        self
    }

    /// Set assertion config. Returns self for const chaining.
    pub const fn assert(mut self, assert: crate::assert::Assert) -> Self {
        self.assert = assert;
        self
    }

    /// Set cgroup parent path. The init creates
    /// `/sys/fs/cgroup/{path}` before starting the scheduler, and
    /// `--cell-parent-cgroup {path}` is injected into scheduler args.
    pub const fn cgroup_parent(mut self, path: &'static str) -> Self {
        self.cgroup_parent = Some(path);
        self
    }

    /// Set scheduler CLI args prepended before per-test
    /// `extra_sched_args`.
    pub const fn sched_args(mut self, args: &'static [&'static str]) -> Self {
        self.sched_args = args;
        self
    }

    /// Set the default VM topology for tests using this scheduler.
    /// Tests inherit this unless they override `sockets`, `cores`, or
    /// `threads` explicitly in `#[ktstr_test]`.
    pub const fn topology(mut self, sockets: u32, cores: u32, threads: u32) -> Self {
        self.topology = Topology {
            sockets,
            cores_per_socket: cores,
            threads_per_core: threads,
        };
        self
    }

    /// Names of all flags this scheduler supports.
    pub fn supported_flag_names(&self) -> Vec<&str> {
        self.flags.iter().map(|f| f.name).collect()
    }

    /// Dependencies of a flag (from its `FlagDecl.requires`).
    pub fn flag_requires(&self, name: &str) -> Vec<&str> {
        self.flags
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.requires.iter().map(|r| r.name).collect())
            .unwrap_or_default()
    }

    /// Extra CLI arguments associated with a flag.
    pub fn flag_args(&self, name: &str) -> Option<&'static [&'static str]> {
        self.flags.iter().find(|f| f.name == name).map(|f| f.args)
    }

    /// Generate flag profiles scoped to this scheduler's supported flags.
    ///
    /// Uses `FlagDecl::requires` for dependency constraints instead of
    /// the module-level `flag_requires()` hardcoded table.
    pub fn generate_profiles(
        &self,
        required: &[&'static str],
        excluded: &[&'static str],
    ) -> Vec<crate::scenario::FlagProfile> {
        let optional: Vec<&'static str> = self
            .flags
            .iter()
            .map(|f| f.name)
            .filter(|f| !required.contains(f) && !excluded.contains(f))
            .collect();
        let mut out = Vec::new();
        for mask in 0..(1u32 << optional.len()) {
            let mut fl: Vec<&'static str> = required.to_vec();
            for (i, &f) in optional.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    fl.push(f);
                }
            }
            let valid = fl
                .iter()
                .all(|f| self.flag_requires(f).iter().all(|r| fl.contains(r)));
            if valid {
                fl.sort_by_key(|f| {
                    self.flags
                        .iter()
                        .position(|d| d.name == *f)
                        .unwrap_or(usize::MAX)
                });
                out.push(crate::scenario::FlagProfile { flags: fl });
            }
        }
        out
    }
}

/// Host-side BPF map write performed during VM execution.
///
/// The write is event-driven: the host polls for BPF map discoverability
/// (scheduler loaded), then polls the SHM ring for scenario start, then
/// writes.
pub struct BpfMapWrite {
    /// Map name suffix to match (e.g. ".bss").
    pub map_name_suffix: &'static str,
    /// Byte offset within the map's value region.
    pub offset: usize,
    /// u32 value to write.
    pub value: u32,
}

/// Re-export of [`crate::vmm::topology::Topology`] for use in
/// [`KtstrTestEntry`] statics generated by the `#[ktstr_test]` macro.
pub use crate::vmm::topology::Topology;

/// Gauntlet topology filtering constraints.
///
/// Controls which gauntlet presets are eligible for a test entry.
/// Presets that don't meet all constraints are skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyConstraints {
    /// Minimum number of sockets.
    pub min_sockets: u32,
    /// Minimum number of LLCs.
    pub min_llcs: u32,
    /// Whether the test requires SMT (threads_per_core > 1).
    pub requires_smt: bool,
    /// Minimum total CPU count.
    pub min_cpus: u32,
}

impl TopologyConstraints {
    pub const DEFAULT: TopologyConstraints = TopologyConstraints {
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
    };
}

/// Registration entry for an `#[ktstr_test]`-annotated function.
pub struct KtstrTestEntry {
    pub name: &'static str,
    pub func: fn(&Ctx) -> Result<AssertResult>,
    pub topology: Topology,
    pub constraints: TopologyConstraints,
    pub memory_mb: u32,
    pub scheduler: &'static Scheduler,
    pub auto_repro: bool,
    pub replicas: u32,
    pub assert: crate::assert::Assert,
    pub extra_sched_args: &'static [&'static str],
    /// scx_watchdog_timeout in the guest kernel.
    pub watchdog_timeout: Duration,
    /// Host-side BPF map write to perform during VM execution.
    pub bpf_map_write: Option<&'static BpfMapWrite>,
    /// Flags that must be present in every flag profile for this test.
    pub required_flags: &'static [&'static str],
    /// Flags that must not be present in any flag profile for this test.
    pub excluded_flags: &'static [&'static str],
    /// Pin vCPU threads to host cores matching the virtual topology's LLC
    /// structure, use 2MB hugepages for guest memory, set KVM_HINTS_REALTIME
    /// CPUID hint (disables PV spinlocks, PV TLB flush, PV sched_yield;
    /// enables haltpoll cpuidle), disable PAUSE and HLT VM exits via
    /// KVM_CAP_X86_DISABLE_EXITS (HLT falls back to PAUSE-only when
    /// mitigate_smt_rsb is active), skip KVM_CAP_HALT_POLL (guest haltpoll
    /// cpuidle disables host halt polling via MSR_KVM_POLL_CONTROL), and
    /// validate that the host has enough CPUs and LLCs to satisfy the
    /// request without oversubscription.
    pub performance_mode: bool,
    /// Workload duration.
    pub duration: Duration,
    /// Workers per cgroup.
    pub workers_per_cgroup: u32,
    /// When true, the test expects run_ktstr_test to return Err.
    /// Disables auto_repro (no point probing a deliberately failing test).
    pub expect_err: bool,
    /// When true, the test runs directly on the host instead of
    /// booting a VM. Used for tests that need host tools (cargo,
    /// nested VMs) unavailable in the guest initramfs.
    pub host_only: bool,
}

/// Placeholder function for `KtstrTestEntry::DEFAULT`. Panics if called.
fn default_test_func(_ctx: &Ctx) -> Result<AssertResult> {
    anyhow::bail!("KtstrTestEntry::DEFAULT func called — override func before use")
}

impl KtstrTestEntry {
    /// Sensible defaults for all fields. Override `name`, `func`, and
    /// `scheduler` (at minimum) via struct update syntax:
    ///
    /// ```ignore
    /// static ENTRY: KtstrTestEntry = KtstrTestEntry {
    ///     name: "my_test",
    ///     func: my_test_fn,
    ///     scheduler: &MITOSIS,
    ///     ..KtstrTestEntry::DEFAULT
    /// };
    /// ```
    pub const DEFAULT: KtstrTestEntry = KtstrTestEntry {
        name: "",
        func: default_test_func,
        topology: Topology {
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        },
        constraints: TopologyConstraints::DEFAULT,
        memory_mb: 2048,
        scheduler: &Scheduler::EEVDF,
        auto_repro: true,
        replicas: 1,
        assert: crate::assert::Assert::NONE,
        extra_sched_args: &[],
        watchdog_timeout: Duration::from_secs(4),
        bpf_map_write: None,
        required_flags: &[],
        excluded_flags: &[],
        performance_mode: false,
        duration: Duration::from_secs(2),
        workers_per_cgroup: 2,
        expect_err: false,
        host_only: false,
    };
}

/// Distributed slice collecting all `#[ktstr_test]` entries via linkme.
#[distributed_slice]
pub static KTSTR_TESTS: [KtstrTestEntry];

/// Look up a registered test function by name.
pub fn find_test(name: &str) -> Option<&'static KtstrTestEntry> {
    KTSTR_TESTS.iter().find(|e| e.name == name)
}

/// Validate that `required_flags` and `excluded_flags` on an entry
/// reference flags the scheduler actually declares. Panics on unknown
/// flag names so typos are caught at test discovery time. Also panics
/// if a flag appears in both `required_flags` and `excluded_flags`.
fn validate_entry_flags(entry: &KtstrTestEntry) {
    if entry.scheduler.flags.is_empty() {
        if !entry.required_flags.is_empty() || !entry.excluded_flags.is_empty() {
            panic!(
                "ktstr_test: '{}' specifies flags but scheduler '{}' has no flag declarations",
                entry.name, entry.scheduler.name,
            );
        }
        return;
    }
    let valid: Vec<&str> = entry.scheduler.supported_flag_names();
    for &flag in entry.required_flags {
        if !valid.contains(&flag) {
            panic!(
                "ktstr_test: '{}' references unknown required_flag '{}'; valid flags for scheduler '{}': {}",
                entry.name,
                flag,
                entry.scheduler.name,
                valid.join(", "),
            );
        }
    }
    for &flag in entry.excluded_flags {
        if !valid.contains(&flag) {
            panic!(
                "ktstr_test: '{}' references unknown excluded_flag '{}'; valid flags for scheduler '{}': {}",
                entry.name,
                flag,
                entry.scheduler.name,
                valid.join(", "),
            );
        }
    }
    for &flag in entry.required_flags {
        if entry.excluded_flags.contains(&flag) {
            panic!(
                "ktstr_test: '{}' has flag '{}' in both required_flags and excluded_flags",
                entry.name, flag,
            );
        }
    }
}

/// Optional topology override for `run_ktstr_test`.
pub(crate) struct TopoOverride {
    pub sockets: u32,
    pub cores: u32,
    pub threads: u32,
    pub memory_mb: u32,
}

/// Parse a topology string in "NsNcNt" format (e.g. "2s4c2t").
/// Returns None if the string doesn't match the expected format.
pub(crate) fn parse_topo_string(s: &str) -> Option<(u32, u32, u32)> {
    let s_pos = s.find('s')?;
    let c_pos = s.find('c')?;
    let t_pos = s.find('t')?;
    if s_pos >= c_pos || c_pos >= t_pos {
        return None;
    }
    let sockets: u32 = s[..s_pos].parse().ok()?;
    let cores: u32 = s[s_pos + 1..c_pos].parse().ok()?;
    let threads: u32 = s[c_pos + 1..t_pos].parse().ok()?;
    if sockets == 0 || cores == 0 || threads == 0 {
        return None;
    }
    Some((sockets, cores, threads))
}

/// Whether this is the final nextest attempt for the current test.
///
/// Reads `NEXTEST_ATTEMPT` and `NEXTEST_TOTAL_ATTEMPTS` env vars.
/// Returns `true` when not running under nextest (no retries available)
/// or when on the last attempt.
pub(crate) fn is_final_nextest_attempt() -> bool {
    let attempt = std::env::var("NEXTEST_ATTEMPT")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);
    let total = std::env::var("NEXTEST_TOTAL_ATTEMPTS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(1);
    attempt >= total
}

/// Host-side entry point: build a VM, boot it with `--ktstr-test-fn=NAME`,
/// extract profraw from SHM, and return the test result.
///
/// Validates KVM access and auto-discovers a kernel image via
/// `resolve_test_kernel()` when `KTSTR_TEST_KERNEL` is not set.
pub fn run_ktstr_test(entry: &KtstrTestEntry) -> Result<AssertResult> {
    if entry.host_only {
        return run_host_only_test_inner(entry);
    }
    if entry.bpf_map_write.is_some()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        return Ok(crate::assert::AssertResult::skip(
            "skipped: vmlinux not found, bpf_map_write requires vmlinux",
        ));
    }
    run_ktstr_test_inner(entry, None, &[])
}

/// Like `run_ktstr_test` but with an explicit topology override and
/// active flags that map to
/// scheduler CLI args via `Scheduler::flag_args()`.
pub(crate) fn run_ktstr_test_with_topo_and_flags(
    entry: &KtstrTestEntry,
    topo: &TopoOverride,
    active_flags: &[String],
) -> Result<AssertResult> {
    run_ktstr_test_inner(entry, Some(topo), active_flags)
}

/// Run a test result through expect_err logic and return an exit code.
///
/// Returns 0 on pass, 1 on failure.
///
/// On `ResourceContention`:
/// - Non-final nextest attempt: return 1 so nextest retries with
///   exponential backoff.
/// - Final attempt (or not running under nextest): return 0 with
///   "ignored" message so the test doesn't fail the suite.
fn result_to_exit_code(result: Result<AssertResult>, expect_err: bool) -> i32 {
    match result {
        Ok(_) if expect_err => {
            eprintln!("expected error but test passed");
            1
        }
        Ok(_) => 0,
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            let reason = e
                .downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .unwrap()
                .reason
                .clone();
            if is_final_nextest_attempt() {
                eprintln!("resource contention (ignored): {reason}");
                0
            } else {
                eprintln!("resource contention (will retry): {reason}");
                1
            }
        }
        Err(_) if expect_err => 0,
        Err(e) => {
            eprintln!("{e:#}");
            1
        }
    }
}

/// Whether a base test entry is "ignored" (skipped by default).
///
/// Tests whose names start with `demo_` are ignored -- they are
/// demonstration/benchmarking tests that require manual opt-in.
fn is_ignored(entry: &KtstrTestEntry) -> bool {
    entry.name.starts_with("demo_")
}

/// Collect test names for nextest discovery (--list --format terse).
///
/// Nextest calls the binary twice:
/// - Without `--ignored`: prints ALL tests (ignored and non-ignored).
/// - With `--ignored`: prints ONLY ignored tests.
///
/// Gauntlet variants are always ignored. Base tests are ignored when
/// their name starts with `demo_`.
///
/// When `KTSTR_BUDGET_SECS` is set, applies greedy coverage maximization
/// to select the subset of tests that maximizes feature coverage within
/// the time budget. Only selected tests are printed.
fn list_tests(ignored_only: bool) {
    let raw = std::env::var("KTSTR_BUDGET_SECS").ok();
    let budget_secs: Option<f64> = raw.as_deref().and_then(|s| match s.parse::<f64>() {
        Ok(v) if v > 0.0 => Some(v),
        Ok(v) => {
            eprintln!("ktstr: KTSTR_BUDGET_SECS={v}: must be positive, ignoring");
            None
        }
        Err(e) => {
            eprintln!("ktstr: KTSTR_BUDGET_SECS={s:?}: {e}, ignoring");
            None
        }
    });

    if let Some(budget) = budget_secs {
        list_tests_budget(ignored_only, budget);
    } else {
        list_tests_all(ignored_only);
    }
}

/// List all tests without budget filtering.
fn list_tests_all(ignored_only: bool) {
    let presets = crate::vm::gauntlet_presets();

    for entry in KTSTR_TESTS.iter() {
        validate_entry_flags(entry);

        if !ignored_only || is_ignored(entry) {
            println!("{}: test", entry.name);
        }

        // Host-only tests run on the host without a VM -- gauntlet
        // topology variants are meaningless.
        if entry.host_only {
            continue;
        }

        let profiles = entry
            .scheduler
            .generate_profiles(entry.required_flags, entry.excluded_flags);

        // Gauntlet variants are always ignored.
        for preset in &presets {
            let t = &preset.topology;
            if t.sockets < entry.constraints.min_sockets
                || t.num_llcs() < entry.constraints.min_llcs
                || (entry.constraints.requires_smt && t.threads_per_core < 2)
                || t.total_cpus() < entry.constraints.min_cpus
            {
                continue;
            }
            for profile in &profiles {
                let pname = profile.name();
                println!("gauntlet/{}/{}/{}: test", entry.name, preset.name, pname,);
            }
        }
    }
}

/// List tests with budget-based coverage maximization.
///
/// Collects all eligible tests as candidates, runs greedy selection,
/// and prints only the selected subset.
fn list_tests_budget(ignored_only: bool, budget_secs: f64) {
    use crate::budget::{TestCandidate, estimate_duration, extract_features, select};

    let presets = crate::vm::gauntlet_presets();
    let mut candidates: Vec<TestCandidate> = Vec::new();

    for entry in KTSTR_TESTS.iter() {
        validate_entry_flags(entry);

        let base_ignored = is_ignored(entry);
        let base_topo = entry.topology;

        // Base test
        if !ignored_only || base_ignored {
            candidates.push(TestCandidate {
                name: format!("{}: test", entry.name),
                features: extract_features(entry, &base_topo, &[], false, entry.name),
                estimated_secs: estimate_duration(entry, &base_topo),
            });
        }

        if entry.host_only {
            continue;
        }

        let profiles = entry
            .scheduler
            .generate_profiles(entry.required_flags, entry.excluded_flags);

        for preset in &presets {
            let t = &preset.topology;
            if t.sockets < entry.constraints.min_sockets
                || t.num_llcs() < entry.constraints.min_llcs
                || (entry.constraints.requires_smt && t.threads_per_core < 2)
                || t.total_cpus() < entry.constraints.min_cpus
            {
                continue;
            }
            for profile in &profiles {
                let pname = profile.name();
                let test_name = format!("gauntlet/{}/{}/{}", entry.name, preset.name, pname);
                candidates.push(TestCandidate {
                    name: format!("{}: test", test_name),
                    features: extract_features(entry, t, &profile.flags, true, &test_name),
                    estimated_secs: estimate_duration(entry, t),
                });
            }
        }
    }

    let selected = select(&candidates, budget_secs);
    for &i in &selected {
        println!("{}", candidates[i].name);
    }

    let stats = crate::budget::selection_stats(&candidates, &selected, budget_secs);
    eprintln!(
        "ktstr budget: {}/{} tests, {:.0}/{:.0}s used, {}/{} configurations covered",
        stats.selected,
        stats.total,
        stats.budget_used,
        stats.budget_total,
        stats.bits_covered,
        stats.bits_possible,
    );
}

/// Parse a nextest-style test name and run it.
///
/// Handles both base tests (`entry.name`) and gauntlet variants
/// (`gauntlet/{name}/{preset}/{profile}`). Returns an exit code.
fn run_named_test(test_name: &str) -> i32 {
    if let Some(rest) = test_name.strip_prefix("gauntlet/") {
        return run_gauntlet_test(rest);
    }

    let entry = match find_test(test_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };

    if entry.host_only {
        return run_host_only_test(entry);
    }

    if entry.bpf_map_write.is_some()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("skipped: vmlinux not found, bpf_map_write requires vmlinux");
        return 0;
    }

    let result = run_ktstr_test_inner(entry, None, &[]);
    result_to_exit_code(result, entry.expect_err)
}

/// Run a host-only test directly without booting a VM.
/// Returns an exit code for nextest dispatch.
fn run_host_only_test(entry: &KtstrTestEntry) -> i32 {
    let result = run_host_only_test_inner(entry);
    result_to_exit_code(result, entry.expect_err)
}

/// Inner host-only dispatch returning `Result<AssertResult>`.
///
/// Builds a minimal Ctx and calls the test function on the host.
/// Used for tests that need host tools (cargo, nested VMs).
fn run_host_only_test_inner(entry: &KtstrTestEntry) -> Result<AssertResult> {
    let topo = crate::topology::TestTopology::from_spec(
        entry.topology.sockets,
        entry.topology.cores_per_socket,
        entry.topology.threads_per_core,
    );
    let cgroups = crate::cgroup::CgroupManager::new("/sys/fs/cgroup/ktstr");
    let workers_per_cgroup = entry.workers_per_cgroup as usize;
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    let ctx = crate::scenario::Ctx {
        cgroups: &cgroups,
        topo: &topo,
        duration: entry.duration,
        workers_per_cgroup,
        sched_pid: 0,
        settle: Duration::from_millis(500),
        work_type_override: None,
        assert: merged_assert,
        wait_for_map_write: false,
    };
    (entry.func)(&ctx)
}

/// Run a gauntlet variant test. `rest` is `{name}/{preset}/{profile}`.
fn run_gauntlet_test(rest: &str) -> i32 {
    let parts: Vec<&str> = rest.splitn(3, '/').collect();
    if parts.len() != 3 {
        eprintln!("invalid gauntlet test name: gauntlet/{rest}");
        return 1;
    }
    let (test_name, preset_name, profile_name) = (parts[0], parts[1], parts[2]);

    let entry = match find_test(test_name) {
        Some(e) => e,
        None => {
            eprintln!("unknown test: {test_name}");
            return 1;
        }
    };
    validate_entry_flags(entry);

    let presets = crate::vm::gauntlet_presets();
    let preset = match presets.iter().find(|p| p.name == preset_name) {
        Some(p) => p,
        None => {
            eprintln!("unknown gauntlet preset: {preset_name}");
            return 1;
        }
    };

    let t = &preset.topology;
    let cpus = t.total_cpus();

    // Skip topologies the host cannot support. Without this check,
    // ResourceContention is returned during VM build and silently
    // converted to PASS on the final nextest attempt.
    let host_cpus = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    if cpus > host_cpus {
        eprintln!("skipped: preset {preset_name} needs {cpus} CPUs, host has {host_cpus}",);
        return 0;
    }

    let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
    let topo = TopoOverride {
        sockets: t.sockets,
        cores: t.cores_per_socket,
        threads: t.threads_per_core,
        memory_mb,
    };

    let profiles = entry
        .scheduler
        .generate_profiles(entry.required_flags, entry.excluded_flags);
    let flags: Vec<String> = match profiles.iter().find(|p| p.name() == profile_name) {
        Some(p) => p.flags.iter().map(|s| s.to_string()).collect(),
        None => {
            eprintln!("unknown flag profile: {profile_name}");
            return 1;
        }
    };

    if entry.bpf_map_write.is_some()
        && let Ok(kernel) = resolve_test_kernel()
        && crate::vmm::find_vmlinux(&kernel).is_none()
    {
        eprintln!("skipped: vmlinux not found, bpf_map_write requires vmlinux");
        return 0;
    }

    let result = run_ktstr_test_inner(entry, Some(&topo), &flags);
    result_to_exit_code(result, entry.expect_err)
}

/// Collect sidecar JSON files and return the full gauntlet analysis.
///
/// When `dir` is `Some`, reads sidecars from that directory. Otherwise
/// uses the default sidecar directory (`KTSTR_SIDECAR_DIR` or
/// `target/ktstr/{branch}-{hash}/`).
///
/// Returns the concatenated output of `analyze_rows`, verifier stats,
/// callback profile, and KVM stats. Returns an empty string when no
/// sidecars are found.
pub fn analyze_sidecars(dir: Option<&std::path::Path>) -> String {
    let default_dir;
    let dir = match dir {
        Some(d) => d,
        None => {
            default_dir = sidecar_dir();
            &default_dir
        }
    };
    let sidecars = collect_sidecars(dir);
    if sidecars.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let rows: Vec<_> = sidecars.iter().map(crate::stats::sidecar_to_row).collect();
    if !rows.is_empty() {
        out.push_str(&crate::stats::analyze_rows(&rows));
    }
    let vstats = format_verifier_stats(&sidecars);
    if !vstats.is_empty() {
        out.push_str(&vstats);
    }
    let cprofile = format_callback_profile(&sidecars);
    if !cprofile.is_empty() {
        out.push_str(&cprofile);
    }
    let kstats = format_kvm_stats(&sidecars);
    if !kstats.is_empty() {
        out.push_str(&kstats);
    }
    out
}

/// Nextest protocol handler.
///
/// Called automatically by [`ktstr_test_early_dispatch`] when running
/// under nextest. Not intended for direct use.
///
/// - `--list --format terse`: output `name: test\n` for each test.
/// - `--exact NAME --nocapture`: run the named test, exit 0/1.
pub fn ktstr_main() -> ! {
    let args: Vec<String> = std::env::args().collect();

    // Discovery mode: --list --format terse [--ignored]
    if args.iter().any(|a| a == "--list") {
        let ignored_only = args.iter().any(|a| a == "--ignored");
        list_tests(ignored_only);
        std::process::exit(0);
    }

    // Execution mode: --exact NAME [--nocapture] [--ignored] [--bench]
    if let Some(pos) = args.iter().position(|a| a == "--exact") {
        if let Some(name) = args.get(pos + 1) {
            let code = run_named_test(name);
            std::process::exit(code);
        }
        eprintln!("--exact requires a test name");
        std::process::exit(1);
    }

    // Fallback: no recognized arguments.
    eprintln!("usage: <binary> --list --format terse [--ignored]");
    eprintln!("       <binary> --exact <test_name> --nocapture");
    std::process::exit(1)
}

/// BPF verifier complexity limit (BPF_COMPLEXITY_LIMIT_INSNS).
const VERIFIER_INSN_LIMIT: u32 = 1_000_000;

/// Percentage of the verifier limit that triggers a warning.
const VERIFIER_WARN_PCT: f64 = 75.0;

/// Aggregate BPF verifier stats across sidecars into a summary table.
///
/// verified_insns is deterministic for a given binary, so per-program
/// values are deduplicated (max across observations). Flags programs
/// using >=75% of the 1M verifier complexity limit.
fn format_verifier_stats(sidecars: &[SidecarResult]) -> String {
    use std::collections::BTreeMap;

    let mut by_name: BTreeMap<&str, u32> = BTreeMap::new();
    for sc in sidecars {
        for info in &sc.verifier_stats {
            let entry = by_name.entry(&info.name).or_insert(0);
            *entry = (*entry).max(info.verified_insns);
        }
    }

    if by_name.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n=== BPF VERIFIER STATS ===\n\n");
    out.push_str(&format!(
        "  {:<24} {:>12} {:>8}\n",
        "program", "verified", "limit%"
    ));
    out.push_str(&format!("  {:-<24} {:-<12} {:-<8}\n", "", "", ""));

    let mut warnings = Vec::new();
    let mut total: u64 = 0;

    for (&name, &verified_insns) in &by_name {
        let pct = (verified_insns as f64 / VERIFIER_INSN_LIMIT as f64) * 100.0;
        let flag = if pct >= VERIFIER_WARN_PCT { " !" } else { "" };
        out.push_str(&format!(
            "  {:<24} {:>12} {:>7.1}%{flag}\n",
            name, verified_insns, pct,
        ));
        if pct >= VERIFIER_WARN_PCT {
            warnings.push(format!(
                "  {name}: {pct:.1}% of 1M limit ({verified_insns} verified insns)",
            ));
        }
        total += verified_insns as u64;
    }

    out.push_str(&format!("\n  total verified insns: {total}\n"));

    if !warnings.is_empty() {
        out.push_str("\nWARNING: programs near verifier complexity limit:\n");
        for w in &warnings {
            out.push_str(w);
            out.push('\n');
        }
    }

    out
}

/// Per-test BPF callback profile from monitor prog_stats_deltas.
///
/// Shows per-program invocation count, total CPU time, and average
/// nanoseconds per call. Each test's profile is printed independently.
fn format_callback_profile(sidecars: &[SidecarResult]) -> String {
    let mut out = String::new();

    for sc in sidecars {
        let deltas = match sc
            .monitor
            .as_ref()
            .and_then(|m| m.prog_stats_deltas.as_ref())
        {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        if out.is_empty() {
            out.push_str("\n=== BPF CALLBACK PROFILE ===\n");
        }
        out.push_str(&format!("\n  {} ({}):\n", sc.test_name, sc.topology));
        out.push_str(&format!(
            "    {:<24} {:>12} {:>14} {:>12}\n",
            "program", "cnt", "total_ns", "avg_ns"
        ));
        out.push_str(&format!(
            "    {:-<24} {:-<12} {:-<14} {:-<12}\n",
            "", "", "", ""
        ));
        for d in deltas {
            out.push_str(&format!(
                "    {:<24} {:>12} {:>14} {:>12.0}\n",
                d.name, d.cnt, d.nsecs, d.nsecs_per_call,
            ));
        }
    }

    out
}

/// Aggregate KVM stats across sidecars into a compact summary.
///
/// Averages each stat across all tests that returned `Some(KvmStatsTotals)`.
/// Tests without KVM stats (non-VM tests, old kernels) are excluded
/// from the denominator.
fn format_kvm_stats(sidecars: &[SidecarResult]) -> String {
    let with_stats: Vec<&crate::vmm::KvmStatsTotals> = sidecars
        .iter()
        .filter_map(|sc| sc.kvm_stats.as_ref())
        .collect();

    if with_stats.is_empty() {
        return String::new();
    }

    let n_vms = with_stats.len();

    // Compute cross-VM averages for each stat.
    let vm_avg = |name: &str| -> u64 {
        let sum: u64 = with_stats.iter().map(|d| d.avg(name)).sum();
        sum / n_vms as u64
    };

    let exits = vm_avg("exits");
    let halt = vm_avg("halt_exits");
    let halt_wait_ns = vm_avg("halt_wait_ns");
    let preempted = vm_avg("preemption_reported");
    let signal = vm_avg("signal_exits");
    let hypercalls = vm_avg("hypercalls");

    // Halt poll efficiency across all vCPUs and VMs.
    let total_poll_ok: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_successful_poll"))
        .sum();
    let total_poll_try: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_attempted_poll"))
        .sum();

    if exits == 0 {
        return String::new();
    }

    let halt_wait_ms = halt_wait_ns as f64 / 1_000_000.0;
    let poll_pct = if total_poll_try > 0 {
        (total_poll_ok as f64 / total_poll_try as f64) * 100.0
    } else {
        0.0
    };

    let mut out = format!("\n=== KVM STATS (avg across {n_vms} VMs) ===\n\n");
    out.push_str(&format!(
        "  exits/vcpu  {:>7}   halt/vcpu     {:>5}   halt_wait_ms {:>7.1}\n",
        exits, halt, halt_wait_ms,
    ));
    out.push_str(&format!(
        "  poll_ok%    {:>6.1}%   preempted/vcpu {:>4}   signal/vcpu  {:>7}\n",
        poll_pct, preempted, signal,
    ));
    if hypercalls > 0 {
        out.push_str(&format!("  hypercalls/vcpu {:>4}\n", hypercalls));
    }

    // Trust warnings.
    if preempted > 0 {
        let total: u64 = with_stats
            .iter()
            .map(|d| d.sum("preemption_reported"))
            .sum();
        out.push_str(&format!(
            "\n  WARNING: {total} host preemptions detected \
             -- timing results may be unreliable\n",
        ));
    }

    out
}

fn run_ktstr_test_inner(
    entry: &KtstrTestEntry,
    topo: Option<&TopoOverride>,
    active_flags: &[String],
) -> Result<AssertResult> {
    ensure_kvm()?;
    let kernel = resolve_test_kernel()?;
    let scheduler = resolve_scheduler(&entry.scheduler.binary)?;
    let ktstr_bin = crate::resolve_current_exe()?;

    let guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
    ];

    // Build cmdline: base args + sysctls (as sysctl.key=value) + kargs.
    let mut cmdline_parts = vec!["iomem=relaxed".to_string()];
    for &(key, value) in entry.scheduler.sysctls {
        cmdline_parts.push(format!("sysctl.{}={}", key, value));
    }
    for &karg in entry.scheduler.kargs {
        cmdline_parts.push(karg.to_string());
    }
    // Propagate RUST_BACKTRACE and RUST_LOG to the guest so
    // guest-side code can gate verbose output and tracing.
    if let Ok(bt) = std::env::var("RUST_BACKTRACE") {
        cmdline_parts.push(format!("RUST_BACKTRACE={bt}"));
    }
    if let Ok(log) = std::env::var("RUST_LOG") {
        cmdline_parts.push(format!("RUST_LOG={log}"));
    }
    let cmdline_extra = cmdline_parts.join(" ");

    let (sockets, cores, threads, memory_mb) = match topo {
        Some(t) => (t.sockets, t.cores, t.threads, t.memory_mb),
        None => {
            let cpus = entry.topology.total_cpus();
            let mem = (cpus * 64).max(256).max(entry.memory_mb);
            (
                entry.topology.sockets,
                entry.topology.cores_per_socket,
                entry.topology.threads_per_core,
                mem,
            )
        }
    };

    let mut builder = vmm::KtstrVm::builder()
        .kernel(&kernel)
        .init_binary(&ktstr_bin)
        .topology(sockets, cores, threads)
        .memory_deferred_min(memory_mb)
        .cmdline(&cmdline_extra)
        .shm_size(KTSTR_TEST_SHM_SIZE)
        .run_args(&guest_args)
        .timeout(Duration::from_secs(60))
        .performance_mode(entry.performance_mode);

    // Merge order: default_checks -> scheduler.assert -> per-test assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);

    if let Some(ref sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }
    if let SchedulerSpec::KernelBuiltin { enable, disable } = &entry.scheduler.binary {
        builder = builder.sched_enable_cmds(enable);
        builder = builder.sched_disable_cmds(disable);
    }
    if entry.scheduler.binary.has_active_scheduling() {
        builder = builder.monitor_thresholds(merged_assert.monitor_thresholds());
    }

    // Merge scheduler args: cgroup_parent injection + scheduler sched_args +
    // per-test extra_sched_args + flag-derived args.
    let mut sched_args: Vec<String> = Vec::new();
    if let Some(cgroup_path) = entry.scheduler.cgroup_parent {
        sched_args.push("--cell-parent-cgroup".to_string());
        sched_args.push(cgroup_path.to_string());
    }
    sched_args.extend(entry.scheduler.sched_args.iter().map(|s| s.to_string()));
    sched_args.extend(entry.extra_sched_args.iter().map(|s| s.to_string()));
    for flag_name in active_flags {
        if let Some(args) = entry.scheduler.flag_args(flag_name) {
            sched_args.extend(args.iter().map(|s| s.to_string()));
        }
    }
    if !sched_args.is_empty() {
        builder = builder.sched_args(&sched_args);
    }

    builder = builder.watchdog_timeout(entry.watchdog_timeout);

    if let Some(bpf_write) = entry.bpf_map_write {
        builder =
            builder.bpf_map_write(bpf_write.map_name_suffix, bpf_write.offset, bpf_write.value);
    }

    let vm = builder.build().context("build ktstr_test VM")?;

    let result = vm.run().context("run ktstr_test VM")?;

    // Drop the VM to release CPU/LLC flock fds before auto-repro.
    drop(vm);

    // Log verifier stats count for visibility.
    if !result.verifier_stats.is_empty() {
        eprintln!(
            "ktstr_test: verifier_stats: {} struct_ops programs",
            result.verifier_stats.len(),
        );
    }

    // When running with a struct_ops scheduler, verify that host-side
    // BPF program enumeration found programs with non-zero verified_insns.
    if entry.scheduler.binary.has_active_scheduling()
        && result.success
        && result.verifier_stats.is_empty()
    {
        eprintln!("ktstr_test: WARNING: scheduler loaded but verifier_stats is empty");
    }

    // Extract profraw from SHM ring buffer and collect stimulus events.
    let mut stimulus_events = Vec::new();
    if let Some(ref shm) = result.shm_data {
        for entry in &shm.entries {
            if entry.msg_type == MSG_TYPE_PROFRAW
                && entry.crc_ok
                && !entry.payload.is_empty()
                && let Err(e) = write_profraw(&entry.payload)
            {
                eprintln!("ktstr_test: write guest profraw: {e}");
            }
            if entry.msg_type == crate::vmm::shm_ring::MSG_TYPE_STIMULUS
                && entry.crc_ok
                && let Some(ev) = crate::vmm::shm_ring::StimulusEvent::from_payload(&entry.payload)
            {
                stimulus_events.push(crate::timeline::StimulusEvent {
                    elapsed_ms: ev.elapsed_ms as u64,
                    label: format!("StepStart[{}]", ev.step_index),
                    op_kind: Some(format!("ops={}", ev.op_count)),
                    detail: Some(format!(
                        "{} cgroups, {} workers",
                        ev.cgroup_count, ev.worker_count,
                    )),
                    total_iterations: if ev.total_iterations > 0 {
                        Some(ev.total_iterations)
                    } else {
                        None
                    },
                });
            }
        }
    }

    // auto_repro is enabled when:
    // - entry.auto_repro is true (default)
    // - a scheduler is running (not EEVDF)
    // - the test does not expect failure (expect_err = false)
    let effective_auto_repro = entry.auto_repro && scheduler.is_some() && !entry.expect_err;
    let repro_fn = |output: &str| -> Option<String> {
        if !effective_auto_repro {
            return None;
        }
        let repro = attempt_auto_repro(
            entry,
            &kernel,
            scheduler.as_deref(),
            &ktstr_bin,
            output,
            &result.stderr,
            topo,
        );
        // When auto-repro was attempted but produced no data, return a
        // diagnostic so the user knows it was tried.
        Some(repro.unwrap_or_else(|| {
            "auto-repro: no probe data — the repro VM may have failed to \
             boot, or the kernel may lack the sched_ext_exit tracepoint \
             required for the probe trigger. Check the sched_ext dump \
             and scheduler log sections above for crash details."
                .to_string()
        }))
    };

    evaluate_vm_result(
        entry,
        &result,
        &merged_assert,
        &stimulus_events,
        sockets,
        cores,
        threads,
        &repro_fn,
    )
}

/// Evaluate a VM result and produce the appropriate error or Ok.
///
/// This is the core result-evaluation logic, extracted from
/// `run_ktstr_test_inner` so that error message formatting can be tested
/// without booting a VM. The `repro_fn` callback handles auto-repro
/// (which requires a second VM boot) when provided.
#[allow(clippy::too_many_arguments)]
fn evaluate_vm_result(
    entry: &KtstrTestEntry,
    result: &vmm::VmResult,
    merged_assert: &crate::assert::Assert,
    stimulus_events: &[StimulusEvent],
    sockets: u32,
    cores: u32,
    threads: u32,
    repro_fn: &dyn Fn(&str) -> Option<String>,
) -> Result<AssertResult> {
    // Build timeline from stimulus events + monitor samples.
    let timeline = result
        .monitor
        .as_ref()
        .map(|m| crate::timeline::Timeline::build(stimulus_events, &m.samples));

    let sched_label = scheduler_label(&entry.scheduler.binary);
    let output = &result.output;
    let dump_section = extract_sched_ext_dump(&result.stderr)
        .map(|d| format!("\n\n--- sched_ext dump ---\n{d}"))
        .unwrap_or_default();
    let sched_log_section = parse_sched_output(output)
        .map(|s| {
            let collapsed = crate::verifier::collapse_cycles(s);
            format!("\n\n--- scheduler log ---\n{collapsed}")
        })
        .unwrap_or_default();
    let fingerprint_line = sched_log_fingerprint(output)
        .map(|fp| format!("\x1b[1;31m{fp}\x1b[0m\n"))
        .unwrap_or_default();

    let tl_ctx = crate::timeline::TimelineContext {
        kernel: extract_kernel_version(&result.stderr),
        topology: Some(format!(
            "{}s{}c{}t ({} cpus)",
            sockets,
            cores,
            threads,
            sockets * cores * threads,
        )),
        scheduler: Some(entry.scheduler.name.to_string()),
        scenario: Some(entry.name.to_string()),
        duration_s: Some(result.duration.as_secs_f64()),
    };

    if let Ok(verify_result) =
        parse_assert_result_shm(result.shm_data.as_ref()).or_else(|_| parse_assert_result(output))
    {
        // Write sidecar before checking pass/fail so both outcomes are captured.
        write_sidecar(entry, result, stimulus_events, &verify_result, "CpuSpin");

        if !verify_result.passed {
            let details = verify_result.details.join("\n  ");
            let repro = if entry.scheduler.binary.has_active_scheduling() {
                repro_fn(output)
            } else {
                None
            };
            let repro_section = repro
                .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
                .unwrap_or_default();
            let timeline_section = timeline
                .as_ref()
                .filter(|t| !t.phases.is_empty())
                .map(|t| format!("\n\n{}", t.format_with_context(&tl_ctx)))
                .unwrap_or_default();
            let stats_section = if !verify_result.stats.cgroups.is_empty() {
                let s = &verify_result.stats;
                let mut lines = vec![format!(
                    "\n\n--- stats ---\n{} workers, {} cpus, {} migrations, worst_spread={:.1}%, worst_gap={}ms",
                    s.total_workers,
                    s.total_cpus,
                    s.total_migrations,
                    s.worst_spread,
                    s.worst_gap_ms,
                )];
                for (i, cg) in s.cgroups.iter().enumerate() {
                    lines.push(format!(
                        "  cg{}: workers={} cpus={} spread={:.1}% gap={}ms migrations={} iter={}",
                        i,
                        cg.num_workers,
                        cg.num_cpus,
                        cg.spread,
                        cg.max_gap_ms,
                        cg.total_migrations,
                        cg.total_iterations,
                    ));
                }
                lines.join("\n")
            } else {
                String::new()
            };
            let console_section = if verify_result
                .details
                .iter()
                .any(|d| d.contains("scheduler died") || d.contains("scheduler crashed"))
                || verbose()
            {
                let init_stage = classify_init_stage(output);
                format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
            } else {
                String::new()
            };
            let monitor_section = if entry.scheduler.binary.has_active_scheduling()
                && let Some(ref monitor) = result.monitor
            {
                format_monitor_section(monitor, merged_assert)
            } else {
                String::new()
            };
            let msg = format!(
                "{}ktstr_test '{}'{} failed:\n  {}{}{}{}{}{}{}{}",
                fingerprint_line,
                entry.name,
                sched_label,
                details,
                stats_section,
                console_section,
                timeline_section,
                sched_log_section,
                monitor_section,
                dump_section,
                repro_section,
            );
            anyhow::bail!("{msg}");
        }

        // Evaluate monitor data against thresholds when a scheduler is running.
        // Without a scheduler (EEVDF), monitor reads rq data that may be
        // uninitialized or irrelevant — skip evaluation in that case.
        //
        // Skip early monitor warmup samples: during boot, BPF verification,
        // and initramfs unpacking the scheduler tick may not fire for hundreds
        // of milliseconds. These transient stalls are real but not indicative
        // of scheduler bugs.
        if entry.scheduler.binary.has_active_scheduling()
            && let Some(ref monitor) = result.monitor
        {
            let eval_report = trim_settle_samples(monitor);
            let thresholds = merged_assert.monitor_thresholds();
            let verdict = thresholds.evaluate(&eval_report);
            if !verdict.passed {
                let details = verdict.details.join("\n  ");
                let timeline_section = timeline
                    .as_ref()
                    .filter(|t| !t.phases.is_empty())
                    .map(|t| format!("\n\n{}", t.format_with_context(&tl_ctx)))
                    .unwrap_or_default();
                let monitor_section = format_monitor_section(monitor, merged_assert);
                let msg = format!(
                    "{}ktstr_test '{}'{} passed scenario but monitor failed:\n  {}{}{}{}{}",
                    fingerprint_line,
                    entry.name,
                    sched_label,
                    details,
                    timeline_section,
                    monitor_section,
                    sched_log_section,
                    dump_section,
                );
                anyhow::bail!("{msg}");
            }
        }

        return Ok(verify_result);
    }

    // No parseable result — no AssertResult found in SHM or COM2.
    // When a scheduler is running this typically means the scheduler died;
    // without a scheduler (EEVDF) it means the payload itself failed.
    // Attempt auto-repro if enabled and a scheduler was running.
    // Any scheduler failure that prevents producing a test result warrants
    // repro — BPF verifier failures, scx_bpf_error() exits, crashes, and
    // stalls all land here. Previous code required specific string patterns
    // ("SCHEDULER_DIED", "sched_ext:" + "disabled") which missed mid-test
    // deaths where the sched_exit_monitor writes to SHM but not COM2.
    let repro_section = if entry.scheduler.binary.has_active_scheduling() {
        repro_fn(output)
            .map(|r| format!("\n\n--- auto-repro ---\n{r}"))
            .unwrap_or_default()
    } else {
        String::new()
    };

    // Build a diagnostic section from COM1 kernel console output and exit code.
    // When COM2 has scheduler output markers, sched_log_section and dump_section
    // carry the diagnostics and the kernel console is noise (BIOS, ACPI boot).
    // When COM2 has NO scheduler output (crash before writing), the kernel console
    // is the ONLY source of crash info — include it unconditionally as a fallback.
    let has_sched_output = output.contains("===SCHED_OUTPUT_START===");
    let console_section = if !has_sched_output || verbose() {
        let init_stage = classify_init_stage(output);
        format_console_diagnostics(&result.stderr, result.exit_code, init_stage)
    } else {
        String::new()
    };

    let timeline_section = timeline
        .as_ref()
        .filter(|t| !t.phases.is_empty())
        .map(|t| format!("\n\n{}", t.format_with_context(&tl_ctx)))
        .unwrap_or_default();

    // Build monitor section for error paths where neither SHM nor COM2 had a parseable result.
    let monitor_section = if entry.scheduler.binary.has_active_scheduling()
        && let Some(ref monitor) = result.monitor
    {
        format_monitor_section(monitor, merged_assert)
    } else {
        String::new()
    };

    if result.timed_out {
        let msg = format!(
            "{}ktstr_test '{}'{} timed out (no result in SHM or COM2){}{}{}{}{}{}",
            fingerprint_line,
            entry.name,
            sched_label,
            console_section,
            timeline_section,
            sched_log_section,
            dump_section,
            monitor_section,
            repro_section,
        );
        anyhow::bail!("{msg}");
    }

    let reason = if let Some(ref shm_crash) = result.crash_message {
        format!("guest crashed:\n{shm_crash}")
    } else if let Some(crash_msg) = extract_panic_message(output) {
        format!("guest crashed: {crash_msg}")
    } else if entry.scheduler.binary.has_active_scheduling() {
        "scheduler crashed before the test could produce results".to_string()
    } else {
        "test function produced no output (no test result found)".to_string()
    };
    let msg = format!(
        "{}ktstr_test '{}'{} {}{}{}{}{}{}{}",
        fingerprint_line,
        entry.name,
        sched_label,
        reason,
        console_section,
        timeline_section,
        sched_log_section,
        dump_section,
        monitor_section,
        repro_section,
    );
    anyhow::bail!("{msg}")
}

/// Format the `--- monitor ---` section for failure output.
///
/// Shows peak values, averaged metrics, event counter rates, schedstat
/// rates, and the monitor verdict. All values are from the post-warmup
/// evaluation window (boot-settle samples trimmed).
fn format_monitor_section(
    monitor: &crate::monitor::MonitorReport,
    merged_assert: &crate::assert::Assert,
) -> String {
    let eval_report = trim_settle_samples(monitor);
    let s = &eval_report.summary;
    let thresholds = merged_assert.monitor_thresholds();
    let verdict = thresholds.evaluate(&eval_report);
    let verdict_line = if verdict.passed {
        verdict.summary.clone()
    } else {
        format!("{}: {}", verdict.summary, verdict.details.join("; "))
    };

    let mut lines = vec![
        format!(
            "samples={} max_imbalance={:.2} max_dsq_depth={} stall={}",
            s.total_samples, s.max_imbalance_ratio, s.max_local_dsq_depth, s.stall_detected,
        ),
        format!(
            "avg: imbalance={:.2} nr_running/cpu={:.1} dsq/cpu={:.1}",
            s.avg_imbalance_ratio, s.avg_nr_running, s.avg_local_dsq_depth,
        ),
    ];

    if let Some(ref ev) = s.event_deltas {
        lines.push(format!(
            "events: fallback={} ({:.1}/s) keep_last={} ({:.1}/s) offline={}",
            ev.total_fallback,
            ev.fallback_rate,
            ev.total_dispatch_keep_last,
            ev.keep_last_rate,
            ev.total_dispatch_offline,
        ));
    }

    if let Some(ref ss) = s.schedstat_deltas {
        lines.push(format!(
            "schedstat: csw={} ({:.0}/s) run_delay={:.0}ns/s ttwu={} goidle={}",
            ss.total_sched_count,
            ss.sched_count_rate,
            ss.run_delay_rate,
            ss.total_ttwu_count,
            ss.total_sched_goidle,
        ));
    }

    if let Some(ref progs) = s.prog_stats_deltas {
        for p in progs {
            if p.cnt > 0 {
                lines.push(format!(
                    "bpf: {} cnt={} {:.0}ns/call",
                    p.name, p.cnt, p.nsecs_per_call,
                ));
            }
        }
    }

    lines.push(format!("verdict: {verdict_line}"));

    format!("\n\n--- monitor ---\n{}", lines.join("\n"))
}

/// Number of monitor samples to skip at the start of evaluation.
///
/// During VM boot the kernel performs BPF verification, initramfs
/// unpacking, and scheduler loading. These memory-intensive operations
/// cause the scheduler tick to stall for hundreds of milliseconds.
/// The stalls are real but transient — evaluating them produces false
/// positives, especially in low-memory VMs.
///
/// 20 samples at ~100ms interval = ~2 seconds of warmup. This covers
/// the boot settling period after the scheduler attaches.
const MONITOR_WARMUP_SAMPLES: usize = 20;

/// Skip boot-settle samples from a MonitorReport for threshold evaluation.
///
/// Returns a report with the first `MONITOR_WARMUP_SAMPLES` removed so
/// that transient boot-time stalls don't trigger sustained-window
/// violations.
fn trim_settle_samples(report: &crate::monitor::MonitorReport) -> crate::monitor::MonitorReport {
    if report.samples.len() <= MONITOR_WARMUP_SAMPLES {
        return report.clone();
    }

    let trimmed = report.samples[MONITOR_WARMUP_SAMPLES..].to_vec();
    let summary = crate::monitor::MonitorSummary::from_samples_with_threshold(
        &trimmed,
        report.preemption_threshold_ns,
    );
    crate::monitor::MonitorReport {
        samples: trimmed,
        summary,
        preemption_threshold_ns: report.preemption_threshold_ns,
    }
}

/// Sentinel value for `--ktstr-probe-stack` when no crash stack functions
/// were extracted. Triggers the guest-side probe path so
/// `discover_bpf_symbols()` can dynamically find the scheduler's BPF
/// programs. `filter_traceable` drops it (not in kallsyms).
const DISCOVER_SENTINEL: &str = "__discover__";

/// Attempt auto-repro: extract stack functions from COM2 scheduler output
/// or COM1 kernel console (fallback), boot a second VM with BPF probes
/// attached, and return formatted probe data. When no stack functions are
/// available (e.g. BPF text error without backtrace), falls back to
/// dynamic BPF program discovery in the repro VM.
/// `console_output` is COM1 kernel console text, used when COM2 has no
/// extractable functions (e.g. scheduler died before writing output).
/// Returns `None` if repro cannot be attempted or yields no data.
fn attempt_auto_repro(
    entry: &KtstrTestEntry,
    kernel: &Path,
    scheduler: Option<&Path>,
    ktstr_bin: &Path,
    first_vm_output: &str,
    console_output: &str,
    topo: Option<&TopoOverride>,
) -> Option<String> {
    use crate::probe::stack::extract_stack_functions_all;

    // Extract scheduler log from COM2 output.
    eprintln!(
        "ktstr_test: auto-repro: COM2 length={} has_sched_start={} has_sched_end={}",
        first_vm_output.len(),
        first_vm_output.contains("===SCHED_OUTPUT_START==="),
        first_vm_output.contains("===SCHED_OUTPUT_END==="),
    );
    let sched_output = parse_sched_output(first_vm_output);

    // Extract function names from COM2 scheduler log first, then
    // fall back to COM1 kernel console (which has kernel backtraces
    // including sched_ext_dump output).
    let stack_funcs = if let Some(sched) = sched_output {
        let funcs = extract_stack_functions_all(sched);
        if funcs.is_empty() {
            eprintln!("ktstr_test: auto-repro: no functions from COM2, trying COM1");
            extract_stack_functions_all(console_output)
        } else {
            funcs
        }
    } else {
        eprintln!("ktstr_test: auto-repro: no scheduler output on COM2, trying COM1");
        extract_stack_functions_all(console_output)
    };
    let func_names: Vec<String> = stack_funcs.iter().map(|f| f.raw_name.clone()).collect();

    // When no stack functions were extracted (e.g. BPF text error with no
    // backtrace), still boot the repro VM. The guest-side discover_bpf_symbols()
    // dynamically finds the scheduler's BPF programs. Pass a sentinel value
    // so extract_probe_stack_arg returns Some and the guest probe path activates.
    let probe_arg = if func_names.is_empty() {
        eprintln!("ktstr_test: auto-repro: no stack functions, using BPF discovery in repro VM");
        format!("--ktstr-probe-stack={DISCOVER_SENTINEL}")
    } else {
        eprintln!(
            "ktstr_test: auto-repro: probing {} functions in second VM",
            func_names.len()
        );
        format!("--ktstr-probe-stack={}", func_names.join(","))
    };

    // Build guest args for the repro VM.
    let guest_args = vec![
        "run".to_string(),
        "--ktstr-test-fn".to_string(),
        entry.name.to_string(),
        probe_arg,
    ];

    // Build cmdline: base args + sysctls + kargs (same as first VM).
    let mut cmdline_parts = vec!["iomem=relaxed".to_string()];
    for &(key, value) in entry.scheduler.sysctls {
        cmdline_parts.push(format!("sysctl.{}={}", key, value));
    }
    for &karg in entry.scheduler.kargs {
        cmdline_parts.push(karg.to_string());
    }
    if let Ok(bt) = std::env::var("RUST_BACKTRACE") {
        cmdline_parts.push(format!("RUST_BACKTRACE={bt}"));
    }
    if let Ok(log) = std::env::var("RUST_LOG") {
        cmdline_parts.push(format!("RUST_LOG={log}"));
    }
    let cmdline_extra = cmdline_parts.join(" ");

    let (sockets, cores, threads, memory_mb) = match topo {
        Some(t) => (t.sockets, t.cores, t.threads, t.memory_mb),
        None => {
            let cpus = entry.topology.total_cpus();
            let mem = (cpus * 64).max(256).max(entry.memory_mb);
            (
                entry.topology.sockets,
                entry.topology.cores_per_socket,
                entry.topology.threads_per_core,
                mem,
            )
        }
    };

    let mut builder = vmm::KtstrVm::builder()
        .kernel(kernel)
        .init_binary(ktstr_bin)
        .topology(sockets, cores, threads)
        .memory_deferred_min(memory_mb)
        .cmdline(&cmdline_extra)
        .shm_size(KTSTR_TEST_SHM_SIZE)
        .run_args(&guest_args)
        .timeout(Duration::from_secs(60));

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }

    // Merge scheduler args: cgroup_parent + scheduler sched_args + per-test.
    {
        let mut args: Vec<String> = Vec::new();
        if let Some(cgroup_path) = entry.scheduler.cgroup_parent {
            args.push("--cell-parent-cgroup".to_string());
            args.push(cgroup_path.to_string());
        }
        args.extend(entry.scheduler.sched_args.iter().map(|s| s.to_string()));
        args.extend(entry.extra_sched_args.iter().map(|s| s.to_string()));
        if !args.is_empty() {
            builder = builder.sched_args(&args);
        }
    }

    // Do NOT forward bpf_map_write to the repro VM. The repro VM
    // runs the same workload with probes attached while the scheduler
    // is alive. Probes capture sched_ext events during normal
    // scheduling. No crash needed — the interesting data is how the
    // probed functions behave under the same workload that caused
    // the crash in the first VM.

    let vm = match builder.build() {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("ktstr_test: auto-repro: failed to build VM: {e:#}");
            return None;
        }
    };

    let repro_result = match vm.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ktstr_test: auto-repro: VM run failed: {e:#}");
            return None;
        }
    };

    // Forward guest stderr (COM1) and COM2 probe lines when verbose.
    if verbose() {
        eprintln!(
            "ktstr_test: auto-repro: COM1 stderr length={} COM2 stdout length={}",
            repro_result.stderr.len(),
            repro_result.output.len(),
        );
        for line in repro_result.stderr.lines() {
            eprintln!("  repro-vm-com1: {line}");
        }
        let mut in_probe = false;
        for line in repro_result.output.lines() {
            if line.contains("ktstr_test: probe:") {
                in_probe = true;
            }
            if in_probe {
                eprintln!("  repro-vm-com2: {line}");
            }
        }
    }

    // Extract probe JSON from the repro VM and format on the host with
    // kernel_dir so blazesym can resolve source locations via vmlinux DWARF.
    // Canonicalize the kernel path first so relative paths resolve correctly.
    let canon_kernel = std::fs::canonicalize(kernel).ok();
    let kernel_dir = canon_kernel
        .as_ref()
        .and_then(|p| p.to_str())
        .and_then(|p| {
            #[cfg(target_arch = "x86_64")]
            {
                p.strip_suffix("/arch/x86/boot/bzImage")
            }
            #[cfg(target_arch = "aarch64")]
            {
                p.strip_suffix("/arch/arm64/boot/Image")
            }
        })
        .map(|s| s.to_string());
    let kernel_dir_str = kernel_dir.as_deref();
    extract_probe_output(&repro_result.output, kernel_dir_str)
}

/// Delimiters for probe output in guest COM2 (written by collect_and_print_probe_data).
const PROBE_OUTPUT_START: &str = "===PROBE_OUTPUT_START===";
const PROBE_OUTPUT_END: &str = "===PROBE_OUTPUT_END===";

/// Extract probe JSON from guest COM2, deserialize, and format on the
/// host where vmlinux (DWARF) is available for source locations.
fn extract_probe_output(output: &str, kernel_dir: Option<&str>) -> Option<String> {
    let json = crate::probe::output::extract_section(output, PROBE_OUTPUT_START, PROBE_OUTPUT_END);
    if json.is_empty() {
        return None;
    }
    let payload: ProbePayload = match serde_json::from_str(&json) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("ktstr_test: probe payload deserialize failed: {e}");
            return None;
        }
    };
    let mut out = String::new();

    // Append pipeline diagnostics if present.
    if let Some(ref diag) = payload.diagnostics {
        out.push_str(&format_probe_diagnostics(&diag.pipeline, &diag.skeleton));
    }

    if payload.events.is_empty() {
        if out.is_empty() {
            return None;
        }
        return Some(out);
    }
    out.push_str(&crate::probe::output::format_probe_events_with_bpf_locs(
        &payload.events,
        &payload.func_names,
        kernel_dir,
        &payload.bpf_source_locs,
    ));
    Some(out)
}

/// Format probe pipeline diagnostics into a human-readable summary.
pub(crate) fn format_probe_diagnostics(
    pipeline: &PipelineDiagnostics,
    skeleton: &crate::probe::process::ProbeDiagnostics,
) -> String {
    let mut out = String::new();
    out.push_str("--- probe pipeline ---\n");

    // Stage 1: extraction
    out.push_str(&format!(
        "  extracted:   {} functions from crash backtrace\n",
        pipeline.stack_extracted,
    ));

    // Stage 2: filter
    let passed = pipeline.stack_extracted as usize - pipeline.filter_dropped.len();
    if pipeline.filter_dropped.is_empty() {
        out.push_str(&format!("  traceable:   {passed} passed filter\n"));
    } else {
        out.push_str(&format!(
            "  traceable:   {passed} passed, {} dropped: {}\n",
            pipeline.filter_dropped.len(),
            pipeline.filter_dropped.join(", "),
        ));
    }

    // Stage 3: BPF discovery
    out.push_str(&format!(
        "  bpf_discover: {} programs found\n",
        pipeline.bpf_discovered,
    ));

    // Stage 4: expansion
    out.push_str(&format!(
        "  after_expand: {} total probe targets\n",
        pipeline.total_after_expand,
    ));

    // Stage 5: kprobe attach
    if skeleton.kprobe_attach_failed.is_empty() {
        out.push_str(&format!(
            "  kprobes:     {} attached\n",
            skeleton.kprobe_attached,
        ));
    } else {
        out.push_str(&format!(
            "  kprobes:     {} attached, {} failed: {}\n",
            skeleton.kprobe_attached,
            skeleton.kprobe_attach_failed.len(),
            skeleton
                .kprobe_attach_failed
                .iter()
                .map(|(n, e)| format!("{n} ({e})"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    if !skeleton.kprobe_resolve_failed.is_empty() {
        out.push_str(&format!(
            "  kprobe_miss: {} unresolved: {}\n",
            skeleton.kprobe_resolve_failed.len(),
            skeleton.kprobe_resolve_failed.join(", "),
        ));
    }

    // Stage 6: fentry attach
    if skeleton.fentry_candidates > 0 {
        if skeleton.fentry_attach_failed.is_empty() {
            out.push_str(&format!(
                "  fentry:      {} attached\n",
                skeleton.fentry_attached,
            ));
        } else {
            out.push_str(&format!(
                "  fentry:      {} attached, {} failed: {}\n",
                skeleton.fentry_attached,
                skeleton.fentry_attach_failed.len(),
                skeleton
                    .fentry_attach_failed
                    .iter()
                    .map(|(n, e)| format!("{n} ({e})"))
                    .collect::<Vec<_>>()
                    .join(", "),
            ));
        }
    }

    // Stage 7: trigger
    let trigger_type = if skeleton.trigger_type.is_empty() {
        "unknown"
    } else {
        &skeleton.trigger_type
    };
    if let Some(ref err) = skeleton.trigger_attach_error {
        out.push_str(&format!("  trigger:     attach failed ({err})\n"));
    } else {
        out.push_str(&format!(
            "  trigger:     {} ({})\n",
            if skeleton.trigger_fired {
                "fired"
            } else {
                "not fired"
            },
            trigger_type,
        ));
    }

    // Stage 8: capture
    out.push_str(&format!(
        "  probe_data:  {} keys, {} unmatched IPs\n",
        skeleton.probe_data_keys, skeleton.probe_data_unmatched_ips,
    ));

    // Stage 9: events + stitching
    out.push_str(&format!(
        "  events:      {} captured, {} after stitch\n",
        skeleton.events_before_stitch, skeleton.events_after_stitch,
    ));

    // Stage 10: BPF-side counters
    if skeleton.bpf_kprobe_fires > 0
        || skeleton.bpf_trigger_fires > 0
        || skeleton.bpf_meta_misses > 0
    {
        out.push_str(&format!(
            "  bpf_counts:  {} kprobe fires, {} trigger fires, {} meta misses\n",
            skeleton.bpf_kprobe_fires, skeleton.bpf_trigger_fires, skeleton.bpf_meta_misses,
        ));
        if !skeleton.bpf_miss_ips.is_empty() {
            let ips: Vec<String> = skeleton
                .bpf_miss_ips
                .iter()
                .map(|ip| format!("0x{ip:x}"))
                .collect();
            out.push_str(&format!("  miss_ips:    {}\n", ips.join(", ")));
        }
    }

    out
}

/// Setup function for nextest `setup-script` integration.
///
/// Validates KVM access, discovers a kernel, writes `KTSTR_TEST_KERNEL`
/// to `env_writer`, and warms the SHM initramfs cache for each binary.
pub fn nextest_setup(binaries: &[&Path], env_writer: &mut dyn Write) -> Result<()> {
    ensure_kvm()?;
    let kernel = resolve_test_kernel()?;
    writeln!(env_writer, "KTSTR_TEST_KERNEL={}", kernel.display())
        .context("write KTSTR_TEST_KERNEL to env")?;

    for bin in binaries {
        let key = vmm::BaseKey::new(bin, None)?;
        let _ = vmm::get_or_build_base(bin, &[], &[], false, &key)?;
    }

    Ok(())
}

/// Guest-side dispatch: check for `--ktstr-test-fn=NAME` in args, run the
/// registered function, write the result to SHM and stdout (COM2),
/// and exit. Profraw flush is handled by `try_flush_profraw()` in the
/// ctor before `std::process::exit()`.
///
/// Called from `ktstr_test_early_dispatch()` (ctor) before `main()`, or
/// from `ktstr_guest_init()` when running as PID 1.
///
/// When called from PID 1 context, args must be pre-loaded into the
/// process args (the caller reads `/args` from the initramfs).
/// Returns `Some(exit_code)` if dispatched, `None` if not an
/// ktstr_test invocation.
pub fn maybe_dispatch_vm_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    maybe_dispatch_vm_test_with_args(&args)
}

/// Like `maybe_dispatch_vm_test` but with explicit args. Used by
/// `ktstr_guest_init()` which reads args from `/args` in the initramfs.
pub(crate) fn maybe_dispatch_vm_test_with_args(args: &[String]) -> Option<i32> {
    let name = extract_test_fn_arg(args)?;

    // Propagate RUST_BACKTRACE and RUST_LOG from kernel cmdline to env.
    if let Ok(cmdline) = std::fs::read_to_string("/proc/cmdline") {
        let parts: Vec<&str> = cmdline.split_whitespace().collect();
        if let Some(val) = parts
            .iter()
            .find(|s| s.starts_with("RUST_BACKTRACE="))
            .and_then(|s| s.strip_prefix("RUST_BACKTRACE="))
        {
            // SAFETY: guest-side dispatch runs single-threaded before
            // any test threads are spawned.
            unsafe { std::env::set_var("RUST_BACKTRACE", val) };
        }
        if let Some(val) = parts
            .iter()
            .find(|s| s.starts_with("RUST_LOG="))
            .and_then(|s| s.strip_prefix("RUST_LOG="))
        {
            unsafe { std::env::set_var("RUST_LOG", val) };
        }
    }

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("ktstr_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    // Parse --ktstr-probe-stack=func1,func2,... for auto-repro mode.
    let probe_stack = extract_probe_stack_arg(args);

    // Parse --ktstr-work-type=NAME for work type override.
    let work_type_override = extract_work_type_arg(args).and_then(|s| {
        crate::workload::WorkType::from_name(&s).or_else(|| {
            eprintln!("ktstr_test: unknown work type '{s}'");
            None
        })
    });

    // Set up BPF probes if --ktstr-probe-stack was provided.
    let probe_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let probe_handle: Option<ProbeHandle> = probe_stack.as_ref().and_then(|stack_input| {
        use crate::probe::stack::load_probe_stack;

        eprintln!("ktstr_test: probe: loading probe stack from --ktstr-probe-stack");
        let mut pipe_diag = PipelineDiagnostics::default();
        let raw_functions = load_probe_stack(stack_input);
        pipe_diag.stack_extracted = raw_functions.len() as u32;
        let pre_filter: Vec<String> = raw_functions.iter().map(|f| f.raw_name.clone()).collect();
        let mut functions = crate::probe::stack::filter_traceable(raw_functions);
        // Record which functions were dropped by filter_traceable.
        for name in &pre_filter {
            if !functions.iter().any(|f| f.raw_name == *name) {
                pipe_diag.filter_dropped.push(name.clone());
            }
        }
        // Discover BPF scheduler functions from the running scheduler.
        // Stack-extracted BPF names have stale prog IDs from the first VM;
        // discover_bpf_symbols finds the current scheduler's programs.
        let stack_display_names: Vec<&str> = functions
            .iter()
            .filter(|f| f.is_bpf)
            .map(|f| f.display_name.as_str())
            .collect();
        let bpf_syms = crate::probe::btf::discover_bpf_symbols(&stack_display_names);
        pipe_diag.bpf_discovered = bpf_syms.len() as u32;
        if !bpf_syms.is_empty() {
            eprintln!(
                "ktstr_test: probe: {} BPF symbols discovered",
                bpf_syms.len()
            );
            functions.extend(bpf_syms);
        }
        // Expand BPF functions to kernel-side callers for bridge kprobes,
        // keeping BPF functions for fentry attachment.
        let functions = crate::probe::stack::expand_bpf_to_kernel_callers(functions);
        pipe_diag.total_after_expand = functions.len() as u32;
        if functions.is_empty() {
            eprintln!("ktstr_test: no traceable functions from --ktstr-probe-stack");
            return None;
        }

        eprintln!(
            "ktstr_test: probe: {} functions loaded, spawning probe thread",
            functions.len()
        );

        // Resolve BTF signatures for kernel functions so probe output
        // gets decoded field names instead of raw register values.
        let kernel_names: Vec<&str> = functions
            .iter()
            .filter(|f| !f.is_bpf)
            .map(|f| f.raw_name.as_str())
            .collect();
        let mut btf_funcs = crate::probe::btf::parse_btf_functions(&kernel_names, None);
        // Parse BPF function signatures from BPF program BTF.
        let bpf_btf_args: Vec<(&str, u32)> = functions
            .iter()
            .filter(|f| f.is_bpf)
            .filter_map(|f| Some((f.display_name.as_str(), f.bpf_prog_id?)))
            .collect();
        if !bpf_btf_args.is_empty() {
            btf_funcs.extend(crate::probe::btf::parse_bpf_btf_functions(&bpf_btf_args));
        }

        // Build func_names from the filtered list so indices match
        // the func_idx values assigned by run_probe_skeleton.
        let func_names: Vec<(u32, String)> = functions
            .iter()
            .enumerate()
            .map(|(i, f)| (i as u32, f.display_name.clone()))
            .collect();

        // Pre-open BPF program FDs while the scheduler is alive.
        // Holding these FDs keeps programs alive via kernel refcounting
        // even after the scheduler crashes.
        let bpf_fds = crate::probe::process::open_bpf_prog_fds(&functions);
        let stop = probe_stop.clone();
        let funcs = functions.clone();
        let fn_names = func_names.clone();
        let pd = pipe_diag.clone();
        let output_done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let output_done_thread = output_done.clone();
        let probes_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let probes_ready_thread = probes_ready.clone();
        let handle = std::thread::spawn(move || {
            use crate::probe::process::run_probe_skeleton;
            let (events, diag) =
                run_probe_skeleton(&funcs, &btf_funcs, &stop, &bpf_fds, &probes_ready_thread);
            // Serialize probe output immediately so it reaches COM2
            // even if the test function hangs and never calls
            // collect_and_print_probe_data.
            emit_probe_payload(events.as_deref().unwrap_or(&[]), &fn_names, &pd, &diag);
            output_done_thread.store(true, std::sync::atomic::Ordering::Release);
            (events, diag)
        });

        // Wait for probes to attach before starting the test function.
        // Without this, the test may crash the scheduler before probes
        // are active, resulting in 0 captured events.
        while !probes_ready.load(std::sync::atomic::Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(10));
        }

        Some((handle, func_names, pipe_diag, output_done))
    });

    // Build a minimal Ctx for the test function.
    // Prefer sysfs topology, but fall back to VM spec if sysfs fails or
    // reports the wrong LLC count (e.g. unpatched CPUID cache leaves).
    let spec_topo = crate::topology::TestTopology::from_spec(
        entry.topology.sockets,
        entry.topology.cores_per_socket,
        entry.topology.threads_per_core,
    );
    let topo = match crate::topology::TestTopology::from_system() {
        Ok(sys) if sys.num_llcs() == spec_topo.num_llcs() => sys,
        Ok(sys) => {
            eprintln!(
                "ktstr_test: sysfs reports {} LLCs, VM spec expects {}; using spec fallback",
                sys.num_llcs(),
                spec_topo.num_llcs(),
            );
            spec_topo
        }
        Err(e) => {
            eprintln!("ktstr_test: topology from sysfs failed ({e}), using VM spec fallback");
            spec_topo
        }
    };
    let cgroup_root = resolve_cgroup_root(args);
    let cgroups = crate::cgroup::CgroupManager::new(&cgroup_root);
    if let Err(e) = cgroups.setup(false) {
        eprintln!("ktstr_test: cgroup setup failed: {e}");
    }
    let sched_pid = std::env::var("SCHED_PID")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let workers_per_cgroup = entry.workers_per_cgroup as usize;
    // Three-layer merge: default_checks -> scheduler.assert -> entry.assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    let ctx = Ctx {
        cgroups: &cgroups,
        topo: &topo,
        duration: entry.duration,
        workers_per_cgroup,
        sched_pid,
        settle: Duration::from_millis(500),
        work_type_override,
        assert: merged_assert,
        wait_for_map_write: entry.bpf_map_write.is_some(),
    };

    let result = match (entry.func)(&ctx) {
        Ok(r) => r,
        Err(e) => {
            let r = AssertResult {
                passed: false,
                details: vec![format!("{e:#}")],
                stats: Default::default(),
            };
            print_assert_result(&r);
            collect_and_print_probe_data(probe_stop, probe_handle);
            return Some(1);
        }
    };

    let exit_code = if result.passed { 0 } else { 1 };
    print_assert_result(&result);
    collect_and_print_probe_data(probe_stop, probe_handle);
    Some(exit_code)
}

type ProbeHandle = (
    std::thread::JoinHandle<(
        Option<Vec<crate::probe::process::ProbeEvent>>,
        crate::probe::process::ProbeDiagnostics,
    )>,
    Vec<(u32, String)>,
    PipelineDiagnostics,
    std::sync::Arc<std::sync::atomic::AtomicBool>, // output_done
);

/// Pre-skeleton pipeline diagnostics captured during guest probe setup.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct PipelineDiagnostics {
    /// Functions from --ktstr-probe-stack before filter.
    pub stack_extracted: u32,
    /// Functions dropped by filter_traceable.
    pub filter_dropped: Vec<String>,
    /// BPF symbols discovered from running scheduler.
    pub bpf_discovered: u32,
    /// Functions after expand_bpf_to_kernel_callers.
    pub total_after_expand: u32,
}

/// Serialized probe data sent from guest to host via COM2.
/// The host deserializes and formats with kernel_dir for source locations.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbePayload {
    pub events: Vec<crate::probe::process::ProbeEvent>,
    pub func_names: Vec<(u32, String)>,
    #[serde(default)]
    pub bpf_source_locs: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub diagnostics: Option<ProbePayloadDiagnostics>,
}

/// Combined diagnostics for the probe payload.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbePayloadDiagnostics {
    #[serde(default)]
    pub pipeline: PipelineDiagnostics,
    #[serde(default)]
    pub skeleton: crate::probe::process::ProbeDiagnostics,
}

/// Serialize probe payload to stdout (COM2) between delimiters.
/// Resolves BPF source locations from loaded programs before serializing.
fn emit_probe_payload(
    events: &[crate::probe::process::ProbeEvent],
    func_names: &[(u32, String)],
    pipeline_diag: &PipelineDiagnostics,
    skeleton_diag: &crate::probe::process::ProbeDiagnostics,
) {
    let source_loc_names: Vec<&str> = func_names.iter().map(|(_, name)| name.as_str()).collect();
    let bpf_syms = crate::probe::btf::discover_bpf_symbols(&source_loc_names);
    let bpf_prog_ids: Vec<u32> = func_names
        .iter()
        .filter_map(|(_, name)| {
            bpf_syms
                .iter()
                .find(|s| s.display_name == *name)
                .and_then(|s| s.bpf_prog_id)
        })
        .collect();
    let bpf_source_locs = crate::probe::btf::resolve_bpf_source_locs(&bpf_prog_ids);

    let payload = ProbePayload {
        events: events.to_vec(),
        func_names: func_names.to_vec(),
        bpf_source_locs,
        diagnostics: Some(ProbePayloadDiagnostics {
            pipeline: pipeline_diag.clone(),
            skeleton: skeleton_diag.clone(),
        }),
    };
    println!("{PROBE_OUTPUT_START}");
    if let Ok(json) = serde_json::to_string(&payload) {
        println!("{json}");
    }
    println!("{PROBE_OUTPUT_END}");
}

/// Stop probes, join the probe thread. The probe thread emits output
/// directly when the trigger fires; this function only needs to set
/// `stop` and join. If the probe thread already emitted output, this
/// is a no-op.
fn collect_and_print_probe_data(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    let Some((handle, func_names, pipeline_diag, output_done)) = handle else {
        return;
    };

    stop.store(true, std::sync::atomic::Ordering::Release);
    let (events, skeleton_diag) = match handle.join() {
        Ok((Some(events), diag)) => (events, diag),
        Ok((None, diag)) => (Vec::new(), diag),
        Err(_) => (
            Vec::new(),
            crate::probe::process::ProbeDiagnostics::default(),
        ),
    };

    // The probe thread already emitted output on trigger/stop.
    // Only emit here if it somehow didn't (e.g. thread panicked
    // before reaching emit_probe_payload).
    if !output_done.load(std::sync::atomic::Ordering::Acquire) {
        emit_probe_payload(&events, &func_names, &pipeline_diag, &skeleton_diag);
    }
}

// ---------------------------------------------------------------------------
// Profraw handling
// ---------------------------------------------------------------------------

/// Flush LLVM coverage profraw to the SHM ring buffer.
///
/// Calls `__llvm_profile_set_filename` to set the output path, then
/// `__llvm_profile_write_file` to write profraw to a tmpfs file inside
/// the guest. Reads the file back and writes the contents to the SHM
/// ring for host-side extraction.
///
/// All symbols have hidden visibility in compiler-rt, so we resolve
/// them via ELF .symtab parsing (dlsym cannot find hidden symbols).
///
/// No-op when built without `-C instrument-coverage` or when SHM
/// parameters are absent from the kernel command line.
pub(crate) fn try_flush_profraw() {
    let Some((shm_base, shm_size)) = parse_shm_params() else {
        return;
    };

    let exe = match std::fs::read("/proc/self/exe") {
        Ok(data) => data,
        Err(_) => return,
    };
    let slide = pie_load_bias(&exe);

    // Resolve both symbols in a single pass through the ELF .symtab.
    let vaddrs = find_symbol_vaddrs(
        &exe,
        &["__llvm_profile_initialize", "__llvm_profile_write_file"],
    );

    // Set profraw output path, then call __llvm_profile_initialize to
    // read it and register the atexit handler.
    // SAFETY: single-threaded guest dispatch context.
    unsafe { std::env::set_var("LLVM_PROFILE_FILE", "/tmp/ktstr.profraw") };
    if let Some(vaddr) = vaddrs[0]
        && vaddr != 0
    {
        let f: extern "C" fn() =
            unsafe { std::mem::transmute((vaddr as usize).wrapping_add(slide)) };
        f();
    }

    // Write profraw to the file.
    let write_file_vaddr = match vaddrs[1] {
        Some(v) if v != 0 => v,
        _ => return,
    };
    let write_file: extern "C" fn() -> i32 =
        unsafe { std::mem::transmute((write_file_vaddr as usize).wrapping_add(slide)) };
    if write_file() != 0 {
        return;
    }

    // Read the profraw file and send through SHM ring.
    let data = match std::fs::read("/tmp/ktstr.profraw") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let _ = write_to_shm_ring(shm_base, shm_size, MSG_TYPE_PROFRAW, &data);
}

/// Resolve multiple symbol virtual addresses in a single pass through
/// the ELF .symtab. Returns addresses in the same order as `names`.
fn find_symbol_vaddrs(data: &[u8], names: &[&str]) -> Vec<Option<u64>> {
    let mut results = vec![None; names.len()];
    let mut remaining = names.len();

    let elf = match goblin::elf::Elf::parse(data) {
        Ok(e) => e,
        Err(_) => return results,
    };

    for sym in elf.syms.iter() {
        if remaining == 0 {
            break;
        }
        if sym.st_size == 0 {
            continue;
        }
        let sym_name = match elf.strtab.get_at(sym.st_name) {
            Some(n) => n,
            None => continue,
        };
        for (i, name) in names.iter().enumerate() {
            if results[i].is_none() && sym_name == *name {
                results[i] = Some(sym.st_value);
                remaining -= 1;
                break;
            }
        }
    }
    results
}

/// Compute the ASLR load bias for a PIE binary.
///
/// For ET_DYN (PIE), the kernel loads the binary at an arbitrary base.
/// The bias is `runtime_phdr_addr - file_phdr_offset`. We get the
/// runtime phdr address from AT_PHDR (via getauxval) and the file
/// offset from e_phoff.
///
/// Returns 0 for ET_EXEC (non-PIE), where st_value is already absolute.
fn pie_load_bias(data: &[u8]) -> usize {
    let elf = match goblin::elf::Elf::parse(data) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    if elf.header.e_type != goblin::elf::header::ET_DYN {
        return 0;
    }

    let phdr_file_offset = elf.header.e_phoff as usize;
    // SAFETY: AT_PHDR is a well-defined auxiliary vector key.
    let phdr_runtime = unsafe { libc::getauxval(libc::AT_PHDR) } as usize;
    if phdr_runtime == 0 {
        return 0;
    }
    phdr_runtime.wrapping_sub(phdr_file_offset)
}

/// Parse KTSTR_SHM_BASE and KTSTR_SHM_SIZE from /proc/cmdline.
fn parse_shm_params() -> Option<(u64, u64)> {
    let cmdline = std::fs::read_to_string("/proc/cmdline").ok()?;
    vmm::shm_ring::parse_shm_params_from_str(&cmdline)
}

/// Write a TLV message to the SHM ring buffer via /dev/mem mmap.
fn write_to_shm_ring(shm_base: u64, shm_size: u64, msg_type: u32, payload: &[u8]) -> Result<()> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::OpenOptionsExt;

    let fd = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_SYNC)
        .open("/dev/mem")
        .context("open /dev/mem")?;

    let m = vmm::shm_ring::mmap_devmem(
        std::os::unix::io::AsRawFd::as_raw_fd(&fd),
        shm_base,
        shm_size,
    )
    .ok_or_else(|| anyhow::anyhow!("mmap /dev/mem failed"))?;

    let shm_buf = unsafe { std::slice::from_raw_parts_mut(m.ptr, shm_size as usize) };

    let written = vmm::shm_ring::shm_write(shm_buf, 0, msg_type, payload);

    unsafe {
        libc::munmap(m.map_base, m.map_size);
    }

    if written == 0 {
        anyhow::bail!(
            "SHM ring full: failed to write {} byte payload",
            payload.len()
        );
    }

    Ok(())
}

static PROFRAW_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Write profraw data to the llvm-cov-target directory.
fn write_profraw(data: &[u8]) -> Result<()> {
    let target_dir = target_dir();
    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("create profraw dir: {}", target_dir.display()))?;
    let id = PROFRAW_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = target_dir.join(format!("ktstr-test-{}-{}.profraw", std::process::id(), id));
    std::fs::write(&path, data).with_context(|| format!("write profraw: {}", path.display()))?;
    Ok(())
}

/// Resolve the llvm-cov-target directory for profraw output.
fn target_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LLVM_COV_TARGET_DIR") {
        return PathBuf::from(d);
    }
    if let Some(parent) = std::env::var("LLVM_PROFILE_FILE")
        .ok()
        .as_ref()
        .and_then(|p| Path::new(p).parent())
    {
        return parent.to_path_buf();
    }
    let mut p = crate::resolve_current_exe().unwrap_or_else(|_| std::env::temp_dir());
    p.pop(); // remove binary name
    p.push("llvm-cov-target");
    p
}

// ---------------------------------------------------------------------------
// Result serialization
// ---------------------------------------------------------------------------

/// Delimiters for the AssertResult JSON in guest output.
const RESULT_START: &str = "===KTSTR_TEST_RESULT_START===";
const RESULT_END: &str = "===KTSTR_TEST_RESULT_END===";

/// Write AssertResult to SHM (primary) and stdout/COM2 (fallback).
fn print_assert_result(r: &AssertResult) {
    if let Ok(json) = serde_json::to_string(r) {
        vmm::shm_ring::write_msg(vmm::shm_ring::MSG_TYPE_TEST_RESULT, json.as_bytes());
        println!("{RESULT_START}");
        println!("{json}");
        println!("{RESULT_END}");
    }
}

/// Extract AssertResult from SHM drain entries.
fn parse_assert_result_shm(shm: Option<&vmm::shm_ring::ShmDrainResult>) -> Result<AssertResult> {
    let shm = shm.ok_or_else(|| anyhow::anyhow!("no SHM data"))?;
    let entry = shm
        .entries
        .iter()
        .rev()
        .find(|e| e.msg_type == vmm::shm_ring::MSG_TYPE_TEST_RESULT && e.crc_ok)
        .ok_or_else(|| anyhow::anyhow!("no test result in SHM"))?;
    serde_json::from_slice(&entry.payload).context("parse AssertResult from SHM")
}

/// Parse AssertResult from guest COM2 output between delimiters.
fn parse_assert_result(output: &str) -> Result<AssertResult> {
    let json = crate::probe::output::extract_section(output, RESULT_START, RESULT_END);
    anyhow::ensure!(!json.is_empty(), "missing result delimiters");
    serde_json::from_str(&json).context("parse AssertResult JSON")
}

// ---------------------------------------------------------------------------
// Scheduler output extraction
// ---------------------------------------------------------------------------

/// Delimiters for the scheduler log in guest output (written by init script).
const SCHED_OUTPUT_START: &str = "===SCHED_OUTPUT_START===";
const SCHED_OUTPUT_END: &str = "===SCHED_OUTPUT_END===";

/// Extract the last non-empty line from the scheduler log.
///
/// This serves as a failure fingerprint: when many tests fail with the
/// same scheduler error, the fingerprint makes identical failures
/// visually obvious in nextest output.
fn sched_log_fingerprint(output: &str) -> Option<&str> {
    let log = parse_sched_output(output)?;
    log.lines().rev().find(|l| !l.trim().is_empty())
}

/// Extract the scheduler log from guest output between delimiters.
/// Returns `None` if the delimiters are absent or the content is empty.
fn parse_sched_output(output: &str) -> Option<&str> {
    // Cannot use extract_section here: it returns an owned String,
    // but callers need a borrowed &str tied to `output`'s lifetime.
    let start = output.find(SCHED_OUTPUT_START)?;
    let end = output.find(SCHED_OUTPUT_END)?;
    let after_marker = start + SCHED_OUTPUT_START.len();
    if after_marker >= end {
        return None;
    }
    let content = output[after_marker..end].trim();
    if content.is_empty() {
        return None;
    }
    Some(content)
}

// ---------------------------------------------------------------------------
// sched_ext dump extraction
// ---------------------------------------------------------------------------

/// Extract sched_ext_dump lines from COM1 kernel console (trace_pipe output).
///
/// The trace_pipe stream contains lines with `sched_ext_dump:` prefixes when
/// a SysRq-D dump is triggered. Collects all such lines into a single string.
/// Returns `None` if no dump lines are present.
fn extract_sched_ext_dump(output: &str) -> Option<String> {
    let lines: Vec<&str> = output
        .lines()
        .filter(|l| l.contains("sched_ext_dump"))
        .collect();
    if lines.is_empty() {
        return None;
    }
    Some(lines.join("\n"))
}

/// Extract kernel version from console output (COM1/stderr).
///
/// Looks for "Linux version X.Y.Z..." in boot messages.
fn extract_kernel_version(console: &str) -> Option<String> {
    for line in console.lines() {
        if let Some(rest) = line.split("Linux version ").nth(1) {
            return Some(rest.split_whitespace().next().unwrap_or("").to_string());
        }
    }
    None
}

/// Extract the panic message from guest COM2 output.
///
/// Looks for a line containing "PANIC:" (written by the guest panic hook
/// in `rust_init.rs`). Returns the trimmed text after the "PANIC:" prefix,
/// or `None` if no panic line is present.
fn extract_panic_message(output: &str) -> Option<&str> {
    output.lines().find(|l| l.contains("PANIC:")).map(|l| {
        l.trim()
            .strip_prefix("PANIC:")
            .map(|s| s.trim_start())
            .unwrap_or(l.trim())
    })
}

// ---------------------------------------------------------------------------
// Init sentinels (written to COM2 by the Rust init and guest dispatch)
// ---------------------------------------------------------------------------

/// Written to COM2 by Rust init after filesystem mounts complete.
const SENTINEL_INIT_STARTED: &str = "KTSTR_INIT_STARTED";

/// Written to COM2 by guest dispatch immediately before the test
/// function is called.
const SENTINEL_PAYLOAD_STARTING: &str = "KTSTR_PAYLOAD_STARTING";

/// Classify the failure stage based on which sentinels appear in COM2 output.
fn classify_init_stage(output: &str) -> &'static str {
    if output.contains(SENTINEL_PAYLOAD_STARTING) {
        "payload started but produced no test result"
    } else if output.contains(SENTINEL_INIT_STARTED) {
        "init started but payload never ran (cgroup/scheduler setup failed)"
    } else {
        "init script never started (kernel or mount failure)"
    }
}

/// Format diagnostic info from COM1 kernel console output, VM exit code,
/// and init stage classification.
///
/// Returns an empty string when there is nothing useful to show.
/// Otherwise returns a section starting with a blank line, containing the
/// init stage, exit code, and the last few lines of kernel console output.
fn format_console_diagnostics(console: &str, exit_code: i32, init_stage: &str) -> String {
    const TAIL_LINES: usize = 20;
    let trimmed = console.trim();
    if trimmed.is_empty() && exit_code == 0 {
        return String::new();
    }
    let mut parts = Vec::with_capacity(3);
    parts.push(format!("stage: {init_stage}"));
    let exit_label = if exit_code < 0 {
        // Negative exit codes are typically negated errno values.
        crate::errno_name(-exit_code)
            .map(|name| format!("exit_code={exit_code} ({name})"))
            .unwrap_or_else(|| format!("exit_code={exit_code}"))
    } else {
        format!("exit_code={exit_code}")
    };
    parts.push(exit_label);
    if !trimmed.is_empty() {
        let lines: Vec<&str> = trimmed.lines().collect();
        // Show all lines when a crash is detected (PANIC: in output),
        // otherwise show only the last TAIL_LINES.
        let has_crash = lines.iter().any(|l| l.contains("PANIC:"));
        let limit = if has_crash { lines.len() } else { TAIL_LINES };
        let start = lines.len().saturating_sub(limit);
        let tail = &lines[start..];
        let truncated = !console.ends_with('\n');
        parts.push(format!(
            "console ({} lines{}):\n{}{}",
            tail.len(),
            if truncated { ", truncated" } else { "" },
            tail.join("\n"),
            if truncated { " [truncated]" } else { "" },
        ));
    }
    format!("\n\n--- diagnostics ---\n{}", parts.join("\n"))
}

// ---------------------------------------------------------------------------
// KVM validation
// ---------------------------------------------------------------------------

/// Verify that `/dev/kvm` is accessible for read+write.
fn ensure_kvm() -> Result<()> {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .context(
            "/dev/kvm not accessible — KVM is required for ktstr_test. \
             Check that KVM is enabled and your user is in the kvm group.",
        )?;
    Ok(())
}

/// Format a label for the scheduler spec, for use in test output.
fn scheduler_label(spec: &SchedulerSpec) -> String {
    match spec {
        SchedulerSpec::None => String::new(),
        SchedulerSpec::Name(n) => format!(" [sched={n}]"),
        SchedulerSpec::Path(p) => format!(" [sched={p}]"),
        SchedulerSpec::KernelBuiltin { .. } => " [sched=kernel]".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Scheduler resolution
// ---------------------------------------------------------------------------

/// Resolve a scheduler binary from a `SchedulerSpec`.
///
/// Returns `Ok(None)` for `SchedulerSpec::None` (EEVDF).
/// For `Name`, searches: `KTSTR_SCHEDULER` env, sibling of current_exe,
/// `target/debug/`, `target/release/`.
/// For `Path`, validates the file exists.
pub fn resolve_scheduler(spec: &SchedulerSpec) -> Result<Option<PathBuf>> {
    match spec {
        SchedulerSpec::None | SchedulerSpec::KernelBuiltin { .. } => Ok(None),
        SchedulerSpec::Path(p) => {
            let path = PathBuf::from(p);
            anyhow::ensure!(path.exists(), "scheduler not found: {p}");
            Ok(Some(path))
        }
        SchedulerSpec::Name(name) => {
            // 1. KTSTR_SCHEDULER env var
            if let Ok(p) = std::env::var("KTSTR_SCHEDULER") {
                let path = PathBuf::from(&p);
                if path.exists() {
                    return Ok(Some(path));
                }
            }

            // 2. Sibling of current executable (or parent of deps/)
            if let Ok(exe) = crate::resolve_current_exe()
                && let Some(dir) = exe.parent()
            {
                let candidate = dir.join(name);
                if candidate.exists() {
                    return Ok(Some(candidate));
                }
                // Integration tests and nextest place test binaries in
                // target/{debug,release}/deps/. The scheduler binary is
                // one level up in target/{debug,release}/.
                if dir.file_name().is_some_and(|d| d == "deps")
                    && let Some(parent) = dir.parent()
                {
                    let candidate = parent.join(name);
                    if candidate.exists() {
                        return Ok(Some(candidate));
                    }
                }
            }

            // 3. target/debug/
            let candidate = PathBuf::from("target/debug").join(name);
            if candidate.exists() {
                return Ok(Some(candidate));
            }

            // 4. target/release/
            let candidate = PathBuf::from("target/release").join(name);
            if candidate.exists() {
                return Ok(Some(candidate));
            }

            anyhow::bail!(
                "scheduler '{name}' not found. Set KTSTR_SCHEDULER or \
                 place it next to the test binary or in target/{{debug,release}}/"
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel resolution
// ---------------------------------------------------------------------------

/// Find a kernel image for running tests.
///
/// Checks `KTSTR_TEST_KERNEL` env var first (direct image path),
/// then delegates to [`crate::find_kernel()`] for cache and
/// filesystem discovery. Bails with actionable hints on failure.
pub fn resolve_test_kernel() -> Result<PathBuf> {
    // Check environment variable first.
    if let Ok(path) = std::env::var("KTSTR_TEST_KERNEL") {
        let p = PathBuf::from(&path);
        anyhow::ensure!(p.exists(), "KTSTR_TEST_KERNEL not found: {path}");
        return Ok(p);
    }

    // Standard locations.
    if let Some(p) = crate::find_kernel()? {
        return Ok(p);
    }

    anyhow::bail!(
        "no kernel found\n  \
         hint: run `cargo ktstr kernel build` to download and build the latest stable kernel\n  \
         hint: or set KTSTR_KERNEL=/path/to/linux\n  \
         hint: or set KTSTR_TEST_KERNEL=/path/to/bzImage"
    )
}

// ---------------------------------------------------------------------------
// Argument parsing helper
// ---------------------------------------------------------------------------

/// Extract the test function name from `--ktstr-test-fn=NAME` or
/// `--ktstr-test-fn NAME` in the argument list.
fn extract_test_fn_arg(args: &[String]) -> Option<&str> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if let Some(val) = a.strip_prefix("--ktstr-test-fn=") {
            return Some(val);
        }
        if a == "--ktstr-test-fn" {
            return iter.next().map(|s| s.as_str());
        }
    }
    None
}

/// Extract `--ktstr-probe-stack=func1,func2,...` from the argument list.
fn extract_probe_stack_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-probe-stack=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--ktstr-topo=NsNcNt` from the argument list.
fn extract_topo_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-topo=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--ktstr-flags=borrow,rebal` from the argument list.
fn extract_flags_arg(args: &[String]) -> Option<Vec<String>> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-flags=")
            && !val.is_empty()
        {
            return Some(val.split(',').map(|s| s.to_string()).collect());
        }
    }
    None
}

/// Extract `--ktstr-work-type=NAME` from the argument list.
fn extract_work_type_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-work-type=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Derive the CgroupManager root path for guest-side dispatch.
///
/// Reads `/sched_args` to find `--cell-parent-cgroup <path>`. When
/// found, constructs `/sys/fs/cgroup{path}`. Falls back to
/// `/sys/fs/cgroup/ktstr` when the arg is absent.
fn resolve_cgroup_root(args: &[String]) -> String {
    // Check guest args for --cell-parent-cgroup (passed via sched_args
    // which are written to /sched_args in the initramfs).
    let sched_args = std::fs::read_to_string("/sched_args").unwrap_or_default();
    let parts: Vec<&str> = sched_args.split_whitespace().collect();
    for i in 0..parts.len() {
        if parts[i] == "--cell-parent-cgroup"
            && let Some(&path) = parts.get(i + 1)
        {
            return format!("/sys/fs/cgroup{path}");
        }
    }
    // Also check the process args in case --cell-parent-cgroup was
    // passed directly (e.g., via extra_sched_args on the test entry).
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == "--cell-parent-cgroup"
            && let Some(path) = iter.next()
        {
            return format!("/sys/fs/cgroup{path}");
        }
    }
    "/sys/fs/cgroup/ktstr".to_string()
}

/// Resolve the sidecar output directory.
///
/// Uses `KTSTR_SIDECAR_DIR` if set, otherwise defaults to
/// `target/ktstr/{branch}-{hash}/`.
fn sidecar_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KTSTR_SIDECAR_DIR")
        && !d.is_empty()
    {
        return PathBuf::from(d);
    }
    PathBuf::from(format!(
        "target/ktstr/{}-{}",
        crate::GIT_BRANCH,
        crate::GIT_HASH,
    ))
}

/// Write a sidecar JSON file for post-run analysis.
///
/// Output goes to `KTSTR_SIDECAR_DIR` if set, otherwise to
/// `target/ktstr/{branch}-{hash}/`.
fn write_sidecar(
    entry: &KtstrTestEntry,
    vm_result: &vmm::VmResult,
    stimulus_events: &[StimulusEvent],
    verify_result: &AssertResult,
    work_type: &str,
) {
    let dir = sidecar_dir();
    let topo = format!(
        "{}s{}c{}t",
        entry.topology.sockets, entry.topology.cores_per_socket, entry.topology.threads_per_core,
    );
    let sched_name = match &entry.scheduler.binary {
        SchedulerSpec::None => "eevdf",
        SchedulerSpec::Name(n) => n,
        SchedulerSpec::Path(p) => p,
        SchedulerSpec::KernelBuiltin { .. } => "kernel",
    };
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: topo,
        scheduler: sched_name.to_string(),
        passed: verify_result.passed,
        stats: verify_result.stats.clone(),
        monitor: vm_result.monitor.as_ref().map(|m| m.summary.clone()),
        stimulus_events: stimulus_events.to_vec(),
        work_type: work_type.to_string(),
        verifier_stats: vm_result.verifier_stats.clone(),
        kvm_stats: vm_result.kvm_stats.clone(),
    };
    let path = dir.join(format!("{}.ktstr.json", entry.name));
    if let Ok(json) = serde_json::to_string_pretty(&sidecar) {
        let _ = std::fs::create_dir_all(&dir);
        if let Err(e) = std::fs::write(&path, json) {
            eprintln!("ktstr_test: write sidecar {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::shm_ring::parse_shm_params_from_str;

    /// Serializes tests that mutate env vars.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
        assert_eq!(entry.topology.sockets, 1);
        assert_eq!(entry.topology.cores_per_socket, 2);
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
        let json = r#"{"passed":true,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
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
        let json = r#"{"passed":false,"details":["stuck 3000ms"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
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
        let json = r#"{"passed":false,"details":["err1","err2"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
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
            }],
            func_names: vec![(0, "schedule".to_string())],
            bpf_source_locs: Default::default(),
            diagnostics: None,
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
                },
                ProbeEvent {
                    func_idx: 1,
                    task_ptr: 1,
                    ts: 200,
                    args: [0; 6],
                    fields: vec![("rq:rq.cpu".to_string(), 3)],
                    kstack: vec![],
                    str_val: None,
                },
            ],
            func_names: vec![
                (0, "schedule".to_string()),
                (1, "pick_task_scx".to_string()),
            ],
            bpf_source_locs: Default::default(),
            diagnostics: None,
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
        let s = Scheduler::new("test_sched")
            .binary(SchedulerSpec::Name("test_bin"))
            .flags(FLAGS_A)
            .sysctls(&[("kernel.sched_cfs_bandwidth_slice_us", "1000")])
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
            verifier_stats: vec![],
            kvm_stats: None,
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
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
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

    // -- parse_topo_string tests --

    #[test]
    fn parse_topo_valid() {
        assert_eq!(parse_topo_string("2s4c2t"), Some((2, 4, 2)));
    }

    #[test]
    fn parse_topo_single_digits() {
        assert_eq!(parse_topo_string("1s1c1t"), Some((1, 1, 1)));
    }

    #[test]
    fn parse_topo_large() {
        assert_eq!(parse_topo_string("14s9c2t"), Some((14, 9, 2)));
    }

    #[test]
    fn parse_topo_zero_sockets() {
        assert!(parse_topo_string("0s4c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_cores() {
        assert!(parse_topo_string("2s0c2t").is_none());
    }

    #[test]
    fn parse_topo_zero_threads() {
        assert!(parse_topo_string("2s4c0t").is_none());
    }

    #[test]
    fn parse_topo_missing_suffix() {
        assert!(parse_topo_string("2s4c2").is_none());
    }

    #[test]
    fn parse_topo_empty() {
        assert!(parse_topo_string("").is_none());
    }

    #[test]
    fn parse_topo_garbage() {
        assert!(parse_topo_string("hello").is_none());
    }

    #[test]
    fn parse_topo_wrong_order() {
        assert!(parse_topo_string("2c4s2t").is_none());
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
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
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
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
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
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "Bursty".to_string(),
            verifier_stats: vec![],
            kvm_stats: None,
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.work_type, "Bursty");
    }

    #[test]
    fn write_sidecar_defaults_to_target_dir_without_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::remove_var(key) };

        let dir = sidecar_dir();
        let expected = format!("target/ktstr/{}-{}", crate::GIT_BRANCH, crate::GIT_HASH);
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
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin");

        // Clean up written file.
        let path = dir.join("__sidecar_default_dir__.ktstr.json");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);

        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
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
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin");

        let path = tmp.join("__sidecar_write_test__.ktstr.json");
        assert!(path.exists(), "sidecar file should be written");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__sidecar_write_test__");
        assert!(loaded.passed);

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn parse_topo_double_digit_threads() {
        assert_eq!(parse_topo_string("1s1c12t"), Some((1, 1, 12)));
    }

    #[test]
    fn find_test_from_distributed_slice() {
        // KTSTR_TESTS should contain at least the __unit_test_dummy__ entry.
        assert!(!KTSTR_TESTS.is_empty());
    }

    #[test]
    fn topo_override_fields() {
        let t = TopoOverride {
            sockets: 2,
            cores: 4,
            threads: 2,
            memory_mb: 8192,
        };
        assert_eq!(t.sockets, 2);
        assert_eq!(t.cores, 4);
        assert_eq!(t.threads, 2);
        assert_eq!(t.memory_mb, 8192);
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
            sockets: 1,
            cores_per_socket: 2,
            threads_per_core: 1,
        },
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("test function produced no output"),
            "EEVDF with no COM2 output should say 'test function produced no output', got: {msg}",
        );
        assert!(
            !msg.contains("scheduler crashed"),
            "EEVDF error should not say 'scheduler crashed', got: {msg}",
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler crashed"),
            "scheduler present with no output should say 'scheduler crashed', got: {msg}",
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler crashed"),
            "should say scheduler crashed, got: {msg}",
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &repro_fn).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &repro_fn).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("test function produced no output"),
            "non-parseable COM2 with EEVDF should say 'test function produced no output', got: {msg}",
        );
        assert!(
            !msg.contains("scheduler crashed"),
            "EEVDF should not say scheduler crashed, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_ext_dump_included() {
        let dump_line = "ktstr-0 [001] 0.5: sched_ext_dump: Debug dump line";
        let entry = sched_entry("__eval_dump__");
        let result = make_vm_result("", dump_line, -1, false);
        let assertions = crate::assert::Assert::NONE;
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let json = r#"{"passed":true,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_pass__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        assert!(
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro,).is_ok(),
            "passing AssertResult should return Ok",
        );
    }

    #[test]
    fn eval_verify_result_failed_includes_details() {
        let json = r#"{"passed":false,"details":["stuck 3000ms","spread 45%"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_fail_details__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
        );
        assert!(msg.contains("failed:"), "got: {msg}");
        assert!(msg.contains("stuck 3000ms"), "got: {msg}");
        assert!(msg.contains("spread 45%"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_includes_sched_log() {
        let json = r#"{"passed":false,"details":["worker 0 stuck 5000ms"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nscheduler noise line\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fail_sched_log__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
        );
        assert!(msg.contains("worker 0 stuck 5000ms"), "got: {msg}");
        assert!(msg.contains("scheduler noise"), "got: {msg}");
        assert!(msg.contains("--- scheduler log ---"), "got: {msg}");
    }

    #[test]
    fn eval_assert_failure_has_fingerprint() {
        let json = r#"{"passed":false,"details":["stuck 3000ms"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let error_line = "Error: apply_cell_config BPF program returned error -2";
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nstarting\n{error_line}\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fingerprint__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let json = r#"{"passed":false,"details":["stuck"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = eevdf_entry("__eval_no_fp__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
        );
        assert!(msg.starts_with("ktstr_test"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_has_fingerprint() {
        let pass_json = r#"{"passed":true,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
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
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
        let err =
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err();
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
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            verifier_stats: vstats,
            kvm_stats: None,
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
    fn sidecar_verifier_stats_absent_deserializes_empty() {
        let json = r#"{
            "test_name": "t",
            "topology": "1s1c1t",
            "scheduler": "eevdf",
            "passed": true,
            "stats": {"cgroups":[],"total_workers":0,"total_cpus":0,
                      "total_migrations":0,"worst_spread":0.0,
                      "worst_gap_ms":0,"worst_gap_cpu":0,
                      "total_iterations":0},
            "stimulus_events": [],
            "work_type": "CpuSpin"
        }"#;
        let loaded: SidecarResult = serde_json::from_str(json).unwrap();
        assert!(loaded.verifier_stats.is_empty());
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
        let json = r#"{"passed":false,"details":["scheduler crashed after completing step 1 of 2 (0.5s into test)"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!("{RESULT_START}\n{json}\n{RESULT_END}");
        let entry = sched_entry("__eval_sched_died_console__");
        let result = make_vm_result(&output, "kernel panic\nsched_ext: disabled", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let msg = format!(
            "{}",
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
        );
        assert!(msg.contains("--- diagnostics ---"), "got: {msg}");
        assert!(msg.contains("kernel panic"), "got: {msg}");
    }

    #[test]
    fn eval_sched_died_includes_monitor() {
        let json = r#"{"passed":false,"details":["scheduler crashed during workload (2.0s into test)"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
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
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
        );
        assert!(msg.contains("--- monitor ---"), "got: {msg}");
        assert!(msg.contains("max_imbalance"), "got: {msg}");
    }

    #[test]
    fn eval_monitor_fail_includes_sched_log() {
        let pass_json = r#"{"passed":true,"details":[],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
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
            evaluate_vm_result(&entry, &result, &assertions, &[], 1, 2, 1, &no_repro).unwrap_err()
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
}
