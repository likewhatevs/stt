//! Runtime support for `#[stt_test]` integration tests.
//!
//! Provides the registration type, distributed slice, VM launcher, and
//! guest-side profraw flush for coverage-instrumented test functions.
//!
//! See the [Writing Tests](https://sched-ext.github.io/scx/stt/writing-tests.html)
//! and [`#[stt_test]` Macro](https://sched-ext.github.io/scx/stt/writing-tests/stt-test-macro.html)
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

/// Test result sidecar written to STT_SIDECAR_DIR for `stt test` collection.
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
}

/// Scan a directory for stt sidecar JSON files. Recurses one level
/// into subdirectories to handle per-job gauntlet layouts.
pub fn collect_sidecars(dir: &std::path::Path) -> Vec<SidecarResult> {
    let mut sidecars = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return sidecars,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json")
            && path.to_str().is_some_and(|s| s.contains(".stt."))
            && let Ok(data) = std::fs::read_to_string(&path)
            && let Ok(sc) = serde_json::from_str::<SidecarResult>(&data)
        {
            sidecars.push(sc);
        }
    }
    // Recurse one level for gauntlet per-job subdirectories.
    if let Ok(dirs) = std::fs::read_dir(dir) {
        for d in dirs.flatten() {
            let sub = d.path();
            if sub.is_dir()
                && let Ok(entries) = std::fs::read_dir(&sub)
            {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("json")
                        && path.to_str().is_some_and(|s| s.contains(".stt."))
                        && let Ok(data) = std::fs::read_to_string(&path)
                        && let Ok(sc) = serde_json::from_str::<SidecarResult>(&data)
                    {
                        sidecars.push(sc);
                    }
                }
            }
        }
    }
    sidecars
}

/// Early dispatch for `#[stt_test]` test execution.
///
/// Runs before `main()` in any binary that links against stt.
///
/// - `--stt-test-fn=NAME --stt-topo=NsNcNt`: host-side dispatch —
///   boots a VM with the specified topology and runs the test inside it.
/// - `--stt-test-fn=NAME` (without `--stt-topo`): guest-side dispatch —
///   runs the test function directly (inside a VM that was already booted).
/// - Otherwise: no-op.
#[ctor::ctor]
pub fn stt_test_early_dispatch() {
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
}

/// Host-side dispatch: if both `--stt-test-fn` and `--stt-topo` are
/// present, boot a VM with the specified topology and run the test
/// inside it. Returns `Some(exit_code)` if dispatched, `None` otherwise.
fn maybe_dispatch_host_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_test_fn_arg(&args)?;
    let topo_str = extract_topo_arg(&args)?;

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("stt_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    let (sockets, cores, threads) = match parse_topo_string(&topo_str) {
        Some(t) => t,
        None => {
            eprintln!("stt_test: invalid --stt-topo format '{topo_str}' (expected NsNcNt)");
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
    match run_stt_test_with_topo_and_flags(entry, &topo, &active_flags) {
        Ok(_) => Some(0),
        Err(e) => {
            eprintln!("stt_test: {e:#}");
            Some(1)
        }
    }
}

/// SHM ring message type for profraw data.
pub const MSG_TYPE_PROFRAW: u32 = 0x50524157; // "PRAW"

/// SHM size for stt_test VMs: 4 MB (profraw can be 1-2 MB).
const STT_TEST_SHM_SIZE: u64 = 4 * 1024 * 1024;

/// How to specify the scheduler binary for an `#[stt_test]`.
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
/// its binary, flag declarations, sysctls, kernel args, and monitor
/// thresholds.
pub struct Scheduler {
    pub name: &'static str,
    pub binary: SchedulerSpec,
    pub flags: &'static [&'static FlagDecl],
    pub sysctls: &'static [(&'static str, &'static str)],
    pub kargs: &'static [&'static str],
    pub assert: crate::assert::Assert,
}

impl Scheduler {
    pub const EEVDF: Scheduler = Scheduler {
        name: "eevdf",
        binary: SchedulerSpec::None,
        flags: &[],
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NONE,
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

/// Registration entry for an `#[stt_test]`-annotated function.
pub struct SttTestEntry {
    pub name: &'static str,
    pub func: fn(&Ctx) -> Result<AssertResult>,
    pub sockets: u32,
    pub cores: u32,
    pub threads: u32,
    pub memory_mb: u32,
    pub scheduler: &'static Scheduler,
    pub auto_repro: bool,
    pub replicas: u32,
    pub assert: crate::assert::Assert,
    pub extra_sched_args: &'static [&'static str],
    /// scx_watchdog_timeout in the guest kernel (seconds).
    pub watchdog_timeout_s: u64,
    /// Host-side BPF map write to perform during VM execution.
    pub bpf_map_write: Option<&'static BpfMapWrite>,
    /// Flags that must be present in every flag profile for this test.
    pub required_flags: &'static [&'static str],
    /// Flags that must not be present in any flag profile for this test.
    pub excluded_flags: &'static [&'static str],
    /// Minimum number of sockets for gauntlet topology filtering.
    pub min_sockets: u32,
    /// Minimum number of LLCs for gauntlet topology filtering.
    pub min_llcs: u32,
    /// Whether the test requires SMT (threads > 1) topologies.
    pub requires_smt: bool,
    /// Minimum total CPU count for gauntlet topology filtering.
    pub min_cpus: u32,
    /// Pin vCPU threads to host cores matching the virtual topology's LLC
    /// structure, use 2MB hugepages for guest memory, and validate that the
    /// host has enough CPUs and LLCs to satisfy the request without
    /// oversubscription.
    pub performance_mode: bool,
    /// LLC exclusivity mode. Implies performance_mode. Each virtual socket
    /// reserves an entire physical LLC group.
    pub super_perf_mode: bool,
    /// Workload duration in seconds.
    pub duration_s: u64,
    /// Workers per cgroup.
    pub workers_per_cgroup: u32,
    /// When true, the test expects run_stt_test to return Err.
    /// Disables auto_repro (no point probing a deliberately failing test).
    pub expect_err: bool,
}

/// Placeholder function for `SttTestEntry::DEFAULT`. Panics if called.
fn default_test_func(_ctx: &Ctx) -> Result<AssertResult> {
    anyhow::bail!("SttTestEntry::DEFAULT func called — override func before use")
}

impl SttTestEntry {
    /// Sensible defaults for all fields. Override `name`, `func`, and
    /// `scheduler` (at minimum) via struct update syntax:
    ///
    /// ```ignore
    /// static ENTRY: SttTestEntry = SttTestEntry {
    ///     name: "my_test",
    ///     func: my_test_fn,
    ///     scheduler: &MITOSIS,
    ///     ..SttTestEntry::DEFAULT
    /// };
    /// ```
    pub const DEFAULT: SttTestEntry = SttTestEntry {
        name: "",
        func: default_test_func,
        sockets: 1,
        cores: 2,
        threads: 1,
        memory_mb: 2048,
        scheduler: &Scheduler::EEVDF,
        auto_repro: true,
        replicas: 1,
        assert: crate::assert::Assert::NONE,
        extra_sched_args: &[],
        watchdog_timeout_s: 4,
        bpf_map_write: None,
        required_flags: &[],
        excluded_flags: &[],
        min_sockets: 1,
        min_llcs: 1,
        requires_smt: false,
        min_cpus: 1,
        performance_mode: false,
        super_perf_mode: false,
        duration_s: 2,
        workers_per_cgroup: 2,
        expect_err: false,
    };
}

/// Distributed slice collecting all `#[stt_test]` entries via linkme.
#[distributed_slice]
pub static STT_TESTS: [SttTestEntry];

/// Look up a registered test function by name.
pub fn find_test(name: &str) -> Option<&'static SttTestEntry> {
    STT_TESTS.iter().find(|e| e.name == name)
}

/// Optional topology override for `run_stt_test`.
pub struct TopoOverride {
    pub sockets: u32,
    pub cores: u32,
    pub threads: u32,
    pub memory_mb: u32,
}

/// Parse a topology string in "NsNcNt" format (e.g. "2s4c2t").
/// Returns None if the string doesn't match the expected format.
pub fn parse_topo_string(s: &str) -> Option<(u32, u32, u32)> {
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

/// Check whether a gauntlet preset is compatible with a test entry
/// on this host. `host_llc_count` is the number of LLC groups on
/// the host -- performance_mode tests need one LLC per virtual socket.
pub fn preset_matches(
    preset: &crate::vm::TopoPreset,
    entry: &SttTestEntry,
    host_llc_count: usize,
) -> bool {
    let t = &preset.topology;
    t.sockets >= entry.min_sockets
        && t.num_llcs() >= entry.min_llcs
        && (!entry.requires_smt || t.threads_per_core >= 2)
        && t.total_cpus() >= entry.min_cpus
        // performance_mode maps each virtual socket to a host LLC.
        // +1 for the service CPU that must land outside pinned LLCs.
        && (!entry.performance_mode || (t.sockets as usize) < host_llc_count)
}

/// Number of LLC groups on this host. Returns 0 on error.
pub fn host_llc_count() -> usize {
    crate::vmm::host_topology::HostTopology::from_sysfs()
        .map(|h| h.llc_groups.len())
        .unwrap_or(0)
}

/// Check whether the host has enough CPUs and LLC groups to satisfy
/// a gauntlet preset's topology without oversubscription.
pub fn host_preset_compatible(
    preset: &crate::vm::TopoPreset,
    host: &crate::vmm::host_topology::HostTopology,
) -> bool {
    let t = &preset.topology;
    let total_vcpus = t.total_cpus();
    let vcpus_per_socket = t.cores_per_socket * t.threads_per_core;
    total_vcpus as usize <= host.total_cpus()
        && (t.sockets as usize) <= host.llc_groups.len()
        && (vcpus_per_socket as usize) <= host.max_cores_per_llc()
}

/// Default seconds to wait for LLC/CPU resource locks before skipping.
/// 55s leaves 5s margin before nextest's 60s default slow-timeout.
const RESOURCE_WAIT_DEADLINE_SECS: u64 = 55;

/// Deadline for resource acquisition polling. Uses
/// `STT_RESOURCE_WAIT_SECS` env var if set, otherwise
/// `RESOURCE_WAIT_DEADLINE_SECS` (55s).
pub(crate) fn resource_deadline() -> std::time::Duration {
    if let Ok(val) = std::env::var("STT_RESOURCE_WAIT_SECS")
        && let Ok(secs) = val.parse::<u64>()
    {
        return std::time::Duration::from_secs(secs);
    }
    std::time::Duration::from_secs(RESOURCE_WAIT_DEADLINE_SECS)
}

/// Host-side entry point: build a VM, boot it with `--stt-test-fn=NAME`,
/// extract profraw from SHM, and return the test result.
///
/// Validates KVM access and auto-discovers a kernel image via
/// `resolve_kernel()` when `STT_TEST_KERNEL` is not set.
pub fn run_stt_test(entry: &SttTestEntry) -> Result<AssertResult> {
    run_stt_test_inner(entry, None, &[])
}

/// Like `run_stt_test` but with an explicit topology override.
pub fn run_stt_test_with_topo(entry: &SttTestEntry, topo: &TopoOverride) -> Result<AssertResult> {
    run_stt_test_inner(entry, Some(topo), &[])
}

/// Like `run_stt_test_with_topo` but with active flags that map to
/// scheduler CLI args via `Scheduler::flag_args()`.
pub fn run_stt_test_with_topo_and_flags(
    entry: &SttTestEntry,
    topo: &TopoOverride,
    active_flags: &[String],
) -> Result<AssertResult> {
    run_stt_test_inner(entry, Some(topo), active_flags)
}

/// Run a test result through expect_err logic and return a
/// `Completion` or `Failed`.
fn result_to_completion(
    result: Result<AssertResult>,
    expect_err: bool,
) -> std::result::Result<libtest_mimic::Completion, libtest_mimic::Failed> {
    match result {
        Ok(_) if expect_err => Err("expected error but test passed".into()),
        Ok(_) => Ok(libtest_mimic::Completion::Completed),
        Err(e)
            if e.downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .is_some() =>
        {
            let reason = e
                .downcast_ref::<crate::vmm::host_topology::ResourceContention>()
                .unwrap()
                .reason
                .clone();
            Ok(libtest_mimic::Completion::ignored_with(reason))
        }
        Err(_) if expect_err => Ok(libtest_mimic::Completion::Completed),
        Err(e) => Err(format!("{e:#}").into()),
    }
}

/// Build `libtest_mimic::Trial` entries for all registered `#[stt_test]`
/// entries. Includes base trials and gauntlet variants.
///
/// Uses `Trial::ignorable_test()` so performance_mode tests that cannot
/// acquire resources within the deadline appear as "ignored" instead of
/// failing.
pub fn build_stt_trials() -> Vec<libtest_mimic::Trial> {
    let presets = crate::vm::gauntlet_presets();
    let host_llcs = host_llc_count();
    let host_topo = crate::vmm::host_topology::HostTopology::from_sysfs().ok();
    let mut trials = Vec::new();

    for entry in STT_TESTS.iter() {
        let profiles = entry
            .scheduler
            .generate_profiles(entry.required_flags, entry.excluded_flags);

        // Skip base trial entirely when the host cannot possibly
        // satisfy the entry's topology. performance_mode tests need
        // one LLC per virtual socket plus a service CPU.
        let base_runnable = if entry.performance_mode {
            host_topo.as_ref().is_some_and(|h| {
                (entry.sockets as usize) < h.llc_groups.len()
                    && ((entry.sockets * entry.cores * entry.threads) as usize) < h.total_cpus()
                    && (entry.cores * entry.threads) as usize <= h.max_cores_per_llc()
            })
        } else {
            true
        };

        if !base_runnable {
            continue;
        }

        // Base trial: runs with the entry's own topology.
        // Resource contention (performance_mode) is handled by
        // result_to_completion via ResourceContention downcast.
        let expect_err = entry.expect_err;
        trials.push(
            libtest_mimic::Trial::ignorable_test(entry.name.to_string(), move || {
                let result = run_stt_test_inner(entry, None, &[]);
                result_to_completion(result, expect_err)
            })
            .with_ignored_flag(entry.name.starts_with("demo_")),
        );

        // Gauntlet variants: topology x flags, always ignored by default.
        for preset in &presets {
            if !preset_matches(preset, entry, host_llcs) {
                continue;
            }
            if let Some(ref host) = host_topo
                && !host_preset_compatible(preset, host)
            {
                continue;
            }
            let t = &preset.topology;
            let topo_str = format!(
                "{}s{}c{}t",
                t.sockets, t.cores_per_socket, t.threads_per_core,
            );
            let cpus = t.total_cpus();
            let memory_mb = (cpus * 64).max(256).max(entry.memory_mb);
            let preset_name = preset.name;

            for profile in &profiles {
                let pname = profile.name();
                let name = format!("gauntlet/{}/{}/{}", entry.name, preset_name, pname);
                let topo_str = topo_str.clone();
                let flags: Vec<String> = profile.flags.iter().map(|s| s.to_string()).collect();

                let expect_err = entry.expect_err;
                trials.push(
                    libtest_mimic::Trial::ignorable_test(name, move || {
                        let (sockets, cores, threads) =
                            parse_topo_string(&topo_str).expect("invalid topo string");
                        let topo = TopoOverride {
                            sockets,
                            cores,
                            threads,
                            memory_mb,
                        };
                        let result = run_stt_test_inner(entry, Some(&topo), &flags);
                        result_to_completion(result, expect_err)
                    })
                    .with_ignored_flag(true),
                );
            }
        }
    }

    trials
}

/// Run sidecar collection and stats summary. Called from harness
/// main() after `libtest_mimic::run()`.
pub fn collect_and_print_sidecar_stats() {
    if let Ok(dir) = std::env::var("STT_SIDECAR_DIR") {
        let sidecars = collect_sidecars(std::path::Path::new(&dir));
        let rows: Vec<_> = sidecars.iter().map(crate::stats::sidecar_to_row).collect();
        if !rows.is_empty() {
            eprintln!("{}", crate::stats::analyze_rows(&rows));
        }
    }
}

fn run_stt_test_inner(
    entry: &SttTestEntry,
    topo: Option<&TopoOverride>,
    active_flags: &[String],
) -> Result<AssertResult> {
    ensure_kvm()?;
    let kernel = resolve_kernel()?;
    let scheduler = resolve_scheduler(&entry.scheduler.binary)?;
    let stt_bin = crate::resolve_current_exe()?;

    let guest_args = vec![
        "run".to_string(),
        "--stt-test-fn".to_string(),
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
    // Propagate RUST_BACKTRACE to the guest so guest-side code
    // can gate verbose output.
    if let Ok(bt) = std::env::var("RUST_BACKTRACE") {
        cmdline_parts.push(format!("RUST_BACKTRACE={bt}"));
    }
    let cmdline_extra = cmdline_parts.join(" ");

    let (sockets, cores, threads, memory_mb) = match topo {
        Some(t) => (t.sockets, t.cores, t.threads, t.memory_mb),
        None => {
            let cpus = entry.sockets * entry.cores * entry.threads;
            let mem = (cpus * 64).max(256).max(entry.memory_mb);
            (entry.sockets, entry.cores, entry.threads, mem)
        }
    };

    // Enforce memory floor based on initramfs size estimate.
    let initrd_floor = vmm::estimate_min_memory_mb(&stt_bin, scheduler.as_deref());
    let memory_mb = memory_mb.max(initrd_floor);

    let mut builder = vmm::SttVm::builder()
        .kernel(&kernel)
        .init_binary(&stt_bin)
        .topology(sockets, cores, threads)
        .memory_mb(memory_mb)
        .cmdline(&cmdline_extra)
        .shm_size(STT_TEST_SHM_SIZE)
        .run_args(&guest_args)
        .timeout(Duration::from_secs(60))
        .performance_mode(entry.performance_mode || entry.super_perf_mode)
        .super_perf_mode(entry.super_perf_mode);

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

    // Merge scheduler args: extra_sched_args from the entry + args derived
    // from active flags via Scheduler::flag_args().
    let mut sched_args: Vec<String> = entry
        .extra_sched_args
        .iter()
        .map(|s| s.to_string())
        .collect();
    for flag_name in active_flags {
        if let Some(args) = entry.scheduler.flag_args(flag_name) {
            sched_args.extend(args.iter().map(|s| s.to_string()));
        }
    }
    if !sched_args.is_empty() {
        builder = builder.sched_args(&sched_args);
    }

    builder = builder.watchdog_timeout_s(entry.watchdog_timeout_s);

    if let Some(bpf_write) = entry.bpf_map_write {
        builder =
            builder.bpf_map_write(bpf_write.map_name_suffix, bpf_write.offset, bpf_write.value);
    }

    let vm = builder.build().context("build stt_test VM")?;

    let result = vm.run().context("run stt_test VM")?;

    // Extract profraw from SHM ring buffer and collect stimulus events.
    let mut stimulus_events = Vec::new();
    if let Some(ref shm) = result.shm_data {
        for entry in &shm.entries {
            if entry.msg_type == MSG_TYPE_PROFRAW
                && entry.crc_ok
                && !entry.payload.is_empty()
                && let Err(e) = write_profraw(&entry.payload)
            {
                eprintln!("stt_test: write guest profraw: {e}");
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
        attempt_auto_repro(entry, &kernel, scheduler.as_deref(), &stt_bin, output, topo)
    };

    evaluate_vm_result(
        entry,
        &result,
        scheduler.as_deref(),
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
/// `run_stt_test_inner` so that error message formatting can be tested
/// without booting a VM. The `repro_fn` callback handles auto-repro
/// (which requires a second VM boot) when provided.
#[allow(clippy::too_many_arguments)]
fn evaluate_vm_result(
    entry: &SttTestEntry,
    result: &vmm::VmResult,
    _scheduler: Option<&Path>,
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
    let dump_section = extract_sched_ext_dump(output)
        .map(|d| format!("\n\n--- sched_ext dump ---\n{d}"))
        .unwrap_or_default();
    let sched_log_section = parse_sched_output(output)
        .map(|s| {
            let collapsed = crate::verifier::collapse_cycles(s);
            format!("\n\n--- scheduler log ---\n{collapsed}")
        })
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

    if let Ok(verify_result) = parse_assert_result(output) {
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
            let msg = format!(
                "stt_test '{}'{} failed:\n  {}{}{}{}{}",
                entry.name,
                sched_label,
                details,
                stats_section,
                timeline_section,
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
                let msg = format!(
                    "stt_test '{}'{} passed scenario but monitor failed:\n  {}{}{}",
                    entry.name, sched_label, details, timeline_section, dump_section,
                );
                anyhow::bail!("{msg}");
            }
        }

        return Ok(verify_result);
    }

    // No parseable result — the payload never wrote a AssertResult to COM2.
    // When a scheduler is running this typically means the scheduler died;
    // without a scheduler (EEVDF) it means the payload itself failed.
    // Attempt auto-repro if enabled and a scheduler was running.
    let repro_section = if entry.scheduler.binary.has_active_scheduling()
        && (output.contains("SCHEDULER_DIED")
            || output.contains("scheduler died")
            || (result.stderr.contains("sched_ext:") && result.stderr.contains("disabled")))
    {
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

    // Build monitor section for error paths where COM2 had no parseable result.
    let monitor_section = if entry.scheduler.binary.has_active_scheduling()
        && let Some(ref monitor) = result.monitor
    {
        let s = &monitor.summary;
        let eval_report = trim_settle_samples(monitor);
        let thresholds = merged_assert.monitor_thresholds();
        let verdict = thresholds.evaluate(&eval_report);
        let verdict_line = if verdict.passed {
            verdict.summary.clone()
        } else {
            format!("{}: {}", verdict.summary, verdict.details.join("; "))
        };
        format!(
            "\n\n--- monitor ---\nsamples={} max_imbalance={:.2} max_dsq_depth={} stall={}\nverdict: {}",
            s.total_samples,
            s.max_imbalance_ratio,
            s.max_local_dsq_depth,
            s.stall_detected,
            verdict_line,
        )
    } else {
        String::new()
    };

    if result.timed_out {
        let msg = format!(
            "stt_test '{}'{} timed out (no result in COM2){}{}{}{}{}{}",
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

    let reason = if entry.scheduler.binary.has_active_scheduling() {
        "scheduler died (no test result in COM2)"
    } else {
        "payload produced no output (no test result in COM2)"
    };
    let msg = format!(
        "stt_test '{}'{} {}{}{}{}{}{}{}",
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
    let summary = crate::monitor::MonitorSummary::from_samples(&trimmed);
    crate::monitor::MonitorReport {
        samples: trimmed,
        summary,
        preemption_threshold_ns: report.preemption_threshold_ns,
    }
}

/// Attempt auto-repro: extract stack functions from the scheduler output,
/// boot a second VM with BPF probes attached, and return formatted probe
/// data. Returns `None` if repro cannot be attempted or yields no data.
fn attempt_auto_repro(
    entry: &SttTestEntry,
    kernel: &Path,
    scheduler: Option<&Path>,
    stt_bin: &Path,
    first_vm_output: &str,
    topo: Option<&TopoOverride>,
) -> Option<String> {
    use crate::probe::stack::extract_stack_functions_all;

    // Extract scheduler log from COM2 output.
    eprintln!(
        "stt_test: auto-repro: COM2 length={} has_sched_start={} has_sched_end={}",
        first_vm_output.len(),
        first_vm_output.contains("===SCHED_OUTPUT_START==="),
        first_vm_output.contains("===SCHED_OUTPUT_END==="),
    );
    let sched_output = parse_sched_output(first_vm_output)?;

    // Extract function names from the scheduler log.
    let stack_funcs = extract_stack_functions_all(sched_output);
    if stack_funcs.is_empty() {
        eprintln!("stt_test: auto-repro: no functions extracted from scheduler output");
        return None;
    }

    let func_names: Vec<String> = stack_funcs.iter().map(|f| f.raw_name.clone()).collect();
    let probe_arg = format!("--stt-probe-stack={}", func_names.join(","));

    eprintln!(
        "stt_test: auto-repro: probing {} functions in second VM",
        func_names.len()
    );

    // Build guest args for the repro VM.
    let guest_args = vec![
        "run".to_string(),
        "--stt-test-fn".to_string(),
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
    let cmdline_extra = cmdline_parts.join(" ");

    let (sockets, cores, threads, memory_mb) = match topo {
        Some(t) => (t.sockets, t.cores, t.threads, t.memory_mb),
        None => {
            let cpus = entry.sockets * entry.cores * entry.threads;
            let mem = (cpus * 64).max(256).max(entry.memory_mb);
            (entry.sockets, entry.cores, entry.threads, mem)
        }
    };

    let mut builder = vmm::SttVm::builder()
        .kernel(kernel)
        .init_binary(stt_bin)
        .topology(sockets, cores, threads)
        .memory_mb(memory_mb)
        .cmdline(&cmdline_extra)
        .shm_size(STT_TEST_SHM_SIZE)
        .run_args(&guest_args)
        .timeout(Duration::from_secs(60));

    if let Some(sched_path) = scheduler {
        builder = builder.scheduler_binary(sched_path);
    }

    if !entry.extra_sched_args.is_empty() {
        let args: Vec<String> = entry
            .extra_sched_args
            .iter()
            .map(|s| s.to_string())
            .collect();
        builder = builder.sched_args(&args);
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
            eprintln!("stt_test: auto-repro: failed to build VM: {e:#}");
            return None;
        }
    };

    let repro_result = match vm.run() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("stt_test: auto-repro: VM run failed: {e:#}");
            return None;
        }
    };

    // Forward guest stderr (COM1) and COM2 probe lines when verbose.
    if verbose() {
        eprintln!(
            "stt_test: auto-repro: COM1 stderr length={} COM2 stdout length={}",
            repro_result.stderr.len(),
            repro_result.output.len(),
        );
        for line in repro_result.stderr.lines() {
            eprintln!("  repro-vm-com1: {line}");
        }
        let mut in_probe = false;
        for line in repro_result.output.lines() {
            if line.contains("stt_test: probe:") {
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
            eprintln!("stt_test: probe payload deserialize failed: {e}");
            return None;
        }
    };
    if payload.events.is_empty() {
        return None;
    }
    Some(crate::probe::output::format_probe_events_with_bpf_locs(
        &payload.events,
        &payload.func_names,
        kernel_dir,
        false,
        &payload.bpf_source_locs,
    ))
}

/// Setup function for nextest `setup-script` integration.
///
/// Validates KVM access, discovers a kernel, writes `STT_TEST_KERNEL`
/// to `env_writer`, and warms the SHM initramfs cache for each binary.
pub fn nextest_setup(binaries: &[&Path], env_writer: &mut dyn Write) -> Result<()> {
    ensure_kvm()?;
    let kernel = resolve_kernel()?;
    writeln!(env_writer, "STT_TEST_KERNEL={}", kernel.display())
        .context("write STT_TEST_KERNEL to env")?;

    for bin in binaries {
        let key = vmm::BaseKey::new(bin, None)?;
        let _ = vmm::get_or_build_base(bin, &[], &key)?;
    }

    Ok(())
}

/// Guest-side dispatch: check for `--stt-test-fn=NAME` in args, run the
/// registered function, write the result to stdout (captured by COM2),
/// and exit. Profraw flush is handled by `try_flush_profraw()` in the
/// ctor before `std::process::exit()`.
///
/// Called from `stt_test_early_dispatch()` (ctor) before `main()`.
/// Returns `Some(exit_code)` if dispatched, `None` if not an
/// stt_test invocation.
pub fn maybe_dispatch_vm_test() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let name = extract_test_fn_arg(&args)?;

    // Propagate RUST_BACKTRACE from kernel cmdline to env.
    if let Ok(cmdline) = std::fs::read_to_string("/proc/cmdline")
        && let Some(val) = cmdline
            .split_whitespace()
            .find(|s| s.starts_with("RUST_BACKTRACE="))
            .and_then(|s| s.strip_prefix("RUST_BACKTRACE="))
    {
        // SAFETY: guest-side dispatch runs single-threaded before
        // any test threads are spawned.
        unsafe { std::env::set_var("RUST_BACKTRACE", val) };
    }

    // Coverage instrumentation adds overhead that affects scheduling
    // fairness and causes larger scheduling gaps. Downgrade spread
    // violations to warnings and relax the gap threshold.
    #[cfg(coverage)]
    {
        crate::assert::set_warn_unfair(true);
        crate::assert::set_coverage_gap_ms(5000);
    }

    let entry = match find_test(name) {
        Some(e) => e,
        None => {
            eprintln!("stt_test: unknown test function '{name}'");
            return Some(1);
        }
    };

    // Parse --stt-probe-stack=func1,func2,... for auto-repro mode.
    let probe_stack = extract_probe_stack_arg(&args);

    // Parse --stt-work-type=NAME for work type override.
    let work_type_override = extract_work_type_arg(&args).and_then(|s| {
        crate::workload::WorkType::from_name(&s).or_else(|| {
            eprintln!("stt_test: unknown work type '{s}'");
            None
        })
    });

    // Set up BPF probes if --stt-probe-stack was provided.
    let probe_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let probe_handle: Option<ProbeHandle> = probe_stack.as_ref().and_then(|stack_input| {
        use crate::probe::stack::load_probe_stack;

        eprintln!("stt_test: probe: loading probe stack from --stt-probe-stack");
        let mut functions = crate::probe::stack::filter_traceable(load_probe_stack(stack_input));
        // Discover BPF scheduler functions from the running scheduler.
        // Stack-extracted BPF names have stale prog IDs from the first VM;
        // discover_bpf_symbols finds the current scheduler's programs.
        let bpf_syms = crate::probe::btf::discover_bpf_symbols();
        if !bpf_syms.is_empty() {
            eprintln!("stt_test: probe: {} BPF symbols discovered", bpf_syms.len());
            functions.extend(bpf_syms);
        }
        // Expand BPF functions to kernel-side callers for bridge kprobes,
        // keeping BPF functions for fentry attachment.
        let functions = crate::probe::stack::expand_bpf_to_kernel_callers(functions);
        if functions.is_empty() {
            eprintln!("stt_test: no traceable functions from --stt-probe-stack");
            return None;
        }

        eprintln!(
            "stt_test: probe: {} functions loaded, spawning probe thread",
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

        let stop = probe_stop.clone();
        let funcs = functions.clone();
        let handle = std::thread::spawn(move || {
            use crate::probe::process::run_probe_skeleton;
            run_probe_skeleton(&funcs, &btf_funcs, "scx_disable_workfn", &stop)
        });
        Some((handle, func_names))
    });

    // Build a minimal Ctx for the test function.
    let topo = crate::topology::TestTopology::from_system().unwrap_or_else(|e| {
        eprintln!("stt_test: topology from sysfs failed ({e}), using VM spec fallback");
        crate::topology::TestTopology::from_spec(entry.sockets, entry.cores, entry.threads)
    });
    let cgroups = crate::cgroup::CgroupManager::new("/sys/fs/cgroup/stt");
    if let Err(e) = cgroups.setup(false) {
        eprintln!("stt_test: cgroup setup failed: {e}");
    }
    let sched_pid = std::env::var("SCHED_PID")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let duration = Duration::from_secs(entry.duration_s);
    let workers_per_cgroup = entry.workers_per_cgroup as usize;
    // Three-layer merge: default_checks -> scheduler.assert -> entry.assert.
    let merged_assert = crate::assert::Assert::default_checks()
        .merge(&entry.scheduler.assert)
        .merge(&entry.assert);
    let ctx = Ctx {
        cgroups: &cgroups,
        topo: &topo,
        duration,
        workers_per_cgroup,
        sched_pid,
        settle_ms: 500,
        work_type_override,
        assert: merged_assert,
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
    std::thread::JoinHandle<Option<Vec<crate::probe::process::ProbeEvent>>>,
    Vec<(u32, String)>,
);

/// Serialized probe data sent from guest to host via COM2.
/// The host deserializes and formats with kernel_dir for source locations.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ProbePayload {
    pub events: Vec<crate::probe::process::ProbeEvent>,
    pub func_names: Vec<(u32, String)>,
    #[serde(default)]
    pub bpf_source_locs: std::collections::HashMap<String, String>,
}

/// Stop probes, join the probe thread, and print captured probe data to
/// stdout (COM2) between delimiters so the host can extract it.
fn collect_and_print_probe_data(
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<ProbeHandle>,
) {
    let Some((handle, func_names)) = handle else {
        return;
    };

    stop.store(true, std::sync::atomic::Ordering::Release);
    let events = match handle.join() {
        Ok(Some(events)) if !events.is_empty() => events,
        _ => return,
    };

    // Resolve BPF source locations inside the guest where the BPF
    // programs are loaded. The host doesn't have the prog FDs.
    let bpf_prog_ids: Vec<u32> = func_names
        .iter()
        .filter_map(|(_, name)| {
            // BPF functions have sentinel IPs — find their prog_ids
            // from discover_bpf_symbols cache.
            crate::probe::btf::discover_bpf_symbols()
                .into_iter()
                .find(|s| s.display_name == *name)
                .and_then(|s| s.bpf_prog_id)
        })
        .collect();
    let bpf_source_locs = crate::probe::btf::resolve_bpf_source_locs(&bpf_prog_ids);

    let payload = ProbePayload {
        events,
        func_names,
        bpf_source_locs,
    };
    println!("===PROBE_OUTPUT_START===");
    if let Ok(json) = serde_json::to_string(&payload) {
        println!("{json}");
    }
    println!("===PROBE_OUTPUT_END===");
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
fn try_flush_profraw() {
    let Some((shm_base, shm_size)) = parse_shm_params() else {
        return;
    };

    let exe = match std::fs::read("/proc/self/exe") {
        Ok(data) => data,
        Err(_) => return,
    };
    let slide = pie_load_bias(&exe);

    // Set profraw output path, then call __llvm_profile_initialize to
    // read it and register the atexit handler.
    // SAFETY: single-threaded guest dispatch context.
    unsafe { std::env::set_var("LLVM_PROFILE_FILE", "/tmp/stt.profraw") };
    if let Some(vaddr) = find_symbol_vaddr(&exe, "__llvm_profile_initialize")
        && vaddr != 0
    {
        let f: extern "C" fn() =
            unsafe { std::mem::transmute((vaddr as usize).wrapping_add(slide)) };
        f();
    }

    // Write profraw to the file.
    let write_file_vaddr = match find_symbol_vaddr(&exe, "__llvm_profile_write_file") {
        Some(v) if v != 0 => v,
        _ => return,
    };
    let write_file: extern "C" fn() -> i32 =
        unsafe { std::mem::transmute((write_file_vaddr as usize).wrapping_add(slide)) };
    if write_file() != 0 {
        return;
    }

    // Read the profraw file and send through SHM ring.
    let data = match std::fs::read("/tmp/stt.profraw") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    let _ = write_to_shm_ring(shm_base, shm_size, MSG_TYPE_PROFRAW, &data);
}

/// Find a symbol's file virtual address in an ELF binary's .symtab.
fn find_symbol_vaddr(data: &[u8], name: &str) -> Option<u64> {
    use object::elf;
    use object::read::elf::{FileHeader, Sym};

    let header = elf::FileHeader64::<object::Endianness>::parse(data).ok()?;
    let endian = header.endian().ok()?;
    let sections = header.sections(endian, data).ok()?;
    let symbols = sections.symbols(endian, data, elf::SHT_SYMTAB).ok()?;

    for sym in symbols.iter() {
        if sym.st_size(endian) == 0 {
            continue;
        }
        let sym_name = match sym.name(endian, symbols.strings()) {
            Ok(n) => n,
            Err(_) => continue,
        };
        if sym_name == name.as_bytes() {
            return Some(sym.st_value(endian));
        }
    }
    None
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
    use object::elf;
    use object::read::elf::FileHeader;

    let header = match elf::FileHeader64::<object::Endianness>::parse(data) {
        Ok(h) => h,
        Err(_) => return 0,
    };
    let endian = match header.endian() {
        Ok(e) => e,
        Err(_) => return 0,
    };

    if header.e_type(endian) != elf::ET_DYN {
        return 0;
    }

    let phdr_file_offset = header.e_phoff(endian) as usize;
    // SAFETY: AT_PHDR is a well-defined auxiliary vector key.
    let phdr_runtime = unsafe { libc::getauxval(libc::AT_PHDR) } as usize;
    if phdr_runtime == 0 {
        return 0;
    }
    phdr_runtime.wrapping_sub(phdr_file_offset)
}

/// Parse STT_SHM_BASE and STT_SHM_SIZE from /proc/cmdline.
fn parse_shm_params() -> Option<(u64, u64)> {
    let cmdline = std::fs::read_to_string("/proc/cmdline").ok()?;
    parse_shm_params_from_str(&cmdline)
}

/// Parse STT_SHM_BASE and STT_SHM_SIZE from a kernel command line string.
fn parse_shm_params_from_str(cmdline: &str) -> Option<(u64, u64)> {
    let base = cmdline
        .split_whitespace()
        .find(|s| s.starts_with("STT_SHM_BASE="))?
        .strip_prefix("STT_SHM_BASE=")?;
    let size = cmdline
        .split_whitespace()
        .find(|s| s.starts_with("STT_SHM_SIZE="))?
        .strip_prefix("STT_SHM_SIZE=")?;
    let base =
        u64::from_str_radix(base.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
    let size =
        u64::from_str_radix(size.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
    Some((base, size))
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

    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let aligned_base = shm_base & !(page_size - 1);
    let offset_in_page = (shm_base - aligned_base) as usize;
    let map_size = shm_size as usize + offset_in_page;

    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            map_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            std::os::unix::io::AsRawFd::as_raw_fd(&fd),
            aligned_base as libc::off_t,
        )
    };

    if ptr == libc::MAP_FAILED {
        anyhow::bail!("mmap /dev/mem failed");
    }

    let shm_buf = unsafe {
        std::slice::from_raw_parts_mut((ptr as *mut u8).add(offset_in_page), shm_size as usize)
    };

    let written = vmm::shm_ring::shm_write(shm_buf, 0, msg_type, payload);

    unsafe {
        libc::munmap(ptr, map_size);
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
    let path = target_dir.join(format!("stt-test-{}-{}.profraw", std::process::id(), id));
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
const RESULT_START: &str = "===STT_TEST_RESULT_START===";
const RESULT_END: &str = "===STT_TEST_RESULT_END===";

/// Print AssertResult as delimited JSON to stdout (captured by COM2).
fn print_assert_result(r: &AssertResult) {
    println!("{RESULT_START}");
    if let Ok(json) = serde_json::to_string(r) {
        println!("{json}");
    }
    println!("{RESULT_END}");
}

/// Parse AssertResult from guest output between delimiters.
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

/// Extract sched_ext_dump lines from guest output (trace_pipe output on COM2).
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

// ---------------------------------------------------------------------------
// Init script sentinels (written to COM2 by the init script)
// ---------------------------------------------------------------------------

/// Written to COM2 after proc/sys/devtmpfs mounts, before scheduler start.
const SENTINEL_INIT_STARTED: &str = "STT_INIT_STARTED";

/// Written to COM2 after mounts, scheduler start, and trace setup —
/// immediately before the payload binary is exec'd.
const SENTINEL_PAYLOAD_STARTING: &str = "STT_PAYLOAD_STARTING";

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
    parts.push(format!("exit_code={exit_code}"));
    if !trimmed.is_empty() {
        let lines: Vec<&str> = trimmed.lines().collect();
        let start = lines.len().saturating_sub(TAIL_LINES);
        let tail = &lines[start..];
        parts.push(format!(
            "kernel console (last {} lines):\n{}",
            tail.len(),
            tail.join("\n"),
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
            "/dev/kvm not accessible — KVM is required for stt_test. \
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
/// For `Name`, searches: `STT_SCHEDULER` env, sibling of current_exe,
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
            // 1. STT_SCHEDULER env var
            if let Ok(p) = std::env::var("STT_SCHEDULER") {
                let path = PathBuf::from(&p);
                if path.exists() {
                    return Ok(Some(path));
                }
            }

            // 2. Sibling of current executable
            if let Ok(exe) = crate::resolve_current_exe()
                && let Some(dir) = exe.parent()
            {
                let candidate = dir.join(name);
                if candidate.exists() {
                    return Ok(Some(candidate));
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
                "scheduler '{name}' not found. Set STT_SCHEDULER or \
                 place it next to the test binary or in target/{{debug,release}}/"
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel resolution
// ---------------------------------------------------------------------------

/// Find a kernel bzImage for running tests.
pub fn resolve_kernel() -> Result<PathBuf> {
    // Check environment variable first.
    if let Ok(path) = std::env::var("STT_TEST_KERNEL") {
        let p = PathBuf::from(&path);
        anyhow::ensure!(p.exists(), "STT_TEST_KERNEL not found: {path}");
        return Ok(p);
    }

    // Standard locations.
    if let Some(p) = crate::find_kernel() {
        return Ok(p);
    }

    anyhow::bail!("no kernel found. Set STT_TEST_KERNEL or build one at ../linux/")
}

// ---------------------------------------------------------------------------
// Argument parsing helper
// ---------------------------------------------------------------------------

/// Extract the test function name from `--stt-test-fn=NAME` or
/// `--stt-test-fn NAME` in the argument list.
fn extract_test_fn_arg(args: &[String]) -> Option<&str> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if let Some(val) = a.strip_prefix("--stt-test-fn=") {
            return Some(val);
        }
        if a == "--stt-test-fn" {
            return iter.next().map(|s| s.as_str());
        }
    }
    None
}

/// Extract `--stt-probe-stack=func1,func2,...` from the argument list.
fn extract_probe_stack_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--stt-probe-stack=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--stt-topo=NsNcNt` from the argument list.
fn extract_topo_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--stt-topo=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--stt-flags=borrow,rebal` from the argument list.
fn extract_flags_arg(args: &[String]) -> Option<Vec<String>> {
    for a in args {
        if let Some(val) = a.strip_prefix("--stt-flags=")
            && !val.is_empty()
        {
            return Some(val.split(',').map(|s| s.to_string()).collect());
        }
    }
    None
}

/// Extract `--stt-work-type=NAME` from the argument list.
fn extract_work_type_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--stt-work-type=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Write a sidecar JSON file to STT_SIDECAR_DIR if the env var is set.
/// No-op when the var is absent, so tests remain runnable with plain cargo test.
fn write_sidecar(
    entry: &SttTestEntry,
    vm_result: &vmm::VmResult,
    stimulus_events: &[StimulusEvent],
    verify_result: &AssertResult,
    work_type: &str,
) {
    let dir = match std::env::var("STT_SIDECAR_DIR") {
        Ok(d) if !d.is_empty() => PathBuf::from(d),
        _ => return,
    };
    let topo = format!("{}s{}c{}t", entry.sockets, entry.cores, entry.threads,);
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
    };
    let path = dir.join(format!("{}.stt.json", entry.name));
    if let Ok(json) = serde_json::to_string_pretty(&sidecar) {
        let _ = std::fs::create_dir_all(&dir);
        if let Err(e) = std::fs::write(&path, json) {
            eprintln!("stt_test: write sidecar {}: {e}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that mutate env vars.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // Register a test entry in the distributed slice for unit testing find_test.
    fn __stt_inner_unit_test_dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }

    #[distributed_slice(STT_TESTS)]
    static __STT_ENTRY_UNIT_TEST_DUMMY: SttTestEntry = SttTestEntry {
        name: "__unit_test_dummy__",
        func: __stt_inner_unit_test_dummy,
        ..SttTestEntry::DEFAULT
    };

    #[test]
    fn find_test_registered_entry() {
        let entry = find_test("__unit_test_dummy__");
        assert!(entry.is_some(), "registered entry should be found");
        let entry = entry.unwrap();
        assert_eq!(entry.name, "__unit_test_dummy__");
        assert_eq!(entry.sockets, 1);
        assert_eq!(entry.cores, 2);
    }

    #[test]
    fn find_test_nonexistent() {
        assert!(find_test("__nonexistent_test_xyz__").is_none());
    }

    #[test]
    fn extract_test_fn_arg_equals() {
        let args = vec!["stt".into(), "run".into(), "--stt-test-fn=my_test".into()];
        assert_eq!(extract_test_fn_arg(&args), Some("my_test"));
    }

    #[test]
    fn extract_test_fn_arg_space() {
        let args = vec![
            "stt".into(),
            "run".into(),
            "--stt-test-fn".into(),
            "my_test".into(),
        ];
        assert_eq!(extract_test_fn_arg(&args), Some("my_test"));
    }

    #[test]
    fn extract_test_fn_arg_missing() {
        let args = vec!["stt".into(), "run".into()];
        assert!(extract_test_fn_arg(&args).is_none());
    }

    #[test]
    fn extract_test_fn_arg_trailing() {
        let args = vec!["stt".into(), "run".into(), "--stt-test-fn".into()];
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
        // Host /proc/cmdline does not contain STT_SHM_BASE/STT_SHM_SIZE.
        let result = parse_shm_params();
        assert!(
            result.is_none(),
            "host should not have STT_SHM_BASE in /proc/cmdline"
        );
    }

    // -- parse_shm_params_from_str tests --

    #[test]
    fn parse_shm_params_from_str_lowercase_hex() {
        let cmdline = "console=ttyS0 STT_SHM_BASE=0xfc000000 STT_SHM_SIZE=0x400000 quiet";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_uppercase_hex() {
        let cmdline = "STT_SHM_BASE=0XFC000000 STT_SHM_SIZE=0X400000";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xFC000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_no_prefix() {
        let cmdline = "STT_SHM_BASE=fc000000 STT_SHM_SIZE=400000";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_missing_base() {
        let cmdline = "console=ttyS0 STT_SHM_SIZE=0x400000";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_missing_size() {
        let cmdline = "STT_SHM_BASE=0xfc000000 quiet";
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
        let cmdline = "STT_SHM_BASE=0xZZZZ STT_SHM_SIZE=0x400000";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    // -- extract_test_fn_arg additional tests --

    #[test]
    fn extract_test_fn_arg_empty_value() {
        let args = vec!["stt".into(), "run".into(), "--stt-test-fn=".into()];
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
            std::env::set_var(key_prof, "/tmp/cov-target/stt-%p-%m.profraw");
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

    // -- resolve_kernel tests --

    #[test]
    fn resolve_kernel_with_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "STT_TEST_KERNEL";
        let prev = std::env::var(key).ok();
        let exe = crate::resolve_current_exe().unwrap();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, exe.to_str().unwrap()) };
        let result = resolve_kernel();
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), exe);
    }

    #[test]
    fn resolve_kernel_with_nonexistent_env_path() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "STT_TEST_KERNEL";
        let prev = std::env::var(key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, "/nonexistent/kernel/path") };
        let result = resolve_kernel();
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
        let key = "STT_SCHEDULER";
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
        let key = "STT_SCHEDULER";
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
        let key = "STT_TEST_KERNEL";
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
            output.starts_with("STT_TEST_KERNEL="),
            "expected STT_TEST_KERNEL=..., got: {output}"
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

    // -- extract_probe_stack_arg tests --

    #[test]
    fn extract_probe_stack_arg_equals() {
        let args = vec![
            "stt".into(),
            "run".into(),
            "--stt-probe-stack=func_a,func_b".into(),
        ];
        assert_eq!(
            extract_probe_stack_arg(&args),
            Some("func_a,func_b".to_string())
        );
    }

    #[test]
    fn extract_probe_stack_arg_missing() {
        let args = vec!["stt".into(), "run".into()];
        assert!(extract_probe_stack_arg(&args).is_none());
    }

    #[test]
    fn extract_probe_stack_arg_empty_value() {
        let args = vec!["stt".into(), "--stt-probe-stack=".into()];
        assert!(extract_probe_stack_arg(&args).is_none());
    }

    // -- extract_probe_output tests --

    #[test]
    fn extract_probe_output_valid_json() {
        use crate::probe::process::ProbeEvent;
        let payload = ProbePayload {
            events: vec![ProbeEvent {
                func_idx: 0,
                tid: 1,
                ts: 100,
                args: [0; 6],
                fields: vec![("p:task_struct.pid".to_string(), 42)],
                kstack: vec![],
                str_val: None,
            }],
            func_names: vec![(0, "schedule".to_string())],
            bpf_source_locs: Default::default(),
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
                    tid: 1,
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
                    tid: 1,
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
        let output = "noise\n  stt-0  [001]  0.500: sched_ext_dump: Debug dump\n  stt-0  [001]  0.501: sched_ext_dump: scheduler state\nmore";
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
        shell_cmds: &[],
    };
    static BORROW: FlagDecl = FlagDecl {
        name: "borrow",
        args: &["--borrow"],
        requires: &[],
        shell_cmds: &[],
    };
    static REBAL: FlagDecl = FlagDecl {
        name: "rebal",
        args: &["--rebal"],
        requires: &[],
        shell_cmds: &[],
    };
    static TEST_LLC: FlagDecl = FlagDecl {
        name: "llc",
        args: &["--llc"],
        requires: &[],
        shell_cmds: &[],
    };
    static TEST_STEAL: FlagDecl = FlagDecl {
        name: "steal",
        args: &["--steal"],
        requires: &[&TEST_LLC],
        shell_cmds: &[],
    };
    static BORROW_LONG: FlagDecl = FlagDecl {
        name: "borrow",
        args: &["--enable-borrow"],
        requires: &[],
        shell_cmds: &[],
    };
    static TEST_A: FlagDecl = FlagDecl {
        name: "a",
        args: &["-a"],
        requires: &[],
        shell_cmds: &[],
    };
    static TEST_B: FlagDecl = FlagDecl {
        name: "b",
        args: &["-b"],
        requires: &[],
        shell_cmds: &[],
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
                    avg_runnable_pct: 50.0,
                    min_runnable_pct: 40.0,
                    max_runnable_pct: 60.0,
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
            }),
            stimulus_events: vec![crate::timeline::StimulusEvent {
                elapsed_ms: 500,
                label: "StepStart[0]".to_string(),
                op_kind: Some("SetCpuset".to_string()),
                detail: Some("4 cpus".to_string()),
                total_iterations: None,
            }],
            work_type: "CpuSpin".to_string(),
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
        let args = vec!["bin".into(), "--stt-topo=2s4c2t".into()];
        assert_eq!(extract_topo_arg(&args), Some("2s4c2t".to_string()));
    }

    #[test]
    fn extract_topo_arg_missing() {
        let args = vec!["bin".into(), "--stt-test-fn=test".into()];
        assert!(extract_topo_arg(&args).is_none());
    }

    #[test]
    fn extract_topo_arg_empty_value() {
        let args = vec!["bin".into(), "--stt-topo=".into()];
        assert!(extract_topo_arg(&args).is_none());
    }

    #[test]
    fn extract_topo_arg_with_other_args() {
        let args = vec![
            "bin".into(),
            "--stt-test-fn=my_test".into(),
            "--stt-topo=1s2c1t".into(),
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
        assert!(!s.contains("kernel console"));
    }

    #[test]
    fn format_console_diagnostics_with_console() {
        let console = "line1\nline2\nKernel panic - not syncing";
        let s = format_console_diagnostics(console, -1, "payload started");
        assert!(s.contains("exit_code=-1"));
        assert!(s.contains("kernel console"));
        assert!(s.contains("Kernel panic"));
        assert!(s.contains("stage: payload started"));
    }

    #[test]
    fn format_console_diagnostics_truncates_long() {
        let lines: Vec<String> = (0..50).map(|i| format!("boot line {i}")).collect();
        let console = lines.join("\n");
        let s = format_console_diagnostics(&console, 0, "test");
        assert!(s.contains("last 20 lines"));
        assert!(s.contains("boot line 49"));
        assert!(!s.contains("boot line 29"));
    }

    #[test]
    fn format_console_diagnostics_short_console() {
        let console = "Linux version 6.14.0\nbooted ok";
        let s = format_console_diagnostics(console, 0, "test");
        assert!(s.contains("last 2 lines"));
        assert!(s.contains("Linux version 6.14.0"));
        assert!(s.contains("booted ok"));
    }

    // -- extract_work_type_arg tests --

    #[test]
    fn extract_work_type_arg_equals() {
        let args = vec!["stt".into(), "--stt-work-type=CpuSpin".into()];
        assert_eq!(extract_work_type_arg(&args), Some("CpuSpin".to_string()));
    }

    #[test]
    fn extract_work_type_arg_missing() {
        let args = vec!["stt".into(), "run".into()];
        assert!(extract_work_type_arg(&args).is_none());
    }

    #[test]
    fn extract_work_type_arg_empty_value() {
        let args = vec!["stt".into(), "--stt-work-type=".into()];
        assert!(extract_work_type_arg(&args).is_none());
    }

    // -- collect_sidecars tests --

    #[test]
    fn collect_sidecars_empty_dir() {
        let tmp = std::env::temp_dir().join("stt-sidecars-empty-test");
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
        let tmp = std::env::temp_dir().join("stt-sidecars-json-test");
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
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(tmp.join("test_x.stt.json"), &json).unwrap();
        // Non-stt JSON should be ignored.
        std::fs::write(tmp.join("other.json"), r#"{"key":"val"}"#).unwrap();
        let results = collect_sidecars(&tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "test_x");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_recurses_one_level() {
        let tmp = std::env::temp_dir().join("stt-sidecars-recurse-test");
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
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(sub.join("nested_test.stt.json"), &json).unwrap();
        let results = collect_sidecars(&tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "nested_test");
        assert!(!results[0].passed);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_skips_invalid_json() {
        let tmp = std::env::temp_dir().join("stt-sidecars-invalid-test");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("bad.stt.json"), "not json").unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_skips_non_stt_json() {
        let tmp = std::env::temp_dir().join("stt-sidecars-notstt-test");
        std::fs::create_dir_all(&tmp).unwrap();
        // File ends in .json but does NOT contain ".stt." in the name
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
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.work_type, "Bursty");
    }

    #[test]
    fn write_sidecar_noop_without_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "STT_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::remove_var(key) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = SttTestEntry {
            name: "__sidecar_noop__",
            func: dummy,
            auto_repro: false,
            ..SttTestEntry::DEFAULT
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
        };
        let verify_result = AssertResult::pass();
        // This should be a no-op because STT_SIDECAR_DIR is not set.
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin");

        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn write_sidecar_writes_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "STT_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("stt-sidecar-write-test");
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = SttTestEntry {
            name: "__sidecar_write_test__",
            func: dummy,
            auto_repro: false,
            ..SttTestEntry::DEFAULT
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
        };
        let verify_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin");

        let path = tmp.join("__sidecar_write_test__.stt.json");
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
        // STT_TESTS should contain at least the __unit_test_dummy__ entry.
        assert!(!STT_TESTS.is_empty());
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

    fn eevdf_entry(name: &'static str) -> SttTestEntry {
        SttTestEntry {
            name,
            func: dummy_test_fn,
            auto_repro: false,
            ..SttTestEntry::DEFAULT
        }
    }

    static SCHED_TEST: Scheduler = Scheduler {
        name: "test_sched",
        binary: SchedulerSpec::Name("test_sched_bin"),
        flags: &[],
        sysctls: &[],
        kargs: &[],
        assert: crate::assert::Assert::NONE,
    };

    fn sched_entry(name: &'static str) -> SttTestEntry {
        SttTestEntry {
            name,
            func: dummy_test_fn,
            scheduler: &SCHED_TEST,
            auto_repro: false,
            ..SttTestEntry::DEFAULT
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
        }
    }

    #[test]
    fn eval_eevdf_no_com2_output() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_eevdf_no_out__");
        let result = make_vm_result("", "boot log line\nKernel panic", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("payload produced no output"),
            "EEVDF with no COM2 output should say 'payload produced no output', got: {msg}",
        );
        assert!(
            !msg.contains("scheduler died"),
            "EEVDF error should not say 'scheduler died', got: {msg}",
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
        let sched_path = std::path::Path::new("/fake/sched");
        let err = evaluate_vm_result(
            &entry,
            &result,
            Some(sched_path),
            &assertions,
            &[],
            1,
            2,
            1,
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler died"),
            "scheduler present with no output should say 'scheduler died', got: {msg}",
        );
        assert!(
            !msg.contains("payload produced no output"),
            "should not say 'payload produced no output' when scheduler is set, got: {msg}",
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
        let sched_path = std::path::Path::new("/fake/sched");
        let err = evaluate_vm_result(
            &entry,
            &result,
            Some(sched_path),
            &assertions,
            &[],
            1,
            2,
            1,
            &no_repro,
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("scheduler died"),
            "should say scheduler died, got: {msg}",
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
    fn eval_timeout_no_result() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = eevdf_entry("__eval_timeout__");
        let result = make_vm_result("", "booting...\nstill booting...", 0, true);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("timed out"),
            "should say timed out, got: {msg}",
        );
        assert!(
            msg.contains("no result in COM2"),
            "should mention COM2, got: {msg}",
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
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("payload produced no output"),
            "non-parseable COM2 with EEVDF should say 'payload produced no output', got: {msg}",
        );
        assert!(
            !msg.contains("scheduler died"),
            "EEVDF should not say scheduler died, got: {msg}",
        );
    }

    #[test]
    fn eval_sched_ext_dump_included() {
        let output = "stt-0 [001] 0.5: sched_ext_dump: Debug dump line";
        let entry = sched_entry("__eval_dump__");
        let result = make_vm_result(output, "", -1, false);
        let assertions = crate::assert::Assert::NONE;
        let sched_path = std::path::Path::new("/fake/sched");
        let err = evaluate_vm_result(
            &entry,
            &result,
            Some(sched_path),
            &assertions,
            &[],
            1,
            2,
            1,
            &no_repro,
        )
        .unwrap_err();
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
            evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro,)
                .is_ok(),
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
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("failed:"),
            "failed AssertResult should say 'failed:', got: {msg}",
        );
        assert!(
            msg.contains("stuck 3000ms"),
            "should include failure details, got: {msg}",
        );
        assert!(
            msg.contains("spread 45%"),
            "should include all failure details, got: {msg}",
        );
    }

    #[test]
    fn eval_assert_failure_excludes_sched_log() {
        let json = r#"{"passed":false,"details":["worker 0 stuck 5000ms"],"stats":{"cgroups":[],"total_workers":0,"total_cpus":0,"total_migrations":0,"worst_spread":0.0,"worst_gap_ms":0,"worst_gap_cpu":0}}"#;
        let output = format!(
            "{RESULT_START}\n{json}\n{RESULT_END}\n{SCHED_OUTPUT_START}\nscheduler noise line\n{SCHED_OUTPUT_END}",
        );
        let entry = sched_entry("__eval_fail_no_sched_log__");
        let result = make_vm_result(&output, "", 0, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("worker 0 stuck 5000ms"),
            "should include assertion details, got: {msg}",
        );
        assert!(
            !msg.contains("scheduler noise"),
            "assertion failure should not include scheduler log, got: {msg}",
        );
        assert!(
            !msg.contains("--- scheduler log ---"),
            "assertion failure should not include scheduler log header, got: {msg}",
        );
    }

    #[test]
    fn eval_timeout_with_sched_includes_diagnostics() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("RUST_BACKTRACE", "1") };
        let entry = sched_entry("__eval_timeout_sched__");
        let result = make_vm_result("", "Linux version 6.14.0\nkernel panic here", -1, true);
        let assertions = crate::assert::Assert::NONE;
        let sched_path = std::path::Path::new("/fake/sched");
        let err = evaluate_vm_result(
            &entry,
            &result,
            Some(sched_path),
            &assertions,
            &[],
            1,
            2,
            1,
            &no_repro,
        )
        .unwrap_err();
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
            classify_init_stage("STT_INIT_STARTED\nsome noise"),
            "init started but payload never ran (cgroup/scheduler setup failed)",
        );
    }

    #[test]
    fn classify_payload_starting() {
        let output = "STT_INIT_STARTED\nSTT_PAYLOAD_STARTING\nsome output";
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
            classify_init_stage("STT_PAYLOAD_STARTING"),
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
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
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
        let result = make_vm_result("STT_INIT_STARTED\n", "boot log", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
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
        let output = "STT_INIT_STARTED\nSTT_PAYLOAD_STARTING\ngarbage";
        let result = make_vm_result(output, "", 1, false);
        let assertions = crate::assert::Assert::NONE;
        let err = evaluate_vm_result(&entry, &result, None, &assertions, &[], 1, 2, 1, &no_repro)
            .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("payload started but produced no test result"),
            "both sentinels should indicate payload ran but failed, got: {msg}",
        );
    }
}
