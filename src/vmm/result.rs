//! Public [`VmResult`] returned from [`super::KtstrVm::run`], plus
//! the internal [`VmRunState`] passed from `run_vm` to
//! `collect_results` and the [`KvmStatsTotals`] aggregate of per-vCPU
//! KVM counters.
//!
//! The split keeps the result-shaping types independent of the
//! orchestration code (which still lives in [`super::KtstrVm`]). Test
//! code outside `vmm/` constructs `VmResult` literals and reads
//! `KvmStatsTotals` fields, so both types stay public; `VmRunState`
//! is `pub(crate)`-only because it's an implementation detail of the
//! run-then-collect handoff.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::console;
use super::kvm;
use super::pi_mutex::PiMutex;
use super::shm_ring;
use super::vcpu::VcpuThread;
use super::virtio_blk::VirtioBlkCounters;
use crate::monitor;

/// Result of a VM execution.
#[derive(Debug)]
pub struct VmResult {
    /// Overall success flag: `true` when the test reported a pass AND
    /// the VM exited cleanly without crash, timeout, or watchdog.
    pub success: bool,
    /// Guest exit code as surfaced through the SHM ring
    /// (`MSG_TYPE_EXIT`) or COM2 sentinel.
    pub exit_code: i32,
    /// Wall-clock duration of the VM run.
    pub duration: Duration,
    /// True when the host hit its watchdog before the guest exited.
    pub timed_out: bool,
    /// Captured guest stdout (and any non-dmesg serial console content).
    pub output: String,
    /// Captured guest stderr (separated from `output` when the guest
    /// reported them distinctly).
    pub stderr: String,
    /// Host-side monitor report: sampled per-CPU state, stall
    /// verdicts, and SCX event deltas. `None` when the monitor did
    /// not run (host-only tests, early VM failure).
    pub monitor: Option<monitor::MonitorReport>,
    /// Data drained from the SHM ring buffer after VM exit.
    pub shm_data: Option<shm_ring::ShmDrainResult>,
    /// Stimulus events extracted from SHM ring entries.
    #[allow(dead_code)]
    pub stimulus_events: Vec<shm_ring::StimulusEvent>,
    /// BPF verifier stats collected from host-side memory reads.
    pub verifier_stats: Vec<monitor::bpf_prog::ProgVerifierStats>,
    /// KVM per-vCPU cumulative stats (requires Linux >= 5.15, x86_64 only).
    pub kvm_stats: Option<KvmStatsTotals>,
    /// Crash message from SHM (MSG_TYPE_CRASH). Reliable delivery via
    /// memcpy unlike serial which truncates large backtraces.
    pub crash_message: Option<String>,
    /// Wall-clock time from BSP exit to the moment
    /// [`super::KtstrVm::collect_results`] finishes assembling
    /// [`VmResult`].
    /// Records the host-side cost of every teardown step that runs
    /// after the guest has stopped advancing: watchdog join, AP joins,
    /// monitor join, BPF-writer join, SHM drain, exit/crash-message
    /// extraction, and BPF verifier-stat read. Always `Some(_)` for
    /// VMs whose [`super::KtstrVm::run_vm`] returns normally —
    /// including the host-watchdog timeout path, because
    /// `run_bsp_loop` exits cleanly with `timed_out = true` and
    /// `collect_results` still executes, populating the field.
    /// `None` only when `run_vm` does not complete (a BSP panic
    /// propagated through `?`, or any pre-BSP setup error that
    /// returns an `Err` before `VmRunState` is constructed) and on
    /// the `test_fixture` / skip-sidecar paths that never boot a VM.
    /// Persisted via
    /// [`SidecarResult`](crate::test_support::SidecarResult) so stats
    /// tooling can flag cleanup regressions across runs.
    pub cleanup_duration: Option<Duration>,
    /// Host-side virtio-blk device counters, sampled after the guest
    /// has exited. `Some(_)` when the builder attached a disk via
    /// [`super::KtstrVmBuilder::disk`]; `None` when no disk was
    /// configured and [`super::KtstrVm::init_virtio_blk`] returned
    /// `None`. The Arc is the same handle the device increments on
    /// the vCPU thread during request processing — by the time
    /// `collect_results` constructs the [`VmResult`] every vCPU has
    /// joined and no further mutation occurs, so a single
    /// `.load(Ordering::Relaxed)` per field on the consumer side
    /// observes the final cumulative totals.
    ///
    /// The counter struct exposes seven `AtomicU64` fields, each
    /// bumped from `process_requests`:
    ///
    ///   - `reads_completed` — count of `VIRTIO_BLK_T_IN` requests
    ///     that returned `S_OK` to the guest. Bumped together with
    ///     `bytes_read` per [`VirtioBlkCounters::record_read`] in
    ///     `src/vmm/virtio_blk.rs`.
    ///   - `writes_completed` — count of `VIRTIO_BLK_T_OUT` requests
    ///     that returned `S_OK`. Bumped together with `bytes_written`.
    ///   - `flushes_completed` — count of `VIRTIO_BLK_T_FLUSH`
    ///     requests that returned `S_OK` (real `fdatasync` for
    ///     read-write disks, no-op for `read_only`).
    ///   - `bytes_read` — total bytes returned to the guest for
    ///     completed reads.
    ///   - `bytes_written` — total bytes accepted from the guest for
    ///     completed writes.
    ///   - `throttled_count` — token-bucket stalls. The chain is
    ///     rolled back and the worker arms a retry timerfd; the
    ///     guest does not see `S_IOERR` for a stall (the request
    ///     is deferred until the bucket refills). This counter is
    ///     separate from `io_errors` so operators can distinguish
    ///     "throttle bucket drained, request deferred" from "real
    ///     IO problem".
    ///   - `io_errors` — every path that reports `S_IOERR`:
    ///     spec violations, backend `pread`/`pwrite` errors,
    ///     malformed chains, `add_used` failures.
    ///     Stalls do not report `S_IOERR`; see `throttled_count`.
    ///
    /// Counters are cumulative for the device's lifetime. A guest
    /// driver re-bind (writing `STATUS=0` to `VIRTIO_MMIO_STATUS`
    /// triggers `VirtioBlk::reset`) does NOT zero them — the
    /// counters Arc is shared across reset cycles and an operator
    /// observes a monotonically non-decreasing series spanning the
    /// entire device lifetime, not just a post-reset fragment.
    ///
    /// Reading example:
    ///
    /// ```ignore
    /// use std::sync::atomic::Ordering;
    /// let r: VmResult = builder.run()?;
    /// let c = r.virtio_blk_counters.expect("disk attached");
    /// assert!(c.reads_completed.load(Ordering::Relaxed) > 0);
    /// ```
    ///
    /// `#[allow(dead_code)]` mirrors `stimulus_events` above: the
    /// field is part of the public API surface and read by user
    /// test code outside `lib.rs`, but the lib build doesn't see
    /// any in-tree readers because no lib code path calls
    /// `.virtio_blk_counters` on a `VmResult`. The in-tree readers
    /// live in unit tests.
    #[allow(dead_code)]
    pub virtio_blk_counters: Option<Arc<VirtioBlkCounters>>,
}

impl VmResult {
    /// Minimal "nothing happened" fixture for tests that exercise
    /// code consuming a [`VmResult`] without actually booting a VM
    /// (the sidecar-write tests in `src/test_support/sidecar.rs`
    /// are the primary users). Every field carries the empty /
    /// default / `None` value that `run_vm` would produce for a
    /// VM that launched, exited cleanly with exit code 0, and
    /// produced no telemetry. Tests that need a specific field
    /// override it with a struct-update expression:
    ///
    /// ```ignore
    /// let result = VmResult { success: false, ..VmResult::test_fixture() };
    /// ```
    ///
    /// Gated on `#[cfg(test)]` so the symbol does not appear in
    /// release builds — production `VmResult` values flow from
    /// `run_vm` and never from this fixture. See
    /// `sidecar_vm_result_is_test_fixture_boilerplate` in
    /// `test_support/sidecar.rs` for the motivating deduplication
    /// (7 identical literal constructions collapsed to a single
    /// call).
    #[cfg(test)]
    pub fn test_fixture() -> Self {
        Self {
            success: true,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
        }
    }
}

/// Per-vCPU KVM stats read after VM exit. Each map holds cumulative
/// counter values from the VM's lifetime.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct KvmStatsTotals {
    /// Per-vCPU stat maps. Index is vCPU id.
    pub per_vcpu: Vec<HashMap<String, u64>>,
}

/// KVM stat names surfaced in sidecar output for scheduler testing.
///
/// Covers VM exit rate, halt-polling behavior, preemption notifications,
/// signal-driven exits, and hypercall counts; all fields scheduler
/// authors typically correlate with scx decisions.
#[allow(dead_code)]
pub const KVM_INTERESTING_STATS: &[&str] = &[
    "exits",
    "halt_exits",
    "halt_successful_poll",
    "halt_attempted_poll",
    "halt_wait_ns",
    "preemption_reported",
    "signal_exits",
    "hypercalls",
];

impl KvmStatsTotals {
    /// Sum a stat across all vCPUs.
    pub fn sum(&self, name: &str) -> u64 {
        self.per_vcpu.iter().filter_map(|m| m.get(name)).sum()
    }

    /// Average a stat across all vCPUs (returns 0 if no vCPUs).
    pub fn avg(&self, name: &str) -> u64 {
        if self.per_vcpu.is_empty() {
            return 0;
        }
        self.sum(name) / self.per_vcpu.len() as u64
    }
}

/// State returned by [`super::KtstrVm::run_vm`] after the BSP exits.
/// Passed to [`super::KtstrVm::collect_results`] to produce
/// [`VmResult`].
pub(crate) struct VmRunState {
    pub(crate) exit_code: i32,
    pub(crate) timed_out: bool,
    pub(crate) ap_threads: Vec<VcpuThread>,
    pub(crate) monitor_handle: Option<JoinHandle<monitor::reader::MonitorLoopResult>>,
    pub(crate) bpf_write_handle: Option<JoinHandle<()>>,
    pub(crate) freeze_coordinator: Option<JoinHandle<()>>,
    pub(crate) com1: Arc<PiMutex<console::Serial>>,
    pub(crate) com2: Arc<PiMutex<console::Serial>>,
    pub(crate) kill: Arc<AtomicBool>,
    /// Broadcast freeze flag for the failure-dump coordinator. When the
    /// coordinator receives a guest-side error-exit signal it sets this
    /// to true, kicks every vCPU, waits for all `parked` flags to flip
    /// true, and then reads guest BPF map state. Released to false to
    /// resume normal execution. Lives alongside `kill` so the same Arc
    /// pattern (broadcast + per-vCPU ACK) covers both shutdown and
    /// freeze rendezvous.
    pub(crate) freeze: Arc<AtomicBool>,
    pub(crate) vm: kvm::KtstrKvm,
    /// Captured immediately after the BSP exits its run loop. Subtracted
    /// from `Instant::now()` in [`super::KtstrVm::collect_results`]
    /// right before the [`VmResult`] is returned to populate
    /// [`VmResult::cleanup_duration`]. Records the wall-clock cost of
    /// every host-side teardown step that runs after the guest has
    /// stopped advancing, in execution order: the watchdog-thread join
    /// in [`super::KtstrVm::run_vm`], then the AP-thread joins, the
    /// monitor-thread join, the BPF-map-writer join, the SHM-ring
    /// drain, the post-exit exit-code/crash-message extraction, and
    /// finally the BPF verifier-stat read inside
    /// [`super::KtstrVm::collect_results`].
    pub(crate) cleanup_start: Instant,
    /// Cloned counter handle from [`super::KtstrVm::init_virtio_blk`]
    /// when a disk was attached, captured before the device-arc is
    /// dropped so [`super::KtstrVm::collect_results`] can plumb it
    /// onto [`VmResult::virtio_blk_counters`]. The device increments
    /// these counters on the vCPU thread during request processing;
    /// by the time `collect_results` reads this field every vCPU
    /// thread has joined, so the Arc holds the final cumulative
    /// totals.
    pub(crate) virtio_blk_counters: Option<Arc<VirtioBlkCounters>>,
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn vm_result_fields_carry_values() {
        let r = VmResult {
            success: true,
            exit_code: 0,
            duration: Duration::from_secs(5),
            timed_out: false,
            output: "hello world".into(),
            stderr: "boot log".into(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: Some(Duration::from_millis(50)),
            virtio_blk_counters: None,
        };
        assert!(r.success);
        assert_eq!(r.exit_code, 0);
        assert!(!r.timed_out);
        assert_eq!(r.duration, Duration::from_secs(5));
        assert_eq!(r.output, "hello world");
        assert_eq!(r.stderr, "boot log");
        assert!(r.monitor.is_none());
        assert!(r.shm_data.is_none());
        assert!(r.stimulus_events.is_empty());
        assert_eq!(r.cleanup_duration, Some(Duration::from_millis(50)));
        assert!(r.virtio_blk_counters.is_none());
        // Second construction covers the opposite polarity of
        // every boolean/numeric field so no field is silently
        // dropped by a future refactor that only exercises the
        // success path.
        let r2 = VmResult {
            success: false,
            exit_code: 1,
            duration: Duration::from_millis(500),
            timed_out: true,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: Some(Arc::new(VirtioBlkCounters::default())),
        };
        assert!(!r2.success);
        assert_eq!(r2.exit_code, 1);
        assert!(r2.timed_out);
        assert_eq!(r2.duration, Duration::from_millis(500));
        assert!(r2.cleanup_duration.is_none());
        // Opposite polarity: counters present. Reads must observe
        // the default-zero values for every field — a future field
        // added to VirtioBlkCounters that doesn't initialise to 0
        // would break the "fresh device reports zero activity"
        // contract that VmResult readers rely on. The Arc handle is
        // the same one `init_virtio_blk` clones onto `VmRunState`,
        // so test code that wants to assert on disk activity calls
        // `.load()` on each counter through this field after the VM
        // exits.
        let counters = r2.virtio_blk_counters.as_ref().unwrap();
        assert_eq!(counters.reads_completed.load(Ordering::Relaxed), 0,);
        assert_eq!(counters.writes_completed.load(Ordering::Relaxed), 0,);
        assert_eq!(counters.flushes_completed.load(Ordering::Relaxed), 0,);
        assert_eq!(counters.bytes_read.load(Ordering::Relaxed), 0,);
        assert_eq!(counters.bytes_written.load(Ordering::Relaxed), 0,);
        assert_eq!(counters.throttled_count.load(Ordering::Relaxed), 0,);
        assert_eq!(counters.io_errors.load(Ordering::Relaxed), 0,);
    }

    #[test]
    fn vm_result_without_monitor_has_no_samples() {
        let r = VmResult {
            success: true,
            exit_code: 0,
            duration: Duration::from_secs(1),
            timed_out: false,
            output: "test output".into(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
        };
        assert!(r.monitor.is_none());
        // Output and exit_code must still be accessible.
        assert_eq!(r.output, "test output");
        assert_eq!(r.exit_code, 0);
    }

    #[test]
    fn vm_result_with_monitor_carries_summary() {
        let summary = monitor::MonitorSummary {
            prog_stats_deltas: None,
            total_samples: 5,
            max_imbalance_ratio: 3.5,
            max_local_dsq_depth: 10,
            stall_detected: true,
            event_deltas: None,
            schedstat_deltas: None,
            ..Default::default()
        };
        let report = monitor::MonitorReport {
            samples: vec![],
            summary: summary.clone(),
            ..Default::default()
        };
        let r = VmResult {
            success: false,
            exit_code: 1,
            duration: Duration::from_millis(500),
            timed_out: true,
            output: String::new(),
            stderr: "kernel panic".into(),
            monitor: Some(report),
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
        };
        let mon = r.monitor.as_ref().unwrap();
        assert_eq!(mon.summary.total_samples, 5);
        assert!((mon.summary.max_imbalance_ratio - 3.5).abs() < f64::EPSILON);
        assert_eq!(mon.summary.max_local_dsq_depth, 10);
        assert!(mon.summary.stall_detected);
        assert!(r.timed_out);
        assert_eq!(r.exit_code, 1);
        assert_eq!(r.stderr, "kernel panic");
    }
}
