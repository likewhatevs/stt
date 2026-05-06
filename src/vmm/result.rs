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
use super::host_comms::BulkDrainResult;
use super::kvm;
use super::pi_mutex::PiMutex;
use super::vcpu::{VcpuThread, WatchpointArm};
use super::virtio_blk::VirtioBlkCounters;
use super::virtio_net::VirtioNetCounters;
use super::wire;
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
    /// TLV messages drained from the guest after VM exit. Merges
    /// mid-flight bytes the freeze coordinator pulled off
    /// virtio-console port 1 during the run with the final port-1
    /// `port1_tx_buf` flush.
    pub guest_messages: Option<BulkDrainResult>,
    /// Stimulus events extracted from guest TLV entries.
    #[allow(dead_code)]
    pub stimulus_events: Vec<wire::StimulusEvent>,
    /// BPF verifier stats collected from host-side memory reads.
    pub verifier_stats: Vec<monitor::bpf_prog::ProgVerifierStats>,
    /// KVM per-vCPU cumulative stats (requires Linux >= 5.14).
    pub kvm_stats: Option<KvmStatsTotals>,
    /// Crash message extracted from COM2 output via
    /// [`crate::test_support::extract_panic_message`]. The guest
    /// panic hook in `rust_init.rs` writes `PANIC: <info>\n<bt>\n`
    /// to `/dev/ttyS1` synchronously inside `KVM_RUN`, so the host
    /// captures the full backtrace in `output` even when the guest
    /// is wedged. `None` when no `PANIC:`-prefixed line was seen.
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
    /// `None`. The Arc is the same handle the device increments
    /// from `drain_bracket_impl` (production cfg: on the dedicated
    /// `ktstr-vblk` worker thread; cfg(test): inline on the test
    /// thread) — by the time `collect_results` constructs the
    /// [`VmResult`] every vCPU and the worker have joined and no
    /// further mutation occurs, so a single
    /// `.load(Ordering::Relaxed)` per field on the consumer side
    /// observes the final cumulative totals.
    ///
    /// The counter struct exposes nine `AtomicU64` fields, each
    /// bumped from `drain_bracket_impl` (in `src/vmm/virtio_blk/device.rs`)
    /// via the `VirtioBlkCounters::record_*` helpers. Per-request
    /// cumulative counters, per-event cumulative counters, and
    /// per-request live gauges are kept distinct per the
    /// counter-taxonomy doc on `VirtioBlkCounters`:
    ///
    ///   - `reads_completed` — count of `VIRTIO_BLK_T_IN` requests
    ///     that returned `S_OK` to the guest. Bumped together with
    ///     `bytes_read` per [`VirtioBlkCounters::record_read`].
    ///   - `writes_completed` — count of `VIRTIO_BLK_T_OUT` requests
    ///     that returned `S_OK`. Bumped together with `bytes_written`.
    ///   - `flushes_completed` — count of `VIRTIO_BLK_T_FLUSH`
    ///     requests that returned `S_OK` (real `fdatasync` for
    ///     read-write disks, no-op for `read_only`).
    ///   - `bytes_read` — total bytes returned to the guest for
    ///     completed reads.
    ///   - `bytes_written` — total bytes accepted from the guest for
    ///     completed writes.
    ///   - `throttled_count` — cumulative token-bucket **stall events**
    ///     for the device's lifetime. The chain is rolled back and
    ///     the worker arms a retry timerfd; the guest does not see
    ///     `S_IOERR` for a stall (the request is deferred until the
    ///     bucket refills). This counter is separate from `io_errors`
    ///     so operators can distinguish "throttle bucket drained,
    ///     request deferred" from "real IO problem". Per-event (NOT
    ///     per-request): a single chain that stalls twice produces
    ///     two bumps.
    ///   - `io_errors` — every path that reports `S_IOERR`:
    ///     spec violations, backend `pread`/`pwrite` errors,
    ///     malformed chains, `add_used` failures.
    ///     Stalls do not report `S_IOERR`; see `throttled_count`.
    ///   - `currently_throttled_gauge` — **live gauge**: how many
    ///     requests are RIGHT NOW waiting for throttle tokens.
    ///     Increments when a chain transitions into stalled,
    ///     decrements on retry success or reset. Bounded at 0 or 1
    ///     on this single-queue device. NOT cumulative — answers
    ///     "what's stuck now," distinct from `throttled_count`
    ///     which answers "how many stall events happened over
    ///     time."
    ///   - `invalid_avail_idx_count` — cumulative count of
    ///     `Error::InvalidAvailRingIndex` events observed by
    ///     `drain_bracket_impl` (avail.idx more than `queue.size`
    ///     ahead of `next_avail` — a virtio-v1.2 §2.7.13.3
    ///     avail.idx-distance violation by the guest). Per-event
    ///     counter; the `queue_poisoned` flag short-circuits
    ///     subsequent kicks so one guest fault produces exactly
    ///     one bump regardless of how many notifications follow
    ///     before reset.
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
    /// let r: VmResult = builder.run()?;
    /// let c = r.virtio_blk_counters.expect("disk attached");
    /// assert!(c.reads_completed() > 0);
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
    /// Host-side virtio-net device counters, sampled after the guest
    /// has exited. `Some(_)` when the builder attached a network via
    /// [`super::KtstrVmBuilder::network`]; `None` when no network was
    /// configured and [`super::KtstrVm::init_virtio_net`] returned
    /// `None`. The Arc is the same handle the device increments on
    /// the vCPU thread inside `process_tx_loopback` — by the time
    /// `collect_results` constructs the [`VmResult`] every vCPU has
    /// joined and no further mutation occurs, so a single
    /// `.load(Ordering::Relaxed)` per field on the consumer side
    /// observes the final cumulative totals.
    ///
    /// The counter struct exposes eleven `AtomicU64` fields, each
    /// bumped from `process_tx_loopback`:
    ///
    ///   - `tx_packets` — count of TX chains the device accepted
    ///     and marked used; advances per parsed chain regardless of
    ///     downstream RX outcome.
    ///   - `tx_bytes` — bytes of L2 frame data captured from
    ///     successfully parsed TX chains (excludes the 12-byte
    ///     virtio header).
    ///   - `rx_packets` / `rx_bytes` — count + bytes of RX chains
    ///     successfully written and marked used. In v0's pure-
    ///     loopback mode the steady-state expectation is
    ///     `rx_packets == tx_packets - tx_dropped_no_rx_buffer`;
    ///     asymmetric counts surface RX-side breakage.
    ///   - `tx_dropped_no_rx_buffer` — successfully-captured TX
    ///     frames the device could not deliver because the RX queue
    ///     was empty (back-pressure event).
    ///   - `tx_chain_invalid` / `rx_chain_invalid` — chains rejected
    ///     for malformed shape (short header, wrong direction,
    ///     attacker-controlled descriptor address overflow).
    ///   - `rx_write_failed` — RX chain whose shape was valid but
    ///     whose guest-memory `write_slice` (header or frame) hit
    ///     an unmapped GPA. Distinct from `rx_chain_invalid` so an
    ///     operator can tell "guest violated the RX descriptor-
    ///     direction rule" from "guest posted a buffer at an
    ///     unmapped GPA"; the two are mutually exclusive per chain.
    ///   - `tx_add_used_failures` / `rx_add_used_failures` —
    ///     `add_used` failures, indicating the queue's used-ring
    ///     address itself is unmapped or otherwise inaccessible.
    ///     Distinct from the `*_chain_invalid` / `rx_write_failed`
    ///     counters so an operator can tell "guest sent malformed
    ///     frame" / "guest's posted buffer GPA was unmapped" from
    ///     "queue itself is broken".
    ///   - `invalid_avail_idx_count` — cumulative count of
    ///     `Error::InvalidAvailRingIndex` events observed by
    ///     `process_tx_loopback` (avail.idx more than `queue.size`
    ///     ahead of `next_avail` — virtio-v1.2 §2.7.13.3 violation
    ///     by the guest). Per-event counter; the per-queue
    ///     `queue_poisoned` flag short-circuits subsequent kicks
    ///     so one guest fault produces exactly one bump regardless
    ///     of how many notifications follow before reset.
    ///
    /// Counters are cumulative for the device's lifetime — a guest
    /// driver re-bind (writing `STATUS=0`) does NOT zero them.
    #[allow(dead_code)]
    pub virtio_net_counters: Option<Arc<VirtioNetCounters>>,
    /// Snapshot bridge populated by the freeze coordinator over the
    /// run's lifetime. Every `Op::Snapshot` and `Op::WatchSnapshot`
    /// fire stores a `FailureDumpReport` keyed by its tag.
    ///
    /// `#[ktstr_test]` test bodies whose scenario fires snapshot
    /// ops in the guest assert on the captured reports through a
    /// `post_vm = NAME` attribute. The named callback runs on the
    /// HOST after `vm.run()` returns (see
    /// [`crate::test_support::KtstrTestEntry::post_vm`]) and
    /// receives `&VmResult`; it calls
    /// [`crate::scenario::snapshot::SnapshotBridge::drain`] on
    /// this field to take ownership of the stored reports and
    /// walks them — typically through
    /// [`crate::scenario::snapshot::Snapshot::new`] for typed
    /// access to map values, per-CPU entries, and scalar
    /// variables. Out-of-tree consumers can drain the bridge the
    /// same way: `VmResult` is in `ktstr::prelude`.
    ///
    /// Always present after a successful `run_vm`; `None`-equivalent
    /// (empty) when the VM crashed before any snapshot fired.
    pub snapshot_bridge: crate::scenario::snapshot::SnapshotBridge,
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
            guest_messages: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
        }
    }
}

/// Build an empty `SnapshotBridge` whose capture callback always
/// returns `None`. Used by `VmResult::test_fixture` and the legacy
/// `VmResult` literal constructions in unit tests so they still
/// compile after the snapshot_bridge field landed. Production
/// `run_vm` constructs its own bridge whose callback is
/// intentionally unused — the freeze coordinator stores reports
/// directly via `bridge.store(name, report)`.
#[cfg(test)]
pub(crate) fn empty_snapshot_bridge_for_tests() -> crate::scenario::snapshot::SnapshotBridge {
    let cb: crate::scenario::snapshot::CaptureCallback = std::sync::Arc::new(|_| None);
    crate::scenario::snapshot::SnapshotBridge::new(cb)
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
///
/// Per-arch availability: `halt_exits`, `preemption_reported`, and
/// `hypercalls` are published by KVM only on x86. On aarch64 the
/// kernel does not expose these stats via `KVM_GET_STATS_FD`; they
/// are absent from the per-vCPU map and read as `0` from
/// [`KvmStatsTotals::sum`] / [`KvmStatsTotals::avg`]. The remaining
/// names (`exits`, `halt_successful_poll`, `halt_attempted_poll`,
/// `halt_wait_ns`, `signal_exits`) are published on both arches.
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
    /// Freeze coordinator handle, always `None` in the
    /// production path: [`super::KtstrVm::run_vm`] joins the
    /// coordinator BEFORE the BSP `VcpuFd` falls out of scope so the
    /// coordinator's captured BSP `ImmediateExitHandle` cannot
    /// outlive the kvm_run mmap (UAF prevention). The optional shape
    /// is preserved so the field stays trivially constructible in
    /// any future test-only or alternative-orchestration path that
    /// might not perform the early join.
    pub(crate) freeze_coordinator: Option<JoinHandle<()>>,
    pub(crate) com1: Arc<PiMutex<console::Serial>>,
    pub(crate) com2: Arc<PiMutex<console::Serial>>,
    pub(crate) kill: Arc<AtomicBool>,
    /// Wake fd paired with `kill`. Setters that flip `kill`
    /// (`collect_results`, vCPU shutdown classifier, panic hook)
    /// also write to this EventFd so any consumer blocked in
    /// `epoll_wait` (notably the freeze coordinator and the
    /// monitor sampler) wakes within microseconds of the flip
    /// rather than waiting up to one full poll interval. The
    /// AtomicBool above remains the source of truth — the EventFd
    /// is purely a wake signal. EFD_NONBLOCK so a saturated
    /// counter never stalls the writer.
    pub(crate) kill_evt: Arc<vmm_sys_util::eventfd::EventFd>,
    /// Broadcast freeze flag for the failure-dump coordinator. When the
    /// coordinator receives a guest-side error-exit signal it sets this
    /// to true, kicks every vCPU, waits for all `parked` flags to flip
    /// true, and then reads guest BPF map state. Released to false to
    /// resume normal execution. Lives alongside `kill` so the same Arc
    /// pattern (broadcast + per-vCPU ACK) covers both shutdown and
    /// freeze rendezvous.
    pub(crate) freeze: Arc<AtomicBool>,
    /// Hardware-watchpoint arming state Arc, forwarded so
    /// [`super::KtstrVm::collect_results`] can invalidate the
    /// `kind_host_ptr` and `request_kva` slots after every vCPU
    /// thread joins but BEFORE `vm` drops.
    ///
    /// Without the invalidation, the slots' published values
    /// continue to address (a) a host pointer into `vm.guest_mem`'s
    /// mapping that becomes unmapped when `vm` drops and (b) a
    /// guest KVA whose translation goes through the same mapping.
    /// The freeze coordinator joins before `vm` drops in
    /// `run_vm`, and AP threads join inside `collect_results` —
    /// but defense-in-depth says we zero the slots once every
    /// reader is gone so any future restructuring (a stray Arc
    /// clone surviving past teardown, a follow-up that adds a
    /// new reader path) cannot trip a use-after-free.
    ///
    /// Declared before `vm` so the implicit drop order on
    /// `VmRunState` teardown drops `watchpoint` first: any Arc
    /// clone outliving the struct can no longer dereference its
    /// `kind_host_ptr` after `vm.guest_mem` has unmapped, even if
    /// a future caller forgets the explicit pre-drop
    /// invalidation in `collect_results`.
    pub(crate) watchpoint: Arc<WatchpointArm>,
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
    /// Cloned counter handle from [`super::KtstrVm::init_virtio_net`]
    /// when a network was attached, captured before the device-arc is
    /// dropped so [`super::KtstrVm::collect_results`] can plumb it
    /// onto [`VmResult::virtio_net_counters`]. Same Arc-handoff
    /// pattern as `virtio_blk_counters` above.
    pub(crate) virtio_net_counters: Option<Arc<VirtioNetCounters>>,
    /// Snapshot bridge owning every report captured during the run.
    /// The freeze coordinator clones this bridge into its closure
    /// state; on every guest-side
    /// [`crate::vmm::wire::MSG_TYPE_SNAPSHOT_REQUEST`] frame the
    /// coordinator's TOKEN_TX handler decoded with kind
    /// [`crate::vmm::wire::SNAPSHOT_KIND_CAPTURE`], the dispatch runs
    /// `freeze_and_capture(false)` and stores the resulting
    /// `FailureDumpReport` here keyed by the snapshot name. After
    /// VM exit, [`super::KtstrVm::collect_results`] forwards the
    /// bridge onto [`VmResult::snapshot_bridge`] so the test code
    /// can drain captured snapshots and walk them via the
    /// [`crate::scenario::snapshot::Snapshot`] accessor surface.
    pub(crate) snapshot_bridge: crate::scenario::snapshot::SnapshotBridge,
    /// Cached aarch64 TCR_EL1 register, populated lazily by the BSP
    /// once the guest kernel programs the MMU. Always `None` on
    /// x86_64 (the register does not exist). Threads that construct
    /// a `GuestKernel` for page-table walks (monitor, BPF map writer,
    /// freeze coordinator, post-exit verifier-stats collector) read
    /// this atomic to feed the granule-agnostic walker (4 KB / 16 KB
    /// / 64 KB). A 0 reading on aarch64 means "kernel hasn't reached
    /// MMU bring-up yet"; the walker's T1SZ=0 gate rejects walks in
    /// that state and the affected lookup returns `None` cleanly.
    pub(crate) tcr_el1: Option<Arc<std::sync::atomic::AtomicU64>>,
    /// Cached BSP CR3 (x86_64) / TTBR1_EL1 (aarch64), populated lazily
    /// by the BSP loop after initial page-table setup. Used by
    /// post-exit `GuestKernel` constructions to walk the live page
    /// tables for `phys_base` resolution. `0` means the cache wasn't
    /// populated (early boot crash); the walk fails and `phys_base`
    /// falls back to `0`, which produces correct translations on
    /// non-KASLR boots.
    pub(crate) cr3: Arc<std::sync::atomic::AtomicU64>,
    /// Cached vmlinux bytes for collect_verifier_stats. Avoids
    /// re-reading from disk (14-28s on cold cache).
    pub(crate) vmlinux_data: Option<Arc<Vec<u8>>>,
    /// Pre-built prog accessor from the accessor-init worker.
    /// When present, `collect_verifier_stats` skips the ~4s
    /// ELF/BTF parse and uses this directly.
    pub(crate) prog_accessor: Option<crate::monitor::bpf_prog::GuestMemProgAccessorOwned>,
    /// Virtio-console device shared with vCPU threads. Carries the
    /// port-1 (`/dev/vport0p1`) bulk TLV stream from guest to host;
    /// `collect_results` calls `drain_bulk()` after the run to feed
    /// `parse_tlv_stream` and produce the `BulkDrainResult` that
    /// `VmResult.guest_messages` exposes to test verdicts.
    pub(crate) virtio_con: Arc<crate::vmm::PiMutex<crate::vmm::virtio_console::VirtioConsole>>,
    /// Bulk TLV entries the freeze coordinator parsed from
    /// `port1_tx_buf` mid-run. The coord's TOKEN_TX handler reads
    /// the device's accumulated bulk bytes, feeds them through
    /// [`crate::vmm::bulk::HostAssembler`], and stashes every parsed
    /// frame here so [`super::KtstrVm::collect_results`] can merge
    /// them into `VmResult::guest_messages` alongside the post-exit
    /// `drain_bulk` and the post-mortem SHM CRASH-ring drain.
    /// Without this stash every EXIT / TEST / PAYLOAD_METRICS /
    /// RAW_PAYLOAD_OUTPUT / PROFRAW frame consumed by the coord
    /// would vanish — only the leftover bytes that arrived on
    /// `port1_tx_buf` after the coord exited would reach the
    /// verdict, and a typical run would surface no metrics.
    pub(crate) bulk_messages: Arc<std::sync::Mutex<Vec<crate::vmm::wire::ShmEntry>>>,
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
            guest_messages: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: Some(Duration::from_millis(50)),
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
        };
        assert!(r.success);
        assert_eq!(r.exit_code, 0);
        assert!(!r.timed_out);
        assert_eq!(r.duration, Duration::from_secs(5));
        assert_eq!(r.output, "hello world");
        assert_eq!(r.stderr, "boot log");
        assert!(r.monitor.is_none());
        assert!(r.guest_messages.is_none());
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
            guest_messages: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: Some(Arc::new(VirtioBlkCounters::default())),
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
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
            guest_messages: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
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
            guest_messages: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
            cleanup_duration: None,
            virtio_blk_counters: None,
            virtio_net_counters: None,
            snapshot_bridge: empty_snapshot_bridge_for_tests(),
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
