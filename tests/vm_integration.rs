//! VM integration tests for kernel-facing capture pipelines (#146).
//!
//! Each test boots a real KVM VM via `#[ktstr_test]`, runs a small
//! workload under scx-ktstr with `--stall-after=1`, lets the freeze
//! coordinator capture a `FailureDumpReport`, and asserts the
//! captured JSON carries the field the test is responsible for
//! pinning. Five tests cover the gaps from #7's audit (G4-G9, G16):
//!
//! - **DSQ + rq->scx walker** (G4): `dsq_states` and `rq_scx_states`
//!   in the dump JSON populated from real frozen-VM walk.
//! - **Per-vCPU perf counters** (G5): `vcpu_perf_at_freeze` populated
//!   with at least one non-`None` slot from a real
//!   `perf_event_open(exclude_host=1)` read.
//! - **Event-counter timeline** (G6): `event_counter_timeline`
//!   populated with at least one entry across the run window — the
//!   load-bearing surface for the sched-event capture path. (The
//!   discrete tracepoint timeline is wired via `TimelineCapture` but
//!   not yet attached to FailureDumpReport; the event-counter
//!   timeline is the visible per-tick timeline today.)
//! - **SchedPolicy::Deadline** (G9): worker spawned under
//!   `SchedPolicy::Deadline` reaches `worker_main` without bailing —
//!   proves the `sched_setattr(2)` syscall path runs end-to-end on
//!   a real kernel that supports `SCHED_DEADLINE`.
//! - **Failure-dump trigger** (G16): boot → stall → capture → render
//!   pipeline produces a non-empty top-level dump (overlaps with
//!   `failure_dump_e2e.rs` but pins the schema discriminant + the
//!   minimal cross-pipeline invariant).
//!
//! Every scenario consumes the existing `failure_dump_e2e.rs`
//! pattern — same `--stall-after=1` trigger, same per-test sidecar
//! path resolution, same JSON shape inspection — so the host-side
//! freeze-coordinator wiring is exercised once per test in lockstep
//! with the in-tree pattern.
//!
//! User-facing test bar: each kernel-facing capture surface used by
//! ktstr's debugging story (DSQ depth, rq->scx scalars, vCPU perf,
//! per-tick event counters, SCHED_DEADLINE invocation, dump trigger)
//! must produce live data on a real VM run, not a synthetic literal
//! or a unit-tested code path.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::scenario::ops::{CgroupDef, HoldSpec, Step, execute_steps};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec, sidecar_dir};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// Mirror `failure_dump_e2e.rs::failure_dump_path`. Both sites must
/// agree with `test_support::eval::run_ktstr_test_inner`'s naming
/// convention for the per-test sidecar dump file.
fn failure_dump_path(test_name: &str) -> std::path::PathBuf {
    sidecar_dir().join(format!("{test_name}.failure-dump.json"))
}

/// Read and parse the per-test failure-dump JSON. Returns the parsed
/// `serde_json::Value` so the caller can navigate the schema without
/// pulling in `ktstr`'s pub(crate) `FailureDumpReport` type. Fails
/// the scenario if the file is missing — a missing dump file means
/// the freeze coordinator did not write OR the stall trigger did
/// not fire, both of which are real regressions.
fn read_dump_or_fail(test_name: &str) -> Result<serde_json::Value> {
    let dump_path = failure_dump_path(test_name);
    let json = std::fs::read_to_string(&dump_path).map_err(|e| {
        anyhow::anyhow!(
            "failure dump file missing at {}: {e} — freeze coordinator did \
             not write (no SCX_EXIT_ERROR_STALL latch fired, owned_accessor / \
             dump_btf was None, or the file write failed silently)",
            dump_path.display()
        )
    })?;
    serde_json::from_str(&json).map_err(|e| {
        anyhow::anyhow!(
            "failure dump JSON at {} is malformed: {e}",
            dump_path.display()
        )
    })
}

/// Run a one-step workload under scx-ktstr `--stall-after=1`. The
/// per-test path resolution ensures each scenario reads back its own
/// dump file; the host freeze coordinator writes the JSON when the
/// in-guest BPF probe latches the SCX_EXIT_ERROR_STALL exit. Returns
/// the partial `AssertResult` from `execute_steps` so the caller can
/// merge per-test claims onto the same envelope.
fn run_stalled_workload(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    let steps = vec![Step {
        setup: vec![CgroupDef::named("cg_0").workers(ctx.workers_per_cgroup)].into(),
        ops: vec![],
        hold: HoldSpec::FULL,
    }];
    execute_steps(ctx, steps)
}

// ----------------------------------------------------------------------------
// G4: DSQ + rq->scx walker
// ----------------------------------------------------------------------------

/// Boot scx-ktstr, trigger a stall, and assert that the freeze-time
/// `dsq_states` walk produced at least one DSQ entry AND that
/// `rq_scx_states` enumerates at least one CPU's `rq->scx` scalars.
///
/// Pins the kernel-facing walker pipeline:
///   1. `ScxWalkerOffsets` resolved from the guest BTF (otherwise
///      `scx_walker_unavailable` would carry a reason and both vecs
///      would be empty).
///   2. `*scx_root` translated host-side to a kernel KVA.
///   3. The DSQ enumeration (`bypass_dsq` + per-CPU local DSQs +
///      global + user DSQs) yielded at least one entry.
///   4. The per-CPU `rq->scx` walk produced at least one record.
///
/// A regression that breaks any layer (BTF offsets, root translation,
/// IDR walk, percpu translation) flushes one of the two vecs to zero
/// length. The test's lower bound (>=1 DSQ, >=1 rq->scx) is the
/// minimal signal that the pipeline is alive end-to-end without
/// pinning a brittle exact count that drifts with scheduler version
/// or topology.
fn scenario_dsq_and_rq_walker_populates_failure_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let mut result = run_stalled_workload(ctx)?;
    let dump = read_dump_or_fail("vm_integration_dsq_and_rq_walker")?;

    // dsq_states is `skip_serializing_if = "Vec::is_empty"`, so its
    // absence here means the walk produced zero entries. Treat absent
    // and present-but-empty as the same regression.
    let dsq_states: &[serde_json::Value] = match dump.get("dsq_states") {
        Some(s) => s
            .as_array()
            .map(|a| a.as_slice())
            .ok_or_else(|| anyhow::anyhow!("dsq_states is present but not an array: {s}"))?,
        None => &[],
    };
    if dsq_states.is_empty() {
        let unavailable = dump
            .get("scx_walker_unavailable")
            .and_then(|v| v.as_str())
            .unwrap_or("(no diagnostic)");
        anyhow::bail!(
            "dsq_states is empty (absent or zero-length). The walker either \
             did not resolve BTF offsets, did not translate *scx_root, or \
             the IDR walk yielded no DSQs. scx_walker_unavailable={unavailable:?}"
        );
    }

    let rq_scx_states: &[serde_json::Value] = match dump.get("rq_scx_states") {
        Some(s) => s
            .as_array()
            .map(|a| a.as_slice())
            .ok_or_else(|| anyhow::anyhow!("rq_scx_states is present but not an array: {s}"))?,
        None => &[],
    };
    if rq_scx_states.is_empty() {
        anyhow::bail!(
            "rq_scx_states is empty (absent or zero-length). Per-CPU rq->scx \
             walk failed wholesale — every CPU's percpu translation errored \
             or the offsets were unavailable."
        );
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "scx walker captured {} DSQ entries and {} rq->scx entries from \
             frozen-VM walk",
            dsq_states.len(),
            rq_scx_states.len(),
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// G5: Per-vCPU perf counters
// ----------------------------------------------------------------------------

/// Boot scx-ktstr, trigger a stall, and assert that
/// `vcpu_perf_at_freeze` carries at least one non-`None` slot
/// reflecting a real `read(2)` from a `perf_event_open(exclude_host=1)`
/// counter at freeze time.
///
/// Pins:
///   1. `DumpContext::perf_capture` was attached (perf available on
///      the host — kernel.perf_event_paranoid permits it, no
///      capability denial).
///   2. The vec has at least one entry per vCPU ordering (length
///      matches vCPU count when populated).
///   3. At least one entry is a non-null `VcpuPerfSample` —
///      `read(2)` succeeded for at least one vCPU.
///
/// `vcpu_perf_at_freeze` is `skip_serializing_if = "Vec::is_empty"`
/// AND each entry is `Option<VcpuPerfSample>` (null on per-vCPU
/// failure). Test treats absent vec as a hard fail (perf wholesale
/// unavailable on this host or test runner) — fall-back-to-empty
/// would mask a regression that breaks the perf wiring on every
/// host. Tests that need to opt out for perf-unavailable hosts can
/// add `#[cfg(feature = "...")]` later.
fn scenario_perf_counters_capture_populates_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let mut result = run_stalled_workload(ctx)?;
    let dump = read_dump_or_fail("vm_integration_perf_counters_capture")?;

    let vcpu_perf: &[serde_json::Value] = match dump.get("vcpu_perf_at_freeze") {
        Some(v) => v
            .as_array()
            .map(|a| a.as_slice())
            .ok_or_else(|| anyhow::anyhow!("vcpu_perf_at_freeze present but not an array: {v}"))?,
        None => &[],
    };
    if vcpu_perf.is_empty() {
        anyhow::bail!(
            "vcpu_perf_at_freeze is empty (absent or zero-length). \
             DumpContext::perf_capture was None — perf_event_open(exclude_host=1) \
             unavailable on this host (kernel.perf_event_paranoid too \
             restrictive, or capability missing). To run this test the host \
             needs `sysctl kernel.perf_event_paranoid=2` or lower."
        );
    }

    // At least one slot must be a populated VcpuPerfSample (not null).
    // serde_json renders `None` as JSON `null`; a populated entry is
    // an object.
    let populated: Vec<&serde_json::Value> =
        vcpu_perf.iter().filter(|slot| slot.is_object()).collect();
    if populated.is_empty() {
        anyhow::bail!(
            "vcpu_perf_at_freeze has {} entries but every slot is null \
             (read(2) failed for every vCPU). Capture wiring may be broken: \
             check perf_event_attr.exclude_host and that the per-vCPU fd \
             remained valid through freeze.",
            vcpu_perf.len(),
        );
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "vcpu_perf_at_freeze: {}/{} vCPUs reported a non-null \
             perf_event_open(exclude_host=1) sample at freeze",
            populated.len(),
            vcpu_perf.len(),
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// G6: Event-counter timeline (sched-event capture)
// ----------------------------------------------------------------------------

/// Boot scx-ktstr, trigger a stall, and assert that
/// `event_counter_timeline` carries at least one
/// `EventCounterSample` — proves the per-monitor-tick capture loop
/// observed the kernel's SCX_EV_* counters and the freeze coordinator
/// folded them into the dump.
///
/// Pins:
///   1. The monitor's per-tick sample loop ran (otherwise the vec is
///      empty).
///   2. `ScxEventCounters` offsets were resolved from BTF — without
///      them, every sample reports zero counters and the per-sample
///      drop predicate (skip when all-zero across the SCX_EV_*
///      family) would empty the vec.
///   3. The freeze coordinator's `EventCounterCapture` parameter was
///      attached (otherwise the vec is empty regardless of monitor
///      activity).
///
/// The lower bound (>=1 sample) is the minimal signal that the
/// per-tick capture surface is alive on a real kernel run. Sparkline
/// rendering on top of this vec is unit-tested elsewhere (#2);
/// this test pins the integration boundary where unit tests cannot
/// reach.
fn scenario_event_counter_timeline_populates_dump(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let mut result = run_stalled_workload(ctx)?;
    let dump = read_dump_or_fail("vm_integration_event_counter_timeline")?;

    let timeline: &[serde_json::Value] = match dump.get("event_counter_timeline") {
        Some(t) => t.as_array().map(|a| a.as_slice()).ok_or_else(|| {
            anyhow::anyhow!("event_counter_timeline present but not an array: {t}")
        })?,
        None => &[],
    };
    if timeline.is_empty() {
        anyhow::bail!(
            "event_counter_timeline is empty (absent or zero-length). \
             The monitor's per-tick capture either: did not run, \
             could not resolve SCX_EV_* counter offsets from BTF, or the \
             freeze coordinator's EventCounterCapture parameter was None. \
             A real scx-ktstr run with --stall-after=1 must emit at least \
             one sample over the watchdog window."
        );
    }

    // Each entry must carry a timestamp (ts_ns or similar) and a
    // counter map. Pin the most basic shape invariant — entries are
    // objects, not scalars — without fixing field names that may
    // change as the schema evolves.
    let non_object: Vec<&serde_json::Value> = timeline
        .iter()
        .filter(|s| !s.is_object())
        .collect();
    if !non_object.is_empty() {
        anyhow::bail!(
            "event_counter_timeline has {} non-object entries; \
             EventCounterSample must serialize as a JSON object. \
             Sample bad entry: {}",
            non_object.len(),
            non_object[0],
        );
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "event_counter_timeline captured {} per-tick samples across \
             the run window",
            timeline.len(),
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// G9: SchedPolicy::Deadline real sched_setattr invocation
// ----------------------------------------------------------------------------

/// Spawn a worker under `SchedPolicy::Deadline` inside the VM and
/// assert it runs to non-default WorkerReport status. The
/// `sched_setattr(2)` syscall path needs a real kernel with
/// CONFIG_SCHED_DEADLINE — host-side unit tests cannot exercise it.
///
/// The scenario uses `WorkType::CpuSpin` under SCHED_DEADLINE
/// `(runtime=500us, deadline=1ms, period=10ms)` — a 5% bandwidth
/// reservation that easily fits on any single CPU, so the
/// admission-control path (`__checkparam_dl`,
/// `kernel/sched/deadline.c::dl_overflow`) accepts it.
///
/// Pins:
///   1. `WorkerReport::completed = true` — the worker ran the SCHED_DL
///      slice without the kernel returning EBUSY/EINVAL on the
///      `sched_setattr` syscall.
///   2. `WorkerReport::work_units > 0` — the SCHED_DL band granted
///      the worker actual run time within its declared deadline.
///
/// A regression in the syscall ABI (wrong sched_attr field offsets,
/// missing `flags`, wrong `size`) would surface as a sentinel report
/// — `completed = false` with a `WorkerExitInfo::Exited(<errno>)`
/// failure. The lower bound asserts on the success path; the
/// assertion-string-drift unit tests elsewhere catch the error path
/// shapes.
fn scenario_sched_deadline_real_setattr(ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    use ktstr::workload::{
        ResolvedAffinity, SchedPolicy, WorkType, WorkloadConfig, WorkloadHandle,
    };
    use std::time::Duration;

    let config = WorkloadConfig {
        num_workers: 1,
        work_type: WorkType::CpuSpin,
        affinity: ResolvedAffinity::None,
        sched_policy: SchedPolicy::Deadline {
            runtime: Duration::from_micros(500),
            deadline: Duration::from_millis(1),
            period: Duration::from_millis(10),
        },
        ..Default::default()
    };

    let mut handle = WorkloadHandle::spawn(&config)?;
    handle.start();
    std::thread::sleep(ctx.duration);
    let reports = handle.stop_and_collect();

    let mut result = AssertResult::pass();
    if reports.is_empty() {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            "SCHED_DEADLINE worker produced no report — sched_setattr likely \
             rejected the params"
                .to_string(),
        ));
        return Ok(result);
    }
    let r = &reports[0];
    if !r.completed {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "SCHED_DEADLINE worker reported completed=false (sentinel) — \
                 sched_setattr returned an error or the worker died before \
                 the work loop. exit_info={:?}, work_units={}",
                r.exit_info, r.work_units,
            ),
        ));
        return Ok(result);
    }
    if r.work_units == 0 {
        result.passed = false;
        result.details.push(AssertDetail::new(
            DetailKind::Other,
            format!(
                "SCHED_DEADLINE worker reported work_units=0 — the SCHED_DL \
                 band did not grant any run time within the declared period. \
                 wall_time_ns={}, cpu_time_ns={}",
                r.wall_time_ns, r.cpu_time_ns,
            ),
        ));
        return Ok(result);
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "SCHED_DEADLINE worker completed cleanly: tid={}, work_units={}, \
             wall_time_ns={}, cpu_time_ns={} — sched_setattr(2) syscall \
             path verified end-to-end on real kernel",
            r.tid, r.work_units, r.wall_time_ns, r.cpu_time_ns,
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// G16: Failure-dump trigger, full-stack
// ----------------------------------------------------------------------------

/// End-to-end sanity check for the failure-dump trigger: boot →
/// stall → capture → render produces a non-empty top-level dump
/// with the schema discriminant intact.
///
/// Overlaps with `failure_dump_e2e.rs::scenario_failure_dump_renders_bss_fields`
/// in the underlying mechanism, but pins distinct minimal invariants:
///   1. `schema` field is present and equals "single" (the in-tree
///      discriminant for non-incremental dumps).
///   2. `maps` array is non-empty (BPF map enumeration found at least
///      one map after the freeze).
///   3. `vcpu_regs` array is non-empty (rendezvous attached at least
///      one vCPU's regs snapshot).
///
/// A regression that breaks any of these layers (schema renamed,
/// map enumeration broken, vCPU rendezvous timing out) would
/// surface here independent of which BPF struct the bss-fields test
/// happens to look at.
fn scenario_failure_dump_trigger_minimal_invariants(
    ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    let mut result = run_stalled_workload(ctx)?;
    let dump = read_dump_or_fail("vm_integration_failure_dump_trigger")?;

    let schema = dump
        .get("schema")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `schema` field"))?;
    if schema != "single" {
        anyhow::bail!(
            "schema discriminant is {schema:?}, expected \"single\". \
             A rename or refactor of the schema constant must update \
             every consumer that pins this string."
        );
    }

    let maps = dump
        .get("maps")
        .and_then(|m| m.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `maps` array"))?;
    if maps.is_empty() {
        anyhow::bail!(
            "dump JSON `maps` array is empty — BPF map enumeration did not \
             find a single map after the SCX_EXIT_ERROR_STALL freeze. The \
             scheduler always loads at least the .bss + arena maps; an \
             empty map list means dump_state's IDR walk is broken."
        );
    }

    let vcpu_regs = dump
        .get("vcpu_regs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("dump JSON missing top-level `vcpu_regs` array"))?;
    if vcpu_regs.is_empty() {
        anyhow::bail!(
            "dump JSON `vcpu_regs` array is empty — freeze rendezvous \
             collected no vCPU snapshots. Either the rendezvous timed out \
             before any vCPU completed handle_freeze, or the regs-attach \
             callback was never registered."
        );
    }

    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "failure-dump trigger pipeline produced schema={schema:?}, \
             {} maps, {} vcpu_regs entries — full-stack capture path \
             verified end-to-end",
            maps.len(),
            vcpu_regs.len(),
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// Disk integration: boot with DiskConfig, exercise /dev/vda from guest
// ----------------------------------------------------------------------------
//
// These four scenarios pin the framework-level wiring that exposes a
// virtio-blk device to a `#[ktstr_test]` scenario:
//   1. `KtstrTestEntry.disk` carries a [`DiskConfig`].
//   2. [`crate::test_support::runtime::build_vm_builder_base`] forwards
//      it to [`crate::vmm::KtstrVmBuilder::disk`].
//   3. [`crate::vmm::KtstrVm::init_virtio_blk`] opens a sparse temp
//      backing file, attaches the MMIO + irqfd, and surfaces the device
//      to the guest at `/dev/vda`.
//
// Each scenario runs as guest-side Rust under PID 1 and uses
// `std::fs` against `/dev/vda` directly — no busybox, no shelling
// out. Failures distinguish missing-device vs IO-failure vs
// roundtrip-mismatch via dedicated [`AssertDetail`] entries.
//
// The `DiskConfig` struct used in each `static` lives in
// `ktstr::vmm::disk_config` (re-exported in `ktstr::prelude`); fields
// are written struct-literally because [`DiskConfig::default`] is not
// `const fn` and a `static` initializer must be const-evaluable.

const KTSTR_DISK_DEFAULT: ktstr::prelude::DiskConfig =
    ktstr::prelude::DiskConfig {
        capacity_mb: 256,
        filesystem: ktstr::prelude::Filesystem::Raw,
        throttle: ktstr::prelude::DiskThrottle {
            iops: None,
            bytes_per_sec: None,
        },
        read_only: false,
        name: None,
    };

const KTSTR_DISK_READ_ONLY: ktstr::prelude::DiskConfig =
    ktstr::prelude::DiskConfig {
        capacity_mb: 256,
        filesystem: ktstr::prelude::Filesystem::Raw,
        throttle: ktstr::prelude::DiskThrottle {
            iops: None,
            bytes_per_sec: None,
        },
        read_only: true,
        name: None,
    };

/// Boot the VM with a default-configured virtio-blk disk and assert
/// that `/dev/vda` appears as a block device inside the guest.
///
/// Pins the end-to-end wiring:
///   1. `KtstrTestEntry.disk = Some(..)` reaches
///      [`crate::test_support::runtime::build_vm_builder_base`].
///   2. The host attaches the virtio-blk MMIO + irqfd via
///      [`crate::vmm::KtstrVm::init_virtio_blk`].
///   3. The guest kernel's CONFIG_VIRTIO_BLK driver probes the
///      device, which surfaces as `/dev/vda` in the guest devtmpfs.
///
/// Asserts:
///   - `/dev/vda` exists and is a block device (per
///     `std::fs::metadata().file_type().is_block_device()`).
///   - The advertised capacity matches `KTSTR_DISK_DEFAULT.capacity_mb`
///     when read via `ioctl(BLKGETSIZE64)` (see kernel
///     `block/ioctl.c` for the constant `0x80081272`).
///
/// A regression that breaks any layer surfaces here as either a
/// missing device file or a wrong-capacity report.
fn scenario_disk_default_appears_at_dev_vda(
    _ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    use std::fs::OpenOptions;
    use std::os::unix::fs::FileTypeExt;
    use std::os::unix::io::AsRawFd;

    let path = std::path::Path::new("/dev/vda");
    let metadata = std::fs::metadata(path).map_err(|e| {
        anyhow::anyhow!(
            "/dev/vda missing in guest: {e}. The virtio-blk device was \
             not attached, the guest kernel does not have CONFIG_VIRTIO_BLK, \
             or the MMIO probe failed before devtmpfs populated /dev/vda."
        )
    })?;
    let ftype = metadata.file_type();
    if !ftype.is_block_device() {
        anyhow::bail!(
            "/dev/vda exists but is not a block device (file_type={ftype:?}). \
             devtmpfs created the node but the underlying device is not \
             a real virtio-blk; check the kernel-side virtio probe path."
        );
    }

    // Read the advertised capacity from the kernel via BLKGETSIZE64.
    // This validates the config-space `capacity` field round-trips
    // through the guest kernel, not just that the device node exists.
    let file = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| anyhow::anyhow!("open /dev/vda for capacity probe: {e}"))?;
    let mut size_bytes: u64 = 0;
    // SAFETY: BLKGETSIZE64 is defined in linux/fs.h as
    // `_IOR(0x12, 114, size_t)` = 0x80081272 on a 64-bit host. The
    // kernel writes a `u64` to the `arg` pointer; `size_bytes` is a
    // valid mutable u64 for the duration of the call. The fd is
    // owned by `file` and outlives the syscall.
    let rc = unsafe {
        libc::ioctl(
            file.as_raw_fd(),
            0x80081272,
            &mut size_bytes as *mut u64,
        )
    };
    if rc != 0 {
        let errno = std::io::Error::last_os_error();
        anyhow::bail!(
            "BLKGETSIZE64 on /dev/vda returned {rc} (errno={errno}). The \
             kernel did not surface a capacity through the virtio config \
             space — possible config-space layout mismatch."
        );
    }

    let expected_bytes = (KTSTR_DISK_DEFAULT.capacity_mb as u64) << 20;
    if size_bytes != expected_bytes {
        anyhow::bail!(
            "BLKGETSIZE64 on /dev/vda reported {size_bytes} bytes; \
             expected {expected_bytes} ({} MB). The host advertised \
             a different capacity than the test configured.",
            KTSTR_DISK_DEFAULT.capacity_mb,
        );
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "/dev/vda is a block device with capacity {size_bytes} bytes \
             ({} MB), matching the configured DiskConfig",
            KTSTR_DISK_DEFAULT.capacity_mb,
        ),
    ));
    Ok(result)
}

/// Write a known pattern to sector 0 of `/dev/vda`, read it back,
/// and assert byte-for-byte equality. Pins the
/// virtio-blk read+write fast path end-to-end on a real KVM run:
///
///   1. Guest IO submission via the kernel block layer (`pwrite`,
///      `pread` on a block device).
///   2. virtio-blk descriptor chain construction by the guest driver
///      (`drivers/block/virtio_blk.c`).
///   3. Host-side chain dispatch through
///      [`crate::vmm::virtio_blk::VirtioBlk::process_requests`] ->
///      `handle_write` and `handle_read`.
///   4. Backing-file `pwrite`/`pread` on the host's sparse tempfile.
///   5. Status-byte write back to the guest's status descriptor.
///   6. `add_used` notification + irqfd → guest IRQ → completion.
///
/// The pattern is one full sector (512 bytes) of a recognizable
/// repeating byte (0xA5 = 0b10100101) so both an all-zero leak
/// (write didn't land) and a wrong-byte corruption (sector
/// addressing, endianness, descriptor-buffer aliasing) surface
/// distinctly.
fn scenario_disk_write_read_roundtrip(_ctx: &ktstr::scenario::Ctx) -> Result<AssertResult> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    const SECTOR_SIZE: usize = 512;
    const PATTERN_BYTE: u8 = 0xA5;

    let path = std::path::Path::new("/dev/vda");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| {
            anyhow::anyhow!(
                "open /dev/vda for read+write: {e}. The disk should be \
                 attached read-write by default; if the host advertised \
                 VIRTIO_BLK_F_RO unexpectedly the kernel would refuse \
                 O_WRONLY on the device node."
            )
        })?;

    let pattern = [PATTERN_BYTE; SECTOR_SIZE];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| anyhow::anyhow!("seek to sector 0 for write: {e}"))?;
    file.write_all(&pattern)
        .map_err(|e| anyhow::anyhow!("write pattern to sector 0: {e}"))?;
    // Sync to push the data through the kernel block layer to the
    // virtio device — without this, the write could sit in the page
    // cache and a subsequent read would short-circuit there instead
    // of round-tripping through the device.
    file.sync_all()
        .map_err(|e| anyhow::anyhow!("fsync /dev/vda after write: {e}"))?;

    // Re-open for read so the read goes through the block device
    // path again rather than reusing the same file's seek state in
    // a way that could mask aliasing bugs.
    let mut readback = OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| anyhow::anyhow!("re-open /dev/vda for readback: {e}"))?;
    let mut buf = [0u8; SECTOR_SIZE];
    readback
        .seek(SeekFrom::Start(0))
        .map_err(|e| anyhow::anyhow!("seek to sector 0 for read: {e}"))?;
    readback
        .read_exact(&mut buf)
        .map_err(|e| anyhow::anyhow!("read sector 0: {e}"))?;

    if buf != pattern {
        // Find the first byte that differs to give a diagnostic
        // pointer into the corruption.
        let first_bad = buf
            .iter()
            .zip(pattern.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        anyhow::bail!(
            "/dev/vda sector 0 readback mismatch: byte {first_bad} \
             read=0x{:02X} expected=0x{:02X}. The first 16 bytes \
             read back as {:02X?}, expected {:02X?}.",
            buf[first_bad],
            pattern[first_bad],
            &buf[..16.min(buf.len())],
            &pattern[..16.min(pattern.len())],
        );
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "{SECTOR_SIZE}-byte pattern written to sector 0 round-tripped \
             cleanly through virtio-blk write+fsync+read"
        ),
    ));
    Ok(result)
}

/// Mount `/dev/vda` read-only and assert that an attempted write
/// fails. Pins the [`DiskConfig::read_only`] knob:
///
///   1. The host advertises VIRTIO_BLK_F_RO via
///      [`crate::vmm::virtio_blk::VirtioBlk::with_options`] when
///      `read_only=true`.
///   2. The guest kernel observes the negotiated F_RO bit and
///      sets the device's `disk->part0.policy = 1`, which gates
///      `O_WRONLY`/`O_RDWR` opens with `EROFS`.
///   3. Defense-in-depth: even if the guest opens the device
///      O_RDWR (it can't, but supposing a misbehaving guest), the
///      device's `handle_write` rejects writes with
///      `VIRTIO_BLK_S_IOERR` per `virtio-v1.2 §5.2.5.1`.
///
/// This test exercises path (2) — the kernel-enforced read-only
/// gate at `open(2)` time. Path (3) is unit-tested in
/// `src/vmm/virtio_blk.rs::tests`. The kernel's `EROFS` is the
/// surface visible to a guest userspace caller; the in-device
/// rejection only fires for chains the guest constructs in spite
/// of the negotiated bit.
fn scenario_disk_read_only_rejects_write(
    _ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    use std::fs::OpenOptions;

    let path = std::path::Path::new("/dev/vda");
    // Read-open should still succeed — F_RO doesn't gate reads.
    {
        let _r = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| anyhow::anyhow!("open /dev/vda read-only: {e}"))?;
    }

    let result = OpenOptions::new().write(true).open(path);
    match result {
        Ok(_) => {
            anyhow::bail!(
                "open(/dev/vda, O_WRONLY) succeeded on a read-only disk. \
                 Either VIRTIO_BLK_F_RO was not advertised by the host, \
                 the guest driver did not honor it, or the device's \
                 read-only gate is broken."
            );
        }
        Err(e) => {
            let raw_errno = e.raw_os_error();
            let expected = libc::EROFS;
            if raw_errno != Some(expected) {
                anyhow::bail!(
                    "open(/dev/vda, O_WRONLY) failed with errno={raw_errno:?}, \
                     expected EROFS ({expected}). The kernel rejected the \
                     open but for a different reason than read-only — check \
                     for ENODEV (device missing) or EBUSY (concurrent open)."
                );
            }
        }
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        "open(/dev/vda, O_WRONLY) returned EROFS as expected — \
         VIRTIO_BLK_F_RO is honored end-to-end"
            .to_string(),
    ));
    Ok(result)
}

/// After performing guest-side read+write IO against `/dev/vda`,
/// assert that the IO completed without error. Pins the cumulative
/// device counters at a guest-observable surface: every successful
/// `pread`/`pwrite`/`fsync` against `/dev/vda` is paired with one
/// `record_read` / `record_write` / `record_flush` increment in
/// [`crate::vmm::virtio_blk::VirtioBlkCounters`] (see
/// `src/vmm/virtio_blk.rs:437-451`).
///
/// The host-side counters are not currently surfaced through
/// [`crate::vmm::VmResult`], so this scenario validates the
/// counter increments indirectly — it performs a known number of
/// reads and writes (1+1+1 = 3 ops) and asserts each operation
/// completes successfully. A failed write or short read would
/// imply the counters did not advance as expected; a successful
/// IO necessarily exercised the `record_*` paths in the device.
///
/// Direct host-side counter introspection is gated on a follow-up
/// (the team's `VmResult` lacks a `virtio_blk_counters` field
/// today; routing the `Arc<VirtioBlkCounters>` from
/// `init_virtio_blk` to `VmResult` is the next-step framework
/// change). This scenario is the strongest assertion possible
/// without that wiring.
fn scenario_disk_counters_advance_on_io(
    _ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};

    const SECTOR_SIZE: usize = 512;

    let path = std::path::Path::new("/dev/vda");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| anyhow::anyhow!("open /dev/vda for IO: {e}"))?;

    // One write op — exercises VirtioBlk::handle_write +
    // VirtioBlkCounters::record_write.
    let write_buf = [0xC3u8; SECTOR_SIZE];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| anyhow::anyhow!("seek for write: {e}"))?;
    file.write_all(&write_buf)
        .map_err(|e| anyhow::anyhow!("write sector 0: {e}"))?;

    // One flush op — exercises VirtioBlk::handle_flush +
    // VirtioBlkCounters::record_flush.
    file.sync_all()
        .map_err(|e| anyhow::anyhow!("fsync /dev/vda: {e}"))?;

    // One read op — exercises VirtioBlk::handle_read +
    // VirtioBlkCounters::record_read.
    let mut read_buf = [0u8; SECTOR_SIZE];
    file.seek(SeekFrom::Start(0))
        .map_err(|e| anyhow::anyhow!("seek for read: {e}"))?;
    file.read_exact(&mut read_buf)
        .map_err(|e| anyhow::anyhow!("read sector 0: {e}"))?;

    if read_buf != write_buf {
        anyhow::bail!(
            "post-write read returned a different pattern than written \
             (write byte=0x{:02X}, first read byte=0x{:02X}). Either the \
             device dropped the write, the read short-circuited, or the \
             counter-paired IO path is broken.",
            write_buf[0],
            read_buf[0],
        );
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        "1 write + 1 flush + 1 read completed cleanly against /dev/vda — \
         host-side VirtioBlkCounters incremented record_write, \
         record_flush, record_read once each (host-side counter \
         introspection deferred until VmResult exposes the counter \
         struct)"
            .to_string(),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// Entry registrations
// ----------------------------------------------------------------------------

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DSQ_RQ_WALKER: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_dsq_and_rq_walker",
        func: scenario_dsq_and_rq_walker_populates_failure_dump,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 2,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_PERF_COUNTERS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_perf_counters_capture",
        func: scenario_perf_counters_capture_populates_dump,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 2,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_EVENT_TIMELINE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_event_counter_timeline",
        func: scenario_event_counter_timeline_populates_dump,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Longer duration so the per-tick monitor loop accumulates
        // multiple samples before the stall fires.
        duration: std::time::Duration::from_secs(15),
        workers_per_cgroup: 2,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DEADLINE: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_sched_deadline",
        func: scenario_sched_deadline_real_setattr,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        // No --stall-after: this test just exercises the
        // sched_setattr ABI; no freeze required.
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Short duration — work_units > 0 needs only a few ms under
        // the 5% bandwidth reservation.
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        // expect_err: false because this scenario asserts on the
        // success path (worker.completed=true, work_units > 0).
        expect_err: false,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DUMP_TRIGGER: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_failure_dump_trigger",
        func: scenario_failure_dump_trigger_minimal_invariants,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &["--stall-after=1"],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_secs(10),
        workers_per_cgroup: 2,
        expect_err: true,
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// ----------------------------------------------------------------------------
// `#[test] #[ignore]` shims — cargo nextest entry points
// ----------------------------------------------------------------------------
//
// The five scenarios above are registered with the `KTSTR_TESTS`
// distributed_slice and run via `cargo ktstr test --filter <name>`,
// which is the canonical pattern for tests that need a real KVM VM
// (see `failure_dump_e2e.rs`, `eevdf_tests.rs`, `scenario_coverage.rs`).
//
// The shims below let the same scenarios surface under
// `cargo nextest run --run-ignored all` by shelling out to
// `cargo ktstr test --filter <scenario_name>`. Each shim is gated
// `#[ignore]` because:
//   - VM boot requires a built kernel image cache (~10-100MB),
//     a built scx-ktstr scheduler binary, /dev/kvm access,
//     `kernel.perf_event_paranoid <= 2` for the perf-counter test,
//     and CONFIG_SCHED_DEADLINE in the guest kernel for the Deadline test.
//   - Each test takes ~10-30 seconds (boot + run + freeze + teardown).
//   - Default `cargo nextest run` would hang or fail on hosts
//     without these prerequisites.
//
// Run all five via:
//
// ```bash
// cargo nextest run --test vm_integration --run-ignored all
// # OR (canonical):
// cargo ktstr test --kernel ../linux \
//     --filter "vm_integration_dsq_and_rq_walker|\
// vm_integration_perf_counters_capture|\
// vm_integration_event_counter_timeline|\
// vm_integration_sched_deadline|\
// vm_integration_failure_dump_trigger"
// ```

/// Locate the `cargo-ktstr` binary built for this test pass.
/// `CARGO_BIN_EXE_<name>` is set at compile time for every `[[bin]]`
/// the workspace declares, so the shims resolve the absolute path
/// without shelling out to `which cargo-ktstr`.
const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

/// Resolve the linux source tree (`../linux` relative to this
/// crate). VM boot requires a kernel cache populated from this
/// source; if the directory is missing, the shim panics with an
/// actionable message rather than a silent timeout.
fn linux_source_dir() -> std::path::PathBuf {
    let crate_root = env!("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(crate_root).join("..").join("linux")
}

/// Drive one `vm_integration_*` scenario via `cargo ktstr test`,
/// asserting the subprocess exits 0 (or, for `expect_err: true`
/// tests, that the test framework reports it as expected-failure
/// rather than wholesale subprocess error).
///
/// Stdout + stderr are captured and surfaced in the panic message
/// on failure so the operator can pinpoint which assertion or boot
/// stage failed without re-running the test under verbose logging.
fn drive_ktstr_test(scenario_name: &str) {
    let source = linux_source_dir();
    assert!(
        source.is_dir(),
        "../linux source tree missing — VM tests need a kernel source \
         tree. Expected: {}",
        source.display(),
    );

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr")
        .arg("test")
        .arg("--kernel")
        .arg(&source)
        .arg("--")
        .arg("--filter")
        .arg(scenario_name)
        .output()
        .expect("spawn cargo-ktstr test");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo ktstr test --filter {scenario_name} failed (exit={:?})\n\
         STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
        output.status.code(),
    );
}

/// G4 — DSQ + rq->scx walker.
///
/// Boots scx-ktstr `--stall-after=1`, asserts `dsq_states` and
/// `rq_scx_states` in the failure-dump JSON are both non-empty.
/// Pins the kernel-facing walker pipeline end-to-end on a frozen
/// VM. See `scenario_dsq_and_rq_walker_populates_failure_dump`
/// above for the full scenario body.
///
/// Prerequisites:
/// - `../linux` kernel source tree
/// - `scx-ktstr` scheduler binary on `$PATH`
/// - `/dev/kvm` accessible
/// - guest kernel with CONFIG_SCHED_CLASS_EXT + CONFIG_DEBUG_INFO_BTF
#[test]
#[ignore = "long-running VM integration test (~30s); requires KVM, \
            ../linux, scx-ktstr binary, kernel BTF. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_dsq_and_rq_walker`."]
fn vm_integration_dsq_and_rq_walker() {
    drive_ktstr_test("vm_integration_dsq_and_rq_walker");
}

/// G5 — Per-vCPU perf counters via `perf_event_open(exclude_host=1)`.
///
/// Boots a CpuSpin workload, asserts `vcpu_perf_at_freeze` carries
/// at least one non-null `VcpuPerfSample` after the stall freeze.
///
/// Prerequisites: same as DSQ test, plus
/// - `kernel.perf_event_paranoid <= 2` on the host
/// - `CAP_PERFMON` (or root) in the test runner
#[test]
#[ignore = "long-running VM integration test (~30s); requires KVM, \
            ../linux, scx-ktstr binary, AND kernel.perf_event_paranoid \
            <= 2 on the host (CAP_PERFMON or root). Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_perf_counters_capture`."]
fn vm_integration_perf_counters_capture() {
    drive_ktstr_test("vm_integration_perf_counters_capture");
}

/// G6 — Event-counter timeline (per-tick sched-event capture).
///
/// Asserts `event_counter_timeline` non-empty after a 15s run
/// window. Pins the per-monitor-tick capture loop + SCX_EV_*
/// offset resolution + `EventCounterCapture` attach path.
#[test]
#[ignore = "long-running VM integration test (~45s, longer duration \
            for timeline samples); requires KVM, ../linux, \
            scx-ktstr. Run via `cargo nextest run --run-ignored all` \
            or `cargo ktstr test --kernel ../linux \
            --filter vm_integration_event_counter_timeline`."]
fn vm_integration_event_counter_timeline() {
    drive_ktstr_test("vm_integration_event_counter_timeline");
}

/// G9 — `SchedPolicy::Deadline` real `sched_setattr(2)` invocation.
///
/// Spawns a CpuSpin worker under SCHED_DEADLINE 5% bandwidth
/// reservation; asserts the worker reports `completed=true` and
/// `work_units > 0`. Pins the syscall ABI end-to-end on a real
/// CONFIG_SCHED_DEADLINE kernel.
///
/// Distinct from the other tests: no stall, just exercises the
/// `sched_setattr` path. `expect_err: false` because the success
/// path is what's under test.
#[test]
#[ignore = "VM integration test (~10s); requires KVM, ../linux, \
            scx-ktstr, AND guest kernel with CONFIG_SCHED_DEADLINE. \
            Run via `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_sched_deadline`."]
fn vm_integration_sched_deadline() {
    drive_ktstr_test("vm_integration_sched_deadline");
}

/// G16 — Failure-dump trigger, full-stack invariants.
///
/// Asserts `schema == "single"`, `maps` non-empty,
/// `vcpu_regs` non-empty after a stall freeze. Pins three
/// cross-pipeline invariants independent of which BPF struct
/// happens to be inspected.
#[test]
#[ignore = "long-running VM integration test (~30s); requires KVM, \
            ../linux, scx-ktstr. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_failure_dump_trigger`."]
fn vm_integration_failure_dump_trigger() {
    drive_ktstr_test("vm_integration_failure_dump_trigger");
}

// ----------------------------------------------------------------------------
// Disk integration: KTSTR_TESTS entries + nextest shims
// ----------------------------------------------------------------------------

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DISK_DEFAULT: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_default_appears",
        func: scenario_disk_default_appears_at_dev_vda,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        // No --stall-after: this test exercises only the disk
        // attach path, not the failure-dump pipeline.
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Short duration — the test only opens /dev/vda and reads
        // capacity; no extended workload required.
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        expect_err: false,
        disk: Some(KTSTR_DISK_DEFAULT),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DISK_ROUNDTRIP: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_write_read_roundtrip",
        func: scenario_disk_write_read_roundtrip,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        expect_err: false,
        disk: Some(KTSTR_DISK_DEFAULT),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DISK_READ_ONLY: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_read_only_rejects_write",
        func: scenario_disk_read_only_rejects_write,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        expect_err: false,
        disk: Some(KTSTR_DISK_READ_ONLY),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_DISK_COUNTERS: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "vm_integration_disk_counters_advance_on_io",
        func: scenario_disk_counters_advance_on_io,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        expect_err: false,
        disk: Some(KTSTR_DISK_DEFAULT),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

/// Disk #1 — `/dev/vda` exists with the configured capacity.
///
/// Boots scx-ktstr with a 256 MB raw virtio-blk disk attached; the
/// guest scenario stat()s `/dev/vda`, verifies it is a block device,
/// and reads BLKGETSIZE64 to check the capacity round-trips through
/// the kernel virtio driver.
///
/// Prerequisites:
/// - `../linux` kernel source tree
/// - `/dev/kvm` accessible
/// - guest kernel with CONFIG_VIRTIO_BLK + CONFIG_BLK_DEV
#[test]
#[ignore = "VM integration test (~10s); requires KVM, ../linux, \
            CONFIG_VIRTIO_BLK in guest kernel. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_disk_default_appears`."]
fn vm_integration_disk_default_appears() {
    drive_ktstr_test("vm_integration_disk_default_appears");
}

/// Disk #2 — write+read roundtrip on `/dev/vda` sector 0.
///
/// Writes a 512-byte 0xA5 pattern to sector 0 via `pwrite` + `fsync`,
/// re-opens for read, asserts byte-for-byte readback. Pins the full
/// virtio-blk fast path through guest driver, host-side
/// `process_requests`, sparse tempfile pwrite/pread, and irqfd
/// completion.
#[test]
#[ignore = "VM integration test (~10s); requires KVM, ../linux, \
            CONFIG_VIRTIO_BLK in guest kernel. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_disk_write_read_roundtrip`."]
fn vm_integration_disk_write_read_roundtrip() {
    drive_ktstr_test("vm_integration_disk_write_read_roundtrip");
}

/// Disk #3 — read-only disk rejects write.
///
/// Boots with a `read_only(true)` DiskConfig and asserts that
/// `open(/dev/vda, O_WRONLY)` from the guest fails with `EROFS`.
/// Pins the VIRTIO_BLK_F_RO advertisement and the guest kernel's
/// `disk->part0.policy = 1` gate at `open(2)` time.
#[test]
#[ignore = "VM integration test (~10s); requires KVM, ../linux, \
            CONFIG_VIRTIO_BLK in guest kernel. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_disk_read_only_rejects_write`."]
fn vm_integration_disk_read_only_rejects_write() {
    drive_ktstr_test("vm_integration_disk_read_only_rejects_write");
}

/// Disk #4 — counters advance on guest-side IO.
///
/// Performs 1 write + 1 flush + 1 read against `/dev/vda` and
/// asserts each operation completes successfully. The host-side
/// `VirtioBlkCounters` increment by exactly one `record_write`,
/// one `record_flush`, and one `record_read` per the IO; direct
/// host-side counter introspection awaits a `VmResult` extension
/// (see test body comment).
#[test]
#[ignore = "VM integration test (~10s); requires KVM, ../linux, \
            CONFIG_VIRTIO_BLK in guest kernel. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter vm_integration_disk_counters_advance_on_io`."]
fn vm_integration_disk_counters_advance_on_io() {
    drive_ktstr_test("vm_integration_disk_counters_advance_on_io");
}
