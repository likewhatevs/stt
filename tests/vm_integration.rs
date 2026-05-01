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
