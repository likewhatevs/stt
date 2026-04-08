//! Composable ops/steps system for dynamic cgroup topology changes.
//!
//! [`Op`] is an atomic cgroup operation. [`Step`] sequences ops with a
//! hold period. [`CgroupDef`] bundles create + cpuset + spawn into a
//! single declaration. [`execute_steps()`] runs a step sequence with
//! scheduler liveness checks and stimulus event recording.
//!
//! See the [Ops and Steps](https://sched-ext.github.io/scx/stt/concepts/ops.html)
//! chapter for a guide.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::ops::RangeInclusive;
use std::thread;
use std::time::Duration;

use anyhow::Result;

use crate::assert::{self, AssertResult};
use crate::vmm::shm_ring::{self, StimulusPayload};
use crate::workload::{WorkType, WorkloadConfig, WorkloadHandle};

use super::{CgroupGroup, Ctx, process_alive};

// ---------------------------------------------------------------------------
// Op / CpusetSpec
// ---------------------------------------------------------------------------

/// Atomic operation on the cgroup topology.
///
/// Names use `Cow<'static, str>` so ops can reference compile-time
/// literals (zero-cost) or runtime-generated strings (owned).
pub enum Op {
    AddCgroup {
        name: Cow<'static, str>,
    },
    RemoveCgroup {
        name: Cow<'static, str>,
    },
    SetCpuset {
        cgroup: Cow<'static, str>,
        cpus: CpusetSpec,
    },
    ClearCpuset {
        cgroup: Cow<'static, str>,
    },
    SwapCpusets {
        a: Cow<'static, str>,
        b: Cow<'static, str>,
    },
    Spawn {
        cgroup: Cow<'static, str>,
        workload: WorkloadConfig,
    },
    StopCgroup {
        cgroup: Cow<'static, str>,
    },
    RandomizeAffinity {
        cgroup: Cow<'static, str>,
    },
    /// Set all workers in a cgroup to the given affinity mask.
    SetAffinity {
        cgroup: Cow<'static, str>,
        cpus: BTreeSet<usize>,
    },
    /// Spawn workers in the parent cgroup (not in a managed cgroup).
    SpawnHost {
        workload: WorkloadConfig,
    },
    /// Move all tasks from one cgroup to another via cgroup.procs.
    MoveAllTasks {
        from: Cow<'static, str>,
        to: Cow<'static, str>,
    },
    /// Move the first `count` tasks from one cgroup to another.
    MoveTasks {
        from: Cow<'static, str>,
        to: Cow<'static, str>,
        count: usize,
    },
}

/// How to compute a cpuset from topology.
#[derive(Clone)]
pub enum CpusetSpec {
    /// All CPUs in a given LLC index.
    Llc(usize),
    /// Fractional range of usable CPUs [start_frac..end_frac).
    Range { start_frac: f64, end_frac: f64 },
    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    Disjoint { index: usize, of: usize },
    /// Like Disjoint but each set overlaps neighbors by `frac` of its size.
    Overlap { index: usize, of: usize, frac: f64 },
    /// Exact CPU set (no topology resolution).
    Exact(BTreeSet<usize>),
}

// ---------------------------------------------------------------------------
// CgroupDef
// ---------------------------------------------------------------------------

/// Declarative cgroup definition: name + cpuset + workload.
///
/// Bundles the three ops that always go together (AddCgroup + SetCpuset +
/// Spawn) into a single value. The executor creates the cgroup, optionally
/// sets its cpuset, spawns workers, and moves them into the cgroup.
///
/// ```
/// # use stt::scenario::ops::{CgroupDef, CpusetSpec};
/// # use stt::workload::WorkType;
/// let def = CgroupDef::named("workers")
///     .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
///     .workers(4)
///     .work_type(WorkType::CpuSpin);
///
/// assert_eq!(def.name, "workers");
/// assert_eq!(def.num_workers, 4);
/// ```
#[derive(Clone)]
pub struct CgroupDef {
    pub name: Cow<'static, str>,
    pub cpuset: Option<CpusetSpec>,
    pub num_workers: usize,
    pub work_type: WorkType,
    pub sched_policy: crate::workload::SchedPolicy,
    /// When true, the gauntlet work_type override replaces this def's work_type.
    pub swappable: bool,
}

impl CgroupDef {
    /// Create a CgroupDef with defaults (CpuSpin, Normal, 0 workers).
    /// 0 workers means use ctx.workers_per_cgroup at execution time.
    pub fn named(name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the cpuset for this cgroup. Use when defining cgroups in step
    /// setup (initial topology). For mid-run cpuset changes, use [`Op::SetCpuset`].
    pub fn with_cpuset(mut self, cpus: CpusetSpec) -> Self {
        self.cpuset = Some(cpus);
        self
    }

    /// Set the number of workers. 0 means use `ctx.workers_per_cgroup`.
    pub fn workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the work type for workers in this cgroup.
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.work_type = wt;
        self
    }

    /// Set the Linux scheduling policy for workers.
    pub fn sched_policy(mut self, p: crate::workload::SchedPolicy) -> Self {
        self.sched_policy = p;
        self
    }

    /// When true, the gauntlet work_type override replaces this def's work type.
    pub fn swappable(mut self, swappable: bool) -> Self {
        self.swappable = swappable;
        self
    }
}

impl Default for CgroupDef {
    fn default() -> Self {
        Self {
            name: Cow::Borrowed("cg_0"),
            cpuset: None,
            num_workers: 0,
            work_type: WorkType::CpuSpin,
            sched_policy: crate::workload::SchedPolicy::Normal,
            swappable: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Step / HoldSpec
// ---------------------------------------------------------------------------

/// How to produce the CgroupDefs for a step's setup phase.
pub enum Setup {
    /// Static list of cgroup definitions.
    Defs(Vec<CgroupDef>),
    /// Factory that generates definitions from the runtime context.
    Factory(fn(&Ctx) -> Vec<CgroupDef>),
}

impl Setup {
    fn resolve(&self, ctx: &Ctx) -> Vec<CgroupDef> {
        match self {
            Setup::Defs(defs) => defs.clone(),
            Setup::Factory(f) => f(ctx),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Setup::Defs(defs) => defs.is_empty(),
            Setup::Factory(_) => false,
        }
    }
}

impl From<Vec<CgroupDef>> for Setup {
    fn from(defs: Vec<CgroupDef>) -> Self {
        Setup::Defs(defs)
    }
}

/// A sequence of ops followed by a hold period.
///
/// `setup` cgroups are created, configured, and populated before `ops`
/// are applied. Use `Step::new` to create a step with only ops (no setup).
pub struct Step {
    pub setup: Setup,
    pub ops: Vec<Op>,
    pub hold: HoldSpec,
}

impl Step {
    /// Create a step with ops only (no CgroupDef setup).
    pub fn new(ops: Vec<Op>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(vec![]),
            ops,
            hold,
        }
    }
}

/// How long to hold after a step's ops are applied.
pub enum HoldSpec {
    /// Fraction of the total scenario duration.
    Frac(f64),
    /// Fixed duration.
    Fixed(Duration),
    /// Repeat the step's ops in a loop at the given interval until the
    /// remaining scenario time is exhausted.
    Loop { interval: Duration },
}

impl Op {
    /// Return a unique bit index for each Op variant (for op_kinds bitmask).
    fn discriminant(&self) -> u32 {
        match self {
            Op::AddCgroup { .. } => 0,
            Op::RemoveCgroup { .. } => 1,
            Op::SetCpuset { .. } => 2,
            Op::ClearCpuset { .. } => 3,
            Op::SwapCpusets { .. } => 4,
            Op::Spawn { .. } => 5,
            Op::StopCgroup { .. } => 6,
            Op::RandomizeAffinity { .. } => 7,
            Op::SetAffinity { .. } => 8,
            Op::SpawnHost { .. } => 9,
            Op::MoveAllTasks { .. } => 10,
            Op::MoveTasks { .. } => 11,
        }
    }
}

// ---------------------------------------------------------------------------
// SHM writer for stimulus events
// ---------------------------------------------------------------------------

/// SHM ring writer for guest-to-host data transfer.
///
/// Prefers mmap of /dev/mem for zero-copy access. Falls back to
/// pread/pwrite when mmap of the E820 gap fails (common on kernels
/// that restrict mmap of non-RAM physical ranges).
enum ShmWriter {
    /// mmap succeeded — direct pointer access.
    Mapped {
        ptr: *mut u8,
        map_base: *mut libc::c_void,
        map_size: usize,
        shm_size: usize,
    },
    /// mmap failed — use pread/pwrite on the /dev/mem fd.
    Fd {
        fd: std::fs::File,
        shm_base: u64,
        shm_size: usize,
    },
}

impl ShmWriter {
    /// Try to open the SHM region. Returns None if SHM params are absent
    /// from /proc/cmdline or /dev/mem cannot be opened.
    fn try_open() -> Option<Self> {
        let cmdline = std::fs::read_to_string("/proc/cmdline").ok()?;
        let (shm_base, shm_size) = shm_ring::parse_shm_params_from_str(&cmdline)?;

        use std::fs::OpenOptions;
        use std::os::unix::fs::OpenOptionsExt;

        let fd = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_SYNC)
            .open("/dev/mem")
            .ok()?;

        match shm_ring::mmap_devmem(
            std::os::unix::io::AsRawFd::as_raw_fd(&fd),
            shm_base,
            shm_size,
        ) {
            Some(m) => Some(ShmWriter::Mapped {
                ptr: m.ptr,
                map_base: m.map_base,
                map_size: m.map_size,
                shm_size: shm_size as usize,
            }),
            None => {
                eprintln!(
                    "stt: SHM mmap failed ({}), using pread/pwrite fallback",
                    std::io::Error::last_os_error(),
                );
                Some(ShmWriter::Fd {
                    fd,
                    shm_base,
                    shm_size: shm_size as usize,
                })
            }
        }
    }

    /// Write a TLV message to the SHM ring.
    fn write(&self, msg_type: u32, payload: &[u8]) {
        match self {
            ShmWriter::Mapped { ptr, shm_size, .. } => {
                let buf = unsafe { std::slice::from_raw_parts_mut(*ptr, *shm_size) };
                shm_ring::shm_write(buf, 0, msg_type, payload);
            }
            ShmWriter::Fd {
                fd,
                shm_base,
                shm_size,
            } => {
                use std::os::unix::io::AsRawFd;

                // Read current SHM state, apply the ring write, write back.
                let mut buf = vec![0u8; *shm_size];
                let n = unsafe {
                    libc::pread(
                        fd.as_raw_fd(),
                        buf.as_mut_ptr() as *mut libc::c_void,
                        buf.len(),
                        *shm_base as libc::off_t,
                    )
                };
                if n < 0 {
                    return;
                }

                shm_ring::shm_write(&mut buf, 0, msg_type, payload);

                unsafe {
                    libc::pwrite(
                        fd.as_raw_fd(),
                        buf.as_ptr() as *const libc::c_void,
                        buf.len(),
                        *shm_base as libc::off_t,
                    );
                }
            }
        }
    }
}

impl Drop for ShmWriter {
    fn drop(&mut self) {
        if let ShmWriter::Mapped {
            map_base, map_size, ..
        } = self
        {
            unsafe {
                libc::munmap(*map_base, *map_size);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CpusetSpec resolution
// ---------------------------------------------------------------------------

impl CpusetSpec {
    /// Resolve to a concrete CPU set given the topology.
    pub fn resolve(&self, ctx: &Ctx) -> BTreeSet<usize> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) => ctx.topo.llc_aligned_cpuset(*idx),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } => {
                let start = (usable.len() as f64 * start_frac) as usize;
                let end = (usable.len() as f64 * end_frac) as usize;
                usable[start.min(usable.len())..end.min(usable.len())]
                    .iter()
                    .copied()
                    .collect()
            }
            CpusetSpec::Disjoint { index, of } => {
                let chunk = usable.len() / of;
                let start = index * chunk;
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    (index + 1) * chunk
                };
                usable[start..end].iter().copied().collect()
            }
            CpusetSpec::Overlap { index, of, frac } => {
                let chunk = usable.len() / of;
                let overlap = (chunk as f64 * frac) as usize;
                let start = if *index == 0 {
                    0
                } else {
                    (index * chunk).saturating_sub(overlap)
                };
                let end = if *index == of - 1 {
                    usable.len()
                } else {
                    ((index + 1) * chunk + overlap).min(usable.len())
                };
                usable[start..end].iter().copied().collect()
            }
            CpusetSpec::Exact(cpus) => cpus.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Step executor
// ---------------------------------------------------------------------------

/// State tracked across step execution.
struct StepState<'a> {
    /// RAII cgroup guard — removes cgroups on drop.
    cgroups: CgroupGroup<'a>,
    /// Active workload handles keyed by cgroup name.
    handles: Vec<(String, WorkloadHandle)>,
}

/// Execute a sequence of steps against the given context.
///
/// Convenience wrapper around [`execute_steps_with`] that passes
/// `None` for checks, falling back to `assert_not_starved`. Use
/// [`execute_steps_with`] when you need custom thresholds.
pub fn execute_steps(ctx: &Ctx, steps: Vec<Step>) -> Result<AssertResult> {
    execute_steps_with(ctx, steps, None)
}

/// Execute steps with an explicit [`Assert`](crate::assert::Assert) for
/// worker checks. When `checks` is `Some`, it overrides `ctx.assert`.
/// When `None`, uses `ctx.assert` (the merged three-layer config).
pub fn execute_steps_with(
    ctx: &Ctx,
    steps: Vec<Step>,
    checks: Option<&crate::assert::Assert>,
) -> Result<AssertResult> {
    let effective_checks = checks.unwrap_or(&ctx.assert);
    let mut state = StepState {
        cgroups: CgroupGroup::new(ctx.cgroups),
        handles: Vec::new(),
    };

    // Open SHM once for the entire step sequence. No-op outside a VM.
    let shm = ShmWriter::try_open();

    let scenario_start = std::time::Instant::now();

    // ScenarioStart marker.
    if let Some(ref w) = shm {
        w.write(shm_ring::MSG_TYPE_SCENARIO_START, &[]);
    }

    // When a host-side BPF map write is configured, wait for the host
    // to complete the write before starting the workload.
    if ctx.wait_for_map_write {
        match shm_ring::wait_for(0, std::time::Duration::from_secs(10)) {
            Ok(()) => {
                // Brief delay for the crash trigger to propagate.
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(e) => {
                eprintln!("stt: signal slot 0 wait failed: {e} — proceeding without sync");
            }
        }
    }

    for (step_idx, step) in steps.iter().enumerate() {
        // Check scheduler liveness between steps (skip before first).
        if step_idx > 0 && !process_alive(ctx.sched_pid) {
            let mut r = collect_result(&mut state, effective_checks);
            r.passed = false;
            r.details.push("scheduler died between steps".into());
            return Ok(r);
        }

        match &step.hold {
            HoldSpec::Loop { interval } => {
                // Setup runs once before the loop.
                if !step.setup.is_empty() {
                    let defs = step.setup.resolve(ctx);
                    apply_setup(ctx, &mut state, &defs)?;
                }
                // Loop mode: apply ops repeatedly at interval until
                // the remaining scenario time is exhausted.
                let deadline = scenario_start + ctx.duration;
                while std::time::Instant::now() < deadline {
                    apply_ops(ctx, &mut state, &step.ops)?;
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    thread::sleep(remaining.min(*interval));
                }
            }
            _ => {
                // Ops first (e.g. parent cgroup creation), then
                // CgroupDef setup (children with workers).
                apply_ops(ctx, &mut state, &step.ops)?;
                if !step.setup.is_empty() {
                    let defs = step.setup.resolve(ctx);
                    apply_setup(ctx, &mut state, &defs)?;
                }

                // Write stimulus event after applying ops.
                if let Some(ref w) = shm {
                    let payload = build_stimulus(&scenario_start, step_idx, &step.ops, &state);
                    w.write(
                        shm_ring::MSG_TYPE_STIMULUS,
                        zerocopy::IntoBytes::as_bytes(&payload),
                    );
                }

                let hold_dur = match &step.hold {
                    HoldSpec::Frac(f) => Duration::from_secs_f64(ctx.duration.as_secs_f64() * f),
                    HoldSpec::Fixed(d) => *d,
                    HoldSpec::Loop { .. } => unreachable!(),
                };
                thread::sleep(hold_dur);
            }
        }
    }

    // ScenarioEnd marker.
    if let Some(ref w) = shm {
        let elapsed = scenario_start.elapsed().as_millis() as u32;
        w.write(shm_ring::MSG_TYPE_SCENARIO_END, &elapsed.to_ne_bytes());
    }

    // Final liveness check.
    let sched_dead = !process_alive(ctx.sched_pid);

    let mut result = collect_result(&mut state, effective_checks);

    if sched_dead {
        result.passed = false;
        result.details.push("scheduler died".into());
    }

    Ok(result)
}

/// Build a StimulusPayload from the current step state.
fn build_stimulus(
    scenario_start: &std::time::Instant,
    step_idx: usize,
    ops: &[Op],
    state: &StepState<'_>,
) -> StimulusPayload {
    let mut op_kinds: u32 = 0;
    for op in ops {
        op_kinds |= 1 << op.discriminant();
    }

    let total_iterations: u64 = state
        .handles
        .iter()
        .flat_map(|(_, h)| h.snapshot_iterations())
        .sum();

    StimulusPayload {
        elapsed_ms: scenario_start.elapsed().as_millis() as u32,
        step_index: step_idx as u16,
        op_count: ops.len() as u16,
        op_kinds,
        cgroup_count: state.cgroups.names().len() as u16,
        worker_count: state.handles.len() as u16,
        total_iterations,
    }
}

/// Create cgroups, set cpusets, and spawn workers from CgroupDefs.
fn apply_setup(ctx: &Ctx, state: &mut StepState<'_>, defs: &[CgroupDef]) -> Result<()> {
    use crate::workload::AffinityMode;

    for def in defs {
        state.cgroups.add_cgroup_no_cpuset(&def.name)?;
        if let Some(ref cpuset_spec) = def.cpuset {
            let resolved = cpuset_spec.resolve(ctx);
            ctx.cgroups.set_cpuset(&def.name, &resolved)?;
        }
        let n = if def.num_workers == 0 {
            ctx.workers_per_cgroup
        } else {
            def.num_workers
        };
        let effective_work_type = if def.swappable
            && let Some(override_wt) = ctx.work_type_override
        {
            // Skip grouped-worker overrides when num_workers is not divisible.
            if let Some(gs) = override_wt.worker_group_size()
                && n % gs != 0
            {
                def.work_type
            } else {
                override_wt
            }
        } else {
            def.work_type
        };
        let wl = WorkloadConfig {
            num_workers: n,
            affinity: AffinityMode::None,
            work_type: effective_work_type,
            sched_policy: def.sched_policy,
        };
        let mut h = WorkloadHandle::spawn(&wl)?;
        ctx.cgroups.move_tasks(&def.name, &h.tids())?;
        h.start();
        state.handles.push((def.name.to_string(), h));
    }
    Ok(())
}

/// Apply a slice of Ops to the running state.
fn apply_ops(ctx: &Ctx, state: &mut StepState<'_>, ops: &[Op]) -> Result<()> {
    for op in ops {
        match op {
            Op::AddCgroup { name } => {
                state.cgroups.add_cgroup_no_cpuset(name)?;
            }
            Op::RemoveCgroup { name } => {
                // Stop workers in this cgroup first.
                state.handles.retain(|(n, _)| n.as_str() != *name);
                let _ = ctx.cgroups.remove_cgroup(name);
            }
            Op::SetCpuset { cgroup, cpus } => {
                let resolved = cpus.resolve(ctx);
                ctx.cgroups.set_cpuset(cgroup, &resolved)?;
            }
            Op::ClearCpuset { cgroup } => {
                ctx.cgroups.clear_cpuset(cgroup)?;
            }
            Op::SwapCpusets { a, b } => {
                // Read current cpusets from the cgroup filesystem, swap them.
                let cpus_a = read_cpuset(ctx, a);
                let cpus_b = read_cpuset(ctx, b);
                if let Some(ref ca) = cpus_a {
                    ctx.cgroups.set_cpuset(b, ca)?;
                }
                if let Some(ref cb) = cpus_b {
                    ctx.cgroups.set_cpuset(a, cb)?;
                }
            }
            Op::Spawn { cgroup, workload } => {
                let mut h = WorkloadHandle::spawn(workload)?;
                ctx.cgroups.move_tasks(cgroup, &h.tids())?;
                h.start();
                state.handles.push((cgroup.to_string(), h));
            }
            Op::StopCgroup { cgroup } => {
                state.handles.retain(|(n, _)| n.as_str() != *cgroup);
            }
            Op::RandomizeAffinity { cgroup } => {
                for (name, handle) in &state.handles {
                    if name.as_str() == *cgroup {
                        let all = ctx.topo.all_cpus();
                        let pool: BTreeSet<usize> = all.iter().copied().collect();
                        for idx in 0..handle.tids().len() {
                            let count = (pool.len() / 2).max(1);
                            use rand::seq::IndexedRandom;
                            let v: Vec<usize> = pool.iter().copied().collect();
                            let chosen: BTreeSet<usize> =
                                v.sample(&mut rand::rng(), count).copied().collect();
                            let _ = handle.set_affinity(idx, &chosen);
                        }
                    }
                }
            }
            Op::SetAffinity { cgroup, cpus } => {
                for (name, handle) in &state.handles {
                    if name.as_str() == *cgroup {
                        for idx in 0..handle.tids().len() {
                            let _ = handle.set_affinity(idx, cpus);
                        }
                    }
                }
            }
            Op::SpawnHost { workload } => {
                let mut h = WorkloadHandle::spawn(workload)?;
                h.start();
                // Empty string key: workers in parent cgroup, not a managed cgroup.
                state.handles.push((String::new(), h));
            }
            Op::MoveAllTasks { from, to } => {
                for (name, handle) in &mut state.handles {
                    if name.as_str() == *from {
                        for tid in handle.tids() {
                            let _ = ctx.cgroups.move_task(to, tid);
                        }
                        *name = to.to_string();
                    }
                }
            }
            Op::MoveTasks { from, to, count } => {
                let mut moved = 0usize;
                for (name, handle) in &mut state.handles {
                    if name.as_str() != *from || moved >= *count {
                        continue;
                    }
                    let tids = handle.tids();
                    let n = (*count - moved).min(tids.len());
                    for &tid in &tids[..n] {
                        let _ = ctx.cgroups.move_task(to, tid);
                    }
                    moved += n;
                }
            }
        }
    }
    Ok(())
}

/// Read the effective cpuset for a cgroup by reading cpuset.cpus.
fn read_cpuset(ctx: &Ctx, name: &str) -> Option<BTreeSet<usize>> {
    let path = ctx.cgroups.parent_path().join(name).join("cpuset.cpus");
    let content = std::fs::read_to_string(&path).ok()?;
    let content = content.trim();
    if content.is_empty() {
        return None;
    }
    let mut cpus = BTreeSet::new();
    for part in content.split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                cpus.extend(lo..=hi);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.insert(cpu);
        }
    }
    Some(cpus)
}

/// Collect all worker results and produce an AssertResult.
///
/// Uses `checks` for worker evaluation. When the Assert has no
/// worker-level checks configured (all fields None), falls back
/// to `assert_not_starved`.
fn collect_result(state: &mut StepState<'_>, checks: &crate::assert::Assert) -> AssertResult {
    let mut result = AssertResult::pass();
    let handles = std::mem::take(&mut state.handles);
    for (_name, h) in handles {
        let reports = h.stop_and_collect();
        if checks.has_worker_checks() {
            result.merge(checks.assert_cgroup(&reports, None));
        } else {
            result.merge(assert::assert_not_starved(&reports));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Traverse combinator
// ---------------------------------------------------------------------------

/// Layout strategy for Traverse phases.
pub enum Layout {
    Disjoint,
    /// Overlapping cpusets. (min_frac, max_frac) — PRNG picks a value in range.
    Overlap(f64, f64),
}

/// Generates a random walk of cgroup topology changes across phases.
///
/// Each phase picks a random (cgroup_count, layout) pair, generates SetCpuset
/// ops, spawns workers in new cgroups, and holds for phase_duration.
///
/// `persistent_cgroups` cgroups are created in phase 0 and never removed.
/// Only cgroups at index >= `persistent_cgroups` are added/removed by the
/// random walk. The `cgroup_count` range applies to the total cgroup count
/// (persistent + ephemeral).
///
/// `cgroup_workloads` controls the workload for each cgroup index. If the
/// vec has fewer entries than the cgroup index, the last entry repeats.
pub struct Traverse {
    pub seed: Option<u64>,
    pub cgroup_count: RangeInclusive<usize>,
    pub layouts: Vec<Layout>,
    pub phases: usize,
    pub phase_duration: Duration,
    pub settle: Duration,
    /// Cgroups [0..persistent_cgroups) are created once and never removed.
    pub persistent_cgroups: usize,
    /// Workload config per cgroup index. Last entry repeats for higher indices.
    pub cgroup_workloads: Vec<WorkloadConfig>,
}

impl Traverse {
    /// Generate a `Vec<Step>` from the Traverse configuration.
    pub fn generate(&self, ctx: &Ctx) -> Vec<Step> {
        use rand::RngExt;

        let seed = self.seed.unwrap_or_else(|| std::process::id() as u64);
        let mut rng = seeded_rng(seed);

        let usable_len = ctx.topo.usable_cpus().len();
        let max_cgroups = (*self.cgroup_count.end()).min(usable_len / 2).max(1);
        let min_cgroups = (*self.cgroup_count.start()).max(1).min(max_cgroups);

        let mut steps = Vec::with_capacity(self.phases + 1);
        let mut live_cgroups: Vec<Cow<'static, str>> = Vec::new();

        let names: Vec<Cow<'static, str>> = (0..max_cgroups)
            .map(|i| Cow::Owned(format!("cg_{i}")))
            .collect();

        for phase in 0..self.phases {
            let range = max_cgroups - min_cgroups + 1;
            let target_count = min_cgroups + rng.random_range(0..range);
            let layout_idx = rng.random_range(0..self.layouts.len());
            let layout = &self.layouts[layout_idx];

            let mut ops = Vec::new();

            // Add cgroups if needed.
            while live_cgroups.len() < target_count {
                let idx = live_cgroups.len();
                let name = names[idx].clone();
                let wl = self
                    .cgroup_workloads
                    .get(idx)
                    .or(self.cgroup_workloads.last())
                    .cloned()
                    .unwrap_or_default();
                ops.push(Op::AddCgroup { name: name.clone() });
                ops.push(Op::Spawn {
                    cgroup: name.clone(),
                    workload: wl,
                });
                live_cgroups.push(name);
            }

            // Remove cgroups if needed (never remove persistent cgroups).
            while live_cgroups.len() > target_count && live_cgroups.len() > self.persistent_cgroups
            {
                if let Some(name) = live_cgroups.pop() {
                    ops.push(Op::StopCgroup {
                        cgroup: name.clone(),
                    });
                    ops.push(Op::RemoveCgroup { name });
                }
            }

            // Apply cpuset layout.
            for (i, name) in live_cgroups.iter().enumerate() {
                let spec = match layout {
                    Layout::Disjoint => CpusetSpec::Disjoint {
                        index: i,
                        of: live_cgroups.len(),
                    },
                    Layout::Overlap(min_frac, max_frac) => {
                        let frac = min_frac
                            + rng.random_range(0..100) as f64 / 100.0 * (max_frac - min_frac);
                        CpusetSpec::Overlap {
                            index: i,
                            of: live_cgroups.len(),
                            frac,
                        }
                    }
                };
                ops.push(Op::SetCpuset {
                    cgroup: name.clone(),
                    cpus: spec,
                });
            }

            let hold = if phase == 0 {
                // First phase includes settle time.
                HoldSpec::Fixed(self.settle + self.phase_duration)
            } else {
                HoldSpec::Fixed(self.phase_duration)
            };

            steps.push(Step {
                setup: vec![].into(),
                ops,
                hold,
            });
        }

        steps
    }
}

/// Seeded PRNG for deterministic topology generation.
///
/// Callers seed with a scenario-specific value so gauntlet runs are
/// deterministic across reruns.
fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vmm::shm_ring::parse_shm_params_from_str;

    // -- Op discriminant tests --

    #[test]
    fn op_discriminant_unique() {
        let ops: Vec<Op> = vec![
            Op::AddCgroup { name: "a".into() },
            Op::RemoveCgroup { name: "a".into() },
            Op::SetCpuset {
                cgroup: "a".into(),
                cpus: CpusetSpec::Exact(Default::default()),
            },
            Op::ClearCpuset { cgroup: "a".into() },
            Op::SwapCpusets {
                a: "a".into(),
                b: "b".into(),
            },
            Op::Spawn {
                cgroup: "a".into(),
                workload: Default::default(),
            },
            Op::StopCgroup { cgroup: "a".into() },
            Op::RandomizeAffinity { cgroup: "a".into() },
            Op::SetAffinity {
                cgroup: "a".into(),
                cpus: Default::default(),
            },
            Op::SpawnHost {
                workload: Default::default(),
            },
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into(),
            },
            Op::MoveTasks {
                from: "a".into(),
                to: "b".into(),
                count: 1,
            },
        ];
        let mut seen = std::collections::BTreeSet::new();
        for op in &ops {
            assert!(seen.insert(op.discriminant()), "duplicate discriminant");
        }
    }

    #[test]
    fn op_discriminant_values() {
        assert_eq!(Op::AddCgroup { name: "a".into() }.discriminant(), 0);
        assert_eq!(Op::RemoveCgroup { name: "a".into() }.discriminant(), 1);
        assert_eq!(
            Op::SpawnHost {
                workload: Default::default()
            }
            .discriminant(),
            9
        );
        assert_eq!(
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into()
            }
            .discriminant(),
            10
        );
        assert_eq!(
            Op::MoveTasks {
                from: "a".into(),
                to: "b".into(),
                count: 1
            }
            .discriminant(),
            11
        );
    }

    // -- seeded_rng tests --

    #[test]
    fn seeded_rng_deterministic() {
        use rand::RngExt;
        let mut rng1 = seeded_rng(42);
        let mut rng2 = seeded_rng(42);
        for _ in 0..100 {
            assert_eq!(rng1.random::<u64>(), rng2.random::<u64>());
        }
    }

    #[test]
    fn seeded_rng_different_seeds_differ() {
        use rand::RngExt;
        let mut rng1 = seeded_rng(1);
        let mut rng2 = seeded_rng(2);
        let same = (0..10).all(|_| rng1.random::<u64>() == rng2.random::<u64>());
        assert!(!same);
    }

    // -- HoldSpec variants --

    #[test]
    fn holdspec_frac() {
        let step = Step::new(vec![], HoldSpec::Frac(0.5));
        match step.hold {
            HoldSpec::Frac(f) => assert!((f - 0.5).abs() < f64::EPSILON),
            _ => panic!("expected Frac"),
        }
    }

    #[test]
    fn holdspec_fixed() {
        let step = Step::new(vec![], HoldSpec::Fixed(Duration::from_secs(3)));
        match step.hold {
            HoldSpec::Fixed(d) => assert_eq!(d, Duration::from_secs(3)),
            _ => panic!("expected Fixed"),
        }
    }

    #[test]
    fn holdspec_loop() {
        let step = Step::new(
            vec![],
            HoldSpec::Loop {
                interval: Duration::from_millis(100),
            },
        );
        match step.hold {
            HoldSpec::Loop { interval } => assert_eq!(interval, Duration::from_millis(100)),
            _ => panic!("expected Loop"),
        }
    }

    // -- CpusetSpec::Exact --

    #[test]
    fn cpusetspec_exact_is_passthrough() {
        let cpus: BTreeSet<usize> = [0, 2, 4].iter().copied().collect();
        let spec = CpusetSpec::Exact(cpus.clone());
        let topo = crate::topology::TestTopology::from_spec(1, 4, 1);
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &cgroups,
            topo: &topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: 0,
            settle_ms: 1000,
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        };
        let resolved = spec.resolve(&ctx);
        assert_eq!(resolved, cpus);
    }

    // -- parse_shm_params_from_str (from shm_ring) --

    #[test]
    fn ops_parse_shm_params_valid() {
        let cmdline = "console=ttyS0 STT_SHM_BASE=0xfc000000 STT_SHM_SIZE=0x10000 quiet";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x10000);
    }

    #[test]
    fn ops_parse_shm_params_missing() {
        assert!(parse_shm_params_from_str("console=ttyS0 quiet").is_none());
    }

    // -- CpusetSpec resolution helpers --

    fn make_ctx(
        sockets: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_spec(sockets, cores, threads);
        (cgroups, topo)
    }

    fn ctx_from<'a>(
        cgroups: &'a crate::cgroup::CgroupManager,
        topo: &'a crate::topology::TestTopology,
    ) -> Ctx<'a> {
        Ctx {
            cgroups,
            topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: 0,
            settle_ms: 0,
            work_type_override: None,
            assert: crate::assert::Assert::default_checks(),
            wait_for_map_write: false,
        }
    }

    // -- CpusetSpec::Disjoint --

    #[test]
    fn cpusetspec_disjoint_two_partitions() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Disjoint { index: 0, of: 2 }.resolve(&ctx);
        let b = CpusetSpec::Disjoint { index: 1, of: 2 }.resolve(&ctx);
        // Partitions must be disjoint.
        assert!(a.is_disjoint(&b), "partitions overlap: {:?} vs {:?}", a, b);
        // Together they cover all usable CPUs.
        let usable: BTreeSet<usize> = ctx.topo.usable_cpus().iter().copied().collect();
        let union: BTreeSet<usize> = a.union(&b).copied().collect();
        assert_eq!(union, usable);
    }

    #[test]
    fn cpusetspec_disjoint_remainder_to_last() {
        // 7 usable CPUs / 3 partitions = chunk=2, so partition 0=[0,1], 1=[2,3], 2=[4,5,6].
        // Last partition gets the remainder.
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let usable_len = ctx.topo.usable_cpus().len();
        let c = CpusetSpec::Disjoint { index: 2, of: 3 }.resolve(&ctx);
        let chunk = usable_len / 3;
        // Last partition should be >= chunk size (gets remainder).
        assert!(
            c.len() >= chunk,
            "last partition {}: expected >= {}",
            c.len(),
            chunk
        );
    }

    #[test]
    fn cpusetspec_disjoint_single_partition() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let all = CpusetSpec::Disjoint { index: 0, of: 1 }.resolve(&ctx);
        let usable: BTreeSet<usize> = ctx.topo.usable_cpus().iter().copied().collect();
        assert_eq!(all, usable);
    }

    // -- CpusetSpec::Range --

    #[test]
    fn cpusetspec_range_first_half() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        let expected_len = usable.len() / 2;
        assert_eq!(cpus.len(), expected_len);
        // Should contain the first usable CPUs.
        for &cpu in &cpus {
            assert!(usable.contains(&cpu));
        }
    }

    #[test]
    fn cpusetspec_range_second_half() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 1.0,
        }
        .resolve(&ctx);
        assert!(a.is_disjoint(&b));
    }

    #[test]
    fn cpusetspec_range_full() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 1.0,
        }
        .resolve(&ctx);
        let usable: BTreeSet<usize> = ctx.topo.usable_cpus().iter().copied().collect();
        assert_eq!(cpus, usable);
    }

    #[test]
    fn cpusetspec_range_empty() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 0.5,
        }
        .resolve(&ctx);
        assert!(cpus.is_empty());
    }

    #[test]
    fn cpusetspec_range_clamps_to_bounds() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        // end_frac > 1.0 should be clamped to usable.len().
        let cpus = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: 2.0,
        }
        .resolve(&ctx);
        let usable: BTreeSet<usize> = ctx.topo.usable_cpus().iter().copied().collect();
        assert_eq!(cpus, usable);
    }

    // -- CpusetSpec::Overlap --

    #[test]
    fn cpusetspec_overlap_neighbors_share_cpus() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.5,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Overlap {
            index: 1,
            of: 2,
            frac: 0.5,
        }
        .resolve(&ctx);
        let shared: BTreeSet<usize> = a.intersection(&b).copied().collect();
        assert!(!shared.is_empty(), "overlap=0.5 should produce shared CPUs");
    }

    #[test]
    fn cpusetspec_overlap_zero_frac_is_disjoint() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let a = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 0.0,
        }
        .resolve(&ctx);
        let b = CpusetSpec::Overlap {
            index: 1,
            of: 2,
            frac: 0.0,
        }
        .resolve(&ctx);
        assert!(a.is_disjoint(&b), "frac=0 should be disjoint");
    }

    #[test]
    fn cpusetspec_overlap_last_partition_covers_tail() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let last = CpusetSpec::Overlap {
            index: 2,
            of: 3,
            frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        // Last partition should include the last usable CPU.
        assert!(last.contains(usable.last().unwrap()));
    }

    #[test]
    fn cpusetspec_overlap_first_partition_starts_at_zero() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let first = CpusetSpec::Overlap {
            index: 0,
            of: 3,
            frac: 0.5,
        }
        .resolve(&ctx);
        let usable = ctx.topo.usable_cpus();
        assert!(first.contains(&usable[0]));
    }

    // -- CpusetSpec::Llc --

    #[test]
    fn cpusetspec_llc_index_zero() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Llc(0).resolve(&ctx);
        assert!(!cpus.is_empty());
        // All CPUs in the set should belong to LLC 0.
        let llc0 = ctx.topo.llc_aligned_cpuset(0);
        assert_eq!(cpus, llc0);
    }

    #[test]
    fn cpusetspec_llc_two_sockets_disjoint() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let llc0 = CpusetSpec::Llc(0).resolve(&ctx);
        let llc1 = CpusetSpec::Llc(1).resolve(&ctx);
        assert!(llc0.is_disjoint(&llc1), "LLCs should be disjoint");
    }

    // -- Traverse::generate --

    #[test]
    fn traverse_generate_produces_steps() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(42),
            cgroup_count: 2..=4,
            layouts: vec![Layout::Disjoint],
            phases: 3,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 0,
            cgroup_workloads: vec![WorkloadConfig::default()],
        };
        let steps = t.generate(&ctx);
        assert_eq!(steps.len(), 3);
        for step in &steps {
            assert!(!step.ops.is_empty(), "each phase should have ops");
        }
    }

    #[test]
    fn traverse_generate_deterministic() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(99),
            cgroup_count: 2..=4,
            layouts: vec![Layout::Disjoint, Layout::Overlap(0.2, 0.5)],
            phases: 5,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 1,
            cgroup_workloads: vec![WorkloadConfig::default()],
        };
        let steps1 = t.generate(&ctx);
        let steps2 = t.generate(&ctx);
        assert_eq!(steps1.len(), steps2.len());
        for (s1, s2) in steps1.iter().zip(steps2.iter()) {
            assert_eq!(
                s1.ops.len(),
                s2.ops.len(),
                "deterministic seed should produce same ops"
            );
        }
    }

    #[test]
    fn traverse_generate_persistent_cgroups_preserved() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let t = Traverse {
            seed: Some(42),
            cgroup_count: 1..=4,
            layouts: vec![Layout::Disjoint],
            phases: 5,
            phase_duration: Duration::from_millis(100),
            settle: Duration::from_millis(50),
            persistent_cgroups: 2,
            cgroup_workloads: vec![WorkloadConfig::default()],
        };
        let steps = t.generate(&ctx);
        // Every phase should have at least persistent_cgroups worth of SetCpuset ops
        // (cg_0, cg_1 are never removed).
        for step in &steps {
            let remove_ops: Vec<&Op> = step.ops.iter()
                .filter(|op| matches!(op, Op::RemoveCgroup { name } if name == "cg_0" || name == "cg_1"))
                .collect();
            assert!(
                remove_ops.is_empty(),
                "persistent cgroups should never be removed"
            );
        }
    }

    // -- CgroupDef builder --

    #[test]
    fn cgroup_def_builder_chain() {
        let d = CgroupDef::named("test")
            .with_cpuset(CpusetSpec::Llc(0))
            .workers(8)
            .work_type(WorkType::Bursty {
                burst_ms: 50,
                sleep_ms: 100,
            })
            .sched_policy(crate::workload::SchedPolicy::Batch)
            .swappable(true);
        assert_eq!(d.name, "test");
        assert!(d.cpuset.is_some());
        assert_eq!(d.num_workers, 8);
        assert!(d.swappable);
    }

    #[test]
    fn cgroup_def_default() {
        let d = CgroupDef::default();
        assert_eq!(d.name, "cg_0");
        assert!(d.cpuset.is_none());
        assert_eq!(d.num_workers, 0);
        assert!(!d.swappable);
    }

    // -- Setup --

    #[test]
    fn setup_defs_resolves() {
        let defs = vec![CgroupDef::named("a"), CgroupDef::named("b")];
        let setup = Setup::Defs(defs);
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let resolved = setup.resolve(&ctx);
        assert_eq!(resolved.len(), 2);
        assert!(!setup.is_empty());
    }

    #[test]
    fn setup_defs_empty() {
        let setup = Setup::Defs(vec![]);
        assert!(setup.is_empty());
    }

    #[test]
    fn setup_factory_not_empty() {
        let setup = Setup::Factory(|_| vec![CgroupDef::named("generated")]);
        assert!(!setup.is_empty());
    }
}
