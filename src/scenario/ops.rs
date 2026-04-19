//! Composable ops/steps system for dynamic cgroup topology changes.
//!
//! [`Op`] is an atomic cgroup operation. [`Step`] sequences ops with a
//! hold period. [`CgroupDef`] bundles create + cpuset + spawn into a
//! single declaration. [`execute_steps()`] runs a step sequence with
//! scheduler liveness checks and stimulus event recording.
//!
//! See the [Ops and Steps](https://likewhatevs.github.io/ktstr/guide/concepts/ops.html)
//! chapter for a guide.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::thread;
use std::time::Duration;

use anyhow::Result;

use crate::assert::AssertResult;
use crate::vmm::shm_ring::{self, StimulusPayload};
use crate::workload::{AffinityKind, MemPolicy, Work, WorkType, WorkloadConfig, WorkloadHandle};

use super::{CgroupGroup, Ctx, process_alive};

// ---------------------------------------------------------------------------
// Op / CpusetSpec
// ---------------------------------------------------------------------------

/// Atomic operation on the cgroup topology.
///
/// Names use `Cow<'static, str>` so ops can reference compile-time
/// literals (zero-cost) or runtime-generated strings (owned).
#[derive(Clone, Debug)]
pub enum Op {
    /// Create a new cgroup under the managed cgroup parent.
    AddCgroup { name: Cow<'static, str> },
    /// Remove a cgroup (stops its workers first).
    RemoveCgroup { cgroup: Cow<'static, str> },
    /// Set a cgroup's cpuset to the resolved CPU set.
    SetCpuset {
        cgroup: Cow<'static, str>,
        cpus: CpusetSpec,
    },
    /// Clear a cgroup's cpuset (allow all CPUs).
    ClearCpuset { cgroup: Cow<'static, str> },
    /// Read both cgroups' cpusets and swap them.
    SwapCpusets {
        a: Cow<'static, str>,
        b: Cow<'static, str>,
    },
    /// Spawn workers and move them into the target cgroup.
    ///
    /// The work type is used as-is; gauntlet `work_type_override` does
    /// not apply. Use [`CgroupDef`] with `swappable(true)` when the
    /// work type should be overridable.
    Spawn {
        cgroup: Cow<'static, str>,
        work: Work,
    },
    /// Stop all workers in a cgroup (does not remove the cgroup).
    StopCgroup { cgroup: Cow<'static, str> },
    /// Set worker affinity in a cgroup. Resolved at apply time via
    /// [`resolve_affinity_for_cgroup()`](super::resolve_affinity_for_cgroup).
    SetAffinity {
        cgroup: Cow<'static, str>,
        affinity: AffinityKind,
    },
    /// Spawn workers in the parent cgroup (not in a managed cgroup).
    ///
    /// `Work` is resolved to a `WorkloadConfig` at apply time, matching
    /// the resolution pattern used by `Op::Spawn`.
    SpawnHost { work: Work },
    /// Move all tasks from one cgroup to another.
    ///
    /// Each task is moved via `cgroup.procs`. If any move fails, the
    /// error propagates and handle name keys are left unchanged (workers
    /// remain addressed under `from`). On success, handle name keys are
    /// updated to `to` so subsequent ops address the moved workers.
    MoveAllTasks {
        from: Cow<'static, str>,
        to: Cow<'static, str>,
    },
}

/// How to compute a cpuset from topology.
#[derive(Clone, Debug)]
pub enum CpusetSpec {
    /// All CPUs in a given LLC index.
    Llc(usize),
    /// All CPUs in a given NUMA node index.
    Numa(usize),
    /// Fractional range of usable CPUs [start_frac..end_frac).
    Range { start_frac: f64, end_frac: f64 },
    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    Disjoint { index: usize, of: usize },
    /// Like Disjoint but each set overlaps neighbors by `frac` of its size.
    Overlap { index: usize, of: usize, frac: f64 },
    /// Exact CPU set (no topology resolution).
    Exact(BTreeSet<usize>),
}

impl CpusetSpec {
    /// Construct an `Exact` cpuset from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        CpusetSpec::Exact(cpus.into_iter().collect())
    }

    /// Partition usable CPUs into `of` equal disjoint sets; take the `index`-th.
    pub fn disjoint(index: usize, of: usize) -> Self {
        CpusetSpec::Disjoint { index, of }
    }

    /// Like [`disjoint`](Self::disjoint) but each set overlaps neighbors by `frac` of its size.
    pub fn overlap(index: usize, of: usize, frac: f64) -> Self {
        CpusetSpec::Overlap { index, of, frac }
    }

    /// Fractional range of usable CPUs `[start_frac..end_frac)`.
    pub fn range(start_frac: f64, end_frac: f64) -> Self {
        CpusetSpec::Range {
            start_frac,
            end_frac,
        }
    }

    /// All CPUs in a given LLC index.
    pub fn llc(index: usize) -> Self {
        CpusetSpec::Llc(index)
    }

    /// All CPUs in a given NUMA node index.
    pub fn numa(index: usize) -> Self {
        CpusetSpec::Numa(index)
    }
}

// ---------------------------------------------------------------------------
// CgroupDef
// ---------------------------------------------------------------------------

/// Declarative cgroup definition: name + cpuset + workload(s).
///
/// Bundles the ops that always go together (AddCgroup + SetCpuset +
/// Spawn) into a single value. The executor creates the cgroup, optionally
/// sets its cpuset, spawns workers for each [`Work`] entry, and moves
/// them into the cgroup.
///
/// Multiple [`Work`] entries run in parallel within the cgroup. Each
/// entry spawns its own set of worker processes.
///
/// Use `CgroupDef` in `Step::with_defs` for scenarios where cgroups are
/// created once and run for the step duration. Use `Op::AddCgroup` +
/// `Op::Spawn` directly when you need mid-step cgroup creation, removal,
/// or other dynamic operations between spawn and collect.
///
/// ```
/// # use ktstr::scenario::ops::{CgroupDef, CpusetSpec};
/// # use ktstr::workload::{Work, WorkType};
/// // Single work group via convenience methods.
/// let def = CgroupDef::named("workers")
///     .with_cpuset(CpusetSpec::disjoint(0, 2))
///     .workers(4)
///     .work_type(WorkType::CpuSpin);
///
/// assert_eq!(def.name, "workers");
/// assert_eq!(def.works[0].num_workers, Some(4));
///
/// // Multiple concurrent work groups via .work().
/// let def = CgroupDef::named("mixed")
///     .work(Work::default().workers(4).work_type(WorkType::CpuSpin))
///     .work(Work::default().workers(2).work_type(WorkType::YieldHeavy));
///
/// assert_eq!(def.works.len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct CgroupDef {
    /// Cgroup name relative to the scenario's parent cgroup. Must be a
    /// valid cgroupfs filename.
    pub name: Cow<'static, str>,
    /// Optional cpuset assignment. `None` inherits the parent cgroup's
    /// cpuset (typically the scenario's usable CPU set).
    pub cpuset: Option<CpusetSpec>,
    /// Work groups to spawn. Empty means use a single default Work
    /// (CpuSpin, Normal, ctx.workers_per_cgroup workers).
    pub works: Vec<Work>,
    /// When true, the gauntlet work_type override replaces each Work's
    /// work_type (applied per-Work via resolve_work_type).
    pub swappable: bool,
}

impl CgroupDef {
    /// Create a CgroupDef with defaults (empty works, no cpuset).
    /// Empty works means use a single default Work at execution time.
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

    /// Add a work group. Can be called multiple times for concurrent
    /// work groups within this cgroup.
    pub fn work(mut self, w: Work) -> Self {
        self.works.push(w);
        self
    }

    /// Ensure works[0] exists for single-Work builder methods.
    fn ensure_default_work(&mut self) {
        if self.works.is_empty() {
            self.works.push(Work::default());
        }
    }

    /// Set the number of workers (convenience for single Work).
    pub fn workers(mut self, n: usize) -> Self {
        self.ensure_default_work();
        self.works[0].num_workers = Some(n);
        self
    }

    /// Set the work type (convenience for single Work).
    pub fn work_type(mut self, wt: WorkType) -> Self {
        self.ensure_default_work();
        self.works[0].work_type = wt;
        self
    }

    /// Set the scheduling policy (convenience for single Work).
    pub fn sched_policy(mut self, p: crate::workload::SchedPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].sched_policy = p;
        self
    }

    /// Set the per-worker affinity (convenience for single Work).
    pub fn affinity(mut self, a: crate::workload::AffinityKind) -> Self {
        self.ensure_default_work();
        self.works[0].affinity = a;
        self
    }

    /// Set the NUMA memory placement policy (convenience for single Work).
    pub fn mem_policy(mut self, p: crate::workload::MemPolicy) -> Self {
        self.ensure_default_work();
        self.works[0].mem_policy = p;
        self
    }

    /// Set the NUMA memory policy mode flags (convenience for single Work).
    pub fn mpol_flags(mut self, f: crate::workload::MpolFlags) -> Self {
        self.ensure_default_work();
        self.works[0].mpol_flags = f;
        self
    }

    /// When true, the gauntlet work_type override replaces each Work's work type.
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
            works: vec![],
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

impl Clone for Setup {
    fn clone(&self) -> Self {
        match self {
            Setup::Defs(defs) => Setup::Defs(defs.clone()),
            Setup::Factory(f) => Setup::Factory(*f),
        }
    }
}

impl std::fmt::Debug for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Setup::Defs(defs) => f.debug_tuple("Defs").field(defs).finish(),
            Setup::Factory(_) => f
                .debug_tuple("Factory")
                .field(&"fn(&Ctx) -> Vec<CgroupDef>")
                .finish(),
        }
    }
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
/// For non-`Loop` steps, `ops` are applied first, then `setup` cgroups
/// are created, configured, and populated. For `Loop` steps, `setup`
/// runs once before the ops loop. Use `Step::new` to create a step
/// with only ops (no setup).
#[derive(Clone, Debug)]
pub struct Step {
    /// Cgroup setup applied before (non-`Loop`) or once above (`Loop`)
    /// the ops list. Runtime cgroups are spawned from this spec.
    pub setup: Setup,
    /// Ordered operations applied each time the step body runs:
    /// cpuset edits, task moves, spawn/despawn, etc.
    pub ops: Vec<Op>,
    /// How long, and whether to loop, after the ops finish one pass.
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

    /// Create a step with CgroupDef setup and a hold period.
    ///
    /// Most steps only need cgroup definitions and a hold duration.
    /// Use [`with_ops`](Step::with_ops) to chain ops onto the step.
    pub fn with_defs(defs: Vec<CgroupDef>, hold: HoldSpec) -> Self {
        Self {
            setup: Setup::Defs(defs),
            ops: vec![],
            hold,
        }
    }

    /// Replace the ops for a step, consuming and returning it.
    pub fn with_ops(mut self, ops: Vec<Op>) -> Self {
        self.ops = ops;
        self
    }
}

/// How a step advances after its ops are applied. `Frac` and `Fixed`
/// hold for a duration; `Loop` repeatedly re-applies `Step::ops` at a
/// fixed interval instead of holding.
#[derive(Clone, Debug)]
pub enum HoldSpec {
    /// Fraction of the total scenario duration.
    Frac(f64),
    /// Fixed duration.
    Fixed(Duration),
    /// Repeat the step's ops in a loop at the given interval until the
    /// remaining scenario time is exhausted.
    Loop { interval: Duration },
}

impl HoldSpec {
    /// Hold for the full scenario duration (`Frac(1.0)`).
    pub const FULL: HoldSpec = HoldSpec::Frac(1.0);
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
            Op::SetAffinity { .. } => 7,
            Op::SpawnHost { .. } => 8,
            Op::MoveAllTasks { .. } => 9,
        }
    }

    /// Create a new cgroup.
    pub fn add_cgroup(name: impl Into<Cow<'static, str>>) -> Self {
        Op::AddCgroup { name: name.into() }
    }

    /// Remove a cgroup (stops its workers first).
    pub fn remove_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::RemoveCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set a cgroup's cpuset.
    pub fn set_cpuset(cgroup: impl Into<Cow<'static, str>>, cpus: CpusetSpec) -> Self {
        Op::SetCpuset {
            cgroup: cgroup.into(),
            cpus,
        }
    }

    /// Clear a cgroup's cpuset (allow all CPUs).
    pub fn clear_cpuset(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::ClearCpuset {
            cgroup: cgroup.into(),
        }
    }

    /// Swap cpusets between two cgroups.
    pub fn swap_cpusets(a: impl Into<Cow<'static, str>>, b: impl Into<Cow<'static, str>>) -> Self {
        Op::SwapCpusets {
            a: a.into(),
            b: b.into(),
        }
    }

    /// Spawn workers in a cgroup.
    pub fn spawn(cgroup: impl Into<Cow<'static, str>>, work: Work) -> Self {
        Op::Spawn {
            cgroup: cgroup.into(),
            work,
        }
    }

    /// Stop all workers in a cgroup.
    pub fn stop_cgroup(cgroup: impl Into<Cow<'static, str>>) -> Self {
        Op::StopCgroup {
            cgroup: cgroup.into(),
        }
    }

    /// Set worker affinity in a cgroup.
    pub fn set_affinity(cgroup: impl Into<Cow<'static, str>>, affinity: AffinityKind) -> Self {
        Op::SetAffinity {
            cgroup: cgroup.into(),
            affinity,
        }
    }

    /// Spawn workers in the parent cgroup.
    pub fn spawn_host(work: Work) -> Self {
        Op::SpawnHost { work }
    }

    /// Move all tasks from one cgroup to another.
    pub fn move_all_tasks(
        from: impl Into<Cow<'static, str>>,
        to: impl Into<Cow<'static, str>>,
    ) -> Self {
        Op::MoveAllTasks {
            from: from.into(),
            to: to.into(),
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
                    "ktstr: SHM mmap failed ({}), using pread/pwrite fallback",
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
    ///
    /// Acquires `SHM_WRITE_LOCK` to serialize against concurrent writers
    /// (sched-exit-mon thread via `write_msg`).
    fn write(&self, msg_type: u32, payload: &[u8]) {
        let _guard = shm_ring::SHM_WRITE_LOCK.lock();
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
    /// Check whether this spec can produce a non-empty cpuset for the
    /// given topology. Returns `Err` with a human-readable reason on
    /// failure.
    pub fn validate(&self, ctx: &Ctx) -> std::result::Result<(), String> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) if *idx >= ctx.topo.num_llcs() => Err(format!(
                "Llc({idx}) out of range: topology has {} LLCs",
                ctx.topo.num_llcs()
            )),
            CpusetSpec::Numa(node) if *node >= ctx.topo.num_numa_nodes() => Err(format!(
                "Numa({node}) out of range: topology has {} NUMA nodes",
                ctx.topo.num_numa_nodes()
            )),
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. } if *of == 0 => {
                Err("partition count (of) must be > 0".into())
            }
            CpusetSpec::Disjoint { index, of, .. } | CpusetSpec::Overlap { index, of, .. }
                if *index >= *of =>
            {
                Err(format!("index {index} >= partition count {of}"))
            }
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if !start_frac.is_finite() || !end_frac.is_finite() => Err(format!(
                "Range start_frac ({start_frac}) or end_frac ({end_frac}) is not finite"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if *start_frac < 0.0 || *end_frac > 1.0 => Err(format!(
                "Range fracs must lie in [0.0, 1.0]: start_frac={start_frac}, end_frac={end_frac}"
            )),
            CpusetSpec::Range {
                start_frac,
                end_frac,
            } if start_frac >= end_frac => Err(format!(
                "Range start_frac ({start_frac}) >= end_frac ({end_frac})"
            )),
            CpusetSpec::Overlap { frac, .. } if !frac.is_finite() => {
                Err(format!("Overlap frac ({frac}) is not finite"))
            }
            CpusetSpec::Overlap { frac, .. } if *frac < 0.0 || *frac > 1.0 => {
                Err(format!("Overlap frac ({frac}) must lie in [0.0, 1.0]"))
            }
            CpusetSpec::Disjoint { of, .. } | CpusetSpec::Overlap { of, .. }
                if usable.len() < *of =>
            {
                Err(format!(
                    "not enough usable CPUs ({}) for {} partitions",
                    usable.len(),
                    of
                ))
            }
            _ => Ok(()),
        }
    }

    /// Resolve to a concrete CPU set given the topology.
    ///
    /// **Callers MUST run [`validate`] first and propagate its error.**
    /// `apply_setup` and `apply_ops::SetCpuset` do so via `anyhow::bail!`.
    /// Among rejected inputs, `Disjoint`/`Overlap` with `of == 0` and
    /// inverted `Range` fracs panic here (div-by-zero and slice OOB);
    /// non-finite fracs and out-of-bounds `Overlap.frac` may produce
    /// silently-wrong cpusets or panics depending on the value (e.g.
    /// NaN saturates to 0, inverting a Range and triggering a slice
    /// OOB). Validate rejects all of these before resolve runs.
    ///
    /// Out-of-range `Llc` and `Numa` indices are defensively clamped
    /// to the largest valid index with a `tracing::warn!` rather than
    /// panicking, so a late-bound topology mismatch (e.g. a scenario
    /// authored against 4 LLCs run on a 2-LLC host after validate has
    /// been skipped) degrades into a usable cpuset instead of a crash.
    /// This defense-in-depth is for the Llc/Numa variants only; the
    /// frac/partition variants rely on `validate`.
    pub fn resolve(&self, ctx: &Ctx) -> BTreeSet<usize> {
        let usable = ctx.topo.usable_cpus();
        match self {
            CpusetSpec::Llc(idx) => {
                if *idx >= ctx.topo.num_llcs() {
                    // Graceful fallback: clamp to last LLC instead of panicking.
                    let clamped = ctx.topo.num_llcs().saturating_sub(1);
                    tracing::warn!(
                        llc_idx = idx,
                        num_llcs = ctx.topo.num_llcs(),
                        clamped,
                        "CpusetSpec::Llc index out of range, clamping",
                    );
                    ctx.topo.llc_aligned_cpuset(clamped)
                } else {
                    ctx.topo.llc_aligned_cpuset(*idx)
                }
            }
            CpusetSpec::Numa(idx) => {
                if *idx >= ctx.topo.num_numa_nodes() {
                    let clamped = ctx.topo.num_numa_nodes().saturating_sub(1);
                    tracing::warn!(
                        numa_node = idx,
                        num_numa_nodes = ctx.topo.num_numa_nodes(),
                        clamped,
                        "CpusetSpec::Numa index out of range, clamping",
                    );
                    ctx.topo.numa_aligned_cpuset(clamped)
                } else {
                    ctx.topo.numa_aligned_cpuset(*idx)
                }
            }
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
    /// Resolved cpusets per cgroup name, for isolation checks.
    cpusets: std::collections::HashMap<String, BTreeSet<usize>>,
}

/// Execute a single step with CgroupDefs that hold for the full duration.
///
/// Convenience wrapper around [`execute_steps`] for the common pattern
/// of creating cgroups and running them for [`HoldSpec::FULL`].
pub fn execute_defs(ctx: &Ctx, defs: Vec<CgroupDef>) -> Result<AssertResult> {
    execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])
}

/// Execute a sequence of steps against the given context.
///
/// Convenience wrapper around [`execute_steps_with`] that passes
/// `None` for checks, falling back to `ctx.assert`. Use
/// [`execute_steps_with`] when you need to override `ctx.assert`.
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
        cpusets: std::collections::HashMap::new(),
    };

    // Open SHM once for the entire step sequence. No-op outside a VM.
    let shm = ShmWriter::try_open();

    let scenario_start = std::time::Instant::now();

    // ScenarioStart marker.
    if let Some(ref w) = shm {
        w.write(shm_ring::MSG_TYPE_SCENARIO_START, &[]);
    }

    // When a host-side BPF map write is configured, signal the host that
    // probes are attached and the scenario is starting, then wait for the
    // host to complete the write before starting the workload.
    if ctx.wait_for_map_write {
        shm_ring::signal_value(1, shm_ring::SIGNAL_PROBES_READY);
        match shm_ring::wait_for(0, std::time::Duration::from_secs(10)) {
            Ok(()) => {
                // Brief delay for the crash trigger to propagate.
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
            Err(e) => {
                eprintln!("ktstr: signal slot 0 wait failed: {e} — proceeding without sync");
            }
        }
    }

    for (step_idx, step) in steps.iter().enumerate() {
        // Check scheduler liveness between steps (skip before first).
        if step_idx > 0 && !process_alive(ctx.sched_pid) {
            let mut r = collect_result(&mut state, effective_checks, ctx.topo);
            r.passed = false;
            r.details.push(crate::assert::AssertDetail::new(
                crate::assert::DetailKind::Monitor,
                format!(
                    "scheduler crashed after completing step {} of {} ({:.1}s into test)",
                    step_idx,
                    steps.len(),
                    scenario_start.elapsed().as_secs_f64(),
                ),
            ));
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

    let mut result = collect_result(&mut state, effective_checks, ctx.topo);

    if sched_dead {
        result.passed = false;
        result.details.push(crate::assert::AssertDetail::new(
            crate::assert::DetailKind::Monitor,
            format!(
                "scheduler crashed during test (detected after all {} steps completed, {:.1}s elapsed)",
                steps.len(),
                scenario_start.elapsed().as_secs_f64(),
            ),
        ));
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
///
/// Validate that a MemPolicy's nodes are covered by the NUMA nodes
/// reachable from the resolved cpuset. Returns `Err` with a description
/// when the policy requests nodes outside the cpuset's NUMA coverage.
fn validate_mempolicy_cpuset(
    policy: &MemPolicy,
    cpuset: &BTreeSet<usize>,
    ctx: &Ctx,
    cgroup_name: &str,
) -> Result<()> {
    let policy_nodes = policy.node_set();
    if policy_nodes.is_empty() {
        return Ok(());
    }
    let cpuset_numa = ctx.topo.numa_nodes_for_cpuset(cpuset);
    let uncovered: Vec<usize> = policy_nodes
        .iter()
        .copied()
        .filter(|n| !cpuset_numa.contains(n))
        .collect();
    if !uncovered.is_empty() {
        anyhow::bail!(
            "cgroup '{}': MemPolicy references NUMA node(s) {:?} \
             but cpuset covers only node(s) {:?}",
            cgroup_name,
            uncovered,
            cpuset_numa,
        );
    }
    Ok(())
}

/// Each CgroupDef's `works` vec is iterated, spawning one WorkloadHandle
/// per Work entry. Multiple Works for the same cgroup produce multiple
/// handle entries with the same name key; Ops that filter by cgroup name
/// (StopCgroup, SetAffinity, etc.) naturally apply to all of them.
///
/// When `works` is empty, a single default Work is used (CpuSpin, Normal,
/// ctx.workers_per_cgroup workers).
fn apply_setup(ctx: &Ctx, state: &mut StepState<'_>, defs: &[CgroupDef]) -> Result<()> {
    let default_work = [Work::default()];
    for def in defs {
        state.cgroups.add_cgroup_no_cpuset(&def.name)?;
        if let Some(ref cpuset_spec) = def.cpuset {
            if let Err(reason) = cpuset_spec.validate(ctx) {
                anyhow::bail!(
                    "cgroup '{}': CpusetSpec validation failed: {}",
                    def.name,
                    reason
                );
            }
            let resolved = cpuset_spec.resolve(ctx);
            ctx.cgroups.set_cpuset(&def.name, &resolved)?;
            state.cpusets.insert(def.name.to_string(), resolved);
        }
        let effective_works: &[Work] = if def.works.is_empty() {
            &default_work
        } else {
            &def.works
        };
        for work in effective_works {
            if let Err(reason) = work.mem_policy.validate() {
                anyhow::bail!("cgroup '{}': {}", def.name, reason);
            }
        }
        let cgroup_cpuset = state.cpusets.get(def.name.as_ref());
        if let Some(resolved) = cgroup_cpuset {
            for work in effective_works {
                validate_mempolicy_cpuset(&work.mem_policy, resolved, ctx, &def.name)?;
            }
        }
        for work in effective_works {
            let n = work.num_workers.unwrap_or(ctx.workers_per_cgroup);
            let effective_work_type = crate::workload::resolve_work_type(
                &work.work_type,
                ctx.work_type_override.as_ref(),
                def.swappable,
                n,
            );
            let affinity =
                super::resolve_affinity_for_cgroup(&work.affinity, cgroup_cpuset, ctx.topo);
            let wl = WorkloadConfig {
                num_workers: n,
                affinity,
                work_type: effective_work_type,
                sched_policy: work.sched_policy,
                mem_policy: work.mem_policy.clone(),
                mpol_flags: work.mpol_flags,
            };
            let mut h = WorkloadHandle::spawn(&wl)?;
            ctx.cgroups.move_tasks(&def.name, &h.tids())?;
            h.start();
            state.handles.push((def.name.to_string(), h));
        }
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
            Op::RemoveCgroup { cgroup } => {
                // Stop workers in this cgroup first.
                state.handles.retain(|(n, _)| n.as_str() != *cgroup);
                state.cpusets.remove(cgroup.as_ref());
                let _ = ctx.cgroups.remove_cgroup(cgroup);
            }
            Op::SetCpuset { cgroup, cpus } => {
                if let Err(reason) = cpus.validate(ctx) {
                    anyhow::bail!(
                        "cgroup '{}': CpusetSpec validation failed: {}",
                        cgroup,
                        reason
                    );
                }
                let resolved = cpus.resolve(ctx);
                ctx.cgroups.set_cpuset(cgroup, &resolved)?;
                state.cpusets.insert(cgroup.to_string(), resolved);
            }
            Op::ClearCpuset { cgroup } => {
                ctx.cgroups.clear_cpuset(cgroup)?;
                state.cpusets.remove(cgroup.as_ref());
            }
            Op::SwapCpusets { a, b } => {
                // Read current cpusets from the cgroup filesystem, swap them.
                let cpus_a = read_cpuset(ctx, a);
                let cpus_b = read_cpuset(ctx, b);
                if let Some(ref ca) = cpus_a {
                    ctx.cgroups.set_cpuset(b, ca)?;
                    state.cpusets.insert(b.to_string(), ca.clone());
                }
                if let Some(ref cb) = cpus_b {
                    ctx.cgroups.set_cpuset(a, cb)?;
                    state.cpusets.insert(a.to_string(), cb.clone());
                }
            }
            Op::Spawn { cgroup, work } => {
                if let Err(reason) = work.mem_policy.validate() {
                    anyhow::bail!("cgroup '{}': {}", cgroup, reason);
                }
                let n = work.num_workers.unwrap_or(ctx.workers_per_cgroup);
                let cgroup_cpuset = state.cpusets.get(cgroup.as_ref());
                if let Some(resolved) = cgroup_cpuset {
                    validate_mempolicy_cpuset(&work.mem_policy, resolved, ctx, cgroup)?;
                }
                let affinity =
                    super::resolve_affinity_for_cgroup(&work.affinity, cgroup_cpuset, ctx.topo);
                let wl = WorkloadConfig {
                    num_workers: n,
                    affinity,
                    work_type: work.work_type.clone(),
                    sched_policy: work.sched_policy,
                    mem_policy: work.mem_policy.clone(),
                    mpol_flags: work.mpol_flags,
                };
                let mut h = WorkloadHandle::spawn(&wl)?;
                ctx.cgroups.move_tasks(cgroup, &h.tids())?;
                h.start();
                state.handles.push((cgroup.to_string(), h));
            }
            Op::StopCgroup { cgroup } => {
                state.handles.retain(|(n, _)| n.as_str() != *cgroup);
            }
            Op::SetAffinity { cgroup, affinity } => {
                let cgroup_cpuset = state.cpusets.get(cgroup.as_ref());
                let resolved =
                    super::resolve_affinity_for_cgroup(affinity, cgroup_cpuset, ctx.topo);
                for (name, handle) in &state.handles {
                    if name.as_str() == *cgroup {
                        match &resolved {
                            crate::workload::AffinityMode::None => {}
                            crate::workload::AffinityMode::Fixed(cpus) => {
                                for idx in 0..handle.tids().len() {
                                    let _ = handle.set_affinity(idx, cpus);
                                }
                            }
                            crate::workload::AffinityMode::Random { from, count }
                                if !from.is_empty() =>
                            {
                                use rand::seq::IndexedRandom;
                                let v: Vec<usize> = from.iter().copied().collect();
                                for idx in 0..handle.tids().len() {
                                    let chosen: BTreeSet<usize> =
                                        v.sample(&mut rand::rng(), *count).copied().collect();
                                    let _ = handle.set_affinity(idx, &chosen);
                                }
                            }
                            // Empty Random pool: matches workload::resolve_affinity
                            // — leave affinity unchanged rather than hand
                            // sched_setaffinity an empty mask (which it
                            // rejects with EINVAL).
                            crate::workload::AffinityMode::Random { .. } => {}
                            crate::workload::AffinityMode::SingleCpu(cpu) => {
                                let cpus: BTreeSet<usize> = [*cpu].into_iter().collect();
                                for idx in 0..handle.tids().len() {
                                    let _ = handle.set_affinity(idx, &cpus);
                                }
                            }
                        }
                    }
                }
            }
            Op::SpawnHost { work } => {
                if let Err(reason) = work.mem_policy.validate() {
                    anyhow::bail!("SpawnHost: {}", reason);
                }
                let n = work.num_workers.unwrap_or(ctx.workers_per_cgroup);
                let affinity = super::resolve_affinity_for_cgroup(&work.affinity, None, ctx.topo);
                let wl = WorkloadConfig {
                    num_workers: n,
                    affinity,
                    work_type: work.work_type.clone(),
                    sched_policy: work.sched_policy,
                    mem_policy: work.mem_policy.clone(),
                    mpol_flags: work.mpol_flags,
                };
                let mut h = WorkloadHandle::spawn(&wl)?;
                h.start();
                // Empty string key: workers in parent cgroup, not a managed cgroup.
                state.handles.push((String::new(), h));
            }
            Op::MoveAllTasks { from, to } => {
                // Clear subtree_control on the destination before moving
                // tasks. The kernel's no-internal-process constraint
                // (cgroup_migrate_vet_dst) returns EBUSY when writing to
                // cgroup.procs of a cgroup with subtree_control set.
                if let Err(e) = ctx.cgroups.clear_subtree_control(to) {
                    tracing::warn!(
                        cgroup = to.as_ref(),
                        err = %e,
                        "failed to clear subtree_control before task move"
                    );
                }
                for (name, handle) in state.handles.iter() {
                    if name.as_str() == *from {
                        ctx.cgroups.move_tasks(to, &handle.tids())?;
                    }
                }
                for (name, _) in &mut state.handles {
                    if name.as_str() == *from {
                        *name = to.to_string();
                    }
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
    let cpus: BTreeSet<usize> = crate::topology::parse_cpu_list_lenient(content)
        .into_iter()
        .collect();
    Some(cpus)
}

/// Collect all worker results and produce an AssertResult.
///
/// Drains handles from state and delegates to [`collect_handles`](super::collect_handles),
/// passing each cgroup's tracked cpuset for isolation checks.
fn collect_result(
    state: &mut StepState<'_>,
    checks: &crate::assert::Assert,
    topo: &crate::topology::TestTopology,
) -> AssertResult {
    let handles = std::mem::take(&mut state.handles);
    super::collect_handles(
        handles
            .into_iter()
            .map(|(name, h)| (h, state.cpusets.get(&name))),
        checks,
        Some(topo),
    )
}

#[cfg(test)]
mod tests {
    use std::ops::RangeInclusive;

    use super::*;
    use crate::vmm::shm_ring::parse_shm_params_from_str;

    // -- Traverse combinator (test-only) --

    /// Layout strategy for Traverse phases.
    #[derive(Debug)]
    enum Layout {
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
    #[derive(Debug)]
    struct Traverse {
        seed: Option<u64>,
        cgroup_count: RangeInclusive<usize>,
        layouts: Vec<Layout>,
        phases: usize,
        phase_duration: Duration,
        settle: Duration,
        /// Cgroups [0..persistent_cgroups) are created once and never removed.
        persistent_cgroups: usize,
        /// Work definition per cgroup index. Last entry repeats for higher indices.
        cgroup_workloads: Vec<Work>,
    }

    impl Traverse {
        /// Generate a `Vec<Step>` from the Traverse configuration.
        fn generate(&self, ctx: &Ctx) -> Vec<Step> {
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
                    let w = self
                        .cgroup_workloads
                        .get(idx)
                        .or(self.cgroup_workloads.last())
                        .cloned()
                        .unwrap_or_default();
                    ops.push(Op::AddCgroup { name: name.clone() });
                    ops.push(Op::Spawn {
                        cgroup: name.clone(),
                        work: w,
                    });
                    live_cgroups.push(name);
                }

                // Remove cgroups if needed (never remove persistent cgroups).
                while live_cgroups.len() > target_count
                    && live_cgroups.len() > self.persistent_cgroups
                {
                    if let Some(name) = live_cgroups.pop() {
                        ops.push(Op::StopCgroup {
                            cgroup: name.clone(),
                        });
                        ops.push(Op::RemoveCgroup { cgroup: name });
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
    fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(seed)
    }

    // -- Op discriminant tests --

    #[test]
    fn op_discriminant_unique() {
        let ops: Vec<Op> = vec![
            Op::AddCgroup { name: "a".into() },
            Op::RemoveCgroup { cgroup: "a".into() },
            Op::SetCpuset {
                cgroup: "a".into(),
                cpus: CpusetSpec::exact([]),
            },
            Op::ClearCpuset { cgroup: "a".into() },
            Op::SwapCpusets {
                a: "a".into(),
                b: "b".into(),
            },
            Op::Spawn {
                cgroup: "a".into(),
                work: Default::default(),
            },
            Op::StopCgroup { cgroup: "a".into() },
            Op::SetAffinity {
                cgroup: "a".into(),
                affinity: Default::default(),
            },
            Op::SpawnHost {
                work: Default::default(),
            },
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into(),
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
        assert_eq!(Op::RemoveCgroup { cgroup: "a".into() }.discriminant(), 1);
        assert_eq!(
            Op::SpawnHost {
                work: Default::default()
            }
            .discriminant(),
            8
        );
        assert_eq!(
            Op::MoveAllTasks {
                from: "a".into(),
                to: "b".into()
            }
            .discriminant(),
            9
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
        let topo = crate::topology::TestTopology::from_spec(1, 1, 4, 1);
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let ctx = Ctx {
            cgroups: &cgroups,
            topo: &topo,
            duration: Duration::from_secs(10),
            workers_per_cgroup: 4,
            sched_pid: 0,
            settle: Duration::from_millis(1000),
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
        let cmdline = "console=ttyS0 KTSTR_SHM_BASE=0xfc000000 KTSTR_SHM_SIZE=0x10000 quiet";
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
        llcs: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_spec(1, llcs, cores, threads);
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
            settle: Duration::ZERO,
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
        let usable = ctx.topo.usable_cpuset();
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
        let usable = ctx.topo.usable_cpuset();
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
        let usable = ctx.topo.usable_cpuset();
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
        let usable = ctx.topo.usable_cpuset();
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
    fn cpusetspec_llc_two_llcs_disjoint() {
        let (cg, topo) = make_ctx(2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let llc0 = CpusetSpec::Llc(0).resolve(&ctx);
        let llc1 = CpusetSpec::Llc(1).resolve(&ctx);
        assert!(llc0.is_disjoint(&llc1), "LLCs should be disjoint");
    }

    // -- CpusetSpec::Numa --

    fn make_numa_ctx(
        numa_nodes: u32,
        llcs: u32,
        cores: u32,
        threads: u32,
    ) -> (crate::cgroup::CgroupManager, crate::topology::TestTopology) {
        let cgroups = crate::cgroup::CgroupManager::new("/nonexistent");
        let topo = crate::topology::TestTopology::from_spec(numa_nodes, llcs, cores, threads);
        (cgroups, topo)
    }

    #[test]
    fn cpusetspec_numa_node_zero() {
        // 2 NUMA nodes, 4 LLCs (2 per NUMA), 4 cores, 1 thread
        // LLCs 0,1 -> NUMA 0 (CPUs 0-7), LLCs 2,3 -> NUMA 1 (CPUs 8-15)
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(0).resolve(&ctx);
        let expected: BTreeSet<usize> = (0..8).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn cpusetspec_numa_node_one() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(1).resolve(&ctx);
        let expected: BTreeSet<usize> = (8..16).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn cpusetspec_numa_disjoint() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let node0 = CpusetSpec::Numa(0).resolve(&ctx);
        let node1 = CpusetSpec::Numa(1).resolve(&ctx);
        assert!(
            node0.is_disjoint(&node1),
            "NUMA nodes should be disjoint: {:?} vs {:?}",
            node0,
            node1
        );
        let union: BTreeSet<usize> = node0.union(&node1).copied().collect();
        assert_eq!(union, ctx.topo.all_cpuset());
    }

    #[test]
    fn cpusetspec_numa_single_node_returns_all() {
        let (cg, topo) = make_numa_ctx(1, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpus = CpusetSpec::Numa(0).resolve(&ctx);
        assert_eq!(cpus, ctx.topo.all_cpuset());
    }

    #[test]
    fn cpusetspec_numa_validate_out_of_range() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Numa(5);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn cpusetspec_numa_validate_valid() {
        let (cg, topo) = make_numa_ctx(2, 4, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        assert!(CpusetSpec::Numa(0).validate(&ctx).is_ok());
        assert!(CpusetSpec::Numa(1).validate(&ctx).is_ok());
    }

    #[test]
    fn cpusetspec_numa_convenience_constructor() {
        let spec = CpusetSpec::numa(0);
        assert!(matches!(spec, CpusetSpec::Numa(0)));
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
            cgroup_workloads: vec![Work::default()],
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
            cgroup_workloads: vec![Work::default()],
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
            cgroup_workloads: vec![Work::default()],
        };
        let steps = t.generate(&ctx);
        // Every phase should have at least persistent_cgroups worth of SetCpuset ops
        // (cg_0, cg_1 are never removed).
        for step in &steps {
            let remove_ops: Vec<&Op> = step.ops.iter()
                .filter(|op| matches!(op, Op::RemoveCgroup { cgroup } if cgroup == "cg_0" || cgroup == "cg_1"))
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
            .with_cpuset(CpusetSpec::llc(0))
            .workers(8)
            .work_type(WorkType::bursty(50, 100))
            .sched_policy(crate::workload::SchedPolicy::Batch)
            .swappable(true);
        assert_eq!(d.name, "test");
        assert!(d.cpuset.is_some());
        assert_eq!(d.works.len(), 1);
        assert_eq!(d.works[0].num_workers, Some(8));
        assert!(d.swappable);
    }

    #[test]
    fn cgroup_def_default() {
        let d = CgroupDef::default();
        assert_eq!(d.name, "cg_0");
        assert!(d.cpuset.is_none());
        assert!(d.works.is_empty());
        assert!(!d.swappable);
    }

    #[test]
    fn cgroup_def_multi_work() {
        let d = CgroupDef::named("multi")
            .work(Work::default().workers(4).work_type(WorkType::CpuSpin))
            .work(Work::default().workers(2).work_type(WorkType::YieldHeavy));
        assert_eq!(d.works.len(), 2);
        assert_eq!(d.works[0].num_workers, Some(4));
        assert_eq!(d.works[1].num_workers, Some(2));
    }

    #[test]
    fn cgroup_def_old_api_then_work() {
        let d = CgroupDef::named("mixed")
            .workers(4)
            .work(Work::default().workers(2));
        assert_eq!(d.works.len(), 2);
        assert_eq!(d.works[0].num_workers, Some(4));
        assert_eq!(d.works[1].num_workers, Some(2));
    }

    #[test]
    fn cgroup_def_work_only_no_phantom() {
        let d = CgroupDef::named("explicit").work(Work::default().workers(3));
        assert_eq!(d.works.len(), 1);
        assert_eq!(d.works[0].num_workers, Some(3));
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

    // -- Step::with_defs / with_ops --

    #[test]
    fn step_with_defs_empty() {
        let step = Step::with_defs(vec![], HoldSpec::Frac(0.5));
        assert!(step.setup.is_empty());
        assert!(step.ops.is_empty());
    }

    #[test]
    fn step_with_defs_populated() {
        let step = Step::with_defs(
            vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
            HoldSpec::Fixed(Duration::from_secs(5)),
        );
        assert!(!step.setup.is_empty());
        assert!(step.ops.is_empty());
    }

    #[test]
    fn step_with_defs_then_ops() {
        let step = Step::with_defs(vec![CgroupDef::named("cg_0")], HoldSpec::FULL).with_ops(vec![
            Op::AddCgroup {
                name: "cg_1".into(),
            },
        ]);
        assert!(!step.setup.is_empty());
        assert_eq!(step.ops.len(), 1);
    }

    #[test]
    fn step_with_ops_replaces() {
        let step = Step::new(
            vec![Op::AddCgroup { name: "a".into() }],
            HoldSpec::Frac(0.5),
        )
        .with_ops(vec![
            Op::AddCgroup { name: "b".into() },
            Op::RemoveCgroup { cgroup: "c".into() },
        ]);
        assert_eq!(step.ops.len(), 2);
    }

    // -- CpusetSpec::validate --

    #[test]
    fn cpusetspec_validate_disjoint_of_zero() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 0 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_disjoint_index_ge_of() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 3, of: 3 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("index 3 >= partition count 3"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_of_zero() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 0,
            frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_index_ge_of() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 2,
            of: 2,
            frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("index 2 >= partition count 2"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_start_ge_end() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: 0.2,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("start_frac"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_nan() {
        // Regression: IEEE 754 comparisons with NaN always return false, so
        // `start_frac >= end_frac` failed to reject it. validate() now
        // rejects non-finite fracs explicitly.
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.8,
            end_frac: f64::NAN,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_infinity() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.0,
            end_frac: f64::INFINITY,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_negative() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: -0.5,
            end_frac: 0.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_range_rejects_above_one() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Range {
            start_frac: 0.5,
            end_frac: 1.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_nan_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::NAN,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_infinity_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: f64::INFINITY,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not finite"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_overlap_rejects_out_of_range_frac() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Overlap {
            index: 0,
            of: 2,
            frac: 1.5,
        };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("[0.0, 1.0]"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_too_few_cpus_for_partitions() {
        // 1 LLC, 2 cores, 1 thread => 2 total cpus, 2 usable
        let (cg, topo) = make_ctx(1, 2, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 0, of: 5 };
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("not enough usable CPUs"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_exact_always_ok() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::exact([99]);
        assert!(spec.validate(&ctx).is_ok());
    }

    #[test]
    fn cpusetspec_validate_llc_out_of_range() {
        let (cg, topo) = make_ctx(1, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Llc(5);
        let err = spec.validate(&ctx).unwrap_err();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn cpusetspec_validate_valid_disjoint_ok() {
        let (cg, topo) = make_ctx(1, 8, 1);
        let ctx = ctx_from(&cg, &topo);
        let spec = CpusetSpec::Disjoint { index: 1, of: 2 };
        assert!(spec.validate(&ctx).is_ok());
    }

    // -- MemPolicy + cpuset validation tests --

    #[test]
    fn validate_mempolicy_default_always_ok() {
        // 2 NUMA nodes, 2 LLCs (1 per node), 4 cores, 1 thread = 8 CPUs
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        assert!(validate_mempolicy_cpuset(&MemPolicy::Default, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_local_always_ok() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect();
        assert!(validate_mempolicy_cpuset(&MemPolicy::Local, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_bind_covered() {
        // 2 NUMA nodes, 2 LLCs, 4 cores each = 8 CPUs total
        // LLC 0 (CPUs 0-3) = NUMA 0, LLC 1 (CPUs 4-7) = NUMA 1
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..8).collect(); // covers both nodes
        let policy = MemPolicy::Bind([0, 1].into_iter().collect());
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_bind_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Bind([1].into_iter().collect()); // node 1 not in cpuset
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_err());
    }

    #[test]
    fn validate_mempolicy_preferred_covered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (4..8).collect(); // NUMA node 1
        let policy = MemPolicy::Preferred(1);
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_ok());
    }

    #[test]
    fn validate_mempolicy_preferred_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Preferred(1);
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_err());
    }

    #[test]
    fn validate_mempolicy_interleave_partial_uncovered() {
        let (cg, topo) = make_numa_ctx(2, 2, 4, 1);
        let ctx = ctx_from(&cg, &topo);
        let cpuset: BTreeSet<usize> = (0..4).collect(); // NUMA node 0 only
        let policy = MemPolicy::Interleave([0, 1].into_iter().collect());
        assert!(validate_mempolicy_cpuset(&policy, &cpuset, &ctx, "cg_0").is_err());
    }

    #[test]
    fn cgroupdef_mem_policy_builder() {
        let def = CgroupDef::named("test").mem_policy(MemPolicy::Bind([0].into_iter().collect()));
        assert!(matches!(def.works[0].mem_policy, MemPolicy::Bind(_)));
    }
}
