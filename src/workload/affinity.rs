//! CPU affinity intent + resolution for worker tasks.
//!
//! [`AffinityIntent`] expresses the test author's request before
//! topology resolution; [`ResolvedAffinity`] is the post-resolution
//! shape the spawn pipeline consumes. [`resolve_affinity`] walks the
//! `ResolvedAffinity` to a concrete CPU set; [`set_thread_affinity`]
//! issues the `sched_setaffinity` syscall; [`sched_getcpu`] is the
//! inverse query (which CPU is the current task on right now).
//!
//! Re-exported from the parent module via `pub use affinity::*` so
//! `crate::workload::AffinityIntent` etc. stay valid.

use std::collections::BTreeSet;

use anyhow::{Context, Result};

/// Scenario-level affinity intent for a group of workers.
///
/// Resolved to a concrete [`ResolvedAffinity`] at runtime based on the
/// cgroup's effective cpuset and the VM's topology. When attached to
/// a [`WorkSpec`], determines per-worker `sched_setaffinity` masks.
///
/// Resolution uses [`resolve_affinity_for_cgroup()`](crate::scenario::resolve_affinity_for_cgroup).
///
/// # Naming pattern (Intent vs Resolved)
///
/// [`AffinityIntent`] and [`ResolvedAffinity`] form a pre/post-resolution
/// pair. Variant names line up where the same shape exists on both
/// sides; payload differences encode the intent → concrete-CPU-set
/// distinction:
///
/// | [`AffinityIntent`]                       | [`ResolvedAffinity`]              |
/// |------------------------------------------|-----------------------------------|
/// | `Inherit` (no payload)                   | `None`                            |
/// | `Exact(BTreeSet<usize>)`                 | `Fixed(BTreeSet<usize>)`          |
/// | `RandomSubset { from, count }`           | `Random { from, count }`          |
/// | `SingleCpu` (no payload)                 | `SingleCpu(usize)`                |
/// | `LlcAligned` / `CrossCgroup`             | `Fixed(...)` (resolver expands)   |
/// | `SmtSiblingPair` (no payload)            | `Fixed({sibling_a, sibling_b})`   |
///
/// Constructor helpers: [`AffinityIntent::exact`] takes any
/// `IntoIterator<Item = usize>` for the `Exact` set;
/// [`AffinityIntent::random_subset`] takes the same iterator shape
/// for the `RandomSubset` pool plus a sample-count argument.
///
/// The `SingleCpu` pair specifically: [`AffinityIntent::SingleCpu`]
/// expresses "pin to one CPU; resolver picks which based on cgroup
/// state and worker index", and [`ResolvedAffinity::SingleCpu`]
/// records the concrete CPU id chosen. Reusing the variant name keeps
/// the pre/post mapping lexically obvious — payload presence
/// distinguishes intent from resolution without renaming the variant.
///
/// [`AffinityIntent::RandomSubset`] carries the resolved pool
/// (`from`) and sample size (`count`) — sampling itself is deferred
/// to spawn time so each worker gets an independent draw. The
/// scenario engine's `resolve_affinity_for_cgroup` materialises the
/// pool from cgroup cpuset / topology before constructing this
/// variant; spawn-time `resolve_affinity` samples per-worker.
/// Construct directly via [`AffinityIntent::random_subset`].
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AffinityIntent {
    /// No affinity constraint -- inherit from parent cgroup.
    #[default]
    Inherit,
    /// Pin each worker to a random subset of `from`, sampling `count`
    /// CPUs per worker. Sampling is deferred to spawn time so each
    /// worker gets an independent draw — mirrors
    /// [`ResolvedAffinity::Random`] semantics. Construct with the
    /// resolved pool already materialised; the scenario engine pre-
    /// resolves topology-aware "pick from cgroup state" intent
    /// before building this variant.
    RandomSubset { from: BTreeSet<usize>, count: usize },
    /// Pin to the CPUs in the worker's LLC.
    LlcAligned,
    /// Pin to all CPUs (crosses cgroup boundaries).
    CrossCgroup,
    /// Pin to a single CPU.
    SingleCpu,
    /// Pin to an exact set of CPUs.
    Exact(BTreeSet<usize>),
    /// Pin all workers in the group to the two SMT siblings of one
    /// physical core. Tests how the scheduler handles two
    /// compute-bound tasks placed on SMT siblings — both threads
    /// contend for the core's shared front-end / execution
    /// resources, exposing scheduler decisions about co-running
    /// vs. spreading compute load across cores.
    ///
    /// Designed for [`WorkType::SmtSiblingSpin`] and other
    /// `worker_group_size = 2` variants
    /// ([`WorkType::FutexPingPong`], [`WorkType::AsymmetricWaker`],
    /// [`WorkType::SignalStorm`], etc.) where both workers in a
    /// group are intended to run on a sibling pair. The variant
    /// has no payload — the resolver picks an SMT-sibling pair
    /// from the cgroup's effective cpuset (or the full topology
    /// when no cpuset is active).
    ///
    /// Resolution is performed by the scenario engine's
    /// `resolve_affinity_for_cgroup` (topology-aware, not
    /// available at the bare [`WorkloadHandle::spawn`] gate). The
    /// resolver searches the cpuset for a physical core with at
    /// least two thread siblings present and resolves to
    /// [`ResolvedAffinity::Fixed`] containing those two CPU IDs.
    /// All workers in the group get pinned to that 2-CPU set;
    /// when `num_workers == 2` the kernel runs one worker on each
    /// sibling, which is the contention pattern this intent
    /// targets.
    ///
    /// Returns an error from the resolver — NOT a silent
    /// fallback — when no SMT-sibling pair is available
    /// (`threads_per_core == 1`, or the cpuset isolates each
    /// sibling onto a different CPU set). Callers must handle
    /// the error; running [`WorkType::SmtSiblingSpin`] without
    /// SMT siblings would produce a misleading result.
    SmtSiblingPair,
}

impl AffinityIntent {
    /// Construct an `Exact` affinity from any iterator of CPU indices.
    ///
    /// Accepts arrays, ranges, `Vec`, `BTreeSet`, or any `IntoIterator<Item = usize>`.
    pub fn exact(cpus: impl IntoIterator<Item = usize>) -> Self {
        AffinityIntent::Exact(cpus.into_iter().collect())
    }

    /// Construct a `RandomSubset` from a pool iterator and a sample
    /// size. Mirrors the [`Self::exact`] constructor's iterator
    /// flexibility — accepts arrays, `Vec`, `BTreeSet`, ranges, or
    /// any `IntoIterator<Item = usize>` for the pool.
    ///
    /// Sampling is deferred to spawn time; each worker gets an
    /// independent `count`-sized draw from `from`. `count > from.len()`
    /// is clamped to `from.len()` at sample time (topology fact, not
    /// caller error). `count == 0` and empty `from` are rejected at
    /// the spawn-time affinity gate with an actionable diagnostic —
    /// use [`AffinityIntent::Inherit`] for no affinity constraint.
    pub fn random_subset(from: impl IntoIterator<Item = usize>, count: usize) -> Self {
        AffinityIntent::RandomSubset {
            from: from.into_iter().collect(),
            count,
        }
    }
}

/// Resolved CPU affinity for a worker process.
///
/// Created from [`AffinityIntent`] at runtime based on topology and
/// cpuset assignments. Variant names track [`AffinityIntent`] where the
/// same shape exists pre/post-resolution; payload presence
/// distinguishes intent from concrete CPU id(s). See the
/// [`AffinityIntent`] type doc for the full pre/post mapping table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedAffinity {
    /// No affinity constraint.
    None,
    /// Pin to a specific set of CPUs.
    Fixed(BTreeSet<usize>),
    /// Pin to `count` randomly-chosen CPUs from `from`.
    ///
    /// - `count` must be `> 0`; zero is rejected at resolve time
    ///   (previously it coerced silently to 1 and masked caller bugs).
    /// - `count > from.len()` is clamped to `from.len()` — asking for
    ///   more CPUs than the pool contains is a topology fact, not a
    ///   caller error.
    /// - `from` empty with `count > 0` resolves to no affinity (no
    ///   pool to sample from); downstream treats this as `None`.
    Random { from: BTreeSet<usize>, count: usize },
    /// Pin to a single CPU.
    SingleCpu(usize),
}

/// Resolve a [`ResolvedAffinity`] into the concrete CPU set the
/// spawn pipeline writes into the worker's `sched_setaffinity` mask.
///
/// `Random` samples `count` CPUs from `from` per call (each worker
/// gets an independent draw at spawn time). Empty `from` with
/// `count > 0` returns `Ok(None)` (no affinity applied) and logs a
/// debug-level note; `count == 0` is a caller bug and bails.
pub(crate) fn resolve_affinity(mode: &ResolvedAffinity) -> Result<Option<BTreeSet<usize>>> {
    match mode {
        ResolvedAffinity::None => Ok(None),
        ResolvedAffinity::Fixed(cpus) => Ok(Some(cpus.clone())),
        ResolvedAffinity::SingleCpu(cpu) => Ok(Some([*cpu].into_iter().collect())),
        ResolvedAffinity::Random { from, count } => {
            use rand::seq::IndexedRandom;
            if *count == 0 {
                anyhow::bail!(
                    "ResolvedAffinity::Random.count must be > 0; a zero count \
                     previously silently coerced to 1, masking caller bugs"
                );
            }
            if from.is_empty() {
                tracing::debug!(
                    count = count,
                    "resolve_affinity: empty Random pool, leaving affinity unset"
                );
                return Ok(None);
            }
            let pool: Vec<usize> = from.iter().copied().collect();
            // Clamp count down to the pool size (user asked for more
            // CPUs than exist). Silent clamp is fine here: the pool
            // upper bound is a topology fact, not a caller bug.
            let count = (*count).min(pool.len());
            Ok(Some(
                pool.sample(&mut rand::rng(), count).copied().collect(),
            ))
        }
    }
}

/// Return the CPU the calling task is currently running on.
///
/// Falls back to `0` on syscall failure (rare; would mean
/// `getcpu(2)` is unavailable, which is not the case on any
/// supported kernel). Wraps [`nix::sched::sched_getcpu`].
pub(crate) fn sched_getcpu() -> usize {
    nix::sched::sched_getcpu().unwrap_or(0)
}

/// Set per-thread CPU affinity via `sched_setaffinity(2)`.
///
/// `pid` must be `> 0` — `pid <= 0` has broadcast semantics at the
/// syscall level (target the calling task or every task in a tgid
/// depending on layer) and is rejected up-front so no caller passes
/// an unchecked `0` through.
pub fn set_thread_affinity(pid: libc::pid_t, cpus: &BTreeSet<usize>) -> Result<()> {
    use nix::sched::{CpuSet, sched_setaffinity};
    use nix::unistd::Pid;
    // See `set_sched_policy` for the rationale — pid <= 0 has
    // broadcast semantics at the syscall and must not be passed
    // through unchecked.
    if pid <= 0 {
        anyhow::bail!("sched_setaffinity: invalid pid {pid} (must be > 0)");
    }
    let mut cpu_set = CpuSet::new();
    for &cpu in cpus {
        cpu_set
            .set(cpu)
            .with_context(|| format!("CPU {cpu} out of range"))?;
    }
    sched_setaffinity(Pid::from_raw(pid), &cpu_set)
        .with_context(|| format!("sched_setaffinity pid={pid}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn resolve_affinity_none() {
        let r = resolve_affinity(&ResolvedAffinity::None).unwrap();
        assert!(r.is_none());
    }
    #[test]
    fn resolve_affinity_fixed() {
        let cpus: BTreeSet<usize> = [0, 1, 2].into_iter().collect();
        let r = resolve_affinity(&ResolvedAffinity::Fixed(cpus.clone())).unwrap();
        assert_eq!(r, Some(cpus));
    }
    #[test]
    fn resolve_affinity_single_cpu() {
        let r = resolve_affinity(&ResolvedAffinity::SingleCpu(5)).unwrap();
        assert_eq!(r, Some([5].into_iter().collect()));
    }
    /// `ResolvedAffinity` derives `Debug`; the `SingleCpu` variant
    /// must render its variant name and the embedded CPU id so
    /// failure-dump and tracing output are diagnosable. Pins the
    /// derive against accidental removal.
    #[test]
    fn resolved_affinity_single_cpu_debug_format() {
        let dbg = format!("{:?}", ResolvedAffinity::SingleCpu(7));
        assert!(
            dbg.contains("SingleCpu"),
            "Debug output must name the variant, got: {dbg}"
        );
        assert!(
            dbg.contains('7'),
            "Debug output must include the CPU id payload, got: {dbg}"
        );
    }
    #[test]
    fn resolve_affinity_random() {
        let from: BTreeSet<usize> = (0..8).collect();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 3 }).unwrap();
        let cpus = r.unwrap();
        assert_eq!(cpus.len(), 3);
        assert!(cpus.iter().all(|c| *c < 8));
    }
    #[test]
    fn resolve_affinity_random_clamps_count() {
        let from: BTreeSet<usize> = [0, 1].into_iter().collect();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 10 }).unwrap();
        assert_eq!(r.unwrap().len(), 2);
    }
    #[test]
    fn resolve_affinity_random_single_cpu_pool() {
        let from: BTreeSet<usize> = [7].into_iter().collect();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 1 }).unwrap();
        assert_eq!(r.unwrap(), [7].into_iter().collect());
    }
    #[test]
    fn affinity_mode_debug_shows_cpus() {
        let a = ResolvedAffinity::Fixed([0, 1, 7].into_iter().collect());
        let s = format!("{:?}", a);
        assert!(s.contains("0"), "must show CPU 0");
        assert!(s.contains("1"), "must show CPU 1");
        assert!(s.contains("7"), "must show CPU 7");
        // Different CPU sets produce different output.
        let b = ResolvedAffinity::Fixed([3, 4].into_iter().collect());
        let s2 = format!("{:?}", b);
        assert!(s2.contains("3"), "must show CPU 3");
        assert_ne!(
            s, s2,
            "different CPU sets must produce different debug output"
        );
    }
    #[test]
    fn affinity_mode_clone_preserves_cpus() {
        let cpus: BTreeSet<usize> = [2, 5, 7].into_iter().collect();
        let a = ResolvedAffinity::Random {
            from: cpus.clone(),
            count: 2,
        };
        let b = a.clone();
        match b {
            ResolvedAffinity::Random { from, count } => {
                assert_eq!(from, cpus, "cloned from set must match original");
                assert_eq!(count, 2, "cloned count must match original");
            }
            _ => panic!("clone must preserve variant"),
        }
    }
    // -- resolve_affinity edge cases --

    #[test]
    fn resolve_affinity_random_zero_count_rejected() {
        // Regression: count=0 previously coerced silently to 1, masking
        // caller bugs. Now it must return an Err.
        let from: BTreeSet<usize> = (0..4).collect();
        let err = resolve_affinity(&ResolvedAffinity::Random { from, count: 0 }).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("count") && msg.contains("> 0"),
            "error must name the field: {msg}"
        );
    }
    #[test]
    fn resolve_affinity_random_empty_pool_is_none() {
        // Regression: ResolvedAffinity::Random { from: empty, count } previously
        // produced an empty affinity mask rejected by sched_setaffinity
        // with EINVAL. Empty pool must short-circuit to Ok(None).
        let from: BTreeSet<usize> = BTreeSet::new();
        let r = resolve_affinity(&ResolvedAffinity::Random { from, count: 1 }).unwrap();
        assert!(r.is_none(), "empty Random pool must resolve to no affinity");
    }

    #[test]
    fn sched_getcpu_valid() {
        let cpu = sched_getcpu();
        let max = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(cpu < max, "cpu {cpu} >= max {max}");
    }

    #[test]
    fn set_thread_affinity_cpu_zero() {
        let pid: libc::pid_t = unsafe { libc::getpid() };
        let cpus: BTreeSet<usize> = [0].into_iter().collect();
        let result = set_thread_affinity(pid, &cpus);
        assert!(result.is_ok(), "pinning to CPU 0 should succeed");
    }
}
