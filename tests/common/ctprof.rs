//! Shared test helpers for the ctprof integration suite
//! (compare + show pipelines).
//!
//! `make_thread`, `snapshot`, and `cgroup_stats_entry` are
//! synthetic-fixture builders that both
//! `tests/ctprof_compare.rs` and
//! `tests/ctprof_show.rs` use to drive the public surface
//! (CtprofSnapshot / write/load roundtrip / run_compare /
//! run_show via the binary). Living here under `common/`
//! keeps the integration-test harness from picking the file up
//! as its own test binary AND avoids the duplicated copies of
//! these helpers across the two integration files.
//!
//! All functions return concrete public types from
//! [`ktstr::ctprof`]; no test-only API is exposed.

use std::collections::BTreeMap;

use ktstr::ctprof::{CgroupStats, CtprofSnapshot, ThreadState};
use ktstr::metric_types::{CategoricalString, CpuSet};

/// Build a `ThreadState` populated with sane defaults for
/// integration tests: tid/tgid 1, the supplied pcomm/comm,
/// root cgroup, `SCHED_OTHER` policy, and a 4-CPU affinity set.
/// Callers mutate the returned struct further (e.g.
/// `t.run_time_ns = MonotonicNs(5_000_000)`) before pushing into
/// a snapshot.
#[allow(dead_code)]
pub fn make_thread(pcomm: &str, comm: &str) -> ThreadState {
    let mut t = ThreadState::default();
    t.tid = 1;
    t.tgid = 1;
    t.pcomm = pcomm.into();
    t.comm = comm.into();
    t.cgroup = "/".into();
    t.policy = CategoricalString("SCHED_OTHER".into());
    t.cpu_affinity = CpuSet(vec![0, 1, 2, 3]);
    t
}

/// Wrap a thread vector + cgroup-stats map into a
/// `CtprofSnapshot`. Uses `Default::default()` + per-field
/// assignment because `CtprofSnapshot` is `#[non_exhaustive]`
/// — direct struct-literal construction is rejected outside
/// the defining crate's module.
#[allow(dead_code)]
pub fn snapshot(
    threads: Vec<ThreadState>,
    cgroup_stats: BTreeMap<String, CgroupStats>,
) -> CtprofSnapshot {
    let mut snap = CtprofSnapshot::default();
    snap.threads = threads;
    snap.cgroup_stats = cgroup_stats;
    snap
}

/// Build a `CgroupStats` entry from raw counter values. Same
/// `#[non_exhaustive]` constraint as `snapshot` — populate via
/// `Default::default()` + per-field assignment. Reaches into
/// the nested-controller shape introduced in #61: cpu counters
/// land on the `cpu` sub-struct, memory.current on `memory`.
#[allow(dead_code)]
pub fn cgroup_stats_entry(
    cpu_usage_usec: u64,
    nr_throttled: u64,
    throttled_usec: u64,
    memory_current: u64,
) -> CgroupStats {
    let mut s = CgroupStats::default();
    s.cpu.usage_usec = cpu_usage_usec;
    s.cpu.nr_throttled = nr_throttled;
    s.cpu.throttled_usec = throttled_usec;
    s.memory.current = memory_current;
    s
}
