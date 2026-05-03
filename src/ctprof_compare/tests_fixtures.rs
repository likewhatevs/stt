//! Cross-cutting test fixtures shared by all ctprof_compare
//! per-module test files. Centralized here because helpers like
//! `make_thread`, `snap_with`, `simple_cgroup_stats`, `lookup_metric`,
//! and the `fudge_*` family are used by tests in 4-5 different
//! submodules; duplicating them per-file would invite drift.

#![allow(dead_code)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;

use super::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, ThreadState};
use crate::metric_types::{CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32};

pub(super) fn make_thread(pcomm: &str, comm: &str) -> ThreadState {
    ThreadState {
        tid: 1,
        tgid: 1,
        pcomm: pcomm.into(),
        comm: comm.into(),
        cgroup: "/".into(),
        start_time_clock_ticks: 0,
        policy: CategoricalString("SCHED_OTHER".into()),
        nice: OrdinalI32(0),
        cpu_affinity: CpuSet(vec![0, 1, 2, 3]),
        ..ThreadState::default()
    }
}

pub(super) fn snap_with(threads: Vec<ThreadState>) -> CtprofSnapshot {
    CtprofSnapshot {
        captured_at_unix_ns: 0,
        host: None,
        threads,
        cgroup_stats: BTreeMap::new(),
        probe_summary: None,
        parse_summary: None,
        taskstats_summary: None,
        psi: crate::ctprof::Psi::default(),
        sched_ext: None,
    }
}

/// Build a `CgroupStats` populated with the four primary
/// cpu / memory counter fields used in compare-pipeline
/// tests. Helper because the nested-struct shape makes
/// Default + per-field-assignment noisy at every test
/// fixture; this keeps call-site brevity at the four
/// counter values that drive most compare assertions.
pub(super) fn simple_cgroup_stats(
    cpu_usage_usec: u64,
    nr_throttled: u64,
    throttled_usec: u64,
    memory_current: u64,
) -> CgroupStats {
    let mut cs = CgroupStats::default();
    cs.cpu.usage_usec = cpu_usage_usec;
    cs.cpu.nr_throttled = nr_throttled;
    cs.cpu.throttled_usec = throttled_usec;
    cs.memory.current = memory_current;
    cs
}

/// Test-only helper: look up a registry entry by name and
/// return a static reference. Reduces fixture duplication
/// across the metric_display_name + tag tests below.
pub(super) fn lookup_metric(name: &str) -> &'static CtprofMetricDef {
    CTPROF_METRICS
        .iter()
        .find(|m| m.name == name)
        .unwrap_or_else(|| panic!("metric {name} registered"))
}

/// Build a one-thread snapshot pair where every column has
/// a meaningful value. Used by the display-format /
/// column-set tests below.
pub(super) fn snap_pair_for_display() -> (CtprofSnapshot, CtprofSnapshot) {
    let mut a = make_thread("p", "w");
    a.run_time_ns = MonotonicNs(100);
    a.wait_count = MonotonicCount(4);
    a.wait_sum = MonotonicNs(1000);
    let mut b = make_thread("p", "w");
    b.run_time_ns = MonotonicNs(200);
    b.wait_count = MonotonicCount(4);
    b.wait_sum = MonotonicNs(2000);
    (snap_with(vec![a]), snap_with(vec![b]))
}

/// Helper: build a leader thread with a populated smaps_rollup
/// map. The `tid == tgid` shape lets the leader-dedup gate
/// inside `collect_smaps_rollup` admit the row.
pub(super) fn smaps_thread(pcomm: &str, tgid: u32, rss_kb: u64, pss_kb: u64) -> ThreadState {
    let mut t = ThreadState {
        tid: tgid,
        tgid,
        pcomm: pcomm.into(),
        comm: pcomm.into(),
        cgroup: "/".into(),
        ..ThreadState::default()
    };
    t.smaps_rollup_kb.insert("Rss".into(), rss_kb);
    t.smaps_rollup_kb.insert("Pss".into(), pss_kb);
    t
}

/// Build a snapshot with `n` distinct thread types under a
/// single cgroup. Each thread carries a unique
/// (pcomm, comm) pair so the cgroup's TypeSet has size `n`.
/// Used by the fudge-threshold tests below to exercise the
/// 10-type set-size gate at exact, below, and above
/// boundaries.
///
/// Pcomms are chosen from a fixed alphabetic vocabulary so
/// each one classifies through `pattern_key` to its literal
/// (no shared `prefix-{N}` skeleton). With shared digit
/// suffixes the pattern_key normalizer would collapse
/// `worker-0`...`worker-9` into a single `worker-{N}`
/// bucket, breaking the test's "n distinct types"
/// invariant.
/// Greek-letter words that pattern_key keeps as literals
/// (no shared `prefix-{N}` skeleton). Used by the fudge
/// tests so each thread's (pcomm, comm) pair stays a
/// distinct entry in the cgroup's TypeSet — `worker-0`...
/// `worker-9` collapse into `worker-{N}` under
/// `pattern_key` and would break the "n distinct types"
/// invariant the threshold tests depend on.
///
/// The same vocabulary works for cgroup path components
/// when used as full segments — `/svc-alpha` and
/// `/svc-beta` survive the cgroup normalization because
/// `alpha`/`beta` classify to themselves (pure literals).
/// Avoid `/svc/v1`-style paths in tests: the `v1` token
/// normalizes to `v{N}`, so `/svc/v1` and `/svc/v2` BOTH
/// collapse to `/svc/v{N}` and would match as the same
/// cgroup (defeating the "different cgroups" precondition
/// fudge depends on).
pub(super) const FUDGE_WORDS: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
    "chi", "psi", "omega",
];

pub(super) fn fudge_snap(cgroup: &str, n: usize, _pcomm_prefix: &str) -> CtprofSnapshot {
    assert!(
        n <= FUDGE_WORDS.len(),
        "fudge_snap: requested n={n} exceeds the literal-pcomm vocabulary",
    );
    let mut threads = Vec::new();
    for (i, word) in FUDGE_WORDS.iter().enumerate().take(n) {
        let pcomm = word.to_string();
        let comm = format!("{word}-w");
        let mut t = make_thread(&pcomm, &comm);
        t.tid = (1000 + i) as u32;
        t.tgid = t.tid;
        t.cgroup = cgroup.into();
        threads.push(t);
    }
    snap_with(threads)
}

/// Compose a CtprofSnapshot from N already-built per-cgroup
/// snapshots by concatenating their thread vectors. Each
/// input snapshot is expected to have already been built
/// with `fudge_snap`.
pub(super) fn fudge_compose(snaps: Vec<CtprofSnapshot>) -> CtprofSnapshot {
    let mut threads = Vec::new();
    for snap in snaps {
        threads.extend(snap.threads);
    }
    snap_with(threads)
}

/// Drive the compare under `GroupBy::All` (the only mode
/// that activates fudge) with default options.
pub(super) fn fudge_compare(a: &CtprofSnapshot, b: &CtprofSnapshot) -> CtprofDiff {
    compare(
        a,
        b,
        &CompareOptions {
            group_by: GroupBy::All.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    )
}

/// Build a vector of threads under one cgroup using
/// FUDGE_WORDS (literal pcomms that survive `pattern_key`),
/// applying a per-thread mutator. Used by the N:1 merge
/// tests so the merge arms are exercised under realistic
/// 10-distinct-type sets without per-test boilerplate.
pub(super) fn fudge_threads_with<F: FnMut(&mut ThreadState)>(
    cgroup: &str,
    n: usize,
    mut tweak: F,
) -> Vec<ThreadState> {
    assert!(
        n <= FUDGE_WORDS.len(),
        "fudge_threads_with: requested n={n} exceeds the literal-pcomm vocabulary",
    );
    let mut threads = Vec::new();
    for (i, word) in FUDGE_WORDS.iter().enumerate().take(n) {
        let mut t = make_thread(word, &format!("{word}-w"));
        t.tid = (1000 + i) as u32;
        t.tgid = t.tid;
        t.cgroup = cgroup.into();
        tweak(&mut t);
        threads.push(t);
    }
    threads
}

#[allow(dead_code)]
pub(super) fn _fudge_helpers_used() {
    let _ = fudge_compose;
}

