//! Group construction, per-group aggregation, and the cgroup-path
//! flatten helpers consumed by [`super::compare`].
//!
//! Three layers:
//!
//! 1. [`build_groups`] / [`build_cgroup_key_map`] — partition a
//!    snapshot's threads by [`super::GroupBy`] axis (pcomm, cgroup,
//!    comm, comm-exact), apply pattern-aware skeleton clustering
//!    where applicable, and produce a `BTreeMap<key, ThreadGroup>`
//!    keyed by the post-tightening join key. The N:1 pattern-bucket
//!    promotion ([`super::CompareOptions::no_thread_normalize`])
//!    determines whether buckets share a token-normalized skeleton
//!    or stay at the literal `pcomm` / `comm` value.
//!
//! 2. [`aggregate`] / [`mode_aggregate`] / [`build_row`] — apply
//!    per-metric reduction rules ([`super::AggRule`]) across a
//!    group's threads to produce [`super::Aggregated`] values, then
//!    materialize one [`super::DiffRow`] per `(group, metric)` pair
//!    with side-by-side baseline + candidate values and the
//!    precomputed delta / pct cells. The Mode arm dispatches
//!    through [`mode_aggregate`] so the categorical empty-bucket
//!    contract lives in one place; the numeric arms delegate to
//!    the typed traits in [`crate::metric_types`]
//!    (`Summable::sum_across`, `Maxable::max_across`,
//!    `Rangeable::range_across`, `Modeable::mode_across`).
//!
//! 3. [`collect_smaps_rollup`] / [`collect_smaps_rollup_hierarchical`]
//!    — pull the per-process smaps_rollup map (from the leader
//!    thread of each tgid) and key by the same pattern as the
//!    primary-table buckets so byte-counts join across snapshots
//!    when PIDs are ephemeral. The hierarchical variant honors
//!    cgroup-flatten patterns and the auto-normalize key map so
//!    smaps rows align with the primary table even after
//!    cgroup-path tightening.
//!
//! [`flatten_cgroup_path`] / [`compile_flatten_patterns`] are
//! the helpers the caller side ([`super::compare`] +
//! [`super::flatten_cgroup_stats`]) uses to apply user-supplied
//! glob patterns against cgroup paths before bucket assignment.

use std::collections::BTreeMap;

use crate::ctprof::{CtprofSnapshot, ThreadState};

use super::{
    AffinitySummary, AggRule, Aggregated, CTPROF_METRICS, CtprofMetricDef, DiffRow, GroupBy,
    ThreadGroup,
    pattern::{cgroup_normalize_skeleton, pattern_key, tighten_group},
};

/// Walk a snapshot's threads and pull non-empty smaps_rollup
/// maps off the leader threads (tid == tgid; non-leader threads
/// land at empty map per the leader-dedup contract).
///
/// Keying:
///
/// - Default normalization (`no_thread_normalize: false`): key is
///   `pattern_key(&t.pcomm)` — pcomm only, the `[tgid]` suffix is
///   DROPPED. The tgid digits would always normalize to `{N}` and
///   add no discriminating signal to the join key, so omitting
///   them makes smaps keys match the primary-table Pcomm group
///   keys exactly (`kworker/{N}:{N}`, `firefox`, `worker-{N}`,
///   etc.).
///
///   No singleton revert. Unlike [`build_groups`], which reverts a
///   pattern_key to the literal name when only one contributor
///   shares the skeleton, `collect_smaps_rollup` always normalizes
///   when normalization is enabled regardless of how many PIDs
///   share the bucket. The reason is structural: smaps keys exist
///   to JOIN baseline vs candidate across snapshots, and PIDs are
///   per-snapshot ephemeral. A singleton-revert path would emit a
///   literal `worker[7]` on baseline and a literal `worker[1234]`
///   on candidate — two never-matching keys — orphaning every
///   cross-snapshot row. The build_groups invariant ("don't
///   advertise a pattern that no peer shares") doesn't apply on
///   the smaps axis because the bucket's role isn't intra-
///   snapshot fleet aggregation; it's cross-snapshot memory
///   diffing.
///
/// - Literal mode (`no_thread_normalize: true`): key is
///   `pcomm[tgid]` so each PID stays attributable to its
///   specific instance. The tradeoff is that rows only join
///   across snapshots when the same process instance ran on
///   both sides — the `[tgid]` is preserved precisely so two
///   distinct PIDs sharing a pcomm don't collide within a
///   snapshot.
///
/// Aggregation: multiple leader threads mapping to the same
/// key (default mode: a fleet of `worker-{N}` parents) SUM
/// their per-field byte counts. `Rss`, `Pss`, `Private_*`,
/// `Shared_*` etc. each accumulate via `saturating_add` —
/// memory quantities are additive across the merged bucket.
/// `saturating_add` mirrors the cumulative-counter merge policy
/// elsewhere in this module (cpu_usage_usec, throttled_usec); a
/// u64 byte-count overflow implies more than 16 EiB of resident
/// memory across the bucket, well past any realistic host.
///
/// Caveat on `Shared_*` aggregation: when multiple PIDs in a
/// merged bucket share physical pages (the COW case for forked
/// children, mmap'd shared libraries, etc.), summing each PID's
/// per-process `Shared_*` reading double-counts the overlapping
/// physical residency. The same double-count exists in the
/// un-aggregated display — the operator already sees `Shared_Clean
/// = 500MiB` listed against two distinct PID rows that happen to
/// share the same library mapping — so the merge introduces no
/// new information loss, just preserves the pre-existing kernel-
/// emission characteristic. `Pss` stays the precise read for a
/// merged bucket's resident footprint because the kernel
/// proportionally divides shared pages across mappers
/// (`fs/proc/task_mmu.c::smap_account`); operators tracking actual
/// memory pressure should prefer `Pss` over `Rss + Shared_*`
/// arithmetic on collapsed buckets.
///
/// Values are converted from kB to bytes via
/// [`ThreadState::smaps_rollup_bytes`] up-front, so the
/// downstream renderer can pass cell values directly into the
/// auto_scale "B" ladder without further unit math.
pub fn collect_smaps_rollup(
    snap: &CtprofSnapshot,
    no_thread_normalize: bool,
) -> BTreeMap<String, BTreeMap<String, u64>> {
    collect_smaps_rollup_inner(snap, no_thread_normalize, false, &[], None)
}

pub fn collect_smaps_rollup_hierarchical(
    snap: &CtprofSnapshot,
    no_thread_normalize: bool,
    flatten: &[glob::Pattern],
    cgroup_key_map: Option<&BTreeMap<String, String>>,
) -> BTreeMap<String, BTreeMap<String, u64>> {
    collect_smaps_rollup_inner(snap, no_thread_normalize, true, flatten, cgroup_key_map)
}

fn collect_smaps_rollup_inner(
    snap: &CtprofSnapshot,
    no_thread_normalize: bool,
    compound_cgroup: bool,
    flatten: &[glob::Pattern],
    cgroup_key_map: Option<&BTreeMap<String, String>>,
) -> BTreeMap<String, BTreeMap<String, u64>> {
    let mut out: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for t in &snap.threads {
        if t.smaps_rollup_kb.is_empty() {
            continue;
        }
        let pcomm_key = if no_thread_normalize {
            format!("{}[{}]", t.pcomm, t.tgid)
        } else {
            pattern_key(&t.pcomm)
        };
        let key = if compound_cgroup {
            let cg = flatten_cgroup_path(&t.cgroup, flatten);
            let cg_key = match cgroup_key_map.and_then(|m| m.get(&cg)) {
                Some(k) => k.clone(),
                None => cg,
            };
            format!("{cg_key}\x00{pcomm_key}")
        } else {
            pcomm_key
        };
        let entry = out.entry(key).or_default();
        for (k, b) in t.smaps_rollup_bytes() {
            entry
                .entry(k.clone())
                .and_modify(|v| *v = v.saturating_add(b.0))
                .or_insert(b.0);
        }
    }
    out
}

/// Build the post-flatten-path → final-tightened-key map for
/// [`GroupBy::Cgroup`] under auto-normalization. Walks the union
/// of paths from both snapshots' threads and `cgroup_stats` so
/// that Layer 3 (tighten) sees every contributor to a given
/// Layer-2 skeleton group. Returns the map keyed by post-flatten
/// path; consumers ([`build_groups`], [`flatten_cgroup_stats`])
/// look up the final key for any path they see.
pub fn build_cgroup_key_map(
    baseline: &CtprofSnapshot,
    candidate: &CtprofSnapshot,
    flatten: &[glob::Pattern],
) -> BTreeMap<String, String> {
    use std::collections::BTreeSet;
    let mut paths: BTreeSet<String> = BTreeSet::new();
    for snap in [baseline, candidate] {
        for t in &snap.threads {
            paths.insert(flatten_cgroup_path(&t.cgroup, flatten));
        }
        for k in snap.cgroup_stats.keys() {
            paths.insert(flatten_cgroup_path(k, flatten));
        }
    }
    // Compute (skeleton, post_l1, tokens) for every path.
    let entries: Vec<(String, String, String, Vec<String>)> = paths
        .into_iter()
        .map(|p| {
            let (skeleton, post_l1, tokens) = cgroup_normalize_skeleton(&p);
            (p, skeleton, post_l1, tokens)
        })
        .collect();
    // Group entries by Layer-2 skeleton.
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, (_, skel, _, _)) in entries.iter().enumerate() {
        groups.entry(skel.clone()).or_default().push(idx);
    }
    // Tighten per group.
    let mut tightened: Vec<String> = vec![String::new(); entries.len()];
    for (skeleton, indices) in &groups {
        if indices.len() < 2 {
            // Singleton — Layer-2 skeleton stays as the key. No
            // member set to compare against.
            for &i in indices {
                tightened[i] = skeleton.clone();
            }
        } else {
            let post_l1_paths: Vec<String> =
                indices.iter().map(|&i| entries[i].2.clone()).collect();
            let member_tokens: Vec<Vec<String>> =
                indices.iter().map(|&i| entries[i].3.clone()).collect();
            let key = tighten_group(&post_l1_paths, &member_tokens);
            for &i in indices {
                tightened[i] = key.clone();
            }
        }
    }
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    for (i, (orig, _, _, _)) in entries.into_iter().enumerate() {
        out.insert(orig, tightened[i].clone());
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_row(
    key: &str,
    display_key: &str,
    n_a: usize,
    n_b: usize,
    metric: &'static CtprofMetricDef,
    a: Aggregated,
    b: Aggregated,
    uptime_pct: Option<f64>,
) -> DiffRow {
    let (delta, delta_pct) = match (a.numeric(), b.numeric()) {
        (Some(va), Some(vb)) => {
            let d = vb - va;
            let pct = if va.abs() > f64::EPSILON {
                Some(d / va)
            } else {
                None
            };
            (Some(d), pct)
        }
        _ => (None, None),
    };
    DiffRow {
        group_key: key.to_string(),
        thread_count_a: n_a,
        thread_count_b: n_b,
        uptime_pct,
        metric_name: metric.name,
        metric_ladder: metric.rule.ladder(),
        baseline: a,
        candidate: b,
        delta,
        delta_pct,
        display_key: display_key.to_string(),
        sort_by_cell: None,
        sort_by_delta: None,
    }
}

pub fn build_groups(
    snap: &CtprofSnapshot,
    group_by: GroupBy,
    flatten: &[glob::Pattern],
    pattern_counts: Option<&BTreeMap<String, usize>>,
    cgroup_key_map: Option<&BTreeMap<String, String>>,
    no_thread_normalize: bool,
) -> BTreeMap<String, ThreadGroup> {
    // Pattern-aware grouping (Comm, Pcomm) needs a frequency pass:
    // pattern keys with only one matching thread revert to the
    // literal name so a lone worker stays ungrouped instead of
    // advertising a `worker-{N}` pattern that no other thread
    // shares. Non-pattern groupings (CommExact, Cgroup) skip the
    // pre-pass.
    //
    // When `pattern_counts` is supplied (production: `compare()`
    // passes the union over baseline+candidate), it is used as
    // the gate. When it is `None` (single-snapshot test
    // ergonomics), this fn computes counts from `snap` alone.
    // Suppressed when `no_thread_normalize` is set — the gate is
    // meaningless once each thread groups by its literal name.
    // Pattern_field selects the thread accessor used by the
    // singleton-revert gate inside the GroupBy::Pcomm / Comm
    // grouping arm. GroupBy::All has its own arm that normalizes
    // both pcomm and comm unconditionally (no singleton revert),
    // so it never reads `pattern_field`. CommExact and Cgroup
    // don't normalize either.
    let pattern_field: Option<fn(&ThreadState) -> &str> = match (group_by, no_thread_normalize) {
        (GroupBy::Comm, false) => Some(|t: &ThreadState| t.comm.as_str()),
        (GroupBy::Pcomm, false) => Some(|t: &ThreadState| t.pcomm.as_str()),
        _ => None,
    };
    let local_counts: Option<BTreeMap<String, usize>> = match (pattern_field, pattern_counts) {
        (Some(field), None) => {
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for t in &snap.threads {
                *counts.entry(pattern_key(field(t))).or_insert(0) += 1;
            }
            Some(counts)
        }
        _ => None,
    };
    let counts_ref: Option<&BTreeMap<String, usize>> = pattern_counts.or(local_counts.as_ref());

    let mut buckets: BTreeMap<String, Vec<&ThreadState>> = BTreeMap::new();
    for t in &snap.threads {
        let key = match group_by {
            GroupBy::All => {
                let cg = flatten_cgroup_path(&t.cgroup, flatten);
                let cg_key = match cgroup_key_map.and_then(|m| m.get(&cg)) {
                    Some(k) => k.clone(),
                    None => cg,
                };
                let pcomm_key = if no_thread_normalize {
                    t.pcomm.clone()
                } else {
                    pattern_key(&t.pcomm)
                };
                let comm_key = if no_thread_normalize {
                    t.comm.clone()
                } else {
                    pattern_key(&t.comm)
                };
                format!("{cg_key}\x00{pcomm_key}\x00{comm_key}")
            }
            // Pcomm and Comm share the same shape: when
            // normalization is enabled, route the chosen field
            // through `pattern_key` and revert singletons to the
            // literal name so a lone process / thread does not
            // advertise a pattern that no other contributor
            // shares. The `pattern_field` accessor (already
            // computed for the local_counts pre-pass) selects
            // pcomm vs comm; under `no_thread_normalize` it is
            // `None` and we group by literal name directly.
            GroupBy::Pcomm | GroupBy::Comm => match pattern_field {
                Some(field) => {
                    let name = field(t);
                    let pk = pattern_key(name);
                    let counts = counts_ref.expect("pattern_counts seeded for Pcomm/Comm");
                    if counts.get(&pk).copied().unwrap_or(0) >= 2 {
                        pk
                    } else {
                        name.to_string()
                    }
                }
                None => {
                    // `no_thread_normalize` set — literal grouping.
                    if group_by == GroupBy::Pcomm {
                        t.pcomm.clone()
                    } else {
                        t.comm.clone()
                    }
                }
            },
            GroupBy::CommExact => t.comm.clone(),
            GroupBy::Cgroup => {
                let post_flatten = flatten_cgroup_path(&t.cgroup, flatten);
                // When auto-normalize is enabled, the cgroup key map
                // (built by `compare()` over the union of paths from
                // both snapshots) maps each post-flatten path to its
                // final tightened key (Layer 1 + 2 + 3). Otherwise,
                // group by post-flatten path verbatim.
                match cgroup_key_map.and_then(|m| m.get(&post_flatten)) {
                    Some(k) => k.clone(),
                    None => post_flatten,
                }
            }
        };
        buckets.entry(key).or_default().push(t);
    }

    let mut out = BTreeMap::new();
    for (key, threads) in buckets {
        let mut metrics = BTreeMap::new();
        for m in CTPROF_METRICS {
            metrics.insert(m.name.to_string(), aggregate(m.rule, &threads));
        }
        let cgroup_stats = if group_by == GroupBy::Cgroup {
            // Pick the first sampled thread's (flattened) cgroup
            // path and look up its enrichment. All threads in the
            // bucket share the flattened key by construction, so
            // the first is representative.
            threads
                .first()
                .and_then(|t| snap.cgroup_stats.get(&t.cgroup).cloned())
        } else {
            None
        };
        // `members` feeds the grex display-label path for
        // normalized `GroupBy::Comm` (literal comms) and
        // `GroupBy::Pcomm` (literal pcomms). Other groupings — and
        // either pattern-aware grouping under
        // `no_thread_normalize` — render the join key directly, so
        // skip the per-bucket name collection (saves a
        // clone-per-thread per-bucket on busy hosts).
        let members: Vec<String> = match pattern_field {
            Some(field) => {
                let mut v: Vec<String> = threads.iter().map(|t| field(t).to_string()).collect();
                v.sort();
                v.dedup();
                v
            }
            None => Vec::new(),
        };
        let valid_starts: Vec<u64> = threads
            .iter()
            .map(|t| t.start_time_clock_ticks)
            .filter(|&t| t > 0)
            .collect();
        let avg_start_ticks = if valid_starts.is_empty() {
            0
        } else {
            valid_starts.iter().sum::<u64>() / valid_starts.len() as u64
        };
        out.insert(
            key.clone(),
            ThreadGroup {
                key,
                thread_count: threads.len(),
                metrics,
                cgroup_stats,
                members,
                avg_start_ticks,
            },
        );
    }
    out
}

/// Aggregate one metric across a slice of threads per its rule.
///
/// Each `Sum*` / `Max*` / `Range*` / `Mode*` arm dispatches
/// through the trait method on the typed newtype defined in
/// [`crate::metric_types`] — `sum_across` for [`Summable`],
/// `max_across` for [`Maxable`], `range_across` for [`Rangeable`],
/// `mode_across` for [`Modeable`] — then unwraps to the
/// untyped scalar that [`Aggregated`] carries today; the
/// unit-aware format dispatch will land in phase 4 and reads
/// the registry's `unit` tag rather than the wrapper type, so
/// `Aggregated` stays scalar-shaped after this phase.
///
/// # Empty-bucket contract
///
/// The trait-level shapes split empty handling differently
/// from the dispatch-level shape:
/// - [`Summable::sum_across`] returns the additive identity
///   (zero) on an empty input — the trait surface itself
///   collapses the empty case. The `Sum*` arms therefore feed
///   straight into [`Aggregated::Sum`] without re-checking.
/// - [`Maxable::max_across`] returns `Option<Self>` (`None`
///   for empty) so callers can distinguish "no contributors"
///   from "all contributors had zero." The dispatch in this
///   function collapses `None` to `Aggregated::Max(0)` at the
///   call boundary so the historical empty-bucket contract on
///   this code path (zero rendered for empty groups) holds
///   regardless of the trait's richer shape.
/// - [`Rangeable::range_across`] returns
///   `Option<Range<Self>>`; the dispatch collapses `None` to
///   `Aggregated::OrdinalRange { min: 0, max: 0 }` at the call
///   boundary.
/// - [`Modeable::mode_across`] returns
///   `Option<(Self, count, total)>`; the dispatch collapses
///   `None` to `Aggregated::Mode { value: "", count: 0, total }`
///   where `total` is the bucket size (which is non-zero only
///   when threads exist but the iterator was emptied — for
///   `aggregate`, total tracks the bucket size directly so the
///   `None` arm always carries `total: threads.len()`).
///
/// Downstream delta math therefore sees a well-defined value
/// at every join boundary regardless of which side of a
/// compare carried zero threads under the bucket key.
///
/// [`Summable`]: crate::metric_types::Summable
/// [`Maxable`]: crate::metric_types::Maxable
/// [`Rangeable`]: crate::metric_types::Rangeable
/// [`Modeable`]: crate::metric_types::Modeable
///
/// Mode-arm dispatch helper used by `aggregate`. Routes a typed
/// iterator of [`crate::metric_types::CategoricalString`] through
/// `mode_across`, then projects the result onto
/// [`Aggregated::Mode`] with the supplied `total` (the number of
/// threads in the bucket). Empty buckets surface as
/// `Aggregated::Mode { value: "", count: 0, total }` matching the
/// historical empty-bucket contract — downstream delta math sees
/// a well-defined value at the join boundary regardless of which
/// side carried zero threads. Lifts the otherwise-identical
/// match arms for [`AggRule::Mode`], [`AggRule::ModeChar`], and
/// [`AggRule::ModeBool`] into one site so a future refactor that
/// changes the empty-bucket contract or the `mode_across` return
/// shape only edits one place.
fn mode_aggregate(
    total: usize,
    items: impl IntoIterator<Item = crate::metric_types::CategoricalString>,
) -> Aggregated {
    // Build the full tally map across the bucket — one entry
    // per distinct category, with its occurrence count. The
    // mode (most-frequent value) is derived on demand by
    // [`Aggregated::mode_value`] / [`Aggregated::mode_count`];
    // storing the full distribution (not just the mode) lets
    // the N:1 fudge merge compose tallies correctly across
    // buckets via [`merge_aggregated_into`].
    let mut tallies: BTreeMap<String, usize> = BTreeMap::new();
    for item in items {
        *tallies.entry(item.0).or_insert(0) += 1;
    }
    Aggregated::Mode { tallies, total }
}

pub fn aggregate(rule: AggRule, threads: &[&ThreadState]) -> Aggregated {
    // `Modeable` is imported in `mode_aggregate`; the Mode arms
    // route through that helper so the trait doesn't need to be
    // in scope here. `CategoricalString` is still needed because
    // the ModeChar / ModeBool arms construct one for the
    // coercion path before passing the iterator to
    // `mode_aggregate`.
    use crate::metric_types::{CategoricalString, Maxable, Rangeable, Summable};
    match rule {
        AggRule::SumCount(f) => {
            let s = crate::metric_types::MonotonicCount::sum_across(threads.iter().map(|t| f(t)));
            Aggregated::Sum(s.0)
        }
        AggRule::SumNs(f) => {
            let s = crate::metric_types::MonotonicNs::sum_across(threads.iter().map(|t| f(t)));
            Aggregated::Sum(s.0)
        }
        AggRule::SumTicks(f) => {
            let s = crate::metric_types::ClockTicks::sum_across(threads.iter().map(|t| f(t)));
            Aggregated::Sum(s.0)
        }
        AggRule::SumBytes(f) => {
            let s = crate::metric_types::Bytes::sum_across(threads.iter().map(|t| f(t)));
            Aggregated::Sum(s.0)
        }
        AggRule::MaxPeak(f) => {
            // `max_across` returns `Option<Self>` so callers can
            // distinguish "empty thread bucket" from "all
            // contributors had zero." The historical empty-bucket
            // contract on this code path was `Aggregated::Max(0)`;
            // preserve it by collapsing `None` to the additive
            // identity at the call boundary. Non-empty buckets
            // produce a concrete max regardless of value.
            let m = crate::metric_types::PeakNs::max_across(threads.iter().map(|t| f(t)));
            Aggregated::Max(m.map(|v| v.0).unwrap_or(0))
        }
        AggRule::MaxPeakBytes(f) => {
            // Same Option<Self> + None → Aggregated::Max(0)
            // collapse as MaxPeak; the difference is only the
            // typed accessor's unit family — Bytes vs Ns. The
            // ladder() match maps this variant to
            // ScaleLadder::Bytes so the renderer auto-scales
            // with KiB/MiB/GiB/TiB suffixes.
            let m = crate::metric_types::PeakBytes::max_across(threads.iter().map(|t| f(t)));
            Aggregated::Max(m.map(|v| v.0).unwrap_or(0))
        }
        AggRule::MaxGaugeNs(f) => {
            let m = crate::metric_types::GaugeNs::max_across(threads.iter().map(|t| f(t)));
            Aggregated::Max(m.map(|v| v.0).unwrap_or(0))
        }
        AggRule::MaxGaugeCount(f) => {
            let m = crate::metric_types::GaugeCount::max_across(threads.iter().map(|t| f(t)));
            Aggregated::Max(m.map(|v| v.0).unwrap_or(0))
        }
        AggRule::RangeI32(f) => {
            match crate::metric_types::OrdinalI32::range_across(threads.iter().map(|t| f(t))) {
                // `range_across` returns `None` only on an empty
                // iterator — mirror the historical empty-group
                // contract by collapsing to (0, 0) so the
                // downstream midpoint and delta math sees a
                // well-defined value at the join boundary. The
                // `Some` arm carries a typed `Range<OrdinalI32>`
                // wrapper that guarantees min ≤ max as a
                // type-system invariant; `into_tuple()` extracts
                // the pair without re-checking.
                Some(r) => {
                    let (min, max) = r.into_tuple();
                    Aggregated::OrdinalRange {
                        min: i64::from(min.0),
                        max: i64::from(max.0),
                    }
                }
                None => Aggregated::OrdinalRange { min: 0, max: 0 },
            }
        }
        AggRule::RangeU32(f) => {
            match crate::metric_types::OrdinalU32::range_across(threads.iter().map(|t| f(t))) {
                Some(r) => {
                    let (min, max) = r.into_tuple();
                    Aggregated::OrdinalRange {
                        min: i64::from(min.0),
                        max: i64::from(max.0),
                    }
                }
                None => Aggregated::OrdinalRange { min: 0, max: 0 },
            }
        }
        AggRule::Mode(f) => mode_aggregate(threads.len(), threads.iter().map(|t| f(t))),
        AggRule::ModeChar(f) => mode_aggregate(
            threads.len(),
            // `char` is not Modeable directly; coerce to the
            // CategoricalString reduction so the lex-tiebreak
            // contract is identical to other Mode variants.
            threads.iter().map(|t| CategoricalString(f(t).to_string())),
        ),
        AggRule::ModeBool(f) => mode_aggregate(
            threads.len(),
            // Same coercion path as `ModeChar`. `to_string()`
            // produces `"true"`/`"false"` per `bool::Display`.
            threads.iter().map(|t| CategoricalString(f(t).to_string())),
        ),
        AggRule::Affinity(f) => {
            let mut seen: Vec<Vec<u32>> = Vec::new();
            let mut min_cpus = usize::MAX;
            let mut max_cpus = 0usize;
            for t in threads {
                let cpus = f(t).0;
                min_cpus = min_cpus.min(cpus.len());
                max_cpus = max_cpus.max(cpus.len());
                if !seen.iter().any(|s| s == &cpus) {
                    seen.push(cpus);
                }
            }
            if threads.is_empty() {
                min_cpus = 0;
            }
            let uniform = if seen.len() == 1 {
                seen.into_iter().next()
            } else {
                None
            };
            Aggregated::Affinity(AffinitySummary {
                min_cpus,
                max_cpus,
                uniform,
            })
        }
    }
}

/// Collapse dynamic segments of a cgroup path per every pattern
/// in `patterns`. A pattern is a glob (`*` matches one segment,
/// `**` matches multiple) where the literal portions are preserved
/// and the wildcard portions are replaced with the wildcard token
/// itself. Example: pattern `/kubepods/*/workload` applied to
/// `/kubepods/pod-abc/workload` produces `/kubepods/*/workload`,
/// so two runs with different pod IDs collapse onto the same key.
///
/// Patterns are tried in listed order; the first match wins and
/// subsequent patterns are not applied. A path that matches no
/// pattern is returned verbatim.
pub fn flatten_cgroup_path(path: &str, patterns: &[glob::Pattern]) -> String {
    for p in patterns {
        if p.matches(path) {
            // The pattern itself becomes the canonical key: every
            // path matching `/kubepods/*/workload` collapses onto
            // the literal pattern string.
            return p.as_str().to_string();
        }
    }
    path.to_string()
}

pub fn compile_flatten_patterns(raw: &[String]) -> Vec<glob::Pattern> {
    raw.iter()
        .filter_map(|s| glob::Pattern::new(s).ok())
        .collect()
}
