//! Join, aggregate, and render the comparison between two
//! [`HostStateSnapshot`]s.
//!
//! Design summary: the per-thread profiler emits
//! one snapshot per run. Comparison groups threads within each
//! snapshot by `(pcomm, comm)` (or by cgroup / comm, see
//! [`GroupBy`]), aggregates every metric per the rule on its
//! [`HostStateMetricDef`], then matches groups across the two
//! snapshots and emits one row per `(group, metric)` pair. Groups
//! present on only one side surface as unmatched entries rather
//! than imaginary zero-valued rows — a row is missing because
//! the process did not exist, not because it did zero work.
//!
//! No judgment labels. The comparison prints raw numbers and
//! percent delta; interpretation (regression vs improvement) is
//! scheduler-specific and left to the user. This mirrors the
//! no-label principle for the broader stats comparison pipeline
//! (see the `stats.rs` module doc).

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use anyhow::Context;

use crate::host_state::{CgroupStats, HostStateSnapshot, ThreadState};

/// Grouping key for the host-state compare.
///
/// The default is [`GroupBy::Pcomm`] — aggregate every thread
/// belonging to the same process name together. The other
/// variants exist for operators who want to slice along a
/// different axis: `Cgroup` groups by cgroup path (useful for
/// container-per-workload deployments), `Comm` groups by thread
/// name across every process (useful when a thread-pool name
/// like `tokio-worker` spans many binaries).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum GroupBy {
    /// Group by process name (`pcomm`). Default.
    Pcomm,
    /// Group by cgroup path. Cgroup-level enrichment is surfaced
    /// in the output alongside the aggregated thread metrics.
    Cgroup,
    /// Group by thread name (`comm`) across every process.
    Comm,
}

/// Options controlling [`compare`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CompareOptions {
    pub group_by: GroupByOrDefault,
    /// Glob patterns that collapse dynamic cgroup path segments
    /// to a canonical form before grouping. Applied in listed
    /// order; each pattern that matches a thread's cgroup path
    /// rewrites the matched segments with the literal portions
    /// of the pattern. See [`flatten_cgroup_path`] for the
    /// rewrite rule and examples.
    pub cgroup_flatten: Vec<String>,
}

/// Newtype wrapper around [`GroupBy`] that defaults to
/// [`GroupBy::Pcomm`]. Separate type so `CompareOptions::default()`
/// does not need to spell out every field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupByOrDefault(pub GroupBy);

impl Default for GroupByOrDefault {
    fn default() -> Self {
        Self(GroupBy::Pcomm)
    }
}

impl From<GroupBy> for GroupByOrDefault {
    fn from(g: GroupBy) -> Self {
        Self(g)
    }
}

/// Aggregation rule for a single metric.
///
/// Encoded as an enum rather than a trait object so the registry
/// table ([`HOST_STATE_METRICS`]) can live in static memory. Each
/// variant carries the reader that extracts the per-thread value
/// — the reader and rule are paired by construction so a new
/// metric cannot register a sum rule against a string accessor.
#[derive(Debug, Clone, Copy)]
pub enum AggRule {
    /// Sum across the group. Used for cumulative counters
    /// (run_time, faults, I/O). Delta is the signed difference
    /// between candidate and baseline sums.
    Sum(fn(&ThreadState) -> u64),
    /// Ordinal integer, aggregated as the observed [min, max].
    /// Delta uses the midpoint of each range as the scalar;
    /// output prints both the range and the delta.
    OrdinalRange(fn(&ThreadState) -> i64),
    /// Categorical string, aggregated as the mode (most frequent)
    /// value. Delta is textual: "same" if both modes agree,
    /// "differs" otherwise — there is no arithmetic on a policy
    /// name.
    Mode(fn(&ThreadState) -> String),
    /// CPU affinity set. Aggregated as num_cpus range across the
    /// group plus a uniform-cpuset rendering when every thread
    /// shared the same allowed set.
    Affinity(fn(&ThreadState) -> Vec<u32>),
}

/// One metric exposed by the comparison pipeline.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct HostStateMetricDef {
    pub name: &'static str,
    /// Display unit appended to the numeric value in output
    /// cells. Empty string means "unitless count".
    pub unit: &'static str,
    pub rule: AggRule,
}

/// Registry of per-thread metrics. Order here is the default
/// display order for rows that have no numeric delta to sort by
/// (ties fall back to registry order). Names are the ASCII
/// short-form used in capture code; long-form display is the
/// same — no translation layer.
pub static HOST_STATE_METRICS: &[HostStateMetricDef] = &[
    // identity / structural (non-numeric aggregation)
    HostStateMetricDef {
        name: "policy",
        unit: "",
        rule: AggRule::Mode(|t| t.policy.clone()),
    },
    HostStateMetricDef {
        name: "nice",
        unit: "",
        rule: AggRule::OrdinalRange(|t| t.nice as i64),
    },
    HostStateMetricDef {
        name: "cpu_affinity",
        unit: "",
        rule: AggRule::Affinity(|t| t.cpu_affinity.clone()),
    },
    // scheduling
    HostStateMetricDef {
        name: "run_time_ns",
        unit: "ns",
        rule: AggRule::Sum(|t| t.run_time_ns),
    },
    HostStateMetricDef {
        name: "wait_time_ns",
        unit: "ns",
        rule: AggRule::Sum(|t| t.wait_time_ns),
    },
    HostStateMetricDef {
        name: "timeslices",
        unit: "",
        rule: AggRule::Sum(|t| t.timeslices),
    },
    HostStateMetricDef {
        name: "voluntary_csw",
        unit: "",
        rule: AggRule::Sum(|t| t.voluntary_csw),
    },
    HostStateMetricDef {
        name: "nonvoluntary_csw",
        unit: "",
        rule: AggRule::Sum(|t| t.nonvoluntary_csw),
    },
    HostStateMetricDef {
        name: "nr_wakeups",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_wakeups),
    },
    HostStateMetricDef {
        name: "nr_wakeups_local",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_wakeups_local),
    },
    HostStateMetricDef {
        name: "nr_wakeups_remote",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_wakeups_remote),
    },
    HostStateMetricDef {
        name: "nr_wakeups_sync",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_wakeups_sync),
    },
    HostStateMetricDef {
        name: "nr_wakeups_migrate",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_wakeups_migrate),
    },
    HostStateMetricDef {
        name: "nr_wakeups_idle",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_wakeups_idle),
    },
    HostStateMetricDef {
        name: "nr_migrations",
        unit: "",
        rule: AggRule::Sum(|t| t.nr_migrations),
    },
    HostStateMetricDef {
        name: "wait_sum",
        unit: "ns",
        rule: AggRule::Sum(|t| t.wait_sum),
    },
    HostStateMetricDef {
        name: "wait_count",
        unit: "",
        rule: AggRule::Sum(|t| t.wait_count),
    },
    HostStateMetricDef {
        name: "sleep_sum",
        unit: "ns",
        rule: AggRule::Sum(|t| t.sleep_sum),
    },
    // No `sleep_count` metric: the kernel does not emit
    // that counter. The prior entry matched a ghost field
    // that never populated; removed alongside the parser
    // key fix (`sum_sleep_runtime` is the real key for
    // sleep_sum, and there is no per-event counter pair).
    HostStateMetricDef {
        name: "block_sum",
        unit: "ns",
        rule: AggRule::Sum(|t| t.block_sum),
    },
    HostStateMetricDef {
        name: "block_count",
        unit: "",
        rule: AggRule::Sum(|t| t.block_count),
    },
    HostStateMetricDef {
        name: "iowait_sum",
        unit: "ns",
        rule: AggRule::Sum(|t| t.iowait_sum),
    },
    HostStateMetricDef {
        name: "iowait_count",
        unit: "",
        rule: AggRule::Sum(|t| t.iowait_count),
    },
    // memory
    HostStateMetricDef {
        name: "allocated_bytes",
        unit: "B",
        rule: AggRule::Sum(|t| t.allocated_bytes),
    },
    HostStateMetricDef {
        name: "deallocated_bytes",
        unit: "B",
        rule: AggRule::Sum(|t| t.deallocated_bytes),
    },
    HostStateMetricDef {
        name: "minflt",
        unit: "",
        rule: AggRule::Sum(|t| t.minflt),
    },
    HostStateMetricDef {
        name: "majflt",
        unit: "",
        rule: AggRule::Sum(|t| t.majflt),
    },
    // I/O
    HostStateMetricDef {
        name: "rchar",
        unit: "B",
        rule: AggRule::Sum(|t| t.rchar),
    },
    HostStateMetricDef {
        name: "wchar",
        unit: "B",
        rule: AggRule::Sum(|t| t.wchar),
    },
    HostStateMetricDef {
        name: "syscr",
        unit: "",
        rule: AggRule::Sum(|t| t.syscr),
    },
    HostStateMetricDef {
        name: "syscw",
        unit: "",
        rule: AggRule::Sum(|t| t.syscw),
    },
    HostStateMetricDef {
        name: "read_bytes",
        unit: "B",
        rule: AggRule::Sum(|t| t.read_bytes),
    },
    HostStateMetricDef {
        name: "write_bytes",
        unit: "B",
        rule: AggRule::Sum(|t| t.write_bytes),
    },
];

/// Aggregated metric value for a single [`ThreadGroup`].
///
/// Carries both a numeric projection (used for delta math and
/// sort order) and a display form. Not every rule produces a
/// numeric — [`AggRule::Mode`] aggregates a policy string, which
/// has no scalar — so the numeric is optional and rows without
/// one fall to the bottom of the default sort.
#[derive(Debug, Clone)]
pub enum Aggregated {
    Sum(u64),
    OrdinalRange { min: i64, max: i64 },
    Mode { value: String, count: usize, total: usize },
    Affinity(AffinitySummary),
}

/// CPU-affinity aggregation result.
///
/// `uniform` is `Some(cpus)` when every thread in the group shared
/// the same allowed set; otherwise heterogeneous and the renderer
/// emits "N-M cpus (mixed)".
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AffinitySummary {
    pub min_cpus: usize,
    pub max_cpus: usize,
    pub uniform: Option<Vec<u32>>,
}

impl Aggregated {
    /// Scalar projection for delta math. `None` when the rule
    /// produces no meaningful scalar (categorical mode, affinity
    /// with heterogeneous cpusets).
    pub fn numeric(&self) -> Option<f64> {
        match self {
            Aggregated::Sum(v) => Some(*v as f64),
            Aggregated::OrdinalRange { min, max } => {
                // Midpoint: keeps a min→max shift on one end visible
                // in the delta without privileging either bound.
                Some((*min as f64 + *max as f64) / 2.0)
            }
            Aggregated::Mode { .. } => None,
            Aggregated::Affinity(s) => {
                // Number of allowed CPUs is the natural scalar. When
                // the group is uniform, `min_cpus == max_cpus`; when
                // heterogeneous, midpoint parallels OrdinalRange.
                Some((s.min_cpus as f64 + s.max_cpus as f64) / 2.0)
            }
        }
    }
}

impl fmt::Display for Aggregated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Aggregated::Sum(v) => write!(f, "{v}"),
            Aggregated::OrdinalRange { min, max } => {
                if min == max {
                    write!(f, "{min}")
                } else {
                    write!(f, "{min}..{max}")
                }
            }
            Aggregated::Mode {
                value,
                count,
                total,
            } => {
                if count == total {
                    write!(f, "{value}")
                } else {
                    write!(f, "{value} ({count}/{total})")
                }
            }
            Aggregated::Affinity(s) => {
                if let Some(cpus) = &s.uniform {
                    let n = cpus.len();
                    let range = format_cpu_range(cpus);
                    write!(f, "{n} cpus ({range})")
                } else if s.min_cpus == s.max_cpus {
                    write!(f, "{} cpus (mixed)", s.min_cpus)
                } else {
                    write!(f, "{}-{} cpus (mixed)", s.min_cpus, s.max_cpus)
                }
            }
        }
    }
}

/// Aggregated metrics for every thread matched by one group key.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ThreadGroup {
    pub key: String,
    pub thread_count: usize,
    /// Metric name → aggregated value. Entries are created for
    /// every registered metric; absent keys signal a missed
    /// aggregation step, not a skip.
    pub metrics: BTreeMap<String, Aggregated>,
    /// Only populated when grouping by cgroup — carries the cgroup
    /// v2 enrichment counters (cpu.stat, memory.current) for that
    /// path. Nested here so the renderer can surface them
    /// alongside the thread-metric rows without a second lookup.
    pub cgroup_stats: Option<CgroupStats>,
}

/// One row in the comparison table: `(group, metric)` pair with
/// aggregated values from both sides.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DiffRow {
    pub group_key: String,
    pub thread_count_a: usize,
    pub thread_count_b: usize,
    pub metric_name: &'static str,
    pub metric_unit: &'static str,
    pub baseline: Aggregated,
    pub candidate: Aggregated,
    /// Signed candidate − baseline for numeric-capable rules.
    pub delta: Option<f64>,
    /// `delta / baseline` as a fraction. `None` when baseline is
    /// zero or the row has no numeric projection.
    pub delta_pct: Option<f64>,
}

impl DiffRow {
    /// Sort key for "biggest absolute delta %". Numeric rows
    /// with a non-zero baseline sort by `|delta_pct|`; numeric
    /// rows with a zero baseline sort by `|delta|` scaled by a
    /// large constant so any non-zero candidate dominates
    /// percent-based rows; non-numeric rows sink to the bottom.
    fn sort_key(&self) -> f64 {
        if let Some(p) = self.delta_pct {
            p.abs()
        } else if let Some(d) = self.delta {
            // Baseline was zero (delta_pct undefined) but candidate
            // is some value — still a visible change. Inflate so it
            // beats percent-only rows in the sort.
            d.abs() * 1e9
        } else {
            f64::NEG_INFINITY
        }
    }
}

/// Full comparison result.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct HostStateDiff {
    pub rows: Vec<DiffRow>,
    /// Group keys that appeared in the baseline snapshot but not
    /// in the candidate.
    pub only_baseline: Vec<String>,
    /// Group keys that appeared in the candidate snapshot but not
    /// in the baseline.
    pub only_candidate: Vec<String>,
    /// Baseline-only cgroup-level enrichment rows, keyed by the
    /// cgroup path (after flatten). Populated only for
    /// [`GroupBy::Cgroup`].
    pub cgroup_stats_a: BTreeMap<String, CgroupStats>,
    /// Candidate-only cgroup-level enrichment rows, same shape.
    pub cgroup_stats_b: BTreeMap<String, CgroupStats>,
}

/// Compare two snapshots and produce a [`HostStateDiff`].
pub fn compare(
    baseline: &HostStateSnapshot,
    candidate: &HostStateSnapshot,
    opts: &CompareOptions,
) -> HostStateDiff {
    let flatten = compile_flatten_patterns(&opts.cgroup_flatten);
    let group_by = opts.group_by.0;
    let groups_a = build_groups(baseline, group_by, &flatten);
    let groups_b = build_groups(candidate, group_by, &flatten);

    let mut diff = HostStateDiff::default();

    for (key, group_a) in &groups_a {
        let Some(group_b) = groups_b.get(key) else {
            diff.only_baseline.push(key.clone());
            continue;
        };
        for metric in HOST_STATE_METRICS {
            let Some(a) = group_a.metrics.get(metric.name).cloned() else {
                continue;
            };
            let Some(b) = group_b.metrics.get(metric.name).cloned() else {
                continue;
            };
            diff.rows.push(build_row(
                key,
                group_a.thread_count,
                group_b.thread_count,
                metric,
                a,
                b,
            ));
        }
    }
    for key in groups_b.keys() {
        if !groups_a.contains_key(key) {
            diff.only_candidate.push(key.clone());
        }
    }
    diff.only_baseline.sort();
    diff.only_candidate.sort();

    // Stable-sort by descending |delta_pct|, ties broken by
    // ascending group_key + registry order of metric.
    diff.rows.sort_by(|a, b| {
        b.sort_key()
            .partial_cmp(&a.sort_key())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.group_key.cmp(&b.group_key))
    });

    if group_by == GroupBy::Cgroup {
        diff.cgroup_stats_a = flatten_cgroup_stats(&baseline.cgroup_stats, &flatten);
        diff.cgroup_stats_b = flatten_cgroup_stats(&candidate.cgroup_stats, &flatten);
    }

    diff
}

fn build_row(
    key: &str,
    n_a: usize,
    n_b: usize,
    metric: &'static HostStateMetricDef,
    a: Aggregated,
    b: Aggregated,
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
        metric_name: metric.name,
        metric_unit: metric.unit,
        baseline: a,
        candidate: b,
        delta,
        delta_pct,
    }
}

fn build_groups(
    snap: &HostStateSnapshot,
    group_by: GroupBy,
    flatten: &[glob::Pattern],
) -> BTreeMap<String, ThreadGroup> {
    let mut buckets: BTreeMap<String, Vec<&ThreadState>> = BTreeMap::new();
    for t in &snap.threads {
        let key = match group_by {
            GroupBy::Pcomm => t.pcomm.clone(),
            GroupBy::Comm => t.comm.clone(),
            GroupBy::Cgroup => flatten_cgroup_path(&t.cgroup, flatten),
        };
        buckets.entry(key).or_default().push(t);
    }

    let mut out = BTreeMap::new();
    for (key, threads) in buckets {
        let mut metrics = BTreeMap::new();
        for m in HOST_STATE_METRICS {
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
        out.insert(
            key.clone(),
            ThreadGroup {
                key,
                thread_count: threads.len(),
                metrics,
                cgroup_stats,
            },
        );
    }
    out
}

/// Aggregate one metric across a slice of threads per its rule.
pub fn aggregate(rule: AggRule, threads: &[&ThreadState]) -> Aggregated {
    match rule {
        AggRule::Sum(f) => {
            let s = threads.iter().map(|t| f(t)).fold(0u64, u64::saturating_add);
            Aggregated::Sum(s)
        }
        AggRule::OrdinalRange(f) => {
            let mut it = threads.iter().map(|t| f(t));
            let first = it.next().unwrap_or(0);
            let (mut min, mut max) = (first, first);
            for v in it {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            Aggregated::OrdinalRange { min, max }
        }
        AggRule::Mode(f) => {
            let total = threads.len();
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            for t in threads {
                *counts.entry(f(t)).or_default() += 1;
            }
            // Largest count wins; ties broken by lexicographic
            // order so the output is deterministic.
            let (value, count) = counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                .unwrap_or_else(|| (String::new(), 0));
            Aggregated::Mode {
                value,
                count,
                total,
            }
        }
        AggRule::Affinity(f) => {
            let mut seen: Vec<Vec<u32>> = Vec::new();
            let mut min_cpus = usize::MAX;
            let mut max_cpus = 0usize;
            for t in threads {
                let cpus = f(t);
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

fn compile_flatten_patterns(raw: &[String]) -> Vec<glob::Pattern> {
    raw.iter()
        .filter_map(|s| glob::Pattern::new(s).ok())
        .collect()
}

fn flatten_cgroup_stats(
    stats: &BTreeMap<String, CgroupStats>,
    patterns: &[glob::Pattern],
) -> BTreeMap<String, CgroupStats> {
    // When multiple input paths flatten to the same key, sum the
    // cpu/throttle counters (cumulative) and keep the max of
    // memory.current (instantaneous — summing overstates the
    // instantaneous RSS of the shared bucket). This mirrors the
    // per-thread aggregation: counters sum, instantaneous values
    // take a representative scalar.
    let mut out: BTreeMap<String, CgroupStats> = BTreeMap::new();
    for (path, cs) in stats {
        let key = flatten_cgroup_path(path, patterns);
        let agg = out.entry(key).or_default();
        agg.cpu_usage_usec = agg.cpu_usage_usec.saturating_add(cs.cpu_usage_usec);
        agg.nr_throttled = agg.nr_throttled.saturating_add(cs.nr_throttled);
        agg.throttled_usec = agg.throttled_usec.saturating_add(cs.throttled_usec);
        agg.memory_current = agg.memory_current.max(cs.memory_current);
    }
    out
}

fn format_cpu_range(cpus: &[u32]) -> String {
    // Collapse contiguous runs to `a-b`, join with commas. Assumes
    // sorted ascending; capture layer stores sorted cpusets.
    if cpus.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let mut start = cpus[0];
    let mut prev = cpus[0];
    for &c in &cpus[1..] {
        if c == prev + 1 {
            prev = c;
            continue;
        }
        if !out.is_empty() {
            out.push(',');
        }
        if start == prev {
            out.push_str(&start.to_string());
        } else {
            out.push_str(&format!("{start}-{prev}"));
        }
        start = c;
        prev = c;
    }
    if !out.is_empty() {
        out.push(',');
    }
    if start == prev {
        out.push_str(&start.to_string());
    } else {
        out.push_str(&format!("{start}-{prev}"));
    }
    out
}

/// Arguments for the `ktstr host-state compare` subcommand.
#[derive(Debug, clap::Args)]
pub struct HostStateCompareArgs {
    /// Baseline snapshot (`.hst.zst`) from `ktstr host-state -o`.
    pub baseline: std::path::PathBuf,
    /// Candidate snapshot (`.hst.zst`) from `ktstr host-state -o`.
    pub candidate: std::path::PathBuf,
    /// Grouping key. `pcomm` (default) aggregates per process
    /// name; `cgroup` per cgroup path; `comm` per thread name
    /// across every process.
    #[arg(long, value_enum, default_value_t = GroupBy::Pcomm)]
    pub group_by: GroupBy,
    /// Glob patterns that collapse dynamic cgroup path segments
    /// so structurally-equivalent cgroups across runs join
    /// correctly. Example:
    /// `--cgroup-flatten '/kubepods/*/workload'` treats different
    /// pod IDs as the same group. Repeatable.
    #[arg(long)]
    pub cgroup_flatten: Vec<String>,
}

/// Entry point for the compare CLI. Loads both snapshots,
/// computes the diff, prints the table, and returns `0` on
/// success. Exits non-zero only on I/O or parse errors; a
/// non-empty diff is data, not a failure.
pub fn run_compare(args: &HostStateCompareArgs) -> anyhow::Result<i32> {
    let baseline = HostStateSnapshot::load(&args.baseline)
        .with_context(|| format!("load baseline {}", args.baseline.display()))?;
    let candidate = HostStateSnapshot::load(&args.candidate)
        .with_context(|| format!("load candidate {}", args.candidate.display()))?;

    let opts = CompareOptions {
        group_by: args.group_by.into(),
        cgroup_flatten: args.cgroup_flatten.clone(),
    };
    let diff = compare(&baseline, &candidate, &opts);
    print_diff(&diff, &args.baseline, &args.candidate, args.group_by);
    Ok(0)
}

/// Render [`HostStateDiff`] as a table on stdout. Thin wrapper
/// over [`write_diff`] so the non-test caller keeps the
/// ergonomics of a one-line call; tests drive [`write_diff`]
/// into a `String` buffer.
pub fn print_diff(
    diff: &HostStateDiff,
    baseline_path: &Path,
    candidate_path: &Path,
    group_by: GroupBy,
) {
    let mut out = String::new();
    // Infallible: writing into a String cannot fail.
    let _ = write_diff(&mut out, diff, baseline_path, candidate_path, group_by);
    print!("{out}");
}

/// Render [`HostStateDiff`] into `w`. The formatter layer lives
/// here so tests can inspect exactly what `print_diff` would
/// emit without shelling through stdout capture. Write errors
/// propagate as [`std::fmt::Error`] — callers that write into an
/// infallible sink (`String`) can unwrap or ignore.
pub fn write_diff<W: fmt::Write>(
    w: &mut W,
    diff: &HostStateDiff,
    baseline_path: &Path,
    candidate_path: &Path,
    group_by: GroupBy,
) -> fmt::Result {
    let group_header = match group_by {
        GroupBy::Pcomm => "pcomm",
        GroupBy::Cgroup => "cgroup",
        GroupBy::Comm => "comm",
    };

    let mut table = crate::cli::new_table();
    table.set_header(vec![
        group_header,
        "threads",
        "metric",
        "baseline",
        "candidate",
        "delta",
        "%",
    ]);
    for row in &diff.rows {
        let delta_cell = match row.delta {
            Some(d) => format!("{:+.3}{}", d, row.metric_unit),
            None => match (&row.baseline, &row.candidate) {
                (Aggregated::Mode { value: a, .. }, Aggregated::Mode { value: b, .. }) => {
                    if a == b {
                        "same".to_string()
                    } else {
                        "differs".to_string()
                    }
                }
                _ => "-".to_string(),
            },
        };
        let pct_cell = match row.delta_pct {
            Some(p) => format!("{:+.1}%", p * 100.0),
            None => "-".to_string(),
        };
        let threads_cell = if row.thread_count_a == row.thread_count_b {
            row.thread_count_a.to_string()
        } else {
            format!("{}\u{2192}{}", row.thread_count_a, row.thread_count_b)
        };
        table.add_row(vec![
            row.group_key.clone(),
            threads_cell,
            row.metric_name.to_string(),
            format!("{}{}", row.baseline, row.metric_unit),
            format!("{}{}", row.candidate, row.metric_unit),
            delta_cell,
            pct_cell,
        ]);
    }
    writeln!(w, "{table}")?;

    if group_by == GroupBy::Cgroup
        && (!diff.cgroup_stats_a.is_empty() || !diff.cgroup_stats_b.is_empty())
    {
        writeln!(w)?;
        let mut ct = crate::cli::new_table();
        ct.set_header(vec![
            "cgroup",
            "cpu_usage_usec",
            "nr_throttled",
            "throttled_usec",
            "memory_current",
        ]);
        let mut all_keys: std::collections::BTreeSet<&String> =
            diff.cgroup_stats_a.keys().collect();
        all_keys.extend(diff.cgroup_stats_b.keys());
        for key in all_keys {
            let a = diff.cgroup_stats_a.get(key);
            let b = diff.cgroup_stats_b.get(key);
            ct.add_row(vec![
                key.clone(),
                cgroup_cell(a.map(|s| s.cpu_usage_usec), b.map(|s| s.cpu_usage_usec)),
                cgroup_cell(a.map(|s| s.nr_throttled), b.map(|s| s.nr_throttled)),
                cgroup_cell(a.map(|s| s.throttled_usec), b.map(|s| s.throttled_usec)),
                cgroup_cell(a.map(|s| s.memory_current), b.map(|s| s.memory_current)),
            ]);
        }
        writeln!(w, "{ct}")?;
    }

    if !diff.only_baseline.is_empty() {
        writeln!(
            w,
            "\n{} group(s) only in baseline ({}):",
            diff.only_baseline.len(),
            baseline_path.display()
        )?;
        for k in &diff.only_baseline {
            writeln!(w, "  {k}")?;
        }
    }
    if !diff.only_candidate.is_empty() {
        writeln!(
            w,
            "\n{} group(s) only in candidate ({}):",
            diff.only_candidate.len(),
            candidate_path.display()
        )?;
        for k in &diff.only_candidate {
            writeln!(w, "  {k}")?;
        }
    }
    Ok(())
}

fn cgroup_cell(a: Option<u64>, b: Option<u64>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => {
            let d = b as i128 - a as i128;
            format!("{a} → {b} ({d:+})")
        }
        (Some(a), None) => format!("{a} → -"),
        (None, Some(b)) => format!("- → {b}"),
        (None, None) => "-".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_thread(pcomm: &str, comm: &str) -> ThreadState {
        ThreadState {
            tid: 1,
            tgid: 1,
            pcomm: pcomm.into(),
            comm: comm.into(),
            cgroup: "/".into(),
            start_time_clock_ticks: 0,
            policy: "SCHED_OTHER".into(),
            nice: 0,
            cpu_affinity: vec![0, 1, 2, 3],
            ..ThreadState::default()
        }
    }

    fn snap_with(threads: Vec<ThreadState>) -> HostStateSnapshot {
        HostStateSnapshot {
            captured_at_unix_ns: 0,
            host: None,
            threads,
            cgroup_stats: BTreeMap::new(),
        }
    }

    #[test]
    fn sum_aggregation_totals_across_group() {
        let mut a = make_thread("app", "w1");
        a.run_time_ns = 1_000;
        let mut b = make_thread("app", "w2");
        b.run_time_ns = 3_000;
        let v = aggregate(AggRule::Sum(|t| t.run_time_ns), &[&a, &b]);
        match v {
            Aggregated::Sum(s) => assert_eq!(s, 4_000),
            other => panic!("expected Sum, got {other:?}"),
        }
    }

    #[test]
    fn sum_saturates_on_overflow() {
        let mut a = make_thread("app", "w1");
        a.run_time_ns = u64::MAX;
        let mut b = make_thread("app", "w2");
        b.run_time_ns = 5;
        let v = aggregate(AggRule::Sum(|t| t.run_time_ns), &[&a, &b]);
        match v {
            Aggregated::Sum(s) => assert_eq!(s, u64::MAX),
            other => panic!("expected Sum, got {other:?}"),
        }
    }

    #[test]
    fn ordinal_range_picks_extremes() {
        let mut a = make_thread("app", "w1");
        a.nice = -5;
        let mut b = make_thread("app", "w2");
        b.nice = 10;
        let v = aggregate(AggRule::OrdinalRange(|t| t.nice as i64), &[&a, &b]);
        match v {
            Aggregated::OrdinalRange { min, max } => {
                assert_eq!(min, -5);
                assert_eq!(max, 10);
            }
            other => panic!("expected OrdinalRange, got {other:?}"),
        }
    }

    #[test]
    fn mode_aggregation_picks_most_frequent() {
        let mut a = make_thread("app", "w1");
        a.policy = "SCHED_OTHER".into();
        let mut b = make_thread("app", "w2");
        b.policy = "SCHED_OTHER".into();
        let mut c = make_thread("app", "w3");
        c.policy = "SCHED_FIFO".into();
        let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &[&a, &b, &c]);
        match v {
            Aggregated::Mode {
                value,
                count,
                total,
            } => {
                assert_eq!(value, "SCHED_OTHER");
                assert_eq!(count, 2);
                assert_eq!(total, 3);
            }
            other => panic!("expected Mode, got {other:?}"),
        }
    }

    #[test]
    fn affinity_uniform_preserves_cpuset() {
        let a = make_thread("app", "w1");
        let b = make_thread("app", "w2");
        let v = aggregate(AggRule::Affinity(|t| t.cpu_affinity.clone()), &[&a, &b]);
        match v {
            Aggregated::Affinity(s) => {
                assert_eq!(s.min_cpus, 4);
                assert_eq!(s.max_cpus, 4);
                assert_eq!(s.uniform, Some(vec![0, 1, 2, 3]));
            }
            other => panic!("expected Affinity, got {other:?}"),
        }
    }

    #[test]
    fn affinity_heterogeneous_drops_uniform() {
        let a = make_thread("app", "w1");
        let mut b = make_thread("app", "w2");
        b.cpu_affinity = vec![4, 5];
        let v = aggregate(AggRule::Affinity(|t| t.cpu_affinity.clone()), &[&a, &b]);
        match v {
            Aggregated::Affinity(s) => {
                assert_eq!(s.min_cpus, 2);
                assert_eq!(s.max_cpus, 4);
                assert!(s.uniform.is_none());
            }
            other => panic!("expected Affinity, got {other:?}"),
        }
    }

    #[test]
    fn format_cpu_range_collapses_contiguous_runs() {
        assert_eq!(format_cpu_range(&[0, 1, 2, 3]), "0-3");
        assert_eq!(format_cpu_range(&[0, 1, 4, 5, 7]), "0-1,4-5,7");
        assert_eq!(format_cpu_range(&[3]), "3");
        assert_eq!(format_cpu_range(&[]), "");
    }

    #[test]
    fn flatten_cgroup_path_collapses_via_pattern() {
        let pats = compile_flatten_patterns(&["/kubepods/*/workload".into()]);
        let out = flatten_cgroup_path("/kubepods/pod-abc-123/workload", &pats);
        assert_eq!(out, "/kubepods/*/workload");
    }

    #[test]
    fn flatten_cgroup_path_falls_through_unmatched() {
        let pats = compile_flatten_patterns(&["/kubepods/*/workload".into()]);
        assert_eq!(
            flatten_cgroup_path("/system.slice/sshd.service", &pats),
            "/system.slice/sshd.service",
        );
    }

    #[test]
    fn compare_emits_rows_for_matched_groups() {
        let mut ta = make_thread("app", "w1");
        ta.run_time_ns = 1_000;
        let mut tb = make_thread("app", "w1");
        tb.run_time_ns = 2_000;
        let a = snap_with(vec![ta]);
        let b = snap_with(vec![tb]);
        let diff = compare(&a, &b, &CompareOptions::default());
        let run_time = diff
            .rows
            .iter()
            .find(|r| r.metric_name == "run_time_ns")
            .expect("run_time_ns row");
        assert_eq!(run_time.group_key, "app");
        assert_eq!(run_time.delta, Some(1_000.0));
        assert!((run_time.delta_pct.unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compare_reports_unmatched_groups() {
        let a = snap_with(vec![make_thread("only_a", "w1")]);
        let b = snap_with(vec![make_thread("only_b", "w1")]);
        let diff = compare(&a, &b, &CompareOptions::default());
        assert_eq!(diff.only_baseline, vec!["only_a".to_string()]);
        assert_eq!(diff.only_candidate, vec!["only_b".to_string()]);
    }

    #[test]
    fn compare_sorts_by_abs_delta_pct_descending() {
        // Build two baseline threads and two candidate threads:
        // "big" swings 10x, "small" swings 1.1x. After compare,
        // the "big" row must sort before "small".
        let mut a1 = make_thread("big", "w");
        a1.run_time_ns = 100;
        let mut a2 = make_thread("small", "w");
        a2.run_time_ns = 1_000;
        let mut b1 = make_thread("big", "w");
        b1.run_time_ns = 1_000;
        let mut b2 = make_thread("small", "w");
        b2.run_time_ns = 1_100;
        let diff = compare(
            &snap_with(vec![a1, a2]),
            &snap_with(vec![b1, b2]),
            &CompareOptions::default(),
        );
        let run_rows: Vec<&DiffRow> = diff
            .rows
            .iter()
            .filter(|r| r.metric_name == "run_time_ns")
            .collect();
        assert_eq!(run_rows[0].group_key, "big");
        assert_eq!(run_rows[1].group_key, "small");
    }

    #[test]
    fn group_by_cgroup_applies_flatten_patterns() {
        let mut ta = make_thread("app", "w1");
        ta.cgroup = "/kubepods/pod-xxx/workload".into();
        ta.run_time_ns = 1_000;
        let mut tb = make_thread("app", "w1");
        tb.cgroup = "/kubepods/pod-yyy/workload".into();
        tb.run_time_ns = 2_000;
        let opts = CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec!["/kubepods/*/workload".into()],
        };
        let diff = compare(&snap_with(vec![ta]), &snap_with(vec![tb]), &opts);
        assert!(diff.only_baseline.is_empty(), "{:?}", diff.only_baseline);
        assert!(
            diff.only_candidate.is_empty(),
            "{:?}",
            diff.only_candidate,
        );
        assert!(
            diff.rows
                .iter()
                .any(|r| r.group_key == "/kubepods/*/workload"),
            "rows={:?}",
            diff.rows.iter().map(|r| &r.group_key).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn group_by_cgroup_surfaces_enrichment_on_diff() {
        let mut ta = make_thread("app", "w1");
        ta.cgroup = "/app".into();
        let mut snap_a = snap_with(vec![ta]);
        snap_a.cgroup_stats.insert(
            "/app".into(),
            CgroupStats {
                cpu_usage_usec: 100,
                nr_throttled: 1,
                throttled_usec: 50,
                memory_current: 1 << 20,
            },
        );
        let mut tb = make_thread("app", "w1");
        tb.cgroup = "/app".into();
        let mut snap_b = snap_with(vec![tb]);
        snap_b.cgroup_stats.insert(
            "/app".into(),
            CgroupStats {
                cpu_usage_usec: 500,
                nr_throttled: 3,
                throttled_usec: 250,
                memory_current: 2 << 20,
            },
        );
        let opts = CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec![],
        };
        let diff = compare(&snap_a, &snap_b, &opts);
        assert_eq!(diff.cgroup_stats_a["/app"].cpu_usage_usec, 100);
        assert_eq!(diff.cgroup_stats_b["/app"].cpu_usage_usec, 500);
    }

    #[test]
    fn categorical_row_labels_same_or_differs() {
        let mut ta = make_thread("app", "w1");
        ta.policy = "SCHED_OTHER".into();
        let mut tb = make_thread("app", "w1");
        tb.policy = "SCHED_FIFO".into();
        let diff = compare(
            &snap_with(vec![ta]),
            &snap_with(vec![tb]),
            &CompareOptions::default(),
        );
        let policy_row = diff
            .rows
            .iter()
            .find(|r| r.metric_name == "policy")
            .expect("policy row");
        assert!(policy_row.delta.is_none());
        match (&policy_row.baseline, &policy_row.candidate) {
            (
                Aggregated::Mode { value: a, .. },
                Aggregated::Mode { value: b, .. },
            ) => {
                assert_eq!(a, "SCHED_OTHER");
                assert_eq!(b, "SCHED_FIFO");
            }
            _ => panic!("expected two Mode aggregates"),
        }
    }

    #[test]
    fn delta_pct_absent_when_baseline_zero() {
        // Baseline=0, candidate=100 → numeric delta is 100 but
        // percent is undefined (division by zero). The row must
        // still appear (the absolute-delta inflation in sort_key
        // keeps it visible).
        let mut ta = make_thread("app", "w1");
        ta.run_time_ns = 0;
        let mut tb = make_thread("app", "w1");
        tb.run_time_ns = 100;
        let diff = compare(
            &snap_with(vec![ta]),
            &snap_with(vec![tb]),
            &CompareOptions::default(),
        );
        let row = diff
            .rows
            .iter()
            .find(|r| r.metric_name == "run_time_ns")
            .expect("row");
        assert_eq!(row.delta, Some(100.0));
        assert!(row.delta_pct.is_none());
    }

    // -- Additional coverage per team-lead directive --

    /// Two empty snapshots (no threads, no cgroup enrichment)
    /// produce an empty diff with zero rows and zero unmatched
    /// groups. Gate against a silent panic or spurious
    /// "only in baseline" entries driven by inserting keys into
    /// the group map from empty inputs.
    #[test]
    fn empty_snapshots_produce_empty_diff() {
        let diff = compare(
            &snap_with(vec![]),
            &snap_with(vec![]),
            &CompareOptions::default(),
        );
        assert!(diff.rows.is_empty());
        assert!(diff.only_baseline.is_empty());
        assert!(diff.only_candidate.is_empty());
    }

    /// Baseline empty, candidate populated: every candidate
    /// group surfaces as `only_candidate`; `rows` stays empty
    /// because there is no matched group to produce a delta.
    #[test]
    fn baseline_empty_surfaces_only_candidate_groups() {
        let t = make_thread("new_proc", "t1");
        let diff = compare(
            &snap_with(vec![]),
            &snap_with(vec![t]),
            &CompareOptions::default(),
        );
        assert!(diff.rows.is_empty());
        assert!(diff.only_baseline.is_empty());
        assert_eq!(diff.only_candidate, vec!["new_proc".to_string()]);
    }

    /// Identical snapshots produce rows whose delta is
    /// uniformly zero (for every numeric rule) and whose
    /// delta_pct is zero (for every non-zero baseline) —
    /// categorical rows still get the "same" treatment via
    /// `Aggregated::Mode` equality. Pin a representative
    /// subset: every delta field in `rows` must be `Some(0.0)`
    /// or `None` (the `None` branch belongs only to categorical
    /// / all-zero-baseline cases).
    #[test]
    fn identical_snapshots_produce_zero_deltas() {
        let mut t = make_thread("app", "w1");
        t.run_time_ns = 1_000;
        t.voluntary_csw = 50;
        let snap = snap_with(vec![t]);
        let diff = compare(&snap, &snap, &CompareOptions::default());
        for row in &diff.rows {
            match row.delta {
                Some(d) => assert_eq!(d, 0.0, "metric {} had nonzero delta", row.metric_name),
                None => {
                    // Only policy (Mode) should surface here for
                    // a populated thread.
                    assert_eq!(row.metric_name, "policy");
                }
            }
        }
    }

    /// Single-thread group: registry emits exactly one row per
    /// registered metric. Defends against a future "skip if
    /// only one thread" short-circuit sneaking into
    /// `aggregate`.
    #[test]
    fn single_thread_group_yields_one_row_per_metric() {
        let a = make_thread("solo", "t");
        let mut b = make_thread("solo", "t");
        b.run_time_ns = 1;
        let diff = compare(
            &snap_with(vec![a]),
            &snap_with(vec![b]),
            &CompareOptions::default(),
        );
        let solo_rows: Vec<&DiffRow> = diff
            .rows
            .iter()
            .filter(|r| r.group_key == "solo")
            .collect();
        assert_eq!(solo_rows.len(), HOST_STATE_METRICS.len());
    }

    /// All-zero cumulative counters on both sides still produce
    /// a row for each Sum metric (delta=0, delta_pct=None
    /// because baseline=0). Gate against a "skip zero" filter
    /// hiding newly-introduced metrics that the workload never
    /// exercises.
    #[test]
    fn all_zero_metrics_emit_zero_delta_rows() {
        let a = make_thread("quiet", "t");
        let b = make_thread("quiet", "t");
        let diff = compare(
            &snap_with(vec![a]),
            &snap_with(vec![b]),
            &CompareOptions::default(),
        );
        let run_time = diff
            .rows
            .iter()
            .find(|r| r.metric_name == "run_time_ns")
            .expect("row");
        assert_eq!(run_time.delta, Some(0.0));
        assert!(run_time.delta_pct.is_none());
    }

    /// `GroupBy::Comm` lumps threads with the same thread name
    /// across processes.
    #[test]
    fn group_by_comm_aggregates_across_processes() {
        let mut ta = make_thread("procA", "worker");
        ta.run_time_ns = 100;
        let mut tb = make_thread("procB", "worker");
        tb.run_time_ns = 200;
        let mut candidate = make_thread("procA", "worker");
        candidate.run_time_ns = 500;
        let mut candidate2 = make_thread("procB", "worker");
        candidate2.run_time_ns = 500;
        let diff = compare(
            &snap_with(vec![ta, tb]),
            &snap_with(vec![candidate, candidate2]),
            &CompareOptions {
                group_by: GroupBy::Comm.into(),
                cgroup_flatten: vec![],
            },
        );
        let row = diff
            .rows
            .iter()
            .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker")
            .expect("worker row");
        // Summed across both processes: baseline=300, candidate=1000, delta=700.
        assert_eq!(row.thread_count_a, 2);
        assert_eq!(row.thread_count_b, 2);
        assert_eq!(row.delta, Some(700.0));
    }

    /// Thread-count change between baseline and candidate
    /// renders "a\u{2192}b" in the row. Gate against silent
    /// collapse to a single value when the group grows or
    /// shrinks.
    #[test]
    fn thread_count_diff_surfaces_when_group_grows() {
        let ta = make_thread("pool", "t");
        let tb1 = make_thread("pool", "t");
        let tb2 = make_thread("pool", "t");
        let diff = compare(
            &snap_with(vec![ta]),
            &snap_with(vec![tb1, tb2]),
            &CompareOptions::default(),
        );
        let row = diff
            .rows
            .iter()
            .find(|r| r.metric_name == "run_time_ns")
            .expect("row");
        assert_eq!(row.thread_count_a, 1);
        assert_eq!(row.thread_count_b, 2);
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Pcomm,
        )
        .unwrap();
        assert!(
            out.contains("1\u{2192}2"),
            "expected thread-count diff rendering, got:\n{out}",
        );
    }

    /// Earlier flatten pattern wins when multiple patterns
    /// match the same path. Gate against a later pattern
    /// silently stealing the collapse when an operator layers
    /// broad and narrow patterns.
    #[test]
    fn flatten_first_match_wins_over_later_pattern() {
        let pats = compile_flatten_patterns(&[
            "/kubepods/*/workload".into(),
            "/kubepods/**".into(),
        ]);
        assert_eq!(
            flatten_cgroup_path("/kubepods/pod-abc/workload", &pats),
            "/kubepods/*/workload",
        );
    }

    /// Multi-pattern collapse: several distinct cgroup paths
    /// flatten to the same key → their enrichment counters
    /// aggregate (sum for counters, max for memory.current).
    #[test]
    fn flatten_cgroup_stats_collapses_overlapping_paths() {
        let mut stats = BTreeMap::new();
        stats.insert(
            "/kubepods/pod-a/workload".into(),
            CgroupStats {
                cpu_usage_usec: 100,
                nr_throttled: 1,
                throttled_usec: 10,
                memory_current: 500,
            },
        );
        stats.insert(
            "/kubepods/pod-b/workload".into(),
            CgroupStats {
                cpu_usage_usec: 200,
                nr_throttled: 2,
                throttled_usec: 20,
                memory_current: 800,
            },
        );
        let pats = compile_flatten_patterns(&["/kubepods/*/workload".into()]);
        let out = flatten_cgroup_stats(&stats, &pats);
        let agg = &out["/kubepods/*/workload"];
        assert_eq!(agg.cpu_usage_usec, 300);
        assert_eq!(agg.nr_throttled, 3);
        assert_eq!(agg.throttled_usec, 30);
        // Instantaneous value: max, not sum.
        assert_eq!(agg.memory_current, 800);
    }

    /// Malformed glob patterns are silently dropped by the
    /// compiler (they never match so they never collapse
    /// anything). Gate against a future change that accidentally
    /// starts rejecting valid-looking patterns.
    #[test]
    fn compile_flatten_patterns_skips_malformed() {
        let pats = compile_flatten_patterns(&["[invalid".into(), "/ok/*".into()]);
        assert_eq!(pats.len(), 1);
        assert_eq!(pats[0].as_str(), "/ok/*");
    }

    /// Every `ThreadState` field that names a registered metric
    /// in the registry has a reachable accessor: sum one unit of
    /// that field through a single-thread aggregate and confirm
    /// the Sum result is 1. Defends against a typo in a new
    /// `AggRule::Sum` accessor pointing at the wrong field.
    ///
    /// The test is metric-registry-driven rather than field-
    /// driven because new metrics have to land through the
    /// registry; a drift between the test and the registry
    /// would catch itself.
    #[test]
    fn sum_metric_accessors_read_expected_field() {
        let cases: &[(&str, fn(&mut ThreadState))] = &[
            ("run_time_ns", |t| t.run_time_ns = 1),
            ("wait_time_ns", |t| t.wait_time_ns = 1),
            ("timeslices", |t| t.timeslices = 1),
            ("voluntary_csw", |t| t.voluntary_csw = 1),
            ("nonvoluntary_csw", |t| t.nonvoluntary_csw = 1),
            ("nr_wakeups", |t| t.nr_wakeups = 1),
            ("nr_wakeups_local", |t| t.nr_wakeups_local = 1),
            ("nr_wakeups_remote", |t| t.nr_wakeups_remote = 1),
            ("nr_wakeups_sync", |t| t.nr_wakeups_sync = 1),
            ("nr_wakeups_migrate", |t| t.nr_wakeups_migrate = 1),
            ("nr_wakeups_idle", |t| t.nr_wakeups_idle = 1),
            ("nr_migrations", |t| t.nr_migrations = 1),
            ("wait_sum", |t| t.wait_sum = 1),
            ("wait_count", |t| t.wait_count = 1),
            ("sleep_sum", |t| t.sleep_sum = 1),
            ("allocated_bytes", |t| t.allocated_bytes = 1),
            ("deallocated_bytes", |t| t.deallocated_bytes = 1),
            ("minflt", |t| t.minflt = 1),
            ("majflt", |t| t.majflt = 1),
            ("rchar", |t| t.rchar = 1),
            ("wchar", |t| t.wchar = 1),
            ("syscr", |t| t.syscr = 1),
            ("syscw", |t| t.syscw = 1),
            ("read_bytes", |t| t.read_bytes = 1),
            ("write_bytes", |t| t.write_bytes = 1),
        ];
        for (name, set) in cases {
            let mut t = make_thread("p", "w");
            set(&mut t);
            let def = HOST_STATE_METRICS
                .iter()
                .find(|m| m.name == *name)
                .unwrap_or_else(|| panic!("metric {name} not in registry"));
            let agg = aggregate(def.rule, &[&t]);
            match agg {
                Aggregated::Sum(v) => assert_eq!(
                    v, 1,
                    "accessor for {name} did not read the {name} field",
                ),
                other => panic!("expected Sum for {name}, got {other:?}"),
            }
        }
    }

    /// Every registered metric name must be unique. A
    /// collision would silently shadow the earlier entry in
    /// lookups and still "work" for fields that happen to
    /// match — a slow-burn correctness bug.
    #[test]
    fn host_state_metric_names_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for m in HOST_STATE_METRICS {
            assert!(
                seen.insert(m.name),
                "duplicate metric name in registry: {}",
                m.name,
            );
        }
    }

    /// Mode rule with a deterministic tie-break: when two
    /// values share the top count, the lexicographically
    /// smaller one wins. Pin the rule so the rendered output
    /// is reproducible across runs.
    #[test]
    fn mode_rule_tie_break_is_lexicographic() {
        let mut a = make_thread("app", "w1");
        a.policy = "SCHED_FIFO".into();
        let mut b = make_thread("app", "w2");
        b.policy = "SCHED_OTHER".into();
        let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &[&a, &b]);
        match v {
            Aggregated::Mode { value, count, .. } => {
                assert_eq!(value, "SCHED_FIFO");
                assert_eq!(count, 1);
            }
            other => panic!("expected Mode, got {other:?}"),
        }
    }

    /// Affinity aggregate on an empty thread slice returns
    /// `min_cpus == max_cpus == 0` and no uniform cpuset — the
    /// compare engine cannot produce an empty group today, but
    /// this defends against an upstream refactor that permits
    /// one.
    #[test]
    fn affinity_aggregate_on_empty_threads_is_zero() {
        let empty: Vec<&ThreadState> = vec![];
        let v = aggregate(AggRule::Affinity(|t| t.cpu_affinity.clone()), &empty);
        match v {
            Aggregated::Affinity(s) => {
                assert_eq!(s.min_cpus, 0);
                assert_eq!(s.max_cpus, 0);
                assert!(s.uniform.is_none());
            }
            other => panic!("expected Affinity, got {other:?}"),
        }
    }

    /// Ordinal range collapses `min == max` to a single number
    /// in display. Defends against `nice=0` single-thread
    /// groups rendering as `0..0`.
    #[test]
    fn ordinal_display_collapses_degenerate_range() {
        let r = Aggregated::OrdinalRange { min: 0, max: 0 };
        assert_eq!(r.to_string(), "0");
        let r = Aggregated::OrdinalRange { min: -5, max: 10 };
        assert_eq!(r.to_string(), "-5..10");
    }

    /// Mode display omits the minority ratio when the mode is
    /// unanimous (count == total). Keeps the table compact for
    /// homogeneous groups.
    #[test]
    fn mode_display_hides_ratio_when_unanimous() {
        let m = Aggregated::Mode {
            value: "SCHED_OTHER".into(),
            count: 4,
            total: 4,
        };
        assert_eq!(m.to_string(), "SCHED_OTHER");
        let m = Aggregated::Mode {
            value: "SCHED_OTHER".into(),
            count: 3,
            total: 5,
        };
        assert_eq!(m.to_string(), "SCHED_OTHER (3/5)");
    }

    // -- write_diff: output rendering --

    #[test]
    fn write_diff_emits_expected_column_headers() {
        let diff = compare(
            &snap_with(vec![make_thread("p", "w")]),
            &snap_with(vec![make_thread("p", "w")]),
            &CompareOptions::default(),
        );
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Pcomm,
        )
        .unwrap();
        for h in ["pcomm", "threads", "metric", "baseline", "candidate", "delta", "%"] {
            assert!(out.contains(h), "missing header {h}:\n{out}");
        }
    }

    #[test]
    fn write_diff_header_switches_on_group_by() {
        let empty = HostStateDiff::default();
        let mut out = String::new();
        write_diff(
            &mut out,
            &empty,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Cgroup,
        )
        .unwrap();
        assert!(out.contains("cgroup"));
        let mut out = String::new();
        write_diff(
            &mut out,
            &empty,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Comm,
        )
        .unwrap();
        assert!(out.contains("comm"));
        // "comm" must render as the column header, not as a
        // substring of "pcomm" left over from the Pcomm variant.
        assert!(!out.contains("pcomm"));
    }

    #[test]
    fn write_diff_prints_only_baseline_section() {
        let diff = HostStateDiff {
            only_baseline: vec!["missing_proc".into()],
            ..HostStateDiff::default()
        };
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("/tmp/a.hst.zst"),
            Path::new("/tmp/b.hst.zst"),
            GroupBy::Pcomm,
        )
        .unwrap();
        assert!(out.contains("only in baseline"));
        assert!(out.contains("missing_proc"));
        assert!(out.contains("/tmp/a.hst.zst"));
    }

    #[test]
    fn write_diff_prints_only_candidate_section() {
        let diff = HostStateDiff {
            only_candidate: vec!["new_proc".into()],
            ..HostStateDiff::default()
        };
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("/tmp/a.hst.zst"),
            Path::new("/tmp/b.hst.zst"),
            GroupBy::Pcomm,
        )
        .unwrap();
        assert!(out.contains("only in candidate"));
        assert!(out.contains("new_proc"));
        assert!(out.contains("/tmp/b.hst.zst"));
    }

    #[test]
    fn write_diff_cgroup_enrichment_section_for_cgroup_mode() {
        let mut diff = HostStateDiff::default();
        diff.cgroup_stats_a.insert(
            "/app".into(),
            CgroupStats {
                cpu_usage_usec: 10,
                nr_throttled: 0,
                throttled_usec: 0,
                memory_current: 100,
            },
        );
        diff.cgroup_stats_b.insert(
            "/app".into(),
            CgroupStats {
                cpu_usage_usec: 50,
                nr_throttled: 0,
                throttled_usec: 0,
                memory_current: 200,
            },
        );
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Cgroup,
        )
        .unwrap();
        assert!(out.contains("cpu_usage_usec"), "missing enrichment header:\n{out}");
        assert!(out.contains("10"), "missing baseline usec:\n{out}");
        assert!(out.contains("50"), "missing candidate usec:\n{out}");
        assert!(out.contains("+40"), "missing delta:\n{out}");
    }

    #[test]
    fn write_diff_enrichment_section_absent_when_group_by_pcomm() {
        let mut diff = HostStateDiff::default();
        // Populate enrichment; renderer must ignore it under
        // GroupBy::Pcomm.
        diff.cgroup_stats_a.insert(
            "/app".into(),
            CgroupStats {
                cpu_usage_usec: 10,
                ..CgroupStats::default()
            },
        );
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Pcomm,
        )
        .unwrap();
        assert!(!out.contains("cpu_usage_usec"), "enrichment leaked:\n{out}");
    }

    #[test]
    fn write_diff_delta_cell_has_plus_minus_sign() {
        let mut ta = make_thread("app", "w");
        ta.run_time_ns = 100;
        let mut tb = make_thread("app", "w");
        tb.run_time_ns = 50;
        let diff = compare(
            &snap_with(vec![ta]),
            &snap_with(vec![tb]),
            &CompareOptions::default(),
        );
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Pcomm,
        )
        .unwrap();
        // 50 - 100 = -50ns → renders with minus sign + unit.
        assert!(
            out.contains("-50.000ns"),
            "missing signed delta with unit:\n{out}",
        );
        assert!(out.contains("-50.0%"), "missing signed pct:\n{out}");
    }

    #[test]
    fn write_diff_categorical_delta_labels_same_or_differs() {
        let mut ta = make_thread("app", "w");
        ta.policy = "SCHED_OTHER".into();
        let mut tb = make_thread("app", "w");
        tb.policy = "SCHED_FIFO".into();
        let diff = compare(
            &snap_with(vec![ta]),
            &snap_with(vec![tb]),
            &CompareOptions::default(),
        );
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Pcomm,
        )
        .unwrap();
        assert!(out.contains("differs"), "missing 'differs' label:\n{out}");
    }

    /// Full round-trip via the public loader: two snapshots
    /// written to disk via `HostStateSnapshot::write`, loaded
    /// via `HostStateSnapshot::load`, compared, and the
    /// rendered output inspected. This stitches together the
    /// serialization layer, the comparison engine, and the
    /// formatter — the components `run_compare` composes in
    /// production.
    #[test]
    fn load_compare_render_pipeline_end_to_end() {
        let mut a = make_thread("e2e_proc", "thread_a");
        a.run_time_ns = 1_000_000;
        a.voluntary_csw = 10;
        a.policy = "SCHED_OTHER".into();
        let snap_a = snap_with(vec![a]);
        let mut b = make_thread("e2e_proc", "thread_a");
        b.run_time_ns = 3_000_000;
        b.voluntary_csw = 30;
        b.policy = "SCHED_FIFO".into();
        let snap_b = snap_with(vec![b]);

        let tmp_a = tempfile::NamedTempFile::new().unwrap();
        let tmp_b = tempfile::NamedTempFile::new().unwrap();
        snap_a.write(tmp_a.path()).unwrap();
        snap_b.write(tmp_b.path()).unwrap();
        let loaded_a = HostStateSnapshot::load(tmp_a.path()).unwrap();
        let loaded_b = HostStateSnapshot::load(tmp_b.path()).unwrap();

        let diff = compare(&loaded_a, &loaded_b, &CompareOptions::default());
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            tmp_a.path(),
            tmp_b.path(),
            GroupBy::Pcomm,
        )
        .unwrap();

        // Column headers present.
        assert!(out.contains("pcomm"));
        assert!(out.contains("metric"));
        // Group key made it through.
        assert!(out.contains("e2e_proc"));
        // run_time_ns delta: +2_000_000 → rendered with plus sign
        // and the `ns` unit.
        assert!(
            out.contains("+2000000.000ns"),
            "run_time delta missing in:\n{out}",
        );
        // Policy row renders "differs" because SCHED_FIFO vs
        // SCHED_OTHER — non-numeric delta path exercised.
        assert!(out.contains("differs"));
    }

    // -- comparison coverage expansion --

    /// Pin all four branches of `cgroup_cell` directly. Existing
    /// tests only exercise the (Some, Some) path transitively via
    /// `write_diff_cgroup_enrichment_section_for_cgroup_mode`; the
    /// other three branches (baseline-only, candidate-only,
    /// both-missing) are rendering-critical for the one-sided
    /// enrichment row path (`all_keys` union at the enrichment
    /// table site) and have no current pin.
    #[test]
    fn cgroup_cell_renders_all_four_branches() {
        // (Some, Some) → "a → b (+d)" where d = b - a (signed).
        assert_eq!(cgroup_cell(Some(10), Some(42)), "10 → 42 (+32)");
        // Negative delta uses the signed formatter to keep the
        // sign explicit.
        assert_eq!(cgroup_cell(Some(50), Some(5)), "50 → 5 (-45)");
        // (Some, None) → baseline value then en-dash placeholder.
        assert_eq!(cgroup_cell(Some(7), None), "7 → -");
        // (None, Some) → leading en-dash placeholder.
        assert_eq!(cgroup_cell(None, Some(99)), "- → 99");
        // (None, None) → single en-dash (both sides absent).
        assert_eq!(cgroup_cell(None, None), "-");
    }

    /// Enrichment renderer must union `cgroup_stats_a` and
    /// `cgroup_stats_b` keys so a cgroup that appeared in only one
    /// run still surfaces a row. Drives the one-sided paths of
    /// `cgroup_cell` through `write_diff` so the rendered output
    /// carries the `"X → -"` / `"- → Y"` strings.
    #[test]
    fn write_diff_enrichment_handles_one_sided_cgroup_keys() {
        let mut diff = HostStateDiff::default();
        diff.cgroup_stats_a.insert(
            "/only-baseline".into(),
            CgroupStats {
                cpu_usage_usec: 111,
                ..CgroupStats::default()
            },
        );
        diff.cgroup_stats_b.insert(
            "/only-candidate".into(),
            CgroupStats {
                cpu_usage_usec: 222,
                ..CgroupStats::default()
            },
        );
        let mut out = String::new();
        write_diff(
            &mut out,
            &diff,
            Path::new("a"),
            Path::new("b"),
            GroupBy::Cgroup,
        )
        .unwrap();
        // Both keys present.
        assert!(
            out.contains("/only-baseline"),
            "baseline-only key missing:\n{out}",
        );
        assert!(
            out.contains("/only-candidate"),
            "candidate-only key missing:\n{out}",
        );
        // Each one-sided row emits the en-dash placeholder for
        // the absent side (per `cgroup_cell`'s Some/None branch).
        assert!(
            out.contains("111 → -"),
            "baseline-only row missing '111 → -' cell:\n{out}",
        );
        assert!(
            out.contains("- → 222"),
            "candidate-only row missing '- → 222' cell:\n{out}",
        );
    }

    /// Rows with equal `sort_key()` break ties by ascending
    /// `group_key`. Build two groups that move the same metric by
    /// the same percentage (so their sort keys are identical) and
    /// verify the output order is alphabetical.
    #[test]
    fn write_diff_stable_sort_tie_breaks_by_group_key_ascending() {
        // Same percentage swing, distinct group keys "alpha" and
        // "bravo". Both rise 1_000 → 2_000 (+100%).
        let mut a1 = make_thread("alpha", "w");
        a1.run_time_ns = 1_000;
        let mut a2 = make_thread("bravo", "w");
        a2.run_time_ns = 1_000;
        let mut b1 = make_thread("alpha", "w");
        b1.run_time_ns = 2_000;
        let mut b2 = make_thread("bravo", "w");
        b2.run_time_ns = 2_000;
        let diff = compare(
            &snap_with(vec![a1, a2]),
            &snap_with(vec![b1, b2]),
            &CompareOptions::default(),
        );
        // Filter to run_time_ns rows across the two groups; the
        // tie-break must put "alpha" before "bravo".
        let run_rows: Vec<&DiffRow> = diff
            .rows
            .iter()
            .filter(|r| r.metric_name == "run_time_ns")
            .collect();
        assert_eq!(run_rows.len(), 2);
        assert!(
            (run_rows[0].delta_pct.unwrap() - 1.0).abs() < 1e-9
                && (run_rows[1].delta_pct.unwrap() - 1.0).abs() < 1e-9,
            "test fixture must produce identical delta_pct for both groups",
        );
        assert_eq!(
            run_rows[0].group_key, "alpha",
            "ascending group_key tie-break expected alpha first",
        );
        assert_eq!(run_rows[1].group_key, "bravo");
    }

    /// `sort_key` inflates the zero-baseline-nonzero-candidate
    /// branch (delta=Some, delta_pct=None) by 1e9 so it sorts
    /// above pure zero-delta rows but still below any nonzero
    /// percentage row. Two rows: one zero-delta (delta_pct=0.0),
    /// one zero-baseline (delta=100, delta_pct=None) — the zero-
    /// baseline row must sort FIRST.
    #[test]
    fn sort_key_zero_delta_rows_sink_below_nonzero() {
        // Group "calm": identical values → delta 0, pct 0.0.
        let mut a1 = make_thread("calm", "w");
        a1.run_time_ns = 500;
        let mut b1 = make_thread("calm", "w");
        b1.run_time_ns = 500;
        // Group "birth": baseline 0 → candidate 100 → delta 100,
        // pct undefined (None). sort_key inflates to 100 * 1e9.
        let a2 = make_thread("birth", "w");
        let mut b2 = make_thread("birth", "w");
        b2.run_time_ns = 100;
        let diff = compare(
            &snap_with(vec![a1, a2]),
            &snap_with(vec![b1, b2]),
            &CompareOptions::default(),
        );
        let run_rows: Vec<&DiffRow> = diff
            .rows
            .iter()
            .filter(|r| r.metric_name == "run_time_ns")
            .collect();
        // "birth" row (zero-baseline branch of sort_key) sorts
        // ahead of "calm" (zero-delta branch).
        assert_eq!(run_rows[0].group_key, "birth");
        assert_eq!(run_rows[1].group_key, "calm");
        // Pin the exact shape each branch is meant to carry, so a
        // regression that swapped the inflation with the zero
        // arm surfaces here with a precise diagnostic rather than
        // just "wrong order".
        assert_eq!(run_rows[0].delta, Some(100.0));
        assert!(run_rows[0].delta_pct.is_none());
        assert_eq!(run_rows[1].delta, Some(0.0));
        assert_eq!(run_rows[1].delta_pct, Some(0.0));
    }

    /// Rows with no numeric delta (categorical Mode) sort to the
    /// bottom via `sort_key`'s `f64::NEG_INFINITY` arm. Pin that a
    /// nonzero numeric row sorts ahead of a Mode row whose inputs
    /// differ, and that the Mode row still appears (sinks, not
    /// dropped).
    #[test]
    fn sort_key_none_delta_rows_sink_to_bottom() {
        let mut a = make_thread("app", "w");
        a.run_time_ns = 100;
        a.policy = "SCHED_OTHER".into();
        let mut b = make_thread("app", "w");
        b.run_time_ns = 200;
        b.policy = "SCHED_FIFO".into();
        let diff = compare(
            &snap_with(vec![a]),
            &snap_with(vec![b]),
            &CompareOptions::default(),
        );
        // Locate the positions of run_time_ns (numeric) and
        // policy (Mode, delta=None) in the sorted rows.
        let run_idx = diff
            .rows
            .iter()
            .position(|r| r.metric_name == "run_time_ns")
            .expect("run_time_ns row");
        let policy_idx = diff
            .rows
            .iter()
            .position(|r| r.metric_name == "policy")
            .expect("policy row");
        assert!(
            run_idx < policy_idx,
            "numeric row at {run_idx} must sort above Mode row at {policy_idx}",
        );
        // Mode row really is None-delta — otherwise the ordering
        // wouldn't prove the NEG_INFINITY branch.
        assert!(diff.rows[policy_idx].delta.is_none());
    }

    /// `aggregate(OrdinalRange, &[])` returns `OrdinalRange {
    /// min: 0, max: 0 }` via the `unwrap_or(0)` in the first-value
    /// init. Sibling to the empty-affinity test.
    #[test]
    fn aggregate_ordinal_range_on_empty_threads_is_zero() {
        let empty: Vec<&ThreadState> = vec![];
        let v = aggregate(AggRule::OrdinalRange(|t| t.nice as i64), &empty);
        match v {
            Aggregated::OrdinalRange { min, max } => {
                assert_eq!(min, 0);
                assert_eq!(max, 0);
            }
            other => panic!("expected OrdinalRange, got {other:?}"),
        }
    }

    /// `aggregate(Mode, &[])` returns `Mode { value: "", count:
    /// 0, total: 0 }` via the `unwrap_or_else((String::new(), 0))`
    /// tail.
    #[test]
    fn aggregate_mode_on_empty_threads_is_empty() {
        let empty: Vec<&ThreadState> = vec![];
        let v = aggregate(AggRule::Mode(|t| t.policy.clone()), &empty);
        match v {
            Aggregated::Mode {
                value,
                count,
                total,
            } => {
                assert!(value.is_empty());
                assert_eq!(count, 0);
                assert_eq!(total, 0);
            }
            other => panic!("expected Mode, got {other:?}"),
        }
    }

    /// `aggregate(Sum, &[])` returns `Sum(0)` via the `fold(0u64,
    /// ...)` accumulator. Completes empty-slice coverage across
    /// all four AggRules.
    #[test]
    fn aggregate_sum_on_empty_threads_is_zero() {
        let empty: Vec<&ThreadState> = vec![];
        let v = aggregate(AggRule::Sum(|t| t.run_time_ns), &empty);
        match v {
            Aggregated::Sum(s) => assert_eq!(s, 0),
            other => panic!("expected Sum, got {other:?}"),
        }
    }

    /// `Aggregated::numeric` returns `None` for `Mode` — a
    /// policy name has no scalar projection. Pin the contract
    /// directly rather than via the diff pipeline because the
    /// pipeline only reads numeric through `build_row`'s `(a.numeric(),
    /// b.numeric())` pair and a regression could silently flip the
    /// return to `Some(0.0)` without any currently-visible symptom.
    #[test]
    fn numeric_returns_none_for_mode() {
        let m = Aggregated::Mode {
            value: "SCHED_OTHER".into(),
            count: 4,
            total: 4,
        };
        assert!(m.numeric().is_none());
    }

    /// `Aggregated::numeric` for a heterogeneous `Affinity`
    /// returns `(min_cpus + max_cpus) / 2.0` — the midpoint
    /// projection. Existing affinity tests only exercise uniform
    /// cpusets where `min == max`, so the arithmetic path is
    /// unpinned.
    #[test]
    fn numeric_returns_midpoint_for_affinity_heterogeneous() {
        let a = Aggregated::Affinity(AffinitySummary {
            min_cpus: 2,
            max_cpus: 8,
            uniform: None,
        });
        assert_eq!(a.numeric(), Some(5.0));
        // Single-element (uniform) heterogeneous check is the
        // degenerate case where the midpoint equals either bound.
        let b = Aggregated::Affinity(AffinitySummary {
            min_cpus: 4,
            max_cpus: 4,
            uniform: None,
        });
        assert_eq!(b.numeric(), Some(4.0));
    }

    /// Uniform non-contiguous cpuset `[0, 2]` renders as
    /// `"2 cpus (0,2)"` — exercises the comma-separated branch of
    /// `format_cpu_range` from the Affinity display impl. Existing
    /// uniform test uses `[0,1,2,3]` which collapses to a single
    /// range token.
    #[test]
    fn affinity_display_uniform_noncontiguous_renders_comma_separated() {
        let a = Aggregated::Affinity(AffinitySummary {
            min_cpus: 2,
            max_cpus: 2,
            uniform: Some(vec![0, 2]),
        });
        assert_eq!(a.to_string(), "2 cpus (0,2)");
    }

    /// Heterogeneous affinity where `min_cpus == max_cpus` (every
    /// thread has the same cpuset SIZE but different SETS) renders
    /// as `"N cpus (mixed)"` — pins the specific branch in the
    /// display impl. Current heterogeneous test has min != max so
    /// this branch was unpinned.
    #[test]
    fn affinity_display_heterogeneous_same_count_renders_mixed() {
        let a = Aggregated::Affinity(AffinitySummary {
            min_cpus: 3,
            max_cpus: 3,
            uniform: None,
        });
        assert_eq!(a.to_string(), "3 cpus (mixed)");
    }

    /// `flatten_cgroup_stats` with zero patterns preserves the
    /// input map verbatim — no entry merges, no key rewrites. A
    /// regression that accidentally ran the aggregation step on
    /// the empty-pattern path would collapse distinct cgroup paths
    /// together.
    #[test]
    fn flatten_cgroup_stats_with_no_patterns_preserves_keys() {
        let mut stats = BTreeMap::new();
        stats.insert(
            "/alpha".into(),
            CgroupStats {
                cpu_usage_usec: 10,
                nr_throttled: 1,
                throttled_usec: 5,
                memory_current: 100,
            },
        );
        stats.insert(
            "/beta".into(),
            CgroupStats {
                cpu_usage_usec: 20,
                nr_throttled: 2,
                throttled_usec: 15,
                memory_current: 200,
            },
        );
        let out = flatten_cgroup_stats(&stats, &[]);
        assert_eq!(out.len(), 2);
        assert_eq!(out["/alpha"].cpu_usage_usec, 10);
        assert_eq!(out["/alpha"].memory_current, 100);
        assert_eq!(out["/beta"].cpu_usage_usec, 20);
        assert_eq!(out["/beta"].memory_current, 200);
    }
}
