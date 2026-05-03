//! Tests for `super::metrics` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::*;
use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters,
    merge_max_option, merge_memory_stat, merge_min_option, merge_psi,
};
use super::columns::{compare_columns_for, format_cgroup_only_section_warning};
use super::compare::sort_diff_rows_by_keys;
use super::groups::build_row;
use super::pattern::{
    Segment, apply_systemd_template, cgroup_normalize_skeleton, cgroup_skeleton_tokens,
    classify_token, is_token_separator, pattern_counts_union, pattern_key, split_into_segments,
    tighten_group,
};
use super::render::psi_pair_has_data;
use super::scale::{auto_scale, format_delta_cell};
use super::tests_fixtures::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// Every `ThreadState` field that names a registered metric
/// in the registry has a reachable accessor: sum one unit of
/// that field through a single-thread aggregate and confirm
/// the Sum result is 1. Defends against a typo in any
/// `AggRule::Sum*` variant
/// ([`AggRule::SumCount`] / [`AggRule::SumNs`] /
/// [`AggRule::SumTicks`] / [`AggRule::SumBytes`]) accessor
/// pointing at the wrong field.
///
/// The test is metric-registry-driven rather than field-
/// driven because new metrics have to land through the
/// registry; a drift between the test and the registry
/// would catch itself.
#[test]
fn sum_metric_accessors_read_expected_field() {
    use crate::metric_types::{Bytes, ClockTicks, MonotonicCount, MonotonicNs};
    type MetricSetter = fn(&mut ThreadState);
    let cases: &[(&str, MetricSetter)] = &[
        ("run_time_ns", |t| t.run_time_ns = MonotonicNs(1)),
        ("wait_time_ns", |t| t.wait_time_ns = MonotonicNs(1)),
        ("timeslices", |t| t.timeslices = MonotonicCount(1)),
        ("voluntary_csw", |t| t.voluntary_csw = MonotonicCount(1)),
        ("nonvoluntary_csw", |t| {
            t.nonvoluntary_csw = MonotonicCount(1)
        }),
        ("nr_wakeups", |t| t.nr_wakeups = MonotonicCount(1)),
        ("nr_wakeups_local", |t| {
            t.nr_wakeups_local = MonotonicCount(1)
        }),
        ("nr_wakeups_remote", |t| {
            t.nr_wakeups_remote = MonotonicCount(1)
        }),
        ("nr_wakeups_sync", |t| t.nr_wakeups_sync = MonotonicCount(1)),
        ("nr_wakeups_migrate", |t| {
            t.nr_wakeups_migrate = MonotonicCount(1)
        }),
        ("nr_wakeups_affine", |t| {
            t.nr_wakeups_affine = MonotonicCount(1)
        }),
        ("nr_wakeups_affine_attempts", |t| {
            t.nr_wakeups_affine_attempts = MonotonicCount(1)
        }),
        ("nr_migrations", |t| t.nr_migrations = MonotonicCount(1)),
        ("nr_forced_migrations", |t| {
            t.nr_forced_migrations = MonotonicCount(1)
        }),
        ("nr_failed_migrations_affine", |t| {
            t.nr_failed_migrations_affine = MonotonicCount(1)
        }),
        ("nr_failed_migrations_running", |t| {
            t.nr_failed_migrations_running = MonotonicCount(1)
        }),
        ("nr_failed_migrations_hot", |t| {
            t.nr_failed_migrations_hot = MonotonicCount(1)
        }),
        ("wait_sum", |t| t.wait_sum = MonotonicNs(1)),
        ("wait_count", |t| t.wait_count = MonotonicCount(1)),
        ("voluntary_sleep_ns", |t| {
            t.voluntary_sleep_ns = MonotonicNs(1)
        }),
        ("block_sum", |t| t.block_sum = MonotonicNs(1)),
        ("iowait_sum", |t| t.iowait_sum = MonotonicNs(1)),
        ("iowait_count", |t| t.iowait_count = MonotonicCount(1)),
        ("allocated_bytes", |t| t.allocated_bytes = Bytes(1)),
        ("deallocated_bytes", |t| t.deallocated_bytes = Bytes(1)),
        ("minflt", |t| t.minflt = MonotonicCount(1)),
        ("majflt", |t| t.majflt = MonotonicCount(1)),
        ("utime_clock_ticks", |t| t.utime_clock_ticks = ClockTicks(1)),
        ("stime_clock_ticks", |t| t.stime_clock_ticks = ClockTicks(1)),
        ("rchar", |t| t.rchar = Bytes(1)),
        ("wchar", |t| t.wchar = Bytes(1)),
        ("syscr", |t| t.syscr = MonotonicCount(1)),
        ("syscw", |t| t.syscw = MonotonicCount(1)),
        ("read_bytes", |t| t.read_bytes = Bytes(1)),
        ("write_bytes", |t| t.write_bytes = Bytes(1)),
        ("cancelled_write_bytes", |t| {
            t.cancelled_write_bytes = Bytes(1)
        }),
    ];
    for (name, set) in cases {
        let mut t = make_thread("p", "w");
        set(&mut t);
        let def = CTPROF_METRICS
            .iter()
            .find(|m| m.name == *name)
            .unwrap_or_else(|| panic!("metric {name} not in registry"));
        let agg = aggregate(def.rule, &[&t]);
        match agg {
            Aggregated::Sum(v) => {
                assert_eq!(v, 1, "accessor for {name} did not read the {name} field",)
            }
            other => panic!("expected Sum for {name}, got {other:?}"),
        }
    }
}

/// Every registered metric name must be unique. A
/// collision would silently shadow the earlier entry in
/// lookups and still "work" for fields that happen to
/// match — a slow-burn correctness bug.
#[test]
fn ctprof_metric_names_are_unique() {
    let mut seen = std::collections::BTreeSet::new();
    for m in CTPROF_METRICS {
        assert!(
            seen.insert(m.name),
            "duplicate metric name in registry: {}",
            m.name,
        );
    }
}

/// `metric_display_name` of a fully-ungated metric returns
/// the bare name with no trailing tags. Pins the
/// no-decoration short-circuit for the typical case, and
/// verifies that the borrowed-Cow path is taken (no
/// allocation when nothing decorates).
#[test]
fn metric_display_name_no_gates_returns_bare_name() {
    let policy = lookup_metric("policy");
    assert_eq!(metric_display_name(policy), "policy");
    assert!(metric_tags(policy).is_empty());
    let cpu_aff = lookup_metric("cpu_affinity");
    assert_eq!(metric_display_name(cpu_aff), "cpu_affinity");
    assert!(metric_tags(cpu_aff).is_empty());
}

/// CFS-only + CONFIG_SCHEDSTATS metric renders BOTH tags in
/// stable order: sched_class first, then each config gate.
/// `nr_wakeups_affine` is the load-bearing example here —
/// `kernel/sched/fair.c::wake_affine` is the only call site
/// for the underlying `__schedstat_inc`. The config gate
/// renders compact (`[SCHEDSTATS]` not `[CONFIG_SCHEDSTATS]`)
/// per the strip rule on `metric_display_name`. Pins both
/// decoration paths against drift.
#[test]
fn metric_tags_renders_class_and_config_tags() {
    let m = lookup_metric("nr_wakeups_affine");
    assert_eq!(metric_display_name(m), "nr_wakeups_affine");
    assert_eq!(metric_tags(m), "[cfs-only] [SCHEDSTATS]");
}

/// Multi-gate metric (`core_forceidle_sum` requires both
/// CONFIG_SCHED_CORE and CONFIG_SCHEDSTATS) renders every
/// gate in registry-declared order. Class is `None` here so
/// no class tag emits — only the two config tags. Compact
/// rendering strips the `CONFIG_` prefix from each gate.
#[test]
fn metric_tags_emits_each_config_gate_in_order() {
    let core = lookup_metric("core_forceidle_sum");
    assert_eq!(metric_display_name(core), "core_forceidle_sum");
    assert_eq!(metric_tags(core), "[SCHED_CORE] [SCHEDSTATS]");
}

/// `fair_slice_ns` is fair-policy-only with no config gate.
/// Pins that the class tag emits without any trailing
/// config-gate tag — the for-loop must not produce a
/// trailing `[]` or trailing whitespace when
/// `config_gates` is empty.
#[test]
fn metric_tags_class_only_no_config_gate() {
    let fair = lookup_metric("fair_slice_ns");
    assert_eq!(metric_display_name(fair), "fair_slice_ns");
    assert_eq!(metric_tags(fair), "[fair-policy]");
}

/// Compact rendering: `metric_display_name` strips the
/// `CONFIG_` prefix from each `config_gate` before emission.
/// The data field stays full so an operator can grep their
/// kconfig directly. Pin the rule explicitly so a refactor
/// of `metric_display_name` does not silently regress the
/// strip behavior.
#[test]
fn metric_tags_strips_config_prefix() {
    for m in CTPROF_METRICS {
        for gate in m.config_gates {
            assert!(
                gate.starts_with("CONFIG_"),
                "registry config_gate {gate:?} on metric {} \
                 must spell the literal CONFIG_X kconfig symbol",
                m.name,
            );
            let tags = metric_tags(m);
            let expected_short = gate.strip_prefix("CONFIG_").unwrap();
            assert!(
                tags.contains(&format!("[{expected_short}]")),
                "metric {} tags {tags:?} must contain [{expected_short}]",
                m.name,
            );
            assert!(
                !tags.contains(&format!("[{gate}]")),
                "metric {} tags {tags:?} must not contain full [{gate}]",
                m.name,
            );
        }
    }
}

/// `[dead]` tag rendering remains in the metric-display
/// machinery even though the registry currently has no
/// `is_dead: true` entries (the previously-registered dead
/// counters were dropped). Pin the rendering on a synthetic
/// `CtprofMetricDef` so a regression that drops the
/// `[dead]` clause from `metric_display_name` surfaces here
/// rather than waiting for a future kernel quirk that
/// resurrects the tag.
#[test]
fn metric_tags_marks_synthetic_dead_counter() {
    let m = CtprofMetricDef {
        name: "synthetic_dead",
        rule: AggRule::SumCount(|_| crate::metric_types::MonotonicCount(0)),
        sched_class: None,
        config_gates: &["CONFIG_SCHEDSTATS"],
        is_dead: true,
        description: "synthetic dead-counter test fixture.",
        section: Section::Primary,
    };
    assert_eq!(metric_display_name(&m), "synthetic_dead");
    assert_eq!(metric_tags(&m), "[dead] [SCHEDSTATS]",);
    // Live registry must NOT carry any is_dead: true entries
    // until a kernel resurrects a dead counter or a new
    // always-zero counter is captured. Detects accidental
    // re-introduction.
    for m in CTPROF_METRICS {
        assert!(
            !m.is_dead,
            "{} unexpectedly carries is_dead: true — the \
             registry is currently empty of dead counters; \
             add the entry to the matrix-pin test below if \
             a new dead counter is intentional",
            m.name,
        );
    }
}

/// `non-ext` rendering: the schedstat sleep/wait family is
/// tagged `non-ext` because it accumulates under CFS / RT /
/// DL but not sched_ext. Pin a representative example:
/// `wait_sum [non-ext] [SCHEDSTATS]`. Guards against the
/// matrix regression that previously left these tagged
/// `None`.
#[test]
fn metric_tags_renders_non_ext_class() {
    let m = lookup_metric("wait_sum");
    assert_eq!(metric_display_name(m), "wait_sum");
    assert_eq!(metric_tags(m), "[non-ext] [SCHEDSTATS]",);
}

/// Exhaustive tag pin: every metric in CTPROF_METRICS
/// gets its (sched_class, config_gates, is_dead) triple
/// asserted against the locked matrix. Set-equality on the
/// keys: every registry name must appear in the matrix
/// table, and vice versa. Drift on either side fails the
/// test before reaching the rendered output.
#[test]
fn registry_tag_matrix_is_pinned() {
    // Locked matrix: (name → (sched_class, config_gates, is_dead)).
    // Order matches CTPROF_METRICS for ease of audit.
    let matrix: &[(&str, Option<&str>, &[&str], bool)] = &[
        // structural: group population count
        ("thread_count", None, &[], false),
        // identity / structural
        ("policy", None, &[], false),
        ("nice", None, &[], false),
        ("priority", None, &[], false),
        ("rt_priority", None, &[], false),
        ("cpu_affinity", None, &[], false),
        ("processor", None, &[], false),
        ("state", None, &[], false),
        ("ext_enabled", None, &["CONFIG_SCHED_CLASS_EXT"], false),
        ("nr_threads", None, &[], false),
        // scheduling / schedstat
        ("run_time_ns", None, &["CONFIG_SCHED_INFO"], false),
        ("wait_time_ns", None, &["CONFIG_SCHED_INFO"], false),
        ("timeslices", None, &["CONFIG_SCHED_INFO"], false),
        ("voluntary_csw", None, &[], false),
        ("nonvoluntary_csw", None, &[], false),
        ("nr_wakeups", None, &["CONFIG_SCHEDSTATS"], false),
        ("nr_wakeups_local", None, &["CONFIG_SCHEDSTATS"], false),
        ("nr_wakeups_remote", None, &["CONFIG_SCHEDSTATS"], false),
        ("nr_wakeups_sync", None, &["CONFIG_SCHEDSTATS"], false),
        ("nr_wakeups_migrate", None, &["CONFIG_SCHEDSTATS"], false),
        (
            "nr_wakeups_affine",
            Some("cfs-only"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        (
            "nr_wakeups_affine_attempts",
            Some("cfs-only"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        ("nr_migrations", None, &[], false),
        (
            "nr_forced_migrations",
            Some("cfs-only"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        (
            "nr_failed_migrations_affine",
            Some("cfs-only"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        (
            "nr_failed_migrations_running",
            Some("cfs-only"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        (
            "nr_failed_migrations_hot",
            Some("cfs-only"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        ("wait_sum", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        ("wait_count", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        ("wait_max", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        (
            "voluntary_sleep_ns",
            Some("non-ext"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        ("sleep_max", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        ("block_sum", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        ("block_max", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        ("iowait_sum", Some("non-ext"), &["CONFIG_SCHEDSTATS"], false),
        (
            "iowait_count",
            Some("non-ext"),
            &["CONFIG_SCHEDSTATS"],
            false,
        ),
        ("exec_max", None, &["CONFIG_SCHEDSTATS"], false),
        ("slice_max", Some("cfs-only"), &["CONFIG_SCHEDSTATS"], false),
        (
            "core_forceidle_sum",
            None,
            &["CONFIG_SCHED_CORE", "CONFIG_SCHEDSTATS"],
            false,
        ),
        ("fair_slice_ns", Some("fair-policy"), &[], false),
        // memory
        ("allocated_bytes", None, &[], false),
        ("deallocated_bytes", None, &[], false),
        ("minflt", None, &[], false),
        ("majflt", None, &[], false),
        ("utime_clock_ticks", None, &[], false),
        ("stime_clock_ticks", None, &[], false),
        // I/O — all 7 fields share CONFIG_TASK_IO_ACCOUNTING
        // (the kernel emits /proc/<tid>/io as a single block
        // under that gate; CONFIG_TASK_IO_ACCOUNTING `depends
        // on` CONFIG_TASK_XACCT in init/Kconfig).
        ("rchar", None, &["CONFIG_TASK_IO_ACCOUNTING"], false),
        ("wchar", None, &["CONFIG_TASK_IO_ACCOUNTING"], false),
        ("syscr", None, &["CONFIG_TASK_IO_ACCOUNTING"], false),
        ("syscw", None, &["CONFIG_TASK_IO_ACCOUNTING"], false),
        ("read_bytes", None, &["CONFIG_TASK_IO_ACCOUNTING"], false),
        ("write_bytes", None, &["CONFIG_TASK_IO_ACCOUNTING"], false),
        (
            "cancelled_write_bytes",
            None,
            &["CONFIG_TASK_IO_ACCOUNTING"],
            false,
        ),
        // taskstats delay accounting — every entry is
        // double-gated on CONFIG_TASKSTATS (the netlink family
        // registration in `kernel/taskstats.c`) and
        // CONFIG_TASK_DELAY_ACCT (the per-task counters in
        // `kernel/delayacct.c`). Operator-visible behavior:
        // missing either gate collapses every field to zero.
        (
            "cpu_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "cpu_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "cpu_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "cpu_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "blkio_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "blkio_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "blkio_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "blkio_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "swapin_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "swapin_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "swapin_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "swapin_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "freepages_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "freepages_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "freepages_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "freepages_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "thrashing_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "thrashing_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "thrashing_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "thrashing_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "compact_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "compact_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "compact_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "compact_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "wpcopy_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "wpcopy_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "wpcopy_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "wpcopy_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "irq_delay_count",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "irq_delay_total_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "irq_delay_max_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "irq_delay_min_ns",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_DELAY_ACCT"],
            false,
        ),
        (
            "hiwater_rss_bytes",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_XACCT"],
            false,
        ),
        (
            "hiwater_vm_bytes",
            None,
            &["CONFIG_TASKSTATS", "CONFIG_TASK_XACCT"],
            false,
        ),
    ];
    // Set-equality: registry keys vs matrix keys.
    let registry_names: std::collections::BTreeSet<&str> =
        CTPROF_METRICS.iter().map(|m| m.name).collect();
    let matrix_names: std::collections::BTreeSet<&str> =
        matrix.iter().map(|(n, _, _, _)| *n).collect();
    assert_eq!(
        registry_names, matrix_names,
        "registry vs matrix key mismatch — every metric must be \
         pinned in the locked matrix and the matrix must not name \
         metrics that aren't registered",
    );
    // Per-entry pin: each tuple matches the registry exactly.
    for (name, expected_class, expected_gates, expected_dead) in matrix {
        let m = lookup_metric(name);
        assert_eq!(m.sched_class, *expected_class, "{name}: sched_class drift",);
        assert_eq!(
            m.config_gates, *expected_gates,
            "{name}: config_gates drift",
        );
        assert_eq!(m.is_dead, *expected_dead, "{name}: is_dead drift");
    }
}

/// Closed-set vocabulary: the registry's tag values must
/// stay inside the documented vocabulary. sched_class is
/// one of {None, "non-ext", "cfs-only", "fair-policy"};
/// config_gates is a subset of the documented kconfig set.
/// Defends against a future entry that tags a metric with a
/// freshly-invented label that the doc / display layers
/// don't yet handle.
#[test]
fn registry_tag_vocabulary_is_closed() {
    let allowed_classes: std::collections::BTreeSet<&str> =
        ["non-ext", "cfs-only", "fair-policy"].into_iter().collect();
    let allowed_gates: std::collections::BTreeSet<&str> = [
        "CONFIG_SCHED_INFO",
        "CONFIG_SCHEDSTATS",
        "CONFIG_SCHED_CORE",
        "CONFIG_TASK_DELAY_ACCT",
        "CONFIG_TASK_IO_ACCOUNTING",
        "CONFIG_TASK_XACCT",
        "CONFIG_SCHED_CLASS_EXT",
        "CONFIG_TASKSTATS",
    ]
    .into_iter()
    .collect();
    for m in CTPROF_METRICS {
        if let Some(class) = m.sched_class {
            assert!(
                allowed_classes.contains(class),
                "{}: sched_class {class:?} outside the closed set \
                 {{None, \"non-ext\", \"cfs-only\", \"fair-policy\"}}",
                m.name,
            );
        }
        for gate in m.config_gates {
            assert!(
                gate.starts_with("CONFIG_"),
                "{}: config_gate {gate:?} must start with CONFIG_",
                m.name,
            );
            assert!(
                allowed_gates.contains(gate),
                "{}: config_gate {gate:?} outside the closed set \
                 {allowed_gates:?}",
                m.name,
            );
        }
    }
}

/// Integration test for `write_diff`: when the `Tags` column
/// is included via `--columns`, a tagged metric row
/// (`nr_wakeups_affine`) renders the bracketed tag string in
/// the dedicated `tags` column. Pins that the registry tag →
/// cell rendering plumbing stays connected end-to-end; the
/// default column set (without `tags`) deliberately omits the
/// bracketed tag string from the `metric` cell so plain
/// listings stay narrow.
#[test]
fn write_diff_renders_tagged_metric_cell() {
    let mut a = make_thread("p", "w");
    a.nr_wakeups_affine = MonotonicCount(5);
    let mut b = make_thread("p", "w");
    b.nr_wakeups_affine = MonotonicCount(9);
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    let mut display = DisplayOptions::default();
    display.columns = vec![
        Column::Group,
        Column::Threads,
        Column::Metric,
        Column::Tags,
        Column::Baseline,
        Column::Candidate,
        Column::Delta,
        Column::Pct,
    ];
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &display,
    )
    .unwrap();
    assert!(
        out.contains("[cfs-only] [SCHEDSTATS]"),
        "tagged metric tags missing from rendered tags column:\n{out}",
    );
    assert!(
        out.contains("nr_wakeups_affine"),
        "tagged metric name missing from rendered table:\n{out}",
    );
}

/// Integration test for `write_diff`: when the `Tags` column
/// is requested, a `non-ext` metric (`wait_sum`) surfaces its
/// `[non-ext] [SCHEDSTATS]` tag string in the dedicated tags
/// column. Pins the matrix change end-to-end so a future
/// regression that rolls the class back to `None` fails
/// here as well as in the unit test.
#[test]
fn write_diff_renders_non_ext_metric_cell() {
    let mut a = make_thread("p", "w");
    a.wait_sum = MonotonicNs(100);
    let mut b = make_thread("p", "w");
    b.wait_sum = MonotonicNs(200);
    let diff = compare(
        &snap_with(vec![a]),
        &snap_with(vec![b]),
        &CompareOptions::default(),
    );
    let mut display = DisplayOptions::default();
    display.columns = vec![
        Column::Group,
        Column::Threads,
        Column::Metric,
        Column::Tags,
        Column::Baseline,
        Column::Candidate,
        Column::Delta,
        Column::Pct,
    ];
    let mut out = String::new();
    write_diff(
        &mut out,
        &diff,
        Path::new("a"),
        Path::new("b"),
        GroupBy::Pcomm,
        &display,
    )
    .unwrap();
    assert!(
        out.contains("[non-ext] [SCHEDSTATS]"),
        "non-ext metric tags missing from rendered tags column:\n{out}",
    );
    assert!(
        out.contains("wait_sum"),
        "non-ext metric name missing from rendered table:\n{out}",
    );
}

/// Show-side `parse_columns` accepts the `metric,value`
/// pair — the show-only allowed vocabulary. Pins that the
/// show-side path actually parses both names rather than
/// silently rejecting `value` as if it were compare-only.
#[test]
fn parse_columns_accepts_show_side_metric_value() {
    let cols = parse_columns("metric,value", false).expect("metric,value is show-side valid");
    assert_eq!(cols, vec![Column::Metric, Column::Value]);
}

/// Empty `metrics` Vec on [`DisplayOptions`] means "every
/// metric is enabled" — the no-filter default. Pins the
/// short-circuit in `is_metric_enabled` so a regression
/// that flipped the empty case to "no metric enabled"
/// surfaces here.
#[test]
fn is_metric_enabled_empty_treats_all_as_on() {
    let opts = DisplayOptions::default();
    // Sample a primary and a derived metric — both must
    // be enabled under the empty default.
    assert!(opts.is_metric_enabled("run_time_ns"));
    assert!(opts.is_metric_enabled("cpu_efficiency"));
    // Even a name not in any registry returns true under
    // the empty filter. is_metric_enabled is the gate at
    // render time; parse_metrics enforces validity at CLI
    // parse time, so these two checks compose to "filter
    // restricts only when populated."
    assert!(opts.is_metric_enabled("anything_under_empty_filter"));
}

/// Non-empty `metrics` Vec restricts rendering to the
/// listed names — names IN the filter return true, names
/// NOT in the filter return false. Pins the contains
/// membership check.
#[test]
fn is_metric_enabled_non_empty_restricts_to_listed() {
    let mut opts = DisplayOptions::default();
    opts.metrics = vec!["run_time_ns", "wait_sum"];
    assert!(opts.is_metric_enabled("run_time_ns"));
    assert!(opts.is_metric_enabled("wait_sum"));
    assert!(!opts.is_metric_enabled("nr_wakeups"));
    assert!(!opts.is_metric_enabled("cpu_efficiency"));
}

/// Registry no longer exposes `voluntary_sleep_sum` as a
/// derived metric — the capture-side `voluntary_sleep_ns`
/// field replaced it. Pin the absence so a future
/// re-introduction surfaces here.
#[test]
fn voluntary_sleep_sum_derived_metric_is_removed() {
    let names: std::collections::BTreeSet<&'static str> =
        CTPROF_DERIVED_METRICS.iter().map(|m| m.name).collect();
    assert!(
        !names.contains("voluntary_sleep_sum"),
        "voluntary_sleep_sum derived metric must not exist — \
         the normalization moved to capture-side \
         `voluntary_sleep_ns` (see ThreadState field doc). \
         Got derived metrics: {names:?}",
    );
}

/// Each `avg_<bucket>_delay_ns` compute closure
/// returns `None` when EITHER input is missing from the
/// metrics map. Pulls the closure directly out of
/// `CTPROF_DERIVED_METRICS` and exercises it with a
/// partial `BTreeMap` (only the numerator side present, no
/// denominator). The compute path must short-circuit via
/// `input_scalar`'s `?` rather than panicking or returning
/// `Some(NaN)`.
///
/// `total_offcpu_delay_ns` follows the same pattern: every
/// input must be present; missing any one returns `None`.
/// The all-inputs-present-but-zero case is covered by the
/// extension to `derived_division_by_zero_returns_none`
/// below.
#[test]
fn derived_avg_delay_ns_returns_none_on_missing_input() {
    let lookup = |name: &str| -> &DerivedMetricDef {
        CTPROF_DERIVED_METRICS
            .iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("{name} present in registry"))
    };

    // For each avg_*: insert ONLY the numerator, not the
    // denominator. The compute closure should return None.
    for (name, numerator) in [
        ("avg_cpu_delay_ns", "cpu_delay_total_ns"),
        ("avg_blkio_delay_ns", "blkio_delay_total_ns"),
        ("avg_swapin_delay_ns", "swapin_delay_total_ns"),
        ("avg_freepages_delay_ns", "freepages_delay_total_ns"),
        ("avg_thrashing_delay_ns", "thrashing_delay_total_ns"),
        ("avg_compact_delay_ns", "compact_delay_total_ns"),
        ("avg_wpcopy_delay_ns", "wpcopy_delay_total_ns"),
        ("avg_irq_delay_ns", "irq_delay_total_ns"),
    ] {
        let mut metrics: BTreeMap<String, Aggregated> = BTreeMap::new();
        metrics.insert(numerator.to_string(), Aggregated::Sum(123));
        let def = lookup(name);
        assert!(
            (def.compute)(&metrics).is_none(),
            "{name}: compute must return None when denominator is \
             missing from metrics map (only {numerator} present)",
        );
    }

    // total_offcpu_delay_ns: insert all but ONE input
    // (`compact_delay_total_ns`). Verify None.
    let mut partial: BTreeMap<String, Aggregated> = BTreeMap::new();
    for name in [
        "cpu_delay_total_ns",
        "blkio_delay_total_ns",
        "swapin_delay_total_ns",
        "freepages_delay_total_ns",
        "thrashing_delay_total_ns",
        // compact_delay_total_ns INTENTIONALLY OMITTED
        "wpcopy_delay_total_ns",
        "irq_delay_total_ns",
    ] {
        partial.insert(name.to_string(), Aggregated::Sum(100));
    }
    let total_def = lookup("total_offcpu_delay_ns");
    assert!(
        (total_def.compute)(&partial).is_none(),
        "total_offcpu_delay_ns: compute must return None when ANY \
         input is missing — exercised here with compact_delay_total_ns \
         omitted from the metrics map",
    );
}

/// `--sort-by` accepts derived metric names. Three groups
/// with distinct cpu_efficiency values: sort descending puts
/// the highest first.
#[test]
fn parse_sort_by_accepts_derived_metric_name() {
    let keys = parse_sort_by("cpu_efficiency").expect("derived name parses");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].metric, "cpu_efficiency");
    assert!(keys[0].descending);
}

/// Primary and derived metric namespaces are disjoint — a
/// derived metric may NOT shadow a primary metric name. Pin
/// the disjoint invariant so a future addition that
/// accidentally collides surfaces here.
#[test]
fn registry_and_derived_names_disjoint() {
    let primary: std::collections::BTreeSet<&str> =
        CTPROF_METRICS.iter().map(|m| m.name).collect();
    for d in CTPROF_DERIVED_METRICS {
        assert!(
            !primary.contains(d.name),
            "derived metric {} shadows primary registry name",
            d.name,
        );
    }
}

/// Every derived metric has a non-empty description and a
/// non-empty inputs list. Defends against a future addition
/// that forgets to fill either field.
#[test]
fn registry_derived_metrics_well_formed() {
    for d in CTPROF_DERIVED_METRICS {
        assert!(
            !d.description.is_empty(),
            "derived metric {} has empty description",
            d.name,
        );
        assert!(
            !d.inputs.is_empty(),
            "derived metric {} has empty inputs list",
            d.name,
        );
        // Every input must be a real registered metric name.
        let primary: std::collections::BTreeSet<&str> =
            CTPROF_METRICS.iter().map(|m| m.name).collect();
        for input in d.inputs {
            assert!(
                primary.contains(input),
                "derived metric {} cites unknown input {input}",
                d.name,
            );
        }
    }
}

/// `metric-list` emits the `## Derived metrics` section
/// with every registered derivation listed. Pin set-equality
/// on names so a registry addition automatically surfaces.
#[test]
fn write_metric_list_emits_derived_section() {
    let mut out = String::new();
    write_metric_list(&mut out).unwrap();
    assert!(
        out.contains("## Derived metrics"),
        "metric-list must emit a Derived metrics header:\n{out}",
    );
    for d in CTPROF_DERIVED_METRICS {
        assert!(
            out.contains(d.name),
            "derived metric {} missing from metric-list:\n{out}",
            d.name,
        );
    }
}

/// `metric-list` emits the `## Sections` table listing
/// every Section variant by its CLI name. Discovery
/// companion to the `--sections` flag — operators reading
/// the rendered metric-list output should see the full
/// vocabulary for `--sections` without needing to read
/// source. Pin every cli_name from `Section::ALL`.
#[test]
fn write_metric_list_emits_sections_vocabulary() {
    let mut out = String::new();
    write_metric_list(&mut out).unwrap();
    assert!(
        out.contains("## Sections"),
        "metric-list must emit the Sections vocabulary heading:\n{out}",
    );
    for section in Section::ALL {
        assert!(
            out.contains(section.cli_name()),
            "section cli_name {} missing from Sections \
             vocabulary table:\n{out}",
            section.cli_name(),
        );
    }
}

/// The Sections vocabulary appears BEFORE the Metrics
/// table in the rendered output. Pins the layout order
/// so a future refactor that moves Sections after Metrics
/// (or drops the heading entirely) surfaces here.
#[test]
fn write_metric_list_sections_precedes_metrics() {
    let mut out = String::new();
    write_metric_list(&mut out).unwrap();
    let sections_at = out
        .find("## Sections")
        .expect("Sections heading must be present");
    let metrics_at = out
        .find("## Metrics")
        .expect("Metrics heading must be present");
    assert!(
        sections_at < metrics_at,
        "Sections heading must precede Metrics heading; \
         got Sections@{sections_at} Metrics@{metrics_at}\n{out}",
    );
}

/// `--sort-by avg_wait_ns` ranks groups by the derived
/// metric's delta. End-to-end pin: three pcomm buckets with
/// distinct avg_wait_ns deltas; descending sort puts the
/// largest delta's group first in the rendered table.
#[test]
fn write_diff_sort_by_derived_metric_ranks_groups() {
    // bucket "high": avg_wait grew from 100ns to 300ns (+200ns)
    // bucket "low": avg_wait grew from 100ns to 150ns (+50ns)
    // Descending sort puts "high" first.
    let mut high_a = make_thread("p", "w");
    high_a.pcomm = "high".to_string();
    high_a.wait_sum = MonotonicNs(100);
    high_a.wait_count = MonotonicCount(1);
    let mut high_b = make_thread("p", "w");
    high_b.pcomm = "high".to_string();
    high_b.wait_sum = MonotonicNs(300);
    high_b.wait_count = MonotonicCount(1);
    let mut low_a = make_thread("p", "w");
    low_a.pcomm = "low".to_string();
    low_a.wait_sum = MonotonicNs(100);
    low_a.wait_count = MonotonicCount(1);
    let mut low_b = make_thread("p", "w");
    low_b.pcomm = "low".to_string();
    low_b.wait_sum = MonotonicNs(150);
    low_b.wait_count = MonotonicCount(1);
    let opts = CompareOptions {
        sort_by: vec![SortKey {
            metric: "avg_wait_ns",
            descending: true,
        }],
        ..CompareOptions::default()
    };
    let diff = compare(
        &snap_with(vec![high_a, low_a]),
        &snap_with(vec![high_b, low_b]),
        &opts,
    );
    // Find the first derived row (post-sort) — the group with
    // the largest avg_wait_ns delta.
    let first = &diff.derived_rows[0];
    assert_eq!(
        first.group_key, "high",
        "descending sort by avg_wait_ns must put `high` first; \
         got {:?}",
        first.group_key,
    );
}

/// `write_metric_list` emits the tag legend section with
/// every closed-set tag value documented. Ties the legend
/// content to the closed-set vocabulary the registry pin
/// guards (`registry_tag_vocabulary_is_closed`); a future
/// allowed-class or allowed-gate addition that doesn't
/// extend the legend fails this test.
#[test]
fn write_metric_list_emits_full_tag_legend() {
    let mut out = String::new();
    write_metric_list(&mut out).unwrap();
    // sched_class vocabulary
    assert!(
        out.contains("[cfs-only]"),
        "missing [cfs-only] in legend:\n{out}"
    );
    assert!(
        out.contains("[non-ext]"),
        "missing [non-ext] in legend:\n{out}"
    );
    assert!(
        out.contains("[fair-policy]"),
        "missing [fair-policy] in legend:\n{out}",
    );
    // config_gates vocabulary (compact form)
    assert!(
        out.contains("[SCHED_INFO]"),
        "missing [SCHED_INFO] in legend:\n{out}"
    );
    assert!(
        out.contains("[SCHEDSTATS]"),
        "missing [SCHEDSTATS] in legend:\n{out}",
    );
    assert!(
        out.contains("[SCHED_CORE]"),
        "missing [SCHED_CORE] in legend:\n{out}"
    );
    assert!(
        out.contains("[SCHED_CLASS_EXT]"),
        "missing [SCHED_CLASS_EXT] in legend:\n{out}",
    );
    assert!(
        out.contains("[TASK_DELAY_ACCT]"),
        "missing [TASK_DELAY_ACCT] in legend:\n{out}",
    );
    assert!(
        out.contains("[TASK_IO_ACCOUNTING]"),
        "missing [TASK_IO_ACCOUNTING] in legend:\n{out}",
    );
    assert!(
        out.contains("[TASKSTATS]"),
        "missing [TASKSTATS] in legend:\n{out}",
    );
    assert!(
        out.contains("[TASK_XACCT]"),
        "missing [TASK_XACCT] in legend:\n{out}",
    );
    // status vocabulary
    assert!(out.contains("[dead]"), "missing [dead] in legend:\n{out}");
    // Section headers
    assert!(
        out.contains("## Tag legend"),
        "missing Tag legend section header:\n{out}",
    );
    assert!(
        out.contains("## Metrics"),
        "missing Metrics section header:\n{out}",
    );
}

/// `write_metric_list` covers every metric in the registry.
/// Pin set-equality on the names so a registry addition
/// fails the test until the description is added (which
/// happens automatically — `write_metric_list` iterates the
/// registry).
#[test]
fn write_metric_list_covers_every_registered_metric() {
    let mut out = String::new();
    write_metric_list(&mut out).unwrap();
    for m in CTPROF_METRICS {
        assert!(
            out.contains(m.name),
            "metric {} missing from metric-list output:\n{out}",
            m.name,
        );
        assert!(
            out.contains(m.description),
            "description for {} missing from metric-list output:\n{out}",
            m.name,
        );
    }
}

/// `write_metric_list` puts the tags into their own column —
/// no metric name leaks into the tags cell. Pin a
/// representative example: `nr_wakeups_affine` carries
/// `[cfs-only] [SCHEDSTATS]`, and that exact substring (with
/// a leading space gap before the bracket) is present in the
/// output but the rendered display form
/// `nr_wakeups_affine [cfs-only] [SCHEDSTATS]` is NOT (which
/// would mean the name leaked into the tags cell).
#[test]
fn write_metric_list_tags_column_excludes_metric_name() {
    let mut out = String::new();
    write_metric_list(&mut out).unwrap();
    assert!(
        out.contains("[cfs-only] [SCHEDSTATS]"),
        "expected bare tag pair `[cfs-only] [SCHEDSTATS]` in tags column:\n{out}",
    );
    assert!(
        !out.contains("nr_wakeups_affine [cfs-only]"),
        "metric name must not leak into tags column:\n{out}",
    );
}

/// Every metric carries a non-empty description string.
/// Defends against a future metric addition that forgets to
/// fill the field — leaving an empty cell in the discovery
/// output that defeats the entire purpose of `metric-list`.
#[test]
fn registry_descriptions_are_non_empty() {
    for m in CTPROF_METRICS {
        assert!(
            !m.description.is_empty(),
            "metric {} has empty description",
            m.name,
        );
        // No trailing whitespace, no leading whitespace —
        // the table cell carries the description verbatim.
        assert_eq!(
            m.description.trim(),
            m.description,
            "metric {} description has leading/trailing whitespace",
            m.name,
        );
    }
}

/// Each `*_max` metric in the registry reads the matching
/// per-thread field — guards against a copy-paste mistake
/// like `Max(|t| t.wait_max.0)` for the `block_max` slot.
/// Mirrors `sum_metric_accessors_read_expected_field` for
/// the Max family.
#[test]
fn max_metric_accessors_read_expected_field() {
    type MetricSetter = fn(&mut ThreadState);
    let cases: &[(&str, MetricSetter)] = &[
        ("wait_max", |t| t.wait_max = PeakNs(1)),
        ("sleep_max", |t| t.sleep_max = PeakNs(1)),
        ("block_max", |t| t.block_max = PeakNs(1)),
        ("exec_max", |t| t.exec_max = PeakNs(1)),
        ("slice_max", |t| t.slice_max = PeakNs(1)),
    ];
    for (name, set) in cases {
        let mut t = make_thread("p", "w");
        set(&mut t);
        let def = CTPROF_METRICS
            .iter()
            .find(|m| m.name == *name)
            .unwrap_or_else(|| panic!("metric {name} not in registry"));
        let agg = aggregate(def.rule, &[&t]);
        match agg {
            Aggregated::Max(v) => {
                assert_eq!(v, 1, "accessor for {name} did not read the {name} field")
            }
            other => panic!("expected Max for {name}, got {other:?}"),
        }
    }
}

/// Bare metric name surrounded by whitespace (no colon, no
/// direction) parses as a single descending key. Pins the
/// metric-side trim path on the `None` arm of the
/// `split_once(':')` match — `entry.trim()` runs first to
/// strip the entry-level whitespace, then the `None` arm
/// passes the trimmed string straight through. A regression
/// that dropped either trim layer would surface here as a
/// failed registry lookup on the literal `"  wait_sum  "`.
#[test]
fn parse_sort_by_bare_metric_with_whitespace_no_colon() {
    let keys = parse_sort_by("  wait_sum  ").expect("bare-metric whitespace must parse");
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].metric, "wait_sum");
    assert!(keys[0].descending);
}

/// Metric name with trailing colon and no direction
/// (`"wait_sum:"`) splits to (`"wait_sum"`, `""`). The
/// empty direction is not `asc` or `desc`, so the
/// bad-direction arm fires. A regression that treated empty
/// direction as the default `desc` would silently accept
/// the typo.
#[test]
fn parse_sort_by_rejects_metric_colon_no_direction() {
    let err = parse_sort_by("wait_sum:").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("invalid direction"),
        "metric-colon-no-direction must surface as invalid-direction error, got: {msg}"
    );
}

/// Unknown-metric error message lists the valid registry
/// entries as a sorted comma-separated list (not a
/// `BTreeSet` debug dump). Pins the operator-facing shape:
/// the diagnostic is copy-pasteable and the names appear in
/// alphabetical order so the operator can scan for the one
/// they meant.
#[test]
fn parse_sort_by_unknown_metric_lists_valid_names_sorted() {
    let err = parse_sort_by("not_a_real_metric").unwrap_err();
    let msg = format!("{err:#}");
    // The list is comma-separated. Find two known-adjacent
    // names from the sorted set and pin their relative
    // order in the diagnostic.
    // In alphabetical order, "nice" comes before
    // "policy" and "policy" before "run_time_ns" (registry
    // names live mostly under the `n…` / `p…` / `r…`
    // namespaces). Pick a triple whose alphabetical order
    // is unambiguous.
    let nice_at = msg
        .find("nice")
        .expect("error must list 'nice' from the registry");
    let policy_at = msg
        .find("policy")
        .expect("error must list 'policy' from the registry");
    let run_time_at = msg
        .find("run_time_ns")
        .expect("error must list 'run_time_ns' from the registry");
    assert!(
        nice_at < policy_at,
        "names must appear in alphabetical order: \
         nice@{nice_at} < policy@{policy_at}\nmsg: {msg}",
    );
    assert!(
        policy_at < run_time_at,
        "names must appear in alphabetical order: \
         policy@{policy_at} < run_time_ns@{run_time_at}\nmsg: {msg}",
    );
    // Format must be comma-separated, not BTreeSet debug
    // (`{...}`). Pin the absence of the debug-set delimiters.
    assert!(
        !msg.contains("{\""),
        "error must use comma-separated list, not BTreeSet debug dump:\n{msg}"
    );
}

/// Multi-key sort spec preserves entry order in the
/// returned Vec (left-to-right). Pins the documented
/// "lexicographic in input order" contract — a reordering
/// regression would silently rank by the second key first.
#[test]
fn parse_sort_by_multi_key_preserves_order() {
    // Three keys, distinct names — pick one each from the
    // ns / unitless / count axes so the entries are visibly
    // distinct.
    let keys =
        parse_sort_by("run_time_ns:desc,nr_wakeups:asc,wait_time_ns:desc").expect("parse");
    assert_eq!(keys.len(), 3);
    assert_eq!(keys[0].metric, "run_time_ns");
    assert!(keys[0].descending);
    assert_eq!(keys[1].metric, "nr_wakeups");
    assert!(!keys[1].descending);
    assert_eq!(keys[2].metric, "wait_time_ns");
    assert!(keys[2].descending);
}

/// Multi-key sort: groups rank by the requested metrics'
/// deltas in tuple order. Big regression on the FIRST key
/// dominates regardless of the second key.
///
/// Exercises `sort_diff_rows_by_keys` directly on synthetic
/// `DiffRow` values rather than driving through `compare()`
/// — the function under test is the sort, not the diff
/// pipeline; building the diff via `compare(empty, full)`
/// would route every group into `only_baseline` /
/// `only_candidate` rather than producing the matched-group
/// rows the sort consumes.
#[test]
fn sort_diff_rows_by_keys_ranks_by_first_key_first() {
    // Build synthetic rows: 3 groups × 2 metrics = 6 rows.
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        mk_row("A", "run_time_ns", 1000.0),
        mk_row("A", "wait_sum", 100.0),
        mk_row("B", "run_time_ns", 100.0),
        mk_row("B", "wait_sum", 1000.0),
        mk_row("C", "run_time_ns", 50.0),
        mk_row("C", "wait_sum", 50.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let groups_in_order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    // A has run_time_ns 1000 → first. B has 100 → second. C has 50 → third.
    // Each group's two rows cluster together in registry
    // order (run_time_ns before wait_sum).
    assert_eq!(
        groups_in_order,
        vec!["A", "A", "B", "B", "C", "C"],
        "groups should rank by run_time_ns delta desc",
    );
    // Within each group: run_time_ns row comes first
    // (registry index lower than wait_sum).
    let metrics_first_two: Vec<&str> = rows.iter().take(2).map(|r| r.metric_name).collect();
    assert_eq!(metrics_first_two, vec!["run_time_ns", "wait_sum"]);
}

/// Multi-key sort tie-break: when the first key value is
/// equal across groups, the second key disambiguates. Two
/// groups with the same run_time_ns delta but different
/// wait_sum deltas: the one with the larger wait_sum delta
/// sorts first (under desc,desc).
#[test]
fn sort_diff_rows_by_keys_breaks_ties_with_second_key() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        // A and B tie on run_time_ns (both 500). Use wait_sum
        // to break: A.wait_sum delta is 100, B.wait_sum delta
        // is 200. Under desc,desc → B first.
        mk_row("A", "run_time_ns", 500.0),
        mk_row("A", "wait_sum", 100.0),
        mk_row("B", "run_time_ns", 500.0),
        mk_row("B", "wait_sum", 200.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[
            SortKey {
                metric: "run_time_ns",
                descending: true,
            },
            SortKey {
                metric: "wait_sum",
                descending: true,
            },
        ],
    );
    let groups_in_order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    assert_eq!(groups_in_order, vec!["B", "B", "A", "A"]);
}

/// Ascending direction reverses the sort. Group with the
/// SMALLEST delta should sort first under `:asc`.
#[test]
fn sort_diff_rows_by_keys_respects_ascending_direction() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        mk_row("A", "run_time_ns", 1000.0),
        mk_row("B", "run_time_ns", 100.0),
        mk_row("C", "run_time_ns", 500.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: false, // asc
        }],
    );
    let groups_in_order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    // B (100) < C (500) < A (1000) under asc.
    assert_eq!(groups_in_order, vec!["B", "C", "A"]);
}

/// End-to-end: `compare()` with a non-empty sort_by uses the
/// multi-key path. Pin that two groups with different
/// run_time_ns deltas surface in the operator-requested
/// order, regardless of which group has the larger
/// |delta_pct| (which would have won under the default sort).
#[test]
fn compare_uses_sort_by_when_set() {
    let mut a_pre = make_thread("alpha", "w");
    a_pre.run_time_ns = MonotonicNs(1_000_000_000); // 1B baseline → big abs but tiny pct change
    let mut a_post = make_thread("alpha", "w");
    a_post.run_time_ns = MonotonicNs(1_000_000_500); // +500 abs; +5e-5 % change
    let mut b_pre = make_thread("bravo", "w");
    b_pre.run_time_ns = MonotonicNs(100);
    let mut b_post = make_thread("bravo", "w");
    b_post.run_time_ns = MonotonicNs(200); // +100 abs; +100% change
    // Default sort: bravo wins by |delta_pct|. With
    // sort_by=run_time_ns:desc, alpha wins by absolute delta
    // (500 > 100).
    let diff = compare(
        &snap_with(vec![a_pre, b_pre]),
        &snap_with(vec![a_post, b_post]),
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: vec![SortKey {
                metric: "run_time_ns",
                descending: true,
            }],
        },
    );
    let run_rows: Vec<&DiffRow> = diff
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .collect();
    assert_eq!(
        run_rows[0].group_key, "alpha",
        "sort_by abs delta picks alpha"
    );
    assert_eq!(run_rows[1].group_key, "bravo");
}

/// Final tie-break: when every sort-key value matches across
/// groups, `sort_diff_rows_by_keys` falls through to ascending
/// `group_key` ordering for deterministic output. Pins the
/// last branch in the comparator (`a.cmp(b)`) — without it,
/// equal-delta groups would emerge in BTreeMap-iteration order
/// dependent on hash, which would produce flaky test output.
#[test]
fn sort_diff_rows_by_keys_falls_back_to_ascending_group_key_on_full_tie() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    // Three groups with IDENTICAL deltas — only the
    // group_key tie-break can deterministically order them.
    // Insert in reverse-alphabetical order so the test fails
    // if the tie-break is dropped (BTreeMap iteration would
    // already produce ascending — distinguishable only via
    // explicit reverse-input ordering).
    let mut rows = vec![
        mk_row("charlie", "run_time_ns", 100.0),
        mk_row("bravo", "run_time_ns", 100.0),
        mk_row("alpha", "run_time_ns", 100.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    assert_eq!(
        order,
        vec!["alpha", "bravo", "charlie"],
        "full sort-key tie must fall back to ascending group_key",
    );
}

/// Missing-metric handling under descending direction:
/// when a group has no row for the named metric (or its
/// row's `delta` is `None`), `sort_diff_rows_by_keys`
/// substitutes `f64::NEG_INFINITY` so the group sinks to
/// the bottom under desc. Pin the documented contract — a
/// regression that used 0.0 (or panicked) would surface
/// here.
#[test]
fn sort_diff_rows_by_keys_missing_metric_sinks_under_desc() {
    let mk_row = |group: &str, metric: &'static str, delta: Option<f64>| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta,
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        // alpha has a real run_time_ns delta.
        mk_row("alpha", "run_time_ns", Some(100.0)),
        // bravo has only a wait_time_ns row — its run_time_ns
        // tuple value is missing → NEG_INFINITY under desc.
        mk_row("bravo", "wait_time_ns", Some(999_999.0)),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    // Recover unique group ordering.
    let mut order: Vec<&str> = Vec::new();
    for r in &rows {
        if !order.contains(&r.group_key.as_str()) {
            order.push(r.group_key.as_str());
        }
    }
    assert_eq!(
        order,
        vec!["alpha", "bravo"],
        "missing metric under desc must sink the group (NEG_INFINITY)",
    );
}

/// Missing-metric handling under ascending direction:
/// when the named metric is missing, `sort_diff_rows_by_keys`
/// substitutes `f64::INFINITY` so the group sinks to the
/// bottom under asc. Mirror of the desc test — same shape,
/// inverted polarity. Together they pin both arms of the
/// `if k.descending` branch in the fallback.
#[test]
fn sort_diff_rows_by_keys_missing_metric_sinks_under_asc() {
    let mk_row = |group: &str, metric: &'static str, delta: Option<f64>| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta,
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        // alpha has a real (positive) run_time_ns delta.
        mk_row("alpha", "run_time_ns", Some(100.0)),
        // bravo has only a wait_time_ns row — its run_time_ns
        // tuple value is missing → INFINITY under asc.
        mk_row("bravo", "wait_time_ns", Some(50.0)),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: false,
        }],
    );
    let mut order: Vec<&str> = Vec::new();
    for r in &rows {
        if !order.contains(&r.group_key.as_str()) {
            order.push(r.group_key.as_str());
        }
    }
    assert_eq!(
        order,
        vec!["alpha", "bravo"],
        "missing metric under asc must sink the group (INFINITY)",
    );
}

/// Categorical-only group: every row's `delta` is `None`
/// (the group's metric is Mode and delta math doesn't
/// apply), but the group still appears in `rows`.
/// `sort_diff_rows_by_keys` must surface the group with
/// the missing-metric fallback applied — no panic, no row
/// dropped. This guards the second loop in the function
/// that adds groups present in `rows` but absent from
/// `group_metrics`.
#[test]
fn sort_diff_rows_by_keys_categorical_only_group_does_not_panic() {
    let mk_row = |group: &str, metric: &'static str| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::mode_single("SCHED_OTHER".into(), 1, 1),
        candidate: Aggregated::mode_single("SCHED_OTHER".into(), 1, 1),
        // `Mode` rows carry `delta: None` because mode
        // metrics have no scalar projection — see
        // `Aggregated::numeric()`.
        delta: None,
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![mk_row("alpha", "policy"), mk_row("bravo", "policy")];
    // Sort by run_time_ns — neither group has it, both fall
    // through to the missing-metric fallback. Final tie-break
    // (`a.cmp(b)`) breaks the tie ascending.
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let order: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    assert_eq!(
        order,
        vec!["alpha", "bravo"],
        "categorical-only groups must survive the sort and fall to ascending group_key",
    );
}

/// Within a group, rows appear in `CTPROF_METRICS`
/// registry order regardless of input order or sort spec.
/// Pins the documented "rows within a group keep registry
/// order" contract — a regression that ordered metric rows
/// by `metric_name` lexicographically (or by sort_key
/// position) would produce non-deterministic per-bucket
/// layouts.
#[test]
fn sort_diff_rows_by_keys_within_group_uses_registry_order() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    // Use four metrics from the scheduling block in their
    // registry order: run_time_ns (idx 6), wait_time_ns (7),
    // timeslices (8), nr_wakeups (11). Insert in
    // REVERSE-registry order so a regression that orders by
    // input/sort-spec/lexicographic would surface as a
    // visibly wrong metric_order assertion.
    let mut rows = vec![
        mk_row("alpha", "nr_wakeups", 4.0),
        mk_row("alpha", "timeslices", 3.0),
        mk_row("alpha", "wait_time_ns", 999.0),
        mk_row("alpha", "run_time_ns", 1.0),
    ];
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            // Sort by wait_time_ns to verify the metric
            // rows still emerge in REGISTRY order, not
            // sort-spec order (which would put wait_time_ns
            // first).
            metric: "wait_time_ns",
            descending: true,
        }],
    );
    let metric_order: Vec<&str> = rows.iter().map(|r| r.metric_name).collect();
    assert_eq!(
        metric_order,
        vec!["run_time_ns", "wait_time_ns", "timeslices", "nr_wakeups"],
        "within-group order must be registry, not sort-spec, order",
    );
}

/// NaN-safe partial_cmp: a `delta` that's NaN must not
/// panic the sort. `partial_cmp` returns `None` for NaN,
/// which the comparator maps to `Ordering::Equal` so the
/// remaining keys (or the group_key tie-break) decide. Pin
/// that the function survives the NaN input — without the
/// `unwrap_or(Equal)` in both arms, the sort would panic on
/// the implicit `unwrap()` of an arithmetic NaN result.
#[test]
fn sort_diff_rows_by_keys_nan_delta_does_not_panic() {
    let mk_row = |group: &str, metric: &'static str, delta: f64| DiffRow {
        group_key: group.into(),
        thread_count_a: 1,
        thread_count_b: 1,
        metric_name: metric,
        metric_ladder: ScaleLadder::None,
        baseline: Aggregated::Sum(0),
        candidate: Aggregated::Sum(0),
        delta: Some(delta),
        delta_pct: None,
        display_key: group.into(),
        uptime_pct: None,
        sort_by_cell: None,
        sort_by_delta: None,
    };
    let mut rows = vec![
        mk_row("alpha", "run_time_ns", f64::NAN),
        mk_row("bravo", "run_time_ns", 100.0),
        mk_row("charlie", "run_time_ns", f64::NAN),
    ];
    // The function call must not panic; output ordering is
    // unspecified for NaN-vs-NaN beyond the group_key
    // tie-break, so we only assert that all three groups
    // survive the sort.
    sort_diff_rows_by_keys(
        &mut rows,
        &mut Vec::new(),
        &[SortKey {
            metric: "run_time_ns",
            descending: true,
        }],
    );
    let mut groups: Vec<&str> = rows.iter().map(|r| r.group_key.as_str()).collect();
    groups.sort();
    groups.dedup();
    assert_eq!(
        groups,
        vec!["alpha", "bravo", "charlie"],
        "NaN delta must not drop or duplicate any group",
    );
}

/// `compare()` with empty `sort_by` routes through the
/// default `delta_pct desc` sort, NOT `sort_diff_rows_by_keys`.
/// Pin the routing branch by exercising the same data
/// shape under both `sort_by: empty` and `sort_by: [...]`
/// and confirming they produce *different* orderings.
/// Together with `compare_uses_sort_by_when_set` (the
/// non-empty branch above), this pins both arms of the
/// `if opts.sort_by.is_empty()` check inside `compare()`.
#[test]
fn compare_uses_default_sort_when_sort_by_empty() {
    // `alpha` has 1B baseline, +500 delta → tiny |delta_pct|.
    // `bravo` has 100 baseline, +100 delta → +100% delta_pct.
    // Default sort ranks by |delta_pct| desc → bravo first.
    let mut a_pre = make_thread("alpha", "w");
    a_pre.run_time_ns = MonotonicNs(1_000_000_000);
    let mut a_post = make_thread("alpha", "w");
    a_post.run_time_ns = MonotonicNs(1_000_000_500);
    let mut b_pre = make_thread("bravo", "w");
    b_pre.run_time_ns = MonotonicNs(100);
    let mut b_post = make_thread("bravo", "w");
    b_post.run_time_ns = MonotonicNs(200);

    // Empty sort_by → default delta_pct desc.
    let diff_default = compare(
        &snap_with(vec![a_pre.clone(), b_pre.clone()]),
        &snap_with(vec![a_post.clone(), b_post.clone()]),
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let default_order: Vec<&str> = diff_default
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .map(|r| r.group_key.as_str())
        .collect();
    assert_eq!(
        default_order,
        vec!["bravo", "alpha"],
        "empty sort_by must use default delta_pct desc sort \
         (bravo's +100% beats alpha's +5e-5 %)",
    );

    // Non-empty sort_by → multi-key. Picks alpha first by
    // absolute delta (+500 > +100).
    let diff_sort = compare(
        &snap_with(vec![a_pre, b_pre]),
        &snap_with(vec![a_post, b_post]),
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: vec![SortKey {
                metric: "run_time_ns",
                descending: true,
            }],
        },
    );
    let sort_order: Vec<&str> = diff_sort
        .rows
        .iter()
        .filter(|r| r.metric_name == "run_time_ns")
        .map(|r| r.group_key.as_str())
        .collect();
    assert_eq!(
        sort_order,
        vec!["alpha", "bravo"],
        "non-empty sort_by must use multi-key path (alpha's +500 abs beats bravo's +100)",
    );

    // The two orderings differ — pins that the routing
    // actually swaps paths, not just produces the same
    // result by coincidence.
    assert_ne!(
        default_order, sort_order,
        "empty vs non-empty sort_by must produce different orderings on this fixture",
    );
}

/// Auto-scale edge case: zero values render as bare
/// `0<unit>` across all five unit families. Pin that the
/// `abs() >= threshold` chain short-circuits to "no
/// step-up" at zero and the integer fast-path renders
/// `0ns`, `0µs`, `0B`, `0ticks`, and `0` (the empty-unit
/// case). A regression that flipped the threshold to `>`
/// (so `abs >= 0` matches and the chain over-steps to the
/// largest unit) would surface here.
#[test]
fn format_scaled_u64_zero_renders_at_base_unit_for_all_families() {
    assert_eq!(format_scaled_u64(0, ScaleLadder::Ns), "0ns");
    assert_eq!(format_scaled_u64(0, ScaleLadder::Us), "0µs");
    assert_eq!(format_scaled_u64(0, ScaleLadder::Bytes), "0B");
    assert_eq!(format_scaled_u64(0, ScaleLadder::Ticks), "0ticks");
    // Empty unit: format prints just the integer with no
    // suffix. This is the canonical unitless render path.
    assert_eq!(format_scaled_u64(0, ScaleLadder::Unitless), "0");
}

/// `format_delta_cell` on a negative µs delta auto-scales
/// AND keeps the explicit minus sign. Pin both sides:
/// magnitude is reported in seconds (`-1.500s`, not
/// `-1500000µs`), and the leading `-` survives the scale
/// step.
#[test]
fn format_delta_cell_negative_microseconds_scales_to_seconds() {
    let cell = format_delta_cell(-1_500_000.0, ScaleLadder::Us);
    assert_eq!(cell, "-1.500s");
}

/// `format_delta_cell` on a negative byte delta auto-scales
/// AND keeps the explicit minus sign. Pin the IEC binary
/// path on the negative side; the existing positive-byte
/// path is exercised by other tests but the negative-byte
/// branch was unpinned.
#[test]
fn format_delta_cell_negative_bytes_scales_to_gib() {
    let two_gib_neg = -(2.0 * 1024.0 * 1024.0 * 1024.0);
    let cell = format_delta_cell(two_gib_neg, ScaleLadder::Bytes);
    assert_eq!(cell, "-2.000GiB");
}

/// Asymmetric threshold-crossing: each cell of a
/// `cgroup_cell` triple scales independently. A baseline
/// just below the µs→ms threshold renders as bare µs while
/// the candidate (just above) jumps to ms — and the delta
/// (their difference) picks its own scale based on its own
/// magnitude. Pin that the three cells don't bleed scales
/// into each other.
#[test]
fn cgroup_cell_each_cell_scales_independently() {
    // Baseline 999 µs (below 1000-µs ms threshold) →
    // renders as `999µs`. Candidate 2000 µs (above) → `2.000ms`.
    // Delta +1001 µs (above) → `+1.001ms`.
    let cell = cgroup_cell(Some(999), Some(2000), ScaleLadder::Us);
    assert_eq!(
        cell, "999µs → 2.000ms (+1.001ms)",
        "asymmetric scaling: each cell must pick its own prefix",
    );
}
