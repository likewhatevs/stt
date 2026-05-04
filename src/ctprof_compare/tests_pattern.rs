//! Tests for `super::pattern` (Phase F.2 per-module redistribution).

#![allow(unused_imports)]
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeMap;
use std::path::Path;

use super::aggregate::{format_cpu_range, merge_aggregated_into};
use super::cgroup_merge::{
    merge_cgroup_cpu, merge_cgroup_memory, merge_cgroup_pids, merge_kv_counters, merge_max_option,
    merge_memory_stat, merge_min_option, merge_psi,
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
use super::*;
use crate::ctprof::{CgroupStats, CtprofSnapshot, Psi, ThreadState};
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32, PeakNs,
};
use regex::Regex;

/// Strip-trailing-digit happy path with a variety of separator
/// chars in the prefix: `tokio-worker-12` → `tokio-worker-{N}`,
/// `worker_5` → `worker_{N}`, etc. Pins that a separator before
/// Token-based normalizer: every separator-delimited
/// digit-run is replaced with `{N}` (rule 1), every
/// alpha-prefix-plus-digits token with `prefix{N}` (rule 3).
/// Embedded digit tokens between separators normalize too —
/// `pool-2-thread-7` collapses to `pool-{N}-thread-{N}`. This
/// is the new spec behavior; under the legacy algorithm only
/// the trailing run was stripped.
#[test]
fn pattern_key_strips_trailing_digits() {
    assert_eq!(pattern_key("tokio-worker-12"), "tokio-worker-{N}");
    assert_eq!(pattern_key("worker_5"), "worker_{N}");
    assert_eq!(pattern_key("rayon.pool.7"), "rayon.pool.{N}");
    // Whitespace-separated tokens normalize per the same
    // rules; the run of whitespace separator chars is
    // preserved verbatim.
    assert_eq!(pattern_key("Chrome thread 4"), "Chrome thread {N}");
    // Embedded digit tokens between separators each
    // normalize.
    assert_eq!(pattern_key("pool-2-thread-7"), "pool-{N}-thread-{N}");
}

/// Bare-numeric and dangling-separator inputs.
/// - `"0"` is a single pure-digit token → `{N}` (rule 1).
/// - `"worker-"` is `[Token("worker"), Separator("-")]`; the
///   trailing separator is preserved verbatim, the alpha
///   token has no digits so it stays literal → `worker-`.
#[test]
fn pattern_key_bare_numeric_and_dangling_separator() {
    assert_eq!(pattern_key("0"), "{N}");
    assert_eq!(pattern_key("worker-"), "worker-");
}

/// AlphaPrefix (no separator before the digit run) groups when
/// the prefix length passes the min-prefix gate. This catches
/// CamelCase names like `CamelCaseWord0`/`CamelCaseWord1`/...
/// that compose 40% of the unobserved coverage gap on
/// many-CPU hosts. `cpu0` (prefix `cpu` exactly 3 chars) groups
/// — correct on hosts where every CPU spawns one such thread.
#[test]
fn pattern_key_alpha_prefix_groups_without_separator() {
    assert_eq!(pattern_key("CamelCaseWord0"), "CamelCaseWord{N}");
    assert_eq!(pattern_key("CamelCaseWord175"), "CamelCaseWord{N}");
    assert_eq!(pattern_key("worker7"), "worker{N}");
    // 3-char prefix is the min boundary — `cpu` is exactly 3 chars.
    assert_eq!(pattern_key("cpu0"), "cpu{N}");
    // No trailing digits at all → stays literal.
    assert_eq!(pattern_key("init"), "init");
}

/// Single-letter alpha prefix in a delimited token normalizes
/// (rule 3 with alpha prefix length ≥ 1) — when the alpha char
/// is OUTSIDE `[0-9a-f]`. If the alpha char is inside that
/// range, rule 2 (hex) fires first (it precedes rule 3). So:
/// - `v` is outside → `gadget-v2` → `gadget-v{N}` (rule 3).
/// - `r` is outside → `thingo-r2` → `thingo-r{N}` (rule 3).
/// - `t` is outside → `t1` → `t{N}` (rule 3).
/// - `a` is INSIDE → `a0` → `{H}` (rule 2 hex precedence).
/// - `c0` etc. would also be `{H}` for the same reason.
#[test]
fn pattern_key_single_letter_alpha_prefix_normalizes() {
    assert_eq!(pattern_key("gadget-v2"), "gadget-v{N}");
    assert_eq!(pattern_key("thingo-r2"), "thingo-r{N}");
    assert_eq!(pattern_key("t1"), "t{N}");
    // `a0` falls under rule 2 because both chars are in
    // `[0-9a-f]` and one is a digit.
    assert_eq!(pattern_key("a0"), "{H}");
    // `t-1` splits into [Token("t"), Separator("-"), Token("1")];
    // `t` is pure alpha (no digits, no rule fires), `1` is pure
    // digit (rule 1 → `{N}`). Joined: `t-{N}`.
    assert_eq!(pattern_key("t-1"), "t-{N}");
    // `ab_5` splits into [Token("ab"), Separator("_"), Token("5")];
    // `ab` is hex-eligible chars but no digit → rule 2 fails;
    // alpha-only → rule 3 fails (no digits); literal `ab`.
    // `5` → `{N}`. Joined: `ab_{N}`.
    assert_eq!(pattern_key("ab_5"), "ab_{N}");
}

/// kworker thread names produce the same skeleton across CPUs
/// under the token-based normalizer. Bound bare:
/// `kworker/0:0` → `kworker/{N}:{N}`. Unbound:
/// `kworker/u8:3` → `kworker/u{N}:{N}` (alpha prefix `u`
/// length 1 normalizes per rule 3). Workqueue-bearing:
/// `kworker/0:0-wq_reclaim` → `kworker/{N}:{N}-wq_reclaim`
/// (workqueue suffix tokens are pure alpha → literal).
/// High-priority worker `1H` matches rule 4
/// (`^\d+[A-Za-z]+$`) and normalizes to `{N}H`.
#[test]
fn pattern_key_kworker_shapes_under_token_normalizer() {
    // Bare bound and unbound.
    assert_eq!(pattern_key("kworker/0:0"), "kworker/{N}:{N}");
    assert_eq!(pattern_key("kworker/3:2"), "kworker/{N}:{N}");
    assert_eq!(pattern_key("kworker/u8:3"), "kworker/u{N}:{N}");
    assert_eq!(pattern_key("kworker/u8:7"), "kworker/u{N}:{N}");
    assert_eq!(pattern_key("kworker/u16:0"), "kworker/u{N}:{N}");
    // Workqueue-bearing.
    assert_eq!(
        pattern_key("kworker/0:0-wq_reclaim"),
        "kworker/{N}:{N}-wq_reclaim",
    );
    assert_eq!(
        pattern_key("kworker/47:2-wq_reclaim"),
        "kworker/{N}:{N}-wq_reclaim",
    );
    // High-priority bound worker — `1H` token matches rule 4.
    assert_eq!(pattern_key("kworker/0:1H"), "kworker/{N}:{N}H");
    // High-priority bound worker with workqueue suffix.
    assert_eq!(
        pattern_key("kworker/0:1H-wq_prio"),
        "kworker/{N}:{N}H-wq_prio",
    );
}

/// Rule 4 (digits + alpha suffix → `{N}suffix`) catches the
/// `<id>H` shape kworker high-priority pools emit. Rule 4 sits
/// AFTER rule 2 (hex), so hex-eligible tokens still take the
/// `{H}` path; only tokens whose alpha portion includes a char
/// outside `[0-9a-f]` (uppercase letters, `g..z`) reach rule 4.
#[test]
fn classify_token_digits_alpha_suffix_rule_4() {
    // Pure-digit then non-hex alpha: rule 4 fires.
    assert_eq!(classify_token("1H"), "{N}H");
    assert_eq!(classify_token("0H"), "{N}H");
    // `Hz` contains uppercase (outside `[0-9a-f]`) → rule 2
    // fails → rule 4 fires.
    assert_eq!(classify_token("100Hz"), "{N}Hz");
    // `z` is outside `[0-9a-f]` → rule 2 fails → rule 4.
    assert_eq!(classify_token("3z"), "{N}z");
    // Rule 2 (hex) precedence: tokens whose chars are all in
    // `[0-9a-f]` (with at least one digit, len ≥ 2) classify
    // as hex BEFORE rule 4 runs.
    assert_eq!(classify_token("1a"), "{H}");
    assert_eq!(classify_token("0f"), "{H}");
    // `42abc` has chars `4,2,a,b,c` all in `[0-9a-f]` and
    // contains digits → rule 2 fires (`{H}`), NOT rule 4.
    assert_eq!(classify_token("42abc"), "{H}");
    // Mixed hex-then-non-hex alpha (e.g. `1aZ`): rule 2 fails
    // because `Z` is outside `[0-9a-f]`. Rule 3 fails (alpha
    // prefix length 0). Rule 4 fires (digits=`1`, alpha=`aZ`).
    assert_eq!(classify_token("1aZ"), "{N}aZ");
    // `42xyz` mixes hex digit with non-hex alpha → rule 2
    // fails on `x` → rule 4 fires.
    assert_eq!(classify_token("42xyz"), "{N}xyz");
    // Pure digits: rule 1 fires before rule 4 ever runs.
    assert_eq!(classify_token("42"), "{N}");
}

/// Empty comm input returns empty (no panic).
#[test]
fn pattern_key_empty_input_returns_empty() {
    assert_eq!(pattern_key(""), "");
}

/// Happy path: 8 `worker-N` + 4 `rayon-pool-N` + 1 `main`
/// produce 2 pattern buckets + 1 ungrouped (singleton). The
/// pattern bucket join keys are the `prefix-{N}` placeholder
/// form; the singleton reverts to its literal comm.
#[test]
fn build_groups_comm_produces_pattern_buckets_and_singleton() {
    let mut threads = Vec::new();
    for i in 0..8 {
        threads.push(make_thread("app", &format!("worker-{i}")));
    }
    for i in 0..4 {
        threads.push(make_thread("app", &format!("rayon-pool-{i}")));
    }
    threads.push(make_thread("app", "main"));

    let snap = snap_with(threads);
    let groups = build_groups(&snap, GroupBy::Comm, &[], None, None, false);

    // Two pattern buckets keyed by the stripped prefix and one
    // singleton keyed by the literal name.
    assert!(
        groups.contains_key("worker-{N}"),
        "worker-{{N}} pattern bucket",
    );
    assert_eq!(groups["worker-{N}"].thread_count, 8);
    assert!(
        groups.contains_key("rayon-pool-{N}"),
        "rayon-pool-{{N}} pattern bucket",
    );
    assert_eq!(groups["rayon-pool-{N}"].thread_count, 4);
    assert!(
        groups.contains_key("main"),
        "singleton main reverts to literal comm",
    );
    assert_eq!(groups["main"].thread_count, 1);
    assert_eq!(groups.len(), 3);
}

/// `pattern_display_label` produces a grex regex over the
/// member set for buckets ≥ 2; singletons fall through to the
/// join key. Validates the render-side wiring without
/// asserting a specific regex shape (grex internals may vary).
#[test]
fn pattern_display_label_grex_for_multi_member_else_join_key() {
    let single = vec!["worker-0".to_string()];
    assert_eq!(pattern_display_label("worker-0", &single), "worker-0");
    let empty: Vec<String> = vec![];
    assert_eq!(pattern_display_label("worker", &empty), "worker");
    let multi = vec!["worker-0".to_string(), "worker-1".to_string()];
    let label = pattern_display_label("worker", &multi);
    assert!(
        label.contains("worker"),
        "grex label must mention the shared prefix; got {label:?}",
    );
}

/// End-to-end pin: `compare(GroupBy::Comm, ...)` produces
/// DiffRow whose `group_key` is the `prefix-{N}` placeholder
/// (deterministic across snapshots) and whose `display_key`
/// is fed by [`pattern_display_label`] over the union of
/// baseline + candidate members. The display label may equal
/// the join key when grex's regex is longer than the key
/// (per the high-cardinality fallback in
/// [`pattern_display_label`]); in either case it must
/// contain the shared `worker` prefix.
#[test]
fn compare_comm_pattern_emits_prefix_join_key_and_grex_display() {
    let baseline = snap_with(vec![
        make_thread("app", "worker-0"),
        make_thread("app", "worker-1"),
    ]);
    let candidate = snap_with(vec![
        make_thread("app", "worker-2"),
        make_thread("app", "worker-3"),
    ]);
    let diff = compare(
        &baseline,
        &candidate,
        &CompareOptions {
            group_by: GroupBy::Comm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker-{N}")
        .expect("worker-{N} row");
    assert_eq!(
        row.group_key, "worker-{N}",
        "join key is the placeholder pattern"
    );
    assert!(
        row.display_key.contains("worker"),
        "display key reflects grex (or fallback to join key) over union; got {:?}",
        row.display_key,
    );
    // The display label is whatever `pattern_display_label`
    // produces for the union of distinct members — either
    // the grex regex when it fits within the join key, or
    // the join key itself under the high-cardinality
    // fallback. Pin the contract by computing the expected
    // label directly and comparing.
    let mut union_members: Vec<String> = vec![
        "worker-0".into(),
        "worker-1".into(),
        "worker-2".into(),
        "worker-3".into(),
    ];
    union_members.sort();
    union_members.dedup();
    let expected_label = pattern_display_label("worker-{N}", &union_members);
    assert_eq!(
        row.display_key, expected_label,
        "display label must match pattern_display_label over union"
    );
}

/// Regression pin for the cross-snapshot frequency union: a
/// pattern that has 1 thread in baseline and 3 threads in
/// candidate must still cluster under the same `worker-{N}`
/// key on BOTH sides. Under per-snapshot counts the baseline
/// would gate `worker-7` to literal (count 1 < 2), the
/// candidate would gate `worker-{N}` to pattern (count 3 ≥ 2),
/// and the row would surface as both only-in-baseline AND
/// only-in-candidate — orphaned. The union frequency
/// (1 + 3 = 4 ≥ 2) promotes the pattern on both sides so the
/// row joins.
#[test]
fn compare_comm_pattern_joins_across_asymmetric_resize() {
    let baseline = snap_with(vec![make_thread("app", "worker-7")]);
    let candidate = snap_with(vec![
        make_thread("app", "worker-0"),
        make_thread("app", "worker-1"),
        make_thread("app", "worker-2"),
    ]);
    let diff = compare(
        &baseline,
        &candidate,
        &CompareOptions {
            group_by: GroupBy::Comm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker-{N}")
        .expect("worker-{N} row joined across asymmetric snapshots");
    assert_eq!(row.thread_count_a, 1, "baseline carries 1 worker");
    assert_eq!(row.thread_count_b, 3, "candidate carries 3 workers");
    // No orphan rows for the worker family. The union map
    // ensures both sides use the same `worker-{N}` key.
    let baseline_orphans: Vec<&String> = diff
        .only_baseline
        .iter()
        .filter(|k| k.starts_with("worker"))
        .collect();
    assert!(
        baseline_orphans.is_empty(),
        "no worker-prefixed orphans in only_baseline; got {baseline_orphans:?}",
    );
    let candidate_orphans: Vec<&String> = diff
        .only_candidate
        .iter()
        .filter(|k| k.starts_with("worker"))
        .collect();
    assert!(
        candidate_orphans.is_empty(),
        "no worker-prefixed orphans in only_candidate; got {candidate_orphans:?}",
    );
}

/// Token classifier: pure digits → `{N}` (rule 1).
#[test]
fn classify_token_pure_digits() {
    assert_eq!(classify_token("0"), "{N}");
    assert_eq!(classify_token("42"), "{N}");
    assert_eq!(classify_token("999"), "{N}");
}

/// Token classifier: hex-like (all `[0-9a-f]`, length ≥ 2,
/// at least one digit) → `{H}` (rule 2). `abc` (no digits)
/// is NOT hex-like; `a1` is. Pure-digit tokens fall through
/// to rule 1 first.
#[test]
fn classify_token_hex_like() {
    assert_eq!(classify_token("a1234"), "{H}");
    assert_eq!(classify_token("abc123def456"), "{H}");
    assert_eq!(classify_token("7890ab"), "{H}");
    assert_eq!(classify_token("1a2"), "{H}");
    assert_eq!(classify_token("650ab12cd34ef"), "{H}");
    // Pure alpha (no digits) — rule 2 fails the digit check.
    assert_eq!(classify_token("abc"), "abc");
    // Length 1 — rule 2 fails the length check (digit-only
    // would be rule 1, but `a` has no digit).
    assert_eq!(classify_token("a"), "a");
    // Hex-like length 2 with one digit and one alpha:
    assert_eq!(classify_token("a1"), "{H}");
    // Token containing chars outside `[0-9a-f]` (like `g`,
    // `u`, `H`) is NOT hex-like and falls through to rule 3.
    assert_eq!(classify_token("u8"), "u{N}");
}

/// Token classifier: alpha prefix + trailing digits
/// (`^[A-Za-z]+\d+$`, alpha prefix length ≥ 1) →
/// `prefix{N}` (rule 3). Single-letter alpha prefixes (e.g.
/// `u8`, `v2`, `r2`) qualify.
#[test]
fn classify_token_alpha_prefix_digits() {
    assert_eq!(classify_token("worker7"), "worker{N}");
    assert_eq!(classify_token("CamelCaseWord175"), "CamelCaseWord{N}");
    assert_eq!(classify_token("u8"), "u{N}");
    assert_eq!(classify_token("u16"), "u{N}");
    assert_eq!(classify_token("v2"), "v{N}");
    assert_eq!(classify_token("r2"), "r{N}");
    // Digits-then-alpha matches rule 4 (`^\d+[A-Za-z]+$`) →
    // `{N}suffix`. Rule 2 (hex) takes precedence when chars
    // qualify (e.g. `3a` → `{H}` because both chars are in
    // `[0-9a-f]`); `H` and `z` are outside that range, so
    // rule 4 fires.
    assert_eq!(classify_token("1H"), "{N}H");
    assert_eq!(classify_token("3z"), "{N}z");
    // Alpha-then-digits-then-alpha does NOT match rule 3
    // (the regex requires the digit run to be at the end,
    // anchored by `$`).
    assert_eq!(classify_token("proto303handler"), "proto303handler");
}

/// Token classifier: token with no rule match stays literal.
#[test]
fn classify_token_literal_fallback() {
    assert_eq!(classify_token("BPF"), "BPF");
    assert_eq!(classify_token("CUBIC"), "CUBIC");
    assert_eq!(classify_token("AUTO"), "AUTO");
    assert_eq!(classify_token("FLOWLABEL"), "FLOWLABEL");
    assert_eq!(classify_token("hamster"), "hamster");
    assert_eq!(classify_token("zilch"), "zilch");
}

/// Empty token returns empty (no panic).
#[test]
fn classify_token_empty_returns_empty() {
    assert_eq!(classify_token(""), "");
}

/// Tokenizer: `split_into_segments` alternates token / sep
/// runs, preserving the original separator characters
/// verbatim. Empty input yields zero segments.
#[test]
fn split_into_segments_alternates_token_and_separator_runs() {
    assert!(split_into_segments("").is_empty());
    // Pure alpha → one token.
    let segs = split_into_segments("hamster");
    assert_eq!(segs, vec![Segment::Token("hamster")]);
    // Token-sep-token.
    let segs = split_into_segments("worker-7");
    assert_eq!(
        segs,
        vec![
            Segment::Token("worker"),
            Segment::Separator("-"),
            Segment::Token("7"),
        ],
    );
    // Multi-char separator run preserved as one segment.
    let segs = split_into_segments("a..b");
    assert_eq!(
        segs,
        vec![
            Segment::Token("a"),
            Segment::Separator(".."),
            Segment::Token("b"),
        ],
    );
    // Leading separator run.
    let segs = split_into_segments("/abc");
    assert_eq!(segs, vec![Segment::Separator("/"), Segment::Token("abc")],);
    // Mixed separator chars in one run.
    let segs = split_into_segments("yy._650");
    assert_eq!(
        segs,
        vec![
            Segment::Token("yy"),
            Segment::Separator("._"),
            Segment::Token("650"),
        ],
    );
    // `+` is a separator (per spec): kworker active-worker
    // decoration tokenizes the same way as the idle (`-`)
    // form. Tokens on either side normalize independently.
    let segs = split_into_segments("kworker/0:1+events");
    assert_eq!(
        segs,
        vec![
            Segment::Token("kworker"),
            Segment::Separator("/"),
            Segment::Token("0"),
            Segment::Separator(":"),
            Segment::Token("1"),
            Segment::Separator("+"),
            Segment::Token("events"),
        ],
    );
}

/// `+` is a separator (per spec) so active-kworker comms
/// (`<cpu>:<id>+<wq>`) tokenize the same shape as idle
/// (`<cpu>:<id>-<wq>`) and the digit tokens on each side
/// normalize independently. Active workers across distinct
/// CPUs collapse to one bucket per workqueue. Active and
/// idle workers DO NOT collapse — the separator character
/// (`+` vs `-`) is preserved verbatim in the rejoined
/// skeleton, so they sort into separate buckets per
/// workqueue per decoration.
#[test]
fn pattern_key_kworker_active_decoration_separator() {
    // Active-decoration per-CPU collapse.
    assert_eq!(pattern_key("kworker/0:1+events"), "kworker/{N}:{N}+events",);
    assert_eq!(pattern_key("kworker/1:0+events"), "kworker/{N}:{N}+events",);
    // Active and idle remain distinct buckets.
    assert_ne!(
        pattern_key("kworker/0:1+events"),
        pattern_key("kworker/0:1-events"),
    );
    assert_eq!(pattern_key("kworker/0:1-events"), "kworker/{N}:{N}-events",);
}

/// Layer 1: systemd template normalization. Instances without
/// `[._-]` become `{I}`; instances with any of those stay
/// literal.
#[test]
fn apply_systemd_template_opaque_id_to_placeholder() {
    // Opaque numeric instance — normalize.
    assert_eq!(
        apply_systemd_template("/user.slice/user-0.slice/user@0.service/boot.scope"),
        "/user.slice/user-0.slice/user@{I}.service/boot.scope",
    );
    assert_eq!(
        apply_systemd_template("/user.slice/user-1001.slice/user@1001.service/boot.scope"),
        "/user.slice/user-1001.slice/user@{I}.service/boot.scope",
    );
    // Structured instance with `.` — keep literal.
    assert_eq!(
        apply_systemd_template("/critical.slice/launcher@foo.bar.baz.service"),
        "/critical.slice/launcher@foo.bar.baz.service",
    );
    // No `@<x>.service` — unchanged.
    assert_eq!(
        apply_systemd_template("/system.slice/crond.service"),
        "/system.slice/crond.service",
    );
    // Path with no `@` at all.
    assert_eq!(apply_systemd_template("/"), "/");
}

/// Layer 3 (tighten) keeps placeholders for instance-classified
/// tokens even when those tokens happen to be constant across
/// every member of a multi-member group. The classify_token
/// gate prevents reverting `{N}`, `{H}`, `prefix{N}`, and
/// `{N}suffix` positions — once Layer 2 identifies a token as
/// instance data, that classification is final regardless of
/// cross-member equality. Only positions whose literal already
/// matches its own classification (pure literals) appear in
/// the tightened key, and those are unchanged from the
/// skeleton.
#[test]
fn cgroup_tighten_keeps_instance_placeholders_when_constant() {
    // Two simplified paths that share most tokens but differ
    // at one digit position.
    let path_1 = "/apps.slice/run-17.fluxcap9000_01.zz3";
    let path_2 = "/apps.slice/run-22.fluxcap9000_01.zz3";
    // After Layer 2 (no Layer 1 substitution applies):
    // Expected skeleton tokens (digits / alpha-prefix-digits
    // placeholders):
    //   apps, slice, run, {N}, fluxcap{N}, {N}, zz{N}
    // After Layer 3 (tighten):
    //   The first `{N}` (17 vs 22) varies → stays `{N}`.
    //   `fluxcap{N}` (always 9000) — instance-classified token,
    //     stays `fluxcap{N}` per the classify_token gate.
    //   `{N}` (the `01`) — instance-classified, stays `{N}`.
    //   `zz{N}` (always 3) — instance-classified, stays `zz{N}`.
    //   `apps`, `slice`, `run` are pure literals: classify_token
    //     returns themselves, so they appear unchanged.
    let snap = snap_with(vec![
        {
            let mut t = make_thread("p", "ta");
            t.cgroup = path_1.into();
            t
        },
        {
            let mut t = make_thread("p", "tb");
            t.cgroup = path_2.into();
            t
        },
    ]);
    let diff = compare(
        &snap,
        &snap,
        &CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let group_keys: std::collections::BTreeSet<String> =
        diff.rows.iter().map(|r| r.group_key.clone()).collect();
    // Instance-classified tokens stay as placeholders; pure
    // literals (`apps`, `slice`, `run`) appear unchanged.
    let expected = "/apps.slice/run-{N}.fluxcap{N}_{N}.zz{N}";
    assert!(
        group_keys.contains(expected),
        "tightened key {expected:?} missing; got {group_keys:?}",
    );
}

/// Brackets in cgroup paths split tokens just like every other
/// separator. Two paths with bracketed hex IDs (session
/// scopes, container instance IDs) collapse onto the same
/// skeleton — `[a1b2c3d4]` and `[deadbeef]` both tokenize to
/// `[{H}]`. The Layer-2 token normalizer treats brackets as
/// run boundaries (per [`is_token_separator`]), so the hex
/// payloads inside them flow through rule 2 the same way
/// dotted hex would.
///
/// Pin the cgroup-path bracket case end-to-end through
/// `compare`: two distinct sessions with hex-ID brackets must
/// land in one normalized bucket under
/// [`GroupBy::Cgroup`].
#[test]
fn cgroup_normalize_collapses_bracketed_hex_session_ids() {
    let mut ta = make_thread("p", "ta");
    ta.cgroup = "/user.slice/session-[a1b2c3d4]/scope".into();
    let mut tb = make_thread("p", "tb");
    tb.cgroup = "/user.slice/session-[dead1234]/scope".into();
    let snap_a = snap_with(vec![ta]);
    let snap_b = snap_with(vec![tb]);

    // Sanity-check the lower-level pieces this test composes:
    // (a) `cgroup_normalize_skeleton` produces the expected
    // `[{H}]` skeleton for both paths, and
    // (b) `build_cgroup_key_map` resolves both literal paths
    // to the tightened skeleton key. If either of these
    // returns something different, the resulting bucket key
    // won't match the test's expected string and the
    // outer compare-driven assertion would fail with an
    // unhelpful "got {}" message.
    let (skel_a, post_a, _) = cgroup_normalize_skeleton("/user.slice/session-[a1b2c3d4]/scope");
    let (skel_b, post_b, _) = cgroup_normalize_skeleton("/user.slice/session-[dead1234]/scope");
    assert_eq!(
        skel_a, "/user.slice/session-[{H}]/scope",
        "Layer-2 skeleton for path1 mismatch; got {skel_a:?}",
    );
    assert_eq!(
        skel_b, "/user.slice/session-[{H}]/scope",
        "Layer-2 skeleton for path2 mismatch; got {skel_b:?}",
    );
    // Layer 1 is a no-op for these paths (no @<x>.service).
    assert_eq!(post_a, "/user.slice/session-[a1b2c3d4]/scope");
    assert_eq!(post_b, "/user.slice/session-[dead1234]/scope");
    let key_map = build_cgroup_key_map(&snap_a, &snap_b, &[]);
    assert_eq!(
        key_map.get("/user.slice/session-[a1b2c3d4]/scope"),
        Some(&"/user.slice/session-[{H}]/scope".to_string()),
        "key_map must resolve path1 to the tightened skeleton",
    );
    assert_eq!(
        key_map.get("/user.slice/session-[dead1234]/scope"),
        Some(&"/user.slice/session-[{H}]/scope".to_string()),
        "key_map must resolve path2 to the tightened skeleton",
    );

    let diff = compare(
        &snap_a,
        &snap_b,
        &CompareOptions {
            group_by: GroupBy::Cgroup.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    // Both bracketed-hex IDs collapse to `[{H}]`, so the two
    // paths share one normalized cgroup key after Layer 2.
    let group_keys: std::collections::BTreeSet<String> =
        diff.rows.iter().map(|r| r.group_key.clone()).collect();
    let expected = "/user.slice/session-[{H}]/scope";
    assert!(
        group_keys.contains(expected),
        "missing bracketed-hex cgroup bucket {expected:?}; got {group_keys:?}; \
         diff.only_baseline={:?}; diff.only_candidate={:?}",
        diff.only_baseline,
        diff.only_candidate,
    );
    // No only-side orphans — the union frequency promoted
    // both paths under the same key.
    assert!(
        diff.only_baseline.is_empty(),
        "no orphans under bracketed-hex collapse, got {:?}",
        diff.only_baseline,
    );
    assert!(
        diff.only_candidate.is_empty(),
        "no orphans under bracketed-hex collapse, got {:?}",
        diff.only_candidate,
    );
}

/// Brackets split tokens identically to the existing separator
/// class. `worker[42]` tokenizes to
/// `[Token("worker"), Sep("["), Token("42"), Sep("]")]` so a
/// rejoin under `pattern_key` produces `worker[{N}]`. A
/// regression that removed `[` / `]` from the separator class
/// would surface here as `worker[42]` returning literal because
/// the bracketed-digit token would no longer reach rule 1.
#[test]
fn pattern_key_normalizes_bracketed_digits() {
    assert_eq!(pattern_key("worker[42]"), "worker[{N}]");
    assert_eq!(
        pattern_key("systemd-network[105904]"),
        "systemd-network[{N}]"
    );
    // Both pcomm halves and the tgid normalize when each side
    // is hex/digit-eligible. `bash[4242]` — bash is pure alpha,
    // 4242 is pure digits → `bash[{N}]`.
    assert_eq!(pattern_key("bash[4242]"), "bash[{N}]");
    // Hex-only inside the brackets still picks `{H}` per the
    // hex-rule precedence over rule 4.
    assert_eq!(pattern_key("dev[1ab]"), "dev[{H}]");
}

/// `[` and `]` join the existing separator class — `split_into_segments`
/// emits separator runs that include them verbatim. Pin both
/// the standalone bracket and a multi-char run mixing brackets
/// with other separators.
#[test]
fn split_into_segments_treats_brackets_as_separators() {
    let segs = split_into_segments("worker[42]");
    assert_eq!(
        segs,
        vec![
            Segment::Token("worker"),
            Segment::Separator("["),
            Segment::Token("42"),
            Segment::Separator("]"),
        ],
    );
    // Bracket adjacent to existing separator chars merges into
    // a single separator run.
    let segs = split_into_segments("a-[1]");
    assert_eq!(
        segs,
        vec![
            Segment::Token("a"),
            Segment::Separator("-["),
            Segment::Token("1"),
            Segment::Separator("]"),
        ],
    );
}

/// `is_token_separator` returns true for `[` and `]` directly.
/// Pin the boolean predicate so a regression that drops a
/// bracket from the `matches!` arm surfaces as a unit-test
/// failure rather than only at the end-to-end pattern-rejoin
/// site.
#[test]
fn is_token_separator_includes_brackets() {
    assert!(is_token_separator('['));
    assert!(is_token_separator(']'));
}

/// Cross-snapshot frequency union promotes a `worker-{N}`
/// pattern when baseline has 1 process + candidate has 2 —
/// the union total (3) crosses the >= 2 promotion gate so
/// both sides use the same `worker-{N}` join key. Mirrors
/// [`compare_comm_pattern_joins_across_asymmetric_resize`]
/// for the Pcomm axis. Without the union, baseline's
/// `worker-7` would gate to literal (count 1) while
/// candidate's two would gate to pattern, producing orphaned
/// only-in-baseline / only-in-candidate rows.
#[test]
fn compare_pcomm_pattern_joins_across_asymmetric_resize() {
    let baseline = snap_with(vec![make_thread("worker-7", "t0")]);
    let candidate = snap_with(vec![
        make_thread("worker-0", "t0"),
        make_thread("worker-1", "t1"),
    ]);
    let diff = compare(
        &baseline,
        &candidate,
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker-{N}")
        .expect("worker-{N} pcomm row joined across asymmetric snapshots");
    assert_eq!(row.thread_count_a, 1, "baseline carries 1 worker process");
    assert_eq!(
        row.thread_count_b, 2,
        "candidate carries 2 worker processes"
    );
    // No orphan rows for the worker family.
    let baseline_orphans: Vec<&String> = diff
        .only_baseline
        .iter()
        .filter(|k| k.starts_with("worker"))
        .collect();
    assert!(
        baseline_orphans.is_empty(),
        "no worker-prefixed pcomm orphans in only_baseline; got {baseline_orphans:?}",
    );
    let candidate_orphans: Vec<&String> = diff
        .only_candidate
        .iter()
        .filter(|k| k.starts_with("worker"))
        .collect();
    assert!(
        candidate_orphans.is_empty(),
        "no worker-prefixed pcomm orphans in only_candidate; got {candidate_orphans:?}",
    );
}

/// End-to-end: `compare(GroupBy::Pcomm, ...)` produces a
/// `DiffRow` whose `group_key` is the `prefix-{N}` skeleton
/// (deterministic across snapshots) and whose `display_key`
/// is fed by [`pattern_display_label`] over the union of
/// baseline + candidate pcomm members. Mirrors
/// [`compare_comm_pattern_emits_prefix_join_key_and_grex_display`]
/// for the Pcomm axis.
#[test]
fn compare_pcomm_pattern_emits_prefix_join_key_and_grex_display() {
    let baseline = snap_with(vec![
        make_thread("worker-0", "t0"),
        make_thread("worker-1", "t1"),
    ]);
    let candidate = snap_with(vec![
        make_thread("worker-2", "t0"),
        make_thread("worker-3", "t1"),
    ]);
    let diff = compare(
        &baseline,
        &candidate,
        &CompareOptions {
            group_by: GroupBy::Pcomm.into(),
            cgroup_flatten: vec![],
            no_thread_normalize: false,
            no_cg_normalize: false,
            sort_by: Vec::new(),
        },
    );
    let row = diff
        .rows
        .iter()
        .find(|r| r.metric_name == "run_time_ns" && r.group_key == "worker-{N}")
        .expect("worker-{N} pcomm row");
    assert_eq!(
        row.group_key, "worker-{N}",
        "join key is the placeholder pattern under Pcomm normalization",
    );
    assert!(
        row.display_key.contains("worker"),
        "display key reflects grex (or fallback to join key) over union; got {:?}",
        row.display_key,
    );
    // The display label is whatever `pattern_display_label`
    // produces for the union of distinct members — either
    // the grex regex when it fits within the join key, or
    // the join key itself under the high-cardinality
    // fallback. Pin the contract by computing the expected
    // label directly and comparing.
    let mut union_members: Vec<String> = vec![
        "worker-0".into(),
        "worker-1".into(),
        "worker-2".into(),
        "worker-3".into(),
    ];
    union_members.sort();
    union_members.dedup();
    let expected_label = pattern_display_label("worker-{N}", &union_members);
    assert_eq!(
        row.display_key, expected_label,
        "display label must match pattern_display_label over union"
    );
}

/// Literal bracket name (no digits inside): `pattern_key`
/// returns the input unchanged. `[bar]` tokenizes to
/// `[Sep("["), Token("bar"), Sep("]")]`; `bar` is pure alpha
/// (no rule fires). The whole input thus echoes through
/// — the bracket separators are preserved verbatim.
#[test]
fn pattern_key_bracket_alpha_token_stays_literal() {
    assert_eq!(pattern_key("foo[bar]"), "foo[bar]");
    assert_eq!(pattern_key("a[b]"), "a[b]");
    // Hex-eligible alpha-only inside brackets still stays
    // literal — rule 2 requires at least one digit.
    assert_eq!(pattern_key("dev[abc]"), "dev[abc]");
}

/// `pattern_display_label` over members whose names contain
/// brackets must not panic in `grex` and must produce a
/// regex that contains the bracketed substrings. Brackets
/// are regex metacharacters, so `grex` has to escape them
/// for the resulting regex to be valid. Pin that the labels
/// for `worker[0]` and `worker[1]` come back containing
/// `worker` — the literal-prefix portion — so a regression
/// that drops the bracket escaping (or that crashes on
/// bracket input) surfaces here.
#[test]
fn pattern_display_label_handles_bracket_member_names() {
    let members = vec![
        "worker[0]".to_string(),
        "worker[1]".to_string(),
        "worker[2]".to_string(),
    ];
    let label = pattern_display_label("worker[{N}]", &members);
    assert!(
        label.contains("worker"),
        "grex must produce a label that contains the shared `worker` prefix; got {label:?}",
    );
    // The regex `grex` produces is well-formed — try
    // compiling it. A bracket-escaping regression would
    // produce an invalid regex syntax that `Regex::new`
    // rejects.
    let _: Regex = Regex::new(&label)
        .unwrap_or_else(|e| panic!("grex output {label:?} is not a valid regex: {e}"));
}

/// Cgroup paths with bracketed segments tokenize through
/// the same separator class as comms. A path like
/// `/runner-[xyz]/scope` splits brackets into separator
/// runs around the inner token. Pin that
/// [`cgroup_skeleton_tokens`] handles the brackets without
/// panic and that the resulting skeleton preserves the
/// non-bracket tokens. Sister test to
/// [`cgroup_normalize_collapses_bracketed_hex_session_ids`]
/// but at the lower-level `cgroup_skeleton_tokens` boundary.
#[test]
fn cgroup_skeleton_tokens_handles_bracketed_segments() {
    let (skeleton, tokens) = cgroup_skeleton_tokens("/runner-[xyz]/scope");
    // Tokens come from the non-separator runs only:
    // `runner`, `xyz`, `scope`. Brackets and `/` and `-` are
    // all separators and don't show up in the token list.
    assert_eq!(
        tokens,
        vec!["runner".to_string(), "xyz".to_string(), "scope".to_string(),],
        "bracket separators must split tokens cleanly; got {tokens:?}",
    );
    // Skeleton: `runner` and `xyz` and `scope` are all pure
    // alpha → literal. Separators preserved verbatim.
    assert_eq!(
        skeleton, "/runner-[xyz]/scope",
        "skeleton must preserve separators including brackets; got {skeleton:?}",
    );
}
