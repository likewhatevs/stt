//! Tests for the failure-dump types and helpers.
//!
//! Split out of mod.rs so the data-shape and dispatch logic in mod.rs stay
//! focused on production code; assertions live here.

use super::super::bpf_map::{
    BPF_MAP_TYPE_BLOOM_FILTER, BPF_MAP_TYPE_CGROUP_STORAGE, BPF_MAP_TYPE_HASH,
    BPF_MAP_TYPE_INSN_ARRAY, BPF_MAP_TYPE_LPM_TRIE, BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE,
    BPF_MAP_TYPE_PERCPU_HASH, BPF_MAP_TYPE_QUEUE, BPF_MAP_TYPE_STACK,
};
// `MemReader` brought into scope so the stub readers below can
// call `is_arena_addr` / `read_arena` via dot notation. The FQP
// `super::super::btf_render::MemReader` resolves the trait for
// the `impl ... for StubReader` blocks, but Rust requires the
// trait to be in scope at the method-call site — the test
// bodies invoke `r.is_arena_addr(...)` and `r.read_arena(...)`,
// which fail with E0599 ("no method named `is_arena_addr` found")
// without this `use`.
use super::*;
use crate::monitor::btf_render::MemReader;
// `name_from_str` is the shared helper that packs a `&str` into the
// `(name_bytes, name_len)` representation used by
// [`super::super::bpf_map::BpfMapInfo`]. Single source of truth in
// [`crate::monitor::test_util`] — replaces the prior local
// `map_name_bytes` copy that duplicated the logic verbatim.
use crate::monitor::test_util::name_from_str;

#[test]
fn hex_dump_basic() {
    assert_eq!(hex_dump(&[]), "");
    assert_eq!(hex_dump(&[0]), "00");
    assert_eq!(hex_dump(&[0x12, 0x34, 0xab]), "12 34 ab");
}

/// Empty input renders as empty string. Single-element input
/// renders as one mid-tier glyph (constant non-zero series
/// reads as "no variation"). All-zero series renders as the
/// lowest glyph repeated.
#[test]
fn render_sparkline_edge_cases() {
    assert_eq!(render_sparkline(&[]), "");
    // Single non-zero element: constant series → mid-tier glyph.
    assert_eq!(render_sparkline(&[42]), "▅");
    // All-zero series: lowest glyph for every entry.
    assert_eq!(render_sparkline(&[0, 0, 0]), "▁▁▁");
    // All-equal non-zero series: mid-tier glyph for every entry.
    assert_eq!(render_sparkline(&[5, 5, 5]), "▅▅▅");
}

/// Strictly-increasing series scales linearly across the glyph
/// set: first sample at min lands at lowest glyph, last sample
/// at max lands at highest. Pin both ends so a future scaling
/// regression that broke either bound is caught.
#[test]
fn render_sparkline_monotonic_scales_to_full_range() {
    let s = render_sparkline(&[0, 1, 2, 3, 4, 5, 6, 7]);
    let chars: Vec<char> = s.chars().collect();
    assert_eq!(chars.len(), 8);
    assert_eq!(chars[0], '▁', "min must map to lowest glyph: {s}");
    assert_eq!(chars[7], '█', "max must map to highest glyph: {s}");
}

/// i64 wrapper saturates negative values to 0, then routes
/// through u64 sparkline. Verifies a counter that briefly
/// dips negative (corrupt read) doesn't crash and produces
/// a sane sparkline.
#[test]
fn render_sparkline_i64_clamps_negatives() {
    let s = render_sparkline_i64(&[-5, 0, 5, 10]);
    // After clamp: [0, 0, 5, 10] → first two at lowest, last
    // two scale up. Just pin length and bounds; exact glyphs
    // depend on integer rounding.
    assert_eq!(s.chars().count(), 4);
}

/// Full SCX_EV_* counter timeline construction: build a
/// MonitorSample with two CPUs reporting event counters,
/// fold to EventCounterSample, verify cross-CPU sums and
/// elapsed_ms propagation.
#[test]
fn event_counter_sample_sums_across_cpus() {
    use super::super::{CpuSnapshot, MonitorSample, ScxEventCounters};
    let cpu_a = CpuSnapshot {
        event_counters: Some(ScxEventCounters {
            select_cpu_fallback: 5,
            bypass_dispatch: 100,
            ..Default::default()
        }),
        ..Default::default()
    };
    let cpu_b = CpuSnapshot {
        event_counters: Some(ScxEventCounters {
            select_cpu_fallback: 7,
            bypass_dispatch: 50,
            ..Default::default()
        }),
        ..Default::default()
    };
    let sample = MonitorSample {
        elapsed_ms: 100,
        cpus: vec![cpu_a, cpu_b],
        prog_stats: None,
    };
    let folded = EventCounterSample::from_monitor_sample(&sample)
        .expect("at least one CPU has event_counters");
    assert_eq!(folded.elapsed_ms, 100);
    assert_eq!(folded.select_cpu_fallback, 12);
    assert_eq!(folded.bypass_dispatch, 150);
}

/// MonitorSample with no CPU reporting event_counters folds
/// to None — propagating an all-zero row would mislead the
/// downstream consumer (a real "every counter at 0" tick
/// looks identical to "every CPU's offsets unresolved").
#[test]
fn event_counter_sample_returns_none_when_no_cpu_has_counters() {
    use super::super::{CpuSnapshot, MonitorSample};
    let cpu = CpuSnapshot {
        event_counters: None,
        ..Default::default()
    };
    let sample = MonitorSample {
        elapsed_ms: 200,
        cpus: vec![cpu],
        prog_stats: None,
    };
    assert!(EventCounterSample::from_monitor_sample(&sample).is_none());
}

/// EventCounterSample serde round-trips cleanly: every field
/// is `i64` (kernel-side `s64`), so a wire-format encode →
/// decode preserves bit patterns including the i64::MAX edge.
#[test]
fn event_counter_sample_serde_roundtrip() {
    let s = EventCounterSample {
        elapsed_ms: 123_456,
        select_cpu_fallback: i64::MAX,
        insert_not_owned: -1, // kernel never produces this
        // but the wire format must
        // preserve whatever the read
        // captured rather than silently
        // clamp.
        ..Default::default()
    };
    let json = serde_json::to_string(&s).unwrap();
    let loaded: EventCounterSample = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.elapsed_ms, 123_456);
    assert_eq!(loaded.select_cpu_fallback, i64::MAX);
    assert_eq!(loaded.insert_not_owned, -1);
}

#[test]
fn report_serde_roundtrip() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: vec![FailureDumpMap {
            name: "scx_demo.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            value_size: 8,
            max_entries: 1,
            value: Some(RenderedValue::Uint {
                bits: 32,
                value: 42,
            }),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        }],
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let json = serde_json::to_string(&report).unwrap();
    let parsed: FailureDumpReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.maps.len(), 1);
    assert_eq!(parsed.maps[0].name, "scx_demo.bss");
    assert_eq!(parsed.maps[0].max_entries, 1);
}

#[test]
fn empty_report_serde() {
    let report = FailureDumpReport::default();
    let json = serde_json::to_string(&report).unwrap();
    let parsed: FailureDumpReport = serde_json::from_str(&json).unwrap();
    assert!(parsed.maps.is_empty());
}

// ---- Display impl coverage --------------------------------------
//
// The Display impl is the human-readable form used in test
// failure output. Pin its layout against representative shapes.

fn make_simple_map() -> FailureDumpMap {
    FailureDumpMap {
        name: "scx_demo.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        value_size: 8,
        max_entries: 1,
        value: Some(RenderedValue::Struct {
            type_name: Some("task_ctx".into()),
            members: vec![super::super::btf_render::RenderedMember {
                name: "weight".into(),
                value: RenderedValue::Uint {
                    bits: 32,
                    value: 1024,
                },
            }],
        }),
        entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    }
}

#[test]
fn report_display_empty() {
    let report = FailureDumpReport::default();
    assert_eq!(format!("{report}"), "(empty failure dump)");
}

#[test]
fn report_display_one_map_with_value() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: vec![make_simple_map()],
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    // Map header line.
    assert!(
        out.starts_with("map scx_demo.bss (type="),
        "missing header: {out}"
    );
    // Struct rendering: the inline form is `TypeName{f=v}` — no
    // `struct` keyword, no space before brace, `=` separator.
    assert!(out.contains("task_ctx{"), "missing struct: {out}");
    assert!(out.contains("weight=1024"), "missing member: {out}");
    assert!(out.ends_with('}'), "missing closing brace: {out}");
}

#[test]
fn report_display_multiple_maps_separated() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: vec![make_simple_map(), make_simple_map()],
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    // Maps separated by a blank line (\n\n).
    let blank_line_count = out.matches("\n\n").count();
    assert_eq!(
        blank_line_count, 1,
        "expected one blank-line separator between two maps: {out}"
    );
}

#[test]
fn map_display_includes_error_marker() {
    let mut m = make_simple_map();
    m.value = None;
    m.error = Some("ARRAY value region unreadable".into());
    let out = format!("{m}");
    assert!(
        out.contains("[error: ARRAY value region unreadable]"),
        "missing error marker: {out}"
    );
}

#[test]
fn entry_display_renders_key_and_value() {
    // Scalar key + value: header is `entry: key=7\n  value: 99`.
    // The `=` is the key-assignment marker; `: ` introduces the
    // value field. The renderer doesn't add type breadcrumb for
    // bare scalars.
    let entry = FailureDumpEntry {
        key: Some(RenderedValue::Uint { bits: 32, value: 7 }),
        key_hex: "07 00 00 00".into(),
        value: Some(RenderedValue::Uint {
            bits: 32,
            value: 99,
        }),
        value_hex: "63 00 00 00".into(),
        payload: None,
    };
    let out = format!("{entry}");
    assert!(out.contains("key=7"), "missing key: {out}");
    assert!(out.contains("value: 99"), "missing value: {out}");
}

#[test]
fn entry_display_falls_back_to_hex_when_no_btf() {
    // No BTF → key/value are None; Display surfaces the hex with
    // a `(raw)` marker so the operator distinguishes "no BTF
    // render" from a parsed scalar value.
    let entry = FailureDumpEntry {
        key: None,
        key_hex: "ab cd".into(),
        value: None,
        value_hex: "ef".into(),
        payload: None,
    };
    let out = format!("{entry}");
    assert!(out.contains("ab cd (raw)"), "missing key hex: {out}");
    assert!(out.contains("ef (raw)"), "missing value hex: {out}");
}

#[test]
fn percpu_entry_display_shows_each_cpu() {
    let entry = FailureDumpPercpuEntry {
        key: 0,
        per_cpu: vec![
            Some(RenderedValue::Uint { bits: 32, value: 1 }),
            None,
            Some(RenderedValue::Uint { bits: 32, value: 3 }),
        ],
    };
    let out = format!("{entry}");
    assert!(out.contains("key 0:"));
    assert!(out.contains("cpu 0: 1"));
    assert!(out.contains("cpu 1: <unmapped>"));
    assert!(out.contains("cpu 2: 3"));
}

// ---- vcpu_regs Display coverage ---------------------------------

#[test]
fn report_display_includes_vcpu_regs_section() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: vec![
            Some(VcpuRegSnapshot {
                instruction_pointer: 0x1,
                stack_pointer: 0x2,
                page_table_root: 0x3,
                user_page_table_root: None,
                tcr_el1: None,
            }),
            None,
            Some(VcpuRegSnapshot {
                instruction_pointer: 0xa,
                stack_pointer: 0xb,
                page_table_root: 0xc,
                user_page_table_root: None,
                tcr_el1: None,
            }),
        ],
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    // Section header.
    assert!(out.starts_with("vcpu_regs:"), "missing header: {out}");
    // Three vCPU rows: 0 with values, 1 unavailable, 2 with values.
    assert!(out.contains("vcpu 0: ip=0x"), "missing vcpu 0: {out}");
    assert!(
        out.contains("vcpu 1: <unavailable>"),
        "missing vcpu 1 marker: {out}"
    );
    assert!(out.contains("vcpu 2: ip=0x"), "missing vcpu 2: {out}");
}

#[test]
fn report_display_pairs_maps_and_vcpu_regs_with_blank_line() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: vec![make_simple_map()],
        vcpu_regs: vec![Some(VcpuRegSnapshot {
            instruction_pointer: 0x1,
            stack_pointer: 0x2,
            page_table_root: 0x3,
            user_page_table_root: None,
            tcr_el1: None,
        })],
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    // Map block, blank line, vcpu_regs section.
    assert!(out.contains("\n\nvcpu_regs:"));
}

#[test]
fn report_display_empty_with_only_vcpu_regs_does_not_say_empty_dump() {
    // An all-empty maps Vec but populated vcpu_regs must still
    // render rather than fall through to "(empty failure dump)".
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: vec![None],
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    assert_eq!(out, "vcpu_regs:\n  vcpu 0: <unavailable>");
}

/// Pin the wire shape of a partial dump — the
/// "all_parked but dump prerequisites unavailable" branch in
/// `vmm::run_vm`'s freeze coordinator builds exactly this
/// shape: empty `maps`, populated `vcpu_regs`. Operators
/// reading the JSON / Display output rely on:
///   - Display NOT rendering the "(empty failure dump)"
///     fallback (which would mask the partial),
///   - Display starting with the `vcpu_regs:` section,
///   - JSON serialising `"maps":[]` (NOT skipped, since
///     `Vec::is_empty` is the skip condition only for
///     `vcpu_regs` and a few `Option`/`Vec` fields inside
///     `FailureDumpMap`, not for the top-level `maps` field).
#[test]
fn report_display_partial_with_populated_regs_and_empty_maps() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: vec![Some(VcpuRegSnapshot {
            instruction_pointer: 0xdead,
            stack_pointer: 0xbeef,
            page_table_root: 0xcafe,
            user_page_table_root: None,
            tcr_el1: None,
        })],
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };

    // (a) Display: vcpu_regs section present, no fallback.
    let out = format!("{report}");
    assert!(
        out.contains("vcpu_regs:"),
        "Display must contain the vcpu_regs section: {out}"
    );
    assert!(
        out.contains("vcpu 0: ip=0x"),
        "Display must render the BSP register row: {out}"
    );
    assert!(
        !out.contains("(empty failure dump)"),
        "Display must NOT fall through to empty fallback when \
         vcpu_regs is populated: {out}"
    );

    // (b) JSON: maps key present as empty array, NOT
    // skipped — operators downstream reliably distinguish
    // "no maps captured (partial)" from "maps key absent
    // (regression / older schema)".
    let json = serde_json::to_string(&report).expect("serialize");
    assert!(
        json.contains("\"maps\":[]"),
        "JSON must carry empty `maps` array (not skip): {json}"
    );
    assert!(
        json.contains("\"vcpu_regs\""),
        "JSON must carry vcpu_regs key: {json}"
    );
}

// -- DualFailureDumpReport serde + Display tests --

/// Roundtrip a `DualFailureDumpReport` with a populated early
/// snapshot and non-zero metric/threshold fields. Pins the wire
/// format on the dual-snapshot side: the wrapper deserialises
/// back with `early` present and the jiffies fields preserved.
#[test]
fn dual_report_serde_roundtrip_with_early() {
    let early = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: vec![None],
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let late = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: vec![None, None],
        sdt_allocations: Vec::new(),
        prog_runtime_stats: Vec::new(),
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(early),
        late,
        early_max_age_jiffies: 1234,
        early_threshold_jiffies: 600,
        early_skipped_reason: None,
    };
    let json = serde_json::to_string(&dual).unwrap();
    let parsed: DualFailureDumpReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.schema, SCHEMA_DUAL);
    assert!(parsed.early.is_some(), "early must roundtrip: {json}");
    assert_eq!(parsed.early_max_age_jiffies, 1234);
    assert_eq!(parsed.early_threshold_jiffies, 600);
    assert_eq!(parsed.late.vcpu_regs.len(), 2);
}

/// Zero `early_max_age_jiffies` / `early_threshold_jiffies`
/// must be skipped on serialize (per the
/// `skip_serializing_if = is_zero_u64` attributes). Pinning
/// this keeps the JSON tight when the early snapshot did not
/// fire — a `late`-only run yields a wrapper without the
/// trigger-metric noise.
#[test]
fn dual_report_serde_skips_zero_jiffies_fields() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let json = serde_json::to_string(&dual).unwrap();
    assert!(
        !json.contains("early_max_age_jiffies"),
        "zero early_max_age_jiffies must skip: {json}"
    );
    assert!(
        !json.contains("early_threshold_jiffies"),
        "zero early_threshold_jiffies must skip: {json}"
    );
}

/// Non-zero jiffies fields must serialize so a downstream
/// consumer can recover the trigger condition without
/// recomputing kernel arithmetic. Mirror of the
/// `skips_zero_jiffies_fields` test.
#[test]
fn dual_report_serde_emits_nonzero_jiffies_fields() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 4096,
        early_threshold_jiffies: 2048,
        early_skipped_reason: None,
    };
    let json = serde_json::to_string(&dual).unwrap();
    assert!(
        json.contains("\"early_max_age_jiffies\":4096"),
        "non-zero max_age must serialize: {json}"
    );
    assert!(
        json.contains("\"early_threshold_jiffies\":2048"),
        "non-zero threshold must serialize: {json}"
    );
}

/// The `schema` field is the wire-format discriminant.
/// `FailureDumpReport` carries `"single"`,
/// `DualFailureDumpReport` carries `"dual"`, and the two
/// values are distinguishable so a consumer can inspect a
/// single field before deciding which type to deserialize
/// into.
#[test]
fn dual_report_schema_distinguishes_from_single() {
    let single = FailureDumpReport::default();
    let single_json = serde_json::to_string(&single).unwrap();
    assert!(
        single_json.contains(&format!("\"schema\":\"{SCHEMA_SINGLE}\"")),
        "single carries schema='single': {single_json}"
    );

    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let dual_json = serde_json::to_string(&dual).unwrap();
    assert!(
        dual_json.contains(&format!("\"schema\":\"{SCHEMA_DUAL}\"")),
        "dual carries schema='dual': {dual_json}"
    );
    // The two discriminants are distinct strings — a consumer
    // checking the field can tell the variants apart without
    // attempting deserialization first.
    assert_ne!(SCHEMA_SINGLE, SCHEMA_DUAL);
}

/// Display output for the early=present branch carries the
/// summary header AND the jiffies metadata, so an operator
/// scanning a log can see at a glance whether the early
/// snapshot fired and what trigger condition produced it.
#[test]
fn dual_report_display_present_carries_jiffies() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 9001,
        early_threshold_jiffies: 4500,
        early_skipped_reason: None,
    };
    let s = format!("{dual}");
    assert!(
        s.contains("early=present"),
        "Display must say early=present: {s}"
    );
    assert!(
        s.contains("max_age=9001j"),
        "Display must surface max_age: {s}"
    );
    assert!(
        s.contains("threshold=4500j"),
        "Display must surface threshold: {s}"
    );
}

/// Display output for the early=absent branch carries the
/// summary header AND the documented absence-reason text
/// describing both possible causes (stall fired before the
/// half-way threshold; runnable_at scan setup failed) AND a
/// pointer to the RUST_LOG knob that surfaces scan-resolution
/// diagnostics — so an operator reading "early=absent" knows
/// the next debugging step rather than having to guess.
#[test]
fn dual_report_display_absent_names_both_causes() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let s = format!("{dual}");
    assert!(
        s.contains("early=absent"),
        "Display must say early=absent: {s}"
    );
    assert!(
        s.contains("stall fired before half-way threshold"),
        "Display must name the threshold-not-reached cause: {s}"
    );
    assert!(
        s.contains("runnable_at scan setup failed"),
        "Display must name the scan-setup-failure cause: {s}"
    );
    assert!(
        s.contains("RUST_LOG=ktstr=debug"),
        "Display must point at the RUST_LOG knob for diagnostics: {s}"
    );
}

// -- FailureDumpReportAny serde + Display tests --

/// `FailureDumpReportAny::from_json` picks the `Single` variant
/// for JSON whose `schema` field is `"single"`, the `Dual`
/// variant for `"dual"`, and the `Single` variant for an absent
/// `schema` field (back-compat with pre-discriminant dumps).
/// Unknown schemas return `None` rather than silently falling
/// back to single — mismatching a future richer wrapper as a
/// lossy single shape would be the wrong behaviour. Malformed
/// JSON also returns `None`.
#[test]
fn report_any_dispatch_branches() {
    // Single branch: schema="single".
    let single = FailureDumpReport::default();
    let single_json = serde_json::to_string(&single).expect("serialize single");
    match FailureDumpReportAny::from_json(&single_json) {
        Some(FailureDumpReportAny::Single(_)) => {}
        other => panic!(
            "schema=single must map to Single, got {other:?}",
            other = other.is_some()
        ),
    }

    // Dual branch: schema="dual".
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let dual_json = serde_json::to_string(&dual).expect("serialize dual");
    match FailureDumpReportAny::from_json(&dual_json) {
        Some(FailureDumpReportAny::Dual(_)) => {}
        other => panic!(
            "schema=dual must map to Dual, got {other:?}",
            other = other.is_some()
        ),
    }

    // Absent-schema branch: pre-discriminant dump.
    let absent = r#"{"maps":[],"vcpu_regs":[],"sdt_allocations":[]}"#;
    match FailureDumpReportAny::from_json(absent) {
        Some(FailureDumpReportAny::Single(_)) => {}
        other => panic!(
            "absent schema must default to Single, got {other:?}",
            other = other.is_some()
        ),
    }

    // Unknown schema → None, not a silent single fallback.
    let unknown = r#"{"schema":"triple","maps":[],"vcpu_regs":[],"sdt_allocations":[]}"#;
    assert!(
        FailureDumpReportAny::from_json(unknown).is_none(),
        "unknown schema must return None, not silent fallback"
    );

    // Malformed JSON → None.
    assert!(
        FailureDumpReportAny::from_json("not json").is_none(),
        "garbage input must return None"
    );
}

/// `prog_runtime_stats` populates and round-trips through
/// `FailureDumpReportAny::from_json`. The dispatch test above
/// covers the empty-stats path; this test pins that the field
/// survives wire encoding when populated, mirroring the
/// strict-schema concerns CgroupStats covers in assert.rs.
#[test]
fn report_any_preserves_prog_runtime_stats() {
    use super::super::bpf_prog::ProgRuntimeStats;
    let report = FailureDumpReport {
        prog_runtime_stats: vec![
            ProgRuntimeStats {
                name: "ktstr_enqueue".to_string(),
                cnt: 1_500,
                nsecs: 7_500_000,
                misses: 2,
            },
            ProgRuntimeStats {
                name: "ktstr_dispatch".to_string(),
                cnt: u64::MAX,
                nsecs: u64::MAX,
                misses: u64::MAX,
            },
        ],
        ..Default::default()
    };
    let json = serde_json::to_string(&report).expect("serialize");
    match FailureDumpReportAny::from_json(&json) {
        Some(FailureDumpReportAny::Single(loaded)) => {
            assert_eq!(loaded.prog_runtime_stats.len(), 2);
            assert_eq!(loaded.prog_runtime_stats[0].name, "ktstr_enqueue");
            assert_eq!(loaded.prog_runtime_stats[0].cnt, 1_500);
            assert_eq!(loaded.prog_runtime_stats[0].nsecs, 7_500_000);
            assert_eq!(loaded.prog_runtime_stats[0].misses, 2);
            assert_eq!(loaded.prog_runtime_stats[1].name, "ktstr_dispatch");
            assert_eq!(loaded.prog_runtime_stats[1].cnt, u64::MAX);
            assert_eq!(loaded.prog_runtime_stats[1].nsecs, u64::MAX);
            assert_eq!(loaded.prog_runtime_stats[1].misses, u64::MAX);
        }
        other => panic!(
            "populated single report must round-trip Single, got {:?}",
            other.is_some()
        ),
    }
}

/// Display roundtrip: a Single-wrapped report renders the same
/// as the underlying `FailureDumpReport`'s own Display, and a
/// Dual-wrapped report renders the same as
/// `DualFailureDumpReport`'s Display. The wrapper's Display is
/// transparent.
#[test]
fn report_any_display_matches_underlying() {
    let single = FailureDumpReport::default();
    let single_direct = format!("{single}");
    let single_via_any = format!("{}", FailureDumpReportAny::Single(Box::new(single)));
    assert_eq!(single_direct, single_via_any);

    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 42,
        early_threshold_jiffies: 21,
        early_skipped_reason: None,
    };
    let dual_direct = format!("{dual}");
    let dual_via_any = format!("{}", FailureDumpReportAny::Dual(Box::new(dual)));
    assert_eq!(dual_direct, dual_via_any);
}

// -- ProgRuntimeStats coverage in FailureDumpReport --

/// Roundtrip a populated `prog_runtime_stats` vector through
/// serde, including `u64::MAX` for every counter to lock in
/// the saturation contract documented on
/// [`super::bpf_prog::read_prog_runtime_stats`] (per-CPU sums use
/// `saturating_add`, so observing `u64::MAX` post-deserialize
/// proves the saturation path didn't silently wrap or truncate).
#[test]
fn prog_runtime_stats_serde_roundtrip_with_saturation() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats: vec![
            super::super::bpf_prog::ProgRuntimeStats {
                name: "dispatch".to_string(),
                cnt: 12345,
                nsecs: 67890,
                misses: 3,
            },
            super::super::bpf_prog::ProgRuntimeStats {
                name: "saturated".to_string(),
                cnt: u64::MAX,
                nsecs: u64::MAX,
                misses: u64::MAX,
            },
        ],
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let json = serde_json::to_string(&report).expect("serialize");
    let parsed: FailureDumpReport = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.prog_runtime_stats.len(), 2);
    assert_eq!(parsed.prog_runtime_stats[0].name, "dispatch");
    assert_eq!(parsed.prog_runtime_stats[0].cnt, 12345);
    assert_eq!(parsed.prog_runtime_stats[0].nsecs, 67890);
    assert_eq!(parsed.prog_runtime_stats[0].misses, 3);
    assert_eq!(parsed.prog_runtime_stats[1].cnt, u64::MAX);
    assert_eq!(parsed.prog_runtime_stats[1].nsecs, u64::MAX);
    assert_eq!(parsed.prog_runtime_stats[1].misses, u64::MAX);
}

/// Empty `prog_runtime_stats` skips serialization (the
/// `skip_serializing_if = "Vec::is_empty"` attribute) — same
/// pattern as the other optional vector fields. Pinning this
/// keeps the JSON tight for the common no-struct_ops-loaded
/// case.
#[test]
fn prog_runtime_stats_empty_skips_serialization() {
    let report = FailureDumpReport::default();
    let json = serde_json::to_string(&report).expect("serialize");
    assert!(
        !json.contains("prog_runtime_stats"),
        "empty prog_runtime_stats must be skipped: {json}"
    );
}

/// Display impl renders `prog_runtime_stats` under a labelled
/// section so an operator scanning failure-dump output sees
/// the per-program counters alongside the maps / vcpu_regs /
/// sdt_allocations sections. Pinning this prevents the
/// "rendered fields silently drop" regression that would mask
/// dump enrichment from reaching log readers.
#[test]
fn report_display_renders_prog_runtime_stats() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats: vec![
            super::super::bpf_prog::ProgRuntimeStats {
                name: "dispatch".to_string(),
                cnt: 5,
                nsecs: 1234,
                misses: 0,
            },
            super::super::bpf_prog::ProgRuntimeStats {
                name: "enqueue".to_string(),
                cnt: 99,
                nsecs: 9999,
                misses: 7,
            },
        ],
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    assert!(
        out.contains("prog_runtime_stats:"),
        "Display must render the prog_runtime_stats section: {out}"
    );
    assert!(
        out.contains("dispatch: cnt=5 nsecs=1234 misses=0"),
        "Display must render first program line: {out}"
    );
    assert!(
        out.contains("enqueue: cnt=99 nsecs=9999 misses=7"),
        "Display must render second program line: {out}"
    );
}

/// An all-empty maps/vcpu_regs/sdt_allocations report with
/// only `prog_runtime_stats` populated must still render
/// rather than fall through to the "(empty failure dump)"
/// fallback — the empty-check in the Display impl gates on
/// every optional vector, including `prog_runtime_stats`.
#[test]
fn report_display_only_prog_runtime_stats_does_not_say_empty_dump() {
    let report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::new(),
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats: vec![super::super::bpf_prog::ProgRuntimeStats {
            name: "lone".to_string(),
            cnt: 1,
            nsecs: 2,
            misses: 0,
        }],
        prog_runtime_stats_unavailable: None,
        per_cpu_time: Vec::new(),
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: None,
        task_enrichments: Vec::new(),
        task_enrichments_unavailable: None,
        event_counter_timeline: Vec::new(),
        rq_scx_states: Vec::new(),
        dsq_states: Vec::new(),
        scx_sched_state: None,
        scx_walker_unavailable: None,
        vcpu_perf_at_freeze: Vec::new(),
        dump_truncated_at_us: None,
        probe_counters: None,
        scx_static_ranges: Default::default(),
        is_placeholder: false,
    };
    let out = format!("{report}");
    assert!(
        !out.contains("(empty failure dump)"),
        "Display must NOT fall through to empty fallback when \
         prog_runtime_stats is populated: {out}"
    );
    assert!(
        out.starts_with("prog_runtime_stats:"),
        "Display must lead with prog_runtime_stats section when \
         only that field is populated: {out}"
    );
}

// ---- pin failure-dump error-message strings --------------------
//
// The six REASON_* constants emitted by `dump_state` into the
// `*_unavailable` fields are wire-format markers: an operator
// parsing `.failure-dump.json` looks for these exact strings to
// distinguish "no scheduler attached" from "no walker capture
// supplied" etc. Drift in any of them silently breaks downstream
// parsing. The constants near the top of this module are the
// single source of truth; the tests below pin each constant's
// exact value so a regression that re-words a string trips both
// at the constant declaration AND at the test assertion.
//
// The companion strict-schema and chain-limit tests for
// FailureDumpReport / is_scx_allocator_type live further down in
// this module.

#[test]
fn reason_no_struct_ops_loaded_string_pinned() {
    assert_eq!(REASON_NO_STRUCT_OPS_LOADED, "no struct_ops programs loaded");
}

#[test]
fn reason_prog_accessor_unavailable_string_pinned() {
    assert_eq!(
        REASON_PROG_ACCESSOR_UNAVAILABLE,
        "prog accessor unavailable"
    );
}

#[test]
fn reason_task_walker_zero_tasks_string_pinned() {
    assert_eq!(
        REASON_TASK_WALKER_ZERO_TASKS,
        "task walker yielded zero tasks"
    );
}

#[test]
fn reason_no_task_walker_string_pinned() {
    assert_eq!(REASON_NO_TASK_WALKER, "no task walker available");
}

#[test]
fn reason_scx_walker_no_state_string_pinned() {
    assert_eq!(REASON_SCX_WALKER_NO_STATE, "scx walker reached no state");
}

#[test]
fn reason_scx_root_null_string_pinned() {
    assert_eq!(
        REASON_SCX_ROOT_NULL,
        "scx_root is NULL (no scheduler attached)"
    );
}

#[test]
fn reason_no_scx_walker_string_pinned() {
    assert_eq!(REASON_NO_SCX_WALKER, "no scx walker capture");
}

/// Every reason constant must round-trip through the JSON wire
/// format embedded in the `*_unavailable` fields. A regression
/// that altered the field's serde encoding (renamed the field,
/// added `#[serde(rename = ...)]`, etc.) would also break the
/// operator's string-match parsing — surface that here too.
#[test]
fn reason_strings_round_trip_through_serde() {
    let report = FailureDumpReport {
        prog_runtime_stats_unavailable: Some(REASON_NO_STRUCT_OPS_LOADED.to_string()),
        task_enrichments_unavailable: Some(REASON_TASK_WALKER_ZERO_TASKS.to_string()),
        scx_walker_unavailable: Some(REASON_SCX_WALKER_NO_STATE.to_string()),
        ..Default::default()
    };
    let json = serde_json::to_string(&report).expect("serialize");
    // Each reason must appear verbatim in the JSON; a future
    // wire-format change (e.g. tagged enum) would hide them
    // behind nested objects and trip this assertion.
    assert!(
        json.contains(REASON_NO_STRUCT_OPS_LOADED),
        "JSON must contain prog reason verbatim: {json}",
    );
    assert!(
        json.contains(REASON_TASK_WALKER_ZERO_TASKS),
        "JSON must contain task reason verbatim: {json}",
    );
    assert!(
        json.contains(REASON_SCX_WALKER_NO_STATE),
        "JSON must contain scx reason verbatim: {json}",
    );

    let loaded: FailureDumpReport = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        loaded.prog_runtime_stats_unavailable.as_deref(),
        Some(REASON_NO_STRUCT_OPS_LOADED),
    );
    assert_eq!(
        loaded.task_enrichments_unavailable.as_deref(),
        Some(REASON_TASK_WALKER_ZERO_TASKS),
    );
    assert_eq!(
        loaded.scx_walker_unavailable.as_deref(),
        Some(REASON_SCX_WALKER_NO_STATE),
    );
}

// -- Strict-schema tests for FailureDumpReport -------------------
//
// Mirrors the CgroupStats / ScenarioStats / SidecarResult tests in
// assert.rs and test_support/sidecar.rs. FailureDumpReport's
// contract is narrower than CgroupStats's because most of its
// fields are intentionally optional (capture pipelines may
// legitimately produce no entries), so the CgroupStats
// "remove every field" loop would over-assert here.
//
// Asserted contract:
//   - `maps` is the only required field on the wire.
//   - `schema` is `serde(default = default_schema_single)` —
//     omission yields `SCHEMA_SINGLE`.
//   - Every other field is `serde(default, skip_serializing_if =
//     ...)` — omission MUST succeed.
//
// A regression that softens `maps` to `serde(default)` (e.g. to
// soften a schema migration) would silently produce empty-maps
// dumps that look indistinguishable from a legitimate no-maps
// run. A regression that hardens an optional field to require it
// on the wire would break replay of older dumps. Either drift
// trips this test.

/// Removing the `maps` field MUST fail deserialize. `maps`
/// carries the BPF map enumeration that is the dump's only
/// mandatory payload — every other field is
/// capture-pipeline-optional. The deserialize error MUST name
/// `maps` so a regression produces a debuggable failure rather
/// than a silent default.
#[test]
fn failure_dump_report_strict_schema_maps_required() {
    let report = FailureDumpReport::default();
    let mut full = match serde_json::to_value(&report).unwrap() {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };
    assert!(
        full.remove("maps").is_some(),
        "FailureDumpReport must emit `maps` for this test to be \
         meaningful — the field has been renamed or removed",
    );
    let json = serde_json::Value::Object(full).to_string();
    let err = serde_json::from_str::<FailureDumpReport>(&json)
        .expect_err("deserialize must reject FailureDumpReport with `maps` removed");
    let msg = format!("{err}");
    assert!(
        msg.contains("maps"),
        "missing-field error for `maps` must name the field; got: {msg}",
    );
}

/// Omitting all optional fields (`schema`, `vcpu_regs`,
/// `sdt_allocations`, every diagnostic Option, every capture
/// Vec) MUST succeed and produce a deserialized report whose
/// absent fields take their `serde(default)` value. `schema`
/// gets a positive control: omission MUST yield `SCHEMA_SINGLE`,
/// not the empty string a naive `Default for String` would
/// produce.
#[test]
fn failure_dump_report_optional_fields_round_trip_when_omitted() {
    let minimal = serde_json::json!({ "maps": [] });
    let report: FailureDumpReport = serde_json::from_value(minimal)
        .expect("deserialize must accept FailureDumpReport with only `maps`");
    assert_eq!(
        report.schema, SCHEMA_SINGLE,
        "absent `schema` field must default to SCHEMA_SINGLE \
         (default_schema_single fn); got: {:?}",
        report.schema,
    );
    assert!(report.maps.is_empty());
    assert!(report.vcpu_regs.is_empty());
    assert!(report.sdt_allocations.is_empty());
    assert!(report.prog_runtime_stats.is_empty());
    assert!(report.prog_runtime_stats_unavailable.is_none());
    assert!(report.per_cpu_time.is_empty());
    assert!(report.per_node_numa.is_empty());
    assert!(report.per_node_numa_unavailable.is_none());
    assert!(report.task_enrichments.is_empty());
    assert!(report.task_enrichments_unavailable.is_none());
    assert!(report.event_counter_timeline.is_empty());
    assert!(report.rq_scx_states.is_empty());
    assert!(report.dsq_states.is_empty());
    assert!(report.scx_sched_state.is_none());
    assert!(report.scx_walker_unavailable.is_none());
    assert!(report.vcpu_perf_at_freeze.is_empty());
}

// -- Pin failure-dump error-message strings ----------------------
//
// Pin the EXACT prose of error strings rendered into
// FailureDumpMap.error. Substring tests are permissive against
// drift; this regression suite asserts byte-for-byte equality so
// any re-wording during refactor surfaces in `cargo nextest run`
// before it ships.
//
// The strings are observable via FailureDumpMap.error contents
// and via downstream log scrapers (operators grep these in CI
// logs). Changing them silently breaks log tooling. Each pin
// doubles as documentation: this file shows exactly which prose
// is covered by drift detection.
//
// dump/render_map.rs producers covered here — five distinct
// render-time formats:
//   - BPF_MAP_TYPE_ARENA, no offsets
//   - BPF_MAP_TYPE_ARRAY, multi-entry
//   - BPF_MAP_TYPE_HASH, truncation
//   - BPF_MAP_TYPE_PERCPU_ARRAY, truncation
//   - unsupported map_type wildcard
//
// Each pin reproduces the production format string against a
// known-value placeholder and asserts byte equality with the
// expected literal. A drift in either the prose or the constant
// value (e.g. raising MAX_HASH_ENTRIES from 4096 to 8192) trips
// the test.
//
// The companion REASON_* constants for diagnostic Option fields
// (REASON_NO_STRUCT_OPS_LOADED, REASON_TASK_WALKER_ZERO_TASKS,
// REASON_SCX_WALKER_NO_STATE, ...) are already pinned by tests
// earlier in this module — see `report_unavailable_reasons_*`.

/// `arena BTF offsets unavailable (kernel lacks struct bpf_arena?)`
/// is rendered by the BPF_MAP_TYPE_ARENA arm when arena_offsets
/// is None — surfacing that the kernel lacks struct bpf_arena.
#[test]
fn pinned_error_arena_btf_offsets_unavailable() {
    // The producer has no format placeholders; reproduce the
    // exact `.into()` literal so a rephrasing in dump/render_map.rs trips.
    let rendered: String = "arena BTF offsets unavailable (kernel lacks struct bpf_arena?)".into();
    assert_eq!(
        rendered, "arena BTF offsets unavailable (kernel lacks struct bpf_arena?)",
        "arena-unavailable error string drifted from pin",
    );
}

/// `multi-entry ARRAY: only key 0 of {N} shown` is rendered by
/// the BPF_MAP_TYPE_ARRAY arm when `info.max_entries > 1`.
#[test]
fn pinned_error_multi_entry_array_truncation() {
    let n: u32 = 7;
    let rendered = format!("multi-entry ARRAY: only key 0 of {n} shown");
    assert_eq!(
        rendered, "multi-entry ARRAY: only key 0 of 7 shown",
        "multi-entry ARRAY truncation string drifted from pin",
    );
}

/// `hash map truncated at {MAX_HASH_ENTRIES} entries` is
/// rendered by the BPF_MAP_TYPE_HASH arm. Pins both the prose
/// and `MAX_HASH_ENTRIES` so either drifting trips the test.
#[test]
fn pinned_error_hash_map_truncation() {
    let rendered = format!("hash map truncated at {MAX_HASH_ENTRIES} entries");
    assert_eq!(
        rendered, "hash map truncated at 4096 entries",
        "hash map truncation string OR MAX_HASH_ENTRIES drifted from pin",
    );
}

/// `PERCPU_ARRAY truncated at {MAX_PERCPU_KEYS} keys (max_entries={N})`
/// is rendered by the BPF_MAP_TYPE_PERCPU_ARRAY arm. Pins prose,
/// constant, and placeholder ordering.
#[test]
fn pinned_error_percpu_array_truncation() {
    let max_entries: u32 = 999;
    let rendered =
        format!("PERCPU_ARRAY truncated at {MAX_PERCPU_KEYS} keys (max_entries={max_entries})",);
    assert_eq!(
        rendered, "PERCPU_ARRAY truncated at 256 keys (max_entries=999)",
        "PERCPU_ARRAY truncation string OR MAX_PERCPU_KEYS drifted from pin",
    );
}

/// `unknown map_type {N}` is rendered by the wildcard arm for any
/// map_type past the kernel-uapi enum the dump renderer was built
/// against. The dispatch now enumerates every known map_type
/// (HASH/ARRAY/PERCPU_*/LRU_*/STRUCT_OPS/RINGBUF/storage/FD-array
/// families/QUEUE/STACK/BLOOM_FILTER/LPM_TRIE/INSN_ARRAY/ARENA),
/// so the wildcard only fires for kernels newer than the
/// renderer.
#[test]
fn pinned_error_unknown_map_type() {
    let other: u32 = 42;
    let rendered = format!(
        "unknown map_type {other} (kernel newer than dump renderer; \
         update render_map dispatch)"
    );
    assert_eq!(
        rendered,
        "unknown map_type 42 (kernel newer than dump renderer; update render_map dispatch)",
        "unknown-map-type string drifted from pin",
    );
}

/// Local-storage truncation diagnostic: when a TASK / INODE / SK
/// / CGRP_STORAGE map holds more than [`MAX_HASH_ENTRIES`] selems,
/// the renderer surfaces a wire-stable error so a log scraper
/// sees the cap at the same shape as the `hash map truncated`
/// diagnostic. Pin the literal in case [`MAX_HASH_ENTRIES`] or
/// the format string drift.
#[test]
fn pinned_error_local_storage_truncation() {
    let rendered = format!("local_storage map truncated at {MAX_HASH_ENTRIES} entries");
    assert_eq!(
        rendered, "local_storage map truncated at 4096 entries",
        "local_storage truncation string OR MAX_HASH_ENTRIES drifted",
    );
}

// -- Per-node NUMA stats wire shape -------------------------------
//
// The live walker is a follow-up; this section pins the wire
// contract so the schema is stable before the producer lands.
//
// Asserted properties:
//   - PerNodeNumaStats serde-roundtrips with all fields preserved
//   - empty per_node_numa skips serialization (skip_serializing_if)
//   - REASON_NO_NUMA_WALKER string is exactly pinned (wire-stable)
//   - the dump_state path emits per_node_numa_unavailable until
//     the walker lands, so a downstream consumer sees the
//     diagnostic even on dumps that complete every other capture

#[test]
fn per_node_numa_stats_serde_roundtrip() {
    let s = PerNodeNumaStats {
        node: 1,
        numa_hit: 1_000_000,
        numa_miss: 100,
        numa_foreign: 50,
        numa_interleave_hit: 200,
        numa_local: 999_900,
        numa_other: 100,
    };
    let json = serde_json::to_string(&s).unwrap();
    let parsed: PerNodeNumaStats = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.node, 1);
    assert_eq!(parsed.numa_hit, 1_000_000);
    assert_eq!(parsed.numa_miss, 100);
    assert_eq!(parsed.numa_foreign, 50);
    assert_eq!(parsed.numa_interleave_hit, 200);
    assert_eq!(parsed.numa_local, 999_900);
    assert_eq!(parsed.numa_other, 100);
}

#[test]
fn per_node_numa_empty_skips_serialization() {
    let report = FailureDumpReport::default();
    let json = serde_json::to_string(&report).unwrap();
    assert!(
        !json.contains("per_node_numa"),
        "empty per_node_numa must be skipped: {json}",
    );
}

#[test]
fn per_node_numa_populated_lands_in_wire() {
    let mut report = FailureDumpReport::default();
    report.per_node_numa.push(PerNodeNumaStats {
        node: 0,
        numa_hit: 42,
        ..Default::default()
    });
    let json = serde_json::to_string(&report).unwrap();
    assert!(
        json.contains("\"per_node_numa\""),
        "populated per_node_numa must appear on wire: {json}",
    );
    assert!(
        json.contains("\"numa_hit\":42"),
        "field value must round-trip: {json}",
    );
}

#[test]
fn reason_no_numa_walker_string_pinned() {
    // Wire-stable diagnostic — operators string-match this in
    // failure-dump consumers. Drift would silently break that
    // tooling. The constant is the single source of truth; this
    // test pins it byte-for-byte.
    assert_eq!(
        REASON_NO_NUMA_WALKER,
        "no NUMA walker (host-side walker pending)"
    );
}

// -- FailureDumpPercpuHashEntry coverage --------------------------
//
// The PERCPU_HASH / LRU_PERCPU_HASH render path produces this
// shape: one entry per htab_elem with a rendered (or raw-hex)
// key plus one Vec<Option<RenderedValue>> slot per CPU. Mirror
// of FailureDumpPercpuEntry but keyed by hash key rather than
// an array index.

/// Display impl: with a rendered key and per-CPU values, the
/// output uses the indent-based `entry: key=` header with one
/// `cpu N:` line per slot.
#[test]
fn percpu_hash_entry_display_shows_key_and_cpus() {
    let entry = FailureDumpPercpuHashEntry {
        key: Some(RenderedValue::Uint { bits: 32, value: 7 }),
        key_hex: "07 00 00 00".into(),
        per_cpu: vec![
            Some(RenderedValue::Uint {
                bits: 32,
                value: 100,
            }),
            None,
            Some(RenderedValue::Uint {
                bits: 32,
                value: 300,
            }),
        ],
    };
    let out = format!("{entry}");
    assert!(out.starts_with("entry: key="), "entry header: {out}");
    assert!(out.contains("entry: key=7"), "rendered key: {out}");
    assert!(out.contains("cpu 0: 100"), "cpu 0 value: {out}");
    assert!(out.contains("cpu 1: <unmapped>"), "cpu 1 unmapped: {out}");
    assert!(out.contains("cpu 2: 300"), "cpu 2 value: {out}");
}

/// When BTF is unavailable the rendered key is `None`; Display
/// falls back to the raw hex representation with a `(raw)` tag,
/// mirroring FailureDumpEntry.
#[test]
fn percpu_hash_entry_display_falls_back_to_hex_when_no_btf() {
    let entry = FailureDumpPercpuHashEntry {
        key: None,
        key_hex: "ab cd ef 01".into(),
        per_cpu: vec![Some(RenderedValue::Uint { bits: 32, value: 1 })],
    };
    let out = format!("{entry}");
    assert!(
        out.contains("ab cd ef 01 (raw)"),
        "raw hex with (raw) marker: {out}",
    );
}

/// Empty per-CPU slot list — every CPU `<unmapped>`. The shape
/// stays well-formed (header + body or just unmapped markers)
/// rather than panicking.
#[test]
fn percpu_hash_entry_display_all_unmapped_cpus() {
    let entry = FailureDumpPercpuHashEntry {
        key: Some(RenderedValue::Uint { bits: 32, value: 0 }),
        key_hex: "00 00 00 00".into(),
        per_cpu: vec![None, None, None],
    };
    let out = format!("{entry}");
    assert!(out.contains("cpu 0: <unmapped>"));
    assert!(out.contains("cpu 1: <unmapped>"));
    assert!(out.contains("cpu 2: <unmapped>"));
}

/// Empty per_cpu vec — entry body has no `cpu N:` lines but the
/// `entry: key=` header is still emitted.
#[test]
fn percpu_hash_entry_display_empty_per_cpu() {
    let entry = FailureDumpPercpuHashEntry {
        key: Some(RenderedValue::Uint { bits: 32, value: 0 }),
        key_hex: "00 00 00 00".into(),
        per_cpu: vec![],
    };
    let out = format!("{entry}");
    assert!(out.starts_with("entry: key="), "header: {out}");
    assert!(!out.contains("cpu "), "no cpu lines: {out}");
}

/// Serde roundtrip: every field preserved on encode/decode.
/// Pin the wire shape so a future serde-attribute change
/// (rename, skip_serializing_if) trips the test.
#[test]
fn percpu_hash_entry_serde_roundtrip() {
    let entry = FailureDumpPercpuHashEntry {
        key: Some(RenderedValue::Uint {
            bits: 32,
            value: 42,
        }),
        key_hex: "2a 00 00 00".into(),
        per_cpu: vec![
            Some(RenderedValue::Uint {
                bits: 32,
                value: 100,
            }),
            None,
        ],
    };
    let json = serde_json::to_string(&entry).expect("serialize");
    let parsed: FailureDumpPercpuHashEntry = serde_json::from_str(&json).expect("deserialize");
    assert!(parsed.key.is_some());
    assert_eq!(parsed.key_hex, "2a 00 00 00");
    assert_eq!(parsed.per_cpu.len(), 2);
    assert!(parsed.per_cpu[0].is_some());
    assert!(parsed.per_cpu[1].is_none());
}

/// `key` is skip_serializing_if=Option::is_none — when absent on
/// the wire it must omit, then deserialize back as None.
#[test]
fn percpu_hash_entry_key_skips_when_none() {
    let entry = FailureDumpPercpuHashEntry {
        key: None,
        key_hex: "00".into(),
        per_cpu: vec![],
    };
    let json = serde_json::to_string(&entry).unwrap();
    assert!(
        !json.contains("\"key\":"),
        "None key must skip on wire: {json}",
    );
}

// -- FailureDumpMap with percpu_hash_entries Display --------------

/// A FailureDumpMap of type PERCPU_HASH renders the percpu_hash
/// entries below the header — matches the existing percpu_entries
/// pattern.
#[test]
fn map_display_percpu_hash_entries_render() {
    let m = FailureDumpMap {
        name: "percpu_hash".into(),
        map_type: BPF_MAP_TYPE_PERCPU_HASH,
        value_size: 4,
        max_entries: 100,
        value: None,
        entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: vec![FailureDumpPercpuHashEntry {
            key: Some(RenderedValue::Uint { bits: 32, value: 1 }),
            key_hex: "01 00 00 00".into(),
            per_cpu: vec![
                Some(RenderedValue::Uint {
                    bits: 32,
                    value: 10,
                }),
                Some(RenderedValue::Uint {
                    bits: 32,
                    value: 20,
                }),
            ],
        }],
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let out = format!("{m}");
    assert!(out.contains("map percpu_hash (type="), "header: {out}");
    assert!(out.contains("entry: key=1"), "key surfaces: {out}");
    assert!(out.contains("cpu 0: 10"), "cpu 0: {out}");
    assert!(out.contains("cpu 1: 20"), "cpu 1: {out}");
}

// -- New pinned error strings on render_map arms ------------------
//
// Each new arm in the render_map dispatch produces a wire-stable
// error string. Pin every one byte-for-byte so a re-wording
// refactor surfaces in cargo nextest before it reaches a log
// scraper.

#[test]
fn pinned_error_percpu_hash_truncation() {
    let rendered = format!("percpu hash map truncated at {MAX_HASH_ENTRIES} entries");
    assert_eq!(
        rendered, "percpu hash map truncated at 4096 entries",
        "percpu hash truncation string OR MAX_HASH_ENTRIES drifted",
    );
}

/// STRUCT_OPS produces TWO distinct error strings since the
/// offsets-vs-region split landed: one when struct_ops_offsets
/// are absent, one when the value region is unmapped.
#[test]
fn pinned_error_struct_ops_offsets_unresolved() {
    let expected = "STRUCT_OPS value unreadable: bpf_struct_ops_map BTF offsets unresolved \
         (kernel without struct_ops support, or vmlinux BTF stripped of \
         bpf_struct_ops_map / bpf_struct_ops_value).";
    let rendered: String = expected.into();
    assert_eq!(
        rendered, expected,
        "STRUCT_OPS offsets-unresolved error string drifted",
    );
}

#[test]
fn pinned_error_struct_ops_region_unmapped() {
    let expected = "STRUCT_OPS value unreadable: value region unmapped. Live-host \
         backend reads via BPF_MAP_LOOKUP_ELEM at key=0.";
    let rendered: String = expected.into();
    assert_eq!(
        rendered, expected,
        "STRUCT_OPS region-unmapped error string drifted",
    );
}

#[test]
fn pinned_error_cgroup_storage_deprecated() {
    let rendered: String =
        "deprecated cgroup-attached storage; use CGRP_STORAGE on newer kernels".into();
    assert_eq!(
        rendered,
        "deprecated cgroup-attached storage; use CGRP_STORAGE on newer kernels",
    );
}

#[test]
fn pinned_error_queue_stack_destructive() {
    let expected = "QUEUE/STACK are destructive (peek shows only the head; pop consumes); \
         no enumeration API";
    let rendered: String = expected.into();
    assert_eq!(rendered, expected);
}

#[test]
fn pinned_error_bloom_filter() {
    let rendered: String =
        "BLOOM_FILTER is a probabilistic set; no key enumeration is possible".into();
    assert_eq!(
        rendered,
        "BLOOM_FILTER is a probabilistic set; no key enumeration is possible",
    );
}

#[test]
fn pinned_error_lpm_trie() {
    let expected = "LPM_TRIE walker not implemented (keyed by prefixlen + data); \
         use bpf(2) BPF_MAP_GET_NEXT_KEY for live-host iteration";
    let rendered: String = expected.into();
    assert_eq!(rendered, expected);
}

#[test]
fn pinned_error_unknown_map_type_format() {
    // The wildcard arm format string carries a placeholder.
    let other: u32 = 99;
    let rendered = format!(
        "unknown map_type {other} (kernel newer than dump renderer; \
         update render_map dispatch)"
    );
    assert_eq!(
        rendered,
        "unknown map_type 99 (kernel newer than dump renderer; \
         update render_map dispatch)",
    );
}

// -- FailureDumpMap.percpu_hash_entries field round-trip ----------

/// `percpu_hash_entries` is `skip_serializing_if = Vec::is_empty`
/// — empty must be omitted; populated must round-trip.
#[test]
fn map_percpu_hash_entries_skips_when_empty() {
    let m = FailureDumpMap {
        name: "test".into(),
        map_type: BPF_MAP_TYPE_HASH,
        value_size: 4,
        max_entries: 1,
        value: None,
        entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let json = serde_json::to_string(&m).unwrap();
    assert!(
        !json.contains("percpu_hash_entries"),
        "empty must skip: {json}",
    );
}

#[test]
fn map_percpu_hash_entries_round_trip_when_populated() {
    let m = FailureDumpMap {
        name: "ph".into(),
        map_type: BPF_MAP_TYPE_PERCPU_HASH,
        value_size: 4,
        max_entries: 1,
        value: None,
        entries: Vec::new(),
        percpu_entries: Vec::new(),
        percpu_hash_entries: vec![FailureDumpPercpuHashEntry {
            key: Some(RenderedValue::Uint { bits: 32, value: 1 }),
            key_hex: "01 00 00 00".into(),
            per_cpu: vec![Some(RenderedValue::Uint {
                bits: 32,
                value: 99,
            })],
        }],
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let json = serde_json::to_string(&m).expect("serialize");
    assert!(
        json.contains("percpu_hash_entries"),
        "populated must serialize: {json}",
    );
    let parsed: FailureDumpMap = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.percpu_hash_entries.len(), 1);
    assert_eq!(parsed.percpu_hash_entries[0].key_hex, "01 00 00 00");
    assert_eq!(parsed.percpu_hash_entries[0].per_cpu.len(), 1);
}

// -- AccessorMemReader-shape arena pointer chasing ---------------
//
// The MemReader trait is implemented by AccessorMemReader. The
// `is_arena_addr` and `read_arena` methods don't need a guest
// kernel — they only consult the arena_snapshot field
// (specifically `snap.user_vm_start` and `snap.pages`). Test
// those paths via stand-in types that mirror production logic
// line-for-line.

/// `is_arena_addr` returns false when the snapshot is None
/// (no arena attached). Mirrors the no-arena fast path.
#[test]
fn accessor_mem_reader_no_snapshot_rejects_all_addrs() {
    struct StubReader<'a> {
        arena_snapshot: Option<&'a super::super::arena::ArenaSnapshot>,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn is_arena_addr(&self, addr: u64) -> bool {
            let Some(snap) = self.arena_snapshot else {
                return false;
            };
            if snap.user_vm_start == 0 {
                return false;
            }
            addr >= snap.user_vm_start && addr < snap.user_vm_start.wrapping_add(1 << 32)
        }
    }
    let r = StubReader {
        arena_snapshot: None,
    };
    assert!(!r.is_arena_addr(0));
    assert!(!r.is_arena_addr(0x10000));
    assert!(!r.is_arena_addr(u64::MAX));
}

/// `is_arena_addr` returns false when the snapshot is present
/// but `user_vm_start == 0` (the snapshot bailed before reading
/// the user_vm_start anchor).
#[test]
fn accessor_mem_reader_zero_user_vm_start_rejects_all() {
    use super::super::arena::ArenaSnapshot;
    let snap = ArenaSnapshot {
        user_vm_start: 0,
        ..ArenaSnapshot::default()
    };
    struct StubReader<'a> {
        arena_snapshot: Option<&'a ArenaSnapshot>,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn is_arena_addr(&self, addr: u64) -> bool {
            let Some(snap) = self.arena_snapshot else {
                return false;
            };
            if snap.user_vm_start == 0 {
                return false;
            }
            addr >= snap.user_vm_start && addr < snap.user_vm_start.wrapping_add(1 << 32)
        }
    }
    let r = StubReader {
        arena_snapshot: Some(&snap),
    };
    // Even with a snapshot present, user_vm_start=0 means no
    // arena base anchor → reject every address.
    assert!(!r.is_arena_addr(0));
    assert!(!r.is_arena_addr(0x100000));
}

/// `is_arena_addr` enforces the `[user_vm_start, user_vm_start +
/// 4 GiB)` half-open range reflecting the kernel's SZ_4G
/// enforcement in arena_map_alloc.
#[test]
fn accessor_mem_reader_arena_addr_range_via_snapshot() {
    use super::super::arena::ArenaSnapshot;
    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    struct StubReader<'a> {
        arena_snapshot: Option<&'a ArenaSnapshot>,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn is_arena_addr(&self, addr: u64) -> bool {
            let Some(snap) = self.arena_snapshot else {
                return false;
            };
            if snap.user_vm_start == 0 {
                return false;
            }
            addr >= snap.user_vm_start && addr < snap.user_vm_start.wrapping_add(1 << 32)
        }
    }
    let r = StubReader {
        arena_snapshot: Some(&snap),
    };
    // Below start: rejected.
    assert!(!r.is_arena_addr(0));
    assert!(!r.is_arena_addr(0xf_ffff_ffff));
    // At start: accepted.
    assert!(r.is_arena_addr(0x10_0000_0000));
    // Just below upper bound: accepted.
    assert!(r.is_arena_addr(0x10_0000_0000 + (1 << 32) - 1));
    // At upper bound: rejected (exclusive).
    assert!(!r.is_arena_addr(0x10_0000_0000 + (1 << 32)));
}

/// `read_arena` returns None when no snapshot is attached.
#[test]
fn accessor_mem_reader_read_arena_none_when_no_snapshot() {
    struct StubReader<'a> {
        arena_snapshot: Option<&'a super::super::arena::ArenaSnapshot>,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn read_arena(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            self.arena_snapshot?;
            None
        }
    }
    let r = StubReader {
        arena_snapshot: None,
    };
    assert!(r.read_arena(0x1234, 8).is_none());
}

/// `read_arena` page-aligns the address and returns the matching
/// page's bytes when the full request fits in one page. Mirrors the
/// production `read_arena` logic line-for-line so a regression in
/// either trips the test. Cross-page reads (where `offset + len`
/// exceeds the page) return None per the documented MemReader
/// contract — partial bytes are not handed back.
#[test]
fn accessor_mem_reader_read_arena_page_hit() {
    use super::super::arena::{ArenaPage, ArenaSnapshot};
    let snap = ArenaSnapshot {
        pages: vec![ArenaPage {
            user_addr: 0x1000,
            bytes: (0..=0xffu8).cycle().take(4096).collect(),
        }],
        ..ArenaSnapshot::default()
    };

    struct StubReader<'a> {
        arena_snapshot: &'a ArenaSnapshot,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
            let page_addr = addr & !0xFFF;
            let offset = (addr & 0xFFF) as usize;
            let page = self
                .arena_snapshot
                .pages
                .iter()
                .find(|p| p.user_addr == page_addr)?;
            if offset + len > page.bytes.len() {
                // Contract: full request cannot be satisfied → None.
                return None;
            }
            Some(page.bytes[offset..offset + len].to_vec())
        }
    }
    let r = StubReader {
        arena_snapshot: &snap,
    };

    // Read at page base: byte 0..8 of (0..=0xff cycled).
    let bytes = r.read_arena(0x1000, 8).expect("page-aligned hit");
    assert_eq!(bytes, vec![0, 1, 2, 3, 4, 5, 6, 7]);

    // Read at offset 100: bytes 100..108.
    let bytes = r.read_arena(0x1000 + 100, 8).expect("offset hit");
    assert_eq!(bytes[0], 100);

    // Read past page end: contract says None. The full request
    // cannot be satisfied from one captured page, so the reader
    // declines rather than handing back a short slice.
    assert!(
        r.read_arena(0x1000 + 4090, 100).is_none(),
        "cross-page read must return None per MemReader::read_arena contract",
    );
}

/// `read_arena` returns None when the address falls outside the
/// captured pages (page miss).
#[test]
fn accessor_mem_reader_read_arena_page_miss_returns_none() {
    use super::super::arena::{ArenaPage, ArenaSnapshot};
    let snap = ArenaSnapshot {
        pages: vec![ArenaPage {
            user_addr: 0x1000,
            bytes: vec![0u8; 4096],
        }],
        ..ArenaSnapshot::default()
    };

    struct StubReader<'a> {
        arena_snapshot: &'a ArenaSnapshot,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn read_arena(&self, addr: u64, len: usize) -> Option<Vec<u8>> {
            let page_addr = addr & !0xFFF;
            let offset = (addr & 0xFFF) as usize;
            let page = self
                .arena_snapshot
                .pages
                .iter()
                .find(|p| p.user_addr == page_addr)?;
            if offset + len > page.bytes.len() {
                // Contract: full request cannot be satisfied → None.
                return None;
            }
            Some(page.bytes[offset..offset + len].to_vec())
        }
    }
    let r = StubReader {
        arena_snapshot: &snap,
    };
    // 0x2000 is NOT in the captured pages → None.
    assert!(r.read_arena(0x2000, 8).is_none());
}

/// MemReader default impls: any reader that doesn't override
/// `is_arena_addr` and `read_arena` gets the default `false` /
/// `None` behavior.
#[test]
fn mem_reader_default_impls_skip_arena() {
    struct MinReader;
    impl super::super::btf_render::MemReader for MinReader {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
    }
    let r = MinReader;
    assert!(!r.is_arena_addr(0));
    assert!(!r.is_arena_addr(u64::MAX));
    assert!(r.read_arena(0x1234, 8).is_none());
}

// -- AccessorMemReader cast_lookup --------------------------------
//
// `AccessorMemReader::cast_lookup` (render_map.rs) consults its
// `cast_map` field: `Some(map)` returns
// `map.get(&(parent_type_id, member_byte_offset)).copied()`,
// `None` returns `None`. The struct itself is private to
// render_map.rs, so these tests use a stand-in `StubReader` that
// mirrors the production method body verbatim — same convention
// as the surrounding AccessorMemReader-shape tests
// (`accessor_mem_reader_no_snapshot_rejects_all_addrs` and the
// rest in this section).

/// `cast_lookup` with a populated [`CastMap`] returns the matching
/// [`CastHit`] when the `(parent_type_id, member_byte_offset)` key
/// is present, and `None` when it is not. Mirrors the production
/// `AccessorMemReader::cast_lookup` body line-for-line so a
/// regression in either trips this test.
#[test]
fn accessor_mem_reader_cast_lookup_with_populated_map() {
    use super::super::btf_render::CastHit;
    use super::super::cast_analysis::{AddrSpace, CastMap};

    // Production cast_lookup body (render_map.rs):
    //   let map = self.cast_map?;
    //   map.get(&(parent_type_id, member_byte_offset)).copied()
    struct StubReader<'a> {
        cast_map: Option<&'a CastMap>,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn cast_lookup(&self, parent_type_id: u32, member_byte_offset: u32) -> Option<CastHit> {
            let map = self.cast_map?;
            map.get(&(parent_type_id, member_byte_offset)).copied()
        }
    }

    // Build a CastMap with two entries: one Arena, one Kernel.
    // Parent ids and member offsets are arbitrary u32s — the
    // map's role is opaque key/value storage and the lookup is
    // a plain BTreeMap::get.
    let mut map = CastMap::new();
    map.insert(
        (42, 8),
        CastHit {
            target_type_id: 99,
            addr_space: AddrSpace::Arena,
        },
    );
    map.insert(
        (42, 16),
        CastHit {
            target_type_id: 100,
            addr_space: AddrSpace::Kernel,
        },
    );

    let r = StubReader {
        cast_map: Some(&map),
    };

    // Hit on (42, 8): returns the Arena CastHit.
    let hit_arena = r
        .cast_lookup(42, 8)
        .expect("populated map must return CastHit for present key");
    assert_eq!(
        hit_arena.target_type_id, 99,
        "target_type_id must match the inserted value",
    );
    assert!(
        matches!(hit_arena.addr_space, AddrSpace::Arena),
        "addr_space hint must round-trip through cast_lookup",
    );

    // Hit on (42, 16): returns the Kernel CastHit.
    let hit_kernel = r
        .cast_lookup(42, 16)
        .expect("populated map must return CastHit for present key");
    assert_eq!(hit_kernel.target_type_id, 100);
    assert!(
        matches!(hit_kernel.addr_space, AddrSpace::Kernel),
        "second entry's addr_space must round-trip distinctly from the first",
    );

    // Miss on a non-present key: returns None.
    assert!(
        r.cast_lookup(42, 24).is_none(),
        "key not in map must produce None (no fallback to nearby offsets)",
    );
    assert!(
        r.cast_lookup(99, 8).is_none(),
        "different parent_type_id must produce None even with same offset",
    );
}

/// `cast_lookup` with `cast_map = None` returns `None` for every
/// query. The `?` operator on the Option short-circuits before the
/// BTreeMap lookup. Production code path: when the dump pass runs
/// without a cast analysis (no scheduler binary supplied), every
/// `u64` field renders as a plain counter — no typed-pointer
/// promotion fires.
#[test]
fn accessor_mem_reader_cast_lookup_with_none_map() {
    use super::super::cast_analysis::CastMap;

    struct StubReader<'a> {
        cast_map: Option<&'a CastMap>,
    }
    impl super::super::btf_render::MemReader for StubReader<'_> {
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
        fn cast_lookup(
            &self,
            parent_type_id: u32,
            member_byte_offset: u32,
        ) -> Option<super::super::btf_render::CastHit> {
            let map = self.cast_map?;
            map.get(&(parent_type_id, member_byte_offset)).copied()
        }
    }

    let r = StubReader { cast_map: None };

    // Every query returns None — the `?` on `self.cast_map` fires
    // before any map lookup happens.
    assert!(r.cast_lookup(0, 0).is_none());
    assert!(r.cast_lookup(42, 8).is_none());
    assert!(r.cast_lookup(u32::MAX, u32::MAX).is_none());
}

// -- AccessorMemReader resolve_arena_type -------------------------
//
// `AccessorMemReader::resolve_arena_type` (render_map.rs) gates on
// `is_arena_addr` (snapshot's [user_vm_start, user_vm_start + 4 GiB)),
// masks the chased address with `0xFFFF_FFFF`, then runs a range
// lookup against the per-pass [`ArenaTypeIndex`] keyed on
// slot-start to find the slot containing the address.
// `AccessorMemReader` itself is private to render_map.rs, but the
// gate / range / dispatch logic lives in the free helper
// `resolve_arena_type_in_index`. The tests below use a stand-in
// `ResolveArenaTypeStub` whose `resolve_arena_type` impl delegates
// directly to that helper — so the tests exercise the production
// path without duplicating its body.
//
// Hit shape: `Some(ArenaResolveHit { target_type_id, header_skip })`.
// `header_skip == header_size` for slot-start chases, `0` for
// payload-start chases, and the entry returns `None` for any
// other in-slot offset (or out-of-range address).

/// Stand-in for `AccessorMemReader::resolve_arena_type` shared by
/// every test in this section. The trait method delegates to the
/// free helper [`super::render_map::resolve_arena_type_in_index`]
/// — the gate/range/dispatch logic the production
/// `AccessorMemReader::resolve_arena_type` reaches via the
/// outer [`super::render_map::resolve_arena_type_with_static_fallback`]
/// wrapper. With no scx_static index seeded the wrapper degrades
/// to exactly [`super::render_map::resolve_arena_type_in_index`]'s
/// behaviour, so this stub still exercises the production
/// per-instance allocator path byte for byte. Dedicated tests in
/// the `resolve_arena_type_with_static_fallback` section below
/// cover the wrapper's scx_static fall-through.
///
/// The stub itself only carries the two borrows the helper needs
/// (`arena_snapshot`, `arena_type_index`); `is_arena_addr` is also
/// implemented for parity with the production reader's surface,
/// though the tests below only assert against `resolve_arena_type`.
struct ResolveArenaTypeStub<'a> {
    arena_snapshot: Option<&'a super::super::arena::ArenaSnapshot>,
    arena_type_index: Option<&'a super::render_map::ArenaTypeIndex>,
}

impl super::super::btf_render::MemReader for ResolveArenaTypeStub<'_> {
    fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
        None
    }
    fn is_arena_addr(&self, addr: u64) -> bool {
        super::render_map::is_arena_addr_in_snapshot(self.arena_snapshot, addr)
    }
    fn resolve_arena_type(&self, addr: u64) -> Option<super::super::btf_render::ArenaResolveHit> {
        super::render_map::resolve_arena_type_in_index(
            self.arena_snapshot,
            self.arena_type_index,
            addr,
        )
    }
}

/// Slot-start chase: a chased address that equals the slot's start
/// resolves with `header_skip = header_size`. The renderer reads
/// `header_skip + btf_size` bytes from the chased address and slices
/// off the header before rendering the payload — covers the
/// `scx_task_map_val.data` shape that prompted the range-based
/// rewrite.
#[test]
fn accessor_mem_reader_resolve_arena_type_slot_start_returns_header_skip() {
    use super::super::arena::ArenaSnapshot;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 24, // 8-byte header + 16-byte payload
            header_size: 8,
            target_type_id: 7,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Slot-start chase: full 64-bit address whose low 32 bits
    // match the slot start. Returns the payload type id paired
    // with `header_skip = 8`.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1000),
        Some(super::super::btf_render::ArenaResolveHit {
            target_type_id: 7,
            header_skip: 8,
        }),
        "slot-start address must resolve with header_skip = header_size",
    );
}

/// Payload-start chase: a chased address that lands at
/// `slot_start + header_size` resolves with `header_skip = 0`.
/// The renderer reads `btf_size` bytes from the chased address
/// directly — the historical case (`scx_task_data(p)` return cached
/// in `cached_taskc_raw`) keeps working under the new range index.
#[test]
fn accessor_mem_reader_resolve_arena_type_payload_start_returns_zero_skip() {
    use super::super::arena::ArenaSnapshot;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 24,
            header_size: 8,
            target_type_id: 7,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Payload-start chase: address = slot_start + header_size.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1008),
        Some(super::super::btf_render::ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        }),
        "payload-start address must resolve with header_skip = 0",
    );
}

/// Mid-header / mid-payload addresses fall inside the slot range but
/// at offsets the bridge cannot route into a payload render.
/// Pinning the None return so a future "render mid-struct" extension
/// is a deliberate change of behaviour, not an accidental fall-through.
#[test]
fn accessor_mem_reader_resolve_arena_type_interior_returns_none() {
    use super::super::arena::ArenaSnapshot;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 24,
            header_size: 8,
            target_type_id: 7,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Mid-header (offset 4 < header_size 8): no payload render.
    assert!(
        r.resolve_arena_type(0x10_0000_1004).is_none(),
        "mid-header offset must not resolve",
    );
    // Mid-payload (offset 12, header_size 8 → payload offset 4):
    // bridge does not render mid-struct today.
    assert!(
        r.resolve_arena_type(0x10_0000_100C).is_none(),
        "mid-payload offset must not resolve",
    );
}

/// Range search across multiple seeded slots picks the slot whose
/// `[slot_start, slot_start + elem_size)` range contains the
/// chased address. Pins the `BTreeMap::range(..=key).next_back()`
/// step against a regression that might fall through to an
/// unrelated entry whose key collides on the low 32 bits.
#[test]
fn accessor_mem_reader_resolve_arena_type_range_picks_correct_slot() {
    use super::super::arena::ArenaSnapshot;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    // Two non-overlapping slots, distinct payload type ids.
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 7,
        },
    );
    index.insert(
        0x0000_2000,
        ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 11,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    use super::super::btf_render::ArenaResolveHit;

    // First slot — slot-start of slot 0x1000.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1000),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 8,
        }),
    );
    // First slot — payload-start of slot 0x1000.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1008),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        }),
    );
    // Second slot — slot-start of slot 0x2000.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_2000),
        Some(ArenaResolveHit {
            target_type_id: 11,
            header_skip: 8,
        }),
    );
    // Second slot — payload-start of slot 0x2000.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_2008),
        Some(ArenaResolveHit {
            target_type_id: 11,
            header_skip: 0,
        }),
    );
    // Address between the two slots (past slot 1's end, before
    // slot 2's start): no slot contains it.
    assert!(
        r.resolve_arena_type(0x10_0000_1800).is_none(),
        "address between known slots must not resolve",
    );
    // Address past every seeded slot's end.
    assert!(
        r.resolve_arena_type(0x10_0000_3000).is_none(),
        "address past every known slot must not resolve",
    );
}

/// `resolve_arena_type` returns `None` for an address that lies
/// OUTSIDE the arena window, even when the index has a seeded entry
/// whose low-32 range covers the address's low 32 bits. The
/// `is_arena_addr` gate fires first; without it a stale index entry
/// could surface for any 64-bit value whose low 32 bits land in a
/// captured slot's range by happenstance.
#[test]
fn accessor_mem_reader_resolve_arena_type_rejects_out_of_window() {
    use super::super::arena::ArenaSnapshot;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 7,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Below the window: low-32 maps inside the seeded slot but
    // is_arena_addr rejects.
    assert!(
        r.resolve_arena_type(0x0F_0000_1008).is_none(),
        "below-window address must NOT resolve regardless of low-32 collision",
    );
    // Above the window: same gate.
    assert!(
        r.resolve_arena_type(0x12_0000_1008).is_none(),
        "above-window address must NOT resolve regardless of low-32 collision",
    );
}

/// `resolve_arena_type` returns `None` when the index is absent
/// (`arena_type_index = None`): the `?` short-circuit fires before
/// the gate. Production path: a scheduler that does not link
/// sdt_alloc leaves the index empty; the renderer falls back to
/// the trait default's "no bridge" behaviour.
#[test]
fn accessor_mem_reader_resolve_arena_type_none_index_short_circuits() {
    use super::super::arena::ArenaSnapshot;

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: None,
    };

    // Even an in-window address returns None when the index is
    // absent — the `?` operator on `self.arena_type_index` fires
    // before the gate runs.
    assert!(
        r.resolve_arena_type(0x10_0000_1008).is_none(),
        "None index must short-circuit before is_arena_addr gate",
    );
}

/// Exact slot-end boundary: the address immediately following the
/// slot's last byte (`slot_start + elem_size`) is OUT of range and
/// must not resolve. Pins the `<` (not `<=`) bound check that keeps
/// adjacent-slot lookups from spuriously hitting the prior slot.
#[test]
fn accessor_mem_reader_resolve_arena_type_slot_end_boundary_excluded() {
    use super::super::arena::ArenaSnapshot;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    // One slot at 0x1000 with elem_size=16 → range
    // [0x1000, 0x1010). 0x100F is the last byte; 0x1010 is the
    // first byte of the next (uninstalled) slot.
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 7,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Last byte of the slot (offset = 15 = elem_size - 1): inside
    // the range; falls to the mid-payload branch and returns None
    // because the bridge does not render mid-struct.
    assert!(
        r.resolve_arena_type(0x10_0000_100F).is_none(),
        "last byte of slot must not resolve (mid-payload offset)",
    );
    // Exactly slot_start + elem_size: OUT of range. The `<`
    // comparison rejects.
    assert!(
        r.resolve_arena_type(0x10_0000_1010).is_none(),
        "slot_start + elem_size must not resolve (boundary excluded)",
    );
}

/// Adjacent slots with no gap (`slot_a_end == slot_b_start`): the
/// range lookup picks each slot for its own range; the exact
/// boundary belongs to the second slot, not the first. Pins the
/// behaviour of `BTreeMap::range(..=key).next_back()` against an
/// off-by-one regression where the prior slot might "win" on the
/// next slot's first byte.
#[test]
fn accessor_mem_reader_resolve_arena_type_adjacent_slots_picked_correctly() {
    use super::super::arena::ArenaSnapshot;
    use super::super::btf_render::ArenaResolveHit;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    // Slot A at 0x1000, elem_size=16 → range [0x1000, 0x1010).
    index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 7,
        },
    );
    // Slot B at 0x1010, elem_size=16 → range [0x1010, 0x1020).
    // Adjacent, no gap.
    index.insert(
        0x0000_1010,
        ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 11,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Slot A start (offset 0): payload type 7 with
    // `header_skip = 8`.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1000),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 8,
        }),
    );
    // Slot A payload-start (offset 8): payload type 7 with
    // `header_skip = 0`.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1008),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 0,
        }),
    );
    // Slot B start (offset 0 within B; this is `slot_a_end`):
    // payload type 11 — slot B wins, NOT slot A.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1010),
        Some(ArenaResolveHit {
            target_type_id: 11,
            header_skip: 8,
        }),
    );
    // Slot B payload-start.
    assert_eq!(
        r.resolve_arena_type(0x10_0000_1018),
        Some(ArenaResolveHit {
            target_type_id: 11,
            header_skip: 0,
        }),
    );
}

/// High-edge slot near `u32::MAX`: a slot whose `slot_start +
/// elem_size` would overflow `u32` must still resolve correctly
/// for in-range addresses. Pins the `u64`-widened bound that
/// replaced the old `u32::checked_add` (which silently dropped
/// the last few KiB of a 4 GiB arena window).
#[test]
fn accessor_mem_reader_resolve_arena_type_high_edge_slot_resolves() {
    use super::super::arena::ArenaSnapshot;
    use super::super::btf_render::ArenaResolveHit;
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut index = ArenaTypeIndex::new();
    // Slot near the top of the 4 GiB window. slot_start +
    // elem_size = 0xFFFF_F000 + 4096 = 0x1_0000_0000, which
    // overflows u32. The widened bound resolves both endpoints
    // correctly.
    index.insert(
        0xFFFF_F000,
        ArenaSlotInfo {
            elem_size: 4096,
            header_size: 8,
            target_type_id: 7,
        },
    );

    let r = ResolveArenaTypeStub {
        arena_snapshot: Some(&snap),
        arena_type_index: Some(&index),
    };

    // Slot start: full 64-bit address re-attaches the high 32 bits
    // from `user_vm_start` (0x10_0000_0000) onto the windowed slot
    // start (0xFFFF_F000) → 0x10_FFFF_F000.
    assert_eq!(
        r.resolve_arena_type(0x10_FFFF_F000),
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 8,
        }),
    );
    // Last byte inside the slot: low-32 = 0xFFFF_FFFF (=
    // slot_start + elem_size - 1). Mid-payload offset → None
    // (the bridge does not render mid-struct). The critical
    // assertion is that the bound check did NOT reject this
    // address as "outside the slot" — the wide arithmetic kept
    // the comparison meaningful.
    assert!(
        r.resolve_arena_type(0x10_FFFF_FFFF).is_none(),
        "last byte must reach the mid-payload branch (None), \
         not be rejected as out-of-range by overflow",
    );
}

// -- resolve_arena_type_with_static_fallback ---------------------
//
// The fall-through helper composes the per-instance sdt_alloc index
// (typed) with the scx_static range index (membership-only). When
// the sdt_alloc index resolves the chase, the result is returned
// verbatim; when sdt_alloc misses but the address falls in a live
// scx_static region, the helper deliberately returns `None` (the
// "no invalid data made" contract — the bridge cannot recover a
// per-allocation type from scx_static memory).

/// sdt_alloc hit path: the helper returns the same `ArenaResolveHit`
/// that `resolve_arena_type_in_index` would. Pinning the
/// "fall-through doesn't reorder behaviour" invariant — adding the
/// scx_static fall-through must not change the output for any
/// address the inner index resolves.
#[test]
fn resolve_arena_type_with_static_fallback_returns_sdt_alloc_hit() {
    use super::super::arena::ArenaSnapshot;
    use super::super::btf_render::ArenaResolveHit;
    use super::super::scx_static_alloc::{ScxStaticRangeIndex, ScxStaticSnapshot};
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let mut sdt_index = ArenaTypeIndex::new();
    sdt_index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 24,
            header_size: 8,
            target_type_id: 7,
        },
    );
    // scx_static index intentionally empty for this test; the
    // sdt_alloc hit must fire first regardless.
    let static_index: ScxStaticRangeIndex =
        super::super::scx_static_alloc::build_scx_static_range_index(&ScxStaticSnapshot::default());

    let hit = super::render_map::resolve_arena_type_with_static_fallback(
        Some(&snap),
        Some(&sdt_index),
        Some(&static_index),
        0x10_0000_1000, // slot start
    );
    assert_eq!(
        hit,
        Some(ArenaResolveHit {
            target_type_id: 7,
            header_skip: 8,
        }),
        "sdt_alloc index hit must propagate through fallback helper unchanged",
    );
}

/// scx_static-only hit path: when sdt_alloc misses AND the address
/// falls inside a live `scx_static` range, the helper returns
/// `None` deliberately — the bridge has no per-allocation type to
/// recover. Pinning the fail-closed contract.
#[test]
fn resolve_arena_type_with_static_fallback_scx_static_hit_returns_none() {
    use super::super::arena::ArenaSnapshot;
    use super::super::scx_static_alloc::{ScxStaticRange, ScxStaticSnapshot};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    // No sdt_alloc index → every address misses sdt_alloc.
    let scx_static_snap = ScxStaticSnapshot {
        ranges: vec![ScxStaticRange {
            instance_name: "scx_static".into(),
            start_low32: 0x2000,
            size: 4096,
            capacity: 8192,
        }],
        skipped: 0,
    };
    let static_index =
        super::super::scx_static_alloc::build_scx_static_range_index(&scx_static_snap);

    // Address inside scx_static range. sdt_alloc misses; scx_static
    // hits; helper returns None.
    let hit = super::render_map::resolve_arena_type_with_static_fallback(
        Some(&snap),
        None,
        Some(&static_index),
        0x10_0000_2010,
    );
    assert!(
        hit.is_none(),
        "scx_static-only hit must return None — bridge cannot recover \
         per-allocation type without per-call-site hook from cast analysis",
    );
}

/// Out-of-window address with both indexes seeded → None.
/// `is_arena_addr_in_snapshot` gates the sdt_alloc lookup, and
/// the scx_static fall-through also gates on the same window. An
/// address whose low-32 lands in either index but whose full
/// 64-bit value lives outside the arena window must NOT spuriously
/// hit either path.
#[test]
fn resolve_arena_type_with_static_fallback_out_of_window_returns_none() {
    use super::super::arena::ArenaSnapshot;
    use super::super::scx_static_alloc::{ScxStaticRange, ScxStaticSnapshot};
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex};

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    // Both indexes have entries.
    let mut sdt_index = ArenaTypeIndex::new();
    sdt_index.insert(
        0x0000_1000,
        ArenaSlotInfo {
            elem_size: 24,
            header_size: 8,
            target_type_id: 7,
        },
    );
    let scx_static_snap = ScxStaticSnapshot {
        ranges: vec![ScxStaticRange {
            instance_name: "scx_static".into(),
            start_low32: 0x2000,
            size: 4096,
            capacity: 8192,
        }],
        skipped: 0,
    };
    let static_index =
        super::super::scx_static_alloc::build_scx_static_range_index(&scx_static_snap);

    // Address has correct low-32 to hit sdt_alloc but high bits
    // are outside the window.
    let hit = super::render_map::resolve_arena_type_with_static_fallback(
        Some(&snap),
        Some(&sdt_index),
        Some(&static_index),
        0x05_0000_1000,
    );
    assert!(
        hit.is_none(),
        "out-of-window address must NOT hit sdt_alloc even with low-32 collision",
    );

    // Same idea, low-32 hits scx_static.
    let hit = super::render_map::resolve_arena_type_with_static_fallback(
        Some(&snap),
        Some(&sdt_index),
        Some(&static_index),
        0x05_0000_2010,
    );
    assert!(
        hit.is_none(),
        "out-of-window address must NOT hit scx_static even with low-32 collision",
    );
}

/// Both indexes None → behaves exactly like the trait default
/// (return None). Pinning that the helper short-circuits cleanly
/// when neither index is wired.
#[test]
fn resolve_arena_type_with_static_fallback_both_none_returns_none() {
    use super::super::arena::ArenaSnapshot;

    let snap = ArenaSnapshot {
        user_vm_start: 0x10_0000_0000,
        ..ArenaSnapshot::default()
    };
    let hit = super::render_map::resolve_arena_type_with_static_fallback(
        Some(&snap),
        None,
        None,
        0x10_0000_1000,
    );
    assert!(
        hit.is_none(),
        "both-None must return None — same as trait default",
    );
}

// -- ArenaSnapshot.user_vm_start serde + Display ------------------
//
// The new field is preserved across serde encode/decode and
// present even when the snapshot bailed before reading
// `kern_vm_kva` (preserves the arena anchor for is_arena_addr
// consumers).

#[test]
fn arena_snapshot_user_vm_start_round_trips() {
    use super::super::arena::ArenaSnapshot;
    let snap = ArenaSnapshot {
        user_vm_start: 0x1234_5678_0000_0000,
        ..ArenaSnapshot::default()
    };
    let json = serde_json::to_string(&snap).expect("serialize");
    assert!(
        json.contains("\"user_vm_start\":1311768464867721216"),
        "user_vm_start in JSON: {json}",
    );
    let parsed: ArenaSnapshot = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.user_vm_start, 0x1234_5678_0000_0000);
}

#[test]
fn arena_snapshot_default_user_vm_start_is_zero() {
    use super::super::arena::ArenaSnapshot;
    let snap = ArenaSnapshot::default();
    assert_eq!(
        snap.user_vm_start, 0,
        "default snapshot's user_vm_start is 0 (no anchor)",
    );
}

// -- render_map refactor regression tests (task #42) --------------
//
// The refactor extracted `render_value_or_hex` and
// `render_key_optional` from 6+ duplicated match arms in
// `render_map`, and pushed wildcard explanation strings into the
// `MAP_TYPE_EXPLANATIONS` lookup table. Pin the dispatch shape of
// both helpers and the table-vs-explicit-arm coverage so a
// future refactor that drops a discriminant from the table or
// re-adds the (None, _) case to the BTF render path trips a test
// before the dump loses fidelity.

/// Empty MemReader for tests that don't need pointer-deref. The
/// helpers under test only forward the `&dyn MemReader` to
/// `render_value_with_mem`; for the (None btf) and (type_id=0)
/// paths the MemReader is never consulted.
struct EmptyReader;
impl super::super::btf_render::MemReader for EmptyReader {
    fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
        None
    }
}

/// `render_value_or_hex` falls back to `RenderedValue::Bytes`
/// when no Btf is provided. Hex output matches `hex_dump` so a
/// consumer scanning the failure dump always sees raw bytes for
/// maps whose value type id couldn't be resolved.
#[test]
fn render_value_or_hex_falls_back_to_hex_when_btf_none() {
    let bytes = [0x12u8, 0x34, 0x56];
    let reader = EmptyReader;
    let rendered = render_value_or_hex(None, 0, &bytes, &reader);
    match rendered {
        RenderedValue::Bytes { hex } => {
            assert_eq!(hex, "12 34 56", "hex must match hex_dump output");
        }
        other => panic!("expected Bytes, got {other:?}"),
    }
}

/// `render_value_or_hex` also falls back to `Bytes` when a Btf
/// IS provided but the type id is zero (the kernel libbpf
/// signal for "no BTF type recorded for this slot"). The match
/// arm guard `id != 0` enforces this.
#[test]
fn render_value_or_hex_falls_back_to_hex_when_type_id_zero() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    let bytes = [0xABu8, 0xCD];
    let reader = EmptyReader;
    let rendered = render_value_or_hex(Some(&btf), 0, &bytes, &reader);
    match rendered {
        RenderedValue::Bytes { hex } => {
            assert_eq!(hex, "ab cd", "type_id=0 must surface hex even with btf");
        }
        other => panic!("expected Bytes, got {other:?}"),
    }
}

/// `render_value_or_hex` produces a typed render when both Btf
/// and a non-zero type id are available. Uses the kernel BTF's
/// `int` type, which the renderer decodes as `RenderedValue::Int`.
#[test]
fn render_value_or_hex_renders_via_btf_when_present() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("int") else {
        crate::report::test_skip("BTF missing 'int' type");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'int' to empty id list");
        return;
    };
    // Little-endian 0x42 as a 4-byte int.
    let bytes = 0x42i32.to_le_bytes();
    let reader = EmptyReader;
    let rendered = render_value_or_hex(Some(&btf), id, &bytes, &reader);
    match rendered {
        RenderedValue::Int { bits: 32, value } => {
            assert_eq!(value, 0x42, "BTF render must surface the decoded int value");
        }
        other => panic!("expected Int{{bits:32}}, got {other:?}"),
    }
}

/// `render_key_optional` returns None when no Btf is provided.
/// The hash-map key path keeps `key_hex` regardless and only
/// surfaces a typed render when both Btf and a non-zero type id
/// allow it, so the None-btf branch must NOT silently fall back
/// to `Bytes` like its value-side sibling.
#[test]
fn render_key_optional_returns_none_when_btf_none() {
    let bytes = [0x07u8, 0x00, 0x00, 0x00];
    let reader = EmptyReader;
    let rendered = render_key_optional(None, 0, &bytes, &reader);
    assert!(
        rendered.is_none(),
        "None btf must surface as None: {rendered:?}"
    );
}

/// `render_key_optional` returns None even with a Btf when the
/// type id is zero. Exercises the `id != 0` guard symmetric with
/// `render_value_or_hex_falls_back_to_hex_when_type_id_zero`.
#[test]
fn render_key_optional_returns_none_when_type_id_zero() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    let bytes = [0x07u8, 0x00, 0x00, 0x00];
    let reader = EmptyReader;
    let rendered = render_key_optional(Some(&btf), 0, &bytes, &reader);
    assert!(
        rendered.is_none(),
        "type_id=0 must surface as None even with btf: {rendered:?}",
    );
}

/// `render_key_optional` returns Some(rendered) when both Btf
/// and a non-zero type id are available — the only path that
/// surfaces a typed key.
#[test]
fn render_key_optional_returns_some_via_btf() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("int") else {
        crate::report::test_skip("BTF missing 'int' type");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("BTF resolved 'int' to empty id list");
        return;
    };
    let bytes = 0x99i32.to_le_bytes();
    let reader = EmptyReader;
    let rendered = render_key_optional(Some(&btf), id, &bytes, &reader);
    match rendered {
        Some(RenderedValue::Int { bits: 32, value }) => {
            assert_eq!(value, 0x99, "must surface the decoded int value");
        }
        other => panic!("expected Some(Int{{bits:32}}), got {other:?}"),
    }
}

/// `find_sdt_data_field_offset` returns `None` for `value_type_id == 0`
/// without consulting BTF — pins the explicit early-return so a future
/// caller passing the kernel-libbpf "no BTF" sentinel doesn't trip a
/// spurious BTF probe.
#[test]
fn find_sdt_data_field_offset_zero_type_id_short_circuits() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    assert_eq!(
        super::render_map::find_sdt_data_field_offset(&btf, 0),
        None,
        "type_id=0 must short-circuit to None",
    );
}

/// `find_sdt_data_field_offset` returns `None` for a struct that has no
/// `struct sdt_data __arena *` member — vmlinux's `task_struct` is
/// guaranteed to predate scx and therefore can't carry a member with
/// that pointee. A non-None return on `task_struct` would mean the
/// helper is matching on something other than the pointee struct name.
#[test]
fn find_sdt_data_field_offset_none_for_unrelated_struct() {
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    let Ok(ids) = btf.resolve_ids_by_name("task_struct") else {
        crate::report::test_skip("vmlinux BTF missing 'task_struct'");
        return;
    };
    let Some(&id) = ids.first() else {
        crate::report::test_skip("'task_struct' resolved to empty id list");
        return;
    };
    assert_eq!(
        super::render_map::find_sdt_data_field_offset(&btf, id),
        None,
        "task_struct must not match the sdt_data pointee predicate",
    );
}

/// `chase_sdt_data_payload` returns `None` whenever any of its
/// prerequisite inputs is missing: no BTF, no field offset, no
/// allocator metadata, zero `target_type_id`,
/// `elem_size <= header_size`, or `kern_vm_start == 0`. Each
/// early-return is one of the gates the surface render relies on
/// to NOT spuriously decorate non-arena entries.
#[test]
fn chase_sdt_data_payload_returns_none_on_missing_prereqs() {
    use super::super::btf_render::MemReader;
    use super::render_map::{SdtAllocMeta, chase_sdt_data_payload};
    struct StubReader;
    impl MemReader for StubReader {
        // Returns a zero buffer for every page-table-walked KVA
        // read so the gates we're testing get exercised regardless
        // of the synthetic kva. The gates this test exercises all
        // short-circuit BEFORE the read_kva call, so the contents
        // here are immaterial — the tests pin that the gates fire,
        // not the post-gate render.
        fn read_kva(&self, _: u64, len: usize) -> Option<Vec<u8>> {
            Some(vec![0u8; len])
        }
    }
    let reader = StubReader;
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    // We use any non-zero u32 as a stand-in for a payload type id —
    // the helper short-circuits BEFORE rendering when the elem_size
    // gate fails, so the type id is never dereferenced in those
    // arms.
    let placeholder_type_id: u32 = 1;
    // Every `Some` SdtAllocMeta in this test uses a non-zero
    // `kern_vm_start` so the kern_vm_start gate doesn't pre-empt
    // the gate under test; the dedicated kern_vm_start=0 case
    // appears below.
    let valid_meta = SdtAllocMeta {
        allocator_name: "scx_test_allocator".into(),
        elem_size: 32,
        header_size: 8,
        target_type_id: placeholder_type_id,
        kern_vm_start: 0xFFFF_8000_0000_0000,
    };
    // 24 bytes of value bytes: tid (8) + tptr (8) + data (8 = pointer
    // 0x100000000 in LE).
    let mut value_bytes = vec![0u8; 24];
    value_bytes[16..24].copy_from_slice(&0x1_0000_0000u64.to_le_bytes());

    // No BTF.
    assert!(
        chase_sdt_data_payload(None, Some(16), Some(&valid_meta), &value_bytes, &reader).is_none(),
        "missing btf must yield None",
    );
    // No field offset.
    assert!(
        chase_sdt_data_payload(Some(&btf), None, Some(&valid_meta), &value_bytes, &reader)
            .is_none(),
        "missing field offset must yield None",
    );
    // No allocator metadata.
    assert!(
        chase_sdt_data_payload(Some(&btf), Some(16), None, &value_bytes, &reader).is_none(),
        "missing allocator metadata must yield None",
    );
    // Zero target_type_id (allocator pre-pass returned 0 for
    // ambiguous / no-candidate paths).
    let zero_payload = SdtAllocMeta {
        target_type_id: 0,
        ..valid_meta.clone()
    };
    assert!(
        chase_sdt_data_payload(
            Some(&btf),
            Some(16),
            Some(&zero_payload),
            &value_bytes,
            &reader,
        )
        .is_none(),
        "target_type_id=0 must yield None",
    );
    // elem_size <= header_size: corrupt allocator metadata; payload
    // would slice empty.
    let small_elem = SdtAllocMeta {
        elem_size: 8,
        ..valid_meta.clone()
    };
    assert!(
        chase_sdt_data_payload(
            Some(&btf),
            Some(16),
            Some(&small_elem),
            &value_bytes,
            &reader,
        )
        .is_none(),
        "elem_size <= header_size must yield None",
    );
    // kern_vm_start == 0: arena pre-pass found no kernel-side
    // anchor — the chase has no way to compute a KVA from the
    // user-side pointer.
    let no_kern_vm = SdtAllocMeta {
        kern_vm_start: 0,
        ..valid_meta.clone()
    };
    assert!(
        chase_sdt_data_payload(
            Some(&btf),
            Some(16),
            Some(&no_kern_vm),
            &value_bytes,
            &reader,
        )
        .is_none(),
        "kern_vm_start=0 must yield None",
    );
    // Null arena pointer: scx_task_storage entry created but
    // scx_task_alloc has not populated the data slot yet.
    let mut zero_value_bytes = vec![0u8; 24];
    // explicit zero at offset 16..24 — already zero, but pin
    // intent.
    zero_value_bytes[16..24].copy_from_slice(&0u64.to_le_bytes());
    assert!(
        chase_sdt_data_payload(
            Some(&btf),
            Some(16),
            Some(&valid_meta),
            &zero_value_bytes,
            &reader,
        )
        .is_none(),
        "data_ptr=0 must yield None",
    );
    // Value bytes too short to hold a u64 at the field offset.
    let short_value_bytes = vec![0u8; 20];
    assert!(
        chase_sdt_data_payload(
            Some(&btf),
            Some(16),
            Some(&valid_meta),
            &short_value_bytes,
            &reader,
        )
        .is_none(),
        "value bytes too short for pointer read must yield None",
    );
}

/// `chase_sdt_data_payload` returns `None` when the page-table
/// walker can't resolve the composed KVA. Pages outside the
/// captured guest memory (PA past end-of-DRAM) translate to a
/// failure, matching the `read_kva_bytes_chunked` semantics
/// `mem_reader.read_kva` is built atop.
#[test]
fn chase_sdt_data_payload_yields_none_on_unmapped_kva() {
    use super::super::btf_render::MemReader;
    use super::render_map::{SdtAllocMeta, chase_sdt_data_payload};
    struct UnmappedReader;
    impl MemReader for UnmappedReader {
        // Always returns None — every read_kva fails. Mirrors a
        // page-table walker that has no PTE for the requested KVA
        // (an arena page that's been freed, or a KVA outside the
        // arena window).
        fn read_kva(&self, _: u64, _: usize) -> Option<Vec<u8>> {
            None
        }
    }
    let reader = UnmappedReader;
    let Some(path) = crate::monitor::find_test_vmlinux() else {
        return;
    };
    let Ok(btf) = crate::monitor::btf_offsets::load_btf_from_path(&path) else {
        crate::report::test_skip("could not parse vmlinux BTF");
        return;
    };
    let mut value_bytes = vec![0u8; 24];
    value_bytes[16..24].copy_from_slice(&0x1_0000_0000u64.to_le_bytes());
    let meta = SdtAllocMeta {
        allocator_name: "scx_test_allocator".into(),
        elem_size: 32,
        header_size: 8,
        target_type_id: 1,
        kern_vm_start: 0xFFFF_8000_0000_0000,
    };
    assert!(
        chase_sdt_data_payload(Some(&btf), Some(16), Some(&meta), &value_bytes, &reader,).is_none(),
        "unmapped kva must yield None even with all other prereqs satisfied",
    );
}

/// `FailureDumpEntry::payload` round-trips through serde JSON when
/// populated, and is suppressed by `skip_serializing_if` when None
/// (preserves the wire shape for entries that don't carry a typed
/// payload).
#[test]
fn failure_dump_entry_payload_serde_roundtrip() {
    let entry = FailureDumpEntry {
        key: None,
        key_hex: "00 11 22 33 44 55 66 77".into(),
        value: None,
        value_hex: "AA BB".into(),
        payload: Some(RenderedValue::Uint {
            bits: 64,
            value: 0xDEAD_BEEF,
        }),
    };
    let json = serde_json::to_string(&entry).unwrap();
    assert!(
        json.contains("\"payload\""),
        "populated payload must appear in JSON: {json}",
    );
    let parsed: FailureDumpEntry = serde_json::from_str(&json).unwrap();
    match parsed.payload {
        Some(RenderedValue::Uint { bits: 64, value }) => {
            assert_eq!(value, 0xDEAD_BEEF, "value must round-trip");
        }
        other => panic!("payload didn't round-trip cleanly: {other:?}"),
    }
}

/// Empty payload must NOT appear on the wire — the
/// `skip_serializing_if = "Option::is_none"` predicate keeps the
/// JSON shape unchanged for non-arena map entries (HASH/LRU_HASH and
/// local-storage maps without a discoverable allocator).
#[test]
fn failure_dump_entry_payload_skipped_when_none() {
    let entry = FailureDumpEntry {
        key: None,
        key_hex: "00".into(),
        value: None,
        value_hex: "00".into(),
        payload: None,
    };
    let json = serde_json::to_string(&entry).unwrap();
    assert!(
        !json.contains("\"payload\""),
        "None payload must be skipped: {json}",
    );
}

/// `FailureDumpEntry` Display surfaces the typed payload underneath
/// `value`. Pin both that the payload is rendered AND the relative
/// order — operators read top-to-bottom and the surface struct
/// (key/value) must come before the deref'd payload, matching how
/// a kernel-side debugger would inspect: chase the pointer, then
/// read the dereferenced struct. The format uses
/// `payload <rendered>` (with a space, no colon) so the value's
/// own breadcrumb completes the line.
#[test]
fn failure_dump_entry_display_renders_payload_after_value() {
    let entry = FailureDumpEntry {
        key: Some(RenderedValue::Uint { bits: 64, value: 0 }),
        key_hex: "00".into(),
        value: Some(RenderedValue::Uint {
            bits: 32,
            value: 99,
        }),
        value_hex: "63".into(),
        payload: Some(RenderedValue::Uint {
            bits: 64,
            value: 0xCAFEBABE,
        }),
    };
    let out = format!("{entry}");
    assert!(
        out.contains("\n  .data "),
        "Display must label .data: {out}"
    );
    let value_pos = out.find("value:").expect("value label present");
    let payload_pos = out.find(".data ").expect(".data label present");
    assert!(
        value_pos < payload_pos,
        "Display must order value before .data: {out}",
    );
    assert!(
        out.contains("3405691582"), // 0xCAFEBABE in decimal
        "rendered payload value must appear in Display: {out}",
    );
}

/// Display omits the payload line when payload is None — the
/// existing key/value surface stays unchanged for entries that
/// don't carry a typed payload.
#[test]
fn failure_dump_entry_display_omits_payload_when_none() {
    let entry = FailureDumpEntry {
        key: None,
        key_hex: "ab".into(),
        value: None,
        value_hex: "cd".into(),
        payload: None,
    };
    let out = format!("{entry}");
    assert!(
        !out.contains("payload"),
        "Display must not surface payload when None: {out}",
    );
}

/// `MAP_TYPE_EXPLANATIONS` must carry an entry for every
/// non-walker map type so the wildcard arm produces a precise
/// reason instead of the generic "unknown map_type N" fallback.
///
/// Walker arms (have explicit handling in `render_map`):
///   ARRAY, HASH, LRU_HASH, PERCPU_HASH, LRU_PERCPU_HASH,
///   PERCPU_ARRAY, ARENA, STRUCT_OPS, TASK_STORAGE,
///   INODE_STORAGE, SK_STORAGE, CGRP_STORAGE,
///   RINGBUF, USER_RINGBUF, STACK_TRACE, and the FD-array family
///   (PROG_ARRAY, PERF_EVENT_ARRAY, CGROUP_ARRAY, ARRAY_OF_MAPS,
///   HASH_OF_MAPS, DEVMAP, DEVMAP_HASH, SOCKMAP, SOCKHASH,
///   CPUMAP, XSKMAP, REUSEPORT_SOCKARRAY).
///
/// Non-walker types must each appear in MAP_TYPE_EXPLANATIONS:
///   LPM_TRIE (11), CGROUP_STORAGE (19),
///   PERCPU_CGROUP_STORAGE (21), QUEUE (22), STACK (23),
///   BLOOM_FILTER (30), INSN_ARRAY (34).
#[test]
fn map_type_explanations_covers_every_non_walker_type() {
    let non_walker: &[(u32, &str)] = &[
        (BPF_MAP_TYPE_LPM_TRIE, "LPM_TRIE"),
        (BPF_MAP_TYPE_CGROUP_STORAGE, "CGROUP_STORAGE"),
        (BPF_MAP_TYPE_PERCPU_CGROUP_STORAGE, "PERCPU_CGROUP_STORAGE"),
        (BPF_MAP_TYPE_QUEUE, "QUEUE"),
        (BPF_MAP_TYPE_STACK, "STACK"),
        (BPF_MAP_TYPE_BLOOM_FILTER, "BLOOM_FILTER"),
        (BPF_MAP_TYPE_INSN_ARRAY, "INSN_ARRAY"),
    ];
    for (discriminant, name) in non_walker {
        let found = MAP_TYPE_EXPLANATIONS.iter().any(|(t, _)| t == discriminant);
        assert!(
            found,
            "MAP_TYPE_EXPLANATIONS missing entry for {name} ({discriminant}); \
             wildcard arm would surface the generic 'unknown map_type' fallback",
        );
    }
}

/// Every explanation string in MAP_TYPE_EXPLANATIONS must be a
/// non-empty, actionable reason — no placeholder strings like
/// "not yet supported" / "TODO" / "FIXME" that signal the
/// implementer left work behind. The dump must always tell the
/// operator something concrete about why the map's contents are
/// missing.
#[test]
fn map_type_explanations_strings_are_actionable() {
    for (discriminant, msg) in MAP_TYPE_EXPLANATIONS {
        assert!(
            !msg.is_empty(),
            "explanation for discriminant {discriminant} is empty",
        );
        for placeholder in ["not yet supported", "TODO", "FIXME", "unimplemented"] {
            assert!(
                !msg.contains(placeholder),
                "explanation for discriminant {discriminant} contains placeholder \
                 {placeholder:?}: {msg:?}",
            );
        }
    }
}

// -- serde roundtrip tests ----------------------------------------
//
// Every new dump section (FailureDumpRingbuf, FailureDumpStackTrace,
// FailureDumpFdArray) must round-trip through serde JSON without
// mangling field semantics. Catches drift in `#[serde(default,
// skip_serializing_if = "...")]` annotations: a renamed field, a
// removed default, or a flipped condition would silently corrupt the
// wire format.

/// FailureDumpRingbuf round-trips every field (capacity, consumer_pos,
/// producer_pos, pending_pos, pending_bytes) through JSON intact.
#[test]
fn failure_dump_ringbuf_roundtrip() {
    let original = FailureDumpRingbuf {
        capacity: 0x1_0000,
        consumer_pos: 0x100,
        producer_pos: 0x200,
        pending_pos: 0x180,
        pending_bytes: 0x100,
    };
    let json = serde_json::to_string(&original).expect("ringbuf serialize");
    let restored: FailureDumpRingbuf = serde_json::from_str(&json).expect("ringbuf deserialize");
    assert_eq!(restored.capacity, original.capacity);
    assert_eq!(restored.consumer_pos, original.consumer_pos);
    assert_eq!(restored.producer_pos, original.producer_pos);
    assert_eq!(restored.pending_pos, original.pending_pos);
    assert_eq!(restored.pending_bytes, original.pending_bytes);
}

/// FailureDumpStackTrace empty case: `truncated=false` is skipped on
/// serialize (skip_serializing_if = std::ops::Not::not) and must
/// default-deserialize back to false. `entries=[]` is serialized
/// since FailureDumpStackTrace.entries has no skip annotation.
#[test]
fn failure_dump_stack_trace_empty_roundtrip() {
    let original = FailureDumpStackTrace {
        n_buckets: 0,
        entries: Vec::new(),
        truncated: false,
    };
    let json = serde_json::to_string(&original).expect("stack_trace serialize");
    // truncated=false should NOT appear in the wire format.
    assert!(
        !json.contains("\"truncated\":true") && !json.contains("\"truncated\":false"),
        "skip_serializing_if must elide truncated when false; JSON: {json}",
    );
    let restored: FailureDumpStackTrace =
        serde_json::from_str(&json).expect("stack_trace deserialize");
    assert_eq!(restored.n_buckets, 0);
    assert!(restored.entries.is_empty());
    assert!(!restored.truncated);
}

/// FailureDumpStackTrace populated entries with truncated=true
/// preserves both the entries vector and the truncation flag.
#[test]
fn failure_dump_stack_trace_populated_roundtrip() {
    let original = FailureDumpStackTrace {
        n_buckets: 4,
        entries: vec![
            FailureDumpStackTraceEntry {
                bucket_id: 0,
                nr: 3,
                pcs: vec![
                    0xFFFF_FFFF_8100_0000,
                    0xFFFF_FFFF_8100_0010,
                    0xFFFF_FFFF_8100_0020,
                ],
                data_hex: "00 10 20".into(),
            },
            FailureDumpStackTraceEntry {
                bucket_id: 2,
                nr: 1,
                pcs: vec![0xFFFF_FFFF_8200_0000],
                data_hex: "ff".into(),
            },
        ],
        truncated: true,
    };
    let json = serde_json::to_string(&original).expect("stack_trace serialize");
    assert!(
        json.contains("\"truncated\":true"),
        "truncated=true must appear in JSON: {json}",
    );
    let restored: FailureDumpStackTrace =
        serde_json::from_str(&json).expect("stack_trace deserialize");
    assert_eq!(restored.n_buckets, 4);
    assert_eq!(restored.entries.len(), 2);
    assert_eq!(restored.entries[0].bucket_id, 0);
    assert_eq!(restored.entries[0].nr, 3);
    assert_eq!(restored.entries[0].pcs.len(), 3);
    assert_eq!(restored.entries[0].pcs[0], 0xFFFF_FFFF_8100_0000);
    assert_eq!(restored.entries[0].data_hex, "00 10 20");
    assert_eq!(restored.entries[1].bucket_id, 2);
    assert!(restored.truncated);
}

/// FailureDumpStackTraceEntry build-id mode: empty `pcs` is elided
/// from the wire format (skip_serializing_if = "Vec::is_empty"),
/// `data_hex` is always populated.
#[test]
fn failure_dump_stack_trace_entry_build_id_roundtrip() {
    let original = FailureDumpStackTraceEntry {
        bucket_id: 7,
        nr: 1,
        pcs: Vec::new(),
        data_hex: "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f \
                   10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f"
            .into(),
    };
    let json = serde_json::to_string(&original).expect("entry serialize");
    assert!(
        !json.contains("\"pcs\""),
        "empty pcs must be elided in build-id mode; JSON: {json}",
    );
    let restored: FailureDumpStackTraceEntry =
        serde_json::from_str(&json).expect("entry deserialize");
    assert_eq!(restored.bucket_id, 7);
    assert_eq!(restored.nr, 1);
    assert!(restored.pcs.is_empty());
    assert_eq!(restored.data_hex.len(), 95); // 32 bytes * 3 chars - 1 trailing
}

/// FailureDumpFdArray empty case: `truncated=false` and `indices=[]`
/// both elided. populated/scanned remain since they have no skip
/// annotation.
#[test]
fn failure_dump_fd_array_empty_roundtrip() {
    let original = FailureDumpFdArray {
        populated: 0,
        scanned: 0,
        indices: Vec::new(),
        truncated: false,
        indices_truncated: false,
    };
    let json = serde_json::to_string(&original).expect("fd_array serialize");
    assert!(
        !json.contains("\"truncated\""),
        "truncated=false must be elided; JSON: {json}",
    );
    assert!(
        !json.contains("\"indices_truncated\""),
        "indices_truncated=false must be elided; JSON: {json}",
    );
    let restored: FailureDumpFdArray = serde_json::from_str(&json).expect("fd_array deserialize");
    assert_eq!(restored.populated, 0);
    assert_eq!(restored.scanned, 0);
    assert!(restored.indices.is_empty());
    assert!(!restored.truncated);
}

/// FailureDumpFdArray populated case: indices vector and
/// truncated=true preserved. Defensive check that populated >
/// indices.len() (the truncation-asymmetry case the implementer
/// handles by capping `indices` at MAX_FD_ARRAY_INDICES while
/// continuing to count populated slots) still round-trips coherently.
#[test]
fn failure_dump_fd_array_populated_roundtrip() {
    let original = FailureDumpFdArray {
        populated: 1500,
        scanned: 4096,
        indices: (0..1024).collect(), // capped at MAX_FD_ARRAY_INDICES
        truncated: true,
        indices_truncated: true, // 1500 > 1024
    };
    let json = serde_json::to_string(&original).expect("fd_array serialize");
    assert!(
        json.contains("\"indices_truncated\":true"),
        "indices_truncated=true must be emitted; JSON: {json}",
    );
    let restored: FailureDumpFdArray = serde_json::from_str(&json).expect("fd_array deserialize");
    assert_eq!(restored.populated, 1500);
    assert_eq!(restored.scanned, 4096);
    assert_eq!(restored.indices.len(), 1024);
    assert_eq!(restored.indices[0], 0);
    assert!(
        restored.indices_truncated,
        "indices_truncated must roundtrip"
    );
    assert_eq!(restored.indices[1023], 1023);
    assert!(restored.truncated);
}

/// Defaulted-from-empty deserialization: a stripped-down JSON
/// (only required fields present) deserializes with `Default`
/// values for skipped fields. Validates the
/// `#[serde(default, skip_serializing_if = "...")]` round-trip
/// across the three new dump sections.
#[test]
fn failure_dump_minimal_deserialize_uses_defaults() {
    // Ringbuf has no skip_serializing_if (every field is required),
    // so the minimal form must include all fields.
    let json =
        r#"{"capacity":0,"consumer_pos":0,"producer_pos":0,"pending_pos":0,"pending_bytes":0}"#;
    let rb: FailureDumpRingbuf = serde_json::from_str(json).expect("ringbuf minimal deserialize");
    assert_eq!(rb.capacity, 0);

    // StackTrace minimal: only n_buckets, default-fills the rest.
    let json = r#"{"n_buckets":0,"entries":[]}"#;
    let st: FailureDumpStackTrace =
        serde_json::from_str(json).expect("stack_trace minimal deserialize");
    assert_eq!(st.n_buckets, 0);
    assert!(st.entries.is_empty());
    assert!(!st.truncated, "truncated must default to false");

    // FdArray minimal: populated/scanned required, indices and
    // truncated default.
    let json = r#"{"populated":0,"scanned":0}"#;
    let fa: FailureDumpFdArray = serde_json::from_str(json).expect("fd_array minimal deserialize");
    assert_eq!(fa.populated, 0);
    assert_eq!(fa.scanned, 0);
    assert!(fa.indices.is_empty(), "indices must default to []");
    assert!(!fa.truncated, "truncated must default to false");
}

// -- DualFailureDumpReport Display rendering ----------------------
//
// The post-pass-3 fix in display.rs distinguishes "early snapshot
// present with valid jiffies" from "early snapshot present with
// jiffies bookkeeping not captured" (both early_max_age_jiffies and
// early_threshold_jiffies are 0). Pin the wire-stable header strings
// for each branch so a reformatting regression surfaces immediately
// rather than producing operator-confusing "max_age=0j, threshold=0j"
// output.

/// Both jiffies fields are 0 with an early snapshot present:
/// renders `early=present (jiffies not captured)` instead of the
/// misleading `max_age=0j, threshold=0j`.
#[test]
fn dual_dump_display_zero_jiffies_uses_jiffies_not_captured_branch() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let rendered = format!("{dual}");
    assert!(
        rendered.contains("early=present (jiffies not captured)"),
        "zero-jiffies branch must surface a distinct phrase; got: {rendered}",
    );
    assert!(
        !rendered.contains("max_age=0j"),
        "zero-jiffies header must NOT print max_age=0j; got: {rendered}",
    );
    assert!(
        !rendered.contains("threshold=0j"),
        "zero-jiffies header must NOT print threshold=0j; got: {rendered}",
    );
}

/// Non-zero jiffies preserves the legacy `max_age={N}j, threshold={M}j`
/// format. Pins the format-string layout so a refactor that swaps
/// the field order or drops the `j` suffix surfaces immediately.
#[test]
fn dual_dump_display_nonzero_jiffies_preserves_max_age_format() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 1234,
        early_threshold_jiffies: 5678,
        early_skipped_reason: None,
    };
    let rendered = format!("{dual}");
    assert!(
        rendered.contains("max_age=1234j, threshold=5678j"),
        "non-zero jiffies must preserve the max_age/threshold format; got: {rendered}",
    );
    assert!(
        !rendered.contains("jiffies not captured"),
        "non-zero jiffies must NOT use the not-captured phrase; got: {rendered}",
    );
}

/// One jiffies field zero, the other non-zero — verifies the
/// zero-jiffies branch ONLY fires when BOTH are zero. A single-zero
/// case should still use the legacy format (the bookkeeping was
/// partial but at least one number is meaningful).
#[test]
fn dual_dump_display_one_zero_one_nonzero_uses_legacy_format() {
    // max_age=0, threshold=5: legacy format with max_age=0j.
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 5,
        early_skipped_reason: None,
    };
    let rendered = format!("{dual}");
    assert!(
        rendered.contains("max_age=0j, threshold=5j"),
        "single-zero case must use legacy format; got: {rendered}",
    );
    assert!(
        !rendered.contains("jiffies not captured"),
        "single-zero case must NOT use the not-captured phrase; got: {rendered}",
    );

    // max_age=5, threshold=0: legacy format with threshold=0j.
    let dual2 = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: Some(FailureDumpReport::default()),
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 5,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let rendered2 = format!("{dual2}");
    assert!(
        rendered2.contains("max_age=5j, threshold=0j"),
        "single-zero case must use legacy format; got: {rendered2}",
    );
    assert!(
        !rendered2.contains("jiffies not captured"),
        "single-zero case must NOT use the not-captured phrase; got: {rendered2}",
    );
}

/// `early=absent` branch is independent of the jiffies fields:
/// when `early` is None, neither the legacy nor the
/// jiffies-not-captured phrase appears, even if the jiffies
/// fields are populated (they describe the absent snapshot's
/// trigger metric — surfacing them would be misleading).
#[test]
fn dual_dump_display_early_absent_omits_jiffies_lines() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        // Populated but unused — exercises the assertion below.
        early_max_age_jiffies: 999,
        early_threshold_jiffies: 100,
        early_skipped_reason: None,
    };
    let rendered = format!("{dual}");
    assert!(
        rendered.contains("early=absent"),
        "absent branch must surface 'early=absent'; got: {rendered}",
    );
    assert!(
        !rendered.contains("max_age=999j"),
        "absent branch must NOT surface jiffies values; got: {rendered}",
    );
    assert!(
        !rendered.contains("jiffies not captured"),
        "absent branch must NOT surface the not-captured phrase; got: {rendered}",
    );
}

/// `early_skipped_reason` populated → Display surfaces the structured
/// reason directly in the header AND in the absent-branch body, replacing
/// the legacy "stall fired before half-way threshold, or runnable_at scan
/// setup failed" generic text. Pins the contract for the freeze
/// coordinator's three known reasons (scan prerequisites unavailable,
/// max_age never crossed threshold, scx_tick stall) and any future
/// addition.
#[test]
fn dual_dump_display_early_absent_renders_structured_reason() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: Some("scx_tick stall — no per-task runnable_at data".to_string()),
    };
    let rendered = format!("{dual}");
    assert!(
        rendered.contains("scx_tick stall"),
        "structured reason must appear in absent header; got: {rendered}",
    );
    assert!(
        !rendered.contains("RUST_LOG=ktstr=debug"),
        "RUST_LOG hint must NOT appear when reason is structured; got: {rendered}",
    );
    assert!(
        !rendered.contains("stall fired before half-way threshold"),
        "legacy generic text must NOT appear when reason is structured; got: {rendered}",
    );
}

/// `early_skipped_reason` None on absent path → falls back to the
/// legacy two-cause generic text and the RUST_LOG hint. Preserves
/// rendering for older dump JSONs that predate
/// `early_skipped_reason` (the field deserialises as `None` on
/// missing-key inputs per `#[serde(default)]`).
#[test]
fn dual_dump_display_early_absent_falls_back_when_reason_absent() {
    let dual = DualFailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        early: None,
        late: FailureDumpReport::default(),
        early_max_age_jiffies: 0,
        early_threshold_jiffies: 0,
        early_skipped_reason: None,
    };
    let rendered = format!("{dual}");
    assert!(
        rendered.contains("stall fired before half-way threshold"),
        "legacy generic text must appear when reason is absent; got: {rendered}",
    );
    assert!(
        rendered.contains("RUST_LOG=ktstr=debug"),
        "RUST_LOG hint must appear when reason is absent; got: {rendered}",
    );
}

// -- New per-section Display rendering ----------------------------
//
// Per-CPU CPU-time, per-node NUMA, and the scx_walker section each
// have wire-stable Display headers. Pin the formats so a renaming
// or a count-format swap surfaces in tests rather than as a
// mismatched log scrape.

/// `per_cpu_time` section: one-line summary of the CPU count
/// captured. Format: `per_cpu_time: {N} CPUs captured`.
#[test]
fn failure_dump_display_per_cpu_time_summary() {
    let report = FailureDumpReport {
        per_cpu_time: vec![
            super::PerCpuTimeStats {
                cpu: 0,
                ..super::PerCpuTimeStats::default()
            },
            super::PerCpuTimeStats {
                cpu: 1,
                ..super::PerCpuTimeStats::default()
            },
            super::PerCpuTimeStats {
                cpu: 2,
                ..super::PerCpuTimeStats::default()
            },
        ],
        ..FailureDumpReport::default()
    };
    let rendered = format!("{report}");
    assert!(
        rendered.contains("per_cpu_time: 3 CPUs captured"),
        "per_cpu_time section must surface CPU count; got: {rendered}",
    );
}

/// `per_cpu_time` empty + `per_node_numa` populated: the two
/// sections are independent — populated nodes render even when no
/// CPU rows were captured.
#[test]
fn failure_dump_display_per_node_numa_summary() {
    let report = FailureDumpReport {
        per_node_numa: vec![
            super::PerNodeNumaStats {
                node: 0,
                ..super::PerNodeNumaStats::default()
            },
            super::PerNodeNumaStats {
                node: 1,
                ..super::PerNodeNumaStats::default()
            },
        ],
        ..FailureDumpReport::default()
    };
    let rendered = format!("{report}");
    assert!(
        rendered.contains("per_node_numa: 2 nodes captured"),
        "per_node_numa section must surface node count; got: {rendered}",
    );
    assert!(
        !rendered.contains("per_cpu_time:"),
        "per_cpu_time must be elided when empty; got: {rendered}",
    );
}

/// `per_node_numa_unavailable` reason renders inline when the
/// walker bailed (current `"no NUMA walker"` placeholder until the
/// host-side walker lands).
#[test]
fn failure_dump_display_per_node_numa_unavailable() {
    let report = FailureDumpReport {
        per_node_numa_unavailable: Some("no NUMA walker".into()),
        ..FailureDumpReport::default()
    };
    let rendered = format!("{report}");
    assert!(
        rendered.contains("per_node_numa: <unavailable: no NUMA walker>"),
        "per_node_numa_unavailable must surface the reason inline; got: {rendered}",
    );
}

/// `scx_walker` section: one-line summary `rq_scx={N} dsq={M}
/// sched={captured|absent}`. Pinned because the format reads as a
/// log-scrapable triple a downstream tool can split on.
#[test]
fn failure_dump_display_scx_walker_all_present() {
    use crate::monitor::scx_walker::{DsqState, RqScxState, ScxSchedState};
    let report = FailureDumpReport {
        rq_scx_states: vec![RqScxState::default(); 4],
        dsq_states: vec![DsqState::default(); 2],
        scx_sched_state: Some(ScxSchedState::default()),
        ..FailureDumpReport::default()
    };
    let rendered = format!("{report}");
    assert!(
        rendered.contains("scx_walker: rq_scx=4 dsq=2 sched=captured"),
        "scx_walker present-everywhere must surface counts and 'captured'; got: {rendered}",
    );
}

/// `scx_walker` partial-output: rq_scx populated but no DSQs and no
/// scx_sched scalar. The section still renders (any non-empty
/// triggers the block) and `sched=absent` surfaces explicitly.
#[test]
fn failure_dump_display_scx_walker_partial() {
    use crate::monitor::scx_walker::RqScxState;
    let report = FailureDumpReport {
        rq_scx_states: vec![RqScxState::default()],
        ..FailureDumpReport::default()
    };
    let rendered = format!("{report}");
    assert!(
        rendered.contains("scx_walker: rq_scx=1 dsq=0 sched=absent"),
        "partial scx_walker must show 'sched=absent'; got: {rendered}",
    );
}

/// `scx_walker_unavailable` reason renders inline when the walker
/// could not run (e.g. scx_sched offsets unresolved).
#[test]
fn failure_dump_display_scx_walker_unavailable() {
    let report = FailureDumpReport {
        scx_walker_unavailable: Some("scx_sched offsets unresolved".into()),
        ..FailureDumpReport::default()
    };
    let rendered = format!("{report}");
    assert!(
        rendered.contains("scx_walker: <unavailable: scx_sched offsets unresolved>"),
        "scx_walker_unavailable must surface the reason inline; got: {rendered}",
    );
}

// -- Render-helper unit tests over synthetic guest memory --------
//
// Build a flat host-side buffer that simulates the guest direct
// mapping: every kernel KVA is `pa + page_offset`, and
// `translate_any_kva` in the production read path resolves through
// that mapping unchanged (no page-table walk needed). The synthetic
// `BpfMapOffsets` field offsets are arbitrary — they only need to be
// consistent with how the helpers read from the buffer.

/// Per-test synthetic-memory scene used by the render-helper tests.
/// Owns the guest buffer + the `BpfMapOffsets` block + the
/// `BpfRingbufOffsets` / `BpfStackmapOffsets` instances so a test
/// can borrow `&BpfMapOffsets` for a `GuestMemMapAccessor` without
/// each test re-stitching the offset substructs.
struct RenderScene {
    buf: Vec<u8>,
    page_offset: u64,
    /// `BpfMapOffsets` with `ringbuf_offsets` / `stackmap_offsets` /
    /// `array_value` populated so the synthetic helpers know where
    /// to find each field within the synthetic structs.
    offsets: crate::monitor::btf_offsets::BpfMapOffsets,
}

/// Direct-mapping KVA from a host PA: `kva = pa + page_offset`. The
/// production `translate_any_kva` reverses this with `kva - page_offset`
/// and bounds-checks against `mem.size()`.
fn pa_to_kva(pa: u64, page_offset: u64) -> u64 {
    page_offset.wrapping_add(pa)
}

/// Synthetic ringbuf offsets. The map carries `rb` at offset 0 of
/// `bpf_ringbuf_map` (the synthetic map struct is just the rb
/// pointer); the bpf_ringbuf struct lays out
/// mask/consumer_pos/producer_pos/pending_pos one per cacheline so
/// the production reader exercises the per-field translate.
fn synth_ringbuf_offsets() -> super::super::btf_offsets::BpfRingbufOffsets {
    super::super::btf_offsets::BpfRingbufOffsets {
        rbm_rb: 0,
        rb_mask: 0,
        rb_consumer_pos: 64,
        rb_producer_pos: 128,
        rb_pending_pos: 192,
    }
}

/// Build a `BpfMapOffsets` carrying only the ringbuf substruct.
/// Other walkers/render helpers do not consult these other fields,
/// so they stay zero-valued.
fn synth_ringbuf_map_offsets() -> crate::monitor::btf_offsets::BpfMapOffsets {
    let mut o = crate::monitor::btf_offsets::BpfMapOffsets::EMPTY;
    o.ringbuf_offsets = Some(synth_ringbuf_offsets());
    o
}

/// Lay out a ringbuf scene: bpf_ringbuf_map at PA 0x1000, bpf_ringbuf
/// at PA 0x10_0000. The map's `rb` pointer at offset 0 holds the
/// rb's KVA. The rb's four position fields hold the supplied values
/// at their respective offsets. Returns the scene plus the map's
/// KVA so the test can pass it as `info.map_kva`.
///
/// `rb_kva_override = Some(0)` writes a NULL rb pointer (exercises
/// the rb-NULL error path); `Some(other_kva)` writes an unmapped
/// pointer; `None` writes the rb's real KVA.
fn build_ringbuf_scene(
    mask: u64,
    consumer_pos: u64,
    producer_pos: u64,
    pending_pos: u64,
    rb_kva_override: Option<u64>,
) -> (RenderScene, u64) {
    let rb_offs = synth_ringbuf_offsets();
    let offsets = synth_ringbuf_map_offsets();
    let page_offset = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    let map_pa: u64 = 0x1000;
    let rb_pa: u64 = 0x10_0000;
    let buf_size: usize = (rb_pa as usize) + 0x1000;

    let mut buf = vec![0u8; buf_size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // bpf_ringbuf_map.rb: pointer to bpf_ringbuf.
    let rb_kva = rb_kva_override.unwrap_or_else(|| pa_to_kva(rb_pa, page_offset));
    write_u64(&mut buf, map_pa + rb_offs.rbm_rb as u64, rb_kva);

    // bpf_ringbuf fields.
    write_u64(&mut buf, rb_pa + rb_offs.rb_mask as u64, mask);
    write_u64(
        &mut buf,
        rb_pa + rb_offs.rb_consumer_pos as u64,
        consumer_pos,
    );
    write_u64(
        &mut buf,
        rb_pa + rb_offs.rb_producer_pos as u64,
        producer_pos,
    );
    write_u64(&mut buf, rb_pa + rb_offs.rb_pending_pos as u64, pending_pos);

    let map_kva = pa_to_kva(map_pa, page_offset);
    (
        RenderScene {
            buf,
            page_offset,
            offsets,
        },
        map_kva,
    )
}

/// Build a `BpfMapInfo` for ringbuf tests with the given map_kva.
fn ringbuf_map_info(map_kva: u64) -> super::super::bpf_map::BpfMapInfo {
    let (name_bytes, name_len) = name_from_str("test_ringbuf");
    super::super::bpf_map::BpfMapInfo {
        map_pa: 0,
        map_kva,
        name_bytes,
        name_len,
        map_type: super::super::bpf_map::BPF_MAP_TYPE_RINGBUF,
        map_flags: 0,
        key_size: 0,
        value_size: 0,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    }
}

// -- #55 ringbuf state render tests -------------------------------

/// `render_ringbuf_state` returns the no-offsets error string when
/// `BpfMapOffsets::ringbuf_offsets` is None (kernel built without
/// ringbuf, or BTF stripped).
#[test]
fn render_ringbuf_no_offsets_returns_err() {
    let (scene, map_kva) = build_ringbuf_scene(0xFFFF, 0x100, 0x200, 0x180, None);
    let info = ringbuf_map_info(map_kva);
    // SAFETY: scene.buf is a live local Vec<u8> whose backing
    // storage outlives the GuestMem use.
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };

    let mut offsets = scene.offsets;
    offsets.ringbuf_offsets = None;
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &offsets, 0);
    let result = render_ringbuf_state(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("BTF lacks bpf_ringbuf_map")));
}

/// `render_ringbuf_state` returns the unmapped-map_kva error when
/// `info.map_kva` falls outside the synthetic memory window.
#[test]
fn render_ringbuf_unmapped_map_kva_returns_err() {
    let (scene, _map_kva) = build_ringbuf_scene(0xFFFF, 0x100, 0x200, 0x180, None);
    // Use a KVA that translates outside the buf.
    let bogus_map_kva = scene.page_offset + 0x100_0000;
    let info = ringbuf_map_info(bogus_map_kva);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let result = render_ringbuf_state(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("RINGBUF map_kva unmapped")));
}

/// `render_ringbuf_state` returns the rb-pointer-NULL error when
/// the bpf_ringbuf_map.rb field reads as 0.
#[test]
fn render_ringbuf_null_rb_returns_err() {
    let (scene, map_kva) = build_ringbuf_scene(0xFFFF, 0x100, 0x200, 0x180, Some(0));
    let info = ringbuf_map_info(map_kva);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let result = render_ringbuf_state(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("rb pointer NULL")));
}

/// `render_ringbuf_state` returns the rb-fields-unmapped error when
/// the bpf_ringbuf pointer translates outside the buffer.
#[test]
fn render_ringbuf_unmapped_rb_returns_err() {
    let (scene, map_kva) = build_ringbuf_scene(
        0xFFFF,
        0x100,
        0x200,
        0x180,
        Some(crate::monitor::symbols::DEFAULT_PAGE_OFFSET + 0x100_0000),
    );
    let info = ringbuf_map_info(map_kva);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let result = render_ringbuf_state(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("rb->mask unmapped")));
}

/// Happy path: capacity = mask + 1, pending_bytes = producer - consumer.
#[test]
fn render_ringbuf_basic_capacity_and_pending() {
    // 64 KiB ring (mask = 0xFFFF, capacity = 0x10000), consumer at
    // 0x100, producer at 0x300, pending = 0x200.
    let (scene, map_kva) = build_ringbuf_scene(0xFFFF, 0x100, 0x300, 0x180, None);
    let info = ringbuf_map_info(map_kva);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let rb = render_ringbuf_state(&accessor, &info).expect("happy-path render");
    assert_eq!(rb.capacity, 0x10000);
    assert_eq!(rb.consumer_pos, 0x100);
    assert_eq!(rb.producer_pos, 0x300);
    assert_eq!(rb.pending_pos, 0x180);
    assert_eq!(rb.pending_bytes, 0x200);
}

/// Wraparound: producer < consumer in absolute terms => unsigned
/// wraparound subtraction yields a meaningful pending count.
#[test]
fn render_ringbuf_wraparound_pending_bytes() {
    // consumer beyond producer by 100; wrap subtraction yields
    // u64::MAX - 99.
    let consumer = 200u64;
    let producer = 100u64;
    let (scene, map_kva) = build_ringbuf_scene(0xFFFF, consumer, producer, producer, None);
    let info = ringbuf_map_info(map_kva);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let rb = render_ringbuf_state(&accessor, &info).expect("wraparound render");
    assert_eq!(
        rb.pending_bytes,
        producer.wrapping_sub(consumer),
        "wraparound subtraction must match production semantics",
    );
}

/// `mask = u64::MAX` triggers the "capacity would wrap to 0" guard;
/// the helper bails with an explicit error rather than producing a
/// nonsense capacity = 0.
#[test]
fn render_ringbuf_mask_max_returns_err() {
    let (scene, map_kva) = build_ringbuf_scene(u64::MAX, 0, 0, 0, None);
    let info = ringbuf_map_info(map_kva);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let result = render_ringbuf_state(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("mask = u64::MAX")));
}

// -- #53 stack-trace render tests ---------------------------------

fn synth_stackmap_offsets() -> super::super::btf_offsets::BpfStackmapOffsets {
    super::super::btf_offsets::BpfStackmapOffsets {
        smap_n_buckets: 0,
        smap_buckets: 16,
        smb_nr: 0,
        smb_data: 16,
    }
}

fn synth_stackmap_map_offsets() -> crate::monitor::btf_offsets::BpfMapOffsets {
    let mut o = crate::monitor::btf_offsets::BpfMapOffsets::EMPTY;
    o.stackmap_offsets = Some(synth_stackmap_offsets());
    o
}

/// Build a stack-trace scene. `bucket_pc_lists[i]` carries the PCs
/// for bucket `i` (empty Vec means an empty/null bucket pointer);
/// the layout writes a `bpf_stack_map` with `n_buckets` and
/// `buckets[]` flex array, plus per-bucket `stack_map_bucket`
/// structs containing `nr` + `data[]`.
fn build_stackmap_scene(
    bucket_pc_lists: &[Vec<u64>],
    map_flags: u32,
) -> (RenderScene, super::super::bpf_map::BpfMapInfo) {
    let sm_offs = synth_stackmap_offsets();
    let offsets = synth_stackmap_map_offsets();
    let page_offset = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    // bpf_stack_map at PA 0x1000. Layout:
    //   [0..4) n_buckets (u32)
    //   [16..16 + n*8) buckets[] (each entry is u64 pointer)
    let map_pa: u64 = 0x1000;
    let n_buckets = bucket_pc_lists.len() as u32;
    let map_struct_end = sm_offs.smap_buckets as u64 + (n_buckets as u64) * 8;

    // Each populated bucket gets a stack_map_bucket at fixed strides
    // starting 0x1_0000, each 0x1000 apart.
    let bucket_stride: u64 = 0x1000;
    let buckets_start: u64 = 0x1_0000;
    let buf_size: usize = (buckets_start + bucket_stride * (n_buckets as u64 + 1)) as usize;
    let mut buf = vec![0u8; buf_size];

    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // n_buckets at smap_n_buckets (offset 0).
    write_u32(&mut buf, map_pa + sm_offs.smap_n_buckets as u64, n_buckets);

    // Per-bucket pointers and bucket data.
    let _ = map_struct_end; // silence unused
    for (i, pcs) in bucket_pc_lists.iter().enumerate() {
        let slot_pa = map_pa + sm_offs.smap_buckets as u64 + (i as u64) * 8;
        if pcs.is_empty() {
            write_u64(&mut buf, slot_pa, 0); // null bucket pointer
            continue;
        }
        let bucket_pa = buckets_start + (i as u64) * bucket_stride;
        write_u64(&mut buf, slot_pa, pa_to_kva(bucket_pa, page_offset));
        // stack_map_bucket: nr at smb_nr (0), data[] at smb_data (16).
        write_u32(
            &mut buf,
            bucket_pa + sm_offs.smb_nr as u64,
            pcs.len() as u32,
        );
        for (j, pc) in pcs.iter().enumerate() {
            write_u64(
                &mut buf,
                bucket_pa + sm_offs.smb_data as u64 + (j as u64) * 8,
                *pc,
            );
        }
    }

    let map_kva = pa_to_kva(map_pa, page_offset);
    let (name_bytes, name_len) = name_from_str("test_stack");
    let info = super::super::bpf_map::BpfMapInfo {
        map_pa: 0,
        map_kva,
        name_bytes,
        name_len,
        map_type: super::super::bpf_map::BPF_MAP_TYPE_STACK_TRACE,
        map_flags,
        key_size: 0,
        value_size: 0,
        max_entries: n_buckets,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    (
        RenderScene {
            buf,
            page_offset,
            offsets,
        },
        info,
    )
}

/// `render_stack_traces` returns the BTF-lacks error when
/// stackmap_offsets is None.
#[test]
fn render_stack_traces_no_offsets_returns_err() {
    let (mut scene, info) = build_stackmap_scene(&[vec![]], 0);
    scene.offsets.stackmap_offsets = None;
    // SAFETY: scene.buf is a live local Vec<u8> whose backing
    // storage outlives the GuestMem use.
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let result = render_stack_traces(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("BTF lacks bpf_stack_map")));
}

/// `render_stack_traces` returns the unmapped-map_kva error when
/// info.map_kva translates outside the buffer.
#[test]
fn render_stack_traces_unmapped_map_kva_returns_err() {
    let (scene, _info) = build_stackmap_scene(&[vec![]], 0);
    let mut info = _info;
    info.map_kva = scene.page_offset + 0x100_0000;
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let result = render_stack_traces(&accessor, &info);
    assert!(matches!(result, Err(ref s) if s.contains("STACK_TRACE map_kva unmapped")));
}

/// All-empty buckets: walker returns Ok with `n_buckets` set, an
/// empty entries vec, and `truncated=false`.
#[test]
fn render_stack_traces_empty_returns_no_entries() {
    let (scene, info) = build_stackmap_scene(&[vec![], vec![], vec![], vec![]], 0);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let st = render_stack_traces(&accessor, &info).expect("empty render");
    assert_eq!(st.n_buckets, 4);
    assert!(st.entries.is_empty());
    assert!(!st.truncated);
}

/// Populated buckets surface their PCs in `entries[].pcs`.
#[test]
fn render_stack_traces_populated_pcs() {
    let (scene, info) = build_stackmap_scene(
        &[
            vec![],
            vec![0xFFFF_FFFF_8100_1000, 0xFFFF_FFFF_8100_2000],
            vec![],
            vec![0xFFFF_FFFF_8200_3000],
        ],
        0,
    );
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let st = render_stack_traces(&accessor, &info).expect("populated render");
    assert_eq!(st.n_buckets, 4);
    assert_eq!(st.entries.len(), 2);
    assert_eq!(st.entries[0].bucket_id, 1);
    assert_eq!(st.entries[0].nr, 2);
    assert_eq!(
        st.entries[0].pcs,
        vec![0xFFFF_FFFF_8100_1000, 0xFFFF_FFFF_8100_2000]
    );
    assert_eq!(st.entries[1].bucket_id, 3);
    assert_eq!(st.entries[1].pcs, vec![0xFFFF_FFFF_8200_3000]);
    assert!(!st.truncated);
}

/// Build-id mode: pcs vector stays empty (per-entry shape is
/// bpf_stack_build_id, not u64), data_hex carries raw bytes.
#[test]
fn render_stack_traces_build_id_mode_pcs_empty() {
    const BPF_F_STACK_BUILD_ID: u32 = 1 << 5;
    // One bucket with one "PC slot" (treated as 8 bytes of raw data
    // in build-id mode). The kernel's per-entry size is 32 bytes
    // (bpf_stack_build_id); our synthetic test only needs to verify
    // pcs stays empty regardless of data shape.
    let (scene, info) = build_stackmap_scene(&[vec![0xDEAD_BEEFu64]], BPF_F_STACK_BUILD_ID);
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let st = render_stack_traces(&accessor, &info).expect("build-id render");
    assert_eq!(st.entries.len(), 1);
    assert!(
        st.entries[0].pcs.is_empty(),
        "build-id mode must NOT populate pcs (entry shape is bpf_stack_build_id, not u64)"
    );
}

// -- #53 fd-array render tests ------------------------------------

fn synth_fd_array_offsets() -> crate::monitor::btf_offsets::BpfMapOffsets {
    let mut o = crate::monitor::btf_offsets::BpfMapOffsets::EMPTY;
    // Place ptrs[] at offset 16 within the synthetic bpf_array.
    o.array_value = 16;
    o
}

/// Build a synthetic FD-array scene. `populated_indices` lists the
/// slot indices that should be non-zero. `max_entries` controls the
/// scan upper bound.
fn build_fd_array_scene(
    map_type: u32,
    max_entries: u32,
    populated_indices: &[u32],
) -> (RenderScene, super::super::bpf_map::BpfMapInfo) {
    let offsets = synth_fd_array_offsets();
    let page_offset = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    // bpf_array at PA 0x1000. ptrs[] starts at offset
    // `array_value` (16); each slot is 8 bytes.
    let map_pa: u64 = 0x1000;
    let scan = max_entries.min(super::render_map::MAX_FD_ARRAY_SLOTS);
    let buf_size =
        (map_pa as usize) + (offsets.array_value as usize) + (scan as usize) * 8 + 0x1000;
    let mut buf = vec![0u8; buf_size];
    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    for &idx in populated_indices {
        if idx >= scan {
            continue;
        }
        let slot_pa = map_pa + offsets.array_value as u64 + (idx as u64) * 8;
        // Any non-zero pointer suffices; use the slot index + 1
        // shifted into a recognizable kernel-pointer range.
        write_u64(&mut buf, slot_pa, 0xFFFF_8000_0000_0000 + (idx as u64));
    }

    let map_kva = pa_to_kva(map_pa, page_offset);
    let (name_bytes, name_len) = name_from_str("test_fd_array");
    let info = super::super::bpf_map::BpfMapInfo {
        map_pa: 0,
        map_kva,
        name_bytes,
        name_len,
        map_type,
        map_flags: 0,
        key_size: 0,
        value_size: 0,
        max_entries,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    (
        RenderScene {
            buf,
            page_offset,
            offsets,
        },
        info,
    )
}

/// PROG_ARRAY with three populated slots: walker reports
/// populated=3, indices=[indexes].
#[test]
fn render_fd_array_populated_indices() {
    let (scene, info) = build_fd_array_scene(
        super::super::bpf_map::BPF_MAP_TYPE_PROG_ARRAY,
        16,
        &[0, 5, 10],
    );
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let fa = render_fd_array_slots(&accessor, &info);
    assert_eq!(fa.populated, 3);
    assert_eq!(fa.scanned, 16);
    assert_eq!(fa.indices, vec![0, 5, 10]);
    assert!(!fa.truncated);
    assert!(
        !fa.indices_truncated,
        "populated == indices.len() must NOT set indices_truncated",
    );
}

/// HASH-shaped FD families (SOCKHASH / DEVMAP_HASH / HASH_OF_MAPS)
/// short-circuit to populated=0/scanned=0/empty.
#[test]
fn render_fd_array_hash_shaped_returns_empty() {
    let (scene, info) = build_fd_array_scene(
        super::super::bpf_map::BPF_MAP_TYPE_SOCKHASH,
        16,
        &[0, 5, 10],
    );
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let fa = render_fd_array_slots(&accessor, &info);
    assert_eq!(fa.populated, 0);
    assert_eq!(fa.scanned, 0);
    assert!(fa.indices.is_empty());
    assert!(!fa.truncated);
    assert!(
        !fa.indices_truncated,
        "hash-shaped early exit must NOT set indices_truncated",
    );
}

/// `max_entries > MAX_FD_ARRAY_SLOTS` triggers the truncation flag.
/// Walker still reports populated/indices for the slots it scanned.
#[test]
fn render_fd_array_max_entries_truncation() {
    // max_entries one above the cap. Skip populating slots — the
    // truncation flag fires regardless of population state. Build a
    // scene with the cap as effective scan size.
    let (scene, mut info) = build_fd_array_scene(
        super::super::bpf_map::BPF_MAP_TYPE_PROG_ARRAY,
        super::render_map::MAX_FD_ARRAY_SLOTS,
        &[],
    );
    info.max_entries = super::render_map::MAX_FD_ARRAY_SLOTS + 1;
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let fa = render_fd_array_slots(&accessor, &info);
    assert!(
        fa.truncated,
        "max_entries above MAX_FD_ARRAY_SLOTS must set truncated"
    );
    assert_eq!(fa.scanned, super::render_map::MAX_FD_ARRAY_SLOTS);
    // No populated slots in this scene → indices.len() == 0 == populated,
    // so indices_truncated stays false even though scan-size truncation
    // fires. Pin the orthogonality of the two flags.
    assert!(
        !fa.indices_truncated,
        "scan-size truncation must NOT set indices_truncated when populated == indices.len()",
    );
    let _ = scene; // silence unused
}

// -- #52 STRUCT_OPS render error paths ----------------------------

/// `render_map` STRUCT_OPS arm with `struct_ops_offsets = None`
/// surfaces the BTF-offsets-unresolved diagnostic — the renderer
/// must NOT silently read with `data_off = 0` against a wrapper-
/// inclusive `value_size`.
#[test]
fn render_map_struct_ops_no_offsets_returns_error() {
    let mut offsets = crate::monitor::btf_offsets::BpfMapOffsets::EMPTY;
    // No struct_ops_offsets resolution.
    offsets.struct_ops_offsets = None;
    // Provide buf with map at PA 0x1000 — won't actually be read.
    let buf = vec![0u8; 0x4000];
    let page_offset = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    // SAFETY: buf is a live local Vec<u8>.
    let mem =
        unsafe { super::super::reader::GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &offsets, 0);
    let (name_bytes, name_len) = name_from_str("test_struct_ops");
    let info = super::super::bpf_map::BpfMapInfo {
        map_pa: 0,
        map_kva: pa_to_kva(0x1000, page_offset),
        name_bytes,
        name_len,
        map_type: super::super::bpf_map::BPF_MAP_TYPE_STRUCT_OPS,
        map_flags: 0,
        key_size: 0,
        value_size: 256,
        max_entries: 1,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    let arena_page_index = super::render_map::ArenaPageIndex::new();
    let sdt_alloc_metas: Vec<super::render_map::SdtAllocMeta> = Vec::new();
    let ctx = super::render_map::RenderMapCtx {
        accessor: &accessor,
        btf: None,
        num_cpus: 1,
        arena_offsets: None,
        shared_arena: None,
        arena_page_index: &arena_page_index,
        sdt_alloc_metas: &sdt_alloc_metas,
        cast_map: None,
        arena_type_index: None,
        cross_btf_fwd_index: None,
        scx_static_index: None,
    };
    let rendered = super::render_map::render_map(&ctx, &info);
    let err = rendered
        .error
        .expect("STRUCT_OPS no-offsets must surface error");
    assert!(
        err.contains("STRUCT_OPS value unreadable") && err.contains("BTF offsets unresolved"),
        "STRUCT_OPS no-offsets error must explain the resolution failure; got: {err}",
    );
}

/// `render_map` STRUCT_OPS arm with valid struct_ops_offsets but
/// `value_kva` translating outside the buffer surfaces the
/// "value region unmapped" diagnostic.
#[test]
fn render_map_struct_ops_unmapped_value_returns_error() {
    let mut offsets = crate::monitor::btf_offsets::BpfMapOffsets::EMPTY;
    offsets.struct_ops_offsets = Some(super::super::btf_offsets::StructOpsOffsets {
        kvalue: 64,
        value_data: 8,
    });
    let buf = vec![0u8; 0x4000];
    let page_offset = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    let mem =
        unsafe { super::super::reader::GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &offsets, 0);
    // value_kva points outside the buffer (page_offset + far) so
    // the read_value translate fails.
    let (name_bytes, name_len) = name_from_str("test_struct_ops");
    let info = super::super::bpf_map::BpfMapInfo {
        map_pa: 0,
        map_kva: pa_to_kva(0x1000, page_offset),
        name_bytes,
        name_len,
        map_type: super::super::bpf_map::BPF_MAP_TYPE_STRUCT_OPS,
        map_flags: 0,
        key_size: 0,
        value_size: 256,
        max_entries: 1,
        value_kva: Some(page_offset + 0x100_0000), // far past buffer
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    let arena_page_index = super::render_map::ArenaPageIndex::new();
    let sdt_alloc_metas: Vec<super::render_map::SdtAllocMeta> = Vec::new();
    let ctx = super::render_map::RenderMapCtx {
        accessor: &accessor,
        btf: None,
        num_cpus: 1,
        arena_offsets: None,
        shared_arena: None,
        arena_page_index: &arena_page_index,
        sdt_alloc_metas: &sdt_alloc_metas,
        cast_map: None,
        arena_type_index: None,
        cross_btf_fwd_index: None,
        scx_static_index: None,
    };
    let rendered = super::render_map::render_map(&ctx, &info);
    let err = rendered
        .error
        .expect("STRUCT_OPS unmapped-value must surface error");
    assert!(
        err.contains("STRUCT_OPS value unreadable") && err.contains("value region unmapped"),
        "STRUCT_OPS unmapped-value error must mention the unmapped region; got: {err}",
    );
}

/// `find_all_bpf_maps` populates `value_kva = kvalue + data_off` for
/// STRUCT_OPS maps when `struct_ops_offsets` is resolved. This test
/// covers the static math without the page-walk to confirm the
/// kvalue + data_off chain matches the spec from
/// `bpf_struct_ops_map_alloc`.
#[test]
fn struct_ops_value_kva_math_kvalue_plus_data() {
    let so = super::super::btf_offsets::StructOpsOffsets {
        kvalue: 0x40,
        value_data: 0x10,
    };
    let map_kva = 0xFFFF_8888_0000_0000u64;
    // Production calculation: map_kva + kvalue + value_data.
    let value_kva = map_kva
        .wrapping_add(so.kvalue as u64)
        .wrapping_add(so.value_data as u64);
    assert_eq!(value_kva, 0xFFFF_8888_0000_0050);
}

/// `populated > MAX_FD_ARRAY_INDICES`: indices vector caps at the
/// limit; populated continues to count beyond. The renderer flags
/// the divergence on `indices_truncated` so a downstream consumer
/// can see at a glance that the indices list is partial.
#[test]
fn render_fd_array_indices_capped_at_max_indices() {
    let cap = super::render_map::MAX_FD_ARRAY_INDICES as u32;
    // Populate cap + 5 slots; expect populated=cap+5, indices=cap.
    let pop: Vec<u32> = (0..cap + 5).collect();
    let (scene, info) = build_fd_array_scene(
        super::super::bpf_map::BPF_MAP_TYPE_PROG_ARRAY,
        cap + 5,
        &pop,
    );
    let mem = unsafe {
        super::super::reader::GuestMem::new(scene.buf.as_ptr() as *mut u8, scene.buf.len() as u64)
    };
    let kernel = super::super::guest::GuestKernel::new_for_test(
        std::sync::Arc::new(mem),
        std::collections::HashMap::new(),
        scene.page_offset,
        0,
        false,
    );
    let kernel_ref = unsafe { &*(&kernel as *const _) };
    let accessor =
        super::super::bpf_map::GuestMemMapAccessor::new_for_test(kernel_ref, &scene.offsets, 0);
    let fa = render_fd_array_slots(&accessor, &info);
    assert_eq!(
        fa.populated,
        cap + 5,
        "populated counts every non-zero slot"
    );
    assert_eq!(
        fa.indices.len() as u32,
        cap,
        "indices vector caps at MAX_FD_ARRAY_INDICES",
    );
    assert!(
        fa.indices_truncated,
        "populated > indices.len() must set indices_truncated",
    );
}

/// `dump_truncated_at_us` defaults to None and roundtrips through
/// serde with `skip_serializing_if`. The field is absent on the
/// wire when None (every healthy dump) and surfaces a u64 us-offset
/// when the soft deadline fires.
#[test]
fn report_dump_truncated_at_us_serde() {
    // None: field absent in JSON.
    let r = FailureDumpReport::default();
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        !json.contains("dump_truncated_at_us"),
        "None must skip-serialize: {json}"
    );
    let parsed: FailureDumpReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.dump_truncated_at_us, None);

    // Some(us): field present and roundtrips.
    let r = FailureDumpReport {
        dump_truncated_at_us: Some(15_000),
        ..FailureDumpReport::default()
    };
    let json = serde_json::to_string(&r).unwrap();
    assert!(
        json.contains("\"dump_truncated_at_us\":15000"),
        "Some must serialize: {json}"
    );
    let parsed: FailureDumpReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.dump_truncated_at_us, Some(15_000));
}

/// `MAX_ENRICHED_TASKS` constant is non-zero and large enough to
/// cover any healthy SCX runnable_list depth (bounded by the
/// kernel's stall watchdog, well under 4096 in practice).
#[test]
fn max_enriched_tasks_constant_is_reasonable() {
    // Pin the constant so a future tightening / loosening is a
    // deliberate test edit, not a silent drift.
    assert_eq!(super::MAX_ENRICHED_TASKS, 4096);
}

// ---- Inline small struct rendering for FailureDumpEntry --------
//
// When a key or value is a struct with ≤ 3 non-zero non-fmt-string
// inline-scalar fields, Display collapses it into the
// `{field: value, ...}` form. When BOTH sides qualify, the entry
// renders on a single line. When only one qualifies, that side
// inlines and the other keeps its block form. Payload (when
// present) always renders as a block below the entry.

fn make_small_struct(type_name: &str, fields: &[(&str, u64)]) -> RenderedValue {
    RenderedValue::Struct {
        type_name: Some(type_name.into()),
        members: fields
            .iter()
            .map(|(n, v)| super::super::btf_render::RenderedMember {
                name: (*n).into(),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: *v,
                },
            })
            .collect(),
    }
}

#[test]
fn entry_display_renders_inline_struct_key_and_value() {
    // With the consolidated indent-based format, each entry renders
    // as:
    //   `entry: key=<key>\n  value: <value>`
    // Small structs collapse onto their own line via the btf_render
    // inline form `Type{f=v, f=v}`.
    let entry = FailureDumpEntry {
        key: Some(make_small_struct(
            "cgroup_llc_id",
            &[("cgrp_id", 1), ("llc_id", 5)],
        )),
        key_hex: "01 05".into(),
        value: Some(make_small_struct(
            "cbw_llc_entry",
            &[("llcx", 17_592_186_046_336)],
        )),
        value_hex: "00".into(),
        payload: None,
    };
    let out = format!("{entry}");
    // Header line: `entry: key=cgroup_llc_id{cgrp_id=1, llc_id=5}`
    assert!(
        out.starts_with("entry: key="),
        "missing entry header: {out}",
    );
    assert!(out.contains("cgroup_llc_id{"), "key inline form: {out}");
    assert!(out.contains("cgrp_id=1"), "key field cgrp_id: {out}");
    assert!(out.contains("llc_id=5"), "key field llc_id: {out}");
    // Value line: indented `  value: cbw_llc_entry{...}`.
    assert!(
        out.contains("\n  value: "),
        "missing indented value line: {out}",
    );
    assert!(out.contains("cbw_llc_entry{"), "value inline form: {out}");
    assert!(out.contains("llcx=17592186046336"), "value field: {out}");
    // No `struct` keyword in inline form (it's been dropped from
    // the renderer entirely).
    assert!(
        !out.contains("struct cgroup_llc_id"),
        "inline form must drop `struct` prefix: {out}",
    );
    assert!(
        !out.contains("struct cbw_llc_entry"),
        "inline form must drop `struct` prefix: {out}",
    );
}

#[test]
fn entry_display_inline_zero_fields_dropped_silently() {
    // Zero fields are suppressed silently — no `(N fields zero)`
    // summary appears in the inline braces or anywhere else. The
    // operator infers from the rendered fields that the rest are
    // zero.
    let entry = FailureDumpEntry {
        key: Some(make_small_struct(
            "k",
            &[("real", 7), ("zero1", 0), ("zero2", 0)],
        )),
        key_hex: "07".into(),
        value: Some(make_small_struct("v", &[("real", 3)])),
        value_hex: "03".into(),
        payload: None,
    };
    let out = format!("{entry}");
    assert!(out.contains("real=7"), "non-zero key field present: {out}");
    assert!(
        out.contains("real=3"),
        "non-zero value field present: {out}"
    );
    assert!(!out.contains("zero1"), "zero fields are suppressed: {out}",);
    assert!(
        !out.contains("fields zero"),
        "no zero-count summary anywhere: {out}",
    );
}

#[test]
fn entry_display_value_falls_to_multi_line_when_too_wide() {
    // Value is a struct that's small enough by field count (5) but
    // its rendered inline form may exceed the inline width budget
    // — actually 5 small u64 fields fit comfortably. The btf_render
    // inline path handles arbitrary field counts so long as the
    // rendered length fits 120 chars; this test pins a value that
    // intentionally exceeds the budget to exercise the multi-line
    // breadcrumb fallback.
    let big_value = RenderedValue::Struct {
        type_name: Some("v".into()),
        members: (0..15)
            .map(|i| super::super::btf_render::RenderedMember {
                name: format!("very_long_field_name_{i}"),
                value: RenderedValue::Uint {
                    bits: 64,
                    value: 0x1234_5678_9abc_def0u64.wrapping_add(i as u64),
                },
            })
            .collect(),
    };
    let entry = FailureDumpEntry {
        key: Some(make_small_struct("k", &[("only", 1)])),
        key_hex: "01".into(),
        value: Some(big_value),
        value_hex: "01".into(),
        payload: None,
    };
    let out = format!("{entry}");
    // Multi-line breadcrumb: `value: v:` then indented field rows.
    // Block form has at least one extra newline beyond the header
    // and value lines.
    assert!(
        out.contains("\n  value: v:"),
        "multi-line value uses breadcrumb form: {out}",
    );
    // Key still inlines.
    assert!(out.contains("k{only=1}"), "key inline form: {out}");
}

#[test]
fn entry_display_payload_renders_below_value() {
    // Payload value renders below the entry on its own indented
    // line. The Display impl uses `\n  payload <rendered>` (with
    // a space, no colon) so the value's own breadcrumb completes
    // the line as `payload TypeName:` for multi-line structs or
    // just `payload <scalar>` for scalars.
    let entry = FailureDumpEntry {
        key: Some(make_small_struct("k", &[("a", 1)])),
        key_hex: "01".into(),
        value: Some(make_small_struct("v", &[("b", 2)])),
        value_hex: "02".into(),
        payload: Some(RenderedValue::Uint {
            bits: 64,
            value: 0xDEAD_BEEF,
        }),
    };
    let out = format!("{entry}");
    assert!(out.starts_with("entry: key="), "entry header: {out}");
    assert!(out.contains("\n  value: v{b=2}"), "value inline: {out}");
    // Payload on a separate line. Scalars don't have a breadcrumb,
    // so the rendered form is just the decimal number.
    assert!(out.contains("\n  .data "), ".data label must appear: {out}",);
    assert!(
        out.contains("3735928559"), // 0xDEADBEEF in decimal
        "rendered payload value must appear: {out}",
    );
}

// ---- Table rendering for FailureDumpMap -------------------------
//
// Homogeneous entries (every entry has key+value as same-shape
// inline-scalar struct, no payload) render as a compact table.
// Heterogeneous or non-qualifying batches fall back to the
// per-entry block form.

#[test]
fn map_display_table_for_homogeneous_entries() {
    let m = FailureDumpMap {
        name: "cbw".into(),
        map_type: BPF_MAP_TYPE_HASH,
        value_size: 8,
        max_entries: 64,
        value: None,
        entries: vec![
            FailureDumpEntry {
                key: Some(make_small_struct(
                    "cgroup_llc_id",
                    &[("cgrp_id", 1), ("llc_id", 5)],
                )),
                key_hex: "01 05".into(),
                value: Some(make_small_struct(
                    "cbw_llc_entry",
                    &[("llcx", 17_592_186_046_336)],
                )),
                value_hex: "00".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(make_small_struct(
                    "cgroup_llc_id",
                    &[("cgrp_id", 61), ("llc_id", 3)],
                )),
                key_hex: "3d 03".into(),
                value: Some(make_small_struct(
                    "cbw_llc_entry",
                    &[("llcx", 17_592_186_047_616)],
                )),
                value_hex: "00".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(make_small_struct(
                    "cgroup_llc_id",
                    &[("cgrp_id", 41), ("llc_id", 1)],
                )),
                key_hex: "29 01".into(),
                value: Some(make_small_struct(
                    "cbw_llc_entry",
                    &[("llcx", 17_592_186_047_040)],
                )),
                value_hex: "00".into(),
                payload: None,
            },
        ],
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let out = format!("{m}");
    // Header row carries column names with `|` separating key
    // from value columns.
    assert!(out.contains("cgrp_id"), "key column header missing: {out}",);
    assert!(out.contains("llc_id"), "key column header missing: {out}");
    assert!(out.contains("llcx"), "value column header missing: {out}");
    assert!(out.contains(" | "), "key/value separator missing: {out}");
    // Data rows carry the per-entry values.
    assert!(out.contains("17592186046336"), "row 0 value: {out}");
    assert!(out.contains("17592186047616"), "row 1 value: {out}");
    assert!(out.contains("17592186047040"), "row 2 value: {out}");
    // Per-entry block form should NOT appear (table replaced it).
    assert!(
        !out.contains("entry {"),
        "table form must replace per-entry blocks: {out}",
    );
}

#[test]
fn map_display_skips_table_for_single_entry() {
    // Single-entry maps fall through to per-entry rendering — the
    // table header overhead exceeds the savings.
    let m = FailureDumpMap {
        name: "single".into(),
        map_type: BPF_MAP_TYPE_HASH,
        value_size: 8,
        max_entries: 64,
        value: None,
        entries: vec![FailureDumpEntry {
            key: Some(make_small_struct("k", &[("a", 1)])),
            key_hex: "01".into(),
            value: Some(make_small_struct("v", &[("b", 2)])),
            value_hex: "02".into(),
            payload: None,
        }],
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let out = format!("{m}");
    // Per-entry rendering uses the indent-based format, with each
    // entry starting on its own line as `entry: key=...`. The
    // table form would have started with column headers like
    // `cgrp_id  llc_id |` — verify the per-entry path took over.
    assert!(
        out.contains("entry: key="),
        "single entry must keep per-entry rendering: {out}",
    );
    assert!(
        !out.contains(" | "),
        "single entry must not use table form: {out}",
    );
}

#[test]
fn map_display_skips_table_when_payload_present() {
    // Any entry with a payload disqualifies the whole batch — the
    // table can't carry the per-entry typed payload below each row.
    let m = FailureDumpMap {
        name: "with_payload".into(),
        map_type: BPF_MAP_TYPE_HASH,
        value_size: 8,
        max_entries: 64,
        value: None,
        entries: vec![
            FailureDumpEntry {
                key: Some(make_small_struct("k", &[("a", 1)])),
                key_hex: "01".into(),
                value: Some(make_small_struct("v", &[("b", 2)])),
                value_hex: "02".into(),
                payload: Some(RenderedValue::Uint {
                    bits: 64,
                    value: 99,
                }),
            },
            FailureDumpEntry {
                key: Some(make_small_struct("k", &[("a", 3)])),
                key_hex: "03".into(),
                value: Some(make_small_struct("v", &[("b", 4)])),
                value_hex: "04".into(),
                payload: None,
            },
        ],
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let out = format!("{m}");
    // Per-entry rendering used (each entry has `entry {` opener).
    assert!(
        out.contains("entry: key="),
        "payload-bearing batch must use per-entry form: {out}",
    );
    assert!(out.contains("\n  .data "), ".data still surfaces: {out}",);
}

#[test]
fn map_display_skips_table_for_heterogeneous_types() {
    // Different key type names → not homogeneous → no table.
    let m = FailureDumpMap {
        name: "het".into(),
        map_type: BPF_MAP_TYPE_HASH,
        value_size: 8,
        max_entries: 64,
        value: None,
        entries: vec![
            FailureDumpEntry {
                key: Some(make_small_struct("k1", &[("a", 1)])),
                key_hex: "01".into(),
                value: Some(make_small_struct("v", &[("b", 2)])),
                value_hex: "02".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(make_small_struct("k2", &[("a", 3)])),
                key_hex: "03".into(),
                value: Some(make_small_struct("v", &[("b", 4)])),
                value_hex: "04".into(),
                payload: None,
            },
        ],
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let out = format!("{m}");
    // Per-entry rendering, NOT a table.
    assert!(
        out.contains("entry: key="),
        "heterogeneous types must use per-entry form: {out}",
    );
}

#[test]
fn map_display_skips_table_when_entry_has_no_btf_render() {
    // Any entry with a None key or value (hex-only fallback)
    // disqualifies the table.
    let m = FailureDumpMap {
        name: "no_btf".into(),
        map_type: BPF_MAP_TYPE_HASH,
        value_size: 8,
        max_entries: 64,
        value: None,
        entries: vec![
            FailureDumpEntry {
                key: None,
                key_hex: "ab".into(),
                value: None,
                value_hex: "cd".into(),
                payload: None,
            },
            FailureDumpEntry {
                key: Some(make_small_struct("k", &[("a", 1)])),
                key_hex: "01".into(),
                value: Some(make_small_struct("v", &[("b", 2)])),
                value_hex: "02".into(),
                payload: None,
            },
        ],
        percpu_entries: Vec::new(),
        percpu_hash_entries: Vec::new(),
        arena: None,
        ringbuf: None,
        stack_trace: None,
        fd_array: None,
        error: None,
    };
    let out = format!("{m}");
    assert!(
        out.contains("entry: key="),
        "missing BTF render disqualifies table: {out}",
    );
    assert!(
        out.contains("ab (raw)"),
        "hex fallback must still surface: {out}",
    );
}

// -- append_arena_type_index_for_allocator -----------------------
//
// Coverage for the index-build helper that the dump pre-pass calls
// per allocator. The helper handles size-fits-u32 conversion, the
// dedup-on-duplicate-slot-start `tracing::debug!` path, and the
// "no payload type" short-circuit. Tests run against a synthesized
// `Vec<SdtAllocEntry>` rather than booting a VM — the helper is a
// pure function over its inputs.

/// Construct an [`SdtAllocEntry`] for the index-build tests.
/// `payload` is set to a placeholder `Bytes` value — the index
/// build path only reads `user_addr`, so the rest is filler.
fn mk_alloc_entry(idx: i32, genn: i32, user_addr: u64) -> super::super::sdt_alloc::SdtAllocEntry {
    super::super::sdt_alloc::SdtAllocEntry {
        idx,
        genn,
        user_addr,
        payload: super::super::btf_render::RenderedValue::Bytes { hex: String::new() },
    }
}

/// `target_type_id == 0` short-circuits — the helper does not
/// produce any index entries because the bridge gate would filter
/// them as "no payload type" anyway. Pinning the early bail keeps
/// callers from accidentally polluting the index with zero ids.
#[test]
fn append_arena_type_index_for_allocator_zero_target_type_id_skips() {
    use super::render_map::{ArenaTypeIndex, append_arena_type_index_for_allocator};
    let mut index = ArenaTypeIndex::new();
    let entries = vec![mk_alloc_entry(0, 0, 0x0000_1000)];
    append_arena_type_index_for_allocator(
        &mut index,
        "test_allocator",
        0, // target_type_id == 0 ⇒ short-circuit
        8,
        16,
        &entries,
    );
    assert!(
        index.is_empty(),
        "zero target_type_id must skip every entry; got {} index entries",
        index.len(),
    );
}

/// `header_size` or `elem_size` that would not fit `u32` (only
/// reachable from a corrupted snapshot — the kernel caps both well
/// below `u32::MAX`) skips silently. Pinning the no-panic behaviour
/// keeps a torn read from aborting the whole dump.
#[test]
fn append_arena_type_index_for_allocator_oversized_skips() {
    use super::render_map::{ArenaTypeIndex, append_arena_type_index_for_allocator};
    let mut index = ArenaTypeIndex::new();
    let entries = vec![mk_alloc_entry(0, 0, 0x0000_1000)];
    // elem_size > u32::MAX ⇒ try_from fails, helper bails.
    append_arena_type_index_for_allocator(
        &mut index,
        "test_allocator",
        7,
        8,
        u64::from(u32::MAX) + 1,
        &entries,
    );
    assert!(
        index.is_empty(),
        "elem_size > u32::MAX must skip every entry; got {} entries",
        index.len(),
    );
}

/// Multi-entry insert: each `SdtAllocEntry` becomes one index entry
/// keyed by `user_addr as u32` with the shared `ArenaSlotInfo`.
/// Pinning the per-allocator append shape so a future inner-loop
/// rewrite can't silently drop entries.
#[test]
fn append_arena_type_index_for_allocator_multi_entry_insert() {
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex, append_arena_type_index_for_allocator};
    let mut index = ArenaTypeIndex::new();
    let entries = vec![
        mk_alloc_entry(0, 0, 0x0000_1000),
        mk_alloc_entry(1, 0, 0x0000_2000),
        mk_alloc_entry(2, 0, 0x0000_3000),
    ];
    append_arena_type_index_for_allocator(&mut index, "test_allocator", 7, 8, 16, &entries);
    let expected_info = ArenaSlotInfo {
        elem_size: 16,
        header_size: 8,
        target_type_id: 7,
    };
    assert_eq!(index.len(), 3);
    assert_eq!(index.get(&0x0000_1000), Some(&expected_info));
    assert_eq!(index.get(&0x0000_2000), Some(&expected_info));
    assert_eq!(index.get(&0x0000_3000), Some(&expected_info));
}

/// Duplicate `slot_start` keeps the FIRST entry. The
/// `tracing::debug!` line for the collision is not asserted — the
/// behaviour test is "vacant wins, occupied keeps prior value". Pin
/// the dedup policy against a future flip to last-wins (which would
/// silently overwrite a live slot's metadata with a stale one when
/// a freed allocation racing the freeze surfaces in two passes).
#[test]
fn append_arena_type_index_for_allocator_duplicate_slot_keeps_first() {
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex, append_arena_type_index_for_allocator};
    let mut index = ArenaTypeIndex::new();
    // First call seeds slot 0x1000 with payload type 7.
    let entries_first = vec![mk_alloc_entry(0, 0, 0x0000_1000)];
    append_arena_type_index_for_allocator(&mut index, "alloc_a", 7, 8, 16, &entries_first);
    // Second call tries to insert the same slot start with a
    // distinct payload type 11 (e.g. a stale snapshot after free
    // racing the freeze). Helper must keep the first entry.
    let entries_second = vec![mk_alloc_entry(0, 0, 0x0000_1000)];
    append_arena_type_index_for_allocator(&mut index, "alloc_b", 11, 8, 16, &entries_second);
    assert_eq!(index.len(), 1);
    assert_eq!(
        index.get(&0x0000_1000),
        Some(&ArenaSlotInfo {
            elem_size: 16,
            header_size: 8,
            target_type_id: 7,
        }),
        "duplicate slot_start must keep first entry's payload type",
    );
}

/// Two distinct allocators contribute non-overlapping slot ranges
/// to one index. Pinning the multi-allocator merge against a
/// regression where one allocator's metadata might silently
/// overwrite another's because both used the same low-32 windowed
/// keys.
#[test]
fn append_arena_type_index_for_allocator_multi_allocator_merge() {
    use super::render_map::{ArenaSlotInfo, ArenaTypeIndex, append_arena_type_index_for_allocator};
    let mut index = ArenaTypeIndex::new();
    // Allocator A — payload type 7, two slots.
    let entries_a = vec![
        mk_alloc_entry(0, 0, 0x0000_1000),
        mk_alloc_entry(1, 0, 0x0000_2000),
    ];
    append_arena_type_index_for_allocator(&mut index, "alloc_a", 7, 8, 16, &entries_a);
    // Allocator B — payload type 11, two distinct slots.
    let entries_b = vec![
        mk_alloc_entry(0, 0, 0x0000_3000),
        mk_alloc_entry(1, 0, 0x0000_4000),
    ];
    append_arena_type_index_for_allocator(&mut index, "alloc_b", 11, 8, 16, &entries_b);
    let info_a = ArenaSlotInfo {
        elem_size: 16,
        header_size: 8,
        target_type_id: 7,
    };
    let info_b = ArenaSlotInfo {
        elem_size: 16,
        header_size: 8,
        target_type_id: 11,
    };
    assert_eq!(index.len(), 4);
    assert_eq!(index.get(&0x0000_1000), Some(&info_a));
    assert_eq!(index.get(&0x0000_2000), Some(&info_a));
    assert_eq!(index.get(&0x0000_3000), Some(&info_b));
    assert_eq!(index.get(&0x0000_4000), Some(&info_b));
}

// -- resolve_cross_btf_fwd_in_index --------------------------------
//
// The free helper backs [`AccessorMemReader::cross_btf_resolve_fwd`].
// Tests below exercise gates that are difficult to reach through a
// full `GuestKernel` mock: aggregate-kind mismatch is the most
// important — the indexer keeps Struct/Union entries tagged in
// production, but the helper still validates the kind at lookup
// time against the caller's [`FwdKind`] argument because (a) the
// index format does not encode the kind and (b) a future indexer
// rewrite could let a same-name Union entry slip through that the
// caller specifically asked for as a Struct (or vice versa).

/// Build a minimal `.BTF` blob containing a single named
/// `BTF_KIND_STRUCT` so the helper's `resolve_type_by_id` succeeds
/// for the caller-supplied id. Returns `(blob, struct_foo_id)`.
/// Inlined here rather than reusing `cast_analysis_load::tests`
/// builders because the dump tests module is not in that file's
/// `cfg(test)` scope.
fn build_btf_with_named_struct(name: &str) -> (Vec<u8>, u32) {
    use std::io::Write;
    // String section: leading NUL + "u64\0" + "<name>\0" + "x\0".
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = strings.len() as u32;
    strings.extend_from_slice(b"u64");
    strings.push(0);
    let n_struct = strings.len() as u32;
    strings.extend_from_slice(name.as_bytes());
    strings.push(0);
    let n_x = strings.len() as u32;
    strings.extend_from_slice(b"x");
    strings.push(0);

    // Type section: id 1 = BTF_KIND_INT u64, id 2 = BTF_KIND_STRUCT
    // <name> { u64 x @ 0 }.
    const BTF_KIND_INT: u32 = 1;
    const BTF_KIND_STRUCT: u32 = 4;
    let mut types: Vec<u8> = Vec::new();
    // id 1: Int u64.
    types.extend_from_slice(&n_u64.to_le_bytes());
    let int_info = (BTF_KIND_INT << 24) & 0x1f00_0000;
    types.extend_from_slice(&int_info.to_le_bytes());
    types.extend_from_slice(&8u32.to_le_bytes()); // size
    let int_data: u32 = 64;
    types.extend_from_slice(&int_data.to_le_bytes()); // encoding=0, offset=0, bits=64
    // id 2: Struct <name> { u64 x @ bit 0 }.
    types.extend_from_slice(&n_struct.to_le_bytes());
    let struct_info = ((BTF_KIND_STRUCT << 24) & 0x1f00_0000) | 1u32; // vlen=1
    types.extend_from_slice(&struct_info.to_le_bytes());
    types.extend_from_slice(&8u32.to_le_bytes()); // size
    types.extend_from_slice(&n_x.to_le_bytes()); // member name_off
    types.extend_from_slice(&1u32.to_le_bytes()); // member type id (u64)
    types.extend_from_slice(&0u32.to_le_bytes()); // bit_offset

    // Header (24 bytes) + type section + string section.
    let type_len = types.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob: Vec<u8> = Vec::new();
    blob.write_all(&0xEB9F_u16.to_le_bytes()).unwrap(); // magic
    blob.push(1); // version
    blob.push(0); // flags
    blob.write_all(&24u32.to_le_bytes()).unwrap(); // hdr_len
    blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
    blob.write_all(&type_len.to_le_bytes()).unwrap();
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off = type_len
    blob.write_all(&str_len.to_le_bytes()).unwrap();
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);
    (blob, 2)
}

/// Aggregate-kind gate fires: index has `("foo", FwdIndexEntry { 0,
/// struct_foo_id })` pointing at a `BTF_KIND_STRUCT`, but the
/// caller queries with [`FwdKind::Union`]. The helper's kind-match
/// arm in [`super::render_map::resolve_cross_btf_fwd_in_index`]
/// rejects with `None`, dropping the chase back to the historical
/// Fwd skip.
///
/// Without this gate, a same-name Union body in a sibling BTF
/// could surface for a caller that asked for a Struct (and
/// vice versa), corrupting the rendered subtree's layout
/// interpretation. Pin the rejection so a future indexer rewrite
/// that admits Union entries cannot silently bypass the kind
/// check.
#[test]
fn resolve_cross_btf_fwd_in_index_rejects_kind_mismatch() {
    use crate::monitor::btf_render::FwdKind;
    use crate::vmm::cast_analysis_load::FwdIndexEntry;
    use std::sync::Arc;

    // Build a sibling BTF whose `foo` is a Struct (id 2). The
    // index will key `foo -> (0, 2)`; the kind-match check in
    // the helper resolves type id 2, sees it's a Struct, and
    // when the caller asks for [`FwdKind::Union`] returns None.
    let (blob, struct_foo_id) = build_btf_with_named_struct("foo");
    let btf = Arc::new(btf_rs::Btf::from_bytes(&blob).expect("synthetic BTF parses"));
    let btfs = vec![btf];
    let mut fwd_index: std::collections::HashMap<String, FwdIndexEntry> =
        std::collections::HashMap::new();
    fwd_index.insert(
        "foo".to_string(),
        FwdIndexEntry {
            btfs_idx: 0,
            type_id: struct_foo_id,
        },
    );
    let cross = super::CrossBtfFwdIndex {
        btfs: &btfs,
        fwd_index: &fwd_index,
    };
    // Query with [`FwdKind::Union`] (caller is asking for a Union
    // body). The index entry is a Struct → kind mismatch → helper
    // returns None.
    let result = super::render_map::resolve_cross_btf_fwd_in_index(
        Some(&cross),
        "foo",
        FwdKind::Union,
    );
    assert!(
        result.is_none(),
        "kind mismatch (Struct entry, FwdKind::Union) must reject; \
         got Some(...)",
    );

    // Sanity: same query with [`FwdKind::Struct`] succeeds —
    // proves the index lookup itself works and the kind gate is
    // the rejection cause, not an unrelated absence.
    let success = super::render_map::resolve_cross_btf_fwd_in_index(
        Some(&cross),
        "foo",
        FwdKind::Struct,
    );
    assert!(
        success.is_some(),
        "matching kind (Struct entry, FwdKind::Struct) must succeed; \
         this confirms the rejection above is the kind gate firing",
    );
}
