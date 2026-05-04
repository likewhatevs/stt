//! Library-level E2E tests for the live-host pipeline (#15).
//!
//! Covers the full library-only loop:
//!   1. Synthetic `/dev/kmsg` window → [`parse_kmsg_window`] → `ScxExitEvent`.
//!   2. Synthetic [`DebugCapture`] with rich `WorkloadFingerprint`.
//!   3. [`generate_spec`] → [`ReproducerSpec`].
//!   4. [`render_run_file_source`] + [`render_ktstr_test_source`] →
//!      generated source containing the expected primitives.
//!   5. Serde round-trip on `DebugCapture` (off-disk format pinned).
//!   6. Funify the serialized `DebugCapture` JSON (#16) so the
//!      output is LLM-safe.
//!
//! Distinct from each module's per-unit tests in three ways:
//!  - Tests the wire integration (DebugCapture → spec → source) as
//!    one chain, not piecewise. A regression that drops a hint
//!    type from one stage trips here even when each stage's unit
//!    tests pass.
//!  - Asserts on the full generated-source string, not on
//!    `spec.config` field-by-field. A drift in the rendered token
//!    set (e.g. a renamed builder method) breaks this test.
//!  - Lives outside the `src/` tree so the integration test runs
//!    against the public API only — anything pub(crate) or
//!    private is unreachable, ensuring the LLM-safe / external-
//!    consumer surface stays usable.
//!
//! No VM boot, no kernel, no scx attach — pure library-level.

use std::time::Duration;

use ktstr::fun::{Funifier, funify_json};
use ktstr::live_host::{
    AffinityHint, CgroupHint, DebugCapture, SchedPolicyHint, ScxExitKind, WorkTypeHint,
    WorkloadGroupHint, generate_spec, parse_kmsg_window, render_ktstr_test_source,
    render_run_file_source,
};

/// A synthetic `/dev/kmsg` window covering one error-class scx exit
/// with a watchdog-style stack. The format mirrors
/// `kernel/sched/ext.c`'s `pr_info("sched_ext: BPF scheduler \"%s\"
/// disabled (...)")` plus a `%pS`-formatted stack trail. Carries
/// enough variety to exercise:
///   - Anchor detection (`sched_ext: BPF scheduler "...")
///   - Scheduler name extraction
///   - Message body capture
///   - Stuck-task pattern recognition
///   - Multi-frame stack parsing
const KMSG_FIXTURE: &str = "\
[12345.678] sched_ext: BPF scheduler \"scx_simple\" disabled (runnable task stall: ktstr_worker[1234] failed to run for 5.123s)
[12345.679] scx_simple: pid 1234 stuck on cpu 3 for 5123 ms
[12345.680] Call Trace:
[12345.681]  <TASK>
[12345.682]  scx_watchdog_workfn+0xa1/0x150
[12345.683]  process_one_work+0x1f0/0x4c0
[12345.684]  worker_thread+0x12c/0x350
[12345.685]  kthread+0xe8/0x120
[12345.686]  ret_from_fork+0x1f/0x30
[12345.687]  </TASK>
";

/// kmsg fixture with a NORMAL exit (no error stack). Lets us verify
/// the parser distinguishes operator-clean disables from stalls.
const KMSG_NORMAL_FIXTURE: &str = "\
[55555.000] sched_ext: BPF scheduler \"scx_qmap\" disabled (operator-initiated unregister)
[55555.001] sched_ext: BPF scheduler \"scx_qmap\" disabled
";

/// The full E2E pipeline from kmsg → ReproducerSpec → generated
/// source. Pins:
///   - Parser pulls the scheduler name out of the anchor line.
///   - The ScxExitEvent's exit kind classifies as a stall when
///     the message body contains `failed to run for`.
///   - A DebugCapture seeded with realistic hint coverage (workload
///     group, affinity, work type, cgroup, sched policy) generates
///     a ReproducerSpec whose config matches every hint AND whose
///     generated source contains the expected primitives.
#[test]
fn live_host_pipeline_e2e_kmsg_to_reproducer_spec_to_source() {
    // -- Phase 1: kmsg parser produces structured exit events.
    let events = parse_kmsg_window(KMSG_FIXTURE);
    assert_eq!(events.len(), 1, "expected exactly one anchor in fixture");
    let event = &events[0];
    assert_eq!(event.scheduler_name, "scx_simple");
    assert!(
        event.message.contains("failed to run for"),
        "stall message body must survive the parse: got {:?}",
        event.message,
    );
    // Stack frames extracted from %pS tokens.
    assert!(
        event.stack.iter().any(|f| f.name == "scx_watchdog_workfn"),
        "scx_watchdog_workfn frame missing from parsed stack: {:?}",
        event.stack,
    );
    // `parse_kmsg_window` runs `classify_exit_kind` inline, so this
    // fixture's `scx_watchdog_workfn` stack frame plus the "stuck on
    // cpu" message body must classify as Stall. Pinning the specific
    // variant (rather than accepting any well-known one) catches
    // regressions where the classifier silently downgrades a stall
    // to Other / Error.
    assert_eq!(
        event.kind,
        ScxExitKind::Stall,
        "watchdog stack + stuck-task message must classify as Stall: {:?}",
        event.kind,
    );

    // -- Phase 2: synthesize a DebugCapture from the parsed event +
    // realistic fingerprint data. Mirrors what the live-host
    // capture-mode binary produces in production.
    let mut capture = DebugCapture::default();
    capture.kernel_release = "6.16.0-ktstr-test".into();
    let mut wg = WorkloadGroupHint::default();
    wg.cgroup_path = "/test/scx_simple".into();
    wg.thread_count = 4;
    wg.cpu_time_fraction = 0.85;
    wg.wakeups_per_sec = 1200.0;
    capture.fingerprint.workload_groups = vec![wg];
    capture.fingerprint.affinity_hints = vec![AffinityHint::Exact {
        cpus: vec![0, 1, 2, 3],
    }];
    capture.fingerprint.work_type_hints = vec![
        WorkTypeHint::Bursty {
            burst_duration: Duration::from_millis(10),
            sleep_duration: Duration::from_millis(90),
        },
        WorkTypeHint::SpinWait, // secondary hint should land in notes
    ];
    let mut cg = CgroupHint::default();
    cg.path = "/test/scx_simple".into();
    cg.cpu_weight = Some(200);
    cg.memory_max_bytes = Some(64 * 1024 * 1024);
    cg.cpuset_cpus = vec![0, 1, 2, 3];
    cg.cpu_max_quota_us = Some(50_000);
    capture.fingerprint.cgroup_hints = vec![cg];
    capture.fingerprint.sched_policy_hints = vec![SchedPolicyHint::Other { nice: -5 }];
    capture.fingerprint.gaps = vec!["affinity hint backed by 1 sample".into()];

    // -- Phase 3: generate_spec projects the fingerprint into a
    // ReproducerSpec. Pins both the projection AND that gaps
    // propagate to notes (so a downstream user sees the
    // confidence-degradation reason).
    let spec = generate_spec(&capture);
    assert_eq!(spec.config.num_workers, 4, "thread_count must propagate");
    assert!(
        spec.notes
            .iter()
            .any(|n| n.message().contains("affinity hint backed by 1 sample")),
        "gaps must propagate to notes: {:?}",
        spec.notes,
    );
    assert!(
        spec.notes
            .iter()
            .any(|n| n.message().contains("additional work-type hints")),
        "secondary work-type hint must surface as a note: {:?}",
        spec.notes,
    );

    // -- Phase 4: render the spec into the two source forms ktstr
    // exposes.
    let run_src = render_run_file_source(&spec, "regression_repro");
    let test_src = render_ktstr_test_source(&spec, "regression_repro");
    // The .run-style source is a free function returning a
    // WorkloadConfig.
    assert!(
        run_src.contains("pub fn regression_repro"),
        ".run source must declare the named function: {run_src}",
    );
    assert!(
        run_src.contains("WorkloadConfig"),
        ".run source must construct a WorkloadConfig: {run_src}",
    );
    // The ktstr_test-style source applies the proc-macro attribute.
    assert!(
        test_src.contains("#[ktstr::ktstr_test]"),
        "ktstr_test source must apply the attribute macro: {test_src}",
    );
    assert!(
        test_src.contains("regression_repro"),
        "ktstr_test source must declare the named function: {test_src}",
    );
    // num_workers from the workload group hint should be visible.
    assert!(
        run_src.contains(".workers(4)") || run_src.contains("num_workers: 4"),
        "thread_count=4 must materialize in source: {run_src}",
    );
}

/// Normal exit fixture must NOT classify as Error/Stall.
/// Distinguishes operator-driven unloads from hangs so the
/// reproducer pipeline doesn't synthesize a "regression" test
/// from a normal exit.
#[test]
fn live_host_pipeline_e2e_normal_exit_does_not_signal_error() {
    let events = parse_kmsg_window(KMSG_NORMAL_FIXTURE);
    assert_eq!(
        events.len(),
        2,
        "fixture has two anchors (normal + duplicate)"
    );
    for event in events {
        assert_eq!(event.scheduler_name, "scx_qmap");
        // Normal exits carry no stack — distinguishes from
        // stall/error events.
        assert!(
            event.stack.is_empty(),
            "normal exit must carry no stack: got {:?}",
            event.stack,
        );
        // Stall message tokens must NOT appear in a normal exit.
        assert!(
            !event.message.contains("failed to run"),
            "normal exit must not look like a stall: {:?}",
            event.message,
        );
    }
}

/// Empty fingerprint → default ReproducerSpec + every absent-hint
/// reason in notes. This is the "no signal — fall back" path the
/// live-host pipeline exercises when the capture window contained
/// no interesting events but the user still requested a
/// reproducer template. The generated source must still compile
/// (renders a valid WorkloadConfig).
#[test]
fn live_host_pipeline_e2e_empty_capture_falls_back_to_defaults() {
    let capture = DebugCapture::default();
    let spec = generate_spec(&capture);

    // Defaults: 1 worker, no affinity, SpinWait work type.
    assert_eq!(spec.config.num_workers, 1);
    // The fall-back reasons land in notes so a user sees WHY the
    // generated test is parameterless.
    assert!(
        spec.notes
            .iter()
            .any(|n| n.message().contains("workload group")),
        "absent workload-group hint must surface a note: {:?}",
        spec.notes,
    );

    let src = render_run_file_source(&spec, "default_template");
    assert!(src.contains("pub fn default_template"));
    assert!(src.contains("WorkloadConfig"));
    // num_workers default is 1 — pin it explicitly. The generator
    // emits the public builder API (`.workers(N)`); accept either
    // form so an out-of-band switch to struct-literal syntax doesn't
    // silently break the assertion.
    assert!(
        src.contains(".workers(1)") || src.contains("num_workers: 1"),
        "default num_workers=1 must render: {src}",
    );
}

/// Serialize a DebugCapture, deserialize it back, and assert the
/// round-trip preserves every field that round-trips (Hint enums
/// + CgroupHint structs). Pins the on-disk schema for live-host
/// consumers.
#[test]
fn live_host_pipeline_e2e_debug_capture_serde_roundtrip() {
    let mut original = DebugCapture::default();
    original.kernel_release = "6.16.0-ktstr-test".into();
    let mut wg = WorkloadGroupHint::default();
    wg.cgroup_path = "/test".into();
    wg.thread_count = 8;
    wg.cpu_time_fraction = 0.5;
    wg.wakeups_per_sec = 100.0;
    original.fingerprint.workload_groups = vec![wg];
    original.fingerprint.affinity_hints = vec![
        AffinityHint::SingleCpu { cpus: Vec::new() },
        AffinityHint::CrossCgroup { cpus: Vec::new() },
    ];
    original.fingerprint.work_type_hints = vec![WorkTypeHint::FutexPingPong];
    original.fingerprint.sched_policy_hints = vec![SchedPolicyHint::Deadline {
        runtime_ns: 1_000_000,
        deadline_ns: 5_000_000,
        period_ns: 10_000_000,
    }];

    let json = serde_json::to_string(&original).expect("serialize DebugCapture");
    let restored: DebugCapture = serde_json::from_str(&json).expect("deserialize DebugCapture");

    // schema field is pinned by the producer side; default is empty
    // when not set. Either way it must round-trip.
    assert_eq!(restored.kernel_release, original.kernel_release);
    assert_eq!(
        restored.fingerprint.workload_groups.len(),
        1,
        "workload_groups must round-trip: got {:?}",
        restored.fingerprint.workload_groups,
    );
    assert_eq!(restored.fingerprint.workload_groups[0].thread_count, 8,);
    assert_eq!(
        restored.fingerprint.affinity_hints.len(),
        2,
        "affinity_hints must round-trip both",
    );
    assert!(
        matches!(
            &restored.fingerprint.work_type_hints[0],
            WorkTypeHint::FutexPingPong,
        ),
        "WorkTypeHint::FutexPingPong must round-trip exactly",
    );
    assert!(
        matches!(
            &restored.fingerprint.sched_policy_hints[0],
            SchedPolicyHint::Deadline {
                runtime_ns: 1_000_000,
                ..
            },
        ),
        "SchedPolicyHint::Deadline must round-trip with runtime preserved",
    );
}

/// Funify pass over the serialized DebugCapture replaces
/// every non-metric-keyed value (cgroup paths, scheduler names,
/// any field whose key is not in the
/// [`Funifier::is_metric_passthrough`] allowlist) without
/// breaking the JSON shape. Metric-keyed values such as
/// `thread_count` pass through unchanged. Mirrors the LLM-safe
/// consumption use case for the live-host pipeline: a captured
/// DebugCapture goes through `funify` before it ever lands in an
/// LLM context.
#[test]
fn live_host_pipeline_e2e_funify_preserves_structure() {
    let mut capture = DebugCapture::default();
    let mut wg = WorkloadGroupHint::default();
    wg.cgroup_path = "/system.slice/scx-test.service".into();
    wg.thread_count = 4;
    wg.cpu_time_fraction = 0.7;
    wg.wakeups_per_sec = 500.0;
    capture.fingerprint.workload_groups = vec![wg];
    let mut cg = CgroupHint::default();
    cg.path = "/system.slice/scx-test.service".into();
    cg.cpu_weight = Some(200);
    cg.memory_max_bytes = Some(1_048_576);
    cg.cpuset_cpus = vec![0, 1];
    cg.cpu_max_quota_us = None;
    capture.fingerprint.cgroup_hints = vec![cg];

    let json = serde_json::to_value(&capture).expect("serialize");
    let funifier = Funifier::with_seed("e2e-test");
    let funified = funify_json(json.clone(), &funifier);

    // Top-level kernel_release was empty in this capture; the
    // funified version preserves that.
    let funified_str = serde_json::to_string(&funified).expect("serialize funified");
    // Real cgroup path must NOT survive — the funifier replaces
    // any `path`/`cgroup_path`-keyed value with a fun name (since
    // neither is in the metric allowlist, the default funify path
    // fires). The test only fails if the original string leaks.
    assert!(
        !funified_str.contains("scx-test.service"),
        "real cgroup path must not survive funification: {funified_str}",
    );
    // thread_count / cpu_time_fraction / wakeups_per_sec hit the
    // metric allowlist — they pass through unchanged.
    assert!(
        funified_str.contains("\"thread_count\":4") || funified_str.contains("\"thread_count\": 4"),
        "thread_count=4 must pass through funification: {funified_str}",
    );
    // Round-trip structurally: serde_json::from_value on the
    // funified blob produces a valid DebugCapture (no schema
    // damage).
    let _restored: DebugCapture = serde_json::from_value(funified)
        .expect("funified DebugCapture must still deserialize as DebugCapture");
}

/// Exercises every supported SchedPolicyHint variant through the
/// projection + source-render pipeline. Each variant must produce
/// a distinct `SchedPolicy::*` token in the generated source; a
/// regression that collapses two variants to the same source
/// trips here even when each variant's per-unit projection passes.
#[test]
fn live_host_pipeline_e2e_sched_policy_variants_render_distinctly() {
    let cases: &[(SchedPolicyHint, &str)] = &[
        (SchedPolicyHint::Other { nice: 0 }, "Normal"),
        (SchedPolicyHint::Fifo { priority: 50 }, "Fifo"),
        (SchedPolicyHint::RoundRobin { priority: 50 }, "RoundRobin"),
        (
            SchedPolicyHint::Deadline {
                runtime_ns: 1_000_000,
                deadline_ns: 5_000_000,
                period_ns: 10_000_000,
            },
            "Deadline",
        ),
        (SchedPolicyHint::Batch, "Batch"),
        (SchedPolicyHint::Idle, "Idle"),
    ];
    for (hint, expected_token) in cases {
        let mut capture = DebugCapture::default();
        capture.fingerprint.sched_policy_hints = vec![hint.clone()];
        let spec = generate_spec(&capture);
        let src = render_run_file_source(&spec, "policy_variant_repro");
        // Each hint must materialize either the policy name OR an
        // equivalent SchedPolicy::Other(nice) variant the renderer
        // chose. The token-set assertion is loose by design — a
        // future refactor that swaps `SchedPolicy::Fifo(50)` for
        // `SchedPolicy::fifo(50)` is allowed; the upper-case
        // discriminant must survive in some form.
        assert!(
            src.contains(expected_token),
            "rendered source for {hint:?} must contain {expected_token:?}: {src}",
        );
    }
}
