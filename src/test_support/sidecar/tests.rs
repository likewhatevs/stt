use super::super::test_helpers::{EnvVarGuard, lock_env};
use super::*;
use crate::assert::{AssertResult, CgroupStats};
use crate::scenario::Ctx;
use anyhow::Result;

/// Collect every sidecar file in `dir` whose name starts with
/// `prefix` and ends with `.ktstr.json`. Returns paths in
/// filesystem iteration order; non-UTF-8 filenames are skipped.
///
/// Call sites that write a single sidecar take the first match
/// via `.into_iter().next().expect(..)` (the variant-hash suffix
/// is opaque to the test so prefix match is how the file is
/// recovered); tests that assert on the number of gauntlet
/// variants use `.len()`.
///
/// **Prefer this over hand-rolling read_dir/filter_map in new
/// write_sidecar tests** — the 7 pre-existing call sites were
/// near-identical inline blocks; funneling new tests through
/// this helper keeps the lookup contract in one place.
///
/// The `.ktstr.json` suffix filter is an intentional tightening
/// relative to the original inline pattern in
/// `write_sidecar_variant_hash_distinguishes_work_types`, which
/// filtered only by prefix. The write-side tests only ever
/// produce `.ktstr.json` files in their temp dirs, so the
/// tightening is safe and rules out future stray files (a
/// `.json.tmp` atomic-write residue, for instance) from
/// inflating the count assertions.
fn find_sidecars_by_prefix(dir: &std::path::Path, prefix: &str) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir)
        .expect("sidecar dir must exist for lookup")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".ktstr.json"))
        })
        .collect()
}

/// Single-file variant of [`find_sidecars_by_prefix`] for tests
/// that exercise one variant per run. Asserts exactly one match
/// and returns the owned path.
///
/// What the length assertion catches: a test producing MORE than
/// one sidecar under the given prefix — typically a stray
/// leftover from a prior run (if the temp-dir cleanup is stale),
/// or a call-site bug that invokes the writer twice. A
/// variant-hash collision on its own would overwrite the file
/// in place (same hash → same filename → single file), so this
/// assertion is NOT a collision detector; it's a
/// "one-call-one-file" invariant for single-variant tests.
/// Centralizes the pattern so the 5 single-variant writer tests
/// share one length check + error message.
fn find_single_sidecar_by_prefix(dir: &std::path::Path, prefix: &str) -> std::path::PathBuf {
    let paths = find_sidecars_by_prefix(dir, prefix);
    assert_eq!(
        paths.len(),
        1,
        "single-variant test must produce exactly one sidecar under \
         prefix {prefix:?}; got {paths:?}",
    );
    paths
        .into_iter()
        .next()
        .expect("length-1 vec yields Some on first next()")
}

// -- find_sidecars_by_prefix self-tests --
//
// Pin the helper's filter behavior so changes to its logic
// surface as failures here rather than as behavior shifts in
// call sites.

/// The `.ktstr.json` suffix filter must exclude files that share
/// the prefix but carry a different extension. Without the
/// suffix check, an atomic-write residue (`.json.tmp`) or a
/// non-ktstr `.json` written into the same directory would
/// inflate the match count.
#[test]
fn find_sidecars_by_prefix_filters_suffix() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    std::fs::write(tmp.join("foo-0001.ktstr.json"), b"{}").unwrap();
    std::fs::write(tmp.join("foo-0002.ktstr.json.tmp"), b"{}").unwrap();
    std::fs::write(tmp.join("foo-0003.json"), b"{}").unwrap();
    std::fs::write(tmp.join("foo-0004.ktstr.txt"), b"{}").unwrap();
    let paths = find_sidecars_by_prefix(tmp, "foo-");
    assert_eq!(
        paths.len(),
        1,
        "only the .ktstr.json file must match, got {paths:?}",
    );
}

/// The prefix filter must reject filenames whose prefix does
/// not match, so the count-based gauntlet-variant tests
/// (`write_sidecar_variant_hash_distinguishes_*`) can coexist
/// safely with sidecars from unrelated tests that happen to
/// share a parent directory.
#[test]
fn find_sidecars_by_prefix_filters_prefix() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    std::fs::write(tmp.join("foo-0001.ktstr.json"), b"{}").unwrap();
    std::fs::write(tmp.join("bar-0002.ktstr.json"), b"{}").unwrap();
    std::fs::write(tmp.join("foobar-0003.ktstr.json"), b"{}").unwrap();
    let paths = find_sidecars_by_prefix(tmp, "foo-");
    assert_eq!(
        paths.len(),
        1,
        "only files starting with 'foo-' must match (not 'foobar-'), got {paths:?}",
    );
}

/// A directory that contains nothing matching the `prefix` +
/// `.ktstr.json` contract must yield an empty `Vec`, not panic.
/// Call sites that use `.into_iter().next().expect(..)` rely on
/// this — an empty Vec lets them surface a descriptive "sidecar
/// file ... should be written" error rather than an opaque
/// helper-internal panic.
#[test]
fn find_sidecars_by_prefix_empty_when_no_match() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    std::fs::write(tmp.join("bar-0001.ktstr.json"), b"{}").unwrap();
    let paths = find_sidecars_by_prefix(tmp, "foo-");
    assert!(
        paths.is_empty(),
        "no prefix match must yield empty Vec, got {paths:?}",
    );
}

// -- test_fixture self-tests --
//
// Guard the fixture's observable shape so call-site tests can rely
// on these defaults without re-asserting them.

/// Serializing the fixture and parsing the result back must
/// succeed — proves every field is serde-compatible and no default
/// produces a value that fails to round-trip (e.g. a NaN float or
/// an invalid Option combination).
#[test]
fn test_fixture_round_trips_clean() {
    let sc = SidecarResult::test_fixture();
    let json = serde_json::to_string(&sc).expect("fixture must serialize");
    let _loaded: SidecarResult = serde_json::from_str(&json).expect("fixture JSON must parse back");
}

/// `passed=true, skipped=false` is the fixture's verdict default
/// so tests that only care about the success path don't need to
/// spell either field out. A silent flip of either bit would
/// invert the meaning of every unmodified call-site test.
#[test]
fn test_fixture_is_pass_not_skip() {
    let sc = SidecarResult::test_fixture();
    assert!(sc.passed, "fixture must default to passed=true");
    assert!(!sc.skipped, "fixture must default to skipped=false");
}

/// `host=None` is the fixture's host default so
/// [`sidecar_variant_hash_excludes_host_context`] and every test
/// that asserts the JSON does not carry a host key can rely on
/// the default rather than spelling it out. Production writers
/// populate host explicitly (see `write_sidecar` /
/// `write_skip_sidecar`).
#[test]
fn test_fixture_host_is_none() {
    let sc = SidecarResult::test_fixture();
    assert!(sc.host.is_none(), "fixture must default to host=None");
}

/// `payload=None, metrics=empty` is the fixture's default so
/// tests that verify the serde always-emit contract
/// (e.g. [`sidecar_payload_and_metrics_always_emit_when_empty`])
/// can rely on these defaults rather than re-spelling them.
#[test]
fn test_fixture_payload_and_metrics_empty() {
    let sc = SidecarResult::test_fixture();
    assert!(sc.payload.is_none(), "fixture must default to payload=None");
    assert!(
        sc.metrics.is_empty(),
        "fixture must default to metrics=empty"
    );
}

/// Summary guard on every empty-collection / None-Option /
/// empty-String default. A silent flip of any of these defaults
/// breaks every test that depends on "unset → serialized as
/// null / []" via the symmetric always-emit contract — and
/// there are many such tests across this file. One tripwire
/// here catches the flip in one place rather than fanning out
/// to per-default pins.
///
/// Hash-participating string defaults (`test_name`,
/// `topology`, `scheduler`, `work_type`) are intentionally NOT
/// re-asserted here — their drift is caught by
/// `test_fixture_variant_hash_is_stable` which pins the hash.
#[test]
fn test_fixture_all_collections_empty_by_default() {
    let sc = SidecarResult::test_fixture();
    assert!(sc.metrics.is_empty(), "metrics must default empty");
    assert!(
        sc.stimulus_events.is_empty(),
        "stimulus_events must default empty"
    );
    assert!(
        sc.verifier_stats.is_empty(),
        "verifier_stats must default empty"
    );
    assert!(sc.sysctls.is_empty(), "sysctls must default empty");
    assert!(sc.kargs.is_empty(), "kargs must default empty");
    assert!(sc.payload.is_none(), "payload must default None");
    assert!(sc.monitor.is_none(), "monitor must default None");
    assert!(sc.kvm_stats.is_none(), "kvm_stats must default None");
    assert!(
        sc.kernel_version.is_none(),
        "kernel_version must default None"
    );
    assert!(
        sc.kernel_commit.is_none(),
        "kernel_commit must default None"
    );
    assert!(sc.host.is_none(), "host must default None");
    assert!(
        sc.timestamp.is_empty(),
        "timestamp must default empty String"
    );
    assert!(sc.run_id.is_empty(), "run_id must default empty String");
    assert!(
        sc.stats.cgroups.is_empty(),
        "stats.cgroups must default empty (ScenarioStats::default)",
    );
    // Overlaps deliberately with `test_fixture_is_pass_not_skip`
    // so this single summary test is sufficient to catch a
    // verdict-default flip even if callers forget the other
    // self-test exists. Cheap belt + suspenders.
    assert!(sc.passed, "passed must default true");
    assert!(!sc.skipped, "skipped must default false");
}

/// Two fresh fixtures must hash to the same value and that value
/// must match the pinned constant. Protects against a change to
/// fixture defaults that would silently shift every call-site
/// test that passes the fixture straight into
/// [`sidecar_variant_hash`] (e.g. `sidecar_variant_hash_distinguishes_payload`'s
/// `none` handle). If this constant needs to move, every such
/// call site must be re-read to confirm the shift is intentional.
#[test]
fn test_fixture_variant_hash_is_stable() {
    let a = sidecar_variant_hash(&SidecarResult::test_fixture());
    let b = sidecar_variant_hash(&SidecarResult::test_fixture());
    assert_eq!(a, b, "two fresh fixtures must hash identically");
    assert_eq!(
        a, 0x16b754f538ba41fd,
        "fixture hash drifted — update only if the fixture default \
         change is intentional; verify every call site that passes \
         the fixture straight into sidecar_variant_hash still expresses \
         the intent it had before",
    );
}

/// Full literal intentional: exercises every field through serde so
/// a future addition is caught by a compile error here.
#[test]
fn sidecar_result_roundtrip() {
    let sc = SidecarResult {
        test_name: "my_test".to_string(),
        topology: "1n2l4c2t".to_string(),
        scheduler: "scx_mitosis".to_string(),
        scheduler_commit: Some("abc123".to_string()),
        project_commit: Some("def4567".to_string()),
        payload: None,
        metrics: vec![],
        passed: true,
        skipped: false,
        stats: crate::assert::ScenarioStats {
            cgroups: vec![CgroupStats {
                num_workers: 4,
                num_cpus: 2,
                avg_off_cpu_pct: 50.0,
                min_off_cpu_pct: 40.0,
                max_off_cpu_pct: 60.0,
                spread: 20.0,
                max_gap_ms: 100,
                max_gap_cpu: 1,
                total_migrations: 5,
                ..Default::default()
            }],
            total_workers: 4,
            total_cpus: 2,
            total_migrations: 5,
            worst_spread: 20.0,
            worst_gap_ms: 100,
            worst_gap_cpu: 1,
            ..Default::default()
        },
        monitor: Some(MonitorSummary {
            prog_stats_deltas: None,
            total_samples: 10,
            max_imbalance_ratio: 1.5,
            max_local_dsq_depth: 3,
            stuck_detected: false,
            event_deltas: Some(crate::monitor::ScxEventDeltas {
                total_fallback: 7,
                fallback_rate: 0.5,
                max_fallback_burst: 2,
                total_dispatch_offline: 0,
                total_dispatch_keep_last: 3,
                keep_last_rate: 0.2,
                total_enq_skip_exiting: 0,
                total_enq_skip_migration_disabled: 0,
                ..Default::default()
            }),
            schedstat_deltas: None,
            ..Default::default()
        }),
        stimulus_events: vec![crate::timeline::StimulusEvent {
            elapsed_ms: 500,
            label: "StepStart[0]".to_string(),
            op_kind: Some("SetCpuset".to_string()),
            detail: Some("4 cpus".to_string()),
            total_iterations: None,
        }],
        work_type: "SpinWait".to_string(),
        verifier_stats: vec![],
        kvm_stats: None,
        sysctls: vec![],
        kargs: vec![],
        kernel_version: None,
        kernel_commit: Some("kabcde7".to_string()),
        timestamp: String::new(),
        run_id: String::new(),
        host: None,
        cleanup_duration_ms: Some(123),
        run_source: Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
    };
    let json = serde_json::to_string_pretty(&sc).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    // Exhaustive destructure — `SidecarResult` is `non_exhaustive`
    // only across crates, but in-crate destructure still requires
    // every field to appear by name. Adding a field to
    // `SidecarResult` without extending this pattern fails to
    // compile here, forcing the author to make an explicit
    // roundtrip-coverage decision at the same time they introduce
    // the field. See sibling
    // [`sidecar_payload_and_metrics_always_emit_when_empty`] for
    // the empty-collection variant of this pin.
    let SidecarResult {
        test_name,
        topology,
        scheduler,
        scheduler_commit,
        project_commit,
        payload,
        metrics,
        passed,
        skipped,
        stats,
        monitor,
        stimulus_events,
        work_type,
        verifier_stats,
        kvm_stats,
        sysctls,
        kargs,
        kernel_version,
        kernel_commit,
        timestamp,
        run_id,
        host,
        cleanup_duration_ms,
        run_source,
    } = loaded;
    // Hash-participating string fields round-trip verbatim.
    assert_eq!(test_name, "my_test");
    assert_eq!(topology, "1n2l4c2t");
    assert_eq!(scheduler, "scx_mitosis");
    assert_eq!(work_type, "SpinWait");
    // Nullable string metadata fields.
    assert_eq!(scheduler_commit.as_deref(), Some("abc123"));
    assert_eq!(project_commit.as_deref(), Some("def4567"));
    assert_eq!(
        kernel_commit.as_deref(),
        Some("kabcde7"),
        "kernel_commit must round-trip the literal string \
         populated on the write side, including the 7-char \
         hex shape `detect_kernel_commit` produces. The \
         fixture uses `kabcde7` (hex-only) to make accidental \
         field-swap regressions with project_commit / \
         scheduler_commit obvious — each commit field carries \
         a distinct token.",
    );
    assert_eq!(payload, None, "fixture declared no payload");
    assert_eq!(kvm_stats, None, "fixture declared no kvm_stats");
    assert_eq!(kernel_version, None, "fixture declared no kernel_version");
    assert_eq!(host, None, "fixture declared no host context");
    assert_eq!(timestamp, "", "fixture used empty-string timestamp");
    assert_eq!(run_id, "", "fixture used empty-string run_id");
    // Verdict bits — passed true + skipped false pinned.
    assert!(passed);
    assert!(!skipped, "fixture declared skipped=false");
    // Empty-Vec collections — regression guard against a serde
    // regression that dropped `[]` on round-trip.
    assert!(metrics.is_empty(), "fixture declared empty metrics");
    assert!(
        verifier_stats.is_empty(),
        "fixture declared empty verifier_stats",
    );
    assert!(sysctls.is_empty(), "fixture declared empty sysctls");
    assert!(kargs.is_empty(), "fixture declared empty kargs");
    // Populated nested structs.
    assert_eq!(stats.total_workers, 4);
    assert_eq!(stats.cgroups.len(), 1);
    assert_eq!(stats.cgroups[0].num_workers, 4);
    assert_eq!(stats.worst_spread, 20.0);
    let mon = monitor.unwrap();
    assert_eq!(mon.total_samples, 10);
    assert_eq!(mon.max_imbalance_ratio, 1.5);
    assert_eq!(mon.max_local_dsq_depth, 3);
    assert!(!mon.stuck_detected);
    let deltas = mon.event_deltas.unwrap();
    assert_eq!(deltas.total_fallback, 7);
    assert_eq!(deltas.total_dispatch_keep_last, 3);
    assert_eq!(stimulus_events.len(), 1);
    assert_eq!(stimulus_events[0].label, "StepStart[0]");
    assert_eq!(
        cleanup_duration_ms,
        Some(123),
        "cleanup_duration_ms round-tripped",
    );
    assert_eq!(
        run_source.as_deref(),
        Some(SIDECAR_RUN_SOURCE_LOCAL),
        "run_source must round-trip the literal `local` populated on \
         the write side, including the absent-vs-populated distinction",
    );
}

/// Exhaustive schema-audit gate for `SidecarResult`'s serde
/// round-trip. Every field is populated with a value that is
/// distinct from the `test_fixture` default AND every field is
/// asserted individually after serialization + deserialization.
/// A new field added to `SidecarResult` triggers failure at two
/// independent sites for `SidecarResult` top-level fields; nested
/// structs use `..Default::default()` and rely on their own
/// per-type tests:
/// 1. The construction literal below fails to compile (Rust
///    requires every field in a struct literal without
///    `..Default::default()`).
/// 2. The per-field assertion block below misses the new field,
///    so the audit surfaces as a reviewer note.
///
/// Nested struct literals inside the construction (e.g.
/// `MonitorSummary`, `ScenarioStats`, `HostContext`,
/// `PayloadMetrics`) use `..Default::default()` to remain
/// resilient to unrelated nested-type growth — adding a field
/// to one of those nested types does NOT trip this test. Fields
/// of those nested types that should trigger a similar audit
/// must grow their own all-fields round-trip test in their
/// owning module (e.g.
/// `host_context_populated_round_trips_via_json` for
/// `HostContext`).
///
/// Complements the structurally-populated
/// [`sidecar_result_roundtrip`] which exercises nested-struct
/// shapes but only asserts on a subset of fields. Leaving both
/// is intentional: the structural test proves deep trees survive
/// serde; this test proves every scalar and Option round-trips.
///
/// Distinct non-default values used:
/// - `test_name="audit"` (vs fixture `"t"`).
/// - `topology="8n8l16c2t"` (vs fixture `"1n1l1c1t"`).
/// - `scheduler="scx_audit"` (vs fixture `"eevdf"`).
/// - `work_type="AuditWork"` (vs fixture `"SpinWait"`).
/// - `passed=false, skipped=true` (vs fixture `true`, `false`).
/// - Non-empty collections for every `Vec<_>` field.
/// - `Some(…)` for every `Option<_>` field.
/// - Non-empty Strings for `timestamp`, `run_id`.
#[test]
fn sidecar_result_roundtrip_all_fields_round_trip() {
    use crate::assert::{CgroupStats, ScenarioStats};
    use crate::host_context::HostContext;
    use crate::monitor::MonitorSummary;
    use crate::monitor::bpf_prog::ProgVerifierStats;
    use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};
    use crate::timeline::StimulusEvent;

    let sc = SidecarResult {
        test_name: "audit".to_string(),
        topology: "8n8l16c2t".to_string(),
        scheduler: "scx_audit".to_string(),
        scheduler_commit: Some("deadbeef1234567890abcdef".to_string()),
        project_commit: Some("cafebab-dirty".to_string()),
        payload: Some("audit_payload".to_string()),
        metrics: vec![PayloadMetrics {
            payload_index: 0,
            metrics: vec![Metric {
                name: "audit_metric".to_string(),
                value: 42.0,
                polarity: Polarity::HigherBetter,
                unit: "audits".to_string(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            }],
            exit_code: 7,
        }],
        passed: false,
        skipped: true,
        stats: ScenarioStats {
            cgroups: vec![CgroupStats {
                num_workers: 3,
                ..Default::default()
            }],
            total_workers: 3,
            ..Default::default()
        },
        monitor: Some(MonitorSummary {
            total_samples: 17,
            ..Default::default()
        }),
        stimulus_events: vec![StimulusEvent {
            elapsed_ms: 123,
            label: "audit_event".to_string(),
            op_kind: None,
            detail: None,
            total_iterations: None,
        }],
        work_type: "AuditWork".to_string(),
        verifier_stats: vec![ProgVerifierStats {
            name: "audit_prog".to_string(),
            verified_insns: 999,
        }],
        kvm_stats: Some(crate::vmm::KvmStatsTotals::default()),
        sysctls: vec!["sysctl.kernel.audit_sysctl=1".to_string()],
        kargs: vec!["audit_karg".to_string()],
        kernel_version: Some("6.99.0".to_string()),
        kernel_commit: Some("kabcde7-dirty".to_string()),
        timestamp: "audit-timestamp".to_string(),
        run_id: "audit-run-id".to_string(),
        host: Some(HostContext {
            kernel_name: Some("AuditLinux".to_string()),
            ..Default::default()
        }),
        cleanup_duration_ms: Some(987),
        run_source: Some(SIDECAR_RUN_SOURCE_CI.to_string()),
    };

    let json = serde_json::to_string(&sc).expect("serialize");
    let loaded: SidecarResult = serde_json::from_str(&json).expect("deserialize");

    // Every field asserted, in struct-declaration order.
    assert_eq!(loaded.test_name, "audit");
    assert_eq!(loaded.topology, "8n8l16c2t");
    assert_eq!(loaded.scheduler, "scx_audit");
    assert_eq!(
        loaded.scheduler_commit.as_deref(),
        Some("deadbeef1234567890abcdef"),
        "scheduler_commit must round-trip the literal string \
         populated on the write side — not collapse to None via \
         a missing serde attribute or default fallback",
    );
    assert_eq!(
        loaded.project_commit.as_deref(),
        Some("cafebab-dirty"),
        "project_commit must round-trip the literal string \
         populated on the write side, including the `-dirty` \
         suffix that `detect_project_commit` appends — a \
         regression that stripped the suffix or substituted \
         None for a populated value would surface here. \
         Fixture uses 7-char hex (`cafebab`) to match the \
         `oid::to_hex_with_len(7)` shape `detect_project_commit` \
         produces in production.",
    );
    assert_eq!(loaded.payload.as_deref(), Some("audit_payload"));
    assert_eq!(loaded.metrics.len(), 1);
    assert_eq!(loaded.metrics[0].exit_code, 7);
    assert_eq!(loaded.metrics[0].metrics.len(), 1);
    assert_eq!(loaded.metrics[0].metrics[0].name, "audit_metric");
    assert_eq!(loaded.metrics[0].metrics[0].value, 42.0);
    assert!(!loaded.passed, "passed must survive as false");
    assert!(loaded.skipped, "skipped must survive as true");
    assert_eq!(loaded.stats.total_workers, 3);
    assert_eq!(loaded.stats.cgroups.len(), 1);
    assert_eq!(loaded.stats.cgroups[0].num_workers, 3);
    let mon = loaded.monitor.expect("monitor round-trips");
    assert_eq!(mon.total_samples, 17);
    assert_eq!(loaded.stimulus_events.len(), 1);
    assert_eq!(loaded.stimulus_events[0].label, "audit_event");
    assert_eq!(loaded.stimulus_events[0].elapsed_ms, 123);
    assert_eq!(loaded.work_type, "AuditWork");
    assert_eq!(loaded.verifier_stats.len(), 1);
    assert_eq!(loaded.verifier_stats[0].name, "audit_prog");
    assert_eq!(loaded.verifier_stats[0].verified_insns, 999);
    assert!(
        loaded.kvm_stats.is_some(),
        "kvm_stats must round-trip as Some"
    );
    assert_eq!(loaded.sysctls, vec!["sysctl.kernel.audit_sysctl=1"]);
    assert_eq!(loaded.kargs, vec!["audit_karg"]);
    assert_eq!(loaded.kernel_version.as_deref(), Some("6.99.0"));
    assert_eq!(
        loaded.kernel_commit.as_deref(),
        Some("kabcde7-dirty"),
        "kernel_commit must round-trip the literal string \
         populated on the write side, including the `-dirty` \
         suffix that `detect_kernel_commit` appends. Fixture \
         uses 7-char hex (`kabcde7`) to match the \
         `oid::to_hex_with_len(7)` shape `detect_kernel_commit` \
         produces in production. The leading `k` in the fixture \
         token makes a project_commit / kernel_commit field-swap \
         regression visible — each commit field carries a \
         distinct token in the audit fixture.",
    );
    assert_eq!(loaded.timestamp, "audit-timestamp");
    assert_eq!(loaded.run_id, "audit-run-id");
    let host = loaded.host.expect("host round-trips");
    assert_eq!(host.kernel_name.as_deref(), Some("AuditLinux"));
    assert_eq!(loaded.cleanup_duration_ms, Some(987));
    assert_eq!(
        loaded.run_source.as_deref(),
        Some(SIDECAR_RUN_SOURCE_CI),
        "run_source must round-trip the literal `ci` populated on \
         the write side. Audit fixture uses `ci` (vs `local` in \
         the sibling roundtrip) so a write-vs-read field-swap \
         regression that mapped one tag onto another would \
         surface in this audit pass even if the sibling test \
         did not detect it.",
    );
}

#[test]
fn sidecar_result_roundtrip_no_monitor() {
    let sc = SidecarResult {
        test_name: "eevdf_test".to_string(),
        topology: "1n1l2c1t".to_string(),
        passed: false,
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.test_name, "eevdf_test");
    assert!(!loaded.passed);
    assert!(loaded.monitor.is_none());
    assert!(loaded.stimulus_events.is_empty());
    // `monitor` is emitted as `"monitor":null` when absent — the
    // writer side guarantees full symmetry by always emitting
    // every field. (The reader side tolerates absence on `Option`
    // fields per serde's native rule; non-`Option` fields remain
    // hard-required.) Pinning the emission pattern prevents a
    // drift back to the old asymmetric `skip_serializing_if` form
    // that omitted None-produced fields entirely.
    assert!(
        json.contains("\"monitor\":null"),
        "monitor=None must serialize as `\"monitor\":null`, not be omitted: {json}",
    );
}

/// Strict-schema rejection for non-`Option` fields: a sidecar
/// JSON that omits any required (non-`Option`) top-level field
/// must fail deserialization, not silently default to the empty
/// string / empty Vec / similar. The SidecarResult policy —
/// `serde(default)` removed crate-wide, no `skip_serializing_if`
/// — is stated in the module doc; this test pins the parser-side
/// half by construction. A regression that reintroduces
/// `#[serde(default)]` on any non-`Option` SidecarResult field
/// would cause the `from_str` calls below to succeed instead of
/// error.
///
/// `Option` fields are deliberately excluded: serde's native
/// `Option<T>` deserialize rule treats absence as `None`, and
/// that tolerance is part of the asymmetric contract documented
/// at the module level — writer always emits, reader tolerates
/// absence on `Option`s. The sibling
/// `serialize_always_emits_option_keys` tests pin the writer
/// side; this loop pins the reader side for non-`Option` fields
/// only.
#[test]
fn sidecar_result_missing_required_field_rejected_by_deserialize() {
    // Table-driven expansion covering every non-`Option` field of
    // `SidecarResult`. Each must fail deserialize when absent with
    // a missing-field error naming the removed key.
    //
    // **Why Option fields are excluded**: serde treats
    // `Option<T>` as tolerant-of-absence natively (no explicit
    // `#[serde(default)]` needed — it's a builtin rule), so
    // removing e.g. `payload: Option<String>` from the JSON
    // yields `None` on the parsed struct rather than a rejection.
    // The module doc at src/test_support/sidecar.rs promises
    // "required on deserialize" for Option fields, but that's
    // enforced at the writer (always-emitted) side, not the
    // parser side. The `serialize_always_emits_option_keys`
    // sibling tests pin the writer half; this test pins the
    // parser-side strictness for every non-Option field.
    //
    // Old single-field-sentinel form (checking only `test_name`)
    // would pass silently if e.g. a regression added
    // `#[serde(default)]` to `run_id` alone — this loop catches
    // that class of softening across every non-Option field.
    const REQUIRED_NON_OPTION_FIELDS: &[&str] = &[
        "test_name",
        "topology",
        "scheduler",
        "metrics",
        "passed",
        "skipped",
        "stats",
        "stimulus_events",
        "work_type",
        "verifier_stats",
        "sysctls",
        "kargs",
        "timestamp",
        "run_id",
    ];

    let fixture = SidecarResult::test_fixture();
    let full = match serde_json::to_value(&fixture).unwrap() {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };

    for field in REQUIRED_NON_OPTION_FIELDS {
        let mut obj = full.clone();
        assert!(
            obj.remove(*field).is_some(),
            "SidecarResult test fixture must emit `{field}` for its \
             rejection case to be meaningful — the required-fields \
             list has drifted from the struct definition",
        );
        let json = serde_json::Value::Object(obj).to_string();
        let err = serde_json::from_str::<SidecarResult>(&json)
            .err()
            .unwrap_or_else(|| {
                panic!(
                    "deserialize must reject SidecarResult with `{field}` removed, \
                 but succeeded — a regression may have added \
                 `#[serde(default)]` to this field",
                )
            });
        let msg = format!("{err}");
        assert!(
            msg.contains(field),
            "missing-field error for `{field}` must name the field; got: {msg}",
        );
    }
}

/// Rename contract pin for the `source` → `run_source`
/// schema change. Per the doc on
/// [`SidecarResult::run_source`], no `#[serde(alias =
/// "source")]` is in place, so an archived sidecar carrying
/// the old `"source": "ci"` key deserializes to
/// `run_source: None` (serde silently drops the unknown
/// `"source"` field, then `Option<T>`'s "tolerate absence"
/// rule fires for the missing `"run_source"` key).
///
/// This is the documented data-loss behavior — pre-1.0
/// disposable schema, re-running the test regenerates the
/// sidecar under the new key. The test pins:
///
/// 1. Old key (`"source": "ci"`) → `run_source: None` (the
///    payload IS dropped, not preserved). A regression that
///    added `#[serde(alias = "source")]` would surface here
///    as `Some("ci")`.
/// 2. New key (`"run_source": "ci"`) → `Some("ci")` (the
///    canonical deserialize path under the post-rename
///    schema). A regression that broke the new-key path
///    would surface here as `None`.
/// 3. Old key + new key both present → new key wins (sanity
///    check that the rename did not silently route the new
///    key through the old field's deserialize logic). Pins
///    the post-rename canonical-key precedence.
#[test]
fn sidecar_result_rename_contract_old_source_key_lands_run_source_none() {
    let fixture = SidecarResult::test_fixture();
    let full = match serde_json::to_value(&fixture).unwrap() {
        serde_json::Value::Object(m) => m,
        other => panic!("expected object, got {other:?}"),
    };

    // Arm 1: old `"source"` key only — the new schema has
    // no alias, so this is the documented data-loss path.
    let mut obj_old = full.clone();
    obj_old.remove("run_source");
    obj_old.insert(
        "source".to_string(),
        serde_json::Value::String("ci".to_string()),
    );
    let json_old = serde_json::Value::Object(obj_old).to_string();
    let parsed_old: SidecarResult = serde_json::from_str(&json_old).expect(
        "old-key sidecar must still deserialize — \
         SidecarResult does not set deny_unknown_fields, \
         so the unrecognised `\"source\"` key is silently dropped",
    );
    assert_eq!(
        parsed_old.run_source, None,
        "old `\"source\": \"ci\"` key must land run_source = None \
         per the documented data-loss contract; a regression that \
         added `#[serde(alias = \"source\")]` would yield Some(\"ci\") here",
    );

    // Arm 2: new `"run_source"` key only — the canonical
    // post-rename deserialize path.
    let mut obj_new = full.clone();
    obj_new.insert(
        "run_source".to_string(),
        serde_json::Value::String("ci".to_string()),
    );
    let json_new = serde_json::Value::Object(obj_new).to_string();
    let parsed_new: SidecarResult =
        serde_json::from_str(&json_new).expect("new-key sidecar must deserialize cleanly");
    assert_eq!(
        parsed_new.run_source.as_deref(),
        Some("ci"),
        "new `\"run_source\": \"ci\"` key must populate \
         run_source — a regression breaking the new-key path \
         would yield None here",
    );

    // Arm 3: BOTH keys present — the new key wins because
    // the old `"source"` is unknown and silently dropped.
    // Pins that the rename did not accidentally route the
    // new key through the old field's logic (which would
    // make this case ambiguous).
    let mut obj_both = full.clone();
    obj_both.insert(
        "run_source".to_string(),
        serde_json::Value::String("ci".to_string()),
    );
    obj_both.insert(
        "source".to_string(),
        serde_json::Value::String("local".to_string()),
    );
    let json_both = serde_json::Value::Object(obj_both).to_string();
    let parsed_both: SidecarResult =
        serde_json::from_str(&json_both).expect("both-keys sidecar must deserialize cleanly");
    assert_eq!(
        parsed_both.run_source.as_deref(),
        Some("ci"),
        "with both keys present, new `\"run_source\"` must win \
         — the old `\"source\"` is silently dropped, NOT used \
         as a fallback. A regression that processed `\"source\"` \
         as an alias would surface here as Some(\"local\")",
    );
}

// -- collect_sidecars tests --

#[test]
fn collect_sidecars_empty_dir() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let results = collect_sidecars(tmp_dir.path());
    assert!(results.is_empty());
}

#[test]
fn collect_sidecars_nonexistent_dir() {
    let results = collect_sidecars(std::path::Path::new("/nonexistent/path"));
    assert!(results.is_empty());
}

#[test]
fn collect_sidecars_reads_json() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let sc = SidecarResult {
        test_name: "test_x".to_string(),
        topology: "1n1l2c1t".to_string(),
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    std::fs::write(tmp.join("test_x.ktstr.json"), &json).unwrap();
    // Non-ktstr JSON should be ignored.
    std::fs::write(tmp.join("other.json"), r#"{"key":"val"}"#).unwrap();
    let results = collect_sidecars(tmp);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].test_name, "test_x");
}

#[test]
fn collect_sidecars_recurses_one_level() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let sub = tmp.join("job-0");
    std::fs::create_dir_all(&sub).unwrap();
    let sc = SidecarResult {
        test_name: "nested_test".to_string(),
        topology: "1n2l4c2t".to_string(),
        scheduler: "scx_mitosis".to_string(),
        passed: false,
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    std::fs::write(sub.join("nested_test.ktstr.json"), &json).unwrap();
    let results = collect_sidecars(tmp);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].test_name, "nested_test");
    assert!(!results[0].passed);
}

#[test]
fn collect_sidecars_does_not_recurse_past_one_level() {
    // Companion to `collect_sidecars_recurses_one_level`: pin the
    // "exactly one level, no deeper" contract. A sidecar two
    // directories deep must be ignored. If a future change
    // switches collect_sidecars to a depth-unbounded walk, this
    // test catches the schema-scope regression before stats
    // tooling starts double-counting results from unrelated
    // sub-runs under the same `runs_root`.
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let top_sub = tmp.join("job-0");
    let deep_sub = top_sub.join("replay-0");
    std::fs::create_dir_all(&deep_sub).unwrap();

    let sc = |name: &str| SidecarResult {
        test_name: name.to_string(),
        ..SidecarResult::test_fixture()
    };
    // One level: should be collected.
    std::fs::write(
        top_sub.join("top_level.ktstr.json"),
        serde_json::to_string(&sc("top_level")).unwrap(),
    )
    .unwrap();
    // Two levels: must NOT be collected.
    std::fs::write(
        deep_sub.join("deep_level.ktstr.json"),
        serde_json::to_string(&sc("deep_level")).unwrap(),
    )
    .unwrap();

    let results = collect_sidecars(tmp);
    let names: Vec<&str> = results.iter().map(|r| r.test_name.as_str()).collect();
    assert_eq!(
        names,
        vec!["top_level"],
        "collect_sidecars must see only the one-level-deep sidecar, not the two-level one"
    );
}

#[test]
fn collect_sidecars_skips_invalid_json() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    std::fs::write(tmp.join("bad.ktstr.json"), "not json").unwrap();
    let results = collect_sidecars(tmp);
    assert!(results.is_empty());
}

#[test]
fn collect_sidecars_skips_non_ktstr_json() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    // File ends in .json but does NOT contain ".ktstr." in the name
    std::fs::write(tmp.join("other.json"), r#"{"test":"val"}"#).unwrap();
    let results = collect_sidecars(tmp);
    assert!(results.is_empty());
}

#[test]
fn sidecar_result_work_type_field() {
    let sc = SidecarResult {
        work_type: "Bursty".to_string(),
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.work_type, "Bursty");
}

#[test]
fn write_sidecar_defaults_to_target_dir_without_env() {
    let _lock = lock_env();
    let target_dir = tempfile::TempDir::new().unwrap();
    let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
    let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
    let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");

    let dir = sidecar_dir();
    // Expected layout: `{CARGO_TARGET_DIR}/ktstr/{kernel}-{project_commit}`.
    // `KTSTR_KERNEL` is unset so kernel resolves to `"unknown"`.
    // `{project_commit}` is whatever `detect_project_commit()`
    // resolves on this machine (`Some(hex7)` when cwd is inside
    // a git repo, `None` -> `"unknown"` otherwise). Compute the
    // expected via `runs_root` + `format_run_dirname` so the
    // assertion matches the production path symmetrically and
    // does not depend on the cwd's git state.
    let kernel = detect_kernel_version();
    let commit = detect_project_commit();
    let expected = runs_root().join(format_run_dirname(kernel.as_deref(), commit.as_deref()));
    assert_eq!(dir, expected);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__sidecar_default_dir__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let check_result = AssertResult::pass();
    write_sidecar(&entry, &vm_result, &[], &check_result, "SpinWait", &[]).unwrap();

    // The actual on-disk filename embeds a variant-hash suffix
    // (see `serialize_and_write_sidecar`), so a fixed
    // `test_name + ".ktstr.json"` path never matches — use the
    // prefix-scan helper the sibling tests use. The tempdir's
    // Drop wipes everything when this scope ends, so no manual
    // cleanup is required.
    let paths = find_sidecars_by_prefix(&dir, "__sidecar_default_dir__-");
    // One call to `write_sidecar` above must produce exactly
    // one sidecar under this test's unique prefix. A count
    // above 1 exposes either a variant-hash collision (two
    // distinct test_name + variant-hash pairs hashing to the
    // same filename suffix) or a regression in
    // `pre_clear_run_dir_once` (which is now keyed per-directory
    // via `Mutex<HashSet<PathBuf>>` — every distinct dir gets
    // exactly one pre-clear per process — so a stale file from
    // a prior crashed run should be wiped on the very first
    // call into this dir, regardless of which test runs first).
    assert_eq!(
        paths.len(),
        1,
        "single `write_sidecar` call against prefix \
         `__sidecar_default_dir__-` must produce exactly one \
         file; got {} ({paths:?}). If >1, either the variant \
         hash collided for this test's variant-field tuple or \
         `pre_clear_run_dir_once`'s per-directory keying failed \
         to wipe a stale sidecar from a prior crashed run.",
        paths.len(),
    );
}

// -- KTSTR_SIDECAR_DIR override: empty-string falls back to default --

/// `KTSTR_SIDECAR_DIR=""` (defensively-cleared empty string)
/// must NOT activate the override branch — `sidecar_dir`
/// must compute the default
/// `runs_root().join({kernel}-{project_commit})` path instead
/// of returning an empty path. Pins the
/// `is_empty()` filter on the override read in
/// [`sidecar_dir_override`]: a regression that dropped the
/// filter (e.g. simplified to `std::env::var("...").ok().map(PathBuf::from)`)
/// would surface here as `sidecar_dir()` returning `PathBuf::from("")`
/// — a path that joins onto runs-root as a no-op alias and
/// silently contaminates the runs listing.
///
/// The override branch SHORT-CIRCUITS on a non-empty value
/// (returns the override verbatim, skipping the format-run-dirname
/// computation), so the assertion below — comparing
/// `sidecar_dir()` against the manually-computed default — is
/// proof that the empty-string DID NOT take the short-circuit
/// path. A regression that activated the override on empty
/// would surface as `dir == PathBuf::from("")`, not equal to
/// the computed default.
#[test]
fn sidecar_dir_empty_override_falls_back_to_default() {
    let _lock = lock_env();
    let target_dir = tempfile::TempDir::new().unwrap();
    let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
    // EnvVarGuard::set with an empty path covers the
    // defensively-cleared `KTSTR_SIDECAR_DIR=""` operator
    // pattern. EnvVarGuard accepts AsRef<OsStr>, and a
    // zero-length `&str` ("") satisfies that bound.
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", "");
    let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");

    let dir = sidecar_dir();
    // Compute the expected default the same way `sidecar_dir`
    // does on its default branch. With KTSTR_KERNEL unset the
    // kernel resolves to "unknown"; commit comes from the
    // OnceLock-cached project probe (Some(hash) when running
    // inside the ktstr repo).
    let kernel = detect_kernel_version();
    let commit = detect_project_commit();
    let expected = runs_root().join(format_run_dirname(kernel.as_deref(), commit.as_deref()));
    assert_eq!(
        dir, expected,
        "empty KTSTR_SIDECAR_DIR must fall back to the default \
         `runs_root().join(format_run_dirname(...))` path, NOT \
         return PathBuf::from(\"\"). A regression that dropped \
         the `is_empty()` filter on the override read would \
         surface here as `dir == PathBuf::from(\"\")`.",
    );
    assert_ne!(
        dir,
        std::path::PathBuf::new(),
        "sidecar_dir must never return an empty path",
    );
}

// -- format_run_dirname (pure function, no OnceLock dependency) --

/// Clean commit shape: `{kernel}-{hex7}` — the standard happy
/// path. Pinning the format here means a regression that adds
/// extra punctuation, swaps the order, or drops a component
/// surfaces as a unit-test failure rather than as a downstream
/// stats-tooling miss.
#[test]
fn format_run_dirname_clean_commit() {
    assert_eq!(
        format_run_dirname(Some("6.14.2"), Some("abc1234")),
        "6.14.2-abc1234",
        "clean dirname must be `{{kernel}}-{{project_commit}}`",
    );
}

/// Dirty commit shape: the `-dirty` suffix flows through verbatim
/// because `format_run_dirname` does not interpret the commit
/// string — it simply joins. The suffix is appended upstream by
/// `commit_with_dirty_suffix`. This test pins the verbatim
/// pass-through.
#[test]
fn format_run_dirname_dirty_commit() {
    assert_eq!(
        format_run_dirname(Some("6.14.2"), Some("abc1234-dirty")),
        "6.14.2-abc1234-dirty",
        "dirty dirname must pass the `-dirty` suffix through verbatim",
    );
}

/// Missing commit (non-git cwd or probe failure) collapses to
/// the literal `"unknown"` sentinel in the commit slot, so the
/// dirname is `{kernel}-unknown`. This is the documented
/// dirname-vs-JSON asymmetry: in-memory the
/// `SidecarResult::project_commit` field stays `None`, but the
/// dirname uses a filesystem-safe sentinel.
#[test]
fn format_run_dirname_unknown_commit() {
    assert_eq!(
        format_run_dirname(Some("6.14.2"), None),
        "6.14.2-unknown",
        "missing commit must collapse to `{{kernel}}-unknown` sentinel",
    );
}

/// Missing kernel mirrors the missing-commit shape: `unknown-{project_commit}`.
/// Captures the `KTSTR_KERNEL` unset / detection-failed path
/// so a regression in the unwrap_or fallback surfaces here.
#[test]
fn format_run_dirname_unknown_kernel() {
    assert_eq!(
        format_run_dirname(None, Some("abc1234")),
        "unknown-abc1234",
        "missing kernel must collapse to `unknown-{{project_commit}}` sentinel",
    );
}

/// Both components missing: every run from a non-git cwd with no
/// `KTSTR_KERNEL` set lands in the same `unknown-unknown`
/// directory. Documented collision: the operator must set
/// `KTSTR_SIDECAR_DIR` or place the project tree under git to
/// disambiguate concurrent test runs.
#[test]
fn format_run_dirname_both_unknown_collide() {
    assert_eq!(
        format_run_dirname(None, None),
        "unknown-unknown",
        "both-missing case must produce `unknown-unknown` — the documented \
         collision the operator must disambiguate via KTSTR_SIDECAR_DIR or git",
    );
}

// -- pre_clear_run_dir_once tests --
//
// Pin the four behavioral invariants the doc on
// `pre_clear_run_dir_once` claims:
// 1. *.ktstr.json files in the immediate dir are removed.
// 2. Subdirectories and non-sidecar files are left untouched.
// 3. A missing dir is silent (no panic).
// 4. Per-directory keying via Mutex<HashSet<PathBuf>>: a second
//    call for the SAME dir is a no-op, but a call for a NEW dir
//    fires its own pre-clear.
//
// Each test uses a fresh tempdir so the per-process cache never
// collides across tests; tests do NOT need `lock_env` because
// they do not touch any environment variable — pre_clear is
// env-independent.

/// `pre_clear_run_dir_once` removes every `*.ktstr.json` file in
/// the immediate directory on its first call against that dir.
/// Pins the wipe-on-first-call invariant.
#[test]
fn pre_clear_run_dir_once_wipes_existing_sidecars() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    std::fs::write(tmp.join("test_a-0000.ktstr.json"), b"{}").unwrap();
    std::fs::write(tmp.join("test_b-1111.ktstr.json"), b"{}").unwrap();
    assert_eq!(
        std::fs::read_dir(tmp).unwrap().count(),
        2,
        "fixture precondition: tempdir must contain two sidecars",
    );

    pre_clear_run_dir_once(tmp);

    let remaining: Vec<_> = std::fs::read_dir(tmp)
        .unwrap()
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert!(
        remaining.is_empty(),
        "every *.ktstr.json file must be wiped; got {remaining:?}",
    );
}

/// `pre_clear_run_dir_once` does NOT recurse — subdirectories
/// and any non-sidecar files in the immediate dir are left
/// untouched. Pins the shallow-scope invariant: an external
/// orchestrator that writes per-job subdirectories under the
/// run dir does not lose its fixture state to a sibling
/// invocation's pre-clear.
#[test]
fn pre_clear_run_dir_once_skips_subdirs_and_non_sidecars() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    // Top-level sidecar: should be wiped.
    std::fs::write(tmp.join("victim-0000.ktstr.json"), b"{}").unwrap();
    // Top-level non-sidecar files: should survive.
    std::fs::write(tmp.join("README.md"), b"keep").unwrap();
    std::fs::write(tmp.join("other.json"), b"{}").unwrap();
    std::fs::write(tmp.join("partial.ktstr.json.tmp"), b"{}").unwrap();
    // Subdirectory with a sidecar inside: subdir AND its
    // contents should survive (pre-clear does not recurse).
    let sub = tmp.join("job-1");
    std::fs::create_dir(&sub).unwrap();
    std::fs::write(sub.join("nested-0000.ktstr.json"), b"{}").unwrap();

    pre_clear_run_dir_once(tmp);

    assert!(
        !tmp.join("victim-0000.ktstr.json").exists(),
        "top-level *.ktstr.json file must be wiped",
    );
    assert!(
        tmp.join("README.md").exists(),
        "non-sidecar file must survive",
    );
    assert!(
        tmp.join("other.json").exists(),
        "bare *.json (no .ktstr. infix) must survive",
    );
    assert!(
        tmp.join("partial.ktstr.json.tmp").exists(),
        "non-`.json` extension must survive even with .ktstr. infix",
    );
    assert!(sub.exists(), "subdirectory must survive");
    assert!(
        sub.join("nested-0000.ktstr.json").exists(),
        "sidecar inside subdirectory must survive (pre-clear is shallow)",
    );
}

/// `pre_clear_run_dir_once` is silent when the target directory
/// does not yet exist — `read_dir` errors are swallowed. Pins
/// the helper's API contract that a missing dir is a no-op
/// rather than a panic. The production caller
/// (`serialize_and_write_sidecar`) materializes the dir via
/// `create_dir_all` BEFORE feeding it to this helper, so the
/// missing-dir branch is unreachable in production today; the
/// invariant is preserved for defensive correctness against
/// future direct callers and to keep the helper safe to call
/// from unit tests that probe the missing-dir edge.
#[test]
fn pre_clear_run_dir_once_silent_on_missing_dir() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let nonexistent = tmp_dir.path().join("does_not_exist_yet");
    assert!(
        !nonexistent.exists(),
        "fixture precondition: dir must not exist"
    );

    // Must not panic. The function returns `()`, so the only
    // observable failure mode is a panic — the call returning
    // normally is the test's pass condition.
    pre_clear_run_dir_once(&nonexistent);

    // Sanity: pre_clear must not have created the dir as a
    // side effect either. `serialize_and_write_sidecar`'s
    // create_dir_all is the only path that materializes the
    // directory.
    assert!(
        !nonexistent.exists(),
        "pre_clear must not create the dir as a side effect",
    );
}

/// `pre_clear_run_dir_once` reaps orphaned atomic-write staging
/// files in the same shallow sweep as live sidecars. Pins the
/// staging-cleanup invariant documented at
/// `pre_clear_run_dir_once`'s in-line comment block (line ~2317):
/// a writer that died between `write` and `rename` in
/// `serialize_and_write_sidecar` leaves a
/// `<test>-<hash>.ktstr.json.tmp.<pid>.<run_id>` artifact;
/// `is_sidecar_filename` rejects these (extension is `<run_id>`,
/// not `json`), so without the staging sweep neither
/// `collect_sidecars` nor the next pre-clear pass would ever
/// reap them. A regression that drops the
/// `is_sidecar_staging_filename` arm would surface here as the
/// `.tmp.…` files surviving the pre-clear call.
#[test]
fn pre_clear_run_dir_once_wipes_staging_files() {
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    // Two orphaned staging files in the canonical
    // `<test>-<hash>.ktstr.json.tmp.<pid>.<run_id>` shape.
    std::fs::write(
        tmp.join("test_a-0000000000000000.ktstr.json.tmp.12345.0001"),
        b"{}",
    )
    .unwrap();
    std::fs::write(
        tmp.join("test_b-1111111111111111.ktstr.json.tmp.67890.0002"),
        b"{}",
    )
    .unwrap();
    // One live sidecar from a prior run that should also be
    // wiped — the same sweep handles both shapes so the test
    // exercises the combined cleanup.
    std::fs::write(tmp.join("test_c-2222222222222222.ktstr.json"), b"{}").unwrap();
    // One unrelated file that must survive — guards against an
    // overly-broad sweep that wipes anything containing the
    // `.ktstr.` infix regardless of suffix shape.
    std::fs::write(tmp.join("README.md"), b"keep").unwrap();
    assert_eq!(
        std::fs::read_dir(tmp).unwrap().count(),
        4,
        "fixture precondition: tempdir must contain 2 staging \
         files, 1 sidecar, and 1 unrelated file",
    );

    pre_clear_run_dir_once(tmp);

    assert!(
        !tmp.join("test_a-0000000000000000.ktstr.json.tmp.12345.0001")
            .exists(),
        "orphaned staging file (writer crash before rename) must be reaped",
    );
    assert!(
        !tmp.join("test_b-1111111111111111.ktstr.json.tmp.67890.0002")
            .exists(),
        "second orphaned staging file must be reaped",
    );
    assert!(
        !tmp.join("test_c-2222222222222222.ktstr.json").exists(),
        "live sidecar from prior run must also be wiped in the same sweep",
    );
    assert!(
        tmp.join("README.md").exists(),
        "unrelated file must survive — sweep is shape-scoped, \
         not a wholesale directory wipe",
    );
}

/// Per-directory keying via `Mutex<HashSet<PathBuf>>`: a second
/// call against the SAME dir is a no-op (newly-written sidecars
/// after the first pre-clear must NOT be wiped on the second
/// call), but a call against a DIFFERENT dir fires its own
/// pre-clear. Pins both halves of the per-dir contract:
/// idempotent for repeats, fresh for novel paths.
///
/// The two tempdirs share a process-global `OnceLock<Mutex<HashSet<...>>>`,
/// so the test order is incidental — what matters is that the
/// HashSet has separate entries per dir.
#[test]
fn pre_clear_run_dir_once_keys_per_directory() {
    let tmp_a = tempfile::TempDir::new().unwrap();
    let tmp_b = tempfile::TempDir::new().unwrap();

    // Phase 1: prime dir A. Populate with a sidecar, call
    // pre_clear, verify wiped. The HashSet now contains A's
    // canonicalized path.
    std::fs::write(tmp_a.path().join("a-0000.ktstr.json"), b"{}").unwrap();
    pre_clear_run_dir_once(tmp_a.path());
    assert!(
        !tmp_a.path().join("a-0000.ktstr.json").exists(),
        "first call against A must wipe A's sidecar",
    );

    // Phase 2: write a new sidecar to A (modeling the writer
    // populating the dir AFTER pre-clear), then call pre_clear
    // against A again. The cache hit must short-circuit the
    // wipe — the new sidecar must SURVIVE.
    std::fs::write(tmp_a.path().join("a-1111.ktstr.json"), b"{}").unwrap();
    pre_clear_run_dir_once(tmp_a.path());
    assert!(
        tmp_a.path().join("a-1111.ktstr.json").exists(),
        "second call against A must be a no-op (cache hit) — \
         the post-prime sidecar must survive. A regression to \
         OnceLock<()> or a HashSet that ignores the key would \
         leak this assertion.",
    );

    // Phase 3: prime dir B (new path). The HashSet has no
    // entry for B yet, so this call must wipe B's sidecar
    // — proving the cache distinguishes paths rather than
    // collapsing every call after the first.
    std::fs::write(tmp_b.path().join("b-0000.ktstr.json"), b"{}").unwrap();
    pre_clear_run_dir_once(tmp_b.path());
    assert!(
        !tmp_b.path().join("b-0000.ktstr.json").exists(),
        "first call against B must wipe B's sidecar — proves the \
         per-dir keying distinguishes A from B (a OnceLock<()> \
         that fired once for A would leak this assertion).",
    );
}

// -- warn_unknown_project_commit_inner tests --
//
// Pin the three behavioral invariants the inner helper exposes:
// 1. Calling once writes the warning text to the sink.
// 2. The emitted text contains the operator-actionable substring
//    pointing at `KTSTR_SIDECAR_DIR` so a future doc-drift on the
//    warning prose surfaces here rather than silently changing
//    operator-facing remediation.
// 3. A second call against the SAME `OnceLock<()>` is a no-op —
//    the second call must NOT append additional bytes to the sink.
//
// Each test owns a local `OnceLock<()>` so the tests are
// independent of any other test (or the production wrapper) that
// might already have initialized the process-global gate. No
// `lock_env` needed: the inner helper does not touch any env var
// or any shared global state beyond the gate the caller supplies.

/// First call against a fresh `OnceLock<()>` writes the warning
/// text to the sink. Pins the emit-once invariant on initial
/// invocation and proves the inner helper emits via the
/// caller-provided sink rather than fd 2.
#[test]
fn warn_unknown_project_commit_inner_emits_on_first_call() {
    let gate = std::sync::OnceLock::new();
    let mut sink: Vec<u8> = Vec::new();
    warn_unknown_project_commit_inner(&gate, &mut sink);
    assert!(
        !sink.is_empty(),
        "first call must emit bytes to the sink; got empty",
    );
}

/// Pin the operator-actionable substring of the warning. The
/// test does NOT pin the entire prose verbatim — that would
/// make every wording tweak break here — but it DOES pin the
/// single load-bearing remediation hint (`KTSTR_SIDECAR_DIR`)
/// so a future edit that drops the recommended env var loses
/// this assertion. The `WARNING:` marker is also pinned so a
/// downgrade from warning to info changes the severity tag
/// observably.
#[test]
fn warn_unknown_project_commit_inner_emits_expected_substring() {
    let gate = std::sync::OnceLock::new();
    let mut sink: Vec<u8> = Vec::new();
    warn_unknown_project_commit_inner(&gate, &mut sink);
    let captured = String::from_utf8(sink).expect("warning text must be UTF-8");
    assert!(
        captured.contains("WARNING:"),
        "warning must carry the WARNING severity tag; got: {captured:?}",
    );
    assert!(
        captured.contains("KTSTR_SIDECAR_DIR"),
        "warning must reference KTSTR_SIDECAR_DIR as the remediation \
         knob — operators rely on this hint to disambiguate \
         non-git runs; got: {captured:?}",
    );
}

/// A second call against the SAME `OnceLock<()>` is a no-op —
/// the gate has already been initialized by the first call, so
/// `get_or_init`'s closure does not fire and no additional bytes
/// land in the sink. Pins the once-per-gate contract that
/// gauntlet variants rely on (otherwise the operator would see
/// thousands of duplicate warnings interleaved with test output).
///
/// The assertion compares the sink's length AFTER the second
/// call against its length AFTER the first call. A regression
/// that re-fires the warning would extend the sink and break
/// this equality.
#[test]
fn warn_unknown_project_commit_inner_second_call_is_no_op() {
    let gate = std::sync::OnceLock::new();
    let mut sink: Vec<u8> = Vec::new();
    warn_unknown_project_commit_inner(&gate, &mut sink);
    let after_first = sink.len();
    assert!(
        after_first > 0,
        "fixture precondition: first call must emit bytes",
    );
    warn_unknown_project_commit_inner(&gate, &mut sink);
    assert_eq!(
        sink.len(),
        after_first,
        "second call against the same gate must NOT append bytes — \
         the OnceLock<()> gating is the load-bearing invariant; got \
         len {} (expected {after_first})",
        sink.len(),
    );
}

// -- newest_run_dir tests --
//
// Pin the dotfile filter so the flock sentinel subdirectory
// (`.locks/`) cannot eclipse a real run dir as the "most
// recent run" — `.locks/`'s mtime tracks per-write flock
// activity and would otherwise advance past the run dir's
// own mtime on the most recent sidecar write, claiming the
// newest-run bucket.

/// `newest_run_dir` must pick a real run directory in
/// preference to a NEWER `.locks/` directory at the same
/// runs root. Mtime ordering is stamped via filesystem
/// create order with a sleep between calls so the test
/// deterministically distinguishes "newer .locks ignored"
/// from "older real run picked up because it happened to
/// have the largest mtime."
#[test]
fn newest_run_dir_skips_dotfile_subdirectories() {
    use std::thread::sleep;
    use std::time::Duration;
    let _lock = lock_env();
    let target_dir = tempfile::TempDir::new().unwrap();
    let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
    // `runs_root()` returns `{CARGO_TARGET_DIR}/ktstr/`, so
    // create that intermediate before populating run subdirs.
    let runs = target_dir.path().join("ktstr");
    std::fs::create_dir(&runs).expect("mkdir runs root");
    // Real run dir created first, so its mtime is OLDER.
    let real = runs.join("real-run");
    std::fs::create_dir(&real).expect("mkdir real run dir");
    sleep(Duration::from_millis(50));
    // .locks/ created second, so its mtime is NEWER. Without
    // the dotfile filter, this entry would win the
    // max_by_key contest and `newest_run_dir` would return
    // `.locks/` — the regression that this test guards.
    std::fs::create_dir(runs.join(".locks")).expect("mkdir .locks");
    let got = newest_run_dir().expect("non-empty runs root must yield Some");
    assert_eq!(
        got, real,
        "newest_run_dir must pick the real run dir even when \
         .locks/ has a newer mtime — a regression that drops \
         the dotfile filter would surface here as `.locks/` \
         winning the mtime contest",
    );
}

/// `newest_run_dir` returns `None` when only dotfile-prefixed
/// subdirectories exist under the runs root. Pins the
/// post-filter empty case: even if the runs root itself is
/// non-empty, a fresh repo state (only `.locks/` lives there
/// because no test has ever produced a sidecar) must not
/// surface `.locks/` as a stand-in run.
#[test]
fn newest_run_dir_yields_none_when_only_dotfiles_exist() {
    let _lock = lock_env();
    let target_dir = tempfile::TempDir::new().unwrap();
    let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
    let runs = target_dir.path().join("ktstr");
    std::fs::create_dir(&runs).expect("mkdir runs root");
    std::fs::create_dir(runs.join(".locks")).expect("mkdir .locks");
    std::fs::create_dir(runs.join(".cache")).expect("mkdir .cache");
    let got = newest_run_dir();
    assert!(
        got.is_none(),
        "runs root with only dotfile subdirs must yield None; got {got:?}",
    );
}

// -- is_run_directory predicate tests --
//
// Direct unit tests over the predicate that backs both
// `newest_run_dir` and `sorted_run_entries`'s filter. Pure
// shape contract, no I/O beyond a tempdir to materialize
// DirEntries the predicate can consume.

/// A regular subdirectory whose name does not start with `.`
/// passes the predicate.
#[test]
fn is_run_directory_accepts_non_dotfile_subdir() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("real-run")).unwrap();
    let entry = std::fs::read_dir(tmp.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert!(
        super::is_run_directory(&entry),
        "non-dotfile subdir must be accepted",
    );
}

/// A subdirectory whose name starts with `.` is rejected.
/// Pins the dotfile filter — the load-bearing rule for the
/// `.locks/` exclusion.
#[test]
fn is_run_directory_rejects_dotfile_subdir() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".locks")).unwrap();
    let entry = std::fs::read_dir(tmp.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert!(
        !super::is_run_directory(&entry),
        "dotfile subdir must be rejected",
    );
}

/// A regular file (not a directory) is rejected, regardless
/// of name — the `is_dir()` short-circuit must precede the
/// dotfile check.
#[test]
fn is_run_directory_rejects_regular_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("regular-file"), b"x").unwrap();
    let entry = std::fs::read_dir(tmp.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    assert!(
        !super::is_run_directory(&entry),
        "regular file must be rejected",
    );
}

// -- run_dir_lock_path / acquire_run_dir_flock tests --
//
// Pin the cross-process flock contract added for the
// concurrent-write collision fix:
//
// 1. `run_dir_lock_path` derives the canonical
//    `{parent}/.locks/{leaf}.lock` shape so two callers
//    keying off the same `dir` agree on the lockfile.
// 2. `acquire_run_dir_flock_with_timeout` materializes the
//    parent `.locks/` subdirectory on first call and returns
//    an `OwnedFd` whose Drop releases the lock — so a second
//    call after the first returns can acquire successfully.
// 3. While a peer holds `LOCK_EX` on the lockfile, the helper
//    times out with an actionable error. (Different `OwnedFd`s
//    in the same process are distinct OFDs, so flock(2)
//    serializes them the same way it would two processes —
//    no fork required to exercise the contention path.)
//
// No `lock_env` needed: the helpers don't touch any env var.
// Each test owns a tempdir so the per-test lockfile namespace
// is isolated.

/// `run_dir_lock_path({parent}/{key})` returns
/// `{parent}/.locks/{key}.lock`. Pins the layout so a future
/// edit to [`crate::flock::LOCK_DIR_NAME`] or the join shape
/// surfaces here rather than as a silent cross-call divergence.
#[test]
fn run_dir_lock_path_returns_expected_shape() {
    let dir = std::path::Path::new("/runs-root/6.14.2-deadbee");
    let lock = super::run_dir_lock_path(dir).expect("non-root dir must yield Some");
    assert_eq!(
        lock,
        std::path::PathBuf::from("/runs-root/.locks/6.14.2-deadbee.lock"),
    );
}

/// A path with no parent (root `/`) has no canonical lockfile
/// location — the helper returns `None` rather than constructing
/// an unsafe sentinel. Pins the defensive arm so a regression
/// that unwraps `parent()` surfaces here.
#[test]
fn run_dir_lock_path_no_parent_returns_none() {
    let lock = super::run_dir_lock_path(std::path::Path::new("/"));
    assert!(
        lock.is_none(),
        "root path must yield None (no parent), got {lock:?}",
    );
}

/// First call against a fresh `dir` materializes the parent
/// `.locks/` subdirectory on demand and returns an `OwnedFd`
/// holding `LOCK_EX`. The lockfile itself persists after the
/// fd is dropped (only the kernel-side lock is released);
/// that's what `try_flock`'s own contract guarantees.
#[test]
fn acquire_run_dir_flock_creates_locks_subdir_lazily() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("6.14.2-deadbee");
    // `acquire_run_dir_flock_with_timeout` doesn't require the
    // run-dir itself to exist — the production caller
    // materializes it via `create_dir_all` BEFORE this point.
    // Mirror that pattern.
    std::fs::create_dir_all(&dir).unwrap();

    let fd = super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_secs(1))
        .expect("first acquire must succeed against an uncontended dir");
    assert!(
        tmp.path().join(".locks").exists(),
        ".locks/ subdirectory must be created lazily on first acquire",
    );
    assert!(
        tmp.path().join(".locks/6.14.2-deadbee.lock").exists(),
        "lockfile must exist on disk after acquire",
    );
    // Drop the fd — releases the kernel-side flock. The
    // sentinel file persists (released, but not unlinked).
    drop(fd);
    assert!(
        tmp.path().join(".locks/6.14.2-deadbee.lock").exists(),
        "lockfile sentinel must persist after fd drop — \
         try_flock's contract is fd-bound release, not file unlink",
    );
}

/// A second `acquire_run_dir_flock_with_timeout` against the
/// same dir AFTER the first fd was dropped must succeed —
/// proves the kernel-side release happens via `OwnedFd::drop`
/// (no leaked OFD blocking subsequent acquires).
#[test]
fn acquire_run_dir_flock_releases_on_drop() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("key");
    std::fs::create_dir_all(&dir).unwrap();

    let fd1 = super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_secs(1))
        .expect("first acquire");
    drop(fd1);
    let fd2 = super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_secs(1))
        .expect(
            "second acquire after drop must succeed — a regression that \
         fails to release the kernel flock on OwnedFd::drop would \
         leak this assertion",
        );
    drop(fd2);
}

/// While a peer holds `LOCK_EX` on the same dir's lockfile,
/// `acquire_run_dir_flock_with_timeout` waits and eventually
/// fails with an actionable error message. Pins the
/// cross-process serialization contract.
///
/// In-process collision: two `try_flock` calls open distinct
/// OFDs against the same lockfile, and `flock(2)` serializes
/// them the same way it would two processes — so this test
/// exercises the production contention path without spawning
/// a child.
#[test]
fn acquire_run_dir_flock_times_out_when_peer_holds_lock() {
    let tmp = tempfile::TempDir::new().unwrap();
    let dir = tmp.path().join("contended-key");
    std::fs::create_dir_all(&dir).unwrap();

    // Peer: acquire the lock through the same machinery and
    // hold the fd alive for the duration of the test. Any
    // sibling acquire must time out behind this hold.
    let lock_path = super::run_dir_lock_path(&dir).unwrap();
    std::fs::create_dir_all(lock_path.parent().unwrap()).unwrap();
    let _peer_fd = crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
        .expect("peer flock attempt")
        .expect("peer must acquire on a fresh lockfile");

    let start = std::time::Instant::now();
    let err =
        super::acquire_run_dir_flock_with_timeout(&dir, std::time::Duration::from_millis(300))
            .expect_err("acquire must fail while peer holds LOCK_EX");
    let elapsed = start.elapsed();
    // Sanity: the helper waited at least roughly the requested
    // timeout before erroring — proves it polled rather than
    // returning EWOULDBLOCK on the first try.
    assert!(
        elapsed >= std::time::Duration::from_millis(250),
        "acquire must wait ~timeout before erroring; elapsed={elapsed:?}",
    );
    let msg = format!("{err:#}");
    assert!(
        msg.contains("timed out"),
        "error must surface the timeout cause; got: {msg}",
    );
    assert!(
        msg.contains("LOCK_EX"),
        "error must name the flock mode for operator triage; got: {msg}",
    );
}

// -- write_sidecar reuse-dir behavior --

/// Two `write_sidecar` invocations against the same effective
/// run directory (same `KTSTR_SIDECAR_DIR` here, simulating two
/// invocations from the same kernel + project commit) must
/// produce a directory containing only the second invocation's
/// sidecars — the first invocation's outputs are pre-cleared
/// before the second writes. Pins the last-writer-wins
/// semantics the documented `{kernel}-{project_commit}` keying implies.
///
/// CAVEAT: this test exercises the OVERRIDE path
/// (`KTSTR_SIDECAR_DIR` is set), where pre-clear is currently
/// SKIPPED per the override contract. To exercise pre-clear in
/// the env-overridden context, the test directly calls
/// `pre_clear_run_dir_once` BETWEEN the two writes — modeling
/// what `serialize_and_write_sidecar` does on the default path
/// (env unset). Both writes go through the override path so
/// the test does not depend on the OnceLock-cached cwd.
#[test]
fn write_sidecar_same_dir_is_last_writer_wins_after_pre_clear() {
    let _lock = lock_env();
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    // First invocation: write a sidecar for entry A.
    let entry_a = KtstrTestEntry {
        name: "__reuse_first_run__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let ok = AssertResult::pass();
    write_sidecar(&entry_a, &vm_result, &[], &ok, "SpinWait", &[]).unwrap();
    // Confirm the first invocation's sidecar is on disk.
    assert_eq!(
        find_sidecars_by_prefix(tmp, "__reuse_first_run__-").len(),
        1,
        "first invocation must write its sidecar",
    );

    // Simulate the second invocation: pre-clear the dir (which
    // is what `serialize_and_write_sidecar` does on the default
    // path), then write a sidecar for entry B.
    pre_clear_run_dir_once(tmp);
    // The first invocation's sidecar must be wiped by pre-clear.
    assert_eq!(
        find_sidecars_by_prefix(tmp, "__reuse_first_run__-").len(),
        0,
        "pre-clear must wipe the first invocation's sidecar before \
         the second invocation writes — this is the last-writer-wins \
         contract",
    );

    // Second invocation: distinct entry name to prove the
    // dir-state after pre-clear contains ONLY the second
    // invocation's sidecars.
    let entry_b = KtstrTestEntry {
        name: "__reuse_second_run__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    write_sidecar(&entry_b, &vm_result, &[], &ok, "SpinWait", &[]).unwrap();

    // Final state: only the second invocation's sidecar is
    // present. The first invocation is gone, the second is
    // intact.
    assert_eq!(
        find_sidecars_by_prefix(tmp, "__reuse_first_run__-").len(),
        0,
        "first invocation's sidecar must remain wiped after second invocation writes",
    );
    assert_eq!(
        find_sidecars_by_prefix(tmp, "__reuse_second_run__-").len(),
        1,
        "second invocation's sidecar must be the only sidecar in the dir",
    );
}

// -- KTSTR_SIDECAR_DIR override skips pre-clear --

/// When `KTSTR_SIDECAR_DIR` is set, `serialize_and_write_sidecar`
/// must NOT call `pre_clear_run_dir_once` against the override
/// dir. Pins the contract that operator-chosen directories are
/// preserved verbatim — silent data loss on an explicit env
/// override is unacceptable.
///
/// The test populates the override dir with a pre-existing
/// sidecar (from a hypothetical sibling run or a manual
/// fixture), runs `write_sidecar`, and verifies BOTH the
/// pre-existing sidecar AND the newly-written one are present.
/// A regression that pre-cleared on the override path would
/// leak this assertion (the pre-existing sidecar would be
/// wiped).
#[test]
fn write_sidecar_override_does_not_pre_clear() {
    let _lock = lock_env();
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

    // Pre-existing sidecar in the override dir — modeling a
    // run the operator wants to preserve.
    std::fs::write(tmp.join("__preserved__-0000.ktstr.json"), b"{}").unwrap();

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__override_skips_preclear__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let ok = AssertResult::pass();
    write_sidecar(&entry, &vm_result, &[], &ok, "SpinWait", &[]).unwrap();

    // The pre-existing sidecar must still be there. A regression
    // that fired pre_clear on the override path would have
    // wiped it.
    assert!(
        tmp.join("__preserved__-0000.ktstr.json").exists(),
        "pre-existing sidecar in override dir must NOT be pre-cleared — \
         operator-chosen directories are owned by the operator and \
         must not lose data on `write_sidecar`",
    );
    // Sanity: the new sidecar landed too.
    assert_eq!(
        find_sidecars_by_prefix(tmp, "__override_skips_preclear__-").len(),
        1,
        "new sidecar must be written alongside the preserved one",
    );
}

// -- relative-path canonicalize cache split regression --

/// Two sequential `write_sidecar` calls in the same process
/// against the DEFAULT path (no `KTSTR_SIDECAR_DIR` override)
/// must both survive: the second call must NOT wipe the first.
///
/// Pins the regression:
/// `serialize_and_write_sidecar` invoked `pre_clear_run_dir_once`
/// BEFORE `create_dir_all`. The first call resolved the
/// pre-clear cache key against the raw path because
/// `canonicalize` failed on a missing dir, then created the
/// dir via `create_dir_all` and wrote sidecar 1.
/// On the second call, `canonicalize` SUCCEEDED against the
/// now-existing dir, producing an absolute path that DIFFERED
/// from the cache key inserted by the first call — so the
/// second call missed the cache, fired pre-clear, and wiped
/// sidecar 1.
///
/// The fix moves `create_dir_all` before `pre_clear_run_dir_once`
/// so canonicalize sees the same on-disk dir on both calls and
/// produces the same canonicalized cache key. With the fix,
/// the second call hits the cache and pre-clear is a no-op,
/// so sidecar 1 survives.
///
/// ISOLATION: the test sets `CARGO_TARGET_DIR` to a unique
/// tempdir so the resolved sidecar dir is
/// `{tempdir}/ktstr/{kernel}-{project_commit}/` — uncrossable by
/// sibling test processes that share the workspace's
/// `target/ktstr/`. Without this isolation, a concurrent
/// nextest worker writing to the SAME shared default dir could
/// fire pre-clear for that dir, race with this test's writes,
/// and surface as a flaky `__b3_first__-` count = 0. The test
/// still exercises the REAL default-path flow (sidecar_dir
/// computes from runs_root + format_run_dirname,
/// serialize_and_write_sidecar runs create_dir_all then
/// pre_clear) — the only thing CARGO_TARGET_DIR redirects is
/// the runs-root parent.
///
/// `KTSTR_KERNEL` and `KTSTR_SIDECAR_DIR` are explicitly
/// removed: kernel resolves to `"unknown"` (deterministic),
/// override is unset (so the default-path branch runs).
/// Project commit comes from the test process's
/// OnceLock-cached cwd probe and is shared with every other
/// default-path test in the same process — irrelevant here
/// since the tempdir-scoped runs-root parent is unique to this
/// test, so no other test's pre-clear cache entry collides
/// with ours.
#[test]
fn write_sidecar_default_path_two_writes_both_survive() {
    let _lock = lock_env();
    let target_dir = tempfile::TempDir::new().unwrap();
    let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", target_dir.path());
    let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
    let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");

    // Resolve the default dir AFTER the env mutations so it
    // reflects the tempdir-scoped target. With KTSTR_KERNEL
    // unset and KTSTR_SIDECAR_DIR unset, this is
    // `{tempdir}/ktstr/unknown-{cached_project_commit}/`.
    let dir = sidecar_dir();

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry_first = KtstrTestEntry {
        name: "__b3_first__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let entry_second = KtstrTestEntry {
        name: "__b3_second__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let ok = AssertResult::pass();

    // First write: under the buggy ordering, this resolved
    // canonicalize-fails (dir missing) → cache key under raw
    // path → wipe was a no-op (dir didn't exist) → created
    // dir → wrote sidecar 1.
    write_sidecar(&entry_first, &vm_result, &[], &ok, "SpinWait", &[]).unwrap();
    // Confirm sidecar 1 lands.
    assert_eq!(
        find_sidecars_by_prefix(&dir, "__b3_first__-").len(),
        1,
        "first write must produce its sidecar",
    );

    // Second write: under the buggy ordering, this resolved
    // canonicalize-succeeds (dir now exists) → cache key under
    // absolute canonicalized path → DIFFERENT key than first
    // call → cache MISS → wipe ran → DELETED sidecar 1 → wrote
    // sidecar 2. Under the fix, create_dir_all runs first on
    // both calls, both canonicalize against an existing dir,
    // both produce the same canonicalized key, and the second
    // call hits the cache → no wipe → both survive.
    write_sidecar(&entry_second, &vm_result, &[], &ok, "SpinWait", &[]).unwrap();

    // Both sidecars must be present. A regression to the buggy
    // ordering would surface here as `__b3_first__-` count = 0.
    let first_count = find_sidecars_by_prefix(&dir, "__b3_first__-").len();
    let second_count = find_sidecars_by_prefix(&dir, "__b3_second__-").len();
    assert_eq!(
        first_count, 1,
        "first sidecar must survive the second write — a count of 0 \
         reveals the canonicalize-cache-split regression: pre-clear \
         ran a second time and wiped sidecar 1. Move `create_dir_all` \
         before `pre_clear_run_dir_once` so canonicalize sees the \
         same dir on both calls.",
    );
    assert_eq!(second_count, 1, "second sidecar must land normally",);

    // No explicit cleanup: the TempDir's Drop removes the
    // entire tempdir tree, including the sidecars and any
    // pre-clear residue under `{tempdir}/ktstr/`.
}

#[test]
fn write_sidecar_writes_file() {
    let _lock = lock_env();
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__sidecar_write_test__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let check_result = AssertResult::pass();
    write_sidecar(&entry, &vm_result, &[], &check_result, "SpinWait", &[]).unwrap();

    // Sidecar filename now includes a variant hash suffix so
    // gauntlet variants don't clobber each other. Use the
    // single-match helper, which also guards against stray
    // leftover files from prior runs or double-writer bugs.
    let path = find_single_sidecar_by_prefix(tmp, "__sidecar_write_test__-");
    let data = std::fs::read_to_string(&path).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
    assert_eq!(loaded.test_name, "__sidecar_write_test__");
    assert!(loaded.passed);
    assert!(!loaded.skipped, "pass result is not a skip");
    // write_sidecar must populate the host-context snapshot so
    // downstream `stats compare --runs a b` can diff hosts.
    // Without this assertion, a regression that dropped the
    // `host: Some(collect_host_context())` builder line would
    // land silently. `kernel_name` is always `Some("Linux")`
    // on a running Linux process (uname syscall, no filesystem
    // dependency), matching the baseline asserted by
    // `host_context::tests::collect_host_context_returns_populated_struct_on_linux`.
    let host = loaded
        .host
        .as_ref()
        .expect("write_sidecar must populate host field from collect_host_context");
    assert_eq!(host.kernel_name.as_deref(), Some("Linux"));
    // Pair the uname check with a field that `HostContext::default()`
    // leaves None. A regression that swapped the full
    // `collect_host_context()` call for `HostContext { kernel_name:
    // Some("Linux".into()), ..Default::default() }` would pass the
    // uname assertion but drop every other captured field —
    // `kernel_cmdline` is present on every live Linux process
    // (/proc/cmdline is always readable; see host_context::tests:
    // collect_host_context_captures_cmdline_on_linux) so
    // `kernel_cmdline.is_some()` catches the default-substitution
    // regression.
    assert!(
        host.kernel_cmdline.is_some(),
        "write_sidecar must capture full HostContext, not Default::default() — \
         /proc/cmdline is always readable on Linux (see host_context tests)",
    );
    // Second Default-distinguishing field: `kernel_release` is
    // populated by the uname() syscall on any live Linux host
    // (filesystem-independent — no /proc/sys dependency), so a
    // `None` here would indicate the default-substitution
    // regression reached the uname path. Pairing cmdline
    // (filesystem-sourced) with kernel_release (syscall-sourced)
    // gives two independent capture paths, so a regression that
    // broke only one collection site is still caught.
    assert!(
        host.kernel_release.is_some(),
        "write_sidecar must capture kernel_release — uname() is \
         filesystem-independent; a None here means the default \
         substitution bypassed the full collect_host_context()",
    );
}

/// `sidecar_variant_hash` is order-insensitive for `sysctls`
/// and `kargs` — canonicalized at hash time via a local sort
/// inside `sidecar_variant_hash`. Pinning the invariant
/// directly against the hash function catches a regression
/// that drops the sort block (reverts to iterating
/// `&sidecar.sysctls` / `&sidecar.kargs` in-order) even if all
/// existing stability pins continue to pass — those pins use
/// single-element collections where sorting is a no-op, so
/// they cannot detect this regression by themselves.
///
/// Calls the hash function directly rather than going through
/// `write_sidecar` because the sysctls/kargs come from
/// `entry.scheduler.sysctls()` / `kargs()` — static slices the
/// caller cannot reorder. The only path for a reordered input
/// is a direct `SidecarResult` construction with reordered
/// fields, which this test exercises.
#[test]
fn sidecar_variant_hash_is_order_invariant_for_sysctls_and_kargs() {
    let forward = SidecarResult {
        sysctls: vec![
            "sysctl.a=1".to_string(),
            "sysctl.b=2".to_string(),
            "sysctl.c=3".to_string(),
        ],
        kargs: vec![
            "karg_alpha".to_string(),
            "karg_beta".to_string(),
            "karg_gamma".to_string(),
        ],
        ..SidecarResult::test_fixture()
    };
    let reversed = SidecarResult {
        sysctls: vec![
            "sysctl.c=3".to_string(),
            "sysctl.b=2".to_string(),
            "sysctl.a=1".to_string(),
        ],
        kargs: vec![
            "karg_gamma".to_string(),
            "karg_beta".to_string(),
            "karg_alpha".to_string(),
        ],
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&forward),
        sidecar_variant_hash(&reversed),
        "reversed-order sysctls/kargs must hash identically — \
         the hash sorts both collections lexically before \
         folding bytes in, matching the set-determines-hash \
         contract documented on `sidecar_variant_hash`. A \
         regression that dropped the sort block would produce \
         distinct hashes and duplicate sidecar files for the \
         same semantic variant.",
    );

    // Permutation check: a partial reorder (sysctls same,
    // kargs reversed) must also collapse. Guards against a
    // partial revert that drops the sort in only one of the
    // two collections.
    let partial = SidecarResult {
        sysctls: forward.sysctls.clone(),
        kargs: reversed.kargs.clone(),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&forward),
        sidecar_variant_hash(&partial),
        "kargs-only reversal must still hash identically — \
         partial revert (one of the two sorts dropped) must \
         fail this assertion. Got distinct hashes for: \
         sysctls={:?}, kargs={:?} vs sysctls={:?}, kargs={:?}",
        forward.sysctls,
        forward.kargs,
        partial.sysctls,
        partial.kargs,
    );
}

#[test]
fn write_sidecar_variant_hash_distinguishes_work_types() {
    // Two gauntlet variants differing only in work_type must
    // produce distinct sidecar filenames so neither clobbers the
    // other.
    let _lock = lock_env();
    let tmp_dir = tempfile::TempDir::new().unwrap();
    let tmp = tmp_dir.path();
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__variant_test__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let ok = AssertResult::pass();
    write_sidecar(&entry, &vm_result, &[], &ok, "SpinWait", &[]).unwrap();
    write_sidecar(&entry, &vm_result, &[], &ok, "YieldHeavy", &[]).unwrap();

    let paths = find_sidecars_by_prefix(tmp, "__variant_test__-");
    assert_eq!(
        paths.len(),
        2,
        "two work_type variants must produce two distinct files, got {paths:?}"
    );
}

/// Freeze the `sidecar_variant_hash` wire format to the exact 64-bit
/// value produced for a representative populated SidecarResult.
///
/// Sidecar filenames embed this hash as a hex suffix; gauntlet
/// tooling groups variants by it. A silent change — e.g. bumping
/// `siphasher`, switching keys, or reordering fields fed into the
/// hasher — would let old-version tooling mis-group new-version
/// sidecars and vice versa. Pinning the output against a
/// pre-computed constant catches that drift before it ships.
///
/// Every currently hash-participating field (topology, scheduler,
/// payload, work_type, sysctls, kargs) is set
/// explicitly; non-participating fields come from
/// [`SidecarResult::test_fixture`] so unrelated schema growth does
/// not disturb the constant. If a future change adds, removes, or
/// renames a hash-participating field in [`sidecar_variant_hash`],
/// update the field set here too and recompute the expected
/// constant — otherwise this test silently degrades into a
/// same-defaults check.
#[test]
fn sidecar_variant_hash_stability_populated() {
    // Every currently hash-participating field is spelled out
    // explicitly so a change to `test_fixture` defaults cannot
    // silently shift the pinned constant. If you add, remove,
    // or rename a hash-participating field in
    // `sidecar_variant_hash`, update the field set here and
    // recompute the expected constant.
    let sc = SidecarResult {
        topology: "1n2l4c1t".to_string(),
        scheduler: "scx-ktstr".to_string(),
        payload: None,
        work_type: "SpinWait".to_string(),
        sysctls: vec!["sysctl.kernel.sched_cfs_bandwidth_slice_us=1000".to_string()],
        kargs: vec!["nosmt".to_string()],
        ..SidecarResult::test_fixture()
    };
    // If this assertion trips, the wire format changed. Bumping
    // the expected value is the wrong fix unless you also plan
    // for old sidecars to be regenerated — see the contract on
    // `sidecar_variant_hash`.
    assert_eq!(
        sidecar_variant_hash(&sc),
        0x5a5b5775a01a425f,
        "sidecar_variant_hash output drifted — regenerate expected only if \
         the wire format change is intentional and old sidecars are \
         disposable (which they are per ktstr's pre-1.0 stance)",
    );
}

/// Pair to [`sidecar_variant_hash_stability_populated`] covering
/// the empty-collections path. [`sidecar_variant_hash`] feeds a
/// canonical JSON object (via [`serde_json::to_vec`]) into
/// SipHasher13; the JSON serialization is what keeps an empty
/// sysctls vec distinct from an empty kargs vec — the object-key
/// names (`"sysctls"`, `"kargs"`) and the empty-array tokens
/// (`[]`) plus the `,` / `:` / `"` separators produced by
/// `serde_json` are the only bytes that distinguish e.g.
/// `{"sysctls":[],"kargs":[]}` from
/// `{"sysctls":[],"kargs":["foo"]}` here. Pinning the
/// empty-inputs hash catches regressions that would erase those
/// boundaries (e.g. switching to a key-less concatenation).
#[test]
fn sidecar_variant_hash_stability_empty_collections() {
    // Every currently hash-participating field is spelled out
    // explicitly so a change to `test_fixture` defaults cannot
    // silently shift the pinned constant. If you add, remove,
    // or rename a hash-participating field in
    // `sidecar_variant_hash`, update the field set here and
    // recompute the expected constant.
    let sc = SidecarResult {
        topology: "1n1l1c1t".to_string(),
        scheduler: "eevdf".to_string(),
        payload: None,
        work_type: String::new(),
        sysctls: Vec::new(),
        kargs: Vec::new(),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(sidecar_variant_hash(&sc), 0x36ceceea5c2beda7);
}

/// Two sidecars that differ only in `payload` must produce
/// distinct variant hashes so gauntlet runs composing the same
/// scheduler with different primary payloads (FIO vs STRESS_NG)
/// don't clobber each other's files.
#[test]
fn sidecar_variant_hash_distinguishes_payload() {
    // `none` relies on [`SidecarResult::test_fixture`] defaulting
    // `payload` to `None`. If that default changes, the absent-vs-
    // present comparison below collapses — the assertion below
    // and this comment are intentionally load-bearing.
    let base = SidecarResult::test_fixture;
    let none = base();
    assert!(
        none.payload.is_none(),
        "fixture default for payload must remain None"
    );
    let fio = SidecarResult {
        payload: Some("fio".to_string()),
        ..base()
    };
    let stress = SidecarResult {
        payload: Some("stress-ng".to_string()),
        ..base()
    };
    let h_none = sidecar_variant_hash(&none);
    let h_fio = sidecar_variant_hash(&fio);
    let h_stress = sidecar_variant_hash(&stress);
    assert_ne!(
        h_none, h_fio,
        "absent vs present payload must hash differently",
    );
    assert_ne!(
        h_fio, h_stress,
        "different payload names must hash differently",
    );
}

/// Two sidecars that differ only in `topology` (the rendered preset
/// label, e.g. `1n_1l_2c_1t` vs `1n_2l_4c_1t`) must produce distinct
/// variant hashes so per-preset sidecar files don't clobber each
/// other when a single test fans out across the gauntlet preset
/// list. The sibling
/// `sidecar_variant_hash_distinguishes_payload` /
/// `..._distinguishes_work_types` tests pin the payload and
/// work_type axes; this fills the preset / topology axis gap so a
/// regression that dropped `topology` from
/// [`sidecar_variant_hash`]'s canonical JSON object would surface
/// here as a hash collision rather than silently producing N
/// gauntlet variants under one filename.
#[test]
fn sidecar_variant_hash_distinguishes_presets() {
    let mut sc = SidecarResult::test_fixture();
    sc.topology = "1n_1l_2c_1t".to_string();
    let h1 = sidecar_variant_hash(&sc);
    sc.topology = "1n_2l_4c_1t".to_string();
    let h2 = sidecar_variant_hash(&sc);
    assert_ne!(
        h1, h2,
        "preset must influence sidecar variant hash so per-preset \
         sidecar files don't clobber each other",
    );
}

// -- format_verifier_stats tests --

#[test]
fn format_verifier_stats_empty() {
    assert!(format_verifier_stats(&[]).is_empty());
}

#[test]
fn format_verifier_stats_no_data() {
    let sc = SidecarResult::test_fixture();
    assert!(format_verifier_stats(&[sc]).is_empty());
}

#[test]
fn format_verifier_stats_table() {
    let sc = SidecarResult {
        verifier_stats: vec![
            crate::monitor::bpf_prog::ProgVerifierStats {
                name: "dispatch".to_string(),
                verified_insns: 50000,
            },
            crate::monitor::bpf_prog::ProgVerifierStats {
                name: "enqueue".to_string(),
                verified_insns: 30000,
            },
        ],
        ..SidecarResult::test_fixture()
    };
    let result = format_verifier_stats(&[sc]);
    assert!(result.contains("BPF VERIFIER STATS"));
    assert!(result.contains("dispatch"));
    assert!(result.contains("enqueue"));
    assert!(result.contains("50000"));
    assert!(result.contains("30000"));
    assert!(result.contains("total verified insns: 80000"));
    assert!(!result.contains("WARNING"));
}

#[test]
fn format_verifier_stats_warning() {
    let sc = SidecarResult {
        verifier_stats: vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "heavy".to_string(),
            verified_insns: 800000,
        }],
        ..SidecarResult::test_fixture()
    };
    let result = format_verifier_stats(&[sc]);
    assert!(result.contains("WARNING"));
    assert!(result.contains("heavy"));
    assert!(result.contains("80.0%"));
}

#[test]
fn sidecar_verifier_stats_serde_roundtrip() {
    let sc = SidecarResult {
        verifier_stats: vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "init".to_string(),
            verified_insns: 5000,
        }],
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    assert!(json.contains("verifier_stats"));
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.verifier_stats.len(), 1);
    assert_eq!(loaded.verifier_stats[0].name, "init");
    assert_eq!(loaded.verifier_stats[0].verified_insns, 5000);
}

/// Every `Vec` field emits as `"x":[]` when empty rather than
/// being omitted. Pin the always-emit contract so a regression
/// that re-adds `skip_serializing_if` on `verifier_stats` is
/// caught before it ships.
#[test]
fn sidecar_verifier_stats_empty_emits_as_empty_array() {
    let sc = SidecarResult::test_fixture();
    let json = serde_json::to_string(&sc).unwrap();
    assert!(
        json.contains("\"verifier_stats\":[]"),
        "empty verifier_stats must emit as `\"verifier_stats\":[]`: {json}",
    );
}

#[test]
fn format_verifier_stats_deduplicates() {
    let vstats = vec![crate::monitor::bpf_prog::ProgVerifierStats {
        name: "dispatch".to_string(),
        verified_insns: 50000,
    }];
    let sc1 = SidecarResult {
        verifier_stats: vstats.clone(),
        ..SidecarResult::test_fixture()
    };
    let sc2 = SidecarResult {
        verifier_stats: vstats,
        ..SidecarResult::test_fixture()
    };
    let result = format_verifier_stats(&[sc1, sc2]);
    // Deduplicated: total should be 50000, not 100000.
    assert!(result.contains("total verified insns: 50000"));
}

// -- scheduler_fingerprint --

#[test]
fn scheduler_fingerprint_eevdf_empty_extras() {
    // Default scheduler (EEVDF) has no sysctls/kargs; fingerprint
    // returns the display name and two empty vecs.
    let entry = KtstrTestEntry {
        name: "eevdf_test",
        ..KtstrTestEntry::DEFAULT
    };
    let SchedulerFingerprint {
        scheduler: name,
        scheduler_commit: commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(&entry);
    assert_eq!(name, "eevdf");
    assert!(
        commit.is_none(),
        "Eevdf variant has no userspace binary; \
         scheduler_commit must be None. Got: {commit:?}",
    );
    assert!(sysctls.is_empty());
    assert!(kargs.is_empty());
}

#[test]
fn scheduler_fingerprint_formats_sysctls_with_prefix() {
    use super::super::entry::Sysctl;
    static SYSCTLS: &[Sysctl] = &[
        Sysctl::new("kernel.foo", "1"),
        Sysctl::new("kernel.bar", "yes"),
    ];
    static SCHED: super::super::entry::Scheduler =
        super::super::entry::Scheduler::new("s").sysctls(SYSCTLS);
    let entry = KtstrTestEntry {
        name: "s_test",
        scheduler: &SCHED,
        ..KtstrTestEntry::DEFAULT
    };
    let SchedulerFingerprint {
        scheduler: name,
        scheduler_commit: _,
        sysctls,
        kargs,
    } = scheduler_fingerprint(&entry);
    assert_eq!(name, "s");
    assert_eq!(
        sysctls,
        vec![
            "sysctl.kernel.foo=1".to_string(),
            "sysctl.kernel.bar=yes".to_string(),
        ]
    );
    assert!(kargs.is_empty());
}

#[test]
fn scheduler_fingerprint_forwards_kargs_verbatim() {
    static SCHED: super::super::entry::Scheduler =
        super::super::entry::Scheduler::new("s").kargs(&["quiet", "splash"]);
    let entry = KtstrTestEntry {
        name: "s_test",
        scheduler: &SCHED,
        ..KtstrTestEntry::DEFAULT
    };
    let SchedulerFingerprint {
        scheduler: _,
        scheduler_commit: _,
        sysctls,
        kargs,
    } = scheduler_fingerprint(&entry);
    assert_eq!(kargs, vec!["quiet".to_string(), "splash".to_string()]);
    assert!(sysctls.is_empty());
}

#[test]
fn scheduler_fingerprint_uses_display_name_for_discover() {
    use super::super::entry::SchedulerSpec;
    static SCHED: super::super::entry::Scheduler =
        super::super::entry::Scheduler::new("s").binary(SchedulerSpec::Discover("scx_relaxed"));
    let entry = KtstrTestEntry {
        name: "rel_test",
        scheduler: &SCHED,
        ..KtstrTestEntry::DEFAULT
    };
    let SchedulerFingerprint {
        scheduler: name,
        scheduler_commit: commit,
        sysctls: _,
        kargs: _,
    } = scheduler_fingerprint(&entry);
    assert_eq!(name, "s");
    assert!(
        commit.is_none(),
        "Discover variant currently returns None via \
         `SchedulerSpec::scheduler_commit` — \
         `resolve_scheduler`'s cascade does not guarantee a \
         fresh build, so there is no authoritative source for \
         the scheduler binary's commit and `scheduler_commit` \
         reports None honestly. Got: {commit:?}",
    );
}

/// Pin that `scheduler_fingerprint` reports `scheduler_commit: None`
/// for the EEVDF baseline (the no-scx-scheduler placeholder).
///
/// `scheduler_fingerprint` reads `entry.scheduler.binary.scheduler_commit()`
/// directly. `SchedulerSpec::scheduler_commit` returns `None` for every
/// variant (Eevdf, Discover, Path, KernelBuiltin) — the commit string is
/// not carried in the static spec. A regression that returned `Some(...)`
/// from `scheduler_commit` for any variant would silently populate the
/// sidecar's `scheduler_commit` field with a value not tied to an actual
/// binary commit; this test pins the `None` contract for the EEVDF
/// baseline end-to-end and confirms the `Scheduler::EEVDF` const reports
/// its `name` field (`"eevdf"`) into the sidecar verbatim.
#[test]
fn scheduler_fingerprint_eevdf_has_no_commit() {
    let entry = KtstrTestEntry {
        name: "eevdf_test",
        scheduler: &super::super::entry::Scheduler::EEVDF,
        ..KtstrTestEntry::DEFAULT
    };
    let SchedulerFingerprint {
        scheduler: name,
        scheduler_commit: commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(&entry);
    assert_eq!(
        name, "eevdf",
        "Scheduler::EEVDF carries the compile-time-fixed name \
         \"eevdf\"; got: {name:?}",
    );
    assert!(
        commit.is_none(),
        "EEVDF has no binary at all — scheduler_commit must be \
         None via the SchedulerSpec::scheduler_commit short-circuit \
         (every variant returns None). Got: {commit:?}",
    );
    assert!(
        sysctls.is_empty(),
        "EEVDF declares no sysctls; got: {sysctls:?}",
    );
    assert!(kargs.is_empty(), "EEVDF declares no kargs; got: {kargs:?}",);
}

mod commits;
