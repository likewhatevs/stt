use super::super::super::test_helpers::{EnvVarGuard, lock_env};
use super::super::*;
use super::find_single_sidecar_by_prefix;
use crate::assert::AssertResult;
use crate::scenario::Ctx;
use anyhow::Result;

// -- write_skip_sidecar --

/// `write_skip_sidecar` is the path covered by the ResourceContention
/// skip branch and any early-exit that bails before `run_ktstr_test_inner`
/// reaches the VM-run call site. The sidecar must be flagged
/// `skipped: true, passed: true` so stats tooling that subtracts
/// skipped runs from pass counts sees a recorded skip instead of
/// a missing file. This regression guards that contract against a
/// future change that forgets the passed-true flag or drops skip
/// sidecars entirely for non-VM early exits.
#[test]
fn write_skip_sidecar_records_passed_true_skipped_true() {
    let _lock = lock_env();
    let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-writes-test");
    let _ = std::fs::remove_dir_all(&tmp);
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__skip_sidecar_test__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let active_flags: Vec<String> = vec!["llc".to_string()];
    write_skip_sidecar(&entry, &active_flags).expect("skip sidecar must write");

    let path = find_single_sidecar_by_prefix(&tmp, "__skip_sidecar_test__-");
    let data = std::fs::read_to_string(&path).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
    assert_eq!(loaded.test_name, "__skip_sidecar_test__");
    assert!(
        loaded.passed,
        "skip sidecar must set passed=true so the verdict gate does not flip fail",
    );
    assert!(
        loaded.skipped,
        "skip sidecar must set skipped=true so stats tooling excludes from pass count",
    );
    assert_eq!(
        loaded.work_type, "skipped",
        "skip path uses the 'skipped' work_type bucket so grouping keeps the skip distinguishable",
    );
    assert_eq!(loaded.active_flags, active_flags);
    // write_skip_sidecar shares the host-context capture with
    // write_sidecar (same `collect_host_context()` builder line)
    // so skip paths still give `stats compare --runs` a host
    // baseline. A regression that dropped the skip-path capture
    // would leave `host: None` in only the skip bucket, producing
    // silent per-run partial data.
    let host = loaded
        .host
        .as_ref()
        .expect("write_skip_sidecar must populate host field from collect_host_context");
    assert_eq!(host.kernel_name.as_deref(), Some("Linux"));
    // Pair the uname check with a Default-distinguishing field —
    // see `write_sidecar_writes_file` for the rationale. Keeps
    // both the happy-path writer and the skip-path writer guarded
    // against the same default-substitution regression.
    assert!(
        host.kernel_cmdline.is_some(),
        "write_skip_sidecar must capture full HostContext, not Default::default()",
    );
    // Syscall-sourced companion to the filesystem-sourced
    // `kernel_cmdline` check — see `write_sidecar_writes_file`
    // for the two-independent-paths rationale.
    assert!(
        host.kernel_release.is_some(),
        "write_skip_sidecar must capture kernel_release (syscall-sourced)",
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// When the sidecar directory cannot be created (path collision
/// with a regular file), `write_skip_sidecar` must return `Err`
/// rather than silently eating the failure. Stats tooling relies
/// on the error chain to diagnose missing sidecars; a swallowed
/// error would make skips invisible to post-run analysis.
#[test]
fn write_skip_sidecar_returns_err_when_dir_cannot_be_created() {
    let _lock = lock_env();

    // Create a regular file, then try to use it as the sidecar
    // directory. `create_dir_all` fails because the path exists
    // but is not a directory.
    let blocker = std::env::temp_dir().join("ktstr-sidecar-skip-blocker");
    let _ = std::fs::remove_file(&blocker);
    let _ = std::fs::remove_dir_all(&blocker);
    std::fs::write(&blocker, b"not a dir").unwrap();
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &blocker);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__skip_sidecar_err_test__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let result = write_skip_sidecar(&entry, &[]);
    assert!(
        result.is_err(),
        "skip sidecar write must return Err when the target is a regular file",
    );

    let _ = std::fs::remove_file(&blocker);
}

// -- sidecar payload + metrics fields --

/// Empty `payload` / `metrics` serialize as `"payload":null` /
/// `"metrics":[]` (always-emit symmetric with `host`) rather than
/// being omitted. Pin the wire shape so a regression that re-adds
/// `skip_serializing_if` on either field is caught before it
/// ships, and verify the None/empty round-trip remains correct
/// under the deserialize-requires contract.
#[test]
fn sidecar_payload_and_metrics_always_emit_when_empty() {
    let sc = SidecarResult::test_fixture();
    let json = serde_json::to_string(&sc).unwrap();
    assert!(
        json.contains("\"payload\":null"),
        "empty payload must emit as `\"payload\":null`: {json}",
    );
    assert!(
        json.contains("\"metrics\":[]"),
        "empty metrics must emit as `\"metrics\":[]`: {json}",
    );
    assert!(
        json.contains("\"project_commit\":null"),
        "absent project_commit must emit as `\"project_commit\":null`, \
         not be omitted via `skip_serializing_if`: {json}",
    );
    assert!(
        json.contains("\"kernel_commit\":null"),
        "absent kernel_commit must emit as `\"kernel_commit\":null`, \
         not be omitted via `skip_serializing_if`: {json}",
    );
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    // Exhaustive destructure so a new `Option<_>` / `Vec<_>`
    // field on `SidecarResult` that defaults to `None` / empty
    // forces this test to spell it out and make an
    // always-emit-vs-skip decision at the same time. See
    // [`sidecar_result_roundtrip`] for the same pattern on the
    // populated side — the two together pin the wire contract
    // at both extremes of the default distribution.
    let SidecarResult {
        test_name: _,
        topology: _,
        scheduler: _,
        scheduler_commit,
        project_commit,
        payload,
        metrics,
        passed: _,
        skipped: _,
        stats: _,
        monitor,
        stimulus_events,
        work_type: _,
        active_flags,
        verifier_stats,
        kvm_stats,
        sysctls,
        kargs,
        kernel_version,
        kernel_commit,
        timestamp: _,
        run_id: _,
        host,
        cleanup_duration_ms,
        run_source,
    } = loaded;
    assert!(payload.is_none());
    assert!(metrics.is_empty());
    // The sibling-field defaults on the empty fixture — every
    // nullable must be None and every Vec empty, matching the
    // always-emit invariants that the JSON shape above pins.
    assert!(scheduler_commit.is_none());
    assert!(project_commit.is_none());
    assert!(monitor.is_none());
    assert!(stimulus_events.is_empty());
    assert!(active_flags.is_empty());
    assert!(verifier_stats.is_empty());
    assert!(kvm_stats.is_none());
    assert!(sysctls.is_empty());
    assert!(kargs.is_empty());
    assert!(kernel_version.is_none());
    assert!(kernel_commit.is_none());
    assert!(host.is_none());
    assert!(cleanup_duration_ms.is_none());
    assert!(
        run_source.is_none(),
        "absent run_source must round-trip as None, \
         matching the symmetric serialize/deserialize \
         contract enforced for every other nullable field",
    );
}

/// Populated `payload` + `metrics` survive round-trip with the
/// exact shape stats tooling will consume — one entry per
/// `ctx.payload(X).run()` call, each carrying its exit code and
/// any extracted metrics. Regression guard against a future
/// schema shift that flattens metrics across payloads (which
/// would lose the per-payload provenance the design requires).
#[test]
fn sidecar_payload_and_metrics_roundtrip_populated() {
    use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};
    let pm = PayloadMetrics {
        payload_index: 0,
        metrics: vec![Metric {
            name: "iops".to_string(),
            value: 5000.0,
            polarity: Polarity::HigherBetter,
            unit: "iops".to_string(),
            source: MetricSource::Json,
            stream: MetricStream::Stdout,
        }],
        exit_code: 0,
    };
    let sc = SidecarResult {
        test_name: "fio_run".to_string(),
        topology: "1n1l2c1t".to_string(),
        payload: Some("fio".to_string()),
        metrics: vec![pm],
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    assert!(json.contains("\"payload\":\"fio\""));
    assert!(json.contains("\"metrics\""));
    assert!(json.contains("\"iops\""));
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.payload.as_deref(), Some("fio"));
    assert_eq!(loaded.metrics.len(), 1);
    assert_eq!(loaded.metrics[0].exit_code, 0);
    assert_eq!(loaded.metrics[0].metrics.len(), 1);
    assert_eq!(loaded.metrics[0].metrics[0].name, "iops");
    assert_eq!(loaded.metrics[0].metrics[0].value, 5000.0);
    assert_eq!(
        loaded.metrics[0].metrics[0].stream,
        MetricStream::Stdout,
        "metric stream tag must round-trip through sidecar \
         serde; a regression that lost `stream` serialization \
         or deserialized it to a different variant would break \
         review-tooling's stdout-vs-stderr attribution",
    );
}

/// `write_sidecar` must populate `payload` from `entry.payload`
/// so a test declaring a binary payload writes the payload name
/// into the sidecar even when no payload-metrics have been
/// threaded in yet. This pins the half-wired state the
/// follow-up WOs will extend: stats tooling that already groups
/// by payload name sees the grouping key on the sidecar
/// immediately.
#[test]
fn write_sidecar_records_entry_payload_name() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};

    let _lock = lock_env();
    let tmp = std::env::temp_dir().join("ktstr-sidecar-payload-name-test");
    let _ = std::fs::remove_dir_all(&tmp);
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

    static FIO: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__payload_name_test__",
        func: dummy,
        auto_repro: false,
        payload: Some(&FIO),
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let ok = AssertResult::pass();
    write_sidecar(&entry, &vm_result, &[], &ok, "SpinWait", &[], &[]).unwrap();

    let path = find_single_sidecar_by_prefix(&tmp, "__payload_name_test__-");
    let data = std::fs::read_to_string(&path).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
    assert_eq!(loaded.payload.as_deref(), Some("fio"));
    assert!(
        loaded.metrics.is_empty(),
        "metrics stay empty until a Ctx-level accumulator lands",
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// `write_sidecar` must forward the `payload_metrics` slice
/// into `SidecarResult.metrics` unmodified — once the
/// follow-up Ctx-accumulator WO lands, stats tooling will see
/// every `ctx.payload(X).run()` invocation's output in order.
#[test]
fn write_sidecar_forwards_payload_metrics_slice() {
    use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};

    let _lock = lock_env();
    let tmp = std::env::temp_dir().join("ktstr-sidecar-metrics-slice-test");
    let _ = std::fs::remove_dir_all(&tmp);
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__metrics_slice_test__",
        func: dummy,
        auto_repro: false,
        ..KtstrTestEntry::DEFAULT
    };
    let vm_result = crate::vmm::VmResult::test_fixture();
    let ok = AssertResult::pass();
    let metrics = vec![
        PayloadMetrics {
            payload_index: 0,
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 1200.0,
                polarity: Polarity::HigherBetter,
                unit: "iops".to_string(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            }],
            exit_code: 0,
        },
        PayloadMetrics {
            payload_index: 1,
            metrics: vec![],
            exit_code: 2,
        },
    ];
    write_sidecar(&entry, &vm_result, &[], &ok, "SpinWait", &[], &metrics).unwrap();

    let path = find_single_sidecar_by_prefix(&tmp, "__metrics_slice_test__-");
    let data = std::fs::read_to_string(&path).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
    assert_eq!(loaded.metrics.len(), 2);
    assert_eq!(loaded.metrics[0].exit_code, 0);
    assert_eq!(loaded.metrics[0].metrics.len(), 1);
    assert_eq!(loaded.metrics[0].metrics[0].name, "iops");
    assert_eq!(loaded.metrics[1].exit_code, 2);
    assert!(loaded.metrics[1].metrics.is_empty());

    let _ = std::fs::remove_dir_all(&tmp);
}

/// `write_skip_sidecar` must also carry `entry.payload` through
/// so a ResourceContention or early-skip on a payload-carrying
/// test still records the payload name. Missing this would
/// drop skipped runs out of payload-grouped stats.
#[test]
fn write_skip_sidecar_records_entry_payload_name() {
    use crate::test_support::{OutputFormat, Payload, PayloadKind};

    let _lock = lock_env();
    let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-payload-test");
    let _ = std::fs::remove_dir_all(&tmp);
    let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

    static STRESS: Payload = Payload {
        name: "stress-ng",
        kind: PayloadKind::Binary("stress-ng"),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
        include_files: &[],
        uses_parent_pgrp: false,
        known_flags: None,
        metric_bounds: None,
    };

    fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
        Ok(AssertResult::pass())
    }
    let entry = KtstrTestEntry {
        name: "__skip_payload_name_test__",
        func: dummy,
        auto_repro: false,
        payload: Some(&STRESS),
        ..KtstrTestEntry::DEFAULT
    };
    write_skip_sidecar(&entry, &[]).unwrap();

    let path = find_single_sidecar_by_prefix(&tmp, "__skip_payload_name_test__-");
    let data = std::fs::read_to_string(&path).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
    assert_eq!(loaded.payload.as_deref(), Some("stress-ng"));
    assert!(loaded.skipped);
    assert!(
        loaded.metrics.is_empty(),
        "skip path never accumulates metrics"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// `host` is deliberately excluded from `sidecar_variant_hash`:
/// two gauntlet variants run on different hosts must collapse
/// into the same hash bucket so downstream stats tooling groups
/// them together. If a future change accidentally folds
/// `HostContext` into the hash, this test catches it before
/// the run-key split reaches on-disk sidecars.
#[test]
fn sidecar_variant_hash_excludes_host_context() {
    use crate::host_context::HostContext;
    let populated = HostContext {
        cpu_model: Some("Example CPU".to_string()),
        cpu_vendor: Some("GenuineExample".to_string()),
        total_memory_kb: Some(16_384_000),
        hugepages_total: Some(0),
        hugepages_free: Some(0),
        hugepages_size_kb: Some(2048),
        thp_enabled: Some("always [madvise] never".to_string()),
        thp_defrag: Some("[always] defer madvise never".to_string()),
        sched_tunables: None,
        online_cpus: Some(8),
        numa_nodes: Some(2),
        cpufreq_governor: std::collections::BTreeMap::new(),
        kernel_name: Some("Linux".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        arch: Some("x86_64".to_string()),
        kernel_cmdline: Some("preempt=lazy".to_string()),
        heap_state: None,
    };
    let without_host = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        ..SidecarResult::test_fixture()
    };
    let with_host = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        host: Some(populated),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&without_host),
        sidecar_variant_hash(&with_host),
        "host context must not influence variant hash",
    );
}

/// `scheduler_commit` is metadata, not a variant discriminator:
/// two gauntlet runs differing only in the recorded scheduler
/// commit (e.g. same variant re-run after a scheduler rebuild)
/// must share one hash bucket so `stats compare` treats them as
/// the same semantic variant. If a future change folds
/// `scheduler_commit` into `sidecar_variant_hash`, this test
/// catches it before the run-key split reaches on-disk sidecars
/// and splits previously-comparable runs. Mirrors
/// `sidecar_variant_hash_excludes_host_context`.
#[test]
fn sidecar_variant_hash_excludes_scheduler_commit() {
    let without_commit = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        scheduler_commit: None,
        ..SidecarResult::test_fixture()
    };
    let with_commit = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        scheduler_commit: Some("0000000000000000000000000000000000000000".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&without_commit),
        sidecar_variant_hash(&with_commit),
        "scheduler_commit must not influence variant hash — \
         runs of the same semantic variant on different \
         scheduler-binary builds must remain comparable by \
         `stats compare`",
    );
}

/// `project_commit` is metadata, not a variant discriminator:
/// two gauntlet runs differing only in the recorded ktstr
/// project commit (e.g. same variant re-run after a `git pull`
/// of the harness, or run from two ktstr clones at different
/// HEADs) must share one hash bucket so `stats compare`
/// treats them as the same semantic variant. If a future
/// change folds `project_commit` into `sidecar_variant_hash`,
/// this test catches it before the run-key split reaches
/// on-disk sidecars and splits previously-comparable runs.
/// Mirrors `sidecar_variant_hash_excludes_scheduler_commit` —
/// the same exclusion rationale applies to both metadata
/// fields.
///
/// Three cases pinned: (1) None vs Some, (2) two distinct
/// populated values, (3) clean Some vs `-dirty` Some. Without
/// the populated×populated case, a regression that XOR'd
/// project_commit's bytes into the hash would still pass the
/// None vs Some case if the empty-input contribution happened
/// to be zero; the third case guards specifically against a
/// change that distinguished only the dirty bit.
#[test]
fn sidecar_variant_hash_excludes_project_commit() {
    let without_commit = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        project_commit: None,
        ..SidecarResult::test_fixture()
    };
    let with_commit = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abcdef1-dirty".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&without_commit),
        sidecar_variant_hash(&with_commit),
        "project_commit must not influence variant hash — \
         None vs Some(...) case",
    );

    let with_commit_a = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abc1234".to_string()),
        ..SidecarResult::test_fixture()
    };
    let with_commit_b = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("def5678".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&with_commit_a),
        sidecar_variant_hash(&with_commit_b),
        "project_commit must not influence variant hash — \
         two distinct populated commits case (catches XOR-style \
         regressions where None and one specific Some happen to \
         collide)",
    );

    let with_commit_clean = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abc1234".to_string()),
        ..SidecarResult::test_fixture()
    };
    let with_commit_dirty = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        project_commit: Some("abc1234-dirty".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&with_commit_clean),
        sidecar_variant_hash(&with_commit_dirty),
        "project_commit must not influence variant hash — \
         clean vs `-dirty` of the same hex case (catches a \
         regression that distinguished only the dirty bit)",
    );
}

/// `kernel_commit` is metadata, not a variant discriminator:
/// two gauntlet runs differing only in the recorded kernel
/// source-tree commit (e.g. same variant re-run after a
/// `git pull` of the kernel tree, or the same release rebuilt
/// on top of a WIP patch) must share one hash bucket so
/// `stats compare` treats them as the same semantic variant.
/// If a future change folds `kernel_commit` into
/// `sidecar_variant_hash`, this test catches it before the
/// run-key split reaches on-disk sidecars and splits
/// previously-comparable runs. Mirrors
/// `sidecar_variant_hash_excludes_project_commit` /
/// `sidecar_variant_hash_excludes_scheduler_commit` — the
/// same exclusion rationale applies to all three metadata
/// commit fields.
///
/// Three cases pinned: (1) None vs Some, (2) two distinct
/// populated values, (3) clean Some vs `-dirty` Some. Without
/// the populated×populated case, a regression that XOR'd
/// kernel_commit's bytes into the hash would still pass the
/// None vs Some case if the empty-input contribution happened
/// to be zero; the third case guards specifically against a
/// change that distinguished only the dirty bit.
#[test]
fn sidecar_variant_hash_excludes_kernel_commit() {
    let without_commit = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        kernel_commit: None,
        ..SidecarResult::test_fixture()
    };
    let with_commit = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("abcdef1-dirty".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&without_commit),
        sidecar_variant_hash(&with_commit),
        "kernel_commit must not influence variant hash — \
         None vs Some(...) case",
    );

    let with_commit_a = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("abc1234".to_string()),
        ..SidecarResult::test_fixture()
    };
    let with_commit_b = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("def5678".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&with_commit_a),
        sidecar_variant_hash(&with_commit_b),
        "kernel_commit must not influence variant hash — \
         two distinct populated commits case (catches XOR-style \
         regressions where None and one specific Some happen to \
         collide)",
    );

    let with_commit_clean = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("abc1234".to_string()),
        ..SidecarResult::test_fixture()
    };
    let with_commit_dirty = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        kernel_commit: Some("abc1234-dirty".to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&with_commit_clean),
        sidecar_variant_hash(&with_commit_dirty),
        "kernel_commit must not influence variant hash — \
         clean vs `-dirty` of the same hex case (catches a \
         regression that distinguished only the dirty bit)",
    );
}

/// `run_source` (the run-environment provenance tag) must not
/// influence the variant hash. Two runs of the same semantic
/// variant — one from a developer machine (`run_source: "local"`)
/// and one from a CI runner (`run_source: "ci"`) — must produce
/// the same sidecar filename so `compare_partitions` can diff them
/// across the CI/local boundary without the run-source tag
/// shattering them into per-environment buckets. Mirrors the
/// commit-exclusion tests: covers `None` vs `Some("local")`,
/// `Some("local")` vs `Some("ci")`, and `Some("ci")` vs
/// `Some("archive")` so a regression that distinguished only
/// one specific tag pair would still be caught.
#[test]
fn sidecar_variant_hash_excludes_run_source() {
    let none = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        run_source: None,
        ..SidecarResult::test_fixture()
    };
    let local = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        run_source: Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&none),
        sidecar_variant_hash(&local),
        "run_source must not influence variant hash — None vs \
         Some(\"local\") case",
    );

    let ci = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        run_source: Some(SIDECAR_RUN_SOURCE_CI.to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&local),
        sidecar_variant_hash(&ci),
        "run_source must not influence variant hash — \
         Some(\"local\") vs Some(\"ci\") case (catches XOR-style \
         regressions where two specific tags happen to collide)",
    );

    let archive = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        run_source: Some(SIDECAR_RUN_SOURCE_ARCHIVE.to_string()),
        ..SidecarResult::test_fixture()
    };
    assert_eq!(
        sidecar_variant_hash(&ci),
        sidecar_variant_hash(&archive),
        "run_source must not influence variant hash — \
         Some(\"ci\") vs Some(\"archive\") case",
    );
}

/// `detect_run_source` reads `KTSTR_CI` and returns `"ci"`
/// when set non-empty, `"local"` otherwise. Empty-string env
/// values count as unset so a defensively-cleared variable
/// does not accidentally classify a developer run as CI.
#[test]
fn detect_run_source_routes_on_ktstr_ci_env() {
    let _lock = lock_env();
    let _restore = EnvVarGuard::remove(KTSTR_CI_ENV);
    assert_eq!(
        detect_run_source(),
        Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
        "unset KTSTR_CI must classify as `local`",
    );
    let _set_empty = EnvVarGuard::set(KTSTR_CI_ENV, std::path::Path::new(""));
    assert_eq!(
        detect_run_source(),
        Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
        "empty-string KTSTR_CI must classify as `local` so a \
         defensively-cleared variable does not accidentally \
         flip the tag",
    );
    drop(_set_empty);
    let _set_one = EnvVarGuard::set(KTSTR_CI_ENV, std::path::Path::new("1"));
    assert_eq!(
        detect_run_source(),
        Some(SIDECAR_RUN_SOURCE_CI.to_string()),
        "non-empty KTSTR_CI must classify as `ci`",
    );
}

/// `apply_archive_source_override` rewrites every sidecar's
/// `run_source` to `"archive"` regardless of the prior value, so
/// that `--dir`-loaded pools surface uniformly under the
/// archive bucket. Pin both branches: a populated `run_source`
/// (`"local"` / `"ci"`) is overwritten, and `None` is
/// rewritten to `Some("archive")` rather than left as `None`.
#[test]
fn apply_archive_source_override_rewrites_every_entry() {
    let mut pool = vec![
        SidecarResult {
            run_source: Some(SIDECAR_RUN_SOURCE_LOCAL.to_string()),
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            run_source: Some(SIDECAR_RUN_SOURCE_CI.to_string()),
            ..SidecarResult::test_fixture()
        },
        SidecarResult {
            run_source: None,
            ..SidecarResult::test_fixture()
        },
    ];
    apply_archive_source_override(&mut pool);
    for sc in &pool {
        assert_eq!(
            sc.run_source.as_deref(),
            Some(SIDECAR_RUN_SOURCE_ARCHIVE),
            "every sidecar in a --dir pool must surface as \
             archive after override",
        );
    }
}

/// A `SidecarResult` carrying a fully populated `HostContext`
/// round-trips through serde_json without losing fields.
/// Struct-level `PartialEq` on `HostContext` makes one
/// `assert_eq!(host, ctx)` cover every field, so a future
/// change that breaks composition between the outer
/// `SidecarResult` and the embedded `HostContext` is caught at
/// the seam without needing a per-field assertion.
#[test]
fn sidecar_result_roundtrip_with_populated_host_context() {
    use crate::host_context::HostContext;
    let mut tunables = std::collections::BTreeMap::new();
    tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
    let ctx = HostContext {
        cpu_model: Some("Example CPU".to_string()),
        cpu_vendor: Some("GenuineExample".to_string()),
        total_memory_kb: Some(16_384_000),
        hugepages_total: Some(4),
        hugepages_free: Some(2),
        hugepages_size_kb: Some(2048),
        thp_enabled: Some("always [madvise] never".to_string()),
        thp_defrag: Some("[always] defer madvise never".to_string()),
        sched_tunables: Some(tunables),
        online_cpus: Some(8),
        numa_nodes: Some(2),
        cpufreq_governor: std::collections::BTreeMap::new(),
        kernel_name: Some("Linux".to_string()),
        kernel_release: Some("6.11.0".to_string()),
        arch: Some("x86_64".to_string()),
        kernel_cmdline: Some("preempt=lazy isolcpus=1-3".to_string()),
        heap_state: Some(crate::host_heap::HostHeapState::test_fixture()),
    };
    let sc = SidecarResult {
        topology: "1n1l2c1t".to_string(),
        host: Some(ctx.clone()),
        ..SidecarResult::test_fixture()
    };
    let json = serde_json::to_string(&sc).unwrap();
    let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
    let host = loaded.host.expect("host must round-trip");
    assert_eq!(host, ctx);
}

/// Every sidecar produced within a single ktstr run records the
/// SAME host context — all writers call
/// [`crate::host_context::collect_host_context`], which
/// memoises the static subset in a process-global `OnceLock`
/// (`STATIC_HOST_INFO`) and re-reads the dynamic subset from
/// the same `/proc` / `/sys` sources on every call. Runtime
/// drift in the captured struct across sidecars in one run
/// would mean one of two bad outcomes:
///   - a regression in the static memoisation (cache key / init
///     closure), producing per-call distinct values for fields
///     that cannot change across a process lifetime (uname,
///     CPU model, NUMA topology);
///   - a test concurrently mutating a dynamic field
///     (`thp_enabled`, `sched_tunables`, hugepage reservations)
///     while another test writes a sidecar, which would be a
///     test-isolation bug — every in-tree test treats host
///     tunables as read-only.
///
/// This test runs a deterministic N-iteration loop (NOT a
/// proptest-style property sampler — there is no input-space
/// shrinker and no random seed; the same N calls with the same
/// ordering produce the same comparisons every run) of
/// back-to-back `collect_host_context()` calls simulating the
/// per-test sidecar drumbeat of a gauntlet run. Every resulting
/// `host` field must compare equal across all N samples. The
/// sibling [`crate::host_context`] tests already pin
/// `collect_host_context` internal stability; this test pins
/// the SIDECAR surface so a regression that threaded a partial
/// context through `write_sidecar` / `write_skip_sidecar`
/// would fail here even if `collect_host_context` itself
/// stayed stable.
///
/// Bounded N=8: enough iterations to catch intermittent drift
/// without bloating the test runtime — `collect_host_context`
/// does ~20 sysfs/procfs reads per call, so the cost scales
/// linearly and must stay modest.
///
/// `#[cfg(target_os = "linux")]`: `collect_host_context` only
/// reads meaningful data on Linux — on other hosts every field
/// is `None` and the equality would trivially hold without
/// exercising the contract.
#[cfg(target_os = "linux")]
#[test]
fn sidecars_in_a_run_carry_identical_host_context() {
    const N: usize = 8;
    let samples: Vec<crate::host_context::HostContext> = (0..N)
        .map(|_| crate::host_context::collect_host_context())
        .collect();
    let first = samples
        .first()
        .expect("N > 0 samples must produce at least one host context");

    // Fields expected to stay STRICTLY equal — either memoised
    // in STATIC_HOST_INFO (uname, CPU, memory, topology) or
    // effectively reboot-static (kernel_cmdline). A regression
    // that broke the cache or mis-read /proc would diverge here.
    for (i, s) in samples.iter().enumerate() {
        assert_eq!(
            s.kernel_name, first.kernel_name,
            "sidecar {i}: kernel_name drifted from first sample",
        );
        assert_eq!(
            s.kernel_release, first.kernel_release,
            "sidecar {i}: kernel_release drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.arch, first.arch,
            "sidecar {i}: arch drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.cpu_model, first.cpu_model,
            "sidecar {i}: cpu_model drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.cpu_vendor, first.cpu_vendor,
            "sidecar {i}: cpu_vendor drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.total_memory_kb, first.total_memory_kb,
            "sidecar {i}: total_memory_kb drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.hugepages_size_kb, first.hugepages_size_kb,
            "sidecar {i}: hugepages_size_kb drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.online_cpus, first.online_cpus,
            "sidecar {i}: online_cpus drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.numa_nodes, first.numa_nodes,
            "sidecar {i}: numa_nodes drifted — STATIC_HOST_INFO cache broken?",
        );
        assert_eq!(
            s.kernel_cmdline, first.kernel_cmdline,
            "sidecar {i}: kernel_cmdline drifted — only a reboot can change it",
        );
    }

    // Dynamic fields are allowed to vary in value under
    // concurrent sysctl/THP/hugepage twiddles (see the sibling
    // `collect_host_context_dynamic_subset_is_stable_across_calls`
    // test for the rationale), but the PRESENCE of each field
    // must stay consistent — a sidecar that suddenly loses the
    // THP row means the collector silently degraded, which
    // stats tooling would read as "no THP data on that host"
    // rather than the truth ("collector broke").
    for (i, s) in samples.iter().enumerate() {
        assert_eq!(
            s.hugepages_total.is_some(),
            first.hugepages_total.is_some(),
            "sidecar {i}: hugepages_total presence flipped across sidecars",
        );
        assert_eq!(
            s.hugepages_free.is_some(),
            first.hugepages_free.is_some(),
            "sidecar {i}: hugepages_free presence flipped across sidecars",
        );
        assert_eq!(
            s.thp_enabled.is_some(),
            first.thp_enabled.is_some(),
            "sidecar {i}: thp_enabled presence flipped across sidecars",
        );
        assert_eq!(
            s.thp_defrag.is_some(),
            first.thp_defrag.is_some(),
            "sidecar {i}: thp_defrag presence flipped across sidecars",
        );
        assert_eq!(
            s.sched_tunables.is_some(),
            first.sched_tunables.is_some(),
            "sidecar {i}: sched_tunables presence flipped across sidecars",
        );
    }
}

// -- detect_project_commit branch coverage --
//
// The five branches probed below cover every shape `detect_commit_at`
// can produce: a clean repo (Some(hex)), a dirty tracked-file
// worktree (Some(hex-dirty)), a non-git directory (None), an unborn
// HEAD (None), and a submodule-entry tree with the submodule
// unchecked-out (Some(hex), no -dirty). Fixtures use gix directly
// — no `git` shell-out — so the tests reflect the same library the
// production probe uses.

/// Plant author NAME/EMAIL fallbacks on `repo`'s in-memory config
/// snapshot.
///
/// `gix::Repository::commit` requires both an author and a
/// committer signature. `committer_or_set_generic_fallback` only
/// writes the committer fallback (gix-0.81 has no equivalent
/// `author_or_set_generic_fallback`); the author cascade reads
/// `author -> user`, so without `user.name`/`user.email` in the
/// runner's git config the author is `None` and `commit` bails
/// with `AuthorMissing`. CI runners that do not pre-seed
/// `user.name`/`user.email` hit this. Plant
/// `gitoxide.author.nameFallback` / `emailFallback` directly so
/// the author cascade has a value regardless of ambient git
/// config — same shape gix uses for the committer fallback in
/// `committer_or_set_generic_fallback`.
///
/// Call this AFTER `committer_or_set_generic_fallback` so both
/// fallbacks land in the same config-snapshot append window;
/// either order works in principle, but pairing them next to
/// each other in callers keeps the "we set both" intent
/// readable.
fn set_test_author_fallback(repo: &mut gix::Repository) {
    use gix::config::tree::gitoxide;
    let mut cfg = gix::config::File::new(gix::config::file::Metadata::api());
    cfg.set_raw_value(&gitoxide::Author::NAME_FALLBACK, "ktstr-test")
        .expect("set author name fallback");
    cfg.set_raw_value(
        &gitoxide::Author::EMAIL_FALLBACK,
        "ktstr-test@example.invalid",
    )
    .expect("set author email fallback");
    let mut snap = repo.config_snapshot_mut();
    snap.append(cfg);
}

/// Construct a single-blob tree at `dir`, populate the index from it,
/// write the file content into the worktree, and return the new
/// HEAD commit's id. After this helper the repo is fully clean:
/// HEAD-tree == index == worktree.
///
/// `committer_or_set_generic_fallback` plus
/// [`set_test_author_fallback`] are both invoked because the test
/// process inherits no `user.name|email` git config and the
/// commit/ref-edit path requires a non-empty signature for both
/// committer and author. The committer fallback writes
/// "no name configured" / "noEmailAvailable@…" via gix's
/// built-in helper; the author fallback plants the matching
/// `gitoxide.author.nameFallback` / `emailFallback` keys so
/// `gix::Repository::commit` succeeds on CI runners with no
/// ambient git identity.
fn init_clean_repo_with_file(dir: &std::path::Path) -> gix::ObjectId {
    let mut repo = gix::init(dir).expect("gix::init");
    let _ = repo
        .committer_or_set_generic_fallback()
        .expect("committer fallback");
    set_test_author_fallback(&mut repo);
    let blob_id: gix::ObjectId = repo.write_blob(b"original\n").expect("write blob").detach();
    let tree = gix::objs::Tree {
        entries: vec![gix::objs::tree::Entry {
            mode: gix::objs::tree::EntryKind::Blob.into(),
            filename: "file.txt".into(),
            oid: blob_id,
        }],
    };
    let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
    let commit_id: gix::ObjectId = repo
        .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
        .expect("commit")
        .detach();
    // Populate the index from the tree and persist it so the
    // tree-vs-index check sees no staged drift, then write the
    // worktree file so the index-vs-worktree check sees no
    // unstaged drift.
    let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
    idx.write(gix::index::write::Options::default())
        .expect("write index");
    std::fs::write(dir.join("file.txt"), b"original\n").expect("write worktree file");
    commit_id
}

/// Clean repo: HEAD reachable, no staged or worktree diffs. The
/// short-hash matches `head.to_hex_with_len(7)`, exactly the same
/// shape `detect_commit_at` formats with — pinning the literal
/// also confirms the 7-char truncation is honored end-to-end (a
/// future refactor that swapped to `format!("{}").chars().take(8)`
/// would silently break the cross-run grouping that stats tooling
/// relies on).
#[test]
fn detect_project_commit_clean_repo_returns_short_hash() {
    let tmp = tempfile::TempDir::new().unwrap();
    let head = init_clean_repo_with_file(tmp.path());
    let result = super::super::detect_commit_at(tmp.path()).expect("clean repo must yield Some");
    assert_eq!(
        result.len(),
        7,
        "clean result must be a 7-char hex hash, got {result:?}"
    );
    assert!(
        !result.contains('-'),
        "clean result must not carry a -dirty suffix, got {result:?}"
    );
    assert!(
        result.chars().all(|c| c.is_ascii_hexdigit()),
        "clean result must be pure hex, got {result:?}"
    );
    assert_eq!(
        result,
        head.to_hex_with_len(7).to_string(),
        "clean result must match the HEAD short hash exactly"
    );
}

/// Dirty tracked-file worktree: HEAD reachable, index matches
/// HEAD, but worktree diverges from the index. The result must
/// carry the `-dirty` suffix per the `index_worktree` leg of the
/// dirt probe.
#[test]
fn detect_project_commit_dirty_repo_appends_dirty_suffix() {
    let tmp = tempfile::TempDir::new().unwrap();
    let head = init_clean_repo_with_file(tmp.path());
    // Mutate the tracked file so index-vs-worktree diverges.
    std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
    let result = super::super::detect_commit_at(tmp.path()).expect("dirty repo must yield Some");
    let expected_prefix = head.to_hex_with_len(7).to_string();
    assert_eq!(
        result,
        format!("{expected_prefix}-dirty"),
        "dirty result must be {expected_prefix:?} + -dirty suffix"
    );
}

/// `repo_is_dirty` returns `Some(false)` for a clean repo. Pins
/// the contract that the helper distinguishes "I checked, it's
/// clean" from "I couldn't check" (`None`), so future callers
/// that need that distinction get reliable signal. The
/// callthrough from `detect_commit_at` collapses both via
/// `unwrap_or(false)`, so this test covers a branch the
/// end-to-end `detect_project_commit_clean_repo_returns_short_hash`
/// test cannot pin.
#[test]
fn repo_is_dirty_clean_repo_returns_some_false() {
    let tmp = tempfile::TempDir::new().unwrap();
    init_clean_repo_with_file(tmp.path());
    let repo = gix::open(tmp.path()).expect("gix::open clean repo");
    assert_eq!(
        super::super::repo_is_dirty(&repo),
        Some(false),
        "clean repo must yield Some(false)"
    );
}

/// `repo_is_dirty` returns `Some(true)` when the worktree
/// diverges from the index. Pins the index-vs-worktree leg of
/// the cascade independently of the suffix-formatting logic in
/// `detect_commit_at`.
#[test]
fn repo_is_dirty_dirty_worktree_returns_some_true() {
    let tmp = tempfile::TempDir::new().unwrap();
    init_clean_repo_with_file(tmp.path());
    std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
    let repo = gix::open(tmp.path()).expect("gix::open dirty repo");
    assert_eq!(
        super::super::repo_is_dirty(&repo),
        Some(true),
        "dirty worktree must yield Some(true)"
    );
}

/// Non-git directory: `detect_commit_at` calls
/// `gix::discover` which walks up from the input path
/// looking for a `.git` boundary. When the host's `/tmp`
/// happens to live inside a git checkout, discover finds
/// that ancestor `.git` and returns success — the
/// no-repo-found branch this test pins cannot fire in that
/// environment. Skip when the tempdir resolves to an
/// ancestor repo. Same skip pattern as the sibling
/// `local_source_non_git_*` tests in fetch.rs.
#[test]
fn detect_project_commit_non_git_returns_none() {
    let tmp = tempfile::TempDir::new_in(std::env::temp_dir()).unwrap();
    if super::super::super::test_helpers::tempdir_resolves_to_ancestor_git(tmp.path()) {
        // `skip!` lives at crate root via `#[macro_use]` on
        // `mod test_macros` in lib.rs, so it is reachable
        // from this nested test module without explicit
        // import.
        skip!(
            "tempdir {} resolves to an ancestor git repo; cannot pin \
             non-git path semantics in this environment",
            tmp.path().display()
        );
    }
    let result = super::super::detect_commit_at(tmp.path());
    assert!(
        result.is_none(),
        "non-git directory must yield None, got {result:?}"
    );
}

/// Unborn HEAD: `gix::init` produces a repo whose HEAD points at a
/// branch that has not been written to yet. `head_id()` returns
/// Err on this state; `detect_commit_at` returns None.
#[test]
fn detect_project_commit_unborn_head_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let _repo = gix::init(tmp.path()).expect("gix::init");
    let result = super::super::detect_commit_at(tmp.path());
    assert!(
        result.is_none(),
        "unborn HEAD must yield None, got {result:?}"
    );
}

/// Concurrent invocation stability: the probe is read-only across
/// the gix layer, so N parallel calls against the same repo must
/// all return the same result. Failure here would indicate a
/// thread-safety regression in either gix or our usage of it.
#[test]
fn detect_project_commit_concurrent_calls_agree() {
    let tmp = tempfile::TempDir::new().unwrap();
    init_clean_repo_with_file(tmp.path());
    let path = tmp.path();
    let baseline =
        super::super::detect_commit_at(path).expect("baseline single-thread call must yield Some");

    const THREADS: usize = 8;
    let results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            handles.push(scope.spawn(|| super::super::detect_commit_at(path)));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("thread join"))
            .collect::<Vec<_>>()
    });
    for (i, r) in results.iter().enumerate() {
        assert_eq!(
            r.as_deref(),
            Some(baseline.as_str()),
            "thread {i} disagreed with baseline {baseline:?}: got {r:?}"
        );
    }
}

/// Submodule false-positive guard: an uninitialized submodule (a
/// gitlinks tree+index entry whose checked-out subdirectory has no
/// `.git` artifact yet) must NOT trip the dirty probe.
/// `detect_commit_at` configures `Submodule::Given { ignore: All,
/// .. }` precisely so a parent repo cloned without
/// `--recurse-submodules` does not get erroneously tagged `-dirty`
/// for every sidecar.
///
/// The fixture writes a tree containing a `.gitmodules` blob (the
/// submodule registration gix needs to recognise the gitlinks
/// entry as a submodule rather than a phantom directory) plus a
/// `Commit`-mode tree entry pointing at an arbitrary OID. The
/// worktree contains the `.gitmodules` file and an EMPTY `submod/`
/// directory — modelling a parent that was cloned without
/// `--recurse-submodules`. The probe must still report clean.
#[test]
fn detect_project_commit_submodule_uninit_is_clean() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut repo = gix::init(tmp.path()).expect("gix::init");
    let _ = repo
        .committer_or_set_generic_fallback()
        .expect("committer fallback");
    set_test_author_fallback(&mut repo);

    // A submodule reference needs both a `.gitmodules` registration
    // (so gix recognises the gitlinks entry as a submodule, not a
    // phantom file) and the gitlinks tree entry itself. The
    // submodule directory is INTENTIONALLY left absent from the
    // worktree, which is the "uninitialized" state the production
    // probe must tolerate.
    let gitmodules_content = b"\
[submodule \"submod\"]\n\
\tpath = submod\n\
\turl = https://example.invalid/submod.git\n";
    let gitmodules_blob: gix::ObjectId = repo
        .write_blob(gitmodules_content)
        .expect("write .gitmodules blob")
        .detach();
    // Any 20-byte OID is a syntactically valid commit reference
    // from the tree's perspective. The null id keeps the fixture
    // self-contained — no dependency on an actual submodule commit
    // having been written.
    let null_commit_id = gix::ObjectId::null(gix::hash::Kind::Sha1);
    let tree = gix::objs::Tree {
        entries: vec![
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: ".gitmodules".into(),
                oid: gitmodules_blob,
            },
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Commit.into(),
                filename: "submod".into(),
                oid: null_commit_id,
            },
        ],
    };
    let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
    let head: gix::ObjectId = repo
        .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
        .expect("commit")
        .detach();
    let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
    idx.write(gix::index::write::Options::default())
        .expect("write index");
    // Materialize the .gitmodules blob in the worktree so the
    // index-vs-worktree leg sees no diff for that file. Create an
    // empty `submod/` directory to model a parent that was cloned
    // with `--no-recurse-submodules`: the gitlinks entry is in the
    // tree and index, the directory exists in the worktree, but no
    // `.git` artifact lives inside it (the submodule is
    // unintialized).
    std::fs::write(tmp.path().join(".gitmodules"), gitmodules_content)
        .expect("write .gitmodules worktree");
    std::fs::create_dir(tmp.path().join("submod")).expect("create submod dir");

    let result =
        super::super::detect_commit_at(tmp.path()).expect("submodule repo must still yield Some");
    assert_eq!(
        result,
        head.to_hex_with_len(7).to_string(),
        "uninitialized submodule must not trigger -dirty suffix"
    );
}

// -- detect_kernel_commit branch coverage --
//
// Mirror the `detect_project_commit` branch matrix for the
// kernel-tree probe. The implementations are nearly identical
// except for `gix::open` (NOT `gix::discover`) so the parent
// walk does not surface; the tests pin the open-vs-discover
// shape explicitly.

/// Clean kernel repo: HEAD reachable, no staged or worktree
/// diffs. `detect_kernel_commit` returns the 7-char short
/// hash.
#[test]
fn detect_kernel_commit_clean_repo_returns_short_hash() {
    let tmp = tempfile::TempDir::new().unwrap();
    let head = init_clean_repo_with_file(tmp.path());
    let result =
        super::super::detect_kernel_commit(tmp.path()).expect("clean repo must yield Some");
    assert_eq!(
        result.len(),
        7,
        "clean result must be a 7-char hex hash, got {result:?}"
    );
    assert!(
        !result.contains('-'),
        "clean result must not carry a -dirty suffix, got {result:?}"
    );
    assert!(
        result.chars().all(|c| c.is_ascii_hexdigit()),
        "clean result must be pure hex, got {result:?}"
    );
    assert_eq!(
        result,
        head.to_hex_with_len(7).to_string(),
        "clean result must match the HEAD short hash exactly"
    );
}

/// Dirty tracked-file worktree: HEAD reachable, index matches
/// HEAD, but worktree diverges. The result must carry the
/// `-dirty` suffix.
#[test]
fn detect_kernel_commit_dirty_repo_appends_dirty_suffix() {
    let tmp = tempfile::TempDir::new().unwrap();
    let head = init_clean_repo_with_file(tmp.path());
    std::fs::write(tmp.path().join("file.txt"), b"modified\n").unwrap();
    let result =
        super::super::detect_kernel_commit(tmp.path()).expect("dirty repo must yield Some");
    let expected_prefix = head.to_hex_with_len(7).to_string();
    assert_eq!(
        result,
        format!("{expected_prefix}-dirty"),
        "dirty result must be {expected_prefix:?} + -dirty suffix"
    );
}

/// Non-git directory: `detect_kernel_commit` uses `gix::open`,
/// NOT `gix::discover`. Open requires `kernel_dir` to BE the
/// repo root, so a non-git directory yields None even when an
/// ancestor IS a git repo. This is the critical behavioural
/// difference from `detect_project_commit`: the kernel
/// directory is explicit, not walked-up.
///
/// Reproduces the failure mode `gix::discover` would trip
/// (parent walk resolves to ktstr's repo when the user passes
/// a non-git subdir as KTSTR_KERNEL): a literal subdirectory
/// of a real git tempdir, NOT initialized as its own repo,
/// must still yield None for the kernel probe.
#[test]
fn detect_kernel_commit_non_git_directory_returns_none() {
    let parent = tempfile::TempDir::new().unwrap();
    // Parent IS a git repo — discover() would walk up and find
    // it from any subdir.
    init_clean_repo_with_file(parent.path());
    let nested = parent.path().join("not_a_repo");
    std::fs::create_dir(&nested).expect("create nested non-git subdir");
    // Pin the precondition: discover() WOULD succeed from this
    // path because the parent is a git repo. If `detect_kernel_commit`
    // accidentally used discover instead of open, it would
    // surface the parent's HEAD here — which is exactly the
    // wrong-kernel-commit-recorded bug we want to prevent.
    assert!(
        gix::discover(&nested).is_ok(),
        "gix::discover must succeed from the nested path (parent IS a repo) — \
         this precondition validates that detect_kernel_commit's open-vs-discover \
         choice is the correct one for the test scenario",
    );
    let result = super::super::detect_kernel_commit(&nested);
    assert!(
        result.is_none(),
        "non-git directory must yield None — `detect_kernel_commit` uses \
         `gix::open` (NOT `gix::discover`), so the parent's HEAD must \
         NOT leak through. Got {result:?}",
    );
}

/// Unborn HEAD: `gix::init` produces a repo whose HEAD points at a
/// branch that has not been written to yet. `head_id()` returns
/// Err on this state; `detect_kernel_commit` returns None.
#[test]
fn detect_kernel_commit_unborn_head_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let _repo = gix::init(tmp.path()).expect("gix::init");
    let result = super::super::detect_kernel_commit(tmp.path());
    assert!(
        result.is_none(),
        "unborn HEAD must yield None, got {result:?}"
    );
}

/// Submodule false-positive guard, mirroring
/// `detect_project_commit_submodule_uninit_is_clean` — an
/// uninitialized submodule must NOT trip the dirty probe in
/// the kernel-tree shape either. Kernel trees commonly carry
/// submodules (e.g. `.git` worktrees pointing to lib stubs)
/// without those subdirectories being checked out, and a
/// false-positive `-dirty` would shatter every sidecar's
/// kernel_commit into a unique bucket.
#[test]
fn detect_kernel_commit_submodule_uninit_is_clean() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mut repo = gix::init(tmp.path()).expect("gix::init");
    let _ = repo
        .committer_or_set_generic_fallback()
        .expect("committer fallback");
    set_test_author_fallback(&mut repo);

    let gitmodules_content = b"\
[submodule \"submod\"]\n\
\tpath = submod\n\
\turl = https://example.invalid/submod.git\n";
    let gitmodules_blob: gix::ObjectId = repo
        .write_blob(gitmodules_content)
        .expect("write .gitmodules blob")
        .detach();
    let null_commit_id = gix::ObjectId::null(gix::hash::Kind::Sha1);
    let tree = gix::objs::Tree {
        entries: vec![
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Blob.into(),
                filename: ".gitmodules".into(),
                oid: gitmodules_blob,
            },
            gix::objs::tree::Entry {
                mode: gix::objs::tree::EntryKind::Commit.into(),
                filename: "submod".into(),
                oid: null_commit_id,
            },
        ],
    };
    let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
    let head: gix::ObjectId = repo
        .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
        .expect("commit")
        .detach();
    let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
    idx.write(gix::index::write::Options::default())
        .expect("write index");
    std::fs::write(tmp.path().join(".gitmodules"), gitmodules_content)
        .expect("write .gitmodules worktree");
    std::fs::create_dir(tmp.path().join("submod")).expect("create submod dir");

    let result = super::super::detect_kernel_commit(tmp.path())
        .expect("submodule repo must still yield Some");
    assert_eq!(
        result,
        head.to_hex_with_len(7).to_string(),
        "uninitialized submodule must not trigger -dirty suffix"
    );
}

/// `detect_project_commit` memoizes its probe behind a
/// process-wide [`std::sync::OnceLock`] (declared on the
/// function body). Two consecutive calls in the same process
/// must therefore return identical [`Option<String>`] results
/// — the first successful call seeds the cache with a probe
/// of cwd; subsequent calls collapse to a `Clone` of the
/// cached short-hash.
///
/// The OnceLock is process-global and writes during the FIRST
/// SUCCESSFUL call observed by the test process — that may be
/// this test or any sibling that ran earlier, since the cache
/// survives across test functions in a single binary. Either
/// way, the public-API contract this test pins is
/// "consecutive calls agree", which holds whether the cache
/// is hot from a previous test or warmed by the first call
/// here.
///
/// `None` is NOT cached — a probe failure (cwd outside any
/// git repo, transient I/O hiccup) leaves the OnceLock
/// unset, and the next call retries from scratch. Two
/// consecutive calls still agree because (a) each call
/// reads the same cwd, and (b) the same cwd produces the
/// same probe outcome (modulo the rare transient-failure
/// window the failure-retry behavior is designed to recover
/// from). The test does not constrain whether the result is
/// Some or None because the cwd at test-runner launch is
/// environmental; equality across the two calls is the
/// testable contract.
#[test]
fn detect_project_commit_memoizes_across_consecutive_calls() {
    let first = super::super::detect_project_commit();
    let second = super::super::detect_project_commit();
    assert_eq!(
        first, second,
        "consecutive detect_project_commit calls must return \
         identical Option<String> via the OnceLock cache; \
         got first={first:?}, second={second:?}",
    );
    // Also pin against a third call to catch a regression that
    // re-probes on every non-first call (e.g. one that read
    // the OnceLock but bypassed it on the return path).
    let third = super::super::detect_project_commit();
    assert_eq!(
        first, third,
        "third detect_project_commit call must still match the \
         first; got first={first:?}, third={third:?}",
    );
}

/// `detect_kernel_commit` memoizes its successful probes
/// behind a process-wide
/// [`std::sync::Mutex<HashMap<PathBuf, String>>`] keyed on
/// the canonicalized input path. Two consecutive calls with
/// the SAME path must return identical results — the first
/// successful call seeds the cache with the resolved short
/// hash; the second returns a clone of the cached entry
/// without re-probing.
///
/// `None` outcomes are NOT cached; a re-probe fires on every
/// call until the path resolves successfully.
///
/// Uses a fresh tempdir so the cache key is unique to this
/// test (no collision with other test functions in the same
/// binary). The hashmap key is the canonicalized path, so a
/// stable path argument across calls produces a cache hit on
/// the second invocation.
#[test]
fn detect_kernel_commit_memoizes_across_consecutive_calls_same_path() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let head = init_clean_repo_with_file(tmp.path());
    let expected = head.to_hex_with_len(7).to_string();

    let first = super::super::detect_kernel_commit(tmp.path());
    let second = super::super::detect_kernel_commit(tmp.path());
    let third = super::super::detect_kernel_commit(tmp.path());

    assert_eq!(
        first.as_deref(),
        Some(expected.as_str()),
        "first call must return the clean short hash {expected:?}; \
         got {first:?}",
    );
    assert_eq!(
        first, second,
        "consecutive detect_kernel_commit calls with the same \
         path must agree via the Mutex<HashMap> cache; got \
         first={first:?}, second={second:?}",
    );
    assert_eq!(
        first, third,
        "third detect_kernel_commit call with the same path must \
         still match; got first={first:?}, third={third:?}",
    );
}

/// `detect_kernel_commit`'s path-keyed cache must not
/// cross-contaminate between distinct kernel directories. Two
/// fresh tempdirs with different HEADs (different blob
/// content → different tree → different commit OID) must each
/// return their OWN HEAD short hash. A regression that, e.g.,
/// keyed the cache on a prefix or a hash-collision-prone
/// derivation would surface here as one of the two paths
/// returning the OTHER path's HEAD.
///
/// Mixed call interleaving (`a`, `b`, `a`, `b`) catches a
/// regression that overwrites the entry on every call rather
/// than inserting per-key.
#[test]
fn detect_kernel_commit_distinct_paths_do_not_cross_contaminate() {
    let tmp_a = tempfile::TempDir::new().expect("tempdir A");
    let tmp_b = tempfile::TempDir::new().expect("tempdir B");

    // Distinct HEADs: write a different blob in each repo so
    // the resulting commit OIDs differ. The blob bytes alone
    // determine the tree OID via gix; identical blobs would
    // yield identical commit OIDs and defeat the test.
    let head_a = init_clean_repo_with_file(tmp_a.path());
    // Overwrite the helper's "original\n" content with a
    // distinct payload, then re-commit so HEAD diverges from
    // tmp_b's. We can't reuse `init_clean_repo_with_file`
    // verbatim because that would commit the same content.
    let mut repo_b = gix::init(tmp_b.path()).expect("gix::init B");
    let _ = repo_b
        .committer_or_set_generic_fallback()
        .expect("committer fallback B");
    set_test_author_fallback(&mut repo_b);
    let blob_id_b: gix::ObjectId = repo_b
        .write_blob(b"different\n")
        .expect("write blob B")
        .detach();
    let tree_b = gix::objs::Tree {
        entries: vec![gix::objs::tree::Entry {
            mode: gix::objs::tree::EntryKind::Blob.into(),
            filename: "file.txt".into(),
            oid: blob_id_b,
        }],
    };
    let tree_id_b: gix::ObjectId = repo_b.write_object(&tree_b).expect("write tree B").detach();
    let head_b: gix::ObjectId = repo_b
        .commit(
            "HEAD",
            "init B",
            tree_id_b,
            std::iter::empty::<gix::ObjectId>(),
        )
        .expect("commit B")
        .detach();
    let mut idx_b = repo_b
        .index_from_tree(&tree_id_b)
        .expect("index_from_tree B");
    idx_b
        .write(gix::index::write::Options::default())
        .expect("write index B");
    std::fs::write(tmp_b.path().join("file.txt"), b"different\n").expect("write worktree file B");

    let expected_a = head_a.to_hex_with_len(7).to_string();
    let expected_b = head_b.to_hex_with_len(7).to_string();
    assert_ne!(
        expected_a, expected_b,
        "fixture precondition: the two repos must have distinct \
         HEADs for this test to mean anything; got a={expected_a} \
         b={expected_b}",
    );

    // Interleave the calls: a, b, a, b. A regression that
    // overwrote the cache on each insert (instead of inserting
    // per-key) would surface here as the second `a` call
    // returning B's hash, or the second `b` returning A's.
    let a1 = super::super::detect_kernel_commit(tmp_a.path());
    let b1 = super::super::detect_kernel_commit(tmp_b.path());
    let a2 = super::super::detect_kernel_commit(tmp_a.path());
    let b2 = super::super::detect_kernel_commit(tmp_b.path());

    assert_eq!(
        a1.as_deref(),
        Some(expected_a.as_str()),
        "first call against path A must return A's HEAD short \
         hash {expected_a:?}; got {a1:?}",
    );
    assert_eq!(
        b1.as_deref(),
        Some(expected_b.as_str()),
        "first call against path B must return B's HEAD short \
         hash {expected_b:?}; got {b1:?}",
    );
    assert_eq!(
        a1, a2,
        "second call against path A must match the first \
         (cache hit on the A entry); got a1={a1:?}, a2={a2:?}",
    );
    assert_eq!(
        b1, b2,
        "second call against path B must match the first \
         (cache hit on the B entry, NOT contaminated by A); \
         got b1={b1:?}, b2={b2:?}",
    );
    assert_ne!(
        a2, b2,
        "after interleaved calls, A and B must STILL hold \
         distinct values — a regression that lost per-key \
         distinction would equate them; got a2={a2:?}, b2={b2:?}",
    );
}

/// `detect_kernel_commit` does NOT cache `None` outcomes. The
/// first call against a path that fails (here: an unborn HEAD
/// from a fresh `gix::init`) returns `None`, but a SECOND call
/// after the same path becomes a valid checkout (a commit was
/// made in between) must return the resolved short hash —
/// proof that the failure did NOT poison the per-path cache.
///
/// Pins the failure-retry contract: a transient probe failure
/// (FS hiccup, race against an in-flight checkout, unborn HEAD
/// resolved later in the same process) must not lock in
/// `unknown` for that path's entire process lifetime. The
/// observable invariant is "the same path's outcome can change
/// from None to Some without process restart"; the regression
/// guarded against is "first None gets cached and the path is
/// stuck at None until the process exits."
///
/// SCOPE: this test guards the FAILURE-RETRY half of the cache
/// contract. The success-side memoization is pinned by
/// `detect_kernel_commit_memoizes_across_consecutive_calls_same_path`
/// and `detect_kernel_commit_canonicalizes_symlink_aliases` —
/// once the cache holds Some, mutations to the underlying repo
/// do NOT invalidate the entry.
#[test]
fn detect_kernel_commit_failure_does_not_poison_cache() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    // First call: unborn HEAD — `gix::open` succeeds but
    // `head_id()` fails, so `commit_with_dirty_suffix` returns
    // None. Pre-fix this would seed the cache with None.
    let mut repo = gix::init(tmp.path()).expect("gix::init");
    let _ = repo
        .committer_or_set_generic_fallback()
        .expect("committer fallback");
    set_test_author_fallback(&mut repo);
    let first = super::super::detect_kernel_commit(tmp.path());
    assert!(
        first.is_none(),
        "fixture precondition: unborn HEAD probe must return \
         None; got {first:?}",
    );

    // Materialize a commit so the path now resolves to a real
    // hash. A regression that cached the first None would
    // surface here as `second.is_none()` — the cache would
    // short-circuit the re-probe and the new commit would be
    // invisible.
    let blob_id: gix::ObjectId = repo.write_blob(b"original\n").expect("write blob").detach();
    let tree = gix::objs::Tree {
        entries: vec![gix::objs::tree::Entry {
            mode: gix::objs::tree::EntryKind::Blob.into(),
            filename: "file.txt".into(),
            oid: blob_id,
        }],
    };
    let tree_id: gix::ObjectId = repo.write_object(&tree).expect("write tree").detach();
    let head: gix::ObjectId = repo
        .commit("HEAD", "init", tree_id, std::iter::empty::<gix::ObjectId>())
        .expect("commit")
        .detach();
    let mut idx = repo.index_from_tree(&tree_id).expect("index_from_tree");
    idx.write(gix::index::write::Options::default())
        .expect("write index");
    std::fs::write(tmp.path().join("file.txt"), b"original\n").expect("write worktree file");

    let second = super::super::detect_kernel_commit(tmp.path());
    let expected = head.to_hex_with_len(7).to_string();
    assert_eq!(
        second.as_deref(),
        Some(expected.as_str()),
        "second call after the path becomes a valid checkout \
         must return the resolved short hash {expected:?} — a \
         regression that cached the first None would surface as \
         None here, locking the path at `unknown` for the rest \
         of the process. got {second:?}",
    );
}

/// `detect_kernel_commit` canonicalizes its cache key so two
/// path spellings that resolve to the same on-disk repo share
/// one cache entry. Without canonicalization a symlink alias
/// would re-run the gix-open + dirt-walk on every call,
/// defeating the memoization the cache exists to provide.
///
/// Behavioral proof: prime the cache against the canonical
/// (real) path of a CLEAN repo, then mutate the worktree so a
/// re-probe would surface `-dirty`, then call via a symlink
/// alias. With canonicalization the alias canonicalizes to
/// the real path, hits the cached CLEAN entry, and returns
/// the no-`-dirty` value. Without canonicalization the alias
/// keys the cache under its literal path, misses, re-probes,
/// and surfaces the new dirt as `*-dirty`.
///
/// The cache deliberately does NOT invalidate mid-process
/// (per the `KERNEL_COMMIT_CACHE` doc-comment); the
/// stale-on-purpose cached return is the load-bearing signal
/// that proves the symlink hit the canonicalized entry.
///
/// Unix-only — `std::os::unix::fs::symlink` is gated.
#[cfg(unix)]
#[test]
fn detect_kernel_commit_canonicalizes_symlink_aliases() {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let real = tmp.path().join("real");
    std::fs::create_dir(&real).expect("mkdir real");
    let head = init_clean_repo_with_file(&real);

    // Sibling symlink pointing at `real`. Both paths live
    // inside `tmp` so TempDir's drop cleans up everything.
    let alias = tmp.path().join("alias");
    std::os::unix::fs::symlink(&real, &alias).expect("symlink alias -> real");

    // Prime the cache via the canonical path. The entry is
    // now memoized under `real.canonicalize()` with the clean
    // short hash.
    let real_clean = super::super::detect_kernel_commit(&real)
        .expect("clean canonical-path probe must yield Some");
    assert_eq!(
        real_clean,
        head.to_hex_with_len(7).to_string(),
        "fixture precondition: canonical-path probe must return \
         the clean 7-char head hash; got {real_clean:?}",
    );

    // Introduce dirt — any cache-bypass re-probe would now
    // observe `-dirty`. The cached entry deliberately does
    // not invalidate, so the symlink call (if it canonicalizes
    // correctly) returns the stale clean value.
    std::fs::write(real.join("file.txt"), b"modified-after-prime\n").expect("dirty the worktree");

    // Call via the symlink alias. With canonicalization, the
    // alias canonicalizes to the real path and hits the
    // cached clean entry. Without it, the alias misses, re-
    // probes, and surfaces the new dirt as `*-dirty`.
    let alias_result =
        super::super::detect_kernel_commit(&alias).expect("alias-path probe must yield Some");
    assert!(
        !alias_result.ends_with("-dirty"),
        "alias call must hit the cached pre-dirt entry — a \
         `-dirty` suffix proves the alias bypassed the cache \
         and re-probed the now-dirty repo, which is the \
         regression this test guards against. got {alias_result:?}",
    );
    assert_eq!(
        alias_result, real_clean,
        "alias call must return the EXACT cached clean value \
         from the canonical-path probe; got alias={alias_result:?}, \
         cached={real_clean:?}",
    );
}

/// Helper for `resolve_kernel_source_dir_with_cache` tests:
/// build a [`KernelMetadata`] for a Local-source entry. Used
/// across the fallback-scan and tarball-priority tests.
fn local_metadata_with_source_tree(
    version: &str,
    source_tree_path: std::path::PathBuf,
) -> crate::cache::KernelMetadata {
    crate::cache::KernelMetadata::new(
        crate::cache::KernelSource::Local {
            source_tree_path: Some(source_tree_path),
            git_hash: None,
        },
        std::env::consts::ARCH.to_string(),
        "bzImage".to_string(),
        "2026-04-26T00:00:00Z".to_string(),
    )
    .with_version(Some(version.to_string()))
    .with_config_hash(Some("abc123".to_string()))
    .with_ktstr_kconfig_hash(Some("def456".to_string()))
}

/// Helper: build a fake kernel image file under `dir` and
/// return its path. Cache `store()` requires an existing image
/// file to copy into the entry directory.
fn create_fake_image_in(dir: &std::path::Path) -> std::path::PathBuf {
    let image = dir.join("bzImage");
    std::fs::write(&image, b"fake kernel image").expect("write fake image");
    image
}

/// Tarball-shaped lookup hit yields the entry's source_tree_path
/// directly when the entry is a Local source. Pins the fast
/// path of the Version arm before exercising the fallback scan.
#[test]
fn resolve_kernel_source_dir_with_cache_version_tarball_key_local_source() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let src = tempfile::TempDir::new().expect("src tempdir");
    let image_dir = tempfile::TempDir::new().expect("image tempdir");
    let image = create_fake_image_in(image_dir.path());

    let arch = std::env::consts::ARCH;
    let key = format!("6.14.2-tarball-{arch}-kc{}", crate::cache_key_suffix());
    let meta = local_metadata_with_source_tree("6.14.2", src.path().to_path_buf());
    cache
        .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
        .expect("store cache entry");

    let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert_eq!(
        resolved.as_deref(),
        Some(src.path()),
        "tarball-shaped Local entry must resolve via direct lookup",
    );
}

/// Fallback scan: the tarball-shaped key is absent, but a
/// non-tarball cache entry (e.g. one stored under a `local-`
/// or git-shaped key) carries a matching version + Local
/// source_tree_path. The Version arm must find it via the
/// list-and-match fallback.
#[test]
fn resolve_kernel_source_dir_with_cache_version_falls_back_to_scan_for_local() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let src = tempfile::TempDir::new().expect("src tempdir");
    let image_dir = tempfile::TempDir::new().expect("image tempdir");
    let image = create_fake_image_in(image_dir.path());

    // Store under a non-tarball key shape — mimics a build
    // driven by `--kernel /path/to/linux`.
    let key = format!(
        "local-deadbee-{arch}-kc{suffix}",
        arch = std::env::consts::ARCH,
        suffix = crate::cache_key_suffix(),
    );
    let meta = local_metadata_with_source_tree("6.14.2", src.path().to_path_buf());
    cache
        .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
        .expect("store cache entry");

    let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert_eq!(
        resolved.as_deref(),
        Some(src.path()),
        "fallback scan must find a Local entry by version when \
         the tarball-shaped lookup misses",
    );
}

/// Fallback scan must SKIP non-Local entries even when the
/// version matches. A Tarball or Git entry has no on-disk
/// source tree to probe, so iterating past it to find the
/// Local sibling (or returning `None` when no Local exists)
/// is the correct behavior.
#[test]
fn resolve_kernel_source_dir_with_cache_version_skips_non_local_in_fallback() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let image_dir = tempfile::TempDir::new().expect("image tempdir");
    let image = create_fake_image_in(image_dir.path());

    // Store ONE entry: a tarball-source entry under a non-
    // tarball cache-key shape (so the direct lookup misses and
    // we hit the fallback scan). Version matches the query but
    // source is Tarball, so resolve must yield None.
    let key = format!(
        "weird-key-{arch}-kc{suffix}",
        arch = std::env::consts::ARCH,
        suffix = crate::cache_key_suffix(),
    );
    let meta = crate::cache::KernelMetadata::new(
        crate::cache::KernelSource::Tarball,
        std::env::consts::ARCH.to_string(),
        "bzImage".to_string(),
        "2026-04-26T00:00:00Z".to_string(),
    )
    .with_version(Some("6.14.2".to_string()))
    .with_config_hash(Some("abc123".to_string()))
    .with_ktstr_kconfig_hash(Some("def456".to_string()));
    cache
        .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
        .expect("store cache entry");

    let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert!(
        resolved.is_none(),
        "non-Local entries are transient and must not be returned by the fallback scan; got {resolved:?}",
    );
}

/// Version mismatch: even a Local entry with source_tree_path
/// is skipped when its `metadata.version` differs from the
/// queried version. Pinning this prevents a regression where
/// the fallback scan returns the first Local entry regardless
/// of version (collapsing every Version query to the same
/// path).
#[test]
fn resolve_kernel_source_dir_with_cache_version_skips_mismatched_version_in_fallback() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let src = tempfile::TempDir::new().expect("src tempdir");
    let image_dir = tempfile::TempDir::new().expect("image tempdir");
    let image = create_fake_image_in(image_dir.path());

    let key = format!(
        "local-deadbee-{arch}-kc{suffix}",
        arch = std::env::consts::ARCH,
        suffix = crate::cache_key_suffix(),
    );
    let meta = local_metadata_with_source_tree("6.13.0", src.path().to_path_buf());
    cache
        .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
        .expect("store cache entry");

    let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert!(
        resolved.is_none(),
        "Local entry with mismatched version must not be returned; got {resolved:?}",
    );
}

/// `KernelId::CacheKey` resolves via direct cache.lookup — no
/// fallback scan needed because the key already encodes every
/// detail (source-type prefix, arch, kconfig hash). Pinning
/// the CacheKey arm against a Local entry stored under that
/// exact key.
#[test]
fn resolve_kernel_source_dir_with_cache_cache_key_direct_lookup_local() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let src = tempfile::TempDir::new().expect("src tempdir");
    let image_dir = tempfile::TempDir::new().expect("image tempdir");
    let image = create_fake_image_in(image_dir.path());

    let key = format!(
        "local-deadbee-{arch}-kc{suffix}",
        arch = std::env::consts::ARCH,
        suffix = crate::cache_key_suffix(),
    );
    let meta = local_metadata_with_source_tree("6.14.2", src.path().to_path_buf());
    cache
        .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
        .expect("store cache entry");

    let id = crate::kernel_path::KernelId::CacheKey(key);
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert_eq!(resolved.as_deref(), Some(src.path()));
}

/// CacheKey lookup against a non-Local entry yields None — no
/// transient source tree to probe.
#[test]
fn resolve_kernel_source_dir_with_cache_cache_key_non_local_yields_none() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let image_dir = tempfile::TempDir::new().expect("image tempdir");
    let image = create_fake_image_in(image_dir.path());

    let key = format!(
        "main-git-deadbee-{arch}-kc{suffix}",
        arch = std::env::consts::ARCH,
        suffix = crate::cache_key_suffix(),
    );
    let meta = crate::cache::KernelMetadata::new(
        crate::cache::KernelSource::Git {
            git_hash: Some("deadbee".to_string()),
            git_ref: Some("main".to_string()),
        },
        std::env::consts::ARCH.to_string(),
        "bzImage".to_string(),
        "2026-04-26T00:00:00Z".to_string(),
    )
    .with_version(Some("6.14.2".to_string()))
    .with_config_hash(Some("abc123".to_string()))
    .with_ktstr_kconfig_hash(Some("def456".to_string()));
    cache
        .store(&key, &crate::cache::CacheArtifacts::new(&image), &meta)
        .expect("store cache entry");

    let id = crate::kernel_path::KernelId::CacheKey(key);
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert!(
        resolved.is_none(),
        "Git source has no persisted source tree; got {resolved:?}",
    );
}

/// Empty cache + Version query yields None. Sanity check
/// against a regression that crashes on an empty entries list.
#[test]
fn resolve_kernel_source_dir_with_cache_version_empty_cache_yields_none() {
    let cache_root = tempfile::TempDir::new().expect("cache tempdir");
    let cache = crate::cache::CacheDir::with_root(cache_root.path().to_path_buf());
    let id = crate::kernel_path::KernelId::Version("6.14.2".to_string());
    let resolved = super::super::resolve_kernel_source_dir_with_cache(&id, &cache);
    assert!(resolved.is_none());
}

// -- resolve_kernel_source_dir Path arm --
//
// The Path arm at sidecar.rs:1641 routes via the shared
// `cache::recover_local_source_tree` helper: when
// `KTSTR_KERNEL` points at a CACHE ENTRY directory (the shape
// `cargo-ktstr` exports for clean Path specs), the helper
// reads `metadata.json` and returns the recorded
// `source_tree_path`. When the env value is itself a SOURCE
// TREE (no metadata.json — the dirty path) or the metadata
// doesn't carry a `KernelSource::Local::source_tree_path`,
// the arm falls back to the env value verbatim.

/// (a) Path env points at a cache entry whose metadata.json
/// carries `KernelSource::Local::source_tree_path`. Resolver
/// returns the source-tree path so commit detection probes
/// the actual git repo, not the cache entry dir.
#[test]
fn resolve_kernel_source_dir_path_metadata_local_returns_source_tree() {
    use super::super::super::test_helpers::{EnvVarGuard, lock_env};
    let _lock = lock_env();
    // Cache entry dir + planted metadata.json pointing at the
    // (separate) source tree.
    let cache_entry = tempfile::TempDir::new().expect("cache entry tempdir");
    let src_tree = tempfile::TempDir::new().expect("src tree tempdir");
    let meta = local_metadata_with_source_tree("6.14.2", src_tree.path().to_path_buf());
    std::fs::write(
        cache_entry.path().join("metadata.json"),
        serde_json::to_string(&meta).expect("serialize metadata"),
    )
    .expect("write metadata.json");

    let _guard = EnvVarGuard::set("KTSTR_KERNEL", cache_entry.path());
    assert_eq!(
        super::super::resolve_kernel_source_dir().as_deref(),
        Some(src_tree.path()),
        "Path arm must recover source_tree_path from metadata.json \
         when the env value points at a cache entry with a Local source",
    );
}

/// (b) Path env with no metadata.json present: resolver falls
/// back to the env value verbatim. Mirrors the dirty-source-
/// tree case where `cargo ktstr test --kernel /path/to/linux`
/// skipped the cache store and `KTSTR_KERNEL` is the source
/// tree itself.
#[test]
fn resolve_kernel_source_dir_path_no_metadata_returns_env_value() {
    use super::super::super::test_helpers::{EnvVarGuard, lock_env};
    let _lock = lock_env();
    let dir = tempfile::TempDir::new().expect("dir tempdir");
    // Deliberately no metadata.json — `recover_local_source_tree`
    // returns None and the Path arm's fallback kicks in.

    let _guard = EnvVarGuard::set("KTSTR_KERNEL", dir.path());
    assert_eq!(
        super::super::resolve_kernel_source_dir().as_deref(),
        Some(dir.path()),
        "Path arm with no metadata.json must return the env value verbatim",
    );
}

/// (c) Path env points at a cache entry whose metadata.json
/// carries a non-Local source (`KernelSource::Tarball`). The
/// helper short-circuits to None inside
/// `recover_local_source_tree`; the arm's fallback returns the
/// env value verbatim. Tarball entries lack a persisted
/// source tree, so probing the cache-entry directory itself
/// (an extracted tarball) is the only available answer.
#[test]
fn resolve_kernel_source_dir_path_metadata_non_local_falls_through() {
    use super::super::super::test_helpers::{EnvVarGuard, lock_env};
    let _lock = lock_env();
    let cache_entry = tempfile::TempDir::new().expect("cache entry tempdir");
    let meta = crate::cache::KernelMetadata::new(
        crate::cache::KernelSource::Tarball,
        std::env::consts::ARCH.to_string(),
        "bzImage".to_string(),
        "2026-04-26T00:00:00Z".to_string(),
    )
    .with_version(Some("6.14.2".to_string()));
    std::fs::write(
        cache_entry.path().join("metadata.json"),
        serde_json::to_string(&meta).expect("serialize metadata"),
    )
    .expect("write metadata.json");

    let _guard = EnvVarGuard::set("KTSTR_KERNEL", cache_entry.path());
    assert_eq!(
        super::super::resolve_kernel_source_dir().as_deref(),
        Some(cache_entry.path()),
        "Path arm with non-Local source metadata must fall back \
         to the env value verbatim — Tarball entries have no \
         persisted source_tree_path to recover",
    );
}
