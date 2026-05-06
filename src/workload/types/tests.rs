//! Tests for `WorkType`, `Phase`, `WorkTypeValidationError`, and the
//! WorkType naming surface (`from_name`, `suggest`, `ALL_NAMES`).
//! Co-located with the type definitions in `types/mod.rs`.

#![cfg(test)]
#![allow(unused_imports)]

use super::super::affinity::*;
use super::super::config::*;
use super::super::spawn::WorkerReport;
use super::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Stub closure for [`WorkType::custom`] naming tests. Returns
/// a zeroed [`WorkerReport`] so the tests exercise the
/// enum-variant plumbing (`name`, `worker_group_size`) without
/// spawning a worker.
fn stub_custom_fn(_stop: &AtomicBool) -> WorkerReport {
    WorkerReport {
        tid: 0,
        work_units: 0,
        cpu_time_ns: 0,
        wall_time_ns: 0,
        off_cpu_ns: 0,
        migration_count: 0,
        cpus_used: BTreeSet::new(),
        migrations: vec![],
        max_gap_ms: 0,
        max_gap_cpu: 0,
        max_gap_at_ms: 0,
        resume_latencies_ns: vec![],
        wake_sample_total: 0,
        iteration_costs_ns: vec![],
        iteration_cost_sample_total: 0,
        iterations: 0,
        schedstat_run_delay_ns: 0,
        schedstat_run_count: 0,
        schedstat_cpu_time_ns: 0,
        completed: true,
        numa_pages: BTreeMap::new(),
        vmstat_numa_pages_migrated: 0,
        exit_info: None,
        is_messenger: false,
        group_idx: 0,
        affinity_error: None,
    }
}

#[test]
fn work_type_name_roundtrip() {
    for &name in WorkType::ALL_NAMES {
        // Sequence and Custom have no default from_name.
        if name == "Sequence" || name == "Custom" {
            assert!(WorkType::from_name(name).is_none());
            continue;
        }
        let wt = WorkType::from_name(name).unwrap();
        assert_eq!(wt.name(), name);
    }
}
#[test]
fn work_type_from_name_unknown() {
    assert!(WorkType::from_name("Nonexistent").is_none());
}
/// Each new IO variant's PascalCase name round-trips through
/// `from_name` back to the same variant. Pins the bidirectional
/// API contract — a regression that drops one of the 3 names
/// from `from_name`'s match would surface here rather than at
/// CLI parse time.
#[test]
fn io_variant_names_round_trip() {
    for name in ["IoSyncWrite", "IoRandRead", "IoConvoy"] {
        let wt = WorkType::from_name(name)
            .unwrap_or_else(|| panic!("from_name({name:?}) returned None"));
        assert_eq!(
            wt.name(),
            name,
            "round-trip mismatch: from_name({name:?}).name() == {:?}",
            wt.name()
        );
    }
}
#[test]
fn work_type_bursty_defaults() {
    let wt = WorkType::from_name("Bursty").unwrap();
    if let WorkType::Bursty {
        burst_duration,
        sleep_duration,
    } = wt
    {
        assert_eq!(burst_duration, Duration::from_millis(50));
        assert_eq!(sleep_duration, Duration::from_millis(100));
    } else {
        panic!("expected Bursty variant");
    }
}
#[test]
fn work_type_pipeio_defaults() {
    let wt = WorkType::from_name("PipeIo").unwrap();
    if let WorkType::PipeIo { burst_iters } = wt {
        assert_eq!(burst_iters, 1024);
    } else {
        panic!("expected PipeIo variant");
    }
}
#[test]
fn work_type_debug_shows_field_values() {
    let s = format!(
        "{:?}",
        WorkType::Bursty {
            burst_duration: Duration::from_millis(10),
            sleep_duration: Duration::from_millis(20),
        }
    );
    assert!(s.contains("10"), "must show burst_duration value");
    assert!(s.contains("20"), "must show sleep_duration value");
    // Different field values must produce different output.
    let s2 = format!(
        "{:?}",
        WorkType::Bursty {
            burst_duration: Duration::from_millis(99),
            sleep_duration: Duration::from_millis(1),
        }
    );
    assert!(s2.contains("99"), "must show changed burst_duration");
    assert!(s2.contains("1"), "must show changed sleep_duration");
    assert_ne!(
        s, s2,
        "different field values must produce different debug output"
    );
}
#[test]
fn work_type_clone_preserves_variant() {
    let a = WorkType::PipeIo { burst_iters: 512 };
    let b = a.clone();
    match b {
        WorkType::PipeIo { burst_iters } => assert_eq!(burst_iters, 512),
        _ => panic!("clone must preserve variant and fields"),
    }
}
// -- worker_group_size --

#[test]
fn worker_group_size_paired() {
    assert_eq!(WorkType::pipe_io(100).worker_group_size(), Some(2));
    assert_eq!(WorkType::futex_ping_pong(100).worker_group_size(), Some(2));
    assert_eq!(WorkType::cache_pipe(32, 100).worker_group_size(), Some(2));
}
#[test]
fn worker_group_size_fan_out() {
    assert_eq!(WorkType::futex_fan_out(4, 100).worker_group_size(), Some(5));
    assert_eq!(WorkType::futex_fan_out(1, 100).worker_group_size(), Some(2));
}
#[test]
fn worker_group_size_thundering_herd() {
    // ThunderingHerd collapses every worker into one group:
    // `waiters + 1` (1 waker + N waiters).
    let th = WorkType::thundering_herd(7, 1000, 5);
    assert_eq!(th.worker_group_size(), Some(8));
}
// -- needs_shared_mem --

#[test]
fn needs_shared_mem_futex_types() {
    assert!(WorkType::futex_ping_pong(100).needs_shared_mem());
    assert!(WorkType::futex_fan_out(4, 100).needs_shared_mem());
}
// -- needs_cache_buf --

#[test]
fn needs_cache_buf_cache_types() {
    assert!(WorkType::cache_pressure(32, 64).needs_cache_buf());
    assert!(WorkType::cache_yield(32, 64).needs_cache_buf());
    assert!(WorkType::cache_pipe(32, 100).needs_cache_buf());
}
#[test]
fn custom_name_returns_label() {
    let wt = WorkType::custom("my_work", stub_custom_fn);
    assert_eq!(wt.name(), "my_work");
}
#[test]
fn custom_group_size_is_none() {
    let wt = WorkType::custom("x", stub_custom_fn);
    assert_eq!(wt.worker_group_size(), None);
}
// -- FanOutCompute tests --

#[test]
fn fan_out_compute_name() {
    let wt = WorkType::FanOutCompute {
        fan_out: 4,
        cache_footprint_kb: 256,
        operations: 5,
        sleep_usec: 100,
    };
    assert_eq!(wt.name(), "FanOutCompute");
}
#[test]
fn fan_out_compute_from_name() {
    let wt = WorkType::from_name("FanOutCompute").unwrap();
    match wt {
        WorkType::FanOutCompute {
            fan_out,
            cache_footprint_kb,
            operations,
            sleep_usec,
        } => {
            assert_eq!(fan_out, 4);
            assert_eq!(cache_footprint_kb, 256);
            assert_eq!(operations, 5);
            assert_eq!(sleep_usec, 100);
        }
        _ => panic!("expected FanOutCompute"),
    }
}
#[test]
fn fan_out_compute_group_size() {
    let wt = WorkType::fan_out_compute(4, 256, 5, 100);
    assert_eq!(wt.worker_group_size(), Some(5));
    let wt2 = WorkType::fan_out_compute(1, 256, 5, 100);
    assert_eq!(wt2.worker_group_size(), Some(2));
}
#[test]
fn fan_out_compute_needs_shared_mem() {
    assert!(WorkType::fan_out_compute(4, 256, 5, 100).needs_shared_mem());
}
#[test]
fn fan_out_compute_needs_cache_buf() {
    assert!(WorkType::fan_out_compute(4, 256, 5, 100).needs_cache_buf());
}
// -- PageFaultChurn tests --

#[test]
fn page_fault_churn_name_roundtrip() {
    let wt = WorkType::from_name("PageFaultChurn").unwrap();
    assert_eq!(wt.name(), "PageFaultChurn");
}
#[test]
fn page_fault_churn_from_name_defaults() {
    let wt = WorkType::from_name("PageFaultChurn").unwrap();
    match wt {
        WorkType::PageFaultChurn {
            region_kb,
            touches_per_cycle,
            spin_iters,
        } => {
            assert_eq!(region_kb, 4096);
            assert_eq!(touches_per_cycle, 256);
            assert_eq!(spin_iters, 64);
        }
        _ => panic!("expected PageFaultChurn"),
    }
}
#[test]
fn page_fault_churn_group_size_none() {
    let wt = WorkType::page_fault_churn(4096, 256, 64);
    assert_eq!(wt.worker_group_size(), None);
}
#[test]
fn page_fault_churn_no_shared_mem() {
    assert!(!WorkType::page_fault_churn(4096, 256, 64).needs_shared_mem());
}
#[test]
fn page_fault_churn_no_cache_buf() {
    assert!(!WorkType::page_fault_churn(4096, 256, 64).needs_cache_buf());
}
// -- MutexContention tests --

#[test]
fn mutex_contention_name_roundtrip() {
    let wt = WorkType::from_name("MutexContention").unwrap();
    assert_eq!(wt.name(), "MutexContention");
}
#[test]
fn mutex_contention_from_name_defaults() {
    let wt = WorkType::from_name("MutexContention").unwrap();
    match wt {
        WorkType::MutexContention {
            contenders,
            hold_iters,
            work_iters,
        } => {
            assert_eq!(contenders, 4);
            assert_eq!(hold_iters, 256);
            assert_eq!(work_iters, 1024);
        }
        _ => panic!("expected MutexContention"),
    }
}
#[test]
fn mutex_contention_group_size() {
    let wt = WorkType::mutex_contention(4, 256, 1024);
    assert_eq!(wt.worker_group_size(), Some(4));
    let wt2 = WorkType::mutex_contention(8, 256, 1024);
    assert_eq!(wt2.worker_group_size(), Some(8));
}
#[test]
fn mutex_contention_needs_shared_mem() {
    assert!(WorkType::mutex_contention(4, 256, 1024).needs_shared_mem());
}
#[test]
fn mutex_contention_no_cache_buf() {
    assert!(!WorkType::mutex_contention(4, 256, 1024).needs_cache_buf());
}
/// `Phase` Duration fields serialize as humantime strings, not
/// `{secs, nanos}` objects. Pins the readable wire format that
/// makes captured `WorkSpec` configs operator-editable.
#[test]
fn phase_duration_serializes_as_humantime() {
    let p = Phase::Spin(Duration::from_millis(100));
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(json, r#"{"spin":"100ms"}"#);
    let back: Phase = serde_json::from_str(&json).unwrap();
    match back {
        Phase::Spin(d) => assert_eq!(d, Duration::from_millis(100)),
        _ => panic!("roundtrip lost Spin variant"),
    }
}
/// Table-driven JSON roundtrip for every entry in
/// [`WorkType::ALL_NAMES`] that [`WorkType::from_name`] can
/// construct with default parameters. The single iteration form
/// catches three regression classes a per-variant test would
/// miss:
///
/// - A new variant added without a `from_name` default arm: the
///   `name` lands in [`WorkType::ALL_NAMES`] (driven by the
///   `strum::VariantNames` derive on the enum) but `from_name`
///   returns `None`. The walk asserts that every name except
///   the two documented exceptions (`Sequence`, `Custom`) is
///   constructible.
/// - A variant whose serialized JSON does not deserialize back
///   to the same shape — e.g. a missing `#[serde(rename)]`,
///   a Duration field that lost its `humantime_serde_helper`
///   wrapper, or a default that drifted between `from_name` and
///   the enum's struct-literal form. Each name is serialized,
///   deserialized, and re-serialized; the second JSON string
///   must equal the first.
/// - A renaming of the enum's wire form (snake_case key) that
///   silently drops `from_name`'s lookup. Re-serializing the
///   `from_name` output and comparing the two strings catches
///   this without a hard-coded JSON literal per variant — those
///   live in [`workload_enums_use_snake_case_wire_format`] for
///   the unit-form variants.
///
/// `Sequence` is excluded from the `from_name` walk because
/// `from_name` deliberately refuses to construct it (it has no
/// natural default — phases are mandatory). The `Sequence` arm
/// is exercised by an explicit struct-literal at the end of the
/// test so the test still covers every WorkType variant by
/// kind, not just by `from_name`-reachability. `Custom` is
/// `#[serde(skip)]` and covered by
/// [`worktype_custom_serialize_errors_skipped_variant`].
///
/// Comparison uses re-serialized JSON strings rather than
/// `PartialEq` because `WorkType` does not derive `PartialEq`
/// (its `Custom` variant carries a non-comparable `fn`
/// pointer). The same pattern is used in
/// [`workload_config_default_roundtrips`] below.
#[test]
fn worktype_serde_roundtrip_table_driven() {
    // `Sequence` and `Custom` are exempt from the from_name
    // walk: see the doc comment for the rationale.
    const FROM_NAME_EXCLUSIONS: &[&str] = &["Sequence", "Custom"];

    let mut covered = 0;
    let mut excluded = 0;
    for name in WorkType::ALL_NAMES {
        if FROM_NAME_EXCLUSIONS.contains(name) {
            excluded += 1;
            continue;
        }
        let wt = WorkType::from_name(name).unwrap_or_else(|| {
            panic!(
                "WorkType::from_name({name:?}) returned None — every \
                 name in ALL_NAMES outside the documented exclusions \
                 must have a from_name arm"
            )
        });
        let json =
            serde_json::to_string(&wt).unwrap_or_else(|e| panic!("serialize {name:?} failed: {e}"));
        let back: WorkType = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize {name:?} failed: {e}; json was {json}"));
        let json2 = serde_json::to_string(&back)
            .unwrap_or_else(|e| panic!("re-serialize {name:?} failed after roundtrip: {e}"));
        assert_eq!(
            json, json2,
            "WorkType::{name} JSON roundtrip drift: \
             original={json}, after-roundtrip={json2}"
        );
        // #[serde(skip)] fields don't appear in `json` so the
        // roundtrip can't preserve their values from the wire.
        // Debug renders every field — including skipped ones.
        // If a future variant gains a #[serde(skip)] field that
        // doesn't have Default::default()-equivalent
        // initialization in its from_name arm, the roundtripped
        // variant will reconstruct that field as the type's
        // default while the from_name-constructed one carries
        // the documented default — and the Debug strings will
        // diverge here. Catches the drift BEFORE a downstream
        // consumer tries to roundtrip a config and silently
        // loses the skipped state.
        let from_name_dbg = format!("{wt:?}");
        let after_roundtrip_dbg = format!("{back:?}");
        assert_eq!(
            from_name_dbg, after_roundtrip_dbg,
            "WorkType::{name} Debug drift across serde roundtrip: \
             a #[serde(skip)] field is reconstructing differently \
             than from_name produces. from_name={from_name_dbg}, \
             after-roundtrip={after_roundtrip_dbg}"
        );
        covered += 1;
    }

    // Every name in ALL_NAMES is accounted for: covered +
    // excluded must equal the full list. Catches a future
    // exclusion that was added to FROM_NAME_EXCLUSIONS but
    // never landed in ALL_NAMES (e.g. a typo) — the loop
    // would silently skip nothing and `excluded` would
    // mismatch the constant length.
    assert_eq!(
        covered + excluded,
        WorkType::ALL_NAMES.len(),
        "table walk dropped variants: covered={covered}, \
         excluded={excluded}, ALL_NAMES.len()={}",
        WorkType::ALL_NAMES.len()
    );
    assert_eq!(
        excluded,
        FROM_NAME_EXCLUSIONS.len(),
        "FROM_NAME_EXCLUSIONS contains {excluded} entries that \
         were not seen in ALL_NAMES; one was likely typo'd or \
         renamed"
    );

    // Sequence is excluded from the from_name walk; cover it
    // here with an explicit construction so the roundtrip is
    // still proven for every kind of variant.
    let seq = WorkType::Sequence {
        first: Phase::Spin(Duration::from_millis(10)),
        rest: vec![
            Phase::Sleep(Duration::from_millis(5)),
            Phase::Yield(Duration::from_millis(2)),
            Phase::Io(Duration::from_millis(1)),
        ],
    };
    let json = serde_json::to_string(&seq).unwrap();
    let back: WorkType = serde_json::from_str(&json).unwrap();
    let json2 = serde_json::to_string(&back).unwrap();
    assert_eq!(
        json, json2,
        "WorkType::Sequence roundtrip drift: original={json}, \
         after-roundtrip={json2}"
    );
    // Pin one humantime field's wire form so a regression that
    // drops `humantime_serde_helper` from `Phase` surfaces
    // here even if the other Phase tests are skipped.
    assert!(
        json.contains(r#""spin":"10ms""#),
        "Sequence first-phase Phase::Spin must serialize \
         through humantime_serde_helper as \"10ms\"; got {json}"
    );
}
/// `WorkType::IpcVariance` constructor rejects every
/// zero-value field with the typed
/// [`WorkTypeValidationError::ZeroIpcVarianceParam`] variant.
/// The same rejection fires at spawn time for variants
/// constructed via the struct-literal form — pin both paths
/// produce the same error variant.
#[test]
fn ipc_variance_constructor_rejects_zeros() {
    let err = WorkType::ipc_variance(0, 1, 1).expect_err("hot_iters=0 must reject");
    assert!(
        matches!(
            err,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "hot_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ hot_iters }}; got: {err:?}",
    );

    let err = WorkType::ipc_variance(1, 0, 1).expect_err("cold_iters=0 must reject");
    assert!(
        matches!(
            err,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "cold_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ cold_iters }}; got: {err:?}",
    );

    let err = WorkType::ipc_variance(1, 1, 0).expect_err("period_iters=0 must reject");
    assert!(
        matches!(
            err,
            WorkTypeValidationError::ZeroIpcVarianceParam {
                field: "period_iters",
                group_idx: 0,
            }
        ),
        "expected ZeroIpcVarianceParam {{ period_iters }}; got: {err:?}",
    );

    // Positive case: all three > 0 yields Ok.
    let wt = WorkType::ipc_variance(1, 1, 1).expect("all positive must construct");
    assert!(matches!(wt, WorkType::IpcVariance { .. }));
}

#[test]
fn suggest_then_from_name_roundtrips_for_buildable_variants() {
    // Lowercase user input: from_name misses, suggest hits,
    // from_name on the canonical spelling succeeds.
    assert!(WorkType::from_name("spinwait").is_none());
    let canonical = WorkType::suggest("spinwait").expect("suggest must find SpinWait");
    assert_eq!(canonical, "SpinWait");
    let wt = WorkType::from_name(canonical).expect("from_name must build from canonical spelling");
    assert!(matches!(wt, WorkType::SpinWait));

    // Uppercase user input roundtrips too.
    assert!(WorkType::from_name("YIELDHEAVY").is_none());
    let canonical = WorkType::suggest("YIELDHEAVY").expect("suggest must find YieldHeavy");
    assert_eq!(canonical, "YieldHeavy");
    let wt = WorkType::from_name(canonical).expect("from_name must build");
    assert!(matches!(wt, WorkType::YieldHeavy));

    // Sequence and Custom are suggest-only: suggest emits them
    // so a diagnostic can name them, but from_name returns None
    // because they need explicit phases / function pointers that
    // a bare string cannot carry.
    assert_eq!(WorkType::suggest("sequence"), Some("Sequence"));
    assert!(WorkType::from_name("Sequence").is_none());
    assert_eq!(WorkType::suggest("custom"), Some("Custom"));
    assert!(WorkType::from_name("Custom").is_none());
}

#[test]
fn suggest_is_case_insensitive_and_canonical() {
    assert_eq!(WorkType::suggest("spinwait"), Some("SpinWait"));
    assert_eq!(WorkType::suggest("SPINWAIT"), Some("SpinWait"));
    assert_eq!(WorkType::suggest("SpinWait"), Some("SpinWait"));
    assert_eq!(WorkType::suggest("YIELDHEAVY"), Some("YieldHeavy"));
    assert_eq!(WorkType::suggest("sequence"), Some("Sequence"));
    assert_eq!(WorkType::suggest("custom"), Some("Custom"));
    assert!(WorkType::suggest("nonexistent").is_none());
    assert!(WorkType::suggest("").is_none());
    assert!(WorkType::suggest("cpu").is_none());
}

/// Surrounding / embedded whitespace must NOT silently resolve
/// to a canonical name. The helper's doc commits to strict
/// (non-trimming) matching so a caller that passes unsanitized
/// user input like `" SpinWait"` or `"SpinWait\n"` sees `None`.
#[test]
fn suggest_rejects_whitespace_padded_inputs() {
    assert!(WorkType::suggest(" SpinWait").is_none());
    assert!(WorkType::suggest("SpinWait ").is_none());
    assert!(WorkType::suggest(" SpinWait ").is_none());
    assert!(WorkType::suggest("SpinWait\n").is_none());
    assert!(WorkType::suggest("\tSpinWait").is_none());
    assert!(WorkType::suggest("SpinWait\t").is_none());
    assert!(WorkType::suggest("Cpu Spin").is_none());
    assert!(WorkType::suggest(" ").is_none());
    assert!(WorkType::suggest("\n").is_none());
    assert_eq!(WorkType::suggest("SpinWait"), Some("SpinWait"));
}

#[test]
fn work_type_all_names_count() {
    // 38 = 20 historical + 2 fundamental + 7 pathology + 5 coverage-gap
    //    + 1 idle-transition + 3 compute-primitive variants.
    assert_eq!(WorkType::ALL_NAMES.len(), 38);
}

#[test]
fn worker_group_size_ungrouped() {
    assert_eq!(WorkType::SpinWait.worker_group_size(), None);
    assert_eq!(WorkType::YieldHeavy.worker_group_size(), None);
    assert_eq!(WorkType::Mixed.worker_group_size(), None);
    assert_eq!(WorkType::IoSyncWrite.worker_group_size(), None);
    assert_eq!(WorkType::IoRandRead.worker_group_size(), None);
    assert_eq!(WorkType::IoConvoy.worker_group_size(), None);
    assert_eq!(
        WorkType::bursty(Duration::from_millis(50), Duration::from_millis(100)).worker_group_size(),
        None
    );
    assert_eq!(WorkType::cache_pressure(32, 64).worker_group_size(), None);
    assert_eq!(WorkType::cache_yield(32, 64).worker_group_size(), None);
}

#[test]
fn needs_cache_buf_non_cache() {
    assert!(!WorkType::SpinWait.needs_cache_buf());
    assert!(!WorkType::pipe_io(100).needs_cache_buf());
    assert!(!WorkType::futex_ping_pong(100).needs_cache_buf());
    assert!(!WorkType::futex_fan_out(4, 100).needs_cache_buf());
}

/// Per-enum `serde(rename_all = "snake_case")` discipline check —
/// a representative variant from each enum serializes to the
/// snake_case form. A regression that drops the `rename_all` on
/// any enum surfaces here.
#[test]
fn workload_enums_use_snake_case_wire_format() {
    let json = serde_json::to_string(&CloneMode::Fork).unwrap();
    assert_eq!(json, r#""fork""#);
    let json = serde_json::to_string(&CloneMode::Thread).unwrap();
    assert_eq!(json, r#""thread""#);

    let json = serde_json::to_string(&SchedPolicy::Normal).unwrap();
    assert_eq!(json, r#""normal""#);
    let json = serde_json::to_string(&SchedPolicy::RoundRobin(50)).unwrap();
    assert_eq!(json, r#"{"round_robin":50}"#);

    let json = serde_json::to_string(&FutexLockMode::Plain).unwrap();
    assert_eq!(json, r#""plain""#);

    let json = serde_json::to_string(&SchedClass::Cfs).unwrap();
    assert_eq!(json, r#""cfs""#);

    let json = serde_json::to_string(&MemPolicy::Default).unwrap();
    assert_eq!(json, r#""default""#);

    let json = serde_json::to_string(&AffinityIntent::Inherit).unwrap();
    assert_eq!(json, r#""inherit""#);
    let json = serde_json::to_string(&AffinityIntent::LlcAligned).unwrap();
    assert_eq!(json, r#""llc_aligned""#);

    let json = serde_json::to_string(&ResolvedAffinity::None).unwrap();
    assert_eq!(json, r#""none""#);

    let json = serde_json::to_string(&WorkType::SpinWait).unwrap();
    assert_eq!(json, r#""spin_wait""#);
    let json = serde_json::to_string(&WorkType::ForkExit).unwrap();
    assert_eq!(json, r#""fork_exit""#);
    let json = serde_json::to_string(&WorkType::IoSyncWrite).unwrap();
    assert_eq!(json, r#""io_sync_write""#);
    let json = serde_json::to_string(&WorkType::IoRandRead).unwrap();
    assert_eq!(json, r#""io_rand_read""#);
    let json = serde_json::to_string(&WorkType::IoConvoy).unwrap();
    assert_eq!(json, r#""io_convoy""#);
    let back: WorkType = serde_json::from_str(r#""io_convoy""#).unwrap();
    assert!(matches!(back, WorkType::IoConvoy));
}

/// `WorkType::Custom` is `#[serde(skip)]` because its `run` field
/// is a `fn` pointer with no portable wire format. Serializing
/// fails with a serde error pointing at the skipped variant.
#[test]
fn worktype_custom_serialize_errors_skipped_variant() {
    fn noop(_: &AtomicBool) -> WorkerReport {
        WorkerReport::default()
    }
    let custom = WorkType::custom("my_custom", noop);
    let r = serde_json::to_string(&custom);
    assert!(
        r.is_err(),
        "Custom variant must error on serialize (it's #[serde(skip)])"
    );
}
