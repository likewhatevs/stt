//! Unit coverage for [`super::EarlySnapshotGuard`] — the Drop-based
//! preservation wrapper that flushes a Captured early-trigger snapshot
//! to disk regardless of how the freeze coordinator's closure exits
//! (normal return, error return, panic-unwind).
//!
//! The guard exists to close a silent-drop window: the closure runs
//! on a spawned thread, and a panic anywhere in the closure body
//! would unwind past the end-of-coord drain. Every Captured early
//! MUST reach disk. These tests pin the Drop guarantee directly
//! without booting a VM — they construct the guard, drive it through
//! normal / explicit-drain / panic-unwind / dual-snapshot-disabled
//! scenarios, and verify the tagged sibling file appears (or does
//! not appear) at the [`snapshot_tagged_path`]-derived location.
//!
//! Test 1 covers the panic-unwind path landing at the default
//! NEVER_FIRED tag. Test 2 pins the idempotence contract — explicit
//! drain followed by Drop is a no-op (matters when the end-of-coord
//! drain runs cleanly and the guard then drops out of scope). Test 3
//! exercises the retain_tag override (the late-trigger Suppressed /
//! pre-late-degraded arms set retain_tag so the drain lands at the
//! operator-correct path, not NEVER_FIRED). Test 4 pins the
//! dual_snapshot=false gate so a single-snapshot test never
//! accidentally lands an empty tagged sibling on disk.
//!
//! Distinct from E2E coverage (which boots a VM and triggers a real
//! early-Captured + late-kill sequence): these are guard-level unit
//! tests, fast, and exercise the Drop semantics directly. Both are
//! needed.

use super::EarlySnapshotGuard;
use super::snapshot::snapshot_tagged_path;
use crate::monitor::bpf_map::BPF_MAP_TYPE_ARRAY;
use crate::monitor::btf_render::RenderedValue;
use crate::monitor::dump::{
    ALL_SNAPSHOT_TAGS, FailureDumpMap, FailureDumpReport, SCHEMA_DUAL,
    SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED, SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED,
    SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use tempfile::TempDir;

/// Build a synthetic FailureDumpReport with deterministic field
/// values so the post-deserialize assertions in Test 1 have a
/// distinguishable schema/maps shape rather than a default-zero
/// blob.
fn synthetic_report() -> FailureDumpReport {
    FailureDumpReport {
        schema: SCHEMA_DUAL.to_string(),
        maps: vec![FailureDumpMap {
            name: "synthetic.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            value_size: 8,
            max_entries: 1,
            value: Some(RenderedValue::Uint {
                bits: 32,
                value: 0xCAFE,
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
        sdt_alloc_unavailable: None,
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
    }
}

/// Build the `{base}.failure-dump.json` path inside a TempDir. The
/// `.failure-dump` suffix routes through `snapshot_tagged_path`'s
/// stem-strip so the resulting tagged sibling looks like
/// `coord.snapshot.{tag}.json`.
fn dump_base_path(dir: &TempDir) -> std::path::PathBuf {
    dir.path().join("coord.failure-dump.json")
}

/// Panic-unwind path: Drop flushes the held snapshot to the
/// NEVER_FIRED tagged sibling. Without the Drop guard, the panic
/// would unwind past the end-of-coord drain and the early snapshot
/// would disappear with the closure frame.
///
/// Pins the panic-safety invariant: every Captured early reaches
/// disk regardless of how the closure exits.
#[test]
fn early_snapshot_guard_drops_on_panic_unwind() {
    let tmp = TempDir::new().expect("tempdir");
    let dump_path = dump_base_path(&tmp);
    let synthetic = synthetic_report();
    // Serialize the synthetic ONCE before moving it into the
    // guard, so the post-deserialize all-fields check (below) can
    // compare against the original wire form. JSON-string equality
    // proves every populated FailureDumpReport field (those NOT
    // suppressed by `skip_serializing_if`) round-trips identically
    // through the drained file. Synthetic populates schema + maps;
    // the 20 other fields stay at empty-Vec / None defaults and are
    // suppressed from the wire JSON, so this leg proves both
    // populated-field survival AND suppressed-field-stays-suppressed.
    // Broader infallibility across max-populated shapes is proven
    // by `failure_dump_report_serialization_is_infallible_for_max_synthetic_input`
    // in src/monitor/dump/tests.rs.
    let synthetic_json = serde_json::to_string(&synthetic).expect("serialize synthetic");

    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = EarlySnapshotGuard {
            snapshot: Some(synthetic),
            retain_tag: None,
            dump_path: Some(dump_path.clone()),
            dual_snapshot: true,
        };
        panic!("inject — guard's Drop must still flush to disk");
    }));

    assert!(
        result.is_err(),
        "the closure must propagate the injected panic"
    );

    let expected = snapshot_tagged_path(&dump_path, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED);
    assert!(
        expected.exists(),
        "panic-unwind must land file at NEVER_FIRED tag: {}",
        expected.display()
    );

    let body = std::fs::read_to_string(&expected).expect("read drained file");
    let loaded: FailureDumpReport = serde_json::from_str(&body).expect("deserialize");
    let loaded_json = serde_json::to_string(&loaded).expect("re-serialize loaded");
    assert_eq!(
        loaded_json, synthetic_json,
        "populated FailureDumpReport fields (those NOT suppressed by \
         skip_serializing_if) must roundtrip identically via the drained \
         file — drift here means a field was dropped or mutated in the \
         serialize → write_to_tagged_path → read_to_string → deserialize \
         chain. Max-shape infallibility proof in \
         failure_dump_report_serialization_is_infallible_for_max_synthetic_input."
    );
}

/// Explicit drain followed by Drop is idempotent. The normal-exit
/// path calls `drain_to_disk()` directly before the guard falls
/// out of scope; the subsequent Drop must NOT re-write the file
/// (which would burn an extra syscall and could overwrite an
/// in-flight reader on the consumer side).
///
/// Pins the `snapshot.take()` idempotence contract: the first
/// drain consumes the report, the second drain sees `None` and
/// short-circuits without touching the filesystem. Mtime + size
/// equality across the Drop boundary proves no second write
/// occurred.
#[test]
fn early_snapshot_guard_drain_then_drop_is_idempotent() {
    let tmp = TempDir::new().expect("tempdir");
    let dump_path = dump_base_path(&tmp);
    let synthetic = synthetic_report();
    let expected = snapshot_tagged_path(&dump_path, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED);

    let (mtime_after_drain, len_after_drain) = {
        let mut guard = EarlySnapshotGuard {
            snapshot: Some(synthetic),
            retain_tag: None,
            dump_path: Some(dump_path.clone()),
            dual_snapshot: true,
        };
        guard.drain_to_disk();

        // Bonus per spec: the take() inside drain_to_disk consumes
        // the snapshot; a second drain (Drop) sees None and short-
        // circuits at the `else { return }` arm.
        assert!(
            guard.snapshot.is_none(),
            "drain_to_disk must take() the snapshot — None proves the gate"
        );

        let md = std::fs::metadata(&expected).expect("file landed by explicit drain");
        (md.modified().expect("mtime"), md.len())
        // guard falls out of scope here — Drop fires drain_to_disk
        // again, which short-circuits on snapshot=None.
    };

    let md_after_drop = std::fs::metadata(&expected).expect("file still present after Drop");
    assert_eq!(
        md_after_drop.len(),
        len_after_drain,
        "Drop after explicit drain must NOT alter the file size"
    );
    assert_eq!(
        md_after_drop.modified().expect("mtime"),
        mtime_after_drain,
        "Drop after explicit drain must NOT touch the file mtime — \
         a non-equal mtime proves a second write happened"
    );

    // Drain successfully outside catch_unwind to capture pre-panic
    // mtime, then move guard into the closure and panic so Drop
    // fires during unwind. Mtime equality across the unwind
    // boundary is the load-bearing detection signal —
    // synthetic_report() determinism makes a rewritten file
    // byte-identical, so size equality alone is tautological. A
    // regression that removes the `snapshot.take()` short-circuit
    // in drain_to_disk would fire a second write during Drop with
    // identical content; mtime moves forward, size stays equal —
    // only the mtime check catches this.
    let tmp2 = TempDir::new().expect("tempdir");
    let dump_path2 = dump_base_path(&tmp2);
    let expected2 = snapshot_tagged_path(&dump_path2, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED);

    let mut guard = EarlySnapshotGuard {
        snapshot: Some(synthetic_report()),
        retain_tag: None,
        dump_path: Some(dump_path2.clone()),
        dual_snapshot: true,
    };
    guard.drain_to_disk();
    let md_before_unwind = std::fs::metadata(&expected2).expect("file landed by drain");
    let mtime_before_unwind = md_before_unwind.modified().expect("mtime");

    // Move guard into the closure so Drop fires during unwind.
    let result = catch_unwind(AssertUnwindSafe(move || {
        let _guard = guard;
        panic!("inject: drain succeeded, now unwind with guard alive");
    }));
    assert!(
        result.is_err(),
        "injected panic must propagate after successful drain"
    );

    let md_after_unwind = std::fs::metadata(&expected2).expect("file still present after unwind");
    let mtime_after_unwind = md_after_unwind.modified().expect("mtime");
    assert_eq!(
        mtime_before_unwind, mtime_after_unwind,
        "Drop during unwind after explicit drain must NOT rewrite — \
         mtime change proves a second write fired even though the snapshot \
         was already consumed by the explicit drain (the snapshot.take() \
         short-circuit in drain_to_disk was removed)"
    );
}

/// Panic AFTER retain_tag was set lands the file at the operator-
/// correct path, NOT at NEVER_FIRED. The retain_tag mechanism
/// exists for the late-trigger Suppressed / pre-late-degraded
/// arms: when their tagged-sibling write fails mid-coord, the
/// guard's retain_tag is set so the end-of-coord drain (or panic-
/// unwind Drop) lands the recovered file at the operator-correct
/// tag rather than the default NEVER_FIRED.
///
/// Pins the retain_tag invariant: a retain_tag set before panic is
/// honored by Drop. Parametric across both retain_tag values the
/// production code sets (SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED from
/// the late-Degraded write-failure arm,
/// SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED from the late-Suppressed
/// write-failure arm).
///
/// The negative assertion (no file at NEVER_FIRED) catches a
/// double-write regression where Drop accidentally writes to BOTH
/// the retain_tag and the default path.
///
/// SNAPSHOT_TAG_EARLY_DEGRADED is deliberately excluded from the
/// parametric variants: the early-snapshot Degraded handler writes
/// to that tag DIRECTLY (not via the guard) in the early-Degraded
/// arm of the late-trigger dispatch, so it never appears as a
/// retain_tag value on the guard. The negative scan below asserts
/// NEVER_FIRED AND the other two arm tags AND EARLY_DEGRADED are
/// all absent — catches double-writes to any mismatched tag.
#[test]
fn early_snapshot_guard_drops_with_retain_tag_when_late_failed() {
    for &(tag, label) in &[
        (
            SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED,
            "early-pre-late-degraded",
        ),
        (
            SNAPSHOT_TAG_EARLY_ONLY_LATE_SUPPRESSED,
            "early-only-late-suppressed",
        ),
    ] {
        let tmp = TempDir::new().expect("tempdir");
        let dump_path = dump_base_path(&tmp);
        let synthetic = synthetic_report();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = EarlySnapshotGuard {
                snapshot: Some(synthetic),
                retain_tag: Some(tag),
                dump_path: Some(dump_path.clone()),
                dual_snapshot: true,
            };
            panic!("inject — guard's Drop must use retain_tag, not NEVER_FIRED");
        }));

        assert!(result.is_err(), "[{label}] catch_unwind must return Err");

        let expected = snapshot_tagged_path(&dump_path, tag);
        assert!(
            expected.exists(),
            "[{label}] retain_tag file must exist at {}",
            expected.display()
        );

        // Negative scan across ALL non-matching tags (NEVER_FIRED +
        // the other arm's tag + EARLY_DEGRADED). Catches double-write
        // regressions to any tag that shouldn't have fired. Iterates
        // ALL_SNAPSHOT_TAGS so a future SNAPSHOT_TAG_* added to the
        // dump module auto-flows into this negative scan without
        // touching this test.
        for negative_tag in ALL_SNAPSHOT_TAGS {
            if *negative_tag == tag {
                continue;
            }
            let negative_path = snapshot_tagged_path(&dump_path, negative_tag);
            assert!(
                !negative_path.exists(),
                "[{label}] retain_tag {tag:?} must override all others — \
                 no file may exist at {} (double-write to {negative_tag:?})",
                negative_path.display()
            );
        }
    }
}

/// dual_snapshot=false bypasses Drop entirely. The drain_to_disk
/// gate at the head of the function returns immediately on
/// `!self.dual_snapshot`, so even with Some(snapshot) +
/// Some(dump_path) no write occurs. Both normal-exit AND
/// panic-unwind paths must respect the gate — a regression that
/// dropped the gate from Drop would silently emit single-snapshot
/// runs with an unexpected tagged sibling on disk.
///
/// Pins the `!self.dual_snapshot` early-return at the head of
/// `EarlySnapshotGuard::drain_to_disk`. The negative scan across
/// every SNAPSHOT_TAG_* constant catches a regression that
/// re-routed the no-op case to a different default tag (the gate
/// must produce ZERO files, not just "no file at NEVER_FIRED").
#[test]
fn early_snapshot_guard_drop_no_op_when_dual_snapshot_disabled() {
    // Normal-exit variant.
    let tmp_normal = TempDir::new().expect("tempdir");
    let dump_path_normal = dump_base_path(&tmp_normal);
    {
        let _guard = EarlySnapshotGuard {
            snapshot: Some(synthetic_report()),
            retain_tag: None,
            dump_path: Some(dump_path_normal.clone()),
            dual_snapshot: false,
        };
    } // Drop fires — must NOT write anything.

    for tag in ALL_SNAPSHOT_TAGS {
        let path = snapshot_tagged_path(&dump_path_normal, tag);
        assert!(
            !path.exists(),
            "dual_snapshot=false (normal exit) must NOT write to {}",
            path.display()
        );
    }

    // Panic-unwind variant: even with retain_tag set, the gate
    // still short-circuits.
    let tmp_panic = TempDir::new().expect("tempdir");
    let dump_path_panic = dump_base_path(&tmp_panic);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = EarlySnapshotGuard {
            snapshot: Some(synthetic_report()),
            retain_tag: Some(SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED),
            dump_path: Some(dump_path_panic.clone()),
            dual_snapshot: false,
        };
        panic!("inject — dual_snapshot=false must still bypass the write");
    }));

    assert!(
        result.is_err(),
        "panic must propagate even with dual_snapshot=false"
    );

    for tag in ALL_SNAPSHOT_TAGS {
        let path = snapshot_tagged_path(&dump_path_panic, tag);
        assert!(
            !path.exists(),
            "dual_snapshot=false (panic unwind) must NOT write to {}",
            path.display()
        );
    }
}

/// dump_path=None gate: drain_to_disk early-returns at the
/// `let Some(dump_path) = self.dump_path.as_deref() else { return };`
/// arm without consuming the snapshot. Pins the dump_path gate AND
/// validates the ordering — the gate runs BEFORE
/// `self.snapshot.take()`, so the snapshot is preserved for a
/// future drain attempt rather than dropped on the floor.
///
/// The empty-directory assertion catches a regression where the
/// gate falls through to write_to_tagged_path with a phantom path
/// (e.g. derived from CWD or a hardcoded fallback) — every file
/// would then land in an operator-unreachable location.
#[test]
fn early_snapshot_guard_drain_no_op_when_dump_path_unset() {
    let tmp = TempDir::new().expect("tempdir");
    let mut guard = EarlySnapshotGuard {
        snapshot: Some(synthetic_report()),
        retain_tag: None,
        dump_path: None,
        dual_snapshot: true,
    };
    guard.drain_to_disk();

    // Snapshot remains Some because dump_path gate runs BEFORE the
    // take(). Without this ordering a gated-off guard would consume
    // the snapshot without writing it anywhere.
    assert!(
        guard.snapshot.is_some(),
        "dump_path=None gate must run BEFORE snapshot.take() — \
         a consumed snapshot here means take() ran first and the \
         snapshot is now lost"
    );

    let entries_after_drain: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("readdir")
        .collect();
    assert!(
        entries_after_drain.is_empty(),
        "no file may be created when dump_path is unset — found {} entries",
        entries_after_drain.len()
    );

    drop(guard);

    let entries_after_drop: Vec<_> = std::fs::read_dir(tmp.path())
        .expect("readdir")
        .collect();
    assert!(
        entries_after_drop.is_empty(),
        "Drop must also no-op when dump_path is unset — found {} entries",
        entries_after_drop.len()
    );
}

/// Write failure (parent path component is a regular file → ENOTDIR
/// at File::create in write_to_tagged_path) is swallowed silently
/// inside drain_to_disk's `Err(_) => {}` arm. The Drop body must
/// NOT propagate any panic — a double-panic during stack unwind
/// would abort the entire process via the std-library
/// double-panic-aborts contract. Pins the silent-swallow and
/// Drop-never-panics invariants.
///
/// Variant A: explicit drain on the bad path. Drop fires after as
/// a no-op (snapshot already taken inside drain). Reaching the
/// assertions after the scope proves no panic propagated.
///
/// Variant B: panic injected inside the scope so Drop fires DURING
/// unwind. If Drop's drain attempted to propagate the write error
/// as a panic, the runtime would abort the process (visible as
/// test-binary signal termination, not as a returned Err). Reaching
/// the `result.is_err()` assertion at all proves Drop was
/// unwind-safe.
///
/// Note on stderr fallback: the structured stderr summary must reach
/// stderr when write_to_tagged_path's atomic publish fails. That
/// emission lives INSIDE write_to_tagged_path (eprintln! at the Err
/// arm), and its assertion lives in write_to_tagged_path's own
/// helper-coverage tests (sibling `write_to_tagged_path_tests`
/// module). The guard tests here pin only the no-panic + no-write-
/// retry semantics; the stderr-fallback contract is covered by the
/// helper's tests, not duplicated here.
#[test]
fn early_snapshot_guard_drop_swallows_write_failure_without_panic() {
    let tmp = TempDir::new().expect("tempdir");
    let blocker = tmp.path().join("blocker_file");
    std::fs::write(&blocker, b"not a dir").expect("write blocker");
    // dump_path's parent is the regular file, so File::create on
    // the tagged sibling fails with ENOTDIR. The
    // `let _ = create_dir_all(parent)` in write_to_tagged_path
    // also fails silently — File::create is the load-bearing fault.
    let dump_path = blocker.join("coord.failure-dump.json");

    // Variant A — explicit drain swallows the Err.
    {
        let mut guard = EarlySnapshotGuard {
            snapshot: Some(synthetic_report()),
            retain_tag: None,
            dump_path: Some(dump_path.clone()),
            dual_snapshot: true,
        };
        guard.drain_to_disk();
        // Snapshot consumed despite the Err — drain_to_disk
        // takes() BEFORE calling write_to_tagged_path.
        assert!(
            guard.snapshot.is_none(),
            "drain consumes snapshot even when the helper returns Err"
        );
    } // Drop fires — no-op since snapshot already taken.

    // Variant B — panic during scope. Drop fires while unwinding.
    // Surviving this without a double-panic abort is the
    // load-bearing assertion.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _guard = EarlySnapshotGuard {
            snapshot: Some(synthetic_report()),
            retain_tag: None,
            dump_path: Some(dump_path.clone()),
            dual_snapshot: true,
        };
        panic!("injected — write failure must not turn this into a double-panic abort");
    }));
    assert!(
        result.is_err(),
        "injected panic must propagate without Drop adding its own"
    );
}

/// retain_tag without snapshot is a silent no-op. Currently no
/// production path sets retain_tag with snapshot=None — the late-
/// trigger Suppressed and pre-late-degraded arms always set
/// retain_tag while snapshot is still held. A future refactor
/// that reorders these to take() the snapshot first then set
/// retain_tag would silently produce a guard whose Drop fires
/// the third `else { return }` arm (snapshot.take() returns None)
/// and never reaches the retain_tag lookup at the
/// `unwrap_or(NEVER_FIRED)` line.
///
/// This canary pins the API invariant explicitly: a guard with
/// retain_tag=Some and snapshot=None lands zero files on disk even
/// when dual_snapshot+dump_path are both wired.
#[test]
fn early_snapshot_guard_retain_tag_without_snapshot_no_op() {
    let tmp = TempDir::new().expect("tempdir");
    let dump_path = dump_base_path(&tmp);
    let mut guard = EarlySnapshotGuard {
        snapshot: None,
        retain_tag: Some(SNAPSHOT_TAG_EARLY_PRE_LATE_DEGRADED),
        dump_path: Some(dump_path.clone()),
        dual_snapshot: true,
    };
    guard.drain_to_disk();

    for tag in ALL_SNAPSHOT_TAGS {
        let path = snapshot_tagged_path(&dump_path, tag);
        assert!(
            !path.exists(),
            "retain_tag=Some + snapshot=None must NOT write to {} \
             after explicit drain (retain_tag is unreachable when \
             the snapshot.take() gate returns None)",
            path.display()
        );
    }

    // Defense-in-depth: explicit drop(guard) + re-scan catches a
    // future refactor that splits Drop's behavior from drain_to_disk
    // (e.g. Drop adopts a retain_tag-bypass logic that writes despite
    // snapshot=None). Today Drop just calls drain_to_disk so this
    // re-scan is redundant — but a future Drop divergence would be
    // caught here.
    drop(guard);
    for tag in ALL_SNAPSHOT_TAGS {
        let path = snapshot_tagged_path(&dump_path, tag);
        assert!(
            !path.exists(),
            "retain_tag=Some + snapshot=None must NOT write to {} \
             after Drop fires (Drop must not diverge from drain_to_disk's \
             gate behavior)",
            path.display()
        );
    }
}

/// take() ordering matrix: `self.snapshot.take()` at the third
/// `let Some(early) = ... else { return };` arm runs ONLY when both
/// gates (dual_snapshot AND dump_path) have passed. Load-bearing for
/// the no-silent-loss contract. If take() ran before either gate, a
/// gated-off guard would consume the snapshot without writing it
/// anywhere, dropping it on the floor.
///
/// Matrix exhausts the 4 reachable configs of (dual_snapshot ×
/// dump_path):
///   (a) dual_snapshot=false, dump_path=Some  — dual_snapshot gate fires
///   (b) dual_snapshot=true,  dump_path=None  — dump_path gate fires
///   (c) dual_snapshot=true,  dump_path=Some  — control, both pass
///   (d) dual_snapshot=false, dump_path=None  — dual_snapshot fires first
#[test]
fn early_snapshot_guard_drain_preserves_snapshot_when_gated_off() {
    struct Row {
        label: &'static str,
        dual_snapshot: bool,
        with_dump_path: bool,
        expect_snapshot_consumed: bool,
        expect_file_written: bool,
    }
    let rows = [
        Row {
            label: "(a) dual_snapshot=false, dump_path=Some",
            dual_snapshot: false,
            with_dump_path: true,
            expect_snapshot_consumed: false,
            expect_file_written: false,
        },
        Row {
            label: "(b) dual_snapshot=true, dump_path=None",
            dual_snapshot: true,
            with_dump_path: false,
            expect_snapshot_consumed: false,
            expect_file_written: false,
        },
        Row {
            label: "(c) control: dual_snapshot=true, dump_path=Some",
            dual_snapshot: true,
            with_dump_path: true,
            expect_snapshot_consumed: true,
            expect_file_written: true,
        },
        Row {
            label: "(d) dual_snapshot=false, dump_path=None",
            dual_snapshot: false,
            with_dump_path: false,
            expect_snapshot_consumed: false,
            expect_file_written: false,
        },
    ];

    for row in rows {
        let tmp = TempDir::new().expect("tempdir");
        let dump_path = dump_base_path(&tmp);
        let mut guard = EarlySnapshotGuard {
            snapshot: Some(synthetic_report()),
            retain_tag: None,
            dump_path: row.with_dump_path.then(|| dump_path.clone()),
            dual_snapshot: row.dual_snapshot,
        };
        guard.drain_to_disk();

        assert_eq!(
            guard.snapshot.is_none(),
            row.expect_snapshot_consumed,
            "[{}] snapshot consumption mismatch: expected_consumed={}",
            row.label,
            row.expect_snapshot_consumed
        );

        let expected =
            snapshot_tagged_path(&dump_path, SNAPSHOT_TAG_EARLY_ONLY_LATE_NEVER_FIRED);
        assert_eq!(
            expected.exists(),
            row.expect_file_written,
            "[{}] file-presence mismatch at {}: expected={}",
            row.label,
            expected.display(),
            row.expect_file_written
        );
    }
}
