use super::*;
use crate::metric_types::{
    Bytes, CategoricalString, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32,
};
use tracing_test::traced_test;

fn thread(pcomm: &str, comm: &str, run_time_ns: u64) -> ThreadState {
    ThreadState {
        tid: 1,
        tgid: 1,
        pcomm: pcomm.into(),
        comm: comm.into(),
        cgroup: "/".into(),
        start_time_clock_ticks: 0,
        policy: CategoricalString("SCHED_OTHER".into()),
        nice: OrdinalI32(0),
        cpu_affinity: CpuSet(vec![0, 1]),
        run_time_ns: MonotonicNs(run_time_ns),
        ..ThreadState::default()
    }
}

/// `ThreadState::default()` produces `'~'` (not `'\0'`) for
/// the `state` char so the absent-value sentinel matches the
/// capture-time `unwrap_or_else(default_state_char)`
/// discipline. The bare `char` Default of `'\0'` (U+0000)
/// lex-compares SMALLER than every real kernel state letter
/// (`R`/`S`/`D`/`T`/`t`/`X`/`Z`/`P`/`I`); a Mode-tie-break
/// that picks the lex-smallest would silently elect `'\0'`
/// whenever a default-built thread sat alongside a real one
/// in a group, dragging the cell from a meaningful state
/// letter down to the absent sentinel. The manual
/// [`Default`] impl on [`ThreadState`] pairs with the
/// `serde(default = "default_state_char")` attribute on the
/// field so both construction paths land on `'~'`.
#[test]
fn default_threadstate_state_is_sentinel_tilde() {
    let t = ThreadState::default();
    assert_eq!(
        t.state, '~',
        "ThreadState::default().state must be '~' (the \
         absent-value sentinel chosen to lex-sort AFTER \
         every real kernel state letter), not '\\0' (the \
         bare char Default); see field doc on \
         ThreadState::state"
    );
}

/// Mode tie-break regression: a default-constructed
/// `ThreadState` must NOT lex-beat a real kernel state
/// letter when both contribute to the same Mode aggregation
/// at equal frequency. The kernel's
/// [`crate::ctprof_compare::aggregate`] closure
/// `a.1.cmp(&b.1).then(b.0.cmp(&a.0))` selects
/// LEX-SMALLEST on count-ties, so the sentinel must be
/// LARGER than every real letter to keep the real letter
/// winning. `'~'` (U+007E = 126) is larger than every
/// kernel state letter (`R`=82, `S`=83, `D`=68, `T`=84,
/// `t`=116, `X`=88, `Z`=90, `P`=80, `I`=73), so the
/// tiebreak picks `R`. The original `'?'` (U+003F = 63)
/// sentinel was SMALLER than every real letter, which
/// would have made this test fail.
#[test]
fn mode_tiebreak_against_default_state_picks_real_letter() {
    use crate::ctprof_compare::{AggRule, Aggregated, aggregate};
    let default_thread = ThreadState::default();
    let real_thread = ThreadState {
        state: 'R',
        ..ThreadState::default()
    };
    let agg = aggregate(
        AggRule::ModeChar(|t| t.state),
        &[&default_thread, &real_thread],
    );
    match &agg {
        Aggregated::Mode { .. } => assert_eq!(
            agg.mode_value(),
            "R",
            "Mode tiebreak between '~' (default sentinel) \
             and 'R' (real kernel state) must elect 'R'; \
             got {:?}",
            agg.mode_value(),
        ),
        other => panic!("expected Mode, got {other:?}"),
    }
}

/// Wire-format identity: hand-written JSON with raw
/// primitive values at every newtype-wrapped field position
/// must deserialize cleanly into a post-phase-2
/// `ThreadState` with the wrapper fields holding the
/// expected values. Covers one representative field per
/// newtype family — MonotonicCount, MonotonicNs, ClockTicks,
/// Bytes, PeakNs, GaugeNs, GaugeCount, OrdinalI32,
/// OrdinalU32, CategoricalString, CpuSet — so a regression
/// that breaks `serde(transparent)` on any wrapper would
/// surface here without needing a real .ctprof.zst file from
/// pre-phase-2 capture. Pre-phase-2 snapshot files (raw
/// `u64`/`i32`/`String`/`Vec<u32>` at every position)
/// continue to deserialize identically.
#[test]
fn wire_format_identity_raw_primitives_deserialize_into_wrapped_thread_state() {
    let json = r#"{
        "tid": 1234,
        "tgid": 1234,
        "pcomm": "demo",
        "comm": "demo-w",
        "cgroup": "/app",
        "start_time_clock_ticks": 555000,
        "policy": "SCHED_OTHER",
        "nice": -5,
        "cpu_affinity": [0, 1, 2, 3],
        "processor": 7,
        "state": "R",
        "ext_enabled": false,
        "run_time_ns": 1000000,
        "wait_time_ns": 0,
        "timeslices": 50,
        "voluntary_csw": 100,
        "nonvoluntary_csw": 25,
        "nr_wakeups": 200,
        "nr_wakeups_local": 80,
        "nr_wakeups_remote": 30,
        "nr_wakeups_sync": 10,
        "nr_wakeups_migrate": 5,
        "nr_wakeups_affine": 60,
        "nr_wakeups_affine_attempts": 100,
        "nr_migrations": 8,
        "nr_forced_migrations": 1,
        "nr_failed_migrations_affine": 0,
        "nr_failed_migrations_running": 0,
        "nr_failed_migrations_hot": 0,
        "wait_sum": 5000000,
        "wait_count": 15,
        "wait_max": 250000,
        "voluntary_sleep_ns": 3200000,
        "sleep_max": 180000,
        "block_sum": 1100000,
        "block_max": 60000,
        "iowait_sum": 77000,
        "iowait_count": 18,
        "exec_max": 90000,
        "slice_max": 400000,
        "allocated_bytes": 16777216,
        "deallocated_bytes": 8388608,
        "minflt": 7777,
        "majflt": 8888,
        "utime_clock_ticks": 10,
        "stime_clock_ticks": 11,
        "priority": 25,
        "rt_priority": 99,
        "core_forceidle_sum": 0,
        "fair_slice_ns": 250000,
        "nr_threads": 4,
        "smaps_rollup_kb": {},
        "rchar": 100,
        "wchar": 200,
        "syscr": 10,
        "syscw": 20,
        "read_bytes": 4096,
        "write_bytes": 8192,
        "cancelled_write_bytes": 1024,
        "cpu_delay_count": 0,
        "cpu_delay_total_ns": 0,
        "cpu_delay_max_ns": 0,
        "cpu_delay_min_ns": 0,
        "blkio_delay_count": 0,
        "blkio_delay_total_ns": 0,
        "blkio_delay_max_ns": 0,
        "blkio_delay_min_ns": 0,
        "swapin_delay_count": 0,
        "swapin_delay_total_ns": 0,
        "swapin_delay_max_ns": 0,
        "swapin_delay_min_ns": 0,
        "freepages_delay_count": 0,
        "freepages_delay_total_ns": 0,
        "freepages_delay_max_ns": 0,
        "freepages_delay_min_ns": 0,
        "thrashing_delay_count": 0,
        "thrashing_delay_total_ns": 0,
        "thrashing_delay_max_ns": 0,
        "thrashing_delay_min_ns": 0,
        "compact_delay_count": 0,
        "compact_delay_total_ns": 0,
        "compact_delay_max_ns": 0,
        "compact_delay_min_ns": 0,
        "wpcopy_delay_count": 0,
        "wpcopy_delay_total_ns": 0,
        "wpcopy_delay_max_ns": 0,
        "wpcopy_delay_min_ns": 0,
        "irq_delay_count": 0,
        "irq_delay_total_ns": 0,
        "irq_delay_max_ns": 0,
        "irq_delay_min_ns": 0,
        "hiwater_rss_bytes": 0,
        "hiwater_vm_bytes": 0
    }"#;
    let t: ThreadState = serde_json::from_str(json).expect("deserialize");
    // One representative field per newtype family proves
    // serde(transparent) works post-migration.
    assert_eq!(t.run_time_ns, crate::metric_types::MonotonicNs(1_000_000));
    assert_eq!(t.timeslices, crate::metric_types::MonotonicCount(50));
    assert_eq!(t.utime_clock_ticks, crate::metric_types::ClockTicks(10));
    assert_eq!(t.allocated_bytes, crate::metric_types::Bytes(16_777_216));
    assert_eq!(
        t.cancelled_write_bytes,
        crate::metric_types::Bytes(1024),
        "cancelled_write_bytes round-trips through the JSON \
         wire format alongside the other Bytes-typed fields",
    );
    assert_eq!(t.wait_max, crate::metric_types::PeakNs(250_000));
    assert_eq!(t.fair_slice_ns, crate::metric_types::GaugeNs(250_000));
    assert_eq!(t.nr_threads, crate::metric_types::GaugeCount(4));
    assert_eq!(t.nice, crate::metric_types::OrdinalI32(-5));
    assert_eq!(t.rt_priority, crate::metric_types::OrdinalU32(99));
    assert_eq!(
        t.policy,
        crate::metric_types::CategoricalString::from("SCHED_OTHER")
    );
    assert_eq!(
        t.cpu_affinity,
        crate::metric_types::CpuSet(vec![0, 1, 2, 3])
    );
}

/// Type-pin: nr_threads MUST be `GaugeCount`. A future
/// refactor that flips it to a different newtype (e.g.
/// `MonotonicCount`, which would silently re-enable Summable
/// and let `--group-by comm`/`--group-by cgroup` over-count
/// the parent process N-fold) would break this single
/// `let _: GaugeCount = ...;` assignment. The test compiles
/// only when the type is exactly `GaugeCount`.
#[test]
fn nr_threads_field_pinned_to_gauge_count() {
    let t = ThreadState::default();
    let _: crate::metric_types::GaugeCount = t.nr_threads;
}

/// Type-pin: cancelled_write_bytes MUST be `Bytes`. A future
/// refactor that flipped it to a non-byte type (e.g. plain
/// `MonotonicCount`, dropping the IEC-binary auto-scale
/// ladder and the registry's `unit: "B"` rendering) would
/// break this single `let _: Bytes = ...;` assignment. The
/// test compiles only when the type is exactly `Bytes`.
#[test]
fn cancelled_write_bytes_field_pinned_to_bytes() {
    let t = ThreadState::default();
    let _: crate::metric_types::Bytes = t.cancelled_write_bytes;
}

#[test]
fn snapshot_roundtrip_through_zstd_json() {
    let snap = CtprofSnapshot {
        captured_at_unix_ns: 42,
        host: None,
        threads: vec![
            thread("proc_a", "worker_0", 1_000_000),
            thread("proc_a", "worker_1", 2_000_000),
        ],
        cgroup_stats: BTreeMap::from([("/".into(), {
            let mut cs = CgroupStats::default();
            cs.cpu.usage_usec = 500;
            cs.memory.current = 1 << 20;
            cs
        })]),
        probe_summary: None,
        parse_summary: None,
        taskstats_summary: None,
        psi: Psi::default(),
        sched_ext: None,
    };
    let tmp = tempfile::NamedTempFile::new().unwrap();
    snap.write(tmp.path()).unwrap();
    let back = CtprofSnapshot::load(tmp.path()).unwrap();
    assert_eq!(back.captured_at_unix_ns, 42);
    assert_eq!(back.threads.len(), 2);
    assert_eq!(
        back.threads[1].run_time_ns,
        crate::metric_types::MonotonicNs(2_000_000),
    );
    assert_eq!(back.cgroup_stats["/"].cpu.usage_usec, 500);
}

#[test]
fn load_rejects_non_zstd_payload() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"{\"not\": \"zstd\"}").unwrap();
    let err = CtprofSnapshot::load(tmp.path()).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("zstd"),
        "expected zstd error in context chain, got: {msg}",
    );
}

#[test]
fn load_rejects_zstd_of_garbage_json() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let compressed = zstd::encode_all(&b"not json"[..], 3).unwrap();
    std::fs::write(tmp.path(), compressed).unwrap();
    let err = CtprofSnapshot::load(tmp.path()).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("parse ctprof"),
        "expected parse error in context chain, got: {msg}",
    );
}

/// Decompression-bomb guard: a zstd payload that decompresses
/// past the configured cap surfaces an error tagged with
/// "decompression-bomb guard" — the loader must not allocate
/// past the ceiling. Test uses a small synthetic payload (8
/// KiB of zeros, which compresses to a tiny blob but
/// decompresses to 8192 bytes) against a 1024-byte cap so
/// the test runs in microseconds rather than allocating a
/// production-sized buffer.
#[test]
fn decompress_capped_rejects_decompression_bomb() {
    let payload = vec![0u8; 8192];
    let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
    let cap: u64 = 1024;
    let err = super::decompress_capped(&compressed, cap).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("decompression-bomb guard"),
        "expected decompression-bomb guard error, got: {msg}",
    );
}

/// Boundary case: a payload whose decompressed length is
/// exactly `cap` bytes is accepted (the cap is inclusive).
/// Pins the `>` (not `>=`) discriminator at the cap boundary
/// so a future refactor that flips the comparison surfaces
/// here rather than turning a legal snapshot into a
/// false-positive bomb rejection.
#[test]
fn decompress_capped_accepts_payload_at_cap_boundary() {
    let payload = b"hello world".to_vec();
    let compressed = zstd::encode_all(payload.as_slice(), 3).unwrap();
    let out = super::decompress_capped(&compressed, payload.len() as u64).unwrap();
    assert_eq!(
        out, payload,
        "payload exactly at the cap must round-trip — \
         cap is inclusive (`>` not `>=`)",
    );
}

#[test]
fn parse_stat_robust_against_paren_in_comm() {
    // Field 2 (comm) may contain ')'. The parser must latch on
    // the LAST ')'. Construct a line where comm is
    // `(weird)name)` and fields 3..=22 are 0..=19.
    let mut line = String::from("1234 (weird)name) ");
    for i in 0..20 {
        line.push_str(&format!("{i} "));
    }
    let f = parse_stat(&line);
    assert_eq!(f.start_time_clock_ticks, Some(19));
}

#[test]
fn parse_stat_extracts_all_known_fields() {
    // Fields 3..=41 — tail indices 0..=38. Token at tail[i] = i.
    // minflt at tail[7] = 7; majflt at tail[9] = 9;
    // utime at tail[11] = 11; stime at tail[12] = 12;
    // nice at tail[16] = 16; starttime at tail[19] = 19;
    // processor at tail[36] = 36; policy at tail[38] = 38.
    let mut line = String::from("1 (n) ");
    for i in 0..=38 {
        line.push_str(&format!("{i} "));
    }
    let f = parse_stat(&line);
    assert_eq!(f.minflt, Some(7));
    assert_eq!(f.majflt, Some(9));
    assert_eq!(f.utime_clock_ticks, Some(11));
    assert_eq!(f.stime_clock_ticks, Some(12));
    assert_eq!(f.nice, Some(16));
    assert_eq!(f.start_time_clock_ticks, Some(19));
    assert_eq!(f.processor, Some(36));
    assert_eq!(f.policy, Some(38));
}

#[test]
fn parse_stat_short_line_drops_missing_fields() {
    // Only fields 3..=10 present; minflt at 7 landed, majflt at
    // 9 missing, later fields also missing.
    let line = "1 (n) 0 1 2 3 4 5 6 7";
    let f = parse_stat(line);
    assert_eq!(f.minflt, Some(7));
    assert_eq!(f.majflt, None);
    assert_eq!(f.utime_clock_ticks, None);
    assert_eq!(f.stime_clock_ticks, None);
    assert_eq!(f.nice, None);
    assert_eq!(f.start_time_clock_ticks, None);
    assert_eq!(f.processor, None);
    assert_eq!(f.policy, None);
}

/// `processor` parses signed values via `get_i32`. The mainline
/// kernel never emits a negative value (`task_cpu` is
/// `unsigned int` per `include/linux/sched.h`, zero-extended
/// through `seq_put_decimal_ll`), but the parser accepts
/// negatives anyway — pinning that the type choice (`i32`)
/// does not silently drop a hypothetical out-of-band negative
/// to `None`. Defends against a regression that swapped
/// `get_i32` for `get_u64` and made the field reject any
/// negative token instead of carrying it through.
#[test]
fn parse_stat_processor_accepts_negative() {
    // 36 zero-pad tokens, tail[36] = -1, then more padding to
    // reach tail[38] for the policy field.
    let mut line = String::from("1 (n) ");
    for i in 0..36 {
        line.push_str(&format!("{i} "));
    }
    line.push_str("-1 ");
    line.push_str("0 ");
    line.push_str("0 ");
    let f = parse_stat(&line);
    assert_eq!(
        f.processor,
        Some(-1),
        "negative tokens must flow through as Some(-1) — pins \
         the get_i32 vs get_u64 type choice, not kernel emit \
         behavior (which never emits negative)",
    );
}

#[test]
fn parse_schedstat_three_fields() {
    let (a, b, c) = parse_schedstat("12345 67890 42\n");
    assert_eq!(a, Some(12345));
    assert_eq!(b, Some(67890));
    assert_eq!(c, Some(42));
}

#[test]
fn parse_schedstat_missing_fields_drop_individually() {
    let (a, b, c) = parse_schedstat("12345\n");
    assert_eq!(a, Some(12345));
    assert_eq!(b, None);
    assert_eq!(c, None);
}

#[test]
fn parse_io_extracts_all_seven_fields() {
    let raw = "rchar: 1\n\
               wchar: 2\n\
               syscr: 3\n\
               syscw: 4\n\
               read_bytes: 5\n\
               write_bytes: 6\n\
               cancelled_write_bytes: 7\n";
    let f = parse_io(raw);
    assert_eq!(f.rchar, Some(1));
    assert_eq!(f.wchar, Some(2));
    assert_eq!(f.syscr, Some(3));
    assert_eq!(f.syscw, Some(4));
    assert_eq!(f.read_bytes, Some(5));
    assert_eq!(f.write_bytes, Some(6));
    assert_eq!(f.cancelled_write_bytes, Some(7));
}

#[test]
fn parse_status_extracts_csw_and_affinity() {
    let raw = "Name:\tbash\n\
               State:\tS (sleeping)\n\
               Cpus_allowed_list:\t0-3,5\n\
               voluntary_ctxt_switches:\t100\n\
               nonvoluntary_ctxt_switches:\t5\n";
    let f = parse_status(raw);
    assert_eq!(f.voluntary_csw, Some(100));
    assert_eq!(f.nonvoluntary_csw, Some(5));
    assert_eq!(
        f.state,
        Some('S'),
        "first non-whitespace char of `State:` value is the \
         single-letter code (R/S/D/T/t/X/Z/P/I)",
    );
    assert_eq!(f.cpus_allowed.as_deref(), Some(&[0u32, 1, 2, 3, 5][..]));
}

/// Every kernel-emitted state code parses correctly. Pins each
/// entry of `task_state_array` so a regression that lowercased
/// the match or stripped paren-content would surface. Codes
/// are from `fs/proc/array.c::task_state_array` — all NINE
/// entries (R/S/D/T/t/X/Z/P/I), including the off-by-default
/// `P (parked)` which only appears on kernels that schedule
/// parked tasks.
#[test]
fn parse_status_accepts_every_kernel_state_code() {
    for code in ['R', 'S', 'D', 'T', 't', 'X', 'Z', 'P', 'I'] {
        let raw = format!("State:\t{code} (label)\n");
        assert_eq!(parse_status(&raw).state, Some(code));
    }
}

/// Absent `State:` line lands as `None`; capture site collapses
/// to `'~'`. Pins the absent-default boundary.
#[test]
fn parse_status_absent_state_line_yields_none() {
    let raw = "voluntary_ctxt_switches:\t1\n";
    let f = parse_status(raw);
    assert_eq!(f.state, None);
}

/// PSI parser pins the kernel emission format
/// `kernel/sched/psi.c:1284`. Two-line shape (some + full)
/// is the cpu/memory/io case; cpu's avg/total decomposition
/// hits both halves so a one-side regression surfaces here.
#[test]
fn parse_psi_extracts_some_and_full_halves() {
    let raw = "some avg10=18.59 avg60=24.31 avg300=20.49 total=78097519837\n\
               full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n";
    let r = parse_psi(raw);
    // some: integer + 2-digit fraction → centi-percent.
    assert_eq!(r.some.avg10, 1859);
    assert_eq!(r.some.avg60, 2431);
    assert_eq!(r.some.avg300, 2049);
    assert_eq!(r.some.total_usec, 78_097_519_837);
    assert_eq!(r.full.avg10, 0);
    assert_eq!(r.full.avg60, 0);
    assert_eq!(r.full.avg300, 0);
    assert_eq!(r.full.total_usec, 0);
}

/// IRQ pressure is full-only per
/// `kernel/sched/psi.c:1268` (`only_full = res == PSI_IRQ`),
/// so the some-half stays at the absent-line default of zero.
#[test]
fn parse_psi_irq_full_only_leaves_some_at_zero() {
    let raw = "full avg10=1.09 avg60=1.08 avg300=1.46 total=80506377366\n";
    let r = parse_psi(raw);
    assert_eq!(r.full.avg10, 109);
    assert_eq!(r.full.avg60, 108);
    assert_eq!(r.full.avg300, 146);
    assert_eq!(r.full.total_usec, 80_506_377_366);
    // `some` half left at default zero — the kernel never
    // emitted a `some` line for irq.
    assert_eq!(r.some.avg10, 0);
    assert_eq!(r.some.avg60, 0);
    assert_eq!(r.some.avg300, 0);
    assert_eq!(r.some.total_usec, 0);
}

/// Empty / absent file collapses to all-zero bundle. Pins
/// the absent-counter contract used elsewhere in this module.
#[test]
fn parse_psi_empty_input_yields_default() {
    let r = parse_psi("");
    assert_eq!(r.some.avg10, 0);
    assert_eq!(r.full.total_usec, 0);
}

/// Malformed numeric values default to zero rather than
/// panicking. Mirrors the broader parser's
/// `value.parse::<u64>().ok()` discipline — best-effort
/// capture, never a hard error.
#[test]
fn parse_psi_malformed_value_defaults_to_zero() {
    let raw = "some avg10=NaN avg60=0.50 avg300=- total=abc\n";
    let r = parse_psi(raw);
    assert_eq!(r.some.avg10, 0, "NaN parses to zero");
    assert_eq!(r.some.avg60, 50, "well-formed neighbor still parses");
    assert_eq!(r.some.avg300, 0, "lone dash parses to zero");
    assert_eq!(r.some.total_usec, 0, "non-numeric total parses to zero");
}

/// Centi-percent conversion exhausts the fixed-point range:
/// `100.00%` maps to 10_000. Pins the upper boundary
/// against the u16 storage choice.
#[test]
fn parse_psi_full_saturation_maps_to_10000() {
    let raw = "some avg10=100.00 avg60=100.00 avg300=100.00 total=42\n";
    let r = parse_psi(raw);
    assert_eq!(r.some.avg10, 10_000);
    assert_eq!(r.some.avg60, 10_000);
    assert_eq!(r.some.avg300, 10_000);
    assert_eq!(r.some.total_usec, 42);
}

/// Unknown tokens are silently dropped (forward-compat with
/// a future kernel that adds a 4th avg or new field).
#[test]
fn parse_psi_unknown_keys_ignored() {
    let raw = "some avg10=1.00 avg600=99.99 future_field=42 total=10\n";
    let r = parse_psi(raw);
    assert_eq!(r.some.avg10, 100);
    assert_eq!(r.some.total_usec, 10);
}

/// `parse_centi_percent` pads/truncates the fractional part
/// to exactly 2 digits before combining. The kernel always
/// emits `%02lu` per `kernel/sched/psi.c:1284`, but a robust
/// parser must not silently rescale `"1.5"` (one digit) as
/// `1*100+5 = 105` (1.05%) — that would corrupt the value.
/// Mirrors `parsed_ns_from_dotted`'s zero-pad-to-six rule.
#[test]
fn parse_centi_percent_zero_pads_short_fraction() {
    // No fraction → 0.
    assert_eq!(parse_centi_percent("0"), 0);
    assert_eq!(parse_centi_percent("42"), 4200);
    // One-digit fraction → pad with trailing zero.
    assert_eq!(parse_centi_percent("1.5"), 150, "1.5 must read as 1.50%");
    assert_eq!(parse_centi_percent("0.7"), 70, "0.7 must read as 0.70%");
    // Two-digit fraction → kernel-canonical case.
    assert_eq!(parse_centi_percent("18.59"), 1859);
    // Three+ digit fraction → truncate to 2.
    assert_eq!(
        parse_centi_percent("1.501"),
        150,
        "1.501 truncates to 1.50%"
    );
    // Empty fraction (trailing dot) → 0.
    assert_eq!(parse_centi_percent("3."), 300);
    // EWMA-rounding ceiling per loadavg.h:35.
    assert_eq!(parse_centi_percent("100.99"), 10099);
}

/// Stage a synthetic `<proc_root>/pressure/{cpu,memory,io,irq}`
/// tree and verify [`read_host_psi_at`] returns a fully
/// populated [`Psi`] bundle. Pins the file naming and the
/// per-resource bundling — a regression that swapped two
/// resource sources (e.g. read `pressure/io` into `psi.cpu`)
/// surfaces here as wrong-resource-wrong-value.
#[test]
fn read_host_psi_at_populates_all_four_resources() {
    let tmp = tempfile::TempDir::new().unwrap();
    let pressure = tmp.path().join("pressure");
    std::fs::create_dir_all(&pressure).unwrap();
    std::fs::write(
        pressure.join("cpu"),
        "some avg10=1.00 avg60=2.00 avg300=3.00 total=100\n\
         full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
    )
    .unwrap();
    std::fs::write(
        pressure.join("memory"),
        "some avg10=4.50 avg60=5.50 avg300=6.50 total=200\n\
         full avg10=7.50 avg60=8.50 avg300=9.50 total=150\n",
    )
    .unwrap();
    std::fs::write(
        pressure.join("io"),
        "some avg10=10.10 avg60=20.20 avg300=30.30 total=300\n\
         full avg10=40.40 avg60=50.50 avg300=60.60 total=250\n",
    )
    .unwrap();
    std::fs::write(
        pressure.join("irq"),
        "full avg10=0.50 avg60=0.60 avg300=0.70 total=80\n",
    )
    .unwrap();

    let psi = read_host_psi_at(tmp.path());

    // cpu: both halves populated, full all-zero.
    assert_eq!(psi.cpu.some.avg10, 100);
    assert_eq!(psi.cpu.some.avg60, 200);
    assert_eq!(psi.cpu.some.avg300, 300);
    assert_eq!(psi.cpu.some.total_usec, 100);
    assert_eq!(psi.cpu.full.avg10, 0);
    assert_eq!(psi.cpu.full.total_usec, 0);

    // memory: both halves carry distinct nonzero values —
    // catches a regression that returned the same half
    // twice.
    assert_eq!(psi.memory.some.avg10, 450);
    assert_eq!(psi.memory.full.avg10, 750);
    assert_eq!(psi.memory.some.total_usec, 200);
    assert_eq!(psi.memory.full.total_usec, 150);

    // io: largest distinct values; ensures resource-source
    // routing isn't swapped against memory or cpu.
    assert_eq!(psi.io.some.avg10, 1010);
    assert_eq!(psi.io.full.avg300, 6060);
    assert_eq!(psi.io.some.total_usec, 300);

    // irq: full-only; some-half stays at the absent-line
    // default of zero.
    assert_eq!(psi.irq.full.avg10, 50);
    assert_eq!(psi.irq.full.avg60, 60);
    assert_eq!(psi.irq.full.avg300, 70);
    assert_eq!(psi.irq.full.total_usec, 80);
    assert_eq!(psi.irq.some.avg10, 0);
    assert_eq!(psi.irq.some.total_usec, 0);
}

/// Absent `pressure/` directory or absent per-resource files
/// collapse to the all-zero default. Pins the absent-counter
/// contract so a host with `CONFIG_PSI=n` (or older kernels
/// missing `irq.pressure`) doesn't error out — capture is
/// best-effort.
#[test]
fn read_host_psi_at_missing_files_yield_default() {
    // tempdir with no `pressure/` subdir at all.
    let tmp = tempfile::TempDir::new().unwrap();
    let psi = read_host_psi_at(tmp.path());
    assert_eq!(psi.cpu.some.avg10, 0);
    assert_eq!(psi.memory.full.total_usec, 0);
    assert_eq!(psi.io.some.avg300, 0);
    assert_eq!(psi.irq.full.avg60, 0);

    // Partial — only `cpu` exists; the other three should
    // still default cleanly.
    let pressure = tmp.path().join("pressure");
    std::fs::create_dir_all(&pressure).unwrap();
    std::fs::write(
        pressure.join("cpu"),
        "some avg10=12.34 avg60=0 avg300=0 total=0\n\
         full avg10=0 avg60=0 avg300=0 total=0\n",
    )
    .unwrap();
    let psi = read_host_psi_at(tmp.path());
    assert_eq!(psi.cpu.some.avg10, 1234);
    assert_eq!(psi.memory.some.avg10, 0);
    assert_eq!(psi.io.full.total_usec, 0);
    assert_eq!(psi.irq.full.avg10, 0);
}

/// Stage a synthetic cgroup tree and verify
/// [`read_cgroup_psi_at`] reads `<cgroup>/<resource>.pressure`
/// (cgroup v2 file naming, distinct from the host-level
/// `pressure/<resource>` directory layout). Pins the
/// path-strip-leading-slash behavior shared with
/// [`read_cgroup_stats_at`].
#[test]
fn read_cgroup_psi_at_uses_resource_dot_pressure_naming() {
    let cgroup_root = tempfile::TempDir::new().unwrap();
    let cg_dir = cgroup_root.path().join("app");
    std::fs::create_dir_all(&cg_dir).unwrap();
    std::fs::write(
        cg_dir.join("cpu.pressure"),
        "some avg10=11.11 avg60=0 avg300=0 total=42\n\
         full avg10=0 avg60=0 avg300=0 total=0\n",
    )
    .unwrap();
    std::fs::write(
        cg_dir.join("memory.pressure"),
        "some avg10=0 avg60=0 avg300=0 total=0\n\
         full avg10=22.22 avg60=0 avg300=0 total=999\n",
    )
    .unwrap();
    // io.pressure absent → default zero. irq.pressure
    // present but full-only.
    std::fs::write(
        cg_dir.join("irq.pressure"),
        "full avg10=33.33 avg60=0 avg300=0 total=7\n",
    )
    .unwrap();

    let psi = read_cgroup_psi_at(cgroup_root.path(), "/app");

    assert_eq!(psi.cpu.some.avg10, 1111);
    assert_eq!(psi.cpu.some.total_usec, 42);
    assert_eq!(psi.memory.full.avg10, 2222);
    assert_eq!(psi.memory.full.total_usec, 999);
    assert_eq!(psi.io.some.avg10, 0, "absent io.pressure → default zero");
    assert_eq!(psi.io.full.total_usec, 0);
    assert_eq!(psi.irq.full.avg10, 3333);
    assert_eq!(psi.irq.some.avg10, 0, "irq is full-only");
}

/// `parse_kv_counters` reads cgroup v2 key-value files
/// (memory.stat, memory.events). Pins:
/// - well-formed multi-line input populates every key
/// - malformed lines silently elide the offending key (rest
///   of the file still parses)
/// - empty input yields an empty map
/// - unknown key prefixes map verbatim (forward-compat with
///   future kernel additions to memory.stat).
#[test]
fn parse_kv_counters_handles_well_formed_and_malformed_lines() {
    let raw = "anon 12812288\n\
               file 12623872\n\
               pgfault 18\n\
               pgmajfault 4\n\
               workingset_refault_anon 0\n\
               workingset_refault_file 27198\n";
    let m = parse_kv_counters(raw);
    assert_eq!(m.get("anon"), Some(&12_812_288));
    assert_eq!(m.get("file"), Some(&12_623_872));
    assert_eq!(m.get("pgfault"), Some(&18));
    assert_eq!(m.get("pgmajfault"), Some(&4));
    assert_eq!(m.get("workingset_refault_anon"), Some(&0));
    assert_eq!(m.get("workingset_refault_file"), Some(&27_198));
    assert_eq!(m.len(), 6);

    // Empty input → empty map.
    assert!(parse_kv_counters("").is_empty());

    // Malformed: missing value, non-u64 value, blank line —
    // each silently dropped; well-formed neighbors persist.
    let raw = "good 42\n\
               bad_no_value\n\
               bad_negative -5\n\
               bad_text foo\n\
               \n\
               recover 7\n";
    let m = parse_kv_counters(raw);
    assert_eq!(m.get("good"), Some(&42));
    assert_eq!(m.get("recover"), Some(&7));
    assert_eq!(m.len(), 2, "malformed lines must not pollute the map");
}

/// `parse_smaps_rollup` reads cgroup-style `<key>: <u64> kB`
/// lines and returns a `BTreeMap<String, u64>` of kB
/// values. Pins:
/// - well-formed multi-line input populates every key
/// - the kernel's `<vma_range> [rollup]` header (no `:`)
///   is silently skipped
/// - " kB" suffix is dropped via first-whitespace-token
///   extraction (parser doesn't hard-code the unit; a
///   future kernel that drops the suffix still parses)
/// - empty input yields an empty map
/// - lines whose value field doesn't parse as u64 are
///   silently dropped (best-effort, matches the
///   absent-counter contract).
#[test]
fn parse_smaps_rollup_extracts_kb_values_and_skips_header() {
    let raw = "55796dced000-7ffe1f875000 ---p 00000000 00:00 0                          [rollup]\n\
               Rss:                2080 kB\n\
               Pss:                 209 kB\n\
               Pss_Dirty:           136 kB\n\
               Pss_Anon:            136 kB\n\
               Anonymous:           136 kB\n\
               Swap:                  0 kB\n\
               SwapPss:               0 kB\n\
               Locked:                0 kB\n";
    let m = parse_smaps_rollup(raw);
    assert_eq!(m.get("Rss"), Some(&2080), "Rss kB stripped to integer");
    assert_eq!(m.get("Pss"), Some(&209));
    assert_eq!(m.get("Pss_Dirty"), Some(&136));
    assert_eq!(m.get("Pss_Anon"), Some(&136));
    assert_eq!(m.get("Anonymous"), Some(&136));
    assert_eq!(m.get("Swap"), Some(&0));
    assert_eq!(m.get("SwapPss"), Some(&0));
    assert_eq!(m.get("Locked"), Some(&0));
    assert_eq!(
        m.len(),
        8,
        "[rollup] header line is silently elided (no `:` separator)",
    );
}

/// Empty file → empty map. Pins the absent-counter contract
/// for the "kernel pre-4.14 lacks smaps_rollup" path.
#[test]
fn parse_smaps_rollup_empty_input_yields_empty_map() {
    assert!(parse_smaps_rollup("").is_empty());
}

/// Malformed value fields (non-u64) are silently dropped;
/// well-formed neighbors still parse. Pins the parser's
/// best-effort discipline so a future kernel that emits a
/// new key with an unexpected format doesn't break the
/// whole capture.
#[test]
fn parse_smaps_rollup_malformed_value_silently_dropped() {
    let raw = "Rss:                100 kB\n\
               BogusKey:        not_a_number kB\n\
               Pss:                 50 kB\n";
    let m = parse_smaps_rollup(raw);
    assert_eq!(m.get("Rss"), Some(&100));
    assert_eq!(m.get("Pss"), Some(&50), "well-formed neighbor still parses");
    assert!(
        !m.contains_key("BogusKey"),
        "non-u64 value silently dropped"
    );
    assert_eq!(m.len(), 2);
}

/// The kernel's smaps_rollup header line carries a `:` in
/// the device-major:minor pair (`<addr_start>-<addr_end>
/// ---p <off> XX:XX <inode> [rollup]`). A naive
/// `split_once(':')` would mis-extract the long
/// whitespace-laden prefix as a "key" and parse the minor
/// device integer as the "value", producing a junk
/// 0-valued entry on every captured process. Pin the
/// header guard so a regression that drops the
/// whitespace-or-`-` rejection surfaces here.
#[test]
fn parse_smaps_rollup_skips_real_kernel_header_with_device_colon() {
    let raw = "55796dced000-7ffe1f875000 ---p 00000000 00:00 0                          [rollup]\n\
         Rss:                2080 kB\n\
         Pss:                 209 kB\n";
    let m = parse_smaps_rollup(raw);
    // Real keys parsed.
    assert_eq!(m.get("Rss"), Some(&2080));
    assert_eq!(m.get("Pss"), Some(&209));
    // No junk key from the header line — the pre-`:`
    // segment of the header carries whitespace AND `-`,
    // both rejected by the parser's header guard.
    assert_eq!(
        m.len(),
        2,
        "header line with `00:00` device pair must not produce a junk key; got {m:?}",
    );
}

/// Stage a synthetic `<sys_root>/kernel/sched_ext/` tree
/// with all 5 global attrs and verify
/// [`read_sched_ext_sysfs_at`] returns a fully populated
/// [`SchedExtSysfs`]. Pins each file's parse routing.
#[test]
fn read_sched_ext_sysfs_at_populates_all_five_attrs() {
    let sys_root = tempfile::TempDir::new().unwrap();
    let scx_dir = sys_root.path().join("kernel").join("sched_ext");
    std::fs::create_dir_all(&scx_dir).unwrap();
    std::fs::write(scx_dir.join("state"), "enabled\n").unwrap();
    std::fs::write(scx_dir.join("switch_all"), "1\n").unwrap();
    std::fs::write(scx_dir.join("nr_rejected"), "42\n").unwrap();
    std::fs::write(scx_dir.join("hotplug_seq"), "315\n").unwrap();
    std::fs::write(scx_dir.join("enable_seq"), "7\n").unwrap();
    let scx = read_sched_ext_sysfs_at(sys_root.path())
        .expect("populated sched_ext directory must yield Some");
    assert_eq!(scx.state, "enabled");
    assert_eq!(scx.switch_all, 1);
    assert_eq!(scx.nr_rejected, 42);
    assert_eq!(scx.hotplug_seq, 315);
    assert_eq!(scx.enable_seq, 7);
}

/// Absent `<sys_root>/kernel/sched_ext/` directory yields
/// `None`. Pins the CONFIG_SCHED_CLASS_EXT=n / no-sysfs
/// path so a kernel without the feature collapses cleanly
/// into the snapshot's `sched_ext: None`.
#[test]
fn read_sched_ext_sysfs_at_absent_directory_yields_none() {
    let sys_root = tempfile::TempDir::new().unwrap();
    // Empty tempdir — no kernel/sched_ext/ subtree.
    assert!(read_sched_ext_sysfs_at(sys_root.path()).is_none());
}

/// Per-file misses default to 0 / empty string. Pins the
/// absent-counter contract for a half-populated sched_ext
/// directory (older kernel that exposed only a subset of
/// the 5 attrs).
#[test]
fn read_sched_ext_sysfs_at_partial_files_default_zero() {
    let sys_root = tempfile::TempDir::new().unwrap();
    let scx_dir = sys_root.path().join("kernel").join("sched_ext");
    std::fs::create_dir_all(&scx_dir).unwrap();
    // Only state + nr_rejected populated; the other 3 files
    // absent.
    std::fs::write(scx_dir.join("state"), "disabled\n").unwrap();
    std::fs::write(scx_dir.join("nr_rejected"), "100\n").unwrap();
    let scx =
        read_sched_ext_sysfs_at(sys_root.path()).expect("directory exists → returns Some");
    assert_eq!(scx.state, "disabled");
    assert_eq!(scx.nr_rejected, 100);
    assert_eq!(scx.switch_all, 0, "absent file → default 0");
    assert_eq!(scx.hotplug_seq, 0);
    assert_eq!(scx.enable_seq, 0);
}

/// Stage a synthetic procfs tree with a leader-thread
/// (tid==tgid) carrying smaps_rollup, plus a follower
/// thread (tid != tgid). Verifies:
///
/// - leader thread's read populates the map.
/// - follower thread's read returns an empty map without
///   touching the file (no IO cost on per-tid walks).
///
/// This is the leader-dedup contract that makes per-MM
/// data cheap to capture across thousands of threads.
#[test]
fn read_smaps_rollup_at_with_tally_dedups_to_leader_only() {
    let proc_root = tempfile::TempDir::new().unwrap();
    let tgid = 4242;
    let leader_tid = 4242;
    let follower_tid = 4243;

    // Stage `<tgid>/task/<leader_tid>/smaps_rollup`.
    let leader_dir = proc_root
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(leader_tid.to_string());
    std::fs::create_dir_all(&leader_dir).unwrap();
    std::fs::write(
        leader_dir.join("smaps_rollup"),
        "Rss:                2048 kB\n\
         Pss:                 512 kB\n",
    )
    .unwrap();

    // Stage `<tgid>/task/<follower_tid>/smaps_rollup` with a
    // POISON value — if the reader incorrectly opened it for
    // the follower it would read this and break the
    // assertion below.
    let follower_dir = proc_root
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(follower_tid.to_string());
    std::fs::create_dir_all(&follower_dir).unwrap();
    std::fs::write(
        follower_dir.join("smaps_rollup"),
        "Rss:                9999 kB\nPoison:           1 kB\n",
    )
    .unwrap();

    // Leader: file is read, map is populated.
    let m = read_smaps_rollup_at_with_tally(proc_root.path(), tgid, leader_tid, &mut None);
    assert_eq!(m.get("Rss"), Some(&2048));
    assert_eq!(m.get("Pss"), Some(&512));
    assert_eq!(m.len(), 2);

    // Follower: short-circuits to empty map BEFORE opening
    // the file. Catches a regression that flipped the
    // tid/tgid comparison or removed the dedup.
    let m = read_smaps_rollup_at_with_tally(proc_root.path(), tgid, follower_tid, &mut None);
    assert!(
        m.is_empty(),
        "follower thread must short-circuit to empty map; got {m:?}"
    );
}

/// Absent smaps_rollup file yields an empty map (older
/// kernels pre-4.14 lack this file; CAP_SYS_PTRACE-denied
/// reads under typical operator runs collapse the same way).
/// Pins the read-failure path.
#[test]
fn read_smaps_rollup_at_with_tally_absent_file_yields_empty_map() {
    let proc_root = tempfile::TempDir::new().unwrap();
    let tgid = 4242;
    let leader_tid = 4242;
    let leader_dir = proc_root
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(leader_tid.to_string());
    std::fs::create_dir_all(&leader_dir).unwrap();
    // No smaps_rollup file written — capture must not error.
    let m = read_smaps_rollup_at_with_tally(proc_root.path(), tgid, leader_tid, &mut None);
    assert!(m.is_empty(), "absent file → empty map; got {m:?}");
}

/// `parse_max_or_u64` distinguishes the kernel's literal
/// `max` token (no limit → `None`) from a concrete u64
/// (a configured cap). Whitespace-only and malformed input
/// collapses to `None` per the absent-counter contract.
#[test]
fn parse_max_or_u64_distinguishes_max_from_concrete_value() {
    assert_eq!(parse_max_or_u64("max"), None, "literal max → no limit");
    assert_eq!(
        parse_max_or_u64("max\n"),
        None,
        "trailing newline tolerated"
    );
    assert_eq!(
        parse_max_or_u64("9223372036854771712"),
        Some(9_223_372_036_854_771_712)
    );
    assert_eq!(parse_max_or_u64("0"), Some(0));
    assert_eq!(parse_max_or_u64(""), None, "empty input → no limit");
    assert_eq!(parse_max_or_u64("   "), None, "whitespace-only → no limit");
    assert_eq!(parse_max_or_u64("not_a_number"), None);
    // Negative values are not a kernel-emitted shape but the
    // parser tolerates them as malformed input → None.
    assert_eq!(parse_max_or_u64("-1"), None);
}

/// `parse_floor_value` is the FLOOR counterpart of
/// [`parse_max_or_u64`]: literal "max" means "maximum
/// protection" → `Some(u64::MAX)` (NOT `None`). `None` is
/// reserved for absent-file / malformed input. The
/// asymmetry vs. limits is load-bearing for the merge
/// step: `merge_min_option(Some(u64::MAX), Some(5G))`
/// yields 5G instead of None — preserving the lower
/// concrete floor when one contributor has full protection.
#[test]
fn parse_floor_value_treats_max_as_full_protection() {
    assert_eq!(
        parse_floor_value("max"),
        Some(u64::MAX),
        "literal max → maximum protection (NOT no floor)"
    );
    assert_eq!(parse_floor_value("max\n"), Some(u64::MAX));
    assert_eq!(parse_floor_value("0"), Some(0), "zero → no protection");
    assert_eq!(parse_floor_value("1073741824"), Some(1_073_741_824));
    assert_eq!(parse_floor_value(""), None, "empty → absent file");
    assert_eq!(parse_floor_value("not_a_number"), None);
}

/// `parse_cpu_max` decodes the two-token `<quota|max> <period>`
/// format. `period` falls back to the kernel default
/// 100_000 µs when malformed.
#[test]
fn parse_cpu_max_handles_quota_period_pairs() {
    // Concrete cap.
    assert_eq!(parse_cpu_max("50000 100000"), (Some(50_000), 100_000));
    // No cap (`max` token); period preserved.
    assert_eq!(parse_cpu_max("max 100000"), (None, 100_000));
    // Different period (50ms).
    assert_eq!(parse_cpu_max("25000 50000"), (Some(25_000), 50_000));
    // Missing period — period defaults to kernel default.
    assert_eq!(parse_cpu_max("50000"), (Some(50_000), 100_000));
    // Empty input — both default.
    assert_eq!(parse_cpu_max(""), (None, 100_000));
    // Malformed period falls back to the default.
    assert_eq!(parse_cpu_max("50000 garbage"), (Some(50_000), 100_000));
    // Trailing newline tolerated by split_ascii_whitespace.
    assert_eq!(parse_cpu_max("max 100000\n"), (None, 100_000));
}

/// Stage a synthetic cgroup tree with every captured cgroup v2
/// file present and verify [`read_cgroup_stats_at`] populates
/// the nested struct end-to-end. Pins file-naming, parse-routing,
/// and the absent-vs-no-limit distinction.
#[test]
fn read_cgroup_stats_at_populates_nested_controllers_end_to_end() {
    let cgroup_root = tempfile::TempDir::new().unwrap();
    let cg_dir = cgroup_root.path().join("app");
    std::fs::create_dir_all(&cg_dir).unwrap();
    std::fs::write(
        cg_dir.join("cpu.stat"),
        "usage_usec 12345\nnr_throttled 7\nthrottled_usec 8\n",
    )
    .unwrap();
    std::fs::write(cg_dir.join("cpu.max"), "50000 100000\n").unwrap();
    std::fs::write(cg_dir.join("cpu.weight"), "200\n").unwrap();
    std::fs::write(cg_dir.join("cpu.weight.nice"), "-5\n").unwrap();
    std::fs::write(cg_dir.join("memory.current"), "9999\n").unwrap();
    std::fs::write(cg_dir.join("memory.max"), "max\n").unwrap();
    std::fs::write(cg_dir.join("memory.high"), "1073741824\n").unwrap();
    std::fs::write(cg_dir.join("memory.low"), "0\n").unwrap();
    std::fs::write(cg_dir.join("memory.min"), "0\n").unwrap();
    std::fs::write(
        cg_dir.join("memory.stat"),
        "anon 100\nfile 200\npgfault 18\nslab 50\n",
    )
    .unwrap();
    std::fs::write(
        cg_dir.join("memory.events"),
        "low 0\nhigh 1\nmax 0\noom 0\noom_kill 0\n",
    )
    .unwrap();
    std::fs::write(cg_dir.join("pids.current"), "42\n").unwrap();
    std::fs::write(cg_dir.join("pids.max"), "1024\n").unwrap();

    let stats = read_cgroup_stats_at(cgroup_root.path(), "/app");

    // CPU domain.
    assert_eq!(stats.cpu.usage_usec, 12_345);
    assert_eq!(stats.cpu.nr_throttled, 7);
    assert_eq!(stats.cpu.throttled_usec, 8);
    assert_eq!(stats.cpu.max_quota_us, Some(50_000));
    assert_eq!(stats.cpu.max_period_us, 100_000);
    assert_eq!(stats.cpu.weight, Some(200));
    assert_eq!(stats.cpu.weight_nice, Some(-5));

    // Memory domain.
    assert_eq!(stats.memory.current, 9999);
    assert_eq!(stats.memory.max, None, "literal max → no limit");
    assert_eq!(stats.memory.high, Some(1_073_741_824));
    assert_eq!(stats.memory.low, Some(0));
    assert_eq!(stats.memory.min, Some(0));
    assert_eq!(stats.memory.stat.get("anon"), Some(&100));
    assert_eq!(stats.memory.stat.get("file"), Some(&200));
    assert_eq!(stats.memory.stat.get("pgfault"), Some(&18));
    assert_eq!(stats.memory.stat.get("slab"), Some(&50));
    assert_eq!(stats.memory.events.get("oom_kill"), Some(&0));
    assert_eq!(stats.memory.events.get("high"), Some(&1));

    // PIDs domain.
    assert_eq!(stats.pids.current, Some(42));
    assert_eq!(stats.pids.max, Some(1024));
}

/// Root cgroup typically lacks every knob/limit file. Pins
/// the absent-vs-no-limit distinction: `Option<u64>` limits
/// stay `None` (file absent), counters stay 0 (Default),
/// and `max_period_us` defaults to the kernel default
/// rather than zero.
#[test]
fn read_cgroup_stats_at_root_cgroup_collapses_to_defaults() {
    let cgroup_root = tempfile::TempDir::new().unwrap();
    // No files at all under root — simulating a v2 mount
    // root that only carries `cgroup.*` files (no domain
    // controllers populated).
    let stats = read_cgroup_stats_at(cgroup_root.path(), "/");
    assert_eq!(stats.cpu.usage_usec, 0);
    assert_eq!(stats.cpu.max_quota_us, None);
    assert_eq!(
        stats.cpu.max_period_us, CPU_MAX_DEFAULT_PERIOD_US,
        "absent cpu.max → period defaults to kernel default"
    );
    assert_eq!(stats.cpu.weight, None);
    assert_eq!(stats.memory.current, 0);
    assert_eq!(stats.memory.max, None);
    assert_eq!(stats.memory.high, None);
    assert!(stats.memory.stat.is_empty());
    assert!(stats.memory.events.is_empty());
    assert_eq!(stats.pids.current, None);
    assert_eq!(stats.pids.max, None);
}

#[test]
fn parse_cgroup_v2_picks_unified_hierarchy() {
    let raw = "12:cpuset:/legacy/cpuset/path\n\
               0::/unified/path\n\
               5:freezer:/legacy/freezer\n";
    assert_eq!(parse_cgroup_v2(raw), Some("/unified/path".to_string()));
}

#[test]
fn parse_cgroup_v2_none_when_only_legacy_present() {
    let raw = "12:cpuset:/legacy/path\n";
    assert_eq!(parse_cgroup_v2(raw), None);
}

#[test]
fn parse_sched_accepts_prefixed_and_bare_keys() {
    let raw = "se.statistics.nr_wakeups            :     1000\n\
               se.nr_migrations                    :     42\n\
               se.statistics.nr_wakeups_local      :     600\n\
               se.statistics.wait_sum              :     12345.678\n";
    let f = parse_sched(raw, &mut None);
    assert_eq!(f.nr_wakeups, Some(1000));
    assert_eq!(f.nr_migrations, Some(42));
    assert_eq!(f.nr_wakeups_local, Some(600));
    // PN_SCHEDSTAT format: ms.ns_remainder. `12345.678`
    // pads `.678` → `.678000` (= 678_000 ns), then
    // 12345 * 1_000_000 + 678_000 = 12_345_678_000 ns.
    assert_eq!(f.wait_sum, Some(12_345_678_000));
}

#[test]
fn parse_cpu_stat_space_separated_format() {
    let raw = "usage_usec 1234\n\
               user_usec 1000\n\
               system_usec 234\n\
               nr_periods 10\n\
               nr_throttled 2\n\
               throttled_usec 500\n";
    let (usage, throttled, throttled_usec) = parse_cpu_stat(raw);
    assert_eq!(usage, Some(1234));
    assert_eq!(throttled, Some(2));
    assert_eq!(throttled_usec, Some(500));
}

#[test]
fn policy_name_known_and_unknown() {
    assert_eq!(policy_name(libc::SCHED_OTHER), "SCHED_OTHER");
    assert_eq!(policy_name(libc::SCHED_FIFO), "SCHED_FIFO");
    assert_eq!(policy_name(libc::SCHED_RR), "SCHED_RR");
    assert_eq!(policy_name(libc::SCHED_BATCH), "SCHED_BATCH");
    assert_eq!(policy_name(libc::SCHED_IDLE), "SCHED_IDLE");
    assert_eq!(policy_name(6), "SCHED_DEADLINE");
    assert_eq!(policy_name(7), "SCHED_EXT");
    assert_eq!(policy_name(99), "SCHED_UNKNOWN(99)");
}

#[test]
fn iter_tgids_includes_self() {
    let tgids = iter_tgids_at(Path::new(DEFAULT_PROC_ROOT));
    let pid = std::process::id() as i32;
    assert!(tgids.contains(&pid), "self pid {pid} not in /proc walk");
}

#[test]
fn iter_task_ids_self_returns_at_least_main_tid() {
    let pid = std::process::id() as i32;
    let tids = iter_task_ids_at(Path::new(DEFAULT_PROC_ROOT), pid);
    assert!(
        tids.contains(&pid),
        "main tid {pid} absent from /proc/self/task"
    );
}

#[test]
fn read_process_comm_for_self_is_populated() {
    let pid = std::process::id() as i32;
    let comm = read_process_comm_at(Path::new(DEFAULT_PROC_ROOT), pid)
        .expect("self comm must be readable");
    assert!(!comm.is_empty());
}

#[test]
fn capture_thread_self_populates_identity() {
    let pid = std::process::id() as i32;
    let t = capture_thread(pid, pid, "testproc");
    assert_eq!(t.tid, pid as u32);
    assert_eq!(t.tgid, pid as u32);
    assert_eq!(t.pcomm, "testproc");
    assert!(!t.comm.is_empty());
    // On a real /proc, start_time_clock_ticks populates for live tasks.
    assert!(t.start_time_clock_ticks > 0);
    // Policy at minimum resolves to SCHED_OTHER for a normal process.
    assert!(!t.policy.0.is_empty());
}

#[test]
fn capture_produces_non_empty_snapshot() {
    // Scope to self_pid so the probe-attach pass is skipped (the
    // capture pipeline excludes the calling process from the
    // ptrace path because PTRACE_SEIZE rejects self-attach). The
    // global `capture()` would attempt to probe every jemalloc-
    // linked tgid on the host — orders of magnitude slower in a
    // unit-test context, and not what this test is asserting on.
    // The wiring-end-to-end test path lives in
    // `tests/ctprof_capture_jemalloc_wiring.rs`, which spawns
    // a real jemalloc target.
    let pid = std::process::id() as i32;
    let snap = capture_pid(pid);
    assert!(!snap.threads.is_empty());
    let self_threads: Vec<_> = snap
        .threads
        .iter()
        .filter(|t| t.tgid == pid as u32)
        .collect();
    assert!(!self_threads.is_empty(), "own tgid missing from capture");
}

#[test]
fn snapshot_extension_is_stable() {
    // Guard against accidental rename of the canonical extension.
    assert_eq!(SNAPSHOT_EXTENSION, "ctprof.zst");
}

// ------------------------------------------------------------
// Parser edge-case coverage expansion
//
// The existing parse_* tests above cover the documented happy
// paths plus the most-adversarial documented edge cases
// (paren-in-comm, huge ranges, fractional fields). The tests
// below cover MALFORMED, EMPTY, and BOUNDARY inputs that the
// parsers silently absorb — regressions in this family would
// land as stray data in the snapshot rather than loud failures,
// which is exactly the class of drift the capture contract
// ("absent = 0, best-effort, never-fail-the-snapshot") needs a
// test gate against.
// ------------------------------------------------------------

/// parse_io on empty input produces the default `IoFields`
/// (every field `None`). Empty input happens when `/proc/<tid>/io`
/// is present but the kernel was compiled without
/// `CONFIG_TASK_IO_ACCOUNTING` — the file exists with zero
/// bytes. Without this gate the parser would silently accept
/// the no-lines case by producing `IoFields::default()` anyway,
/// but a regression that inverted an `if`/ early-returned a
/// partial default would surface here.
#[test]
fn parse_io_empty_input_yields_all_none() {
    let f = parse_io("");
    assert_eq!(f, IoFields::default());
}

/// parse_io with a non-numeric value for a known key must drop
/// ONLY the offending field — other lines still populate. Proves
/// per-field `parse::<u64>().ok()` isolation rather than a
/// whole-file bail that would zero out unrelated counters.
#[test]
fn parse_io_malformed_value_drops_only_that_field() {
    let raw = "rchar: 100\n\
               wchar: not-a-number\n\
               syscr: 3\n";
    let f = parse_io(raw);
    assert_eq!(f.rchar, Some(100));
    assert_eq!(f.wchar, None, "malformed value drops to None");
    assert_eq!(f.syscr, Some(3));
}

/// parse_stat on a line with NO `)` returns `Default` — the
/// `rfind(')')` guard in parse_stat short-circuits to
/// `StatFields::default()` without tripping on out-of-bounds.
/// A procfs file that got truncated mid-comm (impossible under
/// correct procfs but possible against a fuzzer / synthetic
/// tree) must not panic.
#[test]
fn parse_stat_empty_and_no_paren_return_default() {
    assert_eq!(parse_stat(""), StatFields::default());
    assert_eq!(
        parse_stat("garbage line with no close paren 1 2 3"),
        StatFields::default(),
        "line without `)` must return Default, not panic on \
         out-of-bounds indexing",
    );
    assert_eq!(
        parse_stat("  \n"),
        StatFields::default(),
        "whitespace-only input must also land at Default",
    );
}

/// parse_stat on multi-line input reads ONLY the first line.
/// Production procfs stat is single-line; a synthetic
/// multi-line file (e.g. a test fixture that appended extra
/// rows by mistake, or a fuzz input) must not mix field
/// positions across lines. Pins the `.lines().next()` behavior
/// so a future refactor that concatenated lines would surface
/// here.
#[test]
fn parse_stat_multi_line_input_uses_only_first_line() {
    let mut first = String::from("1 (proc) ");
    for i in 0..=38 {
        first.push_str(&format!("{i} "));
    }
    // Second line carries clearly-different values — if the
    // parser concatenated or mixed them, `nice` would change.
    let second = "2 (other) 999 999 999 999 999 999 999 999 999 999 \
                  999 999 999 999 999 999 999 999 999 999 999 999 999\n";
    let raw = format!("{first}\n{second}");
    let f = parse_stat(&raw);
    // First-line values untouched.
    assert_eq!(f.nice, Some(16));
    assert_eq!(f.start_time_clock_ticks, Some(19));
    assert_eq!(f.policy, Some(38));
}

/// parse_schedstat with more than three leading fields must
/// accept the first three and ignore the rest. Real procfs
/// stops at three, but a future kernel could append more or a
/// synthetic fixture could pad the line — the parser's
/// three-next-calls design already ignores tail tokens, and
/// this test pins that invariant.
///
/// Also covers the "invalid u64 token" path — a non-numeric
/// token routes to None via `.parse::<u64>().ok()`.
#[test]
fn parse_schedstat_extra_fields_and_invalid_tokens() {
    // Four fields — fourth ignored.
    let (a, b, c) = parse_schedstat("1 2 3 4\n");
    assert_eq!((a, b, c), (Some(1), Some(2), Some(3)));
    // Invalid middle token drops only that slot.
    let (a, b, c) = parse_schedstat("1 invalid 3\n");
    assert_eq!(a, Some(1));
    assert_eq!(b, None);
    assert_eq!(c, Some(3));
    // Empty input → all None.
    let (a, b, c) = parse_schedstat("");
    assert_eq!((a, b, c), (None, None, None));
}

/// policy_name on a NEGATIVE integer must format as
/// `"SCHED_UNKNOWN(-N)"` rather than panicking or producing an
/// unsigned-wrapped value. The kernel's `policy` field is
/// signed i32 (see `parse_stat::get_i32`), so a corrupt or
/// out-of-band synthetic fixture could carry a negative value;
/// the fallback branch must handle it cleanly.
#[test]
fn policy_name_negative_integer_renders_unknown() {
    assert_eq!(policy_name(-1), "SCHED_UNKNOWN(-1)");
    assert_eq!(
        policy_name(i32::MIN),
        format!("SCHED_UNKNOWN({})", i32::MIN)
    );
}

/// parse_cpu_stat on empty input produces all-`None`. Same
/// shape as `parse_io_empty_input_yields_all_none`, but
/// exercises the space-separated key/value format rather than
/// the `key: value` colon format — they are distinct parsers.
#[test]
fn parse_cpu_stat_empty_and_keyonly_lines_yield_none() {
    let (u, t, tu) = parse_cpu_stat("");
    assert_eq!((u, t, tu), (None, None, None));
    // Line with key but no value — dropped. The `parts.next()`
    // for value returns None → `continue`.
    let (u, t, tu) = parse_cpu_stat("usage_usec\n");
    assert_eq!((u, t, tu), (None, None, None));
}

/// parse_status with ONLY `voluntary_ctxt_switches` present
/// populates only that field — the other two stay `None`. The
/// production capture path coerces these to `0`; pinning the
/// `None` at the parser layer proves the "absent vs. zero"
/// distinction survives through the pure parser even if a
/// future refactor separates the coercion.
#[test]
fn parse_status_partial_and_malformed_fields_isolate_correctly() {
    // Only voluntary_csw → other two None.
    let only_v = "Name:\tfoo\n\
                  voluntary_ctxt_switches:\t9\n";
    let f = parse_status(only_v);
    assert_eq!(f.voluntary_csw, Some(9));
    assert_eq!(f.nonvoluntary_csw, None);
    assert_eq!(f.cpus_allowed, None);

    // Malformed Cpus_allowed_list → cpus_allowed None (parse_cpu_list
    // returns None on bad tokens). Other fields still populate.
    let bad_cpu_list = "Cpus_allowed_list:\t5-3\n\
                        voluntary_ctxt_switches:\t1\n";
    let f = parse_status(bad_cpu_list);
    assert_eq!(f.voluntary_csw, Some(1));
    assert_eq!(
        f.cpus_allowed, None,
        "malformed cpulist must route parse_cpu_list's None \
         into the StatusFields field — not collapse to empty vec",
    );
}

/// parse_cgroup_v2 with an empty path (`"0::\n"`) returns None
/// because the `!trimmed.is_empty()` guard rejects the blank
/// path. A kernel bug or a synthetic fixture that emitted
/// `0::` without a path must not land an empty-string cgroup
/// in the ThreadState (which would then group with other
/// cgroup-less threads and produce noise).
///
/// Also pins the first-wins behavior when multiple unified
/// lines appear — real procfs emits ONE v2 line per task, but
/// a fixture might pad with duplicates; the parser returns on
/// the first valid match.
#[test]
fn parse_cgroup_v2_empty_path_and_multiple_unified_lines() {
    // Empty path after `0::` — the guard rejects.
    assert_eq!(parse_cgroup_v2("0::\n"), None);
    assert_eq!(parse_cgroup_v2("0::   \n"), None);

    // First unified line wins when duplicates exist.
    let raw = "0::/first\n0::/second\n";
    assert_eq!(parse_cgroup_v2(raw), Some("/first".to_string()));
}

/// `read_thread_comm_at` returns `None` (not `Some("")`) when
/// the comm file exists but contains only whitespace. The
/// trim-then-is-empty guard is load-bearing: a `Some("")` in
/// ThreadState.comm would both (a) disable the empty-comm ghost
/// filter and (b) pollute comparisons grouped by comm.
/// Pins the explicit empty→None routing so a future refactor
/// that simplified the fn to `.ok().map(|s| s.trim().to_string())`
/// (losing the empty guard) would break this test.
#[test]
fn read_thread_comm_at_whitespace_only_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let tgid = 1;
    let tid = 1;
    let task_dir = tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    std::fs::create_dir_all(&task_dir).unwrap();
    std::fs::write(task_dir.join("comm"), "   \n").unwrap();
    assert_eq!(read_thread_comm_at(tmp.path(), tgid, tid), None);

    // Also the missing-file branch (thread exited mid-read).
    assert_eq!(read_thread_comm_at(tmp.path(), tgid, 9999), None);
}

// ------------------------------------------------------------
// Synthetic-tree tests (H1-H5)
//
// Stage a tempdir shaped like `/proc/<tgid>/{comm,
// task/<tid>/{stat,schedstat,status,io,sched,comm,cgroup}}`
// so every capture helper can be driven without touching the
// real procfs. Mirrors the compare-side pattern in
// tests/ctprof_compare.rs but against the capture side.
// ------------------------------------------------------------

/// Build a synthetic `/proc` under `root` carrying exactly one
/// thread. Writes every file capture walks so every counter
/// on `ThreadState` round-trips with a known value. `cpus` is
/// the `Cpus_allowed_list` value (a range string the
/// `parse_cpu_list` helper decodes).
fn stage_synthetic_proc(root: &Path, tgid: i32, tid: i32, pcomm: &str, comm: &str) {
    use std::fs;
    let tgid_dir = root.join(tgid.to_string());
    let task_dir = tgid_dir.join("task").join(tid.to_string());
    fs::create_dir_all(&task_dir).unwrap();

    // /proc/<tgid>/comm
    fs::write(tgid_dir.join("comm"), format!("{pcomm}\n")).unwrap();
    // /proc/<tgid>/task/<tid>/comm
    fs::write(task_dir.join("comm"), format!("{comm}\n")).unwrap();

    // stat: paren-safe comm, fields 1..41. Comm inserted with
    // parens inside so the rfind(')') anchor has to find the
    // LAST close-paren, not the first. Fields past comm start
    // at index 0 in `tail` (tail[0] is `state`, per procfs
    // field-index-minus-three convention that parse_stat uses).
    // Field indices (post-comm):
    //   [0]=state [1]=ppid [2]=pgrp [3]=session [4]=tty
    //   [5]=tpgid [6]=flags [7]=minflt(field 10)
    //   [8]=cminflt [9]=majflt(field 12) [10]=cmajflt
    //   [11..16]=utime/stime/cutime/cstime/priority
    //   [16]=nice (field 19) [17]=num_threads [18]=itrealvalue
    //   [19]=starttime (field 22) [20..37]=vsize/rss/...
    //   [38]=policy (field 41).
    let stat_line = format!(
        "{tid} (proc (with) parens) R 1 2 3 4 5 6 \
         7777 0 8888 0 10 11 12 13 14 {nice} 1 0 \
         {starttime} 100 200 300 400 500 600 700 800 \
         900 1000 1100 1200 1300 1400 1500 1600 1700 1800 {policy}\n",
        tid = tid,
        nice = -10_i32,
        starttime = 555_555u64,
        policy = 0, // SCHED_OTHER
    );
    fs::write(task_dir.join("stat"), stat_line).unwrap();

    // schedstat: run_time_ns wait_time_ns timeslices
    fs::write(task_dir.join("schedstat"), "1000000 200000 50\n").unwrap();

    // status: State + voluntary/nonvoluntary csw + Cpus_allowed_list.
    // parse_status matches the lowercase csw keys verbatim;
    // `State` and `Cpus_allowed_list` use the capitalised
    // leading char of the procfs file.
    let status = "Name:\tfoo\n\
         State:\tR (running)\n\
         voluntary_ctxt_switches:\t42\n\
         nonvoluntary_ctxt_switches:\t7\n\
         Cpus_allowed_list:\t0-3\n";
    fs::write(task_dir.join("status"), status).unwrap();

    // io: cumulative byte counters
    let io = "rchar: 100\n\
         wchar: 200\n\
         syscr: 10\n\
         syscw: 20\n\
         read_bytes: 4096\n\
         write_bytes: 8192\n\
         cancelled_write_bytes: 512\n";
    fs::write(task_dir.join("io"), io).unwrap();

    // sched: every parse_sched-matched key, with the
    // `se.statistics.` prefix for the wakeup family to
    // exercise the rsplit('.') short-key logic. `ext.enabled`
    // is unprefixed (literal kernel key) and tests the
    // full-key gate.
    let sched = "\
         se.statistics.nr_wakeups                       :         11\n\
         se.statistics.nr_wakeups_local                 :          8\n\
         se.statistics.nr_wakeups_remote                :          3\n\
         se.statistics.nr_wakeups_sync                  :          2\n\
         se.statistics.nr_wakeups_migrate               :          1\n\
         se.statistics.nr_wakeups_idle                  :          4\n\
         se.statistics.nr_wakeups_affine                :         12\n\
         se.statistics.nr_wakeups_affine_attempts       :         20\n\
         nr_migrations                                  :          9\n\
         se.statistics.nr_migrations_cold               :          5\n\
         se.statistics.nr_forced_migrations             :          7\n\
         se.statistics.nr_failed_migrations_affine      :          1\n\
         se.statistics.nr_failed_migrations_running     :          2\n\
         se.statistics.nr_failed_migrations_hot         :          3\n\
         wait_sum                                       :    5000.25\n\
         wait_count                                     :         15\n\
         se.statistics.wait_max                         :     250.5\n\
         sum_sleep_runtime                              :    3200.50\n\
         se.statistics.sleep_max                        :     180.25\n\
         sum_block_runtime                              :    1100.75\n\
         se.statistics.block_max                        :      60.75\n\
         iowait_sum                                     :       77.0\n\
         iowait_count                                   :         18\n\
         se.statistics.exec_max                         :      90.0\n\
         se.statistics.slice_max                        :     400.5\n\
         ext.enabled                                    :          1\n";
    fs::write(task_dir.join("sched"), sched).unwrap();

    // cgroup: v2-style single entry (0::path). read_cgroup_at
    // parses the `0::` prefix.
    fs::write(task_dir.join("cgroup"), "0::/ktstr.slice/worker0\n").unwrap();
}

/// Ghost-thread filter: a tid whose directory exists but
/// carries ZERO readable procfs files (classic mid-capture
/// exit — readdir races the reap) assembles an all-Default
/// `ThreadState` and must NOT land in the snapshot. Stages
/// one live thread with real content and one empty-directory
/// ghost tid under the same tgid, calls `capture_with`, and
/// asserts the output contains only the live thread.
///
/// Without the filter, the ghost would land as `{ tid: 202,
/// comm: "", cgroup: "", start_time_clock_ticks: 0, ...all
/// counters zero }` and pollute downstream comparisons — a
/// baseline run captures some number of ghosts, the candidate
/// captures a different number, and the diff surfaces spurious
/// "thread vanished" signal on every report.
#[test]
fn capture_with_filters_ghost_threads_with_empty_comm_and_zero_start() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 42;
    let live_tid: i32 = 101;
    let ghost_tid: i32 = 202;

    // Stage the live thread in full.
    stage_synthetic_proc(proc_tmp.path(), tgid, live_tid, "pcomm-proc", "live-thread");

    // Stage a ghost tid directory with NO inner files —
    // simulates the "readdir saw it, per-file reads all
    // ENOENT'd" race window. `iter_task_ids_at` enumerates
    // it (the numeric dir name parses), every capture read
    // returns the default, and the filter rejects the
    // resulting all-zero entry.
    let ghost_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(ghost_tid.to_string());
    std::fs::create_dir_all(&ghost_dir).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    // Exactly one thread — the live one. The ghost is gone.
    assert_eq!(
        snap.threads.len(),
        1,
        "ghost tid with empty comm + zero start must be filtered; \
         got threads: {:?}",
        snap.threads
            .iter()
            .map(|t| (t.tid, &t.comm))
            .collect::<Vec<_>>(),
    );
    assert_eq!(snap.threads[0].tid, live_tid as u32);
    assert_eq!(snap.threads[0].comm, "live-thread");
}

/// H1 + H2 — `capture_with` against a synthetic procfs:
/// staging every file the capture walks and asserting the
/// assembled `ThreadState` carries the planted values.
#[test]
fn capture_with_synthetic_tree_assembles_thread_state() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 42;
    let tid: i32 = 101;

    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "pcomm-proc", "worker-thread");

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    // Exactly one thread — the one we planted.
    assert_eq!(snap.threads.len(), 1, "synthetic proc has one tid");
    let t = &snap.threads[0];

    // Identity fields (round-trip from /proc/<tgid>/comm +
    // /proc/<tgid>/task/<tid>/comm).
    assert_eq!(t.tid, tid as u32);
    assert_eq!(t.tgid, tgid as u32);
    assert_eq!(t.pcomm, "pcomm-proc");
    assert_eq!(t.comm, "worker-thread");
    assert_eq!(t.cgroup, "/ktstr.slice/worker0");

    use crate::metric_types::{
        Bytes, CategoricalString, ClockTicks, CpuSet, MonotonicCount, MonotonicNs, OrdinalI32,
        PeakNs,
    };

    // /proc/<tid>/stat fields parsed out of the paren-comm
    // tail: nice, utime, stime, starttime, processor, policy,
    // minflt, majflt.
    assert_eq!(t.nice, OrdinalI32(-10));
    assert_eq!(t.start_time_clock_ticks, 555_555);
    assert_eq!(t.policy, CategoricalString::from("SCHED_OTHER"));
    assert_eq!(t.minflt, MonotonicCount(7777));
    assert_eq!(t.majflt, MonotonicCount(8888));
    assert_eq!(
        t.utime_clock_ticks,
        ClockTicks(10),
        "tail[11] of stat fixture lands at utime_clock_ticks",
    );
    assert_eq!(
        t.stime_clock_ticks,
        ClockTicks(11),
        "tail[12] of stat fixture lands at stime_clock_ticks",
    );
    assert_eq!(
        t.processor,
        OrdinalI32(1700),
        "tail[36] of stat fixture (the 17th post-starttime \
         token, value 100*17=1700) lands at processor",
    );

    // schedstat — three-tuple of run/wait/slices.
    assert_eq!(t.run_time_ns, MonotonicNs(1_000_000));
    assert_eq!(t.wait_time_ns, MonotonicNs(200_000));
    assert_eq!(t.timeslices, MonotonicCount(50));

    // status — state + csw + Cpus_allowed_list. With
    // `use_syscall_affinity=false`, the capture path reads
    // cpu_affinity from status only.
    assert_eq!(
        t.state, 'R',
        "first non-whitespace char of `State:\tR (running)` is \
         the single-letter code R",
    );
    assert_eq!(t.voluntary_csw, MonotonicCount(42));
    assert_eq!(t.nonvoluntary_csw, MonotonicCount(7));
    assert_eq!(t.cpu_affinity, CpuSet(vec![0, 1, 2, 3]));

    // io — seven cumulative counters.
    assert_eq!(t.rchar, Bytes(100));
    assert_eq!(t.wchar, Bytes(200));
    assert_eq!(t.syscr, MonotonicCount(10));
    assert_eq!(t.syscw, MonotonicCount(20));
    assert_eq!(t.read_bytes, Bytes(4096));
    assert_eq!(t.write_bytes, Bytes(8192));
    assert_eq!(
        t.cancelled_write_bytes,
        Bytes(512),
        "cancelled_write_bytes round-trips from the 7th line of \
         /proc/<tid>/io",
    );

    // sched — every wakeup field, migrations (live counters
    // only; the dead-counter fields nr_wakeups_idle /
    // nr_migrations_cold / nr_wakeups_passive are no longer
    // surfaced on ThreadState — the kernel never increments
    // them so the registry was the wrong place for them; the
    // synthetic fixture still emits the lines to exercise the
    // parser's silent-drop on unknown keys), the four *_sum
    // fractional-parse fields, the five *_max fractional-parse
    // fields, and the ext.enabled bool.
    assert_eq!(t.nr_wakeups, MonotonicCount(11));
    assert_eq!(t.nr_wakeups_local, MonotonicCount(8));
    assert_eq!(t.nr_wakeups_remote, MonotonicCount(3));
    assert_eq!(t.nr_wakeups_sync, MonotonicCount(2));
    assert_eq!(t.nr_wakeups_migrate, MonotonicCount(1));
    assert_eq!(t.nr_wakeups_affine, MonotonicCount(12));
    assert_eq!(
        t.nr_wakeups_affine_attempts,
        MonotonicCount(20),
        "denominator for the affine-wake success ratio \
         (nr_wakeups_affine / nr_wakeups_affine_attempts = 12/20)",
    );
    assert_eq!(t.nr_migrations, MonotonicCount(9));
    assert_eq!(t.nr_forced_migrations, MonotonicCount(7));
    assert_eq!(t.nr_failed_migrations_affine, MonotonicCount(1));
    assert_eq!(t.nr_failed_migrations_running, MonotonicCount(2));
    assert_eq!(t.nr_failed_migrations_hot, MonotonicCount(3));
    // PN_SCHEDSTAT format is ms.ns_remainder. Reconstructed
    // ns = ms_part * 1_000_000 + zero-right-padded ns_part.
    // `5000.25` → `.25` pads to `.250000` (=250_000 ns) +
    // 5000ms × 1_000_000 = 5_000_250_000 ns total.
    assert_eq!(
        t.wait_sum,
        MonotonicNs(5_000_250_000),
        "PN_SCHEDSTAT 5000.25 reconstructs to 5_000_250_000 ns \
         (5000ms + 250_000ns)",
    );
    assert_eq!(t.wait_count, MonotonicCount(15));
    assert_eq!(
        t.wait_max,
        PeakNs(250_500_000),
        "PN_SCHEDSTAT 250.5 reconstructs to 250_500_000 ns",
    );
    // voluntary_sleep_ns = sum_sleep_runtime - sum_block_runtime,
    // computed at capture: 3_200_500_000 - 1_100_750_000 =
    // 2_099_750_000 ns. The kernel's sum_sleep_runtime
    // double-counts block under sleep, so the normalized
    // voluntary-only residual is what surfaces on ThreadState.
    assert_eq!(
        t.voluntary_sleep_ns,
        MonotonicNs(2_099_750_000),
        "voluntary_sleep_ns = sum_sleep_runtime (3_200_500_000) \
         minus sum_block_runtime (1_100_750_000) = \
         2_099_750_000 ns; capture-side normalization strips \
         the kernel's sleep/block double-count",
    );
    assert_eq!(
        t.sleep_max,
        PeakNs(180_250_000),
        "PN_SCHEDSTAT 180.25 reconstructs to 180_250_000 ns",
    );
    assert_eq!(
        t.block_sum,
        MonotonicNs(1_100_750_000),
        "PN_SCHEDSTAT 1100.75 reconstructs to 1_100_750_000 ns; \
         block_sum is populated from the kernel's `sum_block_runtime` key",
    );
    assert_eq!(
        t.block_max,
        PeakNs(60_750_000),
        "PN_SCHEDSTAT 60.75 reconstructs to 60_750_000 ns",
    );
    assert_eq!(
        t.iowait_sum,
        MonotonicNs(77_000_000),
        "PN_SCHEDSTAT 77.0 reconstructs to 77_000_000 ns",
    );
    assert_eq!(t.iowait_count, MonotonicCount(18));
    assert_eq!(
        t.exec_max,
        PeakNs(90_000_000),
        "PN_SCHEDSTAT 90.0 reconstructs to 90_000_000 ns",
    );
    assert_eq!(
        t.slice_max,
        PeakNs(400_500_000),
        "PN_SCHEDSTAT 400.5 reconstructs to 400_500_000 ns",
    );
    assert!(
        t.ext_enabled,
        "ext.enabled = 1 round-trips through the full-key gate \
         to ThreadState::ext_enabled true",
    );

    // jemalloc TSD counters: synthetic procfs has no real ELF
    // behind /proc/<tgid>/exe, so the probe attach is gated off
    // (use_syscall_affinity=false). Both fields land at the
    // absent-counter default of 0. Pins this so a future
    // regression that always-probes (ignoring use_syscall_affinity)
    // would either crash on the synthetic /proc or surface garbage
    // here.
    assert_eq!(
        t.allocated_bytes,
        Bytes(0),
        "synthetic-tree capture must not probe — allocated_bytes \
         collapses to absent-counter zero",
    );
    assert_eq!(
        t.deallocated_bytes,
        Bytes(0),
        "synthetic-tree capture must not probe — deallocated_bytes \
         collapses to absent-counter zero",
    );
}

/// Capture against an empty `proc_root` (no tgid subdirs at
/// all) must complete without panic and produce an empty
/// snapshot. Pins the rayon parallel-probe phase's empty-input
/// handling: `iter_tgids_at` returns an empty Vec, `par_iter`
/// over zero elements collects to an empty HashMap, and the
/// sequential phase 2 loop runs zero iterations. `use_syscall_affinity=true`
/// is required to enter the rayon block at all (the `false`
/// branch skips probe-attach entirely and assigns an empty
/// HashMap directly). Without this gate test, the rayon
/// par_iter over empty input has zero coverage.
#[test]
fn capture_with_empty_proc_root_produces_empty_snapshot() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Stage `/proc/loadavg` so the parallelism-clamp read at
    // <proc_root>/loadavg succeeds rather than falling back to
    // the 0.0 default. Empty `proc_root` otherwise — no tgid
    // subdirs, so `iter_tgids_at` returns Vec::new().
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "empty proc_root must produce empty snapshot; got {} threads",
        snap.threads.len(),
    );
}

/// Exercises the cache-lookup and insert code path in the
/// rayon probe loop. Two tgids whose `/proc/<tgid>/exe`
/// symlinks resolve to the same underlying inode trigger
/// cache interaction: both attach calls fail with
/// AttachError::MapsReadFailure (the synthetic tree has no
/// `/proc/<tgid>/maps`), and the absent-counter contract
/// holds — both threads land in the snapshot with
/// allocated_bytes==0.
#[test]
fn capture_with_inode_cache_collapses_duplicate_binaries() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Required by the parallelism-clamp logic in capture_with.
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    // One real file, two symlinks pointing at it. Both tgids'
    // exe metadata calls return the same (dev, ino) tuple, so
    // the cache_key matches across them.
    let shared_exe = proc_tmp.path().join("shared-exe");
    std::fs::write(&shared_exe, b"\x7fELFsynthetic\n").unwrap();

    for tgid in [4242, 4243] {
        stage_synthetic_proc(
            proc_tmp.path(),
            tgid,
            tgid + 1,
            "shared-pcomm",
            "shared-comm",
        );
        // `/proc/<tgid>/exe` symlink points at the shared file.
        // `attach_jemalloc_at` will read_link this successfully
        // and then fail on the absent `/proc/<tgid>/maps` →
        // AttachError::MapsReadFailure. The cache stores None
        // keyed by (dev, ino) of the shared file.
        let exe_link = proc_tmp.path().join(tgid.to_string()).join("exe");
        std::os::unix::fs::symlink(&shared_exe, &exe_link).unwrap();
    }

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);

    // Both threads still land in the snapshot — the failed
    // attach just leaves allocated_bytes at the absent-counter
    // default of zero. If the cache-hit branch panicked
    // (poisoned mutex, key collision logic, etc.), the rayon
    // worker would crash and `capture_with` would not return.
    assert_eq!(
        snap.threads.len(),
        2,
        "both staged threads must land in the snapshot",
    );
    for thread in &snap.threads {
        assert_eq!(
            thread.allocated_bytes,
            Bytes(0),
            "synthetic /proc has no maps; attach fails, allocated_bytes \
             collapses to absent-counter zero — cache-hit branch must not \
             fabricate a non-zero counter",
        );
    }
}

// ------------------------------------------------------------
// Capture-pipeline error paths (Batch A + B)
//
// The synthetic-tree happy path is covered by
// capture_with_synthetic_tree_assembles_thread_state above.
// The tests below pin the pipeline's behavior against
// adversarial inputs:
// - missing/empty proc_root and tgid dirs (Batch A)
// - non-numeric junk under proc_root (Batch A)
// - capture_pid_with against pids that don't exist or are
//   ghost (Batch A + B)
// - selectively malformed/corrupted procfs files leaving
//   the matching ThreadState fields zero-defaulted (Batch B)
//
// Each test uses stage_synthetic_proc to lay down a known-
// good baseline, then mutates one specific axis. Assertions
// include observed value, expected value, and likely root
// cause so a regression points the reader at the failure
// mode without re-derivation.
// ------------------------------------------------------------

/// G1 — proc_root pointing at a directory that does NOT
/// exist must NOT panic. Pipeline collapses to an empty
/// snapshot via `iter_tgids_at`'s read_dir-fail-→-empty-Vec
/// guard. Defends against a future change that bubbled the
/// I/O error to the caller.
#[test]
fn capture_with_nonexistent_proc_root_produces_empty_snapshot() {
    let scratch = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    // A path inside a fresh tempdir that we never create —
    // guaranteed to not exist within this test's scope.
    // io::read_dir returns ENOENT, iter_tgids_at returns
    // Vec::new(). Use false for use_syscall_affinity so the
    // parallel probe phase is fully skipped. Reuse the same
    // nonexistent path for sys_root: this test exercises the
    // ENOENT-collapses-cleanly invariant uniformly.
    let nonexistent = scratch.path().join("does-not-exist");
    let snap = capture_with(&nonexistent, cgroup_tmp.path(), &nonexistent, false);
    assert!(
        snap.threads.is_empty(),
        "nonexistent proc_root must produce empty snapshot; got \
         {} threads — iter_tgids_at must collapse ENOENT to empty",
        snap.threads.len(),
    );
}

/// G2 — tgid directory present but missing the inner
/// `task/` subdirectory. `iter_task_ids_at` returns an
/// empty vec, so the per-tid loop runs zero iterations and
/// the tgid contributes no threads. Pins that the missing
/// `task/` does not crash or fabricate a synthetic tid.
#[test]
fn capture_with_tgid_missing_task_dir_yields_no_threads_for_that_tgid() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // tgid 4242: has `task/` and one tid (live thread).
    // tgid 4243: numeric directory but NO `task/` subdir.
    let live_tgid: i32 = 4242;
    let live_tid: i32 = 101;
    stage_synthetic_proc(
        proc_tmp.path(),
        live_tgid,
        live_tid,
        "live-pcomm",
        "live-comm",
    );

    let bare_tgid: i32 = 4243;
    std::fs::create_dir_all(proc_tmp.path().join(bare_tgid.to_string())).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "tgid 4243 has no `task/` subdir → contributes zero threads; \
         only live tgid 4242's tid should land. got {} threads, expected 1",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tgid, live_tgid as u32);
    assert_eq!(snap.threads[0].tid, live_tid as u32);
}

/// G3 — non-numeric directory entries under proc_root
/// (real procfs has `self`, `thread-self`, `sys`, `kpageflags`,
/// etc.) MUST be filtered by the parse-as-i32 step in
/// `iter_tgids_at`. Pins the filter so a future refactor
/// that loosened it (e.g. accepted any digit-prefix) does
/// not surface kernel pseudo-files as fake tgids.
#[test]
fn capture_with_non_numeric_proc_entries_are_filtered() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    // Stage one valid numeric tgid plus several non-numeric
    // names that mimic real procfs entries.
    let live_tgid: i32 = 5151;
    let live_tid: i32 = 5152;
    stage_synthetic_proc(proc_tmp.path(), live_tgid, live_tid, "real", "real-thread");

    for junk in &["self", "thread-self", "sys", "version", "12abc", "abc"] {
        std::fs::create_dir_all(proc_tmp.path().join(junk)).unwrap();
    }
    // Negative or zero are filtered by `> 0` predicate.
    std::fs::create_dir_all(proc_tmp.path().join("0")).unwrap();
    std::fs::create_dir_all(proc_tmp.path().join("-1")).unwrap();

    // Direct check on the parse filter — pins iter_tgids_at
    // independently of the rest of the pipeline. Without this,
    // a loosened parse that accepted "12" from "12abc" would
    // still produce 1 thread downstream (the "12" dir has no
    // task/ subdir → contributes zero threads regardless), so
    // the snap.threads.len()==1 assertion alone wouldn't catch
    // the regression.
    assert_eq!(
        iter_tgids_at(proc_tmp.path()),
        vec![live_tgid],
        "iter_tgids_at must return only the real numeric tgid; \
         non-numeric and `0`/`-1` entries must be filtered by \
         parse::<i32>().ok() + `> 0` predicates",
    );

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "non-numeric proc_root entries (`self`, `12abc`, etc.) and \
         `0`/`-1` must be filtered by iter_tgids_at; got {} threads, \
         expected 1 (only the real tgid {live_tgid})",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tgid, live_tgid as u32);
}

/// G7 — `capture_pid_with` against a pid whose `/proc/<pid>`
/// directory does not exist must NOT panic. `iter_task_ids_at`
/// returns empty, the loop iterates zero times, and the
/// snapshot's `threads` is empty. Pins that the per-pid
/// capture path tolerates the same exit-mid-capture race the
/// global path does.
#[test]
fn capture_pid_with_nonexistent_pid_produces_empty_snapshot() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    // pid 99999 is not staged — `proc_tmp/99999` does not exist.
    let snap = capture_pid_with(
        proc_tmp.path(),
        cgroup_tmp.path(),
        sys_tmp.path(),
        99999,
        false,
    );
    assert!(
        snap.threads.is_empty(),
        "capture_pid_with against nonexistent pid must produce empty \
         snapshot; got {} threads — iter_task_ids_at must collapse \
         ENOENT to empty",
        snap.threads.len(),
    );
}

/// G4a — corrupt the `stat` file so `parse_stat` returns
/// all-None defaults (write a single non-paren token, so
/// `rfind(')')` returns None and `parse_stat`
/// short-circuits to `StatFields::default()`). With `comm`
/// intact, the ghost-filter clause does NOT fire, so the
/// thread lands with stat-derived fields at zero (nice,
/// start_time, policy, processor, utime, stime) while
/// comm + status + io still populate from their intact
/// files.
#[test]
fn capture_with_corrupt_stat_file_zeroes_stat_fields_only() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6161;
    let tid: i32 = 6162;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Corrupt /proc/<tgid>/task/<tid>/stat — write a single
    // non-paren token so rfind(')') fails.
    let stat_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("stat");
    std::fs::write(&stat_path, "garbage no parens here\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "corrupt stat does not block thread landing — comm + status \
         + io still populate; ghost filter only fires when comm AND \
         start_time are both empty/zero. got {} threads",
        snap.threads.len(),
    );
    let t = &snap.threads[0];
    // stat-derived fields collapse to zero/default.
    assert_eq!(
        t.start_time_clock_ticks, 0,
        "corrupt stat → start_time_clock_ticks default 0; got {}",
        t.start_time_clock_ticks
    );
    use crate::metric_types::{
        Bytes, CategoricalString, ClockTicks, MonotonicCount, OrdinalI32,
    };
    assert_eq!(
        t.nice,
        OrdinalI32(0),
        "corrupt stat → nice default 0; got {}",
        t.nice.0,
    );
    assert_eq!(
        t.policy,
        CategoricalString::from(""),
        "corrupt stat → policy default empty; got {:?}",
        t.policy
    );
    assert_eq!(t.utime_clock_ticks, ClockTicks(0));
    assert_eq!(t.stime_clock_ticks, ClockTicks(0));
    assert_eq!(t.processor, OrdinalI32(0));
    // status-derived fields still populate.
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(42),
        "status file is intact → voluntary_csw still populates"
    );
    // io-derived fields still populate.
    assert_eq!(
        t.rchar,
        Bytes(100),
        "io file is intact → rchar still populates"
    );
}

/// G4b — missing `schedstat` file (kernel without
/// CONFIG_SCHEDSTATS) leaves run_time_ns / wait_time_ns /
/// timeslices at zero. The thread still lands because
/// stat/comm are intact.
#[test]
fn capture_with_missing_schedstat_zeroes_schedstat_fields() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7171;
    let tid: i32 = 7172;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Remove /proc/<tgid>/task/<tid>/schedstat.
    let schedstat_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("schedstat");
    std::fs::remove_file(&schedstat_path).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "thread still lands with schedstat absent"
    );
    let t = &snap.threads[0];
    use crate::metric_types::{MonotonicCount, MonotonicNs};
    assert_eq!(
        t.run_time_ns,
        MonotonicNs(0),
        "missing schedstat → run_time_ns default 0; got {}",
        t.run_time_ns.0
    );
    assert_eq!(t.wait_time_ns, MonotonicNs(0));
    assert_eq!(t.timeslices, MonotonicCount(0));
    // start_time still populates from intact stat.
    assert_eq!(t.start_time_clock_ticks, 555_555);
}

/// G4c — malformed `status` file (random text, no recognized
/// keys) leaves status-derived fields (voluntary_csw,
/// nonvoluntary_csw, state, cpu_affinity) at default. With
/// `use_syscall_affinity=false`, cpu_affinity comes from
/// status only — so this also pins that absent
/// Cpus_allowed_list defaults to empty Vec, NOT to the
/// caller process's real affinity.
#[test]
fn capture_with_corrupt_status_zeroes_status_fields_and_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 8181;
    let tid: i32 = 8182;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    // No `:` separators → split_once(':') returns None for
    // every line → no field populates.
    std::fs::write(&status_path, "totally malformed garbage no colons here\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::MonotonicCount;
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(0),
        "corrupt status → voluntary_csw default 0; got {}",
        t.voluntary_csw.0
    );
    assert_eq!(t.nonvoluntary_csw, MonotonicCount(0));
    assert_eq!(
        t.state, '~',
        "corrupt status → state collapses to '~' (capture-time \
         unwrap_or_else(default_state_char)); got {:?}",
        t.state
    );
    assert!(
        t.cpu_affinity.0.is_empty(),
        "use_syscall_affinity=false + corrupt status → cpu_affinity \
         must be empty Vec, NOT inherit caller's real affinity; got {:?}",
        t.cpu_affinity,
    );
}

/// G4d — missing `io` file (CONFIG_TASK_IO_ACCOUNTING off
/// at kernel build) leaves all 6 byte counters at zero.
/// Pins that the capture continues without io data rather
/// than failing the whole snapshot.
#[test]
fn capture_with_missing_io_zeroes_io_fields() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 9191;
    let tid: i32 = 9192;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let io_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("io");
    std::fs::remove_file(&io_path).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::{Bytes, MonotonicCount};
    assert_eq!(
        t.rchar,
        Bytes(0),
        "missing io → rchar default 0; got {}",
        t.rchar.0,
    );
    assert_eq!(t.wchar, Bytes(0));
    assert_eq!(t.syscr, MonotonicCount(0));
    assert_eq!(t.syscw, MonotonicCount(0));
    assert_eq!(t.read_bytes, Bytes(0));
    assert_eq!(t.write_bytes, Bytes(0));
    assert_eq!(t.cancelled_write_bytes, Bytes(0));
    // stat-derived fields still populate.
    assert_eq!(t.start_time_clock_ticks, 555_555);
}

/// G4e — missing `sched` file leaves every sched-derived
/// field at zero (nr_wakeups family, *_sum, *_max,
/// migrations, ext_enabled). The thread still lands because
/// stat is intact.
#[test]
fn capture_with_missing_sched_zeroes_sched_fields() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1010;
    let tid: i32 = 1011;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let sched_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("sched");
    std::fs::remove_file(&sched_path).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::{MonotonicCount, MonotonicNs, PeakNs};
    assert_eq!(
        t.nr_wakeups,
        MonotonicCount(0),
        "missing sched → nr_wakeups default 0; got {}",
        t.nr_wakeups.0,
    );
    assert_eq!(t.nr_migrations, MonotonicCount(0));
    assert_eq!(t.wait_sum, MonotonicNs(0));
    assert_eq!(t.wait_max, PeakNs(0));
    assert_eq!(t.voluntary_sleep_ns, MonotonicNs(0));
    assert_eq!(t.block_sum, MonotonicNs(0));
    assert_eq!(t.iowait_sum, MonotonicNs(0));
    assert_eq!(t.exec_max, PeakNs(0));
    assert_eq!(t.slice_max, PeakNs(0));
    assert!(
        !t.ext_enabled,
        "missing sched → ext.enabled key absent → ext_enabled false; \
         got {}",
        t.ext_enabled
    );
}

/// G5 — selectively delete EVERY non-comm file under one tid
/// to simulate a partial mid-capture race (readdir saw the
/// dir, then the kernel completed exit cleanup before our
/// per-file reads). With comm intact, the thread still
/// lands but every counter is zero. Pins the absent-=-zero
/// contract under the worst plausible mid-capture race.
#[test]
fn capture_with_partial_mid_capture_race_lands_zero_thread() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1212;
    let tid: i32 = 1213;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "racy-pcomm", "racy-comm");
    let task_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    // Remove every per-tid file EXCEPT comm. comm is the
    // ghost filter's anchor — keeping it preserves the
    // thread's identity so the test exercises the
    // counters-zero path rather than the ghost-drop path.
    for f in &["stat", "schedstat", "status", "io", "sched", "cgroup"] {
        std::fs::remove_file(task_dir.join(f)).unwrap();
    }

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1, "comm intact → thread still lands");
    let t = &snap.threads[0];
    use crate::metric_types::{Bytes, MonotonicCount, MonotonicNs};
    assert_eq!(t.comm, "racy-comm", "comm survives the racy partial reads");
    // Every counter zeros.
    assert_eq!(t.start_time_clock_ticks, 0);
    assert_eq!(t.nr_wakeups, MonotonicCount(0));
    assert_eq!(t.run_time_ns, MonotonicNs(0));
    assert_eq!(t.voluntary_csw, MonotonicCount(0));
    assert_eq!(t.rchar, Bytes(0));
    assert_eq!(t.minflt, MonotonicCount(0));
    assert_eq!(t.cgroup, "");
    assert!(
        snap.cgroup_stats.is_empty(),
        "all threads have empty cgroup → enrichment loop skips → \
         cgroup_stats stays empty",
    );
}

/// G6 — `capture_pid_with` ghost filter: a tid directory
/// under the target pid exists but carries zero readable
/// files (mid-capture exit). `capture_pid_with`'s
/// terminal ghost-filter check — same shape as the global
/// `capture_with` path's filter — must drop the
/// all-Default ThreadState. Pins the per-pid path's filter
/// independently of the global path.
#[test]
fn capture_pid_with_filters_ghost_threads() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1313;
    let live_tid: i32 = 1314;
    let ghost_tid: i32 = 1315;

    stage_synthetic_proc(proc_tmp.path(), tgid, live_tid, "p", "live");

    // Ghost tid: directory exists but empty (no files).
    let ghost_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(ghost_tid.to_string());
    std::fs::create_dir_all(&ghost_dir).unwrap();

    let snap = capture_pid_with(
        proc_tmp.path(),
        cgroup_tmp.path(),
        sys_tmp.path(),
        tgid,
        false,
    );

    assert_eq!(
        snap.threads.len(),
        1,
        "capture_pid_with must filter ghost tid {ghost_tid}; got {} \
         threads, expected 1 (only live tid {live_tid})",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tid, live_tid as u32);
}

/// G8 — malformed `Cpus_allowed_list:` value (a reversed
/// range like `5-3`) routes through `parse_cpu_list` which
/// returns `None`. With `use_syscall_affinity=false`, the
/// capture site has no fallback and `cpu_affinity` stays
/// at the default empty Vec. Pins that a malformed cpulist
/// does NOT crash the parse and does NOT silently fabricate
/// a partial range.
#[test]
fn capture_with_malformed_cpus_allowed_list_yields_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1414;
    let tid: i32 = 1415;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    // Reversed range — parse_cpu_list rejects (returns None).
    let status = "Name:\tfoo\n\
         State:\tR (running)\n\
         voluntary_ctxt_switches:\t1\n\
         nonvoluntary_ctxt_switches:\t1\n\
         Cpus_allowed_list:\t5-3\n";
    std::fs::write(&status_path, status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::MonotonicCount;
    assert!(
        t.cpu_affinity.0.is_empty(),
        "malformed Cpus_allowed_list `5-3` → parse_cpu_list returns \
         None → cpu_affinity defaults to empty Vec (NOT a partial \
         range, NOT the caller's affinity); got {:?}",
        t.cpu_affinity,
    );
    // Other status fields still populate (the malformed
    // line failed only the cpulist arm of parse_status).
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(1),
        "malformed cpulist must NOT corrupt csw fields on the same \
         status file — per-arm Option isolation"
    );
}

/// G11 — huge `Cpus_allowed_list:` range (above the
/// MAX_CPU_RANGE_EXPANSION cap at 64 Ki CPUs) routes
/// through the `parse_cpu_list` cap and returns `None`.
/// Same observable effect as G8 (empty Vec) but pins a
/// distinct adversarial input — a hostile /proc with a
/// `0-4294967295` cpulist must NOT allocate gigabytes.
#[test]
fn capture_with_huge_cpu_range_in_status_yields_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 1515;
    let tid: i32 = 1516;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    // u32::MAX-spanning range — well above the 64 Ki cap;
    // parse_cpu_list rejects without expansion.
    let status = "Cpus_allowed_list:\t0-4294967295\n\
         voluntary_ctxt_switches:\t1\n\
         nonvoluntary_ctxt_switches:\t1\n";
    std::fs::write(&status_path, status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    use crate::metric_types::MonotonicCount;
    assert!(
        t.cpu_affinity.0.is_empty(),
        "huge cpulist range `0-4294967295` exceeds the 64 Ki \
         expansion cap → parse_cpu_list returns None → cpu_affinity \
         empty (NOT a 4-billion-element Vec, NOT a partial range); \
         got {} elements",
        t.cpu_affinity.0.len(),
    );
    // Per-arm isolation: the cap-rejected cpulist must NOT
    // crash the rest of parse_status. csw fields on the same
    // file still populate. Mirrors G8's isolation check.
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(1),
        "huge cpulist rejection must not break csw parsing on the \
         same status file — per-arm Option isolation"
    );
}

/// G9 — non-numeric directory entries under `<proc_root>/<tgid>/task/`
/// MUST be filtered by the parse-as-i32 step in
/// `iter_task_ids_at`. Mirrors G3 for the per-tgid `task/` subdir
/// (G3 covers `<proc_root>` itself). Real procfs has only numeric
/// task entries, but a hostile or malformed test fixture could
/// stage non-numeric names; the filter must drop them rather
/// than surface garbage tids.
#[test]
fn capture_with_non_numeric_task_entries_are_filtered() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();

    let live_tgid: i32 = 8181;
    let live_tid: i32 = 8182;
    stage_synthetic_proc(proc_tmp.path(), live_tgid, live_tid, "real", "real-thread");

    // Stage non-numeric entries alongside the real tid under
    // <tgid>/task/. iter_task_ids_at must filter on parse::<i32>().
    let task_dir = proc_tmp.path().join(live_tgid.to_string()).join("task");
    for junk in &["status", "self", "12abc", "abc"] {
        std::fs::create_dir_all(task_dir.join(junk)).unwrap();
    }
    std::fs::create_dir_all(task_dir.join("0")).unwrap();
    std::fs::create_dir_all(task_dir.join("-1")).unwrap();

    // Direct check on the parse filter — pins iter_task_ids_at
    // independently of the rest of the pipeline.
    assert_eq!(
        iter_task_ids_at(proc_tmp.path(), live_tgid),
        vec![live_tid],
        "iter_task_ids_at must return only the real numeric tid; \
         non-numeric and `0`/`-1` entries must be filtered by \
         parse::<i32>().ok() + `> 0` predicates",
    );

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "non-numeric `task/` entries must be filtered by \
         iter_task_ids_at; got {} threads, expected 1",
        snap.threads.len(),
    );
    assert_eq!(snap.threads[0].tid, live_tid as u32);
}

/// G10 — a tgid emitting a v1-only `cgroup` file (legacy
/// hierarchy entries, no `0::` unified line) lands the thread
/// with `cgroup` defaulting to "". The ghost filter does NOT
/// fire because comm + start_time are intact. The empty cgroup
/// is a legitimate observable signal — `capture_with`'s
/// cgroup_stats enrichment loop skips entries with empty
/// `cgroup` so no synthetic stats land for the missing path.
#[test]
fn capture_with_v1_only_cgroup_yields_empty_cgroup_string() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 9191;
    let tid: i32 = 9192;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    // Overwrite the cgroup file with only legacy v1 lines —
    // parse_cgroup_v2 returns None, read_cgroup_at returns
    // None, ThreadState.cgroup defaults to "".
    let cgroup_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("cgroup");
    let v1_only = "12:cpuset:/legacy/cpuset/path\n\
         5:freezer:/legacy/freezer\n\
         3:blkio:/\n";
    std::fs::write(&cgroup_path, v1_only).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);

    assert_eq!(
        snap.threads.len(),
        1,
        "v1-only cgroup does not block thread landing — comm + \
         start_time are intact, ghost filter does not fire; \
         got {} threads",
        snap.threads.len(),
    );
    let t = &snap.threads[0];
    assert_eq!(
        t.cgroup, "",
        "v1-only cgroup file → parse_cgroup_v2 returns None → \
         ThreadState.cgroup defaults to empty; got {:?}",
        t.cgroup,
    );
    // cgroup_stats enrichment skips empty-cgroup threads. The
    // map must not carry an entry keyed on "" (would otherwise
    // accumulate a meaningless aggregate row in the snapshot).
    assert!(
        !snap.cgroup_stats.contains_key(""),
        "empty-cgroup thread must NOT seed an empty-key entry in \
         cgroup_stats — the enrichment loop's `!is_empty()` guard \
         pins the skip; got keys: {:?}",
        snap.cgroup_stats.keys().collect::<Vec<_>>(),
    );
}

/// `capture_to` propagates write errors through anyhow with the
/// destination path in the context chain so an operator who
/// passed an unwritable target sees the path in the diagnostic
/// rather than a bare I/O error. Pins the `with_context` wrapper
/// at the public-API boundary; without it, the error message
/// loses the path and operators can't tell which target failed.
#[test]
fn capture_to_returns_err_on_unwritable_path() {
    // A path under a directory that does not exist — std::fs::write
    // returns ENOENT for the parent; capture_to's with_context
    // wraps it with the destination path.
    let scratch = tempfile::TempDir::new().unwrap();
    let unwritable = scratch.path().join("missing-dir").join("snap.ctprof.zst");
    let err = capture_to(&unwritable).unwrap_err();
    let chain = format!("{err:#}");
    assert!(
        chain.contains(unwritable.to_string_lossy().as_ref()),
        "error chain must name the unwritable target path; got: {chain}",
    );
}

/// `read_cgroup_stats_at` reads from the path string verbatim;
/// when the path names a cgroup directory that does not exist
/// (the thread's cgroup string was captured but the cgroup has
/// since been rmdir'd, or the cgroup_root differs from the live
/// host), every cpu.stat / memory.current read fails with
/// ENOENT and the resulting `CgroupStats` is all-zero. Pins the
/// "absent = 0" contract for the enrichment loop's stale-string
/// race.
#[test]
fn capture_with_stale_cgroup_path_yields_all_zero_stats() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7373;
    let tid: i32 = 7374;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // stage_synthetic_proc writes "0::/ktstr.slice/worker0" into
    // the cgroup file but does NOT create the matching directory
    // under cgroup_root. The enrichment loop calls
    // read_cgroup_stats_at("/ktstr.slice/worker0"), which
    // resolves to a non-existent dir and returns all-zero stats.

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let stats = snap
        .cgroup_stats
        .get("/ktstr.slice/worker0")
        .expect("non-empty cgroup string must seed the stats map");
    assert_eq!(stats.cpu.usage_usec, 0, "stale cgroup → cpu_usage_usec 0");
    assert_eq!(stats.cpu.nr_throttled, 0, "stale cgroup → nr_throttled 0");
    assert_eq!(
        stats.cpu.throttled_usec, 0,
        "stale cgroup → throttled_usec 0"
    );
    assert_eq!(stats.memory.current, 0, "stale cgroup → memory_current 0");
}

/// `read_cgroup_at` returns `None` when the cgroup file is
/// present but contains only v1 hierarchy lines (no `0::`
/// unified prefix). Pins the "v1-only → None" path of
/// `parse_cgroup_v2` from the file-read entry point — distinct
/// from `parse_cgroup_v2_none_when_only_legacy_present` which
/// pins the parse function in isolation.
#[test]
fn read_cgroup_at_v1_only_cgroup_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 4242;
    let tid: i32 = 4243;
    let task_dir = tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    std::fs::create_dir_all(&task_dir).unwrap();
    let v1_only = "12:cpuset:/legacy/cpuset/path\n\
         5:freezer:/legacy/freezer\n";
    std::fs::write(task_dir.join("cgroup"), v1_only).unwrap();

    assert_eq!(
        read_cgroup_at(tmp.path(), tgid, tid),
        None,
        "v1-only cgroup file → read_cgroup_at returns None (no 0:: line)",
    );

    // Symmetric missing-file branch: no cgroup file → None.
    assert_eq!(
        read_cgroup_at(tmp.path(), tgid, 9999),
        None,
        "missing cgroup file → read_cgroup_at returns None",
    );
}

/// `parse_cgroup_v2` accepts the degenerate "/" root path. A
/// process cgrouped at the unified root emits "0::/" and the
/// parser returns Some("/"). Pins the boundary distinct from
/// `parse_cgroup_v2_empty_path_and_multiple_unified_lines`
/// (which covers "0::" with empty-string-after-prefix); this
/// test pins that "/" alone is treated as a valid path, not
/// folded into the empty-string rejection.
#[test]
fn parse_cgroup_v2_root_only_path_returns_slash() {
    // Single "0::/" line — the trim + non-empty guard accepts
    // "/" as a valid path.
    assert_eq!(parse_cgroup_v2("0::/\n"), Some("/".to_string()));
    // Same with trailing whitespace — trim absorbs it but "/"
    // survives as the post-trim value.
    assert_eq!(parse_cgroup_v2("0::/  \n"), Some("/".to_string()));
    // Mixed alongside legacy v1 lines — unified picks "/".
    let raw = "12:cpuset:/legacy/path\n0::/\n5:freezer:/legacy\n";
    assert_eq!(parse_cgroup_v2(raw), Some("/".to_string()));
}

// ------------------------------------------------------------
// H3 — read_cgroup_stats_at synthetic-tree coverage
// ------------------------------------------------------------

/// Write a cgroup v2-style `cpu.stat` file at
/// `<root>/<relative>/cpu.stat`.
fn write_cpu_stat(root: &Path, relative: &str, contents: &str) {
    let dir = root.join(relative.trim_start_matches('/'));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("cpu.stat"), contents).unwrap();
}

fn write_memory_current(root: &Path, relative: &str, contents: &str) {
    let dir = root.join(relative.trim_start_matches('/'));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("memory.current"), contents).unwrap();
}

/// Case (a): both `cpu.stat` and `memory.current` present →
/// every field populated from the file contents.
#[test]
fn read_cgroup_stats_at_both_files_populate_all_fields() {
    let tmp = tempfile::TempDir::new().unwrap();
    write_cpu_stat(
        tmp.path(),
        "worker",
        "usage_usec 12345\nnr_throttled 7\nthrottled_usec 8\n",
    );
    write_memory_current(tmp.path(), "worker", "9999\n");
    let stats = read_cgroup_stats_at(tmp.path(), "/worker");
    assert_eq!(stats.cpu.usage_usec, 12345);
    assert_eq!(stats.cpu.nr_throttled, 7);
    assert_eq!(stats.cpu.throttled_usec, 8);
    assert_eq!(stats.memory.current, 9999);
}

/// Case (b): `cpu.stat` only → CPU fields populated,
/// `memory_current` defaults to 0.
#[test]
fn read_cgroup_stats_at_cpu_stat_only_memory_defaults_zero() {
    let tmp = tempfile::TempDir::new().unwrap();
    write_cpu_stat(
        tmp.path(),
        "cpu-only",
        "usage_usec 500\nnr_throttled 0\nthrottled_usec 0\n",
    );
    let stats = read_cgroup_stats_at(tmp.path(), "/cpu-only");
    assert_eq!(stats.cpu.usage_usec, 500);
    assert_eq!(stats.cpu.nr_throttled, 0);
    assert_eq!(stats.cpu.throttled_usec, 0);
    assert_eq!(
        stats.memory.current, 0,
        "missing memory.current must collapse to 0, not None",
    );
}

/// Case (c): `memory.current` only → memory populated, CPU
/// fields default to 0.
#[test]
fn read_cgroup_stats_at_memory_only_cpu_defaults_zero() {
    let tmp = tempfile::TempDir::new().unwrap();
    write_memory_current(tmp.path(), "mem-only", "2048\n");
    let stats = read_cgroup_stats_at(tmp.path(), "/mem-only");
    assert_eq!(stats.cpu.usage_usec, 0);
    assert_eq!(stats.cpu.nr_throttled, 0);
    assert_eq!(stats.cpu.throttled_usec, 0);
    assert_eq!(stats.memory.current, 2048);
}

/// Case (d): neither file present → every field zero.
/// Distinct from "returns None or errors" — the documented
/// contract is absent = 0.
#[test]
fn read_cgroup_stats_at_both_files_missing_all_zero() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("empty-cg")).unwrap();
    let stats = read_cgroup_stats_at(tmp.path(), "/empty-cg");
    assert_eq!(stats.cpu.usage_usec, 0);
    assert_eq!(stats.cpu.nr_throttled, 0);
    assert_eq!(stats.cpu.throttled_usec, 0);
    assert_eq!(stats.memory.current, 0);
}

/// Case (e): `cpu.stat` present but missing `nr_throttled`
/// key → that field defaults to 0, OTHER known keys still
/// populate. Proves the parser scans by key rather than
/// positionally.
#[test]
fn read_cgroup_stats_at_cpu_stat_missing_key_defaults_field_zero() {
    let tmp = tempfile::TempDir::new().unwrap();
    // Missing `nr_throttled` entirely; other two keys present.
    write_cpu_stat(
        tmp.path(),
        "partial",
        "usage_usec 999\nthrottled_usec 111\n",
    );
    let stats = read_cgroup_stats_at(tmp.path(), "/partial");
    assert_eq!(stats.cpu.usage_usec, 999);
    assert_eq!(stats.cpu.nr_throttled, 0, "absent key collapses to 0");
    assert_eq!(stats.cpu.throttled_usec, 111);
}

// ------------------------------------------------------------
// H4 — parse_sched every-field coverage + parse fallbacks
// ------------------------------------------------------------

/// Populated `/proc/<tid>/sched` with every field
/// parse_sched recognises. Ordering mixed (sync before
/// local) so the test doesn't pin a single-pass scan order
/// that the helper doesn't actually promise. Integer-only
/// PN_SCHEDSTAT values (no fractional part) parse via the
/// no-dot branch of `parsed_ns_from_dotted` — interpreted
/// as plain ns counts — so the values pass through
/// unchanged. The fixture also includes the dead-counter
/// lines (`nr_wakeups_idle`, `nr_migrations_cold`,
/// `nr_wakeups_passive`); the parser silently drops them
/// since they were dropped from the registry.
#[test]
fn parse_sched_populates_all_known_fields() {
    let raw = "\
         se.statistics.nr_wakeups                       :         11\n\
         se.statistics.nr_wakeups_sync                  :          2\n\
         se.statistics.nr_wakeups_local                 :          8\n\
         se.statistics.nr_wakeups_migrate               :          1\n\
         se.statistics.nr_wakeups_remote                :          3\n\
         se.statistics.nr_wakeups_idle                  :          4\n\
         se.statistics.nr_wakeups_affine                :         12\n\
         se.statistics.nr_wakeups_affine_attempts       :         20\n\
         nr_migrations                                  :          9\n\
         se.statistics.nr_migrations_cold               :          5\n\
         se.statistics.nr_forced_migrations             :          7\n\
         se.statistics.nr_failed_migrations_affine      :          1\n\
         se.statistics.nr_failed_migrations_running     :          2\n\
         se.statistics.nr_failed_migrations_hot         :          3\n\
         wait_sum                                       :       500\n\
         wait_count                                     :         15\n\
         se.statistics.wait_max                         :       250\n\
         sum_sleep_runtime                              :       320\n\
         se.statistics.sleep_max                        :       180\n\
         sum_block_runtime                              :       110\n\
         se.statistics.block_max                        :        60\n\
         iowait_sum                                     :         77\n\
         iowait_count                                   :         18\n\
         se.statistics.exec_max                         :        90\n\
         se.statistics.slice_max                        :       400\n\
         ext.enabled                                    :          1\n";
    let s = parse_sched(raw, &mut None);
    assert_eq!(s.nr_wakeups, Some(11));
    assert_eq!(s.nr_wakeups_local, Some(8));
    assert_eq!(s.nr_wakeups_remote, Some(3));
    assert_eq!(s.nr_wakeups_sync, Some(2));
    assert_eq!(s.nr_wakeups_migrate, Some(1));
    assert_eq!(s.nr_wakeups_affine, Some(12));
    assert_eq!(s.nr_wakeups_affine_attempts, Some(20));
    assert_eq!(s.nr_migrations, Some(9));
    assert_eq!(s.nr_forced_migrations, Some(7));
    assert_eq!(s.nr_failed_migrations_affine, Some(1));
    assert_eq!(s.nr_failed_migrations_running, Some(2));
    assert_eq!(s.nr_failed_migrations_hot, Some(3));
    assert_eq!(s.wait_sum, Some(500));
    assert_eq!(s.wait_count, Some(15));
    assert_eq!(s.wait_max, Some(250));
    assert_eq!(
        s.sleep_sum,
        Some(320),
        "sleep_sum (raw kernel sum_sleep_runtime) reads through \
         SchedFields; the capture site subtracts block_sum to \
         produce ThreadState::voluntary_sleep_ns",
    );
    assert_eq!(s.sleep_max, Some(180));
    assert_eq!(
        s.block_sum,
        Some(110),
        "block_sum reads the kernel's `sum_block_runtime` key",
    );
    assert_eq!(s.block_max, Some(60));
    assert_eq!(s.iowait_sum, Some(77));
    assert_eq!(s.iowait_count, Some(18));
    assert_eq!(s.exec_max, Some(90));
    assert_eq!(s.slice_max, Some(400));
    assert_eq!(
        s.ext_enabled,
        Some(true),
        "ext.enabled = 1 → Some(true) — full-key match required \
         because rsplit('.') would yield `enabled` and collide \
         with any future field of that name",
    );
}

/// `ext.enabled = 0` lands as `Some(false)` (CONFIG_SCHED_CLASS_EXT
/// kernel where the task is NOT on sched_ext); absent line lands
/// as `None` and the capture-site `unwrap_or(false)` collapses to
/// the absent default. Pins the bool round-trip.
#[test]
fn parse_sched_ext_enabled_zero_and_absent() {
    let zero = parse_sched("ext.enabled : 0\n", &mut None);
    assert_eq!(zero.ext_enabled, Some(false));
    let absent = parse_sched("nr_wakeups : 1\n", &mut None);
    assert_eq!(absent.ext_enabled, None);
}

/// Full-key match on `ext.enabled` MUST take precedence over the
/// rsplit-on-dot fallback. A line like `foo.enabled : 1` would
/// otherwise route through rsplit to `enabled`, collide with
/// `ext.enabled`, and incorrectly populate the bool. Pins the
/// guard.
#[test]
fn parse_sched_ext_enabled_no_collision_via_rsplit() {
    // foo.enabled is not a real kernel key, but proves the
    // full-key gate: rsplit yields `enabled`, but the match
    // arm only fires on the exact key `ext.enabled`.
    let s = parse_sched("foo.enabled : 1\n", &mut None);
    assert_eq!(s.ext_enabled, None);
}

/// Dotted PN_SCHEDSTAT fractional values reconstruct full ns
/// via `ms * 1_000_000 + zero-right-padded ns_remainder`.
/// Pins the helper for varying fractional widths (1, 2, and
/// 3 digits past the dot — all zero-pad to 6).
#[test]
fn parse_sched_fractional_fields_reconstruct_ns() {
    let raw = "\
         wait_sum                                       :    1234.5\n\
         sum_sleep_runtime                              :     678.9\n\
         sum_block_runtime                              :      42.1\n\
         iowait_sum                                     :       7.999\n";
    let s = parse_sched(raw, &mut None);
    // 1234.5 → .5 pads to .500000 (=500_000) + 1234ms = 1_234_500_000 ns
    assert_eq!(s.wait_sum, Some(1_234_500_000));
    // 678.9 → .9 pads to .900000 (=900_000) + 678ms = 678_900_000 ns
    assert_eq!(s.sleep_sum, Some(678_900_000));
    // 42.1 → .1 pads to .100000 (=100_000) + 42ms = 42_100_000 ns
    assert_eq!(s.block_sum, Some(42_100_000));
    // 7.999 → .999 pads to .999000 (=999_000) + 7ms = 7_999_000 ns
    assert_eq!(s.iowait_sum, Some(7_999_000));
}

/// `parsed_ns_from_dotted` rejects negative integer parts —
/// `u64` parse fails on `-5`. The capture site
/// `unwrap_or(0)`s these into the absent-counter zero per the
/// best-effort capture contract, so a kernel that emits a
/// negative SPLIT_NS (rare; can happen for clock skew on
/// suspend/resume) does not pollute downstream metrics. The
/// tally arg is `&mut None` here — the no-tally branch must
/// still produce None for the negative case so synthetic-tree
/// tests that don't carry a tally still observe the
/// pre-tally semantics.
#[test]
fn parse_sched_negative_value_returns_none() {
    let raw = "wait_sum                                       :   -5.0\n";
    let s = parse_sched(raw, &mut None);
    assert_eq!(
        s.wait_sum, None,
        "negative ms part fails u64 parse → None; downstream \
         unwrap_or(0) collapses this to absent-counter zero",
    );
}

/// Negative dotted-ns value records into the [`ParseTally`]
/// when one is supplied — pinning the tally-bump path so a
/// regression that drops the per-line negative detection
/// surfaces here rather than silently zeroing schedstat
/// fields. Multiple negative lines bump independently;
/// non-negative lines on the same parse pass do NOT bump.
#[test]
fn parse_sched_negative_value_records_into_tally() {
    let raw = "wait_sum                                       :   -5.0\n\
               sum_sleep_runtime                              :   12.5\n\
               sum_block_runtime                              :  -10.0\n";
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    let s = parse_sched(raw, &mut tally_opt);
    assert_eq!(
        s.wait_sum, None,
        "negative wait_sum still reads None — the tally records \
         but does not change the per-field outcome",
    );
    assert_eq!(
        s.sleep_sum,
        Some(12_500_000),
        "non-negative neighbor still parses normally",
    );
    assert_eq!(s.block_sum, None, "negative block_sum reads None");
    // 2 negative dotted values landed in pending. Commit
    // through the Option-wrapped tally (NLL: while `tally_opt`
    // holds &mut tally, direct access to `tally` would be
    // a borrow-check error).
    tally_opt.as_mut().unwrap().commit_pending();
    // After this point, `tally_opt` is no longer used — NLL
    // releases the inner borrow so `tally` is reborrowable.
    let summary = tally.to_public();
    assert_eq!(
        summary.negative_dotted_values, 2,
        "two negative dotted lines bumped the per-snapshot \
         negative_dotted_values counter; non-negative neighbor \
         did not contribute",
    );
}

/// Ghost-filter discipline for the negative-dotted tally: a
/// tid whose pending bumps are unwound via
/// [`ParseTally::discard_pending`] must not contribute to
/// the per-snapshot
/// [`CtprofParseSummary::negative_dotted_values`]. Mirrors
/// the read-failure tally's discard semantics so the two
/// tally families stay symmetric under the ghost-filter
/// path.
#[test]
fn parse_tally_negative_dotted_discard_pending_unwinds_bumps() {
    let raw = "wait_sum :   -5.0\n";
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    let _ = parse_sched(raw, &mut tally_opt);
    // Pending bump landed; discard_pending must unwind it
    // before commit so the ghost-filtered tid leaves no trace
    // in the public surface. Same NLL-through-Option pattern
    // as `parse_sched_negative_value_records_into_tally`.
    tally_opt.as_mut().unwrap().discard_pending();
    let summary = tally.to_public();
    assert_eq!(
        summary.negative_dotted_values, 0,
        "discard_pending must unwind the negative-dotted \
         pending bump so a ghost-filtered tid does not \
         pollute the per-snapshot tally",
    );
}

/// Tally accumulates across multiple commits (multi-tid path
/// — production captures invoke `parse_sched` once per tid
/// and `commit_pending` between them). Pin that negative
/// bumps from a SECOND tid land additively on top of the
/// first tid's contribution rather than replacing it. Total
/// after two commits is the sum of pending counts at each
/// commit.
#[test]
fn parse_tally_negative_dotted_accumulates_across_commits() {
    let raw_a = "wait_sum : -1.0\n";
    let raw_b = "wait_sum   : -2.0\n\
                 sleep_max  : -3.0\n";
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    let _ = parse_sched(raw_a, &mut tally_opt);
    // Commit tid A's 1 pending bump.
    tally_opt.as_mut().unwrap().commit_pending();
    // Now parse tid B's 2 pending bumps.
    let _ = parse_sched(raw_b, &mut tally_opt);
    tally_opt.as_mut().unwrap().commit_pending();
    let summary = tally.to_public();
    assert_eq!(
        summary.negative_dotted_values, 3,
        "1 commit + 2 commit = 3 total — multi-tid commits \
         must add, not overwrite. got {}",
        summary.negative_dotted_values,
    );
}

/// All-positive dotted input MUST NOT bump the
/// `negative_dotted_values` counter. Pins that the negative
/// detection is gated on the leading `-`, not triggered by
/// any other parse path. Without this, a regression that
/// always-bumped (e.g. moving the bump out of the Err arm)
/// would let a clean host emit a non-zero count.
#[test]
fn parse_tally_negative_dotted_zero_for_positive_only_input() {
    let raw = "wait_sum            : 100.5\n\
               sum_sleep_runtime   : 200\n\
               sum_block_runtime   : 0.999\n\
               wait_max            : 0\n\
               exec_max            : 7\n";
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    let _ = parse_sched(raw, &mut tally_opt);
    tally_opt.as_mut().unwrap().commit_pending();
    let summary = tally.to_public();
    assert_eq!(
        summary.negative_dotted_values, 0,
        "all-positive dotted input must not bump the \
         negative-dotted tally; got {}",
        summary.negative_dotted_values,
    );
}

/// Sub-millisecond negative SPLIT_NS shape: kernel emits
/// `0.-NNN` when the integer part is `(x / 1_000_000)` for
/// `x` in `(-1_000_000, 0)` — `%Ld` yields `0` (no sign
/// because integer division of a negative by 1M lands at
/// `0` not `-1`) and `%06ld` carries the negative
/// remainder. Without the fractional-side detection in
/// [`parsed_ns_from_dotted`] the integer-only check would
/// miss this shape entirely. Pin both the parser-level
/// detection and the tally-bump path.
#[test]
fn parsed_ns_from_dotted_sub_millisecond_negative_detected() {
    // Direct parser-level shape.
    assert_eq!(
        parsed_ns_from_dotted("0.-000500"),
        Err(ParseDottedNs::Negative),
        "0.-NNN shape (sub-ms negative SPLIT_NS) MUST route \
         through Negative — most schedstat negatives land \
         sub-millisecond and would otherwise slip through",
    );
    assert_eq!(
        parsed_ns_from_dotted("0.-1"),
        Err(ParseDottedNs::Negative),
        "single-digit sub-ms negative shape detected",
    );
    // End-to-end through parse_sched + tally.
    let raw = "wait_sum : 0.-000500\n\
               sleep_max : 0.-1\n";
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    let s = parse_sched(raw, &mut tally_opt);
    assert_eq!(
        s.wait_sum, None,
        "sub-ms negative wait_sum collapses to None",
    );
    assert_eq!(
        s.sleep_max, None,
        "sub-ms negative sleep_max collapses to None",
    );
    tally_opt.as_mut().unwrap().commit_pending();
    let summary = tally.to_public();
    assert_eq!(
        summary.negative_dotted_values, 2,
        "two sub-ms negatives both bump the tally — pins \
         that the integer-only detection is NOT enough on \
         its own",
    );
}

/// Bare-integer (no-dot) negative value is also recorded —
/// the kernel's PN_SCHEDSTAT format always emits the dotted
/// form, but the `parsed_ns_from_dotted` function's bare
/// branch is exercised by the `slice` (P_SCHEDSTAT, no dot)
/// arm and by graceful degradation against fixtures that
/// drop the fractional part. A bare `-5` lands the same
/// `Negative` arm as `-5.0` so the tally treats both
/// identically.
///
/// `wait_sum` itself is dotted-only in real kernel output,
/// but `parsed_ns_from_dotted`'s bare-integer fallback is
/// reachable via test fixtures that drop the dot — pinning
/// the bare-branch negative detection ensures the two
/// branches stay symmetric.
#[test]
fn parsed_ns_from_dotted_negative_bare_branch_records() {
    // Direct call into the parser: bare-integer negative.
    assert_eq!(
        parsed_ns_from_dotted("-5"),
        Err(ParseDottedNs::Negative),
        "bare-integer negative routes through Negative",
    );
    // Dotted negative.
    assert_eq!(
        parsed_ns_from_dotted("-5.0"),
        Err(ParseDottedNs::Negative),
        "dotted negative routes through Negative",
    );
    // Non-numeric malformed.
    assert_eq!(
        parsed_ns_from_dotted("garbage"),
        Err(ParseDottedNs::Malformed),
        "non-numeric input routes through Malformed, not \
         Negative — the tally must NOT bump on garbage",
    );
    assert_eq!(
        parsed_ns_from_dotted("garbage.5"),
        Err(ParseDottedNs::Malformed),
        "non-numeric integer part with fractional routes \
         through Malformed",
    );
    assert_eq!(
        parsed_ns_from_dotted(""),
        Err(ParseDottedNs::Malformed),
        "empty input routes through Malformed",
    );
    assert_eq!(
        parsed_ns_from_dotted("5"),
        Ok(5),
        "bare positive integer parses",
    );
    assert_eq!(
        parsed_ns_from_dotted("5.500"),
        Ok(5_500_000),
        "positive dotted parses normally",
    );
}

/// Bare-key names (no `se.statistics.` prefix) must still
/// populate — some kernels emit `nr_wakeups : N` at the top
/// level. The parser's `rsplit('.').next()` treats a no-dot
/// string as the whole string. Coverage spans the wakeup
/// family, the migrations counter, and one of the *_max ns
/// fields, to prove the bare-key path lights up every parser
/// arm shape (parsed_u64 + parsed_ns_from_dotted).
#[test]
fn parse_sched_bare_key_names_populate_same_fields() {
    let raw = "\
         nr_wakeups                                     :         11\n\
         nr_wakeups_local                               :          8\n\
         nr_wakeups_remote                              :          3\n\
         nr_wakeups_sync                                :          2\n\
         nr_wakeups_migrate                             :          1\n\
         nr_migrations                                  :         42\n\
         wait_max                                       :     999.5\n";
    let s = parse_sched(raw, &mut None);
    assert_eq!(s.nr_wakeups, Some(11));
    assert_eq!(s.nr_wakeups_local, Some(8));
    assert_eq!(s.nr_wakeups_remote, Some(3));
    assert_eq!(s.nr_wakeups_sync, Some(2));
    assert_eq!(s.nr_wakeups_migrate, Some(1));
    assert_eq!(
        s.nr_migrations,
        Some(42),
        "bare-key `nr_migrations` must populate via \
         rsplit('.').next() returning the whole no-dot string",
    );
    assert_eq!(
        s.wait_max,
        Some(999_500_000),
        "bare-key `wait_max` must populate via the \
         parsed_ns_from_dotted path; 999.5 → 999_500_000 ns",
    );
}

/// Future `stats.` or other prefix variants must also
/// populate — the parser matches on the LAST dot-delimited
/// segment, so any enclosing prefix is ignored by design.
#[test]
fn parse_sched_alternative_prefix_populates_same_fields() {
    let raw = "\
         stats.nr_wakeups                               :         42\n\
         some.other.prefix.nr_migrations                :          9\n";
    let s = parse_sched(raw, &mut None);
    assert_eq!(s.nr_wakeups, Some(42));
    assert_eq!(s.nr_migrations, Some(9));
}

/// Unknown keys don't corrupt populated fields — important
/// because kernel versions add new lines frequently and the
/// parser must skip them rather than mis-route.
#[test]
fn parse_sched_unknown_keys_are_ignored() {
    let raw = "\
         nr_wakeups                                     :         11\n\
         fictional_new_kernel_stat                      :       9999\n\
         nr_migrations                                  :          9\n";
    let s = parse_sched(raw, &mut None);
    assert_eq!(s.nr_wakeups, Some(11));
    assert_eq!(s.nr_migrations, Some(9));
}

// ------------------------------------------------------------
// H5 — ProbeSummary discipline
//
// The capture pipeline tallies every per-tgid attach result and
// every per-tid probe_thread result into a [`ProbeSummary`]
// before emitting one info-level line per snapshot. The tests
// below pin the summary's accounting + EPERM-hint policy
// independently of any real ptrace dispatch — a regression that
// mis-categorised a tag, dropped the dominant-tag tiebreak,
// or flipped the ptrace-dominates threshold lands here loudly.
// ------------------------------------------------------------

/// Construct a populated `ProbeSummary` for unit-test cases.
/// Lifts the otherwise-repetitive default-then-mutate pattern
/// out of every test (clippy's `field_reassign_with_default`
/// flags it; using a constructor keeps the tests terse).
fn make_summary(
    failed: u64,
    attach: &[(&'static str, u64)],
    probe: &[(&'static str, u64)],
) -> ProbeSummary {
    ProbeSummary {
        failed,
        attach_tag_counts: attach.iter().copied().collect(),
        probe_tag_counts: probe.iter().copied().collect(),
        ..ProbeSummary::default()
    }
}

#[test]
fn probe_summary_dominant_tag_picks_highest_count() {
    // dwarf-parse-failure is an ACTIONABLE attach tag (it
    // signals a stripped binary worth surfacing), so it
    // survives the `jemalloc-not-found / readlink-failure`
    // filter in `dominant_tag` and competes against the probe
    // side on raw count.
    let s = make_summary(6, &[("dwarf-parse-failure", 5)], &[("ptrace-seize", 1)]);
    assert_eq!(s.dominant_tag(), Some("dwarf-parse-failure"));
}

/// `dominant_tag` filters `jemalloc-not-found` and
/// `readlink-failure` out of the attach side BEFORE the
/// max-by-count step. Both are the expected outcome on the
/// bulk of system processes (most tgids are not jemalloc-
/// linked; short-lived tgids race readlink mid-walk), so
/// surfacing them as the dominant tag would drown actionable
/// signal under benign noise. This pin proves the filter
/// engages even when the filtered tag has the highest raw
/// count: 100 jemalloc-not-found events lose to a single
/// ptrace-seize because the former does not enter the
/// comparison at all.
///
/// Also covers `readlink-failure` symmetrically — both
/// non-actionable attach tags are filtered, only one is in
/// the production code's matches! arm but the test doubles
/// up to keep the contract from quietly degrading to "only
/// jemalloc-not-found is filtered."
#[test]
fn probe_summary_dominant_tag_filters_non_actionable_attach_tags() {
    // jemalloc-not-found dominates by count but is filtered.
    let s = make_summary(101, &[("jemalloc-not-found", 100)], &[("ptrace-seize", 1)]);
    assert_eq!(
        s.dominant_tag(),
        Some("ptrace-seize"),
        "jemalloc-not-found must be filtered out even at \
         100x the count of an actionable tag",
    );
    // readlink-failure dominates by count but is filtered.
    let s = make_summary(101, &[("readlink-failure", 100)], &[("get-regset", 1)]);
    assert_eq!(
        s.dominant_tag(),
        Some("get-regset"),
        "readlink-failure must be filtered out even at \
         100x the count of an actionable tag",
    );
    // Both filtered tags present together: still filtered;
    // the actionable probe tag wins.
    let s = make_summary(
        201,
        &[("jemalloc-not-found", 100), ("readlink-failure", 100)],
        &[("waitpid", 1)],
    );
    assert_eq!(
        s.dominant_tag(),
        Some("waitpid"),
        "both filtered attach tags together must NOT push their \
         aggregate above an actionable probe tag",
    );
    // Only filtered tags, no actionable counterparts: None
    // (the filter removes them, the chain is empty).
    let s = make_summary(5, &[("jemalloc-not-found", 5)], &[]);
    assert_eq!(
        s.dominant_tag(),
        None,
        "only-filtered-tags case must produce None, not the \
         filtered tag itself",
    );
}

#[test]
fn probe_summary_dominant_tag_breaks_ties_reverse_alphabetically() {
    // Two tags tied at count=2 — the tiebreak's secondary key
    // is `b.0.cmp(a.0)` (note the flip), so the alphabetically-
    // EARLIER tag wins. With "ptrace-seize" vs
    // "dwarf-parse-failure", "dwarf-parse-failure" precedes
    // "ptrace-seize" lexicographically, so it wins. This
    // "reverse-alphabetical" framing matches how the
    // `dominant_tag` doc describes the comparator.
    let s = make_summary(4, &[("ptrace-seize", 2)], &[("dwarf-parse-failure", 2)]);
    assert_eq!(s.dominant_tag(), Some("dwarf-parse-failure"));
}

#[test]
fn probe_summary_ptrace_dominates_when_half_of_failures() {
    // 3/6 failures are ptrace-attach — meets the half
    // threshold so the EPERM hint engages.
    let s = make_summary(6, &[], &[("ptrace-seize", 3), ("waitpid", 3)]);
    assert!(s.ptrace_dominates());
}

#[test]
fn probe_summary_ptrace_does_not_dominate_when_below_half() {
    let s = make_summary(6, &[], &[("ptrace-seize", 2), ("waitpid", 4)]);
    assert!(!s.ptrace_dominates());
}

#[test]
fn probe_summary_no_failures_no_dominant_tag() {
    let s = ProbeSummary::default();
    assert!(!s.ptrace_dominates());
    assert_eq!(s.dominant_tag(), None);
}

/// EPERM remediation hint references `$(which ktstr)` rather
/// than a hardcoded path — pins the wording so a future drift
/// to a fixed install path lands here loudly.
#[test]
fn ptrace_eperm_hint_uses_which_ktstr() {
    assert!(
        PTRACE_EPERM_HINT.contains("$(which ktstr)"),
        "EPERM hint must use $(which ktstr) for portability, got: {PTRACE_EPERM_HINT}",
    );
    assert!(PTRACE_EPERM_HINT.contains("cap_sys_ptrace"));
    assert!(PTRACE_EPERM_HINT.contains("yama.ptrace_scope"));
}

/// `to_public()` carries every counter through verbatim and
/// projects `dominant_tag` to `dominant_failure` as the owned
/// tag string. Pins the public surface contract so a refactor
/// that drops a counter or rewires the projection lands here.
#[test]
fn to_public_carries_counters_and_dominant_tag() {
    let mut s = make_summary(3, &[("dwarf-parse-failure", 2)], &[("ptrace-seize", 1)]);
    s.tgids_walked = 10;
    s.jemalloc_detected = 5;
    s.probed_ok = 4;

    let public = s.to_public();
    assert_eq!(public.tgids_walked, 10);
    assert_eq!(public.jemalloc_detected, 5);
    assert_eq!(public.probed_ok, 4);
    assert_eq!(public.failed, 3);
    assert_eq!(
        public.dominant_failure.as_deref(),
        Some("dwarf-parse-failure"),
        "dominant_tag picks the highest-count actionable tag, \
         projected as an owned String",
    );
    // 1 ptrace-seize out of 3 failed (33%) is below the 50%
    // hint-trigger threshold → privilege_dominant is false.
    assert!(
        !public.privilege_dominant,
        "ptrace 1/3 < 50% → privilege_dominant false",
    );
}

/// Zero-failure summary projects to `dominant_failure: None` —
/// the absence-of-failure case must surface as None, not an
/// empty string. Mirrors the internal `dominant_tag` returning
/// None when no actionable tags remain after the
/// non-actionable filter (the fixture seeds
/// `jemalloc-not-found`, which `dominant_tag` filters out).
/// `privilege_dominant` must also be false (no failures to
/// dominate).
#[test]
fn to_public_dominant_failure_is_none_when_no_failures() {
    let s = make_summary(0, &[("jemalloc-not-found", 12)], &[]);
    let public = s.to_public();
    assert_eq!(public.failed, 0);
    assert!(
        public.dominant_failure.is_none(),
        "no actionable failures means dominant_failure is None; \
         got {:?}",
        public.dominant_failure,
    );
    assert!(
        !public.privilege_dominant,
        "no failures means privilege_dominant is false",
    );
}

/// Privilege-dominated snapshot projects
/// `privilege_dominant: true` so a downstream consumer can
/// reproduce the EPERM-hint trigger condition without parsing
/// the tracing summary. Mirrors the
/// `summary_emits_privilege_hint_when_ptrace_dominates`
/// emission test below.
#[test]
fn to_public_privilege_dominant_when_ptrace_crosses_threshold() {
    // 4 failed total, all ptrace-seize → 100% ≥ 50% → true.
    let s = make_summary(4, &[], &[("ptrace-seize", 4)]);
    let public = s.to_public();
    assert_eq!(public.failed, 4);
    assert!(
        public.privilege_dominant,
        "ptrace 4/4 ≥ 50% → privilege_dominant true",
    );

    // 2 ptrace + 2 dwarf = 50% / 50% → boundary
    // (`total_ptrace * 2 >= self.failed` accepts equality).
    let s = make_summary(4, &[("dwarf-parse-failure", 2)], &[("ptrace-seize", 2)]);
    let public = s.to_public();
    assert!(
        public.privilege_dominant,
        "ptrace 2/4 = 50% boundary → privilege_dominant true (>= threshold)",
    );

    // 1 ptrace + 3 dwarf = 25% < 50% → false.
    let s = make_summary(4, &[("dwarf-parse-failure", 3)], &[("ptrace-seize", 1)]);
    let public = s.to_public();
    assert!(
        !public.privilege_dominant,
        "ptrace 1/4 < 50% → privilege_dominant false",
    );
}

/// `privilege_dominant` covers the full ptrace tag set, the
/// smallest-`failed` corners of the threshold, and the default
/// shape of the public surface. Pins:
///
/// 1. `ptrace-interrupt` alone trips the threshold — proves the
///    `matches!` arm in `ptrace_dominates` covers both tags, not
///    just `ptrace-seize`.
/// 2. `dwarf-parse-failure` (2) plus split ptrace tags
///    (`ptrace-seize` 1 + `ptrace-interrupt` 1) out of 4 failed —
///    proves `privilege_dominant` and `dominant_failure` are
///    independent reductions and can DIVERGE: summed ptrace
///    crosses the 50% gate (`privilege_dominant: true`) while
///    `dominant_failure` names the non-ptrace tag that won the
///    single-tag plurality (`dwarf-parse-failure`).
/// 3. `failed == 1` with one ptrace tag is the smallest input
///    that flips the gate true (1*2 >= 1).
/// 4. `failed == 1` with one non-ptrace tag is the smallest
///    input that keeps the gate false (0*2 < 1) — pins that
///    `total_ptrace == 0` keeps the gate false even when
///    `failed > 0`.
/// 5. `CtprofProbeSummary::default()` has
///    `privilege_dominant: false` — pins
///    `CtprofProbeSummary::default()` for callers that may
///    use struct-update syntax.
/// 6. ptrace wins the single-tag plurality but stays below the
///    50% threshold — the converse of bullet 2: `dominant_failure`
///    names a ptrace tag while `privilege_dominant` is `false`.
///    Pins the converse direction of the independence claim.
#[test]
fn to_public_privilege_dominant_ptrace_interrupt_and_edge_cases() {
    // 1. ptrace-interrupt alone: 2/2 = 100% ≥ 50% → true.
    let s = make_summary(2, &[], &[("ptrace-interrupt", 2)]);
    let public = s.to_public();
    assert!(
        public.privilege_dominant,
        "ptrace-interrupt 2/2 ≥ 50% → privilege_dominant true \
         (matches! arm covers ptrace-interrupt as well as ptrace-seize)",
    );

    // 2. divergence: summed ptrace tags trip the privilege gate
    //    while a non-ptrace tag wins the single-tag plurality.
    //    dwarf-parse-failure (2) + ptrace-seize (1) + ptrace-interrupt (1)
    //    out of 4 failed: total_ptrace = 2, 2*2 = 4 >= 4 →
    //    privilege_dominant true; dominant_tag picks
    //    dwarf-parse-failure as the highest single-tag count (2).
    //    Pins that the two fields reduce independently.
    let s = make_summary(
        4,
        &[("dwarf-parse-failure", 2)],
        &[("ptrace-seize", 1), ("ptrace-interrupt", 1)],
    );
    let public = s.to_public();
    assert!(
        public.privilege_dominant,
        "summed ptrace 2/4 ≥ 50% → privilege_dominant true",
    );
    assert_eq!(
        public.dominant_failure.as_deref(),
        Some("dwarf-parse-failure"),
        "dominant_failure names the non-ptrace tag that won the \
         single-tag plurality while privilege_dominant is true — \
         proves the two fields are independent",
    );

    // 3. smallest true: failed == 1 with one ptrace tag.
    let s = make_summary(1, &[], &[("ptrace-seize", 1)]);
    let public = s.to_public();
    assert!(
        public.privilege_dominant,
        "ptrace 1/1 ≥ 50% → privilege_dominant true at the \
         smallest-failed boundary",
    );

    // 4. smallest false: failed == 1 with no ptrace tag. Guards
    //    that `total_ptrace == 0` keeps the gate false even when
    //    `failed > 0`.
    let s = make_summary(1, &[("dwarf-parse-failure", 1)], &[]);
    let public = s.to_public();
    assert!(
        !public.privilege_dominant,
        "no ptrace tags with failed == 1 → privilege_dominant \
         false (total_ptrace == 0 keeps the gate closed)",
    );

    // 5. default invariant: a freshly-defaulted summary must
    //    not claim privilege dominance.
    assert!(
        !CtprofProbeSummary::default().privilege_dominant,
        "CtprofProbeSummary::default().privilege_dominant \
         must be false",
    );

    // 6. converse: ptrace wins the per-tag plurality but stays
    //    below the 50% threshold → privilege_dominant false while
    //    dominant_failure names the ptrace tag.
    let s = make_summary(
        10,
        &[("dwarf-parse-failure", 3), ("jemalloc-in-dso", 3)],
        &[("ptrace-seize", 4)],
    );
    let public = s.to_public();
    assert!(
        !public.privilege_dominant,
        "ptrace 4/10 < 50% → privilege_dominant false",
    );
    assert_eq!(
        public.dominant_failure.as_deref(),
        Some("ptrace-seize"),
        "dominant_failure names a ptrace tag while privilege_dominant \
         is false — converse of the independence claim",
    );
}

/// `remediation_hint()` returns `Some` exactly when
/// `privilege_dominant` is true, and the returned text matches
/// the same `PTRACE_EPERM_HINT` constant the emission path
/// prints — so a downstream consumer surfaces the same fix-it
/// message the operator-facing tracing summary does. Pins both
/// the gate semantics and the text-equality contract.
#[test]
fn remediation_hint_returns_some_iff_privilege_dominant() {
    // privilege_dominant=true → Some(PTRACE_EPERM_HINT).
    let ps = CtprofProbeSummary {
        privilege_dominant: true,
        ..Default::default()
    };
    assert_eq!(
        ps.remediation_hint(),
        Some(PTRACE_EPERM_HINT),
        "privilege_dominant=true must surface the same hint text \
         the tracing summary prints",
    );

    // privilege_dominant=false → None.
    let ps = CtprofProbeSummary::default();
    assert!(
        !ps.privilege_dominant,
        "default privilege_dominant must be false (sanity)",
    );
    assert_eq!(
        ps.remediation_hint(),
        None,
        "privilege_dominant=false → remediation_hint returns None",
    );
}

// ------------------------------------------------------------
// Summary-line emission discipline (tracing assertions)
//
// emit_probe_summary is the single source of truth for the
// operator-facing per-snapshot summary. The tests below run
// under `#[traced_test]` so the emitted `tracing::info!` /
// `tracing::warn!` events are captured into an in-memory
// buffer queryable via `logs_contain`. Without these, a
// refactor that silently dropped the dominant-tag clause or
// the EPERM hint would be invisible — the structural unit
// tests above pin the helpers that feed the summary, but
// only an emission test pins what the operator actually
// reads.
// ------------------------------------------------------------

/// Zero-failure snapshot emits a clean summary line — no
/// failure-class clause, no privilege hint. Pins the "happy
/// path" shape so a future refactor that always-appended a
/// hint would surface here.
///
/// Test fn names deliberately avoid the substrings asserted
/// against (e.g. "dominant", "hint") because
/// `tracing-test`'s `logs_contain` matches across the entire
/// captured frame INCLUDING the span (which is the test fn
/// name). The terse `summary_emits_*` naming keeps the span
/// text disjoint from the assertions.
#[traced_test]
#[test]
fn summary_emits_clean_line_when_no_failures() {
    let summary = make_summary(0, &[("jemalloc-not-found", 12)], &[]);
    emit_probe_summary(&summary);
    assert!(logs_contain("ctprof probe:"));
    assert!(logs_contain("0 tgids walked"));
    assert!(logs_contain("0 failed"));
    assert!(
        !logs_contain("(dominant:"),
        "no failures means the dominant-tag clause is omitted",
    );
    assert!(
        !logs_contain("hint:"),
        "no failures means the EPERM hint is omitted",
    );
}

/// Privilege-dominated snapshot emits the hint with the
/// `$(which ktstr)` substring intact. Catches a regression
/// that drops the hint when the ptrace-dominates threshold
/// fires.
#[traced_test]
#[test]
fn summary_emits_privilege_hint_when_ptrace_dominates() {
    let summary = ProbeSummary {
        tgids_walked: 4,
        jemalloc_detected: 2,
        probed_ok: 0,
        failed: 4,
        attach_tag_counts: BTreeMap::new(),
        probe_tag_counts: [("ptrace-seize", 4u64)].into_iter().collect(),
    };
    emit_probe_summary(&summary);
    assert!(logs_contain("(dominant: ptrace-seize"));
    assert!(logs_contain("hint:"));
    assert!(logs_contain("$(which ktstr)"));
    assert!(logs_contain("cap_sys_ptrace"));
    assert!(logs_contain("yama.ptrace_scope"));
}

/// `ptrace-interrupt`-dominated snapshot also emits the
/// privilege hint. Pins the `matches!` arm in
/// `ProbeSummary::ptrace_dominates` covering both ptrace
/// tags, not just `ptrace-seize` — a regression that
/// narrowed the gate to `ptrace-seize` only would silently
/// drop the hint on hosts where the per-thread interrupt
/// step (rather than the initial seize) is the failure
/// mode (for example: yama scope=1 lets the seize succeed
/// against an opted-in target but blocks the per-tid
/// `PTRACE_INTERRUPT` step against threads created after
/// the opt-in window).
#[traced_test]
#[test]
fn summary_emits_privilege_hint_when_ptrace_interrupt_dominates() {
    let summary = ProbeSummary {
        tgids_walked: 4,
        jemalloc_detected: 2,
        probed_ok: 0,
        failed: 4,
        attach_tag_counts: BTreeMap::new(),
        probe_tag_counts: [("ptrace-interrupt", 4u64)].into_iter().collect(),
    };
    emit_probe_summary(&summary);
    assert!(logs_contain("(dominant: ptrace-interrupt"));
    assert!(logs_contain("hint:"));
    assert!(logs_contain("$(which ktstr)"));
    assert!(logs_contain("cap_sys_ptrace"));
    assert!(logs_contain("yama.ptrace_scope"));
}

/// Mixed-failure snapshot (DWARF + ptrace) where ptrace
/// stays below the half threshold emits the dominant tag
/// but NOT the privilege hint — a stripped-binary host
/// doesn't need the privilege fix, it needs debuginfo.
#[traced_test]
#[test]
fn summary_omits_privilege_hint_when_debuginfo_failures_lead() {
    let summary = ProbeSummary {
        tgids_walked: 5,
        jemalloc_detected: 3,
        probed_ok: 0,
        failed: 5,
        attach_tag_counts: [("dwarf-parse-failure", 4u64)].into_iter().collect(),
        probe_tag_counts: [("ptrace-seize", 1u64)].into_iter().collect(),
    };
    emit_probe_summary(&summary);
    assert!(logs_contain("(dominant: dwarf-parse-failure"));
    assert!(
        !logs_contain("hint:"),
        "DWARF-dominated failures must NOT trigger the privilege \
         hint — only privilege failures earn the privilege remediation",
    );
}

/// Clean parse-summary emission: zero failures, zero negative
/// dotted values. Pins that no dominant-tag clause, no kconfig
/// hint, and no negative-clause render when the underlying
/// signals are zero. Mirrors the
/// `summary_emits_clean_line_when_no_failures` discipline for
/// the probe summary side.
///
/// Test fn name uses `parse_summary_emits_*` rather than
/// `summary_emits_*` to keep the captured span text disjoint
/// from the asserted substrings (`tracing-test`'s
/// `logs_contain` matches the entire captured frame including
/// the span — same caveat the probe-summary emit tests
/// document).
#[traced_test]
#[test]
fn parse_summary_emits_clean_line_when_no_failures() {
    let tally = ParseTally::default();
    emit_parse_summary(&tally);
    assert!(logs_contain("ctprof parse:"));
    assert!(logs_contain("0 tids walked"));
    assert!(logs_contain("0 read failures"));
    assert!(
        !logs_contain("(dominant:"),
        "no failures means the dominant clause is omitted",
    );
    assert!(
        !logs_contain("hint:"),
        "no failures means the kconfig hint is omitted",
    );
    assert!(
        !logs_contain("negative-dotted"),
        "zero negative-dotted values means the negative \
         clause is omitted",
    );
}

/// Negative-dotted clause renders when the tally carries any
/// negative bumps. Pins the `, N negative-dotted values`
/// substring so a regression that drops the clause when read
/// failures are zero (the emit's failure path) surfaces
/// here.
#[traced_test]
#[test]
fn parse_summary_emits_negative_dotted_clause_when_present() {
    let mut tally = ParseTally {
        tids_walked: 5,
        ..ParseTally::default()
    };
    // Drive the negative-dotted counter through the public
    // path: pending bumps + commit, mirroring the production
    // capture pipeline.
    tally.record_negative_dotted();
    tally.record_negative_dotted();
    tally.record_negative_dotted();
    tally.commit_pending();
    emit_parse_summary(&tally);
    assert!(
        logs_contain("3 negative-dotted values"),
        "negative-dotted clause must surface the count when \
         the tally is non-zero — the operator-visibility \
         motivation depends on this rendering",
    );
    assert!(logs_contain("0 read failures"));
}

/// Kconfig hint renders alongside the dominant clause when
/// schedstat / io failures dominate. Pins both clauses
/// firing together so a refactor that conditioned them
/// independently surfaces here.
#[traced_test]
#[test]
fn parse_summary_emits_kconfig_hint_when_dominant() {
    let mut tally = ParseTally {
        tids_walked: 100,
        ..ParseTally::default()
    };
    // 60 schedstat + 40 io = 100% kconfig share, well above
    // the 50% gate.
    for _ in 0..60 {
        tally.record_failure("schedstat");
    }
    for _ in 0..40 {
        tally.record_failure("io");
    }
    tally.commit_pending();
    emit_parse_summary(&tally);
    assert!(logs_contain("(dominant: schedstat)"));
    assert!(logs_contain("hint:"));
    assert!(logs_contain("CONFIG_SCHEDSTATS"));
    assert!(logs_contain("CONFIG_TASK_IO_ACCOUNTING"));
}

/// `try_attach_probe_for_tgid_at` against a known-bad pid (0,
/// reserved by the kernel) emits a `tracing::warn!` event
/// (not debug) because PidMissing is NOT the
/// jemalloc-not-found case — it's a hard error worth
/// surfacing. Pins the level-routing rule from the helper's
/// doc.
#[traced_test]
#[test]
fn try_attach_probe_for_tgid_at_warns_on_pid_missing() {
    let mut summary = ProbeSummary::default();
    let probe = try_attach_probe_for_tgid_at(Path::new(DEFAULT_PROC_ROOT), 0, &mut summary);
    assert!(probe.is_none(), "pid 0 must not produce a probe");
    // PidMissing → tag "pid-missing", logged at warn, counted as failed.
    assert!(logs_contain("attach failed"));
    assert!(logs_contain("pid-missing"));
    assert_eq!(summary.failed, 1);
    assert_eq!(summary.jemalloc_detected, 0);
    assert_eq!(summary.tgids_walked, 1);
    assert_eq!(
        summary.attach_tag_counts.get("pid-missing").copied(),
        Some(1),
        "PidMissing tag must increment its bucket",
    );
}

/// `try_attach_probe_for_tgid_at` against a real process that
/// is NOT jemalloc-linked (`/bin/sleep` spawned for the
/// duration of the test) returns `None` AND logs at debug,
/// not warn — the JemallocNotFound case is the expected
/// outcome for the bulk of system processes and must not
/// flood the operator's log. Pins the
/// `jemalloc-not-found → debug` routing rule.
#[traced_test]
#[test]
fn try_attach_probe_for_tgid_at_debugs_on_non_jemalloc_target() {
    // /bin/sleep is a coreutils binary not linked against
    // jemalloc; attach_jemalloc walks its /proc/<pid>/maps,
    // finds no TSD symbol, and returns JemallocNotFound.
    let mut child = match std::process::Command::new("sleep")
        .arg("3")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping — /bin/sleep unavailable");
            return;
        }
    };
    // Poll for `/proc/<pid>/exe` to become readable rather than
    // burning a hardcoded settle window. On a fast host the
    // exe symlink resolves within microseconds of fork+exec; on
    // a contended CI runner it can lag a few ms. A 1 s deadline
    // with 1 ms backoff bounds the worst case while keeping the
    // common case nearly instantaneous, and deterministically
    // gates the test on the actual readiness signal rather than
    // a guess. `read_link` is the same syscall the probe attach
    // exercises, so once it succeeds the downstream
    // `try_attach_probe_for_tgid_at` call is guaranteed to find
    // an exe symlink it can resolve.
    let pid = child.id() as i32;
    let exe_link = std::path::PathBuf::from(format!("/proc/{pid}/exe"));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while std::fs::read_link(&exe_link).is_err() {
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "/proc/{pid}/exe did not become readable within 1s — \
                 kernel did not surface the freshly-forked child's exe \
                 symlink in time, the test cannot proceed"
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let mut summary = ProbeSummary::default();
    let probe = try_attach_probe_for_tgid_at(Path::new(DEFAULT_PROC_ROOT), pid, &mut summary);

    let _ = child.kill();
    let _ = child.wait();

    assert!(probe.is_none(), "sleep is not jemalloc-linked");
    assert_eq!(summary.tgids_walked, 1);
    assert_eq!(summary.jemalloc_detected, 0);
    assert_eq!(
        summary.failed, 0,
        "jemalloc-not-found must NOT count as failure — it's the \
         expected outcome for the bulk of system processes",
    );
    assert_eq!(
        summary.attach_tag_counts.get("jemalloc-not-found").copied(),
        Some(1),
    );
    // The debug event carries the "attach skipped" message;
    // tracing-test's logs_contain looks across all captured
    // events including debug.
    assert!(
        logs_contain("attach skipped"),
        "JemallocNotFound must emit the debug 'attach skipped' \
         event so log filters can route it separately from \
         actionable warnings",
    );
    assert!(
        !logs_contain("attach failed"),
        "jemalloc-not-found must NOT emit the warn 'attach failed' \
         event — that level is reserved for actionable failures",
    );
}

// ------------------------------------------------------------
// T28 — CtprofParseSummary: per-file read-failure tally
// ------------------------------------------------------------

/// Stage a synthetic procfs tree for parse-summary tests:
/// a single live tgid + tid with `comm` and `stat` populated
/// so the ghost filter does NOT fire (start_time is parseable
/// from `stat`). The caller then deletes the specific
/// per-file targets they want to fail. `cgroup` and other
/// non-asserted files are populated so the surrounding reads
/// succeed and the tally only counts the targeted failures.
fn stage_minimal_proc_for_parse(root: &Path, tgid: i32, tid: i32) {
    use std::fs;
    let tgid_dir = root.join(tgid.to_string());
    let task_dir = tgid_dir.join("task").join(tid.to_string());
    fs::create_dir_all(&task_dir).unwrap();
    fs::write(tgid_dir.join("comm"), "p\n").unwrap();
    fs::write(task_dir.join("comm"), "live\n").unwrap();
    // Non-zero start_time keeps the ghost filter from firing
    // even when other files vanish.
    let stat_line = format!(
        "{tid} (live) R 1 2 3 4 5 6 7 0 8 0 10 11 12 13 14 0 1 0 \
         555555 100 200 300 400 500 600 700 800 900 1000 1100 \
         1200 1300 1400 1500 1600 1700 1800 0\n"
    );
    fs::write(task_dir.join("stat"), stat_line).unwrap();
    fs::write(task_dir.join("schedstat"), "0 0 0\n").unwrap();
    fs::write(
        task_dir.join("status"),
        "voluntary_ctxt_switches:\t0\n\
         nonvoluntary_ctxt_switches:\t0\n",
    )
    .unwrap();
    fs::write(task_dir.join("io"), "rchar: 0\n").unwrap();
    fs::write(task_dir.join("sched"), "").unwrap();
    fs::write(task_dir.join("cgroup"), "0::/\n").unwrap();
}

/// Per-file-kind tally: deleting `schedstat` lands a single
/// `"schedstat"` failure in the summary's per-file map. Other
/// categories stay at zero (key absent from the map).
#[test]
fn parse_summary_records_schedstat_failure() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5050;
    let tid: i32 = 5051;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
    // Delete schedstat so the read fails.
    std::fs::remove_file(
        proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("schedstat"),
    )
    .unwrap();

    // capture_with(_, _, false) skips the production gate so
    // parse_summary is None; use true and stage a /proc tree
    // that the host_context probe absorbs without panicking.
    // For the synthetic-tree pattern, stage a tally directly.
    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(summary.tids_walked, 1);
    assert_eq!(summary.read_failures, 1);
    assert_eq!(summary.read_failures_by_file.get("schedstat"), Some(&1));
    assert!(!summary.read_failures_by_file.contains_key("stat"));
    assert!(!summary.read_failures_by_file.contains_key("io"));
}

/// Per-file-kind tally: deleting `io` lands an `"io"` failure.
#[test]
fn parse_summary_records_io_failure() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5060;
    let tid: i32 = 5061;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
    std::fs::remove_file(
        proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("io"),
    )
    .unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(summary.read_failures_by_file.get("io"), Some(&1));
}

/// Per-file-kind tally: a fully populated synthetic /proc
/// (every reader succeeds) lands an empty map and zero
/// `read_failures`. Pins the "absent key = zero" contract.
#[test]
fn parse_summary_clean_proc_yields_empty_map() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5070;
    let tid: i32 = 5071;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(summary.tids_walked, 1);
    assert_eq!(summary.read_failures, 0);
    assert!(
        summary.read_failures_by_file.is_empty(),
        "clean procfs must yield an empty map, got {:?}",
        summary.read_failures_by_file,
    );
    assert!(summary.dominant_read_failure.is_none());
    assert!(!summary.kernel_config_dominant);
}

/// Ghost filter discipline (T28.2): a tid that exits between
/// readdir and the per-file reads (every read fails with
/// ENOENT, comm is empty, ghost filter rejects the tid) must
/// NOT contribute to the parse-summary tally. Otherwise a
/// busy host with mid-capture exits would inflate
/// `read_failures` with bumps that correspond to threads the
/// snapshot doesn't even contain.
#[test]
fn parse_summary_excludes_ghost_filtered_tids() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5080;
    let tid: i32 = 5081;
    // Stage only the empty task directory (no comm, no stat,
    // no other files) so every read fails AND the ghost filter
    // fires (empty comm + zero start_time).
    let task_dir = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string());
    std::fs::create_dir_all(&task_dir).unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let t =
        capture_thread_at_with_tally(proc_tmp.path(), tgid, tid, "", "", false, &mut tally_opt);
    // Ghost filter: empty comm + zero start_time → discard.
    if t.comm.is_empty() && t.start_time_clock_ticks == 0 {
        tally_opt.as_mut().unwrap().discard_pending();
    } else {
        tally_opt.as_mut().unwrap().commit_pending();
    }

    let summary = tally.to_public();
    assert_eq!(
        summary.read_failures, 0,
        "ghost-filtered tid must NOT contribute to read_failures; \
         got {} failures (the discard_pending unwind is broken)",
        summary.read_failures,
    );
    assert!(summary.read_failures_by_file.is_empty());
    // tids_walked still incremented — the tid was attempted.
    assert_eq!(summary.tids_walked, 1);
}

/// Serde round-trip: a populated `CtprofParseSummary`
/// preserves every field through JSON.
#[test]
fn parse_summary_serde_round_trip() {
    let mut by_file = BTreeMap::new();
    by_file.insert("schedstat".to_string(), 100);
    by_file.insert("io".to_string(), 50);
    let summary = CtprofParseSummary {
        tids_walked: 1000,
        read_failures: 150,
        read_failures_by_file: by_file,
        dominant_read_failure: Some("schedstat".to_string()),
        kernel_config_dominant: true,
        negative_dotted_values: 7,
    };
    let json = serde_json::to_string(&summary).unwrap();
    let back: CtprofParseSummary = serde_json::from_str(&json).unwrap();
    assert_eq!(back.tids_walked, 1000);
    assert_eq!(back.read_failures, 150);
    assert_eq!(back.read_failures_by_file.get("schedstat"), Some(&100));
    assert_eq!(back.read_failures_by_file.get("io"), Some(&50));
    assert_eq!(back.dominant_read_failure.as_deref(), Some("schedstat"));
    assert!(back.kernel_config_dominant);
    assert_eq!(
        back.negative_dotted_values, 7,
        "negative_dotted_values surfaces in the public surface \
         and round-trips through JSON",
    );
}

/// `dominant_read_failure` picks the file kind with the most
/// failures. Ties resolve REVERSE-alphabetically (mirrors the
/// probe-summary comparator) — alphabetically-EARLIER tag
/// wins.
#[test]
fn parse_summary_dominant_picks_max_file_kind() {
    let mut tally = ParseTally::default();
    // schedstat: 10 failures, io: 5, status: 5. schedstat wins.
    for _ in 0..10 {
        tally.record_failure("schedstat");
    }
    for _ in 0..5 {
        tally.record_failure("io");
    }
    for _ in 0..5 {
        tally.record_failure("status");
    }
    tally.commit_pending();
    let summary = tally.to_public();
    assert_eq!(summary.dominant_read_failure.as_deref(), Some("schedstat"));

    // Tie between io and status (same count) — io wins (earlier
    // alphabetical, matches the reverse-alphabetical comparator).
    let mut tally2 = ParseTally::default();
    for _ in 0..3 {
        tally2.record_failure("io");
    }
    for _ in 0..3 {
        tally2.record_failure("status");
    }
    tally2.commit_pending();
    let summary2 = tally2.to_public();
    assert_eq!(
        summary2.dominant_read_failure.as_deref(),
        Some("io"),
        "tie must resolve to alphabetically-earlier tag — \
         `io` beats `status`",
    );
}

/// `kernel_config_hint` returns Some(_) when ≥ 50% of failures
/// land in `schedstat`/`io`. Pins the gate equality at the
/// boundary.
#[test]
fn parse_summary_kernel_config_hint_gate() {
    // 50/50 split: 5 schedstat + 5 status. Kconfig share = 50%.
    let mut tally = ParseTally::default();
    for _ in 0..5 {
        tally.record_failure("schedstat");
    }
    for _ in 0..5 {
        tally.record_failure("status");
    }
    tally.commit_pending();
    let summary = tally.to_public();
    assert!(
        summary.kernel_config_dominant,
        "50% kconfig share must hit the gate (>= 50% boundary inclusive)",
    );
    assert!(summary.kernel_config_hint().is_some());

    // Below threshold: 1 schedstat, 9 status. Kconfig share 10%.
    let mut tally2 = ParseTally::default();
    tally2.record_failure("schedstat");
    for _ in 0..9 {
        tally2.record_failure("status");
    }
    tally2.commit_pending();
    let summary2 = tally2.to_public();
    assert!(!summary2.kernel_config_dominant);
    assert!(summary2.kernel_config_hint().is_none());

    // Zero failures: kconfig_dominant must be false (no failures
    // to dominate), hint is None.
    let summary3 = ParseTally::default().to_public();
    assert!(!summary3.kernel_config_dominant);
    assert!(summary3.kernel_config_hint().is_none());
}

/// `dominant_read_failure` is None when zero failures landed,
/// even though the tally was constructed.
#[test]
fn parse_summary_dominant_none_when_zero_failures() {
    let summary = ParseTally::default().to_public();
    assert_eq!(summary.read_failures, 0);
    assert!(summary.dominant_read_failure.is_none());
}

/// `capture_with(_, _, false)` skips the production gate so
/// `parse_summary` stays `None` on the assembled snapshot —
/// mirrors the `probe_summary` discipline. Synthetic-tree
/// tests must not see a populated parse summary.
#[test]
fn capture_with_synthetic_tree_yields_no_parse_summary() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 5090;
    let tid: i32 = 5091;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert!(
        snap.parse_summary.is_none(),
        "use_syscall_affinity=false must skip parse_summary; \
         got Some — production-gate discipline is broken",
    );
}

// ------------------------------------------------------------
// T43 — Additional capture-pipeline error-path tests
// ------------------------------------------------------------

/// Phase-1 loadavg missing: capture_with must not panic when
/// the parallelism-clamp `proc_root/loadavg` read fails. The
/// reader's `.ok().and_then(...).unwrap_or(0.0)` chain folds
/// the missing-file branch into the 0.0 default, so the
/// headroom calculation continues to clamp at
/// `[1, num_cpus/2 + 1]`. Pins the missing-loadavg branch.
#[test]
fn capture_with_phase1_loadavg_missing_does_not_panic() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    // No loadavg file. iter_tgids_at returns Vec::new() so the
    // probe-attach loop iterates zero times — but the clamp
    // computation runs unconditionally inside the
    // use_syscall_affinity=true branch, exercising the
    // missing-file path.
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "missing loadavg + empty proc_root → empty snapshot, \
         got {} threads",
        snap.threads.len(),
    );
}

/// Phase-1 loadavg malformed: a non-float first token must
/// fold into the 0.0 default via the `.parse::<f64>().ok()`
/// step. Pins that a hostile `proc_root/loadavg` cannot crash
/// the parallelism-clamp computation.
#[test]
fn capture_with_phase1_loadavg_malformed_does_not_panic() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(proc_tmp.path().join("loadavg"), "not_a_number\n").unwrap();
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "malformed loadavg → 0.0 default, empty proc_root → empty \
         snapshot; got {} threads",
        snap.threads.len(),
    );
}

/// Non-UTF-8 bytes in `comm`: `fs::read_to_string` returns Err
/// on invalid UTF-8, so [`read_thread_comm_at`] yields None
/// and the caller defaults to "". With `start_time` non-zero
/// (intact `stat`), the ghost filter does NOT fire and the
/// thread lands with empty comm.
#[test]
fn capture_with_non_utf8_comm_treated_as_absent() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6161;
    let tid: i32 = 6162;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite tid/comm with non-UTF-8 bytes (lone 0xFF, then
    // 0xFE — never valid UTF-8 lead bytes).
    let comm_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("comm");
    std::fs::write(&comm_path, [0xFF, 0xFE]).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "non-UTF-8 comm folds to empty; ghost filter does NOT \
         fire because start_time is intact; thread still lands. \
         got {} threads",
        snap.threads.len(),
    );
    assert_eq!(
        snap.threads[0].comm, "",
        "non-UTF-8 comm must collapse to empty (read_to_string \
         returns Err on invalid UTF-8)",
    );
    assert_ne!(
        snap.threads[0].start_time_clock_ticks, 0,
        "start_time must be intact for the ghost filter NOT to fire",
    );
}

/// Cgroup path traversal: a `0::/../escape` payload in the
/// per-tid cgroup file lands in `ThreadState.cgroup` verbatim
/// (no sanitization at parse time), and the cgroup_stats
/// enrichment loop calls `read_cgroup_stats_at` with the same
/// string. The current behaviour bounds the read inside the
/// configured `cgroup_root` via `Path::join` — which DOES NOT
/// reject `..` components. Pin that the path-traversal string
/// round-trips through the snapshot but does not surface
/// out-of-tree cgroup data: the stats land at the all-zero
/// default because no matching cgroup directory exists under
/// `cgroup_root`.
#[test]
fn capture_with_cgroup_path_traversal_yields_zero_stats() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6262;
    let tid: i32 = 6263;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite cgroup with a traversal string.
    let cgroup_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("cgroup");
    std::fs::write(&cgroup_path, "0::/../escape\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    assert_eq!(
        snap.threads[0].cgroup, "/../escape",
        "traversal string round-trips verbatim through ThreadState.cgroup",
    );
    let stats = snap
        .cgroup_stats
        .get("/../escape")
        .expect("non-empty cgroup string must seed the stats map");
    assert_eq!(
        stats.cpu.usage_usec, 0,
        "no matching cgroup dir under cgroup_root → all-zero stats; \
         a traversal that escaped the cgroup_root would have \
         non-zero values from the parent directory",
    );
}

/// Empty `Cpus_allowed_list:` value: `parse_cpu_list("")`
/// returns None at the empty-input guard, so `cpu_affinity`
/// lands as the empty Vec. Same observable effect as a
/// malformed range (G8) but pins the empty-string branch
/// distinctly.
#[test]
fn capture_with_empty_cpus_allowed_yields_empty_affinity() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6363;
    let tid: i32 = 6364;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    let status_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("status");
    let status = "Cpus_allowed_list:\t\n\
         voluntary_ctxt_switches:\t1\n\
         nonvoluntary_ctxt_switches:\t1\n";
    std::fs::write(&status_path, status).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(snap.threads.len(), 1);
    let t = &snap.threads[0];
    assert!(
        t.cpu_affinity.0.is_empty(),
        "empty Cpus_allowed_list value → parse_cpu_list returns \
         None at the empty-input guard → cpu_affinity empty; \
         got {} elements",
        t.cpu_affinity.0.len(),
    );
    assert_eq!(
        t.voluntary_csw,
        MonotonicCount(1),
        "empty cpulist must not break csw parsing on the same \
         status file",
    );
}

/// Ghost filter AND-semantics: an empty `comm` paired with a
/// NON-zero `start_time_clock_ticks` does NOT fire the filter.
/// The clause requires BOTH conditions (see
/// `t.comm.is_empty() && t.start_time_clock_ticks == 0`). Pins
/// the AND so a future refactor that flipped to OR would
/// surface here rather than hiding legitimate threads with
/// transient empty comms.
#[test]
fn capture_with_empty_comm_nonzero_start_time_keeps_thread() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 6464;
    let tid: i32 = 6465;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite comm with whitespace so read_thread_comm_at
    // returns None → comm defaults to "". start_time stays
    // intact at 555_555 (the value stage_synthetic_proc writes).
    let comm_path = proc_tmp
        .path()
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join("comm");
    std::fs::write(&comm_path, "   \n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "empty comm + nonzero start_time MUST NOT fire ghost filter \
         (AND-semantics requires both empty); got {} threads",
        snap.threads.len(),
    );
    let t = &snap.threads[0];
    assert_eq!(t.comm, "", "empty-comm thread surfaces with empty comm");
    assert_ne!(
        t.start_time_clock_ticks, 0,
        "start_time must be non-zero so the AND-clause has a `false` half",
    );
}

// ------------------------------------------------------------
// T45 — Additional parse_summary + capture-pipeline coverage
// ------------------------------------------------------------

/// W2: every tid is ghost-filtered. With N empty task dirs the
/// ghost filter rejects every tid, so each tid's pending failure
/// bumps unwind via `discard_pending`. `tids_walked` is bumped
/// at the call site BEFORE the discard, so it still reads N.
/// `read_failures` lands at zero (every bump unwound), the per-
/// file map is empty, and `dominant_read_failure` is None. Pins
/// the "tids_walked counts attempts; failure tallies count only
/// committed bumps" split end-to-end through `capture_with`.
#[test]
fn parse_summary_all_ghosts_yields_nonzero_tids_walked_zero_failures() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7070;
    let n: u64 = 4;
    // Stage one tgid with N empty task dirs (no comm, no stat,
    // no other files). Every read fails; ghost filter fires for
    // every tid; every pending tally is unwound.
    let tgid_dir = proc_tmp.path().join(tgid.to_string());
    for k in 0..n {
        let tid = (tgid as u64 + 1 + k) as i32;
        std::fs::create_dir_all(tgid_dir.join("task").join(tid.to_string())).unwrap();
    }
    // Stage `loadavg` so the parallelism-clamp read in phase 1
    // resolves cleanly (the missing-file fallback is exercised
    // by capture_with_phase1_loadavg_missing_does_not_panic).
    std::fs::write(proc_tmp.path().join("loadavg"), "0.10 0.05 0.01 1/1 1\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.threads.is_empty(),
        "every tid is ghost-filtered → threads must be empty, got {}",
        snap.threads.len(),
    );
    let summary = snap
        .parse_summary
        .expect("use_syscall_affinity=true must populate parse_summary");
    assert_eq!(
        summary.tids_walked, n,
        "tids_walked counts every walk attempt, not committed reads — \
         got {}, want {n}",
        summary.tids_walked,
    );
    assert_eq!(
        summary.read_failures, 0,
        "ghost-filtered tids' failures unwind via discard_pending — \
         got {} failures, want 0",
        summary.read_failures,
    );
    assert!(
        summary.read_failures_by_file.is_empty(),
        "no failure bucket survives the ghost-filter unwind, got {:?}",
        summary.read_failures_by_file,
    );
    assert!(
        summary.dominant_read_failure.is_none(),
        "zero failures → dominant_read_failure is None, got {:?}",
        summary.dominant_read_failure,
    );
    assert!(
        !summary.kernel_config_dominant,
        "zero failures → kernel_config_dominant is false, got true",
    );
}

/// W3: pin which file-kind tokens count as kernel-config-gated.
/// `kernel_config_dominates` filters on `matches!(t, "schedstat"
/// | "io")`. Iterate every recognised kebab token solo (one
/// failure of that kind, no others) and assert the gate flips
/// the way the implementation says it should — schedstat/io
/// land 100% kconfig and the gate fires; stat/status/sched/cgroup
/// land 0% kconfig and the gate stays false. A future refactor
/// that added or removed a token from the kconfig set without
/// updating the docs would surface here.
#[test]
fn parse_summary_kernel_config_token_list_pinned() {
    let kconfig_tokens: &[&'static str] = &["schedstat", "io"];
    for tag in kconfig_tokens {
        let mut tally = ParseTally::default();
        tally.record_failure(tag);
        tally.commit_pending();
        let summary = tally.to_public();
        assert!(
            summary.kernel_config_dominant,
            "solo `{tag}` failure must flip kernel_config_dominant true \
             (kconfig share = 100%); got false — token dropped from the \
             kconfig set",
        );
    }

    let non_kconfig_tokens: &[&'static str] = &["stat", "status", "sched", "cgroup"];
    for tag in non_kconfig_tokens {
        let mut tally = ParseTally::default();
        tally.record_failure(tag);
        tally.commit_pending();
        let summary = tally.to_public();
        assert!(
            !summary.kernel_config_dominant,
            "solo `{tag}` failure must keep kernel_config_dominant false \
             (kconfig share = 0%); got true — token incorrectly added to \
             the kconfig set",
        );
    }
}

/// W5: tally aggregates across multiple tids. Stage 2 tids
/// where each fails a different file (one missing io, one
/// missing schedstat). Both bumps must commit (neither tid is
/// ghost-filtered) and the per-file map carries one entry per
/// failure kind with count 1, total `read_failures` = 2.
#[test]
fn parse_summary_aggregates_across_multiple_tids() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7080;
    let tid_a: i32 = 7081;
    let tid_b: i32 = 7082;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid_a);
    // Second tid under the same tgid: write a fresh task dir.
    let tgid_dir = proc_tmp.path().join(tgid.to_string());
    let task_b = tgid_dir.join("task").join(tid_b.to_string());
    std::fs::create_dir_all(&task_b).unwrap();
    std::fs::write(task_b.join("comm"), "live\n").unwrap();
    let stat_line = format!(
        "{tid_b} (live) R 1 2 3 4 5 6 7 0 8 0 10 11 12 13 14 0 1 0 \
         555555 100 200 300 400 500 600 700 800 900 1000 1100 \
         1200 1300 1400 1500 1600 1700 1800 0\n"
    );
    std::fs::write(task_b.join("stat"), stat_line).unwrap();
    std::fs::write(task_b.join("schedstat"), "0 0 0\n").unwrap();
    std::fs::write(
        task_b.join("status"),
        "voluntary_ctxt_switches:\t0\n\
         nonvoluntary_ctxt_switches:\t0\n",
    )
    .unwrap();
    std::fs::write(task_b.join("io"), "rchar: 0\n").unwrap();
    std::fs::write(task_b.join("sched"), "").unwrap();
    std::fs::write(task_b.join("cgroup"), "0::/\n").unwrap();

    // tid_a: delete io. tid_b: delete schedstat.
    std::fs::remove_file(tgid_dir.join("task").join(tid_a.to_string()).join("io")).unwrap();
    std::fs::remove_file(task_b.join("schedstat")).unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    for tid in [tid_a, tid_b] {
        tally_opt.as_mut().unwrap().tids_walked += 1;
        let _ = capture_thread_at_with_tally(
            proc_tmp.path(),
            tgid,
            tid,
            "p",
            "live",
            false,
            &mut tally_opt,
        );
        tally_opt.as_mut().unwrap().commit_pending();
    }
    let summary = tally.to_public();
    assert_eq!(summary.tids_walked, 2);
    assert_eq!(
        summary.read_failures, 2,
        "two tids, one failure each → 2 total; got {}",
        summary.read_failures,
    );
    assert_eq!(
        summary.read_failures_by_file.get("io"),
        Some(&1),
        "tid_a missing io → io bucket = 1; got {:?}",
        summary.read_failures_by_file.get("io"),
    );
    assert_eq!(
        summary.read_failures_by_file.get("schedstat"),
        Some(&1),
        "tid_b missing schedstat → schedstat bucket = 1; got {:?}",
        summary.read_failures_by_file.get("schedstat"),
    );
}

/// W7: deleting cgroup lands a `"cgroup"` failure. Mirrors the
/// schedstat/io single-failure tests so the cgroup-read tally
/// path is exercised explicitly — `read_cgroup_at_with_tally`
/// is the only producer of the `"cgroup"` tag and a future
/// refactor that bypassed the tally would surface here.
#[test]
fn parse_summary_records_cgroup_failure() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7090;
    let tid: i32 = 7091;
    stage_minimal_proc_for_parse(proc_tmp.path(), tgid, tid);
    std::fs::remove_file(
        proc_tmp
            .path()
            .join(tgid.to_string())
            .join("task")
            .join(tid.to_string())
            .join("cgroup"),
    )
    .unwrap();

    let mut tally = ParseTally::default();
    let mut tally_opt: Option<&mut ParseTally> = Some(&mut tally);
    tally_opt.as_mut().unwrap().tids_walked += 1;
    let _ = capture_thread_at_with_tally(
        proc_tmp.path(),
        tgid,
        tid,
        "p",
        "live",
        false,
        &mut tally_opt,
    );
    tally_opt.as_mut().unwrap().commit_pending();

    let summary = tally.to_public();
    assert_eq!(
        summary.read_failures_by_file.get("cgroup"),
        Some(&1),
        "missing cgroup file → cgroup bucket = 1; got {:?}",
        summary.read_failures_by_file.get("cgroup"),
    );
}

/// W6: the production gate (`use_syscall_affinity=true`)
/// populates `parse_summary` end-to-end. Mirror of
/// `capture_with_synthetic_tree_yields_no_parse_summary` but
/// with the gate flipped — pins that the production-path
/// assignment is wired through.
#[test]
fn capture_with_production_gate_populates_parse_summary() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7100;
    let tid: i32 = 7101;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // loadavg lets the parallelism-clamp read resolve cleanly.
    std::fs::write(proc_tmp.path().join("loadavg"), "0.10 0.05 0.01 1/1 1\n").unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    assert!(
        snap.parse_summary.is_some(),
        "use_syscall_affinity=true must populate parse_summary on \
         the assembled snapshot — production-gate wiring is broken",
    );
}

/// X2: non-UTF-8 bytes in `<tgid>/comm` (the pcomm path).
/// `read_process_comm_at` calls `fs::read_to_string`, which
/// returns Err on invalid UTF-8; `.ok()?` propagates None and
/// the caller defaults `pcomm` to "" via `.unwrap_or_default()`.
/// Pin that capture does not panic and the per-thread `pcomm`
/// surfaces empty. Mirror of
/// `capture_with_non_utf8_comm_treated_as_absent` but for the
/// process-level (`<tgid>/comm`) read rather than the per-tid
/// (`<tgid>/task/<tid>/comm`) read.
#[test]
fn capture_with_non_utf8_pcomm_treated_as_absent() {
    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    let tgid: i32 = 7110;
    let tid: i32 = 7111;
    stage_synthetic_proc(proc_tmp.path(), tgid, tid, "p", "live");
    // Overwrite the pcomm path (`<tgid>/comm`) with non-UTF-8
    // lead bytes (0xFF and 0xFE — never valid UTF-8 starts).
    let pcomm_path = proc_tmp.path().join(tgid.to_string()).join("comm");
    std::fs::write(&pcomm_path, [0xFF, 0xFE]).unwrap();

    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), false);
    assert_eq!(
        snap.threads.len(),
        1,
        "non-UTF-8 pcomm must not break the capture — the thread still \
         lands; got {} threads",
        snap.threads.len(),
    );
    assert_eq!(
        snap.threads[0].pcomm, "",
        "non-UTF-8 pcomm collapses to empty (read_to_string returns Err \
         on invalid UTF-8 and unwrap_or_default → \"\")",
    );
}

/// Y1: panic-injection harness for rayon worker panics.
///
/// `attach_jemalloc_at` reads `/proc/<pid>/exe`, opens the ELF
/// file, and walks DWARF — every step can panic under fd
/// exhaustion or OOM. Without the `catch_unwind` guard in
/// `capture_with`'s phase-1 worker closure, a single panicking
/// tgid would propagate through `pool.install` and tear down
/// the whole snapshot. No realistic synthetic input can force
/// the underlying readers to panic, so this test installs an
/// explicit injection seam (`PANIC_INJECT_TGID`) that fires
/// inside `attach_probe_for_tgid_at` for the matching tgid and
/// drives the rayon worker into a panic. The capture pipeline
/// must absorb it, surface it as a `worker-panic` attach tag,
/// and still walk the surviving tgid's threads.
///
/// Asserts:
///   - `capture_with(.., true)` returns rather than unwinding,
///   - the surviving tgid's thread lands in the snapshot,
///   - `probe_summary.failed >= 1` (the panic is counted),
///   - `dominant_failure == Some("worker-panic")` (the new tag
///     surfaces in the curated public surface).
#[test]
fn capture_with_rayon_worker_panic_is_caught_and_surfaced() {
    // Serialize panic-hook test against any future test that
    // also installs a custom hook, so the silenced hook below
    // is not clobbered. `Mutex<()>` is enough — the lock is
    // only held for the duration of the capture call.
    static PANIC_INJECT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = PANIC_INJECT_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let proc_tmp = tempfile::TempDir::new().unwrap();
    let cgroup_tmp = tempfile::TempDir::new().unwrap();
    let sys_tmp = tempfile::TempDir::new().unwrap();
    // Required by the parallelism-clamp logic in capture_with.
    std::fs::write(proc_tmp.path().join("loadavg"), "0.0 0.0 0.0 1/1 1\n").unwrap();

    // Two tgids: the survivor (clean attach attempt → fails
    // benignly with `readlink-failure` because the synthetic
    // /proc has no `<tgid>/exe` symlink — the dominant-tag
    // filter suppresses this, leaving worker-panic as the
    // sole dominant candidate) and the panic target (the
    // sentinel tgid the seam matches against). Sentinel value
    // 99001 is intentionally outside any other test's range so
    // a parallel run cannot cross-fire.
    let survivor_tgid: i32 = 99000;
    let survivor_tid: i32 = 99002;
    let panic_tgid: i32 = 99001;
    let panic_tid: i32 = 99003;
    stage_synthetic_proc(
        proc_tmp.path(),
        survivor_tgid,
        survivor_tid,
        "ok-pcomm",
        "ok-comm",
    );
    stage_synthetic_proc(
        proc_tmp.path(),
        panic_tgid,
        panic_tid,
        "panic-pcomm",
        "panic-comm",
    );

    // Silence the default panic hook: rayon's worker panic
    // would otherwise dump a stack trace to stderr and pollute
    // the test output. Restore the hook before the lock
    // releases so subsequent tests see the real hook again.
    let saved_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_info| {}));

    // Arm the seam, run capture, then disarm BEFORE restoring
    // the hook so a panic during disarm (none expected) still
    // hits the silenced hook rather than the real one.
    PANIC_INJECT_TGID.store(panic_tgid, std::sync::atomic::Ordering::Release);
    let snap = capture_with(proc_tmp.path(), cgroup_tmp.path(), sys_tmp.path(), true);
    PANIC_INJECT_TGID.store(0, std::sync::atomic::Ordering::Release);

    std::panic::set_hook(saved_hook);

    // Survivor thread must land. The panicking tgid's threads
    // are walked too (phase 2 still iterates every tgid in
    // `tgids`), so total threads is 2.
    assert_eq!(
        snap.threads.len(),
        2,
        "rayon worker panic must not block phase 2 — both staged tgids \
         walk their threads; got {} threads",
        snap.threads.len(),
    );

    let summary = snap
        .probe_summary
        .expect("use_syscall_affinity=true must populate probe_summary");
    assert!(
        summary.failed >= 1,
        "worker-panic must count as a failure; got failed={}",
        summary.failed,
    );
    assert_eq!(
        summary.dominant_failure.as_deref(),
        Some("worker-panic"),
        "worker-panic is the only ACTIONABLE failure tag in this \
         scenario. The survivor's synthetic /proc has no `exe` \
         symlink, so attach short-circuits with `readlink-failure` \
         — the dominant-tag comparator filters that benign tag out \
         (same `matches!` arm `record_attach_outcome` uses to log it \
         at debug rather than warn), leaving worker-panic as the \
         sole candidate. A regression that demoted worker-panic \
         out of the dominant set, or that miscounted the panic, \
         would fail here. Got {:?}",
        summary.dominant_failure,
    );
}
