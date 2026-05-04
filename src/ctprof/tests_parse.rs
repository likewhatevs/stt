//! /proc parse_* helpers + read_* helpers (stat, schedstat, io, status, psi, smaps_rollup, sched_ext sysfs, cgroup v2, cpu_max), identity helpers, and read_thread_comm.
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;
use std::path::Path;

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
    let scx = read_sched_ext_sysfs_at(sys_root.path()).expect("directory exists → returns Some");
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
