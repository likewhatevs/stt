//! Unit tests for the `parse_stall_duration_seconds` helper used by
//! the `watchdog_override_timing_precision` gauntlet entry in
//! `tests/ktstr_sched_tests.rs`. The helper parses the sched_ext
//! SCX_EXIT_ERROR_STALL message
//! (`"{task}[{pid}] failed to run for {secs}.{millis}s"`) emitted by
//! `kernel/sched/ext.c` scx_exit — the watchdog timing test reads
//! the guest kernel ring buffer via `ktstr::read_kmsg` and feeds the
//! result through this helper to recover the kernel's own
//! stall-duration measurement.
//!
//! Kept in its own file (rather than nested in
//! `ktstr_sched_tests.rs`) because the ktstr early-dispatch path
//! intercepts nextest `--list` and `--exact` for any test binary
//! that registers `KtstrTestEntry` entries — plain `#[test]`
//! functions in such a binary are filtered out of the listing.
//! This file registers no gauntlet entries, so the standard
//! harness discovers its `#[test]` functions normally.

use grok::Grok;

/// Re-implementation of the helper under test. Kept in sync with
/// `parse_stall_duration_seconds` in `tests/ktstr_sched_tests.rs`
/// — both files compile separately under cargo (each integration
/// test is its own binary), so a shared function would require a
/// helper crate or a `#[path = ...]` include. Duplicating the
/// 7-line helper keeps the dep graph flat; the unit tests here
/// pin the contract via direct input, and the production call
/// site in `ktstr_sched_tests.rs` uses the same grok pattern.
fn parse_stall_duration_seconds(kmsg: &str) -> Option<f64> {
    // Kernel emits `failed to run for %u.%03us` (integer dot integer 's')
    // per `kernel/sched/ext.c` scx_exit. Decompose into two `INT`
    // captures — NOT `NUMBER`, because NUMBER expands to BASE10NUM
    // which already matches `2.004` as a whole decimal and would
    // greedily consume the `.`, leaving nothing for the second
    // capture and making the pattern fail to match.
    // `INT` matches `[+-]?[0-9]+` — an integer with optional sign —
    // which matches each side of the kernel's printf output
    // individually.
    let grok = Grok::with_default_patterns();
    let pattern = grok
        .compile(r"failed to run for %{INT:seconds}\.%{INT:millis}s", false)
        .expect("grok pattern compiles with fancy-regex backend");
    let matches = pattern.match_against(kmsg)?;
    let seconds: u64 = matches.get("seconds")?.parse().ok()?;
    let millis: u64 = matches.get("millis")?.parse().ok()?;
    Some(seconds as f64 + (millis as f64) / 1000.0)
}

#[test]
fn parses_kernel_stall_message() {
    let kmsg = "\
[   42.001] sched_ext: enabled scx-ktstr
[   44.105] kworker/0:1[42] failed to run for 2.004s
[   44.120] sched_ext: scx-ktstr: BPF scheduler \"scx-ktstr\" errored, disabling
";
    assert_eq!(parse_stall_duration_seconds(kmsg), Some(2.004));
}

#[test]
fn returns_none_when_no_stall_message() {
    let kmsg = "[   42.001] sched_ext: enabled scx-ktstr\n";
    assert_eq!(parse_stall_duration_seconds(kmsg), None);
}

#[test]
fn parses_the_first_stall_line_when_multiple() {
    // grok's match_against returns the FIRST successful match over
    // the input. Earliest stall line wins.
    let kmsg = "\
a[1] failed to run for 1.500s
b[2] failed to run for 3.700s
";
    assert_eq!(parse_stall_duration_seconds(kmsg), Some(1.500));
}

#[test]
fn parses_exact_second_boundary() {
    // Kernel printf `%u.%03u` always emits three-digit millis, even
    // on an exact-second boundary: 5 seconds exactly renders as
    // `5.000s`, not `5s`. Pin that the parser combines the two INT
    // captures back into the correct f64 seconds value.
    let kmsg = "x[1] failed to run for 5.000s\n";
    assert_eq!(parse_stall_duration_seconds(kmsg), Some(5.0));
}

#[test]
fn computes_seconds_plus_millis_correctly() {
    // Spot-check the two-part reconstruction: `0.123s` is
    // 0 + 123/1000 = 0.123. `7.999s` is 7.999.
    assert_eq!(
        parse_stall_duration_seconds("a[1] failed to run for 0.123s\n"),
        Some(0.123),
    );
    assert_eq!(
        parse_stall_duration_seconds("b[2] failed to run for 7.999s\n"),
        Some(7.999),
    );
}
