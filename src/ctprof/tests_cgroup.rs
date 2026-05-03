//! read_cgroup_stats_at + parse_sched coverage (cgroup v2 cpu.stat / memory.current + sched-stat key parsing).
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;
use std::path::Path;


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
