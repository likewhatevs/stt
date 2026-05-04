//! ProbeSummary tag tallies, ptrace EPERM dominance hint, summary line composition, try_attach_probe routing.
//!
//! Co-located with `super::mod.rs`; one of the topic-grouped
//! split files that replace the monolithic `tests.rs`.

#![cfg(test)]

use super::*;
use std::path::Path;
use tracing_test::traced_test;

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
    //
    // Sleep duration choice: 3 s. Budget breakdown for the
    // child-process critical section:
    // - Up to 1 s waiting for `/proc/<pid>/exe` readability
    //   (the deadline below). Worst case on a contended
    //   runner.
    // - The `try_attach_probe_for_tgid_at` call itself reads
    //   /proc/<pid>/maps and walks ELF/DWARF only when the
    //   binary is jemalloc-linked. /bin/sleep is not, so it
    //   short-circuits with JemallocNotFound before any heavy
    //   work — on the order of milliseconds.
    // - The `child.kill()` + `child.wait()` reap below
    //   completes in microseconds.
    //
    // Total expected wall-clock: well under 1.5 s. The 3 s
    // budget gives ~2x headroom for CI runners under
    // load — enough that an unexpectedly slow procfs read
    // doesn't let the child exit before the attach call
    // (which would race the test into a "pid vanished" path
    // with PidMissing instead of the JemallocNotFound the
    // test pins). 5 s would be excessive (extra wall-clock
    // for every test run); 1 s would be too tight (the 1 s
    // exe-readability deadline alone could exhaust it).
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
