//! Temporal-assertion patterns over a periodic
//! [`SampleSeries`](crate::scenario::sample::SampleSeries).
//!
//! `SeriesField<T>` is a per-sample column extracted from the
//! series via [`SampleSeries::bpf`] or [`SampleSeries::stats`] (or
//! the typed `bpf_map` / `stats_path` projectors). It carries a
//! parallel `(tag, elapsed_ms, SnapshotResult<T>)` triple per
//! sample so any failure-path message can name the offending tag
//! and timestamp without re-walking the source data.
//!
//! The six built-in patterns are:
//!   1. `nondecreasing` / `strictly_increasing` — counter
//!      monotonicity with optional warmup skip.
//!   2. `rate_within(lo, hi)` — bounded delta-per-millisecond
//!      between consecutive samples.
//!   3. `steady_within(warmup, tolerance)` — post-warmup samples
//!      stay inside `[mean·(1-tol), mean·(1+tol)]`.
//!   4. `converges_to(target, tol, deadline_ms)` — three
//!      consecutive samples land inside `[target-tol, target+tol]`
//!      before `deadline_ms`.
//!   5. `always_true` — boolean invariant at every sample
//!      (`SeriesField<bool>` only).
//!   6. `ratio_within(other, lo, hi)` — cross-field correlation
//!      between two same-length series.
//!
//! Per-sample scalar checks bypass the temporal patterns via
//! [`SeriesField::each`], which yields an [`EachClaim`] supporting
//! `at_least` / `at_most` / `between`. All patterns route through
//! [`Verdict`] and tag failures with [`DetailKind::Temporal`].

use crate::scenario::snapshot::{SnapshotError, SnapshotResult};

use super::{AssertDetail, DetailKind, Verdict};

/// Per-sample column extracted from a
/// [`SampleSeries`](crate::scenario::sample::SampleSeries). Each
/// slot is a [`SnapshotResult<T>`] so a missing or
/// type-mismatched field does NOT abort the whole projection — it
/// surfaces at the temporal-assertion site as a per-sample error
/// the caller decides how to handle.
///
/// The label, tags, and per-sample timestamps are carried so
/// failure-path messages name the offending sample without the
/// caller re-threading the series. Tags and elapsed-ms vectors
/// are always the same length as `values`.
#[derive(Debug, Clone)]
#[must_use = "SeriesField records nothing until a temporal pattern is invoked"]
pub struct SeriesField<T> {
    label: &'static str,
    tags: Vec<String>,
    elapsed_ms: Vec<u64>,
    values: Vec<SnapshotResult<T>>,
}

impl<T> SeriesField<T> {
    /// Build a new field. Internal — projection helpers in
    /// [`crate::scenario::sample`] call this on the series side.
    pub fn from_parts(
        label: &'static str,
        tags: Vec<String>,
        elapsed_ms: Vec<u64>,
        values: Vec<SnapshotResult<T>>,
    ) -> Self {
        debug_assert_eq!(tags.len(), values.len());
        debug_assert_eq!(elapsed_ms.len(), values.len());
        Self {
            label,
            tags,
            elapsed_ms,
            values,
        }
    }

    /// Label for failure-message rendering.
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Number of samples in the field.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when no samples are present.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Tag of sample `i`, panics on out-of-range. Used by the
    /// temporal-assertion failure-message rendering.
    pub fn tag(&self, i: usize) -> &str {
        self.tags[i].as_str()
    }

    /// Elapsed-ms timestamp of sample `i`.
    pub fn elapsed_ms(&self, i: usize) -> u64 {
        self.elapsed_ms[i]
    }

    /// Iterate over per-sample values (each a [`SnapshotResult<T>`]).
    pub fn values_iter(&self) -> impl Iterator<Item = &SnapshotResult<T>> {
        self.values.iter()
    }

    /// Open a per-sample claim builder for scalar comparators
    /// (`at_least`, `at_most`, `between`). Each successful sample
    /// runs the comparator independently; the first failure
    /// records a detail; subsequent failures pile on so the
    /// timeline shows every offending sample, not just the first.
    /// Borrows the verdict mutably for the duration of the
    /// comparator chain.
    pub fn each<'v>(&self, verdict: &'v mut Verdict) -> EachClaim<'_, 'v, T> {
        EachClaim {
            field: self,
            verdict,
        }
    }
}

/// Per-sample scalar claim builder returned by
/// [`SeriesField::each`]. Provides `at_least` / `at_most` /
/// `between` — comparators that apply to every (successfully
/// projected) sample independently. Per-sample errors from the
/// projection (missing field, type mismatch) are routed through
/// the verdict as failures so coverage gaps are never silent.
#[must_use = "EachClaim records nothing until a comparator is invoked"]
pub struct EachClaim<'f, 'v, T> {
    field: &'f SeriesField<T>,
    verdict: &'v mut Verdict,
}

impl<'f, 'v, T> EachClaim<'f, 'v, T>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    /// Pass when every sample's value satisfies `value >= floor`.
    /// Per-sample errors and per-sample violations both record a
    /// [`DetailKind::Temporal`] detail and flip the verdict to
    /// failed; the chain returns the verdict so further claims
    /// can stack.
    ///
    /// On `T = f64`, an incomparable value (NaN) is a failure: a
    /// NaN sample silently passing `value < floor`/`value > ceiling`
    /// (which IEEE-754 semantics give you on raw `<`/`>`) would
    /// hide a coverage gap, so the pattern uses `partial_cmp` and
    /// reports the offending sample distinctly.
    pub fn at_least(self, floor: T) -> &'v mut Verdict {
        let label = self.field.label;
        for (i, slot) in self.field.values.iter().enumerate() {
            match slot {
                Ok(v) => match v.partial_cmp(&floor) {
                    Some(std::cmp::Ordering::Less) => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_least {floor}): sample {tag} (+{elapsed_ms}ms): \
                             value {v}",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    ),
                    None => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_least {floor}): sample {tag} (+{elapsed_ms}ms): \
                             value {v} is incomparable (NaN)",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    ),
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Greater) => {}
                },
                Err(e) => {
                    push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_least {floor}): sample {tag} (+{elapsed_ms}ms): \
                             projection error: {e}",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    );
                }
            }
        }
        self.verdict
    }

    /// Pass when every sample's value satisfies `value <= ceiling`.
    /// NaN samples (on `T = f64`) report an incomparable failure
    /// for the same reason documented on [`Self::at_least`].
    pub fn at_most(self, ceiling: T) -> &'v mut Verdict {
        let label = self.field.label;
        for (i, slot) in self.field.values.iter().enumerate() {
            match slot {
                Ok(v) => match v.partial_cmp(&ceiling) {
                    Some(std::cmp::Ordering::Greater) => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_most {ceiling}): sample {tag} (+{elapsed_ms}ms): \
                             value {v}",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    ),
                    None => push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_most {ceiling}): sample {tag} (+{elapsed_ms}ms): \
                             value {v} is incomparable (NaN)",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    ),
                    Some(std::cmp::Ordering::Equal | std::cmp::Ordering::Less) => {}
                },
                Err(e) => {
                    push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.at_most {ceiling}): sample {tag} (+{elapsed_ms}ms): \
                             projection error: {e}",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    );
                }
            }
        }
        self.verdict
    }

    /// Pass when every sample's value satisfies `lo <= value <= hi`.
    /// Caller error (`lo > hi`) lands as a single
    /// [`DetailKind::Temporal`] detail rather than evaluating each
    /// sample against an inverted range. NaN samples report an
    /// incomparable failure (see [`Self::at_least`]).
    pub fn between(self, lo: T, hi: T) -> &'v mut Verdict {
        let label = self.field.label;
        if lo > hi {
            push_detail(
                self.verdict,
                format!("{label} (each.between): caller error: lo={lo} > hi={hi}"),
            );
            return self.verdict;
        }
        for (i, slot) in self.field.values.iter().enumerate() {
            match slot {
                Ok(v) => {
                    let lo_cmp = v.partial_cmp(&lo);
                    let hi_cmp = v.partial_cmp(&hi);
                    if lo_cmp.is_none() || hi_cmp.is_none() {
                        push_detail(
                            self.verdict,
                            format!(
                                "{label} (each.between [{lo}, {hi}]): sample {tag} \
                                 (+{elapsed_ms}ms): value {v} is incomparable (NaN)",
                                tag = self.field.tags[i],
                                elapsed_ms = self.field.elapsed_ms[i],
                            ),
                        );
                    } else if matches!(lo_cmp, Some(std::cmp::Ordering::Less))
                        || matches!(hi_cmp, Some(std::cmp::Ordering::Greater))
                    {
                        push_detail(
                            self.verdict,
                            format!(
                                "{label} (each.between [{lo}, {hi}]): sample {tag} \
                                 (+{elapsed_ms}ms): value {v}",
                                tag = self.field.tags[i],
                                elapsed_ms = self.field.elapsed_ms[i],
                            ),
                        );
                    }
                }
                Err(e) => {
                    push_detail(
                        self.verdict,
                        format!(
                            "{label} (each.between [{lo}, {hi}]): sample {tag} \
                             (+{elapsed_ms}ms): projection error: {e}",
                            tag = self.field.tags[i],
                            elapsed_ms = self.field.elapsed_ms[i],
                        ),
                    );
                }
            }
        }
        self.verdict
    }
}

// ----- Six temporal patterns -----

impl<T> SeriesField<T>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    /// Pass when every consecutive pair satisfies
    /// `values[i] <= values[i+1]`. A common shape for kernel
    /// counters whose only legal direction is up. Per-sample
    /// projection errors are SKIPPED — the affected pair-comparison
    /// is dropped, the skip count is logged as a verdict Note so
    /// coverage gaps stay visible, and the verdict is NOT flipped
    /// on a missing-sample condition (which is structurally
    /// missing data, not a regression). Adjacent samples on
    /// either side of a gap are still checked against each other.
    pub fn nondecreasing<'v>(&self, verdict: &'v mut Verdict) -> &'v mut Verdict {
        self.monotonicity_check(verdict, false)
    }

    /// Pass when every consecutive pair satisfies
    /// `values[i] < values[i+1]`. Useful for counters that MUST
    /// advance every period (e.g. a heartbeat tick). Same skip-on-
    /// projection-error semantics as [`Self::nondecreasing`].
    pub fn strictly_increasing<'v>(&self, verdict: &'v mut Verdict) -> &'v mut Verdict {
        self.monotonicity_check(verdict, true)
    }

    fn monotonicity_check<'v>(&self, verdict: &'v mut Verdict, strict: bool) -> &'v mut Verdict {
        let pat = if strict {
            "strictly_increasing"
        } else {
            "nondecreasing"
        };
        if self.values.len() < 2 {
            // Vacuous: 0 or 1 samples cannot violate monotonicity.
            // Surface an informational note via the verdict's
            // notes path so the timeline summary records that the
            // pattern was opened against an under-populated
            // series — without this, a bug that drops every
            // periodic capture would silently pass every
            // monotonicity claim.
            verdict.note(format!(
                "{label} ({pat}): only {n} samples — pattern vacuously holds; \
                 ensure num_snapshots >= 2 for meaningful coverage",
                label = self.label,
                n = self.values.len(),
            ));
            return verdict;
        }
        // Per-sample projection errors are NOT temporal failures —
        // they indicate the underlying field was missing on that
        // sample (e.g. placeholder report from a freeze-rendezvous
        // timeout). Skip the affected pair-comparisons and surface
        // the skip count as a Note on the verdict so a coverage
        // gap is visible without flipping a temporal pattern that
        // is structurally about value monotonicity. The compare
        // proceeds across the rest of the series without bridging
        // the gap (a gap means we cannot conclude anything about
        // monotonicity ACROSS the missing sample, only on either
        // side of it).
        let mut skipped: Vec<String> = Vec::new();
        for i in 0..self.values.len() - 1 {
            let left = match &self.values[i] {
                Ok(v) => v,
                Err(_) => {
                    skipped.push(format!(
                        "{tag}(+{elapsed_ms}ms)",
                        tag = self.tags[i],
                        elapsed_ms = self.elapsed_ms[i],
                    ));
                    continue;
                }
            };
            let right = match &self.values[i + 1] {
                Ok(v) => v,
                Err(_) => {
                    // Skip recorded when the (i+1) slot becomes
                    // the `i` slot of the next iteration; avoid
                    // double-counting by only logging on the
                    // forward-edge here.
                    continue;
                }
            };
            let ok = if strict { right > left } else { right >= left };
            if !ok {
                push_detail(
                    verdict,
                    format!(
                        "{label} ({pat}): regression at sample {tag} (+{elapsed_ms}ms): \
                         value {right} after prior value {left} at sample {prev_tag} \
                         (+{prev_elapsed}ms)",
                        label = self.label,
                        tag = self.tags[i + 1],
                        elapsed_ms = self.elapsed_ms[i + 1],
                        prev_tag = self.tags[i],
                        prev_elapsed = self.elapsed_ms[i],
                    ),
                );
            }
        }
        // The final sample's err state was not visited by the
        // loop's left-arm; check it explicitly so the skip count
        // is exhaustive.
        if let Some(last) = self.values.last()
            && last.is_err()
        {
            let i = self.values.len() - 1;
            skipped.push(format!(
                "{tag}(+{elapsed_ms}ms)",
                tag = self.tags[i],
                elapsed_ms = self.elapsed_ms[i],
            ));
        }
        if !skipped.is_empty() {
            verdict.note(format!(
                "{label} ({pat}): skipped {n} sample(s) with projection errors: \
                 {samples}",
                label = self.label,
                n = skipped.len(),
                samples = skipped.join(", "),
            ));
        }
        verdict
    }
}

impl SeriesField<f64> {
    /// Pass when every consecutive (delta_value / delta_ms) lies
    /// in `[lo, hi]`. The rate is computed with millisecond
    /// resolution from the per-sample elapsed-ms timestamps, so
    /// a counter that should advance at ~1 unit/ms reads cleanly
    /// as `rate_within(0.5, 2.0)`. A zero-time delta between
    /// adjacent samples lands as a caller-side or framework
    /// failure (samples too close to compute a rate); the detail
    /// names the offending pair.
    pub fn rate_within<'v>(&self, verdict: &'v mut Verdict, lo: f64, hi: f64) -> &'v mut Verdict {
        if lo > hi {
            push_detail(
                verdict,
                format!(
                    "{label} (rate_within): caller error: lo={lo} > hi={hi}",
                    label = self.label,
                ),
            );
            return verdict;
        }
        if self.values.len() < 2 {
            verdict.note(format!(
                "{label} (rate_within): only {n} samples — pattern vacuously holds",
                label = self.label,
                n = self.values.len(),
            ));
            return verdict;
        }
        // Per-sample projection errors are treated as GAPS — no
        // rate is computed across the gap. Log how many gaps were
        // encountered as a Note so a coverage problem is visible
        // without flipping the verdict on what is structurally a
        // missing-data condition, not a rate violation.
        let mut gap_pairs: usize = 0;
        for i in 0..self.values.len() - 1 {
            let (left, right) = match (&self.values[i], &self.values[i + 1]) {
                (Ok(l), Ok(r)) => (*l, *r),
                _ => {
                    gap_pairs += 1;
                    continue;
                }
            };
            let dt_ms = self.elapsed_ms[i + 1].saturating_sub(self.elapsed_ms[i]) as f64;
            if dt_ms <= 0.0 {
                push_detail(
                    verdict,
                    format!(
                        "{label} (rate_within): zero-time delta between sample {prev_tag} \
                         (+{prev_elapsed}ms) and {tag} (+{elapsed_ms}ms) — cannot compute rate",
                        label = self.label,
                        prev_tag = self.tags[i],
                        prev_elapsed = self.elapsed_ms[i],
                        tag = self.tags[i + 1],
                        elapsed_ms = self.elapsed_ms[i + 1],
                    ),
                );
                continue;
            }
            let rate = (right - left) / dt_ms;
            // NaN can arise from inf-inf or NaN endpoints; raw `<`/`>`
            // against NaN is always false, so a NaN rate would
            // silently slip past the band check. Infinite rates
            // (inf endpoint, or finite endpoints whose difference
            // overflows f64) are also an upstream data corruption
            // signal — caller has no use for the band comparison
            // when the value is non-finite. Both cases get a
            // structured detail naming the sample pair so the
            // operator sees the offending span.
            if !rate.is_finite() {
                push_detail(
                    verdict,
                    format!(
                        "{label} (rate_within [{lo}, {hi}]): non-finite rate between \
                         samples {prev_tag} (+{prev_elapsed}ms, value {left}) and \
                         {tag} (+{elapsed_ms}ms, value {right}) — endpoint is NaN \
                         or produced inf in the delta",
                        label = self.label,
                        prev_tag = self.tags[i],
                        prev_elapsed = self.elapsed_ms[i],
                        tag = self.tags[i + 1],
                        elapsed_ms = self.elapsed_ms[i + 1],
                    ),
                );
            } else if rate < lo || rate > hi {
                push_detail(
                    verdict,
                    format!(
                        "{label} (rate_within [{lo}, {hi}]): rate {rate:.4}/ms between \
                         samples {prev_tag} (+{prev_elapsed}ms, value {left}) and \
                         {tag} (+{elapsed_ms}ms, value {right})",
                        label = self.label,
                        prev_tag = self.tags[i],
                        prev_elapsed = self.elapsed_ms[i],
                        tag = self.tags[i + 1],
                        elapsed_ms = self.elapsed_ms[i + 1],
                    ),
                );
            }
        }
        if gap_pairs > 0 {
            verdict.note(format!(
                "{label} (rate_within): {gap_pairs} consecutive-pair gap(s) skipped \
                 due to projection errors on at least one endpoint",
                label = self.label,
            ));
        }
        verdict
    }

    /// Pass when every post-warmup sample (`elapsed_ms >=
    /// warmup_ms`) lies inside `mean·(1-tolerance), mean·(1+tolerance)`.
    /// `tolerance` is a fraction (0.10 = ±10%). The mean is
    /// computed over the post-warmup samples only — the warmup
    /// region is excluded so ramp-up does not bias the steady-
    /// state baseline. Per-sample projection errors are SKIPPED
    /// (with a verdict Note logging the count and tags); they are
    /// treated as gaps in coverage, not band violations, so a
    /// missing post-warmup sample does not flip the verdict.
    pub fn steady_within<'v>(
        &self,
        verdict: &'v mut Verdict,
        warmup_ms: u64,
        tolerance: f64,
    ) -> &'v mut Verdict {
        if tolerance < 0.0 {
            push_detail(
                verdict,
                format!(
                    "{label} (steady_within): caller error: tolerance {tolerance} negative",
                    label = self.label,
                ),
            );
            return verdict;
        }
        let mut active: Vec<(usize, f64)> = Vec::new();
        let mut skipped: Vec<String> = Vec::new();
        // Track whether any sample's elapsed_ms reached or exceeded
        // warmup_ms — distinguishes "warmup window absorbed every
        // sample" (genuine vacuous-pass) from "post-warmup samples
        // existed but all errored" (skip-Note already covers it).
        let mut any_post_warmup = false;
        for (i, slot) in self.values.iter().enumerate() {
            if self.elapsed_ms[i] < warmup_ms {
                continue;
            }
            any_post_warmup = true;
            match slot {
                Ok(v) => active.push((i, *v)),
                // Per-sample projection errors are treated as
                // gaps: a missing post-warmup sample cannot
                // violate the steady-state band (we have no value
                // to compare). Surface the skip count via a Note
                // so a coverage hole is visible without flipping
                // the verdict on what is structurally missing
                // data, not a band violation.
                Err(_) => skipped.push(format!(
                    "{tag}(+{elapsed_ms}ms)",
                    tag = self.tags[i],
                    elapsed_ms = self.elapsed_ms[i],
                )),
            }
        }
        if !skipped.is_empty() {
            verdict.note(format!(
                "{label} (steady_within): skipped {n} post-warmup sample(s) with \
                 projection errors: {samples}",
                label = self.label,
                n = skipped.len(),
                samples = skipped.join(", "),
            ));
        }
        if active.is_empty() {
            // Only emit the vacuous-warmup Note when the warmup
            // window genuinely absorbed every sample (no
            // post-warmup samples existed). When post-warmup
            // samples existed but all errored, the
            // skipped-with-projection-errors Note above already
            // explained the empty active set; emitting a second
            // Note here would falsely claim "no samples beyond
            // warmup".
            if !any_post_warmup {
                verdict.note(format!(
                    "{label} (steady_within): no samples beyond warmup_ms={warmup_ms} — \
                     pattern vacuously holds",
                    label = self.label,
                ));
            }
            return verdict;
        }
        let mean: f64 = active.iter().map(|(_, v)| *v).sum::<f64>() / (active.len() as f64);
        let lo = mean * (1.0 - tolerance);
        let hi = mean * (1.0 + tolerance);
        // For negative means (pathological), the multiplication
        // flips the band; protect by sorting.
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        for (i, v) in active {
            if v < lo || v > hi {
                push_detail(
                    verdict,
                    format!(
                        "{label} (steady_within mean {mean:.4} ±{pct:.1}%): \
                         sample {tag} (+{elapsed_ms}ms): value {v} outside [{lo:.4}, {hi:.4}]",
                        label = self.label,
                        pct = tolerance * 100.0,
                        tag = self.tags[i],
                        elapsed_ms = self.elapsed_ms[i],
                    ),
                );
            }
        }
        verdict
    }

    /// Pass when three consecutive samples land inside
    /// `[target-tolerance, target+tolerance]` AT OR BEFORE
    /// `deadline_ms`. The intent is "the system stabilizes near
    /// `target` by the deadline" — three consecutive in-band
    /// samples are the convergence-witness shape. Failures fire
    /// when the deadline passes without a witness.
    pub fn converges_to<'v>(
        &self,
        verdict: &'v mut Verdict,
        target: f64,
        tolerance: f64,
        deadline_ms: u64,
    ) -> &'v mut Verdict {
        if tolerance < 0.0 {
            push_detail(
                verdict,
                format!(
                    "{label} (converges_to): caller error: tolerance {tolerance} negative",
                    label = self.label,
                ),
            );
            return verdict;
        }
        // Pre-check: counting all successfully-projected samples
        // (within the deadline window) do we have enough evidence
        // to even attempt a 3-consecutive witness? When fewer
        // than 3 successfully-projected samples exist before the
        // deadline, record an explicit Note (NOT a verdict
        // failure) and return — absence of data is a coverage gap
        // surfaced for the operator, not a negative finding the
        // verdict should fail on. Distinguishes "did not collect
        // enough samples" (Note here) from "collected enough
        // samples but never converged" (the no-witness Temporal
        // detail emitted below by the witness loop).
        let projected_count: usize = self
            .values
            .iter()
            .enumerate()
            .filter(|(i, slot)| self.elapsed_ms[*i] <= deadline_ms && slot.is_ok())
            .count();
        if projected_count < 3 {
            verdict.note(format!(
                "{label} (converges_to {target} ±{tolerance}, deadline_ms={deadline_ms}): \
                 insufficient samples for converges_to (need ≥3, have {projected_count})",
                label = self.label,
            ));
            return verdict;
        }
        let lo = target - tolerance;
        let hi = target + tolerance;
        let mut consecutive: usize = 0;
        let mut witness_idx: Option<usize> = None;
        for (i, slot) in self.values.iter().enumerate() {
            if self.elapsed_ms[i] > deadline_ms {
                break;
            }
            match slot {
                Ok(v) => {
                    if *v >= lo && *v <= hi {
                        consecutive += 1;
                        if consecutive >= 3 {
                            witness_idx = Some(i);
                            break;
                        }
                    } else {
                        consecutive = 0;
                    }
                }
                Err(_) => consecutive = 0,
            }
        }
        if witness_idx.is_none() {
            push_detail(
                verdict,
                format!(
                    "{label} (converges_to {target} ±{tolerance}, deadline_ms={deadline_ms}): \
                     no 3-consecutive-in-band witness before deadline ({n} samples evaluated)",
                    label = self.label,
                    n = self.values.len(),
                ),
            );
        }
        verdict
    }

    /// Pass when every consecutive `(self_value / other_value)`
    /// lies in `[lo, hi]`. Cross-field correlation: e.g. ensure a
    /// per-cgroup utilization always tracks a per-cgroup runtime
    /// within a fixed band. The two series MUST have matching
    /// length and tags; mismatches fire a single caller-error
    /// detail. Per-sample projection errors on EITHER lhs or rhs
    /// are SKIPPED — the affected pair is dropped, the skip count
    /// is logged as a verdict Note, and the verdict is NOT flipped
    /// on missing-data conditions.
    pub fn ratio_within<'v>(
        &self,
        verdict: &'v mut Verdict,
        other: &SeriesField<f64>,
        lo: f64,
        hi: f64,
    ) -> &'v mut Verdict {
        if lo > hi {
            push_detail(
                verdict,
                format!(
                    "{label} (ratio_within): caller error: lo={lo} > hi={hi}",
                    label = self.label,
                ),
            );
            return verdict;
        }
        if self.values.len() != other.values.len() {
            push_detail(
                verdict,
                format!(
                    "{label} (ratio_within {other}): caller error: length mismatch \
                     (this {n}, other {m})",
                    label = self.label,
                    other = other.label,
                    n = self.values.len(),
                    m = other.values.len(),
                ),
            );
            return verdict;
        }
        // Per-sample projection errors on either lhs or rhs are
        // treated as gaps — no ratio is computed across the pair.
        // Surface skip count as a Note so a coverage hole is
        // visible without flipping the verdict on what is
        // structurally missing data.
        let mut gap_pairs: usize = 0;
        for (i, (lhs_slot, rhs_slot)) in self.values.iter().zip(other.values.iter()).enumerate() {
            let (lhs, rhs) = match (lhs_slot, rhs_slot) {
                (Ok(l), Ok(r)) => (*l, *r),
                _ => {
                    gap_pairs += 1;
                    continue;
                }
            };
            if rhs == 0.0 {
                push_detail(
                    verdict,
                    format!(
                        "{label} (ratio_within): rhs == 0 at sample {tag} (+{elapsed_ms}ms) — \
                         cannot compute ratio",
                        label = self.label,
                        tag = self.tags[i],
                        elapsed_ms = self.elapsed_ms[i],
                    ),
                );
                continue;
            }
            let ratio = lhs / rhs;
            if ratio < lo || ratio > hi {
                push_detail(
                    verdict,
                    format!(
                        "{label} (ratio_within {other_label} [{lo}, {hi}]): \
                         ratio {ratio:.4} at sample {tag} (+{elapsed_ms}ms) — \
                         lhs={lhs} rhs={rhs}",
                        label = self.label,
                        other_label = other.label,
                        tag = self.tags[i],
                        elapsed_ms = self.elapsed_ms[i],
                    ),
                );
            }
        }
        if gap_pairs > 0 {
            verdict.note(format!(
                "{label} (ratio_within): {gap_pairs} pair(s) skipped due to projection \
                 errors on lhs or rhs",
                label = self.label,
            ));
        }
        verdict
    }
}

impl SeriesField<bool> {
    /// Pass when every sample's value is `true`. Per-sample
    /// projection errors fail the assertion. Use for boolean
    /// invariants — e.g. "scheduler is alive at every periodic
    /// boundary" projected as `snap.var("scheduler_alive").as_bool()`.
    pub fn always_true<'v>(&self, verdict: &'v mut Verdict) -> &'v mut Verdict {
        for (i, slot) in self.values.iter().enumerate() {
            match slot {
                Ok(v) => {
                    if !*v {
                        push_detail(
                            verdict,
                            format!(
                                "{label} (always_true): sample {tag} (+{elapsed_ms}ms): \
                                 value false",
                                label = self.label,
                                tag = self.tags[i],
                                elapsed_ms = self.elapsed_ms[i],
                            ),
                        );
                    }
                }
                Err(e) => {
                    push_detail(
                        verdict,
                        format!(
                            "{label} (always_true): sample {tag} (+{elapsed_ms}ms): \
                             projection error: {e}",
                            label = self.label,
                            tag = self.tags[i],
                            elapsed_ms = self.elapsed_ms[i],
                        ),
                    );
                }
            }
        }
        verdict
    }
}

fn push_detail(verdict: &mut Verdict, message: String) {
    let result = verdict.result_mut();
    result.passed = false;
    result
        .details
        .push(AssertDetail::new(DetailKind::Temporal, message));
}

// Bridge into Verdict's internal AssertResult — added below as an
// associated method on Verdict so the temporal module does not
// reach into internals from a sibling.

#[allow(dead_code)]
fn _silence_snapshot_error_import(_: SnapshotError) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::sample::SampleSeries;
    use crate::scenario::snapshot::{SnapshotError, SnapshotResult};

    fn synthetic_field<T: Copy>(label: &'static str, values: Vec<(u64, T)>) -> SeriesField<T> {
        let tags: Vec<String> = (0..values.len())
            .map(|i| format!("periodic_{i:03}"))
            .collect();
        let elapsed: Vec<u64> = values.iter().map(|(t, _)| *t).collect();
        let vals: Vec<SnapshotResult<T>> = values.into_iter().map(|(_, v)| Ok(v)).collect();
        SeriesField::from_parts(label, tags, elapsed, vals)
    }

    #[test]
    fn nondecreasing_passes_on_monotonic_series() {
        let f = synthetic_field("counter", vec![(100, 1u64), (200, 2u64), (300, 3u64)]);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        assert!(v.passed());
    }

    #[test]
    fn nondecreasing_fails_on_regression() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 3u64)]);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(r.details.iter().any(|d| d.kind == DetailKind::Temporal));
        assert!(r.details.iter().any(|d| d.message.contains("counter")));
    }

    #[test]
    fn strictly_increasing_fails_on_plateau() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 5u64)]);
        let mut v = Verdict::new();
        f.strictly_increasing(&mut v);
        let r = v.into_result();
        assert!(!r.passed);
    }

    #[test]
    fn rate_within_in_band_passes() {
        // Counter advances 1 unit per 100ms = 0.01/ms.
        let f = synthetic_field("ticks", vec![(100, 1.0f64), (200, 2.0f64), (300, 3.0f64)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.005, 0.02);
        assert!(v.passed());
    }

    #[test]
    fn rate_within_out_of_band_fails() {
        let f = synthetic_field("ticks", vec![(100, 1.0f64), (200, 100.0f64)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 0.5);
        assert!(!v.passed());
    }

    #[test]
    fn steady_within_skips_warmup_and_passes() {
        // Warmup at +0..200ms; steady at 10.0 from +300..500.
        let f = synthetic_field(
            "util",
            vec![
                (100, 100.0f64),
                (200, 50.0f64),
                (300, 10.0f64),
                (400, 10.0f64),
                (500, 10.0f64),
            ],
        );
        let mut v = Verdict::new();
        f.steady_within(&mut v, 250, 0.01);
        assert!(v.passed(), "{:?}", v.into_result().details);
    }

    #[test]
    fn steady_within_post_warmup_outlier_fails() {
        let f = synthetic_field("util", vec![(300, 10.0f64), (400, 10.0f64), (500, 50.0f64)]);
        let mut v = Verdict::new();
        f.steady_within(&mut v, 0, 0.10);
        assert!(!v.passed());
    }

    #[test]
    fn converges_to_finds_witness() {
        let f = synthetic_field(
            "load",
            vec![
                (100, 10.0f64),
                (200, 5.0f64),
                (300, 1.0f64),
                (400, 1.0f64),
                (500, 1.0f64),
            ],
        );
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 1000);
        assert!(v.passed());
    }

    #[test]
    fn converges_to_no_witness_fails() {
        let f = synthetic_field("load", vec![(100, 10.0f64), (200, 10.0f64), (300, 10.0f64)]);
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 500);
        assert!(!v.passed());
    }

    #[test]
    fn always_true_passes_on_all_true() {
        let f = synthetic_field("alive", vec![(100, true), (200, true)]);
        let mut v = Verdict::new();
        f.always_true(&mut v);
        assert!(v.passed());
    }

    #[test]
    fn always_true_fails_on_false() {
        let f = synthetic_field("alive", vec![(100, true), (200, false)]);
        let mut v = Verdict::new();
        f.always_true(&mut v);
        assert!(!v.passed());
    }

    #[test]
    fn ratio_within_in_band_passes() {
        let lhs = synthetic_field("lhs", vec![(100, 10.0f64), (200, 20.0f64), (300, 30.0f64)]);
        let rhs = synthetic_field("rhs", vec![(100, 5.0f64), (200, 10.0f64), (300, 15.0f64)]);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        assert!(v.passed());
    }

    #[test]
    fn ratio_within_length_mismatch_fails_caller_error() {
        let lhs = synthetic_field("lhs", vec![(100, 10.0f64)]);
        let rhs = synthetic_field("rhs", vec![(100, 5.0f64), (200, 10.0f64)]);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        assert!(!v.passed());
    }

    #[test]
    fn each_at_least_passes() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 7u64)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_least(3u64);
        assert!(v.passed());
    }

    #[test]
    fn each_at_most_fails_on_outlier() {
        let f = synthetic_field("counter", vec![(100, 5u64), (200, 99u64)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_most(10u64);
        assert!(!v.passed());
    }

    #[test]
    fn each_propagates_per_sample_projection_error() {
        let tags = vec!["periodic_000".to_string(), "periodic_001".to_string()];
        let elapsed = vec![100u64, 200u64];
        let values: Vec<SnapshotResult<u64>> = vec![
            Ok(5u64),
            Err(SnapshotError::VarNotFound {
                requested: "missing".to_string(),
                available: vec!["a".to_string()],
            }),
        ];
        let f = SeriesField::from_parts("x", tags, elapsed, values);
        let mut v = Verdict::new();
        f.each(&mut v).at_least(1u64);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(
            r.details
                .iter()
                .any(|d| d.message.contains("projection error"))
        );
    }

    /// Vacuous holding when num_snapshots < 2 records a Note, not a
    /// failure.
    #[test]
    fn nondecreasing_with_one_sample_records_note() {
        let f = synthetic_field("counter", vec![(100, 1u64)]);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(r.passed);
        assert!(r.details.iter().any(|d| d.kind == DetailKind::Note));
    }

    /// End-to-end sample: sanity-check that a series projection
    /// flowing through a temporal pattern produces a coherent
    /// verdict. The `SampleSeries` shape exercise lives in
    /// `src/scenario/sample.rs`; this test only confirms the
    /// integration handshake works.
    #[test]
    fn series_projection_into_temporal_pattern_smoke_check() {
        // Empty series — every pattern should be vacuously ok.
        let series = SampleSeries::empty();
        let field = series.bpf("x", |snap| snap.var("missing").as_u64());
        let mut v = Verdict::new();
        field.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(r.passed);
    }

    // ---- Skip-on-projection-error semantics ----

    /// nondecreasing skips errored samples, logs skip count, does
    /// NOT flip the verdict on missing data.
    #[test]
    fn nondecreasing_skips_projection_errors_with_note() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let values: Vec<SnapshotResult<u64>> = vec![
            Ok(1u64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(2u64),
        ];
        let f = SeriesField::from_parts("counter", tags, elapsed, values);
        let mut v = Verdict::new();
        f.nondecreasing(&mut v);
        let r = v.into_result();
        assert!(
            r.passed,
            "nondecreasing must NOT flip on projection error: {:?}",
            r.details
        );
        assert!(
            r.details.iter().any(|d| d.kind == DetailKind::Note
                && d.message.contains("skipped 1 sample")
                && d.message.contains("periodic_001")),
            "expected skip Note: {:?}",
            r.details
        );
    }

    /// rate_within treats errored samples as gaps (no rate
    /// computed across the gap), records skip count via a Note.
    #[test]
    fn rate_within_skips_gaps_with_note() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let values: Vec<SnapshotResult<f64>> = vec![
            Ok(1.0f64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(2.0f64),
        ];
        let f = SeriesField::from_parts("ticks", tags, elapsed, values);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 1.0);
        let r = v.into_result();
        assert!(
            r.passed,
            "rate_within must NOT flip on gap: {:?}",
            r.details
        );
        assert!(
            r.details
                .iter()
                .any(|d| d.kind == DetailKind::Note && d.message.contains("gap")),
            "expected gap Note: {:?}",
            r.details
        );
    }

    /// steady_within skips errored post-warmup samples, records a
    /// Note, does NOT flip the verdict on missing data.
    #[test]
    fn steady_within_skips_projection_errors_with_note() {
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![300u64, 400u64, 500u64];
        let values: Vec<SnapshotResult<f64>> = vec![
            Ok(10.0f64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(10.0f64),
        ];
        let f = SeriesField::from_parts("util", tags, elapsed, values);
        let mut v = Verdict::new();
        f.steady_within(&mut v, 0, 0.10);
        let r = v.into_result();
        assert!(r.passed, "{:?}", r.details);
        assert!(
            r.details.iter().any(|d| d.kind == DetailKind::Note
                && d.message.contains("skipped")
                && d.message.contains("periodic_001")),
            "expected skip Note: {:?}",
            r.details
        );
    }

    /// ratio_within skips pairs where either side errored, records
    /// gap count, does NOT flip on missing data.
    #[test]
    fn ratio_within_skips_gaps_with_note() {
        let lhs_values: Vec<SnapshotResult<f64>> = vec![
            Ok(10.0f64),
            Err(SnapshotError::VarNotFound {
                requested: "x".to_string(),
                available: vec![],
            }),
            Ok(20.0f64),
        ];
        let rhs_values: Vec<SnapshotResult<f64>> = vec![Ok(5.0f64), Ok(7.0f64), Ok(10.0f64)];
        let tags = vec![
            "periodic_000".to_string(),
            "periodic_001".to_string(),
            "periodic_002".to_string(),
        ];
        let elapsed = vec![100u64, 200u64, 300u64];
        let lhs = SeriesField::from_parts("lhs", tags.clone(), elapsed.clone(), lhs_values);
        let rhs = SeriesField::from_parts("rhs", tags, elapsed, rhs_values);
        let mut v = Verdict::new();
        lhs.ratio_within(&mut v, &rhs, 1.5, 2.5);
        let r = v.into_result();
        assert!(r.passed, "{:?}", r.details);
        assert!(
            r.details
                .iter()
                .any(|d| d.kind == DetailKind::Note && d.message.contains("1 pair")),
            "expected gap Note: {:?}",
            r.details
        );
    }

    /// converges_to with fewer than 3 successfully-projected
    /// samples in window records an explicit Note (not a verdict
    /// failure) — absence of data is a coverage gap, not a
    /// negative finding. The Note message names the count and the
    /// requirement so an operator can distinguish "did not collect
    /// enough samples" from "collected enough samples but never
    /// converged".
    #[test]
    fn converges_to_insufficient_samples_records_note() {
        let f = synthetic_field("load", vec![(100, 1.0f64), (200, 1.0f64)]);
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 1000);
        let r = v.into_result();
        assert!(
            r.passed,
            "insufficient-samples must NOT flip the verdict: {:?}",
            r.details
        );
        assert!(
            r.details.iter().any(|d| d.kind == DetailKind::Note
                && d.message.contains("insufficient samples")
                && d.message.contains("need ≥3, have 2")),
            "expected insufficient-samples Note with count: {:?}",
            r.details
        );
    }

    /// converges_to with 3+ samples in window but none in band
    /// produces the "no witness" structured failure (the
    /// pre-existing code path), distinct from the
    /// insufficient-samples message.
    #[test]
    fn converges_to_no_witness_distinct_from_insufficient() {
        let f = synthetic_field(
            "load",
            vec![
                (100, 10.0f64),
                (200, 10.0f64),
                (300, 10.0f64),
                (400, 10.0f64),
            ],
        );
        let mut v = Verdict::new();
        f.converges_to(&mut v, 1.0, 0.5, 1000);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(
            r.details
                .iter()
                .any(|d| d.message.contains("no 3-consecutive-in-band witness")),
            "expected no-witness message: {:?}",
            r.details
        );
        assert!(
            !r.details
                .iter()
                .any(|d| d.message.contains("insufficient samples")),
            "must NOT report insufficient-samples when there ARE enough samples: {:?}",
            r.details
        );
    }

    // ---- NaN handling ----

    /// each.at_least on NaN sample reports an incomparable
    /// failure rather than silently passing the comparison.
    /// Without the partial_cmp fix, IEEE-754 `<` against NaN
    /// is always false, so a NaN sample would silently pass
    /// `at_least(0.0)`.
    #[test]
    fn each_at_least_flags_nan_sample() {
        let f = synthetic_field("util", vec![(100, 50.0f64), (200, f64::NAN)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_least(0.0f64);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(
            r.details
                .iter()
                .any(|d| d.message.contains("NaN") && d.message.contains("periodic_001")),
            "expected NaN failure naming the sample: {:?}",
            r.details
        );
    }

    /// each.at_most on NaN sample reports an incomparable failure.
    #[test]
    fn each_at_most_flags_nan_sample() {
        let f = synthetic_field("util", vec![(100, 50.0f64), (200, f64::NAN)]);
        let mut v = Verdict::new();
        f.each(&mut v).at_most(100.0f64);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(
            r.details
                .iter()
                .any(|d| d.message.contains("NaN") && d.message.contains("periodic_001")),
            "expected NaN failure naming the sample: {:?}",
            r.details
        );
    }

    /// each.between on NaN sample reports an incomparable failure.
    #[test]
    fn each_between_flags_nan_sample() {
        let f = synthetic_field("util", vec![(100, 50.0f64), (200, f64::NAN)]);
        let mut v = Verdict::new();
        f.each(&mut v).between(0.0f64, 100.0f64);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(
            r.details
                .iter()
                .any(|d| d.message.contains("NaN") && d.message.contains("periodic_001")),
            "expected NaN failure naming the sample: {:?}",
            r.details
        );
    }

    /// rate_within reports a non-finite-rate failure when the
    /// computed rate is NaN or Infinity (e.g. inf-inf endpoints,
    /// NaN in either endpoint, or a finite endpoint difference
    /// that overflows f64). Without the `rate.is_finite()` check,
    /// IEEE-754 `<` against NaN is always false and `<` against
    /// Inf trivially passes any finite ceiling, so non-finite
    /// rates would silently slip past the band check.
    #[test]
    fn rate_within_flags_non_finite_rate() {
        let f = synthetic_field("ticks", vec![(100, f64::INFINITY), (200, f64::INFINITY)]);
        let mut v = Verdict::new();
        f.rate_within(&mut v, 0.0, 1.0);
        let r = v.into_result();
        assert!(!r.passed);
        assert!(
            r.details
                .iter()
                .any(|d| d.kind == DetailKind::Temporal && d.message.contains("non-finite rate")),
            "expected non-finite-rate failure: {:?}",
            r.details
        );
    }

    /// nondecreasing skips placeholder samples (is_placeholder=true)
    /// with a Note rather than treating them as monotonicity
    /// regressions or generic projection errors. Verifies F10:
    /// placeholder reports must NOT silently register as zero
    /// progress on a counter.
    #[test]
    fn nondecreasing_skips_placeholder_samples() {
        use crate::monitor::dump::FailureDumpReport;
        let report_a = FailureDumpReport::default(); // not a placeholder; will yield VarNotFound
        let placeholder = FailureDumpReport::placeholder("rendezvous timeout");
        let report_b = FailureDumpReport::default();
        let drained = vec![
            ("periodic_000".to_string(), report_a, None, Some(100u64)),
            ("periodic_001".to_string(), placeholder, None, Some(200u64)),
            ("periodic_002".to_string(), report_b, None, Some(300u64)),
        ];
        let series = SampleSeries::from_drained(drained);
        // Project a missing var so non-placeholder samples also
        // produce errors — but the placeholder sample's Err must
        // be the dedicated PlaceholderSample variant. The skip-
        // with-Note path collects all skipped samples; we verify
        // the placeholder tag appears in the skip list.
        let field: SeriesField<u64> = series.bpf("counter", |snap| snap.var("missing").as_u64());
        let mut v = Verdict::new();
        field.nondecreasing(&mut v);
        let r = v.into_result();
        // Verdict passes (nondecreasing skips errored samples).
        assert!(r.passed, "{:?}", r.details);
        // The Note message names the placeholder sample.
        assert!(
            r.details
                .iter()
                .any(|d| d.kind == DetailKind::Note && d.message.contains("periodic_001")),
            "expected skip Note naming placeholder sample: {:?}",
            r.details
        );
    }
}
