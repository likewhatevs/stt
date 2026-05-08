//! Unified periodic-sample bundle and series projection.
//!
//! At every periodic boundary (see [`super::snapshot`] and the
//! freeze coordinator's periodic-capture loop), the framework
//! captures a coupled [`FailureDumpReport`] + scx_stats JSON pair.
//! [`Sample`] is the borrowed-view tuple over that pair plus the
//! per-sample tag and elapsed-millisecond timestamp;
//! [`SampleSeries`] is the ordered sequence of samples drained
//! from a [`SnapshotBridge`] after VM exit.
//!
//! Test authors do not construct samples manually — they call
//! [`SampleSeries::from_drained`] on the periodic bundle the
//! bridge surfaces via
//! [`SnapshotBridge::drain_ordered_with_stats`], then project the
//! series along either the BPF or the stats axis through
//! [`SampleSeries::bpf`] / [`SampleSeries::stats`] / the typed
//! [`SampleSeries::bpf_map`] / [`SampleSeries::stats_path`]
//! auto-projection helpers. Each projection yields a
//! [`crate::assert::temporal::SeriesField`] that
//! flows into the temporal-assertion patterns
//! (`nondecreasing`, `rate_within`, `steady_within`,
//! `converges_to`, `always_true`, `ratio_within`) defined in
//! [`crate::assert::temporal`].
//!
//! # Lifetime model
//!
//! `SampleSeries` owns the drained `Vec<(tag, report, stats,
//! elapsed_ms)>` so projection closures can borrow into the
//! reports / stats without copying. Constructing a `Sample` only
//! borrows; [`SampleSeries::iter_samples`] yields `Sample<'_>`
//! bound by the series' own lifetime.

use crate::monitor::dump::FailureDumpReport;

use super::snapshot::{JsonField, Snapshot, SnapshotField, SnapshotResult, stats_path};
use crate::assert::temporal::SeriesField;

/// One captured periodic sample: a frozen BPF snapshot paired with
/// the scx_stats JSON observed just before the freeze rendezvous,
/// labelled with the periodic tag (`periodic_000` …
/// `periodic_NNN`) and tagged with the elapsed milliseconds since
/// `run_start`.
///
/// Constructed by [`SampleSeries::iter_samples`] — test authors do
/// not invoke `Sample::new` directly. The `'a` lifetime ties the
/// borrowed `tag`, `snapshot`, and `stats` references back to the
/// owning [`SampleSeries`].
#[derive(Debug)]
#[non_exhaustive]
pub struct Sample<'a> {
    /// Periodic tag the freeze coordinator stamped onto this
    /// sample. Always begins with `"periodic_"` followed by a
    /// zero-padded ordinal — see
    /// [`crate::vmm::freeze_coord::periodic_tag`].
    pub tag: &'a str,
    /// Wall-clock elapsed milliseconds (pause-adjusted: the
    /// coordinator subtracts cumulative ScenarioPause/Resume
    /// pause time and any in-flight pause window) since the
    /// coordinator's `run_start` instant at stats-request
    /// completion time, pre-freeze. The coordinator captures
    /// this timestamp AFTER the scx_stats request returns
    /// (or fails) and BEFORE entering the freeze rendezvous,
    /// so the value reflects when the running scheduler's
    /// stats were observed. BPF state is observed up to
    /// `FREEZE_RENDEZVOUS_TIMEOUT` later than this anchor.
    /// `0` when the bridge could not record a timestamp
    /// (legacy stores without elapsed metadata, or
    /// non-periodic captures surfaced through the same drain).
    pub elapsed_ms: u64,
    /// Frozen BPF state captured at this boundary. The view is
    /// cheap to build — accessor methods walk the underlying
    /// [`FailureDumpReport`] in place.
    pub snapshot: Snapshot<'a>,
    /// scx_stats JSON observed by a stats request issued just
    /// BEFORE the freeze rendezvous. `None` when the stats client
    /// was not wired (`scheduler_binary` is absent) or the request
    /// failed (relay rejected, non-zero envelope errno, scheduler
    /// not yet listening). Temporal assertions over `.stats(...)`
    /// must tolerate the `None` case — typically by skipping the
    /// sample or by failing fast when stats coverage is required.
    pub stats: Option<&'a serde_json::Value>,
}

/// Ordered collection of [`Sample`]s drained from a
/// [`SnapshotBridge`](super::snapshot::SnapshotBridge) after a VM
/// run completes. Owns the underlying tuples so projection
/// closures can borrow into the reports / stats without copying.
///
/// Test authors construct a `SampleSeries` from
/// [`super::snapshot::SnapshotBridge::drain_ordered_with_stats`]
/// via [`Self::from_drained`]; non-periodic tags (e.g. `Op::Snapshot`
/// captures) coexist in the drain output and are tolerated by the
/// projection helpers — the typical pattern is to pre-filter to
/// periodic tags via [`Self::periodic_only`] before asserting.
#[derive(Debug)]
pub struct SampleSeries {
    rows: Vec<SampleRow>,
}

/// Owned tuple stored inside [`SampleSeries`]. Mirrors the shape of
/// [`super::snapshot::SnapshotBridge::drain_ordered_with_stats`]
/// but carries the timestamp explicitly (defaulted to `0` when
/// the bridge omitted it) so iteration does not have to handle
/// the `Option<u64>` repeatedly.
#[derive(Debug)]
struct SampleRow {
    tag: String,
    report: FailureDumpReport,
    stats: Option<serde_json::Value>,
    elapsed_ms: u64,
}

impl SampleSeries {
    /// Build a series from the bridge's drained tuple. Every entry
    /// is preserved in the order the bridge surfaced, including
    /// non-periodic tags — callers that want the periodic-only
    /// view chain `.periodic_only()`.
    pub fn from_drained(
        drained: Vec<(
            String,
            FailureDumpReport,
            Option<serde_json::Value>,
            Option<u64>,
        )>,
    ) -> Self {
        let rows = drained
            .into_iter()
            .map(|(tag, report, stats, elapsed_ms)| SampleRow {
                tag,
                report,
                stats,
                elapsed_ms: elapsed_ms.unwrap_or(0),
            })
            .collect();
        Self { rows }
    }

    /// Empty series. Useful for tests and for the no-periodic-
    /// capture case where every assertion vacuously passes.
    pub fn empty() -> Self {
        Self { rows: Vec::new() }
    }

    /// True when no samples are present.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of samples in the series.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Filter the series to entries whose tag begins with
    /// `"periodic_"`. Periodic captures are the only entries the
    /// temporal-assertion patterns are designed for; on-demand
    /// `Op::Snapshot` and watchpoint-fire captures share the
    /// bridge's tag namespace and would otherwise mix into the
    /// timeline as off-cadence outliers. Consumes `self` because
    /// the filter rebuilds the owning row vec — when a borrowed
    /// view is needed instead, see [`Self::periodic_ref`] which
    /// iterates the same rows without taking ownership.
    #[must_use = "periodic_only returns a filtered series; bind the result"]
    pub fn periodic_only(self) -> Self {
        Self {
            rows: self
                .rows
                .into_iter()
                .filter(|r| r.tag.starts_with("periodic_"))
                .collect(),
        }
    }

    /// Borrowed equivalent of [`Self::periodic_only`]: yields a
    /// borrowed-view iterator over [`Sample`]s whose tag starts
    /// with `"periodic_"`, without consuming the series. Use when
    /// a single test asserts on both periodic-only and
    /// all-captures views from the same series.
    pub fn periodic_ref(&self) -> impl Iterator<Item = Sample<'_>> {
        self.iter_samples()
            .filter(|s| s.tag.starts_with("periodic_"))
    }

    /// Iterate over [`Sample`] views borrowing into this series.
    /// Each yielded `Sample<'_>` carries the tag, elapsed-ms,
    /// borrowed [`Snapshot`], and borrowed `Option<&Value>` stats.
    pub fn iter_samples(&self) -> impl Iterator<Item = Sample<'_>> {
        self.rows.iter().map(|r| Sample {
            tag: r.tag.as_str(),
            elapsed_ms: r.elapsed_ms,
            snapshot: Snapshot::new(&r.report),
            stats: r.stats.as_ref(),
        })
    }

    /// Project the series along the BPF axis. The closure receives
    /// each sample's [`Snapshot`] and returns a
    /// [`SnapshotResult<T>`] — typically a typed value extracted
    /// via `snap.var(...).as_u64()` or
    /// `snap.map(...).at(...).get(...).as_u64()`. Errors flow
    /// through into the resulting [`SeriesField`] as per-sample
    /// `Err` slots so a temporal-assertion pattern can decide
    /// whether to fail or skip on a missing field.
    ///
    /// `label` is owned (`impl Into<String>`) and lands in
    /// [`crate::assert::temporal::SeriesField::label`] for failure-
    /// message rendering. Callers may pass a `&'static str` literal
    /// or a runtime-built `String` (for auto-discovered struct or
    /// JSON key names).
    pub fn bpf<T, F>(&self, label: impl Into<String>, project: F) -> SeriesField<T>
    where
        F: Fn(&Snapshot<'_>) -> SnapshotResult<T>,
    {
        let mut values: Vec<SnapshotResult<T>> = Vec::with_capacity(self.rows.len());
        let mut tags: Vec<String> = Vec::with_capacity(self.rows.len());
        let mut elapsed: Vec<u64> = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            tags.push(row.tag.clone());
            elapsed.push(row.elapsed_ms);
            // Placeholder reports carry no real BPF state — the
            // freeze rendezvous timed out (or the capture pipeline
            // otherwise failed). Surface a dedicated PlaceholderSample
            // error variant BEFORE invoking the projection closure
            // so the temporal-assertion patterns can branch on
            // "placeholder, skip" distinctly from "field missing,
            // skip" when rendering the verdict's skip-Note.
            if row.report.is_placeholder {
                values.push(Err(
                    crate::scenario::snapshot::SnapshotError::PlaceholderSample {
                        tag: row.tag.clone(),
                        reason: row
                            .report
                            .scx_walker_unavailable
                            .clone()
                            .unwrap_or_else(|| "placeholder report".to_string()),
                    },
                ));
                continue;
            }
            let snap = Snapshot::new(&row.report);
            values.push(project(&snap));
        }
        SeriesField::from_parts(label, tags, elapsed, values)
    }

    /// Project the series along the stats axis. The closure
    /// receives each sample's stats JSON (when present) and
    /// returns a [`SnapshotResult<T>`]. Samples whose `stats` is
    /// `None` get a `Err(MissingStats)` slot — temporal assertions
    /// surface that as a per-sample missing-stats failure rather
    /// than vacuously skipping it, so a coverage gap is never
    /// silent.
    ///
    /// `label` is owned (`impl Into<String>`) and matches the
    /// shape of [`Self::bpf`] — pass a literal or a runtime-built
    /// `String` for auto-discovered keys.
    pub fn stats<T, F>(&self, label: impl Into<String>, project: F) -> SeriesField<T>
    where
        F: Fn(StatsValue<'_>) -> SnapshotResult<T>,
    {
        let mut values: Vec<SnapshotResult<T>> = Vec::with_capacity(self.rows.len());
        let mut tags: Vec<String> = Vec::with_capacity(self.rows.len());
        let mut elapsed: Vec<u64> = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            tags.push(row.tag.clone());
            elapsed.push(row.elapsed_ms);
            let outcome = match row.stats.as_ref() {
                Some(v) => project(StatsValue { value: v }),
                None => Err(crate::scenario::snapshot::SnapshotError::MissingStats {
                    tag: row.tag.clone(),
                }),
            };
            values.push(outcome);
        }
        SeriesField::from_parts(label, tags, elapsed, values)
    }

    /// Auto-project a top-level BPF map's struct members. The
    /// returned [`BpfMapProjector`] auto-discovers struct member
    /// names at sample 0 and exposes them via `.field_u64(name)` /
    /// `.field_i64(name)` / `.field_f64(name)` — a caller that
    /// wants every scalar field of a BSS struct without
    /// enumerating each one by hand calls
    /// `series.bpf_map("scx_obj.bss").at(0)` and then
    /// `.field_u64("nr_dispatched")` for the field of interest.
    ///
    /// **Top-level scalar fields only.** The auto-projector reads
    /// directly-named struct members (e.g. `"nr_dispatched"`,
    /// `"stall"`). Nested struct members (e.g. `"ctx.weight"`) and
    /// deeper paths are NOT auto-discoverable through the typed
    /// `field_*` helpers — for those, use the manual closure
    /// projection [`SampleSeries::bpf`] with
    /// `|snap| snap.var("ctx").get("weight").as_u64()` (or the
    /// equivalent map-walking shape). Per-CPU maps are also out
    /// of scope: they require an explicit `.cpu(N)` narrow on
    /// the [`Snapshot`] accessor surface, so callers route
    /// through the manual closure path for those as well.
    pub fn bpf_map<'a>(&'a self, map_name: &'a str) -> BpfMapProjector<'a> {
        BpfMapProjector {
            series: self,
            map_name,
            entry_index: 0,
        }
    }

    /// Auto-project a stats-JSON sub-tree. The returned
    /// [`StatsPathProjector`] resolves the tree at sample 0 and
    /// exposes object keys via `.key(name)` (for nested layer /
    /// cgroup objects) or `.field(name)` (for scalar leaves).
    /// `path` may be empty — `series.stats_path("")` projects from
    /// the root and is the canonical entry for system-level stats
    /// fields like `busy`, `antistall`, `system_cpu_util_ewma`,
    /// etc.
    pub fn stats_path<'a>(&'a self, path: &str) -> StatsPathProjector<'a> {
        StatsPathProjector {
            series: self,
            path: path.to_string(),
        }
    }
}

/// Newtype carrier handed to the [`SampleSeries::stats`] closure.
/// Wraps a borrowed [`serde_json::Value`] and exposes [`Self::path`]
/// as a thin facade over [`stats_path`] so the closure body reads
/// `s.path("layers.batch.util").as_f64()` without an explicit
/// import.
#[derive(Debug, Clone, Copy)]
pub struct StatsValue<'a> {
    value: &'a serde_json::Value,
}

impl<'a> StatsValue<'a> {
    /// Underlying JSON value.
    pub fn raw(&self) -> &'a serde_json::Value {
        self.value
    }

    /// Walk along a dotted path. Empty path returns the root.
    pub fn path(&self, path: &str) -> JsonField<'a> {
        stats_path(self.value, path)
    }
}

/// Auto-projector handle returned by [`SampleSeries::bpf_map`].
/// Lazily resolves the named map's value at the requested entry
/// index when [`Self::field`] is invoked.
pub struct BpfMapProjector<'a> {
    series: &'a SampleSeries,
    map_name: &'a str,
    entry_index: usize,
}

impl<'a> BpfMapProjector<'a> {
    /// Pin the entry index for the projection. Defaults to `0`
    /// (typical for ARRAY / `.bss` / `.data` / `.rodata` maps,
    /// which carry a single value at index 0). Use this to walk
    /// into a HASH map at a specific ordinal.
    pub fn at(mut self, index: usize) -> Self {
        self.entry_index = index;
        self
    }

    /// Project a single named struct field as `u64` (the most
    /// common temporal-assertion shape — counters, byte counts).
    /// The label routed onto the resulting [`SeriesField`] is the
    /// caller-supplied field name; combined with the map name in
    /// the diagnostic the failure message reads
    /// `"<map>.<entry_index>.<field>"`.
    pub fn field_u64(&self, field: &str) -> SeriesField<u64> {
        let map_name = self.map_name.to_string();
        let entry_index = self.entry_index;
        let field_owned = field.to_string();
        self.series.bpf(field, move |snap| {
            let entry = match snap.map(&map_name) {
                Ok(m) => m.at(entry_index),
                Err(e) => return Err(e),
            };
            entry.get(&field_owned).as_u64()
        })
    }

    /// Project a single named struct field as `i64`.
    pub fn field_i64(&self, field: &str) -> SeriesField<i64> {
        let map_name = self.map_name.to_string();
        let entry_index = self.entry_index;
        let field_owned = field.to_string();
        self.series.bpf(field, move |snap| {
            let entry = match snap.map(&map_name) {
                Ok(m) => m.at(entry_index),
                Err(e) => return Err(e),
            };
            entry.get(&field_owned).as_i64()
        })
    }

    /// Project a single named struct field as `f64`.
    pub fn field_f64(&self, field: &str) -> SeriesField<f64> {
        let map_name = self.map_name.to_string();
        let entry_index = self.entry_index;
        let field_owned = field.to_string();
        self.series.bpf(field, move |snap| {
            let entry = match snap.map(&map_name) {
                Ok(m) => m.at(entry_index),
                Err(e) => return Err(e),
            };
            entry.get(&field_owned).as_f64()
        })
    }

    /// Discover the struct member names of the map's first
    /// rendered value. Empty when the map is missing in sample 0
    /// or its value is not a struct. Useful for tests that want
    /// to enumerate every scalar field for a blanket assertion.
    pub fn member_names(&self) -> Vec<String> {
        let row = match self.series.rows.first() {
            Some(r) => r,
            None => return Vec::new(),
        };
        let snap = Snapshot::new(&row.report);
        let map = match snap.map(self.map_name) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let entry = map.at(self.entry_index);
        // Walk the entry's value — SnapshotEntry doesn't expose
        // its struct members directly, but the rendered_value()
        // accessor on the field-with-empty-path does.
        let field = entry.get("");
        match field {
            SnapshotField::Value(crate::monitor::btf_render::RenderedValue::Struct {
                members,
                ..
            }) => members.iter().map(|m| m.name.clone()).collect(),
            _ => Vec::new(),
        }
    }

    /// Project every struct member that resolves as `u64` for at
    /// least one sample. Iterates [`Self::member_names`], calls
    /// [`Self::field_u64`] for each, and keeps the entries whose
    /// resulting [`SeriesField`] has at least one `Ok` value —
    /// non-numeric members (strings, nested structs, floats) drop
    /// out because their `as_u64()` cast always errors.
    pub fn u64_fields(&self) -> Vec<(String, SeriesField<u64>)> {
        self.member_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_u64(&name);
                // Bind the predicate result and drop the
                // values_iter borrow before moving `field`. A
                // chained `.values_iter().any(...).then_some(...)`
                // keeps the iterator alive across the move and
                // fails the borrow check.
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }

    /// Project every struct member that resolves as `f64` for at
    /// least one sample. Mirrors [`Self::u64_fields`] using
    /// [`Self::field_f64`].
    pub fn f64_fields(&self) -> Vec<(String, SeriesField<f64>)> {
        self.member_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_f64(&name);
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }
}

/// Auto-projector handle returned by [`SampleSeries::stats_path`].
/// Walks a stats sub-tree per sample and exposes scalar / nested
/// projections for the keys at that level.
pub struct StatsPathProjector<'a> {
    series: &'a SampleSeries,
    path: String,
}

impl<'a> StatsPathProjector<'a> {
    /// Project a JSON key under the resolved path as `u64`.
    pub fn field_u64(&self, key: &str) -> SeriesField<u64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.path(&full_path).as_u64())
    }

    /// Project a JSON key under the resolved path as `i64`.
    pub fn field_i64(&self, key: &str) -> SeriesField<i64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.path(&full_path).as_i64())
    }

    /// Project a JSON key under the resolved path as `f64`.
    pub fn field_f64(&self, key: &str) -> SeriesField<f64> {
        let full_path = join_paths(&self.path, key);
        self.series
            .stats(key, move |sv| sv.path(&full_path).as_f64())
    }

    /// Return a sub-projector rooted under `key`. Composable —
    /// `series.stats_path("layers").key("batch").field_f64("util")`
    /// drills into the per-layer scheduler stats one segment at a
    /// time without each call site re-typing the full dotted
    /// path.
    pub fn key(&self, key: &str) -> StatsPathProjector<'a> {
        StatsPathProjector {
            series: self.series,
            path: join_paths(&self.path, key),
        }
    }

    /// Discover the JSON object keys of the resolved path at
    /// sample 0. Empty when the path is missing or resolves to a
    /// non-object; populated when the projection lands on a
    /// `serde_json::Value::Object`.
    pub fn key_names(&self) -> Vec<String> {
        let row = match self.series.rows.first() {
            Some(r) => r,
            None => return Vec::new(),
        };
        let stats = match row.stats.as_ref() {
            Some(s) => s,
            None => return Vec::new(),
        };
        let resolved = stats_path(stats, &self.path);
        let raw = match resolved.raw() {
            Some(v) => v,
            None => return Vec::new(),
        };
        match raw {
            serde_json::Value::Object(map) => {
                let mut names: Vec<String> = map.keys().cloned().collect();
                names.sort();
                names
            }
            _ => Vec::new(),
        }
    }

    /// Project every object key that resolves as `u64` for at
    /// least one sample. Iterates [`Self::key_names`], calls
    /// [`Self::field_u64`] for each, and keeps the entries whose
    /// resulting [`SeriesField`] has at least one `Ok` value —
    /// non-numeric leaves (strings, nested objects, floats) drop
    /// out.
    pub fn u64_fields(&self) -> Vec<(String, SeriesField<u64>)> {
        self.key_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_u64(&name);
                // Bind the predicate result and drop the
                // values_iter borrow before moving `field`.
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }

    /// Project every object key that resolves as `f64` for at
    /// least one sample. Mirrors [`Self::u64_fields`] using
    /// [`Self::field_f64`].
    pub fn f64_fields(&self) -> Vec<(String, SeriesField<f64>)> {
        self.key_names()
            .into_iter()
            .filter_map(|name| {
                let field = self.field_f64(&name);
                let any_ok = field.values_iter().any(|r| r.is_ok());
                any_ok.then_some((name, field))
            })
            .collect()
    }
}

fn join_paths(base: &str, leaf: &str) -> String {
    if base.is_empty() {
        leaf.to_string()
    } else if leaf.is_empty() {
        base.to_string()
    } else {
        format!("{base}.{leaf}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::btf_render::{RenderedMember, RenderedValue};
    use crate::monitor::dump::{FailureDumpMap, FailureDumpReport, SCHEMA_SINGLE};

    fn synthetic_report(value: u64) -> FailureDumpReport {
        let bss_value = RenderedValue::Struct {
            type_name: Some(".bss".into()),
            members: vec![
                RenderedMember {
                    name: "nr_dispatched".into(),
                    value: RenderedValue::Uint { bits: 64, value },
                },
                RenderedMember {
                    name: "stall".into(),
                    value: RenderedValue::Uint { bits: 8, value: 0 },
                },
            ],
        };
        let bss_map = FailureDumpMap {
            name: "scx_obj.bss".into(),
            map_type: 2,
            value_size: 16,
            max_entries: 1,
            value: Some(bss_value),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            percpu_hash_entries: Vec::new(),
            arena: None,
            ringbuf: None,
            stack_trace: None,
            fd_array: None,
            error: None,
        };
        FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![bss_map],
            ..Default::default()
        }
    }

    fn synthetic_stats(busy: f64) -> serde_json::Value {
        serde_json::json!({
            "busy": busy,
            "antistall": 0,
            "layers": {
                "batch": { "util": busy * 0.5 }
            }
        })
    }

    #[test]
    fn from_drained_preserves_order() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                Some(synthetic_stats(50.0)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                Some(synthetic_stats(60.0)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained);
        assert_eq!(series.len(), 2);
        let tags: Vec<&str> = series.iter_samples().map(|s| s.tag).collect();
        assert_eq!(tags, vec!["periodic_000", "periodic_001"]);
    }

    #[test]
    fn periodic_only_filters_non_periodic_tags() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                None,
                Some(100),
            ),
            (
                "user_watchpoint_kind".to_string(),
                synthetic_report(99),
                None,
                Some(150),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained).periodic_only();
        assert_eq!(series.len(), 2);
    }

    #[test]
    fn bpf_projection_extracts_field_per_sample() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                None,
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained);
        let field: SeriesField<u64> =
            series.bpf("nr_dispatched", |snap| snap.var("nr_dispatched").as_u64());
        let values: Vec<u64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn stats_projection_handles_missing_stats_as_error() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                Some(synthetic_stats(50.0)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained);
        let field: SeriesField<f64> = series.stats("busy", |s| s.path("busy").as_f64());
        let outcomes: Vec<SnapshotResult<f64>> = field.values_iter().cloned().collect();
        assert_eq!(outcomes.len(), 2);
        assert_eq!(
            outcomes[0].as_ref().copied(),
            Ok(50.0),
            "sample with stats present must project the `busy` field verbatim"
        );
        match &outcomes[1] {
            Err(crate::scenario::snapshot::SnapshotError::MissingStats { tag }) => {
                assert_eq!(
                    tag, "periodic_001",
                    "MissingStats tag must identify the sample whose stats slot was None"
                );
            }
            other => panic!(
                "sample with stats=None must surface SnapshotError::MissingStats, got {other:?}"
            ),
        }
    }

    #[test]
    fn bpf_map_projector_field_u64_extracts_field() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(10),
                None,
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(20),
                None,
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained);
        let field = series
            .bpf_map("scx_obj.bss")
            .at(0)
            .field_u64("nr_dispatched");
        let values: Vec<u64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values, vec![10, 20]);
    }

    #[test]
    fn bpf_map_projector_member_names_lists_struct_fields() {
        let drained = vec![(
            "periodic_000".to_string(),
            synthetic_report(10),
            None,
            Some(100),
        )];
        let series = SampleSeries::from_drained(drained);
        let names = series.bpf_map("scx_obj.bss").at(0).member_names();
        assert!(names.contains(&"nr_dispatched".to_string()));
        assert!(names.contains(&"stall".to_string()));
    }

    #[test]
    fn stats_path_projector_field_f64_extracts_root_scalar() {
        let drained = vec![
            (
                "periodic_000".to_string(),
                synthetic_report(0),
                Some(synthetic_stats(50.0)),
                Some(100),
            ),
            (
                "periodic_001".to_string(),
                synthetic_report(0),
                Some(synthetic_stats(60.0)),
                Some(200),
            ),
        ];
        let series = SampleSeries::from_drained(drained);
        let field = series.stats_path("").field_f64("busy");
        let values: Vec<f64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 50.0).abs() < f64::EPSILON);
        assert!((values[1] - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_path_projector_key_names_at_root() {
        let drained = vec![(
            "periodic_000".to_string(),
            synthetic_report(0),
            Some(synthetic_stats(50.0)),
            Some(100),
        )];
        let series = SampleSeries::from_drained(drained);
        let names = series.stats_path("").key_names();
        assert!(names.contains(&"busy".to_string()));
        assert!(names.contains(&"layers".to_string()));
    }

    #[test]
    fn stats_path_projector_nested_key_drills_in() {
        let drained = vec![(
            "periodic_000".to_string(),
            synthetic_report(0),
            Some(synthetic_stats(50.0)),
            Some(100),
        )];
        let series = SampleSeries::from_drained(drained);
        // Note: drilling deeper than 2 levels via key() chain works
        // because key() returns the same kind of projector.
        let field = series.stats_path("layers").key("batch").field_f64("util");
        let values: Vec<f64> = field
            .values_iter()
            .filter_map(|v| v.as_ref().ok().copied())
            .collect();
        assert_eq!(values.len(), 1);
        assert!((values[0] - 25.0).abs() < f64::EPSILON);
    }
}
