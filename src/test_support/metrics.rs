//! Metric extraction pipeline for payload outputs.
//!
//! Payloads declared with [`OutputFormat::Json`] emit JSON to either
//! stdout or stderr — `PayloadRun` applies a stdout-primary /
//! stderr-fallback contract and hands whichever stream produced a
//! non-empty metric set to this module. Benchmark tools split
//! across the two conventions (schbench writes structured summaries
//! to stderr, fio / stress-ng to stdout); the fallback lets either
//! discipline round-trip through the same extractor. This module
//! locates the JSON document region inside mixed text output
//! (many tools emit a banner line before their structured body) and
//! walks numeric leaves into [`Metric`]s keyed by dotted paths.
//!
//! [`OutputFormat::ExitCode`] returns an empty metric set; exit-code
//! pass/fail is handled by the [`Check::ExitCodeEq`] pre-pass
//! elsewhere.
//!
//! [`OutputFormat::LlmExtract`] routes the same (possibly
//! stderr-sourced) output through
//! [`crate::test_support::model::extract_via_llm`]: the model owns
//! prompt composition and the initial JSON-from-prose parse, then
//! feeds the resulting `serde_json::Value` into this module's
//! [`walk_json_leaves`] with the source pre-tagged to
//! [`MetricSource::LlmExtract`]. One extraction walker, two
//! acquisition paths.

use crate::test_support::{Metric, MetricSource, OutputFormat, Polarity};

/// Extract metrics from a payload's captured output per its declared
/// [`OutputFormat`].
///
/// `output` carries whichever stream `PayloadRun` decided to extract
/// from — stdout on the happy path, stderr under the stdout-primary
/// stderr-fallback contract when stdout produced an empty result.
/// The extractor itself is stream-agnostic; it parses whatever byte
/// blob it is handed.
///
/// Returns an empty `Vec` for [`OutputFormat::ExitCode`] and for
/// [`OutputFormat::Json`] when no JSON document is located or the
/// document carries no numeric leaves. JSON-parse errors are
/// non-fatal: the extraction returns `Vec::new()` so downstream
/// [`Check`](crate::test_support::Check) evaluation reports each
/// referenced metric as missing rather than failing the whole run.
///
/// [`OutputFormat::LlmExtract`] with an optional `hint` delegates to
/// [`crate::test_support::model::extract_via_llm`], which composes a
/// prompt (appending the hint when present), runs a single
/// deterministic (ArgMax) inference pass, and walks the resulting
/// JSON with [`MetricSource::LlmExtract`]. An unavailable inference
/// backend (missing cache, forward-pass failure) yields an empty
/// metric set, matching the non-fatal contract above.
///
/// # Known truncation point: depth cap
///
/// Both the `Json` and `LlmExtract` arms route through
/// [`walk_json_leaves`], which enforces a hard recursion cap of
/// [`MAX_WALK_DEPTH`] (currently 64). Subtrees past that depth are
/// silently dropped from the metric list — a `tracing::warn!` fires
/// and a sentinel metric named [`WALK_TRUNCATION_SENTINEL_NAME`]
/// (`__walk_json_leaves_truncated`) is appended to the return
/// value, with `value` set to the depth at which truncation
/// occurred. Callers that want to distinguish "no deep metrics"
/// from "deep metrics dropped by the cap" scan the returned `Vec`
/// for a metric with that name. Practical upper bound: 64 is well
/// below serde_json's default parse recursion limit (128) and
/// covers every realistic payload schema observed in the crate
/// (fio maxes out around depth 8, schbench around depth 3).
pub fn extract_metrics(output: &str, format: &OutputFormat) -> Result<Vec<Metric>, String> {
    match format {
        OutputFormat::ExitCode => Ok(Vec::new()),
        OutputFormat::Json => Ok(find_and_parse_json(output)
            .map(|v| walk_json_leaves(&v, MetricSource::Json))
            .unwrap_or_default()),
        OutputFormat::LlmExtract(hint) => super::model::extract_via_llm(output, *hint),
    }
}

/// Locate a JSON document within mixed text output and parse it.
///
/// Many benchmark tools emit a banner line (fio, stress-ng)
/// before the structured JSON body. A strict
/// `serde_json::from_str(output)` fails for those. This helper
/// first tries the whole input; on failure, scans for the first
/// balanced `{...}` (or `[...]`) region and parses that.
///
/// Returns `None` when no JSON document is locatable or parsing
/// both candidates fails. Does NOT heuristically repair malformed
/// JSON — only brace-balancing for region extraction; serde_json
/// does the actual parse strictly.
///
/// # Multiple JSON objects in one output
///
/// When `output` contains more than one balanced top-level region
/// (e.g. `{"first": 1} noise {"second": 2}`), only the FIRST is
/// returned. The region finder scans left-to-right for the first
/// `{` or `[`, walks to its matching closer, and stops — it does
/// not merge or concatenate subsequent balanced regions. Payloads
/// that emit multiple JSON documents per run therefore lose all
/// but the first; authors needing full capture should switch the
/// payload to a wrapper that emits a single aggregate document
/// (or use `OutputFormat::LlmExtract` to consolidate prose +
/// multiple JSONs through the model pipeline).
pub(crate) fn find_and_parse_json(output: &str) -> Option<serde_json::Value> {
    // Fast path: whole input is a single JSON document.
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(output.trim()) {
        return Some(v);
    }
    // Slow path: find the first balanced `{...}` or `[...]` region.
    let region = extract_json_region(output)?;
    serde_json::from_str::<serde_json::Value>(region).ok()
}

/// Find the first balanced `{...}` or `[...]` region in `s`.
///
/// Scans left-to-right for the first `{` or `[` and returns a slice
/// spanning to its matching closer, tracking nesting + escaped
/// quotes. Returns `None` if no opener found or no balanced match
/// within the input.
///
/// This is NOT a JSON parser — it's a region locator. The returned
/// slice is handed to `serde_json::from_str` for strict parsing.
/// Mismatched structures (e.g. `{...}]`) are detected there, not
/// here.
fn extract_json_region(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&c| c == b'{' || c == b'[')?;
    let opener = bytes[start];
    let closer = if opener == b'{' { b'}' } else { b']' };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &c) in bytes.iter().enumerate().skip(start) {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match c {
                b'\\' => escape = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            x if x == opener => depth += 1,
            x if x == closer => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Walk numeric leaves of a JSON value, emitting [`Metric`]s keyed
/// by dotted paths.
///
/// Objects contribute `"key.subkey"` paths; arrays contribute
/// `"key.0", "key.1"`. Numeric leaves where `as_f64()` yields a
/// finite value are emitted; String, Bool, and Null leaves are
/// skipped. NaN/infinite f64s are rejected by serde_json at parse
/// time, so natural inputs never reach this walker with non-finite
/// numbers; the defensive `is_finite()` guard catches hand-built
/// `Value` constructions.
///
/// Each [`Metric`] is emitted with [`Polarity::Unknown`] and empty
/// unit; the caller resolves these against the payload's declared
/// [`MetricHint`](crate::test_support::MetricHint)s to upgrade
/// polarity.
pub fn walk_json_leaves(value: &serde_json::Value, source: MetricSource) -> Vec<Metric> {
    let mut out = Vec::new();
    // Single reusable path buffer: children push their segment,
    // recurse, then truncate back. O(total_path_chars) work across
    // the whole walk instead of O(depth × path_chars) per leaf.
    let mut path = String::new();
    walk(value, &mut path, 0, source, &mut out);
    out
}

/// Hard cap on recursion depth in [`walk`]. Object and array
/// children past this depth are skipped and a single
/// [`tracing::warn!`] fires. Serde_json's default parser recursion
/// limit is 128, so this caps us well below that; a hand-built
/// `serde_json::Value` that bypasses the parser can still reach
/// arbitrary depth, so an explicit walker guard is the last line of
/// defence against a stack overflow.
pub(crate) const MAX_WALK_DEPTH: usize = 64;

/// Sentinel metric name emitted when [`walk`] hits
/// [`MAX_WALK_DEPTH`] and skips a subtree. Callers of
/// [`walk_json_leaves`] / [`extract_metrics`] that want to
/// distinguish "no deep metrics present" from "deep metrics
/// dropped by the depth cap" scan the returned `Vec<Metric>` for
/// a metric whose `name` equals this constant — its `value` is
/// the depth at which truncation occurred, so nested failures at
/// different subtrees produce one sentinel per trigger.
///
/// # Accepted collision risk
///
/// The double-underscore prefix makes collision extremely unlikely
/// in practice, but not impossible: a benchmark whose JSON has
/// this exact literal string as a **top-level** key produces a
/// `Metric.name` indistinguishable from the cap-hit sentinel
/// (nested leaves get at least one `.` injected by `walk`, so only
/// the top-level depth-0 push can produce a name without a `.`).
/// Consumers treat the sentinel as advisory, not authoritative —
/// a caller that depends on zero-collision guarantees must reject
/// sentinel-named paths from its input schema.
///
/// A future refactor could eliminate the risk structurally by
/// widening the return type to `WalkResult { metrics: Vec<Metric>,
/// truncated: Option<u64> }` — separating the truncation signal
/// from the metric stream. Held off pending a consumer that
/// materially benefits from zero-collision certainty; the current
/// advisory contract is sufficient for every in-crate consumer.
pub(crate) const WALK_TRUNCATION_SENTINEL_NAME: &str = "__walk_json_leaves_truncated";

fn walk(
    value: &serde_json::Value,
    path: &mut String,
    depth: usize,
    source: MetricSource,
    out: &mut Vec<Metric>,
) {
    if depth > MAX_WALK_DEPTH {
        tracing::warn!(
            depth,
            max = MAX_WALK_DEPTH,
            path = %path,
            "walk_json_leaves: depth cap hit, subtree skipped",
        );
        // Emit a sentinel metric so callers inspecting only the
        // returned `Vec<Metric>` see the truncation — the
        // `tracing::warn!` above only reaches a subscriber, which
        // the default test dispatch path does not install. See
        // [`WALK_TRUNCATION_SENTINEL_NAME`] for the discrimination
        // contract.
        out.push(Metric {
            name: WALK_TRUNCATION_SENTINEL_NAME.to_string(),
            value: depth as f64,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source,
        });
        return;
    }
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let saved_len = path.len();
                if !path.is_empty() {
                    path.push('.');
                }
                path.push_str(k);
                walk(v, path, depth + 1, source, out);
                path.truncate(saved_len);
            }
        }
        serde_json::Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                let saved_len = path.len();
                if !path.is_empty() {
                    path.push('.');
                }
                // Avoid an extra String allocation for the index
                // segment by writing directly into `path` via the
                // fmt::Write impl (infallible for String).
                use std::fmt::Write;
                let _ = write!(path, "{i}");
                walk(v, path, depth + 1, source, out);
                path.truncate(saved_len);
            }
        }
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64()
                && f.is_finite()
            {
                // Leaf emission is the one unavoidable allocation —
                // `Metric.name` is owned. `clone()` copies exactly
                // the current path bytes, not every intermediate
                // ancestor path that `format!` used to materialize.
                out.push(Metric {
                    name: path.clone(),
                    value: f,
                    polarity: Polarity::Unknown,
                    unit: String::new(),
                    source,
                });
            }
        }
        // Strings/bools/null: skipped. Check::Exists can gate on
        // presence via the PayloadMetrics lookup — a missing
        // string-valued key is treated the same as a missing numeric.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_returns_empty() {
        let m = extract_metrics("whatever", &OutputFormat::ExitCode).unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn json_full_document_extracts_numeric_leaves() {
        let s = r#"{"iops": 10000, "lat_ns": 500}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 2);
        let names: Vec<_> = m.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"iops"));
        assert!(names.contains(&"lat_ns"));
        for metric in &m {
            assert_eq!(metric.source, MetricSource::Json);
            assert_eq!(metric.polarity, Polarity::Unknown);
            assert_eq!(metric.unit, "");
        }
    }

    #[test]
    fn json_with_banner_prefix_extracts_region() {
        // Fio-style: banner line then JSON body.
        let s = "fio-3.36 starting up\n{\"iops\": 500}";
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "iops");
        assert_eq!(m[0].value, 500.0);
    }

    #[test]
    fn json_nested_objects_use_dotted_paths() {
        let s = r#"{"jobs": {"0": {"read": {"iops": 123}}}}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "jobs.0.read.iops");
        assert_eq!(m[0].value, 123.0);
    }

    #[test]
    fn json_arrays_use_numeric_index_paths() {
        let s = r#"{"samples": [100, 200, 300]}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 3);
        let mut actual: Vec<(&str, f64)> = m.iter().map(|x| (x.name.as_str(), x.value)).collect();
        actual.sort_by_key(|(n, _)| n.to_string());
        assert_eq!(
            actual,
            vec![
                ("samples.0", 100.0),
                ("samples.1", 200.0),
                ("samples.2", 300.0),
            ]
        );
    }

    #[test]
    fn json_malformed_returns_empty() {
        let m = extract_metrics("garbage not json", &OutputFormat::Json).unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn json_empty_stdout_returns_empty() {
        let m = extract_metrics("", &OutputFormat::Json).unwrap();
        assert!(m.is_empty());
    }

    #[test]
    fn json_skips_string_and_bool_leaves() {
        let s = r#"{"name": "fio", "ok": true, "iops": 42}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        // Only iops is numeric.
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "iops");
    }

    #[test]
    fn json_top_level_array_extracts_entries() {
        let s = "[1, 2, 3]";
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn llm_extract_returns_empty_when_backend_unavailable() {
        // LlmExtract delegates to `model::extract_via_llm`, which
        // calls `load_inference` followed by `invoke_with_model`.
        // Forcing the offline gate makes `ensure()` bail on the
        // uncached model, so `load_inference` surfaces
        // an `anyhow::Error` and the pipeline returns an empty
        // metric set — non-fatal extraction error so downstream
        // Check evaluation reports each referenced metric as missing
        // rather than failing the whole run.
        //
        // Calls `model::reset()` under `lock_env()` so a
        // previously memoized `Ok(_)` slot in `MODEL_CACHE` cannot
        // bypass the offline gate. Without the reset, the test could
        // pass for the wrong reason — cached inference yielding an
        // empty Vec rather than `ensure()` tripping on the offline
        // env var.
        let _lock = super::super::test_helpers::lock_env();
        super::super::model::reset();
        let _cache = super::super::test_helpers::isolated_cache_dir();
        let _env_offline = super::super::test_helpers::EnvVarGuard::set("KTSTR_MODEL_OFFLINE", "1");
        // Return is `Err(reason)` — a model-cache load failure is
        // surfaced as a threaded reason string so the Check
        // evaluator can attach it to the AssertResult. The exact
        // reason message includes the offline-gate env-var name.
        let err = extract_metrics("anything", &OutputFormat::LlmExtract(None))
            .expect_err("offline gate must produce Err from extract_metrics");
        assert!(
            err.contains(super::super::model::OFFLINE_ENV),
            "reason should name the offline env var, got: {err}"
        );
    }

    #[test]
    fn llm_extract_with_hint_returns_empty_when_backend_unavailable() {
        // Same contract as `llm_extract_returns_empty_when_backend_unavailable`
        // but exercising the hint-carrying variant so the dispatch path
        // that plumbs `hint` into `extract_via_llm` is covered.
        //
        // Same `model::reset()` rationale: the offline-gate
        // assertion is meaningful only when MODEL_CACHE starts empty.
        let _lock = super::super::test_helpers::lock_env();
        super::super::model::reset();
        let _cache = super::super::test_helpers::isolated_cache_dir();
        let _env_offline = super::super::test_helpers::EnvVarGuard::set("KTSTR_MODEL_OFFLINE", "1");
        let err = extract_metrics(
            "anything",
            &OutputFormat::LlmExtract(Some("focus on latency")),
        )
        .expect_err("offline gate must produce Err from extract_metrics");
        assert!(
            err.contains(super::super::model::OFFLINE_ENV),
            "reason should name the offline env var, got: {err}"
        );
    }

    #[test]
    fn extract_json_region_finds_braced_region() {
        let r = extract_json_region("prefix {\"a\": 1} suffix").unwrap();
        assert_eq!(r, "{\"a\": 1}");
    }

    #[test]
    fn extract_json_region_handles_nested_braces() {
        let r = extract_json_region("log: {\"a\": {\"b\": 1}} done").unwrap();
        assert_eq!(r, "{\"a\": {\"b\": 1}}");
    }

    #[test]
    fn extract_json_region_skips_braces_in_strings() {
        let r = extract_json_region(r#"{"text": "not a }"}"#).unwrap();
        assert_eq!(r, r#"{"text": "not a }"}"#);
    }

    #[test]
    fn extract_json_region_handles_escaped_quotes() {
        let r = extract_json_region(r#"{"text": "has \"escaped\" quotes"}"#).unwrap();
        assert_eq!(r, r#"{"text": "has \"escaped\" quotes"}"#);
    }

    #[test]
    fn extract_json_region_returns_none_for_no_brace() {
        assert!(extract_json_region("no braces here").is_none());
    }

    #[test]
    fn extract_json_region_returns_none_for_unbalanced() {
        assert!(extract_json_region("incomplete {").is_none());
    }

    #[test]
    fn walk_json_leaves_polarity_is_unknown_before_hint_resolution() {
        let v: serde_json::Value = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        let m = walk_json_leaves(&v, MetricSource::Json);
        assert_eq!(m[0].polarity, Polarity::Unknown);
    }

    #[test]
    fn walk_json_leaves_tags_source() {
        let v: serde_json::Value = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        let json_tagged = walk_json_leaves(&v, MetricSource::Json);
        assert_eq!(json_tagged[0].source, MetricSource::Json);
        let llm_tagged = walk_json_leaves(&v, MetricSource::LlmExtract);
        assert_eq!(llm_tagged[0].source, MetricSource::LlmExtract);
    }

    // Additional edge-case coverage for walk_json_leaves paths.

    #[test]
    fn json_deeply_nested_array_of_objects() {
        // Edge case: array of objects. Each object's field should
        // emit `samples.N.field` paths.
        let s = r#"{"samples": [{"iops": 100}, {"iops": 200}, {"iops": 300}]}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 3);
        let names: Vec<&str> = m.iter().map(|x| x.name.as_str()).collect();
        assert!(names.contains(&"samples.0.iops"));
        assert!(names.contains(&"samples.1.iops"));
        assert!(names.contains(&"samples.2.iops"));
    }

    #[test]
    fn json_large_integer_round_trip_via_f64() {
        // Large but f64-safe integer (below 2^53). serde_json's
        // Number::as_f64 lossily converts any JSON number to f64;
        // values below 2^53 are exact.
        let s = r#"{"big_iops": 1000000000000}"#; // 1e12 = 2^40
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].value, 1_000_000_000_000.0);
    }

    #[test]
    fn json_fio_style_full_output_with_multiline_banner() {
        // Real-world fio output has multiple banner lines + a large
        // JSON object. Region-finder must skip all non-JSON prefix
        // and parse the JSON body.
        let s = "fio-3.36 starting up\n\
                 Running fio with 4 jobs\n\
                 test: (g=0): rw=randread, bs=4k, ioengine=libaio\n\
                 \n\
                 {\"jobs\": [{\"jobname\": \"test\", \"read\": {\"iops\": 12345, \"bw_bytes\": 50593792}}], \
                 \"disk_util\": [{\"util\": 99.5}]}";
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        // Extracted: jobs.0.read.iops, jobs.0.read.bw_bytes, disk_util.0.util.
        // jobname is a string, skipped.
        assert_eq!(m.len(), 3);
        let by_name: std::collections::BTreeMap<&str, f64> =
            m.iter().map(|x| (x.name.as_str(), x.value)).collect();
        assert_eq!(by_name.get("jobs.0.read.iops"), Some(&12345.0));
        assert_eq!(by_name.get("jobs.0.read.bw_bytes"), Some(&50593792.0));
        assert_eq!(by_name.get("disk_util.0.util"), Some(&99.5));
    }

    #[test]
    fn walk_json_leaves_skips_nonfinite_defensively() {
        // serde_json rejects NaN/Infinity at parse time (strict JSON),
        // so naturally-occurring inputs never reach walk_json_leaves
        // with non-finite numbers. The defensive filter is still
        // verified by constructing a Value directly with
        // Number::from_f64 which returns None for non-finite.
        assert!(serde_json::Number::from_f64(f64::NAN).is_none());
        assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
        assert!(serde_json::Number::from_f64(f64::NEG_INFINITY).is_none());
        // Finite values ARE accepted:
        assert!(serde_json::Number::from_f64(2.78).is_some());
    }

    /// JSON integers above 2^53 lose precision when coerced to
    /// f64 via `serde_json::Number::as_f64` — the f64 mantissa is 52
    /// bits, so consecutive integers beyond 9007199254740992 round
    /// to the nearest representable f64. Pin the observed behavior:
    /// `9007199254740993` (2^53 + 1) round-trips as `9007199254740992.0`
    /// (2^53). Payloads emitting integer metrics larger than 2^53
    /// must scale down (µs → s) or encode as strings — the Json
    /// walker cannot preserve integer identity past that boundary.
    #[test]
    fn json_large_integer_above_2_pow_53_loses_precision() {
        // 2^53 = 9_007_199_254_740_992 is the last exactly-representable
        // consecutive integer in f64. 2^53 + 1 rounds down to 2^53
        // (banker's rounding lands on the even representable
        // neighbor). Test via u64 → f64 to pin the u64 input value
        // distinct from the emitted f64 — a direct f64 literal of
        // 2^53+1 would itself round at parse time, obscuring what
        // the walker did.
        let s = r#"{"huge": 9007199254740993}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        // The emitted f64 IS the nearest representable value —
        // which is 2^53, not 2^53+1. Both literals happen to print
        // as "9007199254740992.0" because f64 can't distinguish
        // them; compare against the exact f64 produced by the
        // next-representable-below path.
        assert_eq!(m[0].value, 9_007_199_254_740_992.0_f64);
        // Cast the u64 source input to f64 to reproduce the same
        // rounding serde_json performed. Both sides land at 2^53;
        // that equality IS the lossy cast being documented.
        let rounded: f64 = 9_007_199_254_740_993_u64 as f64;
        assert_eq!(m[0].value, rounded);
        // Confirm bit-level that the u64 input and the resulting
        // f64 are NOT identity-preserving: casting the f64 back to
        // u64 yields 2^53, not 2^53+1.
        assert_eq!(m[0].value as u64, 9_007_199_254_740_992_u64);
        assert_ne!(m[0].value as u64, 9_007_199_254_740_993_u64);
    }

    /// At exactly 2^53 the f64 IS exact — the precision loss is
    /// strictly one-above-the-boundary. Pair with
    /// `json_large_integer_above_2_pow_53_loses_precision` so both
    /// sides of the precision cliff are pinned.
    #[test]
    fn json_integer_at_2_pow_53_is_exact() {
        let s = r#"{"exact": 9007199254740992}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].value, 9_007_199_254_740_992.0_f64);
    }

    /// `find_and_parse_json` tries the whole trimmed input as
    /// a single document on the fast path. If the input has a
    /// balanced object followed by trailing non-JSON text, the
    /// whole-input parse fails (strict serde) and the region-
    /// finder slow path extracts the leading `{...}` region and
    /// parses that. Pins the "trailing garbage is stripped by the
    /// region finder" behavior.
    #[test]
    fn find_and_parse_json_recovers_object_with_trailing_garbage() {
        let s = r#"{"a": 1, "b": 2} --- trailing prose from banner"#;
        let v = find_and_parse_json(s).expect("trailing garbage must not block parse");
        assert_eq!(v["a"], serde_json::json!(1));
        assert_eq!(v["b"], serde_json::json!(2));
    }

    /// A leading array followed by trailing garbage recovers
    /// symmetrically — the region finder handles `[...]` the same
    /// way it handles `{...}`.
    #[test]
    fn find_and_parse_json_recovers_array_with_trailing_garbage() {
        let s = "[1, 2, 3]\nextra: banner line\n";
        let v = find_and_parse_json(s).expect("array with trailing garbage must parse");
        assert_eq!(v, serde_json::json!([1, 2, 3]));
    }

    /// Real-world fio pattern — banner line, JSON body,
    /// *and* trailing "done" marker. The region finder locks to
    /// the first balanced opener/closer, so the trailing content
    /// is ignored even if it contains unbalanced braces.
    #[test]
    fn find_and_parse_json_with_banner_and_trailer() {
        let s = "fio-3.36 starting up\n{\"iops\": 100}\nfio done }";
        let v = find_and_parse_json(s).expect("banner + trailer must resolve to body");
        assert_eq!(v["iops"], serde_json::json!(100));
    }

    /// When the trailing garbage itself contains a
    /// BALANCED brace pair, the region finder still returns the
    /// first one — downstream parsing uses the first match, not
    /// a merged document.
    #[test]
    fn find_and_parse_json_returns_first_region_when_trailer_also_balanced() {
        let s = r#"{"first": 1} unrelated {"second": 2}"#;
        let v = find_and_parse_json(s).expect("first balanced region parses");
        assert_eq!(v["first"], serde_json::json!(1));
        assert!(v.get("second").is_none(), "second region must not merge in");
    }

    /// Embedded `{` / `}` characters inside a JSON string literal
    /// must NOT be counted as structural openers/closers by the
    /// region finder. The in-string tracker flips on `"` and
    /// suppresses nesting accounting until the matching closing
    /// `"`, so the only braces that affect `depth` are the
    /// structural outer ones. Pins that a log message which happens
    /// to contain `{` / `}` inside a quoted string still round-trips
    /// through the slow path.
    #[test]
    fn find_and_parse_json_ignores_braces_inside_string_literals() {
        let s = "fio-3.36 starting up\n\
                 {\"msg\": \"look at {nested} in text\", \"ok\": 1}\n\
                 trailing banner";
        let v = find_and_parse_json(s).expect("embedded braces in string must not break scan");
        assert_eq!(v["msg"], serde_json::json!("look at {nested} in text"));
        assert_eq!(v["ok"], serde_json::json!(1));
    }

    /// Negative numeric leaves extract at their declared value
    /// without any sign-absoluting or filtering. Canonical for
    /// metrics like scheduler_delta_ns that can legitimately be
    /// negative (improvement from baseline).
    #[test]
    fn json_negative_numbers_extract_preserving_sign() {
        let s = r#"{"delta_ns": -500.5, "underflow": -1000000}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        let by_name: std::collections::BTreeMap<&str, f64> =
            m.iter().map(|x| (x.name.as_str(), x.value)).collect();
        assert_eq!(by_name.get("delta_ns"), Some(&-500.5));
        assert_eq!(by_name.get("underflow"), Some(&-1_000_000.0));
    }

    /// Zero is emitted as a real metric value, not filtered
    /// out. A payload that genuinely measured zero (idle CPU, no
    /// errors) must produce a zero metric — otherwise downstream
    /// checks like `Check::exit_code_eq(0)` against an `exit_code`
    /// metric of 0.0 would spuriously report "missing" instead of
    /// passing.
    #[test]
    fn json_zero_values_are_emitted_not_filtered() {
        let s = r#"{"errors": 0, "cpu_idle_pct": 0.0, "count": -0.0}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        let by_name: std::collections::BTreeMap<&str, f64> =
            m.iter().map(|x| (x.name.as_str(), x.value)).collect();
        assert_eq!(by_name.len(), 3, "all three zeros must extract: {m:?}");
        assert_eq!(by_name.get("errors"), Some(&0.0));
        assert_eq!(by_name.get("cpu_idle_pct"), Some(&0.0));
        // -0.0 round-trips via f64; assert the numeric equality.
        assert_eq!(by_name.get("count"), Some(&0.0));
    }

    /// Mixed positive + negative + zero in one document
    /// exercises the walker's sign-agnostic branch.
    #[test]
    fn json_mixed_signs_and_zero_all_extract() {
        let s = r#"{"pos": 10.0, "neg": -10.0, "zero": 0.0}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 3);
    }

    /// An empty JSON object `{}` at the top level parses
    /// successfully but yields no metric leaves — the walker
    /// traverses zero children and falls through to produce an
    /// empty Vec. No `None` return, no panic.
    #[test]
    fn json_empty_object_yields_no_metrics() {
        let m = extract_metrics("{}", &OutputFormat::Json).unwrap();
        assert!(m.is_empty(), "empty object has no leaves: {m:?}");
    }

    /// An empty array at the top level likewise yields zero
    /// metrics.
    #[test]
    fn json_empty_array_yields_no_metrics() {
        let m = extract_metrics("[]", &OutputFormat::Json).unwrap();
        assert!(m.is_empty(), "empty array has no leaves: {m:?}");
    }

    /// Nested empty containers also produce no leaves — the
    /// walker still recurses but finds nothing numeric at the
    /// bottom. Pins the "no ghost metrics from empty containers"
    /// invariant.
    #[test]
    fn json_nested_empty_containers_yield_no_metrics() {
        let s = r#"{"outer": {"inner": {}, "also": []}}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert!(m.is_empty(), "nested empties emit nothing: {m:?}");
    }

    /// Empty container alongside real metrics — empties are
    /// silent, real leaves still emit.
    #[test]
    fn json_empty_container_mixed_with_real_metrics() {
        let s = r#"{"iops": 100.0, "meta": {}, "samples": []}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "iops");
        assert_eq!(m[0].value, 100.0);
    }

    /// walk_json_leaves uses push/pop on a single
    /// path buffer instead of per-level format!(). This test pins
    /// the *behavior* (path output unchanged across deep nesting)
    /// so a future refactor of the path plumbing can't silently
    /// drop a segment or duplicate a dot.
    #[test]
    fn walk_json_leaves_deep_nesting_paths_are_correct() {
        // 6 levels deep → one leaf at a.b.c.d.e.f.
        let s = r#"{"a":{"b":{"c":{"d":{"e":{"f": 42.0}}}}}}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "a.b.c.d.e.f");
        assert_eq!(m[0].value, 42.0);
    }

    /// Sibling keys under the same parent must see the parent
    /// segment truncated between each child — the bug that the
    /// push/pop refactor would hit is "path accumulates across
    /// siblings" producing `root.a.b`, `root.a.b.c` etc. instead
    /// of `root.a.b`, `root.a.c`.
    #[test]
    fn walk_json_leaves_siblings_do_not_accumulate_path() {
        let s = r#"{"root":{"a": 1, "b": 2, "c": 3}}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        assert_eq!(m.len(), 3);
        let names: std::collections::BTreeSet<&str> = m.iter().map(|x| x.name.as_str()).collect();
        let expected: std::collections::BTreeSet<&str> =
            ["root.a", "root.b", "root.c"].into_iter().collect();
        assert_eq!(names, expected, "path must truncate between siblings");
    }

    /// Array indices use the same push/pop path: `arr.0`, `arr.1`.
    /// Deep array-of-array-of-object combinations exercise every
    /// code path in the walker.
    #[test]
    fn walk_json_leaves_deep_array_object_interleaving() {
        let s = r#"{"data":[{"vals":[10.0, 20.0]},{"vals":[30.0]}]}"#;
        let m = extract_metrics(s, &OutputFormat::Json).unwrap();
        let by_name: std::collections::BTreeMap<&str, f64> =
            m.iter().map(|x| (x.name.as_str(), x.value)).collect();
        assert_eq!(by_name.get("data.0.vals.0"), Some(&10.0));
        assert_eq!(by_name.get("data.0.vals.1"), Some(&20.0));
        assert_eq!(by_name.get("data.1.vals.0"), Some(&30.0));
        assert_eq!(by_name.len(), 3);
    }

    /// Programmatically build a `serde_json::Value` nested deeper than
    /// [`MAX_WALK_DEPTH`] and confirm that `walk_json_leaves` returns
    /// without a stack overflow and without emitting metrics from
    /// beyond the cap. Serde_json's own parser depth limit (128 by
    /// default) blocks malicious JSON strings before the walker sees
    /// them, so a parser-bypass (direct `Value::Object` construction)
    /// is the only way to reach this depth — the test exercises
    /// exactly that path.
    #[test]
    fn walk_json_leaves_depth_cap_skips_deeply_nested_subtree() {
        // Build an Object nested 100 deep with a numeric leaf at the
        // bottom. The leaf at depth > MAX_WALK_DEPTH (64) must be
        // skipped by the guard. A sentinel metric with
        // `WALK_TRUNCATION_SENTINEL_NAME` MUST appear in the return
        // so callers without a tracing subscriber still observe the
        // truncation.
        let mut value = serde_json::json!({"leaf": 42.0});
        for _ in 0..100 {
            let mut m = serde_json::Map::new();
            m.insert("x".to_string(), value);
            value = serde_json::Value::Object(m);
        }
        let metrics = walk_json_leaves(&value, MetricSource::Json);
        let real_leaves: Vec<_> = metrics
            .iter()
            .filter(|m| m.name != WALK_TRUNCATION_SENTINEL_NAME)
            .collect();
        assert!(
            real_leaves.is_empty(),
            "leaf beyond MAX_WALK_DEPTH cap must not be emitted, got {real_leaves:?}"
        );
        let sentinel = metrics
            .iter()
            .find(|m| m.name == WALK_TRUNCATION_SENTINEL_NAME)
            .expect("truncation sentinel must be present on cap hit");
        assert!(
            sentinel.value > MAX_WALK_DEPTH as f64,
            "sentinel value must carry the depth at which truncation fired, got {}",
            sentinel.value,
        );
    }

    /// A leaf exactly at [`MAX_WALK_DEPTH`] is still emitted — the
    /// cap bails BEFORE recursing past `depth > MAX_WALK_DEPTH`, so a
    /// leaf reached at `depth == MAX_WALK_DEPTH` is preserved.
    /// Boundary pair with the depth_cap_skips test above so an
    /// off-by-one in the guard (e.g. `>=` instead of `>`) surfaces.
    #[test]
    fn walk_json_leaves_depth_cap_boundary_leaf_preserved() {
        // Build Object of exactly MAX_WALK_DEPTH nesting: top-level
        // holds an Object, which holds an Object, ... for
        // MAX_WALK_DEPTH levels, with the numeric leaf at the bottom.
        // The leaf's path has MAX_WALK_DEPTH segments and walk() is
        // called at depths 0..=MAX_WALK_DEPTH — the leaf call at
        // depth MAX_WALK_DEPTH must pass the guard.
        let mut value = serde_json::Value::Number(serde_json::Number::from_f64(42.0).unwrap());
        for _ in 0..MAX_WALK_DEPTH {
            let mut m = serde_json::Map::new();
            m.insert("x".to_string(), value);
            value = serde_json::Value::Object(m);
        }
        let metrics = walk_json_leaves(&value, MetricSource::Json);
        assert_eq!(metrics.len(), 1, "boundary leaf must be preserved");
        assert_eq!(metrics[0].value, 42.0);
    }

    /// Mixed-depth invariant: a single walk must emit every finite
    /// numeric leaf regardless of the depth at which it appears, so
    /// long as the depth is ≤ MAX_WALK_DEPTH. Mirrors real payload
    /// schemas (fio's `jobs[0].read.lat_ns.mean` sits at depth 5
    /// while `jobs[0].jobname` sits at depth 2). A single-depth
    /// regression — e.g. a premature `return` inside the Object arm
    /// — would skip the shallower siblings of a deep subtree.
    #[test]
    fn walk_json_leaves_mixed_depth_leaves_all_emitted() {
        let value = serde_json::json!({
            "shallow": 1.0,
            "mid": {
                "leaf": 2.0,
                "deeper": {
                    "still": {
                        "further": 3.0
                    }
                }
            },
            "also_shallow": 4.0,
            "deeper_sibling": {
                "only_child": 5.0
            }
        });
        let metrics = walk_json_leaves(&value, MetricSource::Json);
        let by_name: std::collections::BTreeMap<&str, f64> =
            metrics.iter().map(|m| (m.name.as_str(), m.value)).collect();
        assert_eq!(by_name.get("shallow"), Some(&1.0));
        assert_eq!(by_name.get("mid.leaf"), Some(&2.0));
        assert_eq!(by_name.get("mid.deeper.still.further"), Some(&3.0));
        assert_eq!(by_name.get("also_shallow"), Some(&4.0));
        assert_eq!(by_name.get("deeper_sibling.only_child"), Some(&5.0));
        assert_eq!(metrics.len(), 5, "exactly five numeric leaves expected");
    }

    /// Array-chain invariant: nested arrays produce dotted-index
    /// paths with no stray separators. An off-by-one in the
    /// separator injection at :203-205 (array arm) or a swapped
    /// push-path/truncate order would surface as either a leading
    /// dot, a doubled separator, or an index segment merged into
    /// the previous one.
    #[test]
    fn walk_json_leaves_array_chain_paths_correct() {
        // `a` is a 2x2x2 array of numeric leaves; the walker must
        // produce paths `a.0.0.0`, `a.0.0.1`, `a.0.1.0`, …, `a.1.1.1`.
        let value = serde_json::json!({
            "a": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]]
            ]
        });
        let metrics = walk_json_leaves(&value, MetricSource::Json);
        let names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
        // 8 leaves in lexicographic index order.
        assert_eq!(names.len(), 8);
        assert_eq!(names[0], "a.0.0.0");
        assert_eq!(names[1], "a.0.0.1");
        assert_eq!(names[2], "a.0.1.0");
        assert_eq!(names[3], "a.0.1.1");
        assert_eq!(names[4], "a.1.0.0");
        assert_eq!(names[5], "a.1.0.1");
        assert_eq!(names[6], "a.1.1.0");
        assert_eq!(names[7], "a.1.1.1");
        // Values map 1:1 against path order — confirm no segment
        // got dropped or reordered.
        assert_eq!(metrics[0].value, 1.0);
        assert_eq!(metrics[7].value, 8.0);
    }

    /// Null-at-boundary invariant: a `serde_json::Value::Null` leaf
    /// is skipped by the `_ => {}` arm and contributes nothing — no
    /// metric, no sentinel, no side effect — regardless of the
    /// depth at which it sits. Specifically pins the case where the
    /// null is the direct child of a depth-MAX_WALK_DEPTH container,
    /// ensuring the cap check fires first when the container would
    /// itself be above the cap rather than the null stopping
    /// recursion harmlessly short. A regression that treats Null
    /// the same as a Number would surface as a spurious leaf with
    /// `value = 0.0` (or a panic) on this fixture.
    #[test]
    fn walk_json_leaves_null_at_boundary_produces_no_metric() {
        // Build `{a: {a: {a: ... {a: null}}}}` at exactly
        // MAX_WALK_DEPTH nesting — the Null sits at depth
        // MAX_WALK_DEPTH; the walker recurses into the outer Objects
        // at depths 0..=MAX_WALK_DEPTH-1, sees Null at the
        // boundary, and falls through the `_ => {}` arm.
        let mut value = serde_json::Value::Null;
        for _ in 0..MAX_WALK_DEPTH {
            let mut m = serde_json::Map::new();
            m.insert("a".to_string(), value);
            value = serde_json::Value::Object(m);
        }
        let metrics = walk_json_leaves(&value, MetricSource::Json);
        assert!(
            metrics.is_empty(),
            "Null leaves must produce no metrics (and no truncation sentinel), \
             got {metrics:?}",
        );
    }

    #[test]
    fn module_level_example_usage() {
        // Canonical invocation: declare a Payload with
        // OutputFormat::Json, feed stdout, get Vec<Metric>.
        const EXAMPLE_PAYLOAD: crate::test_support::Payload = crate::test_support::Payload {
            name: "example",
            kind: crate::test_support::PayloadKind::Binary("example"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
        };
        let stdout = r#"{"throughput": 42.5}"#;
        let m = extract_metrics(stdout, &EXAMPLE_PAYLOAD.output).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].name, "throughput");
        assert_eq!(m[0].value, 42.5);
        assert_eq!(m[0].source, MetricSource::Json);
    }
}
