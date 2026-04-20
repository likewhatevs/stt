//! Local LLM model cache + LlmExtract runtime for
//! [`OutputFormat::LlmExtract`] payloads.
//!
//! `OutputFormat::LlmExtract` routes stdout through a small local
//! model that emits JSON, which the existing
//! [`walk_json_leaves`](crate::test_support::metrics) pipeline then
//! consumes. The model binary itself lives under
//! `~/.cache/ktstr/models/`. This module owns both the cache
//! surface (locate + fetch + verify) and the LlmExtract pipeline
//! that composes a prompt, invokes inference, and routes the
//! response back through the JSON walker.
//!
//! # Cache layout
//!
//! The cache root follows the same resolution order as [`crate::cache`]
//! (kernel images):
//!
//! 1. `KTSTR_CACHE_DIR` — explicit override.
//! 2. `$XDG_CACHE_HOME/ktstr/models/`.
//! 3. `$HOME/.cache/ktstr/models/`.
//!
//! Each cache entry is `{cache_root}/{model.file_name}`. Downloads
//! land in a tempfile next to the final path and atomically
//! `rename()` into place only after SHA-256 matches the declared
//! pin, so a killed process never leaves a partial file masquerading
//! as a cached model.
//!
//! # Eager-conditional prefetch
//!
//! [`prefetch_if_required`] scans [`KTSTR_TESTS`] for any registered
//! entry whose payload or workloads declare
//! [`OutputFormat::LlmExtract`] and invokes [`ensure`] when at least
//! one match is found. Offline runs set `KTSTR_MODEL_OFFLINE=1` to
//! skip the fetch entirely; a missing model then surfaces as a
//! per-test failure rather than a nextest setup abort, which matches
//! the semantics test authors already expect from other offline env
//! gates.
//!
//! # Wiring into `nextest_setup`
//!
//! [`prefetch_if_required`] is invoked by
//! [`nextest_setup`](crate::test_support::nextest_setup) during
//! nextest's test-setup phase, after the kernel + initramfs warm
//! but before any test body executes. Tests that need
//! `OutputFormat::LlmExtract` therefore observe the model either
//! already cached on disk or missing with a surfaced fetch error —
//! they never race against a half-downloaded artifact. Tests that
//! do not declare LlmExtract skip the prefetch entirely thanks to
//! [`any_test_requires_model`]'s scan.
//!
//! # LlmExtract extraction pipeline
//!
//! [`extract_via_llm`] is the runtime entry point called by
//! [`extract_metrics`](crate::test_support::extract_metrics) when a
//! payload's [`OutputFormat::LlmExtract`] fires:
//!
//! 1. [`compose_prompt`] assembles `{LLM_EXTRACT_PROMPT_TEMPLATE}\n\n{focus}STDOUT:\n{body}`.
//! 2. [`invoke_inference`] runs the (currently stubbed) backend.
//! 3. On `Ok`, [`super::metrics::find_and_parse_json`] extracts the
//!    JSON region; parsed values flow through
//!    [`super::metrics::walk_json_leaves`] pre-tagged with
//!    [`MetricSource::LlmExtract`](crate::test_support::MetricSource::LlmExtract).
//! 4. A JSON-parse failure retries the inference call ONCE. Infra
//!    errors (backend not wired, model missing) never retry — the
//!    second call would hit the same wall.

use anyhow::{Context, Result};
use std::path::PathBuf;

use super::KTSTR_TESTS;
use super::payload::OutputFormat;

/// Pinned description of a model artifact the cache knows how to
/// fetch and verify.
///
/// The fields are `&'static` so a `ModelSpec` can live in a
/// top-level `const` — the default model used by
/// [`OutputFormat::LlmExtract`] sits in [`DEFAULT_MODEL`] and the
/// tests below cover the invariants (size sanity, URL+SHA shape)
/// without any heap allocation.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    /// Human-readable identifier embedded in status output. Also used
    /// as the cache filename (concatenated with `suffix`) so two
    /// distinct pins never overwrite each other.
    pub file_name: &'static str,
    /// HTTPS URL the fetcher downloads from. `http://` is rejected
    /// before the request issues so a placeholder URL typo doesn't
    /// pull bytes over cleartext.
    pub url: &'static str,
    /// Hex-encoded SHA-256 digest of the expected file. Case-
    /// insensitive; the comparator normalizes both sides to lower.
    pub sha256_hex: &'static str,
    /// Approximate on-disk size in bytes; surfaced in status output
    /// so users can tell at a glance whether the cache entry is the
    /// right artifact. Not used for verification (SHA is the gate).
    pub size_bytes: u64,
}

/// Default model served when a payload declares
/// [`OutputFormat::LlmExtract`] without pointing at a custom pin.
///
/// URL + SHA-256 are placeholder constants until the real Qwen2.5
/// 0.5B Q4 artifact is mirrored. The SHA is not the zero digest
/// (that would silently validate an empty file); instead the all-
/// `?` marker trips the hex-decode step in [`verify_sha256`] with a
/// clear error.
pub const DEFAULT_MODEL: ModelSpec = ModelSpec {
    file_name: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    url: "https://UNPINNED-model-url.example/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    sha256_hex: "????????????????????????????????????????????????????????????????",
    size_bytes: 400 * 1024 * 1024,
};

/// Environment variable that opts out of the eager prefetch.
/// `KTSTR_MODEL_OFFLINE=1` (or any non-empty value) leaves the cache
/// untouched; `LlmExtract` tests then surface missing-model errors
/// at invocation time instead of at nextest setup.
pub const OFFLINE_ENV: &str = "KTSTR_MODEL_OFFLINE";

/// Read [`OFFLINE_ENV`] and return the trimmed value IFF it is set
/// to a non-empty string. Centralizes the "non-empty env-var means
/// opt-in" predicate used by [`ensure`] and [`prefetch_if_required`]
/// so the condition stays uniform (e.g. both treat
/// `KTSTR_MODEL_OFFLINE=` as "not set" — the empty-string case).
///
/// Returns `None` when the env var is absent or set to empty
/// string. Returns `Some(value)` when set to any non-empty string;
/// callers that want to surface the user-supplied value in error
/// messages (`"KTSTR_MODEL_OFFLINE=1 set but ..."`) get it for
/// free. Callers that echo the value into user-facing output MUST
/// funnel it through [`sanitize_env_value`] first.
fn read_offline_env() -> Option<String> {
    match std::env::var(OFFLINE_ENV) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Sanitize an env-var value for inclusion in a user-facing error
/// or log message. Control characters (including TAB/CR/LF) are
/// replaced with `?` so a malicious or accidental payload cannot
/// disturb terminal state or forge log-line boundaries, and the
/// result is truncated to `MAX_ENV_ECHO_LEN` bytes with an ellipsis
/// marker when longer so a multi-kilobyte value doesn't blow up the
/// error line. The returned string is always ASCII-safe-to-display.
fn sanitize_env_value(raw: &str) -> String {
    const MAX_ENV_ECHO_LEN: usize = 64;
    let mut cleaned: String = raw
        .chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect();
    if cleaned.len() > MAX_ENV_ECHO_LEN {
        // Truncate on a char boundary by collecting a prefix of chars
        // whose cumulative byte-length stays within the budget. Saves
        // allocating a shrink_to then fixing up a mid-codepoint cut.
        let mut end = 0usize;
        for (idx, c) in cleaned.char_indices() {
            let next = idx + c.len_utf8();
            if next > MAX_ENV_ECHO_LEN {
                break;
            }
            end = next;
        }
        cleaned.truncate(end);
        cleaned.push_str("...");
    }
    cleaned
}

/// Status record returned by [`status`]: where the model would live
/// on disk and whether a verified copy is already there.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub spec: ModelSpec,
    pub path: PathBuf,
    pub cached: bool,
    pub sha_matches: bool,
}

/// Resolve the cache root, creating it lazily when a writer needs it.
/// Mirrors [`crate::cache`]'s kernel cache resolver so the same env
/// overrides (`KTSTR_CACHE_DIR`, `XDG_CACHE_HOME`) govern both.
pub(crate) fn resolve_cache_root() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("KTSTR_CACHE_DIR")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Ok(PathBuf::from(xdg).join("ktstr").join("models"));
    }
    let home = std::env::var("HOME").map_err(|_| {
        anyhow::anyhow!(
            "HOME not set; cannot resolve model cache directory. \
             Set KTSTR_CACHE_DIR to specify a cache location."
        )
    })?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("ktstr")
        .join("models"))
}

/// Return the on-disk path the spec would occupy and whether a
/// verified copy is already present. Used by both the CLI's
/// `model status` subcommand and the eager prefetch fast-path.
pub fn status(spec: &ModelSpec) -> Result<ModelStatus> {
    let root = resolve_cache_root()?;
    let path = root.join(spec.file_name);
    let (cached, sha_matches) = match std::fs::metadata(&path) {
        Ok(meta) if meta.is_file() => {
            // A cached file is considered "matched" only when the
            // SHA agrees with the pin; anything else is a corrupt /
            // interrupted download that ensure() will replace.
            let matches = verify_sha256(&path, spec.sha256_hex).unwrap_or(false);
            (true, matches)
        }
        _ => (false, false),
    };
    Ok(ModelStatus {
        spec: *spec,
        path,
        cached,
        sha_matches,
    })
}

/// Ensure the model artifact described by `spec` is present and
/// SHA-verified in the cache, downloading if necessary.
///
/// Fast path: existing file whose SHA matches — no-op.
/// Slow path: tempfile download + SHA verify + atomic rename.
///
/// Respects `KTSTR_MODEL_OFFLINE`: when set to a non-empty value,
/// returns `Err` immediately without issuing a network request. This
/// lets CI pipelines that pre-seed the cache fail loudly when the
/// pre-seed mechanism skipped an artifact, rather than silently
/// falling through to an online fetch.
pub fn ensure(spec: &ModelSpec) -> Result<PathBuf> {
    let st = status(spec)?;
    if st.cached && st.sha_matches {
        return Ok(st.path);
    }
    if let Some(v) = read_offline_env() {
        let v_safe = sanitize_env_value(&v);
        anyhow::bail!(
            "{OFFLINE_ENV}={v_safe} set but model '{}' is not cached at {}; \
             pre-seed the cache or unset {OFFLINE_ENV} to fetch.",
            spec.file_name,
            st.path.display(),
        );
    }
    fetch(spec, &st.path)
}

/// Download the spec to `final_path` through a tempfile, verify SHA,
/// then atomically rename. Errors are actionable (includes URL +
/// final path) so a test author can reproduce the fetch by hand.
fn fetch(spec: &ModelSpec, final_path: &std::path::Path) -> Result<PathBuf> {
    reject_insecure_url(spec.url)?;
    let parent = final_path.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "model cache path {} has no parent directory",
            final_path.display()
        )
    })?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create model cache dir {}", parent.display()))?;

    // NamedTempFile keeps the partial artifact next to the final
    // path so the subsequent rename is an atomic filesystem op
    // (same filesystem guaranteed). A tempfile in /tmp could sit on
    // a separate fs and fall back to a copy+remove under the hood.
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create tempfile in {}", parent.display()))?;
    let tmp_path = tmp.path().to_path_buf();

    // Use an explicit Client with connect + overall timeouts rather
    // than `reqwest::blocking::get`, which has no timeout and will
    // hang forever on a slow or unreachable mirror. 5m overall is
    // enough for a few-hundred-MB model download on a typical
    // connection; 30s connect catches DNS/TLS wedges early. Tests
    // that don't actually hit the network (offline gate, cached
    // path) never enter this branch.
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("build reqwest::blocking::Client for model fetch")?;
    let mut response = client
        .get(spec.url)
        .send()
        .with_context(|| format!("GET {} (download model '{}')", spec.url, spec.file_name))?;
    if !response.status().is_success() {
        anyhow::bail!(
            "GET {} returned HTTP {} — download of model '{}' failed",
            spec.url,
            response.status(),
            spec.file_name,
        );
    }
    // Stream the body straight into the tempfile via `std::io::copy`
    // so a 400 MiB model doesn't first materialize in a heap Vec.
    // `response` implements `std::io::Read`; the tempfile handle
    // from `NamedTempFile` implements `Write`. Previous buffer-then-
    // write approach held the full body in memory (#116, #106).
    {
        use std::io::Write;
        let file = tmp.as_file_mut();
        let mut writer = std::io::BufWriter::new(file);
        std::io::copy(&mut response, &mut writer)
            .with_context(|| format!("stream body from {} to {}", spec.url, tmp_path.display()))?;
        writer
            .flush()
            .with_context(|| format!("flush {} after body stream", tmp_path.display()))?;
    }

    if !verify_sha256(&tmp_path, spec.sha256_hex)? {
        anyhow::bail!(
            "SHA-256 mismatch for model '{}' downloaded from {}: expected {}, \
             got something else. Pin or source is wrong; refusing to cache \
             the bytes.",
            spec.file_name,
            spec.url,
            spec.sha256_hex,
        );
    }

    tmp.persist(final_path).map_err(|e| {
        anyhow::anyhow!(
            "atomically move {} to {}: {}",
            tmp_path.display(),
            final_path.display(),
            e.error,
        )
    })?;
    Ok(final_path.to_path_buf())
}

/// Return `Ok(true)` when the file's SHA-256 matches the expected
/// hex pin (case-insensitive), `Ok(false)` otherwise. `Err` only on
/// I/O error reading the file or a malformed expected hex string
/// (non-64 chars / non-hex chars), which would render the check
/// itself useless and must surface instead of silently pretending
/// the file is good.
fn verify_sha256(path: &std::path::Path, expected_hex: &str) -> Result<bool> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    if expected_hex.len() != 64 {
        anyhow::bail!(
            "expected SHA-256 hex must be 64 chars, got {} ({:?})",
            expected_hex.len(),
            expected_hex,
        );
    }
    if !expected_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        anyhow::bail!(
            "expected SHA-256 hex contains non-hex chars: {:?}",
            expected_hex,
        );
    }

    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = hex_encode(&hasher.finalize());
    Ok(got.eq_ignore_ascii_case(expected_hex))
}

/// Lowercase hex encoder — avoids pulling in the `hex` crate for a
/// 64-byte-output helper used exactly twice (verify + debug).
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(*b >> 4) as usize] as char);
        out.push(HEX[(*b & 0x0f) as usize] as char);
    }
    out
}

/// Reject `http://` URLs so a placeholder typo can't leak the SHA-
/// pinned artifact request over cleartext. The fetcher is only ever
/// correct for `https://`.
fn reject_insecure_url(url: &str) -> Result<()> {
    if !url.starts_with("https://") {
        anyhow::bail!("model cache fetcher refuses non-HTTPS URL: {}", url,);
    }
    Ok(())
}

/// True iff any entry in `KTSTR_TESTS` declares
/// [`OutputFormat::LlmExtract`] on its primary payload or any
/// workload. The prefetcher uses this to decide whether the fetch is
/// worth attempting — a scheduler-only or binary-only test run does
/// not need the model.
pub fn any_test_requires_model() -> bool {
    KTSTR_TESTS.iter().any(|entry| {
        let primary_needs = entry
            .payload
            .is_some_and(|p| matches!(p.output, OutputFormat::LlmExtract(_)));
        let workload_needs = entry
            .workloads
            .iter()
            .any(|w| matches!(w.output, OutputFormat::LlmExtract(_)));
        primary_needs || workload_needs
    })
}

/// Prefetch [`DEFAULT_MODEL`] when at least one registered test
/// needs it. No-op when `KTSTR_MODEL_OFFLINE` is set (skips fetch,
/// leaves per-test failures to surface downstream) or when no test
/// declares [`OutputFormat::LlmExtract`].
///
/// Returns `Ok(None)` when no fetch was attempted; `Ok(Some(path))`
/// when the model is now cached; `Err` on fetch/verify failure.
pub fn prefetch_if_required() -> Result<Option<PathBuf>> {
    if !any_test_requires_model() {
        return Ok(None);
    }
    if let Some(v) = read_offline_env() {
        let v_safe = sanitize_env_value(&v);
        eprintln!("ktstr: {OFFLINE_ENV}={v_safe} set; skipping eager model prefetch");
        return Ok(None);
    }
    ensure(&DEFAULT_MODEL).map(Some)
}

// ---------------------------------------------------------------------------
// LlmExtract runtime
// ---------------------------------------------------------------------------

/// Default prompt template prepended to every
/// [`OutputFormat::LlmExtract`] invocation. Kept here as a const so
/// tests can assert its exact contents (retry-semantics tests in
/// particular pin the prompt so a silent wording drift doesn't
/// invalidate them).
///
/// The wording is deliberately terse: the model's role is narrow —
/// look at benchmark stdout, produce a single JSON object of
/// numeric leaves. Every word that isn't load-bearing here costs
/// context tokens on a tiny local model.
pub(crate) const LLM_EXTRACT_PROMPT_TEMPLATE: &str = "\
You are a benchmark-output parser. Read the following program stdout \
and emit ONLY a single JSON object whose keys are metric names \
(dotted paths for nested values are fine) and whose values are \
numbers. No prose, no code fences, no commentary. If no numeric \
metrics are present, emit `{}`.";

/// Compose the full prompt sent to the inference backend for an
/// [`OutputFormat::LlmExtract`] invocation.
///
/// Shape: `{TEMPLATE}\n\n{hint_line}STDOUT:\n{stdout}` — the hint is
/// appended as its own line before the stdout block so the model
/// sees the user-declared focus before the raw content. An empty or
/// absent hint degrades to the bare template without leaving a
/// dangling "Focus:" header.
pub(crate) fn compose_prompt(stdout: &str, hint: Option<&str>) -> String {
    let mut out = String::with_capacity(
        LLM_EXTRACT_PROMPT_TEMPLATE.len() + stdout.len() + 64 + hint.map_or(0, |h| h.len() + 16),
    );
    out.push_str(LLM_EXTRACT_PROMPT_TEMPLATE);
    out.push_str("\n\n");
    if let Some(h) = hint
        && !h.trim().is_empty()
    {
        out.push_str("Focus: ");
        out.push_str(h.trim());
        out.push_str("\n\n");
    }
    out.push_str("STDOUT:\n");
    out.push_str(stdout);
    out
}

/// Errors surfaced by [`invoke_inference`]. Kept distinct from the
/// top-level `anyhow::Error` so [`extract_via_llm`] can decide
/// whether to retry (parse failures retry; infra failures don't).
#[derive(Debug)]
pub(crate) enum InferenceError {
    /// The backend itself is not wired yet. The prompt was never
    /// evaluated; retrying would hit the same wall and waste
    /// wallclock.
    NotWired,
    /// The backend ran but failed to load the model artifact
    /// (missing file, corrupt GGUF, etc.). Surfaces via `ensure()`
    /// errors so test authors can tell whether the cache is stale
    /// or the pipeline is broken.
    ModelLoad(anyhow::Error),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::NotWired => write!(
                f,
                "LlmExtract inference backend is not yet wired in this build; \
                 set KTSTR_MODEL_OFFLINE=1 to skip, or declare OutputFormat::Json \
                 if your payload emits structured stdout natively"
            ),
            InferenceError::ModelLoad(e) => write!(f, "LlmExtract model load: {e:#}"),
        }
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            InferenceError::NotWired => None,
            InferenceError::ModelLoad(e) => e.source(),
        }
    }
}

/// Invoke the local inference backend with `prompt` and return the
/// model's raw string output (expected to carry a JSON object per
/// [`LLM_EXTRACT_PROMPT_TEMPLATE`]).
///
/// **This function runs on the HOST, not inside the guest VM.**
/// `extract_via_llm` (and by extension `extract_metrics`) is
/// invoked host-side after the guest's payload has finished and
/// its stdout has been captured via the sidecar pipeline. The
/// model cache lives at `~/.cache/ktstr/models/` on the host and
/// is consulted by the host-side test process — nothing about
/// this path touches the guest initramfs, and guest image size
/// is unaffected by backend selection.
///
/// Backend selection (candle / llama-cpp-rs / similar) is still
/// open. The blockers for wiring a real backend are host-side:
/// binary size of the host test process (candle + tokenizer
/// alone ~40 MiB), and a C/C++ toolchain dependency that would
/// affect every `cargo build` of the ktstr crate. The pipeline
/// lands ahead of the backend so downstream callers
/// (`extract_via_llm`, `extract_metrics`) wire end-to-end; the
/// backend lands as a follow-up that swaps this function's body.
///
/// `ensure(&DEFAULT_MODEL)` is still called so a misconfigured
/// cache surfaces as `ModelLoad` rather than silently passing as
/// "backend not wired". Tests that do not set `KTSTR_MODEL_OFFLINE`
/// and have no cached model therefore get an actionable error
/// naming the missing artifact.
pub(crate) fn invoke_inference(_prompt: &str) -> std::result::Result<String, InferenceError> {
    // Probe the cache so a missing / corrupt model surfaces clearly
    // rather than hiding behind NotWired. A NotWired verdict must
    // mean "backend code not yet written"; it must NOT absorb a
    // missing-artifact error that the test author could fix.
    if let Err(e) = ensure(&DEFAULT_MODEL) {
        return Err(InferenceError::ModelLoad(e));
    }
    Err(InferenceError::NotWired)
}

/// Run the full LlmExtract pipeline against `stdout` and return the
/// resulting metrics, all pre-tagged with
/// [`MetricSource::LlmExtract`](super::MetricSource::LlmExtract).
///
/// Retry semantics: the inference call is made once; on a
/// JSON-parse failure of the model's first response a second
/// inference call is made with the same prompt. A second parse
/// failure (or an infra error at any point) returns an empty
/// metric set — matching the [`extract_metrics`] contract that
/// extraction errors are non-fatal and the downstream
/// [`Check`](crate::test_support::Check) evaluation reports each
/// referenced metric as missing.
///
/// Inference-backend errors never retry — a `NotWired` backend
/// cannot succeed on a second try and retrying would only burn
/// wall time. Only JSON-parse failures get a second attempt.
pub(crate) fn extract_via_llm(stdout: &str, hint: Option<&str>) -> Vec<super::Metric> {
    let prompt = compose_prompt(stdout, hint);

    // First inference attempt.
    let response = match invoke_inference(&prompt) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ktstr: LlmExtract inference failed (attempt 1): {e}");
            return Vec::new();
        }
    };
    if let Some(json) = super::metrics::find_and_parse_json(&response) {
        return super::metrics::walk_json_leaves(&json, super::MetricSource::LlmExtract);
    }

    // Retry once — ONLY on JSON parse failure. The assumption is
    // that a tiny local model can occasionally emit a leading
    // preamble / code fence despite the prompt's "no prose"
    // instruction; a second roll of the dice often lands a clean
    // JSON response. Infra-layer failures (model missing, backend
    // not wired) don't retry because the second call would hit the
    // same wall.
    eprintln!(
        "ktstr: LlmExtract first response was not parseable JSON \
         ({} bytes); retrying once",
        response.len(),
    );
    let retry = match invoke_inference(&prompt) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ktstr: LlmExtract inference failed (attempt 2): {e}");
            return Vec::new();
        }
    };
    match super::metrics::find_and_parse_json(&retry) {
        Some(json) => super::metrics::walk_json_leaves(&json, super::MetricSource::LlmExtract),
        None => {
            eprintln!(
                "ktstr: LlmExtract retry response also not parseable JSON \
                 ({} bytes); returning empty metric set",
                retry.len(),
            );
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_cache_root_honors_ktstr_cache_dir() {
        // Nextest runs tests in parallel within a binary and
        // `std::env::set_var` is process-wide. ENV_LOCK serializes
        // the save/mutate/restore window against every other
        // env-touching test in this crate so concurrent runners in
        // sidecar.rs / eval.rs don't race on KTSTR_CACHE_DIR.
        // Poisoned-lock recovery: env tests don't establish shared
        // invariants beyond the save/restore pair, so a panic inside
        // the critical section is safe to unwrap through.
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // SAFETY: `std::env::set_var` / `remove_var` are `unsafe` in
        // Rust 2024 because a concurrent thread calling
        // `std::env::var_os` may UB-observe a half-updated environ
        // table. The guards here are:
        //   (a) `ENV_LOCK` above serializes every env-mutating test
        //       in this crate — no parallel writer exists within
        //       the ktstr test binary's threads.
        //   (b) No reader thread is spawned inside this test — the
        //       only consumers of the env var (`resolve_cache_root`)
        //       run on the test thread after the mutation returns.
        //   (c) The save-before / restore-after pair keeps other
        //       tests' environment state intact across the critical
        //       section, so a subsequent test that reads the same
        //       key sees its prior value.
        // Remaining residual risk: a signal handler or child process
        // spawned during the critical section would observe mid-
        // mutation env state. No tests in this module spawn such.
        let prev = std::env::var("KTSTR_CACHE_DIR").ok();
        unsafe { std::env::set_var("KTSTR_CACHE_DIR", "/explicit/override") };
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, PathBuf::from("/explicit/override"));
        unsafe {
            match prev {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
        }
    }

    #[test]
    fn reject_insecure_url_rejects_http() {
        let e = reject_insecure_url("http://example.com/model.gguf").unwrap_err();
        assert!(
            format!("{e:#}").contains("non-HTTPS"),
            "unexpected err: {e:#}"
        );
    }

    #[test]
    fn reject_insecure_url_accepts_https() {
        reject_insecure_url("https://example.com/model.gguf").unwrap();
    }

    #[test]
    fn hex_encode_matches_known_vectors() {
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_encode(&[0x00]), "00");
        assert_eq!(hex_encode(&[0xff]), "ff");
        assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }

    #[test]
    fn verify_sha256_matches_empty_file() {
        // SHA-256 of the empty string — a stable external anchor
        // that proves the hasher is wired correctly, independent of
        // the placeholder DEFAULT_MODEL digest.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), []).unwrap();
        let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert!(verify_sha256(tmp.path(), expected).unwrap());
    }

    #[test]
    fn verify_sha256_mismatch_returns_false() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not empty").unwrap();
        let empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        assert!(!verify_sha256(tmp.path(), empty_sha).unwrap());
    }

    #[test]
    fn verify_sha256_is_case_insensitive() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), []).unwrap();
        let upper = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
        assert!(verify_sha256(tmp.path(), upper).unwrap());
    }

    #[test]
    fn verify_sha256_rejects_malformed_hex_length() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), []).unwrap();
        let err = verify_sha256(tmp.path(), "tooshort").unwrap_err();
        assert!(format!("{err:#}").contains("64 chars"), "err: {err:#}");
    }

    #[test]
    fn verify_sha256_rejects_non_hex_chars() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), []).unwrap();
        // 64 chars but includes `?`.
        let bad = "????????????????????????????????????????????????????????????????";
        let err = verify_sha256(tmp.path(), bad).unwrap_err();
        assert!(format!("{err:#}").contains("non-hex"), "err: {err:#}");
    }

    /// #108: Non-empty short file — SHA-256 of ASCII "abc" is a
    /// well-known external anchor (NIST FIPS 180-2 appendix). Pins
    /// the non-empty happy path between the empty-file test above
    /// and the multi-chunk test below; a regression that broke
    /// single-chunk non-empty hashing would surface here.
    #[test]
    fn verify_sha256_matches_abc() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"abc").unwrap();
        // Known SHA-256("abc") — NIST FIPS 180-2 / RFC 6234 test vector.
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(verify_sha256(tmp.path(), expected).unwrap());
    }

    /// #108: Multi-chunk file (larger than a single read buffer)
    /// exercises the streaming `Read`-loop branch of `verify_sha256`
    /// (vs the single-buffer fast path for small files). 192 KiB of
    /// repeated "a" bytes is large enough to cross any reasonable
    /// BufReader default (8 KiB) multiple times; the expected SHA
    /// is computed once here from a known constant so the test
    /// remains deterministic.
    #[test]
    fn verify_sha256_matches_multi_chunk_file() {
        use sha2::{Digest, Sha256};
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // 192 KiB of 'a' bytes. 192 * 1024 = 196_608; several
        // 64 KiB BufReader refills.
        let data: Vec<u8> = std::iter::repeat_n(b'a', 192 * 1024).collect();
        std::fs::write(tmp.path(), &data).unwrap();
        // Compute the expected digest in-process so the test does
        // not hard-code a magic number against the body size.
        let mut h = Sha256::new();
        h.update(&data);
        let expected_bytes = h.finalize();
        let expected_hex = hex_encode(&expected_bytes);
        assert!(verify_sha256(tmp.path(), &expected_hex).unwrap());

        // Negative: flip one byte at the far end and verify the
        // digest rejects, proving the hasher walked past the first
        // chunk.
        let mut tampered = data;
        *tampered.last_mut().unwrap() = b'b';
        std::fs::write(tmp.path(), &tampered).unwrap();
        assert!(!verify_sha256(tmp.path(), &expected_hex).unwrap());
    }

    #[test]
    fn default_model_size_is_in_expected_ballpark() {
        // The placeholder is 400 MiB; the frozen design targets an
        // artifact in that order of magnitude. A wildly different
        // size signals someone swapped the placeholder for a
        // mistaken pin.
        const { assert!(DEFAULT_MODEL.size_bytes > 100 * 1024 * 1024) };
        const { assert!(DEFAULT_MODEL.size_bytes < 2 * 1024 * 1024 * 1024) };
    }

    #[test]
    fn ensure_in_offline_mode_fails_loudly_when_uncached() {
        // See `resolve_cache_root_honors_ktstr_cache_dir` for the
        // ENV_LOCK rationale.
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_offline = std::env::var(OFFLINE_ENV).ok();
        let prev_cache = std::env::var("KTSTR_CACHE_DIR").ok();
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: ENV_LOCK guards process-wide env mutations; the
        // save/restore pair preserves other tests' state.
        unsafe {
            std::env::set_var(OFFLINE_ENV, "1");
            std::env::set_var("KTSTR_CACHE_DIR", tmp.path());
        }
        let fake = ModelSpec {
            file_name: "does-not-exist.gguf",
            url: "https://placeholder.example/none.gguf",
            sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
            size_bytes: 1,
        };
        let err = ensure(&fake).unwrap_err();
        assert!(format!("{err:#}").contains(OFFLINE_ENV), "err: {err:#}");
        unsafe {
            match prev_offline {
                Some(v) => std::env::set_var(OFFLINE_ENV, v),
                None => std::env::remove_var(OFFLINE_ENV),
            }
            match prev_cache {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
        }
    }

    /// #98: status() on a file that exists but whose SHA does not
    /// match must report `cached = true, sha_matches = false`. That
    /// is the branch ensure() consults to decide between "reuse
    /// cached copy" and "re-download"; a regression that lost the
    /// mismatch would silently re-validate any garbage bytes sitting
    /// at the expected path.
    #[test]
    fn status_reports_cached_but_sha_mismatch_for_garbage_bytes() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_cache = std::env::var("KTSTR_CACHE_DIR").ok();
        let tmp = tempfile::tempdir().unwrap();
        let spec = ModelSpec {
            file_name: "bogus.gguf",
            url: "https://placeholder.example/bogus.gguf",
            // Anything but the SHA of whatever bytes we write.
            sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
            size_bytes: 16,
        };
        let on_disk = tmp.path().join(spec.file_name);
        std::fs::write(&on_disk, b"definitely-not-zero-sha").unwrap();
        // SAFETY: ENV_LOCK serializes, save/restore preserves state.
        unsafe { std::env::set_var("KTSTR_CACHE_DIR", tmp.path()) };
        let st = status(&spec).unwrap();
        assert!(st.cached, "file exists, status must report cached=true");
        assert!(
            !st.sha_matches,
            "SHA is a fixed zero pin — garbage bytes must not match",
        );
        assert_eq!(st.path, on_disk);
        unsafe {
            match prev_cache {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
        }
    }

    /// #100: With `KTSTR_CACHE_DIR` unset, `resolve_cache_root` falls
    /// through to `XDG_CACHE_HOME` and appends `ktstr/models`.
    #[test]
    fn resolve_cache_root_honors_xdg_cache_home() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_ktstr = std::env::var("KTSTR_CACHE_DIR").ok();
        let prev_xdg = std::env::var("XDG_CACHE_HOME").ok();
        // SAFETY: ENV_LOCK serializes, save/restore preserves state.
        unsafe {
            std::env::remove_var("KTSTR_CACHE_DIR");
            std::env::set_var("XDG_CACHE_HOME", "/xdg/caches");
        }
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            PathBuf::from("/xdg/caches").join("ktstr").join("models"),
        );
        unsafe {
            match prev_ktstr {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
            match prev_xdg {
                Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
                None => std::env::remove_var("XDG_CACHE_HOME"),
            }
        }
    }

    /// #100: With both `KTSTR_CACHE_DIR` and `XDG_CACHE_HOME` unset,
    /// `resolve_cache_root` falls through to `$HOME/.cache/ktstr/models`.
    /// The third-tier fallback must hold so `~/.cache` remains the
    /// documented default on a fresh system.
    #[test]
    fn resolve_cache_root_falls_back_to_home_cache() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_ktstr = std::env::var("KTSTR_CACHE_DIR").ok();
        let prev_xdg = std::env::var("XDG_CACHE_HOME").ok();
        let prev_home = std::env::var("HOME").ok();
        // SAFETY: ENV_LOCK serializes, save/restore preserves state.
        unsafe {
            std::env::remove_var("KTSTR_CACHE_DIR");
            std::env::remove_var("XDG_CACHE_HOME");
            std::env::set_var("HOME", "/home/fake");
        }
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            PathBuf::from("/home/fake")
                .join(".cache")
                .join("ktstr")
                .join("models"),
        );
        unsafe {
            match prev_ktstr {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
            match prev_xdg {
                Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
                None => std::env::remove_var("XDG_CACHE_HOME"),
            }
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    /// #100: Empty `KTSTR_CACHE_DIR` must fall through to XDG
    /// exactly like "unset", mirroring the `!dir.is_empty()` gate in
    /// `resolve_cache_root`. A regression that treated the empty
    /// string as a valid root would produce an empty `PathBuf` and
    /// silently write cache entries into the current working dir.
    #[test]
    fn resolve_cache_root_treats_empty_ktstr_cache_dir_as_unset() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_ktstr = std::env::var("KTSTR_CACHE_DIR").ok();
        let prev_xdg = std::env::var("XDG_CACHE_HOME").ok();
        // SAFETY: ENV_LOCK serializes, save/restore preserves state.
        unsafe {
            std::env::set_var("KTSTR_CACHE_DIR", "");
            std::env::set_var("XDG_CACHE_HOME", "/xdg/caches");
        }
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            PathBuf::from("/xdg/caches").join("ktstr").join("models"),
            "empty KTSTR_CACHE_DIR must be treated as unset so XDG wins",
        );
        unsafe {
            match prev_ktstr {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
            match prev_xdg {
                Some(v) => std::env::set_var("XDG_CACHE_HOME", v),
                None => std::env::remove_var("XDG_CACHE_HOME"),
            }
        }
    }

    /// #117: `sanitize_env_value` replaces control characters (newline,
    /// tab, backspace, escape) with `?` and passes printable ASCII +
    /// Unicode through unchanged. Pins the predicate used before
    /// echoing a user-controlled env value into error output — a
    /// regression that let `\x1b` flow through could escape-sequence
    /// the terminal of whoever reads the error message.
    #[test]
    fn sanitize_env_value_replaces_control_chars() {
        // Printable ASCII passes through untouched.
        assert_eq!(sanitize_env_value("1"), "1");
        assert_eq!(sanitize_env_value("true"), "true");
        assert_eq!(sanitize_env_value("/path/to/thing"), "/path/to/thing");
        // Every standard control-character class is masked.
        assert_eq!(sanitize_env_value("a\nb"), "a?b");
        assert_eq!(sanitize_env_value("a\tb"), "a?b");
        assert_eq!(sanitize_env_value("a\x1bb"), "a?b");
        assert_eq!(sanitize_env_value("\x08"), "?");
        assert_eq!(sanitize_env_value("\r\n"), "??");
    }

    /// #117: An overlong value is truncated to a byte-bounded prefix
    /// with a `...` marker. The marker (three ASCII dots) makes it
    /// obvious the value was cut, and the truncation walks a char
    /// boundary so a multi-byte UTF-8 codepoint straddling the limit
    /// isn't split mid-sequence.
    #[test]
    fn sanitize_env_value_truncates_overlong_value() {
        let raw: String = "x".repeat(200);
        let out = sanitize_env_value(&raw);
        assert!(out.ends_with("..."), "truncation marker missing: {out:?}");
        // 64-byte cap + 3-byte marker = 67. Any longer means the
        // truncation didn't fire; any shorter means the marker path
        // ran on input that shouldn't have tripped it.
        assert_eq!(out.len(), 67);
    }

    /// #117: ensure()'s offline-bail error echoes the env value
    /// through `sanitize_env_value`. Set `OFFLINE_ENV` to a value
    /// containing both control chars and overlong content, and
    /// verify the error string contains neither a raw newline nor
    /// the full 200-char payload.
    #[test]
    fn ensure_offline_error_sanitizes_env_value_in_message() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_offline = std::env::var(OFFLINE_ENV).ok();
        let prev_cache = std::env::var("KTSTR_CACHE_DIR").ok();
        let tmp = tempfile::tempdir().unwrap();
        // Embed a newline + a very long tail; both get rewritten.
        let hostile = format!("inject\nbreak{}", "z".repeat(200));
        // SAFETY: ENV_LOCK serializes, save/restore preserves state.
        unsafe {
            std::env::set_var(OFFLINE_ENV, &hostile);
            std::env::set_var("KTSTR_CACHE_DIR", tmp.path());
        }
        let fake = ModelSpec {
            file_name: "not-here.gguf",
            url: "https://placeholder.example/not-here.gguf",
            sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
            size_bytes: 1,
        };
        let msg = format!("{:#}", ensure(&fake).unwrap_err());
        assert!(!msg.contains('\n'), "raw newline leaked: {msg:?}");
        assert!(
            !msg.contains(&"z".repeat(200)),
            "overlong tail leaked un-truncated: {msg:?}"
        );
        assert!(
            msg.contains("inject?break"),
            "sanitized stem missing: {msg:?}"
        );
        unsafe {
            match prev_offline {
                Some(v) => std::env::set_var(OFFLINE_ENV, v),
                None => std::env::remove_var(OFFLINE_ENV),
            }
            match prev_cache {
                Some(v) => std::env::set_var("KTSTR_CACHE_DIR", v),
                None => std::env::remove_var("KTSTR_CACHE_DIR"),
            }
        }
    }

    // -- LlmExtract pipeline --

    /// The default prompt is constant and load-bearing: a silent
    /// drift would re-baseline every cached retry-behavior
    /// expectation. Anchor on a prefix + tail so whitespace cleanup
    /// still catches surprise edits.
    #[test]
    fn llm_extract_prompt_template_is_stable() {
        assert!(LLM_EXTRACT_PROMPT_TEMPLATE.starts_with("You are a benchmark-output parser."));
        assert!(LLM_EXTRACT_PROMPT_TEMPLATE.contains("emit ONLY a single JSON object"));
        assert!(LLM_EXTRACT_PROMPT_TEMPLATE.contains("If no numeric metrics are present"));
    }

    #[test]
    fn compose_prompt_without_hint_omits_focus_header() {
        let p = compose_prompt("benchmark stdout", None);
        assert!(p.contains(LLM_EXTRACT_PROMPT_TEMPLATE));
        assert!(p.ends_with("STDOUT:\nbenchmark stdout"));
        assert!(
            !p.contains("Focus:"),
            "absent hint must not leave a dangling Focus header: {p}"
        );
    }

    #[test]
    fn compose_prompt_with_hint_inserts_focus_line() {
        let p = compose_prompt("stdout body", Some("throughput only"));
        assert!(p.contains("Focus: throughput only\n\n"));
        // Hint comes before STDOUT block so the model sees the focus
        // before the raw content.
        let focus_idx = p.find("Focus:").expect("Focus header present");
        let stdout_idx = p.find("STDOUT:").expect("STDOUT header present");
        assert!(focus_idx < stdout_idx);
    }

    #[test]
    fn compose_prompt_trims_hint_whitespace() {
        let p = compose_prompt("x", Some("  trim me \n "));
        assert!(p.contains("Focus: trim me\n\n"));
    }

    #[test]
    fn compose_prompt_empty_hint_degrades_to_no_focus() {
        // Whitespace-only hint is effectively absent; don't emit a
        // dangling "Focus: " header the model would treat as noise.
        let p = compose_prompt("x", Some("   "));
        assert!(
            !p.contains("Focus:"),
            "whitespace-only hint should not emit Focus header: {p}"
        );
    }

    #[test]
    fn inference_error_not_wired_message_mentions_offline_escape() {
        let e = InferenceError::NotWired;
        let msg = format!("{e}");
        assert!(msg.contains("KTSTR_MODEL_OFFLINE"));
        assert!(msg.contains("OutputFormat::Json"));
    }

    #[test]
    fn inference_error_display_includes_source_chain() {
        // ModelLoad wraps an anyhow::Error; {:#} should traverse the
        // source chain so a test author can see *why* the load failed.
        let inner = anyhow::anyhow!("root cause: cache corrupt");
        let e = InferenceError::ModelLoad(inner);
        let msg = format!("{e:#}");
        assert!(msg.contains("root cause: cache corrupt"), "msg: {msg}");
    }

    /// With the stub backend, `invoke_inference` must NEVER return
    /// `Ok`. If the stub is ever lifted accidentally (e.g. someone
    /// stubs a successful response for testing without wiring the
    /// retry-on-parse-failure path), this test fires first so the
    /// pipeline gets the real treatment rather than a half-stub.
    #[test]
    fn invoke_inference_stub_always_errs() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_offline = std::env::var(OFFLINE_ENV).ok();
        // Forcing offline ensures `ensure` bails on the uncached
        // placeholder model rather than attempting a network fetch;
        // the downstream invoke_inference should map that to a
        // `ModelLoad` error, not `NotWired`.
        // SAFETY: ENV_LOCK serializes process-wide env mutations.
        unsafe { std::env::set_var(OFFLINE_ENV, "1") };
        let r = invoke_inference("ignored");
        assert!(r.is_err());
        if let Err(InferenceError::ModelLoad(e)) = &r {
            assert!(
                format!("{e:#}").contains(OFFLINE_ENV),
                "expected offline gate error, got: {e:#}"
            );
        } else {
            panic!("expected ModelLoad(...) under offline gate, got {r:?}");
        }
        unsafe {
            match prev_offline {
                Some(v) => std::env::set_var(OFFLINE_ENV, v),
                None => std::env::remove_var(OFFLINE_ENV),
            }
        }
    }

    /// End-to-end stub behavior: the LlmExtract pipeline must return
    /// an empty metric set when the backend is unwired/unavailable,
    /// and must not panic on any stdout shape. This covers the
    /// contract metrics.rs relies on for its
    /// `llm_extract_returns_empty_when_backend_unwired` test.
    #[test]
    fn extract_via_llm_returns_empty_under_stub_backend() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let prev_offline = std::env::var(OFFLINE_ENV).ok();
        // SAFETY: ENV_LOCK serializes process-wide env mutations.
        unsafe { std::env::set_var(OFFLINE_ENV, "1") };
        let metrics = extract_via_llm("arbitrary stdout", None);
        assert!(metrics.is_empty());
        let metrics = extract_via_llm("stdout with hint", Some("focus"));
        assert!(metrics.is_empty());
        unsafe {
            match prev_offline {
                Some(v) => std::env::set_var(OFFLINE_ENV, v),
                None => std::env::remove_var(OFFLINE_ENV),
            }
        }
    }
}
