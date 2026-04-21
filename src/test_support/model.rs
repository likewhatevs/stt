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
//! 2. `load_inference` (module-private) routes both artifact paths
//!    through [`ensure`] — SHA-verifying the cached GGUF model and
//!    tokenizer or surfacing the offline-gate/missing-cache error —
//!    then opens the GGUF, builds the Qwen3 `ModelWeights`, loads
//!    the tokenizer, and resolves the `<|im_end|>` EOS token id.
//!    This is the failure point for `KTSTR_MODEL_OFFLINE=1` with an
//!    uncached artifact and for a placeholder/malformed SHA pin.
//!    Result is memoized in the process-wide `MODEL_CACHE` static
//!    on success so subsequent calls skip the 2.44 GiB load.
//! 3. `invoke_with_model` (module-private) runs one greedy
//!    generation pass against the loaded state: clears the KV cache
//!    up front (idempotence guarantee), feeds the ChatML-wrapped
//!    `/no_think`-directed prompt through the forward loop under
//!    `Sampling::ArgMax` with a fixed seed (deterministic output
//!    per prompt+weights), then passes the decoded text through
//!    `strip_think_block` (module-private) to remove any leaked
//!    `<think>…</think>` region before returning.
//! 4. On `Ok`, [`super::metrics::find_and_parse_json`] extracts the
//!    JSON region; parsed values flow through
//!    [`super::metrics::walk_json_leaves`] pre-tagged with
//!    [`MetricSource::LlmExtract`](crate::test_support::MetricSource::LlmExtract).
//! 5. A JSON-parse failure or infra error (model missing, forward-
//!    pass error) returns an empty metric set. No retry: under
//!    deterministic ArgMax sampling a second call on the same
//!    prompt produces byte-identical output.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use super::KTSTR_TESTS;
use super::payload::OutputFormat;

static MODEL_CACHE: OnceLock<Mutex<LoadedInference>> = OnceLock::new();

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
/// Qwen3-4B Q4_K_M GGUF (~2.44 GiB — 2500 MiB per [`ModelSpec::size_bytes`]).
/// The 4B-parameter tier gives usable structured-JSON extraction
/// quality (better than 1-2B tiers for the "emit ONLY a JSON object"
/// constraint the `LLM_EXTRACT_PROMPT_TEMPLATE` enforces) at an
/// artifact size small enough that host-side post-test extraction
/// loads and runs in reasonable wall time on CPU. Larger 7B-14B tiers
/// would multiply both disk footprint and inference latency without
/// a commensurate gain on a narrow "parse benchmark stdout" task.
///
/// URL points at the official `Qwen/Qwen3-4B-GGUF` repo on Hugging Face.
pub const DEFAULT_MODEL: ModelSpec = ModelSpec {
    file_name: "Qwen3-4B-Q4_K_M.gguf",
    url: "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
    sha256_hex: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
    size_bytes: 2500 * 1024 * 1024,
};

/// Tokenizer artifact paired with [`DEFAULT_MODEL`]. The GGUF file
/// carries model weights but not the byte-level BPE (BBPE) merge
/// table used for encode/decode, so a separate `tokenizer.json`
/// sits alongside the model in the cache. Both entries are
/// prefetched together so LlmExtract inference never trips on a
/// half-ready cache.
///
/// URL sources the tokenizer from `Qwen/Qwen3-4B` (the non-GGUF
/// upstream repo) because the GGUF repo `Qwen/Qwen3-4B-GGUF` that
/// hosts [`DEFAULT_MODEL`] only ships the quantized weight file —
/// its `tokenizer.json` is not published there. Pulling each
/// artifact from its authoritative repo keeps the two pins in sync
/// with the same Qwen3-4B release.
pub const DEFAULT_TOKENIZER: ModelSpec = ModelSpec {
    file_name: "Qwen3-4B-tokenizer.json",
    url: "https://huggingface.co/Qwen/Qwen3-4B/resolve/main/tokenizer.json",
    sha256_hex: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    size_bytes: 11 * 1024 * 1024,
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
    // SHA-pin shape check runs before the offline-gate check. A
    // malformed or placeholder pin is a programmer error in the
    // `ModelSpec` itself — it does not depend on runtime state. Pre-
    // seeding a cache under the offline gate cannot rescue a broken
    // pin, so surfacing the shape failure first gives the clearer
    // diagnostic ("fix the ModelSpec") instead of the downstream
    // "KTSTR_MODEL_OFFLINE set but not cached" red herring. `status()`
    // swallowed a malformed pin via `unwrap_or(false)` on
    // `verify_sha256`, so a placeholder (all-`?`) `sha256_hex` would
    // silently skip the fast path and drop through to `fetch` —
    // wasting a 2.44 GiB download before the post-download
    // `verify_sha256` bails. Catch it here so the error surfaces
    // before either the offline bail or the network request.
    if !is_valid_sha256_hex(spec.sha256_hex) {
        anyhow::bail!(
            "model '{}' has a placeholder or malformed SHA-256 pin \
             ({:?}); refusing to download {} until a real digest is \
             recorded. Replace the pin in the ModelSpec before re-running.",
            spec.file_name,
            spec.sha256_hex,
            spec.url,
        );
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
    // hang forever on a slow or unreachable mirror. 30s connect
    // catches DNS/TLS wedges early. Tests that don't actually hit
    // the network (offline gate, cached path) never enter this
    // branch.
    //
    // 15-minute overall timeout sizing: the pinned Qwen3-4B GGUF is
    // ~2.44 GiB (see DEFAULT_MODEL.size_bytes). 900s tolerates ~2.7
    // MiB/s sustained throughput, a margin the previous 300s cap
    // (which required ~8.5 MiB/s) did not hold on CDN-throttled or
    // bandwidth-limited CI runners. The fetch is idempotent —
    // subsequent test runs hit the cached-and-verified fast path,
    // so the slow-path cost is paid once per cache warmup.
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(900))
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
    // from `NamedTempFile` implements `Write`. A buffer-then-write
    // approach would hold the full body in memory.
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

/// Canonical predicate for a well-formed SHA-256 hex pin: exactly
/// 64 ASCII characters, each a hex digit (`0-9a-fA-F`). Shared by
/// [`ensure`] (pre-fetch shape check on [`ModelSpec::sha256_hex`])
/// and [`verify_sha256`] (post-read validation of the expected pin);
/// centralizing the rule prevents drift between the two call sites.
/// Callers that need to distinguish "wrong length" from "non-hex"
/// for diagnostics still need their own branching — this helper
/// collapses the predicate, not the error messages.
fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
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

    if !is_valid_sha256_hex(expected_hex) {
        // The helper unified the predicate; pick the specific
        // diagnostic wording from the same two branches the tests
        // pin (`verify_sha256_rejects_malformed_hex_length` expects
        // "64 chars", `verify_sha256_rejects_non_hex_chars` expects
        // "non-hex"). Length is checked first so a non-64 string of
        // all hex digits surfaces as a length error.
        if expected_hex.len() != 64 {
            anyhow::bail!(
                "expected SHA-256 hex must be 64 chars, got {} ({:?})",
                expected_hex.len(),
                expected_hex,
            );
        }
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
    let got = hex::encode(hasher.finalize());
    Ok(got.eq_ignore_ascii_case(expected_hex))
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

/// Prefetch [`DEFAULT_MODEL`] and [`DEFAULT_TOKENIZER`] when at least
/// one registered test needs the LlmExtract backend. No-op when
/// `KTSTR_MODEL_OFFLINE` is set (skips fetch, leaves per-test failures
/// to surface downstream) or when no test declares
/// [`OutputFormat::LlmExtract`].
///
/// Both artifacts are ensured together because inference needs both —
/// deferring the tokenizer to first-call would only move a failure
/// that setup already has the authority to surface.
///
/// Returns `Ok(None)` when no fetch was attempted; `Ok(Some(path))`
/// when both artifacts are now cached (the model path is returned so
/// callers can log the inference entry point); `Err` on fetch/verify
/// failure.
pub fn prefetch_if_required() -> Result<Option<PathBuf>> {
    if !any_test_requires_model() {
        return Ok(None);
    }
    if let Some(v) = read_offline_env() {
        let v_safe = sanitize_env_value(&v);
        tracing::warn!(
            env_var = OFFLINE_ENV,
            value = %v_safe,
            "offline gate set; skipping eager model prefetch",
        );
        return Ok(None);
    }
    let model_path = ensure(&DEFAULT_MODEL)?;
    ensure(&DEFAULT_TOKENIZER)?;
    Ok(Some(model_path))
}

// ---------------------------------------------------------------------------
// LlmExtract runtime
// ---------------------------------------------------------------------------

/// Default prompt template prepended to every
/// [`OutputFormat::LlmExtract`] invocation. Kept here as a const so
/// tests can assert its exact contents — a silent wording drift
/// would re-baseline every downstream behavior expectation.
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

/// Upper bound on generated tokens for a single LlmExtract
/// invocation. The prompt template instructs the model to emit a
/// single JSON object; 512 tokens is enough for a dense metric bag
/// (hundreds of keys) without letting a runaway generation burn
/// wall time.
///
/// 512 is sufficient even with `<think>…</think>` leakage: the
/// `/no_think` directive (see [`invoke_with_model`]) suppresses
/// Qwen3's reasoning trace to at most an empty `<think></think>`
/// shell (~8 tokens), which the post-decode [`strip_think_block`]
/// removes before the JSON walker sees the response. The cap-hit
/// warning in the generation loop fires if the shell grows or a
/// full trace leaks despite `/no_think`, surfacing the regression
/// rather than silently truncating.
const SAMPLE_LEN: usize = 512;

/// Deterministic seed. ArgMax sampling ignores the RNG but
/// `LogitsProcessor::from_sampling` still consumes a seed; pinning
/// it keeps the constructor call a pure function of the pinned
/// model weights.
const SEED: u64 = 299_792_458;

/// Loaded inference state: the Qwen3 weights, its tokenizer, and the
/// resolved EOS token id. Threaded through `load_inference` and
/// `invoke_with_model` — both module-private. Nothing outside
/// `model.rs` constructs or observes this type.
struct LoadedInference {
    model: candle_transformers::models::quantized_qwen3::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    eos_id: u32,
    device: candle::Device,
}

/// Ensure both artifacts are cached, open the GGUF, build the Qwen3
/// `ModelWeights`, and load the tokenizer. Returns the bundled state.
///
/// Every failure point flows through `anyhow::Error` so
/// `extract_via_llm` surfaces the full chain (path, cause) when
/// logging.
fn load_inference() -> anyhow::Result<LoadedInference> {
    use candle::{Device, quantized::gguf_file};
    use candle_transformers::models::quantized_qwen3::ModelWeights;
    use tokenizers::Tokenizer;

    let model_path = ensure(&DEFAULT_MODEL)?;
    let tokenizer_path = ensure(&DEFAULT_TOKENIZER)?;

    let device = Device::Cpu;

    let mut file = std::fs::File::open(&model_path).map_err(|e| {
        anyhow::Error::new(e).context(format!("open GGUF model at {}", model_path.display()))
    })?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::Error::msg(e.with_path(&model_path)))?;
    let model = ModelWeights::from_gguf(content, &mut file, &device).map_err(anyhow::Error::msg)?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        anyhow::Error::msg(e).context(format!("load tokenizer at {}", tokenizer_path.display()))
    })?;

    let eos_id = *tokenizer
        .get_vocab(true)
        .get("<|im_end|>")
        .ok_or_else(|| anyhow::anyhow!("tokenizer vocab missing '<|im_end|>' EOS token"))?;

    Ok(LoadedInference {
        model,
        tokenizer,
        eos_id,
        device,
    })
}

/// Run one greedy generation pass against the already-loaded model
/// and return the decoded assistant text with any `<think>…</think>`
/// block stripped.
///
/// Idempotent: repeated calls with the same `LoadedInference` are
/// safe; each invocation starts with a clean KV cache. The cache is
/// cleared up front on every call so repeated invocations don't
/// carry state across the prompt forward pass's position offsets —
/// the callee owns that invariant rather than pushing it onto
/// callers.
///
/// Greedy: `Sampling::ArgMax` with a fixed seed. Output is a
/// deterministic function of the prompt + weights.
fn invoke_with_model(state: &mut LoadedInference, prompt: &str) -> anyhow::Result<String> {
    use candle::Tensor;
    use candle_transformers::generation::{LogitsProcessor, Sampling};

    // A prior invocation may have populated per-layer K/V tensors
    // addressed by absolute positions `[0, prompt_len + generated)`
    // of the previous prompt. This call re-starts at `index_pos=0`,
    // so the stale entries would alias the new prompt's slots and
    // poison the forward pass. Clearing unconditionally costs a
    // layer-scoped vec walk per call (see
    // `candle_transformers::models::quantized_qwen3::ModelWeights::clear_kv_cache`)
    // and is the lone safety gate keeping `invoke_with_model`
    // idempotent across repeated calls on the same state.
    state.model.clear_kv_cache();

    // Qwen3 ChatML prompt. The `/no_think` directive at the end of
    // the user turn switches the model out of thinking mode per the
    // Qwen3 model card: the assistant skips the `<think>…</think>`
    // block and emits the final answer directly, keeping the SAMPLE_LEN
    // token budget available for the JSON response rather than burning
    // it on a reasoning trace the downstream walker would discard.
    // The post-decode `strip_think_block` below remains as a belt-
    // and-suspenders defense because the `/no_think` directive is a
    // soft switch and the model can still emit an empty
    // `<think></think>` shell.
    let chat_prompt =
        format!("<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n");
    let encoding = state
        .tokenizer
        .encode(chat_prompt, true)
        .map_err(anyhow::Error::msg)?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    let mut logits_processor = LogitsProcessor::from_sampling(SEED, Sampling::ArgMax);

    // Prompt pass: feed the whole prompt at index_pos=0. qwen3's
    // forward already narrows to the last position and returns shape
    // `(b, vocab)`; the caller's `squeeze(0)` strips the batch dim.
    let input = Tensor::new(prompt_tokens.as_slice(), &state.device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(anyhow::Error::msg)?;
    let logits = state
        .model
        .forward(&input, 0)
        .and_then(|l| l.squeeze(0))
        .map_err(anyhow::Error::msg)?;
    let mut next_token = logits_processor
        .sample(&logits)
        .map_err(anyhow::Error::msg)?;

    let mut generated: Vec<u32> = Vec::with_capacity(SAMPLE_LEN);
    if next_token != state.eos_id {
        generated.push(next_token);
    }

    // Generation loop. index_pos advances to the absolute position
    // of the token being processed — the KV cache uses it to place
    // each new token's Q/K/V in the right slot. On step 0 the position
    // is `prompt_tokens.len()`, i.e. the slot immediately after the
    // prompt pass's last token.
    for step in 0..SAMPLE_LEN.saturating_sub(1) {
        if next_token == state.eos_id {
            break;
        }
        let input = Tensor::new(&[next_token], &state.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(anyhow::Error::msg)?;
        let logits = state
            .model
            .forward(&input, prompt_tokens.len() + step)
            .and_then(|l| l.squeeze(0))
            .map_err(anyhow::Error::msg)?;
        next_token = logits_processor
            .sample(&logits)
            .map_err(anyhow::Error::msg)?;
        if next_token == state.eos_id {
            break;
        }
        generated.push(next_token);
    }

    if next_token != state.eos_id {
        tracing::warn!(
            "generation hit {} token cap without EOS — output may be truncated",
            SAMPLE_LEN,
        );
    }

    let decoded = state
        .tokenizer
        .decode(&generated, true)
        .map_err(anyhow::Error::msg)?;
    Ok(strip_think_block(&decoded).to_string())
}

/// Strip one or more `<think>…</think>` blocks from the model's raw
/// output. Qwen3 emits a thinking trace by default; `/no_think` in
/// the user prompt suppresses it, but an empty `<think></think>`
/// shell can still appear and a stray trace under other prompts is
/// also possible. The downstream JSON walker doesn't care about
/// prose surrounding the JSON region, but a half-balanced think
/// block (missing closing tag from a truncated generation) is left
/// as-is to avoid hiding corruption.
///
/// Tags are matched by depth: each outer `<think>` opens a block and
/// is closed by the `</think>` whose running depth hits zero. This
/// handles both nested blocks (`<think><think>x</think></think>` →
/// both tags belong to one outer block, fully removed) and sibling
/// blocks (`<think>a</think>mid<think>b</think>end` → each block
/// closes independently, yielding `"midend"`). `find`-first would
/// bleed an orphan `</think>` through for nested input; `rfind`-last
/// would merge siblings into a single phantom block.
fn strip_think_block(s: &str) -> std::borrow::Cow<'_, str> {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if !s.contains(OPEN) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    'outer: while let Some(open_idx) = rest.find(OPEN) {
        out.push_str(&rest[..open_idx]);
        // Depth scanner: start with depth 1 (the opening tag we just
        // found) and consume tags until it returns to 0 or we run
        // out of input. `cursor` indexes into `rest` — its absolute
        // position within `rest` moves forward monotonically as each
        // tag is consumed. Using byte positions in a &str is safe
        // because OPEN and CLOSE are both ASCII, so find() only
        // returns byte offsets that fall on char boundaries.
        let mut cursor = open_idx + OPEN.len();
        let mut depth: usize = 1;
        while depth > 0 {
            let tail = &rest[cursor..];
            let next_open = tail.find(OPEN);
            let next_close = tail.find(CLOSE);
            match (next_open, next_close) {
                (Some(o), Some(c)) if o < c => {
                    depth += 1;
                    cursor += o + OPEN.len();
                }
                (_, Some(c)) => {
                    depth -= 1;
                    cursor += c + CLOSE.len();
                    if depth == 0 {
                        rest = &rest[cursor..];
                        continue 'outer;
                    }
                }
                (Some(_), None) | (None, None) => {
                    // Unterminated outer block: no `</think>` remains
                    // to close depth. Emit everything from the
                    // opening `<think>` (`&rest[open_idx..]`) through
                    // end-of-input verbatim — including any inner
                    // `<think>` openers we already counted into
                    // `depth`. Preserving the full tail unstripped
                    // keeps a truncation bug visible rather than
                    // masking it behind a partial strip that would
                    // look like a complete response.
                    out.push_str(&rest[open_idx..]);
                    rest = "";
                    break 'outer;
                }
            }
        }
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
}

/// Run the full LlmExtract pipeline against `stdout` and return the
/// resulting metrics, all pre-tagged with
/// [`MetricSource::LlmExtract`](super::MetricSource::LlmExtract).
///
/// A single inference call is made. An infra error or a
/// JSON-parse failure of the model's response returns an empty
/// metric set — matching the [`extract_metrics`] contract that
/// extraction errors are non-fatal and the downstream
/// [`Check`](crate::test_support::Check) evaluation reports each
/// referenced metric as missing.
///
/// No retry: under `Sampling::ArgMax` with a fixed seed, a second
/// inference call on the same prompt + weights produces byte-
/// identical output. Retrying would only burn wall time without
/// changing the result.
pub(crate) fn extract_via_llm(stdout: &str, hint: Option<&str>) -> Vec<super::Metric> {
    let prompt = compose_prompt(stdout, hint);

    let cache = match MODEL_CACHE.get() {
        Some(c) => c,
        None => {
            let loaded = match load_inference() {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = ?e, "LlmExtract model load failed");
                    return Vec::new();
                }
            };
            if MODEL_CACHE.set(Mutex::new(loaded)).is_err() {
                tracing::debug!("MODEL_CACHE init race; discarding duplicate load");
            }
            MODEL_CACHE
                .get()
                .expect("MODEL_CACHE populated by this or a racing peer")
        }
    };
    let mut state = cache.lock().unwrap_or_else(|e| e.into_inner());

    let response = match invoke_with_model(&mut state, &prompt) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = ?e, "LlmExtract inference failed");
            return Vec::new();
        }
    };
    match super::metrics::find_and_parse_json(&response) {
        Some(json) => super::metrics::walk_json_leaves(&json, super::MetricSource::LlmExtract),
        None => {
            tracing::warn!(
                response_bytes = response.len(),
                "LlmExtract response was not parseable JSON; returning empty metric set",
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
        let _env =
            super::super::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", "/explicit/override");
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, PathBuf::from("/explicit/override"));
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

    /// Non-empty short file — SHA-256 of ASCII "abc" is a
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

    /// Multi-chunk file (larger than a single read buffer)
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
        let expected_hex = hex::encode(expected_bytes);
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
        // The pinned Qwen3-4B Q4_K_M GGUF is ~2.44 GiB. The upper
        // bound is tight at 3 GiB so a silent swap to a higher-bit
        // quantization (Q5/Q6/Q8) of the same 4B-parameter base —
        // which would balloon the artifact to 3-4+ GiB and multiply
        // inference latency — fails this check instead of slipping
        // through. A wildly different size signals someone swapped
        // the pin for a mistaken artifact.
        const { assert!(DEFAULT_MODEL.size_bytes > 100 * 1024 * 1024) };
        const { assert!(DEFAULT_MODEL.size_bytes < 3 * 1024 * 1024 * 1024) };
    }

    #[test]
    fn ensure_in_offline_mode_fails_loudly_when_uncached() {
        // See `resolve_cache_root_honors_ktstr_cache_dir` for the
        // ENV_LOCK rationale.
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let _env_offline = super::super::test_helpers::EnvVarGuard::set(OFFLINE_ENV, "1");
        let _env_cache = super::super::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path().to_str().expect("tempdir path is UTF-8"),
        );
        let fake = ModelSpec {
            file_name: "does-not-exist.gguf",
            url: "https://placeholder.example/none.gguf",
            sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
            size_bytes: 1,
        };
        let err = ensure(&fake).unwrap_err();
        assert!(format!("{err:#}").contains(OFFLINE_ENV), "err: {err:#}");
    }

    /// status() on a file that exists but whose SHA does not
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
        let _env_cache = super::super::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path().to_str().expect("tempdir path is UTF-8"),
        );
        let st = status(&spec).unwrap();
        assert!(st.cached, "file exists, status must report cached=true");
        assert!(
            !st.sha_matches,
            "SHA is a fixed zero pin — garbage bytes must not match",
        );
        assert_eq!(st.path, on_disk);
    }

    /// With `KTSTR_CACHE_DIR` unset, `resolve_cache_root` falls
    /// through to `XDG_CACHE_HOME` and appends `ktstr/models`.
    #[test]
    fn resolve_cache_root_honors_xdg_cache_home() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _env_ktstr = super::super::test_helpers::EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _env_xdg =
            super::super::test_helpers::EnvVarGuard::set("XDG_CACHE_HOME", "/xdg/caches");
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            PathBuf::from("/xdg/caches").join("ktstr").join("models"),
        );
    }

    /// With both `KTSTR_CACHE_DIR` and `XDG_CACHE_HOME` unset,
    /// `resolve_cache_root` falls through to `$HOME/.cache/ktstr/models`.
    /// The third-tier fallback must hold so `~/.cache` remains the
    /// documented default on a fresh system.
    #[test]
    fn resolve_cache_root_falls_back_to_home_cache() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _env_ktstr = super::super::test_helpers::EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _env_xdg = super::super::test_helpers::EnvVarGuard::remove("XDG_CACHE_HOME");
        let _env_home = super::super::test_helpers::EnvVarGuard::set("HOME", "/home/fake");
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            PathBuf::from("/home/fake")
                .join(".cache")
                .join("ktstr")
                .join("models"),
        );
    }

    /// Empty `KTSTR_CACHE_DIR` must fall through to XDG
    /// exactly like "unset", mirroring the `!dir.is_empty()` gate in
    /// `resolve_cache_root`. A regression that treated the empty
    /// string as a valid root would produce an empty `PathBuf` and
    /// silently write cache entries into the current working dir.
    #[test]
    fn resolve_cache_root_treats_empty_ktstr_cache_dir_as_unset() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _env_ktstr = super::super::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _env_xdg =
            super::super::test_helpers::EnvVarGuard::set("XDG_CACHE_HOME", "/xdg/caches");
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            PathBuf::from("/xdg/caches").join("ktstr").join("models"),
            "empty KTSTR_CACHE_DIR must be treated as unset so XDG wins",
        );
    }

    /// `sanitize_env_value` replaces control characters (newline,
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

    /// An overlong value is truncated to a byte-bounded prefix
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

    /// ensure()'s offline-bail error echoes the env value
    /// through `sanitize_env_value`. Set `OFFLINE_ENV` to a value
    /// containing both control chars and overlong content, and
    /// verify the error string contains neither a raw newline nor
    /// the full 200-char payload.
    #[test]
    fn ensure_offline_error_sanitizes_env_value_in_message() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        // Embed a newline + a very long tail; both get rewritten.
        let hostile = format!("inject\nbreak{}", "z".repeat(200));
        let _env_offline = super::super::test_helpers::EnvVarGuard::set(OFFLINE_ENV, &hostile);
        let _env_cache = super::super::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            tmp.path().to_str().expect("tempdir path is UTF-8"),
        );
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
    }

    // -- LlmExtract pipeline --

    /// The default prompt is constant and load-bearing: a silent
    /// drift would re-baseline every downstream behavior
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

    /// Under the offline gate with no cached artifacts,
    /// `load_inference` must surface an error whose message echoes
    /// the offline env var — that is the signal the caller needs to
    /// distinguish a user-requested skip from a pipeline bug. Pins
    /// the offline-gate trip point so a regression that swallowed
    /// the env var context would fire here first.
    #[test]
    fn load_inference_errs_with_offline_message_under_offline_gate() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Forcing offline ensures `ensure` bails on the uncached
        // placeholder model rather than attempting a network fetch.
        let _env_offline = super::super::test_helpers::EnvVarGuard::set(OFFLINE_ENV, "1");
        let r = load_inference();
        match r {
            Err(e) => {
                assert!(
                    format!("{e:#}").contains(OFFLINE_ENV),
                    "expected offline gate error, got: {e:#}"
                );
            }
            Ok(_) => panic!("expected Err under offline gate, got Ok"),
        }
    }

    /// End-to-end unavailable-backend behavior: the LlmExtract
    /// pipeline must return an empty metric set when inference
    /// cannot run (uncached artifacts under the offline gate), and
    /// must not panic on any stdout shape. The offline gate trips
    /// `ensure()` before any model load, so the inference call
    /// fails cleanly and the pipeline reports no metrics.
    #[test]
    fn extract_via_llm_returns_empty_when_backend_unavailable() {
        let _guard = super::super::test_helpers::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _env_offline = super::super::test_helpers::EnvVarGuard::set(OFFLINE_ENV, "1");
        let metrics = extract_via_llm("arbitrary stdout", None);
        assert!(metrics.is_empty());
        let metrics = extract_via_llm("stdout with hint", Some("focus"));
        assert!(metrics.is_empty());
    }

    // -- strip_think_block --

    #[test]
    fn strip_think_block_noop_on_absent_tag() {
        let s = "plain output with no think block";
        let out = strip_think_block(s);
        // Borrowed fast path: input without `<think>` must not
        // allocate a new String.
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
        assert_eq!(out, s);
    }

    #[test]
    fn strip_think_block_removes_complete_block() {
        let s = "pre <think>reasoning trace</think> post";
        assert_eq!(strip_think_block(s), "pre  post");
    }

    #[test]
    fn strip_think_block_removes_empty_shell() {
        // /no_think suppresses thinking but an empty shell can still
        // leak through. Must be stripped so `find_and_parse_json`
        // doesn't see the tags at all.
        let s = "<think></think>{\"latency_ms\": 42}";
        assert_eq!(strip_think_block(s), "{\"latency_ms\": 42}");
    }

    #[test]
    fn strip_think_block_removes_multiple_blocks() {
        let s = "<think>a</think>middle<think>b</think>end";
        assert_eq!(strip_think_block(s), "middleend");
    }

    #[test]
    fn strip_think_block_preserves_unterminated_open_tag() {
        // Unterminated trace (e.g. SAMPLE_LEN cut mid-think) is kept
        // verbatim so the truncation is visible downstream instead
        // of silently masked by a partial strip.
        let s = "before <think>unclosed trace and then garbage";
        assert_eq!(strip_think_block(s), s);
    }

    /// Nested `<think>` tags must match by depth: the outermost open
    /// pairs with the outermost close, and everything in between —
    /// including the inner `<think>inner</think>` — is stripped as
    /// part of the outer block. A depth-blind `find`-first
    /// implementation closes on the inner `</think>` and leaves the
    /// outer `</think>` as an orphan, which is the bug this case
    /// regression-guards.
    #[test]
    fn strip_think_block_handles_nested_tags() {
        let s = "<think><think>inner</think></think>{\"k\": 1}";
        assert_eq!(strip_think_block(s), "{\"k\": 1}");
    }

    /// Nested block embedded between plain text on both sides.
    /// Verifies that the depth scanner emits pre/post context
    /// unchanged while collapsing the full outer block (both inner
    /// and outer `</think>` pair consumed).
    #[test]
    fn strip_think_block_handles_nested_tags_with_surrounding_text() {
        let s = "pre <think>a<think>b</think>c</think> post";
        assert_eq!(strip_think_block(s), "pre  post");
    }

    /// Mixed: a nested block followed by an independent sibling
    /// block. The scanner must close the outer of the first nested
    /// pair (depth 1→2→1→0) on its own `</think>`, then restart for
    /// the sibling block — NOT merge the two into a single phantom
    /// block spanning the intervening text.
    #[test]
    fn strip_think_block_handles_nested_then_sibling() {
        let s = "<think><think>x</think></think>mid<think>y</think>end";
        assert_eq!(strip_think_block(s), "midend");
    }
}
