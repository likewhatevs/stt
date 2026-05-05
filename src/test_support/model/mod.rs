//! Local LLM model cache + LlmExtract runtime for
//! [`OutputFormat::LlmExtract`] payloads.
//!
//! `OutputFormat::LlmExtract` routes stdout through a small local
//! model that emits JSON, which the existing
//! [`walk_json_leaves`](crate::test_support::metrics) pipeline then
//! consumes. The model binary itself lives under
//! `~/.cache/ktstr/models/`. This module owns both the cache
//! surface (locate + fetch + check) and the LlmExtract pipeline
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
//! # Lazy model load
//!
//! There is no eager prefetch step. The model is loaded on first
//! [`extract_via_llm`] call by [`load_inference`]'s `ensure(&DEFAULT_MODEL)`
//! invocation, which fetches the GGUF on cache miss, SHA-checks the
//! cached file on hit, and respects `KTSTR_MODEL_OFFLINE=1` (offline
//! runs skip the fetch and surface a per-test load failure). The first
//! LlmExtract test in the process pays the cold-cache fetch + SHA-verify
//! cost (seconds on warm cache, minutes on cold cache with download);
//! subsequent tests see the memoized result.
//!
//! # LlmExtract extraction pipeline
//!
//! [`extract_via_llm`] is the runtime entry point called by
//! [`extract_metrics`](crate::test_support::extract_metrics) when a
//! payload's [`OutputFormat::LlmExtract`] fires:
//!
//! 1. [`compose_prompt`] assembles `{LLM_EXTRACT_PROMPT_TEMPLATE}\n\n{focus}STDOUT:\n{body}`.
//! 2. `load_inference` (module-private) routes the GGUF model
//!    artifact through [`ensure`] — SHA-checking the cached file or
//!    surfacing the offline-gate/missing-cache error — then loads
//!    the model via `llama_cpp_2::LlamaModel::load_from_file`
//!    against the process-wide `LlamaBackend`, with the GGUF
//!    carrying its own tokenizer + EOS metadata so no separate
//!    tokenizer artifact is involved — failures here surface for
//!    `KTSTR_MODEL_OFFLINE=1` with an uncached artifact, for a
//!    placeholder/malformed SHA pin, and for a corrupt GGUF, with
//!    the result memoized in the process-wide [`MODEL_CACHE`]
//!    `Mutex<Option<Arc<Result<Mutex<LoadedInference>, String>>>>`
//!    via [`memoized_inference`] (concurrent first-call races
//!    serialize on the outer `Mutex` so at most one load runs
//!    end-to-end, and a failed load is cached as `Err` so
//!    subsequent calls fail-closed without repeating the 2.55 GiB
//!    load; the inner `Mutex` then serializes repeated generation
//!    passes against the shared `LlamaModel`); tests that mutate
//!    `KTSTR_MODEL_OFFLINE` or `KTSTR_CACHE_DIR` call [`reset`]
//!    (cfg(test)-only) before asserting offline-gate trip behavior
//!    so a previously-memoized `Ok(_)` does not bypass the gate.
//! 3. `invoke_with_model` (module-private) builds a fresh
//!    `LlamaContext` per call — fresh-context-per-call sidesteps the
//!    self-referential lifetime issue that storing the context on
//!    `LoadedInference` would create — feeds the ChatML-wrapped
//!    `/no_think`-directed prompt as a single batched decode, then
//!    samples token-by-token via `LlamaSampler::greedy()` (greedy
//!    ArgMax — output is a deterministic function of prompt + weights
//!    without a separate seed). EOS detection uses
//!    `LlamaModel::is_eog_token`. The decoded text passes through
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
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use llama_cpp_2::llama_backend::LlamaBackend;

/// Process-wide [`LlamaBackend`] handle. The llama.cpp C library uses
/// a single global init/teardown pair (`llama_backend_init` /
/// `llama_backend_free`) and the [`LlamaBackend::init`] wrapper
/// enforces "exactly one live instance per process" — calling
/// `init()` a second time while the first is still alive returns
/// `LlamaCppError::BackendAlreadyInitialized`. A `OnceLock` matches
/// that contract: every caller observes the same `&'static
/// LlamaBackend`, and the lazy init lives until the process exits
/// (we never drop it; doing so would void [`LlamaModel`]s loaded
/// against it).
///
/// `init()` returns `Err` only on `BackendAlreadyInitialized` per
/// llama-cpp-2's documented contract, which the `OnceLock` makes
/// unreachable. A failure here is a programmer/environment bug —
/// panic with the rendered reason rather than threading a fallible
/// return through every caller.
///
/// Log routing: `send_logs_to_tracing(LogOptions::default())` is
/// called inside the OnceLock initializer, BEFORE any
/// `LlamaModel::load_from_file` call hits the C side. The default
/// `LogOptions` has logs ENABLED, routing llama.cpp's internal log
/// stream (model-load progress, GGML init chatter, KV-cache
/// reservation notes, error reasons) into the tracing subscriber.
/// This is the ONLY surface that exposes the upstream reason behind
/// an [`InferenceError::ModelLoad`] /
/// `LlamaModelLoadError::NullResult` failure — the C side writes
/// its actual rejection reason (mmap failure, vocab parse error,
/// version mismatch, etc.) into that log stream, and without it the
/// wrapper just surfaces "null result from llama cpp" with no
/// detail.
///
/// The upstream wrapper tracks log-state via a `OnceLock`-backed
/// singleton itself ("TODO: Reinitialize the state to support
/// calling send_logs_to_tracing multiple times" in upstream
/// `lib.rs`), so we get exactly one call per process. Operators
/// who want to suppress llama.cpp's log noise on a one-off basis
/// can install a tracing-subscriber filter that drops
/// `target = "llama-cpp-2"` events (the upstream metadata name
/// at `llama-cpp-2/src/log.rs:18`, with hyphens — not the Rust
/// path `llama_cpp_2`); suppression is no longer the default
/// because the diagnostic value of the upstream stream
/// outweighs the test-output noise.
static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn global_backend() -> &'static LlamaBackend {
    BACKEND.get_or_init(|| {
        // Install a minimal `tracing-subscriber` BEFORE
        // `send_logs_to_tracing` — without a subscriber, llama.cpp's
        // log events route into tracing but get silently dropped, so
        // load failures surface as bare "null result from llama cpp"
        // with no upstream detail. `try_init` is a no-op when a
        // subscriber is already installed (e.g. a sibling test using
        // `tracing-test` to capture events into a per-test buffer);
        // the `.ok()` discard mirrors the upstream pattern of
        // best-effort install since the existing subscriber's events
        // already cover whatever sink that test wanted.
        //
        // Routes to stderr by default; CI captures and redirected
        // stderr both pick up the events automatically. Operators who
        // want to suppress llama.cpp's log noise on a one-off basis
        // can install their own subscriber FIRST (this function's
        // try_init becomes a no-op) with whatever EnvFilter / target
        // filter they want.
        let _ = tracing_subscriber::fmt::try_init();
        // Enable llama.cpp's internal logs via tracing.
        // `send_logs_to_tracing` runs once per process, so calling it
        // before the first `LlamaModel::load_from_file` is the only
        // window where the configuration takes effect. The default
        // `LogOptions` has logs enabled — surfacing the C-side
        // diagnostic stream is now the default behavior.
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        LlamaBackend::init().expect("llama_cpp_2::LlamaBackend::init must succeed exactly once")
    })
}

/// Structured error type for the inference engine path
/// ([`load_inference`] + [`invoke_with_model`]).
///
/// Each variant maps to one upstream `llama-cpp-2` failure surface,
/// preserving the source error via `#[source]` so
/// `anyhow::Error::new(InferenceError::...)` retains the full chain
/// downstream callers can walk via `.chain()` / `.root_cause()`.
///
/// The variants split along upstream fallible boundaries:
///
/// - [`Self::ModelLoad`] — `LlamaModel::load_from_file` failed (path
///   not readable, GGUF metadata corrupt, the linked llama.cpp
///   build's loader rejected the format). Carries the resolved
///   `PathBuf` because the offline-gate / cache resolution is
///   already handled upstream and the operator wants to know which
///   artifact slot tripped.
/// - [`Self::ContextCreate`] — `LlamaModel::new_context` failed.
///   Practically only fires under exotic context-param shapes
///   (negative `n_ctx`, oversize KV reservations) — the reason
///   string carries the upstream Display.
/// - [`Self::Tokenize`] — `LlamaModel::str_to_token` failed
///   (NUL-byte in the prompt, or `c_int` overflow on prompt length;
///   the latter is theoretically reachable via a multi-GiB prompt).
///   The `prompt_excerpt` carries the first 64 bytes so an operator
///   debugging tokenization can see what hit the boundary without
///   the full prompt body in the error chain.
/// - [`Self::Decode`] — `LlamaContext::decode` failed (KV-cache
///   exhaustion via `NoKvCacheSlot`, empty batch via `NTokensZero`,
///   or an unknown ffi code).
/// - [`Self::Generation`] — catch-all for the per-token-step
///   failures that are not first-class llama.cpp surfaces:
///   `LlamaBatch::add` (`InsufficientSpace`) and
///   `LlamaModel::token_to_piece` (`UnknownTokenType`,
///   `InsufficientBufferSpace`, `FromUtf8Error`). Each call site
///   threads its own `reason` string identifying the step
///   ("seed prompt batch", "decode generated token", etc.) so the
///   error chain is actionable without a typed source variant per
///   distinct llama-cpp-2 error.
#[derive(Debug, thiserror::Error)]
pub(crate) enum InferenceError {
    #[error(
        "GGUF model load failed at {path}. The file may be corrupt or \
         incompatible with the linked llama.cpp version — delete the \
         file and re-run `cargo ktstr model fetch` to download a fresh \
         copy. Check stderr for the upstream llama.cpp rejection reason."
    )]
    ModelLoad {
        path: PathBuf,
        #[source]
        source: llama_cpp_2::LlamaModelLoadError,
    },

    #[error("create LlamaContext for inference")]
    ContextCreate {
        #[source]
        source: llama_cpp_2::LlamaContextLoadError,
    },

    #[error("tokenize ChatML prompt (excerpt: {prompt_excerpt:?})")]
    Tokenize {
        prompt_excerpt: String,
        #[source]
        source: llama_cpp_2::StringToTokenError,
    },

    #[error("llama_decode failed")]
    Decode {
        #[source]
        source: llama_cpp_2::DecodeError,
    },

    #[error("inference generation step failed: {reason}")]
    Generation { reason: String },
}

/// Truncation byte count for [`InferenceError::Tokenize::prompt_excerpt`].
/// The full ChatML-wrapped prompt body can run into multiple KiB
/// — surfacing all of it in an error chain would crowd the
/// rendering downstream consumers print. 64 bytes is enough to
/// fingerprint which prompt category triggered the failure
/// (compose_prompt always opens with the literal
/// `<|im_start|>user\n` ChatML header).
const PROMPT_EXCERPT_BYTES: usize = 64;

/// Take the first [`PROMPT_EXCERPT_BYTES`] bytes of `prompt`,
/// snapped backward to a char boundary so a multi-byte UTF-8
/// codepoint at the boundary doesn't panic the slice. Used by
/// [`InferenceError::Tokenize`] to keep the error chain compact.
fn prompt_excerpt(prompt: &str) -> String {
    if prompt.len() <= PROMPT_EXCERPT_BYTES {
        return prompt.to_string();
    }
    // Walk backward from PROMPT_EXCERPT_BYTES until we hit a char
    // boundary. The first byte (offset 0) is always a boundary, so
    // this loop terminates.
    let mut end = PROMPT_EXCERPT_BYTES;
    while end > 0 && !prompt.is_char_boundary(end) {
        end -= 1;
    }
    prompt[..end].to_string()
}

/// Process-wide memoized inference state.
///
/// The outer `Mutex` serializes initialization and gates access to the
/// `Option`; the inner `Arc` lets concurrent callers each hold the
/// shared inference state for the duration of their generation pass
/// without keeping the outer mutex locked. The double layer of
/// `Mutex`es (outer over the slot, inner over the model) is
/// deliberate — see "Lock layering" below.
///
/// # Serialization guarantee
///
/// The outer `Mutex` makes [`memoized_inference`] atomic: a caller
/// arriving with the slot still `None` runs `load_inference`
/// end-to-end and stores the result; competing callers block on the
/// `lock()` until the initializer returns, then read the now-`Some`
/// slot and proceed. So the 2.55 GiB GGUF load in `load_inference`
/// happens at most once per process rather than once per racing
/// thread.
///
/// # Fail-closed on load error
///
/// The stored value is a [`Result`] so a load failure (missing model
/// under the offline gate, malformed SHA pin, corrupt GGUF) is
/// memoized as `Err(message)`. Subsequent calls
/// observe the cached error and return an empty metric set without
/// re-attempting the load. Retrying would repeat the same failure —
/// the offline gate does not flip, a placeholder pin does not become
/// real, and a corrupt cache entry does not self-heal — so re-trying
/// would only burn wall time. The error is stored pre-rendered as a
/// `String` (the full `{e:#}` chain of the original `anyhow::Error`)
/// because every cached-miss call wants the same human-readable
/// message in its `tracing::warn` line — rendering once at
/// memoization time keeps the hot path a cheap `&str` borrow.
///
/// # Panic vs. returned Err
///
/// Only a returned `Err` is fail-closed — a panic inside the
/// `load_inference` closure leaves the slot still `None` (the
/// assignment that stores the `Some(_)` runs after `load_inference`
/// returns) and the next caller re-runs the initializer. The outer
/// `Mutex` will be marked poisoned by the panic, but
/// [`memoized_inference`] recovers via `unwrap_or_else(|e|
/// e.into_inner())` so a poisoned lock does not wedge later callers.
/// Fail-closed memoization therefore applies exclusively to errors
/// returned through the normal `Result` channel; load paths that can
/// panic (e.g. llama.cpp-side allocation failure surfacing through
/// the FFI as a non-Result panic) do not poison the cache.
///
/// The panic-then-retry behavior described above only applies under
/// the `panic = "unwind"` strategy — i.e. the default debug/test
/// profile. ktstr's release profile sets `panic = "abort"` (see
/// `Cargo.toml [profile.release]`); under abort a panic inside the
/// initializer aborts the process before control returns to
/// [`memoized_inference`], so there is no "next caller" within the
/// same process and the "both `Ok` and `Err` are cached" guarantee
/// is moot for panics. Only values returned through the `Result`
/// channel are cached in release builds; panics terminate.
///
/// # Lock layering
///
/// The outer `Mutex<Option<Arc<...>>>` is held only across the slot
/// read/init/clone window. After initialization, a subsequent caller
/// sees `Some(arc)` and the critical section collapses to a mutex
/// lock + clone + unlock (sub-microsecond).
///
/// **First-call blocking.** The first caller to reach
/// [`memoized_inference`] with an empty slot runs `load_inference`
/// inside the outer lock. That load opens the pinned Qwen3-4B
/// Q4_K_M GGUF (~2.55 GiB) via
/// `llama_cpp_2::LlamaModel::load_from_file`, which mmap's the file
/// and routes the per-layer quantized tensors into a `LlamaModel`
/// owned by llama.cpp. Every concurrent caller queued behind the
/// outer mutex blocks for that entire window. Under nextest's
/// default parallel execution, every `LlmExtract` test racing
/// into the first call serializes here until the loader returns.
/// This is deliberate — the single-loader contract is what gives
/// the cached `Arc<CachedInference>` its "load exactly once per
/// process" invariant and avoids paying 2+ GiB of wasted load
/// work per additional concurrent first-caller. The first
/// `LlmExtract` test in a process pays the load cost once;
/// subsequent tests reuse the memoized [`MODEL_CACHE`] slot.
///
/// The inner `Mutex<LoadedInference>` is held for the full duration
/// of a generation pass and serializes concurrent inference calls
/// against the shared `LlamaModel`. Holding the inner mutex via
/// the cloned `Arc` (rather than via the outer slot) means a caller
/// running inference does not block other callers from observing
/// the slot is already populated. A fresh `LlamaContext` is built
/// per call from `&LlamaModel` (the model's `new_context` borrows
/// `&self`) so the per-generation KV state never aliases across
/// invocations — KV state lives on the `LlamaContext`, which is
/// constructed and destroyed per call, so no cross-invocation
/// `clear_kv_cache` step is needed.
///
/// # Test-only reset
///
/// [`reset`] clears the slot and is the hook tests use to
/// re-exercise `load_inference` (and through it, `ensure()`'s
/// offline-gate trip) when they have just mutated `KTSTR_MODEL_OFFLINE`
/// or `KTSTR_CACHE_DIR`. Without that reset, a successful load in any
/// earlier test (real or future) would memoize an `Ok(_)` slot that
/// silently bypassed the offline gate on every subsequent call. The
/// reset is `#[cfg(test)]`-only — production code never clears the
/// memoized state.
///
/// # Fail-closed-forever policy (production)
///
/// An `Err(_)` slot is memoized exactly like an `Ok(_)` slot.
/// If the first call fails — missing weights, SHA mismatch, a
/// corrupt GGUF read, offline-gate trip — every subsequent call in
/// the same process returns that cached error without retrying the
/// load. Production has no escape hatch: there is no public
/// `clear_model_cache()` and `reset` is `#[cfg(test)]`-only.
/// Downstream consumers that embed ktstr must treat a first-call
/// failure as terminal for the lifetime of the process and surface
/// the error through their own orchestration rather than expecting
/// a retry to succeed. The rationale: a retry under a load pipeline
/// that already failed (bad SHA, truncated download, OOM) almost
/// always hits the same failure; a stable cached error keeps the
/// `LlmExtract` surface deterministic across the process lifetime
/// and lets callers log the error exactly once on the first
/// extraction attempt rather than on every subsequent one.
///
/// # Blast radius for transient failures (intentional)
///
/// The fail-closed-forever policy applies UNIFORMLY across both
/// permanent and transient failure modes. A first-call failure
/// from a transient cause poisons the slot for the entire process
/// lifetime exactly the same as a permanent cause:
///
/// * **NFS hiccup / network pause** during the initial GGUF read —
///   `LlamaModel::load_from_file` returns an I/O error, that error
///   is memoized, and every later `LlmExtract` test in the same
///   process gets `LlmExtract model load failed: <io error>`
///   without re-attempting the read even after the network
///   recovers.
/// * **OOM kill survival** — a transient memory-pressure event that
///   caused the loader to fail (e.g. concurrent test consumed the
///   page cache, leaving llama.cpp's mmap to thrash and produce a
///   read failure) sticks for the whole process even after memory
///   pressure clears.
/// * **Tempfile race during fetch** — if [`ensure`] landed a
///   partial file under the pinned name and the loader saw a
///   truncated read, the cached Err sticks until process exit
///   even if a later writer completes the file under the same path.
/// * **NFS file-handle stale** after a server-side rename — the
///   first `read` returns ESTALE, that error memoizes, and every
///   later call observes it even after the client revalidates
///   the handle.
///
/// The blast radius is **session-wide** in nextest's default
/// concurrent test execution: every `#[ktstr_test]` annotated as
/// `OutputFormat::LlmExtract` shares the same process and the same
/// `MODEL_CACHE`. A single transient failure on the FIRST call
/// surfaces an `LlmExtract model load failed` AssertDetail on
/// every subsequent LlmExtract test in the run. Operator
/// observation: a CI report shows the failing tests clustered
/// with the same error message, and the fix is a re-run rather
/// than a per-test retry.
///
/// This is INTENTIONAL despite the wider blast radius. Three
/// reasons rule against per-call retry:
///
/// 1. **Discriminating transient from permanent at the load
///    boundary is unreliable.** A `std::io::Error` does not carry
///    a bit that says "transient" — an ESTALE, ETIMEDOUT, or
///    ENOMEM is recoverable in principle but the loader sees the
///    same `Err(io)` shape as ENOENT or EACCES. A retry policy
///    keyed on errno would mis-classify a real configuration
///    error as transient and burn 30s+ of wasted retries on each
///    of dozens of tests.
/// 2. **A retry under load pressure compounds the original
///    failure.** Re-attempting a 2.55 GiB mmap that just OOM-killed
///    a peer most likely re-OOMs. Keeping the Err sticky lets the
///    operator restart the process in a less-pressured environment
///    rather than blocking forward progress on a doomed retry.
/// 3. **Test determinism is more important than blast-radius
///    minimization.** A flaky retry policy that sometimes recovers
///    and sometimes doesn't would surface as intermittent
///    "LlmExtract worked in run N, failed in run N+1, worked in
///    N+2" reports — exactly the failure mode `LlmExtract` tests
///    must avoid (they already produce model-driven outputs that
///    drift across runs without a retry-induced jitter on top).
///    A sticky-Err keeps a failed run failing identically, which
///    operators can investigate once and fix at the source.
///
/// **Mitigation for the operator**: the cached error string
/// captures the full anyhow chain (the `{e:#}` rendering at the
/// memoization site, see "Fail-closed on load error" above). An
/// operator who sees a transient-flavored error (ESTALE, ETIMEDOUT,
/// ENOMEM, EAGAIN) can re-run the test process to retry from a
/// clean slate. CI orchestration should treat first-call LlmExtract
/// failures as a re-run signal rather than a hard fail when the
/// underlying error is recognizably transient — the framework
/// surfaces the cause verbatim to enable that decision.
///
/// **Mitigation considered and rejected**: a timestamp-based
/// retry where the cached Err expires after N seconds and the
/// next call re-attempts the load. Rejected because (a) the retry
/// would race against the caller's own deadline (LlmExtract
/// tests run with `timeout` on the payload and expect the host
/// to either succeed quickly or surface a stable error), and (b)
/// the timestamp would need to be tested for monotonicity across
/// concurrent calls, adding lock contention to a hot path. A
/// future revision could differentiate `LoadFailureKind::Transient`
/// vs `Permanent` in `load_inference` and apply retry to the
/// transient subset only — but that requires a structured error
/// type at the loader boundary which llama-cpp-2 currently does
/// not surface, so the work is gated on upstream API support.
type CachedInference = Result<Mutex<LoadedInference>, String>;
static MODEL_CACHE: Mutex<Option<Arc<CachedInference>>> = Mutex::new(None);

/// Test-only counter incremented each time [`memoized_inference`]
/// takes the slow path (observes `None` in [`MODEL_CACHE`] and calls
/// [`load_inference`]). Pins the at-most-one-load-per-slot invariant
/// empirically: a cached `Ok`/`Err` entry must short-circuit every
/// future call without re-invoking the load pipeline. Under the
/// outer `Mutex`, increments are serialized with slot population, so
/// a plain `AtomicUsize` suffices.
#[cfg(test)]
static MODEL_CACHE_LOAD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Pinned description of a model artifact the cache knows how to
/// fetch and check.
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
    ///
    /// Computing the digest for a new pin (see the "model pin
    /// rotation" section on [`DEFAULT_MODEL`]): fetch the artifact,
    /// then run
    ///
    /// ```text
    /// sha256sum <file>      # GNU coreutils (Linux)
    /// shasum -a 256 <file>  # BSD / macOS
    /// ```
    ///
    /// The leading 64-hex token of the output is this field. The
    /// `is_valid_sha256_hex` gate at module scope compile-fails
    /// any pin that is not exactly 64 ASCII hex chars, so
    /// pasting the trailing filename or a truncated prefix trips a
    /// `const { assert!(...) }` at crate build time rather than at
    /// first fetch.
    pub sha256_hex: &'static str,
    /// Approximate on-disk size in bytes; surfaced in status output
    /// so users can tell at a glance whether the cache entry is the
    /// right artifact. Not used for the integrity check (SHA is the gate).
    pub size_bytes: u64,
}

/// Default model served when a payload declares
/// [`OutputFormat::LlmExtract`] without pointing at a custom pin.
///
/// Qwen3.5-4B Q4_K_M GGUF (~2.55 GiB).
/// The 4B-parameter tier gives usable structured-JSON extraction
/// quality at an artifact size small enough that host-side post-test
/// extraction loads and runs in reasonable wall time on CPU.
///
/// URL points at the official `Qwen/Qwen3.5-4B-GGUF` repo on
/// Hugging Face.
///
/// # Model pin rotation
///
/// When upgrading to a newer Qwen release (or swapping quantization),
/// update all three fields in lockstep — a partial edit produces a
/// `sha256 mismatch` on the next fetch at best, or a silently-wrong
/// artifact pulled over a stale digest at worst:
///
/// 1. **`url`** — point at the new artifact on Hugging Face. Must be
///    `https://` (the fetcher rejects `http://` unconditionally).
/// 2. **`sha256_hex`** — re-compute via
///    ```text
///    curl -fL <new_url> | sha256sum
///    ```
///    and paste the 64-hex token.
/// 3. **`size_bytes`** — set to the new artifact's on-disk byte count.
pub const DEFAULT_MODEL: ModelSpec = ModelSpec {
    file_name: "Qwen3.5-4B-Q4_K_M.gguf",
    url: "https://huggingface.co/Qwen/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
    sha256_hex: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
    size_bytes: 2740937888,
};

/// Canonical list of every [`ModelSpec`] declared in this module.
/// Single source of truth for the "iterate all specs at compile
/// time" shape checks below — adding a new `ModelSpec` const
/// anywhere in the file requires appending a reference here, which
/// forces the new pin through the compile-time validator without
/// requiring the author to hand-roll per-spec `const _: () =
/// assert!(..)` blocks at the declaration site.
///
/// The array is `&[&ModelSpec]` so the compile-time iterator below
/// walks pointers, not values — the entries are `const` references
/// to the module-level `DEFAULT_*` constants. Currently a single
/// entry — the GGUF carries its own tokenizer surface via
/// `llama-cpp-2`'s `LlamaModel`, so no separate tokenizer artifact
/// is registered — the slice is kept as a slice (rather than a bare
/// `const`) so future additions slot in without rewriting the
/// validator below.
const ALL_MODEL_SPECS: &[&ModelSpec] = &[&DEFAULT_MODEL];

// Module-scope compile-time shape check on every ModelSpec's SHA
// pin: 64 ASCII hex chars, anything else is a typo. Placed at
// module scope (not inside a `#[cfg(test)] fn`) so the assertion
// fires on every `cargo check` / `cargo build`, not only under
// `cargo check --tests`. A pin swap with a malformed hex string
// now fails the default build before any runtime test hits it.
// The `is_valid_sha256_hex` helper is const-evaluated, so the
// entire check folds at compile time with no runtime cost.
//
// Iterating [`ALL_MODEL_SPECS`] with a const `while` loop means the
// validator auto-applies to every future spec, and forgetting to
// register a new spec is the more likely (and easier-to-review)
// failure mode than forgetting to hand-roll a matching assert.
const _: () = {
    let mut i = 0;
    while i < ALL_MODEL_SPECS.len() {
        assert!(
            is_valid_sha256_hex(ALL_MODEL_SPECS[i].sha256_hex),
            "ModelSpec.sha256_hex must be 64 ASCII hex characters — \
             see ALL_MODEL_SPECS; add a registration line there when \
             declaring a new ModelSpec const",
        );
        i += 1;
    }
};

// Ballpark size bounds on the pinned artifact. The pinned Qwen3.5-4B
// Q4_K_M GGUF is ~2.55 GiB; bound tight at 3 GiB so a silent swap to a
// higher-bit quantization (Q5/Q6/Q8) of the same 4B-parameter base —
// which would balloon the artifact past 3 GiB and multiply inference
// latency — fails this check instead of slipping through. The lower
// bound of 100 MiB rejects a wildly truncated or placeholder pin.
//
// Module scope (not inside `#[test]`) so a pin rotation that slips
// past the ballpark fails `cargo check` without `--tests`, mirroring
// the SHA-hex pin guards above.
const _: () = assert!(
    DEFAULT_MODEL.size_bytes > 100 * 1024 * 1024,
    "DEFAULT_MODEL.size_bytes must exceed 100 MiB — pin truncation suspected",
);
const _: () = assert!(
    DEFAULT_MODEL.size_bytes < 3 * 1024 * 1024 * 1024,
    "DEFAULT_MODEL.size_bytes must stay under 3 GiB — higher-bit quant swap suspected",
);

// Every registered ModelSpec must declare a POSITIVE size_bytes. A
// zero byte count degenerates the free-space gate (`needed == 0`
// lets any available-space value pass even under full-disk
// conditions) and the fetch-timeout computation (`size_bytes / 3MBps
// == 0` collapses to the 60s floor, hiding the relationship between
// size and timeout). Rejecting at `ModelSpec` declaration time means
// [`compute_margin`]'s `max(1)` floor is belt-and-braces rather than
// load-bearing for any production pin — the floor only ever matters
// for the unit-test fixtures that explicitly exercise boundary
// inputs to the helper. Applied per-spec via `ALL_MODEL_SPECS` so
// future `ModelSpec` additions cannot slip past the check by
// forgetting a hand-rolled assertion.
const _: () = {
    let mut i = 0;
    while i < ALL_MODEL_SPECS.len() {
        assert!(
            ALL_MODEL_SPECS[i].size_bytes > 0,
            "ModelSpec.size_bytes must be positive — a zero-size pin \
             degenerates the free-space gate and fetch-timeout \
             computation; see ALL_MODEL_SPECS, add a registration \
             line there when declaring a new ModelSpec const",
        );
        i += 1;
    }
};

/// Environment variable that opts out of the lazy model fetch.
/// `KTSTR_MODEL_OFFLINE=1` (or any non-empty value) leaves the cache
/// untouched; `LlmExtract` tests then surface missing-model errors
/// at `ensure()` invocation time instead of fetching the GGUF on demand.
pub const OFFLINE_ENV: &str = "KTSTR_MODEL_OFFLINE";

/// Environment variable that opts into raw-response tracing for
/// LlmExtract. When set to any non-empty value,
/// [`extract_via_llm`] emits the full model output on every call as
/// a `tracing::debug!` event (field `response`) alongside the
/// existing parse-outcome warn. Off by default: a single debug
/// emission can run to multiple KiB of chat-formatted text with
/// leaked `<think>` traces under pathological prompts, which floods
/// CI logs and leaks prompt-dependent content when enabled blindly.
///
/// # When to enable
///
/// Debugging an LlmExtract test that lands in the "response was not
/// parseable JSON; returning empty metric set" branch. The warn-level
/// event only carries `response_bytes` (a byte count) by policy — a
/// short count suggests "empty response", a long count suggests
/// "large response missing JSON region" — but neither diagnoses the
/// actual content. Flipping this env routes the body through
/// `tracing::debug!` so a follow-up run with
/// `RUST_LOG=ktstr::test_support::model=debug` surfaces exactly what
/// the model emitted, letting the tester adjust the prompt, the hint,
/// or the JSON extraction window.
///
/// # Why opt-in, not always-on
///
/// The warn at byte-count granularity is the designed steady-state
/// signal: it is always safe to log, bounded in size, and answers
/// the first triage question (did the model produce anything?).
/// Routing the full body is reserved for explicit debugging because
/// (a) it multiplies log volume by orders of magnitude, and (b) it
/// can carry prompt-dependent content that would be noise in shared
/// CI transcripts.
pub const LLM_DEBUG_RESPONSES_ENV: &str = "KTSTR_LLM_DEBUG_RESPONSES";

/// Shared non-trim "env var is opt-in" predicate for boolean gates
/// like [`LLM_DEBUG_RESPONSES_ENV`]. Returns `true` iff `val` is
/// `Some` and non-empty; `None` and `Some("")` both map to `false`.
///
/// Callers pass `std::env::var(NAME).ok().as_deref()`; the pure
/// signature lets the predicate be unit-tested without touching the
/// process environment (which would require the
/// `ENV_LOCK`-serialised env-mutation dance).
fn env_value_is_opt_in(val: Option<&str>) -> bool {
    matches!(val, Some(s) if !s.is_empty())
}

/// Read [`OFFLINE_ENV`] and return the trimmed value IFF it is set
/// to a non-empty string. Centralizes the "non-empty env-var means
/// opt-in" predicate used by [`ensure`] (treating
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

/// Outcome of the SHA-256 integrity check for a potentially-cached
/// model artifact.
///
/// Collapses the former `(sha_matches: bool, sha_check_error:
/// Option<String>)` pair on [`ModelStatus`] into a single enum so
/// the impossible `(true, Some(_))` combination — "check succeeded
/// AND recorded an error" — is unrepresentable at the type level.
/// The four variants span every outcome [`status`] can produce; no
/// other combination is constructible.
///
/// Remediation differs by variant so keeping them distinct matters:
/// a [`Self::Mismatches`] points at the bytes (re-fetch or re-pin);
/// a [`Self::CheckFailed`] points at the filesystem entry
/// (permissions, truncation, filesystem errors). The CLI
/// `model status` readout and the offline-gate bail in [`ensure`]
/// both branch on the variant to name the specific remediation
/// rather than defaulting to a generic "doesn't match."
#[derive(Debug, Clone)]
pub enum ShaVerdict {
    /// No cached file was present at the expected path; no SHA-256
    /// check was performed. The `_ => ...` arm of [`status`]'s
    /// metadata probe produces this.
    NotCached,
    /// SHA-256 digest of the cached file equals the declared pin.
    /// Ok(true) from [`check_sha256`].
    Matches,
    /// SHA-256 digest was computed successfully but did not equal
    /// the declared pin. Ok(false) from [`check_sha256`].
    /// Remediation: re-fetch, or re-pin if the cached bytes are
    /// known-correct.
    Mismatches,
    /// Cached file existed but its SHA-256 could not be computed
    /// due to an I/O failure (open/read/permission error). Carries
    /// the rendered error chain (`{e:#}`) for diagnostic output.
    /// Produced only when the pin itself parses as valid hex; a
    /// malformed pin is a programmer error and still surfaces as
    /// an `Err` from [`status`] rather than being folded in here.
    CheckFailed(String),
}

impl ShaVerdict {
    /// Whether a cached file is present. `true` for every variant
    /// except [`Self::NotCached`]. Convenience for call sites that
    /// only care about presence (e.g. the CLI readout's `cached:`
    /// line, test assertions asserting a file landed on disk).
    pub fn is_cached(&self) -> bool {
        !matches!(self, Self::NotCached)
    }

    /// Whether the cached file passed its SHA-256 check. `true`
    /// iff the variant is [`Self::Matches`]. [`ensure`]'s fast path
    /// gates on this. Named `is_match` (not `matches`) to match the
    /// `is_*` accessor convention used by sibling enums
    /// (e.g. `KconfigStatus::{is_stale, is_untracked}` and the
    /// `ShaVerdict::is_cached` accessor right above) and to avoid
    /// collision with the `matches!` macro in call-site patterns.
    pub fn is_match(&self) -> bool {
        matches!(self, Self::Matches)
    }

    /// Rendered I/O-error string iff the variant is
    /// [`Self::CheckFailed`], else `None`. Used by the CLI readout
    /// and the offline-gate bail to name the underlying failure.
    pub fn check_error(&self) -> Option<&str> {
        match self {
            Self::CheckFailed(e) => Some(e.as_str()),
            _ => None,
        }
    }
}

/// Status record returned by [`status`]: where the model would live
/// on disk and the outcome of the SHA-256 check. Presence (former
/// `cached: bool`) and check outcome (former `sha_matches: bool` +
/// `sha_check_error: Option<String>`) are now unified in
/// [`sha_verdict`](Self::sha_verdict); call sites use
/// [`ShaVerdict::is_cached`] / [`ShaVerdict::is_match`] /
/// [`ShaVerdict::check_error`] to read the fields they need.
#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub spec: ModelSpec,
    pub path: PathBuf,
    pub sha_verdict: ShaVerdict,
}

/// Resolve the model cache root, creating it lazily when a writer
/// needs it. Delegates to
/// [`crate::cache::resolve_cache_root_with_suffix`] with the
/// `"models"` suffix so the kernel cache and the model cache share a
/// single source of truth for env-variable handling
/// (`KTSTR_CACHE_DIR` non-UTF-8 bail, `XDG_CACHE_HOME`) and
/// HOME validation (3 arms: unset/empty, literal `/`, non-absolute
/// path). The thin wrapper preserves the per-call
/// `tracing::debug!` env-snapshot for operators diagnosing
/// cache-resolution surprises with `RUST_LOG=debug`.
pub(crate) fn resolve_cache_root() -> Result<PathBuf> {
    // Trace the env-var snapshot at debug level. The earlier
    // implementation emitted this on every call as an unconditional
    // `eprintln!`, which spammed every CI test boot with HOME /
    // XDG_CACHE_HOME / KTSTR_CACHE_DIR diagnostics that operators
    // never asked for. Routed through `tracing::debug!` so the
    // information is available with `RUST_LOG=debug` for operators
    // diagnosing a cache-resolution surprise without cluttering the
    // default test output.
    tracing::debug!(
        home = ?std::env::var("HOME"),
        xdg_cache_home = ?std::env::var("XDG_CACHE_HOME"),
        ktstr_cache_dir = ?std::env::var("KTSTR_CACHE_DIR"),
        "model::resolve_cache_root: env snapshot",
    );
    crate::cache::resolve_cache_root_with_suffix("models")
}

/// Compute the [`ShaVerdict`] for the cached artifact at `path`
/// against the pin recorded in `spec`. Shared between [`status`]
/// (which passes `use_sidecar_fastpath = true` for the quick
/// "cache health" read) and [`ensure`] (which passes `false` to
/// force a full re-hash so the integrity-gate answer does not
/// inherit any warm-cache sidecar false-positive).
///
/// The sidecar fast path is a performance optimization, not a
/// security boundary. mtime-preserving operations (`rsync -t`,
/// `tar -xp`, `touch -r`, coarse-mtime filesystems that round to
/// second or coarser granularity) can produce a sidecar match
/// after the file content has changed. Callers that gate
/// downstream code on byte-exact integrity (LlmExtract expecting
/// the pinned model, test harnesses that compare outputs across
/// runs) must pass `use_sidecar_fastpath = false` so an mtime
/// spoof cannot slip past the SHA check. A cached file that was
/// just downloaded by this process is indistinguishable on the
/// fast path from one touched externally minutes ago, so
/// callers that want true integrity cannot rely on the sidecar
/// alone.
///
/// Error handling mirrors the prior inline implementation:
/// a malformed SHA pin is a `ModelSpec` programmer error and
/// bubbles out as `Err`, while a transient I/O failure on the
/// cached file maps to `ShaVerdict::CheckFailed` so the
/// downstream offline-gate bail can name the specific reason.
/// On `Ok(true)` the sidecar is refreshed (best-effort); on
/// `Ok(false)` the stale sidecar is removed so a future verify
/// cannot short-circuit against rejected bytes.
fn compute_sha_verdict(
    path: &std::path::Path,
    spec: &ModelSpec,
    use_sidecar_fastpath: bool,
) -> Result<ShaVerdict> {
    Ok(match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => {
            if use_sidecar_fastpath && sidecar_confirms_prior_sha_match(path, &meta) {
                ShaVerdict::Matches
            } else {
                match check_sha256(path, spec.sha256_hex) {
                    Ok(true) => {
                        // Best-effort sidecar refresh so the
                        // next status() call short-circuits.
                        // Write failures are logged and
                        // swallowed — the fast path is an
                        // optimization, not correctness.
                        if let Err(e) = write_mtime_size_sidecar(path) {
                            tracing::debug!(
                                artifact = %path.display(),
                                %e,
                                "mtime-size sidecar write failed; next status() will re-hash",
                            );
                        }
                        ShaVerdict::Matches
                    }
                    Ok(false) => {
                        // Drop the stale warm-cache sidecar:
                        // its recorded (mtime, size) now
                        // describes bytes that the pin
                        // explicitly rejects. Leaving it on
                        // disk risks a future status() call
                        // short-circuiting against those bad
                        // bytes if an operator repairs the
                        // cache WITHOUT the mtime or size
                        // changing (touch-replace, rsync -t,
                        // coarse-mtime fs rounding). Removing
                        // the sidecar forces the next call to
                        // re-hash and rewrite, ensuring
                        // sidecar state tracks the artifact's
                        // true integrity.
                        remove_mtime_size_sidecar(path);
                        ShaVerdict::Mismatches
                    }
                    Err(e) => {
                        if !is_valid_sha256_hex(spec.sha256_hex) {
                            return Err(e).with_context(|| {
                                format!("check SHA-256 pin for cached model '{}'", spec.file_name,)
                            });
                        }
                        ShaVerdict::CheckFailed(format!("{e:#}"))
                    }
                }
            }
        }
        _ => ShaVerdict::NotCached,
    })
}

/// Return the on-disk path the spec would occupy and the outcome
/// of the SHA-256 integrity check as a [`ShaVerdict`]. Used by the
/// CLI's `model status` subcommand. Uses the warm-cache sidecar
/// short-circuit for responsiveness; see [`compute_sha_verdict`] for
/// the strict-integrity alternative consumed by [`ensure`].
pub fn status(spec: &ModelSpec) -> Result<ModelStatus> {
    let root = resolve_cache_root()?;
    let path = root.join(spec.file_name);
    // `status()` uses the warm-cache sidecar fast path because its
    // callers (the CLI `model status` subcommand and operators running
    // `cargo ktstr model status`) want an inexpensive "is the cache
    // healthy enough to skip re-fetching" answer. `ensure()`, the
    // integrity gate that hands out the cached path to downstream
    // LlmExtract, calls [`compute_sha_verdict`] with
    // `use_sidecar_fastpath = false` to bypass the sidecar and
    // re-hash, trading the ~10s SHA walk for strict integrity.
    let sha_verdict = compute_sha_verdict(&path, spec, true)?;
    Ok(ModelStatus {
        spec: *spec,
        path,
        sha_verdict,
    })
}

/// Report emitted by [`clean`]: which files were deleted (or absent)
/// and how many bytes each freed. The two paths are returned even
/// when their `*_freed_bytes` field is `None` so a caller rendering
/// the operator-facing message can name the path that was checked
/// (and confirm the cache root resolved as expected) regardless of
/// whether the file was actually present.
///
/// Pre-1.0: callers (`cargo ktstr model clean`) read these fields
/// directly; no `Display` impl is provided because the renderer
/// belongs to the consumer (CLI) layer rather than the library.
#[derive(Debug, Clone)]
pub struct CleanReport {
    /// Cache path of the GGUF artifact (`{cache_root}/models/{file}`).
    /// Always populated: even on the absent-file branch the caller
    /// wants to report which path was checked.
    pub artifact_path: PathBuf,
    /// `Some(N)` when the artifact existed at `artifact_path` and
    /// was deleted (N is the file size in bytes captured before
    /// `remove_file`). `None` when the artifact was absent — no
    /// deletion happened.
    pub artifact_freed_bytes: Option<u64>,
    /// Path of the `.mtime-size` warm-cache sidecar that lives
    /// alongside `artifact_path`. Always populated for the same
    /// reason as `artifact_path`.
    pub sidecar_path: PathBuf,
    /// `Some(N)` when the sidecar existed and was deleted; `None`
    /// when absent. Independent of `artifact_freed_bytes` because
    /// the sidecar can be present without the artifact (sidecar
    /// is a warm-cache helper, not a guard) and vice versa.
    pub sidecar_freed_bytes: Option<u64>,
}

impl CleanReport {
    /// `true` when neither the artifact nor the sidecar existed —
    /// the "no cached model found" case. Callers branch on this to
    /// emit a single "nothing to clean" line instead of two
    /// "(absent)" lines.
    pub fn is_empty(&self) -> bool {
        self.artifact_freed_bytes.is_none() && self.sidecar_freed_bytes.is_none()
    }

    /// Total bytes freed by the clean operation (artifact + sidecar).
    /// Sidecar size is typically ~50 bytes (a magic header line and
    /// a `mtime size` line); artifact is the multi-GiB GGUF. The
    /// sum is what operators want to see as "freed" — splitting
    /// the two would over-emphasize the sidecar.
    pub fn total_freed_bytes(&self) -> u64 {
        self.artifact_freed_bytes.unwrap_or(0) + self.sidecar_freed_bytes.unwrap_or(0)
    }
}

/// Remove the cached GGUF artifact for `spec` plus its `.mtime-size`
/// warm-cache sidecar, returning a [`CleanReport`] describing what
/// was deleted and how many bytes were freed.
///
/// Both files are removed independently — a caller cleaning up
/// after a corrupt fetch may have one file but not the other on
/// disk (e.g. the partial download landed at the artifact path
/// without the sidecar ever being written, or a manual edit
/// removed the artifact but left the sidecar pointing at stale
/// metadata). Each file's size is captured BEFORE `remove_file`
/// so the report is accurate even if the unlink race-loses to
/// another process.
///
/// Errors:
///  - [`resolve_cache_root`] failure (HOME unset, KTSTR_CACHE_DIR
///    non-UTF-8, etc.) propagates up — the operator needs the
///    cache root before any deletion can happen.
///  - `metadata` errors other than `NotFound` propagate up so a
///    permission-denied or I/O failure surfaces actionably
///    instead of being swallowed.
///  - `remove_file` errors propagate up for the same reason. A
///    successful metadata read followed by a failed remove is the
///    main case here (concurrent unlink, read-only filesystem).
///
/// Subsequent `cargo ktstr model fetch` re-downloads the pin from
/// scratch; subsequent `cargo ktstr model status` reports
/// `NotCached`.
pub fn clean(spec: &ModelSpec) -> Result<CleanReport> {
    let root = resolve_cache_root()?;
    let artifact_path = root.join(spec.file_name);
    let sidecar_path = mtime_size_sidecar_path(&artifact_path);

    let artifact_freed_bytes = remove_if_present(&artifact_path)?;
    let sidecar_freed_bytes = remove_if_present(&sidecar_path)?;

    Ok(CleanReport {
        artifact_path,
        artifact_freed_bytes,
        sidecar_path,
        sidecar_freed_bytes,
    })
}

/// Capture the size and remove the file at `path`. Returns
/// `Ok(Some(size))` when the file existed and was deleted,
/// `Ok(None)` when absent (no error), and propagates other I/O
/// failures (permission denied, read-only filesystem, dangling
/// symlink whose target is unreachable, etc.) so [`clean`] surfaces
/// them rather than silently dropping the cleanup.
///
/// Size is captured BEFORE `remove_file` so the returned count
/// describes what was actually freed even if a peer process
/// races to truncate the file between metadata and remove.
fn remove_if_present(path: &std::path::Path) -> Result<Option<u64>> {
    use anyhow::Context;

    match std::fs::metadata(path) {
        Ok(meta) => {
            let size = meta.len();
            std::fs::remove_file(path)
                .with_context(|| format!("remove cached model file {}", path.display()))?;
            Ok(Some(size))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => {
            Err(e).with_context(|| format!("stat cached model file {} for cleanup", path.display()))
        }
    }
}

/// Magic header line prefixing every
/// `{artifact}.mtime-size` warm-cache sidecar. A sidecar whose
/// first line does not match this literal is rejected as
/// truncated, corrupted, or written by an incompatible schema
/// version — [`read_mtime_size_sidecar`] treats any such file as
/// absent and drops through to the slow SHA-256 walk. The
/// explicit `_V1` suffix lets a future rewrite that carries
/// additional fields (e.g. a file inode) bump to `_V2` and have
/// older sidecars deserialize as "absent" rather than as
/// accidental matches against the new layout.
const MTIME_SIZE_SIDECAR_MAGIC: &str = "KTSTR_SHA_MTIME_SIZE_V1";

/// Path of the warm-cache revalidation sidecar alongside
/// `artifact`. Named with a `.mtime-size` suffix so operators
/// inspecting the cache directory can identify it without
/// guessing.
fn mtime_size_sidecar_path(artifact: &std::path::Path) -> PathBuf {
    let mut s = artifact.as_os_str().to_owned();
    s.push(".mtime-size");
    PathBuf::from(s)
}

/// Return `true` iff a `{artifact}.mtime-size` sidecar exists and
/// records the same (mtime_ns, size_bytes) as `meta`. Any I/O error
/// or parse failure returns `false` — callers fall back to the
/// slow path.
fn sidecar_confirms_prior_sha_match(artifact: &std::path::Path, meta: &std::fs::Metadata) -> bool {
    let current = match mtime_size_from_metadata(meta) {
        Some(v) => v,
        None => return false,
    };
    match read_mtime_size_sidecar(artifact) {
        Some(stored) => stored == current,
        None => false,
    }
}

/// Read a previously-written (mtime_ns, size_bytes) pair from the
/// sidecar, or `None` on any error (sidecar missing, missing or
/// mismatching magic header, truncated, malformed contents,
/// unreadable).
///
/// Format: two lines.
///   1. Exactly the [`MTIME_SIZE_SIDECAR_MAGIC`] literal.
///   2. Whitespace-separated `{mtime_ns} {size_bytes}`.
///
/// A partial write (power loss or process kill between the
/// `std::fs::write` syscall and fs writeback flushing the full
/// payload) typically surfaces as a zero-length file or a file
/// carrying only the magic line; the tokeniser below then fails
/// to find the second field and returns `None`. The `None`
/// routes the caller to the slow-path re-hash, and a subsequent
/// successful verify rewrites the sidecar to a valid state.
/// This turns "truncated sidecar" from a silent cache-poisoning
/// risk (reading corrupted mtime/size and matching it spuriously
/// against current metadata) into a reliable fall-through.
fn read_mtime_size_sidecar(artifact: &std::path::Path) -> Option<(u128, u64)> {
    let contents = std::fs::read_to_string(mtime_size_sidecar_path(artifact)).ok()?;
    let mut lines = contents.lines();
    // Magic-header gate: reject anything whose first line is not
    // exactly the versioned literal. An absent line (empty file),
    // a truncated line, or an older-schema sidecar all fail this
    // check and fall through to the slow path.
    if lines.next()? != MTIME_SIZE_SIDECAR_MAGIC {
        return None;
    }
    let payload = lines.next()?;
    let mut toks = payload.split_whitespace();
    let mtime: u128 = toks.next()?.parse().ok()?;
    let size: u64 = toks.next()?.parse().ok()?;
    Some((mtime, size))
}

/// Write the current mtime+size of `artifact` to its sidecar. The
/// sidecar's existence plus matching contents tells a future
/// [`status`] call it can skip the SHA-256 walk.
///
/// Writes the two-line format documented on
/// [`read_mtime_size_sidecar`]: magic header line + `{mtime}
/// {size}` payload line.
fn write_mtime_size_sidecar(artifact: &std::path::Path) -> std::io::Result<()> {
    let meta = std::fs::metadata(artifact)?;
    let (mtime, size) = mtime_size_from_metadata(&meta).ok_or_else(|| {
        std::io::Error::other("cannot capture mtime/size for revalidation sidecar")
    })?;
    std::fs::write(
        mtime_size_sidecar_path(artifact),
        format!("{MTIME_SIZE_SIDECAR_MAGIC}\n{mtime} {size}\n"),
    )
}

/// Best-effort removal of the `{artifact}.mtime-size` sidecar.
/// Called when the SHA-256 check against the artifact has
/// definitively rejected the cached bytes
/// ([`ShaVerdict::Mismatches`]): the sidecar's recorded
/// (mtime_ns, size_bytes) now describes bytes that the pin no
/// longer accepts, so leaving it on disk risks a future
/// fast-path short-circuit against bad bytes if the cache is
/// repaired WITHOUT the mtime/size changing (e.g. a rebuild that
/// preserves timestamps, or a touch-replace under coarse-mtime).
/// Unlink fails are logged — worst case, the next verify
/// recomputes the SHA and rewrites the sidecar with the correct
/// metadata, which is the desired end state anyway.
fn remove_mtime_size_sidecar(artifact: &std::path::Path) {
    let sidecar = mtime_size_sidecar_path(artifact);
    match std::fs::remove_file(&sidecar) {
        Ok(()) => tracing::debug!(
            sidecar = %sidecar.display(),
            artifact = %artifact.display(),
            "removed stale mtime-size sidecar after SHA mismatch",
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // No sidecar to remove — legitimate when the verify
            // ran on a freshly-downloaded entry that never
            // reached the write step, or when cleanup already
            // ran for this mismatch.
        }
        Err(e) => tracing::warn!(
            sidecar = %sidecar.display(),
            err = %format!("{e:#}"),
            "failed to remove stale mtime-size sidecar; next successful \
             verify will overwrite it",
        ),
    }
}

/// Pull mtime (as UNIX-epoch nanoseconds) and size from `meta`.
/// Returns `None` if the platform's mtime clock is unsupported or
/// predates the epoch; callers treat None as "fast path
/// unavailable, fall back to SHA".
fn mtime_size_from_metadata(meta: &std::fs::Metadata) -> Option<(u128, u64)> {
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some((mtime, meta.len()))
}

/// Ensure the model artifact described by `spec` is present and
/// SHA-checked in the cache, downloading if necessary.
///
/// Fast path: existing file whose SHA matches — no-op.
/// Slow path: tempfile download + SHA check + atomic rename.
///
/// Respects `KTSTR_MODEL_OFFLINE`: when set to a non-empty value,
/// returns `Err` immediately without issuing a network request. This
/// lets CI pipelines that pre-seed the cache fail loudly when the
/// pre-seed mechanism skipped an artifact, rather than silently
/// falling through to an online fetch.
pub fn ensure(spec: &ModelSpec) -> Result<PathBuf> {
    // BYPASS the warm-cache mtime/size sidecar: callers of
    // `ensure()` (LlmExtract, test harnesses handing the cached
    // path to the llama.cpp loader, anyone pinning a specific
    // model-weights commit) expect byte-exact integrity against
    // the declared SHA-256 pin. The sidecar fast path lets
    // mtime-preserving tampering (`rsync -t`, `touch -r`,
    // coarse-mtime fs rounding to 1 s or worse) produce a cached
    // artifact whose `{mtime, size}` matches the sidecar record
    // but whose BYTES do not match the pin. `status()` accepts
    // that trade-off for responsiveness, but `ensure()` is the
    // integrity gate. The cost is one full SHA-256 walk per
    // `ensure()` call against an existing cache entry (~10 s for
    // the 2.55 GiB Qwen3-4B pin); the prefetch at nextest
    // bootstrap amortises this over every test in the binary, and
    // the in-test cache reuses the post-ensure `ModelStatus` so
    // the walk fires at most once per process run.
    let root = resolve_cache_root()?;
    let path = root.join(spec.file_name);
    let verdict = compute_sha_verdict(&path, spec, false)?;
    let st = ModelStatus {
        spec: *spec,
        path,
        sha_verdict: verdict,
    };
    if st.sha_verdict.is_match() {
        return Ok(st.path);
    }
    // SHA-pin shape check runs before the offline-gate check. A
    // malformed or placeholder pin is a programmer error in the
    // `ModelSpec` itself — it does not depend on runtime state. Pre-
    // seeding a cache under the offline gate cannot rescue a broken
    // pin, so surfacing the shape failure first gives the clearer
    // diagnostic ("fix the ModelSpec") instead of the downstream
    // "KTSTR_MODEL_OFFLINE set but not cached" red herring.
    // `status()` already propagates a malformed-pin Err for the
    // cached-file branch (so ensure() never reaches this check with
    // a cached file + malformed pin). This explicit check covers the
    // no-cache case: status() returned `ShaVerdict::NotCached` without
    // calling `check_sha256`, so without this gate a placeholder
    // (all-`?`) pin would drop through to `fetch` and waste a
    // 2.55 GiB download before the post-download `check_sha256`
    // bails.
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
        // Distinguish the three paths that reach here: a missing
        // cache entry, a present-but-unreadable one (SHA check
        // failed with an I/O error), and a present-but-stale one
        // (SHA computed successfully but didn't match the pin).
        // All three trip the offline gate, but the remediation
        // differs — a no-cache case needs pre-seeding, a bytes-
        // mismatch case needs re-pinning or re-fetching the bytes,
        // and an I/O-unreadable case needs attention to the cache
        // entry's filesystem state (permissions, truncation,
        // missing extents). Collapsing these into a single generic
        // message misroutes the user.
        match &st.sha_verdict {
            ShaVerdict::CheckFailed(err) => anyhow::bail!(
                "{OFFLINE_ENV}={v_safe} set but model '{}' is cached at {} \
                 and the SHA-256 check could not complete ({}); \
                 inspect the cache entry (permissions, truncation, \
                 filesystem errors) or unset {OFFLINE_ENV} to re-fetch.",
                spec.file_name,
                st.path.display(),
                err,
            ),
            ShaVerdict::Mismatches => anyhow::bail!(
                "{OFFLINE_ENV}={v_safe} set but model '{}' is cached at {} \
                 with bytes that do not match the declared SHA-256 pin; \
                 replace the cache entry with bytes matching the pin (or \
                 unset {OFFLINE_ENV} to re-fetch).",
                spec.file_name,
                st.path.display(),
            ),
            ShaVerdict::NotCached => anyhow::bail!(
                "{OFFLINE_ENV}={v_safe} set but model '{}' is not cached at {}; \
                 pre-seed the cache or unset {OFFLINE_ENV} to fetch.",
                spec.file_name,
                st.path.display(),
            ),
            // `ShaVerdict::Matches` is the fast-path return at the
            // top of `ensure`; reaching the offline-gate with a
            // matching verdict would be a logic bug in `ensure`
            // itself, not a user-facing condition to diagnose.
            ShaVerdict::Matches => unreachable!(
                "fast path returned on Matches; reaching the \
                 offline-gate match with Matches is a logic bug"
            ),
        }
    }
    fetch(spec, &st.path)
}

/// Compute the overall HTTP-request timeout for a download of
/// `size_bytes`. Formula:
///
/// `min(FETCH_MAX_TIMEOUT_SECS,
///      max(FETCH_MIN_TIMEOUT_SECS,
///          size_bytes / FETCH_MIN_BANDWIDTH_BYTES_PER_SEC))`
///
/// where `FETCH_MIN_BANDWIDTH_BYTES_PER_SEC` is 3 MB/s
/// (`3_000_000`), `FETCH_MIN_TIMEOUT_SECS` is 60 s, and
/// `FETCH_MAX_TIMEOUT_SECS` is 1800 s (30 min). The proportional
/// term budgets a 3 MB/s sustained-throughput floor over the
/// artifact body; the 60 s floor keeps small artifacts from getting
/// a sub-second cap that TLS handshake + request/response round-trip
/// would blow past before the first body byte arrives. A regression
/// below the 3 MB/s floor surfaces as a timeout rather than hanging
/// the test setup until an external watchdog fires.
///
/// The 30 min ceiling bounds the wall clock that a single fetch can
/// consume regardless of how large the declared size is — without it,
/// a typo'd or unexpectedly large pin (e.g. a 20 GiB `size_bytes`)
/// would demand roughly 2 h of linear budget with no CI wall-clock
/// cap to stop it. The ceiling kicks in at `1800 s × 3 MB/s =
/// 5.4 GB` of body; the current pin (`DEFAULT_MODEL` ≈ 2.55 GiB) is
/// well under that crossover and continues to receive its linear
/// budget unchanged, and a future 5
/// GiB model pin (`5 × 1024³ / 3_000_000 ≈ 1789 s`) also sits just
/// under the cap. Pins beyond ~5 GB are the ones we explicitly want
/// bounded — the ceiling says "any artifact this codebase fetches
/// either finishes within 30 min or is pathological and should
/// fail fast so the operator notices."
///
/// No overflow path exists: integer division by the nonzero constant
/// `FETCH_MIN_BANDWIDTH_BYTES_PER_SEC` cannot panic and produces a
/// `u64` bounded by `size_bytes`; `u64::max` / `u64::min` return one
/// of their `u64` operands unchanged; and `Duration::from_secs`
/// accepts any `u64` without panicking.
fn fetch_timeout_for_size(size_bytes: u64) -> std::time::Duration {
    const FETCH_MIN_TIMEOUT_SECS: u64 = 60;
    const FETCH_MAX_TIMEOUT_SECS: u64 = 1800;
    const FETCH_MIN_BANDWIDTH_BYTES_PER_SEC: u64 = 3_000_000;
    let body_secs = size_bytes / FETCH_MIN_BANDWIDTH_BYTES_PER_SEC;
    let raw = body_secs.max(FETCH_MIN_TIMEOUT_SECS);
    std::time::Duration::from_secs(raw.min(FETCH_MAX_TIMEOUT_SECS))
}

/// Combine `blocks_available` and `fragment_size` from statvfs into
/// an available-byte count. Saturates at `u64::MAX` for pathological
/// FUSE mounts reporting enormous synthetic block/fragment counts;
/// `u64::MAX` is treated as unbounded space by [`ensure_free_space`]
/// so the gate passes — deliberate, since a false bail on spurious
/// overflow is worse than trusting the filesystem. Extracted so the
/// saturation predicate is addressable in tests that don't want to
/// mock a real filesystem.
fn bytes_from_statvfs_parts(blocks: u64, frag: u64) -> u64 {
    blocks.saturating_mul(frag)
}

/// Return the free space (in bytes, available to unprivileged users)
/// on the filesystem that holds `dir`. Wraps
/// [`nix::sys::statvfs::statvfs`]: `blocks_available` (`f_bavail`) is
/// expressed in units of `fragment_size` (`f_frsize`), so the
/// byte-level answer is the product.
///
/// `blocks_available` is used rather than `blocks_free` so the reading
/// honors the reserved-for-root slice POSIX filesystems carry — an
/// unprivileged process cannot actually consume the reserved slack,
/// and the fetcher runs unprivileged in the normal case.
///
/// Product is computed via [`bytes_from_statvfs_parts`], which
/// saturates at `u64::MAX` for pathological statvfs returns (FUSE
/// filesystems reporting enormous synthetic counts). A saturated
/// `u64::MAX` is effectively "unbounded space" for the subsequent
/// comparison; the gate will always pass.
fn filesystem_available_bytes(dir: &std::path::Path) -> Result<u64> {
    let vfs =
        nix::sys::statvfs::statvfs(dir).with_context(|| format!("statvfs {}", dir.display()))?;
    let blocks = vfs.blocks_available() as u64;
    let frag = vfs.fragment_size() as u64;
    Ok(bytes_from_statvfs_parts(blocks, frag))
}

/// Pre-flight gate in [`fetch`]: refuse to start a download when the
/// filesystem backing `parent` does not carry the declared artifact
/// size plus a 10% safety buffer against concurrent writers
/// consuming space between this snapshot check and the download's
/// final byte (see the "Best-effort only" paragraph below). Returns
/// `Ok(())` when enough room exists and `Err` with an actionable
/// diagnostic —
/// `"Need 2.69 GiB free at /path/to/cache; have 512 MiB"` — otherwise.
///
/// Needed bytes = `size_bytes + size_bytes / 10` (size plus 10%
/// margin). The division itself cannot overflow. The sum can
/// overflow only when `size_bytes` is greater than about
/// `u64::MAX * 10 / 11` (≈ 1.68e19), i.e. within the topmost ~9% of
/// the u64 range — a range no real `ModelSpec` pin reaches, but the
/// gate uses `saturating_add` anyway so a pathological or typo'd
/// value saturates at `u64::MAX` instead of wrapping to a smaller
/// `needed` that the `available < needed` check would let past.
///
/// Sizes are rendered through [`indicatif::HumanBytes`] so the error
/// message speaks in human-scale IEC prefixes (`GiB` / `MiB` / `KiB`)
/// instead of raw byte counts. A user reading
/// `"Need 2.69 GiB free ... ; have 512.03 MiB"` learns both the gap
/// and the order of magnitude at a glance; the raw-byte form
/// (`"Need 2883584000 bytes ..."`) forces mental arithmetic that
/// obscures the actionable "free up a couple of gigs" conclusion.
/// The file_name and the margin's 10% share are intentionally absent
/// from the one-line format — the former rarely matters to an
/// operator clearing disk, and the latter is an implementation detail
/// documented here in the source rather than echoed every time the
/// gate fires.
///
/// Best-effort only: the answer is a snapshot from statvfs at call
/// time. A concurrent writer on the same filesystem can still exhaust
/// space mid-download (surfacing later as the same ENOSPC error this
/// gate pre-empts). The gate catches the common "cache filesystem
/// nearly full" case before the HTTP request runs — it does not
/// claim reservation semantics.
/// The 10% safety buffer over a spec's declared size, floored
/// at 1 byte.
///
/// Integer division by 10 collapses to 0 for any
/// `size_bytes < 10`, which contradicts the "10% safety buffer"
/// claim in [`ensure_free_space`]'s doc. Clamping at `max(1)`
/// keeps the buffer > 0 for micro-specs — a defense-in-depth
/// floor that is redundant under the module-scope
/// `ALL_MODEL_SPECS[i].size_bytes > 0` + ballpark-size guards
/// above, which pin every production `ModelSpec` safely above
/// the `size_bytes < 10` regime. The floor stays so the helper
/// remains well-behaved under direct unit-test inputs that
/// explicitly exercise the `size_bytes < 10` boundary (see the
/// `compute_margin_respects_floor_*` family) without relying on
/// callers to pre-validate.
///
/// Specific size constants are NOT quoted in this doc so a
/// pin rotation that changes a ballpark does not drift this
/// comment. The module-scope `const _: () = assert!(...)` blocks
/// at the head of this file are the single authority for
/// production ballpark bounds; this helper's doc is intentionally
/// agnostic to them.
fn compute_margin(size_bytes: u64) -> u64 {
    (size_bytes / 10).max(1)
}

/// Render the free-space bail message, with an optional
/// FUSE/quota hint when `available == 0`.
///
/// Extracted from [`ensure_free_space`] so the message shape is
/// unit-testable without calling `statvfs` — the inputs
/// `needed`, `parent`, and `available` are pure values the caller
/// supplies. FUSE filesystems, quota-enforced mounts, and some
/// container overlays can report `blocks_available == 0` when no
/// user-visible free-space quota applies — the number reflects
/// the quota/overlay's view, not the underlying backing store.
/// Surfacing the hint only when `available == 0` keeps the
/// "normal" full-disk case's message un-cluttered.
fn format_free_space_error(needed: u64, parent: &std::path::Path, available: u64) -> String {
    let hint = if available == 0 {
        " (blocks_available reported 0 — if this is a FUSE \
         or quota-enforced mount, the free-space report may \
         be a filesystem-side misreport rather than a real \
         out-of-space condition; confirm with `df -h <mount>` \
         or `stat -f <mount>` to see the raw fs_bavail value, \
         then re-run with `XDG_CACHE_HOME` pointing at a \
         directory on a mount without the overlay — e.g. \
         `XDG_CACHE_HOME=/var/tmp/ktstr-cache` — so ktstr's \
         model cache lands on a filesystem the kernel reports \
         normally)"
    } else {
        ""
    };
    format!(
        "Need {} free at {}; have {}{hint}",
        indicatif::HumanBytes(needed),
        parent.display(),
        indicatif::HumanBytes(available),
    )
}

fn ensure_free_space(parent: &std::path::Path, spec: &ModelSpec) -> Result<()> {
    let available = filesystem_available_bytes(parent)?;
    let margin = compute_margin(spec.size_bytes);
    let needed = spec.size_bytes.saturating_add(margin);
    if available < needed {
        anyhow::bail!("{}", format_free_space_error(needed, parent, available));
    }
    Ok(())
}

/// Download the spec to `final_path` through a tempfile, check SHA,
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

    // Pre-flight free-space gate. Without this, a nearly-full cache
    // filesystem lets std::io::copy run until ENOSPC and surfaces a
    // generic I/O error that doesn't name "disk space" as the cause.
    // Checking here — after create_dir_all so statvfs(parent)
    // resolves, before NamedTempFile::new_in so a failed gate does
    // not leave a zero-byte tempfile behind — turns the failure into
    // an actionable bail with the available/needed byte counts in
    // the message.
    ensure_free_space(parent, spec)?;

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
    // catches DNS/TLS wedges early. The overall timeout scales with
    // `spec.size_bytes` via [`fetch_timeout_for_size`] so a 2.55 GiB
    // model does not share a single one-size-fits-all cap — the
    // previous fixed 15-minute ceiling either let a wedged download
    // hang for 15 minutes past any
    // reasonable budget or starved the model on slow CI CDNs. Tests
    // that don't actually hit the network (offline gate, cached path)
    // never enter this branch.
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(fetch_timeout_for_size(spec.size_bytes))
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
    //
    // TTY-aware progress bar: when stderr is a terminal, wrap the
    // reader with [`indicatif::ProgressBar`] so the user sees a
    // live "N/total MiB — ETA" readout during the multi-minute
    // download. indicatif auto-detects whether stderr is a
    // terminal and hides the bar (silently no-ops all draw calls)
    // when it is not — so CI captures, redirected stderr, and
    // nohup'd invocations get zero noise while interactive runs
    // get the progress UI for free. No explicit draw-target
    // override is needed; the default stderr target does the
    // right thing.
    let total_bytes = response.content_length().unwrap_or(spec.size_bytes);
    let progress = indicatif::ProgressBar::new(total_bytes);
    progress.set_style(
        indicatif::ProgressStyle::with_template(
            "  {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
        )
        .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
        .progress_chars("=>-"),
    );
    progress.set_message(spec.file_name);
    {
        use std::io::Write;
        let file = tmp.as_file_mut();
        let mut writer = std::io::BufWriter::new(file);
        let mut reader = progress.wrap_read(&mut response);
        std::io::copy(&mut reader, &mut writer)
            .with_context(|| format!("stream body from {} to {}", spec.url, tmp_path.display()))?;
        writer
            .flush()
            .with_context(|| format!("flush {} after body stream", tmp_path.display()))?;
    }
    progress.finish_and_clear();

    if !check_sha256(&tmp_path, spec.sha256_hex)? {
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
    // Seed the revalidation sidecar so the next `status()` hits the
    // warm-cache fast path instead of re-hashing the full artifact.
    // Write failure is non-fatal — the next status() simply falls
    // back to the SHA walk and tries again.
    if let Err(e) = write_mtime_size_sidecar(final_path) {
        tracing::debug!(
            artifact = %final_path.display(),
            %e,
            "mtime-size sidecar write failed post-fetch; next status() will re-hash",
        );
    }
    Ok(final_path.to_path_buf())
}

/// Canonical length of a SHA-256 digest rendered as ASCII hex:
/// 32 bytes × 2 hex chars per byte. Named constant so the length
/// gate in [`is_valid_sha256_hex`] and the matching diagnostic in
/// [`validate_sha256_hex`] share one source of truth.
const SHA256_HEX_LEN: usize = 64;

/// True iff `s` contains only ASCII hex digits (`0-9a-fA-F`).
/// Length is not checked. Shared between [`is_valid_sha256_hex`]
/// (the const-context bool predicate) and [`validate_sha256_hex`]
/// (the runtime diagnostic-producing validator); centralizing the
/// hex check in one const helper prevents drift between the two
/// surfaces.
const fn is_all_hex_ascii(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_hexdigit() {
            return false;
        }
        i += 1;
    }
    true
}

/// Canonical predicate for a well-formed SHA-256 hex pin: exactly
/// [`SHA256_HEX_LEN`] ASCII characters, each a hex digit
/// (`0-9a-fA-F`). `const fn` so module-scope compile-time asserts
/// on [`DEFAULT_MODEL`] pins fold to a
/// no-op at build time, and so [`status`] / [`ensure`] can gate on
/// it without runtime diagnostic construction (they produce
/// context-specific error messages themselves).
///
/// Runtime callers that want the "wrong length" vs "non-hex" kind
/// distinction in the error string use [`validate_sha256_hex`]
/// instead, which returns `Result<()>` with a pre-formatted
/// diagnostic. The two surfaces share [`SHA256_HEX_LEN`] and
/// [`is_all_hex_ascii`] so a change to either constraint updates
/// both call sites by construction.
const fn is_valid_sha256_hex(s: &str) -> bool {
    // `const fn` requires byte-level iteration — `.chars().all(...)`
    // depends on non-const iterator adapters. `u8::is_ascii_hexdigit`
    // has been `const fn` since Rust 1.47.
    s.len() == SHA256_HEX_LEN && is_all_hex_ascii(s)
}

/// Runtime validator for a SHA-256 hex pin that produces a
/// kind-specific diagnostic on failure. Length failure and non-hex
/// failure surface as distinct bail messages so a caller (CLI
/// readout, I/O-error wrapper, test assertion) can name the
/// underlying problem rather than defaulting to a generic "SHA
/// check failed."
///
/// Previously [`check_sha256`] open-coded the length+hex checks
/// inline to produce these two distinct diagnostics while the
/// const bool [`is_valid_sha256_hex`] sat alongside doing the
/// same check without the diagnostic — two representations of
/// the same predicate that could drift if edited independently.
/// Pushing the diagnostic into this sibling Result-returning
/// validator collapses that duplication: both surfaces now share
/// [`SHA256_HEX_LEN`] and [`is_all_hex_ascii`] and the wording
/// lives in one place.
///
/// Substrings pinned by the call-site tests
/// (`check_sha256_rejects_malformed_hex_length`,
/// `check_sha256_rejects_non_hex_chars`): `"64 chars"` for the
/// length kind and `"non-hex"` for the character kind. Any
/// rewording must preserve those substrings.
fn validate_sha256_hex(s: &str) -> Result<()> {
    if s.len() != SHA256_HEX_LEN {
        anyhow::bail!(
            "expected SHA-256 hex must be {SHA256_HEX_LEN} chars, got {} ({:?})",
            s.len(),
            s,
        );
    }
    if !is_all_hex_ascii(s) {
        anyhow::bail!("expected SHA-256 hex contains non-hex chars: {:?}", s);
    }
    Ok(())
}

/// Return `Ok(true)` when the file's SHA-256 matches the expected
/// hex pin (case-insensitive), `Ok(false)` otherwise. `Err` only on
/// I/O error reading the file or a malformed expected hex string
/// (non-64 chars / non-hex chars), which would render the check
/// itself useless and must surface instead of silently pretending
/// the file is good.
fn check_sha256(path: &std::path::Path, expected_hex: &str) -> Result<bool> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    // Delegate shape validation to `validate_sha256_hex` so the
    // length-vs-hex diagnostic lives in one place. Previously this
    // function open-coded the same length+hex check inline,
    // duplicating what the const bool `is_valid_sha256_hex`
    // expressed without diagnostics.
    validate_sha256_hex(expected_hex)?;

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
///
/// Scheme match is case-SENSITIVE: only the exact lowercase
/// `"https://"` prefix passes. Uppercase (`"HTTPS://"`) or mixed
/// case (`"Https://"`) variants are rejected alongside `http://`
/// and every other scheme. RFC 3986 §3.1 declares URL schemes
/// case-insensitive, so in principle this is stricter than the
/// spec — but every pin in this crate ([`DEFAULT_MODEL`] and the
/// fixtures in the nearby tests) uses lowercase, the compile-time
/// `is_valid_sha256_hex` guards
/// do not reach scheme validation, and a mixed-case scheme in a
/// `ModelSpec::url` field is almost certainly a typo worth failing
/// closed on rather than silently normalizing. The
/// `reject_insecure_url_rejects_non_https_schemes` test pins the
/// strict behavior against `HTTPS://`.
fn reject_insecure_url(url: &str) -> Result<()> {
    if !url.starts_with("https://") {
        anyhow::bail!("model cache fetcher refuses non-HTTPS URL: {}", url,);
    }
    Ok(())
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
/// Shape: `{TEMPLATE}\n\n{hint_line}STDOUT:\n{sanitized_output}` —
/// the hint is appended as its own line before the STDOUT block so
/// the model sees the user-declared focus before the raw content. An
/// empty or absent hint degrades to the bare template without
/// leaving a dangling "Focus:" header. A hint that reduces to empty
/// or whitespace-only after ChatML sanitization is treated the same
/// way.
///
/// Both the `output` body and the `hint` pass through
/// [`strip_chatml_control_tokens`] before they are embedded.
/// [`wrap_chatml_no_think`] later wraps the composed prompt in Qwen3
/// ChatML (`<|im_start|>user\n…<|im_end|>`); any literal
/// `<|im_start|>`, `<|im_end|>`, or `<|im_sep|>` substring inside the
/// user turn would tokenize as a real ChatML control token and close
/// or reopen turn boundaries from inside that turn. Stripping those
/// three substrings from both the body and the hint is the gate that
/// preserves the wrapper's turn framing. The template is a
/// module-level `const` so it is not re-scanned. The hint originates
/// from a `&'static str` on the payload's [`OutputFormat::LlmExtract`]
/// variant (compile-time source text, inside the trust boundary), so
/// the scrub is defense-in-depth against a future API change that
/// could route caller-supplied strings into the hint; it is not a
/// response to a current exploit path.
pub(crate) fn compose_prompt(output: &str, hint: Option<&str>) -> String {
    let safe_output = strip_chatml_control_tokens(output);
    let safe_hint = hint
        .map(|h| h.trim())
        .map(strip_chatml_control_tokens)
        .filter(|h| !h.trim().is_empty());
    let mut out = String::with_capacity(
        LLM_EXTRACT_PROMPT_TEMPLATE.len()
            + safe_output.len()
            + 64
            + safe_hint.as_deref().map_or(0, |h| h.len() + 16),
    );
    out.push_str(LLM_EXTRACT_PROMPT_TEMPLATE);
    out.push_str("\n\n");
    if let Some(h) = safe_hint.as_deref() {
        out.push_str("Focus: ");
        out.push_str(h);
        out.push_str("\n\n");
    }
    // Label frozen as "STDOUT:" for LLM prompt compatibility even
    // when `output` is stderr-sourced via the fallback contract —
    // renaming would re-key the model's prompt/response pattern.
    out.push_str("STDOUT:\n");
    out.push_str(&safe_output);
    out
}

/// Remove the literal ChatML control token strings `<|im_start|>`,
/// `<|im_end|>`, and `<|im_sep|>` from `s`. Matching is byte-exact:
/// case variants (`<|IM_END|>`), whitespace-padded variants
/// (`< |im_end| >`), and shape variants (missing punctuation, unknown
/// token names, attribute-style tokens) are left alone. The Qwen3
/// tokenizer encodes these three exact strings as single ChatML
/// control-token ids that close or reopen the assistant/user turn
/// boundaries [`wrap_chatml_no_think`] establishes; other shapes
/// tokenize as ordinary text and do not produce control-token ids,
/// so the byte-exact match covers the three ChatML turn-framing
/// tokens (which is what this sanitization is responsible for)
/// without over-stripping benchmark output that happens to echo
/// ChatML-looking bytes. Other prompt-injection vectors (semantic
/// manipulation via visible text, model-specific special tokens
/// outside the ChatML turn-framing set) are out of scope for this
/// helper.
///
/// Iterates to a fixed point: a single pass through `TOKENS` can
/// produce a fresh control token via splice-recombination. For
/// example, input `<|im_<|im_start|>start|>` strips the inner
/// `<|im_start|>` on the first pass, leaving the outer prefix +
/// suffix abutted as a fresh `<|im_start|>`. Looping until no token
/// remains forecloses that attack class. Termination is bounded:
/// every iteration that does not reach the `break` strips at least
/// one token (≥ 10 bytes — the shortest token is `<|im_end|>`), so
/// the loop runs at most `s.len() / 10` times.
///
/// When none of the three substrings appear in `s`, the input is
/// returned as a borrowed `Cow::Borrowed` so the common path
/// (benchmark stdout almost never contains these tokens) skips the
/// `String` allocation that the loop body would otherwise force.
fn strip_chatml_control_tokens(s: &str) -> std::borrow::Cow<'_, str> {
    const TOKENS: [&str; 3] = ["<|im_start|>", "<|im_end|>", "<|im_sep|>"];
    if !TOKENS.iter().any(|t| s.contains(t)) {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = s.to_string();
    loop {
        let mut changed = false;
        for token in TOKENS {
            if out.contains(token) {
                out = out.replace(token, "");
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    std::borrow::Cow::Owned(out)
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

/// Context window passed to `LlamaContextParams::with_n_ctx`. Sized
/// at 2048 because the prompt template is short (~120 tokens) and
/// every benchmark this pipeline targets emits at most a few hundred
/// metric leaves; the body almost always fits in the remaining
/// budget after [`SAMPLE_LEN`] is reserved for generation. A larger
/// context would cost KV memory linearly without adding headroom for
/// the realistic input shape.
///
/// Promoted to a module-level `const` so the prompt-budget
/// arithmetic in `invoke_with_model` and the
/// `n_ctx_budget_*` test fixtures share one source of truth.
const N_CTX_TOKENS: usize = 2048;

/// Per-invocation token budget for the prompt — the prompt's
/// `str_to_token` output must not exceed this count, or `ctx.decode`
/// would either reject the batch (`NTokensZero` / `NoKvCacheSlot`)
/// or silently truncate the KV cache, producing degenerate output.
/// The budget reserves [`SAMPLE_LEN`] tokens for generation plus a
/// 64-token cushion for the ChatML wrapper Qwen3 layers around the
/// composed prompt (`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`
/// at ~12-16 tokens, with margin for tokenizer drift across model
/// variants).
///
/// `invoke_with_model` enforces this budget post-tokenization: if
/// the prompt's token count exceeds it, the body is byte-truncated
/// (snapped to a UTF-8 boundary) and re-tokenized. Byte truncation
/// is approximate — Qwen3-4B's BBPE tokenizer averages ~3.5 chars /
/// token on English benchmark text — so we use a 3:1 chars-per-token
/// floor to size the byte budget conservatively, then verify with a
/// second tokenization pass that the truncated prompt fits.
const MAX_PROMPT_TOKENS: usize = N_CTX_TOKENS - SAMPLE_LEN - 64;

/// Approximate bytes-per-token floor for the Qwen3 BBPE tokenizer on
/// English text. Used by the byte-truncation pre-pass that bounds
/// prompt body size before re-tokenization. Conservative — real
/// English text averages ~3.5-4 chars/token, so a 3:1 ratio under-
/// estimates token count and over-truncates body bytes when in
/// doubt. The verification pass that follows re-tokenization
/// catches any case where this floor was still optimistic.
const BYTES_PER_TOKEN_FLOOR: usize = 3;

/// Truncate `prompt` so its tokenization fits inside
/// [`MAX_PROMPT_TOKENS`]. Returns the (possibly truncated) prompt
/// alongside an indicator flagging whether truncation occurred so
/// the caller can `tracing::warn!` and the test fixture can pin
/// the truncation behavior.
///
/// Strategy: tokenize the full prompt first. If the result fits,
/// return it as-is. Otherwise, byte-truncate the prompt to a
/// conservative budget computed from
/// [`BYTES_PER_TOKEN_FLOOR`] × [`MAX_PROMPT_TOKENS`], snap to a
/// UTF-8 char boundary so we never split a multi-byte codepoint,
/// re-tokenize, and pin a final assertion that the result is now
/// within budget. The conservative ratio means a single retry pass
/// is sufficient for English benchmark output; pathological inputs
/// (e.g. long runs of single-byte tokens like raw whitespace
/// emoji) would need a second retry, but those don't exist in any
/// realistic benchmark stdout this pipeline targets.
///
/// On the (theoretical) failure of the second tokenization to fit,
/// returns an error rather than silently shipping an oversize
/// prompt — the caller wraps that into the
/// [`InferenceError::Tokenize`] failure surface. Failing closed
/// here keeps the inference path's "ctx.decode either succeeds or
/// produces an actionable error" contract intact.
fn fit_prompt_to_context(
    model: &llama_cpp_2::model::LlamaModel,
    prompt: &str,
) -> Result<Vec<llama_cpp_2::token::LlamaToken>, InferenceError> {
    use llama_cpp_2::model::AddBos;

    // First-pass tokenization: most inputs fit and short-circuit
    // here without any allocation past the token vec.
    let initial = model
        .str_to_token(prompt, AddBos::Never)
        .map_err(|source| InferenceError::Tokenize {
            prompt_excerpt: prompt_excerpt(prompt),
            source,
        })?;
    if initial.len() <= MAX_PROMPT_TOKENS {
        return Ok(initial);
    }

    // Over budget. Byte-truncate to a conservative budget computed
    // from the chars-per-token floor, snapping back to a UTF-8 char
    // boundary so we never produce an invalid-UTF-8 fragment.
    let byte_budget = MAX_PROMPT_TOKENS.saturating_mul(BYTES_PER_TOKEN_FLOOR);
    let mut end = byte_budget.min(prompt.len());
    while end > 0 && !prompt.is_char_boundary(end) {
        end -= 1;
    }
    let truncated = &prompt[..end];
    let retokenized = model
        .str_to_token(truncated, AddBos::Never)
        .map_err(|source| InferenceError::Tokenize {
            prompt_excerpt: prompt_excerpt(truncated),
            source,
        })?;

    if retokenized.len() > MAX_PROMPT_TOKENS {
        // Pathological shape — the BPE tokenizer ran below the
        // chars-per-token floor for this input. Surface as a
        // typed error rather than slicing further; the operator
        // can re-tune `BYTES_PER_TOKEN_FLOOR` if a real workload
        // hits this.
        return Err(InferenceError::Generation {
            reason: format!(
                "prompt token count {} still exceeds budget {} after \
                 byte-truncation to {} bytes — tokenizer ran below the \
                 {} chars-per-token floor; tune BYTES_PER_TOKEN_FLOOR",
                retokenized.len(),
                MAX_PROMPT_TOKENS,
                end,
                BYTES_PER_TOKEN_FLOOR,
            ),
        });
    }

    tracing::warn!(
        original_tokens = initial.len(),
        truncated_tokens = retokenized.len(),
        max_prompt_tokens = MAX_PROMPT_TOKENS,
        truncated_bytes = prompt.len() - end,
        "LlmExtract prompt exceeded context budget; truncated body to fit",
    );
    Ok(retokenized)
}

/// Loaded inference state: the GGUF-backed `LlamaModel`. The model
/// owns its tokenizer + EOS metadata internally — no separate
/// tokenizer handle is needed. `LlamaContext` is intentionally NOT
/// stored here: it borrows from `&LlamaModel` (`new_context<'a>(&'a
/// self, ...)`), so caching one alongside the model would create a
/// self-referential struct. `invoke_with_model` builds a fresh
/// context per call instead, which also gives every invocation a
/// clean KV state without an explicit `clear_kv_cache` step.
///
/// Threaded through `load_inference` and `invoke_with_model` — both
/// module-private. Nothing outside `model.rs` constructs or observes
/// this type.
struct LoadedInference {
    model: llama_cpp_2::model::LlamaModel,
}

/// Load the bundled Qwen3 weights via `llama-cpp-2`.
///
/// Resolves the cached model via [`ensure`] so first use triggers a
/// SHA check; subsequent in-process calls hit the memoized
/// [`MODEL_CACHE`] slot below and never re-enter this function.
///
/// Production callers reach this only through [`memoized_inference`];
/// [`MODEL_CACHE`] caches the returned `Result` (Ok or Err), so this
/// body runs at most once per process. The `cfg(test)`-only `reset`
/// hook is the sole way to clear the slot and re-enter this function.
///
/// Errors surface through [`InferenceError`]: cache-resolution
/// failures bubble out of `ensure()` as anyhow chains, while engine-
/// level load failures wrap into [`InferenceError::ModelLoad`]
/// carrying the resolved `PathBuf` plus the upstream
/// `LlamaModelLoadError` source.
fn load_inference() -> anyhow::Result<LoadedInference> {
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_2::model::params::LlamaModelParams;

    let model_path = ensure(&DEFAULT_MODEL)?;

    // CPU-only: no GPU layer offload. The
    // process-wide `BACKEND` is a `OnceLock<LlamaBackend>` that
    // initializes lazily on first call here; subsequent calls in
    // the same process reuse the same handle.
    let model =
        LlamaModel::load_from_file(global_backend(), &model_path, &LlamaModelParams::default())
            .map_err(|source| InferenceError::ModelLoad {
                path: model_path.clone(),
                source,
            })?;

    Ok(LoadedInference { model })
}

/// Wrap a raw user prompt in Qwen3 ChatML with the `/no_think`
/// directive appended.
///
/// The `/no_think` directive at the end of the user turn switches the
/// model out of thinking mode per the Qwen3 model card: the assistant
/// skips the `<think>…</think>` block and emits the final answer
/// directly, keeping the [`SAMPLE_LEN`] token budget available for the
/// JSON response rather than burning it on a reasoning trace the
/// downstream walker would discard. The post-decode
/// [`strip_think_block`] in [`invoke_with_model`] remains as a belt-
/// and-suspenders defense because the directive is a soft switch and
/// the model can still emit an empty `<think></think>` shell.
///
/// Returned shape is exactly:
/// `<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n`.
/// A single ASCII space separates the prompt from the directive; the
/// closing `<|im_end|>` sits on the same line as the directive and the
/// assistant turn opens on the next line with no content so the model
/// begins generation at byte 0 of its own turn.
fn wrap_chatml_no_think(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n")
}

/// Resolve the per-context thread count for llama-cpp inference.
///
/// Falls back to 4 when conversion to `i32` fails or
/// `available_parallelism` returned `None`, and clamps to 16.
/// OpenMP matmul scales sub-linearly past ~16 threads on the
/// quantized model used here.
fn inference_thread_count(available: Option<std::num::NonZero<usize>>) -> i32 {
    available
        .and_then(|p| i32::try_from(p.get()).ok())
        .unwrap_or(4)
        .min(16)
}

/// Run one greedy generation pass against the already-loaded model
/// and return the decoded assistant text with any `<think>…</think>`
/// block stripped.
///
/// Idempotent: a fresh `LlamaContext` is built per call from the
/// cached `&LlamaModel`, so each invocation starts with an empty
/// KV cache. Greedy: `LlamaSampler::greedy()` selects the ArgMax
/// token — output is a deterministic function of prompt + weights.
fn invoke_with_model(state: &mut LoadedInference, prompt: &str) -> anyhow::Result<String> {
    use std::num::NonZeroU32;

    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::sampling::LlamaSampler;

    // Context window: [`N_CTX_TOKENS`] = 2048 tokens. Prompt +
    // generation must fit within this; `fit_prompt_to_context`
    // below enforces the prompt budget post-tokenization, so any
    // oversize body is truncated before `ctx.decode` would reject
    // it. The remaining budget after [`SAMPLE_LEN`] reservation is
    // [`MAX_PROMPT_TOKENS`].
    //
    // Threading: `LlamaContextParams::default()` caps both
    // `n_threads` and `n_threads_batch` at 4 (see upstream
    // `llama-cpp-2-0.1.145/src/context/params/get_set.rs:154` and
    // `:184`). Inference is matmul-bound for every quantized layer
    // pass, so on any host with more than 4 cores the default
    // strands the matmul on a fraction of the box — the prompt
    // pre-fill and per-token generation both stretch from
    // milliseconds to seconds. Pull the actual core count from
    // `std::thread::available_parallelism` (which reads
    // `sched_getaffinity` for the current thread; honors cgroup
    // cpuset when the harness propagates affinity into the test
    // process — a constrained worker on a 64-core Threadripper that
    // has been pinned to 8 cores reads 8 here, matching the
    // workload's actual budget). The OpenMP build path (the
    // `openmp` feature on `llama-cpp-2`) further parallelizes
    // matmul across the threads we hand it.
    //
    // `available_parallelism` returns `Result` only because the
    // syscall can fail under unusual containerization shapes
    // (no /proc, mountns drop, etc.). Falling back to the static
    // default of 4 on that path is safe — the run will be slow but
    // correct, and the fallback only fires under environments where
    // the operator has already chosen to constrain visibility.
    // `i32::try_from` cannot fail in practice (modern hosts top out
    // around 256 cores; `i32::MAX = 2^31 - 1`), but the fallible
    // form keeps the conversion explicit.
    let n_threads: i32 = inference_thread_count(std::thread::available_parallelism().ok());
    // Cap at 16: matmul throughput plateaus past ~16 threads on
    // cross-NUMA hosts due to memory bandwidth saturation and
    // OpenMP synchronization overhead.
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX_TOKENS as u32))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads);
    let mut ctx = state
        .model
        .new_context(global_backend(), ctx_params)
        .map_err(|source| InferenceError::ContextCreate { source })?;

    let chat_prompt = wrap_chatml_no_think(prompt);
    // ChatML control tokens (`<|im_start|>`, `<|im_end|>`) carry the
    // turn structure — [`fit_prompt_to_context`] tokenizes with
    // `AddBos::Never` because the prompt template already opens with
    // `<|im_start|>user`. A leading BOS would shift attention
    // positions and mis-align the model's expected ChatML turn
    // structure. The helper enforces the [`MAX_PROMPT_TOKENS`]
    // budget and byte-truncates the body if a pathologically long
    // benchmark output would otherwise overflow the context window.
    let prompt_tokens = fit_prompt_to_context(&state.model, &chat_prompt)?;

    // Prompt batch sized to fit `N_CTX_TOKENS` — any prompt that
    // fits the context after `fit_prompt_to_context` truncation fits
    // the batch.
    let mut batch = LlamaBatch::new(N_CTX_TOKENS, 1);

    let last_index: i32 = (prompt_tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(prompt_tokens.iter().copied()) {
        // logits=true only for the last prompt token — we sample
        // from the position immediately after the prompt, and
        // requesting logits on every prompt token would burn memory
        // bandwidth on output we don't read.
        let is_last = i == last_index;
        batch
            .add(token, i, &[0], is_last)
            .map_err(|e| InferenceError::Generation {
                reason: format!("seed prompt batch at position {i}: {e}"),
            })?;
    }
    ctx.decode(&mut batch)
        .map_err(|source| InferenceError::Decode { source })?;

    let mut sampler = LlamaSampler::greedy();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut decoded = String::new();

    // Each generation step writes the just-sampled token at the
    // absolute position immediately after the prompt — `prompt_len`,
    // `prompt_len + 1`, `prompt_len + 2`, … — so the KV slot for
    // the new token doesn't alias the prompt. `prompt_len` is the
    // batch's `n_tokens` after the prompt-pass `decode`; iterating
    // from there with a `zip` ties the position counter directly to
    // the loop iterator without a separate `n_cur += 1` step.
    let prompt_len = batch.n_tokens();
    let mut hit_eos = false;
    for (n_cur, _) in (prompt_len..).zip(0..SAMPLE_LEN) {
        // Sample from the latest decoded position — `batch.n_tokens()
        // - 1` is the index of the most recently decoded slot, and
        // its logits are what `greedy()` selects from.
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if state.model.is_eog_token(token) {
            hit_eos = true;
            break;
        }

        // Stateful UTF-8 decode: a single token may end mid-codepoint,
        // so the decoder buffers partial bytes between calls. Append
        // each piece to `decoded` as the bytes resolve.
        let piece = state
            .model
            .token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| InferenceError::Generation {
                reason: format!("token_to_piece for token at position {n_cur}: {e}"),
            })?;
        decoded.push_str(&piece);

        // Feed the just-sampled token back as the next batch input.
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| InferenceError::Generation {
                reason: format!("seed generation batch at position {n_cur}: {e}"),
            })?;
        ctx.decode(&mut batch)
            .map_err(|source| InferenceError::Decode { source })?;
    }

    if !hit_eos {
        tracing::warn!(
            "generation hit {} token cap without EOS — output may be truncated",
            SAMPLE_LEN,
        );
    }

    Ok(strip_think_block(&decoded))
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
///
/// Matching is byte-exact: only literal `<think>` and `</think>`
/// are recognized. Case variants, self-closing tags, attribute-
/// carrying tags, and whitespace-injected tags pass through
/// verbatim.
fn strip_think_block(s: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if !s.contains(OPEN) {
        return s.to_string();
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
    out
}

/// Return the memoized [`MODEL_CACHE`] entry, populating it under the
/// outer mutex on the first call.
///
/// Returns `Arc<CachedInference>` so the caller releases the outer
/// mutex before running inference: the inner `Mutex<LoadedInference>`
/// is held for the full generation pass and another thread initiating
/// `extract_via_llm` should be free to observe the populated slot
/// without waiting on the inference in flight. Cloning the `Arc` is
/// cheap (one atomic increment); the only synchronization on the
/// outer mutex is the lock + (slot read or load + store) + unlock.
fn memoized_inference() -> Arc<CachedInference> {
    let mut guard = MODEL_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(arc) = guard.as_ref() {
        return Arc::clone(arc);
    }
    #[cfg(test)]
    // Relaxed is sufficient: the test helper `lock_env` serializes
    // every test that reads MODEL_CACHE_LOAD_COUNT against every
    // test that drives the slow path here, and that mutex
    // provides the real happens-before edge between the increment
    // and the read. The MODEL_CACHE lock covers the write side
    // only — counter reads in
    // `model_cache_loads_at_most_once_per_populated_slot` happen
    // outside MODEL_CACHE, so the MODEL_CACHE lock is not a
    // read-side gate.
    MODEL_CACHE_LOAD_COUNT.fetch_add(1, Ordering::Relaxed);
    let result = load_inference()
        .map(Mutex::new)
        .map_err(|e| format!("{e:#}"));
    let arc = Arc::new(result);
    *guard = Some(Arc::clone(&arc));
    arc
}

/// Clear [`MODEL_CACHE`] so the next [`extract_via_llm`] /
/// [`load_inference`] call re-runs the load path end-to-end
/// (including [`ensure`]'s offline-gate check).
///
/// # When to call
///
/// Tests that mutate `KTSTR_MODEL_OFFLINE` or `KTSTR_CACHE_DIR` and
/// then assert offline-gate / load-failure behavior. Without this
/// reset, an `Ok(_)` slot left by an earlier successful load — in any
/// test or any prior call within the same process — would short-
/// circuit `extract_via_llm` and return cached inference state
/// without ever consulting `ensure()`, silently bypassing the gate.
///
/// # Locking order
///
/// Callers must hold
/// [`crate::test_support::test_helpers::lock_env`] across both the
/// reset and any subsequent env-var mutations + `extract_via_llm`
/// calls that depend on the freshly cleared slot. The lock keeps the
/// reset, the env mutation, and the next initialization atomic with
/// respect to other env-touching tests in this process.
///
/// # cfg(test)-only
///
/// Production code never resets the cache: the memoized state is
/// load-once-per-process by design. The reset hook is a test-only
/// affordance for re-exercising the load path.
#[cfg(test)]
pub(crate) fn reset() {
    MODEL_CACHE_LOAD_COUNT.store(0, Ordering::Relaxed);
    let mut guard = MODEL_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    *guard = None;
}

/// Run the full LlmExtract pipeline against `output` and return the
/// resulting metrics, all pre-tagged with
/// [`MetricSource::LlmExtract`](super::MetricSource::LlmExtract).
///
/// A single inference call is made. An infra error or a
/// JSON-parse failure of the model's response returns an empty
/// metric set — matching the [`extract_metrics`] contract that
/// extraction errors are non-fatal and the downstream
/// [`MetricCheck`](crate::test_support::MetricCheck) evaluation reports
/// each referenced metric as missing.
///
/// No retry: under `LlamaSampler::greedy()` (deterministic ArgMax,
/// no RNG state), a second inference call on the same prompt +
/// weights produces byte-identical output. Retrying would only burn
/// wall time without
/// changing the result.
/// Return signature — `Result<Vec<Metric>, String>` — distinguishes
/// three outcomes:
///
/// - `Ok(metrics)` — inference ran, JSON parsed; `metrics` may be
///   empty if the model emitted a valid-JSON-but-no-numeric-leaves
///   response (documented contract).
/// - `Ok(Vec::new())` — model inference ran but response was not
///   parseable JSON, or inference itself failed mid-forward-pass.
///   These paths are non-fatal (documented "empty metric set on
///   inference hiccup") and the caller keeps going.
/// - `Err(reason)` — the **model cache load** failed. This is a
///   setup-level problem (missing weights, bad SHA, corrupt GGUF).
///   The `MetricCheck` evaluator translates the reason into a
///   `DetailKind::Other` entry on the `AssertResult` so the user
///   sees "LlmExtract model load failed: <reason>" instead of an
///   opaque "metric 'foo' not found" when the real failure was
///   that the model never loaded.
///
/// # MODEL_CACHE memoization + panic-retry invariant (external harness authors)
///
/// Each caller routes through [`memoized_inference`]: the first call
/// runs `load_inference` once under the global [`MODEL_CACHE`] mutex
/// and stores the `Result` (success OR error); every subsequent call
/// observes the cached value with no re-load. Three outcomes follow
/// from this shape:
///
/// 1. **`Ok(cache)` — cached forever**. The loaded model stays in
///    memory for the process lifetime; the ~2.55 GiB slot is never
///    evicted. Subsequent `extract_via_llm` calls reuse the same
///    inference state.
/// 2. **`Err(reason)` — cached forever**. A load failure is cached
///    identically to a successful load: every subsequent call returns
///    the same `Err(reason)` string without re-attempting. This is
///    the "fail-closed-forever" contract documented on
///    [`MODEL_CACHE`] — there is no public `clear_model_cache()` hook,
///    so external harnesses must treat a first `Err` as terminal for
///    the process lifetime.
/// 3. **Panic mid-load — NOT cached, but process-terminal in release**.
///    A panic inside `load_inference` does NOT populate the cache slot:
///    - Under the test/debug profile (`panic = "unwind"`) the mutex
///      unwinds, the slot stays `None`, and the next caller retries
///      `load_inference` from scratch. Panic-retry is observable.
///    - Under the release profile (`panic = "abort"`, see
///      `Cargo.toml [profile.release]`) the process aborts before
///      control returns to the caller. Retry is process-terminal
///      rather than next-call-observable — there is no "next call."
///
/// External-harness checklist: (a) a first `Err(reason)` is
/// terminal for the process lifetime; (b) a panic during load
/// aborts the process under release builds, so plan for no
/// re-entry on that path.
pub(crate) fn extract_via_llm(
    output: &str,
    hint: Option<&str>,
    stream: super::MetricStream,
) -> Result<Vec<super::Metric>, String> {
    let prompt = compose_prompt(output, hint);

    // `memoized_inference` serializes concurrent first-call races on
    // the outer mutex: every caller observes the same stored value,
    // and exactly one caller's closure runs end-to-end. A failed load
    // is memoized as `Err` so subsequent calls return the same
    // reason string without repeating the 2.55 GiB load.
    let cached = memoized_inference();
    let cache = match cached.as_ref() {
        Ok(c) => c,
        Err(msg) => {
            tracing::warn!(%msg, "LlmExtract model load failed (cached)");
            return Err(msg.clone());
        }
    };
    let mut state = cache.lock().unwrap_or_else(|e| e.into_inner());

    let response = match invoke_with_model(&mut state, &prompt) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(err = %format!("{e:#}"), "LlmExtract inference failed");
            return Ok(Vec::new());
        }
    };
    // Opt-in raw-response tracing: off by default (see
    // `LLM_DEBUG_RESPONSES_ENV` doc). A non-empty env value routes
    // the full model output through `tracing::debug!` so users
    // debugging a "response was not parseable JSON" warn can see
    // exactly what the model emitted without patching the source.
    if env_value_is_opt_in(std::env::var(LLM_DEBUG_RESPONSES_ENV).ok().as_deref()) {
        tracing::debug!(
            response_bytes = response.len(),
            response = %response,
            "LlmExtract raw response (debug env enabled)",
        );
    }
    Ok(parse_llm_response(&response, stream))
}

/// Parse a model-emitted response into the Metric list for the
/// `LlmExtract` pipeline. Returns an empty vector when the response
/// contains no JSON region the
/// [`find_and_parse_json`](super::metrics::find_and_parse_json)
/// recovery walker can lift out — a non-JSON response is a recoverable
/// "no metrics this time" outcome, not an error, because LLM output
/// is inherently stochastic and a single failed inference should not
/// fail the whole test run.
///
/// Extracted from [`extract_via_llm`] so the response-to-metrics step
/// is unit-testable without standing up the model backend: the caller
/// injects any response string it likes and asserts on the result.
/// `extract_via_llm` owns the model load and the `invoke_with_model`
/// round-trip; this helper owns the parse contract alone.
fn parse_llm_response(response: &str, stream: super::MetricStream) -> Vec<super::Metric> {
    match super::metrics::find_and_parse_json(response) {
        Some(json) => {
            super::metrics::walk_json_leaves(&json, super::MetricSource::LlmExtract, stream)
        }
        None => {
            // Intentionally log only `response.len()` (byte count), not
            // the body. The response can run up to SAMPLE_LEN tokens —
            // multi-KB chat output with leaked `<think>` traces under
            // pathological inputs — and dumping that into the tracing
            // subscriber floods CI logs while leaking prompt-dependent
            // content. The byte count plus the emitted event is enough
            // to diagnose "empty response" vs "large response missing
            // JSON region" without the payload itself.
            tracing::warn!(
                response_bytes = response.len(),
                "LlmExtract response was not parseable JSON; returning empty metric set",
            );
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests_cache;
#[cfg(test)]
mod tests_fetch;
#[cfg(test)]
mod tests_inference;
#[cfg(test)]
mod tests_prompt;
