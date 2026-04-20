//! Local LLM model cache for [`OutputFormat::LlmExtract`] payloads.
//!
//! The frozen #162 design routes `OutputFormat::LlmExtract` stdout
//! through a small local model that emits JSON, which the existing
//! [`walk_json_leaves`](crate::test_support::metrics) pipeline then
//! consumes. The model binary itself lives under
//! `~/.cache/ktstr/models/`; this module owns locate + fetch +
//! verify semantics without loading or invoking the model (that
//! belongs to the forthcoming model-invocation module).
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
    if let Ok(v) = std::env::var(OFFLINE_ENV)
        && !v.is_empty()
    {
        anyhow::bail!(
            "{OFFLINE_ENV}={v} set but model '{}' is not cached at {}; \
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
    let tmp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create tempfile in {}", parent.display()))?;
    let tmp_path = tmp.path().to_path_buf();

    let response = reqwest::blocking::get(spec.url)
        .with_context(|| format!("GET {} (download model '{}')", spec.url, spec.file_name))?;
    if !response.status().is_success() {
        anyhow::bail!(
            "GET {} returned HTTP {} — download of model '{}' failed",
            spec.url,
            response.status(),
            spec.file_name,
        );
    }
    let bytes = response
        .bytes()
        .with_context(|| format!("read body from {}", spec.url))?;
    std::fs::write(&tmp_path, &bytes)
        .with_context(|| format!("write downloaded model to {}", tmp_path.display()))?;

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
    if let Ok(v) = std::env::var(OFFLINE_ENV)
        && !v.is_empty()
    {
        eprintln!("ktstr: {OFFLINE_ENV}={v} set; skipping eager model prefetch");
        return Ok(None);
    }
    ensure(&DEFAULT_MODEL).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_cache_root_honors_ktstr_cache_dir() {
        // SAFETY: test-only, single-threaded env mutation; tests
        // under #[cfg(test)] run serially within this module.
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
        let prev_offline = std::env::var(OFFLINE_ENV).ok();
        let prev_cache = std::env::var("KTSTR_CACHE_DIR").ok();
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: test-only, single-threaded env mutation.
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
}
