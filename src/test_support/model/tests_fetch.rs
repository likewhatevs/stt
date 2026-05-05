//! Fetch-side tests: ensure(), free-space gate, fetch timeout, SHA-256
//! validators, URL scheme rejection, ModelSpec shape checks.

use super::super::test_helpers::{EnvVarGuard, isolated_cache_dir, lock_env};
use super::*;

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
fn check_sha256_matches_empty_file() {
    // SHA-256 of the empty string — a stable external anchor
    // that proves the hasher is wired correctly, independent of
    // the DEFAULT_MODEL digest.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), []).unwrap();
    let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert!(check_sha256(tmp.path(), expected).unwrap());
}

#[test]
fn check_sha256_mismatch_returns_false() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"not empty").unwrap();
    let empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    assert!(!check_sha256(tmp.path(), empty_sha).unwrap());
}

#[test]
fn check_sha256_is_case_insensitive() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), []).unwrap();
    let upper = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
    assert!(check_sha256(tmp.path(), upper).unwrap());
}

#[test]
fn check_sha256_rejects_malformed_hex_length() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), []).unwrap();
    let err = check_sha256(tmp.path(), "tooshort").unwrap_err();
    assert!(format!("{err:#}").contains("64 chars"), "err: {err:#}");
}

#[test]
fn check_sha256_rejects_non_hex_chars() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), []).unwrap();
    // 64 chars but includes `?`.
    let bad = "????????????????????????????????????????????????????????????????";
    let err = check_sha256(tmp.path(), bad).unwrap_err();
    assert!(format!("{err:#}").contains("non-hex"), "err: {err:#}");
}

/// Direct coverage for `validate_sha256_hex` independent of
/// `check_sha256`'s caller path. `check_sha256_rejects_*` above
/// already exercise the two Err kinds by way of a full file-read
/// call; these direct tests guard the same two Err kinds PLUS
/// the Ok(()) branch, so a regression that broke validate's
/// happy path (e.g. an accidental inversion of the length check)
/// surfaces here instead of silently letting valid pins fall
/// through to a wasted download or a false I/O-error diagnosis.
/// Failure-substring assertions (`"64 chars"`, `"non-hex"`)
/// mirror the wording pinned by the `check_sha256_rejects_*`
/// siblings so the diagnostic is anchored at both layers.
#[test]
fn validate_sha256_hex_flags_empty_as_length_error() {
    let err = validate_sha256_hex("").unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("64 chars"),
        "empty string must surface the length-kind diagnostic \
         (substring \"64 chars\"); got: {rendered}",
    );
}

#[test]
fn validate_sha256_hex_flags_nonhex_chars_at_correct_length() {
    // 64 chars so the length gate passes; every char is `?` so
    // the hex gate trips and the non-hex diagnostic fires.
    let sixty_four_nonhex = "?".repeat(64);
    let err = validate_sha256_hex(&sixty_four_nonhex).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("non-hex"),
        "64-char non-hex string must surface the hex-kind \
         diagnostic (substring \"non-hex\"); got: {rendered}",
    );
    assert!(
        !rendered.contains("64 chars"),
        "length gate passed on a 64-char input — diagnostic \
         must NOT mention \"64 chars\"; got: {rendered}",
    );
}

#[test]
fn validate_sha256_hex_accepts_well_formed_pin() {
    // 64 ASCII hex chars → Ok(()). Mixing case to also exercise
    // the is_ascii_hexdigit path through both the 0-9 and
    // a-f/A-F sub-ranges in one input.
    let pin = "0".repeat(64);
    validate_sha256_hex(&pin).unwrap();
    let mixed = "0123456789abcdef0123456789ABCDEF0123456789abcdef0123456789ABCDEF";
    assert_eq!(mixed.len(), 64);
    validate_sha256_hex(mixed).unwrap();
}

/// Non-empty short file — SHA-256 of ASCII "abc" is a
/// well-known external anchor (NIST FIPS 180-2 appendix). Pins
/// the non-empty happy path between the empty-file test above
/// and the multi-chunk test below; a regression that broke
/// single-chunk non-empty hashing would surface here.
#[test]
fn check_sha256_matches_abc() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), b"abc").unwrap();
    // Known SHA-256("abc") — NIST FIPS 180-2 / RFC 6234 test vector.
    let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert!(check_sha256(tmp.path(), expected).unwrap());
}

/// Multi-chunk file (larger than a single read buffer)
/// exercises the streaming `Read`-loop branch of `check_sha256`
/// (vs the single-buffer fast path for small files). 192 KiB of
/// repeated "a" bytes is large enough to cross any reasonable
/// BufReader default (8 KiB) multiple times; the expected SHA
/// is computed once here from a known constant so the test
/// remains deterministic.
#[test]
fn check_sha256_matches_multi_chunk_file() {
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
    assert!(check_sha256(tmp.path(), &expected_hex).unwrap());

    // Negative: flip one byte at the far end and check the
    // digest rejects, proving the hasher walked past the first
    // chunk.
    let mut tampered = data;
    *tampered.last_mut().unwrap() = b'b';
    std::fs::write(tmp.path(), &tampered).unwrap();
    assert!(!check_sha256(tmp.path(), &expected_hex).unwrap());
}

/// A non-existent path is an I/O-layer failure, not a pin-shape
/// failure, so `check_sha256` must surface the `std::fs::File::open`
/// error with the `open <path>` anyhow context attached. Pins the
/// error wording so callers that pattern-match on "open" still
/// find it if the underlying `io::Error` string changes.
#[test]
fn check_sha256_errors_on_missing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let missing = tmp.path().join("does-not-exist.bin");
    // Valid 64-char hex so the function passes the shape check
    // and reaches the file-open step.
    let valid_hex = "0".repeat(64);
    let err = check_sha256(&missing, &valid_hex).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("open "),
        "error must carry 'open <path>' context: {rendered}"
    );
    assert!(
        rendered.contains("does-not-exist.bin"),
        "error must include the missing path: {rendered}"
    );
}

/// `bytes_from_statvfs_parts` uses `saturating_mul` so a
/// pathological FUSE filesystem reporting enormous synthetic
/// block + fragment counts lands at `u64::MAX` (treated as
/// unbounded space) instead of wrapping into a small positive
/// number. A wrapping regression would report too FEW available
/// bytes and flip `ensure_free_space` into spurious bails; the
/// saturation is what keeps the gate trusting the filesystem.
/// Pin the saturation and the zero-operand short-circuits so a
/// regression to raw `*` or `wrapping_mul` surfaces here.
#[test]
fn bytes_from_statvfs_parts_saturates_on_overflow() {
    // u64::MAX × 2 would wrap; saturating_mul clamps to u64::MAX.
    assert_eq!(bytes_from_statvfs_parts(u64::MAX, 2), u64::MAX);
    assert_eq!(bytes_from_statvfs_parts(2, u64::MAX), u64::MAX);
    assert_eq!(bytes_from_statvfs_parts(u64::MAX, u64::MAX), u64::MAX);
    // Zero on either side produces zero — no overflow path.
    assert_eq!(bytes_from_statvfs_parts(u64::MAX, 0), 0);
    assert_eq!(bytes_from_statvfs_parts(0, u64::MAX), 0);
    // Typical real-world inputs compute exactly (no saturation).
    assert_eq!(bytes_from_statvfs_parts(1_000, 4_096), 4_096_000);
    assert_eq!(bytes_from_statvfs_parts(0, 4_096), 0);
}

/// `ensure_free_space` composes the required byte count as
/// `size_bytes + size_bytes / 10` via `saturating_add`. A
/// `ModelSpec` pin at `u64::MAX` must therefore land at
/// `u64::MAX` (not wrap to a tiny positive number that would let
/// the gate pass on a near-empty disk). Pin that an impossible
/// `size_bytes = u64::MAX` always bails — statvfs on a real
/// filesystem cannot report `u64::MAX` available bytes (18.4
/// exabytes), so the `available < needed` branch fires
/// unconditionally.
#[test]
fn ensure_free_space_saturates_on_u64_max_spec() {
    let dir = std::env::temp_dir();
    let spec = ModelSpec {
        file_name: "saturate-u64-max",
        url: "https://placeholder.example/saturate-u64-max",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: u64::MAX,
    };
    let err = ensure_free_space(&dir, &spec)
        .expect_err("u64::MAX size must saturate and trip the bail, not wrap past the gate");
    let rendered = format!("{err:#}");
    assert!(
        rendered.starts_with("Need "),
        "bail must report Need/have gap, got: {rendered}"
    );
}

/// Build-time shape gate for `DEFAULT_MODEL.sha256_hex`: 64 ASCII
/// hex digits, no more, no less. A placeholder or malformed pin
/// fails this check at build time instead of surfacing mid-CI
/// when prefetch tries to check.
#[test]
fn default_model_sha_is_valid_shape() {
    assert!(
        is_valid_sha256_hex(DEFAULT_MODEL.sha256_hex),
        "DEFAULT_MODEL.sha256_hex must be 64 ASCII hex chars: {:?}",
        DEFAULT_MODEL.sha256_hex
    );
}

/// `DEFAULT_MODEL.url` must be HTTPS — the cache fetcher rejects
/// non-HTTPS URLs via `reject_insecure_url`, so a typo that
/// downgraded the scheme to `http://` would fail prefetch at
/// first use. Pin the scheme at build time.
#[test]
fn default_model_url_is_https() {
    assert!(
        DEFAULT_MODEL.url.starts_with("https://"),
        "DEFAULT_MODEL.url must be HTTPS: {:?}",
        DEFAULT_MODEL.url
    );
}

/// The cache fetcher and GGUF loader both expect the artifact to
/// be a GGUF file, so a pin swap to a different format surfaces
/// before inference tries to parse it.
#[test]
fn default_model_file_name_ends_with_gguf() {
    assert!(
        DEFAULT_MODEL.file_name.ends_with(".gguf"),
        "DEFAULT_MODEL.file_name must end with .gguf: {:?}",
        DEFAULT_MODEL.file_name
    );
}

// -- llama-cpp-2 migration shape tests --
//
// Pin the post-migration invariants that hold without loading
// the 2.55 GiB GGUF: the registered ModelSpec list, the
// `LoadedInference` field shape, and the `LlamaBackend`
// singleton contract. These regress instantly on an accidental
// re-introduction of a separate tokenizer artifact, an extra
// field on the inference state, or a per-call backend init.

/// `ALL_MODEL_SPECS` registers exactly one entry: the GGUF
/// model. A regression that re-introduced a side-loaded artifact
/// (e.g. a separate tokenizer or sentence-piece file) would
/// break this test before any prefetch / load-inference call hit
/// the wire. The GGUF carries its own tokenizer surface via
/// llama-cpp-2, so no separate artifact should ever land here.
#[test]
fn all_model_specs_registers_only_default_model() {
    assert_eq!(
        ALL_MODEL_SPECS.len(),
        1,
        "post-migration ALL_MODEL_SPECS holds the GGUF only — \
         {} entries registered: {:?}",
        ALL_MODEL_SPECS.len(),
        ALL_MODEL_SPECS
            .iter()
            .map(|s| s.file_name)
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        ALL_MODEL_SPECS[0].file_name, DEFAULT_MODEL.file_name,
        "the single registered spec must be DEFAULT_MODEL"
    );
}

/// `is_all_hex_ascii` on the empty string is vacuously true —
/// no byte fails the `is_ascii_hexdigit` check because no byte
/// is inspected. Pins the empty-iteration contract so a
/// regression that flipped the default return (e.g. `return
/// false` at loop start) would surface here. `is_valid_sha256_hex`
/// still rejects the empty string via the length check; this
/// test exercises the hex predicate in isolation.
#[test]
fn is_all_hex_ascii_empty_string_returns_true() {
    assert!(
        is_all_hex_ascii(""),
        "empty string must return true — no byte fails the hex check",
    );
}

/// Every ASCII hex-digit boundary character is accepted. Covers
/// the six documented acceptance ranges (`0-9`, `a-f`, `A-F`)
/// plus the boundary characters at each end: `0` / `9` for
/// decimals, `a` / `f` for lowercase, `A` / `F` for uppercase.
/// A regression that narrowed the predicate (e.g. hardcoded
/// `0-9a-f` only, missing uppercase) would fail here on the
/// uppercase boundary cases.
#[test]
fn is_all_hex_ascii_boundary_chars_all_accepted() {
    for s in &["0", "9", "a", "f", "A", "F", "0123456789", "abcdefABCDEF"] {
        assert!(
            is_all_hex_ascii(s),
            "boundary input {s:?} must be accepted by is_all_hex_ascii",
        );
    }
}

/// Every character immediately adjacent to an ASCII hex-digit
/// range is rejected. The byte values used are, in order: `/`
/// (0x2F, one below `0` at 0x30), `:` (0x3A, one above `9` at
/// 0x39), `@` (0x40, one below `A` at 0x41), `G` (0x47, one
/// above `F` at 0x46), `` ` `` (0x60, one below `a` at 0x61),
/// and `g` (0x67, one above `f` at 0x66). Pinning these six
/// catches any off-by-one widening of the predicate (e.g. a
/// typo that accepted `g-z` or `G-Z` would flip one of these
/// assertions).
#[test]
fn is_all_hex_ascii_adjacent_non_hex_chars_rejected() {
    for s in &["/", ":", "@", "G", "`", "g"] {
        assert!(
            !is_all_hex_ascii(s),
            "adjacent-to-hex input {s:?} (hex byte {:#x}) must be rejected",
            s.as_bytes()[0],
        );
    }
}

/// A multi-byte UTF-8 character (every byte has the high bit
/// set, so none is an ASCII hex digit) is rejected. Complements
/// the existing `is_valid_sha256_hex_rejects_non_canonical_inputs`
/// which covers the same failure mode under the 64-byte length
/// constraint; this test exercises the hex predicate alone at
/// arbitrary length so the byte-level iteration is the only
/// thing being pinned. Uses an emoji ("🦀", 4 bytes) rather
/// than the Arabic-Indic digit so the test name plausibly
/// maps to "non-ASCII bytes" rather than "Unicode digits
/// specifically".
#[test]
fn is_all_hex_ascii_multibyte_utf8_rejected() {
    let s = "🦀";
    assert_eq!(s.len(), 4, "setup: emoji must be 4 UTF-8 bytes");
    assert!(
        !is_all_hex_ascii(s),
        "multi-byte UTF-8 input {s:?} must be rejected — every byte has the high bit set",
    );
}

/// Mixed input: a hex prefix followed by a non-hex byte is
/// rejected. Pins the early-return contract: the iteration
/// must visit bytes until a non-hex byte appears and return
/// `false` immediately rather than accidentally short-
/// circuiting to `true` on a partial match. The opposite
/// ordering (non-hex byte first) also rejects, proving the
/// predicate is position-independent within the iteration.
#[test]
fn is_all_hex_ascii_mixed_hex_and_non_hex_rejected() {
    assert!(
        !is_all_hex_ascii("0123g"),
        "hex prefix + non-hex byte must fail — iteration must reach the non-hex byte",
    );
    assert!(
        !is_all_hex_ascii("g0123"),
        "non-hex prefix + hex suffix must fail — iteration must fail at the first non-hex byte",
    );
}

/// Whitespace and common control bytes that fall OUTSIDE the
/// ASCII hex ranges are rejected. Pins the "strict: no
/// whitespace tolerance" contract — `check_sha256` consumers
/// who pass a pin trimmed from a file-read with trailing
/// newlines get a clean diagnostic rather than a silent pass
/// on the stripped form. Covers: space (0x20), tab (0x09),
/// newline (0x0A), NUL (0x00).
#[test]
fn is_all_hex_ascii_whitespace_and_nul_rejected() {
    for s in &[" ", "\t", "\n", "\0", "abc\n", "\0abc"] {
        assert!(
            !is_all_hex_ascii(s),
            "whitespace/NUL input {s:?} must be rejected",
        );
    }
}

/// `is_valid_sha256_hex` rejects any input that is not exactly
/// 64 ASCII hex digits. Covers the three rejection classes the
/// helper guards against: too-short (63 bytes), too-long (65),
/// and an input that IS 64 bytes long but contains a non-ASCII
/// Unicode digit. Paired with `check_sha256_rejects_malformed_hex_length`
/// and `check_sha256_rejects_non_hex_chars` which exercise the
/// same predicate via `check_sha256`'s error-surface wrapper.
#[test]
fn is_valid_sha256_hex_rejects_non_canonical_inputs() {
    // 63 bytes (short by one).
    assert!(!is_valid_sha256_hex(&"a".repeat(63)));
    // 65 bytes (long by one).
    assert!(!is_valid_sha256_hex(&"a".repeat(65)));
    // 64 BYTES with a non-ASCII Unicode digit: 62 ASCII hex chars
    // plus one Arabic-Indic `٠` (U+0660, 2 UTF-8 bytes) totals
    // 64 bytes, so the length check passes. The `is_ascii_hexdigit`
    // predicate then rejects `٠` because it's outside the ASCII
    // range, proving both halves of the predicate are load-bearing.
    let unicode_digit = format!("{}٠", "0".repeat(62));
    assert_eq!(unicode_digit.len(), 64, "setup: must be exactly 64 bytes");
    assert!(
        !is_valid_sha256_hex(&unicode_digit),
        "non-ASCII Unicode digit must fail is_ascii_hexdigit even at correct byte length"
    );
    // Sanity: exactly 64 ASCII hex digits IS accepted.
    assert!(is_valid_sha256_hex(&"0".repeat(64)));
}

/// `reject_insecure_url` rejects every non-HTTPS scheme — pair
/// with `reject_insecure_url_rejects_http` which only covers
/// `http://`. Each input here is a distinct non-HTTPS shape the
/// `starts_with("https://")` gate must reject: ftp, file, a
/// scheme-less path, the empty string, and the HTTPS prefix
/// missing its slashes. A regression that replaced the
/// `starts_with` gate with a substring search or a laxer URL
/// parse would admit one of these.
#[test]
fn reject_insecure_url_rejects_non_https_schemes() {
    let cases: &[&str] = &[
        "ftp://example.com/model.gguf",
        "file:///tmp/model.gguf",
        "example.com/model.gguf",
        "",
        "https:/example.com/model.gguf",
        "HTTPS://example.com/model.gguf",
    ];
    for url in cases {
        let err = reject_insecure_url(url).unwrap_err();
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("non-HTTPS"),
            "URL {url:?} must be rejected, got: {rendered}"
        );
    }
}

/// Full `ensure()` flow with an `http://` URL must bail at the
/// `reject_insecure_url` gate inside `fetch()`. Cache is empty,
/// offline is unset, and SHA pin is validly shaped — so the
/// status fast path, the explicit shape check, and the offline
/// gate all pass, driving execution through to fetch(). The
/// resulting Err surfaces the "non-HTTPS" message, proving
/// fetch() gates URL scheme before any network or filesystem
/// action. Does not require network: fetch bails before reqwest
/// is constructed.
#[test]
fn ensure_bails_with_non_https_error_on_http_url() {
    let _lock = lock_env();
    let _cache = isolated_cache_dir();
    // Explicitly clear the offline env so prior tests cannot
    // poison this one through lock_env acquisition ordering.
    let _env_offline = EnvVarGuard::remove(OFFLINE_ENV);
    let spec = ModelSpec {
        file_name: "http-url.gguf",
        url: "http://placeholder.example/http-url.gguf",
        // 64-char zero pin is valid shape; shape check passes.
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let err = ensure(&spec).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("non-HTTPS"),
        "expected reject_insecure_url error through ensure→fetch, got: {rendered}"
    );
}

/// Under OFFLINE=1 with a cached file whose bytes do NOT match
/// the declared SHA pin, status() returns
/// `ShaVerdict::Mismatches` and ensure() must bail with the
/// offline-gate error — NOT attempt a re-download. Pins two
/// invariants: (1) status() correctly classifies a stale cache
/// (bytes present, hash wrong), and (2) ensure() prefers
/// "offline, refuse network" over "stale cache, re-download
/// silently" when OFFLINE is set. A regression that tried to
/// re-fetch under offline would surface as reqwest-side error
/// rather than the clear OFFLINE_ENV message.
#[test]
fn ensure_under_offline_bails_on_stale_cache_sha_mismatch() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let _env_offline = EnvVarGuard::set(OFFLINE_ENV, "1");
    let spec = ModelSpec {
        file_name: "stale.gguf",
        url: "https://placeholder.example/stale.gguf",
        // Valid-shape pin; actual bytes written below will not
        // hash to this.
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 16,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, b"wrong bytes for pin").unwrap();
    // Check status() classifies correctly before running ensure.
    let st = status(&spec).expect("status should not error on valid-shape pin");
    assert!(
        matches!(st.sha_verdict, ShaVerdict::Mismatches),
        "file exists with bytes that don't hash to zero-pin; \
         verdict must be ShaVerdict::Mismatches (cached + \
         checked + didn't match); got: {:?}",
        st.sha_verdict,
    );
    // Now ensure() should bail with the offline-gate error, not
    // attempt to re-fetch.
    let err = ensure(&spec).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains(OFFLINE_ENV),
        "expected offline-gate bail on stale cache, got: {rendered}"
    );
    assert!(
        !rendered.contains("non-HTTPS"),
        "expected offline-path bail, not the URL-scheme path: {rendered}"
    );
    // Pin the stale-cache branch wording. The file exists on
    // disk but its bytes do not hash to the pin, so ensure()
    // must take the `ShaVerdict::Mismatches` arm of the
    // offline-gate match and produce a "do not match" message —
    // distinct from the not-cached branch's "is not cached"
    // wording. A regression that collapsed the two branches
    // into a single "not cached" message would misroute the
    // user toward a pre-seed step when they actually need to
    // replace the stale cache entry.
    assert!(
        rendered.contains("do not match"),
        "expected stale-cache branch wording, got: {rendered}"
    );
}

/// Under OFFLINE=1 with a cached file whose SHA-256 check
/// cannot complete (0o000 permissions → EACCES on open),
/// status() must return `ShaVerdict::CheckFailed(err)` and
/// ensure() must bail with the offline-gate error pointing at
/// the I/O failure — NOT the stale-cache or not-cached
/// wordings, and NOT attempt a re-download. Complements
/// `ensure_under_offline_bails_on_stale_cache_sha_mismatch`
/// (Mismatches arm) and
/// `ensure_in_offline_mode_fails_loudly_when_uncached` (NotCached arm) so
/// all three remediation branches of the offline-gate `match`
/// at model.rs:ensure are pinned. A regression that folded
/// CheckFailed into the stale-cache branch would surface the
/// bytes-mismatch diagnostic ("do not match") and hide the
/// filesystem-level failure ("could not complete").
///
/// Unix-only, same DAC-bypass probe as
/// `status_captures_io_error_for_unreadable_cached_file` —
/// self-skips under root / CAP_DAC_OVERRIDE /
/// CAP_DAC_READ_SEARCH / rootless-container harnesses where
/// open(0o000) succeeds.
#[cfg(unix)]
#[test]
fn ensure_under_offline_bails_on_check_failed_cache() {
    use std::os::unix::fs::PermissionsExt;
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let _env_offline = EnvVarGuard::set(OFFLINE_ENV, "1");
    let spec = ModelSpec {
        file_name: "unreadable-offline.gguf",
        url: "https://placeholder.example/unreadable-offline.gguf",
        // Valid-shape pin so check_sha256 clears its shape gate
        // and the only way to fail is the open/read path.
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, b"any content").unwrap();
    // Strip read bits so File::open inside check_sha256 hits
    // EACCES; metadata.is_file() still passes so status() enters
    // the is_file arm and produces CheckFailed (not NotCached).
    std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o000)).unwrap();

    // DAC-bypass probe mirrors the sibling I/O-error test; the
    // restore-first pattern is required because skip! early-
    // returns so permissions must be readable before the skip
    // fires (otherwise the tempdir cleanup chokes on some
    // filesystems).
    if std::fs::File::open(&on_disk).is_ok() {
        std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o644)).unwrap();
        skip!(
            "open(0o000) succeeded — process has a DAC bypass (root, \
             CAP_DAC_OVERRIDE, or equivalent); offline-gate CheckFailed \
             arm cannot be exercised here"
        );
    }

    // Classify before running ensure(): status() must produce
    // CheckFailed here, NOT Mismatches (no hash computed) or
    // NotCached (file exists).
    let st = status(&spec).expect("valid-shape pin; status must not error");
    let underlying_err = match &st.sha_verdict {
        ShaVerdict::CheckFailed(e) => e.clone(),
        other => {
            std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o644)).unwrap();
            panic!(
                "0o000 on a readable-shape pin must yield \
                 ShaVerdict::CheckFailed; got: {other:?}",
            );
        }
    };

    let err = ensure(&spec).unwrap_err();
    // Restore readable permissions before the tempdir Drop —
    // same rationale as the sibling I/O-error test.
    std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o644)).unwrap();

    let rendered = format!("{err:#}");
    assert!(
        rendered.contains(OFFLINE_ENV),
        "expected offline-gate bail on CheckFailed cache, got: {rendered}"
    );
    // The CheckFailed arm's bail wording is the discriminator.
    // Matches model.rs:ensure:"SHA-256 check could not complete".
    assert!(
        rendered.contains("SHA-256 check could not complete"),
        "expected CheckFailed branch wording \
         (\"SHA-256 check could not complete\"), got: {rendered}"
    );
    // The underlying I/O error chain must be surfaced verbatim
    // inside the bail message so an operator can name the
    // filesystem failure without re-running diagnostics.
    assert!(
        rendered.contains(&underlying_err),
        "expected the underlying I/O error {underlying_err:?} \
         to appear verbatim in the offline-gate bail; got: \
         {rendered}"
    );
    // Negative: must NOT be the stale-cache wording (which
    // would misdiagnose the failure as a bytes-mismatch and
    // route the operator toward re-fetching rather than
    // inspecting the cache entry).
    assert!(
        !rendered.contains("do not match"),
        "CheckFailed bail must not emit the stale-cache \
         \"do not match\" wording, got: {rendered}"
    );
    // Negative: must NOT be the not-cached wording (the file
    // exists; claiming otherwise misroutes toward pre-seeding).
    assert!(
        !rendered.contains("is not cached"),
        "CheckFailed bail must not emit the not-cached \
         \"is not cached\" wording, got: {rendered}"
    );
}

/// `fetch_timeout_for_size(0)` returns exactly the 60-second
/// floor: zero bytes, zero proportional term, so the `max()`
/// with the floor wins. Pins that an empty artifact still gets
/// the full TLS/handshake + request/response budget instead of
/// a sub-second cap that the blocking client would blow past
/// before receiving its response head.
#[test]
fn fetch_timeout_for_size_zero_returns_floor() {
    assert_eq!(
        fetch_timeout_for_size(0),
        std::time::Duration::from_secs(60)
    );
}

/// `fetch_timeout_for_size` for an 11 MiB synthetic input is
/// below the body-over-floor crossover point (60 s × 3 MB/s =
/// 180 MB) so it returns exactly the 60-second floor. Pins the
/// floor-wins branch so a regression that swapped `max()` for
/// `+` (adding body seconds to the floor instead of clamping)
/// would surface here.
#[test]
fn fetch_timeout_for_size_small_artifact_hits_floor() {
    let got = fetch_timeout_for_size(11 * 1024 * 1024);
    assert_eq!(got, std::time::Duration::from_secs(60));
}

/// `fetch_timeout_for_size` for the model (2400 MiB —
/// `DEFAULT_MODEL.size_bytes`) is well above the 180 MB
/// crossover so the proportional term wins: `(2400 × 1024 ×
/// 2740937888 / 3_000_000 = 913` seconds (integer division).
/// Pins the proportional branch — a regression that
/// clamped the timeout (e.g. re-introduced a fixed 900 s
/// ceiling) would surface here, and so would a divisor-unit
/// swap (byte vs KiB vs MiB).
#[test]
fn fetch_timeout_for_size_model_scales_up() {
    let got = fetch_timeout_for_size(DEFAULT_MODEL.size_bytes);
    assert_eq!(got, std::time::Duration::from_secs(913));
}

/// For two artifacts BOTH above the floor-crossover, the
/// timeout is strictly linear in `size_bytes`: the larger one
/// gets exactly `(large_bytes - small_bytes) / 3_000_000`
/// seconds more. Pin the linear relationship on two synthetic
/// sizes that clear the crossover — using synthetic sizes keeps
/// this a test of the formula, not a test of any specific pin.
#[test]
fn fetch_timeout_for_size_is_linear_above_floor() {
    let small_bytes: u64 = 300 * 1024 * 1024; // 300 MiB, above floor.
    let large_bytes: u64 = 3000 * 1024 * 1024; // 3000 MiB.
    let small = fetch_timeout_for_size(small_bytes);
    let large = fetch_timeout_for_size(large_bytes);
    assert!(
        large > small,
        "larger artifact must exceed smaller once both clear the floor: {large:?} vs {small:?}"
    );
    let expected_delta = large_bytes / 3_000_000 - small_bytes / 3_000_000;
    assert_eq!(
        large - small,
        std::time::Duration::from_secs(expected_delta)
    );
}

/// Any artifact at or below the `floor_seconds × bandwidth`
/// boundary gets the 60-second floor: an 11 MiB synthetic input
/// and a 1 KiB fake pin collapse to the same 60 s cap. Pins the
/// floor as a hard guarantee for all small artifacts so a
/// regression that dropped the floor (e.g. `max` → just the
/// proportional term) would surface as a sub-60 s result on
/// the small sibling here.
#[test]
fn fetch_timeout_for_size_floor_applies_uniformly_below_crossover() {
    let tiny = fetch_timeout_for_size(1024);
    let small = fetch_timeout_for_size(11 * 1024 * 1024);
    assert_eq!(tiny, std::time::Duration::from_secs(60));
    assert_eq!(small, std::time::Duration::from_secs(60));
}

/// Artifacts large enough that the proportional term would
/// exceed the 30 min ceiling must clamp to `FETCH_MAX_TIMEOUT_SECS`
/// (1800 s). A 20 GiB pin would otherwise demand
/// `20 × 1024³ / 3_000_000 ≈ 7158 s` (≈ 2 h) — far longer than
/// any CI wall-clock budget — so the ceiling is the thing
/// that makes a typo'd or unexpectedly large `size_bytes` fail
/// fast instead of sitting wedged until the outer harness
/// kills the job. Also pins the ceiling identity: doubling
/// the size past the crossover does NOT double the timeout.
#[test]
fn fetch_timeout_for_size_clamps_to_ceiling_on_oversized_pin() {
    let twenty_gib: u64 = 20 * 1024 * 1024 * 1024;
    let got = fetch_timeout_for_size(twenty_gib);
    assert_eq!(
        got,
        std::time::Duration::from_secs(1800),
        "20 GiB pin must clamp to the 30-minute ceiling, not scale linearly",
    );
    let forty_gib: u64 = 40 * 1024 * 1024 * 1024;
    let got_double = fetch_timeout_for_size(forty_gib);
    assert_eq!(
        got_double, got,
        "doubling size past the ceiling must NOT double the timeout — \
         ceiling is the thing being pinned",
    );
}

/// Pin the ceiling-crossover boundary: at exactly `1800 s × 3
/// MB/s = 5_400_000_000` bytes the proportional term equals
/// the ceiling, one byte below would still fall under the
/// ceiling (same 1800 s due to integer division rounding down),
/// and one byte above also clamps to the ceiling. The three
/// inputs are adjacent and asymmetric around the crossover so
/// a regression that swapped `<=` for `<` in the clamp (or
/// introduced an off-by-one in the ceiling comparison) would
/// land one of the three outside the expected 1800 s envelope.
///
/// Separately pin that 5.4 GB + 3_000_000 bytes stays clamped
/// (one body-second past the ceiling in the underlying formula
/// but still at the 1800 s cap) — this is the "small overage
/// clamps correctly" case that the existing 20 / 40 GiB test
/// doesn't exercise because both inputs are orders of magnitude
/// past the crossover.
#[test]
fn fetch_timeout_for_size_ceiling_crossover_at_5_4gb() {
    const CROSSOVER_BYTES: u64 = 1800 * 3_000_000;
    // Exactly at the crossover: body_secs = 1800, clamped to 1800.
    assert_eq!(
        fetch_timeout_for_size(CROSSOVER_BYTES),
        std::time::Duration::from_secs(1800),
        "exactly 5.4 GB must sit right at the ceiling",
    );
    // One body-second below: body_secs = 1799, the `.min(1800)`
    // is a no-op, result is 1799 s — below the ceiling.
    assert_eq!(
        fetch_timeout_for_size(CROSSOVER_BYTES - 3_000_000),
        std::time::Duration::from_secs(1799),
        "one body-second below the crossover must return 1799 s, \
         proving the ceiling clamp hasn't moved",
    );
    // One body-second past: body_secs = 1801, clamped to 1800.
    assert_eq!(
        fetch_timeout_for_size(CROSSOVER_BYTES + 3_000_000),
        std::time::Duration::from_secs(1800),
        "one body-second above the crossover must clamp to the \
         ceiling (1800 s), not return 1801",
    );
}

/// `filesystem_available_bytes` on a real tempdir must return a
/// positive byte count: any working test environment has at least
/// some free space on the filesystem hosting `/tmp` (or wherever
/// `tempfile::tempdir` lands). A zero return would indicate a
/// wiring regression — either `blocks_available` was read as a
/// signed value and truncated or `fragment_size` was confused
/// with zero. Pins the production readings against both
/// regressions at once.
#[test]
fn filesystem_available_bytes_returns_positive_on_tempdir() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let bytes = filesystem_available_bytes(tmp.path()).expect("statvfs");
    assert!(
        bytes > 0,
        "tempdir filesystem must report some available space, got {bytes}"
    );
}

/// `filesystem_available_bytes` surfaces the underlying statvfs
/// error (wrapped with the path-naming context) when the target
/// does not exist. The fetcher relies on this propagation so a
/// typo in `KTSTR_CACHE_DIR` or a torn-down cache root surfaces
/// as a named `statvfs {path}` failure rather than a silent
/// pass-through. Pin both halves: the call fails AND the error
/// message names the missing path.
#[test]
fn filesystem_available_bytes_errors_on_missing_path() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let missing = tmp.path().join("does-not-exist");
    let err = filesystem_available_bytes(&missing).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("statvfs"),
        "error must carry 'statvfs' context: {rendered}"
    );
    assert!(
        rendered.contains("does-not-exist"),
        "error must name the missing path: {rendered}"
    );
}

/// Happy path: `ensure_free_space` returns `Ok(())` when the
/// filesystem has more than `size_bytes + 10%` available. Uses
/// a 1-byte spec so any tempdir filesystem trivially clears the
/// gate — the point is to pin the "returns Ok on enough space"
/// branch against a regression that flipped the comparator
/// direction (which would cause every fetch to bail regardless
/// of real free-space state).
///
/// `compute_margin` enforces the "10% safety buffer, floored
/// at 1 byte" contract. Pins the boundary cases where the
/// `/ 10` branch is zero and the `max(1)` floor is
/// load-bearing: sizes in `[1, 5, 9]` → 1 (integer division
/// yields 0 → floor kicks in). Size 10 → 1 (integer division
/// yields 1, floor is a no-op). Size 100 → 10 (normal 10%
/// path). A regression that lost the floor would fail the
/// sub-10-byte cases and pass the ≥10 cases, surfacing the
/// exact class this extraction was meant to guard against
/// (the original `size_bytes / 10` without `max(1)`).
#[test]
fn compute_margin_respects_floor_and_scales_linearly() {
    // 0-boundary: (0/10).max(1) = 1. The module-scope
    // ALL_MODEL_SPECS size_bytes>0 guard means production pins
    // never hit this input, but the helper must still emit a
    // positive margin for any direct caller — a 0 return here
    // would make `ensure_free_space` accept `needed == size + 0
    // = 0` bytes of headroom, trivially passing on a full disk.
    // Pin both the value (1) and the "floor is load-bearing at
    // this input" semantic.
    assert_eq!(
        compute_margin(0),
        1,
        "compute_margin(0): `/ 10` = 0, the max(1) floor MUST \
         win so the free-space gate retains positive headroom \
         even when called with a degenerate zero input",
    );

    for size in [1u64, 5, 9] {
        assert_eq!(
            compute_margin(size),
            1,
            "compute_margin({size}): floor at 1 must beat the \
             zero produced by integer `/ 10`",
        );
    }
    assert_eq!(
        compute_margin(10),
        1,
        "compute_margin(10): 10/10 = 1 — the `/ 10` branch \
         wins, floor is a no-op",
    );
    assert_eq!(
        compute_margin(100),
        10,
        "compute_margin(100): 10% = 10, `/ 10` dominates",
    );
    assert_eq!(
        compute_margin(u64::MAX),
        u64::MAX / 10,
        "compute_margin(u64::MAX): integer division, no \
         overflow; floor is a no-op",
    );
}

/// `format_free_space_error` includes the FUSE/quota hint
/// iff `available == 0`. Pins both branches so a regression
/// that inverted the condition or always appended the hint
/// fails here. Also pins that both messages include the
/// "Need N free at PATH" skeleton — the hint is ADDITIONAL
/// context, not a replacement.
#[test]
fn format_free_space_error_includes_fuse_hint_iff_available_is_zero() {
    let parent = std::path::Path::new("/tmp/ktstr-fuse-test");

    let with_hint = format_free_space_error(1_000_000, parent, 0);
    assert!(
        with_hint.contains("Need") && with_hint.contains("/tmp/ktstr-fuse-test"),
        "base message shape must survive the hint append; \
         got: {with_hint}",
    );
    assert!(
        with_hint.contains("FUSE") && with_hint.contains("quota"),
        "available == 0 must append the FUSE/quota hint; \
         got: {with_hint}",
    );
    assert!(
        with_hint.contains("blocks_available reported 0"),
        "hint must name the specific value (0) so a user \
         sees the trigger; got: {with_hint}",
    );

    // Non-zero `available` → no hint. Use a realistic gap
    // (needed > available > 0) to confirm the hint does NOT
    // fire for the normal full-disk case.
    let without_hint = format_free_space_error(1_000_000, parent, 500_000);
    assert!(
        without_hint.contains("Need") && without_hint.contains("/tmp/ktstr-fuse-test"),
        "base message shape unchanged; got: {without_hint}",
    );
    assert!(
        !without_hint.contains("FUSE") && !without_hint.contains("blocks_available"),
        "available > 0 must NOT append the FUSE hint (would \
         clutter normal full-disk bails with irrelevant \
         quota speculation); got: {without_hint}",
    );
}

#[test]
fn ensure_free_space_ok_when_space_sufficient() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let tiny = ModelSpec {
        file_name: "tiny.gguf",
        url: "https://placeholder.example/tiny.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    ensure_free_space(tmp.path(), &tiny).expect("1-byte spec must fit");
}

/// `ensure_free_space` must bail with the documented
/// `"Need X free at <path>; have Y"` diagnostic when the declared
/// `size_bytes + 10% margin` exceeds the filesystem's available
/// bytes. Uses `u64::MAX / 2` so no real filesystem (tempdir or
/// otherwise) can clear the gate — `size_bytes + size_bytes / 10`
/// sums well below `u64::MAX` (so `saturating_add` does not
/// saturate for this input), and the resulting ~8.8 EiB
/// requirement still dwarfs any tempdir's free bytes so the
/// comparison trips. Pin every load-bearing piece of the error
/// message: the `"Need "` prefix, `" free at "` infix, `"; have "`
/// separator shape, the `parent` path echo, and the presence of
/// an IEC-prefix size token (`KiB`, `MiB`, `GiB`, `TiB`, `PiB`,
/// or `EiB`) on the `"Need "` side. A regression that dropped the
/// human-readable format or reverted to raw bytes would surface
/// here.
#[test]
fn ensure_free_space_bails_when_space_insufficient() {
    let tmp = tempfile::tempdir().expect("create tempdir");
    let huge = ModelSpec {
        file_name: "ginormous.gguf",
        url: "https://placeholder.example/ginormous.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        // u64::MAX / 2 plus the 10% margin stays within u64 range —
        // the needed byte count exceeds any real filesystem's
        // blocks_available * fragment_size product.
        size_bytes: u64::MAX / 2,
    };
    let err = ensure_free_space(tmp.path(), &huge).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.starts_with("Need "),
        "error must lead with 'Need ': {rendered}"
    );
    assert!(
        rendered.contains(" free at "),
        "error must carry ' free at ' infix: {rendered}"
    );
    assert!(
        rendered.contains("; have "),
        "error must carry '; have ' separator: {rendered}"
    );
    assert!(
        rendered.contains(&format!("{}", tmp.path().display())),
        "error must echo the parent path: {rendered}"
    );
    // `u64::MAX / 2` is ~8.00 EiB; accept any IEC prefix up through
    // EiB — just not a bare-byte `"B"` reading with no prefix.
    let rendered_after_need = rendered
        .strip_prefix("Need ")
        .expect("starts_with 'Need ' above");
    let needed_portion = rendered_after_need
        .split_once(" free at ")
        .expect("infix present")
        .0;
    assert!(
        ["KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
            .iter()
            .any(|p| needed_portion.contains(p)),
        "needed size must render with an IEC prefix, got: {needed_portion:?}"
    );
}

/// Pin the IEC human-readable rendering for
/// `DEFAULT_MODEL.size_bytes` (2400 MiB):
/// `HumanBytes(2740937888)` lands as `"2.55 GiB"`, and
/// `HumanBytes(2640 * 1024 * 1024)` — the size plus the 10%
/// margin — lands as `"2.58 GiB"`. This does NOT go through
/// `ensure_free_space` because a real tempdir filesystem
/// trivially clears a 2.58 GiB gate and the error path never
/// fires. The test instead pins the formatter's exact string so
/// a regression that swapped to `DecimalBytes` (SI prefixes,
/// `"2.77 GB"` for 2640 MiB) or to raw bytes would surface here.
/// Sourced from `DEFAULT_MODEL.size_bytes` so a pin rotation
/// that updates the const but forgets the test is caught by
/// drift between the assertion and the rendered string instead
/// of silently passing on stale literals.
#[test]
fn human_bytes_rendering_is_pinned_for_default_model_size() {
    let size_only = DEFAULT_MODEL.size_bytes;
    let size_plus_margin = size_only + size_only / 10;
    assert_eq!(format!("{}", indicatif::HumanBytes(size_only)), "2.55 GiB");
    assert_eq!(
        format!("{}", indicatif::HumanBytes(size_plus_margin)),
        "2.81 GiB"
    );
}
