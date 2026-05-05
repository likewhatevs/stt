//! Cache-surface tests: status, clean, ShaVerdict, mtime-size warm-cache
//! sidecar, resolve_cache_root, env-value sanitization.

use super::super::test_helpers::{EnvVarGuard, isolated_cache_dir, lock_env};
use super::*;

#[test]
fn resolve_cache_root_honors_ktstr_cache_dir() {
    // Nextest runs tests in parallel within a binary and
    // `std::env::set_var` is process-wide. `lock_env()`
    // serializes the save/mutate/restore window against every
    // other env-touching test in this crate so concurrent
    // runners in sidecar.rs / eval.rs don't race on
    // KTSTR_CACHE_DIR. Poisoned-lock recovery is handled
    // inside `lock_env()` itself, so a panic inside the
    // critical section is safe to recover through.
    let _lock = lock_env();
    let _env = EnvVarGuard::set("KTSTR_CACHE_DIR", "/explicit/override");
    let root = resolve_cache_root().unwrap();
    assert_eq!(root, PathBuf::from("/explicit/override"));
}

#[test]
fn ensure_in_offline_mode_fails_loudly_when_uncached() {
    // See `resolve_cache_root_honors_ktstr_cache_dir` for the
    // lock_env() rationale.
    let _lock = lock_env();
    let _cache = isolated_cache_dir();
    let _env_offline = EnvVarGuard::set(OFFLINE_ENV, "1");
    let fake = ModelSpec {
        file_name: "does-not-exist.gguf",
        url: "https://placeholder.example/none.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let err = ensure(&fake).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(rendered.contains(OFFLINE_ENV), "err: {rendered}");
    // Pin the not-cached branch wording: the file does not exist
    // on disk, so ensure() must take the `ShaVerdict::NotCached`
    // arm of the offline-gate match and produce "is not cached
    // at {path}". A regression that routed this case through
    // the stale-cache branch (or collapsed the two messages into
    // one generic wording) would mask the distinction from the
    // user.
    assert!(
        rendered.contains("is not cached"),
        "expected not-cached branch wording, got: {rendered}"
    );
}

/// `ensure()` must check the SHA pin shape BEFORE the offline
/// gate. A malformed pin is a programmer error that no runtime
/// state can fix — surfacing it first gives the actionable
/// "fix the ModelSpec" error instead of the downstream "OFFLINE
/// set but not cached" red herring. This test sets OFFLINE=1 AND
/// supplies a placeholder (all-`?`) SHA pin; the error must call
/// out the placeholder pin, NOT the offline gate.
#[test]
fn ensure_surfaces_sha_shape_error_before_offline_gate() {
    let _lock = lock_env();
    let _cache = isolated_cache_dir();
    let _env_offline = EnvVarGuard::set(OFFLINE_ENV, "1");
    // Placeholder-shape SHA (all-`?`, 64 chars) is 64 bytes long
    // but contains no ASCII hex digits, so is_valid_sha256_hex
    // rejects it at the shape-check step inside ensure() BEFORE
    // reaching the offline bail.
    let bad_pin = ModelSpec {
        file_name: "placeholder-pin.gguf",
        url: "https://placeholder.example/placeholder-pin.gguf",
        sha256_hex: "????????????????????????????????????????????????????????????????",
        size_bytes: 1,
    };
    let err = ensure(&bad_pin).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("placeholder or malformed"),
        "expected SHA-shape error, got: {rendered}"
    );
    assert!(
        !rendered.contains(&format!("{OFFLINE_ENV}=")),
        "shape error must NOT mention the offline gate: {rendered}"
    );
}

/// status() on a file whose bytes DO hash to the declared pin
/// must report `ShaVerdict::Matches`. Complements the three
/// failure-path tests
/// (`status_reports_cached_but_sha_mismatch_for_garbage_bytes`,
/// `status_captures_io_error_for_unreadable_cached_file`,
/// `status_surfaces_malformed_pin_error_for_cached_file`) by
/// pinning the success path — without this, a regression that
/// silently returned `Mismatches` on a good cache would break
/// [`ensure`]'s fast-path (every call would re-download) but
/// still pass every other test since they assert non-`Matches`
/// variants. The pin is computed in-process from the bytes
/// written so the assertion does not hard-code a magic digest
/// against a specific byte sequence.
#[test]
fn status_reports_matches_for_correctly_pinned_file() {
    use sha2::{Digest, Sha256};
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let bytes: &[u8] = b"model body pinned by its own hash";
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hex::encode(hasher.finalize());
    // Leak the digest to 'static so it lives inside the
    // `ModelSpec` (which holds `&'static str` for sha256_hex
    // to stay copyable). Test-only allocation, bounded by
    // the digest length (64 hex chars).
    let pin: &'static str = Box::leak(digest.into_boxed_str());
    let spec = ModelSpec {
        file_name: "pinned.gguf",
        url: "https://placeholder.example/pinned.gguf",
        sha256_hex: pin,
        size_bytes: bytes.len() as u64,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, bytes).unwrap();
    let st = status(&spec).expect("status on well-pinned file must not error");
    assert_eq!(st.path, on_disk);
    assert!(
        matches!(st.sha_verdict, ShaVerdict::Matches),
        "bytes hash to their declared pin — verdict must be \
         ShaVerdict::Matches (fast path in ensure() depends on \
         this); got: {:?}",
        st.sha_verdict,
    );
    // Defensive: the helper path `ensure()` / `prefetch` relies
    // on should line up with the variant. If the helper ever
    // drifts from the variant (e.g. is_match() returns false on
    // Matches), `sha_verdict_helpers_match_variant_semantics`
    // catches it — this assertion here catches the
    // complementary drift where `status()` constructs a Matches
    // but `is_match()` returns false.
    assert!(
        st.sha_verdict.is_match(),
        "Matches variant must answer true to .is_match(); if \
         this fails but the variant is Matches, the helper is \
         broken — see sha_verdict_helpers_match_variant_semantics",
    );
}

/// status() on a path where no file exists must report
/// `ShaVerdict::NotCached`. Complements the three cached-file
/// tests (`status_reports_cached_but_sha_mismatch_for_garbage_bytes`,
/// `status_captures_io_error_for_unreadable_cached_file`,
/// `status_surfaces_malformed_pin_error_for_cached_file`) so all
/// four `ShaVerdict` variants are pinned on the production path.
/// A regression that folded the no-file branch into a different
/// variant (e.g. `Mismatches` via a "nothing to match" read)
/// would break downstream dispatch — `ensure()` expects
/// `NotCached` to trigger a fetch, and the CLI readout expects
/// it to print the "no cached copy" hint.
#[test]
fn status_reports_not_cached_when_file_absent() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        // File is not written to disk — `metadata()` returns
        // Err(NotFound) and status() lands on the `_ => ...`
        // arm that produces `ShaVerdict::NotCached`.
        file_name: "absent.gguf",
        url: "https://placeholder.example/absent.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let st = status(&spec).expect("status on absent file must not error");
    assert_eq!(st.path, cache.path().join(spec.file_name));
    assert!(
        matches!(st.sha_verdict, ShaVerdict::NotCached),
        "absent file must produce ShaVerdict::NotCached (no \
         check performed); got: {:?}",
        st.sha_verdict,
    );
}

// ─── clean() — `cargo ktstr model clean` library helper ──────

/// `clean()` on a populated cache deletes both the GGUF
/// artifact and its `.mtime-size` warm-cache sidecar, returning
/// per-file freed-byte counts that match what was on disk.
/// Pins the happy-path contract that `cargo ktstr model clean`
/// builds its rendered output on.
#[test]
fn clean_removes_artifact_and_sidecar_and_reports_freed_bytes() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "to-clean.gguf",
        url: "https://placeholder.example/to-clean.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let artifact_path = cache.path().join(spec.file_name);
    let sidecar_path = mtime_size_sidecar_path(&artifact_path);
    let artifact_bytes = b"fake gguf body, exact length pinned by the assertion below";
    let sidecar_bytes = b"KTSTR_SHA_MTIME_SIZE_V1\n123 456\n";
    std::fs::write(&artifact_path, artifact_bytes).expect("plant artifact");
    std::fs::write(&sidecar_path, sidecar_bytes).expect("plant sidecar");

    let report = clean(&spec).expect("clean must succeed when files exist");

    assert_eq!(report.artifact_path, artifact_path);
    assert_eq!(report.sidecar_path, sidecar_path);
    assert_eq!(
        report.artifact_freed_bytes,
        Some(artifact_bytes.len() as u64),
        "artifact_freed_bytes must equal the planted artifact size",
    );
    assert_eq!(
        report.sidecar_freed_bytes,
        Some(sidecar_bytes.len() as u64),
        "sidecar_freed_bytes must equal the planted sidecar size",
    );
    assert!(
        !artifact_path.exists(),
        "artifact must be removed from disk after clean",
    );
    assert!(
        !sidecar_path.exists(),
        "sidecar must be removed from disk after clean",
    );
    assert!(
        !report.is_empty(),
        "is_empty() must be false when at least one file was removed",
    );
    assert_eq!(
        report.total_freed_bytes(),
        (artifact_bytes.len() + sidecar_bytes.len()) as u64,
        "total_freed_bytes() must sum artifact + sidecar bytes",
    );
}

/// `clean()` on an empty cache returns a [`CleanReport`] whose
/// freed-byte fields are both `None` and whose `is_empty()`
/// helper returns `true`. The CLI surface (`cargo ktstr model
/// clean`) branches on `is_empty()` to print the "no cached
/// model found" line, so the contract is load-bearing.
#[test]
fn clean_empty_cache_reports_is_empty() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "absent.gguf",
        url: "https://placeholder.example/absent.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let report = clean(&spec).expect("clean must succeed when nothing is cached");
    assert_eq!(report.artifact_path, cache.path().join(spec.file_name));
    assert_eq!(
        report.sidecar_path,
        mtime_size_sidecar_path(&cache.path().join(spec.file_name)),
    );
    assert!(
        report.artifact_freed_bytes.is_none(),
        "artifact_freed_bytes must be None when artifact was absent; got {:?}",
        report.artifact_freed_bytes,
    );
    assert!(
        report.sidecar_freed_bytes.is_none(),
        "sidecar_freed_bytes must be None when sidecar was absent; got {:?}",
        report.sidecar_freed_bytes,
    );
    assert!(
        report.is_empty(),
        "is_empty() must be true when no files were removed",
    );
    assert_eq!(
        report.total_freed_bytes(),
        0,
        "total_freed_bytes() must be 0 on an empty cache",
    );
}

/// `clean()` removes whichever of (artifact, sidecar) exists and
/// reports `None` for the absent one — the two unlinks are
/// independent. Catches a regression that gated sidecar removal
/// on artifact presence (or vice versa), which would leave stale
/// sidecars behind after a manual artifact-only delete.
#[test]
fn clean_removes_orphaned_sidecar_when_artifact_absent() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "orphan.gguf",
        url: "https://placeholder.example/orphan.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let artifact_path = cache.path().join(spec.file_name);
    let sidecar_path = mtime_size_sidecar_path(&artifact_path);
    // Plant ONLY the sidecar — the artifact stays absent.
    let sidecar_bytes = b"KTSTR_SHA_MTIME_SIZE_V1\n111 222\n";
    std::fs::write(&sidecar_path, sidecar_bytes).expect("plant orphan sidecar");

    let report = clean(&spec).expect("clean must succeed on a sidecar-only cache");

    assert!(
        report.artifact_freed_bytes.is_none(),
        "no artifact on disk → artifact_freed_bytes must be None",
    );
    assert_eq!(
        report.sidecar_freed_bytes,
        Some(sidecar_bytes.len() as u64),
        "orphaned sidecar must be removed and its size reported",
    );
    assert!(
        !sidecar_path.exists(),
        "orphaned sidecar must be removed from disk",
    );
    assert!(
        !report.is_empty(),
        "is_empty() must be false when the sidecar was removed",
    );
}

/// Symmetric to `clean_removes_orphaned_sidecar_when_artifact_absent`:
/// when only the artifact is on disk (no sidecar), `clean()` must
/// still remove the artifact and report `None` for the sidecar.
/// Catches the inverse coupling regression — a code change that
/// only invokes `remove_if_present` for the sidecar when the
/// artifact removal succeeded would pass the orphan-sidecar
/// test (sidecar removed regardless) but fail this test (the
/// artifact must still be removed when the sidecar is absent).
/// Together the two tests pin both directions of the
/// independence contract.
#[test]
fn clean_removes_artifact_when_sidecar_absent() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "artifact-only.gguf",
        url: "https://placeholder.example/artifact-only.gguf",
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let artifact_path = cache.path().join(spec.file_name);
    let sidecar_path = mtime_size_sidecar_path(&artifact_path);
    // Plant ONLY the artifact — the sidecar stays absent.
    let artifact_bytes = b"artifact-only body, sidecar will not be planted";
    std::fs::write(&artifact_path, artifact_bytes).expect("plant artifact-only");

    let report = clean(&spec).expect("clean must succeed on an artifact-only cache");

    assert_eq!(
        report.artifact_freed_bytes,
        Some(artifact_bytes.len() as u64),
        "artifact must be removed and its size reported",
    );
    assert!(
        report.sidecar_freed_bytes.is_none(),
        "no sidecar on disk → sidecar_freed_bytes must be None",
    );
    assert!(
        !artifact_path.exists(),
        "artifact must be removed from disk",
    );
    assert!(
        !sidecar_path.exists(),
        "sidecar that was never planted must remain absent",
    );
    assert!(
        !report.is_empty(),
        "is_empty() must be false when the artifact was removed",
    );
}

/// Exercises every [`ShaVerdict`] variant's helper methods
/// (`is_cached`, `is_match`, `check_error`) against a
/// hand-constructed instance of that variant. This guards the
/// helper contract independently of [`status`]'s construction
/// path: a regression that left the enum fine but broke a
/// helper (e.g. `is_match()` returning true on `Mismatches`, or
/// `is_cached()` returning true on `NotCached`) would pass the
/// construction tests above — those only look at the variant
/// the path produced — but fail here. The helpers are relied on
/// by `ensure()`'s fast path, the CLI readout, and the `model
/// status` integration test; a silent helper regression would
/// cascade into all of them.
#[test]
fn sha_verdict_helpers_match_variant_semantics() {
    // NotCached: no file present → is_cached=false, is_match=false, check_error=None.
    let v = ShaVerdict::NotCached;
    assert!(
        !v.is_cached(),
        "NotCached.is_cached() must be false; got true for {v:?}",
    );
    assert!(
        !v.is_match(),
        "NotCached.is_match() must be false; got true for {v:?}",
    );
    assert_eq!(
        v.check_error(),
        None,
        "NotCached.check_error() must be None; got Some for {v:?}",
    );

    // Matches: file present, SHA equals pin → is_cached=true, is_match=true, check_error=None.
    let v = ShaVerdict::Matches;
    assert!(
        v.is_cached(),
        "Matches.is_cached() must be true; got false for {v:?}",
    );
    assert!(
        v.is_match(),
        "Matches.is_match() must be true; got false for {v:?}",
    );
    assert_eq!(
        v.check_error(),
        None,
        "Matches.check_error() must be None; got Some for {v:?}",
    );

    // Mismatches: file present, SHA differs → is_cached=true, is_match=false, check_error=None.
    let v = ShaVerdict::Mismatches;
    assert!(
        v.is_cached(),
        "Mismatches.is_cached() must be true; got false for {v:?}",
    );
    assert!(
        !v.is_match(),
        "Mismatches.is_match() must be false; got true for {v:?}",
    );
    assert_eq!(
        v.check_error(),
        None,
        "Mismatches.check_error() must be None (the check ran \
         to completion); got Some for {v:?}",
    );

    // CheckFailed: file present, check errored → is_cached=true,
    // is_match=false, check_error=Some(the carried string).
    let err = "open /tmp/x: Permission denied (os error 13)";
    let v = ShaVerdict::CheckFailed(err.to_string());
    assert!(
        v.is_cached(),
        "CheckFailed.is_cached() must be true (file exists, \
         couldn't check it); got false for {v:?}",
    );
    assert!(
        !v.is_match(),
        "CheckFailed.is_match() must be false (check didn't \
         complete successfully); got true for {v:?}",
    );
    assert_eq!(
        v.check_error(),
        Some(err),
        "CheckFailed.check_error() must surface the carried \
         string verbatim so the CLI readout and the offline \
         bail can name the underlying failure; got: {:?}",
        v.check_error(),
    );
}

/// status() on a file that exists but whose SHA does not match
/// must report `ShaVerdict::Mismatches` (cached, checked,
/// didn't match). That is the branch ensure() consults to
/// decide between "reuse cached copy" and "re-download"; a
/// regression that lost the mismatch would silently re-validate
/// any garbage bytes sitting at the expected path.
#[test]
fn status_reports_cached_but_sha_mismatch_for_garbage_bytes() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "bogus.gguf",
        url: "https://placeholder.example/bogus.gguf",
        // Anything but the SHA of whatever bytes we write.
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 16,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, b"definitely-not-zero-sha").unwrap();
    let st = status(&spec).unwrap();
    assert_eq!(st.path, on_disk);
    // Pin the exact variant: garbage bytes hash cleanly to some
    // non-zero digest, so `check_sha256` returns `Ok(false)` and
    // the verdict is `Mismatches`. The complementary I/O-error
    // case produces `CheckFailed(_)`; ensure() and the CLI
    // `model status` readout branch on the variant to name the
    // specific remediation. Asserting the exact variant catches
    // a regression that might fold Mismatches into CheckFailed
    // or NotCached.
    assert!(
        matches!(st.sha_verdict, ShaVerdict::Mismatches),
        "SHA is a fixed zero pin — garbage bytes must hash to a \
         non-matching digest, producing ShaVerdict::Mismatches \
         (not CheckFailed, not NotCached); got: {:?}",
        st.sha_verdict,
    );
}

/// Complement of [`status_reports_cached_but_sha_mismatch_for_garbage_bytes`]:
/// when the cached file exists (so `metadata().is_file()` passes)
/// but `File::open()` fails with a permission error, status()
/// must report `ShaVerdict::CheckFailed(err)` carrying the
/// rendered I/O-error chain — NOT silently collapse into the
/// bytes-mismatch (`Mismatches`) branch. Exercises the
/// I/O-error arm of the `check_sha256` match in status() that
/// the structural change capturing I/O failures into the
/// `CheckFailed` variant wired up.
///
/// Unix-only: relies on POSIX permission semantics (mode 0o000
/// blocks reads). Skipped under any environment that bypasses
/// DAC on open(2) — root, a process granted CAP_DAC_OVERRIDE or
/// CAP_DAC_READ_SEARCH (e.g. via `setcap`), or certain rootless
/// container harnesses. Detection is a direct open probe on the
/// freshly chmod'd file: if `File::open` succeeds under mode
/// 0o000 this environment cannot trigger EACCES, so the
/// I/O-error arm is unreachable and the test self-skips. The
/// probe is strictly stronger than a euid check (which caught
/// root but missed every capability-bypass path) and needs no
/// `libc::capget` plumbing. Skips are logged via `eprintln!` so
/// a user invoking the suite manually sees which specific case
/// was bypassed rather than silently passed.
#[cfg(unix)]
#[test]
fn status_captures_io_error_for_unreadable_cached_file() {
    use std::os::unix::fs::PermissionsExt;
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "unreadable.gguf",
        url: "https://placeholder.example/unreadable.gguf",
        // Valid-shape pin so the shape-check branch of
        // check_sha256 doesn't fire; the only way to reach the
        // I/O-error capture path is a valid pin + open/read
        // failure on the cached file.
        sha256_hex: "0000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, b"any content").unwrap();
    // Mode 0o000 strips owner/group/other read bits so the
    // subsequent File::open inside check_sha256 hits EACCES.
    // The file itself remains in the directory (metadata.is_file
    // still returns true), so status() enters the is_file arm
    // rather than the `_ => (false, false, None)` fallback.
    std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o000)).unwrap();

    // DAC-bypass probe: if an open against the just-chmod'd file
    // succeeds, the process has a read bypass (euid 0,
    // CAP_DAC_OVERRIDE/CAP_DAC_READ_SEARCH, or equivalent
    // sandbox behavior). Restore readable permissions first
    // (skip! early-returns, so the restore must precede it) and
    // emit through the centralized skip reporter.
    if std::fs::File::open(&on_disk).is_ok() {
        std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o644)).unwrap();
        skip!(
            "open(0o000) succeeded — process has a DAC bypass (root, \
             CAP_DAC_OVERRIDE, or equivalent)"
        );
    }

    let st = status(&spec).unwrap();

    // Restore readable permissions before the tempdir Drop runs
    // its remove_dir_all. Unlink on the file needs write+execute
    // on the PARENT directory (not the file), so 0o000 on the
    // file itself wouldn't block cleanup on Linux — but some
    // filesystems and some tempfile paths are less tolerant,
    // and leaving a world-unreadable file in the tempdir after
    // assertion failures would make debug output harder. Reset
    // defensively.
    std::fs::set_permissions(&on_disk, std::fs::Permissions::from_mode(0o644)).unwrap();

    let err = match &st.sha_verdict {
        ShaVerdict::CheckFailed(e) => e.as_str(),
        other => panic!(
            "metadata().is_file() passed despite 0o000 and \
             check_sha256 hit EACCES — status must report \
             ShaVerdict::CheckFailed(_); got: {other:?}",
        ),
    };
    // `{e:#}` on a File::open failure at permission-denied yields
    // something like "open /tmp/.../unreadable.gguf: Permission
    // denied (os error 13)". The exact phrasing of std's
    // io::Error Display for EACCES is "Permission denied" on
    // Linux — pin against "ermission" (case-ambiguity safe
    // relative to "Permission") OR "denied" to survive small
    // libc-side wording drift across platforms while still
    // requiring a substantively permission-related diagnostic.
    assert!(
        err.contains("ermission") || err.contains("denied"),
        "expected permission-denied error in rendered chain, got: {err}"
    );
}

/// status() on a file that exists but whose SHA pin is malformed
/// (non-hex chars) must surface the check_sha256 error instead
/// of coercing it into `ShaVerdict::Mismatches`. A malformed pin
/// is a programmer error in the ModelSpec — silently reporting
/// "SHA doesn't match" hides the defect and misroutes downstream
/// logic into a pointless re-download branch.
#[test]
fn status_surfaces_malformed_pin_error_for_cached_file() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "malformed-pin.gguf",
        url: "https://placeholder.example/malformed-pin.gguf",
        // 64 chars, all `?` — right length, zero hex digits.
        sha256_hex: "????????????????????????????????????????????????????????????????",
        size_bytes: 1,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, b"any bytes will do").unwrap();
    let err = status(&spec).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("non-hex"),
        "expected malformed-pin error from check_sha256, got: {rendered}"
    );
    // Pin the context wrapper that names the offending
    // ModelSpec's file_name. Without this assertion, a regression
    // that dropped the .with_context layer would strip the
    // file-name annotation and leave CLI users to guess which
    // pin was malformed when multiple ModelSpec entries exist.
    assert!(
        rendered.contains(spec.file_name),
        "expected status() context to name the file, got: {rendered}"
    );
}

/// Sibling of [`status_surfaces_malformed_pin_error_for_cached_file`]
/// for the other malformed-pin branch: the pin is all ASCII hex
/// digits but has the wrong length. Exercises the
/// `expected_hex.len() != 64` branch of `check_sha256`, which
/// status() routes through the malformed-pin surface path (per
/// the is_valid_sha256_hex predicate, wrong length is as much a
/// ModelSpec defect as wrong chars). Pins the "64 chars" diagnostic
/// from `check_sha256`'s length branch so a regression that
/// collapsed the two wordings into a single generic message would
/// surface here.
#[test]
fn status_surfaces_length_fail_pin_error_for_cached_file() {
    let _lock = lock_env();
    let cache = isolated_cache_dir();
    let spec = ModelSpec {
        file_name: "short-pin.gguf",
        url: "https://placeholder.example/short-pin.gguf",
        // 63 ASCII hex digits — valid chars, wrong length.
        sha256_hex: "000000000000000000000000000000000000000000000000000000000000000",
        size_bytes: 1,
    };
    let on_disk = cache.path().join(spec.file_name);
    std::fs::write(&on_disk, b"any bytes will do").unwrap();
    let err = status(&spec).unwrap_err();
    let rendered = format!("{err:#}");
    assert!(
        rendered.contains("64 chars"),
        "expected length-fail error from check_sha256, got: {rendered}"
    );
    assert!(
        rendered.contains(spec.file_name),
        "expected status() context to name the file, got: {rendered}"
    );
}

/// With `KTSTR_CACHE_DIR` unset, `resolve_cache_root` falls
/// through to `XDG_CACHE_HOME` and appends `ktstr/models`.
#[test]
fn resolve_cache_root_honors_xdg_cache_home() {
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _env_xdg = EnvVarGuard::set("XDG_CACHE_HOME", "/xdg/caches");
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
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _env_xdg = EnvVarGuard::remove("XDG_CACHE_HOME");
    let _env_home = EnvVarGuard::set("HOME", "/home/fake");
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
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::set("KTSTR_CACHE_DIR", "");
    let _env_xdg = EnvVarGuard::set("XDG_CACHE_HOME", "/xdg/caches");
    let root = resolve_cache_root().unwrap();
    assert_eq!(
        root,
        PathBuf::from("/xdg/caches").join("ktstr").join("models"),
        "empty KTSTR_CACHE_DIR must be treated as unset so XDG wins",
    );
}

/// HOME=`/` is rejected — the resulting `/.cache/ktstr/models`
/// path's statvfs reports the root filesystem's free space
/// (typically a small constrained mount), not a usable user
/// cache. A legitimate root user without a configured home
/// should set KTSTR_CACHE_DIR or XDG_CACHE_HOME explicitly.
/// The shared validation in
/// [`crate::cache::resolve_cache_root_with_suffix`] surfaces
/// the literal-`/` arm with a path-shape-specific diagnostic
/// naming `/.cache/ktstr` so the operator immediately sees
/// what would have been written.
#[test]
fn resolve_cache_root_rejects_root_slash_home() {
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _env_xdg = EnvVarGuard::remove("XDG_CACHE_HOME");
    let _env_home = EnvVarGuard::set("HOME", "/");
    let err = resolve_cache_root().unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("HOME is `/`"),
        "expected HOME=/ specific rejection, got: {msg}"
    );
    assert!(
        msg.contains("/.cache/ktstr"),
        "diagnostic must cite the offending cache path, got: {msg}"
    );
    assert!(
        msg.contains("KTSTR_CACHE_DIR"),
        "error must suggest KTSTR_CACHE_DIR, got: {msg}"
    );
}

/// HOME=`""` is rejected by the empty-string arm of the
/// shared validator. Joining `.cache` onto an empty PathBuf
/// yields a relative `.cache` rooted at CWD instead of a
/// stable user cache. The diagnostic explicitly names the
/// empty-string shape (`Ok("")`) so an operator can identify
/// a Dockerfile `ENV HOME=` or shell-rc `export HOME=` typo
/// rather than confusing it with the container-init-dropped-HOME
/// case, which is rejected by a separate arm with a distinct
/// message.
#[test]
fn resolve_cache_root_rejects_empty_home() {
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _env_xdg = EnvVarGuard::remove("XDG_CACHE_HOME");
    let _env_home = EnvVarGuard::set("HOME", "");
    let err = resolve_cache_root().unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("HOME is set to the empty string"),
        "expected empty-HOME-specific rejection, got: {msg}"
    );
}

/// HOME unset (`Err(NotPresent)`) is rejected by a separate
/// arm of the shared validator — distinct from the empty-string
/// shape. The diagnostic names the unset case so an operator
/// debugging a container init that dropped HOME sees the actual
/// misconfiguration shape rather than a generic message that
/// conflates unset with empty.
#[test]
fn resolve_cache_root_rejects_unset_home() {
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _env_xdg = EnvVarGuard::remove("XDG_CACHE_HOME");
    let _env_home = EnvVarGuard::remove("HOME");
    let err = resolve_cache_root().unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("HOME is unset"),
        "expected unset-HOME-specific rejection, got: {msg}"
    );
    assert!(
        !msg.contains("HOME is set to the empty string"),
        "unset HOME must NOT use the empty-string diagnostic, got: {msg}",
    );
}

/// HOME=relative-path is rejected by the third arm of the
/// shared validation. Pin the model-cache resolver inherits
/// the same protection — a regression that bypassed the
/// shared helper would leave the model cache silently
/// resolving against CWD even though the kernel cache caught
/// the same shape.
#[test]
fn resolve_cache_root_rejects_relative_home() {
    let _lock = lock_env();
    let _env_ktstr = EnvVarGuard::remove("KTSTR_CACHE_DIR");
    let _env_xdg = EnvVarGuard::remove("XDG_CACHE_HOME");
    let _env_home = EnvVarGuard::set("HOME", "relative/dir");
    let err = resolve_cache_root().unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("not an absolute path"),
        "expected relative-path rejection, got: {msg}"
    );
    assert!(
        msg.contains("relative/dir"),
        "diagnostic must cite the offending HOME value, got: {msg}"
    );
}

/// Non-UTF-8 KTSTR_CACHE_DIR must bail with the actionable
/// diagnostic the shared validation surfaces. Pre-unification
/// (model.rs:806) the model resolver silently fell through on
/// `Err(VarError::NotUnicode)` and the operator's override
/// vanished without a trace; the shared helper now catches
/// this for both caches.
#[test]
#[cfg(unix)]
fn resolve_cache_root_rejects_non_utf8_ktstr_cache_dir() {
    let _lock = lock_env();
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let bytes: &[u8] = b"/tmp/ktstr-\xFFmodels";
    let value = OsStr::from_bytes(bytes);
    let _env_ktstr = EnvVarGuard::set("KTSTR_CACHE_DIR", value);
    let err = resolve_cache_root()
        .expect_err("non-UTF-8 KTSTR_CACHE_DIR must bail through the shared helper");
    let msg = err.to_string();
    assert!(
        msg.contains("KTSTR_CACHE_DIR"),
        "error must name the offending variable, got: {msg}",
    );
    assert!(
        msg.contains("non-UTF-8"),
        "error must mention non-UTF-8, got: {msg}",
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

/// Exactly `MAX_ENV_ECHO_LEN` bytes (64) must NOT trip the
/// truncation branch — the gate is `> 64`, not `>= 64`. Pins the
/// off-by-one so a future refactor that tightens to `>=` surfaces
/// here.
#[test]
fn sanitize_env_value_at_exact_cap_does_not_truncate() {
    let raw: String = "x".repeat(64);
    let out = sanitize_env_value(&raw);
    assert_eq!(out, raw, "64-byte input must pass through unchanged");
    assert!(
        !out.ends_with("..."),
        "64-byte input must not gain a truncation marker: {out:?}"
    );
}

/// A multi-byte UTF-8 codepoint straddling the byte cap must be
/// dropped whole, not split mid-sequence. 63 ASCII bytes plus
/// one `β` (2 UTF-8 bytes) totals 65 bytes, which trips the
/// truncation branch. The char_indices walk stops at the last
/// whole char whose end ≤ 64: 'x' #63 ends at byte 63, while
/// placing 'β' next would reach byte 65. So the prefix truncates
/// at byte 63, yielding 63 x's plus the `...` marker (66 bytes).
#[test]
fn sanitize_env_value_truncates_on_char_boundary_for_utf8_straddle() {
    let raw: String = format!("{}β", "x".repeat(63));
    assert_eq!(raw.len(), 65, "setup: input must be 65 bytes");
    let out = sanitize_env_value(&raw);
    assert_eq!(out.len(), 66, "63 truncated + 3 marker = 66 bytes");
    assert!(out.ends_with("..."), "marker missing: {out:?}");
    assert_eq!(&out[..63], &"x".repeat(63), "prefix must be 63 x's");
    assert!(
        !out.contains('β'),
        "straddling codepoint must be dropped whole: {out:?}"
    );
}

/// ensure()'s offline-bail error echoes the env value
/// through `sanitize_env_value`. Set `OFFLINE_ENV` to a value
/// containing both control chars and overlong content, and
/// check the error string contains neither a raw newline nor
/// the full 200-char payload.
#[test]
fn ensure_offline_error_sanitizes_env_value_in_message() {
    let _lock = lock_env();
    let _cache = isolated_cache_dir();
    // Embed a newline + a very long tail; both get rewritten.
    let hostile = format!("inject\nbreak{}", "z".repeat(200));
    let _env_offline = EnvVarGuard::set(OFFLINE_ENV, &hostile);
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

// -- mtime-size warm-cache sidecar helpers --

/// Pin the `.mtime-size` suffix derivation for the warm-cache
/// sidecar path. A future reshape of the naming scheme breaks
/// caches symmetrically across every ktstr invocation, so the
/// path is a hard-coded contract the test captures verbatim.
#[test]
fn mtime_size_sidecar_path_appends_suffix() {
    let artifact = std::path::Path::new("/tmp/model.gguf");
    assert_eq!(
        mtime_size_sidecar_path(artifact),
        std::path::PathBuf::from("/tmp/model.gguf.mtime-size"),
    );
    // No artifact extension — suffix appends to the bare name.
    let bare = std::path::Path::new("/tmp/model");
    assert_eq!(
        mtime_size_sidecar_path(bare),
        std::path::PathBuf::from("/tmp/model.mtime-size"),
    );
}

/// Round-trip: write the sidecar for an artifact, read it
/// back, get the same `(mtime_ns, size_bytes)` tuple that
/// metadata reports for the live file. Covers the happy path
/// that the warm-cache fast path relies on.
#[test]
fn write_then_read_mtime_size_sidecar_roundtrips() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"hello world").unwrap();

    write_mtime_size_sidecar(&artifact).expect("write must succeed");
    let meta = std::fs::metadata(&artifact).unwrap();
    let expected = mtime_size_from_metadata(&meta).unwrap();
    let read_back = read_mtime_size_sidecar(&artifact).expect("sidecar must read back");
    assert_eq!(
        read_back, expected,
        "round-trip must recover the (mtime, size) tuple written",
    );
}

/// `sidecar_confirms_prior_sha_match` returns `true` only
/// when the on-disk metadata matches the sidecar record. A
/// post-write touch that changes mtime must break the match
/// — this is the core semantic the fast path depends on.
#[test]
fn sidecar_confirms_match_tracks_mtime_change() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"contents").unwrap();
    write_mtime_size_sidecar(&artifact).expect("write must succeed");
    let meta = std::fs::metadata(&artifact).unwrap();
    assert!(
        sidecar_confirms_prior_sha_match(&artifact, &meta),
        "fresh sidecar must confirm match for unchanged file",
    );

    // Advance mtime by 2 seconds — enough to cross even the
    // coarsest filesystem's mtime granularity (most are
    // nanosecond, some tmpfs / older FAT are second).
    let meta_before = std::fs::metadata(&artifact).unwrap();
    let now = meta_before.modified().unwrap() + std::time::Duration::from_secs(2);
    filetime_set(&artifact, now);
    let meta_after = std::fs::metadata(&artifact).unwrap();
    assert!(
        !sidecar_confirms_prior_sha_match(&artifact, &meta_after),
        "mtime bump must invalidate the sidecar match so the \
         slow SHA path re-runs",
    );
}

/// Fallback path 1: missing sidecar → `None`. The fast path
/// must not trust absent state as a match; the slow path
/// re-runs SHA-256.
#[test]
fn read_mtime_size_sidecar_missing_file_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact-never-had-sidecar.bin");
    std::fs::write(&artifact, b"x").unwrap();
    // No write_mtime_size_sidecar call — sidecar never created.
    assert!(
        read_mtime_size_sidecar(&artifact).is_none(),
        "absent sidecar must return None, not silently default",
    );
}

/// Fallback path 2: sidecar file exists but is empty. A
/// zero-length file typically surfaces after a crash during
/// write — the kernel created the inode but the `write(2)`
/// payload never flushed. The magic-header gate rejects it.
#[test]
fn read_mtime_size_sidecar_empty_file_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"x").unwrap();
    // Plant an empty sidecar: simulates the zero-length
    // crash-truncation failure mode.
    std::fs::write(mtime_size_sidecar_path(&artifact), b"").unwrap();
    assert!(
        read_mtime_size_sidecar(&artifact).is_none(),
        "empty sidecar must fail the magic-header gate",
    );
}

/// Fallback path 3: sidecar carries only the magic line
/// (payload truncated mid-write). The second `lines.next()`
/// returns `None` and the helper falls through to None.
#[test]
fn read_mtime_size_sidecar_magic_only_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"x").unwrap();
    std::fs::write(
        mtime_size_sidecar_path(&artifact),
        format!("{MTIME_SIZE_SIDECAR_MAGIC}\n"),
    )
    .unwrap();
    assert!(
        read_mtime_size_sidecar(&artifact).is_none(),
        "sidecar missing the mtime/size payload must fail parse",
    );
}

/// Fallback path 4: wrong / older magic header. A v0 sidecar
/// that happened to carry a valid-looking `{mtime} {size}`
/// pair without a magic header must NOT deserialize as a v1
/// match; otherwise a schema bump would silently accept stale
/// data as fresh.
#[test]
fn read_mtime_size_sidecar_wrong_magic_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"x").unwrap();
    // Older-schema shape: `{mtime} {size}` on line 1, no magic.
    std::fs::write(mtime_size_sidecar_path(&artifact), b"12345 100\n").unwrap();
    assert!(
        read_mtime_size_sidecar(&artifact).is_none(),
        "sidecar missing the magic header must fail the version gate",
    );

    // A different magic (future v2) must also be rejected by
    // this v1 reader.
    std::fs::write(
        mtime_size_sidecar_path(&artifact),
        b"KTSTR_SHA_MTIME_SIZE_V2\n12345 100\n",
    )
    .unwrap();
    assert!(
        read_mtime_size_sidecar(&artifact).is_none(),
        "sidecar with a newer magic must fail the v1 gate",
    );
}

/// Fallback path 5: malformed payload line. Non-numeric
/// tokens or a single-token line fail the `.parse()` chain
/// and return None.
#[test]
fn read_mtime_size_sidecar_malformed_payload_returns_none() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"x").unwrap();
    // Magic + non-numeric mtime.
    std::fs::write(
        mtime_size_sidecar_path(&artifact),
        format!("{MTIME_SIZE_SIDECAR_MAGIC}\nnot-a-number 100\n"),
    )
    .unwrap();
    assert!(read_mtime_size_sidecar(&artifact).is_none());
    // Magic + single token (size missing).
    std::fs::write(
        mtime_size_sidecar_path(&artifact),
        format!("{MTIME_SIZE_SIDECAR_MAGIC}\n12345\n"),
    )
    .unwrap();
    assert!(read_mtime_size_sidecar(&artifact).is_none());
}

/// `remove_mtime_size_sidecar` unlinks an existing sidecar
/// and is silent when none exists — the post-mismatch
/// cleanup path must not error on a double-call or on a
/// cache entry that never wrote a sidecar (e.g. a
/// freshly-downloaded file that bailed before the
/// write-sidecar step).
#[test]
fn remove_mtime_size_sidecar_is_idempotent() {
    let tmp = tempfile::TempDir::new().unwrap();
    let artifact = tmp.path().join("artifact.bin");
    std::fs::write(&artifact, b"x").unwrap();
    write_mtime_size_sidecar(&artifact).unwrap();
    assert!(mtime_size_sidecar_path(&artifact).exists());
    remove_mtime_size_sidecar(&artifact);
    assert!(!mtime_size_sidecar_path(&artifact).exists());
    // Double-call: no-op, no panic.
    remove_mtime_size_sidecar(&artifact);
}

/// Set `path`'s mtime directly via libc — tmpfs / nextest
/// parallelism would make `std::thread::sleep(2s)` a flake
/// magnet, so use `utimes` to jump mtime forward without
/// wall-clock waits. Test-only; mirrors the similar helper in
/// sidecar.rs.
fn filetime_set(path: &std::path::Path, new_mtime: std::time::SystemTime) {
    use std::os::unix::ffi::OsStrExt;
    let secs = new_mtime
        .duration_since(std::time::UNIX_EPOCH)
        .expect("mtime before UNIX_EPOCH")
        .as_secs() as i64;
    let times = [
        libc::timeval {
            tv_sec: secs,
            tv_usec: 0,
        },
        libc::timeval {
            tv_sec: secs,
            tv_usec: 0,
        },
    ];
    let cstr = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    let rc = unsafe { libc::utimes(cstr.as_ptr(), times.as_ptr()) };
    assert_eq!(rc, 0, "utimes must succeed for the test helper");
}
