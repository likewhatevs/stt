//! [`CacheDir`] handle, lock guards, and cache-lock timeout policy.
//!
//! Public surface: [`CacheDir`] (the operator-facing handle exposed
//! via `crate::cache::CacheDir`), [`SharedLockGuard`] /
//! [`ExclusiveLockGuard`] (RAII wrappers around per-key flock
//! acquisitions), and the [`CacheDir::store`] /
//! [`CacheDir::lookup`] / [`CacheDir::list`] /
//! [`CacheDir::clean`] lifecycle methods. The internal
//! `warn_if_unstripped_vmlinux` and `should_warn_unstripped`
//! helpers gate a per-lookup warning on entries whose vmlinux
//! sidecar took the strip-failure fallback in
//! [`super::vmlinux_strip::strip_vmlinux_debug`].
//!
//! Sibling modules:
//! - [`super::metadata`] — pure types ([`KernelSource`],
//!   [`KernelMetadata`], [`CacheArtifacts`], [`KconfigStatus`],
//!   [`CacheEntry`], [`ListedEntry`]) plus the
//!   [`super::metadata::classify_corrupt_reason`] dispatcher and
//!   [`super::metadata::format_image_missing_reason`] helper that
//!   `list` uses to emit corrupt-entry reason strings.
//! - [`super::housekeeping`] — atomic-rename install primitives
//!   ([`super::housekeeping::atomic_swap_dirs`],
//!   [`super::housekeeping::TmpDirGuard`]), cache-key /
//!   filename validators, the JSON metadata reader
//!   ([`super::housekeeping::read_metadata`]), and the cross-PID
//!   orphan-tempdir sweep
//!   ([`super::housekeeping::clean_orphaned_tmp_dirs`]).
//! - [`super::vmlinux_strip`] — the ELF strip pipeline
//!   ([`super::vmlinux_strip::strip_vmlinux_debug`]) `store()`
//!   invokes when an artifact carries a vmlinux sidecar.
//! - [`super::resolve`] — env-cascade root resolution that
//!   `CacheDir::new` and `CacheDir::default_root` flow through.
//!
//! Reader/writer asymmetry: shared (reader) lock blocks 10 s — the
//! reader timeout is fixed and not operator-tunable. The exclusive
//! (writer) lock blocks 5 minutes by default but is the ONLY one
//! overridable, via the [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`]
//! environment variable. Writer must outlast every concurrent test
//! reader; reader bails fast on a stuck writer. See
//! [`SHARED_LOCK_DEFAULT_TIMEOUT`] and
//! [`STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT`] for the literal
//! durations and their rationale.
//!
//! Tests live in a sibling file `cache_dir_tests.rs`, pulled in
//! below via `#[path]` so they remain the `cache_dir::tests`
//! submodule. That preserves access to private items
//! (`lookup_silent`, `should_emit_unstripped_warn`,
//! `store_exclusive_lock_timeout`, the `STORE_EXCLUSIVE_LOCK_*`
//! constants) and `super::*` resolution; the split is purely a
//! file-size measure.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::Context;

use super::housekeeping::{
    TmpDirGuard, atomic_swap_dirs, clean_orphaned_tmp_dirs, read_metadata, validate_cache_key,
    validate_filename,
};
#[cfg(test)]
use super::metadata::KconfigStatus;
use super::metadata::{
    CacheArtifacts, CacheEntry, KernelMetadata, ListedEntry, format_image_missing_reason,
};
use super::resolve::resolve_cache_root;
use super::vmlinux_strip::strip_vmlinux_debug;
use super::{LOCK_DIR_NAME, TMP_DIR_PREFIX};
use crate::flock::{FlockMode, acquire_flock_with_timeout};

/// Default wall-clock timeout for [`CacheDir::acquire_shared_lock`].
const SHARED_LOCK_DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Default timeout for [`CacheDir::store`]'s internal `LOCK_EX`
/// acquire when [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] is unset.
///
/// 5 minutes covers a `store` peer's full critical section in the
/// worst case: under heavy parallelism N concurrent runners may
/// contend on the SAME `cache_key`, where the head writer holds
/// `LOCK_EX` while it copies the boot image, runs the 3-stage
/// vmlinux strip pipeline ([`super::vmlinux_strip::strip_vmlinux_debug`]),
/// writes `metadata.json`, and finishes the
/// [`super::housekeeping::atomic_swap_dirs`] swap. A real vmlinux
/// strip on a debug-symbol-rich build can spend tens of seconds
/// inside the strip pipeline alone, and stacking N peers in series
/// behind that producer scales the wait linearly. 60 s was tight
/// enough that 5–10 contending peers reliably timed out before
/// the head writer finished. The new 5-minute default leaves
/// headroom for ~50 contending peers behind a slow strip without
/// losing the "fail loud rather than block forever" property of a
/// finite timeout.
const STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(300);

/// Environment variable name that overrides
/// [`STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT`]. Parsed via
/// [`humantime::parse_duration`] so operators can tune with
/// human-readable units (`30s`, `2m`, `10min`, `1h`). An invalid
/// value falls back to the default with a `warn!` so a typo never
/// silently disables the lock — the operator can see the
/// fall-through in their tracing output and fix the setting.
const STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV: &str = "KTSTR_CACHE_STORE_LOCK_TIMEOUT";

/// Resolve the per-store `LOCK_EX` acquire timeout, honoring the
/// [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] override. Pure function so
/// tests can exercise the parse/fall-through branches without
/// driving a full `store()` cycle.
fn store_exclusive_lock_timeout() -> std::time::Duration {
    match std::env::var(STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV) {
        Ok(v) if !v.is_empty() => match humantime::parse_duration(&v) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    env = %STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV,
                    value = %v,
                    err = %e,
                    "invalid cache-store lock timeout env value; \
                     falling back to default timeout",
                );
                STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT
            }
        },
        _ => STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT,
    }
}

/// Handle to the kernel image cache directory.
#[derive(Debug)]
#[non_exhaustive]
pub struct CacheDir {
    root: PathBuf,
}

/// Process-level dedup set for the unstripped-vmlinux warning.
///
/// `lookup()` is the user-visible entry point and may be called many
/// times per CLI invocation against the same cache_key (for example,
/// a multi-kernel gauntlet does N lookups of the same stale entry
/// across its scenario fan-out). Without dedup, every lookup would
/// re-emit the strip-fallback warn — N copies of the same line drowns
/// out unrelated diagnostics. The set holds every cache_key for which
/// the warn has already fired in this process; on hit, the warn
/// helper skips re-emission.
///
/// `OnceLock` rather than `LazyLock` to keep the lazy init explicit.
/// The mutex is held only across an O(1) HashSet insert; contention
/// under realistic lookup fan-out is negligible.
fn warned_keys() -> &'static Mutex<HashSet<String>> {
    static SET: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    SET.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Pure dedup-gate logic for [`warn_if_unstripped_vmlinux`].
///
/// Returns `true` iff a fresh `tracing::warn!` should fire for this
/// entry: `should_warn_unstripped` accepts the entry AND the entry's
/// cache_key is being recorded in `set` for the first time. Returns
/// `false` if the entry does not need warning at all OR if the key
/// was already in the set (already-warned suppression).
///
/// Takes `&Mutex<HashSet<String>>` rather than reaching into the
/// process-wide [`warned_keys`] static so tests can drive the gate
/// against a fresh per-test mutex without polluting (or being
/// polluted by) the global set. Production callers pass
/// `warned_keys()`; the bool return decouples the side effect (the
/// `tracing::warn!`) from the decision so the latter is unit-testable.
fn should_emit_unstripped_warn(entry: &CacheEntry, set: &Mutex<HashSet<String>>) -> bool {
    if !should_warn_unstripped(entry) {
        return false;
    }
    let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
    guard.insert(entry.key.clone())
}

/// Emit a per-lookup warning when a cache entry was created with an
/// unstripped vmlinux.
///
/// **Once per cache_key per process.** A `static` HashSet (see
/// [`warned_keys`]) records every key for which the warn has already
/// fired; subsequent calls for the same key are silent. Suppression
/// covers callers that lookup the same stale entry repeatedly within
/// one CLI invocation (e.g. multi-kernel gauntlet). The dedup
/// decision is delegated to [`should_emit_unstripped_warn`], which
/// is independently unit-tested.
///
/// Uses [`tracing::warn!`] so the message routes through the same
/// observability pipeline as every other cache-layer diagnostic
/// (the cargo-ktstr binary's `tracing_subscriber::fmt` writes warns
/// to stderr; library consumers can subscribe a different layer).
/// `eprintln!` would bypass that pipeline and force every consumer
/// to live with raw-stderr output regardless of their tracing
/// configuration.
///
/// The mutex is held only across the O(1) HashSet insert inside
/// `should_emit_unstripped_warn`; the `tracing::warn!` macro fires
/// AFTER lock release so a slow tracing subscriber cannot serialise
/// concurrent lookups.
fn warn_if_unstripped_vmlinux(entry: &CacheEntry) {
    if should_emit_unstripped_warn(entry, warned_keys()) {
        tracing::warn!(
            cache_key = %entry.key,
            "cache: using unstripped vmlinux (strip failed on a prior build; \
             re-run with a clean cache to retry)",
        );
    }
}

/// Pure decision logic for [`warn_if_unstripped_vmlinux`].
pub(crate) fn should_warn_unstripped(entry: &CacheEntry) -> bool {
    entry.metadata.has_vmlinux() && !entry.metadata.vmlinux_stripped()
}

/// Whether the existing `cached` cache entry already satisfies a
/// caller's intent to `store` an artifact under the same cache key.
///
/// Pure decision logic for [`CacheDir::store`]'s in-lock re-lookup
/// (step 3 of the docs). When N concurrent peers race on the same
/// `cache_key` they all miss the pre-lock cache check, serialise
/// behind `LOCK_EX`, and would otherwise each repeat the head
/// writer's copy / strip / atomic-publish work. This predicate
/// answers the post-lock question: "is the head writer's output
/// byte-equivalent to what I'd publish?" If yes, the late peers
/// short-circuit — only the head writer pays the publish cost.
///
/// Compares only the metadata fields that drive the on-disk bytes
/// `store()` would write:
///
/// - `config_hash` (CRC32 of the final `.config`) — pins the
///   kernel image identity.
/// - `ktstr_kconfig_hash` (CRC32 of `ktstr.kconfig`) — kconfig
///   fragment that produced the build.
/// - `extra_kconfig_hash` (CRC32 of the user `--extra-kconfig`
///   fragment) — same.
/// - `caller_has_vmlinux` — whether the caller passed a vmlinux
///   sidecar in `CacheArtifacts`. This is the actual switch
///   `store()` keys on (it overwrites `metadata.has_vmlinux`
///   from the artifacts argument), so the predicate compares
///   against the artifacts shape, not the caller's metadata
///   field.
///
/// Excludes:
///
/// - `built_at` — wall-clock timestamp that drifts every build;
///   pinning it would break the early-return and serialise every
///   peer through a redundant publish.
/// - `version` — display-only string, not a byte-difference.
/// - `source` — acquire-time provenance (Tarball / Git / Local +
///   payload). Two peers may publish the same image under
///   different `source` payloads (e.g. one from a tarball mirror,
///   one from a git checkout) and still produce byte-equivalent
///   bytes. The kconfig hash is the authoritative content key.
/// - `arch`, `image_name` — fixed by the cache key shape.
/// - `vmlinux_stripped` — set by `store()` based on
///   strip pipeline success/failure, not caller intent. The head
///   writer either succeeded (stripped) or fell back (unstripped);
///   late peers would just observe the head writer's outcome.
/// - `source_vmlinux_size`, `source_vmlinux_mtime_secs` —
///   DWARF-routing hints, not cached content.
///
/// Pure function so a unit test can pin every accept/reject branch
/// without driving a full `store()` cycle through a temp cache.
pub(crate) fn cache_content_matches(
    cached: &KernelMetadata,
    caller: &KernelMetadata,
    caller_has_vmlinux: bool,
) -> bool {
    cached.config_hash == caller.config_hash
        && cached.ktstr_kconfig_hash == caller.ktstr_kconfig_hash
        && cached.extra_kconfig_hash == caller.extra_kconfig_hash
        && cached.has_vmlinux() == caller_has_vmlinux
}

impl CacheDir {
    /// Open a cache directory at the resolved root path.
    pub fn new() -> anyhow::Result<Self> {
        let root = resolve_cache_root()?;
        Ok(CacheDir { root })
    }

    /// Open a cache directory at a specific path.
    pub fn with_root(root: PathBuf) -> Self {
        CacheDir { root }
    }

    /// Resolve the default cache root path without side effects.
    pub fn default_root() -> anyhow::Result<PathBuf> {
        resolve_cache_root()
    }

    /// Root directory this `CacheDir` is anchored at.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Look up a cached kernel by cache key.
    ///
    /// On hit, emits a `tracing::warn!` via
    /// [`warn_if_unstripped_vmlinux`] when the cached entry took the
    /// strip-failure fallback (see [`should_warn_unstripped`] for the
    /// exact predicate). Caller-facing call sites want the warning;
    /// internal call sites that look the entry up only to compare
    /// against caller intent (notably [`Self::store`]'s in-lock
    /// recheck) use [`Self::lookup_silent`] to avoid double-emitting
    /// the same warning the caller will see on its next `lookup`.
    pub fn lookup(&self, cache_key: &str) -> Option<CacheEntry> {
        let entry = self.lookup_silent(cache_key)?;
        warn_if_unstripped_vmlinux(&entry);
        Some(entry)
    }

    /// Look up a cached kernel without emitting the unstripped-vmlinux
    /// warning. Internal callers that consume the entry's metadata
    /// without surfacing it to the user — specifically the in-lock
    /// recheck inside [`Self::store`] — use this variant so a recheck
    /// hit on a strip-fallback entry does not log a duplicate warning
    /// that the user-facing [`Self::lookup`] will already log on their
    /// next call.
    fn lookup_silent(&self, cache_key: &str) -> Option<CacheEntry> {
        if let Err(e) = validate_cache_key(cache_key) {
            tracing::warn!("invalid cache key: {e}");
            return None;
        }
        let entry_dir = self.root.join(cache_key);
        if !entry_dir.is_dir() {
            return None;
        }
        let metadata = read_metadata(&entry_dir).ok()?;
        if !entry_dir.join(&metadata.image_name).exists() {
            return None;
        }
        Some(CacheEntry {
            key: cache_key.to_string(),
            path: entry_dir,
            metadata,
        })
    }

    /// List all cached kernel entries, sorted by build time (newest
    /// first).
    pub fn list(&self) -> anyhow::Result<Vec<ListedEntry>> {
        let mut entries: Vec<ListedEntry> = Vec::new();
        let read_dir = match fs::read_dir(&self.root) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
            Err(e) => return Err(e.into()),
        };
        for dir_entry in read_dir {
            let dir_entry = dir_entry?;
            let path = dir_entry.path();
            let name = match dir_entry.file_name().into_string() {
                Ok(n) => n,
                Err(_) => continue,
            };
            // Skip every dotfile child — ktstr reserves all
            // dot-prefixed names (current uses: `.locks/`, `.tmp-*`).
            // `validate_cache_key` rejects leading-dot inputs, so a
            // dotfile in the cache root is either ktstr bookkeeping or
            // an external artifact; either way `list()` must not
            // surface it as a cache entry.
            if name.starts_with('.') {
                continue;
            }
            if !path.is_dir() {
                continue;
            }
            match read_metadata(&path) {
                Ok(metadata) => {
                    let image_path = path.join(&metadata.image_name);
                    if image_path.exists() {
                        entries.push(ListedEntry::Valid(Box::new(CacheEntry {
                            key: name,
                            path,
                            metadata,
                        })));
                    } else {
                        entries.push(ListedEntry::Corrupt {
                            key: name,
                            path,
                            reason: format_image_missing_reason(&metadata.image_name),
                        });
                    }
                }
                Err(reason) => {
                    tracing::info!(
                        entry = %name,
                        path = %path.display(),
                        %reason,
                        "cache entry corrupt at list-time",
                    );
                    entries.push(ListedEntry::Corrupt {
                        key: name,
                        path,
                        reason,
                    });
                }
            }
        }
        entries.sort_by(|a, b| {
            let a_time = a.as_valid().map(|e| e.metadata.built_at.as_str());
            let b_time = b.as_valid().map(|e| e.metadata.built_at.as_str());
            b_time.cmp(&a_time)
        });
        Ok(entries)
    }

    /// Store a kernel image (and optional vmlinux sidecar) in the
    /// cache under `cache_key`. Atomic install via temp directory +
    /// `renameat2(RENAME_EXCHANGE)`, so a concurrent reader never
    /// observes a partially-written entry.
    ///
    /// # Steps (in order)
    ///
    /// 1. **Validate inputs.** [`validate_cache_key`] rejects
    ///    `..`, slashes, NUL, leading-dot keys (the `TMP_DIR_PREFIX`
    ///    reservation plus any other dotfile-shaped key, since
    ///    `list()` skips every dotfile child);
    ///    [`validate_filename`] rejects path-separator characters in
    ///    the image basename. Invalid input fails before any I/O.
    /// 2. **Acquire the per-key store lock.** `LOCK_EX` on
    ///    `<root>/.locks/<cache_key>.lock`. Timeout defaults to
    ///    [`STORE_EXCLUSIVE_LOCK_DEFAULT_TIMEOUT`] (5 minutes) and
    ///    can be overridden via [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`]
    ///    for environments where a slow vmlinux strip stacks many
    ///    contending peers behind the head writer. The lock
    ///    excludes other writers for the same key while letting
    ///    readers and writers for unrelated keys proceed. Timeout
    ///    produces an error rather than blocking forever — a hung
    ///    writer cannot indefinitely block a fresh rebuild attempt.
    /// 3. **Double-checked re-lookup inside the lock.** After
    ///    acquiring `LOCK_EX`, re-run [`Self::lookup_silent`] for
    ///    `cache_key`. When N peers race to publish the same key
    ///    they all miss the pre-lock cache check, queue on
    ///    `LOCK_EX`, and serialise behind the head writer. Without
    ///    this recheck, every peer re-runs the full copy + strip +
    ///    publish steps in series even though the head writer's
    ///    output already satisfies them. The recheck early-returns
    ///    when the existing cached entry's content-defining metadata
    ///    fields ([`cache_content_matches`] — config_hash,
    ///    ktstr_kconfig_hash, extra_kconfig_hash, has_vmlinux) match
    ///    the caller's intent for this publish, so only the head
    ///    writer pays the strip/copy/rename cost. Cache-relevant
    ///    differences (a fresh kconfig hash, a different vmlinux
    ///    presence) bypass the early-return and proceed to a real
    ///    overwrite-publish. Cache-irrelevant differences (a fresh
    ///    `built_at` timestamp, a different `version` display
    ///    string) trigger the early-return — the on-disk bytes the
    ///    overwrite would write are byte-equivalent to what's
    ///    already cached, so the publish is redundant.
    /// 4. **Stage into a temp directory.** `<root>/.tmp-<key>-<pid>`
    ///    is created (or pruned and recreated if a previous attempt
    ///    by the same PID exists), with [`TmpDirGuard`] enrolling the
    ///    path for cleanup on any subsequent error. A best-effort
    ///    [`clean_orphaned_tmp_dirs`] pass also runs here so dead
    ///    sibling temp directories from crashed PIDs are GC'd before
    ///    we add another one.
    /// 5. **Copy the boot image.** `metadata.image_name` lands at
    ///    `tmp/<image_name>` via `fs::copy`.
    /// 6. **Strip and copy vmlinux (if supplied).** When
    ///    `artifacts.vmlinux` is `Some`, [`strip_vmlinux_debug`]
    ///    runs the 3-stage strip pipeline and the result is written
    ///    to `tmp/vmlinux`. **Strip-fallback rationale:** if the
    ///    strip pipeline returns an error (e.g. an unrecognised ELF
    ///    layout from a future toolchain or an exotic config), the
    ///    write does NOT abort — it falls back to copying the raw
    ///    unstripped vmlinux and records `vmlinux_stripped: false`
    ///    in metadata. The cache trades a much larger on-disk
    ///    payload for "still usable for monitoring/probes," and
    ///    `cargo ktstr kernel list --json` exposes the
    ///    `vmlinux_stripped` field so operators can spot entries
    ///    that need rebuilding once the strip-failure root cause is
    ///    fixed. A hard failure here would be worse: it would
    ///    effectively brick the cache for that build.
    /// 7. **Write `metadata.json`.** A pretty-printed serde dump of
    ///    `KernelMetadata` (with `has_vmlinux` and `vmlinux_stripped`
    ///    set from step 6) at `tmp/metadata.json`. Pretty-print is
    ///    intentional — operators inspect this file directly when
    ///    debugging cache state.
    /// 8. **Atomic publish.** `fs::rename(tmp → final)` if `final`
    ///    does not exist; otherwise [`atomic_swap_dirs`] uses
    ///    `renameat2(RENAME_EXCHANGE)` to swap the two directories
    ///    in a single atomic syscall. Either way, no reader observes
    ///    a partial entry; the swap path also cleans up the
    ///    now-stale prior version under the temp name.
    pub fn store(
        &self,
        cache_key: &str,
        artifacts: &CacheArtifacts<'_>,
        metadata: &KernelMetadata,
    ) -> anyhow::Result<CacheEntry> {
        validate_cache_key(cache_key)?;
        validate_filename(&metadata.image_name)?;

        let _store_lock =
            self.acquire_exclusive_lock_blocking(cache_key, store_exclusive_lock_timeout())?;

        // Double-checked re-lookup inside LOCK_EX: when N peers race
        // on the same cache_key they all miss the pre-lock cache
        // check, queue on the lock, and would otherwise repeat the
        // head writer's copy/strip/publish work in series. The
        // recheck early-returns when the existing entry's
        // content-defining metadata fields match what we'd publish
        // (see [`cache_content_matches`] for the predicate). The
        // matched entry is returned to the caller verbatim — its
        // on-disk bytes are byte-equivalent to what we would write,
        // so no overwrite-publish is needed.
        //
        // The recheck-hit early-return BYPASSES the orphan tempdir
        // sweep at step 4. That is intentional: every orphan-sweep
        // call costs an opendir + readdir + N kill(pid, 0) probes,
        // and the recheck-hit path is the hot path for serialised
        // peer fan-out — adding the sweep here would charge every
        // late peer a syscall budget the head writer already paid.
        // Orphans accumulate only on the cache-miss / overwrite
        // path, which is also where new tempdirs are created, so
        // the GC runs proportionally to tempdir creation. Uses the
        // private `lookup_silent` variant (no warn) so the recheck
        // does not double-emit the unstripped-vmlinux warn that
        // store()'s caller would see again on its next lookup().
        if let Some(existing) = self.lookup_silent(cache_key)
            && cache_content_matches(&existing.metadata, metadata, artifacts.vmlinux.is_some())
        {
            tracing::debug!(
                cache_key = cache_key,
                "cache.store: in-lock recheck hit; skipping copy/strip/publish",
            );
            return Ok(existing);
        }

        let final_dir = self.root.join(cache_key);
        let tmp_dir = self.root.join(format!(
            "{TMP_DIR_PREFIX}{}-{}",
            cache_key,
            std::process::id(),
        ));

        if tmp_dir.exists() {
            fs::remove_dir_all(&tmp_dir)?;
        }
        if let Err(e) = clean_orphaned_tmp_dirs(&self.root) {
            tracing::warn!(err = %format!("{e:#}"), "clean_orphaned_tmp_dirs failed; continuing store");
        }
        fs::create_dir_all(&tmp_dir)?;

        let _guard = TmpDirGuard(&tmp_dir);

        let image_dest = tmp_dir.join(&metadata.image_name);
        fs::copy(artifacts.image, &image_dest)
            .map_err(|e| anyhow::anyhow!("copy kernel image to cache: {e}"))?;

        let (has_vmlinux, vmlinux_stripped) = if let Some(vmlinux) = artifacts.vmlinux {
            let vmlinux_dest = tmp_dir.join("vmlinux");
            match strip_vmlinux_debug(vmlinux) {
                Ok(stripped) => {
                    fs::copy(stripped.path(), &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy stripped vmlinux to cache: {e}"))?;
                    (true, true)
                }
                Err(e) => {
                    tracing::warn!(
                        cache_key = cache_key,
                        err = %format!("{e:#}"),
                        "vmlinux strip failed, caching unstripped \
                         (larger on-disk payload). See \
                         `cargo ktstr kernel list --json` \
                         vmlinux_stripped field.",
                    );
                    fs::copy(vmlinux, &vmlinux_dest)
                        .map_err(|e| anyhow::anyhow!("copy vmlinux to cache: {e}"))?;
                    (true, false)
                }
            }
        } else {
            (false, false)
        };

        let mut meta = metadata.clone();
        meta.set_has_vmlinux(has_vmlinux);
        meta.set_vmlinux_stripped(vmlinux_stripped);
        let meta_json = serde_json::to_string_pretty(&meta)?;
        fs::write(tmp_dir.join("metadata.json"), meta_json)
            .map_err(|e| anyhow::anyhow!("write cache metadata: {e}"))?;

        match fs::rename(&tmp_dir, &final_dir) {
            Ok(()) => {}
            Err(e)
                if e.raw_os_error() == Some(libc::ENOTEMPTY)
                    || e.raw_os_error() == Some(libc::EEXIST) =>
            {
                atomic_swap_dirs(&tmp_dir, &final_dir)?;
            }
            Err(e) => {
                return Err(anyhow::anyhow!("atomic rename cache entry: {e}"));
            }
        }

        Ok(CacheEntry {
            key: cache_key.to_string(),
            path: final_dir,
            metadata: meta,
        })
    }

    /// Remove every cached entry. Returns the number of entries
    /// removed. Preserves the `.locks/` subdirectory.
    pub fn clean_all(&self) -> anyhow::Result<usize> {
        self.remove_entries(self.list()?)
    }

    /// Remove every cached entry except the `keep` most recent ones
    /// (by `built_at` timestamp). Preserves the `.locks/`
    /// subdirectory.
    pub fn clean_keep(&self, keep: usize) -> anyhow::Result<usize> {
        self.remove_entries(self.list()?.into_iter().skip(keep))
    }

    fn remove_entries<I: IntoIterator<Item = ListedEntry>>(
        &self,
        iter: I,
    ) -> anyhow::Result<usize> {
        let to_remove: Vec<_> = iter.into_iter().collect();
        let count = to_remove.len();
        for entry in &to_remove {
            fs::remove_dir_all(entry.path())?;
        }
        Ok(count)
    }

    // ---------------- Per-entry coordination locks ----------------

    /// Absolute path to the coordination lockfile for `cache_key`.
    pub(crate) fn lock_path(&self, cache_key: &str) -> PathBuf {
        self.root
            .join(LOCK_DIR_NAME)
            .join(format!("{cache_key}.lock"))
    }

    /// Create the `{cache_root}/.locks/` subdirectory if absent.
    pub(crate) fn ensure_lock_dir(&self) -> anyhow::Result<()> {
        let dir = self.root.join(LOCK_DIR_NAME);
        fs::create_dir_all(&dir)
            .with_context(|| format!("create lock subdirectory {}", dir.display()))
    }

    /// Acquire `LOCK_SH` on the cache-entry lockfile.
    pub fn acquire_shared_lock(&self, cache_key: &str) -> anyhow::Result<SharedLockGuard> {
        validate_cache_key(cache_key)?;
        let path = self.lock_path(cache_key);
        let fd = acquire_flock_with_timeout(
            &path,
            FlockMode::Shared,
            SHARED_LOCK_DEFAULT_TIMEOUT,
            &format!("cache entry {cache_key:?}"),
            None,
        )?;
        Ok(SharedLockGuard { fd })
    }

    /// Acquire `LOCK_EX` on the cache-entry lockfile, blocking up
    /// to `timeout`. On timeout, the error message surfaces the
    /// [`STORE_EXCLUSIVE_LOCK_TIMEOUT_ENV`] override so an operator
    /// hitting a contended `store()` discovers the env-var
    /// remediation without reading the docs.
    pub fn acquire_exclusive_lock_blocking(
        &self,
        cache_key: &str,
        timeout: std::time::Duration,
    ) -> anyhow::Result<ExclusiveLockGuard> {
        validate_cache_key(cache_key)?;
        let path = self.lock_path(cache_key);
        let fd = acquire_flock_with_timeout(
            &path,
            FlockMode::Exclusive,
            timeout,
            &format!("cache entry {cache_key:?}"),
            Some(
                "override the timeout via KTSTR_CACHE_STORE_LOCK_TIMEOUT (humantime: 30s, 2m, 1h)",
            ),
        )?;
        Ok(ExclusiveLockGuard { fd })
    }

    /// Non-blocking `LOCK_EX` attempt on the cache-entry lockfile.
    pub fn try_acquire_exclusive_lock(
        &self,
        cache_key: &str,
    ) -> anyhow::Result<ExclusiveLockGuard> {
        validate_cache_key(cache_key)?;
        // try_flock doesn't lazily create the parent directory like
        // acquire_flock_with_timeout does — must materialise .locks/
        // here so the open(O_CREAT) inside try_flock has a parent.
        self.ensure_lock_dir()?;
        let path = self.lock_path(cache_key);
        match crate::flock::try_flock(&path, crate::flock::FlockMode::Exclusive)? {
            Some(fd) => Ok(ExclusiveLockGuard { fd }),
            None => {
                let holders = crate::flock::read_holders(&path).unwrap_or_default();
                anyhow::bail!(
                    "cache entry {cache_key:?} is locked by active test runs \
                     (lockfile {lockfile}, holders: {holders}). Wait for \
                     those tests to finish, or kill them, then retry.",
                    lockfile = path.display(),
                    holders = crate::flock::format_holder_list(&holders),
                );
            }
        }
    }
}

/// RAII guard for a `LOCK_SH` hold on a cache-entry lockfile.
#[derive(Debug)]
pub struct SharedLockGuard {
    #[allow(dead_code)]
    fd: std::os::fd::OwnedFd,
}

/// RAII guard for a `LOCK_EX` hold on a cache-entry lockfile.
#[derive(Debug)]
pub struct ExclusiveLockGuard {
    #[allow(dead_code)]
    fd: std::os::fd::OwnedFd,
}

#[cfg(test)]
#[path = "cache_dir_tests.rs"]
mod tests;
