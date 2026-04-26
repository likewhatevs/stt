//! Kernel source acquisition: tarball download, git clone, local tree.
//!
//! Three entry points — [`download_tarball`], [`git_clone`], and
//! [`local_source`] — each return an [`AcquiredSource`] carrying the
//! source directory, cache key, and metadata the caller needs to
//! proceed to configuration and build.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;

/// Process-wide [`reqwest::blocking::Client`] lazily initialized on
/// first access via [`shared_client`]. Keeping a single `Client`
/// instance across the fetch-family reuses its TCP connection pool
/// and TLS session cache across repeated calls to the same host
/// within a CLI run. Cross-host fetches in the same run still
/// re-handshake because reqwest's connection pool keys on host.
static SHARED_CLIENT: OnceLock<Client> = OnceLock::new();

/// Return the process-wide shared [`reqwest::blocking::Client`]. First
/// call constructs it with `reqwest::blocking::Client::new()`; every
/// subsequent call returns a reference to the same instance. Call sites
/// that need fault-injection seams (httpmock-style tests) should
/// construct a local `Client` directly and pass it to `fetch_*`; this
/// helper is for top-level CLI entries that want the default client.
///
/// Tests that need to verify a network round-trip (rather than a
/// cache hit) must NOT pass `shared_client()` to a cache-routed
/// helper (`cached_releases`, `cached_releases_with`,
/// [`fetch_latest_stable_version`], [`fetch_version_for_prefix`]) —
/// [`RELEASES_CACHE`] may already be populated by a peer test, in
/// which case the helper returns cached data and the network is
/// never touched. Construct a local `Client` instead; the
/// pointer-equality gate in [`cached_releases_with`] routes it to
/// the bypass branch.
///
/// # Panics
///
/// Panics on the first call if `reqwest::blocking::Client::new()` fails
/// to build a default client — inherited behavior from reqwest, which
/// uses it as the infallible constructor. The documented failure modes
/// are TLS backend initialization (e.g. rustls/native-tls subsystem
/// unreachable) and are treated as setup bugs rather than runtime
/// errors; a failing first call would have failed just as hard under
/// the pre-singleton `Client::new()` per-callsite pattern.
pub fn shared_client() -> &'static Client {
    SHARED_CLIENT.get_or_init(Client::new)
}

/// Process-wide cache of the parsed `releases.json` payload.
/// Populated by [`cached_releases_with`] on its first successful
/// singleton-path fetch; every subsequent singleton call returns a
/// clone of the cached vector without re-issuing the HTTP request.
/// Lifetime matches the process — `releases.json` does not change
/// underneath a single CLI invocation, so a per-process cache
/// cannot serve stale data in any way the user would notice.
///
/// Failures are NOT cached: a transient kernel.org outage that
/// errors the first call must allow a later caller to retry, since
/// the underlying network condition may have cleared. Storing
/// `Vec<Release>` rather than `Result<Vec<Release>>` enforces this
/// at the type level — there's no way to populate the cache with
/// a failure.
///
/// Companion to [`SHARED_CLIENT`]: both amortize per-invocation
/// network cost across the resolve pipeline. Without this cache,
/// `cargo ktstr test --kernel 6.10..6.12 --kernel 6.14..6.16`
/// fetches `releases.json` twice — once per Range spec — under
/// the rayon par_iter that drives `resolve_kernel_set`. With
/// the cache the first Range to reach `expand_kernel_range`
/// populates the slot; the second observes the populated slot
/// and skips the network entirely.
static RELEASES_CACHE: OnceLock<Vec<Release>> = OnceLock::new();

/// Fetch `releases.json` via the process-wide [`shared_client`],
/// routing through [`RELEASES_CACHE`].
///
/// Thin wrapper for callers that don't already thread a `&Client`
/// — top-level CLI entries like [`crate::cli::expand_kernel_range`]
/// (under the rayon-driven `cargo ktstr` resolve pipeline) and
/// [`crate::cli::fetch_active_prefixes`] (the EOL-annotation pass).
/// Caching, race semantics, and fault-injection routing are all
/// documented on [`cached_releases_with`].
pub(crate) fn cached_releases() -> Result<Vec<Release>> {
    cached_releases_with(shared_client())
}

/// Pointer-equality against the [`OnceLock`]-backed
/// [`shared_client`] singleton is the correct predicate because
/// `shared_client()` returns a stable `&'static Client` address.
/// The [`cached_releases_with`] gate uses this predicate to
/// decide whether to consult [`RELEASES_CACHE`]: the singleton
/// hits the cache, every other (test-constructed) `Client`
/// bypasses it and exercises the underlying [`fetch_releases`]
/// path.
///
/// Caveat: `shared_client().clone()` produces a distinct
/// `Client` at a different address even though it shares the
/// singleton's connection pool via the inner `Arc`, so the
/// clone bypasses the cache. Always pass `shared_client()`
/// directly — never a clone — when cache routing is desired.
///
/// Side-effect-free when [`SHARED_CLIENT`] is uninitialized:
/// no client can equal a not-yet-allocated singleton, so we
/// return `false` without triggering `get_or_init` — tests
/// that pass a local `Client` before any production code path
/// has touched the singleton skip the construction entirely.
fn is_shared_client(client: &Client) -> bool {
    match SHARED_CLIENT.get() {
        Some(singleton) => std::ptr::eq(client, singleton),
        None => false,
    }
}

/// Unified cache-aware entry point for `releases.json`. Routes
/// the process-wide [`shared_client`] singleton through
/// [`RELEASES_CACHE`]; any other (test-constructed) `Client`
/// falls through to a direct [`fetch_releases`] call against
/// whatever endpoint that `Client` is configured for (e.g. an
/// httpmock-style local mock).
///
/// Used by every in-file caller that already threads a `&Client`
/// — [`fetch_latest_stable_version`], [`fetch_version_for_prefix`],
/// [`latest_in_series`] — so production callers reuse
/// [`RELEASES_CACHE`] and tests still get fault-injection via
/// the pointer-equality gate. [`cached_releases`] is the
/// no-`Client` wrapper for top-level CLI entries.
///
/// Failures are propagated without populating [`RELEASES_CACHE`],
/// so a transient kernel.org outage on the first call lets the
/// next caller retry. Storing `Vec<Release>` (not
/// `Result<Vec<Release>>`) enforces this at the type level.
///
/// Concurrent population on the singleton path is safe via the
/// `OnceLock::set` race: the loser's `set` returns `Err(clone)`
/// (the cloned vector that was passed in is moved back), the
/// returned `Err` is discarded via `let _ = …`, and the loser
/// returns its own original `fresh` vector. Both winner and
/// loser return content-equivalent data since both fetched the
/// same `releases.json`. Worst case under concurrent first
/// calls: both callers issue the network round-trip, only one
/// populates [`RELEASES_CACHE`]; every later call — from any
/// thread — observes the populated slot via the `get` fast-path
/// and skips the network.
fn cached_releases_with(client: &Client) -> Result<Vec<Release>> {
    // Non-singleton clients bypass the cache (test fault injection).
    if !is_shared_client(client) {
        return fetch_releases(client);
    }
    if let Some(cached) = RELEASES_CACHE.get() {
        return Ok(cached.clone());
    }
    let fresh = fetch_releases(client)?;
    // Race-loss: `set` returns `Err(clone)` carrying back the
    // clone we passed in; we discard it and return the original
    // `fresh` below. See the rustdoc above for full semantics.
    let _ = RELEASES_CACHE.set(fresh.clone());
    Ok(fresh)
}

/// Downloaded/cloned kernel source ready for building.
#[non_exhaustive]
pub struct AcquiredSource {
    /// Path to the kernel source directory.
    pub source_dir: PathBuf,
    /// Cache key for this source (e.g. "6.14.2-tarball-x86_64-kc{kconfig_hash}").
    pub cache_key: String,
    /// Version string if known (e.g. "6.14.2", "6.15-rc3").
    pub version: Option<String>,
    /// How the source was acquired, with per-variant payload
    /// (git hash/ref for `Git`, source tree path and git hash for
    /// `Local`).
    pub kernel_source: crate::cache::KernelSource,
    /// Whether the source is a temporary directory that should be
    /// cleaned up after building.
    pub is_temp: bool,
    /// For local sources: whether the working tree is dirty.
    /// Dirty trees must not be cached.
    pub is_dirty: bool,
    /// For local sources: whether the source is an actual git
    /// repository. `true` when `gix::discover` succeeded and the
    /// crate could compute index + worktree dirty state; `false`
    /// for non-git source trees (tarball-extracted, rsync'd,
    /// hand-assembled) where dirty detection is impossible and
    /// the source is always cache-skipped pessimistically. Lets
    /// the cache-skip hint branch on whether `commit` / `stash`
    /// are actionable remediations (they aren't for non-git
    /// sources).
    ///
    /// For non-local sources (tarball, git clone) the field is
    /// set to `true` by convention — these paths are always
    /// `is_dirty = false`, so the cache-skip branch that reads
    /// `is_git` is never reached and the value is inert. Pinning
    /// to `true` (rather than leaving the field meaningless)
    /// keeps the invariant "is_git is meaningful only when
    /// is_dirty is true, but always set" so a future code path
    /// that reaches `is_git` outside the cache-skip context does
    /// not trip on an `is_git = false` under a known-good source.
    pub is_git: bool,
}

/// Target architecture string and boot image name.
pub fn arch_info() -> (&'static str, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        ("x86_64", "bzImage")
    }
    #[cfg(target_arch = "aarch64")]
    {
        ("aarch64", "Image")
    }
}

/// Parse a version string into its major version for URL construction.
///
/// "6.14.2" -> 6, "6.15-rc3" -> 6.
fn major_version(version: &str) -> Result<u32> {
    let major_str = version
        .split('.')
        .next()
        .ok_or_else(|| anyhow!("invalid version: {version}"))?;
    major_str
        .parse::<u32>()
        .with_context(|| format!("invalid major version in {version}"))
}

/// Determine if a version string represents an RC release.
///
/// RC releases use a different URL pattern and gzip compression
/// (vs xz for stable).
fn is_rc(version: &str) -> bool {
    version.contains("-rc")
}

/// One (`moniker`, `version`) row from kernel.org's `releases.json`.
///
/// A named struct instead of a bare `(String, String)` tuple so every
/// call site reads its field by name (`r.moniker`, `r.version`) rather
/// than positional destructuring — the two strings are trivially
/// swappable at a tuple-destructure call site, and a silent swap
/// would mis-drive `is_skippable_release_moniker` while the
/// now-misnamed "moniker" string flows into `version_prefix`
/// downstream. Naming the fields removes that class of bug at the
/// type-checker level and shows up in IDE hints on every iteration
/// site.
///
/// Both fields are owned `String` (not `&str`) because the values are
/// parsed out of a `reqwest::Response` body whose lifetime ends when
/// `fetch_releases` returns; downstream callers iterate the vector
/// long after that borrow would dangle.
#[derive(Clone, Debug)]
pub(crate) struct Release {
    /// releases.json `moniker` field — stable / longterm / mainline /
    /// linux-next / etc. Consumed by
    /// [`is_skippable_release_moniker`] and by
    /// [`fetch_latest_stable_version`]'s stable/longterm filter.
    pub moniker: String,
    /// releases.json `version` field — e.g. `"6.14.2"`, `"6.15-rc3"`,
    /// `"6.16-rc2-next-20260420"`. Consumed by
    /// [`version_tuple`], [`patch_level`], and
    /// `cli::version_prefix`.
    pub version: String,
}

/// Is this releases.json moniker one that the version-resolution
/// pipeline should skip?
///
/// `linux-next` is a rolling integration branch whose version strings
/// carry a date suffix rather than a stable tag, so it does not fit
/// the major.minor.patch resolution model used by `latest_in_series`,
/// `fetch_version_for_prefix`, and `cli::fetch_active_prefixes`. The
/// release iteration in all three sites filters it out; this helper
/// is the single point of truth for that decision so a future moniker
/// that also warrants skipping can be added in one place.
pub(crate) fn is_skippable_release_moniker(moniker: &str) -> bool {
    moniker == "linux-next"
}

/// Find the latest version in the same major.minor series from releases.json.
///
/// Returns `Some("6.14.10")` for prefix `"6.14"` if that series exists in
/// releases.json. Returns `None` if the series is not found (EOL or invalid).
fn latest_in_series(client: &Client, version: &str) -> Option<String> {
    let prefix = {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() >= 2 {
            format!("{}.{}", parts[0], parts[1])
        } else {
            return None;
        }
    };

    // Routes through [`RELEASES_CACHE`] for the singleton; see
    // [`cached_releases_with`] for the bypass gate.
    let releases = cached_releases_with(client).ok()?;
    let mut best: Option<(String, (u32, u32, u32))> = None;
    for r in &releases {
        if is_skippable_release_moniker(&r.moniker) {
            continue;
        }
        if !r.version.starts_with(&prefix) {
            continue;
        }
        if r.version.len() != prefix.len() && r.version.as_bytes()[prefix.len()] != b'.' {
            continue;
        }
        if let Some(tuple) = version_tuple(&r.version)
            && (best.is_none() || tuple > best.as_ref().unwrap().1)
        {
            best = Some((r.version.clone(), tuple));
        }
    }
    best.map(|(v, _)| v)
}

/// Build a user-facing error message for a version that was not found.
///
/// Suggests the latest version in the same major.minor series when
/// releases.json contains one.
fn version_not_found_msg(client: &Client, version: &str) -> String {
    let parts: Vec<&str> = version.split('.').collect();
    let prefix = if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        version.to_string()
    };
    match latest_in_series(client, version) {
        Some(latest) if latest != version => {
            format!("version {version} not found. latest {prefix}.x: {latest}")
        }
        _ => format!("version {version} not found"),
    }
}

/// Reject responses where the server returned HTML instead of a binary
/// archive. Some CDN error pages return 200 with text/html.
fn reject_html_response(response: &reqwest::blocking::Response, url: &str) -> Result<()> {
    if let Some(ct) = response.headers().get(reqwest::header::CONTENT_TYPE)
        && let Ok(ct_str) = ct.to_str()
        && ct_str.contains("text/html")
    {
        anyhow::bail!(
            "download {url}: server returned HTML instead of tarball (URL may be invalid)"
        );
    }
    Ok(())
}

/// Print download size from Content-Length header if available.
///
/// `cli_label` prefixes the diagnostic line so the message matches the
/// binary the user invoked (`"ktstr"` vs `"cargo ktstr"`).
fn print_download_size(response: &reqwest::blocking::Response, url: &str, cli_label: &str) {
    if let Some(len) = response.content_length() {
        let mb = len as f64 / (1024.0 * 1024.0);
        eprintln!("{cli_label}: downloading {url} ({mb:.1} MB)");
    } else {
        eprintln!("{cli_label}: downloading {url}");
    }
}

/// Download a stable kernel tarball (.tar.xz) from cdn.kernel.org.
fn download_stable_tarball(
    client: &Client,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
) -> Result<PathBuf> {
    let major = major_version(version)?;
    let url = format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{version}.tar.xz");

    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("download {url}"))?;
    if !response.status().is_success() {
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            anyhow::bail!("{}", version_not_found_msg(client, version));
        }
        anyhow::bail!("download {url}: HTTP {}", response.status());
    }
    reject_html_response(&response, &url)?;
    print_download_size(&response, &url, cli_label);

    eprintln!("{cli_label}: extracting tarball (xz)");
    let decoder = xz2::read::XzDecoder::new(response);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(dest_dir)
        .with_context(|| "extract tarball")?;

    let source_dir = dest_dir.join(format!("linux-{version}"));
    if !source_dir.is_dir() {
        anyhow::bail!("expected directory linux-{version} after extraction");
    }
    Ok(source_dir)
}

/// Download an RC kernel tarball (.tar.gz) from git.kernel.org.
fn download_rc_tarball(
    client: &Client,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
) -> Result<PathBuf> {
    let url = format!("https://git.kernel.org/torvalds/t/linux-{version}.tar.gz");

    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("download {url}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        anyhow::bail!(
            "RC tarball not found: {url}\n  \
             RC releases are removed from git.kernel.org after the stable version ships."
        );
    }
    if !response.status().is_success() {
        anyhow::bail!("download {url}: HTTP {}", response.status());
    }
    reject_html_response(&response, &url)?;
    print_download_size(&response, &url, cli_label);

    eprintln!("{cli_label}: extracting tarball (gzip)");
    let decoder = flate2::read::GzDecoder::new(response);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(dest_dir)
        .with_context(|| "extract tarball")?;

    let source_dir = dest_dir.join(format!("linux-{version}"));
    if !source_dir.is_dir() {
        anyhow::bail!("expected directory linux-{version} after extraction");
    }
    Ok(source_dir)
}

/// Download a kernel tarball (stable or RC) and extract it.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn download_tarball(
    client: &Client,
    version: &str,
    dest_dir: &Path,
    cli_label: &str,
) -> Result<AcquiredSource> {
    let (arch, _) = arch_info();
    let source_dir = if is_rc(version) {
        download_rc_tarball(client, version, dest_dir, cli_label)?
    } else {
        download_stable_tarball(client, version, dest_dir, cli_label)?
    };

    Ok(AcquiredSource {
        source_dir,
        cache_key: format!("{version}-tarball-{arch}-kc{}", crate::cache_key_suffix()),
        version: Some(version.to_string()),
        kernel_source: crate::cache::KernelSource::Tarball,
        is_temp: true,
        is_dirty: false,
        is_git: true,
    })
}

/// Parse the patch level from a kernel version string.
/// "6.12.8" → Some(8), "7.0" → Some(0), "abc" → None.
fn patch_level(version: &str) -> Option<u32> {
    let parts: Vec<&str> = version.split('.').collect();
    match parts.len() {
        2 => Some(0), // "7.0" has patch level 0
        3 => parts[2].parse().ok(),
        _ => None,
    }
}

/// Fetch releases.json from kernel.org and return a vector of
/// [`Release`] records. Issues an HTTP GET unconditionally — no
/// cache consultation.
///
/// Production callers reach this function via
/// [`cached_releases_with`] (or [`cached_releases`]); the cache
/// helper only invokes `fetch_releases` on a cache miss for the
/// singleton path or on the bypass branch for non-singleton
/// clients. Tests that need to exercise the underlying GET
/// directly — without the cache layer — call this function with
/// a locally-constructed `Client` (e.g. pointed at an httpmock-
/// style local mock server).
pub(crate) fn fetch_releases(client: &Client) -> Result<Vec<Release>> {
    let url = "https://www.kernel.org/releases.json";
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("fetch {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!("fetch {url}: HTTP {}", response.status());
    }
    let body = response.text().with_context(|| "read response body")?;
    let json: serde_json::Value =
        serde_json::from_str(&body).with_context(|| "parse releases.json")?;
    let releases = json
        .get("releases")
        .and_then(|r| r.as_array())
        .ok_or_else(|| anyhow!("releases.json: missing releases array"))?;
    Ok(releases
        .iter()
        .filter_map(|r| {
            let moniker = r.get("moniker")?.as_str()?;
            let version = r.get("version")?.as_str()?;
            Some(Release {
                moniker: moniker.to_string(),
                version: version.to_string(),
            })
        })
        .collect())
}

/// Fetch the latest stable kernel version from kernel.org.
///
/// Selects from the `releases` array (moniker "stable" or "longterm"),
/// requiring patch version >= 8 to avoid brand-new major versions
/// that may have build issues on CI runners.
///
/// When `client` is the process-wide [`shared_client`] singleton,
/// routes through [`RELEASES_CACHE`]; other clients bypass the
/// cache via pointer-equality and exercise [`fetch_releases`]
/// directly — see [`cached_releases_with`] for details.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn fetch_latest_stable_version(client: &Client, cli_label: &str) -> Result<String> {
    eprintln!("{cli_label}: fetching latest kernel version");
    let releases = cached_releases_with(client)?;

    let mut best: Option<&str> = None;
    for r in &releases {
        if r.moniker != "stable" && r.moniker != "longterm" {
            continue;
        }
        if patch_level(&r.version).unwrap_or(0) < 8 {
            continue;
        }
        // Pick the first matching release — releases.json is ordered
        // newest first, so the first stable with patch >= 8 is the best.
        best = Some(r.version.as_str());
        break;
    }

    let version =
        best.ok_or_else(|| anyhow!("no stable kernel with patch >= 8 found in releases.json"))?;
    eprintln!("{cli_label}: latest stable kernel: {version}");
    Ok(version.to_string())
}

/// Parse a version string into numeric components for comparison.
/// "6.14.2" → Some((6, 14, 2)), "6.14" → Some((6, 14, 0)),
/// "7.0" → Some((7, 0, 0)). Returns None for unparseable versions.
fn version_tuple(version: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<&str> = version.split('.').collect();
    match parts.len() {
        2 => {
            let major = parts[0].parse().ok()?;
            let minor = parts[1].parse().ok()?;
            Some((major, minor, 0))
        }
        3 => {
            let major = parts[0].parse().ok()?;
            let minor = parts[1].parse().ok()?;
            let patch = parts[2].parse().ok()?;
            Some((major, minor, patch))
        }
        _ => None,
    }
}

/// Return true when `s` is a kernel major.minor prefix like
/// `"6.14"` (as opposed to a full patch version `"6.14.2"` or an rc
/// tag `"6.15-rc3"`). Callers use this to decide whether the input
/// needs prefix resolution via [`fetch_version_for_prefix`].
///
/// Accepts any string with fewer than 2 dots and no `-rc` substring,
/// so `"7"` (single-segment) and `""` both return true. This matches
/// the historical inline check used by kernel-build dispatchers.
pub fn is_major_minor_prefix(s: &str) -> bool {
    s.matches('.').count() < 2 && !s.contains("-rc")
}

/// Resolve the highest version matching a prefix.
///
/// E.g., "6.12" → "6.12.81", "6" → "6.19.12" (highest 6.x.y).
///
/// Scans all monikers in releases.json except linux-next. If no
/// match is found (EOL series), probes cdn.kernel.org with HEAD
/// requests to find the highest patch version with a tarball.
///
/// When `client` is the process-wide [`shared_client`] singleton,
/// routes through [`RELEASES_CACHE`]; other clients bypass the
/// cache via pointer-equality and exercise [`fetch_releases`]
/// directly — see [`cached_releases_with`] for details. Cache
/// scope is releases.json only; the EOL-series HEAD-probe
/// fallback in [`probe_latest_patch`] always hits the network.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn fetch_version_for_prefix(client: &Client, prefix: &str, cli_label: &str) -> Result<String> {
    eprintln!("{cli_label}: fetching latest {prefix}.x kernel version");
    let releases = cached_releases_with(client)?;

    let mut best: Option<(&str, (u32, u32, u32))> = None;
    for r in &releases {
        if is_skippable_release_moniker(&r.moniker) {
            continue;
        }
        if !r.version.starts_with(prefix) {
            continue;
        }
        if r.version.len() != prefix.len() && r.version.as_bytes()[prefix.len()] != b'.' {
            continue;
        }
        let Some(tuple) = version_tuple(&r.version) else {
            continue;
        };
        if best.is_none() || tuple > best.unwrap().1 {
            best = Some((r.version.as_str(), tuple));
        }
    }

    if let Some((version, _)) = best {
        eprintln!("{cli_label}: latest {prefix}.x kernel: {version}");
        return Ok(version.to_string());
    }

    eprintln!("{cli_label}: {prefix}.x not in releases.json (EOL series), probing cdn.kernel.org");
    probe_latest_patch(client, prefix, cli_label)
}

/// Upper bound for the search range in [`probe_latest_patch`].
/// No kernel minor has ever produced this many patch releases; the bound
/// exists only to terminate the exponential-expansion phase when a CDN
/// misbehaves and returns success for every probe.
const PROBE_PATCH_MAX: u32 = 500;

/// HEAD one cdn.kernel.org tarball URL for `{prefix}.{patch}`.
///
/// Returns `Ok(true)` iff the server returned a 2xx status AND the
/// response body is not HTML (some CDN error pages return 200 with
/// text/html). Network / transport failures propagate as `Err`.
fn probe_patch_exists(client: &Client, major: u32, prefix: &str, patch: u32) -> Result<bool> {
    let url =
        format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{prefix}.{patch}.tar.xz");
    let response = client
        .head(&url)
        .send()
        .with_context(|| format!("HEAD {url}"))?;
    if !response.status().is_success() {
        return Ok(false);
    }
    if let Some(ct) = response.headers().get(reqwest::header::CONTENT_TYPE)
        && let Ok(ct_str) = ct.to_str()
        && ct_str.contains("text/html")
    {
        return Ok(false);
    }
    Ok(true)
}

/// Probe cdn.kernel.org to find the highest patch version for an EOL series.
///
/// Probes patches in parallel batches that double in size each round
/// (16, 32, 64, ...). Each batch HEADs its entire window concurrently
/// via rayon; scanning the ordered results short-circuits at the first
/// non-existent patch. This replaces the former "serial HEAD 1..=500"
/// scan, which issued up to 500 sequential HTTP requests — each ~1 RTT
/// — even for minors with only a handful of published patches, and
/// stalled interactive runs by ~500x the single-request RTT on the
/// slowest path.
///
/// Complexity: the largest patch N is pinpointed in `O(log N)` batches
/// rather than `O(N)` serial requests, and every batch completes in
/// roughly one RTT.
fn probe_latest_patch(client: &Client, prefix: &str, cli_label: &str) -> Result<String> {
    use rayon::prelude::*;

    let major = major_version(prefix)?;

    /// Initial batch size. Each subsequent round doubles the window so
    /// minors with many patches still finish in log-time rounds.
    const PROBE_PATCH_INITIAL_BATCH: u32 = 16;

    // Cap the window at the rayon pool size: HEAD requests beyond that
    // cannot run in parallel anyway, they just queue behind the pool's
    // threads and add latency without widening the probe. Floor at
    // PROBE_PATCH_INITIAL_BATCH so small-core hosts (2-4 core CI
    // runners) still get the log-time search — work-stealing handles
    // the initial queuing cheaply, and the cap only kicks in on large
    // hosts whose growth phase would otherwise run absurdly wide.
    let pool_cap = rayon::current_num_threads().max(PROBE_PATCH_INITIAL_BATCH as usize) as u32;

    let mut last_good: u32 = 0;
    let mut lo: u32 = 1;
    let mut window: u32 = PROBE_PATCH_INITIAL_BATCH.min(pool_cap);
    'expand: loop {
        let hi = (lo + window - 1).min(PROBE_PATCH_MAX);
        // HEAD the entire window concurrently. Any transport error
        // short-circuits via `collect::<Result<_, _>>()`.
        let results: Vec<(u32, bool)> = (lo..=hi)
            .into_par_iter()
            .map(|patch| probe_patch_exists(client, major, prefix, patch).map(|ok| (patch, ok)))
            .collect::<Result<Vec<_>>>()?;
        // rayon preserves input order, so iterating advances `last_good`
        // through increasing patch numbers and stops at the first 404.
        for (patch, ok) in results {
            if !ok {
                break 'expand;
            }
            last_good = patch;
        }
        if hi >= PROBE_PATCH_MAX {
            break;
        }
        lo = hi + 1;
        window = window.saturating_mul(2).min(pool_cap);
    }

    if last_good == 0 {
        anyhow::bail!("no tarball found for {prefix}.x on cdn.kernel.org");
    }
    let version = format!("{prefix}.{last_good}");
    eprintln!("{cli_label}: latest {prefix}.x kernel (from cdn probe): {version}");
    Ok(version)
}

/// Clone a git repository with shallow depth.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn git_clone(
    url: &str,
    git_ref: &str,
    dest_dir: &Path,
    cli_label: &str,
) -> Result<AcquiredSource> {
    let (arch, _) = arch_info();
    eprintln!("{cli_label}: cloning {url} (ref: {git_ref}, depth: 1)");

    let clone_dir = dest_dir.join("linux");

    let mut prep = gix::prepare_clone(url, &clone_dir)
        .with_context(|| "prepare clone")?
        .with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
            NonZeroU32::new(1).expect("1 is nonzero"),
        ))
        .with_ref_name(Some(git_ref))
        .with_context(|| "set ref name")?;

    let (mut checkout, _outcome) = prep
        .fetch_then_checkout(
            gix::progress::Discard,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .with_context(|| "clone fetch")?;

    let (_repo, _outcome) = checkout
        .main_worktree(
            gix::progress::Discard,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .with_context(|| "checkout")?;

    let repo = gix::open(&clone_dir).with_context(|| "open cloned repo")?;
    let head = repo.head_id().with_context(|| "read HEAD")?;
    let short_hash = format!("{}", head).chars().take(7).collect::<String>();

    let cache_key = format!(
        "{git_ref}-git-{short_hash}-{arch}-kc{}",
        crate::cache_key_suffix()
    );

    Ok(AcquiredSource {
        source_dir: clone_dir,
        cache_key,
        version: None,
        kernel_source: crate::cache::KernelSource::Git {
            git_hash: Some(short_hash),
            git_ref: Some(git_ref.to_string()),
        },
        is_temp: true,
        is_dirty: false,
        is_git: true,
    })
}

/// Use a local kernel source tree.
///
/// Dirty detection uses gix `tree_index_status` (HEAD-vs-index) and
/// `status().into_index_worktree_iter()` (index-vs-worktree) to check
/// for modifications to tracked files. Submodule checks are skipped
/// entirely. Untracked files do not affect the dirty flag.
///
/// When the tree is dirty, the HEAD commit does not describe the
/// source actually being built, so `git_hash` is dropped — no
/// commit identifies a dirty worktree. `is_dirty=true` carries that
/// fact forward; callers (see [`crate::cli`]) use it to bypass the
/// kernel cache entirely.
///
/// No diagnostic output: all operator-visible messaging for a
/// local source is routed through `kernel_build_pipeline`'s
/// cache-skip hint (`DIRTY_TREE_CACHE_SKIP_HINT` /
/// `NON_GIT_TREE_CACHE_SKIP_HINT`), which has the full context
/// to emit a single informational line rather than two redundant
/// warnings. Sibling entries (`download_tarball`, `git_clone`)
/// still take a `cli_label` because they genuinely print
/// progress lines — `local_source` does not.
pub fn local_source(source_path: &Path) -> Result<AcquiredSource> {
    let (arch, _) = arch_info();

    if !source_path.is_dir() {
        anyhow::bail!("{}: not a directory", source_path.display());
    }

    let canonical = source_path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", source_path.display()))?;

    // Git hash extraction and dirty detection via gix.
    // Submodule checks are skipped (false positives on kernel
    // trees with uninitialized submodules). The tuple return carries
    // `(hash, is_dirty, is_git)` so the non-git arm can signal
    // "this isn't a git repo" to the cache-skip hint at the caller.
    let (short_hash, is_dirty, is_git) = match gix::discover(&canonical) {
        Ok(repo) => {
            let head = repo.head_id().with_context(|| "read HEAD")?;
            let short_hash = format!("{}", head).chars().take(7).collect::<String>();

            // tree_index_status compares a TREE id against the index;
            // the HEAD commit id is not itself a tree, so peel HEAD
            // to its root tree before diffing or the diff silently
            // returns an error and index dirt goes undetected.
            let head_tree = repo.head_tree().with_context(|| "read HEAD tree")?;
            let head_tree_id = head_tree.id;

            // Check HEAD-vs-index for tracked file changes.
            let mut index_dirty = false;
            let index = repo.index_or_empty().with_context(|| "open index")?;
            let _ = repo.tree_index_status(
                &head_tree_id,
                &index,
                None,
                gix::status::tree_index::TrackRenames::Disabled,
                |_, _, _| {
                    index_dirty = true;
                    Ok::<_, std::convert::Infallible>(std::ops::ControlFlow::Break(()))
                },
            );

            // Check index-vs-worktree for modified tracked files,
            // skipping submodules entirely (Ignore::All).
            let worktree_dirty = if !index_dirty {
                repo.status(gix::progress::Discard)
                    .with_context(|| "status")?
                    .index_worktree_rewrites(None)
                    .index_worktree_submodules(gix::status::Submodule::Given {
                        ignore: gix::submodule::config::Ignore::All,
                        check_dirty: false,
                    })
                    .index_worktree_options_mut(|opts| {
                        opts.dirwalk_options = None;
                    })
                    .into_index_worktree_iter(Vec::new())
                    .map(|mut iter| iter.next().is_some())
                    .unwrap_or(false)
            } else {
                false
            };

            let is_dirty = index_dirty || worktree_dirty;
            // Drop the HEAD hash when dirty — the commit does not
            // describe the actual source being built, so publishing
            // it via git_hash / cache_key would misidentify the
            // build input.
            let hash = if is_dirty { None } else { Some(short_hash) };
            (hash, is_dirty, true)
        }
        Err(_) => {
            // The downstream kernel_build_pipeline (cli::kernel_build_pipeline)
            // emits `NON_GIT_TREE_CACHE_SKIP_HINT` — a single
            // informational line that names both the cause and the
            // remediation paths — once the is_dirty=true branch
            // decides to skip the cache. Emitting a second
            // "not a git repository" warning here duplicated that
            // content for every non-git `--source` run. The
            // `(None, true, false)` tuple silently communicates
            // the non-git state to the cache-skip decision site;
            // no separate stderr line is needed on this path.
            (None, true, false)
        }
    };

    let suffix = crate::cache_key_suffix();
    let cache_key = match &short_hash {
        Some(hash) => format!("local-{hash}-{arch}-kc{suffix}"),
        None => format!("local-unknown-{arch}-kc{suffix}"),
    };

    Ok(AcquiredSource {
        source_dir: canonical.clone(),
        cache_key,
        version: None,
        kernel_source: crate::cache::KernelSource::Local {
            source_tree_path: Some(canonical),
            git_hash: short_hash,
        },
        is_temp: false,
        is_dirty,
        is_git,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- arch_info --

    #[test]
    fn fetch_arch_info_returns_known_arch() {
        let (arch, image) = arch_info();
        assert!(
            (arch == "x86_64" && image == "bzImage") || (arch == "aarch64" && image == "Image"),
            "unexpected arch/image: {arch}/{image}"
        );
    }

    // -- is_major_minor_prefix --

    #[test]
    fn is_major_minor_prefix_accepts_two_segment() {
        assert!(is_major_minor_prefix("6.14"));
        assert!(is_major_minor_prefix("7.0"));
    }

    #[test]
    fn is_major_minor_prefix_rejects_patch_version() {
        assert!(!is_major_minor_prefix("6.14.2"));
        assert!(!is_major_minor_prefix("5.4.0"));
    }

    #[test]
    fn is_major_minor_prefix_rejects_rc_tag() {
        assert!(!is_major_minor_prefix("6.15-rc3"));
        assert!(!is_major_minor_prefix("6.14-rc1"));
    }

    #[test]
    fn is_major_minor_prefix_historical_edge_cases() {
        // Historical behavior: accepts single-segment and empty inputs.
        // Callers are expected to gate upstream.
        assert!(is_major_minor_prefix("7"));
        assert!(is_major_minor_prefix(""));
    }

    // -- major_version --

    #[test]
    fn fetch_major_version_stable() {
        assert_eq!(major_version("6.14.2").unwrap(), 6);
    }

    #[test]
    fn fetch_major_version_rc() {
        assert_eq!(major_version("6.15-rc3").unwrap(), 6);
    }

    #[test]
    fn fetch_major_version_two_part() {
        assert_eq!(major_version("5.4").unwrap(), 5);
    }

    #[test]
    fn fetch_major_version_invalid() {
        assert!(major_version("abc").is_err());
    }

    // -- is_rc --

    #[test]
    fn fetch_is_rc_true() {
        assert!(is_rc("6.15-rc3"));
        assert!(is_rc("6.14.2-rc1"));
    }

    #[test]
    fn fetch_is_rc_false() {
        assert!(!is_rc("6.14.2"));
        assert!(!is_rc("6.14"));
    }

    // -- URL construction --

    /// Stable tarball URL pattern (same logic as download_stable_tarball).
    fn stable_tarball_url(version: &str) -> Result<String> {
        let major = major_version(version)?;
        Ok(format!(
            "https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{version}.tar.xz"
        ))
    }

    /// RC tarball URL pattern (same logic as download_rc_tarball).
    fn rc_tarball_url(version: &str) -> String {
        format!("https://git.kernel.org/torvalds/t/linux-{version}.tar.gz")
    }

    #[test]
    fn fetch_stable_url_construction() {
        let url = stable_tarball_url("6.14.2").unwrap();
        assert_eq!(
            url,
            "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.14.2.tar.xz"
        );
    }

    #[test]
    fn fetch_stable_url_v5() {
        let url = stable_tarball_url("5.4.0").unwrap();
        assert_eq!(
            url,
            "https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.0.tar.xz"
        );
    }

    #[test]
    fn fetch_rc_url_construction() {
        let url = rc_tarball_url("6.15-rc3");
        assert_eq!(
            url,
            "https://git.kernel.org/torvalds/t/linux-6.15-rc3.tar.gz"
        );
    }

    // -- patch_level --

    #[test]
    fn fetch_patch_level_three_part() {
        assert_eq!(patch_level("6.12.8"), Some(8));
    }

    #[test]
    fn fetch_patch_level_two_part() {
        assert_eq!(patch_level("7.0"), Some(0));
    }

    #[test]
    fn fetch_patch_level_single_part() {
        assert_eq!(patch_level("6"), None);
    }

    #[test]
    fn fetch_patch_level_four_part() {
        assert_eq!(patch_level("6.1.2.3"), None);
    }

    #[test]
    fn fetch_patch_level_non_numeric_patch() {
        assert_eq!(patch_level("6.1.rc3"), None);
    }

    #[test]
    fn fetch_patch_level_zero() {
        assert_eq!(patch_level("6.14.0"), Some(0));
    }

    #[test]
    fn fetch_patch_level_large() {
        assert_eq!(patch_level("6.12.99"), Some(99));
    }

    // -- local_source dirty detection --

    /// Initialise a git repo at `dir` with one committed file, using
    /// the `git` CLI with explicit identity + empty global config so
    /// the test is deterministic on developer machines and CI runners
    /// regardless of the ambient git setup.
    fn init_repo_with_commit(dir: &Path) {
        use std::process::Command;

        let run = |args: &[&str]| {
            let out = Command::new("git")
                .args(args)
                .current_dir(dir)
                // Empty system/global config: the test owns identity
                // and default-branch config via -c flags below.
                .env("GIT_CONFIG_GLOBAL", "/dev/null")
                .env("GIT_CONFIG_SYSTEM", "/dev/null")
                .env("GIT_AUTHOR_NAME", "ktstr-test")
                .env("GIT_AUTHOR_EMAIL", "ktstr-test@localhost")
                .env("GIT_COMMITTER_NAME", "ktstr-test")
                .env("GIT_COMMITTER_EMAIL", "ktstr-test@localhost")
                .output()
                .expect("spawn git");
            assert!(
                out.status.success(),
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&out.stderr)
            );
        };

        run(&["init", "-q", "-b", "main"]);
        std::fs::write(dir.join("file.txt"), "original\n").unwrap();
        run(&["add", "file.txt"]);
        run(&[
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "initial",
        ]);
    }

    /// On a clean repo, `local_source` must report `is_dirty=false` and
    /// populate both the cache key and KernelSource::Local.git_hash
    /// with the HEAD short-hash.
    #[test]
    fn local_source_clean_repo_populates_hash() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("git CLI unavailable");
        }
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_commit(tmp.path());

        let acquired = local_source(tmp.path()).expect("local_source ok");
        assert!(!acquired.is_dirty, "clean tree must not be dirty");

        let git_hash = match &acquired.kernel_source {
            crate::cache::KernelSource::Local { git_hash, .. } => git_hash.clone(),
            other => panic!("expected KernelSource::Local, got {other:?}"),
        };
        let hash = git_hash.expect("clean repo must carry a git_hash");
        assert_eq!(hash.len(), 7, "short hash must be 7 chars, got {hash:?}");
        assert!(
            hash.chars().all(|c| c.is_ascii_hexdigit()),
            "hash must be hex, got {hash:?}"
        );
        assert!(
            acquired.cache_key.contains(&hash),
            "clean cache_key must embed the short hash, got {}",
            acquired.cache_key
        );
    }

    /// On a dirty tracked-file worktree (worktree mutation after
    /// commit), `local_source` must report `is_dirty=true` AND clear
    /// `KernelSource::Local.git_hash`. The HEAD commit does not
    /// describe a dirty tree, so surfacing the HEAD hash as the
    /// build's source identity would mislead a reproducer.
    #[test]
    fn local_source_dirty_tracked_file_clears_hash() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("git CLI unavailable");
        }
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_commit(tmp.path());
        // Mutate the tracked file — index-vs-worktree becomes dirty.
        std::fs::write(tmp.path().join("file.txt"), "modified\n").unwrap();

        let acquired = local_source(tmp.path()).expect("local_source ok");
        assert!(acquired.is_dirty, "worktree mutation must mark dirty");
        match &acquired.kernel_source {
            crate::cache::KernelSource::Local { git_hash, .. } => {
                assert!(
                    git_hash.is_none(),
                    "dirty tree must not publish git_hash, got {git_hash:?}"
                );
            }
            other => panic!("expected KernelSource::Local, got {other:?}"),
        }
        // Cache key must also fall through to the unknown bucket so
        // a dirty build can never collide with a clean build at the
        // same HEAD if caching is ever attempted.
        assert!(
            acquired.cache_key.starts_with("local-unknown-"),
            "dirty cache_key must use local-unknown prefix, got {}",
            acquired.cache_key
        );
    }

    /// Staged-but-not-committed changes are dirty via the HEAD-vs-index
    /// check (`tree_index_status`) rather than index-vs-worktree. The
    /// same `git_hash=None` invariant applies.
    #[test]
    fn local_source_dirty_staged_only_clears_hash() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("git CLI unavailable");
        }
        let tmp = tempfile::TempDir::new().unwrap();
        init_repo_with_commit(tmp.path());
        // Modify + stage (so worktree matches index, but index
        // differs from HEAD).
        std::fs::write(tmp.path().join("file.txt"), "staged\n").unwrap();
        let status = std::process::Command::new("git")
            .args(["add", "file.txt"])
            .current_dir(tmp.path())
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .status()
            .expect("git add");
        assert!(status.success());

        let acquired = local_source(tmp.path()).expect("local_source ok");
        assert!(acquired.is_dirty, "staged-only change must mark dirty");
        match &acquired.kernel_source {
            crate::cache::KernelSource::Local { git_hash, .. } => {
                assert!(
                    git_hash.is_none(),
                    "dirty (staged) tree must not publish git_hash, got {git_hash:?}"
                );
            }
            other => panic!("expected KernelSource::Local, got {other:?}"),
        }
    }

    /// Non-git directories are treated as permanently dirty and
    /// produce `git_hash=None` — there is no commit to reference.
    #[test]
    fn local_source_non_git_is_dirty_without_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("file.txt"), "no git here\n").unwrap();

        let acquired = local_source(tmp.path()).expect("local_source ok");
        assert!(acquired.is_dirty, "non-git tree must mark dirty");
        match &acquired.kernel_source {
            crate::cache::KernelSource::Local { git_hash, .. } => {
                assert!(
                    git_hash.is_none(),
                    "non-git tree must not publish git_hash, got {git_hash:?}"
                );
            }
            other => panic!("expected KernelSource::Local, got {other:?}"),
        }
        assert!(
            acquired.cache_key.starts_with("local-unknown-"),
            "non-git cache_key must use local-unknown prefix, got {}",
            acquired.cache_key
        );
    }

    // -- cached_releases --

    /// Pin every routing property of [`cached_releases_with`]
    /// in one test, since the underlying [`RELEASES_CACHE`]
    /// `OnceLock` only allows one populating `set` per process.
    /// Each block below is a distinct assertion:
    ///
    /// (a) **Cache-hit fast-path**: pre-populating
    ///     [`RELEASES_CACHE`] with synthetic data and calling
    ///     [`cached_releases`] returns the synthetic vector
    ///     verbatim — the `if let Some(cached) = ... .get()`
    ///     path is exercised, not [`fetch_releases`].
    ///
    /// (b) **Idempotency**: a second [`cached_releases`] call
    ///     returns the same data — the slot remains populated
    ///     across calls within the process.
    ///
    /// (c) **Singleton-path public-fn routing**:
    ///     [`fetch_latest_stable_version`] called with
    ///     [`shared_client`] reaches [`RELEASES_CACHE`] via
    ///     [`cached_releases_with`] and selects from the
    ///     synthetic data without touching the network.
    ///
    /// (d) **Bypass-branch routing**: a non-singleton local
    ///     [`Client`] passed to [`cached_releases_with`] does
    ///     NOT consult [`RELEASES_CACHE`] even when populated.
    ///
    /// Cross-test contamination: this test populates the
    /// process-wide [`RELEASES_CACHE`] AND initializes the
    /// process-wide [`SHARED_CLIENT`] (via the
    /// [`shared_client`] call in block (c)). Both are
    /// `OnceLock` statics — peer tests in the same binary
    /// observe both as populated/initialized after this test
    /// runs. No other test in this binary calls
    /// [`cached_releases`] or any cache-routed `fetch_*` entry
    /// ([`fetch_latest_stable_version`],
    /// [`fetch_version_for_prefix`], `latest_in_series`) with
    /// [`shared_client`] — the `expand_kernel_range`-shaped
    /// tests in `cli.rs` bypass the network by calling
    /// `filter_and_sort_range` directly with synthetic
    /// releases. The
    /// `is_shared_client_recognizes_process_singleton` and
    /// `is_shared_client_rejects_test_constructed_clients`
    /// tests touch [`SHARED_CLIENT`] but not
    /// [`RELEASES_CACHE`], so they coexist with this test. A
    /// future test that calls any cache-routed entry with
    /// [`shared_client`] must run in a separate binary or
    /// accept the synthetic-data side effect.
    #[test]
    fn cached_releases_routing_singleton_and_bypass() {
        let synthetic = vec![
            Release {
                moniker: "stable".to_string(),
                version: "6.14.2".to_string(),
            },
            Release {
                moniker: "longterm".to_string(),
                version: "6.12.81".to_string(),
            },
            Release {
                moniker: "mainline".to_string(),
                version: "6.16-rc3".to_string(),
            },
        ];

        // Pre-populate the cache. `set` returns `Err(value)` if
        // the slot was already populated by an earlier test in
        // the same binary; in that case our test cannot prove
        // the cache contract because we'd be reading the prior
        // value instead. The static is initialized fresh per
        // process and no other test in this binary populates it
        // — `cli.rs`'s `expand_kernel_range`-shaped tests bypass
        // the network entirely and no test passes
        // `shared_client()` into the cache-routed `fetch_*`
        // family. If a future test breaks that invariant, the
        // assertion below catches it loudly rather than
        // silently passing.
        let set_result = super::RELEASES_CACHE.set(synthetic.clone());
        assert!(
            set_result.is_ok(),
            "test precondition: RELEASES_CACHE must be empty before \
             this test runs — a peer test populated it, breaking the \
             cache-hit assertion below. nextest runs all unit tests \
             of this crate in one process; reorder or split the \
             offending test into a separate binary.",
        );

        // Cache hit: should return the synthetic data verbatim
        // without any network round-trip. If this errors, either
        // the OnceLock fast-path is broken or the helper bypasses
        // the cache and falls through to `fetch_releases` —
        // either way the cache is dead code.
        let result = super::cached_releases().expect(
            "cache hit must return Ok — a network attempt indicates \
             the OnceLock fast-path is bypassed",
        );
        assert_eq!(
            result.len(),
            synthetic.len(),
            "cached vector length must match populated value; got \
             {} entries, populated {}",
            result.len(),
            synthetic.len(),
        );
        for (got, want) in result.iter().zip(synthetic.iter()) {
            assert_eq!(
                got.moniker, want.moniker,
                "cached entry moniker must match populated value",
            );
            assert_eq!(
                got.version, want.version,
                "cached entry version must match populated value",
            );
        }

        // Idempotency: a second call must return the same data.
        // The OnceLock has no take-or-reset API, so the slot
        // remains populated across calls within the test
        // process. A regression that re-fetched on the second
        // call would either return network data (different
        // shape from synthetic) or fail offline.
        let second = super::cached_releases().expect(
            "second cache hit must also return Ok — a regression that \
             cleared the cache between calls would surface here",
        );
        assert_eq!(
            second.len(),
            synthetic.len(),
            "second call must return the same length as the first",
        );
        for (got, want) in second.iter().zip(synthetic.iter()) {
            assert_eq!(got.moniker, want.moniker);
            assert_eq!(got.version, want.version);
        }

        // End-to-end singleton path through a public fetch
        // function: `fetch_latest_stable_version(shared_client(),
        // ...)` must consult `RELEASES_CACHE` via
        // `cached_releases_with` and return "6.12.81" without
        // issuing any network request. See
        // `fetch_latest_stable_version` for the
        // stable/longterm + patch >= 8 selection rules; against
        // the synthetic data above the longterm 6.12.81 entry
        // is the first match. A regression that bypassed the
        // cache would attempt a real kernel.org fetch.
        let latest = super::fetch_latest_stable_version(super::shared_client(), "test")
            .expect("public-fn singleton path must reach cache");
        assert_eq!(
            latest, "6.12.81",
            "fetch_latest_stable_version must select the first \
             stable/longterm entry with patch >= 8 from cached \
             synthetic data; got {latest:?}",
        );

        // Bypass branch through `cached_releases_with` with a
        // non-singleton `Client`: the cache must NOT be consulted
        // even though it is populated above. The 1ms connect
        // timeout fires before any TCP handshake to kernel.org
        // can complete on essentially any real network, producing
        // an `Err`. If a fast-enough route somehow returns Ok,
        // the response cannot be the synthetic data — proving
        // routing took the bypass branch either way.
        let local = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_millis(1))
            .build()
            .expect("build local Client");
        let bypass_result = super::cached_releases_with(&local);
        match bypass_result {
            Err(_) => {
                // Expected: 1ms timeout fires before the real
                // network can answer. Bypass branch correctly
                // delegated to fetch_releases(&local).
            }
            Ok(real) => {
                assert!(
                    real.len() != synthetic.len()
                        || real.iter().zip(synthetic.iter()).any(|(got, want)| {
                            got.version != want.version || got.moniker != want.moniker
                        },),
                    "bypass branch returned the synthetic cache contents — \
                     cache routing leaked across the pointer-equality gate",
                );
            }
        }
    }

    // -- is_shared_client --

    /// `is_shared_client` recognizes the process-wide singleton:
    /// the [`shared_client`] address is stable across every call
    /// within a process (`OnceLock::get_or_init` returns the same
    /// pointer), so passing it to the predicate must yield `true`.
    /// This is the cache-routing branch of [`cached_releases_with`].
    #[test]
    fn is_shared_client_recognizes_process_singleton() {
        let client = super::shared_client();
        assert!(
            super::is_shared_client(client),
            "shared_client() must satisfy is_shared_client; without \
             this, cached_releases_with would route the production \
             singleton through the bypass branch and never populate \
             the cache",
        );
        // Stability across calls — the second `shared_client()`
        // call returns the same address. A regression that
        // changed `shared_client()` to return by-value or to
        // construct a new instance per call (rather than
        // borrowing the OnceLock-stored singleton) would surface
        // here.
        assert!(
            super::is_shared_client(super::shared_client()),
            "shared_client() must return a stable pointer across \
             repeated calls; the OnceLock contract guarantees this",
        );
    }

    /// `is_shared_client` rejects test-constructed clients: a
    /// `reqwest::blocking::Client::new()` call lives at a
    /// different address from the singleton, so the predicate
    /// returns `false`. This is the bypass branch of
    /// [`cached_releases_with`] — fault-injection tests that
    /// build their own `Client` (e.g. for httpmock-style
    /// scenarios) MUST land here, otherwise their mock would be
    /// ignored in favor of the cache.
    #[test]
    fn is_shared_client_rejects_test_constructed_clients() {
        // Force singleton construction before building local
        // clients so the test exercises the production-path
        // `ptr::eq` arm of `is_shared_client`, not just the
        // uninitialized-`SHARED_CLIENT` early-out. Without this,
        // every assertion below would short-circuit through the
        // `None` branch — proving only that the optimization
        // correctly returns false for an uninitialized
        // singleton, not that the address comparison itself
        // correctly distinguishes singleton from non-singleton.
        // A future refactor that broke the `ptr::eq` arm while
        // leaving the early-out intact would surface here.
        let _ = super::shared_client();
        let local = reqwest::blocking::Client::new();
        assert!(
            !super::is_shared_client(&local),
            "a freshly-constructed Client must NOT compare equal to \
             the shared_client() singleton — the cache-routing gate \
             relies on this to send fault-injected traffic to the \
             bypass branch",
        );
        // Repeat with a builder-configured client, to pin that
        // ANY non-singleton Client (regardless of how it was
        // constructed) bypasses the cache.
        let configured = reqwest::blocking::Client::builder()
            .connect_timeout(std::time::Duration::from_millis(100))
            .build()
            .expect("build local Client");
        assert!(
            !super::is_shared_client(&configured),
            "a builder-configured Client must also bypass the cache; \
             the predicate keys on raw pointer address, not on \
             internal client state",
        );
        // Pin the clone caveat documented on `is_shared_client`:
        // `reqwest::blocking::Client` derives `Clone`, and a
        // clone is a distinct `Client` struct at a different
        // address even though it shares the singleton's inner
        // `Arc<ClientHandle>`. A clone of `shared_client()`
        // must therefore bypass the cache. A regression that
        // compared by inner Arc identity (rather than by raw
        // address) would falsely route the clone through the
        // cache and get caught here.
        let cloned = super::shared_client().clone();
        assert!(
            !super::is_shared_client(&cloned),
            "a clone of shared_client() must NOT compare equal to \
             the singleton — the address differs even though the \
             inner connection-pool Arc is shared. Always pass \
             shared_client() directly when cache routing is desired.",
        );
    }

    // -- proptest --

    use proptest::prop_assert;

    proptest::proptest! {
        /// Any arbitrary input must not panic AND, on success, return
        /// only values the input string can justify. Broadened from
        /// 0..20 to 0..100 characters to exercise long/multi-dot
        /// pathological inputs the 20-char range missed.
        #[test]
        fn prop_major_version_never_panics(s in "\\PC{0,100}") {
            if let Ok(major) = major_version(&s) {
                // Ok(major) is only valid when the first dot-segment
                // parses as the returned integer.
                let first = s.split('.').next().unwrap_or("");
                prop_assert!(first.parse::<u32>().ok() == Some(major));
            }
        }

        #[test]
        fn prop_is_rc_contains_dash_rc(s in "\\PC{0,20}") {
            assert_eq!(is_rc(&s), s.contains("-rc"));
        }

        #[test]
        fn prop_patch_level_valid_three_part(
            major in 1u32..100,
            minor in 0u32..100,
            patch in 0u32..100,
        ) {
            let v = format!("{major}.{minor}.{patch}");
            assert_eq!(patch_level(&v), Some(patch));
        }

        #[test]
        fn prop_patch_level_valid_two_part(major in 1u32..100, minor in 0u32..100) {
            let v = format!("{major}.{minor}");
            assert_eq!(patch_level(&v), Some(0));
        }

        #[test]
        fn prop_major_version_valid(major in 1u32..100, minor in 0u32..100) {
            let v = format!("{major}.{minor}");
            assert_eq!(major_version(&v).unwrap(), major);
        }
    }
}
