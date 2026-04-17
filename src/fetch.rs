//! Kernel source acquisition: tarball download, git clone, local tree.
//!
//! All source types produce a directory containing a kernel source tree
//! ready for configuration and building.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

/// Downloaded/cloned kernel source ready for building.
#[non_exhaustive]
pub struct AcquiredSource {
    /// Path to the kernel source directory.
    pub source_dir: PathBuf,
    /// Cache key for this source (e.g. "6.14.2-tarball-x86_64-kc{kconfig_hash}").
    pub cache_key: String,
    /// Version string if known (e.g. "6.14.2", "6.15-rc3").
    pub version: Option<String>,
    /// Git commit hash (short form) if available.
    pub git_hash: Option<String>,
    /// Git ref used for checkout, if applicable.
    pub git_ref: Option<String>,
    /// Source type for cache metadata.
    pub source_type: crate::cache::SourceType,
    /// Whether the source is a temporary directory that should be
    /// cleaned up after building.
    pub is_temp: bool,
    /// For local sources: whether the working tree is dirty.
    /// Dirty trees must not be cached.
    pub is_dirty: bool,
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
fn major_version(version: &str) -> Result<u32, String> {
    let major_str = version
        .split('.')
        .next()
        .ok_or_else(|| format!("invalid version: {version}"))?;
    major_str
        .parse::<u32>()
        .map_err(|e| format!("invalid major version in {version}: {e}"))
}

/// Determine if a version string represents an RC release.
///
/// RC releases use a different URL pattern and gzip compression
/// (vs xz for stable).
fn is_rc(version: &str) -> bool {
    version.contains("-rc")
}

/// Reject responses where the server returned HTML instead of a binary
/// archive. Some CDN error pages return 200 with text/html.
fn reject_html_response(response: &reqwest::blocking::Response, url: &str) -> Result<(), String> {
    if let Some(ct) = response.headers().get(reqwest::header::CONTENT_TYPE)
        && let Ok(ct_str) = ct.to_str()
        && ct_str.contains("text/html")
    {
        return Err(format!(
            "download {url}: server returned HTML instead of tarball (URL may be invalid)"
        ));
    }
    Ok(())
}

/// Print download size from Content-Length header if available.
fn print_download_size(response: &reqwest::blocking::Response, url: &str) {
    if let Some(len) = response.content_length() {
        let mb = len as f64 / (1024.0 * 1024.0);
        eprintln!("ktstr: downloading {url} ({mb:.1} MB)");
    } else {
        eprintln!("ktstr: downloading {url}");
    }
}

/// Download a stable kernel tarball (.tar.xz) from cdn.kernel.org.
fn download_stable_tarball(version: &str, dest_dir: &Path) -> Result<PathBuf, String> {
    let major = major_version(version)?;
    let url = format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{version}.tar.xz");

    let response = reqwest::blocking::get(&url).map_err(|e| format!("download {url}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("download {url}: HTTP {}", response.status()));
    }
    reject_html_response(&response, &url)?;
    print_download_size(&response, &url);

    eprintln!("ktstr: extracting tarball (xz)");
    let decoder = xz2::read::XzDecoder::new(response);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(dest_dir)
        .map_err(|e| format!("extract tarball: {e}"))?;

    // Tarballs extract to linux-{version}/ subdirectory.
    let source_dir = dest_dir.join(format!("linux-{version}"));
    if !source_dir.is_dir() {
        return Err(format!(
            "expected directory linux-{version} after extraction"
        ));
    }
    Ok(source_dir)
}

/// Download an RC kernel tarball (.tar.gz) from git.kernel.org.
fn download_rc_tarball(version: &str, dest_dir: &Path) -> Result<PathBuf, String> {
    let url = format!("https://git.kernel.org/torvalds/t/linux-{version}.tar.gz");

    let response = reqwest::blocking::get(&url).map_err(|e| format!("download {url}: {e}"))?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(format!(
            "RC tarball not found: {url}\n  \
             RC releases are removed from git.kernel.org after the stable version ships."
        ));
    }
    if !response.status().is_success() {
        return Err(format!("download {url}: HTTP {}", response.status()));
    }
    reject_html_response(&response, &url)?;
    print_download_size(&response, &url);

    eprintln!("ktstr: extracting tarball (gzip)");
    let decoder = flate2::read::GzDecoder::new(response);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(dest_dir)
        .map_err(|e| format!("extract tarball: {e}"))?;

    // RC tarballs extract to linux-{version}/ subdirectory.
    let source_dir = dest_dir.join(format!("linux-{version}"));
    if !source_dir.is_dir() {
        return Err(format!(
            "expected directory linux-{version} after extraction"
        ));
    }
    Ok(source_dir)
}

/// Download a kernel tarball (stable or RC) and extract it.
pub fn download_tarball(version: &str, dest_dir: &Path) -> Result<AcquiredSource, String> {
    let (arch, _) = arch_info();
    let source_dir = if is_rc(version) {
        download_rc_tarball(version, dest_dir)?
    } else {
        download_stable_tarball(version, dest_dir)?
    };

    Ok(AcquiredSource {
        source_dir,
        cache_key: format!("{version}-tarball-{arch}-kc{}", crate::cache_key_suffix()),
        version: Some(version.to_string()),
        git_hash: None,
        git_ref: None,
        source_type: crate::cache::SourceType::Tarball,
        is_temp: true,
        is_dirty: false,
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

/// Fetch releases.json from kernel.org and return (moniker, version) pairs.
pub(crate) fn fetch_releases() -> Result<Vec<(String, String)>, String> {
    let url = "https://www.kernel.org/releases.json";
    let response = reqwest::blocking::get(url).map_err(|e| format!("fetch {url}: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("fetch {url}: HTTP {}", response.status()));
    }
    let body = response
        .text()
        .map_err(|e| format!("read response body: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("parse releases.json: {e}"))?;
    let releases = json
        .get("releases")
        .and_then(|r| r.as_array())
        .ok_or_else(|| "releases.json: missing releases array".to_string())?;
    Ok(releases
        .iter()
        .filter_map(|r| {
            let moniker = r.get("moniker")?.as_str()?;
            let version = r.get("version")?.as_str()?;
            Some((moniker.to_string(), version.to_string()))
        })
        .collect())
}

/// Fetch the latest stable kernel version from kernel.org.
///
/// Selects from the `releases` array (moniker "stable" or "longterm"),
/// requiring at least 8 patch releases to avoid brand-new major versions
/// that may have build issues on CI runners.
pub fn fetch_latest_stable_version() -> Result<String, String> {
    eprintln!("ktstr: fetching latest kernel version");
    let releases = fetch_releases()?;

    let mut best: Option<&str> = None;
    for (moniker, version) in &releases {
        if moniker != "stable" && moniker != "longterm" {
            continue;
        }
        if patch_level(version).unwrap_or(0) < 8 {
            continue;
        }
        // Pick the first matching release — releases.json is ordered
        // newest first, so the first stable with patch >= 8 is the best.
        best = Some(version.as_str());
        break;
    }

    let version =
        best.ok_or_else(|| "no stable kernel with patch >= 8 found in releases.json".to_string())?;
    eprintln!("ktstr: latest stable kernel: {version}");
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

/// Resolve the highest version matching a prefix.
///
/// E.g., "6.12" → "6.12.81", "6" → "6.19.12" (highest 6.x.y).
///
/// Scans all monikers in releases.json except linux-next. If no
/// match is found (EOL series), probes cdn.kernel.org with HEAD
/// requests to find the highest patch version with a tarball.
pub fn fetch_version_for_prefix(prefix: &str) -> Result<String, String> {
    eprintln!("ktstr: fetching latest {prefix}.x kernel version");
    let releases = fetch_releases()?;

    let mut best: Option<(&str, (u32, u32, u32))> = None;
    for (moniker, version) in &releases {
        if moniker == "linux-next" {
            continue;
        }
        if !version.starts_with(prefix) {
            continue;
        }
        if version.len() != prefix.len() && version.as_bytes()[prefix.len()] != b'.' {
            continue;
        }
        let Some(tuple) = version_tuple(version) else {
            continue;
        };
        if best.is_none() || tuple > best.unwrap().1 {
            best = Some((version.as_str(), tuple));
        }
    }

    if let Some((version, _)) = best {
        eprintln!("ktstr: latest {prefix}.x kernel: {version}");
        return Ok(version.to_string());
    }

    eprintln!("ktstr: {prefix}.x not in releases.json (EOL series), probing cdn.kernel.org");
    probe_latest_patch(prefix)
}

/// Probe cdn.kernel.org to find the highest patch version for an EOL series.
///
/// Sends HEAD requests for {prefix}.1, {prefix}.2, ... until a 404.
/// Returns the last version that returned 200.
fn probe_latest_patch(prefix: &str) -> Result<String, String> {
    let major = major_version(prefix)?;
    let client = reqwest::blocking::Client::new();

    let mut last_good: Option<String> = None;
    for patch in 1u32.. {
        let version = format!("{prefix}.{patch}");
        let url =
            format!("https://cdn.kernel.org/pub/linux/kernel/v{major}.x/linux-{version}.tar.xz");
        let response = client
            .head(&url)
            .send()
            .map_err(|e| format!("HEAD {url}: {e}"))?;
        if !response.status().is_success() {
            break;
        }
        last_good = Some(version);
    }

    let version =
        last_good.ok_or_else(|| format!("no tarball found for {prefix}.x on cdn.kernel.org"))?;
    eprintln!("ktstr: latest {prefix}.x kernel (from cdn probe): {version}");
    Ok(version)
}

/// Clone a git repository with shallow depth.
pub fn git_clone(url: &str, git_ref: &str, dest_dir: &Path) -> Result<AcquiredSource, String> {
    let (arch, _) = arch_info();
    eprintln!("ktstr: cloning {url} (ref: {git_ref}, depth: 1)");

    let clone_dir = dest_dir.join("linux");

    let mut prep = gix::prepare_clone(url, &clone_dir)
        .map_err(|e| format!("prepare clone: {e}"))?
        .with_shallow(gix::remote::fetch::Shallow::DepthAtRemote(
            NonZeroU32::new(1).expect("1 is nonzero"),
        ))
        .with_ref_name(Some(git_ref))
        .map_err(|e| format!("set ref name: {e}"))?;

    let (mut checkout, _outcome) = prep
        .fetch_then_checkout(
            gix::progress::Discard,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .map_err(|e| format!("clone fetch: {e}"))?;

    let (_repo, _outcome) = checkout
        .main_worktree(
            gix::progress::Discard,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .map_err(|e| format!("checkout: {e}"))?;

    // Extract the short commit hash from HEAD.
    let repo = gix::open(&clone_dir).map_err(|e| format!("open cloned repo: {e}"))?;
    let head = repo.head_id().map_err(|e| format!("read HEAD: {e}"))?;
    let short_hash = format!("{}", head).chars().take(7).collect::<String>();

    let cache_key = format!(
        "{git_ref}-git-{short_hash}-{arch}-kc{}",
        crate::cache_key_suffix()
    );

    Ok(AcquiredSource {
        source_dir: clone_dir,
        cache_key,
        version: None,
        git_hash: Some(short_hash),
        git_ref: Some(git_ref.to_string()),
        source_type: crate::cache::SourceType::Git,
        is_temp: true,
        is_dirty: false,
    })
}

/// Use a local kernel source tree.
///
/// Performs dirty detection via gix (`Repository::is_dirty`) to
/// determine if the working tree has modifications to tracked files.
/// Untracked files do not affect the dirty flag.
pub fn local_source(source_path: &Path) -> Result<AcquiredSource, String> {
    let (arch, _) = arch_info();

    if !source_path.is_dir() {
        return Err(format!("{}: not a directory", source_path.display()));
    }

    let canonical = source_path
        .canonicalize()
        .map_err(|e| format!("canonicalize {}: {e}", source_path.display()))?;

    // Git hash extraction and dirty detection via gix.
    // Use the status API directly instead of is_dirty() to skip
    // submodule checks (which produce false positives on kernel
    // trees with uninitialized submodules).
    let (short_hash, is_dirty) = match gix::discover(&canonical) {
        Ok(repo) => {
            let head = repo.head_id().map_err(|e| format!("read HEAD: {e}"))?;
            let short_hash = format!("{}", head).chars().take(7).collect::<String>();

            // Check HEAD-vs-index for tracked file changes.
            let mut index_dirty = false;
            let index = repo
                .index_or_empty()
                .map_err(|e| format!("open index: {e}"))?;
            let _ = repo.tree_index_status(
                &head,
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
                    .map_err(|e| format!("status: {e}"))?
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

            (Some(short_hash), index_dirty || worktree_dirty)
        }
        Err(_) => {
            eprintln!(
                "ktstr: warning: {} is not a git repository, cannot detect dirty state",
                source_path.display()
            );
            (None, true)
        }
    };

    let suffix = crate::cache_key_suffix();
    let cache_key = match &short_hash {
        Some(hash) => format!("local-{hash}-{arch}-kc{suffix}"),
        None => format!("local-unknown-{arch}-kc{suffix}"),
    };

    Ok(AcquiredSource {
        source_dir: canonical,
        cache_key,
        version: None,
        git_hash: short_hash,
        git_ref: None,
        source_type: crate::cache::SourceType::Local,
        is_temp: false,
        is_dirty,
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
    fn stable_tarball_url(version: &str) -> Result<String, String> {
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

    // -- proptest --

    proptest::proptest! {
        #[test]
        fn prop_major_version_never_panics(s in "\\PC{0,20}") {
            let _ = major_version(&s);
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
