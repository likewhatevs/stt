//! Kernel-resolution dispatch: cache lookup, version download,
//! source-tree auto-build, range expansion, git fetch.
//!
//! Holds the entry points that turn a `--kernel` spec into a
//! bootable image path or cache-entry directory:
//! [`resolve_kernel_image`], [`resolve_kernel_dir`],
//! [`resolve_kernel_dir_to_entry`], [`resolve_cached_kernel`].
//! Range and git wrappers live alongside ([`expand_kernel_range`],
//! [`resolve_git_kernel`]). `--include-files` resolution
//! ([`resolve_include_files`]) and the rayon-pool sizing helper
//! ([`resolve_kernel_parallelism`]) share the module because both
//! are dispatch-time helpers.

use std::path::Path;

use anyhow::{Result, anyhow, bail};

use super::kernel_build::kernel_build_pipeline;
use super::util::{Spinner, status, success};

/// Pre-flight check for /dev/kvm availability and permissions.
pub fn check_kvm() -> Result<()> {
    use std::path::Path;
    if !Path::new("/dev/kvm").exists() {
        bail!(
            "/dev/kvm not found. KVM requires:\n  \
             - Linux kernel with KVM support (CONFIG_KVM)\n  \
             - Access to /dev/kvm (check permissions or add user to 'kvm' group)\n  \
             - Hardware virtualization enabled in BIOS (VT-x/AMD-V)"
        );
    }
    if let Err(e) = std::fs::File::open("/dev/kvm") {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            bail!(
                "/dev/kvm: permission denied. Add your user to the 'kvm' group:\n  \
                 sudo usermod -aG kvm $USER\n  \
                 then log out and back in."
            );
        }
        bail!("/dev/kvm: {e}");
    }
    Ok(())
}

/// Resolve the rayon pool width for `cargo ktstr`'s
/// `resolve_kernel_set` per-spec fan-out.
///
/// Reads [`crate::KTSTR_KERNEL_PARALLELISM_ENV`] first; if the env
/// var is set to a non-zero, parseable `usize`, that value wins.
/// Otherwise falls back to [`std::thread::available_parallelism`]
/// — the host's logical CPU count, the right ceiling for
/// download-bound work that should not outnumber the threads the
/// host can drive without thrashing the local network. Final
/// fallback is `1` if `available_parallelism` errors (a sandboxed
/// or container-limited host), preserving forward progress.
///
/// Sentinel handling: `0` and unparseable values fall through
/// (`from_str` errs on non-digits, and the explicit `n > 0`
/// guard rejects the parsed-zero case). A typoed export
/// (`KTSTR_KERNEL_PARALLELISM=abc` or `=0`) degrades to the
/// host-CPU default rather than disabling parallelism — a
/// disabled-pool resolve would serialize multi-spec invocations
/// with no observable signal that the env var was the cause.
/// The fall-through path emits a `tracing::warn!` carrying the
/// raw value so the operator sees their typo'd export was
/// ignored; the default still applies so forward progress is
/// preserved. Leading/trailing whitespace is trimmed before
/// parsing so a shell-quoted `=" 8 "` behaves the same as the
/// unquoted form.
///
/// Extracted from cargo-ktstr's `resolve_kernel_set` so the
/// parsing rules live in one place; the cargo-ktstr binary
/// invokes this and feeds the result into
/// [`rayon::ThreadPoolBuilder::num_threads`]. Lives in the
/// `cli` module rather than in the binary so it's reachable
/// from rustdoc and from the lib's unit-test harness.
pub fn resolve_kernel_parallelism() -> usize {
    if let Ok(raw) = std::env::var(crate::KTSTR_KERNEL_PARALLELISM_ENV) {
        let trimmed = raw.trim();
        match trimmed.parse::<usize>() {
            Ok(n) if n > 0 => return n,
            _ => {
                tracing::warn!(
                    env_var = crate::KTSTR_KERNEL_PARALLELISM_ENV,
                    value = %raw,
                    "KTSTR_KERNEL_PARALLELISM={raw:?} failed to parse, using default",
                );
            }
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Search PATH for a bare executable name.
fn resolve_in_path(name: &std::path::Path) -> Option<std::path::PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if let Ok(meta) = std::fs::metadata(&candidate)
            && meta.is_file()
            && meta.permissions().mode() & 0o111 != 0
        {
            return Some(candidate);
        }
    }
    None
}

/// Resolve `--include-files` arguments into `(archive_path, host_path)` pairs.
///
/// Each path is resolved as follows:
/// - Explicit paths (starting with `/`, `.`, `..`, or containing `/`): must exist.
/// - Bare names: searched in PATH.
/// - Directories: walked recursively via `walkdir`, following symlinks.
///   The directory's basename becomes the root under `include-files/`.
///   Non-regular files (sockets, pipes, device nodes) are skipped.
///   Empty directories produce a warning to stderr.
/// - Regular files: included directly as `include-files/<filename>`.
pub fn resolve_include_files(
    paths: &[std::path::PathBuf],
) -> Result<Vec<(String, std::path::PathBuf)>> {
    use std::path::{Component, PathBuf};

    let mut resolved_includes: Vec<(String, PathBuf)> = Vec::new();
    for path in paths {
        let is_explicit_path = {
            matches!(
                path.components().next(),
                Some(Component::RootDir | Component::CurDir | Component::ParentDir)
            ) || path.components().count() > 1
        };
        let resolved = if is_explicit_path {
            anyhow::ensure!(
                path.exists(),
                "--include-files path not found: {}",
                path.display()
            );
            path.clone()
        } else {
            // Bare name: search PATH.
            if path.exists() {
                path.clone()
            } else {
                resolve_in_path(path).ok_or_else(|| {
                    anyhow::anyhow!("-i {}: not found in filesystem or PATH", path.display())
                })?
            }
        };
        if resolved.is_dir() {
            let dir_name = resolved
                .file_name()
                .ok_or_else(|| {
                    anyhow::anyhow!("include directory has no name: {}", resolved.display())
                })?
                .to_string_lossy()
                .to_string();
            let prefix = format!("include-files/{dir_name}");
            let mut count = 0usize;
            for entry in walkdir::WalkDir::new(&resolved).follow_links(true) {
                let entry = entry.map_err(|e| anyhow::anyhow!("-i {}: {e}", resolved.display()))?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let rel = entry
                    .path()
                    .strip_prefix(&resolved)
                    .expect("walkdir entry is under root");
                let archive_path = format!("{prefix}/{}", rel.display());
                resolved_includes.push((archive_path, entry.into_path()));
                count += 1;
            }
            if count == 0 {
                eprintln!(
                    "warning: -i {}: directory contains no regular files",
                    resolved.display()
                );
            }
        } else {
            let file_name = resolved
                .file_name()
                .ok_or_else(|| {
                    anyhow::anyhow!("include file has no filename: {}", resolved.display())
                })?
                .to_string_lossy();
            let archive_path = format!("include-files/{file_name}");
            resolved_includes.push((archive_path, resolved));
        }
    }

    // Detect duplicate archive paths (e.g. `-i ./a/dir -i ./b/dir` both
    // containing the same relative file). The cpio format silently
    // overwrites earlier entries, so duplicates must be caught here.
    let mut seen = std::collections::HashMap::<&str, &std::path::Path>::new();
    for (archive_path, host_path) in &resolved_includes {
        if let Some(prev) = seen.insert(archive_path.as_str(), host_path.as_path()) {
            anyhow::bail!(
                "duplicate include path '{}': provided by both {} and {}",
                archive_path,
                prev.display(),
                host_path.display(),
            );
        }
    }

    Ok(resolved_includes)
}

/// Look up a cache key, checking local first, then remote (if enabled).
///
/// `cli_label` prefixes diagnostic output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
pub fn cache_lookup(
    cache: &crate::cache::CacheDir,
    cache_key: &str,
    cli_label: &str,
) -> Option<crate::cache::CacheEntry> {
    // `CacheDir::lookup` emits the per-lookup "unstripped vmlinux"
    // warning on any local hit whose entry was stored via the
    // strip-fallback path. The remote-lookup path here funnels
    // downloads through `CacheDir::store`, which runs its own strip
    // pipeline and reports via eprintln at store time — so the
    // warning coverage is uniform across local and remote cache
    // hits without an additional check here.
    if let Some(entry) = cache.lookup(cache_key) {
        return Some(entry);
    }

    if crate::remote_cache::is_enabled() {
        return crate::remote_cache::remote_lookup(cache, cache_key, cli_label);
    }

    None
}

/// Resolve a Version or CacheKey identifier to a cache entry directory.
///
/// Lookup order: local cache, then the remote GHA cache when
/// `remote_cache::is_enabled()` returns true. Miss behavior differs
/// by variant:
/// - **Version**: major.minor prefixes (e.g. `"6.14"`) resolve to
///   the latest patch via [`crate::fetch::fetch_version_for_prefix`]
///   first. On full miss, downloads the kernel from kernel.org,
///   builds it, and stores it in the cache via
///   [`download_and_cache_version`].
/// - **CacheKey**: errors on miss — cache keys are content-hashes
///   and not downloadable. The error hint suggests running
///   `{cli_label} kernel list`.
///
/// `cli_label` is the human-facing command name (`"ktstr"` or
/// `"cargo ktstr"`) threaded into status output and error messages.
pub fn resolve_cached_kernel(
    id: &crate::kernel_path::KernelId,
    cli_label: &str,
) -> Result<std::path::PathBuf> {
    use crate::kernel_path::KernelId;
    match id {
        KernelId::Version(ver) => {
            // Major.minor prefix (e.g. "6.14") → resolve to latest patch.
            let resolved = if crate::fetch::is_major_minor_prefix(ver) {
                crate::fetch::fetch_version_for_prefix(
                    crate::fetch::shared_client(),
                    ver,
                    cli_label,
                )?
            } else {
                ver.clone()
            };
            let cache = crate::cache::CacheDir::new()?;
            let (arch, _) = crate::fetch::arch_info();
            let cache_key = format!("{resolved}-tarball-{arch}-kc{}", crate::cache_key_suffix());
            if let Some(entry) = cache_lookup(&cache, &cache_key, cli_label) {
                // lookup() returns Some only for valid-metadata entries.
                return Ok(entry.path);
            }
            // Cache miss: download and build the requested version.
            // cpu_cap is None here — resolve_cached_kernel is reached
            // from test/coverage/shell/run/verifier (via
            // resolve_kernel_image), and --cpu-cap is scoped to the
            // explicit `kernel build` / `shell --no-perf-mode` paths
            // only; the auto-build-on-miss codepath is outside that
            // scope by design.
            download_and_cache_version(&resolved, cli_label, None)
        }
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new()?;
            if let Some(entry) = cache_lookup(&cache, key, cli_label) {
                return Ok(entry.path);
            }
            bail!(
                "cache key {key} not found. \
                 Run `{cli_label} kernel list` to see available entries."
            )
        }
        KernelId::Path(_) => bail!("resolve_cached_kernel called with Path variant"),
        // Multi-kernel specs cannot resolve to a single cache entry.
        // This function returns one path; range/git fan-out belongs
        // upstream in the dispatch loop that iterates kernels. Bail
        // with an actionable redirect that cites the value the user
        // wrote — `KernelId::Display` renders Range as `start..end`
        // and Git as `git+URL#REF`, matching the sibling cache-key
        // bail above that cites `{key}`.
        //
        // Run `validate()` first so an inverted range surfaces the
        // specific "swap the endpoints" diagnostic before the
        // generic "not yet supported" redirect masks it. Operators
        // with a typo see the actionable fix; valid-but-unsupported
        // specs get the redirect.
        KernelId::Range { .. } | KernelId::Git { .. } => {
            id.validate()
                .map_err(|e| anyhow::anyhow!("--kernel {id}: {e}"))?;
            bail!(
                "--kernel {id}: kernel ranges and git sources are not \
                 yet supported in this context — use a single kernel \
                 version, cache key, or path"
            )
        }
    }
}

/// Policy controlling `resolve_kernel_image` behavior across binaries.
///
/// The resolution pipeline — directory auto-build, version
/// auto-download, cache lookup — is shared. `KernelResolvePolicy`
/// carries the per-binary knobs documented on each field.
pub struct KernelResolvePolicy<'a> {
    /// Accept raw kernel image files (e.g. `bzImage`, `Image`) passed
    /// as `--kernel`. `ktstr` uses `false` (rejects); `cargo ktstr`
    /// uses `true` (accepts).
    pub accept_raw_image: bool,
    /// CLI label for diagnostic status messages (e.g. `"ktstr"`,
    /// `"cargo ktstr"`), threaded into auto-build and auto-download
    /// status output.
    pub cli_label: &'a str,
}

/// Resolve a kernel identifier to a bootable image path.
///
/// Handles `KernelId` variants: directory (auto-build), version
/// string, and cache key. Raw image file acceptance is controlled by
/// `policy.accept_raw_image`. The `None` case resolves automatically
/// via cache then filesystem, falling back to auto-download.
pub fn resolve_kernel_image(
    kernel: Option<&str>,
    policy: &KernelResolvePolicy<'_>,
) -> Result<std::path::PathBuf> {
    use crate::kernel_path::KernelId;

    if let Some(val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let path = std::path::PathBuf::from(&p);
                if path.is_dir() {
                    // `None` for cpu_cap: resolve_kernel_image is
                    // called by test/coverage/shell/run/verifier —
                    // subcommands where --cpu-cap is not exposed.
                    // The two kernel-build entry points
                    // (ktstr/cargo-ktstr `kernel build`) call
                    // resolve_kernel_dir directly with their flag-
                    // derived cap and do NOT go through
                    // resolve_kernel_image.
                    resolve_kernel_dir(&path, policy.cli_label, None)
                } else if path.is_file() {
                    if policy.accept_raw_image {
                        Ok(path)
                    } else {
                        // Raw kernel image file — reject. Use a source
                        // directory or version string so kconfig validation
                        // and caching work correctly.
                        bail!(
                            "--kernel {}: raw image files are not supported. \
                             Pass a source directory, version, or cache key.",
                            path.display()
                        )
                    }
                } else {
                    bail!("kernel path not found: {}", path.display())
                }
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = resolve_cached_kernel(&id, policy.cli_label)?;
                crate::kernel_path::find_image_in_dir(&cache_dir).ok_or_else(|| {
                    anyhow::anyhow!("no kernel image found in {}", cache_dir.display())
                })
            }
            // Multi-kernel specs cannot resolve to a single image
            // here. The dispatch loop that fans out range expansion
            // and git fetch lives one level up at the test/coverage/
            // verifier subcommand entry; this resolver is the
            // single-kernel leaf. Bail with an actionable redirect
            // so the user knows the spec is recognised but the
            // calling subcommand hasn't wired up the multi-kernel
            // pipeline yet.
            //
            // Run `validate()` first so an inverted range surfaces
            // the specific "swap the endpoints" diagnostic before
            // the generic "not yet supported" redirect masks it.
            id @ (KernelId::Range { .. } | KernelId::Git { .. }) => {
                id.validate()
                    .map_err(|e| anyhow::anyhow!("--kernel {val}: {e}"))?;
                bail!(
                    "--kernel {val}: kernel ranges and git sources are not \
                     yet supported in this context — use a single kernel \
                     version, cache key, or path"
                )
            }
        }
    } else {
        match crate::find_kernel()? {
            Some(image) => Ok(image),
            None => auto_download_kernel(policy.cli_label),
        }
    }
}

/// Auto-download, build, and cache the latest stable kernel.
///
/// Called when no --kernel is specified and no kernel is found via
/// cache or filesystem. Resolves the latest stable version and
/// delegates to [`download_and_cache_version`]. `cli_label` prefixes
/// status output (e.g. `"ktstr"`, `"cargo ktstr"`).
pub fn auto_download_kernel(cli_label: &str) -> Result<std::path::PathBuf> {
    status(&format!(
        "{cli_label}: no kernel found, downloading latest stable"
    ));

    let sp = Spinner::start("Fetching latest kernel version...");
    let ver = crate::fetch::fetch_latest_stable_version(crate::fetch::shared_client(), cli_label)?;
    sp.finish(format!("Latest stable: {ver}"));

    let cache_dir = download_and_cache_version(&ver, cli_label, None)?;
    let (_, image_name) = crate::fetch::arch_info();
    Ok(cache_dir.join(image_name))
}

/// Download a specific kernel version, build it, and store in the
/// cache. Returns the cache entry directory path (NOT the image path).
///
/// Checks the cache one more time with the resolved version to cover
/// races and prefix-resolved entries. Delegates to
/// [`kernel_build_pipeline`] for configure/build/validate/cache.
///
/// `cpu_cap` forwards the resource-budget cap to the pipeline so
/// the LLC flock + cgroup sandbox phases honour it. `None` means
/// "reserve 30% of the allowed-CPU set" (see
/// [`CpuCap::resolve`](crate::vmm::host_topology::CpuCap::resolve)).
pub fn download_and_cache_version(
    version: &str,
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<std::path::PathBuf> {
    let (arch, _) = crate::fetch::arch_info();
    let cache_key = format!("{version}-tarball-{arch}-kc{}", crate::cache_key_suffix());

    // Check cache one more time with the resolved version.
    if let Ok(cache) = crate::cache::CacheDir::new()
        && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label)
    {
        return Ok(entry.path);
    }

    let tmp_dir = tempfile::TempDir::new()?;

    let sp = Spinner::start("Downloading kernel...");
    let acquired = crate::fetch::download_tarball(
        crate::fetch::shared_client(),
        version,
        tmp_dir.path(),
        cli_label,
    )?;
    sp.finish("Downloaded");

    let cache = crate::cache::CacheDir::new()?;
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, false, cpu_cap, None)?;

    match result.entry {
        Some(entry) => Ok(entry.path),
        None => bail!(
            "kernel built but cache store failed — cannot return image from temporary directory"
        ),
    }
}

/// Expand a kernel-version range to the list of stable / longterm
/// releases that fall inside `[start, end]` inclusive.
///
/// Fetches kernel.org's `releases.json` once via
/// [`crate::fetch::cached_releases`], filters to rows whose `moniker`
/// is `stable` or `longterm` (matching the policy
/// [`crate::fetch::fetch_latest_stable_version`] uses for "is this a
/// production release we want to test against"), drops any version
/// outside the inclusive interval, and returns the surviving versions
/// sorted ascending by `(major, minor, patch, rc)` tuple. Empty result
/// is a hard error — an empty range either reflects a typo (start/end
/// don't bracket any active series) or releases.json missing rows
/// the operator expected, and silently iterating over zero kernels
/// would mask both. The `KernelId::Range` doc comment promises "every
/// release in the range" which a quiet no-op contradicts.
///
/// Range endpoints are NOT required to appear in releases.json — the
/// interval is half-the-numeric, half-presence: `6.10..6.16` selects
/// every stable release strictly inside that span, regardless of
/// whether `6.10` and `6.16` themselves are still listed (e.g. one
/// has been pruned from active maintenance). This matches the
/// inclusive-numeric-comparison semantics in
/// [`crate::kernel_path::KernelId::validate`] and lets a range from
/// an EOL series survive even after the endpoint version itself
/// becomes unavailable.
///
/// `cli_label` prefixes the kernel.org-fetch status line so the
/// diagnostic matches the binary that triggered the lookup
/// (`"ktstr"` vs `"cargo ktstr"`).
///
/// Pre-release filter: `mainline` and `linux-next` rows are
/// excluded by the moniker filter; rc tags carrying a stable
/// moniker would also be excluded but kernel.org publishes rcs
/// under `mainline`, so the filter is double-coverage in practice.
/// Operators who want to test against an rc spell it out as a
/// single `--kernel 6.16-rc3` rather than expecting the range
/// expansion to surface it.
pub fn expand_kernel_range(start: &str, end: &str, cli_label: &str) -> Result<Vec<String>> {
    use crate::kernel_path::decompose_version_for_compare;

    let start_key = decompose_version_for_compare(start).ok_or_else(|| {
        anyhow!(
            "kernel range start `{start}` is not a parseable version. \
             Endpoints must match `MAJOR.MINOR[.PATCH][-rcN]`."
        )
    })?;
    let end_key = decompose_version_for_compare(end).ok_or_else(|| {
        anyhow!(
            "kernel range end `{end}` is not a parseable version. \
             Endpoints must match `MAJOR.MINOR[.PATCH][-rcN]`."
        )
    })?;

    eprintln!("{cli_label}: expanding kernel range {start}..{end}");
    // Cached fetch: peer Range specs running in parallel under
    // `cargo ktstr`'s `resolve_kernel_set` rayon pipeline share
    // one network round-trip. The first Range to reach this
    // helper populates [`crate::fetch::RELEASES_CACHE`]; every
    // subsequent `--kernel A..B` call clones the cached vector
    // and skips the kernel.org GET. A transient outage on the
    // first call returns Err and leaves the cache un-populated,
    // so the next caller re-attempts the network — failures
    // never poison the cache.
    let releases = crate::fetch::cached_releases()?;

    let versions = filter_and_sort_range(&releases, start_key, end_key);
    if versions.is_empty() {
        bail!(
            "kernel range {start}..{end} expanded to 0 stable releases. \
             releases.json has no `stable` or `longterm` rows in this \
             interval — verify the endpoints, or use a single \
             `--kernel <version>` if you want a pre-release or \
             archived version."
        );
    }

    eprintln!(
        "{cli_label}: range expanded to {n} kernel(s): {list}",
        n = versions.len(),
        list = versions.join(", "),
    );
    Ok(versions)
}

/// Filter [`Release`](crate::fetch::Release) rows to stable+longterm
/// versions inside `[start_key, end_key]` and return them sorted
/// ascending by version tuple.
///
/// Separated from [`expand_kernel_range`] so the pure filter+sort
/// logic — moniker rejection, version-tuple bounds check, sort
/// order — is testable without hitting the network. The wrapper is
/// a thin adapter that fetches `releases.json` and reports the
/// outcome to stderr; this helper carries no I/O. Mirrors the
/// `active_prefixes_from_releases` split applied above.
fn filter_and_sort_range(
    releases: &[crate::fetch::Release],
    start_key: (u64, u64, u64, u64),
    end_key: (u64, u64, u64, u64),
) -> Vec<String> {
    use crate::kernel_path::decompose_version_for_compare;

    let mut selected: Vec<(String, (u64, u64, u64, u64))> = Vec::new();
    for r in releases {
        if r.moniker != "stable" && r.moniker != "longterm" {
            continue;
        }
        let Some(key) = decompose_version_for_compare(&r.version) else {
            continue;
        };
        if key < start_key || key > end_key {
            continue;
        }
        selected.push((r.version.clone(), key));
    }
    selected.sort_by_key(|s| s.1);
    selected.into_iter().map(|(v, _)| v).collect()
}

/// Resolve a `git+URL#REF` kernel spec to a cache-entry directory.
///
/// Mirrors [`download_and_cache_version`] for the git source path:
/// shallow-clones the repo into a temp directory via
/// [`crate::fetch::git_clone`], checks the resulting cache key for an
/// existing entry (so two consecutive `cargo ktstr test --kernel
/// git+URL#main` invocations against an unchanged tip skip the rebuild),
/// and on miss delegates to [`kernel_build_pipeline`] for
/// configure/build/validate/cache. Returns the cache entry directory
/// path — the same shape `download_and_cache_version` returns and the
/// same shape callers feed into the [`crate::KTSTR_KERNEL_ENV`] export.
///
/// Branches resolve at clone time (the shallow fetch lands on the
/// branch's current tip; the resulting `short_hash` is what the cache
/// key embeds). Two operators cloning `git+URL#main` at different
/// times produce different cache keys when the branch tip has moved
/// — that is intentional for this stage. A future Stage 3 ls-remote
/// pre-resolution would collapse identical-sha-different-spelling
/// invocations to one cache entry; until then the doc comment on
/// [`crate::kernel_path::KernelId::Git`] tracks that as future work.
///
/// `cli_label` matches the contract the sibling helpers
/// (`download_and_cache_version`, `resolve_kernel_dir`) use:
/// it prefixes diagnostic status output and is threaded into
/// [`kernel_build_pipeline`].
pub fn resolve_git_kernel(url: &str, git_ref: &str, cli_label: &str) -> Result<std::path::PathBuf> {
    let tmp_dir = tempfile::TempDir::new()?;

    let acquired = crate::fetch::git_clone(url, git_ref, tmp_dir.path(), cli_label)?;

    // Open cache once, reuse for both lookup (post-clone cache_key
    // embeds the resolved short_hash, so a repeat invocation against
    // an unchanged branch tip skips the rebuild) and the build
    // pipeline below on miss.
    let cache = crate::cache::CacheDir::new()?;
    if let Some(entry) = cache_lookup(&cache, &acquired.cache_key, cli_label) {
        return Ok(entry.path);
    }

    // is_local_source = false: a freshly cloned tree is treated the
    // same as a tarball download — no `make mrproper` skip-warning,
    // no compile_commands.json generation (acquired.is_temp gates
    // that inside the pipeline).
    // extra_kconfig = None: this entry path serves auto-discovery
    // (cargo ktstr test/coverage/llvm-cov), which doesn't expose
    // `--extra-kconfig`. The flag is `cargo ktstr kernel build`-only.
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, false, None, None)?;

    match result.entry {
        Some(entry) => Ok(entry.path),
        None => bail!(
            "kernel built from git+{url}#{git_ref} but cache store failed — \
             cannot return image from temporary directory"
        ),
    }
}

/// Cache-hit signal returned from [`resolve_kernel_dir_to_entry`]
/// when a clean source tree's cache entry was found and reused
/// without invoking [`kernel_build_pipeline`].
///
/// Carries the cache key and the persisted `built_at` ISO-8601
/// timestamp so callers can render a user-facing line that names
/// both the cache identity and the build age. `None`
/// ([`KernelDirOutcome::cache_hit`] returns `None`) means the build
/// pipeline ran — either to populate the cache (clean-tree cache
/// miss) or to build directly without storing (dirty-tree path).
#[derive(Debug, Clone)]
pub struct KernelDirCacheHit {
    /// Cache key that resolved to this entry, e.g.
    /// `local-abc1234-x86_64-kc{suffix}` or
    /// `local-abc1234-x86_64-cfgdeadbeef-kc{suffix}` when the source
    /// tree carried a user `.config`.
    pub cache_key: String,
    /// ISO-8601 timestamp recorded in the entry's `metadata.json`
    /// at store time. Suitable for `humantime::parse_rfc3339`.
    pub built_at: String,
}

/// Result bundle from [`resolve_kernel_dir_to_entry`].
///
/// Bundles the resolved boot-image directory, the cache-hit
/// signal, and the dirty-tree flag so callers do not have to
/// re-run `gix::open` to learn whether the build was reproducible.
/// The dirty flag is the single source of truth for downstream
/// label decoration ([`crate::test_support::sanitize_kernel_label`]'s
/// upstream caller appends `_dirty` so test reports show the run
/// used a non-reproducible build).
///
/// `non_exhaustive` so a future field (e.g. cache miss vs cache
/// store-failed distinction) can land without breaking external
/// destructuring. Construction goes through field literals at the
/// definition site only — every external consumer reads via the
/// public field accessors.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct KernelDirOutcome {
    /// Directory that holds the resolved boot image.
    ///
    /// - Clean tree: cache entry directory under one of the
    ///   `local-{hash7}-{arch}[-cfg{user_config}]-kc{suffix}` or
    ///   `local-unknown-{path_hash}-{arch}-kc{suffix}` shapes (see
    ///   [`crate::fetch::compose_local_cache_key`]); boot image at
    ///   `<dir>/<image_name>`.
    /// - Dirty tree: canonical source-tree directory, boot image at
    ///   `<dir>/arch/<arch>/boot/<image_name>`.
    ///
    /// Both shapes are valid inputs to
    /// [`crate::kernel_path::find_image_in_dir`].
    pub dir: std::path::PathBuf,
    /// `Some` when the resolution short-circuited on a cache hit —
    /// the build pipeline did not run. `None` when the build
    /// pipeline ran (clean-tree miss-then-build OR dirty-tree
    /// build-without-store). [`is_dirty`](Self::is_dirty)
    /// distinguishes the two `None` cases.
    pub cache_hit: Option<KernelDirCacheHit>,
    /// Whether the source tree was non-reproducible. Union of two
    /// signals:
    ///
    /// - Acquire-time inspection by
    ///   [`crate::fetch::local_source`] (uncommitted modifications
    ///   before the build started, OR a non-git tree that has no
    ///   commit hash to record).
    /// - Post-build re-check from
    ///   [`crate::fetch::inspect_local_source_state`] (worktree
    ///   edited, branch flipped, or commit landed during `make`).
    ///
    /// Either signal flips this to `true`. Always `false` on a cache
    /// hit — the cache lookup gate requires a clean tree at acquire
    /// time and the build pipeline does not run.
    pub is_dirty: bool,
}

/// Resolve a source-tree path through the local-kernel cache,
/// returning a [`KernelDirOutcome`] that carries the boot-image
/// directory, the cache-hit signal, and the dirty-tree flag.
///
/// For a clean source tree:
///   - Cache hit → `outcome.dir` is the cache entry directory,
///     `outcome.cache_hit` is `Some(KernelDirCacheHit)`,
///     `outcome.is_dirty` is `false`. The build pipeline does not
///     run.
///   - Cache miss, no mid-build mutation → runs
///     [`kernel_build_pipeline`] which builds in the source tree and
///     stores a stripped vmlinux + boot image under the cache entry;
///     `outcome.dir` is the cache entry directory,
///     `outcome.cache_hit` is `None`, `outcome.is_dirty` is `false`.
///   - Cache miss, mid-build mutation observed by the pipeline's
///     post-build re-check → the cache store is skipped to avoid
///     recording a stale identity, `outcome.dir` is the canonical
///     source-tree directory, `outcome.cache_hit` is `None`,
///     `outcome.is_dirty` is `true`.
///
/// For a dirty source tree:
///   - [`kernel_build_pipeline`] skips the cache store
///     (`is_dirty` short-circuit at the cache-store boundary) and
///     returns the source-tree image. `outcome.dir` is the
///     canonical source-tree directory (boot image at
///     `<source>/arch/<arch>/boot/<image_name>`),
///     `outcome.cache_hit` is `None`, `outcome.is_dirty` is
///     `true`. Callers use the dirty flag to mark the run as
///     non-reproducible in test reports — e.g. `cargo-ktstr`'s
///     Path-spec resolver appends `_dirty` to the kernel label
///     so a `path_linux_a3b1c2_dirty` row in the gauntlet output
///     surfaces the divergence from the cache-stored
///     `path_linux_a3b1c2` clean variant.
///
/// Both directory return shapes are valid inputs to
/// [`crate::kernel_path::find_image_in_dir`], which probes both
/// layouts. Callers that need the boot-image FILE path (not the
/// directory) should use [`resolve_kernel_dir`] instead — that
/// function applies the same pipeline but returns the image path.
///
/// Used by `cargo-ktstr`'s Path-spec resolver to wire `--kernel
/// PATH` invocations through the same cache pipeline that
/// Version/CacheKey/Git specs use, so a clean source-tree rebuild
/// hits the cache instead of re-running `make`.
///
/// `cli_label` prefixes status output and is threaded into
/// [`kernel_build_pipeline`]'s diagnostic surface. `cpu_cap`
/// forwards the resource-budget cap; `None` keeps the
/// 30%-of-allowed default. See [`resolve_kernel_dir`] for the
/// matching image-returning sibling's `cpu_cap` rationale —
/// identical here because both functions reach the same pipeline.
pub fn resolve_kernel_dir_to_entry(
    path: &std::path::Path,
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<KernelDirOutcome> {
    let acquired = acquire_local_source_tree(path)?;
    let cache_key = acquired.cache_key.clone();
    let is_dirty = acquired.is_dirty;
    // Open the cache once and reuse for both the clean-tree
    // lookup and the post-build store. Both legs need the same
    // root resolution; opening twice is wasted work and risks
    // a TOCTOU split if `KTSTR_CACHE_DIR` changes between calls.
    // A failure here is fatal — we cannot proceed without a cache
    // root for either lookup or store.
    let cache = crate::cache::CacheDir::new()?;

    // Clean trees: cache lookup before build.
    if !is_dirty && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label) {
        // `entry.path` is the cache entry directory; the boot
        // image lives at `<entry.path>/<image_name>`. Verify the
        // image is actually present before returning, so a
        // partially-corrupt entry doesn't bypass the
        // build-and-restore path.
        if entry.image_path().exists() {
            let hit = KernelDirCacheHit {
                cache_key: cache_key.clone(),
                built_at: entry.metadata.built_at.clone(),
            };
            return Ok(KernelDirOutcome {
                dir: entry.path,
                cache_hit: Some(hit),
                // Cache-hit gate already required clean tree —
                // restate the invariant in the outcome instead of
                // reading `is_dirty` again, so the bit cannot drift
                // if the gate condition above evolves.
                is_dirty: false,
            });
        }
    }

    // extra_kconfig = None: this path serves cargo ktstr
    // test/coverage/llvm-cov / shell / verifier resolution, none of
    // which expose `--extra-kconfig`. The flag is `cargo ktstr
    // kernel build`-only and feeds extras directly through that
    // dispatch.
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true, cpu_cap, None)?;

    // Prefer the cached entry directory (stable across rebuilds).
    // For dirty trees, `entry` is `None` — fall back to the
    // canonical source directory, which `local_source` already
    // resolved into `acquired.source_dir`.
    let dir = match result.entry {
        Some(entry) => entry.path,
        None => acquired.source_dir,
    };
    // The pipeline observes the dirty signal twice: once at acquire
    // time (captured in `is_dirty` above) and once via the post-build
    // re-check that detects mid-build mutations. Either source
    // flipping the bit means the run is non-reproducible — surface
    // the union here so the kernel-label downstream gets the `_dirty`
    // suffix even when the tree was clean at acquire and only
    // dirtied during `make`.
    Ok(KernelDirOutcome {
        dir,
        cache_hit: None,
        is_dirty: is_dirty || result.post_build_is_dirty,
    })
}

/// Resolve a kernel directory: auto-build from source tree.
///
/// Requires Makefile + Kconfig. Checks cache for clean trees,
/// delegates to [`kernel_build_pipeline`] on miss. `cli_label`
/// prefixes status output and is passed through to
/// [`kernel_build_pipeline`] as the diagnostic label.
///
/// `cpu_cap` forwards the resource-budget cap to the pipeline.
/// `None` is the default for non-kernel-build callers
/// (test/coverage/shell auto-build paths) — `--cpu-cap` lives on
/// the explicit kernel-build entrypoint, not test-running
/// commands, because the auto-build-on-miss path already runs
/// inside a test invocation where perf-mode constraints dominate.
pub fn resolve_kernel_dir(
    path: &std::path::Path,
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<std::path::PathBuf> {
    let acquired = acquire_local_source_tree(path)?;
    let cache_key = acquired.cache_key.clone();
    // Open the cache once and reuse for both the clean-tree
    // lookup and the post-build store. Both legs need the same
    // root resolution; opening twice is wasted work and risks
    // a TOCTOU split if `KTSTR_CACHE_DIR` changes between calls.
    // A failure here is fatal — we cannot proceed without a cache
    // root for either lookup or store. Mirrors the same hoist
    // applied in [`resolve_kernel_dir_to_entry`].
    let cache = crate::cache::CacheDir::new()?;

    // Clean trees: cache lookup before build.
    // Dirty trees: skip cache, always build.
    if !acquired.is_dirty
        && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label)
    {
        let image = entry.image_path();
        if image.exists() {
            success(&format!("{cli_label}: using cached kernel {cache_key}"));
            return Ok(image);
        }
    }

    // extra_kconfig = None: matches the sibling
    // `resolve_kernel_dir_to_entry` rationale — `--extra-kconfig` is
    // a `cargo ktstr kernel build`-only flag.
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true, cpu_cap, None)?;

    // Prefer the cached image path (stable across rebuilds).
    match result.entry {
        Some(entry) => Ok(entry.image_path()),
        None => Ok(result.image_path),
    }
}

/// Validate `path` is a kernel source tree (Makefile + Kconfig at
/// the root) and return the [`AcquiredSource`](crate::fetch::AcquiredSource)
/// computed by [`crate::fetch::local_source`].
///
/// Shared across [`resolve_kernel_dir`] and
/// [`resolve_kernel_dir_to_entry`] so the validation diagnostic
/// and `local_source` error stringification live in one place.
fn acquire_local_source_tree(path: &Path) -> Result<crate::fetch::AcquiredSource> {
    let is_source_tree = path.join("Makefile").exists() && path.join("Kconfig").exists();
    if !is_source_tree {
        bail!(
            "no kernel image found in {} (not a kernel source tree — \
             missing Makefile or Kconfig)",
            path.display()
        );
    }
    crate::fetch::local_source(path).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset env: returns the host-CPU fallback, never zero.
    #[test]
    fn resolve_kernel_parallelism_unset_returns_host_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::remove(crate::KTSTR_KERNEL_PARALLELISM_ENV);
        let n = resolve_kernel_parallelism();
        assert!(
            n >= 1,
            "fallback must yield at least 1; got {n} which would defeat \
             ThreadPoolBuilder::num_threads",
        );
    }

    /// Valid usize override: env-supplied value wins.
    #[test]
    fn resolve_kernel_parallelism_valid_override_wins() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "4");
        assert_eq!(
            resolve_kernel_parallelism(),
            4,
            "valid usize env value must override the host-CPU default",
        );
    }

    /// Zero is sentinel — falls through to default.
    #[test]
    fn resolve_kernel_parallelism_zero_falls_through_to_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "0");
        let n = resolve_kernel_parallelism();
        assert!(
            n >= 1,
            "zero env value must fall through to host-CPU default; got {n}",
        );
    }

    /// Unparseable falls through.
    #[test]
    fn resolve_kernel_parallelism_unparseable_falls_through_to_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "abc");
        let n = resolve_kernel_parallelism();
        assert!(n >= 1);
    }

    /// Negative falls through.
    #[test]
    fn resolve_kernel_parallelism_negative_falls_through_to_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "-1");
        let n = resolve_kernel_parallelism();
        assert!(n >= 1);
    }

    /// Trims whitespace.
    #[test]
    fn resolve_kernel_parallelism_trims_surrounding_whitespace() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "  8  ");
        assert_eq!(resolve_kernel_parallelism(), 8);
    }

    /// Pin env-var name literal.
    #[test]
    fn ktstr_kernel_parallelism_env_const_matches_literal() {
        assert_eq!(
            crate::KTSTR_KERNEL_PARALLELISM_ENV,
            "KTSTR_KERNEL_PARALLELISM",
        );
    }

    #[test]
    fn resolve_in_path_finds_sh() {
        let result = resolve_in_path(std::path::Path::new("sh"));
        assert!(result.is_some(), "sh should be in PATH");
        assert!(result.unwrap().exists());
    }

    #[test]
    fn resolve_in_path_nonexistent() {
        let result = resolve_in_path(std::path::Path::new("nonexistent_binary_xyz_12345"));
        assert!(result.is_none());
    }

    #[test]
    fn resolve_include_files_single_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("test.txt");
        std::fs::write(&file, "hello").unwrap();
        let result = resolve_include_files(&[file]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].0.contains("test.txt"));
    }

    #[test]
    fn resolve_include_files_nonexistent() {
        let result = resolve_include_files(&[std::path::PathBuf::from("/nonexistent/file.txt")]);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_include_files_bare_name_in_path() {
        let result = resolve_include_files(&[std::path::PathBuf::from("sh")]);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].0.contains("sh"));
    }

    /// Inverted-range diagnostic must surface ahead of the generic
    /// "not yet supported" bail when resolve_cached_kernel sees a Range.
    #[test]
    fn resolve_cached_kernel_surfaces_inverted_range_diagnostic() {
        let id = crate::kernel_path::KernelId::Range {
            start: "6.16".to_string(),
            end: "6.12".to_string(),
        };
        let err = resolve_cached_kernel(&id, "ktstr-test").expect_err("inverted range must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("inverted kernel range"),
            "validate() diagnostic must surface ahead of the generic \
             'not yet supported' bail; got: {msg}",
        );
        assert!(msg.contains("6.12..6.16"));
        assert!(!msg.contains("not yet supported in this context"));
    }

    /// Same wiring guarantee for resolve_kernel_image.
    #[test]
    fn resolve_kernel_image_surfaces_inverted_range_diagnostic() {
        let policy = KernelResolvePolicy {
            cli_label: "ktstr-test",
            accept_raw_image: false,
        };
        let err = resolve_kernel_image(Some("6.16..6.12"), &policy)
            .expect_err("inverted range must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("inverted kernel range"),
            "validate() diagnostic must surface ahead of the generic bail; got: {msg}",
        );
        assert!(msg.contains("6.12..6.16"));
        assert!(!msg.contains("not yet supported in this context"));
    }

    fn release(moniker: &str, version: &str) -> crate::fetch::Release {
        crate::fetch::Release {
            moniker: moniker.to_string(),
            version: version.to_string(),
        }
    }

    /// Stable+longterm rows inside the interval are kept; mainline,
    /// linux-next, and rows outside the interval are dropped.
    #[test]
    fn filter_and_sort_range_basic() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![
            release("mainline", "6.18-rc2"),
            release("stable", "6.16.5"),
            release("longterm", "6.12.40"),
            release("linux-next", "6.18-rc2-next-20260420"),
            release("longterm", "6.6.99"),
            release("stable", "6.14.10"),
            release("stable", "6.10.0"),
        ];
        let start_key = decompose_version_for_compare("6.12").unwrap();
        let end_key = decompose_version_for_compare("6.16.5").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert_eq!(
            out,
            vec![
                "6.12.40".to_string(),
                "6.14.10".to_string(),
                "6.16.5".to_string(),
            ],
        );
    }

    /// Endpoints absent from releases.json still bracket correctly.
    #[test]
    fn filter_and_sort_range_endpoints_absent_from_releases() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![
            release("stable", "6.12.5"),
            release("stable", "6.14.2"),
            release("stable", "6.15.0"),
        ];
        let start_key = decompose_version_for_compare("6.10").unwrap();
        let end_key = decompose_version_for_compare("6.16").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert_eq!(
            out,
            vec![
                "6.12.5".to_string(),
                "6.14.2".to_string(),
                "6.15.0".to_string(),
            ],
        );
    }

    /// Inclusive at both endpoints.
    #[test]
    fn filter_and_sort_range_inclusive_both_endpoints() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![
            release("stable", "6.12.5"),
            release("stable", "6.13.0"),
            release("stable", "6.14.2"),
        ];
        let start_key = decompose_version_for_compare("6.12.5").unwrap();
        let end_key = decompose_version_for_compare("6.14.2").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert_eq!(
            out,
            vec![
                "6.12.5".to_string(),
                "6.13.0".to_string(),
                "6.14.2".to_string(),
            ],
        );
    }

    /// rc-as-MAX ordering for rc tags under stable moniker.
    #[test]
    fn filter_and_sort_range_rc_under_stable_moniker_orders_after_release() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![
            release("stable", "6.14.0-rc3"),
            release("stable", "6.14.0"),
            release("stable", "6.13.0"),
        ];
        let start_key = decompose_version_for_compare("6.13").unwrap();
        let end_key = decompose_version_for_compare("6.15").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert_eq!(
            out,
            vec![
                "6.13.0".to_string(),
                "6.14.0-rc3".to_string(),
                "6.14.0".to_string(),
            ],
        );
    }

    /// Empty interval returns empty vec.
    #[test]
    fn filter_and_sort_range_empty_when_no_overlap() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![release("stable", "5.10.0"), release("stable", "5.15.0")];
        let start_key = decompose_version_for_compare("6.10").unwrap();
        let end_key = decompose_version_for_compare("6.16").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert!(out.is_empty(), "no overlap → empty result, got {out:?}");
    }

    /// Mainline/linux-next monikers are dropped even when in interval.
    #[test]
    fn filter_and_sort_range_drops_non_stable_monikers() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![
            release("mainline", "6.14.0"),
            release("linux-next", "6.14.0-next-20260420"),
            release("stable", "6.14.5"),
        ];
        let start_key = decompose_version_for_compare("6.14").unwrap();
        let end_key = decompose_version_for_compare("6.15").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert_eq!(out, vec!["6.14.5".to_string()]);
    }

    /// Unparseable versions silently dropped.
    #[test]
    fn filter_and_sort_range_drops_unparseable_versions() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![
            release("stable", "6.14.0"),
            release("stable", "embargoed-cve-tag"),
            release("stable", "6.14.5"),
        ];
        let start_key = decompose_version_for_compare("6.14").unwrap();
        let end_key = decompose_version_for_compare("6.15").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert_eq!(out, vec!["6.14.0".to_string(), "6.14.5".to_string()]);
    }

    /// expand_kernel_range rejects unparseable start endpoint.
    #[test]
    fn expand_kernel_range_rejects_unparseable_start() {
        let err = expand_kernel_range("garbage", "6.14", "ktstr-test")
            .expect_err("unparseable start must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("kernel range start `garbage`"));
    }

    #[test]
    fn expand_kernel_range_rejects_unparseable_end() {
        let err = expand_kernel_range("6.10", "garbage", "ktstr-test")
            .expect_err("unparseable end must error");
        let msg = format!("{err:#}");
        assert!(msg.contains("kernel range end `garbage`"));
    }

    // ---------------------------------------------------------------
    // resolve_kernel_dir_to_entry — success-path tests
    // ---------------------------------------------------------------
    //
    // Error paths (nonexistent path, not-a-source-tree) live next to
    // [`resolve_path_kernel`] in `bin/cargo_ktstr/kernel/mod.rs`. The success
    // paths exercise the full resolve → cache-lookup → outcome
    // pipeline with a real Makefile / Kconfig fixture, an
    // isolated `KTSTR_CACHE_DIR`, and a pre-populated cache entry
    // for the cache-hit case. The cache-miss + dirty-tree branches
    // are exercised through their predicate (`is_dirty=true` ⇒
    // skip cache lookup) without actually invoking
    // `kernel_build_pipeline`'s `make` subprocess — that would
    // require a real kernel toolchain and exceed unit-test scope.
    // The `is_dirty` branch is exercised by mutating the worktree
    // after commit and asserting the cache lookup is skipped (the
    // pre-populated entry is still present, so a successful lookup
    // would land it as the outcome — failing to do so proves the
    // dirty short-circuit fires).

    /// Initialise a git repo with one committed file, mirroring
    /// the helper in `fetch.rs`. Inlined here so the
    /// `resolve_kernel_dir_to_entry` tests are self-contained
    /// rather than reaching across the test-module boundary.
    /// `dir` MUST exist; the helper does not create it.
    fn init_repo_with_commit_for_resolve_test(dir: &std::path::Path) {
        use std::process::Command;
        let run = |args: &[&str]| {
            let out = Command::new("git")
                .args(args)
                .current_dir(dir)
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
        std::fs::write(dir.join("Makefile"), "# kernel makefile fixture\n").unwrap();
        std::fs::write(dir.join("Kconfig"), "# kernel kconfig fixture\n").unwrap();
        std::fs::write(dir.join("README"), "fixture\n").unwrap();
        run(&["add", "Makefile", "Kconfig", "README"]);
        run(&[
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "initial",
        ]);
    }

    /// Pre-populate a cache entry under `cache_root/{cache_key}/`
    /// containing a synthetic boot image and a `metadata.json`
    /// marking the entry as a [`crate::cache::KernelSource::Local`]
    /// build. Returns the entry path. The metadata's
    /// `source_tree_path` is NOT pinned to the test's source tree
    /// — `resolve_kernel_dir_to_entry`'s lookup gates only on
    /// cache key match, so any persisted metadata that round-trips
    /// is sufficient for the cache-hit assertion.
    fn populate_cache_entry_for_resolve_test(
        cache_root: &std::path::Path,
        cache_key: &str,
    ) -> std::path::PathBuf {
        let cache = crate::cache::CacheDir::with_root(cache_root.to_path_buf());
        let (arch, image_name) = crate::fetch::arch_info();
        let staging = tempfile::TempDir::new().expect("staging tempdir");
        let fake_image = staging.path().join(image_name);
        std::fs::write(&fake_image, b"fake kernel image bytes").expect("write fake image");
        let metadata = crate::cache::KernelMetadata::new(
            crate::cache::KernelSource::Local {
                source_tree_path: None,
                git_hash: None,
            },
            arch.to_string(),
            image_name.to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        let artifacts = crate::cache::CacheArtifacts::new(&fake_image);
        let entry = cache
            .store(cache_key, &artifacts, &metadata)
            .expect("pre-populate cache entry");
        entry.path
    }

    /// Cache hit — clean tree whose `local_source` cache key
    /// resolves to a pre-populated entry must short-circuit the
    /// build pipeline and surface `KernelDirOutcome` with
    /// `cache_hit = Some(...)`, `is_dirty = false`, and `dir`
    /// pointing at the cache entry directory (NOT the source
    /// tree).
    #[test]
    fn resolve_kernel_dir_to_entry_clean_tree_cache_hit() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("git CLI unavailable");
        }
        let _lock = crate::test_support::test_helpers::lock_env();
        let cache_tmp = tempfile::TempDir::new().expect("cache tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            cache_tmp.path(),
        );
        let src_tmp = tempfile::TempDir::new().expect("src tempdir");
        init_repo_with_commit_for_resolve_test(src_tmp.path());

        let acquired =
            crate::fetch::local_source(src_tmp.path()).expect("local_source must succeed");
        assert!(!acquired.is_dirty, "fixture must be clean before lookup");
        let cache_key = acquired.cache_key.clone();

        let entry_path = populate_cache_entry_for_resolve_test(cache_tmp.path(), &cache_key);

        let outcome = resolve_kernel_dir_to_entry(src_tmp.path(), "test", None)
            .expect("resolve must succeed on cache hit");
        assert_eq!(
            outcome.dir, entry_path,
            "cache-hit path must return the cache entry directory, NOT the source tree"
        );
        let hit = outcome
            .cache_hit
            .expect("cache hit must produce KernelDirCacheHit");
        assert_eq!(
            hit.cache_key, cache_key,
            "cache hit must report the resolved key"
        );
        assert_eq!(
            hit.built_at, "2026-04-12T10:00:00Z",
            "cache hit must surface the persisted built_at timestamp",
        );
        assert!(
            !outcome.is_dirty,
            "cache-hit gate requires a clean tree; outcome.is_dirty must be false",
        );
    }

    /// Dirty-tree resolve must short-circuit the cache lookup
    /// even when an entry under the dirty tree's would-be key
    /// already exists. `is_dirty=true` flips `outcome.is_dirty`
    /// so the caller (cargo-ktstr) appends `_dirty` to the
    /// kernel label and the test report distinguishes the
    /// non-reproducible run from a subsequent clean rebuild.
    #[test]
    fn resolve_kernel_dir_to_entry_dirty_tree_skips_cache_lookup() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("git CLI unavailable");
        }
        if std::process::Command::new("make")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("make not in PATH");
        }
        let _lock = crate::test_support::test_helpers::lock_env();
        let cache_tmp = tempfile::TempDir::new().expect("cache tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            cache_tmp.path(),
        );
        let _bypass_env =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_BYPASS_LLC_LOCKS", "1");
        let src_tmp = tempfile::TempDir::new().expect("src tempdir");
        init_repo_with_commit_for_resolve_test(src_tmp.path());

        std::fs::write(src_tmp.path().join("README"), "modified\n").expect("dirty README");
        let dirty_acquired = crate::fetch::local_source(src_tmp.path())
            .expect("local_source on dirty tree must succeed");
        assert!(
            dirty_acquired.is_dirty,
            "post-mutation tree must be dirty for the test to be meaningful"
        );
        populate_cache_entry_for_resolve_test(cache_tmp.path(), &dirty_acquired.cache_key);

        let result = resolve_kernel_dir_to_entry(src_tmp.path(), "test", None);
        match result {
            Ok(outcome) => panic!(
                "dirty tree must skip the cache lookup, but resolve returned \
                 Ok with dir={:?}, cache_hit={:?}, is_dirty={}",
                outcome.dir, outcome.cache_hit, outcome.is_dirty,
            ),
            Err(_) => {
                let entry_dir = cache_tmp.path().join(&dirty_acquired.cache_key);
                assert!(
                    entry_dir.is_dir(),
                    "pre-populated entry must still be present after the \
                     dirty resolve; the gate proved short-circuit by NOT \
                     returning this directory as the outcome.dir",
                );
            }
        }
    }

    /// Cache miss on a clean tree must surface as a build attempt
    /// rather than as a successful shortcut. Pinned through the
    /// build's eventual failure on a fixture without a real kernel
    /// toolchain — same shape as the dirty-tree test, but proves
    /// the cache MISS path also fans out to the pipeline (rather
    /// than a silent no-op that would erase the per-tree
    /// invariant).
    #[test]
    fn resolve_kernel_dir_to_entry_clean_tree_cache_miss_attempts_build() {
        if std::process::Command::new("git")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("git CLI unavailable");
        }
        if std::process::Command::new("make")
            .arg("--version")
            .output()
            .is_err()
        {
            skip!("make not in PATH");
        }
        let _lock = crate::test_support::test_helpers::lock_env();
        let cache_tmp = tempfile::TempDir::new().expect("cache tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            cache_tmp.path(),
        );
        let _bypass_env =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_BYPASS_LLC_LOCKS", "1");
        let src_tmp = tempfile::TempDir::new().expect("src tempdir");
        init_repo_with_commit_for_resolve_test(src_tmp.path());

        let acquired =
            crate::fetch::local_source(src_tmp.path()).expect("local_source must succeed");
        assert!(!acquired.is_dirty, "fixture must be clean before resolve");

        let result = resolve_kernel_dir_to_entry(src_tmp.path(), "test", None);
        assert!(
            result.is_err(),
            "cache miss without a real kernel toolchain must surface the build failure, \
             got Ok({:?})",
            result.as_ref().ok().map(|o| &o.dir),
        );
    }
}
