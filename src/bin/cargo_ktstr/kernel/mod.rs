//! Kernel resolution and build helpers for the `cargo ktstr` binary.
//!
//! Houses the spec-resolution pipeline that converts `--kernel
//! <SPEC>` arguments into bootable kernel directories, the
//! `kernel build` subcommand dispatcher
//! (`kernel_build` / `kernel_build_one` / `cache_lookup`), and the
//! `format_built_age` cache-hit log helper.
//!
//! The flat `(label, kernel_dir)` list this module emits is what
//! the `test` / `coverage` / `llvm-cov` dispatchers (in
//! [`super::run_cargo`]) hand to the test binary as the kernel
//! dimension of the gauntlet expansion via the
//! [`ktstr::KTSTR_KERNEL_LIST_ENV`] wire format.
//!
//! Pure label emission, the `KTSTR_KERNEL_LIST` wire encoder, and
//! the dedup / collision-detection helpers live in the
//! [`wire_format`] submodule — that subsystem is independently
//! unit-testable without driving the rayon resolve pipeline (every
//! `resolve_one` arm performs real I/O).

mod wire_format;

use std::path::{Path, PathBuf};

use ktstr::cache::{CacheDir, CacheEntry};
use ktstr::cli;
use ktstr::fetch;

pub(crate) use wire_format::{
    cache_key_to_version_label, decorate_path_label_for_dirty, dedupe_resolved,
    detect_label_collisions, encode_kernel_list, git_kernel_label, path_kernel_label,
    preflight_collision_check,
};

/// Resolve a `KernelId::Path` to a directory suitable for export
/// via [`ktstr::KTSTR_KERNEL_ENV`] plus the dirty-tree flag the
/// caller uses to decorate the kernel label.
///
/// Routes Path specs through [`cli::resolve_kernel_dir_to_entry`]
/// so they share the same cache pipeline as Version / CacheKey /
/// Git specs:
///   - Clean source tree, cache miss → build, store at
///     `local-{hash7}-{arch}-kc{suffix}`, return cache entry dir
///     with `is_dirty=false`.
///   - Clean source tree, cache hit → skip build, emit a stderr
///     line referencing the user's raw input path, the resolved
///     cache key, and the build age, then return cache entry dir
///     with `is_dirty=false`.
///   - Dirty source tree → build in source, skip cache store,
///     return canonical source dir with `is_dirty=true`. The
///     caller appends `_dirty` to the kernel label so the test
///     report distinguishes the non-reproducible run from a
///     subsequent clean rebuild of the same tree.
///
/// Both directory shapes are valid inputs to
/// [`crate::kernel_path::find_image_in_dir`]'s child consumers;
/// the cache-entry layout (`<dir>/<image_name>`) and source-tree
/// layout (`<dir>/arch/<arch>/boot/<image_name>`) are both probed.
///
/// `raw_input` is the verbatim user-supplied `--kernel` argument
/// before canonicalization — used in the cache-hit stderr line so
/// the operator sees the path they actually typed (e.g.
/// `../linux`) rather than the resolved canonical form, and in
/// the resolve-failure error so a typo names whatever the user
/// supplied.
///
/// On canonicalize / source-tree-validation failure, the inner
/// error is re-wrapped with the user's raw input + the standard
/// `KTSTR_KERNEL_HINT` so the diagnostic shape matches the
/// behaviour the previous inline `canonicalize` call provided.
/// The single canonicalize then lives inside
/// [`crate::fetch::local_source`] (called via
/// [`cli::resolve_kernel_dir_to_entry`]); doing it twice in this
/// function and again in `local_source` produced redundant
/// syscalls without changing the resulting path.
pub(crate) fn resolve_path_kernel(p: &Path, raw_input: &str) -> Result<(PathBuf, bool), String> {
    // Boundary bridge: `cli::resolve_kernel_dir_to_entry` returns
    // `anyhow::Result<KernelDirOutcome>` while this function
    // returns `Result<_, String>`, so we stringify at the call
    // site. A broader anyhow migration across cargo-ktstr.rs is
    // pending and would drop this last bridge.
    let outcome = cli::resolve_kernel_dir_to_entry(p, "cargo ktstr", None).map_err(|e| {
        format!(
            "--kernel {raw_input}: {e:#}. {hint}",
            hint = ktstr::KTSTR_KERNEL_HINT,
        )
    })?;
    if let Some(hit) = outcome.cache_hit {
        eprintln!(
            "cargo ktstr: cache hit for {raw_input} ({key}{age})",
            key = hit.cache_key,
            age = format_built_age(&hit.built_at),
        );
    }
    Ok((outcome.dir, outcome.is_dirty))
}

/// Resolve `--kernel <SPEC>` (or absence) to a bootable image path
/// for the `shell` and `verifier` subcommands.
///
/// `KERNEL_POLICY` is declared as a function-local const because
/// this is the sole consumer — the policy describes cargo-ktstr's
/// host-side conventions for [`cli::resolve_kernel_image`]:
///   - `accept_raw_image: true` allows `--kernel /path/to/bzImage`
///     to short-circuit the source-tree / cache-key path resolution
///     pipeline (the test harness routes through
///     [`resolve_kernel_set`] which only accepts directory inputs by
///     construction, so per-test labels are deterministic).
///   - `cli_label: "cargo ktstr"` is the user-facing prefix that
///     [`cli::resolve_kernel_image`] embeds in its diagnostic
///     messages so failures cite the binary the operator invoked.
///
/// On `Some(spec)` the call delegates to
/// [`cli::resolve_kernel_image`] which dispatches on the parsed
/// [`ktstr::kernel_path::KernelId`] variant (Path / Version /
/// CacheKey / Git). On `None` the same helper falls back to
/// [`ktstr::find_kernel`] (cache-then-filesystem auto-discovery)
/// followed by a kernel.org download if nothing is found.
///
/// Errors stringify the underlying anyhow chain via `{e:#}` so the
/// shell / verifier dispatchers stay on the `Result<_, String>`
/// surface this binary uses end-to-end.
pub(crate) fn resolve_kernel_image(kernel: Option<&str>) -> Result<PathBuf, String> {
    /// Policy for cargo-ktstr's shell + verifier kernel resolution:
    /// accept raw image files, use "cargo ktstr" as the CLI label.
    const KERNEL_POLICY: cli::KernelResolvePolicy<'static> = cli::KernelResolvePolicy {
        accept_raw_image: true,
        cli_label: "cargo ktstr",
    };
    cli::resolve_kernel_image(kernel, &KERNEL_POLICY).map_err(|e| format!("{e:#}"))
}

/// Format a cache entry's `built_at` ISO-8601 timestamp as a
/// human-readable age suffix for the cache-hit log line.
///
/// Returns `, built {age} ago` (with the leading comma+space) on
/// successful parse + elapsed-since-now computation, so the call
/// site can splice it directly into the parenthesised message:
/// `(local-..., built 2h 15m ago)`. Returns the empty string when
/// either the timestamp can't be parsed (malformed metadata) or
/// the build moment is in the future relative to local clock
/// (clock skew on a shared cache); callers see `(local-...)` with
/// no age suffix in those degenerate cases.
pub(crate) fn format_built_age(built_at: &str) -> String {
    let Ok(parsed) = humantime::parse_rfc3339(built_at) else {
        return String::new();
    };
    let Ok(elapsed) = std::time::SystemTime::now().duration_since(parsed) else {
        return String::new();
    };
    // Truncate to whole-second granularity. `format_duration` on
    // sub-second remainders renders nanos that aren't useful
    // ("2h 15m 32s 184ms 7us 12ns") and clutter the cache-hit
    // line.
    let elapsed = std::time::Duration::from_secs(elapsed.as_secs());
    format!(", built {} ago", humantime::format_duration(elapsed))
}

/// Canonicalize a cache-entry directory before exporting it via
/// [`ktstr::KTSTR_KERNEL_ENV`] / [`ktstr::KTSTR_KERNEL_LIST_ENV`].
/// `CacheDir` roots at the XDG cache home (or `KTSTR_CACHE_DIR`),
/// both typically absolute — but an operator-supplied
/// `KTSTR_CACHE_DIR=./cache` would produce a relative path here and
/// reach the same cwd-divergence bug the `Path` branch defends
/// against. `canonicalize` resolves that from the parent's cwd; a
/// failure means the cache dir was removed between lookup and
/// export (rare race), in which case we fall back to the original
/// path rather than bailing — the child will re-enter its own cache
/// lookup and surface the real missing-entry error.
pub(crate) fn canonicalize_cache_dir(cache_dir: PathBuf) -> PathBuf {
    std::fs::canonicalize(&cache_dir).unwrap_or(cache_dir)
}

/// Resolve one already-validated [`KernelId`] (NOT `Range` — the
/// caller fans Range out to per-version `Version` ids before
/// calling here) to a `(label, dir)` tuple.
///
/// Extracted from `resolve_kernel_set`'s rayon body so the per-
/// spec match arm is one function call rather than five inline
/// arms duplicated across the parallel and sequential paths.
/// Each non-Range arm here mirrors what the original sequential
/// loop did.
///
/// Range fan-out lives on the caller because the
/// `expand_kernel_range` step yields a `Vec<String>` that has to
/// be expanded into the same parallel pool — `flat_map_iter` is
/// the wrong shape for "fan out N items into the parent
/// iterator." See the parallel comment block in
/// [`resolve_kernel_set`] for the full strategy.
pub(crate) fn resolve_one(id: ktstr::kernel_path::KernelId) -> Result<(String, PathBuf), String> {
    use ktstr::kernel_path::KernelId;
    match id {
        KernelId::Path(p) => {
            // Capture the user's raw input string before any
            // canonicalization so cache-hit diagnostics inside
            // `resolve_path_kernel` can name the path they
            // actually typed (`../linux`) instead of the
            // resolved canonical form.
            let raw_input = p.display().to_string();
            // Compute the BASE label from the CANONICAL SOURCE TREE
            // path, NOT the directory `resolve_path_kernel`
            // returns. The returned dir may be a cache entry
            // (`<cache>/local-{hash7}-{arch}-kc{suffix}`); a
            // basename-derived label off that would render as
            // `path_local-{hash7}-{arch}-kc{suffix}_{hash6}` and
            // change between cache-miss runs (when
            // `path_kernel_label` would have observed the source
            // tree dir) and cache-hit runs (when it would have
            // observed the cache entry dir). Pinning the label
            // to the canonical SOURCE path keeps the operator-
            // facing identifier stable across cache states for
            // the same `--kernel /path/to/linux` invocation.
            //
            // The dirty-tree flag from `resolve_path_kernel`
            // appends a `_dirty` suffix when `local_source`
            // observed uncommitted modifications. The dirty-tree
            // build skips the cache store, so a `path_linux_a3b1c2`
            // row in the test report is not interchangeable with
            // a `path_linux_a3b1c2_dirty` row — the former is
            // reproducible from the recorded git hash, the latter
            // is not. Surfacing the divergence in the label keeps
            // the gauntlet output honest about which runs the
            // operator can re-run from cache.
            let canon_input = std::fs::canonicalize(&p).map_err(|e| {
                format!(
                    "--kernel {}: path does not exist or cannot be \
                     canonicalized ({e:#}). {hint}",
                    p.display(),
                    hint = ktstr::KTSTR_KERNEL_HINT,
                )
            })?;
            let base_label = path_kernel_label(&canon_input);
            let (dir, is_dirty) = resolve_path_kernel(&p, &raw_input)?;
            let label = decorate_path_label_for_dirty(&base_label, is_dirty);
            Ok((label, dir))
        }
        KernelId::Version(ref ver) => {
            let cache_dir = ktstr::cli::resolve_cached_kernel(&id, "cargo ktstr")
                .map_err(|e| format!("{e:#}"))?;
            let dir = canonicalize_cache_dir(cache_dir);
            Ok((ver.clone(), dir))
        }
        KernelId::CacheKey(ref key) => {
            let cache_dir = ktstr::cli::resolve_cached_kernel(&id, "cargo ktstr")
                .map_err(|e| format!("{e:#}"))?;
            let dir = canonicalize_cache_dir(cache_dir);
            // Extract a discriminating label from the cache key —
            // tarball keys yield the version prefix
            // (`6.14.2-tarball-…` → `6.14.2`), git keys yield the
            // ref (`for-next-git-…` → `for-next`), local keys yield
            // `local_{hash6}` (or `local_unknown` for non-git
            // trees). See [`cache_key_to_version_label`] for the
            // full per-shape contract and fallback behavior.
            let label = cache_key_to_version_label(key).to_string();
            Ok((label, dir))
        }
        KernelId::Git {
            ref url,
            ref git_ref,
        } => {
            let cache_dir = ktstr::cli::resolve_git_kernel(url, git_ref, "cargo ktstr")
                .map_err(|e| format!("resolve git+{url}#{git_ref}: {e:#}"))?;
            let dir = canonicalize_cache_dir(cache_dir);
            let label = git_kernel_label(url, git_ref);
            Ok((label, dir))
        }
        KernelId::Range { start, end } => {
            // Defensive: the caller fans Range out to per-version
            // Version ids before calling here. This arm exists
            // only so the compiler accepts the exhaustive match;
            // hitting it indicates a programming error in the
            // caller's flat-map shape rather than a user-visible
            // condition, so the diagnostic is descriptive enough
            // to point a developer at the wrong call site.
            Err(format!(
                "internal: resolve_one called with Range {start}..{end}; \
                 caller must expand Range via `expand_kernel_range` and \
                 call `resolve_one` per version"
            ))
        }
    }
}

/// Resolve every `--kernel` spec to a flat list of `(kernel_label,
/// kernel_dir)` pairs. Each Range expands to one entry per release
/// in the interval; each Path / Version / CacheKey / Git produces
/// exactly one entry.
///
/// The flat list is what `cargo ktstr test` (and `coverage` /
/// `llvm-cov`) hand to the test binary as the kernel dimension of
/// the gauntlet expansion: every (test × scenario × topology ×
/// flags × kernel) tuple becomes a distinct nextest test case so
/// nextest's parallelism, retries, and `-E` filtering apply
/// natively. A single `cargo nextest run` (or `cargo llvm-cov
/// nextest`) invocation services every variant; profraw lands per-
/// child so cargo-llvm-cov merges all of them automatically.
///
/// Build / download / clone failures abort the resolution before
/// any test runs — there's no useful state to continue from
/// (a missing kernel can't be tested, and continuing would mask
/// which kernel was requested-but-unavailable in the operator-
/// visible error stream).
///
/// `kernel_label` for each entry is a semantic, operator-readable
/// identity:
/// - Path → `path_{basename}_{hash6}` (basename + 6-char hash of the
///   canonical path so two distinct directories with the same name
///   don't collide).
/// - Version / Range expansion → the version string verbatim
///   (e.g. `6.14.2`, `6.15-rc3`).
/// - CacheKey → the version prefix (everything before the first
///   `-tarball-` / `-git-` / `-local-` component).
/// - Git → `git_{owner}_{repo}_{ref}` extracted from the URL +
///   git ref.
///
/// The downstream `sanitize_kernel_label` in
/// [`crate::test_support::dispatch`] applies the `kernel_` prefix
/// and `[a-z0-9_]+` normalisation; this label is the human-meaningful
/// payload it operates on.
pub(crate) fn resolve_kernel_set(specs: &[String]) -> Result<Vec<(String, PathBuf)>, String> {
    use ktstr::kernel_path::KernelId;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    preflight_collision_check(specs)?;

    // Each spec resolves independently:
    //   - Path → cache lookup → maybe build (no network).
    //     Clean source trees hit the local-source cache key
    //     `local-{hash7}-{arch}-kc{suffix}`; cache miss reaches
    //     the same `kernel_build_pipeline` Version/CacheKey/Git
    //     specs use, with the result stored back at the same key.
    //     Dirty trees skip the cache store and build in place.
    //   - Version / CacheKey → cache lookup → maybe download +
    //     build.
    //   - Range → fetch releases.json once, then per-version
    //     cache lookup → maybe download + build for each
    //     expanded version.
    //   - Git → shallow clone → cache lookup → maybe build.
    //
    // Two phases of work happen behind the per-spec resolvers:
    // (1) network I/O — kernel.org tarball download or
    //     `git_clone` shallow fetch — which is independent
    //     across specs and overlaps freely.
    // (2) build — `make -j$(nproc)` invoked under an LLC flock
    //     plus a cgroup v2 sandbox (`acquire_build_reservation`
    //     in `kernel_build_pipeline`). The flock serializes
    //     concurrent builders against each other, so parallel
    //     resolvers queue at the LLC level even when their
    //     downloads overlapped.
    //
    // Net effect: parallelizing `resolve_kernel_set` overlaps
    // every download / clone phase, while the build phase
    // remains serialized via the LLC flock the build pipeline
    // already holds. `make -j$(nproc)` inside a single build
    // saturates CPU on its own — running multiple builds
    // concurrently would only contend with the active build's
    // reserved LLCs, so the flock-driven serialization is the
    // correct ceiling. The cache-store path is also flock-
    // protected (`store_succeeds_under_internal_exclusive_lock`
    // in `cache.rs`) so concurrent stores against different
    // cache keys are safe.
    //
    // Concurrent resolves of the SAME spec (e.g. a duplicated
    // `--kernel 6.14.2` flag) racing on the same cache key are
    // also safe — the cache's exclusive store lock means the
    // second resolver re-checks the cache after acquiring its
    // own lock and finds the just-written entry, skipping the
    // redundant build.
    //
    // `flat_map_iter` flattens Range expansion under one rayon
    // worker: the closure resolves every version of a single
    // Range spec sequentially via `.map(...).collect::<Vec<_>>()`
    // before yielding the iterator, so a 5-version range
    // serializes its five resolves against itself within one
    // worker. Peer specs (other top-level `--kernel` arguments)
    // still parallelize across workers — only versions WITHIN
    // one Range are serial. The serialization is acceptable
    // because the per-version build phase is already serialized
    // at the LLC-flock layer (see comment above), so the lost
    // intra-range download overlap is a small fraction of total
    // wall time on multi-version Range invocations.
    //
    // Result-collecting fail-fast: rayon's `collect` on
    // `Result<_, _>` short-circuits on the first error, so a
    // single failed spec aborts the rest. This matches the
    // pre-parallel loop's `?` propagation; the operator sees
    // the first failure even though peers may still be in
    // flight (their cleanup is owned by their tempdirs going
    // out of scope, see `download_and_cache_version` /
    // `resolve_git_kernel` for the `tempfile::TempDir`-driven
    // teardown).
    // Cap rayon parallelism via a bounded ThreadPool installed
    // ONLY for this resolve pipeline. Without the cap, an
    // operator passing `--kernel A --kernel B ... --kernel Z`
    // (10+ specs) would saturate the global rayon pool with
    // simultaneous git_clone + tarball downloads. Each download
    // / clone is network-bound and largely cooperative on local
    // CPU, but the spawn cost (rayon worker steal-and-park,
    // tempdir creation, gix or reqwest init) compounds in
    // proportion to spec count, and a contended local network
    // (the kernel.org CDN's per-IP throttle, a developer's home
    // ISP, a CI runner's shared NIC) degrades when too many
    // streams overlap.
    //
    // The cap defaults to `available_parallelism()` — the host's
    // logical CPU count, std-lib provided so no extra dependency
    // is pulled in. Saturating local parallelism is the right
    // ceiling: download streams shouldn't outnumber the threads
    // the host can drive without thrashing, and the build phase
    // is already serialized at the LLC-flock layer (see comment
    // above) so additional download fan-out wouldn't accelerate
    // builds anyway.
    //
    // Operators can override the cap via the
    // `KTSTR_KERNEL_PARALLELISM` env var (see
    // [`ktstr::KTSTR_KERNEL_PARALLELISM_ENV`]) — useful when the
    // default is wrong for the host: a fast NIC + slow CPU
    // benefits from more in-flight downloads; a contended CI
    // runner with concurrent jobs benefits from a lower cap to
    // leave bandwidth for siblings. Parsing rules and fallback
    // behavior live in [`ktstr::cli::resolve_kernel_parallelism`]
    // so a typoed export (`=abc`, `=0`) silently degrades to the
    // host-CPU default rather than disabling parallelism.
    //
    // Bounded ThreadPool via `pool.install(|| ...)` scopes the
    // cap to this pipeline only — the global rayon pool is
    // unaffected, so any other rayon-using code in the same
    // process (test parallelism in nextest's harness, polars'
    // groupby, etc.) keeps its own width. Falls back to the
    // global pool if `ThreadPoolBuilder::build` fails (e.g. on
    // a host that's already maxed its thread limits) — better
    // to run the resolve under the default global pool than
    // bail with a cap-construction error that has nothing to
    // do with the user's `--kernel` input.
    let max_threads = ktstr::cli::resolve_kernel_parallelism();
    let bounded_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build()
        .ok();

    // Per-resolve progress feedback. A user passing `--kernel
    // 6.10..6.20` (10+ versions) sees `cargo ktstr: resolved
    // kernel "6.10"` lines as each version finishes its
    // download+build cycle, instead of staring at silence for
    // the multi-minute resolve. Emitted at the Ok-arm of each
    // `resolve_one` call so failures still propagate via the
    // existing fail-fast `collect::<Result<_, _>>?` chain
    // upstream — only successful resolves print. Single-kernel
    // runs emit ONE line; that's negligible noise versus the
    // multi-kernel UX gain. Output is `eprintln!` (stderr) so
    // it doesn't pollute stdout pipelines that consume the
    // tool's other output (e.g. shell scripts piping through
    // jq).
    //
    // `tracing::info!` would respect `RUST_LOG`, but the
    // command spends most of its wall time in
    // `resolve_kernel_set` and operators expect progress
    // visibility by default — gating it behind a verbosity
    // flag would defeat the point. Keep it as unconditional
    // `eprintln!` matching the pattern other long-running
    // helpers (`expand_kernel_range`, `kernel_build_pipeline`)
    // already use.
    let resolve_one_with_progress = |id: KernelId| -> Result<(String, PathBuf), String> {
        let result = resolve_one(id);
        if let Ok((label, _)) = &result {
            eprintln!("cargo ktstr: resolved kernel {label:?}");
        }
        result
    };

    let resolve_in_pool = || -> Result<Vec<(String, PathBuf)>, String> {
        specs
            .into_par_iter()
            .filter_map(|raw| {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .flat_map_iter(|trimmed| {
                // `flat_map_iter` returns an iterator per input. The
                // Range arm below pre-collects every version's
                // `resolve_one` result into a Vec before yielding,
                // so versions WITHIN a single Range spec resolve
                // sequentially under one rayon worker; only PEER
                // specs (other top-level `--kernel` args) parallelize
                // across workers. Each yielded item is an opaque
                // `Result<(String, PathBuf), String>` driven by the
                // shared `resolve_one` helper; rayon's `collect` on
                // `Result` short-circuits on the first error.
                //
                // Validation runs first so an inverted Range bails
                // before any I/O — same diagnostic timing the
                // sequential loop preserved.
                let id = KernelId::parse(&trimmed);
                if let Err(e) = id.validate() {
                    return vec![Err(format!("--kernel {id}: {e}"))].into_iter();
                }
                match id {
                    KernelId::Range { start, end } => {
                        match ktstr::cli::expand_kernel_range(&start, &end, "cargo ktstr") {
                            Ok(versions) => versions
                                .into_iter()
                                .map(|ver| {
                                    resolve_one_with_progress(KernelId::Version(ver.clone()))
                                        .map_err(|e| format!("resolve kernel {ver}: {e}"))
                                })
                                .collect::<Vec<_>>()
                                .into_iter(),
                            Err(e) => vec![Err(format!("{e:#}"))].into_iter(),
                        }
                    }
                    other => vec![resolve_one_with_progress(other)].into_iter(),
                }
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let resolved: Vec<(String, PathBuf)> = match bounded_pool {
        Some(pool) => pool.install(resolve_in_pool)?,
        None => resolve_in_pool()?,
    };

    let resolved = dedupe_resolved(resolved);

    detect_label_collisions(&resolved)?;
    Ok(resolved)
}

/// Acquire source, configure, build, and cache a kernel image.
///
/// `version` accepts `MAJOR.MINOR[.PATCH][-rcN]` for a single tarball,
/// `MAJOR.MINOR` (a major.minor prefix that resolves to the latest
/// patch in that series), or `START..END` for a range that expands
/// against kernel.org's `releases.json` to every `stable` /
/// `longterm` release inside the inclusive interval. A range is
/// detected via [`KernelId::parse`] and dispatched here to
/// [`kernel_build_one`] per resolved version, sharing the
/// download / cache-lookup / build pipeline that single-version
/// invocations use. Range mode collects per-version errors as a
/// best-effort summary: a build failure on one version is reported
/// and the iteration continues to the next, so a stale endpoint
/// doesn't block the rest of the range from caching.
///
/// `--git` and `--source` paths bypass range expansion (range
/// applies to tarball downloads only) and forward unchanged to
/// [`kernel_build_one`].
pub(crate) fn kernel_build(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
    extra_kconfig: Option<PathBuf>,
) -> Result<(), String> {
    // Read the extra-kconfig fragment ONCE up front so a range
    // expansion doesn't re-read the same file per version (and so a
    // bad path surfaces before any download / build work fires).
    // [`ktstr::cli::read_extra_kconfig`] does the 4-arm error
    // classification (ENOENT/EISDIR/EACCES/UTF-8) and emits an
    // empty-file warning so a 0-byte fragment doesn't silently
    // produce an "extras present but nothing merged" build.
    let extra_content: Option<String> = match extra_kconfig.as_ref() {
        Some(p) => Some(cli::read_extra_kconfig(p, "cargo ktstr")?),
        None => None,
    };

    // Range dispatch only applies to tarball mode. `--source` and
    // `--git` carry their own source-of-truth that ranges don't
    // overlap with: a path identifies one tree, a git ref names one
    // commit. A range argument alongside either is undefined input;
    // clap's existing `conflicts_with` already rejects
    // `version + source` and `version + git` combinations, so the
    // range branch only fires when neither --source nor --git is
    // present.
    if source.is_none()
        && git.is_none()
        && let Some(ref v) = version
    {
        use ktstr::kernel_path::KernelId;
        let id = KernelId::parse(v);
        // Validate before any I/O: an inverted range surfaces the
        // "swap the endpoints" diagnostic ahead of any download.
        id.validate().map_err(|e| format!("--kernel {id}: {e}"))?;
        if let KernelId::Range { start, end } = id {
            let versions = ktstr::cli::expand_kernel_range(&start, &end, "cargo ktstr")
                .map_err(|e| format!("{e:#}"))?;
            let total = versions.len();
            let mut failures: Vec<(String, String)> = Vec::new();
            for (i, ver) in versions.iter().enumerate() {
                eprintln!("cargo ktstr: [{}/{total}] kernel build {ver}", i + 1);
                if let Err(e) = kernel_build_one(
                    Some(ver.clone()),
                    None,
                    None,
                    None,
                    force,
                    clean,
                    cpu_cap,
                    extra_content.as_deref(),
                ) {
                    eprintln!("cargo ktstr: {ver}: {e}");
                    failures.push((ver.clone(), e));
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                // Surface the failure summary on the way out so an
                // automated invocation can scrape one log line per
                // failing version. Continue-on-error is the right
                // default for ranges (a stale endpoint shouldn't
                // gate the rest of the build cohort), but a
                // non-zero exit still flags the cohort as
                // partial.
                Err(format!(
                    "kernel build range {start}..{end}: {failed}/{total} \
                     version(s) failed: {names}",
                    start = start,
                    end = end,
                    failed = failures.len(),
                    names = failures
                        .iter()
                        .map(|(v, _)| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                ))
            }
        } else {
            kernel_build_one(
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_content.as_deref(),
            )
        }
    } else {
        kernel_build_one(
            version,
            source,
            git,
            git_ref,
            force,
            clean,
            cpu_cap,
            extra_content.as_deref(),
        )
    }
}

/// Single-version variant of [`kernel_build`]: handles one tarball,
/// `--source`, or `--git` invocation. Carries the `kernel_build`
/// implementation as it stood before range dispatch was wired in;
/// extracted into a helper so the range loop in `kernel_build` can
/// reuse the same download + cache + build pipeline per resolved
/// version without duplicating it.
///
/// `extra_kconfig` is the pre-loaded user fragment from
/// `--extra-kconfig PATH` (the file is read once in [`kernel_build`]
/// before fanning out to per-version invocations). `Some(content)`
/// folds into the cache key suffix via
/// [`ktstr::cache_key_suffix_with_extra`] and into the configure
/// pass via the Cow merge construction in
/// [`ktstr::cli::kernel_build_pipeline`].
#[allow(clippy::too_many_arguments)]
fn kernel_build_one(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
    extra_kconfig: Option<&str>,
) -> Result<(), String> {
    // Resolve the CLI --cpu-cap flag against KTSTR_CPU_CAP env
    // and the implicit "no cap" default. Conflict with
    // KTSTR_BYPASS_LLC_LOCKS=1 surfaces here so operators see
    // the parse-time error, not an opaque pipeline bail later.
    if cpu_cap.is_some()
        && std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_some_and(|v| !v.is_empty())
    {
        return Err(
            "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
             --cpu-cap is a resource contract; bypass disables the contract entirely."
                .to_string(),
        );
    }
    let resolved_cap = cli::CpuCap::resolve(cpu_cap).map_err(|e| format!("{e:#}"))?;

    let cache = CacheDir::new().map_err(|e| format!("open cache: {e:#}"))?;

    // Temporary directory for tarball/git source extraction.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e:#}"))?;

    // Acquire source.
    let client = fetch::shared_client();
    let mut acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path).map_err(|e| format!("{e:#}"))?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path(), "cargo ktstr")
            .map_err(|e| format!("{e:#}"))?
    } else {
        // Tarball download: explicit version, prefix, or latest stable.
        let ver = match version {
            Some(v) if fetch::is_major_minor_prefix(&v) => {
                // Major.minor prefix (e.g., "6.12") — resolve latest patch.
                fetch::fetch_version_for_prefix(client, &v, "cargo ktstr")
                    .map_err(|e| format!("{e:#}"))?
            }
            Some(v) => v,
            None => fetch::fetch_latest_stable_version(client, "cargo ktstr")
                .map_err(|e| format!("{e:#}"))?,
        };
        // Check cache before downloading. Cache key folds in the
        // merged-kconfig hash so an `--extra-kconfig` build looks up
        // a distinct slot from a vanilla baked-in-only build —
        // `cache_key_suffix_with_extra(None)` equals
        // `cache_key_suffix()` so the no-extra path is byte-identical
        // to pre-flag behavior.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!(
            "{ver}-tarball-{arch}-kc{}",
            ktstr::cache_key_suffix_with_extra(extra_kconfig),
        );
        if !force && let Some(entry) = cache_lookup(&cache, &cache_key) {
            eprintln!("cargo ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("cargo ktstr: use --force to rebuild");
            return Ok(());
        }
        let sp = cli::Spinner::start("Downloading kernel...");
        let result = fetch::download_tarball(client, &ver, tmp_dir.path(), "cargo ktstr");
        drop(sp);
        let mut acquired = result.map_err(|e| format!("{e:#}"))?;
        // `download_tarball` builds its `cache_key` using the bare
        // `cache_key_suffix()` (see `fetch::download_tarball`).
        // Override with the merged-suffix key we looked up under so
        // the post-build cache store lands at the same slot we'd
        // hit on a re-run.
        acquired.cache_key = cache_key;
        acquired
    };

    // For `--source` and `--git` paths, `local_source` and `git_clone`
    // build `acquired.cache_key` against the bare `cache_key_suffix()`
    // — already shaped `...-kc{baked_hash}`. With `--extra-kconfig`
    // set, lift the `-xkc{extra_hash}` append to
    // [`cli::append_extra_kconfig_suffix`] so both binaries share
    // one merge path; the cache lookup + post-build store both target
    // the extras-aware slot.
    if source.is_some() || git.is_some() {
        cli::append_extra_kconfig_suffix(&mut acquired.cache_key, extra_kconfig);
    }

    // Check cache for --source and --git (tarball already checked
    // pre-download above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cache_lookup(&cache, &acquired.cache_key)
    {
        eprintln!("cargo ktstr: cached kernel found: {}", entry.path.display());
        eprintln!("cargo ktstr: use --force to rebuild");
        return Ok(());
    }

    // `--force` fail-fast pre-check: if tests are actively holding
    // the cache-entry lock, bail with the PID list rather than
    // silently waiting to stomp the in-use entry. The guard drops
    // at the end of this `if` before `kernel_build_pipeline` runs.
    if force {
        let _force_check = cache
            .try_acquire_exclusive_lock(&acquired.cache_key)
            .map_err(|e| format!("{e:#}"))?;
    }

    cli::kernel_build_pipeline(
        &acquired,
        &cache,
        "cargo ktstr",
        clean,
        source.is_some(),
        resolved_cap,
        extra_kconfig,
    )
    .map_err(|e| format!("{e:#}"))?;

    Ok(())
}

/// Look up a cache key, checking local first, then remote (if enabled).
fn cache_lookup(cache: &CacheDir, cache_key: &str) -> Option<CacheEntry> {
    cli::cache_lookup(cache, cache_key, "cargo ktstr")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // format_built_age — cache-hit log line age suffix
    // ---------------------------------------------------------------
    //
    // The helper renders the persisted `built_at` ISO-8601 stamp as
    // `, built {age} ago`. It must produce the empty string on
    // unparseable inputs (so the cache-hit line still renders
    // gracefully without a malformed suffix), and must include a
    // leading comma+space prefix on the success path so the call
    // site can splice it directly into `(cache_key{age})`.

    #[test]
    fn format_built_age_unparseable_returns_empty_string() {
        // Malformed timestamp must not panic and must not yield a
        // half-formed suffix. The cache-hit log line stays valid
        // even when metadata is corrupt: `(cache_key)` with no
        // age portion.
        assert_eq!(format_built_age("not-a-timestamp"), "");
        assert_eq!(format_built_age(""), "");
        // Almost-valid RFC 3339 (missing trailing Z) must also
        // collapse to empty rather than returning a partial.
        assert_eq!(format_built_age("2026-01-02T03:04:05"), "");
    }

    #[test]
    fn format_built_age_future_timestamp_returns_empty_string() {
        // A timestamp far in the future fails
        // `duration_since` because the build moment hasn't
        // occurred yet relative to local clock — clock skew on a
        // shared cache between two hosts can produce this. The
        // helper collapses to empty rather than rendering
        // `built -2h ago` or panicking.
        assert_eq!(format_built_age("9999-12-31T23:59:59Z"), "");
    }

    #[test]
    fn format_built_age_past_timestamp_includes_leading_comma_and_seconds() {
        // A reachable past timestamp must produce the
        // `, built ... ago` shape. We don't pin the exact age
        // string (it depends on `SystemTime::now()` at test time),
        // but assert the structural invariants:
        //   * non-empty
        //   * starts with `, built ` (the leading comma+space lets
        //     the caller splice into `(cache_key{age})` without a
        //     conditional separator)
        //   * ends with ` ago` (the trailing keyword renders the
        //     duration as relative-past in human language)
        let one_hour_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .saturating_sub(3600);
        let timestamp = humantime::format_rfc3339(
            std::time::UNIX_EPOCH + std::time::Duration::from_secs(one_hour_ago),
        )
        .to_string();
        let age = format_built_age(&timestamp);
        assert!(
            age.starts_with(", built "),
            "age suffix must start with the splice prefix `, built `, got {age:?}",
        );
        assert!(
            age.ends_with(" ago"),
            "age suffix must end with the relative-past keyword ` ago`, got {age:?}",
        );
    }

    // ---------------------------------------------------------------
    // resolve_path_kernel — Path-spec error diagnostics
    // ---------------------------------------------------------------
    //
    // The diagnostic shape `--kernel {raw}: {inner}. {KTSTR_KERNEL_HINT}`
    // is what the user sees when `--kernel /path/...` fails. The
    // raw input must appear verbatim so a typo names the exact
    // string they passed; the inner error must come from the
    // shared resolution pipeline (currently
    // `cli::resolve_kernel_dir_to_entry`); the hint must guide
    // the user toward the supported `--kernel` shapes.

    /// Nonexistent path: the source-tree validation
    /// (`Makefile + Kconfig` exist) fails inside
    /// [`cli::resolve_kernel_dir_to_entry`] (via
    /// [`cli::resolve_kernel_dir_to_entry`] →
    /// `acquire_local_source_tree`), and `resolve_path_kernel`
    /// re-wraps the error with the user's raw input + the
    /// standard hint. Pins the `--kernel {raw}: ...` prefix and
    /// the trailing hint marker so a regression that dropped
    /// either surfaces here.
    #[test]
    fn resolve_path_kernel_nonexistent_returns_actionable_error() {
        let raw = "/this/path/should/not/exist/under/test";
        let result = resolve_path_kernel(std::path::Path::new(raw), raw);
        let err = result.expect_err("nonexistent path must surface as Err");
        assert!(
            err.contains(&format!("--kernel {raw}")),
            "error must lead with `--kernel {{raw_input}}:` so a typo \
             names the exact string the user passed. got: {err}",
        );
        // The hint string carries the documented `--kernel` value
        // shapes; pin its presence rather than its prose so a
        // future hint rewrite doesn't break this test.
        assert!(
            err.contains(ktstr::KTSTR_KERNEL_HINT),
            "error must end with KTSTR_KERNEL_HINT so the user sees \
             the supported `--kernel` shapes. got: {err}",
        );
    }

    /// Empty tempdir (real directory, no Makefile or Kconfig):
    /// `acquire_local_source_tree` rejects it as "not a kernel
    /// source tree" and `resolve_path_kernel` re-wraps with the
    /// user's raw input + hint. Distinct from
    /// `resolve_path_kernel_nonexistent_returns_actionable_error`
    /// because the inner error path differs (existence-check
    /// pass, content-shape fail) — both must surface the same
    /// outer wrapping.
    #[test]
    fn resolve_path_kernel_empty_tempdir_returns_not_a_source_tree_error() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let raw = tmp.path().display().to_string();
        let result = resolve_path_kernel(tmp.path(), &raw);
        let err = result.expect_err("empty tempdir must surface as Err");
        assert!(
            err.contains(&format!("--kernel {raw}")),
            "error must lead with `--kernel {{raw_input}}:`. got: {err}",
        );
        assert!(
            err.contains("not a kernel source tree"),
            "error must include the `not a kernel source tree` phrase \
             from `acquire_local_source_tree`'s diagnostic. got: {err}",
        );
        assert!(
            err.contains(ktstr::KTSTR_KERNEL_HINT),
            "error must end with KTSTR_KERNEL_HINT. got: {err}",
        );
    }
}
