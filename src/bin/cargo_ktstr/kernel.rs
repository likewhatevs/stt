//! Kernel resolution and build helpers for the `cargo ktstr` binary.
//!
//! Houses the spec-resolution pipeline that converts `--kernel
//! <SPEC>` arguments into bootable kernel directories, the
//! collision detection that protects nextest's test-name suffix
//! map, the per-spec label generator (`path_kernel_label`,
//! `git_kernel_label`, `cache_key_to_version_label`), and the
//! `kernel build` subcommand dispatcher
//! (`kernel_build` / `kernel_build_one` / `cache_lookup`).
//!
//! The flat `(label, kernel_dir)` list this module emits is what
//! the `test` / `coverage` / `llvm-cov` dispatchers (in
//! [`super::run_cargo`]) hand to the test binary as the kernel
//! dimension of the gauntlet expansion via the
//! [`ktstr::KTSTR_KERNEL_LIST_ENV`] wire format.

use std::path::{Path, PathBuf};

use ktstr::cache::{CacheDir, CacheEntry};
use ktstr::cli;
use ktstr::fetch;

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
///     pipeline (the test harness in [`super::run_cargo`] uses
///     `false` because gauntlet expansion always operates on
///     directories so per-test labels are deterministic).
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

/// Pre-flight collision detection on cheap-to-label kernel specs
/// (Version / CacheKey / Git refs). Returns `Err(message)` when
/// two distinct producer-side labels sanitize to the same nextest
/// identifier; `Ok(())` otherwise.
///
/// Versions, CacheKeys, and Git refs all yield labels through
/// pure string manipulation (`ver.clone()`,
/// `cache_key_to_version_label(key)`, `git_kernel_label(url,
/// ref)`) — no I/O. We can compute and compare the sanitized
/// forms of those labels BEFORE the parallel resolve fires any
/// downloads, builds, or git clones. That moves the collision
/// diagnostic from a multi-minute build cost ("downloaded 6.14.2,
/// downloaded git+...#main, both rebuilt their kernel, NOW we
/// tell you they collide") to a sub-millisecond pre-flight.
///
/// Path and Range specs are intentionally EXCLUDED:
/// - Path: `path_kernel_label(dir)` requires `dir` to be
///   canonicalized first (its hash6 component is over the
///   canonical path's UTF-8 bytes). Canonicalization is real
///   filesystem I/O — admissible at resolve time but not here,
///   where the goal is "fast pre-flight". Path specs that
///   collide still surface via the post-resolve
///   `detect_label_collisions` call after their canonical labels
///   are known.
/// - Range: expanding a range to its per-version label set
///   requires a `releases.json` fetch — admissible at resolve
///   time but not pre-flight (and the resolve pipeline already
///   does it once; doing it twice is waste). Range-vs-Range or
///   Range-vs-Version collisions surface post-resolve.
///
/// Identical labels appearing twice are NOT a collision under
/// this check (the `prior != label` guard on the same-label
/// case). Two `--kernel 6.14.2` specs resolve to the same
/// `(label, path)` post-resolve, get folded by `dedupe_resolved`,
/// and reach `detect_label_collisions` as a single entry.
///
/// Inverted ranges and other malformed inputs fail validation
/// here, BEFORE the network fetch the rayon resolve would
/// otherwise run — preserves the same diagnostic timing the
/// parallel path would produce on its own.
///
/// Extracted from `resolve_kernel_set` so the pre-flight
/// algorithm is unit-testable on contrived inputs without driving
/// the rayon resolve pipeline (every `resolve_one` arm performs
/// real I/O — canonicalize+build for Path, cache lookup+download
/// for Version/CacheKey, shallow git clone for Git).
pub(crate) fn preflight_collision_check(specs: &[String]) -> Result<(), String> {
    use ktstr::kernel_path::KernelId;
    let mut preflight: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for raw in specs {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let id = KernelId::parse(trimmed);
        if let Err(e) = id.validate() {
            return Err(format!("--kernel {id}: {e}"));
        }
        let label: Option<String> = match &id {
            KernelId::Version(v) => Some(v.clone()),
            KernelId::CacheKey(k) => Some(cache_key_to_version_label(k).to_string()),
            KernelId::Git { url, git_ref } => Some(git_kernel_label(url, git_ref)),
            // Path / Range deferred to post-resolve check.
            KernelId::Path(_) | KernelId::Range { .. } => None,
        };
        if let Some(label) = label {
            let sanitized = ktstr::test_support::sanitize_kernel_label(&label);
            if let Some(prior) = preflight.insert(sanitized.clone(), label.clone())
                && prior != label
            {
                return Err(format!(
                    "--kernel: pre-flight check found collision before any \
                     download or build started — labels {prior:?} and {label:?} \
                     both sanitize to {sanitized:?}, which the nextest \
                     test-name suffix cannot disambiguate. Spell each \
                     --kernel value distinctly so its sanitized form is \
                     unique. (Path and Range specs are checked post-resolve.)"
                ));
            }
        }
    }
    Ok(())
}

/// Dedupe identical `(label, path)` tuples before
/// `detect_label_collisions` fires.
///
/// Two `--kernel 6.14.2` specs (or a Range that overlaps a
/// separate Version spec) resolve to the same `(label, path)`
/// pair by construction — `resolve_one` is deterministic per
/// spec, so identical inputs produce identical outputs. Letting
/// the duplicate flow into `detect_label_collisions` would trip
/// its same-label diagnostic on a fundamentally benign input.
/// Tuple-level dedup keeps the intent ("dedupe identical
/// specs") narrow: two specs that produce the SAME label but
/// DIFFERENT paths represent a real cache-key collision that
/// `detect_label_collisions` must still catch — those rows
/// survive dedup because their tuples differ on the path.
///
/// Order-preserving dedup via a sequential first-seen pass: the
/// rayon pipeline upstream may have shuffled the input order, so
/// we honor whatever order arrived (the downstream wire format
/// is `;`-separated and order-insensitive at the dispatch layer;
/// preserving order keeps stderr diagnostics operator-readable).
/// HashSet membership check + Vec push is O(n) — acceptable on
/// the ~10s-of-kernels scale this function targets.
///
/// Extracted from `resolve_kernel_set` so the dedupe algorithm
/// is unit-testable on contrived inputs without driving the
/// rayon resolve pipeline.
pub(crate) fn dedupe_resolved(resolved: Vec<(String, PathBuf)>) -> Vec<(String, PathBuf)> {
    let mut seen: std::collections::HashSet<(String, PathBuf)> =
        std::collections::HashSet::with_capacity(resolved.len());
    let mut deduped: Vec<(String, PathBuf)> = Vec::with_capacity(resolved.len());
    for entry in resolved {
        if seen.insert(entry.clone()) {
            deduped.push(entry);
        }
    }
    deduped
}

/// Detect two distinct producer-side labels that normalize to the
/// same nextest identifier via [`ktstr::test_support::sanitize_kernel_label`].
/// A collision would shatter two cache directories under one test-
/// name suffix, so the dispatch-side label-to-dir map in
/// `parse_kernel_list` would silently retain only the last entry
/// and every prior collision would route to the wrong kernel.
///
/// On collision: returns `Err(message)` naming both labels and the
/// shared sanitized form so the operator can disambiguate the
/// inputs (e.g. spell `6.14.2` and `git+...#6.14.2` distinctly
/// rather than relying on suffix-encoded identity).
///
/// Identical (label, path) tuples are deduped UPSTREAM in
/// `resolve_kernel_set` before this helper runs, so two identical
/// `--kernel 6.14.2` specs resolving to the same (label, path)
/// pair never reach this check. What CAN reach this check is two
/// distinct producer-side labels that sanitize to the same nextest
/// suffix — that IS a real collision (different kernel content,
/// same routing identity), and surfaces here. Same-label-different-
/// path inputs (e.g. a hypothetical future producer that emits a
/// label with cache-collision shape) also reach here because the
/// upstream tuple-level dedup leaves them distinct, and
/// `seen.insert` then finds the prior label and surfaces the
/// `labels "X" and "X"` diagnostic. This helper is the last line
/// of defense against the silent-routing class of bug.
///
/// Extracted from `resolve_kernel_set` so the collision-detection
/// algorithm is unit-testable on contrived inputs without driving
/// the rayon resolve pipeline (every `resolve_one` arm performs
/// real I/O — canonicalize+build for Path, cache lookup+download
/// for Version/CacheKey, shallow git clone for Git).
pub(crate) fn detect_label_collisions(resolved: &[(String, PathBuf)]) -> Result<(), String> {
    let mut seen: std::collections::HashMap<String, &str> =
        std::collections::HashMap::with_capacity(resolved.len());
    for (label, _) in resolved {
        let sanitized = ktstr::test_support::sanitize_kernel_label(label);
        if let Some(prior) = seen.insert(sanitized.clone(), label.as_str()) {
            return Err(format!(
                "--kernel: labels {prior:?} and {label:?} both sanitize to {sanitized:?} — \
                 the nextest test-name suffix cannot disambiguate them. \
                 Spell each --kernel value distinctly so its sanitized form is unique."
            ));
        }
    }
    Ok(())
}

/// Build the `path_{basename}_{hash6}` label for a `Path`-resolved
/// kernel. The basename keeps the label operator-readable; the 6-char
/// hex hash of the canonical path's UTF-8 bytes disambiguates two
/// `linux` directories under different parents. `crc32fast` is
/// already a workspace dep (see `cli::kernel_build_pipeline` for the
/// existing consumer), so re-using it costs nothing extra.
pub(crate) fn path_kernel_label(dir: &Path) -> String {
    let basename = dir.file_name().and_then(|n| n.to_str()).unwrap_or("kernel");
    let hash = crc32fast::hash(dir.display().to_string().as_bytes());
    // `{:08x}` would emit 8 hex digits; ruling specifies a 6-char
    // hash prefix. Truncating to the leading 6 is sufficient
    // disambiguation for the operator's purpose (collision risk is
    // only a UI nuisance, not a correctness issue — the kernel_dir
    // path itself is the actual identity).
    format!("path_{basename}_{:06x}", hash & 0x00ff_ffff)
}

/// Append a `_dirty` suffix to a Path-spec kernel label when the
/// build skipped the cache store because the source tree carried
/// uncommitted modifications. Returns the label unchanged when the
/// tree was clean.
///
/// Suffix format: literal `"_dirty"` (underscore + lowercase
/// `dirty`), appended directly with no separator beyond the
/// underscore. The underscore is load-bearing — it matches the
/// existing token boundary convention used by every other label
/// emitter in this module (`path_{basename}_{hash6}`,
/// `local_{hash6}`, `kernel_{label}` in nextest output) so the
/// downstream sanitizer treats `_dirty` as one more token rather
/// than introducing a punctuation class change. The character is
/// stable across the codebase: `sanitize_kernel_label` keeps
/// alphanumerics and underscores verbatim, so the suffix does not
/// require escaping at any consumer site.
///
/// Test reports key on the (sanitized) kernel label as the
/// per-kernel column header; without the suffix, a dirty-tree run
/// and a clean-tree run on the same path render identically and
/// the operator cannot tell which row came from a non-reproducible
/// build. With the suffix:
///   - clean tree: `path_linux_a3b1c2`
///   - dirty tree: `path_linux_a3b1c2_dirty`
///
/// Downstream [`ktstr::test_support::sanitize_kernel_label`]
/// preserves alphanumerics and converts `-` / `.` to `_`, so the
/// `_dirty` suffix passes through verbatim and surfaces in the
/// nextest test-name suffix as `kernel_path_linux_a3b1c2_dirty`.
pub(crate) fn decorate_path_label_for_dirty(base_label: &str, is_dirty: bool) -> String {
    if is_dirty {
        format!("{base_label}_dirty")
    } else {
        base_label.to_string()
    }
}

/// Extract a discriminating label from a cache-entry key.
///
/// Cache keys follow three shapes:
/// - tarball: `{version}-tarball-{arch}-kc{hash}` — version is a
///   PROPER PREFIX, e.g. `6.14.2-tarball-x86_64-kcabc` → `6.14.2`.
/// - git: `{ref}-git-{short_hash}-{arch}-kc{hash}` — ref is a
///   PROPER PREFIX, e.g. `for-next-git-deadbee-x86_64-kcabc` →
///   `for-next`.
/// - local: `local-{discriminator}-{arch}-kc{hash}` — the `local-`
///   PREFIX is the source tag, with `{discriminator}` being the
///   git short_hash of the source tree (or the literal `unknown`
///   when the tree is not a git repo, see
///   `crate::fetch::local_source`). Label is `local_{hash6}`,
///   where `{hash6}` is the 6-char prefix of the discriminator —
///   collapsing every local entry to bare `"local"` would erase
///   distinct local trees from the operator-visible label and
///   cause two different `--kernel /path/A` and `--kernel /path/B`
///   builds to render identically in `kernel list` /
///   `--a-kernel` / `--b-kernel` outputs. The hash6 disambiguates
///   without leaking the full short_hash (which is meaningful at
///   the git layer but redundant in the operator-facing label).
///   For `local-unknown-...` (non-git tree), the label is
///   `local_unknown` — a single shared bucket is the correct
///   render because non-git trees lack a discriminator entirely.
///
/// Returns `Cow<str>` because the local arm builds an owned label
/// (`local_{hash6}` requires a fresh allocation), while the
/// tarball/git arms return a borrow into the input.
///
/// Falls back to the full key (borrowed) if no recognised tag is
/// present — a future cache-key shape with an unknown tag still
/// produces a non-empty label rather than a panic.
pub(crate) fn cache_key_to_version_label(key: &str) -> std::borrow::Cow<'_, str> {
    use std::borrow::Cow;
    // Local prefix has no preceding version segment — the source
    // tag is the leading token. Match the prefix shape and pull
    // the discriminator (git short_hash or `unknown`) for
    // labelling.
    if key == "local" {
        return Cow::Borrowed("local");
    }
    if let Some(rest) = key.strip_prefix("local-") {
        // `rest` shape: `{discriminator}-{arch}-kc{hash}`. Take the
        // first segment as the discriminator. Empty discriminator
        // (e.g. `local--x86_64-...`, malformed) collapses to bare
        // `local` — defensive, never produced by `fetch::local_source`.
        let discriminator = rest.split('-').next().unwrap_or("");
        if discriminator.is_empty() {
            return Cow::Borrowed("local");
        }
        // Truncate to 6 chars. `unknown` (7 chars) collapses to
        // `unknow` if truncated mid-word, which is unhelpful — keep
        // the special-case literal that `fetch::local_source` emits
        // at full length so non-git trees render as
        // `local_unknown`.
        let suffix: String = if discriminator == "unknown" {
            "unknown".to_string()
        } else {
            // Truncate to 6 chars via `chars().take(6)` to avoid
            // panicking on a non-UTF-8-aligned byte slice. Today's
            // `fetch::local_source` only emits ASCII hex
            // discriminators, but a future producer that synthesizes
            // a non-ASCII discriminator (or a malformed cache key
            // hand-typed via `KTSTR_KERNEL=local-…`) would crash
            // under `&discriminator[..6]` byte-slicing if the 6th
            // byte fell mid-char. `chars().take(6)` is UTF-8 safe by
            // construction.
            discriminator.chars().take(6).collect::<String>()
        };
        return Cow::Owned(format!("local_{suffix}"));
    }
    for tag in &["-tarball-", "-git-"] {
        if let Some(prefix_end) = key.find(tag) {
            return Cow::Borrowed(&key[..prefix_end]);
        }
    }
    Cow::Borrowed(key)
}

/// Build the `git_{owner}_{repo}_{ref}` label for a `Git`-resolved
/// kernel. Extracts the `owner` and `repo` segments from the URL's
/// path component, drops the scheme/host, strips a trailing `.git`,
/// and pairs them with the operator-supplied git ref.
///
/// Examples:
/// - `git+https://github.com/tj/sched_ext#for-next` →
///   `git_tj_sched_ext_for-next`
/// - `git+https://gitlab.com/foo/bar.git#v6.14` →
///   `git_foo_bar_v6.14`
/// - URL without a recognisable owner/repo (path with only one
///   segment, e.g. a local mirror `/srv/linux.git`) → `git_<first
///   non-empty segment>_<ref>` (defensively avoids producing an
///   ambiguous `git_` prefix on its own).
pub(crate) fn git_kernel_label(url: &str, git_ref: &str) -> String {
    // Strip scheme: everything up to and including `://`.
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    // Strip user@host: split off the leading host segment by
    // dropping everything before the FIRST `/` in the post-scheme
    // remainder, leaving the path component.
    let path = after_scheme
        .split_once('/')
        .map(|(_, rest)| rest)
        .unwrap_or(after_scheme);
    // Trim leading `/`, drop trailing `.git`, then pull the last
    // two non-empty segments as `(owner, repo)`. A single-segment
    // path (e.g. local mirror) gives `(segment, "")` which we
    // collapse to `git_{segment}_{ref}`.
    let trimmed = path.trim_start_matches('/').trim_end_matches('/');
    let trimmed = trimmed.strip_suffix(".git").unwrap_or(trimmed);
    let mut segments: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    let repo = segments.pop().unwrap_or("repo");
    let owner = segments.pop().unwrap_or("");
    if owner.is_empty() {
        format!("git_{repo}_{git_ref}")
    } else {
        format!("git_{owner}_{repo}_{git_ref}")
    }
}

/// Encode a flat `(label, kernel_dir)` list into the wire format that
/// the test binary's [`ktstr::KTSTR_KERNEL_LIST_ENV`] reader parses:
/// `label1=path1;label2=path2;...`. Semicolon is the entry separator
/// (paths can contain `:` on POSIX); `=` separates the label from the
/// path. Empty input returns an empty string so the env var is
/// idempotent — an empty value means "no list, single-kernel mode."
///
/// The label is encoded verbatim — sanitization into nextest-safe
/// `[a-z0-9_]+` identifiers happens on the test-binary side via
/// `dispatch::sanitize_kernel_label`. The producer-side label is
/// already a semantic, operator-readable identifier (a version
/// string like `6.14.2`, `git_owner_repo_ref`, `path_basename_hash6`,
/// or `local`), so the env var inspected directly via `printenv
/// KTSTR_KERNEL_LIST` reads as a meaningful kernel→path map rather
/// than as raw cache-key plumbing.
pub(crate) fn encode_kernel_list(resolved: &[(String, PathBuf)]) -> Result<String, String> {
    // KTSTR_KERNEL_LIST wire format is
    // `label1=path1;label2=path2;...`. Both metacharacters MUST be
    // rejected on the label side: `;` would split the label into
    // two pseudo-entries (the parser's `split(';')` upstream of
    // `split_once('=')`); `=` would split label/path
    // pathologically (the parser's `split_once('=')` consumes the
    // FIRST `=`, so a label `a=b` paired with path `/x` would
    // emit `a=b=/x` — the parser would treat `a` as the label
    // and `b=/x` as the path). Rejecting at encode time bails
    // with an actionable error rather than silently producing a
    // malformed env var that the test-binary parser would split
    // into garbage.
    //
    // Producers feeding this helper today (the encoder family
    // around `path_kernel_label` / `git_kernel_label` /
    // `version_kernel_label`) never emit either character in
    // practice — basenames are `[a-zA-Z0-9._-]+`, version
    // strings have `[0-9.-]`, and git labels are
    // `git_{owner}_{repo}_{ref}` with hash-stripped refs. The
    // checks here guard against a future producer change OR a
    // direct caller of `encode_kernel_list` (e.g. a unit test
    // injecting synthetic input) that violates the wire-format
    // invariant.
    for (label, _) in resolved {
        if label.contains(';') {
            return Err(format!(
                "kernel label {label:?} contains a `;`; \
                 KTSTR_KERNEL_LIST uses `;` as the entry separator. \
                 The label-emission path must produce `;`-free identifiers — \
                 if a producer is emitting this label, fix the producer to \
                 sanitize/strip `;` from its output."
            ));
        }
        if label.contains('=') {
            return Err(format!(
                "kernel label {label:?} contains a `=`; \
                 KTSTR_KERNEL_LIST uses `=` to separate label from path within an entry. \
                 The label-emission path must produce `=`-free identifiers — \
                 if a producer is emitting this label, fix the producer to \
                 sanitize/strip `=` from its output."
            ));
        }
    }
    // POSIX permits `;` in paths but the wire format uses it as
    // entry separator. Bail with an actionable error rather than
    // silently producing a malformed env var that the test-binary
    // parser would split into garbage. `=` in paths is fine — the
    // parser's `split_once('=')` only consumes the first `=`,
    // which sits inside the label↔path boundary; subsequent `=`s
    // become part of the path payload verbatim.
    for (label, dir) in resolved {
        let path = dir.display().to_string();
        if path.contains(';') {
            return Err(format!(
                "kernel directory path for {label:?} contains a `;` ({path:?}); \
                 KTSTR_KERNEL_LIST uses `;` as the entry separator and cannot encode \
                 such paths. Move or symlink the kernel cache to a path without `;`."
            ));
        }
    }
    let mut out = String::new();
    for (i, (label, dir)) in resolved.iter().enumerate() {
        if i > 0 {
            out.push(';');
        }
        out.push_str(label);
        out.push('=');
        out.push_str(&dir.display().to_string());
    }
    Ok(out)
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
    // Kernel label encoding for the multi-kernel test-name suffix
    // ---------------------------------------------------------------

    #[test]
    fn cache_key_to_version_label_tarball() {
        assert_eq!(
            cache_key_to_version_label("6.14.2-tarball-x86_64-kcabc1234"),
            "6.14.2",
        );
    }

    #[test]
    fn cache_key_to_version_label_rc_tarball() {
        assert_eq!(
            cache_key_to_version_label("6.15-rc3-tarball-x86_64-kcabc"),
            "6.15-rc3",
        );
    }

    #[test]
    fn cache_key_to_version_label_git() {
        // Git keys carry the git ref as the prefix; the label
        // captures the ref, not the post-`-git-` short hash.
        assert_eq!(
            cache_key_to_version_label("for-next-git-deadbee-x86_64-kcabc"),
            "for-next",
        );
    }

    #[test]
    fn cache_key_to_version_label_local_emits_hash6_disambiguator() {
        // Local cache keys carry the source tree's git short_hash
        // as the discriminator after `local-`. The label preserves
        // the first 6 chars so two distinct local builds (different
        // source trees, different short_hashes) render with
        // distinct labels in `kernel list` / per-side filter
        // outputs. Truncating to 6 keeps the label compact while
        // still disambiguating against the typical 7-char git
        // short_hash space.
        assert_eq!(
            cache_key_to_version_label("local-deadbee-x86_64-kcabc"),
            "local_deadbe",
            "must emit `local_{{first 6 chars of discriminator}}` so \
             distinct local trees do not collide on label",
        );
    }

    #[test]
    fn cache_key_to_version_label_local_distinct_hashes_render_distinct_labels() {
        // Anti-collision pin: two local cache keys with different
        // discriminators must produce different labels. Bare
        // `"local"` for both would erase the distinction in the
        // operator UI.
        let a = cache_key_to_version_label("local-aaaaaa1-x86_64-kcabc");
        let b = cache_key_to_version_label("local-bbbbbb2-x86_64-kcabc");
        assert_ne!(
            a, b,
            "distinct local discriminators must render distinct labels"
        );
        assert_eq!(a, "local_aaaaaa");
        assert_eq!(b, "local_bbbbbb");
    }

    #[test]
    fn cache_key_to_version_label_local_unknown_renders_local_unknown() {
        // `local-unknown-...` is the literal `fetch::local_source`
        // emits when the source tree is not a git repo (no commit
        // hash to discriminate on). The label uses the full
        // `unknown` literal rather than truncating to `unknow`.
        assert_eq!(
            cache_key_to_version_label("local-unknown-x86_64-kcabc"),
            "local_unknown",
        );
    }

    #[test]
    fn cache_key_to_version_label_local_bare_yields_bare_local() {
        // Defensive: bare `local` (no trailing segments) yields
        // bare `"local"`. Not produced by `fetch::local_source`,
        // but the function must not panic on it.
        assert_eq!(cache_key_to_version_label("local"), "local");
    }

    #[test]
    fn cache_key_to_version_label_unknown_tag_falls_through() {
        // A future cache-key shape with an unrecognised source
        // tag must still produce a non-empty label rather than
        // panicking. Operator can read the raw key in the test
        // name and infer.
        assert_eq!(
            cache_key_to_version_label("6.14.2-novel-tag-kcabc"),
            "6.14.2-novel-tag-kcabc",
        );
    }

    #[test]
    fn git_kernel_label_github_https() {
        assert_eq!(
            git_kernel_label("https://github.com/tj/sched_ext", "for-next"),
            "git_tj_sched_ext_for-next",
        );
    }

    #[test]
    fn git_kernel_label_github_https_with_dot_git() {
        assert_eq!(
            git_kernel_label("https://github.com/tj/sched_ext.git", "for-next"),
            "git_tj_sched_ext_for-next",
        );
    }

    #[test]
    fn git_kernel_label_gitlab_with_ref_tag() {
        assert_eq!(
            git_kernel_label("https://gitlab.com/foo/bar.git", "v6.14"),
            "git_foo_bar_v6.14",
        );
    }

    #[test]
    fn git_kernel_label_local_mirror_two_segment_path() {
        // Two-segment path (`/srv/linux.git`) renders as
        // `git_{owner}_{repo}_{ref}` even when the "owner" is just
        // a parent directory — the helper does not heuristically
        // distinguish "meaningful" ownership from filesystem
        // hierarchy. Deterministic and unique-per-URL is good
        // enough; over-cleverness would risk silently colliding
        // labels across distinct mirrors.
        assert_eq!(
            git_kernel_label("file:///srv/linux.git", "v6.14"),
            "git_srv_linux_v6.14",
        );
    }

    #[test]
    fn git_kernel_label_truly_single_segment_path() {
        // True single-segment path (just one component after the
        // host strip) — e.g. a bare hostname-rooted URL like
        // `file://linux.git` (no `/` after the scheme). The
        // helper's host-strip splits on `://` and takes everything
        // after the first `/` post-scheme; with no `/` to split
        // on, the entire post-scheme string IS the path. After
        // `.git` strip we have one segment, owner pops empty, and
        // the helper falls back to `git_{repo}_{ref}` to avoid
        // emitting `git__{ref}`.
        assert_eq!(
            git_kernel_label("file://linux.git", "v6.14"),
            "git_linux_v6.14",
        );
    }

    #[test]
    fn git_kernel_label_ssh_style_url() {
        // `git+ssh://git@github.com/tj/sched_ext` — the helper's
        // scheme-strip splits on `://`, then the first `/` after
        // the host, yielding the same `tj/sched_ext` path
        // component as the https variant.
        assert_eq!(
            git_kernel_label("ssh://git@github.com/tj/sched_ext", "main"),
            "git_tj_sched_ext_main",
        );
    }

    #[test]
    fn path_kernel_label_includes_basename_and_hash() {
        // `path_kernel_label` builds `path_{basename}_{hash6}`.
        // We don't pin the exact hash (it's a CRC32 of the path)
        // but assert the shape: prefix + basename + 6 hex chars.
        let p = std::path::Path::new("/tmp/somewhere/linux");
        let label = path_kernel_label(p);
        assert!(
            label.starts_with("path_linux_"),
            "label must start with `path_<basename>_`, got: {label}"
        );
        let hash_part = label.strip_prefix("path_linux_").unwrap();
        assert_eq!(hash_part.len(), 6, "hash suffix must be 6 chars: {label}");
        assert!(
            hash_part.chars().all(|c| c.is_ascii_hexdigit()),
            "hash suffix must be hex: {label}"
        );
    }

    #[test]
    fn path_kernel_label_distinguishes_paths_sharing_basename() {
        // Two different parent directories with the same `linux`
        // basename must produce DIFFERENT labels (the hash
        // disambiguates them). Pins the "collision risk is only a
        // UI nuisance" claim in the doc.
        let a = std::path::Path::new("/srv/a/linux");
        let b = std::path::Path::new("/srv/b/linux");
        assert_ne!(
            path_kernel_label(a),
            path_kernel_label(b),
            "distinct path parents must produce distinct labels",
        );
    }

    /// `decorate_path_label_for_dirty` is the seam where a
    /// dirty-tree Path resolve attaches its `_dirty` suffix to
    /// the operator-readable kernel label. Clean trees pass
    /// through unchanged so the cache-stored vs in-tree label
    /// shapes remain stable for the same canonical path.
    #[test]
    fn decorate_path_label_for_dirty_clean_tree_passthrough() {
        let base = "path_linux_a3b1c2";
        assert_eq!(
            decorate_path_label_for_dirty(base, false),
            base,
            "clean trees must not append a `_dirty` suffix",
        );
    }

    /// Dirty trees must append `_dirty` so the test report shows
    /// a non-reproducible run as distinct from the same path's
    /// clean rebuild. The suffix is deliberately placed after
    /// the hash6 segment (rather than between basename and
    /// hash6) so the `path_{basename}_{hash6}` invariant
    /// `path_kernel_label` relies on still parses cleanly.
    #[test]
    fn decorate_path_label_for_dirty_dirty_tree_appends_suffix() {
        let base = "path_linux_a3b1c2";
        assert_eq!(
            decorate_path_label_for_dirty(base, true),
            "path_linux_a3b1c2_dirty",
            "dirty trees must append `_dirty` to the base label",
        );
    }

    /// The `_dirty` suffix survives `sanitize_kernel_label`
    /// transformation verbatim — `_` is alphanumeric-equivalent
    /// in the sanitizer's preservation table, so the nextest
    /// test-name suffix renders as `kernel_path_..._dirty`.
    /// Pins the producer↔consumer round-trip so a future
    /// sanitizer change that mangles `_` is caught here rather
    /// than only in operator-visible test reports.
    #[test]
    fn decorate_path_label_for_dirty_survives_sanitize() {
        let dirty_label = decorate_path_label_for_dirty("path_linux_a3b1c2", true);
        let sanitized = ktstr::test_support::sanitize_kernel_label(&dirty_label);
        assert_eq!(
            sanitized, "kernel_path_linux_a3b1c2_dirty",
            "`_dirty` must survive sanitize verbatim so the test report \
             distinguishes dirty runs from clean runs in the nextest suffix",
        );
    }

    /// Sanity pin on the clean-tree counterpart: the same base
    /// label without the dirty decoration sanitizes to a label
    /// that differs from the dirty form. The two test-report
    /// rows MUST be distinct identifiers downstream so the
    /// per-kernel column keys do not collide.
    #[test]
    fn decorate_path_label_for_dirty_clean_dirty_sanitize_to_distinct_ids() {
        let base = "path_linux_a3b1c2";
        let clean =
            ktstr::test_support::sanitize_kernel_label(&decorate_path_label_for_dirty(base, false));
        let dirty =
            ktstr::test_support::sanitize_kernel_label(&decorate_path_label_for_dirty(base, true));
        assert_ne!(
            clean, dirty,
            "clean ({clean:?}) and dirty ({dirty:?}) sanitized labels must \
             produce distinct nextest identifiers so test reports do not \
             collapse non-reproducible runs into the cache-stored row",
        );
    }

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

    // ---------------------------------------------------------------
    // encode_kernel_list — KTSTR_KERNEL_LIST wire-format encoding
    // ---------------------------------------------------------------

    #[test]
    fn encode_kernel_list_empty_input_returns_empty_string() {
        // Pin the idempotent empty case — `cargo ktstr` skips the
        // env-var export entirely on empty kernel sets, but the
        // encoder must not panic or produce garbage if it ever does
        // see an empty slice.
        let encoded = encode_kernel_list(&[]).expect("empty input must succeed");
        assert!(
            encoded.is_empty(),
            "empty resolved list must encode to empty string, got {encoded:?}",
        );
    }

    #[test]
    fn encode_kernel_list_single_entry_has_no_separator() {
        // Single-entry payload omits the `;` separator entirely:
        // the format is `label=path`, NOT `label=path;`.
        let resolved = vec![("6.14.2".to_string(), PathBuf::from("/cache/foo"))];
        let encoded = encode_kernel_list(&resolved).expect("single entry must succeed");
        assert_eq!(
            encoded, "6.14.2=/cache/foo",
            "single-entry encoding must be `label=path` with no trailing separator",
        );
    }

    #[test]
    fn encode_kernel_list_two_entries_uses_semicolon_separator() {
        // Two-entry payload uses `;` as the entry separator; `=`
        // separates the label from the path within each entry.
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("6.15.0".to_string(), PathBuf::from("/cache/b")),
        ];
        let encoded = encode_kernel_list(&resolved).expect("two entries must succeed");
        assert_eq!(
            encoded, "6.14.2=/cache/a;6.15.0=/cache/b",
            "two-entry encoding must be `label=path;label=path`",
        );
    }

    #[test]
    fn encode_kernel_list_three_entries_preserves_order() {
        // The encoder iterates `resolved` in input order and writes
        // entries in that order. A regression that sorted entries
        // (e.g. by label alphabetically) would silently reorder the
        // multi-kernel test-name suffix dimension and break
        // operator-stable test naming.
        let resolved = vec![
            ("z-late".to_string(), PathBuf::from("/cache/z")),
            ("a-early".to_string(), PathBuf::from("/cache/a")),
            ("m-mid".to_string(), PathBuf::from("/cache/m")),
        ];
        let encoded = encode_kernel_list(&resolved).expect("three entries must succeed");
        assert_eq!(
            encoded, "z-late=/cache/z;a-early=/cache/a;m-mid=/cache/m",
            "encoder must preserve input order; sorting would change test-name suffix order",
        );
    }

    #[test]
    fn encode_kernel_list_rejects_semicolon_in_path() {
        let resolved = vec![("6.14.2".to_string(), PathBuf::from("/cache/has;semicolon"))];
        let err = encode_kernel_list(&resolved)
            .expect_err("path containing `;` must be rejected by encoder");
        assert!(
            err.contains("`;`"),
            "error must reference the offending separator: {err}",
        );
        assert!(
            err.contains("6.14.2"),
            "error must name the offending label so the operator can locate the entry: {err}",
        );
        assert!(
            err.contains("/cache/has;semicolon"),
            "error must include the offending path: {err}",
        );
    }

    /// `;` in a label is a wire-format violation distinct from `;`
    /// in a path: the parser's outer `split(';')` upstream of
    /// `split_once('=')` would split a `;`-bearing label into two
    /// pseudo-entries. The encoder rejects with an actionable error
    /// before any output is built so the corrupted env never reaches
    /// the test-binary parser. Pins the label-side label-validation
    /// loop (sibling check to the path-side `;` rejection above).
    #[test]
    fn encode_kernel_list_rejects_semicolon_in_label() {
        let resolved = vec![("evil;label".to_string(), PathBuf::from("/cache/clean"))];
        let err = encode_kernel_list(&resolved)
            .expect_err("label containing `;` must be rejected by encoder");
        assert!(
            err.contains("`;`"),
            "error must reference the offending separator: {err}",
        );
        assert!(
            err.contains("evil;label"),
            "error must name the offending label so the operator \
             can locate the producer that emitted it: {err}",
        );
        // The error explicitly identifies it as a LABEL error, not
        // a path error — distinguishes from the path-side check
        // whose message starts with `kernel directory path`.
        assert!(
            err.contains("kernel label"),
            "error must classify the violation as a label problem (not \
             a path problem) so an operator reading the diagnostic \
             knows which side of the wire format is at fault: {err}",
        );
    }

    /// `=` in a label is a wire-format violation: the parser's
    /// inner `split_once('=')` consumes the FIRST `=` to separate
    /// label from path, so a label `a=b` paired with path `/x` would
    /// emit `a=b=/x`, and the parser would treat `a` as the label
    /// and `b=/x` as the path — silently misrouting the kernel
    /// directory. Pins the second label-validation check in
    /// `encode_kernel_list`. (Note: `=` in PATHS is fine — the
    /// parser only consumes the first `=` and subsequent ones land
    /// inside the path payload — so there is no symmetric path-side
    /// `=` rejection.)
    #[test]
    fn encode_kernel_list_rejects_equals_in_label() {
        let resolved = vec![("evil=label".to_string(), PathBuf::from("/cache/clean"))];
        let err = encode_kernel_list(&resolved)
            .expect_err("label containing `=` must be rejected by encoder");
        assert!(
            err.contains("`=`"),
            "error must reference the offending separator: {err}",
        );
        assert!(
            err.contains("evil=label"),
            "error must name the offending label so the operator \
             can locate the producer that emitted it: {err}",
        );
        assert!(
            err.contains("kernel label"),
            "error must classify the violation as a label problem: {err}",
        );
    }

    #[test]
    fn encode_kernel_list_first_entry_with_semicolon_rejected_before_emit() {
        // Even on a multi-entry payload where ONLY the first entry's
        // path has a `;`, the encoder must bail without emitting
        // anything — partial encoding would mean the caller exec's
        // a child with a corrupted env value where the early entries
        // succeeded.
        let resolved = vec![
            ("first".to_string(), PathBuf::from("/cache/has;semicolon")),
            ("second".to_string(), PathBuf::from("/cache/clean")),
        ];
        let err = encode_kernel_list(&resolved)
            .expect_err("path containing `;` must be rejected even when other entries are clean");
        assert!(err.contains("first"));
    }

    #[test]
    fn encode_kernel_list_later_entry_with_semicolon_still_rejected() {
        // The validation loop scans every entry before emit, so a
        // `;` in the second/later entry's path also bails.
        let resolved = vec![
            ("first".to_string(), PathBuf::from("/cache/clean")),
            ("second".to_string(), PathBuf::from("/cache/has;semicolon")),
        ];
        let err = encode_kernel_list(&resolved)
            .expect_err("`;` anywhere in any path must abort the encode");
        assert!(err.contains("second"));
    }

    // ---------------------------------------------------------------
    // detect_label_collisions — sanitization-collision guard
    // ---------------------------------------------------------------

    #[test]
    fn detect_label_collisions_empty_input_succeeds() {
        // Trivial: an empty resolved set has no pairs to compare;
        // the helper must return Ok without error.
        let resolved: Vec<(String, PathBuf)> = Vec::new();
        detect_label_collisions(&resolved).expect("empty input must succeed");
    }

    #[test]
    fn detect_label_collisions_unique_labels_succeed() {
        // Two distinct labels that sanitize to distinct nextest
        // identifiers — no collision, no error.
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("6.15.0".to_string(), PathBuf::from("/cache/b")),
        ];
        detect_label_collisions(&resolved).expect("distinct sanitized identifiers must succeed");
    }

    #[test]
    fn detect_label_collisions_period_vs_dash_collides() {
        // `sanitize_kernel_label` replaces both `.` and `-` with
        // `_` — so `6.14.2` and `6-14-2` both sanitize to
        // `kernel_6_14_2`. This is the canonical collision shape
        // referenced in the doc comment ("e.g. spell `6.14.2` and
        // `git+...#6.14.2` distinctly").
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("6-14-2".to_string(), PathBuf::from("/cache/b")),
        ];
        let err = detect_label_collisions(&resolved)
            .expect_err("colliding sanitized identifiers must surface an error");
        // Both labels named in the diagnostic so the operator can
        // disambiguate without grepping the resolver source.
        assert!(
            err.contains("6.14.2"),
            "error must name first colliding label: {err}",
        );
        assert!(
            err.contains("6-14-2"),
            "error must name second colliding label: {err}",
        );
        // Sanitized form named so the operator sees the shared
        // identifier the dispatch side would have used.
        assert!(
            err.contains("kernel_6_14_2"),
            "error must include the shared sanitized identifier: {err}",
        );
        // Diagnostic carries the actionable hint.
        assert!(
            err.contains("Spell each --kernel value distinctly"),
            "error must include the actionable remediation hint: {err}",
        );
    }

    #[test]
    fn detect_label_collisions_uppercase_vs_lowercase_collides() {
        // `sanitize_kernel_label` lowercases its input, so `ABC`
        // and `abc` both sanitize to `kernel_abc`. Distinct
        // collision shape from the period-vs-dash case — pins the
        // case-folding contract.
        let resolved = vec![
            ("ABC".to_string(), PathBuf::from("/cache/a")),
            ("abc".to_string(), PathBuf::from("/cache/b")),
        ];
        let err = detect_label_collisions(&resolved)
            .expect_err("uppercase vs lowercase labels must collide post-sanitize");
        assert!(err.contains("kernel_abc"));
    }

    #[test]
    fn detect_label_collisions_identical_labels_collide() {
        // De-duplication of identical `--kernel` specs is the
        // operator's responsibility; this helper is the LAST line
        // of defense and must surface the duplicate as a collision
        // rather than silently letting both entries through.
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("6.14.2".to_string(), PathBuf::from("/cache/b")),
        ];
        let err = detect_label_collisions(&resolved)
            .expect_err("two identical labels must surface as a collision");
        assert!(err.contains("6.14.2"));
        assert!(err.contains("kernel_6_14_2"));
    }

    #[test]
    fn detect_label_collisions_three_entries_two_collide_one_unique() {
        // First two collide after sanitization; third is distinct.
        // The helper must bail on the first detected collision —
        // the unique third entry never reaches the diagnostic but
        // its absence from the error message is intentional (the
        // operator only needs to know which two labels conflict).
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("6-14-2".to_string(), PathBuf::from("/cache/b")),
            ("7.0.0".to_string(), PathBuf::from("/cache/c")),
        ];
        let err = detect_label_collisions(&resolved)
            .expect_err("collision in the first two entries must surface");
        assert!(err.contains("6.14.2"));
        assert!(err.contains("6-14-2"));
        // Third entry's label not mentioned — only the conflicting
        // pair is named (the API contract is "name the first
        // colliding pair", not "enumerate every collision").
        assert!(
            !err.contains("7.0.0"),
            "non-conflicting label should not appear in the collision diagnostic: {err}",
        );
    }

    #[test]
    fn detect_label_collisions_first_two_unique_third_collides_with_first() {
        // First and third collide; second is unique. Ensures the
        // detection scans past the unique second entry rather than
        // bailing as soon as a non-collision is seen.
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("7.0.0".to_string(), PathBuf::from("/cache/b")),
            ("6-14-2".to_string(), PathBuf::from("/cache/c")),
        ];
        let err = detect_label_collisions(&resolved)
            .expect_err("late-arriving collision against an earlier entry must surface");
        // The diagnostic names the EARLIER entry (the one already
        // in `seen`) as the `prior` label and the LATER entry as
        // the `label`. The shared sanitized form is also named.
        assert!(err.contains("6.14.2"), "earlier (prior) label must appear");
        assert!(err.contains("6-14-2"), "later label must appear");
        assert!(err.contains("kernel_6_14_2"));
    }

    // ---------------------------------------------------------------
    // preflight_collision_check — pre-resolve fast-fail
    // ---------------------------------------------------------------

    #[test]
    fn preflight_collision_check_empty_input_succeeds() {
        // Empty spec set has no pairs to compare; the helper must
        // return Ok without iterating anything.
        preflight_collision_check(&[]).expect("empty input must succeed");
    }

    #[test]
    fn preflight_collision_check_unique_versions_succeed() {
        // Two distinct Version specs that sanitize to distinct
        // identifiers — no collision, no error.
        let specs = vec!["6.14.2".to_string(), "6.15.0".to_string()];
        preflight_collision_check(&specs)
            .expect("distinct sanitized identifiers must succeed at pre-flight");
    }

    #[test]
    fn preflight_collision_check_period_vs_dash_collides() {
        // The canonical collision shape: `6.14.2` parses as
        // KernelId::Version (label = "6.14.2"); `6-14-2` parses as
        // KernelId::CacheKey (no `.` → fails version-string check)
        // and its `cache_key_to_version_label` falls through to the
        // raw key "6-14-2" because no `-tarball-` / `-git-` /
        // `local-` tag matches. Both labels sanitize to
        // `kernel_6_14_2`. Pre-flight must bail with both labels and
        // the shared sanitized form named.
        let specs = vec!["6.14.2".to_string(), "6-14-2".to_string()];
        let err = preflight_collision_check(&specs)
            .expect_err("colliding labels must surface a pre-flight error");
        assert!(err.contains("6.14.2"), "error must name first label: {err}");
        assert!(
            err.contains("6-14-2"),
            "error must name second label: {err}"
        );
        assert!(
            err.contains("kernel_6_14_2"),
            "error must include the shared sanitized identifier: {err}",
        );
        // Pre-flight diagnostic distinguishes itself from the
        // post-resolve `detect_label_collisions` error by prefixing
        // with "pre-flight check found collision before any
        // download or build started" — the two diagnostics are
        // distinct so an operator can tell which gate fired.
        assert!(
            err.contains("pre-flight check found collision"),
            "error must be the pre-flight diagnostic, not the post-resolve one: {err}",
        );
    }

    #[test]
    fn preflight_collision_check_identical_versions_succeed() {
        // Two identical `--kernel 6.14.2` specs sanitize to the same
        // identifier but the `prior != label` guard inside
        // `preflight_collision_check` skips the bail on identical
        // labels — those folder into a single entry by
        // `dedupe_resolved` post-resolve. Pins that the helper does
        // NOT confuse "operator passed the same spec twice" with
        // "two distinct specs that collide".
        let specs = vec!["6.14.2".to_string(), "6.14.2".to_string()];
        preflight_collision_check(&specs)
            .expect("identical specs must NOT bail at pre-flight (handled by dedupe post-resolve)");
    }

    #[test]
    fn preflight_collision_check_skips_path_and_range_specs() {
        // Path specs (recognized by `/` prefix per
        // `KernelId::parse`) and Range specs (`A..B` shape) are
        // EXCLUDED from pre-flight because their labels require
        // I/O. Two paths that would collide on their `path_basename
        // _hash6` labels must NOT bail at pre-flight — they reach
        // post-resolve `detect_label_collisions` after
        // canonicalization. Pin the deferred branch by passing two
        // Path specs that, sans I/O, cannot have their labels
        // computed at pre-flight time.
        let specs = vec![
            "/tmp/kernel-a".to_string(),
            "/tmp/kernel-b".to_string(),
            "6.14.2..6.14.4".to_string(),
        ];
        preflight_collision_check(&specs).expect(
            "Path and Range specs must skip pre-flight — their labels are deferred to post-resolve",
        );
    }

    #[test]
    fn preflight_collision_check_skips_empty_and_whitespace_specs() {
        // `resolve_kernel_set` skips trim()-empty specs at the
        // parallel iterator (filter_map). The pre-flight loop
        // applies the same trim+empty skip so a spurious blank
        // `--kernel ""` doesn't reach `KernelId::parse` (which
        // would parse `""` as KernelId::CacheKey("") and produce
        // `sanitize_kernel_label("") == "kernel_"` — a real but
        // useless collision risk). Pin the upstream filter so a
        // regression that dropped the empty-skip guard surfaces
        // as a behavior change.
        let specs = vec!["".to_string(), "   ".to_string(), "6.14.2".to_string()];
        preflight_collision_check(&specs)
            .expect("blank / whitespace-only specs must be silently skipped");
    }

    #[test]
    fn preflight_collision_check_inverted_range_fails_validation() {
        // An inverted Range (`6.15..6.14`) fails `KernelId::validate`
        // pre-resolve. Pre-flight surfaces the inversion diagnostic
        // BEFORE the rayon resolve fires — matches the timing the
        // parallel pipeline preserved on its own pre-extraction.
        let specs = vec!["6.15..6.14".to_string()];
        let err = preflight_collision_check(&specs)
            .expect_err("inverted range must fail pre-flight validation");
        assert!(
            err.contains("inverted kernel range") || err.contains("--kernel"),
            "error must surface the inversion diagnostic with --kernel framing: {err}",
        );
    }

    #[test]
    fn preflight_collision_check_git_url_collision() {
        // Two distinct `git+URL#REF` specs that produce
        // `git_owner_repo_ref`-shape labels can collide if they
        // share owner/repo/ref segments. Construct two URLs whose
        // git_kernel_label outputs differ only in `.` vs `-`
        // characters that sanitize to `_`.
        // - `git+ssh://h/foo/bar#v6.14` → `git_foo_bar_v6.14`
        //   sanitizes to `kernel_git_foo_bar_v6_14`.
        // - `git+ssh://h/foo/bar#v6-14` → `git_foo_bar_v6-14`
        //   sanitizes to the same `kernel_git_foo_bar_v6_14`.
        let specs = vec![
            "git+ssh://host/foo/bar#v6.14".to_string(),
            "git+ssh://host/foo/bar#v6-14".to_string(),
        ];
        let err = preflight_collision_check(&specs)
            .expect_err("colliding git refs must surface a pre-flight error");
        assert!(err.contains("git_foo_bar_v6.14") || err.contains("git_foo_bar_v6-14"));
        assert!(err.contains("kernel_git_foo_bar_v6_14"));
    }

    // ---------------------------------------------------------------
    // dedupe_resolved — order-preserving tuple-level dedup
    // ---------------------------------------------------------------

    #[test]
    fn dedupe_resolved_empty_input_returns_empty() {
        let resolved: Vec<(String, PathBuf)> = Vec::new();
        let deduped = dedupe_resolved(resolved);
        assert!(deduped.is_empty());
    }

    #[test]
    fn dedupe_resolved_unique_inputs_pass_through() {
        // No duplicates → output identical to input, in order.
        let resolved = vec![
            ("a".to_string(), PathBuf::from("/cache/a")),
            ("b".to_string(), PathBuf::from("/cache/b")),
            ("c".to_string(), PathBuf::from("/cache/c")),
        ];
        let deduped = dedupe_resolved(resolved.clone());
        assert_eq!(deduped, resolved);
    }

    #[test]
    fn dedupe_resolved_two_identical_tuples_collapse_to_one() {
        // The canonical dedupe case: two `--kernel 6.14.2` specs
        // resolve to the same `(label, path)` tuple. Output must be
        // a single entry.
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/v")),
            ("6.14.2".to_string(), PathBuf::from("/cache/v")),
        ];
        let deduped = dedupe_resolved(resolved);
        assert_eq!(
            deduped.len(),
            1,
            "identical tuples must collapse to one entry"
        );
        assert_eq!(deduped[0].0, "6.14.2");
        assert_eq!(deduped[0].1, PathBuf::from("/cache/v"));
    }

    #[test]
    fn dedupe_resolved_same_label_different_paths_both_survive() {
        // CRITICAL: two specs that resolve to the SAME label but
        // DIFFERENT paths represent a real cache-key collision.
        // Tuple-level dedup must NOT fold them — both rows must
        // survive so the post-dedupe `detect_label_collisions`
        // catches the same-label collision.
        let resolved = vec![
            ("6.14.2".to_string(), PathBuf::from("/cache/a")),
            ("6.14.2".to_string(), PathBuf::from("/cache/b")),
        ];
        let deduped = dedupe_resolved(resolved);
        assert_eq!(
            deduped.len(),
            2,
            "same label + different paths must NOT dedupe — \
             this is a real cache-key collision that detect_label_collisions \
             must still catch downstream",
        );
    }

    #[test]
    fn dedupe_resolved_preserves_input_order() {
        // The downstream wire format is `;`-separated and
        // order-insensitive at the dispatch layer, but stderr
        // diagnostics list kernels in the order the operator passed
        // them — the order-preserving dedup keeps that mapping
        // intact across the rayon shuffle. Pin the order via a
        // first-seen pass on a 4-entry input where the duplicate
        // sits between two other unique entries.
        let resolved = vec![
            ("a".to_string(), PathBuf::from("/cache/a")),
            ("b".to_string(), PathBuf::from("/cache/b")),
            ("a".to_string(), PathBuf::from("/cache/a")),
            ("c".to_string(), PathBuf::from("/cache/c")),
        ];
        let deduped = dedupe_resolved(resolved);
        // Output: a, b, c — `a` first-seen at index 0, second
        // occurrence at index 2 dropped.
        assert_eq!(
            deduped,
            vec![
                ("a".to_string(), PathBuf::from("/cache/a")),
                ("b".to_string(), PathBuf::from("/cache/b")),
                ("c".to_string(), PathBuf::from("/cache/c")),
            ],
        );
    }

    #[test]
    fn dedupe_resolved_three_identical_tuples_collapse_to_one() {
        // Larger duplicate count: three identical tuples fold to
        // one. Pins that the dedupe is set-membership, not
        // pairwise — a regression that compared adjacent entries
        // only would still pass for two duplicates but produce
        // two outputs for three identical inputs.
        let resolved = vec![
            ("v".to_string(), PathBuf::from("/cache/v")),
            ("v".to_string(), PathBuf::from("/cache/v")),
            ("v".to_string(), PathBuf::from("/cache/v")),
        ];
        let deduped = dedupe_resolved(resolved);
        assert_eq!(deduped.len(), 1);
    }
}
