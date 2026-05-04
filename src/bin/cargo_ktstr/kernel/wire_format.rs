//! Kernel-label emission, dedup, collision detection, and the
//! [`KTSTR_KERNEL_LIST_ENV`](ktstr::KTSTR_KERNEL_LIST_ENV) wire format.
//!
//! Houses the pure helpers that translate `KernelId` parses into
//! operator-readable labels (`path_kernel_label`, `git_kernel_label`,
//! `cache_key_to_version_label`, `decorate_path_label_for_dirty`),
//! detect label collisions before and after resolution
//! ([`preflight_collision_check`], [`detect_label_collisions`]),
//! fold benign duplicates ([`dedupe_resolved`]), and serialize the
//! resolved `(label, dir)` set into the `;`-separated wire format
//! the test binary reads ([`encode_kernel_list`]).
//!
//! Extracted from the resolution / build dispatcher so the per-
//! function unit tests run without driving the rayon resolve
//! pipeline (every `resolve_one` arm performs real I/O —
//! canonicalize+build for Path, cache lookup+download for
//! Version/CacheKey, shallow git clone for Git).

use std::path::{Path, PathBuf};

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
