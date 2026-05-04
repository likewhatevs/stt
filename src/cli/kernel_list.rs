//! `kernel list` and `kernel clean` implementations.
//!
//! Holds the table renderer ([`format_entry_row`]), the EOL gate
//! ([`is_eol`] / [`entry_is_eol`]), the cache enumeration entry
//! points ([`kernel_list`], [`kernel_list_range_preview`]) and the
//! per-bucket retention partitioner ([`partition_clean_candidates`])
//! plus the [`kernel_clean`] driver.

use std::io::{BufRead, Write};

use anyhow::{Result, bail};

use crate::cache::{CacheDir, CacheEntry, KconfigStatus};

use super::kernel_cmd::{
    corrupt_footer_if_any, embedded_kconfig_hash, eol_legend_if_any, stale_legend_if_any,
    untracked_legend_if_any,
};
use super::resolve::expand_kernel_range;

/// Extract the `major.minor` series prefix from a version string.
///
/// The minor component is normalized to its leading ASCII-digit run
/// so RC, linux-next, and any other `-suffix` strings collapse to
/// the same prefix as a released kernel in the same series:
/// - `"6.12.81"` → `"6.12"`
/// - `"7.0"` → `"7.0"`
/// - `"6.15-rc3"` → `"6.15"` (RC folds into series)
/// - `"6.16-rc2-next-20260420"` → `"6.16"` (linux-next folds too)
/// - `"7.0-rc1"` → `"7.0"` (brand-new RC matches non-RC same-series)
/// - `"abc"` → `None` (no `.`)
/// - `"6.abc"` → `None` (no digits in minor)
///
/// Returning the same prefix for both sides of the
/// [`is_eol`] comparison is what makes the predicate immune to
/// releases.json and local-cache versions using different
/// RC / pre-release suffixes within the same series.
fn version_prefix(version: &str) -> Option<String> {
    let (major, rest) = version.split_once('.')?;
    let minor_digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if minor_digits.is_empty() {
        return None;
    }
    Some(format!("{major}.{minor_digits}"))
}

/// Return `true` when `version`'s major.minor series is absent
/// from a non-empty `active_prefixes` list — i.e. the version is
/// end-of-life relative to the kernel.org releases snapshot the
/// caller supplied.
///
/// Returns `false` in three cases:
/// - `active_prefixes` is empty. Callers pass an empty slice to
///   signal "active list unknown" (fetch failure, or skipped
///   lookup), per the `kernel list --json` doc contract that
///   fetch failure must not flag any entry EOL. Without the
///   explicit empty-slice guard, `!any(..)` on an empty iterator
///   is `true` and every entry would be tagged EOL — the exact
///   opposite of the contract.
/// - `version` has no parseable major.minor prefix (e.g. a cache
///   key or freeform string).
/// - `version`'s major.minor prefix appears in `active_prefixes`.
fn is_eol(version: &str, active_prefixes: &[String]) -> bool {
    if active_prefixes.is_empty() {
        return false;
    }
    let Some(prefix) = version_prefix(version) else {
        return false;
    };
    !active_prefixes.iter().any(|p| p == &prefix)
}

/// Whether a cache entry is end-of-life relative to the supplied
/// active-prefix list. Handles the `version == None` / `"-"`
/// short-circuit once for both the text-path `(EOL)` tag render in
/// [`format_entry_row`] and the JSON-path `eol` field emission in
/// [`kernel_list`], so the two surfaces cannot drift: any change to
/// the predicate or the missing-version gate lands in both by
/// construction. `kernel_list_eol_json_human_parity` pins this
/// invariant.
pub(crate) fn entry_is_eol(entry: &CacheEntry, active_prefixes: &[String]) -> bool {
    let v = entry.metadata.version.as_deref().unwrap_or("-");
    v != "-" && is_eol(v, active_prefixes)
}

/// Fetch active kernel series prefixes from releases.json.
///
/// Returns major.minor prefixes for every stable/longterm/mainline
/// entry on success. Propagates the underlying
/// [`crate::fetch::cached_releases`] error on failure (network error,
/// HTTP status, JSON parse failure, missing releases array) so
/// callers can distinguish "fetched and empty" (kernel.org shipped
/// no active series — a violated assumption) from "fetch failed"
/// (transient outage where EOL annotation must degrade, not flip).
///
/// See [`is_eol`]'s empty-slice guard for the recommended fallback pattern.
pub(crate) fn fetch_active_prefixes() -> anyhow::Result<Vec<String>> {
    // Route through the process-wide releases.json cache so the
    // EOL-annotation pass shares its fetch with the rayon-driven
    // resolve pipeline that calls [`expand_kernel_range`] under
    // `cargo ktstr`'s `resolve_kernel_set`. First caller across
    // the whole process pays the network cost; every subsequent
    // caller (within this command or peer Range/active-prefix
    // consumers) clones the cached vector.
    let releases = crate::fetch::cached_releases()?;
    Ok(active_prefixes_from_releases(&releases))
}

/// Reduce [`Release`](crate::fetch::Release) rows to the deduplicated
/// list of major.minor prefixes the `(EOL)` annotation compares
/// against.
///
/// Separated from [`fetch_active_prefixes`] so the normalization path
/// — `linux-next` skip, RC-suffix collapse via [`version_prefix`], and
/// first-seen dedup preserving input order — is testable without
/// hitting the network. The on-network wrapper is a one-line adapter
/// over this helper, so any future change to the normalization lands
/// here once and both call sites consume it.
fn active_prefixes_from_releases(releases: &[crate::fetch::Release]) -> Vec<String> {
    let mut prefixes = Vec::new();
    for r in releases {
        if crate::fetch::is_skippable_release_moniker(&r.moniker) {
            continue;
        }
        if let Some(prefix) = version_prefix(&r.version)
            && !prefixes.contains(&prefix)
        {
            prefixes.push(prefix);
        }
    }
    prefixes
}

/// Format a human-readable table row for a cache entry.
pub fn format_entry_row(
    entry: &CacheEntry,
    kconfig_hash: &str,
    active_prefixes: &[String],
) -> String {
    let meta = &entry.metadata;
    let version = meta.version.as_deref().unwrap_or("-");
    let source = meta.source.to_string();
    let mut tags = String::new();
    // Compose the kconfig tag from `KconfigStatus`'s `Display` impl
    // so the tag word ("stale" / "untracked") and the JSON
    // `kconfig_status` field both flow through one source of truth.
    // `Matches` emits no tag — `kernel list` only annotates entries
    // that deviate from the current kconfig.
    let status = entry.kconfig_status(kconfig_hash);
    if !matches!(status, KconfigStatus::Matches) {
        tags.push_str(&format!(" ({status} kconfig)"));
    }
    // `(extra kconfig)` is orthogonal to baked-in status: an entry
    // can be Matches/Stale/Untracked AND carry user extras. Emit
    // independently so an operator reading the table sees both
    // signals without one masking the other.
    if entry.has_extra_kconfig() {
        tags.push_str(" (extra kconfig)");
    }
    if entry_is_eol(entry, active_prefixes) {
        tags.push_str(" (EOL)");
    }
    format!(
        "  {:<48} {:<12} {:<8} {:<7} {}{}",
        entry.key, version, source, meta.arch, meta.built_at, tags,
    )
}

/// List cached kernel images.
///
/// # JSON output schema (`--json`)
///
/// ```json
/// {
///   "current_ktstr_kconfig_hash": "abc123...",
///   "active_prefixes_fetch_error": null,
///   "entries": [
///     {
///       "key": "7.1.0-rc2",
///       "path": "/path/to/cache/entry",
///       "version": "7.1.0-rc2",
///       "source": { "type": "tarball" },
///       "arch": "x86_64",
///       "built_at": "2026-04-15T12:34:56Z",
///       "ktstr_kconfig_hash": "abc123...",
///       "extra_kconfig_hash": null,
///       "kconfig_status": "matches",
///       "eol": false,
///       "config_hash": "def456...",
///       "image_name": "bzImage",
///       "image_path": "/path/to/cache/entry/bzImage",
///       "has_vmlinux": true,
///       "vmlinux_stripped": true
///     },
///     {
///       "key": "6.12.0-broken",
///       "path": "/path/to/cache/broken-entry",
///       "error": "metadata.json schema drift: missing field `source` at line 1 column 21",
///       "error_kind": "schema_drift"
///     }
///   ]
/// }
/// ```
///
/// **Wrapper fields:**
/// - `current_ktstr_kconfig_hash`: hex digest of the kconfig fragment
///   the running binary was built against, so consumers can detect
///   entries that were built with a different fragment.
/// - `active_prefixes_fetch_error`: `null` on success, human-readable
///   error string on failure to fetch the active kernel-series list
///   from kernel.org. When non-null, `eol` annotation is disabled for
///   the run (no series data to compare against) and every entry's
///   `eol` is `false` regardless of actual support status — so
///   consumers must check this field before trusting `eol`.
/// - `entries`: heterogeneous array; each element is either a valid
///   entry (object with the full field set) or a corrupt entry
///   (object with only `key`, `path`, and `error`). Corrupt entries
///   have a structurally different shape — consumers should detect the
///   `"error"` key and branch.
///
/// **Entry fields (valid entries):**
/// - `kconfig_status`: one of `"matches"`, `"stale"`, or `"untracked"`
///   (the Display forms of `cache::KconfigStatus`). `matches` means
///   the entry's `ktstr_kconfig_hash` equals
///   `current_ktstr_kconfig_hash`; `stale` means they differ;
///   `untracked` means the entry has no recorded kconfig hash (pre-dates
///   kconfig hash tracking).
/// - `extra_kconfig_hash`: CRC32 (8 hex chars, lowercase) of the user
///   `--extra-kconfig` fragment as raw bytes (no canonicalization), or
///   `null` when the entry was built without `--extra-kconfig`. Cache
///   keys grow from `kc{baked}` to `kc{baked}-xkc{extra}` when extras
///   are present; this field stores the `xkc` segment so `kernel list`
///   is self-describing for entries that carry user modifications.
///   Independent of `kconfig_status` — an entry can match the baked-in
///   hash AND carry a non-null extras hash.
/// - `eol`: `true` iff the entry's version series does not appear in
///   the active-prefix list. Only meaningful when
///   `active_prefixes_fetch_error` is `null`.
/// - `has_vmlinux`: whether the cache entry includes the uncompressed
///   `vmlinux` (needed for DWARF-driven probes); when `false`, only
///   the compressed `image_path` is available.
/// - `vmlinux_stripped`: whether the cached vmlinux came from a
///   successful strip pass (`true`) or the raw-fallback path
///   (`false`). A `false` here indicates the strip pipeline errored
///   on this kernel and the unstripped bytes were copied instead —
///   the entry still works but carries a large on-disk payload that
///   signals a parseability regression worth investigating. Always
///   `false` when `has_vmlinux` is `false`.
/// - `source`: tagged object (serde internally tagged on `"type"`).
///   Variants: `{"type": "tarball"}`, `{"type": "git", "git_hash": ?,
///   "ref": ?}`, `{"type": "local", "source_tree_path": ?, "git_hash":
///   ?}`. Variant-specific fields are nullable — consumers must
///   dispatch on `"type"` before reading them. See `cache::KernelSource`.
///
/// **Entry fields (corrupt entries):**
/// - `error`: human-readable reason from `cache::read_metadata`,
///   prefixed by failure class so programmatic consumers can branch
///   on `starts_with` without parsing the free-form tail. Prefixes:
///   - `"metadata.json missing"` — file absent (not a cache entry).
///   - `"metadata.json unreadable: ..."` — I/O error on
///     `fs::read_to_string` other than ENOENT (e.g. EISDIR,
///     permission).
///   - `"metadata.json schema drift: ..."` — JSON parsed but does
///     not match the `KernelMetadata` shape (serde_json
///     `Category::Data`). Typical cause: older cache from a ktstr
///     whose schema has since changed.
///   - `"metadata.json malformed: ..."` — not valid JSON at all
///     (serde_json `Category::Syntax`).
///   - `"metadata.json truncated: ..."` — JSON ends mid-value
///     (serde_json `Category::Eof`), e.g. a partially-written
///     metadata from a crashed `store()`.
///   - `"metadata.json parse error: ..."` — fallback for an
///     unexpected `Category::Io` from `from_str`; does not fire on
///     the current serde_json version but kept as a defense-in-depth
///     fallback so the field is never absent.
///   - `"image file <name> missing from entry directory"` —
///     metadata parsed cleanly but the declared image file is gone
///     (partial download, manual deletion, failed strip+rename).
///
///   The example above shows the schema-drift case; consumers that
///   treat corrupt entries as a single category can key on the
///   `"error"` key alone.
/// - `error_kind`: machine-readable classification of the failure
///   mode — a stable snake_case identifier CI scripts can dispatch
///   on without parsing the free-form `error`. Values:
///   `"missing"`, `"unreadable"`, `"schema_drift"`, `"malformed"`,
///   `"truncated"`, `"parse_error"`, `"image_missing"`, and
///   `"unknown"` as a defensive fallback for a future producer
///   prefix that has not yet been taught to the classifier. Always
///   present on corrupt entries; always absent on valid entries.
///   See [`crate::cache::ListedEntry::error_kind`] for the
///   classifier contract.
pub fn kernel_list(json: bool) -> Result<()> {
    kernel_list_inner(json, None)
}

/// Range-preview variant of [`kernel_list`].
///
/// Routes through [`kernel_list_inner`] with `range = Some(spec)`,
/// switching the subcommand from "walk the cache and list local
/// entries" to "fetch releases.json once and print the versions
/// `spec` expands to." See the `range` arg's doc on
/// [`super::KernelCommand::List`] for operator-facing semantics.
///
/// Surfaced as a thin wrapper because the binary dispatch sites
/// (`ktstr::kernel kernel list --range R` /
/// `cargo ktstr kernel list --range R`) read more naturally as
/// `cli::kernel_list_range_preview(json, R)` than as
/// `cli::kernel_list_inner(json, Some(R))`. The shared inner
/// function keeps a single `--json` formatter and a single test
/// surface.
pub fn kernel_list_range_preview(json: bool, range: &str) -> Result<()> {
    kernel_list_inner(json, Some(range))
}

fn kernel_list_inner(json: bool, range: Option<&str>) -> Result<()> {
    if let Some(spec) = range {
        return run_kernel_list_range(json, spec);
    }
    let cache = CacheDir::new()?;
    let entries = cache.list()?;
    let kconfig_hash = embedded_kconfig_hash();

    // Track the fetch result so the `--json` path can surface the
    // error string to scripted consumers. Before this, a failure
    // was eprintln'd but never appeared in the JSON wrapper, so
    // downstream tooling could only observe "all entries are
    // non-EOL" without any signal that the prefix list was
    // actually empty because the network fetch failed.
    let (active_prefixes, active_prefixes_fetch_error): (Vec<String>, Option<String>) =
        match fetch_active_prefixes() {
            Ok(p) => (p, None),
            Err(e) => {
                let msg = format!("{e:#}");
                eprintln!(
                    "kernel list: failed to fetch active kernel series ({msg}); \
                     EOL annotation disabled for this run. \
                     Check that kernel.org is reachable from this host.",
                );
                (Vec::new(), Some(msg))
            }
        };

    if json {
        let json_entries: Vec<serde_json::Value> = entries
            .iter()
            .map(|e| match e {
                crate::cache::ListedEntry::Valid(entry) => {
                    let meta = &entry.metadata;
                    let eol = entry_is_eol(entry, &active_prefixes);
                    let kconfig_status = entry.kconfig_status(&kconfig_hash).to_string();
                    serde_json::json!({
                        "key": entry.key,
                        "path": entry.path.display().to_string(),
                        "version": meta.version,
                        "source": meta.source,
                        "arch": meta.arch,
                        "built_at": meta.built_at,
                        "ktstr_kconfig_hash": meta.ktstr_kconfig_hash,
                        "extra_kconfig_hash": meta.extra_kconfig_hash,
                        "kconfig_status": kconfig_status,
                        "eol": eol,
                        "config_hash": meta.config_hash,
                        "image_name": meta.image_name,
                        "image_path": entry.image_path().display().to_string(),
                        "has_vmlinux": meta.has_vmlinux(),
                        "vmlinux_stripped": meta.vmlinux_stripped(),
                    })
                }
                crate::cache::ListedEntry::Corrupt { key, path, reason } => {
                    // `error_kind` is the machine-readable classification
                    // of the failure mode (snake_case identifier); `error`
                    // keeps the human-readable reason. Both fields emit
                    // on every corrupt entry so consumers that dispatch
                    // on `error_kind` AND consumers that display `error`
                    // work without a version gate. See
                    // `ListedEntry::error_kind` for the classifier.
                    let error_kind = e.error_kind().unwrap_or("unknown");
                    serde_json::json!({
                        "key": key,
                        "path": path.display().to_string(),
                        "error": reason,
                        "error_kind": error_kind,
                    })
                }
            })
            .collect();
        // `active_prefixes_fetch_error` is `null` on success and a
        // human-readable string on fetch failure, so JSON consumers
        // can distinguish "no active prefixes learned" (fetch
        // failed, EOL annotation was disabled for this run) from
        // "all kernels are current" (fetch succeeded, list is
        // simply not gating any entry).
        let wrapper = serde_json::json!({
            "current_ktstr_kconfig_hash": kconfig_hash,
            "active_prefixes_fetch_error": active_prefixes_fetch_error,
            "entries": json_entries,
        });
        println!("{}", serde_json::to_string_pretty(&wrapper)?);
        return Ok(());
    }

    eprintln!("cache: {}", cache.root().display());

    if entries.is_empty() {
        println!("no cached kernels. Run `kernel build` to download and build a kernel.");
        return Ok(());
    }

    println!(
        "  {:<48} {:<12} {:<8} {:<7} BUILT",
        "KEY", "VERSION", "SOURCE", "ARCH"
    );
    let mut any_stale = false;
    let mut any_untracked = false;
    let mut any_eol = false;
    let mut corrupt_count: usize = 0;
    for listed in &entries {
        match listed {
            crate::cache::ListedEntry::Valid(entry) => {
                let status = entry.kconfig_status(&kconfig_hash);
                if status.is_stale() {
                    any_stale = true;
                }
                if status.is_untracked() {
                    any_untracked = true;
                }
                if entry_is_eol(entry, &active_prefixes) {
                    any_eol = true;
                }
                println!(
                    "{}",
                    format_entry_row(entry, &kconfig_hash, &active_prefixes)
                );
            }
            crate::cache::ListedEntry::Corrupt { key, reason, .. } => {
                corrupt_count += 1;
                println!("  {key:<48} (corrupt: {reason})");
            }
        }
    }
    // Annotation footers. The emission order is fixed and load-bearing
    // — the integration test
    // `kernel_list_legend_ordering_pins_untracked_stale_corrupt` in
    // `tests/ktstr_cli.rs` pins the sequence against regressions by
    // running the real binary against a fixture cache:
    //
    //   1. EOL        (informational, inherent-to-upstream-release)
    //   2. untracked  (informational, actionable with a rebuild)
    //   3. stale      (informational, actionable with a rebuild)
    //   4. corrupt    (operational, requires manual inspection + clean)
    //
    // Rationale: informational legends come first because they do
    // not demand operator action to resolve — an EOL tag is a state
    // of the world, not a cache pathology. The `untracked` and
    // `stale` legends share a remediation shape (`kernel build
    // --force VERSION`) and are grouped adjacent so an operator who
    // needs to batch-rebuild sees the two one-line recipes together.
    // The corrupt footer comes last because its remediation is the
    // most disruptive (`kernel clean`), runs against a separate
    // command, and interpolates a runtime cache-root path that is
    // irrelevant to the preceding tags; surfacing it last keeps the
    // informational/operational distinction visually obvious in the
    // output stream.
    //
    // Each legend surfaces only when a tag was actually rendered, so
    // the normal no-tag case stays noise-free. Decisions are routed
    // through the `*_legend_if_any` / `*_footer_if_any` helpers so
    // both branches per legend are unit-testable.
    //
    // Channel: stderr (diagnostic). The rendered entry rows above
    // flow to stdout so `kernel list | awk` / `kernel list >
    // kernels.txt` downstream scripts receive table data without
    // legend text mixed in; the legends only become visible on an
    // interactive terminal where both channels are typically
    // displayed. Pinned by `kernel_list_legends_emit_on_stderr` in
    // `tests/ktstr_cli.rs`.
    if let Some(legend) = eol_legend_if_any(any_eol) {
        eprintln!("{legend}");
    }
    if let Some(legend) = untracked_legend_if_any(any_untracked) {
        eprintln!("{legend}");
    }
    if let Some(legend) = stale_legend_if_any(any_stale) {
        eprintln!("{legend}");
    }
    if let Some(footer) = corrupt_footer_if_any(corrupt_count, cache.root()) {
        eprintln!("{footer}");
    }
    Ok(())
}

/// Render a `kernel list --range START..END` preview by parsing
/// `spec` as a [`crate::kernel_path::KernelId::Range`], expanding
/// it via [`expand_kernel_range`], and printing the resulting
/// version list.
///
/// Performs no cache reads or builds — only the single
/// `releases.json` fetch [`expand_kernel_range`] already runs for
/// real range resolves. Bails when:
/// - `spec` does not parse as a `Range` (passes through
///   `KernelId::parse` and rejects non-Range variants with an
///   actionable diagnostic naming the expected shape);
/// - `KernelId::Range::validate` rejects the endpoints (inverted
///   range, malformed version components — same diagnostics the
///   real resolver emits);
/// - the network fetch fails or the range expands to zero
///   versions (the same hard-error contract documented on
///   [`expand_kernel_range`]).
///
/// Output shape mirrors `kernel list`:
/// - text: one version per line on stdout, prefixed with the
///   parsed range and version count on stderr so shell pipelines
///   (`| awk`, `| grep`) see clean stdout.
/// - JSON: a single object with the literal range, the parsed
///   start / end strings, and the expanded version array.
fn run_kernel_list_range(json: bool, spec: &str) -> Result<()> {
    use crate::kernel_path::KernelId;

    let id = KernelId::parse(spec);
    let (start, end) = match &id {
        KernelId::Range { start, end } => (start.clone(), end.clone()),
        _ => {
            bail!(
                "kernel list --range: `{spec}` does not parse as a \
                 `START..END` range. Expected `MAJOR.MINOR[.PATCH][-rcN]..\
                 MAJOR.MINOR[.PATCH][-rcN]` (e.g. `6.12..6.14`)."
            );
        }
    };
    id.validate()
        .map_err(|e| anyhow::anyhow!("kernel list --range {spec}: {e}"))?;

    let versions = expand_kernel_range(&start, &end, "kernel list")?;

    if json {
        let payload = serde_json::json!({
            "range": spec,
            "start": start,
            "end": end,
            "versions": versions,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    // Text output: versions on stdout (one per line) so
    // `kernel list --range R | xargs -I{} kernel build {}`
    // works without tearing on legend lines. The header on
    // stderr matches `expand_kernel_range`'s own status output
    // shape so the operator gets the same "expanded to N
    // kernel(s)" context they would see during a real resolve.
    for v in &versions {
        println!("{v}");
    }
    Ok(())
}

/// Pure partitioner for [`kernel_clean`]: given an ordered
/// (newest-first per `cache::list()`) slice of entries, return the
/// subset that should be removed.
///
/// Split from [`kernel_clean`] so the policy is covered by
/// fixture tests without touching the filesystem: selection
/// semantics are a four-axis matrix (`Valid` vs `Corrupt`, `keep`
/// vs no keep, `corrupt_only` true vs false) and the previous
/// inline loop made every edge regress-only-at-runtime.
///
/// Rules:
/// - `Corrupt` entries are always removal candidates (they occupy
///   disk without being usable, and never consume a `keep` slot).
/// - `Valid` entries are removal candidates only when
///   `corrupt_only = false`; the first `keep.unwrap_or(0)` valid
///   entries PER
///   `(version, ktstr_kconfig_hash, extra_kconfig_hash)` BUCKET in
///   input order are retained, every subsequent valid entry in
///   that bucket is a candidate.
/// - Input order is preserved in the output — `cache.list()` sorts
///   `built_at`-descending, so the retained `keep` prefix per
///   bucket is the most recent entries.
///
/// **Bucketing rationale**: a single `--keep N` pool would let a
/// flurry of builds at one configuration evict useful entries at a
/// different configuration. Bucketing by the
/// `(version, baked-in-kconfig, extras)` tuple preserves the N
/// newest entries in each configuration variant independently, so
/// users iterating on extras for one kernel don't lose unrelated
/// cache slots, and a `ktstr.kconfig` bump (changes
/// `ktstr_kconfig_hash`) doesn't push out the prior baked-in
/// build's slots before the new one is fully exercised.
///
/// `None` participates as its own bucket key value:
/// - `version: None` (local/untagged builds) is distinct from any
///   `Some(version)`.
/// - `ktstr_kconfig_hash: None` (entries that predate the field)
///   is distinct from any `Some(hash)`.
/// - `extra_kconfig_hash: None` (no user extras) is distinct from
///   any `Some(hash)`.
fn partition_clean_candidates<'a>(
    entries: &'a [crate::cache::ListedEntry],
    keep: Option<usize>,
    corrupt_only: bool,
) -> Vec<&'a crate::cache::ListedEntry> {
    let skip = keep.unwrap_or(0);
    // Bucket key groups Valid entries by `(version,
    // ktstr_kconfig_hash, extra_kconfig_hash)` — three optional
    // strings, distinct shapes need distinct retention counters.
    type BucketKey = (Option<String>, Option<String>, Option<String>);
    let mut bucket_kept: std::collections::HashMap<BucketKey, usize> =
        std::collections::HashMap::new();
    let mut to_remove: Vec<&'a crate::cache::ListedEntry> = Vec::new();
    for listed in entries {
        match listed {
            crate::cache::ListedEntry::Valid(entry) => {
                if corrupt_only {
                    continue;
                }
                let bucket_key = (
                    entry.metadata.version.clone(),
                    entry.metadata.ktstr_kconfig_hash.clone(),
                    entry.metadata.extra_kconfig_hash.clone(),
                );
                let kept = bucket_kept.entry(bucket_key).or_insert(0);
                if *kept < skip {
                    *kept += 1;
                    continue;
                }
                to_remove.push(listed);
            }
            crate::cache::ListedEntry::Corrupt { .. } => {
                to_remove.push(listed);
            }
        }
    }
    to_remove
}

/// Remove cached kernels with optional keep-N and confirmation prompt.
///
/// `corrupt_only = true` narrows removal to `ListedEntry::Corrupt`
/// (metadata missing or unparseable, image file absent); valid
/// entries are left untouched regardless of `keep` / `force`.
///
/// `keep = Some(N)` retains the N newest **valid** entries.
pub fn kernel_clean(keep: Option<usize>, force: bool, corrupt_only: bool) -> Result<()> {
    let cache = CacheDir::new()?;
    let entries = cache.list()?;

    if entries.is_empty() {
        println!("nothing to clean");
        return Ok(());
    }

    let kconfig_hash = embedded_kconfig_hash();

    let to_remove = partition_clean_candidates(&entries, keep, corrupt_only);

    if to_remove.is_empty() {
        println!("nothing to clean");
        return Ok(());
    }

    if !force {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            bail!("confirmation requires a terminal. Use --force to skip.");
        }
        // Fetch active-series prefixes for the (EOL) annotation on
        // the confirmation prompt. Scoped to the `!force` branch —
        // force mode skips the prompt, so there's no point burning
        // a network roundtrip to kernel.org. A fetch failure is
        // surfaced via `eprintln!` (mirroring `kernel_list`'s
        // diagnostic) so the operator knows why the `(EOL)`
        // annotations are missing instead of silently degrading.
        let active_prefixes = match fetch_active_prefixes() {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "kernel clean: failed to fetch active kernel series ({e:#}); \
                     EOL annotation disabled for this run. \
                     Check that kernel.org is reachable from this host."
                );
                Vec::new()
            }
        };
        println!("the following entries will be removed:");
        for listed in &to_remove {
            match listed {
                crate::cache::ListedEntry::Valid(entry) => {
                    println!(
                        "{}",
                        format_entry_row(entry, &kconfig_hash, &active_prefixes)
                    );
                }
                crate::cache::ListedEntry::Corrupt { key, reason, .. } => {
                    println!("  {key:<48} (corrupt: {reason})");
                }
            }
        }
        eprint!("remove {} entries? [y/N] ", to_remove.len());
        std::io::stderr().flush()?;
        let mut answer = String::new();
        std::io::stdin().lock().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y") {
            println!("aborted");
            return Ok(());
        }
    }

    let total = to_remove.len();
    let mut removed = 0usize;
    let mut last_err: Option<String> = None;
    for listed in &to_remove {
        match std::fs::remove_dir_all(listed.path()) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                removed += 1;
            }
            Err(e) => {
                last_err = Some(format!("remove {}: {e}", listed.key()));
            }
        }
    }

    println!("removed {removed} cached kernel(s).");
    if let Some(err) = last_err {
        bail!("removed {removed} of {total} entries; {err}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stable release path.
    #[test]
    fn version_prefix_stable_release() {
        assert_eq!(version_prefix("6.14.2").as_deref(), Some("6.14"));
        assert_eq!(version_prefix("6.12.81").as_deref(), Some("6.12"));
        assert_eq!(version_prefix("7.0").as_deref(), Some("7.0"));
    }

    /// RC suffix collapses into series.
    #[test]
    fn version_prefix_strips_rc_suffix() {
        assert_eq!(version_prefix("6.15-rc1").as_deref(), Some("6.15"));
        assert_eq!(version_prefix("6.15-rc3").as_deref(), Some("6.15"));
        assert_eq!(version_prefix("7.0-rc1").as_deref(), Some("7.0"));
    }

    /// linux-next collapses to merge target.
    #[test]
    fn version_prefix_strips_linux_next_suffix() {
        assert_eq!(
            version_prefix("6.16-rc2-next-20260420").as_deref(),
            Some("6.16"),
        );
        assert_eq!(
            version_prefix("7.1-rc1-next-20260501").as_deref(),
            Some("7.1"),
        );
    }

    /// No dot → None.
    #[test]
    fn version_prefix_rejects_no_dot() {
        assert!(version_prefix("abc").is_none());
        assert!(version_prefix("6").is_none());
        assert!(version_prefix("").is_none());
    }

    /// Non-numeric minor → None.
    #[test]
    fn version_prefix_rejects_non_numeric_minor() {
        assert!(version_prefix("6.x").is_none());
        assert!(version_prefix("6.-rc1").is_none());
        assert!(version_prefix("6.").is_none());
    }

    /// Empty active_prefixes is the "active list unknown" signal.
    #[test]
    fn is_eol_empty_active_prefixes_returns_false() {
        assert!(!is_eol("6.14.2", &[]));
    }

    #[test]
    fn is_eol_prefix_in_active_list_returns_false() {
        assert!(!is_eol("6.14.2", &["6.14".to_string()]));
    }

    #[test]
    fn is_eol_prefix_absent_from_active_list_returns_true() {
        assert!(is_eol(
            "5.10.200",
            &["6.14".to_string(), "6.12".to_string()],
        ));
    }

    #[test]
    fn is_eol_unparseable_version_returns_false() {
        assert!(!is_eol("abc", &["6.14".to_string()]));
    }

    #[test]
    fn is_eol_rc_suffix_mismatch_does_not_flag() {
        let active = ["6.15".to_string()];
        assert!(!is_eol("6.15-rc1", &active));
        assert!(!is_eol("6.15-rc4", &active));
    }

    #[test]
    fn is_eol_linux_next_matches_mainline_prefix() {
        let active = ["6.16".to_string()];
        assert!(!is_eol("6.16-rc2-next-20260420", &active));
    }

    #[test]
    fn is_eol_brand_new_major_matches_rc_variant() {
        assert!(!is_eol("7.0", &["7.0".to_string()]));
        assert!(!is_eol("7.0-rc1", &["7.0".to_string()]));
    }

    #[test]
    fn is_eol_brand_new_zero_release_in_active_list() {
        let active = ["7.0".to_string()];
        assert!(!is_eol("7.0", &active));
        assert!(!is_eol("7.0.0", &active));
    }

    #[test]
    fn is_eol_linux_next_version_not_falsely_tagged() {
        assert!(is_eol(
            "6.16-rc1",
            &["6.14".to_string(), "6.13".to_string()]
        ));
    }

    fn owned(pairs: &[(&str, &str)]) -> Vec<crate::fetch::Release> {
        pairs
            .iter()
            .map(|(m, v)| crate::fetch::Release {
                moniker: (*m).to_string(),
                version: (*v).to_string(),
            })
            .collect()
    }

    /// RC-suffixed mainline normalizes to series.
    #[test]
    fn active_prefixes_from_releases_normalizes_rc_versions() {
        let releases = owned(&[
            ("mainline", "6.16-rc3"),
            ("stable", "6.15.2"),
            ("longterm", "6.12.81"),
        ]);
        let prefixes = active_prefixes_from_releases(&releases);
        assert_eq!(
            prefixes,
            vec!["6.16".to_string(), "6.15".to_string(), "6.12".to_string()],
        );
    }

    #[test]
    fn active_prefixes_from_releases_skips_linux_next_moniker() {
        let releases = owned(&[
            ("linux-next", "6.17-rc2-next-20260421"),
            ("mainline", "6.16-rc3"),
            ("stable", "6.15.2"),
        ]);
        let prefixes = active_prefixes_from_releases(&releases);
        assert!(!prefixes.contains(&"6.17".to_string()));
        assert_eq!(prefixes, vec!["6.16".to_string(), "6.15".to_string()]);
    }

    #[test]
    fn active_prefixes_from_releases_dedups_in_input_order() {
        let releases = owned(&[
            ("stable", "6.14.2"),
            ("longterm", "6.14.1"),
            ("longterm", "6.12.81"),
        ]);
        let prefixes = active_prefixes_from_releases(&releases);
        assert_eq!(prefixes, vec!["6.14".to_string(), "6.12".to_string()]);
    }

    /// kernel_list_range_preview rejects non-Range spec.
    #[test]
    fn kernel_list_range_preview_rejects_non_range_spec() {
        let err = run_kernel_list_range(false, "6.14.2")
            .expect_err("bare version must not parse as a Range");
        let msg = format!("{err:#}");
        assert!(msg.contains("does not parse as a `START..END` range"));
        assert!(msg.contains("`6.14.2`"));
    }

    /// kernel_list_range_preview rejects inverted range.
    #[test]
    fn kernel_list_range_preview_rejects_inverted_range() {
        let err = run_kernel_list_range(false, "6.16..6.12")
            .expect_err("inverted range must not be accepted");
        let msg = format!("{err:#}");
        assert!(msg.contains("kernel list --range 6.16..6.12"));
    }

    fn mk_valid(key: &str) -> crate::cache::ListedEntry {
        use crate::cache::{CacheEntry, KernelMetadata, KernelSource};
        let path = std::path::PathBuf::from(format!("/tmp/fixture/{key}"));
        let metadata = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-22T00:00:00Z".to_string(),
        );
        crate::cache::ListedEntry::Valid(Box::new(CacheEntry {
            key: key.to_string(),
            path,
            metadata,
        }))
    }

    fn mk_corrupt(key: &str) -> crate::cache::ListedEntry {
        crate::cache::ListedEntry::Corrupt {
            key: key.to_string(),
            path: std::path::PathBuf::from(format!("/tmp/fixture/{key}")),
            reason: "test fixture corrupt".to_string(),
        }
    }

    #[test]
    fn partition_clean_candidates_empty_input_yields_empty_output() {
        let out = partition_clean_candidates(&[], None, false);
        assert!(out.is_empty());
        let out = partition_clean_candidates(&[], Some(5), true);
        assert!(out.is_empty());
    }

    #[test]
    fn partition_clean_candidates_corrupt_only_skips_valid_entries() {
        let entries = vec![mk_valid("v1"), mk_corrupt("c1"), mk_valid("v2")];
        let out = partition_clean_candidates(&entries, None, true);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].key(), "c1");
    }

    #[test]
    fn partition_clean_candidates_no_keep_removes_every_entry() {
        let entries = vec![mk_valid("v1"), mk_corrupt("c1"), mk_valid("v2")];
        let out = partition_clean_candidates(&entries, None, false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["v1", "c1", "v2"]);
    }

    #[test]
    fn partition_clean_candidates_keep_retains_n_newest_valid_preserves_corrupt() {
        let entries = vec![
            mk_valid("v_new1"),
            mk_corrupt("c_mid"),
            mk_valid("v_new2"),
            mk_valid("v_old"),
        ];
        let out = partition_clean_candidates(&entries, Some(2), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["c_mid", "v_old"]);
    }

    #[test]
    fn partition_clean_candidates_keep_never_preserves_corrupt() {
        let entries = vec![mk_corrupt("c1"), mk_valid("v1"), mk_valid("v2")];
        let out = partition_clean_candidates(&entries, Some(3), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["c1"]);
    }

    /// Defensive cell: corrupt_only=true makes keep inert.
    #[test]
    fn partition_clean_candidates_corrupt_only_ignores_keep() {
        let entries = vec![
            mk_valid("v_new1"),
            mk_corrupt("c_mid"),
            mk_valid("v_new2"),
            mk_valid("v_old"),
        ];
        let out = partition_clean_candidates(&entries, Some(2), true);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["c_mid"]);
    }

    /// Bucket-by-version retention: keep=1 with 4 entries split
    /// across two versions retains the newest from EACH version.
    fn mk_valid_bucketed(
        key: &str,
        version: Option<&str>,
        extra_kconfig_hash: Option<&str>,
    ) -> crate::cache::ListedEntry {
        mk_valid_bucketed_full(key, version, None, extra_kconfig_hash)
    }

    fn mk_valid_bucketed_full(
        key: &str,
        version: Option<&str>,
        ktstr_kconfig_hash: Option<&str>,
        extra_kconfig_hash: Option<&str>,
    ) -> crate::cache::ListedEntry {
        use crate::cache::{CacheEntry, KernelMetadata, KernelSource};
        let path = std::path::PathBuf::from(format!("/tmp/fixture/{key}"));
        let metadata = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-22T00:00:00Z".to_string(),
        )
        .with_version(version.map(String::from))
        .with_ktstr_kconfig_hash(ktstr_kconfig_hash.map(String::from))
        .with_extra_kconfig_hash(extra_kconfig_hash.map(String::from));
        crate::cache::ListedEntry::Valid(Box::new(CacheEntry {
            key: key.to_string(),
            path,
            metadata,
        }))
    }

    #[test]
    fn partition_clean_candidates_keep_buckets_by_version() {
        let entries = vec![
            mk_valid_bucketed("v6_14_new", Some("6.14.2"), None),
            mk_valid_bucketed("v6_15_new", Some("6.15.0"), None),
            mk_valid_bucketed("v6_14_old", Some("6.14.2"), None),
            mk_valid_bucketed("v6_15_old", Some("6.15.0"), None),
        ];
        let out = partition_clean_candidates(&entries, Some(1), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["v6_14_old", "v6_15_old"]);
    }

    #[test]
    fn partition_clean_candidates_keep_buckets_by_extra_kconfig_hash() {
        let entries = vec![
            mk_valid_bucketed("v6_14_xkc_aaaa_new", Some("6.14.2"), Some("aaaa")),
            mk_valid_bucketed("v6_14_xkc_bbbb_new", Some("6.14.2"), Some("bbbb")),
            mk_valid_bucketed("v6_14_xkc_aaaa_old", Some("6.14.2"), Some("aaaa")),
            mk_valid_bucketed("v6_14_xkc_bbbb_old", Some("6.14.2"), Some("bbbb")),
        ];
        let out = partition_clean_candidates(&entries, Some(1), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["v6_14_xkc_aaaa_old", "v6_14_xkc_bbbb_old"]);
    }

    #[test]
    fn partition_clean_candidates_keep_distinguishes_none_from_some_extras() {
        let entries = vec![
            mk_valid_bucketed("bare_new", Some("6.14.2"), None),
            mk_valid_bucketed("xkc_new", Some("6.14.2"), Some("aaaa")),
            mk_valid_bucketed("bare_old", Some("6.14.2"), None),
            mk_valid_bucketed("xkc_old", Some("6.14.2"), Some("aaaa")),
        ];
        let out = partition_clean_candidates(&entries, Some(1), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["bare_old", "xkc_old"]);
    }

    #[test]
    fn partition_clean_candidates_keep_per_bucket_with_corrupt_interleaved() {
        let entries = vec![
            mk_valid_bucketed("v6_14_new", Some("6.14.2"), None),
            mk_corrupt("c_mid"),
            mk_valid_bucketed("v6_15_new", Some("6.15.0"), None),
            mk_valid_bucketed("v6_14_old", Some("6.14.2"), None),
        ];
        let out = partition_clean_candidates(&entries, Some(1), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["c_mid", "v6_14_old"]);
    }

    #[test]
    fn partition_clean_candidates_keep_buckets_by_ktstr_kconfig_hash() {
        let entries = vec![
            mk_valid_bucketed_full("baked_v2_new", Some("6.14.2"), Some("v2hash"), None),
            mk_valid_bucketed_full("baked_v1_new", Some("6.14.2"), Some("v1hash"), None),
            mk_valid_bucketed_full("baked_v2_old", Some("6.14.2"), Some("v2hash"), None),
            mk_valid_bucketed_full("baked_v1_old", Some("6.14.2"), Some("v1hash"), None),
        ];
        let out = partition_clean_candidates(&entries, Some(1), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["baked_v2_old", "baked_v1_old"]);
    }

    #[test]
    fn partition_clean_candidates_keep_local_untagged_builds_form_own_bucket() {
        let entries = vec![
            mk_valid_bucketed("local_new", None, None),
            mk_valid_bucketed("v6_14_new", Some("6.14.2"), None),
            mk_valid_bucketed("local_old", None, None),
            mk_valid_bucketed("v6_14_old", Some("6.14.2"), None),
        ];
        let out = partition_clean_candidates(&entries, Some(1), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["local_old", "v6_14_old"]);
    }

    /// `format_entry_row` with `extra_kconfig_hash = Some(_)` must
    /// emit `(extra kconfig)`. Tag is orthogonal to the kconfig-status
    /// tag — entry can be Matches AND carry extras.
    #[test]
    fn format_entry_row_emits_extra_kconfig_tag() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();
        let current_hash = "abc1234";
        let meta_with = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
        .with_ktstr_kconfig_hash(Some(current_hash.to_string()))
        .with_extra_kconfig_hash(Some("deadbeef".to_string()));
        let entry_with = cache
            .store("with-extras", &CacheArtifacts::new(&image), &meta_with)
            .unwrap();
        let row_with = format_entry_row(&entry_with, current_hash, &[]);
        assert!(row_with.contains("(extra kconfig)"));
        assert!(!row_with.contains("(stale kconfig)"));
        assert!(!row_with.contains("(untracked kconfig)"));

        let meta_without = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
        .with_ktstr_kconfig_hash(Some(current_hash.to_string()));
        let entry_without = cache
            .store(
                "without-extras",
                &CacheArtifacts::new(&image),
                &meta_without,
            )
            .unwrap();
        let row_without = format_entry_row(&entry_without, current_hash, &[]);
        assert!(!row_without.contains("(extra kconfig)"));
    }

    /// Empty `active_prefixes` must NOT tag any entry EOL — that's
    /// the fetch-failed fallback signal.
    #[test]
    fn format_entry_row_empty_active_prefixes_does_not_tag_eol() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("2.6.32".to_string()));
        let entry = cache
            .store("fetch-failed-fallback", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        let row_fallback = format_entry_row(&entry, "kconfig_hash", &[]);
        assert!(!row_fallback.contains("(EOL)"));
        let row_with_active = format_entry_row(&entry, "kconfig_hash", &["6.14".to_string()]);
        assert!(row_with_active.contains("(EOL)"));
    }

    /// Tag-ordering invariant: kconfig-state tag must precede `(EOL)`.
    #[test]
    fn format_entry_row_tags_appear_in_stable_order() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();
        let current_hash = "a1b2c3d4";
        let active_prefixes = ["6.14".to_string()];

        let stale_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("2.6.32".to_string()))
        .with_ktstr_kconfig_hash(Some("deadbeef".to_string()));
        let stale_entry = cache
            .store("stale-eol", &CacheArtifacts::new(&image), &stale_meta)
            .unwrap();
        let stale_row = format_entry_row(&stale_entry, current_hash, &active_prefixes);
        let stale_idx = stale_row
            .find("(stale kconfig)")
            .expect("stale-kconfig tag must appear on dual-tag row");
        let eol_idx = stale_row
            .find("(EOL)")
            .expect("EOL tag must appear on dual-tag row");
        assert!(stale_idx < eol_idx);

        let untracked_meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("2.6.32".to_string()))
        .with_ktstr_kconfig_hash(None);
        let untracked_entry = cache
            .store(
                "untracked-eol",
                &CacheArtifacts::new(&image),
                &untracked_meta,
            )
            .unwrap();
        let untracked_row = format_entry_row(&untracked_entry, current_hash, &active_prefixes);
        let untracked_idx = untracked_row
            .find("(untracked kconfig)")
            .expect("untracked-kconfig tag must appear on dual-tag row");
        let eol_idx = untracked_row
            .find("(EOL)")
            .expect("EOL tag must appear on dual-tag row");
        assert!(untracked_idx < eol_idx);
    }

    /// JSON/human parity: rows where `(EOL)` appears in text must
    /// also produce `eol: true` via entry_is_eol.
    #[test]
    fn kernel_list_eol_json_human_parity() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = tmp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let image = src_dir.join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();

        let make_entry = |key: &str, version: &str| {
            let meta = KernelMetadata::new(
                KernelSource::Tarball,
                "x86_64".to_string(),
                "bzImage".to_string(),
                "2026-04-12T10:00:00Z".to_string(),
            )
            .with_version(Some(version.to_string()));
            cache
                .store(key, &CacheArtifacts::new(&image), &meta)
                .unwrap()
        };

        let cases: &[(&str, &str, &[&str])] = &[
            ("active", "6.14.2", &["6.14"]),
            ("eol", "2.6.32", &["6.14"]),
            ("fetch-fail", "2.6.32", &[]),
        ];
        for (label, version, active) in cases {
            let entry = make_entry(&format!("parity-{label}"), version);
            let active_vec: Vec<String> = active.iter().map(|s| s.to_string()).collect();
            let row = format_entry_row(&entry, "kconfig_hash", &active_vec);
            let json_eol = entry_is_eol(&entry, &active_vec);
            let human_eol = row.contains("(EOL)");
            assert_eq!(
                json_eol, human_eol,
                "JSON/human parity broken for case {label}: \
                 json_eol={json_eol}, human_eol={human_eol}, row={row:?}",
            );
        }
    }

    /// `format_corrupt_footer` is the helper kernel_list emits at
    /// the bottom of the table when at least one cache entry was
    /// surfaced as `ListedEntry::Corrupt`. Pin both the gating
    /// predicate (`any_corrupt` over the entries the caller built)
    /// and the wording invariants the operator depends on:
    /// `(corrupt)` tag, `kernel clean --force` recommendation, the
    /// `ALL` clarifier, the partial-cleanup `--keep N` alternative,
    /// and the cache root's path so the operator knows where to
    /// inspect.
    #[test]
    fn kernel_list_corrupt_footer_fires_iff_any_corrupt() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};
        use crate::cli::kernel_cmd::format_corrupt_footer;

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = tmp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let image = src_dir.join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();

        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-22T00:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()));
        let valid_1 = cache
            .store("valid-entry-a", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        let valid_2 = cache
            .store("valid-entry-b", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        let corrupt_entry = crate::cache::ListedEntry::Corrupt {
            key: "corrupt-entry".to_string(),
            path: cache.root().join("corrupt-entry"),
            reason: "metadata.json missing".to_string(),
        };

        let entries_with_corrupt = [
            crate::cache::ListedEntry::Valid(Box::new(valid_1)),
            corrupt_entry,
        ];
        let entries_clean_only = [crate::cache::ListedEntry::Valid(Box::new(valid_2))];

        fn any_corrupt(entries: &[crate::cache::ListedEntry]) -> bool {
            entries
                .iter()
                .any(|e| matches!(e, crate::cache::ListedEntry::Corrupt { .. }))
        }

        assert!(
            any_corrupt(&entries_with_corrupt),
            "mixed list must trip the footer",
        );
        assert!(
            !any_corrupt(&entries_clean_only),
            "clean-only list must not trip the footer",
        );

        let footer = format_corrupt_footer(cache.root());
        assert!(
            footer.contains("(corrupt)"),
            "footer must reference the tag users see",
        );
        assert!(
            footer.contains("kernel clean --force"),
            "footer must offer a remediation command",
        );
        assert!(
            footer.contains("ALL cached entries"),
            "footer must spell out that `kernel clean --force` is not surgical",
        );
        assert!(
            footer.contains("kernel clean --keep N --force"),
            "footer must offer a partial-cleanup alternative",
        );
        assert!(
            footer.contains(&cache.root().display().to_string()),
            "footer must name the cache root so operators know where to inspect",
        );
    }

    /// JSON/human parity for stale kconfig.
    #[test]
    fn kernel_list_stale_kconfig_json_human_parity() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelSource};
        fn metadata_with_hash(hash: Option<&str>) -> crate::cache::KernelMetadata {
            crate::cache::KernelMetadata::new(
                KernelSource::Tarball,
                "x86_64".to_string(),
                "bzImage".to_string(),
                "2026-04-12T10:00:00Z".to_string(),
            )
            .with_version(Some("6.14.2".to_string()))
            .with_ktstr_kconfig_hash(hash.map(str::to_string))
        }
        let cases: &[(&str, Option<&str>, &str)] = &[
            ("matches", Some("same"), "same"),
            ("stale", Some("old"), "new"),
            ("untracked", None, "anything"),
        ];
        for &(label, entry_hash, current_hash) in cases {
            let tmp = tempfile::TempDir::new().unwrap();
            let cache = CacheDir::with_root(tmp.path().join("cache"));
            let src = tempfile::TempDir::new().unwrap();
            let image = src.path().join("bzImage");
            std::fs::write(&image, b"fake kernel").unwrap();
            let meta = metadata_with_hash(entry_hash);
            let entry = cache
                .store(label, &CacheArtifacts::new(&image), &meta)
                .unwrap();
            let json_stale = entry.kconfig_status(current_hash).is_stale();
            let human_row = format_entry_row(&entry, current_hash, &[]);
            let human_stale = human_row.contains("stale kconfig");
            assert_eq!(
                json_stale, human_stale,
                "kernel_list JSON/human stale-kconfig disagreement on `{label}` \
                 (entry_hash={entry_hash:?}, current_hash={current_hash:?})",
            );
        }
    }

    /// Snapshot pin for `format_entry_row` across the 6-case outcome
    /// matrix over (EOL, not-EOL) × (Matches, Stale, Untracked); empty
    /// and unparseable `active_prefixes` branches are pinned by sibling
    /// `is_eol_` tests. A 7th case fixes the `version == "-"`
    /// short-circuit where a missing version skips the EOL tag even
    /// under a non-empty active list. c8 and c9 pin column-boundary
    /// behavior (exactly 48 chars vs overflow). c10 / c11 pin the
    /// RC-version → series-strip → active-list compare path: c10
    /// catches "suffix left attached to compare key", c11 catches "RC
    /// compare skipped entirely."
    ///
    /// Inline snapshot captures exact padding and tag ordering so any
    /// drift — column width change, tag reorder, `(EOL)` string rename,
    /// Display-impl tweak on `KconfigStatus` — fails this one test.
    #[test]
    fn format_entry_row_renders_eol_kconfig_matrix() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = tmp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let image = src_dir.join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();

        let current_hash = "a1b2c3d4";
        let active_prefixes = ["6.14".to_string()];

        let build_row = |key: &str, version: Option<&str>, entry_hash: Option<&str>| -> String {
            let meta = KernelMetadata::new(
                KernelSource::Tarball,
                "x86_64".to_string(),
                "bzImage".to_string(),
                "2026-04-12T10:00:00Z".to_string(),
            )
            .with_version(version.map(str::to_string))
            .with_ktstr_kconfig_hash(entry_hash.map(str::to_string));
            let entry = cache
                .store(key, &CacheArtifacts::new(&image), &meta)
                .unwrap();
            format_entry_row(&entry, current_hash, &active_prefixes)
        };

        let c8_key = "c8-long-key-exactly-forty-eight-chars-xxxxxxxxxx";
        let c9_key = "c9-key-longer-than-forty-eight-chars-by-twelve-xxxxxxxxxxxx";
        debug_assert_eq!(c8_key.len(), 48);
        debug_assert_eq!(c9_key.len(), 59);
        let rows = [
            build_row("c1-active-matches", Some("6.14.2"), Some(current_hash)),
            build_row("c2-active-stale", Some("6.14.2"), Some("deadbeef")),
            build_row("c3-active-untracked", Some("6.14.2"), None),
            build_row("c4-eol-matches", Some("2.6.32"), Some(current_hash)),
            build_row("c5-eol-stale", Some("2.6.32"), Some("deadbeef")),
            build_row("c6-eol-untracked", Some("2.6.32"), None),
            build_row("c7-active-no-version", None, Some(current_hash)),
            build_row(c8_key, Some("6.14.2"), Some(current_hash)),
            build_row(c9_key, Some("6.14.2"), Some(current_hash)),
            build_row("c10-active-rc", Some("6.14-rc2"), Some(current_hash)),
            build_row("c11-eol-rc", Some("7.0-rc1"), Some(current_hash)),
        ];
        let joined = rows.join("\n");
        insta::assert_snapshot!(joined, @r"
          c1-active-matches                                6.14.2       tarball  x86_64  2026-04-12T10:00:00Z
          c2-active-stale                                  6.14.2       tarball  x86_64  2026-04-12T10:00:00Z (stale kconfig)
          c3-active-untracked                              6.14.2       tarball  x86_64  2026-04-12T10:00:00Z (untracked kconfig)
          c4-eol-matches                                   2.6.32       tarball  x86_64  2026-04-12T10:00:00Z (EOL)
          c5-eol-stale                                     2.6.32       tarball  x86_64  2026-04-12T10:00:00Z (stale kconfig) (EOL)
          c6-eol-untracked                                 2.6.32       tarball  x86_64  2026-04-12T10:00:00Z (untracked kconfig) (EOL)
          c7-active-no-version                             -            tarball  x86_64  2026-04-12T10:00:00Z
          c8-long-key-exactly-forty-eight-chars-xxxxxxxxxx 6.14.2       tarball  x86_64  2026-04-12T10:00:00Z
          c9-key-longer-than-forty-eight-chars-by-twelve-xxxxxxxxxxxx 6.14.2       tarball  x86_64  2026-04-12T10:00:00Z
          c10-active-rc                                    6.14-rc2     tarball  x86_64  2026-04-12T10:00:00Z
          c11-eol-rc                                       7.0-rc1      tarball  x86_64  2026-04-12T10:00:00Z (EOL)
        ");
    }
}
