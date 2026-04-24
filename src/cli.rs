//! CLI support functions shared between `ktstr` and `cargo-ktstr`.
//!
//! Validation, configuration, and kernel/KVM resolution logic used
//! by both binaries.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::Subcommand;

use crate::cache::{CacheDir, CacheEntry, KconfigStatus};
use crate::runner::RunConfig;
use crate::scenario::{Scenario, flags};
use crate::workload::WorkType;

/// Re-export of the internal `vmm::host_topology::CpuCap` type so
/// the `ktstr` and `cargo-ktstr` CLI binaries (which import this
/// module through the `pub mod cli` surface) can resolve
/// `--cpu-cap N` without depending on the `pub(crate)` `vmm`
/// module. Keeping the canonical definition in `vmm::host_topology`
/// (so the `acquire_llc_plan` internal call site consumes its own
/// type without needing `cli`) and re-exporting here — versus
/// inverting the dependency — avoids pulling the CLI module into
/// the VMM internals.
pub use crate::vmm::host_topology::CpuCap;

/// Shared `kernel` subcommand tree used by both `ktstr` and
/// `cargo ktstr`. The two binaries embed this as
/// `ktstr kernel <subcmd>` / `cargo ktstr kernel <subcmd>` and
/// dispatch identically; defining the variants once means a new
/// `kernel` subcommand (or a flag change) lands in both surfaces by
/// construction.
#[derive(Subcommand, Debug)]
pub enum KernelCommand {
    /// List cached kernel images.
    #[command(long_about = KERNEL_LIST_LONG_ABOUT)]
    List {
        /// Output in JSON format for CI scripting.
        #[arg(long)]
        json: bool,
    },
    /// Download, build, and cache a kernel image.
    Build {
        /// Kernel version to download (e.g. 6.14.2, 6.15-rc3). A
        /// major.minor prefix (e.g. 6.12) resolves to the highest
        /// patch release in that series, falling back to probing
        /// cdn.kernel.org for EOL series no longer in releases.json.
        #[arg(conflicts_with_all = ["source", "git"])]
        version: Option<String>,
        /// Path to existing kernel source directory.
        #[arg(long, conflicts_with_all = ["version", "git"])]
        source: Option<PathBuf>,
        /// Git URL to clone kernel source from. Cloned shallow (depth 1)
        /// at the ref supplied via --ref.
        #[arg(long, requires = "git_ref", conflicts_with_all = ["version", "source"])]
        git: Option<String>,
        /// Git ref to checkout (branch, tag, commit). Required with --git.
        #[arg(long = "ref", requires = "git")]
        git_ref: Option<String>,
        /// Rebuild even if a cached image exists.
        #[arg(long)]
        force: bool,
        /// Run `make mrproper` before configuring. Only meaningful
        /// with `--source`: downloaded tarball and freshly cloned
        /// git sources start clean, so this flag prints a notice
        /// and is ignored in those modes.
        #[arg(long)]
        clean: bool,
        #[arg(long, help = CPU_CAP_HELP)]
        cpu_cap: Option<usize>,
    },
    /// Remove cached kernel images.
    Clean {
        /// Keep the N most recent VALID cached kernels. When absent,
        /// removes every valid entry. Corrupt entries are always
        /// candidates for removal regardless of this value — they
        /// waste disk space and serve no build — so a corrupt entry
        /// never consumes a keep slot.
        #[arg(long)]
        keep: Option<usize>,
        /// Skip the y/N confirmation prompt before deleting. Always
        /// required in non-interactive contexts: without `--force`
        /// the command bails on a non-tty stdin rather than hang
        /// waiting for input. In an interactive shell, omit
        /// `--force` to be prompted.
        #[arg(long)]
        force: bool,
        /// Remove only corrupt cache entries (metadata missing or
        /// unparseable, image file absent). Valid entries are left
        /// untouched regardless of `--force`. Useful for clearing
        /// broken entries after an interrupted build without
        /// risking the curated set of good kernels. Mutually
        /// exclusive with `--keep`: `--corrupt-only` never touches
        /// valid entries, so a keep budget would silently be
        /// ignored; rejecting at parse time surfaces the
        /// misunderstanding instead.
        #[arg(long, conflicts_with = "keep")]
        corrupt_only: bool,
    },
}

/// Help text for `--kernel` in contexts that reject raw image files:
/// `cargo ktstr test`, `cargo ktstr coverage`, and `ktstr shell`.
/// Matches `KernelResolvePolicy { accept_raw_image: false, .. }`.
///
/// Raw images are rejected here because these commands depend on a
/// matching `vmlinux` and the cached kconfig fragment alongside the
/// image (test/coverage need BTF, `ktstr shell` reuses the cache
/// entry for kconfig discovery). A bare `bzImage`/`Image` passed
/// directly carries neither, so silently accepting it would produce
/// hard-to-diagnose mid-run failures. The verifier and
/// `cargo ktstr shell` accept raw images because their flows do not
/// need that companion metadata; see [`KERNEL_HELP_RAW_OK`].
pub const KERNEL_HELP_NO_RAW: &str = "Kernel identifier: a source directory \
     path (e.g. `../linux`), a version (`6.14.2`, or major.minor prefix \
     `6.14` for latest patch), or a cache key (see `kernel list`). Raw \
     image files are rejected. Source directories auto-build (can be slow \
     on a fresh tree); versions auto-download from kernel.org on cache \
     miss.";

/// Help text for `--kernel` in contexts that accept raw image files:
/// `cargo ktstr verifier` and `cargo ktstr shell`. Matches
/// `KernelResolvePolicy { accept_raw_image: true, .. }`. See
/// [`KERNEL_HELP_NO_RAW`] for the converse and the rationale for
/// the asymmetry.
pub const KERNEL_HELP_RAW_OK: &str = "Kernel identifier: a source directory \
     path (e.g. `../linux`), a raw image file (`bzImage` / `Image`), a \
     version (`6.14.2`, or major.minor prefix `6.14` for latest patch), \
     or a cache key (see `kernel list`). Source directories auto-build \
     (can be slow on a fresh tree); versions auto-download from kernel.org \
     on cache miss. When absent, resolves via cache then filesystem, \
     falling back to downloading the latest stable kernel.";

/// Help text for the `--cpu-cap N` flag. Shared across `ktstr kernel build`,
/// `cargo ktstr kernel build`, and `ktstr shell` so the operator-facing
/// wording is identical regardless of entry point.
///
/// This flag is the resource-budget contract: the operator promises
/// (and the framework enforces) that the build or no-perf-mode shell
/// VM will stay within N CPUs' worth of reservation and the NUMA
/// nodes hosting them. Setting `--cpu-cap N` flips several internal
/// defaults on this run: the LLC discovery walks whole LLCs in
/// consolidation- and NUMA-aware order until the CPU budget is met;
/// make's `-jN` parallelism matches the plan's CPU count so gcc
/// can't fan out beyond the budget; a cgroup v2 sandbox binds make +
/// gcc's cpuset to the plan's CPUs and `cpuset.mems` to the plan's
/// NUMA nodes, so any degradation is fatal under the flag rather
/// than a silent warning.
pub const CPU_CAP_HELP: &str = "Reserve exactly N host CPUs for the build or \
     no-perf-mode shell. Integer ≥ 1; must be ≤ the calling process's \
     sched_getaffinity cpuset size (the allowed CPU count, NOT the \
     host's total online CPUs — under a cgroup-restricted runner the \
     allowed set is typically smaller). When absent, 30% of the \
     allowed CPUs are reserved (minimum 1). The planner walks whole \
     LLCs in consolidation- and NUMA-aware order, filtered to the \
     allowed cpuset, partial-taking the last LLC so `plan.cpus.len() \
     == N` exactly. The flock set may cover more LLCs than strictly \
     required (flock coordination is per-LLC even when the last LLC \
     is only partially used for the CPU budget). Run `ktstr locks \
     --watch 1s` to observe NUMA placement live. Under --cpu-cap, \
     make's `-jN` parallelism matches the reserved CPU count and the \
     kernel build runs inside a cgroup v2 sandbox that pins gcc/ld \
     to the reserved CPUs + NUMA nodes; if the sandbox cannot be \
     installed (missing cgroup v2, missing cpuset controller, \
     permission denied), the build aborts rather than running \
     without enforcement. Mutually exclusive with \
     KTSTR_BYPASS_LLC_LOCKS=1. On `ktstr shell`, requires \
     --no-perf-mode (perf-mode already holds every LLC exclusively). \
     Also settable via KTSTR_CPU_CAP env var (CLI flag wins when both \
     are present).";

/// Literal text of the `(EOL)` tag explanation. Lives inside a macro
/// (instead of a `pub const`) so that downstream `concat!` callers
/// — specifically [`KERNEL_LIST_LONG_ABOUT`] — can embed the bytes at
/// compile time without duplicating the string. `concat!` requires
/// each argument to be a string literal at expansion, and a macro
/// call that expands to a literal satisfies that requirement while
/// a `&'static str` reference does not. Expansion order: the inner
/// macro is expanded first, `concat!` then sees a literal.
macro_rules! eol_explanation_literal {
    () => {
        "(EOL) marks entries whose major.minor series is absent from \
         kernel.org's current active releases. Suppressed when the \
         active-release list cannot be fetched."
    };
}

/// Explanation of the `(EOL)` tag, shared between the text-output
/// legend printed after `kernel list` and the `kernel list --help`
/// long description (via [`KERNEL_LIST_LONG_ABOUT`], which embeds this
/// exact byte sequence at its head through the shared
/// `eol_explanation_literal!` macro). One literal → one source of
/// truth, so a wording drift cannot put the two surfaces out of
/// sync. `pub` matches the visibility of the sibling
/// `KERNEL_HELP_*` constants so downstream consumers (e.g.
/// documentation generators) can reference the exact text the CLI
/// prints.
pub const EOL_EXPLANATION: &str = eol_explanation_literal!();

/// `long_about` for `kernel list --help`. Embeds [`EOL_EXPLANATION`]
/// verbatim (via `eol_explanation_literal!`) so the tag legend
/// cannot drift between the post-table output and the help copy,
/// then appends a plain-text rendering of the `--json` output
/// schema so scripted consumers can discover the contract from the
/// terminal without running `cargo doc`. The schema wording
/// mirrors the Rust-doc schema on [`kernel_list`]; keeping both
/// surfaces terse makes a drift obvious on review. A plain-text
/// (not JSON/markdown) rendering is used because clap applies no
/// JSON/markdown formatting pass, so the schema reads as plain
/// text. Clap does apply terminal-width wrapping, so the embedded
/// EOL sentence re-flows to the width of the host terminal; the
/// schema block's explicit `\n` line breaks survive wrapping and
/// preserve the column-aligned field table.
pub const KERNEL_LIST_LONG_ABOUT: &str = concat!(
    eol_explanation_literal!(),
    "\n\n",
    "--json emits one JSON object with three top-level fields:\n",
    "\n",
    "  current_ktstr_kconfig_hash   hex digest of the kconfig fragment the\n",
    "                               running binary was built with, for\n",
    "                               stale-entry detection.\n",
    "  active_prefixes_fetch_error  null on success; error string on\n",
    "                               active-series fetch failure. When\n",
    "                               non-null, every entry's `eol` is false\n",
    "                               regardless of actual support status —\n",
    "                               check this field before trusting `eol`.\n",
    "  entries                      array of per-entry objects. Each\n",
    "                               element is either a VALID entry (full\n",
    "                               field set) or a CORRUPT entry (only\n",
    "                               `key`, `path`, `error`). Detect\n",
    "                               corruption by the presence of `error`.\n",
    "\n",
    "Valid entry fields: key, path, version (nullable), source, arch,\n",
    "built_at, ktstr_kconfig_hash (nullable), kconfig_status, eol,\n",
    "config_hash (nullable), image_name, image_path, has_vmlinux,\n",
    "vmlinux_stripped.\n",
    "\n",
    "  path             absolute path to the cache entry DIRECTORY.\n",
    "  image_path       absolute path to the boot image file INSIDE\n",
    "                   that directory. `path` points at the dir, not\n",
    "                   the image — scripts that want the kernel\n",
    "                   artifact to pass to qemu/vm-loaders should\n",
    "                   read `image_path`, not join `path` with a\n",
    "                   hardcoded filename.\n",
    "  kconfig_status   one of \"matches\", \"stale\", \"untracked\"\n",
    "                   (Display form of cache::KconfigStatus).\n",
    "  source           internally-tagged on \"type\":\n",
    "                     {\"type\": \"tarball\"}\n",
    "                     {\"type\": \"git\",   \"git_hash\": ?, \"ref\": ?}\n",
    "                     {\"type\": \"local\", \"source_tree_path\": ?,\n",
    "                                       \"git_hash\": ?}\n",
    "                   Dispatch on \"type\" before reading variant\n",
    "                   fields.\n",
    "  eol              true iff the entry's major.minor series is absent\n",
    "                   from the active-prefix list. Meaningful only when\n",
    "                   active_prefixes_fetch_error is null. Also false\n",
    "                   whenever version is null (the missing-version\n",
    "                   short-circuit in `entry_is_eol`).\n",
    "  has_vmlinux      true iff the uncompressed vmlinux is cached\n",
    "                   alongside the compressed image (required for\n",
    "                   DWARF-driven probes).\n",
    "  vmlinux_stripped true iff the cached vmlinux came from a\n",
    "                   successful strip pass. false marks the\n",
    "                   raw-fallback path — a larger on-disk payload\n",
    "                   indicating the strip pipeline errored on this\n",
    "                   kernel; the entry is still usable but the\n",
    "                   fallback is a signal to investigate. Meaningful\n",
    "                   only when has_vmlinux is true (false otherwise).\n",
    "  config_hash      CRC32 of the final merged .config; distinct\n",
    "                   from ktstr_kconfig_hash which covers only the\n",
    "                   ktstr fragment."
);

/// Emitted by `kernel build` when a local source tree has
/// uncommitted index/worktree changes. Caching would key the built
/// artifact on a git hash that does not describe the actual tree,
/// so the build completes but the result is not archived. The
/// hint names the two remediation paths (commit or stash) so an
/// operator re-running the build after cleaning the tree benefits
/// from the cache. Extracted from the call site so a wording drift
/// between what's printed and what's documented elsewhere is
/// impossible by construction; pinned by
/// `dirty_tree_cache_skip_hint_shape` below.
pub const DIRTY_TREE_CACHE_SKIP_HINT: &str = "skipping cache — working tree has uncommitted changes; \
     commit or stash to enable caching";

/// Hint shown in place of [`DIRTY_TREE_CACHE_SKIP_HINT`] when the
/// source tree is not a git repository at all. `commit` / `stash`
/// are not actionable remediations in that case — the operator's
/// only path to caching is to put the source under git (or use a
/// kernel-source fetch mode that produces a git-tracked tree).
/// Pinned by `non_git_tree_cache_skip_hint_shape` below so a
/// wording drift is caught in unit tests.
pub const NON_GIT_TREE_CACHE_SKIP_HINT: &str = "skipping cache — source tree is not a git repository so dirty \
     state cannot be detected; put the source under git, or replace \
     `--source` with one of the content-keyed fetch modes that does \
     not need dirty-state detection — `kernel build VERSION` \
     (downloads the tarball from kernel.org) or \
     `kernel build --git URL --ref REF` (shallow-clones the given \
     ref) — to enable caching";

/// Decide whether to emit the `(EOL)` legend under the `kernel list`
/// table. Returns `Some(EOL_EXPLANATION)` iff at least one rendered
/// row carried the tag, else `None`. Splitting the conditional out
/// of `kernel_list` lets both branches be pinned in unit tests
/// without capturing stderr.
pub(crate) fn eol_legend_if_any(any_eol: bool) -> Option<&'static str> {
    if any_eol { Some(EOL_EXPLANATION) } else { None }
}

/// Explanation of the `(untracked kconfig)` tag. Consumer-facing
/// wording mirrors `EOL_EXPLANATION`'s "one-const, one-surface"
/// pattern so a doc-drift between the tag word and the legend
/// cannot silently slip. Mirrors [`STALE_KCONFIG_EXPLANATION`] so
/// the kconfig tag pair shares one shape.
///
/// The `(corrupt)` tag is deliberately not in this legend family —
/// its remediation is operational, not informational. See
/// [`format_corrupt_footer`] for the full rationale.
pub const UNTRACKED_KCONFIG_EXPLANATION: &str = "(untracked kconfig) marks entries with no recorded ktstr.kconfig hash \
     (pre-dates kconfig hash tracking). Rebuild with: kernel build --force VERSION";

/// Decide whether to emit the `(untracked kconfig)` legend under the
/// `kernel list` table. Parallels [`eol_legend_if_any`] so both
/// branches are unit-testable without stderr capture.
pub(crate) fn untracked_legend_if_any(any_untracked: bool) -> Option<&'static str> {
    if any_untracked {
        Some(UNTRACKED_KCONFIG_EXPLANATION)
    } else {
        None
    }
}

/// Explanation of the `(stale kconfig)` tag. Mirrors
/// [`UNTRACKED_KCONFIG_EXPLANATION`] so the kconfig tag pair
/// shares one shape — every kconfig-status legend in the
/// informational trio (EOL / UNTRACKED / STALE) is now a const
/// surfaced via a `*_legend_if_any` helper. Verbatim wording
/// preserved from the prior inline `eprintln!` in `kernel_list`
/// so existing operators see no behavioural change.
pub const STALE_KCONFIG_EXPLANATION: &str = "warning: entries marked (stale kconfig) were built against a different ktstr.kconfig. \
     Rebuild with: kernel build --force <entry version>";

/// Decide whether to emit the `(stale kconfig)` legend under the
/// `kernel list` table. Mirrors [`eol_legend_if_any`] and
/// [`untracked_legend_if_any`] so all three informational legends
/// share one shape (boolean in, `Option<&'static str>` out) and
/// every branch is unit-testable without stderr capture.
pub(crate) fn stale_legend_if_any(any_stale: bool) -> Option<&'static str> {
    if any_stale {
        Some(STALE_KCONFIG_EXPLANATION)
    } else {
        None
    }
}

/// Footer emitted by `kernel_list` when at least one entry is
/// corrupt. Pure function of the cache-root path so tests pin the
/// exact same string the production path prints — not a hand-copied
/// duplicate. Extracted alongside [`eol_legend_if_any`] so the
/// three actionable elements (the `(corrupt)` tag label, the
/// `kernel clean` variants, and the cache-root path) are enforced
/// by one source of truth.
///
/// Scope-safe wording: callers inspecting the footer in isolation
/// must not be able to misread `kernel clean --force` as surgical.
/// The text explicitly spells out "ALL cached entries" and
/// surfaces `--corrupt-only --force` (the surgical form that leaves
/// valid entries intact) ahead of the broader `--force` and
/// `--keep N --force` escalation paths, so an operator with valid
/// alongside corrupt entries reaches for the safe option first
/// rather than blowing them all away in a single command.
///
/// Design decision: `(corrupt)` is deliberately NOT promoted to a
/// one-line tag-explanation const in the [`EOL_EXPLANATION`] /
/// [`UNTRACKED_KCONFIG_EXPLANATION`] / [`STALE_KCONFIG_EXPLANATION`]
/// legend family. Two constraints drive the decision:
///
/// 1. **Runtime cache-root path.** The remediation must surface
///    the actual cache-root directory so operators know where to
///    inspect, and a `&'static str` cannot interpolate a runtime
///    value. [`UNTRACKED_KCONFIG_EXPLANATION`] fits on one line
///    precisely because its remediation (`kernel build --force
///    VERSION`) is a literal string with no runtime context;
///    corrupt's is not, and splitting definition from remediation
///    is only a fallback — not a solution — since the runtime
///    path still has to land somewhere adjacent to the tag.
///
/// 2. **Duplication avoidance.** The footer's first sentence
///    already IS the legend — it names the tag, states the
///    unusable meaning, and enumerates the three corruption modes
///    (missing metadata, malformed metadata, missing image). A
///    separate `CORRUPT_EXPLANATION` const would duplicate that
///    content at two surfaces (const + footer), create drift risk
///    as either wording is edited, and pay for nothing: a reader
///    who sees `(corrupt)` in a row and scrolls to the footer
///    already hits the definition in the first line. Test
///    `corrupt_footer_is_self_documenting` pins that invariant.
///
/// Consistency note: the informational trio (EOL / UNTRACKED /
/// STALE) all share the const + `*_legend_if_any` shape;
/// `(corrupt)` is the sole tag whose remediation requires runtime
/// state (the cache-root path), which is why it stays in the
/// footer family rather than joining the informational trio.
///
/// Command ordering inside the footer: `--corrupt-only --force`
/// is listed FIRST because it is the zero-risk surgical option
/// for the common case (a cache with both valid and corrupt
/// entries — leaves valid alone). The broader `--force` (removes
/// ALL) and `--keep N --force` (preserves N newest) variants
/// follow as escalation paths for operators who want to expand
/// scope beyond corrupt entries alone.
pub(crate) fn format_corrupt_footer(cache_root: &Path) -> String {
    format!(
        "warning: entries marked (corrupt) cannot be used — cached metadata is \
         missing, malformed, or references a missing image. Inspect the entry \
         directory under {} to remove it manually, or run \
         `kernel clean --corrupt-only --force` which removes ONLY corrupt \
         entries and leaves valid ones intact. For broader cleanup, \
         `kernel clean --force` removes ALL cached entries (valid and corrupt \
         alike); `kernel clean --keep N --force` preserves the N newest \
         cached entries while removing the rest.",
        cache_root.display(),
    )
}

/// Decide whether to emit the corrupt-entry footer under the
/// `kernel list` table. Mirrors [`eol_legend_if_any`] and
/// [`untracked_legend_if_any`] so the three "tag → footer" gates
/// share one shape (count in, `Option<String>` out) and every
/// branch is unit-testable without stderr capture. The
/// unconditional emission-only-when-tag-rendered invariant is the
/// signal that keeps the normal no-corrupt case noise-free; a
/// regression that unconditionally emitted the footer would show
/// up as a red test on the `corrupt_count == 0` branch here.
///
/// Prepends a `"N corrupt entr{y|ies}. Run `cargo ktstr kernel
/// clean --corrupt-only` to remove.\n"` summary line before the
/// full [`format_corrupt_footer`] body so an operator sees the
/// count and the short remediation FIRST, with the multi-option
/// escalation detail following. The pluralized form ("entry" vs
/// "entries") matches the count, making the line read naturally
/// at both the 1-entry and N>1-entry boundaries.
pub(crate) fn corrupt_footer_if_any(corrupt_count: usize, cache_root: &Path) -> Option<String> {
    if corrupt_count == 0 {
        return None;
    }
    let noun = if corrupt_count == 1 {
        "entry"
    } else {
        "entries"
    };
    let summary = format!(
        "{corrupt_count} corrupt {noun}. \
         Run `cargo ktstr kernel clean --corrupt-only` to remove.",
    );
    let detail = format_corrupt_footer(cache_root);
    Some(format!("{summary}\n{detail}"))
}

/// ktstr.kconfig embedded at compile time.
pub const EMBEDDED_KCONFIG: &str = crate::EMBEDDED_KCONFIG;

/// Compute CRC32 of the embedded ktstr.kconfig fragment.
pub fn embedded_kconfig_hash() -> String {
    crate::kconfig_hash()
}

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
/// [`crate::fetch::fetch_releases`] error on failure (network error,
/// HTTP status, JSON parse failure, missing releases array) so
/// callers can distinguish "fetched and empty" (kernel.org shipped
/// no active series — a violated assumption) from "fetch failed"
/// (transient outage where EOL annotation must degrade, not flip).
///
/// See [`is_eol`]'s empty-slice guard for the recommended fallback pattern.
pub(crate) fn fetch_active_prefixes() -> anyhow::Result<Vec<String>> {
    let releases = crate::fetch::fetch_releases(crate::fetch::shared_client())?;
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
///   entries in input order are retained, every subsequent valid
///   entry is a candidate.
/// - Input order is preserved in the output — `cache.list()` sorts
///   `built_at`-descending, so the retained `keep` prefix is the
///   most recent entries.
fn partition_clean_candidates(
    entries: &[crate::cache::ListedEntry],
    keep: Option<usize>,
    corrupt_only: bool,
) -> Vec<&crate::cache::ListedEntry> {
    let skip = keep.unwrap_or(0);
    let mut valid_kept = 0usize;
    let mut to_remove: Vec<&crate::cache::ListedEntry> = Vec::new();
    for listed in entries {
        match listed {
            crate::cache::ListedEntry::Valid(_) => {
                if corrupt_only {
                    continue;
                }
                if valid_kept < skip {
                    valid_kept += 1;
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

/// Run make in a kernel directory.
pub fn run_make(kernel_dir: &Path, args: &[&str]) -> Result<()> {
    let status = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .status()?;
    anyhow::ensure!(status.success(), "make {} failed", args.join(" "));
    Ok(())
}

/// Ensure the kconfig fragment is applied to the kernel's .config.
///
/// Creates a default .config via `make defconfig` if none exists.
/// Pure check used by [`configure_kernel`]: every non-empty line of
/// `fragment` (including disable directives like
/// `# CONFIG_X is not set`) must appear as an exact line of `config`.
///
/// Exact-line matching avoids the prefix-aliasing hazard of the prior
/// `config.contains(fragment_line)` formulation, where a fragment line
/// false-matches when it appears as a substring of an unrelated
/// `.config` line — e.g. fragment `CONFIG_NR_CPUS=1` appearing inside
/// `CONFIG_NR_CPUS=128`, or any numeric-tail option where the
/// requested value is a prefix of the existing value.
///
/// `# CONFIG_X is not set` comments ARE kconfig semantics (the
/// canonical way to disable an option), so they participate in the
/// check; the only lines skipped are genuinely empty ones.
fn all_fragment_lines_present(fragment: &str, config: &str) -> bool {
    let existing: std::collections::HashSet<&str> = config.lines().map(str::trim).collect();
    fragment
        .lines()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .all(|t| existing.contains(t))
}

/// Checks each non-empty line of the fragment against the current
/// `.config` via [`all_fragment_lines_present`]. If every fragment
/// line already appears in `.config`, the file is not touched
/// (preserving mtime for make's dependency tracking). If any are
/// missing, appends the full fragment and runs `make olddefconfig`
/// to resolve new options with defaults — without this, the
/// subsequent `make` launches interactive `conf` prompts that hang
/// when stdout/stderr are piped.
pub fn configure_kernel(kernel_dir: &Path, fragment: &str) -> Result<()> {
    let config_path = kernel_dir.join(".config");
    if !config_path.exists() {
        run_make(kernel_dir, &["defconfig"])?;
    }

    let config_content = std::fs::read_to_string(&config_path)?;
    if all_fragment_lines_present(fragment, &config_content) {
        return Ok(());
    }

    let mut config = std::fs::OpenOptions::new()
        .append(true)
        .open(&config_path)?;
    std::io::Write::write_all(&mut config, fragment.as_bytes())?;

    run_make(kernel_dir, &["olddefconfig"])?;

    Ok(())
}

/// Drain a reader into a `Vec<String>`, one entry per newline-delimited
/// chunk, with a final partial chunk (no trailing newline) emitted
/// with the same lossy-UTF-8 conversion. Byte-oriented so non-UTF-8
/// input survives via `from_utf8_lossy` (U+FFFD replacement) instead
/// of being dropped at the line boundary. Strips the trailing `\n`
/// and an optional preceding `\r` so CRLF input matches LF semantics.
/// Calls `on_line` for each line before appending to the returned
/// `Vec`.
///
/// Returned entries and the `on_line` argument never carry their
/// terminating `\n` (or `\r\n`) — the strip runs before emission, so
/// callers that re-emit with `println!` get clean single-newline
/// formatting and callers that persist the strings do not double-
/// count line terminators. Interior `\r` bytes (lone CR not paired
/// with a trailing LF) pass through verbatim, matching the unit
/// coverage in `drain_lines_lossy_lone_cr_at_eof_is_preserved` and
/// `drain_lines_lossy_interior_cr_is_preserved`.
///
/// Extracted from [`run_make_with_output`] so the read logic is
/// testable with in-memory readers (the caller still owns child
/// kill+wait).
fn drain_lines_lossy(
    mut reader: impl BufRead,
    mut on_line: impl FnMut(&str),
) -> std::io::Result<Vec<String>> {
    let mut captured = Vec::new();
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        let mut slice: &[u8] = &buf;
        if let Some(rest) = slice.strip_suffix(b"\n") {
            slice = rest;
            if let Some(rest) = slice.strip_suffix(b"\r") {
                slice = rest;
            }
        }
        let line = String::from_utf8_lossy(slice).into_owned();
        on_line(&line);
        captured.push(line);
    }
    Ok(captured)
}

/// Run make with merged stdout+stderr piped through a spinner.
///
/// Creates a single pipe via `nix::unistd::pipe2(O_CLOEXEC)`, hands
/// the write end to the child's stdout AND stderr (a clone), and
/// reads from the read end. `O_CLOEXEC` prevents the raw pipe fds
/// from leaking into any concurrently-spawned children on other
/// threads — without the flag, a race between `pipe()` and the
/// `Stdio::from()` consumption could let an unrelated `fork+exec`
/// inherit the write end and hold the reader open indefinitely.
/// One pipe, one reader — no threads, no channel, no chance of a
/// deadlock where reading stdout blocks while stderr fills its
/// buffer. Same merged-stream semantics that `sh -c "make … 2>&1"`
/// gives, without the shell-out.
///
/// When a spinner is active, each line is printed via `println()`
/// so the spinner redraws below the output. When no spinner,
/// output is captured and shown only on failure.
///
/// Pipe-read I/O errors propagate via `Err` rather than silently
/// ending the read loop. The prior line-iterator formulation
/// (`.lines()` + `Result::ok`) dropped every error-tagged item —
/// a mid-stream read failure just looked like EOF and the child's
/// tail output disappeared without a diagnostic. The byte-oriented
/// [`drain_lines_lossy`] now surfaces such failures with `anyhow`
/// context naming the merged-stream read, so a broken-pipe or EIO
/// during make's output is caught at the call site.
///
/// Lines observed by `spinner.println()` and retained in the
/// on-failure replay buffer are LF-normalized: `drain_lines_lossy`
/// strips the trailing `\n`, and a preceding `\r` (the CRLF form
/// Make emits on some toolchain + terminal combinations) is
/// stripped too, so every line the caller sees is LF-only and
/// terminator-less. Interior lone `\r` bytes — e.g. a progress
/// bar using carriage-return redraw — pass through verbatim (see
/// `drain_lines_lossy_interior_cr_is_preserved`), which keeps
/// the on-failure replay readable without mangling tools that
/// legitimately use `\r` mid-line.
pub fn run_make_with_output(
    kernel_dir: &Path,
    args: &[&str],
    spinner: Option<&Spinner>,
) -> Result<()> {
    let (read_fd, write_fd) = nix::unistd::pipe2(nix::fcntl::OFlag::O_CLOEXEC)
        .context("create pipe for merged make stdout+stderr")?;
    let write_fd_err = write_fd
        .try_clone()
        .context("clone pipe write end for stderr")?;

    let mut child = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .stdout(std::process::Stdio::from(write_fd))
        .stderr(std::process::Stdio::from(write_fd_err))
        .spawn()
        .with_context(|| format!("spawn make {}", args.join(" ")))?;

    // Parent has no remaining writer handles. `Stdio::from(OwnedFd)`
    // consumed `write_fd` and `write_fd_err` into the Command
    // builder; during `.spawn()` the builder installs them as the
    // child's stdout/stderr via `dup2`, then drops its own OwnedFd
    // copies. The child therefore holds the only live write ends
    // (its dup2'd stdout/stderr, fd 1/2). When `make` exits, those
    // fds are closed and the reader here sees EOF naturally.
    //
    // Read as bytes and convert each line via `from_utf8_lossy` at
    // the boundary. Compiler output can include non-UTF-8 bytes —
    // source paths on exotic filesystems, embedded binary fragments
    // from diagnostic tools, locale-encoded text — and a pure-String
    // reader would drop those lines via the `Result::ok` filter,
    // hiding real compiler errors in CI logs. Lossy conversion keeps
    // every line visible with U+FFFD where the bytes were not valid
    // UTF-8.
    let reader = std::io::BufReader::new(std::fs::File::from(read_fd));
    let captured = match drain_lines_lossy(reader, |line| {
        if let Some(sp) = spinner {
            sp.println(line);
        }
    }) {
        Ok(v) => v,
        Err(e) => {
            // On pipe-read I/O failure, kill and reap the child
            // before propagating so `make` doesn't linger as a
            // zombie — stdlib's Child does not auto-wait on drop.
            // Both ops use `.ok()` because the read-side error is
            // the actionable diagnostic; a secondary wait/kill
            // failure should not mask it.
            child.kill().ok();
            child.wait().ok();
            return Err(e).context("read merged make stdout+stderr");
        }
    };

    let status = child.wait()?;
    if !status.success() {
        // Always show captured output on failure so CI logs contain
        // the actual compiler errors, not just "make failed".
        for line in &captured {
            eprintln!("{line}");
        }
        bail!("make {} failed", args.join(" "));
    }
    Ok(())
}

/// Build the kernel with output piped through a spinner.
///
/// `jobs_override` supplies the `-jN` count when set (used by
/// `kernel_build_pipeline` under `--cpu-cap` to keep gcc's
/// parallelism aligned with the reserved CPU count). `None`
/// falls back to `std::thread::available_parallelism`.
pub fn make_kernel_with_output(
    kernel_dir: &Path,
    spinner: Option<&Spinner>,
    jobs_override: Option<usize>,
) -> Result<()> {
    let nproc = jobs_override.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let args = build_make_args(nproc);
    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_make_with_output(kernel_dir, &arg_refs, spinner)
}

/// Resolve flag names, erroring on unknown flags.
pub fn resolve_flags(flag_arg: Option<Vec<String>>) -> Result<Option<Vec<&'static str>>> {
    match flag_arg {
        Some(fs) => {
            let mut resolved = Vec::new();
            for f in &fs {
                match flags::from_short_name(f) {
                    Some(name) => resolved.push(name),
                    None => bail!(
                        "unknown flag: '{f}'. valid flags: {}",
                        flags::ALL.join(", "),
                    ),
                }
            }
            Ok(Some(resolved))
        }
        None => Ok(None),
    }
}

/// Parse and validate a work type name.
pub fn parse_work_type(name: Option<&str>) -> Result<Option<WorkType>> {
    match name {
        Some(name) => match WorkType::from_name(name) {
            Some(wt) => Ok(Some(wt)),
            None => bail!(
                "unknown work type: '{name}'. valid types: {}",
                WorkType::ALL_NAMES.join(", "),
            ),
        },
        None => Ok(None),
    }
}

/// Parse a comma-separated topology string into its four dimensions:
/// `(numa_nodes, llcs, cores, threads)`. The canonical format is
/// `"numa_nodes,llcs,cores,threads"` — the same shape accepted by the
/// `ktstr shell --topology` and `cargo ktstr shell --topology` flags.
///
/// Validation:
/// - Exactly four comma-separated components are required.
/// - Each component must parse as `u32`. A parse failure names the
///   failing field explicitly (e.g. `"invalid llcs value: 'abc'"`)
///   so the user can see which dimension they mistyped without
///   counting commas.
/// - Every dimension must be at least 1 — a zero in any position
///   produces an unusable VM topology, so we reject it up front.
///
/// Consolidating the parse + validate in one helper eliminates the
/// identical 4-arm `parts[i].parse().map_err(...)` block that the two
/// binary entry points (`src/bin/ktstr.rs` Command::Shell and
/// `src/bin/cargo-ktstr.rs` `run_shell`) would otherwise drift on.
/// Error shape is `anyhow::Error`; callers that need a `String` (like
/// cargo-ktstr's `Result<(), String>` surface) bridge via
/// `.map_err(|e| format!("{e:#}"))` at the call site.
pub fn parse_topology_string(topology: &str) -> Result<(u32, u32, u32, u32)> {
    let parts: Vec<&str> = topology.split(',').collect();
    if parts.len() != 4 {
        bail!(
            "invalid topology '{topology}': expected 'numa_nodes,llcs,cores,threads' \
             (e.g. '1,2,4,1')"
        );
    }
    // Stable field order mirrors the 4-tuple return so a future
    // field-rename lands consistently in one place.
    let fields: [(&str, &str); 4] = [
        ("numa_nodes", parts[0]),
        ("llcs", parts[1]),
        ("cores", parts[2]),
        ("threads", parts[3]),
    ];
    let mut vals: [u32; 4] = [0; 4];
    for (i, (name, raw)) in fields.iter().enumerate() {
        vals[i] = raw
            .parse::<u32>()
            .map_err(|_| anyhow::anyhow!("invalid {name} value: '{raw}'"))?;
    }
    let [numa_nodes, llcs, cores, threads] = vals;
    if numa_nodes == 0 || llcs == 0 || cores == 0 || threads == 0 {
        bail!("invalid topology '{topology}': all values must be >= 1");
    }
    Ok((numa_nodes, llcs, cores, threads))
}

/// Filter scenarios by name substring.
///
/// When a non-None `filter` matches zero scenarios, the bail
/// message appends a `Did you mean \`<name>\`?` hint sourced from
/// [`suggest_closest_scenario_name`] — strsim Levenshtein match
/// against the registry — so an operator typo surfaces a concrete
/// correction candidate instead of the generic "run 'ktstr list'"
/// redirect.
pub fn filter_scenarios<'a>(
    scenarios: &'a [Scenario],
    filter: Option<&str>,
) -> Result<Vec<&'a Scenario>> {
    let refs: Vec<&Scenario> = scenarios
        .iter()
        .filter(|s| filter.is_none_or(|f| s.name.contains(f)))
        .collect();
    if refs.is_empty() {
        let hint = filter
            .and_then(suggest_closest_scenario_name)
            .map(|s| format!(" Did you mean `{s}`?"))
            .unwrap_or_default();
        bail!("no scenarios matched filter.{hint} Run 'ktstr list' to see available scenarios.",);
    }
    Ok(refs)
}

/// Build a RunConfig from parsed CLI arguments.
#[allow(clippy::too_many_arguments)]
pub fn build_run_config(
    parent_cgroup: String,
    duration: u64,
    workers: usize,
    active_flags: Option<Vec<&'static str>>,
    repro: bool,
    probe_stack: Option<String>,
    auto_repro: bool,
    kernel_dir: Option<String>,
    work_type_override: Option<WorkType>,
) -> RunConfig {
    RunConfig {
        parent_cgroup,
        duration: Duration::from_secs(duration),
        workers_per_cgroup: workers,
        active_flags,
        repro,
        probe_stack,
        auto_repro,
        kernel_dir,
        work_type_override,
        ..Default::default()
    }
}

/// Check if a kernel .config contains CONFIG_SCHED_CLASS_EXT=y.
pub fn has_sched_ext(kernel_dir: &std::path::Path) -> bool {
    let config = kernel_dir.join(".config");
    std::fs::read_to_string(config)
        .map(|s| s.lines().any(|l| l == "CONFIG_SCHED_CLASS_EXT=y"))
        .unwrap_or(false)
}

/// Validate the output .config for critical options that the kconfig
/// fragment requested but the kernel build system may have silently
/// disabled (e.g. CONFIG_DEBUG_INFO_BTF requires pahole).
///
/// Call after `make` succeeds. Returns `Err` with a diagnostic
/// message listing missing options and likely causes.
/// Critical `.config` options checked by [`validate_kernel_config`].
///
/// Each entry pairs a `CONFIG_X` name with a diagnostic hint —
/// human-readable context on the dependency that typically causes the
/// option to be silently dropped during `make`. The list is curated:
/// every entry here is an option whose absence at the post-build
/// check has historically surfaced as a specific tool-install or
/// arch-default-override. The companion test
/// `critical_options_are_in_embedded_kconfig` proves every name is
/// present in [`EMBEDDED_KCONFIG`] as `=y`, so a kconfig edit that
/// removes a critical entry fails the test immediately instead of
/// surfacing later as a build that passes validation but behaves
/// differently.
const VALIDATE_CONFIG_CRITICAL: &[(&str, &str)] = &[
    (
        "CONFIG_SCHED_CLASS_EXT",
        "depends on CONFIG_DEBUG_INFO_BTF — ensure pahole >= 1.16 is installed (dwarves package)",
    ),
    (
        "CONFIG_DEBUG_INFO_BTF",
        "requires pahole >= 1.16 (dwarves package)",
    ),
    ("CONFIG_BPF_SYSCALL", "required for BPF program loading"),
    (
        "CONFIG_FTRACE",
        "gate for all tracing infrastructure — arm64 defconfig disables it, \
         silently dropping KPROBE_EVENTS and BPF_EVENTS",
    ),
    (
        "CONFIG_KPROBE_EVENTS",
        "required for ktstr probe pipeline (depends on FTRACE + KPROBES)",
    ),
    (
        "CONFIG_BPF_EVENTS",
        "required for BPF kprobe/tracepoint attachment (depends on KPROBE_EVENTS + PERF_EVENTS)",
    ),
];

pub fn validate_kernel_config(kernel_dir: &std::path::Path) -> Result<()> {
    let config_path = kernel_dir.join(".config");
    let config = std::fs::read_to_string(&config_path)
        .with_context(|| format!("read {}", config_path.display()))?;

    let mut missing = Vec::new();
    for &(option, hint) in VALIDATE_CONFIG_CRITICAL {
        let enabled = format!("{option}=y");
        if !config.lines().any(|l| l == enabled) {
            missing.push((option, hint));
        }
    }

    if !missing.is_empty() {
        let mut msg =
            String::from("kernel build completed but critical config options are missing:\n");
        for (option, hint) in &missing {
            msg.push_str(&format!("  {option} not set — {hint}\n"));
        }
        msg.push_str(
            "\nThe kernel build system silently disables options whose dependencies \
             are not met. Install missing tools and rebuild with --force.",
        );
        bail!("{msg}");
    }
    Ok(())
}

/// Result of the post-acquisition kernel build pipeline.
///
/// Returned by [`kernel_build_pipeline`] so callers can inspect
/// the cache entry and built image path.
#[non_exhaustive]
pub struct KernelBuildResult {
    /// Cache entry, if the build was cached. `None` for dirty trees
    /// or when cache store fails.
    pub entry: Option<crate::cache::CacheEntry>,
    /// Path to the built kernel image.
    pub image_path: std::path::PathBuf,
}

/// Two-phase build reservation handles (LLC flock plan + cgroup v2
/// sandbox + make -jN hint). Consumed by
/// [`kernel_build_pipeline`]; the factored-out
/// [`acquire_build_reservation`] builds it from `cpu_cap` without
/// depending on kernel source, enabling integration tests that
/// exercise the reservation logic against synthetic topologies.
///
/// Drop order is load-bearing: `_sandbox` drops BEFORE `plan`
/// because struct fields drop in declaration order and
/// `_sandbox` is declared after `plan` (swapped relative to the
/// prior inline let-bindings in `kernel_build_pipeline` — the
/// inline form relied on LIFO binding-drop order, so the LATER
/// binding dropped FIRST; the struct form relies on IN-ORDER
/// field-drop, so the LATER field also drops FIRST. Same
/// outcome, different mechanism). The sandbox's cgroup rmdir
/// must run while the LLC flocks are still held; otherwise a
/// peer could observe the LLC released before the cgroup is
/// gone and mint a conflicting plan.
#[derive(Debug)]
pub(crate) struct BuildReservation {
    /// cgroup v2 sandbox. `None` when `plan` is `None` (no reservation
    /// to enforce). Drops FIRST per struct field order — cgroup
    /// rmdir runs while LLC flocks are still held. `_` prefix
    /// keeps the binding alive through Drop but marks it as
    /// not-read — the RAII invariant IS the read.
    pub(crate) _sandbox: Option<crate::vmm::cgroup_sandbox::BuildSandbox>,
    /// LLC plan (flock fds + cpus + mems). `None` under
    /// `KTSTR_BYPASS_LLC_LOCKS=1` or sysfs-unreadable host without
    /// `--cpu-cap`. Drops SECOND per struct field order —
    /// flocks release AFTER the sandbox rmdir lands.
    pub(crate) plan: Option<crate::vmm::host_topology::LlcPlan>,
    /// `make -jN` parallelism hint. `Some(N)` under an active
    /// `plan`; `None` when no reservation exists (caller falls
    /// back to `nproc`).
    pub(crate) make_jobs: Option<usize>,
}

/// Acquire the two-phase reservation (LLC flocks + cgroup sandbox)
/// for a kernel build. Factored out of [`kernel_build_pipeline`]
/// so integration tests can exercise the cpu_cap → acquire →
/// sandbox → make_jobs decision tree without requiring a real
/// kernel source tree.
///
/// Returns a `BuildReservation` whose fields are the three values
/// `kernel_build_pipeline` used to bind inline; Drop order
/// matches the prior inline let-bindings so the LIFO cgroup-
/// rmdir-before-LLC-unlock invariant is preserved.
///
/// `cli_label` prefixes operator-facing error text.
///
/// `cpu_cap` is the resolved CPU-count cap from
/// [`CpuCap::resolve`](crate::vmm::host_topology::CpuCap::resolve);
/// `None` means "reserve 30% of the calling process's allowed-CPU
/// set", applied inside the planner at acquire time.
pub(crate) fn acquire_build_reservation(
    cli_label: &str,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<BuildReservation> {
    let bypass = std::env::var("KTSTR_BYPASS_LLC_LOCKS")
        .ok()
        .is_some_and(|v| !v.is_empty());
    // INVARIANT: `plan` is the first field of BuildReservation but
    // Drop runs fields in declaration order — we therefore list
    // `plan` BEFORE `_sandbox` in the struct def and rely on the
    // LIFO Drop-on-the-struct to drop `_sandbox` first. This
    // mirrors the original inline let-bindings (plan declared
    // first, sandbox after) — reordering either would either
    // (a) unlock LLCs while the sandbox still enforces the
    // cpuset — a concurrent peer could claim the LLC and stomp
    // gcc children that haven't exited — or (b) leave the cgroup
    // hierarchy non-empty when its parent tries to rmdir.
    let plan: Option<crate::vmm::host_topology::LlcPlan> = if bypass {
        if cpu_cap.is_some() {
            anyhow::bail!(
                "{cli_label}: --cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; \
                 unset one of them. --cpu-cap is a resource contract; bypass \
                 disables the contract entirely."
            );
        }
        None
    } else if let Ok(host_topo) = crate::vmm::host_topology::HostTopology::from_sysfs() {
        let test_topo = crate::topology::TestTopology::from_system()?;
        let acquired_plan =
            crate::vmm::host_topology::acquire_llc_plan(&host_topo, &test_topo, cpu_cap)?;
        crate::vmm::host_topology::warn_if_cross_node_spill(&acquired_plan, &host_topo);
        Some(acquired_plan)
    } else {
        if cpu_cap.is_some() {
            anyhow::bail!(
                "{cli_label}: --cpu-cap set but host LLC topology unreadable \
                 from sysfs — cannot enforce the resource budget. Run on a \
                 host with /sys/devices/system/cpu populated, or drop \
                 --cpu-cap to build without enforcement."
            );
        }
        tracing::warn!(
            "{cli_label}: could not read host LLC topology from sysfs; \
             skipping kernel-build LLC reservation. Concurrent perf-mode \
             runs on this host will NOT be serialized against this build"
        );
        None
    };

    // Phase 2: cgroup v2 sandbox that enforces cpu+mem binding on
    // make/gcc children. `hard_error_on_degrade` is driven by
    // whether `--cpu-cap` was set explicitly: degradation is fatal
    // under the flag (the flag promises enforcement), and warn-only
    // when the 30%-of-allowed default was expanded (the default
    // contract is best-effort — a parent cgroup narrowing the
    // reservation should not fail the build).
    let sandbox: Option<crate::vmm::cgroup_sandbox::BuildSandbox> = match plan.as_ref() {
        Some(p) => Some(crate::vmm::cgroup_sandbox::BuildSandbox::try_create(
            &p.cpus,
            &p.mems,
            cpu_cap.is_some(),
        )?),
        None => None,
    };

    // `make -jN` parallelism hint. `N` = `plan.cpus.len()` via
    // `make_jobs_for_plan` — the reserved CPU count, whether that
    // came from an explicit `--cpu-cap N` or the 30%-of-allowed
    // default. See `make_kernel_with_output` for the resolution.
    let make_jobs = plan
        .as_ref()
        .map(crate::vmm::host_topology::make_jobs_for_plan);

    Ok(BuildReservation {
        plan,
        _sandbox: sandbox,
        make_jobs,
    })
}

/// Post-acquisition kernel build pipeline.
///
/// Handles: clean, configure, build, validate config, generate
/// compile_commands.json for local trees, find image, strip vmlinux,
/// compute metadata, cache store, and remote cache store (when
/// enabled). Callers handle source acquisition.
///
/// `cli_label` prefixes diagnostic status output (e.g. `"ktstr"` or
/// `"cargo ktstr"`).
///
/// `is_local_source` should be true when the user passed `--source`.
/// It controls the mrproper warning and `source_tree_path` in metadata.
pub fn kernel_build_pipeline(
    acquired: &crate::fetch::AcquiredSource,
    cache: &crate::cache::CacheDir,
    cli_label: &str,
    clean: bool,
    is_local_source: bool,
    cpu_cap: Option<crate::vmm::host_topology::CpuCap>,
) -> Result<KernelBuildResult> {
    let source_dir = &acquired.source_dir;
    let (arch, image_name) = crate::fetch::arch_info();

    // Two-phase reservation. A concurrent perf-mode test run must
    // not have its measured CPUs stomped by a `make -j$(nproc)`
    // explosion of gcc children, and vice-versa a concurrent
    // kernel build must not have its compile window extended by
    // a test pinning RT-FIFO on shared cores. Phase 1 of the
    // reservation is the LLC-level flock from
    // [`acquire_llc_plan`]: whole-LLC flocks whose count is
    // chosen to cover the CPU budget (either an explicit
    // `--cpu-cap N` or the 30%-of-allowed default). Phase 2 is
    // the cgroup v2 sandbox from
    // [`BuildSandbox::try_create`] that binds make/gcc's
    // cpu+mem sets to the plan's CPUs + NUMA nodes so the
    // parallelism hint is enforced, not just advisory.
    //
    // Binding order is load-bearing: `plan` is declared BEFORE
    // `_sandbox` so the sandbox's Drop runs FIRST (LIFO), which
    // migrates the build pid out of the cgroup and rmdirs the
    // child while the LLC flocks are still held. Otherwise a peer
    // could observe the LLC released before the cgroup is gone,
    // mint a new plan against the same LLCs, and see an orphan
    // cgroup lingering for up to the 24h sweep window.
    //
    // Escape hatches:
    //   - `KTSTR_BYPASS_LLC_LOCKS=1`: skip the LLC plan+flock
    //     acquisition entirely; the build proceeds immediately
    //     without coordinating with any concurrent perf-mode run.
    //     Use when the operator explicitly accepts measurement
    //     noise (one shell doing unrelated work, an isolated
    //     developer workstation, or a CI queue that already
    //     serializes jobs at a higher layer). Mutually exclusive
    //     with `--cpu-cap` at CLI parse time — see the CLI
    //     binaries' pre-dispatch conflict check.
    //   - Sysfs-unreadable host (non-Linux, degraded container):
    //     `HostTopology::from_sysfs()` returns `Err`. Without
    //     `--cpu-cap`, we emit a `tracing::warn!` and proceed
    //     without locks. With `--cpu-cap`, the flag cannot be
    //     honoured and we fail hard — cpu_cap is a contract, not
    //     a hint: a silent degrade would let a build exceed the
    //     declared resource budget without surfacing.
    // `_plan` + `_sandbox` are kept alive via RAII — their Drops
    // release the LLC flocks and cgroup on scope exit. Struct
    // field order in BuildReservation ensures `_sandbox` drops
    // BEFORE `plan`, matching the inline LIFO invariant.
    let BuildReservation {
        plan: _plan,
        _sandbox,
        make_jobs,
    } = acquire_build_reservation(cli_label, cpu_cap)?;

    if clean {
        if !is_local_source {
            eprintln!(
                "{cli_label}: --clean is only meaningful with --source (downloaded sources start clean)"
            );
        } else {
            eprintln!("{cli_label}: make mrproper");
            run_make(source_dir, &["mrproper"])?;
        }
    }

    if !has_sched_ext(source_dir) {
        Spinner::with_progress("Configuring kernel...", "Kernel configured", |_| {
            configure_kernel(source_dir, EMBEDDED_KCONFIG)
        })?;
    }

    Spinner::with_progress("Building kernel...", "Kernel built", |sp| {
        make_kernel_with_output(source_dir, Some(sp), make_jobs)
    })?;

    // Validate critical config options were not silently disabled.
    validate_kernel_config(source_dir)?;

    // Generate compile_commands.json for local trees (LSP support).
    if !acquired.is_temp {
        Spinner::with_progress(
            "Generating compile_commands.json...",
            "compile_commands.json generated",
            |sp| run_make_with_output(source_dir, &["compile_commands.json"], Some(sp)),
        )?;
    }

    // Find the built kernel image and vmlinux.
    let image_path = crate::kernel_path::find_image_in_dir(source_dir)
        .ok_or_else(|| anyhow::anyhow!("no kernel image found in {}", source_dir.display()))?;
    let vmlinux_path = source_dir.join("vmlinux");
    let vmlinux_ref = if vmlinux_path.exists() {
        let orig_mb = std::fs::metadata(&vmlinux_path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0);
        eprintln!("{cli_label}: caching vmlinux ({orig_mb:.0} MB, will be stripped)");
        Some(vmlinux_path.as_path())
    } else {
        eprintln!("{cli_label}: warning: vmlinux not found, BTF will not be cached");
        None
    };

    // Cache (skip for dirty local trees).
    if acquired.is_dirty {
        eprintln!("{cli_label}: kernel built at {}", image_path.display());
        // Branch the hint wording: commit/stash is only an actionable
        // remediation for an actual git repo. A non-git source tree
        // is force-marked dirty (see `acquire_local_source` in
        // `fetch.rs`) because dirty detection is impossible, and
        // telling the operator to "commit or stash" leads nowhere.
        let hint = if acquired.is_git {
            DIRTY_TREE_CACHE_SKIP_HINT
        } else {
            NON_GIT_TREE_CACHE_SKIP_HINT
        };
        eprintln!("{cli_label}: {hint}");
        return Ok(KernelBuildResult {
            entry: None,
            image_path,
        });
    }

    let config_path = source_dir.join(".config");
    let config_hash = if config_path.exists() {
        let data = std::fs::read(&config_path)?;
        Some(format!("{:08x}", crc32fast::hash(&data)))
    } else {
        None
    };

    let kconfig_hash = embedded_kconfig_hash();

    let metadata = crate::cache::KernelMetadata::new(
        acquired.kernel_source.clone(),
        arch.to_string(),
        image_name.to_string(),
        crate::test_support::now_iso8601(),
    )
    .with_version(acquired.version.clone())
    .with_config_hash(config_hash)
    .with_ktstr_kconfig_hash(Some(kconfig_hash));

    let mut artifacts = crate::cache::CacheArtifacts::new(&image_path);
    if let Some(v) = vmlinux_ref {
        artifacts = artifacts.with_vmlinux(v);
    }
    let entry = match cache.store(&acquired.cache_key, &artifacts, &metadata) {
        Ok(entry) => {
            success(&format!("\u{2713} Kernel cached: {}", acquired.cache_key));
            eprintln!("{cli_label}: image: {}", entry.image_path().display());
            if crate::remote_cache::is_enabled() {
                crate::remote_cache::remote_store(&entry, cli_label);
            }
            Some(entry)
        }
        Err(e) => {
            warn(&format!("{cli_label}: cache store failed: {e:#}"));
            None
        }
    };

    Ok(KernelBuildResult { entry, image_path })
}

/// Build the make arguments for a kernel build.
///
/// Returns the argument list that would be passed to `make` for a
/// parallel kernel build: `["-jN", "KCFLAGS=-Wno-error"]`.
fn build_make_args(nproc: usize) -> Vec<String> {
    vec![format!("-j{nproc}"), "KCFLAGS=-Wno-error".into()]
}

/// Read sidecar JSON files and return the gauntlet analysis report.
///
/// Source directory:
/// - `KTSTR_SIDECAR_DIR` if set, else
/// - the most recently modified subdirectory under
///   `{CARGO_TARGET_DIR or "target"}/ktstr/`.
///
/// `cargo ktstr stats` doesn't itself run a kernel, so it can't
/// reconstruct the `{kernel}-{timestamp}` key the test process used; the
/// mtime fallback mirrors "show me the report from my last test run."
///
/// Returns `None` with a warning on stderr when no sidecars are found.
/// This is not an error -- regular test runs that skip gauntlet tests
/// produce no sidecar files.
pub fn print_stats_report() -> Option<String> {
    let dir = match std::env::var("KTSTR_SIDECAR_DIR") {
        Ok(d) if !d.is_empty() => Some(std::path::PathBuf::from(d)),
        _ => crate::test_support::newest_run_dir(),
    };
    let report = dir
        .as_deref()
        .map(|d| crate::test_support::analyze_sidecars(Some(d)))
        .filter(|r| !r.is_empty());
    if report.is_none() {
        eprintln!("cargo ktstr: no sidecar data found (skipped)");
    }
    report
}

/// List test runs under `{CARGO_TARGET_DIR or "target"}/ktstr/`.
pub fn list_runs() -> Result<()> {
    crate::stats::list_runs()
}

/// Render the metric registry for `cargo ktstr stats list-metrics`.
///
/// Thin wrapper over [`crate::stats::list_metrics`] — exposed through
/// `cli::` to match the `list_runs` / `compare_runs` / `show_host`
/// convention where every stats-subcommand dispatch arm lands on a
/// `cli::*` helper before reaching the private `stats` module. The
/// returned `String` is printed verbatim by the dispatch site.
pub fn list_metrics(json: bool) -> Result<String> {
    crate::stats::list_metrics(json)
}

/// Compare two test runs and report regressions.
pub fn compare_runs(
    a: &str,
    b: &str,
    filter: Option<&str>,
    policy: &ComparisonPolicy,
    dir: Option<&Path>,
) -> Result<i32> {
    crate::stats::compare_runs(a, b, filter, policy, dir)
}

/// Re-export the comparison-policy type so downstream crates using
/// `ktstr::cli` as their public surface don't need to reach into
/// the internal `ktstr::stats` module (which is `pub(crate)` —
/// see `lib.rs` — and therefore not a stable public path). The
/// policy is the only item in `stats` that a CLI or external
/// consumer constructs directly; every other item is internal
/// plumbing reached via `cli::compare_runs`.
pub use crate::stats::ComparisonPolicy;

/// Collect the current host context via
/// [`crate::host_context::collect_host_context`] and render it as
/// a human-readable multi-line report via
/// [`crate::host_context::HostContext::format_human`]. The output
/// ends with a newline; callers print it verbatim.
pub fn show_host() -> String {
    crate::host_context::collect_host_context().format_human()
}

/// Restore SIGPIPE to its default action (terminate the process)
/// so piping a ktstr binary's output to a reader that closes
/// early (e.g. `... | head`) does not panic inside `print!` /
/// `println!`. Rust's startup code sets SIGPIPE to `SIG_IGN`,
/// which turns the broken-pipe write into an `io::Error` that
/// `print!` escalates to a panic. Setting `SIG_DFL` restores the
/// POSIX "process terminates on SIGPIPE" convention that Unix
/// CLI tools rely on.
///
/// Call this at the TOP of each of the three user-facing CLIs'
/// `main` — `ktstr`, `cargo-ktstr`, and `ktstr-jemalloc-probe` —
/// before the tracing subscriber installs its stderr handler and
/// before any stdout write. Shared across `src/bin/ktstr.rs`,
/// `src/bin/cargo-ktstr.rs`, and `src/bin/jemalloc_probe.rs` so
/// the three CLIs behave identically under `|` pipelines and a
/// future reword of the SAFETY rationale lands in one place. The
/// `ktstr-jemalloc-alloc-worker` binary does NOT call this — it
/// is a test-fixture target spawned by the probe's closed-loop
/// integration tests, never piped by a human operator, and its
/// stdout emission path prints a single "ready" breadcrumb that
/// the test body ignores, so SIGPIPE restoration there would
/// add noise without benefit.
///
/// No return value; the call is effectively infallible (libc's
/// `signal(2)` can't fail for a standard signal + SIG_DFL
/// handler on a live process).
///
/// # Safety (FFI)
///
/// `libc::signal` is an FFI call with no memory effects (no
/// pointer dereferences, no mutation of Rust state). `SIG_DFL`
/// is a well-known constant handler. Call must run before any
/// stdout writes so the handler is in place by the time
/// `print!` fires.
pub fn restore_sigpipe_default() {
    // SAFETY: see fn-level doc comment.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

/// Render the archived host context for the named run, resolved
/// against `dir` (or `test_support::runs_root()` when `dir` is
/// `None`). Loads sidecars under the run directory and returns the
/// `HostContext::format_human` of the first sidecar that has a
/// populated `host` field — every sidecar in a single run captures
/// the same host, so first-wins is adequate.
///
/// Returns `Err` when:
/// - The run directory does not exist (actionable message names
///   the expected root),
/// - The run directory exists but has no sidecar data (matches
///   the `compare_runs` error shape),
/// - Every sidecar carried `host: None` (older pre-enrichment
///   runs won't have the field).
pub fn show_run_host(run: &str, dir: Option<&Path>) -> Result<String> {
    let root: std::path::PathBuf = match dir {
        Some(d) => d.to_path_buf(),
        None => crate::test_support::runs_root(),
    };
    let run_dir = root.join(run);
    if !run_dir.exists() {
        // Name the discovery command alongside the error so an
        // operator who fat-fingered a run key (common failure mode —
        // the keys are auto-generated hex identifiers) doesn't have
        // to reach for `--help` or source-reading to find the
        // enumeration surface. `cargo ktstr stats list` is the
        // authoritative run-key directory — see
        // [`StatsCommand::List`] wiring in src/bin/cargo-ktstr.rs.
        bail!(
            "run '{run}' not found under {}. \
             Run `cargo ktstr stats list` to enumerate available run keys.",
            root.display(),
        );
    }
    let sidecars = crate::test_support::collect_sidecars(&run_dir);
    if sidecars.is_empty() {
        bail!("run '{run}' has no sidecar data");
    }
    // First sidecar with a populated host wins. Every sidecar in a
    // single run captures the same host; pre-enrichment sidecars
    // may have `host: None`. Scan forward rather than take the
    // first entry so older data doesn't force a "no host context"
    // error when newer sidecars in the same run DO have it.
    let host = sidecars
        .iter()
        .find_map(|sc| sc.host.as_ref())
        .ok_or_else(|| {
            anyhow!(
                "run '{run}' has {} sidecar(s) but none carries a populated \
                 host context; this usually means the run predates host-context \
                 enrichment. Re-run the test to produce a sidecar with the \
                 current schema.",
                sidecars.len(),
            )
        })?;
    Ok(host.format_human())
}

/// Return the registered test name whose Levenshtein edit distance
/// from `query` is smallest AND within the closeness threshold, or
/// `None` if no candidate is close enough.
///
/// Threshold: `distance <= max(3, query.len() / 3)`. Two
/// considerations drove this shape:
/// - A flat `distance <= 3` cap misses legitimate typos on long
///   snake_case names — e.g. a 5-character drop from a 50-char test
///   name is a clear "did you mean" case but would fall outside
///   cargo's conventional 3-char cap.
/// - A pure relative cap like `query.len() / 3` under-tolerates on
///   short names (a 9-char query tolerates only 3 edits, identical
///   to the absolute cap; a 6-char query tolerates 2 — strict, but
///   proportionate).
///
/// Taking the max preserves the absolute-3 floor while letting
/// longer names benefit from proportional slack. Empty registry (no
/// tests declared in the running process) returns `None` cleanly.
/// Ties in distance resolve to the FIRST name encountered in the
/// `KTSTR_TESTS` iteration — stable across runs because linkme's
/// distributed slice preserves declaration order.
fn suggest_closest_test_name(query: &str) -> Option<&'static str> {
    let threshold = std::cmp::max(3, query.len() / 3);
    let mut best: Option<(usize, &'static str)> = None;
    for entry in crate::test_support::KTSTR_TESTS.iter() {
        let d = strsim::levenshtein(query, entry.name);
        if d > threshold {
            continue;
        }
        match best {
            Some((best_d, _)) if best_d <= d => continue,
            _ => best = Some((d, entry.name)),
        }
    }
    best.map(|(_, name)| name)
}

/// Return the registered scenario name whose Levenshtein edit
/// distance from `query` is smallest AND within the closeness
/// threshold, or `None` if no candidate is close enough. Sibling
/// of [`suggest_closest_test_name`] — same threshold shape
/// (`max(3, query.len() / 3)`), same first-wins tie rule, same
/// "no match" contract — but queries the scenario registry
/// ([`crate::scenario::all_scenarios`]) instead of `KTSTR_TESTS`.
///
/// Used by [`filter_scenarios`] and the `ktstr list` empty-output
/// surface to surface "Did you mean `<name>`?" hints when a
/// `--filter` value produces zero matches (typo UX per dev-advocate
/// finding 12.b).
///
/// The returned `&'static str` points into the scenario registry —
/// `Scenario.name` is `&'static str` by construction, so the
/// helper owns no allocations on the happy path.
fn suggest_closest_scenario_name(query: &str) -> Option<&'static str> {
    let threshold = std::cmp::max(3, query.len() / 3);
    let mut best: Option<(usize, &'static str)> = None;
    for s in crate::scenario::all_scenarios() {
        let d = strsim::levenshtein(query, s.name);
        if d > threshold {
            continue;
        }
        match best {
            Some((best_d, _)) if best_d <= d => continue,
            _ => best = Some((d, s.name)),
        }
    }
    best.map(|(_, name)| name)
}

/// Public "did you mean?" helper for callers that print a
/// zero-match scenario-filter diagnostic without routing through
/// [`filter_scenarios`]. Returns a formatted suffix like
/// `" Did you mean \`steady_state\`?"` on a near match, or `None`
/// when no scenario is close enough — callers concatenate the
/// suffix onto their own error message.
///
/// Used by the `ktstr list` inline filter (which does not bail on
/// zero matches, intentionally — an empty list is not an error)
/// to enrich the trailing banner with a typo suggestion. The
/// `filter_scenarios` error path uses [`suggest_closest_scenario_name`]
/// directly rather than this wrapper because it owns its own
/// message shape.
pub fn scenario_filter_hint(filter: &str) -> Option<String> {
    suggest_closest_scenario_name(filter).map(|s| format!(" Did you mean `{s}`?"))
}

/// Render the resolved, merged `Assert` thresholds for the named
/// test — the same merge chain evaluated at run time in
/// `run_ktstr_test_inner`:
/// `Assert::default_checks().merge(entry.scheduler.assert()).merge(&entry.assert)`.
///
/// Returns `Err` when no registered test matches `test_name`. The
/// CLI wiring (`cargo ktstr show-thresholds <test>`) surfaces this
/// to the operator without requiring them to read the source, the
/// nextest `--list` output, or the Debug impl of `Assert`.
pub fn show_thresholds(test_name: &str) -> Result<String> {
    let entry = crate::test_support::find_test(test_name).ok_or_else(|| {
        let suggestion = suggest_closest_test_name(test_name)
            .map(|s| format!(" Did you mean `{s}`?"))
            .unwrap_or_default();
        anyhow!(
            "no registered ktstr test named '{test_name}'.{suggestion} \
             Run `cargo nextest list` to see the available test names \
             — then pass just the function-name component to \
             `show-thresholds`, not the `<binary>::` prefix that \
             nextest prepends to each line."
        )
    })?;
    let merged = crate::assert::Assert::default_checks()
        .merge(entry.scheduler.assert())
        .merge(&entry.assert);
    let mut out = format!("Test: {}\n", entry.name);
    out.push_str(&format!(
        "Scheduler: {}\n",
        entry.scheduler.scheduler_name(),
    ));
    // The `Test:` + `Scheduler:` lines above establish context for
    // the indented threshold rows that follow; `Assert::format_human`
    // renders only the rows (caller owns the section header).
    // Prepending `Resolved assertion thresholds:` here keeps the
    // operator-visible output unchanged from the pre-fold shape so
    // shell pipelines grepping for the banner still match.
    out.push_str("Resolved assertion thresholds:\n");
    out.push_str(&merged.format_human());
    Ok(out)
}

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

/// List cgroup directories that `ktstr cleanup` / `cargo ktstr cleanup`
/// target by default: `/sys/fs/cgroup/ktstr` (test-harness parent) and
/// any `/sys/fs/cgroup/ktstr-<pid>` left behind by a `ktstr run` that
/// crashed or was SIGKILLed.
///
/// Returns only entries that exist and are directories. Silently
/// returns empty when `/sys/fs/cgroup` isn't a cgroup v2 mount. Skips
/// `ktstr-<pid>` directories whose pid still owns a live ktstr (or
/// cargo-ktstr) process, so a concurrent cleanup run doesn't rmdir an
/// active run's cgroup out from under it.
pub fn default_cleanup_parents() -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new("/sys/fs/cgroup");
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let Ok(ty) = entry.file_type() else { continue };
        if !ty.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name == "ktstr" {
            out.push(entry.path());
            continue;
        }
        if let Some(pid_str) = name.strip_prefix("ktstr-")
            && !pid_str.is_empty()
            && pid_str.bytes().all(|b| b.is_ascii_digit())
        {
            if is_ktstr_pid_alive(pid_str) {
                eprintln!("ktstr: skipping {} (live process)", entry.path().display());
                continue;
            }
            out.push(entry.path());
        }
    }
    out.sort();
    out
}

/// Return true when `/proc/{pid}/comm` identifies a live ktstr or
/// cargo-ktstr process. Returns false on any read error (pid exited,
/// non-Linux host, /proc not mounted) so the caller treats the cgroup
/// as cleanable.
pub fn is_ktstr_pid_alive(pid: &str) -> bool {
    let comm_path = format!("/proc/{pid}/comm");
    let Ok(comm) = std::fs::read_to_string(&comm_path) else {
        return false;
    };
    let comm = comm.trim();
    comm == "ktstr" || comm == "cargo-ktstr"
}

/// Reap leftover ktstr cgroup directories.
///
/// With `parent_cgroup` set, cleans only that path and leaves the
/// directory itself in place (matches `CgroupManager::cleanup_all`
/// semantics: purge children, keep parent). With `parent_cgroup` as
/// `None`, scans `/sys/fs/cgroup` for the default ktstr parents
/// reported by [`default_cleanup_parents`] and rmdirs each after
/// cleaning. Per-directory failures print to stderr and do not halt
/// the remaining sweep.
pub fn cleanup(parent_cgroup: Option<String>) -> Result<()> {
    use crate::cgroup::CgroupManager;

    match parent_cgroup {
        Some(path) => {
            if !std::path::Path::new(&path).exists() {
                bail!("cgroup path not found: {path}");
            }
            let cgroups = CgroupManager::new(&path);
            cgroups.cleanup_all()?;
            println!("cleaned up {path}");
        }
        None => {
            let parents = default_cleanup_parents();
            if parents.is_empty() {
                println!("no leftover cgroups found");
            } else {
                for path in parents {
                    let cgroups = CgroupManager::new(path.to_str().unwrap_or_default());
                    if let Err(e) = cgroups.cleanup_all() {
                        eprintln!("ktstr: cleanup_all failed on {}: {e}", path.display());
                        continue;
                    }
                    match std::fs::remove_dir(&path) {
                        Ok(()) => println!("cleaned up {}", path.display()),
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            println!("cleaned up {}", path.display());
                        }
                        Err(e) => {
                            eprintln!("ktstr: failed to remove {}: {e}", path.display());
                        }
                    }
                }
            }
        }
    }
    Ok(())
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
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, false, cpu_cap)?;

    match result.entry {
        Some(entry) => Ok(entry.path),
        None => bail!(
            "kernel built but cache store failed — cannot return image from temporary directory"
        ),
    }
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
    let is_source_tree = path.join("Makefile").exists() && path.join("Kconfig").exists();
    if !is_source_tree {
        bail!(
            "no kernel image found in {} (not a kernel source tree — \
             missing Makefile or Kconfig)",
            path.display()
        );
    }

    let acquired = crate::fetch::local_source(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let cache_key = acquired.cache_key.clone();

    // Clean trees: cache lookup before build.
    // Dirty trees: skip cache, always build.
    if !acquired.is_dirty
        && let Ok(cache) = crate::cache::CacheDir::new()
        && let Some(entry) = cache_lookup(&cache, &cache_key, cli_label)
    {
        let image = entry.image_path();
        if image.exists() {
            success(&format!("{cli_label}: using cached kernel {cache_key}"));
            return Ok(image);
        }
    }

    let cache = crate::cache::CacheDir::new()?;
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true, cpu_cap)?;

    // Prefer the cached image path (stable across rebuilds).
    match result.entry {
        Some(entry) => Ok(entry.image_path()),
        None => Ok(result.image_path),
    }
}

/// Whether stderr supports color (cached per process).
pub fn stderr_color() -> bool {
    use std::io::IsTerminal;
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| std::io::stderr().is_terminal())
}

/// Whether stdout supports color (cached per process). Distinct from
/// [`stderr_color`] because `cargo ktstr stats compare > report.txt`
/// pipes stdout to a file while leaving stderr on the TTY — gating
/// stdout tables on the stderr TTY state would leave ANSI escapes
/// in the file. Table-rendering code paths gate on this reading;
/// diagnostic/status prints use [`stderr_color`].
pub fn stdout_color() -> bool {
    use std::io::IsTerminal;
    static COLOR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *COLOR.get_or_init(|| std::io::stdout().is_terminal())
}

/// Build a borderless comfy-table with styling gated on
/// [`stdout_color`]. When stdout is not a TTY (CI, piped-to-file),
/// `force_no_tty` suppresses cell color escapes so a log or grep
/// capture does not land raw `\x1b[...` sequences. The NOTHING preset
/// skips box-drawing characters and keeps whitespace-padded columns,
/// matching the previous hand-rolled `format!("{:<30}…")` look while
/// auto-measuring each column from actual cell contents.
pub fn new_table() -> comfy_table::Table {
    use comfy_table::{ContentArrangement, Table, presets::NOTHING};
    let mut t = Table::new();
    t.load_preset(NOTHING);
    t.set_content_arrangement(ContentArrangement::Disabled);
    if !stdout_color() {
        t.force_no_tty();
    }
    t
}

// ---------------------------------------------------------------------------
// `ktstr locks` — observational enumeration of every ktstr flock on the host
// ---------------------------------------------------------------------------
//
// Troubleshooting companion to `--cpu-cap`: when a build or test is
// stalled behind a peer's reservation, `ktstr locks` names the peer
// (PID + cmdline) without disturbing any of its flocks. Reads
// `/tmp/ktstr-llc-*.lock`, `/tmp/ktstr-cpu-*.lock`, and
// `{cache_root}/.locks/*.lock`; calls [`crate::flock::read_holders`]
// once per file, which does a single `/proc/locks` parse internally.

/// One LLC-lock row in the `ktstr locks` output.
///
/// `pub(crate)` so the test-only [`collect_locks_snapshot_from`]
/// seam can return the type from outside its defining module.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct LlcLockRow {
    pub(crate) llc_idx: usize,
    pub(crate) numa_node: Option<usize>,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// One per-CPU-lock row. `numa_node` carries the host NUMA node the
/// CPU lives on, looked up via [`crate::topology::TestTopology`]'s
/// `cpu_to_node` map; `None` when the sysfs probe failed and the
/// host topology is unavailable.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct CpuLockRow {
    pub(crate) cpu: usize,
    pub(crate) numa_node: Option<usize>,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// One cache-entry-lock row. Cache locks live at
/// `{cache_root}/.locks/{cache_key}.lock`; `cache_key` is parsed
/// from the filename stem.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct CacheLockRow {
    pub(crate) cache_key: String,
    pub(crate) lockfile: String,
    pub(crate) holders: Vec<crate::flock::HolderInfo>,
}

/// Snapshot of every ktstr flock discoverable on the host at the
/// moment this is built. Assembled by [`collect_locks_snapshot`] and
/// rendered by either the human [`render_locks_human`] or JSON
/// [`serde_json::to_string_pretty`] path.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct LocksSnapshot {
    pub(crate) llcs: Vec<LlcLockRow>,
    pub(crate) cpus: Vec<CpuLockRow>,
    pub(crate) cache: Vec<CacheLockRow>,
}

/// Enumerate every ktstr lockfile reachable on the host, attach the
/// holder list parsed from `/proc/locks`, and return a structured
/// snapshot suitable for either human or JSON rendering.
///
/// Missing paths (no `/tmp` glob matches, no cache `.locks/`) produce
/// empty row vectors — not an error. The lockfile glob pattern uses
/// the `glob` crate (already a dep); failures to expand are treated
/// as "no files matched" and surfaced via `tracing::warn!` so the
/// operator still sees a populated snapshot for the paths that did
/// work.
fn collect_locks_snapshot() -> Result<LocksSnapshot> {
    let cache_root = CacheDir::default_root().ok();
    collect_locks_snapshot_from(Path::new("/tmp"), cache_root.as_deref())
}

/// Seam behind [`collect_locks_snapshot`]: enumerate LLC, per-CPU,
/// and cache-entry lockfiles under the given roots. Tests inject a
/// tempdir for `tmp_root` + `cache_root` so the `ktstr locks`
/// snapshot shape can be pinned without touching the real
/// host `/tmp` or the operator's cache directory.
///
/// `tmp_root` is the directory containing `ktstr-llc-*.lock` and
/// `ktstr-cpu-*.lock` (in production: `/tmp`). `cache_root` is the
/// cache-directory whose `.locks/` subdirectory holds per-entry
/// locks (in production: `CacheDir::default_root()`); `None`
/// suppresses the cache-lock enumeration entirely, matching the
/// "home unresolvable" production fallback.
pub(crate) fn collect_locks_snapshot_from(
    tmp_root: &Path,
    cache_root: Option<&Path>,
) -> Result<LocksSnapshot> {
    use crate::vmm::host_topology::HostTopology;

    // Sysfs probe is best-effort — a container without
    // /sys/devices/system/cpu populated still gets to see its flocks,
    // just without NUMA node annotation (degrades to `None` in JSON
    // and `"?"` in the human table). Both the LLC-index→node lookup
    // (via `llc_numa_node`) and the per-CPU→node lookup (via
    // `cpu_to_node`) live on HostTopology — no TestTopology needed.
    let host_topo = HostTopology::from_sysfs().ok();

    // LLC locks: {tmp_root}/ktstr-llc-{N}.lock
    let llc_pattern = format!("{}/ktstr-llc-*.lock", tmp_root.display());
    let mut llcs: Vec<LlcLockRow> = Vec::new();
    for entry in glob::glob(&llc_pattern)
        .map_err(|e| anyhow!("glob {llc_pattern}: {e}"))?
        .flatten()
    {
        let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        // Stem is like "ktstr-llc-0"; strip prefix to get the index.
        let Some(idx_str) = stem.strip_prefix("ktstr-llc-") else {
            continue;
        };
        let Ok(llc_idx) = idx_str.parse::<usize>() else {
            continue;
        };
        let holders = crate::flock::read_holders(&entry).unwrap_or_default();
        let numa_node = host_topo.as_ref().and_then(|t| {
            if llc_idx < t.llc_groups.len() {
                Some(t.llc_numa_node(llc_idx))
            } else {
                None
            }
        });
        llcs.push(LlcLockRow {
            llc_idx,
            numa_node,
            lockfile: entry.display().to_string(),
            holders,
        });
    }
    llcs.sort_by_key(|r| r.llc_idx);

    // Per-CPU locks: {tmp_root}/ktstr-cpu-{C}.lock
    let cpu_pattern = format!("{}/ktstr-cpu-*.lock", tmp_root.display());
    let mut cpus: Vec<CpuLockRow> = Vec::new();
    for entry in glob::glob(&cpu_pattern)
        .map_err(|e| anyhow!("glob {cpu_pattern}: {e}"))?
        .flatten()
    {
        let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(idx_str) = stem.strip_prefix("ktstr-cpu-") else {
            continue;
        };
        let Ok(cpu) = idx_str.parse::<usize>() else {
            continue;
        };
        let holders = crate::flock::read_holders(&entry).unwrap_or_default();
        let numa_node = host_topo
            .as_ref()
            .and_then(|t| t.cpu_to_node.get(&cpu).copied());
        cpus.push(CpuLockRow {
            cpu,
            numa_node,
            lockfile: entry.display().to_string(),
            holders,
        });
    }
    cpus.sort_by_key(|r| r.cpu);

    // Cache-entry locks: {cache_root}/.locks/*.lock — skipped when
    // `cache_root` is None (unresolvable home / test isolation).
    let mut cache: Vec<CacheLockRow> = Vec::new();
    if let Some(cache_root) = cache_root {
        let locks_dir = cache_root.join(".locks");
        let pattern = format!("{}/*.lock", locks_dir.display());
        if let Ok(expanded) = glob::glob(&pattern) {
            for entry in expanded.flatten() {
                let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let holders = crate::flock::read_holders(&entry).unwrap_or_default();
                cache.push(CacheLockRow {
                    cache_key: stem.to_string(),
                    lockfile: entry.display().to_string(),
                    holders,
                });
            }
        }
    }
    cache.sort_by(|a, b| a.cache_key.cmp(&b.cache_key));

    Ok(LocksSnapshot { llcs, cpus, cache })
}

/// Render a [`LocksSnapshot`] as three stacked comfy-tables for
/// interactive reading. Empty sections print "(none)" under their
/// header so the operator can distinguish "no locks of this kind" from
/// a display bug. NUMA column renders the numeric node when available
/// or `"?"` when the sysfs probe failed.
fn render_locks_human(snap: &LocksSnapshot) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let fmt_holders = |hs: &[crate::flock::HolderInfo]| -> String {
        if hs.is_empty() {
            crate::flock::NO_HOLDERS_RECORDED.to_string()
        } else {
            // Newline-separated so multi-holder lockfile rows
            // don't wrap mid-cmdline on narrow terminals (the
            // prior comma-joined form did). Within a comfy-table
            // cell, each holder now renders on its own line.
            hs.iter()
                .map(|h| format!("{} ({})", h.pid, h.cmdline))
                .collect::<Vec<_>>()
                .join("\n")
        }
    };
    let fmt_node = |n: Option<usize>| -> String {
        match n {
            Some(v) => v.to_string(),
            None => "?".to_string(),
        }
    };

    writeln!(out, "LLC locks:").unwrap();
    if snap.llcs.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["LLC", "NODE", "LOCKFILE", "HOLDERS"]);
        for r in &snap.llcs {
            t.add_row([
                r.llc_idx.to_string(),
                fmt_node(r.numa_node),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    writeln!(out, "\nPer-CPU locks:").unwrap();
    if snap.cpus.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["CPU", "NODE", "LOCKFILE", "HOLDERS"]);
        for r in &snap.cpus {
            t.add_row([
                r.cpu.to_string(),
                fmt_node(r.numa_node),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    writeln!(out, "\nCache-entry locks:").unwrap();
    if snap.cache.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["CACHE KEY", "LOCKFILE", "HOLDERS"]);
        for r in &snap.cache {
            t.add_row([
                r.cache_key.clone(),
                r.lockfile.clone(),
                fmt_holders(&r.holders),
            ]);
        }
        writeln!(out, "{t}").unwrap();
    }

    out
}

/// Shared kill flag for the `ktstr locks --watch` SIGINT handler.
/// `libc::signal` installs the C-level handler; the handler flips
/// this atomic so the redraw loop can exit cleanly between frames
/// instead of being torn down mid-print. The flag stays set for the
/// remainder of the process lifetime — `ktstr locks` is a one-shot
/// observational command, so re-arming is unnecessary.
static LOCKS_WATCH_KILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// SIGINT handler for `--watch`: flip the kill flag and return. The
/// main loop observes the flag between frames (at most one interval
/// after Ctrl-C) and exits. No buffered output is flushed here —
/// stdout is line-buffered on a pipe and fully flushed per-frame by
/// `println!`, so a mid-frame interrupt at worst drops the unwritten
/// portion of the final table, not the prior frame.
extern "C" fn locks_watch_sigint_handler(_sig: libc::c_int) {
    LOCKS_WATCH_KILL.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// `ktstr locks` entry point. See module-level doc block above.
///
/// When `watch` is `Some(interval)`, runs a redraw loop that prints
/// the snapshot, sleeps `interval`, and repeats until SIGINT. JSON
/// mode under `--watch` emits one JSON object per interval with a
/// trailing newline so streaming consumers can read frame-by-frame
/// via newline-delimited JSON.
pub fn list_locks(json: bool, watch: Option<std::time::Duration>) -> Result<()> {
    // One-shot: snapshot, render, done.
    if watch.is_none() {
        let snap = collect_locks_snapshot()?;
        if json {
            println!("{}", serde_json::to_string_pretty(&snap)?);
        } else {
            print!("{}", render_locks_human(&snap));
        }
        return Ok(());
    }
    let interval = watch.unwrap();

    // Install the SIGINT handler once. `libc::signal` returns the
    // previous handler; we discard it — `ktstr locks` is a terminal
    // command, nothing restores the prior handler on exit.
    // SAFETY: libc::signal is an FFI call with no memory effects.
    // `locks_watch_sigint_handler` is an `extern "C" fn` with the
    // correct `void(int)` signature. The handler only writes to a
    // static AtomicBool, which is async-signal-safe. Cast routes
    // fn-item → `*const ()` → `sighandler_t` so the
    // `function_casts_as_integer` lint is satisfied.
    unsafe {
        libc::signal(
            libc::SIGINT,
            locks_watch_sigint_handler as *const () as libc::sighandler_t,
        );
    }

    loop {
        if LOCKS_WATCH_KILL.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let snap = collect_locks_snapshot()?;
        if json {
            // Newline-delimited JSON: one frame = one line-terminated
            // object. `to_string_pretty` emits embedded newlines; use
            // the compact form under --watch so streaming consumers
            // can parse per-line.
            println!("{}", serde_json::to_string(&snap)?);
        } else {
            // ANSI clear-screen + home cursor, then the table.
            // `\x1b[2J` clears; `\x1b[H` moves to (1,1). Both standard
            // VT100 — every terminal ktstr supports honors them.
            print!("\x1b[2J\x1b[H{}", render_locks_human(&snap));
        }
        std::thread::sleep(interval);
    }

    Ok(())
}

/// Print a styled status message to stderr.
fn status(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[1m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Print a green success message to stderr.
fn success(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[32m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Print a blue warning to stderr.
fn warn(msg: &str) {
    if stderr_color() {
        eprintln!("\x1b[34m{msg}\x1b[0m");
    } else {
        eprintln!("{msg}");
    }
}

/// Stash of the pre-spinner termios for the panic hook's restore
/// path. Populated by [`Spinner::disable_echo`] before the ECHO flag
/// is cleared, and cleared by [`Spinner::teardown`] on normal exit.
/// The panic hook reads this mutex — when populated, it replays the
/// stashed termios to the terminal BEFORE the default panic handler
/// emits its message. Under `panic = "abort"`, `Spinner::Drop` never
/// runs, so without the hook the terminal stays in echo-disabled /
/// non-canonical mode and the multi-line panic message staircases
/// (LF without CR) before SIGABRT kills the process.
static SPINNER_SAVED_TERMIOS: std::sync::Mutex<Option<libc::termios>> = std::sync::Mutex::new(None);

/// Tracks whether a [`Spinner`] is currently alive. `Spinner::start`
/// flips this from `false` to `true`; `Drop` flips it back. A
/// `debug_assert!` at start-time fires when the previous value was
/// already `true`, catching nested `Spinner::start()` calls that
/// would clobber [`SPINNER_SAVED_TERMIOS`]: the second `start` saves
/// the outer spinner's ALREADY-ECHO-disabled termios, and the outer
/// teardown then restores to the disabled state instead of the
/// original. Release builds skip the check (the assertion compiles
/// away) rather than panic in production; the flag is still
/// maintained so a future `debug_assert` → `assert` upgrade would
/// not need a second seam.
static SPINNER_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Install a panic hook that restores stdin termios from
/// [`SPINNER_SAVED_TERMIOS`] before the default panic handler prints.
/// Called via [`std::sync::Once`] from [`Spinner::disable_echo`], so
/// every Spinner that actually mutates termios triggers the install
/// exactly once per process. Idempotent — subsequent calls hit the
/// `Once` guard and no-op.
///
/// The hook delegates to the default `take_hook()` output after
/// restoring, preserving the full panic-message contract (message,
/// location, backtrace under `RUST_BACKTRACE`).
///
/// # Panic-hook stacking convention
///
/// ktstr installs hooks in two places: this spinner-termios restorer
/// and the vCPU classifier (`crate::vmm::vcpu_panic::install_once`).
/// `std::panic::set_hook` is process-wide — whichever site installs
/// LAST wins, and earlier hooks are reached only via the previous-
/// hook chain each site captures at install time. Every ktstr-side
/// installer MUST follow the stacking pattern used here: call
/// `std::panic::take_hook()` to capture the current hook, then
/// `set_hook` a closure that runs its own work AND calls the
/// captured `prev(info)` at the end. Skipping the delegation
/// breaks the chain and silently drops every earlier-installed
/// hook. See the module-level doc on `src/vmm/vcpu_panic.rs` for
/// the full rationale (limitations section) and an alternative
/// `make_hook(prev)` factoring; the pattern is identical, just
/// packaged differently.
fn install_spinner_termios_panic_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // try_lock, not lock: if the panicking thread is the
            // one mid-mutation inside Spinner::disable_echo (holds
            // the mutex across its own libc::tcsetattr call), a
            // blocking lock would deadlock the hook. try_lock
            // failure ≈ "mutex held by someone mid-mutation" — the
            // terminal state is indeterminate and the hook
            // cannot safely restore, so we fall through to the
            // default handler unchanged.
            if let Ok(guard) = SPINNER_SAVED_TERMIOS.try_lock()
                && let Some(termios) = *guard
            {
                unsafe {
                    libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios);
                }
            }
            default(info);
        }));
    });
}

/// Progress spinner for long-running CLI operations.
///
/// When stderr is a TTY, draws an animated spinner via indicatif,
/// ticks in the background, and disables stdin echo to prevent
/// keypress jank. When stderr is not a TTY, skips all indicatif
/// machinery and falls back to plain stderr writes.
/// Call `finish` with a completion message to replace it with a
/// final line, or let it drop to remove it silently; [`Drop`] also
/// restores echo and clears the bar so a panic or early `?`
/// propagation leaves the terminal in a usable state. Under
/// `panic = "abort"`, Drop does NOT run on a panic — the panic hook
/// installed by [`install_spinner_termios_panic_hook`] restores
/// termios instead, so the panic message renders cleanly before
/// SIGABRT kills the process. Note: Drop also does NOT run on
/// SIGINT/SIGTERM kill; if the spinner is interrupted mid-operation,
/// run `stty sane` to restore echo.
pub struct Spinner {
    /// None when stderr is not a TTY — no indicatif overhead.
    pb: Option<indicatif::ProgressBar>,
    /// Saved termios for echo restore. None when stdin is not a tty
    /// or when the spinner is inactive (non-TTY stderr). Owned directly
    /// (not Arc<Mutex>) because Spinner is not Clone.
    saved_termios: Option<libc::termios>,
}

impl Spinner {
    /// Start a spinner with the given message (e.g. "Building kernel...").
    ///
    /// When stderr is not a TTY, no ProgressBar or ticker thread is
    /// created — all output methods fall back to plain `eprintln!`.
    pub fn start(msg: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        // Nesting rejection: a second `Spinner::start()` while
        // another Spinner is still live would overwrite
        // SPINNER_SAVED_TERMIOS with the ALREADY-ECHO-disabled
        // termios that the outer spinner installed; the outer's
        // Drop / teardown would then restore the disabled state
        // instead of the pre-spinner state, leaving the terminal
        // broken after both exit. `debug_assert!` catches the
        // misuse under `cargo test` / `cargo nextest` without
        // paying a release-mode cost. Release builds allow the
        // nesting and accept the terminal-leakage risk (the
        // alternative — panicking release binaries — would be
        // worse than a terminal that needs `reset` after a crash
        // path that was never exercised in testing). If nesting
        // is genuinely needed in the future, flip this guard and
        // add depth-aware save/restore logic to `teardown()`.
        //
        // The flag is swapped unconditionally at start (before the
        // TTY-absence short-circuit) AND cleared in both Drop and
        // the `is_hidden()` early-return below, so the invariant
        // `SPINNER_ACTIVE == true iff a Spinner exists` holds
        // across every exit path.
        debug_assert!(
            !SPINNER_ACTIVE.swap(true, std::sync::atomic::Ordering::SeqCst),
            "Spinner::start called while another Spinner is already \
             active. Nested spinners clobber SPINNER_SAVED_TERMIOS — \
             the outer spinner's restore path would reset to the \
             already-modified termios state instead of the original. \
             If nesting is genuinely needed, refactor the save/restore \
             path to depth-count before lifting this assertion.",
        );

        if !stderr_color() {
            return Spinner {
                pb: None,
                saved_termios: None,
            };
        }

        let pb = indicatif::ProgressBar::new_spinner();
        pb.set_style(
            indicatif::ProgressStyle::with_template("{spinner:.cyan} {msg}")
                .expect("valid template"),
        );
        pb.set_message(msg);
        pb.enable_steady_tick(Duration::from_millis(80));

        // indicatif hides the bar when NO_COLOR is set or TERM is
        // dumb, even on a real TTY. Downgrade to the non-TTY path
        // so println/finish output is not silently dropped.
        if pb.is_hidden() {
            return Spinner {
                pb: None,
                saved_termios: None,
            };
        }

        let saved_termios = Self::disable_echo();

        Spinner {
            pb: Some(pb),
            saved_termios,
        }
    }

    fn disable_echo() -> Option<libc::termios> {
        use std::io::IsTerminal;
        if !std::io::stdin().is_terminal() {
            return None;
        }
        unsafe {
            let fd = libc::STDIN_FILENO;
            let mut termios: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(fd, &mut termios) != 0 {
                return None;
            }
            let saved = termios;
            // Stash the pre-mutation termios for the panic hook's
            // restore path. Under `panic=abort` the Spinner's Drop
            // never runs, so if a panic fires while the spinner is
            // active the terminal stays in echo-disabled mode and
            // the panic message renders with a "staircase" effect
            // (LF without CR). The hook replays the saved termios
            // before the default panic handler prints, producing a
            // readable diagnostic on the way to SIGABRT.
            install_spinner_termios_panic_hook();
            *SPINNER_SAVED_TERMIOS.lock().unwrap() = Some(saved);
            termios.c_lflag &= !libc::ECHO;
            libc::tcsetattr(fd, libc::TCSANOW, &termios);
            Some(saved)
        }
    }

    /// Restore stdin echo if we disabled it, consuming `saved_termios`
    /// via [`Option::take`]. Idempotent — `finish` and the `Drop`
    /// impl both call this; only the first call has any effect. The
    /// old standalone `clear` method was consolidated into `Drop`
    /// (calling `drop(spinner)` produces the same effect).
    fn teardown(&mut self) {
        if let Some(termios) = self.saved_termios.take() {
            unsafe {
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios);
            }
            // Clear the panic-hook stash — further panics without a
            // live Spinner should NOT try to restore a termios we
            // already restored via the normal path.
            *SPINNER_SAVED_TERMIOS.lock().unwrap() = None;
        }
    }

    /// Update the spinner message.
    pub fn set_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        if let Some(ref pb) = self.pb {
            pb.set_message(msg);
        }
    }

    /// Finish the spinner, replacing it with a completion message.
    ///
    /// In non-TTY mode, prints the message to stderr directly.
    pub fn finish(mut self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.teardown();
        match self.pb.take() {
            Some(pb) => pb.finish_with_message(msg),
            None => eprintln!("{}", msg.into()),
        }
    }

    /// Print a line above the spinner. The spinner redraws below.
    ///
    /// In non-TTY mode, prints directly to stderr.
    pub fn println(&self, msg: impl AsRef<str>) {
        match self.pb {
            Some(ref pb) => pb.println(msg),
            None => eprintln!("{}", msg.as_ref()),
        }
    }

    /// Suspend the spinner tick, execute a closure, then resume.
    /// Use for terminal output that must not race with the spinner.
    ///
    /// In non-TTY mode, calls `f` directly (no spinner to suspend).
    pub fn suspend<F: FnOnce() -> R, R>(&self, f: F) -> R {
        match self.pb {
            Some(ref pb) => pb.suspend(f),
            None => f(),
        }
    }

    /// Run `f` under a spinner that starts with `start_msg`, replaces
    /// itself with `success_msg` on `Ok`, and drops silently on `Err`
    /// so the error propagates without a stale progress bar obscuring
    /// the caller's diagnostics. The closure receives the live
    /// `&Spinner` so it can call [`Self::println`] / [`Self::suspend`]
    /// / [`Self::set_message`] during the operation.
    pub fn with_progress<T, E, F>(
        start_msg: impl Into<std::borrow::Cow<'static, str>>,
        success_msg: impl Into<std::borrow::Cow<'static, str>>,
        f: F,
    ) -> Result<T, E>
    where
        F: FnOnce(&Spinner) -> Result<T, E>,
    {
        let sp = Spinner::start(start_msg);
        let result = f(&sp);
        match result {
            Ok(v) => {
                sp.finish(success_msg);
                Ok(v)
            }
            Err(e) => {
                drop(sp);
                Err(e)
            }
        }
    }
}

impl Drop for Spinner {
    /// Restore terminal echo and clear any live progress bar on drop.
    ///
    /// [`finish`](Self::finish) calls [`Self::teardown`] and takes
    /// `self.pb` via [`Option::take`], so this impl is a no-op after
    /// an explicit end. When the spinner is dropped implicitly
    /// (panic, `?` propagation, `drop(sp)`, or scope exit), this
    /// restores the termios saved in [`Self::disable_echo`] and
    /// clears the live bar so stdin is usable afterwards.
    fn drop(&mut self) {
        self.teardown();
        if let Some(pb) = self.pb.take() {
            pb.finish_and_clear();
        }
        // Release the nesting guard. Paired with the `swap(true)` in
        // `Spinner::start`: Drop fires exactly once per Spinner
        // (owned value), so the flag returns to `false` and the
        // next call to `start` can succeed. Unconditional store
        // rather than a swap — a nested misuse already panicked
        // under `debug_assert`, so the ordering of the counter
        // value on the first observer side is less important than
        // releasing the guard for the next legitimate caller.
        SPINNER_ACTIVE.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario;

    // -- parse_topology_string --

    /// Happy path: a canonical `"n,l,c,t"` string round-trips to the
    /// four u32 dimensions in positional order. Pins the field order
    /// so a future refactor that reshuffles (numa_nodes/llcs/cores/
    /// threads) → something else can't silently swap one dimension
    /// for another without flipping this pin.
    #[test]
    fn parse_topology_string_happy_path() {
        let (n, l, c, t) = parse_topology_string("1,2,4,8").expect("valid");
        assert_eq!((n, l, c, t), (1, 2, 4, 8));
    }

    /// Wrong component count: fewer than 4 parts names the expected
    /// shape in the error so the user sees the canonical format.
    #[test]
    fn parse_topology_string_rejects_too_few_parts() {
        let err = parse_topology_string("1,2,4").expect_err("3 parts must fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("invalid topology '1,2,4'"),
            "error must echo the bad input: {rendered}",
        );
        assert!(
            rendered.contains("numa_nodes,llcs,cores,threads"),
            "error must name the expected shape: {rendered}",
        );
    }

    /// Too MANY parts is rejected the same way. Pairs with the
    /// too-few case so the guard is symmetric.
    #[test]
    fn parse_topology_string_rejects_too_many_parts() {
        let err = parse_topology_string("1,2,4,8,16").expect_err("5 parts must fail");
        assert!(format!("{err:#}").contains("invalid topology"));
    }

    /// A non-numeric component fails with a message that names the
    /// offending FIELD, not just the bad token — a user who mistypes
    /// the second dimension sees `"invalid llcs value: 'abc'"` and
    /// knows immediately which dimension needs fixing. Pin all four
    /// position-to-name mappings so a field-order refactor surfaces
    /// here.
    #[test]
    fn parse_topology_string_names_failing_field() {
        for (pos, field) in [(0, "numa_nodes"), (1, "llcs"), (2, "cores"), (3, "threads")] {
            let mut parts = ["1"; 4];
            parts[pos] = "abc";
            let input = parts.join(",");
            let err = parse_topology_string(&input).expect_err("non-numeric must fail");
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains(&format!("invalid {field} value: 'abc'")),
                "pos {pos}: error must name the `{field}` field, got: {rendered}",
            );
        }
    }

    /// Zero in any position fails the `>= 1` guard with the
    /// "all values must be >= 1" phrasing. A zero topology would
    /// build a non-bootable VM, so rejecting it up-front is a
    /// correctness requirement, not a style choice.
    #[test]
    fn parse_topology_string_rejects_zero_dimensions() {
        for pos in 0..4 {
            let mut parts = ["1"; 4];
            parts[pos] = "0";
            let input = parts.join(",");
            let err = parse_topology_string(&input).expect_err("zero must fail");
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains(">= 1"),
                "pos {pos}: error must cite the >=1 rule: {rendered}",
            );
        }
    }

    /// Upper bound: u32::MAX in every position parses successfully.
    /// Pins the return-type decision (u32, not u16 / usize) so a
    /// future refactor that narrows the type surfaces here rather
    /// than truncating large-host topology strings.
    #[test]
    fn parse_topology_string_accepts_u32_max() {
        let big = u32::MAX;
        let input = format!("{big},{big},{big},{big}");
        let (n, l, c, t) = parse_topology_string(&input).expect("u32::MAX valid");
        assert_eq!((n, l, c, t), (big, big, big, big));
    }

    /// u32 overflow (value above u32::MAX) fails with the field
    /// name, not a generic parse error. Exercises the `parse::<u32>`
    /// failure path rather than only the non-numeric path.
    #[test]
    fn parse_topology_string_rejects_u32_overflow() {
        let too_big = (u32::MAX as u64) + 1;
        let input = format!("1,{too_big},4,1");
        let err = parse_topology_string(&input).expect_err("overflow must fail");
        assert!(
            format!("{err:#}").contains(&format!("invalid llcs value: '{too_big}'")),
            "overflow must surface field + bad token: {err:#}",
        );
    }

    // -- show_host smoke --

    /// `show_host` must return a non-empty, newline-terminated
    /// human-readable report. On Linux, `kernel_name` is
    /// populated from the `uname()` syscall regardless of `/proc`
    /// or `/sys` availability, so the output is guaranteed to
    /// contain that key. This also guards the
    /// [`print!("{}", cli::show_host())`] dispatch in
    /// `cargo-ktstr` — a future change that returns a
    /// trailing-newline-less string would drop the final line
    /// on the terminal.
    #[test]
    fn show_host_returns_populated_report() {
        let out = show_host();
        assert!(!out.is_empty(), "show_host must return non-empty output");
        assert!(
            out.ends_with('\n'),
            "show_host output must end with a newline for print! use: {out:?}",
        );
        assert!(
            out.contains("kernel_name"),
            "show_host must surface the kernel_name field: {out}",
        );
    }

    // -- show_run_host error paths + happy path --
    //
    // Each test builds an isolated `runs_root()` via `tempdir()`
    // and passes it through `--dir`, so no test touches the real
    // target/ktstr tree. Sidecars are constructed via
    // `SidecarResult::test_fixture()` to keep every required field
    // populated without hand-rolling 20 field values per test.

    /// Error path: the named run directory does not exist. Returns
    /// `Err` whose message names the missing run + expected root
    /// so an operator running `cargo ktstr stats show-host --run
    /// typo` sees an actionable message instead of a generic file-
    /// not-found error.
    #[test]
    fn show_run_host_missing_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = show_run_host("nonexistent-run", Some(tmp.path())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("run 'nonexistent-run' not found"),
            "missing-run error must name the run: {msg}",
        );
        // The error must also carry a discovery hint so an operator
        // hitting a typo sees the enumeration command inline. Without
        // this, the next recovery step is `--help` / source-reading.
        assert!(
            msg.contains("cargo ktstr stats list"),
            "missing-run error must name the `stats list` discovery \
             command so operators can enumerate available run keys \
             without extra lookups: {msg}",
        );
    }

    /// Error path: the run directory exists but has no sidecars.
    /// Returns `Err` with the `no sidecar data` diagnostic to
    /// match the `compare_runs` error shape — consistency across
    /// the two stats subcommands.
    #[test]
    fn show_run_host_empty_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("run-empty")).unwrap();
        let err = show_run_host("run-empty", Some(tmp.path())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no sidecar data"),
            "empty-run error must name the condition: {msg}",
        );
    }

    /// Error path: every sidecar in the run carries `host: None`
    /// (older pre-enrichment runs). Returns `Err` explaining the
    /// likely cause so the operator doesn't mistake this for a
    /// tooling bug.
    #[test]
    fn show_run_host_all_host_none_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-no-host");
        std::fs::create_dir(&run_dir).unwrap();
        // Write a sidecar with host: None. `SidecarResult::test_fixture`
        // defaults `host: None`, so the fixture shape matches the
        // pre-enrichment scenario.
        let sc = crate::test_support::SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(run_dir.join("t-0000000000000000.ktstr.json"), json).unwrap();
        let err = show_run_host("run-no-host", Some(tmp.path())).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no sidecar with a populated host")
                || msg.contains("none carries a populated host context"),
            "all-host-None error must name the pre-enrichment likely cause: {msg}",
        );
    }

    /// Happy path: a run with at least one sidecar carrying a
    /// populated host context returns the host's `format_human`
    /// output. Uses `HostContext::test_fixture()` to produce a
    /// predictable host; asserts the output contains a stable
    /// field that survives the fixture defaults.
    #[test]
    fn show_run_host_populated_sidecar_returns_format_human() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-with-host");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.host = Some(crate::host_context::HostContext::test_fixture());
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(run_dir.join("t-0000000000000000.ktstr.json"), json).unwrap();

        let out = show_run_host("run-with-host", Some(tmp.path())).unwrap();
        // `HostContext::format_human` always surfaces kernel_name;
        // the fixture populates it to a plausible value.
        assert!(
            out.contains("kernel_name"),
            "populated host output must include the kernel_name row: {out}",
        );
        assert!(
            out.ends_with('\n'),
            "output must end with newline for print!: {out:?}",
        );
    }

    /// Happy-path forward-scan: multiple sidecars where the FIRST
    /// has `host: None` but a later one has a populated host. The
    /// scanner uses `iter().find_map` so the populated sidecar is
    /// picked up; a regression that switched to `iter().next()`
    /// alone would fail here with the "all-host-None" error even
    /// though at least one sidecar carries host data.
    #[test]
    fn show_run_host_forward_scans_past_none_sidecars() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed");
        std::fs::create_dir(&run_dir).unwrap();
        // First sidecar: host None. File-name prefix `a-` ensures
        // this sorts before `b-` so iteration sees it first on
        // typical filesystems. `collect_sidecars` does not
        // guarantee order across filesystems; the forward-scan
        // invariant is order-independent, but the test is
        // deterministic within this single-filesystem env.
        let sc_none = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc_none).unwrap(),
        )
        .unwrap();
        // Second sidecar: host populated.
        let mut sc_host = crate::test_support::SidecarResult::test_fixture();
        sc_host.host = Some(crate::host_context::HostContext::test_fixture());
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc_host).unwrap(),
        )
        .unwrap();

        let out = show_run_host("run-mixed", Some(tmp.path()))
            .expect("forward scan must find the populated sidecar");
        assert!(
            out.contains("kernel_name"),
            "output from populated sidecar must include kernel_name: {out}",
        );
    }

    // -- Spinner Drop --

    #[test]
    fn spinner_drop_without_finish_does_not_panic_in_non_tty() {
        // Regression: Spinner previously had no Drop impl so early return
        // or panic leaked the disabled-ECHO termios. The added Drop must
        // run cleanly even on the non-TTY path (pb is None, saved_termios
        // is None) that nextest exercises under stderr capture.
        let sp = Spinner::start("test");
        drop(sp);
    }

    #[test]
    fn spinner_finish_then_drop_is_idempotent() {
        // finish() takes pb via Option::take so Drop's pb.take() sees None
        // and is a no-op on the progress bar side. teardown() is
        // idempotent because it consumes saved_termios via Option::take;
        // the second call finds None and does nothing. This test
        // exercises that lifecycle end-to-end.
        let sp = Spinner::start("test");
        sp.finish("done");
    }

    /// Nesting guard pin: starting a second Spinner while another is
    /// live must panic under `debug_assert!`. Exercises the
    /// SPINNER_ACTIVE swap — without the guard, the inner spinner
    /// would stash the outer's already-ECHO-disabled termios into
    /// SPINNER_SAVED_TERMIOS, and the outer's teardown would restore
    /// to that broken state instead of the pre-spinner original.
    ///
    /// `#[should_panic]` is gated on `debug_assertions` because the
    /// assertion compiles away in release builds; running the test
    /// without the debug gate under a release harness would make
    /// the test fail when the expected panic doesn't fire. The
    /// sibling `spinner_start_releases_guard_on_drop` test covers
    /// the happy path (non-nested sequential spinners) and runs
    /// under both profiles.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Spinner::start called while another Spinner is already active")]
    fn spinner_nested_start_panics_under_debug_assertions() {
        let _outer = Spinner::start("outer");
        // This call must fire the debug_assert! — the outer is
        // still live in scope. The test framework captures the
        // panic via `#[should_panic]`.
        let _inner = Spinner::start("inner");
    }

    /// Happy path paired with the nesting-panic test: starting two
    /// spinners SEQUENTIALLY (with the first dropped before the
    /// second starts) must succeed. Guards against a regression that
    /// forgot to clear SPINNER_ACTIVE in Drop and would one-shot the
    /// guard after a single use.
    #[test]
    fn spinner_start_releases_guard_on_drop() {
        {
            let _sp = Spinner::start("first");
            // Drop at end of block.
        }
        // After the first Spinner is dropped, the guard must be
        // cleared so a fresh start succeeds without panicking.
        let _sp = Spinner::start("second");
    }

    // -- drain_lines_lossy --

    #[test]
    fn drain_lines_lossy_eof_terminated_happy_path() {
        let input: &[u8] = b"alpha\nbeta\ngamma\n";
        let mut seen = Vec::new();
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |line| {
            seen.push(line.to_string())
        })
        .unwrap();
        assert_eq!(captured, vec!["alpha", "beta", "gamma"]);
        assert_eq!(seen, captured);
    }

    #[test]
    fn drain_lines_lossy_strips_crlf() {
        let input: &[u8] = b"one\r\ntwo\r\nthree\r\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["one", "two", "three"]);
    }

    #[test]
    fn drain_lines_lossy_non_utf8_bytes_survive_via_replacement() {
        // 0xFF is not valid UTF-8 in any position. `from_utf8_lossy`
        // replaces it with U+FFFD instead of dropping the line.
        let input: &[u8] = b"valid\n\xffbroken\ntail\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["valid", "\u{FFFD}broken", "tail"]);
    }

    #[test]
    fn drain_lines_lossy_empty_stream_yields_empty_vec() {
        let input: &[u8] = b"";
        let mut calls = 0usize;
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| calls += 1).unwrap();
        assert!(captured.is_empty());
        assert_eq!(calls, 0);
    }

    #[test]
    fn drain_lines_lossy_single_line_without_trailing_newline() {
        // Final chunk without a trailing newline should still be
        // emitted; BufRead::read_until returns the partial buffer
        // on EOF.
        let input: &[u8] = b"no-newline";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["no-newline"]);
    }

    #[test]
    fn drain_lines_lossy_lone_cr_at_eof_is_preserved() {
        // Bare CR without a following LF is NOT stripped — the CR
        // strip is nested inside the LF strip, so only `\r\n` is
        // normalized. A final chunk ending in `\r` keeps it.
        let input: &[u8] = b"foo\r";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["foo\r"]);
    }

    #[test]
    fn drain_lines_lossy_interior_cr_is_preserved() {
        // Only the trailing `\r` before `\n` is stripped; an
        // interior CR in the line body passes through verbatim.
        let input: &[u8] = b"ab\rcd\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["ab\rcd"]);
    }

    #[test]
    fn drain_lines_lossy_propagates_io_error_after_first_read() {
        use std::io::{BufReader, ErrorKind, Read};

        // Reader returns "line1\n" on the first read, then BrokenPipe
        // on the second. `drain_lines_lossy` must surface the Err
        // (return `Err(BrokenPipe)`) rather than silently ending the
        // read loop on mid-stream I/O failure. A mid-stream
        // broken-pipe that looks like EOF would drop the child's tail
        // output without a diagnostic; this test pins the contract
        // "error after first read propagates as Err".
        struct FlakyReader {
            calls: usize,
        }
        impl Read for FlakyReader {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                self.calls += 1;
                match self.calls {
                    1 => {
                        let data = b"line1\n";
                        let n = data.len().min(buf.len());
                        buf[..n].copy_from_slice(&data[..n]);
                        Ok(n)
                    }
                    _ => Err(std::io::Error::new(ErrorKind::BrokenPipe, "pipe closed")),
                }
            }
        }

        let err = drain_lines_lossy(BufReader::new(FlakyReader { calls: 0 }), |_| {})
            .expect_err("flaky reader must surface Err");
        assert_eq!(err.kind(), ErrorKind::BrokenPipe);
    }

    #[test]
    fn drain_lines_lossy_mixed_lf_and_crlf() {
        // A single stream with both LF-only and CRLF line endings.
        // Each line is stripped independently: the CR strip is nested
        // inside the LF strip, so an LF-only line passes through
        // without CR stripping while a CRLF line loses the CR.
        let input: &[u8] = b"lf-line\ncrlf-line\r\nlf-again\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["lf-line", "crlf-line", "lf-again"]);
    }

    #[test]
    fn drain_lines_lossy_empty_lines_lf() {
        // A bare `\n` between two non-empty lines produces an empty
        // string in the captured Vec — after strip_suffix(b"\n")
        // the remaining slice is empty and from_utf8_lossy("") == "".
        let input: &[u8] = b"a\n\nb\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["a", "", "b"]);
    }

    #[test]
    fn drain_lines_lossy_empty_lines_crlf() {
        // A bare `\r\n` produces an empty string after both the LF
        // and the preceding CR are stripped.
        let input: &[u8] = b"\r\n\r\n";
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_| {}).unwrap();
        assert_eq!(captured, vec!["", ""]);
    }

    #[test]
    fn drain_lines_lossy_callback_fires_once_per_line_in_order() {
        // Pin the externally-observable callback contract: `on_line`
        // is invoked exactly once per emitted line, in the same order
        // the lines appear in the returned Vec. Each invocation
        // records the count of prior invocations, yielding [0, 1, 2]
        // across three lines — proving once-per-line invocation and
        // stable ordering.
        let input: &[u8] = b"a\nb\nc\n";
        let lens = std::cell::RefCell::new(Vec::<usize>::new());
        let captured = drain_lines_lossy(std::io::Cursor::new(input), |_line| {
            let mut v = lens.borrow_mut();
            let current = v.len();
            v.push(current);
        })
        .unwrap();
        assert_eq!(captured, vec!["a", "b", "c"]);
        assert_eq!(lens.into_inner(), vec![0, 1, 2]);
    }

    // -- run_make_with_output --

    /// `Command::current_dir` on a non-existent path causes
    /// `Command::spawn` to fail before exec, with an underlying
    /// `io::Error` of kind `NotFound`. `run_make_with_output` wraps
    /// that error via `.with_context(|| format!("spawn make {}", ...))`,
    /// so the anyhow chain must surface BOTH the `"spawn make <args>"`
    /// annotation (proving the wrapping landed on the failing
    /// operation) AND the underlying `io::Error` with
    /// `ErrorKind::NotFound` (proving the io::Error chain was
    /// preserved through context). A regression that dropped either
    /// layer — bare `?` losing the context, or `Error::msg` losing
    /// the source — would surface here.
    ///
    /// The io::Error check uses `downcast_ref::<io::Error>()` on the
    /// anyhow chain rather than matching on the rendered
    /// `"No such file or directory"` string: the latter is the
    /// English-locale output of glibc's `strerror(ENOENT)`, but
    /// glibc translates per `LC_MESSAGES` — a CI runner with
    /// `LANG=fr_FR.UTF-8` would see "Aucun fichier ou dossier de ce
    /// type" and fail the test spuriously. `ErrorKind::NotFound` is
    /// structural and locale-free.
    ///
    /// Pipe2 + try_clone run before spawn, so reaching this error
    /// proves they succeeded too.
    #[test]
    fn run_make_with_output_surfaces_actionable_error_when_kernel_dir_missing() {
        // Per-invocation tempdir gives us a guaranteed-unique parent;
        // joining a nonexistent child name produces a path that is
        // provably-absent (the parent is fresh empty on every run).
        // Avoids a hardcoded `/this/path/should/not/exist/...` literal
        // that could collide across parallel test-runner instances and
        // becomes wrong the moment someone pollutes /this on the host.
        let tmp = tempfile::TempDir::new().unwrap();
        let missing = tmp.path().join("nonexistent_child");
        let err = run_make_with_output(&missing, &["foo"], None)
            .expect_err("nonexistent kernel_dir must surface a spawn failure");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("spawn make foo"),
            "expected `spawn make foo` context layer, got: {rendered}"
        );
        let has_not_found = err.chain().any(|e| {
            e.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
        });
        assert!(
            has_not_found,
            "expected underlying io::Error with ErrorKind::NotFound in anyhow chain, \
             got: {rendered}"
        );
    }

    /// End-to-end exercise of the merged-pipe path against a real
    /// `make` invocation that emits to BOTH stdout and stderr in
    /// large enough volume to fill a 64 KiB pipe buffer (Linux
    /// default), then exits non-zero. Two invariants:
    ///
    /// 1. **No-deadlock.** The production code creates ONE pipe
    ///    shared between stdout and stderr and reads it with a
    ///    single BufReader (no threads, no select). If the merge
    ///    were broken — e.g. if stderr were left attached to the
    ///    inherited fd 2 instead of `try_clone`'d onto the pipe
    ///    write end — high-volume stderr writes would still complete
    ///    via the inherited fd, but a regression that wired stderr
    ///    to a SECOND independent pipe with no reader would hang
    ///    the child after the first ~64 KiB of stderr writes. The
    ///    pipe-buffer-overflow lines below force that scenario; if
    ///    the test completes, no-deadlock holds.
    ///
    /// 2. **Failure-path Err.** A non-zero exit must surface as
    ///    `Err` with the `"make ... failed"` wording from the
    ///    `bail!` at the end of `run_make_with_output`. A regression
    ///    that swallowed the exit status or routed it through `Ok`
    ///    would hide compiler errors in CI logs.
    ///
    /// Skipped (passes silently) when `make` is not on PATH so the
    /// test suite stays runnable on minimal CI containers without
    /// build tools. The companion
    /// [`run_make_with_output_surfaces_actionable_error_when_kernel_dir_missing`]
    /// test exercises the spawn-failure path without needing make.
    #[test]
    fn run_make_with_output_drains_high_volume_failing_make_without_deadlock() {
        if resolve_in_path(std::path::Path::new("make")).is_none() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        // Default target prints 1 KiB to stdout and 1 KiB to stderr
        // for 100 iterations (~200 KiB total — well past the Linux
        // 64 KiB pipe buffer), then `false` exits 1. Recipe lines
        // MUST start with a TAB, not spaces — make rejects spaces.
        // `printf` with a 1 KiB byte string per call avoids relying
        // on a specific shell loop construct; the Makefile uses
        // make's own iteration via repetition.
        let stdout_chunk: String = "S".repeat(1024);
        let stderr_chunk: String = "E".repeat(1024);
        let mut recipe = String::new();
        for _ in 0..100 {
            recipe.push_str(&format!("\t@printf '%s\\n' '{stdout_chunk}'\n"));
            recipe.push_str(&format!("\t@printf '%s\\n' '{stderr_chunk}' >&2\n"));
        }
        let makefile = format!("default:\n{recipe}\t@false\n");
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();
        // Passing `default` explicitly anchors the error message
        // wording to `"make default failed"` rather than the
        // double-space `"make  failed"` form that `&[]` produces.
        let err = run_make_with_output(dir.path(), &["default"], None)
            .expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording from bail!, got: {rendered}"
        );
    }

    /// Redirect the test process's `stderr` (fd 2) to a tempfile for
    /// the duration of `f`, then restore. Returns the bytes written
    /// to fd 2 during the call. Used by tests that need to observe
    /// what the production code emitted via `eprintln!` — there is
    /// no in-band way to capture that without process-level fd
    /// manipulation because `eprintln!` writes straight through to
    /// fd 2.
    ///
    /// Uses [`nix::unistd::dup`] (returns an `OwnedFd` for the
    /// saved-stderr handle) and [`nix::unistd::dup2_stderr`] (the
    /// nix 0.31 purpose-built wrapper for redirecting fd 2); the
    /// latter sidesteps the generic `dup2`'s `&mut OwnedFd` newfd
    /// requirement — which would be awkward here, because fd 2 is
    /// not an `OwnedFd` we can lawfully construct. Dropping `saved`
    /// at scope exit closes it; fd 2 retains its own kernel-level
    /// reference to the original stderr open file description.
    ///
    /// Test-only utility — no production caller. Lives in this test
    /// module so its scope is bounded.
    ///
    /// Alias for the crate-shared stderr-capture helper from
    /// `test_support::test_helpers`. Kept under the local
    /// `capture_test_stderr` name so existing call sites below do
    /// not need renames. The prior module-local
    /// `STDERR_CAPTURE_LOCK` + `StderrRestoreGuard` + capture
    /// function that lived here were moved to
    /// `test_helpers::capture_stderr` so this module, `report.rs`,
    /// and any future capture site share ONE process-wide mutex on
    /// fd 2. Per-module mutexes would fail to serialize cross-module
    /// captures and race on the fd-2 swap.
    use crate::test_support::test_helpers::capture_stderr as capture_test_stderr;

    /// Direct proof of the merge: emit a unique marker on stdout and
    /// a different unique marker on stderr from a child process,
    /// then exit non-zero so `run_make_with_output` enters the
    /// failure branch and `eprintln!`s every captured line. The
    /// captured Vec is internal to the function, but the production
    /// code's failure-path `eprintln!` loop (lines 521-525) dumps
    /// every captured line to fd 2 before the `bail!` fires. Capture
    /// fd 2 around the call via [`capture_test_stderr`] and assert
    /// BOTH markers appear in the output.
    ///
    /// If the merge were broken (e.g. stderr installed on a
    /// separate pipe with no reader, or left attached to the
    /// parent's fd 2), the stderr marker would either deadlock the
    /// child or end up on the test process's original stderr — NOT
    /// in the captured Vec, NOT re-emitted by the failure-branch
    /// eprintln loop, NOT in the bytes this test reads from the
    /// sink. Asserting both markers appear is the empirical merge
    /// proof that complements the no-deadlock invariant pinned by
    /// `run_make_with_output_drains_high_volume_failing_make_without_deadlock`.
    ///
    /// Skipped when `make` is not on PATH for the same reason the
    /// high-volume test is.
    #[test]
    fn run_make_with_output_merges_stderr_into_captured_output() {
        if resolve_in_path(std::path::Path::new("make")).is_none() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        // Two distinguishable markers so the assertion can detect a
        // half-merge regression where only one stream made it through.
        let stdout_marker = "KTSTR_STDOUT_MARKER_e7f9";
        let stderr_marker = "KTSTR_STDERR_MARKER_a1b2";
        let makefile = format!(
            "default:\n\
             \t@printf '%s\\n' '{stdout_marker}'\n\
             \t@printf '%s\\n' '{stderr_marker}' >&2\n\
             \t@false\n"
        );
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();

        let (result, captured_bytes) =
            capture_test_stderr(|| run_make_with_output(dir.path(), &["default"], None));
        let err = result.expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording, got: {rendered}"
        );
        let captured = String::from_utf8_lossy(&captured_bytes);
        assert!(
            captured.contains(stdout_marker),
            "stdout marker missing from captured output (eprintln'd via failure path) — \
             expected `{stdout_marker}` in: {captured:?}"
        );
        assert!(
            captured.contains(stderr_marker),
            "stderr marker missing from captured output — proves the merge is BROKEN: \
             stderr did not reach the captured Vec. expected `{stderr_marker}` in: {captured:?}"
        );
    }

    /// Stderr-only high-volume burst: emit ~128 KiB to stderr alone
    /// (no interleaved stdout writes), then exit non-zero. This
    /// isolates the stderr-merge invariant from the stdout path.
    /// A regression that wired stderr to a separate unread pipe
    /// would deadlock the child after the first ~64 KiB (Linux
    /// default pipe buffer) since no reader exists on the broken
    /// stderr pipe. 128 KiB is double the buffer — definitely
    /// triggers the deadlock condition. Distinct from the
    /// interleaved high-volume test in
    /// [`run_make_with_output_drains_high_volume_failing_make_without_deadlock`]:
    /// that test interleaves so partial-merge regressions could
    /// "look" like they work because alternating stdout drains the
    /// pipe between stderr writes; this test forces stderr to drain
    /// alone. Test completion = no-deadlock pass.
    ///
    /// Skipped when `make` is not on PATH.
    #[test]
    fn run_make_with_output_drains_stderr_only_high_volume_without_deadlock() {
        if resolve_in_path(std::path::Path::new("make")).is_none() {
            skip!("make not in PATH");
        }
        let dir = tempfile::TempDir::new().unwrap();
        // 128 iterations * 1 KiB = 128 KiB of stderr — 2x the default
        // 64 KiB pipe buffer. No stdout writes at all so the buffer
        // can only drain via the merged-pipe reader.
        let chunk: String = "X".repeat(1024);
        let mut recipe = String::new();
        for _ in 0..128 {
            recipe.push_str(&format!("\t@printf '%s\\n' '{chunk}' >&2\n"));
        }
        let makefile = format!("default:\n{recipe}\t@false\n");
        std::fs::write(dir.path().join("Makefile"), makefile).unwrap();
        let err = run_make_with_output(dir.path(), &["default"], None)
            .expect_err("non-zero exit must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make default failed"),
            "expected `make default failed` wording, got: {rendered}"
        );
    }

    /// Spawn-failure path must not leak the pipe2 OwnedFds. Three
    /// fds are allocated before spawn: `read_fd` (pipe2 read end),
    /// `write_fd` (pipe2 write end, consumed by `Stdio::from`), and
    /// `write_fd_err` (`try_clone` of write_fd). When spawn fails:
    /// - `write_fd` is owned by the Command builder, which is
    ///   dropped on the early-return path.
    /// - `write_fd_err` is still in scope as an `OwnedFd` and drops
    ///   when the function returns.
    /// - `read_fd` is still in scope and drops on return.
    ///
    /// All three should release via OwnedFd's Drop (which calls
    /// `close()` on the inner fd). Count `/proc/self/fd` entries
    /// before and after a guaranteed-spawn-failure call; the count
    /// must not increase. A regression that switched to raw fd
    /// integers (no Drop) or that consumed the write_fd via a path
    /// other than Stdio::from (leaving it dangling on early return)
    /// would surface here as a leak of 1-3 fds per call.
    ///
    /// Linux-only: relies on `/proc/self/fd` enumeration. Skipped
    /// silently when /proc isn't a procfs mount.
    #[test]
    fn run_make_with_output_releases_fds_on_spawn_failure() {
        let proc_fd = std::path::Path::new("/proc/self/fd");
        if !proc_fd.is_dir() {
            skip!("/proc/self/fd not available");
        }
        let count_fds = || -> usize {
            std::fs::read_dir(proc_fd)
                .expect("read /proc/self/fd")
                .filter_map(|e| e.ok())
                .count()
        };
        // Warm-up pass: the first call may allocate process-wide
        // resources (thread-local buffers, lazy_static-like state in
        // the std/anyhow paths) that look like fd growth on a single
        // before/after measurement. Run once outside the measurement
        // window so steady-state fd usage is what we sample.
        //
        // Per-invocation tempdir yields a guaranteed-absent child path
        // without relying on a hardcoded `/this/path/should/not/exist`
        // literal that could collide across parallel test-runner
        // instances or become wrong the moment /this is polluted on
        // the host.
        let tmp = tempfile::TempDir::new().unwrap();
        let missing = tmp.path().join("nonexistent_child");
        let _ = run_make_with_output(&missing, &["foo"], None);
        let before = count_fds();
        // Run the failing path 128 times — a per-call leak of even
        // one fd would compound into a 128-fd delta, far above any
        // realistic transient-churn noise. Scaled up from the
        // previous 16-iteration loop because a 16-fd growth was
        // still in the range where process-wide churn (background
        // tempfile cleanup, tracing subscriber sinks, log rotators)
        // could mask a small per-call leak under a tolerant `<=`
        // comparison. 128 keeps the test under ~1s on a warm cache
        // while giving the signal-to-noise margin a genuine leak
        // needs to surface.
        const FD_LEAK_ITERATIONS: u32 = 128;
        for _ in 0..FD_LEAK_ITERATIONS {
            let _ = run_make_with_output(&missing, &["foo"], None);
        }
        let after = count_fds();
        assert!(
            after <= before,
            "fd leak on spawn failure: {before} -> {after} \
             ({FD_LEAK_ITERATIONS} calls, expected no growth)"
        );
    }

    // -- resolve_flags --

    #[test]
    fn cli_resolve_flags_none_returns_none() {
        assert!(resolve_flags(None).unwrap().is_none());
    }

    #[test]
    fn cli_resolve_flags_valid_single() {
        let result = resolve_flags(Some(vec!["llc".into()])).unwrap().unwrap();
        assert_eq!(result, vec!["llc"]);
    }

    #[test]
    fn cli_resolve_flags_valid_multiple() {
        let result = resolve_flags(Some(vec!["llc".into(), "borrow".into()]))
            .unwrap()
            .unwrap();
        assert_eq!(result, vec!["llc", "borrow"]);
    }

    #[test]
    fn cli_resolve_flags_all_valid() {
        let all: Vec<String> = flags::ALL.iter().map(|s| s.to_string()).collect();
        let result = resolve_flags(Some(all)).unwrap().unwrap();
        assert_eq!(result.len(), flags::ALL.len());
    }

    #[test]
    fn cli_resolve_flags_unknown_errors() {
        let err = resolve_flags(Some(vec!["nonexistent".into()])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown flag: 'nonexistent'"), "{msg}");
        assert!(msg.contains("valid flags:"), "{msg}");
    }

    #[test]
    fn cli_resolve_flags_mixed_valid_and_unknown_errors() {
        let err = resolve_flags(Some(vec!["llc".into(), "bogus".into()])).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown flag: 'bogus'"), "{msg}");
    }

    // -- parse_work_type --

    #[test]
    fn cli_parse_work_type_none_returns_none() {
        assert!(parse_work_type(None).unwrap().is_none());
    }

    #[test]
    fn cli_parse_work_type_cpu_spin() {
        let wt = parse_work_type(Some("CpuSpin")).unwrap().unwrap();
        assert_eq!(wt.name(), "CpuSpin");
    }

    #[test]
    fn cli_parse_work_type_yield_heavy() {
        let wt = parse_work_type(Some("YieldHeavy")).unwrap().unwrap();
        assert_eq!(wt.name(), "YieldHeavy");
    }

    #[test]
    fn cli_parse_work_type_all_valid() {
        for &name in WorkType::ALL_NAMES {
            if name == "Sequence" || name == "Custom" {
                continue;
            }
            let wt = parse_work_type(Some(name)).unwrap().unwrap();
            assert_eq!(wt.name(), name);
        }
    }

    #[test]
    fn cli_parse_work_type_unknown_errors() {
        let err = parse_work_type(Some("Nonexistent")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown work type: 'Nonexistent'"), "{msg}");
        assert!(msg.contains("valid types:"), "{msg}");
    }

    #[test]
    fn cli_parse_work_type_sequence_errors() {
        let err = parse_work_type(Some("Sequence")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("unknown work type: 'Sequence'"), "{msg}");
    }

    #[test]
    fn cli_parse_work_type_case_sensitive() {
        let err = parse_work_type(Some("cpuspin")).unwrap_err();
        assert!(format!("{err}").contains("unknown work type:"));
    }

    // -- filter_scenarios --

    #[test]
    fn cli_filter_scenarios_no_filter_returns_all() {
        let scenarios = scenario::all_scenarios();
        let result = filter_scenarios(&scenarios, None).unwrap();
        assert_eq!(result.len(), scenarios.len());
    }

    #[test]
    fn cli_filter_scenarios_matching_filter() {
        let scenarios = scenario::all_scenarios();
        let first_name = scenarios[0].name;
        let result = filter_scenarios(&scenarios, Some(first_name)).unwrap();
        assert!(!result.is_empty());
        for s in &result {
            assert!(s.name.contains(first_name));
        }
    }

    #[test]
    fn cli_filter_scenarios_no_match_errors() {
        let scenarios = scenario::all_scenarios();
        let err = filter_scenarios(&scenarios, Some("__nonexistent_scenario_xyz__")).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("no scenarios matched"), "{msg}");
        assert!(msg.contains("ktstr list"), "{msg}");
    }

    #[test]
    fn cli_filter_scenarios_partial_match() {
        let scenarios = scenario::all_scenarios();
        let result = filter_scenarios(&scenarios, Some("steady")).unwrap();
        assert!(!result.is_empty());
    }

    // -- build_run_config --

    #[test]
    fn cli_build_run_config_defaults() {
        let config = build_run_config(
            "/sys/fs/cgroup/ktstr".into(),
            20,
            4,
            None,
            false,
            None,
            false,
            None,
            None,
        );
        assert_eq!(config.parent_cgroup, "/sys/fs/cgroup/ktstr");
        assert_eq!(config.duration, Duration::from_secs(20));
        assert_eq!(config.workers_per_cgroup, 4);
        assert!(config.active_flags.is_none());
        assert!(!config.repro);
        assert!(config.probe_stack.is_none());
        assert!(!config.auto_repro);
        assert!(config.kernel_dir.is_none());
        assert!(config.work_type_override.is_none());
    }

    #[test]
    fn cli_build_run_config_all_fields() {
        let config = build_run_config(
            "/sys/fs/cgroup/test".into(),
            30,
            8,
            Some(vec!["llc", "borrow"]),
            true,
            Some("do_enqueue_task".into()),
            true,
            Some("/usr/src/linux".into()),
            Some(WorkType::Mixed),
        );
        assert_eq!(config.parent_cgroup, "/sys/fs/cgroup/test");
        assert_eq!(config.duration, Duration::from_secs(30));
        assert_eq!(config.workers_per_cgroup, 8);
        let af = config.active_flags.unwrap();
        assert_eq!(af, vec!["llc", "borrow"]);
        assert!(config.repro);
        assert_eq!(config.probe_stack.as_deref(), Some("do_enqueue_task"));
        assert!(config.auto_repro);
        assert_eq!(config.kernel_dir.as_deref(), Some("/usr/src/linux"));
        assert!(config.work_type_override.is_some());
    }

    #[test]
    fn cli_build_run_config_duration_converts() {
        let config = build_run_config("cg".into(), 60, 1, None, false, None, false, None, None);
        assert_eq!(config.duration, Duration::from_secs(60));
    }

    // -- scenario catalog --

    #[test]
    fn cli_all_scenarios_non_empty() {
        let scenarios = scenario::all_scenarios();
        assert!(!scenarios.is_empty());
    }

    #[test]
    fn cli_all_scenarios_have_names() {
        for s in &scenario::all_scenarios() {
            assert!(!s.name.is_empty());
            assert!(!s.category.is_empty());
        }
    }

    // -- has_sched_ext --

    #[test]
    fn cli_has_sched_ext_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_SCHED_CLASS_EXT=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_absent() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "CONFIG_SOMETHING=y\nCONFIG_OTHER=m\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_module_not_builtin() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "CONFIG_SCHED_CLASS_EXT=m\n").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_commented_out() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(
            tmp.path().join(".config"),
            "# CONFIG_SCHED_CLASS_EXT is not set\n",
        )
        .unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_no_config_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    #[test]
    fn cli_has_sched_ext_empty_config() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join(".config"), "").unwrap();
        assert!(!has_sched_ext(tmp.path()));
    }

    // -- build_make_args --

    #[test]
    fn cli_build_make_args_single_core() {
        let args = build_make_args(1);
        assert_eq!(args, vec!["-j1", "KCFLAGS=-Wno-error"]);
    }

    #[test]
    fn cli_build_make_args_multi_core() {
        let args = build_make_args(16);
        assert_eq!(args, vec!["-j16", "KCFLAGS=-Wno-error"]);
    }

    // -- analyze_sidecars (library API used by print_stats_report) --

    #[test]
    fn cli_analyze_sidecars_empty_dir() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = crate::test_support::analyze_sidecars(Some(tmp.path()));
        assert!(result.is_empty());
    }

    #[test]
    fn cli_analyze_sidecars_nonexistent_dir() {
        let result =
            crate::test_support::analyze_sidecars(Some(std::path::Path::new("/nonexistent/path")));
        assert!(result.is_empty());
    }

    // days_to_ymd tests moved to test_support::timefmt tests
    // (days_to_ymd_2024_jan_1, _2024_leap_day, _2023_end_of_year)
    // since the single implementation now lives there.

    // -- validate_kernel_config --

    /// Every entry in `VALIDATE_CONFIG_CRITICAL` must appear as `=y`
    /// in the embedded kconfig fragment. If a critical option is
    /// dropped from the fragment, builds would skip the option but
    /// validation would keep flagging it as missing — the user sees
    /// a build failure that no amount of tool installation fixes.
    /// This test catches the drift at compile-test time.
    #[test]
    fn critical_options_are_in_embedded_kconfig() {
        let fragment = crate::EMBEDDED_KCONFIG;
        for &(option, _) in VALIDATE_CONFIG_CRITICAL {
            let enabled = format!("{option}=y");
            assert!(
                fragment.lines().any(|l| l.trim() == enabled),
                "VALIDATE_CONFIG_CRITICAL lists {option:?} but ktstr.kconfig does not \
                 enable it; either add `{option}=y` to the fragment or drop the entry \
                 from VALIDATE_CONFIG_CRITICAL",
            );
        }
    }

    #[test]
    fn validate_kernel_config_all_present() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".config"),
            "CONFIG_SCHED_CLASS_EXT=y\n\
             CONFIG_DEBUG_INFO_BTF=y\n\
             CONFIG_BPF_SYSCALL=y\n\
             CONFIG_FTRACE=y\n\
             CONFIG_KPROBE_EVENTS=y\n\
             CONFIG_BPF_EVENTS=y\n",
        )
        .unwrap();
        assert!(validate_kernel_config(dir.path()).is_ok());
    }

    #[test]
    fn validate_kernel_config_missing_btf() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(".config"),
            "CONFIG_SCHED_CLASS_EXT=y\n\
             CONFIG_BPF_SYSCALL=y\n\
             CONFIG_FTRACE=y\n\
             CONFIG_KPROBE_EVENTS=y\n\
             CONFIG_BPF_EVENTS=y\n",
        )
        .unwrap();
        let err = validate_kernel_config(dir.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CONFIG_DEBUG_INFO_BTF"), "got: {msg}");
    }

    #[test]
    fn validate_kernel_config_missing_multiple() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF_SYSCALL=y\n").unwrap();
        let err = validate_kernel_config(dir.path()).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("CONFIG_SCHED_CLASS_EXT"), "got: {msg}");
        assert!(msg.contains("CONFIG_DEBUG_INFO_BTF"), "got: {msg}");
    }

    #[test]
    fn validate_kernel_config_no_config_file() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(validate_kernel_config(dir.path()).is_err());
    }

    // -- configure_kernel --

    #[test]
    fn configure_kernel_appends_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join(".config"), "CONFIG_BPF=y\n").unwrap();
        // configure_kernel runs `make olddefconfig` after appending.
        // Provide a stub Makefile so `make` succeeds without a real
        // kernel tree.
        std::fs::write(dir.path().join("Makefile"), "olddefconfig:\n\t@true\n").unwrap();
        let fragment = "CONFIG_EXTRA=y\n";
        configure_kernel(dir.path(), fragment).unwrap();
        let config = std::fs::read_to_string(dir.path().join(".config")).unwrap();
        assert!(config.contains("CONFIG_EXTRA=y"));
        assert!(config.contains("CONFIG_BPF=y"));
    }

    #[test]
    fn configure_kernel_skips_when_present() {
        let dir = tempfile::TempDir::new().unwrap();
        let initial = "CONFIG_BPF=y\nCONFIG_EXTRA=y\n";
        std::fs::write(dir.path().join(".config"), initial).unwrap();
        let fragment = "CONFIG_EXTRA=y\n";
        configure_kernel(dir.path(), fragment).unwrap();
        let config = std::fs::read_to_string(dir.path().join(".config")).unwrap();
        // Should not have appended (mtime preserved behavior).
        assert_eq!(config, initial);
    }

    #[test]
    fn configure_kernel_rejects_numeric_prefix_false_match() {
        // Fragment asks `CONFIG_NR_CPUS=1`, .config has
        // `CONFIG_NR_CPUS=128`. A plain
        // `config_content.contains(fragment_line)` would treat the
        // substring "CONFIG_NR_CPUS=1" as present inside
        // "CONFIG_NR_CPUS=128" (numeric prefix) and skip the append.
        // Exact-line matching via the HashSet helper correctly
        // distinguishes the two and appends.
        let dir = tempfile::TempDir::new().unwrap();
        let initial = "CONFIG_NR_CPUS=128\n";
        std::fs::write(dir.path().join(".config"), initial).unwrap();
        std::fs::write(dir.path().join("Makefile"), "olddefconfig:\n\t@true\n").unwrap();
        let fragment = "CONFIG_NR_CPUS=1\n";
        configure_kernel(dir.path(), fragment).unwrap();
        let config = std::fs::read_to_string(dir.path().join(".config")).unwrap();
        assert!(
            config.lines().any(|l| l.trim() == "CONFIG_NR_CPUS=1"),
            "CONFIG_NR_CPUS=1 must be appended as its own line: {config:?}"
        );
        assert!(
            config.lines().any(|l| l.trim() == "CONFIG_NR_CPUS=128"),
            "original CONFIG_NR_CPUS=128 must be preserved: {config:?}"
        );
    }

    // -- all_fragment_lines_present pure helper --

    #[test]
    fn all_fragment_lines_present_exact_match() {
        let config = "CONFIG_FOO=y\nCONFIG_BAR=m\n";
        assert!(all_fragment_lines_present("CONFIG_FOO=y\n", config));
        assert!(all_fragment_lines_present("CONFIG_BAR=m\n", config));
        assert!(all_fragment_lines_present(
            "CONFIG_FOO=y\nCONFIG_BAR=m\n",
            config
        ));
    }

    #[test]
    fn all_fragment_lines_present_numeric_prefix_not_present() {
        // The bug case. Substring match would incorrectly report present.
        let config = "CONFIG_NR_CPUS=128\n";
        assert!(!all_fragment_lines_present("CONFIG_NR_CPUS=1\n", config));
        assert!(!all_fragment_lines_present("CONFIG_NR_CPUS=12\n", config));
    }

    #[test]
    fn all_fragment_lines_present_disable_directive_participates() {
        // `# CONFIG_X is not set` is a real kconfig semantic (disable),
        // not a comment to be skipped. It must participate in the check.
        let config = "CONFIG_BPF=y\n";
        // Fragment disables CONFIG_BPF via the standard kconfig comment
        // syntax. Since .config has it enabled, the disable line is
        // NOT present and the helper must return false.
        assert!(!all_fragment_lines_present(
            "# CONFIG_BPF is not set\n",
            config
        ));
    }

    #[test]
    fn all_fragment_lines_present_empty_lines_skipped() {
        // Truly empty lines in the fragment carry no kconfig state.
        let config = "CONFIG_FOO=y\n";
        assert!(all_fragment_lines_present("\n\nCONFIG_FOO=y\n\n", config));
    }

    // -- resolve_in_path --

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

    // -- resolve_include_files --

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
        // "sh" is in PATH on all systems.
        let result = resolve_include_files(&[std::path::PathBuf::from("sh")]);
        assert!(result.is_ok());
        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].0.contains("sh"));
    }

    // -- kernel_list JSON/human parity --

    /// Pin the [`format_entry_row`] staleness mapping: the human
    /// `(stale kconfig)` tag appears iff `CacheEntry::kconfig_status`
    /// returns `KconfigStatus::Stale`. The test exercises every
    /// `KconfigStatus` variant (Matches, Stale, Untracked), so a
    /// regression that tightened the variant-to-tag mapping — e.g.
    /// surfacing Untracked as stale or dropping the Stale branch —
    /// surfaces as a tag/status disagreement.
    ///
    /// `kernel list --json` emits `kconfig_status` as a 3-value
    /// string (`"matches"` / `"stale"` / `"untracked"`) via
    /// `CacheEntry::kconfig_status(...).to_string()` at the
    /// JSON-branch call site — NOT a `stale_kconfig` boolean. The
    /// "JSON/human parity" phrasing in the test name refers to the
    /// shared `kconfig_status` gate both branches key off.
    ///
    /// The current body only evaluates the human branch against the
    /// `kconfig_status` method return; it does not exercise the
    /// JSON emission path, so a regression that broke the
    /// JSON-branch string serialization would slip through this
    /// test.
    #[test]
    fn kernel_list_stale_kconfig_json_human_parity() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        fn metadata_with_hash(hash: Option<&str>) -> KernelMetadata {
            KernelMetadata::new(
                KernelSource::Tarball,
                "x86_64".to_string(),
                "bzImage".to_string(),
                "2026-04-12T10:00:00Z".to_string(),
            )
            .with_version(Some("6.14.2".to_string()))
            .with_ktstr_kconfig_hash(hash.map(str::to_string))
        }

        // (case label, entry's recorded ktstr_kconfig_hash, caller's
        // current hash). These cover every KconfigStatus variant:
        // Matches, Stale, Untracked.
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

            // Human branch: format_entry_row emits "(stale kconfig)"
            // iff kconfig_status returns Stale.
            let human_row = format_entry_row(&entry, current_hash, &[]);
            let human_stale = human_row.contains("stale kconfig");

            assert_eq!(
                json_stale, human_stale,
                "kernel_list JSON/human stale-kconfig disagreement on `{label}` \
                 (entry_hash={entry_hash:?}, current_hash={current_hash:?}); \
                 json_stale={json_stale}, human_row={human_row:?}"
            );
        }
    }

    // -- version_prefix normalization --
    //
    // The minor-digit-only normalization is what makes `is_eol`
    // immune to RC and linux-next suffix drift between releases.json
    // and the local cache. Each case here pins one class of input
    // that previously produced a prefix collision mismatch.

    /// Stable release path — the original supported form.
    #[test]
    fn version_prefix_stable_release() {
        assert_eq!(version_prefix("6.14.2").as_deref(), Some("6.14"));
        assert_eq!(version_prefix("6.12.81").as_deref(), Some("6.12"));
        assert_eq!(version_prefix("7.0").as_deref(), Some("7.0"));
    }

    /// RC suffix must collapse into the same series as the final
    /// release. releases.json mainline cycles `6.15-rc1` → `-rc2` →
    /// … → `6.15`; every step must share the `6.15` prefix so a
    /// cache entry from an earlier RC stays non-EOL once a newer
    /// RC ships.
    #[test]
    fn version_prefix_strips_rc_suffix() {
        assert_eq!(version_prefix("6.15-rc1").as_deref(), Some("6.15"));
        assert_eq!(version_prefix("6.15-rc3").as_deref(), Some("6.15"));
        assert_eq!(version_prefix("7.0-rc1").as_deref(), Some("7.0"));
    }

    /// linux-next has versions like `6.16-rc2-next-20260420`. The
    /// `fetch_active_prefixes` walk already skips the `linux-next`
    /// moniker, but the PREFIX of a linux-next-derived cache key
    /// must still collapse to the target merge window (`6.16`) so
    /// it matches mainline's entry and stays non-EOL.
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

    /// `version.split_once('.')` returns `None` on a string with no
    /// `.`, so `"abc"`, `"6"`, and the empty string all fall through
    /// to the outer `None`.
    #[test]
    fn version_prefix_rejects_no_dot() {
        assert!(version_prefix("abc").is_none());
        assert!(version_prefix("6").is_none());
        assert!(version_prefix("").is_none());
    }

    /// A minor component with no leading digits (e.g. `"6.x"`) fails
    /// the `minor_digits.is_empty()` guard. Distinct from the no-dot
    /// path: we got past `split_once` but could not extract a
    /// numeric series. `"6."` is the degenerate case of the same
    /// guard — `split_once('.')` yields `("6", "")`, and the empty
    /// rest produces an empty digit collection.
    #[test]
    fn version_prefix_rejects_non_numeric_minor() {
        assert!(version_prefix("6.x").is_none());
        assert!(version_prefix("6.-rc1").is_none());
        assert!(version_prefix("6.").is_none());
    }

    // -- is_eol predicate --
    //
    // Pure function, no env / fixtures. Pins every return branch
    // documented on `is_eol`: the empty-slice guard, the
    // prefix-in-list branch, the prefix-absent branch, and the
    // unparseable-prefix branch.

    /// Empty `active_prefixes` is the "active list unknown" signal
    /// (fetch failure, skipped lookup). The empty-slice guard must
    /// return false so the `kernel list --json` contract holds:
    /// releases.json failure means no entry is tagged EOL. Without
    /// the guard, `!any(..)` on an empty iterator is `true` and the
    /// predicate would flip to tagging every entry EOL — the exact
    /// opposite of the contract.
    #[test]
    fn is_eol_empty_active_prefixes_returns_false() {
        assert!(!is_eol("6.14.2", &[]));
    }

    /// Happy path for an active series: the major.minor prefix
    /// (`6.14`) appears in the supplied `active_prefixes` list, so
    /// the any-match arm fires and the overall predicate returns
    /// false (not EOL).
    #[test]
    fn is_eol_prefix_in_active_list_returns_false() {
        assert!(!is_eol("6.14.2", &["6.14".to_string()]));
    }

    /// The failure path the predicate exists to detect: the
    /// version's `5.10` prefix is absent from a non-empty active
    /// list, so `!any(..)` fires and the predicate returns true.
    /// Sanity-checks the only code path that produces `true` in the
    /// current implementation.
    #[test]
    fn is_eol_prefix_absent_from_active_list_returns_true() {
        assert!(is_eol(
            "5.10.200",
            &["6.14".to_string(), "6.12".to_string()],
        ));
    }

    /// A version string with no parseable major.minor prefix (e.g.
    /// a cache key or freeform identifier) short-circuits via
    /// `version_prefix` and returns false. Distinct from the
    /// empty-slice branch above: the active list is non-empty here,
    /// so reaching false requires the prefix-absent short-circuit to
    /// fire.
    #[test]
    fn is_eol_unparseable_version_returns_false() {
        assert!(!is_eol("abc", &["6.14".to_string()]));
    }

    /// Regression guard: a cache entry on `6.15-rc1` compared
    /// against an active list that advanced to `6.15-rc4` must NOT
    /// be tagged EOL. Both prefixes normalize to `6.15`, so the
    /// comparison succeeds regardless of which RC the two sides
    /// happened to observe.
    #[test]
    fn is_eol_rc_suffix_mismatch_does_not_flag() {
        let active = ["6.15".to_string()];
        assert!(!is_eol("6.15-rc1", &active));
        assert!(!is_eol("6.15-rc4", &active));
    }

    /// linux-next cache keys (`6.16-rc2-next-YYYYMMDD`) must match
    /// mainline's entry for `6.16`. The prefix normalization in
    /// `version_prefix` collapses the `-rcN-next-*` chain.
    #[test]
    fn is_eol_linux_next_matches_mainline_prefix() {
        let active = ["6.16".to_string()];
        assert!(!is_eol("6.16-rc2-next-20260420", &active));
    }

    /// Brand-new major: releases.json has `"7.0-rc1"` mainline and
    /// the cache has a just-built `"7.0"` (or vice versa). Both
    /// collapse to `"7.0"` → not EOL.
    #[test]
    fn is_eol_brand_new_major_matches_rc_variant() {
        assert!(!is_eol("7.0", &["7.0".to_string()]));
        assert!(!is_eol("7.0-rc1", &["7.0".to_string()]));
    }

    /// First-release flavor: a brand-new `.0` release whose series is
    /// present in the active list must NOT be tagged EOL. Distinct
    /// from `is_eol_brand_new_major_matches_rc_variant` — that test
    /// pins the RC-cache-vs-release-list collapse; this one pins the
    /// pure released-.0 path, where both sides carry the `"7.0"`
    /// prefix directly. Includes the `"7.0.0"` alias form (same
    /// prefix `"7.0"` via `version_prefix`) so a regression that
    /// re-slices the minor component on a patch-suffixed brand-new
    /// release still holds.
    #[test]
    fn is_eol_brand_new_zero_release_in_active_list() {
        let active = ["7.0".to_string()];
        assert!(
            !is_eol("7.0", &active),
            "brand-new 7.0 release matching active prefix 7.0 must not be EOL",
        );
        assert!(
            !is_eol("7.0.0", &active),
            "7.0.0 carries prefix 7.0 via version_prefix and must not be EOL",
        );
    }

    /// A linux-next-derived cache entry targets the NEXT merge
    /// window, so its `major.minor` can precede any entry in the
    /// current stable/longterm active list. After the prefix
    /// normalization, `"6.16-rc1"` → `"6.16"`; when the active list
    /// (`fetch_active_prefixes` skips `linux-next` monikers by
    /// construction — see src/cli.rs) only carries older stable
    /// series like `"6.14"` / `"6.13"`, `"6.16"` is absent and the
    /// predicate returns true. This is INTENTIONAL — a linux-next
    /// kernel whose target merge window has not yet shipped on
    /// mainline is not a maintained series, so `(EOL)` accurately
    /// tells the user it is not receiving upstream fixes. The
    /// tag transitions to not-EOL as soon as mainline catches up
    /// (covered by `is_eol_linux_next_matches_mainline_prefix`).
    #[test]
    fn is_eol_linux_next_version_not_falsely_tagged() {
        assert!(
            is_eol("6.16-rc1", &["6.14".to_string(), "6.13".to_string()]),
            "linux-next cache entry whose merge-window target is ahead \
             of every stable series must be tagged EOL",
        );
    }

    // -- active_prefixes_from_releases normalization --
    //
    // Pure reducer feeding `fetch_active_prefixes`. Tests pin the
    // three behaviors the network wrapper could never exercise in
    // isolation: RC-suffix normalization, `linux-next` skip, and
    // first-seen dedup with input order preserved.

    fn owned(pairs: &[(&str, &str)]) -> Vec<crate::fetch::Release> {
        pairs
            .iter()
            .map(|(m, v)| crate::fetch::Release {
                moniker: (*m).to_string(),
                version: (*v).to_string(),
            })
            .collect()
    }

    /// Mainline entries in releases.json carry an `-rcN` suffix
    /// (`"6.16-rc3"`) whereas cached entries from an already-released
    /// series carry the bare form (`"6.16.2"`). Both sides must land
    /// on the same `"6.16"` prefix after normalization, otherwise the
    /// `(EOL)` annotation would flip on mainline entries every time a
    /// new `-rcN` ships.
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
            "RC-suffixed mainline entry must normalize to its merge-window series",
        );
    }

    /// `linux-next` moniker must be filtered BEFORE
    /// `version_prefix` so a `"6.17-rc2-next-YYYYMMDD"` entry does
    /// not seed a phantom `"6.17"` prefix that then shadows the
    /// mainline entry for a genuine merge window. The filter is
    /// moniker-based (not version-based) so the skip survives any
    /// future shape change in the linux-next version string.
    #[test]
    fn active_prefixes_from_releases_skips_linux_next_moniker() {
        let releases = owned(&[
            ("linux-next", "6.17-rc2-next-20260421"),
            ("mainline", "6.16-rc3"),
            ("stable", "6.15.2"),
        ]);
        let prefixes = active_prefixes_from_releases(&releases);
        assert!(
            !prefixes.contains(&"6.17".to_string()),
            "linux-next moniker must not seed a 6.17 prefix, got {prefixes:?}",
        );
        assert_eq!(
            prefixes,
            vec!["6.16".to_string(), "6.15".to_string()],
            "surviving prefixes come from mainline + stable only",
        );
    }

    /// First-seen dedup. releases.json ships the same series in
    /// both `stable` and a `longterm` row during the backport window
    /// around an LTS cut-over; both normalize to the same prefix
    /// and the helper must emit it once.
    #[test]
    fn active_prefixes_from_releases_dedups_in_input_order() {
        let releases = owned(&[
            ("stable", "6.14.2"),
            ("longterm", "6.14.1"),
            ("longterm", "6.12.81"),
        ]);
        let prefixes = active_prefixes_from_releases(&releases);
        assert_eq!(
            prefixes,
            vec!["6.14".to_string(), "6.12".to_string()],
            "dedup preserves first-seen order; 6.14 appears once",
        );
    }

    /// Pins JSON/human-output parity for the `(EOL)` annotation:
    /// any cache entry where the rendered text row contains
    /// `(EOL)` must also produce `eol: true` in the JSON view, and
    /// vice versa. Both code paths in `kernel_list` delegate to
    /// `is_eol`, so this test guards against a future change that
    /// introduces a second `is_eol`-like predicate in one branch
    /// and leaves the other behind — the exact drift mode that the
    /// kconfig-status parity test already guards against.
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

        // Three cases: active-not-EOL, EOL (non-empty active list),
        // and empty active list (fetch-failed fallback — neither side
        // may flag EOL).
        let cases: &[(&str, &str, &[&str])] = &[
            ("active", "6.14.2", &["6.14"]),
            ("eol", "2.6.32", &["6.14"]),
            ("fetch-fail", "2.6.32", &[]),
        ];

        for (label, version, active) in cases {
            let entry = make_entry(&format!("parity-{label}"), version);
            let active_vec: Vec<String> = active.iter().map(|s| s.to_string()).collect();
            let row = format_entry_row(&entry, "kconfig_hash", &active_vec);
            // Ask the SAME helper both emission paths delegate to
            // — not a re-derivation via `is_eol` inside the test.
            // This enforces parity by construction: a future
            // change that factors the eol computation into a new
            // helper on one side but leaves the other pointing
            // here would be caught immediately when `entry_is_eol`
            // is no longer used by both sites.
            let json_eol = entry_is_eol(&entry, &active_vec);
            let human_eol = row.contains("(EOL)");
            assert_eq!(
                json_eol, human_eol,
                "JSON/human parity broken for case {label}: \
                 json_eol={json_eol}, human_eol={human_eol}, row={row:?}",
            );
        }
    }

    /// Corrupt-entry footer fires iff at least one `ListedEntry::Corrupt`
    /// was rendered in the text path of `kernel_list`. The
    /// `kernel_list` function itself mixes cache IO + stdout/stderr,
    /// so unit-testing the footer shape requires re-constructing
    /// the in-loop decision. Here we mirror the any_corrupt logic
    /// against a small fixture and pin that the footer message
    /// contains the three actionable pieces (key, `kernel clean
    /// --force`, and the cache root path) — a regression that
    /// deletes any of those leaves users with an inactionable
    /// warning.
    #[test]
    fn kernel_list_corrupt_footer_fires_iff_any_corrupt() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = tmp.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        let image = src_dir.join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();

        // Create a valid entry, then build a synthetic Corrupt via
        // the listed-entry enum so the test does not depend on
        // CacheDir::list's corruption-detection internals.
        let meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-22T00:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()));
        // Store two separately-keyed entries so each test case owns
        // one — `CacheEntry` doesn't derive `Clone`, so building two
        // `ListedEntry::Valid` values requires two `store` calls.
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

        let entries_with_corrupt = [crate::cache::ListedEntry::Valid(valid_1), corrupt_entry];
        let entries_clean_only = [crate::cache::ListedEntry::Valid(valid_2)];

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

        // Call the SAME helper production uses so a regression in
        // the footer wording surfaces here. A local re-construction
        // would be tautological (tests itself, not production).
        let footer = format_corrupt_footer(cache.root());
        assert!(
            footer.contains("(corrupt)"),
            "footer must reference the tag users see",
        );
        assert!(
            footer.contains("kernel clean --force"),
            "footer must offer a remediation command",
        );
        // Scope safety: the `--force` variant removes every cached
        // entry — pin the "ALL" call-out so an operator reading the
        // footer in isolation cannot misread it as surgical.
        assert!(
            footer.contains("ALL cached entries"),
            "footer must spell out that `kernel clean --force` is not surgical",
        );
        // Actionable partial cleanup: preserve N newest valid
        // entries while removing the rest. Pinning this ensures the
        // less-destructive option stays surfaced.
        assert!(
            footer.contains("kernel clean --keep N --force"),
            "footer must offer a partial-cleanup alternative",
        );
        assert!(
            footer.contains(&cache.root().display().to_string()),
            "footer must name the cache root so operators know where to inspect",
        );
    }

    /// `eol_legend_if_any` is the sole gate on whether the text
    /// output under `kernel list` emits the `(EOL)` legend. Pinning
    /// both branches avoids a regression that would print the legend
    /// on clean no-EOL runs (noise) or suppress it on EOL runs
    /// (users cannot interpret the tag). The returned `&'static str`
    /// is the same `EOL_EXPLANATION` literal embedded at the head of
    /// `KERNEL_LIST_LONG_ABOUT` (which drives `kernel list --help`) via the
    /// shared `eol_explanation_literal!` macro, so drift between the
    /// legend and the help copy is impossible by construction.
    #[test]
    fn eol_legend_if_any_branches() {
        assert_eq!(eol_legend_if_any(true), Some(EOL_EXPLANATION));
        assert_eq!(eol_legend_if_any(false), None);
    }

    /// `KERNEL_LIST_LONG_ABOUT` drives `kernel list --help` and must expose
    /// the `--json` output contract so scripted consumers can
    /// discover the schema from the terminal alone. Pins:
    /// 1. the `(EOL)` legend text appears verbatim at the head
    ///    (enforced at compile time by the shared
    ///    `eol_explanation_literal!` macro; the runtime assert here
    ///    catches any future edit that causes `EOL_EXPLANATION` and
    ///    `KERNEL_LIST_LONG_ABOUT`'s head to diverge in content);
    /// 2. every top-level wrapper field (`current_ktstr_kconfig_hash`,
    ///    `active_prefixes_fetch_error`, `entries`) appears, so a
    ///    scripted consumer finds them without reading source;
    /// 3. every field listed in the test array is present in
    ///    `KERNEL_LIST_LONG_ABOUT`; the test array is maintained in
    ///    lockstep with the JSON emitter by code-review discipline,
    ///    not at compile time (a new field added to the emitter
    ///    without touching this test array will not trip the
    ///    assertion);
    /// 4. each field declared `Option<T>` in `cache::KernelMetadata`
    ///    is annotated `(nullable)` in the help copy, so consumers
    ///    know to handle `null` without reading source;
    /// 5. the corrupt-entry `error` marker appears so the two
    ///    entry shapes (valid vs corrupt) are both documented.
    ///
    /// Not pinning the entire byte sequence because a small
    /// wording tweak should not require a snapshot update — the
    /// discoverability contract is the invariant.
    #[test]
    fn kernel_list_long_about_exposes_json_schema() {
        assert!(
            KERNEL_LIST_LONG_ABOUT.starts_with(EOL_EXPLANATION),
            "KERNEL_LIST_LONG_ABOUT must embed EOL_EXPLANATION verbatim at its \
             head so the --help and post-table legend share one source of \
             truth; got: {KERNEL_LIST_LONG_ABOUT:?}",
        );

        for wrapper_field in [
            "current_ktstr_kconfig_hash",
            "active_prefixes_fetch_error",
            "entries",
        ] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(wrapper_field),
                "KERNEL_LIST_LONG_ABOUT must mention top-level wrapper field \
                 `{wrapper_field}` so scripted consumers discover the \
                 schema without `cargo doc`; got: {KERNEL_LIST_LONG_ABOUT:?}",
            );
        }

        for valid_entry_field in [
            "key",
            "path",
            "version",
            "source",
            "arch",
            "built_at",
            "ktstr_kconfig_hash",
            "kconfig_status",
            "eol",
            "config_hash",
            "image_name",
            "image_path",
            "has_vmlinux",
            "vmlinux_stripped",
            // Source-variant payload fields (internally-tagged
            // `source` object). `"ref"` is asserted in its quoted
            // JSON-field form to avoid matching substrings like
            // "prefixes" / "reference" elsewhere in the help copy;
            // the bare token would produce a false-positive pass
            // even if the field were removed. Without these, a
            // `KernelSource` variant that adds, removes, or
            // renames a payload field could ship without a
            // matching help update — silently breaking scripted
            // consumers that dispatch on `source.type`.
            "git_hash",
            "\"ref\"",
            "source_tree_path",
        ] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(valid_entry_field),
                "KERNEL_LIST_LONG_ABOUT must mention valid-entry JSON \
                 field `{valid_entry_field}`; a JSON emitter that adds \
                 a field without updating the help copy silently \
                 breaks the discoverability contract; got: \
                 {KERNEL_LIST_LONG_ABOUT:?}",
            );
        }

        // Corrupt-entry shape: `ListedEntry::Corrupt` emits a
        // structurally different object with only `key`, `path`,
        // `error`. `key` and `path` overlap with valid-entry fields
        // and are covered above. `error` is corrupt-only — asserted
        // separately so the failure message does not mislead a
        // future reader into grepping the valid-entry code path.
        {
            let corrupt_entry_field = "error";
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(corrupt_entry_field),
                "KERNEL_LIST_LONG_ABOUT must mention corrupt-entry JSON \
                 field `{corrupt_entry_field}` so consumers know the \
                 corrupt-entry shape and can branch on its presence; \
                 got: {KERNEL_LIST_LONG_ABOUT:?}",
            );
        }

        // Nullable markers: every `Option<T>` field declared in
        // `cache::KernelMetadata` must carry a `(nullable)` tag in
        // the help copy so scripted consumers handle `null` without
        // reading source. Sourced from cache.rs: `version`,
        // `ktstr_kconfig_hash`, `config_hash` are the three
        // `Option<String>` fields flowing into the JSON emitter.
        for nullable_field in ["version", "ktstr_kconfig_hash", "config_hash"] {
            let marker = format!("{nullable_field} (nullable)");
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(&marker),
                "KERNEL_LIST_LONG_ABOUT must mark `{nullable_field}` \
                 as `(nullable)` (expected substring `{marker}`) so \
                 consumers know to handle `null`; got: \
                 {KERNEL_LIST_LONG_ABOUT:?}",
            );
        }

        for source_variant_tag in ["\"tarball\"", "\"git\"", "\"local\""] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(source_variant_tag),
                "KERNEL_LIST_LONG_ABOUT must list source variant tag \
                 `{source_variant_tag}` so consumers can dispatch on \
                 the internally-tagged `source.type` field; got: \
                 {KERNEL_LIST_LONG_ABOUT:?}",
            );
        }

        for status_variant in ["\"matches\"", "\"stale\"", "\"untracked\""] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(status_variant),
                "KERNEL_LIST_LONG_ABOUT must list kconfig_status variant \
                 `{status_variant}` so consumers can branch on the \
                 three-value enum without reading source; got: \
                 {KERNEL_LIST_LONG_ABOUT:?}",
            );
        }
    }

    /// Pins that the `#[command(long_about = KERNEL_LIST_LONG_ABOUT)]`
    /// attribute on `KernelCommand::List` is actually wired through
    /// clap's registered metadata — i.e. the bytes that reach a
    /// terminal user via `kernel list --help` equal
    /// `KERNEL_LIST_LONG_ABOUT` byte-for-byte. Complements the
    /// content-discovery test above: that test guards the string's
    /// shape; this test guards the wiring. A regression that drops
    /// the attribute, points it at a different const, or leaves the
    /// attribute pointing at stale text would pass the shape test
    /// (the const is still well-formed) but fail here. Follows the
    /// `TestCli` pattern used by `kernel_clean_rejects_corrupt_only_
    /// with_keep` at cli.rs below.
    #[test]
    fn kernel_list_long_about_wired_via_clap() {
        use clap::CommandFactory as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let cmd = TestCli::command();
        let list = cmd
            .find_subcommand("list")
            .expect("clap must register a `list` subcommand on KernelCommand");
        let long_about = list
            .get_long_about()
            .expect(
                "`list` subcommand must have a long_about set (drives \
                 `kernel list --help`)",
            )
            .to_string();
        assert_eq!(
            long_about, KERNEL_LIST_LONG_ABOUT,
            "clap's registered long_about for `list` must equal \
             KERNEL_LIST_LONG_ABOUT byte-for-byte; a mismatch means \
             the `#[command(long_about = ...)]` attribute is missing, \
             pointing at a different const, or clap mutated the \
             content on its way into the registry",
        );
    }

    /// `untracked_legend_if_any` mirrors `eol_legend_if_any`'s
    /// gate-is-the-contract discipline. Both branches pinned so
    /// the legend can't silently disappear on an untracked run or
    /// show as noise on a clean run. Gives `(untracked kconfig)` tag
    /// readers a one-line explanation of the tag on par with `(EOL)`.
    #[test]
    fn untracked_legend_if_any_branches() {
        assert_eq!(
            untracked_legend_if_any(true),
            Some(UNTRACKED_KCONFIG_EXPLANATION),
        );
        assert_eq!(untracked_legend_if_any(false), None);
    }

    /// `stale_legend_if_any` completes the kconfig legend pair —
    /// every kconfig-status legend (UNTRACKED, STALE) now flows
    /// through one `*_legend_if_any` shape. True branch returns the
    /// `STALE_KCONFIG_EXPLANATION` const verbatim (no per-call
    /// formatting), false branch returns `None` so the noise-free
    /// invariant on clean runs is enforced by the gate.
    #[test]
    fn stale_legend_if_any_branches() {
        assert_eq!(stale_legend_if_any(true), Some(STALE_KCONFIG_EXPLANATION),);
        assert_eq!(stale_legend_if_any(false), None);
    }

    /// `STALE_KCONFIG_EXPLANATION` carries the four actionable
    /// elements its inline-eprintln predecessor did: the warning
    /// preamble, the tag word `(stale kconfig)`, the cause
    /// (different ktstr.kconfig fragment), and the
    /// `kernel build --force <entry version>` remediation. A reword
    /// that drops any of them silently degrades operator guidance
    /// — pinning each piece catches the regression here. The
    /// `<entry version>` placeholder (angle brackets, lowercase)
    /// is the convention ktstr uses to mark operator-substituted
    /// tokens; a regression to a bare all-caps `VERSION` would
    /// let a reader mistake it for a literal shell token.
    #[test]
    fn stale_kconfig_explanation_shape() {
        assert!(
            STALE_KCONFIG_EXPLANATION.starts_with("warning"),
            "stale legend must keep the warning preamble: {STALE_KCONFIG_EXPLANATION}",
        );
        assert!(
            STALE_KCONFIG_EXPLANATION.contains("(stale kconfig)"),
            "stale legend must name the tag verbatim: {STALE_KCONFIG_EXPLANATION}",
        );
        assert!(
            STALE_KCONFIG_EXPLANATION.contains("different ktstr.kconfig"),
            "stale legend must name the cause: {STALE_KCONFIG_EXPLANATION}",
        );
        assert!(
            STALE_KCONFIG_EXPLANATION.contains("kernel build --force <entry version>"),
            "stale legend must name the rebuild remediation with the \
             `<entry version>` placeholder: {STALE_KCONFIG_EXPLANATION}",
        );
    }

    /// `corrupt_footer_if_any` completes the trio of `*_if_any`
    /// tag-gate helpers. A positive count must yield a footer that
    /// carries the full `format_corrupt_footer` content (so
    /// operators see the cache_root path and all three `kernel
    /// clean` variants), prefixed with the count-summary sentence
    /// a dev-advocate flagged as load-bearing (an operator looking
    /// at `kernel list` should see HOW MANY entries are broken
    /// without re-counting rows). A zero count must be `None` — a
    /// regression that emitted the footer on every run would ship
    /// a "0 corrupt entries" line to every clean invocation.
    #[test]
    fn corrupt_footer_if_any_branches() {
        let root = std::path::Path::new("/tmp/ktstr-cache-test-root");
        // Zero → None.
        assert_eq!(corrupt_footer_if_any(0, root), None);
        // Positive count → Some, with the count-summary line AND the
        // full format_corrupt_footer body.
        let one = corrupt_footer_if_any(1, root).expect("positive count must yield Some(footer)");
        assert!(
            one.contains("1 corrupt entry."),
            "singular form (count == 1) must render as `1 corrupt entry.`; got: {one}",
        );
        assert!(
            one.contains("cargo ktstr kernel clean --corrupt-only"),
            "summary must name the surgical-cleanup command: {one}",
        );
        // Existing detail paragraph must still follow — same content
        // as `format_corrupt_footer` on its own.
        assert!(
            one.contains(&format_corrupt_footer(root)),
            "footer must embed the full format_corrupt_footer \
             detail after the count summary: {one}",
        );

        let many = corrupt_footer_if_any(3, root).expect("positive count must yield Some(footer)");
        assert!(
            many.contains("3 corrupt entries."),
            "plural form (count > 1) must render as `N corrupt entries.`; got: {many}",
        );
    }

    /// Pins the design decision that `(corrupt)` is NOT in the
    /// one-line legend family — the footer carries the
    /// legend-equivalent first sentence AND the operational
    /// remediation block. Positive content assertions only; this
    /// test cannot (and does not try to) enforce the absence of a
    /// future `CORRUPT_EXPLANATION` const.
    ///
    /// The invariant the test pins: the footer's FIRST SENTENCE
    /// names the tag, states the unusable meaning, and enumerates
    /// the three corruption modes — so a reader who sees
    /// `(corrupt)` in a row finds the definition in the first line
    /// of the footer, not buried after the remediation block. A
    /// future reword that moves the legend content elsewhere (e.g.
    /// appends it after the command list) would force readers to
    /// scan past operational guidance to find the meaning — this
    /// test catches that.
    ///
    /// The remediation elements that justify the footer format
    /// over a one-line legend const are also pinned: the cache-root
    /// path (the runtime-interpolation constraint that actually
    /// forces the footer) and all three `kernel clean` command
    /// variants — the surgical `--corrupt-only --force` (leaves
    /// valid entries alone) and the two escalation paths `--force`
    /// (removes ALL) and `--keep N --force` (preserves N newest).
    /// A regression that dropped any of them invalidates the
    /// rationale for keeping corrupt out of the legend family and
    /// trips here.
    #[test]
    fn corrupt_footer_is_self_documenting() {
        let root = std::path::Path::new("/tmp/ktstr-cache-test-root");
        let footer = format_corrupt_footer(root);
        // First sentence = legend. A sentence boundary is the
        // period followed by a space; split once so a future
        // multi-paragraph footer still pins the first sentence
        // against the same invariants. Use `.expect()` — a footer
        // that fails to end its first sentence with `". "` breaks
        // the invariant (the legend would no longer be separable),
        // so failing loudly at this split is the right default;
        // a silent `unwrap_or(&footer)` would let the whole footer
        // satisfy the downstream assertions and hide the
        // regression.
        let first_sentence = footer
            .split_once(". ")
            .map(|(head, _)| head)
            .expect("footer must terminate legend sentence with period-space");
        assert!(
            first_sentence.contains("(corrupt)"),
            "first sentence must name the tag so a reader who sees \
             `(corrupt)` in a row finds the definition in the \
             first line of the footer, not buried after the \
             remediation block; got: {first_sentence:?}",
        );
        assert!(
            first_sentence.contains("cannot be used"),
            "first sentence must carry the definitional meaning \
             (legend-equivalent wording); got: {first_sentence:?}",
        );
        // Reasons a cache entry is corrupt — the three the JSON
        // schema (kernel_list corrupt-entry shape) enumerates:
        // metadata-missing, metadata-malformed, image-missing.
        // Tokens chosen to be distinct substrings of the actual
        // legend wording so each mode is independently guarded:
        // "metadata is missing" anchors to the
        // "cached metadata is missing" clause, "malformed"
        // stands alone, and "missing image" anchors to the
        // "references a missing image" tail. A bare `"missing"`
        // would match two clauses simultaneously and mask a
        // regression that dropped one of them.
        for reason_token in ["metadata is missing", "malformed", "missing image"] {
            assert!(
                first_sentence.contains(reason_token),
                "legend sentence must enumerate corruption modes; \
                 expected `{reason_token}`, got: {first_sentence:?}",
            );
        }
        // Operational elements that justify the footer format
        // over a one-line legend const. If any of these disappear,
        // the design decision (exclude from legend family) loses
        // its rationale and the task should be revisited.
        assert!(
            footer.contains(&root.display().to_string()),
            "footer must surface the cache-root path verbatim so \
             operators know which directory to inspect; got: \
             {footer:?}",
        );
        assert!(
            footer.contains("kernel clean --corrupt-only --force"),
            "footer must name the `kernel clean --corrupt-only \
             --force` surgical variant — the zero-risk option for \
             operators with valid alongside corrupt entries; got: \
             {footer:?}",
        );
        assert!(
            footer.contains("kernel clean --force"),
            "footer must name the `kernel clean --force` escalation \
             variant (removes ALL entries, valid and corrupt); \
             got: {footer:?}",
        );
        assert!(
            footer.contains("kernel clean --keep N --force"),
            "footer must name the `kernel clean --keep N --force` \
             escalation variant (preserves the N newest entries) \
             alongside the surgical `--corrupt-only --force` and \
             the broader `--force`, so every operator position \
             (corrupt-only, preserve-N-newest, everything) has a \
             documented command; got: {footer:?}",
        );
        assert!(
            footer.contains("ALL cached entries"),
            "footer must carry the safety wording that distinguishes \
             the surgical `--corrupt-only --force` (leaves valid \
             entries alone) from its escalation paths `--force` \
             (removes ALL) and `--keep N --force` (preserves N \
             newest); got: {footer:?}",
        );

        // Command ordering: `--corrupt-only --force` must appear
        // FIRST so an operator scanning top-to-bottom reaches the
        // zero-risk surgical option before the escalation paths.
        // Doc on `format_corrupt_footer` calls this out explicitly;
        // pin it here so a future reword that re-orders the three
        // commands trips before landing. Each assertion uses
        // `<` rather than `<=` to require strict precedence —
        // identical positions are impossible (the substrings
        // differ) but the strict form makes the intent obvious to
        // a reader.
        let pos_corrupt_only = footer
            .find("kernel clean --corrupt-only --force")
            .expect("`--corrupt-only --force` must appear in footer");
        let pos_force = footer
            .find("kernel clean --force")
            .expect("`--force` must appear in footer");
        let pos_keep = footer
            .find("kernel clean --keep N --force")
            .expect("`--keep N --force` must appear in footer");
        assert!(
            pos_corrupt_only < pos_force,
            "`--corrupt-only --force` must precede `--force` in the footer \
             (surgical option goes before escalation); got positions \
             corrupt_only={pos_corrupt_only}, force={pos_force} in: {footer:?}",
        );
        assert!(
            pos_force < pos_keep,
            "`--force` must precede `--keep N --force` in the footer so the \
             escalation path reads in widening order (surgical → broadest \
             → preserve-N); got positions force={pos_force}, keep={pos_keep} \
             in: {footer:?}",
        );
    }

    /// `DIRTY_TREE_CACHE_SKIP_HINT` is the exact text the
    /// `kernel build` path prints after the `{cli_label}: ` prefix
    /// when the local tree is dirty. Pinning the shape here means a
    /// reword that drops either the cache-skip reason or the
    /// remediation path trips this test instead of slipping into a
    /// release. The three actionable elements — the "skipping cache"
    /// preamble, the dirty-tree cause, and the commit/stash remedy —
    /// are each asserted so a partial truncation (e.g. losing the
    /// remedy) is caught.
    #[test]
    fn dirty_tree_cache_skip_hint_shape() {
        assert!(
            DIRTY_TREE_CACHE_SKIP_HINT.contains("skipping cache"),
            "dirty-tree hint must name the cache-skip outcome: {DIRTY_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            DIRTY_TREE_CACHE_SKIP_HINT.contains("uncommitted changes"),
            "dirty-tree hint must name the cause: {DIRTY_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            DIRTY_TREE_CACHE_SKIP_HINT.contains("commit")
                && DIRTY_TREE_CACHE_SKIP_HINT.contains("stash"),
            "dirty-tree hint must name the commit-or-stash remediation: {DIRTY_TREE_CACHE_SKIP_HINT}",
        );
    }

    /// `NON_GIT_TREE_CACHE_SKIP_HINT` is the hint fired when the
    /// local source tree is not a git repository — `commit` / `stash`
    /// do not apply, so the wording must avoid them and instead
    /// point to the actionable remediations: put the source under
    /// git, OR switch to a content-keyed fetch mode (`kernel build
    /// VERSION` for tarball, `kernel build --git URL --ref REF` for
    /// shallow clone) that does not need dirty-state detection.
    /// Pinning the shape catches a reword that (a) drops the
    /// "skipping cache" preamble, (b) invents a commit/stash
    /// remediation that's not actionable, or (c) regresses to a
    /// vague "use a kernel fetch mode that produces a git-tracked
    /// tree" pointer that leaves the operator guessing which CLI
    /// invocation actually fixes it.
    #[test]
    fn non_git_tree_cache_skip_hint_shape() {
        assert!(
            NON_GIT_TREE_CACHE_SKIP_HINT.starts_with("skipping cache"),
            "non-git hint must be left-anchored on the cache-skip outcome: {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            NON_GIT_TREE_CACHE_SKIP_HINT.contains("not a git repository"),
            "non-git hint must name the cause: {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            NON_GIT_TREE_CACHE_SKIP_HINT.contains("put the source under git"),
            "non-git hint must name the actionable remediation: {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            NON_GIT_TREE_CACHE_SKIP_HINT.contains("kernel build VERSION"),
            "non-git hint must name the concrete tarball-fetch alternative: {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            NON_GIT_TREE_CACHE_SKIP_HINT.contains("kernel build --git URL --ref REF"),
            "non-git hint must name the concrete git-clone alternative: {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            !NON_GIT_TREE_CACHE_SKIP_HINT.contains("stash"),
            "non-git hint must NOT suggest stash (no git = no stash): {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
        assert!(
            !NON_GIT_TREE_CACHE_SKIP_HINT.contains("commit"),
            "non-git hint must NOT suggest committing existing tree changes (no git = no commit): {NON_GIT_TREE_CACHE_SKIP_HINT}",
        );
    }

    /// `show_thresholds` happy path: a known registered test resolves
    /// to a Result carrying both the `Test:` and `Scheduler:` header
    /// lines plus the human-formatted threshold dump from
    /// `Assert::format_human`. Pins the composition order — header
    /// lines before the threshold block — so a future reorder that
    /// mixed the headers into the field rows would trip this test.
    #[test]
    fn show_thresholds_known_test_returns_populated_report() {
        // Pick the first registered test — any KTSTR_TESTS entry is
        // an acceptable target for the shape assertions below, and
        // iterating avoids hardcoding a specific test name that
        // could be renamed later.
        let Some(entry) = crate::test_support::KTSTR_TESTS.iter().next() else {
            // No registered tests in this build — skip without
            // failing (some lean build configs exclude the
            // test-registry).
            eprintln!(
                "ktstr: SKIP: show_thresholds_known_test_returns_populated_report — no entries in KTSTR_TESTS",
            );
            return;
        };
        let out = show_thresholds(entry.name).expect("show_thresholds must resolve known test");
        assert!(
            out.contains("Test:"),
            "output missing `Test:` header: {out}",
        );
        assert!(
            out.contains("Scheduler:"),
            "output missing `Scheduler:` header: {out}",
        );
        assert!(
            out.contains("Resolved assertion thresholds:"),
            "output missing thresholds section: {out}",
        );
        // Header precedes threshold block (composition order).
        let test_idx = out.find("Test:").unwrap();
        let thresholds_idx = out.find("Resolved assertion thresholds:").unwrap();
        assert!(
            test_idx < thresholds_idx,
            "`Test:` header must precede threshold dump",
        );
    }

    /// `show_thresholds` error path: an unknown test name surfaces
    /// the actionable `no registered ktstr test named` diagnostic
    /// plus the operator hint pointing at `cargo nextest list` and
    /// the `<binary>::` prefix caveat. Pins the wording so a
    /// reword that drops the name or the nextest pointer fails
    /// this test.
    #[test]
    fn show_thresholds_unknown_test_returns_actionable_error() {
        let err = show_thresholds("definitely_not_a_registered_test_xyz123").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no registered ktstr test named"),
            "error must name the missing-test condition: {msg}",
        );
        assert!(
            msg.contains("cargo nextest list"),
            "error must point at the discovery command: {msg}",
        );
        assert!(
            msg.contains("function-name component"),
            "error must flag the nextest binary:: prefix caveat: {msg}",
        );
    }

    /// `suggest_closest_test_name` — positive case: a query that is
    /// a one-character typo of a registered test returns that
    /// registered name. Uses the linkme distributed slice directly
    /// to pick a real entry, then mutates a single byte to guarantee
    /// edit distance 1 (well inside the `max(3, len/3)` threshold).
    ///
    /// Picks a long name (> 9 chars) so the absolute-3 floor is NOT
    /// the binding constraint on the len/3 formula either way —
    /// either branch correctly admits distance-1.
    #[test]
    fn suggest_closest_test_name_finds_near_match() {
        let Some(entry) = crate::test_support::KTSTR_TESTS.iter().find(|e| {
            e.name.len() >= 10 && !(e.name.starts_with("__unit_test_") && e.name.ends_with("__"))
        }) else {
            skip!(
                "no registered non-sentinel test with name >= 10 chars \
                 — cannot construct a positive strsim probe"
            );
        };
        // Mutate one byte to a different ASCII letter to produce
        // edit distance 1 without colliding with another registered
        // name.
        let mut mutated: Vec<u8> = entry.name.bytes().collect();
        mutated[0] = if mutated[0] == b'z' { b'a' } else { b'z' };
        let query = std::str::from_utf8(&mutated).expect("ASCII mutation stays UTF-8");
        let suggestion = suggest_closest_test_name(query)
            .expect("distance-1 typo on a registered name must yield a suggestion");
        assert_eq!(
            suggestion, entry.name,
            "a single-byte typo must suggest the exact name it was derived from",
        );
    }

    /// `suggest_closest_test_name` — negative case: a query totally
    /// unrelated to any registered name must return `None` instead
    /// of over-suggesting a distant match. A 40-char random ASCII
    /// string has no expected Levenshtein relationship to the
    /// snake_case test names in `KTSTR_TESTS`, so the threshold
    /// `max(3, len/3) == 13` filters every candidate out.
    #[test]
    fn suggest_closest_test_name_returns_none_for_unrelated_query() {
        // 40 chars of `x` — distance to any real snake_case test
        // name is dominated by the cardinality difference (most
        // test names share few chars with a uniform string), so
        // every candidate exceeds the threshold and the helper
        // correctly declines.
        let unrelated = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        assert_eq!(
            suggest_closest_test_name(unrelated),
            None,
            "a query with no lexical relationship to any registered \
             test name must yield no suggestion (not an over-reach)",
        );
    }

    /// `suggest_closest_test_name` — boundary case: a query whose
    /// edit distance EXACTLY matches the threshold still yields a
    /// suggestion. Constructs a 12-char query from a registered
    /// name by flipping 4 bytes: distance = 4, threshold for a
    /// 12-char query = `max(3, 12/3) = 4`, so the candidate just
    /// fits.
    ///
    /// Guards against an off-by-one that changes `<=` to `<` on the
    /// threshold check, which would silently drop boundary matches
    /// and make the helper stricter than documented.
    #[test]
    fn suggest_closest_test_name_accepts_at_threshold_boundary() {
        // Look for a registered test name with length >= 12 so we
        // can take its 12-char prefix and mutate 4 bytes.
        let Some(entry) = crate::test_support::KTSTR_TESTS.iter().find(|e| {
            e.name.len() >= 12 && !(e.name.starts_with("__unit_test_") && e.name.ends_with("__"))
        }) else {
            skip!(
                "no registered non-sentinel test with name >= 12 chars \
                 — cannot construct a boundary strsim probe"
            );
        };
        // Preserve entry.name's full length so the helper doesn't
        // fall back to a longer/shorter registered name with smaller
        // distance. Flip exactly 4 bytes at the START (positions
        // chosen to avoid `_` separators that would land on many
        // snake_case names at predictable offsets).
        let mut mutated: Vec<u8> = entry.name.bytes().collect();
        // Flip 4 distinct positions to ASCII letters that differ
        // from the original at each index. Using 2, 4, 6, 8 — all
        // likely to be alphanumeric in a snake_case test name.
        for &pos in &[2usize, 4, 6, 8] {
            if pos >= mutated.len() {
                skip!("entry.name too short for boundary probe");
            }
            mutated[pos] = if mutated[pos] == b'z' { b'a' } else { b'z' };
        }
        let query = std::str::from_utf8(&mutated).expect("ASCII mutation stays UTF-8");
        // Guard: the 4-byte flip above MUST produce exactly
        // distance-4 against the source name. If the registered
        // set contains a near-neighbor name that happens to be
        // closer to `query`, the suggestion could drift and the
        // test would false-fail — skip in that case rather than
        // ship a flaky assertion.
        if strsim::levenshtein(query, entry.name) != 4 {
            skip!("mutation did not produce distance-4 against source");
        }
        let threshold = std::cmp::max(3, query.len() / 3);
        assert_eq!(
            threshold, 4,
            "boundary test presumes threshold == 4 for 12-char query; \
             got {threshold}. If query length changed, update the \
             test OR the mutation count to maintain the boundary.",
        );
        let suggestion = suggest_closest_test_name(query)
            .expect("distance equal to threshold must still yield a suggestion");
        // The suggestion may legitimately be a CLOSER name than
        // `entry.name` if the registry contains one within
        // distance 4 of `query`; the load-bearing invariant is
        // "SOME suggestion is emitted at the threshold boundary",
        // not "the suggestion matches our mutated source".
        assert!(
            !suggestion.is_empty(),
            "boundary-distance query must yield a non-empty suggestion",
        );
    }

    /// Sibling of `suggest_closest_test_name_finds_near_match` for
    /// the scenario-registry helper. A single-byte typo of a
    /// registered scenario name must round-trip to the exact
    /// scenario name through the Levenshtein match. Uses
    /// `all_scenarios()` so the test is data-driven against the
    /// live registry rather than a hardcoded name that could drift.
    #[test]
    fn suggest_closest_scenario_name_finds_near_match() {
        let scenarios = crate::scenario::all_scenarios();
        let Some(s) = scenarios.iter().find(|s| s.name.len() >= 10) else {
            skip!(
                "no registered scenario with name >= 10 chars — cannot \
                 construct a positive strsim probe"
            );
        };
        // Flip one byte at position 0 to an ASCII letter different
        // from the original. Guarantees distance 1, which is well
        // inside `max(3, len/3)` for a ≥10-char name.
        let mut mutated: Vec<u8> = s.name.bytes().collect();
        mutated[0] = if mutated[0] == b'z' { b'a' } else { b'z' };
        let query = std::str::from_utf8(&mutated).expect("ASCII mutation stays UTF-8");
        let suggestion = suggest_closest_scenario_name(query)
            .expect("distance-1 typo of a registered scenario must yield a suggestion");
        assert_eq!(
            suggestion, s.name,
            "single-byte typo must resolve back to the exact scenario name",
        );
    }

    /// Unrelated query → None. Parallel to the test-name sibling.
    /// A 40-char uniform string exceeds `max(3, 40/3) == 13` edits
    /// against every snake_case scenario name, so the helper
    /// correctly declines to emit a suggestion instead of reaching
    /// for the nearest-but-distant candidate.
    #[test]
    fn suggest_closest_scenario_name_returns_none_for_unrelated_query() {
        let unrelated = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        assert_eq!(
            suggest_closest_scenario_name(unrelated),
            None,
            "a query unrelated to every registered scenario must yield \
             no suggestion (no over-reach)",
        );
    }

    /// Empty registry edge case — if `all_scenarios()` is empty in
    /// this build, the helper must return None cleanly rather than
    /// panicking. The `all_scenarios` registry currently has many
    /// entries, so this test is guarded to only assert the "no
    /// panic" contract (any result is acceptable for the empty
    /// path; the scenarios-present path is covered by the
    /// near-match test above).
    #[test]
    fn suggest_closest_scenario_name_handles_any_registry_size() {
        let _ = suggest_closest_scenario_name("arbitrary");
    }

    /// `scenario_filter_hint` wraps the suggestion in the ` Did you
    /// mean \`<name>\`?` suffix shape that `filter_scenarios` and
    /// the `ktstr list` empty-output path both consume. Pins the
    /// format so a reword to `Perhaps you meant` or dropping the
    /// leading space lands here first, before the CLI surfaces
    /// drift away from each other.
    #[test]
    fn scenario_filter_hint_formats_suffix_on_near_match() {
        let scenarios = crate::scenario::all_scenarios();
        let Some(s) = scenarios.iter().find(|s| s.name.len() >= 10) else {
            skip!("no registered scenario with name >= 10 chars");
        };
        let mut mutated: Vec<u8> = s.name.bytes().collect();
        mutated[0] = if mutated[0] == b'z' { b'a' } else { b'z' };
        let query = std::str::from_utf8(&mutated).unwrap();
        let hint = scenario_filter_hint(query).expect("near match must produce a hint");
        assert!(
            hint.starts_with(" Did you mean `"),
            "hint must start with ` Did you mean \\`` prefix: {hint}",
        );
        assert!(
            hint.ends_with("`?"),
            "hint must end with the backtick-close + question mark: {hint}",
        );
        assert!(
            hint.contains(s.name),
            "hint must embed the matched scenario name: {hint}",
        );
    }

    /// Negative case for the public hint wrapper: an unrelated
    /// query yields `None` rather than a blank suffix. The caller
    /// in `ktstr list` conditions its whole eprintln line on this
    /// `Option`, so a spurious `Some("")` would print a dangling
    /// "Did you mean `?" line. Pin the None result explicitly.
    #[test]
    fn scenario_filter_hint_returns_none_on_unrelated_query() {
        assert!(
            scenario_filter_hint("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx").is_none(),
            "unrelated query must produce no hint, not a blank suffix",
        );
    }

    /// End-to-end pin on the `filter_scenarios` error path: a
    /// filter string that's a close typo of a registered scenario
    /// must surface a "Did you mean `<name>`?" snippet embedded in
    /// the anyhow bail message. Operator's first-line error text
    /// goes from generic ("no scenarios matched filter") to
    /// specific (names the likely intended scenario) without them
    /// having to cross-reference `ktstr list`.
    #[test]
    fn filter_scenarios_empty_match_includes_did_you_mean_hint() {
        let scenarios = crate::scenario::all_scenarios();
        let Some(s) = scenarios.iter().find(|s| s.name.len() >= 10) else {
            skip!("no registered scenario with name >= 10 chars");
        };
        // Build a query that's a single-byte typo — guaranteed
        // inside the Levenshtein threshold — AND that doesn't
        // substring-match any scenario name (so the filter path
        // actually reaches the empty-match bail branch). The
        // single-byte mutation at position 0 produces a string
        // that cannot substring-match the source, since `contains`
        // on the unmutated name would require the full original
        // byte at position 0 to appear somewhere in the query.
        let mut mutated: Vec<u8> = s.name.bytes().collect();
        mutated[0] = if mutated[0] == b'z' { b'a' } else { b'z' };
        let query = std::str::from_utf8(&mutated).unwrap().to_string();
        // Sanity: no registered scenario contains the mutated
        // query as a substring — otherwise the filter would pass
        // through and we'd never see the bail message. A legitimate
        // collision is possible in theory (another scenario shares
        // a suffix with the mutated query) but exceedingly unlikely
        // given the distinct snake_case prefixes in the catalog.
        // Skip if we hit one so the test doesn't flake on a future
        // catalog addition.
        if scenarios.iter().any(|sc| sc.name.contains(&query)) {
            skip!(
                "mutated query accidentally substring-matches a \
                 registered scenario; cannot exercise the bail branch"
            );
        }
        let err =
            filter_scenarios(&scenarios, Some(&query)).expect_err("non-matching filter must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no scenarios matched filter"),
            "bail must name the condition: {msg}",
        );
        assert!(
            msg.contains("Did you mean"),
            "bail must include the strsim suggestion on a near match: {msg}",
        );
        assert!(
            msg.contains(s.name),
            "bail must name the suggested scenario: {msg}",
        );
    }

    /// filter_scenarios with a totally-unrelated filter still
    /// bails, but the Did-you-mean suffix is absent — the message
    /// degrades back to the generic "run 'ktstr list'" redirect
    /// rather than over-suggesting a distant candidate.
    #[test]
    fn filter_scenarios_unrelated_filter_bails_without_hint() {
        let scenarios = crate::scenario::all_scenarios();
        let err = filter_scenarios(&scenarios, Some("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"))
            .expect_err("unrelated filter must bail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no scenarios matched filter"),
            "bail must still name the condition: {msg}",
        );
        assert!(
            !msg.contains("Did you mean"),
            "unrelated filter must NOT over-suggest a distant match: {msg}",
        );
        assert!(
            msg.contains("ktstr list"),
            "bail must fall back to the generic 'ktstr list' pointer: {msg}",
        );
    }

    /// The `UNTRACKED_KCONFIG_EXPLANATION` must reference the tag word
    /// `untracked kconfig` verbatim so the legend under the table
    /// matches the per-row tag produced by `format_entry_row`
    /// (which formats via `KconfigStatus`'s Display impl at
    /// cache.rs). If the Display word drifts, this test catches
    /// the mismatch before the CLI ships a legend that doesn't
    /// describe what users see.
    #[test]
    fn untracked_legend_names_the_tag_word() {
        assert!(
            UNTRACKED_KCONFIG_EXPLANATION.contains("(untracked kconfig)"),
            "legend must name the tag it explains: {UNTRACKED_KCONFIG_EXPLANATION}",
        );
    }

    /// Snapshot pin for `format_entry_row` across the 6-case outcome
    /// matrix over (EOL, not-EOL) × (Matches, Stale, Untracked);
    /// empty and unparseable `active_prefixes` branches are pinned by
    /// sibling `is_eol_` tests. A 7th case fixes the `version == "-"`
    /// short-circuit at cli.rs where a missing version skips the EOL
    /// tag even under a non-empty active list. A 10th case (c10) pins
    /// the end-to-end RC-version render: an entry whose version
    /// carries an `-rc` suffix must have that suffix stripped by
    /// [`version_prefix`] before the active-prefix compare, so an RC
    /// whose stripped series IS active emits no `(EOL)` tag. Sibling
    /// unit tests (`version_prefix_strips_rc_suffix`,
    /// `is_eol_rc_suffix_mismatch_does_not_flag`) pin the pieces; c10
    /// pins the assembled behavior through `format_entry_row` itself
    /// so a regression that bypasses `is_eol` in the render path is
    /// still caught. An 11th case (c11) pins the complementary
    /// negative: an RC whose stripped series is NOT in the active
    /// list must emit `(EOL)`, catching a regression that skips
    /// the active-list compare entirely for RC versions (e.g. an
    /// early `if has_rc(v) { return false }` short-circuit).
    /// c10 and c11 guard different regressions: c10 catches
    /// "suffix left attached to the compare key" (6.14-rc2 would
    /// miss the 6.14 prefix), c11 catches "RC compare skipped"
    /// (since 7.0 is absent from the active list either way, only
    /// a skipped compare suppresses its EOL tag).
    ///
    /// Inline snapshot captures exact padding and tag ordering so any
    /// drift — column width change, tag reorder, `(EOL)` string
    /// rename, Display-impl tweak on `KconfigStatus` — fails this one
    /// test. Uses `KernelSource::Tarball` because it is the simplest
    /// variant to construct; `Display` on `KernelSource` strips
    /// payload fields for every variant, so source choice only
    /// affects the rendered column when the Display impl changes.
    /// Fixed `built_at` timestamp keeps the snapshot date-stable.
    /// Key lengths vary (10-20 chars) across c1-c7, c10, and c11;
    /// all fit within the 48-char column, so padding drift surfaces
    /// at multiple pad counts. c8 and c9 pin column-boundary
    /// behavior: c8 is exactly 48 chars to verify the `:<48` pad
    /// emits no truncation at the column edge, and c9 is 59 chars
    /// so the key overflows the nominal column to stress the
    /// min-width (not fixed-width) semantics.
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

        // version "6.14.2" is in active list → not EOL; version
        // "2.6.32" is long-EOL; version None → rendered as "-" and
        // short-circuits the EOL guard regardless of active list.
        // entry_hash == current → Matches; entry_hash != current →
        // Stale; None → Untracked.
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

        // c8 is exactly 48 chars to pin column-boundary behavior:
        // `format!("{:<48}", key)` emits the key with exactly ONE
        // trailing space before `version`, matching the normal-
        // padding rows. Any regression that shrinks the `:<48` pad
        // or applies a hidden truncation surfaces on this row.
        //
        // c9 is 59 chars to pin overflow behavior: Rust's `:<48`
        // pads to AT LEAST 48 chars without truncating, so the long
        // key spills past the nominal column. Regressing to
        // `{:48.48}` (which would truncate) fails this snapshot.
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
            // c10: RC version whose stripped series (`6.14`) is in
            // `active_prefixes` — must render the raw `6.14-rc2`
            // string in the version column with NO `(EOL)` tag.
            // Proves `version_prefix` strips the `-rc2` suffix inside
            // `format_entry_row`'s `entry_is_eol` call, not only in
            // the unit-tested helper.
            build_row("c10-active-rc", Some("6.14-rc2"), Some(current_hash)),
            // c11: RC version whose stripped series (`7.0`) is NOT
            // in `active_prefixes` — must render the raw `7.0-rc1`
            // string AND carry the `(EOL)` tag. Guards specifically
            // against a regression that skips the active-list
            // compare for RC versions (e.g. an early-return
            // `if has_rc(v) { return false }` in the EOL path).
            // "Suffix left attached" regressions are c10's job —
            // c11 does not detect them because `7.0` is absent
            // from the active list regardless of whether the
            // compare key is `7.0` or `7.0-rc1`.
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

    /// Regression pin for `format_entry_row` with empty
    /// `active_prefixes` — the fallback path `kernel_list` enters
    /// when [`fetch_active_prefixes`] returns `Err`. The `(EOL)` tag
    /// must not appear on the rendered row regardless of how old the
    /// entry's version is, since "fetch failed" is an
    /// unknown-active-list signal, not a universal-EOL signal.
    /// Cross-checked against the non-empty branch so the suppression
    /// is owned by the empty-slice fallback, not by some other code
    /// path that happens to be quiet on this fixture.
    #[test]
    fn format_entry_row_empty_active_prefixes_does_not_tag_eol() {
        use crate::cache::{CacheArtifacts, CacheDir, KernelMetadata, KernelSource};

        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join("bzImage");
        std::fs::write(&image, b"fake kernel").unwrap();
        // Ancient long-EOL version: if the active list contained only
        // modern series, is_eol would return true. The empty slice
        // forces the guard branch instead.
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
        assert!(
            !row_fallback.contains("(EOL)"),
            "empty active_prefixes (fetch-failed fallback) must not tag any entry EOL, \
             got row: {row_fallback:?}",
        );

        // Sanity: the same entry IS tagged EOL when a non-empty active
        // list excludes its prefix. Confirms the suppression above
        // flows through the empty-slice guard, not some unrelated
        // short-circuit.
        let row_with_active = format_entry_row(&entry, "kconfig_hash", &["6.14".to_string()]);
        assert!(
            row_with_active.contains("(EOL)"),
            "non-empty active_prefixes excluding entry's prefix must tag EOL, \
             got row: {row_with_active:?}",
        );
    }

    /// Tag-ordering invariant: when a row carries BOTH a kconfig-state
    /// tag (`(stale kconfig)` / `(untracked kconfig)`) AND the
    /// `(EOL)` tag, the kconfig tag must appear FIRST. Dual-tag
    /// snapshot rows in
    /// [`format_entry_row_renders_eol_kconfig_matrix`] pin exact
    /// column widths AND tag sequence together — a column-spacing
    /// change forces re-snapshotting, which could accidentally
    /// hide a reordered-tag regression. This test isolates the
    /// order invariant so it stays pinned independent of snapshot
    /// cadence: a tag swap surfaces here even if the matrix
    /// snapshot is re-blessed for an unrelated column-width tweak.
    ///
    /// Exercises both dual-tag combinations (stale+EOL and
    /// untracked+EOL) because the two kconfig-tag variants are
    /// produced by different branches of [`KconfigStatus`] — a
    /// regression that reordered only one of them (e.g. swapped
    /// the push order in a match arm) would leak past a
    /// single-variant test.
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

        // stale kconfig + EOL: entry hash differs from current AND
        // entry version is outside the active-series list.
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
        assert!(
            stale_idx < eol_idx,
            "(stale kconfig) must precede (EOL) in the rendered row — a \
             regression that reordered the two tags would break operator \
             grep pipelines that key on the kconfig-tag being the first \
             tag on a row:\n{stale_row}",
        );

        // untracked kconfig + EOL: separate branch of KconfigStatus,
        // same ordering contract.
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
        assert!(
            untracked_idx < eol_idx,
            "(untracked kconfig) must precede (EOL) — same ordering \
             contract as the stale branch:\n{untracked_row}",
        );
    }

    // -- partition_clean_candidates fixture coverage -----------------
    //
    // Four-axis matrix: Valid vs Corrupt, keep present vs absent,
    // corrupt_only true vs false, empty vs populated. Builds the
    // `ListedEntry` values directly in the test to avoid touching
    // `$HOME/.cache/ktstr`; the partitioner is a pure function of its
    // inputs.

    fn mk_valid(key: &str) -> crate::cache::ListedEntry {
        use crate::cache::{CacheEntry, KernelMetadata, KernelSource};
        let path = std::path::PathBuf::from(format!("/tmp/fixture/{key}"));
        let metadata = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-22T00:00:00Z".to_string(),
        );
        crate::cache::ListedEntry::Valid(CacheEntry {
            key: key.to_string(),
            path,
            metadata,
        })
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
        // Input order is `cache.list()`'s built_at-desc, so index 0 is
        // newest. keep=2 retains v_new1 + v_new2 (the first two valid
        // entries), removes v_old, and ALWAYS removes the corrupt
        // entry regardless of position.
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
        // Even when keep=3 would "cover" every entry, corrupt ones
        // still surface as removal candidates — they don't consume a
        // keep slot by design.
        let entries = vec![mk_corrupt("c1"), mk_valid("v1"), mk_valid("v2")];
        let out = partition_clean_candidates(&entries, Some(3), false);
        let keys: Vec<&str> = out.iter().map(|e| e.key()).collect();
        assert_eq!(keys, vec!["c1"]);
    }

    /// Defensive cell: (keep=Some, corrupt_only=true). Clap's
    /// `#[arg(long, conflicts_with = "keep")]` on `--corrupt-only`
    /// rejects this combination at parse time (pinned by
    /// `kernel_clean_rejects_corrupt_only_with_keep` below), so the
    /// partitioner never sees it in practice. But `partition_clean_candidates`
    /// is a pure function reachable from any internal caller that
    /// bypasses clap — a future direct caller, a unit test, or a
    /// programmatic entry point. Pin the fallback behavior: when
    /// `corrupt_only=true`, `keep` is inert (valid entries skipped
    /// regardless, corrupt entries removed regardless of keep slot).
    /// This removes the "wait, what would it do?" answer from the
    /// code review for any future call-site that passes both.
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
        assert_eq!(
            keys,
            vec!["c_mid"],
            "corrupt_only=true must make keep inert: valid entries preserved, only corrupt removed",
        );
    }

    // -- clap argument-parse pin: --corrupt-only conflicts_with --keep
    //
    // `#[arg(long, conflicts_with = "keep")]` on the `corrupt_only`
    // field enforces the exclusion at parse time. This fixture pins
    // the invariant so a future refactor that drops or renames the
    // `conflicts_with` attr (or renames the `keep` target) trips a
    // unit-test regression immediately rather than surfacing as a
    // quiet "keep budget silently ignored" at runtime.

    #[test]
    fn kernel_clean_rejects_corrupt_only_with_keep() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let err = TestCli::try_parse_from(["prog", "clean", "--keep", "2", "--corrupt-only"])
            .expect_err("--keep together with --corrupt-only must fail parsing");
        let msg = err.to_string();
        assert!(
            msg.to_ascii_lowercase().contains("cannot be used with")
                || msg.to_ascii_lowercase().contains("conflict"),
            "clap error must surface the conflict between --keep and --corrupt-only, got: {msg}",
        );
    }

    #[test]
    fn kernel_clean_accepts_corrupt_only_alone() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "clean", "--corrupt-only"])
            .expect("--corrupt-only without --keep must parse cleanly");
        match parsed.cmd {
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => {
                assert_eq!(keep, None);
                assert!(!force);
                assert!(corrupt_only);
            }
            other => panic!("expected KernelCommand::Clean, got {other:?}"),
        }
    }

    /// `kernel build --cpu-cap N` parses to `KernelCommand::Build
    /// { cpu_cap: Some(N), .. }`. Unlike the shell subcommand,
    /// kernel-build's `--cpu-cap` has NO `requires` constraint:
    /// builds always use the LLC plan regardless of perf-mode
    /// (builds don't have a perf-mode toggle at all). A clap
    /// regression that accidentally added `requires =
    /// "no_perf_mode"` here would break `ktstr kernel build
    /// --cpu-cap 4` and surface through this test.
    #[test]
    fn kernel_build_parses_cpu_cap_without_extra_flags() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "6.14.2", "--cpu-cap", "4"])
            .expect("kernel build --cpu-cap N must parse");
        match parsed.cmd {
            KernelCommand::Build {
                cpu_cap, version, ..
            } => {
                assert_eq!(cpu_cap, Some(4));
                assert_eq!(version.as_deref(), Some("6.14.2"));
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build` without `--cpu-cap` parses with
    /// `cpu_cap: None` — the "unset" sentinel the downstream planner
    /// expands into the 30%-of-allowed default. Pins the no-flag
    /// path so a future rename of the clap field or a stray
    /// `default_value = "0"` surfaces as a test failure, not a
    /// silent runtime behavior change.
    #[test]
    fn kernel_build_without_cpu_cap_defaults_to_none() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "6.14.2"])
            .expect("kernel build without --cpu-cap must parse");
        match parsed.cmd {
            KernelCommand::Build { cpu_cap, .. } => {
                assert_eq!(cpu_cap, None, "no --cpu-cap must produce None, not Some(0)",);
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build --cpu-cap 0` parses successfully at clap level
    /// — the "must be ≥ 1" check lives in [`CpuCap::new`], not in
    /// the clap value parser. Pins the two-layer validation: clap
    /// accepts any usize; runtime resolution via `CpuCap::resolve`
    /// is responsible for the "0 is rejected" diagnostic. A future
    /// refactor that moved the ≥1 check into clap would trip this
    /// test AND require updating the
    /// `cpu_cap_resolve_zero_env_rejected` error-wording pin.
    #[test]
    fn kernel_build_cpu_cap_zero_passes_clap() {
        use clap::Parser as _;
        #[derive(clap::Parser, Debug)]
        struct TestCli {
            #[command(subcommand)]
            cmd: KernelCommand,
        }
        let parsed = TestCli::try_parse_from(["prog", "build", "6.14.2", "--cpu-cap", "0"])
            .expect("clap-level parse must accept 0; runtime validation rejects");
        match parsed.cmd {
            KernelCommand::Build { cpu_cap, .. } => {
                assert_eq!(
                    cpu_cap,
                    Some(0),
                    "clap parses 0 verbatim; validation is downstream",
                );
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    // Channel-routing and ordering pins previously lived here as
    // `eol_legend_emits_via_eprintln` + `kernel_list_footer_ordering_pin`,
    // scanning cli.rs via `include_str!` + a hand-rolled brace-
    // balanced matcher. Both have moved to
    // `tests/ktstr_cli.rs` as `kernel_list_legends_emit_on_stderr`
    // and `kernel_list_legend_ordering_pins_untracked_stale_corrupt`,
    // which exercise the real `ktstr kernel list` binary against a
    // fixture cache and assert on captured stdout / stderr — the
    // behaviour operators actually observe, not the source form of
    // the code that produces it. The old source-scanning machinery
    // (brace-balance walker, identifier-presence bootstrap) has
    // been removed along with the two tests it supported.

    // ─── `ktstr locks` snapshot + JSON serde pins ────────────────

    /// `LocksSnapshot` JSON top-level keys are stable: `llcs`,
    /// `cpus`, `cache`. Downstream consumers of `ktstr locks --json`
    /// (shell scripts piping through `jq`, the mdbook recipe pages,
    /// future dashboards) parse against these names — a refactor
    /// that renames them would silently break every consumer.
    ///
    /// Also pins the `rename_all = "snake_case"` contract on the
    /// nested row structs: LlcLockRow's "llc_idx" and "numa_node"
    /// must NOT emit as camelCase.
    #[test]
    fn locks_snapshot_json_field_names_are_stable() {
        let snap = LocksSnapshot {
            llcs: vec![LlcLockRow {
                llc_idx: 0,
                numa_node: Some(1),
                lockfile: "/tmp/ktstr-llc-0.lock".to_string(),
                holders: Vec::new(),
            }],
            cpus: vec![CpuLockRow {
                cpu: 3,
                numa_node: None,
                lockfile: "/tmp/ktstr-cpu-3.lock".to_string(),
                holders: Vec::new(),
            }],
            cache: vec![CacheLockRow {
                cache_key: "6.14.2-tarball-x86_64".to_string(),
                lockfile: "/tmp/.locks/6.14.2-tarball-x86_64.lock".to_string(),
                holders: Vec::new(),
            }],
        };
        let val = serde_json::to_value(&snap).expect("serde serialize");
        // Top-level keys.
        assert!(
            val.get("llcs").is_some(),
            "top-level must have 'llcs': {val}"
        );
        assert!(
            val.get("cpus").is_some(),
            "top-level must have 'cpus': {val}"
        );
        assert!(
            val.get("cache").is_some(),
            "top-level must have 'cache': {val}"
        );
        // Nested LLC row.
        let llc0 = &val["llcs"][0];
        assert!(
            llc0.get("llc_idx").is_some(),
            "llc_idx (snake_case): {llc0}"
        );
        assert!(llc0.get("numa_node").is_some(), "numa_node: {llc0}");
        assert!(llc0.get("lockfile").is_some(), "lockfile: {llc0}");
        assert!(llc0.get("holders").is_some(), "holders: {llc0}");
        // Nested CPU row.
        let cpu0 = &val["cpus"][0];
        assert!(cpu0.get("cpu").is_some());
        assert!(cpu0.get("numa_node").is_some());
        // Nested Cache row — cache_key stays snake_case.
        let cache0 = &val["cache"][0];
        assert!(cache0.get("cache_key").is_some(), "cache_key: {cache0}");
    }

    /// `collect_locks_snapshot_from` on a fresh tempdir with no
    /// ktstr lockfiles returns an empty LocksSnapshot (all three
    /// row vectors empty). Production wrapper always sees the same
    /// behavior when `/tmp` has no `ktstr-*.lock` files and the
    /// cache dir has no `.locks/` subdirectory.
    #[test]
    fn collect_locks_snapshot_empty_roots() {
        use tempfile::TempDir;
        let tmp_dir = TempDir::new().expect("tempdir tmp_root");
        let cache_dir = TempDir::new().expect("tempdir cache_root");
        let snap = collect_locks_snapshot_from(tmp_dir.path(), Some(cache_dir.path()))
            .expect("collect must succeed on empty roots");
        assert!(snap.llcs.is_empty(), "no ktstr-llc-*.lock → empty llcs");
        assert!(snap.cpus.is_empty(), "no ktstr-cpu-*.lock → empty cpus");
        assert!(snap.cache.is_empty(), "no .locks/ → empty cache");
    }

    /// `collect_locks_snapshot_from` discovers synthetic LLC + CPU
    /// lockfiles placed under the injected tmp_root. Also pins:
    /// llc_idx / cpu parse from filename stem, sort ascending,
    /// exclude files whose stem doesn't match the expected
    /// `ktstr-{llc,cpu}-{N}` format.
    #[test]
    fn collect_locks_snapshot_discovers_lockfiles() {
        use tempfile::TempDir;
        let tmp_dir = TempDir::new().expect("tempdir");
        let path = tmp_dir.path();
        // Plant 2 LLC lockfiles (out of order — snapshot must sort
        // ascending), 1 CPU lockfile, and a junk file that mustn't
        // appear in the snapshot.
        std::fs::write(path.join("ktstr-llc-5.lock"), b"").expect("plant llc-5");
        std::fs::write(path.join("ktstr-llc-2.lock"), b"").expect("plant llc-2");
        std::fs::write(path.join("ktstr-cpu-7.lock"), b"").expect("plant cpu-7");
        // Junk: looks close but doesn't match the prefix-N-.lock
        // pattern. The parse::<usize>() on "oops" fails → skip.
        std::fs::write(path.join("ktstr-llc-oops.lock"), b"").expect("plant junk");
        let snap = collect_locks_snapshot_from(path, None).expect("collect must succeed");
        // LLC rows, ascending.
        assert_eq!(snap.llcs.len(), 2);
        assert_eq!(snap.llcs[0].llc_idx, 2, "sort ascending: llc 2 first");
        assert_eq!(snap.llcs[1].llc_idx, 5, "sort ascending: llc 5 second");
        // CPU row.
        assert_eq!(snap.cpus.len(), 1);
        assert_eq!(snap.cpus[0].cpu, 7);
        // Cache row empty because cache_root=None.
        assert!(snap.cache.is_empty());
    }

    // ---------------------------------------------------------------
    // kernel_build_pipeline reservation phase — factored-out
    // `acquire_build_reservation` covers the cpu_cap → acquire →
    // sandbox → make_jobs flow without needing a real kernel source.
    // ---------------------------------------------------------------

    /// Serialize `KTSTR_BYPASS_LLC_LOCKS` env-var mutation across
    /// test threads — same pattern as host_topology's env_lock. Two
    /// parallel tests can't both mutate the same process-wide env
    /// var without coordinating.
    fn bypass_env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// RAII guard for scoped `KTSTR_BYPASS_LLC_LOCKS` mutation.
    /// Caller holds `bypass_env_lock()` before constructing.
    struct BypassGuard;
    impl BypassGuard {
        fn set(value: &str) -> Self {
            // SAFETY: env_lock held by caller; serializes with
            // every other env-mutating test.
            unsafe {
                std::env::set_var("KTSTR_BYPASS_LLC_LOCKS", value);
            }
            BypassGuard
        }
        fn remove() -> Self {
            // SAFETY: caller holds env_lock.
            unsafe {
                std::env::remove_var("KTSTR_BYPASS_LLC_LOCKS");
            }
            BypassGuard
        }
    }
    impl Drop for BypassGuard {
        fn drop(&mut self) {
            // SAFETY: guard lifetime bounded by env_lock held by
            // caller; Drop runs before the mutex guard releases.
            unsafe {
                std::env::remove_var("KTSTR_BYPASS_LLC_LOCKS");
            }
        }
    }

    /// `acquire_build_reservation` with
    /// `KTSTR_BYPASS_LLC_LOCKS=1` + `cpu_cap=None` returns a
    /// no-reservation `BuildReservation`: plan is None, sandbox
    /// is None, make_jobs is None. Pins the "bypass escape hatch
    /// disables both layers" contract at the factored-out entry
    /// point so an integration test can exercise the bypass path
    /// without a real kernel source tree.
    #[test]
    fn acquire_build_reservation_bypass_returns_no_reservation() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let r = acquire_build_reservation("test", None).expect("bypass + no cap must succeed");
        assert!(r.plan.is_none(), "bypass must produce no LLC plan");
        assert!(
            r._sandbox.is_none(),
            "bypass must produce no cgroup sandbox",
        );
        assert!(
            r.make_jobs.is_none(),
            "bypass must fall back to nproc (None signals to caller)",
        );
    }

    /// `acquire_build_reservation` with
    /// `KTSTR_BYPASS_LLC_LOCKS=1` + `cpu_cap=Some(_)` must error
    /// with the "resource contract" substring. Pins the conflict
    /// check at the pipeline's reservation entry point, independent
    /// of the CLI-layer conflict check (separate tests pin the CLI layer).
    #[test]
    fn acquire_build_reservation_bypass_with_cap_errors() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::set("1");
        let cap = crate::vmm::host_topology::CpuCap::new(2).expect("cap=2 valid");
        let err =
            acquire_build_reservation("test", Some(cap)).expect_err("bypass + cap must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("resource contract"),
            "err must name the resource contract: {msg}",
        );
    }

    /// `acquire_build_reservation` without bypass on a functional
    /// sysfs-capable host: returns a `BuildReservation` whose
    /// fields are populated consistently — if `plan` is Some, then
    /// `make_jobs` is Some (same plan), and vice-versa. Pins the
    /// "plan and make_jobs must never diverge" invariant; a future
    /// refactor that built `make_jobs` from something other than
    /// `plan.as_ref()` would drift here.
    ///
    /// Runs on any Linux host with `/sys/devices/system/cpu`; the
    /// host's actual LLC count is irrelevant to the invariant.
    #[test]
    fn acquire_build_reservation_plan_and_make_jobs_consistent() {
        let _lock = bypass_env_lock();
        let _env = BypassGuard::remove();
        match acquire_build_reservation("test", None) {
            Ok(r) => {
                // Invariant: plan.is_some() iff make_jobs.is_some()
                assert_eq!(
                    r.plan.is_some(),
                    r.make_jobs.is_some(),
                    "plan and make_jobs must agree on reservation presence",
                );
                if let (Some(p), Some(jobs)) = (r.plan.as_ref(), r.make_jobs) {
                    assert_eq!(
                        jobs,
                        crate::vmm::host_topology::make_jobs_for_plan(p),
                        "make_jobs must equal make_jobs_for_plan(&plan)",
                    );
                }
                // sandbox presence tracks plan presence.
                assert_eq!(
                    r.plan.is_some(),
                    r._sandbox.is_some(),
                    "sandbox and plan must agree on reservation presence",
                );
            }
            Err(e) => {
                // Sysfs-unreadable host or contested LLCs. Accept
                // either outcome; the test's intent is to pin the
                // invariant in the success case, not force success.
                eprintln!("acquire_build_reservation unavailable on this host: {e:#}");
            }
        }
    }
}
