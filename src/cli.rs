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

/// Re-exports of the dimensional-slicing types used by
/// `cargo-ktstr`'s `BuildCompareFilters::build()` plumbing. The
/// `stats` module is `pub(crate)` (its tabular reporting types
/// have no stable surface yet), but the `cargo-ktstr` binary needs
/// `Dimension` and `derive_slicing_dims` to construct compare
/// requests and to unit-test the filter-builder shape. Same
/// pattern as `CpuCap` above: keep the canonical definitions in
/// `stats` (where the comparison plumbing consumes them
/// internally) and re-export the slim slicing surface through
/// `cli` so the binaries reach them through the public `cli`
/// module.
pub use crate::stats::{Dimension, derive_slicing_dims};

/// Shared `kernel` subcommand tree used by both `ktstr` and
/// `cargo ktstr`. The two binaries embed this as
/// `ktstr kernel <subcmd>` / `cargo ktstr kernel <subcmd>` and
/// dispatch identically; defining the variants once means a new
/// `kernel` subcommand (or a flag change) lands in both surfaces by
/// construction.
#[derive(Subcommand, Debug)]
pub enum KernelCommand {
    /// List cached kernel images, or preview a range expansion
    /// without downloading or building.
    ///
    /// Default mode (no `--range`): walks the local cache and
    /// reports every cached kernel image. `--range START..END`
    /// switches to PREVIEW mode: fetches kernel.org's
    /// `releases.json`, expands the inclusive range against the
    /// `stable` / `longterm` releases, and prints the resulting
    /// version list. Preview mode performs no downloads or builds
    /// and ignores the local cache — operators can use it to
    /// answer "what does `--kernel 6.12..6.16` actually cover?"
    /// before paying the network or cache-store cost of a real
    /// resolve.
    #[command(long_about = KERNEL_LIST_LONG_ABOUT)]
    List {
        /// Output in JSON format for CI scripting.
        #[arg(long)]
        json: bool,
        /// Range preview. When supplied, switches the subcommand
        /// from "list cached kernels" to "fetch releases.json and
        /// print the versions a `START..END` range expands to."
        /// Format: `MAJOR.MINOR[.PATCH][-rcN]..MAJOR.MINOR[.PATCH][-rcN]`,
        /// matching [`crate::kernel_path::KernelId::Range`].
        /// Example: `--range 6.12..6.14` → every stable/longterm
        /// release in `[6.12, 6.14]` inclusive.
        ///
        /// In preview mode the subcommand performs no cache
        /// reads or kernel.org tarball downloads — only the
        /// single `releases.json` fetch that
        /// [`crate::cli::expand_kernel_range`] already runs for
        /// real range resolves. `--json` (when also supplied)
        /// emits a JSON object with the literal range string and
        /// the expanded version array; without `--json` the
        /// versions are written one per line to stdout for shell
        /// pipelines.
        #[arg(long)]
        range: Option<String>,
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
/// `cargo ktstr test`, `cargo ktstr coverage`, `cargo ktstr llvm-cov`,
/// and `ktstr shell`. Matches
/// `KernelResolvePolicy { accept_raw_image: false, .. }`.
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
     `6.14` for latest patch), a cache key (see `kernel list`), a \
     version range (`6.12..6.14`), or a git source (`git+URL#REF`). Raw \
     image files are rejected. Source directories auto-build (can be slow \
     on a fresh tree); versions auto-download from kernel.org on cache \
     miss. The flag is REPEATABLE on `test`, `coverage`, and `llvm-cov` \
     — passing multiple `--kernel` flags fans the gauntlet across every \
     resolved kernel; each (test × scenario × topology × flags × kernel) \
     tuple becomes a distinct nextest test case so nextest's parallelism, \
     retries, and `-E` filtering work natively. Ranges expand to every \
     `stable` and `longterm` release inside `[START, END]` inclusive \
     (mainline / linux-next dropped). Git sources clone shallow at the \
     ref and build once. In contrast, `ktstr shell` accepts a single \
     kernel only — pass exactly one `--kernel`.";

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
     falling back to downloading the latest stable kernel. Ranges \
     (`START..END`) and git sources (`git+URL#REF`) are not supported \
     in this context; pass a single kernel.";

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
    "                   ktstr fragment.\n",
    "\n",
    "When --range is set, the subcommand SWITCHES to range-preview\n",
    "mode and emits a structurally different JSON shape — the cache\n",
    "is not walked at all, only kernel.org's releases.json is fetched\n",
    "to expand the inclusive range. The --json output is one object\n",
    "with four top-level fields:\n",
    "\n",
    "  range     literal range string supplied to --range\n",
    "            (e.g. \"6.12..6.14\").\n",
    "  start     parsed start endpoint\n",
    "            (MAJOR.MINOR[.PATCH][-rcN]).\n",
    "  end       parsed end endpoint, same shape as start.\n",
    "  versions  array of resolved version strings inside\n",
    "            [start, end] inclusive, ascending by\n",
    "            (major, minor, patch, rc) tuple. Stable and\n",
    "            longterm releases only — mainline / linux-next\n",
    "            are excluded by the moniker filter.\n",
    "\n",
    "Range-mode output never carries cache metadata\n",
    "(no current_ktstr_kconfig_hash, no entries) — to inspect cached\n",
    "kernels for one of the resolved versions, run `kernel list`\n",
    "without --range. Consumers should dispatch on the presence of\n",
    "the `range` key (range mode) versus `entries` key (list mode)\n",
    "to branch the parse."
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
    kernel_list_inner(json, None)
}

/// Range-preview variant of [`kernel_list`].
///
/// Routes through [`kernel_list_inner`] with `range = Some(spec)`,
/// switching the subcommand from "walk the cache and list local
/// entries" to "fetch releases.json once and print the versions
/// `spec` expands to." See the `range` arg's doc on
/// [`KernelCommand::List`] for operator-facing semantics.
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

/// Run make in a kernel directory under a wall-clock timeout.
///
/// Used for non-build make invocations (`defconfig`, `olddefconfig`,
/// `mrproper`, etc.) where the parent inherits stdout/stderr — the
/// pipe-drained sibling [`run_make_with_output`] handles the full-
/// build path with a separate EOF-driven termination.
///
/// The timeout protects against a wedged make holding the calling
/// pipeline forever. Without it, a stuck `olddefconfig` (e.g. an
/// interactive `conf` prompt that the configure_kernel pre-step
/// failed to bypass, or a kernel-tree inconsistency that wedges
/// `make`) would block the parent process indefinitely. The
/// ceiling is intentionally generous — a single `make defconfig`
/// completes in seconds on any hardware, but large WIP kernel
/// trees with many out-of-tree patches can stretch
/// `mrproper` / `olddefconfig` past the typical seconds-scale; 30
/// minutes covers every legitimate caller while still bounding a
/// genuine wedge.
///
/// Polls `try_wait` at 100ms granularity — small enough that a
/// completed make is reaped within one tick, large enough that
/// the polling itself is not measurable load. On timeout, the
/// child is killed (SIGKILL via `kill_on_drop`-style semantics)
/// and reaped before bailing so no zombie outlives the function.
pub fn run_make(kernel_dir: &Path, args: &[&str]) -> Result<()> {
    const RUN_MAKE_TIMEOUT: Duration = Duration::from_secs(30 * 60);
    // Production poll cadence: small enough that a completed
    // make is reaped within one tick, large enough that the
    // polling itself is not measurable load. Tests pass a
    // sub-millisecond override directly to
    // [`poll_child_with_timeout`] so timeout-fires-and-reaps
    // assertions complete quickly.
    const POLL_INTERVAL: Duration = Duration::from_millis(100);

    let child = std::process::Command::new("make")
        .args(args)
        .current_dir(kernel_dir)
        .spawn()
        .with_context(|| format!("spawn make {}", args.join(" ")))?;

    poll_child_with_timeout(
        child,
        RUN_MAKE_TIMEOUT,
        POLL_INTERVAL,
        &format!("make {}", args.join(" ")),
    )
}

/// Polling-loop body extracted from [`run_make`] so the timeout
/// mechanics can be exercised against synthetic [`std::process::Child`]
/// fixtures with sub-second deadlines (real `make` invocations
/// would burn the full 30-minute production timeout). Production
/// callers funnel through [`run_make`] which spawns `make`,
/// constructs the production deadline, and delegates here.
///
/// `label` is the human-facing name embedded in error messages
/// (e.g. `"make defconfig"`) — pinning a synthetic label in the
/// test surface lets the assertion match the bail wording without
/// depending on `make` being installed on the runner.
///
/// `timeout` is the wall-clock budget AFTER `child` has already
/// spawned (the deadline is computed inside the helper relative
/// to the call instant). `poll_interval` controls the
/// `try_wait` polling cadence — small enough that a completed
/// child is reaped within one tick, large enough that polling
/// itself is not measurable load. Production uses 100ms; tests
/// use 1ms so a sub-second timeout assertion completes quickly.
///
/// On timeout: kill + reap before bailing so no zombie outlives
/// the function. On a `try_wait` error: same kill+reap cleanup
/// before propagating, so a transient probe failure doesn't leak
/// the child.
fn poll_child_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
    poll_interval: Duration,
    label: &str,
) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                anyhow::ensure!(status.success(), "{label} failed");
                return Ok(());
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    // Wedged — kill + reap before bailing so no
                    // zombie persists after we return Err.
                    let _ = child.kill();
                    let _ = child.wait();
                    bail!("{label} timed out after {timeout:?}; child killed");
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                // Reap before propagating so a transient try_wait
                // failure doesn't leak the child.
                let _ = child.kill();
                let _ = child.wait();
                return Err(e).with_context(|| format!("wait on {label}"));
            }
        }
    }
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

    // Build a HashSet of trimmed `.config` lines once, then probe
    // each critical option in O(1). The previous formulation
    // walked `config.lines()` once per critical option (O(N×M)
    // with M≈6 critical options and N≈5000 .config lines), which
    // turned every kernel-build pipeline into a 30K-line scan.
    // Trimming each line matches `all_fragment_lines_present`'s
    // configure-time behavior so the same `.config` parses
    // identically across both checks — without trim, a
    // configure-time write that produced trailing whitespace (or
    // a `.config` edited by hand on a Windows host with `\r\n`
    // line endings) would silently flag every critical option as
    // missing here while passing the configure-time check.
    let existing: std::collections::HashSet<&str> = config.lines().map(str::trim).collect();

    let mut missing = Vec::new();
    for &(option, hint) in VALIDATE_CONFIG_CRITICAL {
        let enabled = format!("{option}=y");
        if !existing.contains(enabled.as_str()) {
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
    /// Whether the source tree was dirty as observed by the build
    /// pipeline. `true` if either the acquire-time inspection
    /// reported dirty OR the post-build re-check observed a
    /// mid-build mutation (worktree edit, branch flip, mid-build
    /// commit). The downstream label decoration in cargo-ktstr's
    /// `resolve_one` uses this to append `_dirty` so a
    /// non-reproducible run is distinguishable from a clean rebuild
    /// of the same path.
    pub post_build_is_dirty: bool,
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

/// Acquire an exclusive flock on a per-source-canonical-path lockfile
/// so two concurrent `cargo ktstr test --kernel <path>` runs against
/// the SAME source tree don't race in `make` (defconfig vs
/// olddefconfig vs compile_commands.json) and stomp each other's
/// `.config` and build artifacts.
///
/// The lockfile lives at
/// `{KTSTR_CACHE_DIR}/.locks/source-{path_hash}.lock` where
/// `{path_hash}` is the full 8-char CRC32 hex of the canonical
/// source-path bytes (same shape and helper the
/// `local-unknown-{path_hash}` cache key uses, see
/// [`crate::fetch::canonical_path_hash`] /
/// [`crate::fetch::compose_local_cache_key`]) — one per-tree
/// identifier ties the source-tree flock to the cache key it gates.
///
/// Lockfile placement piggybacks on the cache root's `.locks/`
/// subdirectory ([`crate::flock::LOCK_DIR_NAME`]) so source-tree
/// flocks share the same filesystem-residency story as cache-entry
/// flocks: never under `/tmp`, where `tmpwatch` (or the equivalent
/// `systemd-tmpfiles` cleanup) can sweep stale-mtime files out from
/// under an active flock holder. flock(2) does NOT update the
/// inode's mtime, so a /tmp-resident lockfile would be a candidate
/// for sweep on every run, with the resulting `unlink(2)` racing
/// any peer trying to `open(2)` the same path. The `.locks/`
/// directory under the user-controlled cache root is exempt from
/// those sweeps.
///
/// Non-blocking — fails fast with an actionable error pointing the
/// operator at `cargo ktstr locks` when a concurrent peer holds the
/// lock. A blocking acquire would silently stall the operator's
/// terminal with no signal why; surfacing the contention immediately
/// lets them inspect peers (or wait deliberately and retry).
///
/// Distinct from the cache-entry flock acquired inside
/// [`crate::cache::CacheDir::store`]: that lock serializes the
/// atomic install of an artifact bundle into a cache slot; this
/// lock serializes the BUILD itself against the source-tree
/// `make` invocations.
pub(crate) fn acquire_source_tree_lock(
    canonical: &Path,
    cli_label: &str,
) -> Result<std::os::fd::OwnedFd> {
    use anyhow::Context;

    // Share the per-path CRC32 with `local-unknown-{hash}` cache
    // keys so a single per-tree identifier ties the source-tree
    // flock to the cache slot it gates.
    let path_hash = crate::fetch::canonical_path_hash(canonical);
    let cache = crate::cache::CacheDir::new()
        .with_context(|| "open cache root for source-tree lockfile placement")?;
    cache
        .ensure_lock_dir()
        .with_context(|| "create cache `.locks/` subdir for source-tree lock")?;
    let lock_path = cache.lock_path(&format!("source-{path_hash}"));

    let fd = crate::flock::try_flock(&lock_path, crate::flock::FlockMode::Exclusive)
        .with_context(|| format!("acquire source-tree flock {}", lock_path.display()))?
        .ok_or_else(|| {
            // Best-effort holder lookup: if /proc/locks reports a
            // peer, surface the pid + cmdline so the operator can
            // identify the conflict without running `cargo ktstr
            // locks` separately. A holder-lookup failure is
            // non-fatal — the EWOULDBLOCK message is already
            // actionable on its own.
            let holders = crate::flock::read_holders(&lock_path).unwrap_or_default();
            let holder_text = if holders.is_empty() {
                String::new()
            } else {
                format!("\n{}", crate::flock::format_holder_list(&holders))
            };
            anyhow::anyhow!(
                "{cli_label}: source tree {} is locked by a concurrent ktstr build \
                 (lockfile {}). Wait for the peer to finish, or run \
                 `cargo ktstr locks` to identify it.{holder_text}",
                canonical.display(),
                lock_path.display(),
            )
        })?;
    Ok(fd)
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
/// `is_local_source` should be true when the source is a local
/// kernel source tree, regardless of how the caller arrived there
/// (`kernel build --source`, `cargo ktstr test --kernel <path>`,
/// or any other Path-spec entry that funnels through
/// [`resolve_kernel_dir`] / [`resolve_kernel_dir_to_entry`]). It
/// controls the mrproper warning and `source_tree_path` in
/// metadata.
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

    // Source-tree flock for local sources. Two parallel
    // `cargo ktstr test --kernel ./linux` runs would otherwise race
    // in `make` against the same source tree (e.g. one's
    // `make defconfig` racing with another's `make compile_commands.json`)
    // and produce inconsistent .config / build artifacts. The flock is
    // taken on the SOURCE TREE itself (per canonical path), distinct from
    // the cache-entry flock acquired inside `cache.store` (per cache key).
    // The two are complementary: the source-tree flock serializes the
    // build phase; the cache-entry flock serializes the atomic install.
    //
    // Held via `OwnedFd` for the lifetime of `_source_lock` — drops at
    // end of pipeline. Skipped under `KTSTR_BYPASS_LLC_LOCKS` to share
    // the operator's escape hatch with the LLC-flock bypass; that
    // env var already declares "I accept noise from concurrent runs."
    //
    // `try_flock` is non-blocking — if a concurrent peer holds the
    // lock, it returns `Ok(None)` and we bail with an actionable error
    // pointing at `cargo ktstr locks` for diagnosis. A blocking acquire
    // here would silently stall the operator's terminal with no
    // indication why; a fail-fast surfaces the contention immediately.
    let _source_lock = if is_local_source
        && std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_none_or(|v| v.is_empty())
    {
        Some(acquire_source_tree_lock(source_dir, cli_label)?)
    } else {
        None
    };

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
            post_build_is_dirty: true,
        });
    }

    // Post-build dirty re-check. `local_source` captures
    // `is_dirty` ONCE at acquire time. The operator may then edit a
    // tracked file (`.config` mutation, source patch) DURING the
    // build window. The acquire-time `is_dirty=false` would say
    // "safe to cache" but the on-disk content actually built
    // differs from the HEAD commit recorded in the cache key —
    // a future cache hit on that key would serve a build that no
    // longer matches its identity. Re-running the same gix probes
    // catches the race. On any change (dirty flip OR HEAD-hash
    // shift from a concurrent commit), skip the cache store and
    // emit a one-liner explaining why the cache slot was passed
    // over.
    //
    // Errors from the re-check are surfaced as a warning rather
    // than a hard fail — the build itself succeeded; refusing to
    // store on a re-check probe failure would penalize an
    // otherwise-clean run for a transient gix glitch. The cache
    // store proceeds with the original key, on the same
    // pessimistic basis as a tree the re-check could not classify.
    if is_local_source {
        match crate::fetch::inspect_local_source_state(source_dir) {
            Ok(post) => {
                let hash_changed = post.short_hash
                    != acquired
                        .kernel_source
                        .as_local_git_hash()
                        .map(str::to_string);
                if post.is_dirty || hash_changed {
                    eprintln!(
                        "{cli_label}: source tree changed during build \
                         (acquire-time dirty={}, post-build dirty={}; \
                         hash_changed={hash_changed}); skipping cache store \
                         to avoid recording a stale identity. Re-run after \
                         the working tree settles to populate the cache.",
                        acquired.is_dirty, post.is_dirty,
                    );
                    return Ok(KernelBuildResult {
                        entry: None,
                        image_path,
                        // Mid-build mutation flips the run's
                        // reproducibility — the cache key recorded at
                        // acquire time no longer identifies the actual
                        // build input. Mirror that into the outcome so
                        // the kernel-label downstream gets the
                        // `_dirty` suffix.
                        post_build_is_dirty: true,
                    });
                }
            }
            Err(e) => {
                tracing::warn!(
                    cli_label = cli_label,
                    err = %format!("{e:#}"),
                    "post-build dirty re-check failed; proceeding to cache store",
                );
            }
        }
    }

    let config_path = source_dir.join(".config");
    let config_hash = if config_path.exists() {
        let data = std::fs::read(&config_path)?;
        Some(format!("{:08x}", crc32fast::hash(&data)))
    } else {
        None
    };

    let kconfig_hash = embedded_kconfig_hash();

    // Source-tree vmlinux stat (size + mtime seconds) so a later
    // `prefer_source_tree_for_dwarf` lookup can detect a user
    // rebuild between cache store and DWARF read. Only meaningful
    // for local sources whose vmlinux survived the build —
    // `vmlinux_ref` is `None` if vmlinux wasn't found, in which
    // case there's nothing to stat. mtime read is best-effort:
    // failure leaves the validation pair `None` and prefers the
    // pre-validation behavior for this entry.
    let source_vmlinux_stat = vmlinux_ref.and_then(|v| {
        let stat = std::fs::metadata(v).ok()?;
        let mtime_secs = stat.modified().ok().and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .ok()
                .or_else(|| {
                    std::time::UNIX_EPOCH
                        .duration_since(t)
                        .ok()
                        .map(|d| -(d.as_secs() as i64))
                })
        })?;
        Some((stat.len(), mtime_secs))
    });

    let mut metadata = crate::cache::KernelMetadata::new(
        acquired.kernel_source.clone(),
        arch.to_string(),
        image_name.to_string(),
        crate::test_support::now_iso8601(),
    )
    .with_version(acquired.version.clone())
    .with_config_hash(config_hash)
    .with_ktstr_kconfig_hash(Some(kconfig_hash));
    if let Some((size, mtime_secs)) = source_vmlinux_stat {
        metadata = metadata.with_source_vmlinux_stat(size, mtime_secs);
    }

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

    Ok(KernelBuildResult {
        entry,
        image_path,
        post_build_is_dirty: false,
    })
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
/// reconstruct the `{kernel}-{project_commit}` key the test process
/// used; the mtime fallback mirrors "show me the report from my
/// last test run."
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
/// `cli::` to match the `list_runs` / `compare_partitions` / `show_host`
/// convention where every stats-subcommand dispatch arm lands on a
/// `cli::*` helper before reaching the private `stats` module. The
/// returned `String` is printed verbatim by the dispatch site.
pub fn list_metrics(json: bool) -> Result<String> {
    crate::stats::list_metrics(json)
}

/// Render the distinct-value catalogue for the sidecar pool, for
/// `cargo ktstr stats list-values`.
///
/// Thin wrapper over [`crate::stats::list_values`] — exposed
/// through `cli::` for the same surface-stability reason as
/// [`list_metrics`]. The returned `String` is printed verbatim by
/// the dispatch site.
pub fn list_values(json: bool, dir: Option<&Path>) -> Result<String> {
    crate::stats::list_values(json, dir)
}

/// Compare two filter-defined partitions of the sidecar pool and
/// report regressions across slicing dimensions. See
/// [`crate::stats::compare_partitions`] for the full contract.
pub fn compare_partitions(
    filter_a: &RowFilter,
    filter_b: &RowFilter,
    filter: Option<&str>,
    policy: &ComparisonPolicy,
    dir: Option<&Path>,
    no_average: bool,
) -> Result<i32> {
    crate::stats::compare_partitions(filter_a, filter_b, filter, policy, dir, no_average)
}

/// Re-export the comparison-policy type so downstream crates using
/// `ktstr::cli` as their public surface don't need to reach into
/// the internal `ktstr::stats` module (which is `pub(crate)` —
/// see `lib.rs` — and therefore not a stable public path). The
/// policy is the only item in `stats` that a CLI or external
/// consumer constructs directly; every other item is internal
/// plumbing reached via `cli::compare_partitions`.
pub use crate::stats::{AveragedGroup, ComparisonPolicy, RowFilter, group_and_average};

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

/// Return the run-directory leaf name under `root` whose Levenshtein
/// edit distance from `query` is smallest AND within the closeness
/// threshold, or `None` if no candidate is close enough (or if
/// `root` cannot be enumerated).
///
/// Threshold is `max(3, query.len() / 3)` — same shape as
/// [`suggest_closest_test_name`] / [`suggest_closest_scenario_name`]
/// so the "did you mean?" UX stays uniform across the test-name,
/// scenario-name, and run-key surfaces. The absolute-3 floor lets
/// short keys (e.g. `6.14`) tolerate small typos while the
/// proportional `len/3` lets longer keys (e.g.
/// `6.14-abcdef1-dirty`) tolerate roughly one bit-flip per 3
/// chars.
///
/// Ties resolve to the FIRST name encountered in `read_dir`
/// iteration order — non-deterministic across filesystems but
/// consistent within a single invocation. The returned `String`
/// owns the leaf name (heap allocation per match) because
/// `read_dir` yields `OsString` filenames that the suggestion
/// outlives.
///
/// `read_dir` failure (root doesn't exist, permission denied)
/// silently degrades to `None` — the caller's primary diagnostic
/// is "run not found"; the "did you mean?" hint is best-effort
/// gravy and must not gate the bail path.
///
/// Filters via [`crate::test_support::is_run_directory`] so the
/// flock sentinel subdirectory (`.locks/`) and any other
/// dotfile-prefixed entry under [`runs_root`] cannot surface as
/// a "did you mean?" suggestion — the same predicate that
/// [`newest_run_dir`] and `sorted_run_entries` use, so all three
/// run-listing surfaces agree on what counts as a run dir.
fn suggest_closest_run_key(query: &str, root: &Path) -> Option<String> {
    let threshold = std::cmp::max(3, query.len() / 3);
    let entries = std::fs::read_dir(root).ok()?;
    let mut best: Option<(usize, String)> = None;
    for entry in entries.flatten() {
        if !crate::test_support::is_run_directory(&entry) {
            continue;
        }
        let name = match entry.file_name().to_str() {
            Some(s) => s.to_string(),
            None => continue,
        };
        let d = strsim::levenshtein(query, &name);
        if d > threshold {
            continue;
        }
        match best {
            Some((best_d, _)) if best_d <= d => continue,
            _ => best = Some((d, name)),
        }
    }
    best.map(|(_, name)| name)
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
///   the `compare_partitions` error shape),
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
        // Pair the discovery hint with a "did you mean?"
        // suggestion (when one is close enough) so the operator
        // doesn't have to retype the whole key — a Levenshtein
        // probe over the run-dir leaves catches one-character
        // typos directly in the error.
        let suggestion = suggest_closest_run_key(run, &root)
            .map(|name| format!(" Did you mean `{name}`?"))
            .unwrap_or_default();
        bail!(
            "run '{run}' not found under {}.{suggestion} \
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

/// Whether a `None` value on a [`crate::test_support::SidecarResult`]
/// `Option` field is the expected steady-state shape (e.g. `payload`
/// for a scheduler-only test) or signals a recoverable gap an
/// operator can remediate (e.g. `kernel_commit` from a tarball-cache
/// kernel that has no on-disk source tree to probe).
///
/// Used by [`explain_sidecar`] to label every diagnostic block; the
/// `JSON` shape exposes this as a `"classification"` string per
/// field so dashboards can color-code "expected" vs "actionable"
/// blocks without re-deriving the rule from causes prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoneClassification {
    /// `None` is the expected steady state for this field — no
    /// operator action recovers it (the source data does not
    /// exist or has not been wired yet).
    Expected,
    /// `None` indicates a recoverable gap — re-running the test in
    /// a different environment (in-repo cwd, non-tarball kernel,
    /// non-host-only test) would populate the field.
    Actionable,
}

impl NoneClassification {
    /// Stable string token for the JSON `classification` key on
    /// [`explain_sidecar`]'s machine-readable output. The text
    /// renderer uses the same token so a human reading the
    /// terminal output sees the same label they would scrape from
    /// JSON.
    fn as_str(self) -> &'static str {
        match self {
            Self::Expected => "expected",
            Self::Actionable => "actionable",
        }
    }
}

/// Catalog entry for one [`crate::test_support::SidecarResult`]
/// `Option` field — wired into [`SIDECAR_NONE_CATALOG`] so the
/// diagnostic surface stays in lockstep with the on-disk schema.
///
/// Each entry was derived from the rustdoc on the corresponding
/// field in `src/test_support/sidecar.rs` — see the per-field
/// references at the catalog site for the source of truth. This
/// catalog is static (no live probing): explain-sidecar's purpose
/// is post-hoc archive diagnosis, not live debugging, and dynamic
/// probes against an absent host (e.g. checking `KTSTR_KERNEL` on
/// a CI runner that produced an archived sidecar) would return
/// nonsense.
struct NoneCatalogEntry {
    /// Field name as serialized on disk (matches the struct field
    /// identifier verbatim — serde uses the field name without
    /// rename attributes for [`crate::test_support::SidecarResult`]).
    field: &'static str,
    /// Classification when this field is `None`. See
    /// [`NoneClassification`].
    classification: NoneClassification,
    /// Human-readable cause prose, one entry per documented cause
    /// from the field's rustdoc. The text renderer prints each
    /// cause on its own bulleted line; the JSON shape emits them
    /// as a JSON string array verbatim.
    causes: &'static [&'static str],
    /// Operator-actionable remediation, when one applies.
    /// `Some(...)` for fields where re-running in a different
    /// configuration would populate the value (e.g.
    /// `kernel_commit` recovers when `KTSTR_KERNEL` points at a
    /// local source tree); `None` for fields whose `None` is the
    /// steady-state shape with no recourse (e.g.
    /// `scheduler_commit` is reserved on the schema for future
    /// enrichment).
    ///
    /// One fix per entry: this is the most-common-case
    /// remediation, picked when a field has multiple causes that
    /// all converge on the same operator action. The current
    /// shape covers the typical case without forcing a deeper
    /// data-model change; a per-cause split is possible if the
    /// catalog grows fields whose causes legitimately diverge
    /// in their remediation.
    fix: Option<&'static str>,
}

/// Static catalog covering every `Option<T>` field on
/// [`crate::test_support::SidecarResult`]. Order matches the on-
/// disk schema declaration order so a human diff against
/// `SidecarResult` reads top-to-bottom.
///
/// Causes prose is sourced FROM the per-field rustdoc on
/// `SidecarResult` — see `src/test_support/sidecar.rs` for the
/// single source of truth on what each `None` means. A future
/// schema change that adds, removes, or renames an `Option`
/// field MUST update this catalog; the
/// `none_catalog_covers_every_option_field` test enforces
/// `SIDECAR_NONE_CATALOG.len() == EXPECTED_OPTION_FIELD_COUNT`
/// (a hand-coded `10`) and asserts the projection helper
/// enumerates the same field names in the same order. A new
/// `Option` field on `SidecarResult` requires bumping the
/// constant, extending [`project_optional_fields`]'s array
/// (which has compile-checked length `10` and will fail to
/// compile on a missing entry), and adding a catalog row.
const SIDECAR_NONE_CATALOG: &[NoneCatalogEntry] = &[
    // See SidecarResult::scheduler_commit — currently always
    // None for every SchedulerSpec variant.
    NoneCatalogEntry {
        field: "scheduler_commit",
        classification: NoneClassification::Expected,
        causes: &["no SchedulerSpec variant currently exposes a reliable \
             commit source — reserved on the schema for future \
             enrichment (e.g. --version probe or ELF-note read on \
             the resolved scheduler binary)"],
        // Steady-state None — no operator action recovers this
        // until a future SchedulerSpec wires up a commit source.
        fix: None,
    },
    // See SidecarResult::project_commit (cause split mirrors
    // detect_project_commit's documented None cases).
    NoneCatalogEntry {
        field: "project_commit",
        classification: NoneClassification::Actionable,
        causes: &[
            "current_dir() could not be resolved at sidecar-write \
             time (process cwd was rmdir'd while alive)",
            "test process cwd was not inside any git repository",
            "HEAD could not be read (unborn HEAD on a fresh \
             `git init` with zero commits, or a corrupt repository)",
        ],
        // Most-common cause is "cwd not inside a git repo";
        // running from inside any git-tracked source tree with
        // at least one commit reaches the `gix::discover`
        // walk-up that `detect_project_commit` performs. Stays
        // project-agnostic so external scheduler crates
        // depending on ktstr see prose that applies to their
        // own clones; also covers the unborn-HEAD cause by
        // requiring "at least one commit".
        fix: Some(
            "run from inside a git-tracked source tree with at \
             least one commit",
        ),
    },
    // See SidecarResult::payload — None when no binary payload
    // declared.
    NoneCatalogEntry {
        field: "payload",
        classification: NoneClassification::Expected,
        causes: &["test declared no binary payload (scheduler-only test \
             or pure-scenario test that never invokes \
             ctx.payload(...))"],
        // Steady-state None for scheduler-only tests — declaring
        // a payload would be a test-design change, not an
        // operator-side remediation.
        fix: None,
    },
    // See SidecarResult::monitor — None for host-only / early VM
    // failure / no valid samples.
    NoneCatalogEntry {
        field: "monitor",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only test path: monitor loop never started",
            "early VM failure: monitor loop terminated before \
             producing samples",
            "sample collection produced no valid data",
        ],
        // No single operator-actionable fix: causes span
        // host-only test choice (test-design), VM failure
        // (debug the failure), and sample-collection issues
        // (likely a kernel/sched_ext bug). Per-cause
        // remediations are tracked separately for a future
        // refactor.
        fix: None,
    },
    // See SidecarResult::kvm_stats — None when VM did not run
    // or KVM stats were unavailable.
    NoneCatalogEntry {
        field: "kvm_stats",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only test path: VM did not run",
            "KVM stats were unavailable on this host (e.g. KVM \
             module not loaded, /dev/kvm permissions, or kernel \
             missing the stats interface)",
        ],
        // No single operator-actionable fix: causes span
        // host-only test choice (test-design) and host KVM
        // setup (load module / fix permissions). Both are
        // distinct remediations — left as None to avoid
        // suggesting a fix that doesn't apply.
        fix: None,
    },
    // See SidecarResult::kernel_version — None for host-only or
    // missing metadata.
    NoneCatalogEntry {
        field: "kernel_version",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only test path: no kernel under test",
            "neither cache metadata nor `include/config/kernel.release` \
             yielded a version string",
        ],
        // No single operator-actionable fix: causes span
        // host-only test choice (test-design) and missing
        // metadata (cache regeneration). Per-cause remediations
        // would require splitting the entry; left as None.
        fix: None,
    },
    // See SidecarResult::kernel_commit — five enumerated None
    // causes per the field's rustdoc.
    NoneCatalogEntry {
        field: "kernel_commit",
        classification: NoneClassification::Actionable,
        causes: &[
            "KTSTR_KERNEL is unset or empty",
            "kernel source is a Tarball or Git transient cache \
             entry (no on-disk source tree to probe)",
            "resolved kernel directory is not a git repository \
             (gix::open failed)",
            "HEAD cannot be read (unborn HEAD on a fresh `git init` \
             with zero commits)",
            "gix probe failed for another reason — metadata, not \
             a gate",
        ],
        // Most-common causes (env-unset and tarball/git
        // transient cache) both recover by pointing
        // `KTSTR_KERNEL` at a real on-disk kernel source
        // tree. The "git repository" qualifier is load-bearing:
        // tarball extractions have no `.git`, so a path at a
        // bare-tree extraction will still produce
        // `kernel_commit = None` even with `KTSTR_KERNEL` set.
        fix: Some(
            "set KTSTR_KERNEL to a local kernel source tree that \
             is a git repository (e.g. a git clone of the kernel)",
        ),
    },
    // See SidecarResult::host — production writers always
    // populate this field; None on a non-fixture sidecar
    // signals a pre-enrichment archive predating the
    // host-context landing.
    NoneCatalogEntry {
        field: "host",
        classification: NoneClassification::Actionable,
        causes: &[
            "test-fixture path: not the production sidecar \
             writer (production writers always populate `host`)",
            "pre-enrichment archive: sidecar predates the \
             host-context landing — re-run the test to \
             regenerate under the current schema",
        ],
        // The pre-enrichment archive case is the only one an
        // operator can recover; the test-fixture path is by
        // construction not a production sidecar — calling that
        // out in the prose so an operator inspecting fixture
        // output doesn't try to re-run a non-production
        // sidecar.
        fix: Some(
            "for pre-enrichment archives, re-run the test to \
             regenerate under the current schema; test-fixture \
             sidecars are not production runs and cannot be \
             recovered by re-running",
        ),
    },
    // See SidecarResult::cleanup_duration_ms — None for
    // watchdog-kill or host-only.
    NoneCatalogEntry {
        field: "cleanup_duration_ms",
        classification: NoneClassification::Actionable,
        causes: &[
            "host-only / host-only-stub test path: no VM teardown \
             window to time",
            "run was killed by the watchdog before \
             `KtstrVm::collect_results` returned",
        ],
        // No single operator-actionable fix: causes span
        // host-only test choice (test-design) and watchdog
        // kill (debug the underlying timeout). Per-cause
        // remediations would require splitting the entry;
        // left as None.
        fix: None,
    },
    // See SidecarResult::run_source — only None case in the
    // current writer is a pre-rename archive whose `source`
    // key was dropped as unknown by the renamed schema.
    NoneCatalogEntry {
        field: "run_source",
        classification: NoneClassification::Actionable,
        causes: &["pre-rename archive: sidecar carries the old `source` \
             key which the current schema drops as an unknown \
             field, leaving `run_source` to fall back to None via \
             serde's tolerate-absence rule. Re-run the test to \
             regenerate under the new schema, or rename the key \
             in-place before deserialize"],
        // The only None case has two distinct recoveries:
        // re-run (regenerates the sidecar under the new schema)
        // or rename the on-disk JSON key in place before load.
        fix: Some(
            "re-run the test to regenerate, or rename the on-disk \
             `source` key to `run_source`",
        ),
    },
];

/// Project one [`crate::test_support::SidecarResult`] onto its
/// `Option` fields, returning `(field_name, is_some)` pairs in the
/// same order as [`SIDECAR_NONE_CATALOG`].
///
/// Hand-written rather than derived because:
/// - Only the 10 `Option<T>` fields are diagnostic surface; the
///   non-`Option` fields (`test_name`, `passed`, `stats`, etc.)
///   are always populated by deserialize and would clutter the
///   output without adding signal.
/// - The order MUST match the catalog so the field-by-field
///   lookup in [`render_explain_sidecar_text`] resolves
///   correctly.
///
/// A future schema addition that introduces a new `Option<T>`
/// field on `SidecarResult` MUST update this projection; the
/// `[(_, _); 10]` array literal makes the length compile-checked
/// — adding an entry without updating the length is a compile
/// error — and the `none_catalog_covers_every_option_field` test
/// asserts the catalog and projection enumerate the same names
/// in the same order.
fn project_optional_fields(sc: &crate::test_support::SidecarResult) -> [(&'static str, bool); 10] {
    [
        ("scheduler_commit", sc.scheduler_commit.is_some()),
        ("project_commit", sc.project_commit.is_some()),
        ("payload", sc.payload.is_some()),
        ("monitor", sc.monitor.is_some()),
        ("kvm_stats", sc.kvm_stats.is_some()),
        ("kernel_version", sc.kernel_version.is_some()),
        ("kernel_commit", sc.kernel_commit.is_some()),
        ("host", sc.host.is_some()),
        ("cleanup_duration_ms", sc.cleanup_duration_ms.is_some()),
        ("run_source", sc.run_source.is_some()),
    ]
}

/// File-walk statistics for the run directory under
/// [`explain_sidecar`]. `walked` counts every `.ktstr.json`
/// file the walker visited; `valid` counts how many parsed
/// into a [`crate::test_support::SidecarResult`]; `errors`
/// carries the per-file parse failures as
/// [`crate::test_support::SidecarParseError`] records (named
/// fields: `path`, `raw_error`, `enriched_message`); `io_errors`
/// carries the per-file IO failures as
/// [`crate::test_support::SidecarIoError`] records (named fields:
/// `path`, `raw_error`) — files whose `read_to_string` failed
/// before parsing could begin (permission denied, mid-rotate
/// truncation, broken symlink). All three vecs are sourced from
/// [`crate::test_support::collect_sidecars_with_errors`].
///
/// The `enriched_message` field on parse errors is
/// `Some(prose)` for failures where a known schema-drift
/// remediation applies (currently the `host` missing-field
/// case) and `None` otherwise. Both raw and enriched are
/// exposed so JSON consumers can render the raw serde message
/// for grep-friendly parse-error tracking AND the enriched
/// prose for human-facing remediation.
///
/// In the steady state, every predicate-matching file lands
/// in exactly one of `valid` (parsed OK), `errors` (read OK,
/// parse failed), or `io_errors` (read failed) — so
/// `walked == valid + errors.len() + io_errors.len()`. The
/// previous "implicit silent-drop count" (where IO failures
/// vanished from every channel) is gone.
///
/// This invariant holds when the run directory is stable for
/// the duration of [`explain_sidecar`]. It is NOT enforced by
/// a single atomic walk: `walked` is computed by
/// [`count_sidecar_files`] in a separate `read_dir` pass from
/// the parse-and-load pass in
/// [`crate::test_support::collect_sidecars_with_errors`].
/// Filesystem mutations between the two passes can perturb
/// the equality:
/// - A file appearing between passes lands in `valid` /
///   `errors` / `io_errors` but was not in `walked` —
///   `walked < valid + errors + io_errors`.
/// - A file disappearing between passes was in `walked` but
///   never reaches a parse outcome — `walked > valid + errors
///   + io_errors`.
/// - A path's type changing between passes (e.g.
///   delete+recreate as a different kind — file→dir or
///   dir→file; POSIX does not support in-place type flips,
///   so the mechanism is always unlink-then-create) can
///   shift it across categories.
///
/// Operators driving `explain-sidecar` against a quiescent
/// archive directory will not observe these effects; the
/// invariant is documented as steady-state because that's
/// the supported use case. CI consumers that need a hard
/// invariant should drain in-flight writes (or copy to a
/// snapshot) before invoking explain-sidecar.
///
/// `errors` and `io_errors` surface in both render paths: text
/// appends a `corrupt sidecars` block (parse failures, optional
/// enriched prose) and an `io errors` block (IO failures, raw
/// `std::io::Error` Display); JSON exposes them under
/// `_walk.errors` as `{path, error, enriched_message}` entries
/// and `_walk.io_errors` as `{path, error}` entries. Dashboard
/// consumers can surface them without parsing prose.
struct WalkStats {
    walked: usize,
    valid: usize,
    errors: Vec<crate::test_support::SidecarParseError>,
    io_errors: Vec<crate::test_support::SidecarIoError>,
}

/// Count `.ktstr.json` files under `run_dir` using
/// [`crate::test_support::collect_sidecars`]'s walk shape (flat
/// files plus one level of subdirectories). Pure file-existence
/// pass — no parsing, no `serde_json` work — used purely to
/// derive the `walked` count for [`WalkStats`].
///
/// Filename predicate is shared with the parsing walker via
/// [`crate::test_support::is_sidecar_filename`] so the count and
/// parse-outcome walkers cannot disagree on what qualifies as a
/// sidecar.
fn count_sidecar_files(run_dir: &Path) -> usize {
    let mut count = 0usize;
    let entries = match std::fs::read_dir(run_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut subdirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        if crate::test_support::is_sidecar_filename(&path) {
            count += 1;
        }
    }
    for sub in subdirs {
        if let Ok(entries) = std::fs::read_dir(&sub) {
            for entry in entries.flatten() {
                if crate::test_support::is_sidecar_filename(&entry.path()) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Load sidecars under `run_dir` and report file-walk statistics.
///
/// Delegates parsing to
/// [`crate::test_support::collect_sidecars_with_errors`] — that
/// helper emits per-file `eprintln!` hints on every failure
/// (parse: actionable schema-drift message; IO: raw error
/// string) AND returns two structured vecs:
/// `Vec<`[`crate::test_support::SidecarParseError`]`>` (each
/// record carries `path`, `raw_error`, and `enriched_message`)
/// that flows into [`WalkStats::errors`], and
/// `Vec<`[`crate::test_support::SidecarIoError`]`>` (each record
/// carries `path` and `raw_error`) that flows into
/// [`WalkStats::io_errors`]. Both feed the JSON / text
/// renderers alongside the per-field breakdown. The `walked`
/// count is derived independently via [`count_sidecar_files`]
/// so the diagnostic header reports total `.ktstr.json` files
/// visited (every predicate-matching file), independent of any
/// parse-failure or IO-failure short-circuit.
fn walk_run_with_stats(run_dir: &Path) -> (Vec<crate::test_support::SidecarResult>, WalkStats) {
    let walked = count_sidecar_files(run_dir);
    let (sidecars, errors, io_errors) = crate::test_support::collect_sidecars_with_errors(run_dir);
    let valid = sidecars.len();
    (
        sidecars,
        WalkStats {
            walked,
            valid,
            errors,
            io_errors,
        },
    )
}

/// Diagnose `Option`-field absences for a run's sidecars. Mirrors
/// [`show_run_host`]'s shape (`--run` + optional `--dir`,
/// printable string return) but renders one block per sidecar
/// listing populated fields plus per-`None` cause-and-classification
/// from [`SIDECAR_NONE_CATALOG`]. Different gauntlet variants on
/// the same run legitimately differ on which fields are populated
/// (host-only vs VM-backed, scheduler-only vs payload-bearing),
/// so the report is per-sidecar rather than aggregate.
///
/// Loads sidecars verbatim — this command does NOT call
/// [`crate::test_support::apply_archive_source_override`] even
/// when `dir` is set, because rewriting `run_source` to
/// `"archive"` at load time would destroy the only signal that
/// surfaces the pre-rename `source`-key drop case (see the
/// `run_source` entry in [`SIDECAR_NONE_CATALOG`]). The
/// diagnostic value of explain-sidecar depends on observing the
/// on-disk shape unmodified; deviation from the
/// `stats compare` / `stats list-values` archive-override
/// pattern is intentional. This DOES match
/// [`show_run_host`]'s pattern.
///
/// `json: true` emits a JSON object with three top-level keys:
/// `"_schema_version"` (a string version stamp — currently
/// `"1"` — that consumers can gate on for incompatible shape
/// changes), `"_walk"` (an envelope carrying `walked` / `valid`
/// counts — the same numbers the text header reports — plus an
/// `errors` array of `{path, error, enriched_message}` entries
/// covering every parse failure (`enriched_message` is a
/// human-facing remediation string when one applies, JSON null
/// otherwise) AND an `io_errors` array of `{path, error}`
/// entries covering every IO failure (file matched the
/// predicate but `read_to_string` failed before parsing — e.g.
/// permission denied, mid-rotate truncation). With both arrays,
/// `walked == valid + errors.len() + io_errors.len()` in the
/// steady state — every predicate-matching file lands in
/// exactly one bucket when the run directory is stable across
/// the count and load passes (see [`WalkStats`] for the
/// filesystem-race caveat). `"fields"` is a map keyed by
/// [`SIDECAR_NONE_CATALOG`] field name where each value is
/// `{ "none_count": N, "some_count": M, "classification": "...",
/// "causes": [...], "fix": "..." }`). `none_count` and
/// `some_count` are the across-all-sidecars-in-this-run counts
/// of `None` and `Some(_)` for that field, summing to
/// `_walk.valid` — both are emitted so dashboard consumers do
/// not need to derive the second from the first. `fix` carries
/// an operator-actionable remediation string for fields where
/// one applies, or JSON null otherwise. This shape is
/// dashboard-friendly: a CI consumer can ingest the JSON
/// without parsing per-sidecar prose. The text form retains
/// per-sidecar detail for human triage and appends trailing
/// `corrupt sidecars` (parse failures) and `io errors` (IO
/// failures) blocks when either occurred.
///
/// All-corrupt and all-IO-failure runs (every predicate-
/// matching file failed to parse, or every one failed to
/// read) are NOT a hard error — both renderers fall through
/// with `valid = 0`, no per-sidecar blocks (none parsed),
/// the per-field `fields` entries reporting zero counts, and
/// the relevant trailing block(s): `_walk.errors` /
/// corrupt-sidecars block for parse failures,
/// `_walk.io_errors` / io-errors block for IO failures, both
/// when failures of both classes occurred. This gives
/// dashboard consumers structured per-file visibility into
/// total-failure runs of either class rather than a
/// single-line bail.
///
/// Exit-code contract: this command exits 0 even when every
/// sidecar in the run failed to parse OR failed to read —
/// the diagnostic surface is the structured `_walk.errors`
/// and `_walk.io_errors` arrays (or the trailing
/// `corrupt sidecars` / `io errors` text blocks), not the
/// process exit code. CI scripts must inspect the JSON
/// channel for failure detection rather than relying on exit
/// code; two distinct gating policies cover different
/// operational stances:
///
/// - **Lenient** (treat partial failures as warnings):
///   `_walk.valid > 0`. Accepts any run with at least one
///   parsed sidecar; per-file failures still surface in the
///   JSON arrays for triage but do not fail the gate.
/// - **Strict** (fail on any sidecar failure):
///   `_walk.errors.len() == 0 && _walk.io_errors.len() == 0`.
///   Requires every predicate-matching file to parse cleanly.
///   Both checks are required because the two arrays cover
///   disjoint failure classes (parse vs read).
///
/// The two policies are NOT equivalent: a run with one valid
/// and one corrupt sidecar passes lenient (`valid == 1 > 0`)
/// but fails strict (`errors.len() == 1 > 0`). Consumers
/// pick the policy that matches their operational tolerance
/// for partial data.
///
/// The only exit-code failures are the two errors documented
/// below: missing run directory and zero predicate-matching
/// files (an empty run, not a corrupt or unreadable one).
///
/// # Errors
///
/// - The `run` argument is empty, equals `.`, escapes the
///   run-root via `..` segments, or is absolute — rejected
///   before path resolution to keep `--dir` (which an operator
///   may point at a shared archive pool) from being used to
///   read arbitrary filesystem locations under
///   attacker-controlled input. Empty and `.` both resolve via
///   `Path::join` to the unmodified pool root, which would
///   walk every archived run; explicitly rejecting these forces
///   operators to name the run they want.
/// - The run directory does not exist.
/// - The run directory exists but the walker found zero
///   `.ktstr.json` files at all. This case covers an empty
///   run directory AND a directory whose top-level
///   `read_dir` itself failed (e.g. read-permission denied
///   on the run directory). Per-file read-permission
///   failures DO NOT bail here — they surface through
///   `WalkStats::io_errors` and the renderers' `io errors`
///   block, returning Ok(...) with `valid = 0` and the IO
///   failures structured. Distinguishing the two: a
///   directory-level failure produces `walked = 0` and
///   bails; a per-file failure produces `walked > 0` and
///   renders.
pub fn explain_sidecar(run: &str, dir: Option<&Path>, json: bool) -> Result<String> {
    // Reject pool-root-aliasing and path-traversal inputs BEFORE
    // joining onto `root`. The `Path::join` rules that matter:
    // joining an empty string is a no-op; joining `.` is a no-op;
    // joining an absolute path REPLACES `root` with the absolute
    // argument. Each shape would let `--run` resolve outside the
    // intended single-run scope:
    //
    // - `""` and `.` both alias the pool root, walking every
    //   archived run instead of the requested one.
    // - `..` escapes upward toward arbitrary filesystem locations.
    // - `/` (RootDir) and Windows `Prefix` produce absolute paths
    //   that bypass `root` entirely.
    //
    // The empty-string case is checked before the component loop
    // because `Path::new("").components()` yields ZERO components
    // and would silently pass an iterate-and-reject validator.
    if run.is_empty() {
        bail!(
            "run argument must not be empty. The run argument is \
             joined onto the run-root via `Path::join` and must \
             contain at least one `Normal` path component — i.e. \
             must not be empty, `.`, `..`, or absolute (e.g. a \
             typical run key shape: `6.14-abc1234` or \
             `6.14-abc1234-dirty`). To point at a different pool \
             root, use `--dir`. Run `cargo ktstr stats list` to \
             enumerate available run keys.",
        );
    }
    // `Path::new(run)`'s `Component` iterator gives a structural
    // view of the input. A bare run-key like
    // `6.14-abc1234` only emits `Normal` components and passes;
    // every other component variant is a pool-root alias or
    // traversal vector and bails with an actionable message.
    for component in std::path::Path::new(run).components() {
        match component {
            std::path::Component::CurDir
            | std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                bail!(
                    "run '{run}' contains pool-root-aliasing or \
                     path-traversal components (`.`, `..`, or absolute \
                     path). The run argument is joined onto the \
                     run-root via `Path::join` and must contain only \
                     `Normal` path components — no `.`, `..`, or \
                     absolute prefix (e.g. a typical run key shape: \
                     `6.14-abc1234` or `6.14-abc1234-dirty`; \
                     multi-component paths like `gauntlet/job-1` are \
                     also accepted). To point at a different pool \
                     root, use `--dir`. Run `cargo ktstr stats list` \
                     to enumerate available run keys.",
                );
            }
            std::path::Component::Normal(_) => {}
        }
    }
    let root: std::path::PathBuf = match dir {
        Some(d) => d.to_path_buf(),
        None => crate::test_support::runs_root(),
    };
    let run_dir = root.join(run);
    if !run_dir.exists() {
        let suggestion = suggest_closest_run_key(run, &root)
            .map(|name| format!(" Did you mean `{name}`?"))
            .unwrap_or_default();
        bail!(
            "run '{run}' not found under {}.{suggestion} \
             Run `cargo ktstr stats list` to enumerate available run keys.",
            root.display(),
        );
    }
    let (sidecars, walk_stats) = walk_run_with_stats(&run_dir);
    if walk_stats.walked == 0 {
        bail!(
            "run '{run}' has no sidecar data (searched {})",
            run_dir.display(),
        );
    }
    // All-corrupt runs (sidecars.is_empty() && walk_stats.walked > 0)
    // fall through to the renderers — _walk.errors / the trailing
    // corrupt-sidecars block carry the structured per-file
    // visibility. Bailing here would lose the same diagnostic data
    // the JSON channel was designed to surface.
    if json {
        Ok(render_explain_sidecar_json(&sidecars, &walk_stats))
    } else {
        Ok(render_explain_sidecar_text(&sidecars, &walk_stats))
    }
}

/// Render the per-sidecar text block for [`explain_sidecar`].
/// Header line names the run-wide walked / valid counts so a
/// corrupt-skip surfaces in human output too. Each block lists
/// populated `Option` fields, then `None` fields with their
/// classification and the causes catalog entry.
///
/// Sidecars are sorted by `test_name` (with `run_id` as a tie
/// breaker for deterministic order across same-test variants)
/// before rendering so output is stable across filesystem
/// `read_dir` orderings — operators diffing two `explain-sidecar`
/// runs see content changes, not iteration-order noise.
///
/// When `walk_stats.errors` is non-empty, a trailing block
/// "corrupt sidecars (N):" lists each
/// [`crate::test_support::SidecarParseError`] entry — `path`
/// followed by `raw_error` and (when present) `enriched_message`
/// — so operators see parse failures alongside the per-sidecar
/// breakdown rather than relying on the eprintln-only stderr
/// path. When `walk_stats.io_errors` is non-empty, a parallel
/// "io errors (N):" block lists each
/// [`crate::test_support::SidecarIoError`] entry — `path` and
/// `raw_error` (Display of `std::io::Error`). Each block emits
/// independently and only when its source vec is non-empty —
/// the common all-valid case stays unchanged.
fn render_explain_sidecar_text(
    sidecars: &[crate::test_support::SidecarResult],
    walk_stats: &WalkStats,
) -> String {
    use std::fmt::Write as _;
    let mut sorted: Vec<&crate::test_support::SidecarResult> = sidecars.iter().collect();
    sorted.sort_by(|a, b| {
        a.test_name
            .cmp(&b.test_name)
            .then_with(|| a.run_id.cmp(&b.run_id))
    });
    let mut out = String::new();
    let _ = writeln!(
        out,
        "walked {} sidecar file(s), parsed {} valid\n",
        walk_stats.walked, walk_stats.valid,
    );
    for sc in &sorted {
        let _ = writeln!(out, "test: {}", sc.test_name);
        let _ = writeln!(out, "  topology: {}", sc.topology);
        let _ = writeln!(out, "  scheduler: {}", sc.scheduler);
        let _ = writeln!(out, "  run_id: {}", sc.run_id);
        // Arch is sourced from `host.arch`; renders as `-` when
        // either `host` is `None` (pre-host-context-landing archive
        // or host-only-stub) or `arch` is `None` (arch probe
        // failed) so the line reads consistently regardless of
        // which leg of the option chain dropped.
        let arch = sc
            .host
            .as_ref()
            .and_then(|h| h.arch.as_deref())
            .unwrap_or("-");
        let _ = writeln!(out, "  arch: {arch}");
        let projected = project_optional_fields(sc);
        let populated: Vec<&'static str> = projected
            .iter()
            .filter(|(_, b)| *b)
            .map(|(n, _)| *n)
            .collect();
        let none_fields: Vec<&'static str> = projected
            .iter()
            .filter(|(_, b)| !*b)
            .map(|(n, _)| *n)
            .collect();
        let populated_text = if populated.is_empty() {
            "<none>".to_string()
        } else {
            populated.join(", ")
        };
        let _ = writeln!(
            out,
            "  populated optional fields ({}): {populated_text}",
            populated.len(),
        );
        if none_fields.is_empty() {
            let _ = writeln!(out, "  none fields: <all populated>\n");
            continue;
        }
        let _ = writeln!(out, "  none fields ({}):", none_fields.len());
        for field in none_fields {
            // Catalog lookup is O(catalog) per lookup, but
            // catalog is fixed at 10 entries and a sidecar has
            // at most 10 None fields — O(100) per sidecar in
            // the worst case. A HashMap would add hashing
            // overhead without measurable benefit.
            let entry = SIDECAR_NONE_CATALOG
                .iter()
                .find(|e| e.field == field)
                .expect(
                    "catalog must cover every projected field — \
                     guarded by none_catalog_covers_every_option_field",
                );
            let _ = writeln!(
                out,
                "    {} [{}]",
                entry.field,
                entry.classification.as_str(),
            );
            for cause in entry.causes {
                let _ = writeln!(out, "      - {cause}");
            }
            if let Some(fix) = entry.fix {
                let _ = writeln!(out, "      fix: {fix}");
            }
        }
        out.push('\n');
    }
    if !walk_stats.errors.is_empty() {
        let _ = writeln!(out, "corrupt sidecars ({}):", walk_stats.errors.len());
        for err in &walk_stats.errors {
            let _ = writeln!(out, "  {}", err.path.display());
            let _ = writeln!(out, "    error: {}", err.raw_error);
            // Enriched prose lands on its own indented line below
            // the raw serde error so an operator sees both the
            // grep-friendly raw message and the human remediation
            // without losing either. Suppressed when no
            // enrichment applies (the common case — only the
            // host-missing schema-drift case enriches today).
            if let Some(prose) = &err.enriched_message {
                let _ = writeln!(out, "    enriched: {prose}");
            }
        }
        out.push('\n');
    }
    if !walk_stats.io_errors.is_empty() {
        let _ = writeln!(out, "io errors ({}):", walk_stats.io_errors.len());
        for err in &walk_stats.io_errors {
            let _ = writeln!(out, "  {}", err.path.display());
            let _ = writeln!(out, "    error: {}", err.raw_error);
        }
        out.push('\n');
    }
    out
}

/// JSON schema version stamp emitted on
/// [`ExplainOutput::_schema_version`]. Bumped on any incompatible
/// shape change (key rename, key removal, semantic shift) so
/// dashboard consumers can gate on a known shape rather than
/// guessing from the keys present. Additive shape changes (new
/// optional keys, new entries in `fields`) do NOT bump this.
///
/// Consumers should parse this string as an integer before
/// comparing — gate on `parsed >= 1` (integer comparison, not
/// lexicographic string comparison).
const EXPLAIN_SIDECAR_SCHEMA_VERSION: &str = "1";

/// Top-level JSON shape for [`explain_sidecar`] in `--json` mode.
/// Serialized via `serde_json::to_string_pretty`. Field order
/// matches the on-disk JSON output order — `_schema_version`
/// first so consumers gating on it see it before scanning the
/// rest of the document, then `_walk`, then `fields`.
///
/// **Per-field ordering across the two channels diverges**:
/// `fields` uses [`std::collections::BTreeMap`] so JSON output
/// orders entries alphabetically (deterministic across
/// invocations); a [`std::collections::HashMap`] would surface
/// hash-randomized ordering. The text renderer iterates the
/// projection helper's array directly, so text output orders
/// `None` fields by their declaration order in
/// [`SIDECAR_NONE_CATALOG`] (which matches struct declaration
/// order on [`crate::test_support::SidecarResult`]). Consumers
/// diffing JSON vs text outputs should expect `kernel_commit`
/// before `kernel_version` in JSON (alphabetical) but
/// `kernel_version` before `kernel_commit` in text (catalog
/// order).
///
/// **`_` prefix convention**: keys whose names begin with `_`
/// (`_schema_version`, `_walk`) are envelope keys carrying
/// metadata about the response itself rather than per-field
/// diagnostic data. The struct declares them before `fields`,
/// so they appear first in the serialized output (serde
/// preserves struct declaration order, not lexicographic
/// order). The `_` prefix is a convention signaling
/// envelope/metadata keys to consumers.
#[derive(serde::Serialize)]
struct ExplainOutput<'a> {
    _schema_version: &'a str,
    _walk: WalkStatsJson<'a>,
    fields: std::collections::BTreeMap<&'a str, FieldDiagnostic<'a>>,
}

/// Walk-statistics envelope under
/// [`ExplainOutput::_walk`]. Mirrors [`WalkStats`] in shape and
/// holds two freshly built error vecs: `errors`
/// (`Vec<WalkError>`) sourced from [`WalkStats::errors`] for
/// parse failures, and `io_errors` (`Vec<WalkIoError>`) sourced
/// from [`WalkStats::io_errors`] for files that matched the
/// sidecar predicate but failed to read. Paths are rendered via
/// `Path::display()` (a `String` allocation per entry, required
/// because `PathBuf` has no stable JSON-string `Serialize`
/// surface across platforms), but error messages are borrowed
/// as `&'a str` to avoid cloning.
///
/// `walked == valid + errors.len() + io_errors.len()` by
/// construction — every predicate-matching file lands in
/// exactly one of the three buckets.
#[derive(serde::Serialize)]
struct WalkStatsJson<'a> {
    walked: usize,
    valid: usize,
    errors: Vec<WalkError<'a>>,
    io_errors: Vec<WalkIoError<'a>>,
}

/// Per-file parse-failure entry for
/// [`WalkStatsJson::errors`]. `path` renders as an owned
/// `String` via `Path::display().to_string()` — matches the text
/// output's path encoding and side-steps the lossy / platform-
/// specific `PathBuf` → JSON conversion. `error` borrows the
/// raw serde error message verbatim from [`WalkStats::errors`];
/// `enriched_message` borrows the optional human-facing
/// remediation prose, JSON null when no enrichment applies (the
/// common case — only the `host` missing-field schema-drift
/// case produces an enrichment today).
///
/// Both keys emit on every entry so dashboard consumers see a
/// uniform shape across enriched and non-enriched failures.
#[derive(serde::Serialize)]
struct WalkError<'a> {
    path: String,
    error: &'a str,
    enriched_message: Option<&'a str>,
}

/// Per-file IO-failure entry for
/// [`WalkStatsJson::io_errors`]. Mirrors [`WalkError`]'s `path`
/// encoding (owned `String` via `Path::display().to_string()`)
/// and `error` borrowing (`&'a str` from
/// [`crate::test_support::SidecarIoError::raw_error`]). No
/// `enriched_message` — IO failures have no schema-drift
/// remediation catalog (causes vary per host: fix permissions,
/// fix the filesystem, retry the test); the raw message is the
/// remediation surface.
#[derive(serde::Serialize)]
struct WalkIoError<'a> {
    path: String,
    error: &'a str,
}

/// Per-field diagnostic entry under [`ExplainOutput::fields`].
/// Mirrors the prior manual `Map::insert` shape exactly:
/// `none_count` + `some_count` (counts across all valid
/// sidecars, summing to `_walk.valid`), `classification` as a
/// short tag string, `causes` as a borrowed slice, and `fix` as
/// `Option<&str>` (JSON string when Some, JSON null when None).
#[derive(serde::Serialize)]
struct FieldDiagnostic<'a> {
    none_count: usize,
    some_count: usize,
    classification: &'a str,
    causes: &'a [&'a str],
    fix: Option<&'a str>,
}

/// Render the aggregate JSON shape for [`explain_sidecar`]. The
/// top-level object has three keys: `_schema_version` (stamps
/// the shape so dashboards can gate on a known version),
/// `_walk` (walked / valid counts plus a per-file `errors` list
/// covering every parse failure), and `fields` (a map keyed by
/// [`SIDECAR_NONE_CATALOG`] field name with per-field
/// none_count/some_count/classification/causes/fix). The `fix`
/// key is JSON null for fields whose `None` is the steady-state
/// shape.
///
/// Construction uses `#[derive(serde::Serialize)]` structs and
/// `serde_json::to_string_pretty` rather than manual
/// `serde_json::Map::insert` calls — the derive path keeps the
/// shape definition in one place (struct fields), so a future
/// shape change is a struct edit rather than coordinating
/// matching insert order across construction code and
/// documentation.
fn render_explain_sidecar_json(
    sidecars: &[crate::test_support::SidecarResult],
    walk_stats: &WalkStats,
) -> String {
    let fields: std::collections::BTreeMap<&str, FieldDiagnostic<'_>> = SIDECAR_NONE_CATALOG
        .iter()
        .map(|entry| {
            let none_count = sidecars
                .iter()
                .filter(|sc| {
                    project_optional_fields(sc)
                        .iter()
                        .any(|(n, b)| *n == entry.field && !*b)
                })
                .count();
            // `some_count = total - none_count`. Total is the
            // count of sidecars in this run (`sidecars.len()` ==
            // `walk_stats.valid`); subtract rather than count
            // separately so the two never disagree on rounding /
            // off-by-one. Saturating subtraction is defensive
            // against an underflow that the projection helper's
            // boolean partition makes impossible — every sidecar
            // contributes exactly 0 or 1 to `none_count` per field.
            let some_count = sidecars.len().saturating_sub(none_count);
            (
                entry.field,
                FieldDiagnostic {
                    none_count,
                    some_count,
                    classification: entry.classification.as_str(),
                    causes: entry.causes,
                    fix: entry.fix,
                },
            )
        })
        .collect();
    let errors: Vec<WalkError<'_>> = walk_stats
        .errors
        .iter()
        .map(|err| WalkError {
            path: err.path.display().to_string(),
            error: &err.raw_error,
            enriched_message: err.enriched_message.as_deref(),
        })
        .collect();
    let io_errors: Vec<WalkIoError<'_>> = walk_stats
        .io_errors
        .iter()
        .map(|err| WalkIoError {
            path: err.path.display().to_string(),
            error: &err.raw_error,
        })
        .collect();
    let output = ExplainOutput {
        _schema_version: EXPLAIN_SIDECAR_SCHEMA_VERSION,
        _walk: WalkStatsJson {
            walked: walk_stats.walked,
            valid: walk_stats.valid,
            errors,
            io_errors,
        },
        fields,
    };
    serde_json::to_string_pretty(&output).expect(
        "static-shape JSON serialization is infallible — every \
         field in ExplainOutput / WalkStatsJson / WalkError / WalkIoError / \
         FieldDiagnostic is a primitive, &str, or Vec/BTreeMap \
         of those — no NaN, no non-string keys, no unsupported \
         types",
    )
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
/// (`KTSTR_KERNEL_PARALLELISM=abc` or `=0`) silently degrades to
/// the host-CPU default rather than disabling parallelism — a
/// disabled-pool resolve would serialize multi-spec invocations
/// with no observable signal that the env var was the cause.
/// Leading/trailing whitespace is trimmed before parsing so a
/// shell-quoted `=" 8 "` behaves the same as the unquoted form.
///
/// Extracted from cargo-ktstr's `resolve_kernel_set` so the
/// parsing rules live in one place; the cargo-ktstr binary
/// invokes this and feeds the result into
/// [`rayon::ThreadPoolBuilder::num_threads`]. Lives here in
/// `cli.rs` rather than in the binary so it's reachable from
/// rustdoc and from the lib's unit-test harness.
pub fn resolve_kernel_parallelism() -> usize {
    if let Ok(raw) = std::env::var(crate::KTSTR_KERNEL_PARALLELISM_ENV) {
        let trimmed = raw.trim();
        if let Ok(n) = trimmed.parse::<usize>()
            && n > 0
        {
            return n;
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
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, false, cpu_cap)?;

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
    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, false, None)?;

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

    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true, cpu_cap)?;

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

    let result = kernel_build_pipeline(&acquired, &cache, cli_label, false, true, cpu_cap)?;

    // Prefer the cached image path (stable across rebuilds).
    match result.entry {
        Some(entry) => Ok(entry.image_path()),
        None => Ok(result.image_path),
    }
}

/// Validate `path` is a kernel source tree (Makefile + Kconfig at
/// the root) and return the [`AcquiredSource`] computed by
/// [`crate::fetch::local_source`].
///
/// Shared across [`resolve_kernel_dir`] and
/// [`resolve_kernel_dir_to_entry`] so the validation diagnostic
/// and `local_source` error stringification live in one place.
fn acquire_local_source_tree(path: &std::path::Path) -> Result<crate::fetch::AcquiredSource> {
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

/// One run-dir-lock row. Per-run-key sidecar-write locks live at
/// `{runs_root}/.locks/{run_key}.lock` where `{run_key}` is the
/// `{kernel}-{project_commit}` directory name; `run_key` is parsed
/// from the filename stem (same shape as
/// [`CacheLockRow::cache_key`] — the row exists as a distinct type
/// so the "this lock serializes sidecar writes" semantic is
/// visible at the schema level rather than buried under the cache
/// table's heading).
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct RunDirLockRow {
    pub(crate) run_key: String,
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
    pub(crate) run_dirs: Vec<RunDirLockRow>,
}

/// Enumerate every ktstr lockfile reachable on the host, attach the
/// holder list parsed from `/proc/locks`, and return a structured
/// snapshot suitable for either human or JSON rendering.
///
/// Missing paths (no `/tmp` glob matches, no `cache_root/.locks/`,
/// no `runs_root/.locks/`) produce empty row vectors — not an
/// error. The lockfile glob pattern uses the `glob` crate (already
/// a dep); failures to expand are treated as "no files matched"
/// and surfaced via `tracing::warn!` so the operator still sees a
/// populated snapshot for the paths that did work.
fn collect_locks_snapshot() -> Result<LocksSnapshot> {
    let cache_root = CacheDir::default_root().ok();
    let runs_root = crate::test_support::runs_root();
    collect_locks_snapshot_from(Path::new("/tmp"), cache_root.as_deref(), Some(&runs_root))
}

/// Seam behind [`collect_locks_snapshot`]: enumerate LLC, per-CPU,
/// cache-entry, and per-run-key lockfiles under the given roots.
/// Tests inject tempdirs for each of `tmp_root`, `cache_root`, and
/// `runs_root` so the `ktstr locks` snapshot shape can be pinned
/// without touching the real host `/tmp`, the operator's cache
/// directory, or the workspace's `target/ktstr/`.
///
/// `tmp_root` is the directory containing `ktstr-llc-*.lock` and
/// `ktstr-cpu-*.lock` (in production: `/tmp`). `cache_root` is the
/// cache-directory whose `.locks/` subdirectory holds per-entry
/// locks (in production: `CacheDir::default_root()`); `None`
/// suppresses the cache-lock enumeration entirely, matching the
/// "home unresolvable" production fallback. `runs_root` is the
/// directory whose `.locks/` subdirectory holds per-run-key
/// sidecar-write locks (in production:
/// [`crate::test_support::runs_root`]); `None` suppresses run-dir
/// lock enumeration.
pub(crate) fn collect_locks_snapshot_from(
    tmp_root: &Path,
    cache_root: Option<&Path>,
    runs_root: Option<&Path>,
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
    // Subdirectory name sourced from `crate::flock::LOCK_DIR_NAME`
    // so the cache scan and the run-dir scan below stay in sync
    // with the cache module and sidecar module's on-disk layout.
    let mut cache: Vec<CacheLockRow> = Vec::new();
    if let Some(cache_root) = cache_root {
        let locks_dir = cache_root.join(crate::flock::LOCK_DIR_NAME);
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

    // Per-run-key sidecar-write locks: {runs_root}/.locks/*.lock —
    // skipped when `runs_root` is None (test isolation). Mirrors
    // the cache-lock loop's shape (single-segment file_stem →
    // run_key), but the row carries a distinct `RunDirLockRow`
    // type so the JSON surface and human heading distinguish
    // "this lock serializes sidecar writes" from "this lock
    // serializes a kernel cache install".
    let mut run_dirs: Vec<RunDirLockRow> = Vec::new();
    if let Some(runs_root) = runs_root {
        let locks_dir = runs_root.join(crate::flock::LOCK_DIR_NAME);
        let pattern = format!("{}/*.lock", locks_dir.display());
        if let Ok(expanded) = glob::glob(&pattern) {
            for entry in expanded.flatten() {
                let Some(stem) = entry.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                let holders = crate::flock::read_holders(&entry).unwrap_or_default();
                run_dirs.push(RunDirLockRow {
                    run_key: stem.to_string(),
                    lockfile: entry.display().to_string(),
                    holders,
                });
            }
        }
    }
    run_dirs.sort_by(|a, b| a.run_key.cmp(&b.run_key));

    Ok(LocksSnapshot {
        llcs,
        cpus,
        cache,
        run_dirs,
    })
}

/// Render a [`LocksSnapshot`] as four stacked comfy-tables for
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

    writeln!(out, "\nRun-dir locks:").unwrap();
    if snap.run_dirs.is_empty() {
        writeln!(out, "  (none)").unwrap();
    } else {
        let mut t = new_table();
        t.set_header(["RUN KEY", "LOCKFILE", "HOLDERS"]);
        for r in &snap.run_dirs {
            t.add_row([
                r.run_key.clone(),
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

    // -- explain-sidecar test helpers (#140) --
    //
    // Twelve+ tests in this module follow the same on-disk layout
    // boilerplate: open a tempdir, create a `<tmp>/<run-name>/`
    // directory, write one or more `.ktstr.json` files into it,
    // call `explain_sidecar(<run-name>, Some(tmp.path()), ...)`.
    // The helpers below collapse that boilerplate so:
    //
    // - new tests do not have to re-derive the directory layout
    //   (parent test code is the only place gauntlet-job nesting
    //   has to land);
    // - a regression that drifts the `.ktstr.json` filename
    //   convention surfaces in one place rather than across
    //   every test that hand-rolls the path.
    //
    // Each helper holds the `tempfile::TempDir` alive via the
    // returned tuple — dropping the helper drops the directory.
    // Per CLAUDE.md "follow established patterns": these helpers
    // mirror the existing `let tmp = tempfile::tempdir().unwrap();
    // let run_dir = tmp.path().join(name); std::fs::create_dir
    // (&run_dir).unwrap();` pattern used at every prior test
    // site, just lifted into one place.

    /// Create a tempdir + a named run directory inside it. Returns
    /// `(TempDir, run_dir_path)`. The TempDir guard MUST be kept
    /// alive in the test scope so the directory survives until
    /// the test asserts; Rust drops it at the end of the function
    /// scope, which deletes the directory tree.
    fn make_test_run(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let run_dir = tmp.path().join(name);
        std::fs::create_dir(&run_dir).expect("create run dir");
        (tmp, run_dir)
    }

    /// Write a serialized [`crate::test_support::SidecarResult`] to
    /// `<dir>/<key>.ktstr.json`. `key` is the variant-hash-shaped
    /// prefix used by the production writer (see
    /// `sidecar_variant_hash`); tests typically use `"a-0000…0"`
    /// or a per-test `t-…` for filename-sort determinism.
    fn write_sidecar(
        dir: &std::path::Path,
        key: &str,
        sc: &crate::test_support::SidecarResult,
    ) -> std::path::PathBuf {
        let path = dir.join(format!("{key}.ktstr.json"));
        let json = serde_json::to_string(sc).expect("fixture must serialize");
        std::fs::write(&path, json).expect("write sidecar");
        path
    }

    /// Write raw bytes (intended to be unparseable JSON or an
    /// alternate serialization of `SidecarResult` with mutated
    /// keys) to `<dir>/<key>.ktstr.json`. Used by parse-failure
    /// and old-key-archive tests. Returns the resolved path so
    /// callers can assert against `path.display().to_string()`.
    fn write_corrupt_sidecar(dir: &std::path::Path, key: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(format!("{key}.ktstr.json"));
        std::fs::write(&path, body).expect("write corrupt sidecar");
        path
    }

    /// `Vec<T>` field names on [`crate::test_support::SidecarResult`].
    /// These fields are hard-required (serde fails deserialize on
    /// absence) and serialize as `[]` when empty — distinct from
    /// the 10 `Option<T>` fields the diagnostic surface enumerates.
    /// The catalog and projection helper MUST never surface these
    /// names, since "missing Option" and "empty Vec" are different
    /// invariants.
    ///
    /// Pinned as a constant so the
    /// [`explain_sidecar_does_not_flag_empty_vec_fields_as_none`]
    /// test and any future Vec-aware test source the same list.
    /// A schema change that adds, removes, or renames a Vec
    /// field MUST update this constant — the
    /// [`sidecar_vec_fields_drift_guard`] test fires when the
    /// runtime fixture's Vec field set diverges.
    pub(super) const SIDECAR_VEC_FIELDS: &[&str] = &[
        "metrics",
        "stimulus_events",
        "active_flags",
        "verifier_stats",
        "sysctls",
        "kargs",
    ];

    /// Drift guard: serialize `SidecarResult::test_fixture()` and
    /// confirm every name in [`SIDECAR_VEC_FIELDS`] appears as an
    /// array-typed key in the JSON, AND that the count of array
    /// keys equals the constant's length. A schema change that
    /// promotes a Vec to an Option (or vice versa), renames a
    /// Vec field, or adds a new one without updating
    /// [`SIDECAR_VEC_FIELDS`] surfaces here — preventing the
    /// constant from going stale relative to the live struct.
    ///
    /// Distinct from
    /// [`none_catalog_covers_every_option_field`]: that test
    /// guards Option-arity, this one guards Vec-arity. A schema
    /// change must satisfy both.
    #[test]
    fn sidecar_vec_fields_drift_guard() {
        let sc = crate::test_support::SidecarResult::test_fixture();
        let value = serde_json::to_value(&sc).expect("fixture must serialize");
        let obj = value.as_object().expect("fixture is an Object");
        // Every name in SIDECAR_VEC_FIELDS must serialize as a
        // JSON array.
        for name in SIDECAR_VEC_FIELDS {
            let v = obj.get(*name).unwrap_or_else(|| {
                panic!(
                    "SIDECAR_VEC_FIELDS lists `{name}` but \
                                            it is not on the serialized fixture — \
                                            schema rename or removal not propagated \
                                            to the constant"
                )
            });
            assert!(
                v.is_array(),
                "SIDECAR_VEC_FIELDS lists `{name}` but it is not a \
                 JSON array on the fixture — schema flipped Vec→Option \
                 or another shape; update the constant",
            );
        }
        // Count guard: every JSON array key on the fixture must
        // be in the constant. Catches Vec additions that didn't
        // update the constant.
        let array_keys: Vec<&str> = obj
            .iter()
            .filter(|(_, v)| v.is_array())
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(
            array_keys.len(),
            SIDECAR_VEC_FIELDS.len(),
            "SidecarResult has {} JSON-array fields, SIDECAR_VEC_FIELDS \
             lists {}. Drift detected — update the constant. \
             Live array keys: {array_keys:?}; constant: {SIDECAR_VEC_FIELDS:?}",
            array_keys.len(),
            SIDECAR_VEC_FIELDS.len(),
        );
    }

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
    /// match the `compare_partitions` error shape — consistency across
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

    // -- explain_sidecar --

    /// Drift guard: every `Option<T>` field on `SidecarResult` must
    /// have a matching catalog entry in `SIDECAR_NONE_CATALOG`, and
    /// the projected-fields helper must enumerate the same set. A
    /// schema change that adds, removes, or renames an `Option`
    /// field MUST update both — this test fires when they drift.
    ///
    /// Counts must match the actual `Option<T>` count on
    /// `SidecarResult`: the 10 documented at the top of
    /// `src/test_support/sidecar.rs`. Any future addition flips
    /// this count and forces a co-update.
    #[test]
    fn none_catalog_covers_every_option_field() {
        const EXPECTED_OPTION_FIELD_COUNT: usize = 10;
        assert_eq!(
            super::SIDECAR_NONE_CATALOG.len(),
            EXPECTED_OPTION_FIELD_COUNT,
            "SIDECAR_NONE_CATALOG must cover every Option<T> field on \
             SidecarResult; expected {EXPECTED_OPTION_FIELD_COUNT}, got \
             {}. A schema change must update the catalog in lockstep.",
            super::SIDECAR_NONE_CATALOG.len(),
        );
        let sc = crate::test_support::SidecarResult::test_fixture();
        let projected = super::project_optional_fields(&sc);
        assert_eq!(
            projected.len(),
            EXPECTED_OPTION_FIELD_COUNT,
            "project_optional_fields must enumerate every Option<T> \
             field; expected {EXPECTED_OPTION_FIELD_COUNT}, got {}. Co-update \
             with the catalog when adding a new Option field.",
            projected.len(),
        );
        // Cross-check: every projected field name appears in the
        // catalog and vice versa. Stable order between the two is
        // a separate invariant the renderer relies on.
        for (i, (name, _)) in projected.iter().enumerate() {
            let catalog = &super::SIDECAR_NONE_CATALOG[i];
            assert_eq!(
                *name, catalog.field,
                "projected field {i} ({name:?}) must match catalog \
                 entry at the same index ({:?}) — order drift breaks \
                 the renderer's catalog-lookup expectation",
                catalog.field,
            );
        }
    }

    /// Catalog `causes` arrays must be non-empty for every entry.
    /// An empty causes list would render as a classification with
    /// no rationale — defeats the diagnostic surface's purpose.
    #[test]
    fn none_catalog_every_entry_has_causes() {
        for entry in super::SIDECAR_NONE_CATALOG {
            assert!(
                !entry.causes.is_empty(),
                "catalog entry for {} has no causes — every field's \
                 None case must document at least one cause",
                entry.field,
            );
        }
    }

    /// Expected-classified entries (steady-state None) must NOT
    /// carry a `fix:` — there is no operator action that
    /// recovers an Expected None, so emitting one would mislead.
    /// Pin the invariant so a future entry that flips
    /// classification without removing its fix gets caught.
    #[test]
    fn none_catalog_expected_entries_have_no_fix() {
        for entry in super::SIDECAR_NONE_CATALOG {
            if matches!(entry.classification, super::NoneClassification::Expected) {
                assert!(
                    entry.fix.is_none(),
                    "Expected-classified field {} must not carry a `fix:` \
                     — there is no operator action that recovers a \
                     steady-state None",
                    entry.field,
                );
            }
        }
    }

    /// Per the design ruling: the most-common-case `fix:`
    /// assignments are Some for project_commit (run from a
    /// git-tracked source tree), kernel_commit (set
    /// KTSTR_KERNEL), host (re-run), and run_source (re-run or
    /// rename). Other Actionable fields (monitor / kvm_stats /
    /// kernel_version / cleanup_duration_ms) span causes that
    /// don't converge on a single operator action and
    /// intentionally carry None. Encoding the assignment matrix
    /// in a test makes a future reviewer's "why doesn't monitor
    /// have a fix?" question answerable from the test name
    /// alone.
    #[test]
    fn none_catalog_fix_assignments_match_design_ruling() {
        let by_field: std::collections::HashMap<&'static str, Option<&'static str>> =
            super::SIDECAR_NONE_CATALOG
                .iter()
                .map(|e| (e.field, e.fix))
                .collect();
        // Fields that MUST carry a fix.
        let must_fix = ["project_commit", "kernel_commit", "host", "run_source"];
        // Fields that intentionally carry no fix because no
        // single operator action covers their multi-cause set.
        let must_not_fix = [
            "scheduler_commit",
            "payload",
            "monitor",
            "kvm_stats",
            "kernel_version",
            "cleanup_duration_ms",
        ];
        // Total-count guard: every catalog entry must be placed
        // in exactly one of the two lists. Without this, a new
        // `Option` field added to `SidecarResult` could land in
        // the catalog with neither `must_fix` nor `must_not_fix`
        // covering it, and the test below would silently
        // ignore the placement decision.
        assert_eq!(
            must_fix.len() + must_not_fix.len(),
            super::SIDECAR_NONE_CATALOG.len(),
            "every catalog entry must be classified as either \
             must-fix or must-not-fix; expected sum = catalog len \
             ({}), got must_fix={} + must_not_fix={}",
            super::SIDECAR_NONE_CATALOG.len(),
            must_fix.len(),
            must_not_fix.len(),
        );
        for field in &must_fix {
            let fix = by_field.get(field).copied().flatten();
            assert!(
                fix.is_some(),
                "field {field} must carry a `fix:` per the design ruling",
            );
        }
        for field in &must_not_fix {
            let fix = by_field.get(field).copied().flatten();
            assert!(
                fix.is_none(),
                "field {field} must NOT carry a `fix:` (multi-cause or \
                 steady-state None) — got: {fix:?}",
            );
        }
    }

    /// Error path: the named run directory does not exist. Mirrors
    /// `show_run_host_missing_run_returns_error`'s error shape so
    /// operators see consistent diagnostics across the two
    /// run-named subcommands.
    #[test]
    fn explain_sidecar_missing_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = super::explain_sidecar("nonexistent-run", Some(tmp.path()), false).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("run 'nonexistent-run' not found"),
            "missing-run error must name the run: {msg}",
        );
        assert!(
            msg.contains("cargo ktstr stats list"),
            "missing-run error must name the discovery command: {msg}",
        );
    }

    /// Error path: run directory exists but is empty. Walked count
    /// is zero — distinct from "files present but parse-failed."
    /// Diagnostic must say "no sidecar data" to match
    /// `show_run_host`'s error shape AND name the resolved
    /// run-directory path so an operator can confirm which
    /// directory was searched (catches `--dir` typos and pool-
    /// root mismatches without a separate `find` invocation).
    #[test]
    fn explain_sidecar_empty_run_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-empty");
        std::fs::create_dir(&run_dir).unwrap();
        let err = super::explain_sidecar("run-empty", Some(tmp.path()), false).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no sidecar data"),
            "empty-run error must use the canonical message: {msg}",
        );
        assert!(
            msg.contains("searched"),
            "empty-run error must name the searched directory: {msg}",
        );
        assert!(
            msg.contains(&run_dir.display().to_string()),
            "empty-run error must include the resolved run_dir path \
             ({}): {msg}",
            run_dir.display(),
        );
    }

    /// All-corrupt run is NOT a hard error — the renderer falls
    /// through with valid=0, no per-sidecar blocks, and the
    /// trailing corrupt-sidecars block carries every parse
    /// failure. Operators get the same structured per-file
    /// diagnostic that the JSON channel exposes, rather than a
    /// single bail line that loses per-file detail.
    #[test]
    fn explain_sidecar_all_corrupt_renders_structured_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-corrupt");
        std::fs::create_dir(&run_dir).unwrap();
        // Two `.ktstr.json` files with garbage content. Walker
        // counts both; parser rejects both.
        std::fs::write(run_dir.join("a-0000000000000000.ktstr.json"), "not json {").unwrap();
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            "{\"missing\": \"required-fields\"}",
        )
        .unwrap();
        let out = super::explain_sidecar("run-corrupt", Some(tmp.path()), false)
            .expect("all-corrupt is no longer a hard error — must render");
        assert!(
            out.contains("walked 2"),
            "header must name the walked count: {out}",
        );
        assert!(
            out.contains("parsed 0 valid"),
            "header must distinguish walked-vs-parsed (zero valid): {out}",
        );
        assert!(
            out.contains("corrupt sidecars (2):"),
            "all-corrupt run must surface the corrupt-sidecars \
             block listing every parse failure: {out}",
        );
        // Per-sidecar `test:` blocks must NOT appear — there are
        // no parsed sidecars to render.
        assert!(
            !out.contains("test:"),
            "no sidecar parsed — must not emit any per-sidecar \
             block: {out}",
        );
    }

    /// Happy path: one sidecar from `test_fixture` (every
    /// `Option<T>` field starts as `None`). Text output must
    /// list ALL ten fields under "none fields" with their
    /// classification + at least one cause string.
    #[test]
    fn explain_sidecar_text_lists_all_none_fields_for_fixture() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-all-none");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-all-none", Some(tmp.path()), false).unwrap();
        assert!(out.contains("walked 1"), "header must report walked: {out}");
        assert!(out.contains("parsed 1"), "header must report parsed: {out}");
        assert!(
            out.contains("none fields (10)"),
            "fixture has every Option as None — count must be 10: {out}",
        );
        // Spot-check that each catalog field name appears.
        for entry in super::SIDECAR_NONE_CATALOG {
            assert!(
                out.contains(entry.field),
                "output must mention field {}: {out}",
                entry.field,
            );
        }
        // Classification labels must surface.
        assert!(
            out.contains("[expected]"),
            "expected-class fields must surface their tag: {out}",
        );
        assert!(
            out.contains("[actionable]"),
            "actionable-class fields must surface their tag: {out}",
        );
        // The fix: line must surface for entries that carry one.
        // Catalog-derived rather than hardcoded — sourcing the
        // expected string from `SIDECAR_NONE_CATALOG` makes the
        // assertion auto-update if the catalog's fix prose changes
        // (e.g. wording polish). A regression that drops the fix
        // line entirely still surfaces; only intentional prose
        // edits avoid the test churn that hardcoding would force.
        let project_commit_fix = super::SIDECAR_NONE_CATALOG
            .iter()
            .find(|e| e.field == "project_commit")
            .and_then(|e| e.fix)
            .expect("project_commit must carry a fix per the design ruling");
        assert!(
            out.contains(&format!("fix: {project_commit_fix}")),
            "project_commit's fix: line must render its catalog \
             prose verbatim ({project_commit_fix:?}): {out}",
        );
        // Entries without a fix must NOT emit a `fix:` line for
        // that field. The test fixture has every Option as None,
        // so the rendered output must contain exactly one
        // `fix:` line per catalog entry whose `fix` is `Some(_)`,
        // and zero for entries whose `fix` is `None`. A
        // regression that emitted an empty-string fix would
        // surface as a stray `fix: \n` somewhere and inflate the
        // count beyond the catalog's true fix-bearing population.
        //
        // Catalog-derived count: a future entry that adds or
        // drops a `fix:` updates `SIDECAR_NONE_CATALOG`, and
        // this assertion auto-tracks. Hardcoding `== 4` would
        // require a coordinated edit at every catalog change.
        let fix_line_count = out.matches("\n      fix:").count();
        let expected_fix_count = super::SIDECAR_NONE_CATALOG
            .iter()
            .filter(|e| e.fix.is_some())
            .count();
        assert_eq!(
            fix_line_count, expected_fix_count,
            "exactly {expected_fix_count} entries carry a fix: in \
             the catalog (count derived via \
             SIDECAR_NONE_CATALOG.iter().filter(|e| e.fix.is_some()).count()); \
             output emitted {fix_line_count}: {out}",
        );
    }

    /// JSON shape: aggregate per-field with `none_count`,
    /// `some_count`, `classification`, and `causes`. With one
    /// fixture sidecar (every Option None), every field must
    /// report `none_count: 1` and `some_count: 0`, and the two
    /// counts must sum to `_walk.valid`. The `_walk` envelope
    /// must carry walked / valid counts.
    #[test]
    fn explain_sidecar_json_shape_aggregates_none_counts() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-json");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-json", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk key");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(walk.get("valid").and_then(|v| v.as_u64()), Some(1));
        let fields = parsed.get("fields").expect("must have fields key");
        for entry in super::SIDECAR_NONE_CATALOG {
            let f = fields
                .get(entry.field)
                .unwrap_or_else(|| panic!("missing field {}", entry.field));
            let none_count = f
                .get("none_count")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("missing none_count for {}", entry.field));
            let some_count = f
                .get("some_count")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("missing some_count for {}", entry.field));
            assert_eq!(
                none_count, 1,
                "every field in fixture is None — none_count must be 1 for {}",
                entry.field,
            );
            assert_eq!(
                some_count, 0,
                "every field in fixture is None — some_count must be 0 for {}",
                entry.field,
            );
            assert_eq!(
                none_count + some_count,
                1,
                "none_count + some_count must sum to _walk.valid (1) for {}",
                entry.field,
            );
            assert_eq!(
                f.get("classification").and_then(|v| v.as_str()),
                Some(entry.classification.as_str()),
                "classification must round-trip for {}",
                entry.field,
            );
            let causes = f
                .get("causes")
                .and_then(|v| v.as_array())
                .unwrap_or_else(|| panic!("missing causes for {}", entry.field));
            assert_eq!(
                causes.len(),
                entry.causes.len(),
                "causes array length must match catalog for {}",
                entry.field,
            );
            // `fix` must round-trip as the catalog's value:
            // string when Some, JSON null when None. Emitting
            // the key uniformly across entries (even on null)
            // saves dashboard consumers a `contains_key`
            // branch.
            let fix_value = f
                .get("fix")
                .unwrap_or_else(|| panic!("missing fix for {}", entry.field));
            match entry.fix {
                Some(expected) => {
                    assert_eq!(
                        fix_value.as_str(),
                        Some(expected),
                        "fix string must round-trip for {}",
                        entry.field,
                    );
                }
                None => {
                    assert!(
                        fix_value.is_null(),
                        "fix must be JSON null for fix=None entry {}: \
                         got {fix_value:?}",
                        entry.field,
                    );
                }
            }
        }
    }

    /// Mixed populated/None: some `Option` fields populated,
    /// others None. The diagnostic block must list both
    /// "populated" and "none fields" sections with the right
    /// counts. Catches a regression that emitted populated
    /// fields under "none fields" or vice versa.
    #[test]
    fn explain_sidecar_text_distinguishes_populated_from_none() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.payload = Some("ipc_pingpong".to_string());
        sc.kernel_version = Some("6.14.2".to_string());
        sc.run_source = Some("local".to_string());
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-mixed", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("populated optional fields (3)"),
            "must report 3 populated: {out}",
        );
        assert!(
            out.contains("payload"),
            "populated `payload` must appear: {out}",
        );
        assert!(out.contains("none fields (7)"), "must report 7 None: {out}",);
    }

    /// Per-sidecar text output: the `arch:` line surfaces under
    /// each sidecar's block, sourced from `host.arch`. Pins
    /// GAP-B: a sidecar with `host: Some(arch=x86_64)` must
    /// surface `arch: x86_64`; a sidecar with `host: None`
    /// must surface `arch: -` so the line is present whether
    /// or not host context was captured (uniform shape across
    /// host-populated and host-absent sidecars makes the line
    /// scriptable without conditional grep).
    #[test]
    fn explain_sidecar_text_renders_arch_line() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-arch");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.host = Some(crate::host_context::HostContext::test_fixture());
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-arch", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("arch: x86_64"),
            "host-populated sidecar must surface `arch: x86_64` per the \
             test_fixture default: {out}",
        );
    }

    /// Per-sidecar text output: when `host` is `None`, the
    /// `arch:` line still emits with the `-` sentinel so the
    /// line is uniform across host-populated and host-absent
    /// sidecars. Pins the fallback rendering arm of GAP-B.
    #[test]
    fn explain_sidecar_text_arch_line_falls_back_to_dash_when_host_none() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-arch-none");
        std::fs::create_dir(&run_dir).unwrap();
        // SidecarResult::test_fixture defaults host: None.
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-arch-none", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("arch: -"),
            "host-None sidecar must surface `arch: -` (consistent \
             sentinel with `list_runs`'s arch column): {out}",
        );
    }

    /// Per-sidecar text output: two sidecars in the same run
    /// with different None patterns must each get their own
    /// block. Aggregate-only output would conflate them.
    #[test]
    fn explain_sidecar_text_emits_one_block_per_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-two");
        std::fs::create_dir(&run_dir).unwrap();
        let mut a = crate::test_support::SidecarResult::test_fixture();
        a.test_name = "test_a".to_string();
        let mut b = crate::test_support::SidecarResult::test_fixture();
        b.test_name = "test_b".to_string();
        b.payload = Some("ipc_pingpong".to_string());
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&a).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            serde_json::to_string(&b).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-two", Some(tmp.path()), false).unwrap();
        assert!(out.contains("test: test_a"), "test_a block missing: {out}");
        assert!(out.contains("test: test_b"), "test_b block missing: {out}");
        assert!(out.contains("walked 2"), "walked count must be 2: {out}");
        assert!(out.contains("parsed 2"), "parsed count must be 2: {out}");
    }

    /// JSON aggregation across multiple sidecars: one sidecar
    /// has `payload = Some(...)`, the other has `payload = None`.
    /// `none_count` for `payload` must be 1 (not 2, not 0), and
    /// `some_count` must be 1 — both surfaced so dashboards do
    /// not need to derive the second from `_walk.valid` minus the
    /// first.
    #[test]
    fn explain_sidecar_json_aggregates_partial_none_correctly() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-partial");
        std::fs::create_dir(&run_dir).unwrap();
        let a = crate::test_support::SidecarResult::test_fixture();
        let mut b = crate::test_support::SidecarResult::test_fixture();
        b.payload = Some("ipc_pingpong".to_string());
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&a).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run_dir.join("b-0000000000000000.ktstr.json"),
            serde_json::to_string(&b).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-partial", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let payload = parsed
            .get("fields")
            .and_then(|f| f.get("payload"))
            .expect("payload field must be present");
        assert_eq!(
            payload.get("none_count").and_then(|v| v.as_u64()),
            Some(1),
            "payload None in 1 of 2 sidecars — none_count must be 1",
        );
        assert_eq!(
            payload.get("some_count").and_then(|v| v.as_u64()),
            Some(1),
            "payload Some in 1 of 2 sidecars — some_count must be 1",
        );
        // Sanity: `host` is None in both sidecars.
        let host = parsed
            .get("fields")
            .and_then(|f| f.get("host"))
            .expect("host field must be present");
        assert_eq!(
            host.get("none_count").and_then(|v| v.as_u64()),
            Some(2),
            "host None in 2 of 2 sidecars — none_count must be 2",
        );
        assert_eq!(
            host.get("some_count").and_then(|v| v.as_u64()),
            Some(0),
            "host Some in 0 of 2 sidecars — some_count must be 0",
        );
    }

    /// Walker counts both valid and corrupt `.ktstr.json` files.
    /// One valid sidecar + one corrupt produces walked=2,
    /// parsed=1; the diagnostic still emits the valid sidecar's
    /// block.
    #[test]
    fn explain_sidecar_walks_corrupt_files_into_count() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed-parse");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "garbage{").unwrap();
        let out = super::explain_sidecar("run-mixed-parse", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("walked 2"),
            "walker must visit both files: {out}",
        );
        assert!(
            out.contains("parsed 1"),
            "only the valid file parses: {out}",
        );
    }

    /// Walker recurses one level into subdirectories — matches
    /// `collect_sidecars`'s gauntlet-job layout. A sidecar in
    /// `run/job-1/foo.ktstr.json` must be loaded.
    #[test]
    fn explain_sidecar_walks_one_level_subdirectory() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-sub");
        let sub = run_dir.join("job-x");
        std::fs::create_dir_all(&sub).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            sub.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-sub", Some(tmp.path()), false).unwrap();
        assert!(out.contains("walked 1"), "must walk into job-x: {out}");
        assert!(
            out.contains("parsed 1"),
            "must parse the nested file: {out}"
        );
    }

    /// Walker MUST ignore non-`.ktstr.json` files even when they
    /// have a `.json` extension. A `metadata.json` or other
    /// adjacent JSON in the run directory must NOT inflate the
    /// walked count or trigger a parse.
    #[test]
    fn explain_sidecar_ignores_non_ktstr_json() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-with-other-json");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        // Adjacent non-ktstr JSON file — must be skipped.
        std::fs::write(run_dir.join("metadata.json"), "{}").unwrap();
        let out = super::explain_sidecar("run-with-other-json", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("walked 1"),
            "non-ktstr JSON must not inflate the walked count: {out}",
        );
    }

    /// JSON output must be a single valid JSON document — round-
    /// trips through `serde_json::from_str` cleanly. Catches
    /// regressions that trailing-comma or invalid escape would
    /// produce.
    #[test]
    fn explain_sidecar_json_is_valid_document() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-roundtrip");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-roundtrip", Some(tmp.path()), true).unwrap();
        // Strict round-trip — any malformed delimiter would error.
        let _: serde_json::Value = serde_json::from_str(&out).expect("output must be valid JSON");
    }

    /// Partial population: the 7 string/u64-shaped `Option`
    /// fields are populated while monitor / kvm_stats / host
    /// stay None (those carry struct shapes whose full fixtures
    /// are deliberately out of scope here). The diagnostic
    /// surface must split the report into "populated optional
    /// fields (7)" and "none fields (3)" — proving the
    /// projection helper distinguishes the two arms correctly
    /// at non-degenerate populated counts.
    #[test]
    fn explain_sidecar_text_handles_partial_population() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-partial-pop");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        sc.scheduler_commit = Some("aaaa111".to_string());
        sc.project_commit = Some("bbbb222".to_string());
        sc.payload = Some("payload".to_string());
        sc.kernel_version = Some("6.14.2".to_string());
        sc.kernel_commit = Some("cccc333".to_string());
        sc.cleanup_duration_ms = Some(123);
        sc.run_source = Some("local".to_string());
        // 7 Options populated; 3 still None (monitor, kvm_stats, host).
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-partial-pop", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("populated optional fields (7)"),
            "7 of 10 Options populated must be reflected in the count: {out}",
        );
        assert!(
            out.contains("none fields (3)"),
            "3 of 10 Options remain None — must report (3): {out}",
        );
    }

    /// Classification labels are stable strings ("expected" vs
    /// "actionable") usable as JSON enum tokens. A regression that
    /// renamed one would break dashboard consumers.
    #[test]
    fn none_classification_as_str_returns_stable_tokens() {
        assert_eq!(super::NoneClassification::Expected.as_str(), "expected");
        assert_eq!(super::NoneClassification::Actionable.as_str(), "actionable",);
    }

    /// `kernel_commit` is the most-multi-cause field (5 documented
    /// causes per its rustdoc). Catalog must enumerate all 5 so
    /// the diagnostic surface mirrors the schema documentation.
    #[test]
    fn kernel_commit_catalog_lists_five_causes() {
        let entry = super::SIDECAR_NONE_CATALOG
            .iter()
            .find(|e| e.field == "kernel_commit")
            .expect("kernel_commit must be in the catalog");
        assert_eq!(
            entry.causes.len(),
            5,
            "kernel_commit rustdoc enumerates 5 None causes; catalog \
             must mirror that",
        );
    }

    /// Schema version stamp is `"1"` — pin it as a constant test
    /// so a future shape change that bumps the version surfaces
    /// here. Dashboard consumers gate on this string; a silent
    /// bump would mask incompatibility.
    #[test]
    fn explain_sidecar_schema_version_constant_is_one() {
        assert_eq!(super::EXPLAIN_SIDECAR_SCHEMA_VERSION, "1");
    }

    /// JSON output stamps `_schema_version: "1"` so dashboard
    /// consumers can gate on a known shape. Surfaces at the
    /// top level alongside `_walk` and `fields`.
    #[test]
    fn explain_sidecar_json_includes_schema_version() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-schema");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-schema", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        assert_eq!(
            parsed.get("_schema_version").and_then(|v| v.as_str()),
            Some(super::EXPLAIN_SIDECAR_SCHEMA_VERSION),
            "JSON output must stamp _schema_version: {out}",
        );
    }

    /// JSON `_walk.errors` is an empty array when every walked
    /// file parses cleanly. Dashboard consumers expect the key to
    /// be present even when empty (uniform shape, no
    /// `contains_key` branching).
    #[test]
    fn explain_sidecar_json_walk_errors_empty_when_all_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-clean-walk");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-clean-walk", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let errors = parsed
            .get("_walk")
            .and_then(|w| w.get("errors"))
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be a JSON array");
        assert!(
            errors.is_empty(),
            "no parse failures — _walk.errors must be empty: {out}",
        );
    }

    /// JSON `_walk.errors` lists `{path, error, enriched_message}`
    /// triples for every file that failed to parse. The valid file
    /// stays counted in `_walk.valid`; the corrupt one surfaces
    /// under errors. `enriched_message` is JSON null for generic
    /// parse failures (no schema-drift remediation applies). Pins
    /// the structured-error channel — previously parse failures
    /// were eprintln-only and dashboard consumers had no
    /// programmatic way to surface them.
    #[test]
    fn explain_sidecar_json_walk_errors_lists_corrupt_files() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed-errs-json");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        let corrupt_path = run_dir.join("b-0000000000000000.ktstr.json");
        std::fs::write(&corrupt_path, "garbage{").unwrap();
        let out = super::explain_sidecar("run-mixed-errs-json", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk key");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(walk.get("valid").and_then(|v| v.as_u64()), Some(1));
        let errors = walk
            .get("errors")
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be a JSON array");
        assert_eq!(errors.len(), 1, "exactly one parse failure expected: {out}",);
        let entry = &errors[0];
        let path = entry
            .get("path")
            .and_then(|v| v.as_str())
            .expect("each error entry must carry a string `path`");
        assert_eq!(
            path,
            corrupt_path.display().to_string(),
            "error path must match the corrupt file's resolved path",
        );
        let error = entry
            .get("error")
            .and_then(|v| v.as_str())
            .expect("each error entry must carry a string `error`");
        assert!(
            !error.is_empty(),
            "error message must not be empty (serde_json should produce \
             a parse-error message for `garbage{{`): {out}",
        );
        // `enriched_message` MUST be present on every entry as a
        // uniform shape (no contains_key branching for dashboard
        // consumers). For generic parse failures like `garbage{`,
        // no schema-drift remediation applies — the value is JSON
        // null.
        let enriched = entry
            .get("enriched_message")
            .expect("each error entry must carry an enriched_message key");
        assert!(
            enriched.is_null(),
            "generic parse failure has no schema-drift remediation; \
             enriched_message must be JSON null: {enriched:?}",
        );
    }

    /// `enriched_parse_error_message` returns operator-facing
    /// remediation prose for the host-missing schema-drift
    /// pattern (a serde error mentioning both "missing field"
    /// and "`host`"). Verifies the helper directly against a
    /// synthetic error string — modern `SidecarResult` declares
    /// `host: Option<HostContext>`, so serde tolerates absence
    /// and never emits the missing-field error against the live
    /// schema. The enrichment exists as a forward-compat
    /// remediation surface for a future schema bump that
    /// promotes `host` to a non-Option field, or for sidecars
    /// produced under such a bump's transitional window.
    #[test]
    fn enriched_parse_error_message_returns_prose_for_host_missing_pattern() {
        // Construct a synthetic serde-style error string that
        // matches the enrichment trigger.
        let raw = "missing field `host` at line 1 column 100";
        let path = std::path::Path::new("/tmp/example-run/sidecar.ktstr.json");
        let enriched = crate::test_support::enriched_parse_error_message_for_test(path, raw)
            .expect("host-missing pattern must produce enrichment prose");
        assert!(
            enriched.contains("host"),
            "enrichment must mention the host field: {enriched}",
        );
        assert!(
            enriched.contains("re-run"),
            "enrichment must point at the re-run remediation: {enriched}",
        );
        assert!(
            enriched.contains("disposable-sidecar"),
            "enrichment must reference the pre-1.0 disposable-sidecar \
             policy: {enriched}",
        );
        // Generic parse errors return None — no enrichment prose.
        let raw_generic = "expected ident at line 1 column 2";
        let no_enrichment =
            crate::test_support::enriched_parse_error_message_for_test(path, raw_generic);
        assert!(
            no_enrichment.is_none(),
            "generic parse error must produce no enrichment: {no_enrichment:?}",
        );
    }

    /// All-corrupt run renders structured JSON with `valid: 0`,
    /// every parse failure populated under `_walk.errors`, and
    /// each field's counts at zero (no valid sidecars to count
    /// over). Pins the no-bail behavior — dashboard consumers
    /// that ingest the JSON channel see the same shape regardless
    /// of partial vs full corruption, just with `valid` collapsed
    /// to 0 and `none_count`/`some_count` mirroring that.
    #[test]
    fn explain_sidecar_all_corrupt_json_renders_structured_diagnostic() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-all-corrupt-json");
        std::fs::create_dir(&run_dir).unwrap();
        std::fs::write(run_dir.join("a-0000000000000000.ktstr.json"), "{").unwrap();
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "garbage{").unwrap();
        let out = super::explain_sidecar("run-all-corrupt-json", Some(tmp.path()), true)
            .expect("all-corrupt JSON must render, not bail");
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk key");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(
            walk.get("valid").and_then(|v| v.as_u64()),
            Some(0),
            "all-corrupt run must report valid=0: {out}",
        );
        let errors = walk
            .get("errors")
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be present");
        assert_eq!(
            errors.len(),
            2,
            "every parse failure must surface in _walk.errors: {out}",
        );
        // Every field entry exists with zero counts — no valid
        // sidecars to count over, but the catalog shape is
        // preserved so dashboard consumers see a uniform schema.
        let fields = parsed
            .get("fields")
            .and_then(|f| f.as_object())
            .expect("fields must be present");
        for entry in super::SIDECAR_NONE_CATALOG {
            let f = fields
                .get(entry.field)
                .unwrap_or_else(|| panic!("field {} must be present", entry.field));
            assert_eq!(
                f.get("none_count").and_then(|v| v.as_u64()),
                Some(0),
                "{}: zero valid sidecars — none_count must be 0",
                entry.field,
            );
            assert_eq!(
                f.get("some_count").and_then(|v| v.as_u64()),
                Some(0),
                "{}: zero valid sidecars — some_count must be 0",
                entry.field,
            );
        }
        // Schema-version stamp still emits — the all-corrupt
        // path is a regular render, not a degenerate one.
        assert_eq!(
            parsed.get("_schema_version").and_then(|v| v.as_str()),
            Some(super::EXPLAIN_SIDECAR_SCHEMA_VERSION),
            "schema_version must stamp on every render: {out}",
        );
    }

    /// Text output renders the corrupt-sidecars block with
    /// `enriched:` lines below `error:` lines for failures
    /// carrying enrichment prose. Generic failures emit only
    /// `error:`. The renderer is private; this test drives the
    /// public surface with a generic-failure fixture (the only
    /// reliably-triggerable parse-failure shape against the live
    /// schema, since `host` is `Option<HostContext>` and serde
    /// tolerates absence) and verifies the absence of an
    /// `enriched:` line. The host-missing enrichment payload
    /// itself is exercised by
    /// `enriched_parse_error_message_returns_prose_for_host_missing_pattern`.
    #[test]
    fn explain_sidecar_text_omits_enriched_line_for_generic_failure() {
        let (tmp, run_dir) = make_test_run("run-generic-fail-text");
        // `garbage{` triggers a generic serde parse error — no
        // enrichment applies, so the `enriched:` line must NOT
        // appear in the rendered output.
        write_corrupt_sidecar(&run_dir, "a-0000000000000000", "garbage{");
        let out = super::explain_sidecar("run-generic-fail-text", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("corrupt sidecars (1):"),
            "generic parse failure must surface in the corrupt \
             block: {out}",
        );
        assert!(
            out.contains("    error:"),
            "generic parse failure must emit raw `error:` line: {out}",
        );
        assert!(
            !out.contains("    enriched:"),
            "generic parse failure has no enrichment — `enriched:` \
             line must NOT appear: {out}",
        );
    }

    /// Text output appends a trailing "corrupt sidecars (N):"
    /// block when parse failures occurred. Each entry lists the
    /// path on its own line, then the error message indented as
    /// "    error: ...". Operators see parse failures inline
    /// rather than relying on stderr eprintln.
    ///
    /// Positional invariant: the corrupt-sidecars block is a
    /// TRAILING block — it must appear AFTER the
    /// `walked N sidecar file(s), parsed M valid` header AND
    /// after every per-sidecar `test:` block. A regression that
    /// reordered output (e.g. emitted parse failures before the
    /// header) would mislead operators reading top-down. Pin
    /// the relative position so that ordering is enforced.
    #[test]
    fn explain_sidecar_text_appends_corrupt_sidecars_block() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-text-corrupt");
        std::fs::create_dir(&run_dir).unwrap();
        let mut valid = crate::test_support::SidecarResult::test_fixture();
        valid.test_name = "valid_test".to_string();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        let corrupt_path = run_dir.join("b-0000000000000000.ktstr.json");
        std::fs::write(&corrupt_path, "garbage{").unwrap();
        let out = super::explain_sidecar("run-text-corrupt", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("corrupt sidecars (1):"),
            "text output must include trailing corrupt-sidecars block \
             when errors exist: {out}",
        );
        assert!(
            out.contains(&corrupt_path.display().to_string()),
            "corrupt-sidecars block must list the corrupt file's path: {out}",
        );
        assert!(
            out.contains("    error:"),
            "corrupt-sidecars block must indent each error under its path: {out}",
        );
        // Positional ordering: the header (`walked N sidecar file(s)`),
        // then the per-sidecar block (`test: valid_test`), then
        // the trailing `corrupt sidecars (N):` block. Failing
        // any of these orderings would produce a confusing
        // top-down read.
        let header_pos = out
            .find("walked 2 sidecar file(s)")
            .expect("walked-header must precede everything");
        let test_block_pos = out
            .find("test: valid_test")
            .expect("per-sidecar block must emit for the valid file");
        let corrupt_pos = out
            .find("corrupt sidecars (1):")
            .expect("corrupt-sidecars block must emit");
        assert!(
            header_pos < test_block_pos,
            "header must precede per-sidecar blocks: {out}",
        );
        assert!(
            test_block_pos < corrupt_pos,
            "per-sidecar blocks must precede the trailing corrupt \
             block — operators read top-down: {out}",
        );
    }

    /// The "corrupt sidecars" block is suppressed when the walk
    /// produced zero parse failures. Common case for clean runs;
    /// emitting an empty header would be visual noise.
    ///
    /// First test in the module migrated to the
    /// [`make_test_run`] / [`write_sidecar`] helpers (#140) — the
    /// boilerplate previously open-coded across 12+ tests now
    /// flows through the helpers, so a future regression in the
    /// shared on-disk layout surfaces in one place. Existing
    /// tests will migrate as they're touched for unrelated
    /// reasons; a wholesale sweep is out of scope for the
    /// helper-extraction task itself.
    #[test]
    fn explain_sidecar_text_omits_corrupt_block_when_no_errors() {
        let (tmp, run_dir) = make_test_run("run-text-clean");
        let sc = crate::test_support::SidecarResult::test_fixture();
        write_sidecar(&run_dir, "t-0000000000000000", &sc);
        let out = super::explain_sidecar("run-text-clean", Some(tmp.path()), false).unwrap();
        assert!(
            !out.contains("corrupt sidecars"),
            "no parse failures — corrupt-sidecars block must be \
             suppressed: {out}",
        );
    }

    /// `Vec` fields on `SidecarResult` (metrics, stimulus_events,
    /// active_flags, verifier_stats, sysctls, kargs) are
    /// hard-required and serialize as `[]` when empty; they are NOT
    /// `Option<T>`. Catalog only covers the 10 `Option` fields, so
    /// the diagnostic must NEVER name a Vec field — neither under
    /// "populated optional fields" nor under "none fields".
    /// Construct a fixture with all 10 Options Some + every Vec
    /// empty and assert the report says "none fields: <all
    /// populated>" with zero mentions of any Vec field name.
    /// Guards against a future refactor that confuses Option-None
    /// with Vec-empty.
    #[test]
    fn explain_sidecar_does_not_flag_empty_vec_fields_as_none() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-vecs");
        std::fs::create_dir(&run_dir).unwrap();
        let mut sc = crate::test_support::SidecarResult::test_fixture();
        // Populate every Option<_> so "none fields" should be empty.
        sc.scheduler_commit = Some("aaaa111".to_string());
        sc.project_commit = Some("bbbb222".to_string());
        sc.payload = Some("payload".to_string());
        sc.kernel_version = Some("6.14.2".to_string());
        sc.kernel_commit = Some("cccc333".to_string());
        sc.cleanup_duration_ms = Some(123);
        sc.run_source = Some("local".to_string());
        sc.monitor = Some(crate::monitor::MonitorSummary::default());
        sc.kvm_stats = Some(crate::vmm::KvmStatsTotals::default());
        sc.host = Some(crate::host_context::HostContext::test_fixture());
        // Vec fields stay empty (test_fixture defaults).
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-vecs", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("none fields: <all populated>"),
            "all Options populated — must report no None fields: {out}",
        );
        // Source the Vec field names from the SIDECAR_VEC_FIELDS
        // constant — single source of truth shared with
        // `sidecar_vec_fields_drift_guard`. A schema change that
        // renames or adds a Vec field updates the constant in
        // one place; this assertion picks up the new name
        // without an independent edit. Substring match is safe:
        // these are distinctive snake_case identifiers, no false
        // positives in the rendered output.
        for vec_field in SIDECAR_VEC_FIELDS {
            assert!(
                !out.contains(vec_field),
                "Vec field '{vec_field}' is hard-required (not Option) and \
                 must never appear in explain-sidecar output: {out}",
            );
        }
    }

    /// Pre-rename archives carry the on-disk key `"source"`; the
    /// current schema reads `"run_source"`. With the old key
    /// present, serde's tolerate-absence rule populates
    /// `run_source: None`, and the diagnostic must surface that
    /// None with the catalog's `run_source` cause prose mentioning
    /// the rename. Guards the diagnostic value of the
    /// pre-rename detection path.
    #[test]
    fn explain_sidecar_handles_old_source_key_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-old-source-key");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        // Serialize then mutate the JSON to drop `run_source` and
        // inject `source` — emulates a pre-rename archive on disk.
        let mut value = serde_json::to_value(&sc).expect("fixture must serialize");
        let obj = value.as_object_mut().expect("fixture is an Object");
        obj.remove("run_source");
        obj.insert(
            "source".to_string(),
            serde_json::Value::String("archive".to_string()),
        );
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&value).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-old-source-key", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("run_source"),
            "explain-sidecar must surface run_source as None for \
             pre-rename archive: {out}",
        );
        // Catalog's run_source cause prose says "tolerate-absence"
        // and "rename"; a regression that drops the rename
        // diagnostic prose would lose the actionable context.
        assert!(
            out.contains("rename"),
            "run_source None cause must mention the rename: {out}",
        );
    }

    /// `dir=None` defaults to [`crate::test_support::runs_root`],
    /// which derives `{CARGO_TARGET_DIR or "target"}/ktstr/`. Pin
    /// the env to a tempdir, write a sidecar under
    /// `<tmp>/ktstr/<run-key>/`, and call `explain_sidecar(run,
    /// None, false)`. Covers the implicit-root resolution path
    /// that all prior tests skip via explicit `Some(tmp.path())`.
    /// `lock_env()` serializes against any other env-touching
    /// test in this binary.
    #[test]
    fn explain_sidecar_resolves_dir_default_to_runs_root() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let tmp = tempfile::tempdir().unwrap();
        let _env_target = EnvVarGuard::set("CARGO_TARGET_DIR", tmp.path());
        let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
        // runs_root is `<CARGO_TARGET_DIR>/ktstr` — create the run
        // directory under that path and write a sidecar inside.
        let runs_root = tmp.path().join("ktstr");
        let run_dir = runs_root.join("run-default-root");
        std::fs::create_dir_all(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        // dir=None — explain_sidecar must resolve runs_root via
        // CARGO_TARGET_DIR and find the sidecar.
        let out = super::explain_sidecar("run-default-root", None, false)
            .expect("dir=None must resolve via runs_root() and succeed");
        assert!(
            out.contains("walked 1"),
            "default-dir resolution must walk into the run dir: {out}",
        );
        assert!(
            out.contains("parsed 1 valid"),
            "default-dir resolution must parse the sidecar: {out}",
        );
    }

    /// A 0-byte `.ktstr.json` file is a parse failure (serde_json
    /// rejects empty input), not a silent skip. The walker counts
    /// it in `walked` and emits a parse error; the valid sibling
    /// still parses and renders. Guards the empty-file edge that
    /// could regress to "treat as missing" if `read_to_string` ever
    /// short-circuits on length-zero input.
    #[test]
    fn explain_sidecar_handles_zero_byte_file() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-zero-byte");
        std::fs::create_dir(&run_dir).unwrap();
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        // Zero-byte file: empty string parses as a serde_json
        // error, not as a successful Object.
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "").unwrap();
        let out = super::explain_sidecar("run-zero-byte", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("walked 2"),
            "walker must visit both files (valid + zero-byte): {out}",
        );
        assert!(
            out.contains("parsed 1"),
            "only the valid file parses; zero-byte is a parse \
             failure: {out}",
        );
        assert!(
            out.contains("corrupt sidecars (1):"),
            "zero-byte file must surface in the corrupt-sidecars \
             block as a parse failure, not be silently dropped: {out}",
        );
    }

    /// `SidecarResult` does NOT set `#[serde(deny_unknown_fields)]`,
    /// so a future-schema sidecar that adds a key serde does not
    /// recognize must still deserialize cleanly. Forward-compat
    /// invariant: a CI consumer running an older ktstr binary
    /// against a sidecar from a newer one must NOT lose the run.
    #[test]
    fn explain_sidecar_tolerates_unknown_extra_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-extra-fields");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        // Serialize then inject an extra key serde_json doesn't know.
        let mut value = serde_json::to_value(&sc).expect("fixture must serialize");
        let obj = value.as_object_mut().expect("fixture is an Object");
        obj.insert(
            "future_field".to_string(),
            serde_json::Value::String("hypothetical".to_string()),
        );
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&value).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-extra-fields", Some(tmp.path()), false).unwrap();
        assert!(
            out.contains("walked 1"),
            "walker must visit the file: {out}",
        );
        assert!(
            out.contains("parsed 1 valid"),
            "extra `future_field` key must NOT block deserialize \
             (SidecarResult does not deny_unknown_fields): {out}",
        );
        // Also confirm the sidecar is named in the output — i.e.
        // the parsed value reaches the renderer, not just the
        // count.
        assert!(
            out.contains("test: t"),
            "parsed sidecar must render its test_name: {out}",
        );
    }

    /// Catalog classification per field is operator-visible as a
    /// stable tag (`expected` vs `actionable`). A typo regression
    /// flipping any field's classification would silently mislead
    /// dashboards and triage. Pin the exact mapping per the
    /// catalog's documented design ruling.
    ///
    /// HashMap dedup guard: collecting catalog entries into a
    /// `HashMap` keyed by field name silently overwrites
    /// duplicates — two entries with the same `field` would
    /// land as one, and the test would read whichever survived
    /// without ever signaling the duplication. Asserting that
    /// `by_field.len() == SIDECAR_NONE_CATALOG.len()` catches
    /// a regression where two catalog entries share a field
    /// name (rename collision, copy-paste).
    #[test]
    fn explain_sidecar_classification_accuracy_per_field() {
        let by_field: std::collections::HashMap<&'static str, super::NoneClassification> =
            super::SIDECAR_NONE_CATALOG
                .iter()
                .map(|e| (e.field, e.classification))
                .collect();
        // Dedup guard: HashMap collapses duplicate keys silently.
        // If two catalog entries shared a field name, this length
        // would diverge from the catalog length and the
        // per-field assertions below would silently miss the
        // overwritten entry.
        assert_eq!(
            by_field.len(),
            super::SIDECAR_NONE_CATALOG.len(),
            "SIDECAR_NONE_CATALOG must have unique `field` values \
             — HashMap collected {} entries, catalog has {}. Two \
             entries sharing a name would silently overwrite during \
             collect.",
            by_field.len(),
            super::SIDECAR_NONE_CATALOG.len(),
        );
        // Per-field expected classification — pinned from the
        // catalog's documented design ruling. Two Expected
        // (steady-state None with no operator action), eight
        // Actionable (operator can recover or environment is
        // wrong).
        let expected_pairs: &[(&str, super::NoneClassification)] = &[
            ("scheduler_commit", super::NoneClassification::Expected),
            ("payload", super::NoneClassification::Expected),
            ("project_commit", super::NoneClassification::Actionable),
            ("monitor", super::NoneClassification::Actionable),
            ("kvm_stats", super::NoneClassification::Actionable),
            ("kernel_version", super::NoneClassification::Actionable),
            ("kernel_commit", super::NoneClassification::Actionable),
            ("host", super::NoneClassification::Actionable),
            ("cleanup_duration_ms", super::NoneClassification::Actionable),
            ("run_source", super::NoneClassification::Actionable),
        ];
        // Total-count guard: every catalog entry must be pinned
        // here. A future Option field added to SidecarResult that
        // lands in the catalog without a pinned classification
        // would silently slip past this assertion otherwise.
        assert_eq!(
            expected_pairs.len(),
            super::SIDECAR_NONE_CATALOG.len(),
            "every catalog entry must have a pinned classification \
             in this test (catalog len {}, pinned len {})",
            super::SIDECAR_NONE_CATALOG.len(),
            expected_pairs.len(),
        );
        for (field, expected) in expected_pairs {
            let actual = by_field
                .get(field)
                .copied()
                .unwrap_or_else(|| panic!("catalog must contain field {field}"));
            assert_eq!(
                actual, *expected,
                "field {field}: classification mismatch — expected \
                 {expected:?}, got {actual:?}",
            );
        }
    }

    // -- explain_sidecar IO-error surface (#124) --

    /// IO failures (file matched the sidecar predicate but
    /// `read_to_string` failed before parsing) surface as a
    /// distinct `io errors` block in text output and as a distinct
    /// `_walk.io_errors` array in JSON output. Trigger: a SUBDIR
    /// child whose name matches the sidecar predicate but is
    /// itself a directory — `read_to_string` returns EISDIR.
    /// `collect_sidecars_with_errors`'s subdir-recursion loop
    /// passes every child to `try_load` without filtering on
    /// `is_dir()`, so a directory named `foo.ktstr.json` reaches
    /// the read step and fails. Walked counts the file via
    /// `count_sidecar_files`'s identical predicate; the IO error
    /// lands in `io_errors`; per-field counts at zero (no parsed
    /// sidecar exists).
    ///
    /// Exit-code contract: explain-sidecar still returns Ok(...)
    /// — IO failures are diagnostic surface, not bail conditions
    /// (matches the parse-failure all-corrupt test at
    /// `explain_sidecar_all_corrupt_renders_structured_diagnostic`).
    #[test]
    fn explain_sidecar_io_errors_surface_in_text_block_and_json() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-io-err");
        std::fs::create_dir(&run_dir).unwrap();
        let sub = run_dir.join("sub");
        std::fs::create_dir(&sub).unwrap();
        // Directory named like a sidecar: predicate matches,
        // read_to_string returns EISDIR, IO error captured.
        std::fs::create_dir(sub.join("eisdir.ktstr.json")).unwrap();

        // Text channel: trailing `io errors (N):` block lists
        // the path; per-sidecar blocks empty (zero parsed).
        let text_out = super::explain_sidecar("run-io-err", Some(tmp.path()), false).unwrap();
        assert!(
            text_out.contains("walked 1"),
            "predicate-matching dir must count as walked: {text_out}",
        );
        assert!(
            text_out.contains("parsed 0 valid"),
            "no parsed sidecar — header must report 0 valid: {text_out}",
        );
        assert!(
            text_out.contains("io errors (1):"),
            "IO failure must surface in the trailing io-errors block: {text_out}",
        );
        assert!(
            text_out.contains("eisdir.ktstr.json"),
            "io-errors block must name the failing path: {text_out}",
        );
        assert!(
            !text_out.contains("corrupt sidecars"),
            "no parse failures — corrupt-sidecars block must be \
             absent (IO and parse channels are distinct): {text_out}",
        );

        // JSON channel: `_walk.io_errors` populated, `_walk.errors`
        // empty, walked / valid agree with text.
        let json_out = super::explain_sidecar("run-io-err", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&json_out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk");
        assert_eq!(walk.get("walked").and_then(|v| v.as_u64()), Some(1));
        assert_eq!(walk.get("valid").and_then(|v| v.as_u64()), Some(0));
        let parse_errs = walk
            .get("errors")
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be present as array");
        assert!(
            parse_errs.is_empty(),
            "no parse failures — _walk.errors must be empty: {json_out}",
        );
        let io_errs = walk
            .get("io_errors")
            .and_then(|e| e.as_array())
            .expect("_walk.io_errors must be present as array (#124)");
        assert_eq!(
            io_errs.len(),
            1,
            "exactly one IO failure expected: {json_out}",
        );
        let entry = &io_errs[0];
        let path = entry
            .get("path")
            .and_then(|v| v.as_str())
            .expect("each io-error entry must carry a string `path`");
        assert!(
            path.ends_with("eisdir.ktstr.json"),
            "io-error path must name the failing file: got {path}",
        );
        let error = entry
            .get("error")
            .and_then(|v| v.as_str())
            .expect("each io-error entry must carry a string `error`");
        assert!(
            !error.is_empty(),
            "io-error message must not be empty: {json_out}",
        );
        // io-error entries do NOT carry `enriched_message` —
        // distinct from parse-error entries (no schema-drift
        // catalog applies to filesystem incidents). Pin the
        // shape difference so a future implementer doesn't
        // accidentally add the field expecting symmetry.
        assert!(
            entry.get("enriched_message").is_none(),
            "io-error entries must NOT have enriched_message: {json_out}",
        );
    }

    /// `walked == valid + errors.len() + io_errors.len()` by
    /// construction — every predicate-matching file lands in
    /// exactly one of the three buckets. Mixed-failure run
    /// (one valid + one parse-fail + one io-fail) pins the
    /// invariant against future regressions where a class of
    /// failure might be silently dropped.
    #[test]
    fn explain_sidecar_walk_counts_reconcile_across_outcomes() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-mixed-outcomes");
        std::fs::create_dir(&run_dir).unwrap();
        // Valid sidecar.
        let valid = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("a-0000000000000000.ktstr.json"),
            serde_json::to_string(&valid).unwrap(),
        )
        .unwrap();
        // Parse-failing sidecar.
        std::fs::write(run_dir.join("b-0000000000000000.ktstr.json"), "garbage{").unwrap();
        // IO-failing entry: directory named like a sidecar inside
        // a level-1 subdir.
        let sub = run_dir.join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::create_dir(sub.join("c-0000000000000000.ktstr.json")).unwrap();

        let json_out =
            super::explain_sidecar("run-mixed-outcomes", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&json_out).expect("json output must round-trip parse");
        let walk = parsed.get("_walk").expect("must have _walk");
        let walked = walk.get("walked").and_then(|v| v.as_u64()).unwrap();
        let valid_n = walk.get("valid").and_then(|v| v.as_u64()).unwrap();
        let parse_errs = walk.get("errors").and_then(|e| e.as_array()).unwrap().len() as u64;
        let io_errs = walk
            .get("io_errors")
            .and_then(|e| e.as_array())
            .unwrap()
            .len() as u64;
        assert_eq!(
            walked,
            valid_n + parse_errs + io_errs,
            "walked must equal valid + errors + io_errors — every \
             predicate-matching file lands in exactly one bucket. \
             walked={walked}, valid={valid_n}, errors={parse_errs}, \
             io_errors={io_errs}: {json_out}",
        );
        assert_eq!(walked, 3, "three predicate-matching entries");
        assert_eq!(valid_n, 1, "one valid sidecar");
        assert_eq!(parse_errs, 1, "one parse failure");
        assert_eq!(io_errs, 1, "one io failure");
    }

    /// `_walk.io_errors` is an empty array on the all-clean
    /// happy path. Pins the uniform-shape contract (key always
    /// emits) so dashboard consumers don't need `contains_key`
    /// branching.
    #[test]
    fn explain_sidecar_json_walk_io_errors_empty_when_no_io_failures() {
        let tmp = tempfile::tempdir().unwrap();
        let run_dir = tmp.path().join("run-clean-io");
        std::fs::create_dir(&run_dir).unwrap();
        let sc = crate::test_support::SidecarResult::test_fixture();
        std::fs::write(
            run_dir.join("t-0000000000000000.ktstr.json"),
            serde_json::to_string(&sc).unwrap(),
        )
        .unwrap();
        let out = super::explain_sidecar("run-clean-io", Some(tmp.path()), true).unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");
        let io_errs = parsed
            .get("_walk")
            .and_then(|w| w.get("io_errors"))
            .and_then(|e| e.as_array())
            .expect("_walk.io_errors must be present as array even when empty");
        assert!(
            io_errs.is_empty(),
            "no IO failures — _walk.io_errors must be empty: {out}",
        );
    }

    // -- explain_sidecar E2E enrichment rendering (#139) --

    /// Direct-renderer test: construct a synthetic [`WalkStats`]
    /// carrying a [`crate::test_support::SidecarParseError`] with a
    /// non-`None` `enriched_message`, call
    /// [`super::render_explain_sidecar_text`] in isolation, and
    /// assert the trailing `corrupt sidecars` block emits both
    /// the raw `error:` line AND the `enriched:` line below it.
    ///
    /// Why bypass `explain_sidecar`'s public surface: producing
    /// an enriched parse failure end-to-end requires fabricating
    /// a sidecar whose serde error contains both `"missing field"`
    /// and ``"`host`"`` substrings (the
    /// `enriched_parse_error_message` trigger). Modern
    /// [`crate::test_support::SidecarResult`] declares `host:
    /// Option<HostContext>`, so serde tolerates absence and never
    /// emits that error — meaning the enrichment surface is
    /// otherwise unreachable from the live schema. Driving the
    /// renderer directly with a synthetic `SidecarParseError`
    /// pins the rendering invariant without depending on a
    /// schema bump that promotes `host` to non-Option.
    ///
    /// Sibling [`explain_sidecar_json_e2e_enrichment_renders_in_walk_errors`]
    /// pins the JSON-channel rendering of the same input.
    /// Together they guarantee the enrichment payload reaches
    /// both output channels — the prior coverage matrix
    /// exercised the catalog (`enriched_parse_error_message_returns_prose_for_host_missing_pattern`)
    /// and the renderer (`explain_sidecar_text_omits_enriched_line_for_generic_failure`)
    /// but never the enriched-rendering path itself.
    #[test]
    fn explain_sidecar_text_e2e_enrichment_renders_in_corrupt_block() {
        let parse_err = crate::test_support::SidecarParseError {
            path: std::path::PathBuf::from("/tmp/example-run/sidecar.ktstr.json"),
            raw_error: "missing field `host` at line 1 column 100".to_string(),
            enriched_message: Some(
                "ktstr_test: skipping /tmp/example-run/sidecar.ktstr.json: \
                 missing field `host` ... — re-run the test"
                    .to_string(),
            ),
        };
        let walk = super::WalkStats {
            walked: 1,
            valid: 0,
            errors: vec![parse_err],
            io_errors: Vec::new(),
        };
        let out = super::render_explain_sidecar_text(&[], &walk);

        // The corrupt-sidecars header MUST emit (errors vec non-empty).
        assert!(
            out.contains("corrupt sidecars (1):"),
            "non-empty errors must surface the trailing block: {out}",
        );
        // Both the `error:` line (raw serde) and the `enriched:`
        // line (catalog prose) must emit, in that order, indented
        // under the path. Asserting on the substring catches a
        // regression that drops either channel.
        assert!(
            out.contains("    error: missing field `host`"),
            "raw serde error must render verbatim: {out}",
        );
        assert!(
            out.contains("    enriched: "),
            "enriched line must render below the raw error: {out}",
        );
        // Positional ordering: the `error:` line must appear
        // BEFORE the `enriched:` line in the rendered text. A
        // regression that swaps the two would surface here.
        let error_pos = out
            .find("    error: ")
            .expect("error: substring must be present");
        let enriched_pos = out
            .find("    enriched: ")
            .expect("enriched: substring must be present");
        assert!(
            error_pos < enriched_pos,
            "raw `error:` line must precede `enriched:` line in \
             the rendered text — operator reads grep-friendly raw \
             first, then human remediation: {out}",
        );
    }

    /// JSON-channel mirror of
    /// [`explain_sidecar_text_e2e_enrichment_renders_in_corrupt_block`].
    /// Constructs a synthetic [`WalkStats`] with one enriched
    /// [`crate::test_support::SidecarParseError`] and validates
    /// that [`super::render_explain_sidecar_json`] emits the
    /// enriched prose under `_walk.errors[].enriched_message` as
    /// a JSON string (not null).
    ///
    /// Counterweight to the existing
    /// [`explain_sidecar_json_walk_errors_lists_corrupt_files`]
    /// test which only covers the generic-failure case
    /// (enriched_message: null). Without this test, a regression
    /// that dropped `enriched_message.as_deref()` and emitted a
    /// constant null would pass every prior JSON test.
    #[test]
    fn explain_sidecar_json_e2e_enrichment_renders_in_walk_errors() {
        let prose = "ktstr_test: skipping path: missing field `host` \
                     — re-run the test to regenerate";
        let parse_err = crate::test_support::SidecarParseError {
            path: std::path::PathBuf::from("/tmp/example-run/sidecar.ktstr.json"),
            raw_error: "missing field `host` at line 1 column 100".to_string(),
            enriched_message: Some(prose.to_string()),
        };
        let walk = super::WalkStats {
            walked: 1,
            valid: 0,
            errors: vec![parse_err],
            io_errors: Vec::new(),
        };
        let out = super::render_explain_sidecar_json(&[], &walk);
        let parsed: serde_json::Value =
            serde_json::from_str(&out).expect("json output must round-trip parse");

        let errors = parsed
            .get("_walk")
            .and_then(|w| w.get("errors"))
            .and_then(|e| e.as_array())
            .expect("_walk.errors must be a JSON array");
        assert_eq!(
            errors.len(),
            1,
            "synthetic input has exactly one parse error: {out}",
        );
        let entry = &errors[0];
        // `enriched_message` MUST be a JSON string carrying the
        // catalog prose verbatim — not null, not a different
        // string. A regression that emitted `null` for non-None
        // `enriched_message` would fail here.
        let enriched = entry
            .get("enriched_message")
            .and_then(|v| v.as_str())
            .expect("enriched_message must be a JSON string for enriched failures");
        assert_eq!(
            enriched, prose,
            "enriched_message must round-trip the catalog prose verbatim: {out}",
        );
        // The raw `error` field still emits alongside — both
        // channels are exposed so dashboards can pick.
        let raw = entry
            .get("error")
            .and_then(|v| v.as_str())
            .expect("error must be a JSON string");
        assert!(
            raw.contains("missing field"),
            "raw error must round-trip verbatim alongside enriched: {out}",
        );
    }

    // -- explain_sidecar path-traversal rejection (#135) --

    /// `--run` containing `..` segments must bail before any
    /// path resolution — operators using `--dir` to point at a
    /// shared archive pool should not be able to read arbitrary
    /// filesystem locations under attacker-controlled `--run`
    /// values. Tests both the leading `..` form and a mid-path
    /// `..` form so the validator does not depend on prefix
    /// matching.
    #[test]
    fn explain_sidecar_rejects_parent_dir_traversal_in_run() {
        let tmp = tempfile::tempdir().unwrap();
        for traversal in ["../escape", "subdir/../../escape"] {
            let err = super::explain_sidecar(traversal, Some(tmp.path()), false).expect_err(
                "path-traversal `..` in --run must be rejected before \
                     resolution",
            );
            let msg = format!("{err:#}");
            assert!(
                msg.contains("path-traversal"),
                "rejection message must name the cause for {traversal}: \
                 {msg}",
            );
            assert!(
                msg.contains(traversal),
                "rejection message must include the offending input \
                 ({traversal}): {msg}",
            );
        }
    }

    /// Absolute paths in `--run` must also bail. Without this
    /// guard, `--run /etc/passwd` would resolve via `root.join`
    /// to `/etc/passwd` (Rust's `PathBuf::join` replaces with
    /// the absolute argument when given one), leaking arbitrary
    /// filesystem reads. The validator rejects every component
    /// variant except `Normal`, so the bail message uses the
    /// "pool-root-aliasing or path-traversal" wording —
    /// asserting on the substring `"path-traversal"` matches
    /// the absolute-path case too.
    #[test]
    fn explain_sidecar_rejects_absolute_path_in_run() {
        let tmp = tempfile::tempdir().unwrap();
        let err = super::explain_sidecar("/etc/passwd", Some(tmp.path()), false)
            .expect_err("absolute path in --run must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path-traversal"),
            "absolute-path rejection must name the cause: {msg}",
        );
    }

    /// Empty `--run` must bail. `Path::new("").components()`
    /// yields zero components, so an iterate-and-reject
    /// validator silently accepts the input; `Path::join("")`
    /// then returns the pool root unchanged, walking every
    /// archived run instead of the requested one. The
    /// explicit empty-string check before the component loop
    /// is the gate that catches this — pin its bail behavior
    /// here so the gate cannot regress.
    #[test]
    fn explain_sidecar_rejects_empty_run() {
        let tmp = tempfile::tempdir().unwrap();
        let err = super::explain_sidecar("", Some(tmp.path()), false)
            .expect_err("empty --run must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("must not be empty"),
            "empty-string rejection must name the cause: {msg}",
        );
    }

    /// `--run .` must bail. `Path::new(".")` yields one
    /// `Component::CurDir`, which `Path::join(".")` treats as
    /// a no-op — so the unmodified pool root would be walked,
    /// aliasing every archived run. Rejecting `CurDir` in the
    /// component-match arm is the gate; this test pins it.
    /// Symmetric with the empty-string test: both inputs are
    /// pool-root aliases.
    #[test]
    fn explain_sidecar_rejects_curdir_run() {
        let tmp = tempfile::tempdir().unwrap();
        let err = super::explain_sidecar(".", Some(tmp.path()), false)
            .expect_err("`.` --run must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path-traversal"),
            "`.` rejection must surface the pool-root-aliasing \
             cause: {msg}",
        );
    }

    /// Bare run keys with `Normal`-only components must pass
    /// the traversal validator and proceed to the normal
    /// not-found / empty-run paths. Pins that the validator
    /// does not over-reject legitimate inputs after the
    /// `CurDir`-rejection tightening.
    #[test]
    fn explain_sidecar_accepts_bare_run_key_after_traversal_check() {
        let tmp = tempfile::tempdir().unwrap();
        // `6.14-abc1234` is a typical run key shape
        // (Normal-only components: `{kernel}-{project_commit}`
        // per `sidecar_dir`). It does not exist under tmp, so
        // the call lands in the not-found path — which is the
        // SECOND validation gate, proving the traversal check
        // let it through.
        let err = super::explain_sidecar("6.14-abc1234", Some(tmp.path()), false)
            .expect_err("non-existent run must surface the not-found error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not found"),
            "bare run key must reach the not-found gate, not the \
             traversal gate: {msg}",
        );
        assert!(
            !msg.contains("path-traversal"),
            "bare run key must NOT trip the traversal check: {msg}",
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

    // -- poll_child_with_timeout --

    /// Helper: spawn a long-sleeping child via `sh -c`. `sh` is in
    /// PATH on every supported platform, and the `sleep N` command
    /// is a coreutils builtin reachable from `sh`. The child PID
    /// is returned alongside the `Child` so the test can verify
    /// reaping at the OS layer.
    ///
    /// Lives in this test module rather than a shared helper file
    /// because every caller needs the PID-extraction line as well,
    /// and the inlined helper keeps the assertion-pid-check pair
    /// in one readable block.
    fn spawn_sleeping_child(seconds: u64) -> (std::process::Child, u32) {
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("sleep {seconds}"))
            .spawn()
            .expect("spawn sh -c sleep N");
        let pid = child.id();
        (child, pid)
    }

    /// Helper: probe whether `pid` is still a live process via
    /// `kill -0` semantics through `nix`. Returns `true` while the
    /// process is alive (or in zombie state — the kernel keeps the
    /// PID slot reserved until the parent reaps), `false` once
    /// the slot is fully released after reap. The helper exists
    /// to confirm the timeout path's `child.wait()` actually
    /// reaps; without the wait, the child would linger as a
    /// zombie and `kill -0` would still return Ok.
    ///
    /// Lifecycle vs zombie: `kill -0` can NOT distinguish "alive"
    /// from "zombie" — both return Ok because the PID slot is
    /// still allocated until the parent reaps. The test must
    /// instead rely on `child.try_wait`'s contract via the helper's
    /// own reap-on-timeout path: after the helper returns, the
    /// `Child` value has been moved into the helper and dropped,
    /// and the helper's explicit `child.wait()` (called before
    /// `bail!`) reaps the zombie. The kernel then reclaims the
    /// PID slot. After that point, `kill -0 pid` returns ESRCH
    /// (the helper returns false). ESRCH may take a brief moment
    /// to propagate; the assertion polls with a short timeout
    /// rather than asserting immediately.
    ///
    /// Caveat: PID reuse can theoretically cause a false-positive
    /// (the kernel reassigned the slot to an unrelated process),
    /// but on Linux the PID space is large (32k by default,
    /// 4M with the kernel.pid_max sysctl) and the test runs for
    /// ~50ms total — collision is astronomically unlikely.
    fn pid_is_alive(pid: u32) -> bool {
        use nix::sys::signal::kill;
        use nix::unistd::Pid;
        // signal=None means "send no signal, just probe" — kill(2)
        // with sig=0 is the canonical liveness probe.
        kill(Pid::from_raw(pid as i32), None).is_ok()
    }

    /// Timeout fires when the child outlives the deadline; the
    /// helper bails with the labeled timeout error AND reaps the
    /// child (no zombie persists past the helper return). Three
    /// invariants:
    ///
    /// 1. **Bail wording.** The error must include the `label`
    ///    parameter and the literal `"timed out after"` phrase
    ///    so operators can pattern-match a wedged-make scenario
    ///    in CI logs. A regression that dropped the label or
    ///    re-worded the timeout substring would surface here.
    ///
    /// 2. **Wall-clock budget.** The helper must return within a
    ///    small multiple of the configured timeout — not block
    ///    indefinitely. The 1ms poll-interval ensures the loop
    ///    notices the deadline within one tick of expiry.
    ///
    /// 3. **No zombie.** After the helper returns, the child's
    ///    PID must be reclaimed by the kernel (no zombie left
    ///    over). Probed via `kill(pid, 0)` returning ESRCH after
    ///    a short propagation poll. A regression that dropped the
    ///    `child.wait()` after `child.kill()` would leak a
    ///    zombie that this assertion catches.
    ///
    /// `sh` is in PATH on every supported runner; if it is not,
    /// the helper bails on spawn and the test fails — that's the
    /// correct signal because every other test in this module
    /// relies on `sh` being available too.
    #[test]
    fn poll_child_with_timeout_bails_and_reaps_on_timeout() {
        // Spawn a child that will outlive the timeout by orders
        // of magnitude (60s > 100ms timeout × 600x margin).
        let (child, pid) = spawn_sleeping_child(60);
        assert!(
            pid_is_alive(pid),
            "fixture precondition: spawned child pid {pid} must be \
             alive before the helper runs",
        );

        let start = std::time::Instant::now();
        let result = super::poll_child_with_timeout(
            child,
            Duration::from_millis(100),
            Duration::from_millis(1),
            "make wedged-target",
        );
        let elapsed = start.elapsed();

        // Invariant 1: bail wording carries label + timeout phrase.
        let err = result.expect_err("timed-out child must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make wedged-target"),
            "timeout bail must include the label parameter; got: {rendered}",
        );
        assert!(
            rendered.contains("timed out after"),
            "timeout bail must include the literal `timed out after` \
             phrase so CI log scrapers can pattern-match wedged builds; \
             got: {rendered}",
        );

        // Invariant 2: wall-clock budget. The 100ms timeout +
        // ~1ms poll interval should fire within ~150ms; allow
        // 5s of slack for slow CI runners and tempfile churn.
        // A regression that ignored the deadline would block
        // for the full 60-second sleep.
        assert!(
            elapsed < Duration::from_secs(5),
            "helper must return within a small multiple of the \
             configured timeout (100ms); took {elapsed:?} which \
             suggests the deadline check is broken",
        );

        // Invariant 3: no zombie. The helper's explicit
        // `child.wait()` after `child.kill()` reaps the child.
        // Poll briefly to give the kernel time to propagate the
        // ESRCH state — most systems clear the PID slot within
        // a few milliseconds of the wait, but a slow CI runner
        // may take longer. Bound the poll at 1s so a regression
        // that leaked the zombie surfaces clearly.
        let zombie_check_deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            if !pid_is_alive(pid) {
                break;
            }
            if std::time::Instant::now() >= zombie_check_deadline {
                panic!(
                    "child pid {pid} still alive 1s after helper returned — \
                     timeout path leaked a zombie (missing child.wait() \
                     after child.kill()?)",
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Successful exit before the deadline: the helper observes
    /// `Ok(Some(status))` with a successful status, returns Ok,
    /// and reaps via the natural process-exit path (no kill
    /// needed). Pins that the timeout machinery does not
    /// false-fire on a fast-exiting child.
    ///
    /// `true` exits 0 immediately; the helper's first `try_wait`
    /// tick should see the completed status. The 5s timeout is
    /// a wide margin so a slow CI runner does not flake the
    /// test on a transient scheduling delay.
    #[test]
    fn poll_child_with_timeout_succeeds_when_child_exits_clean() {
        let child = std::process::Command::new("true")
            .spawn()
            .expect("spawn true");
        let pid = child.id();

        let result = super::poll_child_with_timeout(
            child,
            Duration::from_secs(5),
            Duration::from_millis(1),
            "make happy-target",
        );
        assert!(
            result.is_ok(),
            "child that exits 0 must surface as Ok; got: {result:?}",
        );
        // Cleanup invariant: a successful exit also reaps via
        // the `Ok(Some(status))` arm's implicit Drop of `child`.
        // Verify the PID slot was freed.
        let zombie_check_deadline = std::time::Instant::now() + Duration::from_secs(1);
        loop {
            if !pid_is_alive(pid) {
                break;
            }
            if std::time::Instant::now() >= zombie_check_deadline {
                panic!(
                    "child pid {pid} still alive 1s after Ok return — \
                     successful-exit path leaked a zombie",
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Failed exit before the deadline: the helper observes
    /// `Ok(Some(status))` with an unsuccessful status and
    /// surfaces as Err with the `{label} failed` wording.
    /// Distinct from the timeout case because the bail message
    /// shape differs (`failed` vs `timed out after`); CI log
    /// scrapers must distinguish the two so wedged-make
    /// (operations issue) is not confused with build-failed
    /// (code issue).
    ///
    /// `false` exits 1 immediately; the assertion pins both the
    /// label propagation AND the `failed` wording.
    #[test]
    fn poll_child_with_timeout_surfaces_nonzero_exit_as_err() {
        let child = std::process::Command::new("false")
            .spawn()
            .expect("spawn false");
        let result = super::poll_child_with_timeout(
            child,
            Duration::from_secs(5),
            Duration::from_millis(1),
            "make broken-target",
        );
        let err = result.expect_err("child that exits non-zero must surface as Err");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("make broken-target"),
            "non-zero-exit bail must include the label; got: {rendered}",
        );
        assert!(
            rendered.contains("failed"),
            "non-zero-exit bail must use the `failed` wording so it is \
             distinguishable from the timeout-path's `timed out after`; \
             got: {rendered}",
        );
        // Negative pin: a non-zero exit is NOT a timeout, so the
        // bail message must NOT contain `timed out`. A regression
        // that conflated the two error paths would cross-wire
        // the wording and break CI log triage.
        assert!(
            !rendered.contains("timed out"),
            "non-zero-exit bail must NOT contain `timed out` — that \
             phrase belongs to the deadline-fired path only; got: {rendered}",
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

    /// Pin the per-line `str::trim` semantics in
    /// [`validate_kernel_config`]. The function builds its
    /// presence-check `HashSet` from `config.lines().map(str::trim)`,
    /// so a `.config` carrying CRLF line endings (`\r\n` from a
    /// Windows-edited file or a configure step that wrote with
    /// the wrong newline) and trailing-space lines must STILL
    /// validate when the option-after-trim equals the expected
    /// `CONFIG_X=y` form.
    ///
    /// A regression that dropped `.map(str::trim)` would cause
    /// `lines()` to yield `"CONFIG_SCHED_CLASS_EXT=y\r"` (the
    /// `\r` survives `lines()` per `str::lines` semantics —
    /// `lines` splits on `\n` and on `\r\n` strips ONLY the
    /// final `\n`, leaving the `\r`). The HashSet's contains
    /// check against the bare `CONFIG_X=y` would miss, every
    /// critical option would surface as missing, and CI on
    /// Windows-edited fragments would break silently.
    /// Independently, a trailing-space line like `"CONFIG_X=y "`
    /// without trim would also miss against the bare
    /// `CONFIG_X=y` lookup.
    ///
    /// Mixes BOTH whitespace shapes in one fixture so a
    /// regression that handled one but not the other surfaces
    /// here as a missing-options error message.
    #[test]
    fn validate_kernel_config_trim_handles_crlf_and_trailing_whitespace() {
        let dir = tempfile::TempDir::new().unwrap();
        // Mix CRLF-terminated lines and trailing-space lines so
        // both whitespace forms are exercised. Every entry in
        // VALIDATE_CONFIG_CRITICAL appears once with one of the
        // two whitespace shapes; trim must collapse both back
        // to the bare `CONFIG_X=y` form for the HashSet probe.
        std::fs::write(
            dir.path().join(".config"),
            "CONFIG_SCHED_CLASS_EXT=y\r\n\
             CONFIG_DEBUG_INFO_BTF=y \n\
             CONFIG_BPF_SYSCALL=y\r\n\
             CONFIG_FTRACE=y \n\
             CONFIG_KPROBE_EVENTS=y\r\n\
             CONFIG_BPF_EVENTS=y \n",
        )
        .unwrap();
        let result = validate_kernel_config(dir.path());
        assert!(
            result.is_ok(),
            "validate_kernel_config must trim per-line whitespace \
             before the HashSet probe — a regression dropping \
             `.map(str::trim)` would treat \\r-suffixed and \
             trailing-space lines as distinct from the bare \
             `CONFIG_X=y` form and report every option as \
             missing; got: {result:?}",
        );
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

    // -- resolve_kernel_parallelism --

    /// Unset env: returns the host-CPU fallback, never zero. The
    /// fallback chain is `available_parallelism() → 1`, so the
    /// result is always ≥ 1 — a zero return would set the rayon
    /// pool width to zero, which `ThreadPoolBuilder::build`
    /// rejects, defeating the entire fan-out pipeline.
    #[test]
    fn resolve_kernel_parallelism_unset_returns_host_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::remove(crate::KTSTR_KERNEL_PARALLELISM_ENV);
        let n = super::resolve_kernel_parallelism();
        assert!(
            n >= 1,
            "fallback must yield at least 1; got {n} which would defeat \
             ThreadPoolBuilder::num_threads",
        );
    }

    /// Valid usize override: env-supplied value wins over the
    /// host-CPU default. `KTSTR_KERNEL_PARALLELISM=4` must
    /// produce `4` regardless of the host's logical CPU count.
    #[test]
    fn resolve_kernel_parallelism_valid_override_wins() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "4");
        assert_eq!(
            super::resolve_kernel_parallelism(),
            4,
            "valid usize env value must override the host-CPU default; \
             a regression that ignored the env var would yield \
             available_parallelism() instead",
        );
    }

    /// `KTSTR_KERNEL_PARALLELISM=0`: zero is a sentinel meaning
    /// "ignore me" rather than "disable parallelism." A
    /// `ThreadPoolBuilder::num_threads(0)` does NOT bound the
    /// pool to zero — rayon's documented behavior treats `0` as
    /// "auto-size" and resolves the width via `RAYON_NUM_THREADS`
    /// (if set) or `available_parallelism()`, which would defeat
    /// the explicit cap our caller is trying to install. The
    /// explicit `n > 0` guard in `resolve_kernel_parallelism`
    /// rejects the parsed-zero case in our own helper so the
    /// host-CPU default we compute (rather than rayon's auto-
    /// sizing) drives the pool — keeping cap semantics
    /// predictable and independent of rayon's internal sizing
    /// rules and of any ambient `RAYON_NUM_THREADS` in the
    /// environment.
    #[test]
    fn resolve_kernel_parallelism_zero_falls_through_to_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "0");
        let n = super::resolve_kernel_parallelism();
        assert!(
            n >= 1,
            "zero env value must fall through to host-CPU default \
             (always ≥ 1); got {n} which would crash the pool builder",
        );
    }

    /// Unparseable value: a typoed export
    /// (`KTSTR_KERNEL_PARALLELISM=abc`) silently degrades to the
    /// default cap rather than propagating the parse error or
    /// disabling parallelism. The user gets host-CPU width with
    /// no observable signal that the env var was wrong; the
    /// alternative (bail) would block resolves on a typo, and
    /// the alternative (silently disable) would serialize
    /// multi-spec invocations with no visible cause.
    #[test]
    fn resolve_kernel_parallelism_unparseable_falls_through_to_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "abc");
        let n = super::resolve_kernel_parallelism();
        assert!(
            n >= 1,
            "unparseable env value must fall through to host-CPU \
             default (always ≥ 1); got {n}",
        );
    }

    /// Negative value: `usize::from_str` rejects the leading `-`,
    /// so the parse fails and the fallback fires. Distinct from
    /// the alphabetic-typo case because `-1` is the more likely
    /// "I expected a signed integer" mistake.
    #[test]
    fn resolve_kernel_parallelism_negative_falls_through_to_default() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "-1");
        let n = super::resolve_kernel_parallelism();
        assert!(
            n >= 1,
            "negative env value must fall through to host-CPU \
             default (usize::from_str rejects leading `-`); got {n}",
        );
    }

    /// Surrounding whitespace: a shell-quoted
    /// `KTSTR_KERNEL_PARALLELISM=" 8 "` parses as 8. Matches the
    /// trim-tolerant convention used by every other KTSTR_*
    /// reader (see `ktstr_kernel_env`'s whitespace-tolerance
    /// tests in `lib.rs`).
    #[test]
    fn resolve_kernel_parallelism_trims_surrounding_whitespace() {
        use crate::test_support::test_helpers::{EnvVarGuard, lock_env};
        let _lock = lock_env();
        let _guard = EnvVarGuard::set(crate::KTSTR_KERNEL_PARALLELISM_ENV, "  8  ");
        assert_eq!(
            super::resolve_kernel_parallelism(),
            8,
            "trimmed env value must parse; whitespace tolerance \
             matches the rest of the KTSTR_* env-reading suite",
        );
    }

    /// Pin the literal env-var name. A future rename must update
    /// every reader in lockstep (the constant is the single
    /// source of truth, but if this test fails alongside an
    /// unrelated change, it surfaces the rename to the team
    /// reviewer).
    #[test]
    fn ktstr_kernel_parallelism_env_const_matches_literal() {
        assert_eq!(
            crate::KTSTR_KERNEL_PARALLELISM_ENV,
            "KTSTR_KERNEL_PARALLELISM",
        );
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

    /// `suggest_closest_run_key` — positive case: a query that is
    /// a one-character typo of an actual run-directory leaf
    /// returns that leaf. Plants `6.14-abc1234` under a tempdir,
    /// queries `6.14-abc1235` (final byte flipped — edit distance
    /// 1), and asserts the planted name is returned. The query
    /// length (12 chars) gives a threshold of `max(3, 12/3) = 4`,
    /// so distance-1 is well inside.
    ///
    /// Sibling of `suggest_closest_test_name_finds_near_match` —
    /// same shape, different registry source (filesystem
    /// `read_dir` vs static slice).
    #[test]
    fn suggest_closest_run_key_finds_near_match() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("6.14-abc1234")).expect("plant run dir");

        let suggestion = suggest_closest_run_key("6.14-abc1235", tmp.path())
            .expect("distance-1 typo on a planted run dir must yield a suggestion");
        assert_eq!(
            suggestion, "6.14-abc1234",
            "a single-character typo must suggest the planted dir name",
        );
    }

    /// `suggest_closest_run_key` — negative case: a query whose
    /// edit distance from every candidate exceeds the threshold
    /// returns `None` rather than over-suggesting a distant match.
    /// A 13-char string of `x` against a `6.14-abc1234` candidate
    /// has Levenshtein distance >= 12 (every char differs); the
    /// threshold for a 13-char query is `max(3, 13/3) = 4`, so the
    /// candidate is correctly rejected.
    #[test]
    fn suggest_closest_run_key_returns_none_for_distant_query() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("6.14-abc1234")).expect("plant run dir");
        assert_eq!(
            suggest_closest_run_key("xxxxxxxxxxxxx", tmp.path()),
            None,
            "a query with no lexical relationship to any planted run \
             dir must yield no suggestion",
        );
    }

    /// `suggest_closest_run_key` — empty root: the helper returns
    /// `None` when `read_dir` succeeds but yields zero
    /// subdirectories. Pins the no-candidates path. Operators
    /// hitting a typo on a freshly-created runs-root see the bare
    /// bail message without a misleading near-match suggestion.
    #[test]
    fn suggest_closest_run_key_returns_none_for_empty_root() {
        let tmp = tempfile::tempdir().unwrap();
        // tempdir is fresh and contains no entries; read_dir succeeds
        // and yields zero candidates, so the helper returns None.
        assert_eq!(
            suggest_closest_run_key("6.14-abc1234", tmp.path()),
            None,
            "empty root must yield None — no candidates to match against",
        );
    }

    /// `suggest_closest_run_key` — file entries are skipped, and
    /// when both a FILE and a DIR with similar names exist, only
    /// the directory is considered a candidate. Plants
    /// `6.14-abc1234` as a regular file AND `6.14-abc1235` as a
    /// directory under one tempdir, then queries `6.14-abc1234`
    /// (the FILE's exact name). The helper must return
    /// `Some("6.14-abc1235")` — the DIRECTORY at distance 1 — and
    /// must NOT return the file at distance 0 because the file
    /// fails [`crate::test_support::is_run_directory`]'s
    /// `is_dir()` short-circuit.
    ///
    /// Catches a regression that drops the
    /// [`crate::test_support::is_run_directory`] filter and
    /// allows files to leak into the suggestion surface.
    #[test]
    fn suggest_closest_run_key_skips_files() {
        let tmp = tempfile::tempdir().unwrap();
        // File: distance 0 against the query but is NOT a dir → must be skipped.
        std::fs::write(tmp.path().join("6.14-abc1234"), b"not a dir").expect("plant file");
        // Dir: distance 1 against the query → must win because the
        // file at distance 0 is filtered out.
        std::fs::create_dir(tmp.path().join("6.14-abc1235")).expect("plant dir");

        let suggestion = suggest_closest_run_key("6.14-abc1234", tmp.path())
            .expect("the planted directory must yield a suggestion despite the same-name file");
        assert_eq!(
            suggestion, "6.14-abc1235",
            "a regression that drops the is_dir() filter would surface \
             here as `Some(\"6.14-abc1234\")` (the file at distance 0) \
             instead of `Some(\"6.14-abc1235\")` (the dir at distance 1)",
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
    /// `cpus`, `cache`, `run_dirs`. Downstream consumers of
    /// `ktstr locks --json` (shell scripts piping through `jq`, the
    /// mdbook recipe pages, future dashboards) parse against these
    /// names — a refactor that renames them would silently break
    /// every consumer.
    ///
    /// Also pins the `rename_all = "snake_case"` contract on the
    /// nested row structs: LlcLockRow's "llc_idx" and "numa_node",
    /// RunDirLockRow's "run_key", must NOT emit as camelCase.
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
            run_dirs: vec![RunDirLockRow {
                run_key: "6.14-abc1234".to_string(),
                lockfile: "/tmp/.locks/6.14-abc1234.lock".to_string(),
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
        assert!(
            val.get("run_dirs").is_some(),
            "top-level must have 'run_dirs': {val}"
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
        // Nested RunDir row — run_key stays snake_case.
        let run0 = &val["run_dirs"][0];
        assert!(run0.get("run_key").is_some(), "run_key: {run0}");
        assert!(run0.get("lockfile").is_some(), "lockfile: {run0}");
        assert!(run0.get("holders").is_some(), "holders: {run0}");
    }

    /// `collect_locks_snapshot_from` on a fresh tempdir with no
    /// ktstr lockfiles returns an empty LocksSnapshot (all four
    /// row vectors empty). Production wrapper always sees the same
    /// behavior when `/tmp` has no `ktstr-*.lock` files, the cache
    /// dir has no `.locks/` subdirectory, and the runs root has
    /// no `.locks/` subdirectory.
    #[test]
    fn collect_locks_snapshot_empty_roots() {
        use tempfile::TempDir;
        let tmp_dir = TempDir::new().expect("tempdir tmp_root");
        let cache_dir = TempDir::new().expect("tempdir cache_root");
        let runs_dir = TempDir::new().expect("tempdir runs_root");
        let snap = collect_locks_snapshot_from(
            tmp_dir.path(),
            Some(cache_dir.path()),
            Some(runs_dir.path()),
        )
        .expect("collect must succeed on empty roots");
        assert!(snap.llcs.is_empty(), "no ktstr-llc-*.lock → empty llcs");
        assert!(snap.cpus.is_empty(), "no ktstr-cpu-*.lock → empty cpus");
        assert!(snap.cache.is_empty(), "no .locks/ → empty cache");
        assert!(
            snap.run_dirs.is_empty(),
            "no .locks/ under runs_root → empty run_dirs",
        );
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
        let snap = collect_locks_snapshot_from(path, None, None).expect("collect must succeed");
        // LLC rows, ascending.
        assert_eq!(snap.llcs.len(), 2);
        assert_eq!(snap.llcs[0].llc_idx, 2, "sort ascending: llc 2 first");
        assert_eq!(snap.llcs[1].llc_idx, 5, "sort ascending: llc 5 second");
        // CPU row.
        assert_eq!(snap.cpus.len(), 1);
        assert_eq!(snap.cpus[0].cpu, 7);
        // Cache row empty because cache_root=None.
        assert!(snap.cache.is_empty());
        // Run-dir row empty because runs_root=None.
        assert!(snap.run_dirs.is_empty());
    }

    /// `collect_locks_snapshot_from` discovers synthetic per-run-key
    /// lockfiles planted under `{runs_root}/.locks/*.lock`. Pins:
    /// run_key parses as the file stem (the `{kernel}-{project_commit}`
    /// dirname), rows sort ascending by run_key, and the cache /
    /// run-dir scans live in independent code paths (passing
    /// `cache_root=None` does NOT suppress run-dir enumeration).
    #[test]
    fn collect_locks_snapshot_discovers_run_dir_lockfiles() {
        use tempfile::TempDir;
        let runs_dir = TempDir::new().expect("tempdir runs_root");
        let locks_dir = runs_dir.path().join(crate::flock::LOCK_DIR_NAME);
        std::fs::create_dir_all(&locks_dir).expect("mkdir .locks/");
        // Plant two run-dir lockfiles out of order — snapshot must
        // sort ascending. Use realistic run-key shapes.
        std::fs::write(locks_dir.join("7.0-def5678.lock"), b"").expect("plant 7.0");
        std::fs::write(locks_dir.join("6.14-abc1234.lock"), b"").expect("plant 6.14");
        // tmp_root tempdir is empty so the LLC/CPU scans yield no
        // rows — keeps the assertion focused on the run_dirs path.
        let tmp_dir = TempDir::new().expect("tempdir tmp_root");
        let snap = collect_locks_snapshot_from(tmp_dir.path(), None, Some(runs_dir.path()))
            .expect("collect must succeed");
        assert_eq!(snap.run_dirs.len(), 2);
        assert_eq!(
            snap.run_dirs[0].run_key, "6.14-abc1234",
            "sort ascending: 6.14 lexically before 7.0",
        );
        assert_eq!(snap.run_dirs[1].run_key, "7.0-def5678");
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

    // -- inverted-range diagnostic wiring --

    /// `resolve_cached_kernel` must call `KernelId::validate()` BEFORE
    /// the generic "not yet supported" bail so an inverted range
    /// surfaces the actionable "swap the endpoints" diagnostic
    /// instead of getting masked by the redirect. A future regression
    /// that drops the validate() call would flip the error text from
    /// the specific message to the generic one, landing here.
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
        assert!(
            msg.contains("6.12..6.16"),
            "swap suggestion must appear in the error; got: {msg}",
        );
        // Ensure we did NOT get the generic bail's text.
        assert!(
            !msg.contains("not yet supported in this context"),
            "validate() must short-circuit before the generic bail; got: {msg}",
        );
    }

    /// Companion for `resolve_kernel_image`: same wiring guarantee,
    /// different entry point. The function takes `&str` not
    /// `&KernelId` so we pass the raw spec; internally `KernelId::parse`
    /// produces a Range, then `validate()` rejects.
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
            "validate() diagnostic must surface ahead of the generic \
             'not yet supported' bail; got: {msg}",
        );
        assert!(
            msg.contains("6.12..6.16"),
            "swap suggestion must appear in the error; got: {msg}",
        );
        assert!(
            !msg.contains("not yet supported in this context"),
            "validate() must short-circuit before the generic bail; got: {msg}",
        );
    }

    // -- expand_kernel_range / filter_and_sort_range --

    fn release(moniker: &str, version: &str) -> crate::fetch::Release {
        crate::fetch::Release {
            moniker: moniker.to_string(),
            version: version.to_string(),
        }
    }

    /// Stable+longterm rows inside the interval are kept; mainline,
    /// linux-next, and rows outside the interval are dropped. The
    /// surviving versions sort ascending by `(major, minor, patch,
    /// rc)` regardless of input order, so a regression that left
    /// the releases.json order leaking through (newest-first instead
    /// of oldest-first per the coordinator's ascending ruling) lands
    /// here.
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
            "stable+longterm only, ascending, [start, end] inclusive",
        );
    }

    /// Endpoints that are absent from releases.json still bracket the
    /// surviving versions correctly. `6.10..6.16` brackets `6.12`,
    /// `6.14`, and `6.15` even when none of `6.10` / `6.16` themselves
    /// appear as rows. Pins the "interval is half-the-numeric, half-
    /// presence" semantics documented on `expand_kernel_range`.
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

    /// Inclusive endpoint comparison: `6.12.5..6.14.2` keeps the
    /// versions matching either endpoint exactly. A regression to
    /// strict inequality (`<` / `>`) would silently drop both
    /// endpoints from the result and land here.
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

    /// rc-tagged rows under stable/longterm monikers (kernel.org
    /// publishes rcs under `mainline` so this is a synthetic input)
    /// are still kept by the moniker filter — but ordering relative
    /// to non-rc versions follows the rc-as-MAX rule from
    /// `decompose_version_for_compare`. Synthetic case to pin the
    /// ordering invariant in case kernel.org ever ships an rc under
    /// a stable moniker.
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
        // rc-as-MAX: `6.14.0` (rc=MAX) sorts STRICTLY ABOVE
        // `6.14.0-rc3` (rc=3). Operators expecting "rc before final"
        // must read the comment on `decompose_version_for_compare`.
        assert_eq!(
            out,
            vec![
                "6.13.0".to_string(),
                "6.14.0-rc3".to_string(),
                "6.14.0".to_string(),
            ],
        );
    }

    /// Empty interval (no stable+longterm rows fall inside the
    /// bounds) returns an empty vec from the pure helper. The
    /// network-touching `expand_kernel_range` wrapper translates that
    /// into the actionable "expanded to 0 stable releases" bail; this
    /// pure layer just reports nothing matched, leaving the bail
    /// decision to the outer layer.
    #[test]
    fn filter_and_sort_range_empty_when_no_overlap() {
        use crate::kernel_path::decompose_version_for_compare;
        let releases = vec![release("stable", "5.10.0"), release("stable", "5.15.0")];
        let start_key = decompose_version_for_compare("6.10").unwrap();
        let end_key = decompose_version_for_compare("6.16").unwrap();
        let out = filter_and_sort_range(&releases, start_key, end_key);
        assert!(out.is_empty(), "no overlap → empty result, got {out:?}");
    }

    /// Mainline/linux-next/etc. monikers are dropped even when they
    /// fall inside the interval. Pins the stable+longterm-only filter
    /// the coordinator's ruling A specified.
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
        assert_eq!(
            out,
            vec!["6.14.5".to_string()],
            "only stable/longterm survive the filter"
        );
    }

    /// Unparseable version strings (a kernel.org row whose `version`
    /// field doesn't match the major.minor[.patch][-rcN] grammar) are
    /// dropped silently rather than aborting the whole expansion. A
    /// future kernel.org schema change that introduced a new format
    /// (e.g. an embargoed CVE patch tag) would land here as one
    /// untestable row, not a hard failure for the entire run.
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
        assert_eq!(out, vec!["6.14.0".to_string(), "6.14.5".to_string()],);
    }

    /// The wrapper `expand_kernel_range` rejects unparseable range
    /// endpoints up front before any network call. Pins the
    /// validation gate at the public API surface.
    #[test]
    fn expand_kernel_range_rejects_unparseable_start() {
        let err = expand_kernel_range("garbage", "6.14", "ktstr-test")
            .expect_err("unparseable start must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("kernel range start `garbage`"),
            "error must cite the bad endpoint, got: {msg}"
        );
    }

    #[test]
    fn expand_kernel_range_rejects_unparseable_end() {
        let err = expand_kernel_range("6.10", "garbage", "ktstr-test")
            .expect_err("unparseable end must error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("kernel range end `garbage`"),
            "error must cite the bad endpoint, got: {msg}"
        );
    }

    /// `kernel list --range R` with a non-Range spec must reject
    /// at parse time with a diagnostic naming the expected shape,
    /// before any network fetch. A bare version `6.14.2` parses
    /// as `KernelId::Version`, NOT `KernelId::Range`, so the
    /// preview path must surface a parse error rather than fall
    /// through into `expand_kernel_range` with synthesized
    /// endpoints.
    #[test]
    fn kernel_list_range_preview_rejects_non_range_spec() {
        let err = run_kernel_list_range(false, "6.14.2")
            .expect_err("bare version must not parse as a Range");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("does not parse as a `START..END` range"),
            "error must name the expected range shape, got: {msg}"
        );
        assert!(
            msg.contains("`6.14.2`"),
            "error must cite the bad input verbatim, got: {msg}"
        );
    }

    /// `kernel list --range` with an inverted range must surface
    /// the `validate()` diagnostic ("swap the endpoints") rather
    /// than trying to fetch and silently expanding to zero
    /// versions. The preview path must run the same validation
    /// gate every real resolver runs.
    #[test]
    fn kernel_list_range_preview_rejects_inverted_range() {
        let err = run_kernel_list_range(false, "6.16..6.12")
            .expect_err("inverted range must not be accepted");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("kernel list --range 6.16..6.12"),
            "error must cite the operator-supplied range, got: {msg}"
        );
    }

    // ---------------------------------------------------------------
    // resolve_kernel_dir_to_entry — success-path tests
    // ---------------------------------------------------------------
    //
    // Error paths (nonexistent path, not-a-source-tree) live next to
    // [`resolve_path_kernel`] in `bin/cargo-ktstr.rs`. The success
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
        // Stage a fake image as the source path for the store
        // (which COPIES the bytes into the cache atomically).
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

        // Compute the cache key the same way `local_source` would.
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
    ///
    /// The test asserts the bypass directly: pre-populate the
    /// cache entry under the DIRTY tree's `local-unknown-...`
    /// key (so a cache lookup from the dirty resolve WOULD
    /// match if the gate were absent) and confirm the resolve
    /// does NOT short-circuit on it. The actual build pipeline
    /// then fails on the non-kernel fixture, which is
    /// independent evidence that the dirty path entered the
    /// build branch rather than the cache-hit branch.
    ///
    /// `KTSTR_BYPASS_LLC_LOCKS=1` skips the resource-budget
    /// reservation since this test does not represent a real
    /// build to coordinate against peer measurements.
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

        // Dirty the tree FIRST so the `local-unknown-...` key
        // shape is the one the resolver would look up.
        std::fs::write(src_tmp.path().join("README"), "modified\n").expect("dirty README");
        let dirty_acquired = crate::fetch::local_source(src_tmp.path())
            .expect("local_source on dirty tree must succeed");
        assert!(
            dirty_acquired.is_dirty,
            "post-mutation tree must be dirty for the test to be meaningful"
        );
        // Pre-populate under the EXACT key the dirty tree would
        // hit if the gate were absent. A regression that drops
        // the `if !is_dirty` short-circuit would land this entry
        // as the outcome and the assertion below would fail.
        populate_cache_entry_for_resolve_test(cache_tmp.path(), &dirty_acquired.cache_key);

        let result = resolve_kernel_dir_to_entry(src_tmp.path(), "test", None);
        // The dirty path must not return the pre-populated entry
        // as a cache hit. The build pipeline then fails on the
        // fixture (no real kernel toolchain), surfacing as Err —
        // that is the expected outcome. Any `Ok(KernelDirOutcome
        // { cache_hit: Some(_), .. })` would prove the dirty gate
        // regressed.
        match result {
            Ok(outcome) => panic!(
                "dirty tree must skip the cache lookup, but resolve returned \
                 Ok with dir={:?}, cache_hit={:?}, is_dirty={}",
                outcome.dir, outcome.cache_hit, outcome.is_dirty,
            ),
            Err(_) => {
                // The pre-populated entry is still on disk; the
                // resolve did not consume it.
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
    ///
    /// `KTSTR_BYPASS_LLC_LOCKS=1` skips the resource-budget
    /// reservation (LLC flocks + cgroup v2 sandbox) since the
    /// fixture has no real build to coordinate. The build's `make`
    /// subprocess still runs and still fails on the
    /// non-kernel-target Makefile — that is the assertion's
    /// substrate.
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

        // No pre-populated cache entry → cache miss must reach
        // the build pipeline. The Makefile fixture has no real
        // kernel targets, so the resulting build attempt fails;
        // we observe that as evidence the miss path was taken
        // rather than a silent no-op surfacing as `Ok`.
        let acquired =
            crate::fetch::local_source(src_tmp.path()).expect("local_source must succeed");
        assert!(!acquired.is_dirty, "fixture must be clean before resolve");

        let result = resolve_kernel_dir_to_entry(src_tmp.path(), "test", None);
        // Either the build attempt fails (kernel_build_pipeline's
        // make subprocess doesn't find real kernel targets) or
        // there's an environmental skip (no make / kvm / cgroup);
        // both are acceptable evidence that the cache MISS path
        // entered the build pipeline rather than masking as
        // cache-hit.
        assert!(
            result.is_err(),
            "cache miss without a real kernel toolchain must surface the build failure, \
             got Ok({:?})",
            result.as_ref().ok().map(|o| &o.dir),
        );
    }
}
