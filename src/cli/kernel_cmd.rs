//! `kernel` subcommand definition and help-text constants.
//!
//! Holds the `KernelCommand` enum (shared by `ktstr` and
//! `cargo-ktstr`), the `--help` text constants for kernel-related
//! flags (`--kernel`, `--cpu-cap`, `--extra-kconfig`, `--disk`),
//! and the legend / footer helpers that flow through `kernel list`'s
//! tag-emission gates.

use std::path::{Path, PathBuf};

use clap::Subcommand;

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
        /// Path to an additional kconfig fragment merged on top of
        /// the baked-in `ktstr.kconfig`.
        ///
        /// # Format
        ///
        /// One declaration per line, same shapes the kernel itself
        /// uses:
        ///
        /// ```text
        /// # comment lines start with `#` and a space
        /// CONFIG_FOO=y                  # boolean enable
        /// CONFIG_FOO=m                  # build as module
        /// CONFIG_FOO=n                  # disable (equivalent to is-not-set)
        /// CONFIG_BAR="some value"       # string
        /// CONFIG_BAR=42                 # integer / hex
        /// # CONFIG_FOO is not set       # explicit disable directive
        /// ```
        ///
        /// The baked-in fragment lives at `ktstr.kconfig` in the
        /// ktstr repository root. See [`EMBEDDED_KCONFIG`] for the
        /// const that loads it at compile time.
        ///
        /// # Conflict resolution
        ///
        /// User values win on conflict — kbuild's `.config` parser
        /// (`scripts/kconfig/confdata.c::conf_read_simple`) emits
        /// "override: reassigning to symbol X" and keeps the
        /// last-occurring assignment, so appending the user fragment
        /// AFTER the baked-in fragment makes user values take
        /// precedence. Non-conflicting user lines combine with the
        /// baked-in set verbatim.
        ///
        /// Override warnings: `kernel build` emits one
        /// `tracing::warn!` per user line that overrides a baked-in
        /// symbol (format: "--extra-kconfig overrides baked-in
        /// CONFIG_FOO (was =y, now =n)"). The build proceeds; the
        /// warning lets the operator see they are shadowing a
        /// baked-in setting before make olddefconfig runs.
        ///
        /// # Dependency resolution
        ///
        /// `make olddefconfig` runs after the merge to resolve any
        /// added symbols' dependencies. Options whose deps are not
        /// met land as `# CONFIG_X is not set` in the final
        /// `.config`; those silent drops surface as `tracing::warn!`
        /// lines (not errors) so the operator sees the diagnostic
        /// without the build failing.
        ///
        /// # Critical-symbol protection
        ///
        /// After build, [`super::validate_kernel_config`] rejects entries
        /// that disabled symbols required by ktstr (CONFIG_BPF,
        /// CONFIG_DEBUG_INFO_BTF, CONFIG_FTRACE,
        /// CONFIG_SCHED_CLASS_EXT, etc.). The error names
        /// `--extra-kconfig` as the likely cause when extras were
        /// supplied. So a fragment with
        /// `# CONFIG_BPF is not set` will fail
        /// `validate_kernel_config` post-build with an actionable
        /// message — the override warning fires pre-build and the
        /// validation error fires post-build, giving the operator
        /// two chances to catch a fatal override.
        ///
        /// # Caching
        ///
        /// The cache key suffix grows from `kc{baked}` to
        /// `kc{baked}-xkc{extra}` when extras are present (see
        /// [`crate::cache_key_suffix_with_extra`]). Two builds
        /// with distinct extra-kconfig content land at distinct
        /// cache entries (different content = cache miss; same
        /// content = cache hit on re-run). Builds with NO
        /// `--extra-kconfig` keep using the bare `kc{baked}` suffix,
        /// so existing cached kernels are not orphaned. An
        /// `--extra-kconfig`-built kernel is only addressable by a
        /// matching `--extra-kconfig` invocation or by an explicit
        /// `--source` / `KTSTR_KERNEL` path — `cargo ktstr test
        /// --kernel 6.14.2` (which doesn't take `--extra-kconfig`)
        /// will not surface the extra-built artifact.
        ///
        /// `kernel list` tags entries built with extras as
        /// `(extra kconfig)` so an operator can spot which cached
        /// kernels carry user modifications.
        #[arg(long = "extra-kconfig", value_name = "PATH", help = EXTRA_KCONFIG_HELP)]
        extra_kconfig: Option<PathBuf>,
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

/// Short clap-help for `--extra-kconfig`. Mirrors the [`CPU_CAP_HELP`]
/// pattern: terse first sentence on the clap surface, full rustdoc
/// on the [`KernelCommand::Build`] variant for `--help` long-about.
///
/// The full rustdoc covers: accepted line shapes, kbuild last-wins
/// rule, `make olddefconfig` dependency resolution, post-build
/// `validate_kernel_config` interaction, two-segment cache key,
/// override warnings, and the unaddressable-from-other-flags
/// rationale.
pub const EXTRA_KCONFIG_HELP: &str = "Additional kconfig fragment merged on top of \
     the baked-in `ktstr.kconfig`. Same line shapes the kernel uses: \
     `CONFIG_FOO=y`, `CONFIG_FOO=m`, `CONFIG_FOO=\"value\"`, and \
     `# CONFIG_FOO is not set`. User values win on conflict; \
     `make olddefconfig` resolves dependencies. Each unique fragment \
     produces a distinct cache slot via the `kc{baked}-xkc{extra}` \
     key suffix. After build, `validate_kernel_config` rejects \
     entries that disabled critical baked-in symbols \
     (CONFIG_SCHED_CLASS_EXT, CONFIG_DEBUG_INFO_BTF, CONFIG_BPF_SYSCALL, \
     CONFIG_FTRACE, CONFIG_KPROBE_EVENTS, CONFIG_BPF_EVENTS). \
     The baked-in fragment lives at `ktstr.kconfig` in the ktstr \
     repository root.";

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
/// mirrors the Rust-doc schema on [`super::kernel_list`]; keeping both
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
    "built_at, ktstr_kconfig_hash (nullable), extra_kconfig_hash\n",
    "(nullable), kconfig_status, eol, config_hash (nullable),\n",
    "image_name, image_path, has_vmlinux, vmlinux_stripped.\n",
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
    "  extra_kconfig_hash\n",
    "                   CRC32 of the user `--extra-kconfig` fragment\n",
    "                   (raw bytes, no canonicalization), or null when\n",
    "                   the entry was built without --extra-kconfig.\n",
    "                   The cache key suffix grows from `kc{baked}` to\n",
    "                   `kc{baked}-xkc{extra}` when extras are present,\n",
    "                   and this field stores the `xkc` segment so\n",
    "                   `kernel list` is self-describing for entries\n",
    "                   that carry user modifications.\n",
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
     (pre-dates kconfig hash tracking). Rebuild with: kernel build --force VERSION \
     (add --extra-kconfig PATH if the original entry was built with a user fragment).";

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
     Rebuild with: kernel build --force <entry version> \
     (add --extra-kconfig PATH if the entry also carries the (extra kconfig) tag).";

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

#[cfg(test)]
mod tests {
    use super::*;

    /// `eol_legend_if_any` is the sole gate on whether the text
    /// output under `kernel list` emits the `(EOL)` legend.
    #[test]
    fn eol_legend_if_any_branches() {
        assert_eq!(eol_legend_if_any(true), Some(EOL_EXPLANATION));
        assert_eq!(eol_legend_if_any(false), None);
    }

    /// `untracked_legend_if_any` mirrors `eol_legend_if_any`.
    #[test]
    fn untracked_legend_if_any_branches() {
        assert_eq!(
            untracked_legend_if_any(true),
            Some(UNTRACKED_KCONFIG_EXPLANATION),
        );
        assert_eq!(untracked_legend_if_any(false), None);
    }

    /// `stale_legend_if_any` completes the kconfig legend pair.
    #[test]
    fn stale_legend_if_any_branches() {
        assert_eq!(stale_legend_if_any(true), Some(STALE_KCONFIG_EXPLANATION));
        assert_eq!(stale_legend_if_any(false), None);
    }

    /// `STALE_KCONFIG_EXPLANATION` shape pin.
    #[test]
    fn stale_kconfig_explanation_shape() {
        assert!(STALE_KCONFIG_EXPLANATION.starts_with("warning"));
        assert!(STALE_KCONFIG_EXPLANATION.contains("(stale kconfig)"));
        assert!(STALE_KCONFIG_EXPLANATION.contains("different ktstr.kconfig"));
        assert!(STALE_KCONFIG_EXPLANATION.contains("kernel build --force <entry version>"));
    }

    /// `corrupt_footer_if_any` branches.
    #[test]
    fn corrupt_footer_if_any_branches() {
        let root = std::path::Path::new("/tmp/ktstr-cache-test-root");
        assert_eq!(corrupt_footer_if_any(0, root), None);
        let one = corrupt_footer_if_any(1, root).expect("positive count must yield Some(footer)");
        assert!(one.contains("1 corrupt entry."));
        assert!(one.contains("cargo ktstr kernel clean --corrupt-only"));
        assert!(one.contains(&format_corrupt_footer(root)));
        let many = corrupt_footer_if_any(3, root).expect("positive count must yield Some(footer)");
        assert!(many.contains("3 corrupt entries."));
    }

    /// Pin design decision: `(corrupt)` first sentence IS the
    /// legend; the footer carries it AND the operational
    /// remediation block.
    #[test]
    fn corrupt_footer_is_self_documenting() {
        let root = std::path::Path::new("/tmp/ktstr-cache-test-root");
        let footer = format_corrupt_footer(root);
        let first_sentence = footer
            .split_once(". ")
            .map(|(head, _)| head)
            .expect("footer must terminate legend sentence with period-space");
        assert!(first_sentence.contains("(corrupt)"));
        assert!(first_sentence.contains("cannot be used"));
        for reason_token in ["metadata is missing", "malformed", "missing image"] {
            assert!(
                first_sentence.contains(reason_token),
                "legend sentence must enumerate corruption modes; \
                 expected `{reason_token}`, got: {first_sentence:?}",
            );
        }
        assert!(footer.contains(&root.display().to_string()));
        assert!(footer.contains("kernel clean --corrupt-only --force"));
        assert!(footer.contains("kernel clean --force"));
        assert!(footer.contains("kernel clean --keep N --force"));
        assert!(footer.contains("ALL cached entries"));
        let pos_corrupt_only = footer
            .find("kernel clean --corrupt-only --force")
            .expect("--corrupt-only must appear");
        let pos_force = footer
            .find("kernel clean --force")
            .expect("--force must appear");
        let pos_keep = footer
            .find("kernel clean --keep N --force")
            .expect("--keep must appear");
        assert!(pos_corrupt_only < pos_force);
        assert!(pos_force < pos_keep);
    }

    /// `DIRTY_TREE_CACHE_SKIP_HINT` shape pin.
    #[test]
    fn dirty_tree_cache_skip_hint_shape() {
        assert!(DIRTY_TREE_CACHE_SKIP_HINT.contains("skipping cache"));
        assert!(DIRTY_TREE_CACHE_SKIP_HINT.contains("uncommitted changes"));
        assert!(
            DIRTY_TREE_CACHE_SKIP_HINT.contains("commit")
                && DIRTY_TREE_CACHE_SKIP_HINT.contains("stash")
        );
    }

    /// `NON_GIT_TREE_CACHE_SKIP_HINT` shape pin.
    #[test]
    fn non_git_tree_cache_skip_hint_shape() {
        assert!(NON_GIT_TREE_CACHE_SKIP_HINT.starts_with("skipping cache"));
        assert!(NON_GIT_TREE_CACHE_SKIP_HINT.contains("not a git repository"));
        assert!(NON_GIT_TREE_CACHE_SKIP_HINT.contains("put the source under git"));
        assert!(NON_GIT_TREE_CACHE_SKIP_HINT.contains("kernel build VERSION"));
        assert!(NON_GIT_TREE_CACHE_SKIP_HINT.contains("kernel build --git URL --ref REF"));
        assert!(!NON_GIT_TREE_CACHE_SKIP_HINT.contains("stash"));
        assert!(!NON_GIT_TREE_CACHE_SKIP_HINT.contains("commit"));
    }

    /// `untracked_legend_names_the_tag_word` — legend mentions tag.
    #[test]
    fn untracked_legend_names_the_tag_word() {
        assert!(UNTRACKED_KCONFIG_EXPLANATION.contains("(untracked kconfig)"));
    }

    /// kernel_clean rejects `--keep` together with `--corrupt-only`.
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

    /// `kernel build --cpu-cap N` parses to `Build { cpu_cap: Some(N) }`.
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

    /// `kernel build` without `--cpu-cap` parses with cpu_cap: None.
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
                assert_eq!(cpu_cap, None, "no --cpu-cap must produce None, not Some(0)");
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `kernel build --cpu-cap 0` passes clap (validation runs at runtime).
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
                assert_eq!(cpu_cap, Some(0));
            }
            other => panic!("expected KernelCommand::Build, got {other:?}"),
        }
    }

    /// `KERNEL_LIST_LONG_ABOUT` drives `kernel list --help` and must
    /// expose the `--json` output contract so scripted consumers can
    /// discover the schema from the terminal alone. Pins:
    /// 1. the `(EOL)` legend text appears verbatim at the head;
    /// 2. every top-level wrapper field appears;
    /// 3. every valid-entry field appears;
    /// 4. each `Option<T>` field carries a `(nullable)` tag;
    /// 5. each `KernelSource` variant tag and `kconfig_status`
    ///    enum value is documented.
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
                 schema without `cargo doc`",
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
            "git_hash",
            "\"ref\"",
            "source_tree_path",
        ] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(valid_entry_field),
                "KERNEL_LIST_LONG_ABOUT must mention valid-entry JSON \
                 field `{valid_entry_field}`",
            );
        }

        assert!(
            KERNEL_LIST_LONG_ABOUT.contains("error"),
            "KERNEL_LIST_LONG_ABOUT must mention corrupt-entry JSON \
             field `error` so consumers know the corrupt-entry shape",
        );

        for nullable_field in ["version", "ktstr_kconfig_hash", "config_hash"] {
            let marker = format!("{nullable_field} (nullable)");
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(&marker),
                "KERNEL_LIST_LONG_ABOUT must mark `{nullable_field}` \
                 as `(nullable)` (expected substring `{marker}`)",
            );
        }

        for source_variant_tag in ["\"tarball\"", "\"git\"", "\"local\""] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(source_variant_tag),
                "KERNEL_LIST_LONG_ABOUT must list source variant tag \
                 `{source_variant_tag}`",
            );
        }

        for status_variant in ["\"matches\"", "\"stale\"", "\"untracked\""] {
            assert!(
                KERNEL_LIST_LONG_ABOUT.contains(status_variant),
                "KERNEL_LIST_LONG_ABOUT must list kconfig_status variant \
                 `{status_variant}`",
            );
        }
    }

    /// Pin that the `#[command(long_about = KERNEL_LIST_LONG_ABOUT)]`
    /// attribute on `KernelCommand::List` is wired through clap.
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
            .expect("`list` subcommand must have a long_about set")
            .to_string();
        assert_eq!(long_about, KERNEL_LIST_LONG_ABOUT);
    }
}
