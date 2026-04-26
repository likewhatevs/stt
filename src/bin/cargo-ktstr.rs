#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::{ArgAction, CommandFactory, Parser, Subcommand};
use ktstr::cache::{CacheDir, CacheEntry};
use ktstr::cli;
use ktstr::cli::KernelCommand;
use ktstr::cli::{KERNEL_HELP_NO_RAW, KERNEL_HELP_RAW_OK};
use ktstr::fetch;

#[derive(Parser)]
#[command(name = "cargo-ktstr", bin_name = "cargo")]
struct Cargo {
    #[command(subcommand)]
    command: CargoSub,
}

#[derive(Subcommand)]
enum CargoSub {
    /// ktstr dev workflow: build kernel + run tests.
    Ktstr(Ktstr),
}

#[derive(Parser)]
struct Ktstr {
    #[command(subcommand)]
    command: KtstrCommand,
}

// Same rationale as `StatsCommand`'s sibling `#[allow]` — clap's
// derive expands every variant into a struct of `Option<T>` /
// `Vec<T>` per CLI flag, which after #117's per-side slicing
// flags pushes the Stats-via-Compare variant past clippy's
// large-variant heuristic. The enum is constructed once per CLI
// invocation and dispatched immediately; boxing every variant
// would distort the match ergonomics without measurable benefit.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum KtstrCommand {
    /// Build the kernel (if needed) and run tests via cargo nextest.
    #[command(visible_alias = "nextest")]
    Test {
        /// Repeatable. See [`KERNEL_HELP_NO_RAW`] for accepted shapes
        /// (path, version, cache key, range `START..END`, git source
        /// `git+URL#REF`). Multiple `--kernel` flags fan out the
        /// gauntlet across kernels: each `(test × scenario × topology
        /// × flags × kernel)` tuple becomes a distinct nextest test
        /// case so nextest's parallelism, retries, and `-E`
        /// filtering all apply natively.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Build and run tests with the release profile
        /// (`--cargo-profile release` to nextest).
        ///
        /// Release mode uses STRICTER assertion thresholds
        /// (`gap_threshold_ms` 2000 vs debug's 3000, `spread_threshold_pct`
        /// 15% vs debug's 35%) — tests that barely pass in debug may
        /// fail under `--release`. `catch_unwind`-based tests are
        /// skipped because release sets `panic = "abort"` (see
        /// `Cargo.toml [profile.release]`). Tests gated on
        /// `#[cfg(debug_assertions)]` also skip.
        #[arg(long)]
        release: bool,
        /// Arguments passed through to cargo nextest run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build the kernel (if needed) and run tests with coverage via
    /// cargo llvm-cov nextest. For other llvm-cov subcommands
    /// (`report`, `clean`, `show-env`), use `cargo ktstr llvm-cov`.
    Coverage {
        /// Repeatable. Same shapes and multi-kernel semantics as
        /// `cargo ktstr test --kernel`: each (test × kernel) variant
        /// runs as its own nextest subprocess so cargo-llvm-cov
        /// merges every variant's profraw automatically.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Build and collect coverage with the release profile
        /// (`--cargo-profile release` to llvm-cov nextest).
        ///
        /// Release mode uses STRICTER assertion thresholds
        /// (`gap_threshold_ms` 2000 vs debug's 3000, `spread_threshold_pct`
        /// 15% vs debug's 35%) — tests that barely pass in debug may
        /// fail under `--release`. `catch_unwind`-based tests are
        /// skipped because release sets `panic = "abort"`.
        #[arg(long)]
        release: bool,
        /// Arguments passed through to cargo llvm-cov nextest.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Run `cargo llvm-cov` with arbitrary arguments.
    ///
    /// When you want `cargo llvm-cov nextest`, prefer `cargo ktstr
    /// coverage` — this subcommand is the raw passthrough for
    /// `llvm-cov` invocations that don't fit the coverage flow
    /// (e.g. `report`, `clean`, `show-env`).
    ///
    /// Note: bare `cargo ktstr llvm-cov` (no subcommand) dispatches
    /// to `cargo llvm-cov` which runs `cargo test` — not useful for
    /// ktstr tests. Always pass a subcommand.
    LlvmCov {
        /// Repeatable. Same shapes and multi-kernel semantics as
        /// `cargo ktstr test --kernel`. Profraw aggregation across
        /// kernel variants happens inside cargo-llvm-cov; this raw-
        /// passthrough hands every other argument to the user's
        /// chosen llvm-cov subcommand.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Arguments passed through to cargo llvm-cov.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Print sidecar analysis from the most recent test run.
    ///
    /// Reads sidecar JSON files from the newest subdirectory under
    /// `{CARGO_TARGET_DIR or "target"}/ktstr/` (overridable with
    /// `KTSTR_SIDECAR_DIR`) and prints gauntlet analysis, BPF
    /// verifier stats, callback profile, and KVM stats. Each test
    /// run is its own subdirectory keyed `{kernel}-{timestamp}`;
    /// the runs ARE the baselines.
    ///
    /// Use `list` to see runs; `compare <a> <b>` to diff two.
    Stats {
        #[command(subcommand)]
        command: Option<StatsCommand>,
    },
    /// Manage cached kernel images.
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    /// Manage the LLM model cache used by `OutputFormat::LlmExtract`
    /// payloads. `fetch` downloads the default pinned model to
    /// `~/.cache/ktstr/models/` (respecting `KTSTR_CACHE_DIR` /
    /// `XDG_CACHE_HOME`); `status` reports whether a SHA-checked copy
    /// is already cached.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Collect BPF verifier statistics for a scheduler.
    ///
    /// Builds the scheduler (or uses a pre-built binary), boots a VM,
    /// loads the scheduler's BPF programs, and reports per-program
    /// verified instruction counts from host-side memory introspection.
    Verifier {
        /// Scheduler package name to build and analyze.
        #[arg(long)]
        scheduler: Option<String>,
        /// Path to pre-built scheduler binary (alternative to --scheduler).
        #[arg(long, conflicts_with = "scheduler")]
        scheduler_bin: Option<PathBuf>,
        /// Repeatable. See [`KERNEL_HELP_NO_RAW`] for accepted shapes
        /// (path / version / cache key / range / git source). When
        /// the resolved set has 2+ entries, the verifier collects
        /// stats per kernel sequentially and outputs per-kernel
        /// blocks separated by a header line — there is no
        /// cross-kernel summary table. Single-kernel runs are
        /// unchanged. Raw image files are rejected here for the
        /// same reason as `cargo ktstr test`/`coverage`/`llvm-cov`:
        /// the verifier needs the cached `vmlinux` and kconfig
        /// fragment alongside the image, which a bare `bzImage`
        /// path does not carry.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Print raw verifier output without formatting.
        #[arg(long)]
        raw: bool,
        /// Run verifier for all flag profiles. Discovers flags via
        /// `--ktstr-list-flags`, constructs profiles (power set
        /// respecting requires dependencies), and collects verifier
        /// stats per profile.
        #[arg(long)]
        all_profiles: bool,
        /// Run verifier for specific profiles only (comma-separated
        /// names, e.g. `default,llc,llc+steal`). Implies --all-profiles
        /// for flag discovery.
        #[arg(long, value_delimiter = ',')]
        profiles: Vec<String>,
    },
    /// Generate shell completions for cargo-ktstr.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
        /// Binary name for completions.
        #[arg(long, default_value = "cargo")]
        binary: String,
    },
    /// Print the current host context used by sidecar collection:
    /// CPU identity, memory/hugepage config, transparent-hugepage
    /// policy, NUMA node count, kernel uname triple
    /// (sysname/release/machine), kernel cmdline, and every
    /// `/proc/sys/kernel/sched_*` tunable. Useful for diagnosing
    /// cross-run regressions that trace back to host-context drift
    /// (sysctl change, THP policy flip, hugepage reservation).
    ///
    /// For historical drift between archived runs, use
    /// `cargo ktstr stats compare` — its host-delta section
    /// reports which host-context fields changed between run A
    /// and run B using the same [`HostContext::diff`] logic.
    ShowHost,
    /// Print the resolved assertion thresholds for the named test.
    ///
    /// Dumps the merged `Assert` produced by the runtime merge chain
    /// `Assert::default_checks().merge(entry.scheduler.assert()).merge(&entry.assert)`
    /// — the same value `run_ktstr_test_inner` evaluates against
    /// worker reports. Surfaces every threshold field (or `none`
    /// when inherited / unset) so an operator can see what the test
    /// will actually check against without reading source or
    /// guessing which layer contributed each bound.
    ///
    /// Fails with an actionable message when no registered test
    /// matches the given name. Use `cargo nextest list` to
    /// enumerate test names — then pass just the FUNCTION-NAME
    /// component to `show-thresholds`, not the `<binary>::`
    /// prefix that nextest prepends to each line. The
    /// `#[ktstr_test]` registry keys on the bare function name,
    /// so `ktstr::preempt_regression_fault_under_load` (as
    /// printed by nextest) must be trimmed to
    /// `preempt_regression_fault_under_load` before it resolves.
    ShowThresholds {
        /// Function-name-only test identifier as registered in
        /// `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`).
        /// Do NOT include the `<binary>::` prefix that
        /// `cargo nextest list` prepends — strip it before
        /// invoking this command.
        test: String,
    },
    /// Clean up leftover ktstr cgroups.
    ///
    /// Without `--parent-cgroup`, scans `/sys/fs/cgroup` for the default
    /// ktstr parents (`ktstr` and `ktstr-<pid>`, the paths that `ktstr
    /// run` and the in-process test harness create) and rmdirs each.
    /// `ktstr-<pid>` directories whose pid is still a running ktstr or
    /// cargo-ktstr process are skipped, so a concurrent cleanup run
    /// doesn't yank an active run's cgroup.
    Cleanup {
        /// Parent cgroup path. When set, cleans only this path and
        /// leaves the parent directory in place; when omitted, scans
        /// `/sys/fs/cgroup` for the default ktstr parents
        /// (`ktstr/` and `ktstr-<pid>/`) and rmdirs each.
        #[arg(long)]
        parent_cgroup: Option<String>,
    },
    /// Enumerate every ktstr flock held on this host.
    ///
    /// Troubleshooting companion for `--cpu-cap` contention. Scans
    /// `/tmp/ktstr-llc-*.lock`, `/tmp/ktstr-cpu-*.lock`, and
    /// `{cache_root}/.locks/*.lock`, cross-referenced against
    /// `/proc/locks` via [`ktstr::cli::list_locks`] to name the holder
    /// process (PID + cmdline) for each held lock. Read-only — does
    /// NOT attempt any flock acquire.
    Locks {
        /// Emit the snapshot as JSON (compact object under --watch,
        /// pretty-printed otherwise). Stable field names; schema
        /// documented at [`ktstr::cli::list_locks`].
        #[arg(long)]
        json: bool,
        /// Redraw the snapshot on the given interval until SIGINT.
        /// Value is parsed by `humantime`: `100ms`, `1s`, `5m`, `1h`.
        /// Human output clears and redraws in place; `--json` emits
        /// one line-terminated object per interval (ndjson-style).
        #[arg(long, value_parser = humantime::parse_duration)]
        watch: Option<std::time::Duration>,
    },
    /// Boot an interactive shell in a KVM virtual machine.
    ///
    /// Launches a VM with busybox and drops into a shell. Files and
    /// directories passed via -i are available at /include-files/<name>
    /// inside the guest. Directories are walked recursively, preserving
    /// structure. Dynamically-linked ELF binaries get automatic shared
    /// library resolution via ELF DT_NEEDED parsing.
    Shell {
        #[arg(long, help = KERNEL_HELP_RAW_OK)]
        kernel: Option<String>,
        /// Virtual topology as "numa_nodes,llcs,cores,threads".
        #[arg(long, default_value = "1,1,1,1")]
        topology: String,
        /// Files or directories to include in the guest. Repeatable.
        #[arg(short = 'i', long = "include-files", action = ArgAction::Append)]
        include_files: Vec<PathBuf>,
        /// Guest memory in MB (minimum 128). When absent, estimated
        /// from payload and include file sizes.
        #[arg(long = "memory-mb", value_parser = clap::value_parser!(u32).range(128..))]
        memory_mb: Option<u32>,
        /// Forward kernel console (COM1/dmesg) to stderr in real-time.
        /// Sets loglevel=7 for verbose kernel output.
        #[arg(long)]
        dmesg: bool,
        /// Run a command in the VM instead of an interactive shell.
        /// The VM exits after the command completes.
        #[arg(long)]
        exec: Option<String>,

        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,

        /// Reserve only N host CPUs for the shell VM. Requires
        /// `--no-perf-mode` — perf-mode already holds every LLC
        /// exclusively, so capping under perf-mode would
        /// double-reserve. See `ktstr::cli::CPU_CAP_HELP` for the
        /// full contract.
        #[arg(long, requires = "no_perf_mode", help = ktstr::cli::CPU_CAP_HELP)]
        cpu_cap: Option<usize>,
    },
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download the default pinned model and check its SHA-256.
    /// No-op when the cache already holds a SHA-checked copy.
    /// Respects `KTSTR_MODEL_OFFLINE` — set to `1` to refuse network
    /// fetches.
    Fetch,
    /// Print the cache path for the default model and whether a
    /// SHA-checked copy is already present.
    Status,
}

// `clippy::large_enum_variant` triggers because clap's argument
// derives produce variant-sized cells of `Option<String>` /
// `Option<PathBuf>` per CLI flag. Boxing each variant would
// distort every match arm's pattern shape (`Some(StatsCommand::
// Compare { .. })` becomes `Some(StatsCommand::Compare(box))`)
// and force every dispatch site through an extra deref. The enum
// is constructed once per CLI invocation and immediately
// pattern-matched into a single subcommand call — no allocation
// hot path, no cache pressure. Suppress at the enum level rather
// than wrapping each variant in `Box`.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum StatsCommand {
    /// List test runs under `{CARGO_TARGET_DIR or "target"}/ktstr/`.
    List,
    /// List the registered regression metrics and their default
    /// thresholds.
    ///
    /// Enumerates the `ktstr::stats::METRICS` registry: metric name,
    /// polarity (higher/lower better), default absolute-delta gate,
    /// default relative-delta gate, display unit, and a one-line
    /// description. Use this to see which metric names
    /// `ComparisonPolicy.per_metric_percent` keys can reference, and
    /// what each default_abs / default_rel gate starts at before an
    /// override.
    ///
    /// Default output is a human-readable table; `--json` emits a
    /// JSON array with the same fields (the row accessor function is
    /// omitted — `#[serde(skip)]` in the registry).
    ListMetrics {
        /// Emit JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
    /// List the distinct values present per filterable dimension in
    /// the sidecar pool.
    ///
    /// Walks every run directory under `runs_root()` (or `--dir`),
    /// pools the sidecars, and reports the set of distinct values
    /// found across all eight filterable dimensions: `kernel`,
    /// `commit`, `kernel_commit`, `source`, `scheduler`,
    /// `topology`, `work_type`, and `flags` (individual flag
    /// names). The JSON keys `commit` and `source` map to the
    /// internal `SidecarResult::project_commit` /
    /// `SidecarResult::run_source` fields; the per-side filter
    /// flags spell `--project-commit` / `--run-source` on the
    /// `compare` subcommand. Use this before crafting a
    /// `cargo ktstr stats compare` invocation to discover what
    /// `--a-X` / `--b-X` values the pool actually carries — a
    /// `--a-kernel 6.20` against an empty pool fails downstream
    /// with "no rows match filter A", and `list-values` is the
    /// upstream answer to "what kernels do I have?".
    ///
    /// Default output renders one block per dimension with values
    /// one per line; `--json` emits a single JSON object keyed by
    /// dimension name. The four optional dimensions (`kernel`,
    /// `commit`, `kernel_commit`, `source`) surface absent values
    /// as the textual sentinel `unknown` in the table shape and as
    /// JSON `null` in the JSON shape.
    ListValues {
        /// Emit JSON instead of a per-dimension text block.
        #[arg(long)]
        json: bool,
        /// Alternate run root to walk. Defaults to
        /// `test_support::runs_root()` (typically `target/ktstr/`).
        /// Same semantics as `cargo ktstr stats compare --dir` and
        /// `cargo ktstr stats show-host --dir`: useful when
        /// inspecting archived sidecar trees copied off a CI host.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
    },
    /// Print the archived host context for a specific run.
    ///
    /// Resolves `--run <id>` against `test_support::runs_root()`
    /// (or `--dir` when set), loads any sidecar file under that
    /// run directory, and renders the `host` field via
    /// `HostContext::format_human`. Useful for inspecting the
    /// CPU model, memory config, THP policy, and sched_* tunables
    /// captured at archive time — the same fingerprint
    /// `compare_partitions` uses for its host-delta section, now
    /// available on a single run.
    ///
    /// Scans sidecars in iteration order and returns the FIRST
    /// sidecar with a populated host field. Every sidecar in a
    /// single run captures the same host, but older pre-
    /// enrichment sidecars may have `host: None`; the forward
    /// scan tolerates those without false-failing as long as at
    /// least one sidecar carries the data. If NO sidecar has a
    /// populated host field, the command fails with an actionable
    /// error naming the likely cause (pre-enrichment run) rather
    /// than silently returning empty output.
    ShowHost {
        /// Run key (from `cargo ktstr stats list`).
        #[arg(long)]
        run: String,
        /// Alternate run root to resolve `--run` against. Defaults
        /// to `test_support::runs_root()` (typically
        /// `target/ktstr/`). Same semantics as
        /// `cargo ktstr stats compare --dir`.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
    },
    /// Diagnose missing optional fields across a run's sidecars.
    ///
    /// Loads every `*.ktstr.json` under `--run <id>` and reports,
    /// per sidecar, which optional fields landed as null along
    /// with the documented reasons each one can be missing. Every
    /// such field carries a classification:
    ///
    /// - `expected` — null is the steady-state shape; no operator
    ///   action recovers it (e.g. payload metadata for a
    ///   scheduler-only test).
    /// - `actionable` — null indicates a recoverable gap;
    ///   re-running in a different environment (in-repo cwd,
    ///   non-tarball kernel, non-host-only test) would populate
    ///   the field.
    ///
    /// Different gauntlet variants on the same run legitimately
    /// differ on which fields populate (host-only vs VM-backed,
    /// scheduler-only vs payload-bearing), so the report is
    /// per-sidecar rather than aggregate.
    ///
    /// Sidecars are loaded verbatim. Diverges intentionally from
    /// `stats compare` / `stats list-values` (which rewrite the
    /// `run_source` field to `"archive"` when `--dir` is set):
    /// the override would erase the only signal that surfaces a
    /// pre-rename archive whose `run_source` field was lost on
    /// load. Matches `stats show-host` semantics.
    ///
    /// Default output is per-sidecar text blocks with a header
    /// line reporting walked / parsed counts (so a corrupt
    /// `.ktstr.json` file surfaces as a parse-failure delta
    /// against the file count). `--json` emits a single object
    /// with `_walk` carrying the same counts and `fields`
    /// carrying one aggregated entry per optional field with a
    /// run-wide `none_count`.
    ExplainSidecar {
        /// Run key (from `cargo ktstr stats list`).
        #[arg(long)]
        run: String,
        /// Alternate run root to resolve `--run` against.
        /// Defaults to `target/ktstr/`. Same semantics as
        /// `cargo ktstr stats compare --dir`.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// Emit aggregate JSON instead of per-sidecar text. The
        /// text shape is per-sidecar (different gauntlet variants
        /// have different None patterns); the JSON shape is
        /// across-the-run aggregate by field, suitable for
        /// dashboards and CI ingestion.
        #[arg(long)]
        json: bool,
    },
    /// Compare two filter-defined partitions of the sidecar pool
    /// and report regressions across slicing dimensions.
    ///
    /// Each `--a-X` / `--b-X` pair pins a different value on
    /// dimension `X` for the A and B sides; the dimensions on
    /// which A and B differ are the SLICING dimensions, the
    /// dimensions on which they agree are the PAIRING dimensions
    /// the comparison joins on. Shared `--X` flags pin BOTH sides
    /// to the same value (sugar that narrows pre-slicing scope).
    /// Per-side `--a-X` / `--b-X` flags REPLACE the corresponding
    /// shared `--X` value for that side — "more-specific replaces."
    Compare {
        /// Substring filter. Matches against scenario, topology,
        /// scheduler, work_type.
        #[arg(short = 'E', long)]
        filter: Option<String>,
        /// Uniform relative significance threshold in percent
        /// (e.g. 10 for 10%). When set, overrides the per-metric
        /// default threshold for ALL metrics — intentionally, so
        /// callers can loosen a tight default or tighten a loose
        /// one from the CLI without per-metric knobs. Omit to use
        /// each metric's built-in default.
        ///
        /// Sugar for `--policy` with `{default_percent: N}` and an
        /// empty per-metric map. Mutually exclusive with `--policy`
        /// — if you need per-metric overrides, spell them out in a
        /// policy file and pass `--policy`.
        #[arg(long, conflicts_with = "policy")]
        threshold: Option<f64>,
        /// Path to a JSON-persisted `ktstr::cli::ComparisonPolicy`
        /// file with per-metric thresholds. Mutually exclusive
        /// with `--threshold`. Use `--threshold` as sugar for a
        /// uniform default; use `--policy` for the per-metric
        /// override map.
        ///
        /// Priority: per-metric override → `default_percent` →
        /// each metric's registry `default_rel`.
        ///
        /// Schema (every field optional; empty object produces
        /// the "registry defaults everywhere" policy):
        ///
        ///   {
        ///     "default_percent": 10.0,
        ///     "per_metric_percent": {
        ///       "worst_spread": 5.0,
        ///       "worst_p99_wake_latency_us": 20.0,
        ///       "worst_mean_run_delay_us": 15.0
        ///     }
        ///   }
        ///
        /// Values are PERCENT (e.g. `10.0` → 10%). Negative
        /// values are rejected. Per-metric keys must match a
        /// metric name in the `METRICS` registry — a typo
        /// (e.g. `wrost_spread`) is rejected at load time so it
        /// does not silently fall through to `default_percent`.
        /// Use `cargo ktstr stats list-metrics` to discover
        /// available metric names and their default thresholds.
        #[arg(long, conflicts_with = "threshold")]
        policy: Option<std::path::PathBuf>,
        /// Alternate run root to resolve `a` / `b` against. Defaults
        /// to `test_support::runs_root()` (typically `target/ktstr/`).
        /// Useful when comparing archived sidecar trees copied off a
        /// CI host.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
        /// Match against the sidecar's `kernel_version` field
        /// (e.g. `--kernel 6.14.2`). **Match shape depends on
        /// segment count**: a two-segment value (`--kernel 6.12`)
        /// is a major.minor PREFIX — it matches `6.12`, `6.12.0`,
        /// `6.12.5`, etc., letting the operator narrow on a stable
        /// series without naming every patch release. A
        /// three-or-more-segment value (`--kernel 6.14.2`,
        /// `--kernel 6.15-rc3`) is STRICT EQUALITY — `6.14.2` does
        /// NOT match `6.14.20`. See [`kernel_filter_matches`] in
        /// stats.rs for the cutoff implementation.
        ///
        /// Repeatable: `--kernel A --kernel B` keeps rows whose
        /// `kernel_version` equals A OR B (each value applies its
        /// own match shape per the segment-count rule). Rows whose
        /// `kernel_version` is `None` (sidecar writer could not
        /// extract a version) NEVER match a populated filter —
        /// passing `--kernel` is an opt-in that demands a
        /// known-version row. Same flag name as on `cargo ktstr
        /// test`/`coverage`/`llvm-cov` for consistency: every
        /// subcommand that accepts a kernel filter spells it
        /// `--kernel`. The per-side overrides `--a-kernel` /
        /// `--b-kernel` carry the same match-shape rule.
        #[arg(long, action = ArgAction::Append)]
        kernel: Vec<String>,
        /// Repeatable OR-combined filter on the sidecar's
        /// `project_commit` field (e.g. `--project-commit abcdef1`
        /// or `--project-commit abcdef1-dirty`).
        /// `--project-commit A --project-commit B` keeps rows
        /// whose `project_commit` equals A OR B; each entry uses
        /// strict equality (no prefix matching — `abcdef1` does
        /// not match `abcdef10`). Rows whose `project_commit` is
        /// `None` (sidecar writer's gix probe failed, or cwd was
        /// outside any git repo at write time) NEVER match a
        /// populated filter — same opt-in policy as `--kernel`.
        ///
        /// Filters on the ktstr framework commit
        /// (`SidecarResult::project_commit`); the scheduler
        /// binary's commit (`SidecarResult::scheduler_commit`,
        /// currently always `None`) is a separate concept and is
        /// not currently exposed as a filter.
        ///
        /// The recorded commit is whatever
        /// `detect_project_commit` reads from `gix::discover`
        /// walking up from the test process's cwd at sidecar-write
        /// time; the `-dirty` suffix lands when HEAD-vs-index or
        /// index-vs-worktree changes are detected, so a clean run
        /// and a dirty run of the same HEAD bucket separately
        /// under this filter.
        ///
        /// Symmetric with `--kernel-commit` (which filters on the
        /// kernel SOURCE TREE commit). Together the pair lets the
        /// operator narrow on either or both commit dimensions.
        #[arg(long = "project-commit", action = ArgAction::Append)]
        project_commit: Vec<String>,
        /// Repeatable OR-combined filter on the sidecar's
        /// `kernel_commit` field (e.g. `--kernel-commit abcdef1`
        /// or `--kernel-commit abcdef1-dirty`).
        /// `--kernel-commit A --kernel-commit B` keeps rows whose
        /// `kernel_commit` equals A OR B; each entry uses strict
        /// equality (no prefix matching — `abcdef1` does not
        /// match `abcdef10`). Rows whose `kernel_commit` is
        /// `None` (KTSTR_KERNEL pointed at a non-git path, the
        /// underlying source was Tarball / Git rather than a
        /// `Local` tree, or `detect_kernel_commit`'s gix probe
        /// failed) NEVER match a populated filter — same opt-in
        /// policy as `--project-commit` / `--kernel`.
        ///
        /// Filters on the kernel SOURCE TREE commit
        /// (`SidecarResult::kernel_commit`), NOT on the kernel
        /// release version (`SidecarResult::kernel_version` —
        /// filter that with `--kernel`). Two runs of the same
        /// `kernel_version` with different `kernel_commit` values
        /// represent the same release rebuilt from different
        /// trees (e.g. WIP patches on top of a tagged release);
        /// `--kernel-commit` distinguishes them, `--kernel` does
        /// not.
        ///
        /// The recorded value is whatever
        /// `detect_kernel_commit` reads via
        /// `gix::open(<kernel-dir>)` (NOT `discover` — the
        /// kernel directory is explicit, not walked-up); the
        /// `-dirty` suffix lands when HEAD-vs-index or
        /// index-vs-worktree changes are detected, so a clean
        /// kernel tree and a dirty one at the same HEAD bucket
        /// separately under this filter.
        #[arg(long, action = ArgAction::Append)]
        kernel_commit: Vec<String>,
        /// Repeatable OR-combined filter on the sidecar's
        /// `scheduler` field (e.g. `--scheduler scx_rusty`).
        /// `--scheduler A --scheduler B` keeps rows whose
        /// `scheduler` equals A OR B; each entry uses strict
        /// equality (no prefix matching).
        /// Distinct from `-E`, which matches a substring across
        /// the joined fields. Use this when the operator wants to
        /// pin specific schedulers rather than narrow on a
        /// fragment. Empty (no `--scheduler` flag) is the no-op
        /// default and matches every row's scheduler.
        #[arg(long, action = ArgAction::Append)]
        scheduler: Vec<String>,
        /// Repeatable OR-combined filter on the rendered topology
        /// label (e.g. `--topology 1n2l4c2t`). The label is what
        /// `Topology::Display` produces; `cargo ktstr stats list`
        /// shows the form per-row. `--topology A --topology B`
        /// keeps rows whose `topology` equals A OR B; each entry
        /// uses strict equality (no prefix matching). Empty is
        /// the no-op default.
        #[arg(long, action = ArgAction::Append)]
        topology: Vec<String>,
        /// Repeatable OR-combined filter on the sidecar's
        /// `work_type` field (e.g. `--work-type CpuSpin`). Valid
        /// names are the PascalCase variants of `WorkType`. See
        /// `WorkType::ALL_NAMES` for the canonical variant list, or
        /// `doc/guide/src/concepts/work-types.md`. `--work-type A
        /// --work-type B` keeps rows whose `work_type` equals A OR
        /// B; each entry uses strict equality (no prefix
        /// matching). Empty is the no-op default.
        #[arg(long = "work-type", action = ArgAction::Append)]
        work_type: Vec<String>,
        /// Repeatable OR-combined filter on the sidecar's
        /// `run_source` field (e.g. `--run-source local`,
        /// `--run-source ci`, `--run-source archive`).
        /// `--run-source A --run-source B` keeps rows whose
        /// `run_source` equals A OR B; each entry uses strict
        /// equality (case-sensitive, no prefix matching). Rows
        /// whose `run_source` is `None` (sidecar pre-dates the
        /// field) NEVER match a populated filter — same opt-in
        /// policy as `--kernel` / `--project-commit` /
        /// `--kernel-commit`.
        ///
        /// Filters on the run-environment provenance recorded by
        /// `detect_run_source` at sidecar-write time (`"local"`
        /// for developer runs, `"ci"` when `KTSTR_CI` was set),
        /// or rewritten to `"archive"` at load time when this
        /// command's `--dir` flag points at a non-default pool
        /// root. Combine with `--a-run-source` / `--b-run-source`
        /// to contrast across run environments (e.g.
        /// `--a-run-source ci --b-run-source local` to diff CI
        /// runs against developer runs of the same scenarios).
        ///
        /// Named `--run-source` (rather than `--source`) to
        /// disambiguate from `KernelSource` — every other
        /// `source`-shaped CLI surface in the workspace
        /// (`kernel build --source`, `KernelMetadata.source`)
        /// refers to a kernel-source kind, not a run-environment
        /// tag.
        #[arg(long = "run-source", action = ArgAction::Append)]
        run_source: Vec<String>,
        /// Repeatable AND-combined flag filter (e.g.
        /// `--flag llc --flag rusty_balance`). Every flag listed
        /// must be present in the sidecar's `active_flags`; the row
        /// may carry additional flags beyond the filter set. Empty
        /// repeats are rejected by clap (zero-width match).
        #[arg(long = "flag")]
        flags: Vec<String>,
        /// A-side overrides: replace the corresponding shared
        /// `--X` value for the A side only. See the per-side
        /// semantics on each `--X` flag's doc.
        ///
        /// `--a-kernel` carries the same match-shape rule as the
        /// shared `--kernel`: a two-segment value (e.g.
        /// `--a-kernel 6.12`) is a major.minor PREFIX matching
        /// every patch release in that series; a three-or-more-
        /// segment value (`6.14.2`, `6.15-rc3`) is strict
        /// equality. NOT strict equality across the board — see
        /// [`kernel_filter_matches`] for the cutoff implementation.
        #[arg(long = "a-kernel", action = ArgAction::Append)]
        a_kernel: Vec<String>,
        #[arg(long = "a-project-commit", action = ArgAction::Append)]
        a_project_commit: Vec<String>,
        #[arg(long = "a-kernel-commit", action = ArgAction::Append)]
        a_kernel_commit: Vec<String>,
        #[arg(long = "a-run-source", action = ArgAction::Append)]
        a_run_source: Vec<String>,
        #[arg(long = "a-scheduler", action = ArgAction::Append)]
        a_scheduler: Vec<String>,
        #[arg(long = "a-topology", action = ArgAction::Append)]
        a_topology: Vec<String>,
        #[arg(long = "a-work-type", action = ArgAction::Append)]
        a_work_type: Vec<String>,
        #[arg(long = "a-flag")]
        a_flags: Vec<String>,

        /// B-side overrides: replace the corresponding shared
        /// `--X` value for the B side only. See the per-side
        /// semantics on each `--X` flag's doc.
        ///
        /// `--b-kernel` carries the same match-shape rule as the
        /// shared `--kernel`: a two-segment value (e.g.
        /// `--b-kernel 6.12`) is a major.minor PREFIX matching
        /// every patch release in that series; a three-or-more-
        /// segment value (`6.14.2`, `6.15-rc3`) is strict
        /// equality. NOT strict equality across the board — see
        /// [`kernel_filter_matches`] for the cutoff implementation.
        #[arg(long = "b-kernel", action = ArgAction::Append)]
        b_kernel: Vec<String>,
        #[arg(long = "b-project-commit", action = ArgAction::Append)]
        b_project_commit: Vec<String>,
        #[arg(long = "b-kernel-commit", action = ArgAction::Append)]
        b_kernel_commit: Vec<String>,
        #[arg(long = "b-run-source", action = ArgAction::Append)]
        b_run_source: Vec<String>,
        #[arg(long = "b-scheduler", action = ArgAction::Append)]
        b_scheduler: Vec<String>,
        #[arg(long = "b-topology", action = ArgAction::Append)]
        b_topology: Vec<String>,
        #[arg(long = "b-work-type", action = ArgAction::Append)]
        b_work_type: Vec<String>,
        #[arg(long = "b-flag")]
        b_flags: Vec<String>,

        /// Disable averaging. By default the comparison folds
        /// every matching sidecar within each side into a single
        /// arithmetic-mean row per pairing key; `--no-average`
        /// keeps each sidecar distinct and bails with an
        /// actionable diagnostic if multiple sidecars on the
        /// same side share the same pairing key (otherwise
        /// pairing across A/B sides is ambiguous).
        ///
        /// Aggregation rules under the default (averaging-on)
        /// path: failing/skipped contributors are excluded from
        /// the metric mean (they carry failure-mode telemetry,
        /// not scheduler behaviour); the aggregated row's
        /// `passed` is the AND across every contributor (a
        /// single failure flips the aggregate to `failed`,
        /// which routes the pair through `compare_rows`'
        /// `skipped_failed` gate).
        #[arg(long = "no-average")]
        no_average: bool,
    },
}

/// Configure if needed and build the kernel.
///
/// Returns `anyhow::Result<()>` so the five `cli::*` calls below
/// chain directly with `?` — the outer bin surface is still
/// `Result<(), String>` (a broader anyhow migration across
/// cargo-ktstr.rs is pending), and callers bridge at the
/// boundary. Internally, anyhow lets this function propagate the
/// already-`anyhow::Error` returns from `cli::run_make` /
/// `configure_kernel` / `make_kernel_with_output` /
/// `validate_kernel_config` / `run_make_with_output` without a
/// `.map_err(|e| format!("{e:#}"))` stringification per call.
fn build_kernel(kernel_dir: &Path, clean: bool) -> anyhow::Result<()> {
    if !kernel_dir.is_dir() {
        anyhow::bail!("{}: not a directory", kernel_dir.display());
    }

    if clean {
        eprintln!("cargo ktstr: make mrproper");
        cli::run_make(kernel_dir, &["mrproper"])?;
    }

    if !cli::has_sched_ext(kernel_dir) {
        cli::Spinner::with_progress("Configuring kernel...", "Kernel configured", |_| {
            cli::configure_kernel(kernel_dir, cli::EMBEDDED_KCONFIG)
        })?;
    }

    cli::Spinner::with_progress("Building kernel...", "Kernel built", |sp| {
        cli::make_kernel_with_output(kernel_dir, Some(sp), None)
    })?;

    cli::validate_kernel_config(kernel_dir)?;

    cli::Spinner::with_progress("Generating compile_commands.json...", "Done", |sp| {
        cli::run_make_with_output(kernel_dir, &["compile_commands.json"], Some(sp))
    })?;
    Ok(())
}

/// Resolve a `KernelId::Path` to a canonicalized, ready-to-build
/// directory and trigger the auto-build pipeline. Returns the
/// canonical path suitable for export via [`ktstr::KTSTR_KERNEL_ENV`].
///
/// The previous inline `unwrap_or_else(|_| PathBuf::from(&p))`
/// silently fell back to the raw string on canonicalize failure —
/// that behaviour is preserved here as a hard error so a typo can't
/// exec into a successful-looking run that silently picks up a
/// different kernel via `find_kernel`'s fallback chain.
fn resolve_path_kernel(p: &Path) -> Result<PathBuf, String> {
    let dir = std::fs::canonicalize(p).map_err(|e| {
        format!(
            "--kernel {}: path does not exist or cannot be \
             canonicalized ({e:#}). {hint}",
            p.display(),
            hint = ktstr::KTSTR_KERNEL_HINT,
        )
    })?;
    // Boundary bridge: `build_kernel` returns `anyhow::Result<()>`
    // while this function returns `Result<_, String>`, so we
    // stringify at the call site. A broader anyhow migration across
    // cargo-ktstr.rs is pending and would drop this last bridge.
    build_kernel(&dir, false).map_err(|e| format!("{e:#}"))?;
    Ok(dir)
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
fn canonicalize_cache_dir(cache_dir: PathBuf) -> PathBuf {
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
fn resolve_one(id: ktstr::kernel_path::KernelId) -> Result<(String, PathBuf), String> {
    use ktstr::kernel_path::KernelId;
    match id {
        KernelId::Path(p) => {
            let dir = resolve_path_kernel(&p)?;
            let label = path_kernel_label(&dir);
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

fn resolve_kernel_set(specs: &[String]) -> Result<Vec<(String, PathBuf)>, String> {
    use ktstr::kernel_path::KernelId;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    preflight_collision_check(specs)?;

    // Each spec resolves independently:
    //   - Path → just canonicalize on disk (no network).
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
fn preflight_collision_check(specs: &[String]) -> Result<(), String> {
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
fn dedupe_resolved(resolved: Vec<(String, PathBuf)>) -> Vec<(String, PathBuf)> {
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
fn detect_label_collisions(resolved: &[(String, PathBuf)]) -> Result<(), String> {
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
fn path_kernel_label(dir: &Path) -> String {
    let basename = dir.file_name().and_then(|n| n.to_str()).unwrap_or("kernel");
    let hash = crc32fast::hash(dir.display().to_string().as_bytes());
    // `{:08x}` would emit 8 hex digits; ruling specifies a 6-char
    // hash prefix. Truncating to the leading 6 is sufficient
    // disambiguation for the operator's purpose (collision risk is
    // only a UI nuisance, not a correctness issue — the kernel_dir
    // path itself is the actual identity).
    format!("path_{basename}_{:06x}", hash & 0x00ff_ffff)
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
fn cache_key_to_version_label(key: &str) -> std::borrow::Cow<'_, str> {
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
fn git_kernel_label(url: &str, git_ref: &str) -> String {
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
fn encode_kernel_list(resolved: &[(String, PathBuf)]) -> Result<String, String> {
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

/// Shared runner for `cargo ktstr test`, `cargo ktstr coverage`, and
/// `cargo ktstr llvm-cov`.
///
/// All three subcommands share the same plumbing: resolve `--kernel`
/// to a flat `(label, kernel_dir)` set, propagate `--no-perf-mode`
/// via an env var, optionally prepend `--cargo-profile release`,
/// append the user's trailing args, and `cmd.exec()` once. The
/// cargo subcommand name (`["nextest","run"]` vs `["llvm-cov",
/// "nextest"]` vs `["llvm-cov"]`) and the log / error-message
/// prefix are the only static differences.
///
/// Multi-kernel fan-out lives entirely in the test binary's
/// gauntlet expansion (`src/test_support/dispatch.rs`): when the
/// resolved set has more than one entry, the test binary's
/// `--list` handler prints `gauntlet/{name}/{preset}/{profile}/
/// {kernel_label}` for every kernel and the `--exact` handler
/// strips the kernel suffix and re-exports `KTSTR_KERNEL` to that
/// kernel's directory before booting the VM. `cargo nextest`
/// already handles parallelism, retries, and `-E` filtering;
/// cargo-ktstr never spawns its own loop.
///
/// Empty `--kernel` (the default): no `KTSTR_KERNEL` /
/// `KTSTR_KERNEL_LIST` export — the test binary resolves its own
/// kernel via the existing `find_kernel` chain.
///
/// Single-entry `--kernel` (one Path / Version / CacheKey / Git, OR a
/// Range that expanded to exactly one release): export
/// `KTSTR_KERNEL` only. Test names stay backward-compatible — no
/// kernel suffix is appended in `--list` output.
///
/// Multi-entry `--kernel` (≥ 2 entries after expansion): export
/// `KTSTR_KERNEL_LIST` AND set `KTSTR_KERNEL` to the first entry so
/// downstream code that reads `KTSTR_KERNEL` directly (e.g. budget
/// listing in dispatch.rs that needs ANY kernel for vmlinux probe)
/// still gets a valid path. The test binary's `--list` / `--exact`
/// handlers prefer `KTSTR_KERNEL_LIST` when set.
///
/// `release` is always `false` for the raw `llvm-cov` passthrough —
/// that subcommand hands every argument to the user, so the profile
/// is set via the user's trailing args (or not at all). `test` and
/// `coverage` wire their `--release` flag through to this argument.
fn run_cargo_sub(
    sub_argv: &[&str],
    label: &str,
    kernel: Vec<String>,
    no_perf_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    let mut cmd = Command::new("cargo");
    cmd.args(sub_argv);
    if release {
        // Prepend `--cargo-profile release` BEFORE the user's
        // trailing args so the profile selection applies to the
        // whole invocation. nextest reads `--cargo-profile` directly;
        // `cargo llvm-cov nextest` forwards it to its inner nextest
        // invocation. For `cargo llvm-cov <sub>` (the raw-passthrough
        // binding), the release arg is never passed here — the raw
        // path relies on user-supplied `--release` / `--profile`.
        cmd.args(["--cargo-profile", "release"]);
    }
    cmd.args(&args);
    if no_perf_mode {
        cmd.env("KTSTR_NO_PERF_MODE", "1");
    }

    if !kernel.is_empty() {
        let resolved = resolve_kernel_set(&kernel)?;
        if resolved.is_empty() {
            // `resolve_kernel_set` skips arguments that trim to
            // empty, so `--kernel ""` or `--kernel "  "` reach
            // here without ever entering the per-spec resolve
            // branch. Bail with an actionable error rather than
            // letting the child reach for `find_kernel` as if
            // `--kernel` had never been passed (which would mask
            // the operator's intent).
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        // `KTSTR_KERNEL` always points at the first resolved entry
        // so downstream code that inspects the env directly (e.g.
        // budget listing's vmlinux probe in `dispatch.rs`) sees a
        // valid kernel even when running under multi-kernel.
        let first_dir = &resolved[0].1;
        eprintln!("cargo ktstr: using kernel {}", first_dir.display());
        cmd.env(ktstr::KTSTR_KERNEL_ENV, first_dir);

        if resolved.len() > 1 {
            let encoded = encode_kernel_list(&resolved)?;
            eprintln!(
                "cargo ktstr: fanning gauntlet across {n} kernels",
                n = resolved.len(),
            );
            cmd.env(ktstr::KTSTR_KERNEL_LIST_ENV, encoded);
        }
    }

    eprintln!("cargo ktstr: running {label}");
    let err = cmd.exec();
    Err(format!("exec cargo {}: {err}", sub_argv.join(" ")))
}

/// Cargo sub-argv that `run_test` passes to `run_cargo_sub`. Named
/// constant so the dispatch wiring is pinnable from a test — see
/// `cargo_sub_argv_constants_are_pinned`.
const TEST_SUB_ARGV: &[&str] = &["nextest", "run"];
/// Cargo sub-argv for the `coverage` subcommand (cargo llvm-cov
/// nextest).
const COVERAGE_SUB_ARGV: &[&str] = &["llvm-cov", "nextest"];
/// Cargo sub-argv for the `llvm-cov` raw-passthrough subcommand.
/// Single element — the user's trailing args supply the llvm-cov
/// subcommand (`report`, `clean`, `show-env`, ...).
const LLVM_COV_SUB_ARGV: &[&str] = &["llvm-cov"];

fn run_test(
    kernel: Vec<String>,
    no_perf_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    run_cargo_sub(TEST_SUB_ARGV, "tests", kernel, no_perf_mode, release, args)
}

fn run_coverage(
    kernel: Vec<String>,
    no_perf_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    run_cargo_sub(
        COVERAGE_SUB_ARGV,
        "coverage",
        kernel,
        no_perf_mode,
        release,
        args,
    )
}

fn run_llvm_cov(kernel: Vec<String>, no_perf_mode: bool, args: Vec<String>) -> Result<(), String> {
    // `llvm-cov` is raw passthrough — the user supplies every
    // argument after the subcommand name, including any profile
    // selection. `release: false` here means "don't inject a profile
    // ourselves"; the user decides.
    run_cargo_sub(
        LLVM_COV_SUB_ARGV,
        "llvm-cov",
        kernel,
        no_perf_mode,
        false,
        args,
    )
}

fn run_stats(command: &Option<StatsCommand>) -> Result<(), String> {
    match command {
        None => {
            if let Some(output) = cli::print_stats_report() {
                print!("{output}");
            }
            Ok(())
        }
        Some(StatsCommand::List) => cli::list_runs().map_err(|e| format!("{e:#}")),
        Some(StatsCommand::ListMetrics { json }) => match cli::list_metrics(*json) {
            Ok(s) => {
                print!("{s}");
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        Some(StatsCommand::ListValues { json, dir }) => {
            match cli::list_values(*json, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::ShowHost { run, dir }) => {
            match cli::show_run_host(run, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::ExplainSidecar { run, dir, json }) => {
            match cli::explain_sidecar(run, dir.as_deref(), *json) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::Compare {
            filter,
            threshold,
            policy,
            dir,
            kernel,
            project_commit,
            kernel_commit,
            run_source,
            scheduler,
            topology,
            work_type,
            flags,
            a_kernel,
            a_project_commit,
            a_kernel_commit,
            a_run_source,
            a_scheduler,
            a_topology,
            a_work_type,
            a_flags,
            b_kernel,
            b_project_commit,
            b_kernel_commit,
            b_run_source,
            b_scheduler,
            b_topology,
            b_work_type,
            b_flags,
            no_average,
        }) => {
            // Resolve `--threshold N` / `--policy PATH` / neither
            // into a single `ComparisonPolicy`. Clap's
            // `conflicts_with` guarantees at most one of
            // (threshold, policy) is set, so the three branches
            // are exhaustive on user-visible input.
            let resolved_policy = match (threshold, policy.as_ref()) {
                (Some(t), None) => {
                    let p = ktstr::cli::ComparisonPolicy::uniform(*t);
                    // `uniform` is infallible, but the user-supplied
                    // percent still needs a sign check. `validate`
                    // rejects negatives before they reach
                    // `compare_rows`' dual-gate math.
                    p.validate().map_err(|e| format!("{e:#}"))?;
                    p
                }
                (None, Some(path)) => {
                    ktstr::cli::ComparisonPolicy::load_json(path).map_err(|e| format!("{e:#}"))?
                }
                (None, None) => ktstr::cli::ComparisonPolicy::default(),
                (Some(_), Some(_)) => {
                    // Defence-in-depth: clap's `conflicts_with` is
                    // load-bearing here, but a regression that
                    // dropped either attribute would silently pick
                    // one path and ignore the other. Panic loudly.
                    unreachable!(
                        "clap `conflicts_with` on --threshold / --policy \
                         must enforce mutual exclusion at parse time",
                    );
                }
            };
            // Construct the BuildCompareFilters from the raw CLI
            // inputs. Sugar logic (shared `--X` pins both sides;
            // per-side `--a-X` / `--b-X` REPLACES the shared value
            // for that side) lives inside `build()` so it's
            // unit-testable in isolation. The dispatch site stays
            // a dumb data carrier.
            let build = BuildCompareFilters {
                shared_kernel: kernel.clone(),
                shared_project_commit: project_commit.clone(),
                shared_kernel_commit: kernel_commit.clone(),
                shared_run_source: run_source.clone(),
                shared_scheduler: scheduler.clone(),
                shared_topology: topology.clone(),
                shared_work_type: work_type.clone(),
                shared_flags: flags.clone(),
                a_kernel: a_kernel.clone(),
                a_project_commit: a_project_commit.clone(),
                a_kernel_commit: a_kernel_commit.clone(),
                a_run_source: a_run_source.clone(),
                a_scheduler: a_scheduler.clone(),
                a_topology: a_topology.clone(),
                a_work_type: a_work_type.clone(),
                a_flags: a_flags.clone(),
                b_kernel: b_kernel.clone(),
                b_project_commit: b_project_commit.clone(),
                b_kernel_commit: b_kernel_commit.clone(),
                b_run_source: b_run_source.clone(),
                b_scheduler: b_scheduler.clone(),
                b_topology: b_topology.clone(),
                b_work_type: b_work_type.clone(),
                b_flags: b_flags.clone(),
            };
            let (filter_a, filter_b) = build.build();
            let exit = cli::compare_partitions(
                &filter_a,
                &filter_b,
                filter.as_deref(),
                &resolved_policy,
                dir.as_deref(),
                *no_average,
            )
            .map_err(|e| format!("{e:#}"))?;
            if exit != 0 {
                std::process::exit(exit);
            }
            Ok(())
        }
    }
}

/// Symmetric-sugar resolver for `cargo ktstr stats compare`'s
/// shared `--X` and per-side `--a-X` / `--b-X` filter flags.
///
/// CLI flag semantics:
/// - Shared `--X` pins BOTH sides to the same value(s). E.g.
///   `--kernel 6.14` is equivalent to
///   `--a-kernel 6.14 --b-kernel 6.14`.
/// - Per-side `--a-X` REPLACES the shared `--X` value for the A
///   side only (and `--b-X` replaces for B only). "More-specific
///   replaces" — the per-side flag takes precedence over the
///   shared default for that side, but does not affect the
///   other side.
///
/// Constructed from the raw clap-parsed values; `build()` does
/// the sugar resolution and returns `(filter_a, filter_b)`. The
/// struct is unit-testable in isolation so the sugar logic does
/// not require booting a real comparison.
#[derive(Debug, Clone, Default)]
struct BuildCompareFilters {
    shared_kernel: Vec<String>,
    shared_project_commit: Vec<String>,
    shared_kernel_commit: Vec<String>,
    shared_run_source: Vec<String>,
    shared_scheduler: Vec<String>,
    shared_topology: Vec<String>,
    shared_work_type: Vec<String>,
    shared_flags: Vec<String>,
    a_kernel: Vec<String>,
    a_project_commit: Vec<String>,
    a_kernel_commit: Vec<String>,
    a_run_source: Vec<String>,
    a_scheduler: Vec<String>,
    a_topology: Vec<String>,
    a_work_type: Vec<String>,
    a_flags: Vec<String>,
    b_kernel: Vec<String>,
    b_project_commit: Vec<String>,
    b_kernel_commit: Vec<String>,
    b_run_source: Vec<String>,
    b_scheduler: Vec<String>,
    b_topology: Vec<String>,
    b_work_type: Vec<String>,
    b_flags: Vec<String>,
}

impl BuildCompareFilters {
    /// Resolve sugar into per-side `RowFilter` instances.
    /// "More-specific replaces": a per-side Vec is applied
    /// verbatim when non-empty, otherwise the shared Vec is
    /// used. Every dimension on `RowFilter` is now a `Vec<String>`
    /// (after the #13 conversion from `Option<String>` to repeatable
    /// Vec for scheduler/topology/work_type), so a single `pick_vec`
    /// helper handles every dim uniformly — the prior `pick_opt`
    /// branch is no longer reachable.
    fn build(&self) -> (ktstr::cli::RowFilter, ktstr::cli::RowFilter) {
        let pick_vec = |a: &[String], shared: &[String]| -> Vec<String> {
            if a.is_empty() {
                shared.to_vec()
            } else {
                a.to_vec()
            }
        };
        let filter_a = ktstr::cli::RowFilter {
            kernels: pick_vec(&self.a_kernel, &self.shared_kernel),
            project_commits: pick_vec(&self.a_project_commit, &self.shared_project_commit),
            kernel_commits: pick_vec(&self.a_kernel_commit, &self.shared_kernel_commit),
            run_sources: pick_vec(&self.a_run_source, &self.shared_run_source),
            schedulers: pick_vec(&self.a_scheduler, &self.shared_scheduler),
            topologies: pick_vec(&self.a_topology, &self.shared_topology),
            work_types: pick_vec(&self.a_work_type, &self.shared_work_type),
            flags: pick_vec(&self.a_flags, &self.shared_flags),
        };
        let filter_b = ktstr::cli::RowFilter {
            kernels: pick_vec(&self.b_kernel, &self.shared_kernel),
            project_commits: pick_vec(&self.b_project_commit, &self.shared_project_commit),
            kernel_commits: pick_vec(&self.b_kernel_commit, &self.shared_kernel_commit),
            run_sources: pick_vec(&self.b_run_source, &self.shared_run_source),
            schedulers: pick_vec(&self.b_scheduler, &self.shared_scheduler),
            topologies: pick_vec(&self.b_topology, &self.shared_topology),
            work_types: pick_vec(&self.b_work_type, &self.shared_work_type),
            flags: pick_vec(&self.b_flags, &self.shared_flags),
        };
        (filter_a, filter_b)
    }
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
fn kernel_build(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
) -> Result<(), String> {
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
                if let Err(e) =
                    kernel_build_one(Some(ver.clone()), None, None, None, force, clean, cpu_cap)
                {
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
            kernel_build_one(version, source, git, git_ref, force, clean, cpu_cap)
        }
    } else {
        kernel_build_one(version, source, git, git_ref, force, clean, cpu_cap)
    }
}

/// Single-version variant of [`kernel_build`]: handles one tarball,
/// `--source`, or `--git` invocation. Carries the `kernel_build`
/// implementation as it stood before range dispatch was wired in;
/// extracted into a helper so the range loop in `kernel_build` can
/// reuse the same download + cache + build pipeline per resolved
/// version without duplicating it.
fn kernel_build_one(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
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
    let acquired = if let Some(ref src_path) = source {
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
        // Check cache before downloading.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!("{ver}-tarball-{arch}-kc{}", ktstr::cache_key_suffix());
        if !force && let Some(entry) = cache_lookup(&cache, &cache_key) {
            eprintln!("cargo ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("cargo ktstr: use --force to rebuild");
            return Ok(());
        }
        let sp = cli::Spinner::start("Downloading kernel...");
        let result = fetch::download_tarball(client, &ver, tmp_dir.path(), "cargo ktstr");
        drop(sp);
        result.map_err(|e| format!("{e:#}"))?
    };

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
    )
    .map_err(|e| format!("{e:#}"))?;

    Ok(())
}

/// Look up a cache key, checking local first, then remote (if enabled).
fn cache_lookup(cache: &CacheDir, cache_key: &str) -> Option<CacheEntry> {
    cli::cache_lookup(cache, cache_key, "cargo ktstr")
}

/// Policy for cargo-ktstr's shell + verifier kernel resolution:
/// accept raw image files, use "cargo ktstr" as the CLI label.
const KERNEL_POLICY: ktstr::cli::KernelResolvePolicy<'static> = ktstr::cli::KernelResolvePolicy {
    accept_raw_image: true,
    cli_label: "cargo ktstr",
};

/// Resolve a kernel identifier to a bootable image path via the
/// shared `ktstr::cli::resolve_kernel_image` helper with cargo-ktstr's
/// policy.
fn resolve_kernel_image(kernel: Option<&str>) -> Result<PathBuf, String> {
    ktstr::cli::resolve_kernel_image(kernel, &KERNEL_POLICY).map_err(|e| format!("{e:#}"))
}

#[allow(clippy::too_many_arguments)]
fn run_shell(
    kernel: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
    memory_mb: Option<u32>,
    dmesg: bool,
    exec: Option<String>,
    no_perf_mode: bool,
    cpu_cap: Option<usize>,
) -> Result<(), String> {
    if no_perf_mode {
        // SAFETY: single-threaded at this point — no concurrent env readers.
        unsafe { std::env::set_var("KTSTR_NO_PERF_MODE", "1") };
    }
    if let Some(cap) = cpu_cap {
        // Parse-time conflict with KTSTR_BYPASS_LLC_LOCKS — see
        // ktstr.rs Shell dispatch for the same check.
        if std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_some_and(|v| !v.is_empty())
        {
            return Err(
                "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
                 --cpu-cap is a resource contract; bypass disables the contract entirely."
                    .to_string(),
            );
        }
        // Validate early so a bad cap surfaces at CLI-parse time.
        cli::CpuCap::new(cap).map_err(|e| format!("{e:#}"))?;
        // SAFETY: single-threaded at this point — no concurrent env readers.
        unsafe { std::env::set_var("KTSTR_CPU_CAP", cap.to_string()) };
    }
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    let (numa_nodes, llcs, cores, threads) =
        cli::parse_topology_string(&topology).map_err(|e| format!("{e:#}"))?;

    let resolved_includes =
        cli::resolve_include_files(&include_files).map_err(|e| format!("{e:#}"))?;

    let include_refs: Vec<(&str, &Path)> = resolved_includes
        .iter()
        .map(|(a, p)| (a.as_str(), p.as_path()))
        .collect();

    ktstr::run_shell(
        kernel_path,
        numa_nodes,
        llcs,
        cores,
        threads,
        &include_refs,
        memory_mb,
        dmesg,
        exec.as_deref(),
    )
    .map_err(|e| format!("{e:#}"))
}

/// Query a scheduler binary's flag declarations via `--ktstr-list-flags`.
///
/// Runs the binary with `--ktstr-list-flags` and parses its stdout as
/// JSON. Returns an empty vec if the binary doesn't support the flag
/// (exits non-zero or produces no output).
fn query_scheduler_flags(
    sched_bin: &Path,
) -> Result<Vec<ktstr::scenario::flags::FlagDeclJson>, String> {
    let output = Command::new(sched_bin)
        .arg("--ktstr-list-flags")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("run scheduler --ktstr-list-flags: {e:#}"))?;

    if !output.status.success() {
        return Ok(Vec::new());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    serde_json::from_str(trimmed).map_err(|e| format!("parse --ktstr-list-flags output: {e:#}"))
}

/// Generate flag profiles from flag declarations.
///
/// Produces the power set of flags, filtered by requires constraints,
/// via the shared [`ktstr::scenario::compute_flag_profiles`] generator.
/// Each profile's flags are sorted in declaration order. The profile
/// name is the flags joined with `+`, or `"default"` when empty.
fn generate_flag_profiles(
    flags: &[ktstr::scenario::flags::FlagDeclJson],
) -> Vec<(String, Vec<String>)> {
    let n = flags.len();
    if n > 31 {
        eprintln!(
            "cargo ktstr: error: scheduler has {n} flags, power set too large (2^{n}). \
             Use --profiles to select specific profiles."
        );
        return Vec::new();
    }

    let all: Vec<String> = flags.iter().map(|f| f.name.clone()).collect();
    let requires_fn = |name: &String| -> Vec<String> {
        flags
            .iter()
            .find(|f| f.name == *name)
            .map(|f| f.requires.clone())
            .unwrap_or_default()
    };

    ktstr::scenario::compute_flag_profiles(&all, requires_fn, &[], &[])
        .into_iter()
        .map(|flag_names| {
            let name = if flag_names.is_empty() {
                "default".to_string()
            } else {
                flag_names.join("+")
            };
            (name, flag_names)
        })
        .collect()
}

/// Collect the extra scheduler args for a set of active flags.
///
/// Returns `Err` if any flag in `active_flags` is not declared in
/// `all_flags`. Silently dropping unknown flags masked typos in
/// CLI `--profiles` lists and version-drift in cached nextest args —
/// the caller would see "flag applied" in the profile name but the
/// scheduler was actually invoked without the corresponding CLI arg.
fn profile_sched_args(
    active_flags: &[String],
    all_flags: &[ktstr::scenario::flags::FlagDeclJson],
) -> Result<Vec<String>, String> {
    let mut args = Vec::new();
    for flag_name in active_flags {
        match all_flags.iter().find(|f| f.name == *flag_name) {
            Some(decl) => args.extend(decl.args.iter().cloned()),
            None => {
                let known: Vec<&str> = all_flags.iter().map(|f| f.name.as_str()).collect();
                return Err(format!(
                    "unknown flag {flag_name:?} (known: {})",
                    known.join(", ")
                ));
            }
        }
    }
    Ok(args)
}

fn run_verifier(
    scheduler: Option<String>,
    scheduler_bin: Option<PathBuf>,
    kernel: Vec<String>,
    raw: bool,
    all_profiles: bool,
    profiles_filter: Vec<String>,
) -> Result<(), String> {
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;

    // Resolve scheduler binary.
    let sched_bin = match (scheduler, scheduler_bin) {
        (Some(package), None) => {
            ktstr::build_and_find_binary(&package).map_err(|e| format!("build scheduler: {e:#}"))?
        }
        (None, Some(path)) => {
            if !path.exists() {
                return Err(format!("scheduler binary not found: {}", path.display()));
            }
            path
        }
        (None, None) => {
            return Err("either --scheduler or --scheduler-bin is required".to_string());
        }
        // clap conflicts_with prevents this.
        (Some(_), Some(_)) => unreachable!(),
    };

    // Resolve --kernel into a flat (label, kernel_dir) list. Empty
    // input falls through to the single-kernel auto-discovery path
    // below (`resolve_kernel_image(None)` → `find_kernel`'s
    // fallback chain), preserving the no-flag behaviour. A
    // single-entry list is treated identically to the historical
    // single-kernel path: one verifier run, no kernel-prefixed
    // output. Two or more entries (multiple `--kernel` flags, OR
    // a single `--kernel` Range that expanded to multiple
    // releases) iterate sequentially with per-kernel header lines.
    let kernel_paths: Vec<(String, PathBuf)> = if kernel.is_empty() {
        // Auto-discovery: route through `resolve_kernel_image(None)`
        // so the existing `find_kernel` cascade applies, then label
        // the result `"auto"` for diagnostic visibility on the rare
        // path where the user neither passed `--kernel` nor exported
        // `KTSTR_KERNEL`.
        let path = resolve_kernel_image(None)?;
        vec![("auto".to_string(), path)]
    } else {
        // Multi-kernel resolution shares its plumbing with the test
        // path (`run_cargo_sub`'s `resolve_kernel_set` call),
        // including Range expansion and Git fetch. Each resolved
        // entry is a built / cached kernel directory; convert it to
        // a bootable image via `find_image_in_dir` since the
        // verifier collects stats from a loaded image rather than
        // a directory.
        let resolved = resolve_kernel_set(&kernel)?;
        if resolved.is_empty() {
            return Err(
                "--kernel: every supplied value parsed to empty / whitespace; \
                 omit the flag for auto-discovery, or supply a kernel \
                 identifier"
                    .to_string(),
            );
        }
        let mut out: Vec<(String, PathBuf)> = Vec::with_capacity(resolved.len());
        for (label, dir) in resolved {
            let image = ktstr::kernel_path::find_image_in_dir(&dir).ok_or_else(|| {
                format!(
                    "no kernel image found in {} (resolved from --kernel {label})",
                    dir.display()
                )
            })?;
            out.push((label, image));
        }
        out
    };

    // Build the ktstr init binary.
    let ktstr_bin =
        ktstr::build_and_find_binary("ktstr").map_err(|e| format!("build ktstr: {e:#}"))?;

    let multi_kernel = kernel_paths.len() > 1;
    for (i, (label, kernel_path)) in kernel_paths.iter().enumerate() {
        if multi_kernel {
            eprintln!(
                "cargo ktstr: [kernel {}/{}] {label}",
                i + 1,
                kernel_paths.len(),
            );
            // Header on stdout so a redirected `>` capture
            // separates kernels even when stderr isn't pulled.
            println!("\n=== kernel: {label} ===");
        }

        if all_profiles || !profiles_filter.is_empty() {
            run_verifier_all_profiles(&sched_bin, &ktstr_bin, kernel_path, raw, &profiles_filter)?;
            continue;
        }

        eprintln!("cargo ktstr: collecting verifier stats");
        let result =
            ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, kernel_path, &[])
                .map_err(|e| format!("collect verifier output: {e:#}"))?;

        let output = ktstr::verifier::format_verifier_output("verifier", &result, raw);
        print!("{output}");
    }

    Ok(())
}

fn run_verifier_all_profiles(
    sched_bin: &Path,
    ktstr_bin: &Path,
    kernel_path: &Path,
    raw: bool,
    profiles_filter: &[String],
) -> Result<(), String> {
    let flags = query_scheduler_flags(sched_bin)?;
    if flags.is_empty() {
        eprintln!(
            "cargo ktstr: scheduler does not support --ktstr-list-flags, \
             running with default profile only"
        );
        let result =
            ktstr::verifier::collect_verifier_output(sched_bin, ktstr_bin, kernel_path, &[])
                .map_err(|e| format!("collect verifier output: {e:#}"))?;
        let output = ktstr::verifier::format_verifier_output("default", &result, raw);
        print!("{output}");
        return Ok(());
    }

    let all_profiles = generate_flag_profiles(&flags);

    // Filter profiles if --profiles was specified.
    let profiles: Vec<&(String, Vec<String>)> = if profiles_filter.is_empty() {
        all_profiles.iter().collect()
    } else {
        let filtered: Vec<_> = all_profiles
            .iter()
            .filter(|(name, _)| profiles_filter.iter().any(|f| f == name))
            .collect();
        if filtered.is_empty() {
            return Err(format!(
                "no matching profiles found. Available: {}",
                all_profiles
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        filtered
    };

    let total = profiles.len();
    if total == 0 {
        // Differentiate the empty-profile cases so the user gets an
        // actionable error rather than a generic "0 profiles" message.
        return Err(if flags.len() > 31 {
            format!(
                "no profiles to verify: power-set generation is capped at \
                 31 flags (found {}); use --profiles to select a subset",
                flags.len(),
            )
        } else {
            format!(
                "no profiles to verify: {} flag(s) advertised but profile \
                 generation produced 0 profiles — check `requires` \
                 dependencies and exclusions for cycles or conflicts",
                flags.len(),
            )
        });
    }
    if total > 32 {
        eprintln!(
            "cargo ktstr: warning: {total} profiles to verify (>32). \
             Use --profiles to select a subset."
        );
    }

    eprintln!(
        "cargo ktstr: verifying {total} profile{}",
        if total == 1 { "" } else { "s" }
    );

    // Per-profile summary table: (profile_name, Vec<(prog_name, verified_insns)>).
    let mut summary: Vec<(String, Vec<(String, u32)>)> = Vec::new();

    for (i, (profile_name, active_flags)) in profiles.iter().enumerate() {
        eprintln!(
            "cargo ktstr: [{}/{}] profile: {}",
            i + 1,
            total,
            profile_name
        );

        let extra_args = profile_sched_args(active_flags, &flags)
            .map_err(|e| format!("profile {profile_name}: {e}"))?;
        let result = ktstr::verifier::collect_verifier_output(
            sched_bin,
            ktstr_bin,
            kernel_path,
            &extra_args,
        )
        .map_err(|e| format!("profile {profile_name}: {e:#}"))?;

        let output = ktstr::verifier::format_verifier_output(profile_name, &result, raw);
        print!("{output}");

        let prog_stats: Vec<(String, u32)> = result
            .stats
            .iter()
            .map(|ps| (ps.name.clone(), ps.verified_insns))
            .collect();
        summary.push((profile_name.clone(), prog_stats));
    }

    // Print per-profile summary table.
    if summary.len() > 1 {
        print_profile_summary(&summary);
    }

    Ok(())
}

/// Print a summary table comparing verified_insns across profiles.
fn print_profile_summary(summary: &[(String, Vec<(String, u32)>)]) {
    // Collect all unique program names in insertion order.
    let mut prog_names: Vec<String> = Vec::new();
    for (_, progs) in summary {
        for (name, _) in progs {
            if !prog_names.contains(name) {
                prog_names.push(name.clone());
            }
        }
    }

    println!("\n--- profile summary ---");

    let profile_names: Vec<&str> = summary.iter().map(|(n, _)| n.as_str()).collect();
    let mut table = ktstr::cli::new_table();
    let mut header: Vec<&str> = Vec::with_capacity(1 + profile_names.len());
    header.push("program");
    header.extend(profile_names.iter().copied());
    table.set_header(header);

    for prog in &prog_names {
        let mut row: Vec<String> = Vec::with_capacity(1 + profile_names.len());
        row.push(prog.clone());
        for (_, progs) in summary {
            let insns = progs
                .iter()
                .find(|(n, _)| n == prog)
                .map(|(_, v)| *v)
                .unwrap_or(0);
            row.push(insns.to_string());
        }
        table.add_row(row);
    }

    println!("{table}");
}

fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cargo::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}

/// `cargo ktstr model fetch` — download + SHA-check the default model
/// into the user's cache. Wraps `ktstr::test_support::ensure` with a
/// human-readable progress line; the status is printed after so
/// users can see the final cache path regardless of whether the
/// fetch did any work.
fn run_model_fetch() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    match ktstr::test_support::ensure(&spec) {
        Ok(path) => {
            println!(
                "ktstr: model '{}' ready at {}",
                spec.file_name,
                path.display()
            );
            Ok(())
        }
        Err(e) => Err(format!("fetch model '{}': {e:#}", spec.file_name)),
    }
}

/// `cargo ktstr model status` — report the cache path and whether a
/// SHA-checked copy of the default model is already present.
fn run_model_status() -> Result<(), String> {
    let spec = ktstr::test_support::DEFAULT_MODEL;
    let status = ktstr::test_support::status(&spec).map_err(|e| format!("{e:#}"))?;
    println!("model:    {}", status.spec.file_name);
    println!("path:     {}", status.path.display());
    println!("cached:   {}", status.sha_verdict.is_cached());
    println!("checked:  {}", status.sha_verdict.is_match());
    // Distinguish the four verdict variants so each gets a
    // remediation-specific line: absent cache, I/O failure during
    // the SHA check, successful hash that didn't match, and the
    // all-clear case (no annotation needed). An I/O failure points
    // at the filesystem entry (permissions, truncation); a mismatch
    // points at the bytes themselves.
    // "Re-fetch to replace it" is the shared remediation tail for
    // every non-Matches cached-file branch (both CheckFailed and
    // Mismatches land on the same operator action — the cause
    // differs but the fix does not). Factoring the tail into one
    // string keeps the two arms in lock-step so a wording change
    // lands in both places by construction.
    const RE_FETCH_TAIL: &str = "re-fetch to replace it";
    match &status.sha_verdict {
        ktstr::test_support::ShaVerdict::NotCached => println!(
            "(no cached copy — run `cargo ktstr model fetch` to download {} MiB)",
            status.spec.size_bytes / 1024 / 1024,
        ),
        ktstr::test_support::ShaVerdict::CheckFailed(err) => {
            // Defensively collapse any embedded `\n` into `; `
            // before placing `err` inside the "(single
            // parenthesized note)" wrapper. The alternate
            // anyhow format (`{e:#}`) that produced `err` joins
            // causes with `: ` and is single-line in practice;
            // this replace is a guard against a future error
            // source whose Display impl injects its own
            // newlines (std::io errors wrapping multi-line OS
            // messages, third-party crates formatting
            // call-chain trees). Keeping the output on one
            // line preserves the visual grouping the other
            // match arms use.
            let single_line = err.replace('\n', "; ");
            println!(
                "(cached file could not be checked: {single_line}; \
                 inspect the cache entry or {RE_FETCH_TAIL})",
            );
        }
        ktstr::test_support::ShaVerdict::Mismatches => {
            println!("(cached file failed SHA-256 check; {RE_FETCH_TAIL})",);
        }
        ktstr::test_support::ShaVerdict::Matches => {}
    }
    Ok(())
}

fn main() {
    // Restore SIGPIPE so piping `cargo ktstr ... | head` doesn't
    // panic inside `print!`. See `ktstr::cli::restore_sigpipe_default`
    // for the full rationale; shared across all three ktstr bins so
    // the rationale + SAFETY text lives in one place.
    ktstr::cli::restore_sigpipe_default();
    // Mirror `ktstr`'s tracing init (src/bin/ktstr.rs main()) so
    // `tracing::warn!` calls inside `cli::` / `test_support::` surface
    // on stderr instead of being silently dropped. Default to `warn`
    // so normal CLI invocations (kernel build, model fetch, etc.) stay
    // quiet; users who want finer detail set `RUST_LOG=info,debug,...`.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let Cargo {
        command: CargoSub::Ktstr(ktstr),
    } = Cargo::parse();

    // Match-arm order mirrors the `KtstrCommand` enum declaration at
    // the top of this file. Keeping the two orderings in lockstep lets
    // a reviewer eyeball "every variant is dispatched" in one linear
    // scan instead of cross-referencing two different orders; a future
    // variant addition then lands in the matching enum position and
    // here without requiring the reader to rebuild the mapping.
    let result = match ktstr.command {
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            release,
            args,
        } => run_test(kernel, no_perf_mode, release, args),
        KtstrCommand::Coverage {
            kernel,
            no_perf_mode,
            release,
            args,
        } => run_coverage(kernel, no_perf_mode, release, args),
        KtstrCommand::LlvmCov {
            kernel,
            no_perf_mode,
            args,
        } => run_llvm_cov(kernel, no_perf_mode, args),
        KtstrCommand::Stats { ref command } => run_stats(command),
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json, range } => match range {
                Some(r) => cli::kernel_list_range_preview(json, &r).map_err(|e| format!("{e:#}")),
                None => cli::kernel_list(json).map_err(|e| format!("{e:#}")),
            },
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
            } => kernel_build(version, source, git, git_ref, force, clean, cpu_cap),
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => cli::kernel_clean(keep, force, corrupt_only).map_err(|e| format!("{e:#}")),
        },
        KtstrCommand::Model { command } => match command {
            ModelCommand::Fetch => run_model_fetch(),
            ModelCommand::Status => run_model_status(),
        },
        KtstrCommand::Verifier {
            scheduler,
            scheduler_bin,
            kernel,
            raw,
            all_profiles,
            profiles,
        } => run_verifier(
            scheduler,
            scheduler_bin,
            kernel,
            raw,
            all_profiles,
            profiles,
        ),
        KtstrCommand::Completions { shell, binary } => {
            run_completions(shell, &binary);
            Ok(())
        }
        KtstrCommand::ShowHost => {
            print!("{}", cli::show_host());
            Ok(())
        }
        KtstrCommand::ShowThresholds { test } => match cli::show_thresholds(&test) {
            Ok(s) => {
                print!("{s}");
                Ok(())
            }
            Err(e) => Err(format!("{e:#}")),
        },
        KtstrCommand::Cleanup { parent_cgroup } => {
            cli::cleanup(parent_cgroup).map_err(|e| format!("{e:#}"))
        }
        KtstrCommand::Locks { json, watch } => {
            cli::list_locks(json, watch).map_err(|e| format!("{e:#}"))
        }
        KtstrCommand::Shell {
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
            no_perf_mode,
            cpu_cap,
        } => run_shell(
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
            no_perf_mode,
            cpu_cap,
        ),
    };

    if let Err(e) = result {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use ktstr::cache::KernelMetadata;

    // -- structural validation --

    #[test]
    fn cli_debug_assert() {
        Cargo::command().debug_assert();
    }

    // -- try_get_matches_from: test subcommand --

    #[test]
    fn parse_test_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "test"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_test_with_kernel() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "test", "--kernel", "6.14.2"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    /// `--release` on `test` parses to `KtstrCommand::Test { release:
    /// true, .. }` so `run_test` prepends `--cargo-profile release`
    /// to the cargo nextest invocation. A clap regression that
    /// dropped the flag would turn the user-visible `--release` into
    /// either a silent no-op (default false) or a passthrough-arg
    /// typo — this test pins the clap-level wiring.
    #[test]
    fn parse_test_with_release_flag() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "test", "--release"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Test { release, .. } => {
                assert!(release, "`--release` must set `release=true`");
            }
            _ => panic!("expected Test"),
        }
    }

    #[test]
    fn parse_test_with_passthrough_args() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "test",
            "--",
            "-p",
            "ktstr",
            "--no-capture",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Test { args, .. } => {
                assert_eq!(args, vec!["-p", "ktstr", "--no-capture"]);
            }
            _ => panic!("expected Test"),
        }
    }

    // -- try_get_matches_from: `test` visible alias `nextest` --

    /// `cargo ktstr nextest` resolves to the canonical `Test`
    /// variant. `visible_alias = "nextest"` on the variant makes
    /// the alias user-facing (shows in --help) and dispatch-
    /// transparent (the existing `KtstrCommand::Test` arm handles
    /// both spellings). A regression that dropped the attribute
    /// would fail this test at runtime.
    #[test]
    fn parse_nextest_alias_dispatches_to_test() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "nextest"]).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            matches!(k.command, KtstrCommand::Test { .. }),
            "`nextest` alias must dispatch to the Test variant",
        );
    }

    /// `nextest` alias carries trailing args through the same
    /// `trailing_var_arg` pipeline as `test`. Pins the alias's
    /// passthrough behaviour byte-exactly so a clap regression
    /// that treated the alias as a distinct parse tree surfaces
    /// here rather than in runtime dispatch.
    #[test]
    fn parse_nextest_alias_with_passthrough_args() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "nextest",
            "--",
            "-p",
            "ktstr",
            "--no-capture",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Test { args, .. } => {
                assert_eq!(args, vec!["-p", "ktstr", "--no-capture"]);
            }
            _ => panic!("expected Test (via `nextest` alias)"),
        }
    }

    /// Verify the `nextest` alias preserves all Test fields in a
    /// single invocation: `--kernel`, `--no-perf-mode`, and empty
    /// trailing `args`. A clap regression that silently dropped a
    /// field on the alias path (e.g. a derive bug that re-generated
    /// the subcommand without inheriting the Test variant's args)
    /// would surface here.
    #[test]
    fn parse_nextest_alias_with_kernel_and_no_perf_mode() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "nextest",
            "--kernel",
            "6.14.2",
            "--no-perf-mode",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Test {
                kernel,
                no_perf_mode,
                release,
                args,
            } => {
                assert_eq!(kernel, vec!["6.14.2".to_string()]);
                assert!(no_perf_mode);
                assert!(!release, "bare invocation must default --release to false");
                assert!(args.is_empty());
            }
            _ => panic!("expected Test (via `nextest` alias)"),
        }
    }

    // -- try_get_matches_from: coverage subcommand --

    #[test]
    fn parse_coverage_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_coverage_with_kernel() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--kernel", "6.14.2"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    /// `--release` on `coverage` parses to `KtstrCommand::Coverage
    /// { release: true, .. }` so `run_coverage` prepends
    /// `--cargo-profile release` to the cargo llvm-cov nextest
    /// invocation. Same rationale as the sibling
    /// `parse_test_with_release_flag` — pins the clap-level wiring
    /// against a regression that turns the flag into a no-op.
    #[test]
    fn parse_coverage_with_release_flag() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "coverage", "--release"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Coverage { release, .. } => {
                assert!(release, "`--release` must set `release=true`");
            }
            _ => panic!("expected Coverage"),
        }
    }

    #[test]
    fn parse_coverage_with_passthrough_args() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "coverage",
            "--",
            "--workspace",
            "--lcov",
            "--output-path",
            "lcov.info",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Coverage { args, .. } => {
                assert_eq!(
                    args,
                    vec!["--workspace", "--lcov", "--output-path", "lcov.info"]
                );
            }
            _ => panic!("expected Coverage"),
        }
    }

    /// Combined round-trip for Coverage: `--kernel`, `--no-perf-mode`,
    /// AND trailing args all populate on a single invocation. Mirrors
    /// `parse_llvm_cov_with_kernel_and_no_perf_mode` — a clap
    /// regression that dropped one field on the multi-flag path (or
    /// mis-ordered `--` with flags) would surface here for the
    /// Coverage variant.
    #[test]
    fn parse_coverage_with_kernel_and_no_perf_mode() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "coverage",
            "--kernel",
            "6.14.2",
            "--no-perf-mode",
            "--",
            "--workspace",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Coverage {
                kernel,
                no_perf_mode,
                release,
                args,
            } => {
                assert_eq!(kernel, vec!["6.14.2".to_string()]);
                assert!(no_perf_mode);
                assert!(!release, "bare invocation must default --release to false");
                assert_eq!(args, vec!["--workspace"]);
            }
            _ => panic!("expected Coverage"),
        }
    }

    // -- try_get_matches_from: llvm-cov raw passthrough subcommand --

    #[test]
    fn parse_llvm_cov_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_llvm_cov_with_kernel() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "llvm-cov", "--kernel", "6.14.2"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::LlvmCov { kernel, .. } => {
                assert_eq!(kernel, vec!["6.14.2".to_string()]);
            }
            _ => panic!("expected LlvmCov"),
        }
    }

    #[test]
    fn parse_llvm_cov_with_passthrough_args() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "llvm-cov",
            "--",
            "report",
            "--lcov",
            "--output-path",
            "lcov.info",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::LlvmCov { args, .. } => {
                assert_eq!(args, vec!["report", "--lcov", "--output-path", "lcov.info"]);
            }
            _ => panic!("expected LlvmCov"),
        }
    }

    /// Combined round-trip: `--kernel`, `--no-perf-mode`, AND
    /// trailing args all populate on a single LlvmCov invocation.
    /// A clap regression that dropped one field on the multi-flag
    /// path (or mis-ordered `--` with flags) would surface here.
    #[test]
    fn parse_llvm_cov_with_kernel_and_no_perf_mode() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "llvm-cov",
            "--kernel",
            "6.14.2",
            "--no-perf-mode",
            "--",
            "report",
            "--lcov",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::LlvmCov {
                kernel,
                no_perf_mode,
                args,
            } => {
                assert_eq!(kernel, vec!["6.14.2".to_string()]);
                assert!(no_perf_mode);
                assert_eq!(args, vec!["report", "--lcov"]);
            }
            _ => panic!("expected LlvmCov"),
        }
    }

    /// Negative pin: the variant is `LlvmCov`, and clap derive's
    /// default casing is kebab-case (see clap_derive
    /// `DEFAULT_CASING`), so the subcommand name is `llvm-cov`,
    /// NOT `llvm_cov`. A regression that switched the derive's
    /// rename_all default (or silently aliased the underscore
    /// form) would turn this negative pin positive. The parent-
    /// level `aliases` slot is empty, so clap rejects the
    /// underscore form with an unknown-subcommand error.
    #[test]
    fn parse_llvm_cov_underscore_rejected() {
        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "llvm_cov"]);
        assert!(
            rejected.is_err(),
            "`llvm_cov` (underscore) must be rejected — the \
             canonical name is `llvm-cov` (kebab-case)",
        );
    }

    // -- try_get_matches_from: shell subcommand --

    #[test]
    fn parse_shell_minimal() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "shell"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_shell_with_topology() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--topology", "1,2,4,1"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { topology, .. } => {
                assert_eq!(topology, "1,2,4,1");
            }
            _ => panic!("expected Shell"),
        }
    }

    #[test]
    fn parse_shell_default_topology() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell"]).unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { topology, .. } => {
                assert_eq!(topology, "1,1,1,1");
            }
            _ => panic!("expected Shell"),
        }
    }

    #[test]
    fn parse_shell_include_files() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "-i", "/tmp/a", "-i", "/tmp/b"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell { include_files, .. } => {
                assert_eq!(include_files.len(), 2);
            }
            _ => panic!("expected Shell"),
        }
    }

    // -- try_get_matches_from: stats subcommand --

    #[test]
    fn parse_stats_bare() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "stats"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_stats_list() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    /// `cargo ktstr stats list-metrics` parses (no flags required)
    /// and dispatches to the `ListMetrics` variant with `json=false`.
    #[test]
    fn parse_stats_list_metrics_bare() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ListMetrics { json }),
                ..
            } => {
                assert!(
                    !json,
                    "bare `list-metrics` must default to text mode (json=false)",
                );
            }
            _ => panic!("expected Stats ListMetrics"),
        }
    }

    /// `cargo ktstr stats list-metrics --json` sets `json=true`.
    /// Pins the flag name so a clap-derive-default rename
    /// (kebab-case) cannot drift — `--json` is the same flag name
    /// other list-style subcommands use (e.g. `kernel list --json`).
    #[test]
    fn parse_stats_list_metrics_json() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics", "--json"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ListMetrics { json }),
                ..
            } => {
                assert!(json, "--json must set the flag true");
            }
            _ => panic!("expected Stats ListMetrics"),
        }
    }

    /// `list-metrics` takes no positional args — a stray positional
    /// must be rejected by clap so a typo like `list-metrics
    /// worst_spread` doesn't silently look like success.
    #[test]
    fn parse_stats_list_metrics_rejects_positional() {
        let rejected =
            Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-metrics", "worst_spread"]);
        assert!(
            rejected.is_err(),
            "list-metrics must reject positional arguments",
        );
    }

    /// `cargo ktstr stats list-values` parses with no flags and
    /// dispatches to the `ListValues` variant with `json=false` and
    /// `dir=None`. Pins the bare-call defaults.
    #[test]
    fn parse_stats_list_values_bare() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ListValues { json, dir }),
                ..
            } => {
                assert!(!json, "bare `list-values` must default to text mode");
                assert!(
                    dir.is_none(),
                    "bare `list-values` must default to no --dir override"
                );
            }
            _ => panic!("expected Stats ListValues"),
        }
    }

    /// `cargo ktstr stats list-values --json` sets `json=true`.
    /// Pins the flag name so the same `--json` convention used by
    /// `list-metrics` and `kernel list` carries here too.
    #[test]
    fn parse_stats_list_values_json() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values", "--json"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ListValues { json, .. }),
                ..
            } => {
                assert!(json, "--json must set the flag true");
            }
            _ => panic!("expected Stats ListValues"),
        }
    }

    /// `cargo ktstr stats list-values --dir PATH` round-trips the
    /// path through clap to the dispatch site. Same `--dir`
    /// convention as `compare --dir` and `show-host --dir`.
    #[test]
    fn parse_stats_list_values_with_dir() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "list-values",
            "--dir",
            "/tmp/archived-runs",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ListValues { dir, json }),
                ..
            } => {
                assert_eq!(
                    dir.as_deref(),
                    Some(std::path::Path::new("/tmp/archived-runs")),
                    "--dir must round-trip to Some(PathBuf)",
                );
                assert!(!json, "bare --dir must not spuriously set --json");
            }
            _ => panic!("expected Stats ListValues"),
        }
    }

    /// `list-values` takes no positional args — clap must reject
    /// strays so a typo like `list-values kernel` (intending a
    /// per-dim filter) fails loudly rather than getting silently
    /// dropped.
    #[test]
    fn parse_stats_list_values_rejects_positional() {
        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "list-values", "kernel"]);
        assert!(
            rejected.is_err(),
            "list-values must reject positional arguments",
        );
    }

    #[test]
    fn parse_stats_compare() {
        // Minimal partition shape: --a-kernel + --b-kernel define
        // the slicing dimension. The dispatch site rejects empty
        // slicing dims, so a bare `cargo ktstr stats compare`
        // would fail at run time — but the CLI parser accepts
        // it (validation belongs in `compare_partitions`, not
        // clap). This test pins the parse layer only.
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_stats_compare_with_filter() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "-E",
            "cgroup_steady",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        filter,
                        threshold,
                        policy,
                        dir,
                        a_kernel,
                        b_kernel,
                        ..
                    }),
                ..
            } => {
                assert_eq!(a_kernel, vec!["6.14"]);
                assert_eq!(b_kernel, vec!["6.15"]);
                assert_eq!(filter.as_deref(), Some("cgroup_steady"));
                assert!(threshold.is_none());
                assert!(policy.is_none());
                assert!(dir.is_none());
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    #[test]
    fn parse_stats_compare_with_threshold() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "--threshold",
            "5.0",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        threshold, filter, ..
                    }),
                ..
            } => {
                assert_eq!(threshold, Some(5.0));
                assert!(filter.is_none());
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// Proves the `dir: Option<PathBuf>` field is wired on
    /// `StatsCommand::Compare` and round-trips through clap's arg
    /// parser. A regression that removed the struct field would
    /// fail this test at compile time; a regression that dropped
    /// the dispatch wiring (cargo-ktstr.rs → cli.rs → stats.rs) is
    /// outside parse-scope and covered by the resolver's own
    /// tests. The sibling `*_with_filter` test pins the
    /// `dir.is_none()` default; this one pins the `Some(PathBuf)`
    /// branch byte-exactly. Uses an absolute `/tmp/...` path
    /// (synthetic, not required to exist) because the parse path
    /// does not touch the filesystem — clap produces the `PathBuf`
    /// from the raw argument, full stop.
    #[test]
    fn parse_stats_compare_with_dir() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "--dir",
            "/tmp/archived-runs",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        filter,
                        threshold,
                        policy,
                        dir,
                        ..
                    }),
                ..
            } => {
                assert_eq!(
                    dir.as_deref(),
                    Some(std::path::Path::new("/tmp/archived-runs")),
                    "--dir must round-trip to Some(PathBuf); \
                     parse-scope only — resolver coverage lives \
                     with compare_partitions' own tests",
                );
                assert!(
                    filter.is_none(),
                    "bare --dir must not spuriously populate filter",
                );
                assert!(
                    threshold.is_none(),
                    "bare --dir must not spuriously populate threshold",
                );
                assert!(
                    policy.is_none(),
                    "bare --dir must not spuriously populate policy",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// Positive parse pin: `--policy PATH` round-trips to
    /// `StatsCommand::Compare { policy: Some(PathBuf(PATH)),
    /// threshold: None, ... }`. Mirrors `parse_stats_compare_with_dir`
    /// for the `dir` field. Uses an obviously-synthetic path that
    /// does not need to exist — the parse path never touches the
    /// filesystem; policy loading happens downstream in the
    /// dispatch.
    #[test]
    fn parse_stats_compare_with_policy() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "--policy",
            "/tmp/policy.json",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        threshold, policy, ..
                    }),
                ..
            } => {
                assert_eq!(
                    policy.as_deref(),
                    Some(std::path::Path::new("/tmp/policy.json")),
                    "--policy must round-trip to Some(PathBuf); got {policy:?}",
                );
                assert!(
                    threshold.is_none(),
                    "bare --policy must not populate --threshold",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--threshold` and `--policy` are mutually exclusive via
    /// clap `conflicts_with`. Passing both must be rejected at
    /// parse time, NOT reach the dispatch-level `unreachable!()`
    /// branch. A regression that dropped the `conflicts_with`
    /// attribute on either field would turn the `unreachable!()`
    /// into a panic at runtime instead of a parse error at parse
    /// time — this test catches that at compile-time parse
    /// behaviour.
    #[test]
    fn parse_stats_compare_rejects_both_threshold_and_policy() {
        // Avoid `expect_err` / `unwrap_err` because they require
        // the `Ok` type (`Cargo`) to implement `Debug`, which the
        // clap `Parser` derive does not add. A direct `match`
        // sidesteps the bound and keeps the test compiling
        // independently of whether `Cargo` gains `#[derive(Debug)]`
        // elsewhere.
        let result = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "--threshold",
            "5.0",
            "--policy",
            "/tmp/policy.json",
        ]);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!(
                "clap conflicts_with must reject both --threshold \
                 and --policy being set together"
            ),
        };
        let rendered = err.to_string();
        assert!(
            rendered
                .to_ascii_lowercase()
                .contains("cannot be used with")
                || rendered.to_ascii_lowercase().contains("conflict"),
            "clap error must surface the conflict between \
             --threshold and --policy; got: {rendered}",
        );
    }

    /// Bare `compare` defaults `--no-average` to `false` —
    /// averaging is the default since #117. `--no-average`
    /// must be opt-in for "keep each sidecar distinct"
    /// semantics.
    #[test]
    fn parse_stats_compare_no_average_default_false() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { no_average, .. }),
                ..
            } => {
                assert!(
                    !no_average,
                    "bare compare must default --no-average to false so \
                     averaging-on remains the default — operators get \
                     trial-set folding without an explicit flag.",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--no-average` parses as a bare flag (no value) and lifts
    /// the `no_average: bool` field on `StatsCommand::Compare`
    /// to true. Pins the clap binding so a regression that
    /// dropped the derive arg, renamed the flag, or accidentally
    /// made it take a value lands at parse time.
    #[test]
    fn parse_stats_compare_with_no_average() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "--no-average",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        no_average,
                        threshold,
                        policy,
                        dir,
                        ..
                    }),
                ..
            } => {
                assert!(no_average, "--no-average must lift the flag to true");
                assert!(
                    threshold.is_none(),
                    "bare --no-average must not spuriously populate --threshold",
                );
                assert!(
                    policy.is_none(),
                    "bare --no-average must not spuriously populate --policy",
                );
                assert!(
                    dir.is_none(),
                    "bare --no-average must not spuriously populate --dir",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--project-commit V` round-trips to `Compare { project_commit:
    /// vec![V], .. }`. Pins the clap binding for the shared
    /// `--project-commit` filter on the stats compare subcommand; a
    /// regression that removed the derive arg, renamed the flag, or
    /// dropped its `ArgAction::Append` would land here at parse time.
    #[test]
    fn parse_stats_compare_with_project_commit_single() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--project-commit",
            "abc1234",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        project_commit,
                        a_project_commit,
                        b_project_commit,
                        ..
                    }),
                ..
            } => {
                assert_eq!(project_commit, vec!["abc1234"]);
                assert!(
                    a_project_commit.is_empty(),
                    "shared --project-commit must not populate --a-project-commit",
                );
                assert!(
                    b_project_commit.is_empty(),
                    "shared --project-commit must not populate --b-project-commit",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--project-commit A --project-commit B` produces a Vec with two
    /// entries — the flag is `ArgAction::Append`, so multiple
    /// occurrences accumulate into the OR-combined filter the dispatch
    /// applies. A regression that lost the Append action would
    /// drop the first occurrence.
    #[test]
    fn parse_stats_compare_with_project_commit_repeatable() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--project-commit",
            "a",
            "--project-commit",
            "b",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { project_commit, .. }),
                ..
            } => {
                assert_eq!(project_commit, vec!["a", "b"]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--kernel-commit V` round-trips to `Compare {
    /// kernel_commit: vec![V], .. }`. Pins the clap binding for
    /// the shared `--kernel-commit` filter on the stats compare
    /// subcommand; a regression that removed the derive arg,
    /// renamed the flag, or dropped its `ArgAction::Append`
    /// would land here at parse time. Mirrors
    /// `parse_stats_compare_with_project_commit_single` for the
    /// `kernel_commit` dimension.
    #[test]
    fn parse_stats_compare_with_kernel_commit_single() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--kernel-commit",
            "abc1234",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        kernel_commit,
                        a_kernel_commit,
                        b_kernel_commit,
                        ..
                    }),
                ..
            } => {
                assert_eq!(kernel_commit, vec!["abc1234"]);
                assert!(
                    a_kernel_commit.is_empty(),
                    "shared --kernel-commit must not populate --a-kernel-commit",
                );
                assert!(
                    b_kernel_commit.is_empty(),
                    "shared --kernel-commit must not populate --b-kernel-commit",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--kernel-commit A --kernel-commit B` produces a Vec with
    /// two entries via `ArgAction::Append`. Mirrors
    /// `parse_stats_compare_with_commit_repeatable` for the
    /// kernel-commit dimension.
    #[test]
    fn parse_stats_compare_with_kernel_commit_repeatable() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--kernel-commit",
            "a",
            "--kernel-commit",
            "b",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { kernel_commit, .. }),
                ..
            } => {
                assert_eq!(kernel_commit, vec!["a", "b"]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--scheduler A --scheduler B` produces a Vec with two
    /// entries — the flag is `ArgAction::Append` (Vec, not
    /// Option), so multiple occurrences accumulate into the
    /// OR-combined filter the dispatch applies. Mirrors
    /// `parse_stats_compare_with_project_commit_repeatable` for
    /// the scheduler dimension. A regression that reverted
    /// `scheduler` to `Option<String>` (the pre-#13 shape) would
    /// fail this test at parse time — clap's `Option` derive
    /// rejects multiple occurrences with a "supplied more than
    /// once" diagnostic.
    #[test]
    fn parse_stats_compare_with_scheduler_repeatable() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--scheduler",
            "scx_alpha",
            "--scheduler",
            "scx_beta",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { scheduler, .. }),
                ..
            } => {
                assert_eq!(scheduler, vec!["scx_alpha", "scx_beta"]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--topology A --topology B` produces a Vec with two
    /// entries via `ArgAction::Append`. Mirrors the scheduler
    /// sibling above for the topology dimension. The Display form
    /// of `Topology` (e.g. `1n2l4c2t`) is the operator-visible
    /// label that flows verbatim through clap into this Vec.
    #[test]
    fn parse_stats_compare_with_topology_repeatable() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--topology",
            "1n2l4c2t",
            "--topology",
            "1n4l2c1t",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { topology, .. }),
                ..
            } => {
                assert_eq!(topology, vec!["1n2l4c2t", "1n4l2c1t"]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--work-type A --work-type B` produces a Vec with two
    /// entries via `ArgAction::Append`. Mirrors the scheduler /
    /// topology siblings above for the work_type dimension.
    /// Hyphenated CLI flag (`--work-type`) maps to underscored
    /// field name (`work_type`) per clap's default kebab-case
    /// rename — pin the field-vs-flag mapping by reading from the
    /// underscored field after a hyphenated invocation.
    #[test]
    fn parse_stats_compare_with_work_type_repeatable() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--work-type",
            "CpuSpin",
            "--work-type",
            "PageFaultChurn",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { work_type, .. }),
                ..
            } => {
                assert_eq!(work_type, vec!["CpuSpin", "PageFaultChurn"]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--a-kernel-commit X --b-kernel-commit Y` populates the
    /// per-side fields without touching the shared
    /// `kernel_commit`. Pins the clap binding for the per-side
    /// kernel-commit slicers — required for the
    /// `derive_slicing_dims` path to put `KernelCommit` in the
    /// slicing-dim set when the operator wants to slice by
    /// kernel HEAD.
    #[test]
    fn parse_stats_compare_with_per_side_kernel_commit() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel-commit",
            "abc1234",
            "--b-kernel-commit",
            "def5678",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        kernel_commit,
                        a_kernel_commit,
                        b_kernel_commit,
                        ..
                    }),
                ..
            } => {
                assert!(
                    kernel_commit.is_empty(),
                    "per-side --a-kernel-commit / --b-kernel-commit must not \
                     populate the shared --kernel-commit vec",
                );
                assert_eq!(a_kernel_commit, vec!["abc1234"]);
                assert_eq!(b_kernel_commit, vec!["def5678"]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    // -- BuildCompareFilters: symmetric sugar resolution --

    /// Empty input → both sides default. No filters populated
    /// anywhere; the dispatch site rejects this with the
    /// "specify at least one --a-X" error, but the builder
    /// itself just returns two empty filters.
    #[test]
    fn build_compare_filters_empty_yields_default_default() {
        let b = BuildCompareFilters::default();
        let (fa, fb) = b.build();
        assert!(fa.kernels.is_empty());
        assert!(fa.project_commits.is_empty());
        assert!(fa.kernel_commits.is_empty());
        assert!(fa.run_sources.is_empty());
        assert!(fa.schedulers.is_empty());
        assert!(fa.topologies.is_empty());
        assert!(fa.work_types.is_empty());
        assert!(fa.flags.is_empty());
        assert_eq!(fa.kernels, fb.kernels);
        assert_eq!(fa.project_commits, fb.project_commits);
        assert_eq!(fa.kernel_commits, fb.kernel_commits);
        assert_eq!(fa.run_sources, fb.run_sources);
        assert_eq!(fa.schedulers, fb.schedulers);
        assert_eq!(fa.topologies, fb.topologies);
        assert_eq!(fa.work_types, fb.work_types);
    }

    /// Per-side `--a-kernel-commit` overrides shared
    /// `--kernel-commit` for A only; B retains the shared value.
    /// Same "more-specific replaces" semantics as `--a-kernel`.
    /// The per-side override path is what populates the slicing
    /// dim on `KernelCommit` — without it, two sides with
    /// different live kernel HEADs cannot be contrasted in one
    /// `compare` invocation.
    #[test]
    fn build_compare_filters_per_side_kernel_commit_overrides_shared() {
        let b = BuildCompareFilters {
            shared_kernel_commit: vec!["abcdef1".to_string(), "fedcba2".to_string()],
            a_kernel_commit: vec!["111aaaa".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(
            fa.kernel_commits,
            vec!["111aaaa"],
            "A overrides shared kernel-commit",
        );
        assert_eq!(
            fb.kernel_commits,
            vec!["abcdef1", "fedcba2"],
            "B retains shared kernel-commit default",
        );
    }

    /// `--a-kernel-commit X --b-kernel-commit Y` slices on the
    /// `KernelCommit` dimension. Pins the slicing-dim derivation
    /// for the kernel-commit axis so a regression that dropped
    /// the dim from `derive_slicing_dims` lands here.
    #[test]
    fn build_compare_filters_disjoint_per_side_kernel_commit_slices() {
        let b = BuildCompareFilters {
            a_kernel_commit: vec!["abcdef1".to_string()],
            b_kernel_commit: vec!["fedcba2".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernel_commits, vec!["abcdef1"]);
        assert_eq!(fb.kernel_commits, vec!["fedcba2"]);
        let slicing = ktstr::cli::derive_slicing_dims(&fa, &fb);
        assert_eq!(
            slicing,
            vec![ktstr::cli::Dimension::KernelCommit],
            "differing per-side kernel-commit must derive as a single \
             KernelCommit slicing dim",
        );
    }

    /// Shared `--kernel V` pins BOTH sides to the same vec.
    /// Sugar for `--a-kernel V --b-kernel V`.
    #[test]
    fn build_compare_filters_shared_kernel_pins_both_sides() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.14"]);
    }

    /// Per-side `--a-kernel` overrides shared `--kernel` for A
    /// only; B retains the shared value. "More-specific
    /// replaces" semantics.
    #[test]
    fn build_compare_filters_per_side_overrides_shared_for_that_side_only() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string(), "6.15".to_string()],
            a_kernel: vec!["6.13".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.13"], "A overrides shared");
        assert_eq!(fb.kernels, vec!["6.14", "6.15"], "B retains shared default",);
    }

    /// Per-side overrides on the SAME dimension on BOTH sides
    /// produce the disjoint per-side filters the dispatch
    /// expects. This is the typical "slice on kernel" call shape:
    /// `--a-kernel A --b-kernel B`.
    #[test]
    fn build_compare_filters_disjoint_per_side_kernel_yields_two_filters() {
        let b = BuildCompareFilters {
            a_kernel: vec!["6.14".to_string()],
            b_kernel: vec!["6.15".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.15"]);
    }

    /// Per-side `--a-scheduler` overrides shared `--scheduler` for
    /// A only. Sibling test for the scheduler dimension after the
    /// #13 conversion from `Option<String>` to repeatable
    /// `Vec<String>` — the override semantics now mirror every
    /// other Vec dim ("non-empty per-side replaces shared
    /// verbatim"), so this test pins the same shape every other
    /// override-test pins for kernel / commit / source / etc.
    #[test]
    fn build_compare_filters_per_side_scheduler_overrides_shared() {
        let b = BuildCompareFilters {
            shared_scheduler: vec!["scx_default".to_string()],
            a_scheduler: vec!["scx_alpha".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(
            fa.schedulers,
            vec!["scx_alpha".to_string()],
            "A overrides shared scheduler",
        );
        assert_eq!(
            fb.schedulers,
            vec!["scx_default".to_string()],
            "B retains shared scheduler when only --a-scheduler overrides",
        );
    }

    /// Multi-dim sugar: shared `--kernel` pins both sides AND
    /// per-side `--a-scheduler` / `--b-scheduler` slice on
    /// scheduler. The resulting filters share kernel but slice
    /// on scheduler — exactly what the
    /// "narrow scope, slice on one axis" workflow needs.
    #[test]
    fn build_compare_filters_shared_pin_plus_per_side_slice() {
        let b = BuildCompareFilters {
            shared_kernel: vec!["6.14".to_string()],
            a_scheduler: vec!["scx_alpha".to_string()],
            b_scheduler: vec!["scx_beta".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.kernels, vec!["6.14"]);
        assert_eq!(fb.kernels, vec!["6.14"]);
        assert_eq!(fa.schedulers, vec!["scx_alpha".to_string()]);
        assert_eq!(fb.schedulers, vec!["scx_beta".to_string()]);
        // The slicing-dim derivation for these two filters
        // returns just [Scheduler] — kernel pins both sides
        // so the comparison joins on kernel and contrasts on
        // scheduler.
        let slicing = ktstr::cli::derive_slicing_dims(&fa, &fb);
        assert_eq!(slicing, vec![ktstr::cli::Dimension::Scheduler]);
    }

    /// `--a-flag` / `--b-flag` (AND-combined Vec) compose the
    /// same way as `--a-kernel` / `--b-kernel` (OR-combined
    /// Vec) — per-side empty defers to shared, per-side non-
    /// empty replaces. Pin the shape for the AND-combined dim
    /// to ensure no accidental special-case for OR-vs-AND.
    #[test]
    fn build_compare_filters_per_side_flag_overrides_shared() {
        let b = BuildCompareFilters {
            shared_flags: vec!["llc".to_string()],
            a_flags: vec!["steal".to_string(), "borrow".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.flags, vec!["steal", "borrow"]);
        assert_eq!(fb.flags, vec!["llc"]);
    }

    /// Sibling of `build_compare_filters_empty_yields_default_default`
    /// for the `run_sources` field. The existing empty-default test
    /// asserts on `run_sources` already (see line ~3499) — this
    /// companion adds the cross-side equality check that ensures
    /// `fa.run_sources == fb.run_sources` under the empty default,
    /// matching the same pattern other dimensions have. A regression
    /// that diverged the per-side `run_sources` defaults (e.g. by
    /// forgetting to thread `shared_run_source` into BOTH
    /// constructors in `BuildCompareFilters::build`) would surface
    /// here.
    #[test]
    fn build_compare_filters_empty_run_sources_field_equal_on_both_sides() {
        let b = BuildCompareFilters::default();
        let (fa, fb) = b.build();
        assert!(
            fa.run_sources.is_empty(),
            "empty BuildCompareFilters must produce A-side filter with empty run_sources",
        );
        assert!(
            fb.run_sources.is_empty(),
            "empty BuildCompareFilters must produce B-side filter with empty run_sources",
        );
        assert_eq!(
            fa.run_sources, fb.run_sources,
            "both sides must agree on empty run_sources",
        );
    }

    /// Per-side `--a-run-source` / `--b-run-source` produce
    /// disjoint per-side filters with the shared `run_sources`
    /// left empty. Mirrors
    /// `build_compare_filters_disjoint_per_side_kernel_yields_two_filters`
    /// for the run-source dimension. Pins the wiring of the
    /// `run_sources` field through `build()` so a regression that
    /// dropped it from the per-side branch — silently leaving
    /// `fa.run_sources` / `fb.run_sources` empty under per-side
    /// input — surfaces here.
    #[test]
    fn build_compare_filters_disjoint_per_side_source_yields_two_filters() {
        let b = BuildCompareFilters {
            a_run_source: vec!["ci".to_string()],
            b_run_source: vec!["local".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()]);
        assert_eq!(fb.run_sources, vec!["local".to_string()]);
    }

    /// Shared `--run-source` pins BOTH sides to the same vec.
    /// Sugar for `--a-run-source V --b-run-source V`. Mirrors
    /// `build_compare_filters_shared_kernel_pins_both_sides` for
    /// the run-source dimension.
    #[test]
    fn build_compare_filters_shared_source_pins_both_sides() {
        let b = BuildCompareFilters {
            shared_run_source: vec!["ci".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()]);
        assert_eq!(fb.run_sources, vec!["ci".to_string()]);
    }

    /// Per-side `--a-run-source` overrides shared `--run-source`
    /// for A only; B retains the shared value. "More-specific
    /// replaces" semantics — same shape as the existing
    /// `per_side_overrides_shared_for_that_side_only` for kernels.
    /// Pins the override resolution path for the run-source
    /// dimension.
    #[test]
    fn build_compare_filters_per_side_source_overrides_shared_for_that_side_only() {
        let b = BuildCompareFilters {
            shared_run_source: vec!["local".to_string(), "archive".to_string()],
            a_run_source: vec!["ci".to_string()],
            ..BuildCompareFilters::default()
        };
        let (fa, fb) = b.build();
        assert_eq!(fa.run_sources, vec!["ci".to_string()], "A overrides shared");
        assert_eq!(
            fb.run_sources,
            vec!["local".to_string(), "archive".to_string()],
            "B retains shared default",
        );
    }

    /// `cargo ktstr stats show-host --run X` parses to
    /// `StatsCommand::ShowHost { run: X, dir: None }`.
    #[test]
    fn parse_stats_show_host_with_run() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "show-host", "--run", "my-run-id"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ShowHost { run, dir }),
                ..
            } => {
                assert_eq!(run, "my-run-id");
                assert!(dir.is_none(), "bare --run must not populate --dir");
            }
            _ => panic!("expected Stats ShowHost"),
        }
    }

    /// `cargo ktstr stats show-host --run X --dir PATH` carries
    /// both flags through. Same --dir threading contract as
    /// `compare` — parse layer preserves the PathBuf; resolution
    /// against `runs_root()` is `cli::show_run_host`'s job.
    #[test]
    fn parse_stats_show_host_with_dir() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "show-host",
            "--run",
            "archive-2024-01-15",
            "--dir",
            "/tmp/archived-runs",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ShowHost { run, dir }),
                ..
            } => {
                assert_eq!(run, "archive-2024-01-15");
                assert_eq!(
                    dir.as_deref(),
                    Some(std::path::Path::new("/tmp/archived-runs")),
                );
            }
            _ => panic!("expected Stats ShowHost"),
        }
    }

    /// `cargo ktstr stats show-host` WITHOUT `--run` must fail at
    /// parse time — the flag is required and clap's default shape
    /// says so. A regression that accidentally made `--run`
    /// optional would silently let operators invoke the command
    /// with no target, producing a no-op failure.
    #[test]
    fn parse_stats_show_host_missing_run_rejected() {
        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "show-host"]);
        assert!(rejected.is_err(), "stats show-host must require --run",);
    }

    /// `cargo ktstr stats explain-sidecar --run X` parses to
    /// `StatsCommand::ExplainSidecar { run: X, dir: None,
    /// json: false }`. Mirrors `parse_stats_show_host_with_run`
    /// for the explain-sidecar shape.
    #[test]
    fn parse_stats_explain_sidecar_with_run() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "explain-sidecar",
            "--run",
            "my-run-id",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ExplainSidecar { run, dir, json }),
                ..
            } => {
                assert_eq!(run, "my-run-id");
                assert!(dir.is_none(), "bare --run must not populate --dir");
                assert!(!json, "default output is text, not json");
            }
            _ => panic!("expected Stats ExplainSidecar"),
        }
    }

    /// `cargo ktstr stats explain-sidecar --run X --dir PATH
    /// --json` carries all three flags. Same --dir threading
    /// contract as `show-host`; the `--json` flag toggles the
    /// aggregate-by-field output shape.
    #[test]
    fn parse_stats_explain_sidecar_with_dir_and_json() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "explain-sidecar",
            "--run",
            "archive-2024-01-15",
            "--dir",
            "/tmp/archived-runs",
            "--json",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::ExplainSidecar { run, dir, json }),
                ..
            } => {
                assert_eq!(run, "archive-2024-01-15");
                assert_eq!(
                    dir.as_deref(),
                    Some(std::path::Path::new("/tmp/archived-runs")),
                );
                assert!(json, "--json must toggle aggregate JSON output");
            }
            _ => panic!("expected Stats ExplainSidecar"),
        }
    }

    /// `cargo ktstr stats explain-sidecar` WITHOUT `--run` must
    /// fail at parse time. Same required-flag contract as
    /// `show-host`; without it, an operator could invoke the
    /// command with no target.
    #[test]
    fn parse_stats_explain_sidecar_missing_run_rejected() {
        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "stats", "explain-sidecar"]);
        assert!(
            rejected.is_err(),
            "stats explain-sidecar must require --run",
        );
    }

    // -- try_get_matches_from: kernel list --

    #[test]
    fn parse_kernel_list() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_list_json() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--json"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    /// `kernel list --range R` round-trips to
    /// `KernelCommand::List { range: Some(R), .. }` so the
    /// dispatch site routes through `kernel_list_range_preview`
    /// rather than the cache-walk path. Pins the clap binding
    /// for the new `--range` flag — a regression that dropped
    /// the `range` field from the Subcommand enum would surface
    /// here as a parse rejection.
    #[test]
    fn parse_kernel_list_range() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "list", "--range", "6.12..6.14"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Kernel { command } => match command {
                KernelCommand::List { json, range } => {
                    assert!(!json, "bare --range must not enable --json");
                    assert_eq!(
                        range.as_deref(),
                        Some("6.12..6.14"),
                        "--range must round-trip the literal spec for \
                         dispatch to pass to `expand_kernel_range`",
                    );
                }
                other => panic!("expected KernelCommand::List, got {other:?}"),
            },
            _ => panic!("expected Kernel"),
        }
    }

    /// `kernel list --range R --json` round-trips both flags.
    /// Pins the JSON-output mode is reachable on the range-preview
    /// path (a regression that wired `--range` only on the text
    /// path would surface here).
    #[test]
    fn parse_kernel_list_range_with_json() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "list",
            "--range",
            "6.12..6.14",
            "--json",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Kernel { command } => match command {
                KernelCommand::List { json, range } => {
                    assert!(json, "--json must round-trip alongside --range");
                    assert_eq!(range.as_deref(), Some("6.12..6.14"));
                }
                other => panic!("expected KernelCommand::List, got {other:?}"),
            },
            _ => panic!("expected Kernel"),
        }
    }

    /// `--run-source V` round-trips to `Compare { run_source: vec![V],
    /// .. }`. Pins the clap binding for the shared `--run-source`
    /// filter. Mirrors `parse_stats_compare_with_project_commit_single`
    /// for the new dimension; per-side `--a-run-source` /
    /// `--b-run-source` are covered by the `_per_side` sibling below.
    #[test]
    fn parse_stats_compare_with_run_source_single() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-kernel",
            "6.14",
            "--b-kernel",
            "6.15",
            "--run-source",
            "ci",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        run_source,
                        a_run_source,
                        b_run_source,
                        ..
                    }),
                ..
            } => {
                assert_eq!(
                    run_source,
                    vec!["ci".to_string()],
                    "shared --run-source must populate the shared vec",
                );
                assert!(
                    a_run_source.is_empty(),
                    "shared --run-source must not populate --a-run-source",
                );
                assert!(
                    b_run_source.is_empty(),
                    "shared --run-source must not populate --b-run-source",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--a-run-source A --b-run-source B` round-trips to populated
    /// per-side vecs with the shared `run_source` left empty. Pins
    /// the per-side override path that
    /// `BuildCompareFilters::build` consumes — a regression that
    /// merged shared and per-side into one bucket would surface
    /// here.
    #[test]
    fn parse_stats_compare_with_run_source_per_side() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "compare",
            "--a-run-source",
            "ci",
            "--b-run-source",
            "local",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        run_source,
                        a_run_source,
                        b_run_source,
                        ..
                    }),
                ..
            } => {
                assert!(
                    run_source.is_empty(),
                    "per-side flags must not populate the shared --run-source vec",
                );
                assert_eq!(a_run_source, vec!["ci".to_string()]);
                assert_eq!(b_run_source, vec!["local".to_string()]);
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    // -- try_get_matches_from: kernel build --

    #[test]
    fn parse_kernel_build_version() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "6.14.2"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_build_source() {
        let m =
            Cargo::try_parse_from(["cargo", "ktstr", "kernel", "build", "--source", "../linux"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_build_source_conflicts_with_version() {
        let m = Cargo::try_parse_from([
            "cargo", "ktstr", "kernel", "build", "--source", "../linux", "6.14.2",
        ]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_kernel_build_git_requires_ref() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "build",
            "--git",
            "https://example.com/linux.git",
        ]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_kernel_build_git_with_ref() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "build",
            "--git",
            "https://example.com/linux.git",
            "--ref",
            "v6.14",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_build_git_conflicts_with_source() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "kernel",
            "build",
            "--git",
            "https://example.com/linux.git",
            "--ref",
            "v6.14",
            "--source",
            "../linux",
        ]);
        assert!(m.is_err());
    }

    // -- try_get_matches_from: kernel clean --

    #[test]
    fn parse_kernel_clean() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_kernel_clean_keep() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "kernel", "clean", "--keep", "3"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Kernel {
                command: KernelCommand::Clean { keep, .. },
            } => {
                assert_eq!(keep, Some(3));
            }
            _ => panic!("expected Kernel Clean"),
        }
    }

    // -- try_get_matches_from: verifier --

    #[test]
    fn parse_verifier_with_scheduler() {
        let m =
            Cargo::try_parse_from(["cargo", "ktstr", "verifier", "--scheduler", "scx_rustland"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_verifier_with_scheduler_bin() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler-bin",
            "/tmp/sched",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_verifier_scheduler_conflicts_with_scheduler_bin() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler",
            "scx_rustland",
            "--scheduler-bin",
            "/tmp/sched",
        ]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_verifier_all_profiles() {
        let m = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler",
            "scx_rustland",
            "--all-profiles",
        ]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_verifier_profiles_filter() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "verifier",
            "--scheduler",
            "scx_rustland",
            "--profiles",
            "default,llc,llc+steal",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Verifier { profiles, .. } => {
                assert_eq!(profiles, vec!["default", "llc", "llc+steal"]);
            }
            _ => panic!("expected Verifier"),
        }
    }

    // -- try_get_matches_from: completions --

    #[test]
    fn parse_completions_bash() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "bash"]);
        assert!(m.is_ok(), "{}", m.err().unwrap());
    }

    #[test]
    fn parse_completions_invalid_shell() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "completions", "noshell"]);
        assert!(m.is_err());
    }

    // -- error cases --

    #[test]
    fn parse_missing_subcommand() {
        let m = Cargo::try_parse_from(["cargo", "ktstr"]);
        assert!(m.is_err());
    }

    #[test]
    fn parse_unknown_subcommand() {
        let m = Cargo::try_parse_from(["cargo", "ktstr", "nonexistent"]);
        assert!(m.is_err());
    }

    // -- topology parsing --

    #[test]
    fn topology_valid() {
        let parts: Vec<&str> = "1,2,4,1".split(',').collect();
        assert_eq!(parts.len(), 4);
        assert!(parts[0].parse::<u32>().is_ok());
        assert!(parts[1].parse::<u32>().is_ok());
        assert!(parts[2].parse::<u32>().is_ok());
        assert!(parts[3].parse::<u32>().is_ok());
    }

    #[test]
    fn topology_invalid_one_component() {
        let parts: Vec<&str> = "abc".split(',').collect();
        assert_ne!(parts.len(), 4);
    }

    #[test]
    fn topology_invalid_non_numeric() {
        let parts: Vec<&str> = "a,b,c,d".split(',').collect();
        assert_eq!(parts.len(), 4);
        assert!(parts[0].parse::<u32>().is_err());
    }

    #[test]
    fn topology_invalid_three_components() {
        let parts: Vec<&str> = "1,2,1".split(',').collect();
        assert_ne!(parts.len(), 4);
    }

    #[test]
    fn topology_invalid_zero_component() {
        // run_shell rejects zero values.
        let parts: Vec<&str> = "0,1,1,1".split(',').collect();
        assert_eq!(parts.len(), 4);
        let val: u32 = parts[0].parse().unwrap();
        assert_eq!(val, 0);
    }

    // -- completions --

    #[test]
    fn completions_bash_non_empty() {
        let mut buf = Vec::new();
        let mut cmd = Cargo::command();
        clap_complete::generate(clap_complete::Shell::Bash, &mut cmd, "cargo", &mut buf);
        assert!(!buf.is_empty());
    }

    #[test]
    fn completions_zsh_contains_subcommands() {
        let mut buf = Vec::new();
        let mut cmd = Cargo::command();
        clap_complete::generate(clap_complete::Shell::Zsh, &mut cmd, "cargo", &mut buf);
        let output = String::from_utf8(buf).expect("completions should be valid UTF-8");
        // clap_complete's zsh generator emits each subcommand as a
        // `'NAME:HELP'` describe-list entry (see `add_subcommands`
        // in clap_complete-4.6.1/src/aot/shells/zsh.rs:163). The
        // `'<name>:` prefix pin identifies an actual subcommand
        // completion, not an incidental substring match inside
        // rendered doc text.
        assert!(
            output.contains("'test:"),
            "zsh completions missing 'test:' describe-list entry"
        );
        assert!(
            output.contains("'coverage:"),
            "zsh completions missing 'coverage:' describe-list entry"
        );
        assert!(
            output.contains("'shell:"),
            "zsh completions missing 'shell:' describe-list entry"
        );
        assert!(
            output.contains("'kernel:"),
            "zsh completions missing 'kernel:' describe-list entry"
        );
        // `visible_alias = "nextest"` on the Test variant makes the
        // alias user-facing — clap_complete's zsh generator iterates
        // `get_visible_aliases` (zsh.rs:177) and emits a dedicated
        // describe entry per alias. A regression that dropped the
        // attribute (or silently switched to `alias` which is
        // NON-visible) would drop the entry and fail this assertion.
        assert!(
            output.contains("'nextest:"),
            "zsh completions missing 'nextest:' describe-list \
             entry (visible alias of `test`)"
        );
        // `LlvmCov` variant renders as the kebab-case `llvm-cov`
        // subcommand (clap derive default rename — see
        // clap_derive-4.6.0/src/item.rs:27 `DEFAULT_CASING =
        // CasingStyle::Kebab`). Pinned with the same `'name:`
        // prefix so an accidental doc-text match doesn't mask a
        // missing registration.
        assert!(
            output.contains("'llvm-cov:"),
            "zsh completions missing 'llvm-cov:' describe-list entry"
        );
    }

    // -- dispatch wiring: sub_argv constants --

    /// Byte-exact pin on the three `*_SUB_ARGV` constants that drive
    /// `run_test`, `run_coverage`, and `run_llvm_cov` into
    /// `run_cargo_sub`. A regression that re-ordered the Coverage
    /// tokens (e.g. swapped `["llvm-cov","nextest"]` → `["nextest",
    /// "llvm-cov"]`) would exec `cargo nextest llvm-cov` which is
    /// not a valid cargo subcommand, silently failing coverage
    /// runs. A regression that added a second token to
    /// `LLVM_COV_SUB_ARGV` (e.g. `["llvm-cov","test"]`) would
    /// prepend an implicit subcommand and override the user's
    /// trailing args. Both are caught here.
    #[test]
    fn cargo_sub_argv_constants_are_pinned() {
        assert_eq!(TEST_SUB_ARGV, &["nextest", "run"]);
        assert_eq!(COVERAGE_SUB_ARGV, &["llvm-cov", "nextest"]);
        assert_eq!(LLVM_COV_SUB_ARGV, &["llvm-cov"]);
    }

    // -- generate_flag_profiles --

    #[test]
    fn generate_flag_profiles_empty() {
        let profiles = generate_flag_profiles(&[]);
        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].0, "default");
        assert!(profiles[0].1.is_empty());
    }

    #[test]
    fn generate_flag_profiles_single_flag() {
        let flags = vec![ktstr::scenario::flags::FlagDeclJson {
            name: "llc".to_string(),
            args: vec!["--llc".to_string()],
            requires: vec![],
        }];
        let profiles = generate_flag_profiles(&flags);
        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].0, "default");
        assert_eq!(profiles[1].0, "llc");
    }

    #[test]
    fn generate_flag_profiles_requires_constraint() {
        let flags = vec![
            ktstr::scenario::flags::FlagDeclJson {
                name: "llc".to_string(),
                args: vec!["--llc".to_string()],
                requires: vec![],
            },
            ktstr::scenario::flags::FlagDeclJson {
                name: "steal".to_string(),
                args: vec!["--steal".to_string()],
                requires: vec!["llc".to_string()],
            },
        ];
        let profiles = generate_flag_profiles(&flags);
        // Valid: default, llc, llc+steal. Invalid: steal alone.
        let names: Vec<&str> = profiles.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(profiles.len(), 3);
        assert!(names.contains(&"default"));
        assert!(names.contains(&"llc"));
        assert!(names.contains(&"llc+steal"));
        assert!(!names.contains(&"steal"));
    }

    // -- profile_sched_args --

    #[test]
    fn profile_sched_args_collects_args() {
        let flags = vec![
            ktstr::scenario::flags::FlagDeclJson {
                name: "llc".to_string(),
                args: vec!["--llc".to_string()],
                requires: vec![],
            },
            ktstr::scenario::flags::FlagDeclJson {
                name: "steal".to_string(),
                args: vec!["--steal".to_string(), "--aggressive".to_string()],
                requires: vec![],
            },
        ];
        let active = vec!["llc".to_string(), "steal".to_string()];
        let args = profile_sched_args(&active, &flags).unwrap();
        assert_eq!(args, vec!["--llc", "--steal", "--aggressive"]);
    }

    #[test]
    fn profile_sched_args_empty() {
        let flags = vec![ktstr::scenario::flags::FlagDeclJson {
            name: "llc".to_string(),
            args: vec!["--llc".to_string()],
            requires: vec![],
        }];
        let active: Vec<String> = vec![];
        let args = profile_sched_args(&active, &flags).unwrap();
        assert!(args.is_empty());
    }

    #[test]
    fn profile_sched_args_unknown_flag_errors() {
        // Silently dropping unknown flag names would mask typos in
        // --profiles CLI lists and version-drift in cached nextest
        // args — the profile NAME would still say "foo+bar" while
        // the scheduler was invoked without the corresponding CLI
        // switches.
        let flags = vec![ktstr::scenario::flags::FlagDeclJson {
            name: "llc".to_string(),
            args: vec!["--llc".to_string()],
            requires: vec![],
        }];
        let active = vec!["llc".to_string(), "unknown_flag".to_string()];
        let err = profile_sched_args(&active, &flags).unwrap_err();
        assert!(
            err.contains("unknown_flag"),
            "error should cite flag: {err}"
        );
        assert!(err.contains("llc"), "error should list known flags: {err}");
    }

    // -- format_entry_row helpers --

    fn test_metadata() -> KernelMetadata {
        KernelMetadata::new(
            ktstr::cache::KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        )
        .with_version(Some("6.14.2".to_string()))
    }

    /// Store a fake kernel image and return the CacheEntry.
    fn store_test_entry(cache: &CacheDir, key: &str, meta: &KernelMetadata) -> CacheEntry {
        let src = tempfile::TempDir::new().unwrap();
        let image = src.path().join(&meta.image_name);
        std::fs::write(&image, b"fake kernel").unwrap();
        cache
            .store(key, &ktstr::cache::CacheArtifacts::new(&image), meta)
            .unwrap()
    }

    // -- format_entry_row --
    //
    // The (Matches / Stale / Untracked) × (not-EOL / EOL) outcome
    // matrix plus the `version == None` → "-" dash-render branch are
    // pinned by `format_entry_row_renders_eol_kconfig_matrix` in
    // `src/cli.rs` tests (cases c1-c7). The test below covers a
    // distinct corner the matrix does not: `KernelSource::Local`
    // rendering through format_entry_row, since the matrix uses
    // `Tarball` exclusively for determinism.

    #[test]
    fn format_entry_row_no_version() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let meta = KernelMetadata::new(
            ktstr::cache::KernelSource::Local {
                source_tree_path: None,
                git_hash: None,
            },
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-12T10:00:00Z".to_string(),
        );
        let entry = store_test_entry(&cache, "local-key", &meta);
        let row = cli::format_entry_row(&entry, "hash", &[]);
        assert!(row.contains("-"), "missing version should show dash");
    }

    // Corrupt-entry formatting moved inline into the caller iteration
    // in cli::kernel_list, so no test on format_entry_row covers it;
    // the helper itself now takes only the valid CacheEntry shape.

    // -- kconfig_status (via CacheEntry method) --

    /// Sibling of `format_entry_row_stale_kconfig`: that test pins the
    /// `(stale kconfig)` tag emitted by `cli::format_entry_row` for a
    /// hash-mismatch entry; this test pins the enum variant
    /// (`KconfigStatus::Stale { cached, current }`) returned by
    /// `CacheEntry::kconfig_status` that drives the tag.
    #[test]
    fn kconfig_status_reports_stale_on_hash_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let meta = test_metadata().with_ktstr_kconfig_hash(Some("old".to_string()));
        let entry = store_test_entry(&cache, "stale", &meta);
        assert_eq!(
            entry.kconfig_status("new"),
            ktstr::cache::KconfigStatus::Stale {
                cached: "old".to_string(),
                current: "new".to_string(),
            }
        );
    }

    /// Sibling of `format_entry_row_matching_kconfig`: that test pins
    /// the no-tag contract emitted by `cli::format_entry_row` when the
    /// hashes agree; this test pins the `KconfigStatus::Matches`
    /// variant returned by `CacheEntry::kconfig_status` that drives
    /// the no-tag branch.
    #[test]
    fn kconfig_status_reports_matches_on_hash_equality() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let meta = test_metadata().with_ktstr_kconfig_hash(Some("same".to_string()));
        let entry = store_test_entry(&cache, "fresh", &meta);
        assert_eq!(
            entry.kconfig_status("same"),
            ktstr::cache::KconfigStatus::Matches
        );
    }

    /// Sibling of
    /// `format_entry_row_untracked_kconfig_tagged_distinctly_from_stale`:
    /// that test pins the `(untracked kconfig)` tag emitted by
    /// `cli::format_entry_row` when an entry has no recorded hash;
    /// this test pins the `KconfigStatus::Untracked` variant returned
    /// by `CacheEntry::kconfig_status` that drives the tag.
    #[test]
    fn kconfig_status_reports_untracked_when_entry_has_no_hash() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let meta = test_metadata();
        let entry = store_test_entry(&cache, "no-hash", &meta);
        assert_eq!(
            entry.kconfig_status("anything"),
            ktstr::cache::KconfigStatus::Untracked
        );
    }

    // Corrupt entries no longer surface as CacheEntry — they are
    // ListedEntry::Corrupt with no metadata-bearing struct — so
    // kconfig_status isn't reachable from that state.

    /// Differential pin on the three `KconfigStatus` strings that flow
    /// into the `kconfig_status` field of `cargo ktstr kernel list
    /// --json`. `cli::kernel_list` emits the JSON field via
    /// `entry.kconfig_status(&kconfig_hash).to_string()`, so CI scripts
    /// that key off the stringified variant break if any of these
    /// three words changes. This test exercises the full
    /// `CacheEntry::kconfig_status(..).to_string()` chain (not just
    /// `KconfigStatus::<variant>.to_string()` in isolation) to pin the
    /// end-to-end JSON contract in a single test covering all three
    /// variants.
    #[test]
    fn kconfig_status_json_string_pins_all_three_variants() {
        use ktstr::cache::KconfigStatus;
        let tmp = tempfile::TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        let matches_meta = test_metadata().with_ktstr_kconfig_hash(Some("h".to_string()));
        let matches_entry = store_test_entry(&cache, "matches-key", &matches_meta);
        let matches_status = matches_entry.kconfig_status("h");
        assert!(
            matches!(matches_status, KconfigStatus::Matches),
            "hash equality must yield KconfigStatus::Matches"
        );
        assert_eq!(matches_status.to_string(), "matches");

        let stale_meta = test_metadata().with_ktstr_kconfig_hash(Some("old".to_string()));
        let stale_entry = store_test_entry(&cache, "stale-key", &stale_meta);
        let stale_status = stale_entry.kconfig_status("new");
        assert!(
            matches!(stale_status, KconfigStatus::Stale { .. }),
            "hash mismatch must yield KconfigStatus::Stale"
        );
        assert_eq!(stale_status.to_string(), "stale");

        let untracked_meta = test_metadata();
        let untracked_entry = store_test_entry(&cache, "untracked-key", &untracked_meta);
        let untracked_status = untracked_entry.kconfig_status("anything");
        assert!(
            matches!(untracked_status, KconfigStatus::Untracked),
            "entry without hash must yield KconfigStatus::Untracked"
        );
        assert_eq!(untracked_status.to_string(), "untracked");
    }

    // -- embedded_kconfig_hash --

    #[test]
    fn embedded_kconfig_hash_deterministic() {
        let h1 = cli::embedded_kconfig_hash();
        let h2 = cli::embedded_kconfig_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn embedded_kconfig_hash_is_hex() {
        let h = cli::embedded_kconfig_hash();
        assert_eq!(h.len(), 8, "CRC32 hex should be 8 chars");
        assert!(
            h.chars().all(|c| c.is_ascii_hexdigit()),
            "should be hex digits: {h}"
        );
    }

    #[test]
    fn embedded_kconfig_hash_matches_manual_crc32() {
        let expected = format!("{:08x}", crc32fast::hash(cli::EMBEDDED_KCONFIG.as_bytes()));
        assert_eq!(cli::embedded_kconfig_hash(), expected);
    }

    // -- show-host --

    /// `cargo ktstr show-host` parses with no arguments and maps to
    /// the `ShowHost` variant. A stray positional argument must be
    /// rejected (clap default) so a typo like
    /// `cargo ktstr show-host host_context` is caught at parse time.
    #[test]
    fn parse_show_host_minimal() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "show-host"])
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(k.command, KtstrCommand::ShowHost));

        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-host", "stray"]);
        assert!(
            rejected.is_err(),
            "show-host must reject positional arguments",
        );
    }

    /// `cargo ktstr show-thresholds <test>` parses with exactly one
    /// positional argument and maps to the `ShowThresholds` variant
    /// carrying the test name. Missing argument rejected at parse
    /// time; extra argument rejected too. Pins the arg count so a
    /// future variadic refactor surfaces here.
    #[test]
    fn parse_show_thresholds_with_test_arg() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds", "my_test_fn"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::ShowThresholds { test } => {
                assert_eq!(test, "my_test_fn");
            }
            _ => panic!("expected ShowThresholds"),
        }
    }

    /// `show-thresholds` without the test-name argument must fail
    /// at parse time — the positional is required.
    #[test]
    fn parse_show_thresholds_without_arg_rejected() {
        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds"]);
        assert!(
            rejected.is_err(),
            "show-thresholds requires a test-name argument",
        );
    }

    /// `show-thresholds <a> <b>` is rejected — variadic inputs would
    /// silently drop the second arg or reinterpret it as a flag.
    #[test]
    fn parse_show_thresholds_extra_arg_rejected() {
        let rejected = Cargo::try_parse_from(["cargo", "ktstr", "show-thresholds", "a", "b"]);
        assert!(
            rejected.is_err(),
            "show-thresholds must accept exactly one positional arg",
        );
    }

    /// `cli::show_host` produces a non-empty report under normal
    /// Linux CI conditions. Catches a regression in the underlying
    /// `HostContext::format_human` (e.g. a panic in the
    /// destructuring bind that surfaces every field) before the
    /// ShowHost dispatch arm reaches it. Named without a
    /// `dispatch_` prefix because this exercises the leaf helper
    /// directly; true dispatch-path coverage lives in the parse
    /// tests above + the binary's `main` call.
    #[test]
    fn show_host_helper_produces_non_empty_output() {
        let out = cli::show_host();
        assert!(
            !out.is_empty(),
            "show_host must return a non-empty report under normal Linux CI",
        );
        // Stronger pin: `HostContext::format_human` always includes
        // `kernel_release` even when most other fields are `None`
        // (uname is a syscall, filesystem-independent). Asserting
        // the stable field name catches a regression that returned
        // a non-empty but garbage report (e.g. only comments).
        assert!(
            out.contains("kernel_release"),
            "show_host output must include the stable `kernel_release` row: {out}",
        );
    }

    /// `cli::show_thresholds` returns `Err` with the actionable
    /// "no registered ktstr test named" diagnostic when called with
    /// an unknown test name. Named without a `dispatch_` prefix for
    /// the same reason as `show_host_helper_produces_non_empty_output`
    /// — this exercises the leaf helper, not the dispatch path
    /// wrapping it.
    #[test]
    fn show_thresholds_helper_unknown_test_returns_error() {
        let err = cli::show_thresholds("definitely_not_a_registered_test_xyz").unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no registered ktstr test named"),
            "error path must preserve the actionable diagnostic: {msg}",
        );
    }

    // -- clap argument-parse pins: Shell --cpu-cap requires --no-perf-mode
    //
    // `#[arg(long, requires = "no_perf_mode", ...)]` on the
    // Shell subcommand's `cpu_cap` field enforces the constraint
    // that --cpu-cap is only meaningful in no-perf-mode (perf-mode
    // already holds every LLC exclusively, so capping under
    // perf-mode would double-reserve). These tests pin the
    // invariant so a future refactor that drops or renames the
    // `requires` attribute trips a unit-test regression instead of
    // surfacing as a runtime double-reservation conflict.

    /// `cargo ktstr shell --cpu-cap 4 --no-perf-mode` parses
    /// successfully with both flags set. Pins the positive path of
    /// the `requires = "no_perf_mode"` constraint — the happy-path
    /// invocation an operator would type.
    #[test]
    fn parse_shell_cpu_cap_with_no_perf_mode_succeeds() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "shell",
            "--cpu-cap",
            "4",
            "--no-perf-mode",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell {
                cpu_cap,
                no_perf_mode,
                ..
            } => {
                assert_eq!(cpu_cap, Some(4));
                assert!(no_perf_mode, "--no-perf-mode must be set");
            }
            _ => panic!("expected Shell"),
        }
    }

    /// `cargo ktstr shell --cpu-cap 4` without `--no-perf-mode`
    /// must FAIL at parse time because of the `requires =
    /// "no_perf_mode"` constraint. Pins the negative path: if
    /// the constraint is ever dropped, this test fails so the
    /// regression can't reach production where it would cause a
    /// silent double-reservation under perf-mode.
    #[test]
    fn parse_shell_cpu_cap_without_no_perf_mode_fails() {
        // `Cargo` intentionally has no Debug derive, so unwrap
        // helpers that format the Ok variant are unavailable.
        // Match on Err directly to extract the clap error.
        let msg = match Cargo::try_parse_from(["cargo", "ktstr", "shell", "--cpu-cap", "4"]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("--cpu-cap without --no-perf-mode must fail the parse"),
        };
        // clap renders "the following required arguments were not provided"
        // or similar; lowercase + substring-match is lenient against
        // clap version-to-version message tweaks while still proving
        // the constraint fired.
        assert!(
            msg.to_ascii_lowercase().contains("no-perf-mode")
                || msg.to_ascii_lowercase().contains("no_perf_mode"),
            "clap error must name the missing --no-perf-mode flag, got: {msg}",
        );
    }

    /// `cargo ktstr shell --no-perf-mode` without `--cpu-cap`
    /// parses successfully with `cpu_cap: None`. Pins the shape of
    /// the unset sentinel (expanded to the 30%-of-allowed default by
    /// the planner) — a user who wants --no-perf-mode with the
    /// implicit default must still be able to invoke the shell. A
    /// regression that tied --cpu-cap to --no-perf-mode
    /// bidirectionally would fail here.
    #[test]
    fn parse_shell_no_perf_mode_without_cpu_cap_succeeds() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "shell", "--no-perf-mode"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Shell {
                cpu_cap,
                no_perf_mode,
                ..
            } => {
                assert_eq!(cpu_cap, None, "no --cpu-cap must produce None");
                assert!(no_perf_mode);
            }
            _ => panic!("expected Shell"),
        }
    }

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

    // ---------------------------------------------------------------
    // encode_kernel_list — KTSTR_KERNEL_LIST wire-format encoding
    // ---------------------------------------------------------------
    //
    // The wire format is `label1=path1;label2=path2;...` per the
    // doc comment on `encode_kernel_list`. Empty input encodes to
    // an empty string (idempotent — env var consumers treat the
    // empty value as "no list, single-kernel mode"). Paths
    // containing `;` are rejected with an actionable error since
    // the separator collision would produce a malformed env var
    // the test-binary parser would split into garbage.

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
        // POSIX permits `;` in paths, but the wire format claims
        // `;` as the entry separator. The encoder must bail with an
        // actionable error rather than silently producing
        // `label=foo;bar` which the parser would split into a
        // malformed `(label, "foo")` + spurious `bar` segment.
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
    //
    // Two distinct producer-side labels that normalize to the same
    // nextest identifier via `sanitize_kernel_label` would shatter
    // their cache directories under one test-name suffix. The
    // helper bails before the encoded `KTSTR_KERNEL_LIST` reaches
    // the test binary so the operator gets an actionable
    // "spell each --kernel value distinctly" diagnostic rather
    // than silent misroute to the wrong kernel.

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
        // operator's responsibility (or future task #21); this
        // helper is the LAST line of defense and must surface the
        // duplicate as a collision rather than silently letting
        // both entries through.
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
    // KERNEL_LIST_LONG_ABOUT — range-mode JSON schema discoverability
    // ---------------------------------------------------------------
    //
    // `cargo ktstr kernel list --range R --json` emits a
    // structurally-different JSON shape from the cache-walk mode:
    // four top-level fields (`range`, `start`, `end`, `versions`)
    // with no cache metadata. The help copy is the
    // discoverability contract for scripted consumers — without a
    // unit-test pin, a JSON emitter that adds, renames, or removes
    // a range-mode field could ship without a matching help update
    // and silently break dispatch-on-key consumers. The sibling
    // `kernel_list_long_about_exposes_json_schema` test in
    // `src/cli.rs` covers cache-walk mode; this companion fills
    // the range-mode gap from the cargo-ktstr binary's perspective
    // and exercises the same `pub const` re-exported through
    // `ktstr::cli::KERNEL_LIST_LONG_ABOUT`.

    /// Pins that every range-mode JSON top-level field name appears
    /// in the help copy by exact substring. Range-mode emits
    /// `{ range, start, end, versions }` per the schema block at
    /// `cli.rs:337-346`. Bare-word substring match is sufficient
    /// because the help copy embeds the field names in column-
    /// aligned table form (e.g. `  range     literal range string`)
    /// — distinct from the cache-walk schema's nullable-marker
    /// pattern which uses `{field} (nullable)` substrings.
    #[test]
    fn kernel_list_long_about_exposes_range_mode_json_keys() {
        let about = ktstr::cli::KERNEL_LIST_LONG_ABOUT;
        for range_field in ["range", "start", "end", "versions"] {
            assert!(
                about.contains(range_field),
                "KERNEL_LIST_LONG_ABOUT must mention range-mode JSON \
                 field `{range_field}` so scripted consumers discover \
                 the schema without `cargo doc`; got: {about:?}",
            );
        }
        // Stronger pin: the help copy must explicitly distinguish
        // range-mode from cache-walk-mode by mentioning that the
        // range-mode shape "never carries cache metadata" (the
        // dispatch-on-key contract). A regression that dropped the
        // range-mode block entirely while keeping the cache-walk
        // block (or vice versa) would pass the bare-word checks
        // above (`start` / `end` could match unrelated copy) but
        // fail this one.
        assert!(
            about.contains("--range"),
            "KERNEL_LIST_LONG_ABOUT must reference the `--range` flag \
             so a `kernel list --help` reader sees the range-mode \
             entry point: got: {about:?}",
        );
        assert!(
            about.contains("range-preview") || about.contains("range mode"),
            "KERNEL_LIST_LONG_ABOUT must explain that --range switches \
             to a structurally-different output shape so scripted \
             consumers know to dispatch on the presence of the \
             `range` key: got: {about:?}",
        );
    }

    // ---------------------------------------------------------------
    // preflight_collision_check — pre-resolve fast-fail
    // ---------------------------------------------------------------
    //
    // Pre-flight runs BEFORE the rayon resolve pipeline so a
    // colliding pair of cheap-to-label specs (Version / CacheKey /
    // Git) bails sub-millisecond rather than after a multi-minute
    // download + build cycle. Path and Range specs are intentionally
    // deferred to the post-resolve `detect_label_collisions` because
    // their labels require I/O (canonicalization for Path, a
    // releases.json fetch for Range).

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
    //
    // Identical `(label, path)` tuples are folded into one row so
    // `detect_label_collisions` does NOT trip on a benign duplicate
    // input. Tuples that share a label but DIFFER on the path are a
    // real cache-key collision and survive dedup intact.

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
