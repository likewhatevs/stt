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

#[derive(Subcommand)]
enum KtstrCommand {
    /// Build the kernel (if needed) and run tests via cargo nextest.
    #[command(visible_alias = "nextest")]
    Test {
        #[arg(long, help = KERNEL_HELP_NO_RAW)]
        kernel: Option<String>,
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
        #[arg(long, help = KERNEL_HELP_NO_RAW)]
        kernel: Option<String>,
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
        #[arg(long, help = KERNEL_HELP_NO_RAW)]
        kernel: Option<String>,
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
        #[arg(long, help = KERNEL_HELP_RAW_OK)]
        kernel: Option<String>,
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
    /// Print the archived host context for a specific run.
    ///
    /// Resolves `--run <id>` against `test_support::runs_root()`
    /// (or `--dir` when set), loads any sidecar file under that
    /// run directory, and renders the `host` field via
    /// `HostContext::format_human`. Useful for inspecting the
    /// CPU model, memory config, THP policy, and sched_* tunables
    /// captured at archive time — the same fingerprint
    /// `compare_runs` uses for its host-delta section, now
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
    /// Compare two test runs and report regressions.
    Compare {
        /// Run key A (from `cargo ktstr stats list`).
        a: String,
        /// Run key B (from `cargo ktstr stats list`).
        b: String,
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
        /// Strict equality match against the sidecar's
        /// `kernel_version` field (e.g. `--kernel-version 6.14.2`).
        /// Rows whose `kernel_version` is `None` (sidecar writer
        /// could not extract a version) NEVER match a `Some` filter
        /// — passing `--kernel-version` is an opt-in that demands a
        /// known-version row.
        #[arg(long)]
        kernel_version: Option<String>,
        /// Strict equality match against the sidecar's `scheduler`
        /// field (e.g. `--scheduler scx_rusty`). Distinct from `-E`,
        /// which matches a substring across the joined fields. Use
        /// this when the operator wants to pin a specific scheduler
        /// rather than narrow on a fragment.
        #[arg(long)]
        scheduler: Option<String>,
        /// Strict equality match against the rendered topology label
        /// (e.g. `--topology 1n2l4c2t`). The label is what
        /// `Topology::Display` produces; `cargo ktstr stats list`
        /// shows the form per-row.
        #[arg(long)]
        topology: Option<String>,
        /// Strict equality match against the sidecar's `work_type`
        /// field (e.g. `--work-type CpuSpin`). Valid names are the
        /// PascalCase variants of `WorkType` — see
        /// `WorkType::ALL_NAMES` in `ktstr::workload` for the
        /// compile-time list.
        #[arg(long)]
        work_type: Option<String>,
        /// Repeatable AND-combined flag filter (e.g.
        /// `--flag llc --flag rusty_balance`). Every flag listed
        /// must be present in the sidecar's `active_flags`; the row
        /// may carry additional flags beyond the filter set. Empty
        /// repeats are rejected by clap (zero-width match).
        #[arg(long = "flag")]
        flags: Vec<String>,
        /// Aggregate every (scenario, topology, work_type, flags)
        /// group into a single row carrying the arithmetic mean
        /// of its passing contributors' metrics. Smooths run-to-
        /// run jitter when each side carries multiple trials per
        /// key. The header line above the comparison table reports
        /// the post-aggregation row counts; a per-key `N/M`
        /// (`passes_observed`/`total_observed`) block prints below
        /// the summary for every group whose contributor count
        /// includes a failure or skip on either side.
        ///
        /// Aggregation rules: failing/skipped contributors are
        /// excluded from the metric mean (they would carry
        /// failure-mode telemetry, not scheduler behaviour); the
        /// aggregated row's `passed` is the AND across every
        /// contributor (a single failure flips the aggregate to
        /// `failed`, which routes the pair through `compare_rows`'
        /// `skipped_failed` gate).
        #[arg(long)]
        average: bool,
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

/// Shared runner for `cargo ktstr test`, `cargo ktstr coverage`, and
/// `cargo ktstr llvm-cov`.
///
/// All three subcommands have the same plumbing: resolve `--kernel` to
/// a cache entry or source-tree build, propagate `--no-perf-mode` via
/// an env var, optionally prepend `--cargo-profile release`, append
/// the user's trailing args, and `exec` into the chosen cargo
/// subcommand. The only differences are the cargo subcommand name
/// (`["nextest","run"]` vs `["llvm-cov","nextest"]` vs `["llvm-cov"]`)
/// and the log / error-message prefix. Consolidating here ensures all
/// flows evolve together — a kernel-resolution fix in one used to
/// drift against the others.
///
/// `release` is always `false` for the raw `llvm-cov` passthrough —
/// that subcommand hands every argument to the user, so the profile
/// is set via the user's trailing args (or not at all). `test` and
/// `coverage` wire their `--release` flag through to this argument.
fn run_cargo_sub(
    sub_argv: &[&str],
    label: &str,
    kernel: Option<String>,
    no_perf_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    use ktstr::kernel_path::KernelId;

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

    if let Some(ref val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                // Canonicalize BEFORE export. The exec'd cargo
                // subcommand changes cwd to the workspace root (or
                // wherever `cargo` itself runs from), so a relative
                // path like `../linux` interpreted from cargo-ktstr's
                // cwd and the same string interpreted from the child's
                // cwd resolve to different directories. `canonicalize`
                // pins the string to an absolute realpath at the
                // parent's cwd so every downstream reader
                // (`find_kernel` in the child, `detect_kernel_version`
                // in the sidecar writer, `find_test_vmlinux` in any
                // nested probe) sees the same directory. The previous
                // `unwrap_or_else(|_| PathBuf::from(&p))` silently
                // fell back to the raw string on canonicalize failure
                // — bail loudly instead so a typo doesn't exec into a
                // successful-looking run that silently picks up a
                // different kernel via `find_kernel`'s fallback chain.
                let dir = std::fs::canonicalize(&p).map_err(|e| {
                    format!(
                        "--kernel {}: path does not exist or cannot be \
                         canonicalized ({e:#}). {hint}",
                        p.display(),
                        hint = ktstr::KTSTR_KERNEL_HINT,
                    )
                })?;
                // Boundary bridge: `build_kernel` returns
                // `anyhow::Result<()>` while this function still
                // returns `Result<(), String>`, so we stringify at
                // the call site. A broader anyhow migration across
                // cargo-ktstr.rs is pending and would drop this
                // last bridge.
                build_kernel(&dir, false).map_err(|e| format!("{e:#}"))?;
                cmd.env(ktstr::KTSTR_KERNEL_ENV, &dir);
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = ktstr::cli::resolve_cached_kernel(&id, "cargo ktstr")
                    .map_err(|e| format!("{e:#}"))?;
                // Canonicalize the cache dir defensively. `CacheDir`
                // roots at the XDG cache home (or `KTSTR_CACHE_DIR`),
                // both of which are typically absolute — but an
                // operator-supplied `KTSTR_CACHE_DIR=./cache` would
                // produce a relative path here and reach the same
                // cwd-divergence bug the Path branch defends against.
                // `canonicalize` resolves that from the parent's cwd;
                // a failure means the cache dir was removed between
                // lookup and export (rare race), in which case we
                // fall back to the original path rather than bailing
                // — the child will re-enter its own cache lookup and
                // surface the real missing-entry error.
                let dir = std::fs::canonicalize(&cache_dir).unwrap_or(cache_dir);
                eprintln!("cargo ktstr: using kernel {}", dir.display());
                cmd.env(ktstr::KTSTR_KERNEL_ENV, &dir);
            }
            // Multi-kernel specs cannot resolve to a single
            // KTSTR_KERNEL export here. The dispatch loop that fans
            // out range expansion and git fetch will land at this
            // call site in a follow-up stage; for now, surface an
            // actionable error so the user knows the spec parsed
            // correctly but the calling subcommand hasn't wired up
            // the multi-kernel pipeline yet.
            //
            // Run `validate()` first so an inverted range surfaces
            // the specific "swap the endpoints" diagnostic before
            // the generic "not yet supported" redirect masks it,
            // matching the pattern in `cli::resolve_kernel_image`
            // and `cli::resolve_cached_kernel`.
            id @ (KernelId::Range { .. } | KernelId::Git { .. }) => {
                id.validate().map_err(|e| format!("--kernel {val}: {e}"))?;
                return Err(format!(
                    "--kernel {val}: kernel ranges and git sources are \
                     not yet supported in this context — use a single \
                     kernel version, cache key, or path"
                ));
            }
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
    kernel: Option<String>,
    no_perf_mode: bool,
    release: bool,
    args: Vec<String>,
) -> Result<(), String> {
    run_cargo_sub(TEST_SUB_ARGV, "tests", kernel, no_perf_mode, release, args)
}

fn run_coverage(
    kernel: Option<String>,
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

fn run_llvm_cov(
    kernel: Option<String>,
    no_perf_mode: bool,
    args: Vec<String>,
) -> Result<(), String> {
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
        Some(StatsCommand::ShowHost { run, dir }) => {
            match cli::show_run_host(run, dir.as_deref()) {
                Ok(s) => {
                    print!("{s}");
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            }
        }
        Some(StatsCommand::Compare {
            a,
            b,
            filter,
            threshold,
            policy,
            dir,
            kernel_version,
            scheduler,
            topology,
            work_type,
            flags,
            average,
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
            // Typed filters compose with the substring `-E` filter:
            // typed narrows happen first inside `compare_runs` (strict
            // equality / AND-combined), then the substring runs over
            // the surviving set inside `compare_rows`. See
            // `RowFilter` doc for the full match-semantics contract.
            let row_filter = ktstr::cli::RowFilter {
                kernel_version: kernel_version.clone(),
                scheduler: scheduler.clone(),
                topology: topology.clone(),
                work_type: work_type.clone(),
                flags: flags.clone(),
            };
            let exit = cli::compare_runs(
                a,
                b,
                filter.as_deref(),
                &row_filter,
                &resolved_policy,
                dir.as_deref(),
                *average,
            )
            .map_err(|e| format!("{e:#}"))?;
            if exit != 0 {
                std::process::exit(exit);
            }
            Ok(())
        }
    }
}

/// Acquire source, configure, build, and cache a kernel image.
fn kernel_build(
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
    kernel: Option<String>,
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

    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    // Build the ktstr init binary.
    let ktstr_bin =
        ktstr::build_and_find_binary("ktstr").map_err(|e| format!("build ktstr: {e:#}"))?;

    if all_profiles || !profiles_filter.is_empty() {
        return run_verifier_all_profiles(
            &sched_bin,
            &ktstr_bin,
            &kernel_path,
            raw,
            &profiles_filter,
        );
    }

    eprintln!("cargo ktstr: collecting verifier stats");
    let result =
        ktstr::verifier::collect_verifier_output(&sched_bin, &ktstr_bin, &kernel_path, &[])
            .map_err(|e| format!("collect verifier output: {e:#}"))?;

    let output = ktstr::verifier::format_verifier_output("verifier", &result, raw);
    print!("{output}");

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
            KernelCommand::List { json } => cli::kernel_list(json).map_err(|e| format!("{e:#}")),
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
                assert_eq!(kernel.as_deref(), Some("6.14.2"));
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
                assert_eq!(kernel.as_deref(), Some("6.14.2"));
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
                assert_eq!(kernel.as_deref(), Some("6.14.2"));
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
                assert_eq!(kernel.as_deref(), Some("6.14.2"));
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

    #[test]
    fn parse_stats_compare() {
        let m =
            Cargo::try_parse_from(["cargo", "ktstr", "stats", "compare", "6.14-abc", "6.15-def"]);
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
            "a",
            "b",
            "-E",
            "cgroup_steady",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        a,
                        b,
                        filter,
                        threshold,
                        policy,
                        dir,
                        ..
                    }),
                ..
            } => {
                assert_eq!(a, "a");
                assert_eq!(b, "b");
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
            "a",
            "b",
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
            "a",
            "b",
            "--dir",
            "/tmp/archived-runs",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        a,
                        b,
                        filter,
                        threshold,
                        policy,
                        dir,
                        ..
                    }),
                ..
            } => {
                assert_eq!(a, "a");
                assert_eq!(b, "b");
                assert_eq!(
                    dir.as_deref(),
                    Some(std::path::Path::new("/tmp/archived-runs")),
                    "--dir must round-trip to Some(PathBuf); \
                     parse-scope only — resolver coverage lives \
                     with stats::compare_runs' own tests",
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
            "a",
            "b",
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
            "a",
            "b",
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

    /// Bare `compare` defaults `--average` to `false` — the
    /// aggregation path must be opt-in so existing CLI invocations
    /// retain their per-trial-row comparison semantics.
    #[test]
    fn parse_stats_compare_average_default_false() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "compare", "a", "b"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command: Some(StatsCommand::Compare { average, .. }),
                ..
            } => {
                assert!(
                    !average,
                    "bare compare must default --average to false so \
                     existing scripts retain per-trial-row semantics",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
    }

    /// `--average` parses as a bare flag (no value) and lifts the
    /// `average: bool` field on `StatsCommand::Compare` to true.
    /// Pins the clap binding so a regression that dropped the
    /// derive arg, renamed the flag, or accidentally made it
    /// take a value lands at parse time.
    #[test]
    fn parse_stats_compare_with_average() {
        let Cargo {
            command: CargoSub::Ktstr(k),
        } = Cargo::try_parse_from(["cargo", "ktstr", "stats", "compare", "a", "b", "--average"])
            .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::Stats {
                command:
                    Some(StatsCommand::Compare {
                        average,
                        threshold,
                        policy,
                        dir,
                        ..
                    }),
                ..
            } => {
                assert!(average, "--average must lift the flag to true");
                assert!(
                    threshold.is_none(),
                    "bare --average must not spuriously populate --threshold",
                );
                assert!(
                    policy.is_none(),
                    "bare --average must not spuriously populate --policy",
                );
                assert!(
                    dir.is_none(),
                    "bare --average must not spuriously populate --dir",
                );
            }
            _ => panic!("expected Stats Compare"),
        }
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
}
