//! CLI argument types for the `cargo ktstr` binary.
//!
//! Houses the clap-derived `Cargo` / `CargoSub` / `Ktstr` /
//! `KtstrCommand` / `ModelCommand` / `StatsCommand` enums and structs
//! the binary entry point parses against. Pulled out of
//! [`super`] so the parent file stays focused on dispatch and
//! sub-helpers — the clap derive expansion is bulky enough to
//! dominate a single-file layout, and consumers (`Subcommand`
//! match arms, `try_parse_from` tests) only need the type
//! shapes here.

use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};
use ktstr::cli::KernelCommand;
use ktstr::cli::{KERNEL_HELP_NO_RAW, KERNEL_HELP_RAW_OK};

#[derive(Parser)]
#[command(name = "cargo-ktstr", bin_name = "cargo")]
pub(crate) struct Cargo {
    #[command(subcommand)]
    pub(crate) command: CargoSub,
}

#[derive(Subcommand)]
pub(crate) enum CargoSub {
    /// ktstr dev workflow: build kernel + run tests.
    Ktstr(Ktstr),
}

#[derive(Parser)]
pub(crate) struct Ktstr {
    #[command(subcommand)]
    pub(crate) command: KtstrCommand,
}

// Same rationale as `StatsCommand`'s sibling `#[allow]` — clap's
// derive expands every variant into a struct of `Option<T>` /
// `Vec<T>` per CLI flag, which after the per-side slicing flags
// were added pushes the Stats-via-Compare variant past clippy's
// large-variant heuristic. The enum is constructed once per CLI
// invocation and dispatched immediately; boxing every variant
// would distort the match ergonomics without measurable benefit.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
pub(crate) enum KtstrCommand {
    /// Build the kernel (if needed) and run tests via cargo nextest.
    #[command(visible_alias = "nextest")]
    Test {
        /// Repeatable. See [`KERNEL_HELP_NO_RAW`] for accepted shapes
        /// (path, version, cache key, range `START..END`, git source
        /// `git+URL#REF`). Multiple `--kernel` flags fan out the
        /// gauntlet across kernels: each `(test × scenario × topology
        /// × kernel)` tuple becomes a distinct nextest test case so
        /// nextest's parallelism, retries, and `-E` filtering all
        /// apply natively.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Promote hardware-driven test skips to hard failures.
        /// `ResourceContention` (no LLC slot / not enough CPUs / KVM
        /// fd budget) and host-topology-insufficient skips become
        /// exit 1 instead of silent passes. For CI environments
        /// where the hardware IS expected to support every test —
        /// a skip means the CI config is wrong, not that the test
        /// is inapplicable. Exports `KTSTR_NO_SKIP_MODE=1`.
        #[arg(long)]
        no_skip_mode: bool,
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
        /// Promote hardware-driven test skips to hard failures.
        /// See `cargo ktstr test --no-skip-mode` for the full
        /// contract. Exports `KTSTR_NO_SKIP_MODE=1`.
        #[arg(long)]
        no_skip_mode: bool,
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
        /// Promote hardware-driven test skips to hard failures.
        /// See `cargo ktstr test --no-skip-mode` for the full
        /// contract. Exports `KTSTR_NO_SKIP_MODE=1`.
        #[arg(long)]
        no_skip_mode: bool,
        /// Arguments passed through to cargo llvm-cov.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Print sidecar analysis from the most recent test run.
    ///
    /// Reads sidecar JSON files from the newest subdirectory under
    /// `{CARGO_TARGET_DIR or "target"}/ktstr/` (overridable with
    /// `KTSTR_SIDECAR_DIR`) and prints gauntlet analysis, BPF
    /// verifier stats, callback profile, and KVM stats. Test runs
    /// are partitioned into `{kernel}-{project_commit}` subdirectories,
    /// where `{project_commit}` is the project HEAD short hex with
    /// `-dirty` when the worktree differs; each subdirectory is
    /// the baseline snapshot of the most recent run at that
    /// (kernel, project commit) pair (re-running at the same key
    /// pre-clears prior sidecars before writing the new run).
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
    /// is already cached; `clean` deletes the cached artifact and
    /// its `.mtime-size` warm-cache sidecar.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Collect BPF verifier statistics for declared schedulers.
    ///
    /// Sweep mode: iterates every `declare_scheduler!` entry across
    /// the linked test binaries × their declared `kernels` × accepted
    /// gauntlet topology presets, reporting per-program verified
    /// instruction counts from host-side memory introspection.
    ///
    /// Temporarily a placeholder while the nextest-generated cell
    /// dispatch design lands; see task tracking for status.
    Verifier {
        /// Repeatable. See [`KERNEL_HELP_NO_RAW`] for accepted shapes
        /// (path / version / cache key / range / git source). Overrides
        /// the per-scheduler declared `kernels` set when supplied.
        #[arg(long, action = ArgAction::Append, help = KERNEL_HELP_NO_RAW)]
        kernel: Vec<String>,
        /// Print raw verifier output without formatting.
        #[arg(long)]
        raw: bool,
    },
    /// Throw a costume party for a JSON dump — replaces every
    /// non-metric value with a deterministic `adjective-animal`
    /// petname so a downstream LLM can reason about the
    /// structural shape of the dump without dragging real
    /// identifiers into its context. The transformation is
    /// one-way: there is no reverse-mapping file by design.
    ///
    /// Since v2 the walker funifies BY DEFAULT — every value
    /// whose containing key is NOT a recognised metric gets
    /// replaced. Two values that share the same key AND the
    /// same payload get the same fun name so cross-references
    /// inside the dump survive (e.g. "swift-otter migrated
    /// from CPU 3 to CPU 7" stays consistent).
    ///
    /// Example. Input:
    ///
    ///   {"comm": "scx_simple", "pid": 1234, "nr_running": 7}
    ///
    /// Output (with `--seed demo`, illustrative — exact funified
    /// values depend on the seed):
    ///
    ///   {"comm": "swift-otter", "pid": 8231554926718902741,
    ///    "nr_running": 7}
    ///
    /// `nr_running` is on the metric allowlist so 7 passes
    /// through; `comm` and `pid` are not, so they get funified.
    ///
    /// Metric allowlist categories (key passes through):
    ///   - structural enums: schema, version, type, kind, status,
    ///     state, result, verdict, outcome, phase, policy
    ///   - position / lifecycle: size, len, length, depth, index,
    ///     idx, level, tier, rank, slot, capacity, epoch,
    ///     generation
    ///   - top-level counts: nr_running, nr_queued, nr_failed,
    ///     nr_switches, runqueue_depth
    ///   - count suffixes: *_count, *_total, *_completed,
    ///     *_dropped, *_failed, *_skipped, *_throttled
    ///   - rates / ratios: *_per_sec, *_per_ms, *_rate, *_hz,
    ///     *_ratio, *_fraction, *_pct, *_percent
    ///   - units: *_ns, *_us, *_ms, *_sec, *_seconds, *_bytes,
    ///     *_kb, *_mb, *_gb, *_pages
    ///   - statistics: *_min, *_max, *_mean, *_avg, *_stddev,
    ///     *_p50, *_p90, *_p95, *_p99
    ///   - I/O counters: bytes_read, bytes_written, io_errors,
    ///     *_read, *_written, *_errors
    ///   - scheduling: priority, nice, weight, prio, static_prio,
    ///     normal_prio, nvcsw, nivcsw, signal_nvcsw,
    ///     signal_nivcsw, nr_threads
    ///   - per-rq SCX state: flags, ops_qseq, kick_sync, nr_immed,
    ///     rq_clock
    ///   - DSQ state: nr, seq
    ///   - NUMA event counters: numa_hit, numa_miss, numa_foreign,
    ///     numa_interleave_hit, numa_local, numa_other
    ///   - SCX exit-info events: select_cpu_fallback,
    ///     dispatch_local_dsq_offline, dispatch_keep_last,
    ///     enq_skip_exiting, enq_skip_migration_disabled,
    ///     reenq_immed, reenq_local_repeat, refill_slice_dfl,
    ///     bypass_duration, bypass_dispatch, bypass_activate,
    ///     insert_not_owned, sub_bypass_dispatch
    ///   - BPF prog runtime: cnt, nsecs, misses, verified_insns
    ///   - hardware perf: cycles, instructions, cache_misses,
    ///     branch_misses
    ///   - additional structural-enum / position suffixes:
    ///     *_kind, *_type, *_state, *_status, *_phase,
    ///     *_verdict, *_outcome, *_version, *_capacity, *_size,
    ///     *_depth, *_len, *_length, *_weight, *_nice,
    ///     *_priority, *_index, *_idx, *_offset, *_generation,
    ///     *_epoch
    ///
    /// Floats always pass through. Sentinel u64 values 0 and
    /// u64::MAX preserve their kthread / "no value" semantics.
    /// Reads JSON from `input` (or stdin when no path is given)
    /// and writes the funified JSON to stdout. Non-JSON input
    /// fails fast with the serde_json parse error.
    ///
    /// The category list above is documentary only — the actual
    /// allowlist lives in [`ktstr::fun::Funifier::is_metric_passthrough`].
    /// If categories are added or removed there, this list must be
    /// updated in lockstep so the user-visible help text matches
    /// runtime behaviour.
    ///
    /// Visible alias `costume` matches the costume-party theme.
    #[command(visible_alias = "costume")]
    Funify {
        /// Path to a JSON file (typically a `failure_dump.json` or
        /// debug-capture .json artefact). Pass `-` (or omit) to
        /// read from stdin.
        #[arg(value_name = "INPUT")]
        input: Option<PathBuf>,
        /// Optional seed string. With a fixed seed, the same input
        /// always produces the same fun output across invocations
        /// of this binary — useful for cross-dump correlation when
        /// multiple `funify` runs need to agree on names. Omit for
        /// a process-fresh ephemeral key (different fun names per
        /// run).
        #[arg(long)]
        seed: Option<String>,
        /// Pretty-print the output JSON. Default emits compact
        /// JSON suitable for piping into another tool.
        #[arg(long)]
        pretty: bool,
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
    /// and run B using the same [`ktstr::host_context::HostContext::diff`] logic.
    ShowHost,
    /// Print the resolved assertion thresholds for the named test.
    ///
    /// Dumps the merged `Assert` produced by the runtime merge chain
    /// `Assert::default_checks().merge(&entry.scheduler.assert).merge(&entry.assert)`
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
    /// Export a registered test as a self-extracting `.run` file
    /// that reproduces the scenario on bare metal without a VM.
    ///
    /// Bundles the running ktstr binary, the scheduler binary, and
    /// every include file the test declares into a gzipped tarball
    /// embedded in a bash preamble. The preamble validates root
    /// access, sched_ext support, cgroup2 mount, sched_ext-conflict
    /// (no other scheduler attached), and topology compatibility
    /// before extracting and launching. Chmod +x on the output so
    /// the operator can execute the `.run` directly.
    ///
    /// The frozen bits (scheduler choice, scheduler args, topology)
    /// match the test as registered. Overridable on the target host:
    /// `--duration`, `--watchdog-timeout`, `--quiet` (suppress
    /// banner). NOT overridable: `--cpus`, `--topology`, `--affinity`
    /// — re-export to change those.
    ///
    /// Out of scope for v1: `host_only` tests (they orchestrate
    /// cargo / nested VMs from inside the test body), tests with
    /// `bpf_map_write` (need the framework's host-side runtime
    /// probe surface), and `KernelBuiltin` schedulers (need the
    /// `enable` / `disable` shell commands the preamble doesn't
    /// emit yet). All three are rejected with actionable errors.
    ///
    /// # Name collisions
    ///
    /// If multiple workspace test binaries register a
    /// `#[ktstr_test]` with the same name, the router visits
    /// candidates in alphabetical order by absolute binary path
    /// and the FIRST binary that admits the test wins. Use
    /// `--package` to scope the search to a specific package and
    /// disambiguate deterministically.
    Export {
        /// Function-name-only test identifier as registered in
        /// `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`).
        /// Strip the `<binary>::` prefix that
        /// `cargo nextest list` prepends — the registry keys on the
        /// bare function name.
        test: String,
        /// Output path for the `.run` file. Defaults to
        /// `<test>.run` in the current directory.
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,
        /// Restrict the workspace search to a specific package. When
        /// omitted, every workspace member's tests is built and
        /// scanned for a matching `#[ktstr_test]` registration.
        /// Pass-through to `cargo build --tests --package <NAME>`.
        #[arg(short = 'p', long)]
        package: Option<String>,
        /// Build the test binaries with the release profile.
        /// Stricter assertion thresholds and `panic = "abort"` —
        /// match the profile the operator will run the .run file
        /// under, otherwise the embedded binary's behavior may
        /// drift from the dev-profile test runs the operator
        /// reproduced from.
        #[arg(long)]
        release: bool,
    },
    /// Enumerate every ktstr flock held on this host.
    ///
    /// Troubleshooting companion for `--cpu-cap` contention. Scans
    /// `{KTSTR_LOCK_DIR}/ktstr-llc-*.lock`,
    /// `{KTSTR_LOCK_DIR}/ktstr-cpu-*.lock` (default `/tmp`), and
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

        #[arg(long, help = ktstr::cli::DISK_HELP)]
        disk: Option<String>,
    },
}

#[derive(Subcommand)]
pub(crate) enum ModelCommand {
    /// Download the default pinned model and check its SHA-256.
    /// No-op when the cache already holds a SHA-checked copy.
    /// Respects `KTSTR_MODEL_OFFLINE` — set to `1` to refuse network
    /// fetches.
    Fetch,
    /// Print the cache path for the default model and whether a
    /// SHA-checked copy is already present.
    Status,
    /// Delete the cached GGUF artifact and its `.mtime-size`
    /// warm-cache sidecar. Subsequent `model fetch` re-downloads
    /// the pin from scratch. No-op when nothing is cached.
    Clean,
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
pub(crate) enum StatsCommand {
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
    /// found across all seven filterable dimensions: `kernel`,
    /// `commit`, `kernel_commit`, `source`, `scheduler`,
    /// `topology`, and `work_type`. The JSON keys `commit` and `source` map to the
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
        /// Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`;
        /// from `cargo ktstr stats list`).
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
    /// against the file count). Each `None` cause carries an
    /// optional `fix:` line with an operator-actionable
    /// remediation when one applies (e.g. "set KTSTR_KERNEL to
    /// a local kernel source tree" recovers `kernel_commit =
    /// None` for env-unset cases). When the walk encounters
    /// parse failures, the text output appends a trailing
    /// `corrupt sidecars (N):` block listing each corrupt path
    /// with the raw serde error message and (when applicable)
    /// an `enriched:` line with operator-facing remediation
    /// prose for known schema-drift cases. All-corrupt runs
    /// render the header + corrupt-block alone (no per-sidecar
    /// breakdown to render), preserving per-file diagnostic
    /// detail rather than collapsing to a single error line.
    ///
    /// `--json` emits a single object with three top-level
    /// keys: `_schema_version` (string version stamp —
    /// currently `"1"` — that consumers can gate on for
    /// incompatible shape changes), `_walk` (carrying the same
    /// walked / valid counts plus an `errors` array of
    /// `{path, error, enriched_message}` entries covering every
    /// parse failure; `enriched_message` is a JSON string
    /// when a known schema-drift remediation applies, JSON null
    /// otherwise), and `fields` (one entry per optional field
    /// with run-wide `none_count` + `some_count` summing to
    /// `_walk.valid`, plus the static `classification` /
    /// `causes` / `fix` catalog entry; `fix` is a JSON string
    /// when a remediation applies, JSON null otherwise). All-
    /// corrupt runs render the same shape with `valid = 0` and
    /// per-field counts at zero — never bail.
    ///
    /// Exit code is 0 even for all-corrupt runs — the
    /// diagnostic surface is the structured `_walk.errors`
    /// array (or the trailing `corrupt sidecars` text block),
    /// not the process exit code. CI scripts that need to fail
    /// on parse failures must gate on `_walk.valid > 0` or
    /// `_walk.errors.len() == 0` rather than the exit status.
    /// The only non-zero exits are missing-run-directory and
    /// empty-run (zero `.ktstr.json` files).
    ExplainSidecar {
        /// Run key (e.g. `6.14-abc1234` or `6.14-abc1234-dirty`;
        /// from `cargo ktstr stats list`).
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
        /// NOT match `6.14.20`. See `kernel_filter_matches` in
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
        /// Also accepts git revspecs (`HEAD`, `HEAD~N`, branch
        /// names, tags, `A..B` ranges) resolved against the
        /// project repo (`gix::discover` from cwd) into the same
        /// 7-char short hashes the sidecar writer records. A range
        /// expands to every commit reachable from `B` but not from
        /// `A`, each treated as an OR-combined exact-match
        /// filter. Example: `--project-commit HEAD~3..HEAD` keeps
        /// rows whose `project_commit` matches every commit
        /// reachable from `HEAD` but not from `HEAD~3` (the walk
        /// is breadth-first across the full commit DAG, so for
        /// linear histories this is "the last 3 commits"; for
        /// histories with merges it includes every commit on every
        /// branch joined since `HEAD~3`).
        /// Unrecognized inputs and `<hash>-dirty` forms (which
        /// revspec parsing rejects) pass through as literal
        /// exact-match filters, preserving compatibility with
        /// hand-typed dirty entries. When run outside any git tree,
        /// every input passes through as a literal — revspec
        /// resolution requires the cwd to be inside a project repo.
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
        /// Also accepts git revspecs (`HEAD`, `HEAD~N`, branch
        /// names, tags, `A..B` ranges) resolved against the
        /// kernel repo (`gix::open` against `KTSTR_KERNEL`'s
        /// path) into the same 7-char short hashes the sidecar
        /// writer records. A range expands to every commit
        /// reachable from `B` but not from `A`, each treated as
        /// an OR-combined exact-match filter. Example:
        /// `--kernel-commit v6.14..v6.15` keeps rows whose
        /// `kernel_commit` falls in that release window.
        /// Unrecognized inputs and `<hash>-dirty` forms (which
        /// revspec parsing rejects) pass through as literal
        /// exact-match filters, preserving compatibility with
        /// hand-typed dirty entries. When `KTSTR_KERNEL` is unset
        /// or points outside any git tree, every input passes
        /// through as a literal — revspec resolution requires the
        /// repo to be available.
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
        /// `work_type` field (e.g. `--work-type SpinWait`). Valid
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
        /// `kernel_filter_matches` for the cutoff implementation.
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
        /// `kernel_filter_matches` for the cutoff implementation.
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
