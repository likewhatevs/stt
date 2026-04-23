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
    Test {
        #[arg(long, help = KERNEL_HELP_NO_RAW)]
        kernel: Option<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Arguments passed through to cargo nextest run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build the kernel (if needed) and run tests with coverage via cargo llvm-cov nextest.
    Coverage {
        #[arg(long, help = KERNEL_HELP_NO_RAW)]
        kernel: Option<String>,
        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
        /// Arguments passed through to cargo llvm-cov nextest.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Print sidecar analysis from the most recent test run.
    ///
    /// Reads sidecar JSON files from the newest subdirectory under
    /// `{CARGO_TARGET_DIR or "target"}/ktstr/` (overridable with
    /// `KTSTR_SIDECAR_DIR`) and prints gauntlet analysis, BPF
    /// verifier stats, callback profile, and KVM stats. Each test
    /// run is its own subdirectory keyed `{kernel}-{git_short}`;
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
    /// enumerate test names.
    ShowThresholds {
        /// Fully qualified test name as registered in
        /// `#[ktstr_test]` (e.g. `preempt_regression_fault_under_load`).
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

#[derive(Subcommand)]
enum StatsCommand {
    /// List test runs under `{CARGO_TARGET_DIR or "target"}/ktstr/`.
    List,
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
        /// Relative significance threshold in percent (e.g. 10 for
        /// 10%). When set, overrides the per-metric default
        /// threshold for ALL metrics — intentionally, so callers
        /// can loosen a tight default or tighten a loose one from
        /// the CLI without per-metric knobs. Omit to use each
        /// metric's built-in default.
        #[arg(long)]
        threshold: Option<f64>,
        /// Alternate run root to resolve `a` / `b` against. Defaults
        /// to `test_support::runs_root()` (typically `target/ktstr/`).
        /// Useful when comparing archived sidecar trees copied off a
        /// CI host.
        #[arg(long)]
        dir: Option<std::path::PathBuf>,
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
        cli::make_kernel_with_output(kernel_dir, Some(sp))
    })?;

    cli::validate_kernel_config(kernel_dir)?;

    cli::Spinner::with_progress("Generating compile_commands.json...", "Done", |sp| {
        cli::run_make_with_output(kernel_dir, &["compile_commands.json"], Some(sp))
    })?;
    Ok(())
}

/// Shared runner for `cargo ktstr test` and `cargo ktstr coverage`.
///
/// Both subcommands have the same plumbing: resolve `--kernel` to a
/// cache entry or source-tree build, propagate `--no-perf-mode` via an
/// env var, append the user's trailing args, and `exec` into the
/// chosen cargo subcommand. The only differences are the cargo
/// subcommand name (`["nextest","run"]` vs `["llvm-cov","nextest"]`)
/// and the log / error-message prefix. Consolidating here ensures
/// both flows evolve together — a kernel-resolution fix in one used
/// to drift against the other.
fn run_cargo_sub(
    sub_argv: &[&str],
    label: &str,
    kernel: Option<String>,
    no_perf_mode: bool,
    args: Vec<String>,
) -> Result<(), String> {
    use ktstr::kernel_path::KernelId;

    let mut cmd = Command::new("cargo");
    cmd.args(sub_argv).args(&args);

    if no_perf_mode {
        cmd.env("KTSTR_NO_PERF_MODE", "1");
    }

    if let Some(ref val) = kernel {
        match KernelId::parse(val) {
            KernelId::Path(p) => {
                let dir = std::fs::canonicalize(&p).unwrap_or_else(|_| PathBuf::from(&p));
                // Boundary bridge: `build_kernel` returns
                // `anyhow::Result<()>` while this function still
                // returns `Result<(), String>`, so we stringify at
                // the call site. A broader anyhow migration across
                // cargo-ktstr.rs is pending and would drop this
                // last bridge.
                build_kernel(&dir, false).map_err(|e| format!("{e:#}"))?;
                cmd.env("KTSTR_KERNEL", &dir);
            }
            id @ (KernelId::Version(_) | KernelId::CacheKey(_)) => {
                let cache_dir = ktstr::cli::resolve_cached_kernel(&id, "cargo ktstr")
                    .map_err(|e| format!("{e:#}"))?;
                eprintln!("cargo ktstr: using kernel {}", cache_dir.display());
                cmd.env("KTSTR_KERNEL", &cache_dir);
            }
        }
    }

    eprintln!("cargo ktstr: running {label}");
    let err = cmd.exec();
    Err(format!("exec cargo {}: {err}", sub_argv.join(" ")))
}

fn run_test(kernel: Option<String>, no_perf_mode: bool, args: Vec<String>) -> Result<(), String> {
    run_cargo_sub(&["nextest", "run"], "tests", kernel, no_perf_mode, args)
}

fn run_coverage(
    kernel: Option<String>,
    no_perf_mode: bool,
    args: Vec<String>,
) -> Result<(), String> {
    run_cargo_sub(
        &["llvm-cov", "nextest"],
        "coverage",
        kernel,
        no_perf_mode,
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
            dir,
        }) => {
            let exit = cli::compare_runs(a, b, filter.as_deref(), *threshold, dir.as_deref())
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
) -> Result<(), String> {
    let cache = CacheDir::new().map_err(|e| format!("open cache: {e:#}"))?;

    // Temporary directory for tarball/git source extraction.
    let tmp_dir = tempfile::TempDir::new().map_err(|e| format!("create temp dir: {e:#}"))?;

    // Acquire source.
    let client = fetch::shared_client();
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path, "cargo ktstr").map_err(|e| format!("{e:#}"))?
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

    cli::kernel_build_pipeline(&acquired, &cache, "cargo ktstr", clean, source.is_some())
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

fn run_shell(
    kernel: Option<String>,
    topology: String,
    include_files: Vec<PathBuf>,
    memory_mb: Option<u32>,
    dmesg: bool,
    exec: Option<String>,
) -> Result<(), String> {
    cli::check_kvm().map_err(|e| format!("{e:#}"))?;
    let kernel_path = resolve_kernel_image(kernel.as_deref())?;

    // Parse topology "N,L,C,T" (numa_nodes,llcs,cores,threads).
    let parts: Vec<&str> = topology.split(',').collect();
    if parts.len() != 4 {
        return Err(format!(
            "invalid topology '{topology}': expected 'numa_nodes,llcs,cores,threads' (e.g. '1,2,4,1')"
        ));
    }
    let numa_nodes: u32 = parts[0]
        .parse()
        .map_err(|_| format!("invalid numa_nodes value: '{}'", parts[0]))?;
    let llcs: u32 = parts[1]
        .parse()
        .map_err(|_| format!("invalid llcs value: '{}'", parts[1]))?;
    let cores: u32 = parts[2]
        .parse()
        .map_err(|_| format!("invalid cores value: '{}'", parts[2]))?;
    let threads: u32 = parts[3]
        .parse()
        .map_err(|_| format!("invalid threads value: '{}'", parts[3]))?;
    if numa_nodes == 0 || llcs == 0 || cores == 0 || threads == 0 {
        return Err(format!(
            "invalid topology '{topology}': all values must be >= 1"
        ));
    }

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
    match &status.sha_verdict {
        ktstr::test_support::ShaVerdict::NotCached => println!(
            "(no cached copy — run `cargo ktstr model fetch` to download {} MiB)",
            status.spec.size_bytes / 1024 / 1024,
        ),
        ktstr::test_support::ShaVerdict::CheckFailed(err) => println!(
            "(cached file could not be checked: {err}; inspect the cache entry \
             or re-fetch to replace it)",
        ),
        ktstr::test_support::ShaVerdict::Mismatches => {
            println!("(cached file failed SHA-256 check; re-fetch to replace it)");
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

    let result = match ktstr.command {
        KtstrCommand::Completions { shell, binary } => {
            run_completions(shell, &binary);
            Ok(())
        }
        KtstrCommand::Test {
            kernel,
            no_perf_mode,
            args,
        } => run_test(kernel, no_perf_mode, args),
        KtstrCommand::Coverage {
            kernel,
            no_perf_mode,
            args,
        } => run_coverage(kernel, no_perf_mode, args),
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
        KtstrCommand::Stats { ref command } => run_stats(command),
        KtstrCommand::Shell {
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
        } => run_shell(kernel, topology, include_files, memory_mb, dmesg, exec),
        KtstrCommand::Kernel { command } => match command {
            KernelCommand::List { json } => cli::kernel_list(json).map_err(|e| format!("{e:#}")),
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
            } => kernel_build(version, source, git, git_ref, force, clean),
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => cli::kernel_clean(keep, force, corrupt_only).map_err(|e| format!("{e:#}"))
        },
        KtstrCommand::Model { command } => match command {
            ModelCommand::Fetch => run_model_fetch(),
            ModelCommand::Status => run_model_status(),
        },
        KtstrCommand::Cleanup { parent_cgroup } => {
            cli::cleanup(parent_cgroup).map_err(|e| format!("{e:#}"))
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
                        dir,
                    }),
                ..
            } => {
                assert_eq!(a, "a");
                assert_eq!(b, "b");
                assert_eq!(filter.as_deref(), Some("cgroup_steady"));
                assert!(threshold.is_none());
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
                        dir,
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
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "stats",
            "show-host",
            "--run",
            "my-run-id",
        ])
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
        assert!(
            rejected.is_err(),
            "stats show-host must require --run",
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
        assert!(output.contains("test"), "zsh completions missing 'test'");
        assert!(
            output.contains("coverage"),
            "zsh completions missing 'coverage'"
        );
        assert!(output.contains("shell"), "zsh completions missing 'shell'");
        assert!(
            output.contains("kernel"),
            "zsh completions missing 'kernel'"
        );
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
        } = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "show-thresholds",
            "my_test_fn",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match k.command {
            KtstrCommand::ShowThresholds { test } => {
                assert_eq!(test, "my_test_fn");
            }
            other => panic!("expected ShowThresholds, got {other:?}"),
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
        let rejected = Cargo::try_parse_from([
            "cargo",
            "ktstr",
            "show-thresholds",
            "a",
            "b",
        ]);
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
        let err = cli::show_thresholds("definitely_not_a_registered_test_xyz")
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no registered ktstr test named"),
            "error path must preserve the actionable diagnostic: {msg}",
        );
    }
}
