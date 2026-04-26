#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use ktstr::cgroup::CgroupManager;
use ktstr::cli;
use ktstr::cli::KernelCommand;
use ktstr::host_state;
use ktstr::host_state_compare;
use ktstr::runner::Runner;
use ktstr::scenario;
use ktstr::topology::TestTopology;

#[derive(Parser)]
#[command(
    name = "ktstr",
    about = "Run ktstr scheduler test scenarios on the host",
    after_help = "See also: `cargo ktstr` for cargo-integrated workflows \
                  (test, coverage, llvm-cov, verifier, stats)."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run test scenarios on the host under whatever scheduler is already active.
    ///
    /// Requires cgroup v2 mounted at `/sys/fs/cgroup` and write
    /// permission on it (typically root, or a delegated subtree).
    /// Each invocation creates `/sys/fs/cgroup/ktstr-<pid>/` and
    /// removes it on exit; `ktstr cleanup` reaps directories left
    /// behind by crashed runs.
    ///
    /// Probe modes interact as follows:
    /// - `--repro` alone: no-op (probe attachment requires a stack).
    /// - `--repro` + `--probe-stack`: attach BPF kprobes on the
    ///   listed functions for the duration of the scenario; emit a
    ///   probe-event report at exit and force the scenario to fail
    ///   so the report is preserved in CI output.
    /// - `--auto-repro`: applies only to VM-based runs (the
    ///   `#[ktstr_test]` harness invoked via `cargo nextest run` /
    ///   `cargo ktstr test`). Has no effect here -- this command runs
    ///   on the host without spawning a VM, so there is no second
    ///   boot to perform.
    Run {
        /// Scenario duration in seconds.
        #[arg(long, default_value = "20")]
        duration: u64,

        /// Workers per cgroup.
        #[arg(long, default_value = "4")]
        workers: usize,

        // Hardcoded list of valid flags below MUST mirror
        // `scenario::flags::ALL`. The drift test
        // `run_help_flags_lists_match_flags_all` (in tests/ktstr_cli.rs)
        // fails the build if the two diverge.
        /// Active flags (comma-separated). Omit for all profiles.
        /// Valid: llc, borrow, steal, rebal, reject-pin, no-ctrl.
        #[arg(long, value_delimiter = ',')]
        flags: Option<Vec<String>>,

        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output results as JSON.
        #[arg(long)]
        json: bool,

        /// Attach BPF probes during the scenario. Has no effect
        /// without `--probe-stack`; see the command help above for
        /// the full probe-mode interaction.
        #[arg(long)]
        repro: bool,

        /// Crash stack for auto-probe (file path or comma-separated function names).
        #[arg(long)]
        probe_stack: Option<String>,

        /// VM-based test re-run on scheduler crash. Applies only when
        /// the scenario is invoked through the `#[ktstr_test]`
        /// harness (`cargo ktstr test` / `cargo nextest run`); has no
        /// effect on this `ktstr run` command, which executes on the
        /// host. Documented here for parity with the test harness's
        /// flag set.
        #[arg(long)]
        auto_repro: bool,

        /// Kernel build directory (for DWARF source locations).
        #[arg(long)]
        kernel_dir: Option<String>,

        // Hardcoded list of valid work types below MUST mirror
        // `WorkType::ALL_NAMES` minus `Sequence` (requires explicit
        // phases) and `Custom` (requires a function pointer). The
        // drift test `run_help_work_type_lists_match_all_names` (in
        // tests/ktstr_cli.rs) fails the build if they diverge.
        /// Override work type for all cgroups. Case-sensitive.
        /// Valid: CpuSpin, YieldHeavy, Mixed, IoSync, Bursty, PipeIo,
        /// FutexPingPong, CachePressure, CacheYield, CachePipe,
        /// FutexFanOut, ForkExit, NiceSweep, AffinityChurn, PolicyChurn,
        /// FanOutCompute, PageFaultChurn, MutexContention.
        #[arg(long)]
        work_type: Option<String>,

        /// Disable all performance mode features (flock, pinning, RT
        /// scheduling, hugepages, NUMA mbind, KVM exit suppression).
        /// For shared runners or unprivileged containers.
        /// Also settable via KTSTR_NO_PERF_MODE env var.
        #[arg(long)]
        no_perf_mode: bool,
    },
    /// List available scenarios.
    List {
        /// Filter scenarios by name substring.
        #[arg(long)]
        filter: Option<String>,

        /// Output in JSON format for CI scripting.
        #[arg(long)]
        json: bool,
    },
    /// Show host CPU topology.
    Topo,
    /// Clean up leftover cgroups.
    ///
    /// Without `--parent-cgroup`, scans `/sys/fs/cgroup` for the
    /// default ktstr parents (`ktstr` and `ktstr-<pid>`, the paths
    /// `ktstr run` and the in-process test harness create) and
    /// rmdirs each. `ktstr-<pid>` directories whose pid is still a
    /// running ktstr or cargo-ktstr process are skipped, so a
    /// concurrent cleanup run doesn't yank an active run's cgroup.
    Cleanup {
        /// Parent cgroup path. When set, cleans only this path and
        /// leaves the parent directory in place; when omitted, scans
        /// `/sys/fs/cgroup` for the default ktstr parents
        /// (`ktstr/` and `ktstr-<pid>/`) and rmdirs each.
        #[arg(long)]
        parent_cgroup: Option<String>,
    },
    /// Manage cached kernel images.
    Kernel {
        #[command(subcommand)]
        command: KernelCommand,
    },
    /// Boot an interactive shell in a KVM virtual machine.
    ///
    /// Launches a VM with busybox and drops into a shell. Files and
    /// directories passed via -i are available at /include-files/<name>
    /// inside the guest. Directories are walked recursively, preserving
    /// structure. Dynamically-linked ELF binaries get automatic shared
    /// library resolution via ELF DT_NEEDED parsing.
    Shell {
        #[arg(long, help = ktstr::cli::KERNEL_HELP_NO_RAW)]
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
    /// Capture or compare a host-wide per-thread state snapshot.
    ///
    /// `capture` walks `/proc` at capture time and writes every
    /// visible thread's cumulative scheduling, memory, and I/O
    /// counters as zstd-compressed JSON (`.hst.zst`). Every
    /// field is cumulative-from-birth so probe attachment time
    /// does not bias the reading — a diff between two captures
    /// measures exactly the activity over the window.
    ///
    /// `compare` joins two snapshots on `(pcomm, comm)` and
    /// renders a per-metric baseline/candidate/delta table.
    HostState {
        #[command(subcommand)]
        command: HostStateCommand,
    },
    /// Generate shell completions for ktstr.
    Completions {
        /// Shell to generate completions for.
        shell: clap_complete::Shell,
        /// Binary name to register the completion under. Override
        /// when invoking ktstr through a symlink with a different
        /// name (the shell looks up completions by argv[0]).
        #[arg(long, default_value = "ktstr")]
        binary: String,
    },
    /// Enumerate every ktstr flock held on this host.
    ///
    /// Troubleshooting companion for `--cpu-cap` contention. Scans
    /// `/tmp/ktstr-llc-*.lock`, `/tmp/ktstr-cpu-*.lock`, and
    /// `{cache_root}/.locks/*.lock`, cross-referenced against
    /// `/proc/locks` via [`ktstr::cli::list_locks`] to name the
    /// holder process (PID + cmdline) for each held lock. Read-only
    /// — does NOT attempt any flock acquire.
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
}

#[derive(Subcommand)]
enum HostStateCommand {
    /// Capture a host-wide per-thread snapshot to `<output>` as
    /// zstd-compressed JSON. Walks `/proc` for every live tgid,
    /// enumerates threads, records schedstat / sched / status
    /// CSW / page faults / io bytes / affinity / cgroup / identity.
    /// Per-cgroup aggregates (cpu.stat, memory.current) are
    /// captured once per distinct path.
    Capture {
        /// Destination path (convention: `.hst.zst`).
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compare two snapshots and render a per-metric diff table.
    Compare(host_state_compare::HostStateCompareArgs),
}

/// RAII guard that cleans up an auto-generated cgroup directory on drop.
struct CgroupGuard {
    path: String,
}

impl Drop for CgroupGuard {
    fn drop(&mut self) {
        let cgroups = CgroupManager::new(&self.path);
        let _ = cgroups.cleanup_all();
        let _ = std::fs::remove_dir(&self.path);
    }
}

/// Acquire source, configure, build, and cache a kernel image.
///
/// `version` accepts `MAJOR.MINOR[.PATCH][-rcN]`, a `MAJOR.MINOR`
/// prefix (resolves to the latest patch), or `START..END` for a
/// range that expands against kernel.org's `releases.json` to every
/// `stable` / `longterm` release inside the inclusive interval. A
/// range is detected via [`KernelId::parse`] and dispatched here to
/// [`kernel_build_one`] per resolved version, sharing the
/// download / cache-lookup / build pipeline that single-version
/// invocations use. Range mode collects per-version errors as a
/// best-effort summary: a build failure on one version is reported
/// and the iteration continues to the next, so a stale endpoint
/// doesn't block the rest of the range from caching. `--git` and
/// `--source` paths bypass range expansion (clap's
/// `conflicts_with` already rejects `version + source` and
/// `version + git` combinations).
fn kernel_build(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
) -> Result<()> {
    if source.is_none()
        && git.is_none()
        && let Some(ref v) = version
    {
        use ktstr::kernel_path::KernelId;
        let id = KernelId::parse(v);
        // Validate before any I/O so an inverted range surfaces the
        // "swap the endpoints" diagnostic ahead of any download.
        id.validate()
            .map_err(|e| anyhow::anyhow!("--kernel {id}: {e}"))?;
        if let KernelId::Range { start, end } = id {
            let versions = ktstr::cli::expand_kernel_range(&start, &end, "ktstr")?;
            let total = versions.len();
            let mut failures: Vec<(String, anyhow::Error)> = Vec::new();
            for (i, ver) in versions.iter().enumerate() {
                eprintln!("ktstr: [{}/{total}] kernel build {ver}", i + 1);
                if let Err(e) =
                    kernel_build_one(Some(ver.clone()), None, None, None, force, clean, cpu_cap)
                {
                    eprintln!("ktstr: {ver}: {e:#}");
                    failures.push((ver.clone(), e));
                }
            }
            if failures.is_empty() {
                Ok(())
            } else {
                anyhow::bail!(
                    "kernel build range {start}..{end}: {failed}/{total} \
                     version(s) failed: {names}",
                    failed = failures.len(),
                    names = failures
                        .iter()
                        .map(|(v, _)| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                );
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
) -> Result<()> {
    use ktstr::cache::CacheDir;
    use ktstr::fetch;

    // Resolve the CLI --cpu-cap flag against KTSTR_CPU_CAP env and
    // the implicit "no cap" default. Conflict with
    // KTSTR_BYPASS_LLC_LOCKS=1 surfaces here so operators see the
    // parse-time error, not an opaque pipeline bail later.
    if cpu_cap.is_some()
        && std::env::var("KTSTR_BYPASS_LLC_LOCKS")
            .ok()
            .is_some_and(|v| !v.is_empty())
    {
        anyhow::bail!(
            "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset one of them. \
             --cpu-cap is a resource contract; bypass disables the contract entirely."
        );
    }
    let resolved_cap = cli::CpuCap::resolve(cpu_cap)?;

    let cache = CacheDir::new()?;

    // Temporary directory for tarball/git source extraction.
    let tmp_dir = tempfile::TempDir::new()?;

    // Acquire source.
    let client = fetch::shared_client();
    let acquired = if let Some(ref src_path) = source {
        fetch::local_source(src_path)?
    } else if let Some(ref url) = git {
        let ref_name = git_ref.as_deref().expect("clap requires --ref with --git");
        fetch::git_clone(url, ref_name, tmp_dir.path(), "ktstr")?
    } else {
        // Tarball download: explicit version, prefix, or latest stable.
        let ver = match version {
            Some(v) if fetch::is_major_minor_prefix(&v) => {
                // Major.minor prefix (e.g., "6.12") — resolve latest patch.
                fetch::fetch_version_for_prefix(client, &v, "ktstr")?
            }
            Some(v) => v,
            None => fetch::fetch_latest_stable_version(client, "ktstr")?,
        };
        // Check cache before downloading.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!("{ver}-tarball-{arch}-kc{}", ktstr::cache_key_suffix());
        if !force && let Some(entry) = cli::cache_lookup(&cache, &cache_key, "ktstr") {
            eprintln!("ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("ktstr: use --force to rebuild");
            return Ok(());
        }
        let sp = cli::Spinner::start("Downloading kernel...");
        let result = fetch::download_tarball(client, &ver, tmp_dir.path(), "ktstr");
        drop(sp);
        result?
    };

    // Check cache for --source and --git (tarball already checked above).
    if !force
        && (source.is_some() || git.is_some())
        && !acquired.is_dirty
        && let Some(entry) = cli::cache_lookup(&cache, &acquired.cache_key, "ktstr")
    {
        eprintln!("ktstr: cached kernel found: {}", entry.path.display());
        eprintln!("ktstr: use --force to rebuild");
        return Ok(());
    }

    // `--force` fail-fast pre-check: if tests are actively holding
    // the cache-entry lock, bail with the PID list instead of
    // silently waiting to stomp the in-use entry. The returned
    // guard drops at the end of this `if` scope before
    // `kernel_build_pipeline` runs; `store()` inside the pipeline
    // takes its own (now-uncontested) blocking lock. The brief
    // window between drop and re-take is acceptable for an
    // interactive `--force` operator action.
    if force {
        let _force_check = cache.try_acquire_exclusive_lock(&acquired.cache_key)?;
    }

    cli::kernel_build_pipeline(
        &acquired,
        &cache,
        "ktstr",
        clean,
        source.is_some(),
        resolved_cap,
    )?;

    Ok(())
}

fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}

fn main() -> Result<()> {
    // Restore SIGPIPE so piping ktstr output to `head` / `less` /
    // similar doesn't panic inside `print!`. Shared helper lives
    // in `cli::restore_sigpipe_default`; see that doc for the
    // rationale + SAFETY text.
    ktstr::cli::restore_sigpipe_default();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = Cli::parse();

    match args.command {
        Command::Run {
            duration,
            workers,
            flags: flag_arg,
            filter,
            json,
            repro,
            probe_stack,
            auto_repro,
            kernel_dir,
            work_type,
            no_perf_mode,
        } => {
            if no_perf_mode {
                // SAFETY: single-threaded at this point — no concurrent env readers.
                unsafe { std::env::set_var("KTSTR_NO_PERF_MODE", "1") };
            }

            let parent_cgroup = format!("/sys/fs/cgroup/ktstr-{}", std::process::id());

            // Guard cleans up auto-generated cgroups on exit (pass or fail).
            let _guard = CgroupGuard {
                path: parent_cgroup.clone(),
            };

            let active_flags = cli::resolve_flags(flag_arg)?;
            let work_type_override = cli::parse_work_type(work_type.as_deref())?;

            let config = cli::build_run_config(
                parent_cgroup,
                duration,
                workers,
                active_flags,
                repro,
                probe_stack,
                auto_repro,
                kernel_dir,
                work_type_override,
            );

            let topo = TestTopology::from_system()?;
            let runner = Runner::new(config, topo)?;

            let scenarios = scenario::all_scenarios();
            let refs = cli::filter_scenarios(&scenarios, filter.as_deref())?;

            let results = runner.run_scenarios(&refs)?;

            if json {
                println!("{}", serde_json::to_string_pretty(&results)?);
            } else {
                for r in &results {
                    let status = if r.skipped {
                        "SKIP"
                    } else if r.passed {
                        "PASS"
                    } else {
                        "FAIL"
                    };
                    println!("[{status}] {} ({:.1}s)", r.scenario_name, r.duration_s);
                    for d in &r.details {
                        println!("  {d}");
                    }
                }
                let passed = results.iter().filter(|r| r.passed && !r.skipped).count();
                let skipped = results.iter().filter(|r| r.skipped).count();
                let total = results.len();
                let failed = total - passed - skipped;
                if skipped > 0 {
                    println!("\n{passed}/{total} passed ({skipped} skipped, {failed} failed)");
                } else {
                    println!("\n{passed}/{total} passed");
                }
            }
        }

        Command::List { filter, json } => {
            let scenarios = scenario::all_scenarios();
            let filtered: Vec<&scenario::Scenario> = scenarios
                .iter()
                .filter(|s| filter.as_ref().is_none_or(|f| s.name.contains(f.as_str())))
                .collect();

            if json {
                let entries: Vec<serde_json::Value> = filtered
                    .iter()
                    .map(|s| {
                        let profiles: Vec<String> = s.profiles().iter().map(|p| p.name()).collect();
                        serde_json::json!({
                            "name": s.name,
                            "category": s.category,
                            "description": s.description,
                            "profiles": profiles,
                        })
                    })
                    .collect();
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else {
                for s in &filtered {
                    let profiles: Vec<String> = s.profiles().iter().map(|p| p.name()).collect();
                    println!(
                        "{:<30} [{:<12}] {} (profiles: {})",
                        s.name,
                        s.category,
                        s.description,
                        profiles.join(", "),
                    );
                }
                println!("\n{} scenarios", filtered.len());
            }
            // Typo-suggestion hint on an empty result with a
            // user-supplied filter. Quiet when filter is None
            // (intentional empty listing) or when the filter
            // matched at least one scenario. Routed to stderr so
            // JSON consumers that parse stdout remain unaffected —
            // the JSON array is still the sole stdout payload, the
            // hint is stderr-side operator UX only.
            if filtered.is_empty()
                && let Some(f) = filter.as_deref()
                && let Some(hint) = cli::scenario_filter_hint(f)
            {
                eprintln!(
                    "ktstr: no scenarios matched filter {f:?}.{hint} \
                     Run 'ktstr list' (no --filter) to see every scenario.",
                );
            }
        }

        Command::Topo => {
            let topo = TestTopology::from_system()?;
            println!("CPUs:       {}", topo.total_cpus());
            println!("LLCs:       {}", topo.num_llcs());
            println!("NUMA nodes: {}", topo.num_numa_nodes());
            for (i, llc) in topo.llcs().iter().enumerate() {
                println!("  LLC {} (node {}): {:?}", i, llc.numa_node(), llc.cpus(),);
            }
        }

        Command::Cleanup { parent_cgroup } => cli::cleanup(parent_cgroup)?,

        Command::Kernel { command } => match command {
            KernelCommand::List { json, range } => match range {
                Some(r) => cli::kernel_list_range_preview(json, &r)?,
                None => cli::kernel_list(json)?,
            },
            KernelCommand::Build {
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
            } => kernel_build(version, source, git, git_ref, force, clean, cpu_cap)?,
            KernelCommand::Clean {
                keep,
                force,
                corrupt_only,
            } => cli::kernel_clean(keep, force, corrupt_only)?,
        },

        Command::Shell {
            kernel,
            topology,
            include_files,
            memory_mb,
            dmesg,
            exec,
            no_perf_mode,
            cpu_cap,
        } => {
            if no_perf_mode {
                // SAFETY: single-threaded at this point — no concurrent env readers.
                unsafe { std::env::set_var("KTSTR_NO_PERF_MODE", "1") };
            }
            if let Some(cap) = cpu_cap {
                // Parse-time conflict with KTSTR_BYPASS_LLC_LOCKS — see
                // kernel_build fn for the same check on the build path.
                if std::env::var("KTSTR_BYPASS_LLC_LOCKS")
                    .ok()
                    .is_some_and(|v| !v.is_empty())
                {
                    anyhow::bail!(
                        "--cpu-cap conflicts with KTSTR_BYPASS_LLC_LOCKS=1; unset \
                         one of them. --cpu-cap is a resource contract; bypass \
                         disables the contract entirely."
                    );
                }
                // Validate the cap up front via CpuCap::new so a value
                // of 0 or a bogus KTSTR_CPU_CAP env overlay surfaces
                // at CLI-parse time, not deep inside VM build. The
                // CpuCap value itself is passed to the VMM via
                // KTSTR_CPU_CAP — KtstrVmBuilder::build re-resolves it
                // from the env there, so every resolve path (direct
                // CLI, env overlay, nested exec) agrees on precedence.
                cli::CpuCap::new(cap)?;
                // SAFETY: single-threaded at this point — no concurrent env readers.
                unsafe { std::env::set_var("KTSTR_CPU_CAP", cap.to_string()) };
            }
            cli::check_kvm()?;
            let kernel_path = cli::resolve_kernel_image(
                kernel.as_deref(),
                &cli::KernelResolvePolicy {
                    accept_raw_image: false,
                    cli_label: "ktstr",
                },
            )?;

            let (numa_nodes, llcs, cores, threads) = cli::parse_topology_string(&topology)?;

            let resolved_includes = cli::resolve_include_files(&include_files)?;

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
            )?;
        }

        Command::HostState { command } => match command {
            HostStateCommand::Capture { output } => {
                host_state::capture_to(&output)?;
                eprintln!("ktstr: wrote host-state snapshot to {}", output.display());
            }
            HostStateCommand::Compare(args) => {
                let code = host_state_compare::run_compare(&args)?;
                if code != 0 {
                    std::process::exit(code);
                }
            }
        },

        Command::Completions { shell, binary } => {
            run_completions(shell, &binary);
        }

        Command::Locks { json, watch } => cli::list_locks(json, watch)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- clap argument-parse pins: Shell --cpu-cap requires --no-perf-mode
    //
    // Mirror of the same constraint on the `cargo ktstr shell`
    // subcommand. ktstr and cargo-ktstr define separate Shell
    // variants (each binary has its own clap tree), so the
    // `requires = "no_perf_mode"` attribute must be pinned on BOTH
    // sides. A drift between the two — e.g. cargo-ktstr keeps the
    // requires, ktstr loses it — would surface here without waiting
    // for a runtime double-reservation bug.

    /// `ktstr shell --cpu-cap 4 --no-perf-mode` parses successfully.
    /// Positive-path pin for the standalone `ktstr` binary's Shell
    /// subcommand — complements the cargo-ktstr mirror test.
    #[test]
    fn parse_shell_cpu_cap_with_no_perf_mode_succeeds() {
        let parsed = Cli::try_parse_from(["ktstr", "shell", "--cpu-cap", "4", "--no-perf-mode"])
            .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::Shell {
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

    /// `ktstr shell --cpu-cap 4` without `--no-perf-mode` FAILS at
    /// parse time via the `requires = "no_perf_mode"` constraint.
    /// Negative-path pin — a regression that drops the requires
    /// attribute would allow the command to parse and then
    /// double-reserve under perf-mode at runtime.
    #[test]
    fn parse_shell_cpu_cap_without_no_perf_mode_fails() {
        // `Cli` intentionally has no Debug derive, so unwrap
        // helpers that format the Ok variant are unavailable.
        // Match on Err directly to extract the clap error.
        let msg = match Cli::try_parse_from(["ktstr", "shell", "--cpu-cap", "4"]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("--cpu-cap without --no-perf-mode must fail the parse"),
        };
        assert!(
            msg.to_ascii_lowercase().contains("no-perf-mode")
                || msg.to_ascii_lowercase().contains("no_perf_mode"),
            "clap error must name the missing --no-perf-mode flag, got: {msg}",
        );
    }

    /// `ktstr shell --no-perf-mode` without `--cpu-cap` parses
    /// successfully with `cpu_cap: None`. Pins the shape of the
    /// unset sentinel (expanded to the 30% default by the planner)
    /// — a regression that made --cpu-cap mandatory-with-no-perf-mode
    /// would break the shared-runner path that uses --no-perf-mode
    /// alone.
    #[test]
    fn parse_shell_no_perf_mode_without_cpu_cap_succeeds() {
        let parsed = Cli::try_parse_from(["ktstr", "shell", "--no-perf-mode"])
            .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::Shell {
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
