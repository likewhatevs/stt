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
    /// Memory fields (`allocated_bytes` / `deallocated_bytes`)
    /// are JEMALLOC-ONLY: they read the per-thread
    /// `tsd_s.thread_allocated` / `thread_deallocated` TSD
    /// counters via ptrace + `process_vm_readv`. Targets not
    /// linked against jemalloc, or for which the probe attach
    /// fails, land their threads at zero per the best-effort
    /// capture contract. Other allocators (glibc malloc, mimalloc)
    /// expose no equivalent per-thread counter the capture
    /// pipeline can reach.
    ///
    /// PRIVILEGE: capture pulls the jemalloc counters by briefly
    /// stopping every probed thread via `ptrace(PTRACE_SEIZE)`
    /// (an unavoidable observer effect; the stop is bounded to a
    /// handful of milliseconds per thread). `PTRACE_SEIZE` requires
    /// root, the `CAP_SYS_PTRACE` capability, OR a host configured
    /// with `kernel.yama.ptrace_scope=0`. When the attach is
    /// denied, the probe falls through to the absent-counter
    /// default of zero — the rest of the snapshot still populates.
    /// See the capture summary line for a per-snapshot tally and
    /// remediation hint.
    ///
    /// `compare` joins two snapshots on the selected grouping
    /// axis (pcomm, cgroup, or comm — see `--group-by`) and
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
    ///
    /// Memory fields are JEMALLOC-ONLY (per-thread TSD
    /// `thread_allocated` / `thread_deallocated` counters); other
    /// allocators land at zero. Capture briefly stops every probed
    /// thread via ptrace (observer effect, bounded ms per thread)
    /// and therefore needs root, `CAP_SYS_PTRACE`, or
    /// `kernel.yama.ptrace_scope=0` to pull the jemalloc counters.
    /// Without the privilege, the rest of the snapshot still
    /// populates and the jemalloc fields fall through to zero.
    ///
    /// PROBE SUMMARY LINE: capture emits a single info-level
    /// `tracing` event per snapshot summarising probe outcomes. The
    /// default tracing config writes to stderr, so the line lands
    /// alongside any other capture-time diagnostics; `RUST_LOG=warn`
    /// suppresses it (alongside other info), `RUST_LOG=debug`
    /// reveals the per-tgid attach events that feed it.
    ///
    /// Format depends on whether any per-tgid failures landed:
    ///
    /// 1. No failures —
    ///    `host-state probe: <N> tgids walked, <N> jemalloc detected,
    ///    <N> probed OK, 0 failed`
    /// 2. Failures, ptrace not dominant —
    ///    `... <N> failed (dominant: <tag>)`
    /// 3. Failures, ptrace dominates (≥50% of failures attributable
    ///    to `ptrace-seize` or `ptrace-interrupt`) —
    ///    `... <N> failed (dominant: <tag>; hint: re-run as root, or
    ///    sudo setcap cap_sys_ptrace+eip $(which ktstr), or set
    ///    kernel.yama.ptrace_scope=0)`
    ///
    /// "tgids walked" counts every live tgid the procfs walk reached
    /// (the snapshot's full denominator); "jemalloc detected" counts
    /// the subset that successfully attached the TSD probe; "probed
    /// OK" counts per-thread reads that returned a counter pair;
    /// "failed" counts attach-or-probe failures whose tag is
    /// classified ACTIONABLE — see filter rule below — so a busy
    /// host typically reports `failed` ≪ (`tgids walked` − `jemalloc
    /// detected`) because the bulk of the system is not jemalloc-
    /// linked and that case does not count as a failure.
    ///
    /// ACTIONABLE TAGS (operator-facing causes for the
    /// `(dominant: <tag>)` clause):
    ///
    /// Pre-thread (attach-time) failures:
    /// - `pid-missing` — `/proc/<pid>` does not exist or is
    ///   unreachable; possible race-with-exit between enumeration
    ///   and attach.
    /// - `maps-read-failure` — `/proc/<pid>/maps` could not be read;
    ///   possible race-with-exit, uid mismatch, or sandbox/namespace
    ///   restriction.
    /// - `jemalloc-in-dso` — target's jemalloc symbol resides in a
    ///   shared object (e.g. `libjemalloc.so`). Static-TLS resolution
    ///   only reaches the main executable; DSO-linked jemalloc needs
    ///   DTV walking the engine does not implement yet.
    /// - `arch-mismatch` — target ELF arch differs from the probe
    ///   binary's. ptrace is same-arch only; rebuild the probe for
    ///   the target's arch (or filter the offending tgids).
    /// - `dwarf-parse-failure` — target binary is stripped without
    ///   reachable external debuginfo, debuglink CRC mismatched, or
    ///   the `tsd_s` struct / its member fields are absent from the
    ///   DWARF. Install matching `-debuginfo` / `-dbg` packages.
    /// - `worker-panic` — the per-tgid attach worker panicked
    ///   (caught via `catch_unwind` so the snapshot still completes).
    ///   Indicates an unexpected fault inside the parallel attach
    ///   pipeline — fd exhaustion / OOM during the ELF parse or
    ///   DWARF walk are the canonical triggers, but any panic-on-bug
    ///   under `attach_jemalloc_at` lands here. Investigate the
    ///   `tracing::error!` log for the offending tgid; the recorded
    ///   payload string identifies the panic site.
    ///
    /// Per-thread (probe-time) failures:
    /// - `ptrace-seize` — `PTRACE_SEIZE` rejected by the kernel.
    ///   ESRCH is benign (thread exited mid-snapshot); EPERM is the
    ///   privilege case the EPERM hint targets; EBUSY means another
    ///   tracer is attached.
    /// - `ptrace-interrupt` — `PTRACE_INTERRUPT` failed after a
    ///   successful seize. Race causes — privilege is already
    ///   established by the preceding successful seize. ESRCH (target
    ///   died between seize and interrupt) is the dominant case.
    /// - `waitpid` — `waitpid` after `PTRACE_INTERRUPT` returned an
    ///   unexpected status, a syscall error, or the post-interrupt
    ///   stop did not arrive within 250ms.
    /// - `get-regset` — `PTRACE_GETREGSET` failed when reading the
    ///   thread pointer. Race causes — privilege established. ESRCH
    ///   (target died mid-probe) is the dominant case.
    /// - `process-vm-readv` — `process_vm_readv` failed or returned
    ///   a short read at the resolved TSD address. Investigate
    ///   target memory protection or a corrupted TSD.
    /// - `tls-arithmetic` — TLS-address arithmetic over- or
    ///   underflowed. Indicates a malformed TLS image — file a bug
    ///   with the target binary's identity.
    ///
    /// FILTER RULE — two AttachError tags are NOT actionable and are
    /// suppressed from both the per-tgid `tracing::warn!` log and
    /// the `(dominant: <tag>)` clause:
    ///
    /// - `jemalloc-not-found` — target is not jemalloc-linked, OR
    ///   links a jemalloc build whose symbol-name prefix is not in
    ///   the recognized set. The dominant outcome on most system
    ///   processes.
    /// - `readlink-failure` — `readlink(/proc/<pid>/exe)` failed.
    ///   Typically a race-with-exit between enumeration and the
    ///   per-tgid attach, OR permission denied under the active
    ///   ptrace policy.
    ///
    /// Filtered tags do NOT contribute to the `failed` counter, so a
    /// snapshot whose only "failures" are these two will emit the
    /// no-failure variant of the line (no `(dominant: ...)` clause).
    /// If the operator sees no `(dominant: ...)` clause despite many
    /// non-jemalloc tgids on the host, that is correct: the noise
    /// floor was filtered.
    Capture {
        /// Destination path (convention: `.hst.zst`).
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compare two snapshots and render a per-metric diff table.
    Compare(host_state_compare::HostStateCompareArgs),
    /// Render one snapshot as a per-group, per-metric table.
    ///
    /// Same grouping axis as `compare` (`pcomm` default, `cgroup`,
    /// `comm`, `comm-exact`) and the same per-metric aggregation
    /// rules from [`host_state_compare::HOST_STATE_METRICS`], but
    /// with no baseline/candidate split — every metric renders one
    /// aggregated value per group rather than a delta against a
    /// reference snapshot. Useful for inspecting a capture in
    /// isolation: spot-check a process's per-thread CSW totals,
    /// audit the cgroup distribution, find which thread-name
    /// pattern has the highest run-time without comparing two
    /// runs.
    ///
    /// Pattern normalization (`comm` token-based clustering and
    /// `cgroup` Layer-1/2/3 path tightening) applies the same way
    /// it does in `compare`; `--no-thread-normalize` and
    /// `--no-cg-normalize` disable each axis. `--cgroup-flatten`
    /// glob patterns also apply unchanged.
    Show(HostStateShowArgs),
}

/// Arguments for the `ktstr host-state show` subcommand. Field
/// shapes mirror [`host_state_compare::HostStateCompareArgs`] so a
/// caller switching from compare to show keeps the same flag
/// vocabulary; the only structural difference is the single
/// positional path (one snapshot rather than baseline+candidate)
/// and the absence of any delta/percentage flags — show renders
/// one value per (group, metric) cell, no diff math.
#[derive(Debug, clap::Args)]
pub struct HostStateShowArgs {
    /// Snapshot path (`.hst.zst`) from `ktstr host-state capture -o`.
    pub snapshot: std::path::PathBuf,
    /// Grouping key. Same semantics as
    /// `ktstr host-state compare --group-by`. `pcomm` (default)
    /// aggregates per process name; `cgroup` per cgroup path;
    /// `comm` aggregates threads by NAME PATTERN under the
    /// token-based normalizer; `comm-exact` is a synonym for
    /// `comm --no-thread-normalize`.
    #[arg(long, value_enum, default_value_t = host_state_compare::GroupBy::Pcomm)]
    pub group_by: host_state_compare::GroupBy,
    /// Glob patterns that collapse dynamic cgroup path segments —
    /// same shape as
    /// `ktstr host-state compare --cgroup-flatten`. Repeatable.
    #[arg(long)]
    pub cgroup_flatten: Vec<String>,
    /// Disable token-based pattern normalization for the thread
    /// axis (`--group-by comm`). Threads group by literal `comm`.
    /// Has no effect under any other grouping.
    #[arg(long)]
    pub no_thread_normalize: bool,
    /// Disable token-based pattern normalization for the cgroup
    /// axis (`--group-by cgroup`). Cgroup paths group by literal
    /// post-`--cgroup-flatten` path. Has no effect under any
    /// other grouping.
    #[arg(long)]
    pub no_cg_normalize: bool,
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

/// Entry point for `ktstr host-state show <snapshot>`. Loads one
/// snapshot, builds groups along the requested axis, aggregates
/// every metric per
/// [`host_state_compare::HOST_STATE_METRICS`], and renders the
/// result as a `(group_key, threads, metric, value)` table on
/// stdout. Exits non-zero only on I/O / parse errors; the
/// rendered table is always considered successful output.
///
/// Reuses the grouping pipeline from
/// [`host_state_compare`] so pattern normalization (`comm` Layer-2
/// token clustering, `cgroup` Layer-1/2/3 tightening) and
/// `--cgroup-flatten` glob handling stay identical to the compare
/// path.
fn run_show(args: &HostStateShowArgs) -> Result<i32> {
    use anyhow::Context;
    let snap = ktstr::host_state::HostStateSnapshot::load(&args.snapshot)
        .with_context(|| format!("load snapshot {}", args.snapshot.display()))?;
    let mut out = String::new();
    // Infallible: writing into a String cannot fail.
    let _ = write_show(
        &mut out,
        &snap,
        args.group_by,
        &args.cgroup_flatten,
        args.no_thread_normalize,
        args.no_cg_normalize,
    );
    print!("{out}");
    Ok(0)
}

/// Render the per-group / per-metric show table into `w`. Mirrors
/// [`host_state_compare::write_diff`]'s shape — the formatter
/// layer is split from the I/O wrapper so unit tests can drive
/// rendering into a `String` buffer without shelling through
/// stdout. Write errors propagate as [`std::fmt::Error`]; callers
/// that write into an infallible sink (`String`) can ignore.
fn write_show<W: std::fmt::Write>(
    w: &mut W,
    snap: &ktstr::host_state::HostStateSnapshot,
    group_by: host_state_compare::GroupBy,
    cgroup_flatten: &[String],
    no_thread_normalize: bool,
    no_cg_normalize: bool,
) -> std::fmt::Result {
    let flatten = host_state_compare::compile_flatten_patterns(cgroup_flatten);
    // Single-snapshot cgroup key map. compare()'s
    // `build_cgroup_key_map` walks the union of paths from two
    // snapshots; for show we feed the same snap on both arguments
    // so the union iteration produces the same key set as a
    // single-snap pass (the inner BTreeSet of paths dedups the
    // duplicate insertions). Skipped under `--no-cg-normalize`
    // and under any grouping other than Cgroup.
    let cgroup_key_map = if group_by == host_state_compare::GroupBy::Cgroup && !no_cg_normalize {
        Some(host_state_compare::build_cgroup_key_map(
            snap, snap, &flatten,
        ))
    } else {
        None
    };
    // Pattern counts: `build_groups` falls back to a per-snapshot
    // local count when None, which is exactly the right behavior
    // for a single-snapshot view (compare()'s union-over-baseline-
    // and-candidate gate doesn't apply when there's only one
    // snapshot to inspect).
    let groups = host_state_compare::build_groups(
        snap,
        group_by,
        &flatten,
        None,
        cgroup_key_map.as_ref(),
        no_thread_normalize,
    );

    let group_header = match group_by {
        host_state_compare::GroupBy::Pcomm => "pcomm",
        host_state_compare::GroupBy::Cgroup => "cgroup",
        host_state_compare::GroupBy::Comm => "comm-pattern",
        host_state_compare::GroupBy::CommExact => "comm",
    };

    let mut table = cli::new_table();
    table.set_header(vec![group_header, "threads", "metric", "value"]);

    // Stable iteration order: BTreeMap on `groups` already orders
    // by key; per-group rows iterate `HOST_STATE_METRICS` so the
    // metric column also lands in registry order. No delta sort
    // because there is no delta to sort by — show renders
    // absolute aggregates, not differences.
    for (key, group) in &groups {
        // Display key: pattern grouping under Comm uses grex to
        // turn the join-key skeleton into a regex label; every
        // other grouping renders the join key directly.
        let display_key = if group_by == host_state_compare::GroupBy::Comm && !no_thread_normalize {
            host_state_compare::pattern_display_label(key, &group.member_comms)
        } else {
            key.clone()
        };
        for metric in host_state_compare::HOST_STATE_METRICS {
            let Some(agg) = group.metrics.get(metric.name) else {
                continue;
            };
            table.add_row(vec![
                display_key.clone(),
                group.thread_count.to_string(),
                metric.name.to_string(),
                host_state_compare::format_value_cell(agg, metric.unit),
            ]);
        }
    }
    writeln!(w, "{table}")?;

    // Cgroup grouping carries cgroup_stats enrichment alongside
    // the per-thread aggregates. Render a second table when
    // present so the show output mirrors compare's two-table
    // layout for `--group-by cgroup`.
    if group_by == host_state_compare::GroupBy::Cgroup && !snap.cgroup_stats.is_empty() {
        let stats = host_state_compare::flatten_cgroup_stats(
            &snap.cgroup_stats,
            &flatten,
            cgroup_key_map.as_ref(),
        );
        if !stats.is_empty() {
            writeln!(w)?;
            let mut ct = cli::new_table();
            ct.set_header(vec![
                "cgroup",
                "cpu_usage_usec",
                "nr_throttled",
                "throttled_usec",
                "memory_current",
            ]);
            for (key, s) in &stats {
                ct.add_row(vec![
                    key.clone(),
                    s.cpu_usage_usec.to_string(),
                    s.nr_throttled.to_string(),
                    s.throttled_usec.to_string(),
                    s.memory_current.to_string(),
                ]);
            }
            writeln!(w, "{ct}")?;
        }
    }

    Ok(())
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
            HostStateCommand::Show(args) => {
                let code = run_show(&args)?;
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

    // -- host-state show CLI tests
    //
    // Pin clap-parse shape on the `Show` variant + the rendered
    // output of `write_show` against a synthetic snapshot. The
    // CLI-parse tests guard against argument-vocabulary drift
    // (matching the pattern compare's args use) and the
    // write_show tests guard the rendered table against
    // regressions in column count, header, and group-axis routing.

    /// `ktstr host-state show <path>` with no flags must parse
    /// into a Show variant carrying the positional path and
    /// default GroupBy::Pcomm. Positive-path pin for argv shape.
    #[test]
    fn parse_host_state_show_positional_only_succeeds() {
        let parsed = Cli::try_parse_from(["ktstr", "host-state", "show", "/tmp/snap.hst.zst"])
            .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::HostState {
                command: HostStateCommand::Show(args),
            } => {
                assert_eq!(args.snapshot, std::path::PathBuf::from("/tmp/snap.hst.zst"));
                assert_eq!(args.group_by, host_state_compare::GroupBy::Pcomm);
                assert!(args.cgroup_flatten.is_empty());
                assert!(!args.no_thread_normalize);
                assert!(!args.no_cg_normalize);
            }
            _ => panic!("expected HostState/Show"),
        }
    }

    /// `--group-by`, `--cgroup-flatten`, `--no-thread-normalize`,
    /// and `--no-cg-normalize` propagate from clap into the
    /// `HostStateShowArgs` struct unchanged. Pins the parse shape
    /// for every flag the show subcommand surfaces.
    #[test]
    fn parse_host_state_show_with_every_flag_succeeds() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "host-state",
            "show",
            "/tmp/snap.hst.zst",
            "--group-by",
            "comm",
            "--cgroup-flatten",
            "/kubepods/*/workload",
            "--no-thread-normalize",
            "--no-cg-normalize",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::HostState {
                command: HostStateCommand::Show(args),
            } => {
                assert_eq!(args.group_by, host_state_compare::GroupBy::Comm);
                assert_eq!(
                    args.cgroup_flatten,
                    vec!["/kubepods/*/workload".to_string()]
                );
                assert!(args.no_thread_normalize);
                assert!(args.no_cg_normalize);
            }
            _ => panic!("expected HostState/Show"),
        }
    }

    /// `write_show` on a two-thread synthetic snapshot under
    /// `GroupBy::Pcomm` renders one row per metric per group with
    /// header `pcomm | threads | metric | value`. Pins the column
    /// vocabulary, the routing of GroupBy → header, and that
    /// every row's "threads" column matches the bucket size. A
    /// regression that swapped the Sum/Mode columns or dropped a
    /// metric would surface here.
    #[test]
    fn write_show_renders_pcomm_grouping_with_expected_columns() {
        let mut snap = ktstr::host_state::HostStateSnapshot::default();
        // Two threads under the same pcomm so the bucket has
        // size 2 — verifies the threads column reflects bucket
        // size, not snapshot total. Use a unitless cumulative
        // counter (`nr_wakeups`) for the value-cell pin so the
        // assertion does not need to model `auto_scale`'s ladder
        // step-up: with values 1000 + 2000 = 3000, an `"ns"`
        // metric would render `3.000µs` (auto-scaled), but
        // `nr_wakeups` is unitless and the empty-unit ladder
        // (""→K→M→G) only steps up at 1e3, so `3000` (>= 1e3)
        // would also scale to `3.000K`. Pick small values
        // (1+2=3) so no scaling fires regardless of unit.
        let mut t1 = ktstr::host_state::ThreadState::default();
        t1.pcomm = "worker-proc".to_string();
        t1.comm = "worker-0".to_string();
        t1.nr_wakeups = 1;
        let mut t2 = ktstr::host_state::ThreadState::default();
        t2.pcomm = "worker-proc".to_string();
        t2.comm = "worker-1".to_string();
        t2.nr_wakeups = 2;
        snap.threads.push(t1);
        snap.threads.push(t2);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            host_state_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
        )
        .expect("write_show into String must not fail");

        assert!(
            out.contains("pcomm"),
            "header must include `pcomm` for GroupBy::Pcomm, got: {out}",
        );
        assert!(
            out.contains("threads"),
            "header must include `threads` column, got: {out}",
        );
        assert!(
            out.contains("metric"),
            "header must include `metric` column, got: {out}",
        );
        assert!(
            out.contains("value"),
            "header must include `value` column, got: {out}",
        );
        assert!(
            out.contains("worker-proc"),
            "group key must surface in the rendered table, got: {out}",
        );
        assert!(
            out.contains("nr_wakeups"),
            "Sum metric `nr_wakeups` must render a row, got: {out}",
        );
        // Sum across the bucket: 1 + 2 = 3. `nr_wakeups` is
        // unitless and below the 1e3 ladder step, so
        // `format_value_cell` renders the integer verbatim.
        assert!(
            out.contains(" 3 "),
            "summed nr_wakeups (1+2) must surface as 3 in a value cell, got: {out}",
        );
    }

    /// Header switches based on the GroupBy axis — `comm-pattern`
    /// for `GroupBy::Comm`, `comm` for `GroupBy::CommExact`,
    /// `cgroup` for `GroupBy::Cgroup`, `pcomm` for the default.
    /// Pins the routing in `write_show`'s match arm against
    /// vocabulary drift between this and `host_state_compare`.
    #[test]
    fn write_show_header_switches_on_group_by() {
        let snap = ktstr::host_state::HostStateSnapshot::default();
        for (axis, expected_header) in [
            (host_state_compare::GroupBy::Pcomm, "pcomm"),
            (host_state_compare::GroupBy::Cgroup, "cgroup"),
            (host_state_compare::GroupBy::Comm, "comm-pattern"),
            (host_state_compare::GroupBy::CommExact, "comm"),
        ] {
            let mut out = String::new();
            write_show(&mut out, &snap, axis, &[], false, false)
                .expect("write_show into String must not fail");
            assert!(
                out.contains(expected_header),
                "header for {axis:?} must contain `{expected_header}`, got: {out}",
            );
        }
    }

    /// Empty snapshot renders only the header row — `write_show`
    /// must not panic on the zero-thread path. Pins that the
    /// no-data case is well-behaved (compare's no-rows path is
    /// driven by both-snapshot mismatch; show's only no-data
    /// path is the single-snap one).
    #[test]
    fn write_show_empty_snapshot_renders_header_only() {
        let snap = ktstr::host_state::HostStateSnapshot::default();
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            host_state_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
        )
        .expect("write_show into String must not fail on empty snapshot");
        assert!(
            out.contains("pcomm"),
            "empty snapshot must still emit the header, got: {out}",
        );
        // Run-time metric does not appear because no thread
        // contributed it — the `for metric in HOST_STATE_METRICS`
        // loop emits no rows when `groups` is empty.
        assert!(
            !out.contains("run_time_ns"),
            "no thread → no metric row; got run_time_ns surfaced: {out}",
        );
    }
}
