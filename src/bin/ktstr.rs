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
    /// Multi-key sort spec for the rendered groups. Format mirrors
    /// `ktstr host-state compare --sort-by`:
    /// `metric1[:dir1],metric2[:dir2],...` where each `metric` is
    /// one of the registered metric names and `dir` is `asc` or
    /// `desc` (default `desc`). Show ranks groups by the tuple
    /// of the named metric's *absolute aggregated value* (not a
    /// delta — there is only one snapshot to inspect), descending
    /// by default; rows within a group keep registry order. Empty
    /// (the default) keeps the alphabetical group-key iteration.
    /// Example: `--sort-by run_time_ns,wait_sum:asc`.
    #[arg(long, default_value = "")]
    pub sort_by: String,
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
    // Parse `--sort-by` BEFORE the snapshot load so an operator
    // typo fails fast without paying for disk I/O. Mirrors
    // run_compare's ordering for the same reason.
    let sort_by = host_state_compare::parse_sort_by(&args.sort_by)
        .with_context(|| format!("parse --sort-by {:?}", args.sort_by))?;
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
        &sort_by,
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
    sort_by: &[host_state_compare::SortKey],
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

    // Iteration order: when `sort_by` is empty, fall through the
    // BTreeMap by-key order (alphabetical group key, registry order
    // metrics); when non-empty, rank groups by the tuple of named
    // metrics' *absolute aggregated values* (no deltas — show has
    // one snapshot, not two). Mirrors compare's
    // `sort_diff_rows_by_keys` shape (per-key direction, lex order
    // on the tuple, deterministic group_key tie-break) but on
    // `Aggregated::numeric()` rather than `DiffRow.delta`. Within a
    // group, metrics still iterate `HOST_STATE_METRICS` so the
    // metric column lands in registry order regardless of sort.
    let group_order: Vec<&String> = if sort_by.is_empty() {
        groups.keys().collect()
    } else {
        let mut keys: Vec<&String> = groups.keys().collect();
        // Per-group sort tuple: missing values (a metric absent from
        // this group's `metrics` map, or one whose `Aggregated`
        // returns `None` from `numeric()` — categorical Mode etc.)
        // sink to the bottom for the requested direction, matching
        // `sort_diff_rows_by_keys`'s NEG_INFINITY/INFINITY handling.
        let group_tuple = |group_key: &str| -> Vec<f64> {
            let group = groups.get(group_key);
            sort_by
                .iter()
                .map(|k| {
                    group
                        .and_then(|g| g.metrics.get(k.metric))
                        .and_then(|a| a.numeric())
                        .unwrap_or(if k.descending {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        })
                })
                .collect()
        };
        keys.sort_by(|a, b| {
            let ta = group_tuple(a);
            let tb = group_tuple(b);
            for (i, key) in sort_by.iter().enumerate() {
                let (va, vb) = (ta[i], tb[i]);
                let ord = if key.descending {
                    vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                };
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            // Final tie-break: ascending group_key for determinism.
            a.cmp(b)
        });
        keys
    };

    for key in group_order {
        let group = &groups[key];
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
            // Route every scalar through `format_scaled_u64` so
            // the same auto-scale ladder that compare's enrichment
            // table uses applies here too — `7.500GiB` instead of
            // `8053063680`, `1.235s` instead of `1234567` µs.
            // Compare's table renders a baseline→candidate→delta
            // triple via `cgroup_cell`; show has a single snapshot
            // so each cell stands alone — `format_scaled_u64`
            // gives just the scaled value with no `→` arrow and
            // no `(+0…)` zero-delta tail. Units mirror compare's
            // call sites:
            //   cpu_usage_usec, throttled_usec → "µs"
            //   memory_current                  → "B"
            //   nr_throttled                    → "" (unitless count)
            for (key, s) in &stats {
                ct.add_row(vec![
                    key.clone(),
                    host_state_compare::format_scaled_u64(s.cpu_usage_usec, "µs"),
                    host_state_compare::format_scaled_u64(s.nr_throttled, ""),
                    host_state_compare::format_scaled_u64(s.throttled_usec, "µs"),
                    host_state_compare::format_scaled_u64(s.memory_current, "B"),
                ]);
            }
            writeln!(w, "{ct}")?;

            // Per-cgroup PSI sub-tables — one per resource.
            // Q8 ruling: per-resource sub-tables, all fields
            // rendered (no --verbose gate). Each resource shows
            // a `some` row + a `full` row with `avg10/avg60/avg300/total`
            // columns. avg fields are stored as centi-percent
            // (0..=10099, see [`PsiHalf`] doc for the kernel's
            // EWMA rounding ceiling); render as `N.NN%` for
            // human-friendliness. total is microseconds; the
            // auto_scale "µs" ladder applies via
            // `format_scaled_u64`. Per-resource zero-suppression
            // mirrors the compare-side write_diff path: skip a
            // resource sub-table when no cgroup in the bucket
            // has any non-zero data for it.
            for (resource_name, accessor) in psi_resources() {
                let any_data = stats.values().any(|s| {
                    let r = accessor(&s.psi);
                    psi_resource_has_data(&r)
                });
                if !any_data {
                    continue;
                }
                writeln!(w)?;
                writeln!(w, "## Pressure / {resource_name}")?;
                let mut pt = cli::new_table();
                pt.set_header(vec!["cgroup", "row", "avg10", "avg60", "avg300", "total"]);
                for (key, s) in &stats {
                    let r = accessor(&s.psi);
                    pt.add_row(vec![
                        key.clone(),
                        "some".into(),
                        format_psi_avg(r.some.avg10),
                        format_psi_avg(r.some.avg60),
                        format_psi_avg(r.some.avg300),
                        host_state_compare::format_scaled_u64(r.some.total_usec, "µs"),
                    ]);
                    pt.add_row(vec![
                        key.clone(),
                        "full".into(),
                        format_psi_avg(r.full.avg10),
                        format_psi_avg(r.full.avg60),
                        format_psi_avg(r.full.avg300),
                        host_state_compare::format_scaled_u64(r.full.total_usec, "µs"),
                    ]);
                }
                writeln!(w, "{pt}")?;
            }
        }
    }

    // Host-level PSI — surface above the per-thread table when
    // any resource has nonzero data. Renders as four per-resource
    // sub-tables (cpu / memory / io / irq) with a `some`+`full`
    // row each, matching the per-cgroup layout above.
    if host_psi_has_data(&snap.psi) {
        for (resource_name, accessor) in psi_resources() {
            let r = accessor(&snap.psi);
            if !psi_resource_has_data(&r) {
                continue;
            }
            writeln!(w)?;
            writeln!(w, "## Host pressure / {resource_name}")?;
            let mut pt = cli::new_table();
            pt.set_header(vec!["row", "avg10", "avg60", "avg300", "total"]);
            pt.add_row(vec![
                "some".into(),
                format_psi_avg(r.some.avg10),
                format_psi_avg(r.some.avg60),
                format_psi_avg(r.some.avg300),
                host_state_compare::format_scaled_u64(r.some.total_usec, "µs"),
            ]);
            pt.add_row(vec![
                "full".into(),
                format_psi_avg(r.full.avg10),
                format_psi_avg(r.full.avg60),
                format_psi_avg(r.full.avg300),
                host_state_compare::format_scaled_u64(r.full.total_usec, "µs"),
            ]);
            writeln!(w, "{pt}")?;
        }
    }

    Ok(())
}

/// One entry in the [`psi_resources`] table — a display name
/// paired with the accessor that pulls one
/// [`host_state::PsiResource`] out of a [`host_state::Psi`]
/// bundle.
type PsiAccessor = (
    &'static str,
    fn(&host_state::Psi) -> host_state::PsiResource,
);

/// Returns the four PSI resource accessors paired with their
/// display names. Centralizing the resource list keeps the
/// host-level and per-cgroup rendering paths in lockstep — adding
/// a fifth resource (e.g. a future `net.pressure`) means one
/// edit here, not four scattered through the renderers.
fn psi_resources() -> [PsiAccessor; 4] {
    [
        ("cpu", |p| p.cpu),
        ("memory", |p| p.memory),
        ("io", |p| p.io),
        ("irq", |p| p.irq),
    ]
}

/// Render a centi-percent PSI average as `N.NN%`. The kernel
/// emits `LOAD_INT.LOAD_FRAC` at `kernel/sched/psi.c:1284` with
/// 2-decimal-digit precision — preserve that on display.
fn format_psi_avg(centi_percent: u16) -> String {
    let int = centi_percent / 100;
    let frac = centi_percent % 100;
    format!("{int}.{frac:02}%")
}

/// Returns true when any host-level PSI resource has a non-zero
/// avg or total reading. Used to suppress the host pressure
/// section when the kernel returned all zeros (CONFIG_PSI off,
/// PSI not yet warmed up, or synthetic test fixture).
fn host_psi_has_data(psi: &host_state::Psi) -> bool {
    [psi.cpu, psi.memory, psi.io, psi.irq]
        .iter()
        .any(psi_resource_has_data)
}

fn psi_resource_has_data(r: &host_state::PsiResource) -> bool {
    let h = |h: &host_state::PsiHalf| {
        h.avg10 != 0 || h.avg60 != 0 || h.avg300 != 0 || h.total_usec != 0
    };
    h(&r.some) || h(&r.full)
}

#[cfg(test)]
mod psi_show_tests {
    //! Tests for the show-side PSI helpers
    //! (`psi_resources`, `format_psi_avg`,
    //! `host_psi_has_data`, `psi_resource_has_data`). The
    //! integration tests for the full show pipeline already
    //! exercise the rendering happy path; these unit tests pin
    //! the helper boundaries (zero-suppression edges,
    //! centi-percent → display rounding) directly so a
    //! regression in either function surfaces with a
    //! pinpointed assertion rather than a rendered-output
    //! diff.
    use super::*;
    use ktstr::host_state::{Psi, PsiHalf, PsiResource};

    #[test]
    fn format_psi_avg_renders_centi_percent_with_two_decimal_digits() {
        // Lossless N.NN% display: integer + 2-digit fraction.
        assert_eq!(format_psi_avg(0), "0.00%");
        assert_eq!(format_psi_avg(1), "0.01%");
        assert_eq!(format_psi_avg(50), "0.50%");
        assert_eq!(format_psi_avg(1859), "18.59%");
        assert_eq!(format_psi_avg(10000), "100.00%");
        // Kernel EWMA rounding ceiling per
        // include/linux/sched/loadavg.h:35; render the boundary
        // verbatim rather than clamping.
        assert_eq!(format_psi_avg(10099), "100.99%");
    }

    #[test]
    fn psi_resources_lists_four_in_canonical_order() {
        let names: Vec<&str> = psi_resources().iter().map(|(n, _)| *n).collect();
        assert_eq!(names, vec!["cpu", "memory", "io", "irq"]);
    }

    /// Build a `PsiResource` with one centi-percent sentinel set
    /// on either the `some` or `full` half. The bin crate is a
    /// distinct compilation unit from the lib, so the
    /// `#[non_exhaustive]` markers on `Psi` / `PsiResource` /
    /// `PsiHalf` block struct-literal construction here — same
    /// constraint that drives `tests/common/host_state.rs`'s
    /// Default + per-field assignment pattern.
    fn psi_resource_with_some_avg10(v: u16) -> PsiResource {
        let mut half = PsiHalf::default();
        half.avg10 = v;
        let mut r = PsiResource::default();
        r.some = half;
        r
    }

    fn psi_resource_with_full_avg10(v: u16) -> PsiResource {
        let mut half = PsiHalf::default();
        half.avg10 = v;
        let mut r = PsiResource::default();
        r.full = half;
        r
    }

    #[test]
    fn psi_resources_accessors_route_to_correct_field() {
        // Distinct sentinel per resource so a swapped accessor
        // surfaces as a wrong-resource value here.
        let mut psi = Psi::default();
        psi.cpu = psi_resource_with_some_avg10(1);
        psi.memory = psi_resource_with_some_avg10(2);
        psi.io = psi_resource_with_some_avg10(3);
        psi.irq = psi_resource_with_full_avg10(4);
        let accessors = psi_resources();
        assert_eq!(accessors[0].1(&psi).some.avg10, 1, "cpu accessor");
        assert_eq!(accessors[1].1(&psi).some.avg10, 2, "memory accessor");
        assert_eq!(accessors[2].1(&psi).some.avg10, 3, "io accessor");
        assert_eq!(accessors[3].1(&psi).full.avg10, 4, "irq accessor");
    }

    #[test]
    fn host_psi_has_data_returns_false_for_all_zero_bundle() {
        assert!(!host_psi_has_data(&Psi::default()));
    }

    #[test]
    fn host_psi_has_data_returns_true_for_any_nonzero_field() {
        // Each per-half field flagged independently so a
        // regression that only checked one half (or one resource)
        // surfaces here.
        for resource_idx in 0..4 {
            for is_full in [false, true] {
                let mut psi = Psi::default();
                let target_resource = match resource_idx {
                    0 => &mut psi.cpu,
                    1 => &mut psi.memory,
                    2 => &mut psi.io,
                    _ => &mut psi.irq,
                };
                let half = if is_full {
                    &mut target_resource.full
                } else {
                    &mut target_resource.some
                };
                half.total_usec = 1;
                assert!(
                    host_psi_has_data(&psi),
                    "resource_idx={resource_idx}, is_full={is_full} should be detected"
                );
            }
        }
    }
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
                // `--sort-by` defaults to the empty string —
                // empty-spec sentinel for "fall through to
                // alphabetical iteration".
                assert!(args.sort_by.is_empty());
            }
            _ => panic!("expected HostState/Show"),
        }
    }

    /// `--group-by`, `--cgroup-flatten`, `--no-thread-normalize`,
    /// `--no-cg-normalize`, and `--sort-by` propagate from clap
    /// into the `HostStateShowArgs` struct unchanged. Pins the
    /// parse shape for every flag the show subcommand surfaces.
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
            "--sort-by",
            "run_time_ns:desc,wait_sum:asc",
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
                // `--sort-by` comes through verbatim — clap stores
                // the raw spec; `parse_sort_by` is the parser layer
                // and runs in `run_show`, not at clap parse time.
                assert_eq!(args.sort_by, "run_time_ns:desc,wait_sum:asc");
            }
            _ => panic!("expected HostState/Show"),
        }
    }

    /// `ktstr host-state show <path> --sort-by run_time_ns` parses
    /// successfully with the spec stored verbatim on
    /// `HostStateShowArgs::sort_by`. Single-key, default-direction
    /// pin — the unmarked form ("--sort-by metric" without a `:dir`
    /// suffix) routes through `parse_sort_by`'s `None` arm in
    /// `run_show` and ranks descending.
    #[test]
    fn parse_host_state_show_sort_by_single_key_succeeds() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "host-state",
            "show",
            "/tmp/snap.hst.zst",
            "--sort-by",
            "run_time_ns",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::HostState {
                command: HostStateCommand::Show(args),
            } => {
                assert_eq!(args.sort_by, "run_time_ns");
            }
            _ => panic!("expected HostState/Show"),
        }
    }

    // -- host-state compare CLI --sort-by clap-parse pins
    //
    // Mirror of the show-side parse tests above for the compare
    // subcommand. Pins that the `--sort-by` flag clap-parses
    // into `HostStateCompareArgs::sort_by` as a raw String
    // (parsing through `parse_sort_by` happens later, in
    // `run_compare`). A regression that drops the field or
    // re-types it as `Vec<String>` (e.g. a misguided "make it
    // repeatable" refactor) would surface here at parse time.
    /// `ktstr host-state compare <a> <b>` with no `--sort-by`
    /// must parse with `sort_by` defaulting to the empty string.
    /// Positive-path pin for the default-sentinel contract.
    #[test]
    fn parse_host_state_compare_sort_by_defaults_to_empty() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "host-state",
            "compare",
            "/tmp/a.hst.zst",
            "/tmp/b.hst.zst",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::HostState {
                command: HostStateCommand::Compare(args),
            } => {
                assert_eq!(args.baseline, std::path::PathBuf::from("/tmp/a.hst.zst"));
                assert_eq!(args.candidate, std::path::PathBuf::from("/tmp/b.hst.zst"));
                // `--sort-by` defaults to the empty string —
                // parse_sort_by treats this as the "fall through
                // to default delta_pct sort" sentinel.
                assert!(args.sort_by.is_empty());
            }
            _ => panic!("expected HostState/Compare"),
        }
    }

    /// `ktstr host-state compare <a> <b> --sort-by <spec>` parses
    /// the spec verbatim into `HostStateCompareArgs::sort_by`.
    /// Pins that clap stores the raw string — `parse_sort_by`'s
    /// validation is deferred to `run_compare`, not parse time.
    /// Use a multi-key spec with mixed directions to exercise
    /// every parser feature reachable through clap.
    #[test]
    fn parse_host_state_compare_with_sort_by_succeeds() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "host-state",
            "compare",
            "/tmp/a.hst.zst",
            "/tmp/b.hst.zst",
            "--sort-by",
            "run_time_ns:desc,wait_time_ns:asc",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::HostState {
                command: HostStateCommand::Compare(args),
            } => {
                assert_eq!(args.sort_by, "run_time_ns:desc,wait_time_ns:asc");
            }
            _ => panic!("expected HostState/Compare"),
        }
    }

    /// `--group-by`, `--cgroup-flatten`, `--no-thread-normalize`,
    /// `--no-cg-normalize`, and `--sort-by` propagate from clap
    /// into the `HostStateCompareArgs` struct unchanged. Mirror
    /// of `parse_host_state_show_with_every_flag_succeeds` for
    /// the compare subcommand — pins the parse shape for every
    /// flag the compare subcommand surfaces. A regression that
    /// drops a flag from the clap struct or re-types a field
    /// (e.g. `--sort-by` to `Vec<String>`) would surface here at
    /// parse time before reaching `run_compare`.
    #[test]
    fn parse_host_state_compare_with_every_flag() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "host-state",
            "compare",
            "/tmp/a.hst.zst",
            "/tmp/b.hst.zst",
            "--group-by",
            "comm",
            "--cgroup-flatten",
            "/kubepods/*/workload",
            "--no-thread-normalize",
            "--no-cg-normalize",
            "--sort-by",
            "run_time_ns:desc,wait_sum:asc",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::HostState {
                command: HostStateCommand::Compare(args),
            } => {
                assert_eq!(args.baseline, std::path::PathBuf::from("/tmp/a.hst.zst"));
                assert_eq!(args.candidate, std::path::PathBuf::from("/tmp/b.hst.zst"));
                assert_eq!(args.group_by, host_state_compare::GroupBy::Comm);
                assert_eq!(
                    args.cgroup_flatten,
                    vec!["/kubepods/*/workload".to_string()]
                );
                assert!(args.no_thread_normalize);
                assert!(args.no_cg_normalize);
                // `--sort-by` comes through verbatim — clap stores
                // the raw spec; `parse_sort_by` is the parser layer
                // and runs in `run_compare`, not at clap parse time.
                assert_eq!(args.sort_by, "run_time_ns:desc,wait_sum:asc");
            }
            _ => panic!("expected HostState/Compare"),
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
            &[],
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
            write_show(&mut out, &snap, axis, &[], false, false, &[])
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
            &[],
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

    /// `write_show` with a non-empty `sort_by` re-orders groups
    /// by the named metric's *absolute aggregated value* in
    /// descending order — pins the new sort path against
    /// alphabetical-by-default. Three pcomm buckets (`alpha`,
    /// `bravo`, `charlie`) carry distinct `run_time_ns` totals so
    /// the bucket with the largest value lands first; that order
    /// is the inverse of the alphabetical default, so a regression
    /// dropping the sort path would surface here as the wrong
    /// first-group cell.
    #[test]
    fn write_show_sort_by_orders_groups_by_metric_descending() {
        let mut snap = ktstr::host_state::HostStateSnapshot::default();
        let mut t_alpha = ktstr::host_state::ThreadState::default();
        t_alpha.pcomm = "alpha".to_string();
        t_alpha.comm = "alpha-w".to_string();
        t_alpha.run_time_ns = 100;
        let mut t_bravo = ktstr::host_state::ThreadState::default();
        t_bravo.pcomm = "bravo".to_string();
        t_bravo.comm = "bravo-w".to_string();
        t_bravo.run_time_ns = 500;
        let mut t_charlie = ktstr::host_state::ThreadState::default();
        t_charlie.pcomm = "charlie".to_string();
        t_charlie.comm = "charlie-w".to_string();
        t_charlie.run_time_ns = 250;
        snap.threads.push(t_alpha);
        snap.threads.push(t_bravo);
        snap.threads.push(t_charlie);

        let sort_by = vec![host_state_compare::SortKey {
            metric: "run_time_ns",
            descending: true,
        }];
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            host_state_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &sort_by,
        )
        .expect("write_show into String must not fail");

        // The three group keys appear in descending run_time_ns
        // order: bravo (500) → charlie (250) → alpha (100). The
        // alphabetical default would place alpha first.
        let bravo_at = out.find("bravo").expect("bravo must surface in output");
        let charlie_at = out.find("charlie").expect("charlie must surface in output");
        let alpha_at = out.find("alpha").expect("alpha must surface in output");
        assert!(
            bravo_at < charlie_at,
            "sort_by run_time_ns:desc must place bravo (500) before charlie (250); \
             alpha={alpha_at} bravo={bravo_at} charlie={charlie_at}\nout:\n{out}",
        );
        assert!(
            charlie_at < alpha_at,
            "sort_by run_time_ns:desc must place charlie (250) before alpha (100); \
             alpha={alpha_at} bravo={bravo_at} charlie={charlie_at}\nout:\n{out}",
        );
    }

    /// `write_show` cgroup-stats secondary table renders each
    /// scalar via `format_scaled_u64` — single auto-scaled value
    /// per cell, no `→` arrow, no `(+0)` zero-delta suffix that
    /// the compare-side `cgroup_cell` helper carries when
    /// rendering a baseline→candidate→delta triple. Pins the UX
    /// fix that distinguishes show (single snapshot) from
    /// compare (paired snapshots): a 7.5 GiB `memory_current`
    /// row reads `7.500GiB`, not `7.500GiB → 7.500GiB (+0B)`.
    #[test]
    fn write_show_cgroup_stats_renders_single_value_no_delta() {
        let mut snap = ktstr::host_state::HostStateSnapshot::default();
        // Cgroup grouping requires at least one thread carrying
        // the cgroup path so build_groups produces a bucket; the
        // secondary table renders for every cgroup_stats key
        // regardless, but the primary table needs a row to
        // surround it (write_show returns Ok early only for an
        // empty cgroup_stats map, not an empty groups map).
        let mut t = ktstr::host_state::ThreadState::default();
        t.pcomm = "worker".to_string();
        t.comm = "w".to_string();
        t.cgroup = "/app".to_string();
        snap.threads.push(t);
        // 1 GiB exactly under IEC binary auto-scale → "1.000GiB".
        // 1_500_000 µs under the µs ladder → "1.500s". The other
        // two scalars (50 unitless, 200 µs) sit below their
        // respective ladders' first step (1e3 either way) so they
        // render as bare integers via `format_scaled_u64`'s
        // no-step-up path. This test asserts the GiB and s forms
        // since those are the load-bearing auto-scaled rows.
        let one_gib: u64 = 1024 * 1024 * 1024;
        // `CgroupStats` is `#[non_exhaustive]`, so direct struct
        // construction is rejected outside its defining module
        // (the `ktstr` binary is a separate translation unit
        // from `host_state`). Build via Default + per-field
        // assignment instead.
        let mut cgs = ktstr::host_state::CgroupStats::default();
        cgs.cpu_usage_usec = 1_500_000;
        cgs.nr_throttled = 50;
        cgs.throttled_usec = 200;
        cgs.memory_current = one_gib;
        snap.cgroup_stats.insert("/app".to_string(), cgs);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            host_state_compare::GroupBy::Cgroup,
            &[],
            false,
            false,
            &[],
        )
        .expect("write_show into String must not fail");

        // Each scaled scalar appears VERBATIM with no arrow or
        // delta tail — distinguishes show's single-value cells
        // from compare's `a → b (+d)` triples.
        assert!(
            out.contains("1.500s"),
            "cpu_usage_usec 1_500_000 µs must scale to '1.500s', got:\n{out}",
        );
        assert!(
            out.contains("1.000GiB"),
            "memory_current 1 GiB must scale to '1.000GiB', got:\n{out}",
        );
        // The arrow form would be a regression — single-snapshot
        // rendering must not pretend to carry a delta.
        assert!(
            !out.contains("→"),
            "show cgroup-stats must not emit a `→` arrow \
             (compare's two-value cell shape); got:\n{out}",
        );
        // `(+0` matches the regression signature — the
        // `format_delta_cell` zero-delta tail (`(+0)`, `(+0B)`,
        // `(+0µs)`) that compare's cgroup_cell emits when
        // baseline equals candidate. Bare `(` would over-match
        // any incidental paren in comfy_table output, so pin
        // the `(+0` prefix exactly.
        assert!(
            !out.contains("(+0"),
            "show cgroup-stats must not carry a `(+0…)` zero-delta tail; \
             got:\n{out}",
        );
    }

    /// `write_show` with an empty `sort_by` keeps the default
    /// alphabetical iteration over the BTreeMap-keyed groups.
    /// Mirror of the descending-sort test: same three buckets,
    /// no flag set, expect ascending alphabetical order
    /// regardless of the run_time_ns spread. Pins the
    /// "fall-through to default" branch of `write_show`.
    #[test]
    fn write_show_empty_sort_by_keeps_alphabetical_default() {
        let mut snap = ktstr::host_state::HostStateSnapshot::default();
        let mut t_zulu = ktstr::host_state::ThreadState::default();
        t_zulu.pcomm = "zulu".to_string();
        t_zulu.comm = "zulu-w".to_string();
        t_zulu.run_time_ns = 999;
        let mut t_alpha = ktstr::host_state::ThreadState::default();
        t_alpha.pcomm = "alpha".to_string();
        t_alpha.comm = "alpha-w".to_string();
        t_alpha.run_time_ns = 1;
        snap.threads.push(t_zulu);
        snap.threads.push(t_alpha);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            host_state_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
        )
        .expect("write_show into String must not fail");

        let alpha_at = out.find("alpha").expect("alpha must surface");
        let zulu_at = out.find("zulu").expect("zulu must surface");
        assert!(
            alpha_at < zulu_at,
            "empty sort_by must keep alphabetical order \
             (alpha before zulu); alpha={alpha_at} zulu={zulu_at}\nout:\n{out}",
        );
    }
}
