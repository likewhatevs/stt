#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use ktstr::cli;
use ktstr::cli::KernelCommand;
use ktstr::ctprof;
use ktstr::ctprof_compare;
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
    /// Show host CPU topology.
    Topo,
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

        #[arg(long, help = ktstr::cli::DISK_HELP)]
        disk: Option<String>,
    },
    /// Capture or compare a host-wide per-thread state snapshot.
    ///
    /// `capture` walks `/proc` at capture time and writes every
    /// visible thread's cumulative scheduling, memory, and I/O
    /// counters as zstd-compressed JSON (`.ctprof.zst`). Every
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
    Ctprof {
        #[command(subcommand)]
        command: CtprofCommand,
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
enum CtprofCommand {
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
    ///    `ctprof probe: <N> tgids walked, <N> jemalloc detected,
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
        /// Destination path (convention: `.ctprof.zst`).
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Compare two snapshots and render a per-metric diff table.
    Compare(ctprof_compare::CtprofCompareArgs),
    /// Render one snapshot as a per-group, per-metric table.
    ///
    /// Same grouping axis as `compare` (`pcomm` default, `cgroup`,
    /// `comm`, `comm-exact`) and the same per-metric aggregation
    /// rules from [`ctprof_compare::CTPROF_METRICS`], but
    /// with no baseline/candidate split — every metric renders one
    /// aggregated value per group rather than a delta against a
    /// reference snapshot. Useful for inspecting a capture in
    /// isolation: spot-check a process's per-thread CSW totals,
    /// audit the cgroup distribution, find which thread-name
    /// pattern has the highest run-time without comparing two
    /// runs.
    ///
    /// Pattern normalization (`comm` and `pcomm` token-based
    /// clustering, `cgroup` Layer-1/2/3 path tightening, and
    /// `## smaps_rollup` per-process pcomm-pattern aggregation)
    /// applies the same way it does in `compare`;
    /// `--no-thread-normalize` disables the name-family axes
    /// (comm, pcomm, smaps) and `--no-cg-normalize` disables the
    /// cgroup axis. `--cgroup-flatten` glob patterns also apply
    /// unchanged.
    Show(CtprofShowArgs),
    /// Print every registered metric with its tags and a
    /// one-line description.
    ///
    /// Discovery companion to `compare` and `show`: rendered
    /// table cells carry `[cfs-only]`, `[non-ext]`,
    /// `[fair-policy]`, `[SCHEDSTATS]`, `[SCHED_INFO]`,
    /// `[SCHED_CORE]`, `[SCHED_CLASS_EXT]`, `[TASK_DELAY_ACCT]`,
    /// `[TASK_IO_ACCOUNTING]`, and `[dead]` suffixes — this
    /// subcommand is the CLI-discoverable place to learn what
    /// each tag means and which kernel counter each metric
    /// surfaces.
    ///
    /// Output is two sections: a tag legend explaining each
    /// vocabulary element, then a `metric | tags | description`
    /// table covering every entry in the registry. Pure
    /// informational output — no snapshot is loaded; always
    /// returns 0.
    MetricList,
}

/// Arguments for the `ktstr ctprof show` subcommand. Field
/// shapes mirror [`ctprof_compare::CtprofCompareArgs`] so a
/// caller switching from compare to show keeps the same flag
/// vocabulary; the only structural difference is the single
/// positional path (one snapshot rather than baseline+candidate)
/// and the absence of any delta/percentage flags — show renders
/// one value per (group, metric) cell, no diff math.
#[derive(Debug, clap::Args)]
pub struct CtprofShowArgs {
    /// Snapshot path (`.ctprof.zst`) from `ktstr ctprof capture -o`.
    pub snapshot: std::path::PathBuf,
    /// Grouping key. Same semantics as
    /// `ktstr ctprof compare --group-by`. `pcomm` (default)
    /// aggregates per process name with token-based pattern
    /// normalization (so `worker-{0..N}` parents collapse into
    /// one `worker-{N}` bucket); `cgroup` per cgroup path; `comm`
    /// aggregates threads by NAME PATTERN under the same
    /// token-based normalizer; `comm-exact` is a synonym for
    /// `comm --no-thread-normalize` on the thread axis only.
    #[arg(long, value_enum, default_value_t = ctprof_compare::GroupBy::Pcomm, help_heading = "Grouping")]
    pub group_by: ctprof_compare::GroupBy,
    /// Glob patterns that collapse dynamic cgroup path segments —
    /// same shape as
    /// `ktstr ctprof compare --cgroup-flatten`. Repeatable.
    #[arg(long, help_heading = "Grouping")]
    pub cgroup_flatten: Vec<String>,
    /// Disable token-based pattern normalization across every
    /// name-family axis: `--group-by comm`, `--group-by pcomm`,
    /// AND the `## smaps_rollup` per-process keying (which
    /// normalizes by the pcomm pattern by default — see
    /// `ctprof_compare::collect_smaps_rollup`). With this flag
    /// set, threads / processes group by their literal name and
    /// smaps rows preserve their per-PID identity
    /// (`pcomm[tgid]`) instead of collapsing to the normalized
    /// pcomm pattern. The digit/hex/alpha-prefix placeholders
    /// are bypassed on every axis. Has no effect under
    /// `--group-by comm-exact` (already literal) or
    /// `--group-by cgroup`.
    #[arg(long, help_heading = "Grouping")]
    pub no_thread_normalize: bool,
    /// Disable token-based pattern normalization for the cgroup
    /// axis (`--group-by cgroup`). Cgroup paths group by literal
    /// post-`--cgroup-flatten` path. Has no effect under any
    /// other grouping.
    #[arg(long, help_heading = "Grouping")]
    pub no_cg_normalize: bool,
    /// Multi-key sort spec for the rendered groups. Format mirrors
    /// `ktstr ctprof compare --sort-by`:
    /// `metric1[:dir1],metric2[:dir2],...` where each `metric` is
    /// one of the primary or derived metric names (run
    /// `ctprof metric-list` for the full vocabulary) and
    /// `dir` is `asc` or `desc` (default `desc`). Show ranks
    /// groups by the tuple of the named metric's *absolute
    /// aggregated value* (not a delta — there is only one
    /// snapshot to inspect), descending by default; rows within
    /// a group keep registry order. Empty (the default) keeps
    /// the alphabetical group-key iteration. Example:
    /// `--sort-by run_time_ns,wait_sum:asc`.
    ///
    /// Affects only the per-thread metric table and the derived-
    /// metrics section. The `## smaps_rollup` sub-table sorts
    /// process rows independently by total Rss descending (its
    /// own built-in default); `--sort-by` does not propagate to
    /// it.
    #[arg(long, default_value = "", help_heading = "Display")]
    pub sort_by: String,
    /// Comma-separated column names to render. Empty (the
    /// default) emits the full `(group | threads | metric |
    /// value)` layout. Valid names: `group`, `threads`,
    /// `metric`, `value`. Order in the spec is the rendered
    /// order. Show is single-snapshot, so the
    /// `baseline`/`candidate`/`delta`/`%`/`arrow` columns
    /// (compare-only) are rejected at parse time. Example:
    /// `--columns metric,value`.
    #[arg(long, default_value = "", help_heading = "Display")]
    pub columns: String,
    /// Comma-separated section names to render. Empty (the
    /// default) renders every section that has data. When
    /// non-empty, restricts output to the listed sub-tables —
    /// every section not named is suppressed before its
    /// data-availability gate runs. Valid names: `primary`,
    /// `taskstats-delay`, `derived`, `cgroup-stats`,
    /// `cgroup-limits`, `memory-stat`, `memory-events`,
    /// `pressure`, `host-pressure`, `smaps-rollup`,
    /// `sched-ext`. Useful for narrowing a wide show to one
    /// area of interest. Example:
    /// `--sections primary,host-pressure`.
    #[arg(long, default_value = "", help_heading = "Filter")]
    pub sections: String,
    /// Comma-separated metric names to render. Empty (the
    /// default) renders every metric in the primary and
    /// derived sub-tables. When non-empty, restricts the
    /// rendered ROWS to the listed names — names must come
    /// from the `ctprof metric-list` vocabulary. Useful
    /// for zooming on a specific counter family without
    /// computing every metric: `--metrics
    /// run_time_ns,wait_sum,affine_success_ratio`. Composes
    /// with `--sections` — naming `--sections primary
    /// --metrics run_time_ns` shows a single primary row.
    #[arg(long, default_value = "", help_heading = "Filter")]
    pub metrics: String,
    /// Wrap table cells to fit the terminal width. Off by
    /// default — wide tables can spill past the terminal edge,
    /// matching the prior shell-pipeline-friendly layout. When
    /// set, cells too wide for the available width wrap inside
    /// the cell rather than overflowing, at the cost of taller
    /// rows. The wrap kicks in only when stdout is a tty (the
    /// terminal width is unknown otherwise); when piped to a
    /// file or another command, the flag is silently dropped
    /// and output stays unwrapped so awk/grep pipelines see
    /// the same byte sequence as without the flag.
    #[arg(long, help_heading = "Display")]
    pub wrap: bool,
    /// Maximum rendered lines per section. Sections whose table
    /// output exceeds this limit are truncated with a notice.
    /// Applies independently to each `## <heading>` sub-table.
    /// `0` disables truncation entirely. Default `500`.
    #[arg(long, default_value_t = 500, help_heading = "Display")]
    pub limit: usize,
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
#[allow(clippy::too_many_arguments)]
fn kernel_build(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
    extra_kconfig: Option<PathBuf>,
) -> Result<()> {
    // Read the extra-kconfig fragment ONCE up front so a range
    // expansion doesn't re-read the same file per version (and so
    // a bad path surfaces before any download / build work fires).
    // [`ktstr::cli::read_extra_kconfig`] does the 4-arm error
    // classification (ENOENT/EISDIR/EACCES/UTF-8) and emits an
    // empty-file warning so a 0-byte fragment doesn't silently
    // produce an "extras present but nothing merged" build.
    let extra_content: Option<String> = match extra_kconfig.as_ref() {
        Some(p) => Some(
            ktstr::cli::read_extra_kconfig(p, "ktstr").map_err(|e| anyhow::anyhow!("{e}"))?,
        ),
        None => None,
    };

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
                if let Err(e) = kernel_build_one(
                    Some(ver.clone()),
                    None,
                    None,
                    None,
                    force,
                    clean,
                    cpu_cap,
                    extra_content.as_deref(),
                ) {
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
            kernel_build_one(
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_content.as_deref(),
            )
        }
    } else {
        kernel_build_one(
            version,
            source,
            git,
            git_ref,
            force,
            clean,
            cpu_cap,
            extra_content.as_deref(),
        )
    }
}

/// Single-version variant of [`kernel_build`]: handles one tarball,
/// `--source`, or `--git` invocation. Carries the `kernel_build`
/// implementation as it stood before range dispatch was wired in;
/// extracted into a helper so the range loop in `kernel_build` can
/// reuse the same download + cache + build pipeline per resolved
/// version without duplicating it.
///
/// `extra_kconfig` is the pre-loaded user fragment from
/// `--extra-kconfig PATH` (read once in [`kernel_build`] before
/// fanning out to per-version invocations). `Some(content)` folds
/// into the cache key suffix via
/// [`ktstr::cache_key_suffix_with_extra`] and into the configure
/// pass via the Cow merge construction in
/// [`ktstr::cli::kernel_build_pipeline`].
fn kernel_build_one(
    version: Option<String>,
    source: Option<PathBuf>,
    git: Option<String>,
    git_ref: Option<String>,
    force: bool,
    clean: bool,
    cpu_cap: Option<usize>,
    extra_kconfig: Option<&str>,
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
    let mut acquired = if let Some(ref src_path) = source {
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
        // Check cache before downloading. Cache key folds in the
        // merged-kconfig hash so an `--extra-kconfig` build looks
        // up a distinct slot from a vanilla baked-in-only build —
        // `cache_key_suffix_with_extra(None)` equals
        // `cache_key_suffix()` so the no-extra path is byte-
        // identical to pre-flag behavior.
        let (arch, _) = fetch::arch_info();
        let cache_key = format!(
            "{ver}-tarball-{arch}-kc{}",
            ktstr::cache_key_suffix_with_extra(extra_kconfig),
        );
        if !force && let Some(entry) = cli::cache_lookup(&cache, &cache_key, "ktstr") {
            eprintln!("ktstr: cached kernel found: {}", entry.path.display());
            eprintln!("ktstr: use --force to rebuild");
            return Ok(());
        }
        let sp = cli::Spinner::start("Downloading kernel...");
        let result = fetch::download_tarball(client, &ver, tmp_dir.path(), "ktstr");
        drop(sp);
        let mut acquired = result?;
        // `download_tarball` builds its `cache_key` against the bare
        // `cache_key_suffix()` (see `fetch::download_tarball`).
        // Override with the merged-suffix key we looked up under so
        // the post-build cache store lands at the same slot we'd
        // hit on a re-run with the same `--extra-kconfig`.
        acquired.cache_key = cache_key;
        acquired
    };

    // For `--source` and `--git` paths, `local_source` and `git_clone`
    // build `acquired.cache_key` against the bare `cache_key_suffix()`
    // — already shaped `...-kc{baked_hash}`. With `--extra-kconfig`
    // set, lift the `-xkc{extra_hash}` append to
    // [`cli::append_extra_kconfig_suffix`] so both binaries share
    // one merge path.
    if source.is_some() || git.is_some() {
        cli::append_extra_kconfig_suffix(&mut acquired.cache_key, extra_kconfig);
    }

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
        extra_kconfig,
    )?;

    Ok(())
}

fn run_completions(shell: clap_complete::Shell, binary: &str) {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, binary, &mut std::io::stdout());
}

/// Entry point for `ktstr ctprof show <snapshot>`. Loads one
/// snapshot, builds groups along the requested axis, aggregates
/// every metric per
/// [`ctprof_compare::CTPROF_METRICS`], and renders the
/// result as a `(group_key, threads, metric, value)` table on
/// stdout. Exits non-zero only on I/O / parse errors; the
/// rendered table is always considered successful output.
///
/// Reuses the grouping pipeline from
/// [`ctprof_compare`] so pattern normalization (`comm` Layer-2
/// token clustering, `cgroup` Layer-1/2/3 tightening) and
/// `--cgroup-flatten` glob handling stay identical to the compare
/// path.
fn run_show(args: &CtprofShowArgs) -> Result<i32> {
    use anyhow::Context;
    // Parse `--sort-by` BEFORE the snapshot load so an operator
    // typo fails fast without paying for disk I/O. Mirrors
    // run_compare's ordering for the same reason. `--columns`
    // and `--sections` share the fail-fast contract — an invalid
    // spec must surface before the snapshot read.
    let sort_by = ctprof_compare::parse_sort_by(&args.sort_by)
        .with_context(|| format!("parse --sort-by {:?}", args.sort_by))?;
    // `--columns` parses with compare_side=false so the
    // baseline/candidate/delta/% columns (compare-only) are
    // rejected before any snapshot work begins.
    let columns = ctprof_compare::parse_columns(&args.columns, false)
        .with_context(|| format!("parse --columns {:?}", args.columns))?;
    let sections = ctprof_compare::parse_sections(&args.sections)
        .with_context(|| format!("parse --sections {:?}", args.sections))?;
    let metrics = ctprof_compare::parse_metrics(&args.metrics)
        .with_context(|| format!("parse --metrics {:?}", args.metrics))?;

    // Mirror run_compare's pre-load warning: explicit
    // `--sections cgroup-stats` (or any other cgroup-only
    // section) under a non-cgroup `--group-by` would render
    // zero rows for that section silently. Surface a
    // diagnostic before the snapshot load so the operator sees
    // it immediately.
    ctprof_compare::warn_cgroup_only_sections_under_non_cgroup(&sections, args.group_by);

    let snap = ktstr::ctprof::CtprofSnapshot::load(&args.snapshot)
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
        &columns,
        &sections,
        &metrics,
        args.wrap,
    );
    if args.limit > 0 {
        print!("{}", ctprof_compare::limit_sections(&out, args.limit));
    } else {
        print!("{out}");
    }
    Ok(0)
}

/// Render the per-group / per-metric show table into `w`. Mirrors
/// [`ctprof_compare::write_diff`]'s shape — the formatter
/// layer is split from the I/O wrapper so unit tests can drive
/// rendering into a `String` buffer without shelling through
/// stdout. Write errors propagate as [`std::fmt::Error`]; callers
/// that write into an infallible sink (`String`) can ignore.
///
/// `sections` is a [`ctprof_compare::Section`] filter — empty
/// renders every section that has data, non-empty restricts to
/// the named entries (mirrors `--sections` on `write_diff`).
/// `metrics` is a row-level filter — empty renders every metric
/// in the primary and derived sub-tables, non-empty restricts
/// the rendered rows to the listed names. `wrap` chooses between
/// `Disabled` (fixed-width, default) and `Dynamic`
/// (terminal-width-aware) comfy-table arrangement.
#[allow(clippy::too_many_arguments)]
fn write_show<W: std::fmt::Write>(
    w: &mut W,
    snap: &ktstr::ctprof::CtprofSnapshot,
    group_by: ctprof_compare::GroupBy,
    cgroup_flatten: &[String],
    no_thread_normalize: bool,
    no_cg_normalize: bool,
    sort_by: &[ctprof_compare::SortKey],
    columns: &[ctprof_compare::Column],
    sections: &[ctprof_compare::Section],
    metrics: &[&'static str],
    wrap: bool,
) -> std::fmt::Result {
    let flatten = ctprof_compare::compile_flatten_patterns(cgroup_flatten);
    // Single-snapshot cgroup key map. compare()'s
    // `build_cgroup_key_map` walks the union of paths from two
    // snapshots; for show we feed the same snap on both arguments
    // so the union iteration produces the same key set as a
    // single-snap pass (the inner BTreeSet of paths dedups the
    // duplicate insertions). Skipped under `--no-cg-normalize`
    // and under any grouping other than Cgroup.
    let cgroup_key_map = if group_by == ctprof_compare::GroupBy::Cgroup && !no_cg_normalize {
        Some(ctprof_compare::build_cgroup_key_map(snap, snap, &flatten))
    } else {
        None
    };
    // Pattern counts: `build_groups` falls back to a per-snapshot
    // local count when None, which is exactly the right behavior
    // for a single-snapshot view (compare()'s union-over-baseline-
    // and-candidate gate doesn't apply when there's only one
    // snapshot to inspect).
    let groups = ctprof_compare::build_groups(
        snap,
        group_by,
        &flatten,
        None,
        cgroup_key_map.as_ref(),
        no_thread_normalize,
    );

    let group_header = match group_by {
        ctprof_compare::GroupBy::Pcomm => "pcomm",
        ctprof_compare::GroupBy::Cgroup => "cgroup",
        ctprof_compare::GroupBy::Comm => "comm-pattern",
        ctprof_compare::GroupBy::CommExact => "comm",
        ctprof_compare::GroupBy::All => unreachable!("All is decomposed before write_show"),
    };

    // Resolve the column set: caller-supplied override wins,
    // otherwise fall back to the show-side default
    // `(group, threads, metric, value)`. DisplayOptions is
    // marked `#[non_exhaustive]` so we mutate the
    // default-constructed struct rather than a struct
    // expression. Plumbing wrap/sections through DisplayOptions
    // means every section's table-builder call site below routes
    // through `display_options.new_table()` (the wrap-aware
    // helper) and every section emission gates on
    // `display_options.is_section_enabled(...)`.
    let mut display_options = ctprof_compare::DisplayOptions::default();
    display_options.columns = columns.to_vec();
    display_options.sections = sections.to_vec();
    display_options.metrics = metrics.to_vec();
    display_options.wrap = wrap;
    let resolved_columns = display_options.resolved_show_columns();

    // Iteration order: when `sort_by` is empty, fall through the
    // BTreeMap by-key order (alphabetical group key, registry order
    // metrics); when non-empty, rank groups by the tuple of named
    // metrics' *absolute aggregated values* (no deltas — show has
    // one snapshot, not two). Mirrors compare's
    // `sort_diff_rows_by_keys` shape (per-key direction, lex order
    // on the tuple, deterministic group_key tie-break) but on
    // `Aggregated::numeric()` rather than `DiffRow.delta`. Within a
    // group, metrics still iterate `CTPROF_METRICS` so the
    // metric column lands in registry order regardless of sort.
    //
    // Computed once outside the section gates because every
    // section that iterates groups (Primary + Derived) reuses
    // it; computing inside the Primary `if` would either
    // duplicate the work or leak the binding into a scope where
    // a `--sections derived` invocation skips the Primary block
    // entirely and then can't reach `group_order`.
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

    // Primary table renders rows whose metric.section is
    // enabled. Two sections share the table — Section::Primary
    // (52 non-taskstats rows) and Section::TaskstatsDelay (34
    // taskstats genetlink rows). The outer gate keeps the table
    // open while EITHER section is enabled. Mirrors the
    // ctprof_compare::write_diff outer-gate semantics so
    // `--sections taskstats-delay` works identically across
    // compare and show.
    if display_options.is_section_enabled(ctprof_compare::Section::Primary)
        || display_options.is_section_enabled(ctprof_compare::Section::TaskstatsDelay)
    {
        writeln!(w, "## Primary metrics")?;
        let mut table = display_options.new_table();
        let header_row: Vec<&str> = resolved_columns
            .iter()
            .map(|c| c.header(group_header))
            .collect();
        table.set_header(header_row);

        for key in &group_order {
            let group = &groups[*key];
            // Display key: pattern grouping under Comm or Pcomm
            // uses grex to turn the join-key skeleton into a regex
            // label; every other grouping (CommExact, Cgroup, or
            // either pattern axis under `--no-thread-normalize`)
            // renders the join key directly.
            let display_key = if matches!(
                group_by,
                ctprof_compare::GroupBy::Comm | ctprof_compare::GroupBy::Pcomm
            ) && !no_thread_normalize
            {
                ctprof_compare::pattern_display_label(key, &group.members)
            } else {
                (*key).clone()
            };
            for metric in ctprof_compare::CTPROF_METRICS {
                // `--metrics` filter: skip metrics not on the
                // operator-supplied allowlist. Empty allowlist
                // = no filter (default) per
                // `is_metric_enabled`'s default-empty contract.
                if !display_options.is_metric_enabled(metric.name) {
                    continue;
                }
                // Per-row section gate: skip metrics whose
                // `section` is not enabled by `--sections`. The
                // outer gate above keeps the table open while
                // either section is enabled; this inner gate
                // restricts which rows appear inside the table.
                if !display_options.is_section_enabled(metric.section) {
                    continue;
                }
                let Some(agg) = group.metrics.get(metric.name) else {
                    continue;
                };
                let metric_name = ctprof_compare::metric_display_name(metric).to_string();
                let value_cell = ctprof_compare::format_value_cell(agg, metric.rule.ladder());
                let tags_cell = ctprof_compare::metric_tags(metric);
                let cells: Vec<String> = resolved_columns
                    .iter()
                    .map(|c| match c {
                        ctprof_compare::Column::Group => display_key.clone(),
                        ctprof_compare::Column::Threads => group.thread_count.to_string(),
                        ctprof_compare::Column::Metric => metric_name.clone(),
                        ctprof_compare::Column::Value => value_cell.clone(),
                        ctprof_compare::Column::Tags => tags_cell.clone(),
                        ctprof_compare::Column::Uptime => "-".to_string(),
                        _ => "-".to_string(),
                    })
                    .collect();
                table.add_row(cells);
            }
        }
        writeln!(w, "{table}")?;
    }

    // Derived metrics: one row per (group, derivation) pair.
    // Mirrors the `## Derived metrics` section emitted by
    // ctprof_compare::write_diff but adapted for the
    // single-snapshot show layout (no baseline/candidate split,
    // one value cell per row).
    // Derived-table outer gate mirrors write_diff: open the
    // table when EITHER `Section::Derived` OR
    // `Section::TaskstatsDelay` is enabled. Per-row gating below
    // keeps `--sections taskstats-delay` from leaking
    // non-taskstats derivations.
    if (display_options.is_section_enabled(ctprof_compare::Section::Derived)
        || display_options.is_section_enabled(ctprof_compare::Section::TaskstatsDelay))
        && !groups.is_empty()
    {
        let mut dt = display_options.new_table();
        let header_row: Vec<&str> = resolved_columns
            .iter()
            .map(|c| c.header(group_header))
            .collect();
        dt.set_header(header_row);
        // Iterate groups in the same order as the main table —
        // group_order has been computed once and applies to
        // every section emitted afterwards.
        for key in &group_order {
            let group = &groups[*key];
            let display_key = if matches!(
                group_by,
                ctprof_compare::GroupBy::Comm | ctprof_compare::GroupBy::Pcomm
            ) && !no_thread_normalize
            {
                ctprof_compare::pattern_display_label(key, &group.members)
            } else {
                (*key).clone()
            };
            for d in ctprof_compare::CTPROF_DERIVED_METRICS {
                if !display_options.is_metric_enabled(d.name) {
                    continue;
                }
                // Per-row section gate: same shape as the primary
                // table loop. Skip derivations whose section is
                // not enabled.
                if !display_options.is_section_enabled(d.section) {
                    continue;
                }
                let value_cell = match (d.compute)(&group.metrics) {
                    Some(v) => ctprof_compare::format_derived_value_cell(v, d.ladder, d.is_ratio),
                    None => "-".to_string(),
                };
                let cells: Vec<String> = resolved_columns
                    .iter()
                    .map(|c| match c {
                        ctprof_compare::Column::Group => display_key.clone(),
                        ctprof_compare::Column::Threads => group.thread_count.to_string(),
                        ctprof_compare::Column::Metric => d.name.to_string(),
                        ctprof_compare::Column::Value => value_cell.clone(),
                        ctprof_compare::Column::Tags => String::new(),
                        ctprof_compare::Column::Uptime => "-".to_string(),
                        _ => "-".to_string(),
                    })
                    .collect();
                dt.add_row(ctprof_compare::color_derived_cells(cells));
            }
        }
        writeln!(w)?;
        writeln!(w, "## Derived metrics")?;
        writeln!(w, "{dt}")?;
    }

    // Cgroup grouping carries cgroup_stats enrichment alongside
    // the per-thread aggregates. Render a second table when
    // present so the show output mirrors compare's two-table
    // layout for `--group-by cgroup`. The `--sections` filter is
    // re-checked per sub-table below so a user can request
    // `--sections pressure` and get only the PSI rollups even
    // though the cgroup-stats prefix is present in the snapshot.
    if group_by == ctprof_compare::GroupBy::Cgroup && !snap.cgroup_stats.is_empty() {
        let stats = ctprof_compare::flatten_cgroup_stats(
            &snap.cgroup_stats,
            &flatten,
            cgroup_key_map.as_ref(),
        );
        if !stats.is_empty() {
            if display_options.is_section_enabled(ctprof_compare::Section::CgroupStats) {
                writeln!(w)?;
                let mut ct = display_options.new_table();
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
                        ctprof_compare::format_scaled_u64(
                            s.cpu.usage_usec,
                            ctprof_compare::ScaleLadder::Us,
                        ),
                        ctprof_compare::format_scaled_u64(
                            s.cpu.nr_throttled,
                            ctprof_compare::ScaleLadder::Unitless,
                        ),
                        ctprof_compare::format_scaled_u64(
                            s.cpu.throttled_usec,
                            ctprof_compare::ScaleLadder::Us,
                        ),
                        ctprof_compare::format_scaled_u64(
                            s.memory.current,
                            ctprof_compare::ScaleLadder::Bytes,
                        ),
                    ]);
                }
                writeln!(w, "{ct}")?;
            }

            // Per-cgroup limits / knobs sub-table — operator-set
            // configuration that's typically static across a run
            // but matters when comparing two snapshots that
            // straddle a deployment. `cpu.max`, `cpu.weight`,
            // `memory.max`, `memory.high`, `pids.current/max` per
            // [`CgroupCpuStats`] / [`CgroupMemoryStats`] /
            // [`CgroupPidsStats`]. Suppressed entirely when no
            // cgroup in the bucket exposes any of these (root
            // cgroup, or a host without pids/memory controllers
            // enabled).
            if display_options.is_section_enabled(ctprof_compare::Section::Limits)
                && stats.values().any(|s| {
                    s.cpu.max_quota_us.is_some()
                        || s.cpu.weight.is_some()
                        || s.memory.max.is_some()
                        || s.memory.high.is_some()
                        || s.pids.current.is_some()
                        || s.pids.max.is_some()
                })
            {
                writeln!(w)?;
                writeln!(w, "## Cgroup limits / knobs")?;
                let mut lt = display_options.new_table();
                lt.set_header(vec![
                    "cgroup",
                    "cpu.max",
                    "cpu.weight",
                    "memory.max",
                    "memory.high",
                    "pids.current",
                    "pids.max",
                ]);
                for (key, s) in &stats {
                    // Per-row gate: skip rows where every column
                    // is unset (the cgroup has no caps, no
                    // weight set, no pids accounting). Without
                    // this, a system-wide table can render N
                    // empty rows for every host-controller
                    // cgroup that doesn't expose any of these.
                    let row_has_data = s.cpu.max_quota_us.is_some()
                        || s.cpu.weight.is_some()
                        || s.memory.max.is_some()
                        || s.memory.high.is_some()
                        || s.pids.current.is_some()
                        || s.pids.max.is_some();
                    if !row_has_data {
                        continue;
                    }
                    lt.add_row(vec![
                        key.clone(),
                        ctprof_compare::format_cpu_max(s.cpu.max_quota_us, s.cpu.max_period_us),
                        s.cpu
                            .weight
                            .map(|v| {
                                ctprof_compare::format_scaled_u64(
                                    v,
                                    ctprof_compare::ScaleLadder::Unitless,
                                )
                            })
                            .unwrap_or_else(|| "-".to_string()),
                        ctprof_compare::format_optional_limit(
                            s.memory.max,
                            ctprof_compare::ScaleLadder::Bytes,
                        ),
                        ctprof_compare::format_optional_limit(
                            s.memory.high,
                            ctprof_compare::ScaleLadder::Bytes,
                        ),
                        s.pids
                            .current
                            .map(|v| {
                                ctprof_compare::format_scaled_u64(
                                    v,
                                    ctprof_compare::ScaleLadder::Unitless,
                                )
                            })
                            .unwrap_or_else(|| "-".to_string()),
                        ctprof_compare::format_optional_limit(
                            s.pids.max,
                            ctprof_compare::ScaleLadder::Unitless,
                        ),
                    ]);
                }
                writeln!(w, "{lt}")?;
            }

            // Per-cgroup memory.stat sub-table — kernel-emitted
            // memory counters per cgroup. Up to 71 keys on a
            // recent kernel. Renders as one row per (cgroup,
            // key) pair to keep column width bounded; sorted by
            // key for stable output. Suppressed when every
            // bucketed cgroup has an empty `memory.stat` map.
            // Show-side zero-suppression: a typical workload
            // touches only a handful of memory.stat keys, so
            // rendering all 71 rows × N cgroups creates a
            // massive table dominated by zeros. Skip rows where
            // the value is exactly 0; if every key in a cgroup
            // is zero, that cgroup contributes no rows. The
            // section still renders if any cgroup has any
            // non-zero key. This trims output ~10x for typical
            // runs.
            if display_options.is_section_enabled(ctprof_compare::Section::MemoryStat)
                && stats
                    .values()
                    .any(|s| s.memory.stat.values().any(|v| *v != 0))
            {
                writeln!(w)?;
                writeln!(w, "## memory.stat")?;
                let mut mt = display_options.new_table();
                mt.set_header(vec!["cgroup", "key", "value"]);
                for (key, s) in &stats {
                    for (stat_key, stat_value) in &s.memory.stat {
                        if *stat_value == 0 {
                            continue;
                        }
                        mt.add_row(vec![
                            key.clone(),
                            stat_key.clone(),
                            ctprof_compare::format_scaled_u64(
                                *stat_value,
                                ctprof_compare::ScaleLadder::Unitless,
                            ),
                        ]);
                    }
                }
                writeln!(w, "{mt}")?;
            }

            // Per-cgroup memory.events sub-table — pressure-event
            // counters (low / high / max / oom / oom_kill etc.).
            // Same long-table layout as memory.stat with the
            // same zero-row suppression.
            if display_options.is_section_enabled(ctprof_compare::Section::MemoryEvents)
                && stats
                    .values()
                    .any(|s| s.memory.events.values().any(|v| *v != 0))
            {
                writeln!(w)?;
                writeln!(w, "## memory.events")?;
                let mut et = display_options.new_table();
                et.set_header(vec!["cgroup", "event", "count"]);
                for (key, s) in &stats {
                    for (event_key, event_value) in &s.memory.events {
                        if *event_value == 0 {
                            continue;
                        }
                        et.add_row(vec![
                            key.clone(),
                            event_key.clone(),
                            ctprof_compare::format_scaled_u64(
                                *event_value,
                                ctprof_compare::ScaleLadder::Unitless,
                            ),
                        ]);
                    }
                }
                writeln!(w, "{et}")?;
            }

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
            if display_options.is_section_enabled(ctprof_compare::Section::Pressure) {
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
                    let mut pt = display_options.new_table();
                    pt.set_header(vec!["cgroup", "row", "avg10", "avg60", "avg300", "total"]);
                    for (key, s) in &stats {
                        let r = accessor(&s.psi);
                        pt.add_row(vec![
                            key.clone(),
                            "some".into(),
                            format_psi_avg(r.some.avg10),
                            format_psi_avg(r.some.avg60),
                            format_psi_avg(r.some.avg300),
                            ctprof_compare::format_scaled_u64(
                                r.some.total_usec,
                                ctprof_compare::ScaleLadder::Us,
                            ),
                        ]);
                        pt.add_row(vec![
                            key.clone(),
                            "full".into(),
                            format_psi_avg(r.full.avg10),
                            format_psi_avg(r.full.avg60),
                            format_psi_avg(r.full.avg300),
                            ctprof_compare::format_scaled_u64(
                                r.full.total_usec,
                                ctprof_compare::ScaleLadder::Us,
                            ),
                        ]);
                    }
                    writeln!(w, "{pt}")?;
                }
            }
        }
    }

    // Host-level PSI — surface above the per-thread table when
    // any resource has nonzero data. Renders as four per-resource
    // sub-tables (cpu / memory / io / irq) with a `some`+`full`
    // row each, matching the per-cgroup layout above.
    if display_options.is_section_enabled(ctprof_compare::Section::HostPressure)
        && host_psi_has_data(&snap.psi)
    {
        for (resource_name, accessor) in psi_resources() {
            let r = accessor(&snap.psi);
            if !psi_resource_has_data(&r) {
                continue;
            }
            writeln!(w)?;
            writeln!(w, "## Host pressure / {resource_name}")?;
            let mut pt = display_options.new_table();
            pt.set_header(vec!["row", "avg10", "avg60", "avg300", "total"]);
            pt.add_row(vec![
                "some".into(),
                format_psi_avg(r.some.avg10),
                format_psi_avg(r.some.avg60),
                format_psi_avg(r.some.avg300),
                ctprof_compare::format_scaled_u64(
                    r.some.total_usec,
                    ctprof_compare::ScaleLadder::Us,
                ),
            ]);
            pt.add_row(vec![
                "full".into(),
                format_psi_avg(r.full.avg10),
                format_psi_avg(r.full.avg60),
                format_psi_avg(r.full.avg300),
                ctprof_compare::format_scaled_u64(
                    r.full.total_usec,
                    ctprof_compare::ScaleLadder::Us,
                ),
            ]);
            writeln!(w, "{pt}")?;
        }
    }

    // Per-process smaps_rollup sub-table. Routes through the
    // shared [`ctprof_compare::collect_smaps_rollup`] so the
    // show-side keying and aggregation match the compare-side
    // exactly: under default normalization the key is
    // `pattern_key(&t.pcomm)` (tgid dropped) and per-PID rows
    // sharing the same pcomm pattern have their byte counts
    // field-summed; under `--no-thread-normalize` the literal
    // `pcomm[tgid]` shape is preserved so each PID stays
    // attributable. Process iteration order: descending by Rss,
    // tiebreak descending Pss, final tiebreak alphabetical
    // (mirrors the compare-side sort). Skip zero-valued entries
    // per-row to keep output bounded — Pss for an unmapped
    // process is meaningfully zero, but ShmemPmdMapped=0 etc.
    // are noise rows. Suppressed when no captured thread has a
    // populated map (older kernels, stripped permissions,
    // synthetic fixtures).
    //
    // Smaps keys can differ from primary-table Pcomm group
    // keys for singleton digit pcomms — smaps always normalizes
    // (`worker-{N}` even when only one PID matches), while the
    // primary table reverts singletons to the literal pcomm
    // (`worker-7`); see [`ctprof_compare::collect_smaps_rollup`]
    // for the asymmetry and its rationale (cross-snapshot diff
    // joining vs. intra-snapshot fleet aggregation).
    //
    // Smaps keying is independent of `--group-by`: the keys
    // reflect the per-process pcomm pattern regardless of
    // whether the operator selected `cgroup`, `comm`, `pcomm`,
    // or `comm-exact` for the primary table. The smaps section
    // reads pcomm directly off each leader thread, not the
    // post-grouping bucket key.
    if display_options.is_section_enabled(ctprof_compare::Section::Smaps) {
        let smaps = ctprof_compare::collect_smaps_rollup(snap, no_thread_normalize);
        if !smaps.is_empty() {
            let mut process_keys: Vec<&String> = smaps.keys().collect();
            process_keys.sort_by(|a, b| {
                let max_for = |pkey: &&String, field: &str| -> u64 {
                    smaps
                        .get(*pkey)
                        .and_then(|m| m.get(field).copied())
                        .unwrap_or(0)
                };
                max_for(b, "Rss")
                    .cmp(&max_for(a, "Rss"))
                    .then_with(|| max_for(b, "Pss").cmp(&max_for(a, "Pss")))
                    .then_with(|| a.cmp(b))
            });
            // Pre-pass: every (process, key) pair with non-zero
            // value emits a row. Suppress the section header when
            // no rows would render (e.g. every value is zero).
            let any_row = process_keys.iter().any(|pkey| {
                smaps
                    .get(*pkey)
                    .map(|m| m.values().any(|v| *v != 0))
                    .unwrap_or(false)
            });
            if any_row {
                writeln!(w)?;
                writeln!(w, "## smaps_rollup")?;
                let mut st = display_options.new_table();
                st.set_header(vec!["process", "key", "value"]);
                for pkey in &process_keys {
                    // `process_keys` is built from `smaps.keys()`,
                    // so every entry resolves — index directly to
                    // make the invariant explicit.
                    let m = &smaps[*pkey];
                    for (key, bytes) in m {
                        if *bytes == 0 {
                            continue;
                        }
                        st.add_row(vec![
                            (*pkey).clone(),
                            key.clone(),
                            ctprof_compare::format_scaled_u64(
                                *bytes,
                                ctprof_compare::ScaleLadder::Bytes,
                            ),
                        ]);
                    }
                }
                writeln!(w, "{st}")?;
            }
        }
    }

    // Global sched_ext sysfs section. Suppressed when the
    // snapshot's `sched_ext` field is None (CONFIG_SCHED_CLASS_EXT=n
    // build, or sysfs directory absent). Single 5-row table
    // mirroring the kernel's exposed scx_global_attrs[] surface.
    if display_options.is_section_enabled(ctprof_compare::Section::SchedExt)
        && let Some(scx) = &snap.sched_ext
    {
        writeln!(w)?;
        writeln!(w, "## sched_ext")?;
        let mut at = display_options.new_table();
        at.set_header(vec!["attr", "value"]);
        // state cell: render "-" when the file was unreadable
        // (empty string) so "no observation" stays visually
        // distinct from an actual scx_enable_state_str[] value.
        // Mirrors the compare-side rendering.
        let state_cell = if scx.state.is_empty() {
            "-".to_string()
        } else {
            scx.state.clone()
        };
        at.add_row(vec!["state".into(), state_cell]);
        at.add_row(vec![
            "switch_all".into(),
            ctprof_compare::format_scaled_u64(
                scx.switch_all,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "nr_rejected".into(),
            ctprof_compare::format_scaled_u64(
                scx.nr_rejected,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "hotplug_seq".into(),
            ctprof_compare::format_scaled_u64(
                scx.hotplug_seq,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        at.add_row(vec![
            "enable_seq".into(),
            ctprof_compare::format_scaled_u64(
                scx.enable_seq,
                ctprof_compare::ScaleLadder::Unitless,
            ),
        ]);
        writeln!(w, "{at}")?;
    }

    Ok(())
}

/// One entry in the [`psi_resources`] table — a display name
/// paired with the accessor that pulls one
/// [`ctprof::PsiResource`] out of a [`ctprof::Psi`]
/// bundle.
type PsiAccessor = (&'static str, fn(&ctprof::Psi) -> ctprof::PsiResource);

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
fn host_psi_has_data(psi: &ctprof::Psi) -> bool {
    [psi.cpu, psi.memory, psi.io, psi.irq]
        .iter()
        .any(psi_resource_has_data)
}

fn psi_resource_has_data(r: &ctprof::PsiResource) -> bool {
    let h =
        |h: &ctprof::PsiHalf| h.avg10 != 0 || h.avg60 != 0 || h.avg300 != 0 || h.total_usec != 0;
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
    use ktstr::ctprof::{Psi, PsiHalf, PsiResource};

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
    /// constraint that drives `tests/common/ctprof.rs`'s
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
        Command::Topo => {
            let topo = TestTopology::from_system()?;
            println!("CPUs:       {}", topo.total_cpus());
            println!("LLCs:       {}", topo.num_llcs());
            println!("NUMA nodes: {}", topo.num_numa_nodes());
            for (i, llc) in topo.llcs().iter().enumerate() {
                println!("  LLC {} (node {}): {:?}", i, llc.numa_node(), llc.cpus(),);
            }
        }

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
                extra_kconfig,
            } => kernel_build(
                version,
                source,
                git,
                git_ref,
                force,
                clean,
                cpu_cap,
                extra_kconfig,
            )?,
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
            disk,
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
            // Shared parse path with `cargo ktstr shell` — surfaces
            // size errors at CLI-argument time, never mid-VM-setup.
            let disk_cfg = cli::parse_disk_arg(disk.as_deref())?;
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
                disk_cfg,
            )?;
        }

        Command::Ctprof { command } => match command {
            CtprofCommand::Capture { output } => {
                ctprof::capture_to(&output)?;
                eprintln!("ktstr: wrote ctprof snapshot to {}", output.display());
            }
            CtprofCommand::Compare(args) => {
                let code = ctprof_compare::run_compare(&args)?;
                if code != 0 {
                    std::process::exit(code);
                }
            }
            CtprofCommand::Show(args) => {
                let code = run_show(&args)?;
                if code != 0 {
                    std::process::exit(code);
                }
            }
            CtprofCommand::MetricList => {
                let code = ctprof_compare::run_metric_list()?;
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
    use ktstr::metric_types::{MonotonicCount, MonotonicNs};

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

    // -- ctprof show CLI tests
    //
    // Pin clap-parse shape on the `Show` variant + the rendered
    // output of `write_show` against a synthetic snapshot. The
    // CLI-parse tests guard against argument-vocabulary drift
    // (matching the pattern compare's args use) and the
    // write_show tests guard the rendered table against
    // regressions in column count, header, and group-axis routing.

    /// `ktstr ctprof show <path>` with no flags must parse
    /// into a Show variant carrying the positional path and
    /// default GroupBy::Pcomm. Positive-path pin for argv shape.
    #[test]
    fn parse_ctprof_show_positional_only_succeeds() {
        let parsed = Cli::try_parse_from(["ktstr", "ctprof", "show", "/tmp/snap.ctprof.zst"])
            .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::Ctprof {
                command: CtprofCommand::Show(args),
            } => {
                assert_eq!(
                    args.snapshot,
                    std::path::PathBuf::from("/tmp/snap.ctprof.zst")
                );
                assert_eq!(args.group_by, ctprof_compare::GroupBy::Pcomm);
                assert!(args.cgroup_flatten.is_empty());
                assert!(!args.no_thread_normalize);
                assert!(!args.no_cg_normalize);
                // `--sort-by` defaults to the empty string —
                // empty-spec sentinel for "fall through to
                // alphabetical iteration".
                assert!(args.sort_by.is_empty());
            }
            _ => panic!("expected Ctprof/Show"),
        }
    }

    /// `--group-by`, `--cgroup-flatten`, `--no-thread-normalize`,
    /// `--no-cg-normalize`, and `--sort-by` propagate from clap
    /// into the `CtprofShowArgs` struct unchanged. Pins the
    /// parse shape for every flag the show subcommand surfaces.
    #[test]
    fn parse_ctprof_show_with_every_flag_succeeds() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "ctprof",
            "show",
            "/tmp/snap.ctprof.zst",
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
            Command::Ctprof {
                command: CtprofCommand::Show(args),
            } => {
                assert_eq!(args.group_by, ctprof_compare::GroupBy::Comm);
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
            _ => panic!("expected Ctprof/Show"),
        }
    }

    /// `ktstr ctprof show <path> --sort-by run_time_ns` parses
    /// successfully with the spec stored verbatim on
    /// `CtprofShowArgs::sort_by`. Single-key, default-direction
    /// pin — the unmarked form ("--sort-by metric" without a `:dir`
    /// suffix) routes through `parse_sort_by`'s `None` arm in
    /// `run_show` and ranks descending.
    #[test]
    fn parse_ctprof_show_sort_by_single_key_succeeds() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "ctprof",
            "show",
            "/tmp/snap.ctprof.zst",
            "--sort-by",
            "run_time_ns",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::Ctprof {
                command: CtprofCommand::Show(args),
            } => {
                assert_eq!(args.sort_by, "run_time_ns");
            }
            _ => panic!("expected Ctprof/Show"),
        }
    }

    // -- ctprof compare CLI --sort-by clap-parse pins
    //
    // Mirror of the show-side parse tests above for the compare
    // subcommand. Pins that the `--sort-by` flag clap-parses
    // into `CtprofCompareArgs::sort_by` as a raw String
    // (parsing through `parse_sort_by` happens later, in
    // `run_compare`). A regression that drops the field or
    // re-types it as `Vec<String>` (e.g. a misguided "make it
    // repeatable" refactor) would surface here at parse time.
    /// `ktstr ctprof compare <a> <b>` with no `--sort-by`
    /// must parse with `sort_by` defaulting to the empty string.
    /// Positive-path pin for the default-sentinel contract.
    #[test]
    fn parse_ctprof_compare_sort_by_defaults_to_empty() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "ctprof",
            "compare",
            "/tmp/a.ctprof.zst",
            "/tmp/b.ctprof.zst",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::Ctprof {
                command: CtprofCommand::Compare(args),
            } => {
                assert_eq!(args.baseline, std::path::PathBuf::from("/tmp/a.ctprof.zst"));
                assert_eq!(
                    args.candidate,
                    std::path::PathBuf::from("/tmp/b.ctprof.zst")
                );
                // `--sort-by` defaults to the empty string —
                // parse_sort_by treats this as the "fall through
                // to default delta_pct sort" sentinel.
                assert!(args.sort_by.is_empty());
            }
            _ => panic!("expected Ctprof/Compare"),
        }
    }

    /// `ktstr ctprof compare <a> <b> --sort-by <spec>` parses
    /// the spec verbatim into `CtprofCompareArgs::sort_by`.
    /// Pins that clap stores the raw string — `parse_sort_by`'s
    /// validation is deferred to `run_compare`, not parse time.
    /// Use a multi-key spec with mixed directions to exercise
    /// every parser feature reachable through clap.
    #[test]
    fn parse_ctprof_compare_with_sort_by_succeeds() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "ctprof",
            "compare",
            "/tmp/a.ctprof.zst",
            "/tmp/b.ctprof.zst",
            "--sort-by",
            "run_time_ns:desc,wait_time_ns:asc",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        match parsed.command {
            Command::Ctprof {
                command: CtprofCommand::Compare(args),
            } => {
                assert_eq!(args.sort_by, "run_time_ns:desc,wait_time_ns:asc");
            }
            _ => panic!("expected Ctprof/Compare"),
        }
    }

    /// `--group-by`, `--cgroup-flatten`, `--no-thread-normalize`,
    /// `--no-cg-normalize`, and `--sort-by` propagate from clap
    /// into the `CtprofCompareArgs` struct unchanged. Mirror
    /// of `parse_ctprof_show_with_every_flag_succeeds` for
    /// the compare subcommand — pins the parse shape for every
    /// flag the compare subcommand surfaces. A regression that
    /// drops a flag from the clap struct or re-types a field
    /// (e.g. `--sort-by` to `Vec<String>`) would surface here at
    /// parse time before reaching `run_compare`.
    #[test]
    fn parse_ctprof_compare_with_every_flag() {
        let parsed = Cli::try_parse_from([
            "ktstr",
            "ctprof",
            "compare",
            "/tmp/a.ctprof.zst",
            "/tmp/b.ctprof.zst",
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
            Command::Ctprof {
                command: CtprofCommand::Compare(args),
            } => {
                assert_eq!(args.baseline, std::path::PathBuf::from("/tmp/a.ctprof.zst"));
                assert_eq!(
                    args.candidate,
                    std::path::PathBuf::from("/tmp/b.ctprof.zst")
                );
                assert_eq!(args.group_by, ctprof_compare::GroupBy::Comm);
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
            _ => panic!("expected Ctprof/Compare"),
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
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
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
        let mut t1 = ktstr::ctprof::ThreadState::default();
        t1.pcomm = "worker-proc".to_string();
        t1.comm = "worker-0".to_string();
        t1.nr_wakeups = MonotonicCount(1);
        let mut t2 = ktstr::ctprof::ThreadState::default();
        t2.pcomm = "worker-proc".to_string();
        t2.comm = "worker-1".to_string();
        t2.nr_wakeups = MonotonicCount(2);
        snap.threads.push(t1);
        snap.threads.push(t2);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &[],
            &[],
            &[],
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
    /// vocabulary drift between this and `ctprof_compare`.
    #[test]
    fn write_show_header_switches_on_group_by() {
        let snap = ktstr::ctprof::CtprofSnapshot::default();
        for (axis, expected_header) in [
            (ctprof_compare::GroupBy::Pcomm, "pcomm"),
            (ctprof_compare::GroupBy::Cgroup, "cgroup"),
            (ctprof_compare::GroupBy::Comm, "comm-pattern"),
            (ctprof_compare::GroupBy::CommExact, "comm"),
        ] {
            let mut out = String::new();
            write_show(
                &mut out,
                &snap,
                axis,
                &[],
                false,
                false,
                &[],
                &[],
                &[],
                &[],
                false,
            )
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
        let snap = ktstr::ctprof::CtprofSnapshot::default();
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &[],
            &[],
            &[],
            false,
        )
        .expect("write_show into String must not fail on empty snapshot");
        assert!(
            out.contains("pcomm"),
            "empty snapshot must still emit the header, got: {out}",
        );
        // Run-time metric does not appear because no thread
        // contributed it — the `for metric in CTPROF_METRICS`
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
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        let mut t_alpha = ktstr::ctprof::ThreadState::default();
        t_alpha.pcomm = "alpha".to_string();
        t_alpha.comm = "alpha-w".to_string();
        t_alpha.run_time_ns = MonotonicNs(100);
        let mut t_bravo = ktstr::ctprof::ThreadState::default();
        t_bravo.pcomm = "bravo".to_string();
        t_bravo.comm = "bravo-w".to_string();
        t_bravo.run_time_ns = MonotonicNs(500);
        let mut t_charlie = ktstr::ctprof::ThreadState::default();
        t_charlie.pcomm = "charlie".to_string();
        t_charlie.comm = "charlie-w".to_string();
        t_charlie.run_time_ns = MonotonicNs(250);
        snap.threads.push(t_alpha);
        snap.threads.push(t_bravo);
        snap.threads.push(t_charlie);

        let sort_by = vec![ctprof_compare::SortKey {
            metric: "run_time_ns",
            descending: true,
        }];
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &sort_by,
            &[],
            &[],
            &[],
            false,
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
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        // Cgroup grouping requires at least one thread carrying
        // the cgroup path so build_groups produces a bucket; the
        // secondary table renders for every cgroup_stats key
        // regardless, but the primary table needs a row to
        // surround it (write_show returns Ok early only for an
        // empty cgroup_stats map, not an empty groups map).
        let mut t = ktstr::ctprof::ThreadState::default();
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
        // from `ctprof`). Build via Default + per-field
        // assignment instead.
        let mut cgs = ktstr::ctprof::CgroupStats::default();
        cgs.cpu.usage_usec = 1_500_000;
        cgs.cpu.nr_throttled = 50;
        cgs.cpu.throttled_usec = 200;
        cgs.memory.current = one_gib;
        snap.cgroup_stats.insert("/app".to_string(), cgs);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Cgroup,
            &[],
            false,
            false,
            &[],
            &[],
            &[],
            &[],
            false,
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

    /// `write_show` with the `Tags` column included surfaces the
    /// registry's bracketed tag string in the dedicated tags
    /// column. Pins that the show plumbs `metric_tags` correctly
    /// through the column dispatch and that the rendered cell
    /// carries the bracketed tag. Mirrors the compare-side
    /// `write_diff_renders_tagged_metric_cell` integration test on
    /// the show path. The default column set deliberately omits
    /// the bracketed tag string from the `metric` cell so plain
    /// listings stay narrow.
    #[test]
    fn write_show_renders_tagged_metric_cell() {
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        let mut t = ktstr::ctprof::ThreadState::default();
        t.pcomm = "worker".to_string();
        t.comm = "w".to_string();
        t.nr_wakeups_affine = MonotonicCount(7);
        snap.threads.push(t);
        let columns = vec![
            ctprof_compare::Column::Group,
            ctprof_compare::Column::Threads,
            ctprof_compare::Column::Metric,
            ctprof_compare::Column::Tags,
            ctprof_compare::Column::Value,
        ];
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &columns,
            &[],
            &[],
            false,
        )
        .expect("write_show into String must not fail");
        assert!(
            out.contains("[cfs-only] [SCHEDSTATS]"),
            "tagged metric tags missing from rendered tags column:\n{out}",
        );
        assert!(
            out.contains("nr_wakeups_affine"),
            "tagged metric name missing from rendered show table:\n{out}",
        );
    }

    /// `write_show` emits the `## Derived metrics` section
    /// after the main per-group table. Mirrors the compare-side
    /// `write_diff_emits_derived_section` integration test.
    /// Sets the inputs for `avg_slice_ns` (run_time_ns /
    /// timeslices) on a single thread so the derivation
    /// resolves to a non-`-` cell, and asserts both the section
    /// header and the metric name surface in the output.
    #[test]
    fn write_show_emits_derived_section() {
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        let mut t = ktstr::ctprof::ThreadState::default();
        t.pcomm = "worker".to_string();
        t.comm = "w".to_string();
        t.run_time_ns = MonotonicNs(4_000);
        t.timeslices = MonotonicCount(8);
        snap.threads.push(t);
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &[],
            &[],
            &[],
            false,
        )
        .expect("write_show into String must not fail");
        assert!(
            out.contains("## Derived metrics"),
            "missing derived section header from show output:\n{out}",
        );
        assert!(
            out.contains("avg_slice_ns"),
            "missing avg_slice_ns row in derived section of show output:\n{out}",
        );
    }

    /// `write_show` with `columns=[Metric, Value]` emits exactly
    /// those two columns — the header must contain `metric` and
    /// `value` but not `threads` (or any other compare-only
    /// column name). Pins that the show-side `--columns` override
    /// reaches the renderer and overrides the default
    /// `(group, threads, metric, value)` set.
    #[test]
    fn write_show_columns_override_emits_only_selected_columns() {
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        let mut t = ktstr::ctprof::ThreadState::default();
        t.pcomm = "worker".to_string();
        t.comm = "w".to_string();
        t.nr_wakeups = MonotonicCount(1);
        snap.threads.push(t);

        let columns = vec![
            ctprof_compare::Column::Metric,
            ctprof_compare::Column::Value,
        ];
        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &columns,
            &[],
            &[],
            false,
        )
        .expect("write_show into String must not fail");

        // The first line of `out` is the section heading
        // `## Primary metrics`; the table column header is the
        // first line containing the `metric` token.
        let header_line = out
            .lines()
            .find(|line| line.contains("metric") && !line.starts_with("##"))
            .unwrap_or("");
        assert!(
            header_line.contains("metric"),
            "metric column must appear in header: {header_line}",
        );
        assert!(
            header_line.contains("value"),
            "value column must appear in header: {header_line}",
        );
        assert!(
            !header_line.contains("threads"),
            "threads column must NOT appear when --columns excludes it: {header_line}",
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
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        let mut t_zulu = ktstr::ctprof::ThreadState::default();
        t_zulu.pcomm = "zulu".to_string();
        t_zulu.comm = "zulu-w".to_string();
        t_zulu.run_time_ns = MonotonicNs(999);
        let mut t_alpha = ktstr::ctprof::ThreadState::default();
        t_alpha.pcomm = "alpha".to_string();
        t_alpha.comm = "alpha-w".to_string();
        t_alpha.run_time_ns = MonotonicNs(1);
        snap.threads.push(t_zulu);
        snap.threads.push(t_alpha);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &[],
            &[],
            &[],
            false,
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

    /// `write_show` smaps section under default normalization
    /// keys by `pattern_key(&t.pcomm)` (drops the tgid) and
    /// field-sums byte counts across PIDs sharing the same
    /// pcomm pattern. Three `worker-{0,1,2}` leaders (each with
    /// its own ephemeral tgid) collapse into one `worker-{N}`
    /// row per smaps_rollup field, with values summed.
    #[test]
    fn write_show_smaps_default_normalization_collapses_pids() {
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        for (pcomm, tgid, rss) in [
            ("worker-0", 100, 1024_u64),
            ("worker-1", 200, 2048),
            ("worker-2", 300, 4096),
        ] {
            let mut t = ktstr::ctprof::ThreadState::default();
            t.tid = tgid;
            t.tgid = tgid;
            t.pcomm = pcomm.to_string();
            t.comm = pcomm.to_string();
            t.smaps_rollup_kb.insert("Rss".into(), rss);
            t.smaps_rollup_kb.insert("Pss".into(), rss / 2);
            snap.threads.push(t);
        }

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false, // no_thread_normalize: false → normalize
            false,
            &[],
            &[],
            &[],
            &[],
            false,
        )
        .expect("write_show must not fail");

        // Find the smaps section.
        let smaps_at = out
            .find("## smaps_rollup")
            .expect("smaps section must render");
        let after = &out[smaps_at..];
        // Normalized key surfaces; literal per-PID keys do not.
        assert!(
            after.contains("worker-{N}"),
            "show smaps must collapse to `worker-{{N}}` under default normalization:\n{after}",
        );
        for literal in &["worker-0[100]", "worker-1[200]", "worker-2[300]"] {
            assert!(
                !after.contains(literal),
                "literal per-PID key {literal:?} must NOT appear under default \
                 normalization:\n{after}",
            );
        }
        // Summed Rss = (1024 + 2048 + 4096) * 1024 = 7168 KiB =
        // 7168*1024 bytes = 7340032 B → step-up to MiB
        // (1024*1024 = 1048576), so 7340032 / 1048576 ≈ 7.000 MiB.
        assert!(
            after.contains("7.000MiB"),
            "summed Rss must render as `7.000MiB` (3-PID collapse via field-sum):\n{after}",
        );
    }

    /// `write_show` smaps section under `--no-thread-normalize`
    /// preserves the literal `pcomm[tgid]` key shape so each
    /// PID stays attributable. Same fixture as the default-mode
    /// test, but with `no_thread_normalize: true` — the section
    /// emits three rows (one per PID), not one collapsed row.
    #[test]
    fn write_show_smaps_literal_mode_preserves_per_pid_keys() {
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        for (pcomm, tgid, rss) in [
            ("worker-0", 100, 1024_u64),
            ("worker-1", 200, 2048),
            ("worker-2", 300, 4096),
        ] {
            let mut t = ktstr::ctprof::ThreadState::default();
            t.tid = tgid;
            t.tgid = tgid;
            t.pcomm = pcomm.to_string();
            t.comm = pcomm.to_string();
            t.smaps_rollup_kb.insert("Rss".into(), rss);
            snap.threads.push(t);
        }

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            true, // no_thread_normalize: true → literal
            false,
            &[],
            &[],
            &[],
            &[],
            false,
        )
        .expect("write_show must not fail");

        let smaps_at = out
            .find("## smaps_rollup")
            .expect("smaps section must render");
        let after = &out[smaps_at..];
        // Literal per-PID keys surface; normalized form does not.
        for literal in &["worker-0[100]", "worker-1[200]", "worker-2[300]"] {
            assert!(
                after.contains(literal),
                "literal per-PID key {literal:?} must surface under \
                 --no-thread-normalize:\n{after}",
            );
        }
        assert!(
            !after.contains("worker-{N}"),
            "normalized `worker-{{N}}` key must NOT surface under \
             --no-thread-normalize:\n{after}",
        );
    }

    /// `write_show` smaps section sorts process rows by Rss
    /// descending, mirroring the compare-side sort. Two
    /// processes — `bash` (literal pcomm, large Rss) and
    /// `worker-{N}` collapsed bucket (small per-thread Rss but
    /// summed across PIDs). The heavier process must render
    /// first regardless of alphabetical key order.
    #[test]
    fn write_show_smaps_orders_by_rss_descending() {
        let mut snap = ktstr::ctprof::CtprofSnapshot::default();
        // bash: 100 MiB Rss.
        let mut bash = ktstr::ctprof::ThreadState::default();
        bash.tid = 1;
        bash.tgid = 1;
        bash.pcomm = "bash".to_string();
        bash.comm = "bash".to_string();
        bash.smaps_rollup_kb.insert("Rss".into(), 100 * 1024);
        bash.smaps_rollup_kb.insert("Pss".into(), 50 * 1024);
        snap.threads.push(bash);
        // zulu: 1 MiB Rss only — much smaller, but
        // alphabetically AFTER bash.
        let mut zulu = ktstr::ctprof::ThreadState::default();
        zulu.tid = 2;
        zulu.tgid = 2;
        zulu.pcomm = "zulu".to_string();
        zulu.comm = "zulu".to_string();
        zulu.smaps_rollup_kb.insert("Rss".into(), 1024);
        zulu.smaps_rollup_kb.insert("Pss".into(), 512);
        snap.threads.push(zulu);

        let mut out = String::new();
        write_show(
            &mut out,
            &snap,
            ctprof_compare::GroupBy::Pcomm,
            &[],
            false,
            false,
            &[],
            &[],
            &[],
            &[],
            false,
        )
        .expect("write_show must not fail");

        let smaps_at = out
            .find("## smaps_rollup")
            .expect("smaps section must render");
        let after = &out[smaps_at..];
        let bash_pos = after.find("bash").expect("bash key must surface");
        let zulu_pos = after.find("zulu").expect("zulu key must surface");
        assert!(
            bash_pos < zulu_pos,
            "smaps must sort by Rss desc — bash (100 MiB) ahead of \
             zulu (1 MiB) regardless of alphabetical order; \
             bash@{bash_pos} zulu@{zulu_pos}\nafter:\n{after}",
        );
    }
}
