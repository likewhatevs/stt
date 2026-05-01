//! `cargo ktstr export` — package a registered test as a self-extracting
//! `.run` file that reproduces the scenario on bare metal without a VM.
//!
//! [`export_test`] is the entry point invoked from
//! `cargo-ktstr.rs`. It locates the named test in the [`KTSTR_TESTS`]
//! distributed slice, gathers the binaries it needs (the running
//! ktstr binary, the scheduler binary, and per-test include files),
//! tarballs them with gzip, and emits a single shell script:
//!
//! ```text
//! #!/bin/bash
//! ... preamble: root check, prereq check, sched_ext conflict check,
//!     topology check, arg parsing, mktemp+trap, archive extract,
//!     scheduler launch, test run ...
//! __ARCHIVE__
//! <base64-encoded gzipped tarball>
//! ```
//!
//! The result is `chmod +x` so the operator can `./repro.run`
//! directly on a target host. `ktstr run --ktstr-test-fn <name>` is
//! the same dispatch the in-guest test harness already uses
//! (`test_support::eval` invokes it after VM boot), so the bare-metal
//! path reuses every existing test entry — no separate registry, no
//! rebuilt scenarios.
//!
//! # Why bare-metal repro?
//!
//! The framework's primary execution path runs every test inside a
//! KVM VM. That gets us deterministic topology, fast spin-up, and
//! kernel/scheduler isolation; it also abstracts away from real
//! hardware. When a test fails on bare metal but passes in the VM
//! (or vice versa) the operator wants to bisect. A self-contained
//! `.run` file means they can hand the failing test to any host with
//! a compatible kernel and topology, run it without re-building the
//! workspace, and capture the output through ordinary stdout/stderr
//! channels.
//!
//! # Out of scope
//!
//! - `host_only` tests: they orchestrate cargo invocations and nested
//!   VMs themselves; running them outside the framework's harness
//!   isn't useful.
//! - `bpf_map_write` tests: they need the framework's runtime
//!   probe-based map-write surface, not yet replicated outside the
//!   VM dispatch.
//! - `KernelBuiltin` schedulers: they activate via shell commands
//!   (`enable` / `disable` slots on the spec) rather than launching a
//!   userspace binary. The preamble doesn't generate those commands
//!   in v1; export rejects the variant with an actionable error.
//!
//! # Include-file directories
//!
//! The framework's full include-file resolver (re-exported as
//! [`crate::cli::resolve_include_files`]) walks directories
//! recursively and produces an `archive_path/host_path` map that
//! preserves directory structure. Export uses a simpler subset:
//! every included entry must be a regular file, and the archive
//! layout flattens by basename to `include/<basename>`. Directory
//! specs error with EISDIR. Recursive directory packaging is a v2
//! enhancement.
//!
//! # Bash-only
//!
//! The preamble's heredoc shebang names `/bin/bash` and uses
//! features bash carries — indexed-array syntax
//! (`RUN_ARGS=(...)`, `${RUN_ARGS[@]}` expansion) and
//! `set -o pipefail` (the `o`-form syntax). Bourne / dash /
//! busybox sh would mis-parse the script; the operator must run
//! on a host with bash installed.

use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use flate2::Compression;
use flate2::write::GzEncoder;

use crate::test_support::{KtstrTestEntry, SchedulerSpec, find_test, resolve_scheduler};

/// Build a self-extracting `.run` file for the given test.
///
/// `test_name` must match a `#[ktstr_test]` registration's `name`
/// field exactly (case-sensitive). Use `cargo nextest list` to
/// enumerate names; strip the `<binary>::` prefix the way
/// `cargo ktstr show-thresholds` does.
///
/// `output` is the destination path. `None` defaults to
/// `<test_name>.run` in the current directory. The output file is
/// written with mode 0o755 so the operator can invoke it directly.
pub fn export_test(test_name: &str, output: Option<PathBuf>) -> Result<()> {
    let entry = find_test(test_name)
        .ok_or_else(|| anyhow::anyhow!("no registered test named '{test_name}'"))?;

    if entry.host_only {
        bail!(
            "test '{test_name}' is host_only — it orchestrates cargo / nested VMs \
             from inside the test body and cannot be reproduced outside the \
             framework harness. host_only tests are out of scope for export."
        );
    }
    if !entry.bpf_map_write.is_empty() {
        bail!(
            "test '{test_name}' uses bpf_map_write — runtime BPF map writes are \
             driven by the framework's host-side probe machinery, which is not \
             reproduced bare-metal. bpf_map_write tests are out of scope for v1 \
             export."
        );
    }
    // KernelBuiltin schedulers don't ship a userspace binary; they
    // activate via shell commands stored on the spec's `enable` /
    // `disable` slots. The framework runs those commands in the VM
    // around the scheduler binary launch (eval.rs builds
    // sched_enable_cmds / sched_disable_cmds on the VmBuilder). The
    // preamble in v1 does not generate equivalent shell commands —
    // running the .run file on a host without applying those
    // settings would silently mis-launch the scheduler. Reject with
    // an actionable diagnostic.
    if let Some(SchedulerSpec::KernelBuiltin { .. }) = entry.scheduler.scheduler_binary() {
        bail!(
            "test '{test_name}' uses a KernelBuiltin scheduler — it activates via \
             host-side shell commands (`enable` / `disable` slots) rather than a \
             userspace binary. The export preamble does not yet emit those \
             commands; KernelBuiltin export is out of scope for v1."
        );
    }

    let ktstr_binary = std::env::current_exe()
        .context("locate currently running cargo-ktstr binary via /proc/self/exe")?;

    let scheduler_path = resolve_scheduler_for_export(entry)?;
    let include_files = resolve_include_files(entry)?;

    let output_path = output.unwrap_or_else(|| PathBuf::from(format!("{test_name}.run")));

    let archive = build_archive(&ktstr_binary, scheduler_path.as_deref(), &include_files)
        .context("build embedded gzip tarball")?;

    let preamble = generate_preamble(entry, scheduler_path.is_some());

    write_runfile(&output_path, &preamble, &archive)
        .with_context(|| format!("write runfile to {}", output_path.display()))?;

    eprintln!(
        "wrote {} ({} bytes archive, {} include files)",
        output_path.display(),
        archive.len(),
        include_files.len()
    );
    Ok(())
}

/// Resolve the scheduler binary for an entry, returning `None` for
/// EEVDF / kernel-builtin payloads (which don't ship a binary).
///
/// Reuses [`crate::test_support::eval::resolve_scheduler`] so the
/// resolution cascade matches the in-guest path: `KTSTR_SCHEDULER`
/// env → sibling exe → target/debug → target/release → auto-build.
/// The cascade walks both target dirs regardless of which build
/// profile invoked cargo-ktstr.
fn resolve_scheduler_for_export(entry: &KtstrTestEntry) -> Result<Option<PathBuf>> {
    let Some(spec) = entry.scheduler.scheduler_binary() else {
        // Binary-kind payload — no scheduler.
        return Ok(None);
    };
    let (path, _source) = resolve_scheduler(spec)
        .with_context(|| format!("resolve scheduler binary for test '{}'", entry.name))?;
    Ok(path)
}

/// Resolve every `all_include_files()` spec to a host-side path.
///
/// The framework's full PATH / directory-walking resolver lives at
/// [`crate::cli::resolve_include_files`] and returns an
/// `archive_path/host_path` map that preserves recursive directory
/// structure. Export uses a deliberately simpler subset:
///   - explicit absolute paths → use as-is when they exist
///   - explicit relative paths (containing `/` or starting with `.`)
///     → relative to current dir
///   - bare names → search `PATH`
///   - directories → reject with EISDIR (export packs files only;
///     recursive packaging is a v2 enhancement)
///
/// The simpler layout (flat `include/<basename>`) keeps the
/// extracted .run tree predictable for the operator, at the cost of
/// not handling tests whose include specs name directories.
///
/// Missing files are surfaced as a hard error so the operator can
/// fix the include spec rather than discovering the gap on the
/// target host.
fn resolve_include_files(entry: &KtstrTestEntry) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for spec in entry.all_include_files() {
        let path = if spec.starts_with('/')
            || spec.starts_with("./")
            || spec.starts_with("../")
            || spec.contains('/')
        {
            PathBuf::from(spec)
        } else {
            // Bare name — search PATH.
            search_path_for(spec).ok_or_else(|| {
                anyhow::anyhow!(
                    "include file '{spec}' not found in PATH (test \
                     declared it but the host doesn't have it; install or \
                     supply an absolute path)"
                )
            })?
        };
        if !path.exists() {
            bail!("include file does not exist on host: {}", path.display());
        }
        // Reject directories explicitly: export packs files only,
        // and a directory spec would silently fail later inside
        // `append_file`'s `std::fs::read` with a less-actionable
        // error message.
        if path.is_dir() {
            bail!(
                "include file '{}' is a directory — export packs regular files \
                 only. Recursive directory packaging is a v2 enhancement; for \
                 now, list each file individually in the test's \
                 `include_files` slot.",
                path.display()
            );
        }
        out.push(path);
    }
    Ok(out)
}

/// Search `PATH` for an executable named `name`. Returns the first
/// match. Mirrors the simplest case of the framework's PATH resolver
/// — sufficient for export's needs since tests typically declare
/// either bare standard tools (stress-ng, schbench) or paths to
/// build artifacts.
///
/// A "match" requires the candidate to be (a) a regular file and
/// (b) executable (any of the user/group/other execute bits set).
/// Without the executable check, a non-binary file with a colliding
/// name (e.g. a `stress-ng` documentation file in a PATH dir) would
/// be picked up first and silently fail at .run time when the guest
/// tries to exec it.
fn search_path_for(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if !candidate.is_file() {
            continue;
        }
        let executable = candidate
            .metadata()
            .map(|m| m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false);
        if executable {
            return Some(candidate);
        }
    }
    None
}

/// Tar+gzip the binaries into an in-memory blob.
///
/// Layout inside the archive:
///   - `ktstr` — the runner binary (the one calling
///     [`export_test`])
///   - `scheduler` — the scheduler binary (when present)
///   - `include/<basename>` — every include file, flattened by
///     basename. Collisions on basename are not allowed.
///
/// Permissions: every entry is chmod 0755. The `.run` extractor
/// preserves these, so the operator can invoke them directly under
/// `$DIR/ktstr` / `$DIR/scheduler` without re-chmod.
fn build_archive(ktstr: &Path, scheduler: Option<&Path>, includes: &[PathBuf]) -> Result<Vec<u8>> {
    let buf: Vec<u8> = Vec::new();
    let gz = GzEncoder::new(buf, Compression::default());
    let mut tar = tar::Builder::new(gz);

    append_file(&mut tar, ktstr, "ktstr")?;
    if let Some(s) = scheduler {
        append_file(&mut tar, s, "scheduler")?;
    }

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for inc in includes {
        let name = inc
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("include file has no basename: {}", inc.display()))?
            .to_string_lossy()
            .into_owned();
        if !seen.insert(name.clone()) {
            bail!(
                "include-file basename collision: two specs both flatten to \
                 'include/{name}'. Rename one or use distinct paths."
            );
        }
        let archive_name = format!("include/{name}");
        append_file(&mut tar, inc, &archive_name)?;
    }

    let gz = tar.into_inner().context("finalise tar stream")?;
    let blob = gz.finish().context("finalise gzip stream")?;
    Ok(blob)
}

/// Append one host file at `host_path` into `tar` under `archive_name`.
/// Forces mode 0o755 so the extracted entry is executable on the
/// target host. Regenerates the tar header rather than reusing the
/// host's path metadata (which could leak environment-specific
/// information into the published artifact).
fn append_file<W: Write>(
    tar: &mut tar::Builder<W>,
    host_path: &Path,
    archive_name: &str,
) -> Result<()> {
    let bytes = std::fs::read(host_path)
        .with_context(|| format!("read {} for archive", host_path.display()))?;
    crate::tar_util::pack_tar_entry(
        tar,
        archive_name,
        0o755,
        bytes.len() as u64,
        bytes.as_slice(),
    )
    .with_context(|| format!("append {archive_name} to tar"))?;
    Ok(())
}

/// Generate the bash preamble. The output is a complete shell script
/// up to (but not including) the `__ARCHIVE__` marker; [`write_runfile`]
/// concatenates the preamble, the marker line, and the base64 archive.
///
/// The preamble is verbose by default: a banner identifying test +
/// scheduler + git provenance, every prereq/conflict check spelled
/// out with actionable error text, and a security-posture line
/// warning the operator to inspect the script (everything before
/// `__ARCHIVE__`) before running on a system they do not control.
/// `--quiet` suppresses the banner only — error paths still print
/// so a failing repro is never silent.
fn generate_preamble(entry: &KtstrTestEntry, has_scheduler: bool) -> String {
    let topology = entry.topology;
    let need_llcs = topology.llcs;
    let need_cores = topology.cores_per_llc;
    let need_threads = topology.threads_per_core;
    let need_numa = topology.numa_nodes;

    // Compose scheduler args = required_flags expansion
    // (`scheduler.flag_args(name)`) followed by entry.extra_sched_args.
    // Production dispatch (dispatch.rs:1182, eval.rs after
    // active_flags resolution) does the same: required_flags are
    // turned into CLI args via Scheduler::flag_args and prepended
    // to extra_sched_args. Without this, a test that declares
    // required_flags would launch the scheduler bare and fail to
    // exercise the gated code path.
    let mut sched_arg_tokens: Vec<String> = Vec::new();
    for flag in entry.required_flags {
        if let Some(args) = entry.scheduler.flag_args(flag) {
            for a in args {
                sched_arg_tokens.push(shell_quote(a));
            }
        }
    }
    for a in entry.extra_sched_args {
        sched_arg_tokens.push(shell_quote(a));
    }
    let sched_args_joined = sched_arg_tokens.join(" ");

    // Defensive shell-quoting on every interpolated runtime value.
    // The names come from compile-time const slots that are
    // `[A-Za-z0-9_-]+` in practice today, but interpolating
    // unquoted lets a future producer regression land an
    // unescaped value with shell metacharacters in the preamble.
    // Quoting at the producer is cheap and matches the same defense
    // applied to extra_sched_args / flag_args above.
    let test_name = shell_quote(entry.name);
    let scheduler_name = shell_quote(entry.scheduler.scheduler_name());
    let git_hash = shell_quote(&git_provenance());

    let duration_secs = entry.duration.as_secs();
    let watchdog_secs = entry.watchdog_timeout.as_secs();

    let scheduler_launch = if has_scheduler {
        format!(
            r#"
# --- scheduler launch ---
echo "ktstr export: launching scheduler $KTSTR_SCHED_NAME"
"$DIR/scheduler" {sched_args_joined} &
SCHED_PID=$!

# Wait up to 10s for the scheduler to attach. The kernel's sysfs
# layout exposes attach state under two files; both are accepted
# so the wait loop works on every kernel that ships sched_ext:
#   - `/sys/kernel/sched_ext/root/ops` — non-empty when a scheduler
#     is currently attached. Present on every kernel revision that
#     has sched_ext, but the path moved structurally between early
#     6.x revisions and the upstream-stabilized layout. Treat the
#     file's absence as "no scheduler attached" rather than an
#     error; the secondary check below catches stabilized kernels.
#   - `/sys/kernel/sched_ext/state` (introduced upstream in 6.12)
#     reads `enabled` once a scheduler attaches, `disabled`
#     otherwise. Use as the primary signal where available; it has
#     a stable wire format across kernel versions.
# Bail if the scheduler exits before attaching, or if the timeout
# elapses while the scheduler is still alive but unattached.
ATTACHED=""
for _ in $(seq 1 100); do
    if ! kill -0 "$SCHED_PID" 2>/dev/null; then
        echo "error: scheduler $KTSTR_SCHED_NAME exited before attaching" >&2
        wait "$SCHED_PID" || true
        exit 1
    fi
    if [ -r /sys/kernel/sched_ext/state ]; then
        STATE=$(cat /sys/kernel/sched_ext/state 2>/dev/null || true)
        if [ "$STATE" = "enabled" ]; then
            ATTACHED="$STATE"
            break
        fi
    fi
    if [ -f /sys/kernel/sched_ext/root/ops ]; then
        OPS=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || true)
        if [ -n "$OPS" ]; then
            ATTACHED="$OPS"
            break
        fi
    fi
    sleep 0.1
done
if [ -z "$ATTACHED" ]; then
    echo "error: scheduler $KTSTR_SCHED_NAME launched but did not attach within 10s" >&2
    echo "       (process is still alive; check kernel log for BPF verifier or load errors)" >&2
    exit 1
fi
"#
        )
    } else {
        // Binary-kind / EEVDF payload — no scheduler.
        String::new()
    };

    format!(
        r#"#!/bin/bash
# Generated by `cargo ktstr export`. Do not edit; regenerate to update.
set -euo pipefail

# --- frozen test specification ---
KTSTR_TEST_NAME={test_name}
KTSTR_SCHED_NAME={scheduler_name}
KTSTR_GIT_HASH={git_hash}
NEED_LLCS={need_llcs}
NEED_CORES_PER_LLC={need_cores}
NEED_THREADS_PER_CORE={need_threads}
NEED_NUMA_NODES={need_numa}
TEST_DURATION_SECS={duration_secs}
TEST_WATCHDOG_SECS={watchdog_secs}

QUIET=0
DURATION_OVERRIDE=""
WATCHDOG_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quiet) QUIET=1; shift ;;
        --duration) DURATION_OVERRIDE="$2"; shift 2 ;;
        --watchdog-timeout) WATCHDOG_OVERRIDE="$2"; shift 2 ;;
        --cpus|--topology|--affinity)
            echo "error: --$1 is frozen for repro fidelity. Re-export to change." >&2
            exit 1 ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--quiet] [--duration SECS] [--watchdog-timeout SECS]

Reproduces ktstr test '$KTSTR_TEST_NAME' on bare metal. The script
extracts an embedded gzip tarball containing the ktstr binary and
the scheduler binary, then dispatches the test directly without
booting a VM.

Frozen (cannot be overridden):
  scheduler         $KTSTR_SCHED_NAME
  topology          $NEED_NUMA_NODES NUMA / $NEED_LLCS LLCs / $NEED_CORES_PER_LLC cores/LLC / $NEED_THREADS_PER_CORE threads/core
  scheduler args    (compiled into the script)
  --cpus, --topology, --affinity reject any override

Overridable:
  --duration SECS         workload duration (default $TEST_DURATION_SECS)
  --watchdog-timeout SECS scheduler watchdog (default $TEST_WATCHDOG_SECS)
  --quiet                 suppress the banner (errors still print)

Requirements:
  Run as root. The script attaches a kernel BPF scheduler and sets
  up cgroup v2 subgroups; both need CAP_SYS_ADMIN.

  Host must satisfy the frozen topology (LLCs, cores per LLC,
  threads per core, NUMA nodes); the script's topology check bails
  with a specific "host has X, test needs Y" message if not.

  /sys/kernel/sched_ext must exist (kernel built with
  CONFIG_SCHED_CLASS_EXT) and no other sched_ext scheduler may be
  attached.

Exit codes:
  0   test passed
  1   prerequisite or topology check failed, scheduler attach
      failed, or test failed
EOF
            exit 0 ;;
        *) echo "error: unknown arg '$1' (use --help)" >&2; exit 1 ;;
    esac
done

if [ "$QUIET" != "1" ]; then
    cat <<EOF
ktstr export: test=$KTSTR_TEST_NAME scheduler=$KTSTR_SCHED_NAME git=$KTSTR_GIT_HASH
Generated by cargo ktstr export. This script attaches a kernel BPF scheduler
and runs as root. Inspect this script (everything before __ARCHIVE__) before
running on a system you do not control.
EOF
fi

# --- root check ---
if [ "$(id -u)" != "0" ]; then
    echo "error: must run as root (need CAP_SYS_ADMIN for sched_ext + cgroup ops)" >&2
    exit 1
fi

# --- prereq checks ---
if [ ! -d /sys/kernel/sched_ext ]; then
    echo "error: kernel lacks sched_ext support (no /sys/kernel/sched_ext)" >&2
    exit 1
fi
if [ ! -d /sys/fs/cgroup ]; then
    echo "error: cgroup2 not mounted at /sys/fs/cgroup" >&2
    exit 1
fi
if ! grep -q '^cgroup2 /sys/fs/cgroup ' /proc/mounts; then
    echo "error: /sys/fs/cgroup is not a cgroup2 mount" >&2
    exit 1
fi

# --- sched_ext conflict check ---
# Mirror the attach-detection logic below: prefer
# /sys/kernel/sched_ext/state (stabilized in 6.12) when readable,
# fall back to /sys/kernel/sched_ext/root/ops otherwise. Either
# file reporting an attached scheduler aborts here so we don't
# silently displace someone else's running scheduler.
if [ -r /sys/kernel/sched_ext/state ]; then
    CURRENT_STATE=$(cat /sys/kernel/sched_ext/state 2>/dev/null || true)
    if [ "$CURRENT_STATE" = "enabled" ]; then
        CURRENT_OPS=""
        if [ -f /sys/kernel/sched_ext/root/ops ]; then
            CURRENT_OPS=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || true)
        fi
        echo "error: another sched_ext scheduler is already attached (state=enabled, ops=${{CURRENT_OPS:-unknown}})." >&2
        echo "       Detach it before running this repro (e.g. kill its supervisor)." >&2
        exit 1
    fi
elif [ -f /sys/kernel/sched_ext/root/ops ]; then
    CURRENT=$(cat /sys/kernel/sched_ext/root/ops 2>/dev/null || true)
    if [ -n "$CURRENT" ]; then
        echo "error: another sched_ext scheduler '$CURRENT' is already attached." >&2
        echo "       Detach it before running this repro (e.g. kill its supervisor)." >&2
        exit 1
    fi
fi

# --- topology check ---
# LLC count: find the highest cache-index level under cpu0 (index3
# on most x86, but skylake-x has a dedicated L4 at index4 and ARM
# machines vary). Sum distinct shared_cpu_lists at that level.
HIGHEST_INDEX=$(ls -d /sys/devices/system/cpu/cpu0/cache/index* 2>/dev/null \
    | sort -V | tail -n1 || true)
if [ -n "$HIGHEST_INDEX" ]; then
    HIGHEST_LEVEL=$(basename "$HIGHEST_INDEX")
    HOST_LLCS=$(ls -d /sys/devices/system/cpu/cpu*/cache/$HIGHEST_LEVEL 2>/dev/null \
        | xargs -I{{}} cat {{}}/shared_cpu_list 2>/dev/null \
        | sort -u | wc -l)
else
    HOST_LLCS=0
fi
HOST_NUMA=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l || echo 0)
[ "$HOST_NUMA" -lt 1 ] && HOST_NUMA=1

# Cores per LLC: count distinct core_id values among cpus that share
# the highest-level cache with cpu0. threads per core: count cpus
# that share the same core_id within one LLC.
if [ -n "$HIGHEST_INDEX" ]; then
    CPU0_LLC=$(cat "$HIGHEST_INDEX/shared_cpu_list" 2>/dev/null || echo "")
else
    CPU0_LLC=""
fi
HOST_CORES_PER_LLC=0
HOST_THREADS_PER_CORE=0
if [ -n "$CPU0_LLC" ]; then
    # Expand cpu list ranges (e.g. "0-3,8-11") into individual ids.
    CPU_IDS=$(echo "$CPU0_LLC" | tr ',' '\n' | while read range; do
        if [ -z "$range" ]; then continue; fi
        if echo "$range" | grep -q '-'; then
            start=$(echo "$range" | cut -d- -f1)
            end=$(echo "$range" | cut -d- -f2)
            seq "$start" "$end"
        else
            echo "$range"
        fi
    done)
    HOST_CORES_PER_LLC=$(for id in $CPU_IDS; do
        cat "/sys/devices/system/cpu/cpu$id/topology/core_id" 2>/dev/null || echo
    done | sort -u | wc -l)
    CPU0_CORE=$(cat /sys/devices/system/cpu/cpu0/topology/core_id 2>/dev/null || echo)
    if [ -n "$CPU0_CORE" ]; then
        HOST_THREADS_PER_CORE=$(for id in $CPU_IDS; do
            this_core=$(cat "/sys/devices/system/cpu/cpu$id/topology/core_id" 2>/dev/null || echo)
            if [ "$this_core" = "$CPU0_CORE" ]; then echo "$id"; fi
        done | wc -l)
    fi
fi

if [ "$HOST_LLCS" = "0" ]; then
    echo "warning: could not detect host LLC count from sysfs (no cache/index* found for cpu0); the topology check below will fail" >&2
fi
if [ "$HOST_LLCS" -lt "$NEED_LLCS" ]; then
    echo "error: host has $HOST_LLCS LLCs, test needs $NEED_LLCS" >&2
    exit 1
fi
if [ "$HOST_NUMA" -lt "$NEED_NUMA_NODES" ]; then
    echo "error: host has $HOST_NUMA NUMA nodes, test needs $NEED_NUMA_NODES" >&2
    exit 1
fi
if [ "$HOST_CORES_PER_LLC" -gt 0 ] && [ "$HOST_CORES_PER_LLC" -lt "$NEED_CORES_PER_LLC" ]; then
    echo "error: host has $HOST_CORES_PER_LLC cores per LLC, test needs $NEED_CORES_PER_LLC" >&2
    exit 1
fi
if [ "$HOST_THREADS_PER_CORE" -gt 0 ] && [ "$HOST_THREADS_PER_CORE" -lt "$NEED_THREADS_PER_CORE" ]; then
    echo "error: host has $HOST_THREADS_PER_CORE threads per core, test needs $NEED_THREADS_PER_CORE" >&2
    exit 1
fi

# --- extract embedded archive ---
DIR=$(mktemp -d -t ktstr-export-XXXXXX)
chmod 700 "$DIR"
# The ktstr in-process dispatch creates its cgroup tree under
# /sys/fs/cgroup/ktstr — the export-relevant path goes through the
# ctor early-dispatch into `test_support::probe::build_dispatch_ctx_parts`
# which calls `test_support::args::resolve_cgroup_root` (args.rs:111
# fallback), and the in-VM init follows the same convention.
# Capture the path here so the trap teardown can clean any subgroups
# the dispatch created. The rmdir must walk depth-first because
# cgroup v2 forbids rmdir on a subtree that still contains child
# groups.
#
# WARNING: this cleanup removes ALL subgroups under
# /sys/fs/cgroup/ktstr, including those created by concurrent
# ktstr processes. Do not run multiple ktstr workloads on the same
# host simultaneously.
KTSTR_CGROUP_PARENT="/sys/fs/cgroup/ktstr"
SCHED_PID=""
cleanup() {{
    if [ -n "$SCHED_PID" ]; then
        kill "$SCHED_PID" 2>/dev/null || true
        wait "$SCHED_PID" 2>/dev/null || true
    fi
    rm -rf "$DIR"
    # Cgroup teardown: depth-first rmdir over every subgroup the
    # test created. cgroup v2's interface files (cgroup.procs,
    # cgroup.controllers, ...) are auto-removed when their parent
    # directory rmdirs, so a recursive `rm -rf` is wrong (would
    # ENOTEMPTY on every interior node). `find -depth` visits
    # leaves before parents; rmdir succeeds at each step because
    # children are gone. Errors swallowed via `2>/dev/null` so a
    # cleanup race with another tool doesn't bleed into the test
    # exit status.
    if [ -d "$KTSTR_CGROUP_PARENT" ]; then
        find "$KTSTR_CGROUP_PARENT" -mindepth 1 -depth -type d \
            -exec rmdir {{}} + 2>/dev/null || true
        rmdir "$KTSTR_CGROUP_PARENT" 2>/dev/null || true
    fi
}}
trap cleanup EXIT

# Decode embedded base64 archive (everything after __ARCHIVE__).
sed -n '/^__ARCHIVE__$/,$p' "$0" | tail -n+2 | base64 -d | tar xz -C "$DIR"

if [ ! -x "$DIR/ktstr" ]; then
    echo "error: extracted ktstr binary missing or not executable" >&2
    exit 1
fi
{scheduler_launch}
# --- run the test ---
# `--ktstr-test-fn $KTSTR_TEST_NAME` is intercepted by the ktstr
# binary's `#[ctor::ctor] ktstr_test_early_dispatch` (in
# `src/test_support/dispatch.rs`), which fires from `.init_array`
# BEFORE `main()` runs. The ctor reads the argv directly via
# `extract_test_fn_arg` and dispatches via
# `maybe_dispatch_vm_test_with_args` (in
# `src/test_support/probe.rs`) which calls `(entry.func)(&ctx)`
# directly, then exits the process on completion. The leading
# `"run"` token is cosmetic — it's never parsed because the ctor
# exits before clap sees it. This early-dispatch path is the
# contract for in-process repro and is load-bearing: a future
# refactor that moves dispatch out of the ctor must keep an
# equivalent argv-intercept path in place, or this preamble must
# change to match the new dispatch shape.
#
# IMPORTANT: do NOT use `exec` here. `exec` replaces the bash
# shell with the ktstr binary and DESTROYS the EXIT trap before
# it can fire — leaking the scheduler PID, the tempdir, and the
# cgroup tree. Run as a child and forward the exit code so the
# trap fires on bash exit.
RUN_ARGS=("run" "--ktstr-test-fn" "$KTSTR_TEST_NAME")
if [ -n "$DURATION_OVERRIDE" ]; then
    RUN_ARGS+=("--duration" "$DURATION_OVERRIDE")
fi
if [ -n "$WATCHDOG_OVERRIDE" ]; then
    RUN_ARGS+=("--watchdog-timeout" "$WATCHDOG_OVERRIDE")
fi
# Disable errexit just for the ktstr invocation so a non-zero
# exit from the test (the legitimate "test failed" outcome)
# propagates as our exit code instead of triggering set -e and
# bypassing the cleanup. The `|| true` would also keep going,
# but `set +e` makes the intent explicit.
set +e
"$DIR/ktstr" "${{RUN_ARGS[@]}}"
EXIT_CODE=$?
set -e
exit $EXIT_CODE
"#
    )
}

/// Best-effort git provenance: the project HEAD short hex, or
/// `"unknown"` when not in a git checkout. Stamped into the
/// preamble's banner so an operator running an old `.run` can tell
/// what code was packaged.
///
/// Uses gix in-process rather than shelling out to `git rev-parse`.
/// Same shape as [`crate::fetch::inspect_local_source_state`]: walk
/// up from the current directory with `gix::discover`, read the head
/// id, format and truncate. No process fork, no PATH dependency —
/// the export pipeline never depends on a `git` binary being
/// installed on the host running `cargo ktstr export`.
fn git_provenance() -> String {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| gix::discover(&cwd).ok())
        .and_then(|repo| {
            // `head_id()` returns an Id<'_> borrowing `repo`, so format
            // and truncate to an owned String inside the same scope as
            // `repo` to satisfy the borrow checker. Mirrors the
            // pattern at fetch.rs:1016-1017.
            repo.head_id()
                .ok()
                .map(|id| format!("{id}").chars().take(7).collect::<String>())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Single-quote a shell argument. Embedded single quotes are
/// terminated, escaped via `'\''`, and re-opened. Sufficient for
/// passing arbitrary `extra_sched_args` strings through the
/// preamble's word-split positional context.
///
/// Empty input is quoted to `''` rather than left as the empty
/// string. An unquoted empty arg word-splits to nothing in bash,
/// silently dropping the slot — quoting preserves the empty
/// positional argument so the scheduler's argv index is preserved.
fn shell_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    if !s.contains('\'')
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-+=/:".contains(c))
    {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if c == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(c);
        }
    }
    out.push('\'');
    out
}

/// Write the final `.run` file: preamble bytes, the `__ARCHIVE__`
/// marker line, then the base64-encoded archive split into 76-column
/// lines (POSIX-friendly width). Sets executable mode 0o755.
fn write_runfile(path: &Path, preamble: &str, archive: &[u8]) -> Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o755)
        .open(path)
        .with_context(|| format!("open {} for write", path.display()))?;

    f.write_all(preamble.as_bytes()).context("write preamble")?;
    f.write_all(b"__ARCHIVE__\n")
        .context("write archive marker")?;

    let encoded = BASE64.encode(archive);
    // Split into 76-char lines so the file works through legacy
    // text-only transports (email MIME, some line editors).
    for chunk in encoded.as_bytes().chunks(76) {
        f.write_all(chunk).context("write base64 chunk")?;
        f.write_all(b"\n").context("write newline")?;
    }
    f.sync_all().context("fsync runfile")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_preserves_safe_strings() {
        assert_eq!(shell_quote("simple"), "simple");
        assert_eq!(shell_quote("--foo=bar"), "--foo=bar");
        assert_eq!(shell_quote("/usr/bin/foo"), "/usr/bin/foo");
        assert_eq!(shell_quote("a.b-c_d"), "a.b-c_d");
    }

    #[test]
    fn shell_quote_wraps_special_chars() {
        // Spaces, quotes, semicolons all force quoting.
        assert_eq!(shell_quote("with space"), "'with space'");
        assert_eq!(shell_quote("a;b"), "'a;b'");
        assert_eq!(shell_quote("$VAR"), "'$VAR'");
    }

    #[test]
    fn shell_quote_escapes_embedded_single_quotes() {
        // POSIX-standard quote-escape pattern: terminate, escape,
        // re-open.
        assert_eq!(shell_quote("don't"), "'don'\\''t'");
    }

    #[test]
    fn shell_quote_empty_string_yields_quoted_empty() {
        // Unquoted empty arg word-splits to nothing in bash, dropping
        // the positional slot. Quoting preserves it as an empty
        // argument.
        assert_eq!(shell_quote(""), "''");
    }

    /// Tab (`\t`) is whitespace under bash IFS — unquoted, the
    /// arg would split on the tab. Quoting preserves the literal
    /// tab byte inside the single-quoted form (single quotes are
    /// fully literal in POSIX shell, no escape interpretation).
    #[test]
    fn shell_quote_tab() {
        assert_eq!(shell_quote("a\tb"), "'a\tb'");
    }

    /// Newline must be quoted: an unquoted newline ends the
    /// command and starts a new one. Inside single quotes the
    /// literal `\n` byte is preserved verbatim.
    #[test]
    fn shell_quote_newline() {
        assert_eq!(shell_quote("a\nb"), "'a\nb'");
    }

    /// Backslash is a non-special byte INSIDE POSIX single quotes
    /// (only `'` ends the quoted region). The shell_quote
    /// implementation must preserve the backslash verbatim, NOT
    /// double it (a common bug from "escape everything" thinking).
    #[test]
    fn shell_quote_backslash() {
        assert_eq!(shell_quote(r"a\b"), r"'a\b'");
        // Trailing backslash is also fine — the closing quote
        // terminates the literal cleanly.
        assert_eq!(shell_quote(r"trail\"), r"'trail\'");
    }

    /// Unicode is preserved byte-for-byte: emoji + CJK both
    /// roundtrip through the single-quoted form. POSIX single
    /// quotes are byte-literal so multi-byte UTF-8 sequences are
    /// safe; we still wrap because the `is_ascii_alphanumeric`
    /// gate rejects non-ASCII and falls into the quoted path.
    #[test]
    fn shell_quote_unicode_emoji_and_cjk() {
        assert_eq!(shell_quote("test ✅"), "'test ✅'");
        assert_eq!(shell_quote("日本語"), "'日本語'");
        assert_eq!(shell_quote("héllo"), "'héllo'");
    }

    /// NUL byte: bash refuses NUL bytes in argv, but `shell_quote`
    /// itself must not panic — emit a quoted form and let the
    /// downstream shell reject it. `\0` inside single quotes is
    /// byte-literal, same as any other non-quote byte.
    #[test]
    fn shell_quote_null_byte() {
        let s = "a\0b";
        let q = shell_quote(s);
        assert_eq!(q, "'a\0b'");
    }

    /// Mixed quote types (`'` and `"`): single quotes are escaped
    /// via the `'\''` close-escape-reopen pattern; double quotes
    /// are byte-literal inside single quotes and pass through
    /// untouched.
    #[test]
    fn shell_quote_mixed_quote_types() {
        // Single inside, double outside: single becomes the
        // escape sequence, double is literal.
        assert_eq!(shell_quote(r#"he said "don't""#), r#"'he said "don'\''t"'"#);
    }

    /// String that already looks like a single-quoted literal
    /// must be re-quoted: the existing surrounding `'` are bytes,
    /// not metacharacters, so the wrapping pattern still applies
    /// and the embedded `'` get the close-escape-reopen treatment.
    #[test]
    fn shell_quote_already_single_quoted() {
        assert_eq!(shell_quote("'pre-quoted'"), r"''\''pre-quoted'\'''");
    }

    /// A bare single quote: edge of the escape pattern. The output
    /// must still be a valid single-quoted shell word.
    #[test]
    fn shell_quote_only_single_quote() {
        let q = shell_quote("'");
        // Must be syntactically valid POSIX: `'\''` is the canonical
        // pattern. Wrapped in outer quotes -> `''\'''`.
        assert_eq!(q, r"''\'''");
    }

    /// Carriage return is non-printable and would terminate a
    /// shell line on systems that treat `\r` as a newline (CRLF
    /// terminals). Single-quoted form preserves the byte verbatim.
    #[test]
    fn shell_quote_carriage_return() {
        assert_eq!(shell_quote("a\rb"), "'a\rb'");
        assert_eq!(shell_quote("a\r\nb"), "'a\r\nb'");
    }

    /// Consecutive embedded single quotes: each one independently
    /// triggers the close-escape-reopen pattern. Tests that the
    /// per-char loop doesn't collapse runs of `'` into a single
    /// escape.
    #[test]
    fn shell_quote_consecutive_single_quotes() {
        assert_eq!(shell_quote("a''b"), r"'a'\'''\''b'");
        assert_eq!(shell_quote("'''"), r"''\'''\'''\'''");
    }

    /// Combination: tab byte AND embedded single quote in the same
    /// input. The tab passes through verbatim (single quotes are
    /// byte-literal) and the apostrophe takes the escape path.
    #[test]
    fn shell_quote_tab_with_single_quote() {
        assert_eq!(shell_quote("a\t'b"), "'a\t'\\''b'");
    }

    /// Other low-byte control characters (vertical tab, form feed,
    /// bell, ESC). All non-printable, all byte-literal inside POSIX
    /// single quotes.
    #[test]
    fn shell_quote_low_control_bytes() {
        assert_eq!(shell_quote("\x07"), "'\x07'"); // bell
        assert_eq!(shell_quote("\x08"), "'\x08'"); // backspace
        assert_eq!(shell_quote("\x0b"), "'\x0b'"); // vertical tab
        assert_eq!(shell_quote("\x0c"), "'\x0c'"); // form feed
        assert_eq!(shell_quote("\x1b[31mred\x1b[0m"), "'\x1b[31mred\x1b[0m'");
    }

    /// Safe-set chars (`+`, `=`, `:`, `/`, `.`, `_`, `-`) at
    /// boundaries: this pins which characters bypass the quote
    /// wrap. The safe set is a positive list; any change to it
    /// flips this test.
    #[test]
    fn shell_quote_safe_set_unquoted() {
        // Each of these should pass through verbatim.
        for raw in [
            "+",
            "=",
            ":",
            "/",
            ".",
            "_",
            "-",
            "abc+def",
            "key=value",
            "ns:resource",
            "/usr/local/bin",
            "v1.0.0",
            "file_name-1.txt",
        ] {
            assert_eq!(
                shell_quote(raw),
                raw,
                "safe-set input must remain unquoted: {raw:?}"
            );
        }
    }

    /// Long string with no special chars stays unquoted; long
    /// string with a single special char anywhere triggers the
    /// full-string wrap. Pins behavior for size-conscious callers.
    #[test]
    fn shell_quote_long_strings() {
        let safe = "a".repeat(1024);
        assert_eq!(
            shell_quote(&safe),
            safe,
            "long safe-set string passes through"
        );

        let with_space = format!("{}{}", "a".repeat(512), " end");
        let q = shell_quote(&with_space);
        assert!(q.starts_with('\'') && q.ends_with('\''));
        assert_eq!(&q[1..q.len() - 1], &with_space);
    }

    /// Shell metacharacters that would otherwise be interpreted
    /// by the parser must all roundtrip through the quoted form
    /// without forking a subshell, expanding a glob, or
    /// dereferencing a variable.
    #[test]
    fn shell_quote_shell_metacharacters() {
        // Each of these would, unquoted, do something dangerous.
        for raw in [
            "a&b", "a|b", "a`b`c", "a$b", "a*b", "a?b", "a[b]c", "a{b}c", "a(b)c", "a~b", "a#b",
            "a!b",
        ] {
            let q = shell_quote(raw);
            assert!(
                q.starts_with('\'') && q.ends_with('\''),
                "metachar input must be wrapped: input={raw:?} output={q:?}"
            );
            // The byte content (between the wrapping quotes) must
            // equal the original — single quotes are byte-literal,
            // no escape interpretation, no `'` to escape in these
            // inputs.
            let inner = &q[1..q.len() - 1];
            assert_eq!(
                inner, raw,
                "metachar input must be byte-preserved inside the wrap"
            );
        }
    }

    #[test]
    fn search_path_for_finds_existing_executable() {
        // /bin/sh is executable on every supported host; PATH lookup
        // must resolve it.
        let found = search_path_for("sh");
        assert!(found.is_some(), "PATH search for `sh` returned None");
        let path = found.unwrap();
        assert!(path.is_file(), "resolved path is not a file: {path:?}");
        let mode = path.metadata().unwrap().permissions().mode();
        assert!(
            mode & 0o111 != 0,
            "resolved path is not executable: {path:?} mode={mode:o}"
        );
    }

    #[test]
    fn search_path_for_returns_none_on_missing() {
        // A name that cannot exist as a binary.
        let found = search_path_for("definitely-not-a-real-binary-xyzzy-987");
        assert!(found.is_none());
    }

    #[test]
    fn search_path_for_skips_non_executable_files() {
        // Plant a non-executable file in a temp dir, prepend that
        // dir to PATH, search for the name. The lookup must skip
        // it (mode lacks any execute bit) rather than return the
        // path.
        let tmp = tempfile::TempDir::new().expect("create temp dir");
        let dummy = tmp.path().join("dummy_non_exec");
        std::fs::write(&dummy, b"#!/bin/sh\necho hi\n").expect("write dummy");
        let mut perms = std::fs::metadata(&dummy).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&dummy, perms).expect("set non-exec perms");

        let original_path = std::env::var_os("PATH").unwrap_or_default();
        let new_path = {
            let mut paths = vec![tmp.path().to_path_buf()];
            paths.extend(std::env::split_paths(&original_path));
            std::env::join_paths(paths).expect("join paths")
        };
        // SAFETY: this test is a unit test that does not spawn
        // threads concurrently and the env var is restored before
        // exit. Other concurrent tests under nextest run in
        // separate processes.
        unsafe { std::env::set_var("PATH", &new_path) };
        let found = search_path_for("dummy_non_exec");
        unsafe { std::env::set_var("PATH", &original_path) };

        assert!(
            found.is_none(),
            "non-executable file must NOT match PATH lookup, got: {found:?}",
        );
    }

    /// Read every entry from a gzip-compressed tar blob. Used by the
    /// archive-shape tests below — keeps the read path consolidated
    /// so a future tar-or-gzip change forces only one update.
    fn read_archive_entries(blob: &[u8]) -> Vec<(String, u32, Vec<u8>)> {
        use flate2::read::GzDecoder;
        use std::io::Read as _;
        let gz = GzDecoder::new(blob);
        let mut archive = tar::Archive::new(gz);
        let mut out = Vec::new();
        for entry in archive.entries().expect("read tar entries") {
            let mut e = entry.expect("entry");
            let name = e.path().expect("entry path").to_string_lossy().into_owned();
            let mode = e.header().mode().expect("entry mode");
            let mut data = Vec::new();
            e.read_to_end(&mut data).expect("read entry body");
            out.push((name, mode, data));
        }
        out
    }

    /// `build_archive` with no scheduler and no includes packs ONLY
    /// the ktstr binary under the name `ktstr`. Pins the
    /// scheduler-less code path: the EEVDF/binary-payload export
    /// must NOT silently embed extra entries when the test has no
    /// scheduler binary to ship.
    #[test]
    fn build_archive_no_scheduler_packs_only_ktstr() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let ktstr_path = tmp.path().join("fake-ktstr");
        std::fs::write(&ktstr_path, b"#!/bin/sh\necho ktstr-stub\n").expect("write fake ktstr");

        let blob = build_archive(&ktstr_path, None, &[]).expect("build_archive");
        let entries = read_archive_entries(&blob);

        assert_eq!(entries.len(), 1, "expected 1 entry, got: {entries:?}");
        let (name, mode, data) = &entries[0];
        assert_eq!(name, "ktstr", "entry must be named 'ktstr'");
        assert_eq!(*mode, 0o755, "entry must be mode 0o755 (executable)");
        assert_eq!(
            data.as_slice(),
            b"#!/bin/sh\necho ktstr-stub\n",
            "entry payload must roundtrip the input file bytes",
        );
    }

    /// `build_archive` with a scheduler and N include files emits
    /// the canonical layout: `ktstr`, `scheduler`, then
    /// `include/<basename>` per include. Mode 0o755 on every entry
    /// so the .run extractor preserves executable bits. Pins the
    /// archive-layout contract the preamble's
    /// `sed | base64 -d | tar xz` extraction depends on.
    #[test]
    fn build_archive_packs_ktstr_scheduler_and_includes() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let ktstr_path = tmp.path().join("fake-ktstr");
        let sched_path = tmp.path().join("fake-sched");
        let inc_a = tmp.path().join("inc_a.txt");
        let inc_b = tmp.path().join("inc_b.txt");
        std::fs::write(&ktstr_path, b"K").expect("write ktstr");
        std::fs::write(&sched_path, b"S").expect("write scheduler");
        std::fs::write(&inc_a, b"A").expect("write inc_a");
        std::fs::write(&inc_b, b"B").expect("write inc_b");

        let includes = vec![inc_a.clone(), inc_b.clone()];
        let blob = build_archive(&ktstr_path, Some(&sched_path), &includes).expect("build_archive");
        let entries = read_archive_entries(&blob);

        let names: Vec<&str> = entries.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "ktstr",
                "scheduler",
                "include/inc_a.txt",
                "include/inc_b.txt"
            ],
            "entry names and order must match the documented layout",
        );
        for (name, mode, _) in &entries {
            assert_eq!(*mode, 0o755, "entry {name} must be mode 0o755");
        }
    }

    /// `build_archive` rejects two include specs whose basenames
    /// collide. The error message must name the colliding basename
    /// so the operator can find and rename one. The flat
    /// `include/<basename>` layout is documented as the contract;
    /// silently dropping a collision would corrupt the archive.
    #[test]
    fn build_archive_rejects_basename_collision() {
        let tmp_a = tempfile::TempDir::new().expect("temp dir a");
        let tmp_b = tempfile::TempDir::new().expect("temp dir b");
        let inc_1 = tmp_a.path().join("dup.txt");
        let inc_2 = tmp_b.path().join("dup.txt");
        std::fs::write(&inc_1, b"first").expect("write inc_1");
        std::fs::write(&inc_2, b"second").expect("write inc_2");

        let ktstr_path = tmp_a.path().join("fake-ktstr");
        std::fs::write(&ktstr_path, b"K").expect("write ktstr");

        let err = build_archive(&ktstr_path, None, &[inc_1.clone(), inc_2.clone()])
            .expect_err("colliding basenames must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("dup.txt"),
            "error must name the colliding basename: '{msg}'",
        );
        assert!(
            msg.contains("collision") || msg.contains("collide"),
            "error must describe the failure as a collision: '{msg}'",
        );
    }

    /// `generate_preamble` emits a syntactically valid bash script
    /// that `bash -n` parses without error. This is the load-bearing
    /// invariant of the export pipeline — a malformed preamble means
    /// every `.run` file the operator generates is a dud, regardless
    /// of whether the embedded archive is correct.
    ///
    /// Skipped (not failed) when `bash` is unavailable on the host
    /// running the unit tests; bash is universal on linux dev hosts
    /// but a CI image without it shouldn't make the rest of the
    /// suite red.
    #[test]
    fn generate_preamble_parses_under_bash_n() {
        if which_bash().is_none() {
            crate::report::test_skip("no bash on PATH");
            return;
        }

        let entry = KtstrTestEntry {
            name: "test_preamble_smoke",
            extra_sched_args: &["--foo", "bar baz"],
            ..KtstrTestEntry::DEFAULT
        };

        for has_scheduler in [true, false] {
            let preamble = generate_preamble(&entry, has_scheduler);
            assert_bash_n_accepts(&preamble, has_scheduler);
        }
    }

    /// `generate_preamble` interpolates the test name, scheduler
    /// name, topology, duration, and watchdog into the script in
    /// shape that an operator can grep. Pins the variable names
    /// (KTSTR_TEST_NAME, KTSTR_SCHED_NAME, NEED_LLCS,
    /// TEST_DURATION_SECS, TEST_WATCHDOG_SECS) and their values
    /// against the entry — a future format change must update this
    /// test in lockstep with the preamble.
    #[test]
    fn generate_preamble_interpolates_entry_fields() {
        let entry = KtstrTestEntry {
            name: "interp_smoke",
            duration: std::time::Duration::from_secs(7),
            watchdog_timeout: std::time::Duration::from_secs(13),
            topology: crate::vmm::topology::Topology {
                llcs: 3,
                cores_per_llc: 5,
                threads_per_core: 2,
                numa_nodes: 4,
                nodes: None,
                distances: None,
            },
            ..KtstrTestEntry::DEFAULT
        };
        let preamble = generate_preamble(&entry, true);

        assert!(
            preamble.contains("KTSTR_TEST_NAME=interp_smoke"),
            "preamble must set KTSTR_TEST_NAME from entry.name",
        );
        assert!(
            preamble.contains("NEED_LLCS=3"),
            "preamble must set NEED_LLCS from entry.topology.llcs",
        );
        assert!(
            preamble.contains("NEED_CORES_PER_LLC=5"),
            "preamble must set NEED_CORES_PER_LLC",
        );
        assert!(
            preamble.contains("NEED_THREADS_PER_CORE=2"),
            "preamble must set NEED_THREADS_PER_CORE",
        );
        assert!(
            preamble.contains("NEED_NUMA_NODES=4"),
            "preamble must set NEED_NUMA_NODES",
        );
        assert!(
            preamble.contains("TEST_DURATION_SECS=7"),
            "preamble must set TEST_DURATION_SECS from entry.duration",
        );
        assert!(
            preamble.contains("TEST_WATCHDOG_SECS=13"),
            "preamble must set TEST_WATCHDOG_SECS from entry.watchdog_timeout",
        );
    }

    /// `write_runfile` produces the exact on-disk shape the
    /// preamble's extraction step depends on:
    /// 1. The preamble bytes verbatim
    /// 2. A literal `__ARCHIVE__\n` marker line
    /// 3. Base64-encoded archive bytes split into lines of <= 76
    ///    characters each
    ///
    /// The runfile must also be mode 0o755 so the operator can run
    /// `./repro.run` directly. Decoding the base64 chunk back must
    /// reproduce the original archive bytes exactly.
    #[test]
    fn runfile_layout_and_archive_roundtrip() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let out = tmp.path().join("smoke.run");
        let preamble = "#!/bin/bash\necho hello\n";
        // Use bytes that look nothing like ASCII so we know we're
        // verifying the binary roundtrip and not picking up a
        // text-mode coincidence.
        let archive: Vec<u8> = (0u8..=255).chain(0u8..=128).collect();

        write_runfile(&out, preamble, &archive).expect("write_runfile");

        // Mode 0o755 on the file itself.
        let mode = std::fs::metadata(&out).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o755, "runfile mode must be 0o755");

        let raw = std::fs::read_to_string(&out).expect("read runfile");
        let marker = "\n__ARCHIVE__\n";
        let split_at = raw
            .find(marker)
            .expect("runfile must contain a __ARCHIVE__ marker line");
        // The preamble portion is everything up to and including the
        // newline that immediately precedes __ARCHIVE__. The preamble
        // string already terminates with `\n`, so the first split
        // point includes that trailing newline.
        assert_eq!(
            &raw[..split_at + 1],
            preamble,
            "preamble must be written verbatim before the marker",
        );

        let after = &raw[split_at + marker.len()..];
        for line in after.lines() {
            assert!(
                line.len() <= 76,
                "base64 line must be <= 76 cols (POSIX MIME width), got {}: {line:?}",
                line.len(),
            );
        }
        let joined: String = after.lines().collect();
        let decoded = BASE64
            .decode(joined.as_bytes())
            .expect("base64 decode of runfile tail must succeed");
        assert_eq!(
            decoded, archive,
            "base64 roundtrip must reproduce the input archive bytes",
        );
    }

    /// End-to-end smoke: build_archive + generate_preamble +
    /// write_runfile compose into a runfile whose extracted archive
    /// contains the expected entries AND whose preamble parses
    /// under `bash -n`. Validates that the three pipeline stages
    /// glue together without truncation, escaping bugs, or shape
    /// drift between layers.
    #[test]
    fn export_pipeline_round_trip_for_eevdf_entry() {
        if which_bash().is_none() {
            crate::report::test_skip("no bash on PATH");
            return;
        }
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let ktstr_path = tmp.path().join("fake-ktstr");
        std::fs::write(&ktstr_path, b"FAKE_KTSTR").expect("write ktstr stub");
        let inc = tmp.path().join("topology.yaml");
        std::fs::write(&inc, b"some: yaml").expect("write include");
        let out = tmp.path().join("e2e.run");

        let entry = KtstrTestEntry {
            name: "export_smoke",
            ..KtstrTestEntry::DEFAULT
        };
        let archive =
            build_archive(&ktstr_path, None, std::slice::from_ref(&inc)).expect("build archive");
        let preamble = generate_preamble(&entry, false);
        write_runfile(&out, &preamble, &archive).expect("write runfile");

        // Preamble half: split at marker, run bash -n on the
        // preamble portion.
        let raw = std::fs::read_to_string(&out).expect("read runfile");
        let split_at = raw.find("\n__ARCHIVE__\n").expect("marker present");
        assert_bash_n_accepts(&raw[..split_at + 1], false);

        // Archive half: decode + verify both expected entries.
        let entries = read_archive_entries(&archive);
        let names: Vec<&str> = entries.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["ktstr", "include/topology.yaml"],
            "archive must contain ktstr and the single include entry",
        );

        // Preamble half (positive content checks): operator-visible
        // identifiers must reflect the entry name and the test's
        // duration default (DEFAULT.duration is 2s).
        assert!(
            raw[..split_at].contains("KTSTR_TEST_NAME=export_smoke"),
            "preamble must name the entry",
        );
        assert!(
            raw[..split_at].contains("TEST_DURATION_SECS=2"),
            "preamble must reflect the entry's duration",
        );
    }

    /// Pipe `script` through `bash -n` and assert exit status 0.
    /// `has_scheduler` is included only to make failure messages
    /// distinguish the two preamble shapes.
    fn assert_bash_n_accepts(script: &str, has_scheduler: bool) {
        use std::io::Write as _;
        use std::process::{Command, Stdio};
        let bash = which_bash().expect("bash should have been checked by caller");
        let mut child = Command::new(&bash)
            .arg("-n")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn bash -n");
        child
            .stdin
            .as_mut()
            .expect("bash stdin")
            .write_all(script.as_bytes())
            .expect("pipe script to bash");
        let output = child.wait_with_output().expect("bash -n wait");
        assert!(
            output.status.success(),
            "bash -n rejected the preamble (has_scheduler={has_scheduler}); \
             stderr:\n{}\nscript:\n{script}",
            String::from_utf8_lossy(&output.stderr),
        );
    }

    /// Locate `bash` on the host running the unit tests. Returns
    /// `None` rather than panicking when bash is absent; callers
    /// use the `None` path to skip the bash-dependent checks.
    fn which_bash() -> Option<PathBuf> {
        search_path_for("bash")
    }
}
