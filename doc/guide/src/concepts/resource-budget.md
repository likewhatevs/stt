# Resource Budget

`--cpu-cap N` adds a third tier between full performance-mode
isolation and unreserved no-perf-mode execution. Instead of
"lock each reserved LLC exclusively" (perf-mode), it reserves a
NUMA-aware, consolidation-aware set of host CPUs under `LOCK_SH`,
enforces the reservation via a cgroup v2 cpuset sandbox, and scales
`make -jN` fan-out to the reserved capacity. The flock granularity
stays per-LLC: every selected LLC is flocked whole, but `plan.cpus`
holds EXACTLY `N` CPUs (the last LLC is partial-taken when the
budget falls mid-LLC). See
[Performance Mode](performance-mode.md#three-way-mode-tier) for
the comparison against the other two tiers.

Every no-perf-mode VM and kernel build runs through this pipeline
— there is no "no cap" path. When `--cpu-cap` is absent, the
planner applies a **30% default** of the calling process's
sched_getaffinity cpuset (minimum 1 CPU). This keeps
`sched_setaffinity` safe under cgroup-restricted CI runners (CI
hosts, systemd slices, sudo-under-a-limited-cpuset) where the
process cannot run on every online CPU even if sysfs lists them.

## When to use it

- **Multi-tenant CI hosts** where unbounded parallelism starves
  concurrent builds but the full performance-mode contract
  (`SCHED_FIFO`, hugepages, NUMA mbind, KVM exit suppression) is
  too heavy.
- **Kernel builds run alongside perf-mode VM tests** — the
  shared `LOCK_SH` coordinates with the perf-mode `LOCK_EX` so
  `make` never stomps a measurement in progress.
- **Concurrent no-perf-mode VMs on a shared host** — a cap of `N`
  CPUs bounds how much capacity each run reserves; peers that
  would exceed the host's flock availability wait rather than
  racing for CPU.

## `CpuCap` — parsed and resolved

`CpuCap::new(N: usize) -> Result<CpuCap>` constructs a cap from a
CLI integer. `N` is a CPU count. `N == 0` is rejected with
`--cpu-cap must be ≥ 1 CPU (got 0)` — zero is a scripting sentinel,
not a silent "no cap" fallback.

`CpuCap::resolve(cli_flag: Option<usize>) -> Result<Option<CpuCap>>`
is the three-tier precedence:

1. CLI flag (`--cpu-cap N`) wins over env var.
2. `KTSTR_CPU_CAP=N` env var applies when the CLI flag is absent.
   Empty string is treated as unset; `0` or non-numeric values
   produce the same rejection as the CLI path.
3. Neither set → `Ok(None)`. The planner expands this into the
   30%-of-allowed default at acquire time.

`CpuCap::effective_count(allowed_cpus: usize) -> Result<usize>`
clamps at acquire time, not construction time.
`N > allowed_cpus` returns a `ResourceContention` error naming
both numbers — operators reading the error see immediately that
the cap exceeds the process's `sched_getaffinity` cpuset, not the
host's total online CPU count. Fixing the cap requires either
lowering `N` or releasing the cgroup restriction on the calling
process.

## `host_allowed_cpus` — the reference set

`host_allowed_cpus()` reads the calling process's allowed CPUs
via `sched_getaffinity(0)` with a `/proc/self/status`
`Cpus_allowed_list:` fallback. Every consumer of the `--cpu-cap`
pipeline plans against this set instead of
`HostTopology::online_cpus`, so `sched_setaffinity` on the plan's
CPU list never produces an empty effective mask under a
cgroup-restricted runner.

An empty allowed set is a bail condition, not a fallback to
"every CPU" — guessing on a misconfigured host is worse than
failing visibly. A host topology that has no LLC overlapping the
allowed set (sysfs and `sched_getaffinity` disagree — e.g. stale
sysfs after hot-plug, cgroup cpuset pinned to CPUs the kernel no
longer reports in LLC groups) also bails with an actionable
diagnostic.

## `LlcPlan` — the ACQUIRE result

`acquire_llc_plan(topo, test_topo, cpu_cap)` runs three phases:

1. **DISCOVER** — for every LLC, stat the canonical
   `/tmp/ktstr-llc-{N}.lock`, read `/proc/locks` once, and build a
   snapshot of holders per LLC. No flocks are taken.
2. **PLAN** — rank LLCs (eligible = at least one allowed CPU):
   consolidation (prefer LLCs with existing holders) first, then
   fresh LLCs, all tiebroken by ascending index. Seed on the
   highest-scored LLC's NUMA node; greedily fill that node before
   spilling to nearest-by-distance nodes via
   `TestTopology::numa_distance`. Accumulate allowed-CPU
   contribution per LLC until the accumulated count meets
   `target_cpus`. Final acquire order is ascending LLC index for
   livelock safety.
3. **ACQUIRE** — non-blocking `LOCK_SH` on every selected LLC. A
   single `EWOULDBLOCK` drops every held fd and retries once
   (one TOCTOU retry — the second DISCOVER's `/proc/locks` read
   IS the backoff; more retries would amplify livelock risk
   without adding coordination signal).

### Partial-take on the last LLC

Post-ACQUIRE, the materialization layer walks each selected LLC's
CPUs in ascending order, intersects with the allowed set, and
STOPS at exactly `target_cpus` total. The last selected LLC
typically contributes only a prefix of its allowed CPUs — the
flock is still held at LLC granularity (coordination with
concurrent ktstr peers is always per-LLC), but `plan.cpus`
reflects the exact CPU budget. `sched_setaffinity` masks and
cgroup `cpuset.cpus` writes narrow to that exact set.

The returned `LlcPlan` carries:

- `locked_llcs: Vec<usize>` — selected host LLC indices, ASC.
- `cpus: Vec<usize>` — flat list of reserved CPUs, sized exactly
  `target_cpus` (a subset of every selected LLC's allowed CPUs,
  with the last LLC possibly contributing only a prefix).
- `mems: BTreeSet<usize>` — NUMA nodes actually hosting
  `plan.cpus` (an LLC that contributes a partial slice only
  registers the nodes of its used CPUs).
- `snapshot: Vec<LlcSnapshot>` — per-LLC discovery trail.
- `locks: Vec<OwnedFd>` — RAII flock handles; Drop releases.

When `mems` spans more than one node
(`warn_if_cross_node_spill` fires), stderr gets a `ktstr:
reserving LLCs […] across N NUMA nodes` warning so the operator
knows to expect cross-node memory latency. Single-node plans are
silent.

## Cgroup v2 cpuset sandbox

`BuildSandbox::try_create(plan_cpus, plan_mems, hard_error_on_degrade)`
writes the plan into a child cgroup under the caller's own cgroup,
in the kernel-required order: `cpuset.cpus` → `cpuset.mems` →
`cgroup.procs`. A task in a cgroup with empty `cpuset.mems` may
be killed by the cpuset allocator, so migration into
`cgroup.procs` MUST happen after both cpuset fields are populated.

After each cpuset write, `.effective` is read back. Narrowing by
a parent cgroup (e.g. systemd slice restriction) is a fatal error
under `--cpu-cap` (`hard_error_on_degrade = true`) and a warn-
only degrade without the flag.

Drop migrates the build pid back to root, tolerates transient
EBUSY on `cgroup.rmdir` (5 × 10 ms retries), and orphans the
directory with a `tag=resource_budget.cgroup_orphan_left` warn-
log if the rmdir still refuses. Orphans older than 24 h are
swept on the next sandbox creation.

## `make -jN` hint

`make_jobs_for_plan(plan)` returns `plan.cpus.len().max(1)`. The
kernel-build pipeline threads this as `make -jN`. Without the
hint, `make -j$(nproc)` fans gcc children across every online
CPU, defeating the cpuset reservation in scheduling terms — the
kernel still enforces cpuset membership at the fs layer, but
gcc's parallel width silently violates the budget. The `.max(1)`
floor guards against `make -j0` (unbounded on GNU make).

## `ktstr locks` — observational surface

`ktstr locks` (or `cargo ktstr locks`) prints every ktstr flock
currently held on the host, cross-referenced against
`/proc/locks` to name each holder by PID + truncated cmdline.
Read-only — takes no flocks. Three categories:

1. **LLC locks** under `/tmp/ktstr-llc-*.lock`
2. **Per-CPU locks** under `/tmp/ktstr-cpu-*.lock`
3. **Cache-entry locks** under `{cache_root}/.locks/*.lock`

Flags:

- `--json` — emit a structured snapshot. One-shot uses
  `to_string_pretty` for readability; under `--watch` each frame
  is compact on its own line (ndjson-style) for streaming
  consumers. Top-level keys: `llcs`, `cpus`, `cache`. Each row
  names its `lockfile` path and a `holders` array; every holder
  has `pid` + `cmdline`.
- `--watch <interval>` — redraw on the given interval until
  SIGINT. Interval uses `humantime` syntax (`100ms`, `1s`,
  `5m`, `1h`).

Use `ktstr locks` when `--cpu-cap` acquires fail with
`ResourceContention`: the error already names busy LLCs, but the
live snapshot shows every contending peer at once.

## `KTSTR_BYPASS_LLC_LOCKS` — escape hatch

Setting `KTSTR_BYPASS_LLC_LOCKS=1` (any non-empty value) skips
`acquire_llc_plan` entirely. The VM boots or the kernel
builds immediately without coordinating against any concurrent
perf-mode run. Use only when the operator explicitly accepts
measurement noise:

- A shell session doing unrelated work alongside tests.
- An isolated developer workstation.
- A CI queue that already serializes jobs at a higher layer.

Mutually exclusive with `--cpu-cap` / `KTSTR_CPU_CAP` at every
entry point (CLI parse for `shell` + `kernel build` on both
`ktstr` and `cargo ktstr`, the `kernel_build_pipeline` reservation
phase, and the library-layer `KtstrVmBuilder::build` no-perf-mode
branch). The error wording always contains `"resource contract"`
so operators can grep for it; the contract and the bypass cannot
coexist at any of those six sites.

Note: the `performance_mode=true` vs `--cpu-cap` exclusion is
weaker. It is enforced at CLI parse (`shell --cpu-cap` requires
`--no-perf-mode` via clap `requires`), but library consumers that
set `performance_mode=true` on `KtstrVmBuilder` directly see
`KTSTR_CPU_CAP` silently ignored — the builder's perf-mode branch
never calls `CpuCap::resolve`, it goes through
`validate_performance_mode` + `acquire_resource_locks`
(LOCK_EX) instead.

## Filesystem requirement

Every ktstr lockfile (`/tmp/ktstr-llc-*.lock`,
`/tmp/ktstr-cpu-*.lock`, `{cache_root}/.locks/*.lock`,
`{runs_root}/.locks/*.lock`) must live on a local filesystem —
tmpfs, ext4, xfs, btrfs, f2fs, or bcachefs are the
explicitly-accepted set. `flock(2)` behavior
on NFS, CIFS, SMB2, CephFS, AFS, and FUSE is unreliable: NFSv3
is advisory-only without an NLM peer and NFSv4 byte-range
locking does not cover `flock(2)`; SMB does not emit
`/proc/locks` entries so ktstr cannot enumerate peer holders;
Ceph MDS does not participate in `flock` serialization across
nodes; AFS does not support `flock(2)` at all; FUSE flock
semantics depend on whether the userspace server implements the
op. `try_flock` statfs-checks every lockfile path at open time
via `reject_remote_fs` in `src/flock.rs` — hitting any
deny-listed filesystem produces an actionable runtime error
naming the filesystem plus the remediation "Move the lockfile
path to a local filesystem (tmpfs, ext4, xfs, btrfs, f2fs,
bcachefs)." Unknown local filesystems (zfs, erofs, etc.) are
not on the deny-list and pass through, on the basis that
rejecting unknown-but-local is more disruptive than accepting a
potentially-unreliable `flock`.

## Related

- [Performance Mode](performance-mode.md) — the full-isolation
  tier; the tier comparison lives there.
- [Environment Variables](../reference/environment-variables.md)
  — `KTSTR_CPU_CAP`, `KTSTR_BYPASS_LLC_LOCKS`, and every other
  ktstr-controlled env var.
