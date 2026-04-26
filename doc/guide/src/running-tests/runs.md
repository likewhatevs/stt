# Runs

Each `cargo nextest run` of ktstr tests writes per-test result
sidecars into a *run directory* under
`{CARGO_TARGET_DIR or "target"}/ktstr/`. The directory is the
record of the latest test run for that (kernel, project commit)
pair -- there is no separate "baselines" cache.

> **Warning:** Re-running the suite at the same kernel and
> project commit reuses the same directory and **deletes prior
> sidecars** at the first sidecar write of the new run. To
> preserve a previous run's outputs, archive the directory
> elsewhere first (e.g. `mv target/ktstr/6.14-abc1234
> target/ktstr/6.14-abc1234.archived-{date}`) or commit your
> changes (or amend to drop a `-dirty` suffix) so the next run
> lands in a separate snapshot directory.

## Layout

```
target/
└── ktstr/
    ├── 6.14-abc1234/        # one run: kernel 6.14, project commit abc1234 (clean)
    │   ├── test_a.ktstr.json
    │   └── test_b.ktstr.json
    └── 7.0-def5678-dirty/   # another run: kernel 7.0, project commit def5678 with uncommitted changes
        ├── test_a.ktstr.json
        └── test_b.ktstr.json
```

Each subdirectory is keyed `{kernel}-{commit}`, where `{kernel}`
is the kernel version resolved from the directory `KTSTR_KERNEL`
points at — first the `version` field in its `metadata.json`, else
the content of `include/config/kernel.release`, else `unknown` (when
`KTSTR_KERNEL` is unset or neither file yields a version) — and
`{commit}` is the project tree's HEAD short hex (7 chars), suffixed
`-dirty` when the worktree differs from HEAD, or the literal
`unknown` when the test process is not running inside a git
repository.

The commit is discovered by walking parents of the test process's
working directory until a `.git` marker is found — for a scheduler
crate using ktstr as a dev-dependency, this is the **scheduler
crate's** commit, not ktstr's. The function
that performs the probe (`detect_project_commit`) is called from
the test process's cwd, so running tests from inside the scheduler
crate's clone yields that crate's HEAD. Run from inside ktstr's
clone if you want to record ktstr's HEAD instead.

Two runs sharing the same kernel and project commit (the typical
"re-run the suite without committing changes" loop) reuse the
same directory: the second run pre-clears any prior
`*.ktstr.json` files in the directory at first sidecar write so
the directory is a last-writer-wins snapshot of (kernel, project
commit), not an append-only archive of every invocation. Re-run
the suite to regenerate the sidecars; commit your changes (or
amend to drop the `-dirty` suffix) to land a separate snapshot
directory.

Pre-clear is **shallow** — only `*.ktstr.json` files in the
immediate run directory are removed. Subdirectories created by
external orchestrators (per-job gauntlet layouts, cluster shards)
are left untouched, but `cargo ktstr stats` walks one level of
subdirectories when collecting sidecars, so stale sidecar files
left in subdirectories from a prior run will still appear in
stats output. Operators driving subdirectory layouts must clean
those subdirectories themselves; pre-clear's contract covers the
top-level only.

### Unknown-commit collisions

When the test process is not inside a git repository (so
`detect_project_commit` returns `None`), the on-disk dirname uses
the literal sentinel `unknown` in the commit slot — every such run
lands in `{kernel}-unknown`. Concurrent or successive non-git runs
collide on this single directory, with the latest run pre-clearing
the previous one's sidecars. To disambiguate non-git runs, set
`KTSTR_SIDECAR_DIR` to a per-run path or place the project tree
under git so each run carries its own commit hash.

The `unknown` sentinel applies to the **dirname only**. The
in-memory `SidecarResult.project_commit` field stays `None`
(serialized as JSON `null`) for these runs — the dirname uses a
filesystem-safe sentinel, while the JSON field preserves the
original probe outcome. As a consequence, `cargo ktstr stats
compare --project-commit unknown` will **not** match a sidecar
whose `project_commit` is `None`; omit the `--project-commit`
filter entirely to include `None`-commit rows in the comparison.

`KTSTR_SIDECAR_DIR` overrides the *sidecar* directory itself
(used as-is, no key suffix), not the parent. The override only
affects where new sidecars are written and what bare
`cargo ktstr stats` reads. When the override is set, **pre-clear
is skipped** — the operator chose that directory and owns its
contents, so any pre-existing sidecars there are preserved.
`cargo ktstr stats list`, `cargo ktstr stats compare`,
`cargo ktstr stats list-values`, and `cargo ktstr stats show-host`
all walk `{CARGO_TARGET_DIR or "target"}/ktstr/` by default —
pass `--dir DIR` on `compare` / `list-values` / `show-host` to
point them at an alternate run root (e.g. an archived sidecar
tree copied off a CI host). They do NOT consult
`KTSTR_SIDECAR_DIR`.

## Workflow

1. **Run tests** for kernel A:

   ```sh
   cargo ktstr test --kernel 6.14
   ```

2. **Run again** for kernel B:

   ```sh
   cargo ktstr test --kernel 7.0
   ```

3. **List** runs:

   ```sh
   cargo ktstr stats list
   ```

   Each row carries `RUN`, `TESTS`, and `DATE` columns. `DATE` is
   the earliest sidecar timestamp present in the directory — under
   the last-writer-wins semantics, this equals the **most recent
   run's first sidecar timestamp** (the prior run's sidecars were
   pre-cleared at the new run's first write, so only the new
   run's timestamps remain).

4. **Compare** across dimensions:

   ```sh
   cargo ktstr stats compare --a-kernel 6.14 --b-kernel 7.0
   cargo ktstr stats compare --a-kernel 6.14 --b-kernel 7.0 -E cgroup_steady
   cargo ktstr stats compare --a-scheduler scx_rusty --b-scheduler scx_lavd --kernel 6.14
   cargo ktstr stats compare --a-project-commit abcdef1 --b-project-commit fedcba2
   cargo ktstr stats compare --a-project-commit abc1234 --b-project-commit abc1234-dirty
   cargo ktstr stats compare --a-kernel-commit abcdef1 --b-kernel-commit fedcba2
   cargo ktstr stats compare --a-run-source ci --b-run-source local
   ```

   The `abc1234` vs `abc1234-dirty` row is the canonical
   WIP-vs-baseline pattern: run the suite once at a clean commit
   to capture the baseline directory `{kernel}-abc1234`, edit the
   tree without committing, run the suite again to capture
   `{kernel}-abc1234-dirty`, then diff the two. Both sidecar pools
   coexist under `target/ktstr/` because the `-dirty` suffix
   makes them distinct directories.

   Per-side filters (`--a-*` / `--b-*`) partition the sidecar pool
   into two sides; shared filters (`--kernel`, `--scheduler`,
   `--project-commit`, `--kernel-commit`, `--run-source`, etc.)
   pin both sides. The eight slicing dimensions are `kernel`,
   `scheduler`, `topology`, `work-type`, `project-commit`,
   `kernel-commit`, `run-source`, and `flags`; differing on any
   subset of them defines the A/B contrast. Per-metric deltas are
   computed using the unified `MetricDef` registry (polarity,
   absolute and relative thresholds). Output is colored: red for
   regressions, green for improvements. The command exits non-zero
   when regressions are detected. Use `cargo ktstr stats
   list-values` to discover available dimension values before
   constructing a comparison.

5. **Print analysis** for the most recent run (no subcommand):

   ```sh
   cargo ktstr stats
   ```

   Picks the newest subdirectory under `target/ktstr/` by mtime and
   prints gauntlet analysis, BPF verifier stats, callback profile,
   and KVM stats.

6. **Inspect the archived host context** for a specific run:

   ```sh
   cargo ktstr stats show-host --run 6.14-abc1234
   cargo ktstr stats show-host --run archive-2024-01-15 --dir /tmp/archived-runs
   ```

   Resolves `--run` against `target/ktstr/` (or `--dir` when set),
   scans the run's sidecars in order, and renders the first populated
   host-context field via `HostContext::format_human`: CPU model,
   memory config, transparent-hugepage policy, NUMA node count, uname
   triple, kernel cmdline, and every `/proc/sys/kernel/sched_*`
   tunable. Same fingerprint `stats compare` uses for its host-delta
   section, but available on a single run. Fails with an actionable
   error when no sidecar carries a host field (pre-enrichment run).

## Metric registry discovery

Before configuring per-metric `ComparisonPolicy` overrides, enumerate
the available metric names:

```sh
cargo ktstr stats list-metrics
cargo ktstr stats list-metrics --json
```

Prints the `ktstr::stats::METRICS` registry: metric name, polarity
(higher / lower better), `default_abs` and `default_rel` gate
thresholds, and display unit. Use the metric names from this list as
keys in `ComparisonPolicy.per_metric_percent`; unknown names are
rejected at `--policy` load time so typos surface loudly. `--json`
emits the same data as a serde array — the row accessor function is
omitted (`#[serde(skip)]`) so the wire surface carries only
wire-stable fields.

## Sidecar format

Each test writes a `SidecarResult` JSON file containing the test name,
topology, scheduler, work type, pass/fail, per-cgroup stats, monitor
summary, stimulus events, verifier stats, KVM stats, effective sysctls,
kernel command-line args, kernel version, timestamp, and run ID. Files
are named with a `.ktstr.` infix for discovery. `cargo ktstr stats`
reads all sidecar files from a run directory (recursing one level for
gauntlet per-job subdirectories).

See also: [`KTSTR_SIDECAR_DIR`](../reference/environment-variables.md).
