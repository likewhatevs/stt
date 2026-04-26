# Runs

Each `cargo nextest run` of ktstr tests writes per-test result
sidecars into a *run directory* under
`{CARGO_TARGET_DIR or "target"}/ktstr/`. The directory is the
record -- there is no separate "baselines" cache.

## Layout

```
target/
└── ktstr/
    ├── 6.14-20260424T014200Z/   # one run: kernel 6.14, invoked 2026-04-24 01:42 UTC
    │   ├── test_a.ktstr.json
    │   └── test_b.ktstr.json
    └── 7.0-20260424T015830Z/    # another run: different kernel, different timestamp
        ├── test_a.ktstr.json
        └── test_b.ktstr.json
```

Each subdirectory is keyed `{kernel}-{timestamp}`, where `{kernel}`
is the kernel version resolved from the directory `KTSTR_KERNEL`
points at — first the `version` field in its `metadata.json`, else
the content of `include/config/kernel.release`, else `unknown` (when
`KTSTR_KERNEL` is unset or neither file yields a version) — and
`{timestamp}` is a compact `YYYYMMDDTHHMMSSZ` UTC stamp captured
once per `cargo ktstr test` invocation. Successive runs always get
distinct directories, so no run ever overwrites another.

`KTSTR_SIDECAR_DIR` overrides the *sidecar* directory itself
(used as-is, no key suffix), not the parent. The override only
affects where new sidecars are written and what bare
`cargo ktstr stats` reads. `cargo ktstr stats list`,
`cargo ktstr stats compare`, `cargo ktstr stats list-values`,
and `cargo ktstr stats show-host` all walk
`{CARGO_TARGET_DIR or "target"}/ktstr/` by default — pass
`--dir DIR` on `compare` / `list-values` / `show-host` to point
them at an alternate run root (e.g. an archived sidecar tree
copied off a CI host). They do NOT consult `KTSTR_SIDECAR_DIR`.

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

4. **Compare** across dimensions:

   ```sh
   cargo ktstr stats compare --a-kernel 6.14 --b-kernel 7.0
   cargo ktstr stats compare --a-kernel 6.14 --b-kernel 7.0 -E cgroup_steady
   cargo ktstr stats compare --a-scheduler scx_rusty --b-scheduler scx_lavd --kernel 6.14
   cargo ktstr stats compare --a-project-commit abcdef1 --b-project-commit fedcba2
   cargo ktstr stats compare --a-kernel-commit abcdef1 --b-kernel-commit fedcba2
   cargo ktstr stats compare --a-run-source ci --b-run-source local
   ```

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
   cargo ktstr stats show-host --run 6.14-20260424T014200Z
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
