# Runs

Each `cargo nextest run` of ktstr tests writes per-test result
sidecars into a *run directory* under
`{CARGO_TARGET_DIR or "target"}/ktstr/`. The directory is the
record -- there is no separate "baselines" cache.

## Layout

```
target/
└── ktstr/
    ├── 6.14-abc1234/        # one run: kernel 6.14, repo commit abc1234
    │   ├── test_a.ktstr.json
    │   └── test_b.ktstr.json
    └── 7.0-def5678/         # another run: different kernel, different commit
        ├── test_a.ktstr.json
        └── test_b.ktstr.json
```

Each subdirectory is keyed `{kernel}-{git_short}`, where `{kernel}`
is the kernel version resolved from the directory `KTSTR_KERNEL`
points at — first the `version` field in its `metadata.json`, else
the content of `include/config/kernel.release`, else `unknown` (when
`KTSTR_KERNEL` is unset or neither file yields a version) — and
`{git_short}` is the short commit hash baked in by `build.rs`.

`KTSTR_SIDECAR_DIR` overrides the *sidecar* directory itself
(used as-is, no key suffix), not the parent. The override only
affects where new sidecars are written and what bare
`cargo ktstr stats` reads. `cargo ktstr stats list` and
`cargo ktstr stats compare` always enumerate
`{CARGO_TARGET_DIR or "target"}/ktstr/` regardless of
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

4. **Compare** two runs:

   ```sh
   cargo ktstr stats compare 6.14-abc1234 7.0-def5678
   cargo ktstr stats compare 6.14-abc1234 7.0-def5678 -E cgroup_steady
   ```

   Per-metric deltas are computed using the unified `MetricDef`
   registry (polarity, absolute and relative thresholds). Output is
   colored: red for regressions, green for improvements. The command
   exits non-zero when regressions are detected.

5. **Print analysis** for the most recent run (no subcommand):

   ```sh
   cargo ktstr stats
   ```

   Picks the newest subdirectory under `target/ktstr/` by mtime and
   prints gauntlet analysis, BPF verifier stats, callback profile,
   and KVM stats.

## Sidecar format

Each test writes a `SidecarResult` JSON file containing the test name,
topology, scheduler, work type, pass/fail, per-cgroup stats, monitor
summary, stimulus events, verifier stats, KVM stats, effective sysctls,
kernel command-line args, kernel version, timestamp, and run ID. Files
are named with a `.ktstr.` infix for discovery. `cargo ktstr stats`
reads all sidecar files from a run directory (recursing one level for
gauntlet per-job subdirectories).

See also: [`KTSTR_SIDECAR_DIR`](../reference/environment-variables.md).
