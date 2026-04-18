# Baselines

ktstr saves test results as baselines and compares subsequent runs
against them.

## Workflow

1. **Run tests** and collect stats. `cargo ktstr stats` auto-saves
   sidecars to `~/.cache/ktstr/baselines/{kernel}-{commit}/`:

   ```sh
   cargo ktstr test --kernel 6.14
   cargo ktstr stats
   ```

2. **Run again** (different kernel, commit, or scheduler):

   ```sh
   cargo ktstr test --kernel 7.0
   cargo ktstr stats
   ```

3. **List** saved baselines:

   ```sh
   cargo ktstr stats list
   ```

4. **Compare** two baselines:

   ```sh
   cargo ktstr stats compare 6.14-abc1234 7.0-def5678
   cargo ktstr stats compare 6.14-abc1234 7.0-def5678 -E cgroup_steady
   ```

   Per-metric deltas are computed using the unified `MetricDef`
   registry (polarity, absolute and relative thresholds). Output is
   colored: red for regressions, green for improvements. The command
   exits non-zero when regressions are detected.

## Sidecar format

Each test writes a `SidecarResult` JSON file containing the test name,
topology, scheduler, work type, pass/fail, per-cgroup stats, monitor
summary, stimulus events, verifier stats, KVM stats, effective sysctls,
kernel command-line args, kernel version, timestamp, and run ID. Files
are named with a `.ktstr.` infix for discovery. `cargo ktstr stats`
reads all sidecar files from a directory (recursing one level for
gauntlet per-job subdirectories).

See also: [`KTSTR_SIDECAR_DIR`](../reference/environment-variables.md).
