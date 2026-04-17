# Baselines

ktstr can save test results as baselines and compare subsequent runs
against them.

## Workflow

1. **Save baseline**: set `KTSTR_SIDECAR_DIR` to a directory. Each test
   writes a `SidecarResult` JSON file there.

   ```sh
   KTSTR_SIDECAR_DIR=./baseline cargo nextest run --workspace
   ```

2. **Run current**: run the same tests with a different sidecar dir.

   ```sh
   KTSTR_SIDECAR_DIR=./current cargo nextest run --workspace
   ```

3. **Compare**: diff the sidecar JSON files between directories.
   Automated comparison tooling is planned but not yet implemented.
   For now, compare pass/fail counts and per-test metrics manually
   or with standard JSON diffing tools.

## Sidecar format

Each test writes a `SidecarResult` JSON file containing the test name,
topology, scheduler, work type, pass/fail, per-cgroup stats, monitor
summary, stimulus events, verifier stats, KVM stats, effective sysctls,
and kernel command-line args. Files are named with a `.ktstr.` infix for
discovery. `cargo ktstr test-stats` reads all sidecar files from a
directory (recursing one level for gauntlet per-job subdirectories).

See also: [`KTSTR_SIDECAR_DIR`](../reference/environment-variables.md).
