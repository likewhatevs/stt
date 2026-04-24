# A/B Compare Branches

Compare scheduler behavior between two branches by running the
same `#[ktstr_test]` suite against each, then using
`cargo ktstr stats compare` to diff per-metric results with
dual-gate (absolute and relative) significance and exit non-zero
on any regression.

## Setup worktrees

The examples below use the `scx` scheduler crate under
`~/opensource/scx`; substitute your own scheduler crate's path and
remote everywhere `scx` appears.

```sh
cd ~/opensource/scx

# Create a worktree for the baseline branch.
git worktree add ~/opensource/scx-main upstream/main
```

## Collect both runs into a shared run root

Each `cargo nextest run --workspace` writes its sidecars into
`target/ktstr/{kernel}-{timestamp}/`. The `{timestamp}` half is
captured once per process at run start (compact
`YYYYMMDDTHHMMSSZ` form), so successive runs always land in
distinct directories — two back-to-back runs of the same kernel
never overwrite each other. Rename each run to a branch-tagged
name so `stats compare` can address them by memorable keys.

Do **not** set `KTSTR_SIDECAR_DIR`: `cargo ktstr stats list` and
`cargo ktstr stats compare` always enumerate
`{CARGO_TARGET_DIR or "target"}/ktstr/` and would not see runs
written to a custom flat directory.

```sh
mkdir -p ~/opensource/scx-runs/ktstr

# Baseline.
cd ~/opensource/scx-main
cargo nextest run --workspace
mv target/ktstr/* ~/opensource/scx-runs/ktstr/baseline

# Experimental.
cd ~/opensource/scx
cargo nextest run --workspace
mv target/ktstr/* ~/opensource/scx-runs/ktstr/current
```

Each `mv` renames the just-produced `{kernel}-{timestamp}/`
subdirectory to a branch-tagged name (`baseline`, `current`) so
both runs coexist under one root.

## List and compare

Point `cargo ktstr stats` at the shared run root via
`CARGO_TARGET_DIR`:

```sh
cd ~/opensource/scx
CARGO_TARGET_DIR=~/opensource/scx-runs cargo ktstr stats list
CARGO_TARGET_DIR=~/opensource/scx-runs cargo ktstr stats compare baseline current
```

`cargo ktstr stats compare` joins on
`(scenario, topology, work_type)`, applies the dual-gate
significance check from the unified `MetricDef` registry to every
metric, and prints colored output (red = regression, green =
improvement). Rows where either side has `passed=false` are
dropped from the math and counted in the summary line. The exit
code is non-zero when any regression is detected, so the command
can gate CI directly.

Narrow the comparison with `-E SUBSTRING` (matches `scenario`,
`topology`, and `work_type`); override the relative gate with
`--threshold PCT`. The absolute gate from each `MetricDef` is
unaffected by `--threshold` -- a delta must clear both gates to
count as significant.

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
rm -rf ~/opensource/scx-runs
```
