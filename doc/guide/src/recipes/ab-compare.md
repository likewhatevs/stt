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
`target/ktstr/{kernel}-{project_commit}/`. The `{project_commit}`
half is the project tree's HEAD short hex captured at first
sidecar write (suffixed `-dirty` when the worktree differs from
HEAD), so two branches with distinct HEADs land in distinct
directories.
Two back-to-back runs of the SAME kernel at the SAME commit
reuse the same directory — the second run pre-clears any prior
sidecars at first write, so each directory is a last-writer-wins
snapshot of (kernel, project commit).

> **Warning:** The two worktrees MUST be at distinct commits for
> A/B comparison to work. If both checkouts share the same HEAD
> (e.g. baseline branch and feature branch happen to be even),
> the second run **overwrites** the first via the last-writer-wins
> pre-clear and the comparison degenerates to "identical pool of
> sidecars." Confirm distinct commits with `git -C ~/opensource/scx
> rev-parse HEAD` and `git -C ~/opensource/scx-main rev-parse
> HEAD` before invoking the second `cargo nextest run`.

Every sidecar also carries its own `project_commit` field (read
from the project tree's git HEAD at sidecar-write time), so
the runs from two branches land disjoint values on the `commit`
dimension regardless of how the directories are named. The
project commit is discovered by walking up from the test
process's **current working directory** to find a `.git` marker
— so the `cd ~/opensource/scx-main` / `cd ~/opensource/scx`
steps below are load-bearing, not stylistic. Without them the
probe would walk up from wherever you happened to invoke
`cargo`, potentially ending at an entirely different repo and
recording the wrong commit on every sidecar. The simplest
collection workflow is to merge both branches' run
subdirectories under one root and rely on
`--a-project-commit` / `--b-project-commit` to partition them:

```sh
mkdir -p ~/opensource/scx-runs/ktstr

# Baseline.
cd ~/opensource/scx-main
cargo nextest run --workspace
mv target/ktstr/* ~/opensource/scx-runs/ktstr/

# Experimental.
cd ~/opensource/scx
cargo nextest run --workspace
mv target/ktstr/* ~/opensource/scx-runs/ktstr/
```

The `{kernel}-{project_commit}` subdirectory names are unique per
(kernel, project commit) pair, so two branches with distinct
HEADs coexist under one root without collision. Within a single
branch, two clean back-to-back runs at the same commit reuse
one directory (last-writer-wins via per-process pre-clear);
mark one of them as `-dirty` (uncommitted change) or commit /
amend between runs to land separate directories.

Do **not** set `KTSTR_SIDECAR_DIR`: `cargo ktstr stats list`
and `cargo ktstr stats compare` walk
`{CARGO_TARGET_DIR or "target"}/ktstr/` by default and would
not see runs written to a custom flat directory unless
`--dir DIR` is passed.

## Discover available dimension values

The framework records the project tree's git commit (discovered
by walking parents of the test process's cwd to find the
enclosing `.git`) on every sidecar via
`SidecarResult::project_commit`, so two runs from different
commits land disjoint values on the `commit` dimension and
`--a-project-commit` / `--b-project-commit` slice between them
without any per-run directory bookkeeping.
Use
`cargo ktstr stats list-values --dir DIR` to enumerate the
distinct values of every filterable dimension (`kernel`,
`commit`, `kernel_commit`, `source`, `scheduler`, `topology`,
`work_type`, `flags`) present in the pool, so per-side filters
target real values. The `commit` and `source` keys map to the
internal `SidecarResult::project_commit` / `run_source` fields;
the per-side filter flags spell as `--a-project-commit` /
`--b-project-commit` and `--a-run-source` / `--b-run-source`
on the [`compare`](../running-tests/cargo-ktstr.md#compare)
subcommand.

```sh
cd ~/opensource/scx
CARGO_TARGET_DIR=~/opensource/scx-runs cargo ktstr stats list
CARGO_TARGET_DIR=~/opensource/scx-runs cargo ktstr stats list-values
```

## Compare per-side filter groups

```sh
cd ~/opensource/scx
CARGO_TARGET_DIR=~/opensource/scx-runs cargo ktstr stats compare \
    --a-project-commit <baseline-short-hex> \
    --b-project-commit <current-short-hex>
```

`stats compare` is pool-driven: every sidecar under the runs
root is loaded into a single pool, and per-side filter flags
(`--a-X` / `--b-X`) partition the pool into the A and B
contrasts. The dimensions on which the A and B filters DIFFER
are the *slicing* dimensions of the contrast; every other
dimension is part of the dynamic *pairing key* the comparison
joins on. Slicing on `project-commit` alone joins each
baseline scenario with its matching experimental counterpart
on every other dimension (kernel, kernel-commit, run-source,
scheduler, topology, work_type, flags).

Other slicing axes work the same way:

```sh
# Slice on kernel.
cargo ktstr stats compare --a-kernel 6.14 --b-kernel 7.0

# Slice on scheduler, pin both sides to one kernel.
cargo ktstr stats compare \
    --a-scheduler scx_rusty --b-scheduler scx_lavd \
    --kernel 6.14
```

Shared `--X` flags pin BOTH sides to the same value; per-side
`--a-X` / `--b-X` REPLACE the corresponding shared `--X` for
that side only ("more-specific replaces"). Slicing on more
than one dimension at once prints a stderr warning but is
supported for cohort sweeps.

`compare` applies the dual-gate significance check from the
unified `MetricDef` registry to every metric and prints colored
output (red = regression, green = improvement). Rows where
either side has `passed=false` are dropped from the math and
counted in the summary line; the exit code is non-zero when
any regression is detected, so the command can gate CI
directly. Narrow further with `-E SUBSTRING` (matches the
joined `scenario topology scheduler work_type flags` string),
override the relative gate uniformly with `--threshold PCT`
or per-metric via `--policy FILE`. The absolute gate from each
`MetricDef` is unaffected by `--threshold` — a delta must
clear both gates to count as significant.

See [`stats compare`](../running-tests/cargo-ktstr.md#compare)
for the full per-side flag table and validation rules, and
[`stats list-values`](../running-tests/cargo-ktstr.md#list-values)
for the discovery counterpart.

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
rm -rf ~/opensource/scx-runs
```
