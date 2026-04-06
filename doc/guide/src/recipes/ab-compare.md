# A/B Compare Branches

Compare scheduler behavior between two branches using git worktrees
and baselines.

## Setup worktrees

```sh
cd ~/opensource/scx

# Create worktree for the baseline branch
git worktree add ~/opensource/scx-main upstream/main
```

## Run baseline and save results

`-p` builds the scheduler from source before each run:

```sh
cd ~/opensource/scx-main
cargo stt vm --gauntlet -p scx_mitosis \
  --save-baseline baseline.json
```

## Run experimental and compare

```sh
cd ~/opensource/scx
cargo stt vm --gauntlet -p scx_mitosis \
  --compare ~/opensource/scx-main/baseline.json
```

The comparison report shows regressions and improvements across all
scenarios and topologies. See [Baselines](../running-tests/baselines.md)
for the save/compare format.

## For integration tests

```sh
cd ~/opensource/scx-main
cargo stt test --save-baseline baseline.json

cd ~/opensource/scx
cargo stt test --compare ~/opensource/scx-main/baseline.json
```

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
```
