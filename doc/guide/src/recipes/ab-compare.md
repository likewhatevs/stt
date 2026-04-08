# A/B Compare Branches

Compare scheduler behavior between two branches by running the
same `#[stt_test]` suite against each.

## Setup worktrees

```sh
cd ~/opensource/scx

# Create worktree for the baseline branch
git worktree add ~/opensource/scx-main upstream/main
```

## Run baseline

```sh
cd ~/opensource/scx-main
cargo nextest run --workspace
```

## Run experimental and compare

```sh
cd ~/opensource/scx
cargo nextest run --workspace
```

Compare test output between the two runs.

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
```
