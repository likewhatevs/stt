# A/B Compare Branches

Compare scheduler behavior between two branches using git worktrees.

## Setup worktrees

```sh
cd ~/opensource/scx

# Create worktree for the baseline branch
git worktree add ~/opensource/scx-main upstream/main
```

## Run baseline

```sh
cd ~/opensource/scx-main
stt vm --sockets 2 --cores 4 --threads 2 \
  --scheduler-bin ./target/release/scx_mitosis \
  -- cgroup_steady --flags=llc,borrow
```

## Run experimental and compare

```sh
cd ~/opensource/scx
stt vm --sockets 2 --cores 4 --threads 2 \
  --scheduler-bin ./target/release/scx_mitosis \
  -- cgroup_steady --flags=llc,borrow
```

Compare output between the two runs.

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
```
