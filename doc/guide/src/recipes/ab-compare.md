# A/B Compare Branches

Compare scheduler behavior between two branches by running the
same `#[stt_test]` suite against each and diffing sidecar results.

## Setup worktrees

```sh
cd ~/opensource/scx

# Create worktree for the baseline branch
git worktree add ~/opensource/scx-main upstream/main
```

## Run baseline with sidecars

```sh
cd ~/opensource/scx-main
STT_SIDECAR_DIR=./baseline cargo nextest run --workspace
```

## Run experimental with sidecars

```sh
cd ~/opensource/scx
STT_SIDECAR_DIR=./current cargo nextest run --workspace
```

## Compare results

Diff the sidecar JSON files between the two directories. See
[Baselines](../running-tests/baselines.md) for the sidecar format.

Compare test pass/fail counts between runs:

```sh
diff <(ls baseline/*.stt.json | wc -l) <(ls current/*.stt.json | wc -l)
```

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
```
