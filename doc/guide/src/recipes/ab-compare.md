# A/B Compare Branches

Compare scheduler behavior between two branches by running the
same `#[ktstr_test]` suite against each and diffing sidecar results.

## Setup worktrees

The examples below use the `scx` scheduler crate under
`~/opensource/scx`; substitute your own scheduler crate's path and
remote everywhere `scx` appears.

```sh
cd ~/opensource/scx

# Create worktree for the baseline branch
git worktree add ~/opensource/scx-main upstream/main
```

## Run baseline with sidecars

```sh
cd ~/opensource/scx-main
KTSTR_SIDECAR_DIR=./baseline cargo nextest run --workspace
```

## Run experimental with sidecars

```sh
cd ~/opensource/scx
KTSTR_SIDECAR_DIR=./current cargo nextest run --workspace
```

## Compare results

Diff the sidecar JSON files between the two directories. See
[Baselines](../running-tests/baselines.md) for the sidecar format.

Sanity-check that the two runs produced the same test set (same file
count and same test names):

```sh
diff <(ls baseline | sort) <(ls current | sort)
```

Compare pass/fail verdicts for each test that appears in both runs:

```sh
for f in baseline/*.ktstr.json; do
    name=$(basename "$f")
    [ -f "current/$name" ] || continue
    a=$(jq -r '.passed' "$f")
    b=$(jq -r '.passed' "current/$name")
    [ "$a" = "$b" ] || echo "$name: baseline=$a current=$b"
done
```

For deeper field-by-field comparison (scheduler telemetry, latency
percentiles, etc.), use `jq` to extract specific keys and diff those
between matching sidecar pairs.

## Cleanup

```sh
git worktree remove ~/opensource/scx-main
```
