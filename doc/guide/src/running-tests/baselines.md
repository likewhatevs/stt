# Baselines

stt can save test results as baselines and compare subsequent runs
against them.

## Saving a baseline

```sh
cargo stt vm --gauntlet --save-baseline before.json
```

Saves `GauntletBaseline` (a set of `GauntletRow` entries) as JSON.

## Comparing against a baseline

```sh
cargo stt vm --gauntlet --compare before.json
```

`compare_baselines()` diffs two sets of gauntlet rows. The output
groups cells into four categories:

- **Regressions** -- pass rate dropped, spread increased, or gap
  increased beyond tolerance thresholds.
- **Improvements** -- pass rate increased or metrics improved beyond
  tolerance.
- **Removed** -- cells present in the baseline but missing from the
  current run.
- **New** -- cells present in the current run but missing from the
  baseline.

The summary line shows how many cells were unchanged (within
tolerance).

## Integration tests

```sh
cargo stt test --save-baseline baseline.json
cargo stt test --compare baseline.json
```

Sidecar results (`SidecarResult` JSON files) are written to
`STT_SIDECAR_DIR` and collected for analysis.
