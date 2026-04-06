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

`compare_baselines()` diffs two sets of gauntlet rows, showing
regressions and improvements.

## Integration tests

```sh
cargo stt test --save-baseline baseline.json
cargo stt test --compare baseline.json
```

Sidecar results (`SidecarResult` JSON files) are written to
`STT_SIDECAR_DIR` and collected for analysis.
