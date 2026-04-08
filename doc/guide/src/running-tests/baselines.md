# Baselines

stt can save test results as baselines and compare subsequent runs
against them.

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

Sidecar results (`SidecarResult` JSON files) are written to
`STT_SIDECAR_DIR` and collected for analysis.
