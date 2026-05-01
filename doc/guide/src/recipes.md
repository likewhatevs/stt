# Recipes

Standalone examples for common tasks. Each recipe is self-contained.

- [Test a new scheduler](recipes/test-new-scheduler.md) -- end-to-end
  from binary to integration tests
- [Investigate a crash](recipes/investigate-crash.md) -- auto-repro,
  reading BPF probe output
- [A/B compare branches](recipes/ab-compare.md) -- worktree setup,
  run and compare
- [Capture and compare host state](recipes/host-state.md) --
  `cargo ktstr show-host` snapshot diff for kernel / sched_\*
  tunable / NUMA layout drift
- [Diagnose a slow scheduler with ctprof](recipes/diagnose-slow-scheduler.md) --
  per-thread profile diff via `ktstr ctprof capture` /
  `compare`, with the taskstats off-CPU lens
- [Customize checking](recipes/custom-checking.md) -- scheduler
  thresholds, per-test overrides
- [Benchmarking and negative tests](recipes/benchmarking-tests.md) --
  performance gates, intentional degradation, Assert checks
