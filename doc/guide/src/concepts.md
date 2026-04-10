# Core Concepts

stt tests compose from four layers:

1. **Scenarios** -- what to test: cgroup layout, CPU partitioning,
   workloads, custom logic.

2. **Flags** -- which scheduler features to enable for each run.

3. **Work types** -- what each worker process does: CPU spin, yield,
   I/O, bursty patterns, pipe-based IPC.

4. **Verification** -- how to evaluate results: starvation, fairness,
   isolation, scheduling gaps, monitor thresholds.

These compose orthogonally. A scenario runs with every valid flag
combination, and verification checks apply uniformly across all runs.

## Why this design

Scenarios are data-driven structs, not test functions. The same scenario
definition works across different schedulers, flag combinations, and
topologies without code changes.

Flags are defined via `#[derive(Scheduler)]` on an enum, generating
typed `FlagDecl` statics with dependency constraints (`steal` requires
`llc`). Dependencies are declared statically and enforced at profile
generation time -- invalid combinations are rejected automatically.

Assertion layers merge: `Assert::default_checks()` provides
baselines, each `Scheduler` can override thresholds, and individual
`#[stt_test]` attributes can override further. This eliminates
per-test boilerplate while allowing scheduler-specific tuning.
