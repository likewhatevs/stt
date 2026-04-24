# Summary

[Overview](overview.md)
[Features](features.md)

# Guide

- [Getting Started](getting-started.md)
- [Running Tests](running-tests.md)
  - [Single Scenario](running-tests/single-scenario.md)
  - [Gauntlet](running-tests/gauntlet.md)
  - [Runs](running-tests/runs.md)
  - [ktstr](running-tests/ktstr.md)
  - [cargo-ktstr](running-tests/cargo-ktstr.md)
  - [Auto-Repro](running-tests/auto-repro.md)
  - [BPF Verifier](running-tests/verifier.md)
- [Core Concepts](concepts.md)
  - [Scenarios](concepts/scenarios.md)
  - [Flags](concepts/flags.md)
  - [Work Types](concepts/work-types.md)
  - [Checking](concepts/checking.md)
  - [Ops and Steps](concepts/ops.md)
  - [TestTopology](concepts/topology.md)
  - [MemPolicy](concepts/mem-policy.md)
  - [Performance Mode](concepts/performance-mode.md)
  - [Resource Budget](concepts/resource-budget.md)
- [Writing Tests](writing-tests.md)
  - [The #\[ktstr_test\] Macro](writing-tests/ktstr-test-macro.md)
  - [Custom Scenarios](writing-tests/custom-scenarios.md)
  - [Scheduler Definitions](writing-tests/scheduler-definitions.md)
  - [Gauntlet Tests](writing-tests/gauntlet-tests.md)

# Recipes

- [Recipes](recipes.md)
  - [Test a New Scheduler](recipes/test-new-scheduler.md)
  - [Investigate a Crash](recipes/investigate-crash.md)
  - [A/B Compare Branches](recipes/ab-compare.md)
  - [Capture and Compare Host State](recipes/host-state.md)
  - [Write a Dynamic Scenario](recipes/dynamic-scenario.md)
  - [Customize Checking](recipes/custom-checking.md)
  - [Benchmarking and Negative Tests](recipes/benchmarking-tests.md)

# Architecture

- [Architecture Overview](architecture.md)
  - [VMM](architecture/vmm.md)
  - [Monitor](architecture/monitor.md)
  - [Worker Processes](architecture/workers.md)
  - [WorkloadHandle](architecture/workload-handle.md)
  - [CgroupManager](architecture/cgroup-manager.md)
  - [CgroupGroup](architecture/cgroup-group.md)

# Reference

- [CI](ci.md)
- [Troubleshooting](troubleshooting.md)
- [Environment Variables](reference/environment-variables.md)
- [Host-State Profiler](reference/host-state-profiler.md)
