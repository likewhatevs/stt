# Summary

[Overview](overview.md)

# Guide

- [Getting Started](getting-started.md)
- [Running Tests](running-tests.md)
  - [Single Scenario](running-tests/single-scenario.md)
  - [Gauntlet](running-tests/gauntlet.md)
  - [Baselines](running-tests/baselines.md)
  - [ktstr-host](running-tests/ktstr-host.md)
  - [Auto-Repro](running-tests/auto-repro.md)
  - [BPF Verifier](running-tests/verifier.md)
- [Core Concepts](concepts.md)
  - [Scenarios](concepts/scenarios.md)
  - [Flags](concepts/flags.md)
  - [Work Types](concepts/work-types.md)
  - [Verification](concepts/verification.md)
  - [Ops and Steps](concepts/ops.md)
  - [TestTopology](concepts/topology.md)
  - [Performance Mode](concepts/performance-mode.md)
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
  - [Write a Dynamic Scenario](recipes/dynamic-scenario.md)
  - [Customize Verification](recipes/custom-verification.md)
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

- [Troubleshooting](troubleshooting.md)
- [Environment Variables](reference/environment-variables.md)
