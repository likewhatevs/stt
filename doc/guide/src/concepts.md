# Core Concepts

ktstr tests compose from four layers:

1. **Scenarios** -- what to test: cgroup layout, CPU partitioning,
   workloads, custom logic.

2. **Flags** -- which scheduler features to enable for each run.

3. **Work types** -- what each worker process does: CPU spin, yield,
   I/O, bursty patterns, pipe-based IPC.

4. **Checking** -- how to evaluate results: starvation, fairness,
   isolation, scheduling gaps, monitor thresholds.

These compose orthogonally. A scenario runs with every valid flag
combination, and checks apply uniformly across all runs.

Four supporting concepts complete the picture:
[Ops and Steps](concepts/ops.md) is the primary API for defining
scenarios -- most tests use `CgroupDef` and `execute_defs` from
this module. [TestTopology](concepts/topology.md) provides CPU and
LLC layout for cpuset partitioning.
[Performance Mode](concepts/performance-mode.md) applies host-side
isolation for noise-sensitive measurements.
[Resource Budget](concepts/resource-budget.md) describes the
`--cpu-cap` tier that coordinates concurrent no-perf-mode VMs and
kernel builds via LLC flocks and cgroup v2 cpuset sandboxes.
