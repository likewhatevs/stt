# Ops and Steps

The ops system is a composable way to express dynamic cgroup topology
changes. It replaces hand-written `Action::Custom` functions for most
dynamic scenarios.

## Op

An `Op` is an atomic operation on the cgroup topology:

| Op | Description |
|---|---|
| `AddCgroup` | Create a cgroup |
| `RemoveCgroup` | Stop workers and remove a cgroup |
| `SetCpuset` | Set a cgroup's cpuset via `CpusetSpec` |
| `ClearCpuset` | Remove cpuset constraints |
| `SwapCpusets` | Swap cpusets between two cgroups |
| `Spawn` | Fork workers into a cgroup |
| `StopCgroup` | Stop a cgroup's workers |
| `RandomizeAffinity` | Set random affinity on workers within the cgroup's cpuset |
| `SetAffinity` | Set explicit affinity on workers |
| `SpawnHost` | Spawn workers in the parent cgroup |
| `MoveAllTasks` | Move all tasks from one cgroup to another |

## CpusetSpec

`CpusetSpec` computes a cpuset from the topology at runtime:

```rust,ignore
pub enum CpusetSpec {
    Llc(usize),                          // All CPUs in an LLC
    Range { start_frac: f64, end_frac: f64 }, // Fraction of usable CPUs
    Disjoint { index: usize, of: usize },     // Equal disjoint partitions
    Overlap { index: usize, of: usize, frac: f64 }, // Overlapping partitions
    Exact(BTreeSet<usize>),              // Exact CPU set
}
```

All fractional specs operate on `usable_cpus()`, which reserves the
last CPU for the root cgroup when the topology has more than 2 CPUs.

## CgroupDef

`CgroupDef` bundles three ops that always go together: create cgroup,
set cpuset, spawn workers. It is the primary way to define cgroups in
ops-based scenarios.

```rust,ignore
let def = CgroupDef::named("cg_0")
    .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
    .workers(4)
    .work_type(WorkType::CpuSpin);
```

### Builder methods

- `.with_cpuset(CpusetSpec)` -- set the cpuset.
- `.workers(n)` -- set worker count.
- `.work_type(WorkType)` -- set work type (default: `CpuSpin`).
- `.sched_policy(SchedPolicy)` -- set Linux scheduling policy
  (default: `Normal`). See [Work Types](work-types.md#scheduling-policies).
- `.swappable(bool)` -- opt into gauntlet work type override.

### Work type overrides and swappable

`CgroupDef` has a `swappable` flag (default: `false`). When `true`
and a work type override is active (`Ctx.work_type_override`), the
override replaces this def's work type.

In contrast, the `Scenario`-level override (in `run_scenario()`) only
replaces `CpuSpin` work types. The two mechanisms serve different
scopes:

- **Scenario-level**: replaces `CpuSpin` in `CgroupWork.work_type`
- **CgroupDef-level**: replaces the work type when `swappable = true`

Both skip overrides to grouped work types when `num_workers` is not
divisible by the work type's group size.

Work type overrides apply only to `CgroupDef` setup, not to raw
`Op::Spawn`. `Op::Spawn` always uses the work type as given. Use
`CgroupDef` with `.swappable(true)` when the work type should
participate in gauntlet overrides.

## Step

A `Step` is a sequence of ops with a hold period:

```rust,ignore
pub struct Step {
    pub setup: Setup,   // CgroupDefs to create after ops
    pub ops: Vec<Op>,   // Operations to apply
    pub hold: HoldSpec, // How long to wait after
}
```

`Setup` is either `Defs(Vec<CgroupDef>)` or `Factory(fn(&Ctx) -> Vec<CgroupDef>)`.
`Vec<CgroupDef>` implements `Into<Setup>`, so you can write
`setup: vec![...].into()` instead of `setup: Setup::Defs(vec![...])`.

### Constructors

**`Step::new(ops, hold)`** -- creates a step with ops only (no
CgroupDef setup). Use when the step only applies dynamic operations
to an existing topology.

**`Step::with_defs(defs, hold)`** -- creates a step with CgroupDef
setup and a hold period. The primary constructor for steps that
create cgroups with workers.

**`Step::with_ops(self, ops)`** -- replaces the ops on a step
(builder method). Chain after `with_defs` to add dynamic operations
to a step that also creates cgroups.

## HoldSpec

How long to hold after a step completes:

| Variant | Description |
|---|---|
| `Frac(f64)` | Fraction of the total scenario duration |
| `Fixed(Duration)` | Fixed time |
| `Loop { interval }` | Repeat ops at interval until time runs out |

`HoldSpec::FULL` is a constant for `Frac(1.0)` (hold for the full
scenario duration).

## execute_defs

`execute_defs(ctx, defs)` is a convenience wrapper for the common
pattern of creating cgroups and running them for the full duration:

```rust,ignore
execute_defs(ctx, vec![
    CgroupDef::named("cg_0").workers(4),
    CgroupDef::named("cg_1").workers(4),
])
```

Equivalent to `execute_steps(ctx, vec![Step::with_defs(defs, HoldSpec::FULL)])`.

## execute_steps

`execute_steps(ctx, steps)` runs a step sequence:

1. For each step: apply ops, then apply setup (create cgroups from
   `CgroupDef`s), hold for the specified duration. Ops run first so
   parent cgroups can be created before children are spawned.
   `Loop` steps reverse this: setup runs once before the loop, then
   ops repeat at the specified interval.
2. Check scheduler liveness between steps.
3. After all steps: collect worker reports and run verification.
4. Writes stimulus events to the SHM ring buffer for timeline analysis.

## execute_steps_with

`execute_steps_with(ctx, steps, assertions)` is the same as
`execute_steps` but accepts an explicit
[`Assert`](verification.md#assert-struct) for worker checks.
`execute_steps` is a convenience wrapper that passes `None`.

```rust,ignore
use stt::prelude::*;
use stt::scenario::ops::execute_steps_with;

fn my_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let assertions = Assert::NONE
        .check_not_starved()
        .max_gap_ms(3000);

    let steps = vec![/* ... */];
    execute_steps_with(ctx, steps, Some(&assertions))
}
```

When `assertions` is `Some`, the provided `Assert` overrides `ctx.assert`
for worker checks. When `None`, uses `ctx.assert` (the merged
three-layer config: `default_checks` -> scheduler -> per-test).

For a complete example using ops/steps, see
[Write a Dynamic Scenario](../recipes/dynamic-scenario.md).
