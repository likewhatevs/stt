# Write a Dynamic Scenario

Use [ops/steps](../concepts/ops.md) to express cgroup topology changes
without hand-written `Action::Custom` functions.

## Basic: two phases with cpuset resize

```rust,ignore
use stt::prelude::*;
use stt::scenario::ops::*;

fn my_resize_scenario(ctx: &Ctx) -> Result<VerifyResult> {
    let steps = vec![
        // Phase 1: two disjoint cgroups, hold for half the duration
        Step {
            setup: vec![
                CgroupDef::named("cell_0")
                    .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
                    .workers(4),
                CgroupDef::named("cell_1")
                    .with_cpuset(CpusetSpec::Disjoint { index: 1, of: 2 })
                    .workers(4),
            ].into(),
            ops: vec![],
            hold: HoldSpec::Frac(0.5),
        },
        // Phase 2: resize cpusets to overlap
        Step {
            setup: vec![].into(),
            ops: vec![
                Op::SetCpuset {
                    cgroup: "cell_0".into(),
                    cpus: CpusetSpec::Overlap { index: 0, of: 2, frac: 0.5 },
                },
                Op::SetCpuset {
                    cgroup: "cell_1".into(),
                    cpus: CpusetSpec::Overlap { index: 1, of: 2, frac: 0.5 },
                },
            ],
            hold: HoldSpec::Frac(0.5),
        },
    ];
    execute_steps(ctx, steps)
}
```

## Using Traverse for random topology walks

```rust,ignore
use stt::prelude::*;
use stt::scenario::ops::*;

fn my_traverse_scenario(ctx: &Ctx) -> Result<VerifyResult> {
    let traverse = Traverse {
        seed: Some(42),
        cgroup_count: 2..=4,
        layouts: vec![Layout::Disjoint, Layout::Overlap(0.2, 0.5)],
        phases: 5,
        phase_duration: std::time::Duration::from_millis(500),
        settle: std::time::Duration::from_millis(200),
        persistent_cells: 1,
        cell_workloads: vec![WorkloadConfig::default()],
    };
    let steps = traverse.generate(ctx);
    execute_steps(ctx, steps)
}
```

## Registering

Register the scenario in `all_scenarios()`. Set `num_cells` to 0 and
`action` to `Custom` -- the step executor handles all cgroup creation
via `CgroupDef`:

```rust,ignore
Scenario {
    name: "my_resize",
    category: "dynamic",
    description: "Resize cpusets from disjoint to overlapping",
    required_flags: &[],
    excluded_flags: &[],
    num_cells: 0,
    cpuset_mode: CpusetMode::None,
    cell_works: vec![],
    action: Action::Custom(my_resize_scenario),
}
```

See [Scenarios](../concepts/scenarios.md) for the full `Scenario` struct
and [Ops and Steps](../concepts/ops.md) for `CpusetSpec`, `HoldSpec`,
and `Traverse`.
