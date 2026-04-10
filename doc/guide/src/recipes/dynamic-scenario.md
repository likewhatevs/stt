# Write a Dynamic Scenario

Use [ops/steps](../concepts/ops.md) to express cgroup topology changes
without hand-written `Action::Custom` functions.

## Basic: two phases with cpuset resize

```rust,ignore
use stt::prelude::*;
use stt::scenario::ops::*;

fn my_resize_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        // Phase 1: two disjoint cgroups, hold for half the duration
        Step {
            setup: vec![
                CgroupDef::named("cg_0")
                    .with_cpuset(CpusetSpec::Disjoint { index: 0, of: 2 })
                    .workers(4),
                CgroupDef::named("cg_1")
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
                    cgroup: "cg_0".into(),
                    cpus: CpusetSpec::Overlap { index: 0, of: 2, frac: 0.5 },
                },
                Op::SetCpuset {
                    cgroup: "cg_1".into(),
                    cpus: CpusetSpec::Overlap { index: 1, of: 2, frac: 0.5 },
                },
            ],
            hold: HoldSpec::Frac(0.5),
        },
    ];
    execute_steps(ctx, steps)
}
```

## Registering

Register the scenario in `all_scenarios()`. Set `num_cgroups` to 0 and
`action` to `Custom` -- the step executor handles all cgroup creation
via `CgroupDef`:

```rust,ignore
Scenario {
    name: "my_resize",
    category: "dynamic",
    description: "Resize cpusets from disjoint to overlapping",
    required_flags: &[],
    excluded_flags: &[],
    num_cgroups: 0,
    cpuset_mode: CpusetMode::None,
    cgroup_works: vec![],
    action: Action::Custom(my_resize_scenario),
}
```

See [Scenarios](../concepts/scenarios.md) for the full `Scenario` struct
and [Ops and Steps](../concepts/ops.md) for `CpusetSpec` and `HoldSpec`.
