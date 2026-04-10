# Write a Dynamic Scenario

Use [ops/steps](../concepts/ops.md) to express cgroup topology changes
without hand-written `Action::Custom` functions.

## Basic: two phases with cpuset resize

```rust,ignore
use scx_ktstr::prelude::*;

fn my_resize_scenario(ctx: &Ctx) -> Result<AssertResult> {
    let steps = vec![
        // Phase 1: two disjoint cgroups, hold for half the duration
        Step::with_defs(
            vec![
                CgroupDef::named("cg_0")
                    .with_cpuset(CpusetSpec::disjoint(0, 2))
                    .workers(4),
                CgroupDef::named("cg_1")
                    .with_cpuset(CpusetSpec::disjoint(1, 2))
                    .workers(4),
            ],
            HoldSpec::Frac(0.5),
        ),
        // Phase 2: resize cpusets to overlap
        Step::new(
            vec![
                Op::set_cpuset("cg_0", CpusetSpec::overlap(0, 2, 0.5)),
                Op::set_cpuset("cg_1", CpusetSpec::overlap(1, 2, 0.5)),
            ],
            HoldSpec::Frac(0.5),
        ),
    ];
    execute_steps(ctx, steps)
}
```

## Registering (scx-ktstr contributors only)

This section applies to contributing scenarios to scx-ktstr's internal
catalog. External test suites call scenario functions directly from
`#[ktstr_test]` -- no registration needed.

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
