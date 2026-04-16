# Gauntlet

The gauntlet runs every test across 24 topology presets (14 on aarch64)
and all valid flag profiles. Gauntlet variants are prefixed with `gauntlet/` and
ignored by default.

```sh
# Run only base tests (default)
cargo nextest run

# Run only gauntlet variants
cargo nextest run --run-ignored ignored-only -E 'test(gauntlet/)'

# Run everything
cargo nextest run --run-ignored all
```

Tests with `host_only = true` never produce gauntlet variants (topology
variation is meaningless without a VM).

## Variant naming

Each gauntlet variant is named `gauntlet/{test_name}/{preset}/{profile}`:

- `{test_name}` -- the `#[ktstr_test]` function name
- `{preset}` -- one of the topology preset names below
- `{profile}` -- `default` when no flags are active, otherwise the
  active flags joined with `+` (e.g. `llc+borrow`)

To run a single variant:

```sh
cargo nextest run --run-ignored ignored-only \
  -E 'test(=gauntlet/my_test/smt-2llc/default)'
```

## Topology presets

| Preset | Topology | CPUs | LLCs | NUMA | Description |
|---|---|---|---|---|---|
| `tiny-1llc` | 1n1l4c1t | 4 | 1 | 1 | Single LLC |
| `tiny-2llc` | 1n2l2c1t | 4 | 2 | 1 | Minimal multi-LLC |
| `odd-3llc` | 1n3l3c1t | 9 | 3 | 1 | Odd CPU count |
| `odd-5llc` | 1n5l3c1t | 15 | 5 | 1 | Prime LLC count |
| `odd-7llc` | 1n7l2c1t | 14 | 7 | 1 | Prime LLC count |
| `smt-2llc` | 1n2l2c2t | 8 | 2 | 1 | SMT enabled |
| `smt-3llc` | 1n3l2c2t | 12 | 3 | 1 | SMT, 3 LLCs |
| `medium-4llc` | 1n4l4c2t | 32 | 4 | 1 | Medium topology |
| `medium-8llc` | 1n8l4c2t | 64 | 8 | 1 | Medium, many LLCs |
| `large-4llc` | 1n4l16c2t | 128 | 4 | 1 | Large, few LLCs |
| `large-8llc` | 1n8l8c2t | 128 | 8 | 1 | Large, many LLCs |
| `near-max-llc` | 1n15l8c2t | 240 | 15 | 1 | Near maximum |
| `max-cpu` | 1n14l9c2t | 252 | 14 | 1 | Near KVM vCPU limit |
| `medium-4llc-nosmt` | 1n4l8c1t | 32 | 4 | 1 | Medium, no SMT |
| `medium-8llc-nosmt` | 1n8l8c1t | 64 | 8 | 1 | Medium, many LLCs, no SMT |
| `large-4llc-nosmt` | 1n4l32c1t | 128 | 4 | 1 | Large, no SMT |
| `large-8llc-nosmt` | 1n8l16c1t | 128 | 8 | 1 | Large, many LLCs, no SMT |
| `near-max-llc-nosmt` | 1n15l16c1t | 240 | 15 | 1 | Near maximum, no SMT |
| `max-cpu-nosmt` | 1n14l18c1t | 252 | 14 | 1 | Near KVM vCPU limit, no SMT |
| `numa2-4llc` | 2n4l4c1t | 16 | 4 | 2 | Multi-NUMA, 2 nodes |
| `numa2-8llc` | 2n8l8c2t | 128 | 8 | 2 | Multi-NUMA, 2 nodes, SMT |
| `numa2-8llc-nosmt` | 2n8l16c1t | 128 | 8 | 2 | Multi-NUMA, 2 nodes, no SMT |
| `numa4-8llc` | 4n8l4c1t | 32 | 8 | 4 | Multi-NUMA, 4 nodes |
| `numa4-12llc` | 4n12l8c2t | 192 | 12 | 4 | Multi-NUMA, 4 nodes, SMT |

Topology format: `{numa_nodes}n{llcs}l{cores_per_llc}c{threads_per_core}t`
(e.g. `1n2l4c2t` = 1 NUMA node, 2 LLCs, 4 cores per LLC, 2 threads
per core = 16 CPUs). Presets are defined in `gauntlet_presets()`.
Multi-NUMA presets are excluded by default
(`max_numa_nodes: Some(1)` in `TopologyConstraints::DEFAULT`), so
tests opt in to NUMA testing by raising `max_numa_nodes`.

> **aarch64:** ARM64 CPUs do not have SMT. Presets with
> `threads_per_core > 1` are excluded on aarch64, leaving 14 presets
> (the 5 small presets, 6 `-nosmt` variants, and 3 non-SMT NUMA presets).

## Constraint filtering

`#[ktstr_test]` topology constraints filter which presets a test runs
on. A preset is skipped when any constraint is not met:

- `num_numa_nodes() < min_numa_nodes`
- `max_numa_nodes` is set and `num_numa_nodes() > max_numa_nodes`
- `num_llcs() < min_llcs`
- `max_llcs` is set and `num_llcs() > max_llcs`
- `requires_smt` and `threads_per_core < 2`
- `total_cpus() < min_cpus`
- `max_cpus` is set and `total_cpus() > max_cpus`

See [Topology Constraints](../writing-tests/ktstr-test-macro.md#topology-constraints)
for the full attribute table and
[Gauntlet Tests](../writing-tests/gauntlet-tests.md#worked-example)
for a worked example showing which presets survive a given constraint
set.

## Flag profiles

Gauntlet runs each test with all valid flag combinations generated from
the scheduler's flag declarations and the test's `required_flags` /
`excluded_flags` constraints. This adds a flag profile dimension:
tests x topologies x flag_profiles.

See [Flags](../concepts/flags.md) and
[Gauntlet Tests](../writing-tests/gauntlet-tests.md) for how profiles
are generated.

## Budget interaction

When `KTSTR_BUDGET_SECS` is set, greedy coverage maximization selects
the most diverse set of test configurations within the time budget.
Each candidate test is represented as a feature bitset (CPU count
bucket, LLC count, SMT vs non-SMT, flag profile, etc.). The selector
greedily picks tests that cover the most uncovered feature bits per
estimated second. The result is a mix of base tests and gauntlet
variants that maximizes configuration diversity within the budget.

See [Budget-based test selection](../running-tests.md#budget-based-test-selection).

## Memory allocation

Each gauntlet VM gets `max(topology_mb, initramfs_floor)` MB of RAM,
where `topology_mb = max(cpus * 64, 256, entry.memory_mb)` is the
topology-requested minimum and `initramfs_floor` is computed from
the actual initramfs size after build. For `max-cpu` (252 CPUs) the
topology minimum is at least 16128 MB.
