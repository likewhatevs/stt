# Gauntlet

The gauntlet runs every test across 19 topology presets (11 on aarch64)
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

| Preset | Topology | CPUs | LLCs | Description |
|---|---|---|---|---|
| `tiny-1llc` | 1s4c1t | 4 | 1 | Single LLC |
| `tiny-2llc` | 2s2c1t | 4 | 2 | Minimal multi-LLC |
| `odd-3llc` | 3s3c1t | 9 | 3 | Odd CPU count |
| `odd-5llc` | 5s3c1t | 15 | 5 | Prime LLC count |
| `odd-7llc` | 7s2c1t | 14 | 7 | Prime LLC count |
| `smt-2llc` | 2s2c2t | 8 | 2 | SMT enabled |
| `smt-3llc` | 3s2c2t | 12 | 3 | SMT, 3 LLCs |
| `medium-4llc` | 4s4c2t | 32 | 4 | Medium topology |
| `medium-8llc` | 8s4c2t | 64 | 8 | Medium, many LLCs |
| `large-4llc` | 4s16c2t | 128 | 4 | Large, few LLCs |
| `large-8llc` | 8s8c2t | 128 | 8 | Large, many LLCs |
| `near-max-llc` | 15s8c2t | 240 | 15 | Near maximum |
| `max-cpu` | 14s9c2t | 252 | 14 | Near KVM vCPU limit |
| `medium-4llc-nosmt` | 4s8c1t | 32 | 4 | Medium, no SMT |
| `medium-8llc-nosmt` | 8s8c1t | 64 | 8 | Medium, many LLCs, no SMT |
| `large-4llc-nosmt` | 4s32c1t | 128 | 4 | Large, no SMT |
| `large-8llc-nosmt` | 8s16c1t | 128 | 8 | Large, many LLCs, no SMT |
| `near-max-llc-nosmt` | 15s16c1t | 240 | 15 | Near maximum, no SMT |
| `max-cpu-nosmt` | 14s18c1t | 252 | 14 | Near KVM vCPU limit, no SMT |

Topology format: `{sockets}s{cores_per_socket}c{threads_per_core}t`.
Presets are defined in `gauntlet_presets()`.

> **aarch64:** ARM64 CPUs do not have SMT. Presets with
> `threads_per_core > 1` are excluded on aarch64, leaving 11 presets
> (the 5 small presets plus 6 `-nosmt` variants).

## Constraint filtering

`#[ktstr_test]` topology constraints filter which presets a test runs
on. A preset is skipped when any constraint is not met:

- `sockets < min_sockets`
- `num_llcs() < min_llcs`
- `requires_smt` and `threads_per_core < 2`
- `total_cpus() < min_cpus`

Example:

```rust,ignore
#[ktstr_test(
    sockets = 2, cores = 4, threads = 2,
    scheduler = MY_SCHED,
    min_llcs = 2,
    requires_smt = true,
    required_flags = ["llc"],
)]
fn cross_llc_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

This test skips `tiny-1llc` (1 LLC) and all non-SMT presets. It runs
on 8 presets: `smt-2llc`, `smt-3llc`, `medium-4llc`, `medium-8llc`,
`large-4llc`, `large-8llc`, `near-max-llc`, `max-cpu`. Every
generated flag profile includes `llc`.

See [Topology Constraints](../writing-tests/ktstr-test-macro.md#topology-constraints)
and [Flag Constraints](../writing-tests/ktstr-test-macro.md#flag-constraints)
for the full attribute lists.

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

Each gauntlet VM gets `max(cpus * 64, 256, entry.memory_mb)` MB of RAM,
where `cpus` is the preset's total CPU count and `entry.memory_mb` is
the per-test minimum (default 2048). For `max-cpu` (252 CPUs) this
yields at least 16128 MB.
