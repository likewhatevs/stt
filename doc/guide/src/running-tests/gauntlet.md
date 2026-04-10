# Gauntlet

The gauntlet runs every scenario across 13 topology presets.
Gauntlet variants are prefixed with `gauntlet/` and ignored by default.

```sh
# Run only base tests (default)
cargo nextest run

# Run only gauntlet variants
cargo nextest run --run-ignored ignored-only -E 'test(gauntlet/)'

# Run everything
cargo nextest run --run-ignored all
```

`#[ktstr_test]` functions declare topology constraints that filter which
presets they run on. Each test's attributes -- `required_flags`,
`excluded_flags`, `min_sockets`, `min_llcs`, `requires_smt`,
`min_cpus`, and its scheduler's flag declarations -- fully determine
which cells of the topology x flag_profile matrix that test populates.
The test definition is the spec.

## Topology presets

| Preset | CPUs | LLCs | Description |
|---|---|---|---|
| `tiny-1llc` | 4 | 1 | Single LLC |
| `tiny-2llc` | 4 | 2 | Minimal multi-LLC |
| `odd-3llc` | 9 | 3 | Odd CPU count |
| `odd-5llc` | 15 | 5 | Prime LLC count |
| `odd-7llc` | 14 | 7 | Prime LLC count |
| `smt-2llc` | 8 | 2 | SMT enabled |
| `smt-3llc` | 12 | 3 | SMT, 3 LLCs |
| `medium-4llc` | 32 | 4 | Medium topology |
| `medium-8llc` | 64 | 8 | Medium, many LLCs |
| `large-4llc` | 128 | 4 | Large, few LLCs |
| `large-8llc` | 128 | 8 | Large, many LLCs |
| `near-max-llc` | 240 | 15 | Near maximum |
| `max-cpu` | 252 | 14 | Near i440fx limit |

Presets are defined in `gauntlet_presets()`.

`#[ktstr_test]` functions can declare topology constraints that filter
which presets they run on. For example:

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

This test skips `tiny-1llc` (1 LLC), `odd-3llc` (no SMT), and all
non-SMT presets. Every generated flag profile includes `llc`. The
test definition controls exactly which gauntlet cells it runs in.

See [Topology Constraints](../writing-tests/ktstr-test-macro.md#topology-constraints)
and [Flag Constraints](../writing-tests/ktstr-test-macro.md#flag-constraints)
for the full attribute lists.

## Budget interaction

When `KTSTR_BUDGET_SECS` is set, the budget selector considers gauntlet
variants alongside base tests. Gauntlet variants contribute topology
and flag profile diversity to the feature coverage model, so a
budget-constrained run may select a mix of base tests (cheap, default
topology) and gauntlet variants (expensive, diverse topologies)
depending on the budget.

See [Budget-based test selection](../running-tests.md#budget-based-test-selection).

## Flag profiles

By default, gauntlet runs each test with all valid flag combinations
generated from the scheduler's flag declarations and the test's
`required_flags` / `excluded_flags` constraints. This adds a flag
profile dimension to the matrix: tests x topologies x flag_profiles.
