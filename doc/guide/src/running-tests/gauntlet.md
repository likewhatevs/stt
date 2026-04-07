# Gauntlet

Gauntlet runs every scenario across 13 topology presets in parallel
VMs.

## Two entry points

There are two ways to run gauntlet mode:

**`cargo stt vm --gauntlet`** -- runs data-driven scenarios (from
`all_scenarios()`) across topology presets.

```sh
cargo stt vm --gauntlet --parallel 4
```

To test a scheduler, use `-p` to build and inject it:

```sh
cargo stt vm --gauntlet -p scx_mitosis --parallel 4
```

**`cargo stt gauntlet`** -- runs `#[stt_test]` integration tests across
topology presets. No separate gauntlet configuration exists. Each
test's `#[stt_test]` attributes -- `required_flags`, `excluded_flags`,
`min_sockets`, `min_llcs`, `requires_smt`, `min_cpus`, and its
scheduler's flag declarations -- fully determine which cells of the
topology x flag_profile matrix that test populates. The test
definition is the spec.

`cargo stt gauntlet` discovers all `#[stt_test]` functions by building
the test binary and querying it via `--stt-list`. For each discovered
test, it:

1. Filters topology presets against the test's constraints
   (`min_sockets`, `min_llcs`, `requires_smt`, `min_cpus`).
2. Generates valid flag profiles from the scheduler's flag declarations
   constrained by the test's `required_flags` and `excluded_flags`.
3. Creates one VM job per (test, matching topology, valid flag profile)
   triple.

```sh
cargo stt gauntlet --parallel 4
```

Use `cargo stt vm --gauntlet` for the catalog scenarios. Use
`cargo stt gauntlet` for `#[stt_test]` functions.

`--parallel N` controls concurrent VMs (default: host CPUs / 8).

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

`#[stt_test]` functions can declare topology constraints that filter
which presets they run on. For example:

```rust,ignore
#[stt_test(
    sockets = 2, cores = 4, threads = 2,
    scheduler = MY_SCHED,
    min_llcs = 2,
    requires_smt = true,
    required_flags = ["llc"],
)]
fn cross_llc_test(ctx: &Ctx) -> Result<VerifyResult> { /* ... */ }
```

This test skips `tiny-1llc` (1 LLC), `odd-3llc` (no SMT), and all
non-SMT presets. Every generated flag profile includes `llc`. The
test definition controls exactly which gauntlet cells it runs in.

See [Topology Constraints](../writing-tests/stt-test-macro.md#topology-constraints)
and [Flag Constraints](../writing-tests/stt-test-macro.md#flag-constraints)
for the full attribute lists.

## Flag profiles

By default, gauntlet runs each test with all valid flag combinations
generated from the scheduler's flag declarations and the test's
`required_flags` / `excluded_flags` constraints. This adds a flag
profile dimension to the matrix: tests x topologies x flag_profiles.

Override with `--flags` to run a single profile:

```sh
cargo stt vm --gauntlet --flags=borrow,rebal
cargo stt gauntlet --flags=borrow,rebal
```

## Work type override

`--work-types` adds a work type dimension. Each test runs once per
specified work type (overriding the default `CpuSpin`):

```sh
cargo stt vm --gauntlet --work-types=CpuSpin,Bursty
cargo stt gauntlet --work-types=CpuSpin,Bursty
```

## Retry on failure

`cargo stt vm --gauntlet` retries failed scenarios automatically.
`--retries N` sets the total number of attempts (default: 3):

```sh
cargo stt vm --gauntlet --retries 5
```

`cargo stt gauntlet` does not have `--retries`.
