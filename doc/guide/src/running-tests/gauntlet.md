# Gauntlet

Gauntlet runs every scenario across 13 topology presets in parallel
VMs.

## Two entry points

There are two ways to run gauntlet mode:

**`cargo stt vm --gauntlet`** -- runs data-driven scenarios (from
`all_scenarios()`) across topology presets. Each VM runs `stt run`
inside it.

```sh
cargo stt vm --gauntlet --parallel 4
```

To test a scheduler, use `-p` to build and inject it:

```sh
cargo stt vm --gauntlet -p scx_mitosis --parallel 4
```

**`cargo stt gauntlet`** -- runs `#[stt_test]` integration tests across
topology presets. Each VM runs the test binary with `--stt-test-fn` and
`--stt-topo` arguments.

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

## Flag override

Override the default per-scenario flag profiles:

```sh
cargo stt vm --gauntlet --flags=borrow,rebal
```

## Retry on failure

Failed scenarios are retried automatically. `--retries N` sets the
total number of attempts (default: 3):

```sh
cargo stt vm --gauntlet --retries 5
```
