# Gauntlet Tests

The gauntlet expands each `#[ktstr_test]` into a matrix of
test x topology_preset x flag_profile variants. The test definition
controls which cells of the matrix it populates.

## Controlling topology coverage

Topology constraints filter which of the 24 gauntlet presets (14 on
aarch64) a test runs on. Set constraints in `#[ktstr_test]` attributes:

| Attribute | Type | Default | Effect |
|---|---|---|---|
| `min_llcs` | `u32` | 1 | Skip presets with fewer LLCs |
| `max_llcs` | `Option<u32>` | 12 | Skip presets with more LLCs |
| `min_cpus` | `u32` | 1 | Skip presets with fewer total CPUs |
| `max_cpus` | `Option<u32>` | 192 | Skip presets with more total CPUs |
| `min_numa_nodes` | `u32` | 1 | Skip presets with fewer NUMA nodes |
| `max_numa_nodes` | `Option<u32>` | 1 | Skip presets with more NUMA nodes |
| `requires_smt` | `bool` | false | Skip presets with `threads_per_core < 2` |

A preset is included only when all constraints are satisfied. Multi-NUMA
presets are excluded by default (`max_numa_nodes = 1`).

### Worked example

A test with `min_llcs = 2`, `requires_smt = true`, and default
`max_numa_nodes = 1`:

| Preset | Topology | LLCs | SMT | NUMA | Included |
|---|---|---|---|---|---|
| `tiny-1llc` | 1s1l4c1t | 1 | no | 1 | no (1 LLC) |
| `tiny-2llc` | 1s2l2c1t | 2 | no | 1 | no (no SMT) |
| `odd-3llc` | 1s3l3c1t | 3 | no | 1 | no (no SMT) |
| `odd-5llc` | 1s5l3c1t | 5 | no | 1 | no (no SMT) |
| `odd-7llc` | 1s7l2c1t | 7 | no | 1 | no (no SMT) |
| `smt-2llc` | 1s2l2c2t | 2 | yes | 1 | **yes** |
| `smt-3llc` | 1s3l2c2t | 3 | yes | 1 | **yes** |
| `medium-4llc` | 1s4l4c2t | 4 | yes | 1 | **yes** |
| `medium-8llc` | 1s8l4c2t | 8 | yes | 1 | **yes** |
| `large-4llc` | 1s4l16c2t | 4 | yes | 1 | **yes** |
| `large-8llc` | 1s8l8c2t | 8 | yes | 1 | **yes** |
| `near-max-llc` | 1s15l8c2t | 15 | yes | 1 | no (max_llcs) |
| `max-cpu` | 1s14l9c2t | 14 | yes | 1 | no (max_cpus) |
| `medium-4llc-nosmt` | 1s4l8c1t | 4 | no | 1 | no (no SMT) |
| `medium-8llc-nosmt` | 1s8l8c1t | 8 | no | 1 | no (no SMT) |
| `large-4llc-nosmt` | 1s4l32c1t | 4 | no | 1 | no (no SMT) |
| `large-8llc-nosmt` | 1s8l16c1t | 8 | no | 1 | no (no SMT) |
| `near-max-llc-nosmt` | 1s15l16c1t | 15 | no | 1 | no (no SMT) |
| `max-cpu-nosmt` | 1s14l18c1t | 14 | no | 1 | no (no SMT) |
| `numa2-4llc` | 2s4l4c1t | 4 | no | 2 | no (NUMA > 1) |
| `numa2-8llc` | 2s8l8c2t | 8 | yes | 2 | no (NUMA > 1) |
| `numa2-8llc-nosmt` | 2s8l16c1t | 8 | no | 2 | no (NUMA > 1) |
| `numa4-8llc` | 4s8l4c1t | 8 | no | 4 | no (NUMA > 1) |
| `numa4-12llc` | 4s12l8c2t | 12 | yes | 4 | no (NUMA > 1) |

Result: 6 of 24 presets survive (default `max_llcs = 12` and
`max_cpus = 192` exclude near-max and max-cpu presets; default
`max_numa_nodes = 1` excludes all NUMA presets). On aarch64, none
survive -- all aarch64 presets lack SMT.

## Controlling flag coverage

Flag constraints control which flag profiles are generated for each
test. The scheduler's `flags` field declares the available flags.
`Scheduler::generate_profiles()` produces the powerset of optional
flags, filtered by dependency constraints (`FlagDecl::requires`).

| Attribute | Effect |
|---|---|
| `required_flags = ["f"]` | `f` is present in every profile |
| `excluded_flags = ["g"]` | `g` is absent from every profile |

Flags not in either list are optional -- each profile either includes
or excludes them. Profiles where a flag's `requires` dependencies are
not satisfied are discarded.

### Example

Scheduler declares flags `llc`, `steal` (requires `llc`), `borrow`.
Test has `required_flags = ["llc"]`, `excluded_flags = ["borrow"]`.

Optional flags: `steal` (the only flag not required or excluded).
Generated profiles:

| Profile name | Active flags |
|---|---|
| `llc` | llc |
| `llc+steal` | llc, steal |

`steal` alone is invalid (requires `llc`), but `llc` is always
present via `required_flags`, so both profiles are valid. `borrow`
never appears.

## Total variant count

The total number of gauntlet variants for a test is:

```
valid_presets × valid_profiles
```

A test with 8 valid presets and 4 valid profiles produces 32 gauntlet
variants.

## Tests that skip gauntlet

- `host_only = true` tests never produce gauntlet variants (no VM to
  vary topology on).
- Tests whose names start with `demo_` are ignored by default. Their
  gauntlet variants are also ignored (all gauntlet variants are
  ignored).

## Cross-references

- [Gauntlet (Running Tests)](../running-tests/gauntlet.md) -- how to
  run gauntlet variants, preset table, budget interaction
- [Flags](../concepts/flags.md) -- flag declarations and profiles
- [The #\[ktstr_test\] Macro](ktstr-test-macro.md) -- full attribute
  reference
