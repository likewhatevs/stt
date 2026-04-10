# Gauntlet Tests

The gauntlet expands each `#[ktstr_test]` into a matrix of
test x topology_preset x flag_profile variants. The test definition
controls which cells of the matrix it populates.

## Controlling topology coverage

Topology constraints filter which of the 13 gauntlet presets a test
runs on. Set constraints in `#[ktstr_test]` attributes:

| Attribute | Type | Default | Effect |
|---|---|---|---|
| `min_sockets` | `u32` | 1 | Skip presets with fewer sockets |
| `min_llcs` | `u32` | 1 | Skip presets with fewer LLCs |
| `min_cpus` | `u32` | 1 | Skip presets with fewer total CPUs |
| `requires_smt` | `bool` | false | Skip presets with `threads_per_core < 2` |

A preset is included only when all four constraints are satisfied.

### Worked example

A test with `min_llcs = 2` and `requires_smt = true`:

| Preset | Topology | LLCs | SMT | Included |
|---|---|---|---|---|
| `tiny-1llc` | 1s4c1t | 1 | no | no (1 LLC) |
| `tiny-2llc` | 2s2c1t | 2 | no | no (no SMT) |
| `odd-3llc` | 3s3c1t | 3 | no | no (no SMT) |
| `odd-5llc` | 5s3c1t | 5 | no | no (no SMT) |
| `odd-7llc` | 7s2c1t | 7 | no | no (no SMT) |
| `smt-2llc` | 2s2c2t | 2 | yes | **yes** |
| `smt-3llc` | 3s2c2t | 3 | yes | **yes** |
| `medium-4llc` | 4s4c2t | 4 | yes | **yes** |
| `medium-8llc` | 8s4c2t | 8 | yes | **yes** |
| `large-4llc` | 4s16c2t | 4 | yes | **yes** |
| `large-8llc` | 8s8c2t | 8 | yes | **yes** |
| `near-max-llc` | 15s8c2t | 15 | yes | **yes** |
| `max-cpu` | 14s9c2t | 14 | yes | **yes** |

Result: 8 of 13 presets survive.

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
