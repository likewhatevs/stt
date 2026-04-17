# Gauntlet Tests

The gauntlet expands each `#[ktstr_test]` into a matrix of
test x topology_preset x flag_profile variants. The test definition
controls which cells of the matrix it populates.

## Controlling topology coverage

Topology constraints in `#[ktstr_test]` filter which gauntlet presets
a test runs on. See
[Topology Constraints](ktstr-test-macro.md#topology-constraints)
for the full attribute table and
[Topology Presets](../running-tests/gauntlet.md#topology-presets) for
the preset list.

### Worked example

A test with `min_llcs = 2`, `requires_smt = true`, and default
`max_numa_nodes = 1` against the
[preset table](../running-tests/gauntlet.md#topology-presets):

- `tiny-1llc` (1 LLC): excluded — below `min_llcs`
- All non-SMT presets (`tiny-2llc`, `odd-*`, `*-nosmt`): excluded — `requires_smt`
- `near-max-llc` (15 LLCs): excluded — above default `max_llcs = 12`
- `max-cpu` (252 CPUs, 14 LLCs): excluded — above default `max_cpus = 192` (also above default `max_llcs = 12`)
- All `numa*` presets: excluded — above default `max_numa_nodes = 1`

Result: 6 of 24 presets survive (`smt-2llc`, `smt-3llc`, `medium-4llc`,
`medium-8llc`, `large-4llc`, `large-8llc`). On aarch64, none survive
— all aarch64 presets lack SMT.

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

- Entries with `host_only = true` never produce gauntlet variants
  (no VM to vary topology on). See
  [`host_only`](ktstr-test-macro.md#execution) for how that flag is
  set.
- Tests whose names start with `demo_` are ignored by default. Their
  gauntlet variants are also ignored (all gauntlet variants are
  ignored).

## Cross-references

- [Gauntlet (Running Tests)](../running-tests/gauntlet.md) -- how to
  run gauntlet variants, preset table, budget interaction
- [Flags](../concepts/flags.md) -- flag declarations and profiles
- [The #\[ktstr_test\] Macro](ktstr-test-macro.md) -- full attribute
  reference
