# Gauntlet Tests

The gauntlet expands each `#[ktstr_test]` into a matrix of
test × topology_preset variants. The test definition controls
which cells of the matrix it populates.

## Controlling topology coverage

Topology constraints in `#[ktstr_test]` filter which gauntlet
presets a test runs on. See
[Topology Constraints](ktstr-test-macro.md#topology-constraints)
for the full attribute table and
[Topology Presets](../running-tests/gauntlet.md#topology-presets)
for the preset list.

### Worked example

A test with `min_llcs = 2`, `requires_smt = true`, and default
`max_numa_nodes = 1` against the
[preset table](../running-tests/gauntlet.md#topology-presets):

- `tiny-1llc` (1 LLC): excluded — below `min_llcs`
- All non-SMT presets (`tiny-2llc`, `odd-*`, `*-nosmt`):
  excluded — `requires_smt`
- `near-max-llc` (15 LLCs): excluded — above default
  `max_llcs = 12`
- `max-cpu` (252 CPUs, 14 LLCs): excluded — above default
  `max_cpus = 192` (also above default `max_llcs = 12`)
- All `numa*` presets: excluded — above default
  `max_numa_nodes = 1`

Result: 6 of 24 presets survive (`smt-2llc`, `smt-3llc`,
`medium-4llc`, `medium-8llc`, `large-4llc`, `large-8llc`). On
aarch64, none survive — all aarch64 presets lack SMT.

## Total variant count

The total number of gauntlet variants for a test is:

```text
valid_presets × resolved_kernels
```

A test with 8 valid presets produces 8 gauntlet variants under
a single-kernel run; passing two kernels (`--kernel A --kernel B`)
doubles that to 16. The kernel dimension is contributed by
`cargo ktstr test` / `coverage` / `llvm-cov` at the CLI surface
(zero or one resolved kernels keeps the historical 3-segment
name shape `gauntlet/{name}/{preset}`; two or more expands the
gauntlet across kernels with an extra `{kernel_label}` segment).
See
[Multi-kernel: kernel as a gauntlet dimension](../running-tests/cargo-ktstr.md#multi-kernel-kernel-as-a-gauntlet-dimension).

## Tests that skip gauntlet

- Entries with `host_only = true` never produce gauntlet
  variants (no VM to vary topology on). They also skip the
  kernel-dim multiplication under multi-kernel runs: a
  `host_only` test lists and runs **once** regardless of
  `KTSTR_KERNEL_LIST` cardinality, since a host-side test
  never observes the kernel directory and N copies of identical
  work would carry no signal. See
  [`host_only`](ktstr-test-macro.md#execution) for how that flag
  is set, and
  [Multi-kernel: kernel as a gauntlet dimension](../running-tests/cargo-ktstr.md#multi-kernel-kernel-as-a-gauntlet-dimension)
  for the kernel-suffix dispatch contract.
- Tests whose names start with `demo_` are ignored by default.
  Their gauntlet variants are also ignored (all gauntlet
  variants are ignored).

## Cross-references

- [Gauntlet (Running Tests)](../running-tests/gauntlet.md) —
  how to run gauntlet variants, preset table, budget
  interaction
- [The #\[ktstr_test\] Macro](ktstr-test-macro.md) — full
  attribute reference
