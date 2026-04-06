# Flags

Flags represent scheduler capabilities. Each flag is a typed static
declaration with dependency constraints.

## Flag declarations

```rust
pub struct FlagDecl {
    pub name: &'static str,
    pub args: &'static [&'static str],
    pub requires: &'static [&'static FlagDecl],
}
```

Six flags are defined:

| Flag | Requires | Description |
|---|---|---|
| `llc` | -- | LLC-aware scheduling |
| `borrow` | -- | CPU borrowing across domains |
| `steal` | `llc` | Work stealing (requires LLC awareness) |
| `rebal` | -- | Rebalancing |
| `reject-pin` | -- | Reject pinned task overrides |
| `no-ctrl` | -- | Disable controller |

## Dependency enforcement

`steal` requires `llc`. This is encoded in the `FlagDecl`:

```rust
pub static STEAL_DECL: FlagDecl = FlagDecl {
    name: "steal",
    args: &[],
    requires: &[&LLC_DECL],
};
```

When generating flag profiles, any combination that includes `steal`
without `llc` is rejected.

## Using flags

From the CLI, pass `--flags=borrow,rebal`:

```sh
cargo stt vm --sockets 2 --cores 4 --threads 2 \
  -- cgroup_steady --flags=borrow,rebal
```

`--all-flags` runs every valid flag combination for the selected
scenarios.

## Flag profiles

A `FlagProfile` is a sorted set of active flags:

```rust
pub struct FlagProfile {
    pub flags: Vec<&'static str>,
}
```

The profile's display name is the flags joined with `+`:
- Empty profile: `"default"`
- `[llc, borrow]`: `"llc+borrow"`

## Profile generation

`generate_profiles(required, excluded)` enumerates all valid
combinations:

- Start with all flags not in `required` or `excluded`.
- For each subset, add `required` flags.
- Filter out combinations where a flag's `requires` dependencies are
  missing.
- Sort flags in canonical order.

Unconstrained: 48 profiles. With `steal` required: all profiles
include both `steal` and `llc`.

For CLI usage, see [Running Tests -- Flags](../running-tests.md#flags).
