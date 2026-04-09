# Flags

Flags represent scheduler capabilities. Each flag is a typed static
declaration with dependency constraints.

## Flag declarations

```rust,ignore
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

```rust,ignore
pub static STEAL_DECL: FlagDecl = FlagDecl {
    name: "steal",
    args: &[],
    requires: &[&LLC_DECL],
};
```

When generating flag profiles, any combination that includes `steal`
without `llc` is rejected.

## Using flags

In `#[stt_test]`, use `required_flags` and `excluded_flags` to
constrain which flag profiles the test runs with:

```rust,ignore
#[stt_test(required_flags = ["llc", "borrow"])]
fn my_test(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

The gauntlet expands each test across all valid flag profiles.

## Flag profiles

A `FlagProfile` is a sorted set of active flags:

```rust,ignore
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

For gauntlet flag expansion, see [Gauntlet](../running-tests/gauntlet.md#flag-profiles).
