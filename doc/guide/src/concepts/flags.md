# Flags

Flags represent scheduler capabilities that can be independently
enabled or disabled. The `#[derive(Scheduler)]` macro defines typed
flags from enum variants, generating compile-time checked constants
and `FlagDecl` statics.

## Defining flags

Annotate an enum with `#[derive(Scheduler)]`. Each variant becomes a
flag. `#[flag(args)]` lists CLI arguments passed to the scheduler when
the flag is active. `#[flag(requires)]` declares dependencies on other
variants.

```rust,ignore
use scx_ktstr::prelude::*;

#[derive(Scheduler)]
#[scheduler(name = "my_sched", binary = "scx_my_sched", topology(2, 4, 1))]
#[allow(dead_code)]
enum MySchedFlag {
    #[flag(args = ["--enable-llc-awareness"])]
    Llc,
    #[flag(args = ["--enable-borrowing"])]
    Borrow,
    #[flag(args = ["--enable-work-stealing"], requires = [Llc])]
    Steal,
}
```

The derive macro generates:

- A `static FlagDecl` for each variant with the flag's kebab-case name,
  CLI args, and dependency references.
- `impl MySchedFlag { pub const LLC: &'static str = "llc"; pub const BORROW: &'static str = "borrow"; ... }`
  -- typed string constants for each variant.
- `const MY_SCHED: Scheduler` -- a `Scheduler` const derived from the
  enum name (strip `Flag`/`Flags` suffix, convert to
  `SCREAMING_SNAKE_CASE`).

Variant names are converted to kebab-case: `Llc` becomes `"llc"`,
`RejectPin` becomes `"reject-pin"`.

## Using flags in tests

`#[ktstr_test]` accepts `required_flags` and `excluded_flags` to
constrain which flag profiles a test runs with. Both path expressions
and string literals work:

```rust,ignore
// Path expressions -- typos are compile errors
#[ktstr_test(
    scheduler = MY_SCHED,
    required_flags = [MySchedFlag::LLC],
    excluded_flags = [MySchedFlag::BORROW],
)]
fn needs_llc(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }

// String literals -- also work
#[ktstr_test(scheduler = MY_SCHED, required_flags = ["llc"])]
fn also_needs_llc(ctx: &Ctx) -> Result<AssertResult> { /* ... */ }
```

Path expressions are preferred: `MySchedFlag::LLC` is checked by the
compiler, while `"llc"` is not.

## Flag profiles and gauntlet

A `FlagProfile` is a sorted set of active flag names. Its display name
is the flags joined with `+` (e.g. `"llc+borrow"`), or `"default"`
when empty.

The gauntlet generates all valid flag combinations from the scheduler's
flag declarations and the test's constraints. For a scheduler with 6
flags and one dependency (`steal` requires `llc`), unconstrained
generation produces 48 profiles (2^6 = 64 minus 16 combinations where
`steal` is active without `llc`).

`required_flags` forces flags into every profile. `excluded_flags`
removes flags from consideration. The remaining flags are combined in
all valid subsets.

See [Gauntlet flag profiles](../running-tests/gauntlet.md#flag-profiles).

## Dependencies

`requires = [Llc]` on a variant means that variant is only active in
profiles where `Llc` is also active. Profile generation rejects any
combination where a flag's dependencies are missing.

```rust,ignore
#[flag(args = ["--enable-work-stealing"], requires = [Llc])]
Steal,
```

Every generated profile containing `steal` also contains `llc`.
Requiring `steal` in a test (`required_flags = [MySchedFlag::STEAL]`)
implicitly forces `llc` into all profiles for that test.

## Underlying mechanism (advanced)

The derive macro generates `FlagDecl` statics and a flags array that
the `Scheduler` const references. Most users never need to write
`FlagDecl` manually -- the macro handles it. The generated code is
equivalent to:

```rust,ignore
static __MY_SCHED_FLAG_DECL_LLC: FlagDecl = FlagDecl {
    name: "llc",
    args: &["--enable-llc-awareness"],
    requires: &[],
};

static __MY_SCHED_FLAG_DECL_STEAL: FlagDecl = FlagDecl {
    name: "steal",
    args: &["--enable-work-stealing"],
    requires: &[&__MY_SCHED_FLAG_DECL_LLC],
};
```

See [`scx-ktstr-macros/src/lib.rs`](https://github.com/likewhatevs/scx-ktstr/blob/main/scx-ktstr-macros/src/lib.rs)
for the macro source.
