//! Test-only macros shared across the crate.
//!
//! Hoisted to crate root via `#[macro_use]` on the module declaration
//! in `lib.rs`, so `skip!` and `skip_on_contention!` are reachable from
//! any `#[cfg(test)]` code without an explicit `use`.

/// Emit a canonical `ktstr: SKIP: ...` message and return from the
/// caller. Centralizes the prefix so every test reports skips in a
/// single grep-able format.
///
/// Only callable from functions returning `()` — the macro expands to
/// an early `return;` with no value. Production code that returns a
/// non-unit type (dispatcher fns returning `i32`, helpers returning
/// `Option<T>`) must emit the `ktstr: SKIP:` line manually and return
/// its own sentinel.
macro_rules! skip {
    ($($arg:tt)*) => {{
        eprintln!("ktstr: SKIP: {}", format_args!($($arg)*));
        return;
    }};
}

/// Evaluate a `Result`-returning builder (or any `anyhow::Result`
/// expression) and either unwrap the value or skip gracefully on
/// [`crate::vmm::host_topology::ResourceContention`]. Any other error
/// panics with `{e:#}`.
///
/// Replaces the recurring `match ... { Ok => v, Err(e) if
/// ResourceContention => skip!(...), Err(e) => panic!(...) }`
/// boilerplate. Inherits `skip!`'s early-return behavior, so callers
/// must return `()`.
macro_rules! skip_on_contention {
    ($expr:expr) => {
        match $expr {
            Ok(v) => v,
            Err(e)
                if e.downcast_ref::<$crate::vmm::host_topology::ResourceContention>()
                    .is_some() =>
            {
                skip!("resource contention: {e}");
            }
            Err(e) => panic!("{e:#}"),
        }
    };
}
