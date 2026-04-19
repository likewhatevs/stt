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
                if e.chain().any(|cause| {
                    cause
                        .downcast_ref::<$crate::vmm::host_topology::ResourceContention>()
                        .is_some()
                }) =>
            {
                skip!("resource contention: {e:#}");
            }
            Err(e) => panic!("{e:#}"),
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::vmm::host_topology::ResourceContention;

    /// Regression for the error-chain fix: a ResourceContention wrapped
    /// in `.context(...)` must still be recognized by the macro and
    /// trigger the `skip!` branch instead of the `panic!` branch.
    #[test]
    fn skip_on_contention_walks_context_chain() {
        let result = std::panic::catch_unwind(|| {
            fn skip_fn() {
                let err: anyhow::Error = anyhow::Error::new(ResourceContention {
                    reason: "simulated contention".into(),
                })
                .context("wrapping context layer 1")
                .context("wrapping context layer 2");
                let _: () = skip_on_contention!(Err::<(), _>(err));
                unreachable!("skip_on_contention! should have early-returned");
            }
            skip_fn();
        });
        assert!(
            result.is_ok(),
            "context-wrapped ResourceContention must skip, not panic"
        );
    }

    /// Unwrapped ResourceContention keeps working (no regression on the
    /// simple path).
    #[test]
    fn skip_on_contention_recognizes_direct_error() {
        let result = std::panic::catch_unwind(|| {
            fn skip_fn() {
                let err: anyhow::Error = anyhow::Error::new(ResourceContention {
                    reason: "direct contention".into(),
                });
                let _: () = skip_on_contention!(Err::<(), _>(err));
                unreachable!("skip_on_contention! should have early-returned");
            }
            skip_fn();
        });
        assert!(
            result.is_ok(),
            "direct ResourceContention must skip, not panic"
        );
    }

    /// Non-contention errors still panic (negative case).
    #[test]
    #[should_panic(expected = "unrelated error")]
    fn skip_on_contention_panics_on_non_contention_error() {
        fn skip_fn() {
            let err = anyhow::anyhow!("unrelated error");
            let _: () = skip_on_contention!(Err::<(), _>(err));
        }
        skip_fn();
    }
}
