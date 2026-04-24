//! Test-only macros shared across the crate.
//!
//! Hoisted to crate root via `#[macro_use]` on the module declaration
//! in `lib.rs`, so `skip!` and `skip_on_contention!` are reachable from
//! any `#[cfg(test)]` code without an explicit `use`.

/// Emit a canonical `ktstr: SKIP: ...` message and return from the
/// caller. Routes through [`crate::report::test_skip`] so the
/// prefix lives in one place — the alternative (15+ open-coded
/// `eprintln!` sites) drifts into inconsistent casings that break
/// every grep-based test-summary tool.
///
/// Only callable from functions returning `()` — the macro expands to
/// an early `return;` with no value. Production code that returns a
/// non-unit type (dispatcher fns returning `i32`, helpers returning
/// `Option<T>`, loop bodies that `continue`) calls
/// [`crate::report::test_skip`] directly and drives its own control
/// flow.
macro_rules! skip {
    // Zero-args arm: `skip!()` emits the banner with an empty
    // reason. `format_args!()` itself requires at least a format
    // string, so the variadic arm below cannot handle this case
    // — a dedicated rule routes it to an empty literal.
    () => {{
        $crate::report::test_skip(format_args!(""));
        return;
    }};
    ($($arg:tt)*) => {{
        $crate::report::test_skip(format_args!($($arg)*));
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
    ///
    /// `#[cfg(panic = "unwind")]`: this test uses `std::panic::catch_unwind`
    /// to assert the macro does NOT panic. Under `panic = "abort"` (the
    /// release profile's setting — see `Cargo.toml [profile.release]`)
    /// panics cannot be caught; the panic aborts the whole test binary
    /// instead of returning an `Err` from `catch_unwind`. Gating the
    /// test on the panic strategy lets `cargo ktstr test --release`
    /// skip it without false-failing the binary.
    #[test]
    #[cfg(panic = "unwind")]
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
    ///
    /// `#[cfg(panic = "unwind")]`: same rationale as the sibling
    /// context-chain test — `catch_unwind` is unusable under
    /// `panic = "abort"`.
    #[test]
    #[cfg(panic = "unwind")]
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

    /// The `skip!` macro must emit the canonical `ktstr: SKIP:
    /// <reason>` banner to stderr AND early-return from the calling
    /// function. Prior tests exercise `test_skip` (the lower-level
    /// emitter) and `skip_on_contention!` (the wrapper macro) but
    /// the bare `skip!` macro was left uncovered — a regression that
    /// silently broke the format_args expansion or the `return;`
    /// tail would slip through until a downstream consumer
    /// parsed the wrong line.
    ///
    /// This test uses the crate-shared stderr-capture helper and
    /// verifies BOTH invariants: the captured bytes carry the
    /// canonical banner, and a post-`skip!` line in the helper fn
    /// is never reached (pinned via a sentinel flag).
    #[test]
    fn skip_macro_emits_banner_and_early_returns() {
        use crate::test_support::test_helpers::capture_stderr;
        use std::sync::atomic::{AtomicBool, Ordering};

        let reached_tail = AtomicBool::new(false);
        let (_, bytes) = capture_stderr(|| {
            // Helper fn returning `()` so `skip!` can emit its
            // `return;` tail. The AtomicBool is set only if the
            // line AFTER `skip!` executes — a regression that
            // dropped the `return;` tail would trip it. The two
            // `#[allow(...)]` attributes are load-bearing: when
            // `skip!` correctly returns, `reached.store` is dead
            // code AND `reached` falls out of the live set —
            // which is exactly what this test is designed to
            // pin. Without the allows, compilation warns about
            // the very invariant the test verifies.
            #[allow(unused_variables, unreachable_code)]
            fn helper(reached: &AtomicBool) {
                skip!("macro-level reason with {} substitution", "format-args");
                reached.store(true, Ordering::SeqCst);
            }
            helper(&reached_tail);
        });
        let text = std::str::from_utf8(&bytes).expect("stderr is UTF-8");
        assert_eq!(
            text, "ktstr: SKIP: macro-level reason with format-args substitution\n",
            "expected canonical banner with format-args substitution",
        );
        assert!(
            !reached_tail.load(Ordering::SeqCst),
            "skip! must early-return; lines after the macro must not execute",
        );
    }

    /// `skip!` with a literal (no format args) still emits the
    /// banner. Pairs with the substitution test above to cover the
    /// no-args branch of the `format_args!($($arg)*)` expansion.
    #[test]
    fn skip_macro_literal_reason_emits_banner() {
        use crate::test_support::test_helpers::capture_stderr;
        let (_, bytes) = capture_stderr(|| {
            fn helper() {
                skip!("literal skip reason");
            }
            helper();
        });
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text, "ktstr: SKIP: literal skip reason\n");
    }

    /// `skip!()` with ZERO arguments expands to
    /// `format_args!()` — an empty reason. The banner still fires
    /// with the canonical prefix + colon + empty tail + newline.
    /// Pins the degenerate-input behavior so a regression that
    /// rejected zero-argument expansion (e.g. a macro arm
    /// requiring at least one token tree) fails here instead of at
    /// some downstream call site that happens to call `skip!()`
    /// for "I don't care why, just skip" semantics.
    #[test]
    fn skip_macro_zero_args_emits_banner_with_empty_reason() {
        use crate::test_support::test_helpers::capture_stderr;
        let (_, bytes) = capture_stderr(|| {
            fn helper() {
                skip!();
            }
            helper();
        });
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text, "ktstr: SKIP: \n");
    }
}
