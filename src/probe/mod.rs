// `probe` is `pub(crate)` in lib.rs, so the whole subtree is crate-
// private. Submodules `btf`, `output`, `process`, and `stack` are
// locally `pub` so siblings in the crate can reach their items
// without the verbosity of `pub(crate)`; `decode` and `scx_defs`
// stay `pub(crate)` because they expose helpers only the rest of
// the probe module consumes. Many helpers are only exercised by the
// macro-generated `#[ktstr_test]` dispatcher and the verifier
// binary, so rustc flags them as unused; the allow below keeps the
// dead-code check at a narrower scope than the old blanket in lib.rs.
#![allow(dead_code)]

//! Crash investigation via BPF kprobes, fentry/fexit, and tracepoints.
//!
//! Attaches kprobes and fentry probes to kernel and BPF functions from
//! a crash stack trace, triggers on `sched_ext_exit` via tp_btf,
//! captures argument state, and formats annotated output with source
//! locations.
//!
//! See the [Investigate a Crash](https://likewhatevs.github.io/ktstr/guide/recipes/investigate-crash.html)
//! recipe.

pub mod btf;
pub(crate) mod decode;
pub mod output;
pub mod process;
pub(crate) mod scx_defs;
pub mod stack;
