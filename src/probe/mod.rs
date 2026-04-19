// Probe exposes helpers the `#[ktstr_test]` dispatcher and the
// `cargo ktstr verifier` path reach for from outside the crate. The
// module is `pub(crate)` today so rustc flags each public helper as
// unused; lifting the allow here keeps the dead-code check at a
// narrower scope than the old blanket in lib.rs.
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
