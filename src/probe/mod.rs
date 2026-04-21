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
