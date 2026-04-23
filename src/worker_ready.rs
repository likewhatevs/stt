//! Shared ready-marker path for the `ktstr-jemalloc-alloc-worker`
//! binary and the integration tests that drive it.
//!
//! The worker writes a pid-scoped file after its allocation + black-box
//! triple completes; the test body polls for that file before launching
//! the probe. Centralizing the path format here keeps the worker
//! (`src/bin/jemalloc_alloc_worker.rs`) and the test
//! (`tests/jemalloc_probe_tests.rs`) in sync — a rename changes one
//! place, not two.

/// Prefix for the pid-scoped ready-marker path. The final segment is
/// the worker's pid rendered as a decimal ASCII integer.
pub const WORKER_READY_MARKER_PREFIX: &str = "/tmp/ktstr-worker-ready-";

/// Construct the ready-marker path for a worker with the given pid.
///
/// The worker uses [`std::process::id()`] (`u32`) as the pid source;
/// the test reads the pid via `PayloadHandle::pid()` which also returns
/// `u32`. The shared `u32` parameter matches both call sites without
/// per-caller casts.
pub fn worker_ready_marker_path(pid: u32) -> String {
    format!("{WORKER_READY_MARKER_PREFIX}{pid}")
}
