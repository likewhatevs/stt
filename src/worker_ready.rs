//! Shared ready-marker path format for the
//! `ktstr-jemalloc-alloc-worker` binary and the integration tests
//! that drive it.
//!
//! The worker writes a pid-scoped file after its allocation + black-box
//! triple completes; the test body polls for that file before launching
//! the probe. Centralizing the path format here keeps the worker
//! (`src/bin/jemalloc_alloc_worker.rs`) and the test
//! (`tests/jemalloc_probe_tests.rs`) in sync — a rename changes one
//! place, not two.
//!
//! This file is also included directly into the worker bin crate via
//! `#[path]` (see the bin's source) to avoid linking the entire ktstr
//! library into the worker binary. Consequently, this module must not
//! reference any other ktstr types or modules. The `wait_for_worker_ready`
//! helper lives in the sibling [`crate::worker_ready_wait`] module
//! because it needs `PayloadHandle` and therefore depends on the rest
//! of the library.
//!
//! # Host ↔ guest assumptions
//!
//! The ready-marker scheme relies on two properties that the ktstr
//! harness supplies:
//!
//! - **Shared `/tmp` between worker and reader.** The `#[ktstr_test]`
//!   integration-test flow reads the marker from the host-side
//!   `PayloadHandle::pid()` after the worker writes the file inside
//!   the guest VM — same logical `/tmp` because the harness bind-
//!   mounts the host's `/tmp` into the guest at boot (see
//!   `build_vm_builder_base`). The marker is NOT transported over a
//!   socket / pipe: if the harness ever stops sharing `/tmp`, the
//!   poll will always time out and the ready-signal must move to a
//!   different medium (unix socket, `vsock`, or a stdout-token parse).
//! - **`PayloadHandle::pid() == std::process::id()` inside the guest.**
//!   Host-side `PayloadHandle::pid()` returns the pid the harness
//!   allocated for the payload process, which is the same pid the
//!   worker observes via `getpid()` / `std::process::id()` inside
//!   the guest — single-namespace because the worker runs without
//!   a separate pid-namespace. Consumers that add a pid-namespace
//!   or run the worker under something like `unshare --fork --pid`
//!   must also translate the pid before constructing the path, or
//!   the reader polls a path that the writer never materialized.
//!
//! Both properties are invariants the `ktstr-jemalloc-alloc-worker`
//! + `jemalloc_probe_tests.rs` pair depends on; breaking either
//! without updating this module's scheme produces silent poll
//! timeouts rather than loud errors.

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the path format. If this ever changes the worker binary
    /// and test body must stay in sync, so a rename here fails every
    /// downstream caller's path-literal expectations at build time.
    #[test]
    fn worker_ready_marker_path_format_is_stable() {
        assert_eq!(WORKER_READY_MARKER_PREFIX, "/tmp/ktstr-worker-ready-");
        assert_eq!(worker_ready_marker_path(0), "/tmp/ktstr-worker-ready-0");
        assert_eq!(worker_ready_marker_path(12345), "/tmp/ktstr-worker-ready-12345");
        assert_eq!(
            worker_ready_marker_path(u32::MAX),
            "/tmp/ktstr-worker-ready-4294967295"
        );
    }
}
