//! Shared ready-marker path format for the
//! `ktstr-jemalloc-alloc-worker` binary and the integration tests
//! that drive it.
//!
//! # Worker → probe ready signaling mechanism (design)
//!
//! The worker writes a pid-scoped file after its allocation +
//! black-box triple completes; the test body polls for that file
//! before launching the probe. Centralizing the path format here
//! keeps the worker (`src/bin/jemalloc_alloc_worker.rs`) and the
//! test (`tests/jemalloc_probe_tests.rs`) in sync — a rename
//! changes one place, not two.
//!
//! Medium: **a file on the shared `/tmp`**. The path is
//! `/tmp/ktstr-worker-ready-<pid>` where `<pid>` is the worker's
//! decimal pid (see [`worker_ready_marker_path`] / [`WORKER_READY_MARKER_PREFIX`]).
//! The worker issues `std::fs::write(path, b"ready\n")` on
//! ready; the test polls `Path::exists` in a bounded loop (see
//! `wait_for_worker_ready` in the sibling `worker_ready_wait`
//! module).
//!
//! Why a file-on-tmp rather than a pipe / unix socket / vsock /
//! stdout-token?
//!
//! - **Minimal setup before the worker's own allocation path.**
//!   The probe must observe the worker's post-allocation heap
//!   state, not any pre-signaling setup cost. `std::fs::write`
//!   against an existing tmp directory is three syscalls
//!   (`openat` + `write` + `close`) on already-hot kernel
//!   caches; a socket would add `socket` + `connect` +
//!   `sendto` against a daemon that would itself need to be
//!   provisioned by the harness.
//! - **Shared filesystem namespace without dedicated plumbing.**
//!   Both the worker and the test body run as subprocesses
//!   inside the SAME `#[ktstr_test]` guest VM.
//!   `PayloadRun::spawn` creates the worker via
//!   `std::process::Command`, which inherits the parent's
//!   (guest-side) filesystem namespace, so the two processes
//!   see the same guest-VM tmpfs `/tmp`. No host involvement,
//!   no bind-mount, no guest↔host bridge. A socket path would
//!   still require a dedicated in-VM dispatcher; vsock would
//!   require a cid allocation.
//! - **No process-of-write hard dependency.** A pipe close or
//!   EOF on the worker's stdout would also signal readiness,
//!   but any crashing worker would look the same — the file
//!   approach surfaces "worker reached the signaling point"
//!   distinctly from "worker died before signaling".
//!
//! # Dual-compilation constraint — MUST STAY STD-ONLY
//!
//! This source file is compiled TWICE by the same `cargo build`:
//! once as `ktstr::worker_ready` (a lib-crate module) and once as the
//! worker bin's own `mod worker_ready` via
//! `#[path = "../worker_ready.rs"]` (see
//! `src/bin/jemalloc_alloc_worker.rs`). The `#[path]` include is
//! deliberate: linking the entire ktstr library into a worker
//! process would pull thousands of unused symbols and perturb the
//! probe's cross-process timing.
//!
//! Consequences for this file:
//! - **No `crate::…` or `super::…` paths**, no `use crate::…`
//!   statements. `crate` resolves to two different crates (ktstr vs.
//!   the bin) on the two compilation paths; anything that names the
//!   other crate's types breaks one of the two builds.
//! - **No ktstr-library types or modules.** Only `std` items,
//!   language primitives, and `core` types are safe. Anything that
//!   depends on `PayloadHandle`, scenario `Ctx`, `anyhow`, or any
//!   other lib-only item must live in
//!   [`crate::worker_ready_wait`](../worker_ready_wait/index.html)
//!   (lib-only) — not here.
//! - **No external crate imports that only the lib or only the bin
//!   has.** Adding a non-std dependency requires a matching `Cargo.toml`
//!   stanza for both the lib and the bin; otherwise one build path
//!   fails to resolve the crate.
//! - **No `#[cfg(feature = "…")]` that differs across the two
//!   crates.** Feature gates evaluate per-crate, so a gate that's
//!   satisfied for the lib but not the bin (or vice versa) will
//!   silently diverge the two compiled copies.
//!
//! The `wait_for_worker_ready` helper lives in the sibling
//! [`crate::worker_ready_wait`] module because it needs
//! `PayloadHandle` and therefore depends on the rest of the
//! library.
//!
//! # In-VM invariants
//!
//! The ready-marker scheme relies on two properties that the
//! `#[ktstr_test]` VM environment supplies:
//!
//! - **Shared `/tmp` between worker and reader.** Both the
//!   worker and the test body run as subprocesses inside the
//!   same guest VM, spawned via `PayloadRun::spawn` →
//!   `std::process::Command`. The child inherits the parent's
//!   guest-side filesystem namespace, so both processes see
//!   the same guest-VM tmpfs `/tmp`. The marker is NOT
//!   transported over a socket / pipe: if a future refactor
//!   puts the worker in a distinct filesystem namespace
//!   (e.g. via `unshare --mount` or a separate VM), the poll
//!   will always time out and the ready-signal must move to a
//!   different medium (unix socket, `vsock`, or a stdout-token
//!   parse).
//! - **`PayloadHandle::pid() == std::process::id()` inside the guest.**
//!   The test body reads `PayloadHandle::pid()` to learn the
//!   worker's pid, which is the same pid the worker observes
//!   via `getpid()` / `std::process::id()` inside the VM —
//!   single-namespace because the worker runs without a
//!   separate pid-namespace. Consumers that add a pid-namespace
//!   or run the worker under something like `unshare --fork
//!   --pid` must also translate the pid before constructing
//!   the path, or the reader polls a path that the writer
//!   never materialized.
//!
//! Both properties are invariants the `ktstr-jemalloc-alloc-worker`
//! + `jemalloc_probe_tests.rs` pair depends on; breaking either
//! without updating this module's scheme produces silent poll
//! timeouts rather than loud errors.

/// Prefix for the pid-scoped ready-marker path. The final segment is
/// the worker's pid rendered as a decimal ASCII integer.
///
/// Visibility: `pub(crate)` — the prefix is an internal
/// implementation detail of the worker-ready-marker convention.
/// [`worker_ready_marker_path`] is the stable surface; hiding the
/// prefix prevents external consumers from inlining the literal and
/// silently drifting on a rename.
pub(crate) const WORKER_READY_MARKER_PREFIX: &str = "/tmp/ktstr-worker-ready-";

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
