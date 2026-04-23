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
//! module). A test-only env override —
//! [`WORKER_READY_MARKER_OVERRIDE_ENV`] — replaces the default
//! pid-scoped path when set and non-empty; production callers
//! never set it.
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
//! - **No `crate::…` paths**, no `use crate::…` statements.
//!   `crate` resolves to two different crates (ktstr vs.
//!   the bin) on the two compilation paths; anything that names the
//!   other crate's types breaks one of the two builds. A future
//!   nested `#[cfg(test)] mod tests { … }` block is safe to add at
//!   the bottom of this file using `super::` to reach items defined
//!   here — `super::` from a child `mod tests` resolves to this
//!   file's own items, which exist identically under both
//!   compilation paths, so the tests would not divide lib vs. bin.
//!   None currently exists; the pin-tests for the path format live
//!   in `tests/jemalloc_alloc_worker_exit_codes.rs` and
//!   `tests/jemalloc_probe_signals_test.rs`, which reach the items
//!   through the `ktstr::worker_ready::…` public surface.
//! - **No ktstr-library types or modules.** Only `std` items,
//!   language primitives, and `core` types are safe. Anything that
//!   depends on `PayloadHandle`, scenario `Ctx`, `anyhow`, or any
//!   other lib-only item must live in
//!   [`crate::worker_ready_wait`] (lib-only) — not here.
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
/// Exported as a `pub const` for symmetry with the other items in
/// this module ([`WORKER_READY_MARKER_OVERRIDE_ENV`],
/// [`worker_ready_marker_path`], [`WORKER_STDERR_PREFIX`]) — every
/// symbol that represents a piece of the worker-ready / worker-
/// stderr wire contract is `pub` so host-side integration tests in
/// `tests/` can assert on the exact literal the worker and the
/// probe depend on, without the test file having to duplicate the
/// string. Downstream callers (the worker binary's ready-path
/// write, the host-side poll in `wait_for_worker_ready`) normally
/// route through [`worker_ready_marker_path`] rather than
/// concatenating this prefix themselves; the prefix is exposed for
/// assertion use, not as the preferred construction path.
pub const WORKER_READY_MARKER_PREFIX: &str = "/tmp/ktstr-worker-ready-";

/// Name of the test-only env var that overrides the pid-scoped
/// default path. When set and non-empty, the worker writes the
/// ready marker at the override path instead of
/// [`worker_ready_marker_path(pid)`](worker_ready_marker_path);
/// when unset (or empty) the default pid-scoped path applies.
///
/// Exported as a `pub const` so both the worker binary and the
/// integration tests that drive it share a single source of truth
/// — eliminating the string-literal drift window where the worker
/// and a test disagree on the env-var name and the override
/// silently fails to take effect.
pub const WORKER_READY_MARKER_OVERRIDE_ENV: &str = "KTSTR_WORKER_READY_MARKER_OVERRIDE";

/// Construct the ready-marker path for a worker with the given pid.
///
/// The worker uses [`std::process::id()`] (`u32`) as the pid source
/// inside its own process. The host-side test reads the worker's
/// pid via `PayloadHandle::pid()`, which returns `Option<u32>` —
/// `Some(pid)` once the child has been spawned and `None` before
/// (or after a kill that tore the child down). Callers must
/// unwrap (or propagate) the `Option` at the call site; this
/// helper takes a bare `u32` so the unwrap decision stays visible
/// to the caller instead of being swallowed inside the formatter.
/// The `u32` parameter matches both the worker's
/// `std::process::id()` return type and the unwrapped payload of
/// `PayloadHandle::pid()` without per-caller casts.
pub fn worker_ready_marker_path(pid: u32) -> String {
    format!("{WORKER_READY_MARKER_PREFIX}{pid}")
}

/// Stderr line prefix the `ktstr-jemalloc-alloc-worker` binary
/// prepends to every fail-fast diagnostic it emits (missing argv,
/// bytes=0, thread self-check, procfs unreadable, ready-marker
/// write fail). Exported as a `pub const` so host-side integration
/// tests can assert against the binary's emitted literal without
/// duplicating it — a rename or a typo on either side shows up
/// here in one place instead of silently desynchronizing.
///
/// The trailing space that separates the prefix from the message
/// body is NOT part of this constant: call sites write
/// `{WORKER_STDERR_PREFIX} …` so the space remains a formatting
/// concern, and test-side substring checks can match on the
/// prefix alone regardless of the specific separator the worker
/// chooses.
pub const WORKER_STDERR_PREFIX: &str = "jemalloc-alloc-worker:";

/// Stdout "ready" breadcrumb the `ktstr-jemalloc-alloc-worker`
/// binary prints once, immediately before parking on `pause()`,
/// after its allocation + `black_box` triple has materialised.
/// The full emitted line is
/// `{WORKER_STDOUT_READY_PREFIX} pid={pid} bytes={bytes}`; this
/// const carries only the prefix so host-side consumers that want
/// to grep the worker's captured stdout (or that fold worker
/// stdout into a larger test log) can match against a single
/// authoritative literal.
///
/// Exported as `pub const` for the same reason as
/// [`WORKER_STDERR_PREFIX`]: a rename or a typo on either side
/// shows up in one place instead of silently desynchronising the
/// worker and any test-side assertion. The breadcrumb is
/// currently not parsed by any automated consumer — readiness is
/// signalled via the marker file ([`worker_ready_marker_path`]),
/// NOT via stdout — but pinning the literal here means a future
/// test that wants to correlate "worker log says ready" with
/// "marker file appeared" has a stable hook.
///
/// The trailing space separating the prefix from the
/// `pid=` / `bytes=` tail is NOT part of the constant, matching
/// the [`WORKER_STDERR_PREFIX`] convention.
pub const WORKER_STDOUT_READY_PREFIX: &str = "jemalloc-alloc-worker ready";

