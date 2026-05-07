//! Lightweight state types and constants shared across the freeze
//! coordinator's call sites.
//!
//! Pure types and constants only — no behaviour. Behaviour lives in
//! the modules that consume these (the run-loop closure in
//! [`super`], the snapshot request handlers in [`super::snapshot`]).
//! Splitting the types out here lets the closure body shrink and
//! lets each consumer use the same vocabulary without re-deriving it
//! locally.
//!
//! Three groups live here:
//!
//! * [`FREEZE_RENDEZVOUS_TIMEOUT`] — wall-clock budget for parked-
//!   vCPU rendezvous and the matching post-thaw barrier.
//! * [`BspExitReason`] — diagnostic enum logged when the BSP run
//!   loop breaks.
//! * [`SnapshotRequest`] — typed view of a guest-side
//!   `MSG_TYPE_SNAPSHOT_REQUEST` TLV.
//! * [`FreezeState`] — the dump state machine the run-loop closure
//!   advances on each freeze cycle.
//!
//! All four were previously defined inline at the top of
//! `freeze_coord.rs` (or, for `FreezeState`, inside the run-loop
//! closure body); the public surface is unchanged.

use std::time::Duration;

/// Maximum wall-clock duration the freeze coordinator will wait for
/// every vCPU to acknowledge parked state before logging a timeout
/// and giving up on the dump. Well above the worst-case drain-dance
/// and single-iteration park latency on healthy guests; a real
/// timeout indicates a vCPU stuck in KVM_RUN that the
/// `immediate_exit` kick failed to interrupt.
pub(super) const FREEZE_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(30);

/// Why [`super::KtstrVm::run_bsp_loop`] exited. Logged at break time
/// so an operator reading stderr (`BSP: loop exit reason=...`) can
/// diagnose a `code=-1` exit without correlating to peer-vCPU
/// stderr or `tracing` output.
///
/// Mapping to the BSP loop's exit_code:
///   - [`Shutdown`](Self::Shutdown) → exit_code = 0 (the only path
///     that overwrites the local `-1` sentinel).
///   - Every other variant → exit_code = -1, but
///     [`super::super::KtstrVm::collect_results`] re-derives the
///     final [`super::super::result::VmResult::exit_code`] from the
///     bulk-port `MSG_TYPE_EXIT` payload (or COM2 `KTSTR_EXIT:`
///     sentinel) when either is present, so a `-1` from the BSP
///     run-loop is not authoritative for caller-visible test
///     outcome.
#[derive(Debug, Clone, Copy)]
pub(super) enum BspExitReason {
    /// `kill.load(Acquire)` returned `true` at the top of the loop —
    /// some peer (an AP that observed [`super::exit_dispatch::ExitAction::Shutdown`] or
    /// [`super::exit_dispatch::ExitAction::Fatal`], the panic hook, the monitor thread on
    /// `MSG_TYPE_SCHED_EXIT`, or `collect_results`) flipped the flag.
    /// In particular, on a clean test exit where the kernel's i8042
    /// reset OUT is dispatched to a non-BSP vCPU, the AP path sets
    /// `kill` and the BSP exits via this branch. The default value
    /// for the local — every break path that does not explicitly
    /// reassign falls into this case.
    ExternalKill,
    /// BSP itself observed [`super::exit_dispatch::ExitAction::Shutdown`] from
    /// `classify_exit` (i8042 reset on x86_64, PSCI SystemEvent /
    /// `VcpuExit::Shutdown` on aarch64). The only path that sets
    /// exit_code to 0.
    Shutdown,
    /// BSP itself observed [`super::exit_dispatch::ExitAction::Fatal`] from `classify_exit`
    /// (`VcpuExit::FailEntry` or `VcpuExit::InternalError`). Kill
    /// flag is propagated to peers before break.
    Fatal,
    /// `bsp.run()` returned a non-EINTR/EAGAIN errno. Indicates a
    /// permanent KVM_RUN failure on the BSP vCPU fd.
    RunError,
}

/// Decoded contents of a guest-side `MSG_TYPE_SNAPSHOT_REQUEST` TLV
/// frame consumed from the virtio-console port-1 TX stream by the
/// coordinator's TOKEN_TX handler. The request id is echoed in the
/// matching `MSG_TYPE_SNAPSHOT_REPLY` payload so the guest's blocking
/// reader can pair the reply against its outstanding request; `kind`
/// selects the CAPTURE / WATCH dispatch path and `tag` carries the
/// snapshot name (CAPTURE) or symbol path (WATCH).
pub(super) struct SnapshotRequest {
    pub(super) request_id: u32,
    pub(super) kind: u32,
    pub(super) tag: String,
}

/// Dual-snapshot state machine the freeze coordinator's run-loop
/// advances on each capture cycle. Only the `TookEarly` variant is
/// reachable when `freeze_coord_dual_snapshot` is true; the single-
/// snapshot path drives the same transitions but skips the early
/// branch entirely.
///
/// * [`Idle`](Self::Idle) — no dump captured yet.
/// * [`TookEarly`](Self::TookEarly) — early snapshot captured
///   (dual-snapshot mode only); waiting for the err_exit latch to
///   fire.
/// * [`Done`](Self::Done) — late snapshot captured and emission
///   complete; coord just idles until kill / bsp_done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FreezeState {
    Idle,
    TookEarly,
    Done,
}
