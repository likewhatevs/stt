//! Guest-only typed senders for the host-bound bulk TLV stream.
//!
//! Every function in this module is callable ONLY from inside a
//! ktstr guest VM. The implementation routes through the underlying
//! port-1 writer ([`super::shm_ring::write_msg`]) which gates on
//! [`super::shm_ring::is_guest`]. Host-context invocations log a
//! `tracing::warn!` and no-op.
//!
//! Each function frames its payload with the corresponding
//! [`super::wire::MsgType`] so call sites do not pass raw u32 ids.
//! The frame format is the [`super::wire::ShmMessage`] header +
//! payload described on the [`super::wire`] module doc.
//!
//! # Backpressure
//!
//! The bulk channel uses the kernel virtio_console TX path: a full
//! virtqueue blocks the writer until the host's `add_used` rate
//! catches up. Callers that cannot block (panic hook, signal
//! handlers, anything called from a critical section) MUST use
//! [`send_crash`] which uses the non-blocking SHM-fallback path.

use crate::vmm::shm_ring;
use crate::vmm::wire::MsgType;

/// Send the guest exit code to the host. Payload: 4-byte LE i32.
///
/// Frames the exit code with [`MsgType::Exit`] and routes through
/// [`shm_ring::write_msg`]. The host's `collect_results` reads the
/// latest `Exit` entry to override the BSP run-loop sentinel.
pub fn send_exit(code: i32) {
    shm_ring::write_msg(MsgType::Exit.wire_value(), &code.to_le_bytes());
}

/// Send a JSON-encoded test result to the host. Payload: opaque bytes
/// (the framework's serialised [`crate::test_support::AssertResult`]
/// — the wire is type-agnostic).
///
/// Frames with [`MsgType::TestResult`].
pub fn send_test_result(result: &[u8]) {
    shm_ring::write_msg(MsgType::TestResult.wire_value(), result);
}

/// Send per-payload-invocation metrics. Payload: JSON-encoded
/// [`crate::test_support::PayloadMetrics`].
///
/// Frames with [`MsgType::PayloadMetrics`].
pub fn send_metrics(payload: &[u8]) {
    shm_ring::write_msg(MsgType::PayloadMetrics.wire_value(), payload);
}

/// Send a coverage profraw blob to the host. Payload: raw `.profraw`
/// bytes produced by `__llvm_profile_get_data`.
///
/// Frames with [`MsgType::Profraw`].
pub fn send_profraw(buf: &[u8]) {
    shm_ring::write_msg(MsgType::Profraw.wire_value(), buf);
}

/// Send a stimulus event from the guest step executor.
///
/// Payload: byte-serialised [`crate::vmm::shm_ring::StimulusPayload`]
/// (24 bytes, `IntoBytes`-derived). Frames with
/// [`MsgType::Stimulus`].
pub fn send_stimulus(payload: &[u8]) {
    shm_ring::write_msg(MsgType::Stimulus.wire_value(), payload);
}

/// Send a crash diagnostic. Non-blocking: this is the ONE message
/// type that may travel via the COM2 serial fallback when the bulk
/// virtio port is not yet open or has wedged on backpressure.
///
/// Routes through [`shm_ring::write_msg_nonblocking`] so a panic on
/// the thread that holds the bulk-port mutex cannot deadlock. The
/// COM2 serial fallback is performed at the call site (the panic
/// hook in `rust_init.rs`) which writes the same UTF-8 bytes to
/// `/dev/ttyS1` after invoking this helper.
///
/// Frames with [`MsgType::Crash`]. Returns `true` when the
/// non-blocking SHM write succeeded; `false` when SHM was unavailable
/// or the lock was contended (the caller should fall back to COM2).
pub fn send_crash(msg: &[u8]) -> bool {
    shm_ring::write_msg_nonblocking(MsgType::Crash.wire_value(), msg)
}

/// Send raw stdout/stderr from an LlmExtract payload. Payload:
/// JSON-encoded [`crate::test_support::RawPayloadOutput`].
///
/// Frames with [`MsgType::RawPayloadOutput`].
pub fn send_raw_output(payload: &[u8]) {
    shm_ring::write_msg(MsgType::RawPayloadOutput.wire_value(), payload);
}

/// Send a scheduler-process exit notification. Payload: 4-byte LE i32
/// containing the scheduler's exit code.
///
/// Frames with [`MsgType::SchedExit`]. The host's freeze coordinator
/// promotes a SchedExit message into the run-wide kill flag so the
/// test ends promptly instead of waiting for the watchdog.
pub fn send_sched_exit(code: i32) {
    shm_ring::write_msg(MsgType::SchedExit.wire_value(), &code.to_le_bytes());
}

/// Send a scenario-start marker.
pub fn send_scenario_start() {
    shm_ring::write_msg(MsgType::ScenarioStart.wire_value(), &[]);
}

/// Send a scenario-end marker. Payload: 8-byte LE u64 elapsed
/// milliseconds since scenario start.
pub fn send_scenario_end(elapsed_ms: u64) {
    shm_ring::write_msg(MsgType::ScenarioEnd.wire_value(), &elapsed_ms.to_le_bytes());
}

/// Request a host-driven snapshot. Wraps
/// [`shm_ring::snapshot_request`] under the typed name so the call
/// site does not name the SHM module directly.
pub fn request_snapshot(
    kind: u32,
    tag: &str,
    timeout: std::time::Duration,
) -> shm_ring::SnapshotRequestResult {
    shm_ring::snapshot_request(kind, tag, timeout)
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the typed sender wrappers.
    //!
    //! Every guest_comms helper routes through `shm_ring::write_msg`
    //! (or `write_msg_nonblocking` for `send_crash`) which gates on
    //! `shm_ring::is_guest()`. The host-context check rejects every
    //! call from these tests — verifying that gate holds is the
    //! safest unit-test scope: it confirms the wrappers do not
    //! accidentally write to a host process's memory.
    //!
    //! End-to-end transport (guest → bulk port → host drain → TLV
    //! parse) is exercised by the integration test suite under
    //! `tests/`.

    use super::*;

    /// `send_exit` from host context must be a no-op (no panic, no
    /// SHM mutation).
    #[test]
    fn send_exit_from_host_context_is_noop() {
        send_exit(0);
        send_exit(-1);
    }

    /// `send_test_result` from host context is a no-op.
    #[test]
    fn send_test_result_from_host_context_is_noop() {
        send_test_result(b"{\"ok\":true}");
    }

    /// `send_metrics` from host context is a no-op.
    #[test]
    fn send_metrics_from_host_context_is_noop() {
        send_metrics(b"{}");
    }

    /// `send_profraw` from host context is a no-op.
    #[test]
    fn send_profraw_from_host_context_is_noop() {
        send_profraw(b"\x01\x02\x03");
    }

    /// `send_stimulus` from host context is a no-op.
    #[test]
    fn send_stimulus_from_host_context_is_noop() {
        send_stimulus(&[0u8; 24]);
    }

    /// `send_crash` from host context returns false (SHM unavailable
    /// outside the guest).
    #[test]
    fn send_crash_from_host_context_returns_false() {
        assert!(!send_crash(b"panic body"));
    }

    /// `send_raw_output` from host context is a no-op.
    #[test]
    fn send_raw_output_from_host_context_is_noop() {
        send_raw_output(b"{\"stdout\":\"\"}");
    }

    /// `send_sched_exit` from host context is a no-op.
    #[test]
    fn send_sched_exit_from_host_context_is_noop() {
        send_sched_exit(0);
        send_sched_exit(-1);
    }

    /// `send_scenario_start` from host context is a no-op.
    #[test]
    fn send_scenario_start_from_host_context_is_noop() {
        send_scenario_start();
    }

    /// `send_scenario_end` from host context is a no-op.
    #[test]
    fn send_scenario_end_from_host_context_is_noop() {
        send_scenario_end(0);
        send_scenario_end(u64::MAX);
    }

    /// `request_snapshot` from host context returns `TransportError`
    /// (the SHM-ring gate rejects host invocations because no SHM is
    /// mapped and no doorbell GPA exists).
    #[test]
    fn request_snapshot_from_host_context_returns_transport_error() {
        let r = request_snapshot(0, "tag", std::time::Duration::from_millis(0));
        match r {
            shm_ring::SnapshotRequestResult::TransportError { .. } => {}
            other => panic!("expected TransportError from host context, got {other:?}"),
        }
    }
}
