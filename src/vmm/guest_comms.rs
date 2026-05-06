//! Guest-only typed senders for the host-bound bulk TLV stream.
//!
//! Every function in this module is callable ONLY from inside a
//! ktstr guest VM. Host-context invocations log a `tracing::warn!`
//! and no-op.
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
//! handlers, anything called from a critical section) MUST write
//! directly to COM2 (`/dev/ttyS1`) — the 16550 UART PIO path
//! commits synchronously inside `KVM_RUN` and never blocks the
//! guest on host backpressure. The panic hook in
//! [`super::rust_init`] follows this discipline.

use crate::vmm::wire::{
    MSG_TYPE_SNAPSHOT_REPLY, MsgType, SNAPSHOT_REASON_MAX, SNAPSHOT_STATUS_ERR, SNAPSHOT_STATUS_OK,
    SNAPSHOT_TAG_MAX, ShmMessage, SnapshotReplyPayload, SnapshotRequestPayload,
    SnapshotRequestResult,
};
use zerocopy::{FromBytes, IntoBytes};

/// Mutex serializing guest-side bulk-port writes. Every guest writer
/// (`write_msg`) takes this lock before submitting bytes to
/// `/dev/vport0p1`, so the in-stream order of bytes on port 1 stays
/// `[header][payload]` regardless of which producer (step executor,
/// sched-exit-mon, profraw flusher) emitted the frame.
pub static GUEST_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---------------------------------------------------------------------------
// is_guest detection
// ---------------------------------------------------------------------------

/// Detect whether the current process is running inside a ktstr guest
/// VM, by looking for the `KTSTR_GUEST=1` token on `/proc/cmdline`.
///
/// PID is NOT a reliable signal: the guest test code runs as forked
/// children of init (PID 1), not as PID 1 itself. The guest kernel
/// command line, populated by the host VMM, is the unique fingerprint.
///
/// The result is cached in a `OnceLock` — `/proc/cmdline` is read at
/// most once per process. False on the host (no cmdline match) and
/// false on any non-Linux platform that lacks `/proc/cmdline` (read
/// fails).
///
/// In test builds, the `IS_GUEST_TEST_OVERRIDE` thread-local takes
/// precedence over the `OnceLock`-cached natural detection; the
/// `OnceLock` is consulted only when no override is set on the
/// calling thread.
pub fn is_guest() -> bool {
    #[cfg(test)]
    {
        // Test-only override: tests run on the host but need to
        // exercise the guest-only path (write_msg). The override is
        // thread-local so parallel tests don't fight over it.
        if let Some(v) = IS_GUEST_TEST_OVERRIDE.with(|c| c.get()) {
            return v;
        }
    }
    static IS_GUEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *IS_GUEST.get_or_init(|| {
        std::fs::read_to_string("/proc/cmdline")
            .ok()
            .is_some_and(|c| c.split_whitespace().any(|tok| tok == "KTSTR_GUEST=1"))
    })
}

// Test-only thread-local override for `is_guest`. `None` means
// "consult /proc/cmdline"; `Some(b)` pins the result for the
// current thread. Per-thread so parallel tests cannot interfere.
#[cfg(test)]
thread_local! {
    static IS_GUEST_TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// RAII guard that overrides [`is_guest`] for the duration of its
/// scope on the current thread, and restores the previous value on
/// drop. Avoids leaking override state across tests sharing a thread
/// (e.g. via test-runner thread pools).
///
/// `pub(crate)` so other test modules in the crate can use the
/// fixture when they need to exercise guest-only paths.
#[cfg(test)]
pub(crate) struct IsGuestOverrideGuard {
    prev: Option<bool>,
}

#[cfg(test)]
impl IsGuestOverrideGuard {
    pub(crate) fn new(value: bool) -> Self {
        let prev = IS_GUEST_TEST_OVERRIDE.with(|c| c.replace(Some(value)));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for IsGuestOverrideGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        IS_GUEST_TEST_OVERRIDE.with(|c| c.set(prev));
    }
}

// ---------------------------------------------------------------------------
// Bulk-port writer (guest → host TLV)
// ---------------------------------------------------------------------------

/// Reject a call to a guest-only entry point when invoked from host
/// context. Returns `true` if the caller may proceed (we're inside a
/// guest VM); `false` after emitting a `tracing::warn!` that names the
/// caller and the message type, so a host-side caller surfaces in the
/// log instead of silently no-op'ing.
fn assert_guest_context(fn_name: &str, msg_type: u32) -> bool {
    if !is_guest() {
        tracing::warn!(
            msg_type = msg_type,
            "guest_comms::{fn_name} called from host context"
        );
        return false;
    }
    true
}

/// Cached `/dev/vport0p1` writer. Opened lazily on the first
/// successful `write_to_bulk_port` call after the kernel's
/// virtio_console driver creates the device node (post multiport
/// handshake). `OnceLock<Option<...>>` so repeated open failures
/// (port not yet ready) do not pin the slot to None permanently —
/// instead we re-attempt until `try_open_bulk_port` succeeds, then
/// cache the file handle for the rest of the process.
static BULK_PORT_FD: std::sync::OnceLock<std::sync::Mutex<Option<std::fs::File>>> =
    std::sync::OnceLock::new();

/// Try to open `/dev/vport0p1` for writing. Returns None when the
/// device is not yet present — the kernel virtio_console driver
/// creates it only after the host emits PORT_OPEN on the c_ivq for
/// port 1 and the kernel's `find_port_by_id` resolves the
/// `/sys/class/virtio-ports/vport0p1` entry.
///
/// Open mode: write-only, blocking. The kernel's `port_fops_write`
/// path blocks the writer when the host's `add_used` rate lags;
/// that's the backpressure mechanism that replaces drop semantics.
fn try_open_bulk_port() -> Option<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/vport0p1")
        .ok()
}

/// Write a TLV-framed message to the host through the bulk channel
/// (virtio-console port 1, `/dev/vport0p1`). The frame format is
/// 16-byte [`ShmMessage`] header + `payload.len()` bytes; the host
/// parses the same byte stream via [`super::host_comms::parse_tlv_stream`].
///
/// Backpressure: the kernel's virtio_console TX path (`hvc_push` /
/// `port_fops_write`) blocks the writer until the host's
/// `add_used` rate catches up. There is no drop path; callers that
/// cannot block (panic hook, signal handlers, anything called from
/// a critical section) MUST write directly to COM2 (`/dev/ttyS1`).
///
/// `assert_guest_context` rejects host-context invocations with a
/// `tracing::warn` so a host-side caller surfaces in the log instead
/// of silently no-op'ing.
fn write_msg(msg_type: u32, payload: &[u8]) {
    if !assert_guest_context("write_msg", msg_type) {
        return;
    }
    let _guard = GUEST_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    write_to_bulk_port(msg_type, payload);
}

/// Try to write a TLV-framed message to `/dev/vport0p1`. Returns
/// true when the message was fully written, false when the bulk
/// port is not yet available or the write failed.
///
/// Lazy-open semantics: the multiport handshake completes
/// asynchronously during kernel virtio_console init, so the device
/// node may appear any time after the first `write_msg` call. We
/// retry the open on every call until it succeeds; once cached,
/// subsequent writes go through the cached `File`.
///
/// Submission shape: header and payload are submitted together via
/// `writev(2)` with two `iovec` slices, avoiding a per-call concat
/// allocation. The host's [`super::bulk::HostAssembler`] tolerates
/// partial frames in the byte stream, so any per-iovec virtqueue
/// submissions reassemble correctly.
fn write_to_bulk_port(msg_type: u32, payload: &[u8]) -> bool {
    let slot = BULK_PORT_FD.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        match try_open_bulk_port() {
            Some(f) => *guard = Some(f),
            None => return false,
        }
    }
    let f = guard.as_mut().expect("bulk port handle just installed");
    let Ok(length_u32) = u32::try_from(payload.len()) else {
        tracing::warn!(
            len = payload.len(),
            msg_type,
            "write_to_bulk_port: payload exceeds u32::MAX; dropping"
        );
        return false;
    };
    let msg = ShmMessage {
        msg_type,
        length: length_u32,
        crc32: crc32fast::hash(payload),
        _pad: 0,
    };
    let header_bytes = msg.as_bytes();
    let total = header_bytes.len() + payload.len();
    let fd = std::os::unix::io::AsRawFd::as_raw_fd(f);
    let mut iovs = [
        std::io::IoSlice::new(header_bytes),
        std::io::IoSlice::new(payload),
    ];
    let mut bufs: &mut [std::io::IoSlice<'_>] = &mut iovs[..];
    let mut written: usize = 0;
    while !bufs.is_empty() {
        // SAFETY: `bufs` is a non-empty slice of `IoSlice<'_>`, which
        // is `#[repr(transparent)]` over `libc::iovec` on unix targets.
        // Casting `*const IoSlice` to `*const libc::iovec` is sound.
        // `fd` is a borrowed raw fd from the cached `File`; the
        // `File` outlives the syscall because `guard` keeps it owned.
        let r = unsafe {
            libc::writev(
                fd,
                bufs.as_ptr() as *const libc::iovec,
                bufs.len() as libc::c_int,
            )
        };
        if r < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            tracing::warn!(
                %err,
                msg_type,
                len = payload.len(),
                "write_to_bulk_port: writev failed"
            );
            // Drop the cached handle so the next call retries the open
            // (the device may have transiently closed during a guest
            // reset path).
            *guard = None;
            return false;
        }
        if r == 0 {
            // `writev` returning 0 with no error is unexpected for a
            // char device; treat as an EOF-like failure.
            tracing::warn!(
                msg_type,
                len = payload.len(),
                written,
                total,
                "write_to_bulk_port: writev returned 0"
            );
            *guard = None;
            return false;
        }
        let n = r as usize;
        written += n;
        std::io::IoSlice::advance_slices(&mut bufs, n);
    }
    debug_assert_eq!(written, total);
    true
}

// ---------------------------------------------------------------------------
// Typed senders
// ---------------------------------------------------------------------------

/// Send the guest exit code to the host. Payload: 4-byte LE i32.
///
/// Frames the exit code with [`MsgType::Exit`] and routes through
/// the bulk port. The host's `collect_results` reads the latest
/// `Exit` entry to override the BSP run-loop sentinel.
pub fn send_exit(code: i32) {
    write_msg(MsgType::Exit.wire_value(), &code.to_le_bytes());
}

/// Send a JSON-encoded test result to the host. Payload: opaque bytes
/// (the framework's serialised [`crate::test_support::AssertResult`]
/// — the wire is type-agnostic).
///
/// Frames with [`MsgType::TestResult`].
pub fn send_test_result(result: &[u8]) {
    write_msg(MsgType::TestResult.wire_value(), result);
}

/// Send per-payload-invocation metrics. Payload: JSON-encoded
/// [`crate::test_support::PayloadMetrics`].
///
/// Frames with [`MsgType::PayloadMetrics`].
pub fn send_metrics(payload: &[u8]) {
    write_msg(MsgType::PayloadMetrics.wire_value(), payload);
}

/// Send a coverage profraw blob to the host. Payload: raw `.profraw`
/// bytes produced by `__llvm_profile_get_data`.
///
/// Frames with [`MsgType::Profraw`].
pub fn send_profraw(buf: &[u8]) {
    write_msg(MsgType::Profraw.wire_value(), buf);
}

/// Send a stimulus event from the guest step executor.
///
/// Payload: byte-serialised [`crate::vmm::wire::StimulusPayload`]
/// (24 bytes, `IntoBytes`-derived). Frames with
/// [`MsgType::Stimulus`].
pub fn send_stimulus(payload: &[u8]) {
    write_msg(MsgType::Stimulus.wire_value(), payload);
}

/// Send raw stdout/stderr from an LlmExtract payload. Payload:
/// JSON-encoded [`crate::test_support::RawPayloadOutput`].
///
/// Frames with [`MsgType::RawPayloadOutput`].
pub fn send_raw_output(payload: &[u8]) {
    write_msg(MsgType::RawPayloadOutput.wire_value(), payload);
}

/// Send a scheduler-process exit notification. Payload: 4-byte LE i32
/// containing the scheduler's exit code.
///
/// Frames with [`MsgType::SchedExit`]. The host's freeze coordinator
/// promotes a SchedExit message into the run-wide kill flag so the
/// test ends promptly instead of waiting for the watchdog.
pub fn send_sched_exit(code: i32) {
    write_msg(MsgType::SchedExit.wire_value(), &code.to_le_bytes());
}

/// Send a scenario-start marker.
pub fn send_scenario_start() {
    write_msg(MsgType::ScenarioStart.wire_value(), &[]);
}

/// Send a scenario-end marker. Payload: 8-byte LE u64 elapsed
/// milliseconds since scenario start.
pub fn send_scenario_end(elapsed_ms: u64) {
    write_msg(MsgType::ScenarioEnd.wire_value(), &elapsed_ms.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Snapshot request (guest → host) + reply read-back
// ---------------------------------------------------------------------------

/// Monotonic guest-side request id counter. Bumped by every call to
/// [`request_snapshot`] before publishing the request frame.
/// `AtomicU32` so concurrent requests from different guest threads do
/// not produce duplicate ids. Wraparound past `u32::MAX` is
/// theoretically possible after billions of requests; the host's
/// reply pairing tolerates it because the comparison is equality
/// against the issuer's most-recent value, not a monotonicity check.
static SNAPSHOT_REQUEST_COUNTER: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(1);

/// Mutex serialising guest-side snapshot requests. Without this two
/// guest threads issuing `Op::Snapshot` concurrently could interleave
/// their TX writes and read each other's replies. The freeze
/// coordinator's `on_demand_in_flight` latch already collapses
/// doorbell floods to one capture per thaw on the host side; this
/// lock keeps the guest-side request/reply pairing well-defined too.
static SNAPSHOT_REQUEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Cached read-side handle on `/dev/vport0p1`. Reused across snapshot
/// requests so the kernel's port-1 read queue refills only once per
/// guest process. `OnceLock<Option<File>>` so a not-yet-ready open
/// (multiport handshake still in flight) does not pin the slot to
/// None — the next call retries.
static BULK_PORT_READ_FD: std::sync::OnceLock<std::sync::Mutex<Option<std::fs::File>>> =
    std::sync::OnceLock::new();

/// Try to open `/dev/vport0p1` for reading (read-only, blocking).
/// Returns `None` when the device is not yet present; the multiport
/// handshake completes asynchronously so the read-side handle may
/// not be available on the first `request_snapshot`.
fn try_open_bulk_port_read() -> Option<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .open("/dev/vport0p1")
        .ok()
}

/// Number of fast-poll iterations at the start of
/// [`bounded_read_exact`] before escalating to the slow-poll cadence.
/// Four iterations of 100µs gives ~400µs of fast-path coverage,
/// enough to absorb a host reply that lands in the virtqueue while
/// the guest is still entering `ppoll`, without burning more than
/// a hundred microseconds of cumulative wake-up budget.
const SNAPSHOT_FAST_POLL_ITERS: u32 = 4;
/// Per-iteration ppoll timeout for the first
/// [`SNAPSHOT_FAST_POLL_ITERS`] iterations (100µs). Sub-millisecond
/// granularity is the reason this path uses `ppoll` rather than
/// `poll(2)` (which only takes millisecond timeouts).
const SNAPSHOT_FAST_POLL_INTERVAL: std::time::Duration =
    std::time::Duration::from_micros(100);
/// Per-iteration ppoll timeout after the fast-poll preamble (5ms).
/// Bounds the worst-case extra latency when virtio_console's
/// `port_fops_poll` does not deliver an early wake, while keeping
/// vCPU-thread wake-up cost low across the full snapshot deadline.
const SNAPSHOT_SLOW_POLL_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(5);

/// Read exactly `buf.len()` bytes from `f`, bounded by `deadline`.
/// Uses `ppoll(POLLIN)` between reads to wait without blocking past
/// the deadline. Returns `ErrorKind::TimedOut` when the deadline
/// expires before the read completes.
///
/// Each `ppoll` call's timeout is capped at an adaptive interval, not
/// the full remaining deadline:
///
/// * The first [`SNAPSHOT_FAST_POLL_ITERS`] iterations use a
///   [`SNAPSHOT_FAST_POLL_INTERVAL`] timeout (100µs). On the common
///   path the host's reply is already buffered in the virtqueue by
///   the time the guest enters `ppoll`, so a sub-millisecond bound
///   keeps wake-up latency low without burning CPU on the vCPU
///   thread.
/// * Subsequent iterations escalate to [`SNAPSHOT_SLOW_POLL_INTERVAL`]
///   (5ms), bounding the per-iteration wakeup cost while still
///   guaranteeing prompt deadline checks across the outer loop.
///
/// Each interval is further capped against the remaining deadline so
/// the loop never overshoots the caller's timeout.
fn bounded_read_exact(
    f: &mut std::fs::File,
    buf: &mut [u8],
    deadline: std::time::Instant,
) -> std::io::Result<()> {
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    let fd = f.as_raw_fd();
    let mut filled = 0usize;
    let mut iter: u32 = 0;
    while filled < buf.len() {
        let now = std::time::Instant::now();
        if now >= deadline {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!(
                    "snapshot reply deadline elapsed after reading {filled} of {} header/payload bytes",
                    buf.len()
                ),
            ));
        }
        let remaining = deadline - now;
        let interval = if iter < SNAPSHOT_FAST_POLL_ITERS {
            SNAPSHOT_FAST_POLL_INTERVAL
        } else {
            SNAPSHOT_SLOW_POLL_INTERVAL
        };
        // Cap the per-iteration sleep at min(interval, remaining) so
        // the last iteration before the deadline does not overshoot.
        let slice = remaining.min(interval);
        let ts = libc::timespec {
            tv_sec: slice.as_secs() as libc::time_t,
            tv_nsec: slice.subsec_nanos() as libc::c_long,
        };
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pfd is a valid &mut to a single pollfd; nfds is 1.
        // `ts` is a local timespec passed by const pointer. sigmask
        // is null so the caller's signal mask applies unchanged.
        // Every poll outcome (ready, timeout, EINTR, error) loops
        // back to the read attempt; EINTR is harmless because the
        // outer loop re-evaluates the deadline on every iteration.
        let pr = unsafe { libc::ppoll(&mut pfd, 1, &ts, std::ptr::null()) };
        iter = iter.saturating_add(1);
        if pr < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if pr == 0 {
            // ppoll timeout — re-check deadline at the loop head.
            continue;
        }
        match f.read(&mut buf[filled..]) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "snapshot reply read returned 0 after {filled} of {} bytes",
                        buf.len()
                    ),
                ));
            }
            Ok(n) => {
                filled += n;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Read a single TLV frame (16-byte header + payload bytes) from
/// `/dev/vport0p1`. Returns the parsed message type and payload on
/// success.
///
/// Reads the header with `bounded_read_exact`, decodes the length, then
/// reads the payload with `bounded_read_exact`. On any I/O failure
/// (premature EOF, EINTR, etc.) the cached handle is dropped so a
/// subsequent call retries the open.
fn read_bulk_port_frame(
    f: &mut std::fs::File,
    deadline: std::time::Instant,
) -> std::io::Result<(u32, Vec<u8>)> {
    let mut header = [0u8; std::mem::size_of::<ShmMessage>()];
    bounded_read_exact(f, &mut header, deadline)?;
    let msg = ShmMessage::read_from_bytes(&header).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "ShmMessage::read_from_bytes failed (header underflow)",
        )
    })?;
    let length = msg.length as usize;
    let mut payload = vec![0u8; length];
    if length > 0 {
        bounded_read_exact(f, &mut payload, deadline)?;
    }
    let computed = crc32fast::hash(&payload);
    if computed != msg.crc32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "TLV CRC mismatch: header crc=0x{:08x} computed=0x{computed:08x} length={length}",
                msg.crc32
            ),
        ));
    }
    Ok((msg.msg_type, payload))
}

/// Request a host-driven snapshot. Publishes a snapshot request via
/// the virtio-console port-1 TLV stream and blocks reading port 1 RX
/// until a matching [`MsgType::SnapshotReply`] arrives (or `timeout`
/// elapses).
///
/// `kind` selects the dispatch path on the host:
/// [`crate::vmm::wire::SNAPSHOT_KIND_CAPTURE`] for a capture-now
/// request, [`crate::vmm::wire::SNAPSHOT_KIND_WATCH`] for a
/// hardware-watchpoint registration.
///
/// `tag` is copied into the request payload's tag buffer up to
/// [`SNAPSHOT_TAG_MAX`] bytes. Longer tags are truncated.
///
/// Returns one of [`SnapshotRequestResult`] variants. The serialised
/// guest lock ensures only one in-flight request per process — this
/// matches the host coordinator's `on_demand_in_flight` invariant.
pub fn request_snapshot(
    kind: u32,
    tag: &str,
    timeout: std::time::Duration,
) -> SnapshotRequestResult {
    if !is_guest() {
        return SnapshotRequestResult::TransportError {
            reason: "request_snapshot called from host context (virtio-console port 1 \
                     is reachable only from inside the guest)"
                .into(),
        };
    }
    let _guard = SNAPSHOT_REQUEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // Allocate a request id. Skip 0 so the wait loop's `reply.request_id
    // == request_id` check cannot accidentally match a zero-initialised
    // reply payload from an earlier protocol version.
    let mut request_id =
        SNAPSHOT_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    if request_id == 0 {
        request_id = SNAPSHOT_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    // Build the request payload.
    let tag_bytes = tag.as_bytes();
    let tag_len = tag_bytes.len().min(SNAPSHOT_TAG_MAX);
    let mut tag_buf = [0u8; SNAPSHOT_TAG_MAX];
    tag_buf[..tag_len].copy_from_slice(&tag_bytes[..tag_len]);
    let payload = SnapshotRequestPayload {
        request_id,
        kind,
        tag: tag_buf,
    };
    // Send via the existing port-1 TX writer. `write_msg` already
    // takes `GUEST_WRITE_LOCK` internally, so this serialises with
    // every other guest TLV producer.
    let bytes = payload.as_bytes();
    write_msg(MsgType::SnapshotRequest.wire_value(), bytes);
    // Open the read side of the bulk port. Lazy because the
    // multiport handshake completes asynchronously; the first
    // `request_snapshot` may arrive before `/dev/vport0p1` is
    // creatable.
    let read_slot = BULK_PORT_READ_FD.get_or_init(|| std::sync::Mutex::new(None));
    let mut read_guard = read_slot.lock().unwrap_or_else(|e| e.into_inner());
    if read_guard.is_none() {
        match try_open_bulk_port_read() {
            Some(f) => *read_guard = Some(f),
            None => {
                return SnapshotRequestResult::TransportError {
                    reason: "/dev/vport0p1 not yet readable on this guest \
                             (multiport handshake still in flight); retry shortly"
                        .into(),
                };
            }
        }
    }
    let f = read_guard
        .as_mut()
        .expect("bulk port read handle just installed");
    // Read TLV reply frames until we observe one whose payload
    // request_id matches ours. Frames addressed to other request ids
    // (none in current protocol — the host only writes replies in
    // response to a specific request) or unknown msg_types are
    // logged + dropped.
    let deadline = std::time::Instant::now() + timeout;
    loop {
        let now = std::time::Instant::now();
        if now >= deadline {
            return SnapshotRequestResult::TransportError {
                reason: format!(
                    "host did not deliver matching snapshot reply within {timeout:?} \
                     (request_id={request_id}, kind={kind})"
                ),
            };
        }
        let frame = match read_bulk_port_frame(f, deadline) {
            Ok(frame) => frame,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return SnapshotRequestResult::TransportError {
                    reason: format!(
                        "snapshot reply deadline elapsed before frame complete \
                         (request_id={request_id}, kind={kind}): {e}"
                    ),
                };
            }
            Err(e) => {
                // I/O error on the read fd — drop the cached
                // handle so the next call retries the open and
                // surface the failure to the caller.
                *read_guard = None;
                return SnapshotRequestResult::TransportError {
                    reason: format!(
                        "snapshot reply read failed (request_id={request_id}): {e}"
                    ),
                };
            }
        };
        let (msg_type, frame_payload) = frame;
        if msg_type != MSG_TYPE_SNAPSHOT_REPLY {
            tracing::warn!(
                msg_type,
                len = frame_payload.len(),
                request_id,
                "request_snapshot: ignoring unexpected TLV on port 1 RX (only \
                 SnapshotReply is expected on this transport in current protocol)"
            );
            continue;
        }
        if frame_payload.len() != std::mem::size_of::<SnapshotReplyPayload>() {
            tracing::warn!(
                request_id,
                got = frame_payload.len(),
                want = std::mem::size_of::<SnapshotReplyPayload>(),
                "request_snapshot: malformed reply payload size; ignoring"
            );
            continue;
        }
        let reply = match SnapshotReplyPayload::read_from_bytes(&frame_payload) {
            Ok(r) => r,
            Err(_) => {
                tracing::warn!(
                    request_id,
                    "request_snapshot: SnapshotReplyPayload::read_from_bytes failed; ignoring"
                );
                continue;
            }
        };
        if reply.request_id != request_id {
            tracing::warn!(
                expected = request_id,
                got = reply.request_id,
                "request_snapshot: stale reply id (likely a leftover from a prior \
                 request that timed out on the guest side); ignoring"
            );
            continue;
        }
        return match reply.status {
            SNAPSHOT_STATUS_OK => SnapshotRequestResult::Ok,
            SNAPSHOT_STATUS_ERR => {
                let len = reply
                    .reason
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(SNAPSHOT_REASON_MAX);
                let reason = String::from_utf8_lossy(&reply.reason[..len]).to_string();
                SnapshotRequestResult::HostError { reason }
            }
            other => SnapshotRequestResult::TransportError {
                reason: format!(
                    "host reply with unknown status {other} \
                     (expected OK={SNAPSHOT_STATUS_OK} or ERR={SNAPSHOT_STATUS_ERR})"
                ),
            },
        };
    }
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the typed sender wrappers.
    //!
    //! Every guest_comms helper routes through `write_msg`
    //! which gates on `is_guest()`. The host-context check
    //! rejects every call from these tests — verifying that gate
    //! holds is the safest unit-test scope: it confirms the wrappers
    //! do not accidentally write to a host process's memory.
    //!
    //! End-to-end transport (guest → bulk port → host drain → TLV
    //! parse) is exercised by the integration test suite under
    //! `tests/`.

    use super::*;

    /// `send_exit` from host context must be a no-op (no panic).
    #[test]
    fn send_exit_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_exit(0);
        send_exit(-1);
    }

    /// `send_test_result` from host context is a no-op.
    #[test]
    fn send_test_result_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_test_result(b"{\"ok\":true}");
    }

    /// `send_metrics` from host context is a no-op.
    #[test]
    fn send_metrics_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_metrics(b"{}");
    }

    /// `send_profraw` from host context is a no-op.
    #[test]
    fn send_profraw_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_profraw(b"\x01\x02\x03");
    }

    /// `send_stimulus` from host context is a no-op.
    #[test]
    fn send_stimulus_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_stimulus(&[0u8; 24]);
    }

    /// `send_raw_output` from host context is a no-op.
    #[test]
    fn send_raw_output_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_raw_output(b"{\"stdout\":\"\"}");
    }

    /// `send_sched_exit` from host context is a no-op.
    #[test]
    fn send_sched_exit_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_sched_exit(0);
        send_sched_exit(-1);
    }

    /// `send_scenario_start` from host context is a no-op.
    #[test]
    fn send_scenario_start_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_scenario_start();
    }

    /// `send_scenario_end` from host context is a no-op.
    #[test]
    fn send_scenario_end_from_host_context_is_noop() {
        let _g = IsGuestOverrideGuard::new(false);
        send_scenario_end(0);
        send_scenario_end(u64::MAX);
    }

    /// `request_snapshot` from host context returns `TransportError`.
    #[test]
    fn request_snapshot_from_host_context_returns_transport_error() {
        let _g = IsGuestOverrideGuard::new(false);
        let r = request_snapshot(0, "tag", std::time::Duration::from_millis(0));
        match r {
            SnapshotRequestResult::TransportError { .. } => {}
            other => panic!("expected TransportError from host context, got {other:?}"),
        }
    }

    #[test]
    fn is_guest_override_round_trips_through_thread_local() {
        // Toggling override should affect is_guest() result.
        {
            let _g = IsGuestOverrideGuard::new(false);
            assert!(!is_guest());
        }
        {
            let _g = IsGuestOverrideGuard::new(true);
            assert!(is_guest());
        }
    }

    #[test]
    fn is_guest_override_guards_nest_correctly() {
        let _outer = IsGuestOverrideGuard::new(true);
        assert!(is_guest());
        {
            let _inner = IsGuestOverrideGuard::new(false);
            assert!(!is_guest());
        }
        // Inner dropped — outer's value is restored.
        assert!(is_guest());
    }
}
