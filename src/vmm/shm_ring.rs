/// Shared-memory ring buffer for guest-to-host data transfer.
///
/// The guest writes TLV-framed messages into a fixed region at the top of
/// guest physical memory. The host drains both mid-flight and after VM exit.
/// Multiple guest-side producers (step executor, sched-exit-mon) serialize
/// writes via `SHM_WRITE_LOCK`. Single consumer (host), no read-side locking.
///
/// Memory layout:
///   [ShmRingHeader (40 bytes)] [data (capacity bytes)]
///
/// The SHM region is excluded from usable RAM: on x86_64 via an E820 gap
/// (no E820 entry covers it), on aarch64 via FDT /reserved-memory and
/// /memreserve/. The guest init binary discovers the region via KTSTR_SHM_BASE
/// and KTSTR_SHM_SIZE parameters on the kernel command line.
use std::ptr;

use zerocopy::{FromBytes, IntoBytes};

/// Result of a successful `/dev/mem` mmap of the SHM region.
pub(crate) struct ShmMmap {
    /// Pointer to the start of the SHM region (page-offset adjusted).
    pub ptr: *mut u8,
    /// Base address passed to munmap (page-aligned). Held so the
    /// caller can perform a manual munmap if needed; the lib path
    /// never invokes munmap explicitly because the mapping
    /// outlives every guest thread (set once via `OnceLock`).
    #[allow(dead_code)]
    pub map_base: *mut libc::c_void,
    /// Size passed to munmap. Same rationale as `map_base`.
    #[allow(dead_code)]
    pub map_size: usize,
}

/// Page-aligned mmap of a physical address range via an open `/dev/mem` fd.
/// Returns the adjusted pointer to `shm_base` within the mapping.
pub(crate) fn mmap_devmem(
    fd: std::os::unix::io::RawFd,
    shm_base: u64,
    shm_size: u64,
) -> Option<ShmMmap> {
    let page_size = super::setup::host_page_size();
    let aligned_base = shm_base & !(page_size - 1);
    let offset_in_page = (shm_base - aligned_base) as usize;
    let map_size = shm_size as usize + offset_in_page;

    let map_base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            map_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            aligned_base as libc::off_t,
        )
    };
    if map_base == libc::MAP_FAILED {
        return None;
    }

    let ptr = unsafe { (map_base as *mut u8).add(offset_in_page) };
    Some(ShmMmap {
        ptr,
        map_base,
        map_size,
    })
}

/// Outcome of a guest-driven snapshot request: ok, error with reason,
/// or transport failure (port unavailable / not in guest / timeout).
#[derive(Debug)]
pub enum SnapshotRequestResult {
    /// Host completed the request. For
    /// [`super::wire::SNAPSHOT_KIND_CAPTURE`] this means the report
    /// was stored on the bridge under the supplied tag; for
    /// [`super::wire::SNAPSHOT_KIND_WATCH`] this means the hardware
    /// watchpoint was armed.
    Ok,
    /// Host accepted the request but completed it as a failure. The
    /// reason carries the host-supplied diagnostic text (truncated to
    /// [`super::wire::SNAPSHOT_REASON_MAX`] bytes).
    HostError { reason: String },
    /// Transport failed (called from host context, port not yet open,
    /// host did not reply within `timeout`, malformed reply frame).
    /// The supplied diagnostic names the underlying cause.
    TransportError { reason: String },
}

/// Monotonic guest-side request id counter. Bumped by every call to
/// [`snapshot_request`] before publishing the request frame.
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
/// not be available on the first `snapshot_request`.
fn try_open_bulk_port_read() -> Option<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .open("/dev/vport0p1")
        .ok()
}

/// Read a single TLV frame (16-byte header + payload bytes) from
/// `/dev/vport0p1`. Returns the parsed message type and payload on
/// success.
///
/// Reads the header with `read_exact`, decodes the length, then
/// reads the payload with `read_exact`. On any I/O failure
/// (premature EOF, EINTR, etc.) the cached handle is dropped so a
/// subsequent call retries the open.
///
/// `deadline` bounds total wait time across header + payload reads:
/// the read fd is set to non-blocking before each `read` and a
/// `poll(POLLIN)` waits up to the remaining budget; on timeout the
/// function returns `Err` so the caller can surface a transport
/// failure rather than blocking forever on a wedged host.
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

/// Read exactly `buf.len()` bytes from `f`, bounded by `deadline`.
/// Uses `poll(POLLIN)` between reads to wait without blocking past
/// the deadline. Returns `ErrorKind::TimedOut` when the deadline
/// expires before the read completes.
fn bounded_read_exact(
    f: &mut std::fs::File,
    buf: &mut [u8],
    deadline: std::time::Instant,
) -> std::io::Result<()> {
    use std::io::Read;
    use std::os::unix::io::AsRawFd;
    let fd = f.as_raw_fd();
    let mut filled = 0usize;
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
        let remaining_ms = (deadline - now).as_millis().min(i32::MAX as u128) as i32;
        let mut pfd = libc::pollfd {
            fd,
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: pfd is a valid &mut to a single pollfd; nfds is 1.
        // Every poll outcome (ready, timeout, EINTR, error) loops
        // back to the read attempt; EINTR is harmless because the
        // outer loop re-evaluates the deadline on every iteration.
        let pr = unsafe { libc::poll(&mut pfd, 1, remaining_ms) };
        if pr < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(err);
        }
        if pr == 0 {
            // poll timeout — re-check deadline at the loop head.
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

/// Guest-only. Publish a snapshot request via the virtio-console
/// port-1 TLV stream and block reading port 1 RX until a matching
/// [`super::wire::MsgType::SnapshotReply`] arrives (or `timeout`
/// elapses).
///
/// `kind` selects the dispatch path on the host:
/// [`super::wire::SNAPSHOT_KIND_CAPTURE`] for a capture-now request,
/// [`super::wire::SNAPSHOT_KIND_WATCH`] for a hardware-watchpoint
/// registration.
///
/// `tag` is copied into the request payload's tag buffer up to
/// [`super::wire::SNAPSHOT_TAG_MAX`] bytes. Longer tags are
/// truncated.
///
/// Returns one of [`SnapshotRequestResult`] variants. The serialised
/// guest lock ensures only one in-flight request per process — this
/// matches the host coordinator's `on_demand_in_flight` invariant.
pub fn snapshot_request(
    kind: u32,
    tag: &str,
    timeout: std::time::Duration,
) -> SnapshotRequestResult {
    use super::wire::{
        MSG_TYPE_SNAPSHOT_REPLY, MsgType, SNAPSHOT_REASON_MAX, SNAPSHOT_STATUS_ERR,
        SNAPSHOT_STATUS_OK, SNAPSHOT_TAG_MAX, SnapshotReplyPayload, SnapshotRequestPayload,
    };
    use zerocopy::IntoBytes;

    if !is_guest() {
        return SnapshotRequestResult::TransportError {
            reason: "snapshot_request called from host context (virtio-console port 1 \
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
    let mut request_id = SNAPSHOT_REQUEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
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
    // takes `SHM_WRITE_LOCK` internally, so this serialises with
    // every other guest TLV producer.
    let bytes = payload.as_bytes();
    write_msg(MsgType::SnapshotRequest.wire_value(), bytes);
    // Open the read side of the bulk port. Lazy because the
    // multiport handshake completes asynchronously; the first
    // `snapshot_request` may arrive before `/dev/vport0p1` is
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
                "snapshot_request: ignoring unexpected TLV on port 1 RX (only \
                 SnapshotReply is expected on this transport in current protocol)"
            );
            continue;
        }
        if frame_payload.len() != std::mem::size_of::<SnapshotReplyPayload>() {
            tracing::warn!(
                request_id,
                got = frame_payload.len(),
                want = std::mem::size_of::<SnapshotReplyPayload>(),
                "snapshot_request: malformed reply payload size; ignoring"
            );
            continue;
        }
        let reply = match SnapshotReplyPayload::read_from_bytes(&frame_payload) {
            Ok(r) => r,
            Err(_) => {
                tracing::warn!(
                    request_id,
                    "snapshot_request: SnapshotReplyPayload::read_from_bytes failed; ignoring"
                );
                continue;
            }
        };
        if reply.request_id != request_id {
            tracing::warn!(
                expected = request_id,
                got = reply.request_id,
                "snapshot_request: stale reply id (likely a leftover from a prior \
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

/// Magic value identifying a valid SHM ring header.
pub const SHM_RING_MAGIC: u32 = 0x5354_4d52; // "STMR"

// Message-type discriminants are defined in [`super::wire`]; the
// SHM ring shares the same TLV format as the virtio-console bulk
// stream so a single source of truth keeps both transports in
// sync. These re-exports preserve the legacy `shm_ring::MSG_TYPE_*`
// import paths used throughout the host-side drain code (eval,
// reader, output, freeze coord). New call sites should import from
// `super::wire` directly; the re-exports stay until the in-tree
// migration completes.
//
// `#[allow(unused_imports)]` mirrors the pattern on the `NetConfig` /
// `KVM_INTERESTING_STATS` re-exports in `vmm/mod.rs`: the lib build
// doesn't see internal readers of every name (PROFRAW, SCENARIO_END,
// SCENARIO_START flow only through `crate::vmm::wire` post-migration),
// but the public path must remain reachable.
#[allow(unused_imports)]
pub use super::wire::{
    MSG_TYPE_CRASH, MSG_TYPE_EXIT, MSG_TYPE_PAYLOAD_METRICS, MSG_TYPE_PROFRAW,
    MSG_TYPE_RAW_PAYLOAD_OUTPUT, MSG_TYPE_SCENARIO_END, MSG_TYPE_SCENARIO_START,
    MSG_TYPE_SCHED_EXIT, MSG_TYPE_STIMULUS, MSG_TYPE_TEST_RESULT,
};

/// Current header version.
pub const SHM_RING_VERSION: u32 = 1;

/// Upper bound on a valid ring capacity, in bytes.
///
/// `ShmRingHeader.capacity` is `u32` so a torn init or a guest writing
/// a wild value (e.g. `0xFFFF_FFFF`) can produce a capacity that
/// vastly exceeds the actual mapped SHM region. Using such a value
/// would make the drain path read beyond the mapped window (live: via
/// `mem.read_volatile` past the end of the region; snapshot: via
/// slice indexing into `buf` past `HEADER_SIZE + actual_capacity`),
/// aborting the host under `panic = "abort"`.
///
/// Chosen at 1 GiB — orders of magnitude above any realistic
/// `KTSTR_SHM_SIZE` (tens of KiB to a few MiB for metric / TLV
/// traffic) while leaving `u32::MAX` / the corrupt-pattern region
/// strictly rejected.
pub const MAX_SHM_CAPACITY: u32 = 1 << 30;

/// Byte offset within the SHM region for the host-to-guest dump request flag.
/// Occupies the first byte of the `control_bytes` field in ShmRingHeader (offset 12).
/// Host writes `DUMP_REQ_SYSRQ_D` to request a SysRq-D dump; guest polls
/// this byte, triggers the dump, and clears it back to 0.
pub const DUMP_REQ_OFFSET: usize = 12;

/// Value written to DUMP_REQ_OFFSET to request a SysRq-D dump.
pub const DUMP_REQ_SYSRQ_D: u8 = b'D';

/// Set the cached SHM base pointer and region size. Called from
/// `shm_poll_loop` (spawned by `start_shm_poll`) in the guest init
/// after the /dev/mem mmap succeeds.
pub fn init_shm_ptr(base: *mut u8, size: usize) {
    let _ = SHM_PTR.set(ShmPtr { ptr: base, size });
}

/// Detect whether the current process is running inside a ktstr guest
/// VM, by looking for `KTSTR_SHM_BASE`/`KTSTR_SHM_SIZE` on
/// `/proc/cmdline`.
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
            .and_then(|c| parse_shm_params_from_str(&c))
            .is_some()
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

/// Reject a call to a guest-only entry point when invoked from host
/// context. Returns `true` if the caller may proceed (we're inside a
/// guest VM); `false` after emitting a `tracing::warn!` that names the
/// caller and the message type, so a host-side caller surfaces in the
/// log instead of silently no-op'ing.
///
/// `fn_name` is the calling function's name (e.g. `"write_msg"`) and
/// is interpolated into the log message text.
fn assert_guest_context(fn_name: &str, msg_type: u32) -> bool {
    if !is_guest() {
        tracing::warn!(
            msg_type = msg_type,
            "shm_ring::{fn_name} called from host context; use GuestMem::write_* instead"
        );
        return false;
    }
    true
}

/// Guest-only. Host-side code must use `GuestMem::write_*` instead.
///
/// Write a TLV-framed message to the host through the bulk channel
/// (virtio-console port 1, `/dev/vport0p1`). The frame format is
/// identical to the legacy SHM ring: 16-byte `ShmMessage` header
/// followed by `payload.len()` bytes; the host parses the same byte
/// stream via [`parse_tlv_stream`].
///
/// Backpressure: the kernel's virtio_console TX path (`hvc_push` /
/// `port_fops_write`) blocks the writer until the host's
/// `add_used` rate catches up. There is no drop path on a full ring
/// here — that was the SHM ring's drop semantics; port 1 trades
/// drops for blocking writes. Callers that cannot block (panic hook,
/// signal handlers, anything called from a critical section) MUST
/// use [`write_msg_nonblocking`] (CRASH-only SHM fallback) instead.
///
/// Falls back to [`shm_write_raw`] on the SHM ring when
/// `/dev/vport0p1` is not yet available — the bulk port appears only
/// after the multiport handshake completes inside the kernel
/// virtio_console driver. Early-boot writers (before the port opens)
/// land in SHM and the host's `bulk_drain` merger picks them up via
/// the same TLV parser.
///
/// `assert_guest_context` rejects host-context invocations with a
/// `tracing::warn` so a host-side caller surfaces in the log instead
/// of silently no-op'ing.
pub fn write_msg(msg_type: u32, payload: &[u8]) {
    if !assert_guest_context("write_msg", msg_type) {
        return;
    }
    let _guard = SHM_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if write_to_bulk_port(msg_type, payload) {
        return;
    }
    // Bulk port unavailable — fall back to SHM. Same path as the
    // historical implementation: the host's bulk_drain merges
    // port-1 bytes with the SHM ring drain, so messages that
    // landed here during early boot still reach the test verdict.
    let Ok((ptr, size)) = shm_ptr() else { return };
    // SAFETY: `ptr` points to a `size`-byte mmap region that outlives
    // every guest thread (set once via `OnceLock` during init).
    unsafe { shm_write_raw(ptr, size, msg_type, payload) };
}

/// Guest-only. Host-side code must use `GuestMem::write_*` instead.
///
/// Try to write a TLV message without blocking. Used by the panic
/// hook (`MSG_TYPE_CRASH`) and other critical-section callers that
/// cannot afford to block on virtio TX backpressure.
///
/// Always writes to the SHM ring (NOT port 1): the kernel's
/// virtio_console write path can block on host TX backpressure, and
/// blocking inside a panic hook would deadlock the guest before the
/// crash diagnostic reaches the host. SHM is shared memory with no
/// kernel-side wait, so writes are bounded by `SHM_WRITE_LOCK`
/// contention only.
///
/// Uses `try_lock()` on `SHM_WRITE_LOCK`. If the lock is held (e.g.,
/// the panic occurred on the thread that holds it), silently returns
/// false so the caller can fall back to serial. Returns false (with
/// a tracing warning) if invoked from a host context, and false (no
/// log) if SHM is not yet initialized inside the guest.
pub fn write_msg_nonblocking(msg_type: u32, payload: &[u8]) -> bool {
    if !assert_guest_context("write_msg_nonblocking", msg_type) {
        return false;
    }
    let Ok((ptr, size)) = shm_ptr() else {
        return false;
    };
    // `try_lock` fails if the lock is contended OR poisoned. Both
    // map to "don't block, just drop this message" in the
    // non-blocking path — the caller is invariably an exit/dump
    // hook that cannot afford to wait for the writer.
    let Ok(_guard) = SHM_WRITE_LOCK.try_lock() else {
        return false;
    };
    // SAFETY: same invariant as `write_msg` — `ptr` is a valid
    // `size`-byte mmap region.
    unsafe { shm_write_raw(ptr, size, msg_type, payload) };
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

/// Try to write a TLV-framed message to `/dev/vport0p1`. Returns
/// true when the message was fully written, false when the bulk
/// port is not yet available (caller should fall back to SHM) or
/// the write failed.
///
/// Lazy-open semantics: the multiport handshake completes asynchronously
/// during kernel virtio_console init, so the device node may appear
/// any time after the first `write_msg` call. We retry the open on
/// every call until it succeeds; once cached, subsequent writes go
/// through the cached `File`.
///
/// The header and payload are submitted via `writev` with two
/// `iovec` slices, avoiding a per-call concat allocation. `SHM_WRITE_LOCK`
/// (held by the caller) plus the per-handle `BULK_PORT_FD` mutex
/// serialise writers, so the in-stream order of bytes on port 1 is
/// `[header][payload]` even though the kernel's virtio-console driver
/// only exposes `.write` (not `.write_iter`) and `vfs_writev` therefore
/// loops `port_fops_write` once per iovec. The host's
/// [`super::bulk::HostAssembler`] tolerates partial frames in the byte
/// stream, so the per-iovec virtqueue submissions reassemble correctly.
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

/// Try to open `/dev/vport0p1` for writing. Returns None when the
/// device is not yet present — the kernel virtio_console driver
/// creates it only after the host emits PORT_OPEN on the c_ivq for
/// port 1 and the kernel's `find_port_by_id` resolves the
/// `/sys/class/virtio-ports/vport0p1` entry.
///
/// Open mode: write-only, blocking. The kernel's `port_fops_write`
/// path blocks the writer when the host's `add_used` rate lags;
/// that's the backpressure mechanism we want — it replaces the SHM
/// ring's drop semantics.
fn try_open_bulk_port() -> Option<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/vport0p1")
        .ok()
}

/// Host-side TLV parser for the bulk-port byte stream. The guest
/// emits frames identical to the SHM ring's format: 16-byte
/// `ShmMessage` header (msg_type, length, crc32, _pad) followed by
/// `length` payload bytes. This walks the buffer end-to-end and
/// returns a [`ShmDrainResult`] with one [`ShmEntry`] per complete
/// frame, leaving partial trailing frames untouched. The freeze
/// coordinator pushes its `bulk_assembler`'s residual partial-frame
/// bytes back into the device's `port1_tx_buf` before exit (see
/// [`super::virtio_console::VirtioConsole::push_back_bulk`]); the
/// end-of-run `drain_bulk` here therefore returns those residual
/// bytes plus any bytes the guest wrote after the last mid-run
/// drain, and this walker assembles them in order.
///
/// Per-frame CRC is computed and compared against the guest's
/// stored value. A CRC mismatch sets `crc_ok=false` on that entry
/// but does not break the walk — subsequent frames are still
/// parsed.
///
/// `drops` is always 0: the bulk port has no drop semantics
/// (backpressure replaces them). The field is kept on
/// `ShmDrainResult` so existing host-side consumers compile; SHM
/// drains still report ring-full drops where applicable.
pub fn parse_tlv_stream(buf: &[u8]) -> ShmDrainResult {
    let mut entries = Vec::new();
    let mut pos = 0usize;
    while pos.saturating_add(MSG_HEADER_SIZE) <= buf.len() {
        let hdr_end = pos + MSG_HEADER_SIZE;
        let hdr_slice = &buf[pos..hdr_end];
        let Ok(msg) = ShmMessage::read_from_bytes(hdr_slice) else {
            // Cannot happen — slice is exactly MSG_HEADER_SIZE bytes
            // and ShmMessage is FromBytes. Defensive break.
            break;
        };
        // Reject torn-frame lengths the same way `shm_drain` does:
        // any length larger than the buffer remainder cannot be a
        // complete frame from the writer (`write_all` is atomic per
        // call), so stop parsing rather than allocating a huge vec.
        if (msg.length as usize) > buf.len().saturating_sub(hdr_end) {
            break;
        }
        let payload_end = hdr_end + msg.length as usize;
        let payload = buf[hdr_end..payload_end].to_vec();
        let computed_crc = crc32fast::hash(&payload);
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok: computed_crc == msg.crc32,
        });
        pos = payload_end;
    }
    ShmDrainResult { entries, drops: 0 }
}

/// Wrapper for a raw pointer + size that is Send+Sync.
/// SAFETY: The SHM pointer is set once via OnceLock during init and
/// points into a /dev/mem mmap that outlives all guest threads.
struct ShmPtr {
    ptr: *mut u8,
    size: usize,
}
unsafe impl Send for ShmPtr {}
unsafe impl Sync for ShmPtr {}

/// Cached SHM mmap pointer for guest-side signal operations.
static SHM_PTR: std::sync::OnceLock<ShmPtr> = std::sync::OnceLock::new();

/// Mutex serializing guest-side SHM ring writes. Every guest writer
/// (`write_msg`, `write_msg_nonblocking`) takes this lock before
/// touching `write_ptr`, so the SHM-ring fallback path is safe across
/// the sched-exit-mon thread, the step executor, and the panic hook.
pub static SHM_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Get the cached SHM mmap pointer and size, initializing from
/// /proc/cmdline if not already set.
fn shm_ptr() -> anyhow::Result<(*mut u8, usize)> {
    if let Some(p) = SHM_PTR.get() {
        return Ok((p.ptr, p.size));
    }
    // Lazy init from /proc/cmdline.
    let cmdline = std::fs::read_to_string("/proc/cmdline")
        .map_err(|e| anyhow::anyhow!("/proc/cmdline: {e}"))?;
    let (shm_base, shm_size) = parse_shm_params_from_str(&cmdline)
        .ok_or_else(|| anyhow::anyhow!("no SHM params in cmdline"))?;

    let fd = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/mem")
        .map_err(|e| anyhow::anyhow!("/dev/mem open: {e}"))?;

    let m = mmap_devmem(
        std::os::unix::io::AsRawFd::as_raw_fd(&fd),
        shm_base,
        shm_size,
    )
    .ok_or_else(|| anyhow::anyhow!("/dev/mem mmap failed: {}", std::io::Error::last_os_error()))?;

    let size = shm_size as usize;
    let _ = SHM_PTR.set(ShmPtr { ptr: m.ptr, size });
    Ok((m.ptr, size))
}

/// Parse KTSTR_SHM_BASE and KTSTR_SHM_SIZE from a kernel command line string.
pub(crate) fn parse_shm_params_from_str(cmdline: &str) -> Option<(u64, u64)> {
    let base = cmdline
        .split_whitespace()
        .find(|s| s.starts_with("KTSTR_SHM_BASE="))?
        .strip_prefix("KTSTR_SHM_BASE=")?;
    let size = cmdline
        .split_whitespace()
        .find(|s| s.starts_with("KTSTR_SHM_SIZE="))?
        .strip_prefix("KTSTR_SHM_SIZE=")?;
    let base =
        u64::from_str_radix(base.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
    let size =
        u64::from_str_radix(size.trim_start_matches("0x").trim_start_matches("0X"), 16).ok()?;
    Some((base, size))
}

/// Maximum length, in bytes, of a guest-supplied snapshot tag (name or
/// symbol path) carried through the SHM snapshot request slot. The
/// guest writes a UTF-8 tag, NUL-terminates if shorter than this
/// bound, and truncates to this size if longer (the host treats the
/// first NUL as the boundary, or stops at this size if no NUL is
/// present).
pub const SHM_SNAPSHOT_TAG_MAX: usize = 64;

/// Snapshot request kind: one of [`SHM_SNAPSHOT_KIND_NONE`],
/// [`SHM_SNAPSHOT_KIND_CAPTURE`], or [`SHM_SNAPSHOT_KIND_WATCH`].
/// Written by the guest into [`ShmRingHeader::snapshot_kind`] before
/// firing the doorbell so the host can dispatch the right handler.
pub const SHM_SNAPSHOT_KIND_NONE: u32 = 0;
/// Capture-now request: host runs `freeze_and_capture(false)`,
/// stores the report on the bridge keyed by the snapshot tag, and
/// signals reply.
pub const SHM_SNAPSHOT_KIND_CAPTURE: u32 = 1;
/// Hardware-watchpoint registration request: host resolves the
/// symbol path through BTF + kallsyms, allocates a free DR slot
/// (DR1..=DR3), arms the watchpoint, and signals reply. A future
/// guest write to the resolved KVA fires KVM_EXIT_DEBUG and runs a
/// synchronous capture tagged by the symbol path.
pub const SHM_SNAPSHOT_KIND_WATCH: u32 = 2;

/// Reply status: success — the host completed the requested action
/// (capture stored, or watchpoint armed).
pub const SHM_SNAPSHOT_STATUS_OK: u32 = 1;
/// Reply status: failure — the host rejected or could not complete
/// the request. The reason is delivered via the snapshot reason slot
/// (UTF-8, NUL-terminated, max [`SHM_SNAPSHOT_TAG_MAX`] bytes).
pub const SHM_SNAPSHOT_STATUS_ERR: u32 = 2;

/// Ring buffer header at the start of the SHM region.
///
/// write_ptr and read_ptr are monotonically increasing byte offsets into
/// the data area. Actual position = ptr % capacity.
#[repr(C)]
#[derive(Clone, Copy, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct ShmRingHeader {
    pub magic: u32,
    pub version: u32,
    /// Data area size in bytes (region_size - sizeof(ShmRingHeader)).
    pub capacity: u32,
    /// Packed host→guest control bytes — NOT padding despite being
    /// declared as `u32` for alignment. Byte 0 = `DUMP_REQ_OFFSET`
    /// (SysRq-D dump trigger). Bytes 1-3 are unused; every other
    /// host→guest signal travels via the virtio-console wake-byte
    /// channel rather than packed SHM bytes. Byte 0 is read/written
    /// byte-wise via the `DUMP_REQ_OFFSET` constant above; the `u32`
    /// spelling exists only so the header remains a plain POD for
    /// zerocopy derive.
    pub control_bytes: u32,
    /// Total bytes written by the guest (monotonic).
    pub write_ptr: u64,
    /// Total bytes read by the host (monotonic).
    pub read_ptr: u64,
    /// Number of messages dropped by `shm_write`. Conflates three
    /// distinct failure modes:
    ///   1. ring-full — the common case, used-space + message would
    ///      exceed `capacity`;
    ///   2. total-size overflow — `MSG_HEADER_SIZE + payload.len()`
    ///      overflows `usize` (pathological payload, effectively
    ///      unreachable);
    ///   3. length-field overflow — `payload.len() > u32::MAX` so the
    ///      `ShmMessage.length` field cannot represent it
    ///      (unreachable in the current schema where `capacity: u32`
    ///      already caps payload size at ~4GB).
    ///
    /// Host telemetry readers treat all three as "producer lost a
    /// message"; the ring-full case dominates and the overflow cases
    /// exist only as defense-in-depth (see `shm_write`). Splitting
    /// this into separate counters would add header bytes and
    /// observable bytes for paths that should never fire in practice,
    /// so the single counter is the right tradeoff.
    pub drops: u64,
    /// Snapshot request id — monotonic counter the guest bumps before
    /// each doorbell write. The host stamps the same value into
    /// `snapshot_reply_id` after completing the request so the guest
    /// can pair its wait against the original request without a
    /// dedicated wait queue.
    pub snapshot_request_id: u32,
    /// Request kind. One of [`SHM_SNAPSHOT_KIND_NONE`],
    /// [`SHM_SNAPSHOT_KIND_CAPTURE`], [`SHM_SNAPSHOT_KIND_WATCH`].
    /// Read by the host's freeze coordinator on each doorbell event.
    pub snapshot_kind: u32,
    /// Snapshot reply id — host writes the matching `snapshot_request_id`
    /// here after the request completes. Guest polls this slot and
    /// breaks out of its wait when the value equals the id it sent.
    pub snapshot_reply_id: u32,
    /// Reply status. One of [`SHM_SNAPSHOT_STATUS_OK`] /
    /// [`SHM_SNAPSHOT_STATUS_ERR`]. Valid only after
    /// `snapshot_reply_id == snapshot_request_id`.
    pub snapshot_status: u32,
    /// Snapshot tag — UTF-8, NUL-terminated when shorter than the
    /// fixed buffer; truncated to `SHM_SNAPSHOT_TAG_MAX` bytes when
    /// longer. For [`SHM_SNAPSHOT_KIND_CAPTURE`] the tag is the
    /// snapshot name (key the bridge stores the report under); for
    /// [`SHM_SNAPSHOT_KIND_WATCH`] the tag is the symbol path the
    /// host resolves through BTF + kallsyms.
    pub snapshot_tag: [u8; SHM_SNAPSHOT_TAG_MAX],
    /// Reply reason buffer — when `snapshot_status ==
    /// SHM_SNAPSHOT_STATUS_ERR`, the host writes a UTF-8 reason here
    /// (NUL-terminated when shorter, truncated to
    /// [`SHM_SNAPSHOT_TAG_MAX`] when longer). The guest renders this
    /// as the bail-out message in the `Op::Snapshot` /
    /// `Op::WatchSnapshot` dispatcher.
    pub snapshot_reason: [u8; SHM_SNAPSHOT_TAG_MAX],
}

const _HEADER_SIZE: () =
    assert!(std::mem::size_of::<ShmRingHeader>() == 40 + 16 + SHM_SNAPSHOT_TAG_MAX * 2);

/// Byte offset within the SHM region of `snapshot_request_id`.
pub const SNAPSHOT_REQUEST_ID_OFFSET: usize = 40;
/// Byte offset within the SHM region of `snapshot_kind`.
pub const SNAPSHOT_KIND_OFFSET: usize = 44;
/// Byte offset within the SHM region of `snapshot_reply_id`.
pub const SNAPSHOT_REPLY_ID_OFFSET: usize = 48;
/// Byte offset within the SHM region of `snapshot_status`.
pub const SNAPSHOT_STATUS_OFFSET: usize = 52;
/// Byte offset within the SHM region of `snapshot_tag`.
pub const SNAPSHOT_TAG_OFFSET: usize = 56;
/// Byte offset within the SHM region of `snapshot_reason`.
pub const SNAPSHOT_REASON_OFFSET: usize = SNAPSHOT_TAG_OFFSET + SHM_SNAPSHOT_TAG_MAX;

impl Default for ShmRingHeader {
    /// Manual `Default` because `[u8; 64]` does not auto-derive
    /// `Default` for arrays larger than 32 elements on stable Rust.
    /// Pinned: every numeric field zeroes; both byte buffers zero-fill.
    fn default() -> Self {
        Self {
            magic: 0,
            version: 0,
            capacity: 0,
            control_bytes: 0,
            write_ptr: 0,
            read_ptr: 0,
            drops: 0,
            snapshot_request_id: 0,
            snapshot_kind: SHM_SNAPSHOT_KIND_NONE,
            snapshot_reply_id: 0,
            snapshot_status: 0,
            snapshot_tag: [0; SHM_SNAPSHOT_TAG_MAX],
            snapshot_reason: [0; SHM_SNAPSHOT_TAG_MAX],
        }
    }
}

impl ShmRingHeader {
    /// Build a fresh ring header for an SHM region of `shm_size` bytes.
    /// Saturating subtraction on `shm_size - HEADER_SIZE` means a mis-
    /// sized region (`shm_size < HEADER_SIZE`) surfaces as a zero-
    /// capacity ring rather than a panic — every `shm_write` then hits
    /// the ring-full branch and the operator sees the empty data area
    /// instead of losing the VMM to an arithmetic underflow before the
    /// layout error can surface.
    ///
    /// Single source of truth for magic/version/capacity field
    /// population: production `VmmState::init_shm_region`
    /// (src/vmm/mod.rs) and the `#[cfg(test)] shm_init` helper both
    /// call this, so a schema edit lands once.
    pub fn new(shm_size: usize) -> Self {
        let capacity = shm_size.saturating_sub(HEADER_SIZE);
        Self {
            magic: SHM_RING_MAGIC,
            version: SHM_RING_VERSION,
            capacity: capacity as u32,
            control_bytes: 0,
            write_ptr: 0,
            read_ptr: 0,
            drops: 0,
            snapshot_request_id: 0,
            snapshot_kind: SHM_SNAPSHOT_KIND_NONE,
            snapshot_reply_id: 0,
            snapshot_status: 0,
            snapshot_tag: [0; SHM_SNAPSHOT_TAG_MAX],
            snapshot_reason: [0; SHM_SNAPSHOT_TAG_MAX],
        }
    }
}

// `ShmMessage` and `ShmEntry` live in [`super::wire`]. The SHM ring
// uses the same TLV header layout as the virtio-console bulk
// stream, so a single source of truth keeps both transports in
// sync. Re-export under the legacy `shm_ring::` paths.
pub use super::wire::{ShmEntry, ShmMessage};

/// Size of the ShmRingHeader.
pub const HEADER_SIZE: usize = std::mem::size_of::<ShmRingHeader>();
/// Size of the ShmMessage TLV header.
pub const MSG_HEADER_SIZE: usize = std::mem::size_of::<ShmMessage>();

/// Result of draining the ring buffer.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ShmDrainResult {
    pub entries: Vec<ShmEntry>,
    pub drops: u64,
}

/// Payload for stimulus events written by the guest step executor.
///
/// Compact 24-byte struct describing the state after each step's ops
/// are applied. The host correlates these with monitor samples to map
/// scheduler telemetry to scenario phases.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout)]
pub struct StimulusPayload {
    /// Milliseconds since scenario start.
    pub elapsed_ms: u32,
    /// Index of the step that was just applied.
    pub step_index: u16,
    /// Number of ops applied in this step.
    pub op_count: u16,
    /// Bitmask of Op variant discriminants present in this step.
    pub op_kinds: u32,
    /// Number of live cgroups after this step: sum of step-local
    /// cgroups (from the current Step's `CgroupDef`s + `Op`s) and
    /// Backdrop-owned cgroups that persist across every Step.
    pub cgroup_count: u16,
    /// Total worker handles after this step: sum of step-local
    /// workers and Backdrop-spawned workers that persist across
    /// every Step.
    pub worker_count: u16,
    /// Sum of all workers' iteration counts at this step boundary.
    /// Read from shared MAP_SHARED counters in the step executor.
    pub total_iterations: u64,
}

const _STIMULUS_SIZE: () = assert!(std::mem::size_of::<StimulusPayload>() == 24);

/// Deserialized stimulus event from the SHM ring.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StimulusEvent {
    pub elapsed_ms: u32,
    pub step_index: u16,
    pub op_count: u16,
    pub op_kinds: u32,
    pub cgroup_count: u16,
    pub worker_count: u16,
    pub total_iterations: u64,
}

impl StimulusEvent {
    /// Deserialize from raw payload bytes.
    pub fn from_payload(data: &[u8]) -> Option<Self> {
        if data.len() < std::mem::size_of::<StimulusPayload>() {
            return None;
        }
        Some(StimulusEvent {
            elapsed_ms: u32::from_ne_bytes(data[0..4].try_into().ok()?),
            step_index: u16::from_ne_bytes(data[4..6].try_into().ok()?),
            op_count: u16::from_ne_bytes(data[6..8].try_into().ok()?),
            op_kinds: u32::from_ne_bytes(data[8..12].try_into().ok()?),
            cgroup_count: u16::from_ne_bytes(data[12..14].try_into().ok()?),
            worker_count: u16::from_ne_bytes(data[14..16].try_into().ok()?),
            total_iterations: u64::from_ne_bytes(data[16..24].try_into().ok()?),
        })
    }
}

/// Test helper — initialize the SHM ring header at the given
/// offset in guest memory.
///
/// `buf` is the full guest memory slice. `shm_offset` is the byte offset
/// where the SHM region starts. `shm_size` is the total region size.
///
/// The header is built via [`ShmRingHeader::new`], the same constructor
/// production `VmmState::init_shm_region` uses — a schema change lands
/// once and both sites pick it up.
#[cfg(test)]
pub fn shm_init(buf: &mut [u8], shm_offset: usize, shm_size: usize) {
    let header = ShmRingHeader::new(shm_size);
    let hdr_bytes = header.as_bytes();
    buf[shm_offset..shm_offset + HEADER_SIZE].copy_from_slice(hdr_bytes);
    // Zero the data area.
    let data_start = shm_offset + HEADER_SIZE;
    let data_end = shm_offset + shm_size;
    buf[data_start..data_end].fill(0);
}

/// Read the ring header from guest memory via zerocopy.
///
/// `ShmRingHeader` derives `FromBytes` + `Immutable` + `KnownLayout`,
/// so any byte slice sized exactly `HEADER_SIZE` is a valid header —
/// all fields are fixed-width scalars with no invalid bit patterns.
/// `read_from_bytes` returns a `Result<Self>` that only fails on
/// size mismatch; the slice is always exactly `HEADER_SIZE` bytes by
/// construction, so `.unwrap()` is justified.
fn read_header(buf: &[u8], shm_offset: usize) -> ShmRingHeader {
    let s = &buf[shm_offset..shm_offset + HEADER_SIZE];
    ShmRingHeader::read_from_bytes(s).expect("HEADER_SIZE matches ShmRingHeader layout")
}

/// Read `len` bytes from the ring's data area starting at monotonic offset
/// `ptr`, handling wraparound.
fn read_ring_bytes(
    buf: &[u8],
    data_start: usize,
    capacity: usize,
    ptr: u64,
    len: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; len];
    read_ring_into(buf, data_start, capacity, ptr, &mut out);
    out
}

/// Read `len` bytes from the ring into an existing buffer, handling wraparound.
fn read_ring_into(buf: &[u8], data_start: usize, capacity: usize, ptr: u64, out: &mut [u8]) {
    let len = out.len();
    let mut remaining = len;
    let mut src_pos = (ptr % capacity as u64) as usize;
    let mut dst_pos = 0;
    while remaining > 0 {
        let chunk = remaining.min(capacity - src_pos);
        out[dst_pos..dst_pos + chunk]
            .copy_from_slice(&buf[data_start + src_pos..data_start + src_pos + chunk]);
        dst_pos += chunk;
        src_pos = 0; // wrap
        remaining -= chunk;
    }
}

/// Drain all complete messages from the ring buffer.
///
/// `buf` is the full guest memory (read-only). `shm_offset` is the byte
/// offset where the SHM region starts.
pub fn shm_drain(buf: &[u8], shm_offset: usize) -> ShmDrainResult {
    // A misconfigured caller (e.g., an `shm_size` smaller than
    // `HEADER_SIZE` propagating into `collect_results`) must not panic
    // the host: `read_header` indexes `buf[shm_offset..shm_offset +
    // HEADER_SIZE]` unconditionally and would slice-panic on a
    // too-small buffer. Bail with an empty drain instead.
    if shm_offset.saturating_add(HEADER_SIZE) > buf.len() {
        return ShmDrainResult::default();
    }
    let header = read_header(buf, shm_offset);
    if header.magic != SHM_RING_MAGIC {
        return ShmDrainResult::default();
    }

    // Guest-supplied `capacity` must be a plausible ring size. Zero
    // would divide-by-zero inside `read_ring_into`'s `ptr % capacity`;
    // values above `MAX_SHM_CAPACITY` indicate torn init or
    // corruption and would indexing-panic on `buf` once the walk
    // advances past the real region. Either case: treat as
    // uninitialized and bail out cleanly.
    if header.capacity == 0 || header.capacity > MAX_SHM_CAPACITY {
        return ShmDrainResult::default();
    }

    let capacity = header.capacity as usize;
    let data_start = shm_offset + HEADER_SIZE;
    let mut read_pos = header.read_ptr;
    let write_pos = header.write_ptr;
    let mut entries = Vec::new();
    // Ring invariant: at most `capacity` bytes are unread at any
    // moment. A torn/corrupt header where `read_ptr > write_ptr`
    // (or where the gap exceeds capacity for any reason) makes
    // `wrapping_sub` return a near-`u64::MAX` distance — without
    // this guard the loop would iterate ~2^60 times before
    // termination, OOMing the host long before that.
    //
    // Mirrors the writer-side invariant check in `shm_write` /
    // `shm_write_raw` (`if used > capacity { return; }`).
    if write_pos.wrapping_sub(read_pos) > capacity as u64 {
        return ShmDrainResult {
            entries,
            drops: header.drops,
        };
    }
    // Largest payload that can fit in a valid message: any
    // `msg.length` larger than this is necessarily torn/corrupt and
    // must not be trusted as an allocation size. Without this cap a
    // guest-controlled `u32` (up to 4 GiB) could trigger an OOM via
    // `vec![0u8; msg.length as usize]` in `read_ring_bytes`.
    let max_payload = (capacity - MSG_HEADER_SIZE.min(capacity)) as u64;

    // Modular distance: `write_pos.wrapping_sub(read_pos)` handles
    // both the normal case and the (extremely rare) u64 overflow of
    // `write_pos` ahead of `read_pos`. Raw addition would overflow
    // when `read_pos` is near `u64::MAX`.
    while write_pos.wrapping_sub(read_pos) >= MSG_HEADER_SIZE as u64 {
        let mut hdr_buf = [0u8; MSG_HEADER_SIZE];
        read_ring_into(buf, data_start, capacity, read_pos, &mut hdr_buf);
        let msg = ShmMessage::read_from_bytes(&hdr_buf)
            .expect("MSG_HEADER_SIZE matches ShmMessage layout");

        // Reject implausible message lengths before allocating: a
        // length larger than the entire ring's payload area cannot
        // come from a complete, non-torn write. Stop draining rather
        // than allocating multi-GiB or chasing further corrupt
        // entries.
        if msg.length as u64 > max_payload {
            break;
        }

        let total_msg_size = MSG_HEADER_SIZE as u64 + msg.length as u64;
        if write_pos.wrapping_sub(read_pos) < total_msg_size {
            // Incomplete message — stop.
            break;
        }

        let payload = read_ring_bytes(
            buf,
            data_start,
            capacity,
            read_pos.wrapping_add(MSG_HEADER_SIZE as u64),
            msg.length as usize,
        );

        let computed_crc = crc32fast::hash(&payload);
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok: computed_crc == msg.crc32,
        });

        read_pos = read_pos.wrapping_add(total_msg_size);
    }

    ShmDrainResult {
        entries,
        drops: header.drops,
    }
}

/// Drain messages from the SHM ring while the guest VM is running.
///
/// Unlike `shm_drain` (which operates on a post-mortem snapshot),
/// this reads from live guest memory via volatile pointers and writes
/// `read_ptr` back so the guest can reclaim ring space.
///
/// `mem` provides volatile access to guest DRAM.
/// `shm_base_pa` is the DRAM-relative offset of the SHM region.
///
/// Returns drained entries. Call periodically (~10ms) from the monitor
/// thread to prevent ring overflow during long scenarios.
pub fn shm_drain_live(mem: &crate::monitor::reader::GuestMem, shm_base_pa: u64) -> ShmDrainResult {
    let magic = mem.read_u32(shm_base_pa, 0);
    if magic != SHM_RING_MAGIC {
        return ShmDrainResult::default();
    }

    let capacity_raw = mem.read_u32(shm_base_pa, 8);
    // `capacity` is guest-supplied and not covered by the magic gate
    // above — a torn init (guest writes magic before capacity) or
    // corruption can leave capacity at zero while magic reads valid.
    // `read_ring_volatile` would then compute `ptr % capacity as u64`
    // and divide by zero, panicking the monitor thread and — under
    // `panic = "abort"` — aborting the entire host. A wild value like
    // `0xFFFF_FFFF` is just as damaging: `read_ring_volatile` would
    // then read_volatile past the end of the mapped SHM region. Reject
    // both ends: treat as uninitialized and let the next tick re-read.
    if capacity_raw == 0 || capacity_raw > MAX_SHM_CAPACITY {
        return ShmDrainResult::default();
    }
    let capacity = capacity_raw as usize;
    let write_ptr = mem.read_u64(shm_base_pa, 16);
    let read_ptr = mem.read_u64(shm_base_pa, 24);
    let drops = mem.read_u64(shm_base_pa, 32);

    let data_start_pa = shm_base_pa + HEADER_SIZE as u64;
    let mut read_pos = read_ptr;
    let mut entries = Vec::new();
    // Ring invariant: at most `capacity` bytes are unread. A
    // torn/corrupt header where `read_ptr > write_ptr` (or where
    // the gap exceeds capacity) makes `wrapping_sub` return a
    // near-`u64::MAX` distance — without this guard the loop
    // would iterate ~2^60 times before terminating and OOM the
    // host monitor thread.
    //
    // Bail without advancing `read_ptr` so the next tick re-reads
    // the (hopefully recovered) header. Mirrors the writer-side
    // invariant check in `shm_write` / `shm_write_raw`.
    if write_ptr.wrapping_sub(read_pos) > capacity as u64 {
        return ShmDrainResult { entries, drops };
    }
    // Same OOM cap as `shm_drain`: a guest-supplied `msg.length`
    // larger than the ring's payload area can never come from a
    // complete write, and trusting it would let an untrusted guest
    // request a multi-GiB host allocation.
    let max_payload = (capacity - MSG_HEADER_SIZE.min(capacity)) as u64;

    // Modular distance via `wrapping_sub` — `read_pos + ...` would
    // overflow when `read_pos` is near `u64::MAX` (post-wraparound
    // case validated by `shm_write_wrapping_sub_handles_u64_overflow_of_write_ptr`).
    while write_ptr.wrapping_sub(read_pos) >= MSG_HEADER_SIZE as u64 {
        // Read message header via volatile.
        let mut hdr_buf = [0u8; MSG_HEADER_SIZE];
        read_ring_volatile(mem, data_start_pa, capacity, read_pos, &mut hdr_buf);
        let msg = ShmMessage::read_from_bytes(&hdr_buf)
            .expect("MSG_HEADER_SIZE matches ShmMessage layout");

        // Reject implausible `msg.length` before allocating: prevents
        // a torn-write or guest-controlled u32 (up to ~4GiB) from
        // triggering an OOM via `vec![0u8; msg.length as usize]`.
        if msg.length as u64 > max_payload {
            break;
        }

        let total_msg_size = MSG_HEADER_SIZE as u64 + msg.length as u64;
        if write_ptr.wrapping_sub(read_pos) < total_msg_size {
            break;
        }

        let mut payload = vec![0u8; msg.length as usize];
        if !payload.is_empty() {
            read_ring_volatile(
                mem,
                data_start_pa,
                capacity,
                read_pos.wrapping_add(MSG_HEADER_SIZE as u64),
                &mut payload,
            );
        }

        let computed_crc = crc32fast::hash(&payload);
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok: computed_crc == msg.crc32,
        });

        read_pos = read_pos.wrapping_add(total_msg_size);
    }

    // Advance read_ptr so the guest can reuse the drained space.
    if read_pos != read_ptr {
        mem.write_u64(shm_base_pa, 24, read_pos);
    }

    ShmDrainResult { entries, drops }
}

/// Read `out.len()` bytes from the ring data area via volatile reads,
/// handling wraparound. Uses byte-by-byte volatile reads since the
/// data area is in guest memory that the guest may be writing to.
fn read_ring_volatile(
    mem: &crate::monitor::reader::GuestMem,
    data_start_pa: u64,
    capacity: usize,
    ptr: u64,
    out: &mut [u8],
) {
    let mut remaining = out.len();
    let mut src_pos = (ptr % capacity as u64) as usize;
    let mut dst_pos = 0;
    while remaining > 0 {
        let chunk = remaining.min(capacity - src_pos);
        for i in 0..chunk {
            let pa = data_start_pa + (src_pos + i) as u64;
            out[dst_pos + i] = mem.read_u8(pa, 0);
        }
        dst_pos += chunk;
        src_pos = 0; // wrap
        remaining -= chunk;
    }
}

// ---------------------------------------------------------------------------
// Helper: write a message into the ring (for testing / guest-side simulation)
// ---------------------------------------------------------------------------

/// Write a TLV message into the ring buffer. Returns the number of bytes
/// written (MSG_HEADER_SIZE + payload.len()), or 0 if the ring is full
/// (and increments the drops counter).
///
/// This is the guest-side write operation, used in tests to simulate a
/// producer.
#[allow(dead_code)]
pub fn shm_write(buf: &mut [u8], shm_offset: usize, msg_type: u32, payload: &[u8]) -> usize {
    let header = read_header(buf, shm_offset);
    let capacity = header.capacity as usize;
    let Some(total) = MSG_HEADER_SIZE.checked_add(payload.len()) else {
        // Pathological payload whose size overflows with the header
        // prefix: treat as ring-full so the drops counter reflects the
        // lost message, matching the capacity-overflow path below.
        let drops_offset = shm_offset + 32;
        let current = u64::from_ne_bytes(buf[drops_offset..drops_offset + 8].try_into().unwrap());
        buf[drops_offset..drops_offset + 8]
            .copy_from_slice(&current.saturating_add(1).to_ne_bytes());
        return 0;
    };

    // Available space: capacity - (write_ptr - read_ptr). Both ptrs
    // are monotonic u64 counters; `wrapping_sub` is the semantically
    // correct distance under modular arithmetic and handles the
    // (extremely rare) u64 overflow of write_ptr ahead of read_ptr.
    //
    // If the distance exceeds capacity, the ring invariant is
    // violated — torn memory, corruption, or a bug elsewhere. Log
    // and drop the message rather than returning a meaningless value.
    let used = header.write_ptr.wrapping_sub(header.read_ptr) as usize;
    if used > capacity {
        tracing::warn!(
            write_ptr = header.write_ptr,
            read_ptr = header.read_ptr,
            capacity = capacity,
            used = used,
            "shm_ring: used > capacity; ring invariant violated (torn memory?)"
        );
        return 0;
    }
    // `checked_add` guards against overflow on a pathological payload
    // (MSG_HEADER_SIZE + usize::MAX). Treat overflow as ring-full.
    let needed = used.checked_add(total);
    if needed.is_none_or(|n| n > capacity) {
        // Ring full — increment drops counter. `saturating_add` because
        // a pinned-at-u64::MAX counter is the right observable state
        // when drops overflow; a wraparound to 0 would masquerade as
        // "no drops" to the host telemetry reader.
        let drops_offset = shm_offset + 32; // offset of `drops` field
        let current = u64::from_ne_bytes(buf[drops_offset..drops_offset + 8].try_into().unwrap());
        buf[drops_offset..drops_offset + 8]
            .copy_from_slice(&current.saturating_add(1).to_ne_bytes());
        return 0;
    }

    let data_start = shm_offset + HEADER_SIZE;

    // Write message header. `ShmMessage.length` is `u32`; a payload
    // whose length exceeds u32::MAX cannot be faithfully represented
    // in the header, so drop it rather than silently truncating and
    // producing a header whose CRC+length mismatch would either
    // crash the reader or cause it to skip downstream messages.
    //
    // Defense-in-depth: in the current schema `capacity: u32` (see
    // `ShmHeader`) makes this branch unreachable — the `needed >
    // capacity` check above already rejects payloads larger than ~4GB
    // well before the u32 conversion here could fail. Kept so that a
    // future refactor widening `capacity` to `u64` cannot silently
    // produce a torn header with a truncated length field.
    let Ok(length_u32) = u32::try_from(payload.len()) else {
        let drops_offset = shm_offset + 32;
        let current = u64::from_ne_bytes(buf[drops_offset..drops_offset + 8].try_into().unwrap());
        buf[drops_offset..drops_offset + 8]
            .copy_from_slice(&current.saturating_add(1).to_ne_bytes());
        return 0;
    };
    let msg = ShmMessage {
        msg_type,
        length: length_u32,
        crc32: crc32fast::hash(payload),
        _pad: 0,
    };
    write_ring_bytes(buf, data_start, capacity, header.write_ptr, msg.as_bytes());

    // Write payload
    if !payload.is_empty() {
        write_ring_bytes(
            buf,
            data_start,
            capacity,
            header.write_ptr + MSG_HEADER_SIZE as u64,
            payload,
        );
    }

    // Update write_ptr
    let new_write = header.write_ptr + total as u64;
    let wp_offset = shm_offset + 16; // offset of `write_ptr` field
    buf[wp_offset..wp_offset + 8].copy_from_slice(&new_write.to_ne_bytes());

    total
}

/// Write bytes into the ring's data area at monotonic offset `ptr`,
/// handling wraparound.
#[allow(dead_code)]
fn write_ring_bytes(buf: &mut [u8], data_start: usize, capacity: usize, ptr: u64, data: &[u8]) {
    let mut remaining = data.len();
    let mut src_pos = 0;
    let mut dst_pos = (ptr % capacity as u64) as usize;
    while remaining > 0 {
        let chunk = remaining.min(capacity - dst_pos);
        buf[data_start + dst_pos..data_start + dst_pos + chunk]
            .copy_from_slice(&data[src_pos..src_pos + chunk]);
        src_pos += chunk;
        dst_pos = 0; // wrap
        remaining -= chunk;
    }
}

/// Raw-pointer mirror of [`shm_write`] used by the guest production
/// writers (`write_msg`, `write_msg_nonblocking`). Operates entirely
/// through `ptr::read_volatile` / `ptr::write_volatile` so the SHM
/// region is never materialized as `&mut [u8]` — the host monitor
/// thread reads the same memory concurrently via `shm_drain_live`,
/// so a `&mut` slice would alias the host's view and violate Rust's
/// reference rules even though `SHM_WRITE_LOCK` serializes guest-side
/// writers.
///
/// SAFETY: caller must ensure `base` points to a valid `size`-byte
/// mapping that outlives the call, and that no other code holds a
/// `&` or `&mut` reference to bytes in that mapping.
#[allow(dead_code)]
unsafe fn shm_write_raw(base: *mut u8, size: usize, msg_type: u32, payload: &[u8]) {
    if size < HEADER_SIZE {
        return;
    }
    // Read the current header field-by-field via volatile loads.
    // Order mirrors `ShmRingHeader`: magic (0), version (4),
    // capacity (8), control_bytes (12), write_ptr (16), read_ptr (24),
    // drops (32). Only `capacity`, `write_ptr`, `read_ptr`, and
    // `drops` are needed by the write path.
    let capacity = unsafe { ptr::read_volatile(base.add(8) as *const u32) } as usize;
    if capacity == 0 || capacity > size - HEADER_SIZE {
        return;
    }
    let write_ptr = unsafe { ptr::read_volatile(base.add(16) as *const u64) };
    let read_ptr = unsafe { ptr::read_volatile(base.add(24) as *const u64) };

    let bump_drops = || {
        // SAFETY: caller invariant; `drops` field at offset 32 lies
        // wholly within the `size`-byte mapping because
        // `size >= HEADER_SIZE` was checked above.
        let drops = unsafe { ptr::read_volatile(base.add(32) as *const u64) };
        unsafe { ptr::write_volatile(base.add(32) as *mut u64, drops.saturating_add(1)) };
    };

    let Some(total) = MSG_HEADER_SIZE.checked_add(payload.len()) else {
        bump_drops();
        return;
    };

    let used = write_ptr.wrapping_sub(read_ptr) as usize;
    if used > capacity {
        return;
    }
    let needed = used.checked_add(total);
    if needed.is_none_or(|n| n > capacity) {
        bump_drops();
        return;
    }

    let Ok(length_u32) = u32::try_from(payload.len()) else {
        bump_drops();
        return;
    };

    let msg = ShmMessage {
        msg_type,
        length: length_u32,
        crc32: crc32fast::hash(payload),
        _pad: 0,
    };
    let msg_bytes = msg.as_bytes();

    // Data area starts immediately after the header.
    let data_base = unsafe { base.add(HEADER_SIZE) };

    // Write the TLV header bytes into the ring data area.
    // SAFETY: caller invariant + all bounds derived from `capacity`
    // which we just verified is `<= size - HEADER_SIZE`.
    unsafe {
        write_ring_volatile(data_base, capacity, write_ptr, msg_bytes);
        if !payload.is_empty() {
            write_ring_volatile(
                data_base,
                capacity,
                write_ptr.wrapping_add(MSG_HEADER_SIZE as u64),
                payload,
            );
        }
    }

    // Publish: bump write_ptr last so a concurrent host reader never
    // observes a partially written message past the previous
    // `write_ptr`. `wrapping_add` matches the modular distance used
    // throughout the drain path (`shm_drain` / `shm_drain_live`),
    // which already tolerates `write_ptr` wrapping past `u64::MAX`.
    let new_write = write_ptr.wrapping_add(total as u64);
    unsafe { ptr::write_volatile(base.add(16) as *mut u64, new_write) };
}

/// Volatile byte-by-byte write of `data` into the ring's data area
/// starting at monotonic offset `ptr`, handling wraparound. Mirror
/// of [`read_ring_volatile`] for guest-side writers.
///
/// SAFETY: `data_base` must point to a `capacity`-byte ring data area
/// that lives for the duration of the call.
#[allow(dead_code)]
unsafe fn write_ring_volatile(data_base: *mut u8, capacity: usize, ptr: u64, data: &[u8]) {
    let mut remaining = data.len();
    let mut src_pos = 0usize;
    let mut dst_pos = (ptr % capacity as u64) as usize;
    while remaining > 0 {
        let chunk = remaining.min(capacity - dst_pos);
        for i in 0..chunk {
            // SAFETY: `dst_pos + i < capacity` because chunk is bounded
            // by `capacity - dst_pos`; caller guarantees `data_base`
            // points to a `capacity`-byte mapping.
            unsafe {
                ptr::write_volatile(data_base.add(dst_pos + i), data[src_pos + i]);
            }
        }
        src_pos += chunk;
        remaining -= chunk;
        dst_pos = 0; // wrap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time size assertions (also present above, but explicit tests
    // for visibility in test output).
    #[test]
    fn header_size_matches_field_offsets() {
        // Pre-snapshot fields: 40 bytes (magic, version, capacity,
        // control_bytes, write_ptr, read_ptr, drops). Snapshot
        // request slot adds: request_id+kind+reply_id+status (4×u32
        // = 16) + tag + reason (2 × SHM_SNAPSHOT_TAG_MAX bytes).
        let expected = 40 + 16 + SHM_SNAPSHOT_TAG_MAX * 2;
        assert_eq!(std::mem::size_of::<ShmRingHeader>(), expected);
        assert_eq!(HEADER_SIZE, expected);
    }

    #[test]
    fn snapshot_offsets_within_header_and_distinct() {
        // The request slot fields must occupy distinct, contiguous,
        // in-bounds byte ranges. A regression that shifted any of
        // them would alias with the legacy ring fields (capacity at
        // 8, write_ptr at 16, read_ptr at 24, drops at 32) or with
        // each other, silently corrupting both halves.
        const _: () = assert!(SNAPSHOT_REQUEST_ID_OFFSET >= 40);
        assert_eq!(SNAPSHOT_KIND_OFFSET, SNAPSHOT_REQUEST_ID_OFFSET + 4);
        assert_eq!(SNAPSHOT_REPLY_ID_OFFSET, SNAPSHOT_KIND_OFFSET + 4);
        assert_eq!(SNAPSHOT_STATUS_OFFSET, SNAPSHOT_REPLY_ID_OFFSET + 4);
        assert_eq!(SNAPSHOT_TAG_OFFSET, SNAPSHOT_STATUS_OFFSET + 4);
        assert_eq!(
            SNAPSHOT_REASON_OFFSET,
            SNAPSHOT_TAG_OFFSET + SHM_SNAPSHOT_TAG_MAX
        );
        assert_eq!(
            SNAPSHOT_REASON_OFFSET + SHM_SNAPSHOT_TAG_MAX,
            std::mem::size_of::<ShmRingHeader>()
        );
    }

    #[test]
    fn message_size_is_16() {
        assert_eq!(std::mem::size_of::<ShmMessage>(), 16);
    }

    // ---- ShmRingHeader::new saturating-sub boundaries ---------------
    //
    // `new` subtracts `HEADER_SIZE` from `shm_size` via `saturating_sub`
    // so a mis-sized region surfaces as a zero-capacity ring rather than
    // panicking the VMM on integer underflow before the layout error can
    // surface. The four fixtures pin the three "too small" shapes and
    // the "just enough for a non-empty data area" shape against the
    // HEADER_SIZE boundary.

    #[test]
    fn shm_ring_header_new_zero_input_clamps_to_zero_capacity() {
        assert_eq!(ShmRingHeader::new(0).capacity, 0);
    }

    #[test]
    fn shm_ring_header_new_below_header_size_clamps_to_zero_capacity() {
        // HEADER_SIZE - 1: saturating_sub returns 0, no underflow.
        assert_eq!(ShmRingHeader::new(HEADER_SIZE - 1).capacity, 0);
    }

    #[test]
    fn shm_ring_header_new_exactly_header_size_yields_zero_capacity() {
        // HEADER_SIZE exactly: data area is empty but the header is
        // well-formed. `shm_write` hits the ring-full branch on every
        // call — internally consistent, not a panic.
        assert_eq!(ShmRingHeader::new(HEADER_SIZE).capacity, 0);
    }

    #[test]
    fn shm_ring_header_new_above_header_size_carries_delta_as_capacity() {
        // HEADER_SIZE + 100: 100 bytes of data area is addressable.
        assert_eq!(ShmRingHeader::new(HEADER_SIZE + 100).capacity, 100);
    }

    /// Allocate a buffer and initialize a ring of the given size.
    fn make_ring(shm_size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        buf
    }

    #[test]
    fn init_sets_magic_and_capacity() {
        let buf = make_ring(1024);
        let hdr = read_header(&buf, 0);
        assert_eq!(hdr.magic, SHM_RING_MAGIC);
        assert_eq!(hdr.version, SHM_RING_VERSION);
        assert_eq!(hdr.capacity, (1024 - HEADER_SIZE) as u32);
        assert_eq!(hdr.write_ptr, 0);
        assert_eq!(hdr.read_ptr, 0);
        assert_eq!(hdr.drops, 0);
    }

    #[test]
    fn shm_write_rejects_torn_header_read_ptr_past_write_ptr() {
        // If the shared header is torn (or corrupt) such that
        // read_ptr > write_ptr, `wrapping_sub` yields a huge "used"
        // value. The capacity check must detect the invariant
        // violation and return 0 rather than dropping via the
        // ordinary "ring full" path or silently corrupting state.
        let mut buf = make_ring(1024);
        // write_ptr is at shm_offset + 16; read_ptr is at shm_offset + 24.
        // Force read_ptr > write_ptr to simulate the torn state.
        let wp_offset = 16;
        let rp_offset = 24;
        buf[wp_offset..wp_offset + 8].copy_from_slice(&0u64.to_ne_bytes());
        buf[rp_offset..rp_offset + 8].copy_from_slice(&100u64.to_ne_bytes());

        let result = shm_write(&mut buf, 0, 1, b"probe");
        assert_eq!(
            result, 0,
            "torn header (read_ptr > write_ptr) must return 0, got {result}"
        );
    }

    #[test]
    fn shm_write_wrapping_sub_handles_u64_overflow_of_write_ptr() {
        // When the monotonic write_ptr overflows u64 (extremely rare
        // but theoretically possible over long runs), `wrapping_sub`
        // gives the correct modular distance while raw subtraction
        // would underflow. Set write_ptr just past wrap (= 10) and
        // read_ptr just before wrap (= u64::MAX - 5). Used distance
        // = 16 via wrapping_sub.
        //
        // Ring is sized generously so `used + total_msg <= capacity`
        // and the write should succeed.
        let mut buf = make_ring(4096);
        let wp_offset = 16;
        let rp_offset = 24;
        let new_write_ptr: u64 = 10;
        let new_read_ptr: u64 = u64::MAX - 5;
        buf[wp_offset..wp_offset + 8].copy_from_slice(&new_write_ptr.to_ne_bytes());
        buf[rp_offset..rp_offset + 8].copy_from_slice(&new_read_ptr.to_ne_bytes());

        // Sanity: wrapping_sub gives 16, well below capacity 4096-40.
        assert_eq!(new_write_ptr.wrapping_sub(new_read_ptr), 16);

        let result = shm_write(&mut buf, 0, 1, b"probe");
        assert!(
            result > 0,
            "post-wraparound write should succeed, got {result}"
        );
    }

    /// A torn-init or corrupt guest can leave `capacity = 0` with a
    /// valid magic. Drain must not divide-by-zero under `ptr %
    /// capacity` inside the ring walk; return an empty drain instead.
    #[test]
    fn drain_rejects_zero_capacity() {
        let mut buf = make_ring(1024);
        // Overwrite capacity (u32 at offset 8) with 0.
        buf[8..12].copy_from_slice(&0u32.to_ne_bytes());
        // Set a non-zero write_ptr so the walk would otherwise start.
        buf[16..24].copy_from_slice(&64u64.to_ne_bytes());

        let result = shm_drain(&buf, 0);
        assert!(
            result.entries.is_empty(),
            "capacity=0 must bail out, got {} entries",
            result.entries.len(),
        );
    }

    /// A wild `capacity` (e.g. `0xFFFF_FFFF` from torn init) must be
    /// rejected, not trusted. Trusting it would index past the end of
    /// `buf` in `read_ring_into` once the walk advances, panicking the
    /// host.
    #[test]
    fn drain_rejects_oversized_capacity() {
        let mut buf = make_ring(1024);
        buf[8..12].copy_from_slice(&u32::MAX.to_ne_bytes());
        buf[16..24].copy_from_slice(&64u64.to_ne_bytes());

        let result = shm_drain(&buf, 0);
        assert!(
            result.entries.is_empty(),
            "capacity>MAX_SHM_CAPACITY must bail out, got {} entries",
            result.entries.len(),
        );
    }

    /// Boundary: a capacity exactly equal to `MAX_SHM_CAPACITY + 1`
    /// must be rejected, while the allocation-realistic small cases
    /// already covered by the other tests continue to work.
    #[test]
    fn drain_rejects_capacity_one_past_max() {
        let mut buf = make_ring(1024);
        let over = (MAX_SHM_CAPACITY as u64) + 1;
        // `as u32` truncates but MAX_SHM_CAPACITY+1 < u32::MAX so it
        // is representable directly.
        let over_u32 = over as u32;
        buf[8..12].copy_from_slice(&over_u32.to_ne_bytes());
        buf[16..24].copy_from_slice(&64u64.to_ne_bytes());

        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn drain_empty_ring() {
        let buf = make_ring(1024);
        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());
        assert_eq!(result.drops, 0);
    }

    #[test]
    fn drain_bad_magic() {
        let mut buf = vec![0u8; 1024];
        // Don't initialize — magic is 0.
        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());

        // Set wrong magic.
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());
        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn write_and_drain_single_message() {
        let mut buf = make_ring(1024);
        let payload = b"hello world";
        let written = shm_write(&mut buf, 0, 1, payload);
        assert_eq!(written, MSG_HEADER_SIZE + payload.len());

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 1);
        assert_eq!(result.entries[0].payload, payload);
        assert!(result.entries[0].crc_ok);
        assert_eq!(result.drops, 0);
    }

    #[test]
    fn write_and_drain_multiple_messages() {
        let mut buf = make_ring(1024);
        shm_write(&mut buf, 0, 1, b"first");
        shm_write(&mut buf, 0, 2, b"second");
        shm_write(&mut buf, 0, 3, b"third");

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 3);
        assert_eq!(result.entries[0].msg_type, 1);
        assert_eq!(result.entries[0].payload, b"first");
        assert_eq!(result.entries[1].msg_type, 2);
        assert_eq!(result.entries[1].payload, b"second");
        assert_eq!(result.entries[2].msg_type, 3);
        assert_eq!(result.entries[2].payload, b"third");
        for e in &result.entries {
            assert!(e.crc_ok);
        }
    }

    #[test]
    fn write_empty_payload() {
        let mut buf = make_ring(1024);
        let written = shm_write(&mut buf, 0, 42, b"");
        assert_eq!(written, MSG_HEADER_SIZE);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 42);
        assert!(result.entries[0].payload.is_empty());
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn ring_full_increments_drops() {
        // Small ring: header (40) + data (60) = 100 bytes.
        // MSG_HEADER_SIZE = 16, so a message with 44 bytes payload = 60 bytes total
        // fills the ring exactly.
        let shm_size = HEADER_SIZE + 60;
        let mut buf = make_ring(shm_size);
        let payload = vec![0xAA; 44]; // 16 + 44 = 60, fills ring
        let written = shm_write(&mut buf, 0, 1, &payload);
        assert_eq!(written, 60);

        // Second write should fail — ring full.
        let written = shm_write(&mut buf, 0, 2, b"x");
        assert_eq!(written, 0);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.drops, 1);
    }

    #[test]
    fn ring_full_multiple_drops() {
        let shm_size = HEADER_SIZE + 32;
        let mut buf = make_ring(shm_size);
        let payload = vec![0xBB; 16]; // 16 + 16 = 32, fills ring
        shm_write(&mut buf, 0, 1, &payload);

        // Three failed writes.
        assert_eq!(shm_write(&mut buf, 0, 2, b"a"), 0);
        assert_eq!(shm_write(&mut buf, 0, 3, b"b"), 0);
        assert_eq!(shm_write(&mut buf, 0, 4, b"c"), 0);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.drops, 3);
    }

    #[test]
    fn wraparound_single_message() {
        // Ring with capacity = 48. Write a 32-byte message (16 hdr + 16 payload)
        // to advance write_ptr to 32. Then simulate the host advancing read_ptr
        // to 32. Then write another 32-byte message that wraps around.
        let shm_size = HEADER_SIZE + 48;
        let mut buf = make_ring(shm_size);

        // First message: 16 + 16 = 32 bytes.
        let payload1 = vec![0x11; 16];
        shm_write(&mut buf, 0, 1, &payload1);

        // Simulate host draining: advance read_ptr to match write_ptr.
        let hdr = read_header(&buf, 0);
        buf[24..32].copy_from_slice(&hdr.write_ptr.to_ne_bytes());

        // Second message: 16 + 16 = 32 bytes. Starts at position 32 in a
        // 48-byte ring, so it wraps around.
        let payload2 = vec![0x22; 16];
        shm_write(&mut buf, 0, 2, &payload2);

        // Drain should see only the second message.
        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 2);
        assert_eq!(result.entries[0].payload, payload2);
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn wraparound_message_header_splits() {
        // Ring with capacity = 40. Write 32 bytes to advance to position 32.
        // Then advance read_ptr. Write another message starting at 32 —
        // the 16-byte message header crosses the 40-byte boundary.
        let shm_size = HEADER_SIZE + 40;
        let mut buf = make_ring(shm_size);

        // First: 16 + 16 = 32 bytes.
        shm_write(&mut buf, 0, 1, &[0xAA; 16]);

        // Advance read_ptr.
        let hdr = read_header(&buf, 0);
        buf[24..32].copy_from_slice(&hdr.write_ptr.to_ne_bytes());

        // Second: 16 + 4 = 20 bytes, starting at position 32 in a 40-byte ring.
        // Header bytes: 32..40 (8 bytes) then 0..8 (8 bytes) — wraps mid-header.
        let payload2 = vec![0xBB; 4];
        shm_write(&mut buf, 0, 2, &payload2);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 2);
        assert_eq!(result.entries[0].payload, payload2);
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn crc_detects_corruption() {
        let mut buf = make_ring(1024);
        shm_write(&mut buf, 0, 1, b"integrity check");

        // Corrupt one byte of the payload in the ring data area.
        let data_start = HEADER_SIZE;
        let payload_start = data_start + MSG_HEADER_SIZE;
        buf[payload_start] ^= 0xFF;

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert!(!result.entries[0].crc_ok);
    }

    #[test]
    fn crc_empty_payload_is_zero_for_empty() {
        // CRC32 of empty input is 0x00000000.
        assert_eq!(crc32fast::hash(b""), 0x0000_0000);
    }

    #[test]
    fn crc32_known_vectors() {
        // Standard CRC32 test vectors.
        assert_eq!(crc32fast::hash(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32fast::hash(b""), 0x0000_0000);
        assert_eq!(crc32fast::hash(b"a"), 0xE8B7_BE43);
    }

    #[test]
    fn nonzero_shm_offset() {
        // SHM region at offset 4096 in a larger buffer (simulating guest memory).
        let offset = 4096;
        let shm_size = 512;
        let total = offset + shm_size;
        let mut buf = vec![0xFFu8; total];
        shm_init(&mut buf, offset, shm_size);

        shm_write(&mut buf, offset, 7, b"offset test");

        let result = shm_drain(&buf, offset);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 7);
        assert_eq!(result.entries[0].payload, b"offset test");
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn large_payload() {
        let mut buf = make_ring(65536);
        let payload = vec![0x42; 60000];
        let written = shm_write(&mut buf, 0, 99, &payload);
        assert_eq!(written, MSG_HEADER_SIZE + 60000);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].payload.len(), 60000);
        assert!(result.entries[0].payload.iter().all(|&b| b == 0x42));
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn incomplete_message_not_drained() {
        let mut buf = make_ring(1024);
        shm_write(&mut buf, 0, 1, b"complete");

        // Manually advance write_ptr by 20 bytes (pretend a message header
        // was written but payload is incomplete).
        let hdr = read_header(&buf, 0);
        let fake_write = hdr.write_ptr + 20;
        // Write a fake message header at the current write position claiming
        // 100 bytes of payload (which we don't actually write).
        let fake_msg = ShmMessage {
            msg_type: 99,
            length: 100,
            crc32: 0,
            _pad: 0,
        };
        let data_start = HEADER_SIZE;
        let capacity = hdr.capacity as usize;
        write_ring_bytes(
            &mut buf,
            data_start,
            capacity,
            hdr.write_ptr,
            fake_msg.as_bytes(),
        );
        // Advance write_ptr to only partially cover the fake message.
        buf[16..24].copy_from_slice(&fake_write.to_ne_bytes());

        let result = shm_drain(&buf, 0);
        // Only the first complete message should be drained.
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 1);
        assert_eq!(result.entries[0].payload, b"complete");
    }

    #[test]
    fn stimulus_payload_size_is_24() {
        assert_eq!(std::mem::size_of::<StimulusPayload>(), 24);
    }

    #[test]
    fn msg_type_stimulus_ascii() {
        let bytes = MSG_TYPE_STIMULUS.to_be_bytes();
        assert_eq!(&bytes, b"STIM");
    }

    #[test]
    fn msg_type_scenario_start_ascii() {
        let bytes = MSG_TYPE_SCENARIO_START.to_be_bytes();
        assert_eq!(&bytes, b"SCST");
    }

    #[test]
    fn msg_type_scenario_end_ascii() {
        let bytes = MSG_TYPE_SCENARIO_END.to_be_bytes();
        assert_eq!(&bytes, b"SCEN");
    }

    #[test]
    fn msg_type_sched_exit_ascii() {
        let bytes = MSG_TYPE_SCHED_EXIT.to_be_bytes();
        assert_eq!(&bytes, b"SCDX");
    }

    #[test]
    fn msg_type_crash_ascii() {
        let bytes = MSG_TYPE_CRASH.to_be_bytes();
        assert_eq!(&bytes, b"CRSH");
    }

    #[test]
    fn stimulus_payload_roundtrip() {
        let payload = StimulusPayload {
            elapsed_ms: 1234,
            step_index: 3,
            op_count: 5,
            op_kinds: 0b1010_0101,
            cgroup_count: 4,
            worker_count: 16,
            total_iterations: 99999,
        };
        let bytes = payload.as_bytes();
        let event = StimulusEvent::from_payload(bytes).unwrap();
        assert_eq!(event.elapsed_ms, 1234);
        assert_eq!(event.step_index, 3);
        assert_eq!(event.op_count, 5);
        assert_eq!(event.op_kinds, 0b1010_0101);
        assert_eq!(event.cgroup_count, 4);
        assert_eq!(event.worker_count, 16);
        assert_eq!(event.total_iterations, 99999);
    }

    #[test]
    fn stimulus_event_from_short_payload() {
        assert!(StimulusEvent::from_payload(&[0u8; 19]).is_none());
        assert!(StimulusEvent::from_payload(&[0u8; 24]).is_some());
    }

    #[test]
    fn stimulus_write_and_drain() {
        let mut buf = make_ring(1024);
        let payload = StimulusPayload {
            elapsed_ms: 500,
            step_index: 1,
            op_count: 3,
            op_kinds: 7,
            cgroup_count: 2,
            worker_count: 8,
            total_iterations: 42000,
        };
        let written = shm_write(&mut buf, 0, MSG_TYPE_STIMULUS, payload.as_bytes());
        assert_eq!(written, MSG_HEADER_SIZE + 24);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, MSG_TYPE_STIMULUS);
        assert!(result.entries[0].crc_ok);
        let event = StimulusEvent::from_payload(&result.entries[0].payload).unwrap();
        assert_eq!(event.elapsed_ms, 500);
        assert_eq!(event.step_index, 1);
        assert_eq!(event.op_count, 3);
    }

    #[test]
    fn header_fields_at_expected_offsets() {
        let mut buf = make_ring(256);
        // Write known values and verify byte-level layout.
        let hdr = ShmRingHeader {
            magic: SHM_RING_MAGIC,
            version: SHM_RING_VERSION,
            capacity: 216,
            control_bytes: 0,
            write_ptr: 0x1122_3344_5566_7788,
            read_ptr: 0xAABB_CCDD_EEFF_0011,
            drops: 42,
            ..ShmRingHeader::default()
        };
        buf[..HEADER_SIZE].copy_from_slice(hdr.as_bytes());

        assert_eq!(
            u32::from_ne_bytes(buf[0..4].try_into().unwrap()),
            SHM_RING_MAGIC
        );
        assert_eq!(
            u32::from_ne_bytes(buf[4..8].try_into().unwrap()),
            SHM_RING_VERSION
        );
        assert_eq!(u32::from_ne_bytes(buf[8..12].try_into().unwrap()), 216);
        assert_eq!(
            u64::from_ne_bytes(buf[16..24].try_into().unwrap()),
            0x1122_3344_5566_7788
        );
        assert_eq!(
            u64::from_ne_bytes(buf[24..32].try_into().unwrap()),
            0xAABB_CCDD_EEFF_0011
        );
        assert_eq!(u64::from_ne_bytes(buf[32..40].try_into().unwrap()), 42);
    }

    #[test]
    fn dump_req_offset_in_control_bytes() {
        assert_eq!(DUMP_REQ_OFFSET, 12);
        assert_eq!(DUMP_REQ_SYSRQ_D, b'D');
    }

    #[test]
    fn stimulus_event_from_exact_size_payload() {
        let payload = StimulusPayload {
            elapsed_ms: 42,
            step_index: 7,
            op_count: 3,
            op_kinds: 0xFF,
            cgroup_count: 2,
            worker_count: 10,
            total_iterations: 4,
        };
        let bytes = payload.as_bytes();
        assert_eq!(bytes.len(), 24);
        let event = StimulusEvent::from_payload(bytes).unwrap();
        assert_eq!(event.elapsed_ms, 42);
        assert_eq!(event.step_index, 7);
        assert_eq!(event.op_count, 3);
        assert_eq!(event.op_kinds, 0xFF);
        assert_eq!(event.cgroup_count, 2);
        assert_eq!(event.worker_count, 10);
        assert_eq!(event.total_iterations, 4);
    }

    #[test]
    fn stimulus_event_from_oversized_payload() {
        let mut bytes = vec![0u8; 32];
        // Set elapsed_ms to 123 at offset 0.
        bytes[0..4].copy_from_slice(&123u32.to_ne_bytes());
        let event = StimulusEvent::from_payload(&bytes).unwrap();
        assert_eq!(event.elapsed_ms, 123);
    }

    #[test]
    fn concurrent_producer_consumer_simulated() {
        // Simulate alternating writes and drains to exercise the read_ptr
        // advancement path.
        let shm_size = HEADER_SIZE + 128;
        let mut buf = make_ring(shm_size);

        // Write 3 messages, drain, advance read_ptr, write 3 more, drain.
        for round in 0..3 {
            let base_type = round * 10;
            shm_write(&mut buf, 0, base_type + 1, b"aa");
            shm_write(&mut buf, 0, base_type + 2, b"bb");
            shm_write(&mut buf, 0, base_type + 3, b"cc");

            let result = shm_drain(&buf, 0);
            assert_eq!(result.entries.len(), 3);
            for e in &result.entries {
                assert!(e.crc_ok);
            }

            // Advance read_ptr to write_ptr (simulate host consuming).
            let hdr = read_header(&buf, 0);
            buf[24..32].copy_from_slice(&hdr.write_ptr.to_ne_bytes());
        }
    }

    #[test]
    fn stimulus_event_from_empty_payload() {
        assert!(StimulusEvent::from_payload(&[]).is_none());
    }

    #[test]
    fn stimulus_event_clone_preserves_fields() {
        let event = StimulusEvent {
            elapsed_ms: 999,
            step_index: 7,
            op_count: 3,
            op_kinds: 0xF0,
            cgroup_count: 5,
            worker_count: 20,
            total_iterations: 16,
        };
        let c = event.clone();
        assert_eq!(c.elapsed_ms, 999);
        assert_eq!(c.step_index, 7);
        assert_eq!(c.op_count, 3);
        assert_eq!(c.op_kinds, 0xF0);
        assert_eq!(c.cgroup_count, 5);
        assert_eq!(c.worker_count, 20);
        assert_eq!(c.total_iterations, 16);
    }

    #[test]
    fn shm_drain_result_default_empty() {
        let r = ShmDrainResult::default();
        assert!(r.entries.is_empty());
        assert_eq!(r.drops, 0);
    }

    #[test]
    fn write_exact_capacity_then_empty() {
        // Exactly fill capacity with one message, drain, verify empty after.
        let data_size = 64;
        let shm_size = HEADER_SIZE + data_size;
        let mut buf = make_ring(shm_size);
        let payload_len = data_size - MSG_HEADER_SIZE;
        let payload = vec![0x55u8; payload_len];
        let written = shm_write(&mut buf, 0, 1, &payload);
        assert_eq!(written, data_size);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert!(result.entries[0].crc_ok);
        assert_eq!(result.entries[0].payload.len(), payload_len);
    }

    #[test]
    fn write_ring_bytes_wraparound_exact() {
        // Data area of 16 bytes, write 8 bytes starting at position 12 —
        // first 4 bytes fit, then wraps to start for remaining 4.
        let data_start = HEADER_SIZE;
        let capacity = 16;
        let shm_size = HEADER_SIZE + capacity;
        let mut buf = vec![0u8; shm_size];
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        write_ring_bytes(&mut buf, data_start, capacity, 12, &data);
        // Bytes at positions 12..16 then 0..4
        assert_eq!(&buf[data_start + 12..data_start + 16], &[1, 2, 3, 4]);
        assert_eq!(&buf[data_start..data_start + 4], &[5, 6, 7, 8]);
    }

    #[test]
    fn read_ring_bytes_wraparound_exact() {
        let data_start = HEADER_SIZE;
        let capacity = 16;
        let shm_size = HEADER_SIZE + capacity;
        let mut buf = vec![0u8; shm_size];
        // Plant data that wraps: positions 14..16 and 0..2
        buf[data_start + 14] = 0xAA;
        buf[data_start + 15] = 0xBB;
        buf[data_start] = 0xCC;
        buf[data_start + 1] = 0xDD;
        let out = read_ring_bytes(&buf, data_start, capacity, 14, 4);
        assert_eq!(out, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn stimulus_payload_as_bytes_roundtrip() {
        let p = StimulusPayload {
            elapsed_ms: u32::MAX,
            step_index: u16::MAX,
            op_count: u16::MAX,
            op_kinds: u32::MAX,
            cgroup_count: u16::MAX,
            worker_count: u16::MAX,
            total_iterations: u64::MAX,
        };
        let bytes = p.as_bytes();
        let e = StimulusEvent::from_payload(bytes).unwrap();
        assert_eq!(e.elapsed_ms, u32::MAX);
        assert_eq!(e.step_index, u16::MAX);
        assert_eq!(e.op_count, u16::MAX);
        assert_eq!(e.op_kinds, u32::MAX);
        assert_eq!(e.cgroup_count, u16::MAX);
        assert_eq!(e.worker_count, u16::MAX);
        assert_eq!(e.total_iterations, u64::MAX);
    }

    #[test]
    fn multiple_writes_fill_and_drop() {
        // Ring with 80 bytes of data. Each message = 16 + 8 = 24 bytes.
        // Can fit 3 messages (72 bytes). 4th should drop.
        let shm_size = HEADER_SIZE + 80;
        let mut buf = make_ring(shm_size);
        assert_eq!(shm_write(&mut buf, 0, 1, &[0xAA; 8]), 24);
        assert_eq!(shm_write(&mut buf, 0, 2, &[0xBB; 8]), 24);
        assert_eq!(shm_write(&mut buf, 0, 3, &[0xCC; 8]), 24);
        assert_eq!(shm_write(&mut buf, 0, 4, &[0xDD; 8]), 0); // dropped

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 3);
        assert_eq!(result.drops, 1);
    }

    // ---- read_ring_volatile multi-region routing -------------------
    //
    // `read_ring_volatile` reads each byte through `mem.read_u8(pa, 0)`,
    // so on a multi-region GuestMem each byte must resolve to the
    // region that contains its DRAM offset. This test wires up a
    // 2-region GuestMem with the ring's data area placed inside
    // region 1 (well past region 0's end) and verifies that the
    // bytes returned reflect region 1's host buffer, not stale
    // memory past region 0.

    #[test]
    fn read_ring_volatile_routes_through_correct_region() {
        use crate::monitor::reader::{GuestMem, MemRegion};

        // Region 0: 4 KiB at DRAM offset 0, filled with 0xAA.
        // Region 1: 4 KiB at DRAM offset 1 MiB, filled with 0xBB
        //           except for a planted 32-byte "ring data area"
        //           starting at byte 0 of region 1.
        let mut buf0 = vec![0xAAu8; 4096];
        let mut buf1 = vec![0xBBu8; 4096];
        let pattern: [u8; 32] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, //
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, //
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, //
            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, //
        ];
        buf1[0..32].copy_from_slice(&pattern);

        let regions = vec![
            MemRegion {
                host_ptr: buf0.as_mut_ptr(),
                offset: 0,
                size: 4096,
            },
            MemRegion {
                host_ptr: buf1.as_mut_ptr(),
                offset: 1 << 20, // 1 MiB
                size: 4096,
            },
        ];
        // SAFETY: buf0 and buf1 outlive the GuestMem use.
        let mem = unsafe { GuestMem::from_regions_for_test(regions) };

        // Read 32 bytes starting at the data area in region 1.
        // capacity is set generously so no wraparound occurs in this
        // straight-line read.
        let data_start_pa: u64 = 1 << 20;
        let capacity: usize = 4096;
        let mut out = vec![0u8; 32];
        read_ring_volatile(&mem, data_start_pa, capacity, 0, &mut out);

        assert_eq!(out, pattern);
        // No byte from region 0 (0xAA) leaked into the result.
        assert!(!out.contains(&0xAA));
    }

    #[test]
    fn read_ring_volatile_wraparound_routes_through_correct_region() {
        // Wraparound case: capacity is small enough that a read of
        // length capacity starting near the end wraps back to byte 0
        // of the data area. With multi-region routing, BOTH halves
        // of the wrapped read must hit the right region.
        use crate::monitor::reader::{GuestMem, MemRegion};

        let mut buf0 = vec![0xAAu8; 4096];
        let mut buf1 = vec![0u8; 4096];
        // Plant a 64-byte pattern; data area starts at offset 0 of
        // region 1, capacity 64. A read of 64 bytes starting at
        // pos 60 wraps: bytes at positions 60..64 then 0..60.
        for (i, slot) in buf1[..64].iter_mut().enumerate() {
            *slot = i as u8;
        }
        let regions = vec![
            MemRegion {
                host_ptr: buf0.as_mut_ptr(),
                offset: 0,
                size: 4096,
            },
            MemRegion {
                host_ptr: buf1.as_mut_ptr(),
                offset: 1 << 20,
                size: 4096,
            },
        ];
        // SAFETY: backing buffers outlive the GuestMem use.
        let mem = unsafe { GuestMem::from_regions_for_test(regions) };

        let data_start_pa: u64 = 1 << 20;
        let capacity: usize = 64;
        let mut out = vec![0u8; 64];
        // ptr 60 -> src_pos = 60 % 64 = 60. First chunk covers 4
        // bytes (positions 60..63), then wraps to position 0 for
        // the remaining 60.
        read_ring_volatile(&mem, data_start_pa, capacity, 60, &mut out);

        assert_eq!(out[0..4], [60u8, 61, 62, 63]);
        for (i, &b) in out[4..].iter().enumerate() {
            assert_eq!(b, i as u8);
        }
    }

    // ---- shm_write_raw round-trip --------------------------------
    //
    // `shm_write_raw` is the production guest-side writer that
    // operates on `*mut u8` instead of `&mut [u8]` so the host's
    // concurrent volatile reader cannot violate Rust aliasing. The
    // round-trip tests pin its byte-level layout against the slice
    // reader (`shm_drain`) — they must agree on every field.

    #[test]
    fn shm_write_raw_round_trips_through_drain() {
        // Allocate a buffer, init the ring header via the slice
        // helper, then write through the raw-pointer path. Drain via
        // the slice reader and verify the message arrived intact.
        let shm_size = 1024usize;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        let payload = b"raw-ptr round trip";
        // SAFETY: `buf` outlives the call; no concurrent reference
        // to its bytes exists during the write.
        unsafe {
            shm_write_raw(buf.as_mut_ptr(), shm_size, MSG_TYPE_STIMULUS, payload);
        }

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, MSG_TYPE_STIMULUS);
        assert_eq!(result.entries[0].payload, payload);
        assert!(result.entries[0].crc_ok);
        assert_eq!(result.drops, 0);
    }

    #[test]
    fn shm_write_raw_handles_wraparound() {
        // Force the ring to wrap mid-payload by pre-advancing
        // read_ptr/write_ptr near the boundary.
        let shm_size = HEADER_SIZE + 48;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        let capacity_u64 = (shm_size - HEADER_SIZE) as u64;
        // Place write_ptr at capacity - 8 so a 16-byte header + 4-byte
        // payload (= 20 bytes) wraps.
        let pre = capacity_u64 - 8;
        buf[16..24].copy_from_slice(&pre.to_ne_bytes());
        buf[24..32].copy_from_slice(&pre.to_ne_bytes());

        let payload = [0xAAu8, 0xBB, 0xCC, 0xDD];
        // SAFETY: see `shm_write_raw_round_trips_through_drain`.
        unsafe {
            shm_write_raw(buf.as_mut_ptr(), shm_size, 0xDEAD_BEEF, &payload);
        }
        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, 0xDEAD_BEEF);
        assert_eq!(result.entries[0].payload, payload);
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn shm_write_raw_rejects_undersized_mapping() {
        // A `size` smaller than HEADER_SIZE must noop without
        // touching memory — the volatile reads at offsets 8/16/24/32
        // would read past the end of the mapping otherwise.
        let mut buf = vec![0xAAu8; HEADER_SIZE - 1];
        // SAFETY: buf is `HEADER_SIZE - 1` bytes; the call must
        // bail before touching offset 8.
        unsafe {
            shm_write_raw(buf.as_mut_ptr(), buf.len(), 1, b"x");
        }
        // No bytes mutated — sentinel intact.
        assert!(buf.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn shm_write_raw_rejects_torn_capacity() {
        // capacity = 0 (torn init): drop without writing.
        let shm_size = 1024usize;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        // Overwrite capacity field with 0.
        buf[8..12].copy_from_slice(&0u32.to_ne_bytes());

        // SAFETY: see `shm_write_raw_round_trips_through_drain`.
        unsafe {
            shm_write_raw(buf.as_mut_ptr(), shm_size, 1, b"probe");
        }
        // write_ptr unchanged.
        assert_eq!(u64::from_ne_bytes(buf[16..24].try_into().unwrap()), 0);
    }

    #[test]
    fn shm_write_raw_increments_drops_when_full() {
        // Fill the ring, then write once more — drops must bump.
        let shm_size = HEADER_SIZE + 32;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        let big = vec![0xCCu8; 16]; // 16 + 16 = 32, fills capacity.
        // SAFETY: see `shm_write_raw_round_trips_through_drain`.
        unsafe {
            shm_write_raw(buf.as_mut_ptr(), shm_size, 1, &big);
            shm_write_raw(buf.as_mut_ptr(), shm_size, 2, b"x");
        }
        // drops field at offset 32.
        let drops = u64::from_ne_bytes(buf[32..40].try_into().unwrap());
        assert_eq!(drops, 1);
    }

    // ---- shm_drain panic-guard / OOM-cap / overflow guards ------

    #[test]
    fn shm_drain_returns_empty_when_buf_smaller_than_header() {
        // A misconfigured caller (e.g. shm_size < HEADER_SIZE)
        // must produce an empty drain instead of slice-panicking
        // inside `read_header`.
        let buf = vec![0u8; HEADER_SIZE - 1];
        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());
        assert_eq!(result.drops, 0);
    }

    #[test]
    fn shm_drain_returns_empty_when_offset_pushes_past_end() {
        // shm_offset positioned so that `shm_offset + HEADER_SIZE`
        // exceeds buf.len(). Must not panic.
        let buf = vec![0u8; 1024];
        let result = shm_drain(&buf, 1024 - HEADER_SIZE + 1);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn shm_drain_caps_torn_msg_length_against_capacity() {
        // Torn header advertises `msg.length = u32::MAX`. The OOM
        // guard must reject before allocation.
        let shm_size = 1024usize;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        // Plant a TLV header at the start of the data area:
        // msg_type=1, length=u32::MAX, crc=0, _pad=0.
        let data_start = HEADER_SIZE;
        buf[data_start..data_start + 4].copy_from_slice(&1u32.to_ne_bytes());
        buf[data_start + 4..data_start + 8].copy_from_slice(&u32::MAX.to_ne_bytes());
        buf[data_start + 8..data_start + 12].copy_from_slice(&0u32.to_ne_bytes());
        buf[data_start + 12..data_start + 16].copy_from_slice(&0u32.to_ne_bytes());
        // Set write_ptr large enough that the loop enters and the
        // torn `length` is read.
        buf[16..24].copy_from_slice(&64u64.to_ne_bytes());

        let result = shm_drain(&buf, 0);
        // Must not have allocated 4GiB; result is empty (cap break).
        assert!(result.entries.is_empty());
    }

    #[test]
    fn shm_drain_handles_read_pos_near_u64_max() {
        // Set read_ptr near u64::MAX and write_ptr just past wrap;
        // the modular distance check (`wrapping_sub`) must drive the
        // loop without overflowing the additive form.
        let shm_size = HEADER_SIZE + 64;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        // First write a normal message via shm_write so the layout
        // is valid, then force read_ptr/write_ptr to the post-wrap
        // window while keeping the pattern length consistent.
        let _ = shm_write(&mut buf, 0, 1, b"abcd"); // total 20.
        // Move write_ptr := 5, read_ptr := u64::MAX - 14
        // (so wrapping_sub(write_ptr, read_ptr) == 20).
        let new_write: u64 = 5;
        let new_read: u64 = u64::MAX - 14;
        buf[16..24].copy_from_slice(&new_write.to_ne_bytes());
        buf[24..32].copy_from_slice(&new_read.to_ne_bytes());
        // Sanity.
        assert_eq!(new_write.wrapping_sub(new_read), 20);

        // The drain enters the loop, computes total_msg_size=20,
        // attempts to read the header from `data_start + (read_pos %
        // capacity)`. Since the original write left valid bytes only
        // at position 0, the header read at position
        // `(u64::MAX - 14) % 64` resolves to a defined byte position
        // in the data area — the test's purpose is to confirm no
        // panic, not to verify the parsed payload. Drain returns
        // either empty (CRC mismatch / max_payload reject) or a
        // single entry, but never panics.
        let result = shm_drain(&buf, 0);
        // The pre-write established `drops = 0` and modular
        // arithmetic must not have aborted.
        assert!(result.entries.len() <= 1);
    }

    #[test]
    fn shm_drain_rejects_used_greater_than_capacity() {
        // Writer-side invariant: `used = write_ptr - read_ptr` is at
        // most `capacity`. A torn header can violate this (e.g.
        // `read_ptr > write_ptr` so `wrapping_sub` returns a
        // near-`u64::MAX` distance). The drain MUST bail before
        // entering the loop — without the guard it would iterate
        // ~2^60 times and OOM the host.
        let shm_size = 1024usize;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        // Force read_ptr > write_ptr to make wrapping_sub huge.
        let bogus_write: u64 = 0;
        let bogus_read: u64 = 100;
        buf[16..24].copy_from_slice(&bogus_write.to_ne_bytes());
        buf[24..32].copy_from_slice(&bogus_read.to_ne_bytes());
        // Sanity: distance vastly exceeds capacity (984 bytes).
        assert!(bogus_write.wrapping_sub(bogus_read) > (shm_size - HEADER_SIZE) as u64);

        // Must return immediately — no allocation, no iteration.
        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn shm_drain_rejects_used_one_past_capacity() {
        // Boundary case: distance is exactly `capacity + 1`. The
        // strict `>` comparison must reject; `==` would accept.
        let shm_size = HEADER_SIZE + 64;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        // capacity = 64. Set distance to 65 via wrapping_sub.
        let new_write: u64 = 65;
        let new_read: u64 = 0;
        buf[16..24].copy_from_slice(&new_write.to_ne_bytes());
        buf[24..32].copy_from_slice(&new_read.to_ne_bytes());

        let result = shm_drain(&buf, 0);
        assert!(result.entries.is_empty());
    }

    #[test]
    fn shm_drain_accepts_used_exactly_capacity() {
        // The invariant permits `used == capacity` (full ring); the
        // guard must NOT trip on this case. Build a full ring via
        // the writer and verify drain succeeds.
        let shm_size = HEADER_SIZE + 32;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);
        let payload = vec![0xCCu8; 16]; // 16 + 16 = 32 = capacity.
        let written = shm_write(&mut buf, 0, 1, &payload);
        assert_eq!(written, 32);

        let result = shm_drain(&buf, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].payload, payload);
    }

    // ---- IS_GUEST guard + write_msg round-trip ---------------------
    //
    // `write_msg` and `write_msg_nonblocking` are guest-only entry
    // points. The `is_guest()` guard rejects host-context invocations
    // before `shm_ptr()` would silently no-op anyway — distinguishing
    // a coding mistake (host caller) from a benign early-boot race
    // (guest caller before SHM init) at the call site.
    //
    // Two test fixtures pin the guard's behavior:
    //   1. host-context — `IsGuestOverrideGuard::new(false)` forces
    //      `is_guest() == false`. `write_msg_nonblocking` must return
    //      false; the global `SHM_PTR` must remain whatever it was.
    //   2. guest-context round-trip — `IsGuestOverrideGuard::new(true)`
    //      forces `is_guest() == true`, then a leaked, mutex-guarded
    //      ring buffer is wired into `SHM_PTR` (or used directly) and
    //      the message is round-tripped back via `shm_drain`.

    /// Test-only: serialize tests that depend on the global
    /// `SHM_PTR`. The OnceLock can only be initialized once, so all
    /// tests touching it share a single leaked buffer; the mutex
    /// ensures one test at a time observes a deterministic ring
    /// state.
    static SHM_PTR_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Test-only: leak a fresh ring buffer and install it into the
    /// global `SHM_PTR`. Idempotent — if `SHM_PTR` is already set
    /// (e.g. installed by an earlier test), the existing pointer is
    /// returned unchanged. Caller holds `SHM_PTR_TEST_LOCK` for the
    /// duration of the test to prevent interleaved writes.
    fn ensure_test_shm_ptr(shm_size: usize) -> (*mut u8, usize) {
        // Leak the buffer so it outlives the OnceLock for the
        // duration of the process — `init_shm_ptr` stores the raw
        // pointer indefinitely.
        let buf: Box<[u8]> = vec![0u8; shm_size].into_boxed_slice();
        let len = buf.len();
        let ptr = Box::leak(buf).as_mut_ptr();
        // `init_shm_ptr` is a no-op if SHM_PTR is already set, so
        // subsequent calls in the same process re-use the original
        // installed buffer. We never free leaked buffers.
        init_shm_ptr(ptr, len);
        let p = SHM_PTR.get().expect("SHM_PTR set above");
        (p.ptr, p.size)
    }

    #[test]
    fn is_guest_override_round_trips_through_thread_local() {
        // The override is a fixture for write_msg tests. Verify
        // that toggling it from false -> true -> back works on the
        // current thread, and that drop restores the previous
        // value. No process-global state is mutated.
        // Initial (no override): on the host, is_guest() reads
        // /proc/cmdline. We don't assert its value (test env may
        // vary); we only verify that overrides take effect.
        {
            let _g = IsGuestOverrideGuard::new(false);
            assert!(!is_guest());
        }
        {
            let _g = IsGuestOverrideGuard::new(true);
            assert!(is_guest());
        }
        // After dropping both guards, override is None. The
        // OnceLock-backed default is consulted; we don't assert
        // its value here since the test env's /proc/cmdline is
        // not under our control.
    }

    #[test]
    fn is_guest_override_guards_nest_correctly() {
        // Outer guard sets true; inner guard overrides to false;
        // dropping inner restores true; dropping outer restores
        // None. Ensures tests that nest contexts don't leak.
        let _outer = IsGuestOverrideGuard::new(true);
        assert!(is_guest());
        {
            let _inner = IsGuestOverrideGuard::new(false);
            assert!(!is_guest());
        }
        // Inner dropped — outer's value is restored.
        assert!(is_guest());
    }

    #[test]
    fn write_msg_nonblocking_rejects_host_context() {
        // Wire up SHM_PTR with a freshly-initialized ring BEFORE
        // forcing host context, so the guard is the ONLY thing that
        // can prevent the write. Without this setup, the test would
        // pass trivially: shm_ptr() would fail, and `assert_guest_context`
        // never gets exercised. This way, if the guard were removed,
        // shm_write_raw would advance write_ptr and the assertion
        // below would catch it.
        let _ptr_lock = SHM_PTR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let shm_size = 1024usize;
        let (ptr, size) = ensure_test_shm_ptr(shm_size);
        // SAFETY: lock held; no other test mutates this buffer
        // while we hold the mutex. `ptr` outlives the call (leaked).
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        shm_init(buf_slice, 0, size);

        // Now force host context and call.
        let _guard = IsGuestOverrideGuard::new(false);
        let result = write_msg_nonblocking(MSG_TYPE_STIMULUS, b"should-not-write");
        assert!(
            !result,
            "write_msg_nonblocking from host context must return false"
        );

        // Ring header MUST be untouched: the guard rejected the call
        // before any write reached shm_write_raw. write_ptr at offset
        // 16 is the canonical "did anything land?" signal — a
        // successful write would have advanced it past 0.
        let buf_slice_ro = unsafe { std::slice::from_raw_parts(ptr, size) };
        let write_ptr_after = u64::from_ne_bytes(buf_slice_ro[16..24].try_into().unwrap());
        assert_eq!(
            write_ptr_after, 0,
            "guard must prevent any write to the ring; write_ptr should remain 0"
        );
        let drops_after = u64::from_ne_bytes(buf_slice_ro[32..40].try_into().unwrap());
        assert_eq!(
            drops_after, 0,
            "guard must short-circuit before drop accounting"
        );
    }

    #[test]
    fn write_msg_does_not_panic_from_host_context() {
        // Mirror of the nonblocking test for the blocking entry
        // point. Wire up SHM_PTR + initialize so the guard is the
        // only thing that can stop the write. Verify (a) no panic,
        // (b) ring header unchanged after the call.
        let _ptr_lock = SHM_PTR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let shm_size = 1024usize;
        let (ptr, size) = ensure_test_shm_ptr(shm_size);
        // SAFETY: see write_msg_nonblocking_rejects_host_context.
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        shm_init(buf_slice, 0, size);

        let _guard = IsGuestOverrideGuard::new(false);
        write_msg(MSG_TYPE_STIMULUS, b"should-not-write");
        // No panic above — first half of the assertion. Now check
        // the ring header is unchanged.
        let buf_slice_ro = unsafe { std::slice::from_raw_parts(ptr, size) };
        let write_ptr_after = u64::from_ne_bytes(buf_slice_ro[16..24].try_into().unwrap());
        assert_eq!(
            write_ptr_after, 0,
            "guard must prevent any write to the ring; write_ptr should remain 0"
        );
    }

    #[test]
    fn write_msg_nonblocking_accepts_guest_context_and_round_trips() {
        // With is_guest() forced true and SHM_PTR wired to a
        // freshly-initialized ring, write_msg_nonblocking must
        // return true and the message must be drainable via
        // shm_drain (the slice-based reader).
        let _ptr_lock = SHM_PTR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guest = IsGuestOverrideGuard::new(true);
        let shm_size = 1024usize;
        let (ptr, size) = ensure_test_shm_ptr(shm_size);
        // Re-init the leaked buffer for this test: clear and
        // initialize the header. Subsequent tests will repeat.
        // SAFETY: lock held; no other test touches this buffer
        // while we hold the mutex.
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        shm_init(buf_slice, 0, size);

        let payload = b"guest-context round-trip";
        let ok = write_msg_nonblocking(MSG_TYPE_TEST_RESULT, payload);
        assert!(ok, "write_msg_nonblocking from guest context must succeed");

        // Drain via the slice reader. We borrow the leaked buffer
        // immutably; the mutex serializes all SHM_PTR-touching
        // tests so no concurrent writer aliases this view.
        let buf_slice_ro = unsafe { std::slice::from_raw_parts(ptr, size) };
        let result = shm_drain(buf_slice_ro, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, MSG_TYPE_TEST_RESULT);
        assert_eq!(result.entries[0].payload, payload);
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn write_msg_accepts_guest_context_and_round_trips() {
        // Mirror of the nonblocking test for the blocking
        // `write_msg` entry point. write_msg returns (); its
        // success is observable only via the resulting ring state.
        let _ptr_lock = SHM_PTR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guest = IsGuestOverrideGuard::new(true);
        let shm_size = 1024usize;
        let (ptr, size) = ensure_test_shm_ptr(shm_size);
        // SAFETY: see write_msg_nonblocking_accepts_guest_context_and_round_trips.
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        shm_init(buf_slice, 0, size);

        let payload = b"blocking round-trip";
        write_msg(MSG_TYPE_PAYLOAD_METRICS, payload);

        let buf_slice_ro = unsafe { std::slice::from_raw_parts(ptr, size) };
        let result = shm_drain(buf_slice_ro, 0);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.entries[0].msg_type, MSG_TYPE_PAYLOAD_METRICS);
        assert_eq!(result.entries[0].payload, payload);
        assert!(result.entries[0].crc_ok);
    }

    #[test]
    fn write_msg_round_trips_multiple_messages_in_guest_context() {
        // Three consecutive write_msg calls from guest context
        // must all land in the ring in order. Pins the SHM_PTR-
        // backed write path against the multi-write semantics
        // already verified for shm_write_raw.
        let _ptr_lock = SHM_PTR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _guest = IsGuestOverrideGuard::new(true);
        let shm_size = 1024usize;
        let (ptr, size) = ensure_test_shm_ptr(shm_size);
        // SAFETY: see write_msg_nonblocking_accepts_guest_context_and_round_trips.
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
        shm_init(buf_slice, 0, size);

        write_msg(MSG_TYPE_SCENARIO_START, b"start");
        write_msg(MSG_TYPE_STIMULUS, b"middle");
        write_msg(MSG_TYPE_SCENARIO_END, b"end");

        let buf_slice_ro = unsafe { std::slice::from_raw_parts(ptr, size) };
        let result = shm_drain(buf_slice_ro, 0);
        assert_eq!(result.entries.len(), 3);
        assert_eq!(result.entries[0].msg_type, MSG_TYPE_SCENARIO_START);
        assert_eq!(result.entries[0].payload, b"start");
        assert_eq!(result.entries[1].msg_type, MSG_TYPE_STIMULUS);
        assert_eq!(result.entries[1].payload, b"middle");
        assert_eq!(result.entries[2].msg_type, MSG_TYPE_SCENARIO_END);
        assert_eq!(result.entries[2].payload, b"end");
        for e in &result.entries {
            assert!(e.crc_ok);
        }
    }
}
