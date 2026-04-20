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
/// and KTSTR_SHM_SIZE environment variables on the kernel command line.
use zerocopy::{FromBytes, IntoBytes};

/// Result of a successful `/dev/mem` mmap of the SHM region.
pub(crate) struct ShmMmap {
    /// Pointer to the start of the SHM region (page-offset adjusted).
    pub ptr: *mut u8,
    /// Base address passed to munmap (page-aligned).
    pub map_base: *mut libc::c_void,
    /// Size passed to munmap.
    pub map_size: usize,
}

/// Page-aligned mmap of a physical address range via an open `/dev/mem` fd.
/// Returns the adjusted pointer to `shm_base` within the mapping.
pub(crate) fn mmap_devmem(
    fd: std::os::unix::io::RawFd,
    shm_base: u64,
    shm_size: u64,
) -> Option<ShmMmap> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
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

/// Magic value identifying a valid SHM ring header.
pub const SHM_RING_MAGIC: u32 = 0x5354_4d52; // "STMR"

/// Message type for stimulus events written by the guest step executor.
pub const MSG_TYPE_STIMULUS: u32 = 0x5354_494D; // "STIM"

/// Message type for scenario start marker.
pub const MSG_TYPE_SCENARIO_START: u32 = 0x5343_5354; // "SCST"

/// Message type for scenario end marker.
pub const MSG_TYPE_SCENARIO_END: u32 = 0x5343_454E; // "SCEN"

/// Message type for guest exit code (payload: 4-byte i32).
pub const MSG_TYPE_EXIT: u32 = 0x4558_4954; // "EXIT"

/// Message type for test result (payload: JSON-encoded AssertResult).
pub const MSG_TYPE_TEST_RESULT: u32 = 0x5445_5354; // "TEST"

/// Message type for scheduler process exit (payload: 4-byte i32 exit code).
/// Written by the guest init when the scheduler child process terminates
/// during test execution. The host monitor thread can detect this via
/// mid-flight SHM drain and terminate the VM early instead of waiting
/// for the full watchdog timeout.
pub const MSG_TYPE_SCHED_EXIT: u32 = 0x5343_4458; // "SCDX"

/// Message type for guest crash (payload: UTF-8 panic message + backtrace).
/// Written by the panic hook in rust_init.rs. SHM delivery is reliable
/// (memcpy to mapped memory) unlike serial which truncates large backtraces
/// because the UART cannot drain fast enough before reboot.
pub const MSG_TYPE_CRASH: u32 = 0x4352_5348; // "CRSH"

/// Message type for per-payload-invocation metrics (payload: JSON-encoded
/// [`PayloadMetrics`](crate::test_support::PayloadMetrics)). One entry
/// per terminal `.run()` / `.wait()` / `.kill()` / `.try_wait()` on a
/// [`PayloadRun`](crate::scenario::payload_run::PayloadRun) or
/// [`PayloadHandle`](crate::scenario::payload_run::PayloadHandle). The
/// host-side eval loop drains every `MSG_TYPE_PAYLOAD_METRICS` entry
/// in order and feeds the resulting `Vec<PayloadMetrics>` to the
/// sidecar writer so per-invocation provenance is preserved across
/// composed payload runs.
pub const MSG_TYPE_PAYLOAD_METRICS: u32 = 0x504d_4554; // "PMET"

/// Current header version.
pub const SHM_RING_VERSION: u32 = 1;

/// Byte offset within the SHM region for the host-to-guest dump request flag.
/// Occupies the first byte of the `control_bytes` field in ShmRingHeader (offset 12).
/// Host writes `DUMP_REQ_SYSRQ_D` to request a SysRq-D dump; guest polls
/// this byte, triggers the dump, and clears it back to 0.
pub const DUMP_REQ_OFFSET: usize = 12;

/// Value written to DUMP_REQ_OFFSET to request a SysRq-D dump.
pub const DUMP_REQ_SYSRQ_D: u8 = b'D';

/// Byte offset within the SHM region for the host-to-guest stall request flag.
/// Occupies the second byte of the `control_bytes` field in ShmRingHeader (offset 13).
/// Host writes `STALL_REQ_ACTIVATE` to request a scheduler stall; guest polls
/// this byte, creates /tmp/ktstr_stall, and clears it back to 0.
pub const STALL_REQ_OFFSET: usize = 13;

/// Value written to STALL_REQ_OFFSET to request a scheduler stall.
pub const STALL_REQ_ACTIVATE: u8 = b'S';

/// Base offset within the SHM region for numbered signal slots.
/// Slots occupy bytes starting at offset 14 (third byte of `control_bytes`)
/// and extending into byte 15 (fourth byte of `control_bytes`), providing
/// 2 slots (0..1). AtomicU8 with Acquire/Release ordering.
pub const SIGNAL_SLOT_BASE: usize = 14;

/// Number of available signal slots.
const SIGNAL_SLOT_COUNT: usize = 2;

/// Value written to signal slot 0 by the host to request graceful shutdown.
/// Distinct from the BPF map write signal (value 1) so the guest poll loop
/// can differentiate.
pub const SIGNAL_SHUTDOWN_REQ: u8 = 0xDD;

/// Value written to slot 1 by the guest when probes are attached and the
/// scenario is about to start. The `start_bpf_map_write` thread polls for
/// this value before writing the crash trigger, ensuring probes capture
/// the crash rather than missing it.
pub const SIGNAL_PROBES_READY: u8 = 2;

/// Guest-side: poll SHM slot until non-zero or timeout.
/// Reads via AtomicU8 with Acquire ordering. The SHM mmap pointer
/// is cached in a OnceLock, initialized from /proc/cmdline during
/// the first call (or from `init_shm_ptr`).
pub fn wait_for(slot: u8, timeout: std::time::Duration) -> anyhow::Result<()> {
    assert!(
        (slot as usize) < SIGNAL_SLOT_COUNT,
        "signal slot {slot} out of range"
    );
    let (ptr, _) = shm_ptr()?;
    let offset = SIGNAL_SLOT_BASE + slot as usize;
    let atom = unsafe { &*(ptr.add(offset) as *const std::sync::atomic::AtomicU8) };
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if atom.load(std::sync::atomic::Ordering::Acquire) != 0 {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    anyhow::bail!("signal slot {slot} timed out after {timeout:?}")
}

/// Guest-side: set a slot to non-zero.
/// Writes via AtomicU8 with Release ordering.
pub fn signal(slot: u8) {
    signal_value(slot, 1);
}

/// Host-side: write 1 to a signal slot in guest memory.
/// `mem` provides direct access to guest DRAM;
/// `shm_base` is the DRAM-relative offset of the SHM region.
pub fn signal_guest(mem: &crate::monitor::reader::GuestMem, shm_base: u64, slot: u8) {
    signal_guest_value(mem, shm_base, slot, 1);
}

/// Host-side: write an arbitrary value to a signal slot in guest memory.
pub fn signal_guest_value(
    mem: &crate::monitor::reader::GuestMem,
    shm_base: u64,
    slot: u8,
    value: u8,
) {
    assert!(
        (slot as usize) < SIGNAL_SLOT_COUNT,
        "signal slot {slot} out of range"
    );
    mem.write_u8(shm_base, SIGNAL_SLOT_BASE + slot as usize, value);
}

/// Guest-side: read the current value of a signal slot.
pub fn read_signal(slot: u8) -> u8 {
    assert!(
        (slot as usize) < SIGNAL_SLOT_COUNT,
        "signal slot {slot} out of range"
    );
    let Ok((ptr, _)) = shm_ptr() else { return 0 };
    let offset = SIGNAL_SLOT_BASE + slot as usize;
    let atom = unsafe { &*(ptr.add(offset) as *const std::sync::atomic::AtomicU8) };
    atom.load(std::sync::atomic::Ordering::Acquire)
}

/// Guest-side: set a slot to a specific value.
pub fn signal_value(slot: u8, value: u8) {
    assert!(
        (slot as usize) < SIGNAL_SLOT_COUNT,
        "signal slot {slot} out of range"
    );
    let Ok((ptr, _)) = shm_ptr() else { return };
    let offset = SIGNAL_SLOT_BASE + slot as usize;
    let atom = unsafe { &*(ptr.add(offset) as *const std::sync::atomic::AtomicU8) };
    atom.store(value, std::sync::atomic::Ordering::Release);
}

/// Set the cached SHM base pointer and region size. Called from
/// `start_shm_poll` in the guest init after parsing /proc/cmdline.
pub fn init_shm_ptr(base: *mut u8, size: usize) {
    let _ = SHM_PTR.set(ShmPtr { ptr: base, size });
}

/// Guest-side: write a TLV message to the SHM ring using the cached
/// mmap pointer. No-op if SHM is not initialized.
///
/// Acquires `SHM_WRITE_LOCK` to serialize against concurrent writers
/// (sched-exit-mon thread and step executor).
pub fn write_msg(msg_type: u32, payload: &[u8]) {
    let Ok((ptr, size)) = shm_ptr() else { return };
    // Safe to re-enter: write_ptr advances only after header + payload
    // land, so a mid-write panic cannot corrupt committed messages.
    let _guard = SHM_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let buf = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
    shm_write(buf, 0, msg_type, payload);
}

/// Guest-side: try to write a TLV message without blocking.
///
/// Uses `try_lock()` on `SHM_WRITE_LOCK`. If the lock is held (e.g.,
/// the panic occurred on the thread that holds it), silently returns
/// false so the caller can fall back to serial. No-op if SHM is not
/// initialized.
pub fn write_msg_nonblocking(msg_type: u32, payload: &[u8]) -> bool {
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
    let buf = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
    shm_write(buf, 0, msg_type, payload);
    true
}

/// Wrapper for a raw pointer + size that is Send+Sync.
/// SAFETY: The SHM pointer is set once during single-threaded init and
/// points into a /dev/mem mmap that outlives all guest threads.
struct ShmPtr {
    ptr: *mut u8,
    size: usize,
}
unsafe impl Send for ShmPtr {}
unsafe impl Sync for ShmPtr {}

/// Cached SHM mmap pointer for guest-side signal operations.
static SHM_PTR: std::sync::OnceLock<ShmPtr> = std::sync::OnceLock::new();

/// Mutex serializing guest-side SHM ring writes. Prevents the sched-exit-mon
/// thread (write_msg) and the step executor (ShmWriter::write) from
/// concurrently modifying the ring's write_ptr.
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

/// Ring buffer header at the start of the SHM region.
///
/// write_ptr and read_ptr are monotonically increasing byte offsets into
/// the data area. Actual position = ptr % capacity.
#[repr(C)]
#[derive(
    Clone, Copy, Default, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout,
)]
pub struct ShmRingHeader {
    pub magic: u32,
    pub version: u32,
    /// Data area size in bytes (region_size - sizeof(ShmRingHeader)).
    pub capacity: u32,
    /// Packed host→guest control bytes — NOT padding despite being
    /// declared as `u32` for alignment. Byte 0 = `DUMP_REQ_OFFSET`
    /// (SysRq-D dump trigger), byte 1 = `STALL_REQ_OFFSET` (scheduler
    /// stall trigger), bytes 2-3 = `SIGNAL_SLOT_BASE + {0, 1}` (2
    /// indexed `AtomicU8` signal slots). Read/written byte-wise with
    /// `Acquire`/`Release` ordering via the `*_OFFSET` / `*_BASE`
    /// constants above; the `u32` spelling exists only so the header
    /// remains a plain POD for zerocopy derive.
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
}

const _HEADER_SIZE: () = assert!(std::mem::size_of::<ShmRingHeader>() == 40);

/// TLV message header preceding each payload in the ring.
///
/// CRC32 covers only the payload bytes (not this header).
#[repr(C)]
#[derive(
    Clone, Copy, Default, FromBytes, IntoBytes, zerocopy::Immutable, zerocopy::KnownLayout,
)]
pub struct ShmMessage {
    pub msg_type: u32,
    pub length: u32,
    pub crc32: u32,
    pub _pad: u32,
}

const _MSG_SIZE: () = assert!(std::mem::size_of::<ShmMessage>() == 16);

/// Size of the ShmRingHeader.
pub const HEADER_SIZE: usize = std::mem::size_of::<ShmRingHeader>();
/// Size of the ShmMessage TLV header.
pub const MSG_HEADER_SIZE: usize = std::mem::size_of::<ShmMessage>();

/// A parsed message from the ring buffer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ShmEntry {
    pub msg_type: u32,
    pub payload: Vec<u8>,
    /// True if the CRC32 matched.
    pub crc_ok: bool,
}

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

/// Initialize the SHM ring header at the given offset in guest memory.
///
/// `buf` is the full guest memory slice. `shm_offset` is the byte offset
/// where the SHM region starts. `shm_size` is the total region size.
#[allow(dead_code)]
pub fn shm_init(buf: &mut [u8], shm_offset: usize, shm_size: usize) {
    // Clamp to 0 when the caller mis-sized the region: a 0-capacity ring
    // is internally consistent (every `shm_write` hits the ring-full
    // branch and drops), whereas an arithmetic underflow would panic the
    // VMM before the shm layout error could surface to the operator.
    let capacity = shm_size.saturating_sub(HEADER_SIZE);
    let header = ShmRingHeader {
        magic: SHM_RING_MAGIC,
        version: SHM_RING_VERSION,
        capacity: capacity as u32,
        control_bytes: 0,
        write_ptr: 0,
        read_ptr: 0,
        drops: 0,
    };
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
    ShmRingHeader::read_from_bytes(s).unwrap()
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
    let header = read_header(buf, shm_offset);
    if header.magic != SHM_RING_MAGIC {
        return ShmDrainResult::default();
    }

    let capacity = header.capacity as usize;
    let data_start = shm_offset + HEADER_SIZE;
    let mut read_pos = header.read_ptr;
    let write_pos = header.write_ptr;
    let mut entries = Vec::new();

    while read_pos + MSG_HEADER_SIZE as u64 <= write_pos {
        let mut hdr_buf = [0u8; MSG_HEADER_SIZE];
        read_ring_into(buf, data_start, capacity, read_pos, &mut hdr_buf);
        let msg = ShmMessage::read_from_bytes(&hdr_buf).unwrap();

        let total_msg_size = MSG_HEADER_SIZE as u64 + msg.length as u64;
        if read_pos + total_msg_size > write_pos {
            // Incomplete message — stop.
            break;
        }

        let payload = read_ring_bytes(
            buf,
            data_start,
            capacity,
            read_pos + MSG_HEADER_SIZE as u64,
            msg.length as usize,
        );

        let computed_crc = crc32fast::hash(&payload);
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok: computed_crc == msg.crc32,
        });

        read_pos += total_msg_size;
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

    let capacity = mem.read_u32(shm_base_pa, 8) as usize;
    let write_ptr = mem.read_u64(shm_base_pa, 16);
    let read_ptr = mem.read_u64(shm_base_pa, 24);
    let drops = mem.read_u64(shm_base_pa, 32);

    let data_start_pa = shm_base_pa + HEADER_SIZE as u64;
    let mut read_pos = read_ptr;
    let mut entries = Vec::new();

    while read_pos + MSG_HEADER_SIZE as u64 <= write_ptr {
        // Read message header via volatile.
        let mut hdr_buf = [0u8; MSG_HEADER_SIZE];
        read_ring_volatile(mem, data_start_pa, capacity, read_pos, &mut hdr_buf);
        let msg = ShmMessage::read_from_bytes(&hdr_buf).unwrap();

        let total_msg_size = MSG_HEADER_SIZE as u64 + msg.length as u64;
        if read_pos + total_msg_size > write_ptr {
            break;
        }

        let mut payload = vec![0u8; msg.length as usize];
        if !payload.is_empty() {
            read_ring_volatile(
                mem,
                data_start_pa,
                capacity,
                read_pos + MSG_HEADER_SIZE as u64,
                &mut payload,
            );
        }

        let computed_crc = crc32fast::hash(&payload);
        entries.push(ShmEntry {
            msg_type: msg.msg_type,
            payload,
            crc_ok: computed_crc == msg.crc32,
        });

        read_pos += total_msg_size;
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
            let byte = unsafe { std::ptr::read_volatile(mem.base_ptr().add(pa as usize)) };
            out[dst_pos + i] = byte;
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

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time size assertions (also present above, but explicit tests
    // for visibility in test output).
    #[test]
    fn header_size_is_40() {
        assert_eq!(std::mem::size_of::<ShmRingHeader>(), 40);
    }

    #[test]
    fn message_size_is_16() {
        assert_eq!(std::mem::size_of::<ShmMessage>(), 16);
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
        // Regression for #30: if the shared header is torn (or corrupt)
        // such that read_ptr > write_ptr, `wrapping_sub` yields a huge
        // "used" value. The new capacity check must detect the
        // invariant violation and return 0 rather than dropping via
        // the ordinary "ring full" path or silently corrupting state.
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
        // Regression for #30: when the monotonic write_ptr overflows
        // u64 (extremely rare but theoretically possible over long
        // runs), `wrapping_sub` gives the correct modular distance
        // while raw subtraction would underflow. Set write_ptr just
        // past wrap (= 10) and read_ptr just before wrap
        // (= u64::MAX - 5). Used distance = 16 via wrapping_sub.
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
    fn stall_req_offset_in_control_bytes() {
        assert_eq!(STALL_REQ_OFFSET, 13);
        assert_eq!(STALL_REQ_ACTIVATE, b'S');
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
}
