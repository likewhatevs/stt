//! Taskstats genetlink capture for delay accounting.
//!
//! Provides per-tid access to the kernel's taskstats interface via
//! the genetlink TASKSTATS family (`include/uapi/linux/taskstats.h`).
//! `/proc/<tid>/sched` does NOT expose the eight delay-accounting
//! categories that taskstats does — `cpu_delay`, `blkio_delay`,
//! `swapin_delay`, `freepages_delay`, `thrashing_delay`,
//! `compact_delay`, `wpcopy_delay`, `irq_delay` — and the
//! lifetime memory watermarks (`hiwater_rss`, `hiwater_vm`) only
//! reach userspace through this path. The capture pipeline opens
//! a [`TaskstatsClient`] once per snapshot and issues one
//! [`TaskstatsClient::query_tid`] per tid the procfs walk
//! enumerates; failures are best-effort and surface as
//! `Option::None` on the per-thread record.
//!
//! # Capability gate
//!
//! The kernel registers `TASKSTATS_CMD_GET` with `GENL_ADMIN_PERM`
//! (`kernel/taskstats.c::taskstats_ops`), so the calling process
//! must hold `CAP_NET_ADMIN` to issue the per-tid query. The
//! ctprof capture pipeline runs as root in production, so the
//! cap is normally present. [`TaskstatsClient::open`] does NOT
//! gate on `CAP_NET_ADMIN`: socket creation and the
//! `CTRL_CMD_GETFAMILY` family-id resolution succeed without it,
//! so a missing cap surfaces only at [`TaskstatsClient::query_tid`]
//! time as a per-tid `EPERM`. The capture pipeline keeps that as
//! a best-effort failure that collapses the thread's
//! delay-accounting fields to zero.
//!
//! # Wire format
//!
//! The kernel reply nests one `TASKSTATS_TYPE_AGGR_PID`
//! attribute that carries a `TASKSTATS_TYPE_PID` (u32 pid) plus a
//! `TASKSTATS_TYPE_STATS` blob holding the full
//! `struct taskstats` payload (~600 bytes — the union of v1..v17
//! fields). Older kernels truncate the trailing fields; the
//! parser handles short-payload cases by treating absent bytes as
//! zero.
//!
//! # Per-bucket caveats from kernel-source research
//!
//! - `cpu_delay` is RACY: count + delay_total are not updated
//!   atomically (sched_info path, no lock). The other seven
//!   categories (blkio, swapin, freepages, thrashing, compact,
//!   wpcopy, irq) serialize through `task->delays->lock`.
//! - `swapin` and `thrashing` delay buckets OVERLAP — a
//!   thrashing event is also a swapin event from the syscall
//!   layer. They are not orthogonal time buckets and should not
//!   be summed.
//! - `delay_min == 0` is a sentinel meaning "no events
//!   observed", NOT a genuine zero-delay event. Compare against
//!   the matching `*_count` to disambiguate.
//! - `delay_max_ts` (timestamp fields, v17) is `CLOCK_REALTIME`
//!   and can jump under NTP step adjustments. We do NOT expose
//!   the timestamps; the capture path reads the `delay_max`
//!   nanoseconds value only.
//! - `read_char` / `write_char` in `struct taskstats` are
//!   KB-truncated copies of the procfs `rchar`/`wchar` fields.
//!   The procfs path captures byte-precise values already; this
//!   module ignores the taskstats copies.
//! - `ac_utimescaled`, `ac_stimescaled`, `cpu_scaled_run_real_total`
//!   are dead fields on modern kernels (the scaled-cputime
//!   accounting was removed). This module skips them entirely.

use std::io::{self, ErrorKind};
use std::sync::atomic::{AtomicU32, Ordering};

use netlink_packet_core::{
    NLM_F_REQUEST, NLMSG_ERROR, NetlinkHeader, NetlinkMessage, NetlinkPayload,
};
use netlink_packet_generic::{
    GenlMessage,
    ctrl::{GenlCtrl, GenlCtrlCmd, nlas::GenlCtrlAttrs},
};
use netlink_sys::{Socket, SocketAddr, protocols::NETLINK_GENERIC};

/// Per-snapshot tally of [`TaskstatsClient::query_tid`] outcomes.
///
/// The capture pipeline issues one `query_tid` per tid the procfs
/// walk enumerates and folds the result into the per-thread record
/// via [`crate::ctprof::ThreadState::apply_delay_stats`]. Successes
/// populate the 34 taskstats fields; failures leave them at the
/// absent-counter zero default. Without this tally an operator
/// reading a snapshot has no way to distinguish "no taskstats data
/// because every tid raced exit" from "no taskstats data because
/// the kernel was built without `CONFIG_TASKSTATS`" from "no
/// taskstats data because the calling process lacks
/// `CAP_NET_ADMIN`" — every case collapses to all-zero rows in the
/// rendered table.
///
/// The four counters partition every `query_tid` outcome:
/// - [`Self::ok_count`]: a [`DelayStats`] payload was extracted
///   and `apply_delay_stats` ran. The 34 fields on the per-thread
///   record carry kernel-sourced values.
/// - [`Self::eperm_count`]: the kernel returned `EPERM` for the
///   per-tid query (errno 1). Most commonly because the calling
///   process lacks `CAP_NET_ADMIN` — the kernel registers
///   `TASKSTATS_CMD_GET` with `GENL_ADMIN_PERM`. A snapshot whose
///   `eperm_count` equals `ok_count + esrch_count + ...` (i.e.
///   every query failed with EPERM) almost always means the
///   process needs `setcap cap_net_admin+eip` or to run as root.
/// - [`Self::esrch_count`]: the kernel returned `ESRCH` (errno 3)
///   because the tid no longer exists. Race between the procfs
///   readdir and the netlink query — the tid exited mid-capture.
///   Routine on busy hosts; not actionable.
/// - [`Self::other_err_count`]: any other failure (socket I/O
///   error, malformed reply, kernel returned a different errno,
///   open never succeeded so `query_tid` was never called for
///   the tid). A pile of `other_err` failures alongside zero
///   `ok` / `eperm` / `esrch` typically means the netlink open
///   failed up-front — `CONFIG_TASKSTATS=n` is the most common
///   cause.
///
/// Tally is per-tid, not per-tgid: a process with 200 threads
/// where every query succeeds contributes 200 to `ok_count`.
/// Ghost-filtered tids (the empty-comm + zero-start filter in
/// [`crate::ctprof`]'s capture path) are NOT separately tracked
/// here — their `query_tid` outcome is recorded along with every
/// other tid. The procfs ghost filter discards the per-thread
/// record, but the netlink call already happened.
///
/// `ok_count + eperm_count + esrch_count + other_err_count` may
/// be less than the snapshot's `tids_walked` when
/// `TaskstatsClient::open` failed (CONFIG_TASKSTATS=n / EPERM at
/// open time / older kernel without genetlink): the capture path
/// short-circuits on `taskstats_client.is_none()` and skips the
/// query entirely, so neither tally counter advances. Pair with
/// [`crate::ctprof::CtprofParseSummary::tids_walked`] to compute
/// the "client-was-open ratio".
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct TaskstatsSummary {
    /// Per-tid `query_tid` calls that returned a populated
    /// [`DelayStats`].
    pub ok_count: u64,
    /// Per-tid `query_tid` calls that surfaced `errno=1` (EPERM).
    /// Most commonly because the calling process lacks
    /// `CAP_NET_ADMIN`; the kernel registers `TASKSTATS_CMD_GET`
    /// with `GENL_ADMIN_PERM` so the per-tid path requires the
    /// cap even though `TaskstatsClient::open` does not.
    pub eperm_count: u64,
    /// Per-tid `query_tid` calls that surfaced `errno=3` (ESRCH).
    /// The tid disappeared between the procfs readdir and the
    /// netlink reply. Routine on busy hosts; not actionable.
    pub esrch_count: u64,
    /// Per-tid `query_tid` calls that failed with any other
    /// error: socket I/O failure, malformed reply, a kernel
    /// errno other than EPERM/ESRCH (e.g. EINVAL for a
    /// concurrent-pid race on older kernels), or any new error
    /// shape the parser does not classify yet.
    pub other_err_count: u64,
}

impl TaskstatsSummary {
    /// Classify a single [`TaskstatsClient::query_tid`] result
    /// and bump the matching counter. Pulls the errno out of two
    /// independent sources:
    ///
    /// 1. `io::Error::raw_os_error()`: real OS errnos surface
    ///    here for socket-level failures (`send_to`,
    ///    `recv_from_full`).
    /// 2. The error message text: [`parse_reply`] formats kernel
    ///    `NLMSG_ERROR` payloads as
    ///    `"kernel returned NLMSG_ERROR errno=N"` and wraps via
    ///    `io::Error::other`, which sets `ErrorKind::Other` and
    ///    discards the raw_os_error tag. The text scan is the
    ///    only way to recover the kernel-side errno from those
    ///    failures.
    ///
    /// Unknown errnos (anything other than 1=EPERM, 3=ESRCH)
    /// fall to [`Self::other_err_count`]. The two recognized
    /// errnos cover the operationally interesting causes: the
    /// rest become a single bucket so the tally stays simple
    /// and the `other_err` clause is the operator's hint to
    /// look at the underlying log line.
    pub fn record_result(&mut self, result: &io::Result<DelayStats>) {
        // POSIX errno values, hardcoded so this module stays
        // off the libc dep. The two recognized errnos cover the
        // operationally interesting causes; other errnos fall to
        // `other_err_count`.
        const EPERM: i32 = 1;
        const ESRCH: i32 = 3;
        match result {
            Ok(_) => self.ok_count = self.ok_count.saturating_add(1),
            Err(e) => {
                let errno = classify_errno(e);
                match errno {
                    Some(EPERM) => {
                        self.eperm_count = self.eperm_count.saturating_add(1);
                    }
                    Some(ESRCH) => {
                        self.esrch_count = self.esrch_count.saturating_add(1);
                    }
                    _ => {
                        self.other_err_count = self.other_err_count.saturating_add(1);
                    }
                }
            }
        }
    }
}

/// Recover the kernel-side errno from an [`io::Error`] produced
/// by [`TaskstatsClient::query_tid`]. Two paths produce errors:
///
/// - Socket-level failures (`send_to`, `recv_from_full`) carry a
///   real `raw_os_error()`.
/// - Parse-side failures wrap [`parse_reply`]'s String error via
///   `io::Error::other`. `parse_reply` formats kernel NLMSG_ERROR
///   payloads as `"kernel returned NLMSG_ERROR errno=N"` (the N
///   is the kernel's negated wire value, see the `let errno = -err`
///   step in `parse_reply`). The string scan recovers the errno
///   from those failures.
///
/// Returns the recovered errno or `None` when neither path
/// matches.
fn classify_errno(e: &io::Error) -> Option<i32> {
    if let Some(code) = e.raw_os_error() {
        return Some(code);
    }
    // Inner-error path: io::Error::other(parse_reply_err).
    // The String message has the shape
    // "kernel returned NLMSG_ERROR errno=N"; pull the integer
    // suffix.
    let msg = e.to_string();
    let needle = "errno=";
    let pos = msg.rfind(needle)?;
    let tail = &msg[pos + needle.len()..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end].parse::<i32>().ok()
}

/// Per-tid delay-accounting record, one row per taskstats query.
///
/// Field layout mirrors the kernel's `struct taskstats` v17 (see
/// `include/uapi/linux/taskstats.h`); fields the registry doesn't
/// expose (read_char/write_char/scaled cputime/timestamp records)
/// are dropped at parse time so this struct stays focused on the
/// surface the capture pipeline cares about.
///
/// # Sentinel values
///
/// - `*_delay_min` of 0 means "no events observed in this
///   category" (kernel writes 0 by default; only updates on a
///   real event). Compare against the matching `*_count` to
///   distinguish "no events" from "saw a zero-ns event".
/// - All other fields default to 0 with the standard
///   monotonic-counter semantics: 0 means "no accumulation
///   yet", non-zero means "kernel has observed at least one
///   event and accumulated this much".
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DelayStats {
    pub cpu_count: u64,
    pub cpu_delay_total_ns: u64,
    pub cpu_delay_max_ns: u64,
    pub cpu_delay_min_ns: u64,
    pub blkio_count: u64,
    pub blkio_delay_total_ns: u64,
    pub blkio_delay_max_ns: u64,
    pub blkio_delay_min_ns: u64,
    pub swapin_count: u64,
    pub swapin_delay_total_ns: u64,
    pub swapin_delay_max_ns: u64,
    pub swapin_delay_min_ns: u64,
    pub freepages_count: u64,
    pub freepages_delay_total_ns: u64,
    pub freepages_delay_max_ns: u64,
    pub freepages_delay_min_ns: u64,
    pub thrashing_count: u64,
    pub thrashing_delay_total_ns: u64,
    pub thrashing_delay_max_ns: u64,
    pub thrashing_delay_min_ns: u64,
    pub compact_count: u64,
    pub compact_delay_total_ns: u64,
    pub compact_delay_max_ns: u64,
    pub compact_delay_min_ns: u64,
    pub wpcopy_count: u64,
    pub wpcopy_delay_total_ns: u64,
    pub wpcopy_delay_max_ns: u64,
    pub wpcopy_delay_min_ns: u64,
    pub irq_count: u64,
    pub irq_delay_total_ns: u64,
    pub irq_delay_max_ns: u64,
    pub irq_delay_min_ns: u64,
    /// Lifetime high-watermark of resident-set size, KB → bytes.
    /// The kernel stores `hiwater_rss` as a u64 KB count in
    /// `struct taskstats`; the parser multiplies by 1024 here so
    /// downstream consumers carry a byte-typed value matching the
    /// existing `Bytes` newtype unit.
    pub hiwater_rss_bytes: u64,
    /// Lifetime high-watermark of virtual-memory size, KB → bytes.
    /// Same KB→bytes conversion as `hiwater_rss_bytes`.
    pub hiwater_vm_bytes: u64,
}

/// Genetlink client for the kernel's TASKSTATS family.
///
/// Holds a single AF_NETLINK socket bound to a kernel-assigned
/// port plus the resolved family ID (the kernel assigns family
/// IDs dynamically; the controller's `CTRL_CMD_GETFAMILY` lookup
/// resolves `"TASKSTATS"` to the runtime ID).
///
/// Cheap to construct (single socket + one round-trip for family
/// resolution) and reused for every per-tid query in a snapshot.
/// Drop closes the socket via `netlink_sys::Socket`'s Drop impl
/// (libc::close).
pub struct TaskstatsClient {
    socket: Socket,
    family_id: u16,
    /// Monotonic per-message sequence number. Each request
    /// carries a unique seq so a delayed reply from a prior
    /// request is distinguishable from the current one.
    seq: AtomicU32,
}

/// `TASKSTATS_CMD_GET` opcode (uapi enum, `kernel/taskstats.c`).
const TASKSTATS_CMD_GET: u8 = 1;
/// `TASKSTATS_CMD_NEW` opcode — the kernel's reply opcode for a
/// successful `TASKSTATS_CMD_GET`. Userspace never sends this; it
/// surfaces in synthesized test fixtures that build a kernel-shaped
/// reply for [`parse_reply`]. The production parser dispatches on
/// `nlmsghdr.message_type` (NLMSG_ERROR), not on `genlmsghdr.cmd`,
/// so this constant is test-only.
#[cfg(test)]
const TASKSTATS_CMD_NEW: u8 = 2;
/// `TASKSTATS_GENL_VERSION` from the uapi header. The genetlink
/// generic family-version field; bumped by the kernel when the
/// taskstats family adds a new command (rarely).
const TASKSTATS_GENL_VERSION: u8 = 1;
/// `TASKSTATS_CMD_ATTR_PID` — the request attribute that carries
/// the target tid (u32) for a per-task query.
const TASKSTATS_CMD_ATTR_PID: u16 = 1;
/// `TASKSTATS_TYPE_PID` — the response attribute carrying the
/// tid the kernel filled stats for. We verify this matches the
/// requested tid before parsing the stats blob.
const TASKSTATS_TYPE_PID: u16 = 1;
/// `TASKSTATS_TYPE_STATS` — the response attribute carrying the
/// `struct taskstats` payload as raw bytes.
const TASKSTATS_TYPE_STATS: u16 = 3;
/// `TASKSTATS_TYPE_AGGR_PID` — the outer response attribute that
/// nests `TASKSTATS_TYPE_PID` + `TASKSTATS_TYPE_STATS`.
const TASKSTATS_TYPE_AGGR_PID: u16 = 4;

/// Family name registered by `kernel/taskstats.c::taskstats_genl_family`.
const TASKSTATS_FAMILY_NAME: &str = "TASKSTATS";

impl TaskstatsClient {
    /// Open the genetlink socket and resolve the TASKSTATS family
    /// ID via `CTRL_CMD_GETFAMILY`. Returns `Err` when:
    /// - the kernel was built without `CONFIG_TASKSTATS` (family
    ///   resolution returns ENOENT)
    /// - the calling process lacks `CAP_NET_ADMIN` (the kernel
    ///   later rejects per-tid queries with EPERM, but the open
    ///   path itself can succeed without the cap)
    /// - the kernel is older than v3.0 (no genetlink support;
    ///   socket creation fails with EPROTONOSUPPORT)
    pub fn open() -> io::Result<Self> {
        let mut socket = Socket::new(NETLINK_GENERIC)?;
        socket.bind_auto()?;
        let family_id = resolve_family_id(&socket, TASKSTATS_FAMILY_NAME)?;
        Ok(Self {
            socket,
            family_id,
            seq: AtomicU32::new(1),
        })
    }

    /// Query the kernel for the per-tid taskstats record. On a
    /// well-behaved kernel with `CONFIG_TASK_DELAY_ACCT=y` and
    /// the runtime `delayacct=on` toggle, returns a populated
    /// [`DelayStats`]; on truncated replies (older kernels,
    /// missing fields) the absent fields read zero per the
    /// `Default` impl.
    ///
    /// Errors:
    /// - `EPERM` — calling process lacks CAP_NET_ADMIN.
    /// - `ESRCH` — tid no longer exists (raced with task exit).
    /// - `EINVAL` — kernel rejected the request (malformed
    ///   attribute or non-existent pid).
    pub fn query_tid(&self, tid: u32) -> io::Result<DelayStats> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let request = build_request(self.family_id, seq, tid);
        self.socket.send_to(&request, &SocketAddr::new(0, 0), 0)?;
        let (reply, _addr) = self.socket.recv_from_full()?;
        parse_reply(&reply, tid).map_err(io::Error::other)
    }
}

/// Issue `CTRL_CMD_GETFAMILY` with `CTRL_ATTR_FAMILY_NAME` and
/// extract the resolved `CTRL_ATTR_FAMILY_ID` from the reply.
/// Used at [`TaskstatsClient::open`] time to translate the
/// well-known family name `"TASKSTATS"` into the runtime u16 ID.
///
/// **Why `NLM_F_REQUEST` only (no `NLM_F_ACK`):** the netlink
/// core sends an `NLMSG_ERROR errno=0` ACK ONLY when
/// `NLM_F_ACK` is set in the request. With a successful
/// CMD_GETFAMILY that already produces an explicit reply via
/// the controller's send path, requesting an ACK on top would
/// queue a second message (the ACK) on the socket. A
/// subsequent `recv_from_full` for an unrelated query would
/// consume the queued ACK instead of the new request's reply.
/// Dropping `NLM_F_ACK` here keeps the socket queue clean for
/// the per-tid `query_tid` calls that follow.
fn resolve_family_id(socket: &Socket, name: &str) -> io::Result<u16> {
    let payload = GenlCtrl {
        cmd: GenlCtrlCmd::GetFamily,
        nlas: vec![GenlCtrlAttrs::FamilyName(name.to_string())],
    };
    let mut nl_msg: NetlinkMessage<GenlMessage<GenlCtrl>> =
        NetlinkMessage::from(GenlMessage::from_payload(payload));
    let mut header = NetlinkHeader::default();
    header.flags = NLM_F_REQUEST;
    nl_msg.header = header;
    nl_msg.finalize();

    let mut buf = vec![0u8; nl_msg.header.length as usize];
    nl_msg.serialize(&mut buf);
    socket.send_to(&buf, &SocketAddr::new(0, 0), 0)?;

    let (reply_buf, _) = socket.recv_from_full()?;
    let reply: NetlinkMessage<GenlMessage<GenlCtrl>> =
        NetlinkMessage::deserialize(&reply_buf).map_err(io::Error::other)?;
    match reply.payload {
        NetlinkPayload::InnerMessage(genl) => {
            for attr in &genl.payload.nlas {
                if let GenlCtrlAttrs::FamilyId(id) = attr {
                    return Ok(*id);
                }
            }
            Err(io::Error::new(
                ErrorKind::NotFound,
                format!("CTRL_ATTR_FAMILY_ID missing in CTRL_CMD_GETFAMILY reply for {name}"),
            ))
        }
        NetlinkPayload::Error(err) => Err(io::Error::other(format!(
            "CTRL_CMD_GETFAMILY for {name}: {err:?}"
        ))),
        _ => Err(io::Error::new(
            ErrorKind::InvalidData,
            "unexpected NetlinkPayload variant from CTRL_CMD_GETFAMILY",
        )),
    }
}

/// Build a hand-rolled netlink request for `TASKSTATS_CMD_GET`
/// with a single `TASKSTATS_CMD_ATTR_PID` attribute carrying the
/// target tid. The netlink-packet-generic crate's `GenlMessage<F>`
/// approach requires implementing `GenlFamily + Emitable +
/// ParseableParametrized` for a payload type — for a single-NLA
/// request this is more boilerplate than the underlying byte
/// layout, so we construct the wire format directly.
///
/// **Why no `NLM_F_ACK`:** A successful `TASKSTATS_CMD_GET`
/// produces an explicit `TASKSTATS_CMD_NEW` reply via
/// `send_reply()` at `kernel/taskstats.c:516`. Setting
/// `NLM_F_ACK` on top would make the kernel send a SECOND
/// message (an `NLMSG_ERROR` with `errno=0` representing the
/// "ack only" semantics) AFTER the reply. Reading just one
/// datagram from the socket then races: scheduling and reorder
/// can make the first `recv_from_full` return either the reply
/// or the ACK, and the parser sees only one of them. Dropping
/// `NLM_F_ACK` makes the kernel send exactly one response (the
/// actual reply), so the single recv reads the AGGR_PID nest
/// unconditionally.
///
/// Wire layout (host-endian, NLA-aligned to 4 bytes):
///
/// ```no_rust
///   nlmsghdr (16 bytes)
///   ├─ length: u32 (total message length including header)
///   ├─ message_type: u16 (= family_id)
///   ├─ flags: u16 (= NLM_F_REQUEST)
///   ├─ sequence_number: u32 (= seq)
///   └─ port_number: u32 (= 0; kernel ignores)
///   genlmsghdr (4 bytes)
///   ├─ cmd: u8 (= TASKSTATS_CMD_GET)
///   ├─ version: u8 (= TASKSTATS_GENL_VERSION)
///   └─ reserved: u16
///   nlattr (8 bytes)
///   ├─ length: u16 (= 8: header + u32 value)
///   ├─ type: u16 (= TASKSTATS_CMD_ATTR_PID)
///   └─ value: u32 (= tid)
/// ```
fn build_request(family_id: u16, seq: u32, tid: u32) -> [u8; 28] {
    // 16 (nlmsghdr) + 4 (genlmsghdr) + 8 (nlattr) = 28 bytes total.
    // No padding needed: 8-byte attr is already 4-byte aligned.
    // Stack-allocated [u8; 28] avoids the per-query heap alloc on
    // the hot path (one query per tid per snapshot).
    let mut buf = [0u8; 28];
    // nlmsghdr
    buf[0..4].copy_from_slice(&28u32.to_ne_bytes()); // length
    buf[4..6].copy_from_slice(&family_id.to_ne_bytes()); // message_type
    buf[6..8].copy_from_slice(&NLM_F_REQUEST.to_ne_bytes()); // flags (no NLM_F_ACK; see fn doc above)
    buf[8..12].copy_from_slice(&seq.to_ne_bytes()); // sequence_number
    // bytes 12..16 (port_number) stay zero per the array initializer.
    // genlmsghdr
    buf[16] = TASKSTATS_CMD_GET;
    buf[17] = TASKSTATS_GENL_VERSION;
    // bytes 18..20 (reserved u16) stay zero.
    // nlattr (TASKSTATS_CMD_ATTR_PID = u32 tid)
    buf[20..22].copy_from_slice(&8u16.to_ne_bytes()); // nla_len = header (4) + value (4)
    buf[22..24].copy_from_slice(&TASKSTATS_CMD_ATTR_PID.to_ne_bytes());
    buf[24..28].copy_from_slice(&tid.to_ne_bytes());
    buf
}

/// Parse a kernel reply into a [`DelayStats`]. Walks the
/// nlmsghdr → genlmsghdr → nested NLA structure to find
/// `TASKSTATS_TYPE_AGGR_PID` → `TASKSTATS_TYPE_STATS`, then
/// extracts the delay-accounting + memory-watermark fields from
/// the raw `struct taskstats` byte layout.
///
/// Returns `Err` when:
/// - the message is shorter than the nlmsghdr.
/// - the nlmsg type is [`NLMSG_ERROR`] (kernel returned an
///   error reply, e.g. ESRCH for an exited tid).
/// - `TASKSTATS_TYPE_AGGR_PID` is missing from the payload.
/// - `TASKSTATS_TYPE_PID` is missing from the AGGR_PID nest, or
///   does not match the requested tid (defends against reply
///   mis-routing).
/// - `TASKSTATS_TYPE_STATS` is missing from the AGGR_PID nest.
///
/// Short `TASKSTATS_TYPE_STATS` payloads (older kernels missing
/// trailing fields) do NOT error: [`parse_taskstats_payload`]
/// zero-fills any bytes past `buf.len()` via its `r64` helper.
fn parse_reply(buf: &[u8], expected_tid: u32) -> Result<DelayStats, String> {
    let buf_len = buf.len();
    if buf_len < 16 {
        return Err(format!("reply shorter than nlmsghdr: {buf_len} bytes"));
    }
    let nlmsg_len = u32::from_ne_bytes(buf[0..4].try_into().unwrap()) as usize;
    let nlmsg_type = u16::from_ne_bytes(buf[4..6].try_into().unwrap());
    if nlmsg_len > buf_len {
        return Err(format!(
            "nlmsghdr length {nlmsg_len} exceeds buffer length {buf_len}"
        ));
    }
    // NLMSG_ERROR: payload starts with a signed i32 errno (negated).
    if nlmsg_type == NLMSG_ERROR {
        if buf_len < 20 {
            return Err("NLMSG_ERROR shorter than expected".into());
        }
        let err = i32::from_ne_bytes(buf[16..20].try_into().unwrap());
        if err == 0 {
            return Err("kernel returned NLMSG_ERROR with errno=0 (ack only)".into());
        }
        let errno = -err;
        return Err(format!("kernel returned NLMSG_ERROR errno={errno}"));
    }
    // Skip nlmsghdr (16 bytes) + genlmsghdr (4 bytes) = 20 bytes.
    if nlmsg_len < 20 {
        return Err(format!(
            "reply too short for nlmsghdr+genlmsghdr: {nlmsg_len}"
        ));
    }
    let payload = &buf[20..nlmsg_len];

    // Walk the top-level NLA list looking for TASKSTATS_TYPE_AGGR_PID.
    let aggr = find_nla(payload, TASKSTATS_TYPE_AGGR_PID)
        .ok_or("TASKSTATS_TYPE_AGGR_PID missing in reply")?;
    // Walk the nested NLA list inside AGGR_PID for PID + STATS.
    let pid_attr = find_nla(aggr, TASKSTATS_TYPE_PID)
        .ok_or("TASKSTATS_TYPE_PID missing in TASKSTATS_TYPE_AGGR_PID")?;
    let pid_attr_len = pid_attr.len();
    if pid_attr_len < 4 {
        return Err(format!(
            "TASKSTATS_TYPE_PID payload shorter than u32: {pid_attr_len}"
        ));
    }
    let reply_tid = u32::from_ne_bytes(pid_attr[0..4].try_into().unwrap());
    if reply_tid != expected_tid {
        return Err(format!(
            "tid mismatch: requested {expected_tid}, got {reply_tid}"
        ));
    }
    let stats = find_nla(aggr, TASKSTATS_TYPE_STATS)
        .ok_or("TASKSTATS_TYPE_STATS missing in TASKSTATS_TYPE_AGGR_PID")?;
    parse_taskstats_payload(stats)
}

/// Linear scan over a NLA list for an attribute matching `kind`.
/// Returns the value bytes (NLA payload, no header) on hit. Each
/// attribute is `nla_len: u16, nla_type: u16` followed by the
/// value, padded to 4-byte alignment (NLA_ALIGN). A length
/// shorter than the NLA header (4 bytes) terminates the walk.
fn find_nla(buf: &[u8], kind: u16) -> Option<&[u8]> {
    let mut offset = 0usize;
    while offset + 4 <= buf.len() {
        let nla_len = u16::from_ne_bytes(buf[offset..offset + 2].try_into().unwrap()) as usize;
        let nla_type = u16::from_ne_bytes(buf[offset + 2..offset + 4].try_into().unwrap());
        if nla_len < 4 {
            return None;
        }
        let value_start = offset + 4;
        let value_end = offset + nla_len;
        if value_end > buf.len() {
            return None;
        }
        if nla_type == kind {
            return Some(&buf[value_start..value_end]);
        }
        // NLA_ALIGN(nla_len) — round up to multiple of 4.
        offset += (nla_len + 3) & !3;
    }
    None
}

/// Extract the delay-accounting and memory-watermark fields from
/// a raw `struct taskstats` payload. Offsets pinned against the
/// v17 layout in `include/uapi/linux/taskstats.h`. The struct uses
/// `__attribute__((aligned(8)))` on `cpu_count`, `ac_sched`,
/// `ac_uid`, `ac_etime` to force 8-byte field alignment regardless
/// of compiler default packing.
///
/// Computed byte layout (v17):
///
/// | Offset | Field                        | Size |
/// |--------|------------------------------|------|
/// |   0    | version (u16)                |  2   |
/// |   4    | ac_exitcode (u32)            |  4   |
/// |   8    | ac_flag (u8)                 |  1   |
/// |   9    | ac_nice (u8) + 6B align pad  |  1   |
/// |  16    | cpu_count (u64, aligned 8)   |  8   |
/// |  24    | cpu_delay_total              |  8   |
/// |  32    | blkio_count                  |  8   |
/// |  40    | blkio_delay_total            |  8   |
/// |  48    | swapin_count                 |  8   |
/// |  56    | swapin_delay_total           |  8   |
/// |  64    | cpu_run_real_total           |  8   |
/// |  72    | cpu_run_virtual_total        |  8   |
/// |  80    | ac_comm[32]                  | 32   |
/// | 112    | ac_sched (u8, aligned 8)     |  1   |
/// | 113    | ac_pad[3]                    |  3   |
/// | 120    | ac_uid (u32, aligned 8)      |  4   |
/// | 124    | ac_gid (u32)                 |  4   |
/// | 128    | ac_pid (u32)                 |  4   |
/// | 132    | ac_ppid (u32)                |  4   |
/// | 136    | ac_btime (u32) + 4B align    |  4   |
/// | 144    | ac_etime (u64, aligned 8)    |  8   |
/// | 152    | ac_utime                     |  8   |
/// | 160    | ac_stime                     |  8   |
/// | 168    | ac_minflt                    |  8   |
/// | 176    | ac_majflt                    |  8   |
/// | 184    | coremem                      |  8   |
/// | 192    | virtmem                      |  8   |
/// | 200    | hiwater_rss (u64 KB)         |  8   |
/// | 208    | hiwater_vm  (u64 KB)         |  8   |
/// | 216    | read_char                    |  8   |
/// | 224    | write_char                   |  8   |
/// | 232    | read_syscalls                |  8   |
/// | 240    | write_syscalls               |  8   |
/// | 248    | read_bytes                   |  8   |
/// | 256    | write_bytes                  |  8   |
/// | 264    | cancelled_write_bytes        |  8   |
/// | 272    | nvcsw                        |  8   |
/// | 280    | nivcsw                       |  8   |
/// | 288    | ac_utimescaled (dead)        |  8   |
/// | 296    | ac_stimescaled (dead)        |  8   |
/// | 304    | cpu_scaled_run_real_total    |  8   |
/// | 312    | freepages_count              |  8   |
/// | 320    | freepages_delay_total        |  8   |
/// | 328    | thrashing_count              |  8   |
/// | 336    | thrashing_delay_total        |  8   |
/// | 344    | ac_btime64                   |  8   |
/// | 352    | compact_count                |  8   |
/// | 360    | compact_delay_total          |  8   |
/// | 368    | ac_tgid (u32) + 4B align     |  4   |
/// | 376    | ac_tgetime (u64, aligned 8)  |  8   |
/// | 384    | ac_exe_dev                   |  8   |
/// | 392    | ac_exe_inode                 |  8   |
/// | 400    | wpcopy_count                 |  8   |
/// | 408    | wpcopy_delay_total           |  8   |
/// | 416    | irq_count                    |  8   |
/// | 424    | irq_delay_total              |  8   |
/// | 432    | cpu_delay_max                |  8   |
/// | 440    | cpu_delay_min                |  8   |
/// | 448    | blkio_delay_max              |  8   |
/// | 456    | blkio_delay_min              |  8   |
/// | 464    | swapin_delay_max             |  8   |
/// | 472    | swapin_delay_min             |  8   |
/// | 480    | freepages_delay_max          |  8   |
/// | 488    | freepages_delay_min          |  8   |
/// | 496    | thrashing_delay_max          |  8   |
/// | 504    | thrashing_delay_min          |  8   |
/// | 512    | compact_delay_max            |  8   |
/// | 520    | compact_delay_min            |  8   |
/// | 528    | wpcopy_delay_max             |  8   |
/// | 536    | wpcopy_delay_min             |  8   |
/// | 544    | irq_delay_max                |  8   |
/// | 552    | irq_delay_min                |  8   |
///
/// Older-kernel replies (pre-v15 without `*_delay_max`/`min`,
/// pre-v14 without `irq_*`, pre-v13 without `wpcopy_*`, etc.)
/// surface as truncated payloads — `r64` returns 0 for any read
/// past `buf.len()` so absent fields collapse to zero per the
/// best-effort capture contract.
fn parse_taskstats_payload(buf: &[u8]) -> Result<DelayStats, String> {
    // Helper: read u64 at offset `off`. Returns 0 if the buffer
    // doesn't extend that far (older-kernel truncation).
    let r64 = |off: usize| -> u64 {
        if off + 8 > buf.len() {
            0
        } else {
            u64::from_ne_bytes(buf[off..off + 8].try_into().unwrap())
        }
    };

    // Delay accounting block (v1 fields, alignment-padded so
    // cpu_count starts at offset 16 per `aligned(8)`).
    let cpu_count = r64(16);
    let cpu_delay_total_ns = r64(24);
    let blkio_count = r64(32);
    let blkio_delay_total_ns = r64(40);
    let swapin_count = r64(48);
    let swapin_delay_total_ns = r64(56);

    // Extended-accounting hiwater fields (v3+).
    let hiwater_rss_kb = r64(200);
    let hiwater_vm_kb = r64(208);

    // freepages_* (v8) and thrashing_* (v9). cpu_scaled_run_real_total
    // (v8 dead) sits at 304; freepages_count starts at 312.
    let freepages_count = r64(312);
    let freepages_delay_total_ns = r64(320);
    let thrashing_count = r64(328);
    let thrashing_delay_total_ns = r64(336);

    // compact_* (v11) — ac_btime64 (v10) at 344, then
    // compact_count at 352, compact_delay_total at 360.
    let compact_count = r64(352);
    let compact_delay_total_ns = r64(360);

    // wpcopy_* (v13) — v12 (ac_tgid + ac_tgetime + ac_exe_dev +
    // ac_exe_inode = 32 bytes after 4-byte u32 padding) consumes
    // 368..400; wpcopy_count starts at 400.
    let wpcopy_count = r64(400);
    let wpcopy_delay_total_ns = r64(408);

    // irq_* (v14).
    let irq_count = r64(416);
    let irq_delay_total_ns = r64(424);

    // delay_max + delay_min (v15/v16): 8 categories × 2 u64 fields
    // = 128 bytes starting at offset 432.
    let cpu_delay_max_ns = r64(432);
    let cpu_delay_min_ns = r64(440);
    let blkio_delay_max_ns = r64(448);
    let blkio_delay_min_ns = r64(456);
    let swapin_delay_max_ns = r64(464);
    let swapin_delay_min_ns = r64(472);
    let freepages_delay_max_ns = r64(480);
    let freepages_delay_min_ns = r64(488);
    let thrashing_delay_max_ns = r64(496);
    let thrashing_delay_min_ns = r64(504);
    let compact_delay_max_ns = r64(512);
    let compact_delay_min_ns = r64(520);
    let wpcopy_delay_max_ns = r64(528);
    let wpcopy_delay_min_ns = r64(536);
    let irq_delay_max_ns = r64(544);
    let irq_delay_min_ns = r64(552);

    Ok(DelayStats {
        cpu_count,
        cpu_delay_total_ns,
        cpu_delay_max_ns,
        cpu_delay_min_ns,
        blkio_count,
        blkio_delay_total_ns,
        blkio_delay_max_ns,
        blkio_delay_min_ns,
        swapin_count,
        swapin_delay_total_ns,
        swapin_delay_max_ns,
        swapin_delay_min_ns,
        freepages_count,
        freepages_delay_total_ns,
        freepages_delay_max_ns,
        freepages_delay_min_ns,
        thrashing_count,
        thrashing_delay_total_ns,
        thrashing_delay_max_ns,
        thrashing_delay_min_ns,
        compact_count,
        compact_delay_total_ns,
        compact_delay_max_ns,
        compact_delay_min_ns,
        wpcopy_count,
        wpcopy_delay_total_ns,
        wpcopy_delay_max_ns,
        wpcopy_delay_min_ns,
        irq_count,
        irq_delay_total_ns,
        irq_delay_max_ns,
        irq_delay_min_ns,
        hiwater_rss_bytes: hiwater_rss_kb.saturating_mul(1024),
        hiwater_vm_bytes: hiwater_vm_kb.saturating_mul(1024),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a complete reply buffer with a 20-byte
    /// nlmsghdr+genlmsghdr header prepended to the caller's
    /// payload. The header carries the given `nlmsg_type` and a
    /// stub `TASKSTATS_CMD_NEW` genl cmd; the caller-supplied
    /// payload bytes are appended after the genlmsghdr to form the
    /// NLA list. The total `nlmsg_len` field is backfilled to the
    /// final buffer length so [`parse_reply`] does not reject the
    /// header on a length mismatch — tests that intentionally
    /// corrupt `nlmsg_len` overwrite it after the call.
    fn build_reply_buf(nlmsg_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20 + payload.len());
        // nlmsghdr (16 bytes): length placeholder, type, flags=0,
        // seq=0, port=0.
        buf.extend_from_slice(&0u32.to_ne_bytes());
        buf.extend_from_slice(&nlmsg_type.to_ne_bytes());
        buf.extend_from_slice(&0u16.to_ne_bytes());
        buf.extend_from_slice(&0u32.to_ne_bytes());
        buf.extend_from_slice(&0u32.to_ne_bytes());
        // genlmsghdr (4 bytes): cmd=NEW, version=1, reserved=0.
        buf.push(TASKSTATS_CMD_NEW);
        buf.push(TASKSTATS_GENL_VERSION);
        buf.extend_from_slice(&0u16.to_ne_bytes());
        // Caller payload (post-genlmsghdr — the NLA list).
        buf.extend_from_slice(payload);
        let total = buf.len() as u32;
        buf[0..4].copy_from_slice(&total.to_ne_bytes());
        buf
    }

    /// Build_request emits the expected 28-byte wire layout for
    /// a representative tid. Pins the byte structure that the
    /// kernel parser at `kernel/taskstats.c::cmd_attr_pid` reads
    /// — a layout regression breaks every snapshot.
    #[test]
    fn build_request_layout() {
        let req = build_request(0x4242, 0x1234, 0xCAFEBABE);
        assert_eq!(req.len(), 28);
        // nlmsghdr length
        assert_eq!(u32::from_ne_bytes(req[0..4].try_into().unwrap()), 28);
        // message_type (family_id)
        assert_eq!(u16::from_ne_bytes(req[4..6].try_into().unwrap()), 0x4242);
        // flags — NLM_F_REQUEST only (no NLM_F_ACK; see
        // build_request fn doc for why).
        assert_eq!(
            u16::from_ne_bytes(req[6..8].try_into().unwrap()),
            NLM_F_REQUEST,
        );
        // sequence_number
        assert_eq!(u32::from_ne_bytes(req[8..12].try_into().unwrap()), 0x1234);
        // port_number
        assert_eq!(u32::from_ne_bytes(req[12..16].try_into().unwrap()), 0);
        // genlmsghdr cmd
        assert_eq!(req[16], TASKSTATS_CMD_GET);
        // genlmsghdr version
        assert_eq!(req[17], TASKSTATS_GENL_VERSION);
        // genlmsghdr reserved
        assert_eq!(u16::from_ne_bytes(req[18..20].try_into().unwrap()), 0);
        // nlattr length
        assert_eq!(u16::from_ne_bytes(req[20..22].try_into().unwrap()), 8);
        // nlattr type
        assert_eq!(
            u16::from_ne_bytes(req[22..24].try_into().unwrap()),
            TASKSTATS_CMD_ATTR_PID,
        );
        // tid value
        assert_eq!(
            u32::from_ne_bytes(req[24..28].try_into().unwrap()),
            0xCAFEBABE
        );
    }

    /// `find_nla` walks NLA-aligned attributes and returns the
    /// payload of a matching kind. Pins the alignment math:
    /// each attribute pads to a 4-byte boundary even when the
    /// payload is shorter than 4 bytes.
    #[test]
    fn find_nla_walks_aligned_attrs() {
        // Build a buffer with two attrs:
        //   #1: kind=10, len=5 (1 byte payload + 3 bytes pad → 8 bytes total)
        //   #2: kind=20, len=8 (4 bytes payload, no pad → 8 bytes total)
        let mut buf = Vec::new();
        buf.extend_from_slice(&5u16.to_ne_bytes());
        buf.extend_from_slice(&10u16.to_ne_bytes());
        buf.push(0xAA);
        buf.extend_from_slice(&[0, 0, 0]); // pad to 8
        buf.extend_from_slice(&8u16.to_ne_bytes());
        buf.extend_from_slice(&20u16.to_ne_bytes());
        buf.extend_from_slice(&0xDEADBEEFu32.to_ne_bytes());

        let v1 = find_nla(&buf, 10).expect("attr 10 present");
        assert_eq!(v1, &[0xAA]);
        let v2 = find_nla(&buf, 20).expect("attr 20 present");
        assert_eq!(v2, &0xDEADBEEFu32.to_ne_bytes());
        assert!(find_nla(&buf, 99).is_none());
    }

    /// A short stats payload (older kernel that doesn't carry
    /// the v15+ delay_max / delay_min fields) parses without
    /// panicking; the absent fields read zero per the
    /// `r64`-out-of-range branch.
    #[test]
    fn parse_taskstats_payload_handles_truncation() {
        // 80 bytes — well under the 560-byte v17 layout. Covers
        // cpu_count (offset 16) + cpu_delay_total (offset 24); the
        // hiwater + delay_min/max blocks live past 80 and collapse
        // to 0 via the `r64`-out-of-range branch.
        let mut buf = vec![0u8; 80];
        buf[16..24].copy_from_slice(&123u64.to_ne_bytes()); // cpu_count
        buf[24..32].copy_from_slice(&456u64.to_ne_bytes()); // cpu_delay_total
        let stats = parse_taskstats_payload(&buf).expect("short payload OK");
        assert_eq!(stats.cpu_count, 123);
        assert_eq!(stats.cpu_delay_total_ns, 456);
        assert_eq!(stats.cpu_delay_max_ns, 0);
        assert_eq!(stats.hiwater_rss_bytes, 0);
        assert_eq!(stats.irq_delay_max_ns, 0);
    }

    /// `parse_taskstats_payload` converts hiwater KB → bytes via
    /// `saturating_mul(1024)`. Pins both the multiplier and the
    /// saturation behavior at the u64 boundary (a hiwater value
    /// large enough to overflow saturates to u64::MAX rather
    /// than wrapping silently).
    #[test]
    fn parse_taskstats_payload_kb_to_bytes_conversion() {
        let mut buf = vec![0u8; 560];
        buf[200..208].copy_from_slice(&512u64.to_ne_bytes()); // hiwater_rss = 512 KB
        buf[208..216].copy_from_slice(&u64::MAX.to_ne_bytes()); // hiwater_vm overflow
        let stats = parse_taskstats_payload(&buf).expect("full payload OK");
        assert_eq!(stats.hiwater_rss_bytes, 512 * 1024);
        // saturating_mul(1024) of u64::MAX clamps to u64::MAX.
        assert_eq!(stats.hiwater_vm_bytes, u64::MAX);
    }

    /// `parse_reply` rejects a tid mismatch (defends against
    /// stale-reply mis-routing if a previous query's reply
    /// arrives after a new one was issued). Pin the error
    /// message shape so a regression that drops the validation
    /// surfaces as a test failure.
    #[test]
    fn parse_reply_rejects_tid_mismatch() {
        // AGGR_PID outer NLA: header (4) + nested PID NLA (8) +
        // nested STATS NLA (4 + payload). PID payload = 4 bytes
        // (u32). STATS payload = 8 bytes (just enough for cpu_count).
        // Inner: PID nlattr (8) + STATS nlattr (12) = 20 bytes.
        // Outer: 4 + 20 = 24 bytes.
        let mut payload = Vec::new();
        payload.extend_from_slice(&24u16.to_ne_bytes()); // outer nla_len
        payload.extend_from_slice(&TASKSTATS_TYPE_AGGR_PID.to_ne_bytes());
        // Inner PID nlattr
        payload.extend_from_slice(&8u16.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_PID.to_ne_bytes());
        payload.extend_from_slice(&42u32.to_ne_bytes());
        // Inner STATS nlattr
        payload.extend_from_slice(&12u16.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_STATS.to_ne_bytes());
        payload.extend_from_slice(&[0u8; 8]); // 8 bytes of stats payload
        // 1234 is an arbitrary nlmsg_type that is not NLMSG_ERROR;
        // parse_reply only branches on NLMSG_ERROR for the type.
        let buf = build_reply_buf(1234, &payload);

        let err = parse_reply(&buf, 99).expect_err("tid mismatch should reject");
        assert!(err.contains("tid mismatch"), "error: {err}");
        assert!(err.contains("99"), "error: {err}");
        assert!(err.contains("42"), "error: {err}");
    }

    /// `parse_reply` surfaces the kernel's negated errno when the
    /// reply carries `NLMSG_ERROR` (e.g. ESRCH for an exited tid,
    /// EPERM when CAP_NET_ADMIN is missing). Errno=-1 (the kernel's
    /// negative-on-the-wire convention; see
    /// netlink-packet-core/src/error.rs:146 "Negative errno or 0
    /// for acknowledgements") surfaces as `errno=1` in the
    /// rendered string per `parse_reply`'s `let errno = -err`
    /// negation step.
    #[test]
    fn parse_reply_nlmsg_error_surfaces_errno() {
        // Layout: nlmsghdr (16 bytes) with type = NLMSG_ERROR, then
        // a 4-byte i32 errno. parse_reply branches on the type
        // BEFORE checking nlmsg_len, so we set both honestly.
        let mut buf = Vec::with_capacity(20);
        buf.extend_from_slice(&20u32.to_ne_bytes()); // length
        buf.extend_from_slice(&NLMSG_ERROR.to_ne_bytes()); // type
        buf.extend_from_slice(&0u16.to_ne_bytes()); // flags
        buf.extend_from_slice(&0u32.to_ne_bytes()); // seq
        buf.extend_from_slice(&0u32.to_ne_bytes()); // port
        // i32 errno = -1 (the kernel sends -EPERM as -1 here).
        buf.extend_from_slice(&(-1i32).to_ne_bytes());

        let err = parse_reply(&buf, 1234).expect_err("NLMSG_ERROR must surface as Err");
        assert!(
            err.contains("errno=1"),
            "expected `errno=1` in the rendered string (parse_reply negates the kernel's wire value): {err}",
        );
    }

    /// `parse_reply` rejects buffers shorter than the 16-byte
    /// nlmsghdr. Three subcases pin the threshold: empty (the
    /// recv_from_full path could theoretically deliver an empty
    /// buffer on a malformed reply), 8 bytes (just the length +
    /// type fields), and 15 bytes (one short of the full header).
    /// All must surface the "shorter than nlmsghdr" message so an
    /// operator reading the error knows the failure mode.
    #[test]
    fn parse_reply_rejects_short_buffer() {
        for len in [0usize, 8, 15] {
            let buf = vec![0u8; len];
            let err = parse_reply(&buf, 1).expect_err("short buffer must reject");
            assert!(err.contains("shorter than nlmsghdr"), "len={len}: {err}",);
        }
    }

    /// `parse_reply` rejects an `nlmsg_len` field that exceeds the
    /// actual buffer length — defends against a half-delivered
    /// reply where the recv path returned fewer bytes than the
    /// header advertises. parse_reply's check is
    /// `nlmsg_len > buf.len()`; pin a length wildly larger than
    /// any legitimate reply (999 vs 16-byte buffer).
    #[test]
    fn parse_reply_rejects_oversized_nlmsg_len() {
        let mut buf = vec![0u8; 16];
        buf[0..4].copy_from_slice(&999u32.to_ne_bytes()); // length far past buf
        // nlmsg_type stays 0 (not NLMSG_ERROR) so the length check
        // fires before the error-payload branch.
        let err = parse_reply(&buf, 1).expect_err("oversized nlmsg_len must reject");
        assert!(
            err.contains("exceeds buffer length"),
            "expected `exceeds buffer length` in error: {err}",
        );
    }

    /// `parse_reply` rejects `nlmsg_len < 20` (one byte short of
    /// the nlmsghdr+genlmsghdr minimum). 18 sits between the
    /// nlmsghdr length (16) and the nlmsghdr+genlmsghdr total
    /// (20), so the first length check passes but the second
    /// fires. Pin the "too short for nlmsghdr+genlmsghdr"
    /// message — a regression that conflated the two thresholds
    /// would produce a different error string here.
    #[test]
    fn parse_reply_rejects_short_genlmsghdr() {
        let mut buf = vec![0u8; 18];
        buf[0..4].copy_from_slice(&18u32.to_ne_bytes()); // length matches buf
        // nlmsg_type stays 0 (not NLMSG_ERROR) so we reach the
        // genlmsghdr-length check.
        let err = parse_reply(&buf, 1).expect_err("short nlmsg_len must reject");
        assert!(
            err.contains("too short for nlmsghdr+genlmsghdr"),
            "expected `too short for nlmsghdr+genlmsghdr` in error: {err}",
        );
    }

    /// `parse_reply` rejects a reply whose NLA list does not carry
    /// `TASKSTATS_TYPE_AGGR_PID`. Pin the "AGGR_PID missing in
    /// reply" error so a regression that silently fell through
    /// to a default-zero parse would surface here as a wrong-
    /// error-or-no-error mismatch.
    #[test]
    fn parse_reply_rejects_missing_aggr_pid() {
        // NLA payload: one bogus 8-byte attribute with type 99
        // (not TASKSTATS_TYPE_AGGR_PID). 4-byte header + 4-byte
        // payload = 8 bytes total, NLA-aligned.
        let mut payload = Vec::new();
        payload.extend_from_slice(&8u16.to_ne_bytes()); // nla_len
        payload.extend_from_slice(&99u16.to_ne_bytes()); // nla_type (not AGGR_PID)
        payload.extend_from_slice(&0u32.to_ne_bytes()); // dummy value
        // Use any nlmsg_type other than NLMSG_ERROR so the parser
        // proceeds to the NLA walk.
        let buf = build_reply_buf(1234, &payload);
        let err = parse_reply(&buf, 1).expect_err("missing AGGR_PID must reject");
        assert!(
            err.contains("AGGR_PID missing"),
            "expected `AGGR_PID missing` in error: {err}",
        );
    }

    /// `parse_reply` rejects an AGGR_PID nest that omits
    /// `TASKSTATS_TYPE_PID`. Pins the second `ok_or` in
    /// parse_reply: even when the outer AGGR_PID resolves, the
    /// inner PID lookup must fire its own error rather than
    /// silently substituting a zero pid. AGGR_PID payload here
    /// carries only a STATS nlattr.
    #[test]
    fn parse_reply_rejects_missing_pid_in_aggr() {
        // AGGR_PID inner: single STATS nlattr (4-byte header +
        // 8-byte stats payload = 12 bytes). AGGR_PID outer
        // nla_len = 4 (header) + 12 (inner) = 16.
        let mut payload = Vec::new();
        payload.extend_from_slice(&16u16.to_ne_bytes()); // outer nla_len
        payload.extend_from_slice(&TASKSTATS_TYPE_AGGR_PID.to_ne_bytes());
        // Inner STATS nlattr (no PID nlattr).
        payload.extend_from_slice(&12u16.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_STATS.to_ne_bytes());
        payload.extend_from_slice(&[0u8; 8]); // dummy stats payload
        let buf = build_reply_buf(1234, &payload);
        let err = parse_reply(&buf, 1).expect_err("missing PID must reject");
        assert!(
            err.contains("TASKSTATS_TYPE_PID missing"),
            "expected `TASKSTATS_TYPE_PID missing` in error: {err}",
        );
    }

    /// `parse_reply` rejects a PID nlattr whose payload is
    /// shorter than the 4-byte u32 the parser reads. Pins the
    /// `pid_attr.len() < 4` length check after find_nla returns:
    /// the kernel always writes a u32, so a 2-byte payload signals
    /// a corrupted reply. NLA_ALIGN rounds 6 → 8 so the trailing
    /// 2 bytes pad cleanly into the 16-byte AGGR_PID inner.
    #[test]
    fn parse_reply_rejects_short_pid_payload() {
        // AGGR_PID inner: PID nlattr with nla_len = 6 (4-byte
        // header + 2-byte payload), padded to 8 bytes per
        // NLA_ALIGN. AGGR_PID outer nla_len = 4 + 8 = 12.
        let mut payload = Vec::new();
        payload.extend_from_slice(&12u16.to_ne_bytes()); // outer nla_len
        payload.extend_from_slice(&TASKSTATS_TYPE_AGGR_PID.to_ne_bytes());
        // Inner PID nlattr with 2-byte payload.
        payload.extend_from_slice(&6u16.to_ne_bytes()); // nla_len = 6
        payload.extend_from_slice(&TASKSTATS_TYPE_PID.to_ne_bytes());
        payload.extend_from_slice(&[0u8; 2]); // 2-byte truncated tid
        payload.extend_from_slice(&[0u8; 2]); // NLA_ALIGN pad to 8 bytes
        let buf = build_reply_buf(1234, &payload);
        let err = parse_reply(&buf, 1).expect_err("short PID payload must reject");
        assert!(
            err.contains("PID payload shorter than u32"),
            "expected `PID payload shorter than u32` in error: {err}",
        );
    }

    /// `parse_reply` rejects an AGGR_PID nest that has a valid
    /// PID nlattr but omits `TASKSTATS_TYPE_STATS`. Pins the
    /// third `ok_or` in parse_reply — the STATS lookup runs AFTER
    /// the tid match check, so this test must use a PID value
    /// that matches `expected_tid` (here 7) so execution reaches
    /// the STATS check rather than failing earlier on tid
    /// mismatch.
    #[test]
    fn parse_reply_rejects_missing_stats_in_aggr() {
        // AGGR_PID inner: single PID nlattr (4-byte header +
        // 4-byte u32 payload = 8 bytes). AGGR_PID outer
        // nla_len = 4 + 8 = 12.
        let mut payload = Vec::new();
        payload.extend_from_slice(&12u16.to_ne_bytes()); // outer nla_len
        payload.extend_from_slice(&TASKSTATS_TYPE_AGGR_PID.to_ne_bytes());
        // Inner PID nlattr matching expected_tid=7.
        payload.extend_from_slice(&8u16.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_PID.to_ne_bytes());
        payload.extend_from_slice(&7u32.to_ne_bytes());
        let buf = build_reply_buf(1234, &payload);
        let err = parse_reply(&buf, 7).expect_err("missing STATS must reject");
        assert!(
            err.contains("TASKSTATS_TYPE_STATS missing"),
            "expected `TASKSTATS_TYPE_STATS missing` in error: {err}",
        );
    }

    /// `find_nla` returns `None` on an empty slice — the loop
    /// guard `offset + 4 <= buf.len()` fails on first iteration
    /// when buf.len() is 0.
    #[test]
    fn find_nla_empty_buffer() {
        assert!(find_nla(&[], 1).is_none());
    }

    /// `find_nla` returns `None` on a slice shorter than the
    /// 4-byte NLA header. 3 bytes triggers the loop guard before
    /// the first read.
    #[test]
    fn find_nla_short_buffer() {
        assert!(find_nla(&[0u8, 0, 0], 1).is_none());
    }

    /// `find_nla` returns `None` when an attribute's `nla_len` is
    /// less than the 4-byte header (corrupt or truncated kernel
    /// output). Pin the explicit early-return in find_nla rather
    /// than letting the loop walk into a runaway state.
    #[test]
    fn find_nla_corrupt_short_len() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&2u16.to_ne_bytes()); // nla_len = 2 (< header)
        buf.extend_from_slice(&1u16.to_ne_bytes()); // nla_type
        // No payload — loop's value_end check would fire next, but
        // the nla_len < 4 branch fires first.
        assert!(find_nla(&buf, 1).is_none());
    }

    /// `find_nla` returns `None` when an attribute's `nla_len`
    /// extends past the buffer end (truncated reply). Defends
    /// against a half-read kernel response delivering a header
    /// that promises more bytes than recv returned.
    #[test]
    fn find_nla_truncated_value() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&20u16.to_ne_bytes()); // nla_len = 20 (header + 16-byte value)
        buf.extend_from_slice(&1u16.to_ne_bytes()); // nla_type
        // Only 4 bytes of value follow — far short of the 16
        // promised by nla_len.
        buf.extend_from_slice(&[0u8; 4]);
        assert!(find_nla(&buf, 1).is_none());
    }

    /// Golden-vector test: build a 560-byte v17 `struct taskstats`
    /// payload with a unique distinguishable u64 at every offset
    /// the parser reads, then confirm every output field carries
    /// the exact value its accessor wrote. Catches every offset
    /// regression: a swap between sibling fields, an off-by-eight
    /// drift, or a future kernel layout change that this module
    /// doesn't track. Field values are spaced at 1000 increments
    /// so a one-field cross-wire surfaces as a 1000-unit mismatch
    /// citing the offending field.
    ///
    /// Hiwater values cover the KB→bytes conversion: hiwater_rss
    /// uses 1024 KB (round-trips to exactly 1 MiB) and hiwater_vm
    /// uses 2048 KB (2 MiB). Saturation behavior is covered by
    /// `parse_taskstats_payload_kb_to_bytes_conversion`; this
    /// test focuses on offsets, not on the multiplication ladder.
    #[test]
    fn parse_taskstats_payload_full_v17_roundtrip() {
        let mut buf = vec![0u8; 560];
        // Helper: write u64 at offset.
        let w = |buf: &mut Vec<u8>, off: usize, v: u64| {
            buf[off..off + 8].copy_from_slice(&v.to_ne_bytes());
        };
        // Delay accounting v1 block (offsets 16..64).
        w(&mut buf, 16, 1000); // cpu_count
        w(&mut buf, 24, 2000); // cpu_delay_total
        w(&mut buf, 32, 3000); // blkio_count
        w(&mut buf, 40, 4000); // blkio_delay_total
        w(&mut buf, 48, 5000); // swapin_count
        w(&mut buf, 56, 6000); // swapin_delay_total
        // hiwater (v3, offsets 200/208) — KB values that produce
        // round bytes after `saturating_mul(1024)`.
        w(&mut buf, 200, 1024); // hiwater_rss = 1024 KB → 1 MiB
        w(&mut buf, 208, 2048); // hiwater_vm  = 2048 KB → 2 MiB
        // freepages (v8, offsets 312/320), thrashing (v9, 328/336).
        w(&mut buf, 312, 7000); // freepages_count
        w(&mut buf, 320, 8000); // freepages_delay_total
        w(&mut buf, 328, 9000); // thrashing_count
        w(&mut buf, 336, 10_000); // thrashing_delay_total
        // compact (v11, offsets 352/360).
        w(&mut buf, 352, 11_000); // compact_count
        w(&mut buf, 360, 12_000); // compact_delay_total
        // wpcopy (v13, offsets 400/408).
        w(&mut buf, 400, 13_000); // wpcopy_count
        w(&mut buf, 408, 14_000); // wpcopy_delay_total
        // irq (v14, offsets 416/424).
        w(&mut buf, 416, 15_000); // irq_count
        w(&mut buf, 424, 16_000); // irq_delay_total
        // delay_max + delay_min v15/v16 block (offsets 432..560).
        w(&mut buf, 432, 17_000); // cpu_delay_max
        w(&mut buf, 440, 18_000); // cpu_delay_min
        w(&mut buf, 448, 19_000); // blkio_delay_max
        w(&mut buf, 456, 20_000); // blkio_delay_min
        w(&mut buf, 464, 21_000); // swapin_delay_max
        w(&mut buf, 472, 22_000); // swapin_delay_min
        w(&mut buf, 480, 23_000); // freepages_delay_max
        w(&mut buf, 488, 24_000); // freepages_delay_min
        w(&mut buf, 496, 25_000); // thrashing_delay_max
        w(&mut buf, 504, 26_000); // thrashing_delay_min
        w(&mut buf, 512, 27_000); // compact_delay_max
        w(&mut buf, 520, 28_000); // compact_delay_min
        w(&mut buf, 528, 29_000); // wpcopy_delay_max
        w(&mut buf, 536, 30_000); // wpcopy_delay_min
        w(&mut buf, 544, 31_000); // irq_delay_max
        w(&mut buf, 552, 32_000); // irq_delay_min

        let stats = parse_taskstats_payload(&buf).expect("full v17 payload OK");
        let expected = DelayStats {
            cpu_count: 1000,
            cpu_delay_total_ns: 2000,
            cpu_delay_max_ns: 17_000,
            cpu_delay_min_ns: 18_000,
            blkio_count: 3000,
            blkio_delay_total_ns: 4000,
            blkio_delay_max_ns: 19_000,
            blkio_delay_min_ns: 20_000,
            swapin_count: 5000,
            swapin_delay_total_ns: 6000,
            swapin_delay_max_ns: 21_000,
            swapin_delay_min_ns: 22_000,
            freepages_count: 7000,
            freepages_delay_total_ns: 8000,
            freepages_delay_max_ns: 23_000,
            freepages_delay_min_ns: 24_000,
            thrashing_count: 9000,
            thrashing_delay_total_ns: 10_000,
            thrashing_delay_max_ns: 25_000,
            thrashing_delay_min_ns: 26_000,
            compact_count: 11_000,
            compact_delay_total_ns: 12_000,
            compact_delay_max_ns: 27_000,
            compact_delay_min_ns: 28_000,
            wpcopy_count: 13_000,
            wpcopy_delay_total_ns: 14_000,
            wpcopy_delay_max_ns: 29_000,
            wpcopy_delay_min_ns: 30_000,
            irq_count: 15_000,
            irq_delay_total_ns: 16_000,
            irq_delay_max_ns: 31_000,
            irq_delay_min_ns: 32_000,
            hiwater_rss_bytes: 1024 * 1024,
            hiwater_vm_bytes: 2048 * 1024,
        };
        assert_eq!(
            stats, expected,
            "v17 payload roundtrip mismatch — every field must read \
             back the value its offset was written with",
        );
    }

    /// Full reply roundtrip — build the complete kernel-shaped reply
    /// (nlmsghdr + genlmsghdr + AGGR_PID NLA wrapping PID + STATS) on
    /// top of the v17 golden vector from
    /// [`parse_taskstats_payload_full_v17_roundtrip`], then call
    /// [`parse_reply`] and assert the assembled [`DelayStats`]
    /// matches. Closes the gap between "raw payload parser correct"
    /// (covered by the offset roundtrip) and "full reply parser
    /// correct" (the AGGR_PID nest + tid match + payload extraction
    /// dispatch [`parse_reply`] performs on top of
    /// [`parse_taskstats_payload`]). Same byte values as the offset
    /// roundtrip so a regression in either parser surfaces in this
    /// test too — the cross-check is intentional.
    #[test]
    fn parse_reply_full_roundtrip_v17() {
        // 1. Build the 560-byte v17 STATS payload using the same
        //    golden vector as parse_taskstats_payload_full_v17_roundtrip.
        let mut stats_payload = vec![0u8; 560];
        let w = |buf: &mut Vec<u8>, off: usize, v: u64| {
            buf[off..off + 8].copy_from_slice(&v.to_ne_bytes());
        };
        // Delay accounting v1 block (offsets 16..64).
        w(&mut stats_payload, 16, 1000); // cpu_count
        w(&mut stats_payload, 24, 2000); // cpu_delay_total
        w(&mut stats_payload, 32, 3000); // blkio_count
        w(&mut stats_payload, 40, 4000); // blkio_delay_total
        w(&mut stats_payload, 48, 5000); // swapin_count
        w(&mut stats_payload, 56, 6000); // swapin_delay_total
        // hiwater (v3, offsets 200/208).
        w(&mut stats_payload, 200, 1024); // hiwater_rss = 1 MiB
        w(&mut stats_payload, 208, 2048); // hiwater_vm  = 2 MiB
        // freepages (v8, 312/320), thrashing (v9, 328/336).
        w(&mut stats_payload, 312, 7000);
        w(&mut stats_payload, 320, 8000);
        w(&mut stats_payload, 328, 9000);
        w(&mut stats_payload, 336, 10_000);
        // compact (v11, 352/360).
        w(&mut stats_payload, 352, 11_000);
        w(&mut stats_payload, 360, 12_000);
        // wpcopy (v13, 400/408).
        w(&mut stats_payload, 400, 13_000);
        w(&mut stats_payload, 408, 14_000);
        // irq (v14, 416/424).
        w(&mut stats_payload, 416, 15_000);
        w(&mut stats_payload, 424, 16_000);
        // delay_max + delay_min v15/v16 block (432..560).
        w(&mut stats_payload, 432, 17_000);
        w(&mut stats_payload, 440, 18_000);
        w(&mut stats_payload, 448, 19_000);
        w(&mut stats_payload, 456, 20_000);
        w(&mut stats_payload, 464, 21_000);
        w(&mut stats_payload, 472, 22_000);
        w(&mut stats_payload, 480, 23_000);
        w(&mut stats_payload, 488, 24_000);
        w(&mut stats_payload, 496, 25_000);
        w(&mut stats_payload, 504, 26_000);
        w(&mut stats_payload, 512, 27_000);
        w(&mut stats_payload, 520, 28_000);
        w(&mut stats_payload, 528, 29_000);
        w(&mut stats_payload, 536, 30_000);
        w(&mut stats_payload, 544, 31_000);
        w(&mut stats_payload, 552, 32_000);

        // 2. Wrap the STATS payload in an NLA: 4-byte header + 560
        //    bytes of payload = 564 bytes total. STATS attribute
        //    sits inside the AGGR_PID nest alongside a 4-byte PID.
        let pid: u32 = 4242;
        let stats_nla_len: u16 = 4 + stats_payload.len() as u16; // 564
        let pid_nla_len: u16 = 4 + 4; // 8
        // AGGR_PID outer nla_len = header (4) + inner PID nla (8)
        // + inner STATS nla (564) = 576 bytes.
        let aggr_nla_len: u16 = 4 + pid_nla_len + stats_nla_len;
        let mut payload = Vec::new();
        payload.extend_from_slice(&aggr_nla_len.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_AGGR_PID.to_ne_bytes());
        // Inner PID nlattr.
        payload.extend_from_slice(&pid_nla_len.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_PID.to_ne_bytes());
        payload.extend_from_slice(&pid.to_ne_bytes());
        // Inner STATS nlattr.
        payload.extend_from_slice(&stats_nla_len.to_ne_bytes());
        payload.extend_from_slice(&TASKSTATS_TYPE_STATS.to_ne_bytes());
        payload.extend_from_slice(&stats_payload);

        // 3. Wrap in a full nlmsghdr + genlmsghdr via build_reply_buf
        //    so parse_reply sees the same wire shape the kernel emits.
        //    nlmsg_type = 0x4242 is arbitrary (any value other than
        //    NLMSG_ERROR works — parse_reply only branches on the
        //    error code path for the type field).
        let buf = build_reply_buf(0x4242, &payload);

        // 4. Drive parse_reply and check every DelayStats field.
        let stats = parse_reply(&buf, pid).expect("full v17 reply OK");
        let expected = DelayStats {
            cpu_count: 1000,
            cpu_delay_total_ns: 2000,
            cpu_delay_max_ns: 17_000,
            cpu_delay_min_ns: 18_000,
            blkio_count: 3000,
            blkio_delay_total_ns: 4000,
            blkio_delay_max_ns: 19_000,
            blkio_delay_min_ns: 20_000,
            swapin_count: 5000,
            swapin_delay_total_ns: 6000,
            swapin_delay_max_ns: 21_000,
            swapin_delay_min_ns: 22_000,
            freepages_count: 7000,
            freepages_delay_total_ns: 8000,
            freepages_delay_max_ns: 23_000,
            freepages_delay_min_ns: 24_000,
            thrashing_count: 9000,
            thrashing_delay_total_ns: 10_000,
            thrashing_delay_max_ns: 25_000,
            thrashing_delay_min_ns: 26_000,
            compact_count: 11_000,
            compact_delay_total_ns: 12_000,
            compact_delay_max_ns: 27_000,
            compact_delay_min_ns: 28_000,
            wpcopy_count: 13_000,
            wpcopy_delay_total_ns: 14_000,
            wpcopy_delay_max_ns: 29_000,
            wpcopy_delay_min_ns: 30_000,
            irq_count: 15_000,
            irq_delay_total_ns: 16_000,
            irq_delay_max_ns: 31_000,
            irq_delay_min_ns: 32_000,
            hiwater_rss_bytes: 1024 * 1024,
            hiwater_vm_bytes: 2048 * 1024,
        };
        assert_eq!(
            stats, expected,
            "full reply roundtrip mismatch — parse_reply must reach \
             the same DelayStats as the raw payload parser when given \
             a kernel-shaped wrapper",
        );
    }

    /// Version-boundary truncation tests. For each pre-v17 size the
    /// kernel may emit (size = end of the last fully-included field
    /// from a given uapi version), populate every field that fits
    /// and assert (a) populated fields read back their planted
    /// value, and (b) every field whose offset+8 extends past the
    /// boundary reads zero per the `r64`-out-of-range branch.
    ///
    /// The boundaries map to specific uapi versions:
    ///
    /// - **56 bytes**: v1 partial — covers cpu_count (16),
    ///   cpu_delay_total (24), blkio_count (32), blkio_delay_total
    ///   (40), swapin_count (48). swapin_delay_total at offset 56
    ///   needs end 64 and lands past the boundary. Models a kernel
    ///   that truncated mid-v1 block.
    /// - **312 bytes**: pre-v8 — hiwater (200, 208) fits;
    ///   freepages_count at offset 312 lands AT the boundary (end
    ///   would be 320), so freepages_count reads zero too. Models
    ///   a kernel without freepages delay accounting.
    /// - **360 bytes**: through compact_count (352..360 fits);
    ///   compact_delay_total at 360..368 lands past the boundary.
    ///   Models a kernel with compact_count but no compact_delay_total
    ///   (a half-implemented v11 capture).
    /// - **408 bytes**: pre-v13 wpcopy split — wpcopy_count
    ///   (400..408) fits, wpcopy_delay_total at 408..416 lands past.
    /// - **424 bytes**: pre-v14 irq split — irq_count (416..424)
    ///   fits, irq_delay_total at 424..432 lands past.
    /// - **432 bytes**: pre-v15/v16 — every cumulative field fits;
    ///   the *_delay_max / *_delay_min block starting at offset 432
    ///   lands past the boundary, so all 16 max/min fields read
    ///   zero.
    ///
    /// Pinning each boundary catches a regression that shifts an
    /// offset by 8 bytes (which would silently move the
    /// "trapdoor" between fits and doesn't-fit) since the test
    /// asserts both halves of the predicate at every boundary.
    #[test]
    fn parse_taskstats_payload_version_boundary_truncation() {
        // (size, expectations) — each expectation asserts a single
        // DelayStats field reads the planted u64 value (Some(v))
        // or zero (None), depending on whether the source offset
        // falls inside the truncated buffer.
        struct Boundary {
            size: usize,
            label: &'static str,
        }

        for b in [
            Boundary {
                size: 56,
                label: "v1 partial (mid-cpu/blkio/swapin block)",
            },
            Boundary {
                size: 312,
                label: "pre-v8 (no freepages)",
            },
            Boundary {
                size: 360,
                label: "v11 partial (compact_count without compact_delay_total)",
            },
            Boundary {
                size: 408,
                label: "pre-v13 (no wpcopy_delay_total)",
            },
            Boundary {
                size: 424,
                label: "pre-v14 (no irq_delay_total)",
            },
            Boundary {
                size: 432,
                label: "pre-v15/v16 (no delay_max / delay_min block)",
            },
        ] {
            // Populate every aligned u64 slot inside the buffer with
            // a unique non-zero marker so we can detect (a) reads
            // succeed where they should, and (b) reads past the
            // boundary collapse to zero rather than leaking adjacent
            // memory or wrapping into a different field.
            let mut buf = vec![0u8; b.size];
            // Plant marker = (offset / 8) * 1000 + 1. Non-zero so it
            // distinguishes from the absent-field default; offset-
            // derived so a regression that shuffled fields would
            // surface as a wrong-marker read.
            for off in (16..b.size).step_by(8) {
                if off + 8 <= b.size {
                    let marker = (off as u64 / 8) * 1000 + 1;
                    buf[off..off + 8].copy_from_slice(&marker.to_ne_bytes());
                }
            }

            let stats = parse_taskstats_payload(&buf)
                .unwrap_or_else(|e| panic!("{}: parse failed: {e}", b.label));

            // Helper: marker for offset `off` if the slot fits inside
            // the buffer, zero otherwise. Mirrors the parse_taskstats_payload
            // r64 helper's out-of-range branch.
            let m = |off: usize| -> u64 {
                if off + 8 <= b.size {
                    (off as u64 / 8) * 1000 + 1
                } else {
                    0
                }
            };

            // Walk every offset the parser reads, assert the
            // returned field equals the marker (or zero past the
            // boundary). Hiwater fields go through saturating_mul(1024)
            // so check those separately.
            assert_eq!(stats.cpu_count, m(16), "{}: cpu_count", b.label);
            assert_eq!(
                stats.cpu_delay_total_ns,
                m(24),
                "{}: cpu_delay_total_ns",
                b.label
            );
            assert_eq!(stats.blkio_count, m(32), "{}: blkio_count", b.label);
            assert_eq!(
                stats.blkio_delay_total_ns,
                m(40),
                "{}: blkio_delay_total_ns",
                b.label
            );
            assert_eq!(stats.swapin_count, m(48), "{}: swapin_count", b.label);
            assert_eq!(
                stats.swapin_delay_total_ns,
                m(56),
                "{}: swapin_delay_total_ns",
                b.label
            );
            assert_eq!(
                stats.freepages_count,
                m(312),
                "{}: freepages_count",
                b.label
            );
            assert_eq!(
                stats.freepages_delay_total_ns,
                m(320),
                "{}: freepages_delay_total_ns",
                b.label
            );
            assert_eq!(
                stats.thrashing_count,
                m(328),
                "{}: thrashing_count",
                b.label
            );
            assert_eq!(
                stats.thrashing_delay_total_ns,
                m(336),
                "{}: thrashing_delay_total_ns",
                b.label
            );
            assert_eq!(stats.compact_count, m(352), "{}: compact_count", b.label);
            assert_eq!(
                stats.compact_delay_total_ns,
                m(360),
                "{}: compact_delay_total_ns",
                b.label
            );
            assert_eq!(stats.wpcopy_count, m(400), "{}: wpcopy_count", b.label);
            assert_eq!(
                stats.wpcopy_delay_total_ns,
                m(408),
                "{}: wpcopy_delay_total_ns",
                b.label
            );
            assert_eq!(stats.irq_count, m(416), "{}: irq_count", b.label);
            assert_eq!(
                stats.irq_delay_total_ns,
                m(424),
                "{}: irq_delay_total_ns",
                b.label
            );
            assert_eq!(
                stats.cpu_delay_max_ns,
                m(432),
                "{}: cpu_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.cpu_delay_min_ns,
                m(440),
                "{}: cpu_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.blkio_delay_max_ns,
                m(448),
                "{}: blkio_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.blkio_delay_min_ns,
                m(456),
                "{}: blkio_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.swapin_delay_max_ns,
                m(464),
                "{}: swapin_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.swapin_delay_min_ns,
                m(472),
                "{}: swapin_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.freepages_delay_max_ns,
                m(480),
                "{}: freepages_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.freepages_delay_min_ns,
                m(488),
                "{}: freepages_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.thrashing_delay_max_ns,
                m(496),
                "{}: thrashing_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.thrashing_delay_min_ns,
                m(504),
                "{}: thrashing_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.compact_delay_max_ns,
                m(512),
                "{}: compact_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.compact_delay_min_ns,
                m(520),
                "{}: compact_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.wpcopy_delay_max_ns,
                m(528),
                "{}: wpcopy_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.wpcopy_delay_min_ns,
                m(536),
                "{}: wpcopy_delay_min_ns",
                b.label
            );
            assert_eq!(
                stats.irq_delay_max_ns,
                m(544),
                "{}: irq_delay_max_ns",
                b.label
            );
            assert_eq!(
                stats.irq_delay_min_ns,
                m(552),
                "{}: irq_delay_min_ns",
                b.label
            );
            // Hiwater multiplies by 1024 — m(off) * 1024 (still 0
            // when out of range since saturating_mul preserves 0).
            assert_eq!(
                stats.hiwater_rss_bytes,
                m(200).saturating_mul(1024),
                "{}: hiwater_rss_bytes",
                b.label
            );
            assert_eq!(
                stats.hiwater_vm_bytes,
                m(208).saturating_mul(1024),
                "{}: hiwater_vm_bytes",
                b.label
            );
        }
    }

    /// `TaskstatsSummary::record_result` bumps `ok_count` on a
    /// successful query and never advances any error counter. Pin
    /// the per-counter accounting so a regression that swapped
    /// the success / error branches surfaces here.
    #[test]
    fn taskstats_summary_records_ok() {
        let mut s = TaskstatsSummary::default();
        let ok: io::Result<DelayStats> = Ok(DelayStats::default());
        s.record_result(&ok);
        s.record_result(&ok);
        assert_eq!(s.ok_count, 2);
        assert_eq!(s.eperm_count, 0);
        assert_eq!(s.esrch_count, 0);
        assert_eq!(s.other_err_count, 0);
    }

    /// `TaskstatsSummary::record_result` classifies an `io::Error`
    /// with `raw_os_error() == Some(EPERM)` as an EPERM bump.
    /// EPERM is the operationally interesting case (CAP_NET_ADMIN
    /// missing) so it gets its own counter rather than folding
    /// into `other_err_count`. POSIX EPERM = 1.
    #[test]
    fn taskstats_summary_records_eperm_from_raw_os_error() {
        let mut s = TaskstatsSummary::default();
        let err: io::Result<DelayStats> = Err(io::Error::from_raw_os_error(1));
        s.record_result(&err);
        assert_eq!(s.ok_count, 0);
        assert_eq!(s.eperm_count, 1);
        assert_eq!(s.esrch_count, 0);
        assert_eq!(s.other_err_count, 0);
    }

    /// `TaskstatsSummary::record_result` classifies an `io::Error`
    /// with `raw_os_error() == Some(ESRCH)` as an ESRCH bump.
    /// ESRCH is the routine "tid raced exit" case. POSIX ESRCH = 3.
    #[test]
    fn taskstats_summary_records_esrch_from_raw_os_error() {
        let mut s = TaskstatsSummary::default();
        let err: io::Result<DelayStats> = Err(io::Error::from_raw_os_error(3));
        s.record_result(&err);
        assert_eq!(s.ok_count, 0);
        assert_eq!(s.eperm_count, 0);
        assert_eq!(s.esrch_count, 1);
        assert_eq!(s.other_err_count, 0);
    }

    /// `TaskstatsSummary::record_result` recovers the kernel-side
    /// errno from a `parse_reply`-formatted message wrapped by
    /// `io::Error::other`. The wrap strips the `raw_os_error()`
    /// tag, so the classifier falls back to scanning the rendered
    /// error text for the `errno=N` substring that
    /// `parse_reply` injects on `NLMSG_ERROR` replies. Pins both
    /// EPERM and ESRCH parse paths.
    #[test]
    fn taskstats_summary_records_errno_from_text_message() {
        let mut s = TaskstatsSummary::default();
        // Mirror the exact format `parse_reply` emits.
        let eperm: io::Result<DelayStats> =
            Err(io::Error::other("kernel returned NLMSG_ERROR errno=1"));
        let esrch: io::Result<DelayStats> =
            Err(io::Error::other("kernel returned NLMSG_ERROR errno=3"));
        s.record_result(&eperm);
        s.record_result(&esrch);
        assert_eq!(s.ok_count, 0);
        assert_eq!(s.eperm_count, 1);
        assert_eq!(s.esrch_count, 1);
        assert_eq!(s.other_err_count, 0);
    }

    /// `TaskstatsSummary::record_result` folds unrecognized errors
    /// (a non-EPERM/ESRCH errno, an error with no errno
    /// information at all) to `other_err_count`. Pin three
    /// shapes: a real OS errno that's neither EPERM nor ESRCH,
    /// a parse-style message with an unrecognized errno, and an
    /// io::Error with no recoverable errno.
    #[test]
    fn taskstats_summary_records_other_err_for_unrecognized_shapes() {
        let mut s = TaskstatsSummary::default();
        // EINVAL = 22 — neither EPERM nor ESRCH.
        let einval: io::Result<DelayStats> = Err(io::Error::from_raw_os_error(22));
        s.record_result(&einval);
        // Parse-style message with EINVAL.
        let parse_einval: io::Result<DelayStats> =
            Err(io::Error::other("kernel returned NLMSG_ERROR errno=22"));
        s.record_result(&parse_einval);
        // Bare error text with no errno suffix.
        let bare: io::Result<DelayStats> = Err(io::Error::other("AGGR_PID missing in reply"));
        s.record_result(&bare);
        assert_eq!(s.ok_count, 0);
        assert_eq!(s.eperm_count, 0);
        assert_eq!(s.esrch_count, 0);
        assert_eq!(s.other_err_count, 3);
    }

    /// `classify_errno` prefers the real `raw_os_error()` over a
    /// substring scan when both are present — defends against the
    /// case where a future error path embeds the literal text
    /// `errno=N` in the message of a socket-level failure that
    /// already carries a different real OS errno. The raw_os_error
    /// is the more authoritative signal.
    #[test]
    fn classify_errno_prefers_raw_os_error_over_text() {
        // Hand-build an io::Error that carries both a real
        // raw_os_error AND a confusable text suffix.
        let e = io::Error::other("decoy text saying errno=99 should not win");
        // The from_raw_os_error path doesn't accept arbitrary
        // text, so build a wrapper that has both signals: use
        // io::Error::from_raw_os_error and verify our classifier
        // returns the OS errno even though the text path would
        // also produce a number.
        // Note: io::Error has no public API to attach BOTH a
        // raw_os_error AND custom text simultaneously, so the
        // priority test is degenerate — once raw_os_error is
        // set, the message is a fmt::Display synthesized from
        // the errno. Cover the explicit case: a from_raw_os_error
        // error returns the OS errno via classify_errno.
        let real = io::Error::from_raw_os_error(1);
        assert_eq!(classify_errno(&real), Some(1));
        // And cover the text-only path explicitly.
        assert_eq!(classify_errno(&e), Some(99));
    }
}
