//! Host-side client for the scheduler-stats virtio-console bridge.
//!
//! The scx_stats userspace protocol (line-delimited JSON over a Unix
//! socket) is forwarded between host and guest over
//! [`super::virtio_console::VirtioConsole`] port 2:
//!
//! 1. Caller invokes [`SchedStatsClient::request_raw`] (or one of the
//!    typed wrappers [`SchedStatsClient::stats`] /
//!    [`SchedStatsClient::stats_meta`]).
//! 2. The request bytes (a complete `\n`-terminated JSON line) are
//!    pushed onto port 2 RX via
//!    [`super::virtio_console::VirtioConsole::queue_input_port2`].
//! 3. The guest's relay thread reads `/dev/vport0p2` and forwards
//!    the bytes to `/var/run/scx/root/stats`.
//! 4. The scheduler's response — also a `\n`-terminated JSON line —
//!    travels back through the same path: scheduler writes the Unix
//!    socket; relay forwards it to `/dev/vport0p2`; the device
//!    accumulates the bytes in `port2_tx_buf` and signals
//!    `stats_tx_evt`.
//! 5. The drainer thread this client owns wakes on `stats_tx_evt`,
//!    drains `port2_tx_buf`, and either appends the bytes to the
//!    response buffer (if a request is in flight) or discards them
//!    (logging the discard count).
//! 6. [`SchedStatsClient::request_raw`] blocks on a [`Condvar`]
//!    until a complete `\n`-terminated line accumulates or the
//!    timeout elapses.
//!
//! No extra framing: scx_stats is already line-delimited JSON, so a
//! TLV layer here would just double-frame.
//!
//! # Threading model
//!
//! Each client owns a dedicated drainer thread that polls
//! `stats_tx_evt` via epoll. Cloning the client is cheap (everything
//! is `Arc`-shared) and does NOT spawn additional threads — the
//! single drainer feeds every clone's response buffer because they
//! all share the same `Arc<(Mutex<Vec<u8>>, Condvar)>`.
//!
//! [`SchedStatsClient`] IS safe for concurrent use. scx_stats has
//! no request-id multiplexing, so concurrent host calls would race
//! for the response stream; the client wraps the entire
//! request/response cycle in an internal `Mutex<()>` so callers
//! that issue overlapping requests SERIALISE through the lock
//! rather than corrupting each other's responses. The drainer
//! reads a separate `request_in_flight: AtomicBool` (set inside the
//! lock, cleared on exit) to decide whether to forward drained
//! bytes to the response buffer or discard them — see [`drainer_loop`].
//!
//! # Lifetime
//!
//! Drop on the last clone tears down the drainer thread by writing
//! the `kill_drainer` eventfd. The drainer wakes from `epoll_wait`,
//! observes the kill flag, and exits.

use std::os::unix::io::AsRawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
// `Duration` is only referenced by the cfg(test) module; `Instant`
// is unused after the move to event-driven (no timeouts) cvar
// waits — keep the production code free of `std::time` so a future
// regression that re-introduces a timeout does not silently
// compile.

use serde::{Deserialize, Serialize};
use vmm_sys_util::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use vmm_sys_util::eventfd::{EFD_NONBLOCK, EventFd};

use super::PiMutex;
use super::virtio_console::VirtioConsole;

/// Synthetic relay-error envelope the drainer injects when the
/// 2x hard cap on the response accumulator is breached. Without
/// this, a runaway guest that bursts > 512 KiB without a newline
/// would silently get its bytes dropped while the host request
/// hung in `cvar.wait` forever. The drainer instead clears the
/// buffer, writes this synthetic envelope, and notifies the cvar
/// so `request_raw` wakes up, parses the envelope, and surfaces
/// [`SchedStatsError::NoScheduler`] with a `cap-overflow` reason.
/// The trailing `\n` matches scx_stats's line-delimited wire
/// format so the existing newline-detection path consumes it
/// without special-casing.
const CAP_OVERFLOW_ERROR_REPLY: &[u8] = b"{\"ktstr_relay_error\":\"response cap overflow\"}\n";

/// Maximum bytes accepted in a single host→guest stats request.
/// scx_stats requests are JSON command lines (`{"req":"stats"}\n`
/// and similar); 256 KiB is far above any legitimate request size.
/// Caps the per-request `queue_input_port2` allocation so a buggy
/// caller can't grow the device's `port2_pending_rx` deque without
/// bound on a single push.
pub const MAX_REQUEST_BYTES: usize = 256 * 1024;

/// Maximum bytes the response accumulator may grow to before
/// [`SchedStatsClient::request_raw`] returns
/// [`SchedStatsError::ResponseTooLarge`]. scx_stats responses are
/// JSON objects on a single line; 256 KiB matches the request cap
/// and protects against a hostile guest (or a runaway relay) that
/// sends bytes without ever emitting a newline.
pub const MAX_RESPONSE_BYTES: usize = 256 * 1024;

/// Errors surfaced by [`SchedStatsClient::request_raw`]. Distinct
/// from `anyhow::Error` so callers can branch on the failure mode
/// without string-matching error chains.
#[derive(Debug)]
pub enum SchedStatsError {
    /// The shared response mutex was poisoned by a previous panic.
    /// The client cannot recover; callers should propagate the
    /// error.
    Poisoned,
    /// The caller passed a request larger than [`MAX_REQUEST_BYTES`].
    RequestTooLarge {
        /// Size the caller passed, in bytes.
        size: usize,
        /// Cap configured on the client (`MAX_REQUEST_BYTES`).
        max: usize,
    },
    /// The response accumulator grew past [`MAX_RESPONSE_BYTES`]
    /// without ever emitting a newline. The buffer is cleared
    /// before the error returns so the next request starts from a
    /// clean slate; the partial bytes that triggered the cap are
    /// discarded.
    ResponseTooLarge {
        /// Accumulator size at the moment the cap was breached.
        size: usize,
        /// Cap configured on the client (`MAX_RESPONSE_BYTES`).
        max: usize,
    },
    /// The host-side coordinator marked the run as freezing while
    /// this request was in flight (or about to start). scx_stats
    /// responses are undefined while the scheduler's userspace
    /// thread is paused.
    DuringFreeze,
    /// The run-wide cancel flag was set (the watchdog fired or
    /// the run is shutting down) while this request was in flight
    /// or about to start. The host-side watchdog is the only
    /// "timeout" in the stats path — when it fires, every
    /// outstanding `request_raw` returns immediately with this
    /// variant rather than blocking forever.
    Cancelled,
    /// The guest relay never connected to the scheduler's Unix
    /// socket (no scheduler running, or the scheduler refused the
    /// connection). Delivered as an inline JSON error response by
    /// the in-guest relay; surfaced here so callers can branch on
    /// "no scheduler" without parsing the error JSON.
    NoScheduler {
        /// Reason buffer carried in the inline error response.
        reason: String,
    },
    /// The scheduler returned a non-zero `errno` in the typed
    /// [`StatsResponse`] envelope. The wire payload's verb-specific
    /// `args` field is preserved so the caller can render an
    /// actionable diagnostic.
    SchedulerError {
        /// `errno` field from the scx_stats response envelope.
        errno: i32,
        /// `args` payload from the response. Typically a JSON
        /// object with an error message; opaque to this client.
        args: serde_json::Value,
    },
    /// The typed envelope was successfully decoded but the inner
    /// `args` map did not contain the expected `"resp"` key.
    /// scx_stats schedulers wrap their verb-specific payload under
    /// this key; its absence indicates either a protocol mismatch
    /// or a non-conforming scheduler implementation.
    MissingResp {
        /// Full `args` payload as decoded — useful for diagnostics.
        args: serde_json::Value,
    },
}

impl std::fmt::Display for SchedStatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poisoned => write!(f, "scx_stats response buffer mutex was poisoned"),
            Self::RequestTooLarge { size, max } => write!(
                f,
                "scx_stats request size {size} bytes exceeds cap of {max}"
            ),
            Self::ResponseTooLarge { size, max } => write!(
                f,
                "scx_stats response accumulator grew to {size} bytes (cap {max}) \
                 without emitting a newline; partial bytes discarded"
            ),
            Self::DuringFreeze => write!(
                f,
                "scx_stats request rejected: freeze rendezvous active \
                 (scheduler userspace is paused; responses undefined)"
            ),
            Self::Cancelled => write!(
                f,
                "scx_stats request cancelled: run-wide kill flag set \
                 (watchdog fired or shutdown in progress)"
            ),
            Self::NoScheduler { reason } => write!(
                f,
                "scx_stats relay reports no scheduler available: {reason}"
            ),
            Self::SchedulerError { errno, args } => {
                write!(f, "scx_stats scheduler returned errno={errno}: {args}")
            }
            Self::MissingResp { args } => write!(
                f,
                "scx_stats response envelope missing \"resp\" key in args: {args}"
            ),
        }
    }
}

impl std::error::Error for SchedStatsError {}

/// Vendored scx_stats request envelope. Wire shape pinned to
/// scx_stats v1.1.0 `StatsRequest` (server.rs:103-117 of the
/// upstream crate): a JSON object with a `"req"` verb and
/// optional `"args"` map of string-valued keys. We do not depend
/// on the scx_stats crate to keep the dependency surface minimal
/// for a 2-field type.
///
/// `args` matches upstream byte-for-byte as `BTreeMap<String,
/// String>` — non-string argument values would deserialize-fail
/// on the scheduler side (the upstream `BTreeMap<String, String>`
/// rejects e.g. integers). Keep this in lockstep with upstream
/// or update both sides together.
///
/// Wire format examples:
/// * `{"req":"stats"}\n`
/// * `{"req":"stats_meta"}\n`
/// * `{"req":"stats","args":{"target":"foo"}}\n`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsRequest {
    /// Verb name. scx_stats schedulers handle `"stats"`,
    /// `"stats_meta"`, and scheduler-specific verbs.
    pub req: String,
    /// Optional argument map. Empty by default; scheduler-specific
    /// verbs may require argument keys. Values are strings on the
    /// wire — see the type-level note above for the upstream
    /// constraint.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub args: std::collections::BTreeMap<String, String>,
}

/// Vendored scx_stats response envelope. Wire shape pinned to
/// scx_stats v1.1.0 `StatsResponse` (server.rs:119-123 of the
/// upstream crate). The scheduler emits a JSON object on a
/// single line; the shape is verb-specific. We model the on-wire
/// envelope as `{"errno": Option<i32>, "args": Value}` —
/// present-and-non-zero `errno` flags scheduler-side errors,
/// otherwise `args` carries the verb's payload.
///
/// `args` is intentionally `serde_json::Value` (more permissive
/// than upstream's `BTreeMap<String, Value>`): every scx_stats
/// scheduler emits an object today, so [`extract_resp`] checks
/// `Value::Object` before unwrapping `args["resp"]` and the
/// permissiveness costs nothing on the inbound path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResponse {
    /// scx_stats error code. `0` (or absent) signals success.
    /// Non-zero matches the scx_stats schedulers' error-mapping
    /// convention (`Err(...)` → libc errno).
    #[serde(default)]
    pub errno: i32,
    /// Verb-specific payload. For `"stats"` this is the scheduler's
    /// stats object; for `"stats_meta"` it is the schema map; for
    /// scheduler-specific verbs it is whatever the scheduler
    /// emits.
    #[serde(default = "default_args_value")]
    pub args: serde_json::Value,
}

fn default_args_value() -> serde_json::Value {
    serde_json::Value::Null
}

/// Inline error payload the in-guest relay emits when it cannot
/// connect to the scheduler's Unix socket. Wire format is
/// `{"ktstr_relay_error":"<reason>"}\n`. Callers receive
/// [`SchedStatsError::NoScheduler`] when this is detected; the
/// JSON shape lets the host translate the error without
/// scheduler-side cooperation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RelayError {
    ktstr_relay_error: String,
}

/// Shared response accumulator written by the drainer thread and
/// drained by [`SchedStatsClient::request_raw`].
///
/// `Mutex<Vec<u8>>` carries the bytes; the [`Condvar`] is
/// signalled after every append so blocked readers wake immediately
/// rather than polling.
type StatsResponseBuf = Arc<(Mutex<Vec<u8>>, Condvar)>;

/// Internal state shared between every clone of a
/// [`SchedStatsClient`] AND the drainer thread.
///
/// `Drop` on the last `Arc` clone fires the kill eventfd so the
/// drainer thread exits.
struct ClientShared {
    /// Device handle. Used by `request_raw` to push request bytes
    /// via `queue_input_port2` AND by the drainer to call
    /// `drain_port2_bulk`. Both are short critical sections.
    virtio_con: Arc<PiMutex<VirtioConsole>>,
    /// Response accumulator + Condvar for blocked readers.
    response_buf: StatsResponseBuf,
    /// Optional shared freeze flag. Set by the freeze coordinator
    /// during failure-dump rendezvous; gates `request_raw` calls
    /// (returns [`SchedStatsError::DuringFreeze`]).
    freeze: Option<Arc<AtomicBool>>,
    /// Optional run-wide cancel flag. When `Some(flag)` and the
    /// flag reads `true` (Acquire), `request_raw` returns
    /// [`SchedStatsError::Cancelled`] without ever blocking. The
    /// drainer also watches a paired eventfd ([`Self::cancel_evt`])
    /// and notifies the response Condvar when it fires so a
    /// blocked `request_raw` wakes immediately on shutdown — no
    /// timeout needed. The host watchdog is the only "timeout"
    /// in the system.
    cancel: Option<Arc<AtomicBool>>,
    /// Internal serialisation lock held for the entire
    /// request/response cycle. scx_stats has no req-id multiplexing,
    /// so concurrent host calls must serialise to avoid mixing
    /// responses; held across `queue_input_port2` and the
    /// `Condvar::wait` wait.
    request_lock: Mutex<()>,
    /// `true` while a `request_raw` call is mid-flight on any
    /// clone of this client. Set inside `request_lock`, cleared
    /// on exit. The drainer reads this to decide whether to
    /// append drained bytes (true) or discard them (false).
    request_in_flight: Arc<AtomicBool>,
    /// Cumulative count of bytes the drainer has discarded because
    /// no request was in flight when they arrived. Logged by the
    /// drainer at every discard event.
    discarded_bytes: Arc<AtomicU64>,
    /// Eventfd written by `Drop` on the last `Arc` clone. The
    /// drainer's epoll wakes on this and exits.
    kill_drainer: EventFd,
}

impl Drop for ClientShared {
    fn drop(&mut self) {
        // Wake the drainer so it observes the implicit "shared
        // state is being torn down" (the EventFd write triggers
        // the `kill_drainer` epoll fd; the drainer thread's
        // epoll_wait returns and the thread exits). Failure to
        // write is harmless — the drainer's next epoll cycle on
        // stats_tx_evt will re-park, and the kernel will reap the
        // thread when the process exits.
        let _ = self.kill_drainer.write(1);
    }
}

/// Host-side client for the scheduler-stats virtio-console bridge.
///
/// Cloning is cheap: every field is `Arc`-shared and no additional
/// drainer thread is spawned. Drop on the LAST clone tears down
/// the drainer.
///
/// # Freeze semantics
///
/// scx_stats responses are undefined during a host-side freeze
/// rendezvous: the freeze coordinator suspends every vCPU thread,
/// the in-guest scheduler userspace stops advancing, and the
/// relay's blocking read on `/var/run/scx/root/stats` does not
/// return until the scheduler resumes. The optional `freeze` flag
/// gates requests on this state — when the flag is `true`,
/// `request_raw` returns [`SchedStatsError::DuringFreeze`] without
/// touching the device.
#[derive(Clone)]
pub struct SchedStatsClient {
    shared: Arc<ClientShared>,
}

impl SchedStatsClient {
    /// Construct a new client and spawn its drainer thread.
    ///
    /// Scope is `pub(crate)` because [`PiMutex<VirtioConsole>`] is
    /// `pub(crate)` — only the in-tree `KtstrVm::run_vm` wiring
    /// constructs clients; user code receives a fully constructed
    /// [`SchedStatsClient`] via
    /// [`crate::vmm::VmResult::stats_client`] and calls only the
    /// `pub` request methods on it.
    pub(crate) fn new(
        virtio_con: Arc<PiMutex<VirtioConsole>>,
        freeze: Option<Arc<AtomicBool>>,
        cancel: Option<Arc<AtomicBool>>,
        cancel_evt: Option<Arc<EventFd>>,
    ) -> std::io::Result<Self> {
        let response_buf: StatsResponseBuf = Arc::new((Mutex::new(Vec::new()), Condvar::new()));
        let request_in_flight = Arc::new(AtomicBool::new(false));
        let discarded_bytes = Arc::new(AtomicU64::new(0));
        let kill_drainer = EventFd::new(EFD_NONBLOCK)?;
        let kill_drainer_for_thread = kill_drainer.try_clone()?;
        let stats_tx_evt = virtio_con.lock().stats_tx_evt().try_clone()?;

        let shared = Arc::new(ClientShared {
            virtio_con: virtio_con.clone(),
            response_buf: Arc::clone(&response_buf),
            freeze,
            cancel,
            request_lock: Mutex::new(()),
            request_in_flight: Arc::clone(&request_in_flight),
            discarded_bytes: Arc::clone(&discarded_bytes),
            kill_drainer,
        });

        // Spawn the drainer thread. It owns the stats_tx_evt clone
        // (parked in epoll), the kill_drainer clone, and an
        // optional cancel_evt clone. Lifetimes are tied to the
        // thread itself so dropping the last `Arc<ClientShared>`
        // writes kill_drainer, the epoll wakes, and the thread
        // exits — and a host-wide cancel edge wakes the drainer
        // first so any blocked `request_raw` cvar wait gets a
        // notify_all + cancel-flag check before its caller's
        // watchdog deadline elapses.
        let drain_virtio_con = virtio_con;
        let drain_response_buf = response_buf;
        let drain_request_in_flight = request_in_flight;
        let drain_discarded_bytes = discarded_bytes;
        std::thread::Builder::new()
            .name("ktstr-sched-stats-drain".into())
            .spawn(move || {
                drainer_loop(
                    stats_tx_evt,
                    kill_drainer_for_thread,
                    cancel_evt,
                    drain_virtio_con,
                    drain_response_buf,
                    drain_request_in_flight,
                    drain_discarded_bytes,
                );
            })?;

        Ok(Self { shared })
    }

    /// Send a raw request line (without trailing `\n`) and return
    /// the raw response line (also without `\n`). The trailing
    /// newline scx_stats expects on the wire is appended internally
    /// — callers pass JSON or any other line-shaped payload as a
    /// `&str` and never have to worry about framing.
    pub fn request_raw(&self, line: &str) -> Result<Vec<u8>, SchedStatsError> {
        // Compute the on-wire byte length (line + trailing '\n')
        // and reject before touching the device or the lock so a
        // misuse that exceeds MAX_REQUEST_BYTES does not stall a
        // concurrent caller.
        let on_wire_len = line.len().saturating_add(1);
        if on_wire_len > MAX_REQUEST_BYTES {
            return Err(SchedStatsError::RequestTooLarge {
                size: on_wire_len,
                max: MAX_REQUEST_BYTES,
            });
        }
        if let Some(flag) = self.shared.freeze.as_ref()
            && flag.load(Ordering::Acquire)
        {
            return Err(SchedStatsError::DuringFreeze);
        }
        if let Some(flag) = self.shared.cancel.as_ref()
            && flag.load(Ordering::Acquire)
        {
            return Err(SchedStatsError::Cancelled);
        }

        // Serialise the entire request/response cycle. The lock is
        // held until this method returns so the drainer's
        // `request_in_flight` flag (set inside the lock, cleared on
        // exit) cannot race with a concurrent caller. Mutex
        // poisoning recovers via `into_inner` — we only ever access
        // the unit `()` payload, which has no invariant a panicked
        // thread could leave broken.
        let _request_guard = self
            .shared
            .request_lock
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Mark in-flight so the drainer routes drained bytes to the
        // response buffer instead of discarding them. RAII clear on
        // scope exit covers every error-return path below.
        self.shared.request_in_flight.store(true, Ordering::Release);
        let _in_flight_guard = InFlightGuard {
            flag: &self.shared.request_in_flight,
        };

        let (lock, cvar) = &*self.shared.response_buf;
        // Drain any stale bytes left over from a prior incomplete
        // call. scx_stats is strict request/response — anything
        // sitting in the buffer when a fresh request starts is
        // either a torn previous response or relay-restart
        // residue. Log the discard so a stuck-relay diagnosis can
        // see how many bytes were thrown away on each request,
        // and bump `discarded_bytes` so the counter exposed via
        // [`Self::discarded_bytes`] reflects this path too — not
        // just the drainer's no-request-in-flight discards.
        {
            let mut buf = lock.lock().map_err(|_| SchedStatsError::Poisoned)?;
            if !buf.is_empty() {
                let stale = buf.len();
                let total = self
                    .shared
                    .discarded_bytes
                    .fetch_add(stale as u64, Ordering::Relaxed)
                    .saturating_add(stale as u64);
                tracing::debug!(
                    stale_bytes = stale,
                    total_discarded = total,
                    "scx_stats request_raw: clearing stale response bytes from prior call"
                );
                buf.clear();
            }
        }

        // Push the request bytes onto port 2 RX. Append the
        // trailing newline scx_stats expects. Bound the device
        // mutex critical section to the queue_input_port2 call.
        //
        // B15 fix: drop any host→guest bytes that are still
        // sitting in `port2_pending_rx` from a prior request that
        // was abandoned mid-push (e.g. a freeze rendezvous landed
        // before the guest read those bytes). Without this clear,
        // the new request would be concatenated onto the dead
        // tail of the previous one and the guest relay would
        // forward torn JSON to the scheduler. Account for the
        // discard via `discarded_bytes` so a stuck-stats
        // post-mortem can see it.
        {
            let mut g = self.shared.virtio_con.lock();
            let stale_in = g.clear_port2_pending_rx();
            if stale_in > 0 {
                let total = self
                    .shared
                    .discarded_bytes
                    .fetch_add(stale_in as u64, Ordering::Relaxed)
                    .saturating_add(stale_in as u64);
                tracing::debug!(
                    stale_pending_rx = stale_in,
                    total_discarded = total,
                    "scx_stats request_raw: clearing stale port2_pending_rx \
                     (prior request abandoned mid-push)"
                );
            }
            // Two pushes are equivalent to one combined push from
            // the device's perspective — the bytes land in the
            // pending-RX deque in order.
            g.queue_input_port2(line.as_bytes());
            g.queue_input_port2(b"\n");
        }

        // Wait (no timeout) for a complete line, freeze, or cancel.
        // The drainer notifies the cvar on every appended byte AND
        // when its kill_drainer eventfd fires (cancel edge or
        // last-Arc drop), so a blocked `cvar.wait` always wakes
        // promptly without polling. The host watchdog is the only
        // backstop "timeout" — when it fires it sets `cancel`,
        // writes the cancel_evt, and the drainer's notify_all
        // wakes us into the cancel-check below.
        let mut buf = lock.lock().map_err(|_| SchedStatsError::Poisoned)?;
        loop {
            if let Some(idx) = buf.iter().position(|&b| b == b'\n') {
                let mut response = buf.split_off(idx + 1);
                std::mem::swap(&mut *buf, &mut response);
                response.pop();
                // Inspect for the relay-emitted error envelope. If
                // the line parses as `{"ktstr_relay_error":"..."}`
                // surface it as the typed NoScheduler error so the
                // caller can branch without parsing the JSON.
                if let Ok(err) = serde_json::from_slice::<RelayError>(&response) {
                    return Err(SchedStatsError::NoScheduler {
                        reason: err.ktstr_relay_error,
                    });
                }
                return Ok(response);
            }
            if buf.len() > MAX_RESPONSE_BYTES {
                let size = buf.len();
                buf.clear();
                return Err(SchedStatsError::ResponseTooLarge {
                    size,
                    max: MAX_RESPONSE_BYTES,
                });
            }
            if let Some(flag) = self.shared.freeze.as_ref()
                && flag.load(Ordering::Acquire)
            {
                buf.clear();
                return Err(SchedStatsError::DuringFreeze);
            }
            if let Some(flag) = self.shared.cancel.as_ref()
                && flag.load(Ordering::Acquire)
            {
                buf.clear();
                return Err(SchedStatsError::Cancelled);
            }
            buf = cvar.wait(buf).map_err(|_| SchedStatsError::Poisoned)?;
        }
    }

    /// Send a typed scx_stats request and parse the typed response
    /// envelope. The envelope's `errno` field is NOT checked here
    /// — callers that want errno-as-error semantics should use
    /// [`Self::stats`] / [`Self::stats_meta`] (which surface
    /// non-zero errno as [`SchedStatsError::SchedulerError`]) or
    /// inspect `StatsResponse.errno` themselves.
    pub fn request(&self, request: &StatsRequest) -> Result<StatsResponse, anyhow::Error> {
        // Encode as JSON without the trailing newline — request_raw
        // appends it. `to_string` is preferable to `to_vec` here
        // because request_raw takes a `&str`.
        let line = serde_json::to_string(request)?;
        let raw = self.request_raw(&line).map_err(|e| anyhow::anyhow!(e))?;
        Ok(serde_json::from_slice::<StatsResponse>(&raw)?)
    }

    /// Convenience: send a `"stats"` request and return the inner
    /// `args["resp"]` payload. `args` is an ordered list of
    /// scheduler-specific argument key/value pairs (e.g.
    /// `&[("target", "top")]` to request the scheduler's "top"
    /// dataset). Pass `&[]` for the default `"stats"` payload.
    ///
    /// Errors:
    /// * [`SchedStatsError::SchedulerError`] when the scheduler's
    ///   response envelope carries `errno != 0`.
    /// * [`SchedStatsError::MissingResp`] when the envelope
    ///   succeeded but `args["resp"]` is absent (protocol
    ///   mismatch).
    /// * Other variants surface from the underlying
    ///   [`Self::request`] / [`Self::request_raw`] flow.
    pub fn stats(&self, args: &[(&str, &str)]) -> Result<serde_json::Value, anyhow::Error> {
        let mut req = StatsRequest {
            req: "stats".to_string(),
            args: std::collections::BTreeMap::new(),
        };
        for (k, v) in args {
            req.args.insert((*k).to_string(), (*v).to_string());
        }
        let resp = self.request(&req)?;
        extract_resp(resp)
    }

    /// Convenience: send a `"stats_meta"` request and return the
    /// inner `args["resp"]` payload (the metadata schema for the
    /// scheduler's `"stats"` payload).
    pub fn stats_meta(&self) -> Result<serde_json::Value, anyhow::Error> {
        let req = StatsRequest {
            req: "stats_meta".to_string(),
            args: std::collections::BTreeMap::new(),
        };
        let resp = self.request(&req)?;
        extract_resp(resp)
    }

    /// Cumulative number of bytes the client has discarded over its
    /// lifetime. Two paths feed this counter:
    /// 1. The drainer thread receives port-2 bytes when no request
    ///    is in flight (a stale response from a torn prior call,
    ///    or relay-emitted bytes the host never asked for).
    /// 2. [`Self::request_raw`] clears stale bytes from the
    ///    response buffer at request start.
    ///
    /// Both paths bump `discarded_bytes`, so this accessor lets
    /// test code (and operators inspecting a stuck stats path)
    /// see whether bytes are leaking past the request/response
    /// envelope.
    pub fn discarded_bytes(&self) -> u64 {
        self.shared.discarded_bytes.load(Ordering::Relaxed)
    }
}

/// Convert a [`StatsResponse`] envelope into the verb-specific
/// inner payload: rejects non-zero `errno`, then extracts the
/// `"resp"` key from `args`. Centralised so [`SchedStatsClient::stats`]
/// and [`SchedStatsClient::stats_meta`] share identical error
/// surfaces.
fn extract_resp(resp: StatsResponse) -> Result<serde_json::Value, anyhow::Error> {
    if resp.errno != 0 {
        return Err(anyhow::anyhow!(SchedStatsError::SchedulerError {
            errno: resp.errno,
            args: resp.args,
        }));
    }
    let serde_json::Value::Object(mut map) = resp.args else {
        return Err(anyhow::anyhow!(SchedStatsError::MissingResp {
            args: resp.args
        }));
    };
    match map.remove("resp") {
        Some(v) => Ok(v),
        None => Err(anyhow::anyhow!(SchedStatsError::MissingResp {
            args: serde_json::Value::Object(map),
        })),
    }
}

impl std::fmt::Debug for SchedStatsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedStatsClient")
            .field(
                "response_buf_len",
                &self
                    .shared
                    .response_buf
                    .0
                    .lock()
                    .map(|b| b.len())
                    .unwrap_or(0),
            )
            .field(
                "discarded_bytes",
                &self.shared.discarded_bytes.load(Ordering::Relaxed),
            )
            .field(
                "request_in_flight",
                &self.shared.request_in_flight.load(Ordering::Relaxed),
            )
            .finish()
    }
}

/// RAII guard: clear `request_in_flight` on drop. Prevents a
/// stuck-true flag if `request_raw_with_timeout` returns early via
/// `?` from any of the lock / mutex calls.
struct InFlightGuard<'a> {
    flag: &'a Arc<AtomicBool>,
}

impl<'a> Drop for InFlightGuard<'a> {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// Drainer-thread main loop. Runs until `kill_drainer` fires (the
/// last `Arc<ClientShared>` clone has dropped) OR `cancel_evt`
/// fires (the run-wide kill flag was set, e.g. watchdog timeout).
/// Drains `port2_tx_buf` on every `stats_tx_evt` wake; appends to
/// the response buffer when a request is in flight, discards
/// otherwise. On either kill edge, performs a final
/// `cvar.notify_all()` so any blocked `request_raw` wakes,
/// observes the cancel flag, and returns
/// [`SchedStatsError::Cancelled`].
fn drainer_loop(
    stats_tx_evt: EventFd,
    kill_drainer: EventFd,
    cancel_evt: Option<Arc<EventFd>>,
    virtio_con: Arc<PiMutex<VirtioConsole>>,
    response_buf: StatsResponseBuf,
    request_in_flight: Arc<AtomicBool>,
    discarded_bytes: Arc<AtomicU64>,
) {
    const TOKEN_DATA: u64 = 0;
    const TOKEN_KILL: u64 = 1;
    const TOKEN_CANCEL: u64 = 2;

    let epoll = match Epoll::new() {
        Ok(e) => e,
        Err(e) => {
            tracing::error!(error = %e, "stats drainer: epoll_create1 failed; aborting drainer");
            return;
        }
    };
    for (fd, token, name) in [
        (stats_tx_evt.as_raw_fd(), TOKEN_DATA, "stats_tx_evt"),
        (kill_drainer.as_raw_fd(), TOKEN_KILL, "kill_drainer"),
    ] {
        if let Err(e) = epoll.ctl(
            ControlOperation::Add,
            fd,
            EpollEvent::new(EventSet::IN, token),
        ) {
            tracing::error!(
                error = %e,
                fd_name = name,
                "stats drainer: epoll_ctl ADD failed; aborting drainer"
            );
            return;
        }
    }
    // Optional cancel_evt: the host's run-wide kill_evt clone.
    // When the watchdog or the BSP shutdown path writes it, our
    // epoll wakes; we notify the response cvar (so blocked
    // `request_raw` waiters re-check the cancel flag) and exit.
    if let Some(cancel) = cancel_evt.as_ref()
        && let Err(e) = epoll.ctl(
            ControlOperation::Add,
            cancel.as_raw_fd(),
            EpollEvent::new(EventSet::IN, TOKEN_CANCEL),
        )
    {
        tracing::warn!(
            error = %e,
            "stats drainer: epoll_ctl ADD on cancel_evt failed; \
             cancel edge will not wake blocked requests promptly"
        );
    }

    let mut events_buf = [EpollEvent::default(); 3];
    loop {
        let event_count = match epoll.wait(-1, &mut events_buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                tracing::error!(error = %e, "stats drainer: epoll_wait failed; exiting");
                return;
            }
        };
        for ev in &events_buf[..event_count] {
            match ev.data() {
                TOKEN_KILL => {
                    let _ = kill_drainer.read();
                    // B5 fix: acquire the response_buf lock BEFORE
                    // notifying the cvar. Without the lock the
                    // notify_all can fire after the request thread
                    // has dropped the lock-guard but BEFORE it has
                    // entered cvar.wait — that wake is then lost
                    // and the request hangs until the next
                    // legitimate notify (which on the kill path
                    // will never come). Holding the lock across
                    // notify_all forces the drainer to interleave
                    // with the request's check-and-park sequence:
                    // if the request has already entered cvar.wait,
                    // notify_all wakes it; if the request is still
                    // inside the response-buf check, the drainer
                    // blocks on the lock until the request enters
                    // cvar.wait, then notifies. Either way the
                    // wake is delivered.
                    //
                    // `lock()` returns LockResult; both Ok and Err
                    // (poisoned) variants contain a MutexGuard
                    // that unlocks on Drop, so binding the
                    // LockResult to `_guard` is sufficient — we
                    // don't read the guarded payload, only need
                    // exclusive ownership for the notify.
                    let (lock, cvar) = &*response_buf;
                    let _guard = lock.lock();
                    cvar.notify_all();
                    return;
                }
                TOKEN_CANCEL => {
                    // Drain the cancel eventfd counter (the writer
                    // set the run-wide cancel AtomicBool first;
                    // request_raw rechecks it after wakeup).
                    if let Some(c) = cancel_evt.as_ref() {
                        let _ = c.read();
                    }
                    // Lock-then-notify, same B5 TOCTOU rationale
                    // as the TOKEN_KILL arm above.
                    let (lock, cvar) = &*response_buf;
                    let _guard = lock.lock();
                    cvar.notify_all();
                    return;
                }
                TOKEN_DATA => {
                    let _ = stats_tx_evt.read();
                    // Drain regardless of in-flight state. F14
                    // ruling: the drainer always drains so a
                    // hostile or runaway guest can't grow
                    // port2_tx_buf without bound when no host
                    // request is outstanding.
                    let bytes = {
                        let mut g = virtio_con.lock();
                        g.drain_port2_bulk()
                    };
                    if bytes.is_empty() {
                        continue;
                    }
                    if request_in_flight.load(Ordering::Acquire) {
                        let (lock, cvar) = &*response_buf;
                        if let Ok(mut guard) = lock.lock() {
                            // Hard 2x cap to prevent unbounded growth
                            // even when a request is mid-flight (a
                            // hostile guest could send bytes faster
                            // than the request can complete). B4
                            // fix: inject a synthetic error envelope
                            // so request_raw wakes from cvar.wait,
                            // observes a complete `\n`-terminated
                            // line, and surfaces NoScheduler — the
                            // prior code dropped bytes silently and
                            // the request blocked forever.
                            let new_total = guard.len().saturating_add(bytes.len());
                            if new_total > MAX_RESPONSE_BYTES.saturating_mul(2) {
                                tracing::warn!(
                                    current = guard.len(),
                                    incoming = bytes.len(),
                                    cap = MAX_RESPONSE_BYTES * 2,
                                    "stats drainer: hard cap reached; injecting cap-overflow error envelope"
                                );
                                guard.clear();
                                guard.extend_from_slice(CAP_OVERFLOW_ERROR_REPLY);
                                cvar.notify_all();
                                continue;
                            }
                            guard.extend_from_slice(&bytes);
                            cvar.notify_all();
                        }
                    } else {
                        // F14 ruling: discard bytes when no request is
                        // outstanding, log the discard count.
                        let total = discarded_bytes
                            .fetch_add(bytes.len() as u64, Ordering::Relaxed)
                            .saturating_add(bytes.len() as u64);
                        tracing::debug!(
                            this_drop = bytes.len(),
                            total_discarded = total,
                            "stats drainer: discarding port-2 bytes (no request in flight)"
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_client_full(
        freeze: Option<Arc<AtomicBool>>,
        cancel: Option<Arc<AtomicBool>>,
        cancel_evt: Option<Arc<EventFd>>,
    ) -> SchedStatsClient {
        let virtio_con = Arc::new(PiMutex::new(VirtioConsole::new()));
        SchedStatsClient::new(virtio_con, freeze, cancel, cancel_evt).expect("client construct")
    }

    fn make_client() -> SchedStatsClient {
        make_client_full(None, None, None)
    }

    /// Helper: simulate the drainer appending a complete response
    /// line. The drainer normally sets `request_in_flight` from the
    /// device side; tests stand it up manually. The thread waits
    /// briefly for `request_in_flight` to flip true (the request
    /// thread sets it after grabbing the request_lock) before
    /// appending so the test races the request rather than racing
    /// the writer.
    fn pre_populate(client: &SchedStatsClient, bytes: &[u8]) -> std::thread::JoinHandle<()> {
        let buf = Arc::clone(&client.shared.response_buf);
        let in_flight = Arc::clone(&client.shared.request_in_flight);
        let bytes = bytes.to_vec();
        std::thread::spawn(move || {
            // Wait for the request thread to mark in-flight. Bound
            // the wait with a finite spin count so a test bug
            // doesn't hang the suite.
            for _ in 0..200 {
                if in_flight.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let (lock, cvar) = &*buf;
            let mut guard = lock.lock().unwrap();
            guard.extend_from_slice(&bytes);
            cvar.notify_all();
        })
    }

    /// Pre-populating the response buffer wakes a blocked
    /// `request_raw` and returns the bytes before the first newline.
    #[test]
    fn drainer_append_then_request_returns_first_line() {
        let client = make_client();
        let writer = pre_populate(&client, b"hello\n");
        let resp = client
            .request_raw("x")
            .expect("must wake when bytes arrive");
        assert_eq!(resp, b"hello");
        writer.join().unwrap();
    }

    /// A request larger than [`MAX_REQUEST_BYTES`] (after appending
    /// the trailing newline) is rejected before any device
    /// interaction.
    #[test]
    fn oversize_request_rejected() {
        let client = make_client();
        // request_raw appends '\n', so a line of MAX_REQUEST_BYTES
        // bytes triggers the cap (MAX + 1 on wire).
        let big = "x".repeat(MAX_REQUEST_BYTES);
        let err = client
            .request_raw(&big)
            .expect_err("must reject oversize request");
        match err {
            SchedStatsError::RequestTooLarge { size, max } => {
                assert_eq!(size, MAX_REQUEST_BYTES + 1);
                assert_eq!(max, MAX_REQUEST_BYTES);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// A response that grows past [`MAX_RESPONSE_BYTES`] without a
    /// newline returns [`SchedStatsError::ResponseTooLarge`].
    #[test]
    fn oversize_response_returns_response_too_large() {
        let client = make_client();
        let payload: Vec<u8> = std::iter::repeat_n(b'A', MAX_RESPONSE_BYTES + 1).collect();
        let writer = pre_populate(&client, &payload);
        let err = client
            .request_raw("x")
            .expect_err("must reject oversize response");
        match err {
            SchedStatsError::ResponseTooLarge { size, max } => {
                assert!(size > MAX_RESPONSE_BYTES);
                assert_eq!(max, MAX_RESPONSE_BYTES);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        writer.join().unwrap();
    }

    /// A request issued while the freeze flag is set returns
    /// [`SchedStatsError::DuringFreeze`] without ever blocking.
    #[test]
    fn during_freeze_rejects_request() {
        let freeze = Arc::new(AtomicBool::new(true));
        let client = make_client_full(Some(freeze.clone()), None, None);

        let err = client
            .request_raw("x")
            .expect_err("must reject during freeze");
        assert!(matches!(err, SchedStatsError::DuringFreeze));

        freeze.store(false, Ordering::Release);
        let writer = pre_populate(&client, b"ok\n");
        let resp = client.request_raw("x").expect("must succeed after thaw");
        assert_eq!(resp, b"ok");
        writer.join().unwrap();
    }

    /// A request issued while the cancel flag is set returns
    /// [`SchedStatsError::Cancelled`] without ever blocking.
    #[test]
    fn cancel_flag_set_before_request_rejects() {
        let cancel = Arc::new(AtomicBool::new(true));
        let cancel_evt = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let client = make_client_full(None, Some(cancel), Some(cancel_evt));
        let err = client
            .request_raw("x")
            .expect_err("must reject when cancel pre-set");
        assert!(matches!(err, SchedStatsError::Cancelled));
    }

    /// Cancel flag flipped + cancel_evt fired DURING a blocked
    /// request wakes the cvar wait via the drainer, surfaces
    /// [`SchedStatsError::Cancelled`], and tears down cleanly.
    #[test]
    fn cancel_during_blocked_request_wakes() {
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_evt = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let client = make_client_full(None, Some(cancel.clone()), Some(cancel_evt.clone()));

        let in_flight = Arc::clone(&client.shared.request_in_flight);
        let waker = std::thread::spawn(move || {
            // Wait until the request actually parks on the cvar
            // (in_flight goes true after grabbing the request_lock
            // and before the wait).
            for _ in 0..200 {
                if in_flight.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            cancel.store(true, Ordering::Release);
            let _ = cancel_evt.write(1);
        });

        let err = client
            .request_raw("x")
            .expect_err("must wake on cancel edge");
        assert!(matches!(err, SchedStatsError::Cancelled));
        waker.join().unwrap();
    }

    /// Two concurrent `request_raw` calls SERIALISE through the
    /// internal request_lock — the second waits for the first to
    /// finish rather than racing for the response buffer. Both
    /// observe their own response.
    #[test]
    fn concurrent_requests_serialise_via_lock() {
        let client = make_client();
        let buf = Arc::clone(&client.shared.response_buf);
        let in_flight = Arc::clone(&client.shared.request_in_flight);

        // Writer feeds two responses in sequence: each fires once
        // the request thread has marked in_flight (so it's
        // already parked on the cvar). Because the request_lock
        // serialises the two requests, in_flight goes
        // true→false→true across the gap between the two requests.
        let writer = std::thread::spawn(move || {
            // First response.
            for _ in 0..200 {
                if in_flight.load(Ordering::Acquire) {
                    let (lock, cvar) = &*buf;
                    let mut guard = lock.lock().unwrap();
                    guard.extend_from_slice(b"first\n");
                    cvar.notify_all();
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            // Wait for the first request to finish (in_flight
            // goes false), then for the second request to grab
            // the lock (in_flight goes true again).
            for _ in 0..200 {
                if !in_flight.load(Ordering::Acquire) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            for _ in 0..200 {
                if in_flight.load(Ordering::Acquire) {
                    let (lock, cvar) = &*buf;
                    let mut guard = lock.lock().unwrap();
                    if guard.is_empty() {
                        guard.extend_from_slice(b"second\n");
                        cvar.notify_all();
                        break;
                    }
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        });

        let c2 = client.clone();
        let h = std::thread::spawn(move || c2.request_raw("a").expect("first request succeeds"));
        // Brief delay so the first request has a chance to take
        // the lock before the second arrives. This is a test-only
        // synchronisation, not a production timeout.
        std::thread::sleep(Duration::from_millis(20));
        let resp_b = client
            .request_raw("b")
            .expect("second request succeeds (after lock release)");
        let resp_a = h.join().expect("first thread joins");
        // Order may flip depending on which thread acquired the
        // lock first — assert the set of responses received,
        // matching the writer's output.
        let mut got = vec![resp_a, resp_b];
        got.sort();
        assert_eq!(got, vec![b"first".to_vec(), b"second".to_vec()]);
        writer.join().unwrap();
    }

    /// A relay-emitted error envelope (`{"ktstr_relay_error":"..."}`)
    /// surfaces as [`SchedStatsError::NoScheduler`].
    #[test]
    fn relay_error_envelope_surfaces_as_no_scheduler() {
        let client = make_client();
        let writer = pre_populate(
            &client,
            br#"{"ktstr_relay_error":"no scheduler running"}
"#,
        );
        let err = client
            .request_raw("x")
            .expect_err("must surface NoScheduler");
        match err {
            SchedStatsError::NoScheduler { reason } => {
                assert_eq!(reason, "no scheduler running");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
        writer.join().unwrap();
    }

    /// Typed `request` round-trips through serde — request
    /// serializes to the scx_stats wire format and response
    /// deserializes from it.
    #[test]
    fn typed_request_round_trip() {
        let req = StatsRequest {
            req: "stats".to_string(),
            args: std::collections::BTreeMap::new(),
        };
        let bytes = serde_json::to_vec(&req).unwrap();
        let decoded: StatsRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.req, "stats");
        assert!(decoded.args.is_empty());

        // Args are serialized as plain JSON strings (matches upstream
        // scx_stats v1.1.0 server.rs:107 `BTreeMap<String, String>`),
        // not wrapped in any other type. A regression to
        // `BTreeMap<String, serde_json::Value>` would emit
        // `{"target":"top"}` either way for string values, so this
        // also pins decoded round-trip equality and rejects integer
        // / object values at parse time on the upstream side.
        let mut req_args = StatsRequest {
            req: "stats".to_string(),
            args: std::collections::BTreeMap::new(),
        };
        req_args.args.insert("target".to_string(), "top".to_string());
        let bytes_args = serde_json::to_vec(&req_args).unwrap();
        assert_eq!(
            std::str::from_utf8(&bytes_args).unwrap(),
            r#"{"req":"stats","args":{"target":"top"}}"#,
        );
        let decoded_args: StatsRequest = serde_json::from_slice(&bytes_args).unwrap();
        assert_eq!(decoded_args.args.get("target"), Some(&"top".to_string()));

        let resp_wire = br#"{"errno":0,"args":{"resp":{"foo":42}}}"#;
        let resp: StatsResponse = serde_json::from_slice(resp_wire).unwrap();
        assert_eq!(resp.errno, 0);
        assert_eq!(resp.args["resp"]["foo"], 42);
    }

    /// `extract_resp` returns the inner `args["resp"]` payload on
    /// success and surfaces [`SchedStatsError::SchedulerError`] on
    /// non-zero errno.
    #[test]
    fn extract_resp_happy_path_and_errno() {
        let resp_ok = StatsResponse {
            errno: 0,
            args: serde_json::json!({"resp": {"counter": 7}}),
        };
        let payload = extract_resp(resp_ok).unwrap();
        assert_eq!(payload["counter"], 7);

        let resp_err = StatsResponse {
            errno: 22,
            args: serde_json::json!({"message": "EINVAL"}),
        };
        let err = extract_resp(resp_err).unwrap_err();
        let downcast = err.downcast_ref::<SchedStatsError>().expect("downcast");
        match downcast {
            SchedStatsError::SchedulerError { errno, args } => {
                assert_eq!(*errno, 22);
                assert_eq!(args["message"], "EINVAL");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    /// `extract_resp` surfaces [`SchedStatsError::MissingResp`]
    /// when `args["resp"]` is absent.
    #[test]
    fn extract_resp_missing_resp_key() {
        let resp = StatsResponse {
            errno: 0,
            args: serde_json::json!({"other": 1}),
        };
        let err = extract_resp(resp).unwrap_err();
        let downcast = err.downcast_ref::<SchedStatsError>().expect("downcast");
        assert!(matches!(downcast, SchedStatsError::MissingResp { .. }));
    }
}
