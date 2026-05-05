//! Virtio-block device with file-backed storage and token-bucket
//! throttle.
//!
//! Single request virtqueue. Advertised features: `VIRTIO_F_VERSION_1`,
//! `VIRTIO_BLK_F_BLK_SIZE`, `VIRTIO_BLK_F_SEG_MAX`,
//! `VIRTIO_BLK_F_SIZE_MAX`, `VIRTIO_BLK_F_FLUSH`,
//! `VIRTIO_RING_F_EVENT_IDX`, plus the optional `VIRTIO_BLK_F_RO`
//! when the disk is configured read-only. MMIO
//! register layout per virtio-v1.2 §4.2.2; block-specific config
//! space at offsets `0x100..` is served from a [`VirtioBlkConfig`]
//! struct whose `repr(C, packed)` layout mirrors the kernel uapi
//! `struct virtio_blk_config` byte-for-byte (virtio-v1.2 §5.2.4).
//! Interrupt delivery via irqfd (eventfd → KVM GSI).
//!
//! Every request flows through chain-shape validation, per-descriptor
//! `SIZE_MAX` enforcement, pre-throttle terminal classification (RO
//! write → `S_IOERR`, RO flush → `S_OK`, unsupported request type →
//! `S_UNSUPP`), then throttle bucket consumption. Validation
//! precedes consumption — a malformed or type-rejected request never
//! drains the bucket or hits `pread`/`pwrite`. See
//! `drain_bracket_impl`.
//!
//! # Execution model: split between vCPU and worker thread
//!
//! The cfg split decides which thread runs `drain_bracket_impl`:
//!
//! - **Production (`cfg(not(test))`):** A dedicated worker thread
//!   (`ktstr-vblk`, spawned in `with_options`) owns the
//!   `BlkWorkerState` for the device's lifetime. The vCPU's
//!   `mmio_write(QUEUE_NOTIFY)` performs a non-blocking
//!   `kick_fd.write(1)` and returns immediately; the worker's
//!   `epoll_wait` resumes and runs one drain iteration per kick.
//!   The vCPU thread never blocks on backing IO — `pread` /
//!   `pwrite` / `fdatasync` all happen on the worker thread, off
//!   the SIGRTMIN-delivery path. The freeze coordinator's
//!   rendezvous timeout therefore is no longer at risk from slow
//!   backing IO. `Drop` writes `stop_fd` and joins the worker.
//!
//! - **Tests (`cfg(test)`):** `process_requests` calls
//!   `drain_inline` on the caller thread synchronously. This
//!   preserves the existing test surface that calls
//!   `process_requests` and immediately reads back queue +
//!   counter state without crossing a thread boundary, and keeps
//!   `dev.worker.queues[REQ_QUEUE].…` direct access valid (the
//!   `BlkQueue` alias resolves to bare `Queue` in test builds).
//!
//! Both paths share the same `drain_bracket_impl` body — the only
//! difference is which thread owns the `BlkWorkerState` it
//! mutates.
//!
//! # Why
//!
//! - **`add_used` gated on status-write success.** A used-ring
//!   advancement without a successfully-written status byte lets
//!   the guest's `virtblk_done` observe its `vbr->in_hdr.status`
//!   byte that's stale from prior blk-mq tag use as `BLK_STS_OK`
//!   — silent data corruption for reads, silent dropped writes
//!   for writes. See `publish_completion`.
//!
//! - **Throttle stalls roll back the chain and arm a timerfd.**
//!   When `can_consume` fails, `drain_bracket_impl` rewinds the
//!   queue cursor with `set_next_avail(prev.wrapping_sub(1))` so
//!   the next pop returns the same head, bumps `throttled_count`,
//!   and returns `DrainOutcome::ThrottleStalled { wait_nanos }`
//!   without writing a status byte, calling `add_used`, or firing
//!   the irqfd — the chain stays invisible to the guest until the
//!   retry. In production the worker arms a CLOCK_MONOTONIC
//!   timerfd registered on its epoll (THROTTLE_TOKEN); when it
//!   fires, the worker re-runs the drain. A QUEUE_NOTIFY kick can
//!   wake the worker before the timer fires; the eventual timer
//!   expiry is then a harmless extra drain. The retry duration is
//!   capped at `RETRY_TIMER_MAX_NANOS` (1 s) — well below the guest's
//!   hung-task watchdog (`kernel.hung_task_timeout_secs`,
//!   default 120 s — virtio_blk has no `mq_ops->timeout` callback
//!   so blk-mq alone never surfaces an unpublished request as an
//!   error) — so a pathological refill rate cannot starve the
//!   guest. The bucket never sleeps: `consume`/`can_consume`
//!   always return promptly, so the worker stays responsive to
//!   STOP_TOKEN and KICK_TOKEN. In `cfg(test)` the inline path
//!   discards `DrainOutcome` because tests step the bucket forward
//!   via `set_last_refill_for_test` and re-issue `QUEUE_NOTIFY` to
//!   exercise the post-stall retry without spawning a worker
//!   thread.
//!
//! # Backing-speed caveat
//!
//! Backend IO is synchronous within `drain_bracket_impl`:
//! `handle_read_impl` / `handle_write_impl` call
//! `FileExt::read_at` / `write_at` (`pread64` / `pwrite64`) and
//! `handle_flush_impl` calls `File::sync_data` (`fdatasync`).
//! There is no `io_uring` and no second-tier async queue — the
//! worker serializes requests through the backing fd one at a
//! time.
//!
//! This is fine when the backing is **fast** — tmpfs (the
//! `tempfile()` default) or warm page cache — where pread / pwrite
//! return in sub-microsecond time and fdatasync is a no-op
//! (`noop_fsync`). With slow backing (cold page cache on spinning
//! media, network-mounted file, fdatasync forcing real journal
//! writes), the worker serializes through it; the guest observes
//! high IO latency, but the vCPU thread is no longer at risk of
//! missing SIGRTMIN. The trade-off shifts: slow backing now means
//! "high guest-observed latency" rather than "stalled vCPU empties
//! the failure dump."
//!
//! v0 still targets small backing files on tmpfs; operators who
//! point a virtio-blk disk at a slow backing simply accept the
//! latency penalty.

// Submodule layout (production code split out for module locality):
//
// - `throttle`: token-bucket primitives + `DiskThrottle`-to-buckets
//   conversion. The throttle is the most-exercised piece of the
//   device, so its types live next to their tests.
// - `worker`: production worker-thread main loop, epoll dispatch
//   tokens, stall-decision policy, and retry-timer clamp. Gated on
//   `cfg(not(test))` for the syscall-bearing pieces; pure helpers
//   (`decide_stall_action`, `worker_dispatch_event`, `clamp_retry_nanos`)
//   are always-compiled so the test block here can drive every
//   variant without spawning a worker.
// - `counters`: `VirtioBlkCounters` struct + `record_*` mutators
//   and `pub fn` readers. The counter taxonomy doc (events vs
//   requests vs gauges) and the per-helper invariants live next
//   to the type they describe.
// - `device`: MMIO read/write, the FSM, the request-state structs,
//   the `VirtioBlk` device, the engine plumbing
//   (handle/reset/respawn), and `Drop`.
// - `handlers`: an `impl VirtioBlk` block with the four
//   `handle_*_impl` per-request-type handlers (T_IN / T_OUT /
//   T_FLUSH / T_GET_ID) and their `cfg(test)` `&self` wrappers.
//   Pure per-request logic with no MMIO/FSM/lifecycle concern.
// - `drain`: `DrainOutcome` and `drain_bracket_impl` — the chain
//   validation, throttle gate, handler dispatch, and completion
//   publish pipeline that runs once per kick.
//
// Tests live in sibling `tests_*.rs` files and reach into module
// internals via `super::*;` — re-exports below glob-route the
// names through `mod.rs`.
//
// Re-exports use `pub(crate) use submodule::*;` so the test modules
// (and `worker.rs`, which `use super::*;` for cross-module references)
// see every item without per-name re-export bookkeeping.

mod throttle;
// `pub(crate) use throttle::*;` feeds the `cfg(test)` modules
// (tests_drain, tests_atomics, tests_handler, etc.) via their
// `use super::*;`. The lib build doesn't reference these symbols
// from this glob (device.rs has its own `use super::throttle::*;`),
// so clippy --lib would otherwise flag the re-export as unused.
#[allow(unused_imports)]
pub(crate) use throttle::*;

mod worker;
// `pub(crate) use worker::*;` feeds the `cfg(test)` modules and
// is referenced from device.rs via direct `use super::worker::*;`
// or per-name imports; clippy --lib otherwise flags this as
// unused for the same reason as throttle above.
#[allow(unused_imports)]
pub(crate) use worker::*;

mod counters;
// `VirtioBlkCounters` is the only pub item in `counters.rs`; it is
// re-exported as `pub` below for upstream consumers (vmm/mod.rs and
// lib.rs). Internal references reach it via `super::VirtioBlkCounters`
// from device/handlers/drain — Rust resolves through the same `pub`
// re-export, so a separate `pub(crate) use counters::*;` glob would be
// redundant (clippy --lib flags it as unused).

mod device;
// The glob is `pub(crate)` so internal items (cfg-test test fixtures,
// `pub(crate)` helpers) reach sibling submodules and the test sub-files
// without leaking outside the crate. The `pub use` block below
// itemizes the symbols that need full `pub` visibility for upstream
// re-exports (vmm/mod.rs and lib.rs re-publish the public-facing
// constants and types); these symbols are themselves `pub` inside
// device.rs, and the explicit listing upgrades the re-export from the
// glob's `pub(crate)` to `pub` for those names only.
pub(crate) use device::*;
// `VIRTIO_BLK_DEFAULT_CAPACITY_BYTES` and `VIRTIO_BLK_SECTOR_SIZE`
// are kept in the `pub` re-export so external consumers can pin
// the same defaults the lib uses internally; the lib's current
// callers reach the constants directly via the device module, so
// the public re-export looks unused in clippy --lib.
pub use counters::VirtioBlkCounters;
#[allow(unused_imports)]
pub use device::{
    VIRTIO_BLK_DEFAULT_CAPACITY_BYTES, VIRTIO_BLK_SECTOR_SIZE, VIRTIO_MMIO_SIZE, VirtioBlk,
    WorkerPlacement,
};

mod handlers;
// `handlers.rs` adds an `impl VirtioBlk` block with the four
// `handle_*_impl` request-type handlers and their `cfg(test)`
// `&self` wrappers. No symbols to re-export — the `impl` block
// extends the type that lives in `device.rs`. `mod handlers;`
// alone wires the file into the build.

mod drain;
// `pub(crate) use drain::*;` exposes `DrainOutcome` and
// `drain_bracket_impl` to `worker.rs` (which references both via
// `super::DrainOutcome` and `super::drain_bracket_impl`) and to
// the test sub-files. The lib build references both via these
// paths, so the glob is consumed without `#[allow(unused_imports)]`.
pub(crate) use drain::*;

#[cfg(test)]
mod testing;

#[cfg(test)]
mod tests_proptest;

#[cfg(test)]
mod tests_atomics;

#[cfg(test)]
mod tests_drain;

#[cfg(test)]
mod tests_drain_validation;

#[cfg(test)]
mod tests_drain_late;

#[cfg(test)]
mod tests_handler;

#[cfg(test)]
mod tests_fsm;

#[cfg(test)]
mod tests_poison;
