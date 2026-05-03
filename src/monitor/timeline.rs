//! Failure-dump timeline support.
//!
//! Two related primitives live here:
//!
//! 1. **Sched-event timeline** ([`TimelineEvent`], [`parse_timeline_buf`],
//!    [`TimelineCapture`]) — host-side consumer for the
//!    `timeline_events` BPF ringbuf populated by `tp_btf/sched_switch`,
//!    `sched_migrate_task`, and `sched_wakeup` handlers (see
//!    `src/bpf/probe.bpf.c::ktstr_tl_*`). The freeze coordinator
//!    drains the ringbuf at failure time, parses the records into
//!    [`TimelineEvent`] values, and stitches them into the failure
//!    dump.
//!
//! 2. **Incremental snapshot ring** ([`SnapshotRing`],
//!    [`IncrementalCapture`]) — periodic VM-freeze capture of raw
//!    BPF state bytes for deferred render at trigger time. Cadence,
//!    ring depth, and rendering policy are tuned per consumer; the
//!    [`DEFAULT_SNAPSHOT_RING_DEPTH`] constant pins the storage
//!    budget at 60 entries.
//!
//! Both surfaces compose into [`super::dump::DumpContext`] as
//! optional captures so frozen-VM and live-host pipelines can
//! supply (or omit) them independently.
//!
//! # Layout pinning
//!
//! [`TimelineEvent`] mirrors the on-the-wire `struct timeline_event`
//! defined in `src/bpf/intf.h`. Field order, sizes, and the type
//! constants must stay in lockstep — a unit test
//! ([`tests::timeline_event_layout_pinned`]) verifies the 40-byte
//! footprint and field offsets against the BPF-side layout.

use serde::{Deserialize, Serialize};

/// Type-byte values from `src/bpf/intf.h::TL_EVT_*`. Pinned here as
/// the userspace-facing identifier for each variant; the parser
/// uses these to discriminate the [`TimelineEvent`] variant.
pub mod tl_evt {
    /// `tp_btf/sched_switch` record. `prev_pid`/`next_pid`/`a` (prev_state)/`b` (preempt).
    pub const SWITCH: u32 = 1;
    /// `tp_btf/sched_migrate_task` record. `prev_pid`/`a` (dest_cpu)/`b` (orig_cpu).
    pub const MIGRATE: u32 = 2;
    /// `tp_btf/sched_wakeup` record. `prev_pid`/`a` (target_cpu).
    pub const WAKEUP: u32 = 3;
    /// `fentry/fexit` rt_mutex_setprio. PI boost record.
    pub const PI_BOOST: u32 = 4;
    /// `tp_btf/lock:contention_begin` record.
    pub const LOCK_CONTEND: u32 = 5;
}

/// Wire-format mirror of `struct timeline_event` from
/// `src/bpf/intf.h`.
///
/// Layout pinning: 40 bytes total (4 type + 4 cpu + 8 ts +
/// 4 prev_pid + 4 next_pid + 8 a + 8 b). Order matches the BPF
/// emit sites in `probe.bpf.c::ktstr_tl_switch/migrate/wakeup`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineEventRaw {
    pub type_: u32,
    pub cpu: u32,
    pub ts: u64,
    pub prev_pid: u32,
    pub next_pid: u32,
    pub a: u64,
    pub b: u64,
}

/// Parsed timeline event with variant-aware field naming.
///
/// `non_exhaustive` so future BPF event types added in `intf.h`
/// (per the TL_EVT_PI_BOOST / TL_EVT_LOCK_CONTEND sites) can land
/// without breaking existing on-disk dumps.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(tag = "kind")]
#[allow(dead_code)] // wired into FailureDumpReport.timeline_events;
                    // freeze coordinator populates via TimelineCapture.
pub enum TimelineEvent {
    /// `tp_btf/sched_switch`. The kernel switched from `prev_pid`
    /// to `next_pid` on `cpu` at `ts` (boot-time ns).
    Switch {
        ts: u64,
        cpu: u32,
        prev_pid: u32,
        next_pid: u32,
        /// Raw `prev_state` bitfield (TASK_RUNNING / TASK_INTERRUPTIBLE
        /// / etc., from `include/linux/sched.h`).
        prev_state: u64,
        /// True when the switch was a preemption (vs voluntary
        /// dequeue).
        preempt: bool,
    },
    /// `tp_btf/sched_migrate_task`. Task `pid` migrated from
    /// `orig_cpu` to `dest_cpu`.
    Migrate {
        ts: u64,
        cpu: u32,
        pid: u32,
        orig_cpu: u32,
        dest_cpu: u32,
    },
    /// `tp_btf/sched_wakeup`. Task `pid` woken up; scheduler
    /// chose `target_cpu` for placement.
    Wakeup {
        ts: u64,
        cpu: u32,
        pid: u32,
        target_cpu: u32,
    },
    /// PI boost. Probe-context tid `prober_tid`; boosted task
    /// `pid`. Old/new prio + sched-class id encoded as
    /// (prio_u32 | (class_u32 << 32)). Field layout per
    /// `src/bpf/intf.h::TL_EVT_PI_BOOST`.
    PiBoost {
        ts: u64,
        cpu: u32,
        prober_tid: u32,
        pid: u32,
        old_prio: u32,
        old_class_id: u32,
        new_prio: u32,
        new_class_id: u32,
    },
    /// Lock contention begin. `tid` is the waiter; `lock_kva` is
    /// the lock's kernel virtual address; `flags` carries the
    /// LCB_* class bits (F_SPIN, F_READ, F_WRITE, F_RT — see
    /// `include/trace/events/lock.h`).
    LockContend {
        ts: u64,
        cpu: u32,
        tid: u32,
        lock_kva: u64,
        flags: u32,
    },
    /// Unrecognized type byte. Library doesn't drop unknown
    /// records — surfacing them as `Unknown` lets the failure
    /// dump preserve forward-compat data the consumer can opt
    /// into rendering later.
    Unknown {
        ts: u64,
        cpu: u32,
        type_: u32,
        prev_pid: u32,
        next_pid: u32,
        a: u64,
        b: u64,
    },
}

/// Parse a single 40-byte ringbuf record.
///
/// Returns `None` when the input is shorter than the on-the-wire
/// size — the caller has truncated buffer / partial read and should
/// stop draining at this slot.
#[allow(dead_code)]
pub fn parse_timeline_record(bytes: &[u8]) -> Option<TimelineEvent> {
    if bytes.len() < std::mem::size_of::<TimelineEventRaw>() {
        return None;
    }
    // SAFETY: TimelineEventRaw is repr(C) plain-data, all fields
    // are integer types so any byte pattern is a valid value.
    // The size check above guarantees we have enough bytes.
    let raw = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const TimelineEventRaw) };
    Some(decode_raw(&raw))
}

/// Parse a contiguous buffer of timeline records into a vec of
/// [`TimelineEvent`] values, in encounter order.
///
/// `bytes` is the concatenation of N timeline_event records.
/// Trailing bytes that don't form a full record are silently
/// dropped (a torn final record at ringbuf wrap is the typical
/// case; the consumer's next drain picks up the remainder).
#[allow(dead_code)]
pub fn parse_timeline_buf(bytes: &[u8]) -> Vec<TimelineEvent> {
    let stride = std::mem::size_of::<TimelineEventRaw>();
    let mut out = Vec::with_capacity(bytes.len() / stride);
    let mut off = 0;
    while off + stride <= bytes.len() {
        if let Some(ev) = parse_timeline_record(&bytes[off..off + stride]) {
            out.push(ev);
        }
        off += stride;
    }
    out
}

fn decode_raw(raw: &TimelineEventRaw) -> TimelineEvent {
    match raw.type_ {
        tl_evt::SWITCH => TimelineEvent::Switch {
            ts: raw.ts,
            cpu: raw.cpu,
            prev_pid: raw.prev_pid,
            next_pid: raw.next_pid,
            prev_state: raw.a,
            preempt: raw.b != 0,
        },
        tl_evt::MIGRATE => TimelineEvent::Migrate {
            ts: raw.ts,
            cpu: raw.cpu,
            pid: raw.prev_pid,
            orig_cpu: raw.b as u32,
            dest_cpu: raw.a as u32,
        },
        tl_evt::WAKEUP => TimelineEvent::Wakeup {
            ts: raw.ts,
            cpu: raw.cpu,
            pid: raw.prev_pid,
            target_cpu: raw.a as u32,
        },
        tl_evt::PI_BOOST => {
            let old_prio = (raw.a & 0xffff_ffff) as u32;
            let old_class_id = (raw.a >> 32) as u32;
            let new_prio = (raw.b & 0xffff_ffff) as u32;
            let new_class_id = (raw.b >> 32) as u32;
            TimelineEvent::PiBoost {
                ts: raw.ts,
                cpu: raw.cpu,
                prober_tid: raw.prev_pid,
                pid: raw.next_pid,
                old_prio,
                old_class_id,
                new_prio,
                new_class_id,
            }
        }
        tl_evt::LOCK_CONTEND => TimelineEvent::LockContend {
            ts: raw.ts,
            cpu: raw.cpu,
            tid: raw.prev_pid,
            lock_kva: raw.a,
            flags: raw.b as u32,
        },
        _ => TimelineEvent::Unknown {
            ts: raw.ts,
            cpu: raw.cpu,
            type_: raw.type_,
            prev_pid: raw.prev_pid,
            next_pid: raw.next_pid,
            a: raw.a,
            b: raw.b,
        },
    }
}

/// Capture handle for the freeze coordinator's drain of the
/// `timeline_events` BPF ringbuf.
///
/// At dump time the coordinator constructs this with the drained
/// raw bytes (concatenated 40-byte records, in ringbuf order) plus
/// the BSS-side drop count. The dump consumer parses the buffer
/// into [`TimelineEvent`] values and surfaces both alongside
/// [`super::dump::FailureDumpReport::timeline_events`] /
/// `timeline_drops`.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct TimelineCapture<'a> {
    /// Raw concatenated record bytes drained from the
    /// `timeline_events` ringbuf. Length must be a multiple of
    /// `size_of::<TimelineEventRaw>()` (40); trailing partial
    /// records are silently dropped at parse time.
    pub records: &'a [u8],
    /// `ktstr_timeline_drops` BSS counter at drain time. Non-zero
    /// indicates the BPF producer hit a full ringbuf and dropped
    /// the newest event(s) on submit.
    pub drops: u64,
}

// ---------------------------------------------------------------
// Incremental capture: periodic VM-freeze ring of raw bytes.
// ---------------------------------------------------------------

/// Default snapshot-ring depth: 60 entries at 1 Hz steady-state
/// covers 60 seconds of pre-trigger context — long enough for the
/// dual-snapshot delta to detect slow drift, short enough that the
/// storage cost stays within the 60-300 MiB envelope of the per-VM
/// budget for incremental capture.
pub const DEFAULT_SNAPSHOT_RING_DEPTH: usize = 60;

/// One incremental snapshot — opaque raw bytes captured at a
/// particular freeze instant + the wall-clock ts of capture.
///
/// "Opaque bytes" because the producer captures BPF state without
/// rendering it. Render is deferred until trigger fires; the
/// failure dump's renderer parses the buffer through the same
/// pipeline as a regular failure dump.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
#[allow(dead_code)]
pub struct IncrementalSnapshot {
    /// Wall-clock timestamp at freeze instant (CLOCK_REALTIME ns).
    /// `0` when the producer didn't stamp it (test fixtures only).
    pub captured_ns: u64,
    /// Boot-time (CLOCK_MONOTONIC_RAW) ns at freeze instant.
    /// Co-axial with `captured_ns`; the dump renderer pairs the
    /// two so the timeline is anchored against both wall-clock
    /// (operator readability) and monotonic (delta math).
    pub monotonic_ns: u64,
    /// Raw captured bytes. Shape is producer-defined: typically
    /// the `.bss` value buffer of the scheduler's BPF object,
    /// concatenated with a serialized [`super::scx_walker`]
    /// snapshot. The renderer treats it as opaque until
    /// trigger time.
    pub bytes: Vec<u8>,
}

/// Bounded ring of [`IncrementalSnapshot`] values.
///
/// Newest snapshot at the back; the oldest is dropped when the ring
/// fills (single-producer-single-consumer pattern — the freeze
/// coordinator pushes, the failure-dump renderer drains).
///
/// Cheap to clone for sidecar persistence: `Vec` of `Vec<u8>`
/// shares no internal state, so the capture-mode binary can
/// snapshot the ring at trigger time without blocking the
/// coordinator's continued sampling.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SnapshotRing {
    capacity: usize,
    snapshots: std::collections::VecDeque<IncrementalSnapshot>,
}

impl SnapshotRing {
    /// New ring with the requested capacity. Pass
    /// [`DEFAULT_SNAPSHOT_RING_DEPTH`] for the storage-budget-tuned
    /// default (60 entries).
    #[allow(dead_code)]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            snapshots: std::collections::VecDeque::with_capacity(capacity.max(1)),
        }
    }

    /// Push a new snapshot. When the ring is full, the oldest
    /// entry is dropped (FIFO eviction).
    #[allow(dead_code)]
    pub fn push(&mut self, snap: IncrementalSnapshot) {
        if self.snapshots.len() == self.capacity {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snap);
    }

    /// Number of snapshots currently held.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// True when the ring holds no snapshots.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Capacity (the depth this ring was constructed with).
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Drain every held snapshot into a vec, oldest-first. Used by
    /// the failure-dump renderer at trigger time to consume the
    /// pre-trigger window.
    #[allow(dead_code)]
    pub fn drain(&mut self) -> Vec<IncrementalSnapshot> {
        self.snapshots.drain(..).collect()
    }

    /// Borrow the held snapshots for read-only inspection (e.g.
    /// the capture-mode binary's "show me what's in the ring
    /// right now" diagnostic).
    #[allow(dead_code)]
    pub fn snapshots(&self) -> impl Iterator<Item = &IncrementalSnapshot> {
        self.snapshots.iter()
    }
}

/// Capture handle the freeze coordinator passes into the dump
/// pipeline when periodic incremental snapshots are enabled.
///
/// At dump time the renderer drains [`Self::snapshots`] and emits
/// each snapshot as a [`super::dump::FailureDumpReport`]-shaped
/// record under
/// `super::dump::FailureDumpReport::incremental_snapshots`.
/// `None` capture means the freeze coordinator wasn't running the
/// periodic loop — typical for one-shot dumps where the
/// dual-snapshot delta is enough.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct IncrementalCapture {
    /// Pre-trigger ring of raw snapshots. Producer drains the
    /// ring into this vec at trigger time; the dump renderer
    /// parses each snapshot's bytes via the same path as a
    /// regular failure dump.
    pub snapshots: Vec<IncrementalSnapshot>,
    /// Steady-state sampling frequency (Hz) — typically 1.
    /// Surfaced so the operator can correlate snapshot timing
    /// against the failure window.
    pub steady_hz: f64,
    /// Escalation frequency (Hz) used during stall detection —
    /// typically 10. Reflects the actual frequency at trigger
    /// time, not the configured ceiling.
    pub trigger_hz: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TimelineEventRaw` matches the BPF-side struct timeline_event
    /// in size and field offsets. Drift here is a wire-protocol
    /// break; the test catches it at compile + run time.
    ///
    /// Verdict-routed so a multi-field layout regression (e.g.
    /// somebody re-orders the struct) surfaces every drift in one
    /// run rather than failing on the first mismatch.
    #[test]
    fn timeline_event_layout_pinned() {
        use crate::assert::Verdict;

        let total_size = std::mem::size_of::<TimelineEventRaw>();
        let off_type = std::mem::offset_of!(TimelineEventRaw, type_);
        let off_cpu = std::mem::offset_of!(TimelineEventRaw, cpu);
        let off_ts = std::mem::offset_of!(TimelineEventRaw, ts);
        let off_prev_pid = std::mem::offset_of!(TimelineEventRaw, prev_pid);
        let off_next_pid = std::mem::offset_of!(TimelineEventRaw, next_pid);
        let off_a = std::mem::offset_of!(TimelineEventRaw, a);
        let off_b = std::mem::offset_of!(TimelineEventRaw, b);

        let mut v = Verdict::new();
        // Total: 4 + 4 + 8 + 4 + 4 + 8 + 8 = 40 bytes.
        crate::claim!(v, total_size).eq(40usize);
        // Field offsets matching src/bpf/intf.h::struct timeline_event.
        crate::claim!(v, off_type).eq(0usize);
        crate::claim!(v, off_cpu).eq(4usize);
        crate::claim!(v, off_ts).eq(8usize);
        crate::claim!(v, off_prev_pid).eq(16usize);
        crate::claim!(v, off_next_pid).eq(20usize);
        crate::claim!(v, off_a).eq(24usize);
        crate::claim!(v, off_b).eq(32usize);
        let r = v.into_result();
        assert!(
            r.passed,
            "timeline_event layout drift detected: {:?}",
            r.details,
        );
    }

    fn raw(type_: u32, cpu: u32, ts: u64, p: u32, n: u32, a: u64, b: u64) -> Vec<u8> {
        let r = TimelineEventRaw {
            type_,
            cpu,
            ts,
            prev_pid: p,
            next_pid: n,
            a,
            b,
        };
        // Safety: r is plain-data; reading its bytes is well-defined.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &r as *const TimelineEventRaw as *const u8,
                std::mem::size_of::<TimelineEventRaw>(),
            )
        };
        bytes.to_vec()
    }

    /// Switch record decodes with prev/next pids + prev_state +
    /// preempt bool. Verdict-routed so every field surfaces its own
    /// labeled detail on regression.
    #[test]
    fn parse_switch_record() {
        use crate::assert::Verdict;

        let bytes = raw(tl_evt::SWITCH, 3, 1_000_000, 100, 200, 0x402, 1);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Switch {
                ts,
                cpu,
                prev_pid,
                next_pid,
                prev_state,
                preempt,
            } => {
                let mut v = Verdict::new();
                crate::claim!(v, ts).eq(1_000_000u64);
                crate::claim!(v, cpu).eq(3u32);
                crate::claim!(v, prev_pid).eq(100u32);
                crate::claim!(v, next_pid).eq(200u32);
                crate::claim!(v, prev_state).eq(0x402u64);
                crate::claim!(v, preempt).eq(true);
                let r = v.into_result();
                assert!(
                    r.passed,
                    "Switch record decode drift: {:?}",
                    r.details,
                );
            }
            other => panic!("expected Switch, got {other:?}"),
        }
    }

    /// Migrate record decodes with pid + orig_cpu + dest_cpu.
    /// Per intf.h: a = dest_cpu, b = orig_cpu.
    #[test]
    fn parse_migrate_record() {
        let bytes = raw(tl_evt::MIGRATE, 1, 2_000_000, 555, 0, 7, 2);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Migrate {
                pid,
                orig_cpu,
                dest_cpu,
                ..
            } => {
                assert_eq!(pid, 555);
                assert_eq!(dest_cpu, 7);
                assert_eq!(orig_cpu, 2);
            }
            other => panic!("expected Migrate, got {other:?}"),
        }
    }

    /// Wakeup record decodes with pid + target_cpu.
    #[test]
    fn parse_wakeup_record() {
        let bytes = raw(tl_evt::WAKEUP, 0, 3_000_000, 777, 0, 4, 0);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Wakeup {
                pid, target_cpu, ..
            } => {
                assert_eq!(pid, 777);
                assert_eq!(target_cpu, 4);
            }
            other => panic!("expected Wakeup, got {other:?}"),
        }
    }

    /// PiBoost record splits a/b into (prio | class_id<<32) pairs.
    #[test]
    fn parse_pi_boost_record() {
        let old_a = 120u64 | (3u64 << 32); // prio=120, class_id=3 (rt)
        let new_b = 100u64 | (1u64 << 32); // prio=100, class_id=1 (cfs)
        let bytes = raw(tl_evt::PI_BOOST, 2, 4_000_000, 10, 11, old_a, new_b);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::PiBoost {
                prober_tid,
                pid,
                old_prio,
                old_class_id,
                new_prio,
                new_class_id,
                ..
            } => {
                assert_eq!(prober_tid, 10);
                assert_eq!(pid, 11);
                assert_eq!(old_prio, 120);
                assert_eq!(old_class_id, 3);
                assert_eq!(new_prio, 100);
                assert_eq!(new_class_id, 1);
            }
            other => panic!("expected PiBoost, got {other:?}"),
        }
    }

    /// LockContend record carries lock_kva + flags.
    #[test]
    fn parse_lock_contend_record() {
        let lock_kva = 0xffff_ffff_8000_1000u64;
        let flags = 0x4u64;
        let bytes = raw(tl_evt::LOCK_CONTEND, 5, 5_000_000, 99, 0, lock_kva, flags);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::LockContend {
                tid,
                lock_kva: kva,
                flags: f,
                ..
            } => {
                assert_eq!(tid, 99);
                assert_eq!(kva, lock_kva);
                assert_eq!(f, 0x4);
            }
            other => panic!("expected LockContend, got {other:?}"),
        }
    }

    /// Unknown type byte surfaces as Unknown variant — preserves
    /// forward-compat data for newer kernels with TL_EVT_* values
    /// the consumer doesn't yet decode.
    #[test]
    fn parse_unknown_type_preserves_fields() {
        let bytes = raw(99, 7, 6_000_000, 1, 2, 3, 4);
        let ev = parse_timeline_record(&bytes).unwrap();
        match ev {
            TimelineEvent::Unknown {
                type_,
                prev_pid,
                a,
                b,
                ..
            } => {
                assert_eq!(type_, 99);
                assert_eq!(prev_pid, 1);
                assert_eq!(a, 3);
                assert_eq!(b, 4);
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    /// Truncated record returns None — the drain loop stops
    /// parsing rather than reading past end-of-buffer.
    #[test]
    fn parse_truncated_record_returns_none() {
        let bytes = vec![0u8; 39]; // 1 byte short of 40
        assert!(parse_timeline_record(&bytes).is_none());
    }

    /// `parse_timeline_buf` parses every full record in a multi-
    /// record buffer and silently drops a partial trailing record.
    #[test]
    fn parse_timeline_buf_multi_record_with_partial_tail() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend(raw(tl_evt::SWITCH, 0, 1, 1, 2, 0, 0));
        buf.extend(raw(tl_evt::WAKEUP, 1, 2, 3, 0, 4, 0));
        // Append 20 bytes of partial record — must not parse.
        buf.extend(vec![0u8; 20]);
        let evs = parse_timeline_buf(&buf);
        assert_eq!(evs.len(), 2);
        assert!(matches!(evs[0], TimelineEvent::Switch { .. }));
        assert!(matches!(evs[1], TimelineEvent::Wakeup { .. }));
    }

    /// Snapshot ring bounds at capacity and evicts oldest first.
    #[test]
    fn snapshot_ring_evicts_oldest() {
        let mut ring = SnapshotRing::new(3);
        for i in 0..5 {
            ring.push(IncrementalSnapshot {
                captured_ns: i,
                monotonic_ns: i,
                bytes: vec![i as u8],
            });
        }
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.capacity(), 3);
        let drained = ring.drain();
        assert_eq!(drained.len(), 3);
        // After eviction we should have ts=2,3,4 (oldest 0,1 dropped).
        assert_eq!(drained[0].captured_ns, 2);
        assert_eq!(drained[2].captured_ns, 4);
    }

    /// Default ring depth matches the documented 60-entry budget.
    #[test]
    fn default_ring_depth_pinned() {
        assert_eq!(DEFAULT_SNAPSHOT_RING_DEPTH, 60);
    }

    /// New ring is empty and matches its capacity.
    #[test]
    fn snapshot_ring_starts_empty() {
        let ring = SnapshotRing::new(8);
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.capacity(), 8);
    }

    /// `IncrementalSnapshot` round-trips through serde so off-disk
    /// captures parse on reload.
    #[test]
    fn incremental_snapshot_serde_roundtrip() {
        let snap = IncrementalSnapshot {
            captured_ns: 1234567890,
            monotonic_ns: 9876543210,
            bytes: vec![1, 2, 3, 4],
        };
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: IncrementalSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.captured_ns, 1234567890);
        assert_eq!(parsed.bytes, vec![1, 2, 3, 4]);
    }

    /// `TimelineEvent` round-trips through serde — every variant
    /// survives the json string.
    #[test]
    fn timeline_event_serde_roundtrip_all_variants() {
        let cases = vec![
            TimelineEvent::Switch {
                ts: 1,
                cpu: 0,
                prev_pid: 10,
                next_pid: 20,
                prev_state: 1,
                preempt: false,
            },
            TimelineEvent::Migrate {
                ts: 2,
                cpu: 1,
                pid: 30,
                orig_cpu: 1,
                dest_cpu: 2,
            },
            TimelineEvent::Wakeup {
                ts: 3,
                cpu: 2,
                pid: 40,
                target_cpu: 5,
            },
            TimelineEvent::PiBoost {
                ts: 4,
                cpu: 3,
                prober_tid: 1,
                pid: 2,
                old_prio: 120,
                old_class_id: 3,
                new_prio: 100,
                new_class_id: 1,
            },
            TimelineEvent::LockContend {
                ts: 5,
                cpu: 4,
                tid: 99,
                lock_kva: 0xffff_ffff,
                flags: 0x4,
            },
        ];
        for ev in cases {
            let json = serde_json::to_string(&ev).expect("serialize");
            let _: TimelineEvent = serde_json::from_str(&json).expect("deserialize");
        }
    }
}
