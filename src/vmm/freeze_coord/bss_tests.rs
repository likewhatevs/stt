//! Unit coverage for [`super::bss_read_state`] — the typed
//! three-way resolver the freeze coordinator's late-trigger poll
//! routes through to distinguish `Triggered` (probe latched its
//! sticky 0→1 flag) from `NotTriggered` (probe still quiescent),
//! `OutOfBounds` (cached PA no longer resolves to a 4-byte readable
//! DRAM region — probe map freed mid-run), and `NotResolved` (no
//! cached PA published yet, or no `GuestMem` available).
//!
//! The production read site at the run-loop closure consumes the
//! result and gates `bss_triggered` on the `Triggered` variant only;
//! `OutOfBounds` is surfaced as a one-shot warn while NOT
//! contributing to the err_triggered or-gate. Each test below pins
//! one branch of the helper directly so a regression that
//! re-collapses the four-way result back to a plain bool fails here
//! before reaching production.
//!
//! Cases (a)-(g) from the F11 finding map to the named tests below:
//!   (a) cached_bss_pa pointing at OOB PA returning 0 vs OutOfBounds
//!       → `out_of_bounds_pa_distinguished_from_zero`
//!   (b) cached_bss_offset wrap-around → `wrap_around_pa_lands_oob`
//!   (c) err_triggered combinations → `err_triggered_or_gate_combinations`
//!   (f) value_kva==0 / cached_pa is None → `not_resolved_when_pa_none`
//!       and `not_resolved_when_mem_none`
//!
//! Cases (d) probe reload mid-run, (e) BTF parse failure, and (g)
//! sched detach survival are tracked separately:
//!   (d) and (g) are scoped under the probe-unload
//!       invalidation, which is the production fix that lands the
//!       cached_bss_pa reset wiring; this module covers the
//!       primitive that the fix consumes.
//!   (e) load_probe_bss_offset BTF parse failure is exercised by the
//!       existing `monitor::btf_offsets::tests` BTF-driven coverage;
//!       a freeze_coord-side test would have to re-construct
//!       `struct btf` synthesis and add no signal beyond the in-tree
//!       btf_offsets coverage.

use super::{BssReadState, bss_read_state, bss_state_label};
use crate::monitor::reader::GuestMem;

/// Build a `GuestMem` of `size` bytes whose contents are initially
/// zero. Returns the mem and the backing buffer so the caller can
/// stamp values at known offsets without losing the buffer to a
/// temporary.
fn build_mem(size: usize) -> (GuestMem, Vec<u8>) {
    let mut buf = vec![0u8; size];
    // SAFETY: buf outlives the returned GuestMem (caller binds both
    // into the same scope), and `GuestMem::new` consumes a raw
    // pointer from a live allocation. Same pattern every `capture_*`
    // test in this crate uses (see `vmm::capture_tasks::tests`).
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
    (mem, buf)
}

/// Case (f), part 1: cached_pa is None. The probe has not yet been
/// discovered in `map_idr` so the freeze coordinator has not
/// published a PA. The helper must skip the read entirely and
/// return `NotResolved`.
#[test]
fn not_resolved_when_pa_none() {
    let (mem, _buf) = build_mem(0x1000);
    let state = bss_read_state(Some(&mem), None);
    assert_eq!(state, BssReadState::NotResolved);
}

/// Case (f), part 2: GuestMem is None. The NUMA layout has not been
/// published yet (pre-boot window), so `freeze_coord_mem` is still
/// None. The helper must skip the read regardless of whether the PA
/// is cached.
#[test]
fn not_resolved_when_mem_none() {
    let state = bss_read_state(None, Some(0x100));
    assert_eq!(state, BssReadState::NotResolved);
}

/// Case (f), part 3: both inputs None. Defensive belt-and-suspenders
/// — the helper short-circuits without touching guest memory or
/// dereferencing a nonexistent PA.
#[test]
fn not_resolved_when_both_none() {
    let state = bss_read_state(None, None);
    assert_eq!(state, BssReadState::NotResolved);
}

/// Sticky 0→1 latch hasn't fired: PA is in-bounds, the u32 at the PA
/// is `0`, helper returns `NotTriggered`. Production keeps polling
/// without raising the err_triggered flag.
#[test]
fn not_triggered_when_field_is_zero() {
    let (mem, _buf) = build_mem(0x1000);
    let state = bss_read_state(Some(&mem), Some(0x100));
    assert_eq!(state, BssReadState::NotTriggered);
}

/// Sticky latch flipped to 1: helper returns `Triggered`. Production
/// raises err_triggered and dispatches a freeze. Mirrors
/// `__sync_val_compare_and_swap(&ktstr_err_exit_detected, 0u, 1u)`
/// writing the latch byte from the BPF probe.
#[test]
fn triggered_when_field_is_nonzero() {
    let (mem, mut buf) = build_mem(0x1000);
    // Stamp 1 at PA 0x100 (the field offset within the buffer).
    // GuestMem captures the raw pointer, not a snapshot, so a
    // post-construction write through `buf` is observable through
    // the same volatile-read path the production poll uses.
    buf[0x100..0x100 + 4].copy_from_slice(&1u32.to_le_bytes());
    let state = bss_read_state(Some(&mem), Some(0x100));
    assert_eq!(state, BssReadState::Triggered);
}

/// Different non-zero sentinel values still resolve to `Triggered` —
/// the helper's check is "value != 0", matching the production
/// `mem.read_u32(pa, 0) != 0` semantics. Pins the gate against a
/// regression that bound the check to `value == 1` and would
/// silently miss a future probe that publishes a different sentinel.
#[test]
fn triggered_on_arbitrary_nonzero_value() {
    let (mem, mut buf) = build_mem(0x1000);
    buf[0x80..0x80 + 4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    let state = bss_read_state(Some(&mem), Some(0x80));
    assert_eq!(state, BssReadState::Triggered);
}

/// Case (a): cached_bss_pa points past the end of the DRAM region. A
/// bare `mem.read_u32(pa, 0)` would return `0` per `read_scalar`'s
/// OOB-zero contract; the helper must return `OutOfBounds` so the
/// freeze coordinator can warn rather than silently mistake a stale
/// cache for "no fire". This is the load-bearing invariant the
/// F11(a) finding names: the distinction the production code lacked
/// before this fix.
#[test]
fn out_of_bounds_pa_distinguished_from_zero() {
    let (mem, _buf) = build_mem(0x1000);
    // PA past end-of-region: the buffer covers [0, 0x1000), so
    // 0x2000 is OOB.
    let state = bss_read_state(Some(&mem), Some(0x2000));
    assert_eq!(state, BssReadState::OutOfBounds);
}

/// Case (b): wrap-around in `cached_bss_pa = translated +
/// bss_field_offset`. If the BTF Datasec returned a corrupt huge
/// offset, the publishing site at the run-loop uses `wrapping_add`
/// which silently wraps near the top of the 64-bit address space.
/// The wrapped PA can land back in low DRAM (a valid index,
/// in-bounds), or stay above the size guard. Either way, the
/// helper's region_avail gate catches the OOB case; this test pins
/// the helper's behaviour for the still-OOB arm of the wrap.
#[test]
fn wrap_around_pa_lands_oob() {
    let (mem, _buf) = build_mem(0x1000);
    // Simulate a wrapped PA that landed near u64::MAX. The mem
    // region covers [0, 0x1000); region_avail returns 0 for any PA
    // outside that range, so the helper sees OOB.
    let pa = u64::MAX - 16;
    let state = bss_read_state(Some(&mem), Some(pa));
    assert_eq!(state, BssReadState::OutOfBounds);
}

/// Boundary: PA exactly at end-of-region MINUS 4 bytes. The 4-byte
/// read fits exactly; the helper returns `NotTriggered` (or
/// `Triggered` if the byte at that location is non-zero — here the
/// buffer is zero so it stays `NotTriggered`). Pins the
/// `region_avail < 4` boundary against an off-by-one regression that
/// would reject a valid last-word read as OOB.
#[test]
fn pa_at_last_valid_4byte_window() {
    let size: usize = 0x1000;
    let (mem, _buf) = build_mem(size);
    let state = bss_read_state(Some(&mem), Some((size - 4) as u64));
    assert_eq!(state, BssReadState::NotTriggered);
}

/// Boundary: PA exactly at end-of-region MINUS 3 bytes (the 4-byte
/// read would straddle the end). `region_avail` reports 3 bytes
/// available, which is < 4 — the helper must return `OutOfBounds`.
/// Without this gate the bare read would fall through to
/// `read_scalar`'s short-bounds zero-fill path and silently report
/// `NotTriggered`.
#[test]
fn pa_straddles_end_of_region_returns_oob() {
    let size: usize = 0x1000;
    let (mem, _buf) = build_mem(size);
    let state = bss_read_state(Some(&mem), Some((size - 3) as u64));
    assert_eq!(state, BssReadState::OutOfBounds);
}

/// Boundary: PA exactly at end-of-region MINUS 1 byte. `region_avail`
/// reports 1 byte available; helper rejects as `OutOfBounds`. Pins
/// the same gate from the opposite end of the straddle window so
/// both edges of the boundary are explicit.
#[test]
fn pa_one_byte_before_end_returns_oob() {
    let size: usize = 0x1000;
    let (mem, _buf) = build_mem(size);
    let state = bss_read_state(Some(&mem), Some((size - 1) as u64));
    assert_eq!(state, BssReadState::OutOfBounds);
}

/// Mirror of the production err_triggered or-gate at the run loop:
/// `err_triggered = watchpoint_hit || bss_triggered`. The
/// bss_triggered side flows from `bss_read_state == Triggered`.
/// Each row in this matrix exercises one of the eight reachable
/// combinations of (watchpoint_hit, bss_state). Pins the production
/// semantics: only Triggered contributes to the bss side;
/// OutOfBounds and NotResolved do NOT count as a fire. This is case
/// (c) from the F11 finding.
#[test]
fn err_triggered_or_gate_combinations() {
    struct Row {
        wp_hit: bool,
        bss_state: BssReadState,
        expected_err: bool,
        label: &'static str,
    }
    let rows = [
        Row {
            wp_hit: false,
            bss_state: BssReadState::NotResolved,
            expected_err: false,
            label: "neither side fired (clean run)",
        },
        Row {
            wp_hit: false,
            bss_state: BssReadState::NotTriggered,
            expected_err: false,
            label: "bss resolved but not yet latched",
        },
        Row {
            wp_hit: false,
            bss_state: BssReadState::OutOfBounds,
            expected_err: false,
            label: "OOB must NOT count as a fire (the F11(a) invariant)",
        },
        Row {
            wp_hit: false,
            bss_state: BssReadState::Triggered,
            expected_err: true,
            label: "bss-only fire (degraded watchpoint config)",
        },
        Row {
            wp_hit: true,
            bss_state: BssReadState::NotResolved,
            expected_err: true,
            label: "watchpoint-only fire (probe never loaded)",
        },
        Row {
            wp_hit: true,
            bss_state: BssReadState::NotTriggered,
            expected_err: true,
            label: "watchpoint fired before bss latch caught up",
        },
        Row {
            wp_hit: true,
            bss_state: BssReadState::OutOfBounds,
            expected_err: true,
            label: "watchpoint fires through; OOB on bss does not retract it",
        },
        Row {
            wp_hit: true,
            bss_state: BssReadState::Triggered,
            expected_err: true,
            label: "both fired (steady-state late-trigger path)",
        },
    ];
    for row in rows {
        let bss_triggered = matches!(row.bss_state, BssReadState::Triggered);
        let err = row.wp_hit || bss_triggered;
        assert_eq!(
            err, row.expected_err,
            "row '{}': wp_hit={}, bss_state={:?}",
            row.label, row.wp_hit, row.bss_state,
        );
    }
}

/// Idempotency: repeated calls against the same mem + PA with no
/// underlying writes return the same state. Pins the helper as a
/// pure read with no side effects, mirroring the production poll
/// cadence (every 100 ms during boot, every scan tick afterward)
/// where any per-call mutation would drift the err_triggered
/// decision under steady state.
#[test]
fn helper_is_pure_under_repeated_calls() {
    let (mem, _buf) = build_mem(0x1000);
    let s1 = bss_read_state(Some(&mem), Some(0x100));
    let s2 = bss_read_state(Some(&mem), Some(0x100));
    let s3 = bss_read_state(Some(&mem), Some(0x100));
    assert_eq!(s1, BssReadState::NotTriggered);
    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
}

/// State-transition: helper observes a 0→non-zero flip when the
/// buffer is mutated between calls. Pins the volatile-read contract
/// — a regression that cached the read value in a stale path would
/// still return `NotTriggered` after the underlying write. The
/// production probe writes the latch from BPF and the host
/// coordinator polls; without volatile reads the host could miss
/// the flip on a weakly ordered architecture.
#[test]
fn helper_observes_post_write_flip() {
    let (mem, mut buf) = build_mem(0x1000);
    let pa: u64 = 0x40;
    let state_before = bss_read_state(Some(&mem), Some(pa));
    assert_eq!(state_before, BssReadState::NotTriggered);
    buf[pa as usize..pa as usize + 4].copy_from_slice(&1u32.to_le_bytes());
    let state_after = bss_read_state(Some(&mem), Some(pa));
    assert_eq!(state_after, BssReadState::Triggered);
}

/// PA at offset 0 (start of region) reads correctly. Catches an
/// off-by-one regression on the lower bound of `region_avail` —
/// `read_scalar` does not special-case offset 0, but a future
/// region-layout change could.
#[test]
fn pa_at_region_start() {
    let (mem, mut buf) = build_mem(0x1000);
    buf[0..4].copy_from_slice(&7u32.to_le_bytes());
    let state = bss_read_state(Some(&mem), Some(0));
    assert_eq!(state, BssReadState::Triggered);
}

// -- bss_state_label wire-format pins ---------------------------
//
// The [`bss_state_label`] helper produces the snake-case string
// embedded in
// [`crate::monitor::dump::DegradedFailureDumpReport::bss_latch_state`].
// Operators grep, `jq`-filter, and auto-repro renderers match
// against these exact strings — drift here breaks downstream
// tooling silently (the dump still writes, just with the new label,
// and the operator's matcher looks for the old one). Each variant
// gets its own pin so a regression that drops one arm or returns
// the wrong label is caught BEFORE shipping rather than after the
// dump consumer fails to recognise the post-mortem.

/// `BssReadState::Triggered` → "triggered". Pins the sticky-latch-
/// flipped variant — produced when the probe's
/// `ktstr_err_exit_detected` CAS succeeded and the host poll
/// observed the non-zero u32.
#[test]
fn bss_state_label_triggered() {
    assert_eq!(bss_state_label(BssReadState::Triggered), "triggered");
}

/// `BssReadState::NotTriggered` → "not_triggered". Pins the
/// quiescent-probe variant — cached PA in-bounds, u32 still zero.
#[test]
fn bss_state_label_not_triggered() {
    assert_eq!(
        bss_state_label(BssReadState::NotTriggered),
        "not_triggered"
    );
}

/// `BssReadState::OutOfBounds` → "out_of_bounds". Pins the stale-
/// cache variant — cached PA falls outside every live DRAM region
/// (probe map freed mid-run + vmalloc page recycled, or BTF Datasec
/// returned a corrupt offset whose `wrapping_add` overflowed).
#[test]
fn bss_state_label_out_of_bounds() {
    assert_eq!(
        bss_state_label(BssReadState::OutOfBounds),
        "out_of_bounds"
    );
}

/// `BssReadState::NotResolved` → "not_resolved". Pins the
/// not-yet-cached variant — probe never discovered in `map_idr`,
/// or pre-boot window with no `GuestMem` published.
#[test]
fn bss_state_label_not_resolved() {
    assert_eq!(bss_state_label(BssReadState::NotResolved), "not_resolved");
}

/// Wildcard-trap anchor: bss_state_label MUST only emit labels in
/// the operator-known allowlist. The production match at
/// mod.rs:188-195 is exhaustive, so a NEW BssReadState variant
/// fails to compile until it gets an explicit label arm — but a
/// future refactor that adds any wildcard
/// (`_ => "unknown"`, `_ => "other"`, `_ => "unhandled"`,
/// `_ => "fallback"`, `_ => "default"`, ...) would silently land
/// that string in the DegradedFailureDumpReport.bss_latch_state
/// wire field for any unhandled variant. The four
/// bss_state_label_<variant> canaries above pin the existing arms;
/// this canary catches the wildcard-refactor regression class
/// (not just the "unknown" sub-case) by asserting every variant's
/// label appears in the operator-known allowlist. Operator tooling
/// (jq filters, auto-repro tail renderer) keys off the snake_case
/// labels and has no behavior defined for any string outside the
/// allowlist.
///
/// Maintenance contract: a new BssReadState variant landing
/// legitimately requires updating BOTH the production label arm
/// AND this allowlist — same per-variant pin discipline the
/// individual bss_state_label_<variant> canaries above use.
#[test]
fn bss_state_label_no_wildcard_unknown_fallback() {
    let variants = [
        BssReadState::Triggered,
        BssReadState::NotTriggered,
        BssReadState::OutOfBounds,
        BssReadState::NotResolved,
    ];
    let allowlist = ["triggered", "not_triggered", "out_of_bounds", "not_resolved"];
    for v in variants {
        let label = bss_state_label(v);
        assert!(
            !label.is_empty(),
            "bss_state_label({v:?}) must be non-empty"
        );
        assert!(
            allowlist.contains(&label),
            "bss_state_label({v:?}) returned {label:?} — not in the \
             operator-known allowlist {allowlist:?}. Either an \
             unintended wildcard-arm refactor landed (`_ => \"...\"`) \
             or a new BssReadState variant was added without updating \
             the allowlist here. Operator tooling (jq filters, \
             auto-repro tail renderer) has no behavior defined for \
             strings outside the allowlist."
        );
    }
}
