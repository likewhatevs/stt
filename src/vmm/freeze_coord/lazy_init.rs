#![allow(dead_code)]
//! Lazy retry wrappers for the boot-race-sensitive accessors.
//!
//! Three coordinator-state caches need to be populated lazily because
//! their construction depends on guest-memory bootstrap symbols that
//! the kernel only writes during boot (`page_offset_base`,
//! `pgtable_l5_enabled`, `init_top_pgt`, `__per_cpu_offset[]`):
//!
//! 1. [`crate::monitor::bpf_map::GuestMemMapAccessorOwned`] — backs
//!    every BPF map discovery the freeze coord performs (probe `.bss`
//!    PA cache, watchpoint target resolution, dump map rendering).
//! 2. [`crate::monitor::bpf_prog::GuestMemProgAccessorOwned`] — backs
//!    the prog-runtime-stats capture in `dump_state`.
//! 3. The per-CPU offset array — read once via
//!    [`crate::monitor::symbols::read_per_cpu_offsets`] and cached
//!    for the rest of the run; gated on every entry being non-zero so
//!    a partially-online VM doesn't poison the cache with a CPU 0
//!    alias for not-yet-online CPUs (the rq PA invariant).
//!
//! All three retry blocks share the same `(mem, vmlinux, tcr_el1,
//! cr3)` input shape, which is the GuestKernel handshake context the
//! coordinator captures at run_vm scope. Lifting them into named
//! `pub(super) fn` lets unit tests drive the boot-race window
//! deterministically: a test constructs a `GuestMem` and feeds
//! controlled `(tcr, cr3)` snapshots through `try_init_*` to assert
//! the cache transitions from None → Some on the first successful
//! attempt and stays Some thereafter.
//!
//! # No state-machine semantics change
//!
//! Each `try_init_*` is byte-for-byte identical to the inline
//! retry block: same Acquire load on the cr3 / tcr_el1 atoms (the
//! cr3 cache may flip mid-run as the BSP loop refines the
//! page-table root, so the load happens INSIDE the helper —
//! capturing it pre-call would freeze a stale value), same gate on
//! every per-CPU offset slot being non-zero before caching. The
//! accessor helpers return the constructor's `anyhow::Result`
//! verbatim so the caller can capture the most recent error
//! message and surface it as a warn after enough retries (the
//! per-CPU offsets helper still returns `Option` because its
//! failure mode includes the "any slot still zero" non-error
//! retry condition).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::monitor;

/// Try to construct a [`monitor::bpf_map::GuestMemMapAccessorOwned`]
/// for the lifetime of `mem` and `vmlinux`. Returns `Err` if the
/// constructor's `GuestKernel` handshake fails (still-booting guest
/// has not yet populated the boot-time symbols); caller leaves its
/// cache `None`, retries on the next scan tick, and tracks the most
/// recent `Err` so a permanent failure (e.g. stripped vmlinux missing
/// `map_idr`) can be surfaced as a warn after enough retries instead
/// of disappearing silently behind `.ok()`.
///
/// `tcr_el1` is `Option<&Arc<AtomicU64>>` because aarch64 holds the
/// register cache while x86_64 always passes `None`. The Acquire load
/// happens INSIDE the helper so a fresh value is observed each
/// iteration — capturing it pre-call would freeze a stale snapshot
/// the BSP loop hasn't refined yet.
///
/// `data` is the cached vmlinux bytes the coordinator reads once at
/// run scope; the helper re-parses the ELF on each retry (parsing the
/// cached bytes is microseconds, the original `std::fs::read` was
/// 14-28 s on cold disk cache). `vmlinux` is still passed through for
/// the BTF sidecar cache lookup inside `BpfMapOffsets::from_elf`.
pub(super) fn try_init_owned_accessor(
    mem: Arc<monitor::reader::GuestMem>,
    data: &[u8],
    vmlinux: &std::path::Path,
    tcr_el1: Option<&Arc<AtomicU64>>,
    cr3: &Arc<AtomicU64>,
) -> anyhow::Result<monitor::bpf_map::GuestMemMapAccessorOwned> {
    let tcr_val = tcr_el1.map(|c| c.load(Ordering::Acquire)).unwrap_or(0);
    let cr3_val = cr3.load(Ordering::Acquire);
    let elf = goblin::elf::Elf::parse(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF: {e}"))?;
    monitor::bpf_map::GuestMemMapAccessorOwned::from_elf(
        mem, &elf, data, vmlinux, tcr_val, cr3_val,
    )
}

/// Sibling of [`try_init_owned_accessor`] for the prog-side
/// accessor. Same boot-race rationale, same Acquire-inside-the-helper
/// pattern. Constructed independently from the map-side accessor
/// because the prog-side lookups (`prog_idr`) and offsets
/// ([`crate::monitor::bpf_prog::BpfProgOffsets`]) are disjoint from
/// the map side, so a kernel that exposes maps but lacks `prog_idr`
/// (theoretical) still gets map rendering.
///
/// Returns `Err` on construction failure so the caller can warn-once
/// on permanent failure modes (matching the
/// [`try_init_owned_accessor`] contract).
///
pub(super) fn try_init_owned_prog_accessor(
    mem: Arc<monitor::reader::GuestMem>,
    vmlinux: &std::path::Path,
    tcr_el1: Option<&Arc<AtomicU64>>,
    cr3: &Arc<AtomicU64>,
) -> anyhow::Result<monitor::bpf_prog::GuestMemProgAccessorOwned> {
    let tcr_val = tcr_el1.map(|c| c.load(Ordering::Acquire)).unwrap_or(0);
    let cr3_val = cr3.load(Ordering::Acquire);
    monitor::bpf_prog::GuestMemProgAccessorOwned::new(mem, vmlinux, tcr_val, cr3_val)
}

/// Resolve and cache the per-CPU offset array. Returns `Some(offsets)`
/// only when every slot is non-zero so a partially-online VM does
/// not poison the cache with a CPU 0 alias for not-yet-online CPUs
/// (rq PA invariant; fix for `compute_rq_pas` wraparound when a
/// `pco_offset == 0` is fed downstream). Returns `None` when:
///
/// * `per_cpu_offset_kva == 0` (caller's symbol cache had no entry
///   for `__per_cpu_offset` — typically a stripped vmlinux image),
///   OR
/// * any slot is still zero (caller leaves cache `None`, retries
///   next scan tick).
///
/// Takes a pre-resolved `per_cpu_offset_kva` from the coordinator's
/// `dump_cpu_time_symbols` cache instead of re-running
/// `KernelSymbols::from_vmlinux` on every scan tick. The previous
/// in-helper parse re-read the entire vmlinux ELF (50 MB+) and re-
/// built every symbol-table entry every 100 ms while waiting for
/// the per-CPU areas to come up — visible as ~MB/s of constant
/// post-boot file I/O on every ktstr run. The KVA is fixed at
/// kernel link time and the caller already resolved it once at
/// coord start; passing it through eliminates the redundant work
/// without changing the post-resolution invariants.
///
/// `phys_base` is sourced by the caller from the owned accessor's
/// `GuestKernel::phys_base()` when it has landed; otherwise `0`
/// (correct on non-KASLR boots and the bootstrap value before the
/// accessor's page-table walk has resolved phys_base for the live
/// kernel).
pub(super) fn try_init_prog_per_cpu_offsets(
    mem: &monitor::reader::GuestMem,
    per_cpu_offset_kva: u64,
    tcr_el1: Option<&Arc<AtomicU64>>,
    phys_base: u64,
    num_cpus: u32,
) -> Option<Vec<u64>> {
    if per_cpu_offset_kva == 0 {
        return None;
    }
    let tcr_val = tcr_el1.map(|c| c.load(Ordering::Acquire)).unwrap_or(0);
    let start_kernel_map = monitor::symbols::start_kernel_map_for_tcr(tcr_val)
        .unwrap_or(monitor::symbols::START_KERNEL_MAP);
    let pco_pa =
        monitor::symbols::text_kva_to_pa_with_base(per_cpu_offset_kva, start_kernel_map, phys_base);
    let offsets = monitor::symbols::read_per_cpu_offsets(mem, pco_pa, num_cpus);
    // Defer caching until every offset slot is non-zero — a guest
    // still populating per-CPU areas yields zero entries for the
    // not-yet-initialised CPUs, and caching that would alias every
    // such CPU's stats to CPU 0. A retry is cheap; a cached miss is
    // permanent for the run. For prog_runtime_stats this means stats
    // for CPUs that haven't booted yet are simply missing —
    // acceptable, those CPUs have no stats anyway.
    if offsets.contains(&0) {
        None
    } else {
        Some(offsets)
    }
}
