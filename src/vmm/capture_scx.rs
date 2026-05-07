//! SCX walker capture builder for the failure-dump freeze path.
//!
//! Owns the per-CPU rq KVA/PA arrays the
//! [`crate::monitor::dump::ScxWalkerCapture`] borrows from. The
//! freeze coordinator calls [`build`] after the vCPU rendezvous;
//! when every prerequisite resolves, the returned
//! [`ScxWalkerOwned`] holds the data the coordinator stack-borrows
//! into the dump's [`ScxWalkerCapture`].
//!
//! The walker passes themselves (per-CPU rq->scx scalars +
//! runnable_list, per-CPU local DSQ, per-CPU bypass DSQ, per-NUMA-node
//! global DSQ, user dsq_hash, top-level scx_sched scalars) live in
//! [`crate::monitor::scx_walker`]. This builder only produces the owned
//! data the borrow-only capture needs; `dump_state` invokes the
//! walker functions on each capture and surfaces partial-degradation
//! (a single sub-group `None` blinds only that pass) via
//! `ScxWalkerOffsets::missing_groups()`.
//!
//! See [`crate::monitor::scx_walker`] for the kernel-source-grounded
//! walker semantics this capture feeds.

use crate::monitor::bpf_map::GuestMemMapAccessorOwned;
use crate::monitor::btf_offsets::ScxWalkerOffsets;
use crate::monitor::symbols::KernelSymbols;

/// Owned data the freeze coordinator stack-allocates to back a
/// [`crate::monitor::dump::ScxWalkerCapture`]. Holds every value the
/// borrow-only capture needs that does not already live on a
/// longer-lived owner (`GuestKernel`, `ScxWalkerOffsets`).
pub(crate) struct ScxWalkerOwned {
    /// Per-CPU rq kernel virtual addresses. Same shape the
    /// runnable_at scanner uses; index = CPU id. Empty when
    /// `per_cpu_offsets` was unavailable at build time (early-boot
    /// freeze before secondary CPUs populated `__per_cpu_offset[]`)
    /// — downstream rq->scx walk and per-CPU local DSQ pass yield
    /// no entries in that case, but `scx_tasks_kva` /
    /// `scx_root_kva` still drive their own walks.
    pub(crate) rq_kvas: Vec<u64>,
    /// Per-CPU rq guest physical addresses (parallel to `rq_kvas`).
    /// Empty in the same degraded-capture case described on
    /// [`Self::rq_kvas`].
    pub(crate) rq_pas: Vec<u64>,
    /// `scx_root` symbol KVA — the walker reads `*scx_root` to find
    /// the active `scx_sched`. Zero when the vmlinux had no
    /// `scx_root` symbol (kernel without sched_ext); the per-CPU
    /// rq->scx walks still run, but the DSQ-via-sched and scx_sched
    /// passes surface no state via the walker's own NULL checks.
    pub(crate) scx_root_kva: u64,
    /// `scx_tasks` symbol KVA — the kernel global LIST_HEAD that
    /// every scx-managed task is linked into via
    /// `task_struct.scx.tasks_node`. `0` when the vmlinux had no
    /// `scx_tasks` symbol (kernel without sched_ext or stripped
    /// vmlinux). The task-enrichment capture walks this list as
    /// the primary task source — it survives the per-rq
    /// runnable_list drain that scheduler teardown triggers
    /// (`scx_bypass`, kernel/sched/ext.c:5304-5404).
    pub(crate) scx_tasks_kva: u64,
}

/// Build the SCX walker owned-data set when the hard prerequisites
/// resolve.
///
/// Returns `None` when any of the following is missing:
/// - `offsets`: no BTF sub-group resolved at all (the freeze coord's
///   `dump_scx_walker_offsets`); without it `dump_state` cannot
///   construct a [`crate::monitor::dump::ScxWalkerCapture`].
/// - `symbols`: no [`KernelSymbols`] (vmlinux ELF parse failed); the
///   `runqueues` percpu section offset and `scx_root`/`scx_tasks`
///   symbol KVAs are unavailable.
///
/// `per_cpu_offsets` is NOT a hard prereq. When `None` (secondary
/// CPUs still booting — `__per_cpu_offset[]` not fully populated
/// yet, or accessor not constructed), the builder still returns a
/// degraded `ScxWalkerOwned` with empty `rq_kvas`/`rq_pas` but
/// populated `scx_root_kva` and `scx_tasks_kva`. This lets the
/// global `scx_tasks` walk still surface enrichments — the per-CPU
/// rq->scx walk and per-CPU local DSQ pass simply produce no
/// entries since the rq arrays are empty. Without this decoupling,
/// a freeze during early boot would lose ALL task-enrichment
/// signal because per_cpu_offsets gates the entire capture.
///
/// `scx_root` being absent on its own does NOT block the capture —
/// the per-CPU rq->scx walks still produce useful state on a kernel
/// without sched_ext. The DSQ-via-sched and scx_sched passes
/// degrade gracefully via the walker's own NULL checks (a `0`
/// `scx_root_kva` translates to an out-of-bounds PA that
/// `GuestMem::read_u64` returns zero for, then
/// `read_scx_sched_state` returns `None` on `sched_kva == 0`).
///
/// Per-sub-group `None` Options inside [`ScxWalkerOffsets`] do NOT
/// block the capture either: every walker pass independently gates
/// on the sub-groups it needs and skips when missing. A 6.12 kernel
/// that lacks `scx_sched_pcpu` (the per-CPU bypass DSQ struct didn't
/// land until v6.18) still produces a useful capture — only the
/// bypass-DSQ pass is blinded; the per-CPU local DSQ pass and the
/// rq->scx scalar capture run normally.
///
/// A `None` return propagates to
/// [`crate::monitor::dump::DumpContext::scx_walker_capture`]
/// being `None`, which leaves `rq_scx_states` / `dsq_states` /
/// `scx_sched_state` empty in the report and stamps
/// `scx_walker_unavailable` with
/// [`crate::monitor::dump::REASON_NO_SCX_WALKER`].
///
/// `owned_accessor` carries the [`crate::monitor::guest::GuestKernel`]
/// the walker reads through (used here only for `page_offset`);
/// `offsets` carries the BTF sub-group offsets the walker needs;
/// `symbols` carries the `runqueues` section offset and `scx_root`
/// symbol KVA; `per_cpu_offsets` is the `__per_cpu_offset[]` array
/// the walker uses to address each CPU's `scx_sched_pcpu.bypass_dsq`.
#[allow(dead_code)]
pub(crate) fn build(
    owned_accessor: &GuestMemMapAccessorOwned,
    offsets: Option<&ScxWalkerOffsets>,
    symbols: Option<&KernelSymbols>,
    per_cpu_offsets: Option<&[u64]>,
) -> Option<ScxWalkerOwned> {
    // Hard prereqs. Each `?` short-circuits to a `None` capture which
    // surfaces as `REASON_NO_SCX_WALKER` in the dump.
    let _offs = offsets?;
    let syms = symbols?;

    let page_offset = owned_accessor.guest_kernel().page_offset();
    // scx_root and scx_tasks are read from symbol metadata and do
    // NOT depend on per_cpu_offsets — the global scx_tasks walk
    // and the *scx_root sched-state read can both run when
    // per_cpu_offsets hasn't cached yet (secondary CPUs still
    // booting). Defaulting absent symbols to 0 is the same
    // graceful-degradation pattern the walker functions expect.
    let scx_root_kva = syms.scx_root.unwrap_or(0);
    let scx_tasks_kva = syms.scx_tasks.unwrap_or(0);

    let pco = match per_cpu_offsets {
        Some(pco) => pco,
        None => {
            // Degraded path: no per-CPU offsets yet. Produce an
            // owned set with empty rq arrays but populated symbol
            // KVAs so the global-task walk and sched_state read
            // still surface signal. The rq->scx walk and per-CPU
            // local DSQ pass yield no entries (their iteration
            // ranges are empty), but `walk_scx_tasks_global` and
            // `read_scx_sched_state` are independent of per-CPU
            // addressing.
            tracing::debug!(
                "capture_scx::build: per_cpu_offsets absent — degraded \
                 capture with no rq arrays; global scx_tasks walk and \
                 *scx_root read still active",
            );
            return Some(ScxWalkerOwned {
                rq_kvas: Vec::new(),
                rq_pas: Vec::new(),
                scx_root_kva,
                scx_tasks_kva,
            });
        }
    };

    Some(compute_owned(
        page_offset,
        syms.runqueues,
        scx_root_kva,
        scx_tasks_kva,
        pco,
    ))
}

/// Pure builder for the owned-data set.
///
/// Splits the `&GuestMemMapAccessorOwned`-touching outer surface from
/// the deterministic per-CPU address math so the helper is testable
/// without a real owned accessor (the type has no `new_for_test`
/// constructor).
///
/// `runqueues_off` is the `.data..percpu` section-relative offset of
/// the per-CPU `runqueues` symbol (NOT a KVA — see
/// [`crate::monitor::symbols::KernelSymbols::runqueues`]); the per-CPU
/// rq KVA for CPU `n` is `runqueues_off + per_cpu_offsets[n]` and the
/// PA is the same KVA minus `page_offset` (direct mapping).
fn compute_owned(
    page_offset: u64,
    runqueues_off: u64,
    scx_root_kva: u64,
    scx_tasks_kva: u64,
    per_cpu_offsets: &[u64],
) -> ScxWalkerOwned {
    // Build rq_pas and rq_kvas in a single pass over per_cpu_offsets
    // rather than calling compute_rq_pas (one walk) then mapping
    // again to recover the kvas (second walk). Saves one Vec
    // allocation and one full iteration on every freeze — matters
    // when nr_cpus is large because this runs while every vCPU is
    // paused.
    let n = per_cpu_offsets.len();
    let mut rq_pas: Vec<u64> = Vec::with_capacity(n);
    let mut rq_kvas: Vec<u64> = Vec::with_capacity(n);
    for &offset in per_cpu_offsets {
        let kva = runqueues_off.wrapping_add(offset);
        let pa = crate::monitor::symbols::kva_to_pa(kva, page_offset);
        rq_pas.push(pa);
        rq_kvas.push(pa.wrapping_add(page_offset));
    }
    ScxWalkerOwned {
        rq_kvas,
        rq_pas,
        scx_root_kva,
        scx_tasks_kva,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    /// Happy path: every prereq resolves. The pure builder produces
    /// per-CPU rq KVA/PA pairs that match
    /// `runqueues_off + per_cpu_offset[cpu]` and pass `scx_root_kva`
    /// through unchanged. Mirrors the runnable scanner's address
    /// derivation in `freeze_coord.rs` so both code paths agree on
    /// the per-CPU rq base.
    #[test]
    fn compute_owned_happy_path() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues_off: u64 = 0x20_0000;
        let per_cpu = [0x10_0000u64, 0x14_0000u64, 0x18_0000u64];
        let scx_root_kva = 0xffff_ffff_8230_0000;
        let scx_tasks_kva = 0xffff_ffff_8240_0000;
        let owned = compute_owned(
            page_offset,
            runqueues_off,
            scx_root_kva,
            scx_tasks_kva,
            &per_cpu,
        );

        assert_eq!(owned.scx_root_kva, scx_root_kva);
        assert_eq!(owned.scx_tasks_kva, scx_tasks_kva);
        assert_eq!(owned.rq_kvas.len(), 3);
        assert_eq!(owned.rq_pas.len(), 3);
        // Cross-check against the same `compute_rq_pas` the runnable
        // scanner uses — any drift between the two would surface
        // here as different per-CPU PAs.
        let expected_pas =
            crate::monitor::symbols::compute_rq_pas(runqueues_off, &per_cpu, page_offset, 0, 0);
        assert_eq!(owned.rq_pas, expected_pas);
        // Every rq_kva is the recovered KVA for the same CPU's PA.
        for (cpu, expected_pa) in expected_pas.iter().enumerate() {
            assert_eq!(owned.rq_kvas[cpu], expected_pa.wrapping_add(page_offset),);
        }
    }

    /// scx_root absent (kernel without sched_ext): builder still
    /// produces the per-CPU rq arrays. The walker's own NULL checks
    /// on the zero `scx_root_kva` degrade the DSQ / scx_sched passes
    /// to "no state" without invalidating the rq->scx pass.
    #[test]
    fn compute_owned_partial_scx_root_zero() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues_off: u64 = 0x20_0000;
        let per_cpu = [0x10_0000u64, 0x14_0000u64];
        let owned = compute_owned(page_offset, runqueues_off, 0, 0, &per_cpu);

        assert_eq!(owned.scx_root_kva, 0);
        assert_eq!(owned.scx_tasks_kva, 0);
        assert_eq!(owned.rq_kvas.len(), 2);
        assert_eq!(owned.rq_pas.len(), 2);
        let expected_pas =
            crate::monitor::symbols::compute_rq_pas(runqueues_off, &per_cpu, page_offset, 0, 0);
        assert_eq!(owned.rq_pas, expected_pas);
    }

    /// scx_tasks absent (stripped vmlinux) but scx_root present:
    /// the global-list walk degrades to "no tasks" without affecting
    /// the rq->scx walk or the DSQ/scx_sched passes that key off
    /// scx_root.
    #[test]
    fn compute_owned_partial_scx_tasks_zero() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues_off: u64 = 0x20_0000;
        let per_cpu = [0x10_0000u64];
        let scx_root_kva = 0xffff_ffff_8230_0000;
        let owned = compute_owned(page_offset, runqueues_off, scx_root_kva, 0, &per_cpu);

        assert_eq!(owned.scx_root_kva, scx_root_kva);
        assert_eq!(owned.scx_tasks_kva, 0);
        assert_eq!(owned.rq_kvas.len(), 1);
        assert_eq!(owned.rq_pas.len(), 1);
    }

    /// Degraded build (per_cpu_offsets absent, secondary CPUs still
    /// booting): the captured `ScxWalkerOwned` must have empty rq
    /// arrays but populated symbol KVAs. This is the shape the
    /// build() degraded path returns when per_cpu_offsets is None
    /// — the global-task walk and *scx_root sched-state read still
    /// surface signal even when per-CPU addressing isn't ready.
    /// Validates the structural contract: the downstream
    /// scx_walker functions iterate over `rq_kvas`/`rq_pas`
    /// (yielding nothing when empty) but consume `scx_tasks_kva`
    /// and `scx_root_kva` independently.
    #[test]
    fn degraded_build_shape_empty_rq_with_symbol_kvas() {
        let scx_root_kva = 0xffff_ffff_8230_0000;
        let scx_tasks_kva = 0xffff_ffff_8240_0000;
        let owned = ScxWalkerOwned {
            rq_kvas: Vec::new(),
            rq_pas: Vec::new(),
            scx_root_kva,
            scx_tasks_kva,
        };
        assert!(owned.rq_kvas.is_empty());
        assert!(owned.rq_pas.is_empty());
        assert_eq!(owned.scx_root_kva, scx_root_kva);
        assert_eq!(owned.scx_tasks_kva, scx_tasks_kva);
        // Iteration over empty rq arrays yields no entries — the
        // rq->scx and per-CPU local DSQ passes both no-op cleanly.
        let zipped: Vec<_> = owned.rq_kvas.iter().zip(owned.rq_pas.iter()).collect();
        assert!(zipped.is_empty());
    }

    /// Degraded build coexists with the global-task walk: even with
    /// empty rq arrays, a populated `scx_tasks_kva` is the input the
    /// `walk_scx_tasks_global` walker uses. Pinning the field
    /// independence at the struct-shape level so a future refactor
    /// that ties `scx_tasks_kva` to `rq_kvas` length surfaces here.
    #[test]
    fn degraded_build_scx_tasks_kva_independent_of_rq_arrays() {
        let scx_tasks_kva = 0xffff_ffff_82e5_e840;
        let owned = ScxWalkerOwned {
            rq_kvas: Vec::new(),
            rq_pas: Vec::new(),
            scx_root_kva: 0,
            scx_tasks_kva,
        };
        // scx_tasks_kva remains addressable even though rq arrays
        // are empty — the global walk doesn't depend on per-CPU
        // addressing.
        assert_eq!(owned.scx_tasks_kva, scx_tasks_kva);
        // scx_root_kva == 0 here demonstrates the second
        // independent-degradation axis: a kernel without sched_ext
        // (no scx_root symbol) plus per_cpu_offsets unavailable
        // still produces a valid owned struct.
        assert_eq!(owned.scx_root_kva, 0);
    }

    /// Empty per_cpu_offsets: builder produces an empty owned set.
    /// Mirrors a freeze before any vCPU has come up — the freeze
    /// coordinator's `per_cpu_offsets.contains(&0)` retry guard
    /// normally rejects this state, but the math itself must stay
    /// well-defined so the pre-retry path doesn't UB.
    #[test]
    fn compute_owned_empty_per_cpu_offsets() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues_off: u64 = 0x20_0000;
        let scx_root_kva = 0xffff_ffff_8230_0000;
        let scx_tasks_kva = 0xffff_ffff_8240_0000;
        let owned = compute_owned(page_offset, runqueues_off, scx_root_kva, scx_tasks_kva, &[]);

        assert!(owned.rq_kvas.is_empty());
        assert!(owned.rq_pas.is_empty());
        assert_eq!(owned.scx_root_kva, scx_root_kva);
        assert_eq!(owned.scx_tasks_kva, scx_tasks_kva);
    }

    /// rq_pa wraps when the runqueues section offset + per_cpu offset
    /// straddles `page_offset`. Every step of the math uses
    /// `wrapping_*` so the result is well-defined; this test pins the
    /// behavior so a future refactor can't silently introduce a
    /// `checked_sub` that panics on the boundary case.
    #[test]
    fn compute_owned_wrapping_arithmetic() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        // Pick a per_cpu offset that, combined with the section-
        // relative runqueues_off, lands at exactly page_offset — the
        // resulting rq_pa is 0 and the recovered rq_kva is page_offset.
        let runqueues_off: u64 = 0x1000;
        let per_cpu = [page_offset.wrapping_sub(runqueues_off)];
        let owned = compute_owned(page_offset, runqueues_off, 0, 0, &per_cpu);

        assert_eq!(owned.rq_pas, vec![0u64]);
        assert_eq!(owned.rq_kvas, vec![page_offset]);
    }

    /// rq_kvas and rq_pas remain index-aligned: for every CPU `i`,
    /// `rq_kvas[i] == rq_pas[i] + page_offset`. This invariant is what
    /// lets the runnable_list walker use `rq_kva` as the loop
    /// terminator and `rq_pa` as the read-base on the same CPU.
    #[test]
    fn compute_owned_kva_pa_pairwise_consistent() {
        let page_offset = DEFAULT_PAGE_OFFSET;
        let runqueues_off: u64 = 0x4_0000;
        let per_cpu = [
            0x10_0000u64,
            0x14_0000u64,
            0x18_0000u64,
            0x1c_0000u64,
            0x20_0000u64,
        ];
        let owned = compute_owned(
            page_offset,
            runqueues_off,
            0xffff_ffff_8000_0000,
            0xffff_ffff_8001_0000,
            &per_cpu,
        );

        assert_eq!(owned.rq_kvas.len(), per_cpu.len());
        assert_eq!(owned.rq_pas.len(), per_cpu.len());
        for cpu in 0..per_cpu.len() {
            assert_eq!(
                owned.rq_kvas[cpu],
                owned.rq_pas[cpu].wrapping_add(page_offset),
                "kva/pa pair mismatch on cpu {cpu}",
            );
        }
    }
}
