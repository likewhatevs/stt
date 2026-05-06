//! Per-node NUMA event capture builder for the failure-dump freeze
//! path.
//!
//! Reads `node_data[]` and walks each `pglist_data.node_zones[]`,
//! summing `zone.vm_numa_event[]` across zones to produce one
//! [`crate::monitor::dump::PerNodeNumaStats`] row per NUMA node.
//!
//! Unlike [`crate::vmm::capture_scx`] / [`crate::vmm::capture_tasks`],
//! NUMA produces an owned `Vec<PerNodeNumaStats>` directly — no
//! borrow-only capture object. The freeze coordinator overwrites
//! [`crate::monitor::dump::FailureDumpReport::per_node_numa`] and
//! clears [`crate::monitor::dump::FailureDumpReport::per_node_numa_unavailable`]
//! when a non-empty Vec lands.

use crate::monitor::bpf_map::GuestMemMapAccessorOwned;
use crate::monitor::btf_offsets::{NR_VM_NUMA_EVENT_ITEMS, NumaStatsOffsets};
use crate::monitor::dump::PerNodeNumaStats;
use crate::monitor::guest::GuestKernel;
use crate::monitor::idr::translate_any_kva;
use crate::monitor::symbols::KernelSymbols;

/// `MAX_NR_ZONES` upper bound from `include/linux/mmzone.h`. With
/// `CONFIG_ZONE_DEVICE` the kernel exposes 5 zones (ZONE_DMA,
/// ZONE_DMA32, ZONE_NORMAL, ZONE_MOVABLE, ZONE_DEVICE); without it
/// the trailing slot is absent. Iterating up to 5 is safe — the
/// surplus slots in `pglist_data.node_zones[]` are zero-initialized
/// on a kernel that lacks them, so the per-zone sum just adds zero.
const MAX_NR_ZONES: usize = 5;

/// Build the per-node NUMA event Vec when every prerequisite
/// resolves.
///
/// Returns `None` when the `node_data` symbol is absent (UMA build,
/// stripped vmlinux, or a kernel without `CONFIG_NUMA`), the BTF
/// offsets failed to resolve (no `pglist_data` / `zone` types), or
/// no node's `pglist_data` pointer was readable. A `None` return
/// leaves [`crate::monitor::dump::FailureDumpReport::per_node_numa`]
/// empty and [`crate::monitor::dump::FailureDumpReport::per_node_numa_unavailable`]
/// set to [`crate::monitor::dump::REASON_NO_NUMA_WALKER`].
///
/// `owned_accessor` carries the [`GuestKernel`] the walker reads
/// through; `offsets` carries the BTF pglist_data / zone offsets;
/// `symbols` carries the `node_data` symbol KVA the walker
/// dereferences; `nr_nodes` is the configured NUMA node count from
/// the topology. A node whose `node_data[i]` slot is 0 (offline at
/// freeze time) is skipped — its pglist_data pointer was never
/// installed.
pub(crate) fn build(
    owned_accessor: &GuestMemMapAccessorOwned,
    offsets: Option<&NumaStatsOffsets>,
    symbols: Option<&KernelSymbols>,
    nr_nodes: u32,
) -> Option<Vec<PerNodeNumaStats>> {
    let offsets = offsets?;
    let symbols = symbols?;
    let node_data_kva = symbols.node_data?;
    if nr_nodes == 0 {
        return None;
    }

    let kernel = owned_accessor.guest_kernel();
    let stats = walk_node_data(kernel, node_data_kva, offsets, nr_nodes);
    if stats.is_empty() { None } else { Some(stats) }
}

/// Walk every entry of `node_data[]` and produce one
/// [`PerNodeNumaStats`] row per node whose pglist_data pointer is
/// non-null and reachable.
///
/// The `node_data` array lives in the kernel image (`.data`/`.bss`),
/// so its slots are read at `kernel.text_kva_to_pa(node_data_kva) + i*8`.
/// Each slot value is a direct-mapping kernel virtual address
/// (`__va(nd_pa)`); per-zone reads use [`translate_any_kva`] which
/// tries the direct-mapping translation first, then the page-table
/// walk.
fn walk_node_data(
    kernel: &GuestKernel<'_>,
    node_data_kva: u64,
    offsets: &NumaStatsOffsets,
    nr_nodes: u32,
) -> Vec<PerNodeNumaStats> {
    let mut out = Vec::with_capacity(nr_nodes as usize);
    let node_data_pa = kernel.text_kva_to_pa(node_data_kva);
    let mem = kernel.mem();
    for node in 0..nr_nodes {
        let pgdat_kva = mem.read_u64(node_data_pa, (node as usize) * 8);
        if pgdat_kva == 0 {
            continue;
        }
        let Some(per_node) = read_per_node_stats(kernel, pgdat_kva, offsets, node) else {
            continue;
        };
        out.push(per_node);
    }
    out
}

/// Read one node's per-zone counters and sum them into a
/// [`PerNodeNumaStats`] row.
///
/// Returns `None` when `pgdat_kva` does not resolve to guest memory
/// — that node's pglist_data was either freed or never installed.
fn read_per_node_stats(
    kernel: &GuestKernel<'_>,
    pgdat_kva: u64,
    offsets: &NumaStatsOffsets,
    node: u32,
) -> Option<PerNodeNumaStats> {
    let mem = kernel.mem();
    let walk = kernel.walk_context();
    let pgdat_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        pgdat_kva,
        walk.l5,
        walk.tcr_el1,
    )?;

    let mut sums = [0u64; NR_VM_NUMA_EVENT_ITEMS];
    let zones_base = pgdat_pa.checked_add(offsets.pglist_data_node_zones as u64)?;
    // Hoist the per-slot offset arithmetic out of the per-zone loop:
    // each slot_off depends only on `slot`, not on `zone_idx`, so the
    // checked_mul/checked_add chain ran MAX_NR_ZONES * NR_VM_NUMA_EVENT_ITEMS
    // times pre-fix despite producing only NR_VM_NUMA_EVENT_ITEMS unique
    // values. Compute once into a fixed-size array, then reuse across
    // zones. Saves ~5 * 6 = 30 redundant checked-arith chains per node.
    let mut slot_offs = [0usize; NR_VM_NUMA_EVENT_ITEMS];
    for (slot, off) in slot_offs.iter_mut().enumerate() {
        *off = offsets
            .zone_vm_numa_event
            .checked_add(slot.checked_mul(8)?)?;
    }
    for zone_idx in 0..MAX_NR_ZONES {
        let zone_off = (zone_idx as u64).checked_mul(offsets.zone_size as u64)?;
        let zone_pa = zones_base.checked_add(zone_off)?;
        for (slot_off, sum) in slot_offs.iter().zip(sums.iter_mut()) {
            let v = mem.read_u64(zone_pa, *slot_off);
            *sum = sum.wrapping_add(v);
        }
    }

    Some(PerNodeNumaStats {
        node,
        numa_hit: sums[crate::monitor::btf_offsets::NUMA_HIT],
        numa_miss: sums[crate::monitor::btf_offsets::NUMA_MISS],
        numa_foreign: sums[crate::monitor::btf_offsets::NUMA_FOREIGN],
        numa_interleave_hit: sums[crate::monitor::btf_offsets::NUMA_INTERLEAVE_HIT],
        numa_local: sums[crate::monitor::btf_offsets::NUMA_LOCAL],
        numa_other: sums[crate::monitor::btf_offsets::NUMA_OTHER],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::reader::GuestMem;
    use crate::monitor::symbols::{
        DEFAULT_PAGE_OFFSET, START_KERNEL_MAP, text_kva_to_pa_with_base,
    };
    use std::collections::HashMap;

    /// Layout helper for synthetic walks. Builds a buffer with
    /// `node_data[]` at PA `node_data_pa` (text mapping; KVA is
    /// `START_KERNEL_MAP + 0x1000`). Per node, a pglist_data sits at
    /// `pgdat_base_pa + i * pgdat_size` in the direct mapping
    /// (KVA `DEFAULT_PAGE_OFFSET + pa`). Each pgdat carries
    /// `MAX_NR_ZONES` zones laid out contiguously at
    /// `pgdat_pa + pglist_data_node_zones`. Every per-zone slot
    /// `vm_numa_event[k]` is filled with `n*1000 + z*100 + k`.
    struct NumaLayout {
        buf: Vec<u8>,
        offsets: NumaStatsOffsets,
        node_data_kva: u64,
        nr_nodes: u32,
    }

    impl NumaLayout {
        fn build(nr_nodes: u32) -> Self {
            // Layout choices kept small but realistic:
            //   pglist_data_node_zones = 0x40
            //   zone_size              = 0x80
            //   zone_vm_numa_event     = 0x10
            // Total pgdat size: 0x40 + MAX_NR_ZONES * 0x80 = 0x340.
            let pglist_data_node_zones = 0x40usize;
            let zone_size = 0x80usize;
            let zone_vm_numa_event = 0x10usize;
            let pgdat_size = pglist_data_node_zones + MAX_NR_ZONES * zone_size;

            let node_data_pa: u64 = 0x1000;
            let pgdat_base_pa: u64 = 0x10000;

            let total = (pgdat_base_pa as usize) + (nr_nodes as usize) * pgdat_size + 0x100;
            let mut buf = vec![0u8; total];

            for n in 0..nr_nodes {
                let pgdat_pa = pgdat_base_pa + (n as u64) * (pgdat_size as u64);
                let pgdat_kva = DEFAULT_PAGE_OFFSET.wrapping_add(pgdat_pa);
                let slot = (node_data_pa as usize) + (n as usize) * 8;
                buf[slot..slot + 8].copy_from_slice(&pgdat_kva.to_le_bytes());

                for z in 0..MAX_NR_ZONES {
                    let zone_pa = pgdat_pa
                        + (pglist_data_node_zones as u64)
                        + (z as u64) * (zone_size as u64);
                    for k in 0..NR_VM_NUMA_EVENT_ITEMS {
                        let slot_pa = zone_pa as usize + zone_vm_numa_event + k * 8;
                        let v: u64 = (n as u64) * 1000 + (z as u64) * 100 + (k as u64);
                        buf[slot_pa..slot_pa + 8].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }

            let node_data_kva = START_KERNEL_MAP.wrapping_add(node_data_pa);
            NumaLayout {
                buf,
                offsets: NumaStatsOffsets {
                    pglist_data_node_zones,
                    zone_vm_numa_event,
                    zone_size,
                },
                node_data_kva,
                nr_nodes,
            }
        }
    }

    /// Build a [`GuestKernel`] over the given buffer. `cr3_pa = 0`
    /// (page-table walks fail; the direct-mapping branch of
    /// [`translate_any_kva`] handles every lookup).
    fn make_kernel(buf: &mut [u8]) -> GuestMem {
        // SAFETY: buf outlives the GuestMem and is the only owner.
        unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) }
    }

    /// Happy path: 2 nodes, every per-node counter populated.
    /// Verifies the per-zone sum lands in the right slot of
    /// [`PerNodeNumaStats`] and the node id is preserved.
    #[test]
    fn two_nodes_summed_across_zones() {
        let mut layout = NumaLayout::build(2);
        let nr_nodes = layout.nr_nodes;
        let node_data_kva = layout.node_data_kva;
        let offsets = layout.offsets;
        let mem = make_kernel(&mut layout.buf);
        let kernel = GuestKernel::new_for_test(&mem, HashMap::new(), DEFAULT_PAGE_OFFSET, 0, false);

        let stats = walk_node_data(&kernel, node_data_kva, &offsets, nr_nodes);
        assert_eq!(stats.len(), 2);

        // Expected sum across zones for slot k on node n:
        //   sum_{z=0..MAX_NR_ZONES} (n*1000 + z*100 + k)
        //   = MAX_NR_ZONES*(n*1000) + 100*sum_{z}z + MAX_NR_ZONES*k
        //   = 5*n*1000 + 100*(0+1+2+3+4) + 5*k
        //   = 5*n*1000 + 1000 + 5*k
        let expected = |n: u64, k: u64| -> u64 { 5 * n * 1000 + 1000 + 5 * k };

        for (idx, st) in stats.iter().enumerate() {
            let n = idx as u64;
            assert_eq!(st.node, idx as u32);
            assert_eq!(st.numa_hit, expected(n, 0));
            assert_eq!(st.numa_miss, expected(n, 1));
            assert_eq!(st.numa_foreign, expected(n, 2));
            assert_eq!(st.numa_interleave_hit, expected(n, 3));
            assert_eq!(st.numa_local, expected(n, 4));
            assert_eq!(st.numa_other, expected(n, 5));
        }
    }

    /// Sparse layout: `node_data[1] = 0`. The walker must skip slot 1
    /// (no pglist_data installed) and still emit slot 0.
    #[test]
    fn offline_node_skipped() {
        let mut layout = NumaLayout::build(2);
        let node_data_pa =
            text_kva_to_pa_with_base(layout.node_data_kva, START_KERNEL_MAP, 0) as usize;
        for b in &mut layout.buf[node_data_pa + 8..node_data_pa + 16] {
            *b = 0;
        }
        let nr_nodes = layout.nr_nodes;
        let node_data_kva = layout.node_data_kva;
        let offsets = layout.offsets;
        let mem = make_kernel(&mut layout.buf);
        let kernel = GuestKernel::new_for_test(&mem, HashMap::new(), DEFAULT_PAGE_OFFSET, 0, false);

        let stats = walk_node_data(&kernel, node_data_kva, &offsets, nr_nodes);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].node, 0);
    }

    /// Every `node_data[i]` slot reads as 0 (UMA build with the
    /// symbol present but no pglist_data pointers installed in the
    /// observed range, or a freed kernel image): the walker yields
    /// an empty Vec.
    #[test]
    fn all_slots_zero_yields_empty() {
        // Allocate a buffer large enough for node_data[] but leave
        // every slot zero — no pgdats installed.
        let mut buf = vec![0u8; 0x1_0000];
        let mem = make_kernel(&mut buf);
        let kernel = GuestKernel::new_for_test(&mem, HashMap::new(), DEFAULT_PAGE_OFFSET, 0, false);
        let offsets = NumaStatsOffsets {
            pglist_data_node_zones: 0x40,
            zone_vm_numa_event: 0x10,
            zone_size: 0x80,
        };
        let kva = START_KERNEL_MAP.wrapping_add(0x1000);
        let stats = walk_node_data(&kernel, kva, &offsets, 4);
        assert!(stats.is_empty());
    }

    /// `nr_nodes = 0` produces an empty Vec — the per-node loop
    /// never iterates. Tests the topology-zero-nodes guard
    /// independent of the symbol/offsets-None gates.
    #[test]
    fn zero_nodes_yields_empty_walk() {
        let mut buf = vec![0u8; 0x1_0000];
        let mem = make_kernel(&mut buf);
        let kernel = GuestKernel::new_for_test(&mem, HashMap::new(), DEFAULT_PAGE_OFFSET, 0, false);
        let offsets = NumaStatsOffsets {
            pglist_data_node_zones: 0x40,
            zone_vm_numa_event: 0x10,
            zone_size: 0x80,
        };
        let kva = START_KERNEL_MAP.wrapping_add(0x1000);
        let stats = walk_node_data(&kernel, kva, &offsets, 0);
        assert!(stats.is_empty());
    }

    /// Mixed populated + zero-tail node_data[]: realistic scenario
    /// where the topology reports nr_nodes greater than the count
    /// of actually-online nodes — slots beyond the populated range
    /// read as 0 and the walker skips them. Distinct from
    /// `all_slots_zero_yields_empty` (every slot zero); this pins
    /// the partial-populated path that real CI sees on hosts where
    /// the topology probe reports the kernel-config max but only a
    /// subset of nodes actually have memory installed.
    #[test]
    fn mixed_populated_and_zero_tail_yields_populated_only() {
        // Build a 2-node fixture, then walk with nr_nodes=4: slots
        // 2 and 3 read as 0 from the post-fixture buffer tail.
        let mut layout = NumaLayout::build(2);
        let nr_nodes_to_walk: u32 = 4;
        let node_data_kva = layout.node_data_kva;
        let offsets = layout.offsets;
        // Verify slot 2 / 3 pre-walk: the layout reserves only
        // 2 * 8 bytes for node_data; the next 16 bytes are 0
        // because the fixture allocates with zero-fill, but we want
        // to be explicit since the walker reads them via
        // `read_u64` which is bounds-checked by GuestMem.
        let mem = make_kernel(&mut layout.buf);
        let kernel = GuestKernel::new_for_test(&mem, HashMap::new(), DEFAULT_PAGE_OFFSET, 0, false);

        let stats = walk_node_data(&kernel, node_data_kva, &offsets, nr_nodes_to_walk);
        assert_eq!(
            stats.len(),
            2,
            "only the 2 populated slots produce stats; trailing zero slots are skipped"
        );
        // Populated slots carry their node id (0, 1).
        assert_eq!(stats[0].node, 0);
        assert_eq!(stats[1].node, 1);
    }
}
