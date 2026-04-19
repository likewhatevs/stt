//! Shared IDR/xarray walk for host-side kernel memory reads.
//!
//! The kernel's IDR uses an xarray internally. These functions walk
//! the xarray tree structure in guest physical memory to enumerate
//! entries by index. Used by both BPF map and BPF program discovery.

use super::reader::GuestMem;
use super::symbols::kva_to_pa;

/// XA_CHUNK_SHIFT = 6, XA_CHUNK_SIZE = 64.
pub(crate) const XA_CHUNK_SIZE: u64 = 64;

/// Translate a kernel virtual address to a GuestMem offset, trying
/// direct mapping first, then page table walk.
///
/// SLAB allocations live in the direct mapping (PAGE_OFFSET..PAGE_END).
/// vmalloc'd and module addresses require a page table walk.
pub(crate) fn translate_any_kva(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    kva: u64,
    l5: bool,
) -> Option<u64> {
    let direct_pa = kva_to_pa(kva, page_offset);
    if direct_pa < mem.size() {
        return Some(direct_pa);
    }
    mem.translate_kva(cr3_pa, kva, l5)
}

/// Load an entry from an xarray by index.
///
/// xa_node structs are SLAB-allocated and live in the direct mapping,
/// so their KVAs are translated via `kva_to_pa(kva, page_offset)`.
/// `slots_off` and `shift_off` are BTF-resolved byte offsets of
/// `slots` and `shift` within `struct xa_node`.
///
/// Returns `Some(0)` for empty slots or `Some(ptr)` for populated
/// entries. Out-of-bounds reads return 0 (empty slot).
pub(crate) fn xa_load(
    mem: &GuestMem,
    page_offset: u64,
    xa_head: u64,
    index: u64,
    slots_off: usize,
    shift_off: usize,
) -> Option<u64> {
    if xa_head == 0 {
        return Some(0);
    }

    // Check if xa_head is an internal node (bit 1 set) or a direct entry.
    if xa_head & 2 == 0 {
        // Single-entry xarray: only index 0 is valid.
        return if index == 0 { Some(xa_head) } else { Some(0) };
    }

    // xa_head is a node pointer. Clear the internal marker bits.
    let mut node_kva = xa_head & !3u64;
    let mut shift = xa_node_shift(mem, page_offset, node_kva, shift_off);

    loop {
        let slot_idx = (index >> shift) & (XA_CHUNK_SIZE - 1);
        let slot_pa = kva_to_pa(node_kva + slots_off as u64 + slot_idx * 8, page_offset);
        let entry = mem.read_u64(slot_pa, 0);

        if entry == 0 {
            return Some(0);
        }

        if entry & 2 == 0 {
            // Leaf entry.
            return Some(entry);
        }

        // Internal node — descend.
        node_kva = entry & !3u64;
        if shift < 6 {
            return Some(0);
        }
        shift -= 6; // XA_CHUNK_SHIFT
    }
}

/// Read the `shift` field from an xa_node (SLAB-allocated, direct mapping).
pub(crate) fn xa_node_shift(
    mem: &GuestMem,
    page_offset: u64,
    node_kva: u64,
    shift_off: usize,
) -> u64 {
    let pa = kva_to_pa(node_kva, page_offset);
    mem.read_u8(pa, shift_off) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::symbols::DEFAULT_PAGE_OFFSET;

    #[test]
    fn xa_node_shift_reads_shift_byte() {
        // Place an xa_node at PA 0x100 with shift=12 at byte offset 2.
        let mut buf = [0u8; 0x200];
        let shift_off = 2;
        buf[0x100 + shift_off] = 12;

        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let node_kva = DEFAULT_PAGE_OFFSET + 0x100;
        assert_eq!(
            xa_node_shift(&mem, DEFAULT_PAGE_OFFSET, node_kva, shift_off),
            12
        );
    }

    #[test]
    fn xa_node_shift_zero() {
        let mut buf = [0u8; 0x200];
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let node_kva = DEFAULT_PAGE_OFFSET;
        // shift_off=0, buf[0]=0 -> shift=0.
        assert_eq!(xa_node_shift(&mem, DEFAULT_PAGE_OFFSET, node_kva, 0), 0);
    }

    #[test]
    fn xa_node_shift_max_u8() {
        let mut buf = [0u8; 0x100];
        buf[0x10] = 255;
        // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
        // whose backing storage outlives the GuestMem use.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let node_kva = DEFAULT_PAGE_OFFSET;
        assert_eq!(
            xa_node_shift(&mem, DEFAULT_PAGE_OFFSET, node_kva, 0x10),
            255
        );
    }
}
