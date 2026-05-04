use super::*;
use crate::monitor::idr::{XA_CHUNK_SIZE, xa_node_shift};
use crate::monitor::symbols::START_KERNEL_MAP;

/// Test-only alias: many value-I/O tests don't thread an
/// `&BpfMapOffsets` through, because `read_value` / `write_value`
/// never touch one. Build the full [`AccessorCtx`] by borrowing
/// [`BpfMapOffsets::EMPTY`] so those call sites stay terse.
#[cfg(target_arch = "x86_64")]
fn value_ctx<'a>(mem: &'a GuestMem, cr3_pa: u64, l5: bool) -> AccessorCtx<'a> {
    AccessorCtx {
        mem,
        cr3_pa: Cr3Pa(cr3_pa),
        page_offset: PageOffset(0),
        offsets: &BpfMapOffsets::EMPTY,
        l5,
    }
}

pub(super) fn lookup_ctx<'a>(
    mem: &'a GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    offsets: &'a BpfMapOffsets,
    l5: bool,
) -> AccessorCtx<'a> {
    AccessorCtx {
        mem,
        cr3_pa: Cr3Pa(cr3_pa),
        page_offset: PageOffset(page_offset),
        offsets,
        l5,
    }
}

// On aarch64, page table entries contain GPAs starting at DRAM_START.
// The walker subtracts DRAM_START to produce GuestMem offsets. Test
// page table entries must include this base so the subtraction yields
// the correct buffer offset.
#[cfg(target_arch = "x86_64")]
pub(super) const PTE_BASE: u64 = 0;
#[cfg(target_arch = "aarch64")]
pub(super) const PTE_BASE: u64 = crate::vmm::kvm::DRAM_START;

// Huge page (block) descriptor flags differ by architecture.
// x86: PS(0x80) | present | rw | accessed | dirty = 0xE3.
// aarch64: block descriptor bits [1:0] = 0b01 = 0x01.
#[cfg(target_arch = "x86_64")]
const BLOCK_FLAGS: u64 = 0xE3;
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)] // used when aarch64 huge page tests are added
const BLOCK_FLAGS: u64 = 0x01;

// -- translate_kva tests --

/// Build a minimal 4-level page table in a buffer, mapping a single
/// 4KB page. Returns (buffer, cr3_pa, mapped_kva, mapped_pa).
#[cfg(target_arch = "x86_64")]
fn setup_page_table() -> (Vec<u8>, u64, u64, u64) {
    // Use a KVA and compute indices dynamically.
    let kva: u64 = 0xFFFF_8880_0000_5000;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;
    let pte_idx = (kva >> 12) & 0x1FF;

    // Page table pages at fixed PAs. PGD needs to be large enough
    // for the highest index entry.
    let pgd_pa: u64 = 0x10000; // 64KB — enough for any index * 8
    let pud_pa: u64 = pgd_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let pte_pa: u64 = pmd_pa + 0x1000;
    let data_pa: u64 = pte_pa + 0x1000;

    let size = (data_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x63);

    // Write known data at the target page.
    buf[data_pa as usize..data_pa as usize + 8]
        .copy_from_slice(&0xDEAD_BEEF_CAFE_1234u64.to_ne_bytes());

    (buf, pgd_pa, kva, data_pa)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_basic() {
    let (buf, cr3_pa, kva, data_pa) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva), false);
    assert_eq!(pa, Some(data_pa));
    // Read through the translated PA to verify correctness.
    assert_eq!(mem.read_u64(pa.unwrap(), 0), 0xDEAD_BEEF_CAFE_1234);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_with_offset() {
    let (buf, cr3_pa, kva, data_pa) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    // KVA + 0x100 should map to data_pa + 0x100
    let pa = mem.translate_kva(cr3_pa, Kva(kva + 0x100), false);
    assert_eq!(pa, Some(data_pa + 0x100));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_unmapped() {
    let (buf, cr3_pa, _, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    // A completely different address that has no PGD entry.
    let pa = mem.translate_kva(cr3_pa, Kva(0xFFFF_FFFF_8000_0000), false);
    assert_eq!(pa, None);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_unmapped_pte() {
    let (buf, cr3_pa, kva, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    // Same PGD/PUD/PMD but next PTE index — not mapped.
    let unmapped_kva = kva + 0x1000;
    let pa = mem.translate_kva(cr3_pa, Kva(unmapped_kva), false);
    assert_eq!(pa, None);
}

// -- translate_kva: 2MB huge page --

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_2mb_huge_page() {
    // Map KVA via a 2MB page (PS bit set in PMD entry).
    let kva: u64 = 0xFFFF_8880_0020_0000; // 2MB-aligned
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let huge_page_pa: u64 = 0x20_0000; // 2MB-aligned physical page

    let size = (huge_page_pa + 0x20_0000) as usize; // room for the 2MB page
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // PGD -> PUD
    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    // PUD -> PMD
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    // PMD entry with PS bit set (bit 7) = 2MB huge page
    write_entry(
        &mut buf,
        pmd_pa,
        pmd_idx,
        (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
    ); // present+rw+PS

    // Write marker data at the huge page base.
    buf[huge_page_pa as usize..huge_page_pa as usize + 8]
        .copy_from_slice(&0xCAFE_BABE_1234_5678u64.to_ne_bytes());

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(pgd_pa, Kva(kva), false);
    assert_eq!(pa, Some(huge_page_pa));
    assert_eq!(mem.read_u64(pa.unwrap(), 0), 0xCAFE_BABE_1234_5678);

    // Offset within the 2MB page.
    let pa_off = mem.translate_kva(pgd_pa, Kva(kva + 0x1000), false);
    assert_eq!(pa_off, Some(huge_page_pa + 0x1000));
}

// -- translate_kva: 1GB huge page --

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_1gb_huge_page() {
    // Map KVA via a 1GB page (PS bit set in PUD entry).
    let kva: u64 = 0xFFFF_8880_4000_0000; // 1GB-aligned
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let huge_page_pa: u64 = 0x4000_0000; // 1GB-aligned

    // Buffer must be large enough to hold PGD + PUD. We don't need
    // the actual 1GB page in the buffer — just verify the PA math.
    let size = (pud_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // PGD -> PUD
    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    // PUD entry with PS bit set = 1GB huge page
    write_entry(
        &mut buf,
        pud_pa,
        pud_idx,
        (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
    );

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(pgd_pa, Kva(kva), false);
    assert_eq!(pa, Some(huge_page_pa));

    // Offset within the 1GB page.
    let pa_off = mem.translate_kva(pgd_pa, Kva(kva + 0x1234_5678), false);
    assert_eq!(pa_off, Some(huge_page_pa + 0x1234_5678));
}

// -- translate_kva: not-present at each level --

#[test]
fn translate_kva_pgd_not_present() {
    // PGD entry with present bit clear.
    let kva: u64 = 0xFFFF_8880_0000_5000;
    let pgd_idx = (kva >> 39) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let size = (pgd_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    // Write PGD entry without present bit.
    let off = (pgd_pa + pgd_idx * 8) as usize;
    buf[off..off + 8].copy_from_slice(&0x2000u64.to_ne_bytes()); // no PRESENT

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(mem.translate_kva(pgd_pa, Kva(kva), false), None);
}

#[test]
fn translate_kva_pud_not_present() {
    let kva: u64 = 0xFFFF_8880_0000_5000;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let size = (pud_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // PGD present -> PUD
    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    // PUD entry without present bit.
    write_entry(&mut buf, pud_pa, pud_idx, 0x3000); // no PRESENT

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(mem.translate_kva(pgd_pa, Kva(kva), false), None);
}

#[test]
fn translate_kva_pmd_not_present() {
    let kva: u64 = 0xFFFF_8880_0000_5000;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let size = (pmd_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    // PMD entry without present bit.
    write_entry(&mut buf, pmd_pa, pmd_idx, 0x4000); // no PRESENT

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(mem.translate_kva(pgd_pa, Kva(kva), false), None);
}

#[test]
fn translate_kva_pte_not_present() {
    let kva: u64 = 0xFFFF_8880_0000_5000;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;
    let pte_idx = (kva >> 12) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let pte_pa: u64 = pmd_pa + 0x1000;
    let size = (pte_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
    // PTE entry without present bit.
    write_entry(&mut buf, pte_pa, pte_idx, 0x5000); // no PRESENT

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(mem.translate_kva(pgd_pa, Kva(kva), false), None);
}

// -- write_bpf_map_value tests --

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_u32_roundtrip() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Write u32 at offset 4 within the value region.
    assert!(write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        4,
        0xABCD_1234,
    ));
    // Read it back via direct PA access.
    assert_eq!(mem.read_u32(data_pa, 4), 0xABCD_1234);
}

#[test]
fn read_bytes_basic() {
    let buf = [1u8, 2, 3, 4, 5, 6, 7, 8];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let mut out = [0u8; 4];
    let n = mem.read_bytes(2, &mut out);
    assert_eq!(n, 4);
    assert_eq!(out, [3, 4, 5, 6]);
}

#[test]
fn read_bytes_past_end() {
    let buf = [1u8, 2, 3, 4];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let mut out = [0u8; 8];
    let n = mem.read_bytes(2, &mut out);
    assert_eq!(n, 2); // Only 2 bytes available from PA 2.
    assert_eq!(out[..2], [3, 4]);
}

#[test]
fn read_bytes_at_boundary() {
    let buf = [0xFFu8; 8];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let mut out = [0u8; 8];
    let n = mem.read_bytes(8, &mut out);
    assert_eq!(n, 0); // PA == size, nothing to read.
}

#[test]
fn write_u32_roundtrip() {
    let mut buf = [0u8; 16];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
    mem.write_u32(4, 0, 0xDEAD_BEEF);
    assert_eq!(mem.read_u32(4, 0), 0xDEAD_BEEF);
    assert_eq!(
        u32::from_ne_bytes(buf[4..8].try_into().unwrap()),
        0xDEAD_BEEF
    );
}

// -- xa_load tests --

#[test]
fn xa_load_zero_head() {
    let buf = [0u8; 64];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(xa_load(&mem, 0, 0, 0, 0, 0), Some(0));
    assert_eq!(xa_load(&mem, 0, 0, 5, 0, 0), Some(0));
}

#[test]
fn xa_load_single_entry_index_zero() {
    // xa_head with bit 1 clear = single-entry xarray.
    // Only index 0 returns the head value; others return 0.
    let xa_head: u64 = 0xFFFF_8880_0001_0000; // bit 1 clear
    assert_eq!(xa_head & 2, 0);
    let buf = [0u8; 8];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(xa_load(&mem, 0, xa_head, 0, 0, 0), Some(xa_head));
}

#[test]
fn xa_load_single_entry_index_nonzero() {
    let xa_head: u64 = 0xFFFF_8880_0001_0000;
    let buf = [0u8; 8];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(xa_load(&mem, 0, xa_head, 1, 0, 0), Some(0));
    assert_eq!(xa_load(&mem, 0, xa_head, 63, 0, 0), Some(0));
}

/// Build a single-level xa_node in a buffer. The node has shift=0
/// (leaf level) and the given slots populated with entry pointers.
/// Returns (buffer, xa_head pointing to the node, page_offset used).
///
/// Layout: node at DRAM offset 0x1000, slots at 0x1000 + slots_off.
/// kva_to_pa(node_kva, page_offset) = 0x1000.
fn setup_xa_node(slots: &[(u64, u64)], slots_off: usize) -> (Vec<u8>, u64, u64) {
    let node_pa: u64 = 0x1000;
    let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    let node_kva = page_offset.wrapping_add(node_pa);

    let size = (node_pa as usize) + slots_off + XA_CHUNK_SIZE as usize * 8 + 8;
    let mut buf = vec![0u8; size];

    // xa_node.shift at offset 0 = 0 (leaf level).
    buf[node_pa as usize] = 0;

    // Populate slots.
    for &(idx, entry) in slots {
        let slot_pa = node_pa + slots_off as u64 + idx * 8;
        buf[slot_pa as usize..slot_pa as usize + 8].copy_from_slice(&entry.to_ne_bytes());
    }

    // xa_head = node_kva | 2 (internal node marker).
    let xa_head = node_kva | 2;
    (buf, xa_head, page_offset)
}

#[test]
fn xa_load_multi_entry_populated_slot() {
    let slots_off = 16; // Simulated offset of slots within xa_node.
    let entry_ptr: u64 = 0xDEAD_0000; // Leaf entry (bit 1 clear).
    let (buf, xa_head, page_offset) = setup_xa_node(&[(3, entry_ptr)], slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 3, slots_off, 0),
        Some(entry_ptr)
    );
}

#[test]
fn xa_load_multi_entry_empty_slot() {
    let slots_off = 16;
    let (buf, xa_head, page_offset) = setup_xa_node(&[], slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    // All slots are zero.
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 0, slots_off, 0),
        Some(0)
    );
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 5, slots_off, 0),
        Some(0)
    );
}

#[test]
fn xa_load_multi_entry_multiple_slots() {
    let slots_off = 16;
    let entries = [
        (0, 0xAAAA_0000u64),
        (7, 0xBBBB_0000u64),
        (63, 0xCCCC_0000u64),
    ];
    let (buf, xa_head, page_offset) = setup_xa_node(&entries, slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 0, slots_off, 0),
        Some(0xAAAA_0000)
    );
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 7, slots_off, 0),
        Some(0xBBBB_0000)
    );
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 63, slots_off, 0),
        Some(0xCCCC_0000)
    );
    // Unpopulated slot.
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, 1, slots_off, 0),
        Some(0)
    );
}

// -- find_bpf_map tests --

/// Build a buffer with a mock IDR + bpf_map for find_bpf_map testing.
///
/// Layout:
/// - IDR at idr_pa (BSS region, translated via text_kva_to_pa)
/// - bpf_map at map_pa (vmalloc'd, translated via page table walk)
/// - Page table mapping map_kva -> map_pa
#[cfg(target_arch = "x86_64")]
fn setup_find_bpf_map(
    map_name: &str,
    map_type: u32,
    value_size: u32,
) -> (Vec<u8>, u64, u64, BpfMapOffsets) {
    // Offsets — use realistic values.
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    // Physical layout:
    // 0x0000..0x10000: padding / page tables
    // 0x10000: PGD
    // 0x11000: PUD
    // 0x12000: PMD
    // 0x13000: PTE
    // 0x14000: bpf_map/bpf_array data page
    // 0x15000: IDR data

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let map_pa: u64 = 0x14000;
    let idr_pa: u64 = 0x15000;

    // Choose a KVA for the bpf_map that will walk through our page table.
    let map_kva: u64 = 0xFFFF_C900_0000_0000;
    let pgd_idx = (map_kva >> 39) & 0x1FF;
    let pud_idx = (map_kva >> 30) & 0x1FF;
    let pmd_idx = (map_kva >> 21) & 0x1FF;
    let pte_idx = (map_kva >> 12) & 0x1FF;

    let size = 0x16000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };

    // Page table: PGD -> PUD -> PMD -> PTE -> map_pa.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte_idx * 8, (map_pa + PTE_BASE) | 0x63);

    // IDR: xa_head is a single-entry xarray pointing directly to map_kva.
    // Single entry = bit 1 clear on map_kva (it has bit 1 clear: 0x...0000).
    write_u64(&mut buf, idr_pa + offsets.idr_xa_head as u64, map_kva);
    // idr_next = 1: one map at index 0.
    write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 1);

    // bpf_map fields at map_pa.
    write_u32(&mut buf, map_pa + offsets.map_type as u64, map_type);
    write_u32(&mut buf, map_pa + offsets.value_size as u64, value_size);

    // Map name.
    let name_bytes = map_name.as_bytes();
    let name_pa = map_pa + offsets.map_name as u64;
    buf[name_pa as usize..name_pa as usize + name_bytes.len()].copy_from_slice(name_bytes);

    // IDR KVA: idr is in BSS, so text_kva_to_pa(idr_kva) = idr_pa.
    // text_kva_to_pa(kva) = kva - START_KERNEL_MAP.
    // So idr_kva = idr_pa + START_KERNEL_MAP.
    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    (buf, pgd_pa, idr_kva, offsets)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_discovers_matching_map() {
    let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map("mitosis.bss", BPF_MAP_TYPE_ARRAY, 64);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let result = find_bpf_map(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".bss",
    );

    let info = result.expect("should find the map");
    assert_eq!(info.name, "mitosis.bss");
    assert_eq!(info.map_type, BPF_MAP_TYPE_ARRAY);
    assert_eq!(info.value_size, 64);
    assert_eq!(info.map_pa, 0x14000);
    // value_kva = map_kva + array_value offset
    let map_kva: u64 = 0xFFFF_C900_0000_0000;
    assert_eq!(info.value_kva, Some(map_kva + offsets.array_value as u64));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_no_match_wrong_suffix() {
    let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map("mitosis.bss", BPF_MAP_TYPE_ARRAY, 64);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let result = find_bpf_map(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".data",
    );
    assert!(result.is_none());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_skips_non_array_type() {
    // map_type = 1 (BPF_MAP_TYPE_HASH), not BPF_MAP_TYPE_ARRAY.
    let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map("test.bss", 1, 64);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let result = find_bpf_map(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".bss",
    );
    assert!(result.is_none());
}

#[test]
fn find_bpf_map_empty_idr() {
    // IDR with xa_head = 0 (empty).
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };
    let idr_pa: u64 = 0x1000;
    let size = 0x2000;
    let buf = vec![0u8; size]; // All zeros, so xa_head = 0.

    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = find_bpf_map(
        &lookup_ctx(&mem, 0x10000, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".bss",
    );
    assert!(result.is_none());
}

// -- 5-level translate_kva tests --

/// Build a 5-level page table mapping a single 4KB page.
/// Returns (buffer, cr3_pa, mapped_kva, mapped_pa).
#[cfg(target_arch = "x86_64")]
fn setup_5level_page_table() -> (Vec<u8>, u64, u64, u64) {
    // Use a KVA with a non-zero PML5 index (bits 56:48).
    let kva: u64 = 0xFF11_8880_0000_5000;
    let pml5_idx = (kva >> 48) & 0x1FF;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;
    let pte_idx = (kva >> 12) & 0x1FF;

    let pml5_pa: u64 = 0x10000;
    let p4d_pa: u64 = pml5_pa + 0x1000;
    let pud_pa: u64 = p4d_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let pte_pa: u64 = pmd_pa + 0x1000;
    let data_pa: u64 = pte_pa + 0x1000;

    let size = (data_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // PML5[pml5_idx] -> P4D
    write_entry(&mut buf, pml5_pa, pml5_idx, (p4d_pa + PTE_BASE) | 0x63);
    // P4D/PGD[pgd_idx] -> PUD
    write_entry(&mut buf, p4d_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    // PUD[pud_idx] -> PMD
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    // PMD[pmd_idx] -> PTE
    write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
    // PTE[pte_idx] -> data page
    write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x63);

    // Write marker at data page.
    buf[data_pa as usize..data_pa as usize + 8]
        .copy_from_slice(&0x5555_AAAA_1234_5678u64.to_ne_bytes());

    (buf, pml5_pa, kva, data_pa)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_basic() {
    let (buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva), true);
    assert_eq!(pa, Some(data_pa));
    assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x5555_AAAA_1234_5678);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_with_offset() {
    let (buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva + 0x100), true);
    assert_eq!(pa, Some(data_pa + 0x100));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_unmapped_pml5() {
    let (buf, cr3_pa, _, _) = setup_5level_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    // Different PML5 index — no entry mapped.
    let unmapped_kva: u64 = 0xFF22_8880_0000_5000;
    assert_eq!(mem.translate_kva(cr3_pa, Kva(unmapped_kva), true), None);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_vs_4level_same_buffer() {
    // With l5=false on the same buffer, the walk starts at PGD (which
    // is our PML5). The PGD index from a 4-level perspective differs,
    // so it should fail to find a mapping.
    let (buf, cr3_pa, kva, _) = setup_5level_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    // 4-level walk uses bits 47:39 for PGD, not bits 56:48 for PML5.
    // The PGD index into our PML5 table won't find the right entry.
    let pa_4level = mem.translate_kva(cr3_pa, Kva(kva), false);
    // Should either be None (unmapped) or a different PA than 5-level.
    let pa_5level = mem.translate_kva(cr3_pa, Kva(kva), true);
    assert_ne!(pa_4level, pa_5level);
}

// -- write_bpf_map_value byte-by-byte across pages --

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_bytes_roundtrip() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 16,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    let payload = [0xDE, 0xAD, 0xBE, 0xEF];
    assert!(write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        &payload
    ));

    // Verify each byte was written.
    for (i, &expected) in payload.iter().enumerate() {
        assert_eq!(buf[data_pa as usize + i], expected);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_fails_on_unmapped_kva() {
    let (mut buf, cr3_pa, _, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 16,
        max_entries: 0,
        value_kva: Some(0xFFFF_FFFF_8000_0000), // Unmapped KVA.
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    assert!(!write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        &[0xFF]
    ));
}

// -- two-level xarray traversal --

/// Build a two-level xarray: root xa_node (shift=6) with one child
/// xa_node (shift=0) containing a leaf entry. Exercises the xa_load
/// loop's descent through internal nodes and the shift decrement.
///
/// Layout:
///   root node at PA 0x1000, shift=6
///   child node at PA 0x2000, shift=0
///   root.slots[child_slot] = child_kva | 2 (internal marker)
///   child.slots[leaf_slot] = leaf_entry (bit 1 clear)
///
/// Index = (child_slot << 6) | leaf_slot.
fn setup_two_level_xarray(
    child_slot: u64,
    leaf_slot: u64,
    leaf_entry: u64,
    slots_off: usize,
) -> (Vec<u8>, u64, u64) {
    let root_pa: u64 = 0x1000;
    let child_pa: u64 = 0x2000;
    let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    let root_kva = page_offset.wrapping_add(root_pa);
    let child_kva = page_offset.wrapping_add(child_pa);

    let size = (child_pa as usize) + slots_off + XA_CHUNK_SIZE as usize * 8 + 8;
    let mut buf = vec![0u8; size];

    // Root node: shift=6 (one level above leaf).
    buf[root_pa as usize] = 6;
    // Root slot[child_slot] -> child node (internal marker: bit 1 set).
    let root_slot_pa = root_pa + slots_off as u64 + child_slot * 8;
    buf[root_slot_pa as usize..root_slot_pa as usize + 8]
        .copy_from_slice(&(child_kva | 2).to_ne_bytes());

    // Child node: shift=0 (leaf level).
    buf[child_pa as usize] = 0;
    // Child slot[leaf_slot] -> leaf entry (bit 1 clear).
    let child_slot_pa = child_pa + slots_off as u64 + leaf_slot * 8;
    buf[child_slot_pa as usize..child_slot_pa as usize + 8]
        .copy_from_slice(&leaf_entry.to_ne_bytes());

    let xa_head = root_kva | 2;
    (buf, xa_head, page_offset)
}

#[test]
fn xa_load_two_level_finds_leaf() {
    let slots_off = 16;
    let child_slot = 1u64; // Root slot index for the child node.
    let leaf_slot = 5u64; // Child slot index for the leaf entry.
    let leaf_entry: u64 = 0xBEEF_0000; // Leaf (bit 1 clear).
    let index = (child_slot << 6) | leaf_slot; // = 69.

    let (buf, xa_head, page_offset) =
        setup_two_level_xarray(child_slot, leaf_slot, leaf_entry, slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    assert_eq!(
        xa_load(&mem, page_offset, xa_head, index, slots_off, 0),
        Some(leaf_entry)
    );
}

#[test]
fn xa_load_two_level_empty_child_slot() {
    let slots_off = 16;
    let child_slot = 2u64;
    let leaf_slot = 10u64;
    let leaf_entry: u64 = 0xAAAA_0000;

    let (buf, xa_head, page_offset) =
        setup_two_level_xarray(child_slot, leaf_slot, leaf_entry, slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    // Index that hits root slot 2, child slot 10 -> populated.
    let populated_idx = (child_slot << 6) | leaf_slot;
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, populated_idx, slots_off, 0),
        Some(leaf_entry)
    );

    // Index that hits root slot 2, but a different child slot -> 0.
    let empty_child_idx = child_slot << 6;
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, empty_child_idx, slots_off, 0),
        Some(0)
    );
}

#[test]
fn xa_load_two_level_empty_root_slot() {
    let slots_off = 16;
    let (buf, xa_head, page_offset) = setup_two_level_xarray(3, 0, 0xDEAD_0000, slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    // Index that maps to root slot 0 (empty, child is at slot 3).
    let empty_root_idx = 5u64; // root slot = 5 >> 6 = 0 (wait, index < 64 => root slot 0).
    // Actually: slot_idx = (index >> shift) & 63 = (5 >> 6) & 63 = 0.
    // Root slot 0 is empty (child is at slot 3).
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, empty_root_idx, slots_off, 0),
        Some(0)
    );
}

#[test]
fn xa_load_two_level_high_index() {
    let slots_off = 16;
    // Child at root slot 63, leaf at child slot 63. Max index for 2-level = 63*64+63 = 4095.
    let (buf, xa_head, page_offset) = setup_two_level_xarray(63, 63, 0xFFFF_0000, slots_off);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let max_index = (63 << 6) | 63; // 4095
    assert_eq!(
        xa_load(&mem, page_offset, xa_head, max_index, slots_off, 0),
        Some(0xFFFF_0000)
    );
}

// -- find_bpf_map: multiple IDR entries --

/// Build a buffer with multiple maps in the IDR (via xa_node).
/// First map has wrong name, second map matches.
#[cfg(target_arch = "x86_64")]
fn setup_find_bpf_map_multi() -> (Vec<u8>, u64, u64, BpfMapOffsets) {
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    // Physical layout:
    // 0x10000: PGD
    // 0x11000: PUD
    // 0x12000: PMD
    // 0x13000: PTE (maps map1_kva -> map1_pa and map2_kva -> map2_pa)
    // 0x14000: bpf_map 1 (wrong name)
    // 0x15000: bpf_map 2 (correct name)
    // 0x16000: IDR data
    // 0x17000: xa_node for IDR

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let map1_pa: u64 = 0x14000;
    let map2_pa: u64 = 0x15000;
    let idr_pa: u64 = 0x16000;
    let xa_node_pa: u64 = 0x17000;

    // Two distinct KVAs with different PTE indices.
    let map1_kva: u64 = 0xFFFF_C900_0000_0000;
    let map2_kva: u64 = 0xFFFF_C900_0000_1000;
    let pgd_idx = (map1_kva >> 39) & 0x1FF;
    let pud_idx = (map1_kva >> 30) & 0x1FF;
    let pmd_idx = (map1_kva >> 21) & 0x1FF;
    let pte1_idx = (map1_kva >> 12) & 0x1FF;
    let pte2_idx = (map2_kva >> 12) & 0x1FF;

    let page_offset: u64 = 0xFFFF_8880_0000_0000;
    let xa_node_kva = xa_node_pa + page_offset;

    let size = 0x18000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };

    // Page table.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte1_idx * 8, (map1_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte2_idx * 8, (map2_pa + PTE_BASE) | 0x63);

    // xa_node at xa_node_pa: shift=0 (leaf), with two entries.
    buf[xa_node_pa as usize] = 0; // shift=0
    // Slot 0 -> map1_kva (leaf, bit 1 clear).
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64,
        map1_kva,
    );
    // Slot 1 -> map2_kva (leaf, bit 1 clear).
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64 + 8,
        map2_kva,
    );

    // IDR xa_head -> xa_node (internal marker: bit 1 set).
    write_u64(
        &mut buf,
        idr_pa + offsets.idr_xa_head as u64,
        xa_node_kva | 2,
    );
    // idr_next = 2: two maps at indices 0 and 1.
    write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

    // Map 1: "other.data", BPF_MAP_TYPE_ARRAY.
    write_u32(
        &mut buf,
        map1_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map1_pa + offsets.value_size as u64, 32);
    let name1 = b"other.data";
    let name1_pa = map1_pa + offsets.map_name as u64;
    buf[name1_pa as usize..name1_pa as usize + name1.len()].copy_from_slice(name1);

    // Map 2: "mitosis.bss", BPF_MAP_TYPE_ARRAY.
    write_u32(
        &mut buf,
        map2_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map2_pa + offsets.value_size as u64, 128);
    let name2 = b"mitosis.bss";
    let name2_pa = map2_pa + offsets.map_name as u64;
    buf[name2_pa as usize..name2_pa as usize + name2.len()].copy_from_slice(name2);

    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    (buf, pgd_pa, idr_kva, offsets)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_skips_wrong_name_finds_second() {
    let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map_multi();
    let page_offset: u64 = 0xFFFF_8880_0000_0000;
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let result = find_bpf_map(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        idr_kva,
        ".bss",
    );
    let info = result.expect("should find second map");
    assert_eq!(info.name, "mitosis.bss");
    assert_eq!(info.map_pa, 0x15000);
    assert_eq!(info.value_size, 128);
}

// -- find_bpf_map with full-length name (no null terminator) --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_full_length_name() {
    // Map name fills all BPF_OBJ_NAME_LEN bytes with no null.
    let full_name = "0123456789a.bss"; // 15 bytes, fits in 16 with null.
    let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map(full_name, BPF_MAP_TYPE_ARRAY, 64);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let result = find_bpf_map(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".bss",
    );
    let info = result.expect("should find map with 15-char name");
    assert_eq!(info.name, full_name);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_max_length_name_no_null() {
    // Exactly 16 bytes, no null terminator.
    let max_name = "0123456789a.bss!"; // 16 bytes
    assert_eq!(max_name.len(), BPF_OBJ_NAME_LEN);
    let (mut buf, cr3_pa, idr_kva, offsets) =
        setup_find_bpf_map("placeholder.bss", BPF_MAP_TYPE_ARRAY, 64);
    // Overwrite the name region with exactly 16 non-null bytes.
    let map_pa: u64 = 0x14000;
    let name_pa = (map_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + 16].copy_from_slice(max_name.as_bytes());
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    // The name doesn't end with ".bss" — the '!' is the 16th char.
    let result = find_bpf_map(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".bss",
    );
    assert!(
        result.is_none(),
        "16-byte name ending with '!' should not match .bss suffix"
    );
}

// -- write_bpf_map_value with nonzero offset --

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_nonzero_offset() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
    // Record the original bytes at data_pa before writing.
    let original_first_byte = buf[data_pa as usize];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Write at offset 8 within the value region.
    let payload = [0x11, 0x22, 0x33, 0x44];
    assert!(write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        8,
        &payload
    ));

    for (i, &expected) in payload.iter().enumerate() {
        assert_eq!(buf[data_pa as usize + 8 + i], expected);
    }
    // Bytes before offset should be untouched (still the marker data).
    assert_eq!(buf[data_pa as usize], original_first_byte);
}

// -- write_bpf_map_value with empty data --

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_empty_data() {
    let (mut buf, cr3_pa, kva, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Zero-length write should succeed without doing anything.
    assert!(write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        &[]
    ));
}

// -- write_bpf_map_value_u32 with 5-level paging --

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_u32_5level() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    assert!(write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, true),
        &info,
        0,
        0xCAFE_BABE,
    ));
    assert_eq!(mem.read_u32(data_pa, 0), 0xCAFE_BABE);
}

// -- 5-level: not-present at P4D level --

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_p4d_not_present() {
    // PML5 entry is present but the P4D (delegated to walk_4level as
    // PGD) has no entry for the requested PGD index.
    let kva: u64 = 0xFF11_8880_0000_5000;
    let pml5_idx = (kva >> 48) & 0x1FF;

    let pml5_pa: u64 = 0x10000;
    let p4d_pa: u64 = pml5_pa + 0x1000;

    // Buffer has PML5 -> P4D, but P4D is all zeros (no PGD entries).
    let size = (p4d_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let off = (pml5_pa + pml5_idx * 8) as usize;
    buf[off..off + 8].copy_from_slice(&((p4d_pa + PTE_BASE) | 0x63).to_ne_bytes());

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(mem.translate_kva(pml5_pa, Kva(kva), true), None);
}

// -- 5-level: 2MB huge page --

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_2mb_huge_page() {
    let kva: u64 = 0xFF11_8880_0020_0000; // 2MB-aligned
    let pml5_idx = (kva >> 48) & 0x1FF;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;

    let pml5_pa: u64 = 0x10000;
    let p4d_pa: u64 = pml5_pa + 0x1000;
    let pud_pa: u64 = p4d_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let huge_page_pa: u64 = 0x20_0000;

    let size = (huge_page_pa + 0x20_0000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pml5_pa, pml5_idx, (p4d_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, p4d_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    write_entry(
        &mut buf,
        pmd_pa,
        pmd_idx,
        (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
    ); // PS bit

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(pml5_pa, Kva(kva), true);
    assert_eq!(pa, Some(huge_page_pa));

    let pa_off = mem.translate_kva(pml5_pa, Kva(kva + 0x1234), true);
    assert_eq!(pa_off, Some(huge_page_pa + 0x1234));
}

// -- 5-level: 1GB huge page --

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_5level_1gb_huge_page() {
    let kva: u64 = 0xFF11_8880_4000_0000; // 1GB-aligned
    let pml5_idx = (kva >> 48) & 0x1FF;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;

    let pml5_pa: u64 = 0x10000;
    let p4d_pa: u64 = pml5_pa + 0x1000;
    let pud_pa: u64 = p4d_pa + 0x1000;
    let huge_page_pa: u64 = 0x4000_0000;

    let size = (pud_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pml5_pa, pml5_idx, (p4d_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, p4d_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(
        &mut buf,
        pud_pa,
        pud_idx,
        (huge_page_pa + PTE_BASE) | BLOCK_FLAGS,
    ); // PS bit

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(pml5_pa, Kva(kva), true);
    assert_eq!(pa, Some(huge_page_pa));

    let pa_off = mem.translate_kva(pml5_pa, Kva(kva + 0x1234_5678), true);
    assert_eq!(pa_off, Some(huge_page_pa + 0x1234_5678));
}

// -- find_bpf_map with translate_kva failure on first entry --

#[test]
fn find_bpf_map_skips_untranslatable_entry() {
    // IDR has a single entry whose KVA cannot be translated
    // (no page table mapping for it). find_bpf_map should continue
    // past it and return None (no other entries).
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    let idr_pa: u64 = 0x1000;
    let pgd_pa: u64 = 0x10000;
    let size = 0x12000;
    let mut buf = vec![0u8; size];

    // IDR xa_head = a KVA with no page table entry.
    // Single-entry xarray (bit 1 clear on the KVA).
    let unmappable_kva: u64 = 0xFFFF_C900_DEAD_0000;
    assert_eq!(unmappable_kva & 2, 0);
    let off = (idr_pa + offsets.idr_xa_head as u64) as usize;
    buf[off..off + 8].copy_from_slice(&unmappable_kva.to_ne_bytes());
    // idr_next = 1.
    let off_next = (idr_pa + offsets.idr_next as u64) as usize;
    buf[off_next..off_next + 4].copy_from_slice(&1u32.to_ne_bytes());

    // PGD exists but is all zeros — no entries.
    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = find_bpf_map(
        &lookup_ctx(&mem, pgd_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
        ".bss",
    );
    assert!(result.is_none());
}

// -- find_bpf_map with 5-level paging --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_5level() {
    // Verify find_bpf_map works when l5=true by constructing a
    // 5-level page table mapping the bpf_map.
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    let map_kva: u64 = 0xFF11_C900_0000_0000;
    let pml5_idx = (map_kva >> 48) & 0x1FF;
    let pgd_idx = (map_kva >> 39) & 0x1FF;
    let pud_idx = (map_kva >> 30) & 0x1FF;
    let pmd_idx = (map_kva >> 21) & 0x1FF;
    let pte_idx = (map_kva >> 12) & 0x1FF;

    let pml5_pa: u64 = 0x10000;
    let p4d_pa: u64 = 0x11000;
    let pud_pa: u64 = 0x12000;
    let pmd_pa: u64 = 0x13000;
    let pte_pa: u64 = 0x14000;
    let map_pa: u64 = 0x15000;
    let idr_pa: u64 = 0x16000;

    let size = 0x17000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };

    // 5-level page table.
    write_u64(&mut buf, pml5_pa + pml5_idx * 8, (p4d_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, p4d_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte_idx * 8, (map_pa + PTE_BASE) | 0x63);

    // IDR: single-entry xarray.
    write_u64(&mut buf, idr_pa + offsets.idr_xa_head as u64, map_kva);
    // idr_next = 1.
    write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 1);

    // bpf_map at map_pa.
    write_u32(
        &mut buf,
        map_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map_pa + offsets.value_size as u64, 96);
    let name = b"test.bss";
    let name_pa = (map_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + name.len()].copy_from_slice(name);

    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = find_bpf_map(
        &lookup_ctx(&mem, pml5_pa, 0xFFFF_8880_0000_0000, &offsets, true),
        idr_kva,
        ".bss",
    );

    let info = result.expect("should find map via 5-level walk");
    assert_eq!(info.name, "test.bss");
    assert_eq!(info.map_pa, map_pa);
    assert_eq!(info.value_size, 96);
    assert_eq!(info.value_kva, Some(map_kva + offsets.array_value as u64));
}

// -- write_bpf_map_value across page boundary --

/// Build a page table mapping two consecutive 4KB virtual pages to
/// two physical pages. Returns (buffer, cr3_pa, base_kva, page1_pa, page2_pa).
#[cfg(target_arch = "x86_64")]
fn setup_two_page_table() -> (Vec<u8>, u64, u64, u64, u64) {
    let kva: u64 = 0xFFFF_8880_0000_5000;
    let kva2: u64 = kva + 0x1000;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;
    let pte1_idx = (kva >> 12) & 0x1FF;
    let pte2_idx = (kva2 >> 12) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let pte_pa: u64 = pmd_pa + 0x1000;
    let page1_pa: u64 = pte_pa + 0x1000;
    let page2_pa: u64 = page1_pa + 0x1000;

    let size = (page2_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pte_pa, pte1_idx, (page1_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pte_pa, pte2_idx, (page2_pa + PTE_BASE) | 0x63);

    (buf, pgd_pa, kva, page1_pa, page2_pa)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_across_page_boundary() {
    let (mut buf, cr3_pa, kva, page1_pa, page2_pa) = setup_two_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 0x2000,
        max_entries: 0,
        // value_kva at the start of page 1.
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Write a u32 at offset 0xFFE within the value region.
    // Bytes 0..2 land on page 1 (PA page1_pa + 0xFFE..0x1000),
    // bytes 2..4 land on page 2 (PA page2_pa + 0x000..0x002).
    let val: u32 = 0xAABB_CCDD;
    assert!(write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0xFFE,
        val,
    ));

    // Verify bytes on page 1 (last 2 bytes of the page).
    let b = val.to_ne_bytes();
    assert_eq!(buf[page1_pa as usize + 0xFFE], b[0]);
    assert_eq!(buf[page1_pa as usize + 0xFFF], b[1]);
    // Verify bytes on page 2 (first 2 bytes).
    assert_eq!(buf[page2_pa as usize], b[2]);
    assert_eq!(buf[page2_pa as usize + 1], b[3]);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn write_bpf_map_value_single_byte_on_second_page() {
    let (mut buf, cr3_pa, kva, _, page2_pa) = setup_two_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 0x2000,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Write exactly at offset 0x1000 — first byte of page 2.
    assert!(write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0x1000,
        &[0x42],
    ));
    assert_eq!(buf[page2_pa as usize], 0x42);
}

// -- find_bpf_map: first entry untranslatable, second succeeds --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_bpf_map_skips_untranslatable_finds_translatable() {
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    // Physical layout:
    // 0x10000: PGD
    // 0x11000: PUD
    // 0x12000: PMD
    // 0x13000: PTE (only maps map2_kva -> map2_pa; no entry for map1_kva)
    // 0x14000: bpf_map 2 (matching)
    // 0x15000: IDR data
    // 0x16000: xa_node

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let map2_pa: u64 = 0x14000;
    let idr_pa: u64 = 0x15000;
    let xa_node_pa: u64 = 0x16000;

    // map1_kva has no PTE entry; map2_kva does.
    let map1_kva: u64 = 0xFFFF_C900_0000_0000;
    let map2_kva: u64 = 0xFFFF_C900_0000_1000;
    let pgd_idx = (map2_kva >> 39) & 0x1FF;
    let pud_idx = (map2_kva >> 30) & 0x1FF;
    let pmd_idx = (map2_kva >> 21) & 0x1FF;
    let pte2_idx = (map2_kva >> 12) & 0x1FF;
    // map1_kva and map2_kva share PGD/PUD/PMD indices (they differ
    // only in bits 12..21). PTE index for map1_kva has no entry.

    let page_offset: u64 = 0xFFFF_8880_0000_0000;
    let xa_node_kva = xa_node_pa + page_offset;

    let size = 0x17000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };

    // Page table — only map2_kva is mapped.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    // Only PTE for map2_kva. map1_kva's PTE slot is zero (not present).
    write_u64(&mut buf, pte_pa + pte2_idx * 8, (map2_pa + PTE_BASE) | 0x63);

    // xa_node: slot 0 -> map1_kva (untranslatable), slot 1 -> map2_kva.
    buf[xa_node_pa as usize] = 0; // shift=0
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64,
        map1_kva,
    );
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64 + 8,
        map2_kva,
    );

    // IDR xa_head -> xa_node.
    write_u64(
        &mut buf,
        idr_pa + offsets.idr_xa_head as u64,
        xa_node_kva | 2,
    );
    // idr_next = 2: entries at slots 0 and 1.
    write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

    // Map 2: "target.bss", BPF_MAP_TYPE_ARRAY.
    write_u32(
        &mut buf,
        map2_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map2_pa + offsets.value_size as u64, 200);
    let name = b"target.bss";
    let name_pa = (map2_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + name.len()].copy_from_slice(name);

    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = find_bpf_map(
        &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
        idr_kva,
        ".bss",
    );

    let info = result.expect("should skip untranslatable entry and find the second");
    assert_eq!(info.name, "target.bss");
    assert_eq!(info.map_pa, map2_pa);
    assert_eq!(info.value_size, 200);
}

// -- read_bpf_map_value tests --

#[test]
#[cfg(target_arch = "x86_64")]
fn read_bpf_map_value_u32_roundtrip() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
    // Write a known u32 at data_pa + 4.
    buf[data_pa as usize + 4..data_pa as usize + 8].copy_from_slice(&0xCAFE_BABEu32.to_ne_bytes());
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    let val = read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 4);
    assert_eq!(val, Some(0xCAFE_BABE));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_bpf_map_value_bytes() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_page_table();
    buf[data_pa as usize..data_pa as usize + 4].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    let bytes = read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 4);
    assert_eq!(bytes, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_bpf_map_value_empty() {
    let (buf, cr3_pa, kva, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    let bytes = read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 0);
    assert_eq!(bytes, Some(vec![]));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_bpf_map_value_unmapped_returns_none() {
    let (buf, cr3_pa, _, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 16,
        max_entries: 0,
        value_kva: Some(0xFFFF_FFFF_8000_0000), // Unmapped KVA.
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    assert_eq!(
        read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 4),
        None
    );
    assert_eq!(
        read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 0),
        None
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn write_then_read_bpf_map_value_roundtrip() {
    let (mut buf, cr3_pa, kva, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Write then read u32.
    assert!(write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        8,
        0x1234_5678,
    ));
    assert_eq!(
        read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 8),
        Some(0x1234_5678)
    );

    // Write then read bytes.
    let payload = [0x11, 0x22, 0x33, 0x44, 0x55];
    assert!(write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        16,
        &payload,
    ));
    assert_eq!(
        read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 16, 5),
        Some(payload.to_vec()),
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_bpf_map_value_across_page_boundary() {
    let (mut buf, cr3_pa, kva, page1_pa, page2_pa) = setup_two_page_table();
    // Write known bytes at the page boundary.
    buf[page1_pa as usize + 0xFFE] = 0xAA;
    buf[page1_pa as usize + 0xFFF] = 0xBB;
    buf[page2_pa as usize] = 0xCC;
    buf[page2_pa as usize + 1] = 0xDD;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 0x2000,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    let bytes = read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0xFFE, 4);
    assert_eq!(bytes, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_bpf_map_value_u32_5level() {
    let (mut buf, cr3_pa, kva, data_pa) = setup_5level_page_table();
    buf[data_pa as usize..data_pa as usize + 4].copy_from_slice(&0xDEAD_BEEFu32.to_ne_bytes());
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    assert_eq!(
        read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, true), &info, 0),
        Some(0xDEAD_BEEF)
    );
}

// -- find_all_bpf_maps tests --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_all_bpf_maps_returns_both_types() {
    // Reuse multi-map helper but change map1 to HASH type.
    let mut setup = setup_find_bpf_map_multi();
    let map1_pa: u64 = 0x14000;
    // Overwrite map1's map_type from ARRAY (2) to HASH (1).
    let map_type_off = setup.3.map_type;
    let off = (map1_pa + map_type_off as u64) as usize;
    setup.0[off..off + 4].copy_from_slice(&1u32.to_ne_bytes());

    let (buf, cr3_pa, idr_kva, offsets) = setup;
    let page_offset: u64 = 0xFFFF_8880_0000_0000;
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        idr_kva,
    );
    assert_eq!(maps.len(), 2);
    let hash_map = maps.iter().find(|m| m.name == "other.data");
    let array_map = maps.iter().find(|m| m.name == "mitosis.bss");
    assert!(hash_map.is_some(), "HASH map should be in results");
    assert!(array_map.is_some(), "ARRAY map should be in results");
    assert_eq!(hash_map.unwrap().map_type, 1); // BPF_MAP_TYPE_HASH
    assert!(hash_map.unwrap().value_kva.is_none());
    assert_eq!(array_map.unwrap().map_type, BPF_MAP_TYPE_ARRAY);
    assert!(array_map.unwrap().value_kva.is_some());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_all_bpf_maps_single_entry() {
    let (buf, cr3_pa, idr_kva, offsets) = setup_find_bpf_map("test.bss", BPF_MAP_TYPE_ARRAY, 64);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
    );
    assert_eq!(maps.len(), 1);
    assert_eq!(maps[0].name, "test.bss");
}

#[test]
fn find_all_bpf_maps_empty_idr() {
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };
    let buf = vec![0u8; 0x2000];
    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = 0x1000 + start_kernel_map;
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, 0x10000, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
    );
    assert!(maps.is_empty());
}

// -- value_kva Option tests --

#[test]
#[cfg(target_arch = "x86_64")]
fn read_value_returns_none_for_non_array_map() {
    let (buf, cr3_pa, _, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "hash.map".into(),
        map_type: 1, // HASH
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 0, 4).is_none());
    assert!(read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 0).is_none());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn write_value_returns_false_for_non_array_map() {
    let (mut buf, cr3_pa, _, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "hash.map".into(),
        map_type: 1, // HASH
        map_flags: 0,
        key_size: 0,
        value_size: 64,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    assert!(!write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        &[1, 2, 3, 4],
    ));
    assert!(!write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        42
    ));
}

// -- map_flags test --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_all_bpf_maps_reads_map_flags() {
    let (mut buf, cr3_pa, idr_kva, offsets) =
        setup_find_bpf_map("flagged.bss", BPF_MAP_TYPE_ARRAY, 64);
    // Write non-zero map_flags at the correct offset.
    let map_pa: u64 = 0x14000;
    let flags_pa = (map_pa + offsets.map_flags as u64) as usize;
    buf[flags_pa..flags_pa + 4].copy_from_slice(&0x0400u32.to_ne_bytes()); // BPF_F_MMAPABLE

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
    );
    assert_eq!(maps.len(), 1);
    assert_eq!(maps[0].map_flags, 0x0400);
}

// -- xa_node_shift non-zero offset test --

#[test]
fn xa_node_shift_nonzero_offset() {
    // Place shift at offset 8 within the xa_node instead of 0.
    let node_pa: u64 = 0x1000;
    let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    let node_kva = page_offset.wrapping_add(node_pa);
    let shift_off: usize = 8;

    let mut buf = vec![0u8; 0x2000];
    // Write shift=6 at node_pa + 8.
    buf[node_pa as usize + shift_off] = 6;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    assert_eq!(xa_node_shift(&mem, page_offset, node_kva, shift_off), 6);
    // With offset 0 (wrong), should read 0 (the byte at node_pa + 0).
    assert_eq!(xa_node_shift(&mem, page_offset, node_kva, 0), 0);
}

// -- xa_load continues past failed entry --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_all_bpf_maps_continues_past_untranslatable_entry() {
    // IDR with two entries via xa_node. First entry has an
    // untranslatable KVA (no page table mapping). Second entry
    // is a valid ARRAY map. find_all_bpf_maps should skip the
    // first and return the second.
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let map_pa: u64 = 0x14000;
    let idr_pa: u64 = 0x15000;
    let xa_node_pa: u64 = 0x16000;

    let map_kva: u64 = 0xFFFF_C900_0000_0000;
    let pgd_idx = (map_kva >> 39) & 0x1FF;
    let pud_idx = (map_kva >> 30) & 0x1FF;
    let pmd_idx = (map_kva >> 21) & 0x1FF;
    let pte_idx = (map_kva >> 12) & 0x1FF;

    // Unmappable KVA: different PGD index, no page table entry.
    let bad_kva: u64 = 0xFFFF_C900_8000_0000;

    let page_offset: u64 = 0xFFFF_8880_0000_0000;
    let xa_node_kva = xa_node_pa + page_offset;

    let size = 0x17000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };

    // Page table for map_kva only.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte_idx * 8, (map_pa + PTE_BASE) | 0x63);

    // xa_node with two entries: slot 0 = bad_kva, slot 1 = map_kva.
    buf[xa_node_pa as usize] = 0; // shift=0
    write_u64(&mut buf, xa_node_pa + offsets.xa_node_slots as u64, bad_kva);
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64 + 8,
        map_kva,
    );

    // IDR xa_head -> xa_node.
    write_u64(
        &mut buf,
        idr_pa + offsets.idr_xa_head as u64,
        xa_node_kva | 2,
    );
    // idr_next = 2: entries at slots 0 and 1.
    write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

    // Valid map at map_pa.
    write_u32(
        &mut buf,
        map_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map_pa + offsets.value_size as u64, 64);
    let name = b"good.bss";
    let name_pa = (map_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + name.len()].copy_from_slice(name);

    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
        idr_kva,
    );

    // Should find the second map despite the first being untranslatable.
    let good = maps.iter().find(|m| m.name == "good.bss");
    assert!(
        good.is_some(),
        "good.bss should be found despite bad entry at slot 0"
    );
    assert_eq!(good.unwrap().map_type, BPF_MAP_TYPE_ARRAY);
}

// -- bounds check tests --

#[test]
#[cfg(target_arch = "x86_64")]
fn read_value_rejects_out_of_bounds() {
    let (buf, cr3_pa, kva, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 8,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Exactly at boundary: offset=4, len=4 -> 4+4=8 == value_size, ok.
    assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 4, 4).is_some());
    // One past: offset=4, len=5 -> 4+5=9 > 8, rejected.
    assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 4, 5).is_none());
    // Offset past end: offset=9, len=1 -> 9+1=10 > 8, rejected.
    assert!(read_bpf_map_value(&value_ctx(&mem, cr3_pa, false), &info, 9, 1).is_none());
    // u32 past end: offset=6, 6+4=10 > 8, rejected.
    assert!(read_bpf_map_value_u32(&value_ctx(&mem, cr3_pa, false), &info, 6).is_none());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn write_value_rejects_out_of_bounds() {
    let (mut buf, cr3_pa, kva, _) = setup_page_table();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 8,
        max_entries: 0,
        value_kva: Some(kva),
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Within bounds: offset=0, len=8.
    assert!(write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        &[0u8; 8],
    ));
    // Past end: offset=0, len=9.
    assert!(!write_bpf_map_value(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        0,
        &[0u8; 9],
    ));
    // u32 past end: offset=6, 6+4=10 > 8.
    assert!(!write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        6,
        42
    ));
    // u32 at boundary: offset=4, 4+4=8, ok.
    assert!(write_bpf_map_value_u32(
        &value_ctx(&mem, cr3_pa, false),
        &info,
        4,
        42
    ));
}

// -- BpfMapInfo btf fields --

#[test]
fn bpf_map_info_btf_fields_default_zero() {
    let info = BpfMapInfo {
        map_pa: 0x1000,
        map_kva: 0,
        name: "test".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 32,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    assert_eq!(info.btf_kva, 0);
    assert_eq!(info.btf_value_type_id, 0);
}

#[test]
fn bpf_map_info_btf_fields_populated() {
    let info = BpfMapInfo {
        map_pa: 0x1000,
        map_kva: 0,
        name: "test".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 0,
        value_size: 32,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0xFFFF_8880_0001_0000,
        btf_value_type_id: 42,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    assert_eq!(info.btf_kva, 0xFFFF_8880_0001_0000);
    assert_eq!(info.btf_value_type_id, 42);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn find_all_bpf_maps_populates_btf_fields() {
    let (mut buf, cr3_pa, idr_kva, mut offsets) =
        setup_find_bpf_map("test.bss", BPF_MAP_TYPE_ARRAY, 64);

    // Place btf fields at offsets that don't overlap existing fields.
    offsets.map_btf = 56;
    offsets.map_btf_value_type_id = 64;

    let map_pa: u64 = 0x14000;
    let btf_off = (map_pa + offsets.map_btf as u64) as usize;
    let btf_tid_off = (map_pa + offsets.map_btf_value_type_id as u64) as usize;

    // Zero out the btf fields first — default from zeroed buf.
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
    );
    assert_eq!(maps.len(), 1);
    assert_eq!(maps[0].btf_kva, 0);
    assert_eq!(maps[0].btf_value_type_id, 0);

    // Write nonzero values and re-scan.
    let btf_kva_val: u64 = 0xFFFF_8880_DEAD_0000;
    buf[btf_off..btf_off + 8].copy_from_slice(&btf_kva_val.to_ne_bytes());
    buf[btf_tid_off..btf_tid_off + 4].copy_from_slice(&7u32.to_ne_bytes());

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, cr3_pa, 0xFFFF_8880_0000_0000, &offsets, false),
        idr_kva,
    );
    assert_eq!(maps[0].btf_kva, btf_kva_val);
    assert_eq!(maps[0].btf_value_type_id, 7);
}

// -- idr_next scan bound --

#[test]
#[cfg(target_arch = "x86_64")]
fn find_all_bpf_maps_respects_idr_next_bound() {
    // Build IDR with 3 slots in xa_node, but set idr_next=2.
    // Only indices 0 and 1 should be scanned.
    let offsets = BpfMapOffsets {
        map_name: 32,
        map_type: 24,
        map_flags: 28,
        key_size: 44,
        value_size: 48,
        max_entries: 52,
        array_value: 256,
        xa_node_slots: 16,
        xa_node_shift: 0,
        idr_xa_head: 8,
        idr_next: 20,
        map_btf: 0,
        map_btf_value_type_id: 0,
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    };

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let map_pa: u64 = 0x14000;
    let map2_pa: u64 = 0x15000;
    let map3_pa: u64 = 0x16000;
    let idr_pa: u64 = 0x17000;
    let xa_node_pa: u64 = 0x18000;

    let map_kva: u64 = 0xFFFF_C900_0000_0000;
    let map2_kva: u64 = 0xFFFF_C900_0000_1000;
    let map3_kva: u64 = 0xFFFF_C900_0000_2000;
    let pgd_idx = (map_kva >> 39) & 0x1FF;
    let pud_idx = (map_kva >> 30) & 0x1FF;
    let pmd_idx = (map_kva >> 21) & 0x1FF;
    let pte1_idx = (map_kva >> 12) & 0x1FF;
    let pte2_idx = (map2_kva >> 12) & 0x1FF;
    let pte3_idx = (map3_kva >> 12) & 0x1FF;

    let page_offset: u64 = 0xFFFF_8880_0000_0000;
    let xa_node_kva = xa_node_pa + page_offset;

    let size = 0x19000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };

    // Page table for all three map KVAs.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte1_idx * 8, (map_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte2_idx * 8, (map2_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte3_idx * 8, (map3_pa + PTE_BASE) | 0x63);

    // xa_node with 3 entries.
    buf[xa_node_pa as usize] = 0; // shift=0
    write_u64(&mut buf, xa_node_pa + offsets.xa_node_slots as u64, map_kva);
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64 + 8,
        map2_kva,
    );
    write_u64(
        &mut buf,
        xa_node_pa + offsets.xa_node_slots as u64 + 2 * 8,
        map3_kva,
    );

    // IDR: xa_head -> xa_node, idr_next = 2 (only scan 0..2).
    write_u64(
        &mut buf,
        idr_pa + offsets.idr_xa_head as u64,
        xa_node_kva | 2,
    );
    write_u32(&mut buf, idr_pa + offsets.idr_next as u64, 2);

    // Map 1 at slot 0.
    write_u32(
        &mut buf,
        map_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map_pa + offsets.value_size as u64, 32);
    let name = b"first.bss";
    let name_pa = (map_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + name.len()].copy_from_slice(name);

    // Map 2 at slot 1.
    write_u32(
        &mut buf,
        map2_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map2_pa + offsets.value_size as u64, 64);
    let name = b"second.bss";
    let name_pa = (map2_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + name.len()].copy_from_slice(name);

    // Map 3 at slot 2 — should NOT be found because idr_next=2.
    write_u32(
        &mut buf,
        map3_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_ARRAY,
    );
    write_u32(&mut buf, map3_pa + offsets.value_size as u64, 128);
    let name = b"third.bss";
    let name_pa = (map3_pa + offsets.map_name as u64) as usize;
    buf[name_pa..name_pa + name.len()].copy_from_slice(name);

    let start_kernel_map: u64 = START_KERNEL_MAP;
    let idr_kva = idr_pa + start_kernel_map;

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let maps = find_all_bpf_maps(
        &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
        idr_kva,
    );

    // Only 2 maps should be found (idr_next=2 means scan 0..2).
    assert_eq!(maps.len(), 2);
    assert!(maps.iter().any(|m| m.name == "first.bss"));
    assert!(maps.iter().any(|m| m.name == "second.bss"));
    assert!(!maps.iter().any(|m| m.name == "third.bss"));
}

// -- translate_kva in kernel image / vmalloc region --

/// Build a page table mapping KVA 0xFFFF_8000_8400_5000 (KIMAGE_VADDR
/// region on aarch64, vmalloc range where BPF maps live).
///
/// x86_64: 4-level walk, 4KB pages, PGD index 256.
/// aarch64 (64KB granule): 3-level walk, PGD index 32.
#[cfg(target_arch = "x86_64")]
fn setup_page_table_vmalloc() -> (Vec<u8>, u64, u64, u64) {
    let kva: u64 = 0xFFFF_8000_8400_5000;
    let pgd_idx = (kva >> 39) & 0x1FF;
    let pud_idx = (kva >> 30) & 0x1FF;
    let pmd_idx = (kva >> 21) & 0x1FF;
    let pte_idx = (kva >> 12) & 0x1FF;

    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = pgd_pa + 0x1000;
    let pmd_pa: u64 = pud_pa + 0x1000;
    let pte_pa: u64 = pmd_pa + 0x1000;
    let data_pa: u64 = pte_pa + 0x1000;

    let size = (data_pa + 0x1000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pgd_pa, pgd_idx, (pud_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pud_pa, pud_idx, (pmd_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x63);
    write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x63);

    // Write known data at the target page.
    buf[data_pa as usize..data_pa as usize + 8]
        .copy_from_slice(&0x1234_5678_ABCD_EF00u64.to_ne_bytes());

    (buf, pgd_pa, kva, data_pa)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_l0_index_256() {
    let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva), false);
    assert_eq!(
        pa,
        Some(data_pa),
        "L0[256] walk should resolve to data page"
    );
    assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x1234_5678_ABCD_EF00);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_l0_index_256_with_offset() {
    let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva + 0x100), false);
    assert_eq!(pa, Some(data_pa + 0x100));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn translate_kva_l0_index_256_unmapped_neighbor() {
    let (buf, cr3_pa, kva, _) = setup_page_table_vmalloc();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let kva_257 = kva + (1u64 << 39);
    assert_eq!(mem.translate_kva(cr3_pa, Kva(kva_257), false), None);
}

// -- aarch64 64KB granule vmalloc region tests --

/// Build a 3-level page table for 64KB granule mapping KVA
/// 0xFFFF_8000_8400_0000 (KIMAGE_VADDR region, PGD index 32).
#[cfg(target_arch = "aarch64")]
fn setup_page_table_vmalloc_64k() -> (Vec<u8>, u64, u64, u64) {
    let kva: u64 = 0xFFFF_8000_8400_0000;
    let pgd_idx = (kva >> 42) & 0x3F; // 32
    let pmd_idx = (kva >> 29) & 0x1FFF; // 4
    let pte_idx = (kva >> 16) & 0x1FFF; // 0

    let pgd_pa: u64 = 0x10000;
    let pmd_pa: u64 = 0x20000;
    let pte_pa: u64 = 0x30000;
    let data_pa: u64 = 0x40000;

    let size = (data_pa + 0x10000) as usize;
    let mut buf = vec![0u8; size];

    let write_entry = |buf: &mut Vec<u8>, base: u64, idx: u64, val: u64| {
        let off = (base + idx * 8) as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    write_entry(&mut buf, pgd_pa, pgd_idx, (pmd_pa + PTE_BASE) | 0x03);
    write_entry(&mut buf, pmd_pa, pmd_idx, (pte_pa + PTE_BASE) | 0x03);
    write_entry(&mut buf, pte_pa, pte_idx, (data_pa + PTE_BASE) | 0x03);

    buf[data_pa as usize..data_pa as usize + 8]
        .copy_from_slice(&0x1234_5678_ABCD_EF00u64.to_ne_bytes());

    (buf, pgd_pa, kva, data_pa)
}

#[test]
#[cfg(target_arch = "aarch64")]
fn translate_kva_vmalloc_64k() {
    let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc_64k();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva), false);
    assert_eq!(pa, Some(data_pa), "64KB vmalloc walk should resolve");
    assert_eq!(mem.read_u64(pa.unwrap(), 0), 0x1234_5678_ABCD_EF00);
}

#[test]
#[cfg(target_arch = "aarch64")]
fn translate_kva_vmalloc_64k_with_offset() {
    let (buf, cr3_pa, kva, data_pa) = setup_page_table_vmalloc_64k();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let pa = mem.translate_kva(cr3_pa, Kva(kva + 0x100), false);
    assert_eq!(pa, Some(data_pa + 0x100));
}

#[test]
#[cfg(target_arch = "aarch64")]
fn translate_kva_vmalloc_64k_unmapped_neighbor() {
    let (buf, cr3_pa, kva, _) = setup_page_table_vmalloc_64k();
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let unmapped = kva + (1u64 << 42);
    assert_eq!(mem.translate_kva(cr3_pa, Kva(unmapped), false), None);
}

mod htab_tests;
mod local_storage_tests;
