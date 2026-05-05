use super::*;

// -- iter_htab_entries tests --

use crate::monitor::btf_offsets::HtabOffsets;

/// Simplified htab offsets for synthetic buffer tests.
/// htab_elem_size_base=32 is a test value, not the real kernel size.
fn test_htab_offsets() -> HtabOffsets {
    HtabOffsets {
        htab_buckets: 200,
        htab_n_buckets: 208,
        bucket_size: 16,
        bucket_head: 0,
        hlist_nulls_head_first: 0,
        hlist_nulls_node_next: 0,
        htab_elem_size_base: 32,
    }
}

fn test_htab_map_offsets() -> BpfMapOffsets {
    BpfMapOffsets {
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
        htab_offsets: Some(test_htab_offsets()),
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
    }
}

#[test]
fn iter_htab_entries_non_hash_map_returns_empty() {
    let buf = [0u8; 256];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let offsets = test_htab_map_offsets();
    let map = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test.bss".into(),
        map_type: BPF_MAP_TYPE_ARRAY,
        map_flags: 0,
        key_size: 4,
        value_size: 8,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, 0, &offsets, false), &map);
    assert!(entries.is_empty());
}

#[test]
fn iter_htab_entries_no_htab_offsets_returns_empty() {
    let buf = [0u8; 256];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let mut offsets = test_htab_map_offsets();
    offsets.htab_offsets = None;
    let map = BpfMapInfo {
        map_pa: 0,
        map_kva: 0,
        name: "test".into(),
        map_type: BPF_MAP_TYPE_HASH,
        map_flags: 0,
        key_size: 4,
        value_size: 8,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, 0, &offsets, false), &map);
    assert!(entries.is_empty());
}

/// Build a synthetic hash map in a flat buffer with direct-mapping
/// address translation. All structures are laid out at known PAs;
/// page_offset is chosen so kva = pa + page_offset.
///
/// Layout:
///   PA 0x0000: bpf_htab struct (contains bpf_map + htab fields)
///   PA 0x1000: buckets array (n_buckets * bucket_size)
///   PA 0x2000+: htab_elem entries (elem_size each)
///
/// Each htab_elem has: hlist_nulls_node at offset 0, key at
/// htab_elem_size_base, value at htab_elem_size_base + round_up(key_size, 8).
fn setup_htab_direct(
    key_size: u32,
    value_size: u32,
    entries: &[(&[u8], &[u8])],
    n_buckets: u32,
) -> (Vec<u8>, u64, BpfMapInfo, BpfMapOffsets) {
    let htab = test_htab_offsets();
    let offsets = test_htab_map_offsets();
    let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    // Direct-mapping KVA = PAGE_OFFSET + dram_offset.
    let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };

    let htab_pa: u64 = 0x0000;
    let buckets_pa: u64 = 0x1000;
    let elems_start: u64 = 0x2000;
    let elem_data_size = htab.htab_elem_size_base
        + ((key_size as usize + 7) & !7)
        + ((value_size as usize + 7) & !7);
    let elem_stride = elem_data_size.max(64); // padding for safety

    let buf_size = elems_start as usize + entries.len() * elem_stride + 0x1000;
    let mut buf = vec![0u8; buf_size];

    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // Write bpf_htab fields.
    write_u32(
        &mut buf,
        htab_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_HASH,
    );
    write_u32(&mut buf, htab_pa + offsets.key_size as u64, key_size);
    write_u32(&mut buf, htab_pa + offsets.value_size as u64, value_size);
    write_u64(
        &mut buf,
        htab_pa + htab.htab_buckets as u64,
        pa_to_kva(buckets_pa),
    );
    write_u32(&mut buf, htab_pa + htab.htab_n_buckets as u64, n_buckets);

    // Initialize all bucket heads to nulls marker (bit 0 set = empty).
    for i in 0..n_buckets {
        let bucket_pa = buckets_pa + (i as u64) * (htab.bucket_size as u64);
        write_u64(
            &mut buf,
            bucket_pa + htab.bucket_head as u64 + htab.hlist_nulls_head_first as u64,
            (i as u64) << 1 | 1, // nulls marker with bucket index
        );
    }

    // Place all entries in bucket 0 as a linked list.
    let mut prev_node_pa: Option<u64> = None;
    for (idx, (key, val)) in entries.iter().enumerate().rev() {
        let elem_pa = elems_start + (idx as u64) * (elem_stride as u64);
        let elem_kva = pa_to_kva(elem_pa);

        // Write key at htab_elem_size_base offset.
        let key_off = elem_pa + htab.htab_elem_size_base as u64;
        buf[key_off as usize..key_off as usize + key.len()].copy_from_slice(key);

        // Write value at htab_elem_size_base + round_up(key_size, 8).
        let val_off = elem_pa + htab.htab_elem_size_base as u64 + ((key_size as u64 + 7) & !7);
        buf[val_off as usize..val_off as usize + val.len()].copy_from_slice(val);

        // Set next pointer: points to previous element or nulls marker.
        let next = match prev_node_pa {
            Some(prev_pa) => pa_to_kva(prev_pa), // KVA of previous elem
            None => 1u64,                        // nulls end marker
        };
        write_u64(&mut buf, elem_pa + htab.hlist_nulls_node_next as u64, next);

        prev_node_pa = Some(elem_pa);

        // First element in reverse order becomes the head.
        if idx == 0 {
            // Update bucket 0's head to point to this element.
            write_u64(
                &mut buf,
                buckets_pa + htab.bucket_head as u64 + htab.hlist_nulls_head_first as u64,
                elem_kva,
            );
        }
    }

    // If entries is non-empty, fix the chain: bucket head -> entries[0],
    // entries[0].next -> entries[1], ..., entries[last].next -> nulls.
    // The reverse iteration above already built this correctly:
    // prev_node_pa tracks the previous elem for forward chaining.

    let map = BpfMapInfo {
        map_pa: htab_pa,
        map_kva: pa_to_kva(htab_pa),
        name: "test_hash".into(),
        map_type: BPF_MAP_TYPE_HASH,
        map_flags: 0,
        key_size,
        value_size,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    (buf, page_offset, map, offsets)
}

#[test]
fn iter_htab_entries_empty_map() {
    let (buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[], 4);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert!(entries.is_empty());
}

#[test]
fn iter_htab_entries_single_entry() {
    let key = 42u32.to_ne_bytes();
    let val = 0xDEAD_BEEF_CAFE_1234u64.to_ne_bytes();
    let (buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key);
    assert_eq!(entries[0].1, val);
}

#[test]
fn iter_htab_entries_multiple_entries() {
    let k1 = 1u32.to_ne_bytes();
    let v1 = 100u64.to_ne_bytes();
    let k2 = 2u32.to_ne_bytes();
    let v2 = 200u64.to_ne_bytes();
    let k3 = 3u32.to_ne_bytes();
    let v3 = 300u64.to_ne_bytes();
    let (buf, page_offset, map, offsets) =
        setup_htab_direct(4, 8, &[(&k1, &v1), (&k2, &v2), (&k3, &v3)], 4);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert_eq!(entries.len(), 3);
    // All entries are in bucket 0, chained in order.
    assert_eq!(entries[0].0, k1);
    assert_eq!(entries[0].1, v1);
    assert_eq!(entries[1].0, k2);
    assert_eq!(entries[1].1, v2);
    assert_eq!(entries[2].0, k3);
    assert_eq!(entries[2].1, v3);
}

#[test]
fn iter_htab_entries_zero_buckets() {
    let key = 1u32.to_ne_bytes();
    let val = 1u64.to_ne_bytes();
    let (mut buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
    // Override n_buckets to 0.
    let htab = test_htab_offsets();
    buf[htab.htab_n_buckets..htab.htab_n_buckets + 4].copy_from_slice(&0u32.to_ne_bytes());
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert!(entries.is_empty());
}

#[test]
fn iter_htab_entries_larger_key_and_value() {
    // 8-byte key, 16-byte value.
    let key = 0xAAAA_BBBB_CCCC_DDDDu64.to_ne_bytes();
    let val = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        0x00,
    ];
    let (buf, page_offset, map, offsets) = setup_htab_direct(8, 16, &[(&key, &val)], 2);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key);
    assert_eq!(entries[0].1, val);
}

#[test]
fn iter_htab_entries_multi_bucket() {
    // Entry in bucket 2, buckets 0 and 1 empty. Exercises the
    // bucket stride calculation: buckets_kva + i * bucket_size.
    let htab = test_htab_offsets();
    let offsets = test_htab_map_offsets();
    let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };
    let key_size: u32 = 4;
    let value_size: u32 = 8;

    let htab_pa: u64 = 0x0000;
    let buckets_pa: u64 = 0x1000;
    let elem_pa: u64 = 0x2000;
    let n_buckets: u32 = 4;

    let buf_size = 0x3000;
    let mut buf = vec![0u8; buf_size];

    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // bpf_htab fields.
    write_u32(
        &mut buf,
        htab_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_HASH,
    );
    write_u32(&mut buf, htab_pa + offsets.key_size as u64, key_size);
    write_u32(&mut buf, htab_pa + offsets.value_size as u64, value_size);
    write_u64(
        &mut buf,
        htab_pa + htab.htab_buckets as u64,
        pa_to_kva(buckets_pa),
    );
    write_u32(&mut buf, htab_pa + htab.htab_n_buckets as u64, n_buckets);

    // All buckets get nulls markers (empty).
    for i in 0..n_buckets {
        let bp = buckets_pa + (i as u64) * (htab.bucket_size as u64);
        write_u64(&mut buf, bp, (i as u64) << 1 | 1);
    }

    // Place one entry in bucket 2.
    let bucket2_pa = buckets_pa + 2 * (htab.bucket_size as u64);
    let elem_kva = pa_to_kva(elem_pa);
    write_u64(&mut buf, bucket2_pa, elem_kva); // bucket 2 head -> elem

    // elem next = nulls marker (end).
    write_u64(&mut buf, elem_pa + htab.hlist_nulls_node_next as u64, 1);

    // key at htab_elem_size_base.
    let key_bytes = 99u32.to_ne_bytes();
    let key_off = elem_pa + htab.htab_elem_size_base as u64;
    buf[key_off as usize..key_off as usize + 4].copy_from_slice(&key_bytes);

    // value at htab_elem_size_base + round_up(key_size, 8).
    let val_bytes = 0xBEEF_CAFEu64.to_ne_bytes();
    let val_off = elem_pa + htab.htab_elem_size_base as u64 + ((key_size as u64 + 7) & !7);
    buf[val_off as usize..val_off as usize + 8].copy_from_slice(&val_bytes);

    let map = BpfMapInfo {
        map_pa: htab_pa,
        map_kva: pa_to_kva(htab_pa),
        name: "multi_bucket".into(),
        map_type: BPF_MAP_TYPE_HASH,
        map_flags: 0,
        key_size,
        value_size,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key_bytes);
    assert_eq!(entries[0].1, val_bytes);
}

// -- read_percpu_array_value tests --

/// Build a buffer simulating a percpu array map with `num_cpus` CPUs
/// and `max_entries` entries. Each per-CPU value region is `value_size`
/// bytes. Uses direct-mapping (page_offset) for per-CPU addresses.
///
/// Layout:
///   0x0000..0x1000: page table pages (PGD/PUD/PMD/PTE)
///   0x10000: bpf_array (containing pptrs at array_value offset)
///   0x11000+: per-CPU value regions
///   per_cpu_offsets[cpu] adjusts the percpu base to per-CPU data
///
/// Returns (buffer, cr3_pa, page_offset, map_info, offsets, per_cpu_offsets).
#[cfg(target_arch = "x86_64")]
fn setup_percpu_array(
    num_cpus: u32,
    max_entries: u32,
    value_size: u32,
) -> (Vec<u8>, u64, u64, BpfMapInfo, BpfMapOffsets, Vec<u64>) {
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

    let page_offset: u64 = 0xFFFF_8880_0000_0000;

    // Page table for translating the bpf_array KVA (vmalloc'd).
    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let array_pa: u64 = 0x14000;

    let map_kva: u64 = 0xFFFF_C900_0000_0000;
    let pgd_idx = (map_kva >> 39) & 0x1FF;
    let pud_idx = (map_kva >> 30) & 0x1FF;
    let pmd_idx = (map_kva >> 21) & 0x1FF;
    let pte_idx = (map_kva >> 12) & 0x1FF;

    // Per-CPU data: each CPU gets value_size bytes per entry, at
    // fixed PAs separated by 0x1000 per CPU. The percpu base is
    // a direct-mapped KVA; per_cpu_offsets adjust it per CPU.
    let percpu_base_pa: u64 = 0x20000;
    let percpu_stride: u64 = 0x1000;
    let elem_size = ((value_size as u64 + 7) & !7) * max_entries as u64;

    let total_size = (percpu_base_pa + percpu_stride * num_cpus as u64 + elem_size) as usize;
    let mut buf = vec![0u8; total_size.max(0x30000)];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        if off + 8 <= buf.len() {
            buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
        }
    };

    // Page table: PGD -> PUD -> PMD -> PTE -> array_pa.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pte_pa + pte_idx * 8, (array_pa + PTE_BASE) | 0x63);

    // percpu base KVA (direct-mapped).
    let percpu_base_kva = percpu_base_pa + page_offset;

    // per_cpu_offsets: CPU 0 at percpu_base, CPU 1 at +stride, etc.
    let per_cpu_offsets: Vec<u64> = (0..num_cpus)
        .map(|cpu| cpu as u64 * percpu_stride)
        .collect();

    // Write pptrs[0..max_entries] into the bpf_array at array_pa.
    let pptrs_pa = array_pa + offsets.array_value as u64;
    for entry in 0..max_entries {
        let pptr_value = percpu_base_kva + entry as u64 * ((value_size as u64 + 7) & !7);
        write_u64(&mut buf, pptrs_pa + entry as u64 * 8, pptr_value);
    }

    let info = BpfMapInfo {
        map_pa: array_pa,
        map_kva,
        name: "test_percpu".into(),
        map_type: BPF_MAP_TYPE_PERCPU_ARRAY,
        map_flags: 0,
        key_size: 4,
        value_size,
        max_entries,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    (buf, pgd_pa, page_offset, info, offsets, per_cpu_offsets)
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_basic() {
    let num_cpus = 4u32;
    let value_size = 8u32;
    let (mut buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
        setup_percpu_array(num_cpus, 1, value_size);

    // Write distinct u64 values for each CPU at key 0.
    let percpu_base_pa: u64 = 0x20000;
    let stride: u64 = 0x1000;
    for cpu in 0..num_cpus {
        let pa = percpu_base_pa + cpu as u64 * stride;
        buf[pa as usize..pa as usize + 8]
            .copy_from_slice(&((cpu as u64 + 1) * 0x1111).to_ne_bytes());
    }

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );

    assert_eq!(result.len(), num_cpus as usize);
    for (cpu, entry) in result.iter().enumerate() {
        let bytes = entry.as_ref().expect("CPU value should be Some");
        let val = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(val, (cpu as u64 + 1) * 0x1111);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_key_out_of_bounds() {
    let (buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) = setup_percpu_array(2, 1, 8);
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    // key=1 is out of bounds for max_entries=1.
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        1,
        &per_cpu_offsets,
    );
    assert!(result.is_empty());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_wrong_map_type() {
    let (buf, cr3_pa, page_offset, mut info, offsets, per_cpu_offsets) =
        setup_percpu_array(2, 1, 8);
    info.map_type = BPF_MAP_TYPE_ARRAY;
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );
    assert!(result.is_empty());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_zero_pptr() {
    let (mut buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
        setup_percpu_array(2, 1, 8);

    // Zero out pptrs[0] so the percpu base is 0.
    let pptrs_pa = (0x14000 + offsets.array_value as u64) as usize;
    buf[pptrs_pa..pptrs_pa + 8].copy_from_slice(&0u64.to_ne_bytes());

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );
    assert!(result.is_empty());
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_multiple_entries() {
    let num_cpus = 2u32;
    let value_size = 4u32;
    let max_entries = 3u32;
    let (mut buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) =
        setup_percpu_array(num_cpus, max_entries, value_size);

    // Write distinct u32 values for each CPU at each key.
    let percpu_base_pa: u64 = 0x20000;
    let stride: u64 = 0x1000;
    let elem_size = 8u64; // round_up(4, 8)
    for key in 0..max_entries {
        for cpu in 0..num_cpus {
            let pa = percpu_base_pa + cpu as u64 * stride + key as u64 * elem_size;
            let val: u32 = key * 100 + cpu;
            buf[pa as usize..pa as usize + 4].copy_from_slice(&val.to_ne_bytes());
        }
    }

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

    for key in 0..max_entries {
        let result = read_percpu_array_value(
            &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
            &info,
            key,
            &per_cpu_offsets,
        );
        assert_eq!(result.len(), num_cpus as usize);
        for (cpu, entry) in result.iter().enumerate() {
            let bytes = entry.as_ref().expect("CPU value should be Some");
            let val = u32::from_ne_bytes(bytes[..4].try_into().unwrap());
            assert_eq!(val, key * 100 + cpu as u32);
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_cpu_out_of_guest_memory() {
    let (buf, cr3_pa, page_offset, info, offsets, _) = setup_percpu_array(2, 1, 8);

    // Craft per_cpu_offsets so CPU 1's PA exceeds guest memory size.
    let bad_offset = buf.len() as u64 + 0x10000;
    let per_cpu_offsets = vec![0u64, bad_offset];

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );

    assert_eq!(result.len(), 2);
    assert!(result[0].is_some(), "CPU 0 should be readable");
    assert!(
        result[1].is_none(),
        "CPU 1 should be None (out of guest memory)"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_zero_cpus() {
    // Use setup_percpu_array with num_cpus=0 so the page table and
    // pptrs[0] are valid but per_cpu_offsets is empty. This exercises
    // the per-CPU loop with an empty slice (not the pptr translation
    // failure path).
    let (buf, cr3_pa, page_offset, info, offsets, per_cpu_offsets) = setup_percpu_array(0, 1, 8);
    assert!(per_cpu_offsets.is_empty());

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );
    assert!(result.is_empty(), "zero CPUs should produce empty result");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_mixed_translatable() {
    let num_cpus = 4u32;
    let value_size = 8u32;
    let (mut buf, cr3_pa, page_offset, info, offsets, _) =
        setup_percpu_array(num_cpus, 1, value_size);

    // Write known data at CPU 0 and CPU 2 (valid offsets).
    let percpu_base_pa: u64 = 0x20000;
    let stride: u64 = 0x1000;
    buf[percpu_base_pa as usize..percpu_base_pa as usize + 8]
        .copy_from_slice(&0xAAAAu64.to_ne_bytes());
    let cpu2_pa = percpu_base_pa + 2 * stride;
    buf[cpu2_pa as usize..cpu2_pa as usize + 8].copy_from_slice(&0xCCCCu64.to_ne_bytes());

    // CPU 0 and 2 have valid offsets; CPU 1 and 3 have offsets
    // that produce PAs beyond the buffer.
    let bad = buf.len() as u64 + 0x10000;
    let per_cpu_offsets = vec![0, bad, 2 * stride, bad + stride];

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );

    assert_eq!(result.len(), 4);
    // CPU 0: valid.
    let v0 = result[0].as_ref().expect("CPU 0 should be Some");
    assert_eq!(u64::from_ne_bytes(v0[..8].try_into().unwrap()), 0xAAAA);
    // CPU 1: out of bounds.
    assert!(result[1].is_none(), "CPU 1 should be None");
    // CPU 2: valid.
    let v2 = result[2].as_ref().expect("CPU 2 should be Some");
    assert_eq!(u64::from_ne_bytes(v2[..8].try_into().unwrap()), 0xCCCC);
    // CPU 3: out of bounds.
    assert!(result[3].is_none(), "CPU 3 should be None");
}

/// Pin the out-of-range-CPU aliasing fix: when callers pass a
/// `per_cpu_offsets` array whose tail slots read as zero (because
/// the underlying `__per_cpu_offset[N]` for N >= nr_cpu_ids is
/// BSS-zero in the kernel's static array), `read_percpu_array_value`
/// must return `None` for those slots rather than silently aliasing
/// to whichever CPU happens to live at `percpu_base + 0`.
///
/// Setup: 4-element `per_cpu_offsets` with [non-zero, non-zero,
/// 0, 0]. The first two slots represent real CPUs (with offsets
/// produced by `setup_per_cpu_areas`), the last two simulate the
/// out-of-range tail. Each real CPU has distinct bytes at
/// `percpu_base + offset`. The bytes at `percpu_base + 0` are
/// also distinct (a marker value) so the test can distinguish
/// "returned None correctly" from "returned the aliased CPU 0
/// bytes incorrectly". Only the first two slots should be Some;
/// the last two must be None.
#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_out_of_range_returns_none_not_alias() {
    let num_cpus = 4u32;
    let value_size = 8u32;
    // setup_percpu_array seeds offsets [0, stride, 2*stride, 3*stride].
    // We override per_cpu_offsets below to model the out-of-range
    // tail; the zero-valued tail must NOT alias to whatever lives
    // at `percpu_base + 0`.
    let (mut buf, cr3_pa, page_offset, info, offsets, _) =
        setup_percpu_array(num_cpus, 1, value_size);
    let percpu_base_pa: u64 = 0x20000;
    let stride: u64 = 0x1000;
    // Write a marker at `percpu_base + 0`. If the buggy
    // implementation aliases out-of-range slots to this region,
    // the test will see the marker bytes and fail the None check.
    buf[percpu_base_pa as usize..percpu_base_pa as usize + 8]
        .copy_from_slice(&0xDEAD_BEEFu64.to_ne_bytes());
    // Write distinct values at the real CPUs' regions.
    let cpu1_pa = percpu_base_pa + stride;
    buf[cpu1_pa as usize..cpu1_pa as usize + 8].copy_from_slice(&0x1111u64.to_ne_bytes());

    // Per-CPU offset layout used for this test:
    // - CPU 0 (cpu_index==0): offset 0 — legitimate per the
    //   UP/identity-relocation case; reads the marker at
    //   percpu_base+0.
    // - CPU 1 (cpu_index>0, non-zero offset): real CPU at
    //   percpu_base+stride; reads the 0x1111 value.
    // - CPU 2/3 (cpu_index>0, zero offset): out-of-range tail —
    //   BSS-zero `__per_cpu_offset[N]` for N >= nr_cpu_ids. The
    //   fix returns None here; the buggy implementation would
    //   alias to percpu_base+0 and read the 0xDEAD_BEEF marker.
    let per_cpu_offsets = vec![0u64, stride, 0, 0];

    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, cr3_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );

    assert_eq!(result.len(), 4);
    // CPU 0 (cpu_index==0): zero offset is legitimate; reads
    // the marker bytes at percpu_base+0.
    let v0 = result[0].as_ref().expect("CPU 0 should be Some");
    assert_eq!(u64::from_ne_bytes(v0[..8].try_into().unwrap()), 0xDEAD_BEEF,);
    // CPU 1 (cpu_index>0, non-zero offset): real CPU.
    let v1 = result[1].as_ref().expect("CPU 1 should be Some");
    assert_eq!(u64::from_ne_bytes(v1[..8].try_into().unwrap()), 0x1111);
    // CPU 2 (cpu_index>0, zero offset): out-of-range, must NOT
    // alias to CPU 0's marker bytes.
    assert!(
        result[2].is_none(),
        "CPU 2 (out-of-range, cpu_off==0) must be None, not aliased to CPU 0; got {:?}",
        result[2],
    );
    // CPU 3 (cpu_index>0, zero offset): out-of-range, ditto.
    assert!(
        result[3].is_none(),
        "CPU 3 (out-of-range, cpu_off==0) must be None, not aliased to CPU 0; got {:?}",
        result[3],
    );
}

#[test]
fn read_percpu_array_unmapped_bpf_array() {
    // bpf_array KVA that cannot be translated (no page table,
    // not in direct mapping) — translate_any_kva returns None.
    let buf = vec![0u8; 0x20000];
    // SAFETY: buf is a live local buffer (Vec<u8> or stack array)
    // whose backing storage outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
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

    // map_kva points to an untranslatable address: outside direct
    // mapping range and page table (cr3=0) is all zeros.
    let info = BpfMapInfo {
        map_pa: 0,
        map_kva: 0xFFFF_C900_DEAD_0000,
        name: "test_percpu".into(),
        map_type: BPF_MAP_TYPE_PERCPU_ARRAY,
        map_flags: 0,
        key_size: 4,
        value_size: 8,
        max_entries: 1,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    let per_cpu_offsets = vec![0u64, 0x1000];
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, 0, 0xFFFF_8880_0000_0000, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );
    assert!(
        result.is_empty(),
        "unmapped bpf_array should return empty vec"
    );
}

/// Regression for the per-CPU aliasing fix: when the per-CPU
/// value lives in vmalloc'd percpu memory (outside the direct
/// mapping), the pre-fix `kva_to_pa` math produced an
/// out-of-bounds PA and the CPU read fell through to `None`.
/// Post-fix `translate_any_kva` falls through to a page-table
/// walk and resolves the value.
///
/// Constructs a single-CPU percpu map whose percpu_base KVA is
/// only reachable via page table entries — `kva_to_pa(kva,
/// page_offset)` would yield a PA past the mapped GuestMem
/// region, so the pre-fix path emitted `None`. The buffer wires
/// PGD -> PUD -> PMD -> PTE for the percpu KVA so the new
/// `translate_any_kva` path translates and reads the planted
/// marker bytes.
#[test]
#[cfg(target_arch = "x86_64")]
fn read_percpu_array_kva_via_page_table() {
    // Percpu base lives in vmalloc range (well above PAGE_OFFSET +
    // buf.len(), so direct-mapping math falls off the end of mem).
    let percpu_base_kva: u64 = 0xFFFF_C900_0010_0000;
    let pgd_idx = (percpu_base_kva >> 39) & 0x1FF;
    let pud_idx = (percpu_base_kva >> 30) & 0x1FF;
    let pmd_idx = (percpu_base_kva >> 21) & 0x1FF;
    let pte_idx = (percpu_base_kva >> 12) & 0x1FF;

    // Layout:
    //   0x10000: PGD
    //   0x11000: PUD
    //   0x12000: PMD
    //   0x13000: PTE -> percpu_base_pa
    //   0x14000: bpf_array (pptrs at array_value offset)
    //   0x15000: PTE -> percpu_base_pa (planted percpu value)
    let pgd_pa: u64 = 0x10000;
    let pud_pa: u64 = 0x11000;
    let pmd_pa: u64 = 0x12000;
    let pte_pa: u64 = 0x13000;
    let array_pa: u64 = 0x14000;
    let percpu_base_pa: u64 = 0x15000;

    // bpf_array KVA still uses direct mapping for simplicity.
    let map_kva: u64 = 0xFFFF_8880_0001_4000; // PAGE_OFFSET + 0x14000.

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

    let size = 0x16000;
    let mut buf = vec![0u8; size];

    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // Page table: PGD -> PUD -> PMD -> PTE -> percpu_base_pa.
    write_u64(&mut buf, pgd_pa + pgd_idx * 8, (pud_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pud_pa + pud_idx * 8, (pmd_pa + PTE_BASE) | 0x63);
    write_u64(&mut buf, pmd_pa + pmd_idx * 8, (pte_pa + PTE_BASE) | 0x63);
    write_u64(
        &mut buf,
        pte_pa + pte_idx * 8,
        (percpu_base_pa + PTE_BASE) | 0x63,
    );

    // pptrs[0] at array_value offset = percpu_base_kva.
    let pptrs_pa = array_pa + offsets.array_value as u64;
    write_u64(&mut buf, pptrs_pa, percpu_base_kva);

    // Plant a marker at the percpu_base_pa for CPU 0 (cpu_off=0).
    let marker: u64 = 0xDEAD_BEEF_F00D_CAFE;
    write_u64(&mut buf, percpu_base_pa, marker);

    let info = BpfMapInfo {
        map_pa: array_pa,
        map_kva,
        name: "vmalloc_percpu".into(),
        map_type: BPF_MAP_TYPE_PERCPU_ARRAY,
        map_flags: 0,
        key_size: 4,
        value_size: 8,
        max_entries: 1,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    // Direct-mapping math for the percpu KVA would yield
    // percpu_base_kva - page_offset = 0x4900_0010_0000 — well
    // past the 0x16000 buffer end. The pre-fix path read this
    // out-of-bounds PA and emitted `None`.
    let page_offset: u64 = 0xFFFF_8880_0000_0000;

    // SAFETY: buf is a live local buffer whose backing storage
    // outlives the GuestMem use.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let per_cpu_offsets = vec![0u64];
    let result = read_percpu_array_value(
        &lookup_ctx(&mem, pgd_pa, page_offset, &offsets, false),
        &info,
        0,
        &per_cpu_offsets,
    );

    assert_eq!(result.len(), 1);
    let bytes = result[0]
        .as_ref()
        .expect("post-fix: vmalloc percpu KVA must resolve via page-table walk");
    assert_eq!(u64::from_ne_bytes(bytes[..8].try_into().unwrap()), marker);
}

/// LRU_HASH shares HASH's htab_elem layout
/// (`kernel/bpf/hashtab.c::htab_elem_value`), so the same
/// walker must produce the same entries when the map_type
/// switches from HASH to LRU_HASH. This is the regression
/// pin for the type-set widening — if a future refactor
/// re-tightens `iter_htab_entries` to HASH-only, this fails.
#[test]
fn iter_htab_entries_accepts_lru_hash() {
    let key = 7u32.to_ne_bytes();
    let val = 0x42_u64.to_ne_bytes();
    let (buf, page_offset, mut map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
    map.map_type = BPF_MAP_TYPE_LRU_HASH;
    // SAFETY: buf is a live local buffer whose storage outlives mem.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_htab_entries(&lookup_ctx(&mem, 0, page_offset, &offsets, false), &map);
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key);
    assert_eq!(entries[0].1, val);
}

/// `iter_percpu_htab_entries` must return empty for a plain
/// HASH map — the value position in PERCPU_HASH is a percpu
/// pointer, so calling the percpu walker on a HASH would
/// dereference inline value bytes as a kernel KVA pointer.
#[test]
fn iter_percpu_htab_entries_rejects_plain_hash() {
    let key = 1u32.to_ne_bytes();
    let val = 0u64.to_ne_bytes();
    let (buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
    // SAFETY: buf is a live local buffer whose storage outlives mem.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_percpu_htab_entries(
        &lookup_ctx(&mem, 0, page_offset, &offsets, false),
        &map,
        &[0u64],
    );
    assert!(entries.is_empty());
}

// -- iter_percpu_htab_entries happy-path tests --------------------
//
// PERCPU_HASH stores a `void __percpu *` pptr at the value position
// of each htab_elem. Per-CPU value bytes are at
// `pptr + per_cpu_offsets[cpu]`. Build a synthetic htab over direct
// mapping plus per-CPU value regions so the walker exercises the
// full chain-of-custody: bucket → elem → pptr → per-CPU value.

/// Build a PERCPU_HASH scene: a one-bucket htab_elem chain with N
/// (key, pptr) entries. Each pptr addresses a percpu base region;
/// per_cpu_offsets[cpu] add to that base to yield each CPU's value.
///
/// Returns (buf, page_offset, map, offsets, per_cpu_offsets, value_size).
fn setup_percpu_htab_direct(
    key_size: u32,
    value_size: u32,
    num_cpus: u32,
    entries: &[(Vec<u8>, Vec<Vec<u8>>)],
) -> (Vec<u8>, u64, BpfMapInfo, BpfMapOffsets, Vec<u64>) {
    let htab = test_htab_offsets();
    let offsets = test_htab_map_offsets();
    let page_offset: u64 = crate::monitor::symbols::DEFAULT_PAGE_OFFSET;
    let pa_to_kva = |pa: u64| -> u64 { page_offset.wrapping_add(pa) };

    let htab_pa: u64 = 0x0000;
    let buckets_pa: u64 = 0x1000;
    let elems_start: u64 = 0x2000;
    // Per-CPU base regions live further into the buffer; each CPU
    // gets its own page so the per_cpu_offset[cpu] math is
    // straightforward.
    let percpu_start: u64 = 0x10_0000;
    let percpu_stride: u64 = 0x1000;

    let elem_data_size = htab.htab_elem_size_base + ((key_size as usize + 7) & !7) + 8; // value position holds a u64 pptr
    let elem_stride = elem_data_size.max(64);

    let total_size = (percpu_start
        + percpu_stride * num_cpus as u64
        + (entries.len() as u64) * (value_size as u64 + 8) * num_cpus as u64
        + 0x1000) as usize;
    let mut buf = vec![0u8; total_size.max(0x40_0000)];

    let write_u32 = |buf: &mut Vec<u8>, pa: u64, val: u32| {
        let off = pa as usize;
        buf[off..off + 4].copy_from_slice(&val.to_ne_bytes());
    };
    let write_u64 = |buf: &mut Vec<u8>, pa: u64, val: u64| {
        let off = pa as usize;
        buf[off..off + 8].copy_from_slice(&val.to_ne_bytes());
    };

    // bpf_htab fields.
    write_u32(
        &mut buf,
        htab_pa + offsets.map_type as u64,
        BPF_MAP_TYPE_PERCPU_HASH,
    );
    write_u32(&mut buf, htab_pa + offsets.key_size as u64, key_size);
    write_u32(&mut buf, htab_pa + offsets.value_size as u64, value_size);
    write_u64(
        &mut buf,
        htab_pa + htab.htab_buckets as u64,
        pa_to_kva(buckets_pa),
    );
    write_u32(&mut buf, htab_pa + htab.htab_n_buckets as u64, 4);

    // Initialize all bucket heads to nulls marker.
    for i in 0..4u32 {
        let bucket_pa = buckets_pa + (i as u64) * (htab.bucket_size as u64);
        write_u64(
            &mut buf,
            bucket_pa + htab.bucket_head as u64 + htab.hlist_nulls_head_first as u64,
            (i as u64) << 1 | 1,
        );
    }

    // Lay out elems in bucket 0, chained back-to-back.
    let mut prev_node_pa: Option<u64> = None;
    let value_off_in_elem = htab.htab_elem_size_base as u64 + ((key_size as u64 + 7) & !7);

    for (idx, (key, _per_cpu_values)) in entries.iter().enumerate().rev() {
        let elem_pa = elems_start + (idx as u64) * (elem_stride as u64);
        let elem_kva = pa_to_kva(elem_pa);

        // Key bytes at htab_elem_size_base.
        let key_off = elem_pa + htab.htab_elem_size_base as u64;
        for (i, b) in key.iter().enumerate() {
            buf[key_off as usize + i] = *b;
        }

        // Per-CPU pptr at value position. Each elem gets its own
        // percpu base region so per-CPU values don't collide.
        let pptr_pa = percpu_start + (idx as u64) * 0x10_0000;
        let pptr_kva = pa_to_kva(pptr_pa);
        write_u64(&mut buf, elem_pa + value_off_in_elem, pptr_kva);

        // Chain link.
        let next = match prev_node_pa {
            Some(prev_pa) => pa_to_kva(prev_pa),
            None => 1u64,
        };
        write_u64(&mut buf, elem_pa + htab.hlist_nulls_node_next as u64, next);
        prev_node_pa = Some(elem_pa);

        // Bucket head -> first elem.
        if idx == 0 {
            write_u64(
                &mut buf,
                buckets_pa + htab.bucket_head as u64 + htab.hlist_nulls_head_first as u64,
                elem_kva,
            );
        }
    }

    // Per-cpu offsets: cpu N gets +(N * percpu_stride). Write each
    // CPU's value bytes at pptr_pa + cpu_off.
    let per_cpu_offsets: Vec<u64> = (0..num_cpus)
        .map(|cpu| cpu as u64 * percpu_stride)
        .collect();

    for (idx, (_key, per_cpu_values)) in entries.iter().enumerate() {
        let pptr_pa = percpu_start + (idx as u64) * 0x10_0000;
        for (cpu, value) in per_cpu_values.iter().enumerate() {
            assert_eq!(value.len(), value_size as usize);
            let cpu_pa = pptr_pa + per_cpu_offsets[cpu];
            for (i, b) in value.iter().enumerate() {
                buf[cpu_pa as usize + i] = *b;
            }
        }
    }

    let map = BpfMapInfo {
        map_pa: htab_pa,
        map_kva: pa_to_kva(htab_pa),
        name: "test_percpu_hash".into(),
        map_type: BPF_MAP_TYPE_PERCPU_HASH,
        map_flags: 0,
        key_size,
        value_size,
        max_entries: 0,
        value_kva: None,
        btf_kva: 0,
        btf_value_type_id: 0,
        btf_vmlinux_value_type_id: 0,
        btf_key_type_id: 0,
    };

    (buf, page_offset, map, offsets, per_cpu_offsets)
}

/// Two CPUs, one entry: each CPU reports its distinct value. Pins
/// the per-CPU value-region read path through the pptr indirection.
#[test]
fn iter_percpu_htab_entries_basic_two_cpus() {
    let key = 0xAAu32.to_ne_bytes();
    let cpu0_val = 0x1111_2222_3333_4444u64.to_ne_bytes();
    let cpu1_val = 0x5555_6666_7777_8888u64.to_ne_bytes();
    let (buf, page_offset, map, offsets, per_cpu_offsets) = setup_percpu_htab_direct(
        4,
        8,
        2,
        &[(key.to_vec(), vec![cpu0_val.to_vec(), cpu1_val.to_vec()])],
    );
    // SAFETY: buf is a live local buffer whose storage outlives mem.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_percpu_htab_entries(
        &lookup_ctx(&mem, 0, page_offset, &offsets, false),
        &map,
        &per_cpu_offsets,
    );
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key);
    assert_eq!(entries[0].1.len(), 2);
    assert_eq!(entries[0].1[0].as_ref().unwrap(), &cpu0_val);
    assert_eq!(entries[0].1[1].as_ref().unwrap(), &cpu1_val);
}

/// Zero pptr at value position: walker yields the key with an
/// empty per-CPU vector. The empty vec signals "pptr was NULL"
/// without dropping the entry from the report.
#[test]
fn iter_percpu_htab_entries_zero_pptr_returns_empty_per_cpu() {
    let key = 0xAAu32.to_ne_bytes();
    let val = 0u64.to_ne_bytes();
    let (mut buf, page_offset, map, offsets) = setup_htab_direct(4, 8, &[(&key, &val)], 4);
    // Promote the map to PERCPU_HASH so the percpu walker accepts it.
    let mut map = map;
    map.map_type = BPF_MAP_TYPE_PERCPU_HASH;
    // Overwrite map_type in the buffer too (htab struct read).
    let htab = test_htab_offsets();
    let map_type_off = offsets.map_type;
    buf[map_type_off..map_type_off + 4].copy_from_slice(&BPF_MAP_TYPE_PERCPU_HASH.to_ne_bytes());
    // val at value position is already zero (built with val=0u64).
    let _ = htab;

    // SAFETY: buf is a live local buffer whose storage outlives mem.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_percpu_htab_entries(
        &lookup_ctx(&mem, 0, page_offset, &offsets, false),
        &map,
        &[0u64, 0x1000u64],
    );
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key);
    // Per-cpu vec is empty (NULL pptr signals "no per-cpu data").
    assert!(entries[0].1.is_empty());
}

/// Out-of-range CPU guard: when `per_cpu_offsets[cpu_index]` is 0
/// AND `cpu_index > 0`, the walker treats the slot as unmapped
/// (None) rather than aliasing CPU 0. Mirrors the matching guard
/// in `read_percpu_array_value`.
#[test]
fn iter_percpu_htab_entries_aliasing_guard() {
    let key = 0xAAu32.to_ne_bytes();
    let cpu0_val = 0x1111_2222_3333_4444u64.to_ne_bytes();
    let cpu1_val = 0xDEAD_BEEF_DEAD_BEEFu64.to_ne_bytes();
    let (buf, page_offset, map, offsets, _per_cpu_offsets) = setup_percpu_htab_direct(
        4,
        8,
        2,
        &[(key.to_vec(), vec![cpu0_val.to_vec(), cpu1_val.to_vec()])],
    );
    // SAFETY: buf is a live local buffer whose storage outlives mem.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    // Pass per_cpu_offsets = [0, 0] — second CPU's offset==0 with
    // index>0 must read None, not alias CPU 0's value.
    let aliasing_offsets = vec![0u64, 0u64];
    let entries = iter_percpu_htab_entries(
        &lookup_ctx(&mem, 0, page_offset, &offsets, false),
        &map,
        &aliasing_offsets,
    );
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].1.len(), 2);
    // CPU 0: cpu_off=0 with index 0 is valid (the percpu base IS
    // cpu 0's region). The reader translates pptr+0 → pptr.
    assert_eq!(entries[0].1[0].as_ref().unwrap(), &cpu0_val);
    // CPU 1: cpu_off=0 with index 1 hits the aliasing guard.
    assert!(
        entries[0].1[1].is_none(),
        "cpu_off=0 with cpu_index>0 must yield None to avoid aliasing CPU 0"
    );
}

/// LRU_PERCPU_HASH variant: same shape as PERCPU_HASH, walker
/// accepts both map types.
#[test]
fn iter_percpu_htab_entries_lru_variant() {
    let key = 0xAAu32.to_ne_bytes();
    let cpu0_val = 0x1234u64.to_ne_bytes();
    let (buf, page_offset, mut map, offsets, per_cpu_offsets) =
        setup_percpu_htab_direct(4, 8, 1, &[(key.to_vec(), vec![cpu0_val.to_vec()])]);
    // Promote to LRU variant; same value layout.
    map.map_type = BPF_MAP_TYPE_LRU_PERCPU_HASH;
    // SAFETY: buf is a live local buffer whose storage outlives mem.
    let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
    let entries = iter_percpu_htab_entries(
        &lookup_ctx(&mem, 0, page_offset, &offsets, false),
        &map,
        &per_cpu_offsets,
    );
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, key);
    assert_eq!(entries[0].1[0].as_ref().unwrap(), &cpu0_val);
}
