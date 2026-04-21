//! Compute the minimum guest memory required to boot, extract the
//! initramfs, and run the post-boot test workload.
//!
//! Used by the deferred-memory path in [`KtstrVm`](super::KtstrVm) to
//! size guest memory from observed initramfs sizes instead of a static
//! caller estimate.

use anyhow::{Context, Result};
use std::path::Path;

/// Parameters for computing minimum guest memory.
pub(crate) struct MemoryBudget {
    /// Uncompressed initramfs size (base + suffix cpio) in bytes.
    pub uncompressed_initramfs_bytes: u64,
    /// LZ4-compressed initrd size in bytes. The compressed initrd
    /// is memblock-reserved in guest physical memory from load until
    /// free_initrd_mem() releases it after extraction.
    pub compressed_initrd_bytes: u64,
    /// Kernel `init_size` from bzImage setup_header (offset 0x260).
    /// The kernel's declared contiguous memory requirement during
    /// boot decompression. Includes compressed payload, decompressed
    /// kernel, and decompression workspace. Overestimates resident
    /// kernel (init sections and workspace are freed post-boot),
    /// absorbing percpu and misc boot allocations.
    pub kernel_init_size: u64,
    /// SHM region carved from the top of guest memory (E820 gap on
    /// x86_64, FDT /reserved-memory and /memreserve/ on aarch64).
    pub shm_bytes: u64,
}

/// Read the kernel's declared memory footprint from the image file.
///
/// x86_64 bzImage: reads `init_size` from setup_header at file offset
/// 0x260 (setup_header starts at 0x1F1, `init_size` is at byte 111
/// within it). This is the kernel's declared contiguous memory
/// requirement during boot decompression.
///
/// aarch64 Image: reads `image_size` from the arm64 image header at
/// file offset 16 (after code0 + code1 + text_offset). For gzip-
/// compressed vmlinuz, falls back to file size * 4 as a conservative
/// estimate of the decompressed Image size.
pub(crate) fn read_kernel_init_size(kernel_path: &Path) -> Result<u64> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(kernel_path)
        .with_context(|| format!("open kernel for init_size: {}", kernel_path.display()))?;

    #[cfg(target_arch = "x86_64")]
    {
        // setup_header starts at 0x1F1, init_size at offset 111.
        f.seek(SeekFrom::Start(0x260))
            .context("seek to init_size in bzImage")?;
        let mut buf = [0u8; 4];
        f.read_exact(&mut buf)
            .context("read init_size from bzImage")?;
        Ok(u32::from_le_bytes(buf) as u64)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Check for gzip magic (0x1f 0x8b).
        let mut magic = [0u8; 2];
        f.read_exact(&mut magic).context("read kernel magic")?;
        if magic == [0x1f, 0x8b] {
            // Compressed vmlinuz — decompress header to read image_size.
            f.seek(SeekFrom::Start(0))
                .context("seek vmlinuz to start")?;
            let mut decoder = flate2::read::GzDecoder::new(&mut f);
            let mut header = [0u8; 24];
            decoder
                .read_exact(&mut header)
                .context("decompress arm64 vmlinuz header for image_size")?;
            return Ok(u64::from_le_bytes(header[16..24].try_into().unwrap()));
        }
        // Raw PE Image: image_size is a little-endian u64 at offset 16.
        f.seek(SeekFrom::Start(16))
            .context("seek to image_size in arm64 Image")?;
        let mut buf = [0u8; 8];
        f.read_exact(&mut buf)
            .context("read image_size from arm64 Image")?;
        Ok(u64::from_le_bytes(buf))
    }
}

/// Minimum guest memory (in MB) needed to boot, extract the initramfs,
/// and run the test workload.
///
/// ```text
/// total = computed_boot_requirement + WORKLOAD_MB + shm
/// ```
///
/// ## Computed boot requirement
///
/// Every term is derived from values known at allocation time. The model
/// follows the kernel's boot memory layout.
///
/// **memblock-reserved regions** (excluded from `totalram_pages`):
///
/// - `kernel_init_size`: bzImage setup_header `init_size` field (offset
///   0x260) — the kernel's declared contiguous memory requirement during
///   boot decompression. Includes compressed payload, decompressed
///   vmlinux, and decompression workspace. Overestimates resident kernel
///   since init sections (`free_initmem`, `init/main.c`) and the
///   decompression workspace are freed post-boot. The slack absorbs
///   percpu allocations (`pcpu_embed_first_chunk` in `mm/percpu.c`
///   reserves `static_size + reserved_size + dyn_size` per CPU via
///   memblock, ~220KB/CPU with ktstr's kconfig which disables LOCKDEP)
///   and misc boot allocations (page tables, slab bootstrap, hash tables).
///
/// - `compressed_initrd`: memblock-reserved by `reserve_initrd_mem()`
///   (`init/initramfs.c:642`: `memblock_reserve(start, size)`) until
///   `free_initrd_mem()` after `unpack_to_rootfs` completes.
///
/// - struct page array: `P / 64` bytes. Each 4KB page requires a
///   `struct page` descriptor. On x86_64: base size = 56 bytes
///   (flags:8 + 5-word union:40 + _mapcount:4 + _refcount:4), rounded
///   to 64 by `CONFIG_HAVE_ALIGNED_STRUCT_PAGE` (16-byte alignment,
///   `include/linux/mm_types.h`). Valid for x86_64 without `CONFIG_KMSAN`.
///
/// **tmpfs constraint** (the binding limit for initramfs extraction):
///
/// The rootfs tmpfs is mounted by `init_mount_tree()` (`fs/namespace.c`)
/// via `vfs_kern_mount(&rootfs_fs_type, 0, ...)` — flags=0, NOT
/// `SB_KERNMOUNT`. `alloc_super` (`fs/super.c`) sets `s->s_flags = flags`,
/// so `SB_KERNMOUNT` is not set. In `shmem_fill_super` (`mm/shmem.c`),
/// the `!(sb->s_flags & SB_KERNMOUNT)` branch runs, and since no
/// `size=` mount option was parsed (`SHMEM_SEEN_BLOCKS` unset), it
/// falls through to `ctx->blocks = shmem_default_max_blocks()` =
/// `totalram_pages() / 2` (`mm/shmem.c:146`).
///
/// `initramfs_options=size=90%` on the cmdline is consumed by
/// `init_mount_tree()` (`fs/namespace.c`) when mounting the rootfs
/// tmpfs. This raises the tmpfs block limit from 50% to 90% of
/// `totalram_pages`, preventing ENOSPC on large initramfs payloads.
///
/// Note: `rootflags=size=90%` would set `root_mount_data`
/// (`init/do_mounts.c:109`), consumed only by `do_mount_root()` via
/// `prepare_namespace()`. With `rdinit=`, `kernel_init_freeable`
/// (`init/main.c`) skips `prepare_namespace()` when `init_eaccess`
/// succeeds, so `rootflags=` is never applied to the rootfs.
///
/// The `SB_KERNMOUNT` (unlimited) tmpfs is the separate `shm_mnt`
/// created by `shmem_init()` via `kern_mount()` — used for anonymous
/// shared memory (`shmem_file_setup`), not the rootfs.
///
/// With `initramfs_options=size=90%`, the tmpfs limit is 90% of
/// `totalram_pages` (not the default 50%):
///
/// ```text
/// totalram_pages(P) = (P - init_size - compressed - P/64) / 4096
/// tmpfs_max_pages = totalram_pages * 9 / 10
/// constraint: tmpfs_max_pages >= uncompressed / 4096
///
/// Solving for P:
/// (P - init_size - compressed - P/64) * 9/10 >= uncompressed
/// P * 63/64 >= uncompressed * 10/9 + init_size + compressed
/// P >= (uncompressed * 10/9 + init_size + compressed) * 64/63
/// ```
///
/// In practice, `ceil(uncompressed * 10/9)` is used to ensure
/// integer rounding does not underallocate.
///
/// ## Workload budget
///
/// 256 MB for scheduler execution, test scenarios, and runtime
/// allocations (cgroup memory, BPF maps, process stacks, slab caches).
/// This is a deliberate budget for post-boot workload, not a guess at
/// kernel overhead.
///
/// ## SHM region
///
/// Carved from the top of guest physical memory. Not part of usable
/// RAM (E820 gap on x86_64, FDT /reserved-memory and /memreserve/ on aarch64).
///
/// ```text
/// total = boot_requirement + 256 + shm
/// ```
///
/// Workload budget (MB): scheduler execution, test scenarios, cgroup
/// memory, BPF maps, and runtime allocations.
const WORKLOAD_MB: u64 = 256;

pub(crate) fn initramfs_min_memory_mb(budget: &MemoryBudget) -> u32 {
    let ceil_mb = |bytes: u64| -> u64 { (bytes + (1 << 20) - 1) >> 20 };

    let init_size_mb = ceil_mb(budget.kernel_init_size);
    let compressed_mb = ceil_mb(budget.compressed_initrd_bytes);
    let shm_mb = ceil_mb(budget.shm_bytes);
    let uncompressed_mb = ceil_mb(budget.uncompressed_initramfs_bytes);

    // Boot requirement: initramfs_options=size=90% sets the rootfs
    // tmpfs limit to 90% of totalram_pages.
    //
    // Constraint: totalram_pages * 9/10 >= uncompressed_pages.
    // totalram_pages = (P - reserved) / PAGE_SIZE.
    // reserved = init_size + compressed + struct_page(P).
    // struct_page(P) = P/64.
    //
    // Solving:
    //   (P - init_size - compressed - P/64) * 9/10 >= uncompressed
    //   P * 63/64 >= uncompressed * 10/9 + init_size + compressed
    //   P >= (ceil(uncompressed * 10/9) + init_size + compressed) * 64/63
    let uncompressed_scaled = (uncompressed_mb * 10).div_ceil(9);
    let content_mb = uncompressed_scaled + init_size_mb + compressed_mb;

    // struct page overhead: P/64 is part of reserved, creating a
    // circular dependency. Solve: P = content * 64/63.
    let boot_mb = (content_mb * 64).div_ceil(63);

    // total = computed boot requirement + workload budget + SHM gap.
    (boot_mb + WORKLOAD_MB + shm_mb) as u32
}
