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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the workload-budget constant. Bumping the value
    /// (`WORKLOAD_MB`) changes the floor for every deferred-memory
    /// VM boot; this test fails any change so the bump goes through
    /// review rather than slipping in unnoticed.
    #[test]
    fn workload_mb_is_256() {
        assert_eq!(WORKLOAD_MB, 256);
    }

    /// All-zero inputs collapse to just the workload budget — no
    /// kernel, no initramfs, no shm reservation. Pins the lower
    /// bound the deferred-memory path always allocates.
    #[test]
    fn initramfs_min_memory_mb_zeros_returns_workload_budget() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 0,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            shm_bytes: 0,
        };
        assert_eq!(initramfs_min_memory_mb(&budget), WORKLOAD_MB as u32);
    }

    /// `shm_bytes` is added linearly to the total. Pin the contract
    /// by varying only the SHM input — the delta in the result must
    /// exactly equal the SHM contribution rounded up to MiB.
    #[test]
    fn initramfs_min_memory_mb_shm_contribution_linear() {
        let zero = MemoryBudget {
            uncompressed_initramfs_bytes: 0,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            shm_bytes: 0,
        };
        let with_shm = MemoryBudget {
            shm_bytes: 16 * (1 << 20), // 16 MiB exactly
            ..zero_budget()
        };
        let base = initramfs_min_memory_mb(&zero);
        let shifted = initramfs_min_memory_mb(&with_shm);
        assert_eq!(shifted, base + 16);
    }

    /// `kernel_init_size` and `compressed_initrd_bytes` flow into
    /// `content_mb` additively, then through the `*64/63` struct-page
    /// circular-dependency factor. Verify the math against a
    /// hand-computed reference. Inputs:
    ///   uncompressed=10 MiB, init_size=5 MiB, compressed=2 MiB,
    ///   shm=8 MiB.
    /// Hand trace per `initramfs_min_memory_mb`:
    ///   uncompressed_scaled = ceil(10*10/9) = ceil(11.111) = 12
    ///   content_mb         = 12 + 5 + 2 = 19
    ///   boot_mb            = ceil(19*64/63) = ceil(19.301) = 20
    ///   total              = 20 + 256 (WORKLOAD_MB) + 8 = 284
    #[test]
    fn initramfs_min_memory_mb_known_input() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 10 * (1 << 20),
            compressed_initrd_bytes: 2 * (1 << 20),
            kernel_init_size: 5 * (1 << 20),
            shm_bytes: 8 * (1 << 20),
        };
        assert_eq!(initramfs_min_memory_mb(&budget), 284);
    }

    /// Sub-MiB inputs round up to 1 MiB before participating in the
    /// math. A 1-byte initramfs (degenerate but reachable when test
    /// fixtures construct empty payloads) must not silently round
    /// down to zero and bypass the tmpfs-90% safety factor. With
    /// uncompressed=1 byte, init=0, compressed=0, shm=0:
    ///   uncompressed_scaled = ceil(1*10/9) = 2
    ///   content_mb         = 2 + 0 + 0 = 2
    ///   boot_mb            = ceil(2*64/63) = ceil(2.031) = 3
    ///   total              = 3 + 256 + 0 = 259
    #[test]
    fn initramfs_min_memory_mb_subbyte_uncompressed_rounds_up() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 1,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            shm_bytes: 0,
        };
        assert_eq!(initramfs_min_memory_mb(&budget), 259);
    }

    /// Larger realistic-shape inputs: uncompressed=200 MiB,
    /// compressed=50 MiB, init_size=30 MiB, shm=16 MiB.
    /// Verifies the math holds at integration-realistic scales (the
    /// production callers in vmm/mod.rs feed values of this order).
    /// Trace:
    ///   uncompressed_scaled = ceil(200*10/9) = ceil(222.222) = 223
    ///   content_mb         = 223 + 30 + 50 = 303
    ///   boot_mb            = ceil(303*64/63) = ceil(307.809) = 308
    ///   total              = 308 + 256 + 16 = 580
    #[test]
    fn initramfs_min_memory_mb_larger_input() {
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: 200 * (1 << 20),
            compressed_initrd_bytes: 50 * (1 << 20),
            kernel_init_size: 30 * (1 << 20),
            shm_bytes: 16 * (1 << 20),
        };
        assert_eq!(initramfs_min_memory_mb(&budget), 580);
    }

    /// `read_kernel_init_size` on x86_64 reads 4 little-endian bytes
    /// at file offset 0x260. Construct a tempfile padded to that
    /// offset with a known init_size value and assert the function
    /// returns it as u64. Pins the exact byte-offset and width
    /// against a future drift in the bzImage setup_header layout.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn read_kernel_init_size_x86_64_reads_offset_0x260() {
        use std::io::Write;

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        // Pad up to 0x260 with zeros, then write 4 bytes of init_size.
        let pad = vec![0u8; 0x260];
        f.write_all(&pad).expect("write pad");
        // Distinct value, large enough that wrong-offset reads would
        // yield zero (the surrounding pad).
        let init_size: u32 = 0x1234_5678;
        f.write_all(&init_size.to_le_bytes()).expect("write init_size");
        f.flush().expect("flush");

        let got = read_kernel_init_size(f.path()).expect("read init_size");
        assert_eq!(got, init_size as u64);
    }

    /// Reading a file shorter than 0x264 bytes (the high end of the
    /// init_size field on x86_64) must surface an error rather than
    /// silently returning 0. Pin the failure shape so a future
    /// "graceful-fallback" refactor that swallows truncated-bzImage
    /// errors can't slip past review.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn read_kernel_init_size_x86_64_short_file_errors() {
        use std::io::Write;

        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        // Only 0x100 bytes — well short of the 0x264 needed.
        let truncated = vec![0u8; 0x100];
        f.write_all(&truncated).expect("write truncated");
        f.flush().expect("flush");

        let result = read_kernel_init_size(f.path());
        assert!(
            result.is_err(),
            "truncated file must fail; got: {result:?}",
        );
    }

    /// Helper: an all-zero MemoryBudget for spread-syntax in tests.
    fn zero_budget() -> MemoryBudget {
        MemoryBudget {
            uncompressed_initramfs_bytes: 0,
            compressed_initrd_bytes: 0,
            kernel_init_size: 0,
            shm_bytes: 0,
        }
    }
}
