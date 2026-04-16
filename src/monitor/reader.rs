//! Guest physical memory access and monitor sampling loop.
//!
//! [`GuestMem`] wraps a host pointer to the start of guest DRAM and
//! provides bounds-checked volatile reads and writes for scalar types;
//! `read_bytes` uses `copy_nonoverlapping` for bulk copies. It also implements
//! 4-level and 5-level x86-64 page table walks and 3-level aarch64 walks
//! (64KB granule) for vmalloc'd addresses.
//!
//! The monitor loop (`monitor_loop`) periodically reads per-CPU
//! runqueue state from guest memory and collects `MonitorSample`s.

use super::btf_offsets::{
    CPU_MAX_IDLE_TYPES, KernelOffsets, SchedDomainOffsets, SchedDomainStatsOffsets,
    SchedstatOffsets, ScxEventOffsets,
};
use super::{
    CpuSnapshot, MonitorSample, RqSchedstat, SchedDomainSnapshot, SchedDomainStats,
    ScxEventCounters,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Host pointer to the start of guest DRAM. Offsets passed to read/write
/// methods are DRAM-relative (x86_64: GPA 0, aarch64: GPA DRAM_START).
///
/// SAFETY: The pointer is valid for the lifetime of the KVM VM (GuestMemoryMmap
/// owns the mmap and outlives all threads).
pub struct GuestMem {
    base: *mut u8,
    size: u64,
}

unsafe impl Send for GuestMem {}
unsafe impl Sync for GuestMem {}

impl GuestMem {
    pub fn new(base: *mut u8, size: u64) -> Self {
        Self { base, size }
    }

    /// Raw pointer to the start of guest DRAM.
    pub fn base_ptr(&self) -> *const u8 {
        self.base
    }

    /// Read a u32 at DRAM offset `pa + offset`.
    pub fn read_u32(&self, pa: u64, offset: usize) -> u32 {
        let addr = pa + offset as u64;
        if addr + 4 > self.size {
            return 0;
        }
        unsafe { std::ptr::read_volatile(self.base.add(addr as usize) as *const u32) }
    }

    /// Read a u64 at DRAM offset `pa + offset`.
    pub fn read_u64(&self, pa: u64, offset: usize) -> u64 {
        let addr = pa + offset as u64;
        if addr + 8 > self.size {
            return 0;
        }
        unsafe { std::ptr::read_volatile(self.base.add(addr as usize) as *const u64) }
    }

    /// Read an i64 at DRAM offset `pa + offset`.
    pub fn read_i64(&self, pa: u64, offset: usize) -> i64 {
        self.read_u64(pa, offset) as i64
    }

    /// Write a u8 at DRAM offset `pa + offset`.
    pub fn write_u8(&self, pa: u64, offset: usize, val: u8) {
        let addr = pa + offset as u64;
        if addr + 1 > self.size {
            return;
        }
        unsafe { std::ptr::write_volatile(self.base.add(addr as usize), val) }
    }

    /// Write a u64 at DRAM offset `pa + offset`.
    pub fn write_u64(&self, pa: u64, offset: usize, val: u64) {
        let addr = pa + offset as u64;
        if addr + 8 > self.size {
            return;
        }
        unsafe { std::ptr::write_volatile(self.base.add(addr as usize) as *mut u64, val) }
    }

    /// Read a u8 at DRAM offset `pa + offset`.
    pub fn read_u8(&self, pa: u64, offset: usize) -> u8 {
        let addr = pa + offset as u64;
        if addr + 1 > self.size {
            return 0;
        }
        unsafe { std::ptr::read_volatile(self.base.add(addr as usize)) }
    }

    /// Read `len` bytes from DRAM offset `pa` into `buf`.
    /// Returns the number of bytes actually read (may be less than `len`
    /// if the read would go past the end of guest memory).
    pub fn read_bytes(&self, pa: u64, buf: &mut [u8]) -> usize {
        let len = buf.len() as u64;
        if pa >= self.size {
            return 0;
        }
        let avail = (self.size - pa).min(len) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(self.base.add(pa as usize), buf.as_mut_ptr(), avail);
        }
        avail
    }

    /// Write a u32 at DRAM offset `pa + offset`.
    pub fn write_u32(&self, pa: u64, offset: usize, val: u32) {
        let addr = pa + offset as u64;
        if addr + 4 > self.size {
            return;
        }
        unsafe { std::ptr::write_volatile(self.base.add(addr as usize) as *mut u32, val) }
    }

    /// Translate a kernel virtual address to guest physical address via
    /// page table walk.
    ///
    /// x86-64: supports 4-level (PGD -> PUD -> PMD -> PTE) and 5-level
    /// (PML5 -> P4D -> PUD -> PMD -> PTE) paging.
    ///
    /// aarch64: 3-level walk with AArch64 translation table descriptors
    /// (64KB granule, 48-bit VA). `l5` is ignored.
    ///
    /// `cr3_pa` is the physical address of the top-level page table.
    /// `l5` selects 5-level paging (x86 LA57); use `resolve_pgtable_l5`
    /// to detect the guest's mode at runtime.
    /// Returns `None` if any level is not present or the address is
    /// out of guest memory bounds.
    pub fn translate_kva(&self, cr3_pa: u64, kva: u64, l5: bool) -> Option<u64> {
        if l5 {
            self.walk_5level(cr3_pa, kva)
        } else {
            self.walk_4level(cr3_pa, kva)
        }
    }

    /// 4-level page table walk (x86-64).
    ///
    /// CR3 -> PGD -> PUD -> PMD -> PTE. Uses PS bit (bit 7) for
    /// huge pages, OA in bits \[51:12\].
    #[cfg(target_arch = "x86_64")]
    fn walk_4level(&self, cr3_pa: u64, kva: u64) -> Option<u64> {
        const PRESENT: u64 = 1;
        const PS: u64 = 1 << 7;
        const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

        let pgd_idx = (kva >> 39) & 0x1FF;
        let pud_idx = (kva >> 30) & 0x1FF;
        let pmd_idx = (kva >> 21) & 0x1FF;
        let pte_idx = (kva >> 12) & 0x1FF;
        let page_off = kva & 0xFFF;

        // PGD
        let pgd_pa = (cr3_pa & ADDR_MASK) + pgd_idx * 8;
        let pgde = self.read_u64(pgd_pa, 0);
        if pgde & PRESENT == 0 {
            return None;
        }

        // PUD
        let pud_pa = (pgde & ADDR_MASK) + pud_idx * 8;
        let pude = self.read_u64(pud_pa, 0);
        if pude & PRESENT == 0 {
            return None;
        }
        if pude & PS != 0 {
            let base = pude & 0x000F_FFFF_C000_0000;
            return Some(base | (kva & 0x3FFF_FFFF));
        }

        // PMD
        let pmd_pa = (pude & ADDR_MASK) + pmd_idx * 8;
        let pmde = self.read_u64(pmd_pa, 0);
        if pmde & PRESENT == 0 {
            return None;
        }
        if pmde & PS != 0 {
            let base = pmde & 0x000F_FFFF_FFE0_0000;
            return Some(base | (kva & 0x1F_FFFF));
        }

        // PTE
        let pte_pa = (pmde & ADDR_MASK) + pte_idx * 8;
        let ptee = self.read_u64(pte_pa, 0);
        if ptee & PRESENT == 0 {
            return None;
        }

        Some((ptee & ADDR_MASK) | page_off)
    }

    /// aarch64 page table walk (64KB granule, 3-level, 48-bit VA).
    ///
    /// TTBR_EL1 -> PGD -> PMD -> PTE.
    /// With 64KB pages and 48-bit VA, the kernel uses 3 levels:
    ///   PGD: bits [47:42] = 6 bits, 64 entries
    ///   PMD: bits [41:29] = 13 bits, 8192 entries
    ///   PTE: bits [28:16] = 13 bits, 8192 entries
    ///   page offset: bits [15:0] = 16 bits
    ///
    /// Descriptor format (ARMv8 D5.3):
    /// - bits [1:0] = 0b00: invalid
    /// - bits [1:0] = 0b01: block descriptor (PGD/PMD levels)
    /// - bits [1:0] = 0b11: table descriptor (PGD/PMD) or page (PTE)
    /// - bits [47:16]: output address for 64KB granule
    ///
    /// Page table entries contain guest physical addresses (GPAs). Since
    /// GuestMem is mapped at DRAM_START, all GPAs are adjusted by
    /// subtracting DRAM_START to produce offsets into the memory region.
    #[cfg(target_arch = "aarch64")]
    fn walk_4level(&self, ttbr_pa: u64, kva: u64) -> Option<u64> {
        use crate::vmm::kvm::DRAM_START;

        const VALID: u64 = 1;
        const TABLE: u64 = 0b11;
        const BLOCK: u64 = 0b01;
        const DESC_MASK: u64 = 0b11;
        // OA mask for 64KB granule: bits [47:16]
        const ADDR_MASK: u64 = 0x0000_FFFF_FFFF_0000;

        let to_offset = |gpa: u64| -> u64 { gpa.wrapping_sub(DRAM_START) };

        // 3-level walk for 64KB granule, 48-bit VA.
        let pgd_idx = (kva >> 42) & 0x3F; // bits [47:42], 6 bits
        let pmd_idx = (kva >> 29) & 0x1FFF; // bits [41:29], 13 bits
        let pte_idx = (kva >> 16) & 0x1FFF; // bits [28:16], 13 bits
        let page_off = kva & 0xFFFF; // bits [15:0], 16 bits

        // PGD — ttbr_pa is already a GuestMem offset.
        let pgd_off = (ttbr_pa & ADDR_MASK) + pgd_idx * 8;
        let pgde = self.read_u64(pgd_off, 0);
        if pgde & VALID == 0 {
            return None;
        }
        // PGD block: 4TB region (unlikely but spec-allowed)
        if pgde & DESC_MASK == BLOCK {
            let base = pgde & 0x0000_FC00_0000_0000;
            return Some(to_offset(base) | (kva & 0x3FF_FFFF_FFFF));
        }

        // PMD
        let pmd_off = to_offset(pgde & ADDR_MASK) + pmd_idx * 8;
        let pmde = self.read_u64(pmd_off, 0);
        if pmde & VALID == 0 {
            return None;
        }
        // PMD block: 512MB region
        if pmde & DESC_MASK == BLOCK {
            let base = pmde & 0x0000_FFFF_E000_0000;
            return Some(to_offset(base) | (kva & 0x1FFF_FFFF));
        }

        // PTE — page descriptor (bits [1:0] = 0b11)
        let pte_off = to_offset(pmde & ADDR_MASK) + pte_idx * 8;
        let ptee = self.read_u64(pte_off, 0);
        if ptee & VALID == 0 {
            return None;
        }
        if ptee & DESC_MASK != TABLE {
            return None;
        }

        Some(to_offset(ptee & ADDR_MASK) | page_off)
    }

    /// 5-level page table walk: CR3 -> PML5 -> P4D -> PUD -> PMD -> PTE.
    /// x86-64 only; aarch64 does not use 5-level paging.
    #[cfg(target_arch = "x86_64")]
    fn walk_5level(&self, cr3_pa: u64, kva: u64) -> Option<u64> {
        const PRESENT: u64 = 1;
        const ADDR_MASK: u64 = 0x000F_FFFF_FFFF_F000;

        // PML5 index: bits 56:48.
        let pml5_idx = (kva >> 48) & 0x1FF;

        let pml5_pa = (cr3_pa & ADDR_MASK) + pml5_idx * 8;
        let pml5e = self.read_u64(pml5_pa, 0);
        if pml5e & PRESENT == 0 {
            return None;
        }

        // P4D is the next level; continue with 4-level walk from there.
        let p4d_pa = pml5e & ADDR_MASK;
        self.walk_4level(p4d_pa, kva)
    }

    /// aarch64 stub: 5-level paging is not used.
    #[cfg(target_arch = "aarch64")]
    fn walk_5level(&self, _cr3_pa: u64, _kva: u64) -> Option<u64> {
        None
    }

    /// Guest memory size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Read scheduler stats from one CPU's struct rq at the given physical address.
pub(crate) fn read_rq_stats(mem: &GuestMem, rq_pa: u64, offsets: &KernelOffsets) -> CpuSnapshot {
    CpuSnapshot {
        nr_running: mem.read_u32(rq_pa, offsets.rq_nr_running),
        scx_nr_running: mem.read_u32(rq_pa, offsets.rq_scx + offsets.scx_rq_nr_running),
        local_dsq_depth: mem.read_u32(
            rq_pa,
            offsets.rq_scx + offsets.scx_rq_local_dsq + offsets.dsq_nr,
        ),
        rq_clock: mem.read_u64(rq_pa, offsets.rq_clock),
        scx_flags: mem.read_u32(rq_pa, offsets.rq_scx + offsets.scx_rq_flags),
        event_counters: None,
        schedstat: None,
        vcpu_cpu_time_ns: None,
        sched_domains: None,
    }
}

/// Read scx event counters from one CPU's per-CPU event stats struct.
/// On 7.1+, `pcpu_pa` points to `scx_sched_pcpu`; on 6.16, it points
/// directly to `scx_event_stats` (`event_stats_off` = 0).
pub(crate) fn read_event_stats(
    mem: &GuestMem,
    pcpu_pa: u64,
    ev: &ScxEventOffsets,
) -> ScxEventCounters {
    let base = pcpu_pa + ev.event_stats_off as u64;
    ScxEventCounters {
        select_cpu_fallback: mem.read_i64(base, ev.ev_select_cpu_fallback),
        dispatch_local_dsq_offline: mem.read_i64(base, ev.ev_dispatch_local_dsq_offline),
        dispatch_keep_last: mem.read_i64(base, ev.ev_dispatch_keep_last),
        enq_skip_exiting: mem.read_i64(base, ev.ev_enq_skip_exiting),
        enq_skip_migration_disabled: mem.read_i64(base, ev.ev_enq_skip_migration_disabled),
    }
}

/// Read schedstat fields from one CPU's struct rq at the given physical address.
pub(crate) fn read_rq_schedstat(mem: &GuestMem, rq_pa: u64, ss: &SchedstatOffsets) -> RqSchedstat {
    let sched_info_pa = rq_pa + ss.rq_sched_info as u64;
    RqSchedstat {
        run_delay: mem.read_u64(sched_info_pa, ss.sched_info_run_delay),
        pcount: mem.read_u64(sched_info_pa, ss.sched_info_pcount),
        yld_count: mem.read_u32(rq_pa, ss.rq_yld_count),
        sched_count: mem.read_u32(rq_pa, ss.rq_sched_count),
        sched_goidle: mem.read_u32(rq_pa, ss.rq_sched_goidle),
        ttwu_count: mem.read_u32(rq_pa, ss.rq_ttwu_count),
        ttwu_local: mem.read_u32(rq_pa, ss.rq_ttwu_local),
    }
}

/// Read a u32 array of `CPU_MAX_IDLE_TYPES` elements from guest memory.
fn read_u32_array(mem: &GuestMem, pa: u64, base_offset: usize) -> [u32; CPU_MAX_IDLE_TYPES] {
    std::array::from_fn(|i| mem.read_u32(pa, base_offset + i * 4))
}

/// Read CONFIG_SCHEDSTATS fields from one sched_domain.
fn read_sd_stats(mem: &GuestMem, sd_pa: u64, so: &SchedDomainStatsOffsets) -> SchedDomainStats {
    SchedDomainStats {
        lb_count: read_u32_array(mem, sd_pa, so.sd_lb_count),
        lb_failed: read_u32_array(mem, sd_pa, so.sd_lb_failed),
        lb_balanced: read_u32_array(mem, sd_pa, so.sd_lb_balanced),
        lb_imbalance_load: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_load),
        lb_imbalance_util: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_util),
        lb_imbalance_task: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_task),
        lb_imbalance_misfit: read_u32_array(mem, sd_pa, so.sd_lb_imbalance_misfit),
        lb_gained: read_u32_array(mem, sd_pa, so.sd_lb_gained),
        lb_hot_gained: read_u32_array(mem, sd_pa, so.sd_lb_hot_gained),
        lb_nobusyg: read_u32_array(mem, sd_pa, so.sd_lb_nobusyg),
        lb_nobusyq: read_u32_array(mem, sd_pa, so.sd_lb_nobusyq),
        alb_count: mem.read_u32(sd_pa, so.sd_alb_count),
        alb_failed: mem.read_u32(sd_pa, so.sd_alb_failed),
        alb_pushed: mem.read_u32(sd_pa, so.sd_alb_pushed),
        sbe_count: mem.read_u32(sd_pa, so.sd_sbe_count),
        sbe_balanced: mem.read_u32(sd_pa, so.sd_sbe_balanced),
        sbe_pushed: mem.read_u32(sd_pa, so.sd_sbe_pushed),
        sbf_count: mem.read_u32(sd_pa, so.sd_sbf_count),
        sbf_balanced: mem.read_u32(sd_pa, so.sd_sbf_balanced),
        sbf_pushed: mem.read_u32(sd_pa, so.sd_sbf_pushed),
        ttwu_wake_remote: mem.read_u32(sd_pa, so.sd_ttwu_wake_remote),
        ttwu_move_affine: mem.read_u32(sd_pa, so.sd_ttwu_move_affine),
        ttwu_move_balance: mem.read_u32(sd_pa, so.sd_ttwu_move_balance),
    }
}

/// Read the `sd->name` string from guest memory.
///
/// `sd->name` is a `char *` pointer to a static string in kernel rodata.
/// Rodata lives in the text mapping (`__START_KERNEL_map`), so
/// `text_kva_to_pa` is tried first. Falls back to direct mapping
/// (`kva_to_pa`) for kernels that place topology name strings
/// differently. Returns an empty string if the pointer is null or
/// translation fails.
fn read_sd_name(mem: &GuestMem, sd_pa: u64, name_offset: usize, page_offset: u64) -> String {
    let name_kva = mem.read_u64(sd_pa, name_offset);
    if name_kva == 0 {
        return String::new();
    }
    // Try text mapping first (rodata), then direct mapping.
    let text_pa = super::symbols::text_kva_to_pa(name_kva);
    let name_pa = if text_pa < mem.size() {
        text_pa
    } else {
        let direct_pa = super::symbols::kva_to_pa(name_kva, page_offset);
        if direct_pa >= mem.size() {
            return String::new();
        }
        direct_pa
    };
    // Domain names are short static strings ("SMT", "MC", "DIE", "NUMA",
    // "PKG", "BOOK", "DRAWER"). Read up to 16 bytes.
    let mut buf = [0u8; 16];
    let n = mem.read_bytes(name_pa, &mut buf);
    let end = buf[..n].iter().position(|&b| b == 0).unwrap_or(n);
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Read the sched_domain tree for one CPU.
///
/// Starts at `rq->sd` (the lowest-level domain), walks `sd->parent`
/// until NULL. Each domain is kmalloc'd and lives in the direct mapping.
///
/// `page_offset` is the runtime `PAGE_OFFSET` for direct-mapping translation.
///
/// Returns `None` if `rq->sd` is null (domain not yet built, or CPU
/// offline). Returns an empty `Vec` if the first domain pointer cannot
/// be translated.
///
/// Maximum depth is bounded to 8 levels to prevent infinite loops from
/// corrupted `sd->parent` chains.
pub(crate) fn read_sched_domain_tree(
    mem: &GuestMem,
    rq_pa: u64,
    sd_offsets: &SchedDomainOffsets,
    page_offset: u64,
) -> Option<Vec<SchedDomainSnapshot>> {
    const MAX_DEPTH: usize = 8;

    // rq->sd is a pointer (KVA).
    let sd_kva = mem.read_u64(rq_pa, sd_offsets.rq_sd);
    if sd_kva == 0 {
        return None;
    }

    let mut domains = Vec::new();
    let mut current_kva = sd_kva;

    for _ in 0..MAX_DEPTH {
        if current_kva == 0 {
            break;
        }

        // sched_domain is kmalloc'd — lives in direct mapping.
        let sd_pa = super::symbols::kva_to_pa(current_kva, page_offset);
        if sd_pa >= mem.size() {
            break;
        }

        let level = mem.read_u32(sd_pa, sd_offsets.sd_level) as i32;
        let name = read_sd_name(mem, sd_pa, sd_offsets.sd_name, page_offset);
        let flags = mem.read_u32(sd_pa, sd_offsets.sd_flags) as i32;
        let span_weight = mem.read_u32(sd_pa, sd_offsets.sd_span_weight);

        let stats = sd_offsets
            .stats_offsets
            .as_ref()
            .map(|so| read_sd_stats(mem, sd_pa, so));

        let snap = SchedDomainSnapshot {
            level,
            name,
            flags,
            span_weight,
            balance_interval: mem.read_u32(sd_pa, sd_offsets.sd_balance_interval),
            nr_balance_failed: mem.read_u32(sd_pa, sd_offsets.sd_nr_balance_failed),
            newidle_call: sd_offsets
                .sd_newidle_call
                .map(|off| mem.read_u32(sd_pa, off)),
            newidle_success: sd_offsets
                .sd_newidle_success
                .map(|off| mem.read_u32(sd_pa, off)),
            newidle_ratio: sd_offsets
                .sd_newidle_ratio
                .map(|off| mem.read_u32(sd_pa, off)),
            max_newidle_lb_cost: mem.read_u64(sd_pa, sd_offsets.sd_max_newidle_lb_cost),
            stats,
        };

        domains.push(snap);

        // Follow sd->parent.
        current_kva = mem.read_u64(sd_pa, sd_offsets.sd_parent);
    }

    Some(domains)
}

/// Resolve per-CPU physical addresses for event counter reads.
///
/// Reads `*scx_root` to find the active `scx_sched` struct, then reads
/// the percpu pointer at `scx_sched_pcpu_off` within it. On 7.1+ this
/// is `scx_sched.pcpu` (pointing to `scx_sched_pcpu`); on 6.16 it is
/// `scx_sched.event_stats_cpu` (pointing directly to `scx_event_stats`).
/// Computes each CPU's PA via `__per_cpu_offset`.
///
/// Returns None if `scx_root` is null (no scheduler loaded).
pub(crate) fn resolve_event_pcpu_pas(
    mem: &GuestMem,
    scx_root_pa: u64,
    ev: &ScxEventOffsets,
    per_cpu_offsets: &[u64],
    page_offset: u64,
) -> Option<Vec<u64>> {
    let scx_sched_kva = mem.read_u64(scx_root_pa, 0);
    if scx_sched_kva == 0 {
        return None;
    }

    let scx_sched_pa = super::symbols::kva_to_pa(scx_sched_kva, page_offset);
    let pcpu_kva = mem.read_u64(scx_sched_pa, ev.scx_sched_pcpu_off);
    if pcpu_kva == 0 {
        return None;
    }

    let pas: Vec<u64> = per_cpu_offsets
        .iter()
        .map(|&cpu_off| super::symbols::kva_to_pa(pcpu_kva.wrapping_add(cpu_off), page_offset))
        .collect();

    Some(pas)
}

/// Per-vCPU host thread timing info for gating stall detection.
///
/// When the host is loaded, vCPU threads get preempted and rq_clock
/// cannot advance. Reading per-thread CPU time distinguishes real
/// stalls (vCPU running but clock stuck) from host preemption
/// (vCPU not scheduled, clock can't advance).
pub(crate) struct VcpuTiming {
    /// pthread_t handles for each vCPU, indexed by vCPU ID.
    /// Used with `pthread_getcpuclockid()` + `clock_gettime()`.
    pub pthreads: Vec<libc::pthread_t>,
}

impl VcpuTiming {
    /// Read CPU time for each vCPU thread. Returns nanoseconds per vCPU.
    fn read_cpu_times(&self) -> Vec<u64> {
        self.pthreads
            .iter()
            .map(|&pt| {
                let mut clk: libc::clockid_t = 0;
                let ret = unsafe { libc::pthread_getcpuclockid(pt, &mut clk) };
                if ret != 0 {
                    return 0;
                }
                let mut ts = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                let ret = unsafe { libc::clock_gettime(clk, &mut ts) };
                if ret != 0 {
                    return 0;
                }
                ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
            })
            .collect()
    }
}

/// Configuration for reactive SysRq-D dump triggering.
///
/// When provided to `monitor_loop`, the monitor evaluates thresholds inline
/// and writes the dump request flag to guest SHM on sustained violation.
pub(crate) struct DumpTrigger {
    /// Physical address of the SHM region base in guest memory.
    pub shm_base_pa: u64,
    /// Thresholds for violation detection.
    pub thresholds: super::MonitorThresholds,
}

/// Override for the scheduler watchdog timeout, written every monitor
/// iteration.
///
/// Two write paths are supported:
/// - 7.1+ (`ScxSched`): deref `*scx_root` to find the runtime
///   `scx_sched` struct, then write at the BTF-resolved offset.
///   Re-derefs each iteration because `scx_sched` is reallocated on
///   scheduler (re)load.
/// - 6.16 (`StaticGlobal`): write directly to the PA of the
///   `scx_watchdog_timeout` static global. No deref needed — the
///   address is fixed for the kernel's lifetime.
pub(crate) enum WatchdogOverride {
    /// 7.1+ path: deref `scx_root` -> `scx_sched` -> write at offset.
    ScxSched {
        /// PA of the `scx_root` global pointer (text mapping).
        scx_root_pa: u64,
        /// Byte offset of `watchdog_timeout` within `struct scx_sched`.
        watchdog_offset: usize,
        /// Jiffies value to write.
        jiffies: u64,
        /// Runtime `PAGE_OFFSET` for KVA-to-PA translation.
        page_offset: u64,
    },
    /// 6.16 path: write directly to the static global's PA.
    StaticGlobal {
        /// PA of the `scx_watchdog_timeout` static global (text mapping).
        watchdog_timeout_pa: u64,
        /// Jiffies value to write.
        jiffies: u64,
    },
}

/// Pre-resolved BPF program stats context for the monitor loop.
pub(crate) struct ProgStatsCtx {
    pub cached: Vec<super::bpf_prog::CachedProgInfo>,
    pub per_cpu_offsets: Vec<u64>,
    pub page_offset: u64,
    pub offsets: super::btf_offsets::BpfProgOffsets,
}

/// Samples, SHM drain, and optional watchdog observation returned by
/// [`monitor_loop`].
pub(crate) struct MonitorLoopResult {
    pub(crate) samples: Vec<MonitorSample>,
    pub(crate) drain: crate::vmm::shm_ring::ShmDrainResult,
    pub(crate) watchdog_observation: Option<super::WatchdogObservation>,
}

/// Configuration for the monitor sampling loop.
///
/// Bundles the parameters that `monitor_loop` needs beyond the
/// required `mem`, `rq_pas`, `offsets`, `interval`, `kill`, and `start`.
pub(crate) struct MonitorConfig<'a> {
    /// Per-CPU physical addresses of `scx_sched_pcpu`. When present (and
    /// `event_offsets` exist), each sample includes event counters.
    pub event_pcpu_pas: Option<&'a [u64]>,
    /// Reactive dump configuration. When a sustained threshold violation is
    /// detected, writes the dump request flag to guest SHM to trigger a
    /// SysRq-D dump inside the guest.
    pub dump_trigger: Option<&'a DumpTrigger>,
    pub watchdog_override: Option<&'a WatchdogOverride>,
    pub vcpu_timing: Option<&'a VcpuTiming>,
    pub preemption_threshold_ns: u64,
    pub shm_base_pa: Option<u64>,
    pub prog_stats_ctx: Option<&'a ProgStatsCtx>,
    /// Runtime `PAGE_OFFSET` for direct-mapping KVA translation. Used by
    /// sched_domain tree walking to translate `rq->sd` and `sd->parent`
    /// pointers.
    pub page_offset: u64,
}

/// Run the monitor loop, sampling all CPUs at the given interval.
/// Returns collected samples when `kill` is set.
pub(crate) fn monitor_loop(
    mem: &GuestMem,
    rq_pas: &[u64],
    offsets: &KernelOffsets,
    interval: Duration,
    kill: &AtomicBool,
    start: Instant,
    cfg: &MonitorConfig<'_>,
) -> MonitorLoopResult {
    let event_pcpu_pas = cfg.event_pcpu_pas;
    let dump_trigger = cfg.dump_trigger;
    let watchdog_override = cfg.watchdog_override;
    let vcpu_timing = cfg.vcpu_timing;
    let preemption_threshold_ns = cfg.preemption_threshold_ns;
    let shm_base_pa = cfg.shm_base_pa;
    let prog_stats_ctx = cfg.prog_stats_ctx;
    let page_offset = cfg.page_offset;
    let preemption_threshold_ns = if preemption_threshold_ns > 0 {
        preemption_threshold_ns
    } else {
        super::vcpu_preemption_threshold_ns(None)
    };
    let mut samples: Vec<MonitorSample> = Vec::new();
    let mut consecutive_imbalance = 0usize;
    let mut consecutive_dsq = 0usize;
    let mut consecutive_stall = vec![0usize; rq_pas.len()];
    let mut dump_requested = false;
    let mut cpus: Vec<CpuSnapshot> = Vec::with_capacity(rq_pas.len());
    let mut prev_vcpu_times: Option<Vec<u64>> = None;
    let mut shm_entries: Vec<crate::vmm::shm_ring::ShmEntry> = Vec::new();
    let mut shm_drops: u64 = 0;
    let mut watchdog_observation: Option<super::WatchdogObservation> = None;

    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        if let Some(wd) = watchdog_override {
            let (write_pa, write_offset, wd_jiffies) = match wd {
                WatchdogOverride::ScxSched {
                    scx_root_pa,
                    watchdog_offset,
                    jiffies,
                    page_offset,
                } => {
                    let sch_kva = mem.read_u64(*scx_root_pa, 0);
                    if sch_kva == 0 {
                        (None, 0, *jiffies)
                    } else {
                        let sch_pa = super::symbols::kva_to_pa(sch_kva, *page_offset);
                        (Some(sch_pa), *watchdog_offset, *jiffies)
                    }
                }
                WatchdogOverride::StaticGlobal {
                    watchdog_timeout_pa,
                    jiffies,
                } => (Some(*watchdog_timeout_pa), 0, *jiffies),
            };
            if let Some(pa) = write_pa {
                mem.write_u64(pa, write_offset, wd_jiffies);
                if watchdog_observation.is_none() {
                    let observed = mem.read_u64(pa, write_offset);
                    watchdog_observation = Some(super::WatchdogObservation {
                        expected_jiffies: wd_jiffies,
                        observed_jiffies: observed,
                    });
                }
            }
        }
        cpus.clear();
        cpus.extend(rq_pas.iter().map(|&pa| read_rq_stats(mem, pa, offsets)));

        // Overlay event counters if available.
        if let (Some(pcpu_pas), Some(ev)) = (event_pcpu_pas, &offsets.event_offsets) {
            for (i, cpu) in cpus.iter_mut().enumerate() {
                if let Some(&pcpu_pa) = pcpu_pas.get(i) {
                    cpu.event_counters = Some(read_event_stats(mem, pcpu_pa, ev));
                }
            }
        }

        // Overlay schedstat fields if available.
        if let Some(ss) = &offsets.schedstat_offsets {
            for (i, cpu) in cpus.iter_mut().enumerate() {
                if let Some(&rq_pa) = rq_pas.get(i) {
                    cpu.schedstat = Some(read_rq_schedstat(mem, rq_pa, ss));
                }
            }
        }

        // Overlay sched domain tree if available.
        if let Some(sd) = &offsets.sched_domain_offsets {
            for (i, cpu) in cpus.iter_mut().enumerate() {
                if let Some(&rq_pa) = rq_pas.get(i) {
                    cpu.sched_domains = read_sched_domain_tree(mem, rq_pa, sd, page_offset);
                }
            }
        }

        // Read vCPU CPU times and store in snapshots for post-hoc analysis.
        let curr_vcpu_times = vcpu_timing.map(|vt| vt.read_cpu_times());
        if let Some(ref times) = curr_vcpu_times {
            for (i, cpu) in cpus.iter_mut().enumerate() {
                if let Some(&t) = times.get(i) {
                    cpu.vcpu_cpu_time_ns = Some(t);
                }
            }
        }

        // Inline threshold evaluation for reactive dump.
        if let Some(trigger) = dump_trigger
            && !dump_requested
            && !cpus.is_empty()
        {
            let t = &trigger.thresholds;

            // Imbalance check.
            let mut min_nr = u32::MAX;
            let mut max_nr = 0u32;
            for cpu in &cpus {
                min_nr = min_nr.min(cpu.nr_running);
                max_nr = max_nr.max(cpu.nr_running);
            }
            let ratio = max_nr as f64 / min_nr.max(1) as f64;
            if ratio > t.max_imbalance_ratio {
                consecutive_imbalance += 1;
            } else {
                consecutive_imbalance = 0;
            }

            // DSQ depth check.
            if cpus
                .iter()
                .any(|c| c.local_dsq_depth > t.max_local_dsq_depth)
            {
                consecutive_dsq += 1;
            } else {
                consecutive_dsq = 0;
            }

            // Stall check: per-CPU sustained window, exempt idle CPUs
            // (nr_running==0 in both samples: NOHZ tick stopped) and
            // preempted vCPUs (CPU time didn't advance: host stole the core).
            if t.fail_on_stall
                && let Some(prev) = samples.last()
            {
                let n = prev.cpus.len().min(cpus.len());
                for i in 0..n {
                    let idle = cpus[i].nr_running == 0 && prev.cpus[i].nr_running == 0;
                    let preempted = match (&prev_vcpu_times, &curr_vcpu_times) {
                        (Some(prev_t), Some(curr_t)) if i < prev_t.len() && i < curr_t.len() => {
                            curr_t[i].saturating_sub(prev_t[i]) < preemption_threshold_ns
                        }
                        _ => false,
                    };
                    let is_stall = cpus[i].rq_clock != 0
                        && cpus[i].rq_clock == prev.cpus[i].rq_clock
                        && !idle
                        && !preempted;
                    if is_stall {
                        consecutive_stall[i] += 1;
                    } else {
                        consecutive_stall[i] = 0;
                    }
                }
            }
            let sustained = consecutive_imbalance >= t.sustained_samples
                || consecutive_dsq >= t.sustained_samples
                || consecutive_stall.iter().any(|&c| c >= t.sustained_samples);

            if sustained {
                mem.write_u8(
                    trigger.shm_base_pa,
                    crate::vmm::shm_ring::DUMP_REQ_OFFSET,
                    crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D,
                );
                dump_requested = true;
            }
        }

        prev_vcpu_times = curr_vcpu_times;

        let prog_stats = prog_stats_ctx.map(|ctx| {
            super::bpf_prog::read_prog_runtime_stats(
                mem,
                &ctx.cached,
                &ctx.per_cpu_offsets,
                ctx.page_offset,
                &ctx.offsets,
            )
        });

        samples.push(MonitorSample {
            elapsed_ms: start.elapsed().as_millis() as u64,
            cpus: cpus.clone(),
            prog_stats,
        });

        // Mid-flight SHM drain: advance read_ptr so the guest can
        // reclaim ring space. Accumulate drained entries for the
        // caller to merge with the post-mortem drain.
        if let Some(shm_pa) = shm_base_pa {
            let drain = crate::vmm::shm_ring::shm_drain_live(mem, shm_pa);
            shm_drops = shm_drops.max(drain.drops);
            // Check for scheduler death signal before accumulating.
            // The guest init writes MSG_TYPE_SCHED_EXIT when the
            // scheduler process exits during test execution.
            if drain
                .entries
                .iter()
                .any(|e| e.msg_type == crate::vmm::shm_ring::MSG_TYPE_SCHED_EXIT && e.crc_ok)
            {
                shm_entries.extend(drain.entries);
                kill.store(true, Ordering::Release);
                break;
            }
            shm_entries.extend(drain.entries);
        }

        std::thread::sleep(interval);
    }
    let shm_result = crate::vmm::shm_ring::ShmDrainResult {
        entries: shm_entries,
        drops: shm_drops,
    };
    MonitorLoopResult {
        samples,
        drain: shm_result,
        watchdog_observation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::thread::JoinHandleExt;

    fn test_config() -> MonitorConfig<'static> {
        MonitorConfig {
            event_pcpu_pas: None,
            dump_trigger: None,
            watchdog_override: None,
            vcpu_timing: None,
            preemption_threshold_ns: 0,
            shm_base_pa: None,
            prog_stats_ctx: None,
            page_offset: 0,
        }
    }

    fn test_offsets() -> KernelOffsets {
        KernelOffsets {
            rq_nr_running: 8,
            rq_clock: 16,
            rq_scx: 100,
            scx_rq_nr_running: 4,
            scx_rq_local_dsq: 20,
            scx_rq_flags: 8,
            dsq_nr: 0,
            event_offsets: None,
            schedstat_offsets: None,
            sched_domain_offsets: None,
            watchdog_offsets: None,
        }
    }

    /// Build a byte buffer simulating a struct rq with the given field values.
    fn make_rq_buffer(
        offsets: &KernelOffsets,
        nr_running: u32,
        scx_nr: u32,
        dsq_nr: u32,
        clock: u64,
        flags: u32,
    ) -> Vec<u8> {
        let size = offsets.rq_scx + offsets.scx_rq_local_dsq + offsets.dsq_nr + 8;
        let mut buf = vec![0u8; size];

        buf[offsets.rq_nr_running..offsets.rq_nr_running + 4]
            .copy_from_slice(&nr_running.to_ne_bytes());
        buf[offsets.rq_clock..offsets.rq_clock + 8].copy_from_slice(&clock.to_ne_bytes());

        let scx_base = offsets.rq_scx;
        buf[scx_base + offsets.scx_rq_nr_running..scx_base + offsets.scx_rq_nr_running + 4]
            .copy_from_slice(&scx_nr.to_ne_bytes());
        buf[scx_base + offsets.scx_rq_flags..scx_base + offsets.scx_rq_flags + 4]
            .copy_from_slice(&flags.to_ne_bytes());

        let dsq_base = scx_base + offsets.scx_rq_local_dsq;
        buf[dsq_base + offsets.dsq_nr..dsq_base + offsets.dsq_nr + 4]
            .copy_from_slice(&dsq_nr.to_ne_bytes());
        buf
    }

    #[test]
    fn read_rq_stats_known_values() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 5, 3, 7, 999_000, 0x1);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, 5);
        assert_eq!(snap.scx_nr_running, 3);
        assert_eq!(snap.local_dsq_depth, 7);
        assert_eq!(snap.rq_clock, 999_000);
        assert_eq!(snap.scx_flags, 0x1);
    }

    #[test]
    fn read_rq_stats_all_zeros() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 0, 0, 0, 0, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, 0);
        assert_eq!(snap.scx_nr_running, 0);
        assert_eq!(snap.local_dsq_depth, 0);
        assert_eq!(snap.rq_clock, 0);
        assert_eq!(snap.scx_flags, 0);
    }

    #[test]
    fn read_rq_stats_max_values() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, u32::MAX, u32::MAX, u32::MAX, u64::MAX, u32::MAX);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert_eq!(snap.nr_running, u32::MAX);
        assert_eq!(snap.scx_nr_running, u32::MAX);
        assert_eq!(snap.local_dsq_depth, u32::MAX);
        assert_eq!(snap.rq_clock, u64::MAX);
        assert_eq!(snap.scx_flags, u32::MAX);
    }

    #[test]
    fn read_u32_out_of_bounds() {
        let buf = [0xFFu8; 8];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        // PA 6 + 4 bytes = 10 > 8, out of bounds
        assert_eq!(mem.read_u32(6, 0), 0);
        // Exactly at boundary: PA 4, offset 0 => addr 4, 4+4=8 == size, not >
        assert_eq!(mem.read_u32(4, 0), u32::from_ne_bytes([0xFF; 4]));
        // One past: PA 5, offset 0 => addr 5, 5+4=9 > 8
        assert_eq!(mem.read_u32(5, 0), 0);
    }

    #[test]
    fn read_u64_out_of_bounds() {
        let buf = [0xFFu8; 16];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        // PA 10 + 8 = 18 > 16
        assert_eq!(mem.read_u64(10, 0), 0);
        // Exactly at boundary: PA 8, 8+8=16 == size
        assert_eq!(mem.read_u64(8, 0), u64::from_ne_bytes([0xFF; 8]));
        // One past
        assert_eq!(mem.read_u64(9, 0), 0);
    }

    #[test]
    fn monitor_loop_kill_immediately() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let kill = AtomicBool::new(true);
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        assert!(samples.is_empty());
    }

    #[test]
    fn monitor_loop_one_iteration() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 2, 1, 3, 500, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert_eq!(samples[0].cpus.len(), 1);
        assert_eq!(samples[0].cpus[0].nr_running, 2);
        assert_eq!(samples[0].cpus[0].scx_nr_running, 1);
        assert_eq!(samples[0].cpus[0].local_dsq_depth, 3);
        assert_eq!(samples[0].cpus[0].rq_clock, 500);
    }

    #[test]
    fn two_cpu_independent_reads() {
        let offsets = test_offsets();
        let buf0 = make_rq_buffer(&offsets, 10, 5, 2, 1000, 0x1);
        let buf1 = make_rq_buffer(&offsets, 20, 15, 8, 2000, 0x2);

        // Concatenate into a single memory region; CPU 1's rq starts after CPU 0's.
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);

        let mem = GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64);

        let snap0 = read_rq_stats(&mem, 0, &offsets);
        let snap1 = read_rq_stats(&mem, pa1, &offsets);

        assert_eq!(snap0.nr_running, 10);
        assert_eq!(snap0.scx_nr_running, 5);
        assert_eq!(snap0.local_dsq_depth, 2);
        assert_eq!(snap0.rq_clock, 1000);
        assert_eq!(snap0.scx_flags, 0x1);

        assert_eq!(snap1.nr_running, 20);
        assert_eq!(snap1.scx_nr_running, 15);
        assert_eq!(snap1.local_dsq_depth, 8);
        assert_eq!(snap1.rq_clock, 2000);
        assert_eq!(snap1.scx_flags, 0x2);
    }

    #[test]
    fn read_u32_nonzero_pa_and_offset() {
        // Verify that PA + offset are combined correctly.
        let mut buf = [0u8; 32];
        // Place 0xDEADBEEF at byte 20 (PA=12, offset=8).
        buf[20..24].copy_from_slice(&0xDEADBEEFu32.to_ne_bytes());
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(mem.read_u32(12, 8), 0xDEADBEEF);
    }

    #[test]
    fn read_u64_nonzero_pa_and_offset() {
        let mut buf = [0u8; 32];
        // Place value at byte 16 (PA=10, offset=6).
        buf[16..24].copy_from_slice(&0x0123456789ABCDEFu64.to_ne_bytes());
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(mem.read_u64(10, 6), 0x0123456789ABCDEF);
    }

    #[test]
    fn monitor_loop_multi_cpu() {
        let offsets = test_offsets();
        let buf0 = make_rq_buffer(&offsets, 3, 2, 1, 100, 0);
        let buf1 = make_rq_buffer(&offsets, 7, 5, 4, 200, 0);
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);

        let mem = GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Each sample should have 2 CPUs.
        for s in &samples {
            assert_eq!(s.cpus.len(), 2);
        }
        // CPU 0 values
        assert_eq!(samples[0].cpus[0].nr_running, 3);
        assert_eq!(samples[0].cpus[0].scx_nr_running, 2);
        // CPU 1 values
        assert_eq!(samples[0].cpus[1].nr_running, 7);
        assert_eq!(samples[0].cpus[1].scx_nr_running, 5);
    }

    #[test]
    fn monitor_loop_elapsed_ms_progresses() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 2,
            "need at least 2 samples, got {}",
            samples.len()
        );
        // elapsed_ms must be monotonically non-decreasing.
        for w in samples.windows(2) {
            assert!(
                w[1].elapsed_ms >= w[0].elapsed_ms,
                "elapsed_ms went backwards: {} -> {}",
                w[0].elapsed_ms,
                w[1].elapsed_ms
            );
        }
        // Last sample should have elapsed > 0.
        assert!(samples.last().unwrap().elapsed_ms > 0);
    }

    fn test_event_offsets() -> ScxEventOffsets {
        ScxEventOffsets {
            scx_sched_pcpu_off: 0,
            event_stats_off: 0,
            ev_select_cpu_fallback: 0,
            ev_dispatch_local_dsq_offline: 8,
            ev_dispatch_keep_last: 16,
            ev_enq_skip_exiting: 24,
            ev_enq_skip_migration_disabled: 32,
        }
    }

    /// Build a byte buffer simulating a scx_sched_pcpu with event_stats.
    fn make_event_stats_buffer(
        ev: &ScxEventOffsets,
        fallback: i64,
        offline: i64,
        keep_last: i64,
        skip_exit: i64,
        skip_mig: i64,
    ) -> Vec<u8> {
        let size = ev.event_stats_off + ev.ev_enq_skip_migration_disabled + 8;
        let mut buf = vec![0u8; size];
        let base = ev.event_stats_off;
        buf[base + ev.ev_select_cpu_fallback..base + ev.ev_select_cpu_fallback + 8]
            .copy_from_slice(&fallback.to_ne_bytes());
        buf[base + ev.ev_dispatch_local_dsq_offline..base + ev.ev_dispatch_local_dsq_offline + 8]
            .copy_from_slice(&offline.to_ne_bytes());
        buf[base + ev.ev_dispatch_keep_last..base + ev.ev_dispatch_keep_last + 8]
            .copy_from_slice(&keep_last.to_ne_bytes());
        buf[base + ev.ev_enq_skip_exiting..base + ev.ev_enq_skip_exiting + 8]
            .copy_from_slice(&skip_exit.to_ne_bytes());
        buf[base + ev.ev_enq_skip_migration_disabled..base + ev.ev_enq_skip_migration_disabled + 8]
            .copy_from_slice(&skip_mig.to_ne_bytes());
        buf
    }

    #[test]
    fn read_event_stats_known_values() {
        let ev = test_event_offsets();
        let buf = make_event_stats_buffer(&ev, 42, 7, 100, 3, 5);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let stats = read_event_stats(&mem, 0, &ev);
        assert_eq!(stats.select_cpu_fallback, 42);
        assert_eq!(stats.dispatch_local_dsq_offline, 7);
        assert_eq!(stats.dispatch_keep_last, 100);
        assert_eq!(stats.enq_skip_exiting, 3);
        assert_eq!(stats.enq_skip_migration_disabled, 5);
    }

    #[test]
    fn read_event_stats_zeros() {
        let ev = test_event_offsets();
        let buf = make_event_stats_buffer(&ev, 0, 0, 0, 0, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let stats = read_event_stats(&mem, 0, &ev);
        assert_eq!(stats.select_cpu_fallback, 0);
        assert_eq!(stats.dispatch_local_dsq_offline, 0);
    }

    #[test]
    fn read_i64_roundtrip() {
        let val: i64 = -12345;
        let buf = val.to_ne_bytes();
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(mem.read_i64(0, 0), -12345);
    }

    #[test]
    fn write_u8_and_read_u8() {
        let mut buf = [0u8; 16];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        mem.write_u8(0, 5, 0xAB);
        assert_eq!(mem.read_u8(0, 5), 0xAB);
        assert_eq!(buf[5], 0xAB);
    }

    #[test]
    fn write_u8_out_of_bounds() {
        let mut buf = [0u8; 4];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        // Should not panic or write.
        mem.write_u8(4, 0, 0xFF);
        assert_eq!(buf, [0u8; 4]);
    }

    #[test]
    fn write_u64_and_read_u64() {
        let mut buf = [0u8; 32];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        mem.write_u64(0, 8, 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(mem.read_u64(0, 8), 0xDEAD_BEEF_CAFE_1234);
        assert_eq!(
            u64::from_ne_bytes(buf[8..16].try_into().unwrap()),
            0xDEAD_BEEF_CAFE_1234
        );
    }

    #[test]
    fn write_u64_out_of_bounds() {
        let mut buf = [0u8; 8];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        // addr 1 + 8 = 9 > 8, out of bounds
        mem.write_u64(1, 0, 0xFF);
        assert_eq!(buf, [0u8; 8]);
    }

    #[test]
    fn write_u64_at_boundary() {
        let mut buf = [0u8; 16];
        let mem = GuestMem::new(buf.as_mut_ptr(), buf.len() as u64);
        // PA 8 + 8 = 16 == size, should succeed
        mem.write_u64(8, 0, 0x0123_4567_89AB_CDEF);
        assert_eq!(mem.read_u64(8, 0), 0x0123_4567_89AB_CDEF);
    }

    #[test]
    fn read_u8_out_of_bounds() {
        let buf = [0xFFu8; 4];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        assert_eq!(mem.read_u8(4, 0), 0);
        assert_eq!(mem.read_u8(3, 0), 0xFF);
    }

    #[test]
    fn read_rq_stats_has_no_event_counters() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let snap = read_rq_stats(&mem, 0, &offsets);
        assert!(snap.event_counters.is_none());
    }

    #[test]
    fn monitor_loop_with_event_counters() {
        let ev = test_event_offsets();
        let mut offsets = test_offsets();
        offsets.event_offsets = Some(ev.clone());

        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let ev_buf = make_event_stats_buffer(&ev, 10, 20, 30, 40, 50);

        let rq_pa = 0u64;
        let ev_pa = rq_buf.len() as u64;
        let mut combined = rq_buf;
        combined.extend_from_slice(&ev_buf);

        let mem = GuestMem::new(combined.as_ptr() as *mut u8, combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let ev_pas = vec![ev_pa];
        let cfg = MonitorConfig {
            event_pcpu_pas: Some(&ev_pas),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[rq_pa],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        let counters = samples[0].cpus[0].event_counters.as_ref().unwrap();
        assert_eq!(counters.select_cpu_fallback, 10);
        assert_eq!(counters.dispatch_local_dsq_offline, 20);
        assert_eq!(counters.dispatch_keep_last, 30);
        assert_eq!(counters.enq_skip_exiting, 40);
        assert_eq!(counters.enq_skip_migration_disabled, 50);
    }

    #[test]
    fn monitor_loop_no_event_counters_when_none() {
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert!(samples[0].cpus[0].event_counters.is_none());
    }

    #[test]
    fn resolve_event_pcpu_pas_null_scx_root() {
        let ev = test_event_offsets();
        // scx_root pointer is 0 (null) — no scheduler loaded.
        let buf = [0u8; 64];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let result = resolve_event_pcpu_pas(&mem, 0, &ev, &[0, 0x4000], 0);
        assert!(result.is_none());
    }

    #[test]
    fn monitor_loop_with_watchdog_override() {
        let offsets = test_offsets();
        // Layout:
        //   [rq_buf]
        //   [scx_root pointer slot @ scx_root_pa] (holds scx_sched KVA)
        //   [scx_sched struct @ sch_pa, with watchdog_timeout at watchdog_offset]
        // The monitor derefs *scx_root_pa -> KVA, translates via PAGE_OFFSET -> PA,
        // then writes jiffies at sch_pa + watchdog_offset.
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let scx_root_pa = rq_buf.len() as u64;
        let sch_pa = scx_root_pa + 8;
        let watchdog_offset: usize = 16;
        let page_offset = super::super::symbols::DEFAULT_PAGE_OFFSET;
        let scx_sched_kva = page_offset.wrapping_add(sch_pa);

        // Buffer = rq_buf | 8 bytes (scx_root slot) | 64 bytes (scx_sched stub).
        let mut combined = rq_buf;
        combined.extend_from_slice(&scx_sched_kva.to_ne_bytes());
        combined.extend_from_slice(&[0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let wd = WatchdogOverride::ScxSched {
            scx_root_pa,
            watchdog_offset,
            jiffies: 99999,
            page_offset,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            watchdog_override: Some(&wd),
            ..test_config()
        };
        let MonitorLoopResult {
            samples,
            watchdog_observation,
            ..
        } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Verify the watchdog value was written at sch_pa + watchdog_offset.
        let write_pa = sch_pa as usize + watchdog_offset;
        let written = u64::from_ne_bytes(combined[write_pa..write_pa + 8].try_into().unwrap());
        assert_eq!(written, 99999);
        // Verify monitor_loop recorded the observation.
        let obs = watchdog_observation.expect("watchdog_observation should be Some after write");
        assert_eq!(obs.expected_jiffies, 99999);
        assert_eq!(obs.observed_jiffies, 99999);
    }

    #[test]
    fn monitor_loop_watchdog_override_skipped_when_scx_root_null() {
        let offsets = test_offsets();
        // Layout: rq_buf | scx_root slot = 0 (no scheduler loaded).
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let scx_root_pa = rq_buf.len() as u64;
        let mut combined = rq_buf;
        combined.extend_from_slice(&[0u8; 8]); // scx_root = null
        // Extra space in case of accidental write via garbage deref.
        combined.extend_from_slice(&[0u8; 128]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));
        let wd = WatchdogOverride::ScxSched {
            scx_root_pa,
            watchdog_offset: 16,
            jiffies: 0xDEADBEEF,
            page_offset: super::super::symbols::DEFAULT_PAGE_OFFSET,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            watchdog_override: Some(&wd),
            ..test_config()
        };
        let MonitorLoopResult {
            watchdog_observation,
            ..
        } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        // No write should have happened: buffer is all zeros past rq_buf.
        assert!(
            combined[scx_root_pa as usize..].iter().all(|&b| b == 0),
            "no write should occur when scx_root is null"
        );
        // No observation should have been recorded.
        assert!(
            watchdog_observation.is_none(),
            "watchdog_observation should be None when scx_root is null"
        );
    }

    #[test]
    fn monitor_loop_watchdog_static_global_writes_directly() {
        let offsets = test_offsets();
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let watchdog_pa = rq_buf.len() as u64;

        let mut combined = rq_buf;
        combined.extend_from_slice(&[0u8; 8]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let wd = WatchdogOverride::StaticGlobal {
            watchdog_timeout_pa: watchdog_pa,
            jiffies: 77777,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            watchdog_override: Some(&wd),
            ..test_config()
        };
        let MonitorLoopResult {
            samples,
            watchdog_observation,
            ..
        } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        let written = u64::from_ne_bytes(
            combined[watchdog_pa as usize..watchdog_pa as usize + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(written, 77777);
        let obs = watchdog_observation.expect("watchdog_observation should be Some");
        assert_eq!(obs.expected_jiffies, 77777);
        assert_eq!(obs.observed_jiffies, 77777);
    }

    #[test]
    fn monitor_loop_dump_trigger_fires_on_imbalance() {
        let offsets = test_offsets();
        // Two rq buffers: CPU0 = 1 task, CPU1 = 20 tasks -> ratio=20 >> threshold.
        let buf0 = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let buf1 = make_rq_buffer(&offsets, 20, 20, 1, 200, 0);
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);
        // Append SHM region (64 bytes minimum for dump req offset).
        let shm_pa = combined.len() as u64;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 2.0,
                sustained_samples: 2,
                fail_on_stall: false,
                ..Default::default()
            },
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Verify dump request was written to SHM.
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        assert_eq!(
            dump_byte,
            crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D,
            "dump request should have been written to SHM"
        );
    }

    #[test]
    fn monitor_loop_dump_trigger_stall_with_sustained_window() {
        // Reactive stall path: stuck rq_clock with nr_running>0 triggers
        // dump after sustained_samples consecutive stall pairs.
        let offsets = test_offsets();
        // Single CPU: nr_running=2 (busy), rq_clock stuck at 5000.
        // Need a second CPU with advancing clock so samples differ
        // (otherwise all-same-clock triggers the uninitialized check in
        // from_samples, though monitor_loop's reactive path doesn't use
        // from_samples — it checks inline).
        let buf = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let shm_pa = buf.len() as u64;
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 2,
                ..Default::default()
            },
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        // Should have enough samples for 2+ stall pairs.
        assert!(
            samples.len() >= 3,
            "need >= 3 samples for 2 stall pairs, got {}",
            samples.len()
        );
        // Dump should have fired due to sustained stall.
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        assert_eq!(
            dump_byte,
            crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D,
            "stall should trigger dump after sustained_samples=2"
        );
    }

    #[test]
    fn monitor_loop_dump_trigger_idle_cpu_no_stall() {
        // Reactive path: nr_running==0 (idle) with stuck rq_clock should
        // NOT trigger the dump, even with fail_on_stall=true.
        let offsets = test_offsets();
        // CPU idle: nr_running=0, rq_clock stuck at 5000.
        let buf = make_rq_buffer(&offsets, 0, 0, 0, 5000, 0);
        let shm_pa = buf.len() as u64;
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 1,
                ..Default::default()
            },
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 2,
            "need >= 2 samples, got {}",
            samples.len()
        );
        // Dump should NOT have fired — idle CPU is exempt.
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        assert_eq!(dump_byte, 0, "idle CPU should not trigger stall dump");
    }

    #[test]
    fn monitor_loop_vcpu_timing_preempted_no_stall() {
        // Sleeping thread: CPU time stays near zero between samples.
        // rq_clock stuck + CPU time not advancing = preempted, suppress stall.
        // 30ms interval gives margin on loaded hosts. Explicit threshold
        // (10ms) avoids host CONFIG_HZ dependency.
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let shm_pa = buf.len() as u64;
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let sleeper_kill = std::sync::Arc::new(AtomicBool::new(false));
        let sleeper_kill_clone = sleeper_kill.clone();
        let sleeper = std::thread::Builder::new()
            .name("vcpu-sleeper".into())
            .spawn(move || {
                while !sleeper_kill_clone.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(100));
                }
            })
            .unwrap();

        let pt = sleeper.as_pthread_t() as libc::pthread_t;
        let vcpu_timing = VcpuTiming { pthreads: vec![pt] };

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 1,
                ..Default::default()
            },
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(150));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            vcpu_timing: Some(&vcpu_timing),
            preemption_threshold_ns: 10_000_000,
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(30),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();
        sleeper_kill.store(true, Ordering::Release);
        let _ = sleeper.join();

        assert!(
            samples.len() >= 2,
            "need >= 2 samples, got {}",
            samples.len()
        );
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        assert_eq!(dump_byte, 0, "preempted vCPU should not trigger stall dump");
    }

    #[test]
    fn monitor_loop_vcpu_timing_running_stall_fires() {
        // Busy-spinning thread: accumulates CPU time every interval.
        // 30ms interval ensures spinner clears the 10ms preemption
        // threshold with margin.
        // rq_clock stuck + CPU time advancing = real stall. Explicit
        // threshold (10ms) avoids host CONFIG_HZ dependency (CONFIG_HZ=250
        // gives 40ms threshold, which would mask 30ms of spin time).
        let offsets = test_offsets();
        let buf = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let shm_pa = buf.len() as u64;
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let spinner_kill = std::sync::Arc::new(AtomicBool::new(false));
        let spinner_kill_clone = spinner_kill.clone();
        let spinner = std::thread::Builder::new()
            .name("vcpu-spinner".into())
            .spawn(move || {
                while !spinner_kill_clone.load(Ordering::Relaxed) {
                    std::hint::spin_loop();
                }
            })
            .unwrap();

        let pt = spinner.as_pthread_t() as libc::pthread_t;
        let vcpu_timing = VcpuTiming { pthreads: vec![pt] };

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds: super::super::MonitorThresholds {
                max_imbalance_ratio: 100.0,
                max_local_dsq_depth: 10000,
                fail_on_stall: true,
                sustained_samples: 2,
                ..Default::default()
            },
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            vcpu_timing: Some(&vcpu_timing),
            preemption_threshold_ns: 10_000_000,
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(30),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();
        spinner_kill.store(true, Ordering::Release);
        let _ = spinner.join();

        assert!(
            samples.len() >= 3,
            "need >= 3 samples for 2 stall pairs, got {}",
            samples.len()
        );
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        assert_eq!(
            dump_byte,
            crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D,
            "real stall (vCPU running, clock stuck, nr_running>0) should trigger dump"
        );
    }

    #[test]
    fn reactive_and_evaluate_stall_consistency() {
        // Verify that the reactive path (monitor_loop with dump_trigger)
        // and the post-hoc path (evaluate) agree on stall detection.
        // Build a scenario where stall fires: stuck rq_clock, nr_running>0,
        // sustained_samples=2.
        // Two CPUs: cpu0 stuck (rq_clock=5000), cpu1 advancing (rq_clock
        // changes each sample because it reads from a different rq buffer).
        // This ensures data_looks_valid passes in evaluate.
        let offsets = test_offsets();
        let buf0 = make_rq_buffer(&offsets, 2, 1, 1, 5000, 0);
        let buf1 = make_rq_buffer(&offsets, 1, 1, 1, 9000, 0);
        let pa1 = buf0.len() as u64;
        let mut combined = buf0;
        combined.extend_from_slice(&buf1);
        let shm_pa = combined.len() as u64;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let thresholds = super::super::MonitorThresholds {
            max_imbalance_ratio: 100.0,
            max_local_dsq_depth: 10000,
            fail_on_stall: true,
            sustained_samples: 2,
            ..Default::default()
        };

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 3,
            "need >= 3 samples, got {}",
            samples.len()
        );

        // Reactive path result: check if dump fired.
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        let reactive_stall = dump_byte == crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D;

        // Post-hoc evaluate path on the same samples.
        let summary = super::super::MonitorSummary::from_samples(&samples);
        let report = super::super::MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let verdict = thresholds.evaluate(&report);

        // Both paths should agree: stall detected on cpu0.
        assert!(reactive_stall, "reactive path should detect stall");
        assert!(
            !verdict.passed,
            "evaluate should detect stall: {:?}",
            verdict.details
        );
        assert!(
            verdict.details.iter().any(|d| d.contains("rq_clock stall")),
            "evaluate details should mention stall: {:?}",
            verdict.details
        );
    }

    #[test]
    fn reactive_and_evaluate_idle_consistency() {
        // Both reactive and evaluate should agree: idle CPU is exempt.
        let offsets = test_offsets();
        // nr_running=0, rq_clock stuck.
        let buf = make_rq_buffer(&offsets, 0, 0, 0, 5000, 0);
        let shm_pa = buf.len() as u64;
        let mut combined = buf;
        combined.extend(vec![0u8; 64]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let thresholds = super::super::MonitorThresholds {
            max_imbalance_ratio: 100.0,
            max_local_dsq_depth: 10000,
            fail_on_stall: true,
            sustained_samples: 1,
            ..Default::default()
        };

        let trigger = DumpTrigger {
            shm_base_pa: shm_pa,
            thresholds,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(200));
                kill.store(true, Ordering::Release);
            })
        };

        let cfg = MonitorConfig {
            dump_trigger: Some(&trigger),
            ..test_config()
        };
        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &cfg,
        );
        handle.join().unwrap();

        assert!(
            samples.len() >= 2,
            "need >= 2 samples, got {}",
            samples.len()
        );

        // Reactive: dump should NOT fire.
        let dump_byte = combined[shm_pa as usize + crate::vmm::shm_ring::DUMP_REQ_OFFSET];
        assert_eq!(
            dump_byte, 0,
            "reactive: idle CPU should not trigger stall dump"
        );

        // Evaluate: from_samples should not detect stall.
        let summary = super::super::MonitorSummary::from_samples(&samples);
        assert!(
            !summary.stall_detected,
            "from_samples: idle CPU should not flag stall"
        );

        // Evaluate verdict: should pass (no stall on idle CPU).
        // Note: evaluate may pass via data_looks_valid returning false
        // (all-same clocks with single CPU) — that's consistent behavior.
        let report = super::super::MonitorReport {
            samples,
            summary,
            ..Default::default()
        };
        let verdict = thresholds.evaluate(&report);
        assert!(
            verdict.passed,
            "evaluate: idle CPU should pass: {:?}",
            verdict.details
        );
    }

    fn test_schedstat_offsets() -> super::super::btf_offsets::SchedstatOffsets {
        super::super::btf_offsets::SchedstatOffsets {
            rq_sched_info: 200,
            sched_info_run_delay: 8,
            sched_info_pcount: 0,
            rq_yld_count: 300,
            rq_sched_count: 304,
            rq_sched_goidle: 308,
            rq_ttwu_count: 312,
            rq_ttwu_local: 316,
        }
    }

    /// Build a byte buffer simulating a struct rq with schedstat fields.
    #[allow(clippy::too_many_arguments)]
    fn make_schedstat_buffer(
        ss: &super::super::btf_offsets::SchedstatOffsets,
        run_delay: u64,
        pcount: u64,
        yld_count: u32,
        sched_count: u32,
        sched_goidle: u32,
        ttwu_count: u32,
        ttwu_local: u32,
    ) -> Vec<u8> {
        let size = ss.rq_ttwu_local + 4 + 8;
        let mut buf = vec![0u8; size];

        let si_base = ss.rq_sched_info;
        buf[si_base + ss.sched_info_pcount..si_base + ss.sched_info_pcount + 8]
            .copy_from_slice(&pcount.to_ne_bytes());
        buf[si_base + ss.sched_info_run_delay..si_base + ss.sched_info_run_delay + 8]
            .copy_from_slice(&run_delay.to_ne_bytes());

        buf[ss.rq_yld_count..ss.rq_yld_count + 4].copy_from_slice(&yld_count.to_ne_bytes());
        buf[ss.rq_sched_count..ss.rq_sched_count + 4].copy_from_slice(&sched_count.to_ne_bytes());
        buf[ss.rq_sched_goidle..ss.rq_sched_goidle + 4]
            .copy_from_slice(&sched_goidle.to_ne_bytes());
        buf[ss.rq_ttwu_count..ss.rq_ttwu_count + 4].copy_from_slice(&ttwu_count.to_ne_bytes());
        buf[ss.rq_ttwu_local..ss.rq_ttwu_local + 4].copy_from_slice(&ttwu_local.to_ne_bytes());
        buf
    }

    #[test]
    fn read_rq_schedstat_known_values() {
        let ss = test_schedstat_offsets();
        let buf = make_schedstat_buffer(&ss, 50000, 10, 3, 100, 20, 80, 40);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let stats = read_rq_schedstat(&mem, 0, &ss);
        assert_eq!(stats.run_delay, 50000);
        assert_eq!(stats.pcount, 10);
        assert_eq!(stats.yld_count, 3);
        assert_eq!(stats.sched_count, 100);
        assert_eq!(stats.sched_goidle, 20);
        assert_eq!(stats.ttwu_count, 80);
        assert_eq!(stats.ttwu_local, 40);
    }

    #[test]
    fn read_rq_schedstat_zeros() {
        let ss = test_schedstat_offsets();
        let buf = make_schedstat_buffer(&ss, 0, 0, 0, 0, 0, 0, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let stats = read_rq_schedstat(&mem, 0, &ss);
        assert_eq!(stats.run_delay, 0);
        assert_eq!(stats.pcount, 0);
        assert_eq!(stats.yld_count, 0);
        assert_eq!(stats.sched_count, 0);
        assert_eq!(stats.sched_goidle, 0);
        assert_eq!(stats.ttwu_count, 0);
        assert_eq!(stats.ttwu_local, 0);
    }

    #[test]
    fn monitor_loop_with_schedstat_overlay() {
        let ss = test_schedstat_offsets();
        let mut offsets = test_offsets();
        offsets.schedstat_offsets = Some(ss.clone());

        // Build a buffer that contains both rq fields and schedstat fields.
        // The rq buffer must be large enough to cover schedstat offsets.
        let rq_size = ss.rq_ttwu_local + 4 + 8;
        let mut buf = vec![0u8; rq_size];

        // Write rq base fields.
        buf[offsets.rq_nr_running..offsets.rq_nr_running + 4].copy_from_slice(&2u32.to_ne_bytes());
        buf[offsets.rq_clock..offsets.rq_clock + 8].copy_from_slice(&500u64.to_ne_bytes());

        // Write schedstat fields.
        let si_base = ss.rq_sched_info;
        buf[si_base + ss.sched_info_run_delay..si_base + ss.sched_info_run_delay + 8]
            .copy_from_slice(&12345u64.to_ne_bytes());
        buf[si_base + ss.sched_info_pcount..si_base + ss.sched_info_pcount + 8]
            .copy_from_slice(&7u64.to_ne_bytes());
        buf[ss.rq_sched_count..ss.rq_sched_count + 4].copy_from_slice(&42u32.to_ne_bytes());

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        let ss_snap = samples[0].cpus[0].schedstat.as_ref().unwrap();
        assert_eq!(ss_snap.run_delay, 12345);
        assert_eq!(ss_snap.pcount, 7);
        assert_eq!(ss_snap.sched_count, 42);
    }

    #[test]
    fn monitor_loop_no_schedstat_when_none() {
        let offsets = test_offsets();
        assert!(offsets.schedstat_offsets.is_none());

        let buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let MonitorLoopResult { samples, .. } = monitor_loop(
            &mem,
            &[0],
            &offsets,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            &test_config(),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        assert!(samples[0].cpus[0].schedstat.is_none());
    }

    fn test_sched_domain_offsets() -> SchedDomainOffsets {
        // Synthetic offsets for a sched_domain struct.
        // Layout: parent(0) level(8) flags(12) name(16) span_weight(24)
        //         balance_interval(28) nr_balance_failed(32)
        //         newidle_call(36) newidle_success(40) newidle_ratio(44)
        //         max_newidle_lb_cost(48)
        //         [stats at 56+]
        SchedDomainOffsets {
            rq_sd: 400,
            sd_parent: 0,
            sd_level: 8,
            sd_flags: 12,
            sd_name: 16,
            sd_span_weight: 24,
            sd_balance_interval: 28,
            sd_nr_balance_failed: 32,
            sd_newidle_call: Some(36),
            sd_newidle_success: Some(40),
            sd_newidle_ratio: Some(44),
            sd_max_newidle_lb_cost: 48,
            stats_offsets: Some(test_sd_stats_offsets()),
        }
    }

    fn test_sd_stats_offsets() -> SchedDomainStatsOffsets {
        SchedDomainStatsOffsets {
            sd_lb_count: 56,
            sd_lb_failed: 68,
            sd_lb_balanced: 80,
            sd_lb_imbalance_load: 92,
            sd_lb_imbalance_util: 104,
            sd_lb_imbalance_task: 116,
            sd_lb_imbalance_misfit: 128,
            sd_lb_gained: 140,
            sd_lb_hot_gained: 152,
            sd_lb_nobusyg: 164,
            sd_lb_nobusyq: 176,
            sd_alb_count: 188,
            sd_alb_failed: 192,
            sd_alb_pushed: 196,
            sd_sbe_count: 200,
            sd_sbe_balanced: 204,
            sd_sbe_pushed: 208,
            sd_sbf_count: 212,
            sd_sbf_balanced: 216,
            sd_sbf_pushed: 220,
            sd_ttwu_wake_remote: 224,
            sd_ttwu_move_affine: 228,
            sd_ttwu_move_balance: 232,
        }
    }

    /// Build a synthetic sched_domain buffer with known values.
    /// `parent_kva`: KVA of parent domain (0 = no parent).
    /// `name_kva`: KVA of name string (0 = no name).
    /// Returns a buffer representing one sched_domain struct.
    #[allow(clippy::too_many_arguments)]
    fn make_sd_buffer(
        sd: &SchedDomainOffsets,
        parent_kva: u64,
        level: i32,
        flags: i32,
        name_kva: u64,
        span_weight: u32,
        balance_interval: u32,
        newidle_call: u32,
        lb_count_0: u32,
        alb_pushed: u32,
        ttwu_wake_remote: u32,
    ) -> Vec<u8> {
        // Size must cover the highest offset used.
        let so = sd.stats_offsets.as_ref().unwrap();
        let size = so.sd_ttwu_move_balance + 4 + 8;
        let mut buf = vec![0u8; size];

        buf[sd.sd_parent..sd.sd_parent + 8].copy_from_slice(&parent_kva.to_ne_bytes());
        buf[sd.sd_level..sd.sd_level + 4].copy_from_slice(&level.to_ne_bytes());
        buf[sd.sd_flags..sd.sd_flags + 4].copy_from_slice(&flags.to_ne_bytes());
        buf[sd.sd_name..sd.sd_name + 8].copy_from_slice(&name_kva.to_ne_bytes());
        buf[sd.sd_span_weight..sd.sd_span_weight + 4].copy_from_slice(&span_weight.to_ne_bytes());
        buf[sd.sd_balance_interval..sd.sd_balance_interval + 4]
            .copy_from_slice(&balance_interval.to_ne_bytes());
        if let Some(off) = sd.sd_newidle_call {
            buf[off..off + 4].copy_from_slice(&newidle_call.to_ne_bytes());
        }
        buf[so.sd_lb_count..so.sd_lb_count + 4].copy_from_slice(&lb_count_0.to_ne_bytes());
        buf[so.sd_alb_pushed..so.sd_alb_pushed + 4].copy_from_slice(&alb_pushed.to_ne_bytes());
        buf[so.sd_ttwu_wake_remote..so.sd_ttwu_wake_remote + 4]
            .copy_from_slice(&ttwu_wake_remote.to_ne_bytes());
        buf
    }

    #[test]
    fn read_sched_domain_tree_null_sd() {
        // rq->sd is null — should return None.
        let sd_off = test_sched_domain_offsets();
        let buf = vec![0u8; 512];
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let result = read_sched_domain_tree(&mem, 0, &sd_off, 0);
        assert!(result.is_none());
    }

    #[test]
    fn read_sched_domain_tree_single_domain() {
        let sd_off = test_sched_domain_offsets();

        // Build: rq at PA 0 with rq->sd pointing to a domain.
        // Domain at some offset in the buffer, parent=0 (no parent).
        // page_offset=0 so KVA == PA for testing.
        let sd_pa: u64 = 1024;
        let name_pa: u64 = 2048;

        let sd_buf = make_sd_buffer(&sd_off, 0, 0, 0x42, name_pa, 4, 64, 15, 10, 3, 7);
        let name_bytes = b"SMT\0";

        // Build combined buffer: rq region + sd region + name region.
        let total_size = (name_pa as usize) + 16;
        let mut buf = vec![0u8; total_size];

        // Write rq->sd pointer (KVA == PA since page_offset=0).
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd_pa.to_ne_bytes());

        // Write sched_domain at sd_pa.
        buf[sd_pa as usize..sd_pa as usize + sd_buf.len()].copy_from_slice(&sd_buf);

        // Write name string.
        buf[name_pa as usize..name_pa as usize + name_bytes.len()].copy_from_slice(name_bytes);

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0).unwrap();

        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].level, 0);
        assert_eq!(domains[0].name, "SMT");
        assert_eq!(domains[0].flags, 0x42);
        assert_eq!(domains[0].span_weight, 4);
        assert_eq!(domains[0].balance_interval, 64);
        assert_eq!(domains[0].newidle_call, Some(15));
        let stats = domains[0].stats.as_ref().unwrap();
        assert_eq!(stats.lb_count[0], 10);
        assert_eq!(stats.alb_pushed, 3);
        assert_eq!(stats.ttwu_wake_remote, 7);
    }

    #[test]
    fn read_sched_domain_tree_two_levels() {
        let sd_off = test_sched_domain_offsets();

        // page_offset=0 so KVA == PA.
        let sd0_pa: u64 = 1024;
        let sd1_pa: u64 = 2048;
        let name0_pa: u64 = 3072;
        let name1_pa: u64 = 3088;

        // Domain 0 (SMT, level 0) -> parent = Domain 1
        let sd0_buf = make_sd_buffer(&sd_off, sd1_pa, 0, 0x10, name0_pa, 2, 32, 8, 5, 1, 2);
        // Domain 1 (MC, level 1) -> parent = 0 (top)
        let sd1_buf = make_sd_buffer(&sd_off, 0, 1, 0x20, name1_pa, 8, 128, 22, 20, 4, 10);

        let total_size = 3104;
        let mut buf = vec![0u8; total_size];

        // rq->sd -> domain 0
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd0_pa.to_ne_bytes());
        buf[sd0_pa as usize..sd0_pa as usize + sd0_buf.len()].copy_from_slice(&sd0_buf);
        buf[sd1_pa as usize..sd1_pa as usize + sd1_buf.len()].copy_from_slice(&sd1_buf);
        buf[name0_pa as usize..name0_pa as usize + 4].copy_from_slice(b"SMT\0");
        buf[name1_pa as usize..name1_pa as usize + 3].copy_from_slice(b"MC\0");

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0).unwrap();

        assert_eq!(domains.len(), 2);
        // First = lowest level (SMT).
        assert_eq!(domains[0].level, 0);
        assert_eq!(domains[0].name, "SMT");
        assert_eq!(domains[0].span_weight, 2);
        assert_eq!(domains[0].balance_interval, 32);
        assert_eq!(domains[0].newidle_call, Some(8));
        let s0 = domains[0].stats.as_ref().unwrap();
        assert_eq!(s0.lb_count[0], 5);
        // Second = higher level (MC).
        assert_eq!(domains[1].level, 1);
        assert_eq!(domains[1].name, "MC");
        assert_eq!(domains[1].span_weight, 8);
        assert_eq!(domains[1].balance_interval, 128);
        assert_eq!(domains[1].newidle_call, Some(22));
        let s1 = domains[1].stats.as_ref().unwrap();
        assert_eq!(s1.lb_count[0], 20);
        assert_eq!(s1.alb_pushed, 4);
        assert_eq!(s1.ttwu_wake_remote, 10);
    }

    #[test]
    fn read_sched_domain_tree_max_depth_bound() {
        let sd_off = test_sched_domain_offsets();

        // Create a circular chain: sd->parent points to itself.
        // The MAX_DEPTH bound (8) should prevent infinite loop.
        let sd_pa: u64 = 1024;
        // Self-referential: parent == self.
        let sd_buf = make_sd_buffer(&sd_off, sd_pa, 0, 0, 0, 1, 0, 0, 0, 0, 0);

        let total_size = sd_pa as usize + sd_buf.len();
        let mut buf = vec![0u8; total_size];
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd_pa.to_ne_bytes());
        buf[sd_pa as usize..sd_pa as usize + sd_buf.len()].copy_from_slice(&sd_buf);

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0).unwrap();

        // Should stop at MAX_DEPTH=8.
        assert_eq!(domains.len(), 8);
    }

    #[test]
    fn read_sched_domain_tree_out_of_bounds_pa() {
        let sd_off = test_sched_domain_offsets();

        // rq->sd points to a KVA that translates to a PA beyond guest memory.
        let bad_kva: u64 = 0xFFFF_FFFF_FFFF_0000;
        let mut buf = vec![0u8; 512];
        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&bad_kva.to_ne_bytes());

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        // page_offset=0 -> PA = bad_kva which is > buf.len().
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0);

        // Should return Some(empty vec) — non-null sd but untranslatable.
        assert!(domains.is_some());
        assert!(domains.unwrap().is_empty());
    }

    #[test]
    fn read_sched_domain_tree_newidle_none() {
        // 6.16 kernel: newidle_call/success/ratio are absent.
        // Other fields (level, name, span_weight, balance_interval) must
        // still populate correctly.
        let mut sd_off = test_sched_domain_offsets();
        sd_off.sd_newidle_call = None;
        sd_off.sd_newidle_success = None;
        sd_off.sd_newidle_ratio = None;

        let sd_pa: u64 = 1024;
        let name_pa: u64 = 2048;

        let sd_buf = make_sd_buffer(&sd_off, 0, 0, 0x42, name_pa, 4, 64, 0, 10, 3, 7);
        let name_bytes = b"SMT\0";

        let total_size = (name_pa as usize) + 16;
        let mut buf = vec![0u8; total_size];

        buf[sd_off.rq_sd..sd_off.rq_sd + 8].copy_from_slice(&sd_pa.to_ne_bytes());
        buf[sd_pa as usize..sd_pa as usize + sd_buf.len()].copy_from_slice(&sd_buf);
        buf[name_pa as usize..name_pa as usize + name_bytes.len()].copy_from_slice(name_bytes);

        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let domains = read_sched_domain_tree(&mem, 0, &sd_off, 0).unwrap();

        assert_eq!(domains.len(), 1);
        assert_eq!(domains[0].level, 0);
        assert_eq!(domains[0].name, "SMT");
        assert_eq!(domains[0].flags, 0x42);
        assert_eq!(domains[0].span_weight, 4);
        assert_eq!(domains[0].balance_interval, 64);
        assert_eq!(domains[0].newidle_call, None);
        assert_eq!(domains[0].newidle_success, None);
        assert_eq!(domains[0].newidle_ratio, None);
        let stats = domains[0].stats.as_ref().unwrap();
        assert_eq!(stats.lb_count[0], 10);
        assert_eq!(stats.alb_pushed, 3);
        assert_eq!(stats.ttwu_wake_remote, 7);
    }

    #[test]
    fn read_u32_array_known_values() {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&10u32.to_ne_bytes());
        buf[4..8].copy_from_slice(&20u32.to_ne_bytes());
        buf[8..12].copy_from_slice(&30u32.to_ne_bytes());
        let mem = GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64);
        let arr = read_u32_array(&mem, 0, 0);
        assert_eq!(arr, [10, 20, 30]);
    }
}
