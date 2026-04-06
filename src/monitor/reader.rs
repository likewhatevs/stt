//! Guest physical memory access and monitor sampling loop.
//!
//! [`GuestMem`] wraps a host pointer to guest physical address 0 and
//! provides bounds-checked volatile reads and writes for scalar types;
//! `read_bytes` uses `copy_nonoverlapping` for bulk copies. It also implements
//! 4-level and 5-level x86-64 page table walks for vmalloc'd addresses.
//!
//! The monitor loop (`monitor_loop`) periodically reads per-CPU
//! runqueue state from guest memory and collects `MonitorSample`s.

use super::btf_offsets::{KernelOffsets, ScxEventOffsets};
use super::{CpuSnapshot, MonitorSample, ScxEventCounters};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Host pointer to guest physical address 0. Guest PA `n` is at `host_base + n`.
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

    /// Raw pointer to guest physical address 0.
    pub fn base_ptr(&self) -> *const u8 {
        self.base
    }

    /// Read a u32 at guest physical address `pa + offset`.
    pub fn read_u32(&self, pa: u64, offset: usize) -> u32 {
        let addr = pa + offset as u64;
        if addr + 4 > self.size {
            return 0;
        }
        unsafe { std::ptr::read_volatile(self.base.add(addr as usize) as *const u32) }
    }

    /// Read a u64 at guest physical address `pa + offset`.
    pub fn read_u64(&self, pa: u64, offset: usize) -> u64 {
        let addr = pa + offset as u64;
        if addr + 8 > self.size {
            return 0;
        }
        unsafe { std::ptr::read_volatile(self.base.add(addr as usize) as *const u64) }
    }

    /// Read an i64 at guest physical address `pa + offset`.
    pub fn read_i64(&self, pa: u64, offset: usize) -> i64 {
        self.read_u64(pa, offset) as i64
    }

    /// Write a u8 at guest physical address `pa + offset`.
    pub fn write_u8(&self, pa: u64, offset: usize, val: u8) {
        let addr = pa + offset as u64;
        if addr + 1 > self.size {
            return;
        }
        unsafe { std::ptr::write_volatile(self.base.add(addr as usize), val) }
    }

    /// Write a u64 at guest physical address `pa + offset`.
    pub fn write_u64(&self, pa: u64, offset: usize, val: u64) {
        let addr = pa + offset as u64;
        if addr + 8 > self.size {
            return;
        }
        unsafe { std::ptr::write_volatile(self.base.add(addr as usize) as *mut u64, val) }
    }

    /// Read a u8 at guest physical address `pa + offset`.
    pub fn read_u8(&self, pa: u64, offset: usize) -> u8 {
        let addr = pa + offset as u64;
        if addr + 1 > self.size {
            return 0;
        }
        unsafe { std::ptr::read_volatile(self.base.add(addr as usize)) }
    }

    /// Read `len` bytes from guest physical address `pa` into `buf`.
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

    /// Write a u32 at guest physical address `pa + offset`.
    pub fn write_u32(&self, pa: u64, offset: usize, val: u32) {
        let addr = pa + offset as u64;
        if addr + 4 > self.size {
            return;
        }
        unsafe { std::ptr::write_volatile(self.base.add(addr as usize) as *mut u32, val) }
    }

    /// Translate a kernel virtual address to guest physical address via
    /// page table walk. Supports both 4-level (PGD -> PUD -> PMD -> PTE)
    /// and 5-level (PML5 -> P4D -> PUD -> PMD -> PTE) paging.
    ///
    /// `cr3_pa` is the physical address of the top-level page table.
    /// `l5` selects 5-level paging (LA57); use `resolve_pgtable_l5`
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

    /// 4-level page table walk: CR3 -> PGD -> PUD -> PMD -> PTE.
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

    /// 5-level page table walk: CR3 -> PML5 -> P4D -> PUD -> PMD -> PTE.
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
    }
}

/// Read scx event counters from one CPU's `scx_sched_pcpu` at the given PA.
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

/// Resolve per-CPU physical addresses for `scx_sched_pcpu`.
///
/// Reads the `scx_root` pointer from guest memory, then reads the
/// `pcpu` percpu pointer from the `scx_sched` struct, then computes
/// each CPU's `scx_sched_pcpu` PA via `__per_cpu_offset`.
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

    let scx_sched_pa = scx_sched_kva.wrapping_sub(page_offset);
    let pcpu_kva = mem.read_u64(scx_sched_pa, ev.scx_sched_pcpu_off);
    if pcpu_kva == 0 {
        return None;
    }

    let pas: Vec<u64> = per_cpu_offsets
        .iter()
        .map(|&cpu_off| pcpu_kva.wrapping_add(cpu_off).wrapping_sub(page_offset))
        .collect();

    Some(pas)
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

/// Override for scx_watchdog_timeout written every monitor iteration.
///
/// The kernel sets scx_watchdog_timeout during scheduler attach, so a
/// one-time write before the scheduler starts is overwritten. Writing
/// every iteration ensures the override takes effect after attachment.
pub(crate) struct WatchdogOverride {
    pub pa: u64,
    pub jiffies: u64,
}

/// Run the monitor loop, sampling all CPUs at the given interval.
/// Returns collected samples when `kill` is set.
///
/// `event_pcpu_pas`: optional per-CPU physical addresses of `scx_sched_pcpu`.
/// When present (and `event_offsets` exist), each sample includes event counters.
///
/// `dump_trigger`: optional reactive dump configuration. When a sustained
/// threshold violation is detected, writes the dump request flag to guest
/// SHM to trigger a SysRq-D dump inside the guest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn monitor_loop(
    mem: &GuestMem,
    rq_pas: &[u64],
    offsets: &KernelOffsets,
    event_pcpu_pas: Option<&[u64]>,
    interval: Duration,
    kill: &AtomicBool,
    start: Instant,
    dump_trigger: Option<&DumpTrigger>,
    watchdog_override: Option<&WatchdogOverride>,
) -> Vec<MonitorSample> {
    let mut samples = Vec::new();
    let mut consecutive_imbalance = 0usize;
    let mut consecutive_dsq = 0usize;
    let mut dump_requested = false;
    let mut cpus: Vec<CpuSnapshot> = Vec::with_capacity(rq_pas.len());

    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        if let Some(wd) = watchdog_override {
            mem.write_u64(wd.pa, 0, wd.jiffies);
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

            // Stall check between last two samples.
            let stall = t.fail_on_stall
                && samples.last().is_some_and(|prev: &MonitorSample| {
                    let n = prev.cpus.len().min(cpus.len());
                    (0..n)
                        .any(|i| cpus[i].rq_clock != 0 && cpus[i].rq_clock == prev.cpus[i].rq_clock)
                });

            let sustained = consecutive_imbalance >= t.sustained_samples
                || consecutive_dsq >= t.sustained_samples
                || stall;

            if sustained {
                mem.write_u8(
                    trigger.shm_base_pa,
                    crate::vmm::shm_ring::DUMP_REQ_OFFSET,
                    crate::vmm::shm_ring::DUMP_REQ_SYSRQ_D,
                );
                dump_requested = true;
            }
        }

        samples.push(MonitorSample {
            elapsed_ms: start.elapsed().as_millis() as u64,
            cpus: cpus.clone(),
        });
        std::thread::sleep(interval);
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let samples = monitor_loop(
            &mem,
            &[0],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            None,
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

        let samples = monitor_loop(
            &mem,
            &[0],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            None,
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

        let samples = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            None,
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
                std::thread::sleep(Duration::from_millis(80));
                kill.store(true, Ordering::Release);
            })
        };

        let samples = monitor_loop(
            &mem,
            &[0],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            None,
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
        let samples = monitor_loop(
            &mem,
            &[rq_pa],
            &offsets,
            Some(&ev_pas),
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            None,
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

        let samples = monitor_loop(
            &mem,
            &[0],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            None,
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
        // Buffer: rq data + 8 bytes for watchdog value.
        let rq_buf = make_rq_buffer(&offsets, 1, 1, 1, 100, 0);
        let wd_pa = rq_buf.len() as u64;
        let mut combined = rq_buf;
        combined.extend_from_slice(&[0u8; 8]);

        let mem = GuestMem::new(combined.as_mut_ptr(), combined.len() as u64);
        let kill = std::sync::Arc::new(AtomicBool::new(false));

        let wd = WatchdogOverride {
            pa: wd_pa,
            jiffies: 99999,
        };

        let handle = {
            let kill = std::sync::Arc::clone(&kill);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                kill.store(true, Ordering::Release);
            })
        };

        let samples = monitor_loop(
            &mem,
            &[0],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            None,
            Some(&wd),
        );
        handle.join().unwrap();

        assert!(!samples.is_empty());
        // Verify the watchdog value was written.
        let written = u64::from_ne_bytes(
            combined[wd_pa as usize..wd_pa as usize + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(written, 99999);
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
                std::thread::sleep(Duration::from_millis(80));
                kill.store(true, Ordering::Release);
            })
        };

        let samples = monitor_loop(
            &mem,
            &[0, pa1],
            &offsets,
            None,
            Duration::from_millis(10),
            &kill,
            Instant::now(),
            Some(&trigger),
            None,
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
}
