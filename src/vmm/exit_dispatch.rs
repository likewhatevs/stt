//! vCPU exit classification and per-arch I/O dispatch.
//!
//! Shared between BSP and AP run loops. Each exit gets classified into
//! [`ExitAction`] (Continue / Shutdown / Fatal); arch-specific I/O is
//! dispatched inline so the surrounding loop only sees the action.
//!
//! - x86_64: serial via port I/O ([`dispatch_io_out`] / [`dispatch_io_in`]),
//!   virtio-console via MMIO inside [`classify_exit`], i8042 reset for reboot.
//! - aarch64: serial + virtio-console both via MMIO ([`dispatch_mmio_write`]
//!   / [`dispatch_mmio_read`]).

use crate::vmm::PiMutex;
use crate::vmm::vcpu::{SCX_EXIT_ERROR_THRESHOLD, WatchpointArm, self_arm_watchpoint};
use crate::vmm::{console, kvm, virtio_blk, virtio_console, virtio_net};
use kvm_ioctls::VcpuExit;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use vmm_sys_util::eventfd::EventFd;

/// Snapshot of a vCPU's architectural state, captured by the vCPU
/// thread itself at freeze time (just before it parks). Surfaced in
/// the failure-dump report so an operator can correlate the observed
/// guest-memory state with where each vCPU was executing.
///
/// Field naming is arch-neutral: each value is set from the matching
/// per-arch register so the layout is identical across x86_64 and
/// aarch64 in JSON / Display output.
///
/// Capture must run ON the vCPU thread (not cross-thread) because
/// `KVM_GET_REGS` / `KVM_GET_SREGS` are vCPU-fd-bound ioctls; their
/// thread-affinity is a KVM API contract documented in the kernel
/// vCPU lifecycle (Documentation/virt/kvm/api.rst). Calling them
/// from a different thread reads stale state at best and races KVM's
/// internal vCPU state machine at worst.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub struct VcpuRegSnapshot {
    /// Instruction pointer at freeze time (`rip` on x86_64,
    /// `pc` on aarch64). Identifies the kernel/userspace function
    /// the vCPU was executing when the freeze coordinator's kick
    /// arrived.
    pub instruction_pointer: u64,
    /// Kernel-side stack pointer at freeze time (`rsp` on x86_64,
    /// `sp_el1` on aarch64 — explicitly NOT `sp_el0`, which is the
    /// userspace stack). Captures the EL1/CPL0 stack frame an
    /// operator can unwind against the BPF map dump for sched_ext
    /// failures, which fire in kernel context.
    pub stack_pointer: u64,
    /// Page-table root at freeze time. Captures arch-specific
    /// kernel-side state suitable for correlating the BPF map
    /// dump with the active address space:
    ///
    ///   - On x86_64: `cr3` — per-process pgd. Distinct from
    ///     [`crate::monitor::guest::GuestKernel::cr3_pa`], which
    ///     captures the boot-time `init_top_pgt` at coordinator
    ///     start. This snapshot field reflects whatever pgd the
    ///     vCPU was running on at freeze time (typically the
    ///     current task's mm); the boot-time value is what the
    ///     freeze coordinator uses for its own page-walks.
    ///
    ///   - On aarch64: `ttbr1_el1` — the kernel pgd. Stays
    ///     stable across context switches (TTBR0_EL1 swaps
    ///     per-task; see [`Self::user_page_table_root`] for the
    ///     userspace half).
    ///
    /// Raw register value with arch-specific flag bits intact
    /// (PCID/PCD/PWT on x86_64 CR3, ASID on aarch64 TTBR);
    /// consumers must mask before walking as a physical address.
    pub page_table_root: u64,
    /// Userspace page-table root at freeze time. arch-specific:
    ///
    ///   - On x86_64: always `None`. CR3 already covers both the
    ///     kernel and userspace halves of the address space —
    ///     [`Self::page_table_root`] alone identifies the active
    ///     mm.
    ///
    ///   - On aarch64: `Some(ttbr0_el1)` when capture succeeds,
    ///     `None` when KVM_GET_ONE_REG fails (mid-shutdown vCPU,
    ///     sysreg gated by the host kernel). TTBR0_EL1 holds the
    ///     userspace pgd that switches per-task, so it
    ///     identifies which task's userspace was active at
    ///     freeze time — useful for diagnosing user-context
    ///     traps. For sched_ext failures (kernel context),
    ///     TTBR1_EL1 in `page_table_root` is the primary signal.
    ///
    /// Raw register value with arch-specific flag bits intact
    /// (PCID/PCD/PWT on x86_64 CR3, ASID on aarch64 TTBR);
    /// consumers must mask before walking as a physical address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_page_table_root: Option<u64>,
    /// aarch64 TCR_EL1 register at freeze time. Drives the
    /// granule-agnostic page-table walker
    /// ([`crate::monitor::reader::GuestMem::translate_kva`]):
    /// TG1 bits [31:30] select the high-half granule (4 KB / 16 KB
    /// / 64 KB) and T1SZ bits [21:16] determine the high-half VA
    /// width (`64 - T1SZ`). Stable after kernel MMU bring-up.
    /// `None` on x86_64 (the register does not exist) and on
    /// aarch64 if the KVM_GET_ONE_REG read fails mid-shutdown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcr_el1: Option<u64>,
}

/// Capture the vCPU's RIP/RSP/CR3 (or PC/SP/TTBR1 on aarch64) on
/// the calling thread. Invoked from `handle_freeze` after the drain
/// dance and before the `parked = true` Release store, so the
/// values reach the freeze coordinator via the same happens-before
/// edge the coordinator relies on for guest-memory reads.
///
/// `None` on capture failure — the get_regs / get_one_reg ioctls
/// can fail mid-shutdown when KVM has begun tearing down the vCPU.
/// The caller stores `None` into the per-vCPU slot in that case;
/// the dump reflects "registers unavailable" rather than panicking
/// the freeze path.
#[cfg(target_arch = "x86_64")]
pub(crate) fn capture_vcpu_regs(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<VcpuRegSnapshot> {
    let regs = vcpu.get_regs().ok()?;
    let sregs = vcpu.get_sregs().ok()?;
    Some(VcpuRegSnapshot {
        instruction_pointer: regs.rip,
        stack_pointer: regs.rsp,
        page_table_root: sregs.cr3,
        // x86_64 has a single CR3 covering both halves of the
        // address space; no separate userspace pgd to capture.
        user_page_table_root: None,
        // TCR_EL1 is an aarch64 register; not present on x86_64.
        tcr_el1: None,
    })
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn capture_vcpu_regs(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<VcpuRegSnapshot> {
    // ARM core register IDs encode
    // `(offsetof(struct kvm_regs, field) / sizeof(u32))` in the low
    // bits, OR'd with KVM_REG_ARM64 + KVM_REG_SIZE_U64 +
    // KVM_REG_ARM_CORE (per kernel uapi/asm/kvm.h
    // `KVM_REG_ARM_CORE_REG` macro). The offset is into
    // `struct kvm_regs`, NOT directly into `struct user_pt_regs`;
    // the two coincide for the first 272 bytes because
    // `kvm_regs.regs` is at offset 0, but adding fields past
    // `user_pt_regs` (e.g. `sp_el1` below) requires the
    // `kvm_regs`-relative encoding.
    //
    // struct kvm_regs (arch/arm64/include/uapi/asm/kvm.h):
    //   struct user_pt_regs regs;     // offset 0..272
    //     u64 regs[31];               //   offset   0..248
    //     u64 sp;       (= sp_el0)    //   offset 248
    //     u64 pc;                     //   offset 256
    //     u64 pstate;                 //   offset 264
    //   u64 sp_el1;                   // offset 272
    //   u64 elr_el1;                  // offset 280
    //   u64 spsr[KVM_NR_SPSR];        // offset 288..
    //   ...
    //
    // The kernel-side stack pointer is `sp_el1`, NOT `regs.sp`
    // (which is `sp_el0` — the userspace stack pointer per the
    // comment at arch/arm64/include/uapi/asm/kvm.h:47). sched_ext
    // exits fire in EL1 (kernel context), so capturing sp_el1
    // yields the kernel stack frame an operator can unwind
    // against the BPF map dump. Capturing sp_el0 here would
    // leak the userspace stack of whatever task happened to be
    // current — typically irrelevant for kernel-side debugging
    // and confusing in the JSON output.
    //
    // Each u32 step is +1 in the encoded ID.
    const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
    const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
    const KVM_REG_ARM_CORE: u64 = 0x0010_0000;
    // SP_EL1 lives at offset 272 in struct kvm_regs (right after
    // the 272-byte user_pt_regs). 272 / 4 = 68.
    const SP_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (272 / 4);
    // PC at offset 256 in user_pt_regs (= same offset in kvm_regs
    // because user_pt_regs.regs is at offset 0). 256 / 4 = 64.
    const PC_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (256 / 4);
    // ARM64 system registers encoded under KVM_REG_ARM64_SYSREG.
    // The 16-bit packing is
    //   (Op0 << 14) | (Op1 << 11) | (CRn << 7) | (CRm << 3) | Op2
    // per arch/arm64/include/uapi/asm/kvm.h `ARM64_SYS_REG` macro.
    const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;
    // TTBR0_EL1: Op0=3, Op1=0, CRn=2, CRm=0, Op2=0
    // = (3 << 14) | (0 << 11) | (2 << 7) | (0 << 3) | 0 = 0xC100
    const TTBR0_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC100;
    // TTBR1_EL1: Op0=3, Op1=0, CRn=2, CRm=0, Op2=1
    // = (3 << 14) | (0 << 11) | (2 << 7) | (0 << 3) | 1 = 0xC101
    const TTBR1_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;
    // TCR_EL1: Op0=3, Op1=0, CRn=2, CRm=0, Op2=2
    // = (3 << 14) | (0 << 11) | (2 << 7) | (0 << 3) | 2 = 0xC102
    const TCR_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC102;

    let mut buf = [0u8; 8];
    let pc = vcpu
        .get_one_reg(PC_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))?;
    let sp = vcpu
        .get_one_reg(SP_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))?;
    // TTBR1 read is best-effort; some kernels gate sysreg access.
    // A failure leaves page_table_root = 0 — the boot-time
    // GuestKernel::cr3_pa is still available to the dump.
    let ttbr1 = vcpu
        .get_one_reg(TTBR1_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))
        .unwrap_or(0);
    // TTBR0 read is best-effort. Stored in user_page_table_root
    // so a failure surfaces as None — distinct from a successful
    // read of 0, which means "no userspace mapping active at
    // freeze time" (e.g. the vCPU was running pure kernel code).
    let ttbr0 = vcpu
        .get_one_reg(TTBR0_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf));
    // TCR_EL1 carries the granule (TG1[31:30]) and high-half VA
    // size (T1SZ[21:16]) the page-table walker needs. Best-effort:
    // a failure leaves None and the walker falls back to the
    // boot-time cached value the freeze coordinator latched at
    // GuestKernel construction.
    let tcr_el1 = vcpu
        .get_one_reg(TCR_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf));
    Some(VcpuRegSnapshot {
        instruction_pointer: pc,
        stack_pointer: sp,
        page_table_root: ttbr1,
        user_page_table_root: ttbr0,
        tcr_el1,
    })
}

/// Read TCR_EL1 directly from a vCPU. Used at GuestKernel
/// construction time to feed the page-table walker its granule and
/// VA-width settings (TG1 in bits [31:30], T1SZ in bits [21:16]).
///
/// Returns `None` on x86_64 (the register does not exist) and on
/// aarch64 if `KVM_GET_ONE_REG` fails. The caller treats `None` as
/// "no walker context yet"; on aarch64 that surfaces as a 0 stored
/// in the GuestKernel's `tcr_el1` field — the walker rejects T1SZ=0
/// and the affected lookups skip cleanly.
#[cfg(target_arch = "x86_64")]
pub(crate) fn read_tcr_el1(_vcpu: &mut kvm_ioctls::VcpuFd) -> Option<u64> {
    None
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn read_tcr_el1(vcpu: &mut kvm_ioctls::VcpuFd) -> Option<u64> {
    // Same encoding constants as `capture_vcpu_regs`. TCR_EL1 packs
    // to (Op0=3, Op1=0, CRn=2, CRm=0, Op2=2) under the
    // KVM_REG_ARM64_SYSREG namespace.
    const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
    const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
    const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;
    const TCR_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC102;
    let mut buf = [0u8; 8];
    vcpu.get_one_reg(TCR_EL1_ID, &mut buf)
        .ok()
        .map(|_| u64::from_le_bytes(buf))
}

impl std::fmt::Display for VcpuRegSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ip=0x{:016x} sp=0x{:016x} ptroot=0x{:016x}",
            self.instruction_pointer, self.stack_pointer, self.page_table_root
        )?;
        // user_page_table_root is x86_64-None / aarch64-Optional;
        // when present, render it inline so an aarch64 failure
        // dump shows both halves of the address space.
        if let Some(uptr) = self.user_page_table_root {
            write!(f, " uptroot=0x{uptr:016x}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// aarch64 MMIO dispatch — serial and virtio over MMIO
// ---------------------------------------------------------------------------

/// Dispatch an MMIO write to serial and virtio devices.
/// Returns `true` if the caller should exit (shutdown detected).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mmio_write(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
    virtio_blk: Option<&PiMutex<virtio_blk::VirtioBlk>>,
    virtio_net: Option<&PiMutex<virtio_net::VirtioNet>>,
    addr: u64,
    data: &[u8],
) -> bool {
    if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL_MMIO_BASE) {
        if let Some(&byte) = data.first() {
            com1.lock().inner_write(offset, byte);
        }
    } else if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL2_MMIO_BASE)
        && let Some(&byte) = data.first()
    {
        com2.lock().inner_write(offset, byte);
    } else if let Some(vc) = virtio_con
        && (kvm::VIRTIO_CONSOLE_MMIO_BASE
            ..kvm::VIRTIO_CONSOLE_MMIO_BASE + virtio_console::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vc.lock()
            .mmio_write(addr - kvm::VIRTIO_CONSOLE_MMIO_BASE, data);
    } else if let Some(vb) = virtio_blk
        && (kvm::VIRTIO_BLK_MMIO_BASE..kvm::VIRTIO_BLK_MMIO_BASE + virtio_blk::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vb.lock().mmio_write(addr - kvm::VIRTIO_BLK_MMIO_BASE, data);
    } else if let Some(vn) = virtio_net
        && (kvm::VIRTIO_NET_MMIO_BASE..kvm::VIRTIO_NET_MMIO_BASE + virtio_net::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vn.lock().mmio_write(addr - kvm::VIRTIO_NET_MMIO_BASE, data);
    }
    false
}

/// Dispatch an MMIO read from serial and virtio-console devices.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_mmio_read(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
    virtio_blk: Option<&PiMutex<virtio_blk::VirtioBlk>>,
    virtio_net: Option<&PiMutex<virtio_net::VirtioNet>>,
    addr: u64,
    data: &mut [u8],
) {
    if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL_MMIO_BASE) {
        if let Some(first) = data.first_mut() {
            *first = com1.lock().inner_read(offset);
        }
    } else if let Some(offset) = mmio_serial_offset(addr, kvm::SERIAL2_MMIO_BASE) {
        if let Some(first) = data.first_mut() {
            *first = com2.lock().inner_read(offset);
        }
    } else if let Some(vc) = virtio_con
        && (kvm::VIRTIO_CONSOLE_MMIO_BASE
            ..kvm::VIRTIO_CONSOLE_MMIO_BASE + virtio_console::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vc.lock()
            .mmio_read(addr - kvm::VIRTIO_CONSOLE_MMIO_BASE, data);
    } else if let Some(vb) = virtio_blk
        && (kvm::VIRTIO_BLK_MMIO_BASE..kvm::VIRTIO_BLK_MMIO_BASE + virtio_blk::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vb.lock().mmio_read(addr - kvm::VIRTIO_BLK_MMIO_BASE, data);
    } else if let Some(vn) = virtio_net
        && (kvm::VIRTIO_NET_MMIO_BASE..kvm::VIRTIO_NET_MMIO_BASE + virtio_net::VIRTIO_MMIO_SIZE)
            .contains(&addr)
    {
        vn.lock().mmio_read(addr - kvm::VIRTIO_NET_MMIO_BASE, data);
    } else {
        for b in data.iter_mut() {
            *b = 0xff;
        }
    }
}

/// Compute register offset for an MMIO address within a serial region.
#[cfg(target_arch = "aarch64")]
fn mmio_serial_offset(addr: u64, base: u64) -> Option<u8> {
    let size = kvm::SERIAL_MMIO_SIZE;
    if addr >= base && addr < base + size {
        Some((addr - base) as u8)
    } else {
        None
    }
}

// -- watchpoint hit dispatch ------------------------------------------
//
// Shared between the AP (`vcpu_run_loop_unified`) and BSP
// (`run_bsp_loop`) paths. Identifies which watchpoint slot fired
// from the per-arch `kvm_debug_exit_arch` payload, gates the slot-0
// trigger on the post-store `exit_kind` value (so a clean
// SCX_EXIT_DONE does not generate a failure dump), and latches the
// matched user slot for the freeze coordinator's epoll loop.

/// `ESR_ELx_EC` decoded value for a watchpoint exception taken from a
/// lower exception level (the only EC that surfaces guest-side data
/// watchpoint hits to userspace via `KVM_EXIT_DEBUG`). Pinned per
/// `arch/arm64/include/asm/esr.h` `ESR_ELx_EC_WATCHPT_LOW = 0x34`.
///
/// `WATCHPT_CUR` (EC=0x35, watchpoint taken at the current EL) is
/// not handled because the kernel's `arm_exit_handlers` table in
/// `arch/arm64/kvm/handle_exit.c` (which routes guest exceptions to
/// userspace via `KVM_EXIT_DEBUG`) only registers a handler for
/// `ESR_ELx_EC_WATCHPT_LOW` — `WATCHPT_CUR` has no entry and is
/// therefore never surfaced to userspace. The guest runs at EL0/EL1
/// and KVM hosts at EL2; from KVM's perspective every guest-side
/// watchpoint trap is "from a lower EL" (LOW), and only EL2's own
/// debug traps would be CUR (which KVM does not arm).
#[cfg(target_arch = "aarch64")]
const ESR_ELx_EC_WATCHPT_LOW: u32 = 0x34;
/// `ESR_ELx_EC` decoded value for a software-step exception taken
/// from a lower exception level. KVM raises this through
/// `KVM_EXIT_DEBUG` after a `KVM_GUESTDBG_SINGLESTEP`-armed
/// `KVM_RUN` retires exactly one instruction (kernel
/// `arch/arm64/kvm/handle_exit.c::kvm_handle_guest_debug` switches
/// on `ESR_ELx_EC_SOFTSTP_LOW` and toggles `DBG_SPSR_SS`). We use
/// it to detect "the offending store has retired" after stepping
/// past a watchpoint trap, so the next `self_arm_watchpoint` call
/// can restore the slot's WCR.E=1. Pinned per
/// `arch/arm64/include/asm/esr.h` `ESR_ELx_EC_SOFTSTP_LOW = 0x32`.
#[cfg(target_arch = "aarch64")]
const ESR_ELx_EC_SOFTSTP_LOW: u32 = 0x32;
/// Bit shift of the `ESR_ELx_EC` field within the lower 32 bits of
/// the ESR_EL2 value KVM hands userspace as
/// `kvm_debug_exit_arch.hsr`. Pinned per `arch/arm64/include/asm/
/// esr.h` `ESR_ELx_EC_SHIFT = 26`.
#[cfg(target_arch = "aarch64")]
const ESR_ELx_EC_SHIFT: u32 = 26;
/// Mask applied to `(hsr >> ESR_ELx_EC_SHIFT)` to extract the
/// 6-bit EC field. Pinned per `ESR_ELx_EC(esr) = (esr & ESR_ELx_
/// EC_MASK) >> ESR_ELx_EC_SHIFT` in the same kernel header.
#[cfg(target_arch = "aarch64")]
const ESR_ELx_EC_MASK: u32 = 0x3F;

/// Dispatch a `KVM_EXIT_DEBUG` watchpoint trap to the matching slot's
/// latch. Reads the per-arch identifier (DR6 on x86_64; ESR EC + FAR
/// on aarch64), gates the slot-0 trigger on the post-store
/// `exit_kind` value, and writes the appropriate `hit` flag for the
/// freeze coordinator to observe.
///
/// `armed_slots` is the per-thread mirror of currently-armed KVAs
/// (one entry per slot) maintained by `self_arm_watchpoint`. On
/// aarch64 it carries the original (un-aligned) KVA so the
/// FAR range check covers the exact 4 bytes the watchpoint targets,
/// not the 8-byte block DBGWVR addresses. On x86_64 the entry is
/// also the requested KVA (DR0..DR3 hold full addresses) but is not
/// used by this helper — DR6 alone identifies which slot fired.
///
/// `single_step_pending` and `single_step_slot` are the per-vCPU
/// loop-local single-step bookkeeping the aarch64 path uses to step
/// past a fired watchpoint:
///
///   - On `EC = ESR_ELx_EC_WATCHPT_LOW (0x34)` with at least one
///     slot whose 4-byte FAR window contains the fault address,
///     after latching `hit` on every matched slot the helper sets
///     `*single_step_pending = true` and stores a 4-bit bitmap of
///     matched slot indices into `*single_step_slot` (bit i set
///     ⇒ slot i was matched). The next loop iteration's
///     `self_arm_watchpoint` call notices the flipped flag,
///     reissues KVM_SET_GUEST_DEBUG with WCR.E=0 on EVERY matched
///     slot (peer arms stay enabled), asserts
///     `KVM_GUESTDBG_SINGLESTEP`, and the following KVM_RUN
///     executes exactly one instruction past the offending store.
///     A 4-bit bitmap is sufficient because there are only four
///     hardware watchpoint slots; multiple matches happen when
///     `arm_user_watchpoint` placed two slots on overlapping
///     KVAs (no duplicate-rejection — see comment on the FAR
///     range loop).
///   - On `EC = ESR_ELx_EC_SOFTSTP_LOW (0x32)` with the flag set,
///     the helper clears `*single_step_pending` (without latching).
///     The next `self_arm_watchpoint` call restores WCR.E=1 on
///     every previously-disabled slot and drops the singlestep
///     bit. The mask in `*single_step_slot` is functionally not
///     consulted on this transition because the per-slot E
///     selector at `vcpu.rs::self_arm_watchpoint` short-circuits
///     on `single_step_pending == false`; restoring WCR.E
///     globally is correct because all slots have valid
///     `request_kva` published.
///
/// Both fields are inert on x86_64 — the x86 watchpoint trap is
/// taken AFTER the store retires (Intel SDM Vol. 3B 17.2.4
/// "Trap-class debug exceptions"), so re-entry advances normally
/// without the disable-step-rearm dance.
pub(crate) fn dispatch_watchpoint_hit(
    watchpoint: &WatchpointArm,
    debug_arch: &kvm_bindings::kvm_debug_exit_arch,
    armed_slots: &[u64; 4],
    single_step_pending: &mut bool,
    single_step_slot: &mut usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // DR6 layout (Intel SDM Vol. 3B 17.2.5): bits 0-3 (B0..B3)
        // indicate which DR fired. Bit 14 (BS) signals single-step.
        // KVM populates `kvm_run.debug.arch.dr6` from the just-
        // fired exit's qualification field (`vmx_get_exit_qual`
        // in `arch/x86/kvm/vmx/vmx.c::handle_exception_nmi`),
        // not from the architectural DR6 register, so the bits we
        // see reflect ONLY the slots that fired on THIS exit —
        // not stale "sticky" bits from prior exits. The dedup
        // gate on `WatchpointArm::hit` (CAS in `latch_*`) handles
        // the cross-vCPU race where two vCPUs each fire on the
        // same slot before either has been processed by the
        // freeze coordinator: only the first false→true transition
        // wakes the coordinator's `hit_evt`.
        //
        // Single-step is aarch64-only — the x86 watchpoint trap
        // is taken AFTER the offending store retires (Intel SDM
        // Vol. 3B 17.2.4 "Trap-class debug exceptions"), so re-
        // entering KVM_RUN advances normally without the
        // disable-step-rearm dance. Consume the unused inputs to
        // keep the per-arch helper signature shared.
        let _ = armed_slots;
        let _ = (&mut *single_step_pending, &mut *single_step_slot);
        let dr6 = debug_arch.dr6;
        let trap_bits = (dr6 & 0xF) as u8;
        if trap_bits == 0 {
            // KVM exited via KVM_EXIT_DEBUG with no DR0..3 trap
            // bits set — possible when a single-step (BS, bit 14)
            // or task-switch (BT, bit 15) fired without a data/
            // code breakpoint match. ktstr never arms BS/BT, so
            // this is either a host-side debug stub leaking
            // through or a synthetic exit — log and ignore.
            // Mirrors the aarch64 "no FAR match" debug log so
            // both arches surface unexpected debug exits the
            // same way.
            tracing::debug!(
                dr6,
                "KVM_EXIT_DEBUG fired with no DR0..DR3 trap bit set \
                 (BS/BT or spurious); not latching"
            );
            return;
        }
        if trap_bits & 0x1 != 0 {
            latch_slot0_with_gate(watchpoint);
        }
        for idx in 0..3 {
            if trap_bits & (1u8 << (idx + 1)) != 0 {
                watchpoint.latch_user_hit(idx);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // ARM debug exit payload (kernel
        // `arch/arm64/kvm/handle_exit.c::kvm_handle_guest_debug`):
        //   `hsr` = lower 32 bits of ESR_EL2
        //   `far` = FAR_EL2 (set only when ESR.EC == WATCHPT_LOW)
        // The EC field at bits [31:26] of ESR distinguishes
        // watchpoint exceptions from breakpoints / soft-step / BRK.
        let ec = (debug_arch.hsr >> ESR_ELx_EC_SHIFT) & ESR_ELx_EC_MASK;
        if ec == ESR_ELx_EC_SOFTSTP_LOW {
            // Software-step exception following a watchpoint hit.
            // The kernel sets cpsr.SS in
            // `kvm_handle_guest_debug` to advertise that exactly
            // one instruction retired since the prior fire; we
            // clear `single_step_pending` so the next
            // `self_arm_watchpoint` call restores the slot's
            // WCR.E=1 and drops `KVM_GUESTDBG_SINGLESTEP`. Do NOT
            // latch any `hit` flag here — the original
            // WATCHPT_LOW exit already latched the freeze
            // trigger; this exit only signals "one instruction
            // executed cleanly past the watched store".
            //
            // If the flag is NOT set we got a soft-step exit
            // we did not request (e.g. host kernel quirk, peer
            // tooling); log and ignore — there is no slot to
            // restore.
            if *single_step_pending {
                *single_step_pending = false;
                // Zero `single_step_slot` defensively. The
                // per-slot E selector in
                // `vcpu.rs::self_arm_watchpoint` already
                // short-circuits on `single_step_pending ==
                // false`, so a stale mask cannot disable a slot
                // — but a future regression that drops the
                // short-circuit would silently disable whatever
                // slots the stale mask still flags. Zeroing
                // here makes the post-step state purely
                // reflect "no slots pending step" so downstream
                // readers cannot trip on a leftover bitmap.
                *single_step_slot = 0;
            } else {
                tracing::debug!(
                    hsr = debug_arch.hsr,
                    "KVM_EXIT_DEBUG soft-step EC with no \
                     single-step pending; ignoring (likely \
                     spurious kernel-side step exit)"
                );
            }
            return;
        }
        if ec != ESR_ELx_EC_WATCHPT_LOW {
            tracing::debug!(
                hsr = debug_arch.hsr,
                ec,
                "KVM_EXIT_DEBUG with non-watchpoint EC; ignoring \
                 (breakpoint/BRK paths are not used by ktstr)"
            );
            return;
        }
        let far = debug_arch.far;
        // ARM ARM D2.10.5: FAR may be imprecise for unaligned
        // accesses. This exact-range check is correct for atomic_t
        // writes (aligned 4-byte stores via atomic_set/cmpxchg) but
        // would miss imprecise hits from unaligned multi-byte
        // accesses spanning the watched range.
        //
        // Range-match FAR against each armed slot's 4-byte
        // window. `armed_slots[i]` is the requested KVA (un-
        // aligned); the watch covers `[kva, kva + 4)`. Multiple
        // slots may match if their watched ranges overlap (e.g.
        // two `Op::WatchSnapshot` registrations on adjacent
        // 4-byte fields of the same struct word). `arm_user_
        // watchpoint` allocates by free-slot search and does NOT
        // reject duplicate KVAs, so overlapping arms are
        // possible. The loop iterates all four slots and latches
        // every match — overlapping arms are never silently
        // dropped.
        let mut matched_mask: u8 = 0;
        for (i, kva) in armed_slots.iter().enumerate() {
            if *kva == 0 {
                continue;
            }
            if far >= *kva && far < kva.saturating_add(4) {
                matched_mask |= 1 << i;
                if i == 0 {
                    latch_slot0_with_gate(watchpoint);
                } else {
                    watchpoint.latch_user_hit(i - 1);
                }
            }
        }
        if matched_mask == 0 {
            tracing::debug!(
                hsr = debug_arch.hsr,
                far,
                armed = ?armed_slots,
                "KVM_EXIT_DEBUG watchpoint fired but FAR matched no \
                 armed slot (possible KVM watchpoint match-distance \
                 fallback or stale arm); not latching"
            );
            return;
        }
        // The aarch64 watchpoint trap fires BEFORE the offending
        // store retires (ARM ARM D2.10.5: "the exception is
        // taken on the instruction that would have made the
        // access"). Re-entering KVM_RUN without intervention
        // replays the same store and re-trips the watchpoint
        // forever. Mirror the kernel's
        // `arch/arm64/kernel/hw_breakpoint.c::do_watchpoint`
        // recipe with a two-mechanism dance:
        //
        //   - `KVM_GUESTDBG_SINGLESTEP` (which sets MDSCR_EL1.SS
        //     in the kernel's `setup_external_mdscr`) is what
        //     causes KVM to retire EXACTLY ONE guest instruction
        //     and exit with `EC = ESR_ELx_EC_SOFTSTP_LOW (0x32)`.
        //     This advances the PC past the watched store. MDSCR.
        //     SS does NOT suppress watchpoint exceptions on the
        //     stepped instruction (per ARM ARM D2.12, software-
        //     step state still respects WCR.E for watchpoints
        //     that match the stepped access).
        //
        //   - Clearing WCR.E=0 on every matched slot is what
        //     prevents the watched store from re-tripping the
        //     watchpoint on the single-step pass. Without this,
        //     the same instruction that originally trapped would
        //     re-trap on its replay (the trap is taken BEFORE
        //     the access; the access has not retired yet).
        //
        // `single_step_slot` carries a 4-bit mask of every
        // matched slot index (bit i set ⇒ slot i must have its
        // WCR.E cleared during the single-step pass). Multiple
        // bits can be set: `arm_user_watchpoint` allocates by
        // free-slot search and does NOT reject duplicate KVAs,
        // so two slots may watch overlapping ranges and fire
        // simultaneously on the same store. `self_arm_watch
        // point` walks the mask and clears WCR.E on every set
        // bit; when `single_step_pending` clears on the
        // following SOFTSTP_LOW exit, all slots get WCR.E=1
        // restored.
        *single_step_pending = true;
        *single_step_slot = matched_mask as usize;
    }
}

/// Slot-0 latch with the post-store `exit_kind` value gate. Reads
/// the host pointer the freeze coordinator published, compares
/// against [`SCX_EXIT_ERROR_THRESHOLD`], and latches the failure-
/// trigger only on error-class transitions.
///
/// `kind_host_ptr` is guaranteed non-null when this helper runs.
/// The freeze coordinator publishes the pair in `kind_host_ptr →
/// request_kva` order with matching `Release` stores (see
/// `freeze_coord.rs::run_coord_loop`, where the err_exit publish
/// issues the `kind_host_ptr` store BEFORE the `request_kva`
/// store). [`super::vcpu::self_arm_watchpoint`] only programs the
/// hardware watchpoint after observing a non-zero `request_kva`
/// via an `Acquire` load — that load synchronises-with the
/// publisher's `Release`, which makes the prior `kind_host_ptr`
/// store visible too. Once armed, a fire reaches this helper only
/// when both stores are visible. The pointer is never invalidated
/// for the VM lifetime: `vm.guest_mem` (which backs the host
/// mapping) is dropped only after every vCPU thread has joined, so
/// the host-side mapping at this address strictly outlives every
/// reader of this pointer.
///
/// On aarch64 an `Acquire` fence pairs with the guest's store: by
/// the time KVM_RUN returns `KVM_EXIT_DEBUG` the trap-into-EL2 +
/// host-context-restore path has already issued an architectural
/// context-synchronization event (ERET, eret-to-EL1 from the
/// hypervisor save/restore), but Rust's memory model does not
/// know about those. The fence makes the `read_volatile` of the
/// host pointer ordered-after that synchronization in Rust's
/// happens-before graph, matching what the x86_64 path gets for
/// free from TSO.
///
/// A null observation here would be a publication-invariant
/// violation; we still check at runtime so a regression in the
/// publisher cannot be turned into a `read_volatile` of a null
/// pointer (UB). The check costs one branch on the cold debug
/// trap path — negligible — and surfaces the invariant break as
/// a `tracing::error!` instead of crashing the run.
fn latch_slot0_with_gate(watchpoint: &WatchpointArm) {
    let host_ptr = watchpoint.kind_host_ptr.load(Ordering::Acquire);
    if host_ptr.is_null() {
        tracing::error!(
            "latch_slot0_with_gate: kind_host_ptr null at fire time — \
             publication invariant broken (request_kva non-zero must \
             imply kind_host_ptr non-null per the Release-store \
             ordering in freeze_coord.rs::run_coord_loop). Skipping \
             slot-0 latch; the BPF .bss late-trigger fallback in the \
             freeze coordinator's poll loop remains active."
        );
        return;
    }
    // Publish ordering: the guest's store is globally visible by
    // the time KVM exits to userspace, but Rust's memory model
    // requires an explicit Acquire fence on weakly-ordered hosts
    // (aarch64) to make the host-pointer read happens-after the
    // guest store in the Rust abstract machine. On x86_64 TSO
    // gives us this for free; the explicit fence is a no-op
    // codegen-wise but keeps the operation ordered in std::sync
    // terms across both arches.
    std::sync::atomic::fence(Ordering::Acquire);
    // SAFETY: `kind_host_ptr` was published by the freeze
    // coordinator before `request_kva` (Release), and the
    // `request_kva` non-zero load that triggered the arm is the
    // synchronizes-with edge for this read. The pointer addresses
    // a u32 inside the guest's `scx_sched` slab page, which stays
    // mapped for the VM lifetime per the `ReservationGuard`
    // contract. Non-null per the `is_null` early-return above.
    let kind = unsafe { std::ptr::read_volatile(host_ptr) };
    if kind >= SCX_EXIT_ERROR_THRESHOLD {
        watchpoint.latch_hit();
    } else {
        tracing::debug!(
            kind,
            threshold = SCX_EXIT_ERROR_THRESHOLD,
            "watchpoint fired on non-error exit_kind transition \
             (e.g. SCX_EXIT_DONE on clean shutdown); skipping \
             freeze trigger"
        );
    }
}

/// Unified per-vCPU KVM_RUN loop for AP threads.
///
/// HLT on APs: check kill + continue on both arches (KVM delivers
/// interrupts to wake the vCPU). Shutdown sets the kill flag so all
/// other vCPUs exit.
///
/// `watchpoint` carries the failure-dump trigger contract: each
/// iteration polls `watchpoint.request_kva` and self-arms a hardware
/// data-write watchpoint on `*scx_root->exit_kind` once the freeze
/// coordinator has resolved its KVA. When the kernel later writes
/// the field, KVM exits via `VcpuExit::Debug`; this loop sets
/// `watchpoint.hit` so the freeze coordinator's late-trigger poll
/// fires immediately. The arm is one-shot per KVA value (the
/// per-vCPU `armed_kva` slot suppresses re-arms after the ioctl
/// lands).
///
/// Freeze handling: when the freeze flag is set, the vCPU thread
/// performs the Cloud Hypervisor pause/snapshot drain dance
/// (set_immediate_exit(1) → vcpu.run() → set_immediate_exit(0)) so
/// any in-flight PIO/MMIO operation completes inside the KVM_RUN
/// ioctl before the thread parks. The drain is necessary because
/// KVM_EXIT_IO/MMIO leave the operation only partially complete on
/// the kernel side; userspace must re-enter KVM_RUN to commit it.
/// After draining, the thread sets `parked=true` (Release-ordered so
/// the host's subsequent guest-memory reads happen-after the
/// drain), then polls freeze on park_timeout. The Acquire load on
/// `parked` from the freeze coordinator IS the memory barrier that
/// makes external-thread guest-memory reads correct on weakly
/// ordered architectures (matches Cloud Hypervisor's pause
/// pattern). The kick that triggers freeze observation uses
/// Firecracker's SIGRTMIN+immediate_exit pattern, but the drain
/// dance itself is Cloud Hypervisor-specific.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vcpu_run_loop_unified(
    vcpu: &mut kvm_ioctls::VcpuFd,
    com1: &Arc<PiMutex<console::Serial>>,
    com2: &Arc<PiMutex<console::Serial>>,
    virtio_con: Option<&Arc<PiMutex<virtio_console::VirtioConsole>>>,
    virtio_blk: Option<&Arc<PiMutex<virtio_blk::VirtioBlk>>>,
    virtio_net: Option<&Arc<PiMutex<virtio_net::VirtioNet>>>,
    kill: &Arc<AtomicBool>,
    kill_evt: &Arc<EventFd>,
    freeze: &Arc<AtomicBool>,
    parked: &Arc<AtomicBool>,
    regs_slot: &Arc<std::sync::Mutex<Option<VcpuRegSnapshot>>>,
    watchpoint: &Arc<WatchpointArm>,
    has_immediate_exit: bool,
    parked_evt: Option<&Arc<EventFd>>,
    thaw_evt: Option<&Arc<EventFd>>,
) {
    // Per-AP `armed_slots` mirrors the BSP-side slot array in
    // `freeze_coord::run_bsp_loop`. Index 0 = DR0 (err_exit watchpoint
    // for `*scx_root->exit_kind`); indices 1..=3 = DR1/DR2/DR3 (user
    // `Op::WatchSnapshot` arms). All start at `0` until the freeze
    // coordinator publishes resolved KVAs. The array is a per-thread
    // local so the per-iteration arm check is four Acquire loads
    // with no cross-thread synchronization beyond the published
    // requests. `arm_failures` counts consecutive non-EINTR ioctl
    // failures; EINTR is transient (SIGRTMIN kick race) and does NOT
    // increment, so a kicked-mid-arm vCPU retries instead of
    // permanently disabling the watchpoint.
    let mut armed_slots: [u64; 4] = [0; 4];
    let mut arm_failures: u8 = 0;
    // aarch64 watchpoint single-step bookkeeping. On aarch64 the
    // hardware watchpoint trap is taken BEFORE the offending store
    // retires (ARM ARM D2.10.5: "the exception is taken on the
    // instruction that would have made the access"), so re-entering
    // KVM_RUN replays the same instruction and re-trips the
    // watchpoint forever. Mirroring the kernel's
    // `arch/arm64/kernel/hw_breakpoint.c` recipe, we disable the
    // fired slot's WCR.E and assert KVM_GUESTDBG_SINGLESTEP for
    // exactly one KVM_RUN; the next KVM_EXIT_DEBUG carries
    // EC=ESR_ELx_EC_SOFTSTP_LOW (0x32), at which point we clear
    // `single_step_pending` and `self_arm_watchpoint` reissues
    // KVM_SET_GUEST_DEBUG with WCR.E restored to 1 and
    // KVM_GUESTDBG_SINGLESTEP cleared. Inert on x86_64 (the trap
    // there is taken AFTER the store, so re-entry advances
    // normally); the locals still pass through to keep the
    // per-arch helper signatures shared.
    let mut single_step_pending: bool = false;
    let mut single_step_slot: usize = 0;
    let mut armed_single_step: bool = false;
    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        // Honour a pending freeze before re-entering KVM_RUN.
        if freeze.load(Ordering::Acquire) {
            handle_freeze(
                vcpu,
                has_immediate_exit,
                kill,
                freeze,
                parked,
                regs_slot,
                parked_evt.map(|a| a.as_ref()),
                thaw_evt.map(|a| a.as_ref()),
                Some(kill_evt.as_ref()),
            );
            if kill.load(Ordering::Acquire) {
                break;
            }
        }
        // Self-arm the failure-dump watchpoint when the coordinator
        // publishes (or republishes) a request KVA. Cheap (atomic
        // load + compare against `armed_kva`) when no new arm is
        // pending. Mirrors `run_bsp_loop`'s arm-before-run pattern;
        // both paths share `WatchpointArm` so a fire on either
        // triggers the late-snapshot rendezvous. Also drives the
        // aarch64 watchpoint single-step transition: when
        // `single_step_pending` is set by the prior watchpoint
        // exit, this call reissues KVM_SET_GUEST_DEBUG with the
        // fired slot's WCR.E cleared and KVM_GUESTDBG_SINGLESTEP
        // asserted; when the SOFTSTP_LOW exit clears the flag, the
        // next call restores WCR.E=1 and drops the singlestep bit.
        self_arm_watchpoint(
            vcpu,
            watchpoint,
            &mut armed_slots,
            &mut arm_failures,
            single_step_pending,
            single_step_slot,
            &mut armed_single_step,
        );

        match vcpu.run() {
            Ok(mut exit) => {
                if matches!(exit, VcpuExit::Hlt) {
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                // KVM_EXIT_DEBUG fires when the armed hardware
                // data-write watchpoint trips on a guest write to
                // `*scx_root->exit_kind`. The kernel writes the
                // field on BOTH error transitions
                // (`scx_error -> SCX_EXIT_ERROR/_BPF/_STALL >=
                // 1024`) AND clean shutdown
                // (`scx_unregister -> SCX_EXIT_DONE = 1`). Only the
                // error transitions should trigger the failure-dump
                // freeze; firing on every clean test exit is a
                // regression. Read the post-store value from the
                // host pointer the coordinator published and gate
                // `hit` on the error threshold. The watchpoint is
                // left armed regardless: the coordinator's freeze +
                // thaw is synchronous with the dump emission, and a
                // future error after a clean transition would still
                // fire (slab page lifetime — the scheduler's
                // `scx_sched` is not freed until well after the
                // last `exit_kind` write).
                if let VcpuExit::Debug(debug_arch) = &exit {
                    dispatch_watchpoint_hit(
                        watchpoint,
                        debug_arch,
                        &armed_slots,
                        &mut single_step_pending,
                        &mut single_step_slot,
                    );
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                match classify_exit(
                    com1,
                    com2,
                    virtio_con.map(|a| a.as_ref()),
                    virtio_blk.map(|a| a.as_ref()),
                    virtio_net.map(|a| a.as_ref()),
                    &mut exit,
                ) {
                    Some(ExitAction::Continue) | None => {}
                    Some(ExitAction::Shutdown) => {
                        kill.store(true, Ordering::Release);
                        // Wake the freeze coordinator's epoll loop
                        // so it sees the kill flag without waiting
                        // up to one full epoll timeout. Failure
                        // (EAGAIN under EFD_NONBLOCK from a
                        // saturated counter) is benign — any prior
                        // pending edge already wakes the coord, and
                        // the AtomicBool above remains the source
                        // of truth.
                        let _ = kill_evt.write(1);
                        break;
                    }
                    Some(ExitAction::Fatal(_)) => {
                        // AP fatal exit (FailEntry / InternalError):
                        // surface in tracing AND propagate the kill
                        // signal. Without `kill.store(true)` and the
                        // kill_evt write, the AP thread silently
                        // exits while peer vCPUs and the freeze
                        // coordinator stay running — peers eventually
                        // hit FREEZE_RENDEZVOUS_TIMEOUT instead of
                        // shutting down promptly. Mirrors the
                        // Shutdown arm's kill-propagation pattern.
                        tracing::error!("AP fatal exit");
                        kill.store(true, Ordering::Release);
                        let _ = kill_evt.write(1);
                        break;
                    }
                }
            }
            Err(e) => {
                if e.errno() == libc::EINTR || e.errno() == libc::EAGAIN {
                    vcpu.set_kvm_immediate_exit(0);
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                if kill.load(Ordering::Acquire) {
                    break;
                }
            }
        }

        if kill.load(Ordering::Acquire) {
            break;
        }
    }
}

/// Drain pending PIO/MMIO state and park the vCPU until freeze
/// clears. Called from the run loop when the freeze flag is observed,
/// and from `mod.rs::run_bsp_loop` for the same purpose on the BSP
/// thread.
///
/// The drain dance — `set_immediate_exit(1) → vcpu.run() →
/// set_immediate_exit(0)` — is the Cloud Hypervisor pause/snapshot
/// pattern for completing in-flight I/O before pausing. KVM_RUN with
/// immediate_exit=1 returns -EINTR without entering the guest but
/// still commits any pending PIO/MMIO state from the previous exit
/// (per the KVM API contract: pending I/O is committed at the start
/// of KVM_RUN even when immediate_exit prevents guest entry).
/// `has_immediate_exit` gates the dance — without
/// KVM_CAP_IMMEDIATE_EXIT, calling `vcpu.run()` here would re-enter
/// the guest instead of returning EINTR, so the drain step is
/// skipped on kernels that lack the cap. The freeze rendezvous
/// itself still works (set parked, await thaw); only the I/O drain
/// is skipped.
///
/// After the drain, the thread sets `parked=true` with Release
/// ordering and polls freeze on `park_timeout(10ms)` until the
/// coordinator clears it. The thaw path uses no explicit unpark —
/// the 10ms park_timeout cadence picks up the cleared freeze flag
/// within at most 10 ms, which is well below the dump latency
/// budget.
///
/// `kill` is honoured throughout: a shutdown signal during the park
/// loop wins over freeze and the function returns to the caller's
/// kill-check at the top of the loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_freeze(
    vcpu: &mut kvm_ioctls::VcpuFd,
    has_immediate_exit: bool,
    kill: &Arc<AtomicBool>,
    freeze: &Arc<AtomicBool>,
    parked: &Arc<AtomicBool>,
    regs_slot: &Arc<std::sync::Mutex<Option<VcpuRegSnapshot>>>,
    parked_evt: Option<&EventFd>,
    thaw_evt: Option<&EventFd>,
    kill_evt: Option<&EventFd>,
) {
    // Drain dance: complete any pending PIO/MMIO before parking.
    // Skipped on kernels without KVM_CAP_IMMEDIATE_EXIT, where
    // calling vcpu.run() with the cap absent would re-enter the
    // guest instead of returning EINTR.
    if has_immediate_exit {
        vcpu.set_kvm_immediate_exit(1);
        let _ = vcpu.run();
        vcpu.set_kvm_immediate_exit(0);
    }

    // Capture vCPU registers BEFORE the Release store on `parked`.
    // KVM_GET_REGS / KVM_GET_SREGS are vCPU-fd-bound ioctls — they
    // must run on the vCPU thread (not cross-thread from the
    // coordinator). Capturing here means the regs slot's Mutex
    // store is happens-before the coordinator's Acquire on
    // `parked`, so the coordinator can read the slot via the same
    // synchronizes-with edge that makes its guest-memory reads
    // correct. A failed capture stores `None`; the dump shows
    // "registers unavailable" rather than panicking the freeze.
    let snapshot = capture_vcpu_regs(vcpu);
    *regs_slot.lock().unwrap_or_else(|e| e.into_inner()) = snapshot;

    // Acknowledge frozen state. The Release store synchronizes-with
    // the coordinator's Acquire load on `parked`, providing the
    // happens-before edge that makes the coordinator's subsequent
    // guest-memory reads correct.
    parked.store(true, Ordering::Release);

    // Wake the freeze coordinator's rendezvous wait — write to the
    // shared `parked_evt` AFTER the Release store on `parked`. The
    // coordinator drains the eventfd once and then re-checks every
    // vCPU's `parked` flag plus the worker's `paused` flag. The
    // ordering is load-bearing: the coordinator's Acquire load on
    // `parked` happens-after this Release, so its subsequent
    // guest-memory reads observe every queue mutation the vCPU
    // performed before the drain dance.
    //
    // EAGAIN under EFD_NONBLOCK from a saturated counter is benign:
    // the AtomicBool is the source of truth, and any prior pending
    // edge already wakes the coordinator. Log so a real eventfd
    // breakage surfaces, but do not propagate.
    if let Some(evt) = parked_evt
        && let Err(e) = evt.write(1)
    {
        tracing::debug!(
            err = %e,
            "handle_freeze: parked_evt write failed (EAGAIN expected on counter saturation)"
        );
    }

    // Park until freeze clears or shutdown wins. The thaw_evt
    // is written by the freeze coordinator alongside
    // `freeze.store(false, Release)`; poll on [thaw_evt, kill_evt]
    // with a 100 ms backstop so a missed eventfd write (counter
    // overflow / EAGAIN) still drops the parked vCPU within
    // bounded latency. Without the thaw_evt the legacy
    // park_timeout(10 ms) cadence applies as the only source of
    // wake.
    use std::os::fd::AsRawFd;
    while freeze.load(Ordering::Acquire) {
        if kill.load(Ordering::Acquire) {
            break;
        }
        match (thaw_evt, kill_evt) {
            (Some(thaw), kev) => {
                let mut pfds = [
                    libc::pollfd {
                        fd: thaw.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                    libc::pollfd {
                        fd: kev.map_or(-1, |k| k.as_raw_fd()),
                        events: libc::POLLIN,
                        revents: 0,
                    },
                ];
                let nfds = if kev.is_some() { 2 } else { 1 };
                unsafe {
                    libc::poll(pfds.as_mut_ptr(), nfds as libc::nfds_t, 100);
                }
                // Do NOT drain the shared `thaw_evt` here. The
                // coordinator writes to thaw_evt ONCE per thaw and
                // every parked AP polls the SAME fd; if the first
                // wake-winner drains the counter, every other AP's
                // poll blocks for the full 100 ms backstop instead
                // of waking immediately. Leaving the eventfd level
                // high means poll returns immediately for every AP
                // — fast thaw across all peers. The `freeze.load
                // (Acquire)` re-check at the top of the loop is the
                // source of truth: once `freeze` clears the loop
                // exits regardless of eventfd state.
            }
            (None, _) => {
                // No thaw_evt plumbed (e.g. interactive shell path
                // that doesn't run a freeze coordinator). Fall back
                // to the legacy park_timeout cadence — the freeze
                // flag will never be set in that path so this
                // branch is structurally unreachable for real
                // shutdowns, but the safe-by-construction fallback
                // keeps the function callable with all None.
                std::thread::park_timeout(std::time::Duration::from_millis(10));
            }
        }
    }

    // Resume: clear parked so subsequent freeze cycles are observable.
    parked.store(false, Ordering::Release);
}

// ---------------------------------------------------------------------------
// I/O dispatch — shared between BSP and AP run loops
// ---------------------------------------------------------------------------

const KVM_SYSTEM_EVENT_SHUTDOWN: u32 = 1;
const KVM_SYSTEM_EVENT_RESET: u32 = 2;

/// Classified vCPU exit action from `classify_exit`.
pub(crate) enum ExitAction {
    /// Continue running (I/O handled, etc.).
    Continue,
    /// Clean shutdown (system reset, VcpuExit::Shutdown, etc.).
    Shutdown,
    /// Fatal error. `Some(reason)` for FailEntry, `None` for InternalError.
    Fatal(Option<u64>),
}

/// Classify a VcpuExit into an ExitAction, dispatching arch-specific I/O.
///
/// Returns `None` for HLT (caller handles: check kill flag, continue).
/// Takes the exit by mutable reference so IoIn/MmioRead data buffers
/// can be written back.
///
/// On aarch64, serial and virtio-console are dispatched via MMIO.
/// On x86_64, serial is dispatched via port I/O; virtio-console via MMIO.
#[allow(clippy::too_many_arguments)]
pub(crate) fn classify_exit(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
    virtio_blk: Option<&PiMutex<virtio_blk::VirtioBlk>>,
    virtio_net: Option<&PiMutex<virtio_net::VirtioNet>>,
    exit: &mut VcpuExit,
) -> Option<ExitAction> {
    match exit {
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoOut(port, data) => {
            if dispatch_io_out(com1, com2, *port, data) {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::IoIn(port, data) => {
            dispatch_io_in(com1, com2, *port, data);
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioWrite(addr, data) => {
            if dispatch_mmio_write(com1, com2, virtio_con, virtio_blk, virtio_net, *addr, data) {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioRead(addr, data) => {
            dispatch_mmio_read(com1, com2, virtio_con, virtio_blk, virtio_net, *addr, data);
            Some(ExitAction::Continue)
        }
        VcpuExit::Hlt => None,
        VcpuExit::Shutdown => Some(ExitAction::Shutdown),
        VcpuExit::SystemEvent(event_type, _) => {
            if *event_type == KVM_SYSTEM_EVENT_SHUTDOWN || *event_type == KVM_SYSTEM_EVENT_RESET {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        VcpuExit::FailEntry(reason, _cpu) => Some(ExitAction::Fatal(Some(*reason))),
        VcpuExit::InternalError => Some(ExitAction::Fatal(None)),
        #[cfg(target_arch = "x86_64")]
        VcpuExit::MmioRead(addr, data) => {
            if let Some(vc) = virtio_con {
                let base = kvm::VIRTIO_CONSOLE_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_console::VIRTIO_MMIO_SIZE {
                    vc.lock().mmio_read(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vb) = virtio_blk {
                let base = kvm::VIRTIO_BLK_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_blk::VIRTIO_MMIO_SIZE {
                    vb.lock().mmio_read(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vn) = virtio_net {
                let base = kvm::VIRTIO_NET_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_net::VIRTIO_MMIO_SIZE {
                    vn.lock().mmio_read(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            for b in data.iter_mut() {
                *b = 0xff;
            }
            Some(ExitAction::Continue)
        }
        #[cfg(target_arch = "x86_64")]
        VcpuExit::MmioWrite(addr, data) => {
            if let Some(vc) = virtio_con {
                let base = kvm::VIRTIO_CONSOLE_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_console::VIRTIO_MMIO_SIZE {
                    vc.lock().mmio_write(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vb) = virtio_blk {
                let base = kvm::VIRTIO_BLK_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_blk::VIRTIO_MMIO_SIZE {
                    vb.lock().mmio_write(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            if let Some(vn) = virtio_net {
                let base = kvm::VIRTIO_NET_MMIO_BASE;
                if *addr >= base && *addr < base + virtio_net::VIRTIO_MMIO_SIZE {
                    vn.lock().mmio_write(*addr - base, data);
                    return Some(ExitAction::Continue);
                }
            }
            Some(ExitAction::Continue)
        }
        _ => None,
    }
}

/// I8042 ports and commands — minimal emulation for x86 guest reboot.
/// The kernel's default reboot method (`reboot=k`) writes CMD_RESET_CPU
/// (0xFE) to the i8042 command port (0x64).
#[cfg(target_arch = "x86_64")]
const I8042_DATA_PORT: u16 = 0x60;
#[cfg(target_arch = "x86_64")]
const I8042_CMD_PORT: u16 = 0x64;
#[cfg(target_arch = "x86_64")]
const I8042_CMD_RESET_CPU: u8 = 0xFE;

/// Dispatch an I/O out to serial ports or system devices.
/// Returns `true` if the caller should exit (system reset detected).
#[cfg(target_arch = "x86_64")]
fn dispatch_io_out(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    port: u16,
    data: &[u8],
) -> bool {
    // I8042 reset: kernel writes 0xFE to port 0x64 during reboot.
    if port == I8042_CMD_PORT && data.first() == Some(&I8042_CMD_RESET_CPU) {
        return true;
    }
    // Only lock the matching serial port based on port range.
    if (console::COM1_BASE..console::COM1_BASE + 8).contains(&port) {
        com1.lock().handle_out(port, data);
    } else if (console::COM2_BASE..console::COM2_BASE + 8).contains(&port) {
        com2.lock().handle_out(port, data);
    }
    false
}

/// Dispatch an I/O in from serial ports or system devices.
/// Handles i8042 reads to satisfy the kernel's keyboard probe.
#[cfg(target_arch = "x86_64")]
fn dispatch_io_in(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    port: u16,
    data: &mut [u8],
) {
    match port {
        // I8042 status: return 0 (no data, buffer empty).
        I8042_CMD_PORT => {
            if let Some(b) = data.first_mut() {
                *b = 0;
            }
        }
        // I8042 data: return 0 (no keypress).
        I8042_DATA_PORT => {
            if let Some(b) = data.first_mut() {
                *b = 0;
            }
        }
        // Only lock the matching serial port based on port range.
        p if (console::COM1_BASE..console::COM1_BASE + 8).contains(&p) => {
            com1.lock().handle_in(port, data);
        }
        p if (console::COM2_BASE..console::COM2_BASE + 8).contains(&p) => {
            com2.lock().handle_in(port, data);
        }
        _ => {}
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_i8042_reset_is_shutdown_signal() {
        // The BSP relies on I8042 reset (port 0x64, 0xFE) for shutdown
        // detection instead of VcpuExit::Hlt. Verify that dispatch_io_out
        // returns true for the reset command.
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(
            dispatch_io_out(&com1, &com2, I8042_CMD_PORT, &[I8042_CMD_RESET_CPU]),
            "I8042 reset (0xFE to port 0x64) must signal shutdown"
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_i8042_non_reset() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, I8042_CMD_PORT, &[0x00]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_serial_com1() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        // Write 'A' to COM1 THR — should not trigger reset.
        assert!(!dispatch_io_out(&com1, &com2, console::COM1_BASE, b"A"));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_serial_com2() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, console::COM2_BASE, b"B"));
        let output = com2.lock().output();
        assert!(output.contains('B'));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_out_unknown_port() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        assert!(!dispatch_io_out(&com1, &com2, 0x1234, &[0xFF]));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_in_i8042_status() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, I8042_CMD_PORT, &mut data);
        assert_eq!(data[0], 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_in_i8042_data() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, I8042_DATA_PORT, &mut data);
        assert_eq!(data[0], 0);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dispatch_io_in_unknown_port() {
        let com1 = PiMutex::new(console::Serial::new(console::COM1_BASE));
        let com2 = PiMutex::new(console::Serial::new(console::COM2_BASE));
        let mut data = [0xFFu8; 1];
        dispatch_io_in(&com1, &com2, 0x1234, &mut data);
        assert_eq!(data[0], 0xFF, "unknown port should not modify data");
    }
}

#[cfg(test)]
mod vcpu_reg_snapshot_tests {
    use super::*;

    #[test]
    fn vcpu_reg_snapshot_display_renders_three_hex_fields() {
        // x86_64-shape snapshot (user_page_table_root=None): only
        // the three core hex fields render; no `uptroot=` suffix.
        let s = VcpuRegSnapshot {
            instruction_pointer: 0xffff_ffff_8100_1234,
            stack_pointer: 0xffff_ffff_8000_0000,
            page_table_root: 0x0123_4567_89ab_cdef,
            user_page_table_root: None,
            tcr_el1: None,
        };
        let out = format!("{s}");
        assert_eq!(
            out,
            "ip=0xffffffff81001234 sp=0xffffffff80000000 ptroot=0x0123456789abcdef"
        );
    }

    #[test]
    fn vcpu_reg_snapshot_display_appends_user_pt_root_when_present() {
        // aarch64-shape snapshot: user_page_table_root populated
        // → Display appends ` uptroot=0x...`. Pinning here so a
        // future Display tweak (e.g. swapping " " for "\n  ")
        // is caught.
        let s = VcpuRegSnapshot {
            instruction_pointer: 0xffff_8000_8100_1234,
            stack_pointer: 0xffff_8000_8000_0000,
            page_table_root: 0x0000_4000_8000_0000,
            user_page_table_root: Some(0x0000_0000_aaaa_bbbb),
            tcr_el1: Some(0xb510_0010),
        };
        let out = format!("{s}");
        assert_eq!(
            out,
            "ip=0xffff800081001234 sp=0xffff800080000000 ptroot=0x0000400080000000 uptroot=0x00000000aaaabbbb"
        );
    }

    #[test]
    fn vcpu_reg_snapshot_serde_round_trip() {
        let s = VcpuRegSnapshot {
            instruction_pointer: 0x1,
            stack_pointer: 0x2,
            page_table_root: 0x3,
            user_page_table_root: None,
            tcr_el1: None,
        };
        let json = serde_json::to_string(&s).expect("serialize");
        // Pin the JSON key names so a future field rename is
        // caught here rather than in downstream consumers
        // (operator JSON parsers, the failure_dump_e2e fixture).
        // Arch-neutral keys: see field doc on
        // VcpuRegSnapshot::page_table_root for the per-arch
        // semantics each one carries.
        assert!(
            json.contains("\"instruction_pointer\""),
            "missing JSON key `instruction_pointer`: {json}"
        );
        assert!(
            json.contains("\"stack_pointer\""),
            "missing JSON key `stack_pointer`: {json}"
        );
        assert!(
            json.contains("\"page_table_root\""),
            "missing JSON key `page_table_root`: {json}"
        );
        // user_page_table_root is None → serde-skipped via
        // skip_serializing_if = "Option::is_none"; assert it does
        // NOT appear in the JSON.
        assert!(
            !json.contains("\"user_page_table_root\""),
            "user_page_table_root must skip-serialize when None: {json}"
        );
        let parsed: VcpuRegSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.instruction_pointer, 0x1);
        assert_eq!(parsed.stack_pointer, 0x2);
        assert_eq!(parsed.page_table_root, 0x3);
        assert!(
            parsed.user_page_table_root.is_none(),
            "missing field must deserialize as None"
        );
    }

    #[test]
    fn vcpu_reg_snapshot_serde_round_trip_with_user_pt_root() {
        // aarch64-shape: user_page_table_root populated → JSON
        // carries the key, deserialize round-trips the value.
        let s = VcpuRegSnapshot {
            instruction_pointer: 0x1,
            stack_pointer: 0x2,
            page_table_root: 0x3,
            user_page_table_root: Some(0xdead_beef_cafe_d00d),
            tcr_el1: None,
        };
        let json = serde_json::to_string(&s).expect("serialize");
        assert!(
            json.contains("\"user_page_table_root\""),
            "user_page_table_root must serialize when Some: {json}"
        );
        let parsed: VcpuRegSnapshot = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.user_page_table_root, Some(0xdead_beef_cafe_d00d));
    }

    #[test]
    fn vcpu_reg_snapshot_serde_round_trip_tcr_el1() {
        // Mirrors the user_page_table_root coverage but for tcr_el1:
        // skip_serializing_if = "Option::is_none" must drop the key
        // when None and emit + round-trip it when Some. Picks a
        // representative TCR_EL1 word (T1SZ=0x10 in [21:16] with
        // TG1=0b10 / 4 KB granule in [31:30]) so the test pins the
        // wire format the page-table walker actually sees on aarch64.
        let some_val: u64 = 0x0000_0000_b510_0010;

        let s_some = VcpuRegSnapshot {
            instruction_pointer: 0x1,
            stack_pointer: 0x2,
            page_table_root: 0x3,
            user_page_table_root: None,
            tcr_el1: Some(some_val),
        };
        let json_some = serde_json::to_string(&s_some).expect("serialize Some");
        assert!(
            json_some.contains("\"tcr_el1\""),
            "tcr_el1 must serialize when Some: {json_some}"
        );

        let s_none = VcpuRegSnapshot {
            instruction_pointer: 0x1,
            stack_pointer: 0x2,
            page_table_root: 0x3,
            user_page_table_root: None,
            tcr_el1: None,
        };
        let json_none = serde_json::to_string(&s_none).expect("serialize None");
        assert!(
            !json_none.contains("\"tcr_el1\""),
            "tcr_el1 must skip-serialize when None: {json_none}"
        );

        // Deserialize Some-flavour JSON back, assert value preserved.
        let parsed_some: VcpuRegSnapshot =
            serde_json::from_str(&json_some).expect("deserialize Some");
        assert_eq!(parsed_some.tcr_el1, Some(some_val));

        // Deserialize JSON without the key (None-flavour); the
        // serde(default) attribute must yield None rather than
        // failing because the field is absent.
        let parsed_none: VcpuRegSnapshot =
            serde_json::from_str(&json_none).expect("deserialize None");
        assert!(
            parsed_none.tcr_el1.is_none(),
            "missing tcr_el1 must deserialize as None"
        );
    }

    #[test]
    fn vcpu_reg_snapshot_zero_renders_zeros() {
        let s = VcpuRegSnapshot {
            instruction_pointer: 0,
            stack_pointer: 0,
            page_table_root: 0,
            user_page_table_root: None,
            tcr_el1: None,
        };
        // 16 hex digits each — leading zeros preserved so widths
        // line up across rows in multi-vcpu output.
        assert_eq!(
            format!("{s}"),
            "ip=0x0000000000000000 sp=0x0000000000000000 ptroot=0x0000000000000000"
        );
    }

    /// Pure-arithmetic test on the aarch64 KVM register IDs the
    /// capture path uses. Verifying the encoding here means a
    /// transcription bug (e.g. wrong byte offset, dropped flag,
    /// wrong sysreg op-code packing) would be caught without
    /// booting an aarch64 VM. Mirrors the kernel's
    /// `KVM_REG_ARM_CORE_REG(name) = offsetof(struct kvm_regs,
    /// name) / sizeof(__u32)` macro for core regs and the
    /// `ARM64_SYS_REG(Op0, Op1, CRn, CRm, Op2)` packing for
    /// sysregs (arch/arm64/include/uapi/asm/kvm.h).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn aarch64_register_ids_match_kernel_encoding() {
        const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
        const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
        const KVM_REG_ARM_CORE: u64 = 0x0010_0000;
        const KVM_REG_ARM64_SYSREG: u64 = 0x0013_0000;

        // PC at byte offset 256 in struct kvm_regs (= same offset
        // in user_pt_regs because user_pt_regs.regs is at offset
        // 0). 256 / 4 = 64.
        const EXPECTED_PC_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | 64;
        // SP_EL1 at byte offset 272 in struct kvm_regs (right
        // after the 272-byte user_pt_regs). 272 / 4 = 68.
        const EXPECTED_SP_EL1_ID: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | 68;
        // TTBR0_EL1 sysreg packing: Op0=3, Op1=0, CRn=2, CRm=0, Op2=0
        // → (3<<14) | (0<<11) | (2<<7) | (0<<3) | 0 = 0xC100.
        const EXPECTED_TTBR0_EL1_ID: u64 =
            KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC100;
        // TTBR1_EL1 sysreg packing: Op0=3, Op1=0, CRn=2, CRm=0, Op2=1
        // → (3<<14) | (0<<11) | (2<<7) | (0<<3) | 1 = 0xC101.
        const EXPECTED_TTBR1_EL1_ID: u64 =
            KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;

        // Reconstruct what capture_vcpu_regs declares to catch
        // any drift between the const declarations there and
        // the kernel ABI. The unsafe cast to *const u64 isn't
        // available across modules, so re-derive the values
        // here using the exact same expression form.
        let pc_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (256 / 4);
        let sp_el1_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_CORE | (272 / 4);
        let ttbr0_el1_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC100;
        let ttbr1_el1_id = KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM64_SYSREG | 0xC101;

        assert_eq!(pc_id, EXPECTED_PC_ID, "PC_ID encoding drift");
        assert_eq!(
            sp_el1_id, EXPECTED_SP_EL1_ID,
            "SP_EL1_ID encoding drift — note offset is 272 (sp_el1), \
             not 248 (sp_el0)"
        );
        assert_eq!(
            ttbr0_el1_id, EXPECTED_TTBR0_EL1_ID,
            "TTBR0_EL1_ID encoding drift — verify (Op0=3, Op1=0, \
             CRn=2, CRm=0, Op2=0) packs to 0xC100"
        );
        assert_eq!(
            ttbr1_el1_id, EXPECTED_TTBR1_EL1_ID,
            "TTBR1_EL1_ID encoding drift — verify (Op0=3, Op1=0, \
             CRn=2, CRm=0, Op2=1) packs to 0xC101"
        );
        // Adjacency check: TTBR0 and TTBR1 differ only in Op2,
        // so the encoding must differ by exactly 1. Catches a
        // typo where one constant gets the other's value.
        assert_eq!(
            ttbr1_el1_id - ttbr0_el1_id,
            1,
            "TTBR0/TTBR1 encodings should differ by exactly 1 (Op2 bit)"
        );
    }

    /// DR7 wire format for a 4-byte write watchpoint in slot 0.
    ///
    /// The KVM hardware-watchpoint freeze trigger arms slot 0 of
    /// the guest's debug registers via `KVM_SET_GUEST_DEBUG` to
    /// catch writes to `sch->exit_kind`. The DR7 byte the VMM
    /// hands KVM must encode the exact slot/length/access pattern
    /// the watchpoint requires, otherwise either KVM rejects the
    /// configuration or the breakpoint catches the wrong access
    /// class — both surface as silent freeze-coordinator failure
    /// (the guest never traps, no failure dump).
    ///
    /// Field layout per Intel SDM Vol 3 Ch 17.2 ("Debug Registers")
    /// and pinned against the production
    /// [`super::super::vcpu::self_arm_watchpoint`] x86_64 path which
    /// composes DR7 as:
    ///
    ///   base = `0x400 | 0x200 | 0x100`  (MBS, GE, LE)
    ///   per slot i: `(0b11) << (2*i)`           (L<i> | G<i>)
    ///             | `(0b01) << (16 + 4*i)`     (R/W<i> = write)
    ///             | `(0b11) << (18 + 4*i)`     (LEN<i> = 4-byte)
    ///
    /// Bit-by-bit:
    ///
    ///   bit  0   L0   (local enable, slot 0) — 1
    ///   bit  1   G0   (global enable, slot 0) — 1
    ///   bit  8   LE   (local exact-match, required for data BPs)
    ///   bit  9   GE   (global exact-match, required for data BPs)
    ///   bit 10   reserved, must be 1 (DR7_FIXED_1)
    ///   bits [17:16]  R/W0 (00=exec, 01=write, 11=rw) — 01
    ///   bits [19:18]  LEN0 (00=1B, 01=2B, 10=8B, 11=4B) — 11
    ///
    /// Expected value 0xD0703 is what the production wire format
    /// emits for (slot=0, type=write, len=4). Pinning the arithmetic
    /// here means a future refactor that flips a bit (e.g. swaps R/W
    /// to exec, drops GE/LE, picks the wrong length encoding, drops
    /// L0 or G0) is caught at unit-test time before the trigger
    /// silently stops firing.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dr7_slot0_write_4byte_encoding() {
        // Field constants — local to the test so a future
        // production-side rename does not silently divorce the
        // assertion from the wire format.
        const DR7_FIXED_1: u64 = 1 << 10;
        const DR_LOCAL_EXACT: u64 = 1 << 8; // LE — local exact-match
        const DR_GLOBAL_EXACT: u64 = 1 << 9; // GE — global exact-match
        const DR_LOCAL_ENABLE: u64 = 1 << 0; // L0 — slot 0 local
        const DR_GLOBAL_ENABLE: u64 = 1 << 1; // G0 — slot 0 global
        const DR_RW_WRITE: u64 = 0b01;
        const DR_LEN_4: u64 = 0b11;
        // Slot 0 occupies bits 16/17 for R/W and 18/19 for LEN.
        const SLOT0_RW_SHIFT: u32 = 16;
        const SLOT0_LEN_SHIFT: u32 = 18;

        let dr7 = DR7_FIXED_1
            | DR_GLOBAL_EXACT
            | DR_LOCAL_EXACT
            | DR_LOCAL_ENABLE
            | DR_GLOBAL_ENABLE
            | (DR_RW_WRITE << SLOT0_RW_SHIFT)
            | (DR_LEN_4 << SLOT0_LEN_SHIFT);
        // 0xD0703 mirrors what `self_arm_watchpoint` in
        // `super::super::vcpu` actually programs into KVM. A drift
        // here means the watchpoint's wire format diverged from the
        // production encoding and `KVM_SET_GUEST_DEBUG` will arm the
        // wrong breakpoint (or none).
        assert_eq!(
            dr7, 0xD0703,
            "DR7 encoding for (slot=0, write, 4B) must match the production wire format"
        );

        // Bit-by-bit cross-check: every contributing bit must be
        // present, and every other bit must be clear. Catches the
        // failure mode where two bugs cancel — e.g. wrong shift
        // for R/W combined with wrong shift for LEN that happen
        // to sum to the right total.
        assert_ne!(dr7 & (1 << 0), 0, "L0 (bit 0) must be set");
        assert_ne!(dr7 & (1 << 1), 0, "G0 (bit 1) must be set");
        assert_ne!(
            dr7 & (1 << 8),
            0,
            "LE (bit 8) must be set for data breakpoints"
        );
        assert_ne!(
            dr7 & (1 << 9),
            0,
            "GE (bit 9) must be set for data breakpoints"
        );
        assert_ne!(dr7 & (1 << 10), 0, "DR7_FIXED_1 (bit 10) must be set");
        // Slot 0 R/W field = 0b01 (write).
        assert_eq!(
            (dr7 >> SLOT0_RW_SHIFT) & 0b11,
            DR_RW_WRITE,
            "slot 0 R/W field must encode write (0b01)"
        );
        // Slot 0 LEN field = 0b11 (4 bytes).
        assert_eq!(
            (dr7 >> SLOT0_LEN_SHIFT) & 0b11,
            DR_LEN_4,
            "slot 0 LEN field must encode 4 bytes (0b11)"
        );
        // No other slot should be enabled.
        assert_eq!(
            dr7 & 0b1111_1100,
            0,
            "slots 1..3 must be disabled (L/G bits clear)"
        );
        // R/W and LEN fields for slots 1..3 must be zero.
        assert_eq!(
            (dr7 >> 20) & 0xFFF,
            0,
            "slots 1..3 R/W + LEN fields must be zero"
        );
    }

    /// DBGWCR wire format for a 4-byte write watchpoint at byte
    /// offset 0 of an 8-byte aligned block. The aarch64 sibling of
    /// the x86_64 [`dr7_slot0_write_4byte_encoding`] test above —
    /// pins the exact bit layout that
    /// [`super::super::vcpu::self_arm_watchpoint`] emits, so a future
    /// refactor that flips a bit (e.g. swaps LSC to read, drops PAC,
    /// picks the wrong BAS shift) is caught at unit-test time
    /// before the trigger silently stops firing.
    ///
    /// Field layout per ARM ARM D7.3.11 ("DBGWCR<n>_EL1, Debug
    /// Watchpoint Control Registers") and pinned against QEMU's
    /// `insert_hw_watchpoint` in `target/arm/hyp_gdbstub.c`:
    ///
    ///   bit 0       E   = 1 (enable)
    ///   bits [2:1]  PAC = 0b11 (EL0+EL1, any security state)
    ///   bits [4:3]  LSC = 0b10 (write-only)
    ///   bits [12:5] BAS = 0xF << byte_offset (4 contiguous bytes)
    ///   bits [15:13] HMC = 0
    ///   bits [19:16] SSC = 0
    ///   bit 20       WT = 0
    ///   bits [23:21] LBN = 0
    ///   bits [28:24] MASK = 0
    ///
    /// Concrete byte_offset=0 wire format: `0x1F7`.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn dbgwcr_slot0_write_4byte_encoding_offset0() {
        let kva: u64 = 0xffff_ffff_8100_1000; // 8-byte aligned (low bits = 0)
        let byte_offset = (kva & 0x7u64) as u32;
        let bas: u64 = 0xFu64 << byte_offset;
        let wcr: u64 = 1u64 | (0b11u64 << 1) | (0b10u64 << 3) | (bas << 5);
        assert_eq!(
            wcr, 0x1F7,
            "DBGWCR encoding for (slot=0, write, 4B, offset=0) must \
             match the QEMU/ARM ARM gold-standard wire format"
        );
        // Bit-by-bit cross-check.
        assert_eq!(wcr & 1, 1, "E (bit 0) must be set");
        assert_eq!(
            (wcr >> 1) & 0b11,
            0b11,
            "PAC (bits 2:1) must be 0b11 (EL0+EL1)"
        );
        assert_eq!(
            (wcr >> 3) & 0b11,
            0b10,
            "LSC (bits 4:3) must be 0b10 (write-only)"
        );
        assert_eq!(
            (wcr >> 5) & 0xFF,
            0x0F,
            "BAS (bits 12:5) must be 0x0F for offset=0 (4 \
             contiguous low bytes)"
        );
        // No other fields must be set.
        assert_eq!(
            (wcr >> 13) & 0xF,
            0,
            "HMC + low SSC bit (bits 16:13) must be zero"
        );
        assert_eq!((wcr >> 20) & 0xF, 0, "WT + LBN must be zero");
        assert_eq!((wcr >> 24) & 0x1F, 0, "MASK must be zero");
    }

    /// DBGWCR encoding for a 4-byte write watchpoint at byte offset
    /// 4 of an 8-byte aligned block. Catches the failure mode where
    /// a 4-byte aligned but not 8-byte aligned KVA (e.g. a struct
    /// field at offset 4 inside an 8-byte aligned struct) gets the
    /// wrong BAS shift. Concrete wire format: `0x1E17`.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn dbgwcr_slot0_write_4byte_encoding_offset4() {
        let kva: u64 = 0xffff_ffff_8100_1004; // 4-byte aligned, byte_offset=4
        let byte_offset = (kva & 0x7u64) as u32;
        let bas: u64 = 0xFu64 << byte_offset;
        let wcr: u64 = 1u64 | (0b11u64 << 1) | (0b10u64 << 3) | (bas << 5);
        assert_eq!(
            wcr, 0x1E17,
            "DBGWCR encoding for (slot=0, write, 4B, offset=4) must \
             match `0x1 | (3<<1) | (2<<3) | (0xF0 << 5)` = 0x1E17"
        );
        assert_eq!(
            (wcr >> 5) & 0xFF,
            0xF0,
            "BAS (bits 12:5) must be 0xF0 for offset=4 (4 \
             contiguous high bytes)"
        );
    }

    /// DBGWVR alignment: the WCR/WVR pair always uses an 8-byte
    /// aligned base and BAS to select the 4 bytes, so a 4-byte
    /// aligned KVA at offset 4 must yield WVR = `kva & ~0x7` (= the
    /// containing 8-byte block's base), not `kva` itself.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn dbgwvr_8byte_aligned_base() {
        let kva: u64 = 0xffff_ffff_8100_1004;
        let wvr = kva & !0x7u64;
        assert_eq!(
            wvr, 0xffff_ffff_8100_1000,
            "DBGWVR base must clear the bottom 3 bits (8-byte align) \
             so BAS picks the 4 watched bytes within the block"
        );
        // Round-trip: reconstructing the watched range from WVR + BAS.
        let byte_offset = (kva & 0x7u64) as u32;
        let bas: u64 = 0xFu64 << byte_offset;
        let watched_lo = wvr + (bas.trailing_zeros() as u64);
        let watched_hi = watched_lo + (bas.count_ones() as u64);
        assert_eq!(
            watched_lo, kva,
            "watched range low must equal the original KVA"
        );
        assert_eq!(
            watched_hi,
            kva + 4,
            "watched range high must equal kva + 4 (4 bytes)"
        );
    }

    /// FAR-based slot decode for the aarch64 watchpoint exit path.
    /// Constructs a synthetic `kvm_debug_exit_arch` with EC =
    /// WATCHPT_LOW and a FAR inside slot 2's 4-byte window, runs
    /// the dispatch helper, and asserts only `user[1].hit` was
    /// latched (slot indices in `armed_slots` are 0=DR0/exit_kind,
    /// 1=DR1/user[0], 2=DR2/user[1], 3=DR3/user[2]). Also pins the
    /// post-fire single-step bookkeeping: the helper must mark
    /// `single_step_pending=true` and record the matched slot
    /// index, so the next `self_arm_watchpoint` call disables that
    /// slot's WCR.E and asserts `KVM_GUESTDBG_SINGLESTEP` (avoiding
    /// the aarch64 watchpoint-replay infinite loop).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn watchpoint_slot_decode_from_far_user_slot() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        // Slot 0 unused (request_kva = 0); slots 1..=3 carry
        // distinct addresses 4 bytes apart so a FAR inside slot 2's
        // window matches exactly one slot.
        let armed_slots: [u64; 4] = [
            0,
            0xffff_ffff_8100_1000,
            0xffff_ffff_8100_1004,
            0xffff_ffff_8100_1008,
        ];
        // Construct a synthetic debug-exit payload pointing at the
        // first byte of slot 2 (DR2 / user[1]).
        let far = 0xffff_ffff_8100_1004u64;
        let hsr = (super::ESR_ELx_EC_WATCHPT_LOW) << super::ESR_ELx_EC_SHIFT;
        let debug_arch = kvm_bindings::kvm_debug_exit_arch {
            hsr,
            hsr_high: 0,
            far,
        };
        let mut single_step_pending = false;
        let mut single_step_slot: usize = 99;
        super::dispatch_watchpoint_hit(
            &watchpoint,
            &debug_arch,
            &armed_slots,
            &mut single_step_pending,
            &mut single_step_slot,
        );
        assert!(
            !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "slot 0 (exit_kind) must not latch when a different \
             slot's range matches FAR"
        );
        assert!(
            !watchpoint.user[0]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[0] / slot 1 must not latch — FAR is inside slot 2's \
             range"
        );
        assert!(
            watchpoint.user[1]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[1] / slot 2 must latch — FAR equals the slot's KVA"
        );
        assert!(
            !watchpoint.user[2]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[2] / slot 3 must not latch — FAR is outside its \
             range"
        );
        // Single-step bookkeeping: a watchpoint match MUST request
        // single-step on the matching slot so the next KVM_RUN
        // advances past the offending store before the slot is
        // re-armed. `single_step_slot` is a 4-bit bitmap of
        // matched slot indices; with FAR inside slot 2's range,
        // bit 2 must be the only bit set so `self_arm_watchpoint`
        // clears WCR[2].E (and leaves peer slots armed) for the
        // single-step pass.
        assert!(
            single_step_pending,
            "single_step_pending must be set when a watchpoint match \
             latches; without this the next KVM_RUN replays the same \
             store and re-trips the watchpoint forever (ARM ARM \
             D2.10.5)"
        );
        assert_eq!(
            single_step_slot, 0b0100,
            "single_step_slot bitmap must encode slot 2 (bit 2 = 1, \
             0b0100) so self_arm_watchpoint clears WCR[2].E and \
             leaves WCR[0/1/3].E armed for the single-step pass"
        );
    }

    /// Non-watchpoint EC values must be ignored. KVM_EXIT_DEBUG can
    /// surface for soft-step (EC = 0x32) or BRK (EC = 0x3C); only
    /// EC = 0x34 (`ESR_ELx_EC_WATCHPT_LOW`) means a data watchpoint
    /// fired. Other ECs must not latch any slot. Soft-step (EC =
    /// 0x32) is treated specially when `single_step_pending` is
    /// set — it clears the flag so `self_arm_watchpoint` can
    /// restore the disabled slot's WCR.E. With the flag clear,
    /// soft-step exits are spurious and must NOT touch any state.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn watchpoint_dispatch_ignores_non_watchpt_ec() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        let armed_slots: [u64; 4] = [
            0xffff_ffff_8100_1000,
            0xffff_ffff_8100_1004,
            0xffff_ffff_8100_1008,
            0xffff_ffff_8100_100C,
        ];
        // EC = 0x32 (software step) with single_step_pending = false
        // — must NOT latch and must not flip pending state.
        let hsr = 0x32u32 << super::ESR_ELx_EC_SHIFT;
        let debug_arch = kvm_bindings::kvm_debug_exit_arch {
            hsr,
            hsr_high: 0,
            far: 0xffff_ffff_8100_1004,
        };
        let mut single_step_pending = false;
        let mut single_step_slot: usize = 99;
        super::dispatch_watchpoint_hit(
            &watchpoint,
            &debug_arch,
            &armed_slots,
            &mut single_step_pending,
            &mut single_step_slot,
        );
        assert!(
            !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "soft-step EC must not latch slot 0"
        );
        for (i, slot) in watchpoint.user.iter().enumerate() {
            assert!(
                !slot.hit.load(std::sync::atomic::Ordering::Acquire),
                "soft-step EC must not latch user[{i}]"
            );
        }
        assert!(
            !single_step_pending,
            "spurious soft-step exit (no pending step) must leave \
             single_step_pending unchanged"
        );
        assert_eq!(
            single_step_slot, 99,
            "spurious soft-step exit must not clobber single_step_slot"
        );
    }

    /// Soft-step exit AFTER a watchpoint trap is the second half of
    /// the aarch64 watchpoint-replay-avoidance dance: it signals
    /// "the offending store has retired, you may rearm the slot
    /// now". The dispatch helper must clear `single_step_pending`
    /// (so the next `self_arm_watchpoint` call restores WCR.E=1)
    /// AND must not latch any new `hit` (the original
    /// WATCHPT_LOW exit already latched the freeze trigger).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn watchpoint_softstep_clears_single_step_pending() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        let armed_slots: [u64; 4] = [0, 0xffff_ffff_8100_1000, 0, 0];
        let hsr = super::ESR_ELx_EC_SOFTSTP_LOW << super::ESR_ELx_EC_SHIFT;
        let debug_arch = kvm_bindings::kvm_debug_exit_arch {
            hsr,
            hsr_high: 0,
            far: 0,
        };
        let mut single_step_pending = true;
        let mut single_step_slot: usize = 1;
        super::dispatch_watchpoint_hit(
            &watchpoint,
            &debug_arch,
            &armed_slots,
            &mut single_step_pending,
            &mut single_step_slot,
        );
        assert!(
            !single_step_pending,
            "SOFTSTP_LOW with pending step must clear \
             single_step_pending so the next self_arm_watchpoint \
             call restores WCR.E=1 and drops KVM_GUESTDBG_SINGLESTEP"
        );
        assert!(
            !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "SOFTSTP_LOW must not latch slot 0 (the WATCHPT_LOW \
             exit that preceded it already did)"
        );
        for (i, slot) in watchpoint.user.iter().enumerate() {
            assert!(
                !slot.hit.load(std::sync::atomic::Ordering::Acquire),
                "SOFTSTP_LOW must not latch user[{i}]"
            );
        }
    }

    /// Slot 0 (`exit_kind`) MUST NOT latch when `kind_host_ptr`
    /// is null. The freeze coordinator publishes `kind_host_ptr`
    /// (Release) BEFORE `request_kva` (Release); the vCPU's
    /// `self_arm_watchpoint` only programs the hardware
    /// watchpoint after observing a non-zero `request_kva` via
    /// an Acquire load — at which point both stores are visible.
    /// A null observation here is a publication-invariant
    /// violation. Pinning the no-latch behavior so a future
    /// regression that re-introduces an unconditional fallback
    /// (e.g. "latch hit just in case") is caught: latching on
    /// null defeats the post-store error-class gate and re-fires
    /// dump emission on every clean SCX_EXIT_DONE shutdown.
    /// The single-step bookkeeping STILL fires on the matched
    /// slot — re-entering KVM_RUN without stepping past the
    /// offending store would replay the same trap forever
    /// regardless of whether the slot-0 latch ran.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn watchpoint_slot0_skips_latch_when_host_ptr_null() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        // host_ptr is null by construction (`AtomicPtr::new(null)`).
        let armed_slots: [u64; 4] = [0xffff_ffff_8100_1000, 0, 0, 0];
        let hsr = (super::ESR_ELx_EC_WATCHPT_LOW) << super::ESR_ELx_EC_SHIFT;
        let debug_arch = kvm_bindings::kvm_debug_exit_arch {
            hsr,
            hsr_high: 0,
            far: 0xffff_ffff_8100_1000,
        };
        let mut single_step_pending = false;
        let mut single_step_slot: usize = 0;
        super::dispatch_watchpoint_hit(
            &watchpoint,
            &debug_arch,
            &armed_slots,
            &mut single_step_pending,
            &mut single_step_slot,
        );
        assert!(
            !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "slot 0 must NOT latch hit when kind_host_ptr is null — \
             a null observation is a publication-invariant violation, \
             not a fallback trigger"
        );
        // Single-step bookkeeping must still record the matched
        // slot in the bitmap so `self_arm_watchpoint` clears
        // WCR.E on it. Without this the offending store replays
        // forever on re-entering KVM_RUN.
        assert!(
            single_step_pending,
            "single_step_pending must be set when the FAR matches a \
             slot, regardless of slot-0 latch outcome"
        );
        assert_eq!(
            single_step_slot, 0b0001,
            "single_step_slot bitmap must include slot 0 (bit 0 = 1) \
             so self_arm_watchpoint clears WCR[0].E during the \
             single-step pass"
        );
    }

    /// x86_64 sibling of `watchpoint_slot_decode_from_far_user_slot`.
    /// Constructs a synthetic `kvm_debug_exit_arch` with DR6=0x4 (B2
    /// set, indicating DR2 fired) and verifies the dispatch helper
    /// latches `user[1].hit` (slot 2 → user[1]) and leaves slot 0
    /// and the other user slots untouched. Pins the DR6→slot
    /// mapping so a future refactor that flips the bit-shift
    /// (e.g. interprets bit 2 as slot 1 instead of slot 2) is
    /// caught at unit-test time.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn watchpoint_dispatch_x86_dr6_b2_latches_user_slot_1() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        // armed_slots is consumed by the aarch64 path only; x86
        // uses DR6 alone.
        let armed_slots: [u64; 4] = [0; 4];
        let debug_arch = kvm_bindings::kvm_debug_exit_arch {
            exception: 0,
            pad: 0,
            pc: 0,
            // DR6 bit 2 (B2) set ⇒ DR2 fired (slot 2 in our
            // 0=DR0..3=DR3 indexing). Other bits cleared per Intel
            // SDM Vol. 3B 17.2.5.
            dr6: 0x4,
            dr7: 0,
        };
        let mut single_step_pending = false;
        let mut single_step_slot: usize = 99;
        super::dispatch_watchpoint_hit(
            &watchpoint,
            &debug_arch,
            &armed_slots,
            &mut single_step_pending,
            &mut single_step_slot,
        );
        assert!(
            !watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "slot 0 (exit_kind) must not latch when DR6 B0 is clear"
        );
        assert!(
            !watchpoint.user[0]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[0] / slot 1 must not latch — DR6 B1 is clear"
        );
        assert!(
            watchpoint.user[1]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[1] / slot 2 must latch — DR6 B2 is set"
        );
        assert!(
            !watchpoint.user[2]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[2] / slot 3 must not latch — DR6 B3 is clear"
        );
        // x86 single-step bookkeeping is inert (the trap is taken
        // AFTER the offending store retires — Intel SDM 17.2.4).
        assert!(
            !single_step_pending,
            "x86 dispatch must never set single_step_pending — \
             single-step is aarch64-only"
        );
        assert_eq!(
            single_step_slot, 99,
            "x86 dispatch must not clobber single_step_slot — \
             single-step is aarch64-only"
        );
    }

    /// x86_64 multi-match: DR6=0x5 (B0 + B2) means DR0 and DR2
    /// both fired on the same exit. The dispatch helper must
    /// latch BOTH slot 0 and user[1] from a single
    /// `kvm_debug_exit_arch`. Catches a refactor that breaks at
    /// the first set bit (e.g. early `return` after `if hits[0]`).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn watchpoint_dispatch_x86_dr6_multi_match() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        // Slot 0 needs a non-null kind_host_ptr to latch via the
        // post-store value gate. Pre-arm it with a host-side u32
        // holding an error-class kind value so
        // `latch_slot0_with_gate` takes the latch branch instead
        // of the gated-suppression branch.
        let kind: u32 = super::SCX_EXIT_ERROR_THRESHOLD;
        let kind_box = Box::new(kind);
        let kind_ptr = Box::into_raw(kind_box);
        watchpoint
            .kind_host_ptr
            .store(kind_ptr, std::sync::atomic::Ordering::Release);
        let armed_slots: [u64; 4] = [0; 4];
        let debug_arch = kvm_bindings::kvm_debug_exit_arch {
            exception: 0,
            pad: 0,
            pc: 0,
            // DR6 bits 0 + 2 set: DR0 and DR2 both fired.
            dr6: 0x5,
            dr7: 0,
        };
        let mut single_step_pending = false;
        let mut single_step_slot: usize = 99;
        super::dispatch_watchpoint_hit(
            &watchpoint,
            &debug_arch,
            &armed_slots,
            &mut single_step_pending,
            &mut single_step_slot,
        );
        assert!(
            watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "slot 0 (exit_kind) must latch — DR6 B0 set + kind ≥ \
             SCX_EXIT_ERROR_THRESHOLD"
        );
        assert!(
            !watchpoint.user[0]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[0] / slot 1 must not latch — DR6 B1 is clear"
        );
        assert!(
            watchpoint.user[1]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[1] / slot 2 must latch — DR6 B2 is set, even \
             though slot 0 latched first in iteration order"
        );
        assert!(
            !watchpoint.user[2]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "user[2] / slot 3 must not latch — DR6 B3 is clear"
        );
        // SAFETY: kind_box ownership round-trip via Box::into_raw
        // / Box::from_raw matches the standard pattern; we own
        // the only pointer to this allocation in the test
        // function and the dispatch helper has finished its
        // read_volatile before this drop.
        let _ = unsafe { Box::from_raw(kind_ptr) };
    }

    /// CAS-based dedup in `latch_hit`: a second call (e.g. two
    /// vCPUs each writing the watched address, or a re-fire on
    /// the same vCPU before the freeze coordinator resets `hit`)
    /// must not write a second eventfd edge. Pinning the dedup so
    /// a future refactor that drops the CAS in favour of an
    /// unconditional `store(true)` is caught here — coordinator
    /// would then rendezvous N times for one logical fire.
    #[test]
    fn latch_hit_is_idempotent_across_repeat_calls() {
        use crate::vmm::vcpu::WatchpointArm;
        use std::os::fd::AsRawFd;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");

        // First latch: must flip false→true and write eventfd.
        watchpoint.latch_hit();
        assert!(
            watchpoint.hit.load(std::sync::atomic::Ordering::Acquire),
            "first latch_hit must flip hit=false→true"
        );

        // Drain the eventfd to verify there is exactly one write
        // pending. EFD_NONBLOCK + counter mode: a single read
        // returns the accumulated count and resets the internal
        // counter.
        let mut buf = [0u8; 8];
        let n = unsafe {
            libc::read(
                watchpoint.hit_evt.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };
        assert_eq!(
            n, 8,
            "first latch_hit must produce one eventfd edge \
             (8-byte counter read)"
        );
        let count_after_first = u64::from_ne_bytes(buf);
        assert_eq!(
            count_after_first, 1,
            "first latch_hit must increment counter by exactly 1"
        );

        // Second latch (without coordinator reset): CAS fails,
        // no eventfd write. Counter stays at 0 — a subsequent
        // read returns EAGAIN (no edge available).
        watchpoint.latch_hit();
        let mut buf2 = [0u8; 8];
        let n2 = unsafe {
            libc::read(
                watchpoint.hit_evt.as_raw_fd(),
                buf2.as_mut_ptr() as *mut libc::c_void,
                buf2.len(),
            )
        };
        let errno = unsafe { *libc::__errno_location() };
        assert!(
            n2 == -1 && errno == libc::EAGAIN,
            "second latch_hit on already-latched slot must NOT \
             write to hit_evt (cross-vCPU dedup); read should \
             return EAGAIN, got n={n2} errno={errno}"
        );
    }

    /// CAS-based dedup in `latch_user_hit`: same invariant as the
    /// slot-0 latch, applied to user slots 1..=3. Catches an off-
    /// by-one in the per-slot CAS (e.g. CAS on slot 0's `hit`
    /// instead of `user[idx].hit`).
    #[test]
    fn latch_user_hit_is_idempotent_across_repeat_calls() {
        use crate::vmm::vcpu::WatchpointArm;
        use std::os::fd::AsRawFd;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");

        watchpoint.latch_user_hit(1);
        assert!(
            watchpoint.user[1]
                .hit
                .load(std::sync::atomic::Ordering::Acquire),
            "first latch_user_hit(1) must flip user[1].hit=false→true"
        );

        let mut buf = [0u8; 8];
        let n = unsafe {
            libc::read(
                watchpoint.hit_evt.as_raw_fd(),
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };
        assert_eq!(n, 8, "first latch_user_hit(1) must write eventfd");
        let count_after_first = u64::from_ne_bytes(buf);
        assert_eq!(count_after_first, 1, "counter increment must be 1");

        watchpoint.latch_user_hit(1);
        let mut buf2 = [0u8; 8];
        let n2 = unsafe {
            libc::read(
                watchpoint.hit_evt.as_raw_fd(),
                buf2.as_mut_ptr() as *mut libc::c_void,
                buf2.len(),
            )
        };
        let errno = unsafe { *libc::__errno_location() };
        assert!(
            n2 == -1 && errno == libc::EAGAIN,
            "second latch_user_hit(1) on already-latched slot \
             must NOT write to hit_evt; read should return \
             EAGAIN, got n={n2} errno={errno}"
        );

        // Out-of-range idx must be a silent no-op — no panic, no
        // hit on any slot, no eventfd write. Catches a future
        // refactor that drops the bounds check.
        watchpoint.latch_user_hit(99);
        for (i, slot) in watchpoint.user.iter().enumerate() {
            if i == 1 {
                assert!(
                    slot.hit.load(std::sync::atomic::Ordering::Acquire),
                    "user[1].hit must remain latched"
                );
            } else {
                assert!(
                    !slot.hit.load(std::sync::atomic::Ordering::Acquire),
                    "user[{i}].hit must remain unlatched after \
                     out-of-range latch_user_hit(99)"
                );
            }
        }
    }

    /// `WatchpointArm::mark_armed` flips the `any_armed` gate
    /// from 0 to 1 and is idempotent. Pins the gate's role: the
    /// publisher (freeze coordinator's err_exit publish or
    /// `arm_user_watchpoint`) calls `mark_armed` AFTER the
    /// `Release` store on `request_kva`; until then
    /// `self_arm_watchpoint` short-circuits without doing the
    /// per-slot Acquire reads.
    #[test]
    fn mark_armed_flips_gate_and_is_idempotent() {
        use crate::vmm::vcpu::WatchpointArm;
        let watchpoint = WatchpointArm::new().expect("WatchpointArm::new");
        assert_eq!(
            watchpoint
                .any_armed
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "newly-constructed WatchpointArm must have any_armed=0 \
             so self_arm_watchpoint short-circuits before any \
             publisher fires"
        );
        watchpoint.mark_armed();
        assert_eq!(
            watchpoint
                .any_armed
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "first mark_armed call must flip the gate to 1"
        );
        // Idempotence: a second call must leave the gate at 1.
        // Catches a refactor that turned mark_armed into an
        // increment / fetch_add (which would saturate eventually
        // but burn cycles on every publisher call).
        watchpoint.mark_armed();
        assert_eq!(
            watchpoint
                .any_armed
                .load(std::sync::atomic::Ordering::Relaxed),
            1,
            "second mark_armed call must leave the gate at 1 \
             (idempotent — mark_armed is `store(1)`, not `+= 1`)"
        );
    }
}
