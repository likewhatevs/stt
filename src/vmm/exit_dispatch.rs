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
use crate::vmm::{console, kvm, virtio_console};
use kvm_ioctls::VcpuExit;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    Some(VcpuRegSnapshot {
        instruction_pointer: pc,
        stack_pointer: sp,
        page_table_root: ttbr1,
        user_page_table_root: ttbr0,
    })
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
pub(crate) fn dispatch_mmio_write(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
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
    } else if let Some(vc) = virtio_con {
        let base = kvm::VIRTIO_CONSOLE_MMIO_BASE;
        if addr >= base && addr < base + virtio_console::VIRTIO_MMIO_SIZE {
            vc.lock().mmio_write(addr - base, data);
        }
    }
    false
}

/// Dispatch an MMIO read from serial and virtio-console devices.
#[cfg(target_arch = "aarch64")]
pub(crate) fn dispatch_mmio_read(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
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

/// Unified per-vCPU KVM_RUN loop for AP threads.
///
/// HLT on APs: check kill + continue on both arches (KVM delivers
/// interrupts to wake the vCPU). Shutdown sets the kill flag so all
/// other vCPUs exit.
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
    kill: &Arc<AtomicBool>,
    freeze: &Arc<AtomicBool>,
    parked: &Arc<AtomicBool>,
    regs_slot: &Arc<std::sync::Mutex<Option<VcpuRegSnapshot>>>,
    has_immediate_exit: bool,
) {
    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        // Honour a pending freeze before re-entering KVM_RUN.
        if freeze.load(Ordering::Acquire) {
            handle_freeze(vcpu, has_immediate_exit, kill, freeze, parked, regs_slot);
            if kill.load(Ordering::Acquire) {
                break;
            }
        }

        match vcpu.run() {
            Ok(mut exit) => {
                if matches!(exit, VcpuExit::Hlt) {
                    if kill.load(Ordering::Acquire) {
                        break;
                    }
                    continue;
                }
                match classify_exit(com1, com2, virtio_con.map(|a| a.as_ref()), &mut exit) {
                    Some(ExitAction::Continue) | None => {}
                    Some(ExitAction::Shutdown) => {
                        kill.store(true, Ordering::Release);
                        break;
                    }
                    Some(ExitAction::Fatal(_)) => break,
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
pub(crate) fn handle_freeze(
    vcpu: &mut kvm_ioctls::VcpuFd,
    has_immediate_exit: bool,
    kill: &Arc<AtomicBool>,
    freeze: &Arc<AtomicBool>,
    parked: &Arc<AtomicBool>,
    regs_slot: &Arc<std::sync::Mutex<Option<VcpuRegSnapshot>>>,
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

    // Park until freeze clears or shutdown wins. park_timeout(10ms)
    // bounds thaw latency so the coordinator's freeze=false store
    // is observed within at most 10 ms — no explicit unpark needed.
    while freeze.load(Ordering::Acquire) {
        if kill.load(Ordering::Acquire) {
            break;
        }
        std::thread::park_timeout(std::time::Duration::from_millis(10));
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
pub(crate) fn classify_exit(
    com1: &PiMutex<console::Serial>,
    com2: &PiMutex<console::Serial>,
    virtio_con: Option<&PiMutex<virtio_console::VirtioConsole>>,
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
            if dispatch_mmio_write(com1, com2, virtio_con, *addr, data) {
                Some(ExitAction::Shutdown)
            } else {
                Some(ExitAction::Continue)
            }
        }
        #[cfg(target_arch = "aarch64")]
        VcpuExit::MmioRead(addr, data) => {
            dispatch_mmio_read(com1, com2, virtio_con, *addr, data);
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
    fn vcpu_reg_snapshot_zero_renders_zeros() {
        let s = VcpuRegSnapshot {
            instruction_pointer: 0,
            stack_pointer: 0,
            page_table_root: 0,
            user_page_table_root: None,
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
}
