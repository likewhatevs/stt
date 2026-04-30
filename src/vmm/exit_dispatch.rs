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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    has_immediate_exit: bool,
) {
    loop {
        if kill.load(Ordering::Acquire) {
            break;
        }
        // Honour a pending freeze before re-entering KVM_RUN.
        if freeze.load(Ordering::Acquire) {
            handle_freeze(vcpu, has_immediate_exit, kill, freeze, parked);
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
