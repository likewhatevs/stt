/// 16550A UART emulation via vm-superio.
///
/// Thin wrapper around vm-superio::Serial providing port-based I/O
/// dispatch and output capture. On x86_64, UARTs are accessed via port
/// I/O at COM1 (0x3F8) and COM2 (0x2F8). On aarch64, UARTs are
/// MMIO-mapped. COM1 carries the kernel console, COM2 carries
/// application stdout/stderr.
pub(crate) const COM1_BASE: u16 = 0x3F8;
pub(crate) const COM2_BASE: u16 = 0x2F8;

/// ISA IRQ lines for the two COM ports (x86_64 port I/O only).
#[cfg(target_arch = "x86_64")]
pub(crate) const COM1_IRQ: u32 = 4;
#[cfg(target_arch = "x86_64")]
pub(crate) const COM2_IRQ: u32 = 3;

const NUM_REGS: u16 = 8;

/// EventFd-based interrupt trigger. Signals the eventfd on each
/// interrupt, which KVM's irqfd mechanism delivers to the guest as
/// the configured IRQ line assertion.
struct EventFdTrigger(vmm_sys_util::eventfd::EventFd);

impl vm_superio::Trigger for EventFdTrigger {
    type E = std::io::Error;
    fn trigger(&self) -> Result<(), Self::E> {
        self.0.write(1)
    }
}

/// Port-addressed serial wrapping vm-superio::Serial with output capture.
pub struct Serial {
    base: u16,
    inner: vm_superio::Serial<EventFdTrigger, vm_superio::serial::NoEvents, Vec<u8>>,
}

impl Default for Serial {
    fn default() -> Self {
        Self::new(COM1_BASE)
    }
}

impl Serial {
    pub fn new(base: u16) -> Self {
        let evt = vmm_sys_util::eventfd::EventFd::new(libc::EFD_NONBLOCK)
            .expect("failed to create serial eventfd");
        Serial {
            base,
            inner: vm_superio::Serial::new(EventFdTrigger(evt), Vec::new()),
        }
    }

    /// Return the interrupt eventfd for registering with KVM's irqfd.
    pub fn irq_evt(&self) -> &vmm_sys_util::eventfd::EventFd {
        &self.inner.interrupt_evt().0
    }

    /// Handle a port I/O write from the guest. Returns true if the port
    /// is in this serial's range.
    pub fn handle_out(&mut self, port: u16, data: &[u8]) -> bool {
        let Some(offset) = self.offset(port) else {
            return false;
        };
        if let Some(&byte) = data.first() {
            let _ = self.inner.write(offset, byte);
        }
        true
    }

    /// Handle a port I/O read from the guest. Returns true if the port
    /// is in this serial's range.
    pub fn handle_in(&mut self, port: u16, data: &mut [u8]) -> bool {
        let Some(offset) = self.offset(port) else {
            return false;
        };
        if let Some(first) = data.first_mut() {
            *first = self.inner.read(offset);
        }
        true
    }

    /// Write a byte to a register at the given offset.
    /// Used by MMIO dispatch where the offset is computed externally.
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn inner_write(&mut self, offset: u8, byte: u8) {
        let _ = self.inner.write(offset, byte);
    }

    /// Read a byte from a register at the given offset.
    /// Used by MMIO dispatch where the offset is computed externally.
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn inner_read(&mut self, offset: u8) -> u8 {
        self.inner.read(offset)
    }

    /// Queue input bytes for host->guest communication.
    pub fn queue_input(&mut self, bytes: &[u8]) {
        let _ = self.inner.enqueue_raw_bytes(bytes);
    }

    /// Return and clear accumulated output. O(1) via buffer swap.
    pub fn drain_output(&mut self) -> Vec<u8> {
        std::mem::take(self.inner.writer_mut())
    }

    /// Get all captured output as a string.
    pub fn output(&self) -> String {
        String::from_utf8_lossy(self.inner.writer()).to_string()
    }

    /// Get output bytes.
    #[cfg(test)]
    pub fn output_bytes(&self) -> &[u8] {
        self.inner.writer()
    }

    /// Clear captured output.
    #[cfg(test)]
    pub fn clear(&mut self) {
        self.inner.writer_mut().clear();
    }

    fn offset(&self, port: u16) -> Option<u8> {
        if port >= self.base && port < self.base + NUM_REGS {
            Some((port - self.base) as u8)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Register offsets for test readability
    const DATA: u16 = 0;
    const IER: u16 = 1;
    const IIR: u16 = 2;
    const FCR: u16 = 2;
    const LCR: u16 = 3;
    const MCR: u16 = 4;
    const LSR: u16 = 5;
    const MSR: u16 = 6;
    const SCR: u16 = 7;

    // Bit constants for tests
    const IER_RDI: u8 = 0x01;
    const IER_THRI: u8 = 0x02;
    const IIR_FIFO_BITS: u8 = 0xC0;
    const IIR_NONE: u8 = 0x01;
    const IIR_THRI: u8 = 0x02;
    const IIR_RDI: u8 = 0x04;
    const LCR_DLAB: u8 = 0x80;
    const LSR_DR: u8 = 0x01;
    const LSR_THRE: u8 = 0x20;
    const LSR_TEMT: u8 = 0x40;
    const MCR_DTR: u8 = 0x01;
    const MCR_RTS: u8 = 0x02;
    const MCR_OUT1: u8 = 0x04;
    const MCR_OUT2: u8 = 0x08;
    const MCR_LOOP: u8 = 0x10;
    const MSR_CTS: u8 = 0x10;
    const MSR_DSR: u8 = 0x20;
    const MSR_RI: u8 = 0x40;
    const MSR_DCD: u8 = 0x80;
    const FIFO_SIZE: usize = 0x40;

    #[test]
    fn new_serial_defaults() {
        let s = Serial::default();
        assert!(s.output().is_empty());
    }

    #[test]
    fn write_thr_captures() {
        let mut s = Serial::default();
        assert!(s.handle_out(COM1_BASE, b"H"));
        assert!(s.handle_out(COM1_BASE, b"i"));
        assert_eq!(s.output(), "Hi");
    }

    #[test]
    fn write_thr_multi_byte() {
        let mut s = Serial::default();
        // handle_out only writes first byte
        s.handle_out(COM1_BASE, b"Hello");
        assert_eq!(s.output(), "H");
    }

    #[test]
    fn read_lsr_thre_temt() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut buf);
        assert_ne!(buf[0] & LSR_THRE, 0, "THR should report empty");
        assert_ne!(buf[0] & LSR_TEMT, 0, "transmitter should report empty");
    }

    #[test]
    fn iir_no_interrupt_at_start() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut buf);
        assert_eq!(buf[0] & 0x0F, IIR_NONE, "no interrupt pending");
        assert_ne!(buf[0] & IIR_FIFO_BITS, 0, "FIFO bits should be set");
    }

    #[test]
    fn iir_read_resets_all_interrupts() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + IER, &[IER_RDI | IER_THRI]);
        s.handle_out(COM1_BASE + DATA, b"X");
        s.queue_input(&[0x42]);

        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & IIR_NONE, 0, "should have interrupt pending");

        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_NONE, "cleared after read");
    }

    #[test]
    fn write_read_lcr() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + LCR, &[0x1B]);
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + LCR, &mut buf);
        assert_eq!(buf[0], 0x1B);
    }

    #[test]
    fn write_read_scr() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + SCR, &[0x42]);
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + SCR, &mut buf);
        assert_eq!(buf[0], 0x42);
    }

    #[test]
    fn out_of_range_port() {
        let mut s = Serial::default();
        assert!(!s.handle_out(0x100, b"x"));
        let mut buf = [0u8; 1];
        assert!(!s.handle_in(0x100, &mut buf));
    }

    #[test]
    fn clear_output() {
        let mut s = Serial::default();
        for c in b"test" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        assert_eq!(s.output(), "test");
        s.clear();
        assert!(s.output().is_empty());
    }

    #[test]
    fn dlab_baud_divisor() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + LCR, &[0x03 | LCR_DLAB]);
        s.handle_out(COM1_BASE + DATA, &[0x01]);
        s.handle_out(COM1_BASE + IER, &[0x00]);
        let mut lo = [0u8; 1];
        let mut hi = [0u8; 1];
        s.handle_in(COM1_BASE + DATA, &mut lo);
        s.handle_in(COM1_BASE + IER, &mut hi);
        assert_eq!(lo[0], 0x01);
        assert_eq!(hi[0], 0x00);
        s.handle_out(COM1_BASE + LCR, &[0x03]);
        s.handle_out(COM1_BASE + DATA, b"X");
        assert_eq!(s.output(), "X");
    }

    #[test]
    fn loopback_mode() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP]);
        s.handle_out(COM1_BASE + DATA, &[0xAA]);
        assert!(s.output().is_empty(), "loopback should not produce output");
        let mut lsr = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0] & LSR_DR, 0, "data should be ready");
        let mut data = [0u8; 1];
        s.handle_in(COM1_BASE + DATA, &mut data);
        assert_eq!(data[0], 0xAA);
    }

    #[test]
    fn loopback_msr_mapping() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP]);
        let mut msr = [0u8; 1];
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0], 0x00, "no MCR outputs set");

        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP | MCR_OUT2 | MCR_RTS]);
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0], MSR_DCD | MSR_CTS, "OUT2->DCD, RTS->CTS");

        s.handle_out(
            COM1_BASE + MCR,
            &[MCR_LOOP | MCR_DTR | MCR_RTS | MCR_OUT1 | MCR_OUT2],
        );
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0], MSR_DSR | MSR_CTS | MSR_RI | MSR_DCD);
    }

    #[test]
    fn thr_interrupt_lifecycle() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + IER, &[IER_THRI]);
        s.handle_out(COM1_BASE + DATA, b"A");
        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_THRI, "THR empty interrupt");
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_NONE, "cleared after read");
    }

    #[test]
    fn receive_interrupt() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + IER, &[IER_RDI]);
        s.queue_input(&[0x42]);
        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_RDI, "receive data interrupt");
        let mut data = [0u8; 1];
        s.handle_in(COM1_BASE + DATA, &mut data);
        assert_eq!(data[0], 0x42);
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0x0F, IIR_NONE, "cleared");
    }

    #[test]
    fn queue_input_respects_fifo_size() {
        let mut s = Serial::default();
        let big = vec![0xAA; FIFO_SIZE + 10];
        s.queue_input(&big);
        // vm-superio caps at FIFO_SIZE
        assert_eq!(s.inner.fifo_capacity(), 0);
    }

    #[test]
    fn kernel_autoconfig_ier_test() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_out(COM1_BASE + IER, &[0x00]);
        s.handle_in(COM1_BASE + IER, &mut buf);
        assert_eq!(buf[0] & 0x0F, 0x00);
        s.handle_out(COM1_BASE + IER, &[0x0F]);
        s.handle_in(COM1_BASE + IER, &mut buf);
        assert_eq!(buf[0] & 0x0F, 0x0F);
    }

    #[test]
    fn kernel_autoconfig_loopback_test() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + MCR, &[MCR_LOOP | MCR_OUT2 | MCR_RTS]);
        let mut msr = [0u8; 1];
        s.handle_in(COM1_BASE + MSR, &mut msr);
        assert_eq!(msr[0] & 0xF0, MSR_DCD | MSR_CTS);
    }

    #[test]
    fn kernel_lsr_safety_check() {
        let mut s = Serial::default();
        let mut lsr = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0], 0xFF, "LSR must not be 0xFF");
    }

    #[test]
    fn kernel_fifo_detection() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE + FCR, &[0x01]);
        let mut iir = [0u8; 1];
        s.handle_in(COM1_BASE + IIR, &mut iir);
        assert_eq!(iir[0] & 0xC0, 0xC0, "16550A FIFO");
    }

    #[test]
    fn com2_write_captures() {
        let mut s = Serial::new(COM2_BASE);
        assert!(s.handle_out(COM2_BASE, b"H"));
        assert!(s.handle_out(COM2_BASE, b"i"));
        assert_eq!(s.output(), "Hi");
    }

    #[test]
    fn com2_rejects_com1_port() {
        let mut s = Serial::new(COM2_BASE);
        assert!(!s.handle_out(COM1_BASE, b"x"));
        let mut buf = [0u8; 1];
        assert!(!s.handle_in(COM1_BASE, &mut buf));
    }

    #[test]
    fn com1_rejects_com2_port() {
        let mut s = Serial::default();
        assert!(!s.handle_out(COM2_BASE, b"x"));
        let mut buf = [0u8; 1];
        assert!(!s.handle_in(COM2_BASE, &mut buf));
    }

    #[test]
    fn com2_loopback_msr() {
        let mut s = Serial::new(COM2_BASE);
        s.handle_out(COM2_BASE + MCR, &[MCR_LOOP | MCR_OUT2 | MCR_RTS]);
        let mut msr = [0u8; 1];
        s.handle_in(COM2_BASE + MSR, &mut msr);
        assert_eq!(msr[0], MSR_DCD | MSR_CTS);
    }

    #[test]
    fn com2_lsr_defaults() {
        let mut s = Serial::new(COM2_BASE);
        let mut lsr = [0u8; 1];
        s.handle_in(COM2_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0] & LSR_THRE, 0);
        assert_ne!(lsr[0] & LSR_TEMT, 0);
    }

    #[test]
    fn dual_serial_isolation() {
        let mut com1 = Serial::new(COM1_BASE);
        let mut com2 = Serial::new(COM2_BASE);
        com1.handle_out(COM1_BASE, b"A");
        com2.handle_out(COM2_BASE, b"B");
        assert_eq!(com1.output(), "A");
        assert_eq!(com2.output(), "B");
    }

    #[test]
    fn queue_input_overflow_caps() {
        let mut s = Serial::new(COM1_BASE);
        let data = vec![0x42; FIFO_SIZE + 100];
        s.queue_input(&data);
        assert_eq!(s.inner.fifo_capacity(), 0);
    }

    #[test]
    fn dlab_rapid_transitions() {
        let mut s = Serial::new(COM1_BASE);
        s.handle_out(COM1_BASE + LCR, &[0x80]);
        s.handle_out(COM1_BASE + DATA, &[0x01]);
        s.handle_out(COM1_BASE + LCR, &[0x03]);
        s.handle_out(COM1_BASE + DATA, b"Z");
        assert_eq!(s.output(), "Z");
    }

    #[test]
    fn write_multi_byte_output() {
        let mut s = Serial::new(COM1_BASE);
        for c in b"HELLO" {
            s.handle_out(COM1_BASE + DATA, &[*c]);
        }
        assert_eq!(s.output(), "HELLO");
    }

    #[test]
    fn msr_default_not_loopback() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + MSR, &mut buf);
        // vm-superio defaults MSR to DEFAULT_MODEM_STATUS (DCD).
        // Kernel autoconfig probes MSR in loopback mode, so default doesn't matter.
        assert_eq!(buf[0] & 0x80, 0x80);
    }

    #[test]
    fn mcr_default_value() {
        let mut s = Serial::default();
        let mut buf = [0u8; 1];
        s.handle_in(COM1_BASE + MCR, &mut buf);
        // vm-superio defaults MCR to MCR_OUT2 (0x08).
        assert_eq!(buf[0], MCR_OUT2);
    }

    #[test]
    fn drain_output_returns_and_clears() {
        let mut s = Serial::default();
        for c in b"hello" {
            s.handle_out(COM1_BASE, &[*c]);
        }
        let drained = s.drain_output();
        assert_eq!(drained, b"hello");
        assert!(s.output().is_empty(), "buffer should be empty after drain");
    }

    #[test]
    fn drain_output_empty() {
        let mut s = Serial::default();
        let drained = s.drain_output();
        assert!(drained.is_empty());
    }

    #[test]
    fn drain_output_incremental() {
        let mut s = Serial::default();
        s.handle_out(COM1_BASE, b"A");
        s.handle_out(COM1_BASE, b"B");
        let first = s.drain_output();
        assert_eq!(first, b"AB");

        s.handle_out(COM1_BASE, b"C");
        let second = s.drain_output();
        assert_eq!(second, b"C");
    }

    #[test]
    fn queue_input_pub_accessible() {
        let mut s = Serial::default();
        s.queue_input(&[0x41, 0x42]);
        let mut lsr = [0u8; 1];
        s.handle_in(COM1_BASE + LSR, &mut lsr);
        assert_ne!(lsr[0] & LSR_DR, 0, "data should be ready after queue_input");
    }
}
