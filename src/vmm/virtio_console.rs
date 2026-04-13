use std::collections::VecDeque;
/// Single-port virtio-console with inline MMIO transport.
///
/// Two virtqueues (RX index 0, TX index 1), no multiport/control queues.
/// VIRTIO_F_VERSION_1 only. MMIO register layout per virtio-v1.2 §4.2.2.
/// Interrupt delivery via irqfd (eventfd → KVM GSI). TX data arrival
/// signaled via `tx_evt` eventfd for zero-latency host-side wakeup.
use virtio_bindings::virtio_config::{
    VIRTIO_CONFIG_S_ACKNOWLEDGE, VIRTIO_CONFIG_S_DRIVER, VIRTIO_CONFIG_S_DRIVER_OK,
    VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1,
};
use virtio_bindings::virtio_ids::VIRTIO_ID_CONSOLE;
use virtio_bindings::virtio_mmio::{
    VIRTIO_MMIO_CONFIG_GENERATION, VIRTIO_MMIO_DEVICE_FEATURES, VIRTIO_MMIO_DEVICE_FEATURES_SEL,
    VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_DRIVER_FEATURES, VIRTIO_MMIO_DRIVER_FEATURES_SEL,
    VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INTERRUPT_STATUS,
    VIRTIO_MMIO_MAGIC_VALUE, VIRTIO_MMIO_QUEUE_AVAIL_HIGH, VIRTIO_MMIO_QUEUE_AVAIL_LOW,
    VIRTIO_MMIO_QUEUE_DESC_HIGH, VIRTIO_MMIO_QUEUE_DESC_LOW, VIRTIO_MMIO_QUEUE_NOTIFY,
    VIRTIO_MMIO_QUEUE_NUM, VIRTIO_MMIO_QUEUE_NUM_MAX, VIRTIO_MMIO_QUEUE_READY,
    VIRTIO_MMIO_QUEUE_SEL, VIRTIO_MMIO_QUEUE_USED_HIGH, VIRTIO_MMIO_QUEUE_USED_LOW,
    VIRTIO_MMIO_STATUS, VIRTIO_MMIO_VENDOR_ID, VIRTIO_MMIO_VERSION,
};
use virtio_queue::{Queue, QueueT};
use vm_memory::{Bytes, GuestMemoryMmap};
use vmm_sys_util::eventfd::EventFd;

const MMIO_MAGIC: u32 = 0x7472_6976; // "virt" in LE
const MMIO_VERSION: u32 = 2; // virtio 1.x MMIO
const VENDOR_ID: u32 = 0;

/// MMIO region size: 4 KB (one page).
pub const VIRTIO_MMIO_SIZE: u64 = 0x1000;

const NUM_QUEUES: usize = 2;
const QUEUE_MAX_SIZE: u16 = 256;

const RXQ: usize = 0;
const TXQ: usize = 1;

/// Maximum bytes accepted from a single TX descriptor. The kernel's
/// virtio-console driver sends PAGE_SIZE chunks; this cap prevents a
/// malformed descriptor (len=0xFFFFFFFF) from triggering a ~4GB alloc.
const TX_DESC_MAX: usize = 32 * 1024;

/// Status bits required before each phase.
const S_ACK: u32 = VIRTIO_CONFIG_S_ACKNOWLEDGE;
const S_DRV: u32 = S_ACK | VIRTIO_CONFIG_S_DRIVER;
const S_FEAT: u32 = S_DRV | VIRTIO_CONFIG_S_FEATURES_OK;
const S_OK: u32 = S_FEAT | VIRTIO_CONFIG_S_DRIVER_OK;

/// Virtio-console MMIO device.
///
/// All state is behind a single struct — no separate transport layer.
/// The caller holds this in a `PiMutex` and dispatches MMIO reads/writes.
pub struct VirtioConsole {
    queues: [Queue; NUM_QUEUES],
    queue_select: u32,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    device_status: u32,
    interrupt_status: u32,
    config_generation: u32,
    /// Eventfd for KVM irqfd — signals guest interrupt.
    irq_evt: EventFd,
    /// Eventfd signaled when TX data is available. The host stdout
    /// thread polls this to wake immediately instead of sleeping.
    tx_evt: EventFd,
    /// Guest memory reference. Set before starting vCPUs.
    mem: Option<GuestMemoryMmap>,
    /// Accumulated output from guest TX queue (guest → host).
    tx_buf: Vec<u8>,
    /// Pending host input that could not be delivered because the guest
    /// RX queue had no available buffers. Drained on RX queue notify.
    pending_rx: VecDeque<u8>,
}

impl Default for VirtioConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtioConsole {
    /// Create a new virtio-console device.
    pub fn new() -> Self {
        let irq_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-console irq eventfd");
        let tx_evt =
            EventFd::new(libc::EFD_NONBLOCK).expect("failed to create virtio-console tx eventfd");
        VirtioConsole {
            queues: [
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
                Queue::new(QUEUE_MAX_SIZE).expect("valid queue size"),
            ],
            queue_select: 0,
            device_features_sel: 0,
            driver_features_sel: 0,
            driver_features: 0,
            device_status: 0,
            interrupt_status: 0,
            config_generation: 0,
            irq_evt,
            tx_evt,
            mem: None,
            tx_buf: Vec::new(),
            pending_rx: VecDeque::new(),
        }
    }

    /// Eventfd for KVM irqfd registration.
    pub fn irq_evt(&self) -> &EventFd {
        &self.irq_evt
    }

    /// Eventfd signaled when new TX data arrives. Use in the host-side
    /// stdout thread's poll set for zero-latency wakeup.
    pub fn tx_evt(&self) -> &EventFd {
        &self.tx_evt
    }

    /// Set guest memory reference. Must be called before starting vCPUs.
    pub fn set_mem(&mut self, mem: GuestMemoryMmap) {
        self.mem = Some(mem);
    }

    fn device_features(&self) -> u64 {
        1u64 << VIRTIO_F_VERSION_1
    }

    fn selected_queue(&self) -> Option<usize> {
        let idx = self.queue_select as usize;
        if idx < NUM_QUEUES { Some(idx) } else { None }
    }

    fn signal_used(&mut self) {
        self.interrupt_status |= VIRTIO_MMIO_INT_VRING;
        let _ = self.irq_evt.write(1);
    }

    /// True when device_status has progressed past FEATURES_OK but not
    /// yet reached DRIVER_OK — the window where queue config is valid.
    fn queue_config_allowed(&self) -> bool {
        self.device_status & S_FEAT == S_FEAT && self.device_status & VIRTIO_CONFIG_S_DRIVER_OK == 0
    }

    /// True when driver features may be written: DRIVER set, FEATURES_OK
    /// not yet set.
    fn features_write_allowed(&self) -> bool {
        self.device_status & S_DRV == S_DRV && self.device_status & VIRTIO_CONFIG_S_FEATURES_OK == 0
    }

    // ------------------------------------------------------------------
    // I/O: guest → host (TX queue)
    // ------------------------------------------------------------------

    /// Process TX descriptors: read data from guest buffers into tx_buf.
    /// TX descriptors are device-readable (guest wrote them). The device
    /// writes nothing back, so add_used len is 0.
    fn process_tx(&mut self) {
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => return,
        };
        let mut had_data = false;
        let q = &mut self.queues[TXQ];
        while let Some(chain) = q.pop_descriptor_chain(mem) {
            let head = chain.head_index();
            for desc in chain {
                if !desc.is_write_only() {
                    let guest_addr = desc.addr();
                    let dlen = (desc.len() as usize).min(TX_DESC_MAX);
                    let start = self.tx_buf.len();
                    self.tx_buf.resize(start + dlen, 0);
                    if mem
                        .read_slice(&mut self.tx_buf[start..], guest_addr)
                        .is_ok()
                    {
                        had_data = true;
                    } else {
                        self.tx_buf.truncate(start);
                    }
                }
            }
            let _ = q.add_used(mem, head, 0);
        }
        if had_data {
            tracing::debug!(
                bytes = self.tx_buf.len(),
                "virtio-console process_tx: data received"
            );
            self.signal_used();
            let _ = self.tx_evt.write(1);
        }
    }

    /// Return and clear accumulated TX output (guest → host).
    pub fn drain_output(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.tx_buf)
    }

    /// Get all accumulated TX output as a string.
    pub fn output(&self) -> String {
        String::from_utf8_lossy(&self.tx_buf).to_string()
    }

    // ------------------------------------------------------------------
    // I/O: host → guest (RX queue)
    // ------------------------------------------------------------------

    /// Push host data into guest RX buffers. Any data that cannot be
    /// delivered (RX queue exhausted) is stored in `pending_rx` and
    /// drained when the guest provides new buffers (RX queue notify).
    pub fn queue_input(&mut self, data: &[u8]) {
        tracing::debug!(bytes = data.len(), "virtio-console queue_input");
        self.pending_rx.extend(data);
        self.drain_pending_rx();
    }

    /// Drain pending_rx into guest RX buffers. Called from queue_input
    /// and on RX queue notify (guest posted new buffers).
    fn drain_pending_rx(&mut self) {
        if self.pending_rx.is_empty() {
            return;
        }
        let mem = match self.mem.as_ref() {
            Some(m) => m,
            None => {
                tracing::debug!(
                    pending = self.pending_rx.len(),
                    "virtio-console drain_pending_rx: no mem"
                );
                return;
            }
        };
        if !self.queues[RXQ].ready() {
            tracing::debug!(
                pending = self.pending_rx.len(),
                "virtio-console drain_pending_rx: RX queue not ready"
            );
            return;
        }
        let q = &mut self.queues[RXQ];
        let mut total_written = 0u32;
        while !self.pending_rx.is_empty() {
            let Some(chain) = q.pop_descriptor_chain(mem) else {
                break;
            };
            let head = chain.head_index();
            let mut written = 0u32;
            for desc in chain {
                if desc.is_write_only() && !self.pending_rx.is_empty() {
                    let guest_addr = desc.addr();
                    let avail = desc.len() as usize;
                    let chunk = self.pending_rx.len().min(avail);
                    // VecDeque may not be contiguous; drain into a
                    // contiguous slice for write_slice.
                    let bytes: Vec<u8> = self.pending_rx.drain(..chunk).collect();
                    if mem.write_slice(&bytes, guest_addr).is_ok() {
                        written += chunk as u32;
                    } else {
                        // Write failed — push bytes back to front.
                        for &b in bytes.iter().rev() {
                            self.pending_rx.push_front(b);
                        }
                    }
                }
            }
            total_written += written;
            let _ = q.add_used(mem, head, written);
        }
        if total_written > 0 {
            tracing::debug!(
                delivered = total_written,
                pending = self.pending_rx.len(),
                "virtio-console drain_pending_rx: delivered to guest",
            );
            self.signal_used();
            // If data remains in pending_rx (all descriptors consumed),
            // signal again to prompt the guest to replenish RX buffers
            // sooner. Without this, large pastes stall until the guest
            // independently reads from hvc0.
            if !self.pending_rx.is_empty() {
                let _ = self.irq_evt.write(1);
            }
        }
    }

    // ------------------------------------------------------------------
    // MMIO register dispatch
    // ------------------------------------------------------------------

    /// Handle MMIO read at `offset` within the device's MMIO region.
    pub fn mmio_read(&self, offset: u64, data: &mut [u8]) {
        if data.len() != 4 {
            for b in data.iter_mut() {
                *b = 0xff;
            }
            return;
        }
        let val: u32 = match offset as u32 {
            VIRTIO_MMIO_MAGIC_VALUE => MMIO_MAGIC,
            VIRTIO_MMIO_VERSION => MMIO_VERSION,
            VIRTIO_MMIO_DEVICE_ID => VIRTIO_ID_CONSOLE,
            VIRTIO_MMIO_VENDOR_ID => VENDOR_ID,
            VIRTIO_MMIO_DEVICE_FEATURES => {
                let page = self.device_features_sel;
                if page == 0 {
                    self.device_features() as u32
                } else if page == 1 {
                    (self.device_features() >> 32) as u32
                } else {
                    0
                }
            }
            VIRTIO_MMIO_QUEUE_NUM_MAX => self
                .selected_queue()
                .map(|i| self.queues[i].max_size() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_QUEUE_READY => self
                .selected_queue()
                .map(|i| self.queues[i].ready() as u32)
                .unwrap_or(0),
            VIRTIO_MMIO_INTERRUPT_STATUS => self.interrupt_status,
            VIRTIO_MMIO_STATUS => self.device_status,
            VIRTIO_MMIO_CONFIG_GENERATION => self.config_generation,
            _ => 0,
        };
        tracing::debug!(offset, val, "virtio-console mmio_read");
        data.copy_from_slice(&val.to_le_bytes());
    }

    /// Handle MMIO write at `offset` within the device's MMIO region.
    pub fn mmio_write(&mut self, offset: u64, data: &[u8]) {
        if data.len() != 4 {
            return;
        }
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        tracing::debug!(offset, val, "virtio-console mmio_write");
        match offset as u32 {
            VIRTIO_MMIO_DEVICE_FEATURES_SEL => self.device_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES_SEL => self.driver_features_sel = val,
            VIRTIO_MMIO_DRIVER_FEATURES => {
                if !self.features_write_allowed() {
                    return;
                }
                let page = self.driver_features_sel;
                if page == 0 {
                    self.driver_features =
                        (self.driver_features & 0xFFFF_FFFF_0000_0000) | val as u64;
                } else if page == 1 {
                    self.driver_features =
                        (self.driver_features & 0x0000_0000_FFFF_FFFF) | ((val as u64) << 32);
                }
            }
            VIRTIO_MMIO_QUEUE_SEL => self.queue_select = val,
            VIRTIO_MMIO_QUEUE_NUM
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_size(val as u16);
            }
            VIRTIO_MMIO_QUEUE_READY
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_ready(val == 1);
            }
            VIRTIO_MMIO_QUEUE_NOTIFY => {
                let idx = val as usize;
                if idx == TXQ {
                    self.process_tx();
                } else if idx == RXQ {
                    self.drain_pending_rx();
                }
            }
            VIRTIO_MMIO_INTERRUPT_ACK => {
                self.interrupt_status &= !val;
            }
            VIRTIO_MMIO_STATUS => {
                if val == 0 {
                    self.reset();
                } else {
                    self.set_status(val);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_LOW
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_desc_table_address(Some(val), None);
            }
            VIRTIO_MMIO_QUEUE_DESC_HIGH
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_desc_table_address(None, Some(val));
            }
            VIRTIO_MMIO_QUEUE_AVAIL_LOW
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_avail_ring_address(Some(val), None);
            }
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_avail_ring_address(None, Some(val));
            }
            VIRTIO_MMIO_QUEUE_USED_LOW
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_used_ring_address(Some(val), None);
            }
            VIRTIO_MMIO_QUEUE_USED_HIGH
                if self.queue_config_allowed()
                    && let Some(i) = self.selected_queue() =>
            {
                self.queues[i].set_used_ring_address(None, Some(val));
            }
            _ => {}
        }
    }

    /// Validate and apply a status transition per virtio-v1.2 §3.1.1.
    /// The driver must not clear bits. Each phase requires the previous
    /// phase's bits to be set. Invalid transitions are ignored.
    fn set_status(&mut self, val: u32) {
        let old = self.device_status;
        // Driver must not clear bits (except via reset, which writes 0).
        if val & self.device_status != self.device_status {
            tracing::debug!(
                old,
                val,
                "virtio-console set_status: rejected (clears bits)"
            );
            return;
        }
        let new_bits = val & !self.device_status;
        let valid = match new_bits {
            VIRTIO_CONFIG_S_ACKNOWLEDGE => self.device_status == 0,
            VIRTIO_CONFIG_S_DRIVER => self.device_status == S_ACK,
            VIRTIO_CONFIG_S_FEATURES_OK => self.device_status == S_DRV,
            VIRTIO_CONFIG_S_DRIVER_OK => self.device_status == S_FEAT,
            _ => false,
        };
        if valid {
            self.device_status = val;
            tracing::debug!(old, new = val, "virtio-console set_status: accepted");
        } else {
            tracing::debug!(
                old,
                val,
                "virtio-console set_status: rejected (invalid transition)"
            );
        }
    }

    fn reset(&mut self) {
        self.device_status = 0;
        self.interrupt_status = 0;
        self.queue_select = 0;
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        self.tx_buf.clear();
        self.pending_rx.clear();
        for q in &mut self.queues {
            q.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::io::AsRawFd;
    use virtio_bindings::virtio_mmio::VIRTIO_MMIO_INT_CONFIG;
    use vm_memory::GuestAddress;

    fn read_reg(dev: &VirtioConsole, offset: u32) -> u32 {
        let mut buf = [0u8; 4];
        dev.mmio_read(offset as u64, &mut buf);
        u32::from_le_bytes(buf)
    }

    fn write_reg(dev: &mut VirtioConsole, offset: u32, val: u32) {
        dev.mmio_write(offset as u64, &val.to_le_bytes());
    }

    /// Drive the device through the full init sequence up to DRIVER_OK.
    fn init_device(dev: &mut VirtioConsole) {
        write_reg(dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(dev, VIRTIO_MMIO_STATUS, S_DRV);
        // Negotiate VIRTIO_F_VERSION_1.
        write_reg(dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(dev, VIRTIO_MMIO_STATUS, S_FEAT);
        write_reg(dev, VIRTIO_MMIO_STATUS, S_OK);
    }

    #[test]
    fn magic_version_device_id() {
        let dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_MAGIC_VALUE), 0x7472_6976);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_VERSION), 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_ID), VIRTIO_ID_CONSOLE);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_VENDOR_ID), 0);
    }

    #[test]
    fn device_features_version_1() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
        let lo = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
        let hi = read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES);
        let features = (hi as u64) << 32 | lo as u64;
        assert_ne!(features & (1 << VIRTIO_F_VERSION_1), 0);
    }

    #[test]
    fn queue_num_max() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
            QUEUE_MAX_SIZE as u32
        );
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 1);
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX),
            QUEUE_MAX_SIZE as u32
        );
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
    }

    #[test]
    fn queue_ready_requires_features_ok() {
        let mut dev = VirtioConsole::new();
        // Before FEATURES_OK, queue config is rejected.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);

        // After FEATURES_OK, queue config is accepted.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_READY, 1);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 1);
    }

    #[test]
    fn status_state_machine() {
        let mut dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);

        // Valid sequence.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(dev.device_status, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        assert_eq!(dev.device_status, S_DRV);

        // Skipping FEATURES_OK to DRIVER_OK is rejected.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_OK);
        assert_eq!(
            dev.device_status, S_DRV,
            "skip FEATURES_OK must be rejected"
        );

        // Clearing bits is rejected (non-zero).
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        assert_eq!(
            dev.device_status, S_DRV,
            "clearing DRIVER bit must be rejected"
        );
    }

    #[test]
    fn status_reset_via_zero() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        assert_eq!(dev.device_status, S_OK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);
        assert_eq!(dev.device_status, 0);
    }

    #[test]
    fn interrupt_status_and_ack() {
        let mut dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
        dev.interrupt_status = VIRTIO_MMIO_INT_VRING;
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
            VIRTIO_MMIO_INT_VRING
        );
    }

    #[test]
    fn interrupt_ack_clears_bits() {
        let mut dev = VirtioConsole::new();
        dev.interrupt_status = VIRTIO_MMIO_INT_VRING | VIRTIO_MMIO_INT_CONFIG;
        write_reg(&mut dev, VIRTIO_MMIO_INTERRUPT_ACK, VIRTIO_MMIO_INT_VRING);
        assert_eq!(
            read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS),
            VIRTIO_MMIO_INT_CONFIG
        );
    }

    #[test]
    fn non_4byte_read_returns_ff() {
        let dev = VirtioConsole::new();
        let mut buf = [0u8; 2];
        dev.mmio_read(0, &mut buf);
        assert_eq!(buf, [0xff, 0xff]);
    }

    #[test]
    fn non_4byte_write_ignored() {
        let mut dev = VirtioConsole::new();
        dev.mmio_write(VIRTIO_MMIO_STATUS as u64, &[0x01, 0x00]);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    }

    #[test]
    fn driver_features_gated_by_status() {
        let mut dev = VirtioConsole::new();
        // Before DRIVER status, features writes are rejected.
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD);
        assert_eq!(dev.driver_features, 0);

        // After ACKNOWLEDGE + DRIVER, features writes are accepted.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xDEAD_BEEF);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xCAFE_BABE);
        assert_eq!(dev.driver_features, 0xCAFE_BABE_DEAD_BEEF);
    }

    #[test]
    fn features_rejected_after_features_ok() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        // After FEATURES_OK, features writes are rejected.
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES, 0xFFFF);
        assert_eq!(dev.driver_features & 0xFFFF_FFFF, 0);
    }

    #[test]
    fn queue_desc_addr_requires_features_ok() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
        // Not accepted before FEATURES_OK.
        assert_ne!(dev.queues[0].desc_table(), 0x1000);

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_ACK);
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_DRV);
        write_reg(&mut dev, VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        write_reg(
            &mut dev,
            VIRTIO_MMIO_DRIVER_FEATURES,
            1 << (VIRTIO_F_VERSION_1 - 32),
        );
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, S_FEAT);

        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_LOW, 0x1000);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_DESC_HIGH, 0);
        assert_eq!(dev.queues[0].desc_table(), 0x1000);
    }

    #[test]
    fn reset_clears_all_state() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        dev.interrupt_status = 0xFF;
        dev.tx_buf.extend_from_slice(b"leftover");

        write_reg(&mut dev, VIRTIO_MMIO_STATUS, 0);

        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_INTERRUPT_STATUS), 0);
        assert_eq!(dev.queue_select, 0);
        assert_eq!(dev.device_features_sel, 0);
        assert_eq!(dev.driver_features, 0);
        assert!(dev.tx_buf.is_empty(), "tx_buf must be cleared on reset");
    }

    #[test]
    fn config_generation_initially_zero() {
        let dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_CONFIG_GENERATION), 0);
    }

    #[test]
    fn new_creates_eventfds() {
        let dev = VirtioConsole::new();
        assert!(dev.irq_evt().as_raw_fd() >= 0);
        assert!(dev.tx_evt().as_raw_fd() >= 0);
        assert_ne!(dev.irq_evt().as_raw_fd(), dev.tx_evt().as_raw_fd());
    }

    #[test]
    fn output_empty_initially() {
        let dev = VirtioConsole::new();
        assert!(dev.output().is_empty());
    }

    #[test]
    fn drain_output_empty() {
        let mut dev = VirtioConsole::new();
        assert!(dev.drain_output().is_empty());
    }

    #[test]
    fn set_mem_stores_reference() {
        let mut dev = VirtioConsole::new();
        assert!(dev.mem.is_none());
        let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        dev.set_mem(mem);
        assert!(dev.mem.is_some());
    }

    #[test]
    fn queue_input_no_mem_no_panic() {
        let mut dev = VirtioConsole::new();
        dev.queue_input(b"hello");
    }

    #[test]
    fn unknown_register_returns_zero() {
        let dev = VirtioConsole::new();
        assert_eq!(read_reg(&dev, 0x300), 0);
    }

    #[test]
    fn unknown_register_write_ignored() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, 0x300, 0xDEAD);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_STATUS), 0);
    }

    #[test]
    fn invalid_queue_select_returns_zero() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 99);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_QUEUE_READY), 0);
    }

    #[test]
    fn signal_used_sets_interrupt_and_writes_eventfd() {
        let mut dev = VirtioConsole::new();
        assert_eq!(dev.interrupt_status, 0);
        dev.signal_used();
        assert_ne!(dev.interrupt_status & VIRTIO_MMIO_INT_VRING, 0);
        let val = dev.irq_evt.read().unwrap();
        assert!(val > 0);
    }

    #[test]
    fn features_page_2_returns_zero() {
        let mut dev = VirtioConsole::new();
        write_reg(&mut dev, VIRTIO_MMIO_DEVICE_FEATURES_SEL, 2);
        assert_eq!(read_reg(&dev, VIRTIO_MMIO_DEVICE_FEATURES), 0);
    }

    #[test]
    fn tx_evt_signaled_on_process_tx_with_no_data() {
        let mut dev = VirtioConsole::new();
        let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        dev.set_mem(mem);
        // process_tx with no descriptors should not signal tx_evt.
        dev.process_tx();
        // tx_evt should not be readable (no data processed).
        assert!(dev.tx_evt.read().is_err());
    }

    #[test]
    fn status_skip_acknowledge_rejected() {
        let mut dev = VirtioConsole::new();
        // Skipping ACKNOWLEDGE, going straight to DRIVER.
        write_reg(&mut dev, VIRTIO_MMIO_STATUS, VIRTIO_CONFIG_S_DRIVER);
        assert_eq!(dev.device_status, 0);
    }

    #[test]
    fn queue_config_rejected_after_driver_ok() {
        let mut dev = VirtioConsole::new();
        init_device(&mut dev);
        assert_eq!(dev.device_status, S_OK);

        // After DRIVER_OK, queue config is rejected.
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_SEL, 0);
        write_reg(&mut dev, VIRTIO_MMIO_QUEUE_NUM, 64);
        // Queue size should still be the default (QUEUE_MAX_SIZE), not 64.
        assert_eq!(dev.queues[0].size(), QUEUE_MAX_SIZE);
    }
}
