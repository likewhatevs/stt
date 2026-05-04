//! Virtio-net device with in-VMM loopback backend.
//!
//! Two virtqueues (RX index 0, TX index 1), no multiqueue, no control
//! virtqueue. Advertised features: `VIRTIO_F_VERSION_1` (mandatory)
//! plus `VIRTIO_NET_F_MAC` (so the guest binds a deterministic MAC
//! rather than a random one). MMIO register layout per virtio-v1.2
//! §4.2.2; net-specific config space at offsets `0x100..` is served
//! from a [`VirtioNetConfig`] struct whose `repr(C, packed)` layout
//! mirrors the kernel uapi `struct virtio_net_config` byte-for-byte
//! (virtio-v1.2 §5.1.4). Interrupt delivery via irqfd
//! (eventfd → KVM GSI).
//!
//! # Header semantics
//!
//! `VIRTIO_F_VERSION_1` forces the guest driver into the
//! `virtio_net_hdr_mrg_rxbuf` size (12 bytes) regardless of whether
//! `VIRTIO_NET_F_MRG_RXBUF` is negotiated. Kernel reference:
//! `drivers/net/virtio_net.c::virtnet_probe` ("else if VERSION_1 …
//! mrg_rxbuf"). This device emits a 12-byte header on every RX
//! delivery — bytes 0..10 zero (no GSO/csum), bytes 10..12 =
//! `num_buffers = 1` LE u16 — and consumes 12 header bytes from
//! every TX chain. A 10-byte header would silently corrupt guest
//! memory because the kernel's frame-delivery path advances past
//! `vi->hdr_len` (=12) before handing the frame to the network
//! stack.
//!
//! On the `num_buffers` byte: with our negotiated feature set
//! (`VIRTIO_F_VERSION_1` + `VIRTIO_NET_F_MAC` only, NOT
//! `VIRTIO_NET_F_MRG_RXBUF`), the kernel's `receive_buf`
//! dispatcher takes the `receive_small` path
//! (`drivers/net/virtio_net.c::receive_small`), which subtracts
//! `vi->hdr_len` from `len` and never reads the `num_buffers`
//! byte at all — `receive_small` builds an skb directly from
//! the `len - hdr_len` payload bytes after the header. The
//! `receive_mergeable` path that DOES consult `num_buffers`
//! only runs when `VIRTIO_NET_F_MRG_RXBUF` is negotiated
//! (`receive_buf`'s `if (vi->mergeable_rx_bufs)` branch). We
//! pin `num_buffers = 1` as forward-compatibility hardening
//! for a future MRG_RXBUF advertisement: when that bit is
//! eventually negotiated, the device's RX path will already be
//! emitting the correct head-of-chain marker without further
//! changes. A `num_buffers = 0` header in the
//! `receive_mergeable` path would make the kernel treat the
//! frame as the head of an unfinished multi-buffer chain and
//! either wait for a non-existent continuation buffer or hit
//! the shouldn't-happen branch.
//!
//! # In-VMM loopback (v0 backend)
//!
//! `process_tx_loopback` reads a TX chain, strips the 12-byte
//! header, captures the L2 frame into a per-device scratch buffer,
//! marks the TX chain used (no bytes written back; TX descriptors
//! are device-readable), then synthesizes an RX delivery: locks the
//! RX queue, pops one chain, writes the 12-byte virtio header
//! (num_buffers=1) followed by the frame data, marks the RX chain
//! used, and signals the irqfd. The guest's TX kick therefore
//! produces a guest TX completion AND a guest RX interrupt in the
//! same vCPU exit.
//!
//! No MAC swap, no ARP synthesis, no IP routing — the loopback is a
//! raw byte echo at the L2 layer. AF_PACKET sockets bound by
//! `ifindex` see their own TX echoed back as RX (with destination
//! MAC unchanged); IP-layer self-traffic to the device's address is
//! intercepted by `RTN_LOCAL` routing in the guest kernel and
//! routed onto `lo`, never reaching virtio-net. The loopback's
//! purpose is to generate real `vring_interrupt` →
//! `NET_RX_SOFTIRQ` activity that scheduler-test scenarios can
//! observe; it is not a host-side network bridge.
//!
//! # No worker thread (v0)
//!
//! Unlike virtio-blk, this device runs the loopback path inline on
//! the vCPU thread inside `mmio_write(QUEUE_NOTIFY)`. The work is
//! a guest-memory read and write — no host syscalls, no backing
//! file, no blocking. The round-trip latency is bounded by the
//! frame size (≤64 KiB per chain by the per-descriptor cap) and
//! the irqfd write. Bounded vCPU thread work below the
//! freeze-rendezvous timeout means no worker is needed; future
//! upgrade to a TAP/AF_PACKET backend would migrate the loopback
//! to a worker thread without changing the device state machine.

mod device;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_poison;

#[cfg(test)]
mod tests_proptest;

// Glob `pub(crate) use device::*` makes test sub-files (`tests.rs`)
// reach internal items (`S_ACK`, `S_OK`, `RXQ`, `TXQ`,
// `VIRTIO_NET_CONFIG_SIZE`, etc.) via `super::device::*` without
// per-name re-export bookkeeping. Items intended for non-test
// crate code reach through this glob too. The `pub use` block
// below itemizes the symbols that need full `pub` visibility for
// upstream re-exports (mod.rs and lib.rs publish `VirtioNet`,
// `VirtioNetCounters`, and the MMIO size constant); these are
// already `pub` inside `device.rs`, and the explicit listing
// upgrades the re-export from the glob's `pub(crate)` to `pub`
// for those names only.
#[allow(unused_imports)]
pub(crate) use device::*;
pub use device::{VIRTIO_MMIO_SIZE, VirtioNet, VirtioNetCounters};
