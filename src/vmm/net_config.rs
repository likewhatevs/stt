//! Network configuration for the virtio-net device.
//!
//! [`NetConfig`] is the descriptor — passed by value, copious defaults,
//! no fd field (the framework owns the in-VMM loopback backend's
//! lifecycle). Mirrors [`super::disk_config::DiskConfig`] for API
//! consistency: chainable setters return `Self`, `Default::default()`
//! produces a working device, and the type lives in the public prelude
//! so test authors can spell it out.
//!
//! v0 backend is in-VMM loopback: TX descriptor bytes are written
//! directly into RX descriptors and the irqfd fires. There is no host
//! networking, no `/dev/net/tun`, no AF_PACKET fd. The guest sees a
//! single virtio-net interface that loops its own TX back to its RX
//! verbatim — no MAC swap, no ARP synthesis, no IP routing. The byte
//! flow lives in [`super::virtio_net`] (see `process_tx_loopback` in
//! the device module). AF_PACKET raw sockets bound to the interface
//! generate real virtio TX kicks and observe real virtio RX
//! interrupts — IP-layer self-traffic is forced onto `lo` by the
//! kernel routing layer (`net/ipv4/route.c::ip_route_output_key_hash_rcu`
//! resolves any local destination as `RTN_LOCAL` →
//! `dev_out = net->loopback_dev`) and never reaches virtio-net.

/// Configuration for the virtio-net device attached to the VM.
///
/// `Default::default()` produces a working device with a deterministic
/// locally-administered MAC. Override the MAC with [`Self::mac`] to
/// pin a value across runs (useful for log correlation against
/// AF_PACKET captures).
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct NetConfig {
    /// MAC address advertised to the guest via `VIRTIO_NET_F_MAC`.
    /// The locally-administered bit (0x02) is set in the default to
    /// avoid collisions with real-hardware OUIs; operators that
    /// override the MAC are responsible for the bit themselves.
    pub mac: [u8; 6],
}

impl Default for NetConfig {
    /// Default MAC: `02:00:00:00:00:01`. The leading `0x02` sets the
    /// locally-administered bit per IEEE 802 (bit 1 of the first
    /// octet), keeping the address out of the IEEE OUI namespace.
    /// The trailing `0x01` distinguishes the default from the
    /// guest-host pair convention (host-side might use `0x02`); v0
    /// runs a single device, so the value is informational.
    fn default() -> Self {
        NetConfig {
            mac: [0x02, 0x00, 0x00, 0x00, 0x00, 0x01],
        }
    }
}

impl NetConfig {
    /// Override the advertised MAC. Returns `self` for chained
    /// configuration matching [`super::disk_config::DiskConfig`]'s
    /// builder style.
    pub fn mac(mut self, mac: [u8; 6]) -> Self {
        self.mac = mac;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_locally_administered_bit() {
        let cfg = NetConfig::default();
        assert_eq!(
            cfg.mac[0] & 0x02,
            0x02,
            "default MAC must have locally-administered bit (IEEE 802 first-octet bit 1)",
        );
    }

    #[test]
    fn default_is_unicast() {
        let cfg = NetConfig::default();
        assert_eq!(
            cfg.mac[0] & 0x01,
            0x00,
            "default MAC must not be multicast (IEEE 802 first-octet bit 0)",
        );
    }

    #[test]
    fn mac_setter_overrides_default() {
        let cfg = NetConfig::default().mac([0x52, 0x54, 0x00, 0xab, 0xcd, 0xef]);
        assert_eq!(cfg.mac, [0x52, 0x54, 0x00, 0xab, 0xcd, 0xef]);
    }

    #[test]
    fn serde_roundtrip_pins_field_names() {
        let cfg = NetConfig::default().mac([1, 2, 3, 4, 5, 6]);
        let json = serde_json::to_string(&cfg).expect("serialize");
        // Pin the field name so a future rename surfaces here.
        assert!(json.contains("\"mac\""), "missing key `mac`: {json}");
        let parsed: NetConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, cfg);
    }
}
