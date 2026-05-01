//! Disk configuration for virtio-blk devices.
//!
//! v0 creates a raw sparse backing file per test via `tempfile()` —
//! the guest sees an unformatted block device at `/dev/vda`, and no
//! mount happens. Filesystem formatting via a template VM and guest
//! auto-mount at `/mnt/disk{N}` are deferred follow-ups; the
//! [`Filesystem`] enum reserves API headroom for that work.
//!
//! `DiskConfig` is the descriptor — passed by value, copious
//! defaults, no path field (the framework owns the per-test backing
//! file's lifecycle).

use std::num::NonZeroU64;

/// Filesystem to format the backing file with.
///
/// Reserved for future variants; v0 is always Raw. `Raw` matches the
/// actual on-disk state: no formatting happens, the guest sees
/// `/dev/vda` as a raw unformatted block device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filesystem {
    /// No filesystem; raw block device. v0 always behaves this way.
    #[default]
    Raw,
}

/// IO throttle for one disk. Each field caps a separate dimension;
/// `None` disables that dimension's throttle. Both `None` =
/// unthrottled (the device runs at host-pread/pwrite speed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiskThrottle {
    /// Maximum operations per second (1 read = 1 op, 1 write = 1
    /// op, 1 flush = 1 op).
    ///
    /// Type-enforced nonzero: `Option<NonZeroU64>` makes
    /// `Some(0) = unlimited` impossible to express at the type
    /// level. To disable IOPS throttling, use `None` (or set 0
    /// through the builder, which the builder converts to `None`).
    pub iops: Option<NonZeroU64>,
    /// Maximum bytes per second across read+write data.
    ///
    /// Type-enforced nonzero, same reasoning as `iops`.
    pub bytes_per_sec: Option<NonZeroU64>,
}

/// Per-disk config. `Default` is raw 256 MB device on `/dev/vda`;
/// formatting and auto-mount are deferred.
///
/// No backing-file path field: the framework owns the per-test
/// backing file (`tempfile()` today). See module docs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiskConfig {
    /// Advertised capacity in megabytes. 256 MB default capacity.
    /// Sized to accommodate common guest filesystem formatters;
    /// smaller values are accepted but may cause mkfs failures
    /// when the template-VM lifecycle lands.
    pub capacity_mb: u32,
    /// Filesystem. Reserved for future variants; v0 is always
    /// [`Filesystem::Raw`].
    pub filesystem: Filesystem,
    /// IO throttle. Default unthrottled.
    pub throttle: DiskThrottle,
    /// Read-only at the device level — the device advertises
    /// VIRTIO_BLK_F_RO so the guest mounts read-only. Useful for
    /// tests that need protection against accidental writes.
    pub read_only: bool,
    /// Optional human-readable label for this disk. `None` (the
    /// default) is an anonymous disk addressable only by index. A
    /// name lets WorkType variants reference the disk symbolically
    /// (e.g. `"data"`, `"log"`) instead of by index, which keeps
    /// tests stable across topology rearrangements.
    pub name: Option<String>,
}

impl Default for DiskConfig {
    /// 256 MB, [`Filesystem::Raw`], no throttle. v0 ignores the
    /// `filesystem` field — every disk arrives raw regardless.
    ///
    /// # Memory footprint
    ///
    /// The 256 MB sparse file lives under the host's `TMPDIR`
    /// (`tempfile()`); actual host disk/RAM consumption equals the
    /// bytes the guest writes, not the advertised capacity. On
    /// tmpfs-backed `TMPDIR` (the default on most Linux distros), a
    /// fully-written disk consumes 256 MB of host **RAM** per test
    /// — operators running large topologies should size host memory
    /// accordingly or override `TMPDIR` to a disk-backed path.
    fn default() -> Self {
        DiskConfig {
            capacity_mb: 256,
            filesystem: Filesystem::Raw,
            throttle: DiskThrottle::default(),
            read_only: false,
            name: None,
        }
    }
}

impl DiskConfig {
    /// Set capacity in megabytes.
    pub fn capacity_mb(mut self, mb: u32) -> Self {
        self.capacity_mb = mb;
        self
    }

    /// Set IOPS throttle. Passing 0 disables IOPS throttling
    /// (equivalent to `None`). To throttle near-zero, use `iops(1)`.
    /// There is no "block all IO" mode — the minimum throttled rate
    /// is 1 op/sec. Any positive value is wrapped in `NonZeroU64`.
    pub fn iops(mut self, iops: u64) -> Self {
        self.throttle.iops = NonZeroU64::new(iops);
        self
    }

    /// Set bandwidth throttle (bytes per second). A zero value
    /// disables bandwidth throttling (stored as `None`); any
    /// positive value is wrapped in `NonZeroU64`.
    pub fn bytes_per_sec(mut self, bytes_per_sec: u64) -> Self {
        self.throttle.bytes_per_sec = NonZeroU64::new(bytes_per_sec);
        self
    }

    /// Mark the disk read-only (advertises `VIRTIO_BLK_F_RO`).
    /// Default is read-write; this builder takes no argument (no
    /// boolean footgun) and only flips the flag on. To return to
    /// read-write, drop the call or reconstruct from
    /// `DiskConfig::default()`.
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Attach a human-readable label to this disk. WorkType variants
    /// that need to address a specific disk (e.g. one of several
    /// attached) can resolve the name instead of relying on
    /// attachment order. Default is anonymous (`None`); calling
    /// `.name(...)` sets it.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Capacity in bytes (`capacity_mb << 20`). Used by the device
    /// for the config-space `capacity` field.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        (self.capacity_mb as u64) << 20
    }

    /// Capacity in 512-byte sectors.
    pub(crate) fn capacity_sectors(&self) -> u64 {
        self.capacity_bytes() / 512
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_256mb_raw() {
        let d = DiskConfig::default();
        assert_eq!(d.capacity_mb, 256);
        assert_eq!(d.filesystem, Filesystem::Raw);
        assert_eq!(d.throttle, DiskThrottle::default());
        assert!(!d.read_only);
        assert!(d.name.is_none());
    }

    #[test]
    fn capacity_helpers() {
        let d = DiskConfig::default();
        assert_eq!(d.capacity_bytes(), 256 * 1024 * 1024);
        assert_eq!(d.capacity_sectors(), 524_288);

        let d = DiskConfig::default().capacity_mb(512);
        assert_eq!(d.capacity_bytes(), 512 * 1024 * 1024);
        assert_eq!(d.capacity_sectors(), 1_048_576);
    }

    #[test]
    fn builder_chain() {
        let d = DiskConfig::default()
            .capacity_mb(128)
            .iops(1000)
            .bytes_per_sec(10 * 1024 * 1024)
            .read_only();
        assert_eq!(d.capacity_mb, 128);
        assert_eq!(d.filesystem, Filesystem::Raw);
        assert_eq!(d.throttle.iops, NonZeroU64::new(1000));
        assert_eq!(d.throttle.bytes_per_sec, NonZeroU64::new(10 * 1024 * 1024));
        assert!(d.read_only);
    }

    #[test]
    fn iops_zero_becomes_none() {
        // The NonZeroU64 type makes Some(0) impossible. The builder
        // accepts u64 for ergonomics and converts 0 → None
        // (= unthrottled) at the type boundary.
        let d = DiskConfig::default().iops(0);
        assert!(d.throttle.iops.is_none());
        let d = DiskConfig::default().bytes_per_sec(0);
        assert!(d.throttle.bytes_per_sec.is_none());
    }

    #[test]
    fn filesystem_default_is_raw() {
        // Default::default() must produce a working v0 config — the
        // `Filesystem::Raw` default matches the actual v0 behaviour
        // (no formatting). #[default] attribute on the enum variant
        // drives this; this test pins it so a future patch that
        // adds a non-Raw variant and changes `#[default]` (regressing
        // the "default works" guarantee) surfaces here.
        assert_eq!(Filesystem::default(), Filesystem::Raw);
    }

    #[test]
    fn filesystem_serde_snake_case() {
        assert_eq!(serde_json::to_string(&Filesystem::Raw).unwrap(), r#""raw""#);
    }

    #[test]
    fn throttle_default_is_unthrottled() {
        let t = DiskThrottle::default();
        assert!(t.iops.is_none());
        assert!(t.bytes_per_sec.is_none());
    }

    #[test]
    fn iops_zero_serde_roundtrip() {
        // Build with iops(0) → throttle.iops is None. Serialize +
        // deserialize the config and confirm the field stays None.
        // Pins the NonZeroU64 type-level invariant against a future
        // serde-derive regression that might silently re-introduce
        // a Some(0) representation (impossible by construction
        // today, but a wrong-typed `Option<u64>` migration would
        // bring it back).
        let original = DiskConfig::default().iops(0).bytes_per_sec(0);
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: DiskConfig = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.throttle.iops.is_none());
        assert!(parsed.throttle.bytes_per_sec.is_none());
        // Round-trip equality works because of the PartialEq derive
        // on DiskConfig.
        assert_eq!(parsed, original);
    }

    /// Full serde roundtrip with every field set to a non-default
    /// value. Pin field-by-field equality after a JSON round trip so
    /// a future `#[serde(rename = ...)]` or `#[serde(skip)]`
    /// regression — the typical drift mode for serde-derived structs
    /// — surfaces here loudly.
    #[test]
    fn disk_config_full_serde_roundtrip() {
        let original = DiskConfig {
            capacity_mb: 256,
            filesystem: Filesystem::Raw,
            throttle: DiskThrottle {
                iops: NonZeroU64::new(2_500),
                bytes_per_sec: NonZeroU64::new(50 * 1024 * 1024),
            },
            read_only: true,
            name: Some("data-disk".to_string()),
        };

        let json = serde_json::to_string(&original).expect("serialize DiskConfig");
        let parsed: DiskConfig =
            serde_json::from_str(&json).expect("deserialize DiskConfig");

        // Whole-struct equality first — catches any field drift.
        assert_eq!(parsed, original);
        // Field-by-field follow-up — each line catches a distinct
        // drift mode on its own (rename, skip, type-narrowing).
        assert_eq!(parsed.capacity_mb, original.capacity_mb);
        assert_eq!(parsed.filesystem, original.filesystem);
        assert_eq!(parsed.throttle.iops, original.throttle.iops);
        assert_eq!(
            parsed.throttle.bytes_per_sec,
            original.throttle.bytes_per_sec
        );
        assert_eq!(parsed.read_only, original.read_only);
        assert_eq!(parsed.name, original.name);
        assert_eq!(parsed.name.as_deref(), Some("data-disk"));
    }

    /// Roundtrip the unthrottled default (both throttle fields
    /// `None`). Distinct from `iops_zero_serde_roundtrip` (which
    /// builds via `.iops(0)/.bytes_per_sec(0)`): this exercises the
    /// pure `DiskConfig::default()` shape, ensuring the `None`/`None`
    /// throttle survives serialize→JSON→deserialize and that the
    /// whole-struct PartialEq holds across the round trip.
    #[test]
    fn disk_config_default_unthrottled_serde_roundtrip() {
        let original = DiskConfig::default();
        assert!(original.throttle.iops.is_none());
        assert!(original.throttle.bytes_per_sec.is_none());
        assert!(original.name.is_none());

        let json = serde_json::to_string(&original).expect("serialize default DiskConfig");
        let parsed: DiskConfig =
            serde_json::from_str(&json).expect("deserialize default DiskConfig");

        assert_eq!(parsed, original);
        assert_eq!(parsed.capacity_mb, original.capacity_mb);
        assert_eq!(parsed.filesystem, original.filesystem);
        assert!(parsed.throttle.iops.is_none());
        assert!(parsed.throttle.bytes_per_sec.is_none());
        assert_eq!(parsed.read_only, original.read_only);
        assert!(parsed.name.is_none());
    }

    #[test]
    fn name_builder_sets_label() {
        let d = DiskConfig::default().name("data-disk");
        assert_eq!(d.name.as_deref(), Some("data-disk"));

        // Accepts both &str (Into<String>) and String — pin the
        // generic-bound coverage so a future tightening to &str-only
        // surfaces here.
        let d = DiskConfig::default().name(String::from("log-disk"));
        assert_eq!(d.name.as_deref(), Some("log-disk"));

        // Last call wins — the builder overwrites.
        let d = DiskConfig::default().name("first").name("second");
        assert_eq!(d.name.as_deref(), Some("second"));
    }
}
