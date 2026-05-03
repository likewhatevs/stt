//! Disk configuration for virtio-blk devices.
//!
//! [`Filesystem::Raw`] gives the guest an unformatted block device at
//! `/dev/vda` (a fresh sparse `tempfile()` backing per test). No mount
//! happens.
//!
//! [`Filesystem::Btrfs`] is the entry point for the disk-template
//! lifecycle. Selecting it routes through
//! [`crate::vmm::disk_template::ensure_template`]: on cache miss
//! the framework boots a one-shot template VM that runs
//! `mkfs.btrfs` against `/dev/vda`, caches the formatted image
//! under the ktstr cache root, and per-test boots reflink-copy
//! that template via `FICLONE` so each per-test filesystem starts
//! pre-formatted with zero host-side mkfs cost. The host never
//! execs mkfs against a real backing file — the kernel's own mkfs
//! (run inside the template VM) is the on-disk-format authority.
//! See [`crate::vmm::disk_template`] for the full cache and
//! template-VM driver implementation.
//!
//! `DiskConfig` is the descriptor — passed by value, copious
//! defaults, no path field (the framework owns the per-test backing
//! file's lifecycle).

use std::num::NonZeroU64;

/// Filesystem to format the backing file with.
///
/// `Raw` matches the actual on-disk state: no formatting happens, the
/// guest sees `/dev/vda` as a raw unformatted block device.
///
/// `Btrfs` activates the template-cache lifecycle (see module docs).
/// Selecting it requires the ktstr cache directory to live on a
/// reflink-capable filesystem (btrfs or xfs) — the per-test fan-out
/// uses `FICLONE` to clone the cached template image and would fail
/// on tmpfs/ext4. The host must also have `mkfs.btrfs` on `PATH` at
/// template-build time so the template-VM initramfs can pack it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filesystem {
    /// No filesystem; raw block device. The guest sees `/dev/vda` as
    /// an unformatted volume of the configured capacity. Default.
    #[default]
    Raw,
    /// btrfs filesystem. Once the template-VM driver lands, per-test
    /// backing will be a reflink clone of a host-cached,
    /// guest-formatted btrfs image at the configured capacity; the
    /// cache lives under the ktstr cache root and requires a
    /// btrfs/xfs mount, and `mkfs.btrfs` must be on the host `PATH`
    /// at template-build time. v0: selecting `Btrfs` returns an
    /// actionable error from `init_virtio_blk` until the driver
    /// referenced by
    /// [`crate::vmm::disk_template::build_template_via_vm`] lands.
    /// See [`crate::vmm::disk_template`].
    Btrfs,
}

impl Filesystem {
    /// Short identifier used in cache keys and diagnostics. The
    /// values are intentionally short (≤8 chars), kebab-free, and
    /// stable across rebuilds — they participate in on-disk cache
    /// path names, so renaming a variant invalidates already-cached
    /// templates. New variants must add a new tag rather than
    /// reusing one.
    pub(crate) fn cache_tag(self) -> &'static str {
        match self {
            Filesystem::Raw => "raw",
            Filesystem::Btrfs => "btrfs",
        }
    }
}

/// IO throttle for one disk. Each field caps a separate dimension;
/// `None` disables that dimension's throttle. All `None` =
/// unthrottled (the device runs at host-pread/pwrite speed).
///
/// Burst capacity is the token-bucket capacity (peak instantaneous
/// burst the device will absorb before throttling kicks in). Refill
/// rate is the steady-state allowance (`iops` / `bytes_per_sec`).
/// When `*_burst_capacity` is `None`, the bucket capacity equals the
/// refill rate, giving a 1-second burst — the historical default.
/// Setting a burst capacity larger than the refill rate models a
/// device that tolerates transient spikes (e.g. a 1-second steady
/// rate of 1000 IOPS with a 5000-IOPS burst capacity allows a
/// 5-second-equivalent burst from a full bucket). A burst capacity
/// without a corresponding rate is meaningless (a bucket that never
/// refills); [`DiskThrottle::validate`] rejects it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiskThrottle {
    /// Maximum operations per second (1 read = 1 op, 1 write = 1
    /// op, 1 flush = 1 op). Refill rate of the IOPS token bucket.
    ///
    /// Type-enforced nonzero: `Option<NonZeroU64>` makes
    /// `Some(0) = unlimited` impossible to express at the type
    /// level. To disable IOPS throttling, use `None` (or set 0
    /// through the builder, which the builder converts to `None`).
    pub iops: Option<NonZeroU64>,
    /// Maximum bytes per second across read+write data. Refill rate
    /// of the bandwidth token bucket.
    ///
    /// Type-enforced nonzero, same reasoning as `iops`.
    pub bytes_per_sec: Option<NonZeroU64>,
    /// IOPS bucket capacity (peak burst). When `None`, capacity
    /// equals the `iops` refill rate (1-second burst). When `Some`,
    /// the value must be `>= iops` (a capacity below the refill rate
    /// would discard refilled tokens immediately and effectively
    /// reduce the steady-state rate); [`DiskThrottle::validate`]
    /// enforces this. Has no effect when `iops` is `None`.
    pub iops_burst_capacity: Option<NonZeroU64>,
    /// Bandwidth bucket capacity (peak burst, in bytes). When
    /// `None`, capacity equals the `bytes_per_sec` refill rate
    /// (1-second burst). When `Some`, the value must be
    /// `>= bytes_per_sec`. Has no effect when `bytes_per_sec` is
    /// `None`.
    pub bytes_burst_capacity: Option<NonZeroU64>,
}

impl DiskThrottle {
    /// Non-panicking validation of throttle/burst consistency.
    ///
    /// Rejects burst capacities below their corresponding refill
    /// rate. A bucket with capacity below its refill rate cannot
    /// hold a full second of refilled tokens, so the effective
    /// steady-state rate would silently be the capacity, not the
    /// configured rate — a user who sets `iops(1000).iops_burst_capacity(500)`
    /// would expect 1000 IOPS and silently get 500.
    ///
    /// A burst capacity set without a corresponding rate is also
    /// rejected: a bucket with no refill rate is functionally
    /// unbounded one-shot capacity, which does not match any
    /// useful throttling model.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(burst) = self.iops_burst_capacity {
            match self.iops {
                Some(rate) if burst < rate => {
                    return Err(format!(
                        "iops_burst_capacity ({}) must be >= iops ({})",
                        burst, rate
                    ));
                }
                None => {
                    return Err(
                        "iops_burst_capacity set without iops refill rate".into(),
                    );
                }
                _ => {}
            }
        }
        if let Some(burst) = self.bytes_burst_capacity {
            match self.bytes_per_sec {
                Some(rate) if burst < rate => {
                    return Err(format!(
                        "bytes_burst_capacity ({}) must be >= bytes_per_sec ({})",
                        burst, rate
                    ));
                }
                None => {
                    return Err(
                        "bytes_burst_capacity set without bytes_per_sec refill rate".into(),
                    );
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Per-disk config. `Default` is raw 256 MB device on `/dev/vda`;
/// formatting and auto-mount are deferred.
///
/// No backing-file path field: the framework owns the per-test
/// backing file (`tempfile()` for `Raw`, FICLONE-cloned template
/// for `Btrfs`). See module docs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DiskConfig {
    /// Advertised capacity in megabytes. 256 MB default capacity.
    /// Sized to accommodate common guest filesystem formatters;
    /// smaller values are accepted but may cause `mkfs` failures
    /// inside the template VM (see
    /// [`crate::vmm::disk_template::build_template_via_vm`]) for
    /// `Filesystem::Btrfs`.
    pub capacity_mb: u32,
    /// Filesystem to format the per-test backing with. `Raw` leaves
    /// the device unformatted; `Btrfs` routes through the
    /// template-cache lifecycle.
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
    /// Opt out of guest-side auto-mount. Default `false` means a
    /// non-`Raw` disk is auto-mounted at `/mnt/disk0` by the guest
    /// init (see
    /// [`crate::vmm::rust_init::auto_mount_data_disks`]); setting
    /// `true` suppresses the auto-mount cmdline tokens and leaves
    /// `/dev/vda` raw to the test author. Has no effect for
    /// `Filesystem::Raw` disks (there is nothing to mount). The
    /// only honest reason to flip this is a test that wants to
    /// drive the mount path itself (e.g. exercise mount-option
    /// fuzzing or fail-injection on the kernel mount syscall).
    pub no_auto_mount: bool,
}

impl Default for DiskConfig {
    /// 256 MB, [`Filesystem::Raw`], no throttle. The `Raw` default
    /// keeps the on-host cost minimal — no template-VM build, no
    /// cache directory required — and the per-test backing is a
    /// fresh sparse `tempfile()` per VM (see
    /// [`crate::vmm::KtstrVm::init_virtio_blk`]).
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
            no_auto_mount: false,
        }
    }
}

impl DiskConfig {
    /// Set capacity in megabytes.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn capacity_mb(mut self, mb: u32) -> Self {
        self.capacity_mb = mb;
        self
    }

    /// Select the on-disk filesystem.
    ///
    /// `Filesystem::Raw` (the default) leaves the device unformatted.
    /// `Filesystem::Btrfs` routes through
    /// [`crate::vmm::disk_template::ensure_template`]: on cache miss
    /// the framework boots a one-shot template VM that runs
    /// `mkfs.btrfs` inside the guest, caches the formatted image,
    /// and per-test boots reflink-clone it. The lifecycle requires
    /// a reflink-capable cache directory (btrfs or xfs) and a host
    /// `mkfs.btrfs` binary on `PATH` at template-build time. See
    /// the module-level docs and [`crate::vmm::disk_template`].
    #[must_use = "builder methods consume self; bind the result"]
    pub fn filesystem(mut self, fs: Filesystem) -> Self {
        self.filesystem = fs;
        self
    }

    /// Set IOPS throttle. Passing 0 disables IOPS throttling
    /// (equivalent to `None`). To throttle near-zero, use `iops(1)`.
    /// There is no "block all IO" mode — the minimum throttled rate
    /// is 1 op/sec. Any positive value is wrapped in `NonZeroU64`.
    ///
    /// Clearing the rate (`iops(0)`) also clears the matching
    /// `iops_burst_capacity` — a burst capacity without a refill
    /// rate is invalid (caught by [`DiskThrottle::validate`]) and
    /// keeping a stale burst around after the user explicitly
    /// disabled the rate is a footgun: the next `validate()` call
    /// would fail with a less-helpful "burst without rate" error
    /// rather than the user's intent (a fully-unthrottled bucket).
    #[must_use = "builder methods consume self; bind the result"]
    pub fn iops(mut self, iops: u64) -> Self {
        self.throttle.iops = NonZeroU64::new(iops);
        if self.throttle.iops.is_none() {
            self.throttle.iops_burst_capacity = None;
        }
        self
    }

    /// Set bandwidth throttle (bytes per second). A zero value
    /// disables bandwidth throttling (stored as `None`); any
    /// positive value is wrapped in `NonZeroU64`.
    ///
    /// Clearing the rate (`bytes_per_sec(0)`) also clears the
    /// matching `bytes_burst_capacity` for the same reason as
    /// `iops` — a burst without a rate is invalid and stale-burst
    /// retention turns a deliberate "drop the throttle" into a
    /// validate-time failure.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn bytes_per_sec(mut self, bytes_per_sec: u64) -> Self {
        self.throttle.bytes_per_sec = NonZeroU64::new(bytes_per_sec);
        if self.throttle.bytes_per_sec.is_none() {
            self.throttle.bytes_burst_capacity = None;
        }
        self
    }

    /// Set IOPS burst capacity (token-bucket peak). A zero value
    /// clears the burst override (stored as `None`), reverting to
    /// the default 1-second burst (capacity equals refill rate).
    /// Any positive value is wrapped in `NonZeroU64`.
    ///
    /// The capacity must be `>= iops` when both are set, and must
    /// not be set without `iops`. Both rules are enforced by
    /// [`DiskThrottle::validate`] at VM build time, not by the
    /// builder — the builder is order-independent (a user may set
    /// burst before rate). Tests should call `validate()` after
    /// chaining, or construct an invalid config and observe the
    /// error from VM build.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn iops_burst_capacity(mut self, capacity: u64) -> Self {
        self.throttle.iops_burst_capacity = NonZeroU64::new(capacity);
        self
    }

    /// Set bandwidth burst capacity in bytes (token-bucket peak).
    /// A zero value clears the burst override (stored as `None`),
    /// reverting to the default 1-second burst. Any positive value
    /// is wrapped in `NonZeroU64`.
    ///
    /// The capacity must be `>= bytes_per_sec` when both are set,
    /// and must not be set without `bytes_per_sec`. Both rules are
    /// enforced by [`DiskThrottle::validate`] at VM build time, not
    /// by the builder.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn bytes_burst_capacity(mut self, capacity: u64) -> Self {
        self.throttle.bytes_burst_capacity = NonZeroU64::new(capacity);
        self
    }

    /// Mark the disk read-only (advertises `VIRTIO_BLK_F_RO`).
    /// Default is read-write; this builder takes no argument (no
    /// boolean footgun) and only flips the flag on. To return to
    /// read-write, drop the call or reconstruct from
    /// `DiskConfig::default()`.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    /// Attach a human-readable label to this disk. WorkType variants
    /// that need to address a specific disk (e.g. one of several
    /// attached) can resolve the name instead of relying on
    /// attachment order. Default is anonymous (`None`); calling
    /// `.name(...)` sets it.
    ///
    /// The name also drives the guest auto-mount path: a disk
    /// named `"data"` auto-mounts at `/mnt/data` instead of the
    /// default `/mnt/disk0`. See [`Self::no_auto_mount`] to opt
    /// out of auto-mount entirely.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Suppress the guest-side auto-mount of this disk. Default
    /// behavior auto-mounts a non-`Raw` disk at the path returned
    /// by [`Self::auto_mount_path`]; calling this method flips
    /// the flag on. Useful for tests that want raw access to
    /// `/dev/vda` after a host-driven mkfs (e.g. mount-option
    /// fuzzing, deliberate mount-failure injection, manual
    /// subvolume traversal).
    ///
    /// No-op for `Filesystem::Raw` disks (there is nothing to
    /// mount). The flag is honored at cmdline-emission time in
    /// [`crate::vmm::KtstrVmBuilder::build`]: when set, the
    /// `KTSTR_DISK0_FS` / `KTSTR_DISK0_MOUNT` / `KTSTR_DISK0_RO`
    /// tokens are not emitted, and the guest's
    /// [`crate::vmm::rust_init::auto_mount_data_disks`] short-
    /// circuits at the missing-token check.
    #[must_use = "builder methods consume self; bind the result"]
    pub fn no_auto_mount(mut self) -> Self {
        self.no_auto_mount = true;
        self
    }

    /// Resolve the guest-side mount path for this disk. Returns
    /// `/mnt/<name>` when [`Self::name`] is set, `/mnt/disk0`
    /// otherwise. Used by the cmdline emission to populate the
    /// `KTSTR_DISK0_MOUNT` token consumed by the guest's
    /// [`crate::vmm::rust_init::auto_mount_data_disks`].
    pub(crate) fn auto_mount_path(&self) -> String {
        match self.name.as_deref() {
            Some(n) => format!("/mnt/{n}"),
            None => "/mnt/disk0".to_string(),
        }
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
    fn filesystem_builder_sets_variant() {
        let d = DiskConfig::default().filesystem(Filesystem::Btrfs);
        assert_eq!(d.filesystem, Filesystem::Btrfs);
        // Builder is overwriting (not OR-ing) — last call wins.
        let d = d.filesystem(Filesystem::Raw);
        assert_eq!(d.filesystem, Filesystem::Raw);
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
        assert_eq!(serde_json::to_string(&Filesystem::Btrfs).unwrap(), r#""btrfs""#);
        let parsed: Filesystem = serde_json::from_str(r#""raw""#).unwrap();
        assert_eq!(parsed, Filesystem::Raw);
        let parsed: Filesystem = serde_json::from_str(r#""btrfs""#).unwrap();
        assert_eq!(parsed, Filesystem::Btrfs);
    }

    #[test]
    fn filesystem_cache_tag_round_trips_serde_name() {
        // The cache_tag is the on-disk identifier used in the
        // template-cache key. Pinning that it matches the serde
        // serialization keeps the two name spaces aligned — a future
        // `#[serde(rename = "...")]` change must update cache_tag in
        // lock-step or the cache stops finding old entries.
        for fs in [Filesystem::Raw, Filesystem::Btrfs] {
            let json = serde_json::to_string(&fs).unwrap();
            let stripped = json.trim_matches('"');
            assert_eq!(fs.cache_tag(), stripped, "cache_tag drift for {fs:?}");
        }
    }

    #[test]
    fn throttle_default_is_unthrottled() {
        let t = DiskThrottle::default();
        assert!(t.iops.is_none());
        assert!(t.bytes_per_sec.is_none());
        assert!(t.iops_burst_capacity.is_none());
        assert!(t.bytes_burst_capacity.is_none());
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
                iops_burst_capacity: NonZeroU64::new(10_000),
                bytes_burst_capacity: NonZeroU64::new(200 * 1024 * 1024),
            },
            read_only: true,
            name: Some("data-disk".to_string()),
            no_auto_mount: false,
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
        assert_eq!(
            parsed.throttle.iops_burst_capacity,
            original.throttle.iops_burst_capacity
        );
        assert_eq!(
            parsed.throttle.bytes_burst_capacity,
            original.throttle.bytes_burst_capacity
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
        assert!(parsed.throttle.iops_burst_capacity.is_none());
        assert!(parsed.throttle.bytes_burst_capacity.is_none());
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

    #[test]
    fn burst_capacity_builders_set_fields() {
        let d = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .bytes_per_sec(10 * 1024 * 1024)
            .bytes_burst_capacity(50 * 1024 * 1024);
        assert_eq!(d.throttle.iops, NonZeroU64::new(1_000));
        assert_eq!(d.throttle.iops_burst_capacity, NonZeroU64::new(5_000));
        assert_eq!(d.throttle.bytes_per_sec, NonZeroU64::new(10 * 1024 * 1024));
        assert_eq!(
            d.throttle.bytes_burst_capacity,
            NonZeroU64::new(50 * 1024 * 1024)
        );
    }

    #[test]
    fn burst_capacity_zero_becomes_none() {
        // Mirrors the iops/bytes_per_sec ergonomics: 0 → None at the
        // type boundary so callers can clear a previously-set burst
        // override without dropping back to a fresh `DiskConfig`.
        let d = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .iops_burst_capacity(0);
        assert!(d.throttle.iops_burst_capacity.is_none());

        let d = DiskConfig::default()
            .bytes_per_sec(1_000)
            .bytes_burst_capacity(5_000)
            .bytes_burst_capacity(0);
        assert!(d.throttle.bytes_burst_capacity.is_none());
    }

    #[test]
    fn burst_capacity_default_is_none() {
        let d = DiskConfig::default();
        assert!(d.throttle.iops_burst_capacity.is_none());
        assert!(d.throttle.bytes_burst_capacity.is_none());
    }

    /// Clearing the rate via `iops(0)` also clears the matching
    /// `iops_burst_capacity`. A burst capacity without a refill
    /// rate is invalid per [`DiskThrottle::validate`]; without
    /// this auto-clear, a `.iops(1000).iops_burst_capacity(5000)
    /// .iops(0)` chain would leave a stale burst that turns the
    /// next `validate()` into a "burst without rate" error
    /// instead of the user's intent (a fully-unthrottled bucket).
    #[test]
    fn clearing_iops_clears_iops_burst() {
        let d = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .iops(0);
        assert!(d.throttle.iops.is_none());
        assert!(
            d.throttle.iops_burst_capacity.is_none(),
            "clearing iops must also clear iops_burst_capacity \
             so validate() doesn't fail with a stale-burst error",
        );
        // bytes side untouched — per-dimension independence.
        let d = DiskConfig::default()
            .bytes_per_sec(2_000)
            .bytes_burst_capacity(8_000)
            .iops(0);
        assert!(d.throttle.bytes_per_sec.is_some());
        assert!(d.throttle.bytes_burst_capacity.is_some());
    }

    /// Clearing the rate via `bytes_per_sec(0)` also clears the
    /// matching `bytes_burst_capacity`. Mirror of
    /// `clearing_iops_clears_iops_burst`.
    #[test]
    fn clearing_bytes_per_sec_clears_bytes_burst() {
        let d = DiskConfig::default()
            .bytes_per_sec(2_000)
            .bytes_burst_capacity(8_000)
            .bytes_per_sec(0);
        assert!(d.throttle.bytes_per_sec.is_none());
        assert!(
            d.throttle.bytes_burst_capacity.is_none(),
            "clearing bytes_per_sec must also clear \
             bytes_burst_capacity",
        );
        // iops side untouched.
        let d = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .bytes_per_sec(0);
        assert!(d.throttle.iops.is_some());
        assert!(d.throttle.iops_burst_capacity.is_some());
    }

    /// After a `clear-rate`-then-validate chain, the result must
    /// validate cleanly. Pins the integration: setting both rate
    /// and burst, then clearing the rate, leaves the throttle in
    /// a state that `validate()` accepts (no orphan-burst error).
    #[test]
    fn clearing_rate_leaves_throttle_validate_clean() {
        let throttle = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .bytes_per_sec(2_000)
            .bytes_burst_capacity(8_000)
            .iops(0)
            .bytes_per_sec(0)
            .throttle;
        assert!(throttle.iops.is_none());
        assert!(throttle.bytes_per_sec.is_none());
        assert!(throttle.iops_burst_capacity.is_none());
        assert!(throttle.bytes_burst_capacity.is_none());
        throttle
            .validate()
            .expect("post-clear throttle must validate clean");
    }

    #[test]
    fn validate_accepts_burst_at_or_above_rate() {
        // burst == rate (the historical 1-second-burst behaviour
        // expressed explicitly).
        DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(1_000)
            .throttle
            .validate()
            .expect("burst == iops accepted");

        // burst > rate (multi-second burst).
        DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .bytes_per_sec(10 * 1024 * 1024)
            .bytes_burst_capacity(50 * 1024 * 1024)
            .throttle
            .validate()
            .expect("burst > rate accepted");

        // No throttle set → trivially valid.
        DiskConfig::default()
            .throttle
            .validate()
            .expect("no throttle accepted");

        // Rate set, burst unset → trivially valid (burst defaults to
        // rate-equivalent at wire-up time).
        DiskConfig::default()
            .iops(1_000)
            .bytes_per_sec(1_000_000)
            .throttle
            .validate()
            .expect("rate without burst accepted");
    }

    #[test]
    fn validate_rejects_burst_below_rate() {
        let err = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(500)
            .throttle
            .validate()
            .expect_err("burst < iops rejected");
        assert!(
            err.contains("iops_burst_capacity") && err.contains("must be >="),
            "unexpected error message: {err}"
        );

        let err = DiskConfig::default()
            .bytes_per_sec(10_000)
            .bytes_burst_capacity(5_000)
            .throttle
            .validate()
            .expect_err("burst < bytes_per_sec rejected");
        assert!(
            err.contains("bytes_burst_capacity") && err.contains("must be >="),
            "unexpected error message: {err}"
        );
    }

    /// Off-by-one boundary: `burst == rate - 1` must be rejected. Pins
    /// the strict `<` vs `<=` direction of the validate predicate
    /// against a future flip that would silently accept a steady-state
    /// rate one below the configured value.
    #[test]
    fn validate_rejects_burst_one_below_rate() {
        let err = DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(999)
            .throttle
            .validate()
            .expect_err("iops burst one below rate must be rejected");
        assert!(
            err.contains("iops_burst_capacity") && err.contains("must be >="),
            "unexpected error message: {err}"
        );

        let err = DiskConfig::default()
            .bytes_per_sec(1_000)
            .bytes_burst_capacity(999)
            .throttle
            .validate()
            .expect_err("bytes burst one below rate must be rejected");
        assert!(
            err.contains("bytes_burst_capacity") && err.contains("must be >="),
            "unexpected error message: {err}"
        );
    }

    /// Builder chain that sets a rate and burst then clears the rate
    /// via `iops(0)` must validate clean — clearing the rate also
    /// clears the matching burst (per the [`DiskConfig::iops`]
    /// auto-clear contract), so the resulting throttle is fully
    /// unthrottled and validate rejects nothing. Distinct from
    /// `clearing_rate_leaves_throttle_validate_clean` (which clears
    /// both rates simultaneously); this one isolates the iops-only
    /// clear path so a regression in just one auto-clear branch
    /// surfaces here.
    #[test]
    fn iops_clear_after_burst_set_validates_clean() {
        DiskConfig::default()
            .iops(1_000)
            .iops_burst_capacity(5_000)
            .iops(0)
            .throttle
            .validate()
            .expect("iops-cleared throttle must validate clean");
    }

    #[test]
    fn validate_rejects_burst_without_rate() {
        let err = DiskConfig::default()
            .iops_burst_capacity(5_000)
            .throttle
            .validate()
            .expect_err("burst without iops rejected");
        assert!(
            err.contains("iops_burst_capacity") && err.contains("without iops"),
            "unexpected error message: {err}"
        );

        let err = DiskConfig::default()
            .bytes_burst_capacity(5_000)
            .throttle
            .validate()
            .expect_err("burst without bytes_per_sec rejected");
        assert!(
            err.contains("bytes_burst_capacity") && err.contains("without bytes_per_sec"),
            "unexpected error message: {err}"
        );
    }

    /// Dedicated serde roundtrip for the burst fields. Distinct from
    /// the full-roundtrip test: that one constructs a `DiskThrottle`
    /// literal, this one drives the builder so a future builder
    /// regression that fails to populate the underlying fields would
    /// surface here even if struct-literal construction stayed
    /// correct.
    #[test]
    fn disk_config_burst_serde_roundtrip() {
        let original = DiskConfig::default()
            .iops(2_500)
            .iops_burst_capacity(10_000)
            .bytes_per_sec(50 * 1024 * 1024)
            .bytes_burst_capacity(200 * 1024 * 1024);

        let json = serde_json::to_string(&original).expect("serialize burst DiskConfig");
        let parsed: DiskConfig =
            serde_json::from_str(&json).expect("deserialize burst DiskConfig");

        assert_eq!(parsed, original);
        assert_eq!(parsed.throttle.iops, NonZeroU64::new(2_500));
        assert_eq!(parsed.throttle.iops_burst_capacity, NonZeroU64::new(10_000));
        assert_eq!(
            parsed.throttle.bytes_per_sec,
            NonZeroU64::new(50 * 1024 * 1024)
        );
        assert_eq!(
            parsed.throttle.bytes_burst_capacity,
            NonZeroU64::new(200 * 1024 * 1024)
        );
    }
}
