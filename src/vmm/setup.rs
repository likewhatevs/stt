//! Boot pipeline for `KtstrVm`: virtio-blk wiring, KVM creation,
//! initramfs resolution and compression, COW overlay, deferred memory
//! computation, x86_64 / aarch64 memory and FDT layout, vCPU register
//! setup.
//!
//! These methods run on the calling thread (no vCPU work yet) and
//! produce a [`KtstrKvm`](super::kvm::KtstrKvm) ready for the
//! [`KtstrVm::run_vm`](super::KtstrVm::run_vm) loop. They are reopened
//! as additional [`impl KtstrVm`](super::KtstrVm) blocks; the canonical
//! struct definition lives in [`super`].

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;
use vm_memory::{Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

use super::KtstrVm;
use super::initramfs_cache::{BaseKey, BaseRef, get_or_build_base};
use super::memory_budget::{MemoryBudget, initramfs_min_memory_mb, read_kernel_init_size};
use super::pi_mutex::PiMutex;
use super::{
    disk_config, disk_template, host_topology, initramfs, shm_ring, virtio_blk, virtio_net,
};

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "aarch64")]
use super::aarch64::boot;
#[cfg(target_arch = "aarch64")]
use super::aarch64::kvm;
#[cfg(target_arch = "x86_64")]
use super::virtio_console;
#[cfg(target_arch = "x86_64")]
use super::x86_64::{acpi, boot, kvm, mptable};

/// Address where initramfs is loaded in guest memory.
#[cfg(target_arch = "x86_64")]
const INITRD_ADDR: u64 = 0x800_0000; // 128 MB

/// Compute initramfs load address at the high end of DRAM, just below
/// the FDT. Matches Firecracker/Cloud Hypervisor placement pattern —
/// avoids conflicts with early kernel allocations near the kernel image.
#[cfg(target_arch = "aarch64")]
fn aarch64_initrd_addr(memory_mb: u32, shm_size: u64, initrd_max_size: u64) -> u64 {
    let fdt_addr = aarch64::fdt::fdt_address(memory_mb, shm_size);
    // Place initrd just below FDT, page-aligned.
    (fdt_addr - initrd_max_size) & !0xFFF
}

/// Build the auto-mount cmdline tokens for one disk. Returns an
/// empty string when no auto-mount is requested (Raw filesystem,
/// or `no_auto_mount` opt-out); otherwise returns the
/// space-prefixed `KTSTR_DISK0_FS=... KTSTR_DISK0_MOUNT=...`
/// pair, with `KTSTR_DISK0_RO=1` appended when `read_only` is
/// set.
///
/// Free fn so cfg(test) unit tests cover all branches without
/// driving a full `setup_memory` call.
///
/// Token contract (consumed by
/// [`crate::vmm::rust_init::auto_mount_data_disks`]):
/// * `KTSTR_DISK0_FS=<cache_tag>` — fstype string for the
///   `mount(2)` syscall. Reuses `Filesystem::cache_tag()` so the
///   on-disk-format identifier and the cmdline value stay in
///   lockstep.
/// * `KTSTR_DISK0_MOUNT=<path>` — guest-side mount point. Driven
///   by `DiskConfig::auto_mount_path` (`/mnt/<name>` when
///   `name` is set, `/mnt/disk0` otherwise).
/// * `KTSTR_DISK0_RO=1` — emitted only when `read_only` is set
///   (matches the host-side virtio-blk F_RO advertisement). The
///   guest sets `MS_RDONLY` proactively rather than letting the
///   kernel fail with -EROFS when bdev RO meets RW mount.
pub(crate) fn disk_auto_mount_cmdline_tokens(disk: &disk_config::DiskConfig) -> String {
    if disk.filesystem == disk_config::Filesystem::Raw || disk.no_auto_mount {
        return String::new();
    }
    let mut s = format!(
        " KTSTR_DISK0_FS={} KTSTR_DISK0_MOUNT={}",
        disk.filesystem.cache_tag(),
        disk.auto_mount_path(),
    );
    if disk.read_only {
        s.push_str(" KTSTR_DISK0_RO=1");
    }
    s
}

impl KtstrVm {
    /// Construct the optional virtio-blk device for the configured
    /// disk in `self.disks`. Returns `Ok(None)` when no disk is
    /// attached.
    ///
    /// On `Ok(Some(_))`, the returned `Arc<PiMutex<VirtioBlk>>` has:
    ///   - the backing file open (sparse temp file when
    ///     `disk.backing_path` is `None`, otherwise the operator-supplied
    ///     path),
    ///   - the file extended to `disk.capacity_bytes()` (so unallocated
    ///     reads return zeros via short-read padding in `handle_read`),
    ///   - the throttle wired in,
    ///   - the irqfd registered with the VM,
    ///   - guest memory set so subsequent `process_requests` calls can
    ///     read/write descriptor data.
    ///
    /// The framework reserves a single MMIO base + IRQ pair
    /// (`VIRTIO_BLK_MMIO_BASE` / `VIRTIO_BLK_IRQ`); the builder's
    /// `.disk()` enforces the single-disk constraint by overwriting
    /// any previous disk on each call.
    pub(super) fn init_virtio_blk(
        &self,
        vm: &kvm::KtstrKvm,
    ) -> Result<Option<Arc<PiMutex<virtio_blk::VirtioBlk>>>> {
        if self.disks.is_empty() {
            return Ok(None);
        }
        let disk = &self.disks[0];
        let capacity = disk.capacity_bytes();

        // Throttle sanity gate. `DiskThrottle::validate` rejects
        // burst capacities below their refill rate (which would
        // silently cap the steady-state at the lower capacity
        // instead of the configured rate) and burst capacities set
        // without a refill rate (a one-shot bucket that never
        // refills). Run BEFORE allocating any backing-file resources
        // so a misconfigured throttle bails before disk-side host
        // commitments.
        //
        // The typed `DiskThrottleValidationError` carries the
        // failing dimension (iops/bytes) so callers downcasting via
        // `err.downcast_ref::<DiskThrottleValidationError>()` can
        // route a programmatic recovery without parsing the
        // rendered message.
        disk.throttle
            .validate()
            .map_err(|e| anyhow::anyhow!(e).context("invalid disk throttle"))?;

        // Per-test backing-file allocation forks on the configured
        // [`disk_config::Filesystem`], with one override for the
        // template-build VM driver:
        //
        //  - **`template_staging_image` set** (internal-only — see
        //    [`KtstrVmBuilder::template_staging_image`]): open the
        //    caller-supplied path RW and hand it to the device. This
        //    branch exists exclusively for
        //    [`disk_template::build_template_via_vm`]: the driver
        //    materialises a sparse staging image, points the
        //    template-build guest at it via this field, and recovers
        //    the now-formatted file after VM exit for
        //    [`disk_template::store_atomic`]. Bypasses both the
        //    `Raw` tempfile and `Btrfs` ensure_template branches so
        //    the template-build VM cannot recursively re-enter the
        //    cache it is itself populating.
        //
        //  - `Raw`: anonymous sparse `tempfile()`. The kernel
        //    reclaims storage when the device drops the File. No
        //    cache, no FICLONE.
        //
        //  - `Btrfs`: FICLONE-clones the host-cached, guest-formatted
        //    template into a per-test tempfile under the cache root
        //    (so FICLONE source and dest share a filesystem), unlinks
        //    the dest immediately after open so the device sees the
        //    same anonymous-file semantics as the `Raw` path, and
        //    hands the open `File` to the `VirtioBlk` device. See
        //    [`crate::vmm::disk_template`] module docs.
        let backing = if let Some(staging) = self.template_staging_image.as_ref() {
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(staging)
                .with_context(|| {
                    format!(
                        "open template staging image {} for virtio-blk",
                        staging.display(),
                    )
                })?
        } else {
            match disk.filesystem {
                disk_config::Filesystem::Raw => {
                    let f = tempfile::tempfile()
                        .context("create virtio-blk sparse temp backing file")?;
                    // Make sure the file covers the advertised capacity.
                    // set_len creates a sparse file: holes don't consume
                    // disk space until written.
                    f.set_len(capacity)
                        .context("set virtio-blk backing file length")?;
                    f
                }
                disk_config::Filesystem::Btrfs => {
                    let template =
                        disk_template::ensure_template(disk_config::Filesystem::Btrfs, capacity)
                            .context("ensure btrfs disk template")?;
                    let cache_root = disk_template::cache_root()
                        .context("resolve disk-template cache root for per-test clone")?;
                    std::fs::create_dir_all(&cache_root)
                        .with_context(|| format!("create cache root {cache_root:?}"))?;
                    // Generate a unique per-test path under the cache
                    // root. Use pid + timestamp_ns + random_u64 so
                    // concurrent tests in the same process and across
                    // processes never collide.
                    let dest = cache_root.join(format!(
                        ".per-test-{pid}-{ns:x}-{rnd:x}.img",
                        pid = std::process::id(),
                        ns = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0),
                        rnd = rand::random::<u64>(),
                    ));
                    let f = disk_template::clone_to_per_test(&template, &dest)
                        .context("FICLONE template into per-test backing")?;
                    // Unlink the dest path immediately. The open File
                    // keeps the inode alive for the device's lifetime;
                    // the kernel reclaims storage on drop, matching the
                    // `tempfile()` semantics of the Raw branch.
                    //
                    // If the unlink fails (very rare — ENOENT means a
                    // peer beat us to it, EACCES means the operator's
                    // cache permissions are broken, EBUSY can come from
                    // some FUSE backings), we keep the open File and
                    // warn — the device still works on the open fd, the
                    // only consequence is a stale path on disk that the
                    // next cache GC sweeps. Do NOT propagate the error,
                    // because the device's per-test backing is already
                    // valid and aborting VM init would be a regression
                    // versus the Raw branch where `tempfile::tempfile()`
                    // returns an already-unlinked file with no failure
                    // mode.
                    if let Err(e) = std::fs::remove_file(&dest) {
                        tracing::warn!(
                            path = %dest.display(),
                            error = %e,
                            "failed to unlink per-test btrfs backing after \
                             FICLONE; the open File still backs the device, \
                             but the leftover path will accumulate in the \
                             cache directory until manual cleanup or the \
                             next disk-template cache GC pass."
                        );
                    }
                    f
                }
            }
        };

        let mut blk =
            virtio_blk::VirtioBlk::with_options(backing, capacity, disk.throttle, disk.read_only);
        // Worker placement extracted from the host-topology plan.
        // Perf-mode produces `pinning_plan.service_cpu` (a dedicated
        // host CPU reserved away from vCPU pins) — the worker pins
        // there to keep its cache footprint out of the workload-
        // measured cpuset. Non-perf + `--cpu-cap` produces
        // `no_perf_plan.cpus` (the LLC mask shared with vCPUs); the
        // worker shares the LLC but stays inside the resource budget.
        // The two paths are orthogonal (perf-mode never has
        // `no_perf_plan` and vice versa); both `None` means inherit
        // the parent's affinity (degraded-sysfs / non-cap-set
        // fallback). The setter only takes effect on the next worker
        // spawn — `with_options` deferred initial spawn to DRIVER_OK
        // (matching the respawn path), so this call lands inside the
        // window and the first worker observes the placement.
        let placement = virtio_blk::WorkerPlacement {
            service_cpu: self.pinning_plan.as_ref().and_then(|p| p.service_cpu),
            no_perf_cpus: self.no_perf_plan.as_ref().map(|p| p.cpus.clone()),
        };
        blk.set_worker_placement(placement);
        blk.set_mem((*vm.guest_mem).clone());
        let blk_arc = Arc::new(PiMutex::new(blk));

        // irqfd registration. On x86_64 with split irqchip, IOAPIC
        // routing is unavailable: the kernel's split-irqchip mode
        // emulates the LAPIC in-kernel and leaves PIC/IOAPIC to
        // userspace. The framework does not implement userspace IOAPIC
        // dispatch for virtio-mmio, and the kernel virtio_blk driver
        // has no `mq_ops->timeout` (drivers/block/virtio_blk.c) and no
        // polling fallback — without an IRQ delivery path, blk-mq
        // hangs on every request until the hung-task watchdog fires
        // (default 120 s). Reject loudly here so a topology that
        // exceeds the 8-bit xAPIC limit (max APIC ID > 254) surfaces
        // immediately instead of producing a silent guest hang.
        #[cfg(target_arch = "x86_64")]
        if vm.split_irqchip {
            anyhow::bail!(
                "virtio-blk requires irqfd; split-irqchip mode has no \
                 IOAPIC and the kernel virtio_mmio driver has no polling \
                 fallback — reduce topology so all APIC IDs are at or below 254 (MAX_XAPIC_ID)",
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
            vm.vm_fd
                .register_irqfd(blk_arc.lock().irq_evt(), kvm::VIRTIO_BLK_IRQ)
                .context("register virtio-blk irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(blk_arc.lock().irq_evt(), kvm::VIRTIO_BLK_IRQ)
                .context("register virtio-blk irqfd")?;
        }

        Ok(Some(blk_arc))
    }

    /// Construct the optional virtio-net device for the configured
    /// network in `self.network`. Returns `Ok(None)` when no network
    /// is attached.
    ///
    /// On `Ok(Some(_))`, the returned `Arc<PiMutex<VirtioNet>>` has:
    ///   - the configured MAC baked into config space,
    ///   - guest memory set so subsequent `process_tx_loopback` calls
    ///     can read TX descriptor data and write into RX descriptors,
    ///   - the irqfd registered with the VM (rejected on x86 split
    ///     irqchip via `bail!()` below, matching virtio-blk).
    ///
    /// The framework reserves a single MMIO base + IRQ pair
    /// (`VIRTIO_NET_MMIO_BASE` / `VIRTIO_NET_IRQ`); the builder's
    /// `.network()` enforces the single-device constraint by
    /// overwriting any previous network on each call.
    pub(super) fn init_virtio_net(
        &self,
        vm: &kvm::KtstrKvm,
    ) -> Result<Option<Arc<PiMutex<virtio_net::VirtioNet>>>> {
        let Some(cfg) = self.network else {
            return Ok(None);
        };
        let mut dev = virtio_net::VirtioNet::new(cfg);
        dev.set_mem((*vm.guest_mem).clone());
        let net_arc = Arc::new(PiMutex::new(dev));

        // irqfd registration. Same split-irqchip rejection rationale
        // as virtio-blk above: the kernel virtio_net driver depends
        // on IRQ-driven NAPI to wake on RX, and an undelivered IRQ
        // produces a silent guest hang. Reject loudly so the test
        // setup is caught here.
        #[cfg(target_arch = "x86_64")]
        if vm.split_irqchip {
            anyhow::bail!(
                "virtio-net requires irqfd; split-irqchip mode has no \
                 IOAPIC and the kernel virtio_mmio driver has no polling \
                 fallback — reduce topology so all APIC IDs are at or below 254 (MAX_XAPIC_ID)",
            );
        }
        #[cfg(target_arch = "x86_64")]
        {
            vm.vm_fd
                .register_irqfd(net_arc.lock().irq_evt(), kvm::VIRTIO_NET_IRQ)
                .context("register virtio-net irqfd")?;
        }
        #[cfg(target_arch = "aarch64")]
        {
            vm.vm_fd
                .register_irqfd(net_arc.lock().irq_evt(), kvm::VIRTIO_NET_IRQ)
                .context("register virtio-net irqfd")?;
        }

        Ok(Some(net_arc))
    }

    /// Create the KVM VM and optionally load the kernel.
    ///
    /// When `memory_mb` is `Some`, allocates guest memory and loads the
    /// kernel immediately (existing path). When `None` (deferred), creates
    /// the VM without memory — allocation and kernel loading happen later
    /// in `setup_memory` after the actual initramfs size is known.
    pub(super) fn create_vm_and_load_kernel(
        &self,
    ) -> Result<(kvm::KtstrKvm, Option<boot::KernelLoadResult>)> {
        let t0 = Instant::now();
        let use_hugepages = self.performance_mode
            && self.memory_mb.is_some_and(|mb| {
                host_topology::hugepages_free() >= host_topology::hugepages_needed(mb)
            });

        let vm = match self.memory_mb {
            Some(mb) => {
                if use_hugepages {
                    kvm::KtstrKvm::new_with_hugepages(self.topology, mb, self.performance_mode)
                        .context("create VM with hugepages")?
                } else {
                    kvm::KtstrKvm::new(self.topology, mb, self.performance_mode)
                        .context("create VM")?
                }
            }
            None => {
                kvm::KtstrKvm::new_deferred(self.topology, use_hugepages, self.performance_mode)
                    .context("create VM (deferred memory)")?
            }
        };
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "kvm_create");

        // When memory is already allocated (non-deferred path), do mbind
        // and load kernel now. Deferred path does this in setup_memory.
        let kernel_result = if self.memory_mb.is_some() {
            if self.performance_mode && !self.mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some on the non-deferred allocation path: \
                     allocate_and_register_memory ran during `vm_new` because \
                     memory_mb was provided up front, and that call sets \
                     numa_layout to Some(...) in src/vmm/{x86_64,aarch64}/kvm.rs",
                );
                layout.mbind_regions(&vm.guest_mem, &self.mbind_node_map);
            }

            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");
            Some(kr)
        } else {
            None
        };

        Ok((vm, kernel_result))
    }

    /// Spawn initramfs resolution on a background thread.
    /// Returns the handle to join later (after KVM creation completes).
    pub(super) fn spawn_initramfs_resolve(&self) -> Option<JoinHandle<Result<(BaseRef, BaseKey)>>> {
        let bin = self.init_binary.as_ref()?;
        let payload = bin.clone();
        let scheduler = self.scheduler_binary.clone();
        let probe = self.jemalloc_probe_binary.clone();
        let worker = self.jemalloc_alloc_worker_binary.clone();
        let include_files = self.include_files.clone();
        let busybox = self.busybox;
        std::thread::Builder::new()
            .name("initramfs-resolve".into())
            .spawn(move || -> Result<(BaseRef, BaseKey)> {
                // Extras are stripped by `build_initramfs_base`
                // before write. The scheduler and probe can lose
                // their DWARF without functional impact — the probe
                // resolves `tsd_s.thread_allocated` offsets against
                // the TARGET process's `/proc/<pid>/exe`, not against
                // its own binary, so its own DWARF is dead weight.
                // The worker (the probe's target) MUST retain DWARF:
                // a stripped worker has no DWARF for the probe to
                // walk. Route scheduler + probe through `extras`
                // (stripped), worker through `include_files`
                // (verbatim). Packing the probe unstripped inflated
                // the initramfs by ~900MB per run in debug builds,
                // which was enough to time out VM init before the
                // test binary loaded.
                let mut extras: Vec<(&str, &std::path::Path)> = Vec::new();
                if let Some(s) = scheduler.as_deref() {
                    extras.push(("scheduler", s));
                }
                if let Some(p) = probe.as_deref() {
                    extras.push(("bin/ktstr-jemalloc-probe", p));
                }
                // Shell-mode cache keying treats ANY include_files
                // as shell-mode. `jemalloc_alloc_worker_binary` is
                // still a real include_file at the cache-key layer —
                // hash it accordingly so a binary-change invalidates
                // the cache. The probe is hashed explicitly regardless
                // of its routing (see `BaseKey::new_shell`). The
                // scheduler stays in the non-shell path.
                let has_jemalloc_extras = probe.as_deref().is_some() || worker.as_deref().is_some();
                let shell_mode = busybox || !include_files.is_empty() || has_jemalloc_extras;

                // Merge include_files with worker so both the cache
                // key and the actual archive build see the same
                // worker entry; the probe is added to extras above.
                let mut merged_includes: Vec<(String, PathBuf)> = include_files.clone();
                if let Some(w) = worker.as_deref() {
                    merged_includes.push((
                        "bin/ktstr-jemalloc-alloc-worker".to_string(),
                        w.to_path_buf(),
                    ));
                }

                let key = if shell_mode {
                    BaseKey::new_shell(
                        &payload,
                        scheduler.as_deref(),
                        probe.as_deref(),
                        worker.as_deref(),
                        &merged_includes,
                        busybox,
                    )?
                } else {
                    BaseKey::new(
                        &payload,
                        scheduler.as_deref(),
                        probe.as_deref(),
                        worker.as_deref(),
                    )?
                };

                let include_refs: Vec<(&str, &std::path::Path)> = merged_includes
                    .iter()
                    .map(|(a, p)| (a.as_str(), p.as_path()))
                    .collect();
                let base = get_or_build_base(&payload, &extras, &include_refs, busybox, &key)?;
                Ok((base, key))
            })
            .ok()
    }

    /// Compress base+suffix as separate LZ4 legacy streams, load into
    /// guest memory via COW overlay (falling back to write_slice), and
    /// verify the write. Returns `total_compressed_size`.
    ///
    /// On a successful COW overlay, the returned `CowOverlayGuard` is
    /// pushed onto `vm.cow_overlay_guards` IMMEDIATELY — before any
    /// subsequent fallible operation (suffix write, read-back verify)
    /// runs. This is deliberate: if a later `?` unwinds this function
    /// after the MAP_FIXED overlay is in place, a locally-held guard
    /// would drop first, releasing `LOCK_SH` while the COW VMAs are
    /// still live. A concurrent writer could then take `LOCK_EX` and
    /// truncate the segment → SIGBUS on the mapped pages. Pushing the
    /// guard onto `vm` transfers ownership to the VM, where Drop
    /// order is structurally enforced (guard drops AFTER
    /// `_reservation` munmaps the COW VMAs).
    #[cfg(target_arch = "x86_64")]
    fn compress_and_load_initrd(
        &self,
        vm: &mut kvm::KtstrKvm,
        base_bytes: &[u8],
        suffix: &[u8],
        key: &BaseKey,
        load_addr: u64,
    ) -> Result<u32> {
        let uncompressed_size = base_bytes.len() + suffix.len();

        // Compress base and suffix as separate LZ4 legacy streams. The
        // kernel initramfs decompressor handles concatenated LZ4 natively
        // (re-encountering the magic mid-stream resets the decoder).
        // Keeping them separate lets us COW-map the base from SHM.
        let t0 = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(suffix);
        let total_compressed = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            lz4_base = lz4_base.len(),
            lz4_suffix = lz4_suffix.len(),
            ratio = format!("{:.1}x", uncompressed_size as f64 / total_compressed as f64),
            "lz4_initramfs",
        );

        tracing::debug!(
            base_magic = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                lz4_base[0], lz4_base[1], lz4_base[2], lz4_base[3]
            ),
            suffix_magic = format!(
                "{:02x}{:02x}{:02x}{:02x}",
                lz4_suffix[0], lz4_suffix[1], lz4_suffix[2], lz4_suffix[3]
            ),
            base_len = lz4_base.len(),
            suffix_len = lz4_suffix.len(),
            total = total_compressed,
            load_addr = format!("{:#x}", load_addr),
            suffix_addr = format!("{:#x}", load_addr + lz4_base.len() as u64),
            "initrd_load_debug",
        );

        // Try COW overlay: mmap compressed base from SHM fd directly
        // into guest memory, sharing physical pages across VMs.
        let t0 = Instant::now();
        let cow_guard = self.try_cow_overlay(&vm.guest_mem, key, lz4_base.len(), load_addr);
        // IMPORTANT: stash the guard on the VM IMMEDIATELY — before
        // any fallible operation below. If a `?` unwinds this function
        // with a locally-held guard still on the stack, the guard
        // drops first, releasing LOCK_SH while the COW VMAs are still
        // live. Owned by `vm`, the guard drops with the VM's
        // declared-order Drop, which is strictly after
        // `_reservation` (and thus the COW VMAs). See
        // `try_cow_overlay_rejects_cross_region_span` and the C4
        // comment on `cow_overlay_guards` in kvm.rs.
        let cow_active = cow_guard.is_some();
        if let Some(guard) = cow_guard {
            vm.cow_overlay_guards.push(guard);
        }
        if cow_active {
            vm.guest_mem
                .write_slice(&lz4_suffix, GuestAddress(load_addr + lz4_base.len() as u64))
                .context("write lz4 suffix after COW base")?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                cow = true,
                "initrd_write"
            );
        } else {
            initramfs::load_initramfs_parts(&vm.guest_mem, &[&lz4_base, &lz4_suffix], load_addr)?;
            tracing::debug!(
                elapsed_us = t0.elapsed().as_micros(),
                cow = false,
                "initrd_write"
            );
        }

        // Read back first 8 bytes from guest memory to check write.
        let mut check_buf = [0u8; 8];
        vm.guest_mem
            .read_slice(&mut check_buf, GuestAddress(load_addr))
            .context("read-back initrd check")?;
        tracing::debug!(
            first_8 = format!(
                "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                check_buf[0],
                check_buf[1],
                check_buf[2],
                check_buf[3],
                check_buf[4],
                check_buf[5],
                check_buf[6],
                check_buf[7]
            ),
            expected_magic = "02214c18",
            "initrd_verify",
        );

        Ok(total_compressed as u32)
    }

    /// Join the initramfs thread and load the result into guest memory.
    /// Memory must already be allocated (non-deferred path). Validates
    /// that allocated memory is sufficient for the initramfs.
    #[cfg(target_arch = "x86_64")]
    fn join_and_load_initramfs(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        load_addr: u64,
    ) -> Result<(Option<u64>, Option<u32>)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Enforce minimum memory for initramfs extraction.
        // This path is only reached when memory_mb was set explicitly.
        let memory_mb = self.memory_mb.expect(
            "join_and_load_initramfs called in deferred mode; \
             use join_compute_memory_and_load instead",
        );
        // Compress first to get actual compressed size for validation.
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        let kernel_init_size = read_kernel_init_size(&self.kernel).unwrap_or(0) as u64;
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            shm_bytes: self.shm_size,
        };
        let min_mb = initramfs_min_memory_mb(&budget);
        if memory_mb < min_mb {
            anyhow::bail!(
                "VM memory {}MB insufficient for initramfs \
                 (uncompressed={}MB, compressed={}MB, \
                 init_size={}MB): need {}MB",
                memory_mb,
                uncompressed_size >> 20,
                compressed_size >> 20,
                kernel_init_size >> 20,
                min_mb,
            );
        }

        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size)))
    }

    /// Deferred memory path: join initramfs, compute memory from actual
    /// size, allocate guest memory, then load initramfs.
    ///
    /// Returns `(initrd_addr, initrd_size, memory_mb)`.
    #[cfg(target_arch = "x86_64")]
    fn join_compute_memory_and_load(
        &self,
        vm: &mut kvm::KtstrKvm,
        handle: JoinHandle<Result<(BaseRef, BaseKey)>>,
        load_addr: u64,
    ) -> Result<(Option<u64>, Option<u32>, u32)> {
        let t0 = Instant::now();
        let (base, key) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "initramfs_join");
        let base_bytes: &[u8] = base.as_ref();

        let t0 = Instant::now();
        let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
        let uncompressed_size = base_bytes.len() + suffix.len();
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            base_bytes = base_bytes.len(),
            suffix_bytes = suffix.len(),
            "build_suffix",
        );

        // Compress before computing memory so the formula uses actual
        // compressed size instead of guessing.
        let t0_compress = Instant::now();
        let lz4_base = self.get_or_compress_base(base_bytes, &key)?;
        let lz4_suffix = initramfs::lz4_legacy_compress(&suffix);
        let compressed_size = lz4_base.len() + lz4_suffix.len();
        tracing::debug!(
            elapsed_us = t0_compress.elapsed().as_micros(),
            uncompressed = uncompressed_size,
            compressed = compressed_size,
            ratio = format!("{:.1}x", uncompressed_size as f64 / compressed_size as f64),
            "deferred_lz4_compress",
        );

        // Compute memory from actual sizes, honoring the
        // topology-requested minimum when non-zero.
        let kernel_init_size = read_kernel_init_size(&self.kernel).unwrap_or(0) as u64;
        let budget = MemoryBudget {
            uncompressed_initramfs_bytes: uncompressed_size as u64,
            compressed_initrd_bytes: compressed_size as u64,
            kernel_init_size,
            shm_bytes: self.shm_size,
        };
        let memory_mb = initramfs_min_memory_mb(&budget).max(self.memory_min_mb);
        tracing::debug!(
            uncompressed_mb = uncompressed_size >> 20,
            compressed_mb = compressed_size >> 20,
            init_size_mb = kernel_init_size >> 20,
            memory_min_mb = self.memory_min_mb,
            memory_mb,
            "deferred_memory_computed",
        );

        // Allocate and register guest memory.
        vm.allocate_and_register_memory(memory_mb)
            .with_context(|| format!("allocate deferred memory ({memory_mb}MB)"))?;

        // Load pre-compressed data into guest memory. The base is already
        // in the LZ4 SHM cache from get_or_compress_base above, so
        // compress_and_load_initrd will hit the cache.
        let size = self.compress_and_load_initrd(vm, base_bytes, &suffix, &key, load_addr)?;
        Ok((Some(load_addr), Some(size), memory_mb))
    }

    pub(super) fn effective_memory_mb(&self, guest_mem: &GuestMemoryMmap) -> u32 {
        use vm_memory::GuestMemoryRegion;
        match self.memory_mb {
            Some(mb) => mb,
            None => {
                let total_bytes: u64 = guest_mem.iter().map(|r| r.len()).sum();
                (total_bytes >> 20) as u32
            }
        }
    }

    /// Get or build the compressed base. Checks LZ4 SHM first, then
    /// compresses and stores.
    #[cfg(target_arch = "x86_64")]
    fn get_or_compress_base(&self, base_bytes: &[u8], key: &BaseKey) -> Result<Vec<u8>> {
        // Try loading compressed base from LZ4 SHM.
        if let Some((fd, len)) = initramfs::shm_open_lz4(key.0) {
            use std::os::fd::AsRawFd;
            let mut buf = vec![0u8; len];
            unsafe {
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                );
                if ptr != libc::MAP_FAILED {
                    std::ptr::copy_nonoverlapping(ptr as *const u8, buf.as_mut_ptr(), len);
                    libc::munmap(ptr, len);
                    initramfs::shm_close_fd(fd);

                    // Validate LZ4 legacy magic. Stale segments from a
                    // previous compression format (zstd) must be discarded.
                    if buf.len() >= 4 && buf[..4] == initramfs::LZ4_LEGACY_MAGIC {
                        tracing::debug!(bytes = len, "lz4_base cache hit (shm)");
                        return Ok(buf);
                    }
                    tracing::warn!(
                        bytes = len,
                        magic = format!("{:02x}{:02x}{:02x}{:02x}", buf[0], buf[1], buf[2], buf[3]),
                        "stale compressed shm segment (wrong magic), recompressing"
                    );
                } else {
                    initramfs::shm_close_fd(fd);
                }
            }
        }

        // Compress with LZ4 legacy format.
        let lz4 = initramfs::lz4_legacy_compress(base_bytes);

        if let Err(e) = initramfs::shm_store_lz4(key.0, &lz4) {
            tracing::warn!("shm_store_lz4: {e:#}");
        }
        Ok(lz4)
    }

    /// Try to COW-overlay the compressed base from LZ4 SHM into guest
    /// memory. Returns `Some(CowOverlayGuard)` on success — the guard
    /// owns the SHM fd and holds `LOCK_SH` for the mapping's lifetime,
    /// and MUST be kept alive as long as the COW overlay is in use
    /// (typically the VM lifetime). Validates the segment starts with
    /// LZ4 legacy magic to reject stale data from a previous
    /// compression format.
    #[cfg(target_arch = "x86_64")]
    fn try_cow_overlay(
        &self,
        guest_mem: &GuestMemoryMmap,
        key: &BaseKey,
        expected_len: usize,
        load_addr: u64,
    ) -> Option<initramfs::CowOverlayGuard> {
        let (fd, len) = initramfs::shm_open_lz4(key.0)?;
        if len != expected_len {
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Validate LZ4 legacy magic before COW-mapping.
        use std::os::fd::AsRawFd;
        let mut magic = [0u8; 4];
        unsafe {
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            );
            if ptr == libc::MAP_FAILED {
                initramfs::shm_close_fd(fd);
                return None;
            }
            std::ptr::copy_nonoverlapping(ptr as *const u8, magic.as_mut_ptr(), 4);
            libc::munmap(ptr, len);
        }
        if magic != initramfs::LZ4_LEGACY_MAGIC {
            tracing::warn!(
                magic = format!(
                    "{:02x}{:02x}{:02x}{:02x}",
                    magic[0], magic[1], magic[2], magic[3]
                ),
                "stale compressed shm segment in COW path, skipping"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Refuse zero-length: mmap(len=0) is EINVAL and serves no
        // purpose; the suffix-write fallback handles empty bases
        // trivially. Also refuse load_addr + len overflow before
        // bounds-checking, since GuestAddress arithmetic wraps
        // silently on u64 overflow.
        if len == 0 || load_addr.checked_add(len as u64).is_none() {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: invalid range (zero-length or overflow), falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        // Bounds-check [load_addr, load_addr + len) against guest
        // memory BEFORE the MAP_FIXED mmap. `get_host_address` only
        // validates the start address — without a length check,
        // MAP_FIXED would silently overwrite whatever host VA happens
        // to follow the region (other guest regions, reserved VA, or
        // unrelated mappings). `get_slice` fails if the range extends
        // past the region's end or spans a region boundary, which is
        // exactly the guarantee MAP_FIXED needs.
        if guest_mem.get_slice(GuestAddress(load_addr), len).is_err() {
            tracing::debug!(
                load_addr = format!("{:#x}", load_addr),
                len,
                "cow_overlay: range exceeds guest memory region, falling back"
            );
            initramfs::shm_close_fd(fd);
            return None;
        }
        let Ok(host_addr) = guest_mem.get_host_address(GuestAddress(load_addr)) else {
            initramfs::shm_close_fd(fd);
            return None;
        };
        // cow_overlay takes ownership of `fd` on both Some and None
        // paths: on success the guard carries it; on failure
        // cow_overlay itself closes it. Do NOT call shm_close_fd here.
        unsafe { initramfs::cow_overlay(host_addr, len, fd) }
    }

    /// Initialize the SHM ring buffer header at `shm_base` in guest memory.
    fn init_shm_region(&self, guest_mem: &GuestMemoryMmap, shm_base: u64) -> Result<()> {
        let header = shm_ring::ShmRingHeader::new(self.shm_size as usize);
        guest_mem
            .write_slice(
                zerocopy::IntoBytes::as_bytes(&header),
                GuestAddress(shm_base),
            )
            .context("write SHM header")
    }

    /// Write cmdline, boot params, SHM header, and topology tables to guest memory.
    ///
    /// When `kernel_result` is `None` (deferred memory mode), this method
    /// first joins the initramfs thread to learn the actual size, allocates
    /// guest memory from that size, does mbind, and loads the kernel — all
    /// before proceeding with the normal initramfs load and boot param setup.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn setup_memory(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path: join initramfs first to learn its size,
        // then allocate memory, load kernel, and load initramfs — all in
        // one shot with no estimation.
        let (kernel_result, initrd_addr, initrd_size) = if let Some(kr) = kernel_result {
            // Non-deferred: memory already allocated, kernel already loaded.
            // compress_and_load_initrd transfers the CowOverlayGuard
            // directly onto vm.cow_overlay_guards before any fallible
            // operation, so a mid-function `?` cannot drop the guard
            // before the COW VMAs are torn down.
            let (initrd_addr, initrd_size) = match initramfs_handle {
                Some(handle) => self.join_and_load_initramfs(vm, handle, INITRD_ADDR)?,
                None => (None, None),
            };
            (kr, initrd_addr, initrd_size)
        } else {
            // Deferred memory path: join initramfs first to learn its size,
            // then allocate memory, load kernel, and load initramfs — all in
            // one shot with no estimation.
            let (initrd_addr, initrd_size, _memory_mb) = match initramfs_handle {
                Some(handle) => self.join_compute_memory_and_load(vm, handle, INITRD_ADDR)?,
                None => {
                    // No initramfs — allocate minimum memory.
                    let memory_mb = 256u32;
                    vm.allocate_and_register_memory(memory_mb)
                        .context("allocate deferred memory (no initramfs)")?;
                    (None, None, memory_mb)
                }
            };

            if self.performance_mode && !self.mbind_node_map.is_empty() {
                let layout = vm.numa_layout.as_ref().expect(
                    "numa_layout is Some after the deferred allocate_and_register_memory \
                     call above: that call sets numa_layout to Some(...) in \
                     src/vmm/{x86_64,aarch64}/kvm.rs before this branch can reach here",
                );
                layout.mbind_regions(&vm.guest_mem, &self.mbind_node_map);
            }

            // Load kernel into the freshly allocated memory.
            let t0 = Instant::now();
            let kr = boot::load_kernel(&vm.guest_mem, &self.kernel).context("load kernel")?;
            tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "load_kernel");

            (kr, initrd_addr, initrd_size)
        };

        // Resolve effective memory_mb for boot params / ACPI / SHM.
        let memory_mb = self.effective_memory_mb(&vm.guest_mem);

        // Kernel cmdline rationale (per flag):
        //   console=ttyS0        — serial console for host-visible output.
        //   nomodules            — no out-of-tree modules are shipped; skip modprobe paths.
        //   mitigations=off      — skip Spectre/Meltdown mitigations for VM perf.
        //   no_timer_check       — suppress APIC timer-calibration failure under KVM.
        //   clocksource=kvm-clock — stable paravirt clock; avoid TSC drift under KVM.
        //   random.trust_cpu=on  — seed RNG from RDRAND so userspace doesn't block on entropy.
        //   swiotlb=noforce      — skip the IOMMU bounce buffer — no passthrough devices.
        //   i8042.*=noaux/nomux/nopnp/dumbkbd — skip legacy PS/2 probing; no keyboard/mouse in VM.
        //   pci=off              — no PCI devices emulated; shave boot time by skipping the scan.
        //   reboot=k             — use keyboard-controller reset method.
        //   panic=-1             — reboot immediately on panic; host detects via exit.
        //   iomem=relaxed        — allow guest /dev/mem mmap of the SHM region (see shm_ring.rs).
        //   nokaslr              — deterministic kernel addresses for symbol/offset resolution.
        //   lockdown=none        — permit /dev/mem and unrestricted BPF needed by the test runtime.
        //   sysctl.kernel.unprivileged_bpf_disabled=0 — allow BPF load from the test runtime.
        //   sysctl.kernel.sched_schedstats=1          — enable /proc/schedstat for workload reports.
        //   delayacct                                 — bare boot param consumed by the
        //                                              kernel's `__setup("delayacct", ...)`
        //                                              handler at kernel/delayacct.c:43-48.
        //                                              The handler sets `delayacct_on = 1`
        //                                              during EARLY boot, BEFORE
        //                                              `delayacct_init()` (line 50-55) reads
        //                                              the variable to decide whether to
        //                                              enable the static branch. This is the
        //                                              authoritative way to turn the
        //                                              delayacct subsystem on at boot.
        //   sysctl.kernel.task_delayacct=1            — backup runtime toggle that flips the
        //                                              delayacct_key static_branch via the
        //                                              `kernel.task_delayacct` sysctl declared
        //                                              at kernel/delayacct.c:80. This path
        //                                              fires later via deferred sysctl
        //                                              registration + proc_handler invocation,
        //                                              which has timing fragility relative to
        //                                              the early-boot increment paths
        //                                              (delayacct_blkio_start/_end gated by
        //                                              static_branch_unlikely(&delayacct_key)
        //                                              at kernel/delayacct.c). Both forms are
        //                                              specified — belt and suspenders — so
        //                                              the runtime toggle is on regardless of
        //                                              whether the early-boot or the deferred
        //                                              sysctl path runs first. Without either,
        //                                              /proc/<tid>/stat field 42 and the
        //                                              taskstats delay-accounting fields stay
        //                                              zero on every kernel built with
        //                                              CONFIG_TASK_DELAY_ACCT=y but boot-time
        //                                              off (the upstream default since v5.14).
        let mut cmdline = concat!(
            "console=ttyS0 nomodules mitigations=off ",
            "no_timer_check clocksource=kvm-clock ",
            "random.trust_cpu=on swiotlb=noforce ",
            "i8042.noaux i8042.nomux i8042.nopnp i8042.dumbkbd ",
            "pci=off reboot=k panic=-1 iomem=relaxed nokaslr lockdown=none ",
            "sysctl.kernel.unprivileged_bpf_disabled=0 ",
            "sysctl.kernel.sched_schedstats=1 ",
            "delayacct ",
            "sysctl.kernel.task_delayacct=1",
        )
        .to_string();
        let verbose = std::env::var("KTSTR_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" earlyprintk=serial loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        // Virtio-console MMIO device on the kernel cmdline. The kernel's
        // virtio_mmio_cmdline_devices driver parses this to register the
        // MMIO transport at the given base address and IRQ.
        cmdline.push_str(&format!(
            " virtio_mmio.device={:#x}@{:#x}:{}",
            virtio_console::VIRTIO_MMIO_SIZE,
            kvm::VIRTIO_CONSOLE_MMIO_BASE,
            kvm::VIRTIO_CONSOLE_IRQ,
        ));
        // Virtio-block MMIO device — appended only when the builder
        // attached at least one disk. The kernel's virtio_mmio_cmdline
        // parser registers a MMIO transport per `virtio_mmio.device=`
        // token; the order on the cmdline determines the device-probe
        // order, which in turn determines the `/dev/vd{a,b,...}`
        // assignment. Console-first then blk matches the expected
        // `/dev/vda = first disk` mapping.
        if !self.disks.is_empty() {
            cmdline.push_str(&format!(
                " virtio_mmio.device={:#x}@{:#x}:{}",
                virtio_blk::VIRTIO_MMIO_SIZE,
                kvm::VIRTIO_BLK_MMIO_BASE,
                kvm::VIRTIO_BLK_IRQ,
            ));
            // Auto-mount handshake. Emit a `KTSTR_DISK0_FS=<tag>`
            // token whenever the first disk has been pre-formatted so
            // the guest init at
            // [`crate::vmm::rust_init::auto_mount_data_disks`]
            // can mount `/dev/vda` at `/mnt/disk0` before the test
            // dispatch runs. `Filesystem::Raw` skips the emission
            // because there is no on-disk fs to mount; the guest
            // sees only the absent token and short-circuits the
            // mount path.
            //
            // `KTSTR_DISK0_RO=1` is emitted when the disk is
            // configured `read_only`. The virtio_blk device
            // advertises `VIRTIO_BLK_F_RO` for that case so the
            // guest's gendisk is RO; mounting RW would fail with
            // `-EROFS` (kernel `do_mount` path: `__btrfs_open_devices`
            // probes the bdev's `bdev_read_only` and returns EROFS
            // when the RW mount tries to write). The token lets the
            // guest set `MS_RDONLY` proactively, surfacing the
            // intent in the cmdline and avoiding the kernel-side
            // EROFS path.
            //
            // The cache_tag() value is reused as the fstype string
            // because it is already kebab-free, ≤8 chars, and
            // matches the on-disk-format identifier the host
            // selected — using the same value for both keeps the
            // guest mount and host cache key in lockstep, so a
            // future `Filesystem` variant rename only has to update
            // one place (the `cache_tag` match in disk_config.rs)
            // and the cmdline / mount automatically follow.
            let disk = &self.disks[0];
            cmdline.push_str(&disk_auto_mount_cmdline_tokens(disk));
        }
        // Virtio-net MMIO device — appended only when the builder
        // attached a `NetConfig`. The kernel's virtio_mmio_cmdline
        // parser registers a MMIO transport per `virtio_mmio.device=`
        // token; placing this after virtio-blk does not affect device
        // ordering on the guest's network stack (ifindex is assigned
        // independently of cmdline order).
        if self.network.is_some() {
            cmdline.push_str(&format!(
                " virtio_mmio.device={:#x}@{:#x}:{}",
                virtio_net::VIRTIO_MMIO_SIZE,
                kvm::VIRTIO_NET_MMIO_BASE,
                kvm::VIRTIO_NET_IRQ,
            ));
        }
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            cmdline.push_str(&format!(
                " KTSTR_SHM_BASE={:#x} KTSTR_SHM_SIZE={:#x}",
                shm_base, self.shm_size
            ));
        }
        if self.topology.has_memory_only_nodes() {
            cmdline.push_str(" numa_balancing=enable");
        } else {
            cmdline.push_str(" numa_balancing=0");
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        boot::write_cmdline(&vm.guest_mem, &cmdline)?;
        boot::write_boot_params(
            &vm.guest_mem,
            &cmdline,
            memory_mb,
            initrd_addr,
            initrd_size,
            kernel_result.setup_header.as_ref(),
            self.shm_size,
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "cmdline_boot_params");

        // Initialize SHM ring buffer.
        let t0 = Instant::now();
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = mem_size - self.shm_size;
            self.init_shm_region(&vm.guest_mem, shm_base)?;
        }
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "shm_ring_init");

        let t0 = Instant::now();
        mptable::setup_mptable(&vm.guest_mem, &self.topology)?;
        let _acpi_layout = acpi::setup_acpi(
            &vm.guest_mem,
            &self.topology,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time setup_acpi runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/x86_64/kvm.rs",
            ),
        )?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "mptable_acpi");

        Ok(kernel_result)
    }

    /// Configure BSP and AP vCPUs.
    #[cfg(target_arch = "x86_64")]
    pub(super) fn setup_vcpus(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        boot::setup_sregs(&vm.guest_mem, &vm.vcpus[0], vm.split_irqchip)?;
        boot::setup_regs(&vm.vcpus[0], kernel_entry)?;
        boot::setup_fpu(&vm.vcpus[0])?;
        boot::setup_msrs(&vm.vcpus[0], None)?;
        boot::setup_lapic(&vm.vcpus[0], true)?;
        vm.vcpus[0]
            .set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_RUNNABLE,
            })
            .context("set BSP mp_state")?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");

        let t0 = Instant::now();
        for vcpu in &vm.vcpus[1..] {
            boot::setup_fpu(vcpu)?;
            boot::setup_lapic(vcpu, false)?;
            vcpu.set_mp_state(kvm_bindings::kvm_mp_state {
                mp_state: kvm_bindings::KVM_MP_STATE_UNINITIALIZED,
            })
            .context("set AP mp_state")?;
        }
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            ap_count = vm.vcpus.len().saturating_sub(1),
            "ap_setup"
        );

        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
impl KtstrVm {
    /// Allocate and register guest memory regions for aarch64, including
    /// NUMA-aware placement.
    pub(super) fn setup_memory_aarch64(
        &self,
        vm: &mut kvm::KtstrKvm,
        kernel_result: Option<boot::KernelLoadResult>,
        initramfs_handle: Option<JoinHandle<Result<(BaseRef, BaseKey)>>>,
    ) -> Result<boot::KernelLoadResult> {
        // Deferred memory path for aarch64.
        let kernel_result = if let Some(kr) = kernel_result {
            kr
        } else {
            // Join initramfs to learn actual size, then allocate memory.
            if let Some(handle) = initramfs_handle {
                let (base, _key) = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
                let base_bytes: &[u8] = base.as_ref();
                let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
                let uncompressed_size = base_bytes.len() + suffix.len();

                // Compress before computing memory so the formula uses
                // actual compressed size.
                let initrd_data = initramfs::lz4_compress_combined(base_bytes, &suffix);
                let total_size = initrd_data.len() as u64;

                let kernel_init_size = read_kernel_init_size(&self.kernel).unwrap_or(0);
                let budget = MemoryBudget {
                    uncompressed_initramfs_bytes: uncompressed_size as u64,
                    compressed_initrd_bytes: total_size,
                    kernel_init_size,
                    shm_bytes: self.shm_size,
                };
                let memory_mb = initramfs_min_memory_mb(&budget).max(self.memory_min_mb);

                vm.allocate_and_register_memory(memory_mb)
                    .with_context(|| {
                        format!("allocate deferred memory ({memory_mb}MB, aarch64)")
                    })?;

                // Load kernel.
                let kr = boot::load_kernel(&vm.guest_mem, &self.kernel)
                    .context("load kernel (aarch64)")?;
                let load_addr = aarch64_initrd_addr(memory_mb, self.shm_size, total_size);
                initramfs::load_initramfs_parts(&vm.guest_mem, &[&initrd_data], load_addr)?;

                // Early-return into finish_aarch64_setup so the
                // deferred path shares the cmdline / FDT assembly
                // with the non-deferred branch below. The shared
                // helper takes the kernel_result + initrd metadata
                // we just produced and writes the cmdline, FDT, and
                // SHM ring on the same code path the non-deferred
                // case takes after its `let kernel_result = if ...`
                // bind resolves.
                return self.finish_aarch64_setup(vm, kr, Some(load_addr), Some(total_size as u32));
            } else {
                let memory_mb = 256u32;
                vm.allocate_and_register_memory(memory_mb)
                    .context("allocate deferred memory (no initramfs, aarch64)")?;
                let kr = boot::load_kernel(&vm.guest_mem, &self.kernel)
                    .context("load kernel (aarch64)")?;
                return self.finish_aarch64_setup(vm, kr, None, None);
            }
        };

        // Non-deferred path: memory already allocated, kernel already loaded.
        let (initrd_addr, initrd_size) = match initramfs_handle {
            Some(handle) => {
                // `self.memory_mb` is required on the non-deferred
                // path: deferred boots take the early-return branches
                // above, so we only reach this site after the builder
                // accepted a concrete `memory_mb`. Surface it as an
                // error rather than `unwrap()` so a future refactor
                // that drops the deferred guard fails loudly with an
                // actionable diagnostic instead of an opaque panic.
                let memory_mb = self
                    .memory_mb
                    .context("internal: non-deferred aarch64 path requires memory_mb to be set")?;
                let (base, _key) = handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("initramfs-resolve thread panicked"))??;
                let base_bytes: &[u8] = base.as_ref();
                let suffix = initramfs::build_suffix(base_bytes.len(), &self.suffix_params())?;
                let initrd_data = initramfs::lz4_compress_combined(base_bytes, &suffix);
                let total_size = initrd_data.len() as u64;
                let load_addr = aarch64_initrd_addr(memory_mb, self.shm_size, total_size);
                initramfs::load_initramfs_parts(&vm.guest_mem, &[&initrd_data], load_addr)?;
                (Some(load_addr), Some(total_size as u32))
            }
            None => (None, None),
        };

        self.finish_aarch64_setup(vm, kernel_result, initrd_addr, initrd_size)
    }

    #[cfg(target_arch = "aarch64")]
    fn finish_aarch64_setup(
        &self,
        vm: &kvm::KtstrKvm,
        kernel_result: boot::KernelLoadResult,
        initrd_addr: Option<u64>,
        initrd_size: Option<u32>,
    ) -> Result<boot::KernelLoadResult> {
        let memory_mb = self.effective_memory_mb(&vm.guest_mem);

        // Kernel cmdline rationale (per flag) — aarch64 subset of the
        // x86_64 block above. Flags present on both arches carry the
        // same justification; see the x86_64 comment for details.
        // aarch64-specific:
        //   kfence.sample_interval=0 — disable KFENCE sampling; no real
        //                              driver faults to catch in the
        //                              test VM, and KFENCE adds boot-time
        //                              page-allocation pressure.
        let mut cmdline = concat!(
            "console=ttyS0 ",
            "nomodules mitigations=off ",
            "random.trust_cpu=on swiotlb=noforce ",
            "panic=-1 iomem=relaxed nokaslr lockdown=none ",
            "sysctl.kernel.unprivileged_bpf_disabled=0 ",
            "sysctl.kernel.sched_schedstats=1 ",
            "delayacct sysctl.kernel.task_delayacct=1 ",
            "kfence.sample_interval=0",
        )
        .to_string();
        // earlycon is always enabled so the kernel has a console from
        // the earliest boot stage. Without it, stdout-path auto-detection
        // is the only path to early output — and that can fail silently
        // if the FDT node isn't matched by OF_EARLYCON_DECLARE.
        cmdline.push_str(" earlycon=uart,mmio,0x09000000");
        let verbose = std::env::var("KTSTR_VERBOSE")
            .map(|v| v == "1")
            .unwrap_or(false)
            || std::env::var("RUST_BACKTRACE").is_ok_and(|v| v == "1" || v == "full");
        if verbose {
            cmdline.push_str(" loglevel=7");
        } else {
            cmdline.push_str(" loglevel=0");
        }
        if self.init_binary.is_some() {
            cmdline.push_str(" rdinit=/init initramfs_options=size=90%");
        }
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = kvm::DRAM_START + mem_size - self.shm_size;
            cmdline.push_str(&format!(
                " KTSTR_SHM_BASE={:#x} KTSTR_SHM_SIZE={:#x}",
                shm_base, self.shm_size
            ));
        }
        if self.topology.has_memory_only_nodes() {
            cmdline.push_str(" numa_balancing=enable");
        } else {
            cmdline.push_str(" numa_balancing=0");
        }
        if !self.cmdline_extra.is_empty() {
            cmdline.push(' ');
            cmdline.push_str(&self.cmdline_extra);
        }

        let t0 = Instant::now();
        boot::validate_cmdline(&cmdline)?;

        let fdt_addr = aarch64::fdt::fdt_address(memory_mb, self.shm_size);
        let mpidrs =
            aarch64::topology::read_mpidrs(&vm.vcpus).context("read vCPU MPIDRs for FDT")?;
        let hw_cache_level = aarch64::topology::host_cache_levels();
        let guest_l1_unified = aarch64::topology::host_l1_is_unified();
        let dtb = aarch64::fdt::create_fdt(
            &self.topology,
            &mpidrs,
            memory_mb,
            &cmdline,
            initrd_addr,
            initrd_size,
            self.shm_size,
            hw_cache_level,
            guest_l1_unified,
            vm.numa_layout.as_ref().expect(
                "numa_layout is Some by the time FDT creation runs: \
                 memory allocation (whether deferred or not) ran earlier \
                 in this function and set numa_layout via \
                 allocate_and_register_memory in src/vmm/aarch64/kvm.rs",
            ),
            !self.disks.is_empty(),
            self.network.is_some(),
            vm.has_pmu,
        )
        .context("create FDT")?;
        vm.guest_mem
            .write_slice(&dtb, GuestAddress(fdt_addr))
            .context("write FDT to guest memory")?;
        tracing::debug!(
            elapsed_us = t0.elapsed().as_micros(),
            fdt_addr,
            fdt_len = dtb.len(),
            "cmdline_fdt",
        );

        // Initialize SHM ring buffer.
        let t0 = Instant::now();
        if self.shm_size > 0 {
            let mem_size = (memory_mb as u64) << 20;
            let shm_base = kvm::DRAM_START + mem_size - self.shm_size;
            self.init_shm_region(&vm.guest_mem, shm_base)?;
        }
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "shm_ring_init");

        Ok(kernel_result)
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn setup_vcpus_aarch64(&self, vm: &kvm::KtstrKvm, kernel_entry: u64) -> Result<()> {
        let t0 = Instant::now();
        let memory_mb = self.effective_memory_mb(&vm.guest_mem);
        let fdt_addr = aarch64::fdt::fdt_address(memory_mb, self.shm_size);
        boot::setup_regs(&vm.vcpus[0], kernel_entry, fdt_addr)?;
        tracing::debug!(elapsed_us = t0.elapsed().as_micros(), "bsp_setup");
        // APs start powered off via PSCI — no register setup needed.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Filesystem::Raw` disks emit no auto-mount cmdline tokens.
    /// The host has nothing to advertise: no on-disk fs to mount,
    /// the guest sees an unformatted `/dev/vda` and the
    /// `auto_mount_data_disks` short-circuits at the absent
    /// `KTSTR_DISK0_FS` check. Pin the empty-string contract so a
    /// future regression that emits Raw-disk tokens (e.g. for a
    /// "mount as raw block device" feature) surfaces here loudly.
    #[test]
    fn disk_auto_mount_cmdline_tokens_raw_emits_nothing() {
        let disk = disk_config::DiskConfig::default();
        assert_eq!(disk.filesystem, disk_config::Filesystem::Raw);
        assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");
    }

    /// `Filesystem::Btrfs` with no name and no read_only emits the
    /// FS + MOUNT pair only — no RO token. Default mount path is
    /// `/mnt/disk0` (driven by `auto_mount_path()` returning the
    /// disk0 fallback when `name` is `None`). The leading space
    /// is the cmdline-concatenation contract: callers paste the
    /// returned string directly.
    #[test]
    fn disk_auto_mount_cmdline_tokens_btrfs_default() {
        let disk = disk_config::DiskConfig::default().filesystem(disk_config::Filesystem::Btrfs);
        assert_eq!(
            disk_auto_mount_cmdline_tokens(&disk),
            " KTSTR_DISK0_FS=btrfs KTSTR_DISK0_MOUNT=/mnt/disk0",
        );
    }

    /// Named `Filesystem::Btrfs` disk emits the name-driven mount
    /// path `/mnt/<name>` instead of `/mnt/disk0`. Pin the name
    /// → mount-path translation so a future `auto_mount_path`
    /// regression (e.g. dropping the name and reverting to fixed
    /// /mnt/disk0) surfaces here.
    #[test]
    fn disk_auto_mount_cmdline_tokens_btrfs_named() {
        let disk = disk_config::DiskConfig::default()
            .filesystem(disk_config::Filesystem::Btrfs)
            .name("data");
        assert_eq!(
            disk_auto_mount_cmdline_tokens(&disk),
            " KTSTR_DISK0_FS=btrfs KTSTR_DISK0_MOUNT=/mnt/data",
        );
    }

    /// Read-only Btrfs disk emits the RO token in addition to FS
    /// + MOUNT. The guest's `auto_mount_data_disks` checks
    /// `KTSTR_DISK0_RO == "1"` and sets `MS_RDONLY` to avoid the
    /// kernel-side -EROFS path on RW mount of a F_RO bdev.
    #[test]
    fn disk_auto_mount_cmdline_tokens_btrfs_read_only() {
        let disk = disk_config::DiskConfig::default()
            .filesystem(disk_config::Filesystem::Btrfs)
            .read_only();
        assert_eq!(
            disk_auto_mount_cmdline_tokens(&disk),
            " KTSTR_DISK0_FS=btrfs KTSTR_DISK0_MOUNT=/mnt/disk0 KTSTR_DISK0_RO=1",
        );
    }

    /// `no_auto_mount` opt-out suppresses every auto-mount token,
    /// even for a Btrfs disk that would otherwise emit them. The
    /// host-side mkfs still happens (Filesystem::Btrfs drives the
    /// template-cache lifecycle); only the guest auto-mount is
    /// skipped, leaving raw `/dev/vda` access to the test author.
    #[test]
    fn disk_auto_mount_cmdline_tokens_no_auto_mount_suppresses() {
        let disk = disk_config::DiskConfig::default()
            .filesystem(disk_config::Filesystem::Btrfs)
            .no_auto_mount();
        assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");

        // RO + named + no_auto_mount: still empty. The opt-out
        // dominates every other config dimension.
        let disk = disk_config::DiskConfig::default()
            .filesystem(disk_config::Filesystem::Btrfs)
            .name("data")
            .read_only()
            .no_auto_mount();
        assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");
    }

    /// Raw disk + no_auto_mount: still empty. The Raw branch is
    /// the gate; no_auto_mount is only meaningful for non-Raw
    /// filesystems but the function tolerates the redundant
    /// combination.
    #[test]
    fn disk_auto_mount_cmdline_tokens_raw_with_no_auto_mount() {
        let disk = disk_config::DiskConfig::default().no_auto_mount();
        assert_eq!(disk.filesystem, disk_config::Filesystem::Raw);
        assert_eq!(disk_auto_mount_cmdline_tokens(&disk), "");
    }

    /// Pin the leading-space cmdline-concatenation contract. The
    /// returned tokens MUST start with a space when non-empty so
    /// they can be appended directly to the cmdline buffer in
    /// `setup_memory`. A regression that drops the leading space
    /// would create a glued-together token like
    /// `virtio_mmio.device=...KTSTR_DISK0_FS=btrfs` which the
    /// kernel cmdline parser would mis-classify as a single token.
    #[test]
    fn disk_auto_mount_cmdline_tokens_starts_with_space() {
        let disk = disk_config::DiskConfig::default().filesystem(disk_config::Filesystem::Btrfs);
        let s = disk_auto_mount_cmdline_tokens(&disk);
        assert!(
            s.starts_with(' '),
            "non-empty tokens must start with a space for safe \
             cmdline concatenation; got {s:?}",
        );
    }
}
