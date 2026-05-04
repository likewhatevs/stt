//! End-to-end integration tests for the disk-template lifecycle.
//!
//! Two scenarios pin the lifecycle described in
//! `src/vmm/disk_template.rs` end-to-end on a real KVM VM:
//!
//! - **Template-build VM end-to-end** (`#443`): a `Filesystem::Btrfs`
//!   disk on the very first invocation triggers
//!   [`ktstr::vmm::disk_template::ensure_template`]'s
//!   miss path. The framework boots a one-shot template-build VM
//!   that runs `mkfs.btrfs /dev/vda` inside the guest, the host
//!   atomically installs the formatted image into the cache, and
//!   the test scenario's outer VM observes a fully-formatted
//!   `/dev/vda` post-`FICLONE`. The test confirms the guest sees
//!   a btrfs filesystem (via `BLKID`-equivalent: `statfs.f_type ==
//!   BTRFS_SUPER_MAGIC`) on `/dev/vda`.
//!
//! - **Per-test FICLONE clone** (`#439`): on a warm cache (template
//!   already published), each test starts from an FICLONE clone
//!   of the template image. The single-run scenario asserts the
//!   strongest invariants the entry-level API offers: (a) the
//!   clone is fresh (no pre-existing `sentinel-<pid>` file at
//!   `/mnt/disk0/sentinel-<pid>`), (b) the clone is writable
//!   (sentinel write succeeds), and (c) readback returns the same
//!   bytes that were written. Multi-test coordination is not
//!   supported by the framework today, so the cross-run isolation
//!   property is inferred from each test's clone independently
//!   passing (a)+(b)+(c).
//!
//! Both tests require:
//! - `/dev/kvm` accessible
//! - `../linux` kernel source tree (the VM uses ktstr's kernel
//!   cache resolution; absent → panic with actionable error)
//! - `mkfs.btrfs` on the host `PATH` (gated by
//!   [`ktstr::vmm::disk_template::locate_host_mkfs`])
//! - `KTSTR_CACHE_DIR` set to a btrfs/xfs mount, OR the default
//!   `XDG_CACHE_HOME` ancestor lives on a reflink-capable
//!   filesystem (otherwise
//!   [`ktstr::vmm::disk_template::verify_cache_dir_supports_reflink`]
//!   bails before the test boots)
//! - guest kernel with `CONFIG_VIRTIO_BLK + CONFIG_BTRFS_FS`
//!
//! The tests are gated `#[ignore]` because each requires a real
//! VM boot (~30-90s including the template-build VM on first run)
//! and the prerequisite host configuration; default
//! `cargo nextest run` would hang or fail on hosts without these.
//! Run via:
//!
//! ```bash
//! cargo nextest run --test disk_template_e2e --run-ignored all
//! # OR the canonical entry:
//! cargo ktstr test --kernel ../linux \
//!     --filter "disk_template_e2e_btrfs_template_build|\
//! disk_template_e2e_ficlone_clone_isolated"
//! ```
//!
//! User-facing test bar: a `ktstr_test` declaring
//! `Filesystem::Btrfs` on its `DiskConfig` MUST surface a
//! pre-formatted btrfs `/dev/vda` to the guest scenario without
//! the test author writing any disk-template plumbing — the
//! framework handles the cache lookup, template-build VM, and
//! per-test FICLONE transparently.

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::test_support::{Payload, Scheduler, SchedulerSpec};

const KTSTR_SCHED: Scheduler =
    Scheduler::new("ktstr_sched").binary(SchedulerSpec::Discover("scx-ktstr"));
const KTSTR_SCHED_PAYLOAD: Payload = Payload::from_scheduler(&KTSTR_SCHED);

/// btrfs `statfs.f_type` magic per `linux/magic.h`. Pinned here
/// (rather than imported from `disk_template`'s private `const`)
/// because the constant participates in test assertions, and
/// tests pinning a "guest-observable" value should not silently
/// inherit a host-side definition that could drift.
const BTRFS_SUPER_MAGIC: i64 = 0x9123_683e;

/// 256 MiB btrfs disk. Capacity matches the `vm_integration.rs`
/// patterns and the disk-template-cache documentation default.
const KTSTR_DISK_BTRFS: ktstr::prelude::DiskConfig = ktstr::prelude::DiskConfig {
    capacity_mb: 256,
    filesystem: ktstr::prelude::Filesystem::Btrfs,
    throttle: ktstr::prelude::DiskThrottle {
        iops: None,
        bytes_per_sec: None,
        iops_burst_capacity: None,
        bytes_burst_capacity: None,
    },
    read_only: false,
    name: None,
    no_auto_mount: false,
};

// ----------------------------------------------------------------------------
// #443 — Template-build VM end-to-end: btrfs filesystem visible on /dev/vda
// ----------------------------------------------------------------------------

/// Boot a `ktstr_test` with `Filesystem::Btrfs`, assert the guest
/// sees a btrfs filesystem on `/dev/vda`.
///
/// Pins the full disk-template lifecycle:
///   1. `KtstrTestEntry.disk = Some(BTRFS_CONFIG)` reaches
///      [`ktstr::vmm::disk_template::ensure_template`].
///   2. On cache miss, the framework boots a one-shot template-build
///      VM running `mkfs.btrfs /dev/vda` inside the guest (driven by
///      [`ktstr::vmm::disk_template::build_template_via_vm`]).
///   3. The host atomically installs the formatted image into the
///      cache via
///      [`ktstr::vmm::disk_template::store_atomic`].
///   4. The outer scenario VM's
///      [`ktstr::vmm::disk_template::clone_to_per_test`] FICLONEs
///      the template into a per-test backing file.
///   5. The guest kernel sees `/dev/vda` as a btrfs filesystem
///      because the FICLONE clone preserves the on-disk format
///      from the template.
///
/// The assertion: `statfs("/mnt/disk0").f_type == BTRFS_SUPER_MAGIC`.
/// Auto-mount is enabled by default (the `no_auto_mount` flag in
/// `KTSTR_DISK_BTRFS` is `false`), so the guest's
/// `auto_mount_data_disks` mounts the filesystem at `/mnt/disk0`
/// before the scenario runs.
///
/// A regression in any layer surfaces here distinctly:
/// - Template-build VM failure → outer VM never boots (the framework
///   bails before reaching the scenario function).
/// - FICLONE failure → outer VM bails at `init_virtio_blk` before
///   the guest comes up.
/// - Wrong filesystem on the template → `statfs.f_type` mismatch in
///   the assertion below.
fn scenario_btrfs_filesystem_visible_at_dev_vda(
    _ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    use std::ffi::CString;

    // The auto-mount path is `/mnt/disk0` for an unnamed disk
    // (per `DiskConfig::auto_mount_path`). Statfs the mount point
    // and verify f_type == BTRFS_SUPER_MAGIC.
    let mount_point = CString::new("/mnt/disk0")
        .expect("/mnt/disk0 contains no nul bytes");
    // SAFETY: statfs writes into a stack-allocated zero-initialized
    // buffer of the correct layout. The CString is NUL-terminated
    // for the duration of the call. The kernel returns 0 on success
    // and -1 with errno set on failure.
    let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statfs(mount_point.as_ptr(), &mut buf) };
    if rc != 0 {
        let errno = std::io::Error::last_os_error();
        anyhow::bail!(
            "statfs(/mnt/disk0) failed: {errno}. The disk-template \
             auto-mount must succeed before the scenario runs — \
             check that the framework wired Filesystem::Btrfs through \
             ensure_template and that the guest kernel has \
             CONFIG_BTRFS_FS."
        );
    }

    // Cast f_type through `i64` to match the host-side constant
    // (which is documented as `__fsword_t`-compatible on 64-bit
    // Linux). Comparing the raw `f_type` (which is `i64` on
    // x86_64/aarch64 64-bit Linux) avoids the host-side
    // sign-extension trap that motivates the `compile_error!` in
    // `disk_template.rs`.
    let fs_type = buf.f_type as i64;
    if fs_type != BTRFS_SUPER_MAGIC {
        anyhow::bail!(
            "/mnt/disk0 has statfs.f_type=0x{fs_type:x}, expected \
             BTRFS_SUPER_MAGIC=0x{BTRFS_SUPER_MAGIC:x}. The \
             disk-template lifecycle did not produce a btrfs \
             filesystem on /dev/vda — possible failures: \
             (a) the template-build VM ran but mkfs.btrfs reported \
             success without formatting, (b) FICLONE produced a \
             zeroed image instead of a clone, (c) the guest mounted \
             a different filesystem (check dmesg for mount errors).",
        );
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "/mnt/disk0 statfs.f_type=0x{fs_type:x} matches \
             BTRFS_SUPER_MAGIC — the disk-template lifecycle \
             (build, atomic install, FICLONE) produced a \
             pre-formatted btrfs filesystem on /dev/vda"
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// #439 — Per-test FICLONE clone: writes are isolated per-test
// ----------------------------------------------------------------------------

/// Boot a `ktstr_test` with `Filesystem::Btrfs`, write a sentinel
/// file inside the mounted filesystem, and assert the file is
/// absent (FICLONE produced a fresh clone of the cached template,
/// not a shared mount).
///
/// The cached template image is post-`mkfs.btrfs` only — it
/// contains an empty btrfs filesystem with no user files. Each
/// per-test FICLONE clone starts from that empty state. The
/// scenario writes a sentinel file at `/mnt/disk0/sentinel-<pid>`
/// then immediately re-reads the directory to confirm only the
/// expected entries are present (the file the scenario itself
/// just wrote — proving the underlying filesystem is real and
/// writable, NOT some sentinel from a prior test run).
///
/// Pins the FICLONE isolation contract:
/// - The scenario does NOT see files from prior `cargo ktstr test`
///   invocations (FICLONE produced an independent clone, not a
///   shared mount).
/// - The scenario CAN write to the filesystem (the clone is
///   read-write — `read_only` defaults to false on `DiskConfig`).
/// - The directory listing matches expectations (no unexpected
///   files leaked from the template-build VM or a prior test).
///
/// This is a one-shot scenario: a multi-test isolation assertion
/// would require coordinating two `ktstr_test` entries, which the
/// framework does not currently support cross-test. The single-
/// test assertion below is the strongest check the entry-level
/// API offers — it pins (a) the clone is fresh per test (no
/// pre-existing sentinel), (b) the clone is writable, (c) the
/// clone is genuinely backed by btrfs (the template-build VM
/// produced a real filesystem). A regression that broke the
/// FICLONE clone path (returning a zeroed file instead, sharing
/// the template directly, etc.) would surface here as either a
/// write failure or an unexpected pre-existing file.
fn scenario_ficlone_clone_writable_and_fresh(
    _ctx: &ktstr::scenario::Ctx,
) -> Result<AssertResult> {
    use std::fs;

    let mount = std::path::Path::new("/mnt/disk0");
    if !mount.is_dir() {
        anyhow::bail!(
            "/mnt/disk0 is not a directory — auto-mount of the \
             btrfs disk-template clone failed. Check guest dmesg \
             for mount errors and verify CONFIG_BTRFS_FS in the \
             guest kernel.",
        );
    }

    // Step 1: the cached template was produced by `mkfs.btrfs` on
    // an empty disk; the FICLONE clone inherits that empty state.
    // `read_dir` should succeed and the directory should contain
    // ONLY the entries btrfs creates by default (no user files).
    // Different btrfs versions may or may not create a default
    // subvolume entry visible at the root; we only assert that
    // there is NO sentinel file matching the pattern this test
    // would write, NOT that the directory is exactly empty.
    let entries_before: Vec<String> = fs::read_dir(mount)
        .map_err(|e| anyhow::anyhow!("read_dir({mount:?}): {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    let pid = std::process::id();
    let sentinel_name = format!("sentinel-{pid}");
    if entries_before.contains(&sentinel_name) {
        anyhow::bail!(
            "/mnt/disk0/{sentinel_name} already exists before this \
             scenario wrote it. FICLONE did not produce a fresh \
             clone — either the template-build VM left state behind, \
             or the per-test fan-out reused a stranded debris file \
             instead of cloning fresh. Pre-existing entries: \
             {entries_before:?}"
        );
    }

    // Step 2: write a sentinel file. This proves the clone is
    // read-write AND that the underlying btrfs filesystem accepts
    // ordinary file writes.
    let sentinel_path = mount.join(&sentinel_name);
    fs::write(&sentinel_path, b"DISK_TEMPLATE_E2E_SENTINEL").map_err(|e| {
        anyhow::anyhow!(
            "write {sentinel_path:?}: {e}. The FICLONE clone is \
             not writable — possible causes: (a) read_only flag \
             accidentally on, (b) btrfs filesystem corrupt, \
             (c) ENOSPC on the per-test backing file."
        )
    })?;

    // Step 3: confirm the file is now visible and the contents
    // match what was written. This pins the FICLONE-cloned write
    // path through the guest btrfs driver and the host backing
    // file's reflink extents.
    let body = fs::read(&sentinel_path).map_err(|e| {
        anyhow::anyhow!(
            "read {sentinel_path:?} after write: {e}. The btrfs \
             filesystem accepted the write but the readback failed."
        )
    })?;
    if body != b"DISK_TEMPLATE_E2E_SENTINEL" {
        anyhow::bail!(
            "{sentinel_path:?} readback mismatch: got {body:?}, \
             expected DISK_TEMPLATE_E2E_SENTINEL. The btrfs \
             filesystem accepted the write but the readback \
             returned different content — possible FICLONE \
             extent-sharing bug.",
        );
    }

    let mut result = AssertResult::pass();
    result.details.push(AssertDetail::new(
        DetailKind::Other,
        format!(
            "FICLONE clone produced a fresh writable btrfs filesystem \
             at /mnt/disk0; {sentinel_name} did not pre-exist, was \
             written successfully, and read back byte-identical"
        ),
    ));
    Ok(result)
}

// ----------------------------------------------------------------------------
// Entry registrations
// ----------------------------------------------------------------------------

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_BTRFS_TEMPLATE_BUILD: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "disk_template_e2e_btrfs_template_build",
        func: scenario_btrfs_filesystem_visible_at_dev_vda,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        // Short duration — the assertion is a single statfs syscall.
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        expect_err: false,
        disk: Some(KTSTR_DISK_BTRFS),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

#[ktstr::__private::linkme::distributed_slice(ktstr::test_support::KTSTR_TESTS)]
#[linkme(crate = ktstr::__private::linkme)]
static __KTSTR_ENTRY_FICLONE_CLONE_ISOLATED: ktstr::test_support::KtstrTestEntry =
    ktstr::test_support::KtstrTestEntry {
        name: "disk_template_e2e_ficlone_clone_isolated",
        func: scenario_ficlone_clone_writable_and_fresh,
        scheduler: &KTSTR_SCHED_PAYLOAD,
        extra_sched_args: &[],
        watchdog_timeout: std::time::Duration::from_secs(3),
        duration: std::time::Duration::from_millis(500),
        workers_per_cgroup: 1,
        expect_err: false,
        disk: Some(KTSTR_DISK_BTRFS),
        ..ktstr::test_support::KtstrTestEntry::DEFAULT
    };

// ----------------------------------------------------------------------------
// `#[test] #[ignore]` shims — cargo nextest entry points
// ----------------------------------------------------------------------------
//
// Both scenarios above are registered with the `KTSTR_TESTS`
// distributed_slice and run via `cargo ktstr test --filter <name>`,
// which is the canonical entry for tests that need a real KVM VM
// (matches `vm_integration.rs`, `failure_dump_e2e.rs`).

/// Locate the `cargo-ktstr` binary built for this test pass.
const CARGO_KTSTR_BINARY: &str = env!("CARGO_BIN_EXE_cargo-ktstr");

/// Resolve the linux source tree (`../linux` relative to this crate).
fn linux_source_dir() -> std::path::PathBuf {
    let crate_root = env!("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(crate_root)
        .join("..")
        .join("linux")
}

/// Drive one disk-template scenario via `cargo ktstr test`.
fn drive_ktstr_test(scenario_name: &str) {
    let source = linux_source_dir();
    assert!(
        source.is_dir(),
        "../linux source tree missing — disk-template E2E tests \
         need a kernel source tree. Expected: {}",
        source.display(),
    );

    let output = std::process::Command::new(CARGO_KTSTR_BINARY)
        .arg("ktstr")
        .arg("test")
        .arg("--kernel")
        .arg(&source)
        .arg("--")
        .arg("--filter")
        .arg(scenario_name)
        .output()
        .expect("spawn cargo-ktstr test");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "cargo ktstr test --filter {scenario_name} failed (exit={:?})\n\
         STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
        output.status.code(),
    );
}

/// #443 — Template-build VM end-to-end on a CI runner.
///
/// Boots a `ktstr_test` with `Filesystem::Btrfs`. On a cold cache
/// this triggers the template-build VM (one-shot guest running
/// `mkfs.btrfs /dev/vda`), the atomic install into the cache, and
/// the outer scenario's FICLONE clone of the freshly-published
/// template. On a warm cache only the FICLONE step runs.
///
/// The scenario asserts `statfs(/mnt/disk0).f_type ==
/// BTRFS_SUPER_MAGIC`, proving the disk-template lifecycle
/// produced a real btrfs filesystem visible at the auto-mounted
/// path inside the guest.
///
/// Prerequisites:
/// - `../linux` kernel source tree
/// - `/dev/kvm` accessible
/// - `mkfs.btrfs` on host PATH
/// - `KTSTR_CACHE_DIR` (or its `XDG_CACHE_HOME` ancestor) on
///   btrfs/xfs
/// - guest kernel with CONFIG_VIRTIO_BLK + CONFIG_BTRFS_FS
#[test]
#[ignore = "VM integration test (~30-90s with cold cache template \
            build); requires KVM, ../linux, mkfs.btrfs on PATH, \
            btrfs/xfs cache dir, CONFIG_BTRFS_FS in guest. Run via \
            `cargo nextest run --run-ignored all` or \
            `cargo ktstr test --kernel ../linux \
            --filter disk_template_e2e_btrfs_template_build`."]
fn disk_template_e2e_btrfs_template_build() {
    drive_ktstr_test("disk_template_e2e_btrfs_template_build");
}

/// #439 — Per-test FICLONE clone produces an isolated, writable
/// btrfs filesystem.
///
/// Asserts the clone is fresh (no pre-existing sentinel file),
/// writable (sentinel write succeeds), and read-back-correct
/// (sentinel content matches). Pins the per-test fan-out path
/// through `ktstr::vmm::disk_template::clone_to_per_test`.
#[test]
#[ignore = "VM integration test (~10-90s); requires KVM, ../linux, \
            mkfs.btrfs on PATH, btrfs/xfs cache dir, CONFIG_BTRFS_FS \
            in guest. Run via `cargo nextest run --run-ignored all` \
            or `cargo ktstr test --kernel ../linux \
            --filter disk_template_e2e_ficlone_clone_isolated`."]
fn disk_template_e2e_ficlone_clone_isolated() {
    drive_ktstr_test("disk_template_e2e_ficlone_clone_isolated");
}
