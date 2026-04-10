/// Minimal initramfs (cpio newc format) creation via the `cpio` crate.
/// Packs the test binary as `/init` along with scheduler binaries and
/// shared libraries into a cpio archive for use as Linux initrd.
/// Init setup is handled by Rust code in `vmm::rust_init`.
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Resolve shared library dependencies for a dynamically-linked binary.
/// Runs `ldd`, parses output, returns `(guest_path, host_path)` pairs.
/// Skips linux-vdso (kernel-provided). Returns empty vec for static binaries.
pub fn resolve_shared_libs(binary: &Path) -> Result<Vec<(String, PathBuf)>> {
    let output = std::process::Command::new("ldd")
        .arg(binary)
        .env_remove("LD_LIBRARY_PATH")
        .output()
        .with_context(|| format!("ldd {}", binary.display()))?;

    if !output.status.success() {
        // ldd exits non-zero for static binaries ("not a dynamic executable")
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut libs = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.contains("linux-vdso") || line.contains("linux-gate") {
            continue;
        }
        if let Some(path) = parse_ldd_line(line) {
            // Canonicalize the host path to resolve symlinks (e.g.,
            // /lib64/libelf.so.1 → /usr/lib64/libelf.so.1 when /lib64
            // is a symlink to usr/lib64). Include both the canonical
            // and ldd-reported paths so the dynamic linker finds libs
            // regardless of whether the guest has matching symlinks.
            let host = PathBuf::from(&path);
            let canonical = std::fs::canonicalize(&host).unwrap_or(host.clone());
            let canon_str = canonical.to_str().unwrap_or(&path);
            let canon_guest = canon_str.strip_prefix('/').unwrap_or(canon_str);
            libs.push((canon_guest.to_string(), canonical.clone()));

            let ldd_guest = path.strip_prefix('/').unwrap_or(&path);
            if ldd_guest != canon_guest {
                libs.push((ldd_guest.to_string(), canonical));
            }
        }
    }

    Ok(libs)
}

/// Parse a single ldd output line into a host path.
fn parse_ldd_line(line: &str) -> Option<String> {
    let line = line.trim();
    if let Some(pos) = line.find("=>") {
        let after = line[pos + 2..].trim();
        let path = after.split_whitespace().next()?;
        if path.starts_with('/') {
            return Some(path.to_string());
        }
    } else if line.starts_with('/') {
        let path = line.split_whitespace().next()?;
        return Some(path.to_string());
    }
    None
}

/// Write one entry (file or directory) into the cpio archive.
fn write_entry(archive: &mut Vec<u8>, name: &str, data: &[u8], mode: u32) -> Result<()> {
    let builder = cpio::newc::Builder::new(name).mode(mode).nlink(1);
    let mut writer = builder.write(archive as &mut dyn Write, data.len() as u32);
    writer
        .write_all(data)
        .with_context(|| format!("write cpio entry '{name}'"))?;
    writer.finish().context("finish cpio entry")?;
    Ok(())
}

/// Strip debug sections from an ELF binary to reduce initramfs size.
/// Debug info can be 10-50x the loadable segment size and is not needed
/// inside the VM. Falls back to the original binary if strip fails.
///
/// When the binary has been deleted (e.g. by `cargo llvm-cov`),
/// retries via `/proc/self/exe` which remains valid as long as the
/// process is alive.
fn strip_debug(path: &Path) -> Result<Vec<u8>> {
    let stripped = std::env::temp_dir().join(format!(
        "ktstr-stripped-{}-{:?}-{}",
        std::process::id(),
        std::thread::current().id(),
        path.file_name().unwrap_or_default().to_string_lossy()
    ));

    // Try strip on the original path first, then /proc/self/exe if the
    // binary was deleted (cargo llvm-cov deletes binaries after
    // instrumenting them).
    let paths_to_try: Vec<&Path> = if is_deleted_self(path) {
        vec![path, Path::new("/proc/self/exe")]
    } else {
        vec![path]
    };

    for src in &paths_to_try {
        let status = std::process::Command::new("strip")
            .args(["--strip-debug", "-o"])
            .arg(&stripped)
            .arg(src)
            .status();
        if let Ok(s) = status
            && s.success()
            && let Ok(data) = std::fs::read(&stripped)
        {
            let _ = std::fs::remove_file(&stripped);
            return Ok(data);
        }
    }

    // strip failed on all paths — fall back to unstripped read.
    for src in &paths_to_try {
        if let Ok(data) = std::fs::read(src) {
            return Ok(data);
        }
    }

    std::fs::read(path).with_context(|| format!("read binary: {}", path.display()))
}

/// Check if `path` is the current executable and has been deleted.
fn is_deleted_self(path: &Path) -> bool {
    let proc_exe = Path::new("/proc/self/exe");
    let Ok(target) = std::fs::read_link(proc_exe) else {
        return false;
    };
    let target_str = target.to_string_lossy();
    target_str.ends_with(" (deleted)")
        && target_str.trim_end_matches(" (deleted)") == path.to_string_lossy().as_ref()
}

/// Build the base cpio archive: /init binary, extra binaries, and shared
/// libraries. Does NOT include /args, trailer, or 512-byte padding. The
/// returned bytes are a valid cpio prefix that `build_suffix` can complete
/// with per-invocation args.
///
/// The test binary is packed as `/init` (the kernel's rdinit entry point).
/// Init setup (mounts, scheduler start, etc.) is handled by the Rust init
/// code in `vmm::rust_init`, which runs when the binary detects PID 1.
pub fn create_initramfs_base(payload: &Path, extra_binaries: &[(&str, &Path)]) -> Result<Vec<u8>> {
    let binary = strip_debug(payload)
        .with_context(|| format!("strip/read binary: {}", payload.display()))?;
    let mut archive = Vec::new();

    // Collect directory entries needed for shared libraries.
    let mut dirs = BTreeSet::new();

    // Resolve shared library dependencies for init binary and extras.
    let mut shared_libs: Vec<(String, PathBuf)> = Vec::new();
    let all_binaries: Vec<&Path> = std::iter::once(payload)
        .chain(extra_binaries.iter().map(|(_, p)| *p))
        .collect();
    for path in &all_binaries {
        let libs = resolve_shared_libs(path)
            .with_context(|| format!("resolve libs for {}", path.display()))?;
        for (guest_path, host_path) in libs {
            if let Some(parent) = Path::new(&guest_path).parent() {
                let mut dir = PathBuf::new();
                for component in parent.components() {
                    dir.push(component);
                    dirs.insert(dir.to_string_lossy().to_string());
                }
            }
            shared_libs.push((guest_path, host_path));
        }
    }
    shared_libs.sort_by(|a, b| a.0.cmp(&b.0));
    shared_libs.dedup_by(|a, b| a.0 == b.0);

    // Directory entries
    for dir in &dirs {
        write_entry(&mut archive, dir, &[], 0o40755)?;
    }

    // Test binary as /init — the Rust init code detects PID 1 and performs
    // all setup (mounts, scheduler, etc.) before running the test function.
    write_entry(&mut archive, "init", &binary, 0o100755)?;

    // Extra binaries (stripped to reduce initramfs size)
    for (name, path) in extra_binaries {
        let data = strip_debug(path)
            .with_context(|| format!("strip/read extra binary '{}': {}", name, path.display()))?;
        write_entry(&mut archive, name, &data, 0o100755)?;
    }

    // Shared libraries
    for (guest_path, host_path) in &shared_libs {
        let data = std::fs::read(host_path).with_context(|| {
            format!("read shared lib '{}': {}", guest_path, host_path.display())
        })?;
        write_entry(&mut archive, guest_path, &data, 0o100755)?;
    }

    Ok(archive)
}

/// Build the suffix that completes a base archive: /args and /sched_args
/// entries, trailer, and 512-byte padding. `base_len` is needed to compute
/// the padding. The returned Vec is typically ~200 bytes.
pub fn build_suffix(base_len: usize, args: &[String], sched_args: &[String]) -> Result<Vec<u8>> {
    build_suffix_full(base_len, args, sched_args, &[], &[])
}

/// Extended suffix builder that also writes /sched_enable and /sched_disable
/// shell scripts for kernel-built schedulers.
pub fn build_suffix_full(
    base_len: usize,
    args: &[String],
    sched_args: &[String],
    sched_enable: &[&str],
    sched_disable: &[&str],
) -> Result<Vec<u8>> {
    let mut suffix = Vec::new();

    // Args file
    let args_data = args.join("\n");
    write_entry(&mut suffix, "args", args_data.as_bytes(), 0o100644)?;

    // Scheduler args file
    if !sched_args.is_empty() {
        let sched_args_data = sched_args.join("\n");
        write_entry(
            &mut suffix,
            "sched_args",
            sched_args_data.as_bytes(),
            0o100644,
        )?;
    }

    // Kernel-built scheduler enable/disable scripts
    if !sched_enable.is_empty() {
        let data = sched_enable.join("\n");
        write_entry(&mut suffix, "sched_enable", data.as_bytes(), 0o100755)?;
    }
    if !sched_disable.is_empty() {
        let data = sched_disable.join("\n");
        write_entry(&mut suffix, "sched_disable", data.as_bytes(), 0o100755)?;
    }

    // Trailer
    cpio::newc::trailer(&mut suffix as &mut dyn Write).context("write cpio trailer")?;

    // Pad to 512-byte boundary (initramfs convention)
    let total = base_len + suffix.len();
    let pad = (512 - (total % 512)) % 512;
    suffix.extend(std::iter::repeat_n(0u8, pad));

    Ok(suffix)
}

/// Create a complete cpio newc archive in one call.
/// Convenience wrapper over `create_initramfs_base` + `build_suffix`.
#[cfg(test)]
pub fn create_initramfs(
    payload: &Path,
    extra_binaries: &[(&str, &Path)],
    args: &[String],
) -> Result<Vec<u8>> {
    let base = create_initramfs_base(payload, extra_binaries)?;
    let suffix = build_suffix(base.len(), args, &[])?;
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    Ok(archive)
}

// ---------------------------------------------------------------------------
// POSIX shared-memory cache for base initramfs
// ---------------------------------------------------------------------------

/// Derive an shm segment name from a content hash. Each distinct
/// combination of payload + scheduler binaries gets its own segment.
pub(crate) fn shm_segment_name(content_hash: u64) -> String {
    format!("/ktstr-base-{content_hash:016x}")
}

/// Read-only mmap of a POSIX shared-memory segment. The mapping stays
/// live until the struct is dropped, so callers can borrow the bytes
/// without copying the entire base archive.
pub struct MappedShm {
    ptr: *const u8,
    len: usize,
}

// SAFETY: The mmap is MAP_SHARED|PROT_READ over a shm segment whose
// contents are immutable after the writer releases flock. The pointer
// and length are valid for the lifetime of the mapping.
unsafe impl Send for MappedShm {}
unsafe impl Sync for MappedShm {}

impl AsRef<[u8]> for MappedShm {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: ptr/len are set by a successful mmap in shm_load_base.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MappedShm {
    fn drop(&mut self) {
        // SAFETY: ptr/len are from mmap; munmap is safe to call once.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// Try to mmap a base initramfs from a POSIX shared-memory segment
/// identified by `content_hash`. Returns a `MappedShm` that borrows
/// the data without copying. Returns `None` on miss or error.
///
/// Acquires a shared flock before mmap to ensure the writer has
/// finished populating the segment. The lock is released after mmap
/// completes -- the mapped pages are stable after that since writers
/// use the same content-hash segment name (idempotent writes).
pub fn shm_load_base(content_hash: u64) -> Option<MappedShm> {
    let name = std::ffi::CString::new(shm_segment_name(content_hash)).ok()?;
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
        if fd < 0 {
            return None;
        }

        // Shared lock -- blocks until any concurrent writer releases LOCK_EX.
        if libc::flock(fd, libc::LOCK_SH) != 0 {
            libc::close(fd);
            return None;
        }

        // Get segment size via fstat.
        let mut stat: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut stat) != 0 || stat.st_size <= 0 {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            return None;
        }
        let len = stat.st_size as usize;

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        );

        // Release lock and fd -- pages are stable.
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);

        if ptr == libc::MAP_FAILED {
            return None;
        }

        Some(MappedShm {
            ptr: ptr as *const u8,
            len,
        })
    }
}

/// Store a base initramfs into the POSIX shared-memory segment
/// identified by `content_hash`. Uses flock(LOCK_EX) for
/// synchronization. The segment name encodes the content hash,
/// so writes are idempotent -- concurrent writers produce
/// identical content, and the last write wins harmlessly.
/// Write `data` to a POSIX SHM segment identified by `name`.
///
/// Creates the segment (O_CREAT | O_RDWR), takes an exclusive flock,
/// truncates to `data.len()`, mmaps, copies, and cleans up.
fn shm_store(name: &std::ffi::CStr, data: &[u8]) -> Result<()> {
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o644);
        anyhow::ensure!(fd >= 0, "shm_open: {}", std::io::Error::last_os_error());

        if libc::flock(fd, libc::LOCK_EX) != 0 {
            libc::close(fd);
            anyhow::bail!("flock: {}", std::io::Error::last_os_error());
        }

        if libc::ftruncate(fd, data.len() as libc::off_t) != 0 {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            anyhow::bail!("ftruncate: {}", std::io::Error::last_os_error());
        }

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        if ptr == libc::MAP_FAILED {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            anyhow::bail!("mmap: {}", std::io::Error::last_os_error());
        }

        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        libc::munmap(ptr, data.len());

        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
    }
    Ok(())
}

pub fn shm_store_base(content_hash: u64, data: &[u8]) -> Result<()> {
    let name =
        std::ffi::CString::new(shm_segment_name(content_hash)).context("shm segment name")?;
    shm_store(&name, data)
}

/// Remove the POSIX shared-memory segment identified by `content_hash`.
#[allow(dead_code)]
pub fn shm_unlink_base(content_hash: u64) {
    if let Ok(name) = std::ffi::CString::new(shm_segment_name(content_hash)) {
        unsafe {
            libc::shm_unlink(name.as_ptr());
        }
    }
}

// ---------------------------------------------------------------------------
// Compressed SHM cache — stores gzip'd base for COW overlay into guest RAM
// ---------------------------------------------------------------------------

/// Segment name for the compressed (gzip) version of a base initramfs.
fn shm_gz_segment_name(content_hash: u64) -> String {
    format!("/ktstr-gz-{content_hash:016x}")
}

/// Open the compressed SHM segment and return a held fd + size.
/// The fd has a shared flock held — caller must close it when done.
/// Returns `None` on miss or error.
pub(crate) fn shm_open_gz(content_hash: u64) -> Option<(std::os::unix::io::RawFd, usize)> {
    let name = std::ffi::CString::new(shm_gz_segment_name(content_hash)).ok()?;
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
        if fd < 0 {
            return None;
        }
        if libc::flock(fd, libc::LOCK_SH) != 0 {
            libc::close(fd);
            return None;
        }
        let mut stat: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut stat) != 0 || stat.st_size <= 0 {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            return None;
        }
        Some((fd, stat.st_size as usize))
    }
}

/// Store compressed initramfs data into a gz SHM segment.
pub(crate) fn shm_store_gz(content_hash: u64, data: &[u8]) -> Result<()> {
    let name = std::ffi::CString::new(shm_gz_segment_name(content_hash)).context("shm gz name")?;
    shm_store(&name, data)
}

/// COW-overlay `len` bytes from `shm_fd` at `host_addr` using
/// MAP_PRIVATE | MAP_FIXED. The guest sees the SHM content but
/// writes go to private anonymous pages (copy-on-write).
///
/// Returns `true` on success, `false` on failure (caller should
/// fall back to write_slice).
pub(crate) unsafe fn cow_overlay(
    host_addr: *mut u8,
    len: usize,
    shm_fd: std::os::unix::io::RawFd,
) -> bool {
    // SAFETY: caller guarantees host_addr points into a valid guest
    // memory region of at least `len` bytes and shm_fd is a valid fd.
    let ptr = unsafe {
        libc::mmap(
            host_addr as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_FIXED,
            shm_fd,
            0,
        )
    };
    ptr != libc::MAP_FAILED
}

/// Close a SHM fd and release its shared flock.
pub(crate) fn shm_close_fd(fd: std::os::unix::io::RawFd) {
    unsafe {
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
    }
}

/// Load an initramfs into guest memory at the given address.
/// Returns (address, size) for boot_params.
#[cfg(test)]
pub fn load_initramfs(
    guest_mem: &vm_memory::GuestMemoryMmap,
    initrd_data: &[u8],
    load_addr: u64,
) -> Result<(u64, u32)> {
    use vm_memory::{Bytes, GuestAddress};
    guest_mem
        .write_slice(initrd_data, GuestAddress(load_addr))
        .context("write initramfs to guest memory")?;
    Ok((load_addr, initrd_data.len() as u32))
}

/// Write multiple byte slices sequentially into guest memory as a single
/// contiguous initramfs. Avoids copying parts into a single Vec first.
/// Returns (address, total_size) for boot_params.
pub fn load_initramfs_parts(
    guest_mem: &vm_memory::GuestMemoryMmap,
    parts: &[&[u8]],
    load_addr: u64,
) -> Result<(u64, u32)> {
    use vm_memory::{Bytes, GuestAddress};
    let mut offset = 0u64;
    for part in parts {
        guest_mem
            .write_slice(part, GuestAddress(load_addr + offset))
            .context("write initramfs part to guest memory")?;
        offset += part.len() as u64;
    }
    Ok((load_addr, offset as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpio_header_format() {
        let mut archive = Vec::new();
        write_entry(&mut archive, "test", b"hello", 0o100644).unwrap();
        assert_eq!(&archive[..6], b"070701");
    }

    #[test]
    fn cpio_trailer() {
        let mut archive = Vec::new();
        write_entry(&mut archive, "test", b"data", 0o100755).unwrap();
        cpio::newc::trailer(&mut archive as &mut dyn std::io::Write).unwrap();
        let s = String::from_utf8_lossy(&archive);
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn create_initramfs_has_init() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        let s = String::from_utf8_lossy(&initrd);
        assert!(s.contains("init"), "should contain init entry");
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn create_initramfs_base_is_valid_cpio() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs_base(&exe, &[]).unwrap();
        assert_eq!(&initrd[..6], b"070701");
        // Base is NOT 512-aligned on its own; only base+suffix is.
        let full = create_initramfs(&exe, &[], &[]).unwrap();
        assert!(initrd.len() <= full.len());
    }

    #[test]
    fn create_initramfs_padded() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        assert_eq!(initrd.len() % 512, 0);
    }

    #[test]
    fn initramfs_nonexistent_file() {
        let result = create_initramfs(Path::new("/nonexistent"), &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn initramfs_nonexistent_extra_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = create_initramfs(&exe, &[("bad", Path::new("/nonexistent"))], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn initramfs_with_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let args = vec!["run".into(), "--json".into(), "scenario".into()];
        let initrd = create_initramfs(&exe, &[], &args).unwrap();
        let s = String::from_utf8_lossy(&initrd);
        assert!(s.contains("args"));
    }

    #[test]
    fn initramfs_empty_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        assert_eq!(initrd.len() % 512, 0);
    }

    // -- base + suffix split tests --

    #[test]
    fn suffix_adds_args_and_trailer() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let args = vec!["run".into(), "--json".into()];
        let suffix = build_suffix(base.len(), &args, &[]).unwrap();
        let s = String::from_utf8_lossy(&suffix);
        assert!(s.contains("args"), "suffix should contain args entry");
        assert!(s.contains("TRAILER!!!"), "suffix should contain trailer");
        assert_eq!(
            (base.len() + suffix.len()) % 512,
            0,
            "base+suffix should be 512-byte aligned"
        );
    }

    #[test]
    fn split_matches_monolithic() {
        let exe = crate::resolve_current_exe().unwrap();
        let args = vec!["run".into(), "--json".into(), "scenario".into()];
        let monolithic = create_initramfs(&exe, &[], &args).unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let suffix = build_suffix(base.len(), &args, &[]).unwrap();
        let mut split = Vec::with_capacity(base.len() + suffix.len());
        split.extend_from_slice(&base);
        split.extend_from_slice(&suffix);
        assert_eq!(
            monolithic, split,
            "split path should produce identical output"
        );
    }

    #[test]
    fn suffix_different_args_differ() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let a = build_suffix(base.len(), &["a".into()], &[]).unwrap();
        let b = build_suffix(base.len(), &["b".into()], &[]).unwrap();
        assert_ne!(a, b, "different args should produce different suffixes");
    }

    #[test]
    fn suffix_empty_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let suffix = build_suffix(base.len(), &[], &[]).unwrap();
        assert_eq!((base.len() + suffix.len()) % 512, 0);
        let s = String::from_utf8_lossy(&suffix);
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn load_initramfs_parts_sequential() {
        let part1 = vec![0xAAu8; 4096];
        let part2 = vec![0xBBu8; 512];
        let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[(
            vm_memory::GuestAddress(0),
            16 << 20,
        )])
        .unwrap();
        let (addr, size) = load_initramfs_parts(&mem, &[&part1, &part2], 0x200000).unwrap();
        assert_eq!(addr, 0x200000);
        assert_eq!(size, 4608);
        let mut buf = vec![0u8; 4608];
        use vm_memory::{Bytes, GuestAddress};
        mem.read_slice(&mut buf, GuestAddress(0x200000)).unwrap();
        assert_eq!(&buf[..4096], &part1[..]);
        assert_eq!(&buf[4096..], &part2[..]);
    }

    // -- ldd / shared lib resolution tests --

    #[test]
    fn parse_ldd_line_arrow_format() {
        let line = "  libelf.so.1 => /lib64/libelf.so.1 (0x00007f...)";
        assert_eq!(parse_ldd_line(line), Some("/lib64/libelf.so.1".into()));
    }

    #[test]
    fn parse_ldd_line_linker() {
        let line = "  /lib64/ld-linux-x86-64.so.2 (0x00007f...)";
        assert_eq!(
            parse_ldd_line(line),
            Some("/lib64/ld-linux-x86-64.so.2".into())
        );
    }

    #[test]
    fn parse_ldd_line_vdso() {
        let line = "  linux-vdso.so.1 (0x00007ffc...)";
        assert_eq!(parse_ldd_line(line), None);
    }

    #[test]
    fn parse_ldd_line_not_found() {
        let line = "  libfoo.so.1 => not found";
        assert_eq!(parse_ldd_line(line), None);
    }

    #[test]
    fn resolve_shared_libs_nonexistent_returns_error() {
        let result = resolve_shared_libs(Path::new("/nonexistent/binary"));
        // ldd on a nonexistent binary fails, returning empty or error.
        if let Ok(libs) = result {
            assert!(libs.is_empty());
        }
    }

    #[test]
    fn resolve_shared_libs_dynamic_binary() {
        let sh = Path::new("/bin/sh");
        if sh.exists() {
            let libs = resolve_shared_libs(sh).unwrap();
            if !libs.is_empty() {
                assert!(
                    libs.iter().any(|(g, _)| g.contains("libc")),
                    "dynamic binary should depend on libc: {:?}",
                    libs
                );
                assert!(
                    !libs.iter().any(|(g, _)| g.contains("vdso")),
                    "vdso should be filtered: {:?}",
                    libs
                );
                for (g, _) in &libs {
                    assert!(!g.starts_with('/'), "guest path should be relative: {g}");
                }
            }
        }
    }

    #[test]
    fn suffix_with_sched_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let sched_args = vec!["--enable-borrow".into(), "--llc".into()];
        let suffix = build_suffix(base.len(), &[], &sched_args).unwrap();
        let s = String::from_utf8_lossy(&suffix);
        assert!(
            s.contains("sched_args"),
            "suffix should contain sched_args entry"
        );
        assert!(s.contains("TRAILER!!!"));
        assert_eq!((base.len() + suffix.len()) % 512, 0);
    }

    #[test]
    fn suffix_without_sched_args_omits_entry() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let suffix = build_suffix(base.len(), &[], &[]).unwrap();
        let s = String::from_utf8_lossy(&suffix);
        assert!(
            !s.contains("sched_args"),
            "empty sched_args should not produce entry"
        );
    }

    #[test]
    fn shm_segment_name_format() {
        let name = shm_segment_name(0xDEADBEEF);
        assert!(name.starts_with("/ktstr-base-"));
        assert!(name.contains("deadbeef"));
    }

    #[test]
    fn is_deleted_self_returns_false_for_nonexistent() {
        assert!(!is_deleted_self(Path::new("/nonexistent/binary")));
    }

    #[test]
    fn is_deleted_self_returns_false_for_current() {
        let exe = crate::resolve_current_exe().unwrap();
        // Current binary is not deleted.
        assert!(!is_deleted_self(&exe));
    }

    #[test]
    fn parse_ldd_line_empty() {
        assert!(parse_ldd_line("").is_none());
    }

    #[test]
    fn parse_ldd_line_whitespace_only() {
        assert!(parse_ldd_line("   ").is_none());
    }

    #[test]
    fn shm_store_load_unlink_roundtrip() {
        let hash = 0xABCD_EF01_2345_6789u64;
        let data = vec![0x42u8; 1024];
        shm_store_base(hash, &data).unwrap();
        let loaded = shm_load_base(hash);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().as_ref(), &data[..]);
        shm_unlink_base(hash);
        // After unlink, load should return None.
        assert!(shm_load_base(hash).is_none());
    }

    #[test]
    fn shm_load_nonexistent_returns_none() {
        let hash = 0xFFFF_FFFF_FFFF_FFFFu64;
        shm_unlink_base(hash); // ensure clean
        assert!(shm_load_base(hash).is_none());
    }

    #[test]
    fn shm_store_overwrite_idempotent() {
        let hash = 0x1234_5678_9ABC_DEF0u64;
        let d1 = vec![0x11u8; 64];
        let d2 = vec![0x22u8; 128];
        shm_store_base(hash, &d1).unwrap();
        shm_store_base(hash, &d2).unwrap();
        let loaded = shm_load_base(hash);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().as_ref(), &d2[..]);
        shm_unlink_base(hash);
    }

    #[test]
    fn shm_segment_name_unique_per_hash() {
        let n1 = shm_segment_name(0);
        let n2 = shm_segment_name(1);
        assert_ne!(n1, n2);
        assert!(n1.starts_with("/ktstr-base-"));
        assert!(n2.starts_with("/ktstr-base-"));
    }

    #[test]
    fn shm_unlink_nonexistent_is_noop() {
        // Should not panic.
        shm_unlink_base(0xDEAD_DEAD_DEAD_DEADu64);
    }

    #[test]
    fn mapped_shm_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MappedShm>();
    }

    #[test]
    fn strip_debug_current_exe() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = strip_debug(&exe).unwrap();
        assert!(!data.is_empty());
        // Stripped binary should be an ELF (first 4 bytes = 0x7f ELF).
        assert_eq!(&data[..4], b"\x7fELF");
    }

    #[test]
    fn strip_debug_nonexistent_fails() {
        let result = strip_debug(Path::new("/nonexistent/binary"));
        assert!(result.is_err());
    }

    #[test]
    fn parse_ldd_line_arrow_no_path() {
        // Arrow but no path after it
        let line = "  libfoo.so => ";
        assert!(parse_ldd_line(line).is_none());
    }

    #[test]
    fn create_initramfs_base_contains_init() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("init"), "base should contain init entry");
    }

    #[test]
    fn create_initramfs_base_includes_extra_shared_libs() {
        let exe = crate::resolve_current_exe().unwrap();
        let sched = std::path::PathBuf::from("target/debug/scx-ktstr-sched");
        if !sched.exists() {
            eprintln!("skipping: scx-ktstr-sched not built");
            return;
        }
        let extras: Vec<(&str, &Path)> = vec![("scheduler", sched.as_path())];
        let base = create_initramfs_base(&exe, &extras).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(
            s.contains("lib64/libelf"),
            "initramfs with scx-ktstr-sched extra should contain libelf; \
             resolved libs: {:?}",
            resolve_shared_libs(sched.as_path()).unwrap()
        );
    }

    #[test]
    fn load_initramfs_to_memory() {
        let data = vec![0xAA; 4096];
        let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[(
            vm_memory::GuestAddress(0),
            16 << 20,
        )])
        .unwrap();
        let (addr, size) = load_initramfs(&mem, &data, 0x200000).unwrap();
        assert_eq!(addr, 0x200000);
        assert_eq!(size, 4096);
        let mut buf = vec![0u8; 4096];
        use vm_memory::{Bytes, GuestAddress};
        mem.read_slice(&mut buf, GuestAddress(0x200000)).unwrap();
        assert_eq!(buf, data);
    }
}
