/// Minimal initramfs (cpio newc format) creation via the `cpio` crate.
/// Packs files into a cpio archive for use as Linux initrd.
/// Includes a shell init script that mounts essential filesystems
/// before running the payload.
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Shell init script that mounts proc/sys/dev/cgroup, runs the payload,
/// captures the exit code via serial, and powers off.
///
/// Payload stdout/stderr and STT_EXIT are directed to /dev/ttyS1 (COM2)
/// to separate them from kernel console output on /dev/ttyS0 (COM1).
const INIT_SCRIPT: &str = r#"#!/bin/sh
export PATH=/bin
busybox mkdir -p /proc /sys /dev /tmp
busybox mount -t proc proc /proc
busybox mount -t sysfs sys /sys
busybox mount -t devtmpfs dev /dev
busybox mknod /dev/ttyS0 c 4 64 2>/dev/null
busybox mknod /dev/ttyS1 c 4 65 2>/dev/null
if ! [ -c /dev/ttyS1 ]; then
    echo "FATAL: /dev/ttyS1 not available" > /dev/ttyS0
    echo "--- ttyS diagnostic dump ---" > /dev/ttyS0
    busybox ls -la /dev/ttyS* > /dev/ttyS0 2>&1
    busybox cat /proc/cmdline > /dev/ttyS0
    busybox cat /proc/consoles > /dev/ttyS0 2>/dev/null
    busybox ls -la /sys/class/tty/ > /dev/ttyS0 2>&1
    echo "--- end diagnostic dump ---" > /dev/ttyS0
    busybox reboot -f
fi
echo "STT_INIT_STARTED" > /dev/ttyS1
busybox mkdir -p /sys/fs/cgroup
busybox mount -t cgroup2 none /sys/fs/cgroup 2>/dev/null
busybox mount -t tmpfs tmpfs /tmp
if [ -x /scheduler ]; then
    SCHED_ARGS=$(busybox cat /sched_args 2>/dev/null)
    /scheduler $SCHED_ARGS >/tmp/sched.log 2>&1 &
    SCHED_PID=$!
    export SCHED_PID
    busybox sleep 1
    if ! busybox kill -0 $SCHED_PID 2>/dev/null; then
        echo "===SCHED_OUTPUT_START===" > /dev/ttyS1
        busybox cat /tmp/sched.log > /dev/ttyS1 2>/dev/null
        echo "===SCHED_OUTPUT_END===" > /dev/ttyS1
        echo "SCHEDULER_DIED" > /dev/ttyS1
        echo "STT_EXIT=1" > /dev/ttyS1
        busybox reboot -f
    fi
fi
# Enable sched_ext_dump tracepoint and stream trace_pipe to COM2.
TRACE_EVENTS=/sys/kernel/tracing/events/sched_ext/sched_ext_dump/enable
if [ -f "$TRACE_EVENTS" ]; then
    echo 1 > "$TRACE_EVENTS"
    busybox cat /sys/kernel/tracing/trace_pipe > /dev/ttyS1 &
fi
# Poll SHM dump request flag. When set to 'D' (written by host monitor),
# trigger SysRq-D for scheduler state dump and clear the flag.
SHM_BASE=
for _w in $(busybox cat /proc/cmdline); do
    case $_w in STT_SHM_BASE=*) SHM_BASE=${_w#STT_SHM_BASE=} ;; esac
done
if [ -n "$SHM_BASE" ]; then
    DUMP_ADDR=$(( SHM_BASE + 12 ))
    STALL_ADDR=$(( SHM_BASE + 13 ))
    printf 'D' > /tmp/dump_req
    printf 'S' > /tmp/stall_req
    while true; do
        busybox dd if=/dev/mem bs=1 count=1 skip=$DUMP_ADDR of=/tmp/dump_byte 2>/dev/null
        if busybox cmp -s /tmp/dump_byte /tmp/dump_req; then
            echo D > /proc/sysrq-trigger
            printf '\0' | busybox dd of=/dev/mem bs=1 count=1 seek=$DUMP_ADDR conv=notrunc 2>/dev/null
        fi
        busybox dd if=/dev/mem bs=1 count=1 skip=$STALL_ADDR of=/tmp/stall_byte 2>/dev/null
        if busybox cmp -s /tmp/stall_byte /tmp/stall_req; then
            busybox touch /tmp/stt_stall
            printf '\0' | busybox dd of=/dev/mem bs=1 count=1 seek=$STALL_ADDR conv=notrunc 2>/dev/null
        fi
        busybox sleep 1
    done &
fi
ARGS=$(busybox cat /args 2>/dev/null)
echo "STT_PAYLOAD_STARTING" > /dev/ttyS1
echo "===STT_JSON_START===" > /dev/ttyS1
/payload $ARGS >/dev/ttyS1 2>&1
RC=$?
echo "===STT_JSON_END===" > /dev/ttyS1
if [ -n "$SCHED_PID" ]; then
    busybox kill $SCHED_PID 2>/dev/null
    busybox wait $SCHED_PID 2>/dev/null
    echo "===SCHED_OUTPUT_START===" > /dev/ttyS1
    busybox cat /tmp/sched.log > /dev/ttyS1 2>/dev/null
    echo "===SCHED_OUTPUT_END===" > /dev/ttyS1
fi
echo "STT_EXIT=$RC" > /dev/ttyS1
busybox reboot -f
"#;

/// Statically-linked busybox binary, embedded at compile time via build.rs.
const BUSYBOX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/busybox"));

/// Resolve shared library dependencies for a dynamically-linked binary.
/// Runs `ldd`, parses output, returns `(guest_path, host_path)` pairs.
/// Skips linux-vdso (kernel-provided). Returns empty vec for static binaries.
pub fn resolve_shared_libs(binary: &Path) -> Result<Vec<(String, PathBuf)>> {
    let output = std::process::Command::new("ldd")
        .arg(binary)
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
            let guest = path.strip_prefix('/').unwrap_or(&path);
            libs.push((guest.to_string(), PathBuf::from(&path)));
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
        "stt-stripped-{}-{:?}-{}",
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

/// Build the base cpio archive: directories, busybox, init script, payload,
/// extra binaries, and shared libraries. Does NOT include /args, trailer,
/// or 512-byte padding. The returned bytes are a valid cpio prefix that
/// `build_suffix` can complete with per-invocation args.
pub fn create_initramfs_base(payload: &Path, extra_binaries: &[(&str, &Path)]) -> Result<Vec<u8>> {
    let binary = strip_debug(payload)
        .with_context(|| format!("strip/read binary: {}", payload.display()))?;
    let busybox = BUSYBOX;
    let mut archive = Vec::new();

    // Collect directory entries needed.
    let mut dirs = BTreeSet::new();
    dirs.insert("bin".to_string());

    // Resolve shared library dependencies for payload and extra binaries.
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

    // Core files
    write_entry(&mut archive, "bin/busybox", busybox, 0o100755)?;
    write_entry(&mut archive, "bin/sh", busybox, 0o100755)?;
    write_entry(&mut archive, "init", INIT_SCRIPT.as_bytes(), 0o100755)?;
    write_entry(&mut archive, "payload", &binary, 0o100755)?;

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
    format!("/stt-base-{content_hash:016x}")
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
pub fn shm_store_base(content_hash: u64, data: &[u8]) -> Result<()> {
    let name =
        std::ffi::CString::new(shm_segment_name(content_hash)).context("shm segment name")?;
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o644);
        anyhow::ensure!(fd >= 0, "shm_open: {}", std::io::Error::last_os_error());

        // Exclusive lock — blocks readers and other writers.
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

/// Remove the POSIX shared-memory segment identified by `content_hash`.
#[allow(dead_code)]
pub fn shm_unlink_base(content_hash: u64) {
    if let Ok(name) = std::ffi::CString::new(shm_segment_name(content_hash)) {
        unsafe {
            libc::shm_unlink(name.as_ptr());
        }
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
    fn create_initramfs_has_init_and_payload() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        let s = String::from_utf8_lossy(&initrd);
        assert!(s.contains("init"), "should contain init entry");
        assert!(s.contains("payload"), "should contain payload entry");
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn create_initramfs_init_script_content() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        let s = String::from_utf8_lossy(&initrd);
        assert!(s.contains("mount -t proc"));
        assert!(s.contains("mount -t sysfs"));
        assert!(s.contains("mount -t devtmpfs"));
        assert!(s.contains("STT_EXIT="));
        assert!(s.contains("===STT_JSON_START==="));
        assert!(s.contains("===STT_JSON_END==="));
        assert!(s.contains("===SCHED_OUTPUT_START==="));
        assert!(s.contains("===SCHED_OUTPUT_END==="));
        assert!(s.contains("sched_ext_dump/enable"));
        assert!(s.contains("trace_pipe"));
        assert!(s.contains("sysrq-trigger"));
        assert!(s.contains("STT_SHM_BASE="));
        assert!(s.contains("STT_INIT_STARTED"));
        assert!(s.contains("STT_PAYLOAD_STARTING"));
        assert!(s.contains("ttyS diagnostic dump"));
        assert!(s.contains("/proc/cmdline"));
        assert!(s.contains("/sys/class/tty/"));
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
    fn resolve_shared_libs_static_binary() {
        let busybox_path = std::path::PathBuf::from(env!("OUT_DIR")).join("busybox");
        if busybox_path.exists() {
            let libs = resolve_shared_libs(&busybox_path).unwrap();
            assert!(libs.is_empty(), "static binary should have no shared libs");
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
        assert!(name.starts_with("/stt-base-"));
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
        assert!(n1.starts_with("/stt-base-"));
        assert!(n2.starts_with("/stt-base-"));
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
    fn create_initramfs_base_contains_busybox() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[]).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("bin/busybox"), "base should contain busybox");
        assert!(s.contains("bin/sh"), "base should contain sh symlink");
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
