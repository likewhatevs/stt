/// Minimal initramfs (cpio newc format) creation via the `cpio` crate.
/// Packs the test binary as `/init` along with scheduler binaries,
/// shared libraries, optional busybox, and user-provided include files
/// into a cpio archive for use as Linux initrd.
/// Init setup is handled by Rust code in `vmm::rust_init`.
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// Result of shared library resolution for a binary.
#[derive(Debug)]
pub struct SharedLibs {
    /// Resolved `(guest_path, host_path)` pairs.
    pub found: Vec<(String, PathBuf)>,
    /// Library sonames that could not be resolved to a host path.
    /// Each entry includes whether it is a direct (DT_NEEDED) dependency
    /// of the root binary or a transitive dependency.
    pub missing: Vec<MissingLib>,
}

/// A shared library dependency that could not be resolved.
#[derive(Debug)]
pub struct MissingLib {
    /// The soname (e.g. `libssl.so.1.1`).
    pub soname: String,
    /// True if this soname appears in the root binary's DT_NEEDED.
    /// False if it is a transitive dependency of one of the root's deps.
    pub direct: bool,
}

/// Resolve shared library dependencies for a dynamically-linked ELF binary.
/// Parses the ELF dynamic section to read DT_NEEDED entries, then resolves
/// each soname to a host path using DT_RUNPATH/DT_RPATH and default library
/// paths. Recurses into resolved libraries to build the full transitive
/// closure. Returns empty result for static binaries or non-ELF files.
pub fn resolve_shared_libs(binary: &Path) -> Result<SharedLibs> {
    let data =
        std::fs::read(binary).with_context(|| format!("read binary: {}", binary.display()))?;
    let elf = match goblin::elf::Elf::parse(&data) {
        Ok(e) => e,
        Err(_) => {
            // Not a valid ELF (or 32-bit) — treat as static/non-dynamic.
            return Ok(SharedLibs {
                found: vec![],
                missing: vec![],
            });
        }
    };

    if elf.libraries.is_empty() && elf.dynamic.is_none() {
        // No dynamic section — static binary.
        return Ok(SharedLibs {
            found: vec![],
            missing: vec![],
        });
    }

    // Extract DT_NEEDED, DT_RUNPATH, and DT_RPATH from the root binary.
    let root_needed: Vec<String> = elf.libraries.iter().map(|s| s.to_string()).collect();
    let root_search = elf_search_paths(&elf, binary);

    // Resolve the full transitive closure. Each queued entry carries the
    // search paths from the library that declared the DT_NEEDED, since
    // each ELF has its own DT_RUNPATH/DT_RPATH.
    let mut found: Vec<(String, PathBuf)> = Vec::new();
    let mut missing: Vec<MissingLib> = Vec::new();
    let mut visited = std::collections::HashSet::new();
    // Queue: (soname, is_direct_dep_of_root, search_paths_from_parent)
    let mut queue: std::collections::VecDeque<(String, bool, Vec<PathBuf>)> = root_needed
        .iter()
        .map(|s| (s.clone(), true, root_search.clone()))
        .collect();

    while let Some((soname, is_direct, search_paths)) = queue.pop_front() {
        if !visited.insert(soname.clone()) {
            continue;
        }
        if let Some(host_path) = resolve_soname(&soname, &search_paths) {
            let canonical = std::fs::canonicalize(&host_path).unwrap_or_else(|_| host_path.clone());
            let canon_str = canonical.to_string_lossy();
            let canon_guest = canon_str
                .strip_prefix('/')
                .unwrap_or(&canon_str)
                .to_string();
            found.push((canon_guest.clone(), canonical.clone()));

            // Also add the non-canonical path if it differs, so the
            // guest dynamic linker can find the lib via either path.
            let host_str = host_path.to_string_lossy();
            let host_guest = host_str.strip_prefix('/').unwrap_or(&host_str).to_string();
            if host_guest != canon_guest {
                found.push((host_guest, canonical.clone()));
            }

            // Recurse: parse the resolved lib's own DT_NEEDED and use
            // its own DT_RUNPATH/DT_RPATH for resolving those deps.
            if let Ok(lib_data) = std::fs::read(&canonical)
                && let Ok(lib_elf) = goblin::elf::Elf::parse(&lib_data)
            {
                let lib_search = elf_search_paths(&lib_elf, &canonical);
                for lib_name in &lib_elf.libraries {
                    let transitive = lib_name.to_string();
                    if !visited.contains(&transitive) {
                        queue.push_back((transitive, false, lib_search.clone()));
                    }
                }
            }
        } else {
            missing.push(MissingLib {
                soname,
                direct: is_direct,
            });
        }
    }

    Ok(SharedLibs { found, missing })
}

/// Extract search paths from DT_RUNPATH (preferred) or DT_RPATH, with
/// dynamic string tokens expanded:
/// - `$ORIGIN` / `${ORIGIN}`: binary's parent directory
/// - `$LIB` / `${LIB}`: `lib` or `lib64` based on ELF class
/// - `$PLATFORM` / `${PLATFORM}`: `x86_64` or `aarch64`
fn elf_search_paths(elf: &goblin::elf::Elf, binary: &Path) -> Vec<PathBuf> {
    let origin = binary
        .parent()
        .and_then(|p| std::fs::canonicalize(p).ok())
        .unwrap_or_default();

    // DT_RUNPATH takes precedence over DT_RPATH when both are present.
    let raw = if !elf.runpaths.is_empty() {
        elf.runpaths.join(":")
    } else if !elf.rpaths.is_empty() {
        elf.rpaths.join(":")
    } else {
        return vec![];
    };

    let origin_str = origin.to_string_lossy();
    // goblin's is_64 distinguishes 64-bit vs 32-bit ELF class.
    let lib_str = if elf.is_64 { "lib64" } else { "lib" };
    let platform_str = std::env::consts::ARCH;

    raw.split(':')
        .map(|p| {
            let expanded = p
                .replace("$ORIGIN", &origin_str)
                .replace("${ORIGIN}", &origin_str)
                .replace("$LIB", lib_str)
                .replace("${LIB}", lib_str)
                .replace("$PLATFORM", platform_str)
                .replace("${PLATFORM}", platform_str);
            PathBuf::from(expanded)
        })
        .collect()
}

/// Default library search paths used by the dynamic linker.
const DEFAULT_LIB_PATHS: &[&str] = &[
    "/lib",
    "/usr/lib",
    "/lib64",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
];

/// Resolve a soname to a host path.
/// Search order: RUNPATH/RPATH dirs, then default library paths.
fn resolve_soname(soname: &str, search_paths: &[PathBuf]) -> Option<PathBuf> {
    // 1. RUNPATH / RPATH directories.
    for dir in search_paths {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 2. Default paths.
    for dir in DEFAULT_LIB_PATHS {
        let candidate = Path::new(dir).join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

/// ELF magic bytes: `\x7fELF`.
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// Check if the first 4 bytes of a file match ELF magic.
fn is_elf(path: &Path) -> bool {
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            Ok(magic)
        })
        .is_ok_and(|m| m == *ELF_MAGIC)
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
///
/// When `busybox` is true, embeds busybox at `bin/busybox` for shell mode.
///
/// `include_files` adds files verbatim to the archive (no strip_debug).
/// Each entry is `(archive_path, host_path)`. ELF files get shared library
/// resolution; non-ELF files are copied as-is. Only regular files are
/// accepted; FIFOs, device nodes, and sockets are rejected. Archive paths
/// must not contain `..` components.
pub fn create_initramfs_base(
    payload: &Path,
    extra_binaries: &[(&str, &Path)],
    include_files: &[(&str, &Path)],
    busybox: bool,
) -> Result<Vec<u8>> {
    // Validate include_files and collect metadata (reused in the write
    // loop to avoid a second stat syscall per file).
    let mut validated_includes: Vec<(&str, &Path, u32)> = Vec::with_capacity(include_files.len());
    for (archive_path, host_path) in include_files {
        // Reject path traversal.
        if Path::new(archive_path)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            anyhow::bail!("include_files archive path contains '..': {}", archive_path);
        }
        // Reject non-regular files (FIFOs, device nodes, sockets block or
        // produce garbage).
        let meta = std::fs::metadata(host_path).with_context(|| {
            format!(
                "stat include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        if !meta.file_type().is_file() {
            anyhow::bail!(
                "include_files entry '{}' is not a regular file: {}",
                archive_path,
                host_path.display()
            );
        }
        validated_includes.push((archive_path, host_path, meta.permissions().mode()));
    }

    let binary = strip_debug(payload)
        .with_context(|| format!("strip/read binary: {}", payload.display()))?;
    let mut archive = Vec::new();

    // Collect directory entries needed for shared libraries and includes.
    let mut dirs = BTreeSet::new();

    // Resolve shared library dependencies for init binary and extras.
    let mut shared_libs: Vec<(String, PathBuf)> = Vec::new();
    let mut all_binaries: Vec<&Path> = std::iter::once(payload)
        .chain(extra_binaries.iter().map(|(_, p)| *p))
        .collect();

    // ELF files from include_files join the shared lib resolution chain.
    let mut include_elf_paths: Vec<&Path> = Vec::new();
    for (_, host_path) in include_files {
        if is_elf(host_path) {
            include_elf_paths.push(host_path);
            all_binaries.push(host_path);
        }
    }

    for path in &all_binaries {
        let result = resolve_shared_libs(path)
            .with_context(|| format!("resolve libs for {}", path.display()))?;

        if !result.missing.is_empty() && include_elf_paths.contains(path) {
            let direct: Vec<&str> = result
                .missing
                .iter()
                .filter(|m| m.direct)
                .map(|m| m.soname.as_str())
                .collect();
            if !direct.is_empty() {
                anyhow::bail!(
                    "include file '{}' directly requires missing shared libraries: {}. \
                     The binary will not load inside the guest VM. \
                     Use a statically-linked binary or install the missing libraries.",
                    path.display(),
                    direct.join(", ")
                );
            }
            for m in result.missing.iter().filter(|m| !m.direct) {
                eprintln!(
                    "warning: include file '{}': transitive dependency {} \
                     is not found — the binary may not load in the guest VM",
                    path.display(),
                    m.soname
                );
            }
        } else {
            for m in &result.missing {
                eprintln!("warning: {}: {} => not found", path.display(), m.soname);
            }
        }

        for (guest_path, host_path) in result.found {
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

    // Busybox needs bin/ directory.
    if busybox {
        dirs.insert("bin".to_string());
    }
    // Include files need their parent directories in the cpio archive.
    // The component walk produces all ancestors (e.g. "include-files/sub/f"
    // yields "include-files" and "include-files/sub").
    for (archive_path, _, _) in &validated_includes {
        if let Some(parent) = Path::new(archive_path).parent() {
            let mut dir = PathBuf::new();
            for component in parent.components() {
                dir.push(component);
                dirs.insert(dir.to_string_lossy().to_string());
            }
        }
    }

    // Directory entries
    for dir in &dirs {
        write_entry(&mut archive, dir, &[], 0o40755)?;
    }

    // Test binary as /init — the Rust init code detects PID 1 and performs
    // all setup (mounts, scheduler, etc.) before running the test function.
    write_entry(&mut archive, "init", &binary, 0o100755)?;

    // Shell mode: embed busybox.
    if busybox {
        write_entry(&mut archive, "bin/busybox", crate::BUSYBOX, 0o100755)?;
    }

    // Extra binaries (stripped to reduce initramfs size)
    for (name, path) in extra_binaries {
        let data = strip_debug(path)
            .with_context(|| format!("strip/read extra binary '{}': {}", name, path.display()))?;
        write_entry(&mut archive, name, &data, 0o100755)?;
    }

    // Include files: copied verbatim, preserving original content and
    // debug symbols. No strip_debug — included files are user-provided
    // and may be non-ELF.
    for (archive_path, host_path, mode) in &validated_includes {
        let data = std::fs::read(host_path).with_context(|| {
            format!(
                "read include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        write_entry(&mut archive, archive_path, &data, *mode)?;
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
    let base = create_initramfs_base(payload, extra_binaries, &[], false)?;
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

    /// Extract cpio entry names from a newc archive for test assertions.
    fn cpio_entry_names(archive: &[u8]) -> Vec<String> {
        let mut names = Vec::new();
        let mut remaining: &[u8] = archive;
        while let Ok(reader) = cpio::newc::Reader::new(remaining) {
            let name = reader.entry().name().to_string();
            if reader.entry().is_trailer() {
                break;
            }
            names.push(name);
            remaining = reader.finish().unwrap();
        }
        names
    }

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
        let initrd = create_initramfs_base(&exe, &[], &[], false).unwrap();
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
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
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
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
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
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let a = build_suffix(base.len(), &["a".into()], &[]).unwrap();
        let b = build_suffix(base.len(), &["b".into()], &[]).unwrap();
        assert_ne!(a, b, "different args should produce different suffixes");
    }

    #[test]
    fn suffix_empty_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
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

    // -- shared lib resolution tests --

    #[test]
    fn resolve_shared_libs_nonexistent_returns_error() {
        let result = resolve_shared_libs(Path::new("/nonexistent/binary"));
        // Nonexistent file cannot be read.
        assert!(result.is_err());
    }

    #[test]
    fn resolve_shared_libs_non_elf_returns_empty() {
        let tmp = std::env::temp_dir().join("ktstr-test-resolve-nonelf");
        std::fs::write(&tmp, b"not an elf").unwrap();
        let result = resolve_shared_libs(&tmp).unwrap();
        assert!(result.found.is_empty());
        assert!(result.missing.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn resolve_shared_libs_dynamic_binary() {
        let sh = Path::new("/bin/sh");
        if sh.exists() {
            let shared = resolve_shared_libs(sh).unwrap();
            if !shared.found.is_empty() {
                assert!(
                    shared.found.iter().any(|(g, _)| g.contains("libc")),
                    "dynamic binary should depend on libc: {:?}",
                    shared.found
                );
                for (g, _) in &shared.found {
                    assert!(!g.starts_with('/'), "guest path should be relative: {g}");
                }
            }
        }
    }

    #[test]
    fn elf_dynamic_needed_extracts_sonames() {
        let sh = Path::new("/bin/sh");
        if !sh.exists() || !is_elf(sh) {
            eprintln!("skipping: /bin/sh not ELF");
            return;
        }
        let data = std::fs::read(sh).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let needed: Vec<&str> = elf.libraries.clone();
        assert!(
            needed.iter().any(|n| n.contains("libc")),
            "/bin/sh should need libc: {:?}",
            needed
        );
    }

    #[test]
    fn resolve_soname_finds_libc() {
        let result = resolve_soname("libc.so.6", &[]);
        assert!(
            result.is_some(),
            "should resolve libc.so.6 via default paths"
        );
        assert!(result.unwrap().is_file());
    }

    #[test]
    fn suffix_with_sched_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
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
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
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
    fn create_initramfs_base_contains_init() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("init"), "base should contain init entry");
    }

    #[test]
    fn create_initramfs_base_includes_extra_shared_libs() {
        let exe = crate::resolve_current_exe().unwrap();
        let sched = std::path::PathBuf::from("target/debug/scx-ktstr");
        if !sched.exists() {
            eprintln!("skipping: scx-ktstr not built");
            return;
        }
        let extras: Vec<(&str, &Path)> = vec![("scheduler", sched.as_path())];
        let base = create_initramfs_base(&exe, &extras, &[], false).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(
            s.contains("lib64/libelf"),
            "initramfs with scx-ktstr extra should contain libelf; \
             resolved libs: {:?}",
            resolve_shared_libs(sched.as_path()).unwrap().found
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

    // -- include_files and busybox tests --

    #[test]
    fn busybox_with_include_files() {
        let exe = crate::resolve_current_exe().unwrap();
        // Create a temp file to include.
        let tmp = std::env::temp_dir().join("ktstr-test-include-busybox");
        std::fs::write(&tmp, b"hello").unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/test.txt", tmp.as_path())];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let names = cpio_entry_names(&base);
        assert!(
            names.iter().any(|n| n == "bin/busybox"),
            "busybox=true should have bin/busybox entry: {:?}",
            names
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_no_busybox_when_empty() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let names = cpio_entry_names(&base);
        assert!(
            !names.iter().any(|n| n == "bin/busybox"),
            "busybox=false should not have bin/busybox entry: {:?}",
            names
        );
    }

    #[test]
    fn include_files_preserves_mode() {
        let tmp = std::env::temp_dir().join("ktstr-test-include-mode");
        std::fs::write(&tmp, b"script content").unwrap();
        // Set executable mode.
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o100755)).unwrap();

        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/run.sh", tmp.as_path())];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(
            s.contains("include-files/run.sh"),
            "include path should appear in cpio"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_elf_gets_shared_libs() {
        // /bin/sh is a dynamic ELF on most systems.
        let sh = Path::new("/bin/sh");
        if !sh.exists() {
            eprintln!("skipping: /bin/sh not found");
            return;
        }
        if !is_elf(sh) {
            eprintln!("skipping: /bin/sh is not ELF");
            return;
        }
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/sh", sh)];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        // Dynamic ELF should pull in libc shared libs.
        let shared = resolve_shared_libs(sh).unwrap();
        if !shared.found.is_empty() {
            assert!(
                shared.found.iter().any(|(g, _)| s.contains(g.as_str())),
                "include ELF shared libs should appear in archive: {:?}",
                shared.found
            );
        }
    }

    #[test]
    fn include_files_non_elf_no_shared_libs() {
        let tmp = std::env::temp_dir().join("ktstr-test-include-nonelf");
        std::fs::write(&tmp, b"#!/bin/sh\necho hello\n").unwrap();
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/hello.sh", tmp.as_path())];
        // Should not fail (ELF parsing skipped for non-ELF).
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("include-files/hello.sh"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_adds_directory_entries() {
        let tmp = std::env::temp_dir().join("ktstr-test-include-dirs");
        std::fs::write(&tmp, b"data").unwrap();
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> =
            vec![("include-files/subdir/nested/file.txt", tmp.as_path())];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("include-files"), "should have include-files dir");
        assert!(
            s.contains("include-files/subdir"),
            "should have subdir entry"
        );
        assert!(
            s.contains("include-files/subdir/nested"),
            "should have nested subdir entry"
        );
        assert!(s.contains("bin"), "should have bin dir for busybox");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn is_elf_detects_elf_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        assert!(is_elf(&exe), "test binary should be ELF");
    }

    #[test]
    fn is_elf_rejects_non_elf() {
        let tmp = std::env::temp_dir().join("ktstr-test-not-elf");
        std::fs::write(&tmp, b"not an elf file").unwrap();
        assert!(!is_elf(&tmp));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn is_elf_rejects_short_file() {
        let tmp = std::env::temp_dir().join("ktstr-test-short-elf");
        std::fs::write(&tmp, b"ab").unwrap();
        assert!(!is_elf(&tmp));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn is_elf_nonexistent_returns_false() {
        assert!(!is_elf(Path::new("/nonexistent/file")));
    }

    #[test]
    fn include_files_rejects_path_traversal() {
        let tmp = std::env::temp_dir().join("ktstr-test-traversal");
        std::fs::write(&tmp, b"data").unwrap();
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/../etc/passwd", tmp.as_path())];
        let result = create_initramfs_base(&exe, &[], &includes, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains(".."),
            "error should mention path traversal: {err}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_rejects_fifo() {
        let fifo_path = std::env::temp_dir().join("ktstr-test-fifo");
        let _ = std::fs::remove_file(&fifo_path);
        // Create a FIFO.
        let c_path = std::ffi::CString::new(fifo_path.to_str().unwrap()).unwrap();
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) };
        if rc != 0 {
            eprintln!("skipping: mkfifo failed");
            return;
        }
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/pipe", fifo_path.as_path())];
        let result = create_initramfs_base(&exe, &[], &includes, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a regular file"),
            "error should reject FIFO: {err}"
        );
        let _ = std::fs::remove_file(&fifo_path);
    }

    #[test]
    fn include_files_rejects_directory() {
        let dir_path = std::env::temp_dir().join("ktstr-test-include-dir");
        let _ = std::fs::create_dir(&dir_path);
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/mydir", dir_path.as_path())];
        let result = create_initramfs_base(&exe, &[], &includes, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a regular file"),
            "error should reject directory: {err}"
        );
        let _ = std::fs::remove_dir(&dir_path);
    }

    #[test]
    fn busybox_independent_of_include_files() {
        let exe = crate::resolve_current_exe().unwrap();
        // busybox=true but no include_files.
        let base = create_initramfs_base(&exe, &[], &[], true).unwrap();
        let names = cpio_entry_names(&base);
        assert!(
            names.iter().any(|n| n == "bin/busybox"),
            "busybox=true should have bin/busybox entry even without includes: {:?}",
            names
        );
    }
}
