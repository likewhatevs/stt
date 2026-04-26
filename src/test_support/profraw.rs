//! Guest-side LLVM coverage profraw flush + host-side write-out.
//!
//! Under `-C instrument-coverage`, the compiler inserts profile counters
//! and registers an atexit handler via `.init_array` that writes
//! `.profraw` at process exit. Inside a ktstr guest VM, `std::process::exit`
//! bypasses the atexit handler when the ktstr `#[ctor]` runs first
//! (the ordering between `.init_array` entries is unspecified). To keep
//! coverage data from being dropped, [`try_flush_profraw`] resolves the
//! LLVM runtime symbols via ELF `.symtab` (they have hidden visibility,
//! so `dlsym` can't see them), serializes profraw into a heap buffer
//! via `__llvm_profile_write_buffer`, and publishes it through the
//! guest-to-host SHM ring under [`MSG_TYPE_PROFRAW`].
//!
//! VP data scope: the buffer flush covers coverage counters and
//! bitmaps only; PGO value-profile data is not preserved.
//! `__llvm_profile_write_buffer` passes a NULL `VPDataReader` to
//! `lprofWriteData` (defined in
//! `compiler-rt/lib/profile/InstrProfilingBuffer.c`),
//! whereas the file-based `__llvm_profile_write_file` path passes
//! `lprofGetVPDataReader()` (`InstrProfilingFile.c`) and DOES
//! capture VP records. This matches the current `-C instrument-coverage`
//! use case, which does not emit VP data. Combining coverage with PGO
//! (`-C profile-generate`) in the same binary would silently lose VP
//! records on this path; switch back to the file-based serializer if
//! that combination becomes a requirement.
//!
//! On the host, [`write_profraw`] receives those bytes via the SHM ring
//! and writes them into `LLVM_COV_TARGET_DIR` (or a fallback sibling
//! directory next to the test binary) as
//! `ktstr-test-{pid}-{counter}.profraw`.
//!
//! Supporting helpers:
//! - [`find_symbol_vaddrs`] walks `.symtab` in one pass for multiple
//!   symbols at once.
//! - [`pie_load_bias`] computes the ASLR slide for PIE binaries so
//!   symbol virtual addresses can be rebased to runtime pointers.
//! - [`parse_shm_params`] extracts the SHM base/size the host injected
//!   via `/proc/cmdline` (`KTSTR_SHM_BASE` / `KTSTR_SHM_SIZE`).
//!
//! `/proc/self/exe` is read via `memmap2::Mmap` rather than
//! `std::fs::read` so the kernel page cache backs the bytes goblin
//! parses; for coverage-instrumented binaries (hundreds of MiB up to
//! ~1 GiB) this avoids the heap allocation + copy of the entire
//! binary on every flush. The ELF is parsed once and reused across
//! [`pie_load_bias`] and [`find_symbol_vaddrs`] so a single
//! `goblin::elf::Elf::parse` covers both lookups.

use anyhow::{Context, Result};
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::vmm;

/// SHM ring message type for profraw data.
///
/// Derived from the ASCII bytes `b"PRAW"` in big-endian order so the
/// constant reads as the tag it represents in a hex dump, not as an
/// opaque 32-bit magic number. Equivalent to `0x50524157`.
pub(crate) const MSG_TYPE_PROFRAW: u32 = u32::from_be_bytes(*b"PRAW");

/// Flush LLVM coverage profraw to the SHM ring buffer.
///
/// Resolves `__llvm_profile_get_size_for_buffer` and
/// `__llvm_profile_write_buffer` from the test binary's `.symtab`,
/// allocates a heap buffer of the reported size, calls
/// `__llvm_profile_write_buffer` to serialize the profile counters
/// into it, and publishes the buffer through the SHM ring for
/// host-side extraction.
///
/// All symbols have hidden visibility in compiler-rt, so we resolve
/// them via ELF `.symtab` parsing (dlsym cannot find hidden symbols).
///
/// No-op when built without `-C instrument-coverage` or when SHM
/// parameters are absent from the kernel command line.
pub(crate) fn try_flush_profraw() {
    if parse_shm_params().is_none() {
        return;
    }

    // Memory-map the test binary's executable image. memmap2's `Mmap`
    // borrows from the underlying file mapping; the page cache is the
    // backing store, so goblin's parse + symtab walk reads pages on
    // demand instead of paying for a full read+copy of the (possibly
    // ~1 GiB) coverage-instrumented binary.
    //
    // SAFETY: `/proc/self/exe` is a kernel symlink to the running
    // binary's underlying file (proc(5)); File::open captures an fd
    // that pins the inode for the mmap's lifetime, so even a concurrent
    // binary replacement on disk leaves the mapped pages valid. No part
    // of ktstr or its callers writes to the test binary during a run,
    // satisfying memmap2's no-concurrent-modification invariant.
    let exe_file = match File::open("/proc/self/exe") {
        Ok(f) => f,
        Err(_) => return,
    };
    let mmap = match unsafe { memmap2::Mmap::map(&exe_file) } {
        Ok(m) => m,
        Err(_) => return,
    };
    let bytes: &[u8] = &mmap;

    // Parse the ELF once and reuse it for both `pie_load_bias` and
    // `find_symbol_vaddrs`; a second parse cost a measurable share of
    // teardown latency on coverage builds.
    let elf = match goblin::elf::Elf::parse(bytes) {
        Ok(e) => e,
        Err(_) => return,
    };

    let slide = pie_load_bias(&elf);

    // Resolve both buffer-API symbols in a single pass through the
    // ELF .symtab.
    let vaddrs = find_symbol_vaddrs(
        &elf,
        &[
            "__llvm_profile_get_size_for_buffer",
            "__llvm_profile_write_buffer",
        ],
    );

    let size_vaddr = match vaddrs[0] {
        Some(v) if v != 0 => v,
        _ => return,
    };
    let write_vaddr = match vaddrs[1] {
        Some(v) if v != 0 => v,
        _ => return,
    };

    // SAFETY: vaddr + slide is the runtime address of the
    // hidden-visibility compiler-rt entry point with the C signature
    // `uint64_t (void)`; treating it as `extern "C" fn() -> u64`
    // matches that ABI. The dispatch context is single-threaded.
    let get_size: extern "C" fn() -> u64 =
        unsafe { std::mem::transmute((size_vaddr as usize).wrapping_add(slide)) };
    // SAFETY: same, for `int (char *)` → `extern "C" fn(*mut c_char) -> i32`.
    let write_buffer: extern "C" fn(*mut std::os::raw::c_char) -> i32 =
        unsafe { std::mem::transmute((write_vaddr as usize).wrapping_add(slide)) };

    let needed = get_size() as usize;
    if needed == 0 {
        return;
    }

    let mut buf: Vec<u8> = vec![0u8; needed];
    // `__llvm_profile_write_buffer` returns 0 on success.
    if write_buffer(buf.as_mut_ptr().cast::<std::os::raw::c_char>()) != 0 {
        return;
    }

    vmm::shm_ring::write_msg(MSG_TYPE_PROFRAW, &buf);
}

/// Resolve multiple symbol virtual addresses in a single pass through
/// the ELF .symtab. Returns addresses in the same order as `names`.
pub(crate) fn find_symbol_vaddrs(elf: &goblin::elf::Elf<'_>, names: &[&str]) -> Vec<Option<u64>> {
    let mut results = vec![None; names.len()];
    let mut remaining = names.len();

    for sym in elf.syms.iter() {
        if remaining == 0 {
            break;
        }
        if sym.st_size == 0 {
            continue;
        }
        let sym_name = match elf.strtab.get_at(sym.st_name) {
            Some(n) => n,
            None => continue,
        };
        for (i, name) in names.iter().enumerate() {
            if results[i].is_none() && sym_name == *name {
                results[i] = Some(sym.st_value);
                remaining -= 1;
                break;
            }
        }
    }
    results
}

/// Compute the ASLR load bias for a PIE binary.
///
/// For ET_DYN (PIE), the kernel loads the binary at an arbitrary base.
/// The bias is `runtime_phdr_addr - file_phdr_offset`. We get the
/// runtime phdr address from AT_PHDR (via getauxval) and the file
/// offset from e_phoff.
///
/// Returns 0 for ET_EXEC (non-PIE), where st_value is already absolute.
pub(crate) fn pie_load_bias(elf: &goblin::elf::Elf<'_>) -> usize {
    if elf.header.e_type != goblin::elf::header::ET_DYN {
        return 0;
    }

    let phdr_file_offset = elf.header.e_phoff as usize;
    // SAFETY: AT_PHDR is a well-defined auxiliary vector key.
    let phdr_runtime = unsafe { libc::getauxval(libc::AT_PHDR) } as usize;
    if phdr_runtime == 0 {
        return 0;
    }
    phdr_runtime.wrapping_sub(phdr_file_offset)
}

/// Parse KTSTR_SHM_BASE and KTSTR_SHM_SIZE from /proc/cmdline.
pub(crate) fn parse_shm_params() -> Option<(u64, u64)> {
    let cmdline = std::fs::read_to_string("/proc/cmdline").ok()?;
    vmm::shm_ring::parse_shm_params_from_str(&cmdline)
}

static PROFRAW_COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Write profraw data to the llvm-cov-target directory.
pub(crate) fn write_profraw(data: &[u8]) -> Result<()> {
    let target_dir = target_dir();
    std::fs::create_dir_all(&target_dir)
        .with_context(|| format!("create profraw dir: {}", target_dir.display()))?;
    let id = PROFRAW_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = target_dir.join(format!("ktstr-test-{}-{}.profraw", std::process::id(), id));
    std::fs::write(&path, data).with_context(|| format!("write profraw: {}", path.display()))?;
    Ok(())
}

/// Resolve the llvm-cov-target directory for profraw output.
///
/// Cascade:
/// 1. `LLVM_COV_TARGET_DIR` — explicit operator override.
/// 2. `LLVM_PROFILE_FILE`'s parent directory — when an outer harness
///    (cargo-llvm-cov, or the cargo-ktstr `LLVM_PROFILE_FILE` injection
///    that prevents host-side `default.profraw` leakage from the
///    `cargo ktstr test` path) has already pinned the output location.
/// 3. `<current_exe parent>/llvm-cov-target/` — workspace-local
///    fallback so an instrumented binary invoked without any
///    coordination still drops profraw next to the build output
///    rather than in cwd.
///
/// `pub` rather than `pub(crate)` so the cargo-ktstr binary can
/// resolve the same directory before exec-ing `cargo nextest run`,
/// keeping host-side and guest-side profraw output co-located in
/// one tree without cargo-ktstr re-implementing the cascade.
pub fn target_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LLVM_COV_TARGET_DIR") {
        return PathBuf::from(d);
    }
    // `LLVM_PROFILE_FILE` may be a bare filename (e.g. `default.profraw`)
    // — `Path::parent` returns `Some("")` in that shape, which would
    // otherwise propagate a structurally-empty `PathBuf` through the
    // cascade and surface as an unusable target dir downstream
    // (`std::fs::create_dir_all("")` errors with EINVAL on Linux).
    // The empty-os-str filter forces those bare-filename cases to fall
    // through to the `current_exe`-relative fallback below.
    if let Some(parent) = std::env::var("LLVM_PROFILE_FILE")
        .ok()
        .as_ref()
        .and_then(|p| Path::new(p).parent())
        .filter(|p| !p.as_os_str().is_empty())
    {
        return parent.to_path_buf();
    }
    let mut p = crate::resolve_current_exe().unwrap_or_else(|_| std::env::temp_dir());
    p.pop(); // remove binary name
    p.push("llvm-cov-target");
    p
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{EnvVarGuard, lock_env};
    use super::*;
    use crate::vmm::shm_ring::parse_shm_params_from_str;

    // -- parse_shm_params (/proc/cmdline) --

    #[test]
    fn parse_shm_params_absent() {
        let cmdline = std::fs::read_to_string("/proc/cmdline").unwrap_or_default();
        if cmdline.contains("KTSTR_SHM_BASE") {
            skip!(
                "host /proc/cmdline has KTSTR_SHM_BASE (self-hosted guest?); \
                 the pure-string branch of parse_shm_params is covered by \
                 parse_shm_params_from_str_*"
            );
        }
        let result = parse_shm_params();
        assert!(
            result.is_none(),
            "host without KTSTR_SHM_BASE in /proc/cmdline must yield None"
        );
    }

    // -- parse_shm_params_from_str --

    #[test]
    fn parse_shm_params_from_str_lowercase_hex() {
        let cmdline = "console=ttyS0 KTSTR_SHM_BASE=0xfc000000 KTSTR_SHM_SIZE=0x400000 quiet";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_uppercase_hex() {
        let cmdline = "KTSTR_SHM_BASE=0XFC000000 KTSTR_SHM_SIZE=0X400000";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xFC000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_no_prefix() {
        let cmdline = "KTSTR_SHM_BASE=fc000000 KTSTR_SHM_SIZE=400000";
        let (base, size) = parse_shm_params_from_str(cmdline).unwrap();
        assert_eq!(base, 0xfc000000);
        assert_eq!(size, 0x400000);
    }

    #[test]
    fn parse_shm_params_from_str_missing_base() {
        let cmdline = "console=ttyS0 KTSTR_SHM_SIZE=0x400000";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_missing_size() {
        let cmdline = "KTSTR_SHM_BASE=0xfc000000 quiet";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_missing_both() {
        let cmdline = "console=ttyS0 quiet";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    #[test]
    fn parse_shm_params_from_str_empty() {
        assert!(parse_shm_params_from_str("").is_none());
    }

    #[test]
    fn parse_shm_params_from_str_invalid_hex() {
        let cmdline = "KTSTR_SHM_BASE=0xZZZZ KTSTR_SHM_SIZE=0x400000";
        assert!(parse_shm_params_from_str(cmdline).is_none());
    }

    // -- target_dir --

    #[test]
    fn target_dir_with_env_var() {
        let _lock = lock_env();
        let _env = EnvVarGuard::set("LLVM_COV_TARGET_DIR", "/tmp/my-cov-dir");
        let dir = target_dir();
        assert_eq!(dir, PathBuf::from("/tmp/my-cov-dir"));
    }

    #[test]
    fn target_dir_from_llvm_profile_file() {
        let _lock = lock_env();
        let _env_cov = EnvVarGuard::remove("LLVM_COV_TARGET_DIR");
        let _env_prof =
            EnvVarGuard::set("LLVM_PROFILE_FILE", "/tmp/cov-target/ktstr-%p-%m.profraw");
        let dir = target_dir();
        assert_eq!(dir, PathBuf::from("/tmp/cov-target"));
    }

    #[test]
    fn target_dir_without_env_var() {
        let _lock = lock_env();
        let _env_cov = EnvVarGuard::remove("LLVM_COV_TARGET_DIR");
        let _env_prof = EnvVarGuard::remove("LLVM_PROFILE_FILE");
        let dir = target_dir();
        // Falls back to current_exe parent + "llvm-cov-target".
        assert!(
            dir.ends_with("llvm-cov-target"),
            "expected path ending in llvm-cov-target, got: {}",
            dir.display()
        );
    }

    /// `LLVM_PROFILE_FILE` set to a bare filename (no parent
    /// directory component, e.g. `default.profraw`) must fall
    /// through to the `current_exe`-relative fallback rather than
    /// surfacing a structurally-empty `PathBuf` through the
    /// cascade. `Path::new("default.profraw").parent()` returns
    /// `Some("")`; without the empty-os-str filter,
    /// `target_dir` would return `PathBuf::from("")` and downstream
    /// `create_dir_all` calls fail with EINVAL.
    #[test]
    fn target_dir_bare_filename_llvm_profile_file_falls_through() {
        let _lock = lock_env();
        let _g_cov = EnvVarGuard::remove("LLVM_COV_TARGET_DIR");
        let _g_prof = EnvVarGuard::set("LLVM_PROFILE_FILE", "default.profraw");
        let dir = target_dir();
        assert!(
            !dir.as_os_str().is_empty(),
            "bare-filename LLVM_PROFILE_FILE must fall through to the \
             current_exe fallback, not return an empty PathBuf",
        );
        assert!(
            dir.ends_with("llvm-cov-target"),
            "fallback must land at the current_exe-relative llvm-cov-target \
             dir, got: {}",
            dir.display(),
        );
    }

    // -- MSG_TYPE_PROFRAW encoding --

    #[test]
    fn msg_type_profraw_ascii() {
        let bytes = MSG_TYPE_PROFRAW.to_be_bytes();
        assert_eq!(&bytes, b"PRAW");
    }

    // -- shm_write full-ring semantics (uses MSG_TYPE_PROFRAW) --

    #[test]
    fn shm_write_returns_zero_on_full_ring() {
        use crate::vmm::shm_ring::{HEADER_SIZE, MSG_HEADER_SIZE, shm_init, shm_write};

        // Small ring: header + 32 bytes data.
        let shm_size = HEADER_SIZE + 32;
        let mut buf = vec![0u8; shm_size];
        shm_init(&mut buf, 0, shm_size);

        // Fill the ring: 16-byte header + 16-byte payload = 32 bytes.
        let payload = vec![0xAA; 16];
        let written = shm_write(&mut buf, 0, MSG_TYPE_PROFRAW, &payload);
        assert_eq!(written, MSG_HEADER_SIZE + 16);

        // Ring is full — next write returns 0.
        let written = shm_write(&mut buf, 0, MSG_TYPE_PROFRAW, b"overflow");
        assert_eq!(written, 0);
    }

    // -- find_symbol_vaddrs --

    #[test]
    fn find_symbol_vaddrs_resolves_known_symbol() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        // "main" is present in the symtab of any Rust test binary.
        let results = find_symbol_vaddrs(&elf, &["main"]);
        assert_eq!(results.len(), 1);
        assert!(
            results[0].is_some(),
            "main symbol should be resolved in test binary"
        );
        assert_ne!(results[0].unwrap(), 0, "main address should be nonzero");
    }

    #[test]
    fn find_symbol_vaddrs_missing_symbol_returns_none() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let results = find_symbol_vaddrs(&elf, &["__nonexistent_symbol_xyz__"]);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_none());
    }

    #[test]
    fn find_symbol_vaddrs_mixed_results() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let results = find_symbol_vaddrs(&elf, &["main", "__nonexistent_symbol_xyz__"]);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some(), "main should resolve");
        assert!(results[1].is_none(), "nonexistent should not resolve");
    }

    // -- pie_load_bias --

    /// `pie_load_bias` on the running test binary returns a non-zero
    /// slide. Rust's default release/test build is PIE (`ET_DYN`), and
    /// on a stock Linux kernel (`randomize_va_space=2`) ASLR places
    /// the image at a random base, so `runtime_phdr - file_phdr_offset`
    /// is virtually always a large non-zero value. Guards both that
    /// the function takes the ET_DYN path AND that the AT_PHDR /
    /// e_phoff arithmetic produces something usable.
    #[test]
    fn pie_load_bias_returns_nonzero_slide_on_pie_test_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = std::fs::read(&exe).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        // Sanity: the test binary really is ET_DYN. If a future
        // toolchain change flips the default to ET_EXEC, this guard
        // surfaces the change before the slide assertion fails for
        // the wrong reason.
        assert_eq!(
            elf.header.e_type,
            goblin::elf::header::ET_DYN,
            "Rust test binaries default to PIE (ET_DYN); got e_type={}",
            elf.header.e_type,
        );
        let slide = pie_load_bias(&elf);
        assert_ne!(
            slide, 0,
            "ET_DYN binary under default ASLR must produce a non-zero \
             load bias; if this assertion fails check that \
             /proc/sys/kernel/randomize_va_space is non-zero",
        );
    }

    /// Mutating `e_type` to ET_EXEC routes `pie_load_bias` through
    /// the early-exit branch and returns 0 — the contract for non-PIE
    /// binaries where `st_value` is already absolute.
    ///
    /// The mutation writes `ET_EXEC` (= 2) into the `e_type` field at
    /// byte offset 16 (= `SIZEOF_IDENT` in `goblin::elf::header`).
    /// Endianness is chosen from `e_ident[EI_DATA]` (= byte 5, see
    /// `EI_DATA`, `ELFDATA2LSB`, `ELFDATA2MSB` in
    /// `goblin::elf::header`) so the test
    /// works on both little-endian (x86_64, aarch64) and big-endian
    /// hosts. Re-parsing the mutated buffer yields a synthetic
    /// `Elf<'_>` whose header reports ET_EXEC; everything else
    /// (sections, segments, strtab) stays consistent because we only
    /// touched the type byte.
    #[test]
    fn pie_load_bias_returns_zero_for_et_exec() {
        let exe = crate::resolve_current_exe().unwrap();
        let mut data = std::fs::read(&exe).unwrap();

        // EI_DATA byte tells us the file's endianness.
        let little_endian = match data[goblin::elf::header::EI_DATA] {
            goblin::elf::header::ELFDATA2LSB => true,
            goblin::elf::header::ELFDATA2MSB => false,
            other => panic!("unexpected EI_DATA byte: 0x{other:02x}"),
        };
        let et_exec_bytes: [u8; 2] = if little_endian {
            goblin::elf::header::ET_EXEC.to_le_bytes()
        } else {
            goblin::elf::header::ET_EXEC.to_be_bytes()
        };
        // e_type sits immediately after e_ident (which is SIZEOF_IDENT
        // = 16 bytes). Overwrite both bytes of the u16.
        data[goblin::elf::header::SIZEOF_IDENT] = et_exec_bytes[0];
        data[goblin::elf::header::SIZEOF_IDENT + 1] = et_exec_bytes[1];

        let elf = goblin::elf::Elf::parse(&data).unwrap();
        assert_eq!(
            elf.header.e_type,
            goblin::elf::header::ET_EXEC,
            "byte mutation should have made the parsed header report ET_EXEC",
        );
        assert_eq!(
            pie_load_bias(&elf),
            0,
            "ET_EXEC binary must short-circuit to 0; absolute st_value \
             needs no slide",
        );
    }
}
