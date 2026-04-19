//! Guest-side LLVM coverage profraw flush + host-side write-out.
//!
//! Under `-C instrument-coverage`, the compiler inserts profile counters
//! and registers an atexit handler via `.init_array` that writes
//! `.profraw` at process exit. Inside a ktstr guest VM, `std::process::exit`
//! bypasses the atexit handler when the ktstr `#[ctor]` runs first
//! (the ordering between `.init_array` entries is unspecified). To keep
//! coverage data from being dropped, [`try_flush_profraw`] resolves the
//! LLVM runtime symbols via ELF `.symtab` (they have hidden visibility,
//! so `dlsym` can't see them), initializes the profile writer against an
//! in-tmpfs path, calls `__llvm_profile_write_file`, reads the file back,
//! and publishes it through the guest-to-host SHM ring under
//! [`MSG_TYPE_PROFRAW`].
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

use anyhow::{Context, Result};
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
/// Sets `LLVM_PROFILE_FILE` and calls `__llvm_profile_initialize` to
/// configure the output path, then `__llvm_profile_write_file` to write
/// profraw to a tmpfs file inside the guest. Reads the file back and
/// writes the contents to the SHM ring for host-side extraction.
///
/// All symbols have hidden visibility in compiler-rt, so we resolve
/// them via ELF .symtab parsing (dlsym cannot find hidden symbols).
///
/// No-op when built without `-C instrument-coverage` or when SHM
/// parameters are absent from the kernel command line.
pub(crate) fn try_flush_profraw() {
    if parse_shm_params().is_none() {
        return;
    }

    let exe = match std::fs::read("/proc/self/exe") {
        Ok(data) => data,
        Err(_) => return,
    };
    let slide = pie_load_bias(&exe);

    // Resolve both symbols in a single pass through the ELF .symtab.
    let vaddrs = find_symbol_vaddrs(
        &exe,
        &["__llvm_profile_initialize", "__llvm_profile_write_file"],
    );

    // Set profraw output path, then call __llvm_profile_initialize to
    // read it and register the atexit handler.
    // SAFETY: single-threaded guest dispatch context.
    unsafe { std::env::set_var("LLVM_PROFILE_FILE", "/tmp/ktstr.profraw") };
    if let Some(vaddr) = vaddrs[0]
        && vaddr != 0
    {
        let f: extern "C" fn() =
            unsafe { std::mem::transmute((vaddr as usize).wrapping_add(slide)) };
        f();
    }

    // Write profraw to the file.
    let write_file_vaddr = match vaddrs[1] {
        Some(v) if v != 0 => v,
        _ => return,
    };
    let write_file: extern "C" fn() -> i32 =
        unsafe { std::mem::transmute((write_file_vaddr as usize).wrapping_add(slide)) };
    if write_file() != 0 {
        return;
    }

    // Read the profraw file and send through SHM ring.
    let data = match std::fs::read("/tmp/ktstr.profraw") {
        Ok(d) if !d.is_empty() => d,
        _ => return,
    };
    vmm::shm_ring::write_msg(MSG_TYPE_PROFRAW, &data);
}

/// Resolve multiple symbol virtual addresses in a single pass through
/// the ELF .symtab. Returns addresses in the same order as `names`.
pub(crate) fn find_symbol_vaddrs(data: &[u8], names: &[&str]) -> Vec<Option<u64>> {
    let mut results = vec![None; names.len()];
    let mut remaining = names.len();

    let elf = match goblin::elf::Elf::parse(data) {
        Ok(e) => e,
        Err(_) => return results,
    };

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
pub(crate) fn pie_load_bias(data: &[u8]) -> usize {
    let elf = match goblin::elf::Elf::parse(data) {
        Ok(e) => e,
        Err(_) => return 0,
    };

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
pub(crate) fn target_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LLVM_COV_TARGET_DIR") {
        return PathBuf::from(d);
    }
    if let Some(parent) = std::env::var("LLVM_PROFILE_FILE")
        .ok()
        .as_ref()
        .and_then(|p| Path::new(p).parent())
    {
        return parent.to_path_buf();
    }
    let mut p = crate::resolve_current_exe().unwrap_or_else(|_| std::env::temp_dir());
    p.pop(); // remove binary name
    p.push("llvm-cov-target");
    p
}
