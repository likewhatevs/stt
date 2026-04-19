//! Runtime configuration primitives shared by `eval` and `probe`.
//!
//! These items live here instead of either `eval` or `probe` so the
//! two siblings do not form a circular import chain. `eval` calls
//! `probe::attempt_auto_repro` from its failure path; `probe`
//! previously borrowed `verbose`, `KTSTR_TEST_SHM_SIZE`, and
//! `config_file_parts` back from `eval`. Hoisting those three into a
//! neutral module breaks the cycle without introducing any new
//! concept — the items remain internal to `test_support`.

use std::path::{Path, PathBuf};

use super::entry::KtstrTestEntry;

/// True when `RUST_BACKTRACE` is set to `"1"` or `"full"`.
///
/// Controls whether the full guest kernel console is appended to the
/// `--- diagnostics ---` section of a failed test, and whether
/// auto-repro forwards the repro VM's COM1/COM2 output to the host
/// terminal in real time. The scheduler-log and sched_ext-dump
/// sections of a failure are always emitted regardless of this flag.
pub(crate) fn verbose() -> bool {
    std::env::var("RUST_BACKTRACE")
        .map(|v| v == "1" || v == "full")
        .unwrap_or(false)
}

/// SHM size for ktstr_test VMs: 16 MB. Sized for profraw (1-2 MB),
/// stimulus events, exit code, and test results with mid-flight drain
/// headroom.
pub(crate) const KTSTR_TEST_SHM_SIZE: u64 = 16 * 1024 * 1024;

/// Derive initramfs archive path, host path, and guest path from a
/// scheduler's `config_file`. Returns `None` when no config file is set.
pub(crate) fn config_file_parts(entry: &KtstrTestEntry) -> Option<(String, PathBuf, String)> {
    let config_path = entry.scheduler.config_file?;
    let file_name = Path::new(config_path)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("config_file must have a valid filename");
    let archive_path = format!("include-files/{file_name}");
    let guest_path = format!("/include-files/{file_name}");
    Some((archive_path, PathBuf::from(config_path), guest_path))
}
