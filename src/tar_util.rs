//! Shared tar-archive helpers for [`crate::export`] and
//! [`crate::remote_cache`] — the only two crate sites that build
//! tar archives in memory.
//!
//! Both modules previously open-coded the same five-line sequence:
//! `Header::new_gnu` + `set_size` + `set_mode` + `set_cksum` +
//! `append`/`append_data`. [`pack_tar_entry`] consolidates the
//! sequence so call sites express what they want (a named entry
//! with a mode and a payload) rather than how to assemble a tar
//! header.
//!
//! `tar::Builder::append_data` already calls `Header::set_cksum`
//! after rewriting the path, so the helper does NOT call it
//! separately — the previous open-coded `set_cksum()` calls in
//! `remote_cache.rs` were redundant double-cksums that the helper
//! drops.
//!
//! Mode is required (not defaulted) because the two callers use
//! distinct modes deliberately: `export` writes 0o755 so the
//! `.run` extractor preserves executable bits on the embedded
//! `ktstr`/`scheduler` binaries, while `remote_cache` writes
//! 0o644 for cached metadata/image files that are never executed
//! directly. Defaulting either mode would silently regress the
//! other site.
use std::io::{Read, Write};

/// Pack one named entry into a tar archive in memory.
///
/// Builds a fresh GNU header, writes `size` and `mode`, and
/// streams `reader` (which must yield exactly `size` bytes per
/// the tar format contract) under `name`. The path is encoded by
/// `append_data` so callers do NOT need to pre-set it on the
/// header.
///
/// Returns the underlying I/O error from `append_data` on
/// failure — typical causes are a write error against the
/// builder's sink or a `reader` that yields fewer bytes than
/// `size` claims.
pub fn pack_tar_entry<W: Write, R: Read>(
    builder: &mut tar::Builder<W>,
    name: &str,
    mode: u32,
    size: u64,
    reader: R,
) -> std::io::Result<()> {
    let mut header = tar::Header::new_gnu();
    header.set_size(size);
    header.set_mode(mode);
    builder.append_data(&mut header, name, reader)
}
