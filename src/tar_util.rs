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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack one entry, finish the archive, then unpack and verify
    /// every property the helper sets — name, mode, size, and the
    /// payload bytes. Pins the helper's contract end-to-end so a
    /// future regression that swaps `set_size`/`set_mode` order, or
    /// drops one of the calls, surfaces here. Routed through Verdict
    /// so failure messages carry the field labels for diagnosis.
    #[test]
    fn pack_tar_entry_roundtrips_through_archive() {
        use crate::assert::Verdict;

        const NAME: &str = "ktstr";
        const MODE: u32 = 0o755;
        let payload: &[u8] = b"hello tar world\n";

        // Pack into an in-memory archive.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut buf);
            pack_tar_entry(
                &mut builder,
                NAME,
                MODE,
                payload.len() as u64,
                payload,
            )
            .expect("pack_tar_entry must succeed for valid inputs");
            builder.finish().expect("tar finish");
        }

        // Unpack and verify each header field plus the payload.
        let mut archive = tar::Archive::new(&buf[..]);
        let mut entries = archive.entries().expect("read entries");
        let mut entry = entries
            .next()
            .expect("at least one entry")
            .expect("entry header readable");

        let header = entry.header().clone();
        let path_str = header
            .path()
            .expect("path bytes must decode")
            .to_string_lossy()
            .to_string();
        let mode = header.mode().expect("mode must decode");
        let size = header.size().expect("size must decode");

        let mut got = Vec::new();
        entry.read_to_end(&mut got).expect("payload readable");

        let mut v = Verdict::new();
        crate::claim!(v, path_str).eq(NAME.to_string());
        crate::claim!(v, mode).eq(MODE);
        crate::claim!(v, size).eq(payload.len() as u64);
        // ClaimBuilder requires T: Display; Vec<u8> doesn't impl
        // Display. Compare via hex strings (already a dep) so the
        // verdict carries a readable diff on mismatch.
        let got_hex = hex::encode(&got);
        let want_hex = hex::encode(payload);
        crate::claim!(v, got_hex).eq(want_hex);
        let r = v.into_result();
        assert!(
            r.passed,
            "tar roundtrip claims must all pass: {:?}",
            r.details,
        );

        // Confirm only one entry was packed.
        assert!(
            entries.next().is_none(),
            "pack_tar_entry must produce exactly one entry per call",
        );
    }

    /// Distinct modes are preserved per call. The module doc says
    /// export uses 0o755 and remote_cache uses 0o644 — a regression
    /// that defaulted the mode (e.g. dropping `set_mode`) would
    /// produce identical modes regardless of the argument.
    #[test]
    fn pack_tar_entry_preserves_distinct_modes_across_calls() {
        use crate::assert::Verdict;

        let mut buf: Vec<u8> = Vec::new();
        {
            let mut builder = tar::Builder::new(&mut buf);
            pack_tar_entry(&mut builder, "a", 0o755, 1, &b"x"[..]).unwrap();
            pack_tar_entry(&mut builder, "b", 0o644, 1, &b"y"[..]).unwrap();
            builder.finish().unwrap();
        }

        let mut archive = tar::Archive::new(&buf[..]);
        let mut iter = archive.entries().unwrap();
        let a = iter.next().unwrap().unwrap();
        let b = iter.next().unwrap().unwrap();

        let a_mode = a.header().mode().unwrap();
        let b_mode = b.header().mode().unwrap();

        let mut v = Verdict::new();
        crate::claim!(v, a_mode).eq(0o755u32);
        crate::claim!(v, b_mode).eq(0o644u32);
        crate::claim!(v, a_mode).ne(b_mode);
        let r = v.into_result();
        assert!(
            r.passed,
            "per-call mode must be preserved distinctly: {:?}",
            r.details,
        );
    }
}
