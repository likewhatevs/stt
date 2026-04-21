//! ELF section-stripping primitives shared between cache vmlinux
//! handling and initramfs payload packaging.
//!
//! Both `cache` and `vmm::initramfs` parse an ELF with the `object`
//! crate's `Builder`, mark sections for deletion, and serialize the
//! result. The per-section filter differs — the cache uses a
//! whitelist, the initramfs a blacklist — but the read/mark/write
//! pipeline is identical.

use object::build::elf::Builder;

/// Parse `data` as an ELF, delete every section for which `should_delete`
/// returns true, and serialize the result.
///
/// Returns the object-crate's build error unchanged so callers can log
/// it (the existing cache and initramfs callers swallow failures into
/// either a fallback strip pass or the unstripped bytes).
pub(crate) fn rewrite<F>(data: &[u8], mut should_delete: F) -> Result<Vec<u8>, object::build::Error>
where
    F: FnMut(&[u8]) -> bool,
{
    let mut builder = Builder::read(data)?;
    for section in builder.sections.iter_mut() {
        if should_delete(section.name.as_slice()) {
            section.delete = true;
        }
    }
    let mut out = Vec::new();
    builder.write(&mut out)?;
    Ok(out)
}
