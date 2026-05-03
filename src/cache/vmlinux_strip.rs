//! ELF strip pipeline for the cached vmlinux sidecar.

use std::fs;
use std::path::{Path, PathBuf};

/// Structural ELF sections that must survive any cache-time strip.
pub(crate) const STRUCTURAL_KEEP_SECTIONS: &[&[u8]] = &[
    b"",          // null section (index 0) — required by ELF spec
    b".shstrtab", // section header string table
];

/// Data sections retained as SHT_NOBITS headers with no current
/// consumer.
pub(crate) const SPECULATIVE_ZERO_DATA_SECTIONS: &[&[u8]] = &[b".init.data"];

/// Union of consumer-declared keep-lists plus structural sections.
fn is_keep_section(name: &[u8]) -> bool {
    STRUCTURAL_KEEP_SECTIONS.contains(&name)
        || crate::monitor::symbols::VMLINUX_KEEP_SECTIONS.contains(&name)
        || crate::monitor::VMLINUX_KEEP_SECTIONS.contains(&name)
        || crate::probe::btf::VMLINUX_KEEP_SECTIONS.contains(&name)
}

/// Union of consumer-declared zero-data lists plus the speculative
/// retention set.
fn is_zero_data_section(name: &[u8]) -> bool {
    SPECULATIVE_ZERO_DATA_SECTIONS.contains(&name)
        || crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS.contains(&name)
}

/// Stripped vmlinux written to a temporary file. Owns the backing
/// `tempfile::TempDir` so the file stays alive until the caller
/// drops the [`StrippedVmlinux`].
#[derive(Debug)]
pub(crate) struct StrippedVmlinux {
    _tmp: tempfile::TempDir,
    path: PathBuf,
}

impl StrippedVmlinux {
    /// Path to the stripped vmlinux file inside the owned temp dir.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Strip vmlinux for caching. Three-way partition: keep / zero / delete.
/// Falls back to [`strip_debug_prefix`] when the keep-list strip errors.
pub(crate) fn strip_vmlinux_debug(vmlinux_path: &Path) -> anyhow::Result<StrippedVmlinux> {
    let raw =
        fs::read(vmlinux_path).map_err(|e| anyhow::anyhow!("read vmlinux for stripping: {e}"))?;
    let original_size = raw.len();
    let data =
        neutralize_relocs(&raw).map_err(|e| anyhow::anyhow!("preprocess vmlinux ELF: {e}"))?;

    let out = match strip_keep_list(&data) {
        Ok(buf) => buf,
        Err(e) => {
            tracing::warn!("keep-list strip failed ({e:#}), falling back to debug-only strip");
            strip_debug_prefix(&data)?
        }
    };

    let stripped_size = out.len();
    let saved_mb = (original_size - stripped_size) as f64 / (1024.0 * 1024.0);
    tracing::debug!(
        original = original_size,
        stripped = stripped_size,
        saved_mb = format!("{saved_mb:.0}"),
        "strip_vmlinux_debug",
    );

    let tmp_dir = tempfile::TempDir::new()
        .map_err(|e| anyhow::anyhow!("create temp dir for stripped vmlinux: {e}"))?;
    let stripped_path = tmp_dir.path().join("vmlinux");
    fs::write(&stripped_path, &out).map_err(|e| anyhow::anyhow!("write stripped vmlinux: {e}"))?;
    Ok(StrippedVmlinux {
        _tmp: tmp_dir,
        path: stripped_path,
    })
}

/// Rewrite every relocation section (`SHT_REL`, `SHT_RELA`,
/// `SHT_RELR`, `SHT_CREL`) to `SHT_PROGBITS` with `sh_size = 0`,
/// regardless of `SHF_ALLOC`. Workaround for object-crate Builder
/// failure modes on malformed reloc inputs (sh_size past EOF,
/// non-entsize sh_size, alignment of empty slices, invalid symbol
/// indices). After rewrite, every reloc section reads through the
/// opaque-data path and writes back as a zero-byte SHT_PROGBITS.
pub(crate) fn neutralize_relocs(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    const SHT_RELR: u32 = object::elf::SHT_RELR;
    const SHT_CREL: u32 = object::elf::SHT_CREL;
    const SHT_PROGBITS: u32 = goblin::elf::section_header::SHT_PROGBITS;

    let elf = goblin::elf::Elf::parse(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF for preprocess: {e}"))?;
    let mut out = data.to_vec();
    let shoff = elf.header.e_shoff as usize;
    let shentsize = elf.header.e_shentsize as usize;
    let (sh_size_offset, sh_size_width) = if elf.is_64 { (32, 8) } else { (20, 4) };
    let sh_type_offset: usize = 4;
    let sh_type_width: usize = 4;
    let le = elf.little_endian;
    use goblin::elf::section_header::{SHT_REL, SHT_RELA};
    for (i, sh) in elf.section_headers.iter().enumerate() {
        let is_reloc = matches!(sh.sh_type, SHT_REL | SHT_RELA | SHT_RELR | SHT_CREL);
        if !is_reloc {
            continue;
        }
        let entry_offset = shoff
            .checked_add(
                i.checked_mul(shentsize)
                    .ok_or_else(|| anyhow::anyhow!("section header table overflow at index {i}"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("section header offset overflow at index {i}"))?;
        let type_offset = entry_offset
            .checked_add(sh_type_offset)
            .ok_or_else(|| anyhow::anyhow!("sh_type offset overflow at index {i}"))?;
        let type_end = type_offset
            .checked_add(sh_type_width)
            .ok_or_else(|| anyhow::anyhow!("sh_type end overflow at index {i}"))?;
        let size_offset = entry_offset
            .checked_add(sh_size_offset)
            .ok_or_else(|| anyhow::anyhow!("sh_size offset overflow at index {i}"))?;
        let size_end = size_offset
            .checked_add(sh_size_width)
            .ok_or_else(|| anyhow::anyhow!("sh_size end overflow at index {i}"))?;
        if type_end > out.len() || size_end > out.len() {
            anyhow::bail!("section header {i} sh_type or sh_size field extends past file end");
        }
        let type_bytes: [u8; 4] = if le {
            SHT_PROGBITS.to_le_bytes()
        } else {
            SHT_PROGBITS.to_be_bytes()
        };
        out[type_offset..type_end].copy_from_slice(&type_bytes);
        out[size_offset..size_end].fill(0);
    }
    Ok(out)
}

/// Keep-list strip: three-way partition of ELF sections by name.
pub(crate) fn strip_keep_list(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut builder = object::build::elf::Builder::read(data)
        .map_err(|e| anyhow::anyhow!("parse vmlinux ELF: {e}"))?;
    for section in builder.sections.iter_mut() {
        let name = section.name.as_slice();
        if is_keep_section(name) {
            continue;
        }
        if is_zero_data_section(name) {
            section.sh_type = object::elf::SHT_NOBITS;
            section.data = object::build::elf::SectionData::UninitializedData(0);
            continue;
        }
        let is_code = section.sh_flags & u64::from(object::elf::SHF_EXECINSTR) != 0;
        if is_code {
            section.sh_type = object::elf::SHT_NOBITS;
            section.data = object::build::elf::SectionData::UninitializedData(0);
        } else {
            section.delete = true;
        }
    }

    let named_syms = builder
        .symbols
        .iter()
        .filter(|s| !s.delete && !s.name.as_slice().is_empty())
        .count();
    if named_syms == 0 {
        anyhow::bail!("keep-list strip emptied symbol table (0 named symbols)");
    }

    let mut out = Vec::new();
    builder
        .write(&mut out)
        .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux: {e}"))?;
    Ok(out)
}

/// Fallback strip: remove `.debug_*`, `.comment`, and neutralized
/// relocation sections.
pub(crate) fn strip_debug_prefix(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    crate::elf_strip::rewrite(data, |name| {
        name.starts_with(b".debug_")
            || name == b".comment"
            || name.starts_with(b".rela.")
            || name.starts_with(b".rel.")
            || name.starts_with(b".relr.")
            || name.starts_with(b".crel.")
    })
    .map_err(|e| anyhow::anyhow!("rewrite stripped vmlinux (fallback): {e}"))
}
