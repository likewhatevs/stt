//! ELF strip pipeline for the cached vmlinux sidecar.
//!
//! The pipeline reduces a full debug vmlinux (potentially hundreds of
//! MB) to the minimal subset that downstream consumers actually read,
//! while preserving the structural layout the kernel emits so probes
//! that expect specific section names can still resolve them by name
//! rather than by index.
//!
//! # 3-way section partition
//!
//! Every section in the input ELF lands in exactly one of three buckets,
//! decided by the unions [`STRUCTURAL_KEEP_SECTIONS`] (must keep
//! verbatim) and [`SPECULATIVE_ZERO_DATA_SECTIONS`] (keep header,
//! zero data) plus the keep-list logic in [`strip_keep_list`]:
//!
//! 1. **Keep verbatim.** Sections the runtime depends on at full
//!    fidelity: BTF (`.BTF`, `.BTF.ext`), kallsyms tables, IKCONFIG
//!    blob, structural `.shstrtab` / null. These pass through with
//!    bytes intact so `monitor`, `probe::btf`, and ktstr's CONFIG_HZ
//!    recovery still find what they need.
//! 2. **Header-only (zeroed data).** Sections whose names matter to
//!    consumers but whose bytes do not (e.g. `.init.data`). The
//!    section header is rewritten to `SHT_NOBITS` so the symbol
//!    address space is preserved without inflating the file.
//! 3. **Drop entirely.** Everything else — debug info (`.debug_*`),
//!    relocations against stripped sections, unreferenced strings.
//!
//! # Pipeline stages
//!
//! [`strip_vmlinux_debug`] is the orchestrator. It runs two stages
//! in order, each operating on the byte buffer the previous stage
//! produced, with a fallback path when the second stage fails:
//!
//! 1. [`neutralize_relocs`] — rewrite reloc sections (`SHT_REL`,
//!    `SHT_RELA`, `SHT_RELR`, `SHT_CREL`) to safe `SHT_PROGBITS`
//!    placeholders so the keep-list strip's section walker doesn't
//!    panic on malformed reloc payloads. Always runs first.
//! 2. [`strip_keep_list`] — apply the 3-way partition above using
//!    the structural / zero-data unions plus the kallsyms-and-friends
//!    keep-list embedded in this file. This is the primary strip
//!    that produces the minimal cached vmlinux.
//!    - **Fallback**: if `strip_keep_list` returns an error (e.g.
//!      ELF section-table inconsistency, an unrecognised section
//!      type, or a future toolchain emitting headers the keep-list
//!      logic can't classify), the orchestrator logs a
//!      `tracing::warn!` and falls through to
//!      [`strip_debug_prefix`], which is a strictly weaker strip
//!      that only drops `SHF_ALLOC=0` `.debug_*` sections. The
//!      fallback succeeds for any well-formed ELF, so a partial
//!      strip is preferred over a total failure that would brick
//!      the cache write.
//!
//! # Consumers
//!
//! - `monitor::btf_offsets` reads `.BTF` and `.BTF.ext` for type
//!   discovery.
//! - `probe::btf` walks BTF for runtime type resolution.
//! - `cache_dir::CacheDir::store` invokes the pipeline to produce
//!   the cached `vmlinux.stripped` artifact at write time.
//! - `super::resolve::prefer_source_tree_for_dwarf` short-circuits
//!   the cache (re-routing back to a full-fat operator-supplied
//!   vmlinux) when the operator's tree is already a kernel source
//!   directory with matching identity — see that helper for the
//!   exact policy.

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

/// Strip a vmlinux ELF for caching by partitioning every section
/// into one of three buckets and rewriting the file accordingly.
///
/// # 3-way partition
///
/// The strip pipeline classifies every input section against the
/// keep-list logic in [`strip_keep_list`] and the unions defined
/// in this file:
///
/// 1. **Keep verbatim** — sections in [`STRUCTURAL_KEEP_SECTIONS`]
///    plus the kallsyms-and-friends keep-list embedded in
///    `strip_keep_list` (BTF, IKCONFIG, `.shstrtab`, `.symtab`,
///    `.strtab`, etc.). Header AND data preserved bit-for-bit so
///    `monitor::btf_offsets`, `probe::btf`, kallsyms lookup, and
///    CONFIG_HZ recovery from the IKCONFIG blob all keep working.
/// 2. **Header-only (zeroed data)** — sections in
///    [`SPECULATIVE_ZERO_DATA_SECTIONS`] (today: `.init.data`).
///    Section header is rewritten to `SHT_NOBITS` so address-space
///    layout and section-name-to-index mapping survive, but the
///    bytes are dropped. No current consumer needs the bytes,
///    only the addressing.
/// 3. **Drop entirely** — every other section. `.debug_*` debug
///    info, dropped relocation tables, unreferenced strings, etc.
///    These contribute the bulk of the size savings (typically
///    >90% reduction).
///
/// # Fallback path
///
/// If [`strip_keep_list`] returns an error (e.g. ELF section-table
/// inconsistency, an unrecognised section type, or a future
/// toolchain that emits headers the keep-list logic can't classify),
/// the helper logs a `tracing::warn!` and falls back to
/// [`strip_debug_prefix`], which is a strictly weaker strip that
/// only drops `.debug_*` sections. The fallback always succeeds
/// for any well-formed ELF, so a partial strip is preferred over
/// a total failure that would brick the cache write entirely.
/// `cache_dir::CacheDir::store` records the "stripped" outcome in
/// `metadata.vmlinux_stripped` regardless of which path ran, so
/// `cargo ktstr kernel list --json` operators can identify entries
/// that took the fallback (and entries where the strip failed
/// outright — see `store()` for the strip-failure-to-unstripped
/// fallback at the next layer up).
///
/// # Pipeline order
///
/// 1. Read `vmlinux_path` into memory.
/// 2. [`neutralize_relocs`] — rewrite reloc sections to safe
///    placeholders so the keep-list strip's section walker doesn't
///    panic on malformed reloc payloads.
/// 3. [`strip_keep_list`] (or fallback to [`strip_debug_prefix`]).
/// 4. Write the stripped bytes to a fresh tempfile and return a
///    [`StrippedVmlinux`] that owns the temp dir.
///
/// # Return format
///
/// Returns [`StrippedVmlinux`], an RAII handle that bundles:
///
/// - `path()` — absolute path to the stripped ELF on disk under a
///   fresh `tempfile::TempDir`. The caller (typically
///   [`super::cache_dir::CacheDir::store`]) `fs::copy`s this path
///   into the cache directory.
/// - The owned `TempDir`, which is unlinked when the
///   `StrippedVmlinux` is dropped. The cache `fs::copy` happens
///   before drop, so the cached entry is independent of the temp
///   path.
///
/// Errors from any pipeline stage propagate as `anyhow::Error`
/// with stage-prefixed context (`"read vmlinux for stripping"`,
/// `"preprocess vmlinux ELF"`, etc.) so the caller can route
/// strip failures to the unstripped-cache fallback in
/// `cache_dir::CacheDir::store`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    use tempfile::TempDir;

    /// Decode an ELF section `sh_type` integer to its `SHT_*` constant
    /// name for actionable failure messages.
    fn sh_type_name(t: u32) -> &'static str {
        use goblin::elf::section_header::{
            SHT_DYNAMIC, SHT_DYNSYM, SHT_HASH, SHT_NOBITS, SHT_NOTE, SHT_NULL, SHT_PROGBITS,
            SHT_REL, SHT_RELA, SHT_SHLIB, SHT_STRTAB, SHT_SYMTAB,
        };
        match t {
            SHT_NULL => "SHT_NULL",
            SHT_PROGBITS => "SHT_PROGBITS",
            SHT_SYMTAB => "SHT_SYMTAB",
            SHT_STRTAB => "SHT_STRTAB",
            SHT_RELA => "SHT_RELA",
            SHT_HASH => "SHT_HASH",
            SHT_DYNAMIC => "SHT_DYNAMIC",
            SHT_NOTE => "SHT_NOTE",
            SHT_NOBITS => "SHT_NOBITS",
            SHT_REL => "SHT_REL",
            SHT_SHLIB => "SHT_SHLIB",
            SHT_DYNSYM => "SHT_DYNSYM",
            _ => "SHT_UNKNOWN",
        }
    }

    /// Check whether `elf` has a defined symbol with the given name.
    fn has_symbol(elf: &goblin::elf::Elf, name: &str) -> bool {
        elf.syms
            .iter()
            .any(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(name))
    }

    /// Build a minimal ELF object with a single `.text` section
    /// anchored by one symbol.
    fn build_base_elf_with_text_symbol(
        arch: object::Architecture,
    ) -> object::write::Object<'static> {
        use object::write;
        let sym_size = match arch {
            object::Architecture::X86_64 => 8,
            object::Architecture::I386 => 4,
            other => panic!(
                "build_base_elf_with_text_symbol: unsupported arch {other:?}; supported: X86_64, I386",
            ),
        };
        let mut obj =
            write::Object::new(object::BinaryFormat::Elf, arch, object::Endianness::Little);
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: sym_size,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        obj
    }

    #[test]
    #[should_panic(expected = "unsupported arch")]
    fn build_base_elf_with_text_symbol_panics_on_unsupported_arch() {
        let _ = build_base_elf_with_text_symbol(object::Architecture::Aarch64);
    }

    // -- keep-list source disjointness --

    #[test]
    fn keep_section_sources_are_disjoint() {
        use std::collections::HashMap;
        let mut origins: HashMap<&[u8], Vec<&str>> = HashMap::new();
        let sources: &[(&str, &[&[u8]])] = &[
            ("cache::STRUCTURAL_KEEP_SECTIONS", STRUCTURAL_KEEP_SECTIONS),
            (
                "monitor::symbols::VMLINUX_KEEP_SECTIONS",
                crate::monitor::symbols::VMLINUX_KEEP_SECTIONS,
            ),
            (
                "monitor::VMLINUX_KEEP_SECTIONS",
                crate::monitor::VMLINUX_KEEP_SECTIONS,
            ),
            (
                "probe::btf::VMLINUX_KEEP_SECTIONS",
                crate::probe::btf::VMLINUX_KEEP_SECTIONS,
            ),
        ];
        for (label, list) in sources {
            for name in *list {
                origins.entry(*name).or_default().push(label);
            }
        }
        let dupes: Vec<_> = origins
            .iter()
            .filter(|(_, lists)| lists.len() > 1)
            .collect();
        assert!(
            dupes.is_empty(),
            "keep-list entries declared by multiple source modules (drift hazard): {dupes:?}",
        );
    }

    #[test]
    fn zero_data_section_sources_are_disjoint() {
        use std::collections::HashSet;
        let speculative: HashSet<&[u8]> = SPECULATIVE_ZERO_DATA_SECTIONS.iter().copied().collect();
        let declared: HashSet<&[u8]> = crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .copied()
            .collect();
        let overlap: Vec<_> = speculative.intersection(&declared).collect();
        assert!(
            overlap.is_empty(),
            "zero-data section declared by both SPECULATIVE and a consumer (drift hazard): {overlap:?}",
        );
    }

    /// Build a minimal ELF covering every strip dispatch branch.
    fn create_strip_test_fixture(dir: &Path) -> std::path::PathBuf {
        use object::write;
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x10,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Metadata);
        obj.append_section_data(btf_id, &[0xEB; 256], 1);
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xCA; 512], 1);
        let bss_id = obj.add_section(
            Vec::new(),
            b".bss".to_vec(),
            object::SectionKind::UninitializedData,
        );
        obj.append_section_bss(bss_id, 256, 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_bss_symbol".to_vec(),
            value: 0x50,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(bss_id),
            flags: object::SymbolFlags::None,
        });
        let btf_ext_id = obj.add_section(
            Vec::new(),
            b".BTF.ext".to_vec(),
            object::SectionKind::Metadata,
        );
        obj.append_section_data(btf_ext_id, &[0xE1; 128], 1);
        let debug_id = obj.add_section(
            Vec::new(),
            b".debug_info".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_id, &[0xAA; 4096], 1);
        let debug_str_id = obj.add_section(
            Vec::new(),
            b".debug_str".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_str_id, &[0xBB; 2048], 1);
        let data_id = obj.add_section(Vec::new(), b".data".to_vec(), object::SectionKind::Data);
        obj.append_section_data(data_id, &[0xDD; 512], 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_data_symbol".to_vec(),
            value: 0x20,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(data_id),
            flags: object::SymbolFlags::None,
        });
        let percpu_id = obj.add_section(
            Vec::new(),
            b".data..percpu".to_vec(),
            object::SectionKind::Data,
        );
        obj.append_section_data(percpu_id, &[0xCC; 256], 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_percpu_symbol".to_vec(),
            value: 0x30,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(percpu_id),
            flags: object::SymbolFlags::None,
        });
        let initdata_id = obj.add_section(
            Vec::new(),
            b".init.data".to_vec(),
            object::SectionKind::Data,
        );
        obj.append_section_data(initdata_id, &[0x11; 1024], 8);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_initdata_symbol".to_vec(),
            value: 0x40,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(initdata_id),
            flags: object::SymbolFlags::None,
        });

        let data = obj.write().unwrap();
        let path = dir.join("vmlinux");
        fs::write(&path, &data).unwrap();
        path
    }

    #[test]
    fn strip_vmlinux_debug_applies_keep_list() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());
        let original_size = fs::metadata(&vmlinux).unwrap().len();

        let source_data = fs::read(&vmlinux).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let source_section_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [
            ".debug_info",
            ".debug_str",
            ".BTF.ext",
            ".BTF",
            ".rodata",
            ".bss",
            ".symtab",
            ".strtab",
        ] {
            assert!(
                source_section_names.contains(&name),
                "fixture missing expected section {name}"
            );
        }

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_path = stripped.path();
        let stripped_size = fs::metadata(stripped_path).unwrap().len();

        assert!(
            stripped_size < original_size,
            "stripped ({stripped_size}) should be smaller than original ({original_size})"
        );

        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let section_names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        assert!(
            !section_names.contains(&".debug_info"),
            "should not contain .debug_info"
        );
        assert!(
            !section_names.contains(&".debug_str"),
            "should not contain .debug_str"
        );
        assert!(
            !section_names.contains(&".BTF.ext"),
            "should not contain .BTF.ext"
        );
        for name in [".BTF", ".rodata", ".bss", ".symtab", ".strtab"] {
            assert!(section_names.contains(&name), "should preserve {name}");
        }
    }

    #[test]
    fn strip_vmlinux_debug_symtab_readable() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();

        assert!(
            has_symbol(&elf, "test_text_symbol"),
            "stripped ELF should contain test_text_symbol in symtab"
        );
        assert!(
            has_symbol(&elf, "test_bss_symbol"),
            "stripped ELF should contain test_bss_symbol in symtab"
        );
    }

    #[test]
    fn strip_vmlinux_debug_zeros_data_sections() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());

        use goblin::elf::section_header::SHT_NOBITS;
        let source_data = fs::read(&vmlinux).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        for name_bytes in crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .chain(SPECULATIVE_ZERO_DATA_SECTIONS.iter())
        {
            let name = std::str::from_utf8(name_bytes).unwrap();
            let sh = source_elf
                .section_headers
                .iter()
                .find(|s| source_elf.shdr_strtab.get_at(s.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("fixture missing expected {name}"));
            assert_ne!(
                sh.sh_type,
                SHT_NOBITS,
                "fixture {name} must start non-SHT_NOBITS so the strip is observable; got sh_type={} ({})",
                sh.sh_type,
                sh_type_name(sh.sh_type),
            );
            assert!(
                sh.sh_size > 0,
                "fixture {name} must start with nonzero sh_size"
            );
        }

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();

        let find_section = |name: &str| {
            elf.section_headers
                .iter()
                .find(|s| elf.shdr_strtab.get_at(s.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("section {name} missing from stripped ELF"))
        };
        let assert_nobits_empty = |name: &str| {
            let sh = find_section(name);
            let sh_type = sh.sh_type;
            let sh_size = sh.sh_size;
            assert_eq!(
                sh_type,
                SHT_NOBITS,
                "section {name} should be SHT_NOBITS after strip, got sh_type={sh_type} ({})",
                sh_type_name(sh_type),
            );
            assert_eq!(
                sh_size, 0,
                "section {name} should have sh_size == 0 after strip, got {sh_size}",
            );
        };

        for name_bytes in crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .chain(SPECULATIVE_ZERO_DATA_SECTIONS.iter())
        {
            let name = std::str::from_utf8(name_bytes).unwrap();
            assert_nobits_empty(name);
        }

        assert_nobits_empty(".text");

        assert!(
            has_symbol(&elf, "test_data_symbol"),
            "test_data_symbol dropped by strip"
        );
        assert!(
            has_symbol(&elf, "test_percpu_symbol"),
            "test_percpu_symbol dropped by strip"
        );
        assert!(
            has_symbol(&elf, "test_initdata_symbol"),
            "test_initdata_symbol dropped by strip"
        );
    }

    #[test]
    fn strip_debug_prefix_removes_debug_and_preserves_rest() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());
        let raw = fs::read(&vmlinux).unwrap();
        let processed = neutralize_relocs(&raw).unwrap();

        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        assert!(
            !names.contains(&".debug_info"),
            "fallback should remove .debug_info"
        );
        assert!(
            !names.contains(&".debug_str"),
            "fallback should remove .debug_str"
        );
        for name in [".BTF", ".BTF.ext", ".text", ".data", ".rodata", ".symtab"] {
            assert!(
                names.contains(&name),
                "fallback must preserve {name}, got sections {names:?}"
            );
        }
    }

    #[test]
    fn strip_debug_prefix_removes_dot_comment() {
        use object::write;
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let comment_id = obj.add_section(
            Vec::new(),
            b".comment".to_vec(),
            object::SectionKind::OtherString,
        );
        obj.append_section_data(comment_id, b"GCC: (GNU) 14.2.1 20250207\0", 1);
        let data = obj.write().unwrap();

        let source_elf = goblin::elf::Elf::parse(&data).unwrap();
        let source_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [".comment", ".text"] {
            assert!(
                source_names.contains(&name),
                "fixture missing expected section {name}; got {source_names:?}"
            );
        }

        let processed = neutralize_relocs(&data).unwrap();
        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        assert!(
            !names.contains(&".comment"),
            "fallback must remove .comment, got sections {names:?}"
        );
        assert!(
            names.contains(&".text"),
            "fallback must preserve .text, got sections {names:?}"
        );
    }

    #[test]
    fn strip_debug_prefix_removes_reloc_prefix_sections() {
        use object::elf::{SHT_REL, SHT_RELA, SHT_RELR};

        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        let rela_id = obj.add_section(
            Vec::new(),
            b".rela.text".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rela_id, &[0xA5; 24], 1);
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.data".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 16], 1);
        let relr_id = obj.add_section(
            Vec::new(),
            b".relr.dyn".to_vec(),
            object::SectionKind::Elf(SHT_RELR),
        );
        obj.append_section_data(relr_id, &[0xD3; 16], 1);
        let crel_id = obj.add_section(
            Vec::new(),
            b".crel.text".to_vec(),
            object::SectionKind::Elf(object::elf::SHT_CREL),
        );
        obj.append_section_data(crel_id, &[0xE4; 8], 1);
        let data = obj.write().unwrap();

        let source_elf = goblin::elf::Elf::parse(&data).unwrap();
        let source_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [
            ".rela.text",
            ".rel.data",
            ".relr.dyn",
            ".crel.text",
            ".text",
        ] {
            assert!(
                source_names.contains(&name),
                "fixture missing expected section {name}; got {source_names:?}"
            );
        }

        let processed = neutralize_relocs(&data).unwrap();
        let stripped = strip_debug_prefix(&processed).unwrap();
        let elf = goblin::elf::Elf::parse(&stripped).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        for name in [".rela.text", ".rel.data", ".relr.dyn", ".crel.text"] {
            assert!(
                !names.contains(&name),
                "fallback must delete {name} (prefix arm), got sections {names:?}"
            );
        }
        assert!(
            names.contains(&".text"),
            "fallback must preserve .text, got sections {names:?}"
        );
    }

    #[test]
    fn neutralize_relocs_zeros_sh_size_of_every_reloc_section() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA, SHT_RELR};

        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        let relr_id = obj.add_section(
            Vec::new(),
            b".relr.dyn".to_vec(),
            object::SectionKind::Elf(SHT_RELR),
        );
        obj.append_section_data(relr_id, &[0xD3; 24], 1);
        obj.section_mut(relr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };

        let data = obj.write().unwrap();

        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        let mut pre_kaslr = None;
        let mut pre_rel = None;
        let mut pre_rdbg = None;
        let mut pre_relr = None;
        let mut pre_text = None;
        for sh in pre_elf.section_headers.iter() {
            let name = pre_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => pre_kaslr = Some(sh.clone()),
                ".rel.foo" => pre_rel = Some(sh.clone()),
                ".rela.debug_info" => pre_rdbg = Some(sh.clone()),
                ".relr.dyn" => pre_relr = Some(sh.clone()),
                ".text" => pre_text = Some(sh.clone()),
                _ => {}
            }
        }
        let pre_kaslr = pre_kaslr.expect("fixture must carry .rela.kaslr");
        let pre_rel = pre_rel.expect("fixture must carry .rel.foo");
        let pre_rdbg = pre_rdbg.expect("fixture must carry .rela.debug_info");
        let pre_relr = pre_relr.expect("fixture must carry .relr.dyn");
        let pre_text = pre_text.expect("fixture must carry .text");
        assert_eq!(pre_kaslr.sh_type, SHT_RELA);
        assert!(pre_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0);
        assert_eq!(pre_kaslr.sh_size, 32);
        assert_eq!(pre_rel.sh_type, SHT_REL);
        assert!(pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0);
        assert_eq!(pre_rel.sh_size, 24);
        assert_eq!(pre_rdbg.sh_type, SHT_RELA);
        assert_eq!(pre_rdbg.sh_flags & u64::from(SHF_ALLOC), 0);
        assert_eq!(pre_rdbg.sh_size, 16);
        assert_eq!(pre_relr.sh_type, SHT_RELR);
        assert_eq!(pre_relr.sh_size, 24);
        assert_eq!(pre_text.sh_size, 64);

        let kaslr_offset = pre_kaslr.sh_offset as usize;
        let kaslr_size = pre_kaslr.sh_size as usize;
        let kaslr_original_data = data[kaslr_offset..kaslr_offset + kaslr_size].to_vec();

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(processed.len(), data.len());

        let post_elf = goblin::elf::Elf::parse(&processed).unwrap();
        let mut post_kaslr = None;
        let mut post_rel = None;
        let mut post_rdbg = None;
        let mut post_relr = None;
        let mut post_text = None;
        for sh in post_elf.section_headers.iter() {
            let name = post_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => post_kaslr = Some(sh.clone()),
                ".rel.foo" => post_rel = Some(sh.clone()),
                ".rela.debug_info" => post_rdbg = Some(sh.clone()),
                ".relr.dyn" => post_relr = Some(sh.clone()),
                ".text" => post_text = Some(sh.clone()),
                _ => {}
            }
        }
        let post_kaslr = post_kaslr.expect(".rela.kaslr must survive");
        let post_rel = post_rel.expect(".rel.foo must survive");
        let post_rdbg = post_rdbg.expect(".rela.debug_info must survive");
        let post_relr = post_relr.expect(".relr.dyn must survive");
        let post_text = post_text.expect(".text must survive");

        assert_eq!(post_kaslr.sh_size, 0);
        assert_eq!(post_rel.sh_size, 0);
        assert_eq!(post_rdbg.sh_size, 0);
        assert_eq!(post_relr.sh_size, 0);
        assert_eq!(post_text.sh_size, pre_text.sh_size);

        assert_eq!(
            &processed[kaslr_offset..kaslr_offset + kaslr_size],
            &kaslr_original_data[..],
            ".rela.kaslr data bytes must be preserved; neutralize only rewrites sh_size"
        );

        assert_eq!(post_kaslr.sh_offset, pre_kaslr.sh_offset);
        assert_eq!(post_kaslr.sh_type, object::elf::SHT_PROGBITS);
        assert_eq!(post_kaslr.sh_flags, pre_kaslr.sh_flags);
        assert_eq!(post_rel.sh_type, object::elf::SHT_PROGBITS);
        assert_eq!(post_rdbg.sh_type, object::elf::SHT_PROGBITS);
        assert_eq!(post_relr.sh_type, object::elf::SHT_PROGBITS);
    }

    #[test]
    fn neutralize_relocs_noop_when_no_reloc_sections() {
        let data = build_base_elf_with_text_symbol(object::Architecture::X86_64)
            .write()
            .unwrap();

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed, data,
            "neutralize_relocs must be a byte-identity no-op when no reloc sections are present"
        );
    }

    #[test]
    fn neutralize_relocs_is_idempotent() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);

        let data = obj.write().unwrap();

        let first_pass = neutralize_relocs(&data).unwrap();
        let second_pass = neutralize_relocs(&first_pass).unwrap();

        assert_ne!(
            first_pass, data,
            "first call must modify bytes on a fixture with reloc sections"
        );
        assert_eq!(
            second_pass, first_pass,
            "neutralize_relocs must be idempotent"
        );
        assert_eq!(first_pass.len(), data.len());
        assert_eq!(second_pass.len(), first_pass.len());

        let post_elf = goblin::elf::Elf::parse(&second_pass)
            .expect("second-pass output must remain parseable as ELF");

        let mut post_kaslr = None;
        let mut post_rel = None;
        let mut post_rdbg = None;
        for sh in post_elf.section_headers.iter() {
            let name = post_elf.shdr_strtab.get_at(sh.sh_name).unwrap_or("");
            match name {
                ".rela.kaslr" => post_kaslr = Some(sh.clone()),
                ".rel.foo" => post_rel = Some(sh.clone()),
                ".rela.debug_info" => post_rdbg = Some(sh.clone()),
                _ => {}
            }
        }
        let post_kaslr = post_kaslr.expect(".rela.kaslr must survive second pass");
        let post_rel = post_rel.expect(".rel.foo must survive second pass");
        let post_rdbg = post_rdbg.expect(".rela.debug_info must survive second pass");

        assert_eq!(post_kaslr.sh_size, 0);
        assert_eq!(post_rel.sh_size, 0);
        assert_eq!(post_rdbg.sh_size, 0);

        assert!(post_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0);
        assert!(post_rel.sh_flags & u64::from(SHF_ALLOC) != 0);
        assert_eq!(post_rdbg.sh_flags & u64::from(SHF_ALLOC), 0);
    }

    #[test]
    fn neutralize_relocs_rejects_invalid_elf() {
        let cases: &[(&str, &[u8])] = &[
            ("bad magic", b"not an ELF at all, just some bytes"),
            (
                "magic ok but invalid EI_CLASS",
                &[0x7f, b'E', b'L', b'F', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];
        for (label, input) in cases {
            let err = neutralize_relocs(input).unwrap_err();
            let rendered = format!("{err:#}");
            assert!(
                rendered.contains("parse vmlinux ELF for preprocess"),
                "[{label}] expected error context to name the ELF parse step; got: {rendered}"
            );
        }
    }

    #[test]
    fn neutralize_relocs_zeros_sh_size_in_elf32_fixture() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        let mut obj = build_base_elf_with_text_symbol(object::Architecture::I386);
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 16], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 12], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };

        let data = obj.write().unwrap();

        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            !pre_elf.is_64,
            "fixture must produce ELF32 (is_64 == false) to exercise the (20, 4) branch"
        );
        let pre_kaslr = pre_elf
            .section_headers
            .iter()
            .find(|sh| pre_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rela.kaslr"))
            .expect("fixture must carry .rela.kaslr")
            .clone();
        let pre_rel = pre_elf
            .section_headers
            .iter()
            .find(|sh| pre_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rel.foo"))
            .expect("fixture must carry .rel.foo")
            .clone();
        assert_eq!(pre_kaslr.sh_type, SHT_RELA);
        assert!(pre_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0);
        assert_eq!(pre_kaslr.sh_size, 16);
        assert_eq!(pre_rel.sh_type, SHT_REL);
        assert!(pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0);
        assert_eq!(pre_rel.sh_size, 12);

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(processed.len(), data.len());

        let post_elf = goblin::elf::Elf::parse(&processed).unwrap();
        assert!(!post_elf.is_64, "post-call parse must still be ELF32");
        let post_kaslr = post_elf
            .section_headers
            .iter()
            .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rela.kaslr"))
            .expect(".rela.kaslr must survive the neutralize pass")
            .clone();
        let post_rel = post_elf
            .section_headers
            .iter()
            .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(".rel.foo"))
            .expect(".rel.foo must survive the neutralize pass")
            .clone();
        assert_eq!(post_kaslr.sh_size, 0);
        assert_eq!(post_rel.sh_size, 0);
    }

    #[test]
    fn neutralize_relocs_noop_when_no_reloc_sections_elf32() {
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::I386,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: 4,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let data = obj.write().unwrap();

        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(!pre_elf.is_64);

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(processed, data);
    }

    #[test]
    fn neutralize_relocs_is_idempotent_elf32() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::I386,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_text_symbol".to_vec(),
            value: 0x0,
            size: 4,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 16], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 12], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 8], 1);

        let data = obj.write().unwrap();

        assert!(!goblin::elf::Elf::parse(&data).unwrap().is_64);

        let first_pass = neutralize_relocs(&data).unwrap();
        let second_pass = neutralize_relocs(&first_pass).unwrap();

        assert_ne!(first_pass, data);
        assert_eq!(second_pass, first_pass);

        let post_elf = goblin::elf::Elf::parse(&second_pass).unwrap();
        for name in [".rela.kaslr", ".rel.foo", ".rela.debug_info"] {
            let sh = post_elf
                .section_headers
                .iter()
                .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("{name} must survive second pass"));
            assert_eq!(sh.sh_size, 0);
        }
    }

    #[test]
    fn strip_vmlinux_debug_nonexistent_file() {
        let result = strip_vmlinux_debug(Path::new("/nonexistent/vmlinux"));
        assert!(result.is_err());
    }

    #[test]
    fn strip_vmlinux_debug_non_elf_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("vmlinux");
        fs::write(&path, b"not an ELF file").unwrap();
        let result = strip_vmlinux_debug(&path);
        assert!(result.is_err());
    }

    /// Build an ELF on disk matching `create_strip_test_fixture`'s
    /// shape plus one extra section provided by the caller.
    fn build_reloc_fixture(
        dir: &Path,
        extra_section_name: &[u8],
        extra_section_sh_type: u32,
        extra_section_data: &[u8],
        mutate_header: impl FnOnce(&mut [u8]),
    ) -> std::path::PathBuf {
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 16);
        let _ = obj.add_symbol(write::Symbol {
            name: b"pipeline_anchor".to_vec(),
            value: 0x10,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Other);
        obj.append_section_data(btf_id, &[0x42; 128], 1);
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xAA; 256], 1);
        let extra_id = obj.add_section(
            Vec::new(),
            extra_section_name.to_vec(),
            object::SectionKind::Elf(extra_section_sh_type),
        );
        obj.append_section_data(extra_id, extra_section_data, 1);

        let mut bytes = obj.write().unwrap();
        mutate_header(&mut bytes);
        let path = dir.join("vmlinux");
        fs::write(&path, &bytes).unwrap();
        path
    }

    fn assert_stripped_preserves_keep_list_and_deletes(stripped: &Path, reloc_name: &str) {
        let data = fs::read(stripped).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [".symtab", ".strtab", ".BTF", ".rodata"] {
            assert!(
                names.contains(&name),
                "keep-list section {name} must survive strip_vmlinux_debug; got {names:?}"
            );
        }
        assert!(
            !names.contains(&reloc_name),
            "reloc section {reloc_name} must be deleted by strip_vmlinux_debug; got {names:?}"
        );
    }

    #[test]
    fn strip_vmlinux_debug_handles_nonalloc_rela_with_invalid_entries() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".rela.invalid",
            object::elf::SHT_RELA,
            &[0xA5; 24],
            |_| {},
        );
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".rela.invalid");
    }

    #[test]
    fn strip_vmlinux_debug_handles_nonalloc_rela_with_non_entsize_sh_size() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".rela.odd",
            object::elf::SHT_RELA,
            &[0x11; 24],
            |bytes| {
                let elf = goblin::elf::Elf::parse(bytes).unwrap();
                let shoff = elf.header.e_shoff as usize;
                let shentsize = elf.header.e_shentsize as usize;
                let idx = elf
                    .section_headers
                    .iter()
                    .position(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some(".rela.odd"))
                    .expect("fixture must carry .rela.odd");
                drop(elf);
                let sh_size_off = shoff + idx * shentsize + 32;
                let bad_size: u64 = 17;
                bytes[sh_size_off..sh_size_off + 8].copy_from_slice(&bad_size.to_le_bytes());
            },
        );
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".rela.odd");
    }

    #[test]
    fn strip_vmlinux_debug_handles_relr_section() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".relr.dyn",
            object::elf::SHT_RELR,
            &[0x77; 16],
            |_| {},
        );

        let raw = fs::read(&vmlinux).unwrap();
        let neutralized = neutralize_relocs(&raw).unwrap();
        let neutralized_elf = goblin::elf::Elf::parse(&neutralized).unwrap();
        let relr_sh = neutralized_elf
            .section_headers
            .iter()
            .find(|sh| neutralized_elf.shdr_strtab.get_at(sh.sh_name) == Some(".relr.dyn"))
            .expect(".relr.dyn must survive neutralize");
        assert_eq!(
            relr_sh.sh_type,
            object::elf::SHT_PROGBITS,
            ".relr.dyn sh_type must be rewritten to SHT_PROGBITS",
        );
        assert_eq!(
            relr_sh.sh_size, 0,
            ".relr.dyn sh_size must be zeroed post-neutralize"
        );

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".relr.dyn");
    }

    #[test]
    fn strip_vmlinux_debug_preserves_monitor_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(stripped_path).unwrap();
        assert_ne!(
            syms.runqueues, 0,
            "runqueues symbol missing from stripped vmlinux"
        );
        assert_ne!(
            syms.per_cpu_offset, 0,
            "__per_cpu_offset symbol missing from stripped vmlinux"
        );
        let source_syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(&path).unwrap();
        assert_eq!(
            source_syms.init_top_pgt.is_some(),
            syms.init_top_pgt.is_some(),
            "strip changed KernelSymbols init_top_pgt presence"
        );
        assert_eq!(
            source_syms.page_offset_base_kva.is_some(),
            syms.page_offset_base_kva.is_some(),
            "strip changed page_offset_base_kva presence"
        );
        assert_eq!(
            source_syms.scx_root.is_some(),
            syms.scx_root.is_some(),
            "strip changed scx_root presence"
        );
        assert_eq!(
            source_syms.pgtable_l5_enabled.is_some(),
            syms.pgtable_l5_enabled.is_some(),
            "strip changed pgtable_l5_enabled presence"
        );
        assert_eq!(
            source_syms.prog_idr.is_some(),
            syms.prog_idr.is_some(),
            "strip changed prog_idr presence"
        );
        assert_eq!(
            source_syms.scx_watchdog_timeout.is_some(),
            syms.scx_watchdog_timeout.is_some(),
            "strip changed scx_watchdog_timeout presence"
        );

        let source_data = fs::read(&path).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let stripped_data = fs::read(stripped_path).unwrap();
        let stripped_elf = goblin::elf::Elf::parse(&stripped_data).unwrap();
        assert_eq!(
            has_symbol(&source_elf, "init_top_pgt"),
            has_symbol(&stripped_elf, "init_top_pgt"),
            "strip changed raw-symtab init_top_pgt presence"
        );
        assert_eq!(
            has_symbol(&source_elf, "swapper_pg_dir"),
            has_symbol(&stripped_elf, "swapper_pg_dir"),
            "strip changed raw-symtab swapper_pg_dir presence"
        );
    }

    #[test]
    fn strip_vmlinux_debug_shrinks_when_source_has_debug_info() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let source_data = fs::read(&path).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let source_has_debug = source_elf
            .section_headers
            .iter()
            .any(|sh| source_elf.shdr_strtab.get_at(sh.sh_name) == Some(".debug_info"));
        if !source_has_debug {
            skip!(
                "source vmlinux has no .debug_info — already stripped \
                 (cached copy or distro-stripped); rebuild source tree \
                 to exercise the size-shrink path"
            );
        }

        let stripped = strip_vmlinux_debug(&path).unwrap();
        let source_size = fs::metadata(&path).unwrap().len();
        let stripped_size = fs::metadata(stripped.path()).unwrap().len();
        assert!(
            stripped_size < source_size,
            "stripped vmlinux ({stripped_size} bytes) should be smaller than \
             source ({source_size} bytes)"
        );
    }

    #[test]
    fn strip_vmlinux_debug_preserves_bpf_idr_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            has_symbol(&elf, "map_idr"),
            "map_idr symbol missing from stripped vmlinux"
        );
        assert!(
            has_symbol(&elf, "prog_idr"),
            "prog_idr symbol missing from stripped vmlinux"
        );
    }

    #[test]
    fn strip_vmlinux_debug_preserves_function_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let source_data = fs::read(&path).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        if !has_symbol(&source_elf, "schedule") {
            skip!(
                "source vmlinux has no `schedule` symbol \
                 (already stripped by older ktstr) -- rebuild the kernel \
                 cache to exercise this test"
            );
        }

        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let data = fs::read(stripped_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            has_symbol(&elf, "schedule"),
            "schedule function symbol dropped by strip"
        );
    }

    #[test]
    fn strip_vmlinux_debug_deletes_reloc_sections_and_preserves_keep_list() {
        use object::write;

        let src = TempDir::new().unwrap();
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 16);
        let _ = obj.add_symbol(write::Symbol {
            name: b"pipeline_anchor".to_vec(),
            value: 0x10,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Other);
        obj.append_section_data(btf_id, &[0x42; 128], 1);
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xAA; 256], 1);
        let rela_id = obj.add_section(
            Vec::new(),
            b".rela.dbg".to_vec(),
            object::SectionKind::Elf(object::elf::SHT_RELA),
        );
        obj.append_section_data(rela_id, &[0xA5; 24], 1);
        let relr_id = obj.add_section(
            Vec::new(),
            b".relr.dyn".to_vec(),
            object::SectionKind::Elf(object::elf::SHT_RELR),
        );
        obj.append_section_data(relr_id, &[0xD3; 24], 1);

        let bytes = obj.write().unwrap();
        let vmlinux = src.path().join("vmlinux");
        fs::write(&vmlinux, &bytes).unwrap();

        let source_elf = goblin::elf::Elf::parse(&bytes).unwrap();
        let source_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        for name in [
            ".text",
            ".BTF",
            ".rodata",
            ".rela.dbg",
            ".relr.dyn",
            ".symtab",
            ".strtab",
        ] {
            assert!(
                source_names.contains(&name),
                "fixture missing expected section {name}; got {source_names:?}"
            );
        }

        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        let data = fs::read(stripped.path()).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();

        for name in [".symtab", ".strtab", ".BTF", ".rodata"] {
            assert!(
                names.contains(&name),
                "keep-list section {name} must survive strip; got {names:?}"
            );
        }
        for name in [".rela.dbg", ".relr.dyn"] {
            assert!(
                !names.contains(&name),
                "reloc section {name} must be deleted by strip; got {names:?}"
            );
        }
    }
}
