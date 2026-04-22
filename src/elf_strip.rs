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

#[cfg(test)]
mod tests {
    //! `object` crate API pin tests. The primary consumer of the
    //! `object::build::elf` surface lives in this file; the pin test
    //! lives alongside it so an upstream API change that affects
    //! ktstr's ELF stripping surfaces at test compile/run time rather
    //! than at first production failure. The `object` crate flags the
    //! `build` feature as experimental, so churn is expected.
    use super::*;

    /// Round-trip + delete path pin — feeds the test binary's own
    /// bytes through [`rewrite`] twice: once with `should_delete`
    /// returning false (exercises `Builder::read` → unchanged iterate
    /// → `Builder::write`), once deleting `.comment` (a section
    /// reliably present on Linux binaries built with gcc/clang).
    ///
    /// Asserts section presence on the no-delete output and section
    /// absence on the strip output — a regression where
    /// `Section.delete = true` is silently ignored passes a
    /// parseability-only check but fails this stronger invariant.
    ///
    /// This pin catches:
    /// - Renamed or removed `Builder::read`/`Builder::write`.
    /// - Layout change on `Section.name` or the `.as_slice` API.
    /// - Semantics change in `Section.delete` (the sole field the
    ///   production `rewrite` mutates).
    #[test]
    fn rewrite_roundtrip_and_delete() {
        use object::read::{File, Object, ObjectSection};

        let exe = std::env::current_exe().expect("test current_exe");
        let bytes = std::fs::read(&exe).expect("read self binary");

        let no_delete = rewrite(&bytes, |_| false).expect("rewrite no-op");
        assert!(
            !no_delete.is_empty(),
            "round-trip output must not be empty",
        );
        let _: Builder<'_> = Builder::read(no_delete.as_slice())
            .expect("round-trip output must parse");
        let round_tripped =
            File::parse(no_delete.as_slice()).expect("round-trip File::parse");
        let has_comment = round_tripped
            .sections()
            .any(|s| s.name().ok() == Some(".comment"));
        assert!(
            has_comment,
            "no-op rewrite must preserve .comment (rustc-built test \
             binaries always carry .comment; its absence here means the \
             round-trip dropped it)",
        );

        let stripped =
            rewrite(&bytes, |name| name == b".comment").expect("rewrite strip");
        assert!(
            !stripped.is_empty(),
            "stripped output must not be empty",
        );
        let _: Builder<'_> =
            Builder::read(stripped.as_slice()).expect("stripped output must parse");
        let stripped_file =
            File::parse(stripped.as_slice()).expect("stripped File::parse");
        let still_has_comment = stripped_file
            .sections()
            .any(|s| s.name().ok() == Some(".comment"));
        assert!(
            !still_has_comment,
            ".comment must be absent after rewrite marked it for deletion \
             — a silent no-op on Section.delete would pass the parse \
             check but fail here",
        );
    }

    /// Behavioral pin: converting a section to `SHT_NOBITS` with
    /// `SectionData::UninitializedData(0)` must RETAIN symbols that
    /// reference the section through `Builder::write`. This is the
    /// invariant `cache::strip_keep_list` relies on to null out debug
    /// section contents while keeping symbol addresses intact for
    /// downstream consumers (monitor, probe, verifier).
    ///
    /// Constructs a minimal ELF with `.text` holding a single data
    /// symbol, sanity-checks that the symbol is present in the
    /// fixture before mutation (distinguishes fixture-generation
    /// failure from an SHT_NOBITS regression), parses with
    /// `Builder::read`, flips `.text` to `SHT_NOBITS +
    /// UninitializedData(0)`, re-serializes with `Builder::write`,
    /// and asserts the symbol by name on the output. A regression in
    /// the object crate's orphan-pruning pass that dropped symbols of
    /// NOBITS sections would fail this test.
    #[test]
    fn sht_nobits_conversion_retains_symbols() {
        use object::build::elf::SectionData;
        use object::read::{File, Object, ObjectSymbol};
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        let text_id = obj.add_section(
            Vec::new(),
            b".text".to_vec(),
            object::SectionKind::Text,
        );
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        let _ = obj.add_symbol(write::Symbol {
            name: b"test_sym".to_vec(),
            value: 0,
            size: 8,
            kind: object::SymbolKind::Data,
            scope: object::SymbolScope::Compilation,
            weak: false,
            section: write::SymbolSection::Section(text_id),
            flags: object::SymbolFlags::None,
        });
        let bytes = obj.write().expect("serialize fixture ELF");

        let fixture_check =
            File::parse(bytes.as_slice()).expect("pre-flip File::parse fixture");
        fixture_check.symbol_by_name("test_sym").expect(
            "fixture sanity check: test_sym must exist in fresh-built ELF \
             before any SHT_NOBITS mutation (distinguishes fixture-build \
             regression from the SHT_NOBITS retention regression under test)",
        );

        let mut builder = Builder::read(bytes.as_slice()).expect("Builder::read fixture");
        let mut flipped = false;
        for section in builder.sections.iter_mut() {
            if section.name.as_slice() == b".text" {
                section.sh_type = object::elf::SHT_NOBITS;
                section.data = SectionData::UninitializedData(0);
                flipped = true;
            }
        }
        assert!(flipped, "fixture must contain a .text section");
        let mut out = Vec::new();
        builder.write(&mut out).expect("Builder::write NOBITS'd ELF");

        let file = File::parse(&out[..]).expect("re-parse output as ELF");
        let sym = file
            .symbol_by_name("test_sym")
            .expect("symbol must survive SHT_NOBITS conversion");
        assert_eq!(sym.name().expect("symbol name is UTF-8"), "test_sym");
    }

    /// ELF constant value pin — `rewrite` and `cache::strip_keep_list`
    /// assume the ABI constants `SHT_NOBITS == 8` and
    /// `SHF_EXECINSTR == 0x4`. Upstream would not change the numeric
    /// values (the ELF spec fixes them) but could rename the constants
    /// or change their types; this test catches both.
    #[test]
    fn elf_constants_pinned() {
        assert_eq!(object::elf::SHT_NOBITS, 8);
        assert_eq!(object::elf::SHF_EXECINSTR, 0x4);
    }
}
