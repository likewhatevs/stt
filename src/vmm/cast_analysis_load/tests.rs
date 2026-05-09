//! Error-path coverage for the host-side BPF cast-analysis driver.
//!
//! Every public function in this module returns an empty
//! [`CastMap`] (or an empty `Vec<FuncEntry>`) on malformed input;
//! tests below exercise each early-return so an unintentionally
//! tightened gate (one that panics or aborts) shows up as a test
//! failure rather than a runtime crash on a stripped scheduler
//! binary.
//!
//! Fixtures are byte arrays built in-test with the
//! [`Elf64Builder`] helper — minimal ELF64 little-endian, only
//! the fields the cast loader inspects (section headers, the
//! shstrtab, `.bpf.objs`, `.BTF`, `.BTF.ext`, `SHF_EXECINSTR`
//! PROGBITS sections, and an optional `.symtab`/`.strtab`
//! pair). The builder produces blobs that pass
//! [`goblin::elf::Elf::parse`].
use super::*;
use crate::monitor::cast_analysis::{AddrSpace, CastHit};
use goblin::elf::header as h;
use goblin::elf::section_header as sh;
use goblin::elf::sym as syms;
use std::io::Write;

// ----- ELF fixture builder ----------------------------------------

/// One section in a synthetic ELF64. Matches the fields the
/// cast loader reads (`sh_type`, `sh_flags`, `sh_addr`,
/// `sh_offset`, `sh_size`) plus a `name` so the builder can
/// own the shstrtab.
struct SecSpec {
    name: &'static str,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    /// Section payload bytes. Empty payload still produces a
    /// section header (e.g. a NULL/SHT_NULL section).
    data: Vec<u8>,
    /// `sh_link` field (for symtab → strtab back-reference, or
    /// for a rel/rela section's symtab back-reference).
    sh_link: u32,
    /// `sh_info` field. For SHT_REL / SHT_RELA sections this is
    /// the index of the section being relocated (per ELF spec).
    /// For SHT_SYMTAB it is one greater than the index of the
    /// last local symbol; we leave it at 0 for tests since no
    /// production code in this module reads SYMTAB sh_info.
    sh_info: u32,
    /// `sh_entsize` field (24 for symtab on ELF64; 16 for SHT_REL,
    /// 24 for SHT_RELA).
    sh_entsize: u64,
}

impl SecSpec {
    fn new(name: &'static str, sh_type: u32) -> Self {
        Self {
            name,
            sh_type,
            sh_flags: 0,
            sh_addr: 0,
            data: Vec::new(),
            sh_link: 0,
            sh_info: 0,
            sh_entsize: 0,
        }
    }
    fn flags(mut self, f: u64) -> Self {
        self.sh_flags = f;
        self
    }
    fn data(mut self, d: Vec<u8>) -> Self {
        self.data = d;
        self
    }
    fn link(mut self, l: u32) -> Self {
        self.sh_link = l;
        self
    }
    fn info(mut self, i: u32) -> Self {
        self.sh_info = i;
        self
    }
    fn entsize(mut self, e: u64) -> Self {
        self.sh_entsize = e;
        self
    }
}

/// Build a synthetic ELF64 little-endian byte blob from a list
/// of [`SecSpec`]s.
///
/// Layout:
/// 1. ELF header at offset 0 (64 bytes).
/// 2. Section data packed back-to-back starting at offset 64.
/// 3. shstrtab (auto-generated) appended after the user data.
/// 4. Section header table appended last.
///
/// A leading `SHT_NULL` section is prepended automatically (the
/// ELF spec mandates `shdr[0]` is null). The shstrtab section is
/// appended automatically and `e_shstrndx` points at it.
fn build_elf64(sections: Vec<SecSpec>, e_machine: u16, e_type: u16) -> Vec<u8> {
    // 1. Build the shstrtab payload up front so each section's
    //    sh_name offset is known.
    let mut shstrtab: Vec<u8> = vec![0u8]; // ELF: index 0 is the empty string.
    let null_name_off = 0u32;
    let mut sec_name_offs: Vec<u32> = Vec::new();
    for s in &sections {
        sec_name_offs.push(shstrtab.len() as u32);
        shstrtab.extend_from_slice(s.name.as_bytes());
        shstrtab.push(0);
    }
    let shstrtab_self_name_off = shstrtab.len() as u32;
    shstrtab.extend_from_slice(b".shstrtab");
    shstrtab.push(0);

    // 2. ELF64 sizes per goblin: SIZEOF_EHDR=64, SIZEOF_SHDR=64.
    let ehdr_size: usize = 64;
    let shdr_size: usize = 64;

    // 3. Lay out section data at growing file offsets after the
    //    header. NULL section (index 0) has zero size and is at
    //    offset 0 (convention).
    let mut data_blob: Vec<u8> = Vec::new();
    let mut sec_file_off: Vec<u64> = Vec::new();
    // Index 0: NULL — placed at offset 0 with size 0.
    sec_file_off.push(0);
    // Indices 1..N: user sections, packed after the ELF header.
    let mut cursor: u64 = ehdr_size as u64;
    for s in &sections {
        sec_file_off.push(cursor);
        data_blob.extend_from_slice(&s.data);
        cursor += s.data.len() as u64;
    }
    // shstrtab section file offset.
    let shstrtab_file_off = cursor;
    data_blob.extend_from_slice(&shstrtab);
    cursor += shstrtab.len() as u64;
    // Section header table file offset.
    let shoff = cursor;

    // 4. Total section count: NULL + user + shstrtab.
    let shnum = (1 + sections.len() + 1) as u16;
    let shstrndx = (1 + sections.len()) as u16;

    // 5. ELF header.
    let mut blob: Vec<u8> = Vec::with_capacity(ehdr_size);
    // e_ident[16]
    blob.extend_from_slice(h::ELFMAG); // \x7FELF
    blob.push(h::ELFCLASS64); // EI_CLASS=2
    blob.push(h::ELFDATA2LSB); // EI_DATA=1
    blob.push(h::EV_CURRENT); // EI_VERSION=1
    blob.push(0); // EI_OSABI=0 (System V)
    blob.push(0); // EI_ABIVERSION
    // EI_PAD: 7 bytes of 0.
    blob.extend_from_slice(&[0u8; 7]);
    // e_type, e_machine, e_version
    blob.extend_from_slice(&e_type.to_le_bytes());
    blob.extend_from_slice(&e_machine.to_le_bytes());
    blob.extend_from_slice(&1u32.to_le_bytes()); // EV_CURRENT=1
    blob.extend_from_slice(&0u64.to_le_bytes()); // e_entry
    blob.extend_from_slice(&0u64.to_le_bytes()); // e_phoff (no program headers)
    blob.extend_from_slice(&shoff.to_le_bytes()); // e_shoff
    blob.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    blob.extend_from_slice(&(ehdr_size as u16).to_le_bytes()); // e_ehsize
    blob.extend_from_slice(&0u16.to_le_bytes()); // e_phentsize
    blob.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    blob.extend_from_slice(&(shdr_size as u16).to_le_bytes()); // e_shentsize
    blob.extend_from_slice(&shnum.to_le_bytes()); // e_shnum
    blob.extend_from_slice(&shstrndx.to_le_bytes()); // e_shstrndx

    // 6. Section data + shstrtab payload.
    blob.extend_from_slice(&data_blob);

    // 7. Section header table.
    let mut write_shdr = |sh_name: u32,
                          sh_type: u32,
                          sh_flags: u64,
                          sh_addr: u64,
                          sh_offset: u64,
                          sh_size: u64,
                          sh_link: u32,
                          sh_info: u32,
                          sh_addralign: u64,
                          sh_entsize: u64| {
        blob.write_all(&sh_name.to_le_bytes()).unwrap();
        blob.write_all(&sh_type.to_le_bytes()).unwrap();
        blob.write_all(&sh_flags.to_le_bytes()).unwrap();
        blob.write_all(&sh_addr.to_le_bytes()).unwrap();
        blob.write_all(&sh_offset.to_le_bytes()).unwrap();
        blob.write_all(&sh_size.to_le_bytes()).unwrap();
        blob.write_all(&sh_link.to_le_bytes()).unwrap();
        blob.write_all(&sh_info.to_le_bytes()).unwrap();
        blob.write_all(&sh_addralign.to_le_bytes()).unwrap();
        blob.write_all(&sh_entsize.to_le_bytes()).unwrap();
    };
    // shdr[0] = NULL.
    write_shdr(null_name_off, sh::SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);
    // User sections.
    for (i, s) in sections.iter().enumerate() {
        write_shdr(
            sec_name_offs[i],
            s.sh_type,
            s.sh_flags,
            s.sh_addr,
            sec_file_off[i + 1],
            s.data.len() as u64,
            s.sh_link,
            s.sh_info,
            1,
            s.sh_entsize,
        );
    }
    // shstrtab section.
    write_shdr(
        shstrtab_self_name_off,
        sh::SHT_STRTAB,
        0,
        0,
        shstrtab_file_off,
        shstrtab.len() as u64,
        0,
        0,
        1,
        0,
    );

    blob
}

/// Build an ELF64 symbol table entry (24 bytes, little-endian).
///
/// Layout per `goblin::elf::sym::sym64::Sym`:
/// `st_name(4) st_info(1) st_other(1) st_shndx(2) st_value(8) st_size(8)`.
fn elf64_sym(st_name: u32, st_info: u8, st_shndx: u16, st_value: u64, st_size: u64) -> [u8; 24] {
    let mut out = [0u8; 24];
    out[0..4].copy_from_slice(&st_name.to_le_bytes());
    out[4] = st_info;
    out[5] = 0; // st_other (visibility) = STV_DEFAULT
    out[6..8].copy_from_slice(&st_shndx.to_le_bytes());
    out[8..16].copy_from_slice(&st_value.to_le_bytes());
    out[16..24].copy_from_slice(&st_size.to_le_bytes());
    out
}

/// Pack the symbol-binding (high 4 bits) and symbol-type (low 4
/// bits) into the `st_info` byte. Mirrors `ELF64_ST_INFO(b,t)`
/// from the SysV ELF spec.
fn st_info(bind: u8, ty: u8) -> u8 {
    (bind << 4) | (ty & 0x0f)
}

// ----- BTF fixture builder ----------------------------------------

/// Build a minimal `.BTF` blob.
///
/// Mirrors the BTF wire format documented in
/// `include/uapi/linux/btf.h`: 24-byte header (magic, version,
/// flags, hdr_len, type_off, type_len, str_off, str_len) +
/// type section + string section. The `types` payload is opaque
/// to these tests — `btf_str_at` only consults the header
/// fields and the string section, so an empty type section is
/// fine.
fn build_btf_blob(types: &[u8], strings: &[u8]) -> Vec<u8> {
    let type_len = types.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.write_all(&0xEB9F_u16.to_le_bytes()).unwrap(); // magic
    blob.push(1); // version
    blob.push(0); // flags
    blob.write_all(&24u32.to_le_bytes()).unwrap(); // hdr_len
    blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // type_len
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off (= type_len)
    blob.write_all(&str_len.to_le_bytes()).unwrap(); // str_len
    blob.extend_from_slice(types);
    blob.extend_from_slice(strings);
    blob
}

// ----- Tests for cached_cast_analysis_for_scheduler error paths --

/// 1. Path that does not exist on the filesystem: the
///    `std::fs::read` arm fires, returns `None`.
#[test]
fn cached_cast_analysis_nonexistent_path_returns_none() {
    let p = std::path::Path::new("/tmp/ktstr-cast-analysis-nonexistent-fixture-path-do-not-create");
    // Sanity: ensure the path really does not exist so the
    // assertion below proves what it claims.
    assert!(
        !p.exists(),
        "fixture path must not exist; remove it before running this test"
    );
    assert!(cached_cast_analysis_for_scheduler(p).is_none());
}

/// 2. Empty file: `goblin::elf::Elf::parse` rejects a 0-byte
///    input; the parse arm fires; empty result collapses to `None`.
#[test]
fn cached_cast_analysis_empty_file_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("empty.bin");
    std::fs::write(&p, b"").expect("write empty file");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// 3. Valid ELF without a `.bpf.objs` section: the section-lookup
///    arm fires, no analysis happens; empty result collapses to
///    `None`.
#[test]
fn cached_cast_analysis_no_bpf_objs_section_returns_none() {
    let blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("no_bpf_objs.elf");
    std::fs::write(&p, &blob).expect("write");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

// ----- Tests for btf_str_at --------------------------------------

/// 4. Empty `btf_bytes`: hits the `< 24` header-length gate.
#[test]
fn btf_str_at_empty_returns_none() {
    assert!(btf_str_at(&[], 0).is_none());
    assert!(btf_str_at(&[0u8; 23], 0).is_none());
}

/// 5. `str_off` past `str_section_len`: the `off >= str_section_len`
///    gate fires.
#[test]
fn btf_str_at_offset_past_strtab_returns_none() {
    // strings: 6 bytes ("\0abc\0\0"); offset 100 is far past.
    let strings = b"\0abc\0\0";
    let blob = build_btf_blob(&[], strings);
    assert!(btf_str_at(&blob, 100).is_none());
}

/// 6. `str_off` exactly at the strtab boundary (= len): the
///    `>=` gate rejects it.
#[test]
fn btf_str_at_offset_at_boundary_returns_none() {
    let strings = b"\0abc\0";
    let blob = build_btf_blob(&[], strings);
    assert!(btf_str_at(&blob, strings.len() as u32).is_none());
}

/// 7. No null terminator in the slice from `base..strtab_end`:
///    the function returns the whole tail as a string. Use a
///    payload that ends without a `\0` to hit the `unwrap_or`
///    branch — the result is still valid UTF-8, exercising the
///    "no null terminator within bounds" path that produces a
///    string instead of `None`. The closer case for `None` is
///    invalid UTF-8 bytes; emit those to confirm `from_utf8`
///    rejection.
#[test]
fn btf_str_at_no_null_terminator_invalid_utf8_returns_none() {
    // Strings: 0xff is not valid UTF-8 as a leading byte and
    // there is no trailing `\0` — `from_utf8` rejects, function
    // returns None.
    let strings = vec![0u8, 0xff, 0xff];
    let blob = build_btf_blob(&[], &strings);
    // str_off=1 points to the first 0xff byte; the slice
    // [base..strtab_end] is `[0xff, 0xff]` (no null), so the
    // `from_utf8` call rejects.
    assert!(btf_str_at(&blob, 1).is_none());
}

/// 8. Valid lookup: returns the expected string.
#[test]
fn btf_str_at_valid_returns_string() {
    let strings = b"\0hello\0world\0";
    let blob = build_btf_blob(&[], strings);
    // Offset 1 = "hello"; offset 7 = "world".
    assert_eq!(btf_str_at(&blob, 1), Some("hello"));
    assert_eq!(btf_str_at(&blob, 7), Some("world"));
    // Offset 0 is the empty string.
    assert_eq!(btf_str_at(&blob, 0), Some(""));
}

// ----- Tests for parse_btf_ext_func_entries ----------------------

/// 9. Data shorter than the minimum 24-byte `.BTF.ext` header:
///    the length gate fires.
#[test]
fn parse_btf_ext_too_short_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    // Build a minimal inner ELF so we can pass &elf to the
    // function (even though we never reach the section walk).
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();
    for short_len in [0usize, 23] {
        let data = vec![0u8; short_len];
        let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
        assert!(out.is_empty(), "len={short_len}");
    }
}

/// 10. Wrong magic: the magic check fires.
#[test]
fn parse_btf_ext_wrong_magic_returns_empty() {
    let mut data = vec![0u8; 24];
    // Magic = 0xDEAD (not 0xEB9F).
    data[0..2].copy_from_slice(&0xDEADu16.to_le_bytes());
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty());
}

/// 11. `hdr_len` below the 24-byte minimum, and `hdr_len` past
///     `data.len()`: both fire the `hdr_len < MIN || hdr_len >
///     data.len()` gate.
#[test]
fn parse_btf_ext_bad_hdr_len_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();

    // (a) hdr_len = 16 (< 24).
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&16u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty(), "hdr_len=16 should be rejected");

    // (b) hdr_len = 1024 (> data.len()).
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&1024u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty(), "hdr_len > data.len should be rejected");
}

/// 12. `func_info_off` + `func_info_len` overflows `data.len()`:
///     the `info_end > data.len()` gate fires.
#[test]
fn parse_btf_ext_func_info_window_oob_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();

    // hdr_len=24, func_info_off=0, func_info_len=10_000;
    // info window runs 24..10024 but data is only 32 bytes.
    let mut data = vec![0u8; 32];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&24u32.to_le_bytes()); // hdr_len
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // func_info_off
    data[12..16].copy_from_slice(&10_000u32.to_le_bytes()); // func_info_len
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty());
}

/// 13. `record_size` < 8: the analyzer requires at least an
///     8-byte `bpf_func_info_min`. Smaller records are rejected.
#[test]
fn parse_btf_ext_record_size_too_small_returns_empty() {
    let btf_bytes = build_btf_blob(&[], b"\0");
    let blob = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bases = HashMap::new();

    // hdr_len=24, func_info_off=0, func_info_len=4 (just the
    // record_size field). record_size=4 < 8 → reject.
    let mut data = vec![0u8; 32];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&24u32.to_le_bytes()); // hdr_len
    data[8..12].copy_from_slice(&0u32.to_le_bytes()); // func_info_off
    data[12..16].copy_from_slice(&8u32.to_le_bytes()); // func_info_len
    // info section starts at offset 24 (hdr_len). Place a
    // record_size of 4 there.
    data[24..28].copy_from_slice(&4u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    assert!(out.is_empty());
}

/// 14. Record with `insn_off` not a multiple of 8: the entry
///     is silently skipped rather than producing a bogus
///     [`FuncEntry`].
///
/// Builds a full valid `.BTF.ext` with one section name pointing
/// at a `.text` PROGBITS+EXECINSTR section, two records — one
/// with `insn_off=8` (valid, kept) and one with `insn_off=12`
/// (not multiple of 8, dropped). Verifies the kept entry has
/// the expected `insn_offset` and the malformed one is absent.
#[test]
fn parse_btf_ext_non_multiple_insn_off_skips_entry() {
    // Build BTF strings with a "txt" entry at offset 1.
    let bytes_strs = b"\0txt\0";
    let btf_bytes = build_btf_blob(&[], bytes_strs);

    // Build inner ELF with a .text section so find_section can
    // resolve "txt"... but the BTF strtab name "txt" must match
    // the ELF section name. So name the section "txt".
    let inner = build_elf64(
        vec![SecSpec::new("txt", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    // The user section "txt" is shdr index 1 (0 is NULL).
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(1, 0);

    // Build the .BTF.ext payload:
    //   header (24 bytes): magic, ver, flags, hdr_len=24,
    //     func_info_off=0, func_info_len=24,
    //     line_info_off=24, line_info_len=0.
    //   info (24 bytes): record_size=8 + 1 sec hdr (8 bytes,
    //     sec_name_off=1 ("txt"), num_info=2) + 2 records of
    //     8 bytes each = 4 + 8 + 16 = 28? Let me recompute:
    //     record_size(4) + sec_hdr(8) + 2*8(16) = 28 bytes.
    // We need func_info_len = 28 then.
    let mut data = Vec::new();
    data.extend_from_slice(&0xEB9F_u16.to_le_bytes()); // magic
    data.push(1); // version
    data.push(0); // flags
    data.extend_from_slice(&24u32.to_le_bytes()); // hdr_len
    data.extend_from_slice(&0u32.to_le_bytes()); // func_info_off
    data.extend_from_slice(&28u32.to_le_bytes()); // func_info_len
    data.extend_from_slice(&28u32.to_le_bytes()); // line_info_off
    data.extend_from_slice(&0u32.to_le_bytes()); // line_info_len
    // func_info data:
    data.extend_from_slice(&8u32.to_le_bytes()); // record_size = 8
    data.extend_from_slice(&1u32.to_le_bytes()); // sec_name_off = "txt"
    data.extend_from_slice(&2u32.to_le_bytes()); // num_info = 2
    // record 0: insn_off=8 (valid; instruction index = 8/8 = 1)
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&42u32.to_le_bytes()); // type_id = 42
    // record 1: insn_off=12 (NOT multiple of 8; skipped)
    data.extend_from_slice(&12u32.to_le_bytes());
    data.extend_from_slice(&99u32.to_le_bytes()); // type_id = 99
    let out = parse_btf_ext_func_entries(&data, &btf_bytes, &elf, &bases);
    // Only the insn_off=8 entry should land.
    assert_eq!(out.len(), 1, "got {out:?}");
    assert_eq!(out[0].insn_offset, 1);
    assert_eq!(out[0].func_proto_id, 42);
}

// ----- Tests for iter_embedded_bpf_objects -----------------------

/// 15. No `STT_OBJECT` symbols pointing into `.bpf.objs`: the
///     fallback branch fires and returns one slice covering the
///     entire section.
#[test]
fn iter_embedded_bpf_objects_no_symbols_falls_back_to_full_section() {
    // Build a scheduler-like ELF: one `.bpf.objs` section, no
    // symbol table at all.
    let payload = b"DUMMY_BPF_OBJ_BYTES".to_vec();
    let payload_len = payload.len();
    let blob = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload)],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    // `.bpf.objs` is at section index 1 (0 = NULL).
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").expect(".bpf.objs");
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1, "expected one fallback slice");
    assert_eq!(out[0].len(), payload_len);
    assert_eq!(out[0], b"DUMMY_BPF_OBJ_BYTES");
}

// ----- Tests for section_data ------------------------------------

/// 16. Section header with `sh_offset + sh_size` overflowing
///     `usize`: `checked_add` returns `None`, function returns
///     `None`.
///
/// Building this through the normal builder is impossible
/// (it always sets a real offset). Instead, we manually patch
/// the section header bytes after construction to set
/// `sh_offset=u64::MAX` and `sh_size=u64::MAX`. Goblin still
/// parses the header successfully; `section_data` then triggers
/// the overflow path.
#[test]
fn section_data_overflow_returns_none() {
    let payload = b"PAYLOAD".to_vec();
    let mut blob = build_elf64(
        vec![SecSpec::new(".x", sh::SHT_PROGBITS).data(payload)],
        h::EM_X86_64,
        h::ET_REL,
    );
    // Patch shdr[1] (".x") sh_offset and sh_size to u64::MAX so
    // the `start.checked_add(size)` overflows. shdr table is at
    // the end of the file; each shdr is 64 bytes; shdr[0] is
    // NULL, so shdr[1] starts at e_shoff+64.
    let elf_view = goblin::elf::Elf::parse(&blob).unwrap();
    let shoff = elf_view.header.e_shoff as usize;
    let shdr1_off = shoff + 64;
    // sh_offset is at byte 24 within the 64-byte ELF64 shdr;
    // sh_size is at byte 32.
    blob[shdr1_off + 24..shdr1_off + 32].copy_from_slice(&u64::MAX.to_le_bytes());
    blob[shdr1_off + 32..shdr1_off + 40].copy_from_slice(&u64::MAX.to_le_bytes());

    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let idx = find_section(&elf, ".x").expect(".x");
    assert!(section_data(&elf, &blob, idx).is_none());
}

// ----- Tests for merge_into --------------------------------------

/// 17. Conflicting entries (same key, different value) collapse
///     to "drop the key" — false negatives are the safe direction.
#[test]
fn merge_into_conflicting_entries_drop_key() {
    let mut into = CastMap::new();
    into.insert(
        (10, 0),
        CastHit {
            target_type_id: 100,
            addr_space: AddrSpace::Arena,
        },
    );
    into.insert(
        (10, 8),
        CastHit {
            target_type_id: 200,
            addr_space: AddrSpace::Kernel,
        },
    );

    let mut from = CastMap::new();
    // Same (parent, offset) but different target id → drop.
    from.insert(
        (10, 0),
        CastHit {
            target_type_id: 101,
            addr_space: AddrSpace::Arena,
        },
    );
    // Same (parent, offset) but different AddrSpace → drop.
    from.insert(
        (10, 8),
        CastHit {
            target_type_id: 200,
            addr_space: AddrSpace::Arena,
        },
    );
    // Disjoint key → kept.
    from.insert(
        (10, 16),
        CastHit {
            target_type_id: 300,
            addr_space: AddrSpace::Kernel,
        },
    );
    // Identical key+value → kept.
    from.insert(
        (20, 0),
        CastHit {
            target_type_id: 400,
            addr_space: AddrSpace::Arena,
        },
    );

    merge_into(&mut into, from);

    // Conflicting keys are gone.
    assert!(!into.contains_key(&(10, 0)));
    assert!(!into.contains_key(&(10, 8)));
    // Disjoint key from `from` is now in `into`.
    assert_eq!(
        into.get(&(10, 16)),
        Some(&CastHit {
            target_type_id: 300,
            addr_space: AddrSpace::Kernel,
        })
    );
    // Identical merge keeps the value.
    assert_eq!(
        into.get(&(20, 0)),
        Some(&CastHit {
            target_type_id: 400,
            addr_space: AddrSpace::Arena,
        })
    );
}

/// Sanity: the unused-helper escape valves (`elf64_sym`,
/// `st_info`) are exercised by a smoke build of a symbol table
/// to keep them from rotting if a future test wants them. The
/// goblin parser must accept the symtab/strtab pair.
#[test]
fn smoke_symtab_helpers_compile() {
    // Build .strtab content: "\0bpf_obj\0".
    let strtab = b"\0bpf_obj\0".to_vec();
    // Single STT_OBJECT symbol named "bpf_obj" pointing at
    // the (theoretical) `.bpf.objs` section index 1.
    let mut symtab = Vec::new();
    // shdr[0] = NULL — the first entry of a symtab is reserved.
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        1, // st_name: offset of "bpf_obj" in .strtab
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1, // st_shndx
        0, // st_value
        8, // st_size
    ));

    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(vec![0u8; 8]),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2) // sh_link → strtab is user-section index 2 = shdr index 3? wait
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    // sh_link must reference the actual shdr index of the
    // strtab. shdr[0]=NULL, [1]=.bpf.objs, [2]=.strtab,
    // [3]=.symtab, [4]=.shstrtab. So sh_link should be 2.
    // We passed link(2) above, which matches.
    let _ = goblin::elf::Elf::parse(&blob).expect("parse");
    // The parser-level smoke completed; nothing further to
    // assert here — this test exists so the helpers stay in
    // active use.
}

// ----- Tests for find_section ------------------------------------

/// Happy path: `find_section` resolves an existing section by
/// name and returns the matching shdr index.
#[test]
fn find_section_locates_named_section() {
    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into()),
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(vec![0u8; 4]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    // shdr[0]=NULL, [1]=.text, [2]=.bpf.objs, [3]=.shstrtab.
    assert_eq!(find_section(&elf, ".text"), Some(1));
    assert_eq!(find_section(&elf, ".bpf.objs"), Some(2));
}

/// `find_section` returns `None` for a name that does not match
/// any section.
#[test]
fn find_section_missing_returns_none() {
    let blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    assert_eq!(find_section(&elf, ".nope"), None);
}

// ----- Tests for section_data happy path -------------------------

/// `section_data` returns the byte slice covering a known
/// section's `[sh_offset, sh_offset + sh_size)` range.
#[test]
fn section_data_returns_section_bytes() {
    let payload = b"section-bytes-payload-12345".to_vec();
    let payload_len = payload.len();
    let blob = build_elf64(
        vec![SecSpec::new(".x", sh::SHT_PROGBITS).data(payload)],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let idx = find_section(&elf, ".x").unwrap();
    let bytes = section_data(&elf, &blob, idx).expect("payload slice");
    assert_eq!(bytes.len(), payload_len);
    assert_eq!(bytes, &b"section-bytes-payload-12345"[..]);
}

/// Out-of-range section index returns `None`.
#[test]
fn section_data_out_of_range_returns_none() {
    let blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS)],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    assert!(section_data(&elf, &blob, 9999).is_none());
}

// ----- iter_embedded_bpf_objects symbol-driven path --------------

/// Symbol-driven path: a single `STT_OBJECT` symbol pointing
/// into `.bpf.objs` produces one slice covering exactly the
/// range `[st_value, st_value + st_size)`.
#[test]
fn iter_embedded_bpf_objects_uses_object_symbol() {
    let payload: Vec<u8> = (0..32u8).collect();
    let strtab = b"\0bpf_obj\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,  // st_shndx — .bpf.objs at shdr[1]
        4,  // st_value: byte offset within .bpf.objs (sh_addr=0)
        24, // st_size
    ));
    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").unwrap();
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), 24);
    let expected: Vec<u8> = (4..28u8).collect();
    assert_eq!(out[0], expected.as_slice());
}

/// Symbol whose `st_value + st_size` exceeds the section bounds
/// is rejected; the iterator falls back to the full section.
#[test]
fn iter_embedded_bpf_objects_rejects_oversized_symbol() {
    let payload = b"0123456789abcdef".to_vec(); // 16 bytes
    let payload_len = payload.len();
    let strtab = b"\0bpf_obj\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    // st_size=200 vs section size=16 → reject → fallback fires.
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,
        0,
        200,
    ));
    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").unwrap();
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1, "fallback yields exactly one slice");
    assert_eq!(out[0].len(), payload_len);
}

/// Symbol whose `st_type` is `STT_FUNC` (not `STT_OBJECT`) is
/// skipped — iterator falls back to the full section.
#[test]
fn iter_embedded_bpf_objects_skips_non_object_symbols() {
    let payload = b"hello-bpf-objects".to_vec();
    let payload_len = payload.len();
    let strtab = b"\0func_sym\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1,
        0,
        8,
    ));
    let blob = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(payload),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).unwrap();
    let bpf_objs_idx = find_section(&elf, ".bpf.objs").unwrap();
    let out = iter_embedded_bpf_objects(&elf, &blob, bpf_objs_idx);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].len(), payload_len);
}

// ----- BPF instruction encoding helpers --------------------------
//
// `BpfInsn` exposes [`BpfInsn::new`] and [`BpfInsn::from_le_bytes`]
// but not a writer. The end-to-end tests below need wire bytes to
// populate `.text` sections, so we re-encode by mirroring the
// little-endian layout the decoder reads.
fn insn_to_bytes(i: BpfInsn) -> [u8; 8] {
    // `regs` field is private in production; rebuild the packed
    // byte from `dst_reg()` (low 4 bits) and `src_reg()` (high
    // 4 bits) — exactly the layout `BpfInsn::new` produces.
    let regs_byte = (i.dst_reg() & 0x0f) | ((i.src_reg() & 0x0f) << 4);
    let mut buf = [0u8; 8];
    buf[0] = i.code;
    buf[1] = regs_byte;
    buf[2..4].copy_from_slice(&i.off.to_le_bytes());
    buf[4..8].copy_from_slice(&i.imm.to_le_bytes());
    buf
}

fn insns_to_text_bytes(insns: &[BpfInsn]) -> Vec<u8> {
    let mut out = Vec::with_capacity(insns.len() * 8);
    for ins in insns {
        out.extend_from_slice(&insn_to_bytes(*ins));
    }
    out
}

// BPF opcode field values (kernel uapi `bpf.h`):
//   class low 3 bits: LDX=1, JMP=5
//   size bits 3..4: DW=0x18
//   mode bits 5..7: MEM=0x60
//   op  bits 4..7: EXIT=0x90
const OP_LDX_DW_MEM: u8 = 0x01 | 0x18 | 0x60; // 0x79
const OP_JMP_EXIT: u8 = 0x05 | 0x90; // 0x95

fn ldx_dw_mem(dst: u8, src: u8, off: i16) -> BpfInsn {
    BpfInsn::new(OP_LDX_DW_MEM, dst, src, off, 0)
}

fn exit_insn() -> BpfInsn {
    BpfInsn::new(OP_JMP_EXIT, 0, 0, 0, 0)
}

/// `BPF_ADDR_SPACE_CAST` for the F1 mitigation: ALU64 | MOV | X
/// with `off=1, imm=1` is the as(1)→as(0) (arena→kernel) cast
/// the analyzer treats as `arena_confirmed` evidence on the
/// source's `(struct, field_offset)` slot.
fn addr_space_cast_insn(dst: u8, src: u8) -> BpfInsn {
    use libbpf_rs::libbpf_sys as bs;
    let code = (bs::BPF_ALU64 | bs::BPF_MOV | bs::BPF_X) as u8;
    BpfInsn::new(code, dst, src, 1, 1)
}

// ----- Synthesizers for full BTF (ints, structs, ptr, FuncProto, Func)
//
// The error-path tests above only need empty BTFs. The
// analyze_one_object_with_btf end-to-end test needs a real BTF
// whose types the analyzer can intersect. This builder mirrors
// `cast_analysis::tests::build_btf` in shape but is local to this
// module so the two test fixtures stay decoupled.
const SYN_BTF_KIND_INT: u32 = 1;
const SYN_BTF_KIND_PTR: u32 = 2;
const SYN_BTF_KIND_STRUCT: u32 = 4;
const SYN_BTF_KIND_FWD: u32 = 7;
const SYN_BTF_KIND_FUNC: u32 = 12;
const SYN_BTF_KIND_FUNC_PROTO: u32 = 13;

/// Append `name` plus a trailing NUL to `s`; return the offset
/// at which it was written. Standard BTF strtab convention.
fn push_btf_name(s: &mut Vec<u8>, name: &str) -> u32 {
    let off = s.len() as u32;
    s.extend_from_slice(name.as_bytes());
    s.push(0);
    off
}

/// Member of a synthetic struct (non-bitfield, byte-aligned).
#[derive(Clone, Copy)]
struct SynMember {
    name_off: u32,
    type_id: u32,
    byte_offset: u32,
}

/// FuncProto parameter record.
#[derive(Clone, Copy)]
struct SynParam {
    name_off: u32,
    type_id: u32,
}

enum SynKind {
    Int {
        name_off: u32,
        size: u32,
        encoding: u32,
        offset: u32,
        bits: u32,
    },
    Ptr {
        type_id: u32,
    },
    Struct {
        name_off: u32,
        size: u32,
        members: Vec<SynMember>,
    },
    /// `BTF_KIND_FWD` (kind 7) — forward declaration with no body.
    /// Layout per `include/uapi/linux/btf.h` is the 12-byte common
    /// `struct btf_type` header (name_off + info + size_type) with
    /// no trailing payload. `kind_flag` (bit 31 of info) is 0 for
    /// struct-kind, 1 for union-kind; `Fwd::is_struct()` /
    /// `Fwd::is_union()` in btf-rs read it back. The third u32 of
    /// the common header is unused for Fwd and emitted as 0.
    Fwd {
        name_off: u32,
        kind_flag: u32,
    },
    Func {
        name_off: u32,
        type_id: u32,
        linkage: u32,
    },
    FuncProto {
        return_type_id: u32,
        params: Vec<SynParam>,
    },
}

/// Encode `types` and `strings` into a BTF byte blob.
fn build_btf_full(types: &[SynKind], strings: &[u8]) -> Vec<u8> {
    let mut type_section = Vec::new();
    for ty in types {
        match ty {
            SynKind::Int {
                name_off,
                size,
                encoding,
                offset,
                bits,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = (SYN_BTF_KIND_INT << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                let int_data = (*encoding << 24) | ((*offset & 0xff) << 16) | (*bits & 0xff);
                type_section.extend_from_slice(&int_data.to_le_bytes());
            }
            SynKind::Ptr { type_id } => {
                type_section.extend_from_slice(&0u32.to_le_bytes());
                let info = (SYN_BTF_KIND_PTR << 24) & 0x1f00_0000;
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynKind::Struct {
                name_off,
                size,
                members,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let vlen = members.len() as u32;
                let info = ((SYN_BTF_KIND_STRUCT << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&size.to_le_bytes());
                for m in members {
                    type_section.extend_from_slice(&m.name_off.to_le_bytes());
                    type_section.extend_from_slice(&m.type_id.to_le_bytes());
                    let bit_off = m.byte_offset * 8;
                    type_section.extend_from_slice(&bit_off.to_le_bytes());
                }
            }
            SynKind::Fwd {
                name_off,
                kind_flag,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                // Fwd info: kind in bits 24-28, kind_flag in bit
                // 31 (struct-vs-union tag), no vlen. The third u32
                // of btf_type (size_type) is unused for Fwd.
                let info = ((SYN_BTF_KIND_FWD << 24) & 0x1f00_0000) | ((*kind_flag & 1) << 31);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&0u32.to_le_bytes());
            }
            SynKind::Func {
                name_off,
                type_id,
                linkage,
            } => {
                type_section.extend_from_slice(&name_off.to_le_bytes());
                let info = ((SYN_BTF_KIND_FUNC << 24) & 0x1f00_0000) | (*linkage & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&type_id.to_le_bytes());
            }
            SynKind::FuncProto {
                return_type_id,
                params,
            } => {
                type_section.extend_from_slice(&0u32.to_le_bytes());
                let vlen = params.len() as u32;
                let info = ((SYN_BTF_KIND_FUNC_PROTO << 24) & 0x1f00_0000) | (vlen & 0xffff);
                type_section.extend_from_slice(&info.to_le_bytes());
                type_section.extend_from_slice(&return_type_id.to_le_bytes());
                for p in params {
                    type_section.extend_from_slice(&p.name_off.to_le_bytes());
                    type_section.extend_from_slice(&p.type_id.to_le_bytes());
                }
            }
        }
    }
    // Header.
    let type_len = type_section.len() as u32;
    let str_len = strings.len() as u32;
    let mut blob = Vec::new();
    blob.write_all(&0xEB9F_u16.to_le_bytes()).unwrap();
    blob.push(1); // version
    blob.push(0); // flags
    blob.write_all(&24u32.to_le_bytes()).unwrap(); // hdr_len
    blob.write_all(&0u32.to_le_bytes()).unwrap(); // type_off
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // type_len
    blob.write_all(&type_len.to_le_bytes()).unwrap(); // str_off
    blob.write_all(&str_len.to_le_bytes()).unwrap();
    blob.extend_from_slice(&type_section);
    blob.extend_from_slice(strings);
    blob
}

/// Build a `.BTF.ext` blob describing one `func_info` section
/// with `records` entries `(insn_off, type_id)`.
fn build_btf_ext(section_name_off: u32, records: &[(u32, u32)], record_size: u32) -> Vec<u8> {
    let header_len = 24u32;
    let info_len = 4 + 4 + 4 + records.len() as u32 * record_size;
    let mut info = Vec::new();
    info.extend_from_slice(&record_size.to_le_bytes());
    info.extend_from_slice(&section_name_off.to_le_bytes());
    info.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for (insn_off, type_id) in records {
        info.extend_from_slice(&insn_off.to_le_bytes());
        info.extend_from_slice(&type_id.to_le_bytes());
        let pad = record_size.saturating_sub(8) as usize;
        info.extend(std::iter::repeat_n(0, pad));
    }
    let mut out = Vec::new();
    out.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    out.push(1);
    out.push(0);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // func_info_off
    out.extend_from_slice(&info_len.to_le_bytes());
    out.extend_from_slice(&info_len.to_le_bytes()); // line_info_off (unused)
    out.extend_from_slice(&0u32.to_le_bytes()); // line_info_len
    out.extend_from_slice(&info);
    out
}

/// Construct a BPF object ELF with `.text`, `.BTF`, and
/// `.BTF.ext` sections — the canonical scx-built shape minus
/// the relocations the loader does not consume.
fn build_full_bpf_object_elf(text: Vec<u8>, btf: Vec<u8>, btf_ext: Vec<u8>) -> Vec<u8> {
    build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf),
            SecSpec::new(".BTF.ext", sh::SHT_PROGBITS).data(btf_ext),
        ],
        h::EM_BPF,
        h::ET_REL,
    )
}

// ----- analyze_one_object_with_btf error paths -------------------

/// Inner ELF whose bytes do not start with a valid ELF magic
/// fails goblin parse — `analyze_one_object_with_btf` returns
/// empty.
#[test]
fn analyze_one_object_corrupt_elf_returns_empty() {
    let bytes = vec![0u8; 64]; // all zeros — bad ELF magic
    let (map, btf) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_none());
}

/// Inner ELF without a `.BTF` section returns an empty map and
/// no parsed BTF.
#[test]
fn analyze_one_object_no_btf_returns_empty() {
    let bytes = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 8]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_none());
}

/// Inner ELF whose `.BTF` bytes do not parse as valid BTF
/// returns empty.
#[test]
fn analyze_one_object_corrupt_btf_returns_empty() {
    let bytes = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(insns_to_text_bytes(&[exit_insn()])),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(vec![0xFFu8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_none());
}

/// Inner ELF with valid BTF but no executable text section
/// produces no instructions to analyze → empty map. The parsed
/// BTF is still returned so its struct/union definitions can
/// feed the cross-BTF Fwd index.
#[test]
fn analyze_one_object_no_text_section_returns_empty() {
    let bytes = build_elf64(
        vec![SecSpec::new(".BTF", sh::SHT_PROGBITS).data(build_btf_blob(&[], b"\0"))],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_some());
}

/// Text section whose byte length is not a multiple of 8 is
/// skipped during decode → empty map. As with the no-text case,
/// the parsed BTF is still returned for cross-BTF Fwd indexing.
#[test]
fn analyze_one_object_misaligned_text_skipped() {
    let bytes = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 7]),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(build_btf_blob(&[], b"\0")),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let (map, btf) = analyze_one_object_with_btf(&bytes);
    assert!(map.is_empty());
    assert!(btf.is_some());
}

// ----- analyze_one_object_with_btf end-to-end recovery -----------

/// Full pipeline: BTF describes T (id=2) with a u64 field at
/// offset 8 and Q (id=3) with a u64 field at offset 0; .text
/// contains a function entry that loads T.f then dereferences
/// it as Q*; .BTF.ext seeds R1=*T at the entry. Expected:
/// CastMap maps `(2, 8) → CastHit { 3, Arena }`.
#[test]
fn analyze_one_object_recovers_arena_cast_end_to_end() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_t = push_btf_name(&mut strings, "T");
    let n_q = push_btf_name(&mut strings, "Q");
    let n_f = push_btf_name(&mut strings, "f");
    let n_x = push_btf_name(&mut strings, "x");
    let n_func = push_btf_name(&mut strings, "myfunc");
    let n_text = push_btf_name(&mut strings, ".text");
    // id=1 u64, id=2 T, id=3 Q, id=4 *T, id=5 FuncProto(*T),
    // id=6 Func(myfunc@5).
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynKind::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::Ptr { type_id: 2 },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 4,
            }],
        },
        SynKind::Func {
            name_off: n_func,
            type_id: 5,
            linkage: 1,
        },
    ];
    let btf_blob = build_btf_full(&types, &strings);
    // r2 = *(u64 *)(r1 + 8); r2 = arena_cast(r2);
    // r3 = *(u64 *)(r2 + 0); exit.
    // The arena_cast adds (T, 8) to arena_confirmed (F1
    // mitigation prerequisite for the shape-inference finding).
    let insns = vec![
        ldx_dw_mem(2, 1, 8),
        addr_space_cast_insn(2, 2),
        ldx_dw_mem(3, 2, 0),
        exit_insn(),
    ];
    let text = insns_to_text_bytes(&insns);
    let btf_ext = build_btf_ext(n_text, &[(0, 5)], 8);

    let bytes = build_full_bpf_object_elf(text, btf_blob, btf_ext);
    let (map, btf) = analyze_one_object_with_btf(&bytes);
    assert!(btf.is_some(), "valid BTF must be returned");
    let hit = map.get(&(2u32, 8u32)).copied();
    assert_eq!(
        hit,
        Some(CastHit {
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
        "expected arena cast T.f → Q*, got {map:?}"
    );
}

// ----- cached_cast_analysis_for_scheduler error & happy paths ---

/// Outer ELF that parses successfully but whose `.bpf.objs`
/// bytes are not a valid inner ELF — outer merge is empty,
/// cache layer collapses to `None`.
#[test]
fn cached_cast_analysis_corrupt_inner_returns_none() {
    let outer = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(b"not-an-elf".to_vec())],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("bad_inner.bin");
    std::fs::write(&p, &outer).expect("write");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// Outer ELF whose `.bpf.objs` carries an inner BPF ELF
/// without a `.BTF` section — outer merge is empty, cache
/// layer collapses to `None`.
#[test]
fn cached_cast_analysis_inner_without_btf_returns_none() {
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 8]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let outer = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(inner)],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("no_inner_btf.bin");
    std::fs::write(&p, &outer).expect("write");
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// Full end-to-end through the public driver: outer host ELF
/// wraps an inner BPF ELF that recovers an arena cast.
#[test]
fn cached_cast_analysis_recovers_arena_cast_end_to_end() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_t = push_btf_name(&mut strings, "T");
    let n_q = push_btf_name(&mut strings, "Q");
    let n_f = push_btf_name(&mut strings, "f");
    let n_x = push_btf_name(&mut strings, "x");
    let n_func = push_btf_name(&mut strings, "myfunc");
    let n_text = push_btf_name(&mut strings, ".text");
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynKind::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::Ptr { type_id: 2 },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 4,
            }],
        },
        SynKind::Func {
            name_off: n_func,
            type_id: 5,
            linkage: 1,
        },
    ];
    let btf_blob = build_btf_full(&types, &strings);
    // F1 mitigation: include arena_space_cast on r2 so the
    // shape-inference finding emits.
    let insns = vec![
        ldx_dw_mem(2, 1, 8),
        addr_space_cast_insn(2, 2),
        ldx_dw_mem(3, 2, 0),
        exit_insn(),
    ];
    let text = insns_to_text_bytes(&insns);
    let btf_ext = build_btf_ext(n_text, &[(0, 5)], 8);

    let inner = build_full_bpf_object_elf(text, btf_blob, btf_ext);
    let outer = build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(inner)],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("full.bin");
    std::fs::write(&p, &outer).expect("write");

    let out = cached_cast_analysis_for_scheduler(&p).expect("non-empty fixture must produce Some");
    let hit = out.cast_map.get(&(2u32, 8u32)).copied();
    assert_eq!(
        hit,
        Some(CastHit {
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
        "expected arena cast T.f → Q*, got {:?}",
        out.cast_map
    );
}

// ----- Tests for cached_cast_analysis_for_scheduler --------------

/// Helper: build the same arena-cast end-to-end fixture used by
/// `cached_cast_analysis_recovers_arena_cast_end_to_end`,
/// returning the outer ELF bytes. Centralised so cache tests
/// share a fixture shape with the path-driven test.
fn build_recovers_arena_cast_outer_elf() -> Vec<u8> {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_t = push_btf_name(&mut strings, "T");
    let n_q = push_btf_name(&mut strings, "Q");
    let n_f = push_btf_name(&mut strings, "f");
    let n_x = push_btf_name(&mut strings, "x");
    let n_func = push_btf_name(&mut strings, "myfunc");
    let n_text = push_btf_name(&mut strings, ".text");
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_t,
            size: 16,
            members: vec![SynMember {
                name_off: n_f,
                type_id: 1,
                byte_offset: 8,
            }],
        },
        SynKind::Struct {
            name_off: n_q,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::Ptr { type_id: 2 },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 4,
            }],
        },
        SynKind::Func {
            name_off: n_func,
            type_id: 5,
            linkage: 1,
        },
    ];
    let btf_blob = build_btf_full(&types, &strings);
    // F1 mitigation: include arena_space_cast on r2 so the
    // shape-inference finding emits.
    let insns = vec![
        ldx_dw_mem(2, 1, 8),
        addr_space_cast_insn(2, 2),
        ldx_dw_mem(3, 2, 0),
        exit_insn(),
    ];
    let text = insns_to_text_bytes(&insns);
    let btf_ext = build_btf_ext(n_text, &[(0, 5)], 8);
    let inner = build_full_bpf_object_elf(text, btf_blob, btf_ext);
    build_elf64(
        vec![SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(inner)],
        h::EM_X86_64,
        h::ET_REL,
    )
}

/// Cache hit by content: two calls on the same bytes (different
/// paths) return the same `Arc<CastAnalysisOutput>`. Proves the
/// cache is content-keyed (SHA-256), not path-keyed.
#[test]
fn cached_cast_analysis_returns_same_arc_for_same_content() {
    let blob = build_recovers_arena_cast_outer_elf();
    let dir = tempfile::tempdir().expect("tempdir");
    let p1 = dir.path().join("first.bin");
    let p2 = dir.path().join("second.bin");
    std::fs::write(&p1, &blob).expect("write 1");
    std::fs::write(&p2, &blob).expect("write 2");

    let first = cached_cast_analysis_for_scheduler(&p1).expect("Some on non-empty analysis");
    let second = cached_cast_analysis_for_scheduler(&p2).expect("cache hit on identical content");

    assert!(
        Arc::ptr_eq(&first, &second),
        "expected pointer-equal Arc when two paths have identical content"
    );
    // Sanity: the cached output carries the recovered cast.
    assert_eq!(
        first.cast_map.get(&(2u32, 8u32)).copied(),
        Some(CastHit {
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
    );
}

/// Cache miss by content: an empty-result blob caches as
/// `None`. Proves the empty-result collapse (cast_map empty
/// AND fwd_index empty) is preserved across cache lookups.
#[test]
fn cached_cast_analysis_collapses_empty_to_none() {
    let empty_blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_X86_64,
        h::ET_REL,
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("empty.bin");
    std::fs::write(&p, &empty_blob).expect("write");

    // First call analyzes; result is empty → None.
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
    // Second call hits the same content-hash cache entry and
    // also resolves to None without re-running the analyzer.
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());
}

/// Read-failure path: a non-existent path produces `None`
/// without polluting the cache. A later call after the file
/// appears must succeed and run the analyzer on demand.
#[test]
fn cached_cast_analysis_read_failure_does_not_pollute_cache() {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("appears_later.bin");

    assert!(!p.exists());
    assert!(cached_cast_analysis_for_scheduler(&p).is_none());

    let blob = build_recovers_arena_cast_outer_elf();
    std::fs::write(&p, &blob).expect("write");
    let out = cached_cast_analysis_for_scheduler(&p)
        .expect("post-creation read should succeed and produce a non-empty CastAnalysisOutput");
    assert_eq!(
        out.cast_map.get(&(2u32, 8u32)).copied(),
        Some(CastHit {
            target_type_id: 3,
            addr_space: AddrSpace::Arena,
        }),
        "post-creation analysis should recover the seeded cast"
    );
}

/// Lazy wrapper: `LazyCastMap::new` runs no analysis. The
/// `OnceLock` is empty until `.get_full()` fires, and
/// `.get_full()` returns identical `Arc`s on every subsequent
/// call (the analyzer ran exactly once).
#[test]
fn lazy_cast_map_get_full_is_idempotent_and_lazy() {
    let blob = build_recovers_arena_cast_outer_elf();
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("lazy.bin");
    std::fs::write(&p, &blob).expect("write");

    let lazy = LazyCastMap::new(Some(p.clone()));
    // Sanity: the lazy slot is empty before any `.get_full()`.
    assert!(
        lazy.inner.get().is_none(),
        "LazyCastMap::new must not run analysis"
    );

    let first = lazy.get_full().expect("non-empty result");
    let second = lazy.get_full().expect("non-empty result");
    assert!(
        Arc::ptr_eq(&first, &second),
        "OnceLock-backed `.get_full()` must return the same Arc on every call"
    );
}

/// `LazyCastMap::get_full` on a binary with no recoverable
/// casts returns `None` (the cache layer collapses empty
/// results). The renderer treats `None` identically to an
/// empty map, so this keeps the pre-integration default
/// behaviour intact.
#[test]
fn lazy_cast_map_get_full_returns_none_for_no_findings() {
    let empty_blob = build_elf64(
        vec![SecSpec::new(".text", sh::SHT_PROGBITS).flags(sh::SHF_EXECINSTR.into())],
        h::EM_X86_64,
        h::ET_REL,
    );
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("no_findings.bin");
    std::fs::write(&p, &empty_blob).expect("write");

    let lazy = LazyCastMap::new(Some(p));
    assert!(
        lazy.get_full().is_none(),
        "no-`.bpf.objs` binary must collapse to None on `.get_full()`"
    );
}

// ----- parse_btf_ext_func_entries happy paths --------------------

/// Records produce one [`FuncEntry`] each, with `insn_offset`
/// measured in instruction indices (byte offset / 8) plus the
/// section base supplied by the caller.
#[test]
fn parse_btf_ext_records_produce_func_entries() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);

    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let text_idx = find_section(&elf, ".text").expect(".text") as u32;
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(text_idx, 0);

    let data = build_btf_ext(n_text, &[(0, 11), (16, 22)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert_eq!(out.len(), 2, "got {out:?}");
    assert_eq!(out[0].insn_offset, 0);
    assert_eq!(out[0].func_proto_id, 11);
    assert_eq!(out[1].insn_offset, 2);
    assert_eq!(out[1].func_proto_id, 22);
}

/// Record offsets are measured relative to the section's base
/// in the concatenated text stream.
#[test]
fn parse_btf_ext_applies_section_base_offset() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let text_idx = find_section(&elf, ".text").expect(".text") as u32;
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(text_idx, 10);
    let data = build_btf_ext(n_text, &[(16, 5)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].insn_offset, 12);
    assert_eq!(out[0].func_proto_id, 5);
}

/// `record_size` larger than the minimum 8 bytes means
/// trailing padding the parser must skip.
#[test]
fn parse_btf_ext_handles_padded_records() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let text_idx = find_section(&elf, ".text").expect(".text") as u32;
    let mut bases: HashMap<u32, usize> = HashMap::new();
    bases.insert(text_idx, 0);
    let data = build_btf_ext(n_text, &[(0, 11), (8, 22)], 16);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].insn_offset, 0);
    assert_eq!(out[0].func_proto_id, 11);
    assert_eq!(out[1].insn_offset, 1);
    assert_eq!(out[1].func_proto_id, 22);
}

/// `sec_name_off` that does not resolve in the BTF strtab
/// causes records to be silently skipped.
#[test]
fn parse_btf_ext_skips_unresolvable_section_name() {
    let strings = vec![0u8];
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases: HashMap<u32, usize> = HashMap::new();
    let data = build_btf_ext(999, &[(0, 7)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

/// `sec_name_off` resolves to a name that does not match any
/// ELF section — records are skipped.
#[test]
fn parse_btf_ext_skips_section_not_in_elf() {
    let mut strings = vec![0u8];
    let n_other = push_btf_name(&mut strings, ".not_in_elf");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases: HashMap<u32, usize> = HashMap::new();
    let data = build_btf_ext(n_other, &[(0, 7)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

/// ELF section exists but `section_bases` lacks an entry —
/// records skipped.
#[test]
fn parse_btf_ext_skips_section_without_base() {
    let mut strings = vec![0u8];
    let n_text = push_btf_name(&mut strings, ".text");
    let btf_blob = build_btf_full(&[], &strings);
    let inner = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(vec![0u8; 32]),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases: HashMap<u32, usize> = HashMap::new();
    let data = build_btf_ext(n_text, &[(0, 7)], 8);
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

/// `func_info_len` of zero short-circuits the record loop.
#[test]
fn parse_btf_ext_zero_func_info_len_returns_empty() {
    let btf_blob = build_btf_full(&[], b"\0");
    let inner = build_elf64(vec![], h::EM_BPF, h::ET_REL);
    let elf = goblin::elf::Elf::parse(&inner).unwrap();
    let bases = HashMap::new();
    let mut data = vec![0u8; 24];
    data[0..2].copy_from_slice(&0xEB9F_u16.to_le_bytes());
    data[4..8].copy_from_slice(&24u32.to_le_bytes());
    let out = parse_btf_ext_func_entries(&data, &btf_blob, &elf, &bases);
    assert!(out.is_empty());
}

// ----- kfunc-relocation patcher tests ----------------------------
//
// Coverage strategy: for `patch_kfunc_calls` we run end-to-end
// tests that synthesize a complete BPF object (program text +
// BTF + .symtab/.strtab + a SHT_REL section) and feed the
// patcher the same `(text_concat, btf, elf, section_bases)`
// tuple `analyze_one_object_with_btf` produces. After patching we re-
// decode the call instruction and assert the analyzer-visible
// state — this is the exact contract `analyze_casts` consumes
// downstream, so any drift between the patcher and the analyzer
// surfaces here.

/// Encode a single BTF type record header. Mirrors the wire
/// format from linux uapi `btf.h`:
/// `name_off(4) info(4) size_or_type(4)`.
fn kfunc_btf_type_header(name_off: u32, kind: u32, vlen: u32, size_or_type: u32) -> [u8; 12] {
    let info = ((kind << 24) & 0x1f00_0000) | (vlen & 0xffff);
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&name_off.to_le_bytes());
    out[4..8].copy_from_slice(&info.to_le_bytes());
    out[8..12].copy_from_slice(&size_or_type.to_le_bytes());
    out
}

/// Build a minimal `.BTF` blob containing a single extern FUNC
/// with a FuncProto that returns a struct pointer.
/// Returns the byte blob plus the BTF id of the extern Func
/// (always 5) and the BTF id of struct T (always 2).
fn build_kfunc_btf_blob(kf_name: &str) -> (Vec<u8>, u32, u32) {
    let mut strings: Vec<u8> = vec![0];
    let push_name = |s: &mut Vec<u8>, name: &str| -> u32 {
        let off = s.len() as u32;
        s.extend_from_slice(name.as_bytes());
        s.push(0);
        off
    };
    let n_u64 = push_name(&mut strings, "u64");
    let n_t = push_name(&mut strings, "T");
    let n_x = push_name(&mut strings, "x");
    let n_func = push_name(&mut strings, kf_name);

    let mut types: Vec<u8> = Vec::new();
    const BTF_KIND_INT: u32 = 1;
    const BTF_KIND_PTR: u32 = 2;
    const BTF_KIND_STRUCT: u32 = 4;
    const BTF_KIND_FUNC: u32 = 12;
    const BTF_KIND_FUNC_PROTO: u32 = 13;
    const BTF_FUNC_EXTERN: u32 = 2;

    // id 1: BTF_KIND_INT u64.
    types.extend_from_slice(&kfunc_btf_type_header(n_u64, BTF_KIND_INT, 0, 8));
    let int_data: u32 = 64;
    types.extend_from_slice(&int_data.to_le_bytes());

    // id 2: BTF_KIND_STRUCT T { u64 x @ 0 } size=8 vlen=1.
    types.extend_from_slice(&kfunc_btf_type_header(n_t, BTF_KIND_STRUCT, 1, 8));
    types.extend_from_slice(&n_x.to_le_bytes());
    types.extend_from_slice(&1u32.to_le_bytes());
    types.extend_from_slice(&0u32.to_le_bytes());

    // id 3: BTF_KIND_PTR -> id 2.
    types.extend_from_slice(&kfunc_btf_type_header(0, BTF_KIND_PTR, 0, 2));

    // id 4: BTF_KIND_FUNC_PROTO returning id 3, no params.
    types.extend_from_slice(&kfunc_btf_type_header(0, BTF_KIND_FUNC_PROTO, 0, 3));

    // id 5: BTF_KIND_FUNC kf_name -> id 4 (proto), linkage=extern.
    types.extend_from_slice(&kfunc_btf_type_header(
        n_func,
        BTF_KIND_FUNC,
        BTF_FUNC_EXTERN,
        4,
    ));

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);
    (blob, 5, 2)
}

/// Build an ELF64 `Elf64_Rel` entry (16 bytes, little-endian).
/// `Elf64_Rel { r_offset(8), r_info(8) }` where
/// `r_info = (sym_idx << 32) | r_type`.
fn elf64_rel(r_offset: u64, sym_idx: u64, r_type: u32) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..8].copy_from_slice(&r_offset.to_le_bytes());
    let r_info = (sym_idx << 32) | (r_type as u64);
    out[8..16].copy_from_slice(&r_info.to_le_bytes());
    out
}

/// Encode a `BPF_JMP|BPF_CALL` with the clang-emitted pre-
/// relocation kfunc form: `code=0x85`, `dst=0`,
/// `src=BPF_PSEUDO_CALL=1`, `off=0`, `imm=-1`.
fn pre_reloc_kfunc_call_bytes() -> [u8; 8] {
    [0x85, 0x10, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff]
}

/// Encode an EXIT instruction (`code=0x95`).
fn kfunc_exit_bytes() -> [u8; 8] {
    [0x95, 0, 0, 0, 0, 0, 0, 0]
}

/// Test 1 — happy path: kfunc call gets rewritten.
#[test]
fn patch_kfunc_calls_happy_path_rewrites_call_site() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, expected_func_id, _t_id) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let kf_str_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        kf_str_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());

    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");

    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    assert_eq!(text_concat[0].code, 0x85);
    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].code, 0x85);
    assert_eq!(
        text_concat[0].src_reg(),
        BPF_PSEUDO_KFUNC_CALL,
        "src_reg now BPF_PSEUDO_KFUNC_CALL"
    );
    assert_eq!(
        text_concat[0].imm, expected_func_id as i32,
        "imm patched to BTF Func id"
    );
    assert_eq!(text_concat[1].code, 0x95);
}

/// Test 2 — non-extern symbol must NOT trigger patching.
#[test]
fn patch_kfunc_calls_skips_non_extern_symbol() {
    let kf_name = "static_helper";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_LOCAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 3 — symbol is extern but its name does NOT resolve to
/// an extern FUNC in the program BTF.
#[test]
fn patch_kfunc_calls_skips_symbol_not_in_btf() {
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob("bpf_task_acquire");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let unknown = "unknown_kfunc";
    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(unknown.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 4 — relocation targets a section we did NOT add to
/// `section_bases` (e.g. `.maps`).
#[test]
fn patch_kfunc_calls_ignores_non_text_relocations() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".maps", sh::SHT_PROGBITS).data(vec![0u8; 8]),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(3)
                .entsize(24),
            SecSpec::new(".rel.maps", sh::SHT_REL)
                .data(rel_data)
                .link(4)
                .info(2)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 5 — relocation byte offset is past the section's end.
#[test]
fn patch_kfunc_calls_rejects_out_of_bounds_offset() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&pre_reloc_kfunc_call_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());
    // r_offset = 100 (past 16-byte .text).
    let rel_data: Vec<u8> = elf64_rel(100, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(pre_reloc_kfunc_call_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, -1);
}

/// Test 6 — the relocation lands on a non-call instruction
/// (LD_IMM64). The patcher's code-byte gate rejects.
#[test]
fn patch_kfunc_calls_rejects_non_call_instruction() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    let ld_imm64_first_slot: [u8; 8] = [0x18, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let ld_imm64_second_slot: [u8; 8] = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&ld_imm64_first_slot);
    text.extend_from_slice(&ld_imm64_second_slot);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 1).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(ld_imm64_first_slot),
        BpfInsn::from_le_bytes(ld_imm64_second_slot),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    let pre = text_concat.clone();
    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat, pre);
}

/// Test 7 — relocation entry whose `imm` is NOT `-1` (a
/// resolved subprog call). Must not be patched.
#[test]
fn patch_kfunc_calls_rejects_non_minus_one_imm() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _func_id, _) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let name_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        name_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    // imm = 42 (not -1).
    let subprog_call: [u8; 8] = [0x85, 0x10, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00];
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&subprog_call);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(subprog_call),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_CALL);
    assert_eq!(text_concat[0].imm, 42);
}

/// Test 8 — `find_extern_func_btf_id` only matches FUNC types,
/// not other kinds that share the same name.
#[test]
fn find_extern_func_btf_id_filters_to_func_kind() {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = strings.len() as u32;
    strings.extend_from_slice(b"u64");
    strings.push(0);
    let n_foo = strings.len() as u32;
    strings.extend_from_slice(b"foo");
    strings.push(0);

    let mut types: Vec<u8> = Vec::new();
    types.extend_from_slice(&kfunc_btf_type_header(n_u64, 1, 0, 8));
    types.extend_from_slice(&64u32.to_le_bytes());
    // BTF_KIND_VAR (kind=14) named "foo".
    types.extend_from_slice(&kfunc_btf_type_header(n_foo, 14, 0, 1));
    types.extend_from_slice(&1u32.to_le_bytes());

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);

    let btf = Btf::from_bytes(&blob).expect("parse btf");
    // VAR id is not returned (kind filter rejects).
    assert_eq!(find_extern_func_btf_id(&btf, "foo"), None);
    // Name not in BTF returns None.
    assert_eq!(find_extern_func_btf_id(&btf, "absent"), None);
}

// ----- build_subprog_returns tests -------------------------------
//
// Coverage strategy: each test synthesises a minimal BPF object
// (program text + symtab/strtab + SHT_REL section) and feeds the
// analyzer-visible tuple `(text_concat, elf, section_bases)` to
// [`build_subprog_returns`]. The four test cases pin the four
// gates the function applies before recording a [`SubprogReturn`]:
//
//   1. Happy path — symbol is `STT_FUNC` + non-`SHN_UNDEF` +
//      `BPF_PSEUDO_CALL` site + name on `ALLOC_SUBPROG_NAMES`.
//      The result must contain exactly one entry pointing at the
//      call PC.
//   2. Gate skip — `BPF_PSEUDO_KFUNC_CALL` (src_reg = 2). Kfunc
//      calls are handled by `handle_kfunc_call` separately; a
//      subprog-return seed at this site would mis-route the
//      arena tag.
//   3. Gate skip — `STT_OBJECT` symbol (data, not function). The
//      relocation might target a call PC by coincidence but the
//      symbol is not a subprog.
//   4. Gate skip — `STT_FUNC` symbol whose name is NOT on
//      `ALLOC_SUBPROG_NAMES`. A regular subprog call must not
//      seed an arena tag — the analyzer's `BPF_OP_CALL` arm
//      relies on the seed list to disambiguate.
//
// These four gates compose the "false-negative-safe" boundary of
// the SubprogReturn pipeline; a regression in any one of them
// would either drop allocator-call sites silently (false
// negative — surfaces as a missing chase) or seed an arena tag
// on an unrelated subprog call (false positive — produces a
// misleading render). The tests below pin each gate independently.

/// Encode a `BPF_PSEUDO_CALL` (src_reg = 1) call instruction:
/// `code=0x85`, `dst=0`, `src=1`, `off=0`, `imm=any`.
/// Mirrors clang's pre-relocation BPF-to-BPF call shape.
fn pseudo_call_bytes(imm: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0] = 0x85; // BPF_JMP | BPF_CALL
    out[1] = 0x10; // dst=0, src=1 (BPF_PSEUDO_CALL)
    out[2..4].copy_from_slice(&0i16.to_le_bytes());
    out[4..8].copy_from_slice(&imm.to_le_bytes());
    out
}

/// Encode a `BPF_PSEUDO_KFUNC_CALL` (src_reg = 2) call
/// instruction. Used by the gate-skip test to confirm kfunc
/// call sites do not seed SubprogReturn entries.
fn pseudo_kfunc_call_bytes(imm: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0] = 0x85;
    out[1] = 0x20; // dst=0, src=2 (BPF_PSEUDO_KFUNC_CALL)
    out[2..4].copy_from_slice(&0i16.to_le_bytes());
    out[4..8].copy_from_slice(&imm.to_le_bytes());
    out
}

/// Build a `(elf, text_concat, section_bases)` triple that
/// [`build_subprog_returns`] consumes. The input is a minimal
/// BPF object with one program text section (call + EXIT) plus
/// a SHT_REL section pointing at the call PC. The symbol the
/// reloc references is parameterised so each gate test can
/// vary it independently. Returns the three tuple elements.
#[allow(clippy::too_many_arguments)]
fn build_subprog_test_scaffold(
    sym_name: &str,
    sym_st_type_bind: u8,
    sym_st_shndx: u16,
    call_bytes: [u8; 8],
) -> (Vec<u8>, Vec<BpfInsn>, HashMap<u32, usize>) {
    let mut strtab: Vec<u8> = vec![0];
    let n_sym = strtab.len() as u32;
    strtab.extend_from_slice(sym_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(n_sym, sym_st_type_bind, sym_st_shndx, 0, 0));

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&call_bytes);
    text.extend_from_slice(&kfunc_exit_bytes());
    // r_offset = 0 (call is at the first slot), sym_idx = 1
    // (the synthesised symbol), r_type = 1 (R_BPF_64_64 — value
    // does not matter for the SubprogReturn walk; the function
    // does not gate on r_type, only on the resolved instruction
    // and symbol shape).
    let rel_data: Vec<u8> = elf64_rel(0, 1, 1).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(call_bytes),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0); // .text at section index 1
    (blob, text_concat, section_bases)
}

/// Test 1 — happy path. Symbol is `STT_FUNC` global non-extern,
/// name is on `ALLOC_SUBPROG_NAMES`, call is `BPF_PSEUDO_CALL`.
/// Must emit exactly one [`SubprogReturn`] at the call PC.
#[test]
fn build_subprog_returns_happy_path_emits_one() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "scx_alloc_internal",
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1, // st_shndx — .text at shdr[1]
        pseudo_call_bytes(123),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert_eq!(out.len(), 1, "happy path: expected 1 entry, got {out:?}");
    assert_eq!(
        out[0].insn_offset, 0,
        "SubprogReturn must point at the call PC"
    );
}

/// Test 2 — gate skip: `BPF_PSEUDO_KFUNC_CALL` site. Even though
/// the symbol is `STT_FUNC` and the name is on the allowlist,
/// the call's `src_reg = 2` (kfunc) must be rejected. Kfunc
/// arena allocators are tagged via
/// [`crate::monitor::cast_analysis::ARENA_ALLOC_KFUNC_NAMES`]
/// inside [`crate::monitor::cast_analysis::Analyzer::handle_kfunc_call`],
/// not via SubprogReturn.
#[test]
fn build_subprog_returns_skips_pseudo_kfunc_call() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "scx_alloc_internal",
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1,
        pseudo_kfunc_call_bytes(0),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "BPF_PSEUDO_KFUNC_CALL must not seed a SubprogReturn: {out:?}"
    );
}

/// Test 3 — gate skip: `STT_OBJECT` symbol. A data symbol
/// (`STT_OBJECT`) referenced by a reloc on a call site is
/// malformed input — the relocation walks over a call PC but
/// the resolved symbol is not a subprog. The
/// `sym.st_type() == STT_FUNC` gate must reject it.
#[test]
fn build_subprog_returns_skips_stt_object() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "scx_alloc_internal",
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,
        pseudo_call_bytes(0),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "STT_OBJECT symbol must not seed a SubprogReturn: {out:?}"
    );
}

/// Test 4 — gate skip: `STT_FUNC` symbol whose name is NOT on
/// `ALLOC_SUBPROG_NAMES`. A regular BPF-to-BPF call to a
/// non-allocator subprog must not seed an arena tag. The
/// allowlist keeps the arena finding path strictly scoped.
#[test]
fn build_subprog_returns_skips_non_allowlist_name() {
    let (blob, text_concat, section_bases) = build_subprog_test_scaffold(
        "ktstr_some_unrelated_helper",
        st_info(syms::STB_GLOBAL, syms::STT_FUNC),
        1,
        pseudo_call_bytes(0),
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let out = build_subprog_returns(&text_concat, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "non-allowlist subprog name must not seed a SubprogReturn: {out:?}"
    );
}

// ----- build_datasec_pointers tests ------------------------------
//
// The eight gates inside `build_datasec_pointers` reject malformed
// input and surface only well-formed `DatasecPointer` annotations
// for `R_BPF_64_64` relocations whose target instruction is a
// `BPF_LD_IMM64` referencing a `BTF_KIND_DATASEC` section. The
// tests below construct one `(elf, btf, section_bases)` tuple per
// gate, run [`build_datasec_pointers`], and assert the gate fired
// (empty result) or did not fire (one result with the expected
// fields).

/// Encode `BPF_LD_IMM64` first-slot wire bytes:
/// `code=0x18`, `dst_reg=0`, `src_reg=0`, `off=0`, `imm`.
/// libbpf-style pre-relocation: the LD_IMM64 second slot
/// (also 8 bytes, all zero except trailing imm-high) is appended
/// separately by callers — only the first slot opcode matters
/// for the `build_datasec_pointers` gate.
fn ld_imm64_first_slot_bytes(imm: i32) -> [u8; 8] {
    // `BPF_LD | BPF_DW | BPF_IMM` = 0x18 in linux uapi `bpf.h`.
    let mut out = [0u8; 8];
    out[0] = 0x18;
    out[1] = 0; // regs byte: dst=0, src=0
    out[2..4].copy_from_slice(&0i16.to_le_bytes());
    out[4..8].copy_from_slice(&imm.to_le_bytes());
    out
}

/// `BPF_LD_IMM64` second slot — 8 bytes with the imm-high field
/// cleared. Production paths use this slot for the high 32 bits
/// of a 64-bit immediate; the test only needs a non-call slot
/// the patcher will skip.
fn ld_imm64_second_slot_bytes() -> [u8; 8] {
    [0u8; 8]
}

/// Append a single `BTF_KIND_DATASEC` type to `types`. Each
/// datasec entry is `name_off(4) info(4) size(4)` (12 bytes,
/// the standard btf_type header) plus N * 12 bytes for each
/// VarSecinfo (`type(4) offset(4) size(4)`). `vsi_entries` is a
/// slice of `(type_id, offset, size)` triples — empty list is
/// allowed (vlen=0), giving a name-only datasec.
fn append_btf_datasec(
    types: &mut Vec<u8>,
    name_off: u32,
    section_size: u32,
    vsi_entries: &[(u32, u32, u32)],
) {
    // BTF_KIND_DATASEC = 15. info packs `(kind << 24) | vlen`.
    const BTF_KIND_DATASEC: u32 = 15;
    let vlen = vsi_entries.len() as u32;
    let info = ((BTF_KIND_DATASEC << 24) & 0x1f00_0000) | (vlen & 0xffff);
    types.extend_from_slice(&name_off.to_le_bytes());
    types.extend_from_slice(&info.to_le_bytes());
    // size_or_type field carries the section's total byte size
    // for DATASEC (matches kernel `btf_type::size_or_type` union
    // when `kind == BTF_KIND_DATASEC`).
    types.extend_from_slice(&section_size.to_le_bytes());
    for (type_id, offset, size) in vsi_entries {
        types.extend_from_slice(&type_id.to_le_bytes());
        types.extend_from_slice(&offset.to_le_bytes());
        types.extend_from_slice(&size.to_le_bytes());
    }
}

/// Build a minimal `.BTF` blob containing one `BTF_KIND_DATASEC`
/// named `sec_name` plus one `BTF_KIND_INT u64` (id=1). Returns
/// the byte blob and the datasec id (always 2). The integer is
/// the underlying type for any VarSecinfo entries the caller adds.
fn build_datasec_btf_blob(sec_name: &str) -> (Vec<u8>, u32) {
    let mut strings: Vec<u8> = vec![0];
    let n_u64 = strings.len() as u32;
    strings.extend_from_slice(b"u64");
    strings.push(0);
    let n_sec = strings.len() as u32;
    strings.extend_from_slice(sec_name.as_bytes());
    strings.push(0);

    let mut types: Vec<u8> = Vec::new();
    // id 1: BTF_KIND_INT u64 size=8 bits=64 (encoding=0).
    types.extend_from_slice(&kfunc_btf_type_header(n_u64, 1, 0, 8));
    let int_data: u32 = 64;
    types.extend_from_slice(&int_data.to_le_bytes());
    // id 2: BTF_KIND_DATASEC named `sec_name`, no VSI entries.
    // `build_datasec_pointers` only resolves the section name
    // to a datasec id; it does NOT walk the VSI list (that's
    // the analyzer's job during STX/LDX). An empty VSI list is
    // acceptable for these gate-focused tests.
    append_btf_datasec(&mut types, n_sec, 32, &[]);

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);
    (blob, 2)
}

/// Construct the standard scaffold the `build_datasec_pointers`
/// gate tests share: an inner ELF with a `.bss`-named PROGBITS
/// section (the "datasec target"), a `.text` section with one
/// LD_IMM64 + EXIT, a `.symtab` + `.strtab`, and an `SHT_REL`
/// section relocating `.text`. Returns `(blob, btf_blob,
/// text_concat, section_bases)` ready for [`build_datasec_pointers`].
///
/// `r_type` selects the relocation type byte (1 = R_BPF_64_64);
/// `r_offset` selects which `.text` slot the reloc lands on
/// (must be 0 for the LD_IMM64 first slot); `sym_st_value`,
/// `sym_st_shndx`, and `sym_st_type_bind` parameterize the
/// referenced symbol; `imm_value` is the LD_IMM64 first-slot
/// `imm` field. `sec_name_in_btf` controls whether the BTF
/// datasec's name matches the ELF section name.
#[allow(clippy::too_many_arguments)]
fn build_datasec_test_scaffold(
    bss_name: &'static str,
    sec_name_in_btf: &str,
    r_type: u32,
    r_offset: u64,
    sym_st_value: u64,
    sym_st_shndx: u16,
    sym_st_type_bind: u8,
    imm_value: i32,
) -> (Vec<u8>, Vec<u8>, Vec<BpfInsn>, HashMap<u32, usize>) {
    // BTF blob: one datasec whose name is `sec_name_in_btf`.
    let (btf_blob, _ds_id) = build_datasec_btf_blob(sec_name_in_btf);

    // ELF strtab: just the symbol name (we use a single named
    // symbol pointing into `.bss`).
    let mut strtab: Vec<u8> = vec![0];
    let n_sym = strtab.len() as u32;
    strtab.extend_from_slice(b"global_var");
    strtab.push(0);

    // Symtab: shdr[0] is the always-null sentinel; shdr[1] is
    // the variable symbol. `st_info` packs (bind, type) per
    // ELF64. The caller controls both via `sym_st_type_bind`.
    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        n_sym,
        sym_st_type_bind,
        sym_st_shndx,
        sym_st_value,
        0,
    ));

    // Text section: one LD_IMM64 + an EXIT slot. The LD_IMM64
    // uses two 8-byte slots; we encode a third slot for EXIT
    // so the section byte size is 24 — matching what the BPF
    // loader sees for a real LD_IMM64 followed by an exit.
    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&ld_imm64_first_slot_bytes(imm_value));
    text.extend_from_slice(&ld_imm64_second_slot_bytes());
    text.extend_from_slice(&kfunc_exit_bytes());

    // SHT_REL entry: `r_offset = r_offset` (caller-controlled),
    // `r_sym = 1` (our named symbol), `r_type = r_type`.
    let rel_data: Vec<u8> = elf64_rel(r_offset, 1, r_type).to_vec();

    // ELF layout (caller-controlled section names so tests can
    // exercise the "unknown section name" gate). Section
    // indices: 1 = `.bss`-named (`bss_name`); 2 = `.text`;
    // 3 = `.strtab`; 4 = `.symtab`; 5 = `.rel.text`; 6 = `.BTF`.
    let blob = build_elf64(
        vec![
            SecSpec::new(bss_name, sh::SHT_PROGBITS).data(vec![0u8; 32]),
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(3)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(4)
                .info(2) // info = target section idx (.text)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob.clone()),
        ],
        h::EM_BPF,
        h::ET_REL,
    );

    // Decoded text — three 8-byte instructions:
    //   slot 0: LD_IMM64 first half (the reloc target)
    //   slot 1: LD_IMM64 second half (zeros)
    //   slot 2: EXIT
    let text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(ld_imm64_first_slot_bytes(imm_value)),
        BpfInsn::from_le_bytes(ld_imm64_second_slot_bytes()),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];

    // section_bases: only the .text section (idx 2 here). The
    // base index is 0 because the test object only has one text
    // section, so its instructions start at concat-idx 0.
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(2, 0);

    (blob, btf_blob, text_concat, section_bases)
}

/// Gate 1 (R_BPF_64_64 type): a relocation whose `r_type` is
/// not `R_BPF_64_64` (= 1) is silently dropped — the function
/// produces no `DatasecPointer` even though every other gate
/// would pass.
#[test]
fn build_datasec_pointers_rejects_non_r_bpf_64_64() {
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        10, // r_type != R_BPF_64_64 (= 1)
        0,
        0,
        1, // st_shndx = .bss (idx 1)
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(out.is_empty(), "non-R_BPF_64_64 reloc must be skipped");
}

/// Gate 2 (`r_offset` alignment): a relocation whose `r_offset`
/// is not a multiple of 8 cannot reference an LD_IMM64
/// instruction (BPF instructions are 8-byte aligned). The
/// alignment gate fires before any other check.
#[test]
fn build_datasec_pointers_rejects_non_multiple_of_8_offset() {
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        4, // r_offset = 4 (not a multiple of 8)
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "r_offset=4 (not multiple of 8) must be rejected"
    );
}

/// Gate 3 (`r_offset` past section end): a relocation whose
/// `r_offset >= section_byte_size` cannot possibly land on a
/// real instruction. The bounds gate fires.
#[test]
fn build_datasec_pointers_rejects_offset_past_section_size() {
    // Text section size = 24 bytes (3 BPF instructions). An
    // r_offset of 100 is far past the end and must be rejected.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        100, // r_offset >= section_byte_size (= 24)
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "r_offset past section size must be rejected"
    );
}

/// Gate 4 (instruction opcode): a relocation that lands on an
/// instruction whose `code` byte is not `BPF_LD_IMM64` (= 0x18)
/// is silently dropped. The renderer relies on the LD_IMM64
/// arm to apply datasec annotations; a reloc on an EXIT or
/// LDX would mis-route the analyzer state.
#[test]
fn build_datasec_pointers_rejects_non_ld_imm64_opcode() {
    // r_offset = 16 → instruction index 2 (the EXIT slot, not
    // an LD_IMM64). The opcode-byte gate fires.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        16, // EXIT slot, not LD_IMM64
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "reloc on non-LD_IMM64 opcode must be rejected"
    );
}

/// Gate 5 (symbol section binding): symbols with `st_shndx`
/// set to `SHN_UNDEF` (0), `SHN_ABS` (0xFFF1), or `SHN_COMMON`
/// (0xFFF2) are not bound to a real section index; the
/// function rejects all three.
#[test]
fn build_datasec_pointers_rejects_special_section_index_symbols() {
    for shndx in [0u16, 0xFFF1, 0xFFF2] {
        let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
            ".bss",
            ".bss",
            1,
            0,
            0,
            shndx,
            st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
            0,
        );
        let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
        let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
        let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
        assert!(
            out.is_empty(),
            "symbol with st_shndx={shndx:#x} must be rejected"
        );
    }
}

/// Gate 6 (BTF datasec lookup): a section name that resolves
/// in the ELF but does NOT exist as a `BTF_KIND_DATASEC` in the
/// program BTF is rejected. Even if the section name is well-
/// formed (`.bss`), without a matching BTF datasec the
/// annotation cannot be emitted — the analyzer would have no
/// VarSecinfo entries to walk.
#[test]
fn build_datasec_pointers_rejects_section_not_in_btf() {
    // ELF section name = `.bss`, BTF datasec name = `.rodata`.
    // The BTF lookup at the section name `.bss` finds no
    // matching datasec → drop.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".rodata", // BTF datasec name mismatches ELF section name
        1,
        0,
        0,
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(
        out.is_empty(),
        "section name not in BTF as DATASEC must be rejected"
    );
}

/// Gate 7 (`sym.st_value` overflow): if `sym.st_value`
/// exceeds `u32::MAX`, the offset cannot be represented in the
/// `base_offset: u32` field of [`DatasecPointer`]. The gate
/// rejects.
#[test]
fn build_datasec_pointers_rejects_st_value_past_u32_max() {
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1,
        0,
        (u32::MAX as u64) + 1, // st_value > u32::MAX
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        0,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert!(out.is_empty(), "sym.st_value > u32::MAX must be rejected");
}

/// Gate 8 (happy path): every gate passes, the function emits
/// exactly one [`DatasecPointer`] with the expected
/// `insn_offset`, `datasec_type_id`, and `base_offset`.
/// The `base_offset` is the sum of `insn.imm` and
/// `sym.st_value`, mirroring the libbpf convention for
/// `STT_OBJECT` symbols carrying the per-variable offset in
/// `st_value` and `STT_SECTION` symbols using `imm`.
#[test]
fn build_datasec_pointers_happy_path_emits_pointer() {
    // `imm = 16`, `st_value = 0`: STT_SECTION-style
    // pre-relocation form where the byte offset of the
    // referenced global is encoded in the LD_IMM64 imm field.
    let (blob, btf_blob, text_concat, section_bases) = build_datasec_test_scaffold(
        ".bss",
        ".bss",
        1, // R_BPF_64_64
        0, // r_offset = 0 (LD_IMM64 first slot)
        0, // st_value = 0
        1, // st_shndx = .bss (idx 1)
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        16, // LD_IMM64 imm = 16 (offset within .bss)
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");
    let out = build_datasec_pointers(&text_concat, &btf, &elf, &section_bases);
    assert_eq!(out.len(), 1, "all gates pass → exactly one entry");
    assert_eq!(out[0].insn_offset, 0, "PC = base + r_offset/8 = 0");
    assert_eq!(
        out[0].datasec_type_id, 2,
        "datasec id is 2 (per build_datasec_btf_blob)"
    );
    assert_eq!(
        out[0].base_offset, 16,
        "base_offset = imm (16) + st_value (0) = 16"
    );
}

/// `find_datasec_btf_id` filters its results to
/// `BTF_KIND_DATASEC` only — a name shared by a `BTF_KIND_VAR`
/// or `BTF_KIND_INT` does not match. Mirrors the kind-filter
/// invariant in [`find_extern_func_btf_id_filters_to_func_kind`]
/// for the kfunc helper.
#[test]
fn find_datasec_btf_id_filters_to_datasec_kind() {
    // Build a BTF with three types named `.bss`:
    //   id 1: BTF_KIND_INT named ".bss" (size=4, bits=32)
    //   id 2: BTF_KIND_VAR named ".bss" (linkage=1)
    //   id 3: BTF_KIND_DATASEC named ".bss" (size=8)
    // The lookup must return id 3 — not id 1 (Int) or id 2
    // (Var) — even though all three share the same name.
    let mut strings: Vec<u8> = vec![0];
    let n_bss = strings.len() as u32;
    strings.extend_from_slice(b".bss");
    strings.push(0);

    let mut types: Vec<u8> = Vec::new();
    // id 1: INT
    types.extend_from_slice(&kfunc_btf_type_header(n_bss, 1, 0, 4));
    let int_data: u32 = 32;
    types.extend_from_slice(&int_data.to_le_bytes());
    // id 2: VAR (kind=14, vlen=0). size_or_type = wrapped int id (1).
    types.extend_from_slice(&kfunc_btf_type_header(n_bss, 14, 0, 1));
    let var_linkage: u32 = 1; // global
    types.extend_from_slice(&var_linkage.to_le_bytes());
    // id 3: DATASEC (kind=15, vlen=0). size_or_type = section
    // byte size (8).
    append_btf_datasec(&mut types, n_bss, 8, &[]);

    let mut blob: Vec<u8> = Vec::new();
    blob.extend_from_slice(&0xEB9F_u16.to_le_bytes());
    blob.push(1);
    blob.push(0);
    blob.extend_from_slice(&24u32.to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(types.len() as u32).to_le_bytes());
    blob.extend_from_slice(&(strings.len() as u32).to_le_bytes());
    blob.extend_from_slice(&types);
    blob.extend_from_slice(&strings);

    let btf = Btf::from_bytes(&blob).expect("parse btf");
    // The datasec is id 3; the helper must filter past Int (1)
    // and Var (2) to return it.
    assert_eq!(
        find_datasec_btf_id(&btf, ".bss"),
        Some(3),
        "kind filter must skip past Int/Var to the Datasec",
    );
    // A name not present in the BTF returns None.
    assert_eq!(find_datasec_btf_id(&btf, ".rodata"), None);
}

/// `patch_kfunc_calls` already-relocated gate: a call whose
/// `src_reg == BPF_PSEUDO_KFUNC_CALL` (= 2) and `imm == 42`
/// has already been rewritten by some prior relocation pass
/// (e.g. an scheduler binary that captures a post-load BPF
/// object). The patcher must NOT overwrite the kernel BTF id
/// already in `imm` — doing so would replace a kernel id with
/// a program-BTF id, sending the analyzer to the wrong BTF
/// universe. Both `src_reg` and `imm` survive unmodified.
#[test]
fn patch_kfunc_calls_skips_already_relocated_src_reg() {
    let kf_name = "bpf_task_acquire";
    let (btf_blob, _expected_func_id, _t_id) = build_kfunc_btf_blob(kf_name);
    let btf = Btf::from_bytes(&btf_blob).expect("parse btf");

    let mut strtab: Vec<u8> = vec![0];
    let kf_str_off = strtab.len() as u32;
    strtab.extend_from_slice(kf_name.as_bytes());
    strtab.push(0);

    let mut symtab: Vec<u8> = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    symtab.extend_from_slice(&elf64_sym(
        kf_str_off,
        st_info(syms::STB_GLOBAL, syms::STT_NOTYPE),
        0,
        0,
        0,
    ));

    // Already-relocated kfunc call:
    //   code = 0x85 (BPF_JMP | BPF_CALL)
    //   dst = 0, src = BPF_PSEUDO_KFUNC_CALL (= 2)
    //   off = 0, imm = 42 (some kernel BTF id)
    // The packed regs byte: dst=0 (low 4) | src=2 (high 4) = 0x20.
    let already_relocated_call: [u8; 8] = [0x85, 0x20, 0x00, 0x00, 42, 0x00, 0x00, 0x00];

    let mut text: Vec<u8> = Vec::new();
    text.extend_from_slice(&already_relocated_call);
    text.extend_from_slice(&kfunc_exit_bytes());
    let rel_data: Vec<u8> = elf64_rel(0, 1, 10).to_vec();

    let blob = build_elf64(
        vec![
            SecSpec::new(".text", sh::SHT_PROGBITS)
                .flags(sh::SHF_EXECINSTR.into())
                .data(text),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
            SecSpec::new(".rel.text", sh::SHT_REL)
                .data(rel_data)
                .link(3)
                .info(1)
                .entsize(16),
            SecSpec::new(".BTF", sh::SHT_PROGBITS).data(btf_blob),
        ],
        h::EM_BPF,
        h::ET_REL,
    );
    let elf = goblin::elf::Elf::parse(&blob).expect("parse elf");
    let mut text_concat: Vec<BpfInsn> = vec![
        BpfInsn::from_le_bytes(already_relocated_call),
        BpfInsn::from_le_bytes(kfunc_exit_bytes()),
    ];
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    section_bases.insert(1, 0);

    // Sanity: pre-call state matches the already-relocated form.
    assert_eq!(text_concat[0].code, 0x85);
    assert_eq!(text_concat[0].src_reg(), BPF_PSEUDO_KFUNC_CALL);
    assert_eq!(text_concat[0].imm, 42);

    patch_kfunc_calls(&mut text_concat, &btf, &elf, &section_bases);

    // Both fields must survive unmodified — the imm gate
    // (`imm != -1`) fires before any BTF lookup, preserving
    // the kernel id intact.
    assert_eq!(
        text_concat[0].src_reg(),
        BPF_PSEUDO_KFUNC_CALL,
        "src_reg must survive unmodified",
    );
    assert_eq!(
        text_concat[0].imm, 42,
        "imm must survive unmodified — kernel BTF id preserved",
    );
}

// ----- build_fwd_index tests -----------------------------------

/// Single BTF carrying complete `Type::Struct` entries indexes
/// each name to `(0, type_id)` — the fwd-resolution index is
/// the input the renderer's cross-BTF chase consults when a
/// `BTF_KIND_FWD` terminal needs a body lookup.
#[test]
fn build_fwd_index_indexes_single_btf_structs() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_foo = push_btf_name(&mut strings, "foo");
    let n_bar = push_btf_name(&mut strings, "bar");
    let n_x = push_btf_name(&mut strings, "x");
    let types = vec![
        // id 1: u64 (skipped by the indexer — only Struct/Union)
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct foo { u64 x @ 0 }
        SynKind::Struct {
            name_off: n_foo,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: struct bar { u64 x @ 0 }
        SynKind::Struct {
            name_off: n_bar,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf_full(&types, &strings);
    let btf = Arc::new(Btf::from_bytes(&blob).expect("parse btf"));
    let btfs = vec![btf];
    let index = build_fwd_index(&btfs);
    assert_eq!(
        index.get("foo"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 2,
        })
    );
    assert_eq!(
        index.get("bar"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 3,
        })
    );
    assert!(!index.contains_key("u64"), "Int names must not be indexed");
}

/// Multiple BTFs: the index records the first BTF seen for any
/// duplicate name, so an entry's `(idx, type_id)` reflects the
/// first-write-wins policy. The renderer only consults the
/// cross-BTF index when local in-BTF resolution failed, so a
/// name conflict resolved locally never reaches the index.
#[test]
fn build_fwd_index_first_write_wins_on_duplicate_name() {
    // BTF #0: struct foo at id 2 (u64 at offset 0)
    let mut strings_0 = vec![0u8];
    let n_int_0 = push_btf_name(&mut strings_0, "u64");
    let n_foo_0 = push_btf_name(&mut strings_0, "foo");
    let n_x_0 = push_btf_name(&mut strings_0, "x");
    let types_0 = vec![
        SynKind::Int {
            name_off: n_int_0,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_foo_0,
            size: 8,
            members: vec![SynMember {
                name_off: n_x_0,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_0 = build_btf_full(&types_0, &strings_0);
    let btf_0 = Arc::new(Btf::from_bytes(&blob_0).expect("parse btf 0"));

    // BTF #1: also has struct foo (different layout!) at id 2.
    // Index keeps the BTF #0 entry per first-write-wins.
    let mut strings_1 = vec![0u8];
    let n_int_1 = push_btf_name(&mut strings_1, "u64");
    let n_foo_1 = push_btf_name(&mut strings_1, "foo");
    let n_y_1 = push_btf_name(&mut strings_1, "y");
    let types_1 = vec![
        SynKind::Int {
            name_off: n_int_1,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_foo_1,
            size: 16,
            members: vec![SynMember {
                name_off: n_y_1,
                type_id: 1,
                byte_offset: 8,
            }],
        },
    ];
    let blob_1 = build_btf_full(&types_1, &strings_1);
    let btf_1 = Arc::new(Btf::from_bytes(&blob_1).expect("parse btf 1"));

    let btfs = vec![btf_0, btf_1];
    let index = build_fwd_index(&btfs);
    // Entry must point at BTF #0, not #1.
    assert_eq!(
        index.get("foo"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 2,
        }),
        "first-write-wins: BTF #0 wins on duplicate name"
    );
}

/// Anonymous structs (empty resolved name) are silently
/// skipped — the index keys on names, so an anonymous type has
/// nothing to look up.
#[test]
fn build_fwd_index_skips_anonymous_structs() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_x = push_btf_name(&mut strings, "x");
    let types = vec![
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // Anonymous struct (name_off = 0)
        SynKind::Struct {
            name_off: 0,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf_full(&types, &strings);
    let btf = Arc::new(Btf::from_bytes(&blob).expect("parse btf"));
    let btfs = vec![btf];
    let index = build_fwd_index(&btfs);
    // Anonymous struct must NOT be indexed under the empty
    // string (would key every anonymous type to the same slot).
    assert!(
        index.is_empty(),
        "anonymous structs must not be indexed: {index:?}"
    );
}

/// PhD-F gap G1: cross-BTF Fwd resolution — `BTF_KIND_FWD` entries
/// must not register in the index even when no complete body
/// shares the name. The renderer's chase consults the index to
/// resolve a Fwd terminal to a complete sibling; a Fwd-keyed
/// entry would point the chase at another Fwd (no body), defeating
/// the index. The function's `if let Type::Struct(s) | Type::Union(s)`
/// filter excludes Fwd before the name lookup.
///
/// Fixture: BTF #0 declares `struct shared;` (Fwd) at id 2; BTF #1
/// defines `struct shared { u64 v @ 0 }` at id 2. Expected:
/// `index["shared"] = (1, 2)` — first-write-wins records BTF #1's
/// complete body, not BTF #0's Fwd. Even if BTF order were
/// reversed (Fwd-bearing BTF traversed first), the Fwd would be
/// filtered out so the complete body in the other BTF would still
/// win. This test pins the Fwd-skip behaviour explicitly.
#[test]
fn build_fwd_index_skips_fwd_when_complete_body_in_later_btf() {
    // BTF #0: forward declaration only — `struct shared;` at id 2.
    let mut strings_0 = vec![0u8];
    let n_int_0 = push_btf_name(&mut strings_0, "u64");
    let n_shared_0 = push_btf_name(&mut strings_0, "shared");
    let types_0 = vec![
        // id 1: u64 (filler so id=2 is the Fwd)
        SynKind::Int {
            name_off: n_int_0,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: BTF_KIND_FWD `struct shared;` (kind_flag=0 -> struct)
        SynKind::Fwd {
            name_off: n_shared_0,
            kind_flag: 0,
        },
    ];
    let blob_0 = build_btf_full(&types_0, &strings_0);
    let btf_0 = Arc::new(Btf::from_bytes(&blob_0).expect("parse btf 0"));

    // BTF #1: complete `struct shared { u64 v @ 0 }` at id 2 — the
    // body the cross-BTF index keys to.
    let mut strings_1 = vec![0u8];
    let n_int_1 = push_btf_name(&mut strings_1, "u64");
    let n_shared_1 = push_btf_name(&mut strings_1, "shared");
    let n_v_1 = push_btf_name(&mut strings_1, "v");
    let types_1 = vec![
        SynKind::Int {
            name_off: n_int_1,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        SynKind::Struct {
            name_off: n_shared_1,
            size: 8,
            members: vec![SynMember {
                name_off: n_v_1,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob_1 = build_btf_full(&types_1, &strings_1);
    let btf_1 = Arc::new(Btf::from_bytes(&blob_1).expect("parse btf 1"));

    // Sanity: BTF #0's id 2 must be a Fwd (proves the synthesizer
    // emitted the right kind). Without this, a Struct-encoding bug
    // could pass the assertion below trivially.
    let ty_0_id_2 = btf_0
        .resolve_type_by_id(2)
        .expect("BTF #0 id 2 must resolve");
    assert!(
        matches!(ty_0_id_2, Type::Fwd(_)),
        "BTF #0 id 2 must be Fwd, got {ty_0_id_2:?}"
    );

    let btfs = vec![btf_0, btf_1];
    let index = build_fwd_index(&btfs);
    // The "shared" key must point at BTF #1 (the complete body),
    // not BTF #0 (the Fwd that should be filtered out).
    assert_eq!(
        index.get("shared"),
        Some(&FwdIndexEntry {
            btfs_idx: 1,
            type_id: 2,
        }),
        "Fwd in BTF #0 must not register; complete body in BTF #1 wins: {index:?}"
    );
    // No spurious entries from BTF #0.
    assert_eq!(
        index.len(),
        1,
        "only the BTF #1 complete body should be indexed: {index:?}"
    );
}

/// PhD-F gap G1: a `BTF_KIND_FWD` with `name_off = 0` (no name in
/// the strtab) must be silently skipped without panicking the
/// id-space walk. The btf-rs parser only registers names when
/// `name_off > 0` (see obj.rs `if bt.name_off > 0`), so the type
/// entry exists in `types[id]` but has no string-table linkage.
/// `build_fwd_index`'s `if let Type::Struct(s) | Type::Union(s)`
/// filter excludes the Fwd before the name lookup runs, so
/// `resolve_name` is never called on the empty-named Fwd.
/// This test pins the no-panic guarantee — a future refactor that
/// drops the kind filter (e.g. broadening to also index Typedefs)
/// must not panic on a name_off=0 Fwd.
#[test]
fn build_fwd_index_handles_empty_name_fwd_without_panic() {
    let mut strings = vec![0u8];
    let n_int = push_btf_name(&mut strings, "u64");
    let n_named = push_btf_name(&mut strings, "named");
    let n_x = push_btf_name(&mut strings, "x");
    let types = vec![
        // id 1: u64
        SynKind::Int {
            name_off: n_int,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: BTF_KIND_FWD with name_off=0 (no strtab linkage).
        // The btf-rs name-cache registration block at obj.rs is
        // skipped because of the `if bt.name_off > 0` gate.
        SynKind::Fwd {
            name_off: 0,
            kind_flag: 0,
        },
        // id 3: a complete named struct so the index has at
        // least one positive entry, proving the walk continued
        // past the empty-named Fwd at id 2 instead of aborting.
        SynKind::Struct {
            name_off: n_named,
            size: 8,
            members: vec![SynMember {
                name_off: n_x,
                type_id: 1,
                byte_offset: 0,
            }],
        },
    ];
    let blob = build_btf_full(&types, &strings);
    let btf = Arc::new(Btf::from_bytes(&blob).expect("parse btf"));

    // Sanity: BTF id 2 must be a Fwd (synthesizer encoded the
    // kind correctly).
    let ty_id_2 = btf.resolve_type_by_id(2).expect("BTF id 2 must resolve");
    assert!(
        matches!(ty_id_2, Type::Fwd(_)),
        "BTF id 2 must be Fwd, got {ty_id_2:?}"
    );

    let btfs = vec![btf];
    // The walk must not panic even though id 2's Fwd has no
    // strtab name. Filter behaviour: Fwd kind never reaches
    // `resolve_name`, so the empty-name path is structurally
    // unreachable for the production filter.
    let index = build_fwd_index(&btfs);
    // The empty-named Fwd at id 2 must NOT be present, neither
    // under "" (anonymous) nor under any other key.
    assert!(
        !index.contains_key(""),
        "empty-string key must not appear (anonymous Fwd): {index:?}"
    );
    // The named struct at id 3 must be indexed — proves the walk
    // continued past the Fwd at id 2 rather than terminating.
    assert_eq!(
        index.get("named"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 3,
        }),
        "named struct at id 3 must register after the empty-named Fwd at id 2: {index:?}"
    );
    assert_eq!(
        index.len(),
        1,
        "only the named struct should be indexed: {index:?}"
    );
}

/// Two-object end-to-end: object A's BTF declares
/// `struct cgx_target;` (a `BTF_KIND_FWD`) and references it
/// via a Ptr field; object B's BTF carries the full body
/// `struct cgx_target { u64 marker @ 0 }`. The cross-BTF index
/// produced by [`build_cast_analysis_from_bytes`] indexes
/// `cgx_target -> (1, 2)` — BTF #1 (object B) at type id 2,
/// the body location.
///
/// Mirrors the deferred-resolve arena cast target shape: a
/// `__arena u64` declared in object A whose true type is the
/// `cgx_target` body in object B. The renderer's chase then
/// resolves the Fwd through the cross-BTF index and renders
/// the payload.
#[test]
fn build_cast_analysis_indexes_cross_object_struct_body() {
    // Object A: declares `struct cgx_target;` as a Fwd at id
    // 2, used as a pointee. The Fwd has no body — just the
    // forward declaration.
    let mut strings_a = vec![0u8];
    let n_int_a = push_btf_name(&mut strings_a, "u64");
    let n_cgx_a = push_btf_name(&mut strings_a, "cgx_target");
    let n_t_a = push_btf_name(&mut strings_a, "outer_a");
    let n_field_a = push_btf_name(&mut strings_a, "ptr_to_target");
    let n_func_a = push_btf_name(&mut strings_a, "func_a");
    let n_text_a = push_btf_name(&mut strings_a, ".text");
    // This test omits a `cgx_target` Fwd in object A entirely;
    // the cross-BTF index lookup is exercised purely via object
    // B's complete body. `build_fwd_index_skips_fwd_when_complete_body_in_later_btf`
    // covers the Fwd-in-A + body-in-B shape directly with the
    // `SynKind::Fwd` synthesizer. The `outer_a` struct here just
    // exists so `analyze_one_object_with_btf` has a non-empty
    // type table to traverse.
    let types_a = vec![
        // id 1: u64
        SynKind::Int {
            name_off: n_int_a,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: outer_a (carries a u64 field; just there so
        // we have any struct at all in this BTF — the
        // cross-BTF assertion is on object B's body)
        SynKind::Struct {
            name_off: n_t_a,
            size: 8,
            members: vec![SynMember {
                name_off: n_field_a,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        // id 3: FuncProto returning void with one u64 param
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 1,
            }],
        },
        // id 4: Func
        SynKind::Func {
            name_off: n_func_a,
            type_id: 3,
            linkage: 1,
        },
    ];
    let _ = n_cgx_a; // unused: this test omits a `cgx_target` Fwd
    let btf_blob_a = build_btf_full(&types_a, &strings_a);
    let insns_a = vec![exit_insn()];
    let text_a = insns_to_text_bytes(&insns_a);
    let btf_ext_a = build_btf_ext(n_text_a, &[(0, 3)], 8);
    let inner_a = build_full_bpf_object_elf(text_a, btf_blob_a, btf_ext_a);

    // Object B: defines `struct cgx_target { u64 marker @ 0 }`
    // as a complete struct at id 2.
    let mut strings_b = vec![0u8];
    let n_int_b = push_btf_name(&mut strings_b, "u64");
    let n_cgx_b = push_btf_name(&mut strings_b, "cgx_target");
    let n_marker_b = push_btf_name(&mut strings_b, "marker");
    let n_func_b = push_btf_name(&mut strings_b, "func_b");
    let n_text_b = push_btf_name(&mut strings_b, ".text");
    let types_b = vec![
        // id 1: u64
        SynKind::Int {
            name_off: n_int_b,
            size: 8,
            encoding: 0,
            offset: 0,
            bits: 64,
        },
        // id 2: struct cgx_target { u64 marker @ 0 }  -- THE
        // BODY the cross-BTF index keys to.
        SynKind::Struct {
            name_off: n_cgx_b,
            size: 8,
            members: vec![SynMember {
                name_off: n_marker_b,
                type_id: 1,
                byte_offset: 0,
            }],
        },
        SynKind::FuncProto {
            return_type_id: 0,
            params: vec![SynParam {
                name_off: 0,
                type_id: 1,
            }],
        },
        SynKind::Func {
            name_off: n_func_b,
            type_id: 3,
            linkage: 1,
        },
    ];
    let btf_blob_b = build_btf_full(&types_b, &strings_b);
    let insns_b = vec![exit_insn()];
    let text_b = insns_to_text_bytes(&insns_b);
    let btf_ext_b = build_btf_ext(n_text_b, &[(0, 3)], 8);
    let inner_b = build_full_bpf_object_elf(text_b, btf_blob_b, btf_ext_b);

    // Outer ELF wraps both inner objects in `.bpf.objs` via
    // STT_OBJECT symbols so [`iter_embedded_bpf_objects`]
    // yields them as separate slices.
    let strtab = b"\0obj_a\0obj_b\0".to_vec();
    let mut symtab = Vec::new();
    symtab.extend_from_slice(&elf64_sym(0, 0, 0, 0, 0));
    // sym for object A: name_off = 1 (b"obj_a"), st_value = 0,
    // size = inner_a.len()
    symtab.extend_from_slice(&elf64_sym(
        1,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1, // st_shndx — .bpf.objs at shdr[1]
        0,
        inner_a.len() as u64,
    ));
    // sym for object B: name_off = 7 (b"obj_b"),
    // st_value = inner_a.len(), size = inner_b.len()
    symtab.extend_from_slice(&elf64_sym(
        7,
        st_info(syms::STB_GLOBAL, syms::STT_OBJECT),
        1,
        inner_a.len() as u64,
        inner_b.len() as u64,
    ));

    // Pack both inner objects back-to-back in `.bpf.objs`.
    let mut bpf_objs_data = Vec::new();
    bpf_objs_data.extend_from_slice(&inner_a);
    bpf_objs_data.extend_from_slice(&inner_b);

    let outer = build_elf64(
        vec![
            SecSpec::new(".bpf.objs", sh::SHT_PROGBITS).data(bpf_objs_data),
            SecSpec::new(".strtab", sh::SHT_STRTAB).data(strtab),
            SecSpec::new(".symtab", sh::SHT_SYMTAB)
                .data(symtab)
                .link(2)
                .entsize(24),
        ],
        h::EM_X86_64,
        h::ET_REL,
    );

    let out = build_cast_analysis_from_bytes(&outer);
    // Both BTFs parsed.
    assert_eq!(
        out.btfs.len(),
        2,
        "both embedded objects' BTFs must be retained: {}",
        out.btfs.len()
    );
    // The cross-BTF index has cgx_target keyed at the FIRST
    // BTF that carries a complete body. Object A in this
    // fixture exposes no `cgx_target` struct (this test omits
    // the Fwd entirely), so object B's id is what gets indexed.
    let cgx_hit = out.fwd_index.get("cgx_target");
    assert_eq!(
        cgx_hit,
        Some(&FwdIndexEntry {
            btfs_idx: 1,
            type_id: 2,
        }),
        "cross-BTF index must point cgx_target to BTF #1 at type id 2: {:?}",
        out.fwd_index
    );
    // Both objects' top-level structs are also indexed.
    assert_eq!(
        out.fwd_index.get("outer_a"),
        Some(&FwdIndexEntry {
            btfs_idx: 0,
            type_id: 2,
        }),
        "object A's struct outer_a must be indexed in BTF #0 at id 2"
    );
}

// ----- LazyCastMap full-output accessor -------------------------

/// `LazyCastMap::new(None).get_full()` returns `None` without
/// touching the filesystem or the process-wide cache. Matches
/// the no-scheduler dump-path contract (every `u64` renders as
/// a plain counter) for the production [`Self::get_full`]
/// accessor that returns the full [`CastAnalysisOutput`]
/// including the cross-BTF Fwd index.
#[test]
fn lazy_cast_map_get_full_returns_none_when_no_scheduler() {
    let lazy = LazyCastMap::new(None);
    assert!(
        lazy.get_full().is_none(),
        "no-scheduler builder must short-circuit `.get_full()` to None",
    );
}

// ----- cached_cast_analysis_for_scheduler concurrency -----------

/// Multi-thread race on the same scheduler binary path: every
/// caller must observe the same `Arc<CastAnalysisOutput>` —
/// pointer-equal — proving the per-hash `OnceLock` inside the
/// process-wide cache deduplicates concurrent first-callers
/// rather than running the analyzer once per caller and
/// returning equivalent-but-distinct Arcs.
///
/// Uses [`std::thread::scope`] so the threads can borrow the
/// path; an [`Arc<std::sync::Barrier>`] coordinates the
/// release point so every thread enters
/// [`cached_cast_analysis_for_scheduler`] within microseconds
/// of one another, maximising the contention on the cache's
/// `Mutex<HashMap>` lookup AND the per-hash
/// `OnceLock::get_or_init` serialisation. Without the barrier
/// the threads might serialise naturally on creation, missing
/// the concurrent-init regression the
/// `Arc<OnceLock<...>>` shape exists to catch.
#[test]
fn cached_cast_analysis_concurrent_callers_share_one_oncelock_init() {
    use std::sync::{Arc as StdArc, Barrier};

    // Build the standard arena-cast end-to-end fixture and
    // write it to a fresh path so the content hash is unique
    // to this test run (won't collide with other tests'
    // cache entries).
    let blob = build_recovers_arena_cast_outer_elf();
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path().join("concurrent.bin");
    std::fs::write(&p, &blob).expect("write");

    const N_THREADS: usize = 8;
    let barrier = StdArc::new(Barrier::new(N_THREADS));
    let path = p.clone();
    let results: Vec<Arc<CastAnalysisOutput>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..N_THREADS)
            .map(|_| {
                let barrier = barrier.clone();
                let path = path.clone();
                s.spawn(move || {
                    // Synchronise the release: every thread
                    // hits `wait()` before any thread enters
                    // the cache lookup.
                    barrier.wait();
                    cached_cast_analysis_for_scheduler(&path)
                        .expect("non-empty fixture must produce Some")
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(results.len(), N_THREADS);
    // Every Arc must be pointer-equal to the first — proves
    // the OnceLock dedup fired and only one analysis ran
    // across all N concurrent callers.
    let first = &results[0];
    for (i, other) in results.iter().enumerate().skip(1) {
        assert!(
            Arc::ptr_eq(first, other),
            "thread {i}: Arc must be pointer-equal to thread 0's; \
                 OnceLock dedup did NOT fire across concurrent callers",
        );
    }
}
