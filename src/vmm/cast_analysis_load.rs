//! Host-side BPF cast analysis driver for the scheduler binary.
//!
//! Bridges the path-based scheduler-binary input (a libbpf-rs / scx-built
//! ELF that embeds its compiled BPF objects into a `.bpf.objs` PROGBITS
//! section) and the pure-data [`crate::monitor::cast_analysis::analyze_casts`]
//! pass that turns BPF instructions plus a parsed [`btf_rs::Btf`] into a
//! [`crate::monitor::cast_analysis::CastMap`].
//!
//! # Pipeline
//!
//! 1. Read the scheduler binary from disk.
//! 2. Parse it as a host ELF via [`goblin::elf::Elf::parse`]; locate the
//!    `.bpf.objs` PROGBITS section. scx schedulers (the only producers
//!    we target) embed their compiled BPF object inline at that section
//!    via the libbpf-rs / scx skel codegen. Each `STT_OBJECT` symbol in
//!    the outer ELF whose containing section is `.bpf.objs` points at a
//!    contiguous embedded ELF blob — the BPF object that the scheduler
//!    will hand to `bpf_object__load` at runtime.
//! 3. For each embedded ELF, parse its `.BTF` (and `.BTF.ext` when
//!    present) plus every program text section (any PROGBITS section
//!    flagged `SHF_EXECINSTR`).
//! 4. Concatenate the program texts in section-header order. Decode each
//!    8-byte slot through [`crate::monitor::cast_analysis::BpfInsn::from_le_bytes`].
//! 5. Walk `.BTF.ext`'s `func_info` and build the [`FuncEntry`] table:
//!    every record's `insn_off` (in BYTES) becomes a function-entry PC
//!    once divided by 8 and offset into the concatenated stream by the
//!    base of the section the record belongs to. The record's `type_id`
//!    is the BTF id of `BTF_KIND_FUNC` whose `func.type` is the
//!    [`btf_rs::Type::FuncProto`] the analyzer reseeds R1..R5 from.
//! 6. Run [`analyze_casts`]; merge the result into a single
//!    [`CastMap`] aggregating every embedded BPF object's findings.
//!
//! # Error policy
//!
//! Any failure (path missing, ELF parse, missing `.bpf.objs`, missing
//! `.BTF`, malformed `.BTF.ext`, BTF parse failure) returns an empty
//! [`CastMap`] after a `warn!` log line. The dump path is best-effort —
//! a missing cast map silently disables typed-pointer promotion in the
//! renderer (every `u64` field renders as a plain counter, the
//! pre-integration default).
//!
//! No libbpf calls, no kernel BPF interaction, no CAP_BPF needed — this
//! runs purely on the on-disk binary bytes.

use std::collections::HashMap;

use crate::monitor::cast_analysis::{BpfInsn, CastMap, FuncEntry, analyze_casts};

use btf_rs::Btf;

/// One BPF instruction's wire size (bytes). Mirrors `sizeof(struct
/// bpf_insn)` in the kernel's UAPI and the [`BpfInsn::from_le_bytes`]
/// 8-byte input. Used to translate `.BTF.ext`-reported byte offsets
/// (`bpf_func_info::insn_off`) into instruction indices for
/// [`FuncEntry::insn_offset`].
const BPF_INSN_SIZE: usize = 8;

/// Resolve a string offset against the BTF string table embedded in
/// the `.BTF` section blob. Per kernel `include/uapi/linux/btf.h`,
/// the BTF header is: magic(2) + version(1) + flags(1) + hdr_len(4)
/// + type_off(4) + type_len(4) + str_off(4) + str_len(4) = 24 bytes.
///
/// The string table starts at `hdr_len + str_off` within the blob.
fn btf_str_at(btf_bytes: &[u8], str_off: u32) -> Option<&str> {
    if btf_bytes.len() < 24 {
        return None;
    }
    let hdr_len = u32::from_le_bytes(btf_bytes[4..8].try_into().ok()?) as usize;
    let str_section_off = u32::from_le_bytes(btf_bytes[16..20].try_into().ok()?) as usize;
    let str_section_len = u32::from_le_bytes(btf_bytes[20..24].try_into().ok()?) as usize;
    let str_start = hdr_len + str_section_off;
    let off = str_off as usize;
    if off >= str_section_len {
        return None;
    }
    let base = str_start + off;
    if base >= btf_bytes.len() {
        return None;
    }
    let strtab_end = (str_start + str_section_len).min(btf_bytes.len());
    if base >= strtab_end {
        return None;
    }
    let end = btf_bytes[base..strtab_end]
        .iter()
        .position(|&b| b == 0)
        .map(|p| base + p)
        .unwrap_or(strtab_end);
    std::str::from_utf8(&btf_bytes[base..end]).ok()
}

/// `.BTF.ext` magic — `0xEB9F` in native byte order.
///
/// Same magic as the `.BTF` section. A mismatch here (truncation,
/// foreign-endian, corruption) triggers the silent-empty-result path:
/// the cast analyzer never sees garbage data.
const BTF_MAGIC: u16 = 0xEB9F;

/// Minimum `.BTF.ext` header byte size. Per kernel
/// `tools/lib/bpf/btf.c:btf_ext_parse`, the minimum is
/// `offsetofend(struct btf_ext_header, line_info_len)` = 24 bytes:
/// magic(2) + version(1) + flags(1) + hdr_len(4) + func_info_off(4)
/// + func_info_len(4) + line_info_off(4) + line_info_len(4).
const BTF_EXT_HEADER_MIN_LEN: u32 = 24;

/// Build the merged cast map for a scheduler binary.
///
/// Returns an empty [`CastMap`] on any failure; the dump path treats an
/// empty map as "no cast analysis available", which produces the same
/// rendering as before this integration landed.
///
/// `path` is the host filesystem path to the scheduler binary (the
/// builder's `scheduler_binary` field). The function reads the file,
/// locates every embedded BPF object inside `.bpf.objs`, and merges
/// each object's per-program cast findings into a single [`CastMap`].
///
/// # Why merge across objects
///
/// scx schedulers ship a single embedded BPF object per binary
/// today. The merge is a no-op in that case. Multi-object schedulers
/// (theoretical) produce one [`CastMap`] per object; merging into a
/// single map keeps the runtime threading uniform — the renderer's
/// per-map [`crate::monitor::btf_render::MemReader::cast_lookup`]
/// dispatches on `(parent_btf_id, offset)` and the BTF type ids in
/// disjoint program BTFs do not collide because each program BTF is
/// loaded under its own `btf_kva` at runtime, and the renderer
/// indexes the cast map only after it has resolved a parent struct
/// in a specific BTF (so `(parent_id, offset)` is implicitly scoped
/// to that BTF). The conservative "false negatives are fine, false
/// positives are not" stance from
/// [`crate::monitor::cast_analysis`] still applies.
pub(crate) fn build_cast_map_from_scheduler(path: &std::path::Path) -> CastMap {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "cast_analysis: read scheduler binary failed; \
                 dump renderer will fall back to plain u64 counters"
            );
            return CastMap::new();
        }
    };
    let outer = match goblin::elf::Elf::parse(&bytes) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cast_analysis: parse outer ELF failed; \
                 dump renderer will fall back to plain u64 counters"
            );
            return CastMap::new();
        }
    };
    let bpf_objs_section = match find_section(&outer, ".bpf.objs") {
        Some(s) => s,
        None => {
            tracing::debug!(
                "cast_analysis: scheduler binary has no .bpf.objs section; \
                 nothing to analyze"
            );
            return CastMap::new();
        }
    };

    let mut merged = CastMap::new();
    for inner in iter_embedded_bpf_objects(&outer, &bytes, bpf_objs_section) {
        merge_into(&mut merged, analyze_one_object(inner));
    }
    merged
}

/// Walk the outer ELF's symbol tables and yield every byte slice that
/// belongs to a `STT_OBJECT` symbol whose section is `.bpf.objs`.
///
/// scx-built schedulers emit a single such symbol per BPF object — the
/// libbpf-rs `bpf_skel::imp::DATA` slice the runtime hands to
/// `bpf_object__load`. A scheduler that statically composes multiple
/// BPF objects (theoretical; not produced by today's scx skel codegen)
/// would emit one symbol per object and the iterator would yield each
/// in turn. The fallback "one slice covering the whole section" path
/// ensures a hand-crafted scheduler that drops the symbol table still
/// gets analyzed: the section name alone is enough to identify the
/// blob.
fn iter_embedded_bpf_objects<'data>(
    outer: &goblin::elf::Elf<'_>,
    file_bytes: &'data [u8],
    bpf_objs_idx: usize,
) -> Vec<&'data [u8]> {
    let mut out: Vec<&[u8]> = Vec::new();
    // Symbol-driven path: every STT_OBJECT pointing into .bpf.objs.
    // st_value is the section-relative virtual address (the section's
    // sh_addr is the section start in the file's virtual layout); a
    // typical `.bpf.objs` is non-allocated and sh_addr matches sh_offset
    // semantics here, but we anchor on the section's file offset
    // explicitly to avoid relying on that coincidence.
    let sh = &outer.section_headers[bpf_objs_idx];
    let sec_file_start = sh.sh_offset as usize;
    let sec_file_end = sec_file_start.saturating_add(sh.sh_size as usize);
    let sec_va_start = sh.sh_addr;
    for sym in outer.syms.iter() {
        // st_type 1 == STT_OBJECT (data symbol); section index match
        // ties the symbol to .bpf.objs. SHN_UNDEF / SHN_ABS / SHN_COMMON
        // are below the section-header range so the equality test
        // already excludes them.
        if sym.st_type() != goblin::elf::sym::STT_OBJECT {
            continue;
        }
        if sym.st_shndx != bpf_objs_idx {
            continue;
        }
        if sym.st_size == 0 {
            continue;
        }
        // Translate virtual address → file offset. For a typical
        // non-allocated `.bpf.objs` section, sh_addr is 0 and st_value
        // is the byte offset within the section. For an allocated
        // section, sh_addr is the load address and st_value is also
        // a virtual address; in either case the per-symbol offset
        // within the section is `st_value - sh_addr`, and the file
        // offset is `sec_file_start + (st_value - sh_addr)`. Using
        // checked arithmetic so a symbol whose st_value somehow
        // precedes sh_addr (corrupted ELF) is rejected rather than
        // wrapping into a wild slice index.
        let Some(rel) = sym.st_value.checked_sub(sec_va_start) else {
            continue;
        };
        let Some(start) = (sec_file_start as u64).checked_add(rel) else {
            continue;
        };
        let Some(end) = start.checked_add(sym.st_size) else {
            continue;
        };
        if (start as usize) < sec_file_start || (end as usize) > sec_file_end {
            continue;
        }
        if let Some(slice) = file_bytes.get(start as usize..end as usize) {
            out.push(slice);
        }
    }
    if out.is_empty() {
        // No matching symbol — fall back to treating the entire
        // section as one BPF object. scx-built binaries always emit
        // a covering symbol; a stripped binary or a custom scheduler
        // that omits it still gets analysis as long as the section's
        // bytes are themselves a valid BPF object ELF.
        if let Some(slice) = file_bytes.get(sec_file_start..sec_file_end) {
            out.push(slice);
        }
    }
    out
}

/// Run cast analysis on one embedded BPF object's bytes.
///
/// The bytes are themselves an ELF (the BPF object); parse it, extract
/// the BTF, the `.BTF.ext`-derived [`FuncEntry`] table, and the
/// concatenated instruction stream, then call [`analyze_casts`].
fn analyze_one_object(obj_bytes: &[u8]) -> CastMap {
    let elf = match goblin::elf::Elf::parse(obj_bytes) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cast_analysis: parse inner BPF object ELF failed"
            );
            return CastMap::new();
        }
    };

    // .BTF is mandatory — no BTF, no struct/field resolution, no
    // analysis output the renderer can use.
    let btf_bytes = match find_section(&elf, ".BTF").and_then(|i| section_data(&elf, obj_bytes, i))
    {
        Some(b) => b,
        None => {
            tracing::debug!("cast_analysis: inner ELF has no .BTF section");
            return CastMap::new();
        }
    };
    let btf = match Btf::from_bytes(btf_bytes) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                error = ?e,
                "cast_analysis: parse .BTF failed"
            );
            return CastMap::new();
        }
    };

    // Instruction sections in section-header order: every
    // SHF_EXECINSTR-flagged PROGBITS section. Concatenating in this
    // order matches how `.BTF.ext` records reference them — each
    // record's `insn_off` is byte-relative to its OWN section, so we
    // record each section's base index in the concatenated stream and
    // translate per-record below.
    let mut text_concat: Vec<BpfInsn> = Vec::new();
    let mut section_bases: HashMap<u32, usize> = HashMap::new();
    for (idx, sh) in elf.section_headers.iter().enumerate() {
        if sh.sh_type != goblin::elf::section_header::SHT_PROGBITS {
            continue;
        }
        if sh.sh_flags & u64::from(goblin::elf::section_header::SHF_EXECINSTR) == 0 {
            continue;
        }
        let Some(data) = section_data(&elf, obj_bytes, idx) else {
            continue;
        };
        if data.len() % BPF_INSN_SIZE != 0 {
            // Non-multiple-of-8 program section: malformed for BPF
            // bytecode. Skip rather than try to decode partial slots.
            continue;
        }
        let base = text_concat.len();
        for chunk in data.chunks_exact(BPF_INSN_SIZE) {
            let mut buf = [0u8; BPF_INSN_SIZE];
            buf.copy_from_slice(chunk);
            text_concat.push(BpfInsn::from_le_bytes(buf));
        }
        section_bases.insert(idx as u32, base);
    }
    if text_concat.is_empty() {
        tracing::debug!("cast_analysis: inner ELF has no executable BPF program sections");
        return CastMap::new();
    }

    // .BTF.ext is optional — without it, every program function still
    // appears in the concatenated insn stream, but the analyzer cannot
    // reseed R1..R5 at function entries. Without entries the
    // analyzer cannot clear stale R6..R9 state at function
    // boundaries, which could produce false positives in theory
    // (stale typed pointer leaks via concatenation fall-through).
    // In practice all scx-built schedulers ship valid .BTF.ext.
    let func_entries = find_section(&elf, ".BTF.ext")
        .and_then(|i| section_data(&elf, obj_bytes, i))
        .map(|d| parse_btf_ext_func_entries(d, btf_bytes, &elf, &section_bases))
        .unwrap_or_default();

    analyze_casts(&text_concat, &btf, &[], &func_entries)
}

/// Parse `.BTF.ext` and emit one [`FuncEntry`] per `bpf_func_info`
/// record in every section.
///
/// Returns an empty Vec on any malformed input. The format matches
/// `struct btf_ext_header` + per-info-section blobs from
/// `tools/lib/bpf/libbpf_internal.h`:
///
/// ```text
/// btf_ext_header { u16 magic; u8 version; u8 flags; u32 hdr_len;
///                  u32 func_info_off; u32 func_info_len;
///                  u32 line_info_off; u32 line_info_len;
///                  // optional: u32 core_relo_off; u32 core_relo_len; }
/// // After header (at offset hdr_len):
/// // func_info section starts at hdr_len + func_info_off:
/// //   u32 record_size
/// //   repeated for each program section that has func_info:
/// //     btf_ext_info_sec { u32 sec_name_off; u32 num_info; }
/// //     bpf_func_info_min[num_info] { u32 insn_off; u32 type_id; }
/// // ...
/// ```
///
/// `insn_off` is in BYTES; we divide by [`BPF_INSN_SIZE`] (8) to
/// translate to an instruction index. Records are scoped to the
/// section named by `sec_name_off` in the `.BTF` strtab; the
/// instruction index gets offset by that section's base in the
/// concatenated text stream. A section whose name we cannot resolve,
/// or that we did not collect into the concatenated stream (e.g. it
/// lacked SHF_EXECINSTR), is silently skipped — its records produce
/// no [`FuncEntry`].
fn parse_btf_ext_func_entries(
    data: &[u8],
    btf_bytes: &[u8],
    inner_elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) -> Vec<FuncEntry> {
    if data.len() < BTF_EXT_HEADER_MIN_LEN as usize {
        return Vec::new();
    }
    let magic = u16::from_le_bytes([data[0], data[1]]);
    if magic != BTF_MAGIC {
        // Wrong-endian or corrupted; we don't try to byteswap. Cast
        // analysis is best-effort.
        return Vec::new();
    }
    // data[2] = version, data[3] = flags — not consulted; the
    // wire layout is documented in the BTF_EXT_HEADER_MIN_LEN comment.
    let hdr_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let func_info_off = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    let func_info_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
    if hdr_len < BTF_EXT_HEADER_MIN_LEN || (hdr_len as usize) > data.len() {
        return Vec::new();
    }
    if func_info_len == 0 {
        return Vec::new();
    }
    // The func_info data starts at `hdr_len + func_info_off` and runs
    // for `func_info_len` bytes. Bound-check that whole window.
    let info_start = (hdr_len as usize).checked_add(func_info_off as usize);
    let info_end = info_start.and_then(|s| s.checked_add(func_info_len as usize));
    let (info_start, info_end) = match (info_start, info_end) {
        (Some(s), Some(e)) => (s, e),
        _ => return Vec::new(),
    };
    if info_end > data.len() {
        return Vec::new();
    }
    let info = &data[info_start..info_end];
    if info.len() < 4 {
        return Vec::new();
    }
    let record_size = u32::from_le_bytes([info[0], info[1], info[2], info[3]]) as usize;
    // Minimum bpf_func_info layout is { u32 insn_off; u32 type_id; }
    // — 8 bytes. Newer kernels may pad to a larger record_size; we
    // only consume the first 8 bytes of each record (`insn_off` and
    // `type_id`) and skip the rest, mirroring `bpf_func_info_min` in
    // libbpf_internal.h.
    if record_size < 8 {
        return Vec::new();
    }
    let mut cursor = 4usize;
    let mut out: Vec<FuncEntry> = Vec::new();
    while cursor + 8 <= info.len() {
        let sec_name_off = u32::from_le_bytes([
            info[cursor],
            info[cursor + 1],
            info[cursor + 2],
            info[cursor + 3],
        ]);
        let num_info = u32::from_le_bytes([
            info[cursor + 4],
            info[cursor + 5],
            info[cursor + 6],
            info[cursor + 7],
        ]) as usize;
        cursor += 8;
        let records_bytes = num_info.saturating_mul(record_size);
        match cursor.checked_add(records_bytes) {
            Some(end) if end <= info.len() => {}
            _ => break,
        }
        // Resolve section name via the BTF string table — per kernel
        // libbpf (tools/lib/bpf/libbpf.c:3328), `.BTF.ext`
        // `sec_name_off` indexes the BTF strtab, NOT the ELF
        // section-header strtab. The BTF strtab starts at
        // `hdr_len + str_off` within the `.BTF` blob.
        let sec_name = match btf_str_at(btf_bytes, sec_name_off) {
            Some(s) => s,
            None => {
                cursor += records_bytes;
                continue;
            }
        };
        let sec_idx = match find_section(inner_elf, sec_name) {
            Some(i) => i as u32,
            None => {
                cursor += records_bytes;
                continue;
            }
        };
        let base = match section_bases.get(&sec_idx) {
            Some(b) => *b,
            None => {
                cursor += records_bytes;
                continue;
            }
        };
        for i in 0..num_info {
            let rec_off = cursor + i * record_size;
            // Read the first 8 bytes (`bpf_func_info_min`); ignore
            // any trailing padding in newer record layouts.
            let insn_off = u32::from_le_bytes([
                info[rec_off],
                info[rec_off + 1],
                info[rec_off + 2],
                info[rec_off + 3],
            ]) as usize;
            let type_id = u32::from_le_bytes([
                info[rec_off + 4],
                info[rec_off + 5],
                info[rec_off + 6],
                info[rec_off + 7],
            ]);
            // insn_off is in BYTES per libbpf docs; translate to an
            // instruction index. A non-multiple-of-8 byte offset is
            // malformed (no real BPF function starts on a non-aligned
            // boundary); skip silently — false negative is safe.
            if insn_off % BPF_INSN_SIZE != 0 {
                continue;
            }
            let entry_idx = base.saturating_add(insn_off / BPF_INSN_SIZE);
            out.push(FuncEntry {
                insn_offset: entry_idx,
                func_proto_id: type_id,
            });
        }
        cursor += records_bytes;
    }
    out
}

/// Merge `from` into `into`. Coalesces conflicting `(source, offset)`
/// keys by dropping the entry — same conservative stance as
/// [`crate::monitor::cast_analysis`]'s own `KptrEntry::Conflicting`
/// rule. Self-consistent merges (same `(target, AddrSpace)` from two
/// program objects, e.g. a shared library type both reference) keep
/// the entry.
fn merge_into(into: &mut CastMap, from: CastMap) {
    use std::collections::btree_map::Entry;
    for (key, val) in from {
        match into.entry(key) {
            Entry::Vacant(v) => {
                v.insert(val);
            }
            Entry::Occupied(o) => {
                if *o.get() != val {
                    // Disagreement across objects: drop the entry.
                    // The renderer falls back to plain u64 for that
                    // slot, which is the original (pre-integration)
                    // behavior.
                    o.remove();
                }
            }
        }
    }
}

/// Find a section by exact name. Returns the section index, or `None`
/// if no section matches. Uses `shdr_strtab.get_at` directly to avoid
/// pulling section data when only the index is needed.
fn find_section(elf: &goblin::elf::Elf<'_>, name: &str) -> Option<usize> {
    for (i, sh) in elf.section_headers.iter().enumerate() {
        if let Some(n) = elf.shdr_strtab.get_at(sh.sh_name)
            && n == name
        {
            return Some(i);
        }
    }
    None
}

/// Get the byte slice covering a section's `[sh_offset, sh_offset +
/// sh_size)` range. Returns `None` if the range is out of bounds (a
/// malformed ELF whose section header points past file end).
fn section_data<'a>(
    elf: &goblin::elf::Elf<'_>,
    file_bytes: &'a [u8],
    idx: usize,
) -> Option<&'a [u8]> {
    let sh = elf.section_headers.get(idx)?;
    let start = sh.sh_offset as usize;
    let end = start.checked_add(sh.sh_size as usize)?;
    file_bytes.get(start..end)
}
