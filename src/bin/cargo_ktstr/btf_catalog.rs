//! BTF type anchor generation for BPF schedulers.
//!
//! BPF compilers eliminate struct types from BTF when all referencing
//! functions are inlined. This module generates a `-include` header
//! with weak global anchors that force the compiler to retain every
//! struct definition discovered in the scheduler's source tree.
//!
//! The pipeline:
//! 1. Extract `.bpf.c` source paths from BTF string tables in prior
//!    build's `.bpf.o` files (clang embeds absolute paths).
//! 2. Run `clang -M` on each source (in parallel) with the build's
//!    cflags to get the transitive include chain, filtering out
//!    system/kernel headers.
//! 3. Parse every file in the dep list with tree-sitter-c to extract
//!    struct definitions.
//! 4. Generate a header with weak global pointer declarations that
//!    anchor each struct in BTF.
//!
//! The anchor is cached in `target/ktstr_btf_anchor.h` with an ahash
//! of all inputs (ktstr version, source paths, cflags, .bpf.o sizes).
//! Regenerated only when inputs change.

use std::collections::{BTreeSet, HashSet};
use std::hash::{BuildHasher as _, Hash as _, Hasher as _};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Discover sources via BTF strings, run clang -M for deps, extract
/// structs via tree-sitter-c, write the anchor header.
pub(crate) fn generate_btf_anchor(
    bpf_object_dir: &Path,
    clang: &str,
    cflags: &[String],
    anchor_path: &Path,
) -> Option<PathBuf> {
    // Step 1: find .bpf.o files and extract source paths from BTF
    let mut bpf_sources = discover_sources_from_objects(bpf_object_dir);
    if bpf_sources.is_empty() {
        tracing::debug!("btf_anchor: no .bpf.c sources found via BTF");
        return None;
    }
    tracing::debug!(
        sources = bpf_sources.len(),
        "btf_anchor: discovered BPF sources via BTF"
    );

    // Fast path: hash all inputs that affect the anchor output.
    // - ktstr version: pipeline logic changes invalidate
    // - source paths: file set changes
    // - .bpf.o sizes: proxy for source content changes (recompilation
    //   changes object size via BTF/code changes)
    // - cflags: different includes change the dep chain
    bpf_sources.sort();
    let input_hash = {
        let mut h = ahash::RandomState::with_seeds(0x6b74, 0x7374, 0x7200, 0x616e).build_hasher();
        env!("CARGO_PKG_VERSION").hash(&mut h);
        for p in &bpf_sources {
            p.to_string_lossy().hash(&mut h);
        }
        for cflag in cflags {
            cflag.hash(&mut h);
        }
        if let Ok(entries) = std::fs::read_dir(bpf_object_dir) {
            let mut sizes: Vec<(String, u64)> = entries
                .flatten()
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    if name.ends_with(".bpf.o") {
                        e.metadata().ok().map(|m| (name, m.len()))
                    } else {
                        None
                    }
                })
                .collect();
            sizes.sort();
            for (name, size) in &sizes {
                name.hash(&mut h);
                size.hash(&mut h);
            }
        }
        h.finish()
    };
    if let Some(old_hash) = read_anchor_hash(anchor_path) {
        if old_hash == input_hash {
            tracing::debug!("btf_anchor: cached anchor is current");
            let abs = std::fs::canonicalize(anchor_path)
                .unwrap_or_else(|_| anchor_path.to_path_buf());
            return Some(abs);
        }
    }

    // Step 2: clang -M on each source to get transitive deps
    let dep_files = collect_dep_files(&bpf_sources, clang, cflags);
    if dep_files.is_empty() {
        tracing::debug!("btf_anchor: clang -M produced no dep files");
        return None;
    }
    tracing::debug!(
        files = dep_files.len(),
        "btf_anchor: collected dep files via clang -M"
    );

    // Step 3: tree-sitter-c parse for struct definitions
    let structs = extract_struct_names(&dep_files);
    if structs.is_empty() {
        tracing::debug!("btf_anchor: no struct definitions found");
        return None;
    }
    tracing::debug!(
        structs = structs.len(),
        "btf_anchor: extracted struct definitions"
    );

    if let Some(parent) = anchor_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    write_anchor_header(anchor_path, &structs, input_hash)?;

    let abs = std::fs::canonicalize(anchor_path)
        .unwrap_or_else(|_| anchor_path.to_path_buf());
    Some(abs)
}

/// Extract .bpf.c source paths from the BTF string table in each
/// .bpf.o file. clang embeds full absolute paths in BTF, making
/// this work regardless of build system.
fn discover_sources_from_objects(dir: &Path) -> Vec<PathBuf> {
    let mut sources: HashSet<PathBuf> = HashSet::new();

    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".bpf.o") || name == "bpf.bpf.o" {
            continue;
        }

        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };

        if let Some(btf_data) = find_btf_section_raw(&bytes) {
            for s in btf_strings(btf_data) {
                if s.ends_with(".bpf.c") {
                    let p = PathBuf::from(s);
                    if p.is_file() {
                        if let Ok(canonical) = std::fs::canonicalize(&p) {
                            sources.insert(canonical);
                        }
                    }
                }
            }
        }
    }

    sources.into_iter().collect()
}

fn find_btf_section_raw(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 64 {
        return None;
    }
    let e_shoff = u64::from_le_bytes(bytes[40..48].try_into().ok()?) as usize;
    let e_shentsize = u16::from_le_bytes(bytes[58..60].try_into().ok()?) as usize;
    let e_shnum = u16::from_le_bytes(bytes[60..62].try_into().ok()?) as usize;
    let e_shstrndx = u16::from_le_bytes(bytes[62..64].try_into().ok()?) as usize;

    if e_shstrndx >= e_shnum || e_shentsize < 64 {
        return None;
    }

    let strtab_base = e_shoff + e_shstrndx * e_shentsize;
    if strtab_base + 64 > bytes.len() {
        return None;
    }
    let strtab_off = u64::from_le_bytes(bytes[strtab_base + 24..strtab_base + 32].try_into().ok()?) as usize;
    let strtab_size = u64::from_le_bytes(bytes[strtab_base + 32..strtab_base + 40].try_into().ok()?) as usize;
    if strtab_off + strtab_size > bytes.len() {
        return None;
    }
    let strtab = &bytes[strtab_off..strtab_off + strtab_size];

    for i in 0..e_shnum {
        let base = e_shoff + i * e_shentsize;
        if base + 64 > bytes.len() {
            break;
        }
        let sh_name = u32::from_le_bytes(bytes[base..base + 4].try_into().ok()?) as usize;
        if sh_name + 4 >= strtab.len() {
            continue;
        }
        if &strtab[sh_name..sh_name + 4] != b".BTF" {
            continue;
        }
        if sh_name + 4 < strtab.len() && strtab[sh_name + 4] != 0 {
            continue;
        }
        let sh_offset = u64::from_le_bytes(bytes[base + 24..base + 32].try_into().ok()?) as usize;
        let sh_size = u64::from_le_bytes(bytes[base + 32..base + 40].try_into().ok()?) as usize;
        if sh_offset + sh_size <= bytes.len() && sh_size >= 24 {
            return Some(&bytes[sh_offset..sh_offset + sh_size]);
        }
    }
    None
}

fn btf_strings(btf: &[u8]) -> Vec<&str> {
    if btf.len() < 24 {
        return Vec::new();
    }
    let hdr_len = u32::from_le_bytes([btf[4], btf[5], btf[6], btf[7]]) as usize;
    let str_off = u32::from_le_bytes([btf[16], btf[17], btf[18], btf[19]]) as usize;
    let str_len = u32::from_le_bytes([btf[20], btf[21], btf[22], btf[23]]) as usize;
    let str_start = hdr_len + str_off;
    let str_end = str_start + str_len;
    if str_end > btf.len() {
        return Vec::new();
    }
    let str_section = &btf[str_start..str_end];
    let mut result = Vec::new();
    for chunk in str_section.split(|&b| b == 0) {
        if let Ok(s) = std::str::from_utf8(chunk) {
            if !s.is_empty() {
                result.push(s);
            }
        }
    }
    result
}

/// Run `clang -M -MG` on each source in parallel and collect every
/// dependency, filtering out system/kernel headers.
fn collect_dep_files(
    sources: &[PathBuf],
    clang: &str,
    cflags: &[String],
) -> Vec<PathBuf> {
    let all_deps = std::sync::Mutex::new(HashSet::<PathBuf>::new());

    std::thread::scope(|s| {
        for source in sources {
            let deps_ref = &all_deps;
            s.spawn(move || {
                let output = Command::new(clang)
                    .arg("-M")
                    .arg("-MG")
                    .arg("-target")
                    .arg("bpf")
                    .args(cflags)
                    .arg(source)
                    .output();

                let Ok(output) = output else { return };
                if !output.status.success() {
                    return;
                }

                let mut local = HashSet::new();
                let stdout = String::from_utf8_lossy(&output.stdout);
                let joined = stdout.replace("\\\n", " ");
                for line in joined.lines() {
                    let deps_part = match line.split_once(':') {
                        Some((_, deps)) => deps,
                        None => line,
                    };
                    for token in deps_part.split_whitespace() {
                        let p = PathBuf::from(token);
                        if p.is_file() {
                            if let Ok(canonical) = std::fs::canonicalize(&p) {
                                if !is_system_header(&canonical) {
                                    local.insert(canonical);
                                }
                            }
                        }
                    }
                }
                deps_ref.lock().unwrap().extend(local);
            });
        }
    });

    all_deps.into_inner().unwrap().into_iter().collect()
}

fn is_system_header(path: &Path) -> bool {
    let s = path.to_string_lossy();
    if s.contains("/usr/include/") || s.contains("/usr/lib/") {
        return true;
    }
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        if name == "vmlinux.h" || name == "vmlinux.bpf.h" {
            return true;
        }
    }
    if s.contains("scx_utils-bpf_h/") {
        return true;
    }
    false
}

/// Parse C files with tree-sitter-c and extract named struct definitions.
fn extract_struct_names(files: &[PathBuf]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
        .expect("tree-sitter-c language");

    for file in files {
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };
        let Some(tree) = parser.parse(&content, None) else {
            continue;
        };
        collect_structs(tree.root_node(), content.as_bytes(), &mut names);
    }
    names
}

fn collect_structs(
    node: tree_sitter::Node,
    source: &[u8],
    names: &mut BTreeSet<String>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "struct_specifier" {
            if child.child_by_field_name("body").is_some() {
                if let Some(name_node) = child.child_by_field_name("name") {
                    if let Ok(name) = std::str::from_utf8(&source[name_node.byte_range()]) {
                        if !name.is_empty() && !name.starts_with("__") {
                            names.insert(name.to_string());
                        }
                    }
                }
            }
        }
        collect_structs(child, source, names);
    }
}

fn read_anchor_hash(path: &Path) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    let line = content.lines().find(|l| l.starts_with("/* ktstr_hash="))?;
    let hex = line.strip_prefix("/* ktstr_hash=")?.strip_suffix(" */")?;
    u64::from_str_radix(hex, 16).ok()
}

fn write_anchor_header(path: &Path, structs: &BTreeSet<String>, hash: u64) -> Option<()> {
    let mut src = String::new();
    src.push_str(&format!("/* ktstr_hash={hash:016x} */\n"));
    src.push_str("#ifndef __KTSTR_BTF_ANCHOR_H\n");
    src.push_str("#define __KTSTR_BTF_ANCHOR_H\n");
    for (i, s) in structs.iter().enumerate() {
        src.push_str(&format!(
            "struct {s} __attribute__((weak)) *__ktstr_keep_{i};\n"
        ));
    }
    src.push_str("#endif\n");
    std::fs::write(path, &src).ok()
}
