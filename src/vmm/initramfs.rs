/// Minimal initramfs (cpio newc format) creation via the `cpio` crate.
/// Packs the test binary as `/init` along with scheduler binaries,
/// shared libraries, optional busybox, and user-provided include files
/// into a cpio archive for use as Linux initrd.
/// Init setup is handled by Rust code in `vmm::rust_init`.
use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Result of shared library resolution for a binary.
#[derive(Debug, Clone)]
pub(crate) struct SharedLibs {
    /// Resolved `(guest_path, host_path)` pairs.
    pub found: Vec<(String, PathBuf)>,
    /// Library sonames that could not be resolved to a host path.
    /// Each entry includes whether it is a direct (DT_NEEDED) dependency
    /// of the root binary or a transitive dependency.
    pub missing: Vec<MissingLib>,
    /// The binary's PT_INTERP path, if present (e.g. `/lib64/ld-linux-x86-64.so.2`).
    pub interpreter: Option<String>,
}

/// A shared library dependency that could not be resolved.
#[derive(Debug, Clone)]
pub(crate) struct MissingLib {
    /// The soname (e.g. `libssl.so.1.1`).
    pub soname: String,
    /// True if this soname appears in the root binary's DT_NEEDED.
    /// False if it is a transitive dependency of one of the root's deps.
    pub direct: bool,
}

/// Parse `/etc/ld.so.conf` and any included files to produce additional
/// library search paths. The format:
/// - One directory path per line
/// - Lines starting with `#` are comments
/// - `include <glob>` directives match files (e.g. `/etc/ld.so.conf.d/*.conf`)
/// - Empty lines are skipped
///
/// Parsed once and cached for the process lifetime.
static LD_SO_CONF_PATHS: LazyLock<Vec<PathBuf>> =
    LazyLock::new(|| parse_ld_so_conf(Path::new("/etc/ld.so.conf")));

/// Maximum recursion depth for ld.so.conf `include` chains. Guards against
/// cyclic or pathologically deep include graphs.
const LD_SO_CONF_MAX_DEPTH: usize = 16;

/// Parse a single ld.so.conf-format file, recursing into `include` directives.
/// Cycles are broken via a visited-set keyed on the canonicalized path
/// (falling back to the raw path when canonicalize fails, e.g. ENOENT);
/// the recursion also stops at [`LD_SO_CONF_MAX_DEPTH`].
fn parse_ld_so_conf_file(
    path: &Path,
    out: &mut Vec<PathBuf>,
    visited: &mut HashSet<PathBuf>,
    depth: usize,
) {
    if depth >= LD_SO_CONF_MAX_DEPTH {
        tracing::warn!(
            path = %path.display(),
            max_depth = LD_SO_CONF_MAX_DEPTH,
            "ld.so.conf include depth limit hit; truncating further includes"
        );
        return;
    }
    let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !visited.insert(key) {
        tracing::warn!(
            path = %path.display(),
            "ld.so.conf already-visited include file; skipping to avoid redundant or cyclic descent"
        );
        return;
    }
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(glob_pattern) = trimmed.strip_prefix("include") {
            let glob_pattern = glob_pattern.trim();
            if glob_pattern.is_empty() {
                continue;
            }
            // Expand the glob by reading the parent directory and matching.
            let pattern_path = Path::new(glob_pattern);
            if let Some(parent) = pattern_path.parent()
                && let Some(file_pattern) = pattern_path.file_name()
            {
                let pat = file_pattern.to_string_lossy();
                if let Ok(entries) = std::fs::read_dir(parent) {
                    let mut paths: Vec<PathBuf> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| {
                            p.file_name()
                                .is_some_and(|n| glob_match(&pat, &n.to_string_lossy()))
                        })
                        .collect();
                    paths.sort();
                    for p in paths {
                        parse_ld_so_conf_file(&p, out, visited, depth + 1);
                    }
                }
            }
        } else {
            let dir = PathBuf::from(trimmed);
            if dir.is_dir() {
                out.push(dir);
            }
        }
    }
}

/// Parse `/etc/ld.so.conf` and return all library search directories.
fn parse_ld_so_conf(path: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut visited = HashSet::new();
    parse_ld_so_conf_file(path, &mut paths, &mut visited, 0);
    paths
}

/// Parsed soname-to-path mappings from `/etc/ld.so.cache`.
///
/// The binary cache is the authoritative lookup used by `ld-linux.so`.
/// It contains entries for every library indexed by `ldconfig`, including
/// libraries in directories added via `ldconfig /path` that may not
/// appear in the text-based `/etc/ld.so.conf` files. Parsing the cache
/// catches libraries the conf-based directory scan misses.
///
/// Format (glibc new format, `glibc-ld.so.cache1.1`):
///   - 48-byte header: magic[20] + nlibs[4] + len_strings[4] + flags[4] + unused[16]
///   - nlibs entries of 24 bytes: flags[4] + key[4] + value[4] + osversion[4] + hwcap[8]
///   - String table: key/value are absolute byte offsets from file start
static LD_SO_CACHE: LazyLock<HashMap<String, PathBuf>> =
    LazyLock::new(|| parse_ld_so_cache(Path::new("/etc/ld.so.cache")));

/// Magic bytes at the start of the glibc new-format `ld.so.cache`.
const LD_CACHE_MAGIC: &[u8; 20] = b"glibc-ld.so.cache1.1";
/// Header size: magic(20) + nlibs(4) + len_strings(4) + flags(4) + unused(16).
const LD_CACHE_HEADER_SIZE: usize = 48;
/// Per-entry size: flags(4) + key(4) + value(4) + osversion(4) + hwcap(8).
const LD_CACHE_ENTRY_SIZE: usize = 24;

/// Parse the binary `/etc/ld.so.cache` file into a soname->path map.
///
/// Scans for the new-format magic because some systems prepend the
/// old format (`ld.so-1.7.0`) before the new-format section.
fn parse_ld_so_cache(path: &Path) -> HashMap<String, PathBuf> {
    let mut map = HashMap::new();
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return map,
    };
    // Scan for new-format magic. Usually at offset 0, but old-format
    // systems prepend the legacy section.
    let Some(magic_pos) = data
        .windows(LD_CACHE_MAGIC.len())
        .position(|w| w == LD_CACHE_MAGIC)
    else {
        return map;
    };
    let hdr = magic_pos;
    if data.len() < hdr + LD_CACHE_HEADER_SIZE {
        return map;
    }
    let nlibs = u32::from_le_bytes(data[hdr + 20..hdr + 24].try_into().unwrap()) as usize;
    let min_size = hdr + LD_CACHE_HEADER_SIZE + nlibs * LD_CACHE_ENTRY_SIZE;
    if data.len() < min_size {
        return map;
    }
    for i in 0..nlibs {
        let off = hdr + LD_CACHE_HEADER_SIZE + i * LD_CACHE_ENTRY_SIZE;
        // key and value are absolute byte offsets from file start.
        let key_off = u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
        let val_off = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap()) as usize;
        if key_off >= data.len() || val_off >= data.len() {
            continue;
        }
        let soname = match read_cstr(&data, key_off) {
            Some(s) => s,
            None => continue,
        };
        let path_str = match read_cstr(&data, val_off) {
            Some(s) => s,
            None => continue,
        };
        // Only accept absolute paths that exist as files.
        if path_str.starts_with('/') {
            let p = PathBuf::from(path_str);
            if p.is_file() {
                map.entry(soname.to_string()).or_insert(p);
            }
        }
    }
    map
}

/// Read a null-terminated C string from `data` at `offset`.
fn read_cstr(data: &[u8], offset: usize) -> Option<&str> {
    let end = data[offset..].iter().position(|&b| b == 0)?;
    std::str::from_utf8(&data[offset..offset + end]).ok()
}

/// Simple glob matching supporting only `*` as a wildcard.
/// Matches the full string (not a substring).
fn glob_match(pattern: &str, s: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        s.starts_with(prefix)
    } else if let Some(suffix) = pattern.strip_prefix('*') {
        s.ends_with(suffix)
    } else if let Some((prefix, suffix)) = pattern.split_once('*') {
        s.starts_with(prefix) && s[prefix.len()..].ends_with(suffix)
    } else {
        pattern == s
    }
}

/// Resolve shared library dependencies for a dynamically-linked ELF binary.
/// Parses the ELF dynamic section to read DT_NEEDED entries, then resolves
/// each soname to a host path matching the host dynamic linker's search
/// order: LD_LIBRARY_PATH → DT_RUNPATH/DT_RPATH → /etc/ld.so.cache →
/// /etc/ld.so.conf paths → default library paths. When the binary uses
/// a non-standard PT_INTERP, the interpreter's parent and sibling lib
/// dirs are prepended to the search path (before RPATH/RUNPATH) and
/// propagated to transitive deps. Walks transitive deps via
/// level-parallel BFS. Returns empty result for static binaries or
/// non-ELF files.
#[tracing::instrument(skip_all, fields(binary = %binary.display()))]
pub(crate) fn resolve_shared_libs(binary: &Path) -> Result<SharedLibs> {
    // Cache results by canonical path — avoids re-resolving the same
    // binary across concurrent initramfs builds (nextest parallelism).
    static CACHE: LazyLock<std::sync::Mutex<HashMap<PathBuf, SharedLibs>>> =
        LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

    let canon = std::fs::canonicalize(binary).unwrap_or_else(|_| binary.to_path_buf());
    if let Ok(cache) = CACHE.lock()
        && let Some(cached) = cache.get(&canon)
    {
        return Ok(cached.clone());
    }

    let data =
        std::fs::read(binary).with_context(|| format!("read binary: {}", binary.display()))?;
    let elf = match goblin::elf::Elf::parse(&data) {
        Ok(e) => e,
        Err(_) => {
            // Not a valid ELF (or 32-bit) — treat as static/non-dynamic.
            return Ok(SharedLibs {
                found: vec![],
                missing: vec![],
                interpreter: None,
            });
        }
    };

    let interpreter = elf.interpreter.map(|s| s.to_string());

    if elf.libraries.is_empty() && elf.dynamic.is_none() {
        // No dynamic section — static binary.
        return Ok(SharedLibs {
            found: vec![],
            missing: vec![],
            interpreter,
        });
    }

    // Extract DT_NEEDED, DT_RUNPATH, and DT_RPATH from the root binary.
    let root_needed: Vec<String> = elf.libraries.iter().map(|s| s.to_string()).collect();
    let mut root_search = elf_search_paths(&elf, binary);

    // When the binary uses a non-standard interpreter (custom toolchain),
    // collect the interpreter's parent dir and sibling lib dirs. These
    // are prepended to root search paths and propagated to transitive
    // deps so the custom environment's libs are found BEFORE system libs.
    // Without this, the system libc gets resolved first, causing version
    // mismatches when the custom ld.so loads a libc that requires GLIBC
    // symbols the custom ld.so doesn't provide.
    let interp_search_dirs: Vec<PathBuf> = match interpreter {
        Some(ref interp) if !is_standard_interpreter(interp) => {
            let interp_path = Path::new(interp);
            let mut dirs = Vec::new();
            if let Some(parent) = interp_path.parent() {
                dirs.push(parent.to_path_buf());
                // Sibling lib dirs: e.g. for /opt/toolchain/lib64/ld.so,
                // parent is lib64, so siblings are at parent.parent()/lib
                // and parent.parent()/lib64.
                if let Some(grandparent) = parent.parent() {
                    dirs.push(grandparent.join("lib"));
                    dirs.push(grandparent.join("lib64"));
                }
            }
            dirs
        }
        _ => Vec::new(),
    };
    if !interp_search_dirs.is_empty() {
        let mut combined = interp_search_dirs.clone();
        combined.append(&mut root_search);
        root_search = combined;
    }

    // Resolve the full transitive closure via level-parallel BFS.
    // Each level's file reads (read + ELF parse) run in parallel via
    // rayon. Soname resolution (resolve_soname) is cheap (cache lookups
    // + stat calls), so it stays sequential per level.
    use rayon::prelude::*;

    let mut found: Vec<(String, PathBuf)> = Vec::new();
    let mut missing: Vec<MissingLib> = Vec::new();
    let mut visited = std::collections::HashSet::new();

    // Current level: (soname, is_direct_dep_of_root, search_paths_from_parent)
    let mut level: Vec<(String, bool, Vec<PathBuf>)> = root_needed
        .iter()
        .map(|s| (s.clone(), true, root_search.clone()))
        .collect();

    while !level.is_empty() {
        // Phase 1: resolve sonames to host paths (sequential, cheap).
        let mut resolved: Vec<(String, PathBuf, PathBuf)> = Vec::new();
        for (soname, is_direct, search_paths) in &level {
            if !visited.insert(soname.clone()) {
                continue;
            }
            if let Some(host_path) = resolve_soname(soname, search_paths) {
                let canonical =
                    std::fs::canonicalize(&host_path).unwrap_or_else(|_| host_path.clone());
                let canon_str = canonical.to_string_lossy();
                let canon_guest = canon_str
                    .strip_prefix('/')
                    .unwrap_or(&canon_str)
                    .to_string();
                found.push((canon_guest.clone(), canonical.clone()));

                // Also add the non-canonical path if it differs, so the
                // guest dynamic linker can find the lib via either path.
                let host_str = host_path.to_string_lossy();
                let host_guest = host_str.strip_prefix('/').unwrap_or(&host_str).to_string();
                if host_guest != canon_guest {
                    found.push((host_guest, canonical.clone()));
                }

                resolved.push((soname.clone(), host_path, canonical));
            } else {
                missing.push(MissingLib {
                    soname: soname.clone(),
                    direct: *is_direct,
                });
            }
        }

        // Phase 2: read + parse resolved libs in parallel to discover
        // their DT_NEEDED entries and search paths.
        let next_deps: Vec<(String, Vec<PathBuf>)> = resolved
            .par_iter()
            .flat_map(|(_, _, canonical)| {
                let Ok(lib_data) = std::fs::read(canonical) else {
                    return Vec::new();
                };
                let Ok(lib_elf) = goblin::elf::Elf::parse(&lib_data) else {
                    return Vec::new();
                };
                let mut lib_search = elf_search_paths(&lib_elf, canonical);
                // Propagate interpreter-relative dirs to transitive deps
                // so custom-environment libs resolve consistently.
                if !interp_search_dirs.is_empty() {
                    let mut combined = interp_search_dirs.clone();
                    combined.append(&mut lib_search);
                    lib_search = combined;
                }
                lib_elf
                    .libraries
                    .iter()
                    .map(|name| (name.to_string(), lib_search.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

        // Build next level from discovered deps, skipping already-visited.
        level = next_deps
            .into_iter()
            .filter(|(soname, _)| !visited.contains(soname))
            .map(|(soname, search)| (soname, false, search))
            .collect();
    }

    let result = SharedLibs {
        found,
        missing,
        interpreter,
    };

    if let Ok(mut cache) = CACHE.lock() {
        cache.insert(canon, result.clone());
    }

    Ok(result)
}

/// Extract search paths from DT_RUNPATH (preferred) or DT_RPATH, with
/// dynamic string tokens expanded:
/// - `$ORIGIN` / `${ORIGIN}`: binary's parent directory
/// - `$LIB` / `${LIB}`: `lib` or `lib64` based on ELF class
/// - `$PLATFORM` / `${PLATFORM}`: `x86_64` or `aarch64`
fn elf_search_paths(elf: &goblin::elf::Elf, binary: &Path) -> Vec<PathBuf> {
    let origin = binary
        .parent()
        .and_then(|p| std::fs::canonicalize(p).ok())
        .unwrap_or_default();

    // DT_RUNPATH takes precedence over DT_RPATH when both are present.
    let raw = if !elf.runpaths.is_empty() {
        elf.runpaths.join(":")
    } else if !elf.rpaths.is_empty() {
        elf.rpaths.join(":")
    } else {
        return vec![];
    };

    let origin_str = origin.to_string_lossy();
    // goblin's is_64 distinguishes 64-bit vs 32-bit ELF class.
    let lib_str = if elf.is_64 { "lib64" } else { "lib" };
    let platform_str = std::env::consts::ARCH;

    raw.split(':')
        .map(|p| {
            let expanded = p
                .replace("$ORIGIN", &origin_str)
                .replace("${ORIGIN}", &origin_str)
                .replace("$LIB", lib_str)
                .replace("${LIB}", lib_str)
                .replace("$PLATFORM", platform_str)
                .replace("${PLATFORM}", platform_str);
            PathBuf::from(expanded)
        })
        .collect()
}

/// Well-known system dynamic linker paths. If a binary's PT_INTERP
/// canonicalizes to the same file as one of these, it uses the standard
/// linker and does not need the interpreter packed separately.
const STANDARD_INTERPRETERS: &[&str] = &[
    "/lib/ld-linux.so.2",
    "/lib/ld-linux-aarch64.so.1",
    "/lib/ld-linux-armhf.so.3",
    "/lib64/ld-linux-x86-64.so.2",
    "/lib/ld-musl-x86_64.so.1",
    "/lib/ld-musl-aarch64.so.1",
    "/libexec/ld-elf.so.1",
];

/// Check if `interp` is a standard system linker. Compares the
/// canonicalized path against canonicalized well-known linker paths
/// to catch symlinks (e.g. `/opt/toolchain/lib/ld-linux-x86-64.so.2`
/// symlinking to `/lib64/ld-linux-x86-64.so.2`).
fn is_standard_interpreter(interp: &str) -> bool {
    let interp_path = Path::new(interp);
    // Direct match first (avoids syscalls for common case).
    if STANDARD_INTERPRETERS.contains(&interp) {
        return true;
    }
    // Canonicalize and compare against canonical standard paths.
    let Ok(canon) = std::fs::canonicalize(interp_path) else {
        return false;
    };
    STANDARD_INTERPRETERS.iter().any(|std_interp| {
        std::fs::canonicalize(std_interp).is_ok_and(|std_canon| std_canon == canon)
    })
}

/// Default library search paths used by the dynamic linker.
const DEFAULT_LIB_PATHS: &[&str] = &[
    "/lib",
    "/usr/lib",
    "/lib64",
    "/usr/lib64",
    "/usr/local/lib",
    "/usr/local/lib64",
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
];

/// Directories from the `LD_LIBRARY_PATH` environment variable, parsed
/// once on first access. Empty when the variable is unset or empty.
static LD_LIBRARY_PATH_DIRS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    std::env::var("LD_LIBRARY_PATH")
        .unwrap_or_default()
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
});

/// Resolve a soname to a host path.
/// Search order matches the host dynamic linker (ld.so):
///   1. LD_LIBRARY_PATH
///   2. DT_RUNPATH / DT_RPATH from the binary
///   3. /etc/ld.so.cache (binary cache from ldconfig)
///   4. /etc/ld.so.conf paths (text config directories)
///   5. Default library paths (/lib, /usr/lib, etc.)
fn resolve_soname(soname: &str, elf_search_dirs: &[PathBuf]) -> Option<PathBuf> {
    // 1. LD_LIBRARY_PATH.
    for dir in LD_LIBRARY_PATH_DIRS.iter() {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 2. DT_RUNPATH / DT_RPATH directories.
    for dir in elf_search_dirs {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 3. ld.so.cache — the binary cache is the real dynamic linker's
    //    primary lookup mechanism. Catches libraries in directories
    //    added via `ldconfig /path` that don't appear in ld.so.conf.
    if let Some(cached_path) = LD_SO_CACHE.get(soname) {
        return Some(cached_path.clone());
    }

    // 4. Paths from /etc/ld.so.conf.
    for dir in LD_SO_CONF_PATHS.iter() {
        let candidate = dir.join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    // 5. Default paths.
    for dir in DEFAULT_LIB_PATHS {
        let candidate = Path::new(dir).join(soname);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

/// ELF magic bytes: `\x7fELF`.
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// Check if the first 4 bytes of a file match ELF magic.
fn is_elf(path: &Path) -> bool {
    std::fs::File::open(path)
        .and_then(|mut f| {
            use std::io::Read;
            let mut magic = [0u8; 4];
            f.read_exact(&mut magic)?;
            Ok(magic)
        })
        .is_ok_and(|m| m == *ELF_MAGIC)
}

/// Write one entry (file or directory) into the cpio archive.
fn write_entry(archive: &mut Vec<u8>, name: &str, data: &[u8], mode: u32) -> Result<()> {
    let builder = cpio::newc::Builder::new(name).mode(mode).nlink(1);
    let mut writer = builder.write(archive as &mut dyn Write, data.len() as u32);
    writer
        .write_all(data)
        .with_context(|| format!("write cpio entry '{name}'"))?;
    writer.finish().context("finish cpio entry")?;
    Ok(())
}

/// Write a cpio symlink entry. `name` is the symlink path, `target` is the
/// absolute path it points to. Mode is S_IFLNK | 0777 = 0o120777.
fn write_symlink_entry(archive: &mut Vec<u8>, name: &str, target: &str) -> Result<()> {
    let target_bytes = target.as_bytes();
    let builder = cpio::newc::Builder::new(name).mode(0o120777).nlink(1);
    let mut writer = builder.write(archive as &mut dyn Write, target_bytes.len() as u32);
    writer
        .write_all(target_bytes)
        .with_context(|| format!("write cpio symlink '{name}' -> '{target}'"))?;
    writer.finish().context("finish cpio symlink entry")?;
    Ok(())
}

/// Section names removed during debug stripping. These contain debug
/// info, compiler metadata, and profiling data that inflate the binary
/// but are not needed inside the VM.
const DEBUG_SECTIONS: &[&[u8]] = &[
    b".debug_info",
    b".debug_abbrev",
    b".debug_line",
    b".debug_line_str",
    b".debug_str",
    b".debug_ranges",
    b".debug_aranges",
    b".debug_frame",
    b".debug_loc",
    b".debug_loclists",
    b".debug_rnglists",
    b".debug_str_offsets",
    b".debug_addr",
    b".debug_pubtypes",
    b".debug_pubnames",
    b".debug_types",
    b".debug_macro",
    b".debug_macinfo",
    b".comment",
];

/// Strip debug sections from an ELF binary to reduce initramfs size.
/// Debug info can be 10-50x the loadable segment size and is not needed
/// inside the VM. Uses the `object` crate to parse and rewrite the ELF,
/// removing non-loadable debug sections. Falls back to the original
/// binary on parse or write failure.
///
/// When the binary has been deleted (e.g. by `cargo llvm-cov`),
/// retries via `/proc/self/exe` which remains valid as long as the
/// process is alive.
fn strip_debug(path: &Path) -> Result<Vec<u8>> {
    // Try the original path first, then /proc/self/exe if the binary
    // was deleted (cargo llvm-cov deletes binaries after instrumenting).
    let paths_to_try: Vec<&Path> = if is_deleted_self(path) {
        vec![path, Path::new("/proc/self/exe")]
    } else {
        vec![path]
    };

    for src in &paths_to_try {
        if let Ok(data) = std::fs::read(src) {
            if let Ok(stripped) = strip_debug_sections(&data) {
                return Ok(stripped);
            }
            // object crate failed to parse/write — return unstripped.
            return Ok(data);
        }
    }

    std::fs::read(path).with_context(|| format!("read binary: {}", path.display()))
}

/// Remove debug sections from ELF data using the object crate's
/// build module. Parses the ELF, marks debug sections for deletion,
/// and writes back a new ELF without them.
fn strip_debug_sections(data: &[u8]) -> std::result::Result<Vec<u8>, object::build::Error> {
    let mut builder = object::build::elf::Builder::read(data)?;
    for section in builder.sections.iter_mut() {
        if DEBUG_SECTIONS.contains(&section.name.as_slice()) {
            section.delete = true;
        }
    }
    let mut out = Vec::new();
    builder.write(&mut out)?;
    Ok(out)
}

/// Check if `path` is the current executable and has been deleted.
fn is_deleted_self(path: &Path) -> bool {
    let proc_exe = Path::new("/proc/self/exe");
    let Ok(target) = std::fs::read_link(proc_exe) else {
        return false;
    };
    let target_str = target.to_string_lossy();
    target_str.ends_with(" (deleted)")
        && target_str.trim_end_matches(" (deleted)") == path.to_string_lossy().as_ref()
}

/// Build the base cpio archive: /init binary, extra binaries, and shared
/// libraries. Does NOT include /args, trailer, or 512-byte padding. The
/// returned bytes are a valid cpio prefix that `build_suffix_full` can complete
/// with per-invocation args.
///
/// The test binary is packed as `/init` (the kernel's rdinit entry point).
/// Init setup (mounts, scheduler start, etc.) is handled by the Rust init
/// code in `vmm::rust_init`, which runs when the binary detects PID 1.
///
/// When `busybox` is true, embeds busybox at `bin/busybox` for shell mode.
///
/// `include_files` adds files verbatim to the archive (no strip_debug).
/// Each entry is `(archive_path, host_path)`. ELF files get shared library
/// resolution; non-ELF files are copied as-is. Only regular files are
/// accepted; FIFOs, device nodes, and sockets are rejected. Archive paths
/// must not contain `..` components. Callers expand directories into
/// individual file entries before calling this function (see
/// `cli::resolve_include_files`).
#[tracing::instrument(skip_all, fields(payload = %payload.display(), includes = include_files.len()))]
pub fn create_initramfs_base(
    payload: &Path,
    extra_binaries: &[(&str, &Path)],
    include_files: &[(&str, &Path)],
    busybox: bool,
) -> Result<Vec<u8>> {
    // Validate include_files and collect metadata (reused in the write
    // loop to avoid a second stat syscall per file).
    let mut validated_includes: Vec<(&str, &Path, u32)> = Vec::with_capacity(include_files.len());
    for (archive_path, host_path) in include_files {
        // Reject path traversal.
        if Path::new(archive_path)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            anyhow::bail!("include_files archive path contains '..': {}", archive_path);
        }
        // Reject paths that collide with internal sentinel files.
        if archive_path.starts_with(".ktstr_") {
            anyhow::bail!(
                "include_files archive path must not start with '.ktstr_': {}",
                archive_path
            );
        }
        // Reject non-regular files (FIFOs, device nodes, sockets block or
        // produce garbage).
        let meta = std::fs::metadata(host_path).with_context(|| {
            format!(
                "stat include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        if !meta.file_type().is_file() {
            anyhow::bail!(
                "include_files entry '{}' is not a regular file: {}",
                archive_path,
                host_path.display()
            );
        }
        validated_includes.push((archive_path, host_path, meta.permissions().mode()));
    }

    let binary = {
        let _s = tracing::debug_span!("strip_debug").entered();
        strip_debug(payload).with_context(|| format!("strip/read binary: {}", payload.display()))?
    };
    let mut archive = Vec::new();

    // Collect directory entries needed for shared libraries and includes.
    let mut dirs = BTreeSet::new();

    // Resolve shared library dependencies for init binary and extras.
    let mut shared_libs: Vec<(String, PathBuf)> = Vec::new();
    let mut all_binaries: Vec<&Path> = std::iter::once(payload)
        .chain(extra_binaries.iter().map(|(_, p)| *p))
        .collect();

    // ELF files from include_files join the shared lib resolution chain.
    let mut include_elf_paths: Vec<&Path> = Vec::new();
    for (_, host_path) in include_files {
        if is_elf(host_path) {
            include_elf_paths.push(host_path);
            all_binaries.push(host_path);
        }
    }

    let _s_resolve = tracing::debug_span!("resolve_all_libs", count = all_binaries.len()).entered();
    for path in &all_binaries {
        let _s_one =
            tracing::debug_span!("resolve_shared_libs", binary = %path.display()).entered();
        let result = resolve_shared_libs(path)
            .with_context(|| format!("resolve libs for {}", path.display()))?;
        drop(_s_one);

        // Include-file ELFs must have all shared libs resolvable.
        if !result.missing.is_empty() && include_elf_paths.contains(path) {
            let names: Vec<&str> = result.missing.iter().map(|m| m.soname.as_str()).collect();
            anyhow::bail!(
                "{}: missing shared libraries: {}",
                path.display(),
                names.join(", ")
            );
        }

        // Pack PT_INTERP (dynamic linker) into the initramfs. The
        // interpreter is not a DT_NEEDED entry and won't appear in the
        // resolved shared libs, so it must be added explicitly.
        // For non-standard interpreters, also resolve their own deps.
        tracing::debug!(
            binary = %path.display(),
            interpreter = ?result.interpreter,
            is_include = include_elf_paths.contains(path),
            "resolved interpreter for binary"
        );
        if let Some(ref interp) = result.interpreter {
            let interp_path = Path::new(interp);
            let is_standard = is_standard_interpreter(interp);
            tracing::debug!(
                interp = %interp_path.display(),
                exists = interp_path.is_file(),
                is_standard,
                "interpreter details"
            );
            if interp_path.is_file() {
                let canonical = std::fs::canonicalize(interp_path)
                    .unwrap_or_else(|_| interp_path.to_path_buf());
                let canon_str = canonical.to_string_lossy();
                let guest = canon_str
                    .strip_prefix('/')
                    .unwrap_or(&canon_str)
                    .to_string();
                if let Some(parent) = Path::new(&guest).parent() {
                    let mut dir = PathBuf::new();
                    for component in parent.components() {
                        dir.push(component);
                        dirs.insert(dir.to_string_lossy().to_string());
                    }
                }
                tracing::debug!(
                    canonical_guest = %guest,
                    canonical_host = %canonical.display(),
                    "packing interpreter canonical path"
                );
                shared_libs.push((guest.clone(), canonical.clone()));

                // Also add the non-canonical path if it differs.
                let orig_guest = interp.strip_prefix('/').unwrap_or(interp).to_string();
                if orig_guest != guest {
                    tracing::debug!(
                        orig_guest = %orig_guest,
                        canonical_guest = %guest,
                        "packing interpreter original (non-canonical) path"
                    );
                    if let Some(parent) = Path::new(&orig_guest).parent() {
                        let mut dir = PathBuf::new();
                        for component in parent.components() {
                            dir.push(component);
                            dirs.insert(dir.to_string_lossy().to_string());
                        }
                    }
                    shared_libs.push((orig_guest, canonical));
                } else {
                    tracing::debug!("interpreter original path matches canonical, no alias needed");
                }

                // Non-standard interpreters may have their own shared lib
                // deps (custom toolchain linkers alongside their libs).
                if !is_standard_interpreter(interp)
                    && let Ok(interp_result) = resolve_shared_libs(interp_path)
                {
                    for (g, h) in interp_result.found {
                        if let Some(parent) = Path::new(&g).parent() {
                            let mut dir = PathBuf::new();
                            for component in parent.components() {
                                dir.push(component);
                                dirs.insert(dir.to_string_lossy().to_string());
                            }
                        }
                        shared_libs.push((g, h));
                    }
                }
            }
        }

        for (guest_path, host_path) in result.found {
            if let Some(parent) = Path::new(&guest_path).parent() {
                let mut dir = PathBuf::new();
                for component in parent.components() {
                    dir.push(component);
                    dirs.insert(dir.to_string_lossy().to_string());
                }
            }
            shared_libs.push((guest_path, host_path));
        }
    }
    let pre_dedup_count = shared_libs.len();
    shared_libs.sort_by(|a, b| a.0.cmp(&b.0));
    shared_libs.dedup_by(|a, b| a.0 == b.0);
    tracing::debug!(
        pre_dedup = pre_dedup_count,
        post_dedup = shared_libs.len(),
        removed = pre_dedup_count - shared_libs.len(),
        "shared_libs dedup"
    );

    // Busybox needs bin/ directory.
    if busybox {
        dirs.insert("bin".to_string());
    }
    // Include files need their parent directories in the cpio archive.
    // The component walk produces all ancestors (e.g. "include-files/sub/f"
    // yields "include-files" and "include-files/sub").
    for (archive_path, _, _) in &validated_includes {
        if let Some(parent) = Path::new(archive_path).parent() {
            let mut dir = PathBuf::new();
            for component in parent.components() {
                dir.push(component);
                dirs.insert(dir.to_string_lossy().to_string());
            }
        }
    }

    drop(_s_resolve);

    tracing::debug!(
        shared_libs_count = shared_libs.len(),
        dirs_count = dirs.len(),
        dirs = ?dirs,
        shared_libs_guests = ?shared_libs.iter().map(|(g, _)| g.as_str()).collect::<Vec<_>>(),
        "pre-write archive contents"
    );

    let _s_write = tracing::debug_span!("write_cpio").entered();
    // Directory entries
    for dir in &dirs {
        write_entry(&mut archive, dir, &[], 0o40755)?;
    }

    // Test binary as /init — the Rust init code detects PID 1 and performs
    // all setup (mounts, scheduler, etc.) before running the test function.
    write_entry(&mut archive, "init", &binary, 0o100755)?;

    // Shell mode: embed busybox.
    if busybox {
        write_entry(&mut archive, "bin/busybox", crate::BUSYBOX, 0o100755)?;
    }

    // Extra binaries (stripped to reduce initramfs size)
    for (name, path) in extra_binaries {
        let data = strip_debug(path)
            .with_context(|| format!("strip/read extra binary '{}': {}", name, path.display()))?;
        write_entry(&mut archive, name, &data, 0o100755)?;
    }

    // Include files: copied verbatim, preserving original content and
    // debug symbols. No strip_debug — included files are user-provided
    // and may be non-ELF.
    for (archive_path, host_path, mode) in &validated_includes {
        let data = std::fs::read(host_path).with_context(|| {
            format!(
                "read include file '{}': {}",
                archive_path,
                host_path.display()
            )
        })?;
        write_entry(&mut archive, archive_path, &data, *mode)?;
    }

    // Shared libraries — write each canonical host file once as a regular
    // file, then write subsequent guest paths that map to the same host
    // file as cpio symlinks. This avoids duplicating large libraries in
    // the initramfs (e.g. libc appearing under both lib64/ and usr/lib64/).
    {
        // canonical host path -> first guest_path written for this file
        let mut written_files: HashMap<PathBuf, String> = HashMap::new();
        for (guest_path, host_path) in &shared_libs {
            let canonical = std::fs::canonicalize(host_path).unwrap_or_else(|_| host_path.clone());
            if let Some(first_guest) = written_files.get(&canonical) {
                // Already written — emit a symlink to the first guest path.
                let target = format!("/{first_guest}");
                write_symlink_entry(&mut archive, guest_path, &target)?;
            } else {
                let data = std::fs::read(host_path).with_context(|| {
                    format!("read shared lib '{}': {}", guest_path, host_path.display())
                })?;
                write_entry(&mut archive, guest_path, &data, 0o100755)?;
                written_files.insert(canonical, guest_path.clone());
            }
        }
    }

    // Sentinel: last entry before the suffix. The guest init checks for
    // this file to detect incomplete initramfs extraction.
    write_entry(&mut archive, ".ktstr_init_ok", &[], 0o100644)?;

    drop(_s_write);

    Ok(archive)
}

/// Build the suffix that completes a base archive: /args and /sched_args
/// entries, trailer, and 512-byte padding. `base_len` is needed to compute
/// the padding. The returned Vec is typically ~200 bytes.
#[cfg(test)]
pub(crate) fn build_suffix(
    base_len: usize,
    args: &[String],
    sched_args: &[String],
) -> Result<Vec<u8>> {
    build_suffix_full(base_len, args, sched_args, &[], &[], None)
}

/// Extended suffix builder that also writes /sched_enable and /sched_disable
/// shell scripts for kernel-built schedulers.
pub fn build_suffix_full(
    base_len: usize,
    args: &[String],
    sched_args: &[String],
    sched_enable: &[&str],
    sched_disable: &[&str],
    exec_cmd: Option<&str>,
) -> Result<Vec<u8>> {
    let mut suffix = Vec::new();

    // Args file
    let args_data = args.join("\n");
    write_entry(&mut suffix, "args", args_data.as_bytes(), 0o100644)?;

    // Scheduler args file
    if !sched_args.is_empty() {
        let sched_args_data = sched_args.join("\n");
        write_entry(
            &mut suffix,
            "sched_args",
            sched_args_data.as_bytes(),
            0o100644,
        )?;
    }

    // Kernel-built scheduler enable/disable scripts
    if !sched_enable.is_empty() {
        let data = sched_enable.join("\n");
        write_entry(&mut suffix, "sched_enable", data.as_bytes(), 0o100755)?;
    }
    if !sched_disable.is_empty() {
        let data = sched_disable.join("\n");
        write_entry(&mut suffix, "sched_disable", data.as_bytes(), 0o100755)?;
    }

    if let Some(cmd) = exec_cmd {
        write_entry(&mut suffix, "exec_cmd", cmd.as_bytes(), 0o100644)?;
    }

    // Trailer
    cpio::newc::trailer(&mut suffix as &mut dyn Write).context("write cpio trailer")?;

    // Pad to 512-byte boundary (initramfs convention)
    let total = base_len + suffix.len();
    let pad = (512 - (total % 512)) % 512;
    suffix.extend(std::iter::repeat_n(0u8, pad));

    Ok(suffix)
}

/// Create a complete cpio newc archive in one call.
/// Convenience wrapper over `create_initramfs_base` + `build_suffix`.
#[cfg(test)]
pub fn create_initramfs(
    payload: &Path,
    extra_binaries: &[(&str, &Path)],
    args: &[String],
) -> Result<Vec<u8>> {
    let base = create_initramfs_base(payload, extra_binaries, &[], false)?;
    let suffix = build_suffix(base.len(), args, &[])?;
    let mut archive = Vec::with_capacity(base.len() + suffix.len());
    archive.extend_from_slice(&base);
    archive.extend_from_slice(&suffix);
    Ok(archive)
}

// ---------------------------------------------------------------------------
// POSIX shared-memory cache for base initramfs
// ---------------------------------------------------------------------------

/// Derive an shm segment name from a content hash. Each distinct
/// combination of payload + scheduler binaries gets its own segment.
pub(crate) fn shm_segment_name(content_hash: u64) -> String {
    format!("/ktstr-base-{content_hash:016x}")
}

/// Read-only mmap of a POSIX shared-memory segment. The mapping stays
/// live until the struct is dropped, so callers can borrow the bytes
/// without copying the entire base archive.
///
/// Holds the shared flock (`LOCK_SH`) for the lifetime of the mapping
/// so that a concurrent writer cannot `ftruncate` the segment beneath
/// us, which would cause `SIGBUS` on access to the truncated pages.
pub(crate) struct MappedShm {
    ptr: *const u8,
    len: usize,
    fd: std::os::unix::io::RawFd,
}

// SAFETY: The mmap is MAP_SHARED|PROT_READ over a shm segment whose
// contents are held stable for the mapping's lifetime by a shared
// flock retained in `fd`. The pointer and length are valid for the
// lifetime of the mapping.
unsafe impl Send for MappedShm {}
unsafe impl Sync for MappedShm {}

impl AsRef<[u8]> for MappedShm {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: ptr/len are set by a successful mmap in
        // shm_load_base, and the SHM segment's contents are held
        // stable for the mapping's lifetime by the shared flock
        // retained in self.fd — a cooperating writer cannot
        // ftruncate the segment out from under us.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MappedShm {
    fn drop(&mut self) {
        // SAFETY: ptr/len are from mmap; fd is from shm_open with
        // LOCK_SH held. Release the mapping first, then the flock, then
        // the fd — order matters so the lock protects the mapping right
        // up until it is torn down.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
            libc::flock(self.fd, libc::LOCK_UN);
            libc::close(self.fd);
        }
    }
}

/// Try to mmap a base initramfs from a POSIX shared-memory segment
/// identified by `content_hash`. Returns a `MappedShm` that borrows
/// the data without copying. Returns `None` on miss or error.
///
/// Acquires a shared flock (`LOCK_SH`) before mmap and keeps it held
/// for the lifetime of the returned `MappedShm`. A concurrent writer
/// calls `ftruncate` under `LOCK_EX` in `shm_store` (and in
/// `shm_write_and_release`, which also `ftruncate`s to 0 on mmap
/// failure); holding `LOCK_SH` for the mapping's lifetime prevents
/// either writer from truncating the segment out from under us, which
/// would turn any access to the truncated pages into `SIGBUS`.
///
/// Note: `flock` is advisory — it only protects against other
/// processes that also call `flock`. A process that writes the
/// segment without taking `LOCK_EX` (e.g. `rm /dev/shm/…` + recreate
/// by an unrelated tool) bypasses this scheme. All callers within
/// this crate cooperate, which is the closed-world guarantee we
/// rely on.
pub(crate) fn shm_load_base(content_hash: u64) -> Option<MappedShm> {
    let name = std::ffi::CString::new(shm_segment_name(content_hash)).ok()?;
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
        if fd < 0 {
            return None;
        }

        // Shared lock -- blocks until any concurrent writer releases
        // LOCK_EX. Held for the mapping's lifetime; released in
        // MappedShm::drop.
        if libc::flock(fd, libc::LOCK_SH) != 0 {
            libc::close(fd);
            return None;
        }

        // Get segment size via fstat.
        let mut stat: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut stat) != 0 || stat.st_size <= 0 {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            return None;
        }
        let len = stat.st_size as usize;

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            fd,
            0,
        );

        if ptr == libc::MAP_FAILED {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            return None;
        }

        Some(MappedShm {
            ptr: ptr as *const u8,
            len,
            fd,
        })
    }
}

/// Write `data` to a POSIX SHM segment identified by `name`.
///
/// Creates (or opens existing) the segment with `O_CREAT | O_RDWR`,
/// takes an exclusive flock, `ftruncate`s to `data.len()`, `mmap`s
/// `PROT_WRITE | MAP_SHARED`, copies, and cleans up.
///
/// Concurrency: `LOCK_EX` blocks while any reader holds `LOCK_SH` on
/// the same segment (e.g. a live `MappedShm` or `CowOverlayGuard`).
/// The writer thus waits for in-flight VMs before truncating — which
/// is what prevents the `SIGBUS` class of bug addressed by the
/// reader-side flock lifetime in `shm_load_base` and `cow_overlay`.
///
/// Writes are content-addressed at the caller: callers hash `data`
/// to form the segment name. When two callers write the same content
/// to the same hash, the payload length and bytes are identical, so
/// the second `ftruncate(same_len)` is a no-op on page contents and
/// the second memcpy writes the same bytes. A third-party caller
/// that writes DIFFERENT data to an already-used hash (e.g. the
/// rename-test pattern) will overwrite — the store does not enforce
/// idempotence itself.
fn shm_store(name: &std::ffi::CStr, data: &[u8]) -> Result<()> {
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_CREAT | libc::O_RDWR, 0o644);
        anyhow::ensure!(fd >= 0, "shm_open: {}", std::io::Error::last_os_error());

        // Surfaces the wait explicitly: with readers holding LOCK_SH
        // for VM lifetime, concurrent test runs can block here for
        // seconds. Without this, the user sees silent hang.
        tracing::info!(
            segment = name.to_string_lossy().as_ref(),
            data_len = data.len(),
            "shm_store: waiting for LOCK_EX"
        );
        if libc::flock(fd, libc::LOCK_EX) != 0 {
            libc::close(fd);
            anyhow::bail!("flock: {}", std::io::Error::last_os_error());
        }

        if libc::ftruncate(fd, data.len() as libc::off_t) != 0 {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            anyhow::bail!("ftruncate: {}", std::io::Error::last_os_error());
        }

        let ptr = libc::mmap(
            std::ptr::null_mut(),
            data.len(),
            libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        );
        if ptr == libc::MAP_FAILED {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            anyhow::bail!("mmap: {}", std::io::Error::last_os_error());
        }

        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        libc::munmap(ptr, data.len());

        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
    }
    Ok(())
}

pub(crate) fn shm_store_base(content_hash: u64, data: &[u8]) -> Result<()> {
    let name =
        std::ffi::CString::new(shm_segment_name(content_hash)).context("shm segment name")?;
    shm_store(&name, data)
}

/// Remove the POSIX shared-memory segment identified by `content_hash`.
#[allow(dead_code)]
pub(crate) fn shm_unlink_base(content_hash: u64) {
    if let Ok(name) = std::ffi::CString::new(shm_segment_name(content_hash)) {
        unsafe {
            libc::shm_unlink(name.as_ptr());
        }
    }
}

// ---------------------------------------------------------------------------
// Compressed SHM cache — stores LZ4-compressed base for COW overlay into
// guest RAM
// ---------------------------------------------------------------------------

/// Segment name for the LZ4-compressed version of a base initramfs.
/// Uses `lz4` prefix to avoid collisions with segments written by
/// previous compression formats (zstd, gzip).
fn shm_lz4_segment_name(content_hash: u64) -> String {
    format!("/ktstr-lz4-{content_hash:016x}")
}

/// Open the compressed SHM segment and return a held fd + size.
/// The fd has a shared flock held — caller must close it when done.
/// Returns `None` on miss or error.
pub(crate) fn shm_open_lz4(content_hash: u64) -> Option<(std::os::unix::io::RawFd, usize)> {
    let name = std::ffi::CString::new(shm_lz4_segment_name(content_hash)).ok()?;
    unsafe {
        let fd = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
        if fd < 0 {
            return None;
        }
        if libc::flock(fd, libc::LOCK_SH) != 0 {
            libc::close(fd);
            return None;
        }
        let mut stat: libc::stat = std::mem::zeroed();
        if libc::fstat(fd, &mut stat) != 0 || stat.st_size <= 0 {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
            return None;
        }
        Some((fd, stat.st_size as usize))
    }
}

/// Store compressed initramfs data into an LZ4 SHM segment.
pub(crate) fn shm_store_lz4(content_hash: u64, data: &[u8]) -> Result<()> {
    let name =
        std::ffi::CString::new(shm_lz4_segment_name(content_hash)).context("shm lz4 name")?;
    shm_store(&name, data)
}

/// RAII guard for a live COW-overlay mapping.
///
/// A COW overlay is `MAP_PRIVATE | MAP_FIXED` onto guest memory from
/// a SHM segment fd. `MAP_PRIVATE` pages are lazily read from the
/// backing file on first access; if the SHM segment is truncated or
/// unlinked-with-retruncate between `mmap` and the guest's first
/// read, the access SIGBUSes (see Linux `filemap_fault` against
/// `i_size`). Holding the fd with `LOCK_SH` for the mapping's
/// lifetime blocks any cooperating writer from taking `LOCK_EX` and
/// `ftruncate`ing the segment until after the mapping is torn down.
///
/// Drop order: the guard releases `LOCK_UN` and `close` only. The
/// MAP_FIXED region itself is owned by the caller's VA reservation
/// (e.g. `ReservationGuard` in the VMM) and is munmapped when that
/// reservation drops — which must happen BEFORE this guard drops,
/// so the lock protects the mapping right up until tear-down.
pub(crate) struct CowOverlayGuard {
    fd: std::os::unix::io::RawFd,
}

impl CowOverlayGuard {
    fn new(fd: std::os::unix::io::RawFd) -> Self {
        Self { fd }
    }
}

// SAFETY: The fd is owned by this guard; no other code reads or
// closes it. flock/close are thread-safe syscalls.
unsafe impl Send for CowOverlayGuard {}
unsafe impl Sync for CowOverlayGuard {}

impl Drop for CowOverlayGuard {
    fn drop(&mut self) {
        // SAFETY: fd was obtained via shm_open in the COW overlay
        // path; we own it and release LOCK_SH before close.
        unsafe {
            libc::flock(self.fd, libc::LOCK_UN);
            libc::close(self.fd);
        }
    }
}

/// COW-overlay `len` bytes from `shm_fd` at `host_addr` using
/// `MAP_PRIVATE | MAP_FIXED | MAP_POPULATE`. The guest sees the SHM
/// content but writes go to private anonymous pages (copy-on-write).
/// `MAP_POPULATE` pre-faults the pages so the initial accesses skip
/// the filemap fault path; the lock guard still protects against
/// truncate of pages that may be refaulted from the page cache
/// (MAP_POPULATE alone is not sufficient — truncate invalidates the
/// page cache via `unmap_mapping_range`).
///
/// On success, returns `Some(CowOverlayGuard)` — the guard owns
/// `shm_fd` and holds `LOCK_SH` for the mapping's lifetime. The
/// caller MUST keep the guard alive for as long as the MAP_FIXED
/// mapping is in use (typically the VM lifetime) and drop the guard
/// AFTER the VA region is munmapped.
///
/// On failure, returns `None` and CLOSES `shm_fd` (releasing
/// `LOCK_SH`) so the caller does not need to clean it up.
///
/// # Safety
///
/// The caller MUST have validated that the entire range
/// `[host_addr, host_addr + len)` lies within one contiguous guest
/// memory region. `MAP_FIXED` unmaps whatever is already present
/// across the full range and replaces it with the new mapping; if
/// `len` extends past the region, `MAP_FIXED` silently corrupts
/// unrelated host mappings (kernel-defined behaviour at the mmap
/// layer) and violates Rust's aliasing invariants at the process
/// level. The caller is also responsible for ensuring `shm_fd` is a
/// valid, open file descriptor with `LOCK_SH` already held (the
/// guard inherits both).
pub(crate) unsafe fn cow_overlay(
    host_addr: *mut u8,
    len: usize,
    shm_fd: std::os::unix::io::RawFd,
) -> Option<CowOverlayGuard> {
    // SAFETY: caller guarantees [host_addr, host_addr + len) is
    // entirely within a single valid guest memory region and shm_fd
    // is a valid fd holding LOCK_SH. See function-level docs.
    let ptr = unsafe {
        libc::mmap(
            host_addr as *mut libc::c_void,
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_FIXED | libc::MAP_POPULATE,
            shm_fd,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        // Close and unlock fd ourselves — caller expects no cleanup
        // responsibility on the None path.
        unsafe {
            libc::flock(shm_fd, libc::LOCK_UN);
            libc::close(shm_fd);
        }
        return None;
    }
    Some(CowOverlayGuard::new(shm_fd))
}

/// Close a SHM fd and release its shared flock.
pub(crate) fn shm_close_fd(fd: std::os::unix::io::RawFd) {
    unsafe {
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
    }
}

/// Load an initramfs into guest memory at the given address.
/// Returns (address, size) for boot_params.
#[cfg(test)]
pub fn load_initramfs(
    guest_mem: &vm_memory::GuestMemoryMmap,
    initrd_data: &[u8],
    load_addr: u64,
) -> Result<(u64, u32)> {
    use vm_memory::{Bytes, GuestAddress};
    guest_mem
        .write_slice(initrd_data, GuestAddress(load_addr))
        .context("write initramfs to guest memory")?;
    Ok((load_addr, initrd_data.len() as u32))
}

/// Write multiple byte slices sequentially into guest memory as a single
/// contiguous initramfs. Avoids copying parts into a single Vec first.
/// Returns (address, total_size) for boot_params.
pub fn load_initramfs_parts(
    guest_mem: &vm_memory::GuestMemoryMmap,
    parts: &[&[u8]],
    load_addr: u64,
) -> Result<(u64, u32)> {
    use vm_memory::{Bytes, GuestAddress};
    let mut offset = 0u64;
    for part in parts {
        guest_mem
            .write_slice(part, GuestAddress(load_addr + offset))
            .context("write initramfs part to guest memory")?;
        offset += part.len() as u64;
    }
    Ok((load_addr, offset as u32))
}

/// LZ4 legacy format magic number (`0x184C2102` little-endian).
/// This is the format the kernel's initramfs decompressor expects
/// (CONFIG_RD_LZ4 / lib/decompress_unlz4.c).
pub(crate) const LZ4_LEGACY_MAGIC: [u8; 4] = 0x184C2102u32.to_le_bytes();

/// Maximum uncompressed chunk size for LZ4 legacy format.
/// Must match `LZ4_DEFAULT_UNCOMPRESSED_CHUNK_SIZE` in the kernel
/// (lib/decompress_unlz4.c: `8 << 20`).
const LZ4_CHUNK_SIZE: usize = 8 << 20;

/// Compress `data` into LZ4 legacy frame format for the kernel's
/// initramfs decompressor. The format is:
///   [4-byte magic] ([4-byte compressed_size LE] [compressed block])*
///
/// Input is split into `LZ4_CHUNK_SIZE` (8MB) chunks, compressed in
/// parallel with rayon, then assembled sequentially.
pub(crate) fn lz4_legacy_compress(data: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;

    // Compress all chunks in parallel.
    let compressed_chunks: Vec<Vec<u8>> = data
        .par_chunks(LZ4_CHUNK_SIZE)
        .map(lz4_flex::block::compress)
        .collect();

    // Assemble: magic + (size + data) per chunk.
    let total: usize = 4 + compressed_chunks.iter().map(|c| 4 + c.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&LZ4_LEGACY_MAGIC);
    for chunk in &compressed_chunks {
        out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extract cpio entry names from a newc archive for test assertions.
    fn cpio_entry_names(archive: &[u8]) -> Vec<String> {
        let mut names = Vec::new();
        let mut remaining: &[u8] = archive;
        while let Ok(reader) = cpio::newc::Reader::new(remaining) {
            let name = reader.entry().name().to_string();
            if reader.entry().is_trailer() {
                break;
            }
            names.push(name);
            remaining = reader.finish().unwrap();
        }
        names
    }

    /// Extract cpio entries with name, size, mode, and inode for diagnostics.
    fn cpio_entries(archive: &[u8]) -> Vec<(String, u32, u32, u32)> {
        let mut entries = Vec::new();
        let mut remaining: &[u8] = archive;
        while let Ok(reader) = cpio::newc::Reader::new(remaining) {
            if reader.entry().is_trailer() {
                break;
            }
            let name = reader.entry().name().to_string();
            let size = reader.entry().file_size();
            let mode = reader.entry().mode();
            let ino = reader.entry().ino();
            entries.push((name, size, mode, ino));
            remaining = reader.finish().unwrap();
        }
        entries
    }

    #[test]
    fn cpio_header_format() {
        let mut archive = Vec::new();
        write_entry(&mut archive, "test", b"hello", 0o100644).unwrap();
        assert_eq!(&archive[..6], b"070701");
    }

    #[test]
    fn cpio_trailer() {
        let mut archive = Vec::new();
        write_entry(&mut archive, "test", b"data", 0o100755).unwrap();
        cpio::newc::trailer(&mut archive as &mut dyn std::io::Write).unwrap();
        let s = String::from_utf8_lossy(&archive);
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn create_initramfs_has_init() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        let s = String::from_utf8_lossy(&initrd);
        assert!(s.contains("init"), "should contain init entry");
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn create_initramfs_base_is_valid_cpio() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs_base(&exe, &[], &[], false).unwrap();
        assert_eq!(&initrd[..6], b"070701");
        // Base is NOT 512-aligned on its own; only base+suffix is.
        let full = create_initramfs(&exe, &[], &[]).unwrap();
        assert!(initrd.len() <= full.len());
    }

    #[test]
    fn create_initramfs_padded() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        assert_eq!(initrd.len() % 512, 0);
    }

    #[test]
    fn initramfs_nonexistent_file() {
        let result = create_initramfs(Path::new("/nonexistent"), &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn initramfs_nonexistent_extra_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        let result = create_initramfs(&exe, &[("bad", Path::new("/nonexistent"))], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn initramfs_with_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let args = vec!["run".into(), "--json".into(), "scenario".into()];
        let initrd = create_initramfs(&exe, &[], &args).unwrap();
        let s = String::from_utf8_lossy(&initrd);
        assert!(s.contains("args"));
    }

    #[test]
    fn initramfs_empty_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let initrd = create_initramfs(&exe, &[], &[]).unwrap();
        assert_eq!(initrd.len() % 512, 0);
    }

    // -- base + suffix split tests --

    #[test]
    fn suffix_adds_args_and_trailer() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let args = vec!["run".into(), "--json".into()];
        let suffix = build_suffix(base.len(), &args, &[]).unwrap();
        let s = String::from_utf8_lossy(&suffix);
        assert!(s.contains("args"), "suffix should contain args entry");
        assert!(s.contains("TRAILER!!!"), "suffix should contain trailer");
        assert_eq!(
            (base.len() + suffix.len()) % 512,
            0,
            "base+suffix should be 512-byte aligned"
        );
    }

    #[test]
    fn split_matches_monolithic() {
        let exe = crate::resolve_current_exe().unwrap();
        let args = vec!["run".into(), "--json".into(), "scenario".into()];
        let monolithic = create_initramfs(&exe, &[], &args).unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let suffix = build_suffix(base.len(), &args, &[]).unwrap();
        let mut split = Vec::with_capacity(base.len() + suffix.len());
        split.extend_from_slice(&base);
        split.extend_from_slice(&suffix);
        assert_eq!(
            monolithic, split,
            "split path should produce identical output"
        );
    }

    #[test]
    fn suffix_different_args_differ() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let a = build_suffix(base.len(), &["a".into()], &[]).unwrap();
        let b = build_suffix(base.len(), &["b".into()], &[]).unwrap();
        assert_ne!(a, b, "different args should produce different suffixes");
    }

    #[test]
    fn suffix_empty_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let suffix = build_suffix(base.len(), &[], &[]).unwrap();
        assert_eq!((base.len() + suffix.len()) % 512, 0);
        let s = String::from_utf8_lossy(&suffix);
        assert!(s.contains("TRAILER!!!"));
    }

    #[test]
    fn try_cow_overlay_rejects_cross_region_span() {
        // The bounds check in try_cow_overlay relies on
        // GuestMemoryMmap::get_slice failing when a range would cross
        // a region boundary. This test locks that semantic in: two
        // non-contiguous regions; a range that starts in region A but
        // extends past its end must be rejected. If this ever passes
        // (e.g. vm-memory swaps in multi-region get_slices semantics
        // here), try_cow_overlay's MAP_FIXED would silently clobber
        // whatever host mapping sits between the regions.
        use vm_memory::{GuestAddress, GuestMemory};
        let region_a_size: usize = 64 * 1024;
        let region_b_size: usize = 64 * 1024;
        let region_a_start: u64 = 0;
        let region_b_start: u64 = 1 << 20; // 1 MiB gap
        let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[
            (GuestAddress(region_a_start), region_a_size),
            (GuestAddress(region_b_start), region_b_size),
        ])
        .unwrap();

        // Range fully inside region A: must succeed.
        assert!(
            mem.get_slice(GuestAddress(region_a_start), region_a_size)
                .is_ok(),
            "full-region slice must succeed"
        );

        // Range starting mid-region-A and extending past region A's
        // end: must fail. This is the exact shape of the hazardous
        // cow_overlay case.
        let overrun_start = region_a_start + (region_a_size as u64 / 2);
        let overrun_len = region_a_size; // well past the region's end
        assert!(
            mem.get_slice(GuestAddress(overrun_start), overrun_len)
                .is_err(),
            "cross-boundary slice must fail"
        );

        // Range starting at a GPA inside the gap between regions:
        // also fails (no region covers the start address).
        let gap_addr = (region_a_start + region_a_size as u64) + 0x1000;
        assert!(
            mem.get_slice(GuestAddress(gap_addr), 4).is_err(),
            "gap-start slice must fail"
        );
    }

    #[test]
    fn try_cow_overlay_preserves_adjacent_region_bytes() {
        // Proves the invariant at the application layer: with the
        // bounds check in place, we never invoke mmap(MAP_FIXED),
        // which means bytes outside the validated range stay
        // untouched. We simulate "before" bytes in region B, run the
        // same bounds check try_cow_overlay uses, observe that it
        // rejects the request, and verify region B's bytes survive.
        use vm_memory::{Bytes, GuestAddress, GuestMemory};
        let region_a_size: usize = 64 * 1024;
        let region_b_size: usize = 64 * 1024;
        let region_a_start: u64 = 0;
        let region_b_start: u64 = 1 << 20;
        let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[
            (GuestAddress(region_a_start), region_a_size),
            (GuestAddress(region_b_start), region_b_size),
        ])
        .unwrap();

        // Seed region B with a detectable marker.
        let marker: Vec<u8> = (0..region_b_size).map(|i| (i & 0xff) as u8).collect();
        mem.write_slice(&marker, GuestAddress(region_b_start))
            .unwrap();

        // Compute an oversized COW request: starts in region A, len
        // spans the whole guest range up to the end of region B.
        let overrun_load_addr = region_a_start;
        let overrun_len = (region_b_start + region_b_size as u64) as usize;

        // This is the same check try_cow_overlay uses; on failure it
        // returns early and never invokes cow_overlay. We assert the
        // rejection and the preservation of region B's contents.
        assert!(
            mem.get_slice(GuestAddress(overrun_load_addr), overrun_len)
                .is_err(),
            "oversized overlay must be rejected before MAP_FIXED"
        );
        let mut readback = vec![0u8; region_b_size];
        mem.read_slice(&mut readback, GuestAddress(region_b_start))
            .unwrap();
        assert_eq!(
            readback, marker,
            "region B must be untouched when bounds check rejects cow_overlay"
        );
    }

    #[test]
    fn load_initramfs_parts_sequential() {
        let part1 = vec![0xAAu8; 4096];
        let part2 = vec![0xBBu8; 512];
        let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[(
            vm_memory::GuestAddress(0),
            16 << 20,
        )])
        .unwrap();
        let (addr, size) = load_initramfs_parts(&mem, &[&part1, &part2], 0x200000).unwrap();
        assert_eq!(addr, 0x200000);
        assert_eq!(size, 4608);
        let mut buf = vec![0u8; 4608];
        use vm_memory::{Bytes, GuestAddress};
        mem.read_slice(&mut buf, GuestAddress(0x200000)).unwrap();
        assert_eq!(&buf[..4096], &part1[..]);
        assert_eq!(&buf[4096..], &part2[..]);
    }

    // -- shared lib resolution tests --

    #[test]
    fn resolve_shared_libs_nonexistent_returns_error() {
        let result = resolve_shared_libs(Path::new("/nonexistent/binary"));
        // Nonexistent file cannot be read.
        assert!(result.is_err());
    }

    #[test]
    fn resolve_shared_libs_non_elf_returns_empty() {
        let tmp = std::env::temp_dir().join("ktstr-test-resolve-nonelf");
        std::fs::write(&tmp, b"not an elf").unwrap();
        let result = resolve_shared_libs(&tmp).unwrap();
        assert!(result.found.is_empty());
        assert!(result.missing.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn resolve_shared_libs_dynamic_binary() {
        let sh = Path::new("/bin/sh");
        if sh.exists() {
            let shared = resolve_shared_libs(sh).unwrap();
            if !shared.found.is_empty() {
                assert!(
                    shared.found.iter().any(|(g, _)| g.contains("libc")),
                    "dynamic binary should depend on libc: {:?}",
                    shared.found
                );
                for (g, _) in &shared.found {
                    assert!(!g.starts_with('/'), "guest path should be relative: {g}");
                }
            }
        }
    }

    #[test]
    fn elf_dynamic_needed_extracts_sonames() {
        let sh = Path::new("/bin/sh");
        if !sh.exists() || !is_elf(sh) {
            eprintln!("skipping: /bin/sh not ELF");
            return;
        }
        let data = std::fs::read(sh).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let needed: Vec<&str> = elf.libraries.clone();
        assert!(
            needed.iter().any(|n| n.contains("libc")),
            "/bin/sh should need libc: {:?}",
            needed
        );
    }

    #[test]
    fn resolve_soname_finds_libc() {
        let result = resolve_soname("libc.so.6", &[]);
        assert!(
            result.is_some(),
            "should resolve libc.so.6 via default paths"
        );
        assert!(result.unwrap().is_file());
    }

    #[test]
    fn suffix_with_sched_args() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let sched_args = vec!["--enable-borrow".into(), "--llc".into()];
        let suffix = build_suffix(base.len(), &[], &sched_args).unwrap();
        let s = String::from_utf8_lossy(&suffix);
        assert!(
            s.contains("sched_args"),
            "suffix should contain sched_args entry"
        );
        assert!(s.contains("TRAILER!!!"));
        assert_eq!((base.len() + suffix.len()) % 512, 0);
    }

    #[test]
    fn suffix_without_sched_args_omits_entry() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let suffix = build_suffix(base.len(), &[], &[]).unwrap();
        let s = String::from_utf8_lossy(&suffix);
        assert!(
            !s.contains("sched_args"),
            "empty sched_args should not produce entry"
        );
    }

    #[test]
    fn shm_segment_name_format() {
        let name = shm_segment_name(0xDEADBEEF);
        assert!(name.starts_with("/ktstr-base-"));
        assert!(name.contains("deadbeef"));
    }

    #[test]
    fn is_deleted_self_returns_false_for_nonexistent() {
        assert!(!is_deleted_self(Path::new("/nonexistent/binary")));
    }

    #[test]
    fn is_deleted_self_returns_false_for_current() {
        let exe = crate::resolve_current_exe().unwrap();
        // Current binary is not deleted.
        assert!(!is_deleted_self(&exe));
    }

    #[test]
    fn shm_store_load_unlink_roundtrip() {
        let hash = 0xABCD_EF01_2345_6789u64;
        let data = vec![0x42u8; 1024];
        shm_store_base(hash, &data).unwrap();
        let loaded = shm_load_base(hash);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().as_ref(), &data[..]);
        shm_unlink_base(hash);
        // After unlink, load should return None.
        assert!(shm_load_base(hash).is_none());
    }

    #[test]
    fn shm_load_nonexistent_returns_none() {
        let hash = 0xFFFF_FFFF_FFFF_FFFFu64;
        shm_unlink_base(hash); // ensure clean
        assert!(shm_load_base(hash).is_none());
    }

    #[test]
    fn shm_store_last_writer_wins_even_with_size_change() {
        // Documents actual semantics: shm_store reuses the segment name,
        // so a second write with different size overwrites the first.
        // Idempotent writes (same content_hash → same contents) rely on
        // callers to derive the hash from the actual content — this test
        // deliberately uses differently-sized payloads to prove the
        // writer does NOT assume the old name's size is still valid.
        let hash = 0x1234_5678_9ABC_DEF0u64;
        let d1 = vec![0x11u8; 64];
        let d2 = vec![0x22u8; 128];
        shm_store_base(hash, &d1).unwrap();
        shm_store_base(hash, &d2).unwrap();
        let loaded = shm_load_base(hash);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().as_ref(), &d2[..]);
        shm_unlink_base(hash);
    }

    #[test]
    fn shm_segment_name_unique_per_hash() {
        let n1 = shm_segment_name(0);
        let n2 = shm_segment_name(1);
        assert_ne!(n1, n2);
        assert!(n1.starts_with("/ktstr-base-"));
        assert!(n2.starts_with("/ktstr-base-"));
    }

    #[test]
    fn shm_unlink_nonexistent_is_noop() {
        // Should not panic.
        shm_unlink_base(0xDEAD_DEAD_DEAD_DEADu64);
    }

    #[test]
    fn mapped_shm_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MappedShm>();
    }

    #[test]
    fn shm_load_base_holds_lock_until_drop() {
        // Invariant: as long as a MappedShm is live, the SHM
        // segment's flock is held in LOCK_SH. A concurrent writer
        // calling LOCK_EX | LOCK_NB must fail with EWOULDBLOCK. Once
        // the MappedShm is dropped, the lock releases and a subsequent
        // LOCK_EX | LOCK_NB must succeed.
        //
        // This is the core invariant of the #2 fix — if it regresses,
        // shm_store's ftruncate can race with a live reader and cause
        // SIGBUS on the mapped pages.
        let hash = 0xD0D0_BEEF_F00D_BA5Eu64;
        shm_unlink_base(hash); // clean any stale segment
        shm_store_base(hash, &vec![0x55u8; 256]).unwrap();
        let loaded = shm_load_base(hash).expect("load must succeed");

        // Open a second fd and attempt LOCK_EX|LOCK_NB. Should fail
        // with EWOULDBLOCK because the MappedShm holds LOCK_SH.
        let name = std::ffi::CString::new(shm_segment_name(hash)).unwrap();
        unsafe {
            let fd2 = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
            assert!(fd2 >= 0, "second shm_open must succeed");
            let rc = libc::flock(fd2, libc::LOCK_EX | libc::LOCK_NB);
            let errno = *libc::__errno_location();
            assert_eq!(
                rc, -1,
                "LOCK_EX|LOCK_NB must be blocked by the live reader's LOCK_SH"
            );
            assert_eq!(
                errno,
                libc::EWOULDBLOCK,
                "lock contention must surface as EWOULDBLOCK"
            );
            libc::close(fd2);
        }

        // Drop the mapping; lock releases.
        drop(loaded);

        // Now LOCK_EX|LOCK_NB must succeed on a fresh fd.
        unsafe {
            let fd3 = libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0);
            assert!(fd3 >= 0, "third shm_open must succeed");
            let rc = libc::flock(fd3, libc::LOCK_EX | libc::LOCK_NB);
            assert_eq!(
                rc, 0,
                "LOCK_EX|LOCK_NB must succeed after the MappedShm is dropped"
            );
            libc::flock(fd3, libc::LOCK_UN);
            libc::close(fd3);
        }
        shm_unlink_base(hash);
    }

    #[test]
    fn strip_debug_current_exe() {
        let exe = crate::resolve_current_exe().unwrap();
        let data = strip_debug(&exe).unwrap();
        assert!(!data.is_empty());
        // Stripped binary should be an ELF (first 4 bytes = 0x7f ELF).
        assert_eq!(&data[..4], b"\x7fELF");
    }

    #[test]
    fn strip_debug_nonexistent_fails() {
        let result = strip_debug(Path::new("/nonexistent/binary"));
        assert!(result.is_err());
    }

    #[test]
    fn create_initramfs_base_contains_init() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("init"), "base should contain init entry");
    }

    #[test]
    fn create_initramfs_base_includes_extra_shared_libs() {
        let exe = crate::resolve_current_exe().unwrap();
        let sched = crate::test_support::require_binary("scx-ktstr");
        let extras: Vec<(&str, &Path)> = vec![("scheduler", sched.as_path())];
        let base = create_initramfs_base(&exe, &extras, &[], false).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(
            s.contains("lib64/libelf"),
            "initramfs with scx-ktstr extra should contain libelf; \
             resolved libs: {:?}",
            resolve_shared_libs(sched.as_path()).unwrap().found
        );
    }

    #[test]
    fn load_initramfs_to_memory() {
        let data = vec![0xAA; 4096];
        let mem = vm_memory::GuestMemoryMmap::<()>::from_ranges(&[(
            vm_memory::GuestAddress(0),
            16 << 20,
        )])
        .unwrap();
        let (addr, size) = load_initramfs(&mem, &data, 0x200000).unwrap();
        assert_eq!(addr, 0x200000);
        assert_eq!(size, 4096);
        let mut buf = vec![0u8; 4096];
        use vm_memory::{Bytes, GuestAddress};
        mem.read_slice(&mut buf, GuestAddress(0x200000)).unwrap();
        assert_eq!(buf, data);
    }

    // -- include_files and busybox tests --

    #[test]
    fn busybox_with_include_files() {
        let exe = crate::resolve_current_exe().unwrap();
        // Create a temp file to include.
        let tmp = std::env::temp_dir().join("ktstr-test-include-busybox");
        std::fs::write(&tmp, b"hello").unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/test.txt", tmp.as_path())];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let names = cpio_entry_names(&base);
        assert!(
            names.iter().any(|n| n == "bin/busybox"),
            "busybox=true should have bin/busybox entry: {:?}",
            names
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_no_busybox_when_empty() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let names = cpio_entry_names(&base);
        assert!(
            !names.iter().any(|n| n == "bin/busybox"),
            "busybox=false should not have bin/busybox entry: {:?}",
            names
        );
    }

    #[test]
    fn include_files_preserves_mode() {
        let tmp = std::env::temp_dir().join("ktstr-test-include-mode");
        std::fs::write(&tmp, b"script content").unwrap();
        // Set executable mode.
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o100755)).unwrap();

        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/run.sh", tmp.as_path())];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(
            s.contains("include-files/run.sh"),
            "include path should appear in cpio"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_elf_gets_shared_libs() {
        // /bin/sh is a dynamic ELF on most systems.
        let sh = Path::new("/bin/sh");
        if !sh.exists() {
            eprintln!("skipping: /bin/sh not found");
            return;
        }
        if !is_elf(sh) {
            eprintln!("skipping: /bin/sh is not ELF");
            return;
        }
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/sh", sh)];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        // Dynamic ELF should pull in libc shared libs.
        let shared = resolve_shared_libs(sh).unwrap();
        if !shared.found.is_empty() {
            assert!(
                shared.found.iter().any(|(g, _)| s.contains(g.as_str())),
                "include ELF shared libs should appear in archive: {:?}",
                shared.found
            );
        }
    }

    #[test]
    fn include_files_non_elf_no_shared_libs() {
        let tmp = std::env::temp_dir().join("ktstr-test-include-nonelf");
        std::fs::write(&tmp, b"#!/bin/sh\necho hello\n").unwrap();
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/hello.sh", tmp.as_path())];
        // Should not fail (ELF parsing skipped for non-ELF).
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("include-files/hello.sh"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_adds_directory_entries() {
        let tmp = std::env::temp_dir().join("ktstr-test-include-dirs");
        std::fs::write(&tmp, b"data").unwrap();
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> =
            vec![("include-files/subdir/nested/file.txt", tmp.as_path())];
        let base = create_initramfs_base(&exe, &[], &includes, true).unwrap();
        let s = String::from_utf8_lossy(&base);
        assert!(s.contains("include-files"), "should have include-files dir");
        assert!(
            s.contains("include-files/subdir"),
            "should have subdir entry"
        );
        assert!(
            s.contains("include-files/subdir/nested"),
            "should have nested subdir entry"
        );
        assert!(s.contains("bin"), "should have bin dir for busybox");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn is_elf_detects_elf_binary() {
        let exe = crate::resolve_current_exe().unwrap();
        assert!(is_elf(&exe), "test binary should be ELF");
    }

    #[test]
    fn is_elf_rejects_non_elf() {
        let tmp = std::env::temp_dir().join("ktstr-test-not-elf");
        std::fs::write(&tmp, b"not an elf file").unwrap();
        assert!(!is_elf(&tmp));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn is_elf_rejects_short_file() {
        let tmp = std::env::temp_dir().join("ktstr-test-short-elf");
        std::fs::write(&tmp, b"ab").unwrap();
        assert!(!is_elf(&tmp));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn is_elf_nonexistent_returns_false() {
        assert!(!is_elf(Path::new("/nonexistent/file")));
    }

    #[test]
    fn include_files_rejects_path_traversal() {
        let tmp = std::env::temp_dir().join("ktstr-test-traversal");
        std::fs::write(&tmp, b"data").unwrap();
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/../etc/passwd", tmp.as_path())];
        let result = create_initramfs_base(&exe, &[], &includes, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains(".."),
            "error should mention path traversal: {err}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn include_files_rejects_fifo() {
        let fifo_path = std::env::temp_dir().join("ktstr-test-fifo");
        let _ = std::fs::remove_file(&fifo_path);
        // Create a FIFO.
        let c_path = std::ffi::CString::new(fifo_path.to_str().unwrap()).unwrap();
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) };
        assert_eq!(
            rc,
            0,
            "ktstr: mkfifo({}) failed -- test infrastructure broken",
            fifo_path.display(),
        );
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/pipe", fifo_path.as_path())];
        let result = create_initramfs_base(&exe, &[], &includes, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a regular file"),
            "error should reject FIFO: {err}"
        );
        let _ = std::fs::remove_file(&fifo_path);
    }

    #[test]
    fn include_files_rejects_directory() {
        let dir_path = std::env::temp_dir().join("ktstr-test-include-dir");
        let _ = std::fs::create_dir(&dir_path);
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/mydir", dir_path.as_path())];
        let result = create_initramfs_base(&exe, &[], &includes, true);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("not a regular file"),
            "error should reject directory: {err}"
        );
        let _ = std::fs::remove_dir(&dir_path);
    }

    #[test]
    fn busybox_independent_of_include_files() {
        let exe = crate::resolve_current_exe().unwrap();
        // busybox=true but no include_files.
        let base = create_initramfs_base(&exe, &[], &[], true).unwrap();
        let names = cpio_entry_names(&base);
        assert!(
            names.iter().any(|n| n == "bin/busybox"),
            "busybox=true should have bin/busybox entry even without includes: {:?}",
            names
        );
    }

    // -- ld.so.conf parsing tests --

    #[test]
    fn parse_ld_so_conf_empty_file() {
        let tmp = std::env::temp_dir().join("ktstr-test-ldso-empty");
        std::fs::write(&tmp, "").unwrap();
        let paths = parse_ld_so_conf(&tmp);
        assert!(paths.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn parse_ld_so_conf_comments_and_blank_lines() {
        let tmp = std::env::temp_dir().join("ktstr-test-ldso-comments");
        std::fs::write(&tmp, "# comment\n\n# another\n").unwrap();
        let paths = parse_ld_so_conf(&tmp);
        assert!(paths.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn parse_ld_so_conf_directory_entries() {
        let tmp = std::env::temp_dir().join("ktstr-test-ldso-dirs");
        // /usr/lib exists on all Linux systems.
        std::fs::write(&tmp, "/usr/lib\n/nonexistent-dir-xyz\n").unwrap();
        let paths = parse_ld_so_conf(&tmp);
        assert!(
            paths.contains(&PathBuf::from("/usr/lib")),
            "should include existing directory: {:?}",
            paths
        );
        assert!(
            !paths.contains(&PathBuf::from("/nonexistent-dir-xyz")),
            "should skip nonexistent directories: {:?}",
            paths
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn parse_ld_so_conf_include_directive() {
        let tmp_dir = std::env::temp_dir().join("ktstr-test-ldso-include");
        let conf_d = tmp_dir.join("conf.d");
        let _ = std::fs::create_dir_all(&conf_d);
        // Create a sub-config that points to /usr/lib.
        std::fs::write(conf_d.join("test.conf"), "/usr/lib\n").unwrap();
        // Create main config with include.
        let main_conf = tmp_dir.join("ld.so.conf");
        std::fs::write(
            &main_conf,
            format!("include {}/conf.d/*.conf\n", tmp_dir.display()),
        )
        .unwrap();
        let paths = parse_ld_so_conf(&main_conf);
        assert!(
            paths.contains(&PathBuf::from("/usr/lib")),
            "include directive should pull in /usr/lib: {:?}",
            paths
        );
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn parse_ld_so_conf_nonexistent_returns_empty() {
        let paths = parse_ld_so_conf(Path::new("/nonexistent/ld.so.conf"));
        assert!(paths.is_empty());
    }

    #[test]
    fn parse_ld_so_conf_self_include_does_not_stack_overflow() {
        // Regression: a config that includes itself previously recursed
        // without bound and overflowed the stack. The visited-set (keyed on
        // canonicalized path) must break the cycle on the second visit.
        let tmp = tempfile::TempDir::new().unwrap();
        let self_conf = tmp.path().join("self.conf");
        std::fs::write(
            &self_conf,
            format!("include {}/self.conf\n/usr/lib\n", tmp.path().display()),
        )
        .unwrap();
        let paths = parse_ld_so_conf(&self_conf);
        assert!(
            paths.contains(&PathBuf::from("/usr/lib")),
            "self-include must still parse the rest of the file: {:?}",
            paths
        );
    }

    #[test]
    fn parse_ld_so_conf_mutual_include_does_not_stack_overflow() {
        // Regression: a ↔ b include cycle previously recursed without bound.
        let tmp = tempfile::TempDir::new().unwrap();
        let a = tmp.path().join("a.conf");
        let b = tmp.path().join("b.conf");
        std::fs::write(
            &a,
            format!("include {}/b.conf\n/usr/lib\n", tmp.path().display()),
        )
        .unwrap();
        std::fs::write(&b, format!("include {}/a.conf\n", tmp.path().display())).unwrap();
        let paths = parse_ld_so_conf(&a);
        assert!(
            paths.contains(&PathBuf::from("/usr/lib")),
            "mutual include must still collect non-cycle directories: {:?}",
            paths
        );
    }

    #[test]
    fn parse_ld_so_conf_long_chain_terminates() {
        // Regression: a deep, acyclic include chain must terminate at
        // LD_SO_CONF_MAX_DEPTH without stack exhaustion. The terminal file
        // (only reachable past the depth limit) writes `/usr/lib`; its
        // absence from the result proves the depth limit actually stopped
        // descent rather than merely avoiding a crash.
        let tmp = tempfile::TempDir::new().unwrap();
        let chain_len = LD_SO_CONF_MAX_DEPTH + 4;
        for i in 0..chain_len {
            let this = tmp.path().join(format!("link_{i}.conf"));
            let next = tmp.path().join(format!("link_{}.conf", i + 1));
            if i + 1 < chain_len {
                std::fs::write(&this, format!("include {}\n", next.display())).unwrap();
            } else {
                std::fs::write(&this, "/usr/lib\n").unwrap();
            }
        }
        let root = tmp.path().join("link_0.conf");
        let paths = parse_ld_so_conf(&root);
        assert!(
            !paths.contains(&PathBuf::from("/usr/lib")),
            "depth limit must prevent reading the terminal file at depth {} > {}: {:?}",
            chain_len - 1,
            LD_SO_CONF_MAX_DEPTH,
            paths,
        );
    }

    // -- glob_match tests --

    #[test]
    fn glob_match_suffix_star() {
        assert!(glob_match("*.conf", "test.conf"));
        assert!(glob_match("*.conf", ".conf"));
        assert!(!glob_match("*.conf", "test.txt"));
    }

    #[test]
    fn glob_match_prefix_star() {
        assert!(glob_match("test*", "test.conf"));
        assert!(glob_match("test*", "test"));
        assert!(!glob_match("test*", "other"));
    }

    #[test]
    fn glob_match_middle_star() {
        assert!(glob_match("lib*.so", "libfoo.so"));
        assert!(!glob_match("lib*.so", "libfoo.a"));
    }

    #[test]
    fn glob_match_no_star() {
        assert!(glob_match("exact", "exact"));
        assert!(!glob_match("exact", "other"));
    }

    // -- ld.so.cache parsing tests --

    #[test]
    fn parse_ld_so_cache_finds_libc() {
        let cache = parse_ld_so_cache(Path::new("/etc/ld.so.cache"));
        // libc.so.6 is in every glibc system's ld.so.cache.
        assert!(
            cache.contains_key("libc.so.6"),
            "ld.so.cache should contain libc.so.6: found {} entries",
            cache.len(),
        );
        let path = &cache["libc.so.6"];
        assert!(
            path.is_file(),
            "cached libc path should exist: {}",
            path.display()
        );
    }

    #[test]
    fn parse_ld_so_cache_nonexistent_returns_empty() {
        let cache = parse_ld_so_cache(Path::new("/nonexistent/ld.so.cache"));
        assert!(cache.is_empty());
    }

    #[test]
    fn parse_ld_so_cache_bad_magic_returns_empty() {
        let tmp = std::env::temp_dir().join("ktstr-test-ldcache-bad");
        std::fs::write(&tmp, b"not a valid cache file").unwrap();
        let cache = parse_ld_so_cache(&tmp);
        assert!(cache.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn parse_ld_so_cache_truncated_returns_empty() {
        let tmp = std::env::temp_dir().join("ktstr-test-ldcache-trunc");
        // Valid magic but truncated header.
        let mut data = LD_CACHE_MAGIC.to_vec();
        data.extend_from_slice(&[0u8; 10]); // not enough for full header
        std::fs::write(&tmp, &data).unwrap();
        let cache = parse_ld_so_cache(&tmp);
        assert!(cache.is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn ld_so_cache_consistent_with_resolve_soname() {
        // If libc.so.6 is in the cache, resolve_soname should find it.
        let result = resolve_soname("libc.so.6", &[]);
        assert!(
            result.is_some(),
            "resolve_soname should find libc.so.6 (cache or paths)"
        );
        assert!(result.unwrap().is_file());
    }

    #[test]
    fn no_duplicate_cpio_entries() {
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let entries = cpio_entries(&base);
        let mut seen = std::collections::HashSet::new();
        let mut duplicates = Vec::new();
        for (name, size, mode, ino) in &entries {
            if !seen.insert(name.clone()) {
                duplicates.push((name.clone(), *size, *mode, *ino));
            }
        }
        assert!(
            duplicates.is_empty(),
            "archive contains duplicate entries: {:?}",
            duplicates
        );
    }

    #[test]
    fn no_duplicate_entries_with_include_files() {
        let exe = crate::resolve_current_exe().unwrap();
        // Create include files in a deeply nested path mimicking custom
        // linker library directories.
        let tmp_dir = std::env::temp_dir().join("ktstr-test-cpio-dedup");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let lib_data = vec![0xCCu8; 4096];
        let f1 = tmp_dir.join("libcustom1.so");
        let f2 = tmp_dir.join("libcustom2.so");
        let f3 = tmp_dir.join("libcustom3.so");
        std::fs::write(&f1, &lib_data).unwrap();
        std::fs::write(&f2, &lib_data).unwrap();
        std::fs::write(&f3, &lib_data).unwrap();

        let includes: Vec<(&str, &Path)> = vec![
            ("usr/local/custom/platform/lib/libcustom1.so", f1.as_path()),
            ("usr/local/custom/platform/lib/libcustom2.so", f2.as_path()),
            ("usr/local/custom/platform/lib/libcustom3.so", f3.as_path()),
        ];

        let base = create_initramfs_base(&exe, &[], &includes, false).unwrap();
        let entries = cpio_entries(&base);
        let entry_names: Vec<&str> = entries.iter().map(|(n, _, _, _)| n.as_str()).collect();

        // Verify all include files are present.
        for (archive_path, _) in &includes {
            assert!(
                entry_names.contains(archive_path),
                "missing include file entry '{}'; archive entries: {:?}",
                archive_path,
                entry_names
            );
        }

        // Verify all include files have correct size.
        for (archive_path, _) in &includes {
            let entry = entries.iter().find(|(n, _, _, _)| n == archive_path);
            assert!(
                entry.is_some_and(|(_, size, _, _)| *size == lib_data.len() as u32),
                "include file '{}' has wrong size: {:?}",
                archive_path,
                entry
            );
        }

        // Verify directory entries exist for the nested path.
        assert!(entry_names.contains(&"usr"), "missing 'usr' dir entry");
        assert!(
            entry_names.contains(&"usr/local"),
            "missing 'usr/local' dir entry"
        );
        assert!(
            entry_names.contains(&"usr/local/custom"),
            "missing 'usr/local/custom' dir entry"
        );
        assert!(
            entry_names.contains(&"usr/local/custom/platform"),
            "missing 'usr/local/custom/platform' dir entry"
        );
        assert!(
            entry_names.contains(&"usr/local/custom/platform/lib"),
            "missing 'usr/local/custom/platform/lib' dir entry"
        );

        // Verify directories come before files they contain.
        let dir_pos = entries
            .iter()
            .position(|(n, _, _, _)| n == "usr/local/custom/platform/lib")
            .unwrap();
        for (archive_path, _) in &includes {
            let file_pos = entries
                .iter()
                .position(|(n, _, _, _)| n == *archive_path)
                .unwrap();
            assert!(
                dir_pos < file_pos,
                "directory entry must precede file '{}': dir at {}, file at {}",
                archive_path,
                dir_pos,
                file_pos
            );
        }

        // No duplicate entries.
        let mut seen = std::collections::HashSet::new();
        let mut duplicates = Vec::new();
        for (name, _, _, _) in &entries {
            if !seen.insert(name.clone()) {
                duplicates.push(name.clone());
            }
        }
        assert!(
            duplicates.is_empty(),
            "duplicate entries in archive: {:?}",
            duplicates
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn include_elf_shared_libs_all_present_in_archive() {
        // Use /bin/sh as an include file — its shared libs must all
        // appear in the archive with non-zero sizes.
        let sh = Path::new("/bin/sh");
        if !sh.exists() || !is_elf(sh) {
            eprintln!("skipping: /bin/sh not available or not ELF");
            return;
        }
        let exe = crate::resolve_current_exe().unwrap();
        let includes: Vec<(&str, &Path)> = vec![("include-files/sh", sh)];
        let base = create_initramfs_base(&exe, &[], &includes, false).unwrap();
        let entries = cpio_entries(&base);
        let entry_map: std::collections::HashMap<&str, (u32, u32, u32)> = entries
            .iter()
            .map(|(n, s, m, i)| (n.as_str(), (*s, *m, *i)))
            .collect();

        let shared = resolve_shared_libs(sh).unwrap();
        for (guest_path, _host_path) in &shared.found {
            assert!(
                entry_map.contains_key(guest_path.as_str()),
                "shared lib '{}' missing from archive; entries: {:?}",
                guest_path,
                entries
                    .iter()
                    .map(|(n, _, _, _)| n.as_str())
                    .collect::<Vec<_>>()
            );
            let (size, _, _) = entry_map[guest_path.as_str()];
            assert!(
                size > 0,
                "shared lib '{}' has zero size in archive",
                guest_path
            );
        }

        // Verify the include file itself is present.
        assert!(
            entry_map.contains_key("include-files/sh"),
            "include file itself missing from archive"
        );
    }

    #[test]
    fn all_inode_zero_entries_have_nlink_one() {
        // Verify that all entries use ino=0 and nlink=1, so the kernel
        // initramfs unpacker never enters the hardlink path.
        let exe = crate::resolve_current_exe().unwrap();
        let base = create_initramfs_base(&exe, &[], &[], false).unwrap();
        let mut remaining: &[u8] = base.as_slice();
        while let Ok(reader) = cpio::newc::Reader::new(remaining) {
            if reader.entry().is_trailer() {
                break;
            }
            let name = reader.entry().name().to_string();
            let ino = reader.entry().ino();
            let nlink = reader.entry().nlink();
            assert_eq!(
                ino, 0,
                "entry '{}' has non-zero inode {}: risk of kernel hardlink confusion",
                name, ino
            );
            assert_eq!(
                nlink, 1,
                "entry '{}' has nlink {}: kernel only hardlinks when nlink >= 2",
                name, nlink
            );
            remaining = reader.finish().unwrap();
        }
    }

    #[test]
    fn lz4_legacy_compress_format() {
        let data = vec![0xAAu8; 4096];
        let compressed = lz4_legacy_compress(&data);
        // Must start with LZ4 legacy magic.
        assert_eq!(
            &compressed[..4],
            &LZ4_LEGACY_MAGIC,
            "output must start with LZ4 legacy magic 0x184C2102"
        );
        // First chunk: 4-byte compressed size follows magic.
        let chunk_size = u32::from_le_bytes(compressed[4..8].try_into().unwrap()) as usize;
        assert!(
            chunk_size > 0 && chunk_size < data.len(),
            "compressed chunk should be non-empty and smaller than input: {}",
            chunk_size
        );
        // Decompress and verify roundtrip.
        let decompressed = lz4_flex::block::decompress(&compressed[8..8 + chunk_size], data.len())
            .expect("lz4 block decompress failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn lz4_legacy_compress_large_input_splits_chunks() {
        // Input larger than LZ4_CHUNK_SIZE (8MB) must produce multiple chunks.
        let data = vec![0xBBu8; LZ4_CHUNK_SIZE + 1024];
        let compressed = lz4_legacy_compress(&data);
        assert_eq!(&compressed[..4], &LZ4_LEGACY_MAGIC);
        // Parse chunks: should be at least 2.
        let mut pos = 4;
        let mut chunk_count = 0;
        let mut total_decompressed = Vec::new();
        while pos + 4 <= compressed.len() {
            let chunk_size =
                u32::from_le_bytes(compressed[pos..pos + 4].try_into().unwrap()) as usize;
            if chunk_size == 0 {
                break;
            }
            pos += 4;
            let remaining_uncompressed = data.len() - total_decompressed.len();
            let expected_chunk_len = remaining_uncompressed.min(LZ4_CHUNK_SIZE);
            let decompressed =
                lz4_flex::block::decompress(&compressed[pos..pos + chunk_size], expected_chunk_len)
                    .expect("lz4 block decompress failed");
            total_decompressed.extend_from_slice(&decompressed);
            pos += chunk_size;
            chunk_count += 1;
        }
        assert!(
            chunk_count >= 2,
            "input > 8MB should produce >= 2 chunks, got {}",
            chunk_count
        );
        assert_eq!(total_decompressed, data);
    }

    #[test]
    fn lz4_legacy_compress_empty_input() {
        let compressed = lz4_legacy_compress(&[]);
        // Empty input: just the magic, no chunks.
        assert_eq!(compressed, LZ4_LEGACY_MAGIC);
    }

    /// Build a synthetic cpio archive from generated data for LZ4 tests.
    /// Uses generic paths to avoid banned terms.
    fn build_synthetic_cpio(total_size: usize) -> Vec<u8> {
        let mut archive = Vec::new();
        // Directory entries.
        write_entry(&mut archive, "lib", &[], 0o40755).unwrap();
        write_entry(&mut archive, "data", &[], 0o40755).unwrap();

        // Fill with generated binary data to reach target size.
        // Use a simple PRNG for reproducible high-entropy content.
        let mut rng_state = 0x12345678u64;
        let entry_size = 256 * 1024; // 256KB per entry
        let mut entry_num = 0;
        while archive.len() + entry_size < total_size {
            let mut payload = vec![0u8; entry_size];
            for byte in &mut payload {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *byte = (rng_state >> 33) as u8;
            }
            let name = format!("lib/test_{entry_num:04}.so");
            write_entry(&mut archive, &name, &payload, 0o100755).unwrap();
            entry_num += 1;
        }

        // Pad remaining space with a data file.
        if archive.len() < total_size {
            let remaining = total_size - archive.len() - 200; // room for header
            let remaining = remaining.min(total_size);
            let mut payload = vec![0u8; remaining];
            for byte in &mut payload {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                *byte = (rng_state >> 33) as u8;
            }
            write_entry(&mut archive, "data/fill.bin", &payload, 0o100644).unwrap();
        }

        // Trailer and padding.
        cpio::newc::trailer(&mut archive as &mut dyn std::io::Write).unwrap();
        let pad = (512 - (archive.len() % 512)) % 512;
        archive.extend(std::iter::repeat_n(0u8, pad));
        archive
    }

    /// Simulate the kernel's unlz4() decompression loop (non-fill path).
    /// This mirrors lib/decompress_unlz4.c behavior:
    ///   1. Read and validate 4-byte magic (0x184C2102)
    ///   2. Loop: read 4-byte LE chunk size, decompress chunk, advance
    ///   3. Handle concatenated magic (re-encounter mid-stream)
    ///   4. Terminate on size < 4 or size == 0
    fn simulate_kernel_unlz4(input: &[u8]) -> Result<Vec<u8>, String> {
        const UNCOMP_CHUNK_SIZE: usize = 8 << 20; // LZ4_DEFAULT_UNCOMPRESSED_CHUNK_SIZE

        if input.len() < 4 {
            return Err("input too short for magic".into());
        }

        let mut inp = 0usize; // current position
        let mut size = input.len() as isize; // remaining bytes

        // Read and validate magic.
        let magic = u32::from_le_bytes(input[inp..inp + 4].try_into().unwrap());
        if magic != 0x184C2102 {
            return Err(format!("invalid header: 0x{magic:08X}"));
        }
        inp += 4;
        size -= 4;

        let mut output = Vec::new();

        loop {
            if size < 4 {
                // End of input — clean exit.
                break;
            }

            let chunksize = u32::from_le_bytes(input[inp..inp + 4].try_into().unwrap()) as usize;

            // Handle concatenated magic mid-stream.
            if chunksize == 0x184C2102 {
                inp += 4;
                size -= 4;
                continue;
            }

            // Zero chunk size — end of stream.
            if chunksize == 0 {
                break;
            }

            inp += 4;
            size -= 4;

            // Kernel: LZ4_decompress_safe(inp, outp, chunksize, dest_len)
            // dest_len = uncomp_chunksize (8MB max output)
            let chunk_data = &input[inp..inp + chunksize];
            let decompressed = lz4_flex::block::decompress(chunk_data, UNCOMP_CHUNK_SIZE)
                .map_err(|e| format!("LZ4_decompress_safe failed: {e}"))?;

            output.extend_from_slice(&decompressed);

            size -= chunksize as isize;
            if size == 0 {
                break;
            } else if size < 0 {
                return Err("data corrupted: size went negative".into());
            }
            inp += chunksize;
        }

        Ok(output)
    }

    /// Roundtrip test with synthetic cpio data through the kernel's
    /// unlz4() decompression logic. Uses generated test data with
    /// generic paths.
    #[test]
    fn lz4_legacy_kernel_unlz4_roundtrip() {
        // Single chunk (< 8MB).
        let small = build_synthetic_cpio(1 << 20); // ~1MB
        let compressed = lz4_legacy_compress(&small);
        let decompressed = simulate_kernel_unlz4(&compressed)
            .expect("kernel unlz4 simulation failed on small input");
        assert_eq!(decompressed, small);

        // Multi-chunk (> 8MB, forces chunk splitting).
        let large = build_synthetic_cpio(10 << 20); // ~10MB
        let compressed = lz4_legacy_compress(&large);
        let decompressed = simulate_kernel_unlz4(&compressed)
            .expect("kernel unlz4 simulation failed on multi-chunk input");
        assert_eq!(decompressed, large);
    }

    /// Test concatenated LZ4 legacy streams (base + suffix) through
    /// the kernel unlz4 simulation. This is the format used when
    /// base and suffix are compressed separately.
    #[test]
    fn lz4_legacy_kernel_unlz4_concatenated() {
        let base = build_synthetic_cpio(2 << 20); // ~2MB
        let suffix_data = b"arg1\narg2\narg3\n";

        let lz4_base = lz4_legacy_compress(&base);
        let lz4_suffix = lz4_legacy_compress(suffix_data);

        // Concatenate the two streams.
        let mut combined = Vec::with_capacity(lz4_base.len() + lz4_suffix.len());
        combined.extend_from_slice(&lz4_base);
        combined.extend_from_slice(&lz4_suffix);

        let decompressed = simulate_kernel_unlz4(&combined)
            .expect("kernel unlz4 simulation failed on concatenated streams");

        let mut expected = Vec::with_capacity(base.len() + suffix_data.len());
        expected.extend_from_slice(&base);
        expected.extend_from_slice(suffix_data);
        assert_eq!(decompressed, expected);
    }

    /// Verify lz4_flex block output is decompressible by the C lz4
    /// library (same decompressor as the kernel's LZ4_decompress_safe).
    /// Uses synthetic cpio data with generic paths.
    #[test]
    fn lz4_legacy_compress_c_compat() {
        let lz4_check = std::process::Command::new("lz4").arg("--version").output();
        if lz4_check.is_err() {
            eprintln!("skipping: lz4 CLI not found");
            return;
        }

        let data = build_synthetic_cpio(2 << 20); // ~2MB
        let compressed = lz4_legacy_compress(&data);
        let compressed_path = std::env::temp_dir().join("ktstr-test-lz4-compat.lz4");
        let decompressed_path = std::env::temp_dir().join("ktstr-test-lz4-compat.bin");
        std::fs::write(&compressed_path, &compressed).unwrap();

        let output = std::process::Command::new("lz4")
            .args(["-d", "-f", "--no-frame-crc"])
            .arg(&compressed_path)
            .arg(&decompressed_path)
            .output()
            .expect("lz4 -d failed to execute");

        let _ = std::fs::remove_file(&compressed_path);

        assert!(
            output.status.success(),
            "lz4 -d failed: stderr={}",
            String::from_utf8_lossy(&output.stderr),
        );

        let result = std::fs::read(&decompressed_path).unwrap();
        let _ = std::fs::remove_file(&decompressed_path);
        assert_eq!(result.len(), data.len(), "decompressed size mismatch");
        assert_eq!(&result[..], &data[..], "decompressed content mismatch");
    }

    /// Verify our output can be decompressed by `lz4 -d` when compressed
    /// with `lz4 -l` as reference. Tests cross-compatibility of our
    /// legacy format framing with the reference implementation.
    #[test]
    fn lz4_legacy_reference_cross_compat() {
        let lz4_check = std::process::Command::new("lz4").arg("--version").output();
        if lz4_check.is_err() {
            eprintln!("skipping: lz4 CLI not found");
            return;
        }

        let data = build_synthetic_cpio(2 << 20);

        // Compress with `lz4 -l` (reference legacy mode).
        let input_path = std::env::temp_dir().join("ktstr-test-lz4-ref-input.bin");
        let ref_path = std::env::temp_dir().join("ktstr-test-lz4-ref.lz4");
        std::fs::write(&input_path, &data).unwrap();

        let ref_output = std::process::Command::new("lz4")
            .args(["-l", "-f"])
            .arg(&input_path)
            .arg(&ref_path)
            .output()
            .expect("lz4 -l failed to execute");
        let _ = std::fs::remove_file(&input_path);

        assert!(
            ref_output.status.success(),
            "lz4 -l failed: stderr={}",
            String::from_utf8_lossy(&ref_output.stderr),
        );

        // Decompress reference output through our kernel simulation.
        let ref_compressed = std::fs::read(&ref_path).unwrap();
        let _ = std::fs::remove_file(&ref_path);

        let ref_decompressed = simulate_kernel_unlz4(&ref_compressed)
            .expect("kernel unlz4 simulation failed on lz4 -l output");
        assert_eq!(
            ref_decompressed, data,
            "reference lz4 -l roundtrip mismatch"
        );

        // Also compress with our encoder, decompress with lz4 -d.
        let our_compressed = lz4_legacy_compress(&data);
        let our_lz4_path = std::env::temp_dir().join("ktstr-test-lz4-ref-ours.lz4");
        let our_decompressed_path = std::env::temp_dir().join("ktstr-test-lz4-ref-ours.bin");
        std::fs::write(&our_lz4_path, &our_compressed).unwrap();

        let our_output = std::process::Command::new("lz4")
            .args(["-d", "-f", "--no-frame-crc"])
            .arg(&our_lz4_path)
            .arg(&our_decompressed_path)
            .output()
            .expect("lz4 -d on our output failed to execute");

        let _ = std::fs::remove_file(&our_lz4_path);

        assert!(
            our_output.status.success(),
            "lz4 -d on our output failed: stderr={}",
            String::from_utf8_lossy(&our_output.stderr),
        );

        let our_result = std::fs::read(&our_decompressed_path).unwrap();
        let _ = std::fs::remove_file(&our_decompressed_path);
        assert_eq!(our_result, data, "our lz4 output cross-compat mismatch");
    }
}
