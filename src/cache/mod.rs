//! Kernel image cache for ktstr.
//!
//! Manages a local cache of built kernel images under an XDG-compliant
//! directory. Each cached kernel is a directory containing the boot
//! image, optionally a stripped vmlinux ELF (symbol table, BTF, and
//! the section headers that monitor/probe code reads), and a
//! `metadata.json` descriptor. `CONFIG_HZ` is recovered from the
//! embedded IKCONFIG blob in the stripped vmlinux (ktstr.kconfig
//! forces `CONFIG_IKCONFIG=y`), so no separate `.config` sidecar is
//! cached.
//!
//! # Cache location
//!
//! Resolved in order:
//! 1. `KTSTR_CACHE_DIR` environment variable
//! 2. `$XDG_CACHE_HOME/ktstr/kernels/`
//! 3. `$HOME/.cache/ktstr/kernels/`
//!
//! # Submodule layout
//!
//! - [`metadata`] — public types: [`KernelSource`], [`KernelMetadata`],
//!   [`CacheArtifacts`], [`KconfigStatus`], [`CacheEntry`],
//!   [`ListedEntry`], plus the internal `classify_corrupt_reason`
//!   dispatcher.
//! - [`cache_dir`] — [`CacheDir`] handle, lock guards
//!   ([`SharedLockGuard`], [`ExclusiveLockGuard`]), store/lookup/list/
//!   clean lifecycle, and reader/writer-asymmetric lock policy.
//! - [`housekeeping`] — atomic-rename install primitives, cache-key
//!   and image-name validators, `read_metadata` decoder, and the
//!   `clean_orphaned_tmp_dirs` cross-PID sweep.
//! - [`vmlinux_strip`] — ELF strip pipeline ([`strip_vmlinux_debug`],
//!   `neutralize_relocs`, `strip_keep_list`, `strip_debug_prefix`)
//!   plus the keep-list / zero-data section-name unions.
//! - [`resolve`] — env-cascade root resolution
//!   (`resolve_cache_root_with_suffix`, `validate_home_for_cache`,
//!   [`path_inside_cache_root`]) and source-tree path helpers
//!   ([`prefer_source_tree_for_dwarf`], [`recover_local_source_tree`]).

use crate::flock::LOCK_DIR_NAME;

mod cache_dir;
mod housekeeping;
mod metadata;
mod resolve;
mod vmlinux_strip;

// Public API re-exports — preserve every `crate::cache::*` path that
// external callers (lib.rs, cli.rs, fetch.rs, monitor/*, probe/btf.rs,
// vmm/disk_template.rs, test_support/*, remote_cache.rs, stats.rs,
// flock.rs) rely on.

pub use cache_dir::{CacheDir, ExclusiveLockGuard, SharedLockGuard};
pub use metadata::{
    CacheArtifacts, CacheEntry, KconfigStatus, KernelMetadata, KernelSource, ListedEntry,
};
pub use resolve::{prefer_source_tree_for_dwarf, recover_local_source_tree};

// Re-export KernelId from kernel_path (canonical definition, std-only).
pub use crate::kernel_path::KernelId;

// Crate-internal API re-exports for callers in other modules
// (test_support/model, vmm/disk_template, monitor/btf_offsets).
pub(crate) use resolve::{path_inside_cache_root, resolve_cache_root_with_suffix};
// Re-exported for rustdoc cross-link resolution (cache/mod.rs:31's
// `[`strip_vmlinux_debug`]` link). No `crate::cache::strip_vmlinux_debug`
// code call sites today; intra-cache callers (cache_dir.rs, tests)
// reach the function via `super::vmlinux_strip::strip_vmlinux_debug`.
#[allow(unused_imports)]
pub(crate) use vmlinux_strip::strip_vmlinux_debug;

/// Filename prefix that marks an in-progress atomic-store directory
/// under the cache root. Format: `{TMP_DIR_PREFIX}{cache_key}-{pid}`.
/// Centralized here so the three roles — emitter
/// ([`cache_dir::CacheDir::store`]), scanner
/// ([`housekeeping::clean_orphaned_tmp_dirs`]), validator
/// ([`housekeeping::validate_cache_key`]) — cannot drift.
pub(crate) const TMP_DIR_PREFIX: &str = ".tmp-";

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    // Test-only re-imports for items that moved into submodules.
    // The split preserves test bodies verbatim; these `use` statements
    // route the references through the new module paths.
    use super::cache_dir::should_warn_unstripped;
    use super::housekeeping::{
        clean_orphaned_tmp_dirs, read_metadata, validate_cache_key, validate_filename,
    };
    use super::metadata::classify_corrupt_reason;
    use super::resolve::{resolve_cache_root, validate_home_for_cache};
    use super::vmlinux_strip::{
        SPECULATIVE_ZERO_DATA_SECTIONS, STRUCTURAL_KEEP_SECTIONS, neutralize_relocs,
        strip_debug_prefix, strip_vmlinux_debug,
    };
    use crate::test_support::test_helpers::{EnvVarGuard, lock_env};

    /// Decode an ELF section `sh_type` integer to its `SHT_*` constant
    /// name. Strip-helper assertions embed the decoded name alongside
    /// the raw integer so a failing diagnostic like "left: 8 right: 1"
    /// reads as "sh_type=8 (SHT_NOBITS)" / "sh_type=1 (SHT_PROGBITS)"
    /// — immediately actionable instead of requiring the reader to
    /// look up the ELF spec table.
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

    fn test_metadata(version: &str) -> KernelMetadata {
        KernelMetadata {
            version: Some(version.to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: Some("abc123".to_string()),
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ktstr_kconfig_hash: Some("def456".to_string()),
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        }
    }

    fn create_fake_image(dir: &Path) -> PathBuf {
        let image = dir.join("bzImage");
        fs::write(&image, b"fake kernel image").unwrap();
        image
    }

    /// Build a minimal ELF object with a single `.text` section (64
    /// bytes of 0xCC) anchored by one symbol. The anchor symbol is
    /// what drives `object::write` to emit `.symtab`/`.strtab`, and
    /// every `neutralize_relocs` test shares this base shape.
    /// Callers that need relocation sections (with or without
    /// `SHF_ALLOC`) add them on top of the returned object before
    /// calling `.write()`.
    ///
    /// `arch` selects the ELF class: `Architecture::X86_64` yields
    /// ELF64 (8-byte anchor symbol), `Architecture::I386` yields
    /// ELF32 (4-byte anchor symbol). The anchor-symbol size is the
    /// only shape difference between the two classes at this
    /// fixture level; everything downstream (section headers, the
    /// `is_reloc` predicate under test) is driven by the
    /// ELF32/ELF64 split `object::write` performs based on `arch`.
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

    /// Regression pin for the explicit `other =>` arm in
    /// [`build_base_elf_with_text_symbol`]. Before the guard, an
    /// unsupported architecture silently fell through to `sym_size = 8`
    /// which is wrong for any future 32-bit arch (or any arch whose
    /// address width isn't 8 bytes). `Aarch64` is a supported object
    /// crate architecture that isn't on the helper's allow-list, so
    /// passing it triggers the panic and the `#[should_panic]`
    /// assertion confirms the guard fires.
    #[test]
    #[should_panic(expected = "unsupported arch")]
    fn build_base_elf_with_text_symbol_panics_on_unsupported_arch() {
        let _ = build_base_elf_with_text_symbol(object::Architecture::Aarch64);
    }

    // -- keep-list source disjointness --

    /// Every entry in `is_keep_section` comes from one of four source
    /// lists owned by independent modules. Overlap is harmless at
    /// strip time but masks drift: if two modules add `.foo` and one
    /// later removes it, the strip still preserves `.foo` via the
    /// other module — the "dead" reference outlives its reader.
    ///
    /// This test locks the four lists as disjoint sets so a removal
    /// in one module immediately drops `.foo` from
    /// `is_keep_section`, and the downstream consumer's loss of its
    /// declared section becomes a visible test break rather than a
    /// silent ALL-tests-pass-but-data-is-missing runtime surprise.
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

    /// Same disjointness contract for the two zero-data lists.
    /// Retained sections here keep symbols but drop bytes — duplicate
    /// declarations would mask the same drift the keep-list test
    /// guards against.
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

    // -- KernelMetadata serde --

    #[test]
    fn cache_metadata_serde_roundtrip() {
        let meta = test_metadata("6.14.2");
        let json = serde_json::to_string_pretty(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version.as_deref(), Some("6.14.2"));
        assert_eq!(parsed.source, KernelSource::Tarball);
        assert_eq!(parsed.arch, "x86_64");
        assert_eq!(parsed.image_name, "bzImage");
        assert_eq!(parsed.config_hash.as_deref(), Some("abc123"));
        assert_eq!(parsed.built_at, "2026-04-12T10:00:00Z");
        assert_eq!(parsed.ktstr_kconfig_hash.as_deref(), Some("def456"));
        assert!(!parsed.has_vmlinux);
        assert!(!parsed.vmlinux_stripped);
    }

    #[test]
    fn cache_metadata_serde_git_with_payload() {
        let meta = KernelMetadata {
            version: Some("6.15-rc3".to_string()),
            source: KernelSource::Git {
                git_hash: Some("a1b2c3d".to_string()),
                git_ref: Some("v6.15-rc3".to_string()),
            },
            arch: "aarch64".to_string(),
            image_name: "Image".to_string(),
            config_hash: None,
            built_at: "2026-04-12T12:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed.source,
            KernelSource::Git {
                git_hash: Some(ref h),
                git_ref: Some(ref r),
            }
            if h == "a1b2c3d" && r == "v6.15-rc3"
        ));
    }

    #[test]
    fn cache_metadata_serde_local_with_source_tree() {
        let meta = KernelMetadata {
            version: Some("6.14.0".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(PathBuf::from("/tmp/linux")),
                git_hash: Some("deadbee".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: Some("fff000".to_string()),
            built_at: "2026-04-12T14:00:00Z".to_string(),
            ktstr_kconfig_hash: Some("aaa111".to_string()),
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: KernelMetadata = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed.source,
            KernelSource::Local {
                source_tree_path: Some(ref p),
                git_hash: Some(ref h),
            }
            if p == &PathBuf::from("/tmp/linux") && h == "deadbee"
        ));
        assert!(parsed.has_vmlinux);
        assert!(parsed.vmlinux_stripped);
    }

    /// git_hash on KernelSource::Local is a plain Option<String> with
    /// no serde attributes — the compat shims (serde(default) +
    /// skip_serializing_if) were removed for pre-1.0, so `None`
    /// serializes as an explicit `null` key and deserialization
    /// accepts `null` back as `None`. This test pins only the
    /// None → null → None round trip; the absent-key branch is
    /// exercised separately by
    /// [`kernel_source_absent_option_keys_deserialize_as_none`].
    #[test]
    fn kernel_source_local_git_hash_serde_round_trip_none() {
        let src = KernelSource::Local {
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
            git_hash: None,
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(
            json.contains(r#""git_hash":null"#),
            "git_hash=None must round-trip as explicit null, got {json}"
        );
        let parsed: KernelSource = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, KernelSource::Local { git_hash: None, .. }));
    }

    /// Pins the post-shim wire format: `Option` payload fields inside
    /// every [`KernelSource`] variant serialize as explicit `null`
    /// rather than being omitted. The `serde(default)` /
    /// `skip_serializing_if` compat shims were removed for pre-1.0;
    /// [`kernel_source_local_git_hash_serde_round_trip_none`] already
    /// covers the round-trip statement for the single Local.git_hash
    /// slot. This test extends that guarantee across every Option
    /// payload on both Git and Local so `cargo ktstr kernel list
    /// --json` consumers see stable key presence regardless of which
    /// optional values are set — absent keys would mean the emitted
    /// schema has silently regressed.
    #[test]
    fn kernel_source_option_fields_serialize_as_explicit_null() {
        let local = KernelSource::Local {
            source_tree_path: None,
            git_hash: None,
        };
        let local_json = serde_json::to_string(&local).unwrap();
        assert!(
            local_json.contains(r#""source_tree_path":null"#),
            "Local.source_tree_path=None must serialize as explicit null, got {local_json}"
        );
        assert!(
            local_json.contains(r#""git_hash":null"#),
            "Local.git_hash=None must serialize as explicit null, got {local_json}"
        );

        let git = KernelSource::Git {
            git_hash: None,
            git_ref: None,
        };
        let git_json = serde_json::to_string(&git).unwrap();
        assert!(
            git_json.contains(r#""git_hash":null"#),
            "Git.git_hash=None must serialize as explicit null, got {git_json}"
        );
        // The struct field is `git_ref` but `#[serde(rename = "ref")]`
        // renames the JSON key — check the renamed key, not the field.
        assert!(
            git_json.contains(r#""ref":null"#),
            "Git.git_ref=None must serialize as explicit null under the `ref` key, got {git_json}"
        );
    }

    /// Older `metadata.json` files written before `Option` fields
    /// were emitted as explicit `null` simply omit the keys. The
    /// [`KernelSource`] doc states absent `Option` keys must
    /// deserialize as `None` — cache-integrity enforcement rides on
    /// the required non-`Option` fields of [`KernelMetadata`], not
    /// on the optional payloads inside variants. Feed each variant
    /// a minimal JSON with every `Option` key omitted, deserialize,
    /// and assert the result carries `None` in every payload slot.
    #[test]
    fn kernel_source_absent_option_keys_deserialize_as_none() {
        // Git with both git_hash and ref omitted.
        let git_bare: KernelSource = serde_json::from_str(r#"{"type":"git"}"#)
            .expect("Git with absent Option keys must deserialize");
        assert!(matches!(
            git_bare,
            KernelSource::Git {
                git_hash: None,
                git_ref: None,
            }
        ));

        // Git with only git_hash present.
        let git_hash_only: KernelSource =
            serde_json::from_str(r#"{"type":"git","git_hash":"abc"}"#)
                .expect("Git with only git_hash must deserialize");
        assert!(matches!(
            git_hash_only,
            KernelSource::Git {
                git_hash: Some(ref h),
                git_ref: None,
            } if h == "abc"
        ));

        // Git with only ref present.
        let git_ref_only: KernelSource = serde_json::from_str(r#"{"type":"git","ref":"main"}"#)
            .expect("Git with only ref must deserialize");
        assert!(matches!(
            git_ref_only,
            KernelSource::Git {
                git_hash: None,
                git_ref: Some(ref r),
            } if r == "main"
        ));

        // Local with both source_tree_path and git_hash omitted.
        let local_bare: KernelSource = serde_json::from_str(r#"{"type":"local"}"#)
            .expect("Local with absent Option keys must deserialize");
        assert!(matches!(
            local_bare,
            KernelSource::Local {
                source_tree_path: None,
                git_hash: None,
            }
        ));

        // Local with only source_tree_path present.
        let local_path_only: KernelSource =
            serde_json::from_str(r#"{"type":"local","source_tree_path":"/tmp/linux"}"#)
                .expect("Local with only source_tree_path must deserialize");
        assert!(matches!(
            local_path_only,
            KernelSource::Local {
                source_tree_path: Some(ref p),
                git_hash: None,
            } if p.to_str() == Some("/tmp/linux")
        ));

        // Local with only git_hash present.
        let local_hash_only: KernelSource =
            serde_json::from_str(r#"{"type":"local","git_hash":"deadbeef"}"#)
                .expect("Local with only git_hash must deserialize");
        assert!(matches!(
            local_hash_only,
            KernelSource::Local {
                source_tree_path: None,
                git_hash: Some(ref h),
            } if h == "deadbeef"
        ));
    }

    #[test]
    fn kernel_source_serde_tagged_representation() {
        // Check the tagged JSON shape on each variant.
        let t = serde_json::to_string(&KernelSource::Tarball).unwrap();
        assert_eq!(t, r#"{"type":"tarball"}"#);
        let g = serde_json::to_string(&KernelSource::Git {
            git_hash: Some("abc".to_string()),
            git_ref: Some("main".to_string()),
        })
        .unwrap();
        assert!(g.contains(r#""type":"git""#));
        assert!(g.contains(r#""git_hash":"abc""#));
        assert!(g.contains(r#""ref":"main""#));
        let l = serde_json::to_string(&KernelSource::Local {
            source_tree_path: Some(PathBuf::from("/tmp/linux")),
            git_hash: Some("a1b2c3d".to_string()),
        })
        .unwrap();
        assert!(l.contains(r#""type":"local""#));
        assert!(l.contains(r#""source_tree_path":"/tmp/linux""#));
        assert!(l.contains(r#""git_hash":"a1b2c3d""#));
    }

    // -- CacheDir --

    #[test]
    fn cache_dir_with_root_does_not_create_dir() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("kernels");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root.clone());
        // Resolution must not create the directory — store() does it
        // lazily on first write.
        assert!(!root.exists());
        assert_eq!(cache.root(), root);
    }

    #[test]
    fn cache_dir_list_returns_empty_for_nonexistent_root() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("never-created");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root);
        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_store_creates_root_lazily() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("lazy-root");
        assert!(!root.exists());
        let cache = CacheDir::with_root(root.clone());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("key", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(root.exists(), "store() must create the cache root");
    }

    #[test]
    fn cache_dir_default_root_returns_path() {
        // `lock_env()` serializes against every other env-touching
        // test in the crate (test_support/model.rs, test_helpers
        // siblings, the cache_resolve_root_* tests below). nextest
        // runs unit tests concurrently within a binary and
        // `std::env::set_var` is process-wide, so a sibling test
        // that mutates HOME / XDG_CACHE_HOME / KTSTR_CACHE_DIR
        // without the lock can race the save / mutate / restore
        // window of an `EnvVarGuard` here. Tester finding T1.
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let resolved = CacheDir::default_root().unwrap();
        assert_eq!(resolved, tmp.path());
        // Side-effect-free: calling default_root() must not create
        // any directories beyond what the env var already pointed at.
    }

    #[test]
    fn cache_dir_list_empty() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_store_and_lookup() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));

        // Create a fake kernel image.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        // Store.
        let entry = cache
            .store("6.14.2-tarball-x86_64", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.key, "6.14.2-tarball-x86_64");
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("metadata.json").exists());

        // Lookup.
        let found = cache.lookup("6.14.2-tarball-x86_64");
        assert!(found.is_some());
        let found = found.unwrap();
        assert_eq!(found.key, "6.14.2-tarball-x86_64");
        assert_eq!(found.metadata.version.as_deref(), Some("6.14.2"));
    }

    #[test]
    fn cache_dir_lookup_missing() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup("nonexistent").is_none());
    }

    #[test]
    fn cache_dir_lookup_corrupt_metadata() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create entry dir with image but corrupt metadata.
        let entry_dir = tmp.path().join("bad-entry");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("bzImage"), b"fake").unwrap();
        fs::write(entry_dir.join("metadata.json"), b"not json").unwrap();

        // lookup returns None because metadata is corrupt and we
        // cannot determine the image_name field.
        let found = cache.lookup("bad-entry");
        assert!(found.is_none());
    }

    #[test]
    fn cache_dir_lookup_missing_image() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create entry dir with valid metadata but no image file.
        let entry_dir = tmp.path().join("no-image");
        fs::create_dir_all(&entry_dir).unwrap();
        let meta = test_metadata("6.14.2");
        let json = serde_json::to_string(&meta).unwrap();
        fs::write(entry_dir.join("metadata.json"), json).unwrap();

        let found = cache.lookup("no-image");
        assert!(found.is_none());
    }

    #[test]
    fn cache_dir_store_overwrites_existing() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta1 = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        cache
            .store(
                "6.14.2-tarball-x86_64",
                &CacheArtifacts::new(&image),
                &meta1,
            )
            .unwrap();

        let meta2 = KernelMetadata {
            built_at: "2026-04-12T11:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        cache
            .store(
                "6.14.2-tarball-x86_64",
                &CacheArtifacts::new(&image),
                &meta2,
            )
            .unwrap();

        let found = cache.lookup("6.14.2-tarball-x86_64").unwrap();
        assert_eq!(found.metadata.built_at, "2026-04-12T11:00:00Z");
    }

    #[test]
    fn cache_dir_list_sorted_newest_first() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta_old = KernelMetadata {
            built_at: "2026-04-10T10:00:00Z".to_string(),
            ..test_metadata("6.13.0")
        };
        let meta_new = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let meta_mid = KernelMetadata {
            built_at: "2026-04-11T10:00:00Z".to_string(),
            ..test_metadata("6.14.0")
        };

        // Store in non-chronological order.
        cache
            .store("old", &CacheArtifacts::new(&image), &meta_old)
            .unwrap();
        cache
            .store("new", &CacheArtifacts::new(&image), &meta_new)
            .unwrap();
        cache
            .store("mid", &CacheArtifacts::new(&image), &meta_mid)
            .unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].key(), "new");
        assert_eq!(entries[1].key(), "mid");
        assert_eq!(entries[2].key(), "old");
    }

    #[test]
    fn cache_dir_list_includes_corrupt_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create a valid entry.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("valid", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Create a corrupt entry (no metadata).
        let bad_dir = tmp.path().join("corrupt");
        fs::create_dir_all(&bad_dir).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 2);
        // Valid entry surfaces as ListedEntry::Valid.
        let valid = entries.iter().find(|e| e.key() == "valid").unwrap();
        assert!(valid.as_valid().is_some());
        // Corrupt entry surfaces as ListedEntry::Corrupt.
        let corrupt = entries.iter().find(|e| e.key() == "corrupt").unwrap();
        assert!(corrupt.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = corrupt else {
            panic!("expected Corrupt variant");
        };
        assert_eq!(
            reason, "metadata.json missing",
            "missing-metadata reason should be the exact missing-file label, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_missing_image_as_corrupt() {
        // Metadata parses cleanly but the image file it references
        // has been deleted (partial download / manual cleanup).
        // list() must surface the entry as ListedEntry::Corrupt with
        // an image-missing reason, so callers don't dispatch to
        // image_path() and get a stale path.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let entry = cache
            .store("missing-image", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Delete only the image file; leave metadata.json in place.
        fs::remove_file(entry.image_path()).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "missing-image");
        assert!(
            listed.as_valid().is_none(),
            "entry with missing image must not surface as Valid",
        );
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for missing-image entry");
        };
        assert!(
            reason.contains("image file") && reason.contains("missing"),
            "reason should cite missing image file, got: {reason}",
        );
        assert!(
            reason.contains(&meta.image_name),
            "reason should name the specific image file, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_unreadable_metadata_as_corrupt() {
        // The `missing` and parse-family branches (schema drift,
        // malformed, truncated) of `read_metadata` are covered
        // elsewhere; the I/O-error branch — any `fs::read_to_string`
        // failure that is NOT `ErrorKind::NotFound` — is exercised
        // here. Forcing a non-ENOENT error without relying on
        // filesystem permissions (which vary across rootless
        // containers and CI sandboxes) is awkward, so we make
        // `metadata.json` a DIRECTORY: `read_to_string` then fails
        // with `EISDIR`, which `read_metadata` must map to the
        // `"metadata.json unreadable: "` prefix rather than the
        // missing or any parse-family label. This pins the
        // distinction so a future refactor that collapses the arms
        // back into a single generic "corrupt" reason breaks this
        // test before it ships a less-actionable diagnostic.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("unreadable-metadata");
        fs::create_dir_all(entry_dir.join("metadata.json")).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "unreadable-metadata");
        assert!(listed.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for entry with unreadable metadata");
        };
        assert!(
            reason.starts_with("metadata.json unreadable: "),
            "unreadable-metadata reason should carry the unreadable prefix distinct from the \
             missing / schema-drift / malformed / truncated prefixes, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_malformed_json_as_corrupt() {
        // metadata.json exists but is not valid JSON at all (unbalanced
        // punctuation / stray characters). `read_metadata` must route
        // this through `serde_json::Error::classify() ==
        // Category::Syntax` to produce the
        // `"metadata.json malformed: {e}"` prefix — distinct from the
        // schema-drift prefix that fires when JSON parses but does
        // not match the `KernelMetadata` shape.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("malformed-json");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("metadata.json"), b"not valid json {[").unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "malformed-json");
        assert!(listed.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for malformed-json entry");
        };
        assert!(
            reason.starts_with("metadata.json malformed: "),
            "malformed-JSON reason should carry the malformed prefix \
             (Category::Syntax route), got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_incomplete_metadata_as_corrupt() {
        // metadata.json is valid JSON but omits fields the current
        // `KernelMetadata` schema requires: `source`, `arch`,
        // `image_name`, `built_at`, `has_vmlinux`, and
        // `vmlinux_stripped`. These are non-`Option`,
        // non-`#[serde(default)]` fields, so `serde_json::from_str`
        // fails with `Category::Data` when they are absent. Note
        // `has_vmlinux: bool` and `vmlinux_stripped: bool` are
        // required even though they are not wrapped in `Option` — a
        // plain `bool` with no `#[serde(default)]` attribute must
        // still be present in the JSON payload. serde_json reports
        // the first missing required field in declaration order
        // (`source`), and `read_metadata` wraps it under the
        // schema-drift prefix so the user sees both the
        // classification ("schema drift") and
        // the specific missing field.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("incomplete-metadata");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("metadata.json"), br#"{"version": "6.14"}"#).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "incomplete-metadata");
        assert!(
            listed.as_valid().is_none(),
            "incomplete-metadata missing required fields must not deserialize as Valid",
        );
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for entry with incomplete metadata");
        };
        assert!(
            reason.starts_with("metadata.json schema drift: "),
            "incomplete-metadata reason should carry the schema-drift \
             prefix (Category::Data route), got: {reason}",
        );
        assert!(
            reason.contains("missing field `source`"),
            "incomplete-metadata reason should name the first missing required field, got: {reason}",
        );
    }

    #[test]
    fn cache_dir_list_classifies_truncated_json_as_corrupt() {
        // metadata.json ends mid-value: `{"source":` stops after the
        // colon with no value byte. serde_json surfaces this as
        // `Category::Eof`, which `read_metadata` wraps under the
        // `"metadata.json truncated: {e}"` prefix. Covers the Eof
        // branch of the classify() match — distinct from the schema-
        // drift (Data) and malformed (Syntax) branches exercised by
        // the sibling tests above.
        //
        // Typical real-world cause: a crashed `store()` whose atomic
        // rename never completed, leaving a partially-written
        // metadata.json in a surviving entry directory.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let entry_dir = tmp.path().join("truncated-json");
        fs::create_dir_all(&entry_dir).unwrap();
        fs::write(entry_dir.join("metadata.json"), br#"{"source":"#).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 1);
        let listed = &entries[0];
        assert_eq!(listed.key(), "truncated-json");
        assert!(listed.as_valid().is_none());
        let ListedEntry::Corrupt { reason, .. } = listed else {
            panic!("expected Corrupt variant for truncated-json entry");
        };
        assert!(
            reason.starts_with("metadata.json truncated: "),
            "truncated-JSON reason should carry the truncated prefix \
             (Category::Eof route), got: {reason}",
        );
    }

    /// Table-drive every prefix → `error_kind` classifier mapping.
    /// Pins each documented value independently so a regression in
    /// one arm surfaces with the specific prefix cited in the
    /// failure message, not as a blanket "classifier broken". The
    /// "unknown" fallback row is the safety net: a future producer
    /// prefix that falls through this table must surface as
    /// `"unknown"` to consumers rather than panic.
    #[test]
    fn classify_corrupt_reason_covers_every_documented_prefix() {
        let cases: &[(&str, &str)] = &[
            ("metadata.json missing", "missing"),
            (
                "metadata.json unreadable: Is a directory (os error 21)",
                "unreadable",
            ),
            (
                "metadata.json schema drift: missing field `source` at line 1 column 21",
                "schema_drift",
            ),
            (
                "metadata.json malformed: expected value at line 1 column 1",
                "malformed",
            ),
            (
                "metadata.json truncated: EOF while parsing a value at line 1 column 10",
                "truncated",
            ),
            (
                "metadata.json parse error: something unexpected",
                "parse_error",
            ),
            (
                "image file bzImage missing from entry directory",
                "image_missing",
            ),
            ("some future prefix nobody wrote yet", "unknown"),
        ];
        for (reason, expected) in cases {
            assert_eq!(
                classify_corrupt_reason(reason),
                *expected,
                "reason `{reason}` should classify as `{expected}`",
            );
        }
    }

    /// `ListedEntry::error_kind()` returns `None` on a Valid entry
    /// and the classifier result on a Corrupt entry. Pins the
    /// Valid → None contract so a consumer that dispatches on
    /// `error_kind().is_some()` can safely gate on the corrupt
    /// path.
    #[test]
    fn listed_entry_error_kind_dispatches_on_variant() {
        // Construct a Valid entry via the normal store path.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        cache
            .store("valid-ek", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // And a Corrupt entry via a missing-metadata directory.
        let bad_dir = tmp.path().join("cache").join("corrupt-ek");
        fs::create_dir_all(&bad_dir).unwrap();

        let entries = cache.list().unwrap();
        assert_eq!(entries.len(), 2);
        let valid = entries
            .iter()
            .find(|e| e.key() == "valid-ek")
            .expect("valid entry must be listed");
        let corrupt = entries
            .iter()
            .find(|e| e.key() == "corrupt-ek")
            .expect("corrupt entry must be listed");
        assert_eq!(
            valid.error_kind(),
            None,
            "Valid entries must report no error_kind",
        );
        assert_eq!(
            corrupt.error_kind(),
            Some("missing"),
            "missing-metadata Corrupt entry must classify as `missing`",
        );
    }

    #[test]
    fn cache_dir_list_skips_tmp_dirs() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create a .tmp- directory (in-progress store).
        let tmp_dir = tmp.path().join(".tmp-in-progress-12345");
        fs::create_dir_all(&tmp_dir).unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_list_skips_regular_files() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());

        // Create a regular file in the cache root.
        fs::write(tmp.path().join("stray-file.txt"), b"stray").unwrap();

        let entries = cache.list().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn cache_dir_clean_all() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store("a", &CacheArtifacts::new(&image), &test_metadata("6.14.0"))
            .unwrap();
        cache
            .store("b", &CacheArtifacts::new(&image), &test_metadata("6.14.1"))
            .unwrap();
        cache
            .store("c", &CacheArtifacts::new(&image), &test_metadata("6.14.2"))
            .unwrap();

        let removed = cache.clean_all().unwrap();
        assert_eq!(removed, 3);
        assert!(cache.list().unwrap().is_empty());
    }

    #[test]
    fn cache_dir_clean_keep_n() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        let meta_old = KernelMetadata {
            built_at: "2026-04-10T10:00:00Z".to_string(),
            ..test_metadata("6.13.0")
        };
        let meta_new = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let meta_mid = KernelMetadata {
            built_at: "2026-04-11T10:00:00Z".to_string(),
            ..test_metadata("6.14.0")
        };

        cache
            .store("old", &CacheArtifacts::new(&image), &meta_old)
            .unwrap();
        cache
            .store("new", &CacheArtifacts::new(&image), &meta_new)
            .unwrap();
        cache
            .store("mid", &CacheArtifacts::new(&image), &meta_mid)
            .unwrap();

        let removed = cache.clean_keep(1).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key(), "new");
    }

    #[test]
    fn cache_dir_clean_keep_more_than_exist() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        cache
            .store(
                "only",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.2"),
            )
            .unwrap();

        let removed = cache.clean_keep(5).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(cache.list().unwrap().len(), 1);
    }

    #[test]
    fn cache_dir_clean_empty_cache() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let removed = cache.clean_all().unwrap();
        assert_eq!(removed, 0);
    }

    // -- resolve_cache_root --

    #[test]
    fn cache_resolve_root_ktstr_cache_dir() {
        // `lock_env()` serializes against sibling env-touching tests
        // in test_support/model.rs and the cache_resolve_root_* group
        // below. See `cache_dir_default_root_returns_path` for the
        // long-form rationale (Tester finding T1).
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("custom-cache");
        // Temporarily set env var for this test.
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", &dir);
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, dir);
    }

    #[test]
    fn cache_resolve_root_xdg_cache_home() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, tmp.path().join("ktstr").join("kernels"));
    }

    #[test]
    fn cache_resolve_root_empty_ktstr_cache_dir_falls_through() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path());
        let root = resolve_cache_root().unwrap();
        assert_eq!(root, tmp.path().join("ktstr").join("kernels"));
    }

    #[test]
    fn cache_resolve_root_empty_xdg_falls_to_home() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::set("XDG_CACHE_HOME", "");
        let _guard3 = EnvVarGuard::set("HOME", tmp.path());
        let root = resolve_cache_root().unwrap();
        assert_eq!(
            root,
            tmp.path().join(".cache").join("ktstr").join("kernels")
        );
    }

    // -- resolve_cache_root error paths --

    #[test]
    fn cache_resolve_root_home_unset_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::remove("HOME");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME is unset"),
            "expected HOME-unset error, got: {msg}"
        );
        assert!(
            !msg.contains("HOME is set to the empty string"),
            "unset HOME must NOT use the empty-string diagnostic — the two \
             cases are distinct now (NotPresent vs Ok(\"\")), got: {msg}",
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    /// A HOME literal of `"/"` (legacy root convention, container
    /// init that forgot to override HOME) must NOT silently produce
    /// `/.cache/ktstr/kernels` — that path's statvfs reports the
    /// root filesystem's free space, which is typically a small
    /// constrained mount and never the user's intended cache
    /// location. Bail with a diagnostic that names the resulting
    /// junk path and points the operator at a remediation.
    #[test]
    fn cache_resolve_root_home_root_slash_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "/");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME is `/`"),
            "expected HOME=/ specific error, got: {msg}"
        );
        assert!(
            msg.contains("/.cache/ktstr"),
            "diagnostic must cite the offending cache path, got: {msg}"
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    /// A HOME literal of `""` (empty string) is just as broken as
    /// unset (`PathBuf::from("").join(".cache")` produces a
    /// relative `.cache` rooted at the process CWD instead of the
    /// user's home), but the diagnostic now distinguishes the two
    /// shapes: empty-string assignment hits the `Ok("")` arm of
    /// `validate_home_for_cache`, surfacing "HOME is set to the
    /// empty string" so an operator can identify a Dockerfile
    /// `ENV HOME=` or shell-rc `export HOME=` typo as the cause
    /// rather than a missing init-time assignment.
    #[test]
    fn cache_resolve_root_home_empty_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HOME is set to the empty string"),
            "empty-HOME bail must use the empty-string diagnostic, got: {msg}",
        );
        assert!(
            !msg.contains("HOME is unset"),
            "empty-HOME must NOT use the unset diagnostic — the two \
             cases are distinct now, got: {msg}",
        );
    }

    /// A relative-path HOME (e.g. `HOME=relative/dir`) silently
    /// resolves the cache against CWD, which silently relocates
    /// the cache as the operator changes directories — a usability
    /// nightmare worse than a deferred error. Pin the explicit
    /// rejection so a regression that drops the absolute-path
    /// check surfaces here instead of as a hard-to-diagnose
    /// "cache contents disappeared" report from the operator.
    #[test]
    fn cache_resolve_root_home_relative_path_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "relative/dir");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not an absolute path"),
            "expected relative-path-specific error, got: {msg}"
        );
        assert!(
            msg.contains("relative/dir"),
            "diagnostic must cite the offending HOME value, got: {msg}"
        );
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error should suggest KTSTR_CACHE_DIR, got: {msg}"
        );
    }

    /// A bare-name HOME (no path separators at all, e.g. `HOME=tmp`)
    /// is also relative — `PathBuf::from("tmp").join(".cache")`
    /// yields `tmp/.cache` against CWD. Pin separately from the
    /// `relative/dir` case to confirm the absolute-path check
    /// isn't accidentally permissive on shapes that lack a `/`.
    #[test]
    fn cache_resolve_root_home_bare_name_relative_error() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _guard3 = EnvVarGuard::set("HOME", "tmp");
        let err = resolve_cache_root().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not an absolute path"),
            "expected relative-path-specific error, got: {msg}"
        );
        assert!(
            msg.contains("\"tmp\""),
            "diagnostic must cite the offending HOME value via its Debug \
             representation, got: {msg}"
        );
    }

    /// Sanity check the happy path: an absolute HOME pointing at a
    /// real directory must resolve through the gate to the expected
    /// `$HOME/.cache/ktstr/kernels` path. Pins that the new
    /// validation does not over-reject — a regression that hardens
    /// the gate further (e.g. requires HOME to exist via metadata)
    /// would break this.
    #[test]
    fn cache_resolve_root_home_absolute_passes() {
        let _lock = lock_env();
        let _guard1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _guard2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let tmp = TempDir::new().expect("tempdir");
        let _guard3 = EnvVarGuard::set("HOME", tmp.path());
        let resolved = resolve_cache_root().expect("absolute HOME must resolve");
        let expected = tmp.path().join(".cache").join("ktstr").join("kernels");
        assert_eq!(
            resolved, expected,
            "absolute HOME must produce $HOME/.cache/ktstr/kernels",
        );
    }

    /// A non-UTF-8 `KTSTR_CACHE_DIR` must fail fast with an
    /// actionable diagnostic rather than silently falling through to
    /// `$XDG_CACHE_HOME` / `$HOME`. Before the `NotUnicode` branch
    /// existed, `std::env::var` returned `Err` and the old `if let
    /// Ok(..)` guard dropped the override without a trace — an
    /// operator who set the variable would see ktstr write to a
    /// surprising directory under `$HOME` and have no clue why the
    /// override was ignored.
    ///
    /// `EnvVarGuard::set` accepts arbitrary `OsStr`, so the test can
    /// plant a lone 0xFF byte (valid on Unix filesystems, invalid as
    /// UTF-8) and observe the bail.
    #[test]
    #[cfg(unix)]
    fn cache_resolve_root_non_utf8_ktstr_cache_dir_bails() {
        // `lock_env()` for the same reason every other env-touching
        // cache.rs test holds it (Tester finding T1).
        let _lock = lock_env();
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let bytes: &[u8] = b"/tmp/ktstr-\xFFcache";
        let value = OsStr::from_bytes(bytes);
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", value);
        let err = resolve_cache_root()
            .expect_err("non-UTF-8 KTSTR_CACHE_DIR must bail, not silently fall through");
        let msg = err.to_string();
        assert!(
            msg.contains("KTSTR_CACHE_DIR"),
            "error must name the offending variable, got: {msg}",
        );
        assert!(
            msg.contains("non-UTF-8"),
            "error must mention non-UTF-8 so the operator knows the encoding, \
             got: {msg}",
        );
        assert!(
            msg.contains("UTF-8") || msg.contains("unset") || msg.contains("ASCII"),
            "error must name a remediation (UTF-8 replacement or unset), \
             got: {msg}",
        );
    }

    // -- path_inside_cache_root direct unit tests --
    //
    // `path_inside_cache_root` is the gate that prevents
    // `<vmlinux>.btf` sidecar writes from polluting kernel source
    // trees and other directories the cache does not own (see
    // `monitor::btf_offsets::load_btf_from_path`). The tests below
    // exercise the helper directly so a regression in the
    // membership logic surfaces against this dedicated entry point
    // as well as the integration paths in btf_offsets.rs.
    //
    // Every test holds `lock_env()` because the helper reads
    // KTSTR_CACHE_DIR / XDG_CACHE_HOME / HOME from the process
    // environment.

    /// Sibling path that lives directly under the cache root resolves
    /// as in-cache. The canonical case the gate must accept.
    #[test]
    fn path_inside_cache_root_accepts_path_inside() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let entry = tmp.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"placeholder").unwrap();
        assert!(
            path_inside_cache_root(&vmlinux),
            "vmlinux directly under cache root must be classified as in-cache",
        );
    }

    /// Path that lives in a sibling tempdir (modeling a kernel source
    /// tree) must NOT classify as in-cache — the source-tree pollution
    /// scenario this helper exists to prevent.
    #[test]
    fn path_inside_cache_root_rejects_path_outside() {
        let _lock = lock_env();
        let cache_root = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
        let source_tree = TempDir::new().unwrap();
        let vmlinux = source_tree.path().join("vmlinux");
        std::fs::write(&vmlinux, b"placeholder").unwrap();
        assert!(
            !path_inside_cache_root(&vmlinux),
            "vmlinux in a sibling tempdir must NOT be classified as in-cache",
        );
    }

    /// Bare filename has no parent component (`Path::parent` returns
    /// `Some("")`). Treating an empty parent as "in cache" would let
    /// any process invoking `path_inside_cache_root("vmlinux")` pass
    /// when its CWD happens to be the cache root — surprising
    /// semantics for what is otherwise a structural check. The
    /// helper short-circuits to false.
    #[test]
    fn path_inside_cache_root_rejects_bare_filename() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        let bare = std::path::Path::new("vmlinux");
        assert!(
            !path_inside_cache_root(bare),
            "bare filename (no parent) must short-circuit to false",
        );
    }

    /// With every cascade variable removed (`KTSTR_CACHE_DIR`,
    /// `XDG_CACHE_HOME`, `HOME`), `resolve_cache_root` errors out and
    /// `path_inside_cache_root` falls through to `false`. A real
    /// vmlinux on disk under those conditions is "outside the cache"
    /// because no cache root exists.
    #[test]
    fn path_inside_cache_root_false_when_unresolvable() {
        let _lock = lock_env();
        let _g1 = EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _g2 = EnvVarGuard::remove("XDG_CACHE_HOME");
        let _g3 = EnvVarGuard::remove("HOME");
        // Use a real file so `Path::parent` is non-empty; the
        // unresolvable cache root is the part being tested.
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("vmlinux");
        std::fs::write(&f, b"x").unwrap();
        assert!(
            !path_inside_cache_root(&f),
            "unresolvable cache root must classify as outside-cache (false)",
        );
    }

    /// Parent that does not exist on disk: `fs::canonicalize(parent)`
    /// fails with ENOENT, and the helper safely returns false.
    /// Models a caller passing a path whose ancestor was rmdir'd
    /// between path construction and the membership check.
    #[test]
    fn path_inside_cache_root_false_when_parent_canonicalize_fails() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", tmp.path());
        // Construct a path whose parent does not exist on disk.
        let nonexistent = std::path::Path::new("/this/parent/should/not/exist/vmlinux");
        assert!(
            !nonexistent.parent().unwrap().exists(),
            "precondition: parent must not exist for the canonicalize \
             failure path to be exercised",
        );
        assert!(
            !path_inside_cache_root(nonexistent),
            "nonexistent parent must surface as outside-cache, not panic",
        );
    }

    /// Symlink whose parent's canonical form lands INSIDE the cache
    /// root: classified as in-cache, even though the symlink itself
    /// lives outside. This is the desired semantic for callers that
    /// follow `find_vmlinux`-style fallbacks where the resolved
    /// target is what matters.
    #[test]
    #[cfg(unix)]
    fn path_inside_cache_root_follows_symlink_into_cache() {
        let _lock = lock_env();
        let cache_root = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
        // Real vmlinux inside the cache.
        let entry = cache_root.path().join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let real = entry.join("vmlinux");
        std::fs::write(&real, b"placeholder").unwrap();
        // Symlink whose PARENT (an outside-cache tempdir) holds a
        // link to the in-cache real file. `path.parent()` is the
        // outside-cache dir, but `fs::canonicalize(parent)`
        // resolves... the parent itself, not the symlink target.
        // To exercise the "symlink resolving INTO cache" semantics
        // we need a parent that IS itself a symlink into the
        // cache. Use a directory symlink for that.
        let outside = TempDir::new().unwrap();
        let alias_parent = outside.path().join("alias");
        std::os::unix::fs::symlink(&entry, &alias_parent).unwrap();
        let through_alias = alias_parent.join("vmlinux");
        assert!(
            through_alias.exists(),
            "precondition: path through symlinked parent must be reachable",
        );
        assert!(
            path_inside_cache_root(&through_alias),
            "path whose parent symlink resolves into cache must classify as in-cache",
        );
    }

    /// Symlink whose parent's canonical form lands OUTSIDE the cache
    /// root: classified as outside-cache. The opposite of
    /// `path_inside_cache_root_follows_symlink_into_cache` — together
    /// the two pin the canonicalize-then-compare contract.
    #[test]
    #[cfg(unix)]
    fn path_inside_cache_root_follows_symlink_out_of_cache() {
        let _lock = lock_env();
        let cache_root = TempDir::new().unwrap();
        let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_root.path());
        // Real vmlinux outside the cache.
        let outside = TempDir::new().unwrap();
        let real = outside.path().join("vmlinux");
        std::fs::write(&real, b"placeholder").unwrap();
        // Symlink directory IN cache pointing at the outside parent.
        let alias_parent = cache_root.path().join("alias");
        std::os::unix::fs::symlink(outside.path(), &alias_parent).unwrap();
        let through_alias = alias_parent.join("vmlinux");
        assert!(
            through_alias.exists(),
            "precondition: path through symlinked parent must be reachable",
        );
        assert!(
            !path_inside_cache_root(&through_alias),
            "path whose parent symlink resolves OUT of cache must classify as outside-cache",
        );
    }

    /// Empty `KTSTR_CACHE_DIR` falls through the cascade per
    /// `resolve_cache_root_with_suffix`'s `Ok(_) => fall through`
    /// arm; the helper then resolves through XDG/HOME like any
    /// other call. With XDG and HOME pointed at a tempdir, a path
    /// inside that tempdir's `ktstr/kernels/` subtree must classify
    /// as in-cache; a sibling outside it must not.
    #[test]
    fn path_inside_cache_root_empty_ktstr_cache_dir_falls_through() {
        let _lock = lock_env();
        let tmp = TempDir::new().unwrap();
        let _g1 = EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _g2 = EnvVarGuard::set("XDG_CACHE_HOME", tmp.path());
        // resolve_cache_root with empty KTSTR_CACHE_DIR + XDG set →
        // root = <tmp>/ktstr/kernels. Stage a vmlinux inside that
        // resolved root.
        let resolved = tmp.path().join("ktstr").join("kernels");
        let entry = resolved.join("kentry");
        std::fs::create_dir_all(&entry).unwrap();
        let vmlinux = entry.join("vmlinux");
        std::fs::write(&vmlinux, b"placeholder").unwrap();
        assert!(
            path_inside_cache_root(&vmlinux),
            "with empty KTSTR_CACHE_DIR, the cascade must resolve via \
             XDG_CACHE_HOME and accept paths inside that resolved root",
        );
    }

    /// Resolution is performed fresh on every call: changing
    /// `KTSTR_CACHE_DIR` between two invocations must yield
    /// different membership decisions for the same input path.
    /// Memoization would surface here as a stale `true` after the
    /// pointer moves away from the path's parent.
    #[test]
    fn path_inside_cache_root_fresh_resolution_per_call() {
        let _lock = lock_env();
        let cache_a = TempDir::new().unwrap();
        let cache_b = TempDir::new().unwrap();
        // Stage a vmlinux inside cache_a.
        let entry_a = cache_a.path().join("kentry");
        std::fs::create_dir_all(&entry_a).unwrap();
        let vmlinux_a = entry_a.join("vmlinux");
        std::fs::write(&vmlinux_a, b"placeholder").unwrap();
        // First call: KTSTR_CACHE_DIR points at cache_a → in-cache.
        {
            let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_a.path());
            assert!(
                path_inside_cache_root(&vmlinux_a),
                "first call: vmlinux is inside cache_a (the active root)",
            );
        }
        // Second call: KTSTR_CACHE_DIR moved to cache_b. Same input
        // path is now outside the active root.
        {
            let _guard = EnvVarGuard::set("KTSTR_CACHE_DIR", cache_b.path());
            assert!(
                !path_inside_cache_root(&vmlinux_a),
                "second call: KTSTR_CACHE_DIR has moved to cache_b, so the \
                 vmlinux (still under cache_a) must be classified outside",
            );
        }
    }

    // -- clean_orphaned_tmp_dirs unit tests --
    //
    // Parser/dispatcher coverage: the scan must remove directories
    // under `.tmp-{key}-{pid}` whose `{pid}` is verifiably dead,
    // must LEAVE malformed entries and non-`.tmp-` entries alone,
    // and must tolerate a nonexistent cache root.

    /// A `.tmp-{key}-{pid}` directory whose pid refers to a dead
    /// process is removed. Uses `libc::pid_t::MAX` — above
    /// `PID_MAX_LIMIT` (2^22), so no live process can ever claim it
    /// (same technique as `process_alive_nonexistent_pid` in
    /// scenario tests, removes the pid-reuse race from the test).
    #[test]
    fn clean_orphaned_tmp_dirs_removes_dead_pid_tempdir() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let orphan = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}somekey-{dead_pid}"));
        std::fs::create_dir_all(&orphan).unwrap();
        // Plant a nested file so a regression that hand-rolled
        // `remove_dir` (non-recursive) instead of `remove_dir_all`
        // would fail with ENOTEMPTY and the dir would survive.
        std::fs::write(orphan.join("inner.txt"), b"data").unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            !orphan.exists(),
            "dead-pid tempdir must be removed by clean_orphaned_tmp_dirs",
        );
    }

    /// A `.tmp-{key}-{pid}` directory whose pid is LIVE (the test
    /// process itself) must be preserved. `kill(getpid(), None)`
    /// returns `Ok(())` inside `clean_orphaned_tmp_dirs`'s liveness
    /// probe, which routes to the `!dead` continue branch.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_live_pid_tempdir() {
        let tmp = TempDir::new().unwrap();
        let live_pid = unsafe { libc::getpid() };
        let keeper = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}somekey-{live_pid}"));
        std::fs::create_dir_all(&keeper).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            keeper.exists(),
            "live-pid tempdir must NOT be removed — its owner is still running",
        );
    }

    /// Entries whose suffix cannot be parsed as a pid (non-numeric
    /// or empty after the trailing `-`) must be left alone — they
    /// do not match our format and may belong to an unrelated
    /// tool. Covers the `rsplit_once` / `parse::<i32>` continue
    /// branches.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_malformed_suffix_alone() {
        let tmp = TempDir::new().unwrap();
        // Case A: non-numeric suffix.
        let nonnum = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-notapid"));
        std::fs::create_dir_all(&nonnum).unwrap();
        // Case B: empty suffix (name ends with `-`).
        let empty_suf = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-"));
        std::fs::create_dir_all(&empty_suf).unwrap();
        // Case C: no `-` at all after the prefix (rsplit_once
        // still finds the `-` inside the prefix itself, but
        // `.tmp` parses as non-numeric → continue).
        let no_dash = tmp.path().join(format!("{TMP_DIR_PREFIX}nokeyhere"));
        std::fs::create_dir_all(&no_dash).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(nonnum.exists(), "non-numeric pid suffix must be left alone");
        assert!(empty_suf.exists(), "empty pid suffix must be left alone");
        assert!(no_dash.exists(), "no-pid-suffix entry must be left alone");
    }

    /// Directories that do not begin with [`TMP_DIR_PREFIX`] must
    /// never be touched. The cache root also holds real cache
    /// entries (hash-keyed directories), and an overbroad scan
    /// would wipe them out.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_unrelated_entries_alone() {
        let tmp = TempDir::new().unwrap();
        let real_entry = tmp.path().join("real-cache-entry");
        std::fs::create_dir_all(&real_entry).unwrap();
        let other = tmp.path().join("not-a-tempdir");
        std::fs::create_dir_all(&other).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            real_entry.exists(),
            "unrelated cache entry must be preserved"
        );
        assert!(other.exists(), "unrelated directory must be preserved");
    }

    /// Non-UTF-8 filenames in the cache root must be skipped
    /// silently — they cannot be a `.tmp-{key}-{pid}` directory
    /// this module created (all our names are ASCII), and bailing
    /// on every stray non-UTF-8 entry would fail the whole cleanup
    /// pass.
    ///
    /// Unix-only because the byte-level name construction uses
    /// `OsStr::from_bytes`, which is Unix-only. Other platforms
    /// cannot produce a non-UTF-8 filesystem name from this test
    /// code.
    #[test]
    #[cfg(unix)]
    fn clean_orphaned_tmp_dirs_skips_non_utf8_names() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let tmp = TempDir::new().unwrap();
        // Name that looks like a tempdir prefix but has a non-UTF-8
        // byte after. `into_string()` in the scan returns Err(_)
        // and the continue branch skips it.
        let mut bytes: Vec<u8> = b".tmp-".to_vec();
        bytes.push(0xFF);
        bytes.extend_from_slice(b"-123");
        let bad_name = OsStr::from_bytes(&bytes);
        let bad_path = tmp.path().join(bad_name);
        std::fs::create_dir(&bad_path).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            bad_path.exists(),
            "non-UTF-8 entry must be left alone — the scan cannot \
             confirm it matches our format, so safe-default is skip",
        );
    }

    /// A nonexistent cache root returns `Ok(())` without error —
    /// called from the `store()` prologue, which may execute
    /// before any cache operation has created the directory.
    #[test]
    fn clean_orphaned_tmp_dirs_handles_missing_cache_root() {
        let tmp = TempDir::new().unwrap();
        let never_created = tmp.path().join("never-created");
        // `is_dir()` short-circuits to Ok(()) without read_dir.
        clean_orphaned_tmp_dirs(&never_created).unwrap();
    }

    /// Multi-entry mix: a DEAD-pid orphan and a LIVE-pid tempdir
    /// side by side — only the dead one is removed. Pins the
    /// per-entry classification logic against a regression that
    /// bailed on the first entry's liveness-probe error or
    /// short-circuited after the first successful remove.
    #[test]
    fn clean_orphaned_tmp_dirs_mixed_entries() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let live_pid = unsafe { libc::getpid() };
        let dead = tmp.path().join(format!("{TMP_DIR_PREFIX}a-{dead_pid}"));
        let live = tmp.path().join(format!("{TMP_DIR_PREFIX}b-{live_pid}"));
        let unrelated = tmp.path().join("c-regular-entry");
        std::fs::create_dir_all(&dead).unwrap();
        std::fs::create_dir_all(&live).unwrap();
        std::fs::create_dir_all(&unrelated).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(!dead.exists(), "dead orphan must be removed");
        assert!(live.exists(), "live-pid entry must survive");
        assert!(unrelated.exists(), "unrelated entry must survive");
    }

    /// `pid == 0` suffix: the scan rejects non-positive pids before
    /// the liveness probe runs. `kill(0, None)` has process-group
    /// broadcast semantics (POSIX: signal pid 0 = self's pgrp) and
    /// could not authoritatively classify a `.tmp-key-0` directory
    /// as belonging to a dead orphan even on `Err(ESRCH)`, so the
    /// `pid <= 0` filter forces the safe default (preserve) without
    /// invoking the probe at all.
    #[test]
    fn clean_orphaned_tmp_dirs_preserves_pid_zero_suffix() {
        let tmp = TempDir::new().unwrap();
        let entry = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey-0"));
        std::fs::create_dir_all(&entry).unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            entry.exists(),
            "pid=0 suffix must be preserved — `pid <= 0` filter \
             skips the entry before kill(0, None)'s pgrp-broadcast \
             ambiguity can reach the liveness probe",
        );
    }

    /// "Negative pid suffix" unreachability pin. The parser uses
    /// `rsplit_once('-')` to extract the suffix AFTER the last `-`,
    /// which by construction never contains a `-` — so
    /// `parse::<i32>()` on the suffix can only produce a
    /// non-negative integer (or fail to parse). A filename like
    /// `.tmp-key--12345` rsplits into `(".tmp-key-", "12345")`:
    /// the suffix `"12345"` parses to pid 12345 (POSITIVE).
    ///
    /// This test documents the invariant and pins the observable
    /// behavior: under `.tmp-key--12345`, the pid parses as a
    /// real positive integer (12345), `kill(12345, None)` likely
    /// returns `Err(ESRCH)` on a fresh pid space, and the entry
    /// is REMOVED (not preserved). The test verifies the REMOVAL
    /// path under this input so a future refactor that changed
    /// `rsplit_once('-')` to `splitn(3, '-')` or a regex — which
    /// COULD emit a `-12345` suffix and open the negative-pid
    /// door — would change the observable behavior and trip this
    /// test's "entry must be gone" assertion.
    ///
    /// Note: the test's observable outcome depends on pid 12345
    /// NOT being alive on the host. If a coincidental live
    /// process happens to hold pid 12345, the entry would be
    /// preserved instead; accept the ≈1-in-N-pids risk
    /// (empirically negligible in CI / dev environments) rather
    /// than contort the test to force a guaranteed-dead pid (the
    /// existing `dead_pid = libc::pid_t::MAX` technique produces
    /// a suffix too large to demonstrate the `--` splitting
    /// behavior).
    #[test]
    fn clean_orphaned_tmp_dirs_double_dash_parses_as_positive_pid() {
        let tmp = TempDir::new().unwrap();
        // Name with a double-dash so the rsplit-once path produces
        // the suffix "12345" (no leading dash — rsplit_once
        // guarantees no delimiter in the suffix). A future regex
        // that emitted "-12345" would behave differently here.
        let entry = tmp.path().join(format!("{TMP_DIR_PREFIX}somekey--12345"));
        std::fs::create_dir_all(&entry).unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();

        // Whether the entry is removed depends on whether pid
        // 12345 is alive at test time. The invariant being pinned
        // is the parse direction (positive, not negative), which
        // is a prerequisite for either the remove or preserve
        // branch — a refactor to a negative-suffix parser would
        // land in the `kill(-12345, None)` broadcast probe
        // instead, which returns `Ok(())` and preserves
        // unconditionally. Testing both "parses as positive" AND
        // "either removed or preserved based on liveness" together
        // requires nothing stronger than a liveness check here.
        //
        // The TEST IS PRIMARILY A DOC — the comment above explains
        // the negative-pid unreachability. The assertion below
        // guards against the most likely concrete regression: a
        // regex that emits a `-N` suffix and thereby lands in the
        // broadcast-probe branch. Under that regression, `kill(-N,
        // None)` returns Ok and the entry is ALWAYS preserved;
        // this assertion is satisfied only if the current parse
        // direction (positive pid, real liveness probe) holds.
        //
        // Use `kill(12345, None)` here to decide what we expect:
        // if the pid is live, the entry is preserved; if dead,
        // removed. Either result confirms positive-pid parse.
        let pid_alive = matches!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(12345), None),
            Ok(()),
        );
        if pid_alive {
            assert!(
                entry.exists(),
                "pid 12345 was alive at probe time → entry must be \
                 preserved; got: entry removed (regression?)",
            );
        } else {
            assert!(
                !entry.exists(),
                "pid 12345 was dead at probe time → entry must be \
                 removed (proves positive-pid parse). A regression to \
                 negative-pid parse would preserve unconditionally; \
                 entry still exists.",
            );
        }
    }

    /// Regular file entry (not a directory) whose name MATCHES the
    /// `.tmp-{key}-{pid}` pattern with a dead pid. `fs::remove_dir_all`
    /// on a regular file returns `ENOTDIR` / `NotADirectory`; the
    /// scan catches the error in its match arm, logs + continues,
    /// and the file stays in place. Pins that the scan does NOT
    /// fall through to `fs::remove_file` on type mismatch —
    /// quietly removing a file with a tempdir-shaped name could
    /// destroy state belonging to an unrelated tool that happened
    /// to pick a colliding name.
    #[test]
    fn clean_orphaned_tmp_dirs_leaves_regular_file_entry() {
        let tmp = TempDir::new().unwrap();
        let dead_pid = libc::pid_t::MAX;
        let file_entry = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}fileshaped-{dead_pid}"));
        std::fs::write(&file_entry, b"not a directory").unwrap();
        clean_orphaned_tmp_dirs(tmp.path()).unwrap();
        assert!(
            file_entry.exists(),
            "regular file with tempdir-shaped name + dead pid must \
             NOT be removed — `remove_dir_all` errors on a file, \
             and the scan's error-tolerance contract leaves it",
        );
    }

    /// Symlink entry whose NAME matches the tempdir pattern but
    /// whose TARGET is an unrelated path outside the cache. The
    /// scan must not follow the symlink — following would risk
    /// `remove_dir_all` deleting the target's contents (the very
    /// bug that a production cache cleaner must never commit).
    ///
    /// Rust's `std::fs::remove_dir_all` on modern platforms uses
    /// `openat` + symlink-aware checks to refuse to follow
    /// symlinks; this test pins that guarantee against a regression
    /// that reached for `fs::remove_dir` (which follows) or hand-
    /// rolled a recursive walk that followed links.
    #[test]
    #[cfg(unix)]
    fn clean_orphaned_tmp_dirs_leaves_symlink_entry() {
        let tmp = TempDir::new().unwrap();

        // Create the real target directory OUTSIDE the cache root
        // — the test asserts the target's contents survive even
        // though the symlink shares the tempdir-like name + dead
        // pid.
        let target_root = TempDir::new().unwrap();
        let target_file = target_root.path().join("sentinel.txt");
        std::fs::write(&target_file, b"must-not-be-deleted").unwrap();

        let dead_pid = libc::pid_t::MAX;
        let symlink = tmp
            .path()
            .join(format!("{TMP_DIR_PREFIX}symkey-{dead_pid}"));
        std::os::unix::fs::symlink(target_root.path(), &symlink).unwrap();

        clean_orphaned_tmp_dirs(tmp.path()).unwrap();

        // Either the symlink itself was removed (modern
        // `remove_dir_all` removes the link without following) OR
        // the symlink stayed (older `remove_dir_all` that errored
        // on symlinks). The LOAD-BEARING invariant is that the
        // TARGET's contents survive — the test's safety guarantee
        // is "data outside the cache root is untouched", not
        // "the symlink entry itself must survive".
        assert!(
            target_file.exists(),
            "symlink target's contents must survive the scan — \
             following symlinks would delete unrelated state \
             outside the cache root, a critical security / data- \
             safety regression",
        );
        assert_eq!(
            std::fs::read(&target_file).unwrap(),
            b"must-not-be-deleted",
            "target file content must be unchanged",
        );
    }

    // -- validate_cache_key unit tests --

    #[test]
    fn cache_validate_key_rejects_empty() {
        let err = validate_cache_key("").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn cache_validate_key_rejects_whitespace_only() {
        let err = validate_cache_key("   ").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn cache_validate_key_rejects_forward_slash() {
        let err = validate_cache_key("a/b").unwrap_err();
        assert!(err.to_string().contains("path separator"));
    }

    #[test]
    fn cache_validate_key_rejects_backslash() {
        let err = validate_cache_key("a\\b").unwrap_err();
        assert!(err.to_string().contains("path separator"));
    }

    #[test]
    fn cache_validate_key_rejects_dotdot() {
        // Use a key without slashes to specifically hit the ".." check
        // (slashes are rejected first by the separator check).
        let err = validate_cache_key("foo..bar").unwrap_err();
        assert!(err.to_string().contains("path traversal"));
    }

    #[test]
    fn cache_validate_key_rejects_null_byte() {
        let err = validate_cache_key("key\0evil").unwrap_err();
        assert!(err.to_string().contains("null"));
    }

    #[test]
    fn cache_validate_key_rejects_tmp_prefix() {
        let err = validate_cache_key(".tmp-in-progress").unwrap_err();
        assert!(
            err.to_string().contains(".tmp-"),
            "expected .tmp- rejection, got: {err}"
        );
    }

    #[test]
    fn cache_validate_key_rejects_dot() {
        let err = validate_cache_key(".").unwrap_err();
        assert!(
            err.to_string().contains("directory reference"),
            "expected dot rejection, got: {err}"
        );
    }

    #[test]
    fn cache_validate_key_rejects_dotdot_bare() {
        let err = validate_cache_key("..").unwrap_err();
        assert!(
            err.to_string().contains("directory reference"),
            "expected dotdot rejection, got: {err}"
        );
    }

    #[test]
    fn cache_validate_key_accepts_valid() {
        assert!(validate_cache_key("6.14.2-tarball-x86_64").is_ok());
        assert!(validate_cache_key("local-deadbeef-x86_64").is_ok());
        assert!(validate_cache_key("v6.14-git-a1b2c3d-aarch64").is_ok());
    }

    // -- validate_filename --

    #[test]
    fn cache_validate_filename_rejects_traversal() {
        assert!(validate_filename("../etc/passwd").is_err());
        assert!(validate_filename("foo/../bar").is_err());
    }

    #[test]
    fn cache_validate_filename_rejects_empty() {
        assert!(validate_filename("").is_err());
    }

    #[test]
    fn cache_validate_filename_accepts_valid() {
        assert!(validate_filename("bzImage").is_ok());
        assert!(validate_filename("Image").is_ok());
    }

    // -- image_name traversal via store --

    #[test]
    fn cache_dir_store_rejects_image_name_traversal() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let mut meta = test_metadata("6.14.2");
        meta.image_name = "../escape".to_string();

        let err = cache
            .store("valid-key", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("image name"),
            "expected image_name rejection, got: {err}"
        );
    }

    // -- .tmp- prefix via store/lookup --

    #[test]
    fn cache_dir_store_tmp_prefix_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store(".tmp-sneaky", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains(".tmp-"),
            "expected .tmp- rejection, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_tmp_prefix_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup(".tmp-sneaky").is_none());
    }

    // -- cache key validation via store/lookup --

    #[test]
    fn cache_dir_store_empty_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty-key error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_empty_key_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup("").is_none());
    }

    #[test]
    fn cache_dir_store_path_traversal_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("../escape", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("path"),
            "expected path-traversal error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_lookup_path_traversal_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        assert!(cache.lookup("../escape").is_none());
        assert!(cache.lookup("foo/../bar").is_none());
    }

    #[test]
    fn cache_dir_store_slash_in_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("a/b", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("path separator"),
            "expected path-separator error, got: {err}"
        );
    }

    #[test]
    fn cache_dir_store_whitespace_only_key_rejected() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let err = cache
            .store("   ", &CacheArtifacts::new(&image), &meta)
            .unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty/whitespace error, got: {err}"
        );
    }

    // -- clean with mixed valid + corrupt entries --

    #[test]
    fn cache_dir_clean_keep_n_with_mixed_entries() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        // Two valid entries with different timestamps.
        let meta_new = KernelMetadata {
            built_at: "2026-04-12T10:00:00Z".to_string(),
            ..test_metadata("6.14.2")
        };
        let meta_old = KernelMetadata {
            built_at: "2026-04-10T10:00:00Z".to_string(),
            ..test_metadata("6.13.0")
        };
        cache
            .store("new", &CacheArtifacts::new(&image), &meta_new)
            .unwrap();
        cache
            .store("old", &CacheArtifacts::new(&image), &meta_old)
            .unwrap();

        // One corrupt entry (no metadata).
        let corrupt_dir = tmp.path().join("cache").join("corrupt");
        fs::create_dir_all(&corrupt_dir).unwrap();

        // list() returns 3 entries. Corrupt entries (no built_at) sort
        // last. keep=1 should keep the newest valid entry and remove
        // the old valid + corrupt entries.
        let removed = cache.clean_keep(1).unwrap();
        assert_eq!(removed, 2);

        let remaining = cache.list().unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].key(), "new");
    }

    // -- atomic write safety --

    /// Regression for the rename-TOCTOU fix: a second `store` with
    /// the same key must atomically replace the previous entry's
    /// content without leaving half-installed state — even though
    /// the underlying code path exercises the
    /// `final_dir-exists → swap` branch rather than the plain
    /// rename. The new content wins, the old content is gone, and
    /// no `.evict-*` staging dir lingers under cache_root.
    #[test]
    fn cache_dir_store_overwrites_existing_key_atomically() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // First install.
        let src_a = TempDir::new().unwrap();
        let image_a = create_fake_image(src_a.path());
        fs::write(&image_a, b"version-a").unwrap();
        let mut meta_a = test_metadata("6.14.2");
        meta_a.built_at = "2026-04-10T00:00:00Z".to_string();
        let entry_a = cache
            .store("collide", &CacheArtifacts::new(&image_a), &meta_a)
            .unwrap();
        assert_eq!(
            fs::read(entry_a.path.join("bzImage")).unwrap(),
            b"version-a"
        );

        // Second install with the same key — exercises the rename-
        // to-staging branch. Different built_at so we can tell
        // which metadata won.
        let src_b = TempDir::new().unwrap();
        let image_b = create_fake_image(src_b.path());
        fs::write(&image_b, b"version-b").unwrap();
        let mut meta_b = test_metadata("6.14.2");
        meta_b.built_at = "2026-04-18T00:00:00Z".to_string();
        let entry_b = cache
            .store("collide", &CacheArtifacts::new(&image_b), &meta_b)
            .unwrap();

        // New content wins.
        assert_eq!(
            fs::read(entry_b.path.join("bzImage")).unwrap(),
            b"version-b",
            "new content must replace old content atomically"
        );
        let installed_meta = read_metadata(&entry_b.path).expect("metadata.json");
        assert_eq!(installed_meta.built_at, "2026-04-18T00:00:00Z");

        // No staging or tmp residue.
        for dirent in fs::read_dir(&cache_root).unwrap() {
            let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".evict-") && !name.starts_with(".tmp-"),
                "unexpected leftover directory under cache_root: {name}"
            );
        }
    }

    #[test]
    fn cache_dir_store_cleans_stale_tmp() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // Create a stale .tmp- directory simulating a prior crash.
        let stale_tmp = cache_root.join(format!(".tmp-mykey-{}", std::process::id()));
        fs::create_dir_all(&stale_tmp).unwrap();
        fs::write(stale_tmp.join("junk"), b"leftover").unwrap();

        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        // Store should succeed despite stale tmp dir.
        let entry = cache
            .store("mykey", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        // Stale tmp dir should be gone.
        assert!(!stale_tmp.exists());
    }

    /// Concurrent readers calling `lookup()` while a writer is
    /// rapidly overwriting the same cache key must never observe a
    /// half-installed entry. The atomic rename-to-staging swap in
    /// `store()` should make every successful lookup return an entry
    /// whose `image_path()` exists and whose contents match one of
    /// the writer's complete versions — never a missing file, never a
    /// truncated image.
    ///
    /// Pinning this behavior catches regressions where the swap
    /// sequence is reordered (e.g. removing `final_dir` before
    /// renaming the tmp dir into place) or replaced with a non-atomic
    /// copy. Such regressions would let a reader observe a cache
    /// entry with valid metadata but a missing `bzImage`, or a
    /// partially-written image with bytes from two generations.
    #[test]
    fn cache_dir_store_atomic_under_concurrent_readers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::thread;

        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = Arc::new(CacheDir::with_root(cache_root.clone()));

        // Writer sources: two distinct full versions with
        // recognizable, long-ish content so a torn read would be
        // detectable by byte comparison, not just length.
        let src_a = TempDir::new().unwrap();
        let image_a = src_a.path().join("bzImage");
        let content_a = b"AAAAAAAA-image-version-a-AAAAAAAA".repeat(64);
        fs::write(&image_a, &content_a).unwrap();

        let src_b = TempDir::new().unwrap();
        let image_b = src_b.path().join("bzImage");
        let content_b = b"BBBBBBBB-image-version-b-BBBBBBBB".repeat(64);
        fs::write(&image_b, &content_b).unwrap();

        // Prime the cache so lookup() has something to find from
        // iteration one onwards. Without priming, early readers would
        // legitimately see None until the writer lands the first
        // store — and we want to assert "never missing once present,"
        // which requires an initial present state.
        let meta_prime = test_metadata("6.14.2");
        cache
            .store("atomic-key", &CacheArtifacts::new(&image_a), &meta_prime)
            .unwrap();

        const WRITE_ITERATIONS: usize = 40;
        let stop = Arc::new(AtomicBool::new(false));
        let lookups_observed = Arc::new(AtomicUsize::new(0));
        let atomicity_violations = Arc::new(AtomicUsize::new(0));

        // Spawn reader threads. Each reader loops until `stop` is
        // set, calling lookup() and checking that when Some(entry)
        // comes back, the image file exists and matches one of the
        // two known writer contents byte-for-byte.
        let reader_count = 4;
        let mut readers = Vec::with_capacity(reader_count);
        for _ in 0..reader_count {
            let cache = Arc::clone(&cache);
            let stop = Arc::clone(&stop);
            let lookups_observed = Arc::clone(&lookups_observed);
            let violations = Arc::clone(&atomicity_violations);
            let expected_a = content_a.clone();
            let expected_b = content_b.clone();
            readers.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let Some(entry) = cache.lookup("atomic-key") else {
                        // Once primed, lookup must always see an
                        // entry. A None here is a real atomicity
                        // violation: the writer briefly removed the
                        // final_dir without immediately replacing it.
                        violations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    let image_path = entry.image_path();
                    let Ok(bytes) = fs::read(&image_path) else {
                        // Entry directory + metadata visible, but
                        // image file missing → non-atomic install.
                        violations.fetch_add(1, Ordering::Relaxed);
                        continue;
                    };
                    if bytes != expected_a && bytes != expected_b {
                        // Torn read: bytes don't match either
                        // complete version. Would indicate the image
                        // was observed mid-copy.
                        violations.fetch_add(1, Ordering::Relaxed);
                    }
                    lookups_observed.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // Writer: alternate between version A and version B,
        // exercising the rename-to-staging branch on every iteration
        // after the first (final_dir already exists from priming).
        for i in 0..WRITE_ITERATIONS {
            let (image, label) = if i % 2 == 0 {
                (&image_a, "a")
            } else {
                (&image_b, "b")
            };
            let mut meta = test_metadata("6.14.2");
            meta.built_at = format!("2026-04-18T00:00:{:02}Z", i % 60);
            meta.config_hash = Some(format!("iter-{i}-{label}"));
            cache
                .store("atomic-key", &CacheArtifacts::new(image), &meta)
                .expect("store under concurrent readers must not fail");
        }

        stop.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().expect("reader thread panicked");
        }

        assert_eq!(
            atomicity_violations.load(Ordering::Relaxed),
            0,
            "lookup observed a missing or torn cache entry during concurrent store; \
             rename-to-staging swap is not atomic",
        );
        assert!(
            lookups_observed.load(Ordering::Relaxed) > 0,
            "readers never observed a successful lookup — test did not \
             actually exercise the concurrency window",
        );

        // Post-condition: final state is intact and no staging or
        // tmp residue leaked out of the write loop.
        let final_entry = cache.lookup("atomic-key").expect("entry must exist");
        let final_bytes = fs::read(final_entry.image_path()).unwrap();
        assert!(
            final_bytes == content_a || final_bytes == content_b,
            "final image must match one of the writer's versions",
        );
        for dirent in fs::read_dir(&cache_root).unwrap() {
            let name = dirent.unwrap().file_name().to_string_lossy().into_owned();
            assert!(
                !name.starts_with(".evict-") && !name.starts_with(".tmp-"),
                "unexpected leftover directory under cache_root: {name}",
            );
        }
    }

    #[test]
    fn cache_dir_store_with_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = src_dir.path().join("vmlinux");
        fs::write(&vmlinux, b"fake vmlinux ELF").unwrap();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store(
                "with-vmlinux",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(entry.path.join("vmlinux").exists());
        assert!(entry.path.join("metadata.json").exists());
        // Metadata records has_vmlinux.
        assert!(entry.metadata.has_vmlinux);
        // Original files still exist (copy, not move).
        assert!(image.exists());
        assert!(vmlinux.exists());
    }

    #[test]
    fn cache_dir_store_without_vmlinux() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store("no-vmlinux", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert!(entry.path.join("bzImage").exists());
        assert!(!entry.path.join("vmlinux").exists());
        assert!(entry.path.join("metadata.json").exists());
        // Metadata records absence of vmlinux; vmlinux_stripped is
        // meaningless without a vmlinux but must still be false (the
        // strip pipeline never ran).
        assert!(!entry.metadata.has_vmlinux);
        assert!(!entry.metadata.vmlinux_stripped);
    }

    #[test]
    fn cache_dir_store_strips_vmlinux_internally() {
        // Real ELF fixture: store() must run strip_vmlinux_debug and
        // the stored vmlinux must reflect the strip (smaller than
        // source, no .debug_* sections).
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = create_strip_test_fixture(src_dir.path());
        let source_size = fs::metadata(&vmlinux).unwrap().len();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store(
                "strip-in-store",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        let cached_vmlinux = entry.path.join("vmlinux");
        let cached_size = fs::metadata(&cached_vmlinux).unwrap().len();
        assert!(
            cached_size < source_size,
            "stored vmlinux ({cached_size} bytes) should be smaller \
             than source ({source_size}) after internal strip"
        );
        let data = fs::read(&cached_vmlinux).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let section_names: Vec<&str> = elf
            .section_headers
            .iter()
            .filter_map(|s| elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        assert!(
            !section_names.contains(&".debug_info"),
            "internal strip should have removed .debug_info"
        );
        assert!(entry.metadata.has_vmlinux);
        assert!(
            entry.metadata.vmlinux_stripped,
            "strip-succeeds path must set vmlinux_stripped = true"
        );
    }

    #[test]
    fn cache_dir_store_falls_back_when_strip_fails() {
        // Unparseable vmlinux: strip errors, store() falls back to
        // copying the raw bytes. has_vmlinux stays true (so consumers
        // still see the sidecar) but vmlinux_stripped is false (so
        // consumers can tell the raw-fallback path ran).
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = src_dir.path().join("vmlinux");
        let raw = b"not an ELF file";
        fs::write(&vmlinux, raw).unwrap();
        let meta = test_metadata("6.14.2");

        let entry = cache
            .store(
                "strip-fallback",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        let cached = fs::read(entry.path.join("vmlinux")).unwrap();
        assert_eq!(cached, raw, "fallback must copy raw bytes verbatim");
        assert!(entry.metadata.has_vmlinux);
        assert!(
            !entry.metadata.vmlinux_stripped,
            "raw-fallback path must set vmlinux_stripped = false so \
             `ktstr cache list --json` surfaces the strip failure"
        );
    }

    // -- should_warn_unstripped (pure decision logic driving
    //    `CacheDir::lookup`'s per-lookup "unstripped vmlinux" warning).

    /// Helper for the three `should_warn_unstripped` tests below:
    /// construct a synthetic [`CacheEntry`] with explicit
    /// `has_vmlinux` / `vmlinux_stripped` bits and the rest of the
    /// metadata filled in from [`KernelMetadata::new`]. The entry-dir
    /// path is never touched (the decision logic only reads the
    /// metadata bools), so a synthetic PathBuf is enough.
    fn make_warn_test_entry(has_vmlinux: bool, vmlinux_stripped: bool) -> CacheEntry {
        let mut meta = KernelMetadata::new(
            KernelSource::Tarball,
            "x86_64".to_string(),
            "bzImage".to_string(),
            "2026-04-24T12:00:00Z".to_string(),
        );
        meta.set_has_vmlinux(has_vmlinux);
        meta.set_vmlinux_stripped(vmlinux_stripped);
        CacheEntry {
            key: "test-key".to_string(),
            path: PathBuf::from("/nonexistent/entry"),
            metadata: meta,
        }
    }

    /// An entry with a vmlinux that came from the raw-fallback path
    /// (strip failed at store time) MUST trigger the warning. The
    /// per-lookup eprintln is the operator's persistent signal to
    /// rebuild the cache.
    #[test]
    fn should_warn_unstripped_fires_when_vmlinux_present_and_unstripped() {
        let entry = make_warn_test_entry(true, false);
        assert!(
            should_warn_unstripped(&entry),
            "has_vmlinux=true + vmlinux_stripped=false must warn"
        );
    }

    /// An entry with a successfully-stripped vmlinux MUST NOT warn.
    /// This is the common case; warning here would be noise that
    /// operators learn to ignore, defeating the signal on the
    /// genuine failure case above.
    #[test]
    fn should_warn_unstripped_silent_when_vmlinux_stripped() {
        let entry = make_warn_test_entry(true, true);
        assert!(
            !should_warn_unstripped(&entry),
            "has_vmlinux=true + vmlinux_stripped=true must not warn"
        );
    }

    /// An entry with no vmlinux at all MUST NOT warn. The
    /// `vmlinux_stripped` bit is meaningless in that shape (always
    /// `false` by construction in [`CacheDir::store`]'s no-vmlinux
    /// branch) and warning would fire on every cache hit that simply
    /// did not cache a vmlinux — pure noise.
    #[test]
    fn should_warn_unstripped_silent_when_no_vmlinux() {
        let entry = make_warn_test_entry(false, false);
        assert!(
            !should_warn_unstripped(&entry),
            "has_vmlinux=false must not warn (no vmlinux to worry about)"
        );
    }

    #[test]
    fn cache_dir_store_preserves_original_vmlinux() {
        // strip_vmlinux_debug reads the source path; check the
        // source file is still there after store() (no move, no
        // truncate).
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = create_strip_test_fixture(src_dir.path());
        let source_size = fs::metadata(&vmlinux).unwrap().len();
        let meta = test_metadata("6.14.2");

        cache
            .store(
                "preserve-src",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &meta,
            )
            .unwrap();
        assert!(vmlinux.exists(), "source vmlinux must survive store()");
        assert_eq!(
            fs::metadata(&vmlinux).unwrap().len(),
            source_size,
            "source vmlinux size must not change"
        );
    }

    #[test]
    fn cache_dir_store_preserves_original_image() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");

        cache
            .store("key", &CacheArtifacts::new(&image), &meta)
            .unwrap();

        // Original image must still exist (copy, not move).
        assert!(image.exists());
    }

    // -- CacheEntry accessors --

    #[test]
    fn cache_entry_image_path_joins_key_with_image_name() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let entry = cache
            .store(
                "key",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        assert_eq!(entry.image_path(), entry.path.join("bzImage"));
        assert!(entry.image_path().exists());
    }

    #[test]
    fn cache_entry_vmlinux_path_some_when_stored() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let vmlinux = create_strip_test_fixture(src_dir.path());
        let entry = cache
            .store(
                "with-vml",
                &CacheArtifacts::new(&image).with_vmlinux(&vmlinux),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        let vml_path = entry.vmlinux_path().expect("vmlinux_path() should be Some");
        assert_eq!(vml_path, entry.path.join("vmlinux"));
        assert!(vml_path.exists());
    }

    #[test]
    fn cache_entry_vmlinux_path_none_when_not_stored() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let entry = cache
            .store(
                "no-vml",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.2"),
            )
            .unwrap();
        assert!(entry.vmlinux_path().is_none());
    }

    // -- KconfigStatus variants --

    #[test]
    fn kconfig_status_matches_when_hash_equal() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2").with_ktstr_kconfig_hash(Some("deadbeef".to_string()));
        let entry = cache
            .store("kc-match", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.kconfig_status("deadbeef"), KconfigStatus::Matches);
    }

    #[test]
    fn kconfig_status_untracked_when_no_hash_in_entry() {
        // Pre-kconfig-tracking cache entries (ktstr_kconfig_hash == None)
        // must surface as `Untracked`, not `Stale`. `find_kernel`'s
        // stale-filter at lib.rs treats `Untracked` as "keep" so legacy
        // entries remain usable — checked here so a regression that
        // conflates "no recorded hash" with "different hash" surfaces
        // at unit-test time.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        // test_metadata seeds ktstr_kconfig_hash = Some("def456"); strip
        // it here to hit the None branch in kconfig_status.
        let meta = KernelMetadata {
            ktstr_kconfig_hash: None,
            ..test_metadata("6.14.2")
        };
        let entry = cache
            .store("kc-untracked", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        assert_eq!(entry.kconfig_status("anything"), KconfigStatus::Untracked);
    }

    #[test]
    fn kconfig_status_stale_pins_cached_and_current_field_order() {
        // `KconfigStatus::Stale { cached, current }` names the two
        // hashes for diagnostics: `cached` is what the entry recorded
        // at build time, `current` is what the caller is comparing
        // against. A swap would invert the "was / is" story in every
        // diagnostic message consuming these fields (e.g. `kernel list`
        // tags, future error-formatting code). Pin the mapping so a
        // refactor that swaps the two construction args breaks this
        // test before it ships a misleading diagnostic.
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2").with_ktstr_kconfig_hash(Some("old_cached".to_string()));
        let entry = cache
            .store("kc-stale", &CacheArtifacts::new(&image), &meta)
            .unwrap();
        match entry.kconfig_status("new_current") {
            KconfigStatus::Stale { cached, current } => {
                assert_eq!(
                    cached, "old_cached",
                    "`cached` must hold the hash recorded in the entry"
                );
                assert_eq!(
                    current, "new_current",
                    "`current` must hold the hash the caller passed in"
                );
            }
            other => panic!("expected KconfigStatus::Stale, got {other:?}"),
        }
    }

    // -- prefer_source_tree_for_dwarf --

    #[test]
    fn prefer_source_tree_local_with_vmlinux() {
        // Local-source cache entry whose source tree is still on disk
        // and has a vmlinux whose size + mtime match the recorded
        // stat pair: helper returns the source tree path.
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len()),
            source_vmlinux_mtime_secs: Some(mtime_secs),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(prefer_source_tree_for_dwarf(&cache_entry), Some(src_tree));
    }

    #[test]
    fn prefer_source_tree_local_without_vmlinux_in_tree() {
        // Local-source cache entry but source tree lacks vmlinux:
        // fall back to None so caller keeps the cache-entry path.
        // The entry still records the stat pair (cache-store time
        // may have written the pair, then the user later removed the
        // source-tree vmlinux); the helper bails on the absent
        // `vmlinux` file before reaching the size/mtime check.
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        // No vmlinux in src_tree.

        let meta = KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: Some(42),
            source_vmlinux_mtime_secs: Some(1_700_000_000),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(prefer_source_tree_for_dwarf(&cache_entry), None);
    }

    #[test]
    fn prefer_source_tree_tarball_source_returns_none() {
        // Tarball source entry has no source_tree_path — return None
        // so caller uses the cache-entry directory (symbol lookup only,
        // no file:line).
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(prefer_source_tree_for_dwarf(&cache_entry), None);
    }

    #[test]
    fn prefer_source_tree_no_metadata_returns_none() {
        // Directory without metadata.json (e.g. a build-tree root, not
        // a cache entry): return None, caller keeps its existing path.
        let tmp = TempDir::new().unwrap();
        assert_eq!(prefer_source_tree_for_dwarf(tmp.path()), None);
    }

    /// Malformed `metadata.json` — present on disk but not valid
    /// [`KernelMetadata`] — must short-circuit to `None`. The
    /// `read_metadata(..).ok()?` guard in
    /// [`prefer_source_tree_for_dwarf`] converts the parse failure
    /// into `None` without bailing, so callers fall back to the
    /// cache directory for symbol-only lookup rather than having
    /// the DWARF path blow up on a corrupted entry.
    ///
    /// A regression that replaced the `.ok()?` with `.unwrap()`,
    /// `.expect(..)`, or an `anyhow::Result` propagation would
    /// break this test — preserving the "silent fallback" contract
    /// documented on the function.
    #[test]
    fn prefer_source_tree_metadata_parse_failure_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();
        // Valid JSON shape but NOT a `KernelMetadata` — missing
        // every required field. serde_json::from_str errors with
        // "missing field", which `read_metadata` maps to
        // `Err(String)`, which `prefer_source_tree_for_dwarf`'s
        // `.ok()?` turns into `None`.
        fs::write(
            cache_entry.join("metadata.json"),
            br#"{"not_kernel_metadata": true}"#,
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "malformed metadata.json must short-circuit to None, not bail",
        );

        // Completely invalid JSON (not parseable at the token level)
        // must also short-circuit. Covers serde's two distinct
        // error classes — tokenizer failure vs shape mismatch —
        // both of which map to `Err(String)` inside `read_metadata`.
        let other_entry = tmp.path().join("other");
        fs::create_dir_all(&other_entry).unwrap();
        fs::write(other_entry.join("metadata.json"), b"not json at all {{{").unwrap();
        assert_eq!(
            prefer_source_tree_for_dwarf(&other_entry),
            None,
            "unparseable metadata.json must short-circuit to None, not bail",
        );
    }

    /// Local-source cache entry whose `source_tree_path` is
    /// explicitly `None` short-circuits at the `let src_path =
    /// source_tree_path?;` line — no filesystem probe runs for the
    /// missing path. Pins the "tree location not recorded" branch
    /// documented on [`prefer_source_tree_for_dwarf`].
    ///
    /// Distinct from `prefer_source_tree_local_without_vmlinux_in_tree`:
    /// that test has `source_tree_path = Some(...)` but the
    /// filesystem lacks `vmlinux`, so the function reaches the
    /// `src_path.join("vmlinux").is_file()` check before returning
    /// None. This test short-circuits earlier, before any filesystem
    /// inspection — a regression that replaced the `?` with a
    /// `.unwrap_or_else(|| default_path)` or a fallback would break
    /// it.
    #[test]
    fn prefer_source_tree_local_with_none_source_tree_path_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: None,
                git_hash: Some("abc123".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(42),
            source_vmlinux_mtime_secs: Some(1_700_000_000),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "Local entry with source_tree_path=None must short-circuit \
             to None at the `let src_path = source_tree_path?;` line \
             — no filesystem probe must run",
        );
    }

    /// When `source_vmlinux_size` and `source_vmlinux_mtime_secs`
    /// match the on-disk vmlinux's current stat, the helper returns
    /// the source-tree path. Pins the validate-and-pass branch
    /// added for the user-rebuild-detection gate.
    #[test]
    fn prefer_source_tree_validates_matching_vmlinux_stat_and_returns_path() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf-bytes").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let meta = KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len()),
            source_vmlinux_mtime_secs: Some(mtime_secs),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            Some(src_tree),
            "matching size + mtime must pass the validation gate"
        );
    }

    /// A size mismatch (user rebuild changed the vmlinux) drops the
    /// helper to None so callers fall back to the cached stripped
    /// vmlinux. Without the gate, DWARF reads would silently route
    /// to a vmlinux whose line numbers no longer correspond to the
    /// cache's BTF.
    #[test]
    fn prefer_source_tree_size_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf-bytes").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Stored size deliberately wrong by one byte.
        let meta = KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len() + 1),
            source_vmlinux_mtime_secs: Some(mtime_secs),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "size mismatch must drop validation and return None"
        );
    }

    /// An mtime mismatch (typical user-rebuild signal — `make`
    /// touches `vmlinux` even when the resulting bytes happen to be
    /// the same length) drops to None on the same fallback path.
    #[test]
    fn prefer_source_tree_mtime_mismatch_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        let vmlinux = src_tree.join("vmlinux");
        fs::write(&vmlinux, b"fake-elf-bytes").unwrap();
        let stat = fs::metadata(&vmlinux).unwrap();
        let mtime_secs = stat
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Stored mtime deliberately offset by an hour.
        let meta = KernelMetadata {
            version: None,
            source: KernelSource::Local {
                source_tree_path: Some(src_tree),
                git_hash: None,
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: Some(stat.len()),
            source_vmlinux_mtime_secs: Some(mtime_secs - 3600),
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(
            prefer_source_tree_for_dwarf(&cache_entry),
            None,
            "mtime mismatch must drop validation and return None"
        );
    }

    // -- recover_local_source_tree --
    //
    // Sibling helper to `prefer_source_tree_for_dwarf` but with
    // distinct semantics: NO `vmlinux` presence check, NO arm for
    // tarball/git entries (returns None when metadata.source isn't
    // `KernelSource::Local`). Used by callers that just need the
    // source tree directory (git open, commit detection); DWARF
    // callers stay on `prefer_source_tree_for_dwarf` for the
    // additional vmlinux gate.

    /// Cache entry built from a local tree: helper returns the
    /// recorded `source_tree_path` regardless of whether the
    /// source tree still has a `vmlinux` on disk. The vmlinux gate
    /// is `prefer_source_tree_for_dwarf`'s concern, NOT this
    /// helper's — git tree opens and commit detection only need
    /// the directory path.
    #[test]
    fn recover_local_source_tree_local_with_path_returns_source_tree() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        let src_tree = tmp.path().join("src");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::create_dir_all(&src_tree).unwrap();
        // Deliberately omit vmlinux from the source tree so this
        // test pins the "no vmlinux gate" contract — a regression
        // that copied prefer_source_tree_for_dwarf's vmlinux check
        // would surface here as a None result.

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: Some(src_tree.clone()),
                git_hash: Some("abc1234".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: false,
            vmlinux_stripped: false,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(recover_local_source_tree(&cache_entry), Some(src_tree));
    }

    /// No `metadata.json` on disk: helper returns `None` and the
    /// caller falls back to using the input dir verbatim
    /// (cargo-ktstr stats compare's `gix::open` site treats the
    /// env value as the source tree itself; sidecar's
    /// `resolve_kernel_source_dir` Path arm does the same).
    #[test]
    fn recover_local_source_tree_no_metadata_returns_none() {
        let tmp = TempDir::new().unwrap();
        // No metadata.json planted — `tmp.path()` is a bare empty
        // tempdir, modeling the dirty-source-tree path that
        // skipped cache store.
        assert_eq!(recover_local_source_tree(tmp.path()), None);
    }

    /// `metadata.json` present but `source` is `Tarball`: helper
    /// returns `None`. Tarball entries never carry a
    /// `source_tree_path` (the extraction dir is transient);
    /// callers should not probe a tarball entry as a git repo.
    #[test]
    fn recover_local_source_tree_tarball_source_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Tarball,
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(recover_local_source_tree(&cache_entry), None);
    }

    /// `Local` source with `source_tree_path: None` (the
    /// dirty-tree case where the source-tree path was not
    /// recorded): helper returns `None`. Distinct from the
    /// no-metadata case but produces the same outcome.
    #[test]
    fn recover_local_source_tree_local_with_none_path_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();

        let meta = KernelMetadata {
            version: Some("6.14.2".to_string()),
            source: KernelSource::Local {
                source_tree_path: None,
                git_hash: Some("abc1234".to_string()),
            },
            arch: "x86_64".to_string(),
            image_name: "bzImage".to_string(),
            config_hash: None,
            built_at: "2026-04-18T10:00:00Z".to_string(),
            ktstr_kconfig_hash: None,
            extra_kconfig_hash: None,
            has_vmlinux: true,
            vmlinux_stripped: true,
            source_vmlinux_size: None,
            source_vmlinux_mtime_secs: None,
        };
        fs::write(
            cache_entry.join("metadata.json"),
            serde_json::to_string(&meta).unwrap(),
        )
        .unwrap();

        assert_eq!(recover_local_source_tree(&cache_entry), None);
    }

    /// Malformed `metadata.json`: helper short-circuits to `None`
    /// silently. Mirrors `prefer_source_tree_for_dwarf`'s contract
    /// — a corrupt cache entry must not blow up callers; they fall
    /// back to using the input dir verbatim.
    #[test]
    fn recover_local_source_tree_malformed_metadata_returns_none() {
        let tmp = TempDir::new().unwrap();
        let cache_entry = tmp.path().join("cache");
        fs::create_dir_all(&cache_entry).unwrap();
        fs::write(
            cache_entry.join("metadata.json"),
            br#"{"not_kernel_metadata": true}"#,
        )
        .unwrap();
        assert_eq!(recover_local_source_tree(&cache_entry), None);
    }

    // -- strip_vmlinux_debug --

    /// Check whether `elf` has a defined symbol with the given name.
    /// Mirrors the `sym_addr` closure inside `KernelSymbols::from_vmlinux`
    /// by requiring `st_value != 0` to reject undefined/absent symbols.
    fn has_symbol(elf: &goblin::elf::Elf, name: &str) -> bool {
        elf.syms
            .iter()
            .any(|s| s.st_value != 0 && elf.strtab.get_at(s.st_name) == Some(name))
    }

    /// Build a minimal ELF covering every strip dispatch branch:
    /// `.text` (code, bytes dropped via SHT_NOBITS), `.BTF` and
    /// `.rodata` (kept whole via the keep-list predicate), `.bss`
    /// (keep-list, already SHT_NOBITS), `.BTF.ext` + `.debug_*`
    /// (deleted), and the zero-data sections (`.data`,
    /// `.data..percpu`, `.init.data`; bytes dropped via SHT_NOBITS).
    /// Each bytes-dropped section has a symbol pointing at an
    /// in-bounds offset so tests can assert the symbols survive
    /// `Builder::delete_orphans`.
    fn create_strip_test_fixture(dir: &Path) -> PathBuf {
        use object::write;
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text — loadable code (not in keep-list, bytes dropped by keep-list path).
        let text_id = obj.add_section(Vec::new(), b".text".to_vec(), object::SectionKind::Text);
        obj.append_section_data(text_id, &[0xCC; 64], 1);
        // Symbol so .symtab and .strtab are generated.
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
        // .BTF — kept by both keep-list and fallback.
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Metadata);
        obj.append_section_data(btf_id, &[0xEB; 256], 1);
        // .rodata — kept by keep-list (IKCONFIG gzip blob at runtime).
        // Bytes are preserved verbatim so read_hz_from_ikconfig can scan
        // for the IKCFG_ST marker; fixture stores an opaque payload.
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xCA; 512], 1);
        // .bss — kept by keep-list; already SHT_NOBITS on any real
        // kernel build. object::write emits it without backing bytes
        // when `kind = UninitializedData`, matching real-vmlinux layout.
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
        // .BTF.ext — deleted by keep-list (no consumer).
        let btf_ext_id = obj.add_section(
            Vec::new(),
            b".BTF.ext".to_vec(),
            object::SectionKind::Metadata,
        );
        obj.append_section_data(btf_ext_id, &[0xE1; 128], 1);
        // .debug_info — always stripped.
        let debug_id = obj.add_section(
            Vec::new(),
            b".debug_info".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_id, &[0xAA; 4096], 1);
        // .debug_str — always stripped.
        let debug_str_id = obj.add_section(
            Vec::new(),
            b".debug_str".to_vec(),
            object::SectionKind::Debug,
        );
        obj.append_section_data(debug_str_id, &[0xBB; 2048], 1);
        // .data — bytes dropped via SHT_NOBITS; symbol must survive.
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
        // .data..percpu — bytes dropped via SHT_NOBITS; symbol must survive.
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
        // .init.data — bytes dropped via SHT_NOBITS; symbol must survive.
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

        // Positive control: the fixture must actually carry the
        // sections this test asserts on. If object::write silently
        // renames or drops one, the post-strip absence assertions
        // would false-pass.
        let source_data = fs::read(&vmlinux).unwrap();
        let source_elf = goblin::elf::Elf::parse(&source_data).unwrap();
        let source_section_names: Vec<&str> = source_elf
            .section_headers
            .iter()
            .filter_map(|s| source_elf.shdr_strtab.get_at(s.sh_name))
            .collect();
        // Positive control covers every section the post-strip
        // assertions inspect — kept, dropped, or deleted. A future
        // fixture regression that silently omits any of these would
        // make the corresponding post-strip check vacuous.
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
        // Debug sections removed.
        assert!(
            !section_names.contains(&".debug_info"),
            "should not contain .debug_info"
        );
        assert!(
            !section_names.contains(&".debug_str"),
            "should not contain .debug_str"
        );
        // .BTF.ext removed (no consumer).
        assert!(
            !section_names.contains(&".BTF.ext"),
            "should not contain .BTF.ext"
        );
        // Keep-list sections preserved (names from all three consumer
        // modules plus structural).
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

        // Smoke check: stripping a synthetic ELF produces a readable
        // symbol table whose strtab still contains our test symbol
        // names. End-to-end symbol preservation on real vmlinuxes is
        // covered by the *_preserves_monitor_symbols tests below.
        assert!(
            has_symbol(&elf, "test_text_symbol"),
            "stripped ELF should contain test_text_symbol in symtab"
        );
        // test_bss_symbol anchors .bss against Builder::delete_orphans.
        // Queryable via has_symbol because its fixture st_value is
        // nonzero (an in-bounds offset within .bss).
        assert!(
            has_symbol(&elf, "test_bss_symbol"),
            "stripped ELF should contain test_bss_symbol in symtab"
        );
    }

    /// Data sections matched by [`is_zero_data_section`] and code
    /// sections must come out as SHT_NOBITS with sh_size == 0, and
    /// symbols pointing at them must survive. Runs on the synthetic
    /// fixture so it exercises the keep-list path in CI environments
    /// without a real vmlinux.
    #[test]
    fn strip_vmlinux_debug_zeros_data_sections() {
        let src = TempDir::new().unwrap();
        let vmlinux = create_strip_test_fixture(src.path());

        // Pre-strip positive control: every zero-data section must
        // start with a non-SHT_NOBITS type AND non-zero sh_size. If
        // the fixture ever emits them as empty or already-SHT_NOBITS,
        // the post-strip assertions below become tautological and
        // would pass even if the strip pipeline regressed.
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

        // Iterate both the consumer-declared zero-data sections and
        // the speculative retention set so the test stays in sync
        // automatically when either source changes.
        for name_bytes in crate::monitor::symbols::VMLINUX_ZERO_DATA_SECTIONS
            .iter()
            .chain(SPECULATIVE_ZERO_DATA_SECTIONS.iter())
        {
            let name = std::str::from_utf8(name_bytes).unwrap();
            assert_nobits_empty(name);
        }

        // Code sections (`.text` in the fixture) receive the same
        // SHT_NOBITS treatment so function symbols survive
        // `Builder::delete_orphans`.
        assert_nobits_empty(".text");

        // Symbols pointing at the zeroed data sections must survive.
        // Fixture symbol values are nonzero (0x20/0x30/0x40, within
        // their section bounds), so has_symbol's st_value != 0 filter
        // matches them.
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

    /// `strip_debug_prefix` is the fallback path `strip_vmlinux_debug`
    /// hits when the keep-list strip errors out. Exercise it
    /// directly on the synthetic fixture so the success path has
    /// coverage independent of the keep-list branch.
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

        // .debug_* sections deleted. The fallback also removes the
        // `.comment` section, but this fixture does not emit one, so
        // that branch of the delete set is not exercised here.
        assert!(
            !names.contains(&".debug_info"),
            "fallback should remove .debug_info"
        );
        assert!(
            !names.contains(&".debug_str"),
            "fallback should remove .debug_str"
        );
        // Every other section the fixture carries survives — unlike
        // the keep-list path, the fallback does not partition by
        // consumer. In particular `.BTF.ext` (which keep-list would
        // delete) remains.
        for name in [".BTF", ".BTF.ext", ".text", ".data", ".rodata", ".symtab"] {
            assert!(
                names.contains(&name),
                "fallback must preserve {name}, got sections {names:?}"
            );
        }
    }

    /// `strip_debug_prefix`'s delete filter matches on six predicates:
    /// `name.starts_with(b".debug_")`, `name == b".comment"`,
    /// `name.starts_with(b".rela.")`, `name.starts_with(b".rel.")`,
    /// `name.starts_with(b".relr.")`, and `name.starts_with(b".crel.")`.
    /// The `.debug_*` branch is exercised by
    /// `strip_debug_prefix_removes_debug_and_preserves_rest` against
    /// the shared fixture; the four reloc-name prefix arms are
    /// exercised by `strip_debug_prefix_removes_reloc_prefix_sections`.
    /// This test covers the `.comment` branch against a focused
    /// fixture that specifically emits one — the shared fixture
    /// deliberately does not, to keep the keep-list assertions
    /// scoped.
    #[test]
    fn strip_debug_prefix_removes_dot_comment() {
        use object::write;
        // Minimal ELF: one loadable .text (fallback must preserve it)
        // plus a .comment section (fallback must delete it). A symbol
        // anchors .text so the `object` writer emits .symtab/.strtab.
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
        // `.comment` is ELF's standard toolchain-producer string table
        // (`object::SectionKind::OtherString`).
        // Real kernel builds carry one stamped by GCC/Clang.
        let comment_id = obj.add_section(
            Vec::new(),
            b".comment".to_vec(),
            object::SectionKind::OtherString,
        );
        obj.append_section_data(comment_id, b"GCC: (GNU) 14.2.1 20250207\0", 1);
        let data = obj.write().unwrap();

        // Positive control: the fixture must actually carry `.comment`
        // and `.text` before strip. If `object::write` silently dropped
        // either (e.g. renaming, or treating OtherString non-standardly),
        // the post-strip absence assertion on `.comment` would
        // false-pass. Mirrors the positive-control pattern in
        // `strip_vmlinux_debug_applies_keep_list`.
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

        // `neutralize_relocs` is a no-op on this fixture (no
        // SHF_ALLOC relocation sections) — run it anyway so the test
        // exercises the exact input pipeline `strip_vmlinux_debug` uses.
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
        // Non-comment, non-debug sections survive untouched — guards
        // against an overly broad filter that accidentally drops
        // unrelated sections.
        assert!(
            names.contains(&".text"),
            "fallback must preserve .text, got sections {names:?}"
        );
    }

    /// `strip_debug_prefix` deletes reloc-named sections via the
    /// `.rela.`, `.rel.`, `.relr.`, and `.crel.` prefix arms so the
    /// fallback output doesn't carry the zero-size ghost headers that
    /// `neutralize_relocs` left behind. Exercise each prefix on a
    /// focused fixture — a real kernel vmlinux might carry only a
    /// subset, so the synthetic shape pins every arm.
    #[test]
    fn strip_debug_prefix_removes_reloc_prefix_sections() {
        use object::elf::{SHT_REL, SHT_RELA, SHT_RELR};

        // Base ELF with .text + anchor symbol so `.symtab`/`.strtab`
        // are present.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        // One section per reloc-prefix arm. `.crel.*` uses SHT_CREL
        // below. Each carries a nonzero payload so `neutralize_relocs`
        // has observable work to do before the fallback runs.
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

        // Positive control: every reloc-named section must exist
        // pre-strip; a silent rename by `object::write` would false-pass
        // the post-strip absence assertions below.
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

        // All four reloc-prefix arms deleted.
        for name in [".rela.text", ".rel.data", ".relr.dyn", ".crel.text"] {
            assert!(
                !names.contains(&name),
                "fallback must delete {name} (prefix arm), got sections {names:?}"
            );
        }
        // Non-reloc section survives — guards against an overly broad
        // filter that would drop e.g. `.text` on an unrelated name prefix.
        assert!(
            names.contains(&".text"),
            "fallback must preserve .text, got sections {names:?}"
        );
    }

    /// `neutralize_relocs` rewrites two section-header fields on every
    /// section whose `sh_type` is `SHT_REL`, `SHT_RELA`, `SHT_RELR`, or
    /// `SHT_CREL`, regardless of the `SHF_ALLOC` flag: `sh_type`
    /// becomes `SHT_PROGBITS` and `sh_size` becomes 0. Pin the
    /// observable invariants against a focused fixture:
    ///
    /// 1a. SHF_ALLOC + SHT_RELA section has sh_type rewritten to
    ///     SHT_PROGBITS and sh_size zeroed post-call.
    /// 1b. SHF_ALLOC + SHT_REL section has sh_type rewritten to
    ///     SHT_PROGBITS and sh_size zeroed post-call.
    /// 1c. Non-ALLOC SHT_RELA section has sh_type rewritten to
    ///     SHT_PROGBITS and sh_size zeroed post-call (the SHF_ALLOC
    ///     gate was dropped — aarch64 kernels emit non-alloc rela
    ///     sections whose byte ranges trip
    ///     `object::build::elf::Builder::read`).
    /// 1d. SHT_RELR section has sh_type rewritten to SHT_PROGBITS and
    ///     sh_size zeroed post-call (defense-in-depth for arm64
    ///     kernels with `CONFIG_PIE` + `CONFIG_RELR` that emit
    ///     `.relr.dyn`).
    /// 2. Non-RELA section (e.g. `.text`) has sh_type and sh_size
    ///    preserved (guards against an accidentally-broader filter).
    ///
    /// Also pins content preservation: `neutralize_relocs` only
    /// mutates the section HEADER's `sh_type` and `sh_size`, not the
    /// section's data bytes. Raw bytes at the original sh_offset must
    /// remain bit-identical post-call.
    #[test]
    fn neutralize_relocs_zeros_sh_size_of_every_reloc_section() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA, SHT_RELR};

        // Base ELF with .text + anchor symbol (so object::write
        // emits .symtab/.strtab). Reloc sections are added below.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        // .rela.kaslr — SHT_RELA + SHF_ALLOC. Shape matches what
        // CONFIG_RELOCATABLE + CONFIG_RANDOMIZE_BASE kernels emit.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. The fn's match arm accepts
        // both SHT_REL and SHT_RELA; exercising only SHT_RELA would
        // let a regression that dropped SHT_REL ride unnoticed.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA WITHOUT SHF_ALLOC. After the
        // SHF_ALLOC gate was dropped, this must also be zeroed — a
        // regression that re-added the gate would preserve sh_size
        // here and fail the new invariant 1c.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        // .relr.dyn — SHT_RELR + SHF_ALLOC. Defense-in-depth for
        // arm64 kernels that emit packed relative relocations.
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

        // Positive-control the fixture: the five sections we assert on
        // must actually exist in the produced ELF with the expected
        // sh_type/sh_flags/sh_size. If `object::write` renamed or
        // reshaped one, the post-call assertions would false-pass.
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
        assert_eq!(
            pre_kaslr.sh_type,
            SHT_RELA,
            ".rela.kaslr sh_type must be SHT_RELA; got sh_type={} ({})",
            pre_kaslr.sh_type,
            sh_type_name(pre_kaslr.sh_type),
        );
        assert!(
            pre_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rela.kaslr must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_kaslr.sh_flags
        );
        assert_eq!(
            pre_kaslr.sh_size, 32,
            ".rela.kaslr sh_size must match 32-byte payload"
        );
        assert_eq!(
            pre_rel.sh_type,
            SHT_REL,
            ".rel.foo sh_type must be SHT_REL; got sh_type={} ({})",
            pre_rel.sh_type,
            sh_type_name(pre_rel.sh_type),
        );
        assert!(
            pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rel.sh_flags
        );
        assert_eq!(
            pre_rel.sh_size, 24,
            ".rel.foo sh_size must match 24-byte payload"
        );
        assert_eq!(
            pre_rdbg.sh_type,
            SHT_RELA,
            ".rela.debug_info sh_type must be SHT_RELA; got sh_type={} ({})",
            pre_rdbg.sh_type,
            sh_type_name(pre_rdbg.sh_type),
        );
        assert_eq!(
            pre_rdbg.sh_flags & u64::from(SHF_ALLOC),
            0,
            ".rela.debug_info must NOT carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rdbg.sh_flags
        );
        assert_eq!(
            pre_rdbg.sh_size, 16,
            ".rela.debug_info sh_size must match 16-byte payload"
        );
        assert_eq!(
            pre_relr.sh_type,
            SHT_RELR,
            ".relr.dyn sh_type must be SHT_RELR (19); got sh_type={} ({})",
            pre_relr.sh_type,
            sh_type_name(pre_relr.sh_type),
        );
        assert_eq!(
            pre_relr.sh_size, 24,
            ".relr.dyn sh_size must match 24-byte payload"
        );
        assert_eq!(
            pre_text.sh_size, 64,
            ".text sh_size must match 64-byte payload"
        );

        // Snapshot the .rela.kaslr data bytes before the call so we
        // can assert they survive the sh_size rewrite.
        let kaslr_offset = pre_kaslr.sh_offset as usize;
        let kaslr_size = pre_kaslr.sh_size as usize;
        let kaslr_original_data = data[kaslr_offset..kaslr_offset + kaslr_size].to_vec();

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed.len(),
            data.len(),
            "neutralize_relocs must not resize the ELF; only sh_size header fields are rewritten"
        );

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

        // Invariant 1a: SHF_ALLOC + SHT_RELA section has sh_size zeroed.
        assert_eq!(
            post_kaslr.sh_size, 0,
            ".rela.kaslr sh_size must be zeroed; got {}",
            post_kaslr.sh_size
        );
        // Invariant 1b: SHF_ALLOC + SHT_REL section has sh_size zeroed
        // (the SHT_REL arm of the filter).
        assert_eq!(
            post_rel.sh_size, 0,
            ".rel.foo sh_size must be zeroed; got {}",
            post_rel.sh_size
        );
        // Invariant 1c: Non-ALLOC SHT_RELA section ALSO zeroed (the
        // SHF_ALLOC gate was dropped so aarch64 non-alloc rela
        // sections get neutralized).
        assert_eq!(
            post_rdbg.sh_size, 0,
            ".rela.debug_info sh_size must be zeroed (SHF_ALLOC gate dropped); got {}",
            post_rdbg.sh_size
        );
        // Invariant 1d: SHT_RELR section zeroed (defense-in-depth
        // for arm64 CONFIG_RELR kernels).
        assert_eq!(
            post_relr.sh_size, 0,
            ".relr.dyn sh_size must be zeroed (SHT_RELR match arm); got {}",
            post_relr.sh_size
        );
        // Invariant 2: Non-RELA section preserved.
        assert_eq!(
            post_text.sh_size, pre_text.sh_size,
            ".text sh_size must be preserved (not a relocation section)"
        );

        // Content preservation: the raw bytes at the section's
        // sh_offset must be bit-identical to pre-call. Only the
        // sh_size header field was rewritten.
        assert_eq!(
            &processed[kaslr_offset..kaslr_offset + kaslr_size],
            &kaslr_original_data[..],
            ".rela.kaslr data bytes must be preserved; neutralize only rewrites sh_size"
        );

        // sh_offset and sh_flags are preserved; sh_type is rewritten
        // to SHT_PROGBITS so the Builder reads the section via the
        // opaque-data arm instead of the rel/rela parse arms that
        // break on zero-length slices with align != 1.
        assert_eq!(
            post_kaslr.sh_offset, pre_kaslr.sh_offset,
            "sh_offset must be preserved"
        );
        assert_eq!(
            post_kaslr.sh_type,
            object::elf::SHT_PROGBITS,
            "sh_type must be rewritten to SHT_PROGBITS; got sh_type={} ({})",
            post_kaslr.sh_type,
            sh_type_name(post_kaslr.sh_type),
        );
        assert_eq!(
            post_kaslr.sh_flags, pre_kaslr.sh_flags,
            "sh_flags must be preserved"
        );
        // The sibling reloc sections should also be re-typed to
        // SHT_PROGBITS (the fn applies sh_type rewrite to every
        // matching section, not just the first).
        assert_eq!(
            post_rel.sh_type,
            object::elf::SHT_PROGBITS,
            ".rel.foo sh_type must be SHT_PROGBITS"
        );
        assert_eq!(
            post_rdbg.sh_type,
            object::elf::SHT_PROGBITS,
            ".rela.debug_info sh_type must be SHT_PROGBITS"
        );
        assert_eq!(
            post_relr.sh_type,
            object::elf::SHT_PROGBITS,
            ".relr.dyn sh_type must be SHT_PROGBITS"
        );
    }

    /// For ELFs that carry no relocation sections at all,
    /// `neutralize_relocs` returns an unchanged copy —
    /// documented as the "no-op" branch in the fn docstring.
    #[test]
    fn neutralize_relocs_noop_when_no_reloc_sections() {
        // Base ELF carries only .text + anchor symbol — no reloc
        // sections at all, so the filter matches nothing.
        let data = build_base_elf_with_text_symbol(object::Architecture::X86_64)
            .write()
            .unwrap();

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed, data,
            "neutralize_relocs must be a byte-identity no-op when no reloc sections are present"
        );
    }

    /// `neutralize_relocs` must be byte-identity idempotent:
    /// `f(f(x)) == f(x)`. The production filter inside
    /// [`neutralize_relocs`] keys on `sh_type` — which IS rewritten
    /// (to `SHT_PROGBITS`) on matching sections. Idempotence still
    /// holds because after the first pass the neutralized sections
    /// no longer match the `is_reloc` predicate (sh_type is now
    /// `SHT_PROGBITS`, not one of `SHT_REL`/`SHT_RELA`/`SHT_RELR`/
    /// `SHT_CREL`), so the second pass walks every section without
    /// touching any header field and the output is byte-identical to
    /// the first-pass output.
    ///
    /// Guards against a future mutation that rewrites sh_type to a
    /// still-matched value (e.g. flipping `SHT_REL` to `SHT_RELA` —
    /// both match `is_reloc`, which would make the second pass
    /// re-neutralize to yet another sh_type value and break
    /// idempotence).
    ///
    /// Uses the same multi-section fixture as
    /// `neutralize_relocs_zeros_sh_size_of_every_reloc_section`
    /// so every reloc-type arm of the filter (SHT_RELA with and
    /// without SHF_ALLOC, SHT_REL) and the non-RELA negative control
    /// re-walk on the second pass.
    #[test]
    fn neutralize_relocs_is_idempotent() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        // Base .text + anchor symbol; the reloc sections added below
        // intentionally mirror the sibling zeros-every-reloc test's
        // fixture so the filter re-walks on the second pass.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::X86_64);
        // .rela.kaslr — SHT_RELA + SHF_ALLOC.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 32], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Exercises the SHT_REL arm
        // of the filter so a regression that special-cased only
        // SHT_RELA on re-entry would surface here.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 24], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA without SHF_ALLOC. After the
        // SHF_ALLOC gate was dropped, this gets neutralized too —
        // but must re-neutralize to byte-identical bytes on the
        // second pass.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 16], 1);
        // flags left as SectionFlags::None — no SHF_ALLOC.

        let data = obj.write().unwrap();

        let first_pass = neutralize_relocs(&data).unwrap();
        let second_pass = neutralize_relocs(&first_pass).unwrap();

        // Non-vacuous guard: the first call must actually modify bytes
        // on this fixture (which carries reloc sections); a degenerate
        // no-op implementation of `neutralize_relocs` would
        // trivially satisfy idempotence and must not pass.
        assert_ne!(
            first_pass, data,
            "first call must modify bytes on a fixture with reloc sections; \
             if this fails, neutralize_relocs is a no-op"
        );

        // Primary idempotence assertion: byte equality between passes.
        assert_eq!(
            second_pass, first_pass,
            "neutralize_relocs must be idempotent: a second pass over its own output produces byte-identical bytes"
        );

        // Length preservation across both passes — the function only
        // rewrites in-place `sh_size` fields, never resizes the buffer.
        assert_eq!(
            first_pass.len(),
            data.len(),
            "first pass must preserve ELF length"
        );
        assert_eq!(
            second_pass.len(),
            first_pass.len(),
            "second pass must preserve ELF length"
        );

        // Re-parse post-second-pass: the ELF header and section
        // header table must still be well-formed after two rewrites.
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

        // All reloc sections stay zeroed on the second pass,
        // regardless of SHF_ALLOC.
        assert_eq!(
            post_kaslr.sh_size, 0,
            ".rela.kaslr sh_size must remain zero after the second pass"
        );
        assert_eq!(
            post_rel.sh_size, 0,
            ".rel.foo sh_size must remain zero after the second pass"
        );
        assert_eq!(
            post_rdbg.sh_size, 0,
            ".rela.debug_info sh_size must remain zero after the second pass (SHF_ALLOC gate dropped)"
        );

        // SHF_ALLOC flag must still be set on the ALLOC sections —
        // the function touches sh_type and sh_size, never sh_flags.
        // The non-ALLOC `.rela.debug_info` likewise retains its
        // (cleared) SHF_ALLOC bit.
        assert!(
            post_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rela.kaslr SHF_ALLOC flag must survive both passes; got sh_flags={:#x}",
            post_kaslr.sh_flags
        );
        assert!(
            post_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo SHF_ALLOC flag must survive both passes; got sh_flags={:#x}",
            post_rel.sh_flags
        );
        assert_eq!(
            post_rdbg.sh_flags & u64::from(SHF_ALLOC),
            0,
            ".rela.debug_info must retain its (cleared) SHF_ALLOC flag across both passes; got sh_flags={:#x}",
            post_rdbg.sh_flags
        );
    }

    /// `neutralize_relocs` fails loudly when fed bytes that do
    /// not parse as an ELF — the goblin parse returns Err and the
    /// function wraps it in an `anyhow::anyhow!("parse vmlinux ELF
    /// for preprocess: {e}")`. Pin only the stable "parse vmlinux ELF
    /// for preprocess" wrapper in `neutralize_relocs`; the
    /// goblin-side error text is version-dependent and not part of
    /// the contract.
    ///
    /// Exercises two distinct goblin failure paths through the same
    /// anyhow wrapper:
    ///
    /// 1. Bad magic: "not an ELF..." passes the 16-byte length gate but
    ///    its first four bytes do not match `\x7fELF`, so goblin's
    ///    `TryFromCtx` for `Header` bails with `Error::BadMagic` before
    ///    inspecting any later field (see goblin 0.10 `elf/header.rs`
    ///    `try_from_ctx` at the `ident[0..SELFMAG] != ELFMAG` branch).
    /// 2. Invalid EI_CLASS: a 16-byte prefix with the correct magic but
    ///    `ident[EI_CLASS] == 0` (ELFCLASSNONE) passes BOTH the length
    ///    and magic gates, and fails on the subsequent class-dispatch
    ///    match with `Error::Malformed("invalid ELF class 0")`. This is
    ///    the "passes magic, fails deeper" path.
    ///
    /// Either failure mode flows through the same `anyhow::anyhow!`
    /// wrapper, so the test pins the wrapper string for each input
    /// without pinning the goblin-side sub-error wording.
    #[test]
    fn neutralize_relocs_rejects_invalid_elf() {
        // Table-driven so a future goblin upgrade that changes either
        // sub-error's wording still surfaces both paths distinctly.
        let cases: &[(&str, &[u8])] = &[
            ("bad magic", b"not an ELF at all, just some bytes"),
            (
                "magic ok but invalid EI_CLASS",
                &[
                    0x7f, b'E', b'L', b'F', // magic
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // ident[4..16]: class=0, rest 0
                ],
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

    /// ELF32 counterpart of
    /// [`neutralize_relocs_zeros_sh_size_of_every_reloc_section`].
    ///
    /// `neutralize_relocs` dispatches on `elf.is_64` at the
    /// `sh_size` offset/width pair — 32-byte offset + 8-byte field for
    /// ELF64, 20-byte offset + 4-byte field for ELF32 (per the
    /// ELF32/ELF64 section header layouts documented at the call site).
    /// The existing fixture-driven coverage is all ELF64 (Architecture::
    /// X86_64), so a regression that swapped the `if elf.is_64` branches
    /// or hardcoded the 64-bit offsets would silently corrupt 32-bit
    /// inputs without tripping any assertion.
    ///
    /// Uses `Architecture::I386` which `object` maps to
    /// `address_size == U32` and therefore emits ELFCLASS32 via the
    /// writer's is_64=false path. goblin then parses the output with
    /// `is_64 == false`, driving `neutralize_relocs` through the
    /// else `(20, 4)` branch.
    ///
    /// Exercises BOTH arms of the
    /// `sh.sh_type == SHT_RELA || sh.sh_type == SHT_REL` filter
    /// predicate on the ELF32 code path. A regression that special-
    /// cased SHT_REL only on ELF64 (e.g. wired it to a 64-bit offset
    /// table), or dropped SHT_REL from the ELF32 filter altogether,
    /// would leave one section un-zeroed here.
    ///
    /// Invariants pinned: SHF_ALLOC + SHT_RELA AND SHF_ALLOC + SHT_REL
    /// both have their sh_size zeroed, the output remains parseable
    /// as ELF32 (is_64 stays false), and the buffer length is
    /// preserved (the fn only rewrites an in-place header field,
    /// never resizes).
    #[test]
    fn neutralize_relocs_zeros_sh_size_in_elf32_fixture() {
        use object::elf::{SHF_ALLOC, SHT_REL, SHT_RELA};

        // Base shape — .text + test_text_symbol with ELF32-sized
        // anchor — is shared with the ELF64 fixtures via the
        // helper. Passing `Architecture::I386` flips the ELF class
        // (is_64 false) and downgrades the symbol size to 4 bytes.
        let mut obj = build_base_elf_with_text_symbol(object::Architecture::I386);
        // .rela.kaslr — SHT_RELA + SHF_ALLOC. A 16-byte payload is
        // large enough that a mis-targeted 4-byte write at offset 20
        // (the correct ELF32 sh_size location) is observable via
        // goblin's post-parse sh_size read.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 16], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Exercises the `SHT_REL` arm
        // of the `is_reloc` match on the ELF32 code path. A regression
        // that dropped SHT_REL from the filter on the 32-bit path
        // would leave this section's sh_size unchanged and trip the
        // post-call assertion below.
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

        // Positive-control the fixture: the output must actually be
        // ELF32 (is_64 == false) — otherwise this test would false-pass
        // through the ELF64 code path the sibling test already covers.
        // A future object-crate change that remapped I386 to ELF64
        // would surface here rather than silently duplicating existing
        // coverage.
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
        assert_eq!(
            pre_kaslr.sh_type,
            SHT_RELA,
            ".rela.kaslr sh_type must be SHT_RELA; got sh_type={} ({})",
            pre_kaslr.sh_type,
            sh_type_name(pre_kaslr.sh_type),
        );
        assert!(
            pre_kaslr.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rela.kaslr must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_kaslr.sh_flags
        );
        assert_eq!(
            pre_kaslr.sh_size, 16,
            ".rela.kaslr sh_size must match 16-byte payload pre-call"
        );
        assert_eq!(
            pre_rel.sh_type,
            SHT_REL,
            ".rel.foo sh_type must be SHT_REL; got sh_type={} ({})",
            pre_rel.sh_type,
            sh_type_name(pre_rel.sh_type),
        );
        assert!(
            pre_rel.sh_flags & u64::from(SHF_ALLOC) != 0,
            ".rel.foo must carry SHF_ALLOC; got sh_flags={:#x}",
            pre_rel.sh_flags
        );
        assert_eq!(
            pre_rel.sh_size, 12,
            ".rel.foo sh_size must match 12-byte payload pre-call"
        );

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed.len(),
            data.len(),
            "neutralize_relocs must not resize the ELF32 buffer"
        );

        let post_elf = goblin::elf::Elf::parse(&processed).unwrap();
        assert!(
            !post_elf.is_64,
            "post-call parse must still be ELF32; the fn must not alter the e_ident class byte"
        );
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
        // Primary invariants: sh_size is zeroed in the ELF32 4-byte
        // slot at offset 20 within both section header entries — one
        // per arm of the SHT_RELA || SHT_REL filter predicate.
        assert_eq!(
            post_kaslr.sh_size, 0,
            "ELF32 .rela.kaslr sh_size must be zeroed (SHT_RELA arm); got {}",
            post_kaslr.sh_size
        );
        assert_eq!(
            post_rel.sh_size, 0,
            "ELF32 .rel.foo sh_size must be zeroed (SHT_REL arm); got {}",
            post_rel.sh_size
        );
    }

    /// ELF32 counterpart of
    /// [`neutralize_relocs_noop_when_no_reloc_sections`].
    ///
    /// When the input carries no reloc sections, the ELF32 code path
    /// in `neutralize_relocs` must return a byte-identity copy
    /// of the input — same invariant as ELF64, but exercised through
    /// the `(20, 4)` offset/width branch. A regression that filled
    /// zeros even on the "no match" path, or mis-read the section
    /// header count / size on 32-bit inputs, would break byte-identity
    /// here without tripping the ELF64 sibling test.
    #[test]
    fn neutralize_relocs_noop_when_no_reloc_sections_elf32() {
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::I386,
            object::Endianness::Little,
        );
        // .text + symbol mirror the sibling fixture so object::write
        // emits a valid ELF32 with the same structural sections but
        // zero reloc entries.
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

        // Positive-control: the fixture must parse as ELF32
        // (is_64 == false) so the no-match path through the
        // `(20, 4)` branch is what gets exercised. A future object
        // change that remapped I386 to ELF64 would turn this into a
        // duplicate of the ELF64 sibling without visible failure.
        let pre_elf = goblin::elf::Elf::parse(&data).unwrap();
        assert!(
            !pre_elf.is_64,
            "fixture must produce ELF32 (is_64 == false) to exercise the (20, 4) branch",
        );

        let processed = neutralize_relocs(&data).unwrap();
        assert_eq!(
            processed, data,
            "neutralize_relocs must be byte-identity on ELF32 when no reloc sections are present",
        );
    }

    /// ELF32 counterpart of
    /// [`neutralize_relocs_is_idempotent`].
    ///
    /// Idempotence (`f(f(x)) == f(x)`) must hold through the ELF32
    /// `(20, 4)` branch of `neutralize_relocs`. The ELF64
    /// sibling covers the `(32, 8)` branch; pinning both prevents a
    /// future offset-width mismatch where e.g. the second pass on
    /// ELF32 reads sh_size through an ELF64 offset and silently
    /// tripped idempotence on 32-bit inputs.
    ///
    /// Uses the same SHT_RELA+ALLOC / SHT_REL+ALLOC / SHT_RELA-no-
    /// ALLOC section mix as the ELF32 zeros fixture so the SHT_REL
    /// and SHT_RELA arms of the `is_reloc` match re-walk on the
    /// second pass. A no-match section is present to rule out a
    /// degenerate "zero every sh_size" implementation.
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
        // .rela.kaslr — SHT_RELA + SHF_ALLOC.
        let kaslr_id = obj.add_section(
            Vec::new(),
            b".rela.kaslr".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(kaslr_id, &[0xA5; 16], 1);
        obj.section_mut(kaslr_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rel.foo — SHT_REL + SHF_ALLOC. Second filter arm.
        let rel_id = obj.add_section(
            Vec::new(),
            b".rel.foo".to_vec(),
            object::SectionKind::Elf(SHT_REL),
        );
        obj.append_section_data(rel_id, &[0xC7; 12], 1);
        obj.section_mut(rel_id).flags = object::SectionFlags::Elf {
            sh_flags: u64::from(SHF_ALLOC),
        };
        // .rela.debug_info — SHT_RELA without SHF_ALLOC. The
        // SHF_ALLOC gate was dropped, so this also gets neutralized
        // on both passes — a regression that re-added the gate would
        // leave sh_size preserved here but still satisfy idempotence,
        // so the post-second-pass assertions below pin the neutralized
        // value directly.
        let rdbg_id = obj.add_section(
            Vec::new(),
            b".rela.debug_info".to_vec(),
            object::SectionKind::Elf(SHT_RELA),
        );
        obj.append_section_data(rdbg_id, &[0xB6; 8], 1);

        let data = obj.write().unwrap();

        // Positive-control ELF32: any post-parse assertion depends on
        // this; a silent promotion to ELF64 would make the idempotence
        // check run through the (32, 8) branch instead.
        assert!(
            !goblin::elf::Elf::parse(&data).unwrap().is_64,
            "fixture must be ELF32 to exercise the (20, 4) idempotence path",
        );

        let first_pass = neutralize_relocs(&data).unwrap();
        let second_pass = neutralize_relocs(&first_pass).unwrap();

        // Non-vacuous guard: first pass must actually rewrite bytes on
        // this fixture. Without this the test could false-pass on a
        // degenerate no-op implementation that trivially satisfies
        // idempotence.
        assert_ne!(
            first_pass, data,
            "first pass must rewrite sh_size on ELF32 reloc sections",
        );
        assert_eq!(
            second_pass, first_pass,
            "neutralize_relocs must be byte-identity idempotent on ELF32",
        );

        // Pin the post-second-pass sh_size values directly so a
        // regression that re-added the SHF_ALLOC gate (leaving
        // `.rela.debug_info` un-zeroed) surfaces even though
        // idempotence alone would still hold.
        let post_elf = goblin::elf::Elf::parse(&second_pass).unwrap();
        for name in [".rela.kaslr", ".rel.foo", ".rela.debug_info"] {
            let sh = post_elf
                .section_headers
                .iter()
                .find(|sh| post_elf.shdr_strtab.get_at(sh.sh_name) == Some(name))
                .unwrap_or_else(|| panic!("{name} must survive second pass"));
            assert_eq!(
                sh.sh_size, 0,
                "ELF32 {name} sh_size must be zeroed after both passes (SHF_ALLOC gate dropped)"
            );
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
    /// shape (keep-list sections, code, debug, data sidecars, and a
    /// symtab anchor) plus one extra section provided by the caller.
    /// Returns the path.
    ///
    /// The helper is generic over the per-test extra section so each
    /// of the four end-to-end pipeline tests can focus on one failure
    /// mode (non-alloc SHT_RELA with invalid entries, non-alloc
    /// SHT_RELA with sh_size past EOF, SHT_RELR with sh_size past
    /// EOF) while sharing the rest of the fixture shape.
    ///
    /// The `mutate_header` closure receives a goblin-parsed view of
    /// the produced ELF plus a mutable byte buffer and can rewrite
    /// the section header of the extra section in-place. Tests use it
    /// to push `sh_size` or `sh_offset` past the file end — a direct
    /// rewrite is safer than trying to coax `object::write` into
    /// emitting malformed headers.
    fn build_reloc_fixture(
        dir: &Path,
        extra_section_name: &[u8],
        extra_section_sh_type: u32,
        extra_section_data: &[u8],
        mutate_header: impl FnOnce(&mut [u8]),
    ) -> PathBuf {
        use object::write;

        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text anchors the symtab.
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
        // .BTF — kept by probe BTF keep-list.
        let btf_id = obj.add_section(Vec::new(), b".BTF".to_vec(), object::SectionKind::Other);
        obj.append_section_data(btf_id, &[0x42; 128], 1);
        // .rodata — kept by monitor CONFIG_IKCONFIG keep-list.
        let rodata_id = obj.add_section(
            Vec::new(),
            b".rodata".to_vec(),
            object::SectionKind::ReadOnlyData,
        );
        obj.append_section_data(rodata_id, &[0xAA; 256], 1);
        // Extra caller-provided section (a reloc section in the tests
        // below). Flags left as SectionFlags::None so it is non-alloc
        // — exercising the SHF_ALLOC-gate-drop path.
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

    /// Assert a successful [`strip_vmlinux_debug`] run on the fixture
    /// preserves the keep-list sections and deletes the extra
    /// (reloc-name) section via `strip_keep_list`'s name-based policy.
    ///
    /// This is the shared oracle for the four end-to-end pipeline
    /// tests: every variant that `strip_vmlinux_debug` accepts must
    /// yield the same output shape — keep-list sections present,
    /// reloc-name section absent.
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

    /// Pipeline pin #1: strip_vmlinux_debug handles a non-ALLOC
    /// `SHT_RELA` section with VALID byte range but entries whose
    /// `r_info` symbol indices are garbage (`0xA5A5...`).
    ///
    /// Before the SHF_ALLOC gate was dropped from
    /// [`neutralize_relocs`], non-ALLOC reloc sections were skipped,
    /// and `object::build::elf::Builder::read` then called
    /// `section.rela()` → `data_as_array` → `read_relocations_impl`
    /// on the raw bytes. With `sh_link == 0` (no linked symbol
    /// table) the impl uses `dynamic_symbols.len() == 0` for the
    /// bounds check; any non-null symbol index fails with
    /// `"Invalid symbol index N in relocation section at index M"`
    /// and `strip_vmlinux_debug` bubbled the error up. This test
    /// FAILS on that pre-fix codepath and PASSES after neutralize
    /// rewrites `sh_type` to `SHT_PROGBITS` on every `SHT_REL`/
    /// `SHT_RELA` section regardless of `SHF_ALLOC`.
    #[test]
    fn strip_vmlinux_debug_handles_nonalloc_rela_with_invalid_entries() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".rela.invalid",
            object::elf::SHT_RELA,
            // 24 bytes = one Elf64_Rela entry. 0xA5 bytes give
            // r_info = 0xA5A5A5A5A5A5A5A5 — a non-null, out-of-range
            // symbol index that `read_relocations_impl`'s bounds
            // check rejects when sh_link=0 directs the parse to the
            // empty dynamic symbol table.
            &[0xA5; 24],
            |_| {},
        );
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".rela.invalid");
    }

    /// Pipeline pin #2: strip_vmlinux_debug handles a non-ALLOC
    /// `SHT_RELA` section whose `sh_size` is not a multiple of the
    /// `Elf64_Rela` entry size (24 bytes) — a shape that passes
    /// goblin's section-bounds check but fails object-crate's
    /// `data_as_array` divisibility check with `"Invalid ELF
    /// relocation section offset or size"`.
    ///
    /// This is the realistic arm64 kernel 7.0 failure mode: the
    /// section's byte range fits inside the file (so goblin accepts
    /// it) but doesn't represent a well-formed stream of `Elf64_Rela`
    /// entries from `object::build::elf::Builder::read`'s
    /// perspective.
    ///
    /// Before the fix, `Builder::read` failed at
    /// `slice_from_all_bytes` (non-exact multiple of entry size ⇒
    /// tail bytes remaining ⇒ Err). After the fix,
    /// [`neutralize_relocs`] rewrites `sh_type` to `SHT_PROGBITS` on
    /// every reloc section before `Builder::read` sees it; the sh_type
    /// mismatch short-circuits `section.rel()`/`section.rela()` at
    /// the type-check line (object-0.37.3/src/read/elf/section.rs:829,
    /// 849) and `data_as_array` is never called.
    #[test]
    fn strip_vmlinux_debug_handles_nonalloc_rela_with_non_entsize_sh_size() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".rela.odd",
            object::elf::SHT_RELA,
            // 24 bytes = one valid Elf64_Rela. We'll rewrite sh_size
            // to 17 below — fits inside the file's byte range (so
            // goblin accepts it) but 17 % 24 != 0 so object-crate's
            // `slice_from_all_bytes::<Rela64>` rejects the size.
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
                // sh_size = 17 bytes, not divisible by 24
                // (sizeof(Elf64_Rela)). In-bounds (section payload is
                // 24 bytes) so goblin accepts, but the Builder's
                // `slice_from_all_bytes` check leaves a 17-byte tail
                // that rejects — matching the arm64 kernel 7.0
                // failure mode.
                let bad_size: u64 = 17;
                bytes[sh_size_off..sh_size_off + 8].copy_from_slice(&bad_size.to_le_bytes());
            },
        );
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".rela.odd");
    }

    /// Pipeline pin #3: strip_vmlinux_debug handles `SHT_RELR`
    /// sections — arm64 kernels with `CONFIG_PIE` + `CONFIG_RELR`
    /// emit `.relr.dyn` with packed relative-relocation entries.
    ///
    /// This test locks in the SHT_RELR match arm in
    /// [`neutralize_relocs`] by checking BOTH invariants:
    ///
    /// 1. **Neutralize reaches SHT_RELR**: after the first pass,
    ///    the `.relr.dyn` section's `sh_type` is `SHT_PROGBITS`
    ///    (not its original `SHT_RELR`) and `sh_size` is 0. A
    ///    regression that drops SHT_RELR from the match arm leaves
    ///    the section with `sh_type = SHT_RELR` (19) — this
    ///    assertion fires.
    ///
    /// 2. **End-to-end strip succeeds**: `strip_vmlinux_debug` runs
    ///    cleanly and the output has `.relr.dyn` removed by
    ///    keep-list policy.
    ///
    /// Even on a well-formed `.relr.dyn` payload (Builder::read
    /// handles SHT_RELR opaquely via `section.data()` with no
    /// alignment check, so a "happy-path" SHT_RELR might pass
    /// `strip_vmlinux_debug` even without neutralization), the
    /// invariant-1 check locks in the neutralize reach to guard
    /// against a future regression that silently stops rewriting
    /// SHT_RELR sections.
    #[test]
    fn strip_vmlinux_debug_handles_relr_section() {
        let src = TempDir::new().unwrap();
        let vmlinux = build_reloc_fixture(
            src.path(),
            b".relr.dyn",
            object::elf::SHT_RELR,
            // 16 bytes = two packed RELR entries (each u64).
            &[0x77; 16],
            |_| {},
        );

        // Invariant 1: `neutralize_relocs` must rewrite the .relr.dyn
        // section's sh_type to SHT_PROGBITS and zero its sh_size.
        // Checking the function output directly locks in the
        // SHT_RELR match arm — a regression that drops SHT_RELR
        // from the arm would leave sh_type == SHT_RELR here.
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
            ".relr.dyn sh_type must be rewritten to SHT_PROGBITS (SHT_RELR arm of the match); got sh_type={}",
            relr_sh.sh_type,
        );
        assert_eq!(
            relr_sh.sh_size, 0,
            ".relr.dyn sh_size must be zeroed post-neutralize",
        );

        // Invariant 2: end-to-end strip succeeds and removes the
        // reloc section.
        let stripped = strip_vmlinux_debug(&vmlinux).unwrap();
        assert_stripped_preserves_keep_list_and_deletes(stripped.path(), ".relr.dyn");
    }

    /// Pipeline pin #4: after strip_vmlinux_debug succeeds on an ELF
    /// carrying BOTH a non-alloc `SHT_RELA` and a `SHT_RELR` section,
    /// the output has every keep-list section (`.symtab`, `.strtab`,
    /// `.BTF`, `.rodata`) present and every reloc-named section
    /// deleted.
    ///
    /// Guards against a regression where the fix skipped one or the
    /// other reloc type (e.g. a future refactor that splits
    /// [`neutralize_relocs`]'s match arm and drops SHT_RELR). The
    /// pipeline pins above each cover one reloc type in isolation;
    /// this combined fixture ensures the fix holds when both types
    /// appear in the same kernel image.
    #[test]
    fn strip_vmlinux_debug_deletes_reloc_sections_and_preserves_keep_list() {
        use object::write;

        let src = TempDir::new().unwrap();
        let mut obj = write::Object::new(
            object::BinaryFormat::Elf,
            object::Architecture::X86_64,
            object::Endianness::Little,
        );
        // .text anchors the symtab.
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
        // Two reloc sections: .rela.dbg (non-alloc SHT_RELA with
        // garbage entries) and .relr.dyn (SHT_RELR). Both must be
        // deleted from the output and neither must break the strip.
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

        // Positive control: fixture must carry all sections the
        // post-strip assertion inspects. A silent rename by
        // object::write would false-pass the absence checks.
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

        // Keep-list sections survive.
        for name in [".symtab", ".strtab", ".BTF", ".rodata"] {
            assert!(
                names.contains(&name),
                "keep-list section {name} must survive strip; got {names:?}"
            );
        }
        // Both reloc sections deleted.
        for name in [".rela.dbg", ".relr.dyn"] {
            assert!(
                !names.contains(&name),
                "reloc section {name} must be deleted by strip; got {names:?}"
            );
        }
    }

    #[test]
    fn strip_vmlinux_debug_preserves_monitor_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
        };
        // find_test_vmlinux may return /sys/kernel/btf/vmlinux (raw BTF,
        // not an ELF), which strip_vmlinux_debug cannot parse.
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        let stripped = strip_vmlinux_debug(&path).unwrap();
        let stripped_path = stripped.path();
        let syms = crate::monitor::symbols::KernelSymbols::from_vmlinux(stripped_path).unwrap();
        // `runqueues` and `per_cpu_offset` are required non-Option
        // fields on KernelSymbols; `from_vmlinux` bails via
        // `Context::context` if either symbol is absent or zero
        // (`sym_addr` filters `st_value != 0`). Reaching the unwrap
        // above therefore guarantees both are nonzero. These asserts
        // are defensive against a future regression that loosens the
        // sym_addr filter or adds a non-error-on-missing path.
        assert_ne!(
            syms.runqueues, 0,
            "runqueues symbol missing from stripped vmlinux"
        );
        assert_ne!(
            syms.per_cpu_offset, 0,
            "__per_cpu_offset symbol missing from stripped vmlinux"
        );
        // For every optional symbol KernelSymbols tracks: presence must
        // survive the strip. A symbol that is absent from the source
        // vmlinux stays absent (kernel-config-dependent); a symbol that
        // is present must still be present.
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

        // KernelSymbols.init_top_pgt collapses init_top_pgt OR
        // swapper_pg_dir via or_else. Check both names directly against
        // the raw symbol table so a regression that keeps one while
        // dropping the other is caught.
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

    /// Guards against a regression where `strip_vmlinux_debug` returns
    /// `Ok` but produces output close to the source size — e.g. if
    /// `.debug_*` removal is silently skipped.
    ///
    /// Skipped when the source vmlinux carries no `.debug_info`,
    /// which is the signature of an already-stripped input: ktstr's
    /// own cache path caches pre-stripped vmlinuxes, and CI that
    /// points this test at a cache-produced vmlinux would see the
    /// DWARF sections already gone. Running strip over an
    /// already-stripped ELF produces output the same size as the
    /// input (the keep-list partition is idempotent once DWARF is
    /// gone), so the `<` inequality no longer observes the strip.
    /// Rebuild the source-tree vmlinux to exercise this test.
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

    /// Function symbols (in `.text` and friends) must survive the
    /// strip so `resolve_addrs_from_elf` can resolve event addresses
    /// from the cached vmlinux. The strip preserves code-section
    /// headers as `SHT_NOBITS` to keep these symbols from being
    /// dropped by `Builder::delete_orphans`.
    #[test]
    fn strip_vmlinux_debug_preserves_function_symbols() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            skip!("no vmlinux found; {}", crate::KTSTR_KERNEL_HINT);
        };
        if path.starts_with("/sys/") {
            skip!("vmlinux is raw BTF (not ELF), cannot strip debug");
        }
        // Skip if the source vmlinux has no `schedule` symbol -- that
        // means it was already stripped by an older build of ktstr
        // and no longer carries .text symbols. The test exercises
        // strip-preserves behavior, not whether a particular cache
        // entry was rebuilt.
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

    // -- KconfigStatus Display impl --
    //
    // Pins the three Display strings that flow through `kernel list
    // --json` as the `kconfig_status` field. CI scripts consume these
    // exact strings, so any rewording is a downstream-visible
    // contract change.

    #[test]
    fn kconfig_status_display_matches_renders_lowercase_word() {
        assert_eq!(KconfigStatus::Matches.to_string(), "matches");
    }

    #[test]
    fn kconfig_status_display_stale_renders_lowercase_word_without_hashes() {
        let s = KconfigStatus::Stale {
            cached: "deadbeef".to_string(),
            current: "cafebabe".to_string(),
        }
        .to_string();
        assert_eq!(
            s, "stale",
            "Display elides the cached/current hashes; callers that need them must match on the variant directly"
        );
    }

    #[test]
    fn kconfig_status_display_untracked_renders_lowercase_word() {
        assert_eq!(KconfigStatus::Untracked.to_string(), "untracked");
    }

    // ------------------------------------------------------------
    // Cache-entry coordination lock tests
    // ------------------------------------------------------------

    /// `acquire_shared_lock` on a fresh cache root creates the
    /// lockfile at `{root}/.locks/{key}.lock` (and the parent
    /// `.locks/` subdirectory) — guards against drift to the old
    /// sibling layout.
    #[test]
    fn acquire_shared_lock_creates_lockfile_at_expected_path() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let _guard = cache.acquire_shared_lock("some-key-123").unwrap();
        assert!(
            tmp.path().join(".locks").is_dir(),
            "parent .locks/ subdirectory must materialize on first acquire",
        );
        assert!(
            tmp.path().join(".locks").join("some-key-123.lock").exists(),
            "lockfile must materialize at {{cache_root}}/.locks/{{key}}.lock on first acquire",
        );
    }

    /// Two concurrent `acquire_shared_lock` calls on the same key
    /// both succeed — LOCK_SH coexists. Uses separate threads so
    /// each gets its own open-file-description (flock is per-OFD).
    #[test]
    fn acquire_shared_lock_permits_concurrent_readers() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().to_path_buf()));
        let key = "concurrent-sh";
        let success = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            let success = Arc::clone(&success);
            handles.push(std::thread::spawn(move || {
                let _g = cache
                    .acquire_shared_lock(key)
                    .expect("LOCK_SH must succeed");
                success.fetch_add(1, Ordering::SeqCst);
                // Hold briefly so all threads concurrently hold the
                // lock. Without this sleep, threads could serialize
                // through a narrow no-contention window and pass
                // even if the lock mistakenly rejected coexistence.
                std::thread::sleep(std::time::Duration::from_millis(50));
            }));
        }
        for h in handles {
            h.join().expect("reader thread panicked");
        }
        assert_eq!(
            success.load(Ordering::SeqCst),
            4,
            "all 4 concurrent LOCK_SH acquires must succeed",
        );
    }

    /// `try_acquire_exclusive_lock` fails with an error naming the
    /// lockfile when a concurrent reader holds LOCK_SH. A
    /// spawned thread takes LOCK_SH and sleeps; the main thread
    /// attempts `try_acquire_exclusive_lock` non-blocking and
    /// asserts the error path fires.
    #[test]
    fn try_acquire_exclusive_lock_fails_with_active_reader() {
        use std::sync::Arc;
        use std::sync::mpsc;
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().to_path_buf()));
        let key = "force-contended";
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let cache_reader = Arc::clone(&cache);
        let reader = std::thread::spawn(move || {
            let _g = cache_reader
                .acquire_shared_lock(key)
                .expect("reader LOCK_SH must succeed");
            ready_tx.send(()).unwrap();
            // Block until the main thread's non-blocking attempt
            // has had its chance to fail. Without this gate, the
            // reader could drop its lock before the main thread's
            // try_acquire_exclusive_lock ran, producing a
            // false-pass.
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader thread did not signal ready in time");
        // Now the reader is holding LOCK_SH. A non-blocking LOCK_EX
        // must bail.
        let err = cache.try_acquire_exclusive_lock(key).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("is locked by active test runs") || msg.contains("holders:"),
            "error must surface the contention diagnostic; got: {msg}",
        );
        assert!(
            msg.contains("lockfile"),
            "error must name the lockfile path: {msg}",
        );
        // Release the reader so the test cleans up.
        release_tx.send(()).unwrap();
        reader.join().expect("reader thread panicked");
    }

    /// `acquire_exclusive_lock_blocking` times out with the
    /// documented wording when a concurrent reader holds LOCK_SH
    /// longer than the timeout allows. Uses a 200ms timeout + a
    /// reader that holds for >500ms to reliably trip the bail.
    #[test]
    fn acquire_exclusive_lock_blocking_times_out_on_contention() {
        use std::sync::Arc;
        use std::sync::mpsc;
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().to_path_buf()));
        let key = "blocking-timeout";
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let cache_reader = Arc::clone(&cache);
        let reader = std::thread::spawn(move || {
            let _g = cache_reader
                .acquire_shared_lock(key)
                .expect("reader LOCK_SH must succeed");
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader did not signal ready in time");
        let start = std::time::Instant::now();
        let err = cache
            .acquire_exclusive_lock_blocking(key, std::time::Duration::from_millis(200))
            .unwrap_err();
        let elapsed = start.elapsed();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timed out"),
            "error must mention the timeout: {msg}",
        );
        assert!(
            elapsed >= std::time::Duration::from_millis(150),
            "acquire should have waited ~timeout (150ms lower bound); \
             got {elapsed:?}",
        );
        release_tx.send(()).unwrap();
        reader.join().expect("reader thread panicked");
    }

    /// `store()` acquires its own exclusive lock and completes
    /// successfully when no readers contend. Regression pin for
    /// the internal `acquire_exclusive_lock_blocking` call.
    #[test]
    fn store_succeeds_under_internal_exclusive_lock() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().join("cache"));
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let entry = cache
            .store("internal-lock", &CacheArtifacts::new(&image), &meta)
            .expect("store must succeed when no readers contend");
        assert!(entry.path.join("bzImage").exists());
        // Lockfile should exist in the .locks/ subdirectory
        // (acquired during store).
        assert!(
            tmp.path()
                .join("cache")
                .join(".locks")
                .join("internal-lock.lock")
                .exists(),
            "lockfile materialized during store must persist after \
             store returns (it's fine; the flock is released on fd \
             drop but the file stays as a reusable sentinel)",
        );
    }

    /// `store()` blocks while a reader holds LOCK_SH, then
    /// completes after the reader releases. Drives the path by
    /// spawning a reader that holds its lock while attempting
    /// store in a thread; probes that store() does NOT complete
    /// within 200ms, then releases the reader and asserts store()
    /// completes within 10s.
    #[test]
    fn store_blocks_while_reader_holds_shared_lock() {
        use std::sync::Arc;
        use std::sync::mpsc;
        let tmp = TempDir::new().unwrap();
        let cache = Arc::new(CacheDir::with_root(tmp.path().join("cache-block")));
        let key = "blocked-store";
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let cache_reader = Arc::clone(&cache);
        let reader = std::thread::spawn(move || {
            let _g = cache_reader
                .acquire_shared_lock(key)
                .expect("reader LOCK_SH must succeed");
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("reader did not signal ready in time");

        // Reader is holding LOCK_SH. A store attempt must block.
        // Spawn the store in a thread and check it hasn't
        // completed within a short window.
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());
        let meta = test_metadata("6.14.2");
        let (store_done_tx, store_done_rx) = mpsc::channel();
        let cache_store = Arc::clone(&cache);
        let image_clone = image.clone();
        let store_thread = std::thread::spawn(move || {
            let _ = cache_store.store(key, &CacheArtifacts::new(&image_clone), &meta);
            store_done_tx.send(()).unwrap();
        });
        // Short probe: store must NOT complete while reader holds lock.
        let early = store_done_rx.recv_timeout(std::time::Duration::from_millis(200));
        assert!(
            early.is_err(),
            "store() must block while reader holds LOCK_SH; got completion signal early",
        );
        // Release the reader — store should now unblock and finish.
        release_tx.send(()).unwrap();
        let finish = store_done_rx.recv_timeout(std::time::Duration::from_secs(10));
        assert!(
            finish.is_ok(),
            "store() must complete after reader releases; got timeout",
        );
        reader.join().expect("reader thread panicked");
        store_thread.join().expect("store thread panicked");
    }

    /// `lock_path` returns `{cache_root}/.locks/{key}.lock` — pins
    /// the exact on-disk shape against a refactor that relocates
    /// the lockfile. Pure path construction, no filesystem access.
    #[test]
    fn lock_path_returns_expected_shape() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let path = cache.lock_path("my-key-42");
        assert_eq!(path, tmp.path().join(".locks").join("my-key-42.lock"));
    }

    /// `.locks/` subdirectory PERSISTS after the lock guard drops.
    /// Kernel_clean and any other walker that relies on list()
    /// filtering dotfiles assumes `.locks/` outlives individual
    /// acquires. A regression that rm'd the directory on guard
    /// drop would cause next-acquire to re-`mkdir` on a different
    /// inode and invalidate any /proc/locks peer-holder lookup
    /// (the peer's inode would be stale).
    #[test]
    fn locks_subdir_persists_after_guard_drop() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        let locks_dir = tmp.path().join(".locks");
        {
            let _guard = cache
                .acquire_shared_lock("persist-test")
                .expect("acquire must succeed");
            assert!(locks_dir.is_dir(), "must exist during guard lifetime");
        }
        // Guard dropped. .locks/ must still exist.
        assert!(
            locks_dir.is_dir(),
            ".locks/ must persist after guard drop — next acquire \
             keys /proc/locks on the existing inode",
        );
    }

    /// `CacheDir::list` skips `.locks/` — pins the dotfile-filter
    /// contract in `CacheDir::list`. kernel_clean iterates what
    /// `list()` returns, so this is the same guard: `.locks/` is
    /// NEVER visible to the cleanup path. A future refactor that
    /// removed the `starts_with('.')` filter would regress
    /// through this test.
    #[test]
    fn list_skips_locks_dotfile_subdirectory() {
        let tmp = TempDir::new().unwrap();
        let cache = CacheDir::with_root(tmp.path().to_path_buf());
        // Materialize .locks/ via acquire, then list().
        let _guard = cache.acquire_shared_lock("dummy").expect("acquire");
        drop(_guard);
        assert!(
            tmp.path().join(".locks").is_dir(),
            ".locks/ must exist after acquire drop",
        );
        let entries = cache.list().expect("list must succeed");
        let keys: Vec<&str> = entries
            .iter()
            .map(|e| match e {
                ListedEntry::Valid(entry) => entry.key.as_str(),
                ListedEntry::Corrupt { key, .. } => key.as_str(),
            })
            .collect();
        assert!(
            !keys.iter().any(|k| k.starts_with('.')),
            "list() must not return dotfile children: {keys:?}",
        );
    }

    /// Empty cache root: acquire creates `.locks/` lazily.
    /// Distinct from `acquire_shared_lock_creates_lockfile_at_expected_path`
    /// above because THAT test asserts the lockfile path; this
    /// one pins the LAZY-create behavior — the cache root can be
    /// totally empty (no kernel entries) and first acquire still
    /// works.
    #[test]
    fn acquire_on_empty_root_creates_locks_dir_lazily() {
        let tmp = TempDir::new().unwrap();
        let root = tmp.path().join("pristine");
        std::fs::create_dir(&root).unwrap();
        let cache = CacheDir::with_root(root.clone());
        // Pre-acquire: no .locks/ yet.
        assert!(!root.join(".locks").exists());
        let _guard = cache
            .acquire_shared_lock("lazy-test")
            .expect("first acquire on empty root must succeed");
        assert!(
            root.join(".locks").is_dir(),
            "first acquire must materialize .locks/ lazily",
        );
    }

    /// `clean_all` MUST preserve the `.locks/` subdirectory. The
    /// `list()` filter skips dotfile children (tested elsewhere);
    /// `clean_all` removes what `list()` returns, so dotfiles — and
    /// specifically `.locks/` — survive. Without this guarantee,
    /// cleaning would delete a live SH flock's lockfile inode,
    /// leaving the next acquirer's `/proc/locks` lookup blind to
    /// the peer that still holds the (now-orphaned) fd.
    ///
    /// Repro sequence: populate an entry, acquire SH, clean_all,
    /// assert `.locks/` still exists AND the lockfile still exists
    /// inside it (the held fd keeps the inode alive even if the
    /// directory entry were removed — we're checking the directory
    /// entry specifically).
    #[test]
    fn cache_dir_clean_all_preserves_locks_subdir() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());
        let src_dir = TempDir::new().unwrap();
        let image = create_fake_image(src_dir.path());

        // Populate a cache entry so clean_all has something to
        // remove (its job is to tell the dotfile filter apart from
        // real entries).
        cache
            .store(
                "entry-a",
                &CacheArtifacts::new(&image),
                &test_metadata("6.14.0"),
            )
            .expect("store must succeed");
        // Acquire a shared lock so .locks/{key}.lock materializes.
        let _guard = cache
            .acquire_shared_lock("entry-a")
            .expect("SH acquire must succeed");

        let locks_dir = cache_root.join(".locks");
        let lockfile = locks_dir.join("entry-a.lock");
        assert!(locks_dir.is_dir(), "precondition: .locks/ must exist");
        assert!(lockfile.exists(), "precondition: lockfile must exist");

        // Clean every entry. .locks/ is a dotfile-prefixed child
        // and must NOT be treated as a cache entry.
        let removed = cache.clean_all().expect("clean_all must succeed");
        assert_eq!(removed, 1, "clean_all must remove exactly 1 entry");

        // Post-clean: .locks/ subdirectory survives so the held SH
        // flock's inode is still the one /proc/locks points at.
        assert!(
            locks_dir.is_dir(),
            ".locks/ subdirectory must survive clean_all — the live \
             SH flock's inode would otherwise orphan",
        );
        assert!(
            lockfile.exists(),
            "lockfile must still exist under .locks/ after clean_all",
        );

        // And the entry itself is gone.
        assert!(
            !cache_root.join("entry-a").exists(),
            "cache entry must be removed by clean_all",
        );
    }

    /// `acquire_shared_lock` MUST reject cache keys containing path
    /// traversal components (`..`, `/`). Without the rejection, a
    /// key of `"../../etc/passwd"` would join against the cache
    /// root and materialize a lockfile OUTSIDE `.locks/`, which is
    /// both a security concern (attacker-controlled write through
    /// a library entry point) and a correctness failure (the lock
    /// file's inode won't match anything in subsequent enumeration).
    ///
    /// Pins the `validate_cache_key` rejection from the two path-
    /// traversal entry points — the `/` separator check and the
    /// `..` component check — with a single test input that
    /// triggers both. The error text must be actionable; asserting
    /// against the `"path"` substring in the message catches both
    /// the separator and traversal rejection arms.
    #[test]
    fn cache_dir_acquire_rejects_path_traversal_key() {
        let tmp = TempDir::new().unwrap();
        let cache_root = tmp.path().join("cache");
        let cache = CacheDir::with_root(cache_root.clone());

        // Attacker-shaped key: contains both `/` separators and
        // `..` traversal, hitting both rejection arms in
        // `validate_cache_key`.
        let err = cache
            .acquire_shared_lock("../../etc/passwd")
            .expect_err("path-traversal key must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path"),
            "error must mention path rejection: {msg}",
        );

        // Critically: no lockfile must have been created anywhere
        // outside `.locks/`. Walk two levels above the cache root
        // to verify nothing landed in `tmp.path()` or a traversal
        // destination. The cache root itself may or may not exist
        // (acquire creates `.locks/` lazily, but the validator
        // rejects BEFORE that materialization).
        let etc_passwd_lock = tmp.path().join("etc").join("passwd.lock");
        assert!(
            !etc_passwd_lock.exists(),
            "path traversal must NOT create a lockfile outside .locks/",
        );
        // And verify .locks/ wasn't touched either — the validator
        // rejects before any FS state is mutated.
        assert!(
            !cache_root.join(".locks").exists()
                || cache_root
                    .join(".locks")
                    .read_dir()
                    .unwrap()
                    .next()
                    .is_none(),
            ".locks/ must be empty if it exists at all — validator \
             rejects before lockfile creation",
        );
    }

    // -- validate_home_for_cache direct unit tests --
    //
    // These tests pin the helper directly. The helper reads
    // `HOME` from the process environment, so each test holds
    // [`lock_env`] across the env mutation and uses
    // [`EnvVarGuard`] to scope the change. The integration-level
    // pins on the full `KTSTR_CACHE_DIR → XDG_CACHE_HOME → HOME`
    // cascade live in model.rs and cache.rs as
    // `resolve_cache_root_*` tests; this set covers the helper's
    // contract surface so a regression in the validation logic
    // surfaces against this dedicated entry point as well as the
    // integration paths.

    /// Unset `HOME` — `env::var()` returns `Err(NotPresent)` and
    /// the validator surfaces "HOME is unset" as the matching
    /// arm. Distinguished from the empty-string case below so an
    /// operator hitting either shape sees the actual misconfiguration
    /// in the diagnostic.
    #[test]
    fn validate_home_for_cache_rejects_unset() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::remove("HOME");
        let err = validate_home_for_cache().expect_err("unset HOME must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("HOME is unset"),
            "diagnostic must call out the unset case specifically: {msg}",
        );
        assert!(
            !msg.contains("HOME is set to the empty string"),
            "unset HOME must NOT use the empty-string diagnostic — the two \
             cases are distinct now (NotPresent vs Ok(\"\")): {msg}",
        );
    }

    /// Empty `HOME` — explicitly assigned to the empty string.
    /// `env::var()` returns `Ok("")` and the validator surfaces
    /// "HOME is set to the empty string" so an operator can
    /// identify a Dockerfile `ENV HOME=` or shell-rc `export HOME=`
    /// typo as the cause rather than confusing it with the
    /// container-init-dropped-HOME case.
    #[test]
    fn validate_home_for_cache_rejects_empty() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::set("HOME", "");
        let err = validate_home_for_cache().expect_err("empty HOME must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("HOME is set to the empty string"),
            "diagnostic must call out the empty-string case specifically: {msg}",
        );
        assert!(
            !msg.contains("HOME is unset"),
            "empty HOME must NOT use the unset diagnostic — the two \
             cases are distinct now: {msg}",
        );
    }

    /// Literal `/` — the container-init / no-home shape. Pins the
    /// dedicated arm (separate from the more general
    /// `is_empty()` check) so the operator-facing diagnostic stays
    /// specific to this case rather than collapsing into a generic
    /// "unset" message.
    #[test]
    fn validate_home_for_cache_rejects_root_slash() {
        let _env_lock = lock_env();
        let _home = EnvVarGuard::set("HOME", "/");
        let err = validate_home_for_cache().expect_err("HOME=/ must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("HOME is `/`"),
            "diagnostic must call out the root-slash case specifically: {msg}",
        );
        assert!(
            msg.contains("/.cache/ktstr"),
            "diagnostic must explain why (/.cache/ktstr aliases root fs): {msg}",
        );
    }

    /// Relative path — would resolve against CWD at every call,
    /// silently relocating the cache as the operator changes
    /// directories. Pins the absolute-path requirement.
    #[test]
    fn validate_home_for_cache_rejects_relative_path() {
        let _env_lock = lock_env();
        for rel in ["relative", "./relative", "home/user", "."] {
            let _home = EnvVarGuard::set("HOME", rel);
            let err = validate_home_for_cache()
                .expect_err(&format!("relative path '{rel}' must be rejected"));
            let msg = format!("{err:#}");
            assert!(
                msg.contains("not an absolute path"),
                "[rel={rel:?}] diagnostic must call out non-absolute: {msg}",
            );
            assert!(
                msg.contains(&format!("{rel:?}")),
                "[rel={rel:?}] diagnostic must echo the offending value verbatim: {msg}",
            );
        }
    }

    /// Acceptable shapes — absolute paths starting with `/` and
    /// longer than just `/`. Pins the happy path so a regression
    /// that tightened one of the rejection arms (e.g. a length
    /// check that accidentally rejected `/a`) surfaces here.
    /// Also pins that the returned PathBuf carries the HOME bytes
    /// verbatim — no canonicalization, no .cache/ktstr suffix.
    #[test]
    fn validate_home_for_cache_accepts_absolute_paths() {
        let _env_lock = lock_env();
        for ok in [
            "/home/user",
            "/var/empty",
            "/root",
            "/a", // shortest non-`/` absolute path
            "/home/user with spaces",
            "/home/user/.local/share",
        ] {
            let _home = EnvVarGuard::set("HOME", ok);
            let got = validate_home_for_cache()
                .unwrap_or_else(|e| panic!("absolute path {ok:?} must be accepted; got: {e:#}"));
            assert_eq!(
                got,
                std::path::PathBuf::from(ok),
                "returned PathBuf must equal the HOME value verbatim — \
                 helper does not append the cache suffix or canonicalize",
            );
        }
    }

    /// Edge: a path that starts with `/` but contains junk later
    /// (e.g. `//`, `/./`, `/.`). The helper does NOT canonicalize —
    /// these accept and surface the OS-level diagnostic at use
    /// time per the body comments above the helper. Pins this
    /// "intentionally not caught" boundary so a future change that
    /// adds canonicalization (which would BREAK this test) is
    /// forced to update the doc comments at the same time.
    #[test]
    fn validate_home_for_cache_does_not_canonicalize_dots_and_doubles() {
        let _env_lock = lock_env();
        for not_normalized in ["//", "/./", "/.", "/foo//bar", "/./home"] {
            let _home = EnvVarGuard::set("HOME", not_normalized);
            validate_home_for_cache().unwrap_or_else(|e| {
                panic!(
                    "non-normalized but absolute path {not_normalized:?} must \
                     pass the helper (downstream OS surfaces the diagnostic); \
                     got: {e:#}",
                )
            });
        }
    }
}
