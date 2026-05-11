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
//!    we target) embed their compiled BPF object(s) inline at that
//!    section via the libbpf-rs / scx skel codegen. Each `STT_OBJECT`
//!    symbol in the outer ELF whose containing section is `.bpf.objs`
//!    points at a contiguous embedded ELF blob — the BPF object that
//!    the scheduler will hand to `bpf_object__load` at runtime.
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
//! Any failure returns an empty [`CastMap`]. The log level depends on
//! the failure kind: scheduler-binary read errors, outer ELF parse
//! failures, missing `.bpf.objs`, inner ELF parse failures, and
//! malformed `.BTF` log at `warn!` (these indicate a likely bug in
//! the scheduler build); a missing `.BTF` section and an inner ELF
//! with no executable BPF program sections log at `debug!` (these
//! shapes are valid for non-scx binaries that ship a `.bpf.objs` for
//! unrelated reasons). The dump path is best-effort — a missing
//! cast map silently disables typed-pointer promotion in the renderer
//! (every `u64` field renders as a plain counter, the pre-integration
//! default).
//!
//! No libbpf calls, no kernel BPF interaction, no CAP_BPF needed — this
//! runs purely on the on-disk binary bytes.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use crate::monitor::cast_analysis::{
    BPF_PSEUDO_CALL, BPF_PSEUDO_KFUNC_CALL, BpfInsn, CastMap, DatasecPointer, FuncEntry,
    SubprogReturn, analyze_casts,
};

use btf_rs::{Btf, Type};

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

/// One entry in the cross-BTF Fwd resolution index — locates a
/// complete struct/union body by `(BTF index, type id)`.
///
/// `btfs_idx` selects which entry of [`CastAnalysisOutput::btfs`]
/// carries the body; `type_id` is the type id WITHIN that BTF's
/// own id space (distinct from the entry BTF's id space the
/// renderer's chase entered with).
///
/// Used as the value type of [`CastAnalysisOutput::fwd_index`] —
/// the renderer's
/// [`crate::monitor::btf_render::MemReader::cross_btf_resolve_fwd`]
/// override looks the entry up by name, picks `btfs[btfs_idx]`,
/// and recurses against `type_id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FwdIndexEntry {
    /// Index into [`CastAnalysisOutput::btfs`] selecting which
    /// embedded BPF object's parsed program BTF carries the body.
    pub(crate) btfs_idx: usize,
    /// Type id within `btfs[btfs_idx]`'s own id space. Distinct
    /// from the entry BTF's id space; the chase code switches the
    /// rendering BTF before resolving the id.
    pub(crate) type_id: u32,
}

/// Output of one full pass of host-side scheduler cast analysis: the
/// `(parent_struct, member_offset) -> CastHit` map, the list of every
/// embedded BPF object's program BTF, and a name-keyed index over
/// every complete (`!is_fwd`) struct/union/typedef across those BTFs.
///
/// The renderer's chase paths consult the cross-BTF index when a
/// declared `BTF_KIND_FWD` pointee has no complete sibling in its
/// own BTF: the index points at the `(btfs[idx], type_id)` pair where
/// the body lives, so a `cgx_target __arena *` declared in object A
/// (Fwd-only) renders as the full `struct cgx_target { ... }` body
/// from object B without dropping into the "forward declaration; body
/// not in this BTF" skip.
///
/// Built once per scheduler binary per process via
/// [`cached_cast_analysis_for_scheduler`] and shared across VMs by
/// content hash. The `btfs` vec is `Arc<Btf>` so the rendered
/// borrows live for the full dump pass without copying the parsed
/// BTF.
pub(crate) struct CastAnalysisOutput {
    /// `(parent_btf_id, member_offset) -> CastHit` recovered by the
    /// instruction-level cast analyzer. The renderer's
    /// [`crate::monitor::btf_render::MemReader::cast_lookup`] hits
    /// against the per-program BTF the rendered map was loaded from.
    /// Even when the cast hit is empty, the wrapping output is still
    /// retained because the cross-BTF [`fwd_index`] is independently
    /// useful — a scheduler whose Fwd pointers all live in
    /// non-typed-pointer-bearing maps still benefits from the index
    /// when the renderer chases those maps' [`Type::Ptr`] arms.
    pub(crate) cast_maps: Vec<Arc<CastMap>>,
    /// Every embedded BPF object's parsed program BTF, in the same
    /// order [`iter_embedded_bpf_objects`] yielded the slices. Index
    /// 0 is the first symbol-driven slice (or the fallback whole-
    /// section blob), index 1 is the next, and so on. Empty when no
    /// BTF parsed successfully — the renderer falls back to the
    /// per-map vmlinux BTF for any cross-BTF resolution that would
    /// have hit this index.
    pub(crate) btfs: Vec<Arc<Btf>>,
    /// `struct_or_union_name -> FwdIndexEntry` for every complete
    /// (`!is_fwd`) [`btf_rs::Type::Struct`] / [`btf_rs::Type::Union`]
    /// across [`btfs`]. `Typedef` is NOT indexed — typedefs add no
    /// body and the chase path peels through them via
    /// [`peel_modifiers_with_id`] before consulting the index.
    ///
    /// First-write-wins: when the same name appears in multiple
    /// BTFs the index keeps the first-seen entry. Two distinct
    /// programs declaring `struct foo` with conflicting layouts
    /// would each see their own program BTF resolve correctly via
    /// the renderer's local Fwd-resolving peel; the cross-BTF index
    /// only fires when the local resolve failed. The first-write-
    /// wins policy keeps the index deterministic across re-runs of
    /// the analyzer on the same binary.
    ///
    /// Anonymous structs/unions are not indexed (no name to key on);
    /// the chase falls through to the existing "forward declaration;
    /// body not in this BTF" skip path for those.
    pub(crate) fwd_index: HashMap<String, FwdIndexEntry>,
    /// Unique alloc_sizes captured from `scx_static_alloc_internal`
    /// call sites via [`build_subprog_returns`]. Threaded to the
    /// renderer as a last-resort fallback for deferred-resolve
    /// arena chases whose CastHit has `alloc_size: None`.
    /// `(alloc_size, struct_name)` pairs: for each captured alloc_size
    /// from `scx_static_alloc_internal`, the struct name that
    /// `discover_payload_btf_id` resolved uniquely in the embedded
    /// `.bpf.o` BTF. The renderer uses the name with
    /// `cross_btf_resolve_fwd` to find the struct body at chase time.
    /// Empty when no sizes resolved or no embedded BTF was available.
    pub(crate) alloc_size_types: Vec<(u64, String)>,
}

/// Per-`KtstrVm` lazy on-demand BPF cast-analysis handle.
///
/// Captures the scheduler binary path at VM build time (no analyzer
/// work runs here) and exposes a lazy accessor (`.get_full()`)
/// that runs the analysis on first call and caches the result
/// inside an [`OnceLock`]. The failure-dump path is the only
/// production caller, so a test that passes without ever dumping
/// pays zero analyzer cost. A test that triggers multiple dumps
/// in the same VM (e.g. periodic-capture + final freeze) only
/// runs the analyzer once.
///
/// # Cross-VM sharing
///
/// `.get_full()` consults the process-wide content-hash cache via
/// [`cached_cast_analysis_for_scheduler`], so two VMs in the same
/// process that share a scheduler binary share one analyzed
/// `Arc<CastAnalysisOutput>`. Production runs under nextest use
/// process-per-test by default, so the cross-VM share helps mostly
/// for the auto-repro path (which boots a second VM in the same
/// process after a primary-test failure) and for any future
/// in-process multi-test driver.
///
/// # Concurrency
///
/// `OnceLock::get_or_init` serialises concurrent first-callers in
/// the same VM: the second caller blocks while the first runs the
/// analysis, then both observe the cached
/// `Option<Arc<CastAnalysisOutput>>`. The inner
/// [`cached_cast_analysis_for_scheduler`] additionally dedupes work
/// across VMs by content hash and uses an inner `OnceLock` per
/// cache entry to avoid the thundering-herd shape where two VMs
/// find the cache empty under the same lock and both run the
/// analyzer after releasing it.
pub(crate) struct LazyCastMap {
    /// Scheduler binary path captured at VM build time. `None`
    /// when the builder had no scheduler binary; `.get_full()`
    /// returns `None` immediately in that case.
    scheduler_binary: Option<std::path::PathBuf>,
    /// One-shot per-VM cache of the analysis result. Populated by
    /// the first `.get_full()` caller via
    /// [`cached_cast_analysis_for_scheduler`]; `None` is cached
    /// when no scheduler binary was set OR the analyzer produced
    /// neither cast findings nor cross-BTF index entries.
    inner: OnceLock<Option<Arc<CastAnalysisOutput>>>,
}

impl LazyCastMap {
    /// Construct a lazy handle for `scheduler_binary`. No file I/O
    /// or analyzer work runs here — both defer to
    /// [`Self::get_full`].
    pub(crate) fn new(scheduler_binary: Option<std::path::PathBuf>) -> Self {
        Self {
            scheduler_binary,
            inner: OnceLock::new(),
        }
    }

    /// Force the lazy analysis (or return the cached result) and
    /// hand back the full [`CastAnalysisOutput`] including the
    /// cross-BTF Fwd index.
    ///
    /// First call runs [`cached_cast_analysis_for_scheduler`] on
    /// the captured path, which itself consults the process-wide
    /// content-hash cache — so two VMs that share a scheduler
    /// binary path produce one analyzer run per process.
    /// Subsequent `.get_full()` calls on the same VM hit the inner
    /// `OnceLock` and return immediately.
    ///
    /// Returns `None` when no scheduler binary was set, the file
    /// read failed, or the analyzer produced neither cast findings
    /// nor cross-BTF index entries.
    pub(crate) fn get_full(&self) -> Option<Arc<CastAnalysisOutput>> {
        self.inner
            .get_or_init(|| {
                self.scheduler_binary
                    .as_deref()
                    .and_then(cached_cast_analysis_for_scheduler)
            })
            .clone()
    }
}

/// Process-wide cache entry: scheduler binary content hash →
/// `Arc<OnceLock<Option<Arc<CastAnalysisOutput>>>>`. The outer
/// `OnceLock` is the deduplication primitive — two VMs that hash
/// to the same content but find the entry uninitialized both call
/// `entry.get_or_init(...)`, which runs the analyzer exactly once.
/// The entry's eventual value is the collapsed
/// `Option<Arc<CastAnalysisOutput>>` (`None` on empty cast map AND
/// empty cross-BTF index, `Some` on any non-empty). Without the
/// inner `OnceLock` shape, two cache misses on the same hash would
/// each release the `Mutex<HashMap>` lock, then race to run the
/// analyzer in parallel — the thundering-herd anti-pattern.
type CastCacheEntry = Arc<OnceLock<Option<Arc<CastAnalysisOutput>>>>;

fn cast_cache() -> &'static Mutex<HashMap<u64, CastCacheEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, CastCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn ahash_bytes(bytes: &[u8]) -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = ahash::RandomState::with_seeds(0, 0, 0, 0).build_hasher();
    hasher.write(bytes);
    hasher.finish()
}

/// Process-wide content-hash-cached entry point.
///
/// Reads the scheduler binary once, hashes the bytes via ahash
/// (AES-NI accelerated, deterministic per-binary with fixed seeds),
/// and either returns the previously-analysed
/// `Option<Arc<CastAnalysisOutput>>` for that hash or runs the
/// analyzer once to populate the cache entry. The cache value is
/// `Option<Arc>` (collapsed empty → `None`) so the dump path's
/// borrow expresses "no analysis available" cleanly without an
/// emptiness check at every freeze.
///
/// # Why content-hash, not path-stat
///
/// `(path, dev, ino, mtime, len)` would be a stale-tolerant cache
/// key when scheduler binaries always rebuild with a fresh mtime,
/// but a `cp -p`-style overwrite or hardlinked rotation can
/// preserve mtime AND length while the bytes change, hitting a
/// stale entry and rendering the wrong cast map for a
/// just-replaced binary. Content-hash over the actual bytes is
/// the only key that is correct for every overwrite shape. The
/// hash cost is dominated by the file read which has to happen
/// anyway.
///
/// # Concurrency
///
/// Two simultaneous misses for the same hash do NOT both run the
/// analyzer — they share an `Arc<OnceLock<...>>` and the second
/// caller blocks inside `OnceLock::get_or_init` until the first
/// finishes. Misses for different hashes proceed in parallel
/// because the `Mutex<HashMap>` is held only across the
/// hash-and-fetch step.
///
/// # Returns
///
/// `None` when the file read fails (transient I/O) OR the
/// analyzer's result is empty AND the cross-BTF index is empty.
/// Otherwise the analyzed `Arc<CastAnalysisOutput>` shared with
/// every prior caller for the same binary content.
pub(crate) fn cached_cast_analysis_for_scheduler(path: &Path) -> Option<Arc<CastAnalysisOutput>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "cast_analysis: read scheduler binary failed; \
                 dump renderer will fall back to plain u64 counters"
            );
            return None;
        }
    };
    let hash_t0 = std::time::Instant::now();
    let hash = ahash_bytes(&bytes);
    tracing::debug!(
        elapsed_us = hash_t0.elapsed().as_micros() as u64,
        len = bytes.len(),
        hash = format_args!("{hash:016x}"),
        "cast_analysis: scheduler binary content hash finished"
    );

    let entry: CastCacheEntry = {
        let mut cache = cast_cache().lock().unwrap();
        cache
            .entry(hash)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone()
    };
    entry
        .get_or_init(|| {
            // Disk cache probe: if a prior process already analyzed
            // this binary, load the result without re-running the
            // instruction walker. BTFs are reparsed from the binary
            // bytes (Btf is not serializable).
            let btfs = parse_btfs_from_bytes(&bytes);
            if let Some((cast_map, fwd_index, alloc_size_types)) = persist::try_load(hash, btfs.len()) {
                tracing::debug!("cast_analysis: disk cache hit");
                let out = CastAnalysisOutput {
                    cast_maps: vec![Arc::new(cast_map)],
                    btfs,
                    fwd_index,
                    alloc_size_types,
                };
                let total: usize = out.cast_maps.iter().map(|m| m.len()).sum();
                return if total == 0 && out.fwd_index.is_empty() {
                    None
                } else {
                    Some(Arc::new(out))
                };
            }

            let analyze_t0 = std::time::Instant::now();
            let out = build_cast_analysis_from_bytes(&bytes);
            tracing::debug!(
                elapsed_ms = analyze_t0.elapsed().as_millis() as u64,
                casts = out.cast_maps.iter().map(|m| m.len()).sum::<usize>(),
                btfs = out.btfs.len(),
                fwd_index = out.fwd_index.len(),
                "cast_analysis: on-demand analysis finished"
            );
            let merged_for_cache: CastMap = out
                .cast_maps
                .iter()
                .flat_map(|m| m.iter())
                .map(|(&k, &v)| (k, v))
                .collect();
            persist::try_save(hash, &merged_for_cache, &out.fwd_index, out.btfs.len(), &out.alloc_size_types);
            let total_casts: usize = out.cast_maps.iter().map(|m| m.len()).sum();
            if total_casts == 0 && out.fwd_index.is_empty() {
                None
            } else {
                Some(Arc::new(out))
            }
        })
        .clone()
}

/// Run the cast-analysis pipeline on already-loaded scheduler
/// binary bytes.
///
/// Locates every embedded BPF object inside `.bpf.objs`, parses
/// each object's program BTF, runs the analyzer per-object, and
/// returns the merged [`CastMap`] alongside the parsed BTFs and a
/// name-keyed cross-BTF Fwd resolution index over every complete
/// struct/union across them. The renderer's chase paths consume
/// the index when a `BTF_KIND_FWD` pointee in one BTF resolves to
/// a complete sibling in another — the typical multi-object
/// scheduler shape where one `.bpf.c` declares
/// `struct cgx_target;` (forward) and a sibling object defines
/// `struct cgx_target { ... }` (full body).
///
/// Returns an empty [`CastAnalysisOutput`] on parse failure
/// (`cast_map` empty, `btfs` empty, `fwd_index` empty). Per-stage
/// timing is emitted at `debug!` so a future regression in any
/// sub-stage is visible without re-instrumenting.
///
/// This is the lowest-level entry point; see
/// [`cached_cast_analysis_for_scheduler`] for the production
/// path-driven, content-hash-cached, lazy-on-demand wrapper.
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
pub(crate) fn build_cast_analysis_from_bytes(bytes: &[u8]) -> CastAnalysisOutput {
    let parse_t0 = std::time::Instant::now();
    let outer = match goblin::elf::Elf::parse(bytes) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cast_analysis: parse outer ELF failed; \
                 dump renderer will fall back to plain u64 counters"
            );
            return CastAnalysisOutput {
                cast_maps: vec![Arc::new(CastMap::new())],
                btfs: Vec::new(),
                fwd_index: HashMap::new(),
                alloc_size_types: Vec::new(),
            };
        }
    };
    let bpf_objs_section = match find_section(&outer, ".bpf.objs") {
        Some(s) => s,
        None => {
            tracing::debug!(
                "cast_analysis: scheduler binary has no .bpf.objs section; \
                 typed-pointer rendering disabled"
            );
            return CastAnalysisOutput {
                cast_maps: vec![Arc::new(CastMap::new())],
                btfs: Vec::new(),
                fwd_index: HashMap::new(),
                alloc_size_types: Vec::new(),
            };
        }
    };
    tracing::debug!(
        elapsed_us = parse_t0.elapsed().as_micros() as u64,
        "cast_analysis: outer ELF parse + .bpf.objs lookup finished"
    );

    let mut cast_maps: Vec<Arc<CastMap>> = Vec::new();
    let mut btfs: Vec<Arc<Btf>> = Vec::new();
    let mut all_alloc_sizes: Vec<u64> = Vec::new();
    let started = std::time::Instant::now();
    tracing::debug!("cast_analysis: starting analyze_casts pipeline");
    for inner in iter_embedded_bpf_objects(&outer, bytes, bpf_objs_section) {
        let one_t0 = std::time::Instant::now();
        let (one, btf_for_obj, obj_alloc_sizes) = analyze_one_object_with_btf(inner);
        tracing::debug!(
            elapsed_ms = one_t0.elapsed().as_millis() as u64,
            casts = one.len(),
            "cast_analysis: analyze_one_object_with_btf finished"
        );
        cast_maps.push(Arc::new(one));
        all_alloc_sizes.extend_from_slice(&obj_alloc_sizes);
        if let Some(btf) = btf_for_obj {
            btfs.push(btf);
        }
    }
    let total_casts: usize = cast_maps.iter().map(|m| m.len()).sum();
    tracing::debug!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        casts = total_casts,
        btfs = btfs.len(),
        objects = cast_maps.len(),
        "cast_analysis: analyze_casts pipeline finished"
    );

    // Build the cross-BTF Fwd resolution index over every parsed
    // BTF. `build_fwd_index` walks each BTF's id space looking for
    // complete struct/union definitions and records `name ->
    // (btfs index, type id)`; first-write-wins on duplicate names
    // (see [`CastAnalysisOutput::fwd_index`]).
    let fwd_t0 = std::time::Instant::now();
    let fwd_index = build_fwd_index(&btfs);
    tracing::debug!(
        elapsed_us = fwd_t0.elapsed().as_micros() as u64,
        entries = fwd_index.len(),
        "cast_analysis: build_fwd_index finished"
    );

    // Demote to debug! when no casts were recovered: a clean
    // analyze on a scheduler with no typed pointers is a normal
    // outcome, not an event the operator needs to see at info!
    // (which would surface as a startup line on every test run).
    // Non-empty results stay at info! so the operator sees the
    // recovery count when it matters.
    if total_casts == 0 {
        tracing::debug!(
            casts = 0,
            "cast_analysis: recovered 0 typed pointers from scheduler"
        );
    } else {
        tracing::info!(
            casts = total_casts,
            "cast_analysis: recovered typed pointers from scheduler"
        );
    }
    all_alloc_sizes.sort_unstable();
    all_alloc_sizes.dedup();
    // For each captured alloc_size, try discover_payload_btf_id
    // against every embedded BTF. The embedded BTFs carry full
    // struct bodies that may be Fwd-only in the kernel's split BTF.
    // Store (size, struct_name) so the renderer can cross-BTF-resolve
    // by name at chase time.
    //
    // Walk each BTF's struct id-space exactly once via
    // [`enumerate_named_structs`] (consecutive-fail-cap to bail at the
    // dense table's end, [`crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE`]
    // backstops a sparse BTF). The cached `(size, name)` table is then
    // probed per alloc_size — replaces a quadratic per-size re-walk
    // AND the prior `take_while().last()` max-id discovery, which
    // bailed on the first id gap and undercounted on sparse split-BTF
    // tables.
    let mut alloc_size_types: Vec<(u64, String)> =
        Vec::with_capacity(all_alloc_sizes.len());
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let per_btf_structs: Vec<Vec<(u64, String)>> = btfs
        .iter()
        .map(|ebtf| enumerate_named_structs(ebtf))
        .collect();
    for &size in &all_alloc_sizes {
        if size == 0 {
            continue;
        }
        for (ebtf, structs) in btfs.iter().zip(per_btf_structs.iter()) {
            let choice = super::super::monitor::sdt_alloc::discover_payload_btf_id(
                ebtf,
                size as usize,
                "",
            );
            if choice.target_type_id != 0 {
                if let Ok(ty) = ebtf.resolve_type_by_id(choice.target_type_id)
                    && let Some(bt) = ty.as_btf_type()
                    && let Ok(name) = ebtf.resolve_name(bt)
                    && !name.is_empty()
                    && seen_names.insert(name.to_string())
                {
                    alloc_size_types.push((size, name.to_string()));
                }
                break;
            }
            // For ambiguous sizes, collect all scheduler-
            // convention candidates (names ending in _ctx,
            // _arena_ctx, or exact task_ctx). The cross-BTF
            // resolution at chase time disambiguates by name.
            for (struct_size, name) in structs {
                if *struct_size != size {
                    continue;
                }
                let dominated = name == "task_ctx"
                    || name.ends_with("_ctx")
                    || name.ends_with("_arena_ctx");
                if dominated && seen_names.insert(name.clone()) {
                    alloc_size_types.push((size, name.clone()));
                }
            }
        }
    }
    CastAnalysisOutput {
        cast_maps,
        btfs,
        fwd_index,
        alloc_size_types,
    }
}

fn parse_btfs_from_bytes(bytes: &[u8]) -> Vec<Arc<Btf>> {
    let outer = match goblin::elf::Elf::parse(bytes) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let bpf_objs_section = match find_section(&outer, ".bpf.objs") {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut btfs = Vec::new();
    for inner in iter_embedded_bpf_objects(&outer, bytes, bpf_objs_section) {
        let elf = match goblin::elf::Elf::parse(inner) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let btf_bytes = match find_section(&elf, ".BTF").and_then(|i| section_data(&elf, inner, i))
        {
            Some(b) => b,
            None => continue,
        };
        if let Ok(btf) = Btf::from_bytes(btf_bytes) {
            btfs.push(Arc::new(btf));
        }
    }
    btfs
}

/// Walk every parsed BTF and collect a `name -> FwdIndexEntry`
/// index of complete (`!is_fwd`) struct/union definitions for the
/// renderer's cross-BTF Fwd resolution path. First-write-wins —
/// see [`CastAnalysisOutput::fwd_index`] for the rationale.
///
/// The id-space walk uses the same `consecutive_fail` cap pattern
/// as [`crate::monitor::sdt_alloc::discover_payload_btf_id`]: real
/// BPF BTFs have dense id tables, so 256 consecutive failed
/// `resolve_type_by_id` calls is safe to treat as "table
/// exhausted". The hard ceiling
/// [`crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE`] backstops a
/// pathological / synthesized BTF.
///
/// Anonymous structs/unions are silently skipped (no name to key
/// the index entry on). Type kinds that are not Struct/Union are
/// also skipped — the index is consumed by the renderer's
/// [`crate::monitor::btf_render::peel_modifiers_resolving_fwd`]
/// extension, which only looks up Fwd terminals against this
/// table.
fn build_fwd_index(btfs: &[Arc<Btf>]) -> HashMap<String, FwdIndexEntry> {
    let mut out: HashMap<String, FwdIndexEntry> = HashMap::new();
    const CONSECUTIVE_FAIL_CAP: u32 = 256;
    for (idx, btf) in btfs.iter().enumerate() {
        let mut tid: u32 = 1;
        let mut consecutive_fail: u32 = 0;
        while tid < crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE {
            match btf.resolve_type_by_id(tid) {
                Ok(ty) => {
                    consecutive_fail = 0;
                    match &ty {
                        Type::Struct(s) | Type::Union(s) => {
                            if let Ok(name) = btf.resolve_name(s) {
                                if !name.is_empty() {
                                    out.entry(name).or_insert(FwdIndexEntry {
                                        btfs_idx: idx,
                                        type_id: tid,
                                    });
                                }
                            }
                        }
                        Type::Typedef(td) => {
                            if let Ok(td_name) = btf.resolve_name(td) {
                                if !td_name.is_empty() {
                                    if let Ok(pid) = <dyn btf_rs::BtfType>::get_type_id(td) {
                                        if let Ok(Type::Struct(s)) = btf.resolve_type_by_id(pid) {
                                            if btf.resolve_name(&s).map_or(true, |n| n.is_empty()) {
                                                let base = td_name.strip_suffix("_t")
                                                    .unwrap_or(&td_name);
                                                out.entry(base.to_string()).or_insert(FwdIndexEntry {
                                                    btfs_idx: idx,
                                                    type_id: pid,
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => {
                    consecutive_fail += 1;
                    if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                        break;
                    }
                }
            }
            tid += 1;
        }
    }
    out
}

/// Enumerate every named [`Type::Struct`] in one BTF as
/// `(struct_size, struct_name)` pairs.
///
/// Mirrors the consecutive-fail-cap pattern from [`build_fwd_index`]
/// and [`crate::monitor::sdt_alloc::discover_payload_btf_id`]: real
/// BPF BTFs have dense id tables, so 256 consecutive `resolve_type_by_id`
/// failures is safe to treat as "table exhausted"; the hard ceiling
/// [`crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE`] backstops a
/// pathological / sparse BTF id space.
///
/// Anonymous structs (empty resolved name) and non-Struct kinds are
/// skipped — the caller looks up by name and only cares about struct
/// kinds.
fn enumerate_named_structs(btf: &Btf) -> Vec<(u64, String)> {
    const CONSECUTIVE_FAIL_CAP: u32 = 256;
    let mut out: Vec<(u64, String)> = Vec::new();
    let mut tid: u32 = 1;
    let mut consecutive_fail: u32 = 0;
    while tid < crate::monitor::sdt_alloc::MAX_BTF_ID_PROBE {
        match btf.resolve_type_by_id(tid) {
            Ok(ty) => {
                consecutive_fail = 0;
                if let Type::Struct(s) = &ty
                    && let Ok(name) = btf.resolve_name(s)
                    && !name.is_empty()
                {
                    out.push((s.size() as u64, name));
                }
            }
            Err(_) => {
                consecutive_fail += 1;
                if consecutive_fail >= CONSECUTIVE_FAIL_CAP {
                    break;
                }
            }
        }
        tid += 1;
    }
    out
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
        // STT_OBJECT (data symbol); section index match ties the
        // symbol to .bpf.objs. SHN_UNDEF / SHN_ABS / SHN_COMMON are
        // below the section-header range so the equality test
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

/// Run cast analysis on one embedded BPF object's bytes and
/// return the parsed BTF alongside the cast map.
///
/// The bytes are themselves an ELF (the BPF object); parse it, extract
/// the BTF, the `.BTF.ext`-derived [`FuncEntry`] table, and the
/// concatenated instruction stream, then call [`analyze_casts`].
///
/// The parsed BTF is returned wrapped in `Arc` so the caller can
/// retain it across the dump pass without copying. `None` for the
/// BTF position indicates a parse failure or an inner ELF without
/// a `.BTF` section — the cast map is still returned (empty in that
/// case) so the merger keeps working without distinguishing the
/// no-BTF inner from one with no recovered casts.
fn analyze_one_object_with_btf(obj_bytes: &[u8]) -> (CastMap, Option<Arc<Btf>>, Vec<u64>) {
    let elf = match goblin::elf::Elf::parse(obj_bytes) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(
                error = %e,
                "cast_analysis: parse inner BPF object ELF failed"
            );
            return (CastMap::new(), None, Vec::new());
        }
    };

    // .BTF is mandatory — no BTF, no struct/field resolution, no
    // analysis output the renderer can use.
    let btf_bytes = match find_section(&elf, ".BTF").and_then(|i| section_data(&elf, obj_bytes, i))
    {
        Some(b) => b,
        None => {
            tracing::debug!("cast_analysis: inner ELF has no .BTF section");
            return (CastMap::new(), None, Vec::new());
        }
    };
    let btf = match Btf::from_bytes(btf_bytes) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                error = ?e,
                "cast_analysis: parse .BTF failed"
            );
            return (CastMap::new(), None, Vec::new());
        }
    };
    let btf = Arc::new(btf);

    // Instruction sections in section-header order: every
    // SHF_EXECINSTR-flagged PROGBITS section. Concatenating in this
    // order matches how `.BTF.ext` records reference them — each
    // record's `insn_off` is byte-relative to its OWN section, so we
    // record each section's base index in the concatenated stream and
    // translate per-record below.
    // Pre-walk to size the concatenated instruction vec — saves a
    // sequence of growth-and-copy reallocations on schedulers with
    // large BPF programs (a single scx scheduler easily hits tens of
    // thousands of instructions). Each `chunks_exact(BPF_INSN_SIZE)`
    // pass below pushes `data.len() / BPF_INSN_SIZE` instructions.
    let total_insns: usize = elf
        .section_headers
        .iter()
        .enumerate()
        .filter(|(_, sh)| {
            sh.sh_type == goblin::elf::section_header::SHT_PROGBITS
                && sh.sh_flags & u64::from(goblin::elf::section_header::SHF_EXECINSTR) != 0
        })
        .filter_map(|(idx, _)| section_data(&elf, obj_bytes, idx))
        .filter(|d| d.len().is_multiple_of(BPF_INSN_SIZE))
        .map(|d| d.len() / BPF_INSN_SIZE)
        .sum();
    let mut text_concat: Vec<BpfInsn> = Vec::with_capacity(total_insns);
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
        // Even on empty text we still return the parsed `Btf` so
        // the cross-BTF Fwd index can pick up its struct/union
        // definitions: a header-only object that contributes no
        // analyzer findings can still expose a complete sibling
        // for a Fwd in another object.
        return (CastMap::new(), Some(btf), Vec::new());
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

    // Pre-relocation .bpf.o files (the production path: an embedded
    // BPF object inside a scheduler binary that has not been through
    // libbpf's RELO_EXTERN_CALL handler yet) emit kfunc call sites
    // as `BPF_JMP|BPF_CALL` with `src_reg = BPF_PSEUDO_CALL = 1` and
    // `imm = -1`. The cast analyzer's `handle_kfunc_call` keys on
    // `src_reg = BPF_PSEUDO_KFUNC_CALL = 2` + `imm = btf_id`, so
    // every pre-relocation kfunc call is invisible to it. Patching
    // mirrors what libbpf does at load time
    // (`bpf_object__relocate_data`'s `RELO_EXTERN_CALL` arm):
    // walk the ELF relocation entries that target each program text
    // section, resolve the symbol name to a `BTF_KIND_FUNC` of
    // extern linkage in the program's own BTF, then rewrite both
    // `src_reg` and `imm` on the call instruction. After patching,
    // `analyze_casts` sees the kfunc id and `handle_kfunc_call`
    // recovers the return type — typically `Ptr -> Struct` for
    // pointer-returning kfuncs (`bpf_task_acquire`,
    // `bpf_cpumask_first`, …), which seeds R0 so the next STX of
    // R0 into a u64 slot records a `(parent, off) -> target,
    // AddrSpace::Kernel` cast entry.
    let patch_t0 = std::time::Instant::now();
    patch_kfunc_calls(&mut text_concat, btf.as_ref(), &elf, &section_bases);
    tracing::debug!(
        elapsed_us = patch_t0.elapsed().as_micros() as u64,
        insns = text_concat.len(),
        "cast_analysis: patch_kfunc_calls finished"
    );

    // BPF-to-BPF subprog call patching. libbpf-rs's Linker leaves
    // every global subprog call as `BPF_PSEUDO_CALL` with
    // `imm = -1`, paired with a `STT_FUNC` relocation. The cast
    // analyzer's `caller_arg_types` mechanism (see
    // [`crate::monitor::cast_analysis::Analyzer::analyze`])
    // computes `callee_pc = pc + 1 + insn.imm`, so an unpatched
    // `imm == -1` resolves to `pc` (the call site itself) and
    // poisons the lookup table with bogus entries. Patching
    // mirrors what libbpf does at load time
    // (`bpf_object__reloc_code` in tools/lib/bpf/libbpf.c):
    // `sub_insn_idx = sym.st_value/8 + insn.imm + 1`, with
    // `insn.imm = -1` for the global-subprog case. We rewrite
    // the placeholder `imm` in place so the analyzer's
    // `pc + 1 + imm` computation lands on the correct callee
    // entry PC in the concatenated text stream.
    let subprog_patch_t0 = std::time::Instant::now();
    patch_subprog_calls(&mut text_concat, &elf, &section_bases);
    tracing::debug!(
        elapsed_us = subprog_patch_t0.elapsed().as_micros() as u64,
        insns = text_concat.len(),
        "cast_analysis: patch_subprog_calls finished"
    );

    // BSS / DATA / RODATA datasec annotations: walk every
    // relocation section in the inner ELF and emit a
    // `DatasecPointer` per `R_BPF_64_64` reloc that targets a
    // section the program BTF exposes as a `BTF_KIND_DATASEC`.
    // The annotation gives the analyzer's `BPF_LD_IMM64` arm the
    // missing `(datasec_id, base_offset)` pair: libbpf's runtime
    // relocator would set `src_reg = BPF_PSEUDO_MAP_VALUE` and
    // patch the imm into a map fd, but the host-side cast loader
    // sees pre-relocation bytecode where the imm is the per-
    // variable byte offset within the section. We translate that
    // directly into the analyzer's `RegState::DatasecPointer`
    // representation so subsequent STX/LDX through the LD_IMM64
    // destination resolve to the right `VarSecinfo` entry via
    // `struct_member_at`.
    let datasec_t0 = std::time::Instant::now();
    let datasec_pointers = build_datasec_pointers(&text_concat, btf.as_ref(), &elf, &section_bases);
    tracing::debug!(
        elapsed_us = datasec_t0.elapsed().as_micros() as u64,
        datasec_pointers = datasec_pointers.len(),
        "cast_analysis: build_datasec_pointers finished"
    );

    // Allocator-return seeds: walk every relocation section to find
    // `BPF_PSEUDO_CALL` sites whose resolved subprog name matches
    // the arena-allocator allowlist (e.g. `scx_static_alloc_internal`).
    // Emit one [`SubprogReturn`] per matching call site so the
    // analyzer's `BPF_OP_CALL` arm tags R0 as
    // [`RegState::ArenaU64FromAlloc`] after the standard R0..=R5
    // clobber. The subsequent STX of the tagged R0 (or its
    // propagation through MOV / stack spill / LDX of an
    // already-arena-tagged slot) records `(parent, off)` as an
    // Arena cast finding via the new STX-flow path. See
    // [`build_subprog_returns`] for the relocation walk.
    let alloc_seed_t0 = std::time::Instant::now();
    let subprog_returns = build_subprog_returns(&text_concat, &elf, &section_bases);
    tracing::debug!(
        elapsed_us = alloc_seed_t0.elapsed().as_micros() as u64,
        subprog_returns = subprog_returns.len(),
        "cast_analysis: build_subprog_returns finished"
    );

    let analyze_t0 = std::time::Instant::now();
    let result = analyze_casts(
        &text_concat,
        btf.as_ref(),
        &[],
        &func_entries,
        &datasec_pointers,
        &subprog_returns,
    );
    tracing::debug!(
        elapsed_ms = analyze_t0.elapsed().as_millis() as u64,
        casts = result.len(),
        "cast_analysis: analyze_casts inner pass finished"
    );
    let mut alloc_sizes: Vec<u64> = subprog_returns
        .iter()
        .filter_map(|sr| sr.alloc_size)
        .collect();
    alloc_sizes.sort_unstable();
    alloc_sizes.dedup();
    (result, Some(btf), alloc_sizes)
}

/// Walk every ELF relocation section in `elf` whose target section
/// is indexed by `section_bases`, validate each relocation's
/// `r_offset`, and yield surviving entries paired with the
/// translated instruction index in the concatenated text stream.
///
/// Used by [`patch_kfunc_calls`], [`build_subprog_returns`], and
/// [`build_datasec_pointers`] — every consumer needs the same
/// "rel section → target program text section → per-reloc
/// `insn_idx`" pipeline. Centralising it here removes the
/// rel-section / bounds / alignment preamble from each consumer
/// and guarantees identical gating across all three.
///
/// # Filtering rules (shared with all consumers)
///
/// 1. The rel section's `sh_info` must point at a real section
///    header. Out-of-range `sh_info` is silently skipped.
/// 2. The target section must appear in `section_bases` — only
///    program text sections we concatenated into `text_concat` are
///    eligible. A rel section targeting `.maps`, `.BTF.ext`, or any
///    non-text section yields no items.
/// 3. The target section header must resolve so we can read its
///    byte size (`sh_size`); a missing header rejects the section.
/// 4. Per-reloc gates: `r_offset` must be a multiple of
///    [`BPF_INSN_SIZE`] (BPF instructions are 8-byte aligned) and
///    strictly less than the target section's byte size. Failures
///    drop the individual relocation.
///
/// Each surviving item is `(insn_idx, reloc)` where `insn_idx`
/// equals `base + r_offset / BPF_INSN_SIZE` (saturating-add against
/// the unlikely-but-possible overflow of a corrupted ELF). The
/// caller then fetches the instruction from `text_concat` (mutably
/// or immutably as needed) and applies its own consumer-specific
/// gates (call opcode, src_reg, datasec lookup, …).
fn iter_text_relocs<'a, 'elf: 'a>(
    elf: &'a goblin::elf::Elf<'elf>,
    section_bases: &'a HashMap<u32, usize>,
) -> impl Iterator<Item = (usize, goblin::elf::Reloc)> + 'a {
    elf.shdr_relocs
        .iter()
        .flat_map(move |(rel_section_idx, reloc_section)| {
            // Resolve which section the relocations target.
            let target_section_idx = elf.section_headers.get(*rel_section_idx).map(|h| h.sh_info);
            // Only program text sections appear in `section_bases`.
            let scope = target_section_idx.and_then(|idx| {
                let base = *section_bases.get(&idx)?;
                let sh = elf.section_headers.get(idx as usize)?;
                Some((base, sh.sh_size as usize))
            });
            // `into_iter` collapses `Option<I>` to an iterator (one
            // pass when `Some`, empty when `None`), so the outer
            // `flat_map` sees the correct shape regardless of
            // whether the rel section was in scope.
            scope
                .into_iter()
                .flat_map(move |(base, section_byte_size)| {
                    reloc_section.iter().filter_map(move |reloc| {
                        let off = reloc.r_offset as usize;
                        if !off.is_multiple_of(BPF_INSN_SIZE) {
                            return None;
                        }
                        if off >= section_byte_size {
                            return None;
                        }
                        let insn_idx = base.saturating_add(off / BPF_INSN_SIZE);
                        Some((insn_idx, reloc))
                    })
                })
        })
}

/// Names of in-tree BPF subprograms whose return values are arena
/// virtual addresses stored in `u64` slots. The cast analyzer's
/// STX-flow path tags any slot the returned value is stored into as
/// an Arena cast finding (resolved via the renderer's
/// [`crate::monitor::btf_render::MemReader::resolve_arena_type`]
/// bridge at chase time).
///
/// Order is alphabetical for readability — the allowlist is
/// consulted by linear scan in [`build_subprog_returns`] (small N,
/// no perf concern). Each entry must be `__always_inline`-d in the
/// scheduler source for the analyzer to see the call site at the
/// stash location; non-inlined helpers move the `STX` of the
/// returned R0 into the helper's own frame (R0 is clobbered at the
/// caller's call site), so the analyzer never sees the tag flow
/// across the call boundary. The F4 mitigation surfaces a warn at
/// finalize when arena STX evidence is present but no LDX→cast
/// chain landed for any slot, prompting operators to mark missing
/// helpers `__always_inline`.
const ALLOC_SUBPROG_NAMES: &[&str] = &[
    // sdt_alloc lib allocator for per-task / per-cgroup contexts
    // (lib/sdt_alloc.bpf.c). Distinct from `scx_static_alloc_internal`
    // — sdt_alloc adds a per-allocation header (`union sdt_id`)
    // before the payload, but the returned u64 is still an arena
    // VA suitable for STX-flow tagging.
    "scx_alloc_internal",
    // scx-shared static allocator that returns a u64 carrying an
    // arena VA with NO per-allocation header (the slot just holds
    // the start of an arbitrary-typed payload, e.g. `struct
    // scx_cgroup_ctx`). Drives the deferred-resolve arena cast
    // path: the renderer's `resolve_arena_type` bridge resolves
    // the payload type at chase time.
    "scx_static_alloc_internal",
    // The kernel kfunc `bpf_arena_alloc_pages` is intentionally
    // NOT in this allowlist — it is a kfunc (`SHN_UNDEF` /
    // `STT_NOTYPE`), not a subprog, so every gate in
    // [`build_subprog_returns`] (`STT_FUNC`, non-`SHN_UNDEF`,
    // `BPF_PSEUDO_CALL`) rejects it. Arena allocator kfuncs are
    // tagged on the kfunc-side allowlist
    // [`crate::monitor::cast_analysis::ARENA_ALLOC_KFUNC_NAMES`]
    // consulted by [`crate::monitor::cast_analysis::Analyzer::handle_kfunc_call`].
    // Putting `bpf_arena_alloc_pages` here would have been dead code
    // — it failed every gate silently — but kept the wrong impression
    // that subprog detection covered kernel arena allocation.
];

/// Walk every ELF relocation section in `elf` and emit one
/// [`SubprogReturn`] per `BPF_PSEUDO_CALL` site whose resolved
/// subprog name matches the arena-allocator allowlist (see
/// [`ALLOC_SUBPROG_NAMES`]).
///
/// Pre-relocation `.bpf.o` (the form embedded inside an scx-built
/// scheduler binary's `.bpf.objs` section) emits BPF-to-BPF calls
/// to in-tree library subprograms as:
///
/// ```text
///     code = BPF_JMP|BPF_CALL = 0x85
///     dst_reg = 0, src_reg = BPF_PSEUDO_CALL = 1
///     off = 0
///     imm = pc-relative offset to the subprog's first insn
/// ```
///
/// paired with an ELF relocation entry at the call's byte offset
/// pointing to the subprog's `STT_FUNC` symbol. Unlike kfunc
/// calls (`SHN_UNDEF`), library subprogs are linked into the same
/// program text section (or a sibling section with `SHF_EXECINSTR`)
/// — the reloc's symbol's `st_shndx` is non-`SHN_UNDEF` and
/// `st_type == STT_FUNC`. The symbol's name is the subprog's name
/// in the program BTF (clang preserves the C identifier).
///
/// The function does NOT patch any instruction; it only records the
/// call PC for the analyzer to consume. Distinct from
/// [`patch_kfunc_calls`] which rewrites kfunc call sites in place.
///
/// # Errors
///
/// Never fails. Symbol resolve failures, relocations on non-call
/// instructions, missing subprog names — all silent no-ops. The
/// analyzer falls through to the existing shape-inference path.
fn build_subprog_returns(
    text_concat: &[BpfInsn],
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) -> Vec<SubprogReturn> {
    let mut out: Vec<SubprogReturn> = Vec::new();
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item is
    // a relocation that targets a known program text section at an
    // 8-byte-aligned, in-bounds offset; the call-site / symbol /
    // allowlist gates below are subprog-specific.
    for (insn_idx, reloc) in iter_text_relocs(elf, section_bases) {
        let Some(insn) = text_concat.get(insn_idx) else {
            continue;
        };
        // Gate 1: the instruction must be a BPF call site.
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            continue;
        }
        // Gate 2: the call must be a `BPF_PSEUDO_CALL`. Kfunc
        // calls (`BPF_PSEUDO_KFUNC_CALL`) and helper calls
        // (`src_reg == 0`) are not subprog calls.
        if insn.src_reg() != BPF_PSEUDO_CALL {
            continue;
        }
        // Resolve the symbol → name. The symbol must be `STT_FUNC`
        // with a defined section (`st_shndx != SHN_UNDEF`) — that's
        // the in-tree-subprog shape. Extern (kfunc) callsites have
        // `st_shndx == SHN_UNDEF` and are handled by
        // [`patch_kfunc_calls`] separately.
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        const STT_FUNC: u8 = goblin::elf::sym::STT_FUNC;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx == SHN_UNDEF {
            continue;
        }
        if sym.st_type() != STT_FUNC {
            continue;
        }
        let name = match elf.strtab.get_at(sym.st_name) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        // Allowlist match: linear scan over the small list. The
        // names are exact (no prefix / glob); a future change to
        // allow prefix matching would require a dedicated test for
        // cross-allocator name collisions (e.g.
        // `scx_static_alloc_internal_v2`).
        if !ALLOC_SUBPROG_NAMES.contains(&name) {
            continue;
        }
        // For `scx_static_alloc_internal` callers, recover the
        // `size` argument from R1 by scanning backward from the call
        // PC for the most recent `BPF_MOV64_IMM r1, <imm>`. The bump
        // allocator emits no per-slot header, so the renderer's
        // [`crate::monitor::btf_render::MemReader::resolve_arena_type`]
        // bridge has no entry to resolve the payload type id from —
        // size-based BTF matching via
        // [`crate::monitor::sdt_alloc::discover_payload_btf_id`] is
        // the only resolution path. The captured size threads from
        // [`SubprogReturn::alloc_size`] all the way to
        // [`crate::monitor::cast_analysis::CastHit::alloc_size`].
        //
        // For other allocators (e.g. `scx_alloc_internal`) the
        // bridge handles resolution via the per-slot header, so
        // the captured size is not needed and we leave
        // `alloc_size: None` to keep the chase on the bridge path.
        //
        // The lookback is bounded at [`ALLOC_SIZE_LOOKBACK`]
        // instructions: clang's allocator inlining emits the
        // `mov r1, <imm>` immediately before the call in real
        // schedulers, but a small budget tolerates conservative
        // codegen (a constant rematerialized into a different
        // register, then moved into r1) without opening the door
        // to spurious matches from unrelated MOVs many
        // instructions back. A failed lookback yields
        // `alloc_size: None`, falling back to the bridge or the
        // skip-with-reason chase outcome.
        let alloc_size = if name == "scx_static_alloc_internal" {
            recover_alloc_size_from_r1(text_concat, insn_idx)
        } else {
            None
        };
        out.push(SubprogReturn {
            insn_offset: insn_idx,
            alloc_size,
        });
    }
    out
}

/// Maximum instructions [`recover_alloc_size_from_r1`] scans backward
/// from a `scx_static_alloc_internal` call site looking for the
/// `BPF_MOV64_IMM r1, <imm>` that materialised R1. Real schedulers
/// emit the MOV adjacent to the call (clang inlines the helper, so
/// the call is preceded by `r1 = sizeof(...)`); 20 instructions is a
/// generous budget that covers a few intervening setup ops without
/// reaching back into unrelated control flow.
const ALLOC_SIZE_LOOKBACK: usize = 20;

/// `BPF_ALU64 | BPF_MOV | BPF_K` opcode byte (`= 0xb7`). Sets
/// `dst_reg = imm` (sign-extended to 64 bits). See linux uapi
/// `bpf.h` and `kernel/bpf/verifier.c` `check_alu_op`. The
/// host-side loader uses this to recognize the
/// `mov rN, <imm>` instructions that clang emits for
/// argument-setup before a BPF-to-BPF subprog call.
const BPF_MOV64_IMM_CODE: u8 = (libbpf_rs::libbpf_sys::BPF_ALU64
    | libbpf_rs::libbpf_sys::BPF_MOV
    | libbpf_rs::libbpf_sys::BPF_K) as u8;

/// Scan backward from `call_pc` in `text` looking for the most recent
/// `BPF_MOV64_IMM r1, <imm>` and return the immediate as a `u64`.
/// Returns `None` when no matching instruction is found within
/// [`ALLOC_SIZE_LOOKBACK`] instructions, when `call_pc` is `0`
/// (no predecessors to scan), or when `call_pc` is out of bounds.
///
/// The scanner stops at the first match — the most recent write to
/// R1 is the one that survived to the call site. Other instructions
/// (ALU on R1, LDX into R1, helper-call clobbers) are NOT modeled
/// here: the host-side loader is intentionally simpler than the
/// analyzer's full register-state walk, and a complex sequence
/// elides into `None` (lookback misses) rather than a wrong
/// capture. False negatives surface as `alloc_size: None` — the
/// chase falls back to the bridge (no static-alloc match), which
/// is the safe direction.
///
/// `imm` is sign-extended to `u64` via the `i32 -> i64 -> u64`
/// chain so a negative `i32` would surface as a very large `u64`.
/// Real `sizeof` arguments are non-negative; the analyzer's
/// downstream chase (`discover_payload_btf_id`) returns
/// `target_type_id == 0` for impossible payload sizes, so a
/// pathological negative `imm` cannot misrender — it falls back
/// to the bridge or skips.
fn recover_alloc_size_from_r1(text: &[BpfInsn], call_pc: usize) -> Option<u64> {
    if call_pc == 0 {
        return None;
    }
    let start = call_pc.saturating_sub(ALLOC_SIZE_LOOKBACK);
    // Walk from `call_pc - 1` down to `start` (inclusive), stopping
    // at the first MOV r1, imm.
    let mut idx = call_pc;
    while idx > start {
        idx -= 1;
        let Some(insn) = text.get(idx) else {
            return None;
        };
        if insn.code == BPF_MOV64_IMM_CODE && insn.dst_reg() == 1 {
            return Some(insn.imm as i64 as u64);
        }
    }
    None
}

/// Walk every ELF relocation section in `elf` and emit a
/// [`DatasecPointer`] for each `R_BPF_64_64` reloc that targets a
/// section the program BTF exposes as a `BTF_KIND_DATASEC`
/// (`.bss`, `.data`, `.rodata`, `.data.<name>`, …).
///
/// Pre-relocation `.bpf.o` (the form embedded inside an scx-built
/// scheduler binary's `.bpf.objs` section) emits `BPF_LD_IMM64`
/// references to global variables in `.bss` / `.data` / `.rodata`
/// with `src_reg = 0`; the relocation entry is the only host-side
/// evidence that the LD_IMM64 targets a specific section. Each
/// reloc's `r_offset` (byte offset within the targeted text
/// section) divided by [`BPF_INSN_SIZE`] gives the instruction PC
/// in `text_concat`. The reloc's symbol resolves either to the
/// section symbol itself (`STT_SECTION`, `st_value == 0`) or to a
/// regular `STT_OBJECT` data symbol whose `st_shndx` points at
/// the section. Either way, the section's name keys the BTF
/// lookup that finds the matching `BTF_KIND_DATASEC` id.
///
/// `base_offset` resolution mirrors libbpf's relocation logic.
/// For SHT_REL (the BPF convention — clang emits SHT_REL, not
/// SHT_RELA, for BPF object files), `r_addend` is absent; the
/// offset comes from `LD_IMM64 insn.imm + sym.st_value`. The
/// LD_IMM64's pre-relocation `imm` field carries the per-variable
/// byte offset within the section (clang emits this for
/// `STT_SECTION` symbols). For `STT_OBJECT` symbols clang emits
/// `imm == 0` and the offset comes from `sym.st_value` (the
/// object symbol's address within its section). The function
/// adds both contributions so both clang patterns produce
/// identical annotations.
///
/// # What gets emitted
///
/// - `R_BPF_64_64` (numeric `r_type == 1`): the LD_IMM64-on-text
///   relocation libbpf rewrites to `BPF_PSEUDO_MAP_VALUE`. Other
///   reloc types are not LD_IMM64-on-text and produce no
///   annotation.
/// - The instruction at the resolved PC must be `BPF_LD_IMM64`
///   (`code == BPF_LD | BPF_DW | BPF_IMM = 0x18`). A reloc on a
///   non-LD_IMM64 instruction is malformed input — drop silently.
/// - The target section must resolve to a `BTF_KIND_DATASEC` in
///   the program BTF. `.text` (executable), `.maps` (BPF map
///   definitions, exposed as a different BTF shape), and `.BTF`
///   itself are not datasecs and produce no annotation.
///
/// # Errors
///
/// Never fails. A relocation we cannot parse, a symbol we cannot
/// resolve, a section name absent from BTF, an out-of-range PC —
/// every failure path produces a silent no-op. False negatives
/// are safe; the analyzer leaves the corresponding LD_IMM64
/// destination as Unknown, which falls through to the original
/// pre-integration u64 counter rendering.
fn build_datasec_pointers(
    text_concat: &[BpfInsn],
    btf: &Btf,
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) -> Vec<DatasecPointer> {
    // R_BPF_64_64 = 1 per linux `tools/lib/bpf/libbpf_internal.h`.
    // goblin's reloc constants table does not expose BPF reloc
    // types, so the numeric value is inlined here. Same gating
    // libbpf applies in `bpf_program__resolve_map_value_relos`.
    const R_BPF_64_64: u32 = 1;
    // BPF_LD | BPF_DW | BPF_IMM opcode byte (= 0x18 per linux
    // uapi `bpf.h`). Used to gate the relocation: a reloc against
    // an instruction whose opcode is not LD_IMM64 must not
    // produce a datasec annotation, since the analyzer's BPF_LD
    // arm only applies datasec annotations on this exact opcode.
    let bpf_ld_imm64_code: u8 = (libbpf_rs::libbpf_sys::BPF_LD
        | libbpf_rs::libbpf_sys::BPF_DW
        | libbpf_rs::libbpf_sys::BPF_IMM) as u8;

    let mut out: Vec<DatasecPointer> = Vec::new();
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item
    // is a relocation that targets a known program text section
    // at an 8-byte-aligned, in-bounds offset; the reloc-type /
    // opcode / symbol / BTF-lookup gates below are datasec-specific.
    for (insn_pc, reloc) in iter_text_relocs(elf, section_bases) {
        // Gate 1: only `R_BPF_64_64` produces a datasec annotation.
        // Other reloc types touch different instruction kinds
        // (call sites, ABS32/64 data references) that are not
        // LD_IMM64.
        if reloc.r_type != R_BPF_64_64 {
            continue;
        }
        // Gate 2: the reloc must target a `BPF_LD_IMM64`
        // instruction.
        let Some(insn) = text_concat.get(insn_pc) else {
            continue;
        };
        if insn.code != bpf_ld_imm64_code {
            continue;
        }
        // Resolve the symbol. `r_sym` indexes the ELF symbol
        // table; the symbol's section (`st_shndx`) identifies
        // the target section, and `st_value` contributes to
        // the base offset for `STT_OBJECT` symbols.
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        // SHN_UNDEF / SHN_ABS / SHN_COMMON: symbols not bound to a
        // real section index. None can refer to a datasec section;
        // drop.
        const SHN_UNDEF: usize = 0;
        const SHN_ABS: usize = 0xFFF1;
        const SHN_COMMON: usize = 0xFFF2;
        if sym.st_shndx == SHN_UNDEF || sym.st_shndx == SHN_ABS || sym.st_shndx == SHN_COMMON {
            continue;
        }
        let target_sec_idx = sym.st_shndx;
        // Resolve the target section's name via the ELF section
        // header strtab.
        let target_sh_for_name = match elf.section_headers.get(target_sec_idx) {
            Some(s) => s,
            None => continue,
        };
        let sec_name = match elf.shdr_strtab.get_at(target_sh_for_name.sh_name) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        // Resolve the section name to a `BTF_KIND_DATASEC` id.
        // `Btf::resolve_ids_by_name` returns every id sharing the
        // name; the helper filters for the Datasec kind.
        let Some(datasec_id) = find_datasec_btf_id(btf, sec_name) else {
            continue;
        };
        // Compute base_offset: pre-relocation LD_IMM64 imm (per-
        // variable offset for `STT_SECTION` syms) plus
        // `sym.st_value` (per-object offset for `STT_OBJECT` syms).
        // Both contributions are non-negative in well-formed input;
        // checked_add guards against overflow that could only arise
        // from a corrupt ELF.
        let imm_off = if insn.imm < 0 { 0 } else { insn.imm as u32 };
        if sym.st_value > u32::MAX as u64 {
            continue;
        }
        let sym_off = sym.st_value as u32;
        let Some(base_offset) = imm_off.checked_add(sym_off) else {
            continue;
        };
        out.push(DatasecPointer {
            insn_offset: insn_pc,
            datasec_type_id: datasec_id,
            base_offset,
        });
    }
    out
}

/// Find the `BTF_KIND_DATASEC` id whose name matches `name`. Returns
/// the first matching id; `None` if no Datasec by that name is
/// indexed in the program BTF.
///
/// Section names are unique per BTF (every `.bss` / `.data` /
/// `.rodata` / `.data.<name>` produces exactly one DATASEC), so
/// the first hit is the only hit in well-formed input. Mirrors
/// the name-keyed lookup style of [`find_extern_func_btf_id`].
fn find_datasec_btf_id(btf: &Btf, name: &str) -> Option<u32> {
    let ids = btf.resolve_ids_by_name(name).ok()?;
    for id in ids {
        let Ok(ty) = btf.resolve_type_by_id(id) else {
            continue;
        };
        if let Type::Datasec(_) = ty {
            return Some(id);
        }
    }
    None
}

/// Mirror libbpf's `RELO_EXTERN_CALL` handler on the host side.
///
/// In a pre-relocation `.bpf.o` (the form embedded inside an scx-
/// built scheduler binary's `.bpf.objs` section), every kfunc call
/// site is emitted by clang as:
///
/// ```text
///     code = BPF_JMP|BPF_CALL = 0x85
///     dst_reg = 0, src_reg = BPF_PSEUDO_CALL = 1
///     off = 0
///     imm = -1                ; placeholder filled in by libbpf
/// ```
///
/// paired with an ELF relocation entry at the call's byte offset
/// pointing to an extern symbol (`STT_NOTYPE`, `STB_GLOBAL` or
/// `STB_WEAK`, `st_shndx == SHN_UNDEF`). At kernel-load time, libbpf
/// resolves the symbol's BTF id (the program's own
/// `BTF_KIND_FUNC` whose name matches the symbol) to the kernel's
/// kfunc BTF id, then rewrites `src_reg` to `BPF_PSEUDO_KFUNC_CALL =
/// 2` and `imm` to the resolved id (libbpf
/// `bpf_object__relocate_data`'s `RELO_EXTERN_CALL` arm).
///
/// The cast analyzer never runs at kernel-load time — it operates
/// purely on the on-disk binary. So this function performs the same
/// rewrite host-side, except that the BTF id we patch in is the
/// program-BTF id of the extern `BTF_KIND_FUNC`, not the running
/// kernel's id. That suffices for cast analysis: the analyzer's
/// [`crate::monitor::cast_analysis::Analyzer::handle_kfunc_call`]
/// resolves `imm` against the same program BTF (it has no kernel
/// BTF here), peels `Func -> FuncProto -> return type` through
/// `Ptr -> Struct/Union`, and types R0 accordingly. The kfunc's
/// program-BTF Func entry shares the same FuncProto a kernel-BTF
/// Func entry would, so the return type is the same.
///
/// # Symbol → BTF FUNC id mapping
///
/// libbpf walks the `.ksyms` `BTF_KIND_DATASEC`, whose
/// [`btf_rs::VarSecinfo`] entries point to the per-kfunc
/// `BTF_KIND_FUNC` types (with `BTF_FUNC_EXTERN` linkage). We don't
/// need to descend the DATASEC explicitly: every FUNC referenced by
/// `.ksyms` is also indexed in the program BTF's name → id map (see
/// `btf-rs::BtfObj::resolve_ids_by_name`), so a name-keyed lookup is
/// enough. We still filter the result to FUNCs with extern linkage
/// to avoid colliding with a same-named static helper that happens
/// to share the symbol name.
///
/// # What gets patched
///
/// - The instruction must be a `BPF_JMP|BPF_CALL` (code byte
///   `0x85`).
/// - The current `src_reg` must be `BPF_PSEUDO_CALL` (the clang-
///   emitted form). If it is already `BPF_PSEUDO_KFUNC_CALL` (post-
///   relocation form, observed when the scheduler binary embeds a
///   pre-loaded BPF object) we leave it alone — the imm already
///   carries the kernel-BTF id, which means nothing in the program
///   BTF.
/// - The current `imm` must be `-1` (the placeholder libbpf fills
///   in). A non-`-1` imm would mean clang resolved this call to a
///   subprog (BPF-to-BPF call), and we must not steal those.
///
/// All three conditions plus the name-resolves-to-extern-FUNC check
/// must hold before any byte is patched. Anything else is a no-op,
/// preserving the cast analyzer's "false negative is safe; false
/// positive is not" stance.
///
/// # Errors
///
/// This function never fails. An ELF without relocation sections, a
/// relocation pointing into a section we did not concatenate, a
/// symbol we cannot resolve, a name that does not map to an extern
/// FUNC, or a bounds-violating reloc offset all produce silent
/// no-ops. The cast map ends up identical to the pre-patching world
/// for those instructions.
fn patch_kfunc_calls(
    text_concat: &mut [BpfInsn],
    btf: &Btf,
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) {
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item
    // is a relocation that targets a known program text section
    // at an 8-byte-aligned, in-bounds offset; the kfunc-specific
    // gates (call opcode, imm == -1, BPF_PSEUDO_CALL src_reg,
    // extern NOTYPE symbol, BTF Func/extern resolve) are applied
    // here. The iterator borrows `elf` and `section_bases`
    // immutably while we take a disjoint mutable borrow on
    // `text_concat`.
    for (insn_idx, reloc) in iter_text_relocs(elf, section_bases) {
        let Some(insn) = text_concat.get_mut(insn_idx) else {
            continue;
        };
        // Gate 1: the instruction must be a BPF call site.
        // `BPF_JMP|BPF_CALL` = `0x05 | 0x80 = 0x85`. Anything
        // else (LD_IMM64 referencing a typeless ksym, BTF data
        // reloc, …) leaves the slot alone.
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            continue;
        }
        // Gate 2: `imm` must be the libbpf placeholder. A non-`-1`
        // imm means clang already resolved this call to a same-
        // section subprog (BPF_PSEUDO_CALL with a pc-relative imm),
        // and patching it as a kfunc would corrupt subprog dispatch
        // in the analyzer's eyes.
        if insn.imm != -1 {
            continue;
        }
        // Gate 3: src_reg must be the clang-emitted
        // `BPF_PSEUDO_CALL` (1). If the embedded object has
        // already been through libbpf's relocation pass (rare;
        // observed only when a scheduler binary captures a
        // post-load object), `src_reg` is already
        // `BPF_PSEUDO_KFUNC_CALL` and `imm` is the kernel BTF id —
        // we must not overwrite the kernel id with the program's
        // id, because the analyzer would then resolve the call
        // against the wrong BTF universe.
        if insn.src_reg() != BPF_PSEUDO_CALL {
            continue;
        }
        // Resolve the symbol → name. goblin parses the symbol
        // table referenced by the rel section's sh_link via
        // `elf.syms`. The symbol's `st_name` indexes the
        // associated string table (`elf.strtab`).
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        // Match libbpf's `sym_is_extern`: the symbol must be an
        // undefined NOTYPE with global or weak binding. Anything
        // else is a subprog, a static helper, or a data symbol;
        // not a kfunc.
        const STT_NOTYPE: u8 = goblin::elf::sym::STT_NOTYPE;
        const STB_GLOBAL: u8 = goblin::elf::sym::STB_GLOBAL;
        const STB_WEAK: u8 = goblin::elf::sym::STB_WEAK;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx != SHN_UNDEF {
            continue;
        }
        if sym.st_type() != STT_NOTYPE {
            continue;
        }
        let bind = sym.st_bind();
        if bind != STB_GLOBAL && bind != STB_WEAK {
            continue;
        }
        // The string-table interning goblin builds gives us a
        // borrow of the symbol's name without copying.
        let name = match elf.strtab.get_at(sym.st_name) {
            Some(s) if !s.is_empty() => s,
            _ => continue,
        };
        // Look up the symbol name in the program BTF. We want a
        // `BTF_KIND_FUNC` with extern linkage (mirroring libbpf's
        // `find_extern_btf_id`). The helper returns every id
        // sharing this name; we accept only Func/extern. A name
        // that resolves to multiple distinct Func ids (impossible
        // in well-formed BPF BTF since extern names are unique)
        // yields the first match — same as libbpf.
        let Some(func_btf_id) = find_extern_func_btf_id(btf, name) else {
            continue;
        };
        // Patch in place. The two changes mirror libbpf's
        // RELO_EXTERN_CALL handler exactly. Note we mutate the
        // packed `regs` byte directly: src_reg occupies the
        // high 4 bits, dst_reg the low 4, and the analyzer's
        // `BpfInsn::src_reg()` accessor reads them back as
        // expected after the rewrite.
        insn.set_src_reg(BPF_PSEUDO_KFUNC_CALL);
        insn.imm = func_btf_id as i32;
    }
}

/// Mirror libbpf's BPF-to-BPF subprog call patching on the host side.
///
/// libbpf-rs's `Linker` leaves every global BPF-to-BPF subprog call
/// as a `BPF_PSEUDO_CALL` with `imm = -1`, paired with an ELF
/// relocation against an `STT_FUNC` symbol whose containing section
/// is one of the program text sections we concatenated into
/// `text_concat`. Without patching, the cast analyzer's
/// [`crate::monitor::cast_analysis::Analyzer::analyze`] computes
/// `callee_pc = pc + 1 + insn.imm = pc + 1 + (-1) = pc` and
/// inserts the caller's R1..R5 snapshot into `caller_arg_types`
/// at the call site itself instead of at the callee's entry PC.
/// The downstream lookup at `entries_by_pc` then reseeds R1..R5
/// at every callee entry with `RegState::Unknown`, dropping all
/// inter-procedural typed-pointer flow.
///
/// At kernel-load time libbpf computes
/// `sub_insn_idx = sym.st_value/8 + insn.imm + 1` per
/// `bpf_object__reloc_code` (tools/lib/bpf/libbpf.c). For a
/// global subprog: `sym.st_value` = byte offset of the callee's
/// first instruction within its section; `insn.imm = -1` is the
/// libbpf placeholder; `+1` accounts for the BPF call ABI
/// (next-instruction-relative). The result `sub_insn_idx` is the
/// callee's entry PC in libbpf's appended-into-main-prog
/// instruction stream — the same shape as our `text_concat`,
/// modulo per-section base offsets we tracked in `section_bases`.
///
/// We patch in place so the analyzer's computation lands on the
/// correct callee entry: target `imm` = `callee_pc - call_pc - 1`
/// where both PCs are absolute indices in `text_concat`. After
/// patching, the analyzer's
/// `callee_pc = pc + 1 + insn.imm = call_pc + 1 + (callee_pc - call_pc - 1)`
/// resolves to the actual callee entry PC.
///
/// # What gets patched
///
/// - Instruction must be `BPF_JMP|BPF_CALL` (code byte `0x85`).
/// - Current `src_reg` must be `BPF_PSEUDO_CALL` (1). After
///   [`patch_kfunc_calls`] runs, kfunc call sites have
///   `src_reg == BPF_PSEUDO_KFUNC_CALL` (2) and skip this gate.
/// - Current `imm` must be `-1` (the libbpf placeholder). Static
///   (file-local) subprog calls have `imm` already pointing at the
///   target byte offset and skip this gate — clang's pre-relocation
///   encoding for static subprogs is correct as-is.
/// - Symbol must be `STT_FUNC` and not `SHN_UNDEF`. Extern calls
///   (`STT_NOTYPE`, `SHN_UNDEF`) were already handled by
///   [`patch_kfunc_calls`]; non-FUNC symbols (data, section,
///   notype) cannot be subprog targets.
/// - Symbol's section must appear in `section_bases` — only
///   sections we concatenated are eligible callee containers.
/// - `sym.st_value` must be a multiple of [`BPF_INSN_SIZE`]; a
///   non-aligned offset is malformed input (no real subprog
///   starts on a non-8-byte-aligned boundary).
///
/// All gates plus the section-base lookup must hold before any
/// byte is patched. Anything else is a no-op.
///
/// # Errors
///
/// This function never fails. An ELF without relocation sections,
/// a relocation pointing into a section we did not concatenate, a
/// symbol we cannot resolve, an out-of-range PC, an unaligned
/// `st_value`, an arithmetic overflow on the imm computation —
/// every failure path produces a silent no-op. The cast map ends
/// up identical to the pre-patching world for those instructions.
/// False negatives are safe per the analyzer's "false negative is
/// safe; false positive is not" stance.
fn patch_subprog_calls(
    text_concat: &mut [BpfInsn],
    elf: &goblin::elf::Elf<'_>,
    section_bases: &HashMap<u32, usize>,
) {
    // The shared `iter_text_relocs` helper handles the rel-section /
    // target-section / `r_offset` validation preamble. Each item
    // is a relocation that targets a known program text section
    // at an 8-byte-aligned, in-bounds offset; the subprog-specific
    // gates (call opcode, imm == -1, BPF_PSEUDO_CALL src_reg,
    // STT_FUNC defined symbol, callee section in `section_bases`,
    // st_value alignment) are applied here.
    //
    // Capture `text_concat.len()` once up front so the callee-PC
    // bound check inside the loop body does not collide with the
    // mutable borrow from `text_concat.get_mut(call_pc)`.
    let text_len = text_concat.len();
    for (call_pc, reloc) in iter_text_relocs(elf, section_bases) {
        let Some(insn) = text_concat.get_mut(call_pc) else {
            continue;
        };
        // Gate 1: the instruction must be a BPF call site.
        if insn.code != cast_analysis_load_consts::BPF_JMP_CALL_CODE {
            continue;
        }
        // Gate 2: `imm` must be the libbpf placeholder for global
        // subprog calls. Static (file-local) subprog calls already
        // carry the correct PC-relative offset in `imm` and must
        // not be touched.
        if insn.imm != -1 {
            continue;
        }
        // Gate 3: src_reg must be the clang-emitted
        // `BPF_PSEUDO_CALL` (1). After [`patch_kfunc_calls`] runs
        // first, kfunc call sites have `src_reg ==
        // BPF_PSEUDO_KFUNC_CALL` (2) and naturally skip this gate.
        if insn.src_reg() != BPF_PSEUDO_CALL {
            continue;
        }
        // Resolve the symbol. The reloc's symbol must be a defined
        // STT_FUNC (the global subprog shape). Extern kfunc calls
        // were already handled upstream; data symbols, section
        // symbols, and STT_NOTYPE entries are not subprog targets.
        let Some(sym) = elf.syms.get(reloc.r_sym) else {
            continue;
        };
        const STT_FUNC: u8 = goblin::elf::sym::STT_FUNC;
        const SHN_UNDEF: usize = 0;
        if sym.st_shndx == SHN_UNDEF {
            continue;
        }
        if sym.st_type() != STT_FUNC {
            continue;
        }
        // The symbol's section must appear in `section_bases` —
        // only sections we concatenated are valid callee
        // containers. A subprog defined in a section we did not
        // collect (e.g. SHF_EXECINSTR-less PROGBITS, or one whose
        // size is not a multiple of [`BPF_INSN_SIZE`]) cannot be
        // resolved to a callee PC and is skipped silently.
        let callee_sec_idx = sym.st_shndx as u32;
        let Some(&callee_section_base) = section_bases.get(&callee_sec_idx) else {
            continue;
        };
        // `sym.st_value` is the byte offset of the callee's first
        // instruction within its section (relative to the section's
        // sh_addr). For BPF .o files, sections are non-allocated
        // and sh_addr is 0, so st_value is a plain byte offset.
        // We still subtract sh_addr defensively to handle any
        // future shape where the inner ELF might surface an
        // allocated text section.
        let Some(callee_section) = elf.section_headers.get(callee_sec_idx as usize) else {
            continue;
        };
        let Some(sym_offset_bytes) = sym.st_value.checked_sub(callee_section.sh_addr) else {
            continue;
        };
        let sym_offset_bytes = sym_offset_bytes as usize;
        if !sym_offset_bytes.is_multiple_of(BPF_INSN_SIZE) {
            continue;
        }
        let callee_pc = match callee_section_base.checked_add(sym_offset_bytes / BPF_INSN_SIZE) {
            Some(p) => p,
            None => continue,
        };
        // Bound-check the callee PC against text_concat — a
        // st_value past the end of the concatenated stream is a
        // corrupt ELF and would produce a meaningless caller_arg
        // entry; drop silently.
        if callee_pc >= text_len {
            continue;
        }
        // Compute the new `imm` so the analyzer's
        // `pc + 1 + imm` lands on `callee_pc`. The signed-
        // arithmetic conversion handles call sites that point
        // backward (callee earlier in the stream than caller).
        let call_pc_i64 = call_pc as i64;
        let callee_pc_i64 = callee_pc as i64;
        let new_imm = callee_pc_i64 - call_pc_i64 - 1;
        // i32 range guard: a single BPF program text plus its
        // siblings cannot exceed 2^31 instructions in any realistic
        // build, but the source ELF is attacker-influenced so we
        // bound-check rather than silently truncate.
        if new_imm < i32::MIN as i64 || new_imm > i32::MAX as i64 {
            continue;
        }
        insn.imm = new_imm as i32;
    }
}

/// Find the `BTF_KIND_FUNC` whose name matches `name` and whose
/// linkage is extern. Returns `None` if the name does not resolve
/// in the BTF or if the only matching id is not a Func / not extern.
///
/// Mirrors libbpf's `find_extern_btf_id` restricted to FUNC kinds
/// — the cast analyzer only consumes FUNCs (it does not type-
/// recover ksym data variables, just kfunc returns).
fn find_extern_func_btf_id(btf: &Btf, name: &str) -> Option<u32> {
    let ids = btf.resolve_ids_by_name(name).ok()?;
    for id in ids {
        let Ok(ty) = btf.resolve_type_by_id(id) else {
            continue;
        };
        if let Type::Func(f) = ty
            && f.is_extern()
        {
            return Some(id);
        }
    }
    None
}

/// Constants this module needs to talk about BPF instruction wire
/// encoding without pulling the full `cast_analysis` constants set
/// into module scope. Kept private so the loader's surface stays
/// minimal.
mod cast_analysis_load_consts {
    use libbpf_rs::libbpf_sys as bs;
    /// `BPF_JMP | BPF_CALL` opcode byte = `0x85`. The single value
    /// every BPF call instruction (helper, subprog, kfunc) carries
    /// in its `code` field. Used by the kfunc-relocation patcher
    /// to confirm the relocated slot is in fact a call site before
    /// rewriting `src_reg` / `imm`.
    pub(super) const BPF_JMP_CALL_CODE: u8 = (bs::BPF_JMP | bs::BPF_CALL) as u8;
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
            if !insn_off.is_multiple_of(BPF_INSN_SIZE) {
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

mod persist;

#[cfg(test)]
mod tests;
