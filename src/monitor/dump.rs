//! BPF map state dump for stall-trigger post-mortem.
//!
//! [`dump_state`] is invoked by the freeze coordinator after the vCPU
//! rendezvous succeeds (see `src/vmm/mod.rs`). It enumerates every
//! BPF map in the guest via [`BpfMapAccessor::maps`], filters out
//! ktstr-internal probes (the framework's own probe and fentry skel
//! maps), and dispatches per map type:
//!
//! - `BPF_MAP_TYPE_ARRAY` (and the `.bss` / `.data` / `.rodata`
//!   global-section maps libbpf creates as single-key arrays) — read
//!   the whole value buffer and render it via [`btf_render::render_value`].
//! - `BPF_MAP_TYPE_HASH` — iterate (key, value) pairs, capped at
//!   [`MAX_HASH_ENTRIES`].
//! - `BPF_MAP_TYPE_PERCPU_ARRAY` — read each CPU's slot for keys
//!   `0..min(max_entries, MAX_PERCPU_KEYS)`.
//! - Other types — recorded as [`StallDumpMap::error`] so the operator
//!   sees the gap rather than a silent omission.
//!
//! # BTF source — per-map program BTF loading
//!
//! The renderer loads each map's program BTF from guest memory at
//! [`BpfMapInfo::btf_kva`], following the kernel `struct btf`'s
//! `data`/`data_size`/`base_btf` fields. Split BTF (program types
//! extending vmlinux) is parsed via [`Btf::from_split_bytes`] with
//! the host's vmlinux BTF as the base (correct when host kernel ==
//! guest kernel — ktstr's default and the common CI configuration).
//! A per-`btf_kva` cache dedupes parses across maps sharing a
//! program's BTF object. When per-map load fails (still-booting
//! guest, untranslatable page, corrupted blob), the renderer falls
//! back to the caller-supplied vmlinux BTF.

use serde::{Deserialize, Serialize};

use btf_rs::Btf;

use super::arena::{ArenaSnapshot, BpfArenaOffsets, snapshot_arena};
use super::bpf_map::{
    BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_ARRAY,
    BpfMapAccessor, BpfMapInfo,
};
use super::btf_render::{RenderedValue, render_value};

/// Top-level stall-dump report. One per freeze trigger.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub struct StallDumpReport {
    /// One entry per BPF map enumerated. Order matches the IDR walk
    /// (i.e. allocation order); the report is otherwise unsorted so
    /// callers that want a stable view should sort by name.
    pub maps: Vec<StallDumpMap>,
}

/// Rendering of one BPF map's contents.
///
/// Unifies the four map-type rendering paths under a single
/// representation: scalar-valued maps (ARRAY) populate `value`; keyed
/// maps (HASH) populate `entries`; per-CPU maps populate
/// `percpu_entries`. Exactly one of these is non-empty for a
/// successful render; on failure `error` is set and the rest empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StallDumpMap {
    /// Map name as registered with the kernel. Truncated to
    /// `BPF_OBJ_NAME_LEN` (16) by the kernel; libbpf composes
    /// "<obj_name>.<section>" for global-section maps.
    pub name: String,
    /// Raw `map_type` from `struct bpf_map` (e.g. `BPF_MAP_TYPE_ARRAY`).
    /// Kept as `u32` rather than an enum to avoid bumping a serde
    /// schema each time the kernel adds a kind.
    pub map_type: u32,
    /// Declared per-entry value size. Captured even when rendering
    /// fails so the operator can see the map shape.
    pub value_size: u32,
    /// Declared maximum entry count from `struct bpf_map.max_entries`.
    /// Surfaces alongside the rendered slice so a consumer can spot
    /// when the dump shows fewer entries than the map declares
    /// (e.g. multi-entry ARRAY rendering only key 0; HASH map
    /// truncated at [`MAX_HASH_ENTRIES`]; PERCPU_ARRAY truncated at
    /// [`MAX_PERCPU_KEYS`]).
    pub max_entries: u32,
    /// Single-value render (set for ARRAY-style maps).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<RenderedValue>,
    /// (key, value) entries for HASH maps.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entries: Vec<StallDumpEntry>,
    /// Per-CPU slots for PERCPU_ARRAY maps. Outer Vec indexed by key,
    /// inner Vec indexed by CPU id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub percpu_entries: Vec<StallDumpPercpuEntry>,
    /// Page snapshot for `BPF_MAP_TYPE_ARENA` maps. `None` for all
    /// other map types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arena: Option<ArenaSnapshot>,
    /// Reason this map's contents are missing or partial. Empty on
    /// successful render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// One (key, value) pair from a hash map. Both sides are rendered via
/// BTF when key/value type ids are available; a `None` rendering
/// preserves the raw bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StallDumpEntry {
    /// Rendered key. `None` when no BTF type is available for the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<RenderedValue>,
    /// Hex-encoded raw key bytes. Kept alongside `key` so the operator
    /// can correlate rendered output with the wire format.
    pub key_hex: String,
    /// Rendered value. `None` when no BTF type is available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<RenderedValue>,
    /// Hex-encoded raw value bytes.
    pub value_hex: String,
}

/// One key from a per-CPU array, with one rendered value per CPU
/// (None for CPUs whose per-CPU page was unmapped or out-of-range).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StallDumpPercpuEntry {
    pub key: u32,
    pub per_cpu: Vec<Option<RenderedValue>>,
}

impl std::fmt::Display for StallDumpReport {
    /// Human-readable rendering of every map. JSON remains the
    /// programmatic form via `serde_json`; this Display is the
    /// default presentation used in test-failure output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.maps.is_empty() {
            return f.write_str("(empty stall dump)");
        }
        for (i, m) in self.maps.iter().enumerate() {
            if i > 0 {
                f.write_str("\n\n")?;
            }
            std::fmt::Display::fmt(m, f)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for StallDumpMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "map {} (type={}, value_size={}, max_entries={})",
            self.name, self.map_type, self.value_size, self.max_entries
        )?;
        if let Some(err) = &self.error {
            write!(f, " [error: {err}]")?;
        }
        if let Some(value) = &self.value {
            f.write_str("\n")?;
            std::fmt::Display::fmt(value, f)?;
        }
        for entry in &self.entries {
            f.write_str("\n")?;
            std::fmt::Display::fmt(entry, f)?;
        }
        for entry in &self.percpu_entries {
            f.write_str("\n")?;
            std::fmt::Display::fmt(entry, f)?;
        }
        if let Some(arena) = &self.arena {
            // Arena snapshots have their own Debug-derived shape; use
            // the debug representation for now (one line per page).
            // The full structured render is in the JSON serialization.
            write!(f, "\narena: {arena:?}")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for StallDumpEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("entry {\n  key: ")?;
        match &self.key {
            Some(k) => std::fmt::Display::fmt(k, f)?,
            None => write!(f, "{} (raw)", self.key_hex)?,
        }
        f.write_str("\n  value: ")?;
        match &self.value {
            Some(v) => std::fmt::Display::fmt(v, f)?,
            None => write!(f, "{} (raw)", self.value_hex)?,
        }
        f.write_str("\n}")
    }
}

impl std::fmt::Display for StallDumpPercpuEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "key {}:", self.key)?;
        for (cpu, slot) in self.per_cpu.iter().enumerate() {
            f.write_str("\n")?;
            match slot {
                Some(v) => {
                    write!(f, "  cpu {cpu}: ")?;
                    std::fmt::Display::fmt(v, f)?;
                }
                None => write!(f, "  cpu {cpu}: <unmapped>")?,
            }
        }
        Ok(())
    }
}

/// Maximum per-CPU array key span the dump path will iterate.
///
/// `BPF_MAP_TYPE_PERCPU_ARRAY` declares `max_entries` at create-time;
/// the dump enumerates `0..min(max_entries, MAX_PERCPU_KEYS)` so a
/// scheduler that allocated a million-entry per-CPU array doesn't
/// blow up the report. Today's scx schedulers use small fixed-size
/// per-CPU arrays (one entry per topology level), so this cap is
/// generous.
const MAX_PERCPU_KEYS: u32 = 256;

/// Maximum (key, value) pairs the dump path will pull from a HASH map.
///
/// Mirrors [`super::btf_render::MAX_ARRAY_ELEMS`] (4096): a HASH map
/// with millions of live entries would OOM the host renderer if
/// iterated unbounded, so the dump caps at 4096 and surfaces an
/// `error` describing the truncation. The unrendered tail is silently
/// dropped — recording it would itself require unbounded memory.
const MAX_HASH_ENTRIES: usize = 4096;

/// Sanity cap on a single BTF blob read.
///
/// BPF program BTF is normally <100 KB; vmlinux BTF caps around
/// ~10 MB. A bogus `data_size` (corrupted `struct btf`) shouldn't
/// pull megabytes of unrelated guest memory into the renderer or the
/// freeze coordinator. Shared between [`load_program_btf`] and
/// `vmm::load_probe_bss_offset`; defining it here keeps the bound
/// in one place so a future tightening doesn't drift between sites.
pub(crate) const MAX_BTF_BLOB: usize = 32 * 1024 * 1024;

/// Bare-named ktstr framework maps to skip during enumeration.
///
/// These are declared in `src/bpf/probe.bpf.c` without a libbpf
/// `<obj>.<section>` prefix (`SEC(".maps")` declarations like
/// `func_meta_map`, `probe_data`, `probe_scratch`, `events`); the
/// kernel registers them under the bare names listed here. They're
/// framework-internal — the user looking at a stall dump for their
/// scheduler doesn't care about ktstr's own kprobe scratch — so the
/// dump path drops them.
///
/// Future ktstr probe additions need to be added here AND the
/// matching `<obj_name>.` prefix needs to be in the
/// [`render_map`-internal] starts_with list (see [`dump_state`]).
const KTSTR_INTERNAL_MAPS: &[&str] = &["func_meta_map", "probe_data", "probe_scratch", "events"];

/// Snapshot every BPF map visible to the host accessor.
///
/// `num_cpus` is the guest's `nr_cpu_ids`; pass `1` for non-percpu-only
/// dumps if the caller doesn't have the value handy.
///
/// `arena_offsets` enables `BPF_MAP_TYPE_ARENA` page snapshotting.
/// `None` skips arena rendering (e.g. older kernel without arena
/// support, or BTF that lacks `struct bpf_arena`).
///
/// The dump is best-effort: a map that fails to render lands in the
/// report with `error: Some(...)` rather than aborting the whole walk,
/// so a single corrupt map can't blind the operator to the rest of
/// the scheduler's state.
pub fn dump_state(
    accessor: &BpfMapAccessor<'_>,
    btf: &Btf,
    num_cpus: u32,
    arena_offsets: Option<&BpfArenaOffsets>,
) -> StallDumpReport {
    let maps = accessor.maps();
    let mut report = StallDumpReport {
        maps: Vec::with_capacity(maps.len()),
    };

    // Per-map program-BTF cache, keyed by `btf_kva`. Each unique
    // `struct btf *` lives in the kernel BTF IDR — multiple maps from
    // the same BPF program point at the same KVA, so caching dedupes
    // the heavy `Btf::from_bytes`/`from_split_bytes` parse across them
    // (a scheduler with N maps backed by one BPF object pays one
    // parse, not N). Lookups go through this cache before falling
    // back to the caller-supplied vmlinux `btf`.
    let mut program_btfs: std::collections::HashMap<u64, Btf> = std::collections::HashMap::new();

    for info in maps {
        // Skip ktstr's own framework maps so the report only shows
        // the scheduler-under-test's state. Three distinct shapes
        // need filtering:
        //
        // 1. Global-section maps from the probe skeleton: libbpf
        //    composes `<obj_name>.<section>` so `probe_bp.bss`,
        //    `probe_bp.data`, `probe_bp.rodata` all match the
        //    `probe_bp.` prefix. (`probe_bp` matching the bare obj
        //    name covers any single-name section the kernel might
        //    surface, though libbpf today always adds the suffix.)
        // 2. Global-section maps from the fentry skeleton, named
        //    with the `fentry_p.` prefix following the same
        //    libbpf convention.
        // 3. Bare-named maps declared via `SEC(".maps")` in
        //    src/bpf/probe.bpf.c — these don't get an obj prefix
        //    because they're not from a global section. The
        //    explicit denylist [`KTSTR_INTERNAL_MAPS`] enumerates
        //    them.
        //
        // A future tighter filter would consult bpf_prog ownership
        // (the program-attachment ID list pinned to each map), but
        // name-based filtering is enough today and avoids loading
        // the full prog_idr walk on the freeze hot path.
        if info.name.starts_with("probe_bp.")
            || info.name.starts_with("fentry_p.")
            || info.name == "probe_bp"
            || info.name == "fentry_p"
            || KTSTR_INTERNAL_MAPS.contains(&info.name.as_str())
        {
            continue;
        }

        // Resolve the per-map BTF.
        //
        // The map's `btf_value_type_id` / `btf_key_type_id` index
        // the *map's own* BTF, NOT the kernel vmlinux BTF — when
        // `btf_kva != 0` the type IDs are program-local and using
        // vmlinux BTF with them would resolve to unrelated kernel
        // types (or out-of-range nonsense). So:
        //
        //   - `btf_kva != 0` AND program BTF loads     → use it.
        //   - `btf_kva != 0` AND program BTF fails     → render
        //     hex-only (None map_btf), no fallback.
        //   - `btf_kva == 0` (kernel-builtin map)      → use the
        //     caller-supplied vmlinux BTF; the type IDs (if any)
        //     genuinely index vmlinux BTF in this case.
        if info.btf_kva != 0
            && !program_btfs.contains_key(&info.btf_kva)
            && let Some(loaded) = load_program_btf(accessor, info.btf_kva, btf)
        {
            program_btfs.insert(info.btf_kva, loaded);
        }
        let map_btf: Option<&Btf> = if info.btf_kva != 0 {
            program_btfs.get(&info.btf_kva)
        } else {
            Some(btf)
        };

        report.maps.push(render_map(
            accessor,
            map_btf,
            &info,
            num_cpus,
            arena_offsets,
        ));
    }

    report
}

/// Load a BPF program's `struct btf` from guest memory.
///
/// Reads the kernel `struct btf` at `btf_kva`, follows its `data` /
/// `data_size` / `base_btf` fields, fetches the raw BTF blob via
/// page-walked vmalloc reads, and parses it. When `base_btf` is
/// non-NULL the program's BTF is split atop the vmlinux BTF (the
/// kernel's own base BTF) — pass the host's already-parsed vmlinux
/// `Btf` as the split base so type IDs resolve correctly.
///
/// Returns `None` when any step fails: missing offsets, untranslatable
/// pages, or `Btf::from_bytes` rejection (truncated / corrupted blob).
/// Failure is silent and the caller falls back to the host vmlinux
/// BTF — the dump is best-effort, a partial render still beats no
/// render.
fn load_program_btf(accessor: &BpfMapAccessor<'_>, btf_kva: u64, base_btf: &Btf) -> Option<Btf> {
    let kernel = accessor.kernel();
    let offsets = accessor.offsets();
    let mem = kernel.mem();

    // `struct btf` may be kmalloc'd (direct map) or vmalloc'd; use
    // translate_any_kva.
    let btf_pa = super::idr::translate_any_kva(
        mem,
        kernel.cr3_pa(),
        kernel.page_offset(),
        btf_kva,
        kernel.l5(),
    )?;
    let data_kva = mem.read_u64(btf_pa, offsets.btf_data);
    let data_size = mem.read_u32(btf_pa, offsets.btf_data_size) as usize;
    let base_kva = mem.read_u64(btf_pa, offsets.btf_base_btf);

    if data_kva == 0 || data_size == 0 {
        return None;
    }

    if data_size > MAX_BTF_BLOB {
        return None;
    }

    // The BTF blob is vmalloc-backed — `btf->data` is allocated via
    // vmalloc / kvmalloc inside `kernel/bpf/btf.c`'s
    // `btf_parse_*` paths. Use the chunked vmalloc reader so a
    // 100 KB blob doesn't pay 100K syscalls of byte-wise translate.
    // The chunked reader honours all-or-nothing semantics, so a
    // short read returns None directly; no extra length check needed.
    let blob = kernel.read_kva_bytes_chunked(data_kva, data_size)?;

    if base_kva != 0 {
        // Split BTF: the program's types extend the kernel's
        // vmlinux BTF. Pass the host's parsed vmlinux Btf as the
        // base so cross-base type IDs (e.g. `task_struct`) resolve.
        //
        // Uses host vmlinux BTF as split base — correct when host
        // kernel == guest kernel (ktstr's default and the common
        // CI configuration). A guest running a different kernel
        // version would silently mis-render cross-base type
        // references; flagged as a known limitation in the module
        // doc above.
        Btf::from_split_bytes(&blob, base_btf).ok()
    } else {
        Btf::from_bytes(&blob).ok()
    }
}

fn render_map(
    accessor: &BpfMapAccessor<'_>,
    btf: Option<&Btf>,
    info: &BpfMapInfo,
    num_cpus: u32,
    arena_offsets: Option<&BpfArenaOffsets>,
) -> StallDumpMap {
    let mut out = StallDumpMap {
        name: info.name.clone(),
        map_type: info.map_type,
        value_size: info.value_size,
        max_entries: info.max_entries,
        value: None,
        entries: Vec::new(),
        percpu_entries: Vec::new(),
        arena: None,
        error: None,
    };

    match info.map_type {
        BPF_MAP_TYPE_ARRAY => {
            // Read the entire value buffer in one shot. Single-entry
            // global-section maps (.bss / .data / .rodata) declare
            // value_size as the section size; multi-entry ARRAY maps
            // declare it as one entry's size — the renderer only sees
            // one entry's worth of bytes here, which matches the
            // kernel's value-region layout for ARRAY (each key is
            // contiguous starting at `bpf_array.value`).
            //
            // The BTF type id `btf_value_type_id` describes one entry,
            // so for max_entries > 1 the renderer would need to be
            // called per-key. ARRAY maps used by sched_ext today are
            // either single-entry global sections or per-CPU arrays;
            // multi-entry plain ARRAYs surface as the first entry
            // only. The truncation is recorded in `error` and
            // `max_entries` so the consumer sees the partial render.
            match accessor.read_value(info, 0, info.value_size as usize) {
                Some(bytes) => {
                    // BTF-driven render only when both a BTF object
                    // is available AND the map declares a value type
                    // id — `info.btf_value_type_id` indexes the
                    // map's program BTF, so without that BTF the id
                    // resolves to nothing meaningful.
                    out.value = match (btf, info.btf_value_type_id) {
                        (Some(b), id) if id != 0 => Some(render_value(b, id, &bytes)),
                        _ => Some(RenderedValue::Bytes {
                            hex: hex_dump(&bytes),
                        }),
                    };
                }
                None => {
                    out.error = Some("ARRAY value region unreadable (unmapped page?)".into());
                }
            }
            // Multi-entry ARRAY: surface the silent truncation. The
            // single-entry global-section maps (.bss/.data/.rodata)
            // declare max_entries=1 so this branch is a no-op for
            // them; only schedulers using BPF_MAP_TYPE_ARRAY with
            // multiple keys hit it.
            if out.error.is_none() && info.max_entries > 1 {
                out.error = Some(format!(
                    "multi-entry ARRAY: only key 0 of {} shown",
                    info.max_entries
                ));
            }
        }
        BPF_MAP_TYPE_HASH => {
            // Both key and value render via BTF when their type IDs
            // are present (`btf_key_type_id` / `btf_value_type_id`
            // captured during map enumeration). Either side falls
            // through to a hex dump alongside the rendered counterpart
            // when its type id is 0 — so an operator always sees the
            // raw bytes, even if BTF didn't help.
            //
            // Hard-cap at MAX_HASH_ENTRIES to keep a million-entry
            // hash from OOMing the host renderer. `iter_hash_map`
            // already enforces its own much-larger HTAB_ITER_MAX
            // (1_000_000) inside the bucket walk, but a million
            // [`RenderedValue`] trees would still pin gigabytes
            // here — surface the truncation in `out.error` so the
            // consumer sees that the rendered slice is partial.
            let raw_entries = accessor.iter_hash_map(info);
            let truncated = raw_entries.len() > MAX_HASH_ENTRIES;
            for (k, v) in raw_entries.into_iter().take(MAX_HASH_ENTRIES) {
                // Both render gates require BTF presence AND
                // non-zero type id; same reasoning as the ARRAY arm.
                let key = match (btf, info.btf_key_type_id) {
                    (Some(b), id) if id != 0 => Some(render_value(b, id, &k)),
                    _ => None,
                };
                let value = match (btf, info.btf_value_type_id) {
                    (Some(b), id) if id != 0 => Some(render_value(b, id, &v)),
                    _ => None,
                };
                out.entries.push(StallDumpEntry {
                    key,
                    key_hex: hex_dump(&k),
                    value,
                    value_hex: hex_dump(&v),
                });
            }
            if truncated {
                out.error = Some(format!("hash map truncated at {MAX_HASH_ENTRIES} entries"));
            }
        }
        BPF_MAP_TYPE_PERCPU_ARRAY => {
            let limit = info.max_entries.min(MAX_PERCPU_KEYS);
            for key in 0..limit {
                let per_cpu_bytes = accessor.read_percpu_array(info, key, num_cpus);
                let per_cpu = per_cpu_bytes
                    .into_iter()
                    .map(|maybe_bytes| {
                        maybe_bytes.map(|b| match (btf, info.btf_value_type_id) {
                            (Some(b_btf), id) if id != 0 => render_value(b_btf, id, &b),
                            _ => RenderedValue::Bytes { hex: hex_dump(&b) },
                        })
                    })
                    .collect();
                out.percpu_entries
                    .push(StallDumpPercpuEntry { key, per_cpu });
            }
            // Surface PERCPU_ARRAY key truncation, mirroring the
            // ARRAY (key 0 of N) and HASH (entries cap) patterns:
            // when the map declares more keys than [`MAX_PERCPU_KEYS`],
            // the dump only walks the first MAX_PERCPU_KEYS slots and
            // the consumer needs to know the rest are dropped.
            if info.max_entries > MAX_PERCPU_KEYS {
                out.error = Some(format!(
                    "PERCPU_ARRAY truncated at {MAX_PERCPU_KEYS} keys (max_entries={})",
                    info.max_entries,
                ));
            }
        }
        BPF_MAP_TYPE_ARENA => {
            // Arena pages live in vmalloc space and are translated
            // via the existing PTE walker; rendering is page-granular
            // (4 KiB per ArenaPage) rather than struct-granular
            // because arena memory has no canonical type schema —
            // BPF programs allocate raw memory regions and impose
            // their own layouts. Operators can post-process the page
            // bytes against the program's data structure docs.
            match arena_offsets {
                Some(off) => {
                    let snap = snapshot_arena(accessor.kernel(), info, off);
                    out.arena = Some(snap);
                }
                None => {
                    out.error = Some(
                        "arena BTF offsets unavailable (kernel lacks struct bpf_arena?)".into(),
                    );
                }
            }
        }
        other => {
            out.error = Some(format!("map_type {other} not yet supported by stall dump"));
        }
    }

    out
}

fn hex_dump(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        // unwrap is safe: write! to String never fails.
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_dump_basic() {
        assert_eq!(hex_dump(&[]), "");
        assert_eq!(hex_dump(&[0]), "00");
        assert_eq!(hex_dump(&[0x12, 0x34, 0xab]), "12 34 ab");
    }

    #[test]
    fn report_serde_roundtrip() {
        let report = StallDumpReport {
            maps: vec![StallDumpMap {
                name: "scx_demo.bss".into(),
                map_type: BPF_MAP_TYPE_ARRAY,
                value_size: 8,
                max_entries: 1,
                value: Some(RenderedValue::Uint {
                    bits: 32,
                    value: 42,
                }),
                entries: Vec::new(),
                percpu_entries: Vec::new(),
                arena: None,
                error: None,
            }],
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: StallDumpReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.maps.len(), 1);
        assert_eq!(parsed.maps[0].name, "scx_demo.bss");
        assert_eq!(parsed.maps[0].max_entries, 1);
    }

    #[test]
    fn empty_report_serde() {
        let report = StallDumpReport::default();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: StallDumpReport = serde_json::from_str(&json).unwrap();
        assert!(parsed.maps.is_empty());
    }

    // ---- Display impl coverage --------------------------------------
    //
    // The Display impl is the human-readable form used in test
    // failure output. Pin its layout against representative shapes.

    fn make_simple_map() -> StallDumpMap {
        StallDumpMap {
            name: "scx_demo.bss".into(),
            map_type: BPF_MAP_TYPE_ARRAY,
            value_size: 8,
            max_entries: 1,
            value: Some(RenderedValue::Struct {
                type_name: Some("task_ctx".into()),
                members: vec![super::super::btf_render::RenderedMember {
                    name: "weight".into(),
                    value: RenderedValue::Uint {
                        bits: 32,
                        value: 1024,
                    },
                }],
            }),
            entries: Vec::new(),
            percpu_entries: Vec::new(),
            arena: None,
            error: None,
        }
    }

    #[test]
    fn report_display_empty() {
        let report = StallDumpReport::default();
        assert_eq!(format!("{report}"), "(empty stall dump)");
    }

    #[test]
    fn report_display_one_map_with_value() {
        let report = StallDumpReport {
            maps: vec![make_simple_map()],
        };
        let out = format!("{report}");
        // Map header line.
        assert!(
            out.starts_with("map scx_demo.bss (type="),
            "missing header: {out}"
        );
        // Struct rendering with one indented member.
        assert!(out.contains("struct task_ctx {"), "missing struct: {out}");
        assert!(out.contains("  weight: 1024"), "missing member: {out}");
        assert!(out.ends_with('}'), "missing closing brace: {out}");
    }

    #[test]
    fn report_display_multiple_maps_separated() {
        let report = StallDumpReport {
            maps: vec![make_simple_map(), make_simple_map()],
        };
        let out = format!("{report}");
        // Maps separated by a blank line (\n\n).
        let blank_line_count = out.matches("\n\n").count();
        assert_eq!(
            blank_line_count, 1,
            "expected one blank-line separator between two maps: {out}"
        );
    }

    #[test]
    fn map_display_includes_error_marker() {
        let mut m = make_simple_map();
        m.value = None;
        m.error = Some("ARRAY value region unreadable".into());
        let out = format!("{m}");
        assert!(
            out.contains("[error: ARRAY value region unreadable]"),
            "missing error marker: {out}"
        );
    }

    #[test]
    fn entry_display_renders_key_and_value() {
        let entry = StallDumpEntry {
            key: Some(RenderedValue::Uint { bits: 32, value: 7 }),
            key_hex: "07 00 00 00".into(),
            value: Some(RenderedValue::Uint {
                bits: 32,
                value: 99,
            }),
            value_hex: "63 00 00 00".into(),
        };
        let out = format!("{entry}");
        assert!(out.contains("key: 7"), "missing key: {out}");
        assert!(out.contains("value: 99"), "missing value: {out}");
    }

    #[test]
    fn entry_display_falls_back_to_hex_when_no_btf() {
        // No BTF → key/value are None; Display surfaces the hex.
        let entry = StallDumpEntry {
            key: None,
            key_hex: "ab cd".into(),
            value: None,
            value_hex: "ef".into(),
        };
        let out = format!("{entry}");
        assert!(out.contains("ab cd (raw)"), "missing key hex: {out}");
        assert!(out.contains("ef (raw)"), "missing value hex: {out}");
    }

    #[test]
    fn percpu_entry_display_shows_each_cpu() {
        let entry = StallDumpPercpuEntry {
            key: 0,
            per_cpu: vec![
                Some(RenderedValue::Uint { bits: 32, value: 1 }),
                None,
                Some(RenderedValue::Uint { bits: 32, value: 3 }),
            ],
        };
        let out = format!("{entry}");
        assert!(out.contains("key 0:"));
        assert!(out.contains("cpu 0: 1"));
        assert!(out.contains("cpu 1: <unmapped>"));
        assert!(out.contains("cpu 2: 3"));
    }
}
