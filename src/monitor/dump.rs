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
//! # BTF source — known limitation
//!
//! Today the renderer uses the host's vmlinux BTF (passed in as `&Btf`
//! by the caller). This resolves kernel-defined primitives — `int`,
//! `__u64`, `task_struct` and friends — but does **not** resolve
//! scheduler-defined types like `task_ctx` or any custom struct the
//! BPF program declares: those type IDs index the program's *own*
//! BTF blob, which lives in guest memory at the kernel KVA captured
//! in [`BpfMapInfo::btf_kva`]. Renders against unknown type IDs land
//! as [`RenderedValue::Unsupported`] rather than crashing, but the
//! operator only sees a hex dump for the value bytes.
//!
//! Loading the per-map program BTF — read the `struct btf` at
//! `btf_kva`, follow its `data`/`data_size` fields, translate via the
//! guest page tables, and feed the bytes to [`Btf::from_bytes`] — is
//! the path forward for full scheduler-defined-type rendering.
//! Tracked as task #49.
//!
//! Practical impact today:
//! - scx-ktstr (the test fixture) exercises only kernel primitives, so
//!   its scheduler state renders fully.
//! - Real schedulers under test will see scheduler-private types fall
//!   back to hex until #49 lands.

use serde::{Deserialize, Serialize};

use btf_rs::Btf;

use super::bpf_map::{
    BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_ARRAY, BpfMapAccessor, BpfMapInfo,
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
/// The dump is best-effort: a map that fails to render lands in the
/// report with `error: Some(...)` rather than aborting the whole walk,
/// so a single corrupt map can't blind the operator to the rest of
/// the scheduler's state.
pub fn dump_state(accessor: &BpfMapAccessor<'_>, btf: &Btf, num_cpus: u32) -> StallDumpReport {
    let maps = accessor.maps();
    let mut report = StallDumpReport {
        maps: Vec::with_capacity(maps.len()),
    };

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
        report.maps.push(render_map(accessor, btf, &info, num_cpus));
    }

    report
}

fn render_map(
    accessor: &BpfMapAccessor<'_>,
    btf: &Btf,
    info: &BpfMapInfo,
    num_cpus: u32,
) -> StallDumpMap {
    let mut out = StallDumpMap {
        name: info.name.clone(),
        map_type: info.map_type,
        value_size: info.value_size,
        max_entries: info.max_entries,
        value: None,
        entries: Vec::new(),
        percpu_entries: Vec::new(),
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
                Some(bytes) if info.btf_value_type_id != 0 => {
                    out.value = Some(render_value(btf, info.btf_value_type_id, &bytes));
                }
                Some(bytes) => {
                    // No BTF type — emit a hex fallback.
                    out.value = Some(RenderedValue::Bytes {
                        hex: hex_dump(&bytes),
                    });
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
            // BpfMapInfo currently captures only `btf_value_type_id`,
            // not a key type id, so hash-map keys surface as hex bytes
            // here. Values render with the value type id when one is
            // present; otherwise keys-and-values both fall through to
            // hex.
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
                let value = if info.btf_value_type_id != 0 {
                    Some(render_value(btf, info.btf_value_type_id, &v))
                } else {
                    None
                };
                out.entries.push(StallDumpEntry {
                    key: None,
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
                        maybe_bytes.map(|b| {
                            if info.btf_value_type_id != 0 {
                                render_value(btf, info.btf_value_type_id, &b)
                            } else {
                                RenderedValue::Bytes { hex: hex_dump(&b) }
                            }
                        })
                    })
                    .collect();
                out.percpu_entries
                    .push(StallDumpPercpuEntry { key, per_cpu });
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
}
