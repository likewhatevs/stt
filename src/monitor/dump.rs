//! BPF map state dump for scheduler-failure post-mortem.
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
//! - Other types — recorded as [`FailureDumpMap::error`] so the operator
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
//!
//! # sdt_alloc post-pass
//!
//! After the per-map walk completes, [`dump_state`] runs a post-pass
//! that locates `sdt_alloc`-backed allocator instances inside the
//! scheduler's `.bss` and surfaces every live per-task / per-cgroup
//! allocation as structured records under
//! [`FailureDumpReport::sdt_allocations`]. The walk runs only when
//! every prerequisite is present:
//!   - the scheduler exposes a `.bss` ARRAY map with non-zero
//!     `btf_kva` (so we can read its raw bytes and have a program
//!     BTF to resolve types against),
//!   - at least one `BPF_MAP_TYPE_ARENA` map snapshot succeeded
//!     (so we have `kern_vm_start` for arena pointer translation),
//!   - the program BTF carries `struct scx_allocator` (the scheduler
//!     links `lib/sdt_alloc.bpf.c`).
//!
//! When any prerequisite is missing, the post-pass leaves
//! `sdt_allocations` empty rather than failing the dump — the
//! per-map page-granular [`super::arena::ArenaSnapshot`] still
//! captures raw arena content for callers that don't need
//! structured rendering. See [`super::sdt_alloc`] for the walker
//! design.

use serde::{Deserialize, Serialize};

use btf_rs::Btf;

use super::arena::{ArenaSnapshot, BpfArenaOffsets, snapshot_arena};
use super::bpf_map::{
    BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_PERCPU_ARRAY,
    BpfMapAccessor, BpfMapInfo,
};
use super::btf_render::{RenderedValue, render_value};
use super::sdt_alloc::{
    SdtAllocOffsets, SdtAllocatorSnapshot, discover_payload_btf_id, walk_sdt_allocator,
};

/// Snapshot of one vCPU's instruction-pointer / stack-pointer / page-
/// table-root at freeze time. Re-export of the freeze-side type so
/// dump consumers don't have to depend on `vmm::exit_dispatch`
/// internals.
pub use crate::vmm::exit_dispatch::VcpuRegSnapshot;

/// Schema discriminant value emitted in `FailureDumpReport.schema`.
///
/// Consumers that read a `.failure-dump.json` file use the `schema`
/// field's value to choose between [`FailureDumpReport`] and
/// [`DualFailureDumpReport`] before attempting deserialization.
/// Values are stable wire constants — extending the dump pipeline
/// with a new shape adds a new constant rather than changing this
/// one.
pub const SCHEMA_SINGLE: &str = "single";

/// Schema discriminant value emitted in `DualFailureDumpReport.schema`.
/// See [`SCHEMA_SINGLE`] for the discriminant contract.
pub const SCHEMA_DUAL: &str = "dual";

fn default_schema_single() -> String {
    SCHEMA_SINGLE.to_string()
}

fn default_schema_dual() -> String {
    SCHEMA_DUAL.to_string()
}

/// Top-level failure-dump report. One per freeze trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpReport {
    /// Wire-format discriminant. Always `"single"` for this variant,
    /// pinning [`SCHEMA_SINGLE`]. Consumers branch on this to
    /// choose between [`FailureDumpReport`] and
    /// [`DualFailureDumpReport`] before deserializing — the two
    /// variants share top-level field names that would collide
    /// without an explicit tag.
    #[serde(default = "default_schema_single")]
    pub schema: String,
    /// One entry per BPF map enumerated. Order matches the IDR walk
    /// (i.e. allocation order); the report is otherwise unsorted so
    /// callers that want a stable view should sort by name.
    pub maps: Vec<FailureDumpMap>,
    /// Per-vCPU register snapshots captured on each vCPU thread at
    /// freeze time. Index matches vCPU id (BSP at 0, APs at 1..N).
    /// `None` when a vCPU never parked (rendezvous timeout) or its
    /// `KVM_GET_REGS` failed mid-shutdown. Attached to the report by
    /// the freeze coordinator after `dump_state` returns.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vcpu_regs: Vec<Option<VcpuRegSnapshot>>,
    /// Structured per-allocation views from sdt_alloc-backed
    /// allocators. One entry per discovered allocator; each carries
    /// every live leaf slot (capped at
    /// [`super::sdt_alloc::MAX_SDT_ALLOC_ENTRIES`]) BTF-rendered to
    /// named field views. Empty when no scheduler-side allocator
    /// could be located, when arena offsets / sdt_alloc offsets are
    /// absent, or when the program BTF lacks the `scx_allocator`
    /// type (scheduler doesn't link `lib/sdt_alloc.bpf.c`).
    ///
    /// Populated alongside the page-granular [`ArenaSnapshot`] in
    /// each map: a consumer can read either representation depending
    /// on whether they want raw bytes or named-field allocations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sdt_allocations: Vec<SdtAllocatorSnapshot>,
}

impl Default for FailureDumpReport {
    /// Empty report with `schema = "single"`. Pinning the schema
    /// here keeps `FailureDumpReport::default()` and a
    /// freshly-constructed `FailureDumpReport { ..., schema:
    /// SCHEMA_SINGLE.into(), ... }` indistinguishable to consumers,
    /// so the schema discriminant is never quietly missing on a
    /// default-built report.
    fn default() -> Self {
        Self {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
        }
    }
}

/// Pair of failure-dump snapshots captured at two points in a stall.
///
/// `early` is taken when the host-side runnable_at scanner observes
/// any task with `jiffies - p->scx.runnable_at > watchdog_timeout/2`
/// (mirrors the kernel's `check_rq_for_timeouts` walk over
/// `rq->scx.runnable_list`). `late` is taken at the same trigger as
/// the single-snapshot path: the BPF probe's
/// `ktstr_err_exit_detected` latch flipping after a sched_ext
/// error-class exit.
///
/// `early == None` when the watchdog half-way threshold never
/// triggered before `late` fired (e.g. an immediate scheduler error
/// in `init_task` before any task became runnable). Diffing
/// `late` against `early` shows what BPF state changed during the
/// stall window — the value-add over the single-snapshot dump.
///
/// **No user toggle — auto-repro engages this automatically.** Only
/// the auto-repro VM emits this shape;
/// [`crate::test_support::probe::attempt_auto_repro`] is the
/// single call site flipping the builder's `dual_snapshot` flag,
/// and there is no public ktstr surface for asking for it from a
/// primary VM. Test authors don't need to know about it — when an
/// auto-repro fires, the file at `<test>.repro.failure-dump.json`
/// changes shape from [`FailureDumpReport`] to this wrapper.
///
/// Note: there is no `Default` impl. The `late` field is required
/// by the doc invariant ("the freeze coordinator only writes a
/// `DualFailureDumpReport` after the late snapshot has been
/// captured"); a `Default::default()` would have produced a wrapper
/// with an empty late report whose `maps`/`vcpu_regs` vectors
/// silently lie about a successful capture. Construct via the
/// struct literal with an explicit `late: FailureDumpReport`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DualFailureDumpReport {
    /// Wire-format discriminant. Always `"dual"` for this variant,
    /// pinning [`SCHEMA_DUAL`]. Mirror of [`FailureDumpReport::schema`]
    /// — consumers branch on it before deserializing.
    #[serde(default = "default_schema_dual")]
    pub schema: String,
    /// Snapshot at the watchdog half-way point. `None` when the
    /// stall fired before the half-way scanner crossed its threshold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early: Option<FailureDumpReport>,
    /// Snapshot at the error-exit latch trigger. Always present
    /// (the freeze coordinator only writes a `DualFailureDumpReport`
    /// after the late snapshot has been captured; if the run ends
    /// with only an early snapshot the file is not written at all).
    pub late: FailureDumpReport,
    /// Maximum `jiffies - p->scx.runnable_at` observed by the
    /// runnable_at scanner at the moment the early snapshot fired.
    /// Zero when `early` is `None`.
    ///
    /// Diff against the kernel's `watchdog_timeout` (carried
    /// alongside as [`Self::early_threshold_jiffies`] doubled — the
    /// scanner trigger is half the watchdog) to see how close the
    /// system was to the SCX_EXIT_ERROR_STALL emission line at the
    /// early-trigger point.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub early_max_age_jiffies: u64,
    /// The half-way trigger threshold the scanner compared against
    /// when capturing the early snapshot, expressed in guest
    /// jiffies. Equals `(watchdog_timeout_ms * CONFIG_HZ) / 1000 / 2`
    /// at the moment the snapshot fired. Zero when `early` is
    /// `None`.
    ///
    /// Surfaced alongside `early_max_age_jiffies` so a downstream
    /// consumer reading the JSON does not have to recompute the
    /// kernel-internal jiffies arithmetic to reproduce the
    /// trigger condition.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub early_threshold_jiffies: u64,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

impl std::fmt::Display for DualFailureDumpReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Summary header: a one-line at-a-glance description so an
        // operator scanning logs sees the shape (early present /
        // absent, late map + vcpu_regs counts, plus the trigger
        // metric and threshold when early fired) before paging
        // through the full body.
        let n_maps = self.late.maps.len();
        let m_vcpu_regs = self.late.vcpu_regs.len();
        if self.early.is_some() {
            write!(
                f,
                "DualFailureDumpReport: early=present (max_age={}j, threshold={}j), \
                 late=({n_maps} maps, {m_vcpu_regs} vcpu_regs)\n\n",
                self.early_max_age_jiffies, self.early_threshold_jiffies,
            )?;
        } else {
            write!(
                f,
                "DualFailureDumpReport: early=absent, late=({n_maps} maps, \
                 {m_vcpu_regs} vcpu_regs)\n\n",
            )?;
        }
        match &self.early {
            Some(early) => {
                f.write_str("early snapshot (sched_ext watchdog half-way):\n")?;
                std::fmt::Display::fmt(early, f)?;
                f.write_str("\n\nlate snapshot (error-exit):\n")?;
                std::fmt::Display::fmt(&self.late, f)
            }
            None => {
                f.write_str(
                    "late snapshot (error-exit; early snapshot absent \
                     (stall fired before half-way threshold, or runnable_at \
                     scan setup failed)):\n",
                )?;
                std::fmt::Display::fmt(&self.late, f)
            }
        }
    }
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
pub struct FailureDumpMap {
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
    pub entries: Vec<FailureDumpEntry>,
    /// Per-CPU slots for PERCPU_ARRAY maps. Outer Vec indexed by key,
    /// inner Vec indexed by CPU id.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub percpu_entries: Vec<FailureDumpPercpuEntry>,
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
pub struct FailureDumpEntry {
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
pub struct FailureDumpPercpuEntry {
    pub key: u32,
    pub per_cpu: Vec<Option<RenderedValue>>,
}

impl std::fmt::Display for FailureDumpReport {
    /// Human-readable rendering of every map plus per-vCPU register
    /// snapshots. JSON remains the programmatic form via
    /// `serde_json`; this Display is the default presentation used
    /// in test-failure output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.maps.is_empty() && self.vcpu_regs.is_empty() && self.sdt_allocations.is_empty() {
            return f.write_str("(empty failure dump)");
        }
        let mut first = true;
        for m in &self.maps {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            std::fmt::Display::fmt(m, f)?;
        }
        if !self.vcpu_regs.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            f.write_str("vcpu_regs:")?;
            for (i, slot) in self.vcpu_regs.iter().enumerate() {
                f.write_str("\n  ")?;
                match slot {
                    Some(s) => write!(f, "vcpu {i}: {s}")?,
                    None => write!(f, "vcpu {i}: <unavailable>")?,
                }
            }
        }
        for snap in &self.sdt_allocations {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            std::fmt::Display::fmt(snap, f)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for FailureDumpMap {
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

impl std::fmt::Display for FailureDumpEntry {
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

impl std::fmt::Display for FailureDumpPercpuEntry {
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
/// framework-internal — the user looking at a failure dump for their
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
) -> FailureDumpReport {
    let maps = accessor.maps();
    let mut report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::with_capacity(maps.len()),
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
    };

    // Per-map program-BTF cache, keyed by `btf_kva`. Each unique
    // `struct btf *` lives in the kernel BTF IDR — multiple maps from
    // the same BPF program point at the same KVA, so caching dedupes
    // the heavy `Btf::from_bytes`/`from_split_bytes` parse across them
    // (a scheduler with N maps backed by one BPF object pays one
    // parse, not N). Lookups go through this cache before falling
    // back to the caller-supplied vmlinux `btf`.
    let mut program_btfs: std::collections::HashMap<u64, Btf> = std::collections::HashMap::new();

    // Bookkeeping for the sdt_alloc walker that runs after the map
    // loop. We need: (1) the raw .bss bytes from the scheduler's
    // global-section ARRAY map, (2) the kern_vm_start from any arena
    // map that snapshot_arena populated, (3) one program BTF
    // (`btf_kva` of the scheduler's BPF object) so we can resolve
    // sdt_alloc struct offsets and the allocator's .bss byte offset.
    let mut sched_bss_bytes: Option<(Vec<u8>, u64)> = None; // (bytes, btf_kva)
    let mut arena_kern_vm_start: u64 = 0;

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

        let rendered = render_map(accessor, map_btf, &info, num_cpus, arena_offsets);

        // Cache the scheduler's `.bss` raw bytes for the post-pass
        // sdt_alloc walker. libbpf composes `<obj>.bss` for the
        // scheduler's global-section map and the framework probes
        // were already filtered above, so the first ARRAY map ending
        // in `.bss` with a non-zero `btf_kva` is the right one. Cap
        // at one — multiple BPF objects in one scheduler is theoretical
        // for ktstr's surface today.
        if sched_bss_bytes.is_none()
            && info.map_type == BPF_MAP_TYPE_ARRAY
            && info.btf_kva != 0
            && info.name.ends_with(".bss")
            && let Some(bytes) = accessor.read_value(&info, 0, info.value_size as usize)
        {
            sched_bss_bytes = Some((bytes, info.btf_kva));
        }

        // Cache kern_vm_start from the first arena map whose
        // snapshot succeeded — sdt_alloc's `__arena` pointers all
        // index this same window, regardless of which map declared
        // it. (lib/arena_map.h declares one __weak arena per BPF
        // object; multiple linked objects would each see their own.)
        if arena_kern_vm_start == 0
            && let Some(snap) = rendered.arena.as_ref()
            && snap.kern_vm_start != 0
        {
            arena_kern_vm_start = snap.kern_vm_start;
        }

        report.maps.push(rendered);
    }

    // Post-pass: walk sdt_alloc trees if all prerequisites lined up.
    // The walk is best-effort and silent: any missing prerequisite
    // (no scheduler .bss, no arena window, no program BTF, no
    // `scx_allocator` type) leaves `sdt_allocations` empty rather
    // than failing the dump.
    if let Some((bss_bytes, btf_kva)) = sched_bss_bytes
        && arena_kern_vm_start != 0
        && let Some(prog_btf) = program_btfs.get(&btf_kva)
        && let Ok(sdt_offsets) = SdtAllocOffsets::from_btf(prog_btf)
    {
        // Locate every sdt_alloc allocator instance declared in
        // `.bss`. The Datasec walk gives us each variable's name and
        // offset; we filter to types matching `struct scx_allocator`
        // by re-resolving the var's chained type. A scheduler may
        // declare more than one allocator (e.g. one per-task, one
        // per-cgroup) so we iterate all of them.
        for (var_name, var_offset, var_type_id) in iter_bss_vars_with_type(prog_btf, ".bss") {
            // Only walk vars whose type is `struct scx_allocator`.
            if !is_scx_allocator_type(prog_btf, var_type_id) {
                continue;
            }
            // Slice the in-bss bytes for one full `struct scx_allocator`.
            // The size comes from BTF (resolved into `allocator_size`
            // by `SdtAllocOffsets::from_btf`); using the BTF-reported
            // size means a future field appended to scx_allocator
            // doesn't silently slip past the slice end.
            let Some(slice_end) = var_offset.checked_add(sdt_offsets.allocator_size) else {
                continue;
            };
            let slice = match bss_bytes.get(var_offset..slice_end) {
                Some(s) => s,
                None => continue,
            };

            // Discover the payload BTF type id from the elem_size
            // we'd read in the walker. We do a small read here just
            // to drive the heuristic; the walker re-reads it.
            let pool_off = sdt_offsets.allocator_pool + sdt_offsets.pool_elem_size;
            let elem_size = if pool_off + 8 <= slice.len() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&slice[pool_off..pool_off + 8]);
                u64::from_le_bytes(buf)
            } else {
                0
            };
            let payload_size =
                elem_size.saturating_sub(sdt_offsets.data_header_size as u64) as usize;
            let choice = discover_payload_btf_id(prog_btf, payload_size);

            let snap = walk_sdt_allocator(
                accessor.kernel(),
                arena_kern_vm_start,
                slice,
                &sdt_offsets,
                prog_btf,
                choice.btf_type_id,
                choice.reason,
                var_name,
            );
            // Surface only allocators with a non-empty result OR a
            // diagnostic elem_size; an all-zero snapshot from a
            // never-initialized allocator is just noise.
            if !snap.entries.is_empty() || snap.elem_size != 0 {
                report.sdt_allocations.push(snap);
            }
        }
    }

    report
}

/// Walk a Datasec section by name, yielding `(var_name, byte_offset,
/// type_id)` for every variable declared in it.
///
/// Used by [`dump_state`] to enumerate `.bss` variables when looking
/// for `scx_allocator` instances. Returns an empty iterator when the
/// Datasec doesn't exist or any chained Var resolution fails — the
/// caller treats that as "no sdt_alloc state to surface" rather than
/// a hard error.
fn iter_bss_vars_with_type(btf: &Btf, section_name: &str) -> Vec<(String, usize, u32)> {
    use btf_rs::BtfType;
    let mut out = Vec::new();
    let Ok(candidates) = btf.resolve_types_by_name(section_name) else {
        return out;
    };
    for ty in candidates {
        let btf_rs::Type::Datasec(ds) = ty else {
            continue;
        };
        for var_info in &ds.variables {
            let Ok(chained) = btf.resolve_chained_type(var_info) else {
                continue;
            };
            let btf_rs::Type::Var(var) = chained else {
                continue;
            };
            let Ok(name) = btf.resolve_name(&var) else {
                continue;
            };
            // The Var's type_id points to the variable's actual
            // type (e.g. struct scx_allocator). var_info.offset() is
            // the byte offset within the Datasec.
            let Ok(type_id) = var.get_type_id() else {
                continue;
            };
            out.push((name, var_info.offset() as usize, type_id));
        }
    }
    out
}

/// True iff `type_id` resolves to a struct named `scx_allocator`,
/// stripping the BTF modifier chain en route. The five modifier
/// kinds the loop unwraps — `Const`, `Volatile`, `Typedef`,
/// `Restrict`, `TypeTag` — are the complete set the kernel BPF
/// pipeline emits for global variable types in `.bss`. Any other
/// kind in the chain (Ptr, Array, etc.) terminates the lookup with
/// a non-match.
fn is_scx_allocator_type(btf: &Btf, type_id: u32) -> bool {
    use btf_rs::Type as T;
    // Mirror the modifier-chain pattern in
    // `btf_offsets::resolve_member_composite` — resolve the
    // chained type via the BtfType trait object so the type
    // aliases (Const = Volatile, TypeTag = Typedef) all share the
    // same path through the loop.
    let Ok(mut t) = btf.resolve_type_by_id(type_id) else {
        return false;
    };
    for _ in 0..20 {
        match t {
            T::Struct(s) => {
                return btf.resolve_name(&s).is_ok_and(|n| n == "scx_allocator");
            }
            T::Const(_) | T::Volatile(_) | T::Typedef(_) | T::Restrict(_) | T::TypeTag(_) => {
                let Some(btf_ty) = t.as_btf_type() else {
                    return false;
                };
                let Ok(next) = btf.resolve_chained_type(btf_ty) else {
                    return false;
                };
                t = next;
            }
            _ => return false,
        }
    }
    false
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
) -> FailureDumpMap {
    let mut out = FailureDumpMap {
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
                out.entries.push(FailureDumpEntry {
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
                    .push(FailureDumpPercpuEntry { key, per_cpu });
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
            // Arena maps render in two phases:
            //
            //   1. Page-granular: arena pages live in vmalloc space
            //      and translate via the existing PTE walker. Each
            //      mapped page surfaces here as a 4 KiB ArenaPage —
            //      raw bytes the operator can post-process against
            //      the program's own layout documentation.
            //
            //   2. Structured (sdt_alloc post-pass): when the
            //      scheduler links `lib/sdt_alloc.bpf.c`, the
            //      `dump_state` post-pass walks `scx_allocator`'s
            //      radix tree and produces named-field
            //      [`super::sdt_alloc::SdtAllocEntry`] records under
            //      [`FailureDumpReport::sdt_allocations`]. That phase
            //      is gated on the program BTF carrying
            //      `struct scx_allocator` — schedulers that don't use
            //      sdt_alloc still get the page-granular fallback
            //      from this arm.
            //
            // Both representations land in the same dump so a
            // consumer can pick whichever fits — raw bytes for ad
            // hoc post-processing, structured records for typed
            // field views.
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
            out.error = Some(format!(
                "map_type {other} not yet supported by failure dump"
            ));
        }
    }

    out
}

/// Render a byte slice as space-separated hex pairs.
///
/// `pub(crate)` so [`super::sdt_alloc`] can reuse the same wire shape
/// for its hex-fallback payload renderings — keeps the dump's hex
/// output consistent across both renderers.
pub(crate) fn hex_dump(bytes: &[u8]) -> String {
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
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![FailureDumpMap {
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
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
        };
        let json = serde_json::to_string(&report).unwrap();
        let parsed: FailureDumpReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.maps.len(), 1);
        assert_eq!(parsed.maps[0].name, "scx_demo.bss");
        assert_eq!(parsed.maps[0].max_entries, 1);
    }

    #[test]
    fn empty_report_serde() {
        let report = FailureDumpReport::default();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: FailureDumpReport = serde_json::from_str(&json).unwrap();
        assert!(parsed.maps.is_empty());
    }

    // ---- Display impl coverage --------------------------------------
    //
    // The Display impl is the human-readable form used in test
    // failure output. Pin its layout against representative shapes.

    fn make_simple_map() -> FailureDumpMap {
        FailureDumpMap {
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
        let report = FailureDumpReport::default();
        assert_eq!(format!("{report}"), "(empty failure dump)");
    }

    #[test]
    fn report_display_one_map_with_value() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![make_simple_map()],
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
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
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![make_simple_map(), make_simple_map()],
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
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
        let entry = FailureDumpEntry {
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
        let entry = FailureDumpEntry {
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
        let entry = FailureDumpPercpuEntry {
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

    // ---- vcpu_regs Display coverage ---------------------------------

    #[test]
    fn report_display_includes_vcpu_regs_section() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: vec![
                Some(VcpuRegSnapshot {
                    instruction_pointer: 0x1,
                    stack_pointer: 0x2,
                    page_table_root: 0x3,
                    user_page_table_root: None,
                }),
                None,
                Some(VcpuRegSnapshot {
                    instruction_pointer: 0xa,
                    stack_pointer: 0xb,
                    page_table_root: 0xc,
                    user_page_table_root: None,
                }),
            ],
            sdt_allocations: Vec::new(),
        };
        let out = format!("{report}");
        // Section header.
        assert!(out.starts_with("vcpu_regs:"), "missing header: {out}");
        // Three vCPU rows: 0 with values, 1 unavailable, 2 with values.
        assert!(out.contains("vcpu 0: ip=0x"), "missing vcpu 0: {out}");
        assert!(
            out.contains("vcpu 1: <unavailable>"),
            "missing vcpu 1 marker: {out}"
        );
        assert!(out.contains("vcpu 2: ip=0x"), "missing vcpu 2: {out}");
    }

    #[test]
    fn report_display_pairs_maps_and_vcpu_regs_with_blank_line() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: vec![make_simple_map()],
            vcpu_regs: vec![Some(VcpuRegSnapshot {
                instruction_pointer: 0x1,
                stack_pointer: 0x2,
                page_table_root: 0x3,
                user_page_table_root: None,
            })],
            sdt_allocations: Vec::new(),
        };
        let out = format!("{report}");
        // Map block, blank line, vcpu_regs section.
        assert!(out.contains("\n\nvcpu_regs:"));
    }

    #[test]
    fn report_display_empty_with_only_vcpu_regs_does_not_say_empty_dump() {
        // An all-empty maps Vec but populated vcpu_regs must still
        // render rather than fall through to "(empty failure dump)".
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: vec![None],
            sdt_allocations: Vec::new(),
        };
        let out = format!("{report}");
        assert_eq!(out, "vcpu_regs:\n  vcpu 0: <unavailable>");
    }

    /// Pin the wire shape of a partial dump — the
    /// "all_parked but dump prerequisites unavailable" branch in
    /// `vmm::run_vm`'s freeze coordinator builds exactly this
    /// shape: empty `maps`, populated `vcpu_regs`. Operators
    /// reading the JSON / Display output rely on:
    ///   - Display NOT rendering the "(empty failure dump)"
    ///     fallback (which would mask the partial),
    ///   - Display starting with the `vcpu_regs:` section,
    ///   - JSON serialising `"maps":[]` (NOT skipped, since
    ///     `Vec::is_empty` is the skip condition only for
    ///     `vcpu_regs` and a few `Option`/`Vec` fields inside
    ///     `FailureDumpMap`, not for the top-level `maps` field).
    #[test]
    fn report_display_partial_with_populated_regs_and_empty_maps() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: vec![Some(VcpuRegSnapshot {
                instruction_pointer: 0xdead,
                stack_pointer: 0xbeef,
                page_table_root: 0xcafe,
                user_page_table_root: None,
            })],
            sdt_allocations: Vec::new(),
        };

        // (a) Display: vcpu_regs section present, no fallback.
        let out = format!("{report}");
        assert!(
            out.contains("vcpu_regs:"),
            "Display must contain the vcpu_regs section: {out}"
        );
        assert!(
            out.contains("vcpu 0: ip=0x"),
            "Display must render the BSP register row: {out}"
        );
        assert!(
            !out.contains("(empty failure dump)"),
            "Display must NOT fall through to empty fallback when \
             vcpu_regs is populated: {out}"
        );

        // (b) JSON: maps key present as empty array, NOT
        // skipped — operators downstream reliably distinguish
        // "no maps captured (partial)" from "maps key absent
        // (regression / older schema)".
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(
            json.contains("\"maps\":[]"),
            "JSON must carry empty `maps` array (not skip): {json}"
        );
        assert!(
            json.contains("\"vcpu_regs\""),
            "JSON must carry vcpu_regs key: {json}"
        );
    }

    // -- DualFailureDumpReport serde + Display tests --

    /// Roundtrip a `DualFailureDumpReport` with a populated early
    /// snapshot and non-zero metric/threshold fields. Pins the wire
    /// format on the dual-snapshot side: the wrapper deserialises
    /// back with `early` present and the jiffies fields preserved.
    #[test]
    fn dual_report_serde_roundtrip_with_early() {
        let early = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: vec![None],
            sdt_allocations: Vec::new(),
        };
        let late = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: vec![None, None],
            sdt_allocations: Vec::new(),
        };
        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: Some(early),
            late,
            early_max_age_jiffies: 1234,
            early_threshold_jiffies: 600,
        };
        let json = serde_json::to_string(&dual).unwrap();
        let parsed: DualFailureDumpReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.schema, SCHEMA_DUAL);
        assert!(parsed.early.is_some(), "early must roundtrip: {json}");
        assert_eq!(parsed.early_max_age_jiffies, 1234);
        assert_eq!(parsed.early_threshold_jiffies, 600);
        assert_eq!(parsed.late.vcpu_regs.len(), 2);
    }

    /// Zero `early_max_age_jiffies` / `early_threshold_jiffies`
    /// must be skipped on serialize (per the
    /// `skip_serializing_if = is_zero_u64` attributes). Pinning
    /// this keeps the JSON tight when the early snapshot did not
    /// fire — a `late`-only run yields a wrapper without the
    /// trigger-metric noise.
    #[test]
    fn dual_report_serde_skips_zero_jiffies_fields() {
        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: None,
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 0,
            early_threshold_jiffies: 0,
        };
        let json = serde_json::to_string(&dual).unwrap();
        assert!(
            !json.contains("early_max_age_jiffies"),
            "zero early_max_age_jiffies must skip: {json}"
        );
        assert!(
            !json.contains("early_threshold_jiffies"),
            "zero early_threshold_jiffies must skip: {json}"
        );
    }

    /// Non-zero jiffies fields must serialize so a downstream
    /// consumer can recover the trigger condition without
    /// recomputing kernel arithmetic. Mirror of the
    /// `skips_zero_jiffies_fields` test.
    #[test]
    fn dual_report_serde_emits_nonzero_jiffies_fields() {
        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: Some(FailureDumpReport::default()),
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 4096,
            early_threshold_jiffies: 2048,
        };
        let json = serde_json::to_string(&dual).unwrap();
        assert!(
            json.contains("\"early_max_age_jiffies\":4096"),
            "non-zero max_age must serialize: {json}"
        );
        assert!(
            json.contains("\"early_threshold_jiffies\":2048"),
            "non-zero threshold must serialize: {json}"
        );
    }

    /// The `schema` field is the wire-format discriminant.
    /// `FailureDumpReport` carries `"single"`,
    /// `DualFailureDumpReport` carries `"dual"`, and the two
    /// values are distinguishable so a consumer can inspect a
    /// single field before deciding which type to deserialize
    /// into.
    #[test]
    fn dual_report_schema_distinguishes_from_single() {
        let single = FailureDumpReport::default();
        let single_json = serde_json::to_string(&single).unwrap();
        assert!(
            single_json.contains(&format!("\"schema\":\"{SCHEMA_SINGLE}\"")),
            "single carries schema='single': {single_json}"
        );

        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: None,
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 0,
            early_threshold_jiffies: 0,
        };
        let dual_json = serde_json::to_string(&dual).unwrap();
        assert!(
            dual_json.contains(&format!("\"schema\":\"{SCHEMA_DUAL}\"")),
            "dual carries schema='dual': {dual_json}"
        );
        // The two discriminants are distinct strings — a consumer
        // checking the field can tell the variants apart without
        // attempting deserialization first.
        assert_ne!(SCHEMA_SINGLE, SCHEMA_DUAL);
    }

    /// Display output for the early=present branch carries the
    /// summary header AND the jiffies metadata, so an operator
    /// scanning a log can see at a glance whether the early
    /// snapshot fired and what trigger condition produced it.
    #[test]
    fn dual_report_display_present_carries_jiffies() {
        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: Some(FailureDumpReport::default()),
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 9001,
            early_threshold_jiffies: 4500,
        };
        let s = format!("{dual}");
        assert!(
            s.contains("early=present"),
            "Display must say early=present: {s}"
        );
        assert!(
            s.contains("max_age=9001j"),
            "Display must surface max_age: {s}"
        );
        assert!(
            s.contains("threshold=4500j"),
            "Display must surface threshold: {s}"
        );
    }

    /// Display output for the early=absent branch carries the
    /// summary header AND the documented absence-reason text
    /// describing both possible causes (stall fired before the
    /// half-way threshold; runnable_at scan setup failed).
    #[test]
    fn dual_report_display_absent_names_both_causes() {
        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: None,
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 0,
            early_threshold_jiffies: 0,
        };
        let s = format!("{dual}");
        assert!(
            s.contains("early=absent"),
            "Display must say early=absent: {s}"
        );
        assert!(
            s.contains("stall fired before half-way threshold"),
            "Display must name the threshold-not-reached cause: {s}"
        );
        assert!(
            s.contains("runnable_at scan setup failed"),
            "Display must name the scan-setup-failure cause: {s}"
        );
    }
}
