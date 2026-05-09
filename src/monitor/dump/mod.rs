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
//!   the whole value buffer and render it via [`btf_render::render_value_with_mem`]
//!   so embedded `__arena` pointers chase into the captured arena pages.
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

mod display;
mod render_map;
#[cfg(test)]
mod tests;
use render_map::*;

use serde::{Deserialize, Serialize};

use btf_rs::Btf;

use super::arena::{ArenaSnapshot, BpfArenaOffsets, snapshot_arena};
use super::bpf_map::{
    BPF_MAP_TYPE_ARENA, BPF_MAP_TYPE_ARRAY, BpfMapAccessor, BpfMapInfo, GuestMemMapAccessor,
};
use super::btf_render::RenderedValue;
use super::sdt_alloc::{
    SdtAllocOffsets, SdtAllocatorSnapshot, discover_payload_btf_id, walk_sdt_allocator,
};

/// Borrow-only capture context for per-program runtime stats
/// (cnt/nsecs/misses) populated alongside the BPF map dump.
///
/// Carries a borrowed [`super::bpf_prog::BpfProgAccessor`] plus the
/// per-CPU offset array obtained from
/// [`super::symbols::read_per_cpu_offsets`]. [`dump_state`] calls
/// [`super::bpf_prog::BpfProgAccessor::struct_ops_runtime_stats`]
/// with the supplied offsets and stores the resulting
/// [`super::bpf_prog::ProgRuntimeStats`] vector in
/// [`FailureDumpReport::prog_runtime_stats`].
///
/// Pass `None` to skip prog-runtime capture (e.g. when the
/// accessor could not be constructed because `prog_idr` is
/// missing or the BPF prog offsets did not resolve). The dump still
/// renders every map the [`super::bpf_map::BpfMapAccessor`] enumerates.
pub struct ProgRuntimeCapture<'a> {
    /// Accessor for walking `prog_idr` and reading per-program
    /// `bpf_prog_stats` slots. Trait dispatch lets the same dump
    /// site consume either the guest-memory backend or the planned
    /// live-host backend without committing to a concrete type.
    pub accessor: &'a dyn super::bpf_prog::BpfProgAccessor,
    /// Per-CPU offset array (`__per_cpu_offset[cpu]`) used to address
    /// each CPU's `bpf_prog_stats` slot for summation.
    pub per_cpu_offsets: &'a [u64],
}

/// Borrow-only capture context for per-CPU CPU-time / softirq / IRQ
/// counters populated alongside the BPF map dump.
///
/// Carries the BTF-resolved field offsets for `kernel_cpustat`,
/// `kernel_stat`, and `tick_sched`, the resolved `.data..percpu`
/// section offsets of the three per-CPU symbols, and the
/// `__per_cpu_offset[cpu]` array used to address each CPU's slot.
///
/// [`dump_state`] reads each CPU's slot via direct guest-memory
/// reads against the supplied [`super::reader::GuestMem`] and
/// records the result into [`FailureDumpReport::per_cpu_time`].
/// Mirrors [`ProgRuntimeCapture`]'s "borrowed-only, optional"
/// shape — when `None`, the dump skips the per-CPU time capture
/// and leaves the field empty.
///
/// Skipped silently when the resolver could not locate any of the
/// three per-CPU symbols (stripped vmlinux), the BTF offsets are
/// not present (CPU-time accounting types missing), or
/// `__per_cpu_offset` resolution returned an empty array. The
/// capture is best-effort diagnostic data; its absence does not
/// fail the dump.
pub struct CpuTimeCapture<'a> {
    /// Guest memory handle used to read each per-CPU slot.
    pub mem: &'a super::reader::GuestMem,
    /// BTF-resolved offsets for `kernel_cpustat::cpustat[]`,
    /// `kernel_stat::softirqs[]`, `kernel_stat::irqs_sum`, and
    /// optionally `tick_sched::iowait_sleeptime`.
    pub offsets: &'a super::btf_offsets::CpuTimeOffsets,
    /// Section-relative `.data..percpu` offset of the
    /// `kernel_cpustat` per-CPU symbol. Each CPU's KVA is
    /// `kernel_cpustat + per_cpu_offsets[cpu]`.
    pub kernel_cpustat_kva: u64,
    /// Section-relative `.data..percpu` offset of the `kstat`
    /// per-CPU symbol.
    pub kstat_kva: u64,
    /// Section-relative `.data..percpu` offset of the `tick_cpu_sched`
    /// per-CPU symbol. `None` when the kernel was built without
    /// `CONFIG_NO_HZ_COMMON`; iowait_sleeptime capture is skipped.
    pub tick_cpu_sched_kva: Option<u64>,
    /// Per-CPU offset array (`__per_cpu_offset[cpu]`) — same array
    /// the BPF prog-stats walker uses (see
    /// [`super::symbols::read_per_cpu_offsets`]). Length determines
    /// how many CPUs the walker visits.
    pub per_cpu_offsets: &'a [u64],
    /// Guest's `PAGE_OFFSET` (resolved via
    /// [`super::symbols::resolve_page_offset`]). Used to translate
    /// each CPU's per-CPU KVA to a guest physical address for the
    /// memory read.
    pub page_offset: u64,
}

/// Borrow-only capture context for per-task enrichment.
///
/// Carries the [`super::guest::GuestKernel`] (guest memory + symbol
/// table), the BTF-resolved task/signal/pid/upid offsets, the cached
/// sched_class symbol KVAs (for class-name decode and the
/// PI-boost-out-of-SCX flag), the lock-slowpath symbol cache (for
/// stack-trace pattern matching), AND the task list itself — a
/// pre-collected `&[TaskWalkerEntry]` produced by a task walker
/// (rq->scx walk, DSQ walk, init_task→tasks enumeration).
///
/// Mirrors the [`ProgRuntimeCapture`] / [`CpuTimeCapture`]
/// borrowed-only-optional shape. When `dump_state` receives
/// `Some(TaskEnrichmentCapture)`, it iterates `tasks` and calls
/// [`super::task_enrichment::walk_task_enrichment`] for each entry,
/// pushing results into [`FailureDumpReport::task_enrichments`]. When
/// `None`, the field stays empty and
/// [`FailureDumpReport::task_enrichments_unavailable`] gets a
/// "no task walker available" diagnostic.
///
/// The walker producer (rq->scx walker etc.) is responsible for
/// building this struct. Until walker dispatch lands, no walker
/// exists; the freeze coordinator passes `None` and the field is
/// plumbed but empty.
pub struct TaskEnrichmentCapture<'a> {
    /// Borrowed GuestKernel — provides memory access, page-table
    /// translation context, and the vmlinux symbol table.
    pub kernel: &'a super::guest::GuestKernel,
    /// BTF-resolved offsets for the task/signal/pid/upid walk.
    pub offsets: &'a super::btf_offsets::TaskEnrichmentOffsets,
    /// Cached sched_class symbol KVAs for class decode + PI-boost
    /// flag.
    pub sched_classes: &'a super::task_enrichment::SchedClassRegistry,
    /// Cached lock-slowpath symbol KVAs for stack-PC pattern
    /// matching.
    pub lock_slowpaths: &'a super::task_enrichment::LockSlowpathRegistry,
    /// Tasks the walker discovered, plus per-task metadata
    /// `walk_task_enrichment` needs (see [`TaskWalkerEntry`]).
    pub tasks: &'a [TaskWalkerEntry],
}

/// One entry produced by a task walker (rq->scx, DSQ, etc.) for the
/// enrichment capture pipeline.
///
/// Each task walker discovers task KVAs by traversing the kernel's
/// own scheduling data structures; the walker also knows which task
/// was reachable via `rq->scx.runnable_list` (used for the
/// PI-boost-out-of-SCX flag) and which vCPU's instruction-pointer
/// matches the running task (used for the lock-slowpath stack
/// matcher). Capturing those signals at the walker site keeps the
/// enrichment surface side-effect free — `walk_task_enrichment` only
/// reads guest memory; it does not perform discovery itself.
#[derive(Debug, Clone, Copy)]
pub struct TaskWalkerEntry {
    /// Kernel virtual address of the `task_struct`.
    pub task_kva: u64,
    /// True iff the task was reached via `rq->scx.runnable_list`.
    /// Required for the PI-boost-out-of-SCX flag — see
    /// [`super::task_enrichment::TaskEnrichment::pi_boosted_out_of_scx`].
    pub is_runnable_in_scx: bool,
    /// Optional instruction pointer for the lock-slowpath stack
    /// matcher. Pass the corresponding vCPU's
    /// [`VcpuRegSnapshot::instruction_pointer`] when this task was
    /// running on that vCPU at freeze time; pass `None` for tasks
    /// not actively running.
    pub running_pc: Option<u64>,
}

/// Per-CPU CPU-time / softirq / IRQ snapshot captured at freeze
/// time. One entry per CPU index visible to the host walker.
///
/// All counter fields are monotonic in the kernel — the freeze
/// captures the instantaneous value at the moment the vCPUs
/// rendezvous-park. Diffing two snapshots (or comparing against a
/// pre-test baseline) is the consumer's job; this type does not
/// derive deltas.
///
/// Field semantics match the kernel sources verified by PhD:
///   - `cpustat_*_ns`: ns counter from
///     `kernel_cpustat::cpustat[CPUTIME_*]`. Updated by
///     `account_user_time` / `account_system_index_time` and
///     siblings (`kernel/sched/cputime.c`). The kernel stores
///     nanoseconds; `/proc/stat` divides by `cputime_to_clock_t`.
///   - `softirqs[i]`: `kernel_stat::softirqs[i]` cumulative count
///     incremented by `kstat_incr_softirqs_this_cpu` on every
///     softirq raise. Indexed by [`super::btf_offsets::SOFTIRQ_NAMES`].
///   - `irqs_sum`: `kernel_stat::irqs_sum` cumulative count
///     incremented by `kstat_incr_irq_this_cpu` on every hardirq.
///   - `iowait_sleeptime_ns`: `tick_sched::iowait_sleeptime`
///     accumulated only under NO_HZ when the CPU enters idle with
///     `nr_iowait > 0`. `None` when CONFIG_NO_HZ_COMMON is off or
///     the resolver couldn't locate `tick_cpu_sched`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PerCpuTimeStats {
    /// CPU index (0-based) this entry describes.
    pub cpu: u32,
    /// `cpustat[CPUTIME_USER]` (ns).
    pub cpustat_user_ns: u64,
    /// `cpustat[CPUTIME_NICE]` (ns).
    pub cpustat_nice_ns: u64,
    /// `cpustat[CPUTIME_SYSTEM]` (ns).
    pub cpustat_system_ns: u64,
    /// `cpustat[CPUTIME_SOFTIRQ]` (ns).
    pub cpustat_softirq_ns: u64,
    /// `cpustat[CPUTIME_IRQ]` (ns).
    pub cpustat_irq_ns: u64,
    /// `cpustat[CPUTIME_IDLE]` (ns).
    pub cpustat_idle_ns: u64,
    /// `cpustat[CPUTIME_IOWAIT]` (ns).
    pub cpustat_iowait_ns: u64,
    /// `cpustat[CPUTIME_STEAL]` (ns).
    pub cpustat_steal_ns: u64,
    /// `kernel_stat::softirqs[]` per-vector cumulative counts.
    /// Indexed by [`super::btf_offsets::SOFTIRQ_NAMES`].
    pub softirqs: [u64; super::btf_offsets::NR_SOFTIRQS],
    /// `kernel_stat::irqs_sum` cumulative hardirq count.
    pub irqs_sum: u64,
    /// `tick_sched::iowait_sleeptime` accumulated NO_HZ idle time
    /// with outstanding IO (ns). `None` when NO_HZ disabled or
    /// `tick_cpu_sched` symbol was absent at resolve time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iowait_sleeptime_ns: Option<u64>,
}

/// Per-node NUMA event counters captured from
/// `pglist_data->node_zones[]->vm_numa_event[]` at freeze time.
///
/// Each row is one row of NUMA event counters summed across all
/// zones on a single node. The six counters mirror the kernel's
/// `enum numa_stat_item` (see [`super::btf_offsets::NUMA_HIT`]
/// etc. for the enum-stable indices). All counters are
/// monotonic-since-boot; consumers diff against a baseline (or
/// against another node's row) to extract the test-window delta.
///
/// Diagnostic value for sched_ext stalls is informational only —
/// the NUMA balancer is not active for ext tasks. The rows
/// surface here so an operator triaging a NUMA-aware workload
/// (e.g. a memory-tiering test) can verify the kernel actually
/// observed the expected node-locality distribution.
///
/// **Live walker status:** the wire shape, BTF offsets
/// ([`super::btf_offsets::NumaStatsOffsets`]), and report field
/// are wired through. The actual host-side walker that resolves
/// `node_data[]` and reads per-zone counters is pending; until it
/// lands, the report's [`FailureDumpReport::per_node_numa`] vec
/// stays empty and
/// [`FailureDumpReport::per_node_numa_unavailable`] carries the
/// `"no NUMA walker"` reason.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct PerNodeNumaStats {
    /// NUMA node id this row describes.
    pub node: u32,
    /// `vm_numa_event[NUMA_HIT]` summed across zones — pages
    /// allocated on the requested node when local was preferred.
    pub numa_hit: u64,
    /// `vm_numa_event[NUMA_MISS]` — local node full, allocation
    /// landed on a non-local node.
    pub numa_miss: u64,
    /// `vm_numa_event[NUMA_FOREIGN]` — process-policy targeted a
    /// different node, this node honored the policy.
    pub numa_foreign: u64,
    /// `vm_numa_event[NUMA_INTERLEAVE_HIT]` — interleave policy
    /// allocations that landed on this node.
    pub numa_interleave_hit: u64,
    /// `vm_numa_event[NUMA_LOCAL]` — allocations on this node by
    /// processes running on this node.
    pub numa_local: u64,
    /// `vm_numa_event[NUMA_OTHER]` — allocations on this node by
    /// processes running on a different node.
    pub numa_other: u64,
}

/// Reason string written into [`FailureDumpReport::per_node_numa_unavailable`]
/// when the per-node NUMA walker has not landed yet. Distinct from
/// other unavailable reasons so a downstream consumer can tell
/// "walker not implemented" apart from "walker ran and produced
/// no data" once the live producer ships.
pub const REASON_NO_NUMA_WALKER: &str = "no NUMA walker (host-side walker pending)";

/// Borrow-only capture context for the per-sample SCX event counter
/// timeline.
///
/// The freeze coordinator forwards the monitor sampler's accumulated
/// [`super::MonitorSample`] vec via [`Self::samples`]; the dump path
/// folds each sample's per-CPU [`super::ScxEventCounters`] into a
/// single cross-CPU sum and produces one [`EventCounterSample`] per
/// monitor tick.
///
/// `None` skips the timeline capture; the dump still renders the
/// rest of the report. Mirrors [`ProgRuntimeCapture`] /
/// [`CpuTimeCapture`]'s "borrowed-only, optional" shape.
pub struct EventCounterCapture<'a> {
    /// Periodic monitor samples gathered between VM start and the
    /// freeze trigger. Each sample carries per-CPU
    /// [`super::ScxEventCounters`] when scx event-stat offsets
    /// resolved; the dump folder skips samples whose CPUs all
    /// reported `event_counters: None`.
    pub samples: &'a [super::MonitorSample],
}

/// Borrow-only capture context for the rq->scx + DSQ walkers.
/// Mirrors [`TaskEnrichmentCapture`] / [`CpuTimeCapture`] shape —
/// `dump_state` consumes everything by reference.
///
/// Carries:
/// - `kernel`: GuestKernel handle for guest-memory reads
///   (PTE walks, symbol resolution).
/// - `offsets`: BTF-resolved
///   [`super::btf_offsets::ScxWalkerOffsets`] covering scx_rq,
///   scx_sched, scx_sched_pcpu, scx_sched_pnode, scx_dispatch_q,
///   sched_ext_entity, scx_dsq_list_node, rhashtable, bucket_table,
///   rhash_head.
/// - `scx_root_kva`: kernel-text-mapped pointer the walker
///   dereferences to find the active `scx_sched`.
/// - `rq_kvas` / `rq_pas`: per-CPU rq KVA + PA arrays; same vecs
///   the runnable_at scanner uses.
/// - `per_cpu_offsets`: `__per_cpu_offset[]` array — needed for
///   per-CPU bypass DSQ resolution.
/// - `nr_nodes`: NUMA node count, for the per-node global-DSQ
///   walk. Pass `1` on UMA / unknown configurations; the walker
///   gracefully skips slots whose pnode pointers are NULL.
///
/// When `None` is passed in [`DumpContext::scx_walker_capture`],
/// the dump emits empty `rq_scx_states` / `dsq_states` and
/// records `scx_walker_unavailable` with a diagnostic reason.
pub struct ScxWalkerCapture<'a> {
    /// Borrowed GuestKernel — provides memory access, page-table
    /// translation context, and the vmlinux symbol table.
    pub kernel: &'a super::guest::GuestKernel,
    /// BTF-resolved offsets for the scx walker.
    pub offsets: &'a super::btf_offsets::ScxWalkerOffsets,
    /// `scx_root` symbol KVA (resolved via vmlinux ELF symtab).
    /// The walker reads `*scx_root` to find the active scx_sched.
    pub scx_root_kva: u64,
    /// Per-CPU rq kernel virtual addresses (one per CPU).
    pub rq_kvas: &'a [u64],
    /// Per-CPU rq guest physical addresses (parallel to rq_kvas).
    pub rq_pas: &'a [u64],
    /// `__per_cpu_offset[]` array, used to address each CPU's
    /// `scx_sched_pcpu.bypass_dsq`.
    pub per_cpu_offsets: &'a [u64],
    /// NUMA node count for the per-node global-DSQ walk. Pass `1`
    /// on UMA / unknown configurations.
    pub nr_nodes: u32,
}

/// One per-monitor-tick snapshot of the 13 SCX_EV_* event counters
/// summed across every CPU at that tick.
///
/// The kernel stores per-CPU `s64` counters in `scx_sched_pcpu`
/// (kernel/sched/ext.c); the monitor sampler reads them at every
/// tick and stores per-CPU `event_counters` on each
/// [`super::CpuSnapshot`]. The dump path sums across CPUs into the
/// fields here so a downstream consumer can render the run's
/// counter timeline (sparkline, delta plot, ...) without
/// re-iterating the per-CPU vec.
///
/// Field semantics match
/// [`super::ScxEventCounters`] one-to-one — see that struct's
/// per-field doc for kernel-source provenance. `total_*` naming
/// here echoes [`super::ScxEventDeltas`]'s aggregate-across-window
/// fields but with per-tick (not per-window) granularity.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct EventCounterSample {
    /// Milliseconds since VM start (mirrors
    /// [`super::MonitorSample::elapsed_ms`]). Zero on the first
    /// sample.
    pub elapsed_ms: u64,
    /// Sum of `select_cpu_fallback` across all CPUs at this tick.
    pub select_cpu_fallback: i64,
    /// Sum of `dispatch_local_dsq_offline` across all CPUs.
    pub dispatch_local_dsq_offline: i64,
    /// Sum of `dispatch_keep_last` across all CPUs.
    pub dispatch_keep_last: i64,
    /// Sum of `enq_skip_exiting` across all CPUs.
    pub enq_skip_exiting: i64,
    /// Sum of `enq_skip_migration_disabled` across all CPUs.
    pub enq_skip_migration_disabled: i64,
    /// Sum of `reenq_immed` across all CPUs.
    pub reenq_immed: i64,
    /// Sum of `reenq_local_repeat` across all CPUs.
    pub reenq_local_repeat: i64,
    /// Sum of `refill_slice_dfl` across all CPUs.
    pub refill_slice_dfl: i64,
    /// Sum of `bypass_duration` across all CPUs (ns).
    pub bypass_duration: i64,
    /// Sum of `bypass_dispatch` across all CPUs.
    pub bypass_dispatch: i64,
    /// Sum of `bypass_activate` across all CPUs.
    pub bypass_activate: i64,
    /// Sum of `insert_not_owned` across all CPUs.
    pub insert_not_owned: i64,
    /// Sum of `sub_bypass_dispatch` across all CPUs.
    pub sub_bypass_dispatch: i64,
}

impl EventCounterSample {
    /// Construct from a [`super::MonitorSample`] by summing every
    /// CPU's [`super::ScxEventCounters`]. CPUs whose
    /// `event_counters` is `None` (event-stat offsets unresolved)
    /// contribute 0 to every field.
    ///
    /// Returns `None` when no CPU on the sample reported event
    /// counters — propagating that to the timeline would emit a
    /// row of all zeros that's indistinguishable from a real
    /// "every counter at zero" tick. Callers filter `None` out.
    pub fn from_monitor_sample(sample: &super::MonitorSample) -> Option<Self> {
        let mut any = false;
        let mut out = Self {
            elapsed_ms: sample.elapsed_ms,
            ..Self::default()
        };
        for cpu in &sample.cpus {
            if let Some(ev) = &cpu.event_counters {
                any = true;
                // Per-CPU SCX event counters are s64 in the kernel
                // and originate from BPF map reads of guest memory.
                // A corrupt counter could trip i64 addition overflow
                // when summed across many CPUs; saturating_add pins
                // the sum at i64::{MIN,MAX} rather than panicking
                // (debug) or wrapping (release) into a misleading
                // value.
                out.select_cpu_fallback = out
                    .select_cpu_fallback
                    .saturating_add(ev.select_cpu_fallback);
                out.dispatch_local_dsq_offline = out
                    .dispatch_local_dsq_offline
                    .saturating_add(ev.dispatch_local_dsq_offline);
                out.dispatch_keep_last =
                    out.dispatch_keep_last.saturating_add(ev.dispatch_keep_last);
                out.enq_skip_exiting = out.enq_skip_exiting.saturating_add(ev.enq_skip_exiting);
                out.enq_skip_migration_disabled = out
                    .enq_skip_migration_disabled
                    .saturating_add(ev.enq_skip_migration_disabled);
                out.reenq_immed = out.reenq_immed.saturating_add(ev.reenq_immed);
                out.reenq_local_repeat =
                    out.reenq_local_repeat.saturating_add(ev.reenq_local_repeat);
                out.refill_slice_dfl = out.refill_slice_dfl.saturating_add(ev.refill_slice_dfl);
                out.bypass_duration = out.bypass_duration.saturating_add(ev.bypass_duration);
                out.bypass_dispatch = out.bypass_dispatch.saturating_add(ev.bypass_dispatch);
                out.bypass_activate = out.bypass_activate.saturating_add(ev.bypass_activate);
                out.insert_not_owned = out.insert_not_owned.saturating_add(ev.insert_not_owned);
                out.sub_bypass_dispatch = out
                    .sub_bypass_dispatch
                    .saturating_add(ev.sub_bypass_dispatch);
            }
        }
        if any { Some(out) } else { None }
    }
}

/// Render a u64 counter series as a 1-line UTF-8 sparkline.
///
/// Maps each value into one of 8 unicode block-element glyphs
/// (`▁▂▃▄▅▆▇█`) by min-max scaling. Empty input renders as the
/// empty string; a constant non-zero series renders as repeated
/// mid-tier glyphs (matches the "no variation" reading in the
/// data, not as misleading monotonic up-bars). A constant zero
/// series renders as repeated lowest glyphs.
///
/// Used by the `Display` impl for the event-counter timeline. Pure
/// helper — no allocation outside the returned `String`.
pub fn render_sparkline(values: &[u64]) -> String {
    const GLYPHS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() {
        return String::new();
    }
    let min = *values.iter().min().expect("non-empty");
    let max = *values.iter().max().expect("non-empty");
    let mut s = String::with_capacity(values.len() * 4);
    if max == min {
        let glyph = if max == 0 {
            GLYPHS[0]
        } else {
            GLYPHS[GLYPHS.len() / 2]
        };
        for _ in values {
            s.push(glyph);
        }
        return s;
    }
    let span = max - min;
    let last_idx = (GLYPHS.len() - 1) as u64;
    for &v in values {
        // Linear scale [min, max] → [0, GLYPHS.len()-1]. Integer
        // math is sufficient (no rounding artifact at the cost
        // of one extra glyph step at boundaries).
        let scaled = ((v - min) * last_idx) / span;
        let idx = scaled.min(last_idx) as usize;
        s.push(GLYPHS[idx]);
    }
    s
}

/// Saturating-cast wrapper around [`render_sparkline`] for signed
/// (i64) counter series. Negative values clamp to 0; the kernel
/// stores SCX_EV_* as `s64` but every counter is non-negative in
/// practice, so the saturation only fires on a corrupt read.
pub fn render_sparkline_i64(values: &[i64]) -> String {
    let widened: Vec<u64> = values.iter().map(|&v| v.max(0) as u64).collect();
    render_sparkline(&widened)
}

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

/// Reason string written into [`FailureDumpReport::prog_runtime_stats_unavailable`]
/// when [`DumpContext::prog_capture`] was supplied but the per-program
/// walker found no struct_ops programs in `prog_idr` at freeze time.
/// Wire-format-stable: an operator parsing the sidecar JSON looks for
/// this exact string to distinguish from the prog-accessor-missing
/// case.
pub const REASON_NO_STRUCT_OPS_LOADED: &str = "no struct_ops programs loaded";

/// Reason string written into [`FailureDumpReport::prog_runtime_stats_unavailable`]
/// when [`DumpContext::prog_capture`] was `None`. Distinguishes from
/// [`REASON_NO_STRUCT_OPS_LOADED`] — the walker never ran in this case
/// because the accessor wasn't constructed (e.g. `prog_idr` symbol
/// missing).
pub const REASON_PROG_ACCESSOR_UNAVAILABLE: &str = "prog accessor unavailable";

/// Reason string written into [`FailureDumpReport::task_enrichments_unavailable`]
/// when [`DumpContext::task_enrichment_capture`] was supplied but
/// every walker entry produced no enrichment (idle guest with no
/// runnable scx tasks at the freeze instant).
pub const REASON_TASK_WALKER_ZERO_TASKS: &str = "task walker yielded zero tasks";

/// Reason string written into [`FailureDumpReport::task_enrichments_unavailable`]
/// when [`DumpContext::task_enrichment_capture`] was `None`.
/// Distinguishes from [`REASON_TASK_WALKER_ZERO_TASKS`] — the walker
/// never ran because the capture wasn't supplied.
pub const REASON_NO_TASK_WALKER: &str = "no task walker available";

/// Reason string written into [`FailureDumpReport::scx_walker_unavailable`]
/// when offsets resolved AND the walker found rq->scx + local DSQ
/// data BUT `*scx_root == 0` — no scheduler attached at the freeze
/// instant. The sched-rooted passes (bypass / global / user-hash)
/// have nothing to walk, but the rq->scx and per-CPU local DSQ
/// captures still produced data. Surfaces a distinct reason so the
/// operator knows the scheduler isn't loaded vs. the walker is
/// broken.
pub const REASON_SCX_ROOT_NULL: &str = "scx_root is NULL (no scheduler attached)";

/// Reason string written into [`FailureDumpReport::scx_walker_unavailable`]
/// when [`DumpContext::scx_walker_capture`] was supplied AND every
/// offset sub-group resolved BUT the walker reached no rq, no DSQ,
/// and no scx_sched state at all (every read failed). Distinct from
/// [`REASON_SCX_ROOT_NULL`]: that case has rq->scx + local DSQ data
/// but no sched_state; this case has nothing.
pub const REASON_SCX_WALKER_NO_STATE: &str = "scx walker reached no state";

/// Reason string written into [`FailureDumpReport::scx_walker_unavailable`]
/// when [`DumpContext::scx_walker_capture`] was `None`. Distinguishes
/// from [`REASON_SCX_WALKER_NO_STATE`] — the walker never ran at all
/// because no capture was supplied.
pub const REASON_NO_SCX_WALKER: &str = "no scx walker capture";

/// Cross-CPU sum of every per-CPU diagnostic counter slot in the
/// probe BPF program's `.bss` `ktstr_pcpu_counters` array.
///
/// The probe declares one fixed-shape per-CPU array
/// (`pcpu_counter ktstr_pcpu_counters[MAX_CPUS][KTSTR_PCPU_NR]` —
/// see `src/bpf/probe.bpf.c`); each tracepoint / kprobe handler
/// bumps a slot via `ktstr_pcpu_inc(KTSTR_PCPU_<NAME>)`. The host
/// reader sums across the CPU axis to recover the cumulative count
/// each handler reports. Field names mirror the slot names from
/// `enum ktstr_pcpu_idx` so an operator can walk back from the
/// failure-dump field to the probe source by exact name.
///
/// All counters are monotonic-since-probe-attach. Zero values
/// indicate either "the corresponding tracepoint never fired" (the
/// common case for `pi_*` and `lock_contend_*` on tests that don't
/// exercise PI / lock contention) or "the tracepoint never attached"
/// (e.g. `preempt_*` on a kernel without
/// `CONFIG_TRACE_PREEMPT_TOGGLE`); the counter alone cannot
/// distinguish those two cases — pair with the attach-state surface
/// in [`super::probe::process::ProbeDiagnostics`] when the
/// distinction matters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ProbeBssCounters {
    /// `KTSTR_PCPU_PROBE_COUNT` summed across CPUs — total kprobe
    /// fires past the `ktstr_enabled` gate.
    pub probe_count: u64,
    /// `KTSTR_PCPU_KPROBE_RETURNS` summed across CPUs — kprobe fires
    /// that committed an entry to `probe_data` (past `func_meta_map`
    /// lookup and scratch-slot allocation).
    pub kprobe_returns: u64,
    /// `KTSTR_PCPU_META_MISS` summed across CPUs — kprobe fires
    /// whose IP missed `func_meta_map`. `probe_count -
    /// kprobe_returns` is the total bail count; `meta_miss` is the
    /// subset whose bail came from the `func_meta_map` lookup.
    pub meta_miss: u64,
    /// `KTSTR_PCPU_RINGBUF_DROPS` summed across CPUs — failed
    /// `bpf_ringbuf_reserve` calls inside the trigger handler.
    pub ringbuf_drops: u64,
    /// `KTSTR_PCPU_TIMELINE_COUNT` summed across CPUs — successful
    /// timeline-event submissions across the three timeline
    /// tracepoints (sched_switch + sched_migrate_task + sched_wakeup).
    pub timeline_count: u64,
    /// `KTSTR_PCPU_TIMELINE_DROPS` summed across CPUs — timeline
    /// submissions that failed because the dedicated
    /// `timeline_events` ringbuf was full at submit time.
    pub timeline_drops: u64,
    /// `KTSTR_PCPU_PI_COUNT` summed across CPUs — PI boost / unboost
    /// records committed via `fexit/rt_mutex_setprio`.
    pub pi_count: u64,
    /// `KTSTR_PCPU_PI_ORPHAN_FEXITS` summed across CPUs — fexit
    /// fires whose entry-side snapshot was never recorded (attach
    /// race or `pi_scratch` overflow).
    pub pi_orphan_fexits: u64,
    /// `KTSTR_PCPU_PI_CLASS_CHANGE_COUNT` summed across CPUs —
    /// PI events that observed a `sched_class` flip from fentry
    /// to fexit (e.g. CFS → RT under a boost).
    pub pi_class_change_count: u64,
    /// `KTSTR_PCPU_PI_DROPS` summed across CPUs — TL_EVT_PI_BOOST
    /// submissions that failed because the timeline ringbuf was
    /// full at the PI fexit handler.
    pub pi_drops: u64,
    /// `KTSTR_PCPU_LOCK_CONTEND_COUNT` summed across CPUs —
    /// `tp_btf/contention_begin` fires that committed a
    /// TL_EVT_LOCK_CONTEND timeline record.
    pub lock_contend_count: u64,
    /// `KTSTR_PCPU_LOCK_CONTEND_DROPS` summed across CPUs —
    /// TL_EVT_LOCK_CONTEND submissions that failed because the
    /// timeline ringbuf was full.
    pub lock_contend_drops: u64,
    /// `KTSTR_PCPU_PREEMPT_DISABLE_COUNT` summed across CPUs —
    /// `tp_btf/preempt_disable` outermost-transition fires.
    pub preempt_disable_count: u64,
    /// `KTSTR_PCPU_PREEMPT_ENABLE_COUNT` summed across CPUs —
    /// `tp_btf/preempt_enable` outermost-transition fires.
    pub preempt_enable_count: u64,
    /// `KTSTR_PCPU_TRIGGER_COUNT` summed across CPUs — every
    /// `tp_btf/sched_ext_exit` fire (including non-error
    /// kinds like DONE / UNREG, not just error-class exits).
    pub trigger_count: u64,
}

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
    /// Per-program BPF runtime stats summed across CPUs at freeze
    /// time (cnt, nsecs, misses). One entry per discovered
    /// struct_ops BPF program. Empty when no struct_ops programs are
    /// loaded OR when the prog accessor was unavailable to
    /// [`dump_state`] — see [`Self::prog_runtime_stats_unavailable`]
    /// for the reason.
    ///
    /// Per-CPU offset resolution failure does NOT empty the vec —
    /// each program still contributes one entry, but with
    /// `cnt`/`nsecs`/`misses` summed only over CPUs whose per-CPU
    /// `bpf_prog_stats` slot translated successfully (out-of-range
    /// CPUs return None per [`super::bpf_map::read_percpu_array_value`]
    /// semantics).
    ///
    /// See [`super::bpf_prog::ProgRuntimeStats`] for field semantics
    /// and the kernel-source-grounded provenance of each counter.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prog_runtime_stats: Vec<super::bpf_prog::ProgRuntimeStats>,
    /// Diagnostic reason for `prog_runtime_stats` being empty.
    ///
    /// Distinguishes the three causes a consumer can't otherwise tell
    /// apart from an empty vec:
    /// - `None` (field absent on wire) → vec was populated normally
    ///   (or the dump path didn't run). Default.
    /// - `Some("no struct_ops programs loaded")` → walker ran, no
    ///   struct_ops programs were in `prog_idr` at freeze time.
    /// - `Some("prog accessor unavailable")` → caller passed
    ///   `prog_capture: None`. Typical causes: `prog_idr` symbol
    ///   missing, `BpfProgOffsets` BTF parse failed, or
    ///   `__per_cpu_offset` resolution didn't yield non-zero offsets
    ///   yet (still-booting guest).
    ///
    /// Set by [`dump_state`] only when prog_runtime_stats ends up
    /// empty AND a definite cause is identifiable; left None
    /// otherwise so the field stays absent in the JSON for
    /// already-populated dumps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prog_runtime_stats_unavailable: Option<String>,
    /// Per-CPU CPU-time / softirq / IRQ counters captured from
    /// `kernel_cpustat`, `kernel_stat`, and (under NO_HZ)
    /// `tick_sched`. One entry per CPU enumerated by the walker.
    /// Empty when the dump caller passed no [`CpuTimeCapture`] or
    /// when symbol/BTF resolution failed.
    ///
    /// See [`PerCpuTimeStats`] for field semantics. Surfaces the
    /// per-CPU interrupt and idle-time data the failure dump
    /// otherwise leaves implicit (the existing scx walker reads
    /// `rq->nr_iowait` but not the cumulative time accounting).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_cpu_time: Vec<PerCpuTimeStats>,
    /// Per-node NUMA event counters captured from
    /// `pglist_data->node_zones[]->vm_numa_event[]`. One row per
    /// NUMA node enumerated by the walker. Empty when the live
    /// walker has not landed yet (the BTF offsets and wire shape
    /// are wired; the reader is a follow-up).
    ///
    /// See [`PerNodeNumaStats`] for field semantics; see
    /// [`Self::per_node_numa_unavailable`] for the "why empty"
    /// diagnostic.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_node_numa: Vec<PerNodeNumaStats>,
    /// Diagnostic reason for `per_node_numa` being empty.
    /// `None` when the vec was populated normally (or the dump
    /// path didn't run); `Some(REASON_NO_NUMA_WALKER)` until the
    /// host-side walker lands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub per_node_numa_unavailable: Option<String>,
    /// Per-task failure-dump enrichments — identity (pid, tgid,
    /// comm), process tree (group_leader, real_parent, pgid, sid,
    /// nr_threads), scheduling (prio family, sched_class name,
    /// scx.weight, core_cookie), context-switch counters, watchdog
    /// disambiguation flag, and lock-slowpath stack matches.
    ///
    /// One entry per task the dump path's task walker reaches —
    /// today's task walkers are the rq->scx walker and the DSQ
    /// walker; both produce task KVAs that get enriched here.
    /// Empty when no task walker ran (typical until walker
    /// dispatch lands) or when the [`TaskEnrichmentCapture`] was
    /// absent.
    ///
    /// See [`super::task_enrichment::TaskEnrichment`] for field
    /// semantics; see [`Self::task_enrichments_unavailable`] for the
    /// "why empty" diagnostic.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub task_enrichments: Vec<super::task_enrichment::TaskEnrichment>,
    /// Diagnostic reason for `task_enrichments` being empty.
    ///
    /// - `None` → vec was populated normally (or the dump path
    ///   didn't run).
    /// - `Some("no task walker available")` → the
    ///   [`TaskEnrichmentCapture`] was missing from
    ///   [`DumpContext`]. Until DSQ + rq->scx walker dispatch
    ///   lands, this is the expected steady state for the dump
    ///   pipeline; the offsets + walker library is wired and
    ///   ready to populate as soon as a task-list producer hooks
    ///   in.
    /// - `Some("task walker yielded zero tasks")` → walker
    ///   produced no task KVAs (frozen guest with no runnable /
    ///   queued scx tasks at the dump instant — possible on a
    ///   completely-idle stall trigger).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_enrichments_unavailable: Option<String>,
    /// Per-monitor-tick SCX_EV_* event counter timeline. Each entry
    /// is the cross-CPU sum of the 13 SCX_EV_* counters at one
    /// monitor sample. Empty when the dump caller passed no
    /// [`EventCounterCapture`] or no sample reported event counters
    /// (event-stat offsets unresolved, scx_root unset). Renderers
    /// build sparklines / per-counter delta plots from this vec.
    ///
    /// See [`EventCounterSample`] for field semantics; the kernel-
    /// source provenance lives on
    /// [`super::ScxEventCounters`] field doc.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub event_counter_timeline: Vec<EventCounterSample>,
    /// Per-CPU `rq->scx` snapshots — scalar fields the kernel's
    /// own `scx_dump_state` reads plus the runnable_list per-task
    /// KVAs that fed into the per-task enrichment capture.
    /// One entry per CPU walked. Empty when the
    /// [`ScxWalkerCapture`] was absent or every CPU's translate
    /// failed.
    ///
    /// See [`super::scx_walker::RqScxState`] for field semantics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rq_scx_states: Vec<super::scx_walker::RqScxState>,
    /// Per-DSQ snapshots — local, bypass, global, and user DSQs
    /// reachable from `*scx_root`. Each entry carries `nr` (depth),
    /// `seq` (BPF-iter counter), and the queued task KVAs.
    /// Surfaces data the kernel's own `scx_dump_state` does not
    /// emit (per-DSQ depth enumeration), so this vec adds value
    /// even on a kernel that prints its own dump.
    ///
    /// Empty when the [`ScxWalkerCapture`] was absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dsq_states: Vec<super::scx_walker::DsqState>,
    /// Top-level `scx_sched` state captured from `*scx_root`:
    /// aborting flag, bypass_depth, exit_kind. `None` when no
    /// scheduler is attached or `*scx_root` was unreadable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scx_sched_state: Option<super::scx_walker::ScxSchedState>,
    /// Diagnostic reason for `rq_scx_states` / `dsq_states` /
    /// `scx_sched_state` being absent. Mirrors the
    /// `prog_runtime_stats_unavailable` / `task_enrichments_unavailable`
    /// pattern.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scx_walker_unavailable: Option<String>,
    /// Per-vCPU hardware perf counter snapshot captured at the
    /// instant the failure dump fired. One entry per vCPU; index
    /// matches vCPU id (0 = BSP, 1..N = APs). `None` per-entry when
    /// the freeze-time `read(2)` failed for that vCPU. Empty vec
    /// when [`DumpContext::perf_capture`] was None (perf
    /// unavailable on this host) or the read errored wholesale.
    ///
    /// `exclude_host=1` means each counter ticks only during guest
    /// execution; the values here record the cumulative count from
    /// the start of the run. Diff against any
    /// [`super::CpuSnapshot::vcpu_perf`] in the monitor timeline to
    /// recover the count over a freeze-aligned window. See
    /// [`super::perf_counters::VcpuPerfSample`] for field semantics
    /// and the multiplexing math.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vcpu_perf_at_freeze: Vec<Option<super::perf_counters::VcpuPerfSample>>,
    /// Microseconds from dump_state entry to the phase that exceeded
    /// the soft deadline supplied via [`DumpContext::deadline`]. `None`
    /// when no deadline was supplied, when every phase finished within
    /// the deadline, or when the deadline check happened before the
    /// dump started any heavy phase. A `Some(us)` value means the dump
    /// truncated remaining work (skipped further maps / tasks /
    /// walkers) at that elapsed offset to keep the freeze window
    /// bounded — the freeze coordinator's parked vCPUs cannot
    /// service guest IRQs or MMIO traps while the dump is running,
    /// so unbounded dump latency stretches every guest's KVM_RUN
    /// pause and risks the freeze rendezvous timeout firing on the
    /// next iteration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dump_truncated_at_us: Option<u64>,
    /// Probe BPF program's per-CPU diagnostic counter snapshot
    /// (see [`ProbeBssCounters`]). Populated by the host-side
    /// reader in [`decode_probe_counters_snapshot`] which sums
    /// each `KTSTR_PCPU_*` slot across CPUs. `None` when the
    /// probe `.bss` map isn't enumerated (probe not loaded), the
    /// program BTF can't be parsed, or the array's offset doesn't
    /// resolve.
    ///
    /// A populated `trigger_count > 0` is the structural signal
    /// that the BPF tp_btf/sched_ext_exit handler fired during
    /// the run — distinct from the boolean `trigger_fired` flag
    /// in [`super::probe::process::ProbeDiagnostics`] (which
    /// also records host-side observations like a watchdog
    /// teardown). The cross-product is the failure-dump E2E
    /// test's structural assertion: a stall scenario must show
    /// both flag=true AND `trigger_count > 0`, otherwise the
    /// probe attached without firing or fired without the host
    /// observing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probe_counters: Option<ProbeBssCounters>,
    /// `true` when this report was produced by
    /// [`Self::placeholder`] — i.e. the capture pipeline could
    /// not produce real data (typical cause: freeze rendezvous
    /// timed out). Periodic-sample temporal assertions skip
    /// placeholder reports rather than treating their empty
    /// vectors as "no progress" signals; the `*_unavailable`
    /// fields carry the reason string for human consumers, but
    /// the boolean flag is the machine-checkable discriminant a
    /// pattern can branch on without re-deriving placeholder
    /// status from the absence of every field.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_placeholder: bool,
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            per_node_numa: Vec::new(),
            per_node_numa_unavailable: None,
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
            dump_truncated_at_us: None,
            probe_counters: None,
            is_placeholder: false,
        }
    }
}

impl FailureDumpReport {
    /// Build a placeholder report for a capture that could not
    /// produce real data. Every `*_unavailable` field is set to
    /// `Some(reason)` so downstream consumers (`stats compare`,
    /// failure-rendering tooling) can distinguish "capture
    /// happened, no data" from "capture path failed for reason X".
    /// All vector / option fields stay at their `Default` empty
    /// state so the report is structurally a real
    /// `FailureDumpReport`, not a sentinel that breaks consumer
    /// type contracts.
    ///
    /// Used by the freeze coordinator's user-watchpoint dispatch,
    /// periodic-capture drain, and final-drain teardown — every
    /// site that needs to publish a "capture attempted, did not
    /// land" entry on the snapshot bridge.
    pub fn placeholder(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        Self {
            prog_runtime_stats_unavailable: Some(reason.clone()),
            per_node_numa_unavailable: Some(reason.clone()),
            task_enrichments_unavailable: Some(reason.clone()),
            scx_walker_unavailable: Some(reason),
            is_placeholder: true,
            ..Self::default()
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
    /// To recover the kernel's full `watchdog_timeout`, double
    /// [`Self::early_threshold_jiffies`] — the scanner trigger
    /// fires at half the watchdog, so the threshold field carries
    /// `watchdog_timeout / 2`. Diff `early_max_age_jiffies` against
    /// `2 * early_threshold_jiffies` to see how close the system
    /// was to the SCX_EXIT_ERROR_STALL emission line at the
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
    /// Structured reason the early snapshot is absent. `None` when
    /// the early snapshot was captured (the [`Self::early`] field is
    /// `Some`). When the early field is `None`, this carries a short
    /// machine-friendly string identifying which of the known
    /// failure modes occurred:
    ///
    /// - `"scan prerequisites unavailable: <prereq>"` — the
    ///   per-CPU `runnable_at` scan never resolved its dependencies
    ///   (most often `<prereq>` names the missing kernel symbol /
    ///   BTF entry).
    /// - `"max_age never crossed threshold (peak={peak}j,
    ///   threshold={threshold}j)"` — the scan ran but the maximum
    ///   observed runnable-age stayed below the half-way mark for
    ///   the whole VM lifetime. Indicates a non-stall err-class exit
    ///   (e.g. `scx_bpf_error()`).
    /// - `"scx_tick stall — no per-task runnable_at data"` — the
    ///   stall path that drove the late capture has no per-task
    ///   `runnable_at` to scan (the kernel's "watchdog failed to
    ///   check in" path raises `SCX_EXIT_ERROR_STALL` from the
    ///   scx_tick kernel side without any task on
    ///   `rq->scx.runnable_list`).
    ///
    /// Display rendering at [`super::display`] surfaces this string
    /// directly; the previous "stall fired before half-way threshold,
    /// or runnable_at scan setup failed" generic text is replaced
    /// with the structured reason whenever this field is `Some`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_skipped_reason: Option<String>,
}

fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// Either-or wrapper that owns a parsed [`FailureDumpReport`] or
/// [`DualFailureDumpReport`]. Lets a consumer hold and render a
/// failure-dump file without prematurely committing to one schema —
/// the discriminant lives in the JSON's `schema` field, not in the
/// type the consumer holds.
///
/// Centralises the schema-tag dispatch logic that previously lived
/// inline at every read site (the auto-repro tail renderer, the
/// failure-dump-e2e test, any future consumer that wants to inspect
/// either shape). Use [`Self::from_json`] to parse an arbitrary
/// failure-dump JSON blob; the Display impl forwards to the
/// underlying report's existing Display so the rendered output is
/// indistinguishable from holding the unwrapped report directly.
///
/// `non_exhaustive` so a future third schema (e.g. a `triple`
/// wrapper that captures snapshots at three points instead of two)
/// can be added without breaking external pattern matches.
#[non_exhaustive]
pub enum FailureDumpReportAny {
    /// Single-snapshot report, schema=`"single"`. Emitted by the
    /// primary VM's freeze coordinator when an error-class SCX exit
    /// fires.
    Single(Box<FailureDumpReport>),
    /// Dual-snapshot wrapper, schema=`"dual"`. Emitted by the
    /// auto-repro VM when the dual-snapshot path is enabled. Carries
    /// optional `early` + required `late` snapshots plus jiffies
    /// metadata for the early-trigger condition.
    ///
    /// Boxed to keep [`FailureDumpReportAny`]'s on-stack size bounded
    /// — `DualFailureDumpReport` carries the early+late snapshots
    /// inline and is roughly 2x the size of [`FailureDumpReport`].
    Dual(Box<DualFailureDumpReport>),
}

impl FailureDumpReportAny {
    /// Parse a failure-dump JSON blob, choosing the variant by the
    /// `schema` field. Returns `None` on any of:
    ///
    /// - the blob does not parse as JSON
    /// - the `schema` field carries an unknown value (no silent
    ///   fallback to single — that would mis-render a richer wrapper
    ///   as a lossy single shape)
    /// - the typed deserialisation under the chosen schema fails
    ///
    /// An absent `schema` field deserialises as
    /// [`Self::Single`] via the
    /// `default_schema_single` serde default — this preserves
    /// backwards compatibility with dumps written before the
    /// schema-tag landed.
    pub fn from_json(json: &str) -> Option<Self> {
        let value: serde_json::Value = serde_json::from_str(json).ok()?;
        let schema = value.get("schema").and_then(|v| v.as_str()).unwrap_or("");
        match schema {
            SCHEMA_DUAL => serde_json::from_str(json)
                .ok()
                .map(|d| Self::Dual(Box::new(d))),
            SCHEMA_SINGLE | "" => serde_json::from_str(json)
                .ok()
                .map(|r| Self::Single(Box::new(r))),
            _ => None,
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
    /// Per-key per-CPU slots for `PERCPU_HASH` / `LRU_PERCPU_HASH`
    /// maps. Same shape as `percpu_entries` but the outer key is
    /// arbitrary bytes (rendered via BTF when a key type id is
    /// available, hex otherwise) instead of the implicit u32 key
    /// of a per-CPU array.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub percpu_hash_entries: Vec<FailureDumpPercpuHashEntry>,
    /// Page snapshot for `BPF_MAP_TYPE_ARENA` maps. `None` for all
    /// other map types.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arena: Option<ArenaSnapshot>,
    /// Position counters and capacity for `BPF_MAP_TYPE_RINGBUF` /
    /// `BPF_MAP_TYPE_USER_RINGBUF` maps. Surfaces stuck-consumer
    /// diagnostics — pending bytes far below the watermark plus
    /// non-zero `pending_pos` indicates a producer holding a
    /// reservation; pending bytes near capacity indicates a stalled
    /// consumer. `None` for non-ringbuf maps or when the BTF offsets
    /// for `bpf_ringbuf_map` / `bpf_ringbuf` weren't resolvable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ringbuf: Option<FailureDumpRingbuf>,
    /// Per-bucket trace summary for `BPF_MAP_TYPE_STACK_TRACE` maps.
    /// `None` for non-STACK_TRACE maps or when the BTF offsets for
    /// `bpf_stack_map` / `stack_map_bucket` weren't resolvable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stack_trace: Option<FailureDumpStackTrace>,
    /// Populated-slot summary for FD-array families (`PROG_ARRAY`,
    /// `PERF_EVENT_ARRAY`, `CGROUP_ARRAY`, `ARRAY_OF_MAPS`,
    /// `HASH_OF_MAPS`, `DEVMAP*`, `SOCKMAP*`, `CPUMAP`, `XSKMAP`,
    /// `REUSEPORT_SOCKARRAY`). `None` for non-FD-array maps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fd_array: Option<FailureDumpFdArray>,
    /// Reason this map's contents are missing or partial. Empty on
    /// successful render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Ringbuf occupancy snapshot read from `struct bpf_ringbuf` at the
/// freeze instant.
///
/// Capacity, consumer/producer positions, and the in-flight reservation
/// frontier (`pending_pos`) are all that's readable without walking the
/// records. Pending bytes (= `producer_pos - consumer_pos`, computed
/// with unsigned wraparound) is the operator-visible indicator: low
/// values = consumer keeping up; values approaching capacity = consumer
/// stuck or kernel producer overrunning. A non-zero gap between
/// `producer_pos` and `pending_pos` means a producer is mid-reserve
/// and the consumer can't advance past `pending_pos`.
///
/// Read via [`crate::monitor::btf_offsets::BpfRingbufOffsets`]; rendered
/// in [`render_ringbuf_state`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpRingbuf {
    /// Ring data area capacity in bytes (= `mask + 1`). Always a
    /// power of two; matches the map's declared `max_entries`.
    pub capacity: u64,
    /// Consumer position. Byte index of the next record userspace
    /// will read. Monotonically advances; the kernel never writes
    /// here.
    pub consumer_pos: u64,
    /// Producer position. Byte index past the last reserved record.
    /// Monotonically advances; updated by the kernel on each
    /// `bpf_ringbuf_reserve`.
    pub producer_pos: u64,
    /// Pending position. Byte index of the oldest in-flight (still
    /// being filled) reservation. Records below `pending_pos` are
    /// committed and visible to the consumer; records between
    /// `pending_pos` and `producer_pos` are reserved but not yet
    /// committed.
    pub pending_pos: u64,
    /// Pending bytes (= `producer_pos.wrapping_sub(consumer_pos)`).
    /// 0 = consumer caught up; capacity = ring full / consumer
    /// stalled. Computed with unsigned wraparound to match the
    /// kernel's dispatch-path arithmetic.
    pub pending_bytes: u64,
}

/// Per-bucket summary of populated stack traces in a STACK_TRACE map.
///
/// Each `entry` is one populated bucket whose pointer was non-null at
/// the freeze instant. `nr` is the number of trace samples (PCs) in
/// the bucket; `pcs` carries the actual u64 PC values when readable
/// (build-id stacks render the raw bytes hex since the per-entry
/// shape is `struct bpf_stack_build_id`, not a u64). The dump caps
/// per-bucket entries at [`MAX_STACK_TRACE_PCS`] to bound memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpStackTrace {
    /// `bpf_stack_map.n_buckets` — the rounded-up power-of-two slot
    /// count. Iteration upper bound; differs from `max_entries` which
    /// the kernel rounds.
    pub n_buckets: u32,
    /// One entry per non-null bucket pointer. Sorted by bucket id.
    pub entries: Vec<FailureDumpStackTraceEntry>,
    /// True when any populated bucket was truncated at
    /// [`MAX_STACK_TRACE_PCS`] PCs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// One populated stack trace from a STACK_TRACE map.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpStackTraceEntry {
    /// Bucket id (= stack ID returned by `bpf_get_stackid`).
    pub bucket_id: u32,
    /// Number of trace samples (kernel `stack_map_bucket.nr`).
    pub nr: u32,
    /// PC values (u64) when the map is in non-build-id mode. Empty
    /// when `BPF_F_STACK_BUILD_ID` is set on the map (each entry
    /// is then a `bpf_stack_build_id` record — its raw bytes land
    /// in `data_hex`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pcs: Vec<u64>,
    /// Hex-encoded raw bucket data bytes. Always populated alongside
    /// `pcs` so the operator can decode build-id stacks or correlate
    /// trace samples with the wire format.
    pub data_hex: String,
}

/// Per-FD-array snapshot of populated indices.
///
/// FD-array families store `void *` slots in `bpf_array.ptrs`; each
/// slot is either NULL (empty) or a kernel pointer (struct bpf_prog *,
/// struct file *, etc.). The dump path reads up to
/// [`MAX_FD_ARRAY_SLOTS`] slots, counts non-zero, and lists the
/// populated indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpFdArray {
    /// Number of populated (non-zero) slots seen.
    pub populated: u32,
    /// Total slots scanned. Capped at [`MAX_FD_ARRAY_SLOTS`].
    pub scanned: u32,
    /// Indices of populated slots. Truncated to
    /// [`MAX_FD_ARRAY_INDICES`] entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<u32>,
    /// True when iteration capped at [`MAX_FD_ARRAY_SLOTS`] and
    /// `scanned < max_entries`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
    /// True when `populated > indices.len()` because
    /// [`MAX_FD_ARRAY_INDICES`] capped the index list.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub indices_truncated: bool,
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
    /// Typed render of the per-entry sdt_alloc payload, when the value
    /// carries a `struct sdt_data __arena *` field that points into a
    /// captured arena page and a payload type id was discovered for
    /// the matching allocator. `None` when the entry carries no arena
    /// pointer to chase, no allocator metadata was found, the payload
    /// type was ambiguous, or the arena read failed.
    ///
    /// `value` already renders the surface struct (e.g.
    /// `scx_task_map_val { tid, tptr, data: 0x100000... -> sdt_data {
    /// tid: { idx, genn } } }`), but `sdt_data.payload[]` is a flex
    /// array — BTF reports its size as 0, so the per-task struct that
    /// actually lives in the payload bytes never decodes through the
    /// surface render. This field carries that decoded payload
    /// alongside the surface struct so the operator sees both views
    /// at once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<RenderedValue>,
}

/// One key from a per-CPU array, with one rendered value per CPU
/// (None for CPUs whose per-CPU page was unmapped or out-of-range).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpPercpuEntry {
    pub key: u32,
    pub per_cpu: Vec<Option<RenderedValue>>,
}

/// One key from a `PERCPU_HASH` / `LRU_PERCPU_HASH` map, with one
/// rendered value per CPU. Mirrors [`FailureDumpEntry`] for the key
/// side (rendered + hex) and [`FailureDumpPercpuEntry`] for the
/// per-CPU value vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FailureDumpPercpuHashEntry {
    /// Rendered key. `None` when no BTF type is available for the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<RenderedValue>,
    /// Hex-encoded raw key bytes.
    pub key_hex: String,
    /// One slot per CPU. `None` when the CPU's per-CPU slot was
    /// unmapped or out-of-range; `Some` rendered (BTF when value type
    /// id is non-zero) or hex bytes otherwise.
    pub per_cpu: Vec<Option<RenderedValue>>,
}

/// Sanity cap on a single BTF blob read.
///
/// BPF program BTF is normally <100 KB; vmlinux BTF caps around
/// ~10 MB. A bogus `data_size` (corrupted `struct btf`) shouldn't
/// pull megabytes of unrelated guest memory into the renderer or the
/// freeze coordinator. Shared between [`load_program_btf_kva`] and
/// `vmm::load_probe_bss_offset`; defining it here keeps the bound
/// in one place so a future tightening doesn't drift between sites.
pub(crate) const MAX_BTF_BLOB: usize = 32 * 1024 * 1024;

/// Hard cap on the per-task enrichment loop inside [`dump_state`].
///
/// A hostile or pathologically broken guest can produce a runnable_list
/// chain whose length is bounded only by the number of
/// `task_struct`s in the kernel — tens of thousands on a busy box.
/// Each enrichment call walks task/signal/pid/upid offsets, the
/// sched_class registry, and the lock-slowpath stack matcher, so an
/// uncapped loop turns the freeze window from milliseconds into
/// minutes. 4096 is well above any healthy SCX runnable_list depth
/// (the kernel's own watchdog fires long before that many tasks
/// queue up) and still bounds the worst-case freeze cost. When the
/// cap fires, [`dump_state`] truncates without enriching the tail
/// and stamps [`FailureDumpReport::dump_truncated_at_us`] so the
/// operator knows to attribute missing tasks to truncation rather
/// than walker failure.
pub const MAX_ENRICHED_TASKS: usize = 4096;

/// Bare-named ktstr framework maps to skip during enumeration.
///
/// These are declared in `src/bpf/probe.bpf.c` without a libbpf
/// `<obj>.<section>` prefix (`SEC(".maps")` declarations like
/// `func_meta_map`, `probe_data`, `probe_scratch`, `ktstr_events`);
/// the kernel registers them under the bare names listed here.
/// They're framework-internal — the user looking at a failure dump
/// for their scheduler doesn't care about ktstr's own kprobe
/// scratch — so the dump path drops them.
///
/// The framework's ringbuf is named `ktstr_events` (not `events`)
/// so a user scheduler that legitimately names its own ringbuf
/// `events` is not silently dropped from the dump.
///
/// Future ktstr probe additions need to be added here AND the
/// matching `<obj_name>.` prefix needs to be in the
/// [`render_map`-internal] starts_with list (see [`dump_state`]).
const KTSTR_INTERNAL_MAPS: &[&str] = &[
    "func_meta_map",
    "probe_data",
    "probe_scratch",
    "ktstr_events",
];

/// All inputs the failure-dump renderer needs, bundled so future
/// capture sites (DSQ walker, rq->scx walker, NUMA stats, ...) can
/// land as new optional fields without churning every call site.
///
/// `accessor` is currently the concrete guest-memory backend. The
/// trait dispatch claim in [`BpfMapAccessor`]'s module-level doc
/// is aspirational: `dump_state` reaches through the accessor for
/// map enumeration AND for the sdt_alloc post-pass walk, which
/// needs the underlying [`super::guest::GuestKernel`] handle —
/// only the guest-memory backend exposes that. When the live-host
/// backend lands, sdt_alloc walking will move into a
/// backend-specific path and `accessor` here can become
/// `&'a dyn BpfMapAccessor`.
///
/// `arena_offsets` and `prog_capture` are both optional borrows
/// (uniform shape): `None` for either disables that
/// capture leg without affecting the rest. A scheduler running on
/// an older kernel without arena support lands here with
/// `arena_offsets: None` and the failure dump renders maps + regs
/// without arena pages; a setup where the BpfProgAccessor couldn't
/// resolve `prog_idr` lands with `prog_capture: None` and
/// `prog_runtime_stats` stays empty.
pub struct DumpContext<'a> {
    /// BPF map accessor. Concrete guest-memory backend today; see
    /// the type-level doc for why this is not `&dyn BpfMapAccessor`.
    pub accessor: &'a GuestMemMapAccessor<'a>,
    /// Host-resolved vmlinux BTF. The renderer uses it as the base
    /// for split-BTF parsing on programs that ship their own type
    /// info; it's also the fallback when a map's program BTF can't
    /// be loaded.
    pub btf: &'a Btf,
    /// Guest's `nr_cpu_ids`. Forwarded into per-CPU map rendering
    /// so PERCPU_ARRAY readers know how many slots to enumerate.
    /// Pass `1` for non-percpu-only dumps if the caller doesn't
    /// have the value handy.
    pub num_cpus: u32,
    /// BTF-resolved arena field offsets. Enables
    /// `BPF_MAP_TYPE_ARENA` page snapshotting via the accessor
    /// trait's `read_arena_pages`. `None` skips arena rendering
    /// (older kernel without arena support, or BTF lacking
    /// `struct bpf_arena`).
    pub arena_offsets: Option<&'a BpfArenaOffsets>,
    /// Per-program runtime stats capture. `None` skips
    /// prog-runtime capture; the dump still renders every map the
    /// accessor enumerates.
    pub prog_capture: Option<&'a ProgRuntimeCapture<'a>>,
    /// Per-CPU CPU-time / softirq / IRQ capture. `None` skips the
    /// per-CPU time walk; the rest of the dump still renders. Same
    /// "borrowed-only, optional" shape as
    /// [`Self::prog_capture`] / [`Self::arena_offsets`] so a
    /// future capture site lands as another optional field without
    /// churning the call sites already plumbed through here.
    pub cpu_time_capture: Option<&'a CpuTimeCapture<'a>>,
    /// Per-task enrichment capture. `None` skips the per-task walk
    /// and `task_enrichments` stays empty; the rest of the dump
    /// still renders.
    ///
    /// Today's freeze coordinator passes `None` because the DSQ
    /// and rq->scx task walkers have not yet landed dispatch. The
    /// `TaskEnrichmentOffsets` + `SchedClassRegistry` + the
    /// `walk_task_enrichment` library are wired and ready —
    /// the producer side just needs to populate
    /// [`TaskEnrichmentCapture::tasks`] from the rq->scx walker.
    pub task_enrichment_capture: Option<&'a TaskEnrichmentCapture<'a>>,
    /// SCX_EV_* event counter timeline capture. `None` skips
    /// timeline rendering and `event_counter_timeline` stays
    /// empty; the rest of the dump still renders. Same
    /// "borrowed-only, optional" shape as
    /// [`Self::cpu_time_capture`].
    pub event_counter_capture: Option<&'a EventCounterCapture<'a>>,
    /// SCX rq->scx + DSQ walker capture. `None` skips the walk;
    /// `rq_scx_states` / `dsq_states` / `scx_sched_state` stay
    /// empty/None and `scx_walker_unavailable` records why.
    pub scx_walker_capture: Option<&'a ScxWalkerCapture<'a>>,
    /// Host-side per-vCPU hardware perf counters (cycles,
    /// instructions, cache-misses, branch-misses) opened with
    /// `exclude_host=1`, so each counter only ticks during guest
    /// execution. `None` skips the freeze-time read; the
    /// [`FailureDumpReport::vcpu_perf_at_freeze`] vec stays empty.
    /// See [`super::perf_counters`] for the kernel-source-grounded
    /// rationale and capture semantics.
    ///
    /// The same capture is shared (via `Arc` in the freeze
    /// coordinator) with the per-tick monitor sampler; per-tick
    /// samples land on each [`super::CpuSnapshot::vcpu_perf`]. The
    /// freeze-time read here records the absolute counter values at
    /// the instant the failure dump fired, which lets a consumer
    /// diff against any earlier sample to compute IPC over a
    /// freeze-aligned window.
    pub perf_capture: Option<&'a super::perf_counters::PerfCountersCapture>,
    /// Soft deadline for the dump's heavy phases (per-map render
    /// loop, walk_rq_scx, walk_local_dsqs, walk_dsqs sched-rooted,
    /// walk_task_enrichment, sdt_alloc post-pass). When supplied,
    /// each phase boundary checks `Instant::now() > deadline`; the
    /// first crossing truncates remaining work and stamps
    /// [`FailureDumpReport::dump_truncated_at_us`]. `None` disables
    /// the bailout — the dump runs every phase to completion.
    ///
    /// Set by the freeze coordinator to `capture_start +
    /// watchdog_timeout/2` so a slow dump can't keep vCPUs parked
    /// past the kernel's own SCX_EXIT_ERROR_STALL emission line. The
    /// deadline is a soft bound: each phase that has already started
    /// runs to completion before checking, so the actual elapsed
    /// time at truncation can exceed the deadline by one phase's
    /// worth of work.
    pub deadline: Option<std::time::Instant>,
    /// BPF cast-analysis output for the scheduler's program object,
    /// produced once at builder time by parsing the scheduler
    /// binary's `.bpf.objs` ELF blob (no libbpf, no kernel
    /// interaction). Threaded into every per-map [`RenderMapCtx`]
    /// so the renderer's
    /// [`super::btf_render::MemReader::cast_lookup`] can promote
    /// `u64` fields the analyzer flagged into typed-pointer
    /// renders. `None` skips cast-driven promotion entirely (every
    /// `u64` renders as a plain unsigned counter, the
    /// pre-integration default); same effect as passing an empty
    /// map but cheaper to thread.
    pub cast_map: Option<&'a super::cast_analysis::CastMap>,
}

/// Reconstruct an `ScxSchedState` from the probe BPF program's
/// `.bss` snapshot (`ktstr_exit_*` vars).
///
/// Used as a fallback by [`dump_state`] when
/// [`super::scx_walker::read_scx_sched_state`] returned `None`
/// because `*scx_root == 0` at freeze time. The probe's tp_btf
/// handler captured the same scalars BEFORE the kernel teardown
/// nulled `scx_root`, so this path produces a coherent view of
/// what the scheduler looked like AT THE INSTANT IT ERRORED OUT —
/// which is exactly the state an operator wants to debug.
///
/// Returns `None` when:
///   - the probe `.bss` map isn't loaded yet (boot-race window),
///   - the probe's program BTF can't be parsed,
///   - the snapshot's `ktstr_exit_kind_snap` is still 0 (latch
///     never fired this run, so the snapshot is empty defaults),
///   - or any individual var lookup / read fails wholesale.
///
/// Variable names match the probe BPF declarations one-for-one
/// (`ktstr_exit_aborting`, `ktstr_exit_bypass_depth`,
/// `ktstr_exit_kind_snap`, `ktstr_exit_sched_kva`,
/// `ktstr_exit_watchdog_timeout`); each is resolved by name via the
/// program-BTF Datasec walk so a future addition / reorder of `.bss`
/// vars does not silently misalign offsets.
/// Decode the probe BPF program's per-CPU counter array
/// (`ktstr_pcpu_counters`) and sum each slot across CPUs.
///
/// The probe declares `pcpu_counter ktstr_pcpu_counters[MAX_CPUS]
/// [KTSTR_PCPU_NR]` in `.bss`; each `pcpu_counter` is a single
/// `long` field forced to 128-byte alignment, so each per-CPU slot
/// occupies its own cacheline. The host walks each
/// `(cpu, slot)` 8-byte slice and sums into a [`ProbeBssCounters`]
/// — see the BPF source for the
/// `ktstr_pcpu_inc(KTSTR_PCPU_<NAME>)` fire sites.
///
/// Returns `None` when:
///   - the probe `.bss` map isn't loaded yet (boot-race window),
///   - the probe's program BTF can't be parsed,
///   - the BTF doesn't carry a `ktstr_pcpu_counters` var (probe
///     build that pre-dates the per-CPU conversion), or
///   - the array's bytes can't be read wholesale.
///
/// All values use `u64` for wire compatibility; the underlying
/// kernel `long` is signed but every fire site only ever
/// increments, so a positive cumulative count is the only outcome
/// in practice. Negative reads (would indicate guest-memory
/// corruption) saturate to 0 via `as u64`.
fn decode_probe_counters_snapshot(
    accessor: &GuestMemMapAccessor<'_>,
    base_btf: &Btf,
) -> Option<ProbeBssCounters> {
    use super::bpf_map::BpfMapAccessor;

    // Slot indices must match `enum ktstr_pcpu_idx` in
    // src/bpf/probe.bpf.c. A reorder in the BPF source breaks
    // every reader; the explicit constants here keep the slot
    // mapping localized and reviewable.
    const PCPU_PROBE_COUNT: usize = 0;
    const PCPU_KPROBE_RETURNS: usize = 1;
    const PCPU_META_MISS: usize = 2;
    const PCPU_RINGBUF_DROPS: usize = 3;
    const PCPU_TIMELINE_COUNT: usize = 4;
    const PCPU_TIMELINE_DROPS: usize = 5;
    const PCPU_PI_COUNT: usize = 6;
    const PCPU_PI_ORPHAN_FEXITS: usize = 7;
    const PCPU_PI_CLASS_CHANGE_COUNT: usize = 8;
    const PCPU_PI_DROPS: usize = 9;
    const PCPU_LOCK_CONTEND_COUNT: usize = 10;
    const PCPU_LOCK_CONTEND_DROPS: usize = 11;
    const PCPU_PREEMPT_DISABLE_COUNT: usize = 12;
    const PCPU_PREEMPT_ENABLE_COUNT: usize = 13;
    const PCPU_TRIGGER_COUNT: usize = 14;
    const PCPU_NR: usize = 15;
    /// Per-CPU slot stride in bytes — `pcpu_counter` is forced to
    /// 128-byte alignment in the BPF source so each slot occupies
    /// one cacheline. Mirroring the alignment here keeps the
    /// host-side walk in lockstep with the BPF storage layout;
    /// any future change to the alignment must update both.
    const PCPU_SLOT_STRIDE: usize = 128;
    /// Per-CPU dimension. Matches `MAX_CPUS` in `src/bpf/probe.bpf.c`
    /// (CPU_MASK + 1 = 256). Walking every CPU slot is cheap (256
    /// CPUs × 15 slots × 8 bytes = 30 KB of reads); slots beyond
    /// the actual `nr_cpus` are zero-init `.bss` and contribute
    /// nothing to the sum.
    const MAX_CPUS: usize = 256;

    // Locate the probe's `.bss` map. Same suffix the freeze
    // coordinator's lazy-discovery path uses (matched by suffix
    // to avoid colliding with a scheduler-under-test's own
    // `.bss`).
    let bss_map = accessor.find_map("probe_bp.bss")?;
    if bss_map.btf_kva == 0 {
        // Probe not yet loaded — accessor enumerated a stub.
        return None;
    }

    // Load the probe's program BTF as split BTF on top of the
    // host vmlinux BTF (matches the freeze coordinator's
    // load_probe_bss_offset pattern). Failure is silent — the
    // dump path stays best-effort and falls through to None so
    // the caller leaves `probe_counters` as None rather than
    // emitting a misleading partial.
    let prog_btf = load_program_btf_kva(accessor, bss_map.btf_kva, base_btf)?;

    // Resolve the array's byte offset within the `.bss` Datasec.
    // A missing var (e.g. probe build that pre-dates the per-CPU
    // conversion) means the snapshot wasn't emitted — bail.
    let array_off = super::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_pcpu_counters",
    )? as usize;

    // Read the entire array as one slab — 256 * 17 * 128 = 544 KB.
    // A single slab read is cheaper than 256 * 17 individual reads
    // through the page-walking accessor; the read primitive
    // tolerates over-large requests (truncates at the map's
    // value_size) so a future MAX_CPUS / PCPU_NR shrink doesn't
    // need a coordinated host update.
    let total_bytes = MAX_CPUS * PCPU_NR * PCPU_SLOT_STRIDE;
    let array_bytes = accessor.read_value(&bss_map, array_off, total_bytes)?;
    if array_bytes.len() < total_bytes {
        // Short read — the map's value_size bounds were tighter
        // than the array's compile-time shape. A future probe
        // build that shrinks MAX_CPUS or PCPU_NR is the expected
        // case; bail rather than misalign the slot indexing.
        return None;
    }

    // Sum every CPU's slot. Each slot's `long value` lives at
    // offset 0 within the cacheline-aligned `pcpu_counter`
    // struct, so the per-(cpu, slot) byte offset is
    // `(cpu * PCPU_NR + slot) * PCPU_SLOT_STRIDE`.
    let sum_slot = |slot: usize| -> u64 {
        let mut total: u64 = 0;
        for cpu in 0..MAX_CPUS {
            let off = (cpu * PCPU_NR + slot) * PCPU_SLOT_STRIDE;
            // BPF runs in little-endian byte order on every
            // host arch ktstr targets (x86_64, aarch64). A future
            // big-endian host would need an arch gate — flagged
            // in the probe BPF source's byte-order section.
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&array_bytes[off..off + 8]);
            // The kernel's `long` is signed but counters only
            // increment; cast through `i64` then to `u64` to
            // saturate any negative value (corruption signal) to 0.
            let v = i64::from_le_bytes(buf);
            if v > 0 {
                total = total.saturating_add(v as u64);
            }
        }
        total
    };

    Some(ProbeBssCounters {
        probe_count: sum_slot(PCPU_PROBE_COUNT),
        kprobe_returns: sum_slot(PCPU_KPROBE_RETURNS),
        meta_miss: sum_slot(PCPU_META_MISS),
        ringbuf_drops: sum_slot(PCPU_RINGBUF_DROPS),
        timeline_count: sum_slot(PCPU_TIMELINE_COUNT),
        timeline_drops: sum_slot(PCPU_TIMELINE_DROPS),
        pi_count: sum_slot(PCPU_PI_COUNT),
        pi_orphan_fexits: sum_slot(PCPU_PI_ORPHAN_FEXITS),
        pi_class_change_count: sum_slot(PCPU_PI_CLASS_CHANGE_COUNT),
        pi_drops: sum_slot(PCPU_PI_DROPS),
        lock_contend_count: sum_slot(PCPU_LOCK_CONTEND_COUNT),
        lock_contend_drops: sum_slot(PCPU_LOCK_CONTEND_DROPS),
        preempt_disable_count: sum_slot(PCPU_PREEMPT_DISABLE_COUNT),
        preempt_enable_count: sum_slot(PCPU_PREEMPT_ENABLE_COUNT),
        trigger_count: sum_slot(PCPU_TRIGGER_COUNT),
    })
}

fn decode_probe_sched_state_snapshot(
    accessor: &GuestMemMapAccessor<'_>,
    base_btf: &Btf,
) -> Option<super::scx_walker::ScxSchedState> {
    use super::bpf_map::BpfMapAccessor;

    // Locate the probe's `.bss` map. Same suffix the freeze
    // coordinator's lazy-discovery path uses (matched by suffix to
    // avoid colliding with a scheduler-under-test's own `.bss`).
    let bss_map = accessor.find_map("probe_bp.bss")?;
    if bss_map.btf_kva == 0 {
        // Probe not yet loaded — accessor enumerated a stub. The
        // var offsets live in the program BTF the loader hasn't
        // attached yet.
        return None;
    }

    // Load the probe's program BTF as a split BTF on top of the
    // host vmlinux BTF (matches the freeze coordinator's
    // load_probe_bss_offset pattern). Failure is silent — the dump
    // path stays best-effort and falls through to None so the
    // caller leaves `scx_sched_state` as None rather than emitting
    // a misleading partial.
    let prog_btf = load_program_btf_kva(accessor, bss_map.btf_kva, base_btf)?;

    // Resolve each `ktstr_exit_*` var's byte offset within the
    // `.bss` Datasec. A missing var (e.g. probe build that pre-
    // dates the snapshot vars) means the snapshot wasn't emitted —
    // bail rather than render zero defaults that would alias as
    // "scheduler healthy and exited cleanly".
    let kind_off = super::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_exit_kind_snap",
    )?;
    let aborting_off = super::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_exit_aborting",
    )?;
    let bypass_depth_off = super::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_exit_bypass_depth",
    )?;
    let sched_kva_off = super::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_exit_sched_kva",
    )?;
    let watchdog_timeout_off = super::btf_offsets::resolve_var_offset_in_section(
        &prog_btf,
        ".bss",
        "ktstr_exit_watchdog_timeout",
    )?;

    // Read each var's bytes via the accessor. `.bss` maps have a
    // single key (zero) and the value bytes ARE the section bytes,
    // so `read_value(map, off, size)` is the read primitive. A
    // failed read on any field bails the whole snapshot — partial
    // values would mislead the consumer.
    let kind_bytes = accessor.read_value(&bss_map, kind_off as usize, 4)?;
    let kind = u32::from_le_bytes(kind_bytes.as_slice().try_into().ok()?);

    // The snapshot is sticky: `ktstr_exit_kind_snap` stays at 0
    // until the BPF tp_btf handler latches an error-class exit. A
    // 0 here means the latch never fired — the snapshot vars are
    // all at their initial 0/false defaults and the dump should
    // honour `*scx_root == 0` as "no scheduler state to surface"
    // rather than render a fake healthy-exit ScxSchedState.
    if kind == 0 {
        return None;
    }

    let aborting_bytes = accessor.read_value(&bss_map, aborting_off as usize, 1)?;
    let aborting = aborting_bytes.first().copied()? != 0;

    let bypass_depth_bytes = accessor.read_value(&bss_map, bypass_depth_off as usize, 4)?;
    let bypass_depth = i32::from_le_bytes(bypass_depth_bytes.as_slice().try_into().ok()?);

    let sched_kva_bytes = accessor.read_value(&bss_map, sched_kva_off as usize, 8)?;
    let sched_kva = u64::from_le_bytes(sched_kva_bytes.as_slice().try_into().ok()?);

    let watchdog_timeout_bytes = accessor.read_value(&bss_map, watchdog_timeout_off as usize, 8)?;
    let watchdog_timeout = u64::from_le_bytes(watchdog_timeout_bytes.as_slice().try_into().ok()?);

    Some(super::scx_walker::ScxSchedState {
        aborting,
        bypass_depth,
        exit_kind: kind,
        watchdog_timeout: Some(watchdog_timeout),
        source: Some(super::scx_walker::SCX_SCHED_STATE_SOURCE_BSS.to_string()),
        // `sched_kva == 0` would mean the BPF probe handler ran
        // BEFORE `*scx_root` was populated (impossibly early — the
        // tp_btf hook is on `sched_ext_exit`, which only fires after
        // a sched_ext scheduler attached and ran). Surface it as
        // None so the consumer can distinguish "snapshot data exists
        // but no slab address" from "snapshot has the address" via
        // a single Option rather than a magic-zero check.
        sched_kva: if sched_kva == 0 {
            None
        } else {
            Some(sched_kva)
        },
    })
}

/// Snapshot every BPF map visible to the host accessor.
///
/// The dump is best-effort: a map that fails to render lands in the
/// report with `error: Some(...)` rather than aborting the whole walk,
/// so a single corrupt map can't blind the operator to the rest of
/// the scheduler's state.
pub fn dump_state(ctx: DumpContext<'_>) -> FailureDumpReport {
    let DumpContext {
        accessor,
        btf,
        num_cpus,
        arena_offsets,
        prog_capture,
        cpu_time_capture,
        task_enrichment_capture,
        event_counter_capture,
        scx_walker_capture,
        perf_capture,
        deadline,
        cast_map,
    } = ctx;
    // Wall-clock origin for per-phase elapsed_us tracing and the
    // soft-deadline bailout. Each heavy phase compares
    // `Instant::now()` against `deadline` AFTER it finishes, so a
    // truncation captures the phase's data before short-circuiting
    // the remaining ones (consistent with the doc on
    // [`DumpContext::deadline`]).
    let dump_start = std::time::Instant::now();
    // Tracks the elapsed_us of the first phase to observe a deadline
    // crossing. Stamped onto [`FailureDumpReport::dump_truncated_at_us`]
    // at the end so the operator can attribute missing maps / tasks /
    // walker results to truncation rather than walker failure.
    let mut truncated_at_us: Option<u64> = None;
    // Helper closure: returns `true` once the deadline (if any) has
    // been crossed. Sets `truncated_at_us` on the FIRST crossing so
    // the report records WHERE truncation began, not the last phase
    // to short-circuit. Idempotent on repeated calls — once stamped,
    // every later phase sees the same elapsed_us.
    let deadline_exceeded = |truncated_at: &mut Option<u64>| -> bool {
        if let Some(deadline) = deadline {
            let now = std::time::Instant::now();
            if now > deadline {
                if truncated_at.is_none() {
                    let elapsed_us = dump_start.elapsed().as_micros() as u64;
                    *truncated_at = Some(elapsed_us);
                    tracing::warn!(
                        elapsed_us,
                        "dump_state: deadline exceeded, truncating remaining phases"
                    );
                }
                return true;
            }
        }
        false
    };
    let maps = accessor.maps();
    let (prog_runtime_stats, prog_runtime_stats_unavailable) = match prog_capture {
        Some(cap) => {
            let stats = cap.accessor.struct_ops_runtime_stats(cap.per_cpu_offsets);
            let reason = if stats.is_empty() {
                Some(REASON_NO_STRUCT_OPS_LOADED.to_string())
            } else {
                None
            };
            (stats, reason)
        }
        None => (
            Vec::new(),
            Some(REASON_PROG_ACCESSOR_UNAVAILABLE.to_string()),
        ),
    };
    let per_cpu_time = match cpu_time_capture {
        Some(cap) => collect_per_cpu_time(cap),
        None => Vec::new(),
    };
    let task_enrichment_t0 = std::time::Instant::now();
    let (task_enrichments, task_enrichments_unavailable) = match task_enrichment_capture {
        Some(cap) => {
            // Cap iteration AND Vec capacity at MAX_ENRICHED_TASKS so
            // a hostile guest with a corrupt or absurdly long
            // runnable_list can't drag the freeze window into the
            // tens-of-seconds range.
            let total = cap.tasks.len();
            let cap_n = total.min(MAX_ENRICHED_TASKS);
            let mut enrichments = Vec::with_capacity(cap_n);
            for entry in cap.tasks.iter().take(cap_n) {
                if let Some(e) = super::task_enrichment::walk_task_enrichment(
                    cap.kernel,
                    entry.task_kva,
                    cap.offsets,
                    cap.sched_classes,
                    cap.lock_slowpaths,
                    entry.is_runnable_in_scx,
                    entry.running_pc,
                ) {
                    enrichments.push(e);
                }
            }
            if total > cap_n {
                tracing::warn!(
                    cap = MAX_ENRICHED_TASKS,
                    total,
                    "dump_state task_enrichment: capped at MAX_ENRICHED_TASKS, dropping tail"
                );
            }
            let reason = if enrichments.is_empty() {
                tracing::debug!(
                    tasks_count = total,
                    "dump_state task_enrichment: walker yielded zero entries — \
                     scx_tasks list and rq->scx.runnable_list both empty, or every \
                     walk_task_enrichment call returned None (translate failures)",
                );
                Some(REASON_TASK_WALKER_ZERO_TASKS.to_string())
            } else {
                None
            };
            (enrichments, reason)
        }
        None => {
            tracing::debug!(
                "dump_state task_enrichment: capture is None — \
                 freeze coordinator passed no TaskEnrichmentCapture \
                 (scx_owned, scx_walker_offsets, or task_enrichment_offsets unresolved)",
            );
            (Vec::new(), Some(REASON_NO_TASK_WALKER.to_string()))
        }
    };
    tracing::debug!(
        elapsed_us = task_enrichment_t0.elapsed().as_micros() as u64,
        enriched = task_enrichments.len(),
        "dump_state phase: walk_task_enrichment"
    );
    deadline_exceeded(&mut truncated_at_us);
    let event_counter_timeline = match event_counter_capture {
        Some(cap) => cap
            .samples
            .iter()
            .filter_map(EventCounterSample::from_monitor_sample)
            .collect(),
        None => Vec::new(),
    };
    let (rq_scx_states, dsq_states, scx_sched_state, scx_walker_unavailable) =
        match scx_walker_capture {
            Some(cap) => {
                // Sub-group offsets resolved per kernel struct;
                // surface the absent groups in the diagnostic so a
                // partial walk announces which passes were skipped.
                let missing = cap.offsets.missing_groups();

                // 1. Read scalar scx_sched state and recover the
                //    sched_pa for the sched-rooted DSQ walker passes.
                //    `sched_state` is None when the BTF lacked the
                //    `sched` sub-group OR when *scx_root == 0
                //    (no scheduler attached) — both surface as a
                //    None scx_sched_state in the report. The
                //    distinction is encoded in `scx_walker_unavailable`
                //    via REASON_SCX_ROOT_NULL.
                let (sched_pa_opt, sched_state) = match super::scx_walker::read_scx_sched_state(
                    cap.kernel,
                    cap.scx_root_kva,
                    cap.offsets,
                ) {
                    Some((sched_kva, state)) => {
                        // Translate sched_kva → PA (slab/vmalloc; use
                        // translate_any_kva via the GuestKernel handle).
                        let mem = cap.kernel.mem();
                        let walk = cap.kernel.walk_context();
                        let pa = super::idr::translate_any_kva(
                            mem,
                            walk.cr3_pa,
                            walk.page_offset,
                            sched_kva,
                            walk.l5,
                            walk.tcr_el1,
                        );
                        (pa, Some(state))
                    }
                    None => {
                        // Live read failed — `*scx_root == 0` because
                        // the scheduler has already torn down by
                        // freeze time. Fall back to the BPF .bss
                        // snapshot the probe's tp_btf handler latched
                        // at err-exit time. The snapshot is the
                        // strict subset of scheduler state the host
                        // renderer needs; the sched_pa stays None
                        // because the slab page that backed the live
                        // `scx_sched` was freed during teardown and
                        // the sched-rooted DSQ passes (per-node
                        // global, user dsq_hash) cannot reach it any
                        // longer. The caller's `unavail` selector
                        // below now sees `Some(state)` and skips
                        // REASON_SCX_ROOT_NULL — the consumer reads
                        // `state.source = "bss_snapshot"` to
                        // distinguish snapshot from live.
                        let snap = decode_probe_sched_state_snapshot(accessor, btf);
                        if snap.is_some() {
                            tracing::debug!(
                                scx_root_kva = format_args!("{:#x}", cap.scx_root_kva),
                                "dump_state scx walker: live read returned None; \
                                 BPF .bss snapshot fallback populated scx_sched_state \
                                 (scheduler torn down before freeze, snapshot \
                                 captured at err-exit instant)",
                            );
                        }
                        (None, snap)
                    }
                };

                // 2. Per-CPU rq->scx walk. Per-CPU runs only when the
                //    rq + scx_rq + task sub-groups are present;
                //    walk_rq_scx returns None to skip otherwise.
                let walk_rq_scx_t0 = std::time::Instant::now();
                let mut rq_states = Vec::with_capacity(cap.rq_kvas.len());
                if !deadline_exceeded(&mut truncated_at_us) {
                    for (cpu, (&rq_kva, &rq_pa)) in
                        cap.rq_kvas.iter().zip(cap.rq_pas.iter()).enumerate()
                    {
                        if let Some((state, _entries)) = super::scx_walker::walk_rq_scx(
                            cap.kernel,
                            cpu as u32,
                            rq_kva,
                            rq_pa,
                            cap.offsets,
                        ) {
                            rq_states.push(state);
                        }
                    }
                }
                tracing::debug!(
                    elapsed_us = walk_rq_scx_t0.elapsed().as_micros() as u64,
                    cpus = cap.rq_kvas.len(),
                    rq_states = rq_states.len(),
                    "dump_state phase: walk_rq_scx"
                );

                // 3. Per-CPU local DSQ walk runs unconditionally —
                //    `rq->scx.local_dsq` is initialized at boot
                //    (init_dsq from kernel/sched/ext.c:7772 for every
                //    possible CPU) and survives scheduler teardown,
                //    so it produces data even when *scx_root is NULL.
                //    This is the data source that survives
                //    scx_bypass's runnable_list drain
                //    (kernel/sched/ext.c:5304-5404) during teardown.
                let walk_local_dsqs_t0 = std::time::Instant::now();
                let mut dsqs: Vec<super::scx_walker::DsqState> = Vec::new();
                if !deadline_exceeded(&mut truncated_at_us)
                    && let Some((local_states, _entries)) = super::scx_walker::walk_local_dsqs(
                        cap.kernel,
                        cap.rq_kvas,
                        cap.rq_pas,
                        cap.per_cpu_offsets,
                        cap.offsets,
                    )
                {
                    dsqs.extend(local_states);
                }
                tracing::debug!(
                    elapsed_us = walk_local_dsqs_t0.elapsed().as_micros() as u64,
                    local_dsqs = dsqs.len(),
                    "dump_state phase: walk_local_dsqs"
                );

                // 4. Sched-rooted DSQ passes (per-CPU bypass, per-node
                //    global, user dsq_hash) require the sched_pa we
                //    resolved in step 1. Without it, no scheduler is
                //    attached and these DSQs don't exist at all.
                let walk_dsqs_t0 = std::time::Instant::now();
                if !deadline_exceeded(&mut truncated_at_us)
                    && let Some(sched_pa) = sched_pa_opt
                {
                    let (sched_states, _entries) = super::scx_walker::walk_dsqs(
                        cap.kernel,
                        sched_pa,
                        cap.per_cpu_offsets,
                        cap.nr_nodes,
                        cap.offsets,
                    );
                    dsqs.extend(sched_states);
                }
                tracing::debug!(
                    elapsed_us = walk_dsqs_t0.elapsed().as_micros() as u64,
                    total_dsqs = dsqs.len(),
                    "dump_state phase: walk_dsqs"
                );

                // Diagnostic priority:
                //   1. Partial-degradation (sub-group(s) missing) —
                //      announces exactly which passes were skipped.
                //   2. *scx_root is NULL — sched/bypass/global/user
                //      passes blinded but rq->scx + local DSQ still
                //      work; surface this distinct reason so the
                //      operator knows the scheduler isn't attached.
                //   3. Walker reached no state at all — typical when
                //      every read fails.
                //   4. None — every pass had data to surface.
                let unavail = if !missing.is_empty() {
                    tracing::debug!(
                        missing_groups = ?missing,
                        rq_states_count = rq_states.len(),
                        dsq_count = dsqs.len(),
                        sched_state_some = sched_state.is_some(),
                        "dump_state scx walker: partial degradation — missing BTF sub-groups",
                    );
                    Some(format!(
                        "scx walker partial: missing offset groups [{}]",
                        missing.join(", ")
                    ))
                } else if sched_state.is_none() {
                    tracing::debug!(
                        scx_root_kva = format_args!("{:#x}", cap.scx_root_kva),
                        rq_states_count = rq_states.len(),
                        dsq_count = dsqs.len(),
                        "dump_state scx walker: scx_root is NULL — no scheduler attached; \
                         rq->scx and local DSQ captures populated, sched/bypass/global/user passes blinded",
                    );
                    Some(REASON_SCX_ROOT_NULL.to_string())
                } else if rq_states.is_empty() && dsqs.is_empty() {
                    tracing::debug!(
                        scx_root_kva = format_args!("{:#x}", cap.scx_root_kva),
                        "dump_state scx walker: every walker read failed — no rq->scx, no DSQ, but sched_state present",
                    );
                    Some(REASON_SCX_WALKER_NO_STATE.to_string())
                } else {
                    None
                };
                (rq_states, dsqs, sched_state, unavail)
            }
            None => {
                tracing::debug!(
                    "dump_state scx walker: capture is None — \
                     freeze coordinator passed no ScxWalkerCapture (offsets/symbols/per_cpu_offsets unresolved)",
                );
                (
                    Vec::new(),
                    Vec::new(),
                    None,
                    Some(REASON_NO_SCX_WALKER.to_string()),
                )
            }
        };
    // Freeze-time per-vCPU perf-counter snapshot. With `exclude_host=1`
    // each counter ticks only during guest execution; the freeze
    // coordinator has parked every vCPU before reaching this site, so
    // the read returns the cumulative count at the last guest exit
    // for each vCPU. A single per-vCPU read failure is recorded as
    // `None` for that entry; a failure on one vCPU does not blank the
    // others. When `perf_capture` is None the vec stays empty (the
    // host lacked perf, or `perf_event_open` failed at run start).
    let vcpu_perf_at_freeze: Vec<Option<super::perf_counters::VcpuPerfSample>> = match perf_capture
    {
        Some(cap) => cap.per_vcpu.iter().map(|p| p.read().ok()).collect(),
        None => Vec::new(),
    };

    // Snapshot the probe's per-CPU diagnostic counters before the
    // per-map render loop walks `.bss` itself — the read goes
    // through the same `read_value` path the renderer uses, but
    // captures the array as a structured `ProbeBssCounters` rather
    // than the BTF Datasec render. Best-effort: a None result
    // (probe not loaded, BTF missing the var) leaves the report's
    // `probe_counters` empty and the existing `.bss` map render
    // still surfaces the raw bytes.
    let probe_counters = decode_probe_counters_snapshot(accessor, btf);

    let mut report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::with_capacity(maps.len()),
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats,
        prog_runtime_stats_unavailable,
        per_cpu_time,
        // Per-node NUMA wire fields: empty Vec + the well-defined
        // diagnostic string until the host-side walker lands.
        per_node_numa: Vec::new(),
        per_node_numa_unavailable: Some(REASON_NO_NUMA_WALKER.to_string()),
        task_enrichments,
        task_enrichments_unavailable,
        event_counter_timeline,
        rq_scx_states,
        dsq_states,
        scx_sched_state,
        scx_walker_unavailable,
        vcpu_perf_at_freeze,
        dump_truncated_at_us: None,
        probe_counters,
        is_placeholder: false,
    };

    // Per-map program-BTF cache, keyed by `btf_kva`. Each unique
    // `struct btf *` lives in the kernel BTF IDR — multiple maps from
    // the same BPF program point at the same KVA, so caching dedupes
    // the heavy `Btf::from_bytes`/`from_split_bytes` parse across them
    // (a scheduler with N maps backed by one BPF object pays one
    // parse, not N). Lookups go through this cache before falling
    // back to the caller-supplied vmlinux `btf`.
    //
    // Populated by an explicit pre-pass below so the sdt_alloc walk
    // can read it before the per-map render loop runs (the renderer
    // needs the resulting allocator metadata via `RenderMapCtx`).
    let mut program_btfs: std::collections::HashMap<u64, Btf> = std::collections::HashMap::new();

    // Pre-pass: locate the first non-internal `BPF_MAP_TYPE_ARENA`
    // map (skipping the same ktstr-internal name set the main loop
    // skips) and snapshot it once before any map renders. This lets
    // the per-map `MemReader` chase `__arena` pointers no matter
    // which slot the arena map occupies in the iteration order —
    // the previous design ran `snapshot_arena` lazily inside
    // `render_map`'s arena arm, so non-arena maps that rendered
    // earlier saw `arena_snapshot: None` and silently failed every
    // arena pointer chase. `lib/arena_map.h` declares one `__weak`
    // arena per BPF object so a single shared snapshot covers every
    // `__arena` pointer the scheduler emits; additional arena maps
    // (multi-object schedulers, theoretical) still get their own
    // snapshot inside `render_map`'s arena arm — they just don't
    // contribute to the cross-map pointer-chase context.
    let shared_arena_snapshot: Option<(BpfMapInfo, ArenaSnapshot)> =
        arena_offsets.and_then(|off| {
            for info in &maps {
                let name = info.name();
                if name.starts_with("probe_bp.")
                    || name.starts_with("fentry_p.")
                    || name == "probe_bp"
                    || name == "fentry_p"
                    || KTSTR_INTERNAL_MAPS.contains(&name.as_ref())
                {
                    continue;
                }
                if info.map_type == BPF_MAP_TYPE_ARENA {
                    let snap = snapshot_arena(accessor.kernel(), info, off);
                    return Some((info.clone(), snap));
                }
            }
            None
        });
    let shared_arena_ref: Option<(&ArenaSnapshot, u64)> = shared_arena_snapshot
        .as_ref()
        .map(|(info, snap)| (snap, info.map_kva));

    // Cache `kern_vm_start` from the pre-pass snapshot for the
    // sdt_alloc walk. Pulling directly from `shared_arena_snapshot`
    // (rather than scraping each rendered map's `arena` field in the
    // main loop) keeps the walk gating decoupled from per-map render
    // order — the data the walker needs is finalized before the
    // loop runs.
    let arena_kern_vm_start: u64 = shared_arena_snapshot
        .as_ref()
        .map(|(_, snap)| snap.kern_vm_start)
        .unwrap_or(0);

    // Pre-pass: load every non-internal map's program BTF and locate
    // the scheduler's `.bss` raw bytes. Both inputs feed the
    // sdt_alloc walk below — moving them out of the main render loop
    // means the allocator metadata that decoration in the
    // TASK_STORAGE arm needs (`elem_size`, `payload_btf_type_id`) is
    // available BEFORE any map renders, instead of getting derived
    // post-loop only to be unusable for per-entry payload chase.
    let mut sched_bss_bytes: Option<(Vec<u8>, u64)> = None; // (bytes, btf_kva)
    for info in &maps {
        let name = info.name();
        if name.starts_with("probe_bp.")
            || name.starts_with("fentry_p.")
            || name == "probe_bp"
            || name == "fentry_p"
            || KTSTR_INTERNAL_MAPS.contains(&name.as_ref())
        {
            continue;
        }
        if info.btf_kva != 0
            && !program_btfs.contains_key(&info.btf_kva)
            && let Some(loaded) = accessor.load_program_btf(info, btf)
        {
            program_btfs.insert(info.btf_kva, loaded);
        }
        if sched_bss_bytes.is_none()
            && info.map_type == BPF_MAP_TYPE_ARRAY
            && info.btf_kva != 0
            && name.ends_with(".bss")
            && let Some(bytes) = accessor.read_value(info, 0, info.value_size as usize)
        {
            sched_bss_bytes = Some((bytes, info.btf_kva));
        }
    }

    // Pre-pass: walk sdt_alloc trees if all prerequisites lined up.
    // Runs BEFORE the main render loop so the allocator metadata it
    // discovers (`elem_size`, `payload_btf_type_id`,
    // `data_header_size`) is available to per-map decoration —
    // specifically, the TASK_STORAGE arm uses it to expand each
    // entry's `struct sdt_data __arena *` pointer into a typed
    // payload render via [`render_map`]'s
    // [`crate::monitor::dump::render_map::SdtAllocMeta`].
    //
    // The walk is best-effort and silent: any missing prerequisite
    // (no scheduler .bss, no arena window, no program BTF, no
    // `scx_allocator` type) leaves `sdt_allocations` empty rather
    // than failing the dump. `sdt_alloc_metas` stays empty in the
    // same cases, so each per-entry payload field also degrades to
    // `None` (the surface struct still renders).
    //
    // Build the dump-pass arena page index here too — once outside
    // the per-map loop so each per-map `mem_reader` borrows the
    // existing table instead of rebuilding it. The sdt_alloc walk
    // below uses the same index for its own MemReader.
    let arena_page_index = crate::monitor::dump::render_map::build_arena_page_index(
        shared_arena_snapshot.as_ref().map(|(_, snap)| snap),
    );
    let sdt_alloc_t0 = std::time::Instant::now();
    // Every typed allocator the program declares; the per-map
    // selector in [`render_map`] picks the matching entry by name
    // (e.g. `scx_task_allocator` matches `scx_task_map`). A
    // single-allocator scheduler hits the unique-candidate path —
    // every map gets that allocator. A multi-allocator scheduler
    // (per-task + per-cgroup) lets each local-storage map render
    // its own payload type instead of forcing the renderer to give
    // up.
    let mut sdt_alloc_metas: Vec<crate::monitor::dump::render_map::SdtAllocMeta> = Vec::new();
    // `payload_start_low32 → payload_btf_type_id` lookup
    // populated as each allocator walk completes.
    // [`MemReader::resolve_arena_type`] consults this to recover
    // the real payload struct id for chased arena pointers whose
    // declared pointee is a `BTF_KIND_FWD` (typical for `struct
    // sdt_data __arena *` fields where the body lives in a
    // separate library BTF). Built incrementally inside the walk
    // loop so the per-allocator snapshot moves into
    // `report.sdt_allocations` after each iteration without a
    // clone.
    //
    // [`crate::monitor::sdt_alloc::TreeWalker::emit_leaf`]
    // populates each [`SdtAllocEntry::user_addr`] as
    // `data_ptr & 0xFFFF_FFFF` — the slot-START address windowed
    // to the low 32 bits. The bridge does NOT key on slot-start
    // because the production trigger is post-header pointers:
    // `scx_task_data(p)` (`lib/sdt_task.bpf.c`) returns
    // `data->payload`, and lavd caches that exact value in
    // `cached_taskc_raw`. Every chase the renderer issues for a
    // typed field whose value comes from those helpers carries
    // the payload start (slot start + header_size), not the slot
    // start. Adding `data_header_size` here (8 bytes —
    // `sizeof(union sdt_id)`) shifts the key from slot-start to
    // payload-start so the renderer's
    // [`MemReader::resolve_arena_type`] override — which masks the
    // chased value with `0xFFFF_FFFF` — finds a match.
    //
    // `checked_add` guards against a `user_addr + header_size`
    // sum overflowing `u32::MAX`: the additive end of the windowed
    // payload start. The mask in [`emit_leaf`] keeps `user_addr`
    // within `u32::MAX`, but a slot at the very top of the window
    // could push the post-header start past it; saturate by
    // skipping those slots rather than wrapping into a
    // low-numbered key that would alias a different slot. Duplicates
    // (two slots reporting the same payload start, indicating a
    // stale snapshot from a freed allocation racing with the
    // freeze) keep the FIRST entry; this matches the
    // [`build_arena_page_index`] policy on duplicate user_addr
    // pages and emits a `tracing::debug!` line so an operator
    // diagnosing a wrong-render can spot the collision.
    let mut arena_type_index = crate::monitor::dump::render_map::ArenaTypeIndex::new();
    if !deadline_exceeded(&mut truncated_at_us)
        && let Some((bss_bytes, btf_kva)) = sched_bss_bytes
        && arena_kern_vm_start != 0
        && let Some(prog_btf) = program_btfs.get(&btf_kva)
        && let Ok(sdt_offsets) = SdtAllocOffsets::from_btf(prog_btf)
    {
        // One MemReader for every leaf payload render, so an
        // arena pointer embedded in a per-task / per-cgroup
        // sdt_alloc payload chases into typed contents instead
        // of opaque hex.
        //
        // The arena type index is intentionally `None` on this
        // pre-pass reader: the walk produces the entries the
        // index is built from, so the index does not yet exist
        // when the leaf payload renders run. A nested `__arena
        // *` pointer inside a payload that targets a separate
        // allocator slot whose payload type is forward-declared
        // in the program BTF degrades to the existing chase
        // behaviour during the pre-pass; the index is wired
        // into the per-map renders below where the typical
        // bridge call site lives (TASK_STORAGE / HASH maps
        // holding `struct sdt_data __arena *` entry pointers).
        let sdt_mem = accessor.mem_reader(
            shared_arena_snapshot.as_ref().map(|(_, snap)| snap),
            &arena_page_index,
            num_cpus,
            // Threaded in from [`DumpContext::cast_map`]: same
            // cast-analysis output the per-map renderer below
            // consumes. Letting the sdt_alloc pre-pass see it
            // means typed-allocator payload chases (per-task /
            // per-cgroup contents inside arena) get the same
            // `u64` → typed-pointer promotion as the rest of
            // the dump, instead of degrading to plain counters
            // for fields the analyzer recovered.
            cast_map,
            None,
        );
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
                choice.reason.clone(),
                var_name.clone(),
                &sdt_mem,
            );
            // Accumulate every allocator with a typed payload AND
            // append its live slots to the bridge index. The
            // per-map selector (`select_sdt_alloc_meta`) picks the
            // right one by matching `var_name` (the .bss symbol —
            // e.g. `scx_task_allocator`) against each rendered map's
            // name (e.g. `scx_task_map`). Schedulers that declare
            // multiple typed allocators no longer lose payload
            // expansion — each map renders against the matching
            // allocator's payload type. Only allocators with a
            // resolved payload type id contribute to the bridge
            // index — without a typed payload there is no useful
            // BTF id to surface to the renderer, and the index
            // would just point every chase at 0 (which the bridge
            // gate filters as "no payload type").
            if choice.btf_type_id != 0 {
                sdt_alloc_metas.push(crate::monitor::dump::render_map::SdtAllocMeta {
                    allocator_name: var_name.clone(),
                    elem_size,
                    header_size: sdt_offsets.data_header_size,
                    payload_btf_type_id: choice.btf_type_id,
                    kern_vm_start: arena_kern_vm_start,
                });
                if let Ok(header_low32) = u32::try_from(sdt_offsets.data_header_size) {
                    for entry in &snap.entries {
                        // user_addr comes back as a `u64` whose top
                        // 32 bits are zero (per
                        // `walk_sdt_allocator`'s mask in
                        // `emit_leaf`). The narrowing cast
                        // preserves the meaningful low half; the
                        // assertion is encoded in the source
                        // comment for the masking site rather than
                        // re-checked here.
                        let user_addr_low32 = entry.user_addr as u32;
                        let Some(payload_start) = user_addr_low32.checked_add(header_low32) else {
                            continue;
                        };
                        match arena_type_index.entry(payload_start) {
                            std::collections::btree_map::Entry::Vacant(v) => {
                                v.insert(choice.btf_type_id);
                            }
                            std::collections::btree_map::Entry::Occupied(o) => {
                                tracing::debug!(
                                    payload_start = format_args!("{:#x}", payload_start),
                                    first_btf_type_id = *o.get(),
                                    duplicate_btf_type_id = choice.btf_type_id,
                                    allocator = %var_name,
                                    "sdt_alloc bridge has duplicate payload_start; keeping first entry",
                                );
                            }
                        }
                    }
                }
            }
            // Surface only allocators with a non-empty result OR a
            // diagnostic elem_size; an all-zero snapshot from a
            // never-initialized allocator is just noise.
            if !snap.entries.is_empty() || snap.elem_size != 0 {
                report.sdt_allocations.push(snap);
            }
        }
    }
    let arena_type_index_ref = if arena_type_index.is_empty() {
        None
    } else {
        Some(&arena_type_index)
    };
    tracing::debug!(
        elapsed_us = sdt_alloc_t0.elapsed().as_micros() as u64,
        allocations = report.sdt_allocations.len(),
        index_entries = arena_type_index.len(),
        "dump_state phase: sdt_alloc"
    );

    let render_map_t0 = std::time::Instant::now();
    let mut maps_rendered: usize = 0;
    let mut maps_truncated: usize = 0;
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
        {
            let info_name = info.name();
            if info_name.starts_with("probe_bp.")
                || info_name.starts_with("fentry_p.")
                || info_name == "probe_bp"
                || info_name == "fentry_p"
                || KTSTR_INTERNAL_MAPS.contains(&info_name.as_ref())
            {
                continue;
            }
        }

        // Deadline check before each map render — bigger maps
        // (large hashes, arenas) can each take a meaningful slice
        // of the freeze window, so we re-check between renders to
        // bound the worst case rather than letting one
        // straggler push us past the watchdog.
        if deadline_exceeded(&mut truncated_at_us) {
            maps_truncated += 1;
            continue;
        }

        // Per-map BTF resolution.
        //
        // The map's `btf_value_type_id` / `btf_key_type_id` index
        // the *map's own* BTF, NOT the kernel vmlinux BTF — when
        // `btf_kva != 0` the type IDs are program-local and using
        // vmlinux BTF with them would resolve to unrelated kernel
        // types (or out-of-range nonsense). So:
        //
        //   - `BPF_MAP_TYPE_STRUCT_OPS`              → use vmlinux
        //     BTF. The wrapper struct `bpf_struct_ops_<name>` is
        //     declared in the kernel's vmlinux BTF and the
        //     wrapper type id stored on the map (in
        //     `btf_vmlinux_value_type_id`) indexes vmlinux. Using
        //     the program BTF here would fail to resolve the
        //     wrapper.
        //   - `btf_kva != 0` AND program BTF loaded by pre-pass → use it.
        //   - `btf_kva != 0` AND program BTF load failed in pre-pass
        //     → render hex-only (None map_btf), no fallback.
        //   - `btf_kva == 0` (kernel-builtin map)      → use the
        //     caller-supplied vmlinux BTF; the type IDs (if any)
        //     genuinely index vmlinux BTF in this case.
        let map_btf: Option<&Btf> = if info.map_type == super::bpf_map::BPF_MAP_TYPE_STRUCT_OPS {
            Some(btf)
        } else if info.btf_kva != 0 {
            program_btfs.get(&info.btf_kva)
        } else {
            Some(btf)
        };

        let rendered = render_map(
            &RenderMapCtx {
                accessor,
                btf: map_btf,
                num_cpus,
                arena_offsets,
                shared_arena: shared_arena_ref,
                arena_page_index: &arena_page_index,
                sdt_alloc_metas: &sdt_alloc_metas,
                // Threaded in from
                // [`DumpContext::cast_map`]: the BPF
                // cast-analysis output for the scheduler's
                // program object. `Some(&map)` lets the
                // renderer promote `u64` fields the analyzer
                // flagged into typed-pointer renders via
                // [`super::btf_render::MemReader::cast_lookup`];
                // `None` keeps every `u64` rendered as a plain
                // unsigned counter (the trait default).
                cast_map,
                // Built from the sdt_alloc pre-pass above:
                // `payload_start → payload_btf_type_id` for
                // every live allocator slot. Lets the renderer
                // recover a `BTF_KIND_FWD` pointee's real
                // struct id via [`MemReader::resolve_arena_type`]
                // — a `struct sdt_data __arena *` field whose
                // pointee body lives in the sdt_alloc library's
                // BTF still chases as the typed per-task /
                // per-cgroup struct, instead of skipping with
                // "forward declaration; body not in this BTF".
                // `None` when no allocator with a typed payload
                // was discovered.
                arena_type_index: arena_type_index_ref,
            },
            &info,
        );

        report.maps.push(rendered);
        maps_rendered += 1;
    }
    tracing::debug!(
        elapsed_us = render_map_t0.elapsed().as_micros() as u64,
        rendered = maps_rendered,
        truncated = maps_truncated,
        "dump_state phase: per-map render"
    );

    report.dump_truncated_at_us = truncated_at_us;
    report
}

/// Walk every CPU's `kernel_cpustat`, `kernel_stat`, and (under
/// NO_HZ) `tick_sched` slots and produce a [`PerCpuTimeStats`]
/// vector — one entry per CPU index in `cap.per_cpu_offsets`.
///
/// Reads against the supplied [`super::reader::GuestMem`]. Each CPU's
/// per-CPU base for a symbol `S` is `S + per_cpu_offsets[cpu]`,
/// converted to a guest physical offset via
/// [`super::symbols::kva_to_pa`] using the supplied `page_offset`
/// (the standard direct-mapping translation; per-CPU pages always
/// live in the direct mapping).
///
/// `cpustat[]` is read as 8 contiguous u64s starting at the
/// resolved offset (length matches the indices captured —
/// CPUTIME_USER through CPUTIME_STEAL — leaving CPUTIME_GUEST /
/// CPUTIME_GUEST_NICE / CPUTIME_FORCEIDLE unread; the dump
/// surfaces them as zero in the unread slots, which is acceptable
/// since they're virt-guest specific or kernel-config gated and
/// distinct from the failure-dump narrative). `softirqs[]` reads
/// as `NR_SOFTIRQS` u32s, widened to u64 for the report. `irqs_sum`
/// is `unsigned long` (read as u64 — 64-bit only kernels are the
/// supported configuration). `iowait_sleeptime` is `ktime_t` /
/// `s64`; the value is cast to u64 (the kernel never produces
/// negative iowait time).
fn collect_per_cpu_time(cap: &CpuTimeCapture<'_>) -> Vec<PerCpuTimeStats> {
    use super::btf_offsets::{
        CPUTIME_IDLE, CPUTIME_IOWAIT, CPUTIME_IRQ, CPUTIME_NICE, CPUTIME_SOFTIRQ, CPUTIME_STEAL,
        CPUTIME_SYSTEM, CPUTIME_USER, NR_SOFTIRQS,
    };
    let mut out = Vec::with_capacity(cap.per_cpu_offsets.len());
    for (cpu_idx, &per_cpu_off) in cap.per_cpu_offsets.iter().enumerate() {
        let cpu = cpu_idx as u32;

        // kernel_cpustat::cpustat[N]: each slot is a u64 in nsec.
        // Read CPUTIME_USER through CPUTIME_STEAL (indices 0..=7).
        let cpustat_kva = cap.kernel_cpustat_kva.wrapping_add(per_cpu_off);
        let cpustat_pa = super::symbols::kva_to_pa(cpustat_kva, cap.page_offset);
        let cpustat_base = cap.offsets.kernel_cpustat_cpustat;
        let read_cpustat = |idx: usize| -> u64 {
            // sizeof(u64) == 8.
            cap.mem.read_u64(cpustat_pa, cpustat_base + idx * 8)
        };
        let cpustat_user_ns = read_cpustat(CPUTIME_USER);
        let cpustat_nice_ns = read_cpustat(CPUTIME_NICE);
        let cpustat_system_ns = read_cpustat(CPUTIME_SYSTEM);
        let cpustat_softirq_ns = read_cpustat(CPUTIME_SOFTIRQ);
        let cpustat_irq_ns = read_cpustat(CPUTIME_IRQ);
        let cpustat_idle_ns = read_cpustat(CPUTIME_IDLE);
        let cpustat_iowait_ns = read_cpustat(CPUTIME_IOWAIT);
        let cpustat_steal_ns = read_cpustat(CPUTIME_STEAL);

        // kernel_stat::softirqs[N]: each slot is a u32 (count).
        // Widen to u64 for reporting consistency with cpustat.
        let kstat_kva = cap.kstat_kva.wrapping_add(per_cpu_off);
        let kstat_pa = super::symbols::kva_to_pa(kstat_kva, cap.page_offset);
        let mut softirqs = [0u64; NR_SOFTIRQS];
        for (i, slot) in softirqs.iter_mut().enumerate() {
            // sizeof(unsigned int) == 4.
            *slot = cap
                .mem
                .read_u32(kstat_pa, cap.offsets.kstat_softirqs + i * 4) as u64;
        }

        // kernel_stat::irqs_sum: unsigned long. 64-bit only
        // kernels are supported, so read as u64.
        let irqs_sum = cap.mem.read_u64(kstat_pa, cap.offsets.kstat_irqs_sum);

        // tick_sched::iowait_sleeptime: ktime_t (s64) ns,
        // accumulated only under NO_HZ when the CPU enters idle
        // with nr_iowait > 0. Skip when the symbol or BTF offset
        // is absent.
        let iowait_sleeptime_ns = cap
            .tick_cpu_sched_kva
            .zip(cap.offsets.tick_sched_iowait_sleeptime)
            .map(|(tick_sym_kva, off)| {
                let kva = tick_sym_kva.wrapping_add(per_cpu_off);
                let pa = super::symbols::kva_to_pa(kva, cap.page_offset);
                cap.mem.read_u64(pa, off)
            });

        out.push(PerCpuTimeStats {
            cpu,
            cpustat_user_ns,
            cpustat_nice_ns,
            cpustat_system_ns,
            cpustat_softirq_ns,
            cpustat_irq_ns,
            cpustat_idle_ns,
            cpustat_iowait_ns,
            cpustat_steal_ns,
            softirqs,
            irqs_sum,
            iowait_sleeptime_ns,
        });
    }
    out
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

/// Load a BPF program's `struct btf` from guest memory at `btf_kva`.
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
///
/// Distinct from the [`super::bpf_map::BpfMapAccessor::load_program_btf`]
/// trait method (which dispatches across backends): this free function
/// is the guest-memory backend's actual KVA-based loader. The trait
/// method on `GuestMemMapAccessor` just forwards here.
pub(super) fn load_program_btf_kva(
    accessor: &GuestMemMapAccessor<'_>,
    btf_kva: u64,
    base_btf: &Btf,
) -> Option<Btf> {
    let kernel = accessor.kernel();
    let offsets = accessor.offsets();
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    // `struct btf` may be kmalloc'd (direct map) or vmalloc'd; use
    // translate_any_kva.
    let btf_pa = super::idr::translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        btf_kva,
        walk.l5,
        walk.tcr_el1,
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
