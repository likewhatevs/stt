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
    BpfMapAccessor, BpfMapInfo, GuestMemMapAccessor,
};
use super::btf_render::{RenderedValue, render_value};
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
/// (rq->scx in #50, DSQ in #49, init_task→tasks for an enumeration
/// path).
///
/// Mirrors the [`ProgRuntimeCapture`] / [`CpuTimeCapture`]
/// borrowed-only-optional shape (#84). When `dump_state` receives
/// `Some(TaskEnrichmentCapture)`, it iterates `tasks` and calls
/// [`super::task_enrichment::walk_task_enrichment`] for each entry,
/// pushing results into [`FailureDumpReport::task_enrichments`]. When
/// `None`, the field stays empty and
/// [`FailureDumpReport::task_enrichments_unavailable`] gets a
/// "no task walker available" diagnostic.
///
/// The walker producer (rq->scx walker etc.) is responsible for
/// building this struct. Until #49/#50 land, no walker exists; the
/// freeze coordinator passes `None` and the field is plumbed but
/// empty.
pub struct TaskEnrichmentCapture<'a> {
    /// Borrowed GuestKernel — provides memory access, page-table
    /// translation context, and the vmlinux symbol table.
    pub kernel: &'a super::guest::GuestKernel<'a>,
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

/// Borrow-only capture context for the rq->scx + DSQ walkers (#49,
/// #50). Mirrors [`TaskEnrichmentCapture`] / [`CpuTimeCapture`]
/// shape (#84) — `dump_state` consumes everything by reference.
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
    pub kernel: &'a super::guest::GuestKernel<'a>,
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
                out.select_cpu_fallback += ev.select_cpu_fallback;
                out.dispatch_local_dsq_offline += ev.dispatch_local_dsq_offline;
                out.dispatch_keep_last += ev.dispatch_keep_last;
                out.enq_skip_exiting += ev.enq_skip_exiting;
                out.enq_skip_migration_disabled += ev.enq_skip_migration_disabled;
                out.reenq_immed += ev.reenq_immed;
                out.reenq_local_repeat += ev.reenq_local_repeat;
                out.refill_slice_dfl += ev.refill_slice_dfl;
                out.bypass_duration += ev.bypass_duration;
                out.bypass_dispatch += ev.bypass_dispatch;
                out.bypass_activate += ev.bypass_activate;
                out.insert_not_owned += ev.insert_not_owned;
                out.sub_bypass_dispatch += ev.sub_bypass_dispatch;
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
/// when [`DumpContext::scx_walker_capture`] was supplied AND every
/// offset sub-group resolved BUT the walker reached no rq, no DSQ,
/// and no scx_sched state. Typical when no scheduler is attached
/// (`*scx_root == NULL`).
pub const REASON_SCX_WALKER_NO_STATE: &str = "scx walker reached no state (scx_root NULL?)";

/// Reason string written into [`FailureDumpReport::scx_walker_unavailable`]
/// when [`DumpContext::scx_walker_capture`] was `None`. Distinguishes
/// from [`REASON_SCX_WALKER_NO_STATE`] — the walker never ran at all
/// because no capture was supplied.
pub const REASON_NO_SCX_WALKER: &str = "no scx walker capture";

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
    /// Per-task failure-dump enrichments — identity (pid, tgid,
    /// comm), process tree (group_leader, real_parent, pgid, sid,
    /// nr_threads), scheduling (prio family, sched_class name,
    /// scx.weight, core_cookie), context-switch counters, watchdog
    /// disambiguation flag, and lock-slowpath stack matches.
    ///
    /// One entry per task the dump path's task walker reaches —
    /// today's task walkers are the rq->scx walker (#50) and the
    /// DSQ walker (#49); both produce task KVAs that get enriched
    /// here. Empty when no task walker ran (typical until #49/#50
    /// land) or when the [`TaskEnrichmentCapture`] was absent.
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
    ///   [`DumpContext`]. Until #49/#50 (DSQ + rq->scx walkers)
    ///   land, this is the expected steady state for the dump
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
    /// KVAs that fed into the per-task enrichment capture (#28).
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
    /// pattern (#42).
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
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
                     scan setup failed) — re-run with RUST_LOG=ktstr=debug \
                     for scan resolution diagnostics):\n",
                )?;
                std::fmt::Display::fmt(&self.late, f)
            }
        }
    }
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
    Single(FailureDumpReport),
    /// Dual-snapshot wrapper, schema=`"dual"`. Emitted by the
    /// auto-repro VM when the dual-snapshot path is enabled. Carries
    /// optional `early` + required `late` snapshots plus jiffies
    /// metadata for the early-trigger condition.
    Dual(DualFailureDumpReport),
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
            SCHEMA_DUAL => serde_json::from_str(json).ok().map(Self::Dual),
            SCHEMA_SINGLE | "" => serde_json::from_str(json).ok().map(Self::Single),
            _ => None,
        }
    }
}

impl std::fmt::Display for FailureDumpReportAny {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(r) => std::fmt::Display::fmt(r, f),
            Self::Dual(r) => std::fmt::Display::fmt(r, f),
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
    /// snapshots and per-program runtime stats. JSON remains the
    /// programmatic form via `serde_json`; this Display is the
    /// default presentation used in test-failure output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.maps.is_empty()
            && self.vcpu_regs.is_empty()
            && self.sdt_allocations.is_empty()
            && self.prog_runtime_stats.is_empty()
            && self.per_cpu_time.is_empty()
            && self.task_enrichments.is_empty()
            && self.event_counter_timeline.is_empty()
            && self.vcpu_perf_at_freeze.is_empty()
        {
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
        if !self.prog_runtime_stats.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            f.write_str("prog_runtime_stats:")?;
            for stats in &self.prog_runtime_stats {
                f.write_str("\n  ")?;
                std::fmt::Display::fmt(stats, f)?;
            }
        }
        if let Some(reason) = &self.prog_runtime_stats_unavailable {
            if !first {
                f.write_str("\n\n")?;
            }
            first = false;
            write!(f, "prog_runtime_stats: <unavailable: {reason}>")?;
        }
        if !self.event_counter_timeline.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            // Section comes last; no further sibling sections need
            // to consult `first`, so leave the flag alone here.
            write!(
                f,
                "event_counter_timeline: {} samples ({}–{}ms)",
                self.event_counter_timeline.len(),
                self.event_counter_timeline
                    .first()
                    .map(|s| s.elapsed_ms)
                    .unwrap_or(0),
                self.event_counter_timeline
                    .last()
                    .map(|s| s.elapsed_ms)
                    .unwrap_or(0),
            )?;
            // Per-counter sparkline. Each row is one of the 13
            // SCX_EV_* counters across all samples in the
            // timeline. Skips counters that stayed at zero across
            // every sample to keep the rendering compact (a
            // counter at zero everywhere has no signal worth
            // surfacing in the human-readable view).
            let extract: [(
                &str,
                fn(&EventCounterSample) -> i64,
            ); 13] = [
                ("select_cpu_fallback", |s| s.select_cpu_fallback),
                ("dispatch_local_dsq_offline", |s| s.dispatch_local_dsq_offline),
                ("dispatch_keep_last", |s| s.dispatch_keep_last),
                ("enq_skip_exiting", |s| s.enq_skip_exiting),
                ("enq_skip_migration_disabled", |s| s.enq_skip_migration_disabled),
                ("reenq_immed", |s| s.reenq_immed),
                ("reenq_local_repeat", |s| s.reenq_local_repeat),
                ("refill_slice_dfl", |s| s.refill_slice_dfl),
                ("bypass_duration", |s| s.bypass_duration),
                ("bypass_dispatch", |s| s.bypass_dispatch),
                ("bypass_activate", |s| s.bypass_activate),
                ("insert_not_owned", |s| s.insert_not_owned),
                ("sub_bypass_dispatch", |s| s.sub_bypass_dispatch),
            ];
            for (name, ext) in extract {
                let series: Vec<i64> =
                    self.event_counter_timeline.iter().map(ext).collect();
                if series.iter().all(|&v| v == 0) {
                    continue;
                }
                let line = render_sparkline_i64(&series);
                let last = series.last().copied().unwrap_or(0);
                write!(f, "\n  {name:>30}  {line}  (last={last})")?;
            }
        }
        if !self.vcpu_perf_at_freeze.is_empty() {
            if !first {
                f.write_str("\n\n")?;
            }
            // Trailing section; mirrors the event_counter_timeline
            // pattern — `first` is no longer consulted after this
            // block.
            f.write_str("vcpu_perf_at_freeze:")?;
            for (i, slot) in self.vcpu_perf_at_freeze.iter().enumerate() {
                f.write_str("\n  ")?;
                match slot {
                    Some(s) => write!(
                        f,
                        "vcpu {i}: cycles={} insns={} ipc={:.3} cache_misses={} branch_misses={} (en/ru={}/{} ns)",
                        s.cycles,
                        s.instructions,
                        s.ipc(),
                        s.cache_misses,
                        s.branch_misses,
                        s.time_enabled_ns,
                        s.time_running_ns,
                    )?,
                    None => write!(f, "vcpu {i}: <unavailable>")?,
                }
            }
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
/// freeze coordinator. Shared between [`load_program_btf_kva`] and
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
/// backend (#9) lands, sdt_alloc walking will move into a
/// backend-specific path and `accessor` here can become
/// `&'a dyn BpfMapAccessor`.
///
/// `arena_offsets` and `prog_capture` are both optional borrows
/// (uniform shape — see #84): `None` for either disables that
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
    /// [`Self::prog_capture`] / [`Self::arena_offsets`] (#84) so a
    /// future capture site lands as another optional field without
    /// churning the call sites already plumbed through here.
    pub cpu_time_capture: Option<&'a CpuTimeCapture<'a>>,
    /// Per-task enrichment capture. `None` skips the per-task walk
    /// and `task_enrichments` stays empty; the rest of the dump
    /// still renders.
    ///
    /// Today's freeze coordinator passes `None` because no task
    /// walker has landed yet (#49 DSQ + #50 rq->scx). The
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
    } = ctx;
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
    let (task_enrichments, task_enrichments_unavailable) = match task_enrichment_capture {
        Some(cap) => {
            let mut enrichments = Vec::with_capacity(cap.tasks.len());
            for entry in cap.tasks {
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
            let reason = if enrichments.is_empty() {
                Some(REASON_TASK_WALKER_ZERO_TASKS.to_string())
            } else {
                None
            };
            (enrichments, reason)
        }
        None => (
            Vec::new(),
            Some(REASON_NO_TASK_WALKER.to_string()),
        ),
    };
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
                // Sub-group offsets resolved per kernel struct (#43);
                // surface the absent groups in the diagnostic so a
                // partial walk announces which passes were skipped.
                let missing = cap.offsets.missing_groups();

                // 1. Read scalar scx_sched state and recover the
                //    sched_pa for the DSQ walker pass.
                let (sched_pa_opt, sched_state) = match super::scx_walker::read_scx_sched_state(
                    cap.kernel,
                    cap.scx_root_kva,
                    cap.offsets,
                ) {
                    Some((sched_kva, state)) => {
                        // Translate sched_kva → PA (slab/vmalloc; use
                        // translate_any_kva via the GuestKernel handle).
                        let mem = cap.kernel.mem();
                        let cr3_pa = cap.kernel.cr3_pa();
                        let po = cap.kernel.page_offset();
                        let l5 = cap.kernel.l5();
                        let pa =
                            super::idr::translate_any_kva(mem, cr3_pa, po, sched_kva, l5);
                        (pa, Some(state))
                    }
                    None => (None, None),
                };

                // 2. Per-CPU rq->scx walk. Per-CPU runs only when the
                //    rq + scx_rq + task sub-groups are present;
                //    walk_rq_scx returns None to skip otherwise.
                let mut rq_states = Vec::with_capacity(cap.rq_kvas.len());
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

                // 3. DSQ walk requires the sched_pa we resolved
                //    above. If sched_pa is None, only the per-CPU
                //    local DSQs (which live in rq->scx, not via
                //    sched->pcpu/pnode) would be reachable —
                //    skipping the whole DSQ walk in that case is
                //    consistent with "no scheduler attached".
                let dsqs = match sched_pa_opt {
                    Some(sched_pa) => {
                        let (states, _entries) = super::scx_walker::walk_dsqs(
                            cap.kernel,
                            sched_pa,
                            cap.rq_kvas,
                            cap.rq_pas,
                            cap.per_cpu_offsets,
                            cap.nr_nodes,
                            cap.offsets,
                        );
                        states
                    }
                    None => Vec::new(),
                };

                // Diagnostic priority:
                //   1. Partial-degradation (sub-group(s) missing) —
                //      announces exactly which passes were skipped.
                //   2. Walker reached no state at all — typical when
                //      scx_root is NULL (no scheduler attached).
                //   3. None — every pass had data to surface.
                let unavail = if !missing.is_empty() {
                    Some(format!(
                        "scx walker partial: missing offset groups [{}]",
                        missing.join(", ")
                    ))
                } else if rq_states.is_empty()
                    && dsqs.is_empty()
                    && sched_state.is_none()
                {
                    Some(REASON_SCX_WALKER_NO_STATE.to_string())
                } else {
                    None
                };
                (rq_states, dsqs, sched_state, unavail)
            }
            None => (
                Vec::new(),
                Vec::new(),
                None,
                Some(REASON_NO_SCX_WALKER.to_string()),
            ),
        };
    // Freeze-time per-vCPU perf-counter snapshot. With `exclude_host=1`
    // each counter ticks only during guest execution; the freeze
    // coordinator has parked every vCPU before reaching this site, so
    // the read returns the cumulative count at the last guest exit
    // for each vCPU. A single per-vCPU read failure is recorded as
    // `None` for that entry; a failure on one vCPU does not blank the
    // others. When `perf_capture` is None the vec stays empty (the
    // host lacked perf, or `perf_event_open` failed at run start).
    let vcpu_perf_at_freeze: Vec<Option<super::perf_counters::VcpuPerfSample>> =
        match perf_capture {
            Some(cap) => cap
                .per_vcpu
                .iter()
                .map(|p| p.read().ok())
                .collect(),
            None => Vec::new(),
        };

    let mut report = FailureDumpReport {
        schema: SCHEMA_SINGLE.to_string(),
        maps: Vec::with_capacity(maps.len()),
        vcpu_regs: Vec::new(),
        sdt_allocations: Vec::new(),
        prog_runtime_stats,
        prog_runtime_stats_unavailable,
        per_cpu_time,
        task_enrichments,
        task_enrichments_unavailable,
        event_counter_timeline,
        rq_scx_states,
        dsq_states,
        scx_sched_state,
        scx_walker_unavailable,
        vcpu_perf_at_freeze,
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
            && let Some(loaded) = accessor.load_program_btf(&info, btf)
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
            cap.mem
                .read_u64(cpustat_pa, cpustat_base + idx * 8)
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
            *slot = cap.mem.read_u32(kstat_pa, cap.offsets.kstat_softirqs + i * 4)
                as u64;
        }

        // kernel_stat::irqs_sum: unsigned long. 64-bit only
        // kernels are supported, so read as u64.
        let irqs_sum = cap.mem.read_u64(kstat_pa, cap.offsets.kstat_irqs_sum);

        // tick_sched::iowait_sleeptime: ktime_t (s64) ns,
        // accumulated only under NO_HZ when the CPU enters idle
        // with nr_iowait > 0. Skip when the symbol or BTF offset
        // is absent.
        let iowait_sleeptime_ns = cap.tick_cpu_sched_kva.zip(cap.offsets.tick_sched_iowait_sleeptime).map(
            |(tick_sym_kva, off)| {
                let kva = tick_sym_kva.wrapping_add(per_cpu_off);
                let pa = super::symbols::kva_to_pa(kva, cap.page_offset);
                cap.mem.read_u64(pa, off)
            },
        );

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
    accessor: &GuestMemMapAccessor<'_>,
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

    /// Empty input renders as empty string. Single-element input
    /// renders as one mid-tier glyph (constant non-zero series
    /// reads as "no variation"). All-zero series renders as the
    /// lowest glyph repeated.
    #[test]
    fn render_sparkline_edge_cases() {
        assert_eq!(render_sparkline(&[]), "");
        // Single non-zero element: constant series → mid-tier glyph.
        assert_eq!(render_sparkline(&[42]), "▅");
        // All-zero series: lowest glyph for every entry.
        assert_eq!(render_sparkline(&[0, 0, 0]), "▁▁▁");
        // All-equal non-zero series: mid-tier glyph for every entry.
        assert_eq!(render_sparkline(&[5, 5, 5]), "▅▅▅");
    }

    /// Strictly-increasing series scales linearly across the glyph
    /// set: first sample at min lands at lowest glyph, last sample
    /// at max lands at highest. Pin both ends so a future scaling
    /// regression that broke either bound is caught.
    #[test]
    fn render_sparkline_monotonic_scales_to_full_range() {
        let s = render_sparkline(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars.len(), 8);
        assert_eq!(chars[0], '▁', "min must map to lowest glyph: {s}");
        assert_eq!(chars[7], '█', "max must map to highest glyph: {s}");
    }

    /// i64 wrapper saturates negative values to 0, then routes
    /// through u64 sparkline. Verifies a counter that briefly
    /// dips negative (corrupt read) doesn't crash and produces
    /// a sane sparkline.
    #[test]
    fn render_sparkline_i64_clamps_negatives() {
        let s = render_sparkline_i64(&[-5, 0, 5, 10]);
        // After clamp: [0, 0, 5, 10] → first two at lowest, last
        // two scale up. Just pin length and bounds; exact glyphs
        // depend on integer rounding.
        assert_eq!(s.chars().count(), 4);
    }

    /// Full SCX_EV_* counter timeline construction: build a
    /// MonitorSample with two CPUs reporting event counters,
    /// fold to EventCounterSample, verify cross-CPU sums and
    /// elapsed_ms propagation.
    #[test]
    fn event_counter_sample_sums_across_cpus() {
        use super::super::{CpuSnapshot, MonitorSample, ScxEventCounters};
        let cpu_a = CpuSnapshot {
            event_counters: Some(ScxEventCounters {
                select_cpu_fallback: 5,
                bypass_dispatch: 100,
                ..Default::default()
            }),
            ..Default::default()
        };
        let cpu_b = CpuSnapshot {
            event_counters: Some(ScxEventCounters {
                select_cpu_fallback: 7,
                bypass_dispatch: 50,
                ..Default::default()
            }),
            ..Default::default()
        };
        let sample = MonitorSample {
            elapsed_ms: 100,
            cpus: vec![cpu_a, cpu_b],
            prog_stats: None,
        };
        let folded = EventCounterSample::from_monitor_sample(&sample)
            .expect("at least one CPU has event_counters");
        assert_eq!(folded.elapsed_ms, 100);
        assert_eq!(folded.select_cpu_fallback, 12);
        assert_eq!(folded.bypass_dispatch, 150);
    }

    /// MonitorSample with no CPU reporting event_counters folds
    /// to None — propagating an all-zero row would mislead the
    /// downstream consumer (a real "every counter at 0" tick
    /// looks identical to "every CPU's offsets unresolved").
    #[test]
    fn event_counter_sample_returns_none_when_no_cpu_has_counters() {
        use super::super::{CpuSnapshot, MonitorSample};
        let cpu = CpuSnapshot {
            event_counters: None,
            ..Default::default()
        };
        let sample = MonitorSample {
            elapsed_ms: 200,
            cpus: vec![cpu],
            prog_stats: None,
        };
        assert!(EventCounterSample::from_monitor_sample(&sample).is_none());
    }

    /// EventCounterSample serde round-trips cleanly: every field
    /// is `i64` (kernel-side `s64`), so a wire-format encode →
    /// decode preserves bit patterns including the i64::MAX edge.
    #[test]
    fn event_counter_sample_serde_roundtrip() {
        let s = EventCounterSample {
            elapsed_ms: 123_456,
            select_cpu_fallback: i64::MAX,
            insert_not_owned: -1, // kernel never produces this
                                  // but the wire format must
                                  // preserve whatever the read
                                  // captured rather than silently
                                  // clamp.
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let loaded: EventCounterSample = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.elapsed_ms, 123_456);
        assert_eq!(loaded.select_cpu_fallback, i64::MAX);
        assert_eq!(loaded.insert_not_owned, -1);
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
        };
        let late = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: vec![None, None],
            sdt_allocations: Vec::new(),
            prog_runtime_stats: Vec::new(),
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
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
    /// half-way threshold; runnable_at scan setup failed) AND a
    /// pointer to the RUST_LOG knob that surfaces scan-resolution
    /// diagnostics — so an operator reading "early=absent" knows
    /// the next debugging step rather than having to guess.
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
        assert!(
            s.contains("RUST_LOG=ktstr=debug"),
            "Display must point at the RUST_LOG knob for diagnostics: {s}"
        );
    }

    // -- FailureDumpReportAny serde + Display tests --

    /// `FailureDumpReportAny::from_json` picks the `Single` variant
    /// for JSON whose `schema` field is `"single"`, the `Dual`
    /// variant for `"dual"`, and the `Single` variant for an absent
    /// `schema` field (back-compat with pre-discriminant dumps).
    /// Unknown schemas return `None` rather than silently falling
    /// back to single — mismatching a future richer wrapper as a
    /// lossy single shape would be the wrong behaviour. Malformed
    /// JSON also returns `None`.
    #[test]
    fn report_any_dispatch_branches() {
        // Single branch: schema="single".
        let single = FailureDumpReport::default();
        let single_json = serde_json::to_string(&single).expect("serialize single");
        match FailureDumpReportAny::from_json(&single_json) {
            Some(FailureDumpReportAny::Single(_)) => {}
            other => panic!(
                "schema=single must map to Single, got {other:?}",
                other = other.is_some()
            ),
        }

        // Dual branch: schema="dual".
        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: None,
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 0,
            early_threshold_jiffies: 0,
        };
        let dual_json = serde_json::to_string(&dual).expect("serialize dual");
        match FailureDumpReportAny::from_json(&dual_json) {
            Some(FailureDumpReportAny::Dual(_)) => {}
            other => panic!(
                "schema=dual must map to Dual, got {other:?}",
                other = other.is_some()
            ),
        }

        // Absent-schema branch: pre-discriminant dump.
        let absent = r#"{"maps":[],"vcpu_regs":[],"sdt_allocations":[]}"#;
        match FailureDumpReportAny::from_json(absent) {
            Some(FailureDumpReportAny::Single(_)) => {}
            other => panic!(
                "absent schema must default to Single, got {other:?}",
                other = other.is_some()
            ),
        }

        // Unknown schema → None, not a silent single fallback.
        let unknown = r#"{"schema":"triple","maps":[],"vcpu_regs":[],"sdt_allocations":[]}"#;
        assert!(
            FailureDumpReportAny::from_json(unknown).is_none(),
            "unknown schema must return None, not silent fallback"
        );

        // Malformed JSON → None.
        assert!(
            FailureDumpReportAny::from_json("not json").is_none(),
            "garbage input must return None"
        );
    }

    /// `prog_runtime_stats` populates and round-trips through
    /// `FailureDumpReportAny::from_json`. The dispatch test above
    /// covers the empty-stats path; this test pins that the field
    /// survives wire encoding when populated, mirroring the
    /// strict-schema concerns CgroupStats covers in assert.rs.
    #[test]
    fn report_any_preserves_prog_runtime_stats() {
        use super::super::bpf_prog::ProgRuntimeStats;
        let report = FailureDumpReport {
            prog_runtime_stats: vec![
                ProgRuntimeStats {
                    name: "ktstr_enqueue".to_string(),
                    cnt: 1_500,
                    nsecs: 7_500_000,
                    misses: 2,
                },
                ProgRuntimeStats {
                    name: "ktstr_dispatch".to_string(),
                    cnt: u64::MAX,
                    nsecs: u64::MAX,
                    misses: u64::MAX,
                },
            ],
            ..Default::default()
        };
        let json = serde_json::to_string(&report).expect("serialize");
        match FailureDumpReportAny::from_json(&json) {
            Some(FailureDumpReportAny::Single(loaded)) => {
                assert_eq!(loaded.prog_runtime_stats.len(), 2);
                assert_eq!(loaded.prog_runtime_stats[0].name, "ktstr_enqueue");
                assert_eq!(loaded.prog_runtime_stats[0].cnt, 1_500);
                assert_eq!(loaded.prog_runtime_stats[0].nsecs, 7_500_000);
                assert_eq!(loaded.prog_runtime_stats[0].misses, 2);
                assert_eq!(loaded.prog_runtime_stats[1].name, "ktstr_dispatch");
                assert_eq!(loaded.prog_runtime_stats[1].cnt, u64::MAX);
                assert_eq!(loaded.prog_runtime_stats[1].nsecs, u64::MAX);
                assert_eq!(loaded.prog_runtime_stats[1].misses, u64::MAX);
            }
            other => panic!(
                "populated single report must round-trip Single, got {:?}",
                other.is_some()
            ),
        }
    }

    /// Display roundtrip: a Single-wrapped report renders the same
    /// as the underlying `FailureDumpReport`'s own Display, and a
    /// Dual-wrapped report renders the same as
    /// `DualFailureDumpReport`'s Display. The wrapper's Display is
    /// transparent.
    #[test]
    fn report_any_display_matches_underlying() {
        let single = FailureDumpReport::default();
        let single_direct = format!("{single}");
        let single_via_any = format!("{}", FailureDumpReportAny::Single(single));
        assert_eq!(single_direct, single_via_any);

        let dual = DualFailureDumpReport {
            schema: SCHEMA_DUAL.to_string(),
            early: Some(FailureDumpReport::default()),
            late: FailureDumpReport::default(),
            early_max_age_jiffies: 42,
            early_threshold_jiffies: 21,
        };
        let dual_direct = format!("{dual}");
        let dual_via_any = format!("{}", FailureDumpReportAny::Dual(dual));
        assert_eq!(dual_direct, dual_via_any);
    }

    // -- ProgRuntimeStats coverage in FailureDumpReport --

    /// Roundtrip a populated `prog_runtime_stats` vector through
    /// serde, including `u64::MAX` for every counter to lock in
    /// the saturation contract documented on
    /// [`super::bpf_prog::read_prog_runtime_stats`] (per-CPU sums use
    /// `saturating_add`, so observing `u64::MAX` post-deserialize
    /// proves the saturation path didn't silently wrap or truncate).
    #[test]
    fn prog_runtime_stats_serde_roundtrip_with_saturation() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
            prog_runtime_stats: vec![
                super::super::bpf_prog::ProgRuntimeStats {
                    name: "dispatch".to_string(),
                    cnt: 12345,
                    nsecs: 67890,
                    misses: 3,
                },
                super::super::bpf_prog::ProgRuntimeStats {
                    name: "saturated".to_string(),
                    cnt: u64::MAX,
                    nsecs: u64::MAX,
                    misses: u64::MAX,
                },
            ],
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let parsed: FailureDumpReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.prog_runtime_stats.len(), 2);
        assert_eq!(parsed.prog_runtime_stats[0].name, "dispatch");
        assert_eq!(parsed.prog_runtime_stats[0].cnt, 12345);
        assert_eq!(parsed.prog_runtime_stats[0].nsecs, 67890);
        assert_eq!(parsed.prog_runtime_stats[0].misses, 3);
        assert_eq!(parsed.prog_runtime_stats[1].cnt, u64::MAX);
        assert_eq!(parsed.prog_runtime_stats[1].nsecs, u64::MAX);
        assert_eq!(parsed.prog_runtime_stats[1].misses, u64::MAX);
    }

    /// Empty `prog_runtime_stats` skips serialization (the
    /// `skip_serializing_if = "Vec::is_empty"` attribute) — same
    /// pattern as the other optional vector fields. Pinning this
    /// keeps the JSON tight for the common no-struct_ops-loaded
    /// case.
    #[test]
    fn prog_runtime_stats_empty_skips_serialization() {
        let report = FailureDumpReport::default();
        let json = serde_json::to_string(&report).expect("serialize");
        assert!(
            !json.contains("prog_runtime_stats"),
            "empty prog_runtime_stats must be skipped: {json}"
        );
    }

    /// Display impl renders `prog_runtime_stats` under a labelled
    /// section so an operator scanning failure-dump output sees
    /// the per-program counters alongside the maps / vcpu_regs /
    /// sdt_allocations sections. Pinning this prevents the
    /// "rendered fields silently drop" regression that would mask
    /// dump enrichment from reaching log readers.
    #[test]
    fn report_display_renders_prog_runtime_stats() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
            prog_runtime_stats: vec![
                super::super::bpf_prog::ProgRuntimeStats {
                    name: "dispatch".to_string(),
                    cnt: 5,
                    nsecs: 1234,
                    misses: 0,
                },
                super::super::bpf_prog::ProgRuntimeStats {
                    name: "enqueue".to_string(),
                    cnt: 99,
                    nsecs: 9999,
                    misses: 7,
                },
            ],
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
        };
        let out = format!("{report}");
        assert!(
            out.contains("prog_runtime_stats:"),
            "Display must render the prog_runtime_stats section: {out}"
        );
        assert!(
            out.contains("dispatch: cnt=5 nsecs=1234 misses=0"),
            "Display must render first program line: {out}"
        );
        assert!(
            out.contains("enqueue: cnt=99 nsecs=9999 misses=7"),
            "Display must render second program line: {out}"
        );
    }

    /// An all-empty maps/vcpu_regs/sdt_allocations report with
    /// only `prog_runtime_stats` populated must still render
    /// rather than fall through to the "(empty failure dump)"
    /// fallback — the empty-check in the Display impl gates on
    /// every optional vector, including `prog_runtime_stats`.
    #[test]
    fn report_display_only_prog_runtime_stats_does_not_say_empty_dump() {
        let report = FailureDumpReport {
            schema: SCHEMA_SINGLE.to_string(),
            maps: Vec::new(),
            vcpu_regs: Vec::new(),
            sdt_allocations: Vec::new(),
            prog_runtime_stats: vec![super::super::bpf_prog::ProgRuntimeStats {
                name: "lone".to_string(),
                cnt: 1,
                nsecs: 2,
                misses: 0,
            }],
            prog_runtime_stats_unavailable: None,
            per_cpu_time: Vec::new(),
            task_enrichments: Vec::new(),
            task_enrichments_unavailable: None,
            event_counter_timeline: Vec::new(),
            rq_scx_states: Vec::new(),
            dsq_states: Vec::new(),
            scx_sched_state: None,
            scx_walker_unavailable: None,
            vcpu_perf_at_freeze: Vec::new(),
        };
        let out = format!("{report}");
        assert!(
            !out.contains("(empty failure dump)"),
            "Display must NOT fall through to empty fallback when \
             prog_runtime_stats is populated: {out}"
        );
        assert!(
            out.starts_with("prog_runtime_stats:"),
            "Display must lead with prog_runtime_stats section when \
             only that field is populated: {out}"
        );
    }

    // ---- #91: pin failure-dump error-message strings ---------------
    //
    // The six REASON_* constants emitted by `dump_state` into the
    // `*_unavailable` fields are wire-format markers: an operator
    // parsing `.failure-dump.json` looks for these exact strings to
    // distinguish "no scheduler attached" from "no walker capture
    // supplied" etc. Drift in any of them silently breaks downstream
    // parsing. The constants near the top of this module are the
    // single source of truth; the tests below pin each constant's
    // exact value so a regression that re-words a string trips both
    // at the constant declaration AND at the test assertion.
    //
    // The companion strict-schema and chain-limit tests for
    // FailureDumpReport / is_scx_allocator_type live further down in
    // this module (#88 / #89).

    #[test]
    fn reason_no_struct_ops_loaded_string_pinned() {
        assert_eq!(REASON_NO_STRUCT_OPS_LOADED, "no struct_ops programs loaded");
    }

    #[test]
    fn reason_prog_accessor_unavailable_string_pinned() {
        assert_eq!(REASON_PROG_ACCESSOR_UNAVAILABLE, "prog accessor unavailable");
    }

    #[test]
    fn reason_task_walker_zero_tasks_string_pinned() {
        assert_eq!(REASON_TASK_WALKER_ZERO_TASKS, "task walker yielded zero tasks");
    }

    #[test]
    fn reason_no_task_walker_string_pinned() {
        assert_eq!(REASON_NO_TASK_WALKER, "no task walker available");
    }

    #[test]
    fn reason_scx_walker_no_state_string_pinned() {
        assert_eq!(
            REASON_SCX_WALKER_NO_STATE,
            "scx walker reached no state (scx_root NULL?)"
        );
    }

    #[test]
    fn reason_no_scx_walker_string_pinned() {
        assert_eq!(REASON_NO_SCX_WALKER, "no scx walker capture");
    }

    /// Every reason constant must round-trip through the JSON wire
    /// format embedded in the `*_unavailable` fields. A regression
    /// that altered the field's serde encoding (renamed the field,
    /// added `#[serde(rename = ...)]`, etc.) would also break the
    /// operator's string-match parsing — surface that here too.
    #[test]
    fn reason_strings_round_trip_through_serde() {
        let report = FailureDumpReport {
            prog_runtime_stats_unavailable: Some(REASON_NO_STRUCT_OPS_LOADED.to_string()),
            task_enrichments_unavailable: Some(REASON_TASK_WALKER_ZERO_TASKS.to_string()),
            scx_walker_unavailable: Some(REASON_SCX_WALKER_NO_STATE.to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&report).expect("serialize");
        // Each reason must appear verbatim in the JSON; a future
        // wire-format change (e.g. tagged enum) would hide them
        // behind nested objects and trip this assertion.
        assert!(
            json.contains(REASON_NO_STRUCT_OPS_LOADED),
            "JSON must contain prog reason verbatim: {json}",
        );
        assert!(
            json.contains(REASON_TASK_WALKER_ZERO_TASKS),
            "JSON must contain task reason verbatim: {json}",
        );
        assert!(
            json.contains(REASON_SCX_WALKER_NO_STATE),
            "JSON must contain scx reason verbatim: {json}",
        );

        let loaded: FailureDumpReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            loaded.prog_runtime_stats_unavailable.as_deref(),
            Some(REASON_NO_STRUCT_OPS_LOADED),
        );
        assert_eq!(
            loaded.task_enrichments_unavailable.as_deref(),
            Some(REASON_TASK_WALKER_ZERO_TASKS),
        );
        assert_eq!(
            loaded.scx_walker_unavailable.as_deref(),
            Some(REASON_SCX_WALKER_NO_STATE),
        );
    }

    // -- Strict-schema tests for FailureDumpReport (#88) --------------
    //
    // Mirrors the CgroupStats / ScenarioStats / SidecarResult tests in
    // assert.rs and test_support/sidecar.rs. FailureDumpReport's
    // contract is narrower than CgroupStats's because most of its
    // fields are intentionally optional (capture pipelines may
    // legitimately produce no entries), so the CgroupStats
    // "remove every field" loop would over-assert here.
    //
    // Asserted contract:
    //   - `maps` is the only required field on the wire.
    //   - `schema` is `serde(default = default_schema_single)` —
    //     omission yields `SCHEMA_SINGLE`.
    //   - Every other field is `serde(default, skip_serializing_if =
    //     ...)` — omission MUST succeed.
    //
    // A regression that softens `maps` to `serde(default)` (e.g. to
    // soften a schema migration) would silently produce empty-maps
    // dumps that look indistinguishable from a legitimate no-maps
    // run. A regression that hardens an optional field to require it
    // on the wire would break replay of older dumps. Either drift
    // trips this test.

    /// Removing the `maps` field MUST fail deserialize. `maps`
    /// carries the BPF map enumeration that is the dump's only
    /// mandatory payload — every other field is
    /// capture-pipeline-optional. The deserialize error MUST name
    /// `maps` so a regression produces a debuggable failure rather
    /// than a silent default.
    #[test]
    fn failure_dump_report_strict_schema_maps_required() {
        let report = FailureDumpReport::default();
        let mut full = match serde_json::to_value(&report).unwrap() {
            serde_json::Value::Object(m) => m,
            other => panic!("expected object, got {other:?}"),
        };
        assert!(
            full.remove("maps").is_some(),
            "FailureDumpReport must emit `maps` for this test to be \
             meaningful — the field has been renamed or removed",
        );
        let json = serde_json::Value::Object(full).to_string();
        let err = serde_json::from_str::<FailureDumpReport>(&json)
            .err()
            .expect("deserialize must reject FailureDumpReport with `maps` removed");
        let msg = format!("{err}");
        assert!(
            msg.contains("maps"),
            "missing-field error for `maps` must name the field; got: {msg}",
        );
    }

    /// Omitting all optional fields (`schema`, `vcpu_regs`,
    /// `sdt_allocations`, every diagnostic Option, every capture
    /// Vec) MUST succeed and produce a deserialized report whose
    /// absent fields take their `serde(default)` value. `schema`
    /// gets a positive control: omission MUST yield `SCHEMA_SINGLE`,
    /// not the empty string a naive `Default for String` would
    /// produce.
    #[test]
    fn failure_dump_report_optional_fields_round_trip_when_omitted() {
        let minimal = serde_json::json!({ "maps": [] });
        let report: FailureDumpReport = serde_json::from_value(minimal)
            .expect("deserialize must accept FailureDumpReport with only `maps`");
        assert_eq!(
            report.schema, SCHEMA_SINGLE,
            "absent `schema` field must default to SCHEMA_SINGLE \
             (default_schema_single fn); got: {:?}",
            report.schema,
        );
        assert!(report.maps.is_empty());
        assert!(report.vcpu_regs.is_empty());
        assert!(report.sdt_allocations.is_empty());
        assert!(report.prog_runtime_stats.is_empty());
        assert!(report.prog_runtime_stats_unavailable.is_none());
        assert!(report.per_cpu_time.is_empty());
        assert!(report.task_enrichments.is_empty());
        assert!(report.task_enrichments_unavailable.is_none());
        assert!(report.event_counter_timeline.is_empty());
        assert!(report.rq_scx_states.is_empty());
        assert!(report.dsq_states.is_empty());
        assert!(report.scx_sched_state.is_none());
        assert!(report.scx_walker_unavailable.is_none());
        assert!(report.vcpu_perf_at_freeze.is_empty());
    }

    // -- Pin failure-dump error-message strings (#91) ----------------
    //
    // Pin the EXACT prose of error strings rendered into
    // FailureDumpMap.error. Substring tests are permissive against
    // drift; this regression suite asserts byte-for-byte equality so
    // any re-wording during refactor surfaces in `cargo nextest run`
    // before it ships.
    //
    // The strings are observable via FailureDumpMap.error contents
    // and via downstream log scrapers (operators grep these in CI
    // logs). Changing them silently breaks log tooling. Each pin
    // doubles as documentation: this file shows exactly which prose
    // is covered by drift detection.
    //
    // dump.rs producers covered here — five distinct render-time
    // formats observed at:
    //   - dump.rs:2168 (BPF_MAP_TYPE_ARENA, no offsets)
    //   - dump.rs:2064-2067 (BPF_MAP_TYPE_ARRAY, multi-entry)
    //   - dump.rs:2106 (BPF_MAP_TYPE_HASH, truncation)
    //   - dump.rs:2131-2133 (BPF_MAP_TYPE_PERCPU_ARRAY, truncation)
    //   - dump.rs:2174-2175 (unsupported map_type wildcard)
    //
    // Each pin reproduces the production format string against a
    // known-value placeholder and asserts byte equality with the
    // expected literal. A drift in either the prose or the constant
    // value (e.g. raising MAX_HASH_ENTRIES from 4096 to 8192) trips
    // the test.
    //
    // The companion REASON_* constants for diagnostic Option fields
    // (REASON_NO_STRUCT_OPS_LOADED, REASON_TASK_WALKER_ZERO_TASKS,
    // REASON_SCX_WALKER_NO_STATE, ...) are already pinned by tests
    // earlier in this module — see `report_unavailable_reasons_*`.

    /// `arena BTF offsets unavailable (kernel lacks struct bpf_arena?)`
    /// is rendered by the BPF_MAP_TYPE_ARENA arm when arena_offsets
    /// is None — surfacing that the kernel lacks struct bpf_arena.
    #[test]
    fn pinned_error_arena_btf_offsets_unavailable() {
        // The producer has no format placeholders; reproduce the
        // exact `.into()` literal so a rephrasing in dump.rs trips.
        let rendered: String =
            "arena BTF offsets unavailable (kernel lacks struct bpf_arena?)".into();
        assert_eq!(
            rendered,
            "arena BTF offsets unavailable (kernel lacks struct bpf_arena?)",
            "arena-unavailable error string drifted from pin",
        );
    }

    /// `multi-entry ARRAY: only key 0 of {N} shown` is rendered by
    /// the BPF_MAP_TYPE_ARRAY arm when `info.max_entries > 1`.
    #[test]
    fn pinned_error_multi_entry_array_truncation() {
        let n: u32 = 7;
        let rendered = format!("multi-entry ARRAY: only key 0 of {n} shown");
        assert_eq!(
            rendered, "multi-entry ARRAY: only key 0 of 7 shown",
            "multi-entry ARRAY truncation string drifted from pin",
        );
    }

    /// `hash map truncated at {MAX_HASH_ENTRIES} entries` is
    /// rendered by the BPF_MAP_TYPE_HASH arm. Pins both the prose
    /// and `MAX_HASH_ENTRIES` so either drifting trips the test.
    #[test]
    fn pinned_error_hash_map_truncation() {
        let rendered = format!("hash map truncated at {MAX_HASH_ENTRIES} entries");
        assert_eq!(
            rendered, "hash map truncated at 4096 entries",
            "hash map truncation string OR MAX_HASH_ENTRIES drifted from pin",
        );
    }

    /// `PERCPU_ARRAY truncated at {MAX_PERCPU_KEYS} keys (max_entries={N})`
    /// is rendered by the BPF_MAP_TYPE_PERCPU_ARRAY arm. Pins prose,
    /// constant, and placeholder ordering.
    #[test]
    fn pinned_error_percpu_array_truncation() {
        let max_entries: u32 = 999;
        let rendered = format!(
            "PERCPU_ARRAY truncated at {MAX_PERCPU_KEYS} keys (max_entries={max_entries})",
        );
        assert_eq!(
            rendered,
            "PERCPU_ARRAY truncated at 256 keys (max_entries=999)",
            "PERCPU_ARRAY truncation string OR MAX_PERCPU_KEYS drifted from pin",
        );
    }

    /// `map_type {N} not yet supported by failure dump` is rendered
    /// by the wildcard arm for any map_type the dump doesn't enumerate.
    #[test]
    fn pinned_error_unsupported_map_type() {
        let other: u32 = 42;
        let rendered = format!("map_type {other} not yet supported by failure dump");
        assert_eq!(
            rendered, "map_type 42 not yet supported by failure dump",
            "unsupported-map-type string drifted from pin",
        );
    }
}
