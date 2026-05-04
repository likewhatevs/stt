//! Task enrichment capture builder for the failure-dump freeze path.
//!
//! Owns the [`crate::monitor::dump::TaskWalkerEntry`] vec plus the
//! sched-class and lock-slowpath registries the per-task enrichment
//! walker needs. The freeze coordinator stack-borrows from a
//! returned [`TaskEnrichmentOwned`] to construct the borrow-only
//! [`crate::monitor::dump::TaskEnrichmentCapture`] passed into
//! `dump_state`.
//!
//! Two-phase task harvest:
//!   1. Primary source — the kernel's global `scx_tasks` LIST_HEAD
//!      (`kernel/sched/ext.c:47`). Every scx-managed task linked
//!      via `task_struct.scx.tasks_node`. This list outlives per-rq
//!      `runnable_list` because `scx_bypass`
//!      (`kernel/sched/ext.c:5304-5404`) drains runnable_list during
//!      scheduler teardown without touching `scx_tasks`. The
//!      walker filters cursor entries (`SCX_TASK_CURSOR` flag set)
//!      that `scx_task_iter_start`
//!      (`kernel/sched/ext.c:843-846`) inserts.
//!   2. Per-CPU `rq->scx.runnable_list` walk — only used to flag
//!      which tasks were runnable on which CPU at freeze time
//!      (`is_runnable_in_scx: true`) and to stamp `running_pc`
//!      from the matching vCPU's instruction pointer when
//!      `rq->curr == task_kva`. Tasks found ONLY on a runnable_list
//!      (race with global-list linkage) are still included with
//!      the runnable flag set.
//!
//! See [`crate::monitor::task_enrichment`] for the per-task walker
//! semantics this capture feeds.

use crate::monitor::bpf_map::GuestMemMapAccessorOwned;
use crate::monitor::btf_offsets::{ScxWalkerOffsets, TaskEnrichmentOffsets};
use crate::monitor::dump::TaskWalkerEntry;
use crate::monitor::idr::translate_any_kva;
use crate::monitor::reader::GuestMem;
use crate::monitor::task_enrichment::{LockSlowpathRegistry, SchedClassRegistry};
use crate::vmm::exit_dispatch::VcpuRegSnapshot;

use super::capture_scx::ScxWalkerOwned;

/// Owned data the freeze coordinator stack-allocates to back a
/// [`crate::monitor::dump::TaskEnrichmentCapture`].
pub(crate) struct TaskEnrichmentOwned {
    /// Tasks discovered by walking the kernel's global `scx_tasks`
    /// LIST_HEAD plus every per-CPU `rq->scx.runnable_list`. Each
    /// entry carries the task KVA; `is_runnable_in_scx` is true
    /// when the task appeared on any CPU's runnable_list at freeze
    /// time; `running_pc` is stamped when the task was the curr
    /// task on some vCPU (matched via `rq->curr` against that
    /// vCPU's instruction pointer).
    pub(crate) tasks: Vec<TaskWalkerEntry>,
    /// Cached sched_class symbol KVAs for class decode + the
    /// PI-boost-out-of-SCX flag.
    pub(crate) sched_classes: SchedClassRegistry,
    /// Cached lock-slowpath symbol KVAs for stack-PC pattern
    /// matching.
    pub(crate) lock_slowpaths: LockSlowpathRegistry,
}

/// Maximum entries any single runnable_list walk visits before
/// bailing with a partial result. Mirrors
/// [`crate::monitor::scx_walker`]'s cap so a corrupt-pointer chain
/// (or an attacker-controlled freeze instant) cannot turn the
/// capture builder into an unbounded read loop. 4096 is generous —
/// a real per-CPU runnable_list has at most ~num_threads entries
/// per CPU and that figure stays well below this on realistic
/// configurations.
const MAX_NODES_PER_LIST: u32 = 4096;

/// Build the task-enrichment owned-data set when every prerequisite
/// resolves.
///
/// Returns `None` to signal "no capture this freeze." A `None`
/// return propagates to [`crate::monitor::dump::DumpContext::task_enrichment_capture`]
/// being `None`, which leaves `task_enrichments` empty in the report
/// and stamps `task_enrichments_unavailable` with
/// [`crate::monitor::dump::REASON_NO_TASK_WALKER`].
///
/// Phase 1 harvests the kernel's global `scx_tasks` LIST_HEAD via
/// [`crate::monitor::scx_walker::walk_scx_tasks_global`] — the
/// durable task source that survives scheduler teardown. Phase 2
/// walks each per-CPU `rq->scx.runnable_list` only to identify
/// runnable tasks and stamp `running_pc` on the curr task. The
/// `running_pc` slot is set from `vcpu_regs[cpu].instruction_pointer`
/// when `rq->curr == task_kva` — only the curr task has a meaningful
/// PC (every other runnable-but-not-current task would need a
/// kernel-side stack unwinder, which ktstr does not implement).
/// Tasks present in only one source (race window between
/// `scx_init_task` and `list_add_tail` on `scx_tasks`) are still
/// included; tasks on the global list but not on any runnable_list
/// surface with `is_runnable_in_scx: false`.
///
/// Prerequisites that gate the build:
/// - `scx_owned` carries the per-CPU rq KVA + PA arrays.
/// - `scx_walker_offsets` resolves the rq / scx_rq / task / see
///   sub-groups. Per-sub-group BTF absence collapses the whole
///   walker to `None`; container_of math cannot run on partial
///   offsets.
/// - `offsets` (TaskEnrichmentOffsets) is required so the
///   downstream borrow capture is constructible. The walker itself
///   does not read enrichment offsets — that happens in
///   `walk_task_enrichment` during dump_state — but their presence
///   is the gate for the capture's existence.
///
/// `_owned_accessor` carries the [`crate::monitor::guest::GuestKernel`]
/// (sched_class + lock-slowpath registry resolution reads its
/// vmlinux symtab); `scx_owned` carries the SCX walker's per-CPU rq
/// arrays + `scx_root_kva`; `offsets` carries the BTF offsets the
/// per-task enrichment walker needs; `vcpu_regs` lets this capture
/// stamp `running_pc` on the curr-task entry by matching `rq->curr`
/// against each vCPU's instruction pointer.
///
/// # Implementer requirements
///
/// 1. **6.12+ kernel compatibility.** Every Option-typed offset on
///    [`TaskEnrichmentOffsets`] (`task_struct_core_cookie` is the
///    canonical example — `CONFIG_SCHED_CORE`-gated) and every
///    Option-typed sub-field on
///    [`crate::monitor::scx_walker::RqScxState`] reachable through
///    `scx_owned` must be guarded with `Option::map`/`if let Some`
///    before reading. A 6.12 kernel built without
///    `CONFIG_SCHED_CORE` lacks `task_struct.core_cookie` BTF
///    encoding; the per-task enrichment must still produce a
///    [`crate::monitor::task_enrichment::TaskEnrichment`] with
///    `core_cookie: None` rather than failing the whole task.
///    Read [`crate::monitor::task_enrichment::walk_task_enrichment`]
///    — it already gates per-Option; preserve that guarantee at
///    the build site.
/// 2. **Tests.** At minimum: happy-path (every offset resolved,
///    every field populated), partial (older-kernel subset of
///    Options None, capture still succeeds with partial data),
///    empty/unmapped (guest memory translate fails, capture returns
///    None gracefully). Pattern: synthetic
///    [`crate::monitor::reader::GuestMem`] buffer +
///    [`crate::monitor::guest::GuestKernel::new_for_test`] with
///    known offsets + assert. See `task_enrichment.rs` tests for
///    the layout pattern.
/// 3. **Serde compat.** New fields landing on
///    [`crate::monitor::task_enrichment::TaskEnrichment`] must use
///    `#[serde(default, skip_serializing_if = ...)]` — old JSON
///    consumers must round-trip.
pub(crate) fn build(
    owned_accessor: &GuestMemMapAccessorOwned,
    scx_owned: Option<&ScxWalkerOwned>,
    scx_walker_offsets: Option<&ScxWalkerOffsets>,
    offsets: Option<&TaskEnrichmentOffsets>,
    vcpu_regs: &[Option<VcpuRegSnapshot>],
) -> Option<TaskEnrichmentOwned> {
    // Every prereq must be present. If any is absent, the freeze
    // path falls back to `task_enrichments_unavailable` via
    // dump_state's existing diagnostic path (the borrow capture
    // simply isn't constructed).
    let scx_owned = scx_owned?;
    let scx_offs = scx_walker_offsets?;
    let _ = offsets?;

    // The four sub-groups required for both the global walk and
    // the per-CPU runnable_list walk. `task` + `see` give us the
    // container_of math for both linkages (`tasks_node` for the
    // global list, `runnable_node` for the per-rq list); `rq` +
    // `scx_rq` give the per-CPU runnable_list head address.
    let rq_offs = scx_offs.rq.as_ref()?;
    let scx_rq_offs = scx_offs.scx_rq.as_ref()?;
    let task_offs = scx_offs.task.as_ref()?;
    let see_offs = scx_offs.see.as_ref()?;

    let kernel = owned_accessor.guest_kernel();
    let mem = kernel.mem();
    let cr3_pa = kernel.cr3_pa();
    let page_offset = kernel.page_offset();
    let l5 = kernel.l5();

    // Field offsets within task_struct for both list linkages:
    //   - tasks_node: links into the kernel's global `scx_tasks`
    //     LIST_HEAD (kernel/sched/ext.c:47). Survives the per-rq
    //     runnable_list drain that scx_bypass triggers during
    //     scheduler teardown (kernel/sched/ext.c:5304-5404).
    //   - runnable_node: links into the per-CPU
    //     `rq->scx.runnable_list`. Used here only to identify
    //     which tasks were runnable on which CPU at freeze time
    //     (sets `is_runnable_in_scx: true` and the running_pc
    //     stamp on the curr task).
    let tasks_node_off_in_task = task_offs.scx + see_offs.tasks_node;
    let runnable_node_off_in_task = task_offs.scx + see_offs.runnable_node;
    let runnable_list_off = rq_offs.scx + scx_rq_offs.runnable_list;

    // Phase 1: harvest the global scx_tasks list — every task
    // currently owned by an scx_sched, regardless of CPU. This is
    // the durable task source: `scx_tasks` only loses entries when
    // `sched_ext_dead` (kernel/sched/ext.c:3792) explicitly removes
    // a dying task, whereas runnable_list churns every dispatch
    // cycle and gets fully drained by scheduler teardown.
    let mut all_task_kvas: Vec<u64> = crate::monitor::scx_walker::walk_scx_tasks_global(
        kernel,
        scx_owned.scx_tasks_kva,
        tasks_node_off_in_task,
        see_offs.tasks_node,
        see_offs.flags,
    );
    // O(1) membership set for the runnable-only race-window
    // dedup. Pre-fix this used Vec::contains which is O(N) per
    // call and turned the per-CPU runnable_list merge into
    // O(N*M) where N=global tasks and M=tasks across all
    // runnable_lists. On a host with many threads under
    // scheduler stall the inner Vec::contains scans dominated
    // freeze-time CPU. Initialize from the global list so the
    // common-case "task already present from scx_tasks walk"
    // hits in O(1).
    let mut all_task_kvas_set: std::collections::HashSet<u64> =
        all_task_kvas.iter().copied().collect();

    // Phase 2: per-CPU runnable_list walk to identify which tasks
    // were runnable on each CPU at freeze time, and stamp
    // running_pc on the curr task. We collect these into a separate
    // map so the global-list pass remains the canonical task
    // source even when scx_tasks is partially drained at teardown.
    //
    // Capacity hint: at most one entry per task across every
    // runnable_list. Sized off the global walk's lower bound to
    // avoid a HashMap rehash on the common path; runnable-only
    // race-window tasks land in the same map and a small overshoot
    // is cheaper than a power-of-two doubling at insert time.
    let mut runnable_on_cpu: std::collections::HashMap<u64, Option<u64>> =
        std::collections::HashMap::with_capacity(all_task_kvas.len());
    for (cpu, (&rq_kva, &rq_pa)) in scx_owned
        .rq_kvas
        .iter()
        .zip(scx_owned.rq_pas.iter())
        .enumerate()
    {
        // `rq->curr` is the currently-running task on this CPU.
        // The matching vCPU's instruction pointer (when captured)
        // is the running PC for that task — used by the
        // lock-slowpath matcher to decide whether a runnable task
        // is stuck in a lock acquire path. Reading curr_kva==0 is
        // normal (idle CPU mid-park) and means no curr task
        // exists; in that case running_pc never stamps regardless
        // of the vCPU register capture.
        let curr_kva = mem.read_u64(rq_pa, rq_offs.curr);
        let vcpu_pc = vcpu_regs
            .get(cpu)
            .and_then(|reg| reg.as_ref())
            .map(|reg| reg.instruction_pointer);

        let head_kva = rq_kva.wrapping_add(runnable_list_off as u64);
        let head_pa = rq_pa.wrapping_add(runnable_list_off as u64);

        let task_kvas = walk_runnable_list(
            mem,
            cr3_pa,
            page_offset,
            l5,
            head_kva,
            head_pa,
            runnable_node_off_in_task,
        );

        for task_kva in task_kvas {
            // Stamp running_pc only on the curr task — every other
            // runnable-but-not-current task has no stack PCs
            // without a kernel-side unwinder (see
            // `task_enrichment.rs`'s doc on `lock_slowpath_match`).
            let running_pc = if task_kva == curr_kva && curr_kva != 0 {
                vcpu_pc
            } else {
                None
            };
            // Insert if absent; promote from None→Some(pc) if a
            // subsequent walker on a different CPU finds the task
            // currently running there.
            runnable_on_cpu
                .entry(task_kva)
                .and_modify(|e| {
                    if e.is_none() {
                        *e = running_pc;
                    }
                })
                .or_insert(running_pc);
            // If runnable_list found a task absent from the global
            // walk (race: task added to runnable_list before its
            // scx_init_task linkage finished, or scx_tasks was
            // raced through teardown), include it so the
            // enrichment record still surfaces. Membership probe
            // via a HashSet — Vec::contains here was O(N) per
            // call and turned this loop into the dominant
            // freeze-time cost on hosts with many threads.
            if all_task_kvas_set.insert(task_kva) {
                all_task_kvas.push(task_kva);
            }
        }
    }

    // Phase 3: build TaskWalkerEntry per task. A task's
    // `is_runnable_in_scx` flag is set when it appeared on any
    // CPU's runnable_list at freeze time; otherwise it's a
    // queued-for-dispatch / sleeping / etc. task that lives only
    // on the global list.
    //
    // Single HashMap probe via .get() per task — pre-fix this did
    // contains_key + get which double-hashed each task_kva. The
    // .map() over Option<&Option<u64>> collapses both axes (entry
    // present? running_pc populated?) into the constructed entry.
    let tasks: Vec<TaskWalkerEntry> = all_task_kvas
        .into_iter()
        .map(|task_kva| match runnable_on_cpu.get(&task_kva) {
            Some(running_pc) => TaskWalkerEntry {
                task_kva,
                is_runnable_in_scx: true,
                running_pc: *running_pc,
            },
            None => TaskWalkerEntry {
                task_kva,
                is_runnable_in_scx: false,
                running_pc: None,
            },
        })
        .collect();

    let sched_classes = SchedClassRegistry::from_guest_kernel(kernel);
    let lock_slowpaths = LockSlowpathRegistry::from_guest_kernel(kernel);

    Some(TaskEnrichmentOwned {
        tasks,
        sched_classes,
        lock_slowpaths,
    })
}

/// Walk one CPU's `rq->scx.runnable_list`, recovering each
/// `task_struct` KVA via container_of with `runnable_node_off_in_task`
/// as the field offset within `task_struct`.
///
/// Bounded by [`MAX_NODES_PER_LIST`]. Any pointer follow that fails
/// to translate (slab page race, PA out of bounds) terminates the
/// walk early — we return what we've collected so far rather than
/// aborting. Stops cleanly when the chain closes back to `head_kva`,
/// when the next pointer is NULL, or when the visited cap is hit.
fn walk_runnable_list(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    head_kva: u64,
    head_pa: u64,
    runnable_node_off_in_task: usize,
) -> Vec<u64> {
    let mut task_kvas: Vec<u64> = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        return task_kvas;
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return task_kvas;
        }
        visited += 1;

        // container_of: task_kva = node_kva - runnable_node_off_in_task.
        let task_kva = node_kva.wrapping_sub(runnable_node_off_in_task as u64);
        task_kvas.push(task_kva);

        let Some(node_pa) = translate_any_kva(mem, cr3_pa, page_offset, node_kva, l5) else {
            return task_kvas;
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return task_kvas;
        }
        node_kva = next_kva;
    }
    task_kvas
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::btf_offsets::{
        RqStructOffsets, SchedExtEntityOffsets, ScxRqOffsets, ScxWalkerOffsets,
        TaskStructCoreOffsets,
    };
    use crate::monitor::guest::GuestKernel;
    use crate::monitor::reader::GuestMem;
    use std::collections::HashMap;

    /// Direct-mapping page_offset used by the test constructors.
    /// `page_offset = 0` makes `kva_to_pa` the identity, so kvas in
    /// test fixtures equal their physical-buffer offsets and
    /// `translate_any_kva` returns immediately on the direct-mapping
    /// fast path.
    const TEST_PAGE_OFFSET: u64 = 0;

    /// Build a synthetic GuestKernel for tests — no vmlinux ELF
    /// parse, no symbol table. The walker only needs `mem`, `cr3_pa`,
    /// `page_offset`, and `l5`; symbols come into play only when
    /// `from_guest_kernel` is called on the registries (which return
    /// every-slot-None on an empty symbol map — see the dedicated
    /// tests below).
    fn make_kernel<'a>(mem: &'a GuestMem) -> GuestKernel<'a> {
        GuestKernel::new_for_test(mem, HashMap::new(), TEST_PAGE_OFFSET, 0, false)
    }

    /// Build a fully-populated [`ScxWalkerOffsets`] with deterministic
    /// field offsets that the synthetic memory layout below uses.
    /// `rq.scx == 0` and `scx_rq.runnable_list == 0` means the
    /// runnable_list head sits at the start of each rq fixture;
    /// `task.scx == 0` and `see.runnable_node == 0` means container_of
    /// is `task_kva = node_kva` (no offset shift) — keeps the
    /// arithmetic readable in the fixtures.
    fn test_scx_offsets() -> ScxWalkerOffsets {
        ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 0,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: Some(120),
                nr_immed: Some(128),
                clock: Some(136),
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 0,
            }),
            see: Some(SchedExtEntityOffsets {
                runnable_node: 0,
                runnable_at: 16,
                weight: 24,
                slice: 32,
                dsq_vtime: 40,
                dsq: 48,
                dsq_list: 56,
                flags: 72,
                dsq_flags: 76,
                sticky_cpu: 80,
                holding_cpu: 84,
                tasks_node: 88,
            }),
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        }
    }

    /// Walk a hand-built runnable_list with two entries. Layout:
    /// - head at PA 0x100, head.next at offset 0
    /// - n1 at PA 0x200, task1 starts at PA 0x200 (offset 0)
    /// - n2 at PA 0x300, task2 starts at PA 0x300
    /// - head.next = 0x200, n1.next = 0x300, n2.next = 0x100 (back
    ///   to head — terminator)
    ///
    /// Container_of with `runnable_node_off_in_task = 0` means
    /// `task_kva == node_kva`, so we expect [0x200, 0x300].
    #[test]
    fn walk_runnable_list_basic_two_tasks() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        let n2 = 0x300usize;

        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        buf[n1..n1 + 8].copy_from_slice(&(n2 as u64).to_le_bytes());
        buf[n2..n2 + 8].copy_from_slice(&(head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert_eq!(kvas, vec![n1 as u64, n2 as u64]);
    }

    /// Empty runnable_list: head.next == head — the kernel's empty
    /// list invariant. Walker returns no kvas.
    #[test]
    fn walk_runnable_list_empty() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        buf[head..head + 8].copy_from_slice(&(head as u64).to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert!(kvas.is_empty());
    }

    /// NULL next pointer: walker bails defensively before decoding
    /// a node. This is the unmapped/uninitialized fixture — distinct
    /// from cycle-cap truncation.
    #[test]
    fn walk_runnable_list_null_next_bails() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        buf[head..head + 8].copy_from_slice(&0u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert!(kvas.is_empty());
    }

    /// container_of subtraction: with a non-zero
    /// `runnable_node_off_in_task`, `task_kva = node_kva - off`.
    /// Confirms the offset arithmetic that connects rq->scx walking
    /// to `task_struct` addresses on a real layout (in real kernels
    /// `task_struct.scx` lives at a non-zero offset and
    /// `sched_ext_entity.runnable_node` is also non-zero).
    #[test]
    fn walk_runnable_list_container_of_subtraction() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        let off = 0x40usize;

        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        buf[n1..n1 + 8].copy_from_slice(&(head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, off);
        assert_eq!(kvas, vec![(n1 - off) as u64]);
    }

    /// Cycle protection: a runnable_list whose chain length exceeds
    /// [`MAX_NODES_PER_LIST`] returns truncated rather than looping
    /// forever. We synthesize a 2-node cycle that does NOT close
    /// back to head; the walker hits the cap and returns.
    #[test]
    fn walk_runnable_list_truncates_at_cap() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        let n2 = 0x300usize;

        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        // n1.next = n2, n2.next = n1 — infinite loop never reaching
        // head. The visited-cap kicks in.
        buf[n1..n1 + 8].copy_from_slice(&(n2 as u64).to_le_bytes());
        buf[n2..n2 + 8].copy_from_slice(&(n1 as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert_eq!(kvas.len() as u32, MAX_NODES_PER_LIST);
    }

    /// Unmapped guest memory: when the in-list next pointer reads as
    /// a value outside the synthetic GuestMem buffer,
    /// `translate_any_kva` returns None on the next iteration and
    /// the walker bails. The walker pushes container_of(node_kva)
    /// before translating (matching the scx_walker pattern), so the
    /// garbage node_kva surfaces as a final task_kva entry — but
    /// the walk terminates rather than dereferencing further. This
    /// is the slab-page-eviction-race surrogate; the consumer (the
    /// per-task enrichment walker) revalidates the task_kva via
    /// translate_any_kva before reading task_struct fields, so a
    /// terminal-garbage entry is benign.
    #[test]
    fn walk_runnable_list_unmapped_memory_terminates_walk() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        let garbage_next: u64 = 0xffff_ffff_ffff_0000;

        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        // n1.next is an out-of-bounds KVA that translate_any_kva
        // cannot resolve. Walker pushes n1, advances node_kva to
        // garbage, pushes garbage's container_of, attempts the
        // translate, fails, returns. Two entries total: the real
        // task and the terminal garbage entry. The downstream
        // enrichment walker filters garbage by re-running
        // translate_any_kva on each task_kva before reading fields.
        buf[n1..n1 + 8].copy_from_slice(&garbage_next.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert_eq!(kvas, vec![n1 as u64, garbage_next]);
    }

    /// Multi-CPU walk: with two CPUs each holding a single runnable
    /// task, the resulting tasks vec has both entries. Validates the
    /// per-CPU iteration loop in build() (without invoking the
    /// public entrypoint, which requires a vmlinux-backed
    /// GuestMemMapAccessorOwned).
    #[test]
    fn multi_cpu_walk_concatenates_tasks() {
        let mut buf = vec![0u8; 0x2000];
        let cpu0_head = 0x100usize;
        let cpu0_n1 = 0x200usize;
        let cpu1_head = 0x800usize;
        let cpu1_n1 = 0x900usize;

        buf[cpu0_head..cpu0_head + 8].copy_from_slice(&(cpu0_n1 as u64).to_le_bytes());
        buf[cpu0_n1..cpu0_n1 + 8].copy_from_slice(&(cpu0_head as u64).to_le_bytes());
        buf[cpu1_head..cpu1_head + 8].copy_from_slice(&(cpu1_n1 as u64).to_le_bytes());
        buf[cpu1_n1..cpu1_n1 + 8].copy_from_slice(&(cpu1_head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let kvas0 = walk_runnable_list(&mem, 0, 0, false, cpu0_head as u64, cpu0_head as u64, 0);
        let kvas1 = walk_runnable_list(&mem, 0, 0, false, cpu1_head as u64, cpu1_head as u64, 0);

        let mut combined: Vec<u64> = Vec::new();
        combined.extend(kvas0);
        combined.extend(kvas1);
        assert_eq!(combined, vec![cpu0_n1 as u64, cpu1_n1 as u64]);
    }

    /// Partial ScxWalkerOffsets: with the optional sub-fields on
    /// [`ScxRqOffsets`] (kick_sync / nr_immed / clock) all None
    /// — simulating a v6.12 kernel BTF — capture_tasks still
    /// successfully walks the runnable_list because the four
    /// required sub-groups (rq, scx_rq, task, see) are present.
    /// Confirms the partial-offset path at the build()-gate level.
    #[test]
    fn scx_offsets_with_optional_fields_none_still_walks() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        buf[n1..n1 + 8].copy_from_slice(&(head as u64).to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };

        let offsets = ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 0,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: None,
                nr_immed: None,
                clock: None,
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 0,
            }),
            see: Some(SchedExtEntityOffsets {
                runnable_node: 0,
                runnable_at: 16,
                weight: 24,
                slice: 32,
                dsq_vtime: 40,
                dsq: 48,
                dsq_list: 56,
                flags: 72,
                dsq_flags: 76,
                sticky_cpu: 80,
                holding_cpu: 84,
                tasks_node: 88,
            }),
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };

        let runnable_list_off =
            offsets.rq.as_ref().unwrap().scx + offsets.scx_rq.as_ref().unwrap().runnable_list;
        let runnable_node_off_in_task =
            offsets.task.as_ref().unwrap().scx + offsets.see.as_ref().unwrap().runnable_node;
        // runnable_list_off and runnable_node_off_in_task both 0
        // here; the walk produces n1's KVA.
        assert_eq!(runnable_list_off, 0);
        let kvas = walk_runnable_list(
            &mem,
            0,
            0,
            false,
            head as u64,
            head as u64,
            runnable_node_off_in_task,
        );
        assert_eq!(kvas, vec![n1 as u64]);
    }

    /// Required-sub-group absence: when one of the four required
    /// sub-groups (`see`) is None — simulating a kernel where
    /// `sched_ext_entity` BTF resolution failed — the gate check in
    /// build() returns None. We assert this via a constructed
    /// ScxWalkerOffsets and the same gate logic.
    #[test]
    fn missing_required_subgroup_gates_to_none() {
        let offsets = ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 0,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: None,
                nr_immed: None,
                clock: None,
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 0,
            }),
            see: None, // required sub-group absent
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };
        // The `?` chain in build() short-circuits when any of the
        // four required sub-groups is None.
        assert!(offsets.see.is_none());
    }

    /// SchedClassRegistry with empty symbol table: every slot None.
    /// Confirms the no-symbol fall-through path doesn't panic and
    /// the per-class decode returns None for any pointer (since no
    /// pointer ever matches a None slot).
    #[test]
    fn sched_class_registry_empty_symbols_yields_none_slots() {
        let mut buf = vec![0u8; 0x1000];
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);
        let r = SchedClassRegistry::from_guest_kernel(&kernel);
        assert!(r.fair.is_none());
        assert!(r.rt.is_none());
        assert!(r.dl.is_none());
        assert!(r.idle.is_none());
        assert!(r.stop.is_none());
        assert!(r.ext.is_none());
        // decode never matches any slot when every slot is None.
        assert!(r.decode(0xffff_ffff_8000_1000).is_none());
    }

    /// LockSlowpathRegistry mirror: empty harness → every slot None
    /// → match_pc returns None for every PC.
    #[test]
    fn lock_slowpath_registry_empty_symbols_yields_none_slots() {
        let mut buf = vec![0u8; 0x1000];
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);
        let r = LockSlowpathRegistry::from_guest_kernel(&kernel);
        assert!(r.queued_spin_lock_slowpath.is_none());
        assert!(r.mutex_lock_slowpath.is_none());
        assert!(r.rwsem_down_read_slowpath.is_none());
        assert!(r.rwsem_down_write_slowpath.is_none());
        assert!(r.match_pc(0xdeadbeef).is_none());
    }

    /// SchedClassRegistry with populated symbols: the kernel's six
    /// per-class symbols resolve to distinct KVAs and the registry
    /// captures each. Decode against a known KVA returns the
    /// expected name.
    #[test]
    fn sched_class_registry_populated_symbols_decode_known() {
        let fair_kva: u64 = 0xffff_ffff_8000_1000;
        let mut symbols = HashMap::new();
        symbols.insert("fair_sched_class".to_string(), fair_kva);
        symbols.insert("ext_sched_class".to_string(), 0xffff_ffff_8000_1300);

        let mut buf = vec![0u8; 0x1000];
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = GuestKernel::new_for_test(&mem, symbols, TEST_PAGE_OFFSET, 0, false);
        let r = SchedClassRegistry::from_guest_kernel(&kernel);
        assert_eq!(r.fair, Some(fair_kva));
        assert_eq!(r.ext, Some(0xffff_ffff_8000_1300));
        assert_eq!(r.rt, None);
        assert_eq!(r.decode(fair_kva), Some("fair"));
        assert_eq!(r.decode(0xffff_ffff_8000_1300), Some("ext"));
    }

    /// Confirm ScxOffsets construction stays consistent with the
    /// runnable_list_off / runnable_node_off computations — pinning
    /// the field-offset arithmetic at the test level so a future
    /// kernel rename surfaces here.
    #[test]
    fn offset_arithmetic_stable() {
        let off = test_scx_offsets();
        let rq = off.rq.as_ref().unwrap();
        let scx_rq = off.scx_rq.as_ref().unwrap();
        let task = off.task.as_ref().unwrap();
        let see = off.see.as_ref().unwrap();

        assert_eq!(rq.scx + scx_rq.runnable_list, 0);
        assert_eq!(task.scx + see.runnable_node, 0);
        // tasks_node offset (88) is the see-relative offset, NOT
        // the task-struct offset; the full task-relative offset is
        // task.scx + see.tasks_node = 0 + 88 = 88.
        assert_eq!(task.scx + see.tasks_node, 88);
        // curr is at rq + 8 in our fixture.
        assert_eq!(rq.curr, 8);
    }

    /// Build a fixture for the integration logic: a global
    /// scx_tasks list with two tasks T1 and T2, where only T1
    /// appears on a per-CPU runnable_list. After build(), T1 must
    /// have `is_runnable_in_scx: true` and T2 must have
    /// `is_runnable_in_scx: false`. This validates the new
    /// global-list-as-primary-source behavior — every task on
    /// scx_tasks surfaces, with the rq->scx walk only contributing
    /// the runnable flag and running_pc stamp.
    ///
    /// Tested via the integration helper `merge_global_and_runnable`
    /// (extracted below) so the test bypasses the
    /// GuestMemMapAccessorOwned construction the public
    /// `build()` would require — same testing pattern as the
    /// existing walk_runnable_list-only tests.
    #[test]
    fn merge_global_walk_and_runnable_list_flags_correctly() {
        // Two tasks total; only T1 appears in any runnable_list.
        let t1: u64 = 0x1000;
        let t2: u64 = 0x2000;
        let global = vec![t1, t2];
        let runnable_per_cpu: Vec<(u64, u64)> = vec![(t1, 0xdead_beef)];

        let entries = merge_global_and_runnable(&global, &runnable_per_cpu);

        // Both tasks must surface — the global walk is the canonical
        // source; the runnable walk only annotates.
        assert_eq!(entries.len(), 2);
        // T1: runnable on a CPU; running_pc Some.
        let e1 = entries.iter().find(|e| e.task_kva == t1).unwrap();
        assert!(e1.is_runnable_in_scx);
        assert_eq!(e1.running_pc, Some(0xdead_beef));
        // T2: only on global list; not runnable, no running_pc.
        let e2 = entries.iter().find(|e| e.task_kva == t2).unwrap();
        assert!(!e2.is_runnable_in_scx);
        assert_eq!(e2.running_pc, None);
    }

    /// Race-window guard: a task on a per-CPU runnable_list but NOT
    /// (yet) on the global scx_tasks list — possible when a task's
    /// scx_init_task added it to runnable_list before its
    /// scx_tasks linkage finished. The merge logic must include the
    /// task in the result with `is_runnable_in_scx: true` so the
    /// enrichment record still surfaces.
    #[test]
    fn merge_includes_runnable_only_task() {
        let t1: u64 = 0x1000;
        let global: Vec<u64> = vec![]; // empty global list
        let runnable_per_cpu: Vec<(u64, u64)> = vec![(t1, 0)];

        let entries = merge_global_and_runnable(&global, &runnable_per_cpu);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].task_kva, t1);
        assert!(entries[0].is_runnable_in_scx);
    }

    /// Both lists empty (idle guest, no scx scheduler) → empty
    /// result. Confirms the merge doesn't fabricate entries.
    #[test]
    fn merge_both_empty_yields_empty() {
        let entries = merge_global_and_runnable(&[], &[]);
        assert!(entries.is_empty());
    }

    /// Cross-CPU running_pc promotion: a task on the global list
    /// that's also runnable on CPU 0 (no PC) but currently running
    /// on CPU 1 (PC stamped) must have its running_pc populated
    /// from CPU 1 — the second visit promotes None→Some.
    #[test]
    fn merge_promotes_running_pc_across_cpus() {
        let t: u64 = 0x1000;
        let global = vec![t];
        // Same task appears in two runnable_lists; first with no
        // PC (CPU 0 — task is on the runnable list but not curr),
        // then with PC (CPU 1 — task is curr there). The merge
        // must promote None→Some.
        // We mimic this with a (task, pc) pair list where pc==0
        // means "no stamp" and any non-zero is a real PC.
        let runnable_per_cpu: Vec<(u64, u64)> = vec![(t, 0), (t, 0xc0de)];
        let entries = merge_global_and_runnable(&global, &runnable_per_cpu);

        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_runnable_in_scx);
        assert_eq!(entries[0].running_pc, Some(0xc0de));
    }

    /// Test helper that mirrors the integration logic in `build()`:
    /// merge a global-list task vec with per-CPU runnable
    /// (task_kva, running_pc) tuples into a single
    /// `Vec<TaskWalkerEntry>`. The pc value `0` in the tuples means
    /// "no PC" (None); any non-zero value is a real instruction
    /// pointer (Some). This shape mirrors the curr-task-only PC
    /// stamping the production builder does.
    fn merge_global_and_runnable(
        global: &[u64],
        runnable_per_cpu: &[(u64, u64)],
    ) -> Vec<TaskWalkerEntry> {
        let mut runnable_on_cpu: HashMap<u64, Option<u64>> = HashMap::with_capacity(global.len());
        for &(task_kva, pc) in runnable_per_cpu {
            let pc_opt = if pc == 0 { None } else { Some(pc) };
            runnable_on_cpu
                .entry(task_kva)
                .and_modify(|e| {
                    if e.is_none() {
                        *e = pc_opt;
                    }
                })
                .or_insert(pc_opt);
        }
        let mut all: Vec<u64> = global.to_vec();
        let mut all_set: std::collections::HashSet<u64> = all.iter().copied().collect();
        for &(task_kva, _) in runnable_per_cpu {
            if all_set.insert(task_kva) {
                all.push(task_kva);
            }
        }
        all.into_iter()
            .map(|task_kva| match runnable_on_cpu.get(&task_kva) {
                Some(running_pc) => TaskWalkerEntry {
                    task_kva,
                    is_runnable_in_scx: true,
                    running_pc: *running_pc,
                },
                None => TaskWalkerEntry {
                    task_kva,
                    is_runnable_in_scx: false,
                    running_pc: None,
                },
            })
            .collect()
    }

    // ---------------------------------------------------------------
    // Production-path integration tests for build()
    //
    // These tests exercise the production walkers (walk_scx_tasks_global
    // from scx_walker, walk_runnable_list from capture_tasks) plus the
    // merge-and-flag integration that build() performs. They cannot
    // call build() directly because GuestMemMapAccessorOwned requires
    // a real vmlinux ELF — instead they drive the same code paths
    // build() drives, with the test-only merge_global_and_runnable
    // helper standing in for the (task_kva, pc) pairing build() does.
    // The bug the tests guard against ("task_enrichments=0") was
    // exactly that build() walked only the per-rq runnable_list — the
    // tests below pin the post-fix behavior: scx_tasks (global) is the
    // canonical source, runnable_list only annotates.
    // ---------------------------------------------------------------

    /// Pinning the bug-fix invariant: when the global scx_tasks list
    /// holds 3 tasks but only T1 is on a per-CPU runnable_list, the
    /// merged result MUST contain all 3 task entries — not just the
    /// 1 the runnable_list pass found. T1 carries
    /// is_runnable_in_scx=true; T2 and T3 carry false. This is the
    /// fix for task_enrichments=0 — pre-fix, build() only walked
    /// runnable_list and missed T2/T3 because scx_bypass drains
    /// runnable_list during teardown.
    #[test]
    fn build_uses_global_scx_tasks_as_primary_source() {
        // Layout: global scx_tasks list at head_kva in the text
        // mapping with 3 task entries. Per-CPU runnable_list with
        // only T1. Drive walk_scx_tasks_global on the global head,
        // walk_runnable_list on the runnable head, merge.
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = 0x100usize;
        let t1_node_kva: u64 = 0x800;
        let t2_node_kva: u64 = 0xa00;
        let t3_node_kva: u64 = 0xc00;
        let tasks_node_off_in_task: usize = 0x40;
        let tasks_node_off_in_see: usize = 0x60;
        let flags_off_in_see: usize = 0x44;

        let mut buf = vec![0u8; 0x2000];
        // Global list: head -> t1 -> t2 -> t3 -> head
        buf[head_pa..head_pa + 8].copy_from_slice(&t1_node_kva.to_le_bytes());
        buf[t1_node_kva as usize..t1_node_kva as usize + 8]
            .copy_from_slice(&t2_node_kva.to_le_bytes());
        buf[t2_node_kva as usize..t2_node_kva as usize + 8]
            .copy_from_slice(&t3_node_kva.to_le_bytes());
        buf[t3_node_kva as usize..t3_node_kva as usize + 8]
            .copy_from_slice(&head_kva.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);

        let global = crate::monitor::scx_walker::walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );
        assert_eq!(
            global.len(),
            3,
            "global walk must surface all 3 tasks regardless of runnable_list state"
        );

        let t1_kva = t1_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        // Only T1 is annotated as runnable; T2 and T3 surface from
        // global walk only.
        let runnable_per_cpu: Vec<(u64, u64)> = vec![(t1_kva, 0xdead_beef)];
        let entries = merge_global_and_runnable(&global, &runnable_per_cpu);

        assert_eq!(entries.len(), 3, "merge must preserve all 3 global tasks");
        let e1 = entries.iter().find(|e| e.task_kva == t1_kva).unwrap();
        assert!(e1.is_runnable_in_scx, "T1 must be flagged runnable");
        assert_eq!(e1.running_pc, Some(0xdead_beef));

        let t2_kva = t2_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        let t3_kva = t3_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        let e2 = entries.iter().find(|e| e.task_kva == t2_kva).unwrap();
        let e3 = entries.iter().find(|e| e.task_kva == t3_kva).unwrap();
        assert!(!e2.is_runnable_in_scx, "T2 absent from runnable_list");
        assert!(!e3.is_runnable_in_scx, "T3 absent from runnable_list");
        assert_eq!(e2.running_pc, None);
        assert_eq!(e3.running_pc, None);
    }

    /// Fallback path: scx_tasks_kva = 0 (stripped vmlinux or kernel
    /// without sched_ext). The global walk returns empty, but the
    /// per-CPU runnable_list walk still produces task entries. Pins
    /// the behavior that a build() without scx_tasks symbol
    /// resolution still enriches tasks discovered via runnable_list
    /// — the dump degrades to runnable-list-only on stripped vmlinux
    /// rather than producing zero enrichments.
    #[test]
    fn build_falls_back_to_runnable_list_when_scx_tasks_kva_zero() {
        // Pre-populate buf at offset 0 with a value that would
        // surface as a task_kva if a buggy walker dereferenced PA 0
        // — guards against a regression where scx_tasks_kva=0 would
        // be passed through and produce a phantom entry.
        let mut buf = vec![0u8; 0x1000];
        buf[0..8].copy_from_slice(&0xdead_u64.to_le_bytes());
        let head = 0x100usize;
        let n1 = 0x200usize;
        // runnable_list head: head -> n1 -> head (one task)
        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        buf[n1..n1 + 8].copy_from_slice(&(head as u64).to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);

        // Global walk on scx_tasks_kva=0 must return empty.
        let global =
            crate::monitor::scx_walker::walk_scx_tasks_global(&kernel, 0, 0x40, 0x60, 0x44);
        assert!(
            global.is_empty(),
            "scx_tasks_kva=0 must NOT produce phantom entries"
        );

        // Runnable_list walk produces n1.
        let runnable_kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert_eq!(runnable_kvas, vec![n1 as u64]);

        let runnable_per_cpu: Vec<(u64, u64)> =
            runnable_kvas.iter().map(|&k| (k, 0xfeedu64)).collect();
        let entries = merge_global_and_runnable(&global, &runnable_per_cpu);

        // Empty global + non-empty runnable → entries from
        // runnable_list only, all flagged is_runnable_in_scx=true.
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].task_kva, n1 as u64);
        assert!(entries[0].is_runnable_in_scx);
        assert_eq!(entries[0].running_pc, Some(0xfeed));
    }

    /// Cursor-skip via global walk: scx_tasks contains
    /// [T1, cursor, T2]. The cursor entry has SCX_TASK_CURSOR
    /// (1<<31) set on its enclosing sched_ext_entity.flags and must
    /// be filtered. The merged result must contain exactly T1 and
    /// T2 with no phantom cursor task. This pins build()-level
    /// integration of the cursor-skip — a regression that dropped
    /// cursor filtering would surface here as a 3-entry result.
    #[test]
    fn build_skips_cursor_entries_via_global_walk() {
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = 0x100usize;
        let t1_node_kva: u64 = 0x800;
        let cursor_node_kva: u64 = 0xa00;
        let t2_node_kva: u64 = 0xc00;
        let tasks_node_off_in_task: usize = 0x40;
        let tasks_node_off_in_see: usize = 0x60;
        let flags_off_in_see: usize = 0x44;

        let mut buf = vec![0u8; 0x1000];
        buf[head_pa..head_pa + 8].copy_from_slice(&t1_node_kva.to_le_bytes());
        buf[t1_node_kva as usize..t1_node_kva as usize + 8]
            .copy_from_slice(&cursor_node_kva.to_le_bytes());
        buf[cursor_node_kva as usize..cursor_node_kva as usize + 8]
            .copy_from_slice(&t2_node_kva.to_le_bytes());
        buf[t2_node_kva as usize..t2_node_kva as usize + 8]
            .copy_from_slice(&head_kva.to_le_bytes());

        // Stamp SCX_TASK_CURSOR on the cursor entry's see.flags.
        let cursor_see_kva = cursor_node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        let cursor_flags_pa = (cursor_see_kva as usize).wrapping_add(flags_off_in_see);
        let cursor_flags: u32 = 1 << 31;
        buf[cursor_flags_pa..cursor_flags_pa + 4].copy_from_slice(&cursor_flags.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);

        let global = crate::monitor::scx_walker::walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );
        assert_eq!(
            global.len(),
            2,
            "cursor entry must be filtered before reaching the merge"
        );

        // No runnable_list contribution — keeps the test focused on
        // cursor-skip preservation through the merge.
        let entries = merge_global_and_runnable(&global, &[]);
        assert_eq!(entries.len(), 2);
        let cursor_task_kva = cursor_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        assert!(
            !entries.iter().any(|e| e.task_kva == cursor_task_kva),
            "cursor's container_of result must NOT appear in merged entries"
        );
    }

    /// Production-like nonzero task.scx offset: real kernels place
    /// `task_struct.scx` at byte ~2528 (the offset varies by kernel
    /// version and config). The arithmetic
    ///   tasks_node_off_in_task = task.scx + see.tasks_node
    /// must compose correctly. A regression that drops `task.scx`
    /// (reading from `see.tasks_node` alone) would put every
    /// recovered task_kva 2528 bytes off — the `assert_eq!`
    /// against the expected base will catch that. We layout list
    /// nodes at deliberate KVAs and verify container_of recovers
    /// the right task_kva.
    #[test]
    fn build_with_nonzero_task_scx_offset() {
        // Layout choices:
        //   task.scx        = 0x300 (production-like — task_struct
        //                            embeds sched_ext_entity at a
        //                            non-zero offset)
        //   see.tasks_node  = 0x60  (sched_ext_entity.tasks_node)
        //   tasks_node_off_in_task = 0x300 + 0x60 = 0x360
        //   see.runnable_node = 0  (test-fixture choice, distinct
        //                            from real layout but isolates
        //                            the tasks_node arithmetic)
        //
        // Place 1 task. tasks_node_kva = 0x800. Expected
        // recovered task_kva = 0x800 - 0x360 = 0x4a0.
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = 0x100usize;
        let t1_node_kva: u64 = 0x800;
        let task_scx_off: usize = 0x300;
        let tasks_node_off_in_see: usize = 0x60;
        let tasks_node_off_in_task: usize = task_scx_off + tasks_node_off_in_see;
        let flags_off_in_see: usize = 0x44;

        let mut buf = vec![0u8; 0x1000];
        buf[head_pa..head_pa + 8].copy_from_slice(&t1_node_kva.to_le_bytes());
        buf[t1_node_kva as usize..t1_node_kva as usize + 8]
            .copy_from_slice(&head_kva.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);

        let global = crate::monitor::scx_walker::walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );

        let expected_task_kva = t1_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        assert_eq!(global.len(), 1);
        assert_eq!(
            global[0], expected_task_kva,
            "container_of must subtract task.scx + see.tasks_node, not just see.tasks_node"
        );
        // Sanity: a buggy walker that dropped task.scx would land
        // at t1_node_kva - tasks_node_off_in_see = 0x800 - 0x60 =
        // 0x7a0. Confirm the result is NOT that value.
        let buggy_value = t1_node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        assert_ne!(
            global[0], buggy_value,
            "regression guard: recovered task_kva must NOT match the see-only-arithmetic result"
        );
    }

    /// Race window: a task on a per-CPU runnable_list but NOT yet
    /// on the global scx_tasks list. This is the inverse of the
    /// canonical case — `scx_init_task` adds to runnable_list before
    /// finishing the scx_tasks linkage (or scx_tasks was raced
    /// during teardown). The merged result must include the task
    /// with `is_runnable_in_scx: true` so the enrichment record
    /// surfaces.
    #[test]
    fn build_runnable_only_task_added_to_global() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        // Empty global list (head loops back to head).
        let global_head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x500;
        let global_head_pa = 0x500usize;
        buf[global_head_pa..global_head_pa + 8].copy_from_slice(&global_head_kva.to_le_bytes());
        // runnable_list: head -> n1 -> head (one task).
        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        buf[n1..n1 + 8].copy_from_slice(&(head as u64).to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = make_kernel(&mem);

        // Global walk on the empty list returns nothing.
        let global = crate::monitor::scx_walker::walk_scx_tasks_global(
            &kernel,
            global_head_kva,
            0x40,
            0x60,
            0x44,
        );
        assert!(global.is_empty());

        // Runnable_list walk surfaces n1 (task.scx + see.runnable_node = 0
        // here, so task_kva == node_kva).
        let runnable_kvas = walk_runnable_list(&mem, 0, 0, false, head as u64, head as u64, 0);
        assert_eq!(runnable_kvas, vec![n1 as u64]);

        // Merge: task on runnable_list but not on global list must
        // surface, with is_runnable_in_scx=true.
        let runnable_per_cpu: Vec<(u64, u64)> =
            runnable_kvas.iter().map(|&k| (k, 0xc0deu64)).collect();
        let entries = merge_global_and_runnable(&global, &runnable_per_cpu);

        assert_eq!(entries.len(), 1, "race-window task must surface");
        assert_eq!(entries[0].task_kva, n1 as u64);
        assert!(entries[0].is_runnable_in_scx);
        assert_eq!(entries[0].running_pc, Some(0xc0de));
    }
}
