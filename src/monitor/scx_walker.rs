//! Host-side rq->scx + DSQ enumeration walkers for the failure dump.
//!
//! Two entry points:
//!
//! 1. [`walk_rq_scx`] — for one CPU's `struct rq.scx`, captures the
//!    scalar fields the kernel's own `scx_dump_state` reads
//!    (nr_running, flags, cpu_released, ops_qseq, kick_sync) plus
//!    nr_immed, clock, and the curr task pid+comm. Walks
//!    `rq->scx.runnable_list` and emits a list of
//!    [`super::dump::TaskWalkerEntry`] tuples for each runnable task —
//!    these feed directly into the per-task enrichment capture
//!    pipeline.
//!
//! 2. [`walk_dsqs`] — enumerates every dispatch queue reachable from
//!    `*scx_root`:
//!    - per-node global DSQs via `scx_sched.pnode[node]->global_dsq`
//!    - per-CPU local DSQs via `rq->scx.local_dsq`
//!    - per-CPU bypass DSQs via `scx_sched_pcpu.bypass_dsq`
//!    - user-allocated DSQs via the `scx_sched.dsq_hash` rhashtable
//!    For each DSQ captures the scalar state (id, nr, seq) and walks
//!    its `list_head` to enumerate queued tasks. The kernel's own
//!    `scx_dump_state` does NOT enumerate per-DSQ depths — this
//!    walker surfaces data even mainline misses (research_debug_probes
//!    phd-debug7).
//!
//! Both walkers are best-effort: any address that fails to translate
//! (slab page race, PA out of bounds) yields a partial result rather
//! than aborting. Cycle protection is per-list (MAX_NODES_PER_LIST);
//! the rhashtable walk caps total bucket-table chain length at
//! MAX_RHT_NODES.
//!
//! # Lock-free reads
//!
//! These walkers run from the freeze coordinator after the vCPU
//! rendezvous. All vCPUs are parked at a known KVM exit; the host
//! reads guest memory directly with no in-guest synchronization. The
//! kernel-side locks (`scx_dispatch_q.lock` raw_spinlock,
//! `rhashtable.mutex`) are not honored — the freeze rendezvous IS
//! the synchronization. A torn read can still happen if a vCPU was
//! mid-write at the freeze instant; the walker treats torn results
//! as best-effort partial output.

use serde::{Deserialize, Serialize};

use super::btf_offsets::{
    RHT_PTR_LOCK_BIT, SCX_DSQ_LNODE_ITER_CURSOR, ScxWalkerOffsets,
};
use super::dump::TaskWalkerEntry;
use super::guest::GuestKernel;
use super::idr::translate_any_kva;
use super::reader::GuestMem;

/// Maximum entries any single list_head walk visits before bailing
/// with what's been collected. Bounds CPU + memory cost on a
/// corrupt-pointer chain that loops back on itself or runs into
/// arbitrary slab. 4096 is generous: real per-CPU runnable_list has
/// at most ~num_threads entries on a given CPU, capped well below
/// this; user DSQs can in principle hold millions of tasks but the
/// per-DSQ walker still bails at this cap so a million-entry DSQ
/// surfaces 4096 task entries plus the `nr` count (truncation
/// surfaces via `truncated: true` on [`DsqState`]).
const MAX_NODES_PER_LIST: u32 = 4096;

/// Maximum total node visits across all rhashtable buckets in the
/// `dsq_hash` walk. Bounds the cost of a runaway bucket chain.
/// Mainline ScxLib creates at most a few hundred user DSQs.
const MAX_RHT_NODES: u32 = 8192;

/// Maximum number of buckets the rhashtable walker enumerates.
/// `bucket_table.size` is normally a small power of two
/// (rhashtable starts at 16, grows by 2x); a pathological torn
/// read could surface a huge value. Caps the bucket walk at 64K to
/// protect freeze-path latency.
const MAX_RHT_BUCKETS: u32 = 65_536;

/// Snapshot of one CPU's `struct rq.scx` state at freeze time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RqScxState {
    /// CPU index (0-based) this state describes.
    pub cpu: u32,
    /// `rq->scx.nr_running`.
    pub nr_running: u32,
    /// `rq->scx.flags`.
    pub flags: u32,
    /// `rq->scx.cpu_released` — `true` when the kernel released
    /// the CPU back to the BPF scheduler (see `scx_pre_release_cpu`
    /// in kernel/sched/ext.c).
    pub cpu_released: bool,
    /// `rq->scx.ops_qseq`.
    pub ops_qseq: u64,
    /// `rq->scx.kick_sync`.
    pub kick_sync: u64,
    /// `rq->scx.nr_immed` — count of ENQ_IMMED tasks on local_dsq.
    pub nr_immed: u32,
    /// `rq->clock` — per-CPU rq clock at the freeze instant.
    pub rq_clock: u64,
    /// `rq->curr->pid` — the currently-running task. `None` when
    /// the curr pointer didn't translate (idle or torn read).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curr_pid: Option<i32>,
    /// `rq->curr->comm`. Mirrors `curr_pid`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curr_comm: Option<String>,
    /// `task_struct` KVAs of every entry walked off
    /// `rq->scx.runnable_list`. The freeze coordinator passes this
    /// vec into the per-task enrichment capture so the same
    /// task list drives both rq->scx state AND per-task records.
    pub runnable_task_kvas: Vec<u64>,
    /// True when the runnable_list walk hit the
    /// [`MAX_NODES_PER_LIST`] safety cap before reaching the head
    /// — typical only on a corrupted chain.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub runnable_truncated: bool,
}

/// Snapshot of one DSQ's state — built-in (per-CPU local, per-CPU
/// bypass, per-node global) or user-allocated.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DsqState {
    /// `scx_dispatch_q.id` — built-in DSQs use synthetic ids
    /// (`SCX_DSQ_LOCAL`, `SCX_DSQ_GLOBAL` per-node); user DSQs
    /// carry the BPF-allocated id.
    pub id: u64,
    /// Operator-facing tag describing where the DSQ came from:
    /// `"local cpu N"`, `"bypass cpu N"`, `"global node N"`, or
    /// `"user"`. Aligned with the kernel's own `scx_dump_state`
    /// terminology where comparable.
    pub origin: String,
    /// `scx_dispatch_q.nr` — number of tasks currently queued.
    pub nr: u32,
    /// `scx_dispatch_q.seq` — BPF-iter seq counter, used by the
    /// dual-snapshot delta to distinguish dead vs busy DSQs:
    /// `Δnr=0 + Δseq=0` is a dead DSQ; `Δseq>>Δ(seq-nr)` indicates
    /// unbounded growth (per research_debug_probes phd-debug7).
    pub seq: u32,
    /// `task_struct` KVAs walked off the DSQ's `list_head`. Same
    /// shape as [`RqScxState::runnable_task_kvas`] — feeds into
    /// the same per-task enrichment pipeline.
    pub task_kvas: Vec<u64>,
    /// True when the DSQ list walk hit the
    /// [`MAX_NODES_PER_LIST`] cap before reaching the head.
    /// Distinct from `nr`: the kernel may report `nr` larger than
    /// our walk cap on legitimately-deep DSQs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Top-level scheduler state captured from `*scx_root`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ScxSchedState {
    /// `scx_sched.aborting`. `true` when the scheduler is in the
    /// abort path; `bypass_depth` typically rises here.
    pub aborting: bool,
    /// `scx_sched.bypass_depth`. Nesting depth of the bypass-mode
    /// stack; non-zero means the kernel is dispatching tasks
    /// without consulting the BPF scheduler.
    pub bypass_depth: i32,
    /// `scx_sched.exit_kind` — the SCX_EXIT_* enum value latched
    /// at `scx_error()` time. 0 means no exit yet; non-zero values
    /// match `enum scx_exit_kind` in
    /// `include/linux/sched/ext.h`.
    pub exit_kind: u32,
}

/// Walk one CPU's `rq->scx` state. Reads the scalar fields and
/// the runnable_list, returning [`RqScxState`] plus a vec of
/// [`TaskWalkerEntry`] entries for the per-task enrichment pipeline.
///
/// `cpu` is the 0-based CPU index; `rq_kva` and `rq_pa` address that
/// CPU's `struct rq`. Caller resolves both via
/// `runqueues + per_cpu_offset[cpu]` (KVA) plus the corresponding PA
/// via `compute_rq_pas`.
///
/// Cap on visited nodes: [`MAX_NODES_PER_LIST`]. A truncated walk
/// surfaces via [`RqScxState::runnable_truncated`].
///
/// Returns `None` when any of the offset sub-groups required for
/// scalar reads (`rq`, `scx_rq`, `task`) is absent — the walker
/// cannot synthesize partial scalars meaningfully without the rq /
/// scx_rq base offsets. Per-CPU runnable_list walking additionally
/// requires `see` (sched_ext_entity); when only `see` is missing the
/// scalar capture still lands but the runnable_list walk yields
/// nothing.
#[allow(dead_code)]
pub fn walk_rq_scx(
    kernel: &GuestKernel<'_>,
    cpu: u32,
    rq_kva: u64,
    rq_pa: u64,
    offsets: &ScxWalkerOffsets,
) -> Option<(RqScxState, Vec<TaskWalkerEntry>)> {
    let rq_offs = offsets.rq.as_ref()?;
    let scx_rq_offs = offsets.scx_rq.as_ref()?;
    let task_offs = offsets.task.as_ref()?;

    let mem = kernel.mem();
    let cr3_pa = kernel.cr3_pa();
    let page_offset = kernel.page_offset();
    let l5 = kernel.l5();

    let scx_off = rq_offs.scx;

    // Scalar reads off rq + scx_rq.
    let nr_running = mem.read_u32(rq_pa, scx_off + scx_rq_offs.nr_running);
    let flags = mem.read_u32(rq_pa, scx_off + scx_rq_offs.flags);
    let cpu_released = mem.read_u8(rq_pa, scx_off + scx_rq_offs.cpu_released) != 0;
    let ops_qseq = mem.read_u64(rq_pa, scx_off + scx_rq_offs.ops_qseq);
    let kick_sync = mem.read_u64(rq_pa, scx_off + scx_rq_offs.kick_sync);
    let nr_immed = mem.read_u32(rq_pa, scx_off + scx_rq_offs.nr_immed);
    let rq_clock = mem.read_u64(rq_pa, scx_off + scx_rq_offs.clock);

    // curr task — pointer follow.
    let curr_kva = mem.read_u64(rq_pa, rq_offs.curr);
    let (curr_pid, curr_comm) = read_task_pid_comm(
        mem,
        cr3_pa,
        page_offset,
        l5,
        curr_kva,
        task_offs.pid,
        task_offs.comm,
    );

    // Walk runnable_list when sched_ext_entity offsets are available.
    // Without `see` we can still report scalar state but cannot
    // container_of a runnable_node back to its task_struct.
    let (runnable_task_kvas, runnable_truncated) = if let Some(see_offs) =
        offsets.see.as_ref()
    {
        let list_head_off = scx_off + scx_rq_offs.runnable_list;
        let head_kva = rq_kva.wrapping_add(list_head_off as u64);
        let head_pa = rq_pa.wrapping_add(list_head_off as u64);

        // container_of offset within task_struct: each runnable_node
        // is at task + task_struct.scx + see.runnable_node.
        let runnable_node_off_in_task = task_offs.scx + see_offs.runnable_node;

        walk_list_head_for_task_kvas(
            mem,
            cr3_pa,
            page_offset,
            l5,
            head_kva,
            head_pa,
            runnable_node_off_in_task,
        )
    } else {
        (Vec::new(), false)
    };

    let walker_entries: Vec<TaskWalkerEntry> = runnable_task_kvas
        .iter()
        .map(|&task_kva| TaskWalkerEntry {
            task_kva,
            // Runnable on this CPU's scx — eligible for the
            // pi_boosted_out_of_scx flag.
            is_runnable_in_scx: true,
            // running_pc only known for the curr task; the
            // freeze coordinator can fill that via
            // VcpuRegSnapshot.instruction_pointer at a higher
            // level. The walker leaves it None.
            running_pc: None,
        })
        .collect();

    let state = RqScxState {
        cpu,
        nr_running,
        flags,
        cpu_released,
        ops_qseq,
        kick_sync,
        nr_immed,
        rq_clock,
        curr_pid,
        curr_comm,
        runnable_task_kvas,
        runnable_truncated,
    };

    Some((state, walker_entries))
}

/// Read scalar `scx_sched` fields off `*scx_root`.
///
/// `scx_root` is a kernel-text-mapped pointer at the resolved KVA;
/// `*scx_root` points at the active `struct scx_sched`. Returns
/// `None` when scx_root is unset (no scheduler attached), the read
/// fails, or the `scx_sched` offset sub-group is missing from BTF.
#[allow(dead_code)]
pub fn read_scx_sched_state(
    kernel: &GuestKernel<'_>,
    scx_root_kva: u64,
    offsets: &ScxWalkerOffsets,
) -> Option<(u64, ScxSchedState)> {
    let sched_offs = offsets.sched.as_ref()?;

    let mem = kernel.mem();
    let cr3_pa = kernel.cr3_pa();
    let page_offset = kernel.page_offset();
    let l5 = kernel.l5();

    let root_pa = super::symbols::text_kva_to_pa(scx_root_kva);
    let sched_kva = mem.read_u64(root_pa, 0);
    if sched_kva == 0 {
        return None;
    }
    let sched_pa = translate_any_kva(mem, cr3_pa, page_offset, sched_kva, l5)?;

    let aborting = mem.read_u8(sched_pa, sched_offs.aborting) != 0;
    let bypass_depth = mem.read_u32(sched_pa, sched_offs.bypass_depth) as i32;
    // `exit_kind` is `atomic_t`; the value lives in the `counter`
    // field at offset 0 of atomic_t. We're already at the
    // outer-struct offset of `exit_kind`, so a u32 read at that
    // offset reads the `counter` directly.
    let exit_kind = mem.read_u32(sched_pa, sched_offs.exit_kind);

    Some((
        sched_kva,
        ScxSchedState {
            aborting,
            bypass_depth,
            exit_kind,
        },
    ))
}

/// Walk every DSQ reachable from a `scx_sched` and produce one
/// `DsqState` per DSQ plus a flat vec of `TaskWalkerEntry` rows for
/// the per-task enrichment pipeline.
///
/// Walks (in this order, gated on the relevant sub-group offsets
/// being present):
///   1. Per-CPU local DSQs at `rq->scx.local_dsq` for every CPU
///      (needs `rq`, `scx_rq`, `dsq`, `dsq_lnode`, `task`, `see`).
///   2. Per-CPU bypass DSQs at `scx_sched_pcpu.bypass_dsq` for
///      every CPU (needs `sched`, `sched_pcpu`, plus the leaf set
///      above).
///   3. Per-node global DSQs at `scx_sched.pnode[node]->global_dsq`
///      for every NUMA node (needs `sched`, `sched_pnode`, plus
///      leaf set).
///   4. User-allocated DSQs walked through `scx_sched.dsq_hash`
///      (needs `sched`, `rht`, plus leaf set).
///
/// Each pass is independent: missing offsets for one pass blind
/// only that pass. A translate failure on one DSQ leaves it out of
/// the result without affecting the others.
#[allow(dead_code)]
pub fn walk_dsqs(
    kernel: &GuestKernel<'_>,
    sched_pa: u64,
    rq_kvas: &[u64],
    rq_pas: &[u64],
    per_cpu_offsets: &[u64],
    nr_nodes: u32,
    offsets: &ScxWalkerOffsets,
) -> (Vec<DsqState>, Vec<TaskWalkerEntry>) {
    let mem = kernel.mem();
    let cr3_pa = kernel.cr3_pa();
    let page_offset = kernel.page_offset();
    let l5 = kernel.l5();

    let mut dsq_states: Vec<DsqState> = Vec::new();
    let mut all_entries: Vec<TaskWalkerEntry> = Vec::new();

    // Leaf offsets common to every pass — all four DSQ-walking
    // passes feed `walk_one_dsq` which needs these. If any leaf
    // group is missing, no pass can run.
    let (Some(dsq_offs), Some(dsq_lnode_offs), Some(task_offs), Some(see_offs)) = (
        offsets.dsq.as_ref(),
        offsets.dsq_lnode.as_ref(),
        offsets.task.as_ref(),
        offsets.see.as_ref(),
    ) else {
        return (dsq_states, all_entries);
    };

    // Pass 1: per-CPU local DSQs. Needs rq + scx_rq sub-groups for
    // the local_dsq embedded in each rq.
    if let (Some(rq_offs), Some(scx_rq_offs)) =
        (offsets.rq.as_ref(), offsets.scx_rq.as_ref())
    {
        for (cpu, (&rq_kva, &rq_pa)) in rq_kvas.iter().zip(rq_pas.iter()).enumerate() {
            let dsq_kva = rq_kva
                .wrapping_add((rq_offs.scx + scx_rq_offs.local_dsq) as u64);
            let dsq_pa = rq_pa
                .wrapping_add((rq_offs.scx + scx_rq_offs.local_dsq) as u64);
            if let Some((state, entries)) = walk_one_dsq(
                mem,
                cr3_pa,
                page_offset,
                l5,
                dsq_kva,
                dsq_pa,
                format!("local cpu {cpu}"),
                dsq_offs,
                dsq_lnode_offs,
                task_offs,
                see_offs,
            ) {
                all_entries.extend(entries);
                dsq_states.push(state);
            }
        }
    }

    // Pass 2: per-CPU bypass DSQs. The percpu base lives at
    // sched->pcpu, dereferenced as a __percpu pointer; each CPU's
    // address is `pcpu_base + per_cpu_offset[cpu] +
    // scx_sched_pcpu.bypass_dsq`.
    if let (Some(sched_offs), Some(pcpu_offs)) =
        (offsets.sched.as_ref(), offsets.sched_pcpu.as_ref())
    {
        let pcpu_kva = mem.read_u64(sched_pa, sched_offs.pcpu);
        if pcpu_kva != 0 {
            for (cpu, &cpu_off) in per_cpu_offsets.iter().enumerate() {
                // Skip out-of-range CPUs — same heuristic as
                // read_percpu_array_value (cpu_off==0 && cpu_index>0
                // means BSS-zero tail).
                if cpu_off == 0 && cpu > 0 {
                    continue;
                }
                let dsq_kva = pcpu_kva
                    .wrapping_add(cpu_off)
                    .wrapping_add(pcpu_offs.bypass_dsq as u64);
                if let Some(dsq_pa) =
                    translate_any_kva(mem, cr3_pa, page_offset, dsq_kva, l5)
                    && let Some((state, entries)) = walk_one_dsq(
                        mem,
                        cr3_pa,
                        page_offset,
                        l5,
                        dsq_kva,
                        dsq_pa,
                        format!("bypass cpu {cpu}"),
                        dsq_offs,
                        dsq_lnode_offs,
                        task_offs,
                        see_offs,
                    )
                {
                    all_entries.extend(entries);
                    dsq_states.push(state);
                }
            }
        }
    }

    // Pass 3: per-node global DSQs. `sched->pnode` is a pointer
    // to an array of `struct scx_sched_pnode *` of length nr_nodes.
    if let (Some(sched_offs), Some(pnode_offs)) =
        (offsets.sched.as_ref(), offsets.sched_pnode.as_ref())
    {
        let pnode_kva = mem.read_u64(sched_pa, sched_offs.pnode);
        if pnode_kva != 0
            && let Some(pnode_arr_pa) =
                translate_any_kva(mem, cr3_pa, page_offset, pnode_kva, l5)
        {
            for node in 0..nr_nodes as u64 {
                let pnode_ptr_kva = mem.read_u64(pnode_arr_pa, (node * 8) as usize);
                if pnode_ptr_kva == 0 {
                    continue;
                }
                let Some(pnode_pa) =
                    translate_any_kva(mem, cr3_pa, page_offset, pnode_ptr_kva, l5)
                else {
                    continue;
                };
                let dsq_kva = pnode_ptr_kva.wrapping_add(pnode_offs.global_dsq as u64);
                let dsq_pa = pnode_pa.wrapping_add(pnode_offs.global_dsq as u64);
                if let Some((state, entries)) = walk_one_dsq(
                    mem,
                    cr3_pa,
                    page_offset,
                    l5,
                    dsq_kva,
                    dsq_pa,
                    format!("global node {node}"),
                    dsq_offs,
                    dsq_lnode_offs,
                    task_offs,
                    see_offs,
                ) {
                    all_entries.extend(entries);
                    dsq_states.push(state);
                }
            }
        }
    }

    // Pass 4: user-allocated DSQs via the scx_sched.dsq_hash
    // rhashtable. Walks at most MAX_RHT_NODES nodes total across
    // all buckets.
    if let (Some(sched_offs), Some(rht_offs)) =
        (offsets.sched.as_ref(), offsets.rht.as_ref())
    {
        let rht_kva = sched_pa.wrapping_add(sched_offs.dsq_hash as u64);
        // dsq_hash is embedded in scx_sched (not a pointer); rht_kva
        // here is a KVA we can translate directly. The walker reads
        // it via the rht sub-group offsets.
        let user_dsqs = walk_user_dsq_hash(
            mem,
            cr3_pa,
            page_offset,
            l5,
            rht_kva,
            rht_offs,
            dsq_offs,
        );
        for dsq_kva in user_dsqs {
            let Some(dsq_pa) = translate_any_kva(mem, cr3_pa, page_offset, dsq_kva, l5)
            else {
                continue;
            };
            if let Some((state, entries)) = walk_one_dsq(
                mem,
                cr3_pa,
                page_offset,
                l5,
                dsq_kva,
                dsq_pa,
                "user".to_string(),
                dsq_offs,
                dsq_lnode_offs,
                task_offs,
                see_offs,
            ) {
                all_entries.extend(entries);
                dsq_states.push(state);
            }
        }
    }

    (dsq_states, all_entries)
}

/// Walk one `scx_dispatch_q`. Returns the DSQ scalar state plus the
/// task entries on its `list`.
#[allow(clippy::too_many_arguments)]
fn walk_one_dsq(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    dsq_kva: u64,
    dsq_pa: u64,
    origin: String,
    dsq_offs: &super::btf_offsets::ScxDispatchQOffsets,
    dsq_lnode_offs: &super::btf_offsets::ScxDsqListNodeOffsets,
    task_offs: &super::btf_offsets::TaskStructCoreOffsets,
    see_offs: &super::btf_offsets::SchedExtEntityOffsets,
) -> Option<(DsqState, Vec<TaskWalkerEntry>)> {
    let id = mem.read_u64(dsq_pa, dsq_offs.id);
    let nr = mem.read_u32(dsq_pa, dsq_offs.nr);
    let seq = mem.read_u32(dsq_pa, dsq_offs.seq);

    // List head at dsq + list.
    let head_kva = dsq_kva.wrapping_add(dsq_offs.list as u64);
    let head_pa = dsq_pa.wrapping_add(dsq_offs.list as u64);

    // The DSQ list links sched_ext_entity.dsq_list.node fields
    // (struct list_head inside scx_dsq_list_node inside
    // sched_ext_entity inside task_struct). container_of computes:
    //   task_kva = node_kva
    //            - task.scx
    //            - see.dsq_list
    //            - dsq_lnode.node
    let dsq_node_off_in_task =
        task_offs.scx + see_offs.dsq_list + dsq_lnode_offs.node;

    let (task_kvas, truncated) = walk_list_head_for_dsq_task_kvas(
        mem,
        cr3_pa,
        page_offset,
        l5,
        head_kva,
        head_pa,
        dsq_node_off_in_task,
        dsq_lnode_offs,
    );

    let entries: Vec<TaskWalkerEntry> = task_kvas
        .iter()
        .map(|&task_kva| TaskWalkerEntry {
            task_kva,
            // Tasks queued on a DSQ are NOT on the per-CPU
            // runnable_list — they're staged for dispatch but not
            // yet runnable in the rq->scx sense. The
            // pi_boosted_out_of_scx flag only fires for
            // runnable_list tasks (the scenario it diagnoses is a
            // task that should have left the runnable_list when
            // its sched_class changed but didn't).
            is_runnable_in_scx: false,
            running_pc: None,
        })
        .collect();

    Some((
        DsqState {
            id,
            origin,
            nr,
            seq,
            task_kvas,
            truncated,
        },
        entries,
    ))
}

/// Walk a generic `list_head` chain starting at `head_kva`/`head_pa`,
/// recovering each task_struct KVA via container_of with
/// `runnable_node_off_in_task` as the field offset within
/// task_struct.
///
/// Returns (task_kvas, truncated). `truncated` is true when the
/// MAX_NODES_PER_LIST cap kicked in before the walk closed back to
/// the head.
fn walk_list_head_for_task_kvas(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    head_kva: u64,
    head_pa: u64,
    runnable_node_off_in_task: usize,
) -> (Vec<u64>, bool) {
    let mut task_kvas = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        return (task_kvas, false);
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return (task_kvas, true);
        }
        visited += 1;

        // container_of: task_kva = node_kva - runnable_node_off_in_task
        let task_kva = node_kva.wrapping_sub(runnable_node_off_in_task as u64);
        task_kvas.push(task_kva);

        // Step to next node — translate node_kva, read .next at offset 0.
        let Some(node_pa) = translate_any_kva(mem, cr3_pa, page_offset, node_kva, l5)
        else {
            return (task_kvas, false);
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return (task_kvas, false);
        }
        node_kva = next_kva;
    }
    (task_kvas, false)
}

/// Walk a DSQ's `list` chain (a list of `scx_dsq_list_node.node`
/// entries embedded in `sched_ext_entity.dsq_list`). Skips iterator
/// cursor entries marked with `SCX_DSQ_LNODE_ITER_CURSOR`.
fn walk_list_head_for_dsq_task_kvas(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    head_kva: u64,
    head_pa: u64,
    dsq_node_off_in_task: usize,
    dsq_lnode_offs: &super::btf_offsets::ScxDsqListNodeOffsets,
) -> (Vec<u64>, bool) {
    let mut task_kvas = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        return (task_kvas, false);
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return (task_kvas, true);
        }
        visited += 1;

        // The list_head we're walking is `scx_dsq_list_node.node`
        // (the inner `struct list_head`). Recover the parent
        // scx_dsq_list_node start by subtracting `dsq_lnode.node`
        // — fixed at 0 in current kernels but we read it from the
        // offsets struct for forward-compatibility.
        let lnode_kva = node_kva.wrapping_sub(dsq_lnode_offs.node as u64);

        // Read the lnode's flags to skip iterator-cursor entries.
        if let Some(lnode_pa) =
            translate_any_kva(mem, cr3_pa, page_offset, lnode_kva, l5)
        {
            let lnode_flags = mem.read_u32(lnode_pa, dsq_lnode_offs.flags);
            if lnode_flags & SCX_DSQ_LNODE_ITER_CURSOR != 0 {
                // Cursor entry — advance without recording.
                let Some(node_pa) =
                    translate_any_kva(mem, cr3_pa, page_offset, node_kva, l5)
                else {
                    return (task_kvas, false);
                };
                let next_kva = mem.read_u64(node_pa, 0);
                if next_kva == 0 {
                    return (task_kvas, false);
                }
                node_kva = next_kva;
                continue;
            }
        }

        // Real task entry: container_of from the inner list_head's
        // node_kva back to task_struct. The full offset within
        // task_struct is task.scx + see.dsq_list + dsq_lnode.node.
        let task_kva = node_kva.wrapping_sub(dsq_node_off_in_task as u64);
        task_kvas.push(task_kva);

        let Some(node_pa) = translate_any_kva(mem, cr3_pa, page_offset, node_kva, l5)
        else {
            return (task_kvas, false);
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return (task_kvas, false);
        }
        node_kva = next_kva;
    }
    (task_kvas, false)
}

/// Walk the user-allocated DSQ rhashtable rooted at `rht_kva`.
///
/// `rht_kva` addresses the embedded `struct rhashtable` inside
/// `scx_sched.dsq_hash`. The walker reads `tbl` (bucket_table
/// pointer), then for each of `tbl.buckets[i]` it strips the
/// LSB tag (`RHT_PTR_LOCK_BIT`) and chases the `rhash_head.next`
/// chain. For each node the walker computes
/// `dsq_kva = node_kva - scx_dispatch_q.hash_node` (container_of).
///
/// Caps:
/// - bucket count at [`MAX_RHT_BUCKETS`]
/// - total nodes visited at [`MAX_RHT_NODES`]
fn walk_user_dsq_hash(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    rht_kva: u64,
    rht_offs: &super::btf_offsets::RhashtableOffsets,
    dsq_offs: &super::btf_offsets::ScxDispatchQOffsets,
) -> Vec<u64> {
    let mut dsq_kvas = Vec::new();

    let Some(rht_pa) = translate_any_kva(mem, cr3_pa, page_offset, rht_kva, l5)
    else {
        return dsq_kvas;
    };

    let tbl_kva = mem.read_u64(rht_pa, rht_offs.tbl);
    if tbl_kva == 0 {
        return dsq_kvas;
    }
    let Some(tbl_pa) = translate_any_kva(mem, cr3_pa, page_offset, tbl_kva, l5)
    else {
        return dsq_kvas;
    };

    let size = mem.read_u32(tbl_pa, rht_offs.bucket_table_size);
    let bucket_count = size.min(MAX_RHT_BUCKETS) as u64;
    let buckets_off = rht_offs.bucket_table_buckets;

    let mut total_nodes: u32 = 0;
    for i in 0..bucket_count {
        if total_nodes >= MAX_RHT_NODES {
            return dsq_kvas;
        }
        let entry_off = buckets_off + (i as usize) * 8;
        let raw_ptr = mem.read_u64(tbl_pa, entry_off);
        // Strip the LSB lock-bit tag. NULL or pure tag (0 with
        // bit 0 unset) means empty bucket.
        let head_kva = raw_ptr & !RHT_PTR_LOCK_BIT;
        if head_kva == 0 {
            continue;
        }
        // Chase the `rhash_head.next` chain. Each node is a
        // `rhash_head` embedded in scx_dispatch_q at
        // `hash_node`; container_of yields the dsq KVA.
        let mut node_kva = head_kva;
        let mut chain_visited: u32 = 0;
        while node_kva != 0 && total_nodes < MAX_RHT_NODES && chain_visited < 1024 {
            chain_visited += 1;
            total_nodes += 1;
            let dsq_kva = node_kva.wrapping_sub(dsq_offs.hash_node as u64);
            dsq_kvas.push(dsq_kva);
            let Some(node_pa) =
                translate_any_kva(mem, cr3_pa, page_offset, node_kva, l5)
            else {
                break;
            };
            let next_raw = mem.read_u64(node_pa, rht_offs.rhash_head_next);
            // The chain terminator is a "nulls" pointer with bit 0
            // set encoding the bucket index; treat any LSB-tagged
            // pointer as terminator.
            if next_raw & RHT_PTR_LOCK_BIT != 0 || next_raw == 0 {
                break;
            }
            node_kva = next_raw;
        }
    }

    dsq_kvas
}

/// Read `(pid, comm)` for a `task_struct *` after a NULL-check and
/// translate. Returns `(None, None)` on NULL or untranslatable.
fn read_task_pid_comm(
    mem: &GuestMem,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    task_kva: u64,
    pid_off: usize,
    comm_off: usize,
) -> (Option<i32>, Option<String>) {
    if task_kva == 0 {
        return (None, None);
    }
    let Some(task_pa) = translate_any_kva(mem, cr3_pa, page_offset, task_kva, l5)
    else {
        return (None, None);
    };
    let pid = mem.read_u32(task_pa, pid_off) as i32;
    let mut buf = [0u8; 16];
    mem.read_bytes(task_pa + comm_off as u64, &mut buf);
    let n = buf.iter().position(|&b| b == 0).unwrap_or(16);
    let comm = String::from_utf8_lossy(&buf[..n]).to_string();
    (Some(pid), Some(comm))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin RqScxState wire shape — every optional field skips
    /// on None, required fields land directly. Same coverage style
    /// as task_enrichment::tests::task_enrichment_serde_skip_none_fields.
    #[test]
    fn rq_scx_state_serde_skip_none() {
        let s = RqScxState {
            cpu: 3,
            nr_running: 4,
            flags: 0x10,
            cpu_released: false,
            ops_qseq: 100,
            kick_sync: 50,
            nr_immed: 0,
            rq_clock: 1234567,
            curr_pid: None,
            curr_comm: None,
            runnable_task_kvas: vec![],
            runnable_truncated: false,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("curr_pid"));
        assert!(!json.contains("curr_comm"));
        assert!(!json.contains("runnable_truncated"));
        assert!(json.contains("\"cpu\":3"));
        assert!(json.contains("\"nr_running\":4"));
    }

    /// Roundtrip every populated field — Verdict-routed so a single
    /// regression in any field surfaces with its own labeled detail
    /// rather than a cliff-edge `assert_eq!` panic that hides the
    /// other field outcomes. Better signal when multiple serde
    /// renames land at once.
    #[test]
    fn rq_scx_state_serde_roundtrip_populated() {
        use crate::assert::Verdict;

        let s = RqScxState {
            cpu: 1,
            nr_running: 2,
            flags: 0x1,
            cpu_released: true,
            ops_qseq: 42,
            kick_sync: 17,
            nr_immed: 1,
            rq_clock: 999_999,
            curr_pid: Some(1234),
            curr_comm: Some("ktstr".into()),
            runnable_task_kvas: vec![0xffff_ffff_8000_1000, 0xffff_ffff_8000_2000],
            runnable_truncated: true,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: RqScxState = serde_json::from_str(&json).unwrap();

        let parsed_cpu = parsed.cpu;
        let parsed_nr_running = parsed.nr_running;
        let parsed_flags = parsed.flags;
        let parsed_cpu_released = parsed.cpu_released;
        let parsed_ops_qseq = parsed.ops_qseq;
        let parsed_kick_sync = parsed.kick_sync;
        let parsed_nr_immed = parsed.nr_immed;
        let parsed_rq_clock = parsed.rq_clock;
        let parsed_curr_pid = parsed.curr_pid;
        let parsed_curr_comm = parsed.curr_comm.clone();
        let parsed_runnable_kvas_len = parsed.runnable_task_kvas.len();
        let parsed_runnable_truncated = parsed.runnable_truncated;

        let mut v = Verdict::new();
        crate::claim!(v, parsed_cpu).eq(1u32);
        crate::claim!(v, parsed_nr_running).eq(2u32);
        crate::claim!(v, parsed_flags).eq(0x1u32);
        crate::claim!(v, parsed_cpu_released).eq(true);
        crate::claim!(v, parsed_ops_qseq).eq(42u64);
        crate::claim!(v, parsed_kick_sync).eq(17u64);
        crate::claim!(v, parsed_nr_immed).eq(1u32);
        crate::claim!(v, parsed_rq_clock).eq(999_999u64);
        // Option<T> doesn't impl Display, so claim on the unwrapped
        // values via match-against-known-shape: bake the expected
        // outcome ("present + value matches") into a single bool.
        let curr_pid_match = parsed_curr_pid == Some(1234);
        let curr_comm_match = parsed_curr_comm.as_deref() == Some("ktstr");
        crate::claim!(v, curr_pid_match).eq(true);
        crate::claim!(v, curr_comm_match).eq(true);
        crate::claim!(v, parsed_runnable_kvas_len).eq(2usize);
        crate::claim!(v, parsed_runnable_truncated).eq(true);
        let r = v.into_result();
        assert!(
            r.passed,
            "rq_scx_state roundtrip claims must all pass: {:?}",
            r.details,
        );
    }

    #[test]
    fn dsq_state_serde_skip_truncated_when_false() {
        let d = DsqState {
            id: 0xdead_beef,
            origin: "user".into(),
            nr: 5,
            seq: 100,
            task_kvas: vec![],
            truncated: false,
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(!json.contains("truncated"));
        assert!(json.contains("\"id\":3735928559"));
        assert!(json.contains("\"nr\":5"));
        assert!(json.contains("\"seq\":100"));
    }

    #[test]
    fn dsq_state_serde_emits_truncated_when_true() {
        let d = DsqState {
            id: 1,
            origin: "global node 0".into(),
            nr: 5000,
            seq: 5001,
            task_kvas: (0..MAX_NODES_PER_LIST as u64).collect(),
            truncated: true,
        };
        let json = serde_json::to_string(&d).unwrap();
        assert!(json.contains("\"truncated\":true"));
    }

    #[test]
    fn scx_sched_state_default_empty() {
        let s = ScxSchedState::default();
        assert!(!s.aborting);
        assert_eq!(s.bypass_depth, 0);
        assert_eq!(s.exit_kind, 0);
    }

    /// Roundtrip every scalar field — Verdict-routed so a serde
    /// rename on one field doesn't mask the other two.
    #[test]
    fn scx_sched_state_serde_roundtrip() {
        use crate::assert::Verdict;

        let s = ScxSchedState {
            aborting: true,
            bypass_depth: 2,
            // SCX_EXIT_ERROR_BPF per include/linux/sched/ext.h
            exit_kind: 1027,
        };
        let json = serde_json::to_string(&s).unwrap();
        let parsed: ScxSchedState = serde_json::from_str(&json).unwrap();

        let parsed_aborting = parsed.aborting;
        let parsed_bypass_depth = parsed.bypass_depth;
        let parsed_exit_kind = parsed.exit_kind;

        let mut v = Verdict::new();
        crate::claim!(v, parsed_aborting).eq(true);
        crate::claim!(v, parsed_bypass_depth).eq(2i32);
        crate::claim!(v, parsed_exit_kind).eq(1027u32);
        let r = v.into_result();
        assert!(
            r.passed,
            "scx_sched_state roundtrip claims must all pass: {:?}",
            r.details,
        );
    }

    /// Walk a hand-built list with two task entries — verifies
    /// the container_of subtraction and the cycle-back termination.
    #[test]
    fn walk_list_head_basic_two_tasks() {
        // Layout (PA == KVA in this test for simplicity):
        //   PA 0x100: head (next at 0, prev at 8)
        //   PA 0x200: task1's runnable_node (next at 0, prev at 8)
        //     task1 starts at PA 0x200 - runnable_node_off_in_task
        //   PA 0x300: task2's runnable_node
        //     task2 starts at PA 0x300 - runnable_node_off_in_task
        //
        // head.next = 0x200, task1.next = 0x300, task2.next = 0x100 (back to head)
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        let n1 = 0x200usize;
        let n2 = 0x300usize;
        // head.next = n1
        buf[head..head + 8].copy_from_slice(&(n1 as u64).to_le_bytes());
        // n1.next = n2
        buf[n1..n1 + 8].copy_from_slice(&(n2 as u64).to_le_bytes());
        // n2.next = head (terminator)
        buf[n2..n2 + 8].copy_from_slice(&(head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        // Identity translation: PA == KVA for this minimal setup,
        // page_offset = 0 so kva_to_pa is identity.
        let runnable_node_off = 0x10usize;
        let (kvas, truncated) = walk_list_head_for_task_kvas(
            &mem, 0, 0, false, head as u64, head as u64, runnable_node_off,
        );
        assert!(!truncated);
        assert_eq!(kvas.len(), 2);
        assert_eq!(kvas[0], (n1 - runnable_node_off) as u64);
        assert_eq!(kvas[1], (n2 - runnable_node_off) as u64);
    }

    /// Empty list: head.next == &head. Walker returns no kvas.
    #[test]
    fn walk_list_head_empty() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        // head.next = head
        buf[head..head + 8].copy_from_slice(&(head as u64).to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };

        let (kvas, truncated) =
            walk_list_head_for_task_kvas(&mem, 0, 0, false, head as u64, head as u64, 0x10);
        assert!(!truncated);
        assert!(kvas.is_empty());
    }

    /// Zero next pointer: walker bails defensively without
    /// truncation flag (different from cycle-cap).
    #[test]
    fn walk_list_head_zero_next_bails() {
        let mut buf = vec![0u8; 0x1000];
        let head = 0x100usize;
        // head.next = 0 (uninitialized / unmapped)
        buf[head..head + 8].copy_from_slice(&0u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_ptr() as *mut u8, buf.len() as u64) };
        let (kvas, truncated) =
            walk_list_head_for_task_kvas(&mem, 0, 0, false, head as u64, head as u64, 0x10);
        assert!(!truncated);
        assert!(kvas.is_empty());
    }

    /// `missing_groups()` reports every absent sub-group when offsets
    /// are constructed empty (every Option None). This is the
    /// degenerate input that surfaces every diagnostic name.
    #[test]
    fn scx_walker_offsets_missing_groups_reports_all_when_empty() {
        let offsets = ScxWalkerOffsets {
            rq: None,
            scx_rq: None,
            task: None,
            see: None,
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };
        let missing = offsets.missing_groups();
        // 10 sub-groups, all missing.
        assert_eq!(missing.len(), 10);
        assert!(missing.contains(&"rq"));
        assert!(missing.contains(&"scx_rq"));
        assert!(missing.contains(&"task_struct"));
        assert!(missing.contains(&"sched_ext_entity"));
        assert!(missing.contains(&"scx_dsq_list_node"));
        assert!(missing.contains(&"scx_dispatch_q"));
        assert!(missing.contains(&"scx_sched"));
        assert!(missing.contains(&"scx_sched_pnode"));
        assert!(missing.contains(&"scx_sched_pcpu"));
        assert!(missing.contains(&"rhashtable/bucket_table/rhash_head"));
    }

    /// `missing_groups()` reports nothing when every sub-group is
    /// resolved — a normal, well-formed BTF parse outcome.
    #[test]
    fn scx_walker_offsets_missing_groups_reports_none_when_full() {
        use super::super::btf_offsets::{
            RhashtableOffsets, RqStructOffsets, SchedExtEntityOffsets, ScxDispatchQOffsets,
            ScxDsqListNodeOffsets, ScxRqOffsets, ScxSchedOffsets, ScxSchedPcpuOffsets,
            ScxSchedPnodeOffsets, TaskStructCoreOffsets,
        };
        let offsets = ScxWalkerOffsets {
            rq: Some(RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(ScxRqOffsets {
                local_dsq: 0,
                runnable_list: 64,
                nr_running: 96,
                flags: 100,
                cpu_released: 104,
                ops_qseq: 112,
                kick_sync: 120,
                nr_immed: 128,
                clock: 136,
            }),
            task: Some(TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 300,
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
            }),
            dsq_lnode: Some(ScxDsqListNodeOffsets { node: 0, flags: 16 }),
            dsq: Some(ScxDispatchQOffsets {
                list: 0,
                nr: 16,
                seq: 20,
                id: 24,
                hash_node: 32,
            }),
            sched: Some(ScxSchedOffsets {
                dsq_hash: 0,
                pnode: 64,
                pcpu: 72,
                aborting: 80,
                bypass_depth: 84,
                exit_kind: 88,
            }),
            sched_pnode: Some(ScxSchedPnodeOffsets { global_dsq: 0 }),
            sched_pcpu: Some(ScxSchedPcpuOffsets { bypass_dsq: 0 }),
            rht: Some(RhashtableOffsets {
                tbl: 0,
                nelems: 8,
                bucket_table_size: 0,
                bucket_table_buckets: 16,
                rhash_head_next: 0,
            }),
        };
        assert!(offsets.missing_groups().is_empty());
    }

    // -- Verdict API integration coverage -------------------------------
    //
    // The walker emits RqScxState / DsqState / ScxSchedState rows that
    // scenario authors will claim against via the new pointwise-claim
    // API. These tests pin the integration shape: walker output flows
    // into Verdict claims via the claim! macro, scalar fields claim
    // through ClaimBuilder, runnable_task_kvas / task_kvas claim
    // through SeqClaim. A regression that breaks the Display impls
    // those claim messages depend on, or that drops the field types
    // claim-able comparators expect, surfaces here.

    /// Author-style claim sequence over a populated RqScxState. The
    /// claims reflect what a scheduler test would actually write —
    /// nr_running ≥ 0, no truncation under healthy load, runnable
    /// task KVA list non-empty when the CPU has running work.
    /// The Verdict accumulates without relying on the legacy Expect
    /// shape; final pass/fail honors every claim.
    #[test]
    fn rq_scx_state_authorial_verdict_claims_compose() {
        use crate::assert::Verdict;

        let s = RqScxState {
            cpu: 2,
            nr_running: 3,
            flags: 0x1,
            cpu_released: false,
            ops_qseq: 4242,
            kick_sync: 100,
            nr_immed: 0,
            rq_clock: 999_999,
            curr_pid: Some(1234),
            curr_comm: Some("ktstr-w".into()),
            runnable_task_kvas: vec![0xffff_ffff_8000_1000, 0xffff_ffff_8000_2000],
            runnable_truncated: false,
        };

        let mut v = Verdict::new();
        // Scalar claims via the claim! macro (label = stringify of expr).
        crate::claim!(v, s.nr_running).at_least(1);
        crate::claim!(v, s.nr_running).at_most(64);
        crate::claim!(v, s.runnable_truncated).eq(false);
        // Sequence claim via claim_seq.
        v.claim_seq("runnable_task_kvas", &s.runnable_task_kvas).nonempty();
        v.claim_seq("runnable_task_kvas", &s.runnable_task_kvas).len_at_most(64);

        let r = v.into_result();
        assert!(
            r.passed,
            "authorial claim sequence on populated RqScxState must pass: {:?}",
            r.details,
        );
    }

    /// Failing claim path: a verdict that calls at_most on
    /// nr_running with a value BELOW the actual count must record
    /// a single kind=Other detail with the field-name label and the
    /// at-most message. Pins the integration of the walker's u32
    /// field type through ClaimBuilder<u32>::at_most's failure
    /// formatter.
    #[test]
    fn rq_scx_state_failing_at_most_records_labeled_detail() {
        use crate::assert::Verdict;

        let s = RqScxState {
            cpu: 0,
            nr_running: 100,
            flags: 0,
            cpu_released: false,
            ops_qseq: 0,
            kick_sync: 0,
            nr_immed: 0,
            rq_clock: 0,
            curr_pid: None,
            curr_comm: None,
            runnable_task_kvas: vec![],
            runnable_truncated: false,
        };

        let mut v = Verdict::new();
        crate::claim!(v, s.nr_running).at_most(10);
        let r = v.into_result();

        assert!(!r.passed, "at_most(10) on nr_running=100 must fail");
        assert_eq!(
            r.details.len(),
            1,
            "exactly one failing detail must record: {:?}",
            r.details,
        );
        let msg = &r.details[0].message;
        assert!(
            msg.contains("s.nr_running"),
            "detail must carry the macro-stringify label: {msg}",
        );
        assert!(
            msg.contains("at most 10"),
            "detail must name the at_most threshold: {msg}",
        );
        assert!(
            msg.contains("100"),
            "detail must include the observed value: {msg}",
        );
    }

    /// DsqState.task_kvas + DsqState.truncated claims compose like
    /// RqScxState's. Pins the walker-DSQ-output shape through the
    /// Verdict surface so a scenario test can write
    /// `claim!(v, dsq.nr).at_most(LIMIT)` and
    /// `v.claim_seq("dsq.task_kvas", &dsq.task_kvas).len_at_most(LIMIT)`
    /// against a real DSQ snapshot.
    #[test]
    fn dsq_state_authorial_verdict_claims_compose() {
        use crate::assert::Verdict;

        let d = DsqState {
            id: 0xdead_beef,
            origin: "user".into(),
            nr: 5,
            seq: 100,
            task_kvas: vec![0xffff_8000_8000_1000; 5],
            truncated: false,
        };

        let mut v = Verdict::new();
        crate::claim!(v, d.nr).at_most(MAX_NODES_PER_LIST);
        crate::claim!(v, d.truncated).eq(false);
        crate::claim!(v, d.seq).at_least(d.nr);
        v.claim_seq("d.task_kvas", &d.task_kvas).len_eq(5);

        let r = v.into_result();
        assert!(
            r.passed,
            "authorial claim sequence on populated DsqState must pass: {:?}",
            r.details,
        );
    }

    /// `ScxSchedState.exit_kind == 0` is the no-error sentinel
    /// (per `enum scx_exit_kind`). Pin via Verdict + claim!(eq) so
    /// scheduler tests can write
    /// `claim!(v, sched.exit_kind).eq(0)` for the healthy-exit
    /// invariant.
    #[test]
    fn scx_sched_state_healthy_exit_kind_claim() {
        use crate::assert::Verdict;

        let healthy = ScxSchedState {
            aborting: false,
            bypass_depth: 0,
            exit_kind: 0,
        };
        let mut v = Verdict::new();
        crate::claim!(v, healthy.aborting).eq(false);
        crate::claim!(v, healthy.bypass_depth).eq(0);
        crate::claim!(v, healthy.exit_kind).eq(0u32);
        let r = v.into_result();
        assert!(r.passed, "healthy-state claims must pass: {:?}", r.details);

        // Inverse: an aborting scheduler with non-zero exit_kind
        // must fail the same claim sequence.
        let aborted = ScxSchedState {
            aborting: true,
            bypass_depth: 4,
            // SCX_EXIT_ERROR_BPF (1027) per include/linux/sched/ext.h.
            exit_kind: 1027,
        };
        let mut v = Verdict::new();
        crate::claim!(v, aborted.exit_kind).eq(0u32);
        let r = v.into_result();
        assert!(!r.passed, "exit_kind=1027 must fail eq(0)");
    }
}
