//! Host-side rq->scx + DSQ + task enumeration walkers for the
//! failure dump.
//!
//! Entry points:
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
//! 2. [`walk_local_dsqs`] — per-CPU local DSQs at
//!    `rq->scx.local_dsq`. Runs unconditionally — local DSQs are
//!    initialized at boot (`init_dsq` at `kernel/sched/ext.c:7772`
//!    for every possible CPU) and exist whether or not a scheduler
//!    is attached, so this surfaces local-DSQ state even when
//!    `*scx_root == 0`.
//!
//! 3. [`walk_dsqs`] — sched-rooted DSQs reachable from `*scx_root`
//!    (excluding per-CPU local DSQs, which [`walk_local_dsqs`]
//!    handles separately):
//!    - per-CPU bypass DSQs via `scx_sched_pcpu.bypass_dsq`
//!    - per-node global DSQs via `scx_sched.pnode[node]->global_dsq`
//!    - user-allocated DSQs via the `scx_sched.dsq_hash` rhashtable
//!
//!    For each DSQ captures the scalar state (id, nr, seq) and walks
//!    its `list_head` to enumerate queued tasks. The kernel's own
//!    `scx_dump_state` does NOT enumerate per-DSQ depths — this
//!    walker surfaces queue depth and per-task ordering that the
//!    in-tree dump path does not.
//!
//! 4. [`walk_scx_tasks_global`] — walks the kernel's global
//!    `scx_tasks` LIST_HEAD via each task's `scx.tasks_node`.
//!    Surfaces every task owned by an scx_sched, surviving the
//!    per-rq runnable_list drain that scheduler teardown
//!    (`scx_bypass`, `kernel/sched/ext.c:5304-5404`) triggers.
//!
//! All walkers are best-effort: any address that fails to translate
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

use super::btf_offsets::{RHT_PTR_LOCK_BIT, SCX_DSQ_LNODE_ITER_CURSOR, ScxWalkerOffsets};
use super::dump::TaskWalkerEntry;
use super::guest::GuestKernel;
use super::idr::translate_any_kva;
use super::reader::{GuestMem, WalkContext};

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

/// Maximum nodes any single rhashtable bucket chain visits before
/// bailing. A healthy rhashtable holds ~1 element per bucket on
/// average; a pathological chain of 1024 entries in one bucket is
/// orders of magnitude beyond legitimate use and almost certainly
/// indicates a corrupted `next` chain or torn read. Bounded
/// independently of [`MAX_RHT_NODES`] so a single runaway bucket
/// cannot starve the walk's per-bucket budget on the way to the
/// global cap. The condition `chain_visited < PER_BUCKET_CHAIN_CAP`
/// admits exactly 1024 body executions: chain_visited starts at 0
/// and increments inside the loop body, so the comparison reads
/// 0,1,...,1023 across the 1024 iterations and exits on the next
/// check (1024 < 1024 is false).
const PER_BUCKET_CHAIN_CAP: u32 = 1024;

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
    /// `rq->scx.kick_sync` — present on post-v7.0-rc5 kernels. None
    /// when the BTF lookup of the field returns absent (v6.14 and
    /// v7.0 release-line layouts predate the `kick_sync` member).
    /// Skipped on serde when None so older dumps stay tight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kick_sync: Option<u64>,
    /// `rq->scx.nr_immed` — count of ENQ_IMMED tasks on local_dsq.
    /// Same kernel-version provenance as [`Self::kick_sync`]: the
    /// field is post-v7.0-rc5 and absent on the v6.14/v7.0 CI matrix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nr_immed: Option<u32>,
    /// `rq->scx.clock` — per-CPU scx_rq clock (the value
    /// `scx_bpf_now()` returns) at the freeze instant. Optional
    /// because the field was added by the `scx_bpf_now()` series in
    /// v6.14 (commit 3a9910b5904d); v6.12 and v6.13 release kernels
    /// have no equivalent member on `struct scx_rq`. None when the
    /// BTF lookup of `rq->scx.clock` resolves absent — consumers
    /// that need the value gate on Some.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rq_clock: Option<u64>,
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
    /// unbounded queue growth.
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
    /// `scx_sched.watchdog_timeout` (jiffies) at the snapshot
    /// instant. `None` when the field was not captured — either
    /// because the live `read_scx_sched_state` path was taken on a
    /// kernel that still exposes `watchdog_timeout` only via the
    /// monitor's `WatchdogOverride` plumbing (not as a BTF field on
    /// every release), or because the BPF .bss fallback was used
    /// without the snapshot var set. Some when populated via the
    /// probe BPF .bss snapshot
    /// (`ktstr_exit_watchdog_timeout`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub watchdog_timeout: Option<u64>,
    /// Provenance tag identifying which path produced this state.
    /// `None` for the default-built / serde-deserialized case where
    /// the source isn't recorded; `Some("live")` when populated by
    /// `read_scx_sched_state` reading `*scx_root` directly;
    /// `Some("bss_snapshot")` when populated from the probe BPF
    /// .bss snapshot fallback (the `ktstr_exit_*` vars). Lets the
    /// dump consumer distinguish "scheduler was alive at freeze
    /// time" from "scheduler had already torn down and we read the
    /// pre-teardown snapshot the BPF probe latched".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Kernel virtual address of the `scx_sched` instance these
    /// values describe. `None` when not captured. Same provenance
    /// rule as [`Self::source`]: live path stamps the resolved
    /// `*scx_root` value; the BPF .bss snapshot stamps the
    /// `ktstr_exit_sched_kva` field. Lets a consumer correlate
    /// dumps across reloads (a different scx_sched instance has a
    /// different KVA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_kva: Option<u64>,
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
    let walk = kernel.walk_context();

    let scx_off = rq_offs.scx;

    // Scalar reads off rq + scx_rq.
    let nr_running = mem.read_u32(rq_pa, scx_off + scx_rq_offs.nr_running);
    let flags = mem.read_u32(rq_pa, scx_off + scx_rq_offs.flags);
    let cpu_released = mem.read_u8(rq_pa, scx_off + scx_rq_offs.cpu_released) != 0;
    let ops_qseq = mem.read_u64(rq_pa, scx_off + scx_rq_offs.ops_qseq);
    // kick_sync / nr_immed are post-v7.0-rc5 fields; offsets resolve
    // as None on v6.14 and v7.0 release-line BTFs. Gate the read on
    // Some so we don't fabricate a u64/u32 from rq_pa+0 (which would
    // alias the local_dsq head pointer — a non-zero garbage read
    // that the dump would render as legitimate kernel state).
    let kick_sync = scx_rq_offs
        .kick_sync
        .map(|off| mem.read_u64(rq_pa, scx_off + off));
    let nr_immed = scx_rq_offs
        .nr_immed
        .map(|off| mem.read_u32(rq_pa, scx_off + off));
    // rq->scx.clock added in v6.14 (commit 3a9910b5904d). Gate the
    // read on Some(off): on v6.12/v6.13 the offset is None and the
    // walker must NOT fall back to rq_pa+0 (would alias local_dsq's
    // raw_spinlock — non-zero junk rendered as a legitimate clock
    // reading). The downstream RqScxState carries an Option<u64> so
    // the JSON elides scx_rq_clock on unsupported kernels.
    let rq_clock = scx_rq_offs
        .clock
        .map(|off| mem.read_u64(rq_pa, scx_off + off));

    // curr task — pointer follow.
    let curr_kva = mem.read_u64(rq_pa, rq_offs.curr);
    let (curr_pid, curr_comm) =
        read_task_pid_comm(mem, walk, curr_kva, task_offs.pid, task_offs.comm);

    // Walk runnable_list when sched_ext_entity offsets are available.
    // Without `see` we can still report scalar state but cannot
    // container_of a runnable_node back to its task_struct.
    let (runnable_task_kvas, runnable_truncated) = if let Some(see_offs) = offsets.see.as_ref() {
        let list_head_off = scx_off + scx_rq_offs.runnable_list;
        let head_kva = rq_kva.wrapping_add(list_head_off as u64);
        let head_pa = rq_pa.wrapping_add(list_head_off as u64);

        // container_of offset within task_struct: each runnable_node
        // is at task + task_struct.scx + see.runnable_node.
        let runnable_node_off_in_task = task_offs.scx + see_offs.runnable_node;

        walk_list_head_for_task_kvas(mem, walk, head_kva, head_pa, runnable_node_off_in_task)
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
///
/// Emits `tracing::debug!` at each gate that returns `None` so an
/// operator parsing the failure-dump trace can pinpoint exactly
/// where the read aborted: BTF sub-group missing, scx_root_kva
/// zero, dereferenced sched_kva zero, or sched_kva translate
/// failure.
#[allow(dead_code)]
pub fn read_scx_sched_state(
    kernel: &GuestKernel<'_>,
    scx_root_kva: u64,
    offsets: &ScxWalkerOffsets,
) -> Option<(u64, ScxSchedState)> {
    let Some(sched_offs) = offsets.sched.as_ref() else {
        tracing::debug!(
            "read_scx_sched_state: ScxSchedOffsets BTF sub-group missing — \
             vmlinux lacks `struct scx_sched` (kernel without sched_ext or stripped vmlinux)",
        );
        return None;
    };

    let mem = kernel.mem();
    let walk = kernel.walk_context();

    if scx_root_kva == 0 {
        tracing::debug!(
            "read_scx_sched_state: scx_root_kva is 0 — vmlinux had no \
             `scx_root` symbol (pre-6.16 kernel or stripped vmlinux)",
        );
        return None;
    }

    let root_pa = kernel.text_kva_to_pa(scx_root_kva);
    let sched_kva = mem.read_u64(root_pa, 0);
    if sched_kva == 0 {
        tracing::debug!(
            scx_root_kva = format_args!("{:#x}", scx_root_kva),
            root_pa = format_args!("{:#x}", root_pa),
            "read_scx_sched_state: *scx_root == 0 — no scheduler attached at the freeze instant",
        );
        return None;
    }
    let Some(sched_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        sched_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        tracing::debug!(
            sched_kva = format_args!("{:#x}", sched_kva),
            "read_scx_sched_state: translate_any_kva failed for sched_kva — \
             page-table walk yielded no PA (slab page race or torn read)",
        );
        return None;
    };

    // `aborting` and `bypass_depth` are dev-only fields (absent on
    // every release tag in our supported range). Gate each read on
    // its offset being Some — falling back to 0 / false matches the
    // semantics of reading from a kernel that never had the field
    // (no in-flight abort, no bypass nesting). The downstream
    // ScxSchedState carries plain bool/i32 because those defaults
    // are meaningful and serializable; an Option wrapper would just
    // complicate every consumer for no extra signal on release
    // kernels.
    let aborting = sched_offs
        .aborting
        .map(|off| mem.read_u8(sched_pa, off) != 0)
        .unwrap_or(false);
    let bypass_depth = sched_offs
        .bypass_depth
        .map(|off| mem.read_u32(sched_pa, off) as i32)
        .unwrap_or(0);
    // `exit_kind` is `atomic_t`; the value lives in the `counter`
    // field at offset 0 of atomic_t. We're already at the
    // outer-struct offset of `exit_kind`, so a u32 read at that
    // offset reads the `counter` directly. Mandatory on every
    // kernel that has `scx_sched`.
    let exit_kind = mem.read_u32(sched_pa, sched_offs.exit_kind);

    Some((
        sched_kva,
        ScxSchedState {
            aborting,
            bypass_depth,
            exit_kind,
            // Live read from `*scx_root` doesn't capture
            // `watchdog_timeout`. The BTF sub-group does not carry
            // an offset for the field today (it would need to be
            // added to `ScxSchedOffsets`), and the host tracks the
            // configured timeout via the `WatchdogOverride` plumbing
            // anyway. Leave None; the BPF .bss snapshot's
            // `ktstr_exit_watchdog_timeout` path populates this when
            // the live read is unavailable.
            watchdog_timeout: None,
            source: Some(SCX_SCHED_STATE_SOURCE_LIVE.to_string()),
            sched_kva: Some(sched_kva),
        },
    ))
}

/// Provenance tag for [`ScxSchedState::source`] when the state was
/// read directly from `*scx_root` via `read_scx_sched_state`. The
/// scheduler was alive at freeze time and the host walked its slab
/// page directly. Pinned as a constant so the dump's display layer
/// and tests reference the same string without drift.
pub const SCX_SCHED_STATE_SOURCE_LIVE: &str = "live";

/// Provenance tag for [`ScxSchedState::source`] when the state was
/// reconstructed from the probe BPF program's `.bss` snapshot
/// (`ktstr_exit_*` vars). The scheduler had already torn down by
/// freeze time (`*scx_root == 0`), so the live walker returned None
/// and the host fell back to the snapshot the BPF tp_btf handler
/// captured at err-exit time.
pub const SCX_SCHED_STATE_SOURCE_BSS: &str = "bss_snapshot";

/// `SCX_TASK_CURSOR` flag value (`1 << 31`) on `sched_ext_entity.flags`.
/// Cursor entries are stack-allocated `sched_ext_entity` placeholders
/// that `scx_task_iter_start` (`kernel/sched/ext.c:843-846`) inserts
/// into `scx_tasks` to mark the iterator's progress; they are NOT
/// embedded in any `task_struct` so the global walker must skip them
/// to avoid container_of producing a bogus task KVA. Pinned per
/// `include/linux/sched/ext.h:142::SCX_TASK_CURSOR`.
const SCX_TASK_CURSOR: u32 = 1 << 31;
/// Walk the kernel's global `scx_tasks` LIST_HEAD and recover every
/// task linked into it via `task_struct.scx.tasks_node`.
///
/// `scx_tasks` is `static LIST_HEAD(scx_tasks)` at
/// `kernel/sched/ext.c:47`. Tasks are added on
/// `scx_init_task` (`kernel/sched/ext.c:3742` —
/// `list_add_tail(&p->scx.tasks_node, &scx_tasks)`) and removed on
/// `sched_ext_dead` (`kernel/sched/ext.c:3803` —
/// `list_del_init(&p->scx.tasks_node)`). The list outlives the
/// per-rq `runnable_list` because `scx_bypass`
/// (`kernel/sched/ext.c:5304-5404`) drains runnable_list during
/// scheduler teardown without touching `scx_tasks` — making this
/// the durable task source for failure-dump enrichment.
///
/// Cursor entries (`scx_task_iter_start` inserts a stack-allocated
/// `sched_ext_entity` with `flags = SCX_TASK_CURSOR` into
/// `scx_tasks` while iterating) are skipped via the
/// `tasks_node_off_in_see` parameter — the walker reads
/// `sched_ext_entity.flags` for each list entry and skips entries
/// whose flag is set.
///
/// `scx_tasks_kva` is the symbol KVA of the global LIST_HEAD;
/// `tasks_node_off_in_task` is the byte offset of `tasks_node`
/// within `task_struct` (`task.scx + see.tasks_node`);
/// `tasks_node_off_in_see` is the byte offset of `tasks_node`
/// within `sched_ext_entity` (`see.tasks_node` alone — used to
/// recover the see base for cursor-flag testing on entries that
/// are not embedded in a `task_struct`); `flags_off_in_see` is the
/// byte offset of `flags` within `sched_ext_entity`.
///
/// Returns an empty vec when `scx_tasks_kva` is 0 (symbol absent —
/// stripped vmlinux or kernel without sched_ext) or when the list
/// head reads as empty (tasks_node points at itself).
///
/// Bounded by [`MAX_NODES_PER_LIST`] to protect against a corrupt
/// chain.
#[allow(dead_code)]
pub fn walk_scx_tasks_global(
    kernel: &GuestKernel<'_>,
    scx_tasks_kva: u64,
    tasks_node_off_in_task: usize,
    tasks_node_off_in_see: usize,
    flags_off_in_see: usize,
) -> Vec<u64> {
    if scx_tasks_kva == 0 {
        tracing::debug!(
            "walk_scx_tasks_global: scx_tasks_kva is 0 — vmlinux had no \
             `scx_tasks` symbol (kernel without sched_ext or stripped vmlinux)",
        );
        return Vec::new();
    }
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    // The LIST_HEAD lives in the kernel text/.data mapping; convert
    // KVA → PA via the GuestKernel's runtime kernel image base. The
    // first u64 at that PA is list_head.next (the LIST_HEAD struct's
    // first field).
    let head_kva = scx_tasks_kva;
    let head_pa = kernel.text_kva_to_pa(scx_tasks_kva);

    let mut task_kvas: Vec<u64> = Vec::new();
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        tracing::debug!(
            scx_tasks_kva = format_args!("{:#x}", scx_tasks_kva),
            head_pa = format_args!("{:#x}", head_pa),
            "walk_scx_tasks_global: head.next read as 0 — list-head bytes \
             unmapped or torn read; no tasks harvested",
        );
        return task_kvas;
    }

    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_LIST {
            return task_kvas;
        }
        visited += 1;

        // Recover the sched_ext_entity base for this list entry so we
        // can read its `flags`. For task-embedded entries this base is
        // inside a task_struct (`task_kva + task.scx`); for cursor
        // entries this base is a stack-allocated sched_ext_entity.
        // Either way, `see_kva = node_kva - see.tasks_node`.
        let see_kva = node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        let cursor = match translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            see_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(see_pa) => {
                let flags = mem.read_u32(see_pa, flags_off_in_see);
                flags & SCX_TASK_CURSOR != 0
            }
            // Translate failure on the see base — be conservative and
            // treat as not-cursor so the entry surfaces; downstream
            // walk_task_enrichment will revalidate via translate and
            // drop it cleanly if the address is bogus.
            None => false,
        };

        if !cursor {
            // container_of: task_kva = node_kva - tasks_node_off_in_task.
            let task_kva = node_kva.wrapping_sub(tasks_node_off_in_task as u64);
            task_kvas.push(task_kva);
        }

        // Advance to the next node via the list_head.next pointer
        // at offset 0 of the tasks_node list_head.
        let Some(node_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
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

/// Walk every per-CPU local DSQ — the DSQs embedded in `rq->scx.local_dsq`.
///
/// This is a strict subset of [`walk_dsqs`]'s pass 1, extracted so
/// the dump path can call it INDEPENDENTLY of `*scx_root`. Per-CPU
/// local DSQs are kernel-initialized at boot (`init_dsq` from
/// `kernel/sched/ext.c:7772`, called for every possible CPU in the
/// `__init` path), so they exist even when no scheduler is attached
/// (`*scx_root == NULL`) and survive scheduler teardown's bypass
/// drain.
///
/// Returns one `DsqState` per CPU whose translate succeeds, plus a
/// flat vec of [`TaskWalkerEntry`] for the per-task enrichment
/// pipeline (these entries carry `is_runnable_in_scx: false` —
/// tasks queued on a DSQ are staged for dispatch, not yet runnable
/// in the rq->scx sense).
///
/// `rq_kvas`, `rq_pas`, and `per_cpu_offsets` index by CPU id
/// (parallel arrays, same shape `walk_rq_scx` consumes). The walker
/// skips BSS-zero-tail CPUs by checking the per-CPU offset directly
/// (`per_cpu_offsets[cpu] == 0 && cpu > 0`) — those entries fall out
/// of un-written `__per_cpu_offset[]` slots past `nr_cpu_ids` and
/// would otherwise surface a phantom DSQ row at the bare `runqueues`
/// symbol KVA. Comparing the resolved `rq_kva` against `rq_kvas[0]`
/// would miss the alias on x86_64 SMP: `setup_per_cpu_areas`
/// (`arch/x86/kernel/setup_percpu.c`) writes
/// `__per_cpu_offset[cpu] = delta + pcpu_unit_offsets[cpu]` with
/// `delta = pcpu_base_addr - __per_cpu_start` non-zero, so CPU 0's
/// `rq_kva` is `runqueues + delta` while a BSS-zero-tail CPU's is
/// `runqueues + 0` — the two differ and `rq_kva == rq_kvas[0]` would
/// let the phantom row through. Mirrors the canonical `cpu_off == 0
/// && cpu_index > 0` guard
/// [`super::bpf_map::read_percpu_array_value`] applies for percpu
/// reads, expressed against the same `__per_cpu_offset[]` array.
///
/// Empty arrays mean "no CPUs walked successfully"; the caller's
/// freeze-path retry guard normally rejects empty inputs before
/// reaching this pass.
///
/// `None` return when any required offset sub-group is missing
/// (`rq`, `scx_rq`, `dsq`, `dsq_lnode`, `task`, `see` — the same
/// leaf set [`walk_dsqs`]'s pass 1 needs). A partial offset set
/// is the same gating condition that blinds every other DSQ pass.
#[allow(dead_code)]
pub fn walk_local_dsqs(
    kernel: &GuestKernel<'_>,
    rq_kvas: &[u64],
    rq_pas: &[u64],
    per_cpu_offsets: &[u64],
    offsets: &ScxWalkerOffsets,
) -> Option<(Vec<DsqState>, Vec<TaskWalkerEntry>)> {
    let Some(rq_offs) = offsets.rq.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.rq sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(scx_rq_offs) = offsets.scx_rq.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.scx_rq sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(dsq_offs) = offsets.dsq.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.dsq sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(dsq_lnode_offs) = offsets.dsq_lnode.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.dsq_lnode sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(task_offs) = offsets.task.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.task sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };
    let Some(see_offs) = offsets.see.as_ref() else {
        tracing::debug!(
            "walk_local_dsqs: ScxWalkerOffsets.see sub-group missing — \
             local DSQ pass blinded",
        );
        return None;
    };

    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let mut states: Vec<DsqState> = Vec::new();
    let mut entries: Vec<TaskWalkerEntry> = Vec::new();

    for (cpu, (&rq_kva, &rq_pa)) in rq_kvas.iter().zip(rq_pas.iter()).enumerate() {
        // BSS-zero-tail guard: kernel `setup_per_cpu_areas`
        // only writes `__per_cpu_offset[cpu]` for CPUs in
        // `for_each_possible_cpu`, leaving slots beyond
        // `nr_cpu_ids` at the BSS-initialized 0. The caller
        // builds rq_kvas via `runqueues + per_cpu_offset[cpu]`,
        // so a BSS-zero-tail entry produces an rq_kva of
        // `runqueues + 0` instead of CPU 0's
        // `runqueues + __per_cpu_offset[0]`. The two are NOT
        // equal on x86_64 SMP because
        // `__per_cpu_offset[0] = pcpu_base_addr - __per_cpu_start`
        // is non-zero (`arch/x86/kernel/setup_percpu.c`); a
        // resolved-rq_kva comparison would let the phantom
        // BSS-zero entry through. Check the per-CPU offset
        // directly instead — `cpu_off == 0 && cpu > 0` is the
        // canonical guard
        // [`super::bpf_map::read_percpu_array_value`] uses for
        // percpu reads and the matching guard at the per-CPU
        // bypass DSQ pass below. A `per_cpu_offsets` slice
        // shorter than `rq_kvas` (length-mismatched caller)
        // is treated conservatively: an absent offset for
        // `cpu > 0` skips the slot, since the walker can't
        // distinguish a real CPU from a BSS-zero tail without
        // the offset.
        let cpu_off = per_cpu_offsets.get(cpu).copied();
        match cpu_off {
            Some(off) if off == 0 && cpu > 0 => continue,
            None if cpu > 0 => continue,
            _ => {}
        }
        let local_dsq_off = rq_offs.scx + scx_rq_offs.local_dsq;
        let dsq_kva = rq_kva.wrapping_add(local_dsq_off as u64);
        let dsq_pa = rq_pa.wrapping_add(local_dsq_off as u64);
        if let Some((state, e)) = walk_one_dsq(
            mem,
            walk,
            dsq_kva,
            dsq_pa,
            || format!("local cpu {cpu}"),
            dsq_offs,
            dsq_lnode_offs,
            task_offs,
            see_offs,
        ) {
            entries.extend(e);
            states.push(state);
        }
    }

    Some((states, entries))
}

/// Walk every DSQ reachable from a `scx_sched` (the bypass / global
/// / user-hash passes — NOT per-CPU local DSQs) and produce one
/// `DsqState` per DSQ plus a flat vec of `TaskWalkerEntry` rows for
/// the per-task enrichment pipeline.
///
/// Per-CPU local DSQs (`rq->scx.local_dsq`) are NOT walked here —
/// they live in each rq independently of `*scx_root`, so callers
/// invoke [`walk_local_dsqs`] separately and unconditionally for
/// the local pass. This split lets the dump path surface local DSQ
/// state even when no scheduler is attached
/// (`*scx_root == NULL`) — the local_dsq struct is initialized at
/// boot per `init_dsq` (`kernel/sched/ext.c:7772`) for every
/// possible CPU, so it has well-defined contents long before any
/// scheduler attaches.
///
/// Walks (in this order, gated on the relevant sub-group offsets
/// being present):
///   1. Per-CPU bypass DSQs at `scx_sched_pcpu.bypass_dsq` for
///      every CPU (needs `sched`, `sched_pcpu`, plus the leaf set).
///   2. Per-node global DSQs at `scx_sched.pnode[node]->global_dsq`
///      for every NUMA node (needs `sched`, `sched_pnode`, plus
///      leaf set).
///   3. User-allocated DSQs walked through `scx_sched.dsq_hash`
///      (needs `sched`, `rht`, plus leaf set).
///
/// Each pass is independent: missing offsets for one pass blind
/// only that pass. A translate failure on one DSQ leaves it out of
/// the result without affecting the others.
#[allow(dead_code)]
pub fn walk_dsqs(
    kernel: &GuestKernel<'_>,
    sched_pa: u64,
    per_cpu_offsets: &[u64],
    nr_nodes: u32,
    offsets: &ScxWalkerOffsets,
) -> (Vec<DsqState>, Vec<TaskWalkerEntry>) {
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let mut dsq_states: Vec<DsqState> = Vec::new();
    let mut all_entries: Vec<TaskWalkerEntry> = Vec::new();

    // Leaf offsets common to every pass — all three DSQ-walking
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

    // Pass 1: per-CPU bypass DSQs. The percpu base lives at
    // sched->pcpu, dereferenced as a __percpu pointer; each CPU's
    // address is `pcpu_base + per_cpu_offset[cpu] +
    // scx_sched_pcpu.bypass_dsq`.
    //
    // Both `sched_offs.pcpu` (v6.18+) and `pcpu_offs.bypass_dsq`
    // (dev-only) are kernel-version-gated. Skip the entire pass
    // unless both offsets resolved — partial state would compute
    // a bogus DSQ KVA from `sched_pa + 0` (aliasing dsq_hash) and
    // surface phantom DSQ entries.
    if let (Some(sched_offs), Some(pcpu_offs)) =
        (offsets.sched.as_ref(), offsets.sched_pcpu.as_ref())
        && let (Some(sched_pcpu_off), Some(bypass_dsq_off)) =
            (sched_offs.pcpu, pcpu_offs.bypass_dsq)
    {
        let pcpu_kva = mem.read_u64(sched_pa, sched_pcpu_off);
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
                    .wrapping_add(bypass_dsq_off as u64);
                if let Some(dsq_pa) = translate_any_kva(
                    mem,
                    walk.cr3_pa,
                    walk.page_offset,
                    dsq_kva,
                    walk.l5,
                    walk.tcr_el1,
                ) && let Some((state, entries)) = walk_one_dsq(
                    mem,
                    walk,
                    dsq_kva,
                    dsq_pa,
                    || format!("bypass cpu {cpu}"),
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

    // Pass 2: per-node global DSQs. `sched->pnode` is a pointer
    // to an array of `struct scx_sched_pnode *` of length nr_nodes.
    // Both `sched_offs.pnode` and `pnode_offs.global_dsq` are
    // dev-only — skip the pass unless both resolved.
    if let (Some(sched_offs), Some(pnode_offs)) =
        (offsets.sched.as_ref(), offsets.sched_pnode.as_ref())
        && let (Some(sched_pnode_off), Some(global_dsq_off)) =
            (sched_offs.pnode, pnode_offs.global_dsq)
    {
        let pnode_kva = mem.read_u64(sched_pa, sched_pnode_off);
        if pnode_kva != 0
            && let Some(pnode_arr_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                pnode_kva,
                walk.l5,
                walk.tcr_el1,
            )
        {
            for node in 0..nr_nodes as u64 {
                let pnode_ptr_kva = mem.read_u64(pnode_arr_pa, (node * 8) as usize);
                if pnode_ptr_kva == 0 {
                    continue;
                }
                let Some(pnode_pa) = translate_any_kva(
                    mem,
                    walk.cr3_pa,
                    walk.page_offset,
                    pnode_ptr_kva,
                    walk.l5,
                    walk.tcr_el1,
                ) else {
                    continue;
                };
                let dsq_kva = pnode_ptr_kva.wrapping_add(global_dsq_off as u64);
                let dsq_pa = pnode_pa.wrapping_add(global_dsq_off as u64);
                if let Some((state, entries)) = walk_one_dsq(
                    mem,
                    walk,
                    dsq_kva,
                    dsq_pa,
                    || format!("global node {node}"),
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

    // Pass 3: user-allocated DSQs via the scx_sched.dsq_hash
    // rhashtable. Walks at most MAX_RHT_NODES nodes total across
    // all buckets.
    if let (Some(sched_offs), Some(rht_offs)) = (offsets.sched.as_ref(), offsets.rht.as_ref()) {
        let rht_kva = sched_pa.wrapping_add(sched_offs.dsq_hash as u64);
        // dsq_hash is embedded in scx_sched (not a pointer); rht_kva
        // here is a KVA we can translate directly. The walker reads
        // it via the rht sub-group offsets.
        let (user_dsqs, user_dsqs_truncated) =
            walk_user_dsq_hash(mem, walk, rht_kva, rht_offs, dsq_offs);
        if user_dsqs_truncated {
            // Surface the cap-hit so an operator parsing the
            // failure dump trace sees that the user-DSQ list is
            // partial. Without this log the dump silently
            // omits the tail of the dsq_hash bucket table or
            // the tail of one bucket's chain.
            tracing::warn!(
                visited = user_dsqs.len(),
                cap_buckets = MAX_RHT_BUCKETS,
                cap_nodes = MAX_RHT_NODES,
                "walk_user_dsq_hash: truncated — bucket-table or node cap fired; \
                 dsq_kvas list is incomplete",
            );
        }
        for dsq_kva in user_dsqs {
            let Some(dsq_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                dsq_kva,
                walk.l5,
                walk.tcr_el1,
            ) else {
                continue;
            };
            if let Some((state, entries)) = walk_one_dsq(
                mem,
                walk,
                dsq_kva,
                dsq_pa,
                || "user".to_string(),
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
///
/// `origin` is taken as a `FnOnce` closure so the per-call
/// `format!("local cpu {cpu}")` / `format!("bypass cpu {cpu}")` /
/// `format!("global node {node}")` heap allocation only fires
/// after the `dsq_pa == 0` early-out has been cleared. Eagerly
/// formatting at every caller wasted one short-string allocation
/// per skipped DSQ on every freeze.
///
/// Returns `None` when `dsq_pa == 0`. Reading at PA 0 would
/// surface the boot-page contents as DSQ scalars (`id`, `nr`,
/// `seq`) and an all-zero list-head as an apparently-empty queue
/// — indistinguishable from a real empty DSQ. The early check
/// rejects that case so the caller does not push a phantom
/// DsqState row built from PA-0 garbage.
#[allow(clippy::too_many_arguments)]
fn walk_one_dsq(
    mem: &GuestMem,
    walk: WalkContext,
    dsq_kva: u64,
    dsq_pa: u64,
    origin: impl FnOnce() -> String,
    dsq_offs: &super::btf_offsets::ScxDispatchQOffsets,
    dsq_lnode_offs: &super::btf_offsets::ScxDsqListNodeOffsets,
    task_offs: &super::btf_offsets::TaskStructCoreOffsets,
    see_offs: &super::btf_offsets::SchedExtEntityOffsets,
) -> Option<(DsqState, Vec<TaskWalkerEntry>)> {
    if dsq_pa == 0 {
        tracing::debug!(
            dsq_kva = format_args!("{:#x}", dsq_kva),
            "walk_one_dsq: dsq_pa == 0 — would alias the boot page; \
             skipping to avoid surfacing phantom all-zero DSQ state",
        );
        return None;
    }
    let origin = origin();
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
    let dsq_node_off_in_task = task_offs.scx + see_offs.dsq_list + dsq_lnode_offs.node;

    let (task_kvas, truncated) = walk_list_head_for_dsq_task_kvas(
        mem,
        walk,
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
    walk: WalkContext,
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
        let Some(node_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
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
    walk: WalkContext,
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
        // is_cursor: Some(true) → cursor (skip), Some(false) → real
        // task entry (push), None → translate failed and we cannot
        // distinguish. When the cursor flag cannot be read, treat
        // the entry as a cursor and skip it: pushing it would
        // record a phantom `task_kva = node_kva - dsq_node_off_in_task`
        // built from a node whose enclosing sched_ext_entity isn't
        // mappable, which downstream task enrichment would surface
        // as bogus pid/comm reads at an arbitrary address.
        let is_cursor = match translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            lnode_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(lnode_pa) => {
                let lnode_flags = mem.read_u32(lnode_pa, dsq_lnode_offs.flags);
                Some(lnode_flags & SCX_DSQ_LNODE_ITER_CURSOR != 0)
            }
            None => None,
        };

        let skip_entry = match is_cursor {
            Some(true) => true,  // cursor entry — advance without recording
            Some(false) => false, // real task entry — push and advance
            None => true,        // cursor-detection unreliable — skip rather than push bogus
        };

        if !skip_entry {
            // Real task entry: container_of from the inner list_head's
            // node_kva back to task_struct. The full offset within
            // task_struct is task.scx + see.dsq_list + dsq_lnode.node.
            let task_kva = node_kva.wrapping_sub(dsq_node_off_in_task as u64);
            task_kvas.push(task_kva);
        }

        // Advance to the next node. The list_head.next pointer
        // lives at offset 0 of the inner list_head we landed on,
        // which is `node_kva` itself.
        let Some(node_pa) = translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) else {
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
/// - per-bucket chain length at [`PER_BUCKET_CHAIN_CAP`]
///
/// Returns `(dsq_kvas, truncated)`. `truncated` is `true` when any
/// cap fired before the walk could reach the natural end of every
/// bucket chain — either `bucket_table.size > MAX_RHT_BUCKETS`,
/// `total_nodes >= MAX_RHT_NODES` mid-walk, or a per-bucket chain
/// reached its `PER_BUCKET_CHAIN_CAP` cap. Without this signal,
/// callers cannot distinguish "small DSQ count" from "cap silently
/// dropped tail entries" — see DsqState.truncated for the same
/// pattern on per-DSQ task lists.
fn walk_user_dsq_hash(
    mem: &GuestMem,
    walk: WalkContext,
    rht_kva: u64,
    rht_offs: &super::btf_offsets::RhashtableOffsets,
    dsq_offs: &super::btf_offsets::ScxDispatchQOffsets,
) -> (Vec<u64>, bool) {
    let mut dsq_kvas = Vec::new();

    let Some(rht_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        rht_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        return (dsq_kvas, false);
    };

    let tbl_kva = mem.read_u64(rht_pa, rht_offs.tbl);
    if tbl_kva == 0 {
        return (dsq_kvas, false);
    }
    let Some(tbl_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        tbl_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
        return (dsq_kvas, false);
    };

    let size = mem.read_u32(tbl_pa, rht_offs.bucket_table_size);
    let bucket_count = size.min(MAX_RHT_BUCKETS) as u64;
    // A bucket_table.size larger than the bucket cap means we'll
    // walk only the first MAX_RHT_BUCKETS buckets and the tail is
    // silently dropped. Surface that as truncation up front.
    let mut truncated = size as u64 > bucket_count;
    let buckets_off = rht_offs.bucket_table_buckets;

    let mut total_nodes: u32 = 0;
    for i in 0..bucket_count {
        if total_nodes >= MAX_RHT_NODES {
            // Hit the global node cap — remaining buckets unwalked.
            return (dsq_kvas, true);
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
        let mut chain_terminated_naturally = false;
        while node_kva != 0
            && total_nodes < MAX_RHT_NODES
            && chain_visited < PER_BUCKET_CHAIN_CAP
        {
            chain_visited += 1;
            total_nodes += 1;
            let dsq_kva = node_kva.wrapping_sub(dsq_offs.hash_node as u64);
            dsq_kvas.push(dsq_kva);
            let Some(node_pa) = translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                node_kva,
                walk.l5,
                walk.tcr_el1,
            ) else {
                // Translate failure — chain ended for this bucket
                // without hitting a cap. Not a truncation signal.
                chain_terminated_naturally = true;
                break;
            };
            let next_raw = mem.read_u64(node_pa, rht_offs.rhash_head_next);
            // The chain terminator is a "nulls" pointer with bit 0
            // set encoding the bucket index; treat any LSB-tagged
            // pointer as terminator.
            if next_raw & RHT_PTR_LOCK_BIT != 0 || next_raw == 0 {
                chain_terminated_naturally = true;
                break;
            }
            node_kva = next_raw;
        }
        // The loop exited; if it wasn't via a natural terminator
        // (LSB-tagged pointer / NULL / translate failure), one of
        // the two caps fired (chain_visited >= PER_BUCKET_CHAIN_CAP
        // or total_nodes >= MAX_RHT_NODES) and we silently dropped
        // the rest of this bucket's chain.
        if !chain_terminated_naturally {
            truncated = true;
        }
    }

    (dsq_kvas, truncated)
}

/// Read `(pid, comm)` for a `task_struct *` after a NULL-check and
/// translate. Returns `(None, None)` on NULL or untranslatable.
fn read_task_pid_comm(
    mem: &GuestMem,
    walk: WalkContext,
    task_kva: u64,
    pid_off: usize,
    comm_off: usize,
) -> (Option<i32>, Option<String>) {
    if task_kva == 0 {
        return (None, None);
    }
    let Some(task_pa) = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    ) else {
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
            kick_sync: Some(50),
            nr_immed: None,
            rq_clock: Some(1234567),
            curr_pid: None,
            curr_comm: None,
            runnable_task_kvas: vec![],
            runnable_truncated: false,
        };
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("curr_pid"));
        assert!(!json.contains("curr_comm"));
        assert!(!json.contains("runnable_truncated"));
        // nr_immed is None — must skip via skip_serializing_if so
        // dumps from v6.14/v7.0 (no kick_sync / nr_immed fields)
        // stay tight.
        assert!(!json.contains("nr_immed"));
        // kick_sync is Some — must serialize the inner value, not
        // a `{ "Some": ... }` shape (Option<T> with default serde
        // serializes the wrapped value bare).
        assert!(json.contains("\"kick_sync\":50"));
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
            kick_sync: Some(17),
            nr_immed: Some(1),
            rq_clock: Some(999_999),
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
        // kick_sync / nr_immed are now Option<…>; the populated test
        // fixture sets both Some(…), so equality on the unwrapped
        // shape is the same correctness check the prior assertions
        // gave on the bare types.
        let kick_sync_match = parsed_kick_sync == Some(17u64);
        let nr_immed_match = parsed_nr_immed == Some(1u32);
        let rq_clock_match = parsed_rq_clock == Some(999_999u64);
        crate::claim!(v, kick_sync_match).eq(true);
        crate::claim!(v, nr_immed_match).eq(true);
        crate::claim!(v, rq_clock_match).eq(true);
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
            ..Default::default()
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
            &mem,
            WalkContext::default(),
            head as u64,
            head as u64,
            runnable_node_off,
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

        let (kvas, truncated) = walk_list_head_for_task_kvas(
            &mem,
            WalkContext::default(),
            head as u64,
            head as u64,
            0x10,
        );
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
        let (kvas, truncated) = walk_list_head_for_task_kvas(
            &mem,
            WalkContext::default(),
            head as u64,
            head as u64,
            0x10,
        );
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
                kick_sync: Some(120),
                nr_immed: Some(128),
                clock: Some(136),
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
                tasks_node: 88,
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
                pnode: Some(64),
                pcpu: Some(72),
                aborting: Some(80),
                bypass_depth: Some(84),
                exit_kind: 88,
            }),
            sched_pnode: Some(ScxSchedPnodeOffsets {
                global_dsq: Some(0),
            }),
            sched_pcpu: Some(ScxSchedPcpuOffsets {
                bypass_dsq: Some(0),
            }),
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
            kick_sync: Some(100),
            nr_immed: Some(0),
            rq_clock: Some(999_999),
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
        v.claim_seq("runnable_task_kvas", &s.runnable_task_kvas)
            .nonempty();
        v.claim_seq("runnable_task_kvas", &s.runnable_task_kvas)
            .len_at_most(64);

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
            kick_sync: None,
            nr_immed: None,
            rq_clock: None,
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
            ..Default::default()
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
            ..Default::default()
        };
        let mut v = Verdict::new();
        crate::claim!(v, aborted.exit_kind).eq(0u32);
        let r = v.into_result();
        assert!(!r.passed, "exit_kind=1027 must fail eq(0)");
    }

    /// `walk_scx_tasks_global` returns an empty vec when the
    /// `scx_tasks` symbol KVA is 0 — kernel without sched_ext or
    /// stripped vmlinux. The walk must NOT attempt to read at PA 0
    /// (which would alias the boot-page region and surface bogus
    /// task entries).
    #[test]
    fn walk_scx_tasks_global_zero_kva_returns_empty() {
        let mut buf = vec![0u8; 0x1000];
        // Pre-populate buf at offset 0 to make the difference visible:
        // a buggy implementation that read from PA 0 would surface
        // 0xdead_beef as a task_kva (after container_of subtraction).
        buf[0..8].copy_from_slice(&0xdead_beef_u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(&kernel, 0, 0x10, 0x60, 0x44);
        assert!(
            kvas.is_empty(),
            "scx_tasks_kva=0 must short-circuit before any read"
        );
    }

    /// `walk_scx_tasks_global` walks an empty global list (head.next
    /// points back at the head itself — kernel's empty-list
    /// invariant). Walker returns no task KVAs.
    #[test]
    fn walk_scx_tasks_global_empty_list_returns_empty() {
        // page_offset = 0 makes the GuestKernel's text_kva_to_pa
        // return KVA itself for KVAs >= __START_KERNEL_map. The KVA
        // we choose is in the text mapping range so the translation
        // lands at a sensible offset within our test buffer.
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = head_kva.wrapping_sub(crate::monitor::symbols::START_KERNEL_MAP) as usize;
        let mut buf = vec![0u8; 0x1000];
        // head.next = head_kva (empty list invariant)
        buf[head_pa..head_pa + 8].copy_from_slice(&head_kva.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0, // page_offset = 0; kva_to_pa identity
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(&kernel, head_kva, 0x10, 0x60, 0x44);
        assert!(kvas.is_empty(), "empty global list must yield no tasks");
    }

    /// `walk_scx_tasks_global` recovers task KVAs via
    /// `task_kva = node_kva - tasks_node_off_in_task`. Two-task list
    /// with the head in the kernel text mapping; the per-task
    /// see/tasks_node lives in a directly-mapped region. Verifies
    /// the container_of math against the kernel's container_of
    /// pattern.
    #[test]
    fn walk_scx_tasks_global_two_tasks_round_trip() {
        // Layout (page_offset = 0 so direct-map kva == pa for the
        // task entries; head lives in the text mapping region so
        // text_kva_to_pa_with_base reaches the buffer):
        //   head_kva = START_KERNEL_MAP + 0x100   → head_pa = 0x100
        //   t1_node_kva = 0x800                   → t1_pa = 0x800
        //   t2_node_kva = 0x900                   → t2_pa = 0x900
        // tasks_node_off_in_task = 0x40 (so task_kva = node_kva - 0x40).
        // Linkage:
        //   head.next = t1_node_kva
        //   t1.next   = t2_node_kva
        //   t2.next   = head_kva (close the list)
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + 0x100;
        let head_pa = 0x100usize;
        let t1_node_kva: u64 = 0x800;
        let t2_node_kva: u64 = 0x900;
        let tasks_node_off_in_task: usize = 0x40;
        let tasks_node_off_in_see: usize = 0x60;
        let flags_off_in_see: usize = 0x44;

        let mut buf = vec![0u8; 0x1000];
        buf[head_pa..head_pa + 8].copy_from_slice(&t1_node_kva.to_le_bytes());
        let t1_pa = t1_node_kva as usize;
        let t2_pa = t2_node_kva as usize;
        buf[t1_pa..t1_pa + 8].copy_from_slice(&t2_node_kva.to_le_bytes());
        buf[t2_pa..t2_pa + 8].copy_from_slice(&head_kva.to_le_bytes());

        // Both task entries are NOT cursors. Their see.flags slot
        // stays zero (the buf is zero-initialized) so the walker's
        // cursor-flag check passes through. The flags slot for each
        // entry sits at `see_kva + flags_off_in_see` =
        // `(node_kva - tasks_node_off_in_see) + flags_off_in_see`.
        // For t1: see_kva = 0x800 - 0x60 = 0x7a0 → flags @ 0x7a0+0x44=0x7e4.
        // For t2: see_kva = 0x900 - 0x60 = 0x8a0 → flags @ 0x8a0+0x44=0x8e4.
        // Both already 0 from buf init, so the cursor bit is unset.

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );
        assert_eq!(kvas.len(), 2, "two-task list must yield two task kvas");
        // container_of: task_kva = node_kva - tasks_node_off_in_task.
        assert_eq!(
            kvas[0],
            t1_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
        assert_eq!(
            kvas[1],
            t2_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
    }

    /// `walk_local_dsqs` returns `None` when any required offset
    /// sub-group is missing — the gate must NOT fabricate partial
    /// state when offsets are incomplete.
    #[test]
    fn walk_local_dsqs_none_when_offsets_missing() {
        let mut buf = vec![0u8; 0x1000];
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = ScxWalkerOffsets {
            rq: None, // missing → walk_local_dsqs gates to None
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

        let r = walk_local_dsqs(&kernel, &[], &[], &[], &offsets);
        assert!(r.is_none(), "missing offsets must gate to None");
    }

    /// `walk_local_dsqs` runs unconditionally — even when
    /// `*scx_root` would be 0 (no scheduler attached). With a
    /// well-formed empty per-CPU local_dsq fixture, the walker
    /// returns `Some(([DsqState{empty list}], []))` for each CPU.
    /// Confirms the new dump-path independence: the local-DSQ
    /// pass surfaces every CPU's DSQ state regardless of
    /// scheduler attachment.
    #[test]
    fn walk_local_dsqs_runs_without_scheduler() {
        // Layout: one CPU. rq fixture lives at PA 0x100 (page_offset=0,
        // identity translation). scx_rq embedded at offset 0; the
        // scx_dispatch_q within scx_rq.local_dsq lives at offset 0
        // of the rq (rq.scx + scx_rq.local_dsq = 0). The DSQ's
        // list_head sits at dsq + dsq.list = 0 + 0 = 0. An empty
        // list means head.next == head_kva.
        let rq_kva: u64 = 0x100;
        let rq_pa: u64 = 0x100;
        let mut buf = vec![0u8; 0x1000];
        // head.next = rq_kva (empty list)
        buf[rq_pa as usize..rq_pa as usize + 8].copy_from_slice(&rq_kva.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = ScxWalkerOffsets {
            rq: Some(crate::monitor::btf_offsets::RqStructOffsets { scx: 0, curr: 8 }),
            scx_rq: Some(crate::monitor::btf_offsets::ScxRqOffsets {
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
            task: Some(crate::monitor::btf_offsets::TaskStructCoreOffsets {
                comm: 100,
                pid: 200,
                scx: 0,
            }),
            see: Some(crate::monitor::btf_offsets::SchedExtEntityOffsets {
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
            dsq_lnode: Some(crate::monitor::btf_offsets::ScxDsqListNodeOffsets {
                node: 0,
                flags: 16,
            }),
            dsq: Some(crate::monitor::btf_offsets::ScxDispatchQOffsets {
                list: 0,
                nr: 16,
                seq: 20,
                id: 24,
                hash_node: 32,
            }),
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };

        // Single-CPU per_cpu_offsets: cpu 0 has any offset (BSP can
        // legitimately be 0 — only `cpu_off == 0 && cpu > 0` triggers
        // the BSS-zero-tail skip).
        let (states, entries) = walk_local_dsqs(&kernel, &[rq_kva], &[rq_pa], &[0], &offsets)
            .expect("offsets present, should yield Some");
        assert_eq!(states.len(), 1, "one CPU → one DSQ state");
        assert_eq!(states[0].origin, "local cpu 0");
        // Empty list → no entries.
        assert!(entries.is_empty());
    }

    /// `walk_scx_tasks_global` skips cursor entries — list nodes
    /// whose enclosing `sched_ext_entity.flags` has `SCX_TASK_CURSOR`
    /// (1<<31) set. Inserts a cursor BETWEEN two real task entries
    /// and asserts the cursor's container_of result is NOT in the
    /// returned vec, but both real tasks are.
    #[test]
    fn walk_scx_tasks_global_skips_cursor_entries() {
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
        let t1_pa = t1_node_kva as usize;
        let cursor_pa = cursor_node_kva as usize;
        let t2_pa = t2_node_kva as usize;
        buf[t1_pa..t1_pa + 8].copy_from_slice(&cursor_node_kva.to_le_bytes());
        buf[cursor_pa..cursor_pa + 8].copy_from_slice(&t2_node_kva.to_le_bytes());
        buf[t2_pa..t2_pa + 8].copy_from_slice(&head_kva.to_le_bytes());

        // Stamp SCX_TASK_CURSOR (1<<31) into the cursor entry's
        // sched_ext_entity.flags. flags slot lives at
        // (cursor_node_kva - tasks_node_off_in_see) + flags_off_in_see.
        let cursor_see_kva = cursor_node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        let cursor_flags_pa = (cursor_see_kva as usize).wrapping_add(flags_off_in_see);
        let cursor_flags: u32 = 1 << 31;
        buf[cursor_flags_pa..cursor_flags_pa + 4].copy_from_slice(&cursor_flags.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = crate::monitor::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let kvas = walk_scx_tasks_global(
            &kernel,
            head_kva,
            tasks_node_off_in_task,
            tasks_node_off_in_see,
            flags_off_in_see,
        );
        assert_eq!(
            kvas.len(),
            2,
            "cursor entry must be filtered; only 2 real tasks remain"
        );
        let cursor_task_kva = cursor_node_kva.wrapping_sub(tasks_node_off_in_task as u64);
        assert!(
            !kvas.contains(&cursor_task_kva),
            "cursor's container_of result must NOT appear in the task list"
        );
        assert_eq!(
            kvas[0],
            t1_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
        assert_eq!(
            kvas[1],
            t2_node_kva.wrapping_sub(tasks_node_off_in_task as u64)
        );
    }

    // ---------------------------------------------------------------
    // walk_dsqs partial-pass + read_scx_sched_state degradation tests
    //
    // The fix for dsq=0 / sched=absent requires that every walker
    // produces what data IT can, even when sibling walkers can't run.
    // Pre-fix, a single missing offset blinded the whole DSQ surface;
    // the contract these tests pin is "each pass independent — missing
    // offsets for one pass blind only that pass."
    // ---------------------------------------------------------------

    /// Build a fully-populated `ScxWalkerOffsets` for DSQ walker
    /// fixtures. All leaf groups present so walk_dsqs's outer
    /// short-circuit doesn't fire.
    fn dsq_test_offsets() -> ScxWalkerOffsets {
        use super::super::btf_offsets::{
            RhashtableOffsets, RqStructOffsets, SchedExtEntityOffsets, ScxDispatchQOffsets,
            ScxDsqListNodeOffsets, ScxRqOffsets, ScxSchedOffsets, ScxSchedPcpuOffsets,
            ScxSchedPnodeOffsets, TaskStructCoreOffsets,
        };
        ScxWalkerOffsets {
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
            dsq_lnode: Some(ScxDsqListNodeOffsets { node: 0, flags: 16 }),
            dsq: Some(ScxDispatchQOffsets {
                list: 0,
                nr: 16,
                seq: 20,
                id: 24,
                hash_node: 32,
            }),
            sched: Some(ScxSchedOffsets {
                dsq_hash: 0x40,
                pnode: Some(0x80),
                pcpu: Some(0x88),
                aborting: Some(0x90),
                bypass_depth: Some(0x94),
                exit_kind: 0x98,
            }),
            sched_pnode: Some(ScxSchedPnodeOffsets {
                global_dsq: Some(0),
            }),
            sched_pcpu: Some(ScxSchedPcpuOffsets {
                bypass_dsq: Some(0),
            }),
            rht: Some(RhashtableOffsets {
                tbl: 0,
                nelems: 8,
                bucket_table_size: 0,
                bucket_table_buckets: 16,
                rhash_head_next: 0,
            }),
        }
    }

    /// REQ 1 / partial passes: leaves all present, sched_pcpu present
    /// (Pass 1 runs), sched_pnode None (Pass 2 skipped), rht None
    /// (Pass 3 skipped). Result must contain Pass 1's bypass DSQ
    /// entries — pinning the "each pass independent" contract.
    #[test]
    fn walk_dsqs_partial_passes_yield_partial_results() {
        // Layout (page_offset = 0; kva_to_pa identity):
        //   sched_pa = 0x100
        //   pcpu_kva = 0x300 (placed at sched_pa + sched.pcpu = 0x100 + 0x88 = 0x188)
        //   per_cpu_offsets = [0]
        //   bypass_dsq_kva for cpu 0 = pcpu_kva + 0 + bypass_dsq_off = 0x300 + 0 + 0 = 0x300
        //   bypass DSQ list head at dsq + dsq.list = 0x300 + 0 = 0x300
        //   We write head.next = head_kva to make the list empty so
        //   walk_one_dsq returns Some with task_kvas = [].
        let mut buf = vec![0u8; 0x2000];
        let sched_pa: u64 = 0x100;
        let pcpu_kva: u64 = 0x300;
        // Place pcpu_kva at sched_pa + sched.pcpu (0x88)
        buf[(sched_pa + 0x88) as usize..(sched_pa + 0x88) as usize + 8]
            .copy_from_slice(&pcpu_kva.to_le_bytes());
        // Empty DSQ at pcpu_kva: head.next = pcpu_kva (head_kva)
        buf[pcpu_kva as usize..pcpu_kva as usize + 8].copy_from_slice(&pcpu_kva.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        // Disable pass 2 + pass 3 by Noneing their offset groups.
        offsets.sched_pnode = None;
        offsets.rht = None;

        let (states, entries) = walk_dsqs(&kernel, sched_pa, &[0u64], 0, &offsets);
        assert_eq!(states.len(), 1, "pass 1 produces one bypass DSQ entry");
        assert_eq!(states[0].origin, "bypass cpu 0");
        assert!(entries.is_empty(), "empty bypass DSQ → no task entries");
    }

    /// REQ 4 / 6.12+ compat: leaves present, ALL three "advanced"
    /// offset groups (sched_pcpu, sched_pnode, rht) None. Result is
    /// (vec![], vec![]) — no panic, no garbage reads. This is the
    /// 6.12-kernel reality: scx_sched_pcpu didn't land until v6.18,
    /// rhashtable shape varies across kernels, and sched_pnode is
    /// dev-only. Without sched layer, walker must NOT crash.
    #[test]
    fn walk_dsqs_all_advanced_offsets_none_yields_empty() {
        let mut buf = vec![0u8; 0x1000];
        // Pre-populate buf at sched_pa to ensure a buggy walker that
        // bypassed the offset gates would surface garbage. With
        // every advanced offset None, the walker must NOT read here.
        let sched_pa: u64 = 0x100;
        buf[sched_pa as usize..sched_pa as usize + 8]
            .copy_from_slice(&0xdead_beef_dead_beef_u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        offsets.sched_pcpu = None;
        offsets.sched_pnode = None;
        offsets.rht = None;

        let (states, entries) = walk_dsqs(&kernel, sched_pa, &[0u64], 1, &offsets);
        assert!(
            states.is_empty(),
            "all advanced offsets None → no DSQ states"
        );
        assert!(entries.is_empty());
    }

    /// REQ 1 / not-all-or-nothing: 2 CPUs, local-DSQ pass produces
    /// one DsqState row per CPU regardless of whether that CPU's
    /// list has tasks. CPU 0 has 1 queued task; CPU 1 is empty. The
    /// result must have 2 DsqState rows — not 0, not 1. This pins
    /// the production guarantee that walk_local_dsqs surfaces every
    /// CPU's DSQ regardless of queue depth.
    #[test]
    fn walk_local_dsqs_one_cpu_empty_one_populated() {
        // Layout (page_offset = 0; identity translation):
        //   CPU 0: rq_kva = rq_pa = 0x100. local_dsq head at
        //          rq + 0 = 0x100. head.next = task1 (0x800).
        //          dsq_lnode at task1 (0x800), dsq_lnode.flags at 0x10
        //          → set to 0 (not cursor).
        //          task1.dsq_lnode.next = head_kva (0x100, terminator).
        //   CPU 1: rq_kva = rq_pa = 0x300. local_dsq head at
        //          rq + 0 = 0x300. head.next = head_kva (empty list).
        //
        // dsq.{nr,seq,id} fields read from rq_pa+{16,20,24}.
        let mut buf = vec![0u8; 0x2000];
        let cpu0_rq: u64 = 0x100;
        let cpu1_rq: u64 = 0x300;
        let task1: u64 = 0x800;

        // CPU 0 list: head.next = task1, task1.next = head_kva.
        buf[cpu0_rq as usize..cpu0_rq as usize + 8].copy_from_slice(&task1.to_le_bytes());
        buf[task1 as usize..task1 as usize + 8].copy_from_slice(&cpu0_rq.to_le_bytes());

        // Stamp DSQ scalars on CPU 0 (id=0xa, nr=1, seq=10).
        buf[(cpu0_rq + 16) as usize..(cpu0_rq + 16) as usize + 4]
            .copy_from_slice(&1u32.to_le_bytes()); // nr
        buf[(cpu0_rq + 20) as usize..(cpu0_rq + 20) as usize + 4]
            .copy_from_slice(&10u32.to_le_bytes()); // seq
        buf[(cpu0_rq + 24) as usize..(cpu0_rq + 24) as usize + 8]
            .copy_from_slice(&0xau64.to_le_bytes()); // id

        // CPU 1 list: head.next = head_kva (empty list).
        buf[cpu1_rq as usize..cpu1_rq as usize + 8].copy_from_slice(&cpu1_rq.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        // Both CPUs onlined: per_cpu_offsets non-zero for cpu>0
        // (otherwise the BSS-zero-tail guard would skip cpu 1).
        let (states, entries) = walk_local_dsqs(
            &kernel,
            &[cpu0_rq, cpu1_rq],
            &[cpu0_rq, cpu1_rq],
            &[0, 0x1000],
            &offsets,
        )
        .expect("offsets present, should yield Some");

        assert_eq!(
            states.len(),
            2,
            "two CPUs → two DSQ rows, regardless of queue depth"
        );
        let cpu0 = states.iter().find(|s| s.origin == "local cpu 0").unwrap();
        let cpu1 = states.iter().find(|s| s.origin == "local cpu 1").unwrap();
        assert_eq!(cpu0.task_kvas.len(), 1, "CPU 0 has one queued task");
        assert!(cpu1.task_kvas.is_empty(), "CPU 1 is empty");
        assert_eq!(cpu0.id, 0xa);
        assert_eq!(cpu0.nr, 1);
        assert_eq!(cpu0.seq, 10);
        // entries vec contains exactly the CPU 0 task.
        assert_eq!(entries.len(), 1);
    }

    /// BSS-zero-tail guard: `__per_cpu_offset[]` is BSS-zero for
    /// CPU slots beyond `nr_cpu_ids` because `setup_per_cpu_areas`
    /// only writes the slots in `for_each_possible_cpu`. The walker
    /// must check `per_cpu_offsets[cpu] == 0 && cpu > 0` to skip
    /// those slots; otherwise it surfaces phantom DSQ rows for
    /// un-onlined CPUs. The phantom rows would land at the bare
    /// `runqueues` symbol KVA (rq_kva = runqueues + 0), aliasing
    /// neither CPU 0's KVA (runqueues + delta on x86_64 SMP) nor
    /// each other in any well-formed way — a resolved-rq_kva
    /// comparison cannot detect the alias on x86_64 SMP because
    /// `__per_cpu_offset[0]` is non-zero (`delta = pcpu_base_addr -
    /// __per_cpu_start`, see `arch/x86/kernel/setup_percpu.c`).
    #[test]
    fn walk_local_dsqs_skips_bss_zero_tail_aliases() {
        // Layout (page_offset = 0; identity translation):
        //   CPU 0: per_cpu_offset = 0x100; rq_kva = rq_pa = 0x100.
        //          Empty list head at PA 0x100.
        //   CPU 1, 2, 3: per_cpu_offset = 0 (BSS-zero tail).
        //
        // CPU 0's rq_kva is NOT shared by the BSS-zero entries
        // (their rq_kva would be `runqueues + 0` = 0, not 0x100),
        // so the old `rq_kva == rq_kvas[0]` guard would not catch
        // them — on x86_64 SMP, the legitimate CPU 0 entry is the
        // ONLY one with rq_kva == runqueues + per_cpu_offset[0].
        // A correct walker emits one DsqState (CPU 0) using the
        // `cpu_off == 0 && cpu > 0` check; a regressed walker
        // emits four (or three, if the old guard caught some
        // accidental alias). Pinning len() == 1 makes the
        // regression visible.
        let mut buf = vec![0u8; 0x1000];
        let cpu0_rq: u64 = 0x100;
        // CPU 0 list: head.next = head_kva (empty).
        buf[cpu0_rq as usize..cpu0_rq as usize + 8].copy_from_slice(&cpu0_rq.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        // For the BSS-zero tail entries, the caller would actually
        // pass an rq_kva of `runqueues + 0`; the test mirrors the
        // production rq_kvas here (every entry equals cpu0_rq) so
        // a regressed `rq_kva == rq_kvas[0]` walker would still
        // catch the alias. The new guard ignores rq_kvas entirely
        // and gates on per_cpu_offsets — so the BSS-zero entries
        // are skipped for that reason instead.
        let (states, entries) = walk_local_dsqs(
            &kernel,
            &[cpu0_rq, cpu0_rq, cpu0_rq, cpu0_rq],
            &[cpu0_rq, cpu0_rq, cpu0_rq, cpu0_rq],
            &[0x100, 0, 0, 0], // CPU 0 onlined; CPUs 1-3 BSS-zero
            &offsets,
        )
        .expect("offsets present, should yield Some");
        assert_eq!(
            states.len(),
            1,
            "BSS-zero-tail aliases must be skipped; only CPU 0 surfaces"
        );
        assert_eq!(states[0].origin, "local cpu 0");
        assert!(entries.is_empty());
    }

    /// Regression: x86_64 SMP layout where `__per_cpu_offset[0]` is
    /// non-zero (the production case — `delta = pcpu_base_addr -
    /// __per_cpu_start` is positive when the percpu allocator places
    /// its base outside the static `.data..percpu` region, see
    /// `setup_per_cpu_areas` in `arch/x86/kernel/setup_percpu.c`).
    /// CPU 0's `rq_kva = runqueues + delta` differs from a BSS-zero
    /// tail entry's `rq_kva = runqueues + 0`, so the prior
    /// `rq_kva == rq_kvas[0]` guard would NOT catch the alias and
    /// would surface a phantom DSQ row. The new
    /// `per_cpu_offsets[cpu] == 0 && cpu > 0` guard catches it
    /// regardless of how the resolved KVAs compare.
    #[test]
    fn walk_local_dsqs_skips_bss_zero_tail_with_nonzero_cpu0_offset() {
        // Layout (page_offset = 0; identity translation):
        //   runqueues_pa  = 0x300 (a non-zero "runqueues" KVA so a
        //                          BSS-zero entry's rq_pa is also
        //                          non-zero — `walk_one_dsq` skips
        //                          dsq_pa==0, which would otherwise
        //                          mask a regressed guard).
        //   per_cpu_offset[0] = 0x100; rq_pa[0] = 0x400 (delta).
        //   per_cpu_offset[1] = 0;     rq_pa[1] = 0x300 (BSS-zero).
        // CPU 0's rq_pa (0x400) differs from CPU 1's BSS rq_pa
        // (0x300); the prior `rq_kva == rq_kvas[0]` guard would
        // NOT catch the alias because the two PAs are distinct.
        // The new guard catches it via `cpu_off == 0 && cpu > 0`.
        let runqueues_pa: u64 = 0x300;
        let cpu0_rq: u64 = runqueues_pa + 0x100; // 0x400
        let bss_rq: u64 = runqueues_pa; // 0x300
        let mut buf = vec![0u8; 0x1000];
        // Stamp empty-list heads at BOTH addresses so a regressed
        // walker would surface DsqState rows for both. The post-
        // guard list walk reads head.next at offset 0; pointing
        // it at itself terminates the list immediately.
        buf[cpu0_rq as usize..cpu0_rq as usize + 8].copy_from_slice(&cpu0_rq.to_le_bytes());
        buf[bss_rq as usize..bss_rq as usize + 8].copy_from_slice(&bss_rq.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        let (states, _entries) = walk_local_dsqs(
            &kernel,
            &[cpu0_rq, bss_rq],
            &[cpu0_rq, bss_rq],
            &[0x100, 0], // CPU 0 onlined (delta=0x100); CPU 1 BSS-zero
            &offsets,
        )
        .expect("offsets present, should yield Some");
        // The new guard skips CPU 1 because per_cpu_offset[1] == 0.
        // If the walker compared resolved KVAs (`rq_kva == rq_kvas[0]`)
        // instead, it would see cpu0_rq != bss_rq and emit a phantom
        // row for CPU 1. Pinning len() == 1 catches that regression.
        assert_eq!(
            states.len(),
            1,
            "BSS-zero entry must be skipped via cpu_off==0 guard \
             even when its rq_pa differs from rq_pas[0]"
        );
        assert_eq!(states[0].origin, "local cpu 0");
    }

    /// REQ 2: read_scx_sched_state with `offsets.sched = None` —
    /// the walker MUST short-circuit before any guest-memory read.
    /// Pre-populating sched_pa with a value that would surface as a
    /// bogus aborting/bypass_depth ensures the gate fires correctly:
    /// a regression that read despite None offsets would emit a
    /// state with the bogus values; the None-return contract pins
    /// "no fabricated state."
    #[test]
    fn read_scx_sched_state_offsets_sched_none_returns_none() {
        let mut buf = vec![0u8; 0x1000];
        // Pre-populate: a buggy walker reading at PA 0 would surface
        // the magic value as exit_kind / bypass_depth.
        buf[0..8].copy_from_slice(&0xdead_beef_u64.to_le_bytes());
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        offsets.sched = None;
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x10;
        let r = read_scx_sched_state(&kernel, scx_root_kva, &offsets);
        assert!(r.is_none(), "sched=None must short-circuit before read");
    }

    /// REQ 2 / *scx_root unset: scx_root_kva resolves but the
    /// pointer it points to reads as 0 (no scheduler attached).
    /// read_scx_sched_state must return None — pinning the
    /// "scheduler not attached" diagnosis without surfacing bogus
    /// state.
    #[test]
    fn read_scx_sched_state_scx_root_pointer_zero_returns_none() {
        // Layout: scx_root_kva is in the text mapping. We choose
        // START_KERNEL_MAP + 0x100 so kernel.text_kva_to_pa(scx_root_kva)
        // = 0x100. Stamp 0 at that PA. The walker reads sched_kva = 0
        // and returns None.
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x100;
        let scx_root_pa = 0x100usize;
        let mut buf = vec![0u8; 0x1000];
        buf[scx_root_pa..scx_root_pa + 8].copy_from_slice(&0u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let offsets = dsq_test_offsets();
        let r = read_scx_sched_state(&kernel, scx_root_kva, &offsets);
        assert!(
            r.is_none(),
            "*scx_root == 0 (no scheduler) → None, no state surfaced"
        );
    }

    /// REQ 4 / dev-only field None: sched.aborting offset is
    /// Option<usize>. On release kernels the field is absent. The
    /// walker must NOT read at sched_pa+0 as a fallback (that would
    /// alias dsq_hash). Pinning: with aborting=None, the returned
    /// state has aborting=false.
    #[test]
    fn read_scx_sched_state_aborting_offset_none_defaults_false() {
        // Layout:
        //   scx_root_kva = START_KERNEL_MAP + 0x100
        //   *scx_root → sched_kva (we put it in direct mapping at 0x800)
        //   sched_pa = 0x800 (page_offset = 0; identity)
        //   exit_kind at sched_pa + 0x98 = 0
        //
        // Stamp a magic value at sched_pa + 0 to detect any bogus
        // fallback read for `aborting`. Without aborting=None being
        // honored, a buggy walker reading that location would
        // surface aborting=true.
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x100;
        let scx_root_pa: usize = 0x100;
        let sched_pa: u64 = 0x800;
        let mut buf = vec![0u8; 0x1000];
        // *scx_root = sched_kva (direct mapping; sched_kva == sched_pa here)
        buf[scx_root_pa..scx_root_pa + 8].copy_from_slice(&sched_pa.to_le_bytes());
        // Stamp 0xff at sched_pa+0 — non-zero, would be true if read as bool.
        buf[sched_pa as usize] = 0xff;

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        // Mark aborting offset absent — release-kernel reality.
        if let Some(s) = offsets.sched.as_mut() {
            s.aborting = None;
        }

        let (sched_kva_out, state) = read_scx_sched_state(&kernel, scx_root_kva, &offsets)
            .expect("should yield Some when sched offsets present");
        assert_eq!(sched_kva_out, sched_pa);
        assert!(
            !state.aborting,
            "aborting=None must default to false, NOT read sched_pa+0"
        );
    }

    /// REQ 4 / dev-only field None: sched.bypass_depth offset is
    /// Option<usize>. Same shape as aborting — None means the
    /// kernel doesn't have the field; walker must default to 0
    /// without reading.
    #[test]
    fn read_scx_sched_state_bypass_depth_offset_none_defaults_zero() {
        let scx_root_kva = super::super::symbols::START_KERNEL_MAP + 0x100;
        let scx_root_pa: usize = 0x100;
        let sched_pa: u64 = 0x800;
        let mut buf = vec![0u8; 0x1000];
        buf[scx_root_pa..scx_root_pa + 8].copy_from_slice(&sched_pa.to_le_bytes());
        // Stamp a magic at sched_pa+0 (would surface as bypass_depth
        // if a buggy walker read there as fallback).
        buf[sched_pa as usize..sched_pa as usize + 4]
            .copy_from_slice(&0xdead_beef_u32.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let kernel = super::super::guest::GuestKernel::new_for_test(
            &mem,
            std::collections::HashMap::new(),
            0,
            0,
            false,
        );

        let mut offsets = dsq_test_offsets();
        if let Some(s) = offsets.sched.as_mut() {
            s.bypass_depth = None;
        }

        let (_, state) = read_scx_sched_state(&kernel, scx_root_kva, &offsets)
            .expect("should yield Some when sched offsets present");
        assert_eq!(
            state.bypass_depth, 0,
            "bypass_depth=None must default to 0, NOT read sched_pa+0"
        );
    }

    // ---------------------------------------------------------------
    // walk_user_dsq_hash truncation-cap tests
    //
    // walk_user_dsq_hash bounds the walk on three independent caps
    // (MAX_RHT_BUCKETS, MAX_RHT_NODES, PER_BUCKET_CHAIN_CAP). Each
    // cap must set the returned `truncated` flag so the failure-dump
    // consumer can distinguish "small DSQ count" from "cap silently
    // dropped tail entries."
    // ---------------------------------------------------------------

    /// Build a minimal `RhashtableOffsets` fixture whose every offset
    /// is small and consistent: tbl at 0, bucket_table.size at 0,
    /// bucket_table.buckets at 16, rhash_head.next at 0. Used by the
    /// three truncation tests below.
    fn rht_test_offsets() -> super::super::btf_offsets::RhashtableOffsets {
        super::super::btf_offsets::RhashtableOffsets {
            tbl: 0,
            nelems: 8,
            bucket_table_size: 0,
            bucket_table_buckets: 16,
            rhash_head_next: 0,
        }
    }

    /// Build a minimal `ScxDispatchQOffsets` with `hash_node = 0` so
    /// container_of yields the node KVA unchanged. Tests below assert
    /// truncation flags, not container_of math.
    fn dsq_test_offsets_for_hash() -> super::super::btf_offsets::ScxDispatchQOffsets {
        super::super::btf_offsets::ScxDispatchQOffsets {
            list: 0,
            nr: 16,
            seq: 20,
            id: 24,
            hash_node: 0,
        }
    }

    /// Per-bucket chain cap (`PER_BUCKET_CHAIN_CAP`): a single
    /// bucket's chain that doesn't terminate naturally must set
    /// `truncated = true` after exactly `PER_BUCKET_CHAIN_CAP`
    /// visits. The fixture uses a 2-node cycle so the chain has no
    /// natural terminator; the walker bails on the per-bucket cap
    /// and the post-loop check sets truncated.
    #[test]
    fn walk_user_dsq_hash_per_bucket_chain_cap_truncates() {
        // Layout (page_offset = 0; identity translation):
        //   rht_pa = 0x100  (struct rhashtable; .tbl at offset 0 = 8 bytes)
        //   tbl_pa = 0x200  (bucket_table; size at off 0, buckets at off 16)
        //   node_a = 0x300, node_b = 0x308: cycle (next-pointer at off 0 each)
        //   tbl.size = 1; buckets[0] = node_a (no LSB tag).
        //
        // Walker chases node_a → node_b → node_a → ... until
        // chain_visited reaches PER_BUCKET_CHAIN_CAP. 1024 visits,
        // cap fires, post-loop sets truncated=true.
        let mut buf = vec![0u8; 0x1000];
        let rht_pa: u64 = 0x100;
        let tbl_kva: u64 = 0x200;
        let tbl_pa: u64 = 0x200;
        let node_a: u64 = 0x300;
        let node_b: u64 = 0x308;

        // rht.tbl = tbl_kva
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // tbl.size = 1 (one bucket)
        buf[tbl_pa as usize..tbl_pa as usize + 4].copy_from_slice(&1u32.to_le_bytes());
        // buckets[0] = node_a
        buf[(tbl_pa + 16) as usize..(tbl_pa + 16) as usize + 8]
            .copy_from_slice(&node_a.to_le_bytes());
        // node_a.next = node_b (LSB clear → not terminator)
        buf[node_a as usize..node_a as usize + 8].copy_from_slice(&node_b.to_le_bytes());
        // node_b.next = node_a (close the cycle, LSB clear)
        buf[node_b as usize..node_b as usize + 8].copy_from_slice(&node_a.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let rht_offs = rht_test_offsets();
        let dsq_offs = dsq_test_offsets_for_hash();

        let (dsq_kvas, truncated) =
            walk_user_dsq_hash(&mem, WalkContext::default(), rht_pa, &rht_offs, &dsq_offs);

        assert!(
            truncated,
            "per-bucket chain cap must set truncated=true on a non-terminating chain",
        );
        assert_eq!(
            dsq_kvas.len(),
            PER_BUCKET_CHAIN_CAP as usize,
            "PER_BUCKET_CHAIN_CAP must admit exactly {} chain visits",
            PER_BUCKET_CHAIN_CAP,
        );
    }

    /// Global node cap (`MAX_RHT_NODES`): when the cumulative node
    /// count across multiple buckets reaches `MAX_RHT_NODES`, the
    /// walker must set `truncated = true` and stop visiting further
    /// buckets. This test constructs `MAX_RHT_NODES + 1` buckets each
    /// with a single-node chain that terminates naturally; the
    /// per-bucket cap never fires (each chain has 1 entry). Truncation
    /// fires only via the global cap.
    #[test]
    fn walk_user_dsq_hash_global_node_cap_truncates() {
        // Layout (page_offset = 0; identity translation):
        //   rht_pa = 0x100         (.tbl at offset 0)
        //   tbl_pa = 0x1000        (size at 0, buckets at 16)
        //   shared_node = 0x40000  (next field at offset 0 = 0 → terminator)
        //   tbl.size = MAX_RHT_NODES + 1 = 8193 buckets, each pointing
        //   at shared_node.
        //
        // Walker enters each bucket, walks 1 node (push, total++),
        // reads next=0 → break with chain_terminated_naturally=true.
        // After bucket 8191, total=MAX_RHT_NODES (8192). Entering
        // bucket 8192, the pre-loop `total_nodes >= MAX_RHT_NODES`
        // gate returns truncated=true.
        let bucket_count: u32 = MAX_RHT_NODES + 1;
        let rht_pa: u64 = 0x100;
        let tbl_kva: u64 = 0x1000;
        let tbl_pa: u64 = 0x1000;
        let buckets_off: u64 = 16;
        let shared_node: u64 = 0x40000;

        // Buffer must cover: rht (0x100..0x108), tbl (0x1000),
        // buckets array (0x1010..0x1010 + bucket_count*8 = 0x1010 +
        // 0x10008 = 0x11018), and shared_node (0x40000..0x40008).
        let buf_size = (shared_node + 16) as usize;
        let mut buf = vec![0u8; buf_size];

        // rht.tbl = tbl_kva
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // tbl.size = bucket_count
        buf[tbl_pa as usize..tbl_pa as usize + 4]
            .copy_from_slice(&bucket_count.to_le_bytes());
        // Stamp every bucket[i] = shared_node
        for i in 0..bucket_count as u64 {
            let off = (tbl_pa + buckets_off + i * 8) as usize;
            buf[off..off + 8].copy_from_slice(&shared_node.to_le_bytes());
        }
        // shared_node.next = 0 (already zero from buffer init — explicit
        // for clarity).
        buf[shared_node as usize..shared_node as usize + 8].copy_from_slice(&0u64.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let rht_offs = rht_test_offsets();
        let dsq_offs = dsq_test_offsets_for_hash();

        let (dsq_kvas, truncated) =
            walk_user_dsq_hash(&mem, WalkContext::default(), rht_pa, &rht_offs, &dsq_offs);

        assert!(
            truncated,
            "global node cap (MAX_RHT_NODES) must set truncated=true",
        );
        // The walker pushes one dsq per bucket up to MAX_RHT_NODES,
        // then short-circuits — never enters bucket MAX_RHT_NODES.
        assert_eq!(
            dsq_kvas.len(),
            MAX_RHT_NODES as usize,
            "global cap halts the walk at exactly {} nodes",
            MAX_RHT_NODES,
        );
    }

    /// Bucket-table cap (`MAX_RHT_BUCKETS`): when
    /// `bucket_table.size > MAX_RHT_BUCKETS`, the walker enumerates
    /// only the first `MAX_RHT_BUCKETS` entries and sets
    /// `truncated = true` upfront — the tail of the bucket table is
    /// silently dropped. The fixture stamps `size = MAX_RHT_BUCKETS +
    /// 1`; bucket reads past the buffer return 0 (empty bucket) so
    /// the walker drains all 65536 reads with no chain work.
    #[test]
    fn walk_user_dsq_hash_bucket_table_cap_truncates() {
        // Layout (page_offset = 0; identity translation):
        //   rht_pa = 0x100   (.tbl at offset 0)
        //   tbl_pa = 0x200   (size at 0, buckets at 16)
        //   tbl.size = MAX_RHT_BUCKETS + 1 = 65537.
        //
        // The walker computes bucket_count = size.min(MAX_RHT_BUCKETS)
        // = 65536, then `truncated = size as u64 > bucket_count`
        // immediately fires. Subsequent bucket reads land outside the
        // buffer and return 0 (empty bucket); no chains walked.
        let mut buf = vec![0u8; 0x300];
        let rht_pa: u64 = 0x100;
        let tbl_kva: u64 = 0x200;
        let tbl_pa: u64 = 0x200;

        // rht.tbl = tbl_kva
        buf[rht_pa as usize..rht_pa as usize + 8].copy_from_slice(&tbl_kva.to_le_bytes());
        // tbl.size = MAX_RHT_BUCKETS + 1 → upfront truncation
        let oversize: u32 = MAX_RHT_BUCKETS + 1;
        buf[tbl_pa as usize..tbl_pa as usize + 4].copy_from_slice(&oversize.to_le_bytes());

        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let rht_offs = rht_test_offsets();
        let dsq_offs = dsq_test_offsets_for_hash();

        let (dsq_kvas, truncated) =
            walk_user_dsq_hash(&mem, WalkContext::default(), rht_pa, &rht_offs, &dsq_offs);

        assert!(
            truncated,
            "bucket-table cap (size > MAX_RHT_BUCKETS) must set truncated=true upfront",
        );
        assert!(
            dsq_kvas.is_empty(),
            "all buckets read as 0 (out-of-buffer) → no DSQ KVAs collected",
        );
    }
}
