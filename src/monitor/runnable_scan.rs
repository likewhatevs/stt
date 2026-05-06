//! Runnable_at scanners for the dual-snapshot freeze coordinator.
//!
//! Two complementary walks contribute to the maximum age the
//! coordinator compares against `watchdog_timeout/2`:
//!
//! 1. **Global `scx_tasks` walk** (`max_runnable_age_global`): every
//!    scx-managed task is linked into the kernel's global `scx_tasks`
//!    LIST_HEAD (`kernel/sched/ext.c:47`) via
//!    `task_struct.scx.tasks_node` at `scx_init_task`
//!    (`kernel/sched/ext.c:3742`); they leave at `sched_ext_dead`
//!    (`kernel/sched/ext.c:3803`). The walker recovers each
//!    `task_struct` KVA via container_of on `task.scx +
//!    see.tasks_node`, reads `task_struct.scx.runnable_at`, and
//!    skips entries that are not `SCX_TASK_QUEUED` (the kernel
//!    re-stamps `runnable_at` on every enqueue, so an unqueued task
//!    on this list carries a stale stamp from its last enqueue) AND
//!    skips entries that have `SCX_TASK_RESET_RUNNABLE_AT` set (the
//!    kernel marks the stamp stale via `clr_task_runnable` and does
//!    not refresh it until the next `set_task_runnable`). The
//!    actual per-rq runnable_list drain happens in
//!    `scx_root_disable` (`kernel/sched/ext.c`), which iterates
//!    `scx_task_iter_next_locked` and switches each task's class
//!    via `scx_disable_and_exit_task`. `scx_bypass`
//!    (`kernel/sched/ext.c:5304-5404`) only cycles tasks
//!    (DEQUEUE_SAVE | DEQUEUE_MOVE) on each per-rq `runnable_list`
//!    so they remain on the list after — it does NOT drain.
//!    Because `scx_bypass` is invoked at the start of
//!    `scx_root_disable` (and again at the end), the global walk
//!    plus the per-rq walk together cover bypass-time captures.
//!
//! 2. **Per-rq `runnable_list` walk** (`max_runnable_age_per_rq`):
//!    each CPU's `rq->scx.runnable_list` head is the same list the
//!    kernel's own watchdog (`check_rq_for_timeouts`,
//!    `kernel/sched/ext.c`) walks. Tasks on a per-rq list are
//!    queued by construction; their `runnable_at` is the live
//!    enqueue stamp and is the truthful evidence of an aged stuck
//!    task, NOT a stale leftover. The global walk's QUEUED gate
//!    paper-over a structural gap: rotating tasks on `scx_tasks`
//!    can mask a single stuck task whose stamp ages past the
//!    threshold even when every visible entry's stamp keeps
//!    sliding forward. The per-rq walk surfaces that aging task
//!    directly because tasks remain on the per-rq list for the
//!    full duration of their runnable window — the very window
//!    `check_rq_for_timeouts` measures.
//!
//! 3. **`scx_watchdog_timestamp` heartbeat** (caller-resolved PA
//!    passed to [`max_runnable_age`] as
//!    `watchdog_timestamp_pa`): the file-scope static
//!    `scx_watchdog_timestamp` (`kernel/sched/ext.c:94`) is
//!    refreshed to `jiffies` by the workqueue callback
//!    `scx_watchdog_workfn` (`kernel/sched/ext.c:3383`). The
//!    kernel's `scx_tick` (`kernel/sched/ext.c:3409`) compares
//!    `jiffies - scx_watchdog_timestamp` against
//!    `root->watchdog_timeout` and fires `SCX_EXIT_ERROR_STALL`
//!    when the workqueue stops running. Reading the same global
//!    value catches the case where the scheduler stopped
//!    dispatching but no individual task is uniquely stuck on a
//!    runnable_list — the per-rq and global walks both contribute
//!    0 in that case because `runnable_at` is per-task and not
//!    every test scheduler stalls a single task.
//!
//! `max_runnable_age` returns the max of all three signals. Any
//! signal contributing zero is harmless. The three paths are
//! complementary across the run's life cycle: the per-rq path is
//! the primary signal during normal operation; the global path
//! covers the bypass window where per-rq lists are drained; the
//! watchdog-timestamp path covers global "scheduler stopped"
//! stalls that the per-task signals miss.
//!
//! The dual-snapshot trigger compares against **half** the
//! configured `watchdog_timeout`, not the full timeout — the goal
//! is to capture a pre-stall snapshot before the kernel itself
//! emits `SCX_EXIT_ERROR_STALL` at the full-timeout mark. Diffing
//! the half-way snapshot against the eventual error-exit snapshot
//! shows what BPF state changed during the second half of the
//! stall window.
//!
//! All reads go through [`super::reader::GuestMem`] and
//! [`super::idr::translate_any_kva`] (the same direct-mapping +
//! page-walk code path the BPF map dump uses). The `scx_tasks`
//! LIST_HEAD itself lives in the kernel text/.data mapping
//! (`text_kva_to_pa_with_base`); each `task_struct` is a slab
//! object inside the kernel direct map, so the walk is
//! page-walk-light: one translate per task to read its
//! runnable_at, plus one translate per node to step `next`.

use super::reader::{GuestMem, WalkContext};
use crate::monitor::btf_offsets::RunnableScanOffsets;

/// `SCX_TASK_CURSOR` flag value (`1 << 31`) on `sched_ext_entity.flags`.
/// Cursor entries are stack-allocated `sched_ext_entity` placeholders
/// that `scx_task_iter_start` (`kernel/sched/ext.c:843-846`) inserts
/// into `scx_tasks` to mark the iterator's progress; they are NOT
/// embedded in any `task_struct` so the global walker must skip them
/// to avoid container_of producing a bogus task KVA. Pinned per
/// `include/linux/sched/ext.h::SCX_TASK_CURSOR`.
const SCX_TASK_CURSOR: u32 = 1 << 31;

/// `SCX_TASK_QUEUED` flag value (`1 << 0`) on `sched_ext_entity.flags`.
/// Set when the task is currently on an ext runqueue (i.e. has a
/// meaningful `runnable_at` stamp); cleared on dequeue. Tasks linked
/// into the global `scx_tasks` list but not currently runnable
/// (sleeping, blocked, freshly init'd) carry stale `runnable_at`
/// values that can age by hours; using their delta against the
/// current `jiffies` would synthesize false stalls and trip the
/// dual-snapshot's early trigger immediately on every guest. Gating
/// the age contribution on this flag mirrors the kernel's
/// `check_rq_for_timeouts` semantics — that walker iterates per-rq
/// `runnable_list`, which only holds queued tasks. Pinned per
/// `include/linux/sched/ext.h::scx_ent_flags::SCX_TASK_QUEUED`.
const SCX_TASK_QUEUED: u32 = 1 << 0;

/// `SCX_TASK_RESET_RUNNABLE_AT` flag value (`1 << 2`) on
/// `sched_ext_entity.flags`. Set by `clr_task_runnable(_,
/// reset_runnable_at=true)` in `kernel/sched/ext.c` to mark
/// `runnable_at` as stale: the kernel does NOT refresh `runnable_at`
/// when the task next becomes runnable (it waits for the next
/// `set_task_runnable` to overwrite). A task with QUEUED set AND
/// RESET_RUNNABLE_AT set therefore has a stamp that is
/// indeterminately old — typically hours for a long-running healthy
/// task that was repeatedly dequeued/enqueued — so subtracting it
/// from `jiffies` would synthesize a false stall. The per-rq walker
/// is unaffected because tasks are removed from `runnable_list` in
/// `clr_task_runnable` (`list_del_init`) before this flag is
/// stamped, so a per-rq entry is always live-stamped. Pinned per
/// `include/linux/sched/ext.h::scx_ent_flags::SCX_TASK_RESET_RUNNABLE_AT`.
const SCX_TASK_RESET_RUNNABLE_AT: u32 = 1 << 2;

/// Cap on the number of nodes the walker visits before bailing out.
///
/// Defends against a corrupted `next` pointer that fails to terminate
/// (cycle that doesn't include the head, or a stray pointer into
/// arbitrary memory). The kernel's `scx_tasks` list holds one entry
/// per scx-managed task on the host — even on a tens-of-thousands-of-
/// threads workload the count stays well under this cap. Picking
/// 65536 keeps the bound generously above realistic kernels and well
/// below the cost of a runaway walk.
const MAX_NODES: u32 = 65_536;

/// Walk every available source and return the maximum `jiffies -
/// runnable_at` observed.
///
/// Combines three signals:
/// 1. the global `scx_tasks` walk
/// 2. a per-CPU `rq->scx.runnable_list` walk for each CPU's `rq` PA
///    in `rq_pas`
/// 3. the global `scx_watchdog_timestamp` heartbeat at
///    `watchdog_timestamp_pa` — `jiffies - timestamp` is the same
///    quantity the kernel's `scx_tick` (`kernel/sched/ext.c:3409`)
///    compares against `root->watchdog_timeout` to decide whether
///    to fire `SCX_EXIT_ERROR_STALL`. The workqueue callback
///    `scx_watchdog_workfn` (`kernel/sched/ext.c:3383`) refreshes
///    the timestamp; when the scheduler stops dispatching the
///    workqueue, the value goes stale and this walk returns the
///    growing age — catching the broader "scheduler stopped"
///    case that the per-rq runnable_list walks miss when no
///    individual task is uniquely stuck.
///
/// Returns `max(global, max_over_per_rq, watchdog_age)`. Any
/// contributing 0 is harmless. The watchdog age skips when
/// `watchdog_timestamp_pa` is `None` (caller lacks the symbol)
/// or when the read returns 0 (unmapped guest memory) — the
/// kernel initialises `scx_watchdog_timestamp = INITIAL_JIFFIES`
/// (`kernel/sched/ext.c:94`), so a real-running guest never
/// shows 0; treating 0 as "no signal" avoids synthesising a
/// spurious giant age out of an unmapped read.
///
/// `rq_pas` is the per-CPU `struct rq` PA array (typically built
/// by [`super::symbols::compute_rq_pas`]). When empty, only the
/// global walk + watchdog signal run — useful for callers that
/// lack per-CPU addressing (no `runqueues` symbol, no
/// `__per_cpu_offset` array, etc.).
///
/// All other parameters mirror [`max_runnable_age_global`] and
/// [`max_runnable_age_per_rq`].
#[allow(clippy::too_many_arguments)]
pub fn max_runnable_age(
    mem: &GuestMem,
    scx_tasks_kva: u64,
    rq_pas: &[u64],
    offsets: &RunnableScanOffsets,
    jiffies: u64,
    walk: WalkContext,
    watchdog_timestamp_pa: Option<u64>,
    start_kernel_map: u64,
    phys_base: u64,
) -> u64 {
    let global = max_runnable_age_global(
        mem,
        scx_tasks_kva,
        offsets,
        jiffies,
        walk,
        start_kernel_map,
        phys_base,
    );
    let mut per_rq_max: u64 = 0;
    for &rq_pa in rq_pas {
        let age = max_runnable_age_per_rq(mem, rq_pa, offsets, jiffies, walk);
        if age > per_rq_max {
            per_rq_max = age;
        }
    }
    let watchdog_age = match watchdog_timestamp_pa {
        Some(pa) => {
            let timestamp = mem.read_u64(pa, 0);
            // A zero read is an unmapped PA (GuestMem bounds-check
            // returned 0). The kernel initialises
            // `scx_watchdog_timestamp = INITIAL_JIFFIES`
            // (`kernel/sched/ext.c:94`), which is non-zero, so a
            // real running guest never shows 0; treat it as
            // "no signal" rather than synthesise `jiffies` of age.
            //
            // Pre-attach guard (cap on implausibly large ages):
            // both `jiffies_64` and `scx_watchdog_timestamp` are
            // initialised to `INITIAL_JIFFIES` at boot. Once a
            // sched_ext scheduler attaches and `scx_watchdog_workfn`
            // runs, the timestamp is refreshed regularly and
            // `jiffies - timestamp` stays small. BEFORE any
            // scheduler attaches the workqueue is dormant, the
            // timestamp stays at its `INITIAL_JIFFIES` boot value,
            // and `jiffies - timestamp` grows unboundedly with
            // wall time — at HZ=1000, after 5 seconds the age
            // would already be 5000 jiffies and would falsely
            // trip a 4-second `watchdog_half` threshold. Capping
            // at 86_400_000 jiffies (24h at HZ=1000; comfortably
            // above any plausible scheduler stall — the kernel's
            // own `scx_tick` would have fired
            // `SCX_EXIT_ERROR_STALL` long before then) suppresses
            // the boot-time false trigger without affecting any
            // real stall. The runnable_list walks remain the
            // primary signal for per-task stalls; this third
            // signal is purely additive for the global "workfn
            // wedged" failure mode that the per-rq walks miss.
            const SANITY_CAP_JIFFIES: u64 = 86_400_000;
            if timestamp == 0 {
                0
            } else {
                // saturating_sub mirrors the per-rq / global
                // walkers' future-stamp guard: if the workqueue
                // updated the timestamp on a CPU after our
                // jiffies read (rare race window), treat it as
                // age 0 instead of wrapping to near-u64::MAX.
                let raw = jiffies.saturating_sub(timestamp);
                if raw > SANITY_CAP_JIFFIES { 0 } else { raw }
            }
        }
        None => 0,
    };
    let mut max_age = global;
    if per_rq_max > max_age {
        max_age = per_rq_max;
    }
    if watchdog_age > max_age {
        max_age = watchdog_age;
    }
    max_age
}

/// Walk the kernel's global `scx_tasks` LIST_HEAD and return the
/// maximum `jiffies - p->scx.runnable_at` observed across every task
/// on the list.
///
/// `scx_tasks_kva` is the symbol KVA of the global `LIST_HEAD`
/// (resolved by `KernelSymbols::scx_tasks`). Returns `0` when the
/// symbol is absent (`scx_tasks_kva == 0`) — a kernel without
/// sched_ext or a stripped vmlinux.
///
/// `jiffies` is the current `jiffies_64` value (caller reads this
/// from the pre-translated `jiffies_64_pa`). `page_offset` is the
/// runtime PAGE_OFFSET for `kva_to_pa` direct-mapping translation;
/// `cr3_pa` and `l5` feed the page-walk fallback through
/// [`super::idr::translate_any_kva`] for KVAs that fall outside
/// the direct map (vmalloc-backed slab objects on some
/// configurations).
///
/// Returns 0 when `scx_tasks` is empty (head.next == &head), when
/// the head reads as zero (unmapped or torn read), or when every
/// task's `runnable_at` is newer than `jiffies` (the
/// `saturating_sub` clamps a future-runnable_at down to 0).
///
/// Tasks whose `runnable_at == 0` are skipped without contributing
/// to max — kernel-side `runnable_at` is zero between slab
/// allocation and the first `set_task_runnable` call, and treating
/// that as age-from-zero would synthesize spurious multi-day
/// stalls. Cursor entries (`SCX_TASK_CURSOR` flag set on the
/// enclosing `sched_ext_entity.flags`) are skipped without
/// contributing to max — they are stack-allocated iterator
/// placeholders, not real tasks, and their container_of result is
/// not a valid `task_struct` KVA.
///
/// Best-effort: a task with an unmapped slab page contributes
/// nothing for that task but does not abort the scan; a corrupted
/// chain contributes whatever was scanned before the bail-out,
/// capped at MAX_NODES visits.
pub fn max_runnable_age_global(
    mem: &GuestMem,
    scx_tasks_kva: u64,
    offsets: &RunnableScanOffsets,
    jiffies: u64,
    walk: WalkContext,
    start_kernel_map: u64,
    phys_base: u64,
) -> u64 {
    if scx_tasks_kva == 0 {
        return 0;
    }
    // The LIST_HEAD lives in the kernel text/.data mapping; convert
    // KVA → PA via text_kva_to_pa_with_base. The first u64 at that PA
    // is list_head.next (the LIST_HEAD struct's first field).
    let head_kva = scx_tasks_kva;
    let head_pa =
        super::symbols::text_kva_to_pa_with_base(scx_tasks_kva, start_kernel_map, phys_base);

    // Read head.next. struct list_head { next; prev; } — `next` at
    // offset 0. Empty list: head.next == &head, so the loop exits
    // on the first iteration without reading any task.
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        // Defensive: a zero `next` pointer means either an
        // uninitialized list head (guest still booting) or guest
        // memory the host could not read (PA out of range —
        // read_u64 returns 0 for unmapped). Skip without
        // contributing.
        return 0;
    }

    let tasks_node_off_in_task = offsets.task_struct_scx + offsets.sched_ext_entity_tasks_node;
    let runnable_at_off_in_task = offsets.task_struct_scx + offsets.sched_ext_entity_runnable_at;
    let flags_off_in_see = offsets.sched_ext_entity_flags;
    let tasks_node_off_in_see = offsets.sched_ext_entity_tasks_node;

    let mut max_age: u64 = 0;
    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES {
            // Cycle or runaway pointer — bail out with what we have.
            // tracing::warn would amplify a transient corruption into
            // log spam, so swallow silently. The freeze coord's normal
            // dump path still surfaces the underlying problem.
            return max_age;
        }
        visited += 1;

        // Recover the sched_ext_entity base for this list entry so we
        // can read its `flags` and (a) skip cursor placeholders,
        // (b) skip non-queued tasks whose `runnable_at` is stale.
        // For task-embedded entries this base is inside a
        // task_struct (`task_kva + task.scx`); for cursor entries it
        // is a stack-allocated `sched_ext_entity`. Either way,
        //   see_kva = node_kva - see.tasks_node
        //
        // The `queued` flag mirrors the kernel's per-rq
        // `runnable_list` membership: only tasks the kernel itself
        // would walk in `check_rq_for_timeouts` carry a meaningful
        // `runnable_at`. Tasks linked into the global `scx_tasks`
        // list but not currently runnable (sleeping, blocked, freshly
        // init'd) keep their last `runnable_at` value indefinitely;
        // a multi-hour-old stamp on a sleeping task would synthesize
        // a false stall and trip the dual-snapshot's early trigger
        // every guest tick.
        //
        // Translate failure on the see base — conservative: treat as
        // not-cursor AND not-queued so the entry contributes nothing
        // but the walk continues. The next-node step below still
        // runs and the rest of the list is scanned.
        let see_kva = node_kva.wrapping_sub(tasks_node_off_in_see as u64);
        let (cursor, queued, reset_runnable_at) = match super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            see_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(see_pa) => {
                let flags = mem.read_u32(see_pa, flags_off_in_see);
                (
                    flags & SCX_TASK_CURSOR != 0,
                    flags & SCX_TASK_QUEUED != 0,
                    flags & SCX_TASK_RESET_RUNNABLE_AT != 0,
                )
            }
            None => (false, false, false),
        };

        // Skip QUEUED tasks whose `runnable_at` is stale per
        // `clr_task_runnable(_, reset_runnable_at=true)`. The kernel
        // sets RESET_RUNNABLE_AT to mark the field as
        // not-yet-refreshed: the value carries the time of the
        // task's previous enqueue, NOT a live runnable window. A
        // long-running healthy task that toggles between runnable
        // and dequeued can carry an hours-old stamp here and would
        // synthesize a false stall. The per-rq walker is unaffected
        // because the task has been list_del_init'd out of
        // `runnable_list` before this flag is set.
        if !cursor && queued && !reset_runnable_at {
            // container_of: task_kva = node_kva - tasks_node_off_in_task.
            let task_kva = node_kva.wrapping_sub(tasks_node_off_in_task as u64);

            // Read p->scx.runnable_at. task_struct lives in slab; on
            // most configs that's the direct map (kva_to_pa works), but
            // some BPF arenas / vmalloc-backed slabs require a page
            // walk. Use translate_any_kva so the latter still resolves.
            let runnable_at_kva = task_kva.wrapping_add(runnable_at_off_in_task as u64);
            if let Some(runnable_at_pa) = super::idr::translate_any_kva(
                mem,
                walk.cr3_pa,
                walk.page_offset,
                runnable_at_kva,
                walk.l5,
                walk.tcr_el1,
            ) {
                let runnable_at = mem.read_u64(runnable_at_pa, 0);
                // Skip the age contribution when `runnable_at == 0`.
                // The kernel stamps `p->scx.runnable_at = jiffies` in
                // `set_task_runnable` on enqueue (kernel/sched/ext.c)
                // and `init_scx_entity` on entity init; a zero value
                // means the task_struct's scx slab bytes have been
                // zeroed but neither stamping site has run yet — most
                // commonly a fresh task_struct between slab allocation
                // and the first set_task_runnable call. Treating 0 as
                // a jiffies-aged-from-time-zero would synthesize a
                // multi-day "stall" out of an entirely fresh task and
                // falsely trigger the early-snapshot path.
                if runnable_at != 0 {
                    // saturating_sub(): if runnable_at > jiffies (set in
                    // the future, which can happen on a wraparound or the
                    // rare case where a task is enqueued from another CPU
                    // mid-sample), a wrapping_sub would produce a near-
                    // u64::MAX value that falsely looks like a stall.
                    // Saturating to 0 is the right semantics — a future
                    // runnable_at is age 0 by definition.
                    let age = jiffies.saturating_sub(runnable_at);
                    if age > max_age {
                        max_age = age;
                    }
                }
            } else {
                // Untranslatable task page — skip without contributing
                // to max.
                tracing::debug!(
                    task_kva = format_args!("{task_kva:#x}"),
                    "runnable_scan: task page untranslatable, skipping",
                );
            }
        }

        // Step to next node. Read node.next (offset 0 of list_head).
        // Translate the node KVA to PA each step — successive
        // task_structs live in different slab pages, so caching a
        // single PA does not help.
        let node_pa = match super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(pa) => pa,
            None => return max_age,
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            // Same defensive bail as the empty-head case.
            return max_age;
        }
        node_kva = next_kva;
    }
    max_age
}

/// Walk a single CPU's `rq->scx.runnable_list` and return the
/// maximum `jiffies - p->scx.runnable_at` observed across every
/// task on the list.
///
/// `rq_pa` is the per-CPU `struct rq` PA (typically built by
/// [`super::symbols::compute_rq_pas`]). The list head sits at
/// `rq_pa + offsets.rq_scx + offsets.scx_rq_runnable_list`; the
/// list links each `sched_ext_entity` via its
/// `runnable_node` field (NOT `tasks_node` — that one wires the
/// global `scx_tasks` list and points to the SAME entity but
/// through a different list_head). container_of recovers the owning
/// `task_struct`:
///   `task_kva = node_kva - (task_struct_scx +
///                           sched_ext_entity_runnable_node)`.
///
/// Per-rq list semantics simplify the walker compared to the global
/// case:
/// - **No cursor entries.** The kernel uses cursors only for
///   `scx_task_iter_*` over `scx_tasks`; per-rq lists never carry
///   stack-allocated placeholders (`check_rq_for_timeouts` walks
///   the list directly with `list_for_each_entry`, no cursor).
/// - **No QUEUED gate.** Tasks are enqueued onto the per-rq list at
///   the same site that stamps `runnable_at` (`set_task_runnable`
///   in kernel/sched/ext.c) and removed at the same site that
///   clears the queued bit. Membership IS queued-ness; a separate
///   flag check would be redundant.
///
/// Returns 0 when `rq_pa == 0`, when the head reads as zero
/// (unmapped or torn read), or when every task's `runnable_at` is
/// newer than `jiffies`. The MAX_NODES bound from the global walker
/// applies here too as a corruption safety net.
pub fn max_runnable_age_per_rq(
    mem: &GuestMem,
    rq_pa: u64,
    offsets: &RunnableScanOffsets,
    jiffies: u64,
    walk: WalkContext,
) -> u64 {
    if rq_pa == 0 {
        return 0;
    }

    // Head address inside `struct rq`. The list_head's `next` field
    // is at offset 0 of the list_head, so reading u64 at the head's
    // offset gives `head.next`.
    let head_offset = offsets.rq_scx + offsets.scx_rq_runnable_list;
    let head_pa = rq_pa.wrapping_add(head_offset as u64);
    // `head.next` is a KVA pointing to the first
    // `sched_ext_entity.runnable_node` on the list, OR back to the
    // head if the list is empty.
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        // Defensive: zero means the head was unreadable or
        // uninitialised. The kernel's INIT_LIST_HEAD points back at
        // itself, never at NULL — so a 0 read here is host-side
        // memory unavailability, not an empty list. Skip without
        // contributing.
        return 0;
    }

    // Reconstruct the head's KVA so the loop can detect "we walked
    // back to the head" the same way the global walker does.
    // Per-rq head lives in the percpu direct-map; KVA = PA +
    // page_offset.
    let head_kva = head_pa.wrapping_add(walk.page_offset);

    let runnable_node_off_in_task =
        offsets.task_struct_scx + offsets.sched_ext_entity_runnable_node;
    let runnable_at_off_in_task = offsets.task_struct_scx + offsets.sched_ext_entity_runnable_at;

    let mut max_age: u64 = 0;
    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES {
            return max_age;
        }
        visited += 1;

        // container_of: task_kva = node_kva - runnable_node_off.
        // Recover the owning task_struct, then read
        // `runnable_at`. Per-rq entries carry live runnable_at
        // stamps because membership coincides with queued-ness;
        // no QUEUED gate or zero-skip mirroring the global walker
        // is needed (a task only enters the list AFTER
        // set_task_runnable stamps a non-zero jiffies).
        let task_kva = node_kva.wrapping_sub(runnable_node_off_in_task as u64);
        let runnable_at_kva = task_kva.wrapping_add(runnable_at_off_in_task as u64);
        if let Some(runnable_at_pa) = super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            runnable_at_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            let runnable_at = mem.read_u64(runnable_at_pa, 0);
            if runnable_at != 0 {
                // saturating_sub: same wraparound rationale as the
                // global walker — a future-stamped runnable_at
                // must surface as age 0, not u64::MAX.
                let age = jiffies.saturating_sub(runnable_at);
                if age > max_age {
                    max_age = age;
                }
            }
        } else {
            tracing::debug!(
                task_kva = format_args!("{task_kva:#x}"),
                "runnable_scan: per-rq task page untranslatable, skipping",
            );
        }

        // Step to next node. Read node.next at offset 0 of the
        // list_head. Translate the node KVA to PA each step —
        // successive task_structs live in different slab pages.
        let node_pa = match super::idr::translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            node_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            Some(pa) => pa,
            None => return max_age,
        };
        let next_kva = mem.read_u64(node_pa, 0);
        if next_kva == 0 {
            return max_age;
        }
        node_kva = next_kva;
    }
    max_age
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::reader::GuestMem;
    use crate::monitor::symbols::START_KERNEL_MAP;

    /// Build a `RunnableScanOffsets` with the test-fixture layout:
    /// `task.scx == 0` and `see.tasks_node == 0` so container_of
    /// is identity (`task_kva == node_kva`); `see.runnable_at`
    /// nonzero so the field is at a distinct offset; `see.flags`
    /// at a distinct offset for cursor-skip tests.
    ///
    /// Per-rq fields (`runnable_node`, `rq_scx`,
    /// `scx_rq_runnable_list`) default to zero. Tests that need
    /// a per-rq layout pin them explicitly via `with_per_rq`.
    fn test_offsets(
        task_struct_scx: usize,
        sched_ext_entity_tasks_node: usize,
        sched_ext_entity_runnable_at: usize,
        sched_ext_entity_flags: usize,
    ) -> RunnableScanOffsets {
        RunnableScanOffsets {
            task_struct_scx,
            sched_ext_entity_tasks_node,
            sched_ext_entity_flags,
            sched_ext_entity_runnable_at,
            sched_ext_entity_runnable_node: 0,
            rq_scx: 0,
            scx_rq_runnable_list: 0,
        }
    }

    /// Helper: extend a `test_offsets` with explicit per-rq layout
    /// values so the per-rq walker tests can pin distinct offsets
    /// for `runnable_node`, `rq.scx`, and `scx_rq.runnable_list`.
    fn with_per_rq(
        mut o: RunnableScanOffsets,
        sched_ext_entity_runnable_node: usize,
        rq_scx: usize,
        scx_rq_runnable_list: usize,
    ) -> RunnableScanOffsets {
        o.sched_ext_entity_runnable_node = sched_ext_entity_runnable_node;
        o.rq_scx = rq_scx;
        o.scx_rq_runnable_list = scx_rq_runnable_list;
        o
    }

    /// scx_tasks_kva == 0 (kernel without sched_ext or stripped
    /// vmlinux): scanner returns 0 immediately, no memory reads.
    #[test]
    fn zero_scx_tasks_kva_returns_zero() {
        let mut buf = vec![0u8; 4096];
        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(0, 0, 16, 8);
        let age = max_runnable_age_global(
            &mem,
            0,
            &offsets,
            1_000,
            WalkContext::default(),
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(age, 0);
    }

    /// Empty list: head.next == &head. Returns 0.
    #[test]
    fn empty_list_returns_zero() {
        // Place head in the text mapping (text_kva_to_pa_with_base
        // subtracts START_KERNEL_MAP). Layout:
        //   PA 0: head — head.next = &head (self-loop terminator).
        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        buf[0..8].copy_from_slice(&head_kva.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(0, 0, 16, 8);
        // page_offset chosen so kva_to_pa(direct_kva, page_offset) =
        // pa for any task we lay out below.
        let page_offset = 0xffff_8880_0000_0000u64;
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            10_000,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(age, 0, "empty list must return age 0");
    }

    /// Single stalled task on the global list: runnable_at = jiffies - 50,
    /// expected age 50.
    #[test]
    fn single_stalled_task_age() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;

        // page_offset chosen so kva_to_pa(task_kva, page_offset) =
        // task_pa (direct mapping). text_kva_to_pa_with_base subtracts
        // START_KERNEL_MAP, so the head sits in the text region while
        // task slabs sit in the direct-map region.
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        // head at PA 0 (text mapping).
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        // task at PA 64, KVA in the direct map.
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva + (task_scx + tasks_node_off) as u64;

        // head.next = node_kva (single-task list).
        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        // node.next = head_kva (terminator). node lives at offset 0
        // of the task_struct, which is at PA 64.
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at at task_pa + runnable_at_off; value = jiffies - 50.
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 50;
        let runnable_at_pa = (task_pa as usize) + task_scx + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&runnable_at.to_le_bytes());
        // Stamp SCX_TASK_QUEUED on the task's see.flags so the
        // walker treats `runnable_at` as live (matches the kernel's
        // per-rq runnable_list invariant).
        let flags_pa = (task_pa as usize) + task_scx + flags_off;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(age, 50);
    }

    /// Future runnable_at (runnable_at > jiffies) saturates to age 0
    /// rather than wrapping to a huge u64.
    #[test]
    fn future_runnable_at_treated_as_age_zero() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;

        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        let jiffies = 1_000u64;
        let future = jiffies + 100;
        let runnable_at_pa = 64usize + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&future.to_le_bytes());
        // Stamp SCX_TASK_QUEUED — required for the walker to even
        // consider the task. The future runnable_at then saturates
        // to age 0 via the saturating_sub invariant.
        let flags_pa = 64usize + flags_off;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(age, 0, "future runnable_at must saturate to age 0");
    }

    /// `runnable_at == 0` is treated as "not yet stamped" and
    /// contributes nothing to max_age. Without this guard a fresh
    /// zero-init task_struct on the global list would synthesize
    /// `jiffies` of "stall age" out of an unrelated task.
    #[test]
    fn zero_runnable_at_skipped() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;

        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at = 0 — task on the list but not yet stamped.
        // (Zeroed by `vec![0u8; 4096]`; explicit for clarity.)
        let runnable_at = 0u64;
        let runnable_at_pa = 64usize + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&runnable_at.to_le_bytes());
        // Stamp SCX_TASK_QUEUED so the walker examines this task —
        // we want the runnable_at == 0 skip path, not the unqueued
        // skip path, to be the test's discriminator.
        let flags_pa = 64usize + flags_off;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let jiffies = 1_000u64;
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 0,
            "runnable_at == 0 must be skipped, not treated as a 1000-jiffy stall"
        );
    }

    /// Two tasks on the global list — max wins.
    #[test]
    fn max_across_tasks() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        // Two tasks. task A age 30, task B age 70.
        let task_a_pa = 64u64;
        let task_a_kva = page_offset.wrapping_add(task_a_pa);
        let task_b_pa = 256u64;
        let task_b_kva = page_offset.wrapping_add(task_b_pa);

        // head.next = task A
        buf[0..8].copy_from_slice(&task_a_kva.to_le_bytes());
        // task_a.next = task B
        buf[64..72].copy_from_slice(&task_b_kva.to_le_bytes());
        // task_b.next = head (terminator)
        buf[256..264].copy_from_slice(&head_kva.to_le_bytes());

        let jiffies = 1_000u64;
        let runnable_at_a = jiffies - 30;
        let runnable_at_b = jiffies - 70;
        buf[(64 + runnable_at_off)..(64 + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_a.to_le_bytes());
        buf[(256 + runnable_at_off)..(256 + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_b.to_le_bytes());
        // Stamp SCX_TASK_QUEUED on both tasks' see.flags so they
        // both contribute to the max.
        buf[(64 + flags_off)..(64 + flags_off + 4)].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());
        buf[(256 + flags_off)..(256 + flags_off + 4)]
            .copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(age, 70, "scan must take the max across the global list");
    }

    /// Cursor entry skipped: scx_task_iter_start inserts a stack-
    /// allocated sched_ext_entity placeholder with SCX_TASK_CURSOR
    /// (1 << 31) set on its `flags`. The walker must skip without
    /// dereferencing — its container_of result is not a real
    /// task_struct.
    #[test]
    fn cursor_entry_skipped() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;

        // Task A at PA 64 — real, age 30.
        let task_a_pa = 64u64;
        let task_a_kva = page_offset.wrapping_add(task_a_pa);
        // Cursor entry at PA 256 — stack-allocated sched_ext_entity
        // (no embedding task_struct), flags has SCX_TASK_CURSOR set.
        let cursor_pa = 256u64;
        let cursor_kva = page_offset.wrapping_add(cursor_pa);

        // head.next = task A → cursor → head.
        buf[0..8].copy_from_slice(&task_a_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&cursor_kva.to_le_bytes());
        buf[256..264].copy_from_slice(&head_kva.to_le_bytes());

        let jiffies = 1_000u64;
        let runnable_at_a = jiffies - 30;
        buf[(64 + runnable_at_off)..(64 + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_a.to_le_bytes());
        // Stamp SCX_TASK_QUEUED on task A's flags so its runnable_at
        // counts.
        buf[(64 + flags_off)..(64 + flags_off + 4)].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // Stamp SCX_TASK_CURSOR (1<<31) on the cursor entry's flags
        // (deliberately WITHOUT QUEUED — cursor entries are stack
        // placeholders and never carry queued tasks). The cursor
        // skip path runs first; the queued gate is irrelevant here
        // but pinning the absence prevents a future regression that
        // promotes "cursor + queued" to "real task with queued".
        let cursor_flags: u32 = 1 << 31;
        buf[(256 + flags_off)..(256 + flags_off + 4)].copy_from_slice(&cursor_flags.to_le_bytes());
        // Stamp a runnable_at on the cursor entry that WOULD synthesize
        // a giant age if the walker failed to skip.
        let cursor_runnable_at: u64 = 1; // jiffies-1=999 — would dominate.
        buf[(256 + runnable_at_off)..(256 + runnable_at_off + 8)]
            .copy_from_slice(&cursor_runnable_at.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 30,
            "cursor entry must be skipped — only task A's age 30 contributes"
        );
    }

    /// Cycle bail-out: if the list does not terminate, the walker
    /// stops at MAX_NODES rather than spinning forever.
    #[test]
    fn cycle_walker_terminates() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;

        // Self-cycle: head.next = node, node.next = node.
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;
        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&node_kva.to_le_bytes());
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 5;
        let runnable_at_pa = 64usize + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&runnable_at.to_le_bytes());
        // Stamp SCX_TASK_QUEUED so the task's age contributes before
        // the cycle bail-out fires.
        let flags_pa = 64usize + flags_off;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        // Walker visited the task at least once, so it captured age 5
        // before bailing out; the test confirms termination (no
        // panic, no infinite loop) AND a sensible reading.
        assert_eq!(age, 5);
    }

    /// Multi-task list with a zero-runnable_at task in the middle:
    /// the walker must skip the zero entry's age contribution but
    /// still continue stepping so that a real stall later in the
    /// list is observed.
    #[test]
    fn zero_runnable_at_does_not_abort_walk() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;

        // Task A at PA 64 — runnable_at = 0 (skip).
        // Task B at PA 256 — runnable_at = jiffies - 80 (real stall).
        let task_a_pa = 64u64;
        let task_a_kva = page_offset.wrapping_add(task_a_pa);
        let task_b_pa = 256u64;
        let task_b_kva = page_offset.wrapping_add(task_b_pa);

        // head -> A -> B -> head.
        buf[0..8].copy_from_slice(&task_a_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&task_b_kva.to_le_bytes());
        buf[256..264].copy_from_slice(&head_kva.to_le_bytes());

        let jiffies = 1_000u64;
        // task A runnable_at left at 0 (vec init zeroed it).
        // task B runnable_at = jiffies - 80.
        let runnable_at_b = jiffies - 80;
        buf[(256 + runnable_at_off)..(256 + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_b.to_le_bytes());
        // Stamp SCX_TASK_QUEUED on both tasks' see.flags. Task A's
        // QUEUED + runnable_at == 0 must hit the zero-skip branch
        // (not the unqueued-skip branch) so the test pins the
        // structural shape of the zero-skip continuation.
        buf[(64 + flags_off)..(64 + flags_off + 4)].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());
        buf[(256 + flags_off)..(256 + flags_off + 4)]
            .copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 80,
            "zero-runnable_at task must not abort the walk; \
             real stall later in the list must still surface",
        );
    }

    /// Production-like task.scx offset: real kernels place
    /// `task_struct.scx` at byte ~2528. The arithmetic
    ///   tasks_node_off_in_task = task.scx + see.tasks_node
    /// must compose correctly. Layout: place the task_struct so the
    /// `tasks_node` field lands at a known PA; verify container_of
    /// recovers the right `task_struct` KVA and the runnable_at
    /// read at `task + task.scx + see.runnable_at` lands the right
    /// value.
    #[test]
    fn nonzero_task_scx_offset() {
        // task_struct layout in the test:
        //   task_pa = 0x100
        //   task_struct.scx starts at offset 0x300 (production-like)
        //   sched_ext_entity.tasks_node at offset 0x60 within see
        //   sched_ext_entity.runnable_at at offset 0x18 within see
        //   sched_ext_entity.flags at offset 0x44 within see
        // Full offsets within task_struct:
        //   tasks_node = 0x300 + 0x60 = 0x360
        //   runnable_at = 0x300 + 0x18 = 0x318
        //   flags (in see) = 0x300 + 0x44 = 0x344
        let task_scx = 0x300usize;
        let tasks_node_off = 0x60usize;
        let runnable_at_off = 0x18usize;
        let flags_off = 0x44usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 0x1000];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        let task_pa: u64 = 0x100;
        let task_kva = page_offset.wrapping_add(task_pa);
        let tasks_node_kva = task_kva + (task_scx + tasks_node_off) as u64;

        // head.next = tasks_node_kva (single-task list).
        buf[0..8].copy_from_slice(&tasks_node_kva.to_le_bytes());
        // tasks_node.next = head (terminator). list_head.next at
        // offset 0 of the tasks_node, which is at task_pa + 0x360.
        let tasks_node_pa = task_pa as usize + task_scx + tasks_node_off;
        buf[tasks_node_pa..tasks_node_pa + 8].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at at task_pa + 0x318.
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 42;
        let runnable_at_pa = task_pa as usize + task_scx + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&runnable_at.to_le_bytes());
        // Stamp SCX_TASK_QUEUED on the task's see.flags so the
        // walker treats `runnable_at` as live. The flags field
        // lives inside the see at byte offset `flags_off`; the
        // see itself starts at byte offset `task_scx` inside the
        // task_struct, so the absolute PA is the same composition
        // the production walker reads.
        let flags_pa = task_pa as usize + task_scx + flags_off;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 42,
            "container_of must subtract task.scx + see.tasks_node, \
             not just see.tasks_node, to recover the right task_struct KVA"
        );
    }

    /// Negative case: a task linked into `scx_tasks` whose
    /// `SCX_TASK_QUEUED` flag is clear (sleeping / blocked /
    /// post-dequeue) carries a stale `runnable_at` value the
    /// kernel never resets — `clr_task_runnable` only sets
    /// `SCX_TASK_RESET_RUNNABLE_AT`, deferring the actual stamp
    /// to the next `set_task_runnable`. Using that stale stamp
    /// against the live `jiffies` would synthesize a multi-second
    /// (or multi-minute, on a long-sleeping task) "stall age" out
    /// of an entirely sleeping task and false-trigger the dual-
    /// snapshot's early path on every guest.
    ///
    /// This test pins the QUEUED-gate path: a non-queued task
    /// with a runnable_at value that WOULD synthesize a giant
    /// age must contribute nothing. Without the gate the walker
    /// would return `jiffies - runnable_at = 900`; with the gate
    /// the only path that survives is the fall-through to the
    /// next-node step, leaving max_age at 0.
    #[test]
    fn unqueued_task_with_stale_runnable_at_skipped() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;

        // Single-task list: head → task → head.
        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at = jiffies - 900. Without the queued gate
        // this would dominate the scan and synthesize a huge
        // false stall age.
        let jiffies = 1_000u64;
        let stale_runnable_at = jiffies - 900;
        let runnable_at_pa = 64usize + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&stale_runnable_at.to_le_bytes());
        // Flags: leave at 0 — neither QUEUED nor CURSOR set.
        // (vec![0u8; 4096] zero-inits; explicit no-op for clarity.)

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 0,
            "non-queued task must not contribute to max_age \
             regardless of runnable_at value (kernel does not \
             clear runnable_at on dequeue)"
        );
    }

    /// Mixed list: a queued task with a recent runnable_at AND
    /// an unqueued task with a much older runnable_at. The
    /// walker must pick the queued task's age (50) and skip the
    /// unqueued task's stale age (900) — without the queued
    /// gate the older value would dominate.
    #[test]
    fn queued_and_unqueued_mixed_takes_queued_age() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = START_KERNEL_MAP + head_pa;

        // Task A: queued, age 50 (recent).
        let task_a_pa = 64u64;
        let task_a_kva = page_offset.wrapping_add(task_a_pa);
        // Task B: unqueued, "age" 900 (stale stamp from a long
        // time ago — must be ignored).
        let task_b_pa = 256u64;
        let task_b_kva = page_offset.wrapping_add(task_b_pa);

        // head → A → B → head.
        buf[0..8].copy_from_slice(&task_a_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&task_b_kva.to_le_bytes());
        buf[256..264].copy_from_slice(&head_kva.to_le_bytes());

        let jiffies = 1_000u64;
        let runnable_at_a = jiffies - 50;
        let runnable_at_b = jiffies - 900;
        buf[(64 + runnable_at_off)..(64 + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_a.to_le_bytes());
        buf[(256 + runnable_at_off)..(256 + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_b.to_le_bytes());
        // A: QUEUED. B: 0 (neither QUEUED nor CURSOR).
        buf[(64 + flags_off)..(64 + flags_off + 4)].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());
        // task B flags left at 0.

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age_global(
            &mem,
            head_kva,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 50,
            "queued task age must win over an unqueued task's \
             stale runnable_at — the unqueued task's 900-jiffy \
             value is a leftover stamp, not a current stall"
        );
    }

    /// Per-rq walker: empty `runnable_list` (head.next == &head)
    /// returns 0. Mirrors the global walker's empty-list test.
    ///
    /// `rq_pa` must be non-zero: the walker treats `rq_pa == 0`
    /// as a sentinel meaning "no per-CPU rq PA available" and
    /// short-circuits before the empty-list check, so a zero
    /// `rq_pa` would make this test pass for the wrong reason
    /// (sentinel return, not empty-list traversal). The sentinel
    /// path is covered separately by `per_rq_zero_rq_pa_returns_zero`.
    #[test]
    fn per_rq_empty_list_returns_zero() {
        // Layout:
        //   rq lives at PA 64 in the direct map (KVA = page_offset + 64).
        //   rq.scx at offset 32; scx_rq.runnable_list at offset 8.
        //   So the head_pa is rq_pa + 32 + 8 = 104.
        //   head.next at head_kva (self-loop terminator) means empty.
        let rq_scx_off = 32usize;
        let scx_rq_runnable_list_off = 8usize;
        let head_offset = rq_scx_off + scx_rq_runnable_list_off;
        let page_offset = 0xffff_8880_0000_0000u64;
        let mut buf = vec![0u8; 4096];
        let rq_pa: u64 = 64;
        let head_pa = rq_pa + head_offset as u64;
        let head_kva = head_pa + page_offset;
        // head.next = &head (empty list).
        buf[head_pa as usize..head_pa as usize + 8].copy_from_slice(&head_kva.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = with_per_rq(
            test_offsets(0, 0, 16, 8),
            24,
            rq_scx_off,
            scx_rq_runnable_list_off,
        );
        let age = max_runnable_age_per_rq(
            &mem,
            rq_pa,
            &offsets,
            1_000,
            WalkContext {
                page_offset,
                ..Default::default()
            },
        );
        assert_eq!(age, 0, "empty per-rq runnable_list must return 0");
    }

    /// Per-rq walker: single stalled task on the per-CPU list.
    /// Verifies container_of through `runnable_node` (NOT
    /// `tasks_node` — those are separate list_heads on the same
    /// `sched_ext_entity`) and the per-rq head address composition
    /// `rq + rq_scx + runnable_list`.
    #[test]
    fn per_rq_single_stalled_task_age() {
        // Layout (chosen so each offset is distinct):
        //   rq at PA 64 in direct map.
        //   rq.scx at offset 32. scx_rq.runnable_list at offset 8.
        //   So the list_head for the per-rq list is at PA 104.
        //   head_kva = page_offset + 104.
        //   task at PA 256 in direct map.
        //   task.scx at offset 0; see.runnable_node at offset 24.
        //   see.runnable_at at offset 16.
        // node_kva (the runnable_node pointer the kernel links into
        // the list) = task_kva + (task_struct_scx +
        // sched_ext_entity_runnable_node) = page_offset + 256 + 24.
        //
        // rq must live at a non-zero PA: the walker treats `rq_pa ==
        // 0` as a sentinel meaning "no per-CPU rq PA available" and
        // short-circuits to age 0 (see `per_rq_zero_rq_pa_returns_zero`).
        let task_scx = 0usize;
        let runnable_node_off = 24usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let rq_scx_off = 32usize;
        let scx_rq_runnable_list_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let rq_pa: u64 = 64;
        let head_offset = rq_scx_off + scx_rq_runnable_list_off;
        let head_pa = rq_pa + head_offset as u64;
        let head_kva = head_pa + page_offset;
        let task_pa: u64 = 256;
        let task_kva = page_offset + task_pa;
        let node_kva = task_kva + (task_scx + runnable_node_off) as u64;

        // head.next = node_kva.
        buf[head_pa as usize..head_pa as usize + 8].copy_from_slice(&node_kva.to_le_bytes());
        // node.next = head_kva (terminator). The node sits at
        // task_pa + (task_scx + runnable_node_off).
        let node_pa = task_pa as usize + task_scx + runnable_node_off;
        buf[node_pa..node_pa + 8].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at at task + (task_scx + runnable_at_off).
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 75;
        let ra_pa = task_pa as usize + task_scx + runnable_at_off;
        buf[ra_pa..ra_pa + 8].copy_from_slice(&runnable_at.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = with_per_rq(
            test_offsets(task_scx, 0, runnable_at_off, flags_off),
            runnable_node_off,
            rq_scx_off,
            scx_rq_runnable_list_off,
        );
        let age = max_runnable_age_per_rq(
            &mem,
            rq_pa,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
        );
        assert_eq!(
            age, 75,
            "per-rq walker must compose rq.scx + scx_rq.runnable_list to find the head, \
             then container_of through runnable_node to reach the task"
        );
    }

    /// Per-rq walker contributes age 0 when `rq_pa == 0` (the
    /// caller is signalling "no per-CPU rq PA available"). Without
    /// this guard the walker would dereference offset-from-zero
    /// memory and synthesise spurious ages from whatever lives at
    /// the start of the GuestMem.
    #[test]
    fn per_rq_zero_rq_pa_returns_zero() {
        // GuestMem with a non-zero head value at PA 0 — if the
        // walker did NOT short-circuit on rq_pa==0 it would read
        // this bogus pointer and try to chase it.
        let page_offset = 0xffff_8880_0000_0000u64;
        let mut buf = vec![0u8; 4096];
        let bogus_head_kva = page_offset + 0xff_ff_ff_00u64;
        buf[40..48].copy_from_slice(&bogus_head_kva.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = with_per_rq(test_offsets(0, 0, 16, 8), 24, 32, 8);
        // Caller passes rq_pa == 0 — should short-circuit and not
        // read anything despite the buf containing a "valid-looking"
        // pointer at the address the head would otherwise compute.
        let age = max_runnable_age_per_rq(
            &mem,
            0,
            &offsets,
            1_000,
            WalkContext {
                page_offset,
                ..Default::default()
            },
        );
        assert_eq!(age, 0, "rq_pa == 0 must short-circuit the per-rq walker");
    }

    /// Combined `max_runnable_age` returns max(global, per_rq).
    /// Verifies the wrapper's max-of-walks invariant: a per-rq scan
    /// observing a higher age than the global walk wins, and vice
    /// versa.
    #[test]
    fn max_runnable_age_takes_max_across_walks() {
        // Global walk surfaces age 30; per-rq walk surfaces age 70.
        // Wrapper returns 70.
        //
        // Layout:
        //   Global head at PA 0 (text mapping).
        //   Global task A at PA 64 (direct map), age 30.
        //   Per-rq lives at rq PA 1024 (direct map). scx offset 32,
        //   runnable_list offset 8. head_pa = 1024 + 40 = 1064.
        //   Per-rq task B at PA 1280 (direct map), age 70.
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_node_off = 24usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let rq_scx_off = 32usize;
        let scx_rq_runnable_list_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 0x2000];
        // Global head at PA 0 (text mapping).
        let head_pa = 0u64;
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + head_pa;

        let task_a_pa = 64u64;
        let task_a_kva = page_offset + task_a_pa;

        // head.next -> task A's tasks_node (which is at task A's
        // base since tasks_node_off == 0).
        let node_a_kva = task_a_kva;
        buf[head_pa as usize..head_pa as usize + 8].copy_from_slice(&node_a_kva.to_le_bytes());
        // task A's tasks_node.next -> head (terminator).
        buf[task_a_pa as usize..task_a_pa as usize + 8].copy_from_slice(&head_kva.to_le_bytes());
        let jiffies = 1_000u64;
        let runnable_at_a = jiffies - 30;
        buf[(task_a_pa as usize + runnable_at_off)..(task_a_pa as usize + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_a.to_le_bytes());
        // Stamp QUEUED on task A so the global walker counts it.
        buf[(task_a_pa as usize + flags_off)..(task_a_pa as usize + flags_off + 4)]
            .copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // Per-rq layout.
        let rq_pa: u64 = 1024;
        let prq_head_pa = rq_pa + (rq_scx_off + scx_rq_runnable_list_off) as u64;
        let prq_head_kva = prq_head_pa + page_offset;
        let task_b_pa: u64 = 1280;
        let task_b_kva = page_offset + task_b_pa;
        let node_b_kva = task_b_kva + (task_scx + runnable_node_off) as u64;
        // per-rq head.next -> task B's runnable_node.
        buf[prq_head_pa as usize..prq_head_pa as usize + 8]
            .copy_from_slice(&node_b_kva.to_le_bytes());
        // task B's runnable_node.next -> per-rq head.
        let node_b_pa = task_b_pa as usize + task_scx + runnable_node_off;
        buf[node_b_pa..node_b_pa + 8].copy_from_slice(&prq_head_kva.to_le_bytes());
        let runnable_at_b = jiffies - 70;
        buf[(task_b_pa as usize + runnable_at_off)..(task_b_pa as usize + runnable_at_off + 8)]
            .copy_from_slice(&runnable_at_b.to_le_bytes());
        // Per-rq path doesn't gate on flags; leave unset.

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = with_per_rq(
            test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off),
            runnable_node_off,
            rq_scx_off,
            scx_rq_runnable_list_off,
        );
        let rq_pas = vec![rq_pa];
        let age = max_runnable_age(
            &mem,
            head_kva,
            &rq_pas,
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            None,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 70,
            "wrapper must take max(global, per-rq) — per-rq's 70 > global's 30"
        );
    }

    /// Combined wrapper takes the global walk's age when the
    /// per-rq path contributes nothing (empty rq_pas slice). The
    /// pre-fix code path used only the global walker; this pins
    /// the equivalent behaviour through the new wrapper API so
    /// callers without per-CPU rq PAs (no `runqueues` symbol,
    /// missing `__per_cpu_offset`, etc.) still get the global
    /// signal.
    #[test]
    fn max_runnable_age_empty_rq_pas_uses_global() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + head_pa;
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;

        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 42;
        let runnable_at_pa = 64usize + runnable_at_off;
        buf[runnable_at_pa..runnable_at_pa + 8].copy_from_slice(&runnable_at.to_le_bytes());
        let flags_pa = 64usize + flags_off;
        buf[flags_pa..flags_pa + 4].copy_from_slice(&SCX_TASK_QUEUED.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age(
            &mem,
            head_kva,
            &[],
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            None,
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 42,
            "empty rq_pas slice must defer to the global walk's age"
        );
    }

    /// `watchdog_timestamp_pa = Some(pa)` and the read yields a
    /// stale timestamp: the wrapper picks up the watchdog age
    /// even when both per-task walks contribute 0. Mirrors the
    /// production case where `scx_tick` would fire
    /// `SCX_EXIT_ERROR_STALL` because the workqueue stopped
    /// running but no individual task is stuck on a runnable_list.
    #[test]
    fn max_runnable_age_uses_watchdog_timestamp() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        // Empty global list (head.next = head, no tasks). PA 0 in
        // the text mapping.
        let head_pa = 0u64;
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + head_pa;
        buf[0..8].copy_from_slice(&head_kva.to_le_bytes());

        // Place the watchdog timestamp at PA 256: jiffies - 800
        // means the workqueue last ran 800 jiffies ago.
        let jiffies = 1_000u64;
        let stale_timestamp = jiffies - 800;
        let watchdog_pa: u64 = 256;
        buf[256..264].copy_from_slice(&stale_timestamp.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age(
            &mem,
            head_kva,
            &[],
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            Some(watchdog_pa),
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 800,
            "watchdog_timestamp must contribute when both per-task walks are empty",
        );
    }

    /// `watchdog_timestamp_pa = Some(pa)` but the read returns 0
    /// (unmapped guest memory or pre-init kernel): the wrapper
    /// must skip the contribution rather than synthesise
    /// `jiffies` worth of "age" out of nothing.
    #[test]
    fn max_runnable_age_zero_watchdog_timestamp_skipped() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + head_pa;
        buf[0..8].copy_from_slice(&head_kva.to_le_bytes());

        // PA 256 left at zero (vec![0u8; 4096] zero-init).
        let jiffies = 1_000u64;
        let watchdog_pa: u64 = 256;

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age(
            &mem,
            head_kva,
            &[],
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            Some(watchdog_pa),
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 0,
            "zero watchdog_timestamp must be skipped, not synthesise jiffies of age",
        );
    }

    /// `watchdog_timestamp_pa = Some(pa)` with an implausibly
    /// large age (timestamp at boot's `INITIAL_JIFFIES` while
    /// `jiffies` has advanced for hours without a sched_ext
    /// scheduler attaching): the sanity cap suppresses the
    /// contribution rather than synthesise a false stall. Models
    /// the boot-time pre-attach window where the workqueue is
    /// dormant.
    #[test]
    fn max_runnable_age_pre_attach_watchdog_capped() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + head_pa;
        buf[0..8].copy_from_slice(&head_kva.to_le_bytes());

        // Pre-attach scenario: timestamp pinned at INITIAL_JIFFIES
        // (large negative-cast value), jiffies advanced 100 million
        // jiffies past it (well over the 86.4M sanity cap).
        let timestamp = 1u64;
        let jiffies = 100_000_000u64;
        let watchdog_pa: u64 = 256;
        buf[256..264].copy_from_slice(&timestamp.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age(
            &mem,
            head_kva,
            &[],
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            Some(watchdog_pa),
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(
            age, 0,
            "implausibly large watchdog age (pre-attach boot window) must be capped to 0",
        );
    }

    /// `watchdog_timestamp_pa = Some(pa)` with a future
    /// timestamp (timestamp > jiffies): saturating_sub clamps
    /// the contribution to 0 instead of wrapping to near
    /// u64::MAX. Models the rare race where the workqueue
    /// updates the timestamp on a different CPU after our
    /// jiffies read.
    #[test]
    fn max_runnable_age_future_watchdog_timestamp_saturates() {
        let task_scx = 0usize;
        let tasks_node_off = 0usize;
        let runnable_at_off = 16usize;
        let flags_off = 8usize;
        let page_offset = 0xffff_8880_0000_0000u64;

        let mut buf = vec![0u8; 4096];
        let head_pa = 0u64;
        let head_kva = crate::monitor::symbols::START_KERNEL_MAP + head_pa;
        buf[0..8].copy_from_slice(&head_kva.to_le_bytes());

        let jiffies = 1_000u64;
        let future_timestamp = jiffies + 100;
        let watchdog_pa: u64 = 256;
        buf[256..264].copy_from_slice(&future_timestamp.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = test_offsets(task_scx, tasks_node_off, runnable_at_off, flags_off);
        let age = max_runnable_age(
            &mem,
            head_kva,
            &[],
            &offsets,
            jiffies,
            WalkContext {
                page_offset,
                ..Default::default()
            },
            Some(watchdog_pa),
            START_KERNEL_MAP,
            0,
        );
        assert_eq!(age, 0, "future watchdog_timestamp must saturate to age 0",);
    }
}
