//! Per-CPU runnable_at scanner for the dual-snapshot freeze coordinator.
//!
//! Mirrors the kernel's `check_rq_for_timeouts`
//! (`kernel/sched/ext.c`) walk:
//!
//! ```ignore
//! list_for_each_entry(p, &rq->scx.runnable_list, scx.runnable_node) {
//!     unsigned long last_runnable = p->scx.runnable_at;
//!     if (time_after(jiffies, last_runnable + sch->watchdog_timeout)) ...
//! }
//! ```
//!
//! Walked from the host: each per-CPU `struct rq`'s
//! `scx.runnable_list` head, follow `list_head.next` pointers,
//! container_of back to `task_struct`, read `scx.runnable_at`,
//! compare against the current `jiffies_64`. The maximum
//! `jiffies - runnable_at` across all runnable tasks on all CPUs is
//! the trigger threshold for the dual-snapshot early capture.
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
//! page-walk code path the BPF map dump uses). Each `task_struct`
//! is a slab object inside the kernel direct map, so the walk is
//! page-walk-light: one translate per task to find the runnable_at
//! field. The list head and `next` pointers live inside the per-CPU
//! `rq`, whose PA is already cached.

use super::reader::GuestMem;
use crate::monitor::btf_offsets::RunnableScanOffsets;

/// Cap on the number of nodes the walker visits per CPU before
/// bailing out.
///
/// Defends against a corrupted `next` pointer that fails to terminate
/// (cycle that doesn't include the head, or a stray pointer into
/// arbitrary memory). The kernel scheduler does not normally have
/// thousands of runnable tasks per CPU; even a stress workload tops
/// out in the low hundreds. Picking 4096 keeps the bound well above
/// realistic kernels and well below the cost of a runaway walk.
const MAX_NODES_PER_CPU: u32 = 4096;

/// Walk every per-CPU `rq->scx.runnable_list` and return the maximum
/// `jiffies - p->scx.runnable_at` observed across all runnable tasks
/// on all CPUs.
///
/// `rq_pas` is the per-CPU array of `struct rq` physical addresses
/// (already computed by [`super::symbols::compute_rq_pas`]).
///
/// `rq_kvas` is the corresponding per-CPU array of `struct rq`
/// kernel virtual addresses (parallel to `rq_pas`, indexed by CPU).
/// Both forms are needed because the `runnable_list` walk uses each
/// CPU's runnable_list head as the loop terminator: the kernel's
/// `list_for_each_entry` exit condition compares each visited
/// node's `next` pointer against the head's KVA (not its PA), so
/// the walker carries the head KVA explicitly. Reads still go
/// through the PA via `mem.read_u64(rq_pa + …, 0)` — only the
/// terminator comparison is KVA-vs-KVA. Caller derives
/// `rq_kvas[i] = rq_pas[i] + page_offset` since DRAM offset 0 maps
/// at PAGE_OFFSET on both x86_64 and aarch64.
///
/// `jiffies` is the current `jiffies_64` value (caller reads this
/// from the pre-translated `jiffies_64_pa`). `page_offset` is the
/// runtime PAGE_OFFSET for `kva_to_pa` direct-mapping translation;
/// `cr3_pa` and `l5` feed the page-walk fallback through
/// [`super::idr::translate_any_kva`] for KVAs that fall outside
/// the direct map (vmalloc-backed slab objects on some
/// configurations).
///
/// Returns 0 when no CPU has a non-empty runnable_list, or when
/// every task's `runnable_at` is newer than `jiffies` (the
/// `saturating_sub` clamps a future-runnable_at down to 0, which
/// the caller's threshold check ignores).
///
/// Tasks whose `runnable_at == 0` are skipped without contributing
/// to max — kernel-side `runnable_at` is zero between slab
/// allocation and the first `set_task_runnable` call, and treating
/// that as age-from-zero would synthesize spurious multi-day
/// stalls.
///
/// Best-effort: a single CPU with an unmapped task page contributes
/// nothing for that task but does not abort the scan, and a
/// corrupted list contributes whatever was scanned before the
/// bail-out, capped at MAX_NODES_PER_CPU visits. The caller still
/// sees the genuine max from every other CPU.
#[allow(clippy::too_many_arguments)]
pub fn max_runnable_age(
    mem: &GuestMem,
    rq_pas: &[u64],
    rq_kvas: &[u64],
    offsets: &RunnableScanOffsets,
    rq_scx_offset: usize,
    jiffies: u64,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
) -> u64 {
    debug_assert_eq!(
        rq_pas.len(),
        rq_kvas.len(),
        "rq_pas and rq_kvas must be the same length",
    );
    let mut max_age: u64 = 0;
    for (cpu_idx, (&rq_pa, &rq_kva)) in rq_pas.iter().zip(rq_kvas.iter()).enumerate() {
        let cpu_max = scan_one_cpu(
            mem,
            rq_pa,
            rq_kva,
            offsets,
            rq_scx_offset,
            jiffies,
            cr3_pa,
            page_offset,
            l5,
            cpu_idx,
        );
        if cpu_max > max_age {
            max_age = cpu_max;
        }
    }
    max_age
}

/// Walk one CPU's `runnable_list` and return the maximum
/// `jiffies - runnable_at` for that CPU. 0 means either the list
/// is empty or the walk failed before finding a stalled task.
#[allow(clippy::too_many_arguments)]
fn scan_one_cpu(
    mem: &GuestMem,
    rq_pa: u64,
    rq_kva: u64,
    offsets: &RunnableScanOffsets,
    rq_scx_offset: usize,
    jiffies: u64,
    cr3_pa: u64,
    page_offset: u64,
    l5: bool,
    cpu_idx: usize,
) -> u64 {
    // Head of the runnable_list lives inside this CPU's rq:
    //   head_kva = rq_kva + rq_scx_offset + scx_rq_runnable_list
    //   head_pa  = rq_pa  + rq_scx_offset + scx_rq_runnable_list
    let list_head_offset = rq_scx_offset + offsets.scx_rq_runnable_list;
    let head_kva = rq_kva.wrapping_add(list_head_offset as u64);
    let head_pa = rq_pa.wrapping_add(list_head_offset as u64);

    // Read head.next. struct list_head { next; prev; } — `next` at
    // offset 0. Empty list: head.next == &head, so the loop exits
    // on the first iteration without reading any task.
    let mut node_kva = mem.read_u64(head_pa, 0);
    if node_kva == 0 {
        // Defensive: a zero `next` pointer means either an
        // uninitialized rq (guest still booting) or guest memory the
        // host could not read (PA out of range — read_u64 returns 0
        // for unmapped). Skip without contributing to max.
        return 0;
    }

    let mut max_age: u64 = 0;
    let mut visited: u32 = 0;
    while node_kva != head_kva {
        if visited >= MAX_NODES_PER_CPU {
            // Cycle or runaway pointer — bail out with what we have.
            // tracing::warn would amplify a transient corruption into
            // log spam, so swallow silently. The freeze coord's normal
            // dump path still surfaces the underlying problem.
            return max_age;
        }
        visited += 1;

        // node_kva points at a `struct list_head` that lives inside
        // a `task_struct`'s `scx.runnable_node`. container_of recovers
        // the task_struct KVA:
        //   task_kva = node_kva
        //            - (task_struct.scx + sched_ext_entity.runnable_node)
        let runnable_node_off_in_task =
            offsets.task_struct_scx + offsets.sched_ext_entity_runnable_node;
        let task_kva = node_kva.wrapping_sub(runnable_node_off_in_task as u64);

        // Read p->scx.runnable_at. task_struct lives in slab; on
        // most configs that's the direct map (kva_to_pa works), but
        // some BPF arenas / vmalloc-backed slabs require a page
        // walk. Use translate_any_kva so the latter still resolves.
        let runnable_at_kva = task_kva
            .wrapping_add(offsets.task_struct_scx as u64)
            .wrapping_add(offsets.sched_ext_entity_runnable_at as u64);
        if let Some(runnable_at_pa) =
            super::idr::translate_any_kva(mem, cr3_pa, page_offset, runnable_at_kva, l5)
        {
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
            // falsely trigger the early-snapshot path. The walk
            // continues past the skip — the next-node step at the
            // bottom of the loop runs unconditionally, so other
            // tasks on this CPU's list still contribute.
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
            // to max. Tag with cpu_idx so a spike of failures per
            // CPU is visible if anyone enables debug logging.
            tracing::debug!(
                cpu = cpu_idx,
                task_kva = format_args!("{task_kva:#x}"),
                "runnable_scan: task page untranslatable, skipping",
            );
        }

        // Step to next node. Read node.next (offset 0 of list_head).
        // Translate the node KVA to PA each step — successive
        // task_structs may live in different slab pages, so caching
        // a single PA does not help.
        let node_pa = match super::idr::translate_any_kva(mem, cr3_pa, page_offset, node_kva, l5) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::reader::GuestMem;

    /// Empty list: head.next == &head. Returns 0.
    #[test]
    fn empty_list_returns_zero() {
        // Layout: at PA 0 is one rq.
        // rq_scx_offset + scx_rq_runnable_list = 0 (we put the head
        // right at the rq base for test brevity).
        // head { next: &head, prev: &head }.
        let mut buf = vec![0u8; 4096];
        // page_offset chosen so kva_to_pa(kva, page_offset) = pa.
        let page_offset = 0xffff_8880_0000_0000u64;
        let head_pa = 0u64;
        let head_kva = page_offset.wrapping_add(head_pa);
        // head.next = head_kva (empty list terminator).
        buf[0..8].copy_from_slice(&head_kva.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list: 0,
            task_struct_scx: 16,
            sched_ext_entity_runnable_node: 0,
            sched_ext_entity_runnable_at: 8,
        };
        let age = max_runnable_age(
            &mem,
            &[head_pa],
            &[head_kva],
            &offsets,
            0,
            10_000,
            0, // cr3_pa unused on direct-map path
            page_offset,
            false,
        );
        assert_eq!(age, 0, "empty list must return age 0");
    }

    /// Future runnable_at (runnable_at > jiffies) saturates to age 0
    /// rather than wrapping to a huge u64.
    #[test]
    fn future_runnable_at_treated_as_age_zero() {
        // Direct-map only: with cr3_pa=0 and a flat buffer,
        // translate_any_kva's direct-map branch should resolve KVAs
        // back into our buffer. Build:
        //   task_struct at KVA T (PA = T - page_offset)
        //   T + task_struct_scx + sched_ext_entity_runnable_node = node KVA N
        //   T + task_struct_scx + sched_ext_entity_runnable_at = read site
        //
        // Layout:
        //   PA 0:  rq with head at offset 0; head.next = N, head.prev = N
        //   PA 64: task_struct (task_struct_scx = 0 in test for brevity)
        //          - sched_ext_entity at offset 0; runnable_node at offset 0
        //            => node KVA = T (PA 64) + 0 + 0 = T
        //          - runnable_at at offset 16; value = jiffies + 100 (future)
        let task_struct_scx = 0usize;
        let sched_ext_entity_runnable_node = 0usize;
        let sched_ext_entity_runnable_at = 16usize;
        let scx_rq_runnable_list = 0usize;

        let mut buf = vec![0u8; 4096];
        let page_offset = 0xffff_8880_0000_0000u64;

        let head_pa = 0u64;
        let head_kva = page_offset.wrapping_add(head_pa);
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva + (task_struct_scx + sched_ext_entity_runnable_node) as u64;

        // head.next = node_kva (single-task list). head.next at PA 0.
        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        // node.next = head_kva (terminator). node lives at offset 0
        // of the task_struct, which is at PA 64.
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at at task_pa + 16; value = future.
        let jiffies = 1_000u64;
        let future = jiffies + 100;
        buf[(64 + sched_ext_entity_runnable_at)..(64 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&future.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list,
            task_struct_scx,
            sched_ext_entity_runnable_node,
            sched_ext_entity_runnable_at,
        };
        let age = max_runnable_age(
            &mem,
            &[head_pa],
            &[head_kva],
            &offsets,
            0,
            jiffies,
            0,
            page_offset,
            false,
        );
        assert_eq!(age, 0, "future runnable_at must saturate to age 0");
    }

    /// Single stalled task: runnable_at = jiffies - 50, expected age 50.
    #[test]
    fn single_stalled_task_age() {
        let task_struct_scx = 0usize;
        let sched_ext_entity_runnable_node = 0usize;
        let sched_ext_entity_runnable_at = 16usize;
        let scx_rq_runnable_list = 0usize;

        let mut buf = vec![0u8; 4096];
        let page_offset = 0xffff_8880_0000_0000u64;

        let head_pa = 0u64;
        let head_kva = page_offset.wrapping_add(head_pa);
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;

        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 50;
        buf[(64 + sched_ext_entity_runnable_at)..(64 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&runnable_at.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list,
            task_struct_scx,
            sched_ext_entity_runnable_node,
            sched_ext_entity_runnable_at,
        };
        let age = max_runnable_age(
            &mem,
            &[head_pa],
            &[head_kva],
            &offsets,
            0,
            jiffies,
            0,
            page_offset,
            false,
        );
        assert_eq!(age, 50);
    }

    /// `runnable_at == 0` is treated as "not yet stamped" and
    /// contributes nothing to max_age. Without this guard a fresh
    /// zero-init task_struct on the runnable_list would synthesize
    /// `jiffies` of "stall age" out of an unrelated task.
    #[test]
    fn zero_runnable_at_skipped() {
        let task_struct_scx = 0usize;
        let sched_ext_entity_runnable_node = 0usize;
        let sched_ext_entity_runnable_at = 16usize;
        let scx_rq_runnable_list = 0usize;

        let mut buf = vec![0u8; 4096];
        let page_offset = 0xffff_8880_0000_0000u64;

        let head_pa = 0u64;
        let head_kva = page_offset.wrapping_add(head_pa);
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;

        // Single-task list: head.next = node, node.next = head.
        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head_kva.to_le_bytes());
        // runnable_at = 0 — task on the list but not yet stamped.
        // (Zeroed by `vec![0u8; 4096]`; explicit for clarity.)
        let runnable_at = 0u64;
        buf[(64 + sched_ext_entity_runnable_at)..(64 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&runnable_at.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list,
            task_struct_scx,
            sched_ext_entity_runnable_node,
            sched_ext_entity_runnable_at,
        };
        // jiffies non-zero so without the guard the synthetic age
        // would be `jiffies - 0 = jiffies` (a giant number).
        let jiffies = 1_000u64;
        let age = max_runnable_age(
            &mem,
            &[head_pa],
            &[head_kva],
            &offsets,
            0,
            jiffies,
            0,
            page_offset,
            false,
        );
        assert_eq!(
            age, 0,
            "runnable_at == 0 must be skipped, not treated as a 1000-jiffy stall"
        );
    }

    /// Two CPUs, each with one task; max wins.
    #[test]
    fn max_across_cpus() {
        let task_struct_scx = 0usize;
        let sched_ext_entity_runnable_node = 0usize;
        let sched_ext_entity_runnable_at = 16usize;
        let scx_rq_runnable_list = 0usize;

        let mut buf = vec![0u8; 4096];
        let page_offset = 0xffff_8880_0000_0000u64;

        // CPU 0: rq head at PA 0, task at PA 64, age = 30.
        let head0_pa = 0u64;
        let head0_kva = page_offset.wrapping_add(head0_pa);
        let task0_pa = 64u64;
        let task0_kva = page_offset.wrapping_add(task0_pa);
        let node0_kva = task0_kva;
        buf[0..8].copy_from_slice(&node0_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&head0_kva.to_le_bytes());

        // CPU 1: rq head at PA 256, task at PA 320, age = 70.
        let head1_pa = 256u64;
        let head1_kva = page_offset.wrapping_add(head1_pa);
        let task1_pa = 320u64;
        let task1_kva = page_offset.wrapping_add(task1_pa);
        let node1_kva = task1_kva;
        buf[256..264].copy_from_slice(&node1_kva.to_le_bytes());
        buf[320..328].copy_from_slice(&head1_kva.to_le_bytes());

        let jiffies = 1_000u64;
        let runnable_at_0 = jiffies - 30;
        let runnable_at_1 = jiffies - 70;
        buf[(64 + sched_ext_entity_runnable_at)..(64 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&runnable_at_0.to_le_bytes());
        buf[(320 + sched_ext_entity_runnable_at)..(320 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&runnable_at_1.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list,
            task_struct_scx,
            sched_ext_entity_runnable_node,
            sched_ext_entity_runnable_at,
        };
        let age = max_runnable_age(
            &mem,
            &[head0_pa, head1_pa],
            &[head0_kva, head1_kva],
            &offsets,
            0,
            jiffies,
            0,
            page_offset,
            false,
        );
        assert_eq!(age, 70, "scan must take the max across CPUs");
    }

    /// Cycle bail-out: if the list does not terminate, the walker
    /// stops at MAX_NODES_PER_CPU rather than spinning forever.
    #[test]
    fn cycle_walker_terminates() {
        let task_struct_scx = 0usize;
        let sched_ext_entity_runnable_node = 0usize;
        let sched_ext_entity_runnable_at = 16usize;
        let scx_rq_runnable_list = 0usize;

        let mut buf = vec![0u8; 4096];
        let page_offset = 0xffff_8880_0000_0000u64;

        // Self-cycle: head.next = node, node.next = node.
        let head_pa = 0u64;
        let head_kva = page_offset.wrapping_add(head_pa);
        let task_pa = 64u64;
        let task_kva = page_offset.wrapping_add(task_pa);
        let node_kva = task_kva;
        buf[0..8].copy_from_slice(&node_kva.to_le_bytes());
        buf[64..72].copy_from_slice(&node_kva.to_le_bytes());
        let jiffies = 1_000u64;
        let runnable_at = jiffies - 5;
        buf[(64 + sched_ext_entity_runnable_at)..(64 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&runnable_at.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list,
            task_struct_scx,
            sched_ext_entity_runnable_node,
            sched_ext_entity_runnable_at,
        };
        let age = max_runnable_age(
            &mem,
            &[head_pa],
            &[head_kva],
            &offsets,
            0,
            jiffies,
            0,
            page_offset,
            false,
        );
        // Walker visited the task at least once, so it captured
        // age 5 before bailing out; the test confirms termination
        // (no panic, no infinite loop) AND a sensible reading.
        assert_eq!(age, 5);
    }

    /// Multi-task list with a zero-runnable_at task in the middle:
    /// the walker must skip the zero entry's age contribution but
    /// still continue stepping so that a real stall later in the
    /// list is observed. This pins the structural shape of the
    /// fix — the skip must not abort the per-CPU walk.
    #[test]
    fn zero_runnable_at_does_not_abort_walk() {
        let task_struct_scx = 0usize;
        let sched_ext_entity_runnable_node = 0usize;
        let sched_ext_entity_runnable_at = 16usize;
        let scx_rq_runnable_list = 0usize;

        let mut buf = vec![0u8; 4096];
        let page_offset = 0xffff_8880_0000_0000u64;

        // Layout:
        //   PA 0:    rq head
        //   PA 64:   task A — runnable_at = 0 (skip)
        //   PA 128:  task B — runnable_at = jiffies - 80 (real stall)
        let head_pa = 0u64;
        let head_kva = page_offset.wrapping_add(head_pa);
        let task_a_pa = 64u64;
        let task_a_kva = page_offset.wrapping_add(task_a_pa);
        let task_b_pa = 128u64;
        let task_b_kva = page_offset.wrapping_add(task_b_pa);

        // head.next = task A
        buf[0..8].copy_from_slice(&task_a_kva.to_le_bytes());
        // task_a.next = task B
        buf[64..72].copy_from_slice(&task_b_kva.to_le_bytes());
        // task_b.next = head (terminator)
        buf[128..136].copy_from_slice(&head_kva.to_le_bytes());

        let jiffies = 1_000u64;
        // task A runnable_at left at 0 (vec init zeroed it).
        // task B runnable_at = jiffies - 80.
        let runnable_at_b = jiffies - 80;
        buf[(128 + sched_ext_entity_runnable_at)..(128 + sched_ext_entity_runnable_at + 8)]
            .copy_from_slice(&runnable_at_b.to_le_bytes());

        // SAFETY: buf outlives mem.
        let mem = unsafe { GuestMem::new(buf.as_mut_ptr(), buf.len() as u64) };
        let offsets = RunnableScanOffsets {
            scx_rq_runnable_list,
            task_struct_scx,
            sched_ext_entity_runnable_node,
            sched_ext_entity_runnable_at,
        };
        let age = max_runnable_age(
            &mem,
            &[head_pa],
            &[head_kva],
            &offsets,
            0,
            jiffies,
            0,
            page_offset,
            false,
        );
        assert_eq!(
            age, 80,
            "zero-runnable_at task must not abort the walk; \
             real stall later in the list must still surface",
        );
    }
}
