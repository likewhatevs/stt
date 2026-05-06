//! Per-task failure-dump enrichment: read every Tier-1 task field the
//! failure-dump renderer surfaces in one pass.
//!
//! Given a task_struct KVA from a freeze-time walker (rq->scx walker
//! for runnable tasks, DSQ walker for queued tasks, init_task→tasks
//! walker for thread-group enumeration), this module reads:
//!
//! - Identity: pid, tgid, comm, group_leader_pid, real_parent_pid+comm
//! - Process tree: pgid, sid, nr_threads (via signal_struct)
//! - Scheduling: prio, static_prio, normal_prio, rt_priority,
//!   sched_class decoded to a name (CFS / RT / DL / IDLE / STOP / EXT),
//!   scx.weight, core_cookie (CONFIG_SCHED_CORE-gated)
//! - Watchdog disambiguation: `pi_boosted_out_of_scx` flag set when
//!   the runnable task's `sched_class` is not `ext_sched_class` (the
//!   PI-boost path moved it out and the failure isn't the BPF
//!   scheduler's fault — see scx core.c rt_mutex_setprio interactions)
//! - Context-switch counters: per-task nvcsw/nivcsw + per-thread-group
//!   signal->nvcsw/nivcsw
//! - Lock-contention hints: stack-trace pattern match against the
//!   sched_class symbol KVAs of `queued_spin_lock_slowpath`,
//!   `__mutex_lock_slowpath`, `rwsem_down_read_slowpath`,
//!   `rwsem_down_write_slowpath`. A PC inside any slowpath function
//!   on a runnable ('R') task indicates lock contention rather than
//!   scheduler fault. Stack-trace walking is only attempted when the
//!   caller supplies a non-empty stack-PC slice (typically harvested
//!   from the freeze coordinator's `VcpuRegSnapshot.instruction_pointer`
//!   for currently-running tasks; runnable-but-not-current tasks have
//!   no stack PCs without a kernel-side unwinder, which ktstr does
//!   not implement).
//!
//! The walker is best-effort: any pointer follow that fails to
//! translate yields a `None` for that derived field rather than
//! aborting the whole enrichment.

use serde::{Deserialize, Serialize};

use super::btf_offsets::{TaskEnrichmentOffsets, pid_type};
use super::guest::GuestKernel;
use super::idr::translate_any_kva;

/// Maximum bytes of `comm` to read.
///
/// Kernel-pinned at 16 by `include/linux/sched.h::TASK_COMM_LEN`. The
/// walker reads exactly this many bytes; trailing nuls are stripped
/// when forming the `String`.
const TASK_COMM_LEN: usize = 16;

/// Sched-class symbol KVAs cached for decode + watchdog
/// disambiguation. All six are resolved via vmlinux ELF symbol table
/// at coordinator start; missing symbols (kernel built without the
/// corresponding scheduling class — typically `dl_sched_class` on
/// CONFIG_SCHED_DEADLINE=n) leave the slot as `None`, and the
/// decoder returns `None` for that class.
///
/// All addresses are KVAs of the per-class `sched_class` static
/// (`fair_sched_class`, `rt_sched_class`, `dl_sched_class`,
/// `idle_sched_class`, `stop_sched_class`, `ext_sched_class`). On a
/// running guest, `task_struct.sched_class` points at exactly one of
/// these — comparing the read pointer to the cached set yields the
/// class name without needing a kallsyms parse.
#[derive(Debug, Clone, Default)]
pub struct SchedClassRegistry {
    pub fair: Option<u64>,
    pub rt: Option<u64>,
    pub dl: Option<u64>,
    pub idle: Option<u64>,
    pub stop: Option<u64>,
    pub ext: Option<u64>,
}

#[allow(dead_code)] // wired through DumpContext::TaskEnrichmentCapture;
// freeze coordinator passes None until the rq->scx
// walker lands a walker producer.
impl SchedClassRegistry {
    /// Resolve all six class symbols via the GuestKernel's vmlinux
    /// symbol table. Each lookup is independent — a missing symbol
    /// for one class doesn't fail the others.
    pub fn from_guest_kernel(kernel: &GuestKernel) -> Self {
        Self {
            fair: kernel.symbol_kva("fair_sched_class"),
            rt: kernel.symbol_kva("rt_sched_class"),
            dl: kernel.symbol_kva("dl_sched_class"),
            idle: kernel.symbol_kva("idle_sched_class"),
            stop: kernel.symbol_kva("stop_sched_class"),
            ext: kernel.symbol_kva("ext_sched_class"),
        }
    }

    /// Decode a `task_struct.sched_class` pointer to a class name.
    /// Returns `None` when the pointer matches no known class
    /// (stripped vmlinux, an out-of-tree class the kernel added,
    /// or a torn read landing on garbage).
    pub fn decode(&self, sched_class_kva: u64) -> Option<&'static str> {
        if sched_class_kva == 0 {
            return None;
        }
        if Some(sched_class_kva) == self.fair {
            return Some("fair");
        }
        if Some(sched_class_kva) == self.rt {
            return Some("rt");
        }
        if Some(sched_class_kva) == self.dl {
            return Some("dl");
        }
        if Some(sched_class_kva) == self.idle {
            return Some("idle");
        }
        if Some(sched_class_kva) == self.stop {
            return Some("stop");
        }
        if Some(sched_class_kva) == self.ext {
            return Some("ext");
        }
        None
    }
}

/// Lock-slowpath function KVAs. Used by the stack-trace lock detector
/// to flag runnable tasks whose stack contains a slowpath PC —
/// indicating the apparent scheduler stall is actually lock
/// contention, not BPF scheduler fault.
///
/// Each address is the function entry; the detector flags any PC in
/// `[start, start + LOCK_SLOWPATH_FN_MAX_SIZE)`. Without ELF symbol
/// size info we can't bound this exactly, so we use a conservative
/// 4 KiB window. False positives on adjacent functions are acceptable
/// for a diagnostic flag; false negatives only matter on slowpaths
/// longer than 4 KiB, none of which occur in mainline.
#[derive(Debug, Clone, Default)]
pub struct LockSlowpathRegistry {
    pub queued_spin_lock_slowpath: Option<u64>,
    pub mutex_lock_slowpath: Option<u64>,
    pub rwsem_down_read_slowpath: Option<u64>,
    pub rwsem_down_write_slowpath: Option<u64>,
}

/// Conservative max function size for stack-PC matching against
/// lock-slowpath entry symbols. See `LockSlowpathRegistry` doc.
const LOCK_SLOWPATH_FN_MAX_SIZE: u64 = 4096;

#[allow(dead_code)] // same wiring rationale as SchedClassRegistry above.
impl LockSlowpathRegistry {
    /// Resolve the four lock-slowpath symbols from the GuestKernel's
    /// vmlinux. Each lookup is independent; absent symbols leave the
    /// corresponding slot None and the matcher silently skips that
    /// pattern.
    pub fn from_guest_kernel(kernel: &GuestKernel) -> Self {
        Self {
            queued_spin_lock_slowpath: kernel.symbol_kva("queued_spin_lock_slowpath"),
            // `__mutex_lock_slowpath` is the historical name; modern
            // kernels (~4.15+) inline the slowpath into
            // `__mutex_lock`, but a leftover symbol remains in many
            // configs. Fall through to `__mutex_lock` if the
            // slowpath symbol is absent — both indicate the same
            // contention pattern at PC granularity.
            mutex_lock_slowpath: kernel
                .symbol_kva("__mutex_lock_slowpath")
                .or_else(|| kernel.symbol_kva("__mutex_lock")),
            rwsem_down_read_slowpath: kernel.symbol_kva("rwsem_down_read_slowpath"),
            rwsem_down_write_slowpath: kernel.symbol_kva("rwsem_down_write_slowpath"),
        }
    }

    /// Match a single PC against the four slowpath windows. Returns
    /// the pattern name when any window contains the PC, or `None`
    /// otherwise.
    pub fn match_pc(&self, pc: u64) -> Option<&'static str> {
        let probe = |start: Option<u64>, name: &'static str| -> Option<&'static str> {
            let s = start?;
            // Symbol KVAs come from the guest's vmlinux. A corrupt
            // ELF could place a slowpath symbol near u64::MAX; the
            // window upper bound `s + 4096` would wrap, and `pc <
            // wrapped` would falsely match every PC. checked_add
            // returning None means "this symbol can't define a
            // valid window" — treat as no match for that pattern.
            let end = s.checked_add(LOCK_SLOWPATH_FN_MAX_SIZE)?;
            if pc >= s && pc < end {
                Some(name)
            } else {
                None
            }
        };
        probe(self.queued_spin_lock_slowpath, "queued_spin_lock_slowpath")
            .or_else(|| probe(self.mutex_lock_slowpath, "mutex_lock_slowpath"))
            .or_else(|| probe(self.rwsem_down_read_slowpath, "rwsem_down_read_slowpath"))
            .or_else(|| probe(self.rwsem_down_write_slowpath, "rwsem_down_write_slowpath"))
    }
}

/// Per-task enrichment captured at freeze time.
///
/// Every field is best-effort: read failures (untranslatable RCU
/// pointer, slab-page eviction race, missing BTF field) yield `None`
/// rather than failing the whole capture. Optional fields cover both
/// "absent on this kernel build" (e.g. `core_cookie` without
/// CONFIG_SCHED_CORE) and "unreadable at this freeze instant" (e.g.
/// `real_parent_pid` when the parent task_struct's slab page didn't
/// translate).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TaskEnrichment {
    /// `task_struct.pid`. The kernel's per-task identifier.
    pub pid: i32,
    /// `task_struct.tgid`. Thread-group identifier (POSIX `getpid()`).
    pub tgid: i32,
    /// `task_struct.comm` truncated at the first nul byte.
    pub comm: String,
    /// `task_struct.group_leader->pid`. Pointer-followed; `None` on
    /// translate failure or NULL group_leader (init_task case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_leader_pid: Option<i32>,
    /// `task_struct.real_parent->pid`. RCU pointer-followed;
    /// `None` on translate failure or NULL real_parent (init_task).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_parent_pid: Option<i32>,
    /// `task_struct.real_parent->comm` truncated at the first nul.
    /// `None` if real_parent unreadable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_parent_comm: Option<String>,
    /// `signal->pids[PIDTYPE_PGID]->numbers[0].nr`. Process group id.
    /// `None` on signal_struct translate failure or NULL pids slot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pgid: Option<i32>,
    /// `signal->pids[PIDTYPE_SID]->numbers[0].nr`. Session id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sid: Option<i32>,
    /// `signal->nr_threads`. Live thread count for the thread group.
    /// `None` on signal_struct translate failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nr_threads: Option<i32>,
    /// `task_struct.scx.weight` (u32). scx-domain CFS-equivalent
    /// weight; 100 default.
    pub weight: u32,
    /// `task_struct.prio`. Effective scheduling priority
    /// (PI-boost-aware).
    pub prio: i32,
    /// `task_struct.static_prio`. User-set priority before PI boost.
    pub static_prio: i32,
    /// `task_struct.normal_prio`. Normal priority for the class.
    pub normal_prio: i32,
    /// `task_struct.rt_priority`. RT priority (1-99) for SCHED_FIFO/RR.
    pub rt_priority: u32,
    /// Decoded sched_class name: "fair", "rt", "dl", "idle", "stop",
    /// or "ext". `None` when the pointer matches no cached class
    /// (stripped vmlinux or out-of-tree class).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_class: Option<String>,
    /// `task_struct.core_cookie` (unsigned long).
    /// CONFIG_SCHED_CORE-gated; `None` on kernels built without it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_cookie: Option<u64>,
    /// True iff the task was on the rq->scx.runnable_list at freeze
    /// time AND `sched_class != ext_sched_class`. Indicates the PI
    /// boost path moved it out of SCX (rt_mutex_setprio) — failure
    /// is not the BPF scheduler's fault. Set only by the runnable
    /// walker; the queued-DSQ walker leaves this `false`.
    pub pi_boosted_out_of_scx: bool,
    /// `task_struct.nvcsw` (unsigned long). Voluntary context
    /// switches; the live thread count.
    pub nvcsw: u64,
    /// `task_struct.nivcsw` (unsigned long). Involuntary context
    /// switches.
    pub nivcsw: u64,
    /// `signal->nvcsw` (unsigned long). Thread-group accumulator
    /// for dead threads. `None` on signal_struct translate failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_nvcsw: Option<u64>,
    /// `signal->nivcsw` (unsigned long). Mirror of `signal_nvcsw`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_nivcsw: Option<u64>,
    /// Lock-slowpath pattern matched on a PC supplied by the caller
    /// (typically `VcpuRegSnapshot.instruction_pointer` for the task
    /// running on a vCPU at freeze time). One of
    /// "queued_spin_lock_slowpath", "mutex_lock_slowpath",
    /// "rwsem_down_read_slowpath", "rwsem_down_write_slowpath", or
    /// `None` when the supplied PC matched nothing OR the caller
    /// supplied no PCs.
    ///
    /// Set only when [`walk_task_enrichment_with_pcs`] is used; the
    /// no-PC entry point [`walk_task_enrichment`] always leaves this
    /// `None`. A stack walker that produces multiple PCs (a future
    /// kernel-side unwinder) would surface them as a `Vec<String>`
    /// in a non_exhaustive struct extension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lock_slowpath_match: Option<String>,
}

/// Walk one task_struct and populate every Tier-1 enrichment field.
/// `task_kva` must point at a valid `struct task_struct` reachable
/// via `translate_any_kva`. `is_runnable_in_scx` is set by the rq->scx
/// walker for tasks read off `rq->scx.runnable_list` (used for the
/// PI-boost-out-of-SCX flag); the queued-DSQ walker passes `false`.
///
/// `pc` (`Option<u64>`) is the task's instruction pointer for the
/// lock-slowpath stack matcher. Pass the corresponding vCPU's
/// `instruction_pointer` when this task was running on that vCPU at
/// freeze time; pass `None` for tasks not actively running (the
/// matcher needs an unwinder we don't have).
#[allow(dead_code)]
pub fn walk_task_enrichment(
    kernel: &GuestKernel,
    task_kva: u64,
    offsets: &TaskEnrichmentOffsets,
    classes: &SchedClassRegistry,
    locks: &LockSlowpathRegistry,
    is_runnable_in_scx: bool,
    pc: Option<u64>,
) -> Option<TaskEnrichment> {
    let mem = kernel.mem();
    let walk = kernel.walk_context();

    let task_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    )?;

    // Identity.
    let pid = mem.read_u32(task_pa, offsets.task_struct_pid) as i32;
    let tgid = mem.read_u32(task_pa, offsets.task_struct_tgid) as i32;
    let comm = read_comm(mem, task_pa, offsets.task_struct_comm);

    // Scheduling fields.
    let prio = mem.read_u32(task_pa, offsets.task_struct_prio) as i32;
    let static_prio = mem.read_u32(task_pa, offsets.task_struct_static_prio) as i32;
    let normal_prio = mem.read_u32(task_pa, offsets.task_struct_normal_prio) as i32;
    let rt_priority = mem.read_u32(task_pa, offsets.task_struct_rt_priority);
    let sched_class_kva = mem.read_u64(task_pa, offsets.task_struct_sched_class);
    let sched_class = classes.decode(sched_class_kva).map(str::to_string);
    let weight = mem.read_u32(task_pa, offsets.task_struct_scx + offsets.see_weight);
    let core_cookie = offsets
        .task_struct_core_cookie
        .map(|off| mem.read_u64(task_pa, off));

    // PI-boost-out-of-SCX flag: set only when the task was reached
    // via the rq->scx.runnable_list AND its current sched_class is
    // not ext_sched_class. This catches the case where rt_mutex_setprio
    // moved the task to a higher-prio class while it remained on the
    // SCX runnable list.
    let pi_boosted_out_of_scx =
        is_runnable_in_scx && classes.ext.is_some() && Some(sched_class_kva) != classes.ext;

    // Per-task context-switch counters.
    let nvcsw = mem.read_u64(task_pa, offsets.task_struct_nvcsw);
    let nivcsw = mem.read_u64(task_pa, offsets.task_struct_nivcsw);

    // Pointer follows: group_leader, real_parent, signal.
    let group_leader_kva = mem.read_u64(task_pa, offsets.task_struct_group_leader);
    let group_leader_pid =
        follow_task_for_pid(mem, walk, group_leader_kva, offsets.task_struct_pid);

    let real_parent_kva = mem.read_u64(task_pa, offsets.task_struct_real_parent);
    let (real_parent_pid, real_parent_comm) = follow_task_for_pid_and_comm(
        mem,
        walk,
        real_parent_kva,
        offsets.task_struct_pid,
        offsets.task_struct_comm,
    );

    let signal_kva = mem.read_u64(task_pa, offsets.task_struct_signal);
    let (nr_threads, signal_nvcsw, signal_nivcsw, pgid, sid) = if signal_kva == 0 {
        (None, None, None, None, None)
    } else {
        match translate_any_kva(
            mem,
            walk.cr3_pa,
            walk.page_offset,
            signal_kva,
            walk.l5,
            walk.tcr_el1,
        ) {
            None => (None, None, None, None, None),
            Some(signal_pa) => {
                let nr_threads_v = mem.read_u32(signal_pa, offsets.signal_struct_nr_threads) as i32;
                let signal_nvcsw_v = mem.read_u64(signal_pa, offsets.signal_struct_nvcsw);
                let signal_nivcsw_v = mem.read_u64(signal_pa, offsets.signal_struct_nivcsw);
                // pids[PIDTYPE_PGID] / pids[PIDTYPE_SID] traversal.
                // Each slot is `struct pid *` (8 bytes); the
                // numbers[0].nr deref reads the canonical root-ns
                // pid number.
                let pgid_v = read_pid_nr_at_index(
                    mem,
                    walk,
                    signal_pa,
                    offsets.signal_struct_pids,
                    pid_type::PGID,
                    offsets.pid_numbers,
                    offsets.upid_size,
                    offsets.upid_nr,
                );
                let sid_v = read_pid_nr_at_index(
                    mem,
                    walk,
                    signal_pa,
                    offsets.signal_struct_pids,
                    pid_type::SID,
                    offsets.pid_numbers,
                    offsets.upid_size,
                    offsets.upid_nr,
                );
                (
                    Some(nr_threads_v),
                    Some(signal_nvcsw_v),
                    Some(signal_nivcsw_v),
                    pgid_v,
                    sid_v,
                )
            }
        }
    };

    // Lock-slowpath PC match, if a PC was supplied.
    let lock_slowpath_match = pc.and_then(|p| locks.match_pc(p)).map(str::to_string);

    Some(TaskEnrichment {
        pid,
        tgid,
        comm,
        group_leader_pid,
        real_parent_pid,
        real_parent_comm,
        pgid,
        sid,
        nr_threads,
        weight,
        prio,
        static_prio,
        normal_prio,
        rt_priority,
        sched_class,
        core_cookie,
        pi_boosted_out_of_scx,
        nvcsw,
        nivcsw,
        signal_nvcsw,
        signal_nivcsw,
        lock_slowpath_match,
    })
}

/// Read `comm` as a `String` truncated at the first nul.
fn read_comm(mem: &super::reader::GuestMem, task_pa: u64, comm_off: usize) -> String {
    let mut buf = [0u8; TASK_COMM_LEN];
    mem.read_bytes(task_pa + comm_off as u64, &mut buf);
    let n = buf.iter().position(|&b| b == 0).unwrap_or(TASK_COMM_LEN);
    String::from_utf8_lossy(&buf[..n]).to_string()
}

/// Translate a `task_struct *` to its physical address and return
/// `(pid, comm)`. Returns `(None, None)` on any failure.
fn follow_task_for_pid_and_comm(
    mem: &super::reader::GuestMem,
    walk: super::reader::WalkContext,
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
    let comm = read_comm(mem, task_pa, comm_off);
    (Some(pid), Some(comm))
}

/// Translate a `task_struct *` and read just the pid.
fn follow_task_for_pid(
    mem: &super::reader::GuestMem,
    walk: super::reader::WalkContext,
    task_kva: u64,
    pid_off: usize,
) -> Option<i32> {
    if task_kva == 0 {
        return None;
    }
    let task_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        task_kva,
        walk.l5,
        walk.tcr_el1,
    )?;
    Some(mem.read_u32(task_pa, pid_off) as i32)
}

/// Read `signal->pids[idx]->numbers[0].nr`.
///
/// Three pointer hops:
///   1. `signal_pa + pids_off + idx * 8` reads the `struct pid *`.
///   2. Translate the pid pointer; the `numbers[0]` element starts at
///      `pid_pa + numbers_off`.
///   3. Read the `nr` field at `numbers[0] + nr_off`.
///
/// Returns `None` on any translate failure or when the pid pointer is
/// NULL (typical for `pids[PIDTYPE_PGID/SID]` on threads that aren't
/// session/process group leaders).
#[allow(clippy::too_many_arguments)]
fn read_pid_nr_at_index(
    mem: &super::reader::GuestMem,
    walk: super::reader::WalkContext,
    signal_pa: u64,
    pids_off: usize,
    idx: usize,
    numbers_off: usize,
    upid_size: usize,
    nr_off: usize,
) -> Option<i32> {
    let pid_kva = mem.read_u64(signal_pa, pids_off + idx * 8);
    if pid_kva == 0 {
        return None;
    }
    let pid_pa = translate_any_kva(
        mem,
        walk.cr3_pa,
        walk.page_offset,
        pid_kva,
        walk.l5,
        walk.tcr_el1,
    )?;
    // numbers[0] is at offset `numbers_off`; subsequent levels are at
    // `numbers_off + level * upid_size`. We always read level 0
    // (root pid namespace) per the kernel's `pid_nr` contract.
    let _ = upid_size; // captured in signature for level>0 callers
    Some(mem.read_u32(pid_pa, numbers_off + nr_off) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sched_class_registry_decode_known_class() {
        let r = SchedClassRegistry {
            fair: Some(0xffff_ffff_8000_1000),
            rt: Some(0xffff_ffff_8000_1100),
            dl: None,
            idle: None,
            stop: None,
            ext: Some(0xffff_ffff_8000_1300),
        };
        assert_eq!(r.decode(0xffff_ffff_8000_1000), Some("fair"));
        assert_eq!(r.decode(0xffff_ffff_8000_1100), Some("rt"));
        assert_eq!(r.decode(0xffff_ffff_8000_1300), Some("ext"));
    }

    #[test]
    fn sched_class_registry_decode_unknown_returns_none() {
        let r = SchedClassRegistry {
            fair: Some(0xffff_ffff_8000_1000),
            rt: None,
            dl: None,
            idle: None,
            stop: None,
            ext: None,
        };
        assert_eq!(r.decode(0xffff_ffff_8000_2000), None);
        // Zero pointer never decodes (would-be-NULL sched_class).
        assert_eq!(r.decode(0), None);
    }

    #[test]
    fn lock_slowpath_match_within_window() {
        let r = LockSlowpathRegistry {
            queued_spin_lock_slowpath: Some(0xffff_ffff_8001_0000),
            mutex_lock_slowpath: Some(0xffff_ffff_8002_0000),
            rwsem_down_read_slowpath: None,
            rwsem_down_write_slowpath: None,
        };
        // Inside the qsl window.
        assert_eq!(
            r.match_pc(0xffff_ffff_8001_0010),
            Some("queued_spin_lock_slowpath")
        );
        // Inside the mutex window.
        assert_eq!(
            r.match_pc(0xffff_ffff_8002_0fff),
            Some("mutex_lock_slowpath")
        );
        // Past the qsl window (4 KiB cap).
        assert!(r.match_pc(0xffff_ffff_8001_2000).is_none());
        // Before the qsl window.
        assert!(r.match_pc(0xffff_ffff_8000_ffff).is_none());
    }

    #[test]
    fn lock_slowpath_no_match_when_all_none() {
        let r = LockSlowpathRegistry::default();
        assert_eq!(r.match_pc(0xdeadbeef), None);
    }

    /// Pin the wire shape of `TaskEnrichment` — every optional field
    /// should skip on `None` so a populated capture renders cleanly
    /// without a wall of nulls in the JSON.
    #[test]
    fn task_enrichment_serde_skip_none_fields() {
        let e = TaskEnrichment {
            pid: 42,
            tgid: 42,
            comm: "ktstr_worker".to_string(),
            group_leader_pid: None,
            real_parent_pid: None,
            real_parent_comm: None,
            pgid: None,
            sid: None,
            nr_threads: None,
            weight: 100,
            prio: 120,
            static_prio: 120,
            normal_prio: 120,
            rt_priority: 0,
            sched_class: Some("fair".to_string()),
            core_cookie: None,
            pi_boosted_out_of_scx: false,
            nvcsw: 0,
            nivcsw: 0,
            signal_nvcsw: None,
            signal_nivcsw: None,
            lock_slowpath_match: None,
        };
        let json = serde_json::to_string(&e).unwrap();
        // Skipped fields must not appear in the JSON.
        assert!(!json.contains("group_leader_pid"));
        assert!(!json.contains("real_parent_pid"));
        assert!(!json.contains("pgid"));
        assert!(!json.contains("nr_threads"));
        assert!(!json.contains("core_cookie"));
        assert!(!json.contains("signal_nvcsw"));
        assert!(!json.contains("lock_slowpath_match"));
        // Required fields must appear.
        assert!(json.contains("\"pid\":42"));
        assert!(json.contains("\"comm\":\"ktstr_worker\""));
        assert!(json.contains("\"weight\":100"));
        assert!(json.contains("\"sched_class\":\"fair\""));
    }

    #[test]
    fn task_enrichment_serde_roundtrip_populated() {
        let e = TaskEnrichment {
            pid: 1234,
            tgid: 1230,
            comm: "stress-ng".to_string(),
            group_leader_pid: Some(1230),
            real_parent_pid: Some(1),
            real_parent_comm: Some("systemd".to_string()),
            pgid: Some(1230),
            sid: Some(1),
            nr_threads: Some(8),
            weight: 200,
            prio: 100,
            static_prio: 120,
            normal_prio: 100,
            rt_priority: 50,
            sched_class: Some("rt".to_string()),
            core_cookie: Some(0xc0c01e),
            pi_boosted_out_of_scx: true,
            nvcsw: 12345,
            nivcsw: 678,
            signal_nvcsw: Some(50_000),
            signal_nivcsw: Some(1_234),
            lock_slowpath_match: Some("queued_spin_lock_slowpath".to_string()),
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: TaskEnrichment = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pid, 1234);
        assert_eq!(parsed.comm, "stress-ng");
        assert_eq!(parsed.real_parent_comm.as_deref(), Some("systemd"));
        assert_eq!(parsed.nr_threads, Some(8));
        assert_eq!(parsed.core_cookie, Some(0xc0c01e));
        assert!(parsed.pi_boosted_out_of_scx);
        assert_eq!(
            parsed.lock_slowpath_match.as_deref(),
            Some("queued_spin_lock_slowpath"),
        );
    }
}
