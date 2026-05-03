//! Cgroup v2 filesystem operations for test cgroup management.
//!
//! Creates, configures, and removes cgroups under a parent path
//! (default `/sys/fs/cgroup/ktstr`). Provides cpuset assignment,
//! task migration, and cleanup.

use crate::topology::TestTopology;
use anyhow::{Context, Result, bail};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

/// Default timeout for cgroup filesystem writes. Normally <1ms; 2s catches
/// real hangs without waiting so long the test result is meaningless.
const CGROUP_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Write `data` to `path` with a timeout. Spawns a thread for the blocking
/// `fs::write` and waits on a channel. If the write does not complete within
/// `timeout`, returns an error (the spawned thread may still be blocked in
/// the kernel but will not prevent the caller from making progress).
fn write_with_timeout(path: &Path, data: &str, timeout: Duration) -> Result<()> {
    let display = path.display().to_string();
    let path = path.to_owned();
    let data = data.to_owned();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = fs::write(&path, &data);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(timeout) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            let errno_suffix = e
                .raw_os_error()
                .and_then(crate::errno_name)
                .map(|name| format!(" ({name})"))
                .unwrap_or_default();
            Err(e).with_context(|| format!("write {display}{errno_suffix}"))
        }
        Err(_) => bail!(
            "cgroup write to {display} timed out after {}ms",
            timeout.as_millis()
        ),
    }
}

/// Walk an `anyhow::Error` chain and return the first
/// `std::io::Error`'s raw errno, if any. Shared helper for errno
/// classification across cgroup orchestration — both this module's
/// ESRCH/EBUSY checks and [`crate::vmm::cgroup_sandbox`]'s
/// EACCES/EPERM/EBUSY branches walk the same chain shape.
pub(crate) fn anyhow_first_io_errno(err: &anyhow::Error) -> Option<i32> {
    err.chain()
        .find_map(|cause| cause.downcast_ref::<std::io::Error>())
        .and_then(|io| io.raw_os_error())
}

/// ESRCH: task exited between listing and migration
/// (`cgroup_procs_write_start` -> `find_task_by_vpid` returns NULL).
fn is_esrch(err: &anyhow::Error) -> bool {
    anyhow_first_io_errno(err) == Some(libc::ESRCH)
}

/// EBUSY: either the cgroup v2 no-internal-process constraint
/// (`cgroup_migrate_vet_dst` when `subtree_control` is set) or a
/// transient rejection from a sched_ext BPF `cgroup_prep_move`
/// callback (`scx_cgroup_can_attach`).
fn is_ebusy(err: &anyhow::Error) -> bool {
    anyhow_first_io_errno(err) == Some(libc::EBUSY)
}

/// RAII manager for cgroup v2 filesystem operations.
///
/// Creates, configures, and removes cgroups under a parent directory.
/// Provides cpuset assignment and task migration.
#[derive(Debug)]
pub struct CgroupManager {
    parent: PathBuf,
}

impl CgroupManager {
    /// Create a manager rooted at the given cgroup v2 path.
    pub fn new(parent: &str) -> Self {
        Self {
            parent: PathBuf::from(parent),
        }
    }
    /// Path to the parent cgroup directory.
    pub fn parent_path(&self) -> &std::path::Path {
        &self.parent
    }

    /// Create the parent directory and enable cgroup controllers
    /// (cpuset, optionally cpu, plus memory + pids + io unconditionally).
    ///
    /// `enable_cpu_controller` gates `+cpu` only — the memory, pids, and
    /// io controllers are always enabled because the test framework's
    /// CgroupDef builders (`memory_max`, `pids_max`, `io_weight`,
    /// `memory_swap_max`) can land on any cgroup the test author defines,
    /// and per-write lazy enablement would race against concurrent
    /// sibling cgroups reading their controller files. Per the cgroup v2
    /// docs ("Documentation/admin-guide/cgroup-v2.rst"), `+memory`
    /// enables memory controller files including `memory.max`,
    /// `memory.high`, `memory.swap.max`; `+pids` enables `pids.max`;
    /// `+io` enables `io.weight`. `cgroup.freeze` is a cgroup-core file
    /// not gated by any controller.
    pub fn setup(&self, enable_cpu_controller: bool) -> Result<()> {
        self.setup_under_root(enable_cpu_controller, &PathBuf::from("/sys/fs/cgroup"))
    }

    /// Inner setup that takes the cgroup-fs root as an explicit
    /// argument so tests can drive the controller-enable path against
    /// a tmpdir without touching `/sys/fs/cgroup`. Production
    /// [`Self::setup`] hardcodes `/sys/fs/cgroup`. The strip-prefix
    /// gate stays — if the parent is outside the supplied root,
    /// directory creation still happens but no subtree_control walk
    /// fires (matches the existing "non-cgroup-mount" early-bail).
    fn setup_under_root(&self, enable_cpu_controller: bool, root: &Path) -> Result<()> {
        if !self.parent.exists() {
            fs::create_dir_all(&self.parent)
                .with_context(|| format!("mkdir {}", self.parent.display()))?;
        }
        let controllers = if enable_cpu_controller {
            "+cpuset +cpu +memory +pids +io"
        } else {
            "+cpuset +memory +pids +io"
        };
        if let Ok(rel) = self.parent.strip_prefix(root) {
            let mut cur = root.to_path_buf();
            for c in rel.components() {
                let sc = cur.join("cgroup.subtree_control");
                if sc.exists()
                    && let Err(e) = write_with_timeout(&sc, controllers, CGROUP_WRITE_TIMEOUT)
                {
                    tracing::warn!(path = %sc.display(), err = %e, "failed to enable controllers");
                }
                cur = cur.join(c);
            }
            let sc = self.parent.join("cgroup.subtree_control");
            if sc.exists()
                && let Err(e) = write_with_timeout(&sc, controllers, CGROUP_WRITE_TIMEOUT)
            {
                tracing::warn!(path = %sc.display(), err = %e, "failed to enable controllers at parent");
            }
        }
        Ok(())
    }

    /// Create a child cgroup directory.
    ///
    /// For nested paths (e.g. `"cg_0/narrow"`), enables controllers on
    /// each intermediate cgroup's `subtree_control` so the leaf has
    /// controller files available. The kernel requires each parent to
    /// have the controller in `subtree_control` for its children to
    /// have the corresponding files (`cgroup_control()` returns
    /// `parent->subtree_control`).
    pub fn create_cgroup(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name);
        if !p.exists() {
            fs::create_dir_all(&p).with_context(|| format!("mkdir {}", p.display()))?;
        }
        self.enable_subtree_cpuset(name);
        Ok(())
    }

    /// Enable a controller on the parent cgroup's `cgroup.subtree_control`.
    ///
    /// Writes `+{controller}` to `{parent}/cgroup.subtree_control` so
    /// children created under the parent inherit the controller and
    /// expose the corresponding `*.cpus`, `*.mems`, etc. files. No-op
    /// (returns `Ok`) when the subtree_control file does not exist —
    /// callers treat that as "parent is not a cgroup v2 node" and
    /// degrade elsewhere.
    ///
    /// Unlike [`Self::setup`] and [`Self::enable_subtree_cpuset`],
    /// which swallow write failures via `tracing::warn!`, this method
    /// propagates the underlying [`std::io::Error`] so callers can
    /// classify errnos (EACCES/EPERM for permission, EBUSY for a
    /// peer holding the subtree) via [`anyhow_first_io_errno`] and
    /// map them to operator-facing degrade variants. Used by
    /// [`crate::vmm::cgroup_sandbox::BuildSandbox::try_create`] under
    /// the `--cpu-cap` hard-error contract.
    pub fn add_parent_subtree_controller(&self, controller: &str) -> Result<()> {
        let p = self.parent.join("cgroup.subtree_control");
        if !p.exists() {
            return Ok(());
        }
        write_with_timeout(&p, &format!("+{controller}"), CGROUP_WRITE_TIMEOUT)
    }

    /// Drain tasks from a child cgroup and remove it.
    pub fn remove_cgroup(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name);
        if !p.exists() {
            return Ok(());
        }
        self.drain_tasks(name)?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::remove_dir(&p).with_context(|| format!("rmdir {}", p.display()))
    }

    /// Write `cpuset.cpus` for a child cgroup.
    pub fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        let p = self.parent.join(name).join("cpuset.cpus");
        write_with_timeout(&p, &TestTopology::cpuset_string(cpus), CGROUP_WRITE_TIMEOUT)
    }

    /// Enable `+cpuset` on `cgroup.subtree_control` for each ancestor
    /// of the leaf in a nested cgroup path. For `"cg_0/narrow"`, writes
    /// `+cpuset` to `{parent}/cgroup.subtree_control` and
    /// `{parent}/cg_0/cgroup.subtree_control`. No-op for
    /// single-component paths.
    fn enable_subtree_cpuset(&self, name: &str) {
        let components: Vec<&str> = name.split('/').collect();
        if components.len() < 2 {
            return;
        }
        let mut cur = self.parent.clone();
        for c in &components[..components.len() - 1] {
            let sc = cur.join("cgroup.subtree_control");
            if sc.exists()
                && let Err(e) = write_with_timeout(&sc, "+cpuset", CGROUP_WRITE_TIMEOUT)
            {
                tracing::warn!(path = %sc.display(), err = %e, "failed to enable cpuset");
            }
            cur = cur.join(c);
        }
        // Write at the last intermediate (direct parent of the leaf).
        let sc = cur.join("cgroup.subtree_control");
        if sc.exists()
            && let Err(e) = write_with_timeout(&sc, "+cpuset", CGROUP_WRITE_TIMEOUT)
        {
            tracing::warn!(path = %sc.display(), err = %e, "failed to enable cpuset");
        }
    }

    /// Clear `cpuset.cpus` for a child cgroup (empty string = inherit parent).
    pub fn clear_cpuset(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name).join("cpuset.cpus");
        write_with_timeout(&p, "", CGROUP_WRITE_TIMEOUT)
    }

    /// Write `cpuset.mems` for a child cgroup. Constrains which NUMA
    /// nodes the cgroup's tasks can allocate memory on.
    ///
    /// Shape mirrors [`set_cpuset`] exactly — [`TestTopology::cpuset_string`]
    /// range-compact-formats the node set, [`write_with_timeout`] bounds
    /// the filesystem-write at 2s. Used by `BuildSandbox` under the
    /// `--cpu-cap` flow to bind build memory to the NUMA nodes hosting
    /// the locked LLCs, avoiding cross-socket DRAM latency for gcc's
    /// symbol tables and linker working sets.
    ///
    /// Must be called AFTER [`set_cpuset`] and BEFORE any
    /// [`move_task`]: a task in a cgroup whose `cpuset.mems` is empty
    /// either fails migration with EINVAL or (if it somehow gets in)
    /// hits SIGKILL on the next allocation per the kernel's
    /// `cpuset_update_task_spread` path.
    pub fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()> {
        let p = self.parent.join(name).join("cpuset.mems");
        write_with_timeout(
            &p,
            &TestTopology::cpuset_string(nodes),
            CGROUP_WRITE_TIMEOUT,
        )
    }

    /// Clear `cpuset.mems` for a child cgroup (empty string = inherit parent).
    /// Parallels [`clear_cpuset`]; callers use it only when tearing
    /// down a cpuset-restricted cgroup that needs to accept a
    /// fresh task binding with a different NUMA budget.
    pub fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name).join("cpuset.mems");
        write_with_timeout(&p, "", CGROUP_WRITE_TIMEOUT)
    }

    /// Write `cpu.max` for a child cgroup. `quota_us = None` writes
    /// `"max <period_us>"` (no upper bound — same as a freshly
    /// created cgroup); `Some(q)` writes `"<q> <period_us>"`.
    ///
    /// Per the kernel's cgroup v2 docs ("Documentation/admin-guide/
    /// cgroup-v2.rst", "CPU Interface Files"): each period the
    /// cgroup gets `quota` microseconds of CPU time across its
    /// CPUs, and is throttled until the next period boundary once
    /// the quota is exhausted. `quota` MAY exceed `period` to let
    /// the cgroup use multiple CPUs concurrently (e.g. quota
    /// 200_000 / period 100_000 = up to 2 CPUs of throughput).
    ///
    /// Requires `+cpu` in the parent's `cgroup.subtree_control`;
    /// missing controller surfaces as ENOENT on the file (handled
    /// generically by [`write_with_timeout`]'s error path with the
    /// errno suffix).
    pub fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()> {
        let p = self.parent.join(name).join("cpu.max");
        let line = match quota_us {
            Some(q) => format!("{q} {period_us}"),
            None => format!("max {period_us}"),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `cpu.weight` for a child cgroup (cgroup v2 weight,
    /// range 1..=10000, default 100). Used together with sibling
    /// cgroups to bias relative CPU share inside the parent's
    /// quota. Independent from `cpu.max` — weights govern share
    /// when CPU is contended, max enforces an absolute ceiling.
    ///
    /// Per "Documentation/admin-guide/cgroup-v2.rst" the legacy
    /// "shares" knob is `cpu.weight.nice` (mapped from nice value);
    /// this method targets the canonical `cpu.weight` knob.
    pub fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()> {
        let p = self.parent.join(name).join("cpu.weight");
        write_with_timeout(&p, &weight.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Write `memory.max` for a child cgroup. `bytes = None` writes
    /// `"max"` (no hard limit). When the cgroup's RSS exceeds the
    /// limit, the kernel OOM-kills tasks per the documented
    /// `memory.max` semantics. Requires `+memory` in the parent's
    /// `cgroup.subtree_control`.
    pub fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        let p = self.parent.join(name).join("memory.max");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `memory.high` for a child cgroup. `bytes = None`
    /// writes `"max"` (no high-water mark). Crossing the high
    /// threshold triggers reclaim throttling but NOT OOM-kill,
    /// distinguishing it from `memory.max`.
    pub fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        let p = self.parent.join(name).join("memory.high");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `memory.low` for a child cgroup. `bytes = None` writes
    /// `"0"` (no low-water protection). The kernel preferentially
    /// reclaims FROM other cgroups before reclaiming this cgroup's
    /// memory below `memory.low`; not a hard reservation.
    pub fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        let p = self.parent.join(name).join("memory.low");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "0".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `io.weight` for a child cgroup (cgroup v2 weight,
    /// range 1..=10000, default 100). Biases relative IO share
    /// across sibling cgroups when the io controller is enabled
    /// in the parent's `cgroup.subtree_control`. The kernel's BFQ
    /// or io.cost backend (whichever is active) applies the
    /// weight when contending devices are saturated.
    ///
    /// `io.max` (per-device throughput cap) is intentionally NOT
    /// surfaced here — the per-device interface needs major:minor
    /// device-id lookup which has no in-tree consumer; surface it
    /// as a follow-up task when a concrete use case lands.
    pub fn set_io_weight(&self, name: &str, weight: u16) -> Result<()> {
        let p = self.parent.join(name).join("io.weight");
        write_with_timeout(&p, &weight.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Write `cgroup.freeze` for a child cgroup. `frozen = true` writes
    /// `"1"`, `frozen = false` writes `"0"`.
    ///
    /// `cgroup.freeze` is a cgroup-core file exposed on every non-root
    /// cgroup automatically — it is NOT gated by `cgroup.subtree_control`.
    /// The kernel's `cgroup_freeze_write`
    /// (kernel/cgroup/cgroup.c:4099-4122) parses the value via
    /// `kstrtoint`, rejects anything outside `{0, 1}` with `-ERANGE`,
    /// and dispatches `cgroup_freeze(cgrp, freeze)`. Writing `1` to a
    /// cgroup containing tasks transitions every task in the subtree to
    /// the frozen state; writing `0` releases. The transition is
    /// asynchronous — `cgroup.events`'s `frozen` field reaches `1` once
    /// every task has parked.
    pub fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
        let p = self.parent.join(name).join("cgroup.freeze");
        let line = if frozen { "1" } else { "0" };
        write_with_timeout(&p, line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `pids.max` for a child cgroup. `max = None` writes `"max"`
    /// (the kernel's `PIDS_MAX_STR` sentinel for unlimited);
    /// `Some(n)` writes the decimal `n`.
    ///
    /// Per the kernel's `pids_max_write`
    /// (kernel/cgroup/pids.c:301-329): the parser short-circuits to the
    /// unlimited limit when `buf == PIDS_MAX_STR`; otherwise
    /// `kstrtoll(buf, 0, &limit)` parses a signed integer and rejects
    /// `< 0` or `>= PIDS_MAX` with `-EINVAL`. The update is atomic
    /// (`atomic64_set(&pids->limit, limit)`); existing tasks are NOT
    /// killed when the limit lands below the current task count — only
    /// future `fork()` / `clone()` calls are blocked.
    ///
    /// Requires `+pids` in the parent's `cgroup.subtree_control`;
    /// [`Self::setup`] enables it unconditionally so this write
    /// succeeds on every ktstr-managed cgroup tree.
    pub fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()> {
        let p = self.parent.join(name).join("pids.max");
        let line = match max {
            Some(n) => n.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `memory.swap.max` for a child cgroup. `bytes = None` writes
    /// `"max"` (no swap cap); `Some(b)` writes the decimal byte count.
    ///
    /// Per the kernel's `swap_max_write` (mm/memcontrol.c:5379-5394):
    /// the value is parsed via `page_counter_memparse(buf, "max", &max)`,
    /// which accepts the literal `"max"` token for unlimited or a
    /// numeric byte count. The store is `xchg(&memcg->swap.max, max)` —
    /// atomic, with no failure path beyond the parse.
    ///
    /// Requires `+memory` in the parent's `cgroup.subtree_control`;
    /// [`Self::setup`] enables it unconditionally.
    pub fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        let p = self.parent.join(name).join("memory.swap.max");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Move a single task into a child cgroup via `cgroup.procs`.
    pub fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        let p = self.parent.join(name).join("cgroup.procs");
        write_with_timeout(&p, &pid.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Move multiple tasks into a child cgroup by PID.
    ///
    /// Tolerates ESRCH (task exited between listing and migration).
    /// Retries EBUSY up to 3 times with 100ms backoff for transient
    /// rejections from sched_ext BPF `cgroup_prep_move` callbacks
    /// (`scx_cgroup_can_attach`). Propagates EBUSY after retries
    /// exhausted. Propagates all other errors immediately.
    pub fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()> {
        for &pid in pids {
            if let Err(e) = self.move_task_with_retry(name, pid) {
                if is_esrch(&e) {
                    tracing::warn!(pid, cgroup = name, "task vanished during migration");
                    continue;
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Move a single task with bounded EBUSY retry.
    fn move_task_with_retry(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY: Duration = Duration::from_millis(100);

        for attempt in 0..MAX_RETRIES {
            match self.move_task(name, pid) {
                Ok(()) => return Ok(()),
                Err(e) if is_ebusy(&e) && attempt + 1 < MAX_RETRIES => {
                    tracing::debug!(
                        pid,
                        cgroup = name,
                        attempt = attempt + 1,
                        "EBUSY on cgroup.procs write, retrying"
                    );
                    std::thread::sleep(RETRY_DELAY);
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!()
    }

    /// Clear `subtree_control` on a child cgroup by writing an empty
    /// string. Disables all controllers for the cgroup's children.
    ///
    /// Required before moving tasks into a cgroup that has
    /// `subtree_control` set: the kernel's no-internal-process
    /// constraint (`cgroup_migrate_vet_dst`) returns EBUSY when
    /// tasks are written to `cgroup.procs` of a cgroup with
    /// controllers in `subtree_control`.
    pub fn clear_subtree_control(&self, name: &str) -> Result<()> {
        let p = self.parent.join(name).join("cgroup.subtree_control");
        if !p.exists() {
            return Ok(());
        }
        // Read current controllers and disable each one.
        let content = fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let content = content.trim();
        if content.is_empty() {
            return Ok(());
        }
        // Each controller name needs a "-" prefix to disable.
        let disable: Vec<String> = content
            .split_whitespace()
            .map(|c| format!("-{c}"))
            .collect();
        let disable_str = disable.join(" ");
        write_with_timeout(&p, &disable_str, CGROUP_WRITE_TIMEOUT)
            .with_context(|| format!("clear subtree_control on {name}"))
    }

    /// Move all tasks from a child cgroup to the cgroup root.
    ///
    /// Drains to `/sys/fs/cgroup/cgroup.procs` instead of the parent
    /// because the parent has `subtree_control` set (enabling cpuset
    /// for children), and the kernel's no-internal-process constraint
    /// rejects writes to `cgroup.procs` when `subtree_control` is
    /// active. The root cgroup is exempt from this constraint.
    pub fn drain_tasks(&self, name: &str) -> Result<()> {
        let src = self.parent.join(name).join("cgroup.procs");
        if !src.exists() {
            return Ok(());
        }
        drain_pids_to_root(&src, name);
        Ok(())
    }

    /// Remove all child cgroups under the parent (keeps the parent itself).
    ///
    /// Returns `Ok` even when individual filesystem probes fail; callers
    /// treat cleanup as best-effort teardown (see the runner's warn-
    /// and-continue in `src/runner.rs`). Per-entry `read_dir` /
    /// `DirEntry` / `file_type` errors are surfaced via
    /// `tracing::warn!` — mirrors `CgroupGroup::drop` so a failure
    /// shows up in logs instead of silently leaving children behind.
    pub fn cleanup_all(&self) -> Result<()> {
        if !self.parent.exists() {
            return Ok(());
        }
        if let Err(err) = for_each_child_dir(&self.parent, "cleanup_all", cleanup_recursive) {
            tracing::warn!(
                parent = %self.parent.display(),
                err = %err,
                "cleanup_all: read_dir failed; child cgroups may remain under parent",
            );
        }
        Ok(())
    }
}

/// Abstraction over the cgroup v2 filesystem surface used by the
/// scenario runtime. The production implementation is [`CgroupManager`],
/// which translates each method into real writes under `/sys/fs/cgroup`.
///
/// Extracted so `scenario::ops::apply_setup` and related orchestration
/// code can be unit-tested against an in-memory double: tests construct
/// a recording or failure-injecting implementor, drive `apply_setup`
/// against it, and assert on the recorded call sequence without
/// touching the host cgroup hierarchy.
///
/// Object-safe by design — scenario code holds the trait object behind
/// `&dyn CgroupOps` rather than being generic. Callers keep writing
/// `ctx.cgroups.set_cpuset(...)` with no syntactic change; dynamic
/// dispatch resolves to `CgroupManager` in production and to the
/// test double under `#[cfg(test)]`. The per-call indirect-call cost
/// is dominated by the filesystem I/O the trait abstracts over.
pub trait CgroupOps {
    /// Path to the parent cgroup directory. See
    /// [`CgroupManager::parent_path`].
    fn parent_path(&self) -> &Path;
    /// Create the parent directory and enable controllers. See
    /// [`CgroupManager::setup`].
    fn setup(&self, enable_cpu_controller: bool) -> Result<()>;
    /// Create a child cgroup. See [`CgroupManager::create_cgroup`].
    fn create_cgroup(&self, name: &str) -> Result<()>;
    /// Drain and remove a child cgroup. See
    /// [`CgroupManager::remove_cgroup`].
    fn remove_cgroup(&self, name: &str) -> Result<()>;
    /// Write `cpuset.cpus`. See [`CgroupManager::set_cpuset`].
    fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()>;
    /// Clear `cpuset.cpus` (inherit from parent). See
    /// [`CgroupManager::clear_cpuset`].
    fn clear_cpuset(&self, name: &str) -> Result<()>;
    /// Write `cpuset.mems`. See [`CgroupManager::set_cpuset_mems`].
    fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()>;
    /// Clear `cpuset.mems` (inherit from parent). See
    /// [`CgroupManager::clear_cpuset_mems`].
    fn clear_cpuset_mems(&self, name: &str) -> Result<()>;
    /// Write `cpu.max`. See [`CgroupManager::set_cpu_max`].
    fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()>;
    /// Write `cpu.weight`. See [`CgroupManager::set_cpu_weight`].
    fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()>;
    /// Write `memory.max`. See [`CgroupManager::set_memory_max`].
    fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Write `memory.high`. See [`CgroupManager::set_memory_high`].
    fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Write `memory.low`. See [`CgroupManager::set_memory_low`].
    fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Write `io.weight`. See [`CgroupManager::set_io_weight`].
    fn set_io_weight(&self, name: &str, weight: u16) -> Result<()>;
    /// Write `cgroup.freeze`. See [`CgroupManager::set_freeze`].
    fn set_freeze(&self, name: &str, frozen: bool) -> Result<()>;
    /// Write `pids.max`. See [`CgroupManager::set_pids_max`].
    fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()>;
    /// Write `memory.swap.max`. See
    /// [`CgroupManager::set_memory_swap_max`].
    fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()>;
    /// Move a single task via `cgroup.procs`. See
    /// [`CgroupManager::move_task`].
    fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()>;
    /// Move multiple tasks (tolerates ESRCH, retries EBUSY). See
    /// [`CgroupManager::move_tasks`].
    fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()>;
    /// Clear `cgroup.subtree_control` on a child. See
    /// [`CgroupManager::clear_subtree_control`].
    fn clear_subtree_control(&self, name: &str) -> Result<()>;
    /// Drain tasks from a child to the cgroup root. See
    /// [`CgroupManager::drain_tasks`].
    fn drain_tasks(&self, name: &str) -> Result<()>;
    /// Remove all child cgroups under the parent. See
    /// [`CgroupManager::cleanup_all`].
    fn cleanup_all(&self) -> Result<()>;
}

// Thin forwarding trait impl: inherent `CgroupManager` methods hold the
// real bodies; this trait impl exists so scenario code can hold
// `&dyn CgroupOps` for test-double injection without threading a generic
// through every caller. Trait default methods cannot access the private
// fields, and macro-generated delegation would lose Go-To-Definition.
impl CgroupOps for CgroupManager {
    fn parent_path(&self) -> &Path {
        CgroupManager::parent_path(self)
    }
    fn setup(&self, enable_cpu_controller: bool) -> Result<()> {
        CgroupManager::setup(self, enable_cpu_controller)
    }
    fn create_cgroup(&self, name: &str) -> Result<()> {
        CgroupManager::create_cgroup(self, name)
    }
    fn remove_cgroup(&self, name: &str) -> Result<()> {
        CgroupManager::remove_cgroup(self, name)
    }
    fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        CgroupManager::set_cpuset(self, name, cpus)
    }
    fn clear_cpuset(&self, name: &str) -> Result<()> {
        CgroupManager::clear_cpuset(self, name)
    }
    fn set_cpuset_mems(&self, name: &str, nodes: &BTreeSet<usize>) -> Result<()> {
        CgroupManager::set_cpuset_mems(self, name, nodes)
    }
    fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
        CgroupManager::clear_cpuset_mems(self, name)
    }
    fn set_cpu_max(&self, name: &str, quota_us: Option<u64>, period_us: u64) -> Result<()> {
        CgroupManager::set_cpu_max(self, name, quota_us, period_us)
    }
    fn set_cpu_weight(&self, name: &str, weight: u32) -> Result<()> {
        CgroupManager::set_cpu_weight(self, name, weight)
    }
    fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_max(self, name, bytes)
    }
    fn set_memory_high(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_high(self, name, bytes)
    }
    fn set_memory_low(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_low(self, name, bytes)
    }
    fn set_io_weight(&self, name: &str, weight: u16) -> Result<()> {
        CgroupManager::set_io_weight(self, name, weight)
    }
    fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
        CgroupManager::set_freeze(self, name, frozen)
    }
    fn set_pids_max(&self, name: &str, max: Option<u64>) -> Result<()> {
        CgroupManager::set_pids_max(self, name, max)
    }
    fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        CgroupManager::set_memory_swap_max(self, name, bytes)
    }
    fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        CgroupManager::move_task(self, name, pid)
    }
    fn move_tasks(&self, name: &str, pids: &[libc::pid_t]) -> Result<()> {
        CgroupManager::move_tasks(self, name, pids)
    }
    fn clear_subtree_control(&self, name: &str) -> Result<()> {
        CgroupManager::clear_subtree_control(self, name)
    }
    fn drain_tasks(&self, name: &str) -> Result<()> {
        CgroupManager::drain_tasks(self, name)
    }
    fn cleanup_all(&self) -> Result<()> {
        CgroupManager::cleanup_all(self)
    }
}

/// Drain all tasks from `procs_path` to the cgroup filesystem root.
///
/// The root cgroup is exempt from the no-internal-process constraint,
/// so writes to `/sys/fs/cgroup/cgroup.procs` succeed even when
/// intermediate cgroups have `subtree_control` set.
/// ESRCH (task exited) is silently tolerated; other errors are logged.
/// A `read_to_string` failure or a malformed pid line is surfaced via
/// `tracing::warn!` — silently dropping either would hide a cgroup
/// that still contains tasks and send it into cleanup, which then
/// fails with EBUSY and compounds the confusion.
fn drain_pids_to_root(procs_path: &Path, context: &str) {
    let dst = Path::new("/sys/fs/cgroup/cgroup.procs");
    let content = match fs::read_to_string(procs_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(
                path = %procs_path.display(),
                cgroup = context,
                err = %e,
                "drain_pids_to_root: read_to_string failed; tasks may remain in cgroup",
            );
            return;
        }
    };
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let pid: u32 = match trimmed.parse() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    path = %procs_path.display(),
                    cgroup = context,
                    line = trimmed,
                    err = %e,
                    "drain_pids_to_root: malformed pid line; skipping",
                );
                continue;
            }
        };
        if let Err(e) = write_with_timeout(dst, &pid.to_string(), CGROUP_WRITE_TIMEOUT)
            && !is_esrch(&e)
        {
            tracing::warn!(pid, cgroup = context, err = %e, "failed to drain task");
        }
    }
}

/// Iterate the direct child directories of `path`, calling `f` on each.
///
/// `context` is a short caller name (e.g. `"cleanup_all"`,
/// `"cleanup_recursive"`) that is prefixed into every per-entry
/// `tracing::warn!` message so operators grepping logs for
/// `"cleanup_all: "` still see both the outer read_dir failure (which
/// stays with the caller) and the per-entry `DirEntry` / `file_type`
/// warnings emitted here.
///
/// `read_dir` failure is surfaced to the caller via `Err`; the caller
/// owns the top-level warn message. Non-directory entries are skipped.
/// Per-entry errors are logged and the iteration continues.
///
/// The structured log field key is normalized to `path =` at this
/// boundary; `cleanup_all`'s outer warn still uses `parent =` for the
/// top-level read_dir failure since that warn is emitted by the
/// caller, not here.
fn for_each_child_dir(path: &Path, context: &str, mut f: impl FnMut(&Path)) -> std::io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    err = %err,
                    "{context}: dir entry read failed; skipping",
                );
                continue;
            }
        };
        match entry.file_type() {
            Ok(t) if t.is_dir() => f(&entry.path()),
            Ok(_) => {}
            Err(err) => tracing::warn!(
                path = %entry.path().display(),
                err = %err,
                "{context}: file_type read failed; skipping entry",
            ),
        }
    }
    Ok(())
}

fn cleanup_recursive(path: &std::path::Path) {
    // Depth-first: clean children before parent
    if let Err(err) = for_each_child_dir(path, "cleanup_recursive", cleanup_recursive) {
        tracing::warn!(
            path = %path.display(),
            err = %err,
            "cleanup_recursive: read_dir failed; child cgroups may remain",
        );
    }
    drain_pids_to_root(&path.join("cgroup.procs"), &path.display().to_string());
    std::thread::sleep(std::time::Duration::from_millis(10));
    if let Err(err) = fs::remove_dir(path) {
        tracing::warn!(
            path = %path.display(),
            err = %err,
            "cleanup_recursive: remove_dir failed; cgroup directory may remain",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cgroup_manager_path() {
        let cg = CgroupManager::new("/sys/fs/cgroup/test");
        assert_eq!(
            cg.parent_path(),
            std::path::Path::new("/sys/fs/cgroup/test")
        );
    }

    #[test]
    fn create_cgroup_in_tmpdir() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("test_cg").unwrap();
        assert!(dir.join("test_cg").exists());
        cg.create_cgroup("nested/deep").unwrap();
        assert!(dir.join("nested/deep").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_cgroup_idempotent() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-idem-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("cg_0").unwrap();
        cg.create_cgroup("cg_0").unwrap(); // should not error
        assert!(dir.join("cg_0").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_all_on_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-test-path");
        assert!(cg.cleanup_all().is_ok());
    }

    #[test]
    fn remove_cgroup_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-test-path");
        assert!(cg.remove_cgroup("no_such_cgroup").is_ok());
    }

    #[test]
    fn cleanup_removes_child_dirs() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-clean-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("a").unwrap();
        cg.create_cgroup("b").unwrap();
        cg.create_cgroup("c/deep").unwrap();
        assert!(dir.join("a").exists());
        assert!(dir.join("c/deep").exists());
        // cleanup_all removes child dirs (not real cgroups, so drain_tasks is a no-op)
        cg.cleanup_all().unwrap();
        assert!(!dir.join("a").exists());
        assert!(!dir.join("b").exists());
        assert!(!dir.join("c").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn drain_tasks_nonexistent_source() {
        let cg = CgroupManager::new("/nonexistent/ktstr-drain-test");
        assert!(cg.drain_tasks("missing_cgroup").is_ok());
    }

    /// `cleanup_all` must skip non-directory entries rather than
    /// recurse into them. Plants a regular file alongside a child
    /// cgroup directory and verifies: (a) the child dir is removed,
    /// (b) the file is left in place. Pins the `Ok(t) if t.is_dir()`
    /// branch in [`for_each_child_dir`] so a future refactor that
    /// drops the `is_dir` guard fails this test instead of silently
    /// deleting arbitrary files under the cgroup parent.
    #[test]
    fn cleanup_all_skips_non_dir_entries() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-nondir-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.create_cgroup("cg_child").unwrap();
        let stray_file = dir.join("stray.txt");
        fs::write(&stray_file, b"do not descend").unwrap();
        assert!(dir.join("cg_child").exists());
        assert!(stray_file.exists());
        cg.cleanup_all().unwrap();
        assert!(
            !dir.join("cg_child").exists(),
            "cleanup_all should remove the child directory",
        );
        assert!(
            stray_file.exists(),
            "cleanup_all must not descend into or remove regular files",
        );
        assert_eq!(fs::read_to_string(&stray_file).unwrap(), "do not descend");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `cleanup_recursive` on a 2-level nested directory structure must
    /// remove leaves before their parents (depth-first). Plants
    /// `root/mid/leaf/` plus `root/sibling/`, invokes
    /// [`cleanup_recursive`] directly on `root`, and verifies every
    /// directory is gone. Exercises the recursive call inside
    /// [`for_each_child_dir`] that item 7's `cleanup_recursive`
    /// function-pointer arg drives.
    #[test]
    fn cleanup_recursive_removes_nested_dirs_depth_first() {
        let base = std::env::temp_dir().join(format!("ktstr-cg-nested-{}", std::process::id()));
        let root = base.join("root");
        fs::create_dir_all(root.join("mid").join("leaf")).unwrap();
        fs::create_dir_all(root.join("sibling")).unwrap();
        assert!(root.join("mid/leaf").exists());
        assert!(root.join("sibling").exists());
        cleanup_recursive(&root);
        assert!(
            !root.exists(),
            "cleanup_recursive should remove root and every descendant",
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn setup_non_cgroup_path() {
        // setup() on a non-cgroup path should still create the dir
        let dir = std::env::temp_dir().join(format!("ktstr-setup-{}", std::process::id()));
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.setup(true).unwrap();
        assert!(dir.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// `setup` writes every expected controller (`+cpuset +cpu
    /// +memory +pids +io`) to the parent cgroup's
    /// `cgroup.subtree_control`. Without this assertion, a future
    /// edit that drops `+pids` (or any other controller) from the
    /// expansion would silently land — apply_setup's pids/swap_max
    /// writes would then fail with ENOENT at scenario-execution
    /// time, NOT at framework-setup time.
    ///
    /// Path: drives [`setup_under_root`] against a tmpdir-rooted
    /// "cgroup tree" (parent dir + pre-created
    /// `cgroup.subtree_control` file at the leaf), then reads the
    /// file back and asserts every controller token is present.
    #[test]
    fn setup_writes_all_controllers() {
        let root = std::env::temp_dir().join(format!("ktstr-setup-controllers-{}", std::process::id()));
        let parent = root.join("ktstr");
        fs::create_dir_all(&parent).unwrap();
        // Pre-create the subtree_control file so the strip-prefix
        // walk in setup_under_root's leaf-write branch sees it
        // exist (the production setup() also depends on the file
        // existing — cgroup v2 creates it at mount time).
        fs::write(parent.join("cgroup.subtree_control"), "").unwrap();

        let cg = CgroupManager::new(parent.to_str().unwrap());
        cg.setup_under_root(true, &root).unwrap();

        let written = fs::read_to_string(parent.join("cgroup.subtree_control")).unwrap();
        for token in [
            "+cpuset",
            "+cpu",
            "+memory",
            "+pids",
            "+io",
        ] {
            assert!(
                written.contains(token),
                "subtree_control must contain {token}; got: {written:?}",
            );
        }

        // enable_cpu_controller=false drops only +cpu; the other
        // four must still appear.
        fs::write(parent.join("cgroup.subtree_control"), "").unwrap();
        cg.setup_under_root(false, &root).unwrap();
        let written = fs::read_to_string(parent.join("cgroup.subtree_control")).unwrap();
        assert!(written.contains("+cpuset"));
        assert!(written.contains("+memory"));
        assert!(written.contains("+pids"));
        assert!(written.contains("+io"));
        assert!(
            !written.contains("+cpu "),
            "+cpu must be absent when enable_cpu_controller=false; got: {written:?}",
        );
        // Distinguish "+cpu " (token bound) from "+cpuset" / "+cpu"
        // tail-without-space: the only +cpu* tokens we expect when
        // enable_cpu_controller=false are +cpuset. Verify the raw
        // string ends without a bare +cpu by asserting the position
        // of "+cpu" only matches the +cpuset prefix.
        let cpu_positions: Vec<usize> = written.match_indices("+cpu").map(|(i, _)| i).collect();
        for pos in cpu_positions {
            let suffix = &written[pos..];
            assert!(
                suffix.starts_with("+cpuset"),
                "every +cpu* token must be +cpuset when enable_cpu_controller=false; \
                 got '{suffix}' at pos {pos} in {written:?}",
            );
        }

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn write_with_timeout_success() {
        let dir = std::env::temp_dir().join(format!("ktstr-wt-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("test_write");
        write_with_timeout(&f, "hello", Duration::from_secs(5)).unwrap();
        assert_eq!(fs::read_to_string(&f).unwrap(), "hello");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_bad_path() {
        let f = Path::new("/nonexistent/dir/file");
        assert!(write_with_timeout(f, "data", Duration::from_secs(5)).is_err());
    }

    #[test]
    fn move_task_nonexistent_cgroup() {
        let cg = CgroupManager::new("/nonexistent/ktstr-move-test");
        assert!(cg.move_task("no_cgroup", 1).is_err());
    }

    #[test]
    fn set_cpuset_empty() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-cpuset-{}", std::process::id()));
        let dir_a = dir.join("cg_a");
        fs::create_dir_all(&dir_a).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Empty BTreeSet → writes empty string via cpuset_string
        cg.set_cpuset("cg_a", &BTreeSet::new()).unwrap();
        assert_eq!(fs::read_to_string(dir_a.join("cpuset.cpus")).unwrap(), "");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn move_tasks_partial_failure() {
        // move_tasks propagates non-ESRCH errors immediately
        let cg = CgroupManager::new("/nonexistent/ktstr-partial");
        let err = cg.move_tasks("cg", &[1, 2, 3]).unwrap_err();
        // The error comes from the first pid (write to nonexistent path)
        let msg = format!("{err:#}");
        assert!(msg.contains("cgroup.procs"), "unexpected error: {msg}");
    }

    #[test]
    fn drain_tasks_empty_cgroup() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-drain-{}", std::process::id()));
        let dir_d = dir.join("cg_d");
        fs::create_dir_all(&dir_d).unwrap();
        // Create an empty cgroup.procs file
        fs::write(dir_d.join("cgroup.procs"), "").unwrap();
        // Parent also needs cgroup.procs
        fs::write(dir.join("cgroup.procs"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // drain_tasks on a cgroup with empty procs file should succeed
        assert!(cg.drain_tasks("cg_d").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn is_esrch_detects_esrch_in_chain() {
        let io_err = std::io::Error::from_raw_os_error(libc::ESRCH);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(is_esrch(&anyhow_err));
    }

    #[test]
    fn is_esrch_rejects_enoent() {
        let io_err = std::io::Error::from_raw_os_error(libc::ENOENT);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(!is_esrch(&anyhow_err));
    }

    #[test]
    fn is_ebusy_detects_ebusy_in_chain() {
        let io_err = std::io::Error::from_raw_os_error(libc::EBUSY);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(is_ebusy(&anyhow_err));
    }

    #[test]
    fn is_ebusy_rejects_esrch() {
        let io_err = std::io::Error::from_raw_os_error(libc::ESRCH);
        let anyhow_err = anyhow::Error::new(io_err).context("write cgroup.procs");
        assert!(!is_ebusy(&anyhow_err));
    }

    #[test]
    fn clear_subtree_control_nonexistent() {
        let cg = CgroupManager::new("/nonexistent/ktstr-clear-sc");
        // No subtree_control file → no-op success.
        assert!(cg.clear_subtree_control("cg_0").is_ok());
    }

    #[test]
    fn clear_subtree_control_empty() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-sc-{}", std::process::id()));
        let dir_a = dir.join("cg_a");
        fs::create_dir_all(&dir_a).unwrap();
        fs::write(dir_a.join("cgroup.subtree_control"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // Empty subtree_control → no-op success.
        assert!(cg.clear_subtree_control("cg_a").is_ok());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_with_timeout_blocks_on_fifo() {
        use std::ffi::CString;
        let dir = std::env::temp_dir().join(format!("ktstr-cg-fifo-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let fifo_path = dir.join("blocked_write");
        let c_path = CString::new(fifo_path.to_str().unwrap()).unwrap();
        let rc = unsafe { libc::mkfifo(c_path.as_ptr(), 0o700) };
        assert_eq!(rc, 0, "mkfifo failed: {}", std::io::Error::last_os_error());
        // Very short timeout — write blocks until a reader opens the FIFO
        let err = write_with_timeout(&fifo_path, "data", Duration::from_millis(50)).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("timed out"), "unexpected error: {msg}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn anyhow_first_io_errno_extracts_raw_errno() {
        let io = std::io::Error::from_raw_os_error(libc::EBUSY);
        let err = anyhow::Error::new(io);
        assert_eq!(anyhow_first_io_errno(&err), Some(libc::EBUSY));
    }

    #[test]
    fn anyhow_first_io_errno_through_context() {
        let io = std::io::Error::from_raw_os_error(libc::ESRCH);
        let err = anyhow::Error::new(io).context("wrapping context");
        assert_eq!(anyhow_first_io_errno(&err), Some(libc::ESRCH));
    }

    #[test]
    fn anyhow_first_io_errno_no_io_returns_none() {
        let err = anyhow::anyhow!("plain text error");
        assert_eq!(anyhow_first_io_errno(&err), None);
    }

    #[test]
    fn add_parent_subtree_controller_missing_file_noop() {
        let cg = CgroupManager::new("/nonexistent/ktstr-add-parent-sc");
        assert!(cg.add_parent_subtree_controller("cpuset").is_ok());
    }

    #[test]
    fn add_parent_subtree_controller_writes_plus_prefixed_token() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-addparent-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        // The subtree_control file in a real cgroup v2 tree echoes the
        // currently-enabled controllers (no `+` prefix) when read back;
        // here we just observe that our write landed verbatim.
        let sc = dir.join("cgroup.subtree_control");
        fs::write(&sc, "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.add_parent_subtree_controller("cpuset").unwrap();
        assert_eq!(fs::read_to_string(&sc).unwrap(), "+cpuset");
        let _ = fs::remove_dir_all(&dir);
    }

    // -- Cgroup v2 resource control writes ----------------------------
    //
    // Each new CgroupOps method writes a single cgroupfs file. The
    // tests below stand up a tmpdir representing the parent cgroup,
    // pre-create the child + the target file (real cgroupfs creates
    // these on directory creation; tmpfs needs them touched), invoke
    // the method, and assert on the resulting file contents.

    fn make_test_cgroup(label: &str) -> (PathBuf, CgroupManager) {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-{label}-{}", std::process::id()));
        fs::create_dir_all(dir.join("cg_x")).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        (dir, cg)
    }

    #[test]
    fn set_cpu_max_writes_quota_and_period_when_some() {
        let (dir, cg) = make_test_cgroup("cpu-max-some");
        let target = dir.join("cg_x").join("cpu.max");
        fs::write(&target, "").unwrap();
        cg.set_cpu_max("cg_x", Some(50_000), 100_000).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "50000 100000");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_cpu_max_writes_max_keyword_when_none() {
        let (dir, cg) = make_test_cgroup("cpu-max-none");
        let target = dir.join("cg_x").join("cpu.max");
        fs::write(&target, "").unwrap();
        cg.set_cpu_max("cg_x", None, 100_000).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max 100000");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_cpu_weight_writes_decimal_value() {
        let (dir, cg) = make_test_cgroup("cpu-weight");
        let target = dir.join("cg_x").join("cpu.weight");
        fs::write(&target, "").unwrap();
        cg.set_cpu_weight("cg_x", 250).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "250");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_memory_max_writes_bytes_or_max_keyword() {
        let (dir, cg) = make_test_cgroup("mem-max");
        let target = dir.join("cg_x").join("memory.max");
        fs::write(&target, "").unwrap();
        cg.set_memory_max("cg_x", Some(1_048_576)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1048576");
        cg.set_memory_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_memory_high_writes_bytes_or_max_keyword() {
        let (dir, cg) = make_test_cgroup("mem-high");
        let target = dir.join("cg_x").join("memory.high");
        fs::write(&target, "").unwrap();
        cg.set_memory_high("cg_x", Some(524_288)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "524288");
        cg.set_memory_high("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `memory.low`'s "no protection" wire value is `"0"`, NOT
    /// `"max"` — the kernel treats `max` as a syntax error on
    /// `memory.low`. Pin both the bytes-set and the cleared paths.
    #[test]
    fn set_memory_low_writes_bytes_or_zero() {
        let (dir, cg) = make_test_cgroup("mem-low");
        let target = dir.join("cg_x").join("memory.low");
        fs::write(&target, "").unwrap();
        cg.set_memory_low("cg_x", Some(2_048)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "2048");
        cg.set_memory_low("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn set_io_weight_writes_decimal_value() {
        let (dir, cg) = make_test_cgroup("io-weight");
        let target = dir.join("cg_x").join("io.weight");
        fs::write(&target, "").unwrap();
        cg.set_io_weight("cg_x", 500).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "500");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `set_freeze(true)` writes the literal `"1"`; `false` writes
    /// `"0"`. Pinned because the kernel's `cgroup_freeze_write` rejects
    /// any other value with `-ERANGE` — a regression that emits "true"
    /// or "frozen" would surface as a syscall failure on real cgroupfs.
    #[test]
    fn set_freeze_writes_zero_or_one() {
        let (dir, cg) = make_test_cgroup("freeze");
        let target = dir.join("cg_x").join("cgroup.freeze");
        fs::write(&target, "").unwrap();
        cg.set_freeze("cg_x", true).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1");
        cg.set_freeze("cg_x", false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `set_pids_max(Some(n))` writes the decimal `n`;
    /// `set_pids_max(None)` writes `"max"` — the kernel's
    /// `PIDS_MAX_STR` sentinel that selects the unlimited path in
    /// `pids_max_write`.
    #[test]
    fn set_pids_max_writes_decimal_or_max_keyword() {
        let (dir, cg) = make_test_cgroup("pids-max");
        let target = dir.join("cg_x").join("pids.max");
        fs::write(&target, "").unwrap();
        cg.set_pids_max("cg_x", Some(1024)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1024");
        cg.set_pids_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `set_memory_swap_max(Some(b))` writes the decimal byte count;
    /// `None` writes `"max"` — the unlimited sentinel
    /// `page_counter_memparse` recognises in `swap_max_write`.
    #[test]
    fn set_memory_swap_max_writes_bytes_or_max_keyword() {
        let (dir, cg) = make_test_cgroup("mem-swap-max");
        let target = dir.join("cg_x").join("memory.swap.max");
        fs::write(&target, "").unwrap();
        cg.set_memory_swap_max("cg_x", Some(2 * 1024 * 1024)).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "2097152");
        cg.set_memory_swap_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
        let _ = fs::remove_dir_all(&dir);
    }
}
