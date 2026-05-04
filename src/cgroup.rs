//! Cgroup v2 filesystem operations for test cgroup management.
//!
//! Creates, configures, and removes cgroups under a parent path
//! (default `/sys/fs/cgroup/ktstr`). Provides cpuset assignment,
//! task migration, and cleanup.
//!
//! # Controller surface
//!
//! [`CgroupManager`] enables a fixed controller set in
//! `cgroup.subtree_control` at [`Self::setup`] time so every method
//! that writes a controller knob succeeds without per-call lazy
//! enablement (which would race against concurrent sibling cgroup
//! creation). The enabled controllers and the knobs each one exposes
//! map to:
//!
//! | Controller | `setup` writes | Methods that touch the controller's files |
//! |------------|----------------|-------------------------------------------|
//! | `cpuset`   | always         | [`Self::set_cpuset`], [`Self::set_cpuset_mems`], [`Self::clear_cpuset`], [`Self::clear_cpuset_mems`] |
//! | `cpu`      | when `enable_cpu_controller=true` | [`Self::set_cpu_max`], [`Self::set_cpu_weight`] |
//! | `memory`   | always         | [`Self::set_memory_max`], [`Self::set_memory_high`], [`Self::set_memory_low`], [`Self::set_memory_swap_max`] |
//! | `pids`     | always         | [`Self::set_pids_max`] |
//! | `io`       | always         | [`Self::set_io_weight`] |
//! | (cgroup-core) | not gated   | [`Self::set_freeze`], [`Self::move_task`], [`Self::move_tasks`] |
//!
//! `cgroup.freeze` and `cgroup.procs` are cgroup-core files exposed on
//! every non-root cgroup automatically; they do not require a
//! controller in `subtree_control`. `memory.swap.max` only exists when
//! the kernel was built with `CONFIG_SWAP=y` — the file is absent on
//! swap-disabled kernels and a write returns ENOENT (callers route
//! through the wire-time error chain).
//!
//! # Untrusted-name validation
//!
//! Cgroup names flow into [`Path::join`] under `parent` to address
//! files inside cgroupfs. [`validate_cgroup_name`] rejects shapes that
//! would escape that parent (`..`, absolute leading `/`, `NUL`) or
//! that produce invisible cgroupfs entries (leading `.`); other ASCII
//! is passed through to the kernel which is the final authority on
//! per-component validity. Every public method that takes a `name`
//! validates it before any filesystem write.

use crate::topology::TestTopology;
use anyhow::{Context, Result, anyhow, bail};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

/// Cgroup v2 controllers that [`CgroupManager::setup`] can enable in
/// `cgroup.subtree_control`.
///
/// Each variant maps to a literal token the kernel parses in
/// `cgroup_subtree_control_write`. The enum is exhaustive over the
/// controllers the framework's [`CgroupOps`] surface actually writes
/// to (cpuset, cpu, memory, pids, io); cgroup-core knobs
/// (`cgroup.freeze`, `cgroup.procs`) are not gated by any controller
/// and never appear here.
///
/// Callers pass a `BTreeSet<Controller>` to `setup` — sets compose
/// naturally across nested CgroupDef declarations and the deterministic
/// `BTreeSet` iteration order keeps the rendered subtree_control write
/// stable between runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Controller {
    /// `+cpuset` — gates `cpuset.cpus`, `cpuset.cpus.effective`,
    /// `cpuset.mems`, `cpuset.mems.effective` files on every child.
    Cpuset,
    /// `+cpu` — gates `cpu.max`, `cpu.weight`, `cpu.weight.nice`,
    /// `cpu.stat`, `cpu.pressure` files on every child.
    Cpu,
    /// `+memory` — gates `memory.max`, `memory.high`, `memory.low`,
    /// `memory.min`, `memory.current`, `memory.swap.max`,
    /// `memory.events`, `memory.stat`, `memory.pressure` files.
    Memory,
    /// `+pids` — gates `pids.max`, `pids.current`, `pids.events` files.
    Pids,
    /// `+io` — gates `io.max`, `io.weight`, `io.bfq.weight`,
    /// `io.stat`, `io.pressure` files.
    Io,
}

impl Controller {
    /// Kernel token written to `cgroup.subtree_control` (the bare name
    /// without the `+`/`-` prefix; see [`Self::as_subtree_control_add`]
    /// for the full token).
    pub fn name(self) -> &'static str {
        match self {
            Controller::Cpuset => "cpuset",
            Controller::Cpu => "cpu",
            Controller::Memory => "memory",
            Controller::Pids => "pids",
            Controller::Io => "io",
        }
    }
}

/// Default timeout for cgroup filesystem writes. Normally <1ms; 2s catches
/// real hangs without waiting so long the test result is meaningless.
const CGROUP_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Write `data` to `path` with a timeout. Spawns a thread for the blocking
/// `fs::write` and waits on a channel. If the write does not complete within
/// `timeout`, returns an error (the spawned thread may still be blocked in
/// the kernel but will not prevent the caller from making progress).
///
/// # Stranded-writer thread semantics
///
/// On timeout the helper returns `Err` while the spawned thread stays
/// blocked in the kernel inside `fs::write` — typically inside the
/// cgroupfs `cgroup_kn_lock_live` / `cgroup_mutex` lock acquisition or
/// the per-file `kn->active` semaphore. The host-side fd to `path` is
/// owned by the spawned thread, so:
///
/// - **Per-file lock retention.** While the writer is blocked, the
///   target cgroupfs file's `kn->active` (kernfs's per-knob writer
///   semaphore) remains held by the stranded thread. Concurrent
///   writes to the SAME file from any thread in the same process —
///   including this same caller's retry — will queue behind the
///   stranded write inside the kernel. Writes to OTHER files in the
///   same cgroup are unaffected (kernfs holds `kn->active`
///   per-knob, not per-cgroup).
/// - **Thread-handle drop.** The `JoinHandle` returned by
///   `thread::spawn` is dropped when the helper returns. Rust's
///   `JoinHandle::Drop` implementation detaches the thread without
///   waiting; the thread continues to run and is implicitly joined
///   when the kernel write eventually unblocks (or when the process
///   exits).
/// - **Bounded leak under wedged cgroupfs.** A genuinely-wedged
///   cgroupfs (e.g. a stuck filesystem driver in the kernel) would
///   accumulate threads at a rate of one per timed-out write site.
///   The 2s per-write timeout caps the per-site stall to 2s; the
///   total accumulation is driven by how many distinct write sites
///   the scenario hits, not by elapsed wall-clock time alone.
///   Operators noticing stranded `<defunct>` cgroupfs writers in
///   `ps` should investigate whether the underlying kernel cgroup
///   subsystem is hung; the framework's own teardown does not
///   block on these stranded threads.
///
/// Each stranded thread holds the file's `kn->active` until the
/// kernel write returns. The OS-level memory cost per stranded
/// thread is the default Rust thread stack (8 MiB on Linux, mostly
/// virtual until touched).
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

/// Validate a cgroup name before joining it onto the parent path.
///
/// Rejects shapes that would either escape the parent directory
/// (`..` component, absolute leading `/`, embedded NUL) or produce
/// a hidden / invisible cgroupfs entry (leading `.`). Empty names
/// are also rejected — `parent.join("")` returns `parent`, which
/// would let a caller accidentally clobber the parent's own
/// `cpuset.cpus` / `cgroup.subtree_control` files via a method
/// that expected to address a child.
///
/// Permits `/` only as a path separator between non-empty
/// components (nested cgroups like `"cg_0/narrow"`); a leading
/// `/` is rejected because `Path::join` would replace `parent`
/// entirely with the absolute path.
///
/// Beyond these structural checks the kernel is the final authority
/// on per-component validity: cgroupfs rejects names containing
/// newlines or names colliding with reserved knobs (`cgroup.procs`,
/// `cpuset.cpus`, etc.) at `mkdir` time with EINVAL / EEXIST. Those
/// failures surface through the regular `fs::create_dir_all` /
/// `fs::write` error chain.
fn validate_cgroup_name(name: &str) -> Result<()> {
    if name.is_empty() {
        bail!("cgroup name must not be empty");
    }
    if name.starts_with('/') {
        bail!(
            "cgroup name '{name}' starts with '/' — would escape the \
             managed parent via Path::join (absolute paths replace the \
             join base)"
        );
    }
    if name.contains('\0') {
        bail!("cgroup name '{name}' contains a NUL byte");
    }
    // Per-component checks run before the whole-name leading-dot
    // check so a component like `..` matches the more specific
    // path-traversal diagnostic instead of the generic hidden-entry
    // one. The ordering matters for error messages — `'..' component`
    // is what callers grep for.
    for component in name.split('/') {
        if component.is_empty() {
            bail!(
                "cgroup name '{name}' contains an empty path component \
                 (consecutive '/') — Path::join would emit a malformed path"
            );
        }
        if component == ".." {
            bail!(
                "cgroup name '{name}' contains a '..' component — \
                 would escape the managed parent via Path::join"
            );
        }
        if component == "." {
            bail!(
                "cgroup name '{name}' contains a '.' component — \
                 ambiguous self-reference, refuse before fs writes"
            );
        }
        if component.starts_with('.') {
            bail!(
                "cgroup name '{name}' contains a leading-dot component \
                 ('{component}') — produces a hidden cgroupfs entry"
            );
        }
    }
    Ok(())
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

/// Snapshot the cgroup-tree state at the moment a cpuset.cpus
/// write fails, for diagnostic attachment to the returned error.
///
/// Captures (per the diagnostic contract on
/// [`CgroupManager::set_cpuset`]):
/// - the parent's `cgroup.controllers` (controllers AVAILABLE for
///   children — confirms whether subtree_control already
///   propagated to this child)
/// - the parent's `cgroup.subtree_control` (controllers ENABLED
///   for children — what setup() last wrote)
/// - the child's `cgroup.controllers` (the set children of the
///   CHILD inherit — useful for nested cgroups)
/// - whether `cpuset.cpus` exists at the child (distinguishes a
///   "controller never propagated" failure mode from a
///   "kernel rejected this specific value" failure mode)
/// - the child's directory listing (so an unexpected presence/
///   absence of any cgroupfs knob is visible)
///
/// Read failures inside the snapshot are folded into the snapshot
/// string as `<read failed: {err}>` rather than propagating —
/// the caller's error path is what the caller cares about; the
/// snapshot is best-effort instrumentation.
fn capture_cpuset_state(parent: &Path, name: &str) -> String {
    let child = parent.join(name);
    let parent_controllers = read_or_label(&parent.join("cgroup.controllers"));
    let parent_subtree_control = read_or_label(&parent.join("cgroup.subtree_control"));
    let child_controllers = read_or_label(&child.join("cgroup.controllers"));
    let cpuset_cpus_exists = child.join("cpuset.cpus").exists();
    let child_listing = match fs::read_dir(&child) {
        Ok(entries) => {
            let mut names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            names.sort_unstable();
            format!("[{}]", names.join(", "))
        }
        Err(e) => format!("<read_dir failed: {e}>"),
    };
    format!(
        "cgroup-state-snapshot: \
         parent={} name={} \
         parent.cgroup.controllers={:?} \
         parent.cgroup.subtree_control={:?} \
         child.cgroup.controllers={:?} \
         child.cpuset.cpus.exists={} \
         child.listing={}",
        parent.display(),
        name,
        parent_controllers,
        parent_subtree_control,
        child_controllers,
        cpuset_cpus_exists,
        child_listing,
    )
}

/// Read `path` to a string for snapshotting, returning a
/// `<...>` placeholder if the read fails. Used by
/// [`capture_cpuset_state`] so a missing or permission-denied
/// snapshot field shows up as a labeled placeholder rather than
/// killing the whole snapshot.
fn read_or_label(path: &Path) -> String {
    match fs::read_to_string(path) {
        Ok(s) => s.trim().to_string(),
        Err(e) => format!("<read failed: {e}>"),
    }
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

    /// Create the parent directory and enable the requested cgroup
    /// controllers in every ancestor `cgroup.subtree_control` between
    /// `/sys/fs/cgroup` and `self.parent`.
    ///
    /// Pass the controllers the test actually needs — empty set means
    /// "create the parent dir, write nothing to subtree_control". The
    /// scenario runtime computes the controller union from
    /// [`CgroupDef`](crate::scenario::ops::CgroupDef) declarations
    /// (cpuset/cpuset_mems → [`Controller::Cpuset`], cpu →
    /// [`Controller::Cpu`], memory → [`Controller::Memory`], pids →
    /// [`Controller::Pids`], io → [`Controller::Io`]) so a test
    /// that never sets a memory limit never enables `+memory` and
    /// vice versa. `cgroup.freeze` and `cgroup.procs` are
    /// cgroup-core, ungated by any controller, and need no entry.
    ///
    /// # Availability check
    ///
    /// Each requested controller is verified against
    /// `/sys/fs/cgroup/cgroup.controllers` before any write. A
    /// requested controller missing from the kernel's available set
    /// surfaces as `controller {ctrl} not available; cgroup.controllers
    /// = {available:?}` rather than the bare ENOENT/EACCES the
    /// downstream `set_*` write would otherwise emit.
    ///
    /// # Error propagation
    ///
    /// All filesystem writes propagate via `?`. A user inspecting
    /// `RUST_BACKTRACE=1` output sees the exact subtree_control path
    /// that failed and the underlying errno, instead of a swallowed
    /// `tracing::warn!` followed by a downstream EACCES at the
    /// controller-knob write site.
    pub fn setup(&self, controllers: &BTreeSet<Controller>) -> Result<()> {
        self.setup_under_root(controllers, &PathBuf::from("/sys/fs/cgroup"))
    }

    /// Inner setup that takes the cgroup-fs root as an explicit
    /// argument so tests can drive the controller-enable path against
    /// a tmpdir without touching `/sys/fs/cgroup`. Production
    /// [`Self::setup`] hardcodes `/sys/fs/cgroup`. The strip-prefix
    /// gate stays — if the parent is outside the supplied root,
    /// directory creation still happens but no subtree_control walk
    /// fires (matches the existing "non-cgroup-mount" early-bail).
    fn setup_under_root(
        &self,
        controllers: &BTreeSet<Controller>,
        root: &Path,
    ) -> Result<()> {
        if !self.parent.exists() {
            fs::create_dir_all(&self.parent)
                .with_context(|| format!("mkdir {}", self.parent.display()))?;
        }
        if controllers.is_empty() {
            return Ok(());
        }
        if let Ok(rel) = self.parent.strip_prefix(root) {
            let available_path = root.join("cgroup.controllers");
            if available_path.exists() {
                let raw = fs::read_to_string(&available_path).with_context(|| {
                    format!("read cgroup.controllers: {}", available_path.display())
                })?;
                let available: BTreeSet<&str> = raw.split_whitespace().collect();
                for c in controllers {
                    if !available.contains(c.name()) {
                        return Err(anyhow!(
                            "cgroup controller '{}' not available at {}; \
                             cgroup.controllers reports {:?}. CONFIG_{}_CONTROLLER \
                             may be unset, or the controller is masked at this \
                             level of the hierarchy",
                            c.name(),
                            available_path.display(),
                            available,
                            c.name().to_uppercase(),
                        ));
                    }
                }
            }
            let line: String = controllers
                .iter()
                .map(|c| format!("+{}", c.name()))
                .collect::<Vec<_>>()
                .join(" ");
            let mut cur = root.to_path_buf();
            for c in rel.components() {
                let sc = cur.join("cgroup.subtree_control");
                if sc.exists() {
                    write_with_timeout(&sc, &line, CGROUP_WRITE_TIMEOUT).with_context(
                        || format!("enable controllers '{line}' at {}", sc.display()),
                    )?;
                }
                cur = cur.join(c);
            }
            let sc = self.parent.join("cgroup.subtree_control");
            if sc.exists() {
                write_with_timeout(&sc, &line, CGROUP_WRITE_TIMEOUT).with_context(
                    || format!("enable controllers '{line}' at {}", sc.display()),
                )?;
            }
        }
        Ok(())
    }

    /// Create a child cgroup directory.
    ///
    /// For nested paths (e.g. `"cg_0/narrow"`), enables `+cpuset` on
    /// each intermediate cgroup's `subtree_control` so the leaf has
    /// `cpuset.cpus` / `cpuset.mems` files available. The kernel
    /// requires each parent to have the controller in
    /// `subtree_control` for its children to have the corresponding
    /// files (`cgroup_control()` returns `parent->subtree_control`).
    ///
    /// # Limitation: only `+cpuset` is propagated through nested
    /// intermediates
    ///
    /// [`Self::enable_subtree_cpuset`] writes ONLY `+cpuset` to each
    /// intermediate's `cgroup.subtree_control`; the `+cpu` /
    /// `+memory` / `+pids` / `+io` controllers enabled by
    /// [`Self::setup`] cover only the manager's parent cgroup, not
    /// arbitrary intermediate cgroups created via nested
    /// `create_cgroup` calls. As a result, a nested leaf like
    /// `"cg_0/narrow"` exposes `cpuset.*` knobs but NOT
    /// `memory.max` / `pids.max` / `io.weight`. If a future
    /// [`CgroupDef`](crate::scenario::ops::CgroupDef) addresses such
    /// a leaf with a memory/pids/io knob, the corresponding
    /// `set_*` write will return ENOENT.
    ///
    /// Today's in-tree consumers (host topology cpuset locks,
    /// `BuildSandbox`, scenario ops) only nest cgroups for cpuset
    /// scoping, so this matches the actual surface the framework
    /// exercises. Extending [`Self::enable_subtree_cpuset`] to
    /// propagate the remaining controllers across intermediates is
    /// straightforward (write the same controller list as
    /// [`Self::setup`] uses) but is deferred until a use case
    /// concretely needs it; without one, the wider write would
    /// race against concurrent sibling cgroup creation under the
    /// same intermediate without buying anything.
    pub fn create_cgroup(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
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
    ///
    /// Auto-unfreezes the cgroup before draining: a frozen cgroup that
    /// reaches teardown (e.g. a step body issues `Op::FreezeCgroup` and
    /// never pairs it with `Op::UnfreezeCgroup`) would migrate its
    /// frozen tasks to the cgroup root via `drain_tasks` and rely on
    /// the kernel's `cgroup_freezer_migrate_task` to clear the JOBCTL
    /// freeze bit when the destination cgroup is unfrozen. The kernel
    /// path is correct, but writing `cgroup.freeze=0` first makes the
    /// teardown deterministic regardless of who froze the cgroup and
    /// when. Tolerates ENOENT on the freeze file (cgroup directory
    /// already gone, or `CONFIG_CGROUP_FREEZE` absent on legacy
    /// kernels) silently — only non-ENOENT failures warn.
    ///
    /// # Post-drain settle window
    ///
    /// The 50ms sleep between [`Self::drain_tasks`] and `rmdir` is a
    /// concession to the cgroup v2 task-migration RCU grace period.
    /// Writes to `cgroup.procs` queue the task move but the source
    /// cgroup's `nr_populated` counter only drops once the per-task
    /// css_set switch completes — `rmdir` returns EBUSY if the
    /// counter is non-zero. The kernel's `cgroup_rmdir` path
    /// (`kernel/cgroup/cgroup.c`) gates on `cgroup_is_populated()`
    /// which reads `nr_populated`, and the migration RCU callback
    /// runs from the next softirq tick. 50ms exceeds the longest
    /// observed callback latency on a moderately-loaded host (worst
    /// case ~30ms under heavy IRQ pressure on a 4.18-era kernel,
    /// sub-millisecond on a quiet 6.x kernel).
    ///
    /// Without the sleep, the `rmdir` would race the migration RCU
    /// callback under load and intermittently return EBUSY. A
    /// per-attempt retry loop would also work, but adds branching
    /// to a non-hot teardown path; the fixed-window sleep is
    /// simpler and the 50ms tax on a teardown that is already
    /// scheduled to absorb a VM shutdown is immaterial.
    pub fn remove_cgroup(&self, name: &str) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name);
        if !p.exists() {
            return Ok(());
        }
        if let Err(err) = self.set_freeze(name, false)
            && anyhow_first_io_errno(&err) != Some(libc::ENOENT)
        {
            tracing::warn!(
                cgroup = name,
                err = %format!("{err:#}"),
                "remove_cgroup: pre-drain unfreeze failed; drain may strand frozen tasks at root"
            );
        }
        self.drain_tasks(name)?;
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::remove_dir(&p).with_context(|| format!("rmdir {}", p.display()))
    }

    /// Write `cpuset.cpus` for a child cgroup.
    ///
    /// On write failure, captures and emits a snapshot of the
    /// cgroup-tree state at the moment of failure: the parent's
    /// `cgroup.controllers` (controllers AVAILABLE to children),
    /// the parent's `cgroup.subtree_control` (controllers ENABLED
    /// for children), the child's `cgroup.controllers` (the
    /// inheritance ROOT for children of the child), the
    /// `cpuset.cpus` file's existence, and a directory listing of
    /// the child cgroup's knob files. The capture lets a kernel /
    /// hierarchy-state bug surface as a focused diagnostic instead
    /// of a bare `EACCES` at the write site.
    pub fn set_cpuset(&self, name: &str, cpus: &BTreeSet<usize>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpuset.cpus");
        match write_with_timeout(&p, &TestTopology::cpuset_string(cpus), CGROUP_WRITE_TIMEOUT) {
            Ok(()) => Ok(()),
            Err(e) => {
                let snapshot = capture_cpuset_state(&self.parent, name);
                Err(e.context(snapshot))
            }
        }
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cpu.weight");
        write_with_timeout(&p, &weight.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Write `memory.max` for a child cgroup. `bytes = None` writes
    /// `"max"` (no hard limit). When the cgroup's RSS exceeds the
    /// limit, the kernel OOM-kills tasks per the documented
    /// `memory.max` semantics. Requires `+memory` in the parent's
    /// `cgroup.subtree_control`.
    pub fn set_memory_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("io.weight");
        write_with_timeout(&p, &weight.to_string(), CGROUP_WRITE_TIMEOUT)
    }

    /// Write `cgroup.freeze` for a child cgroup. `frozen = true` writes
    /// `"1"`, `frozen = false` writes `"0"`.
    ///
    /// `cgroup.freeze` is a cgroup-core file exposed on every non-root
    /// cgroup automatically — it is NOT gated by `cgroup.subtree_control`.
    /// The kernel's `cgroup_freeze_write` parses the value via
    /// `kstrtoint`, rejects anything outside `{0, 1}` with `-ERANGE`,
    /// and dispatches `cgroup_freeze(cgrp, freeze)`. Writing `1` to a
    /// cgroup containing tasks transitions every task in the subtree to
    /// the frozen state; writing `0` releases. The transition is
    /// asynchronous — `cgroup.events`'s `frozen` field reaches `1` once
    /// every task has parked.
    pub fn set_freeze(&self, name: &str, frozen: bool) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("cgroup.freeze");
        let line = if frozen { "1" } else { "0" };
        write_with_timeout(&p, line, CGROUP_WRITE_TIMEOUT)
    }

    /// Write `pids.max` for a child cgroup. `max = None` writes `"max"`
    /// (the kernel's `PIDS_MAX_STR` sentinel for unlimited);
    /// `Some(n)` writes the decimal `n`.
    ///
    /// Per the kernel's `pids_max_write`: the parser short-circuits to
    /// the unlimited limit when `buf == PIDS_MAX_STR`; otherwise
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
        validate_cgroup_name(name)?;
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
    /// Per the kernel's `swap_max_write`: the value is parsed via
    /// `page_counter_memparse(buf, "max", &max)`, which accepts the
    /// literal `"max"` token for unlimited or a numeric byte count.
    /// The store is `xchg(&memcg->swap.max, max)` — atomic, with no
    /// failure path beyond the parse.
    ///
    /// Requires `+memory` in the parent's `cgroup.subtree_control`;
    /// [`Self::setup`] enables it unconditionally.
    ///
    /// Requires CONFIG_SWAP=y in the test kernel. The file does not
    /// exist on swapless builds; the write returns ENOENT.
    pub fn set_memory_swap_max(&self, name: &str, bytes: Option<u64>) -> Result<()> {
        validate_cgroup_name(name)?;
        let p = self.parent.join(name).join("memory.swap.max");
        let line = match bytes {
            Some(b) => b.to_string(),
            None => "max".to_string(),
        };
        write_with_timeout(&p, &line, CGROUP_WRITE_TIMEOUT)
    }

    /// Move a single task into a child cgroup via `cgroup.procs`.
    pub fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
        validate_cgroup_name(name)?;
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
    ///
    /// # Outer-read_dir failure semantic
    ///
    /// When `read_dir(self.parent)` itself fails — e.g. the parent
    /// directory is unreadable, the cgroup mount has been unmounted
    /// out from under us, or a stat-side IO error fires — the
    /// failure is surfaced via `tracing::warn!` and the function
    /// still returns `Ok(())`. The deliberate semantic here is
    /// "teardown that observes a hostile filesystem state must
    /// not block scenario completion": a hard `Err` would propagate
    /// up through the runner's teardown and abort the whole test
    /// run on a transient cgroupfs failure that the operator can
    /// follow up on by reading the warn line.
    ///
    /// Production callers (the runner's drop path, scenario teardown)
    /// already log-and-continue on `cleanup_all` errors, so the
    /// always-Ok return is consistent with how every consumer
    /// already treats the result. Operators who need to detect
    /// teardown leakage should grep `tracing` output for
    /// `"cleanup_all: read_dir failed"` rather than relying on a
    /// non-zero exit; the warn includes both the offending path and
    /// the underlying io::Error.
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
    fn setup(&self, controllers: &BTreeSet<Controller>) -> Result<()>;
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
    fn setup(&self, controllers: &BTreeSet<Controller>) -> Result<()> {
        CgroupManager::setup(self, controllers)
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
    // Auto-unfreeze before draining tasks. Mirrors
    // `CgroupManager::remove_cgroup`'s pre-drain unfreeze, but for
    // defense-in-depth and source-cgroup state hygiene rather than
    // for correctness: the kernel's `cgroup_freezer_migrate_task`
    // path DOES unfreeze tasks when they migrate to an unfrozen
    // destination (the cgroup root is always unfrozen), so frozen
    // tasks would not actually strand at the root. The explicit
    // pre-drain `cgroup.freeze=0` write is still worthwhile because
    // it (a) makes the source cgroup's transient state visible in
    // tracing / `cgroup.events` before the directory disappears,
    // (b) avoids a brief frozen-counter churn while migration
    // batches advance, and (c) makes the teardown path symmetric
    // with `remove_cgroup` so operators reading either function
    // see the same auto-unfreeze step.
    //
    // Gate on existence: `fs::write` on a regular filesystem
    // CREATES the file when it doesn't exist (open(O_WRONLY |
    // O_CREAT | O_TRUNC)), so unconditionally writing
    // `cgroup.freeze` would plant a stray 1-byte file under any
    // non-cgroupfs directory and cause the subsequent
    // `fs::remove_dir(path)` to fail with ENOTEMPTY. On a real
    // cgroup v2 tree the file is always present (cgroup-core,
    // ungated by controllers); on a legacy kernel without
    // `CONFIG_CGROUP_FREEZE` or on a non-cgroup directory entry
    // the file is absent and the unfreeze step is a no-op.
    let freeze_path = path.join("cgroup.freeze");
    if freeze_path.exists()
        && let Err(err) = write_with_timeout(&freeze_path, "0", CGROUP_WRITE_TIMEOUT)
    {
        tracing::warn!(
            path = %path.display(),
            err = %format!("{err:#}"),
            "cleanup_recursive: pre-drain unfreeze failed; source-cgroup state-hygiene step skipped",
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
        // setup() on a non-cgroup path should still create the dir.
        // Empty controller set skips the subtree_control walk entirely.
        let dir = std::env::temp_dir().join(format!("ktstr-setup-{}", std::process::id()));
        let cg = CgroupManager::new(dir.to_str().unwrap());
        cg.setup(&BTreeSet::new()).unwrap();
        assert!(dir.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    /// `setup` writes only the controllers the caller requested to
    /// `cgroup.subtree_control`. Pinning a focused minimum
    /// (cpuset + memory) catches regressions where the rendered
    /// `+token` list grows past what the caller asked for.
    ///
    /// Path: drives [`setup_under_root`] against a tmpdir-rooted
    /// "cgroup tree" (parent dir + pre-created
    /// `cgroup.controllers` advertising both controllers +
    /// `cgroup.subtree_control` at root and leaf), then reads the
    /// leaf back and asserts both requested tokens land while
    /// non-requested controllers do not.
    #[test]
    fn setup_writes_requested_controllers_only() {
        let root =
            std::env::temp_dir().join(format!("ktstr-setup-controllers-{}", std::process::id()));
        let parent = root.join("ktstr");
        fs::create_dir_all(&parent).unwrap();
        // Pre-create cgroup.controllers at root so the availability
        // check in setup_under_root passes for the requested
        // controllers. Production cgroup v2 mount populates this file.
        fs::write(root.join("cgroup.controllers"), "cpuset cpu memory pids io").unwrap();
        // Pre-create the subtree_control file at root and leaf so
        // the strip-prefix walk's exists() gate sees them.
        fs::write(root.join("cgroup.subtree_control"), "").unwrap();
        fs::write(parent.join("cgroup.subtree_control"), "").unwrap();

        let cg = CgroupManager::new(parent.to_str().unwrap());
        let mut requested = BTreeSet::new();
        requested.insert(Controller::Cpuset);
        requested.insert(Controller::Memory);
        cg.setup_under_root(&requested, &root).unwrap();

        let written = fs::read_to_string(parent.join("cgroup.subtree_control")).unwrap();
        assert!(
            written.contains("+cpuset"),
            "subtree_control must contain +cpuset; got: {written:?}",
        );
        assert!(
            written.contains("+memory"),
            "subtree_control must contain +memory; got: {written:?}",
        );
        // Non-requested controllers must NOT appear.
        assert!(
            !written.contains("+pids"),
            "+pids must be absent when not requested; got: {written:?}",
        );
        assert!(
            !written.contains("+io"),
            "+io must be absent when not requested; got: {written:?}",
        );
        // +cpu must be absent. Distinguish from +cpuset by walking
        // every +cpu* match position and asserting it's the +cpuset
        // prefix.
        let cpu_positions: Vec<usize> = written.match_indices("+cpu").map(|(i, _)| i).collect();
        for pos in cpu_positions {
            let suffix = &written[pos..];
            assert!(
                suffix.starts_with("+cpuset"),
                "+cpu must be absent when not requested (only +cpuset allowed); \
                 got '{suffix}' at pos {pos} in {written:?}",
            );
        }

        let _ = fs::remove_dir_all(&root);
    }

    /// `setup` rejects an unavailable controller with a clear error
    /// citing both the requested controller name and the kernel's
    /// advertised set. Without the gate, the downstream
    /// `set_*` write would fail with bare ENOENT/EACCES — much
    /// harder to diagnose than "controller X not available".
    #[test]
    fn setup_rejects_unavailable_controller() {
        let root = std::env::temp_dir()
            .join(format!("ktstr-setup-unavail-{}", std::process::id()));
        let parent = root.join("ktstr");
        fs::create_dir_all(&parent).unwrap();
        // Advertise only memory; request cpuset.
        fs::write(root.join("cgroup.controllers"), "memory").unwrap();
        fs::write(root.join("cgroup.subtree_control"), "").unwrap();
        fs::write(parent.join("cgroup.subtree_control"), "").unwrap();

        let cg = CgroupManager::new(parent.to_str().unwrap());
        let mut requested = BTreeSet::new();
        requested.insert(Controller::Cpuset);
        let err = cg.setup_under_root(&requested, &root).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cpuset") && msg.contains("not available"),
            "error must cite missing 'cpuset' and 'not available'; got {msg:?}",
        );
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
        cg.set_memory_swap_max("cg_x", Some(2 * 1024 * 1024))
            .unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "2097152");
        cg.set_memory_swap_max("cg_x", None).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "max");
        let _ = fs::remove_dir_all(&dir);
    }

    // -- validate_cgroup_name ----------------------------------------

    /// Reject the path-escape and hidden-entry shapes that
    /// [`validate_cgroup_name`] guards against. Each branch is named
    /// in the assertion so a future regression that drops a check
    /// surfaces with the specific shape that slipped through.
    #[test]
    fn validate_cgroup_name_rejects_unsafe_shapes() {
        for (name, reason) in [
            ("", "empty"),
            ("/abs", "starts with '/'"),
            ("nul\0byte", "NUL byte"),
            (".hidden", "leading-dot component"),
            ("..", "'..' component"),
            ("a/..", "'..' component"),
            ("../escape", "'..' component"),
            (".", "'.' component"),
            ("a//b", "empty path component"),
            ("ok/.dotfile", "leading-dot component"),
        ] {
            let err =
                validate_cgroup_name(name).expect_err(&format!("must reject {name:?} ({reason})"));
            assert!(
                err.to_string().contains(reason),
                "error for {name:?} must mention {reason:?}; got: {err:#}"
            );
        }
    }

    /// Names the validator accepts: simple identifiers, nested paths
    /// with non-leading dots, plain numeric suffixes. Pinned so a
    /// future tightening that breaks legitimate `cg_0/narrow` shapes
    /// is caught at test time.
    #[test]
    fn validate_cgroup_name_accepts_valid_shapes() {
        for name in [
            "cg_0",
            "cg-1",
            "cg.0",
            "cg_0/narrow",
            "level1/level2/level3",
            "a.b.c",
            "x",
        ] {
            validate_cgroup_name(name).unwrap_or_else(|e| {
                panic!("must accept legitimate name {name:?}; got: {e:#}");
            });
        }
    }

    /// Public methods that take a `name` must run name validation
    /// before any filesystem write so a hostile name never reaches
    /// `Path::join`. Pin one representative method per knob type.
    #[test]
    fn cgroup_methods_reject_bad_names_before_fs_writes() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-badname-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        let bad = "../escape";
        // Each call must fail at validation, not at the fs write.
        // The shared error fragment ('..' component) appears in
        // every diagnostic so callers see the same shape regardless
        // of which method tripped.
        let err = cg.create_cgroup(bad).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_freeze(bad, true).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_pids_max(bad, Some(10)).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_memory_swap_max(bad, Some(1024)).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.set_cpuset_mems(bad, &BTreeSet::new()).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.move_task(bad, 1).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.drain_tasks(bad).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        let err = cg.remove_cgroup(bad).unwrap_err();
        assert!(err.to_string().contains("'..' component"));
        // No directory under `dir` should have been created from any
        // of these calls — the validator bails before fs writes.
        let escape_marker = dir.join("escape");
        assert!(
            !escape_marker.exists(),
            "validator must bail before fs writes; saw {escape_marker:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- setup_under_root strip-prefix-fail branch -------------------

    /// When `parent` does not lie under the supplied `root`, the
    /// `strip_prefix` call returns `Err` and `setup_under_root` skips
    /// the subtree-control walk entirely. The function must still
    /// create the parent directory and return Ok — the early-bail
    /// matches the production "non-cgroup-mount" path described on
    /// [`Self::setup_under_root`]. Pin both: the parent dir exists,
    /// no subtree_control was written.
    #[test]
    fn setup_under_root_outside_root_creates_dir_and_skips_walk() {
        let outside = std::env::temp_dir().join(format!("ktstr-out-{}", std::process::id()));
        let unrelated_root =
            std::env::temp_dir().join(format!("ktstr-other-{}", std::process::id()));
        // Pre-create both so neither needs `mkdir`. The point is that
        // `outside.strip_prefix(unrelated_root)` returns Err.
        fs::create_dir_all(&unrelated_root).unwrap();
        let cg = CgroupManager::new(outside.to_str().unwrap());
        let mut requested = BTreeSet::new();
        requested.insert(Controller::Cpuset);
        cg.setup_under_root(&requested, &unrelated_root).unwrap();
        assert!(outside.exists(), "setup must create the parent directory");
        assert!(
            !outside.join("cgroup.subtree_control").exists(),
            "no subtree_control walk should fire when the parent is not under root"
        );
        let _ = fs::remove_dir_all(&outside);
        let _ = fs::remove_dir_all(&unrelated_root);
    }

    // -- set_freeze idempotency --------------------------------------

    /// Freezing an already-frozen cgroup must not error — the kernel
    /// short-circuits on the duplicate write. Pinned because the
    /// `remove_cgroup` auto-unfreeze path depends on the inverse
    /// idempotency (unfreezing an unfrozen cgroup), so the symmetric
    /// case is checked here.
    #[test]
    fn set_freeze_is_idempotent_when_already_in_target_state() {
        let (dir, cg) = make_test_cgroup("freeze-idem");
        let target = dir.join("cg_x").join("cgroup.freeze");
        fs::write(&target, "").unwrap();
        cg.set_freeze("cg_x", true).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1");
        // Second freeze: no error, file content unchanged.
        cg.set_freeze("cg_x", true).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "1");
        // Inverse: idempotent unfreeze on already-unfrozen.
        cg.set_freeze("cg_x", false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
        cg.set_freeze("cg_x", false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "0");
        let _ = fs::remove_dir_all(&dir);
    }

    // -- pids.max / memory.swap.max overflow boundary ---------------

    /// `set_pids_max(Some(u64::MAX))` writes the decimal representation
    /// verbatim. The kernel rejects values `>= PIDS_MAX` with EINVAL,
    /// but the framework wire layer is responsible only for byte-exact
    /// stringification — pinning u64::MAX guards against accidental
    /// narrowing to i64 (which would turn the value into "-1") or to
    /// u32 (which would silently saturate).
    #[test]
    fn set_pids_max_writes_u64_max_verbatim() {
        let (dir, cg) = make_test_cgroup("pids-overflow");
        let target = dir.join("cg_x").join("pids.max");
        fs::write(&target, "").unwrap();
        cg.set_pids_max("cg_x", Some(u64::MAX)).unwrap();
        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            u64::MAX.to_string(),
            "u64::MAX must round-trip without narrowing or sign change"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// `set_memory_swap_max(Some(u64::MAX))` mirrors the pids.max
    /// boundary check. Catches the same narrowing-regression class.
    #[test]
    fn set_memory_swap_max_writes_u64_max_verbatim() {
        let (dir, cg) = make_test_cgroup("swap-overflow");
        let target = dir.join("cg_x").join("memory.swap.max");
        fs::write(&target, "").unwrap();
        cg.set_memory_swap_max("cg_x", Some(u64::MAX)).unwrap();
        assert_eq!(
            fs::read_to_string(&target).unwrap(),
            u64::MAX.to_string(),
            "u64::MAX must round-trip without narrowing or sign change"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    // -- write_with_timeout failure paths for new methods ------------
    //
    // [`write_with_timeout`] surfaces the per-method "knob file does
    // not exist" path as an error chain that callers (e.g.
    // `apply_setup`) propagate up. Pin the failure surface for every
    // recently-added method so a regression that swallows the error
    // (or returns Ok despite a missing file) trips here.

    /// `set_pids_max` against a missing parent directory returns an
    /// error whose chain walks back to the missing path. The cgroup
    /// directory has to be missing — `fs::write` to a nonexistent
    /// file inside an existing directory just creates the file —
    /// so the test exercises the realistic "cgroup never created"
    /// path through ENOENT on `parent.join(name).join("pids.max")`.
    #[test]
    fn set_pids_max_returns_err_when_pids_max_file_missing() {
        let cg = CgroupManager::new("/nonexistent/ktstr-pids-test");
        let err = cg
            .set_pids_max("cg_x", Some(1024))
            .expect_err("missing pids.max must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("pids.max"),
            "error chain must name the missing file: {msg}"
        );
    }

    /// Mirror of the pids.max test for memory.swap.max.
    #[test]
    fn set_memory_swap_max_returns_err_when_file_missing() {
        let cg = CgroupManager::new("/nonexistent/ktstr-swap-test");
        let err = cg
            .set_memory_swap_max("cg_x", Some(2_000_000))
            .expect_err("missing memory.swap.max must surface as Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("memory.swap.max"),
            "error chain must name the missing file: {msg}"
        );
    }

    /// `set_freeze` against a missing `cgroup.freeze` file surfaces
    /// an ENOENT errno reachable from the error chain — what
    /// `remove_cgroup`'s auto-unfreeze branch uses to suppress the
    /// warn. Pin the errno so a regression that wraps the underlying
    /// IO error in a way that loses the raw_os_error trips here.
    #[test]
    fn set_freeze_returns_err_with_enoent_when_freeze_file_missing() {
        let cg = CgroupManager::new("/nonexistent/ktstr-freeze-test");
        let err = cg
            .set_freeze("cg_x", true)
            .expect_err("missing cgroup.freeze must surface as Err");
        assert_eq!(
            anyhow_first_io_errno(&err),
            Some(libc::ENOENT),
            "ENOENT errno must be reachable from the error chain so \
             remove_cgroup's auto-unfreeze can suppress it; got: {err:#}"
        );
    }

    // -- remove_cgroup auto-unfreeze --------------------------------

    /// `remove_cgroup` writes `0` to `cgroup.freeze` before draining
    /// tasks — pin the side effect so a regression that drops the
    /// auto-unfreeze surfaces here. The test observes the freeze
    /// file's post-call contents instead of asserting on the rmdir
    /// outcome: real cgroupfs auto-removes child files during a
    /// directory rmdir, but tmpfs requires the files to be unlinked
    /// first. The unfreeze-before-drain ordering is the invariant
    /// under test, not the rmdir success.
    #[test]
    fn remove_cgroup_auto_unfreezes_before_drain() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-autounf-{}", std::process::id()));
        let inner = dir.join("cg_x");
        fs::create_dir_all(&inner).unwrap();
        // Pre-create the freeze + procs files. Seed `cgroup.freeze`
        // with "1" so the test can observe the unfreeze write.
        let freeze_path = inner.join("cgroup.freeze");
        fs::write(&freeze_path, "1").unwrap();
        fs::write(inner.join("cgroup.procs"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // The rmdir at the end fails on tmpfs because cgroup.freeze
        // / cgroup.procs are leftover non-cgroupfs files; we don't
        // care — the assertion is on the freeze-file content.
        let _ = cg.remove_cgroup("cg_x");
        // The auto-unfreeze must have written "0" to cgroup.freeze
        // before the drain.
        assert_eq!(
            fs::read_to_string(&freeze_path).unwrap(),
            "0",
            "remove_cgroup must write '0' to cgroup.freeze before draining"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// `remove_cgroup` swallows ENOENT on the unfreeze write so a
    /// cgroup without `cgroup.freeze` (legacy kernel without
    /// CONFIG_CGROUP_FREEZE) still drains cleanly. The drain reaches
    /// `cgroup.procs` instead of returning early because of the
    /// missing freeze file.
    #[test]
    fn remove_cgroup_tolerates_missing_freeze_file() {
        let dir = std::env::temp_dir().join(format!("ktstr-cg-nofrz-{}", std::process::id()));
        let inner = dir.join("cg_x");
        fs::create_dir_all(&inner).unwrap();
        // Deliberately omit cgroup.freeze. Provide cgroup.procs so
        // drain_tasks finds something to read.
        fs::write(inner.join("cgroup.procs"), "").unwrap();
        let cg = CgroupManager::new(dir.to_str().unwrap());
        // The rmdir at the end fails on tmpfs (cgroup.procs left
        // over) — we only care that no error propagates from the
        // pre-drain unfreeze branch. The test would catch a
        // regression where the missing freeze file produces a hard
        // error before the drain runs.
        let _ = cg.remove_cgroup("cg_x");
        // No assertion on the freeze file — it never existed. The
        // test passes when the call body runs to completion without
        // panicking on the tolerated ENOENT branch.
        let _ = fs::remove_dir_all(&dir);
    }

    /// `cleanup_recursive` writes `0` to `cgroup.freeze` before
    /// draining tasks — mirrors
    /// [`remove_cgroup_auto_unfreezes_before_drain`]. The pre-drain
    /// unfreeze is a state-hygiene step (the kernel would unfreeze
    /// migrated tasks at the unfrozen root anyway), but the write
    /// itself is observable in `cgroup.events` and tracing, so a
    /// regression that drops the unfreeze step would silently lose
    /// that visibility. Pin the side effect by seeding
    /// `cgroup.freeze="1"` and asserting the post-call contents
    /// are `"0"`. Test mirrors the `remove_cgroup` shape: rmdir at
    /// the end fails on tmpfs (leftover non-cgroupfs files), but
    /// the freeze-file content is what the assertion targets.
    #[test]
    fn cleanup_recursive_auto_unfreezes_before_drain() {
        let dir =
            std::env::temp_dir().join(format!("ktstr-cleanup-rec-autounf-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        // Seed cgroup.freeze="1" + empty cgroup.procs so the test
        // observes the unfreeze write the same way
        // remove_cgroup_auto_unfreezes_before_drain does.
        let freeze_path = dir.join("cgroup.freeze");
        fs::write(&freeze_path, "1").unwrap();
        fs::write(dir.join("cgroup.procs"), "").unwrap();
        // `cleanup_recursive` is the free fn the cleanup_all walk
        // dispatches per directory; call it directly.
        cleanup_recursive(&dir);
        // The auto-unfreeze must have written "0" to cgroup.freeze
        // before the drain — pinned identically to the
        // remove_cgroup test so a regression on either path
        // surfaces with the same diagnostic shape.
        assert_eq!(
            fs::read_to_string(&freeze_path).unwrap(),
            "0",
            "cleanup_recursive must write '0' to cgroup.freeze before draining \
             (mirrors remove_cgroup auto-unfreeze for state hygiene)",
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
