//! Build-time cgroup v2 sandbox for kernel compilation under an
//! [`LlcPlan`](super::host_topology::LlcPlan) reservation.
//!
//! Wraps [`CgroupManager`](crate::cgroup::CgroupManager) to enforce CPU
//! + NUMA memory binding for `make` and its gcc/ld children when the
//! user passes `--llc-cap N`. The sandbox contract is:
//!
//!  - A dedicated child cgroup is created under the caller's own
//!    cgroup (parsed from `/proc/self/cgroup`). Name format is
//!    `ktstr-build-{epoch_nanos}-{pid}` so two concurrent builds
//!    never collide and an orphan from a previous run is
//!    identifiable by its `{pid}` suffix.
//!  - Writes are strictly ordered per kernel rules:
//!        cpuset.cpus → cpuset.mems → cgroup.procs
//!    A task in a cgroup with an empty `cpuset.mems` may be killed by
//!    the kernel's cpuset allocation path; migrating the build shell
//!    into `cgroup.procs` before both cpuset fields are populated
//!    risks SIGKILL on the next allocation.
//!  - After each cpuset write, the `.effective` file is read back.
//!    Disagreement between the requested and effective set means the
//!    child's parent had a narrower view than the plan asked for
//!    (e.g. kernel returned `EINVAL` silently, systemd restricted
//!    the slice, or a prior ancestor cpuset shrunk our window).
//!    Degradation under `--llc-cap` is fatal; without the flag, it
//!    warns and proceeds.
//!  - `Drop` migrates the build pid back to root, tolerates
//!    transient EBUSY on `cgroup.rmdir` (up to 5 x 10ms retries),
//!    and warn-logs `tag=resource_budget.cgroup_orphan_left` if the
//!    directory still refuses to go away. The lock file that
//!    anchored the reservation is released separately by
//!    [`LlcPlan`](super::host_topology::LlcPlan)'s own Drop.
//!
//! This module is Linux-only; callers must arrive here already
//! holding an `LlcPlan` from a successful
//! [`acquire_llc_plan`](super::host_topology::acquire_llc_plan)
//! invocation, so the cpu + mem sets are guaranteed to be non-empty.

use crate::cgroup::{CgroupManager, anyhow_first_io_errno};
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// `statfs(2)` magic for cgroup v2.
///
/// Mirrors `libc::CGROUP2_SUPER_MAGIC` (value `0x63677270`). Kept
/// as a named constant in this module so `no_cgroup2` detection
/// reads clearly in `try_create`. Value type chosen as `i64` so the
/// comparison against `StatFs::f_type` works uniformly across
/// glibc architectures that define `__fsword_t` as either `i32`
/// or `i64`.
const CGROUP2_SUPER_MAGIC: i64 = 0x6367_7270;

/// Ceiling on the EBUSY retry loop in Drop. Five attempts at 10ms
/// gives kernel housekeeping 50ms to release references the
/// build process may have left on the cgroup (e.g. a grandchild
/// caught mid-fork). Past that, the directory is orphaned and
/// logged; re-running `ktstr cleanup` (which recurses into
/// `/sys/fs/cgroup/…/ktstr-build-*`) sweeps the orphan later.
const RMDIR_EBUSY_RETRIES: u32 = 5;
/// Per-attempt backoff for `RMDIR_EBUSY_RETRIES`.
const RMDIR_EBUSY_BACKOFF: Duration = Duration::from_millis(10);

/// Orphan sweep age threshold. A `ktstr-build-*` directory whose
/// `mtime` is older than this and whose embedded `{pid}` no longer
/// exists is assumed crash-leftover and removed on sandbox
/// creation. 24h is long enough that a suspended kernel build
/// holding an ancient cgroup remains visible rather than silently
/// destroyed, and short enough that a typical CI host's orphans
/// don't accumulate across days.
const ORPHAN_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// Failure modes for the cgroup v2 sandbox. Surfaces as
/// `BuildSandbox::Degraded(…)` when `--llc-cap` is NOT set; a
/// future tightening under the flag would reject these.
///
/// `#[non_exhaustive]` because the kernel surface we sense grows
/// over time (e.g. cgroup v1-v2 hybrid, userns-restricted child
/// cgroups) and we want to add variants without breaking external
/// consumers' match arms.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SandboxDegraded {
    /// `/sys/fs/cgroup` is not mounted as cgroup v2 (wrong magic or
    /// not mounted at all). We cannot enforce CPU / NUMA binding.
    NoCgroupV2,
    /// cgroup v2 is mounted, but this process's parent cgroup does
    /// not advertise `cpuset` in `cgroup.controllers`. Likely
    /// systemd nspawn / container config that hid the controller.
    NoCpusetController,
    /// Attempted to write `+cpuset` to the parent's
    /// `cgroup.subtree_control` and got rejected (typically EBUSY
    /// when another tool holds the subtree, or EACCES under a
    /// nested userns).
    SubtreeControlRefused,
    /// Lacked permission (EACCES / EPERM) to create the child
    /// cgroup or write its cpuset files. Build continues without
    /// enforcement.
    PermissionDenied,
    /// The caller's own cgroup is the cgroup v2 root
    /// (`/proc/self/cgroup` reads `0::/`). The root has no parent
    /// subtree_control to write `+cpuset` into, and writing to its
    /// own cgroup.procs is a kernel-constraint violation under the
    /// no-internal-process rule. ktstr refuses to create a sandbox
    /// at this location — operators must run under a non-root
    /// cgroup (systemd-run --user --scope, sudo -E under a slice,
    /// or an explicit delegation subtree).
    RootCgroupRefused,
}

impl std::fmt::Display for SandboxDegraded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxDegraded::NoCgroupV2 => {
                write!(f, "cgroup v2 not mounted at /sys/fs/cgroup")
            }
            SandboxDegraded::NoCpusetController => {
                write!(f, "parent cgroup does not expose cpuset controller")
            }
            SandboxDegraded::SubtreeControlRefused => {
                write!(f, "parent cgroup rejected +cpuset subtree_control")
            }
            SandboxDegraded::PermissionDenied => {
                write!(f, "insufficient permission to create / write cgroup files")
            }
            SandboxDegraded::RootCgroupRefused => {
                write!(
                    f,
                    "caller's cgroup is the cgroup v2 root (0::/); sandbox \
                     creation at the root violates the no-internal-process \
                     rule"
                )
            }
        }
    }
}

impl std::error::Error for SandboxDegraded {}

/// Live child cgroup backing an active [`BuildSandbox::Active`].
/// Split out so the `Drop` cleanup path can pattern-match without
/// needing to destructure the whole variant.
#[derive(Debug)]
pub(crate) struct SandboxInner {
    cg: CgroupManager,
    name: String,
    /// Parent cgroup absolute path — kept for diagnostics in the
    /// Drop warn log. Stored rather than recomputed in case
    /// `/proc/self/cgroup` moves between construction and drop
    /// (which would mean our pid got migrated underneath us; we
    /// still need to point the warning at the original parent).
    parent_cgroup: PathBuf,
    /// PID the sandbox was created for — the caller's own pid at
    /// `try_create`. Drop migrates it back to root before rmdir;
    /// migrating any other pid here would mean the caller leaked
    /// a child into our cgroup, which Drop tolerates by walking
    /// `cgroup.procs` wholesale.
    our_pid: u32,
}

/// RAII guard for a `--llc-cap` build sandbox.
///
/// Returned by [`BuildSandbox::try_create`]. Two shapes:
///
///  - `Active`: the child cgroup exists and the current process is
///    a member. Dropping it migrates-out and removes the child.
///  - `Degraded(SandboxDegraded)`: one of the preconditions
///    failed, no cgroup was created, Drop is a no-op. The caller
///    inspects the variant to decide whether to abort (under
///    `--llc-cap`) or continue without enforcement.
#[derive(Debug)]
pub enum BuildSandbox {
    /// Sandbox is live and enforcing CPU / NUMA binding.
    Active(Box<SandboxInner>),
    /// Pre-condition failed; no cgroup was created. `Drop` is a
    /// no-op. The payload surfaces via `Display` from the calling
    /// site's diagnostic rendering (e.g. `degraded_or_err`'s
    /// `"--llc-cap: {kind}. ..."`); readers never destructure the
    /// variant directly, which the dead-code lint flags as an
    /// unread field. Kept intact so future consumers (telemetry,
    /// test-only inspection) can pattern-match the kind without an
    /// API break.
    #[allow(dead_code)]
    Degraded(SandboxDegraded),
}

impl BuildSandbox {
    /// Create the cgroup, write cpuset.cpus + cpuset.mems + migrate
    /// self into `cgroup.procs`, and return an `Active` sandbox.
    ///
    /// `plan_cpus` is the flattened CPU list from
    /// [`LlcPlan::cpus`](super::host_topology::LlcPlan::cpus);
    /// `plan_mems` is the NUMA node union from
    /// [`LlcPlan::mems`](super::host_topology::LlcPlan::mems).
    ///
    /// Signature takes the two fields destructured rather than
    /// `&LlcPlan` so this module carries no import of the
    /// host_topology types. Keeps the sandbox reusable from any
    /// future caller that has a cpu list + mem set without having
    /// to manufacture an `LlcPlan` (e.g. a future `ktstr cgroup`
    /// sub-command, an external test that exercises sandbox Drop
    /// in isolation).
    ///
    /// `hard_error_on_degrade` controls the `--llc-cap` strict
    /// contract: when `true`, any `SandboxDegraded` outcome is
    /// converted into an `Err(anyhow!(…))` with an actionable
    /// message; when `false`, `Ok(Degraded(…))` is returned and
    /// the caller proceeds without enforcement (the pre-flag
    /// behaviour, preserved so existing no-perf-mode callers don't
    /// regress).
    pub fn try_create(
        plan_cpus: &[usize],
        plan_mems: &BTreeSet<usize>,
        hard_error_on_degrade: bool,
    ) -> Result<Self> {
        // Step A: statfs /sys/fs/cgroup to confirm cgroup v2 is the
        // active mount. The f_type field varies in signed-ness across
        // glibc targets; cast to i64 to compare uniformly.
        match rustix::fs::statfs("/sys/fs/cgroup") {
            Ok(sfs) if (sfs.f_type as i64) == CGROUP2_SUPER_MAGIC => {}
            Ok(_) | Err(_) => {
                return Self::degraded_or_err(
                    SandboxDegraded::NoCgroupV2,
                    hard_error_on_degrade,
                );
            }
        }

        // Step B: locate the caller's parent cgroup via
        // /proc/self/cgroup. cgroup v2 always has a single line
        // `0::/relative/path`.
        let parent_rel = match read_self_cgroup_path() {
            Ok(rel) => rel,
            Err(_) => {
                return Self::degraded_or_err(
                    SandboxDegraded::NoCgroupV2,
                    hard_error_on_degrade,
                );
            }
        };
        // Guard against the root cgroup (0::/). The root has no
        // parent's subtree_control we can add +cpuset to (step D
        // below would need to write to /sys/fs/cgroup's own file,
        // which is special-cased by the kernel), and any child we
        // create there is subject to the no-internal-process rule
        // against the root's own tasks. Refuse early so the
        // operator gets an actionable error before step D-G
        // surfaces an opaque EINVAL / EBUSY.
        if is_root_cgroup(&parent_rel) {
            return Self::degraded_or_err(
                SandboxDegraded::RootCgroupRefused,
                hard_error_on_degrade,
            );
        }
        let parent_abs = Path::new("/sys/fs/cgroup").join(parent_rel.trim_start_matches('/'));

        // Step C: parent must expose cpuset in cgroup.controllers,
        // otherwise no cpuset.* files will exist on our child.
        if !parent_controllers_include(&parent_abs, "cpuset") {
            return Self::degraded_or_err(
                SandboxDegraded::NoCpusetController,
                hard_error_on_degrade,
            );
        }

        let parent_str = match parent_abs.to_str() {
            Some(s) => s,
            None => {
                return Self::degraded_or_err(
                    SandboxDegraded::NoCgroupV2,
                    hard_error_on_degrade,
                );
            }
        };
        let cg = CgroupManager::new(parent_str);

        // Enable +cpuset on the parent's subtree_control so children
        // inherit the controller. Failure here is a degradation
        // signal, not a hard kernel error — many systemd slices
        // already have cpuset enabled.
        //
        // Routed through
        // [`CgroupManager::add_parent_subtree_controller`] rather than
        // a direct `std::fs::write` so the write inherits the module's
        // 2-second timeout (`CGROUP_WRITE_TIMEOUT`) and the errno is
        // surfaced through the shared `anyhow`/`io::Error` chain that
        // [`anyhow_first_io_errno`] walks. `CgroupManager::create_cgroup`'s
        // nested-path enablement short-circuits on flat names (early
        // return when `name.split('/').count() < 2`), and sandbox names
        // are flat (`ktstr-build-{epoch_nanos}-{pid}`), so that helper
        // never touches the parent's subtree_control for our child.
        if let Err(e) = cg.add_parent_subtree_controller("cpuset") {
            // EBUSY when something else holds the subtree;
            // EACCES under a hostile userns. Both are "can't
            // enforce" signals rather than true errors.
            let raw = anyhow_first_io_errno(&e);
            if raw == Some(libc::EACCES) || raw == Some(libc::EPERM) {
                return Self::degraded_or_err(
                    SandboxDegraded::PermissionDenied,
                    hard_error_on_degrade,
                );
            }
            if raw == Some(libc::EBUSY) {
                return Self::degraded_or_err(
                    SandboxDegraded::SubtreeControlRefused,
                    hard_error_on_degrade,
                );
            }
            return Err(e).with_context(|| {
                format!(
                    "write +cpuset to {}/cgroup.subtree_control",
                    parent_abs.display()
                )
            });
        }

        // Step D: orphan sweep — before minting a new cgroup name,
        // remove any prior `ktstr-build-{epoch}-{pid}` whose pid no
        // longer exists OR whose mtime is past ORPHAN_MAX_AGE.
        // Best-effort; errors are logged and iteration continues so
        // one weird leftover doesn't block new sandboxes.
        sweep_orphan_sandboxes(&parent_abs);

        // Step E: mkdir `ktstr-build-{epoch_nanos}-{pid}`. Collision
        // is effectively impossible — two invocations at the same
        // epoch_nanos on the same pid would require process-id
        // rollover inside a single nanosecond.
        let epoch_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let our_pid = std::process::id();
        let name = format!("ktstr-build-{epoch_nanos}-{our_pid}");

        // The cpuset-write + migrate sequence is extracted so
        // rollback-ordering tests can drive it through
        // `&dyn CgroupOps` without a real cgroup v2 mount.
        // Production continues to use the concrete CgroupManager
        // inherently — the trait dispatch adds one vtable lookup
        // per call, dominated by filesystem I/O.
        if let Err(e) = cg.create_cgroup(&name) {
            // mkdir can fail with EACCES/EPERM on a
            // permission-restricted cgroup. Treat as degrade, not
            // a hard error, so the caller's decision path is
            // consistent with the later write failures.
            let raw = anyhow_first_io_errno(&e);
            if raw == Some(libc::EACCES) || raw == Some(libc::EPERM) {
                return Self::degraded_or_err(
                    SandboxDegraded::PermissionDenied,
                    hard_error_on_degrade,
                );
            }
            return Err(e).with_context(|| format!("create {name}"));
        }

        // Step F.1: cpuset.cpus — write, then readback .effective.
        // Disagreement means the kernel narrowed the set (parent
        // restriction or invalid bits). Under `--llc-cap` this is
        // unacceptable; without it, warn and proceed.
        let cpu_set: BTreeSet<usize> = plan_cpus.iter().copied().collect();
        if let Err(e) = cg.set_cpuset(&name, &cpu_set) {
            let _ = cg.remove_cgroup(&name);
            return Err(e).context("write cpuset.cpus");
        }
        let effective_cpus =
            read_cpuset_effective(&parent_abs.join(&name).join("cpuset.cpus.effective"));
        if let Some(eff) = &effective_cpus
            && !cpuset_sets_equal(&cpu_set, eff)
        {
            // Cpuset narrowing is operational telemetry — tracing
            // so sched-team log pipelines pick it up with the
            // structured fields. Cross-node spill is a separate
            // UX-facing warning that stays on eprintln!.
            tracing::warn!(
                cgroup = %name,
                requested = ?cpu_set,
                effective = ?eff,
                "tag=resource_budget.cpuset_cpus_degraded",
            );
            if hard_error_on_degrade {
                let _ = cg.remove_cgroup(&name);
                anyhow::bail!(
                    "--llc-cap: cpuset.cpus narrowed by parent cgroup \
                     (requested {cpu_set:?}, effective {eff:?}). \
                     Run `ktstr locks --json` to inspect peers."
                );
            }
        }

        // Step F.2: cpuset.mems — STRICTLY AFTER F.1. A cgroup with
        // empty cpuset.mems + a task in cgroup.procs would hit
        // SIGKILL on the next allocation per the kernel's
        // `cpuset_update_task_spread` path (already doc'd on
        // CgroupManager::set_cpuset_mems).
        if let Err(e) = cg.set_cpuset_mems(&name, plan_mems) {
            let _ = cg.remove_cgroup(&name);
            return Err(e).context("write cpuset.mems");
        }
        let effective_mems =
            read_cpuset_effective(&parent_abs.join(&name).join("cpuset.mems.effective"));
        if let Some(eff) = &effective_mems
            && !cpuset_sets_equal(plan_mems, eff)
        {
            tracing::warn!(
                cgroup = %name,
                requested = ?plan_mems,
                effective = ?eff,
                "tag=resource_budget.cpuset_mems_degraded",
            );
            if hard_error_on_degrade {
                let _ = cg.remove_cgroup(&name);
                anyhow::bail!(
                    "--llc-cap: cpuset.mems narrowed by parent cgroup \
                     (requested {plan_mems:?}, effective {eff:?}).\n\
                     \n\
                     Remediation:\n\
                     \n\
                       1. The parent cgroup's cpuset.mems is narrower than \
                          the plan's NUMA node set. Run `ktstr locks` to \
                          see which parent is active and `cat \
                          /proc/self/cgroup` for the path, then either \
                          widen that parent's cpuset.mems or move this \
                          process under a wider cgroup (systemd-run \
                          --user --scope -p Delegate=cpuset).\n\
                       2. Drop --llc-cap to build under LLC flock \
                          coordination alone, trading NUMA enforcement \
                          for the noisier fallback path."
                );
            }
        }

        // Step G: migrate self into cgroup.procs AFTER both cpuset
        // writes. Use move_task (the single-pid variant) because
        // we're moving exactly one pid (our own) — move_tasks's
        // ESRCH tolerance and EBUSY retry are overkill here but
        // both are cheap; use move_task for clarity.
        if let Err(e) = cg.move_task(&name, our_pid as libc::pid_t) {
            let _ = cg.remove_cgroup(&name);
            let raw = anyhow_first_io_errno(&e);
            if raw == Some(libc::EACCES) || raw == Some(libc::EPERM) {
                return Self::degraded_or_err(
                    SandboxDegraded::PermissionDenied,
                    hard_error_on_degrade,
                );
            }
            return Err(e).context("migrate self into cgroup.procs");
        }

        Ok(BuildSandbox::Active(Box::new(SandboxInner {
            cg,
            name,
            parent_cgroup: parent_abs,
            our_pid,
        })))
    }

    /// Return `true` when the sandbox successfully installed
    /// kernel-enforced CPU binding. Current production callers use
    /// `hard_error_on_degrade = true` and never observe the
    /// `Degraded` variant — an error propagates instead — so this
    /// accessor is unused in-tree today. Kept for the future
    /// `hard_error_on_degrade = false` path (pre-flag default) and
    /// downstream tooling that wants a Boolean rather than a
    /// `matches!` macro dance.
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        matches!(self, BuildSandbox::Active(_))
    }

    /// Wrap a `SandboxDegraded` into either `Ok(Degraded)` or
    /// `Err(anyhow)` depending on `hard_error_on_degrade`.
    /// Centralized so every degrade site renders the same
    /// operator-facing text.
    fn degraded_or_err(
        kind: SandboxDegraded,
        hard_error_on_degrade: bool,
    ) -> Result<Self> {
        if hard_error_on_degrade {
            Err(anyhow::anyhow!(
                "--llc-cap: {kind}. This host cannot enforce the \
                 resource budget.\n\
                 \n\
                 Remediation (pick one):\n\
                 \n\
                   1. Re-run under a systemd transient scope with a \
                      writable cpuset-capable cgroup:\n\
                      \n\
                        systemd-run --user --scope \\\n\
                            -p 'Delegate=cpuset cpu' \\\n\
                            cargo ktstr kernel build --source <path> --llc-cap N\n\
                      \n\
                   2. Re-run with sudo preserving env so KTSTR_LLC_CAP / \
                      KTSTR_CACHE_DIR / RUST_LOG propagate to the root \
                      invocation:\n\
                      \n\
                        sudo -E cargo ktstr kernel build --source <path> --llc-cap N\n\
                      \n\
                   3. Enable cpuset delegation on the caller's cgroup \
                      by adding `cpuset` to the parent's cgroup.subtree_control \
                      (requires CAP_SYS_ADMIN).\n\
                   \n\
                   4. Drop --llc-cap to build without the cgroup-level \
                      resource contract (falls back to LLC flock \
                      coordination only)."
            ))
        } else {
            Ok(BuildSandbox::Degraded(kind))
        }
    }
}

impl Drop for BuildSandbox {
    fn drop(&mut self) {
        let BuildSandbox::Active(boxed) = self else {
            return;
        };
        let inner: &SandboxInner = boxed;

        // Migrate our pid back to root BEFORE rmdir — kernel
        // returns EBUSY on cgroup_rmdir when cgroup.procs is
        // non-empty. drain_tasks already walks cgroup.procs and
        // re-drives each pid to /sys/fs/cgroup/cgroup.procs, which
        // handles the case of extra children the build left behind
        // as well.
        if let Err(e) = inner.cg.drain_tasks(&inner.name) {
            tracing::warn!(
                cgroup = %inner.name,
                parent = %inner.parent_cgroup.display(),
                err = %e,
                "resource_budget: drain_tasks failed during sandbox drop",
            );
        }

        // Remove with bounded EBUSY retries. CgroupManager::remove_cgroup
        // already sleeps 50ms after draining; our retry loop covers the
        // case where a just-exited child still holds a reference.
        for attempt in 0..RMDIR_EBUSY_RETRIES {
            match inner.cg.remove_cgroup(&inner.name) {
                Ok(()) => return,
                Err(e) => {
                    let raw = anyhow_first_io_errno(&e);
                    if raw == Some(libc::EBUSY) && attempt + 1 < RMDIR_EBUSY_RETRIES {
                        std::thread::sleep(RMDIR_EBUSY_BACKOFF);
                        continue;
                    }
                    // Final failure — orphan left behind. Log
                    // with the conventional tag so `ktstr cleanup`
                    // operators can key on it.
                    tracing::warn!(
                        cgroup = %inner.name,
                        parent = %inner.parent_cgroup.display(),
                        our_pid = inner.our_pid,
                        err = %e,
                        "tag=resource_budget.cgroup_orphan_left",
                    );
                    return;
                }
            }
        }
    }
}

/// Return `true` when `parent_rel` (as emitted by
/// [`read_self_cgroup_path`]) identifies the cgroup v2 root. The
/// root has no parent `subtree_control` to write `+cpuset` into,
/// and writing a task to its own `cgroup.procs` is special-cased
/// by the kernel's no-internal-process rule — neither operation
/// makes sense for a ktstr child sandbox.
///
/// Accepts either `"/"` or `""` (both seen in the wild: modern
/// kernels emit `/` unconditionally, but a trimmed / degenerate
/// read could produce an empty string). Trims leading/trailing
/// whitespace before comparing so a stray `\n` from the
/// /proc/self/cgroup line read doesn't desynchronize the check.
///
/// Extracted from the inline guard in [`BuildSandbox::try_create`]
/// so the root-vs-non-root decision is unit-testable without
/// requiring a cgroup v2 test fixture that actually places the
/// caller at the root. Production caller at
/// [`BuildSandbox::try_create`] Step B.
pub(crate) fn is_root_cgroup(parent_rel: &str) -> bool {
    let trimmed = parent_rel.trim();
    trimmed == "/" || trimmed.is_empty()
}

/// Read the caller's own cgroup v2 path from `/proc/self/cgroup`.
///
/// cgroup v2 always reports a single entry of the form
/// `0::/relative/path`. Returns the `/relative/path` portion
/// verbatim — leading slash preserved, suitable for callers that
/// `trim_start_matches('/').join()` against `/sys/fs/cgroup` to
/// form the absolute parent cgroup path.
fn read_self_cgroup_path() -> Result<String> {
    let text = std::fs::read_to_string("/proc/self/cgroup")
        .context("read /proc/self/cgroup")?;
    for line in text.lines() {
        // cgroup v2 line format: `0::/path`. Hybrid cgroup v1+v2
        // may also list `0::/path`; ignore v1 controller-specific
        // lines (they start with a non-zero hierarchy id).
        if let Some(rest) = line.strip_prefix("0::") {
            return Ok(rest.trim().to_string());
        }
    }
    anyhow::bail!("no cgroup v2 entry (0::) in /proc/self/cgroup")
}

/// Return `true` iff the parent cgroup's `cgroup.controllers` file
/// lists `controller` as available. Absent file or unreadable
/// contents count as "not present" (parent doesn't support the
/// controller).
fn parent_controllers_include(parent_abs: &Path, controller: &str) -> bool {
    let path = parent_abs.join("cgroup.controllers");
    match std::fs::read_to_string(&path) {
        Ok(contents) => contents
            .split_whitespace()
            .any(|c| c == controller),
        Err(_) => false,
    }
}

/// Read a `cpuset.{cpus,mems}.effective` file and parse into a
/// `BTreeSet<usize>`. Returns `None` when the file doesn't exist
/// (kernel versions without the `.effective` view — cgroup v2
/// exposes it since 5.12 but some distros backport, so absence is
/// treated as "cannot verify" rather than hard failure).
fn read_cpuset_effective(path: &Path) -> Option<BTreeSet<usize>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        crate::topology::parse_cpu_list_lenient(text.trim())
            .into_iter()
            .collect(),
    )
}

/// Set-equality check — used for verifying readback of cpuset
/// writes. Returns `true` when `requested` and `effective` are
/// the same set. `effective ⊂ requested` is NOT accepted as OK:
/// a narrower effective set means the kernel narrowed our view
/// (parent cgroup restriction), which IS the degrade case the
/// caller must hear about.
fn cpuset_sets_equal(requested: &BTreeSet<usize>, effective: &BTreeSet<usize>) -> bool {
    requested == effective
}

/// Best-effort removal of crashed prior sandboxes.
///
/// For every `ktstr-build-*` child under `parent_abs`:
///   - If the embedded pid is present on the host (`kill(pid, 0)`
///     succeeds), leave the cgroup — a sibling ktstr may legitimately
///     be running.
///   - If the pid is gone AND the directory mtime is older than
///     `ORPHAN_MAX_AGE`, remove it.
///
/// Removal failures are warn-logged; they don't prevent the caller
/// from minting a new sandbox because the new name embeds
/// `epoch_nanos` and so cannot collide with the orphan.
///
/// Concurrent sweepers (two ktstr processes entering `try_create`
/// at the same time) are safe: if both target the same orphan, the
/// second's `CgroupManager::remove_cgroup` hits ENOENT on
/// `fs::remove_dir` (src/cgroup.rs `remove_cgroup`), which bubbles
/// up as an `anyhow::Error` and is tolerated by the
/// `tracing::warn!` branch below. Neither sweeper poisons the
/// other's forward progress — both still mkdir their own
/// `{epoch_nanos}-{pid}`-suffixed child.
/// Reason to skip sweeping a particular `ktstr-build-*` entry.
/// Surfaces which gate (pid-alive or mtime-young) blocks the sweep
/// so tests can pin gate behavior without calling the full sweep
/// (which would hit `remove_cgroup` and fail on non-cgroup tempdirs).
/// Returns `None` ONLY when both gates are open — live=false AND
/// age >= ORPHAN_MAX_AGE.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SweepSkip {
    /// `kill(pid, 0)` returned success or a non-ESRCH errno — the
    /// pid might still be alive (covers EPERM from cross-user pids
    /// plus normal live pids).
    PidLive,
    /// `ktstr-build-*` entry lacks readable metadata (stat failed).
    MetadataUnreadable,
    /// `entry.metadata().modified()` returned an error.
    MtimeUnreadable,
    /// Directory's mtime is less than `ORPHAN_MAX_AGE` old — recent
    /// enough that it might belong to a legitimate in-progress run
    /// whose pid just happened to be recycled.
    MtimeYoung,
}

/// Decide whether a candidate orphan should be skipped. Returns
/// `None` when both gates are open (dead pid AND old mtime) — the
/// only path that proceeds to `remove_cgroup`.
///
/// `meta` is pre-fetched by the caller (`entry.metadata().ok()`) so
/// this predicate stays filesystem-free beyond a single `kill(2)`
/// probe. `now` is injected for test determinism; production passes
/// `SystemTime::now()`.
pub(crate) fn sweep_skip_reason(
    pid: libc::pid_t,
    meta: Option<std::fs::Metadata>,
    now: std::time::SystemTime,
) -> Option<SweepSkip> {
    // kill(pid, 0) is the standard "does this pid exist" probe.
    // ESRCH means gone; other errors (EPERM, e.g. a root-owned pid
    // an unprivileged test user can't signal) we conservatively
    // treat as "still present" to avoid sweeping another user's
    // in-progress build.
    let kill_rc = unsafe { libc::kill(pid, 0) };
    let live = kill_rc == 0
        || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
    if live {
        return Some(SweepSkip::PidLive);
    }
    let meta = match meta {
        Some(m) => m,
        None => return Some(SweepSkip::MetadataUnreadable),
    };
    let mtime = match meta.modified() {
        Ok(m) => m,
        Err(_) => return Some(SweepSkip::MtimeUnreadable),
    };
    let age = now.duration_since(mtime).unwrap_or(Duration::ZERO);
    if age < ORPHAN_MAX_AGE {
        return Some(SweepSkip::MtimeYoung);
    }
    None
}

fn sweep_orphan_sandboxes(parent_abs: &Path) {
    let entries = match std::fs::read_dir(parent_abs) {
        Ok(e) => e,
        Err(_) => return,
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if !name.starts_with("ktstr-build-") {
            continue;
        }
        // Parse the trailing `-{pid}` segment.
        let pid = match name.rsplit_once('-').and_then(|(_, tail)| tail.parse::<i32>().ok()) {
            Some(p) => p,
            None => continue,
        };
        if sweep_skip_reason(pid, entry.metadata().ok(), now).is_some() {
            continue;
        }
        let cg = match parent_abs.to_str() {
            Some(s) => CgroupManager::new(s),
            None => continue,
        };
        if let Err(e) = cg.remove_cgroup(&name) {
            tracing::warn!(
                cgroup = %name,
                parent = %parent_abs.display(),
                err = %e,
                "resource_budget: orphan sweep remove_cgroup failed",
            );
        }
    }
}

/// Perform the sandbox cpuset-write + migrate sequence on an
/// `&dyn CgroupOps`, with rollback-before-error ordering.
///
/// Extracted from `BuildSandbox::try_create` so unit tests can
/// drive the sequence through a mock CgroupOps and verify that
/// on any failure (F.1 `set_cpuset`, F.2 `set_cpuset_mems`, G
/// `move_task`), `remove_cgroup` runs BEFORE the error returns.
/// Production `try_create` reimplements the same sequence inline
/// (with the extra `.effective` readback gating that this helper
/// intentionally skips — the readback path goes through real
/// filesystem I/O that would require a second mocking layer).
///
/// Returns `Ok(())` on successful F.1 → F.2 → G sequence, or
/// the first error wrapped with context naming the failed step.
/// The caller (and test) observe the rollback by counting
/// `remove_cgroup` invocations on the mock.
#[cfg(test)]
fn apply_sandbox_sequence(
    cg: &dyn crate::cgroup::CgroupOps,
    name: &str,
    cpu_set: &BTreeSet<usize>,
    mem_set: &BTreeSet<usize>,
    pid: libc::pid_t,
) -> Result<()> {
    // F.1: cpuset.cpus — rollback before propagating error.
    if let Err(e) = cg.set_cpuset(name, cpu_set) {
        let _ = cg.remove_cgroup(name);
        return Err(e).context("write cpuset.cpus");
    }
    // F.2: cpuset.mems — STRICTLY AFTER F.1.
    if let Err(e) = cg.set_cpuset_mems(name, mem_set) {
        let _ = cg.remove_cgroup(name);
        return Err(e).context("write cpuset.mems");
    }
    // G: migrate pid after both cpuset writes.
    if let Err(e) = cg.move_task(name, pid) {
        let _ = cg.remove_cgroup(name);
        return Err(e).context("migrate self into cgroup.procs");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_self_cgroup_returns_path() {
        // On any cgroup v2 host, /proc/self/cgroup's 0:: line is
        // always present. A non-cgroup-v2 host would skip this test
        // rather than fail it, but since the whole ktstr test
        // matrix runs on Linux-hosted CI with cgroup v2, we assert
        // outright.
        let rel = read_self_cgroup_path().expect("proc self cgroup readable");
        // Always starts with '/'.
        assert!(rel.starts_with('/'), "cgroup path must be absolute-ish: {rel}");
    }

    #[test]
    fn cpuset_sets_equal_identity() {
        let mut a = BTreeSet::new();
        a.insert(0);
        a.insert(2);
        let mut b = BTreeSet::new();
        b.insert(0);
        b.insert(2);
        assert!(
            cpuset_sets_equal(&a, &b),
            "identity sets must be equal",
        );
    }

    #[test]
    fn cpuset_sets_equal_narrower_effective() {
        // Effective is a proper subset — must be flagged as not
        // equal so the caller sees the narrowing.
        let mut req = BTreeSet::new();
        req.insert(0);
        req.insert(1);
        req.insert(2);
        let mut eff = BTreeSet::new();
        eff.insert(0);
        eff.insert(1);
        assert!(
            !cpuset_sets_equal(&req, &eff),
            "narrower effective must not equal requested",
        );
    }

    #[test]
    fn sandbox_degraded_display_text() {
        // Each variant must render an operator-facing string that
        // keys on its specific failure — they surface in the
        // `--llc-cap` hard-error message and the no-flag warn,
        // and a log-scraper keying on keywords like "cpuset" or
        // "subtree_control" must find the matching variant's
        // text. Non-empty was the prior bar; keyword-contains
        // pins the discriminating content per variant so a
        // future Display refactor that collapsed variants to a
        // generic "sandbox error" wouldn't silently regress.
        let nc = format!("{}", SandboxDegraded::NoCgroupV2);
        assert!(nc.contains("cgroup v2"), "NoCgroupV2: {nc}");
        let ncc = format!("{}", SandboxDegraded::NoCpusetController);
        assert!(ncc.contains("cpuset"), "NoCpusetController: {ncc}");
        let scr = format!("{}", SandboxDegraded::SubtreeControlRefused);
        assert!(
            scr.contains("subtree_control"),
            "SubtreeControlRefused: {scr}",
        );
        let pd = format!("{}", SandboxDegraded::PermissionDenied);
        assert!(pd.contains("permission"), "PermissionDenied: {pd}");
        let rcr = format!("{}", SandboxDegraded::RootCgroupRefused);
        assert!(rcr.contains("root"), "RootCgroupRefused: {rcr}");
    }

    #[test]
    fn parent_controllers_include_missing_file() {
        // Non-existent path → treat as "no controllers available".
        let path = Path::new("/nonexistent/ktstr-controllers-test");
        assert!(
            !parent_controllers_include(path, "cpuset"),
            "nonexistent path must report no controllers",
        );
    }

    #[test]
    fn read_cpuset_effective_missing_file_returns_none() {
        let path = Path::new("/nonexistent/ktstr-effective-test/cpuset.cpus.effective");
        assert!(
            read_cpuset_effective(path).is_none(),
            "nonexistent path must return None",
        );
    }

    /// `sweep_orphan_sandboxes` on a nonexistent parent path is a
    /// no-op. Production DISCOVER happens inside a parent cgroup
    /// the caller is already a member of, so the path always
    /// exists; this test pins the defensive "read_dir failure =
    /// early return" branch so a refactor that accidentally
    /// panics on read_dir failure gets caught.
    #[test]
    fn sweep_orphan_sandboxes_on_nonexistent_parent_is_noop() {
        // Must not panic, must not error (fn returns unit).
        sweep_orphan_sandboxes(Path::new("/nonexistent/ktstr-sweep-test-xyz"));
    }

    /// `sweep_orphan_sandboxes` skips children whose names don't
    /// start with `"ktstr-build-"`. A temp directory containing
    /// an unrelated entry must not disturb it.
    #[test]
    fn sweep_orphan_sandboxes_ignores_non_ktstr_entries() {
        let dir = std::env::temp_dir()
            .join(format!("ktstr-sandbox-sweep-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let unrelated = dir.join("some-other-dir");
        std::fs::create_dir(&unrelated).unwrap();
        sweep_orphan_sandboxes(&dir);
        // The unrelated entry is untouched.
        assert!(
            unrelated.exists(),
            "sweep must not remove non-ktstr entries"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `sweep_orphan_sandboxes` skips children whose trailing
    /// `-{pid}` segment doesn't parse — a filename
    /// `ktstr-build-NOT-A-NUMBER` is left alone rather than
    /// crashing the sweep.
    #[test]
    fn sweep_orphan_sandboxes_skips_malformed_pid_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "ktstr-sandbox-malformed-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let malformed = dir.join("ktstr-build-123-NOTAPID");
        std::fs::create_dir(&malformed).unwrap();
        sweep_orphan_sandboxes(&dir);
        assert!(
            malformed.exists(),
            "sweep must skip entries with unparseable pid suffix"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `sweep_orphan_sandboxes` on an entry named exactly
    /// `"ktstr-build-"` (empty tail after rsplit_once): the
    /// starts_with check passes, rsplit_once('-') returns
    /// Some(("ktstr-build", "")), and `"".parse::<i32>()` fails —
    /// `continue` preserves the entry. Distinct from the 3-segment
    /// malformed case above because this exercises the empty-tail
    /// branch specifically.
    #[test]
    fn sweep_orphan_sandboxes_skips_empty_pid_suffix() {
        let dir = std::env::temp_dir().join(format!(
            "ktstr-sandbox-empty-suffix-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let empty_suffix = dir.join("ktstr-build-");
        std::fs::create_dir(&empty_suffix).unwrap();
        sweep_orphan_sandboxes(&dir);
        assert!(
            empty_suffix.exists(),
            "sweep must skip entries with empty (post-rsplit) pid tail",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `sweep_orphan_sandboxes` on a 2-segment entry with a
    /// directly non-numeric tail (`ktstr-build-NOTAPID`). Distinct
    /// from `_skips_malformed_pid_suffix` above which uses
    /// 3 segments (`ktstr-build-123-NOTAPID`) — this covers the
    /// simpler parse-fails-directly-after-prefix case.
    #[test]
    fn sweep_orphan_sandboxes_skips_direct_non_numeric_pid() {
        let dir = std::env::temp_dir().join(format!(
            "ktstr-sandbox-non-numeric-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let non_numeric = dir.join("ktstr-build-NOTAPID");
        std::fs::create_dir(&non_numeric).unwrap();
        sweep_orphan_sandboxes(&dir);
        assert!(
            non_numeric.exists(),
            "sweep must skip entries whose tail is directly non-numeric",
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `ktstr-build-` filename prefix is load-bearing for
    /// `sweep_orphan_sandboxes` to find its own prior orphans
    /// without touching unrelated cgroup directories. Pin the
    /// exact byte string so a refactor that renames the mkdir
    /// pattern but not the sweep pattern gets caught.
    #[test]
    fn sandbox_filename_prefix_pins_to_ktstr_build() {
        // Mirrors the format! in try_create. If the mkdir format
        // ever changes, this test must update in lockstep with
        // the sweep_orphan_sandboxes prefix check.
        let name = format!("ktstr-build-{}-{}", 1_700_000_000_u64, 12345_u32);
        assert!(
            name.starts_with("ktstr-build-"),
            "sweep keys on this prefix: {name}"
        );
        // Trailing segment must parse as i32 for the sweep's
        // `rsplit_once('-').and_then(parse::<i32>)` path.
        let pid_str = name.rsplit_once('-').map(|(_, t)| t).unwrap();
        assert!(
            pid_str.parse::<i32>().is_ok(),
            "trailing segment must be i32-parseable: {pid_str}"
        );
    }

    /// `BuildSandbox::is_active` returns `true` for `Active(_)`
    /// and `false` for `Degraded(_)`. Covers both enum arms so a
    /// future variant addition that forgets to update `is_active`
    /// gets caught.
    ///
    /// The Active case is constructed with a deliberately-fake
    /// parent path (`/nonexistent/...`). Drop runs on scope exit;
    /// `CgroupManager::remove_cgroup` short-circuits on
    /// `!p.exists()` (src/cgroup.rs remove_cgroup), so teardown is
    /// a safe no-op and leaves no filesystem state.
    #[test]
    fn build_sandbox_is_active_discriminates_variants() {
        // Degraded path: Display-only, no filesystem at all.
        let degraded = BuildSandbox::Degraded(SandboxDegraded::NoCgroupV2);
        assert!(!degraded.is_active(), "Degraded must report !is_active()");

        // Active path: fabricate a SandboxInner with a bogus parent.
        // Both `CgroupManager::drain_tasks` and
        // `CgroupManager::remove_cgroup` early-return on nonexistent
        // paths, so Drop is a no-op when the parent cgroup doesn't
        // exist.
        let fake_parent = std::path::PathBuf::from(
            "/nonexistent/ktstr-build-sandbox-is-active-test",
        );
        let active = BuildSandbox::Active(Box::new(SandboxInner {
            cg: CgroupManager::new(
                fake_parent.to_str().expect("utf-8 test path"),
            ),
            name: "ktstr-build-test-name".to_string(),
            parent_cgroup: fake_parent,
            our_pid: 1,
        }));
        assert!(active.is_active(), "Active must report is_active()");
        // active drops here — safe no-op per doc above.
    }

    /// `BuildSandbox::try_create` on a host that lacks cgroup v2 at
    /// `/sys/fs/cgroup` must surface `NoCgroupV2` when
    /// `hard_error_on_degrade=false`, and a hard error naming the
    /// "--llc-cap" contract when `hard_error_on_degrade=true`.
    /// Exercises the step-A statfs guard.
    ///
    /// Can't reliably fake the `statfs` magic test without mount
    /// privileges, so this test runs only on hosts where the
    /// caller's parent cgroup is BELOW the root but we can still
    /// detect that the sandbox code path takes one of the guard
    /// branches. The production statfs succeeds on every realistic
    /// test host (CI and developer machines both mount cgroup2), so
    /// what this test actually verifies is that a non-panicking
    /// return path exists — either a real Active sandbox or a
    /// Degraded variant flagged via is_active().
    #[test]
    fn build_sandbox_try_create_returns_without_panic() {
        // Use an empty plan + empty mems — no CPU binding at all.
        // On a functional cgroup v2 host, this produces an Active
        // sandbox whose Drop rmdir's cleanly.
        let cpus: Vec<usize> = Vec::new();
        let mems: BTreeSet<usize> = BTreeSet::new();

        // hard_error_on_degrade=false: caller that would otherwise
        // not have set --llc-cap. Any degrade variant surfaces as
        // Ok(Degraded(_)).
        let result = BuildSandbox::try_create(&cpus, &mems, false);
        match &result {
            Ok(BuildSandbox::Active(_)) => {
                // Happy path on a fully-configured cgroup v2 host.
            }
            Ok(BuildSandbox::Degraded(kind)) => {
                // Partial-configured host (no cpuset controller,
                // subtree_control refused, permission denied, root
                // cgroup). Assert Display renders non-empty so the
                // degraded_or_err message path won't print blanks.
                assert!(
                    !format!("{kind}").is_empty(),
                    "SandboxDegraded::Display must be non-empty: {kind:?}",
                );
            }
            Err(e) => {
                // try_create only bubbles non-degrade errors (e.g.
                // unexpected kernel rejection, filesystem I/O). On
                // an unexpected error, surface it.
                panic!("try_create unexpected hard error: {e:#}");
            }
        }
        // result drops here — Active variants clean up their cgroup
        // via Drop; Degraded is a no-op.
    }

    /// `BuildSandbox::try_create` with `hard_error_on_degrade=true`
    /// must convert any `Degraded` outcome into an `Err(anyhow)`
    /// with the "--llc-cap" and "Remediation" substrings. On a
    /// fully-configured host that produces an `Active` sandbox, the
    /// test accepts the Ok path and does not assert error text.
    #[test]
    fn build_sandbox_try_create_hard_error_converts_degrade() {
        let cpus: Vec<usize> = Vec::new();
        let mems: BTreeSet<usize> = BTreeSet::new();

        let result = BuildSandbox::try_create(&cpus, &mems, true);
        match &result {
            Ok(BuildSandbox::Active(_)) => {
                // Fully-configured host — acceptable, no assertion.
            }
            Ok(BuildSandbox::Degraded(kind)) => {
                panic!(
                    "hard_error_on_degrade=true must NOT return Degraded; \
                     got {kind:?}"
                );
            }
            Err(e) => {
                // Degraded was converted to a hard error per the
                // `--llc-cap` contract.
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("--llc-cap"),
                    "hard error must name the contract flag: {msg}",
                );
                assert!(
                    msg.contains("Remediation"),
                    "hard error must include remediation block: {msg}",
                );
            }
        }
    }

    /// Drop on an Active sandbox with a non-existent parent cgroup
    /// path must NOT panic. The production Drop calls drain_tasks
    /// + remove_cgroup, both of which short-circuit on
    /// `!parent.exists()` (src/cgroup.rs), so the teardown is a
    /// safe no-op when we hand it a bogus path.
    ///
    /// Distinct from `build_sandbox_is_active_discriminates_variants`:
    /// that test asserts the `is_active()` accessor; this one pins
    /// the Drop invariant that a bogus parent doesn't panic.
    #[test]
    fn build_sandbox_drop_on_nonexistent_parent_does_not_panic() {
        let fake_parent = std::path::PathBuf::from(
            "/nonexistent/ktstr-build-drop-test-xyz",
        );
        let sandbox = BuildSandbox::Active(Box::new(SandboxInner {
            cg: CgroupManager::new(
                fake_parent.to_str().expect("utf-8 test path"),
            ),
            name: "ktstr-build-drop-test-name".to_string(),
            parent_cgroup: fake_parent,
            our_pid: 1,
        }));
        // Drop runs on scope exit and must not panic.
        drop(sandbox);
    }

    /// Drop on a Degraded sandbox is a no-op — no cgroup was ever
    /// created, so drain_tasks and remove_cgroup must not be called.
    /// Pins the Drop arm that early-returns on the Degraded variant.
    #[test]
    fn build_sandbox_drop_on_degraded_is_noop() {
        let sandbox = BuildSandbox::Degraded(SandboxDegraded::NoCgroupV2);
        drop(sandbox);
    }

    /// Rollback path test: every SandboxDegraded variant must have
    /// a Display impl that produces non-empty text. The
    /// `degraded_or_err` remediation block embeds `{kind}` via its
    /// Display — an empty Display would break the error message in
    /// production. Covers all 5 variants to catch a future addition
    /// that forgot to update the Display match.
    ///
    /// Distinct from `sandbox_degraded_display_text` (which only
    /// asserts non-empty) in that this test enumerates every
    /// variant explicitly — a new variant added after refactoring
    /// must be added here in lockstep, catching drift between the
    /// enum and its Display impl.
    #[test]
    fn sandbox_degraded_all_variants_display_non_empty() {
        let variants = [
            SandboxDegraded::NoCgroupV2,
            SandboxDegraded::NoCpusetController,
            SandboxDegraded::SubtreeControlRefused,
            SandboxDegraded::PermissionDenied,
            SandboxDegraded::RootCgroupRefused,
        ];
        for v in variants {
            let text = format!("{v}");
            assert!(
                !text.is_empty(),
                "SandboxDegraded::{v:?} must have non-empty Display",
            );
        }
    }

    /// `is_root_cgroup` treats `"/"` and `""` as root, and a
    /// non-empty non-slash path as non-root. Trimming covers a
    /// stray `\n` in the /proc/self/cgroup read.
    ///
    /// Seam test for the `try_create` step-B guard — lets us
    /// regression-test the RootCgroupRefused path without
    /// requiring a cgroup v2 fixture that places the caller at
    /// the root.
    #[test]
    fn is_root_cgroup_handles_slash_empty_and_whitespace() {
        assert!(is_root_cgroup("/"), "literal / is the root");
        assert!(is_root_cgroup(""), "empty string is treated as root");
        // Whitespace-only should also count — trim() collapses it
        // to empty, which hits the empty-is-root branch.
        assert!(is_root_cgroup("   "), "whitespace-only is treated as root");
        // Trailing newline from a /proc read: trim() strips it.
        assert!(is_root_cgroup("/\n"), "slash + newline trims to root");
        // Non-root paths must NOT match.
        assert!(!is_root_cgroup("/user.slice"), "/user.slice is not root");
        assert!(
            !is_root_cgroup("/user.slice/session-1.scope"),
            "nested slice is not root",
        );
        // A path that starts with `/` but isn't `/` alone.
        assert!(!is_root_cgroup("/a"), "/a is not root");
    }

    // ---------------------------------------------------------------
    // `apply_sandbox_sequence` rollback-ordering tests
    // ---------------------------------------------------------------
    //
    // Hand-rolled CgroupOps mock (rather than mockall) so the dep
    // graph stays flat and the test matches existing ktstr test-
    // double patterns. The mock records call order and injects
    // errors at configurable steps; tests assert the invariant that
    // on any failure, `remove_cgroup` runs BEFORE the error returns.

    use std::sync::Mutex;

    /// Hand-rolled CgroupOps mock that records every call in order
    /// and lets tests inject errors at specific steps.
    #[derive(Default)]
    struct MockCgroupOps {
        calls: Mutex<Vec<String>>,
        set_cpuset_err: Mutex<Option<&'static str>>,
        set_cpuset_mems_err: Mutex<Option<&'static str>>,
        move_task_err: Mutex<Option<&'static str>>,
    }

    impl MockCgroupOps {
        fn record(&self, call: &str) {
            self.calls.lock().unwrap().push(call.to_string());
        }

        fn calls_snapshot(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl crate::cgroup::CgroupOps for MockCgroupOps {
        fn parent_path(&self) -> &Path {
            Path::new("/mock")
        }
        fn setup(&self, _: bool) -> Result<()> {
            self.record("setup");
            Ok(())
        }
        fn create_cgroup(&self, name: &str) -> Result<()> {
            self.record(&format!("create_cgroup({name})"));
            Ok(())
        }
        fn remove_cgroup(&self, name: &str) -> Result<()> {
            self.record(&format!("remove_cgroup({name})"));
            Ok(())
        }
        fn set_cpuset(&self, name: &str, _: &BTreeSet<usize>) -> Result<()> {
            self.record(&format!("set_cpuset({name})"));
            if let Some(msg) = *self.set_cpuset_err.lock().unwrap() {
                anyhow::bail!("{msg}");
            }
            Ok(())
        }
        fn clear_cpuset(&self, name: &str) -> Result<()> {
            self.record(&format!("clear_cpuset({name})"));
            Ok(())
        }
        fn set_cpuset_mems(&self, name: &str, _: &BTreeSet<usize>) -> Result<()> {
            self.record(&format!("set_cpuset_mems({name})"));
            if let Some(msg) = *self.set_cpuset_mems_err.lock().unwrap() {
                anyhow::bail!("{msg}");
            }
            Ok(())
        }
        fn clear_cpuset_mems(&self, name: &str) -> Result<()> {
            self.record(&format!("clear_cpuset_mems({name})"));
            Ok(())
        }
        fn move_task(&self, name: &str, pid: libc::pid_t) -> Result<()> {
            self.record(&format!("move_task({name},{pid})"));
            if let Some(msg) = *self.move_task_err.lock().unwrap() {
                anyhow::bail!("{msg}");
            }
            Ok(())
        }
        fn move_tasks(&self, name: &str, _: &[libc::pid_t]) -> Result<()> {
            self.record(&format!("move_tasks({name})"));
            Ok(())
        }
        fn clear_subtree_control(&self, name: &str) -> Result<()> {
            self.record(&format!("clear_subtree_control({name})"));
            Ok(())
        }
        fn drain_tasks(&self, name: &str) -> Result<()> {
            self.record(&format!("drain_tasks({name})"));
            Ok(())
        }
        fn cleanup_all(&self) -> Result<()> {
            self.record("cleanup_all");
            Ok(())
        }
    }

    fn test_sets() -> (BTreeSet<usize>, BTreeSet<usize>) {
        (
            [0usize, 1, 2].into_iter().collect(),
            [0usize].into_iter().collect(),
        )
    }

    /// Rollback on F.1 failure (`set_cpuset`): sequence must be
    /// `set_cpuset` (fails) → `remove_cgroup` → error return. No
    /// `set_cpuset_mems` or `move_task` should fire.
    #[test]
    fn apply_sandbox_sequence_rolls_back_on_set_cpuset_failure() {
        let mock = MockCgroupOps::default();
        *mock.set_cpuset_err.lock().unwrap() = Some("EINVAL");
        let (cpus, mems) = test_sets();
        let err = apply_sandbox_sequence(&mock, "sbx", &cpus, &mems, 1)
            .expect_err("F.1 failure must propagate");
        let msg = format!("{err:#}");
        assert!(msg.contains("cpuset.cpus"), "err must name the step: {msg}");
        let calls = mock.calls_snapshot();
        assert_eq!(
            calls,
            vec!["set_cpuset(sbx)".to_string(), "remove_cgroup(sbx)".to_string()],
            "exact call order: set_cpuset THEN remove_cgroup, with NO \
             set_cpuset_mems or move_task",
        );
    }

    /// Rollback on F.2 failure (`set_cpuset_mems`): sequence must
    /// be `set_cpuset` (ok) → `set_cpuset_mems` (fails) →
    /// `remove_cgroup` → error return. No `move_task` fires. This
    /// pins the "rollback happens AFTER the failed step, not
    /// before" invariant and the "F.2 runs strictly after F.1".
    #[test]
    fn apply_sandbox_sequence_rolls_back_on_set_cpuset_mems_failure() {
        let mock = MockCgroupOps::default();
        *mock.set_cpuset_mems_err.lock().unwrap() = Some("EINVAL");
        let (cpus, mems) = test_sets();
        let err = apply_sandbox_sequence(&mock, "sbx", &cpus, &mems, 1)
            .expect_err("F.2 failure must propagate");
        let msg = format!("{err:#}");
        assert!(msg.contains("cpuset.mems"), "err must name the step: {msg}");
        let calls = mock.calls_snapshot();
        assert_eq!(
            calls,
            vec![
                "set_cpuset(sbx)".to_string(),
                "set_cpuset_mems(sbx)".to_string(),
                "remove_cgroup(sbx)".to_string(),
            ],
            "exact order: F.1 ok, F.2 fails, remove_cgroup, with NO move_task",
        );
    }

    /// Rollback on G failure (`move_task`): sequence must be
    /// F.1 ok → F.2 ok → `move_task` (fails) → `remove_cgroup` →
    /// error return. Critical for the "no leaked cgroup on the
    /// most common failure" invariant — the task migration step
    /// is the last and most externally-driven source of failure.
    #[test]
    fn apply_sandbox_sequence_rolls_back_on_move_task_failure() {
        let mock = MockCgroupOps::default();
        *mock.move_task_err.lock().unwrap() = Some("ESRCH");
        let (cpus, mems) = test_sets();
        let err = apply_sandbox_sequence(&mock, "sbx", &cpus, &mems, 42)
            .expect_err("G failure must propagate");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("migrate self into cgroup.procs"),
            "err must name the step: {msg}",
        );
        let calls = mock.calls_snapshot();
        assert_eq!(
            calls,
            vec![
                "set_cpuset(sbx)".to_string(),
                "set_cpuset_mems(sbx)".to_string(),
                "move_task(sbx,42)".to_string(),
                "remove_cgroup(sbx)".to_string(),
            ],
            "exact order: F.1, F.2, G fails, remove_cgroup",
        );
    }

    // ---------------------------------------------------------------
    // sweep_skip_reason gate tests — pid-alive and mtime-age gates
    // ---------------------------------------------------------------
    //
    // Each test constructs a fake `ktstr-build-*` tempdir with
    // controlled mtime, then invokes `sweep_skip_reason` directly
    // to assert which gate (if any) blocks the sweep. Avoids the
    // full `sweep_orphan_sandboxes` call because remove_cgroup
    // would fail on a non-cgroup tempdir (no cgroup.procs file)
    // — the gate predicate is the invariant we need to pin.

    /// Helper: build a temp directory with the given mtime and
    /// return its Metadata. Used by the gate tests to feed
    /// `sweep_skip_reason` without calling `sweep_orphan_sandboxes`
    /// itself.
    fn fake_entry_with_mtime(age: Duration) -> (tempfile::TempDir, std::fs::Metadata) {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let child = tmp.path().join("entry");
        std::fs::create_dir(&child).expect("mkdir");
        let target_mtime = std::time::SystemTime::now()
            .checked_sub(age)
            .expect("time arithmetic");
        // Shift mtime via `filetime` — actually, stick to libc
        // `utime(2)` to avoid a new dep. Convert SystemTime → secs
        // since epoch.
        let secs = target_mtime
            .duration_since(std::time::UNIX_EPOCH)
            .expect("post-epoch")
            .as_secs();
        let buf = libc::utimbuf {
            actime: secs as libc::time_t,
            modtime: secs as libc::time_t,
        };
        let cpath = std::ffi::CString::new(child.to_str().expect("utf8"))
            .expect("nul-free");
        // SAFETY: `cpath` is a valid nul-terminated C string built
        // from a just-created directory path; `&buf` is an aligned
        // struct living on this frame. utime(2) is safe to call
        // with these arguments per its kernel contract.
        let rc = unsafe { libc::utime(cpath.as_ptr(), &buf) };
        assert_eq!(rc, 0, "utime must succeed on our tempdir");
        let meta = std::fs::metadata(&child).expect("stat");
        (tmp, meta)
    }

    /// Gate 1: live pid + old mtime → skip (live-pid gate wins).
    /// Our own process id is always alive; mtime is set to 48 h
    /// (well past ORPHAN_MAX_AGE of 24 h). Without the pid gate,
    /// the mtime gate would allow the sweep.
    #[test]
    fn sweep_skip_reason_live_pid_with_old_mtime_blocks_on_pid() {
        let (_tmp, meta) = fake_entry_with_mtime(Duration::from_secs(48 * 60 * 60));
        let our_pid = std::process::id() as libc::pid_t;
        let now = std::time::SystemTime::now();
        assert_eq!(
            sweep_skip_reason(our_pid, Some(meta), now),
            Some(SweepSkip::PidLive),
            "live pid must block sweep even when mtime is old",
        );
    }

    /// Gate 2: dead pid + fresh mtime → skip (mtime gate wins).
    /// Fork a child, wait for its exit, use the (now-dead) pid.
    /// Mtime is the tempdir's natural creation time (fresh).
    /// Without the mtime gate, the dead-pid check would sweep.
    #[test]
    fn sweep_skip_reason_dead_pid_with_fresh_mtime_blocks_on_mtime() {
        // Fork a child that immediately exits. Wait for its
        // reaping so kill(pid, 0) returns ESRCH.
        // SAFETY: we fork and immediately _exit(0) in the child;
        // no Rust destructors between fork and _exit; parent
        // waits for the child to reap the zombie before using
        // the pid.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork must succeed");
        if pid == 0 {
            // Child: exit immediately without running destructors.
            unsafe {
                libc::_exit(0);
            }
        }
        // Parent: wait for the child to be fully reaped.
        let mut status: libc::c_int = 0;
        let rc = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(rc, pid, "waitpid must reap our child");
        // Now kill(pid, 0) will return ESRCH.
        let (_tmp, meta) = fake_entry_with_mtime(Duration::from_secs(60));
        let now = std::time::SystemTime::now();
        assert_eq!(
            sweep_skip_reason(pid, Some(meta), now),
            Some(SweepSkip::MtimeYoung),
            "dead pid with fresh mtime must block on the mtime gate",
        );
    }

    /// Gate 3: dead pid + old mtime → NO skip (both gates agree).
    /// This is the ONLY configuration the sweep actually acts on.
    #[test]
    fn sweep_skip_reason_dead_pid_with_old_mtime_sweeps() {
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork must succeed");
        if pid == 0 {
            unsafe {
                libc::_exit(0);
            }
        }
        let mut status: libc::c_int = 0;
        let rc = unsafe { libc::waitpid(pid, &mut status, 0) };
        assert_eq!(rc, pid, "waitpid must reap our child");
        let (_tmp, meta) = fake_entry_with_mtime(Duration::from_secs(48 * 60 * 60));
        let now = std::time::SystemTime::now();
        assert_eq!(
            sweep_skip_reason(pid, Some(meta), now),
            None,
            "dead pid + old mtime is the only config that sweeps",
        );
    }

    /// Gate 4: EPERM pid → treat as live, skip on pid-live gate.
    /// kill(1, 0) returns EPERM for unprivileged users because pid
    /// 1 (init) is owned by root. The kill_rc != ESRCH branch
    /// treats EPERM as "still present" to avoid sweeping another
    /// user's in-progress build.
    ///
    /// Skips when the test runs as root (kill(1, 0) succeeds → 0
    /// → live → correct outcome, but the test's ability to
    /// distinguish EPERM from 0 is lost). The assertion is still
    /// valid (live=true either way), so we don't skip — we just
    /// note both paths reach the same SweepSkip::PidLive verdict.
    #[test]
    fn sweep_skip_reason_eperm_pid_treated_as_live() {
        // pid 1 is `init` (systemd on most Linux distros, or a
        // container's PID 1 under nspawn/Docker). On unprivileged
        // test users, kill(1, 0) returns -1 with errno=EPERM. As
        // root, it returns 0. In EITHER case, sweep_skip_reason
        // must return PidLive — the "not ESRCH" branch catches
        // EPERM, and the "kill_rc == 0" branch catches root.
        let (_tmp, meta) = fake_entry_with_mtime(Duration::from_secs(48 * 60 * 60));
        let now = std::time::SystemTime::now();
        assert_eq!(
            sweep_skip_reason(1, Some(meta), now),
            Some(SweepSkip::PidLive),
            "pid 1 must be treated as live (EPERM or success, not ESRCH)",
        );
    }

    /// Happy path: no rollback. All three steps complete, no
    /// `remove_cgroup` is called. Pins that `apply_sandbox_sequence`
    /// doesn't accidentally clean up on success — the sandbox
    /// lifecycle owns the cleanup via Drop, the sequence is only
    /// supposed to roll back on FAILURE.
    #[test]
    fn apply_sandbox_sequence_success_does_not_call_remove_cgroup() {
        let mock = MockCgroupOps::default();
        let (cpus, mems) = test_sets();
        apply_sandbox_sequence(&mock, "sbx", &cpus, &mems, 42)
            .expect("all steps ok must succeed");
        let calls = mock.calls_snapshot();
        assert_eq!(
            calls,
            vec![
                "set_cpuset(sbx)".to_string(),
                "set_cpuset_mems(sbx)".to_string(),
                "move_task(sbx,42)".to_string(),
            ],
            "exact F.1→F.2→G order, NO remove_cgroup on success",
        );
        assert!(
            !calls.iter().any(|c| c.starts_with("remove_cgroup")),
            "remove_cgroup must NOT fire on success: {calls:?}",
        );
    }
}
