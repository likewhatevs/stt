//! VM-backed integration test for multi-level
//! `--cell-parent-cgroup` paths.
//!
//! Boots a guest with a 3-level cgroup_parent
//! (`/ktstr-multi-l1/l2/l3`) and reads back every ancestor's
//! `cgroup.subtree_control` from inside the guest to verify the
//! root-to-leaf-parent walk in
//! [`crate::vmm::rust_init::enable_subtree_controllers_to`]
//! enabled `+cpuset +cpu` at every intermediate level.
//!
//! Without that walk the kernel rejects each child's
//! `subtree_control` write with `-ENOENT`
//! (`cgroup_subtree_control_write` /
//! `cgroup_control` in `kernel/cgroup/cgroup.c`); with the walk
//! every level above the leaf advertises both controllers so the
//! scheduler's later attach against the leaf cgroup succeeds.
//!
//! Pinning the multi-level case complements the single-level
//! coverage in the wider scheduler-attach tests: a single-level
//! parent (`/test`) only needs the cgroup root's
//! `subtree_control` populated and would pass even if the walk
//! degraded back to "immediate parent only" behaviour. The
//! 3-level case is the smallest topology that distinguishes
//! "walks the full ancestor chain" from "only writes the
//! immediate parent."

use anyhow::Result;
use ktstr::assert::{AssertDetail, AssertResult, DetailKind};
use ktstr::ktstr_test;
use ktstr::scenario::Ctx;
use ktstr::test_support::{Scheduler, SchedulerSpec};
use std::fs;

/// Scheduler with a 3-level cgroup_parent. The leaf
/// `/sys/fs/cgroup/ktstr-multi-l1/l2/l3` exercises every
/// intermediate level — each must have `+cpuset +cpu` enabled
/// in its own `cgroup.subtree_control` before its child can
/// inherit those controllers.
const MULTI_LEVEL_SCHED: Scheduler = Scheduler::new("ktstr_sched")
    .binary(SchedulerSpec::Discover("scx-ktstr"))
    .cgroup_parent("/ktstr-multi-l1/l2/l3");

/// Verify every intermediate cgroup directory exists and its
/// `cgroup.subtree_control` advertises both controllers
/// (`cpuset` and `cpu`). The check covers four levels:
///
///   * `/sys/fs/cgroup` (cgroup root) — must export both
///     controllers to its children, otherwise the
///     `ktstr-multi-l1` write would be rejected.
///   * `/sys/fs/cgroup/ktstr-multi-l1` — first level under
///     root.
///   * `/sys/fs/cgroup/ktstr-multi-l1/l2` — second level.
///   * `/sys/fs/cgroup/ktstr-multi-l1/l2/l3` — leaf. The leaf's
///     own `subtree_control` is intentionally NOT checked: the
///     walk in `enable_subtree_controllers_to` skips the leaf
///     because enabling controllers IN a cgroup means they are
///     visible inside its CHILDREN, and the scheduler attaches
///     tasks AT the leaf. Only the leaf's existence is asserted.
///
/// Body returns `pass()` only when every level's write
/// succeeded; any missing directory or any
/// `subtree_control` line that does not contain both
/// `cpuset` AND `cpu` produces a `fail` carrying the offending
/// path and observed contents so a regression has actionable
/// diagnostic context without a host-side log dive.
#[ktstr_test(
    scheduler = MULTI_LEVEL_SCHED,
    llcs = 1,
    cores = 1,
    threads = 1,
    memory_mb = 256,
)]
fn cgroup_parent_three_levels_writes_subtree_control_at_every_ancestor(
    _ctx: &Ctx,
) -> Result<AssertResult> {
    // Levels checked for `+cpuset +cpu` in `cgroup.subtree_control`.
    // Ordered root-first, mirroring the write order in
    // `enable_subtree_controllers_to`. The leaf is NOT included
    // (see body docstring): the walk skips it deliberately, and
    // its own subtree_control may be empty depending on the
    // scheduler's later cgroup operations.
    let intermediate_levels: &[&str] = &[
        "/sys/fs/cgroup",
        "/sys/fs/cgroup/ktstr-multi-l1",
        "/sys/fs/cgroup/ktstr-multi-l1/l2",
    ];
    for level in intermediate_levels {
        let control_path = format!("{level}/cgroup.subtree_control");
        let contents = match fs::read_to_string(&control_path) {
            Ok(s) => s,
            Err(e) => {
                return Ok(AssertResult::fail(AssertDetail::new(
                    DetailKind::Other,
                    format!(
                        "read {control_path}: {e}. \
                         enable_subtree_controllers_to should have \
                         created the cgroup directory and populated \
                         its subtree_control before the test body \
                         dispatched."
                    ),
                )));
            }
        };
        // The kernel renders subtree_control as a space-separated
        // list of controller names (e.g. "cpuset cpu memory"). The
        // exact set may include controllers other than the two we
        // wrote, depending on what controllers ancestors above
        // had already enabled. The contract is "cpuset AND cpu
        // present", not "exactly these two".
        let tokens: std::collections::HashSet<&str> = contents.split_whitespace().collect();
        if !tokens.contains("cpuset") || !tokens.contains("cpu") {
            return Ok(AssertResult::fail(AssertDetail::new(
                DetailKind::Other,
                format!(
                    "{control_path} = {contents:?}: missing one of \
                     cpuset/cpu. enable_subtree_controllers_to writes \
                     `+cpuset +cpu` to every ancestor's \
                     cgroup.subtree_control; an absent controller \
                     here means the ancestor walk skipped this \
                     level."
                ),
            )));
        }
    }
    // Confirm the leaf directory itself exists. mkdir_p creates
    // it via repeated `mkdir(2)` calls; a missing directory here
    // proves create_cgroup_parent_from_sched_args never reached
    // its mkdir_p arm — i.e. the `--cell-parent-cgroup` argument
    // wasn't parsed out of /sched_args.
    let leaf = "/sys/fs/cgroup/ktstr-multi-l1/l2/l3";
    if !std::path::Path::new(leaf).is_dir() {
        return Ok(AssertResult::fail(AssertDetail::new(
            DetailKind::Other,
            format!(
                "leaf cgroup {leaf} does not exist. \
                 create_cgroup_parent_from_sched_args should have \
                 parsed `--cell-parent-cgroup /ktstr-multi-l1/l2/l3` \
                 from /sched_args and called mkdir_p before the \
                 scheduler started."
            ),
        )));
    }
    Ok(AssertResult::pass())
}
