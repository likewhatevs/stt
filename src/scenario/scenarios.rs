//! Curated canned scenarios for common scheduler test patterns.
//!
//! Each function takes a [`Ctx`] and returns `Result<AssertResult>`.
//! These are thin wrappers over existing scenario implementations,
//! providing better names in a single discoverable namespace.
//!
//! # Categories
//!
//! - **Basic**: steady-state cgroups with no dynamic ops.
//! - **Cpuset**: cpuset assignment and mid-run mutation.
//! - **Dynamic**: cgroup add/remove during a running workload.
//! - **Affinity**: per-worker CPU affinity patterns.
//! - **Stress**: host/cgroup contention and mixed workload types.
//! - **Nested**: workers in nested sub-cgroups.
//!
//! # Example
//!
//! ```rust,no_run
//! use ktstr::prelude::*;
//!
//! #[ktstr_test(llcs = 2, cores = 4, threads = 1)]
//! fn test_steady(ctx: &Ctx) -> Result<AssertResult> {
//!     scenarios::steady(ctx)
//! }
//! ```

use anyhow::Result;

use crate::assert::AssertResult;
use crate::workload::WorkType;

use super::Ctx;
use super::ops::{CgroupDef, CpusetSpec, execute_defs};

// ---------------------------------------------------------------------------
// Basic
// ---------------------------------------------------------------------------

/// Two cgroups, no cpusets, equal CPU-spin load.
///
/// Simplest possible scenario: tests that the scheduler can handle
/// two cgroups running simultaneously without starvation.
pub fn steady(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(
        ctx,
        vec![CgroupDef::named("cg_0"), CgroupDef::named("cg_1")],
    )
}

/// Two cgroups with LLC-aligned cpusets.
///
/// Each cgroup gets CPUs from a different LLC. Tests scheduler
/// behavior when cgroups are partitioned along cache boundaries.
/// Skips on single-LLC topologies.
pub fn steady_llc(ctx: &Ctx) -> Result<AssertResult> {
    if ctx.topo.num_llcs() < 2 {
        return Ok(AssertResult::skip("skipped: need >=2 LLCs"));
    }
    execute_defs(
        ctx,
        vec![
            CgroupDef::named("cg_0").with_cpuset(CpusetSpec::llc(0)),
            CgroupDef::named("cg_1").with_cpuset(CpusetSpec::llc(1)),
        ],
    )
}

/// Two cgroups with 32 mixed workers each (oversubscribed).
///
/// Worker count far exceeds CPU count, testing dispatch under
/// heavy oversubscription with mixed workload types.
pub fn oversubscribed(ctx: &Ctx) -> Result<AssertResult> {
    execute_defs(
        ctx,
        vec![
            CgroupDef::named("cg_0")
                .workers(32)
                .work_type(WorkType::Mixed),
            CgroupDef::named("cg_1")
                .workers(32)
                .work_type(WorkType::Mixed),
        ],
    )
}

// ---------------------------------------------------------------------------
// Cpuset — delegates to super::cpuset
// ---------------------------------------------------------------------------

/// Two cgroups start without cpusets, then get disjoint cpusets mid-run.
///
/// Tests the scheduler's response to cpuset assignment on running
/// cgroups. Workers must migrate to their assigned CPUs.
pub fn cpuset_apply(ctx: &Ctx) -> Result<AssertResult> {
    super::cpuset::custom_cgroup_cpuset_apply_midrun(ctx)
}

/// Two cgroups start with disjoint cpusets, then cpusets are cleared mid-run.
///
/// Tests the scheduler's response to cpuset removal. Workers that
/// were confined to a subset of CPUs become free to run anywhere.
pub fn cpuset_clear(ctx: &Ctx) -> Result<AssertResult> {
    super::cpuset::custom_cgroup_cpuset_clear_midrun(ctx)
}

/// Two cgroups with cpusets that shrink then grow.
///
/// Three-phase scenario: even split, then shrink cg_0 / grow cg_1,
/// then reverse. Tests scheduler adaptation to cpuset resizing.
pub fn cpuset_resize(ctx: &Ctx) -> Result<AssertResult> {
    super::cpuset::custom_cgroup_cpuset_resize(ctx)
}

// ---------------------------------------------------------------------------
// Dynamic — delegates to super::dynamic
// ---------------------------------------------------------------------------

/// Two cgroups initially, then one or two more added mid-run.
///
/// Tests the scheduler's response to new cgroups appearing while
/// workers are already running.
pub fn cgroup_add(ctx: &Ctx) -> Result<AssertResult> {
    super::dynamic::custom_cgroup_add_midrun(ctx)
}

/// Four cgroups initially, then the second half removed mid-run.
///
/// Tests the scheduler's response to cgroup removal while workers
/// in surviving cgroups continue running.
pub fn cgroup_remove(ctx: &Ctx) -> Result<AssertResult> {
    super::dynamic::custom_cgroup_remove_midrun(ctx)
}

// ---------------------------------------------------------------------------
// Affinity — delegates to super::affinity
// ---------------------------------------------------------------------------

/// Two cgroups with worker affinities randomized mid-run.
///
/// Workers start with no affinity, then get random CPU subsets
/// applied four times during the run.
pub fn affinity_change(ctx: &Ctx) -> Result<AssertResult> {
    super::affinity::custom_cgroup_affinity_change(ctx)
}

/// Two cgroups with workers pinned to a 2-CPU subset.
///
/// All workers in both cgroups share the same narrow affinity mask.
/// Tests scheduler behavior under heavy contention on few CPUs.
pub fn affinity_pinned(ctx: &Ctx) -> Result<AssertResult> {
    super::affinity::custom_cgroup_multicpu_pin(ctx)
}

// ---------------------------------------------------------------------------
// Stress — delegates to super::basic / super::interaction
// ---------------------------------------------------------------------------

/// Host workers competing with cgroup workers for CPU time.
///
/// Two cgroups plus unconstrained host workers (one per CPU).
/// Tests scheduler fairness between cgroup-managed and
/// non-cgroup-managed tasks.
pub fn host_contention(ctx: &Ctx) -> Result<AssertResult> {
    super::basic::custom_host_cgroup_contention(ctx)
}

/// Heavy + bursty + IO cgroups.
///
/// Three cgroups with different workload types: CPU-heavy, bursty
/// wake/sleep, and synchronous IO. Tests fairness across mixed
/// workload patterns.
pub fn mixed_workloads(ctx: &Ctx) -> Result<AssertResult> {
    super::interaction::custom_cgroup_imbalance_mixed_workload(ctx)
}

// ---------------------------------------------------------------------------
// Nested — delegates to super::nested
// ---------------------------------------------------------------------------

/// Workers in nested sub-cgroups.
///
/// Creates a multi-level cgroup hierarchy (cg_0/sub_a, cg_0/sub_b,
/// cg_1/sub_b, cg_1/sub_a/deep) with workers at the leaf level.
/// Tests scheduler handling of nested cgroup hierarchies.
pub fn nested_steady(ctx: &Ctx) -> Result<AssertResult> {
    super::nested::custom_nested_cgroup_steady(ctx)
}

/// Move tasks between nested cgroups.
///
/// Creates nested cgroups, spawns workers in one, then moves them
/// through the hierarchy (sub -> parent -> sibling/sub -> sibling).
/// Tests task migration across nesting levels.
pub fn nested_task_move(ctx: &Ctx) -> Result<AssertResult> {
    super::nested::custom_nested_cgroup_task_move(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all canned scenario functions have the expected signature:
    /// `fn(&Ctx) -> Result<AssertResult>`.
    #[test]
    fn all_scenario_fns_have_correct_signature() {
        let fns: Vec<fn(&Ctx) -> Result<AssertResult>> = vec![
            steady,
            steady_llc,
            cpuset_apply,
            cpuset_clear,
            cpuset_resize,
            cgroup_add,
            cgroup_remove,
            affinity_change,
            affinity_pinned,
            oversubscribed,
            host_contention,
            mixed_workloads,
            nested_steady,
            nested_task_move,
        ];
        // 14 canned scenarios.
        assert_eq!(fns.len(), 14);
    }
}
