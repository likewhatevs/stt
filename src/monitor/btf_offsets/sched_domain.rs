//! BTF offsets for `struct sched_domain` tree walking and the
//! `CONFIG_SCHEDSTATS` load-balancing stats fields it carries.
//!
//! `rq->sd` is a per-CPU pointer to the lowest-level domain; the
//! domain tree is walked via `sd->parent` until NULL. Runtime fields
//! (`balance_interval`, `nr_balance_failed`, `max_newidle_lb_cost`)
//! are unconditional. `newidle_call` / `newidle_success` /
//! `newidle_ratio` were added together in 6.19 (proportional newidle
//! balance — kernel commit 33cf66d88306) and are absent on older
//! kernels, so the resolver returns them as optional. Load-balancing
//! stats fields (`lb_count`, `alb_pushed`, `ttwu_wake_remote`, etc.)
//! are gated on `CONFIG_SCHEDSTATS` and resolved into an optional
//! [`SchedDomainStatsOffsets`].

use anyhow::Result;
use btf_rs::Btf;

use super::{find_struct, member_byte_offset};

/// Byte offsets for walking the `struct sched_domain` tree from guest memory.
///
/// `rq->sd` is a per-CPU pointer to the lowest-level domain. The domain
/// tree is walked via `sd->parent` until NULL. Resolution returns `Err`
/// when the `sd` field is missing from `struct rq` or when `struct
/// sched_domain` is absent from BTF.
///
/// Runtime fields (`balance_interval`, `nr_balance_failed`,
/// `max_newidle_lb_cost`) are always present on `struct sched_domain`.
/// `newidle_call`, `newidle_success`, and `newidle_ratio` were added
/// together in 6.19 and are absent on older kernels, so they are
/// resolved as optional.
/// Load balancing stats (`lb_count`, `alb_pushed`, `ttwu_wake_remote`,
/// etc.) are guarded by `CONFIG_SCHEDSTATS` and resolved separately
/// into an optional [`SchedDomainStatsOffsets`].
#[derive(Debug, Clone)]
pub struct SchedDomainOffsets {
    /// Offset of `sd` (pointer) within `struct rq`.
    pub rq_sd: usize,
    /// Offset of `parent` (pointer) within `struct sched_domain`.
    pub sd_parent: usize,
    /// Offset of `level` (int) within `struct sched_domain`.
    pub sd_level: usize,
    /// Offset of `name` (char *) within `struct sched_domain`.
    pub sd_name: usize,
    /// Offset of `flags` (int) within `struct sched_domain`.
    pub sd_flags: usize,
    /// Offset of `span_weight` (unsigned int) within `struct sched_domain`.
    pub sd_span_weight: usize,

    // -- Runtime fields --
    /// Offset of `balance_interval` (unsigned int) within `struct sched_domain`.
    pub sd_balance_interval: usize,
    /// Offset of `nr_balance_failed` (unsigned int) within `struct sched_domain`.
    pub sd_nr_balance_failed: usize,
    /// Offset of `newidle_call` (unsigned int) within `struct sched_domain`.
    /// None on pre-6.19 kernels where this field had not yet been added.
    pub sd_newidle_call: Option<usize>,
    /// Offset of `newidle_success` (unsigned int) within `struct sched_domain`.
    /// None on pre-6.19 kernels where this field had not yet been added.
    pub sd_newidle_success: Option<usize>,
    /// Offset of `newidle_ratio` (unsigned int) within `struct sched_domain`.
    /// None on pre-6.19 kernels where this field had not yet been added.
    pub sd_newidle_ratio: Option<usize>,
    /// Offset of `max_newidle_lb_cost` (u64) within `struct sched_domain`.
    pub sd_max_newidle_lb_cost: usize,

    /// CONFIG_SCHEDSTATS load balancing stats offsets. None when
    /// CONFIG_SCHEDSTATS is not enabled.
    pub stats_offsets: Option<SchedDomainStatsOffsets>,
}

/// Byte offsets for CONFIG_SCHEDSTATS fields on `struct sched_domain`.
///
/// Array fields are `unsigned int field[CPU_MAX_IDLE_TYPES]` where
/// `CPU_MAX_IDLE_TYPES = 3`. The offset is to element 0; element i
/// is at `offset + i * 4`.
#[derive(Debug, Clone)]
pub struct SchedDomainStatsOffsets {
    // Array fields indexed by cpu_idle_type.
    /// Offset of `lb_count[0]` within `struct sched_domain`.
    pub sd_lb_count: usize,
    /// Offset of `lb_failed[0]` within `struct sched_domain`.
    pub sd_lb_failed: usize,
    /// Offset of `lb_balanced[0]` within `struct sched_domain`.
    pub sd_lb_balanced: usize,
    /// Offset of `lb_imbalance_load[0]` within `struct sched_domain`.
    pub sd_lb_imbalance_load: usize,
    /// Offset of `lb_imbalance_util[0]` within `struct sched_domain`.
    pub sd_lb_imbalance_util: usize,
    /// Offset of `lb_imbalance_task[0]` within `struct sched_domain`.
    pub sd_lb_imbalance_task: usize,
    /// Offset of `lb_imbalance_misfit[0]` within `struct sched_domain`.
    pub sd_lb_imbalance_misfit: usize,
    /// Offset of `lb_gained[0]` within `struct sched_domain`.
    pub sd_lb_gained: usize,
    /// Offset of `lb_hot_gained[0]` within `struct sched_domain`.
    pub sd_lb_hot_gained: usize,
    /// Offset of `lb_nobusyg[0]` within `struct sched_domain`.
    pub sd_lb_nobusyg: usize,
    /// Offset of `lb_nobusyq[0]` within `struct sched_domain`.
    pub sd_lb_nobusyq: usize,

    // Scalar fields.
    /// Offset of `alb_count` within `struct sched_domain`.
    pub sd_alb_count: usize,
    /// Offset of `alb_failed` within `struct sched_domain`.
    pub sd_alb_failed: usize,
    /// Offset of `alb_pushed` within `struct sched_domain`.
    pub sd_alb_pushed: usize,
    /// Offset of `sbe_count` within `struct sched_domain`.
    pub sd_sbe_count: usize,
    /// Offset of `sbe_balanced` within `struct sched_domain`.
    pub sd_sbe_balanced: usize,
    /// Offset of `sbe_pushed` within `struct sched_domain`.
    pub sd_sbe_pushed: usize,
    /// Offset of `sbf_count` within `struct sched_domain`.
    pub sd_sbf_count: usize,
    /// Offset of `sbf_balanced` within `struct sched_domain`.
    pub sd_sbf_balanced: usize,
    /// Offset of `sbf_pushed` within `struct sched_domain`.
    pub sd_sbf_pushed: usize,
    /// Offset of `ttwu_wake_remote` within `struct sched_domain`.
    pub sd_ttwu_wake_remote: usize,
    /// Offset of `ttwu_move_affine` within `struct sched_domain`.
    pub sd_ttwu_move_affine: usize,
    /// Offset of `ttwu_move_balance` within `struct sched_domain`.
    pub sd_ttwu_move_balance: usize,
}

/// Number of idle types for array-indexed sched_domain stats fields.
/// `CPU_MAX_IDLE_TYPES = 3`: CPU_NOT_IDLE, CPU_IDLE, CPU_NEWLY_IDLE.
pub const CPU_MAX_IDLE_TYPES: usize = 3;

/// Resolve BTF offsets for sched_domain tree walking and stats.
/// Returns Err if `rq.sd` is missing or `struct sched_domain` is absent.
/// CONFIG_SCHEDSTATS fields are resolved separately and stored in
/// `stats_offsets` (None when CONFIG_SCHEDSTATS is off).
pub(super) fn resolve_sched_domain_offsets(
    btf: &Btf,
    rq_struct: &btf_rs::Struct,
) -> Result<SchedDomainOffsets> {
    let rq_sd = member_byte_offset(btf, rq_struct, "sd")?;

    let (sd_struct, _) = find_struct(btf, "sched_domain")?;
    let sd_parent = member_byte_offset(btf, &sd_struct, "parent")?;
    let sd_level = member_byte_offset(btf, &sd_struct, "level")?;
    let sd_name = member_byte_offset(btf, &sd_struct, "name")?;
    let sd_flags = member_byte_offset(btf, &sd_struct, "flags")?;
    let sd_span_weight = member_byte_offset(btf, &sd_struct, "span_weight")?;

    // Runtime fields.
    let sd_balance_interval = member_byte_offset(btf, &sd_struct, "balance_interval")?;
    let sd_nr_balance_failed = member_byte_offset(btf, &sd_struct, "nr_balance_failed")?;
    let sd_max_newidle_lb_cost = member_byte_offset(btf, &sd_struct, "max_newidle_lb_cost")?;

    // newidle_call/newidle_success/newidle_ratio were added together
    // in 6.19 and are absent on older kernels. Resolve all-or-nothing:
    // if any is missing, set all to None.
    let (sd_newidle_call, sd_newidle_success, sd_newidle_ratio) = match (
        member_byte_offset(btf, &sd_struct, "newidle_call").ok(),
        member_byte_offset(btf, &sd_struct, "newidle_success").ok(),
        member_byte_offset(btf, &sd_struct, "newidle_ratio").ok(),
    ) {
        (Some(c), Some(s), Some(r)) => (Some(c), Some(s), Some(r)),
        _ => (None, None, None),
    };

    // CONFIG_SCHEDSTATS fields (optional).
    let stats_offsets = resolve_sched_domain_stats_offsets(btf, &sd_struct).ok();

    Ok(SchedDomainOffsets {
        rq_sd,
        sd_parent,
        sd_level,
        sd_name,
        sd_flags,
        sd_span_weight,
        sd_balance_interval,
        sd_nr_balance_failed,
        sd_newidle_call,
        sd_newidle_success,
        sd_newidle_ratio,
        sd_max_newidle_lb_cost,
        stats_offsets,
    })
}

/// Resolve CONFIG_SCHEDSTATS field offsets on struct sched_domain.
/// Returns Err if any required field is missing.
fn resolve_sched_domain_stats_offsets(
    btf: &Btf,
    sd_struct: &btf_rs::Struct,
) -> Result<SchedDomainStatsOffsets> {
    Ok(SchedDomainStatsOffsets {
        sd_lb_count: member_byte_offset(btf, sd_struct, "lb_count")?,
        sd_lb_failed: member_byte_offset(btf, sd_struct, "lb_failed")?,
        sd_lb_balanced: member_byte_offset(btf, sd_struct, "lb_balanced")?,
        sd_lb_imbalance_load: member_byte_offset(btf, sd_struct, "lb_imbalance_load")?,
        sd_lb_imbalance_util: member_byte_offset(btf, sd_struct, "lb_imbalance_util")?,
        sd_lb_imbalance_task: member_byte_offset(btf, sd_struct, "lb_imbalance_task")?,
        sd_lb_imbalance_misfit: member_byte_offset(btf, sd_struct, "lb_imbalance_misfit")?,
        sd_lb_gained: member_byte_offset(btf, sd_struct, "lb_gained")?,
        sd_lb_hot_gained: member_byte_offset(btf, sd_struct, "lb_hot_gained")?,
        sd_lb_nobusyg: member_byte_offset(btf, sd_struct, "lb_nobusyg")?,
        sd_lb_nobusyq: member_byte_offset(btf, sd_struct, "lb_nobusyq")?,
        sd_alb_count: member_byte_offset(btf, sd_struct, "alb_count")?,
        sd_alb_failed: member_byte_offset(btf, sd_struct, "alb_failed")?,
        sd_alb_pushed: member_byte_offset(btf, sd_struct, "alb_pushed")?,
        sd_sbe_count: member_byte_offset(btf, sd_struct, "sbe_count")?,
        sd_sbe_balanced: member_byte_offset(btf, sd_struct, "sbe_balanced")?,
        sd_sbe_pushed: member_byte_offset(btf, sd_struct, "sbe_pushed")?,
        sd_sbf_count: member_byte_offset(btf, sd_struct, "sbf_count")?,
        sd_sbf_balanced: member_byte_offset(btf, sd_struct, "sbf_balanced")?,
        sd_sbf_pushed: member_byte_offset(btf, sd_struct, "sbf_pushed")?,
        sd_ttwu_wake_remote: member_byte_offset(btf, sd_struct, "ttwu_wake_remote")?,
        sd_ttwu_move_affine: member_byte_offset(btf, sd_struct, "ttwu_move_affine")?,
        sd_ttwu_move_balance: member_byte_offset(btf, sd_struct, "ttwu_move_balance")?,
    })
}
