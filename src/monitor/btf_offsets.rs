//! BTF-based struct field offset resolution.
//!
//! Parses BTF from a vmlinux ELF (or raw `/sys/kernel/btf/vmlinux`)
//! to resolve byte offsets of kernel struct fields needed for
//! host-side memory reads: runqueue monitoring ([`KernelOffsets`]),
//! scx event counters ([`ScxEventOffsets`]), schedstat fields
//! ([`SchedstatOffsets`]), sched domain tree walking
//! ([`SchedDomainOffsets`]), BPF map discovery ([`BpfMapOffsets`]),
//! BPF hash map iteration ([`HtabOffsets`]), and BPF program
//! enumeration ([`BpfProgOffsets`]).

use std::path::Path;

use anyhow::{Context, Result, bail};
use btf_rs::{Btf, Type};

/// Load BTF from a path. Handles both raw BTF (/sys/kernel/btf/vmlinux)
/// and ELF files (vmlinux) by extracting the .BTF section.
pub(crate) fn load_btf_from_path(path: &Path) -> Result<Btf> {
    let data = std::fs::read(path).context("read file")?;
    // Try raw BTF first (starts with BTF magic 0x9FEB)
    if data.len() >= 4 && data[0] == 0x9F && data[1] == 0xEB {
        return Btf::from_bytes(&data).map_err(|e| anyhow::anyhow!("{e}"));
    }
    // Try ELF — extract .BTF section
    if let Ok(elf) = object::File::parse(&*data) {
        use object::{Object, ObjectSection};
        if let Some(section) = elf.section_by_name(".BTF") {
            let btf_data = section.data().context(".BTF section data")?;
            return Btf::from_bytes(btf_data).map_err(|e| anyhow::anyhow!("{e}"));
        }
        bail!("vmlinux ELF has no .BTF section");
    }
    // Fallback: try btf-rs directly
    Btf::from_file(path).map_err(|e| anyhow::anyhow!("{e}"))
}

/// Byte offsets of kernel struct fields needed for host-side rq monitoring.
/// All offsets are relative to the start of their containing struct.
#[derive(Debug, Clone)]
pub struct KernelOffsets {
    /// Offset of `nr_running` within `struct rq`.
    pub rq_nr_running: usize,
    /// Offset of `clock` within `struct rq`.
    pub rq_clock: usize,
    /// Offset of `scx` (struct scx_rq) within `struct rq`.
    pub rq_scx: usize,
    /// Offset of `nr_running` within `struct scx_rq`.
    pub scx_rq_nr_running: usize,
    /// Offset of `local_dsq` (struct scx_dispatch_q) within `struct scx_rq`.
    pub scx_rq_local_dsq: usize,
    /// Offset of `flags` within `struct scx_rq`.
    pub scx_rq_flags: usize,
    /// Offset of `nr` within `struct scx_dispatch_q`.
    pub dsq_nr: usize,
    /// Offsets for scx event counters. None if BTF lacks the required types.
    pub event_offsets: Option<ScxEventOffsets>,
    /// Offsets for struct rq schedstat fields. None if CONFIG_SCHEDSTATS
    /// is not enabled (BTF lacks the required fields).
    pub schedstat_offsets: Option<SchedstatOffsets>,
    /// Offsets for sched_domain tree walking and stats. None if BTF
    /// lacks the `sd` field on `struct rq` or `struct sched_domain`.
    pub sched_domain_offsets: Option<SchedDomainOffsets>,
}

/// Byte offsets for reading scx event counters from guest memory.
///
/// Event counters live in `scx_sched_pcpu.event_stats` (per-CPU).
/// The host resolves per-CPU addresses via `scx_root -> scx_sched.pcpu`
/// plus `__per_cpu_offset[cpu]`.
#[derive(Debug, Clone)]
pub struct ScxEventOffsets {
    /// Offset of `pcpu` within `struct scx_sched`.
    pub scx_sched_pcpu_off: usize,
    /// Offset of `event_stats` within `struct scx_sched_pcpu`.
    pub event_stats_off: usize,
    /// Offset of `SCX_EV_SELECT_CPU_FALLBACK` within `struct scx_event_stats`.
    pub ev_select_cpu_fallback: usize,
    /// Offset of `SCX_EV_DISPATCH_LOCAL_DSQ_OFFLINE` within `struct scx_event_stats`.
    pub ev_dispatch_local_dsq_offline: usize,
    /// Offset of `SCX_EV_DISPATCH_KEEP_LAST` within `struct scx_event_stats`.
    pub ev_dispatch_keep_last: usize,
    /// Offset of `SCX_EV_ENQ_SKIP_EXITING` within `struct scx_event_stats`.
    pub ev_enq_skip_exiting: usize,
    /// Offset of `SCX_EV_ENQ_SKIP_MIGRATION_DISABLED` within `struct scx_event_stats`.
    pub ev_enq_skip_migration_disabled: usize,
}

impl KernelOffsets {
    /// Parse BTF from a vmlinux ELF and resolve field offsets for
    /// `struct rq`, `struct scx_rq`, and `struct scx_dispatch_q`.
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let btf =
            load_btf_from_path(path).with_context(|| format!("btf: open {}", path.display()))?;

        let (rq_struct, _) = find_struct(&btf, "rq")?;
        let rq_nr_running = member_byte_offset(&btf, &rq_struct, "nr_running")?;
        let rq_clock = member_byte_offset(&btf, &rq_struct, "clock")?;
        let (rq_scx, scx_member) = member_byte_offset_with_member(&btf, &rq_struct, "scx")?;

        // Resolve the type of rq.scx to get struct scx_rq.
        let scx_rq_struct =
            resolve_member_struct(&btf, &scx_member).context("btf: resolve type of rq.scx")?;
        let scx_rq_nr_running = member_byte_offset(&btf, &scx_rq_struct, "nr_running")?;
        let (scx_rq_local_dsq, local_dsq_member) =
            member_byte_offset_with_member(&btf, &scx_rq_struct, "local_dsq")?;
        let scx_rq_flags = member_byte_offset(&btf, &scx_rq_struct, "flags")?;

        // Resolve the type of scx_rq.local_dsq to get struct scx_dispatch_q.
        let dsq_struct = resolve_member_struct(&btf, &local_dsq_member)
            .context("btf: resolve type of scx_rq.local_dsq")?;
        let dsq_nr = member_byte_offset(&btf, &dsq_struct, "nr")?;

        let event_offsets = resolve_event_offsets(&btf).ok();
        let schedstat_offsets = resolve_schedstat_offsets(&btf).ok();
        let sched_domain_offsets = resolve_sched_domain_offsets(&btf, &rq_struct).ok();

        Ok(Self {
            rq_nr_running,
            rq_clock,
            rq_scx,
            scx_rq_nr_running,
            scx_rq_local_dsq,
            scx_rq_flags,
            dsq_nr,
            event_offsets,
            schedstat_offsets,
            sched_domain_offsets,
        })
    }
}

/// Resolve BTF offsets for scx event counters.
/// Returns Err if any required type/field is missing (old kernel).
fn resolve_event_offsets(btf: &Btf) -> Result<ScxEventOffsets> {
    let (scx_sched_struct, _) = find_struct(btf, "scx_sched")?;
    let scx_sched_pcpu_off = member_byte_offset(btf, &scx_sched_struct, "pcpu")?;

    let (pcpu_struct, _) = find_struct(btf, "scx_sched_pcpu")?;
    let (event_stats_off, event_stats_member) =
        member_byte_offset_with_member(btf, &pcpu_struct, "event_stats")?;

    let event_stats_struct = resolve_member_struct(btf, &event_stats_member)
        .context("btf: resolve type of scx_sched_pcpu.event_stats")?;

    let ev_select_cpu_fallback =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_SELECT_CPU_FALLBACK")?;
    let ev_dispatch_local_dsq_offline = member_byte_offset(
        btf,
        &event_stats_struct,
        "SCX_EV_DISPATCH_LOCAL_DSQ_OFFLINE",
    )?;
    let ev_dispatch_keep_last =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_DISPATCH_KEEP_LAST")?;
    let ev_enq_skip_exiting =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_ENQ_SKIP_EXITING")?;
    let ev_enq_skip_migration_disabled = member_byte_offset(
        btf,
        &event_stats_struct,
        "SCX_EV_ENQ_SKIP_MIGRATION_DISABLED",
    )?;

    Ok(ScxEventOffsets {
        scx_sched_pcpu_off,
        event_stats_off,
        ev_select_cpu_fallback,
        ev_dispatch_local_dsq_offline,
        ev_dispatch_keep_last,
        ev_enq_skip_exiting,
        ev_enq_skip_migration_disabled,
    })
}

/// Find a named struct in BTF. Returns the Struct and its BTF type name.
pub(crate) fn find_struct(btf: &Btf, name: &str) -> Result<(btf_rs::Struct, String)> {
    let types = btf
        .resolve_types_by_name(name)
        .with_context(|| format!("btf: type '{name}' not found"))?;

    for t in &types {
        if let Type::Struct(s) = t {
            let resolved_name = btf.resolve_name(s).unwrap_or_default();
            return Ok((s.clone(), resolved_name));
        }
    }
    bail!("btf: '{name}' exists but is not a struct");
}

/// Find a member by name in a struct and return its byte offset.
///
/// Searches through anonymous struct/union members recursively to
/// handle fields inside `DECLARE_FLEX_ARRAY` and anonymous unions.
pub(crate) fn member_byte_offset(btf: &Btf, s: &btf_rs::Struct, field: &str) -> Result<usize> {
    member_byte_offset_recursive(btf, s, field, 0)
}

fn member_byte_offset_recursive(
    btf: &Btf,
    s: &btf_rs::Struct,
    field: &str,
    base_offset: usize,
) -> Result<usize> {
    for member in &s.members {
        let name = btf.resolve_name(member).unwrap_or_default();
        let bits = member.bit_offset();
        if bits % 8 != 0 {
            if name == field {
                bail!("btf: field '{field}' has non-byte-aligned offset ({bits} bits)");
            }
            continue;
        }
        let member_offset = base_offset + (bits / 8) as usize;

        if name == field {
            return Ok(member_offset);
        }

        // Anonymous member (empty name): recurse into nested struct/union.
        if name.is_empty()
            && let Ok(inner) = resolve_member_composite(btf, member)
            && let Ok(offset) = member_byte_offset_recursive(btf, &inner, field, member_offset)
        {
            return Ok(offset);
        }
    }
    bail!("btf: field '{field}' not found in struct");
}

/// Follow a Member's type_id through modifiers to reach a Struct or Union.
/// btf-rs uses `Union = Struct`, so both return as `btf_rs::Struct`.
fn resolve_member_composite(btf: &Btf, member: &btf_rs::Member) -> Result<btf_rs::Struct> {
    let mut t = btf.resolve_chained_type(member)?;
    for _ in 0..20 {
        match t {
            Type::Struct(s) | Type::Union(s) => return Ok(s),
            Type::Const(_)
            | Type::Volatile(_)
            | Type::Typedef(_)
            | Type::Restrict(_)
            | Type::TypeTag(_) => {
                t = btf.resolve_chained_type(t.as_btf_type().unwrap())?;
            }
            _ => bail!("btf: not a composite type"),
        }
    }
    bail!("btf: type chain too deep")
}

/// Like `member_byte_offset` but also returns the Member for type resolution.
fn member_byte_offset_with_member(
    btf: &Btf,
    s: &btf_rs::Struct,
    field: &str,
) -> Result<(usize, btf_rs::Member)> {
    for member in &s.members {
        let name = btf.resolve_name(member).unwrap_or_default();
        if name == field {
            let bits = member.bit_offset();
            if bits % 8 != 0 {
                bail!("btf: field '{field}' has non-byte-aligned offset ({bits} bits)");
            }
            return Ok(((bits / 8) as usize, member.clone()));
        }
    }
    bail!("btf: field '{field}' not found in struct");
}

/// Follow a Member's type_id through Ptr/Const/Volatile/Typedef/TypeTag
/// chains to reach the underlying Struct.
fn resolve_member_struct(btf: &Btf, member: &btf_rs::Member) -> Result<btf_rs::Struct> {
    use btf_rs::BtfType;
    let tid = member.get_type_id().context("btf: member type_id")?;
    super::bpf_map::resolve_to_struct(btf, tid).context("btf: could not resolve member to struct")
}

/// Byte offsets for reading struct rq schedstat fields from guest memory.
///
/// Schedstat fields are guarded by `CONFIG_SCHEDSTATS` in the kernel.
/// Resolution is optional — `resolve_schedstat_offsets()` returns `Err`
/// when the required fields are missing from BTF.
#[derive(Debug, Clone)]
pub struct SchedstatOffsets {
    /// Offset of `rq_sched_info` (struct sched_info) within `struct rq`.
    pub rq_sched_info: usize,
    /// Offset of `run_delay` within `struct sched_info`.
    pub sched_info_run_delay: usize,
    /// Offset of `pcount` within `struct sched_info`.
    pub sched_info_pcount: usize,
    /// Offset of `yld_count` within `struct rq`.
    pub rq_yld_count: usize,
    /// Offset of `sched_count` within `struct rq`.
    pub rq_sched_count: usize,
    /// Offset of `sched_goidle` within `struct rq`.
    pub rq_sched_goidle: usize,
    /// Offset of `ttwu_count` within `struct rq`.
    pub rq_ttwu_count: usize,
    /// Offset of `ttwu_local` within `struct rq`.
    pub rq_ttwu_local: usize,
}

/// Resolve BTF offsets for struct rq schedstat fields.
/// Returns Err if any required type/field is missing (CONFIG_SCHEDSTATS
/// not enabled in the kernel).
fn resolve_schedstat_offsets(btf: &Btf) -> Result<SchedstatOffsets> {
    let (rq_struct, _) = find_struct(btf, "rq")?;

    // rq.rq_sched_info is struct sched_info embedded in struct rq.
    let (rq_sched_info, sched_info_member) =
        member_byte_offset_with_member(btf, &rq_struct, "rq_sched_info")?;

    let sched_info_struct = resolve_member_struct(btf, &sched_info_member)
        .context("btf: resolve type of rq.rq_sched_info")?;
    let sched_info_run_delay = member_byte_offset(btf, &sched_info_struct, "run_delay")?;
    let sched_info_pcount = member_byte_offset(btf, &sched_info_struct, "pcount")?;

    // Direct unsigned int fields on struct rq.
    let rq_yld_count = member_byte_offset(btf, &rq_struct, "yld_count")?;
    let rq_sched_count = member_byte_offset(btf, &rq_struct, "sched_count")?;
    let rq_sched_goidle = member_byte_offset(btf, &rq_struct, "sched_goidle")?;
    let rq_ttwu_count = member_byte_offset(btf, &rq_struct, "ttwu_count")?;
    let rq_ttwu_local = member_byte_offset(btf, &rq_struct, "ttwu_local")?;

    Ok(SchedstatOffsets {
        rq_sched_info,
        sched_info_run_delay,
        sched_info_pcount,
        rq_yld_count,
        rq_sched_count,
        rq_sched_goidle,
        rq_ttwu_count,
        rq_ttwu_local,
    })
}

/// Byte offsets for walking the `struct sched_domain` tree from guest memory.
///
/// `rq->sd` is a per-CPU pointer to the lowest-level domain. The domain
/// tree is walked via `sd->parent` until NULL. Resolution returns `Err`
/// when the `sd` field is missing from `struct rq` or when `struct
/// sched_domain` is absent from BTF.
///
/// Runtime fields (`balance_interval`, `nr_balance_failed`,
/// `newidle_call`, etc.) are always present on `struct sched_domain`.
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

    // -- Runtime fields (always present) --
    /// Offset of `balance_interval` (unsigned int) within `struct sched_domain`.
    pub sd_balance_interval: usize,
    /// Offset of `nr_balance_failed` (unsigned int) within `struct sched_domain`.
    pub sd_nr_balance_failed: usize,
    /// Offset of `newidle_call` (unsigned int) within `struct sched_domain`.
    pub sd_newidle_call: usize,
    /// Offset of `newidle_success` (unsigned int) within `struct sched_domain`.
    pub sd_newidle_success: usize,
    /// Offset of `newidle_ratio` (unsigned int) within `struct sched_domain`.
    pub sd_newidle_ratio: usize,
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
fn resolve_sched_domain_offsets(
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

    // Runtime fields (always present).
    let sd_balance_interval = member_byte_offset(btf, &sd_struct, "balance_interval")?;
    let sd_nr_balance_failed = member_byte_offset(btf, &sd_struct, "nr_balance_failed")?;
    let sd_newidle_call = member_byte_offset(btf, &sd_struct, "newidle_call")?;
    let sd_newidle_success = member_byte_offset(btf, &sd_struct, "newidle_success")?;
    let sd_newidle_ratio = member_byte_offset(btf, &sd_struct, "newidle_ratio")?;
    let sd_max_newidle_lb_cost = member_byte_offset(btf, &sd_struct, "max_newidle_lb_cost")?;

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

/// Byte offsets within kernel BPF structures needed for host-side
/// BPF map discovery and value access.
#[derive(Debug, Clone)]
pub struct BpfMapOffsets {
    /// Offset of `name` (char\[BPF_OBJ_NAME_LEN\]) within `struct bpf_map`.
    pub map_name: usize,
    /// Offset of `map_type` (enum bpf_map_type, u32) within `struct bpf_map`.
    pub map_type: usize,
    /// Offset of `map_flags` (u32) within `struct bpf_map`.
    pub map_flags: usize,
    /// Offset of `key_size` (u32) within `struct bpf_map`.
    pub key_size: usize,
    /// Offset of `value_size` (u32) within `struct bpf_map`.
    pub value_size: usize,
    /// Offset of `max_entries` (u32) within `struct bpf_map`.
    pub max_entries: usize,
    /// Offset of `value`/`ptrs`/`pptrs` union within `struct bpf_array`.
    /// For `BPF_MAP_TYPE_ARRAY`: inline value data at this offset.
    /// For `BPF_MAP_TYPE_PERCPU_ARRAY`: `__percpu` pointers at this offset.
    pub array_value: usize,
    /// Offset of `slots` within `struct xa_node`.
    pub xa_node_slots: usize,
    /// Offset of `shift` (u8) within `struct xa_node`.
    pub xa_node_shift: usize,
    /// Offset of `xa_head` within `struct idr`.
    /// Computed as idr.idr_rt (xarray) offset + xarray.xa_head offset.
    pub idr_xa_head: usize,
    /// Offset of `idr_next` (unsigned int) within `struct idr`.
    /// The next ID to allocate — scanning `0..idr_next` covers all
    /// allocated entries without wrapping past the xarray's slot count.
    pub idr_next: usize,
    /// Offset of `btf` pointer within `struct bpf_map`.
    pub map_btf: usize,
    /// Offset of `btf_value_type_id` (u32) within `struct bpf_map`.
    pub map_btf_value_type_id: usize,
    /// Offset of `data` pointer within `struct btf`.
    pub btf_data: usize,
    /// Offset of `data_size` (u32) within `struct btf`.
    pub btf_data_size: usize,
    /// Offsets for hash table structures. None if BTF lacks the
    /// required types (e.g. `bpf_htab` is not in vmlinux BTF).
    pub htab_offsets: Option<HtabOffsets>,
}

impl BpfMapOffsets {
    /// Parse BTF from a vmlinux ELF and resolve BPF map field offsets.
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let btf =
            load_btf_from_path(path).with_context(|| format!("btf: open {}", path.display()))?;
        Self::from_btf(&btf)
    }

    /// Resolve BPF map struct offsets from a pre-loaded BTF object.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (bpf_map, _) = find_struct(btf, "bpf_map")?;
        let map_name = member_byte_offset(btf, &bpf_map, "name")?;
        let map_type = member_byte_offset(btf, &bpf_map, "map_type")?;
        let map_flags = member_byte_offset(btf, &bpf_map, "map_flags")?;
        let key_size = member_byte_offset(btf, &bpf_map, "key_size")?;
        let value_size = member_byte_offset(btf, &bpf_map, "value_size")?;
        let max_entries = member_byte_offset(btf, &bpf_map, "max_entries")?;

        let (bpf_array, _) = find_struct(btf, "bpf_array")?;
        let array_value = member_byte_offset(btf, &bpf_array, "value")?;

        let (xa_node, _) = find_struct(btf, "xa_node")?;
        let xa_node_slots = member_byte_offset(btf, &xa_node, "slots")?;
        let xa_node_shift = member_byte_offset(btf, &xa_node, "shift")?;

        // struct idr { struct xarray idr_rt; ... }
        // xa_head offset within idr = idr_rt offset + xa_head offset in xarray.
        let (idr_struct, _) = find_struct(btf, "idr")?;
        let (idr_rt_off, idr_rt_member) =
            member_byte_offset_with_member(btf, &idr_struct, "idr_rt")?;
        let xa_struct = resolve_member_struct(btf, &idr_rt_member)
            .context("btf: resolve type of idr.idr_rt")?;
        let xa_head_off = member_byte_offset(btf, &xa_struct, "xa_head")?;
        let idr_xa_head = idr_rt_off + xa_head_off;

        let idr_next = member_byte_offset(btf, &idr_struct, "idr_next")?;

        let map_btf = member_byte_offset(btf, &bpf_map, "btf")?;
        let map_btf_value_type_id = member_byte_offset(btf, &bpf_map, "btf_value_type_id")?;

        let (btf_struct, _) = find_struct(btf, "btf")?;
        let btf_data = member_byte_offset(btf, &btf_struct, "data")?;
        let btf_data_size = member_byte_offset(btf, &btf_struct, "data_size")?;

        let htab_offsets = resolve_htab_offsets(btf).ok();

        Ok(Self {
            map_name,
            map_type,
            map_flags,
            key_size,
            value_size,
            max_entries,
            array_value,
            xa_node_slots,
            xa_node_shift,
            idr_xa_head,
            idr_next,
            map_btf,
            map_btf_value_type_id,
            btf_data,
            btf_data_size,
            htab_offsets,
        })
    }
}

/// Byte offsets within kernel BPF hash table structures needed for
/// host-side hash map iteration.
///
/// Resolution is optional — `resolve_htab_offsets()` returns `Err`
/// when the required types are missing from BTF.
#[derive(Debug, Clone)]
pub struct HtabOffsets {
    /// Offset of `buckets` pointer within `struct bpf_htab`.
    pub htab_buckets: usize,
    /// Offset of `n_buckets` (u32) within `struct bpf_htab`.
    pub htab_n_buckets: usize,
    /// Size of `struct bucket` in bytes.
    pub bucket_size: usize,
    /// Offset of `head` (`struct hlist_nulls_head`) within `struct bucket`.
    pub bucket_head: usize,
    /// Offset of `first` pointer within `struct hlist_nulls_head`.
    pub hlist_nulls_head_first: usize,
    /// Offset of `next` pointer within `struct hlist_nulls_node`.
    pub hlist_nulls_node_next: usize,
    /// Size of `struct htab_elem` (base size, before flex key[]).
    pub htab_elem_size_base: usize,
}

/// Find the BPF hashtab `struct bucket` among possibly multiple BTF
/// structs named `bucket`. Returns the struct and its `head` field offset.
/// The BPF bucket has a `head` field (`hlist_nulls_head`); other `bucket`
/// structs (e.g. bcache) do not.
fn find_bucket_struct(btf: &Btf) -> Result<(btf_rs::Struct, usize)> {
    let types = btf
        .resolve_types_by_name("bucket")
        .with_context(|| "btf: type 'bucket' not found")?;

    for t in &types {
        if let Type::Struct(s) = t
            && let Ok(head_off) = member_byte_offset(btf, s, "head")
        {
            return Ok((s.clone(), head_off));
        }
    }
    bail!("btf: no 'bucket' struct with 'head' field found");
}

/// Resolve BTF offsets for BPF hash table structures.
/// Returns Err if any required type/field is missing.
fn resolve_htab_offsets(btf: &Btf) -> Result<HtabOffsets> {
    let (bpf_htab, _) = find_struct(btf, "bpf_htab")?;
    let htab_buckets = member_byte_offset(btf, &bpf_htab, "buckets")?;
    let htab_n_buckets = member_byte_offset(btf, &bpf_htab, "n_buckets")?;

    // Multiple structs named `bucket` may exist in BTF (e.g. bcache).
    // Find the one with a `head` field (BPF hashtab's bucket).
    let (bucket_struct, bucket_head) = find_bucket_struct(btf)?;
    let bucket_size = bucket_struct.size();

    let (hlist_nulls_head, _) = find_struct(btf, "hlist_nulls_head")?;
    let hlist_nulls_head_first = member_byte_offset(btf, &hlist_nulls_head, "first")?;

    let (hlist_nulls_node, _) = find_struct(btf, "hlist_nulls_node")?;
    let hlist_nulls_node_next = member_byte_offset(btf, &hlist_nulls_node, "next")?;

    let (htab_elem, _) = find_struct(btf, "htab_elem")?;
    let htab_elem_size_base = htab_elem.size();

    Ok(HtabOffsets {
        htab_buckets,
        htab_n_buckets,
        bucket_size,
        bucket_head,
        hlist_nulls_head_first,
        hlist_nulls_node_next,
        htab_elem_size_base,
    })
}

/// Byte offsets within kernel BPF program structures needed for
/// host-side BPF program enumeration and verifier stats collection.
#[derive(Debug, Clone)]
pub struct BpfProgOffsets {
    /// Offset of `type` (enum bpf_prog_type, u32) within `struct bpf_prog`.
    pub prog_type: usize,
    /// Offset of `aux` pointer within `struct bpf_prog`.
    pub prog_aux: usize,
    /// Offset of `verified_insns` (u32) within `struct bpf_prog_aux`.
    pub aux_verified_insns: usize,
    /// Offset of `name` (char\[BPF_OBJ_NAME_LEN\]) within `struct bpf_prog_aux`.
    pub aux_name: usize,
    /// IDR offsets reused from BpfMapOffsets for walking prog_idr.
    pub xa_node_slots: usize,
    /// Offset of `shift` (u8) within `struct xa_node`.
    pub xa_node_shift: usize,
    /// Offset of `xa_head` within `struct idr` (idr.idr_rt.xa_head).
    pub idr_xa_head: usize,
    /// Offset of `idr_next` (unsigned int) within `struct idr`.
    pub idr_next: usize,
    /// Offset of `stats` (__percpu pointer) within `struct bpf_prog`.
    pub prog_stats: usize,
    /// Offset of `cnt` (u64_stats_t) within `struct bpf_prog_stats`.
    pub stats_cnt: usize,
    /// Offset of `nsecs` (u64_stats_t) within `struct bpf_prog_stats`.
    pub stats_nsecs: usize,
}

impl BpfProgOffsets {
    /// Resolve BPF program struct offsets from a pre-loaded BTF object.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (bpf_prog, _) = find_struct(btf, "bpf_prog")?;
        let prog_type = member_byte_offset(btf, &bpf_prog, "type")?;
        let prog_aux = member_byte_offset(btf, &bpf_prog, "aux")?;

        let (bpf_prog_aux, _) = find_struct(btf, "bpf_prog_aux")?;
        let aux_verified_insns = member_byte_offset(btf, &bpf_prog_aux, "verified_insns")?;
        let aux_name = member_byte_offset(btf, &bpf_prog_aux, "name")?;

        let (xa_node, _) = find_struct(btf, "xa_node")?;
        let xa_node_slots = member_byte_offset(btf, &xa_node, "slots")?;
        let xa_node_shift = member_byte_offset(btf, &xa_node, "shift")?;

        let (idr_struct, _) = find_struct(btf, "idr")?;
        let (idr_rt_off, idr_rt_member) =
            member_byte_offset_with_member(btf, &idr_struct, "idr_rt")?;
        let xa_struct = resolve_member_struct(btf, &idr_rt_member)
            .context("btf: resolve type of idr.idr_rt")?;
        let xa_head_off = member_byte_offset(btf, &xa_struct, "xa_head")?;
        let idr_xa_head = idr_rt_off + xa_head_off;

        let idr_next = member_byte_offset(btf, &idr_struct, "idr_next")?;

        let prog_stats = member_byte_offset(btf, &bpf_prog, "stats")?;

        let (bpf_prog_stats, _) = find_struct(btf, "bpf_prog_stats")?;
        let stats_cnt = member_byte_offset(btf, &bpf_prog_stats, "cnt")?;
        let stats_nsecs = member_byte_offset(btf, &bpf_prog_stats, "nsecs")?;

        Ok(Self {
            prog_type,
            prog_aux,
            aux_verified_insns,
            aux_name,
            xa_node_slots,
            xa_node_shift,
            idr_xa_head,
            idr_next,
            prog_stats,
            stats_cnt,
            stats_nsecs,
        })
    }

    /// Parse BTF from a vmlinux ELF and resolve BPF program field offsets.
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let btf =
            load_btf_from_path(path).with_context(|| format!("btf: open {}", path.display()))?;
        Self::from_btf(&btf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rq_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = KernelOffsets::from_vmlinux(&path).unwrap();
        assert!(offsets.rq_nr_running > 0);
        assert!(offsets.rq_clock > 0);
        assert!(offsets.rq_scx > 0);
        assert!(offsets.dsq_nr > 0);
    }

    #[test]
    fn parse_event_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = KernelOffsets::from_vmlinux(&path).unwrap();
        // Event offsets are optional — only assert if present.
        if let Some(ev) = &offsets.event_offsets {
            // All event counter fields are s64, so offsets must differ.
            let all = [
                ev.ev_select_cpu_fallback,
                ev.ev_dispatch_local_dsq_offline,
                ev.ev_dispatch_keep_last,
                ev.ev_enq_skip_exiting,
                ev.ev_enq_skip_migration_disabled,
            ];
            for i in 0..all.len() {
                for j in (i + 1)..all.len() {
                    assert_ne!(all[i], all[j], "event counter offsets must be distinct");
                }
            }
        }
    }

    #[test]
    fn parse_schedstat_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = KernelOffsets::from_vmlinux(&path).unwrap();
        // Schedstat offsets are optional — only assert if present.
        if let Some(ss) = &offsets.schedstat_offsets {
            // rq_sched_info must be at a nonzero offset (it's not the first
            // field of struct rq).
            assert!(ss.rq_sched_info > 0);
            // pcount is the first field in struct sched_info, so its offset
            // can be 0. run_delay follows pcount, so it must be > 0.
            assert!(
                ss.sched_info_run_delay > 0,
                "run_delay must follow pcount in struct sched_info"
            );
            assert_ne!(
                ss.sched_info_pcount, ss.sched_info_run_delay,
                "pcount and run_delay offsets must be distinct"
            );
            // All rq-level fields must be at distinct nonzero offsets.
            let rq_fields = [
                ss.rq_yld_count,
                ss.rq_sched_count,
                ss.rq_sched_goidle,
                ss.rq_ttwu_count,
                ss.rq_ttwu_local,
            ];
            for &off in &rq_fields {
                assert!(off > 0, "schedstat rq field offset must be nonzero");
            }
            for i in 0..rq_fields.len() {
                for j in (i + 1)..rq_fields.len() {
                    assert_ne!(
                        rq_fields[i], rq_fields[j],
                        "schedstat rq field offsets must be distinct"
                    );
                }
            }
        }
    }

    #[test]
    fn parse_sched_domain_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = KernelOffsets::from_vmlinux(&path).unwrap();
        // Sched domain offsets are optional — only assert if present.
        if let Some(sd) = &offsets.sched_domain_offsets {
            // sd fields must be at distinct nonzero offsets (none is the
            // first field of struct sched_domain — parent is).
            assert!(sd.rq_sd > 0, "rq.sd must be at nonzero offset");
            // parent can be offset 0 (first field). level and name must
            // differ from parent.
            assert_ne!(
                sd.sd_level, sd.sd_parent,
                "level and parent offsets must be distinct"
            );
            assert_ne!(
                sd.sd_name, sd.sd_parent,
                "name and parent offsets must be distinct"
            );
            // Runtime fields must be at distinct nonzero offsets.
            let runtime_fields = [
                sd.sd_balance_interval,
                sd.sd_nr_balance_failed,
                sd.sd_newidle_call,
                sd.sd_newidle_success,
                sd.sd_newidle_ratio,
                sd.sd_max_newidle_lb_cost,
            ];
            for &off in &runtime_fields {
                assert!(off > 0, "sched_domain runtime field offset must be nonzero");
            }
            // Stats offsets are optional (CONFIG_SCHEDSTATS).
            if let Some(so) = &sd.stats_offsets {
                let array_fields = [
                    so.sd_lb_count,
                    so.sd_lb_failed,
                    so.sd_lb_balanced,
                    so.sd_lb_imbalance_load,
                    so.sd_lb_imbalance_util,
                    so.sd_lb_imbalance_task,
                    so.sd_lb_imbalance_misfit,
                    so.sd_lb_gained,
                    so.sd_lb_hot_gained,
                    so.sd_lb_nobusyg,
                    so.sd_lb_nobusyq,
                ];
                for i in 0..array_fields.len() {
                    for j in (i + 1)..array_fields.len() {
                        assert_ne!(
                            array_fields[i], array_fields[j],
                            "sched_domain array field offsets must be distinct"
                        );
                    }
                }
                let scalar_fields = [
                    so.sd_alb_count,
                    so.sd_alb_failed,
                    so.sd_alb_pushed,
                    so.sd_ttwu_wake_remote,
                    so.sd_ttwu_move_affine,
                    so.sd_ttwu_move_balance,
                ];
                for &off in &scalar_fields {
                    assert!(off > 0, "sched_domain scalar field offset must be nonzero");
                }
                for i in 0..scalar_fields.len() {
                    for j in (i + 1)..scalar_fields.len() {
                        assert_ne!(
                            scalar_fields[i], scalar_fields[j],
                            "sched_domain scalar field offsets must be distinct"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parse_bpf_map_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = BpfMapOffsets::from_vmlinux(&path).unwrap();
        // All offsets should be nonzero in a real kernel BTF.
        assert!(offsets.map_name > 0);
        assert!(offsets.map_type > 0);
        assert!(offsets.value_size > 0);
        assert!(offsets.array_value > 0);
        // BTF-related offsets should be resolved.
        // btf_data can be 0 (first field in struct btf), so just verify
        // that parsing succeeded without error. btf_data_size cannot be
        // the first field (data comes before it), so it must be nonzero.
        assert!(offsets.map_btf > 0);
        assert!(offsets.map_btf_value_type_id > 0);
        assert!(offsets.btf_data_size > offsets.btf_data);
    }

    #[test]
    fn parse_bpf_prog_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = BpfProgOffsets::from_vmlinux(&path).unwrap();
        assert!(offsets.prog_aux > 0);
        assert!(offsets.aux_verified_insns > 0);
        assert!(offsets.aux_name > 0);
    }

    #[test]
    fn from_vmlinux_nonexistent() {
        let path = std::path::Path::new("/nonexistent/vmlinux");
        assert!(KernelOffsets::from_vmlinux(path).is_err());
    }

    #[test]
    fn from_vmlinux_empty_file() {
        let dir = std::env::temp_dir().join(format!("stt-btf-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("vmlinux");
        std::fs::write(&f, b"").unwrap();
        assert!(KernelOffsets::from_vmlinux(&f).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
