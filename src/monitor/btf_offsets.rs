use std::path::Path;

use anyhow::{Context, Result, bail};
use btf_rs::{Btf, Type};

/// Load BTF from a path. Handles both raw BTF (/sys/kernel/btf/vmlinux)
/// and ELF files (vmlinux) by extracting the .BTF section.
fn load_btf_from_path(path: &Path) -> Result<Btf> {
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

        Ok(Self {
            rq_nr_running,
            rq_clock,
            rq_scx,
            scx_rq_nr_running,
            scx_rq_local_dsq,
            scx_rq_flags,
            dsq_nr,
            event_offsets,
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
fn find_struct(btf: &Btf, name: &str) -> Result<(btf_rs::Struct, String)> {
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
fn member_byte_offset(btf: &Btf, s: &btf_rs::Struct, field: &str) -> Result<usize> {
    for member in &s.members {
        let name = btf.resolve_name(member).unwrap_or_default();
        if name == field {
            let bits = member.bit_offset();
            if bits % 8 != 0 {
                bail!("btf: field '{field}' has non-byte-aligned offset ({bits} bits)");
            }
            return Ok((bits / 8) as usize);
        }
    }
    bail!("btf: field '{field}' not found in struct");
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
    let mut t = btf.resolve_chained_type(member)?;
    for _ in 0..20 {
        match t {
            Type::Struct(s) => return Ok(s),
            Type::Ptr(_)
            | Type::Const(_)
            | Type::Volatile(_)
            | Type::Typedef(_)
            | Type::Restrict(_)
            | Type::TypeTag(_) => {
                t = btf.resolve_chained_type(t.as_btf_type().unwrap())?;
            }
            _ => bail!(
                "btf: unexpected type '{}' while resolving member struct",
                t.name()
            ),
        }
    }
    bail!("btf: type chain too deep resolving member struct");
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
