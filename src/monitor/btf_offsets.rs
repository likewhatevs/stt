//! BTF-based struct field offset resolution.
//!
//! Parses BTF from a vmlinux ELF (or raw `/sys/kernel/btf/vmlinux`)
//! to resolve byte offsets of kernel struct fields needed for
//! host-side memory reads: runqueue monitoring ([`KernelOffsets`]),
//! scx event counters ([`ScxEventOffsets`]), watchdog timeout override
//! ([`ScxWatchdogOffsets`]), schedstat fields ([`SchedstatOffsets`]),
//! sched domain tree walking ([`SchedDomainOffsets`]) with optional
//! load balancing stats ([`SchedDomainStatsOffsets`]), BPF map
//! discovery ([`BpfMapOffsets`]), BPF hash map iteration
//! ([`HtabOffsets`]), BPF program enumeration
//! ([`BpfProgOffsets`]), and shared `struct idr` walking
//! ([`IdrOffsets`]).

use std::path::Path;

use anyhow::{Context, Result, bail};
use btf_rs::{Btf, Type};

/// Load BTF from a path. Accepts two input shapes:
///
/// - Raw BTF (e.g. `/sys/kernel/btf/vmlinux`) — identified by the
///   leading 0x9FEB magic bytes, parsed via `Btf::from_bytes`.
/// - ELF vmlinux — parsed via `goblin::elf::Elf`; the `.BTF`
///   section bytes are extracted and handed to `Btf::from_bytes`.
///
/// Any other input is rejected with a "not recognized as raw BTF or
/// ELF vmlinux" error.
///
/// # BTF sidecar cache
///
/// For ELF inputs only, the extracted `.BTF` section bytes are cached
/// as a sibling file at `<path>.btf` (e.g. `vmlinux` →
/// `vmlinux.btf`). On subsequent loads, if the sidecar exists and
/// its mtime is greater than or equal to the vmlinux mtime, the
/// cached bytes are read and parsed directly, skipping the goblin
/// ELF parse + `.BTF` section extraction (the slow path on a
/// multi-hundred-MB vmlinux).
///
/// The sidecar is written lazily on first load after a cache miss.
/// Write failures (e.g. read-only directory) are logged at
/// `tracing::warn` level and do not fail the load — the function
/// falls through with the freshly-parsed BTF. Raw-BTF inputs never
/// write a sidecar: the input file IS the BTF blob and a sidecar
/// would just be a redundant copy of itself.
///
/// Staleness: mtime-based, no content hash. `CacheDir::store`'s
/// atomic rename path bumps vmlinux mtime when an entry is replaced,
/// so a previously-written sidecar next to the old vmlinux surfaces
/// as stale (`mtime(sidecar) < mtime(vmlinux)`) and the bytes are
/// re-extracted + re-written on the next load. Source-tree
/// vmlinuxes rebuilt in place bump mtime the same way.
pub(crate) fn load_btf_from_path(path: &Path) -> Result<Btf> {
    let data = std::fs::read(path).context("read file")?;
    // Raw BTF: first 2 bytes are the 0x9FEB magic. Parse directly;
    // never write a sidecar (would be a byte-for-byte self-copy).
    if is_raw_btf(&data) {
        return Btf::from_bytes(&data).map_err(|e| anyhow::anyhow!("{e}"));
    }

    // ELF path: try the BTF sidecar cache before re-parsing ELF.
    let sidecar = btf_sidecar_path(path);
    if sidecar_fresh(&sidecar, path) {
        match std::fs::read(&sidecar) {
            Ok(cached) if is_raw_btf(&cached) => {
                match Btf::from_bytes(&cached) {
                    Ok(btf) => return Ok(btf),
                    Err(e) => {
                        // Parse failure on a fresh-looking sidecar:
                        // treat as corrupt and fall through to ELF
                        // extraction. The subsequent write overwrites
                        // the corrupt file.
                        tracing::warn!(
                            path = %sidecar.display(),
                            err = %e,
                            "btf sidecar parse failed; falling back to ELF extraction",
                        );
                    }
                }
            }
            Ok(_) => {
                tracing::warn!(
                    path = %sidecar.display(),
                    "btf sidecar lacks 0x9FEB magic; falling back to ELF extraction",
                );
            }
            Err(e) => {
                tracing::warn!(
                    path = %sidecar.display(),
                    err = %e,
                    "btf sidecar read failed; falling back to ELF extraction",
                );
            }
        }
    }

    // Fallback: parse ELF, extract `.BTF` section bytes.
    let elf = goblin::elf::Elf::parse(&data).map_err(|_| {
        anyhow::anyhow!(
            "{}: not recognized as raw BTF (missing 0x9FEB magic) or ELF vmlinux",
            path.display()
        )
    })?;
    let btf_shdr = elf
        .section_headers
        .iter()
        .find(|shdr| elf.shdr_strtab.get_at(shdr.sh_name) == Some(".BTF"));
    let shdr = match btf_shdr {
        Some(s) => s,
        None => bail!("vmlinux ELF has no .BTF section"),
    };
    let offset = shdr.sh_offset as usize;
    let size = shdr.sh_size as usize;
    let btf_data = data
        .get(offset..offset + size)
        .context(".BTF section data out of bounds")?;
    let btf = Btf::from_bytes(btf_data).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Write sidecar on successful parse. Errors are non-fatal — the
    // load succeeds regardless, we just miss the cache on future loads.
    if let Err(e) = write_btf_sidecar(&sidecar, btf_data) {
        tracing::warn!(
            path = %sidecar.display(),
            err = %e,
            "btf sidecar write failed; BTF will be re-extracted from ELF on next load",
        );
    }

    Ok(btf)
}

/// Sidecar path for a given vmlinux path: append `.btf` to the
/// existing filename so the sidecar sits next to vmlinux in the
/// same directory (e.g. `<cache-entry>/vmlinux` →
/// `<cache-entry>/vmlinux.btf`). Using append-suffix rather than
/// `with_extension` preserves any existing extension on the input
/// (uncommon for real vmlinuxes, but robust against paths like
/// `vmlinux.elf`).
fn btf_sidecar_path(path: &Path) -> std::path::PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".btf");
    std::path::PathBuf::from(name)
}

/// True iff `data` begins with the little-endian raw-BTF magic
/// (bytes 0x9F then 0xEB in file order, i.e. the u16 0xEB9F read LE).
///
/// `is_raw_btf` accepts only little-endian BTF; the host
/// architectures ktstr supports are LE, so a big-endian BTF blob
/// here would signal an unsupported configuration. `btf-rs` itself
/// would parse big-endian BTF too — it branches on the magic at
/// `cbtf::btf_header::from_reader` and reads the remaining fields
/// through `Endianness::Big` — but such inputs are not a supported
/// ktstr configuration, so we reject them at the sidecar/magic
/// gate and let the caller see the "not recognized as raw BTF"
/// error from the ELF-parse fallback.
fn is_raw_btf(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0x9F && data[1] == 0xEB
}

/// Is the sidecar at least as new as its vmlinux? Returns false when
/// either file is missing or any mtime cannot be read (safe-default:
/// treat as miss and re-extract from ELF).
fn sidecar_fresh(sidecar: &Path, vmlinux: &Path) -> bool {
    let Ok(sidecar_mtime) = std::fs::metadata(sidecar).and_then(|m| m.modified()) else {
        return false;
    };
    let Ok(vmlinux_mtime) = std::fs::metadata(vmlinux).and_then(|m| m.modified()) else {
        return false;
    };
    sidecar_mtime >= vmlinux_mtime
}

/// Atomically write `bytes` to `sidecar`. Creates a tempfile in the
/// sidecar's parent directory and persists it via rename so
/// concurrent readers either see the old sidecar or the new one,
/// never a partial write.
fn write_btf_sidecar(sidecar: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = sidecar
        .parent()
        .context("btf sidecar path has no parent directory")?;
    let mut tmp =
        tempfile::NamedTempFile::new_in(parent).context("create tempfile for btf sidecar")?;
    tmp.write_all(bytes).context("write btf sidecar contents")?;
    tmp.as_file()
        .sync_all()
        .context("fsync btf sidecar before rename")?;
    tmp.persist(sidecar)
        .map_err(|e| anyhow::anyhow!("persist btf sidecar: {}", e.error))?;
    Ok(())
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
    /// Offsets for scx event counters. Resolved via `scx_sched.pcpu`
    /// (6.18+) or `scx_sched.event_stats_cpu` (6.16-6.17) fallback.
    /// None if BTF lacks both paths.
    pub event_offsets: Option<ScxEventOffsets>,
    /// Offsets for struct rq schedstat fields. None when BTF lacks the
    /// required fields (typically CONFIG_SCHEDSTATS=n).
    pub schedstat_offsets: Option<SchedstatOffsets>,
    /// Offsets for sched_domain tree walking and stats. None if BTF
    /// lacks the `sd` field on `struct rq` or `struct sched_domain`.
    pub sched_domain_offsets: Option<SchedDomainOffsets>,
    /// Offsets for runtime `scx_sched.watchdog_timeout` override. None
    /// if BTF lacks `struct scx_sched` or its `watchdog_timeout` field.
    pub watchdog_offsets: Option<ScxWatchdogOffsets>,
}

/// Byte offsets for overriding `scx_sched.watchdog_timeout` from the host.
/// Applies to 7.1+ kernels where `watchdog_timeout` is a field on the
/// runtime-allocated `scx_sched` struct. On pre-7.1 kernels the timeout
/// is a file-scope static (`scx_watchdog_timeout`), handled separately
/// via [`super::reader::WatchdogOverride::StaticGlobal`].
///
/// The host reads `*scx_root` to find the struct, then writes jiffies
/// at this offset.
#[derive(Debug, Clone)]
pub struct ScxWatchdogOffsets {
    /// Offset of `watchdog_timeout` within `struct scx_sched`.
    pub scx_sched_watchdog_timeout_off: usize,
}

/// Byte offsets for reading scx event counters from guest memory.
///
/// Two kernel layouts are supported:
/// - 6.18+: `scx_sched.pcpu` -> `scx_sched_pcpu.event_stats` (percpu
///   pointer to an intermediate struct containing the stats).
/// - 6.16-6.17: `scx_sched.event_stats_cpu` -> `scx_event_stats`
///   directly (percpu pointer to the stats struct, `event_stats_off`
///   = 0).
///
/// The host resolves per-CPU addresses via `scx_root -> scx_sched`
/// plus `__per_cpu_offset[cpu]`.
#[derive(Debug, Clone)]
pub struct ScxEventOffsets {
    /// Offset of the percpu pointer within `struct scx_sched`.
    /// On 6.18+: offset of `pcpu` (`__percpu *scx_sched_pcpu`).
    /// On 6.16-6.17: offset of `event_stats_cpu` (`__percpu *scx_event_stats`).
    pub percpu_ptr_off: usize,
    /// Offset of `event_stats` within the per-CPU struct.
    /// On 6.18+: offset within `struct scx_sched_pcpu`.
    /// On 6.16-6.17: 0 (the percpu pointer points directly to the stats).
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
    /// Offset of `SCX_EV_REENQ_IMMED` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_reenq_immed: Option<usize>,
    /// Offset of `SCX_EV_REENQ_LOCAL_REPEAT` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_reenq_local_repeat: Option<usize>,
    /// Offset of `SCX_EV_REFILL_SLICE_DFL` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_refill_slice_dfl: Option<usize>,
    /// Offset of `SCX_EV_BYPASS_DURATION` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_bypass_duration: Option<usize>,
    /// Offset of `SCX_EV_BYPASS_DISPATCH` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_bypass_dispatch: Option<usize>,
    /// Offset of `SCX_EV_BYPASS_ACTIVATE` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_bypass_activate: Option<usize>,
    /// Offset of `SCX_EV_INSERT_NOT_OWNED` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_insert_not_owned: Option<usize>,
    /// Offset of `SCX_EV_SUB_BYPASS_DISPATCH` within `struct scx_event_stats`.
    /// None on kernels that predate this field.
    pub ev_sub_bypass_dispatch: Option<usize>,
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
        let watchdog_offsets = resolve_watchdog_offsets(&btf).ok();

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
            watchdog_offsets,
        })
    }
}

/// Resolve BTF offsets for scx event counters.
///
/// Tries the 6.18+ layout first (`scx_sched.pcpu` ->
/// `scx_sched_pcpu.event_stats`), then falls back to the 6.16-6.17
/// layout (`scx_sched.event_stats_cpu` -> `scx_event_stats` directly
/// with `event_stats_off` = 0).
///
/// Returns Err if both paths fail.
fn resolve_event_offsets(btf: &Btf) -> Result<ScxEventOffsets> {
    let (scx_sched_struct, _) = find_struct(btf, "scx_sched")?;

    // Try 6.18+ path: scx_sched.pcpu -> scx_sched_pcpu.event_stats.
    let pcpu_path = member_byte_offset(btf, &scx_sched_struct, "pcpu")
        .ok()
        .and_then(|pcpu_off| {
            let (pcpu_struct, _) = find_struct(btf, "scx_sched_pcpu").ok()?;
            let (stats_off, stats_member) =
                member_byte_offset_with_member(btf, &pcpu_struct, "event_stats").ok()?;
            let stats_struct = resolve_member_struct(btf, &stats_member).ok()?;
            Some((pcpu_off, stats_off, stats_struct))
        });

    // Try 6.16-6.17 path: scx_sched.event_stats_cpu -> scx_event_stats directly.
    let (percpu_ptr_off, event_stats_off, event_stats_struct) = match pcpu_path {
        Some(resolved) => resolved,
        None => {
            let (esc_off, esc_member) =
                member_byte_offset_with_member(btf, &scx_sched_struct, "event_stats_cpu")
                    .context("btf: neither scx_sched.pcpu nor scx_sched.event_stats_cpu found")?;
            let stats_struct = resolve_member_struct(btf, &esc_member)
                .context("btf: resolve type of scx_sched.event_stats_cpu")?;
            // 0: the percpu pointer targets scx_event_stats directly.
            (esc_off, 0, stats_struct)
        }
    };

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

    let ev_reenq_immed = member_byte_offset(btf, &event_stats_struct, "SCX_EV_REENQ_IMMED").ok();
    let ev_reenq_local_repeat =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_REENQ_LOCAL_REPEAT").ok();
    let ev_refill_slice_dfl =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_REFILL_SLICE_DFL")
            .or_else(|_| member_byte_offset(btf, &event_stats_struct, "SCX_EV_ENQ_SLICE_DFL"))
            .ok();
    let ev_bypass_duration =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_BYPASS_DURATION").ok();
    let ev_bypass_dispatch =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_BYPASS_DISPATCH").ok();
    let ev_bypass_activate =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_BYPASS_ACTIVATE").ok();
    let ev_insert_not_owned =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_INSERT_NOT_OWNED").ok();
    let ev_sub_bypass_dispatch =
        member_byte_offset(btf, &event_stats_struct, "SCX_EV_SUB_BYPASS_DISPATCH").ok();

    Ok(ScxEventOffsets {
        percpu_ptr_off,
        event_stats_off,
        ev_select_cpu_fallback,
        ev_dispatch_local_dsq_offline,
        ev_dispatch_keep_last,
        ev_enq_skip_exiting,
        ev_enq_skip_migration_disabled,
        ev_reenq_immed,
        ev_reenq_local_repeat,
        ev_refill_slice_dfl,
        ev_bypass_duration,
        ev_bypass_dispatch,
        ev_bypass_activate,
        ev_insert_not_owned,
        ev_sub_bypass_dispatch,
    })
}

/// Resolve BTF offsets for `scx_sched.watchdog_timeout`.
/// Returns Err if the struct or field is missing (kernel without sched_ext).
fn resolve_watchdog_offsets(btf: &Btf) -> Result<ScxWatchdogOffsets> {
    let (scx_sched_struct, _) = find_struct(btf, "scx_sched")?;
    let scx_sched_watchdog_timeout_off =
        member_byte_offset(btf, &scx_sched_struct, "watchdog_timeout")?;
    Ok(ScxWatchdogOffsets {
        scx_sched_watchdog_timeout_off,
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
/// `max_newidle_lb_cost`) are always present on `struct sched_domain`.
/// `newidle_call`, `newidle_success`, and `newidle_ratio` were removed
/// in 6.16 and are resolved as optional.
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
    /// None on 6.16+ where this field was removed.
    pub sd_newidle_call: Option<usize>,
    /// Offset of `newidle_success` (unsigned int) within `struct sched_domain`.
    /// None on 6.16+ where this field was removed.
    pub sd_newidle_success: Option<usize>,
    /// Offset of `newidle_ratio` (unsigned int) within `struct sched_domain`.
    /// None on 6.16+ where this field was removed.
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

    // Runtime fields.
    let sd_balance_interval = member_byte_offset(btf, &sd_struct, "balance_interval")?;
    let sd_nr_balance_failed = member_byte_offset(btf, &sd_struct, "nr_balance_failed")?;
    let sd_max_newidle_lb_cost = member_byte_offset(btf, &sd_struct, "max_newidle_lb_cost")?;

    // newidle_call/newidle_success/newidle_ratio were removed together
    // in 6.16. Resolve all-or-nothing: if any is missing, set all to None.
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

/// Byte offsets for walking a kernel `struct idr` — the shared
/// allocator that both `map_idr` and `prog_idr` use. Resolved once
/// per BTF load and spliced into [`BpfMapOffsets`] and
/// [`BpfProgOffsets`], each of which used to compute the same four
/// fields independently.
#[derive(Debug, Clone, Copy)]
pub struct IdrOffsets {
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
}

impl IdrOffsets {
    /// Resolve IDR + xa_node offsets from a pre-loaded BTF object.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
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

        Ok(Self {
            xa_node_slots,
            xa_node_shift,
            idr_xa_head,
            idr_next,
        })
    }
}

/// Byte offsets within kernel BPF structures needed for host-side
/// BPF map discovery and value access.
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
    /// All-zero offsets. Useful for tests that exercise functions which
    /// do not need any `BpfMapOffsets` field — e.g. the `write_value`
    /// / `read_value` path, which only walks page tables to reach the
    /// value KVA and never reads an offset from `self`. Production code
    /// must use [`from_vmlinux`](Self::from_vmlinux) or
    /// [`from_btf`](Self::from_btf).
    #[cfg(all(test, target_arch = "x86_64"))]
    pub(crate) const EMPTY: Self = Self {
        map_name: 0,
        map_type: 0,
        map_flags: 0,
        key_size: 0,
        value_size: 0,
        max_entries: 0,
        array_value: 0,
        xa_node_slots: 0,
        xa_node_shift: 0,
        idr_xa_head: 0,
        idr_next: 0,
        map_btf: 0,
        map_btf_value_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        htab_offsets: None,
    };

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

        let idr = IdrOffsets::from_btf(btf)?;

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
            xa_node_slots: idr.xa_node_slots,
            xa_node_shift: idr.xa_node_shift,
            idr_xa_head: idr.idr_xa_head,
            idr_next: idr.idr_next,
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
#[allow(dead_code)]
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

        let idr = IdrOffsets::from_btf(btf)?;

        let prog_stats = member_byte_offset(btf, &bpf_prog, "stats")?;

        let (bpf_prog_stats, _) = find_struct(btf, "bpf_prog_stats")?;
        let stats_cnt = member_byte_offset(btf, &bpf_prog_stats, "cnt")?;
        let stats_nsecs = member_byte_offset(btf, &bpf_prog_stats, "nsecs")?;

        Ok(Self {
            prog_type,
            prog_aux,
            aux_verified_insns,
            aux_name,
            xa_node_slots: idr.xa_node_slots,
            xa_node_shift: idr.xa_node_shift,
            idr_xa_head: idr.idr_xa_head,
            idr_next: idr.idr_next,
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
        let offsets = crate::test_support::require_kernel_offsets(&path);
        assert_ne!(
            offsets.rq_nr_running, offsets.rq_clock,
            "rq_nr_running and rq_clock offsets must be distinct"
        );
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
        let offsets = crate::test_support::require_kernel_offsets(&path);
        // Event offsets are optional — only assert if present.
        if let Some(ev) = &offsets.event_offsets {
            // All event counter fields are s64, so offsets must differ.
            let mut all = vec![
                ev.ev_select_cpu_fallback,
                ev.ev_dispatch_local_dsq_offline,
                ev.ev_dispatch_keep_last,
                ev.ev_enq_skip_exiting,
                ev.ev_enq_skip_migration_disabled,
            ];
            for off in [
                ev.ev_reenq_immed,
                ev.ev_reenq_local_repeat,
                ev.ev_refill_slice_dfl,
                ev.ev_bypass_duration,
                ev.ev_bypass_dispatch,
                ev.ev_bypass_activate,
                ev.ev_insert_not_owned,
                ev.ev_sub_bypass_dispatch,
            ]
            .into_iter()
            .flatten()
            {
                all.push(off);
            }
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
        let offsets = crate::test_support::require_kernel_offsets(&path);
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
        let offsets = crate::test_support::require_kernel_offsets(&path);
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
            // Runtime fields that are always present must be at nonzero offsets.
            let always_present = [
                sd.sd_balance_interval,
                sd.sd_nr_balance_failed,
                sd.sd_max_newidle_lb_cost,
            ];
            for &off in &always_present {
                assert!(off > 0, "sched_domain runtime field offset must be nonzero");
            }
            // newidle_call/newidle_success/newidle_ratio are optional (removed in 6.16).
            // When present, they must be at nonzero offsets.
            for off in [
                sd.sd_newidle_call,
                sd.sd_newidle_success,
                sd.sd_newidle_ratio,
            ]
            .into_iter()
            .flatten()
            {
                assert!(
                    off > 0,
                    "optional newidle field offset must be nonzero when present"
                );
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
        let offsets = crate::test_support::require_bpf_map_offsets(&path);
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
        let offsets = crate::test_support::require_bpf_prog_offsets(&path);
        assert!(offsets.prog_aux > 0);
        assert!(offsets.aux_verified_insns > 0);
        assert!(offsets.aux_name > 0);
    }

    /// Validate that optional BTF offsets (watchdog, event) are
    /// internally consistent.
    ///
    /// `watchdog_offsets` requires the post-refactor `scx_sched` layout
    /// (with `watchdog_timeout` field). `event_offsets` can resolve via
    /// either path (6.18+ `pcpu` or 6.16-6.17 `event_stats_cpu`).
    /// `watchdog_offsets` being present implies `event_offsets` is also
    /// present, but not vice versa.
    ///
    /// Assertions that overlap with parse_rq_offsets_from_vmlinux and
    /// parse_event_offsets_from_vmlinux are intentionally omitted.
    #[test]
    fn btf_optional_offsets_consistent() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let offsets = match KernelOffsets::from_vmlinux(&path) {
            Ok(o) => o,
            Err(e) => skip!("vmlinux BTF resolution failed: {e}"),
        };

        assert_ne!(
            offsets.rq_nr_running, offsets.rq_scx,
            "rq_nr_running and rq_scx offsets must be distinct"
        );

        if let Some(ref ev) = offsets.event_offsets {
            assert!(ev.percpu_ptr_off > 0);
        }

        if let Some(ref wd) = offsets.watchdog_offsets {
            assert!(
                wd.scx_sched_watchdog_timeout_off > 0,
                "watchdog_timeout offset must be nonzero within scx_sched"
            );
            assert!(
                offsets.event_offsets.is_some(),
                "watchdog_offsets present implies event_offsets must also resolve"
            );
        }
    }

    #[test]
    fn from_vmlinux_nonexistent() {
        let path = std::path::Path::new("/nonexistent/vmlinux");
        assert!(KernelOffsets::from_vmlinux(path).is_err());
    }

    #[test]
    fn from_vmlinux_empty_file() {
        let dir = std::env::temp_dir().join(format!("ktstr-btf-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("vmlinux");
        std::fs::write(&f, b"").unwrap();
        assert!(KernelOffsets::from_vmlinux(&f).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -- BTF sidecar cache --
    //
    // These tests exercise the `<path>.btf` sidecar pipeline: pure
    // helper functions (path derivation, magic check, freshness)
    // directly, plus end-to-end load_btf_from_path behavior against a
    // real-ELF fixture when one is available on the host.

    #[test]
    fn btf_sidecar_path_appends_dot_btf() {
        let p = std::path::Path::new("/cache/vmlinux");
        assert_eq!(
            btf_sidecar_path(p),
            std::path::PathBuf::from("/cache/vmlinux.btf"),
        );
    }

    #[test]
    fn btf_sidecar_path_preserves_existing_extension() {
        // Append-suffix semantics, NOT `with_extension` which would
        // replace `.elf` with `.btf`.
        let p = std::path::Path::new("/cache/vmlinux.elf");
        assert_eq!(
            btf_sidecar_path(p),
            std::path::PathBuf::from("/cache/vmlinux.elf.btf"),
        );
    }

    #[test]
    fn is_raw_btf_accepts_little_endian_magic() {
        // Little-endian BTF begins with bytes 0x9F 0xEB in file
        // order. `is_raw_btf` accepts only little-endian BTF: the
        // host architectures ktstr supports are LE, so a big-endian
        // BTF blob is an unsupported configuration even though
        // btf-rs itself could parse it (see the sibling
        // `is_raw_btf_rejects_wrong_magic_and_short_input` where
        // the BE magic is explicitly rejected).
        assert!(is_raw_btf(&[0x9F, 0xEB, 0x01, 0x00]));
    }

    #[test]
    fn is_raw_btf_rejects_wrong_magic_and_short_input() {
        // ELF magic — bytes-wise distinct from BTF magic.
        assert!(!is_raw_btf(&[0x7F, b'E', b'L', b'F']));
        // Big-endian BTF magic: file-order bytes 0xEB 0x9F. btf-rs
        // itself would parse such a blob (branches on the magic at
        // cbtf::btf_header::from_reader), but ktstr supports only
        // LE hosts, so `is_raw_btf` deliberately rejects BE and
        // lets the caller surface "not recognized as raw BTF" via
        // the ELF-parse fallback.
        assert!(!is_raw_btf(&[0xEB, 0x9F, 0x01, 0x00]));
        // Too short to carry the 2-byte magic.
        assert!(!is_raw_btf(&[0x9F]));
        assert!(!is_raw_btf(&[]));
    }

    #[test]
    fn sidecar_fresh_false_when_either_file_missing() {
        let dir =
            std::env::temp_dir().join(format!("ktstr-btf-sidecar-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vmlinux = dir.join("vmlinux");
        let sidecar = dir.join("vmlinux.btf");
        std::fs::write(&vmlinux, b"vmlinux-bytes").unwrap();
        // sidecar missing → not fresh
        assert!(!sidecar_fresh(&sidecar, &vmlinux));
        std::fs::write(&sidecar, b"cached-btf").unwrap();
        // both present → fresh (sidecar written after vmlinux)
        assert!(sidecar_fresh(&sidecar, &vmlinux));
        // vmlinux missing → not fresh (safe default)
        std::fs::remove_file(&vmlinux).unwrap();
        assert!(!sidecar_fresh(&sidecar, &vmlinux));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end: first load extracts BTF from ELF vmlinux and
    /// writes the sidecar; second load reads the sidecar bytes
    /// directly and parses them. Exercises both branches of
    /// `load_btf_from_path` against a real vmlinux.
    ///
    /// Skipped when no test vmlinux is available or when
    /// `find_test_vmlinux` resolves to raw BTF (sysfs), which
    /// exercises a different branch that never writes a sidecar.
    #[test]
    fn load_btf_writes_sidecar_then_hits_cache_on_second_load() {
        use std::time::Duration;

        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            // Raw BTF input never writes a sidecar — wrong branch.
            return;
        }

        // Run against a fresh temp copy so the test does not mutate
        // the shared kernel cache or source tree. The copy shares
        // the same directory so the sidecar's tempfile+rename lands
        // on the same filesystem.
        let dir =
            std::env::temp_dir().join(format!("ktstr-btf-sidecar-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vmlinux = dir.join("vmlinux");
        std::fs::copy(&path, &vmlinux).unwrap();
        let sidecar = dir.join("vmlinux.btf");
        // Ensure vmlinux mtime is strictly less than whatever the
        // sidecar write will stamp — avoids a same-second tie that
        // could false-pass the freshness check on low-resolution
        // filesystems.
        std::thread::sleep(Duration::from_millis(10));

        // Pre-state: no sidecar exists.
        assert!(
            !sidecar.exists(),
            "precondition: sidecar should not exist before first load",
        );

        // First load: extract + write sidecar.
        let btf1 = load_btf_from_path(&vmlinux).expect("first load must succeed");
        // Consume btf1 so the optimizer cannot elide the parse.
        let _ = format!("{:?}", btf1.resolve_types_by_name("task_struct").is_ok());
        assert!(
            sidecar.exists(),
            "first load must write sidecar at {}",
            sidecar.display(),
        );
        let sidecar_bytes = std::fs::read(&sidecar).unwrap();
        assert!(
            is_raw_btf(&sidecar_bytes),
            "sidecar contents must carry the raw BTF 0x9FEB magic",
        );

        // Sanity: sidecar mtime is at/after vmlinux mtime so the
        // freshness check on the second load picks it up.
        assert!(
            sidecar_fresh(&sidecar, &vmlinux),
            "sidecar mtime must be ≥ vmlinux mtime after first load",
        );

        // Second load: should hit the cache. We verify by deleting
        // the ELF so that any fallback to ELF parsing would fail —
        // but wait, the function reads the ELF path first for its
        // bytes; deletion would break even the sidecar branch
        // (since the function reads `path` unconditionally at the
        // top). Instead, pin the behavior by checking that a second
        // load still succeeds AND the sidecar mtime is unchanged
        // (a second write would bump it).
        let sidecar_mtime_before = std::fs::metadata(&sidecar).unwrap().modified().unwrap();
        // Sleep a bit so a spurious sidecar rewrite would be
        // detectable via an mtime bump.
        std::thread::sleep(Duration::from_millis(50));
        let btf2 = load_btf_from_path(&vmlinux).expect("second load must succeed");
        let _ = format!("{:?}", btf2.resolve_types_by_name("task_struct").is_ok());
        let sidecar_mtime_after = std::fs::metadata(&sidecar).unwrap().modified().unwrap();
        assert_eq!(
            sidecar_mtime_before, sidecar_mtime_after,
            "second load must hit sidecar cache — mtime bump proves a \
             redundant rewrite",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Simulating a stale sidecar by making its mtime older than
    /// vmlinux's: the next load must ignore the cached bytes and
    /// re-extract from ELF, then overwrite the sidecar. Exercises
    /// the `mtime(sidecar) < mtime(vmlinux)` staleness guard.
    #[test]
    fn load_btf_rejects_stale_sidecar() {
        use std::time::{Duration, SystemTime};

        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let dir =
            std::env::temp_dir().join(format!("ktstr-btf-sidecar-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vmlinux = dir.join("vmlinux");
        std::fs::copy(&path, &vmlinux).unwrap();
        let sidecar = dir.join("vmlinux.btf");

        // Plant a sidecar that predates vmlinux by writing garbage
        // and stamping its mtime into the past. `set_times` is the
        // portable way to force a past mtime.
        std::fs::write(&sidecar, b"stale-sidecar-bytes").unwrap();
        let past = SystemTime::now() - Duration::from_secs(3600);
        let f = std::fs::File::options().write(true).open(&sidecar).unwrap();
        f.set_modified(past).unwrap();
        drop(f);

        // Precondition: sidecar is older than vmlinux.
        assert!(
            !sidecar_fresh(&sidecar, &vmlinux),
            "precondition: planted sidecar must be stale",
        );

        let btf = load_btf_from_path(&vmlinux)
            .expect("load must succeed via ELF fallback despite stale sidecar");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        // Post-condition: sidecar has been overwritten with fresh
        // BTF bytes (must now start with the BTF magic, not the
        // garbage we planted).
        let sidecar_bytes = std::fs::read(&sidecar).unwrap();
        assert!(
            is_raw_btf(&sidecar_bytes),
            "load must overwrite stale sidecar with fresh BTF bytes",
        );
        assert!(
            sidecar_fresh(&sidecar, &vmlinux),
            "sidecar must be fresh again after re-extraction",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Sidecar with correct mtime but garbage contents (no 0x9FEB
    /// magic): the load must recover by falling through to ELF
    /// extraction and overwriting the corrupt sidecar. Exercises
    /// the "fresh but lacks magic" branch of the match inside
    /// `load_btf_from_path`.
    #[test]
    fn load_btf_recovers_from_corrupt_sidecar() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let dir =
            std::env::temp_dir().join(format!("ktstr-btf-sidecar-corrupt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vmlinux = dir.join("vmlinux");
        std::fs::copy(&path, &vmlinux).unwrap();
        let sidecar = dir.join("vmlinux.btf");
        // Plant a sidecar that is newer than vmlinux but whose
        // contents do not carry the BTF magic.
        std::fs::write(&sidecar, b"not-btf-bytes").unwrap();
        assert!(
            sidecar_fresh(&sidecar, &vmlinux),
            "precondition: planted sidecar must be mtime-fresh",
        );

        let btf = load_btf_from_path(&vmlinux)
            .expect("load must recover when sidecar is fresh-but-corrupt");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        // Corrupt sidecar should have been overwritten.
        let sidecar_bytes = std::fs::read(&sidecar).unwrap();
        assert!(
            is_raw_btf(&sidecar_bytes),
            "corrupt sidecar must be overwritten on next load",
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// When the sidecar would be written to a read-only directory,
    /// the load must still succeed — sidecar writes are
    /// best-effort and never surface as errors. Exercises the
    /// tracing::warn fallback in `write_btf_sidecar`'s error path.
    #[test]
    #[cfg(unix)]
    fn load_btf_survives_readonly_sidecar_dir() {
        use std::os::unix::fs::PermissionsExt;

        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }
        // Root skips DAC permission checks entirely, so a
        // read-only directory still lets root write inside. The
        // test cannot distinguish "sidecar write skipped due to
        // best-effort" from "sidecar write succeeded because we
        // are root" — skip under euid 0 to avoid a false-pass on
        // CI runners that sandbox as root.
        if unsafe { libc::geteuid() } == 0 {
            return;
        }

        let dir = std::env::temp_dir().join(format!("ktstr-btf-sidecar-ro-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vmlinux = dir.join("vmlinux");
        std::fs::copy(&path, &vmlinux).unwrap();
        // Mark directory read-only after the vmlinux is in place.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();

        // Load must succeed despite the sidecar write failing.
        let btf = load_btf_from_path(&vmlinux)
            .expect("load must succeed even when sidecar dir is read-only");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        // Sidecar must not exist — write should have failed.
        let sidecar = dir.join("vmlinux.btf");
        assert!(
            !sidecar.exists(),
            "sidecar must not exist after write to read-only dir",
        );

        // Restore permissions for cleanup.
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Raw-BTF inputs (files that already carry 0x9FEB magic) must
    /// never have a sidecar written alongside them — the input file
    /// IS the BTF blob, and a sidecar would be a byte-for-byte
    /// copy of itself. Exercises the raw-BTF early-return branch.
    #[test]
    fn load_btf_skips_sidecar_for_raw_btf_input() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if !path.starts_with("/sys/") {
            // Generate a raw-BTF file from the ELF so this test
            // exercises the raw-BTF path even when
            // find_test_vmlinux returns an ELF.
            let dir =
                std::env::temp_dir().join(format!("ktstr-btf-sidecar-raw-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let src_data = std::fs::read(&path).unwrap();
            let elf = match goblin::elf::Elf::parse(&src_data) {
                Ok(e) => e,
                Err(_) => {
                    // Raw BTF already — skip the ELF extraction.
                    let raw = dir.join("vmlinux.btf-raw");
                    std::fs::copy(&path, &raw).unwrap();
                    let _ = load_btf_from_path(&raw).expect("raw-BTF load must succeed");
                    let sidecar = btf_sidecar_path(&raw);
                    assert!(
                        !sidecar.exists(),
                        "raw-BTF input must not produce a sidecar",
                    );
                    let _ = std::fs::remove_dir_all(&dir);
                    return;
                }
            };
            let btf_shdr = elf
                .section_headers
                .iter()
                .find(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some(".BTF"));
            let shdr = match btf_shdr {
                Some(s) => s,
                None => {
                    let _ = std::fs::remove_dir_all(&dir);
                    return;
                }
            };
            let offset = shdr.sh_offset as usize;
            let size = shdr.sh_size as usize;
            let raw_bytes = &src_data[offset..offset + size];
            let raw = dir.join("vmlinux.btf-raw");
            std::fs::write(&raw, raw_bytes).unwrap();
            let _ = load_btf_from_path(&raw).expect("raw-BTF load must succeed");
            // The sidecar would be at `<raw>.btf` — must NOT exist.
            let sidecar = btf_sidecar_path(&raw);
            assert!(
                !sidecar.exists(),
                "raw-BTF input must not produce a sidecar at {}",
                sidecar.display(),
            );
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        // /sys/kernel/btf/vmlinux path: raw BTF on a read-only
        // filesystem. Any sidecar write would fail anyway, and the
        // sidecar path itself ("/sys/kernel/btf/vmlinux.btf") is
        // not writable — we cannot assert much here beyond "load
        // must succeed," which the pre-existing tests already
        // cover.
    }
}
