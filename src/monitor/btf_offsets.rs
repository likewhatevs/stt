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
/// For ELF inputs whose path lies inside the ktstr kernel cache root
/// (see [`crate::cache::path_inside_cache_root`]), the extracted
/// `.BTF` section bytes are cached as a sibling file at `<path>.btf`
/// (e.g. `vmlinux` → `vmlinux.btf`). On subsequent loads, if the
/// sidecar exists and its mtime is greater than or equal to the
/// vmlinux mtime, the cached bytes are read and parsed directly,
/// skipping the goblin ELF parse + `.BTF` section extraction.
///
/// The sidecar is written lazily on first load after a cache miss,
/// gated on the same membership check. Write failures (e.g. read-only
/// directory) are logged at `tracing::warn` level and do not fail
/// the load — the function falls through with the freshly-parsed
/// BTF. Raw-BTF inputs never write a sidecar: the input file IS the
/// BTF blob and a sidecar would just be a redundant copy of itself.
///
/// Vmlinuxes resolved from outside the cache — kernel source trees
/// (the `<root>/vmlinux` walk-up in [`crate::vmm::find_vmlinux`])
/// and distro debug paths (`/usr/lib/debug/boot/...`,
/// `/lib/modules/<v>/build/vmlinux`) — get neither sidecar reads nor
/// writes. The BTF is re-extracted from ELF on every load. Caching
/// would otherwise pollute directories the cache does not own; a
/// repeated extract is fast relative to VM boot times so the cost
/// is acceptable.
///
/// Staleness: mtime-based, no content hash. `CacheDir::store`'s
/// atomic rename path bumps vmlinux mtime when an entry is replaced,
/// so a previously-written sidecar next to the old vmlinux surfaces
/// as stale (`mtime(sidecar) < mtime(vmlinux)`) and the bytes are
/// re-extracted + re-written on the next load.
pub(crate) fn load_btf_from_path(path: &Path) -> Result<Btf> {
    let data = std::fs::read(path).context("read file")?;
    // Raw BTF: first 2 bytes are the 0x9FEB magic. Parse directly;
    // never write a sidecar (would be a byte-for-byte self-copy).
    if is_raw_btf(&data) {
        return Btf::from_bytes(&data).map_err(|e| anyhow::anyhow!("{e}"));
    }

    // Canonicalize the input path before deriving sidecar artifacts.
    // Both flows that the membership gate must handle correctly
    // depend on this normalization:
    //   (a) Symlink in cache pointing to a source-tree real file
    //       (`/cache/entry/vmlinux` -> `/source-tree/vmlinux`)
    //       would otherwise pass the lexical membership check (the
    //       cache-side parent canonicalizes into the cache) and
    //       deposit a stale-prone sidecar at
    //       `/cache/entry/vmlinux.btf`. The sidecar's mtime tracks
    //       the symlink's target, so an in-place rebuild of the
    //       source-tree real file silently desynchronizes the
    //       cached sidecar.
    //   (b) Symlink in source tree pointing into cache
    //       (`/source-tree/vmlinux` -> `/cache/entry/vmlinux`)
    //       would otherwise fail the membership check (the
    //       source-tree parent canonicalizes outside the cache)
    //       and miss the sidecar cache for what is, after
    //       resolution, a genuine cache entry.
    // Canonicalize-at-top normalizes both flows to use the real
    // file's path: (a) collapses to "outside cache" and suppresses
    // the sidecar; (b) collapses to "inside cache" and writes the
    // sidecar next to the real file in the cache.
    //
    // The fs::read above proves the file is reachable; canonicalize
    // can still fail under EACCES on a parent component or a race
    // with a disappearing symlink target. Any canonicalize failure
    // suppresses the sidecar entirely — without a canonical path
    // there is no way to prove the input is inside the cache, and
    // writing a `<lexical-path>.btf` next to an unresolvable input
    // is exactly the source-tree pollution the gate exists to
    // prevent.
    let (canon_path, sidecar_allowed) = match std::fs::canonicalize(path) {
        Ok(c) => {
            let inside = crate::cache::path_inside_cache_root(&c);
            (c, inside)
        }
        Err(e) => {
            tracing::debug!(
                path = %path.display(),
                err = %e,
                "btf input path canonicalize failed; sidecar suppressed for this load",
            );
            (path.to_path_buf(), false)
        }
    };
    // Sidecar reads and writes are gated on cache-root membership:
    // source trees, distro debug paths, and other non-cache inputs
    // get neither, ensuring ktstr never deposits sibling artifacts
    // in directories it does not own. Resolved on every call so a
    // mid-process `KTSTR_CACHE_DIR` change is honored.
    let sidecar = btf_sidecar_path(&canon_path);

    if sidecar_allowed {
        if sidecar_fresh(&sidecar, &canon_path) {
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
    } else {
        tracing::debug!(
            path = %canon_path.display(),
            "btf sidecar suppressed: vmlinux path is outside the cache root",
        );
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
    let btf_data = offset
        .checked_add(size)
        .and_then(|end| data.get(offset..end))
        .context(".BTF section data out of bounds")?;
    let btf = Btf::from_bytes(btf_data).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Write sidecar on successful parse, gated on cache-root
    // membership. Errors are non-fatal — the load succeeds
    // regardless, we just miss the cache on future loads. Outside
    // the cache the write is suppressed so source-tree and distro
    // paths remain pristine.
    if sidecar_allowed && let Err(e) = write_btf_sidecar(&sidecar, btf_data) {
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

/// Resolve the byte offset of a named global within a BTF section.
///
/// Walks every `BTF_KIND_DATASEC` whose name matches `section_name`
/// (e.g. `".bss"`, `".data"`, `".rodata"`) and returns the
/// `VarSecinfo.offset()` whose chained `Var` resolves to a name
/// matching `var_name`. Returns `None` when the section, the var, or
/// the chained type chain is missing — the caller should then fall
/// back to a known offset (the freeze coordinator falls back to 0
/// during early boot before program BTF is loadable).
///
/// First-match-by-var-name across Datasecs: the walk visits all
/// Datasecs whose name matches `section_name` (libbpf normally emits
/// at most one) and returns on the first matching var, so callers
/// see one offset even if a future libbpf change splits a section
/// into multiple Datasecs.
///
/// Lives next to [`find_struct`] / [`member_byte_offset`] because
/// it's a generic BTF helper, not specific to the dump path. Both
/// the dump renderer and the freeze coordinator consume it.
pub(crate) fn resolve_var_offset_in_section(
    btf: &Btf,
    section_name: &str,
    var_name: &str,
) -> Option<u32> {
    // Iterate every type by name to find the Datasec for the
    // requested section. `resolve_types_by_name` returns Err on no
    // hit — propagate that as None.
    let candidates = btf.resolve_types_by_name(section_name).ok()?;
    for ty in candidates {
        let Type::Datasec(ds) = ty else { continue };
        for var_info in &ds.variables {
            let Ok(chained) = btf.resolve_chained_type(var_info) else {
                continue;
            };
            let Type::Var(var) = chained else { continue };
            let Ok(name) = btf.resolve_name(&var) else {
                continue;
            };
            if name == var_name {
                return Some(var_info.offset());
            }
        }
    }
    None
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
    /// Offset of `btf_key_type_id` (u32) within `struct bpf_map`.
    /// Hash maps render their keys via this type id when present;
    /// ARRAY/PERCPU_ARRAY use it only for the (synthetic) `__u32 key`
    /// that BPF imposes on those map types.
    pub map_btf_key_type_id: usize,
    /// Offset of `data` pointer within `struct btf`.
    pub btf_data: usize,
    /// Offset of `data_size` (u32) within `struct btf`.
    pub btf_data_size: usize,
    /// Offset of `base_btf` (`struct btf *`) within `struct btf`.
    /// `NULL` for a base BTF (e.g. vmlinux's own); non-null for split
    /// BTF (e.g. a BPF program's BTF whose types extend the kernel
    /// vmlinux BTF). Drives the `Btf::from_bytes` vs
    /// `Btf::from_split_bytes(blob, &base)` choice in the
    /// program-BTF loader.
    pub btf_base_btf: usize,
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
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
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
        let map_btf_key_type_id = member_byte_offset(btf, &bpf_map, "btf_key_type_id")?;

        let (btf_struct, _) = find_struct(btf, "btf")?;
        let btf_data = member_byte_offset(btf, &btf_struct, "data")?;
        let btf_data_size = member_byte_offset(btf, &btf_struct, "data_size")?;
        let btf_base_btf = member_byte_offset(btf, &btf_struct, "base_btf")?;

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
            map_btf_key_type_id,
            btf_data,
            btf_data_size,
            btf_base_btf,
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
    /// Offset of `misses` (u64_stats_t) within `struct bpf_prog_stats`.
    /// Incremented from `bpf_prog_inc_misses_counter` (kernel/bpf/syscall.c)
    /// when a recursion-protected program is re-entered on the same CPU,
    /// so the runtime profile shows skipped invocations alongside
    /// `cnt`/`nsecs`.
    pub stats_misses: usize,
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
        let stats_misses = member_byte_offset(btf, &bpf_prog_stats, "misses")?;

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
            stats_misses,
        })
    }

    /// Parse BTF from a vmlinux ELF and resolve BPF program field offsets.
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let btf =
            load_btf_from_path(path).with_context(|| format!("btf: open {}", path.display()))?;
        Self::from_btf(&btf)
    }
}

// ===========================================================================
// Per-struct offset sub-groups
//
// Each group below resolves the field offsets for ONE kernel struct from
// BTF. Higher-level offset structs (RunnableScanOffsets,
// TaskEnrichmentOffsets, ScxWalkerOffsets) compose these groups so:
//
//   1. Each kernel field is resolved exactly once across the codebase
//      (deduplicated source of truth — task #45).
//   2. Higher-level structs that need only some groups can degrade
//      gracefully: a missing `scx_sched_pnode` group blinds the global
//      DSQ walk pass but leaves rq->scx + per-CPU local DSQ walks
//      working (graceful degradation — task #43).
//
// Convention: each sub-group's `from_btf` is all-or-nothing for ITS
// struct (one field missing -> `Err`). Composers store the result as
// `Result<Sub, String>` (or `Option<Sub>`) so a per-struct failure
// doesn't poison unrelated walks.
// ===========================================================================

/// Field offsets within `struct rq` (`kernel/sched/sched.h`).
///
/// Captures the two fields the host-side scx walker dereferences off a
/// `struct rq` pointer: `scx` (the embedded `struct scx_rq`) and
/// `curr` (the currently-running `struct task_struct *`).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // wired via ScxWalkerOffsets; stays alive once the
                    // freeze coordinator populates ScxWalkerCapture.
pub struct RqStructOffsets {
    /// Offset of `scx` (struct scx_rq) within `struct rq`.
    /// Same value as [`KernelOffsets::rq_scx`]; resolved here so the
    /// scx walker doesn't have to depend on `KernelOffsets`.
    pub scx: usize,
    /// Offset of `curr` (`struct task_struct *`) within `struct rq`.
    pub curr: usize,
}

impl RqStructOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (rq, _) = find_struct(btf, "rq")?;
        Ok(Self {
            scx: member_byte_offset(btf, &rq, "scx")?,
            curr: member_byte_offset(btf, &rq, "curr")?,
        })
    }
}

/// Field offsets within `struct scx_rq` (`kernel/sched/sched.h`).
///
/// The fields are split between two consumers:
///
/// - **Read by `scx_dump_state` (kernel/sched/ext.c)** — the
///   per-CPU "CPU states" section emits these directly:
///   `runnable_list` (walked via `list_for_each_entry`),
///   `nr_running`, `flags`, `cpu_released`, `ops_qseq`,
///   `kick_sync`, plus `local_dsq` indirectly (its `.nr` is
///   read in the dispatch-decision path elsewhere). The
///   host-side rq->scx walker mirrors that output and needs
///   each of these offsets.
/// - **Read by other ext.c paths, NOT `scx_dump_state`** —
///   `nr_immed` is incremented / decremented / inspected in
///   the SCX_ENQ_IMMED enqueue path (`do_enqueue_task` and the
///   ENQ_IMMED branches) but is NOT rendered by
///   `scx_dump_state`. ktstr collects the offset because the
///   host-side dumper exposes the same scalar to operators
///   for ENQ_IMMED diagnosis even though the kernel's own
///   debug dump omits it. `clock` is similar — read by
///   in-kernel scheduling paths but not by `scx_dump_state`;
///   ktstr surfaces it for cross-CPU clock-skew analysis.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxRqOffsets {
    /// Offset of `local_dsq` (struct scx_dispatch_q). Read by
    /// `scx_dump_state` indirectly via local-DSQ depth queries.
    pub local_dsq: usize,
    /// Offset of `runnable_list` (struct list_head) — head of
    /// the per-CPU runnable task list. Walked by the
    /// runnable_at scanner and the rq->scx walker; rendered
    /// per-CPU by `scx_dump_state` via `list_for_each_entry`.
    pub runnable_list: usize,
    /// Offset of `nr_running` (u32). Rendered by
    /// `scx_dump_state` ("CPU %d: nr_run=%u …").
    pub nr_running: usize,
    /// Offset of `flags` (u32). Rendered by `scx_dump_state`
    /// ("flags=0x%x").
    pub flags: usize,
    /// Offset of `cpu_released` (bool). Rendered by
    /// `scx_dump_state` ("cpu_rel=%d").
    pub cpu_released: usize,
    /// Offset of `ops_qseq` (unsigned long). Rendered by
    /// `scx_dump_state` ("ops_qseq=%lu").
    pub ops_qseq: usize,
    /// Offset of `kick_sync` (unsigned long). Rendered by
    /// `scx_dump_state` ("ksync=%lu").
    pub kick_sync: usize,
    /// Offset of `nr_immed` (u32). NOT read by
    /// `scx_dump_state`; ktstr collects it for the host-side
    /// dump's ENQ_IMMED diagnosis path (kernel updates the
    /// counter in `do_enqueue_task` ENQ_IMMED branches and
    /// elsewhere in `kernel/sched/ext.c`).
    pub nr_immed: usize,
    /// Offset of `clock` (u64). NOT read by `scx_dump_state`;
    /// surfaced by ktstr for cross-CPU clock-skew analysis.
    pub clock: usize,
}

impl ScxRqOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (scx_rq, _) = find_struct(btf, "scx_rq")?;
        Ok(Self {
            local_dsq: member_byte_offset(btf, &scx_rq, "local_dsq")?,
            runnable_list: member_byte_offset(btf, &scx_rq, "runnable_list")?,
            nr_running: member_byte_offset(btf, &scx_rq, "nr_running")?,
            flags: member_byte_offset(btf, &scx_rq, "flags")?,
            cpu_released: member_byte_offset(btf, &scx_rq, "cpu_released")?,
            ops_qseq: member_byte_offset(btf, &scx_rq, "ops_qseq")?,
            kick_sync: member_byte_offset(btf, &scx_rq, "kick_sync")?,
            nr_immed: member_byte_offset(btf, &scx_rq, "nr_immed")?,
            clock: member_byte_offset(btf, &scx_rq, "clock")?,
        })
    }
}

/// Universal-subset field offsets within `struct task_struct`. The
/// fields here are read by every walker that touches a task — the
/// runnable scanner, the rq->scx walker, the DSQ walker, and the
/// task_enrichment walker. Resolved once and shared so
/// `task_struct.scx` etc. exist as a single source of truth (task #45).
///
/// Walkers that need additional task_struct fields (priority, signal,
/// stack...) compose [`TaskStructEnrichmentOffsets`] alongside this.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct TaskStructCoreOffsets {
    /// Offset of `comm` (char[16]).
    pub comm: usize,
    /// Offset of `pid` (pid_t == int).
    pub pid: usize,
    /// Offset of `scx` (struct sched_ext_entity).
    pub scx: usize,
}

impl TaskStructCoreOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (task_struct, _) = find_struct(btf, "task_struct")?;
        Ok(Self {
            comm: member_byte_offset(btf, &task_struct, "comm")?,
            pid: member_byte_offset(btf, &task_struct, "pid")?,
            scx: member_byte_offset(btf, &task_struct, "scx")?,
        })
    }
}

/// Extended `struct task_struct` field offsets used by the
/// task_enrichment walker. Composed alongside [`TaskStructCoreOffsets`]
/// in [`TaskEnrichmentOffsets`] — the universal three (`comm`, `pid`,
/// `scx`) live in the core struct; everything below here is
/// enrichment-specific.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct TaskStructEnrichmentOffsets {
    /// Offset of `tgid` (pid_t == int).
    pub tgid: usize,
    /// Offset of `prio` (int).
    pub prio: usize,
    /// Offset of `static_prio` (int).
    pub static_prio: usize,
    /// Offset of `normal_prio` (int).
    pub normal_prio: usize,
    /// Offset of `rt_priority` (unsigned int).
    pub rt_priority: usize,
    /// Offset of `sched_class` (`const struct sched_class *`).
    pub sched_class: usize,
    /// Offset of `core_cookie` (unsigned long). `None` when the kernel
    /// was built without `CONFIG_SCHED_CORE` (BTF omits the field).
    pub core_cookie: Option<usize>,
    /// Offset of `real_parent` (`struct task_struct __rcu *`).
    pub real_parent: usize,
    /// Offset of `group_leader` (`struct task_struct *`).
    pub group_leader: usize,
    /// Offset of `signal` (`struct signal_struct *`).
    pub signal: usize,
    /// Offset of `stack` (`void *`).
    pub stack: usize,
    /// Offset of `nvcsw` (unsigned long).
    pub nvcsw: usize,
    /// Offset of `nivcsw` (unsigned long).
    pub nivcsw: usize,
}

impl TaskStructEnrichmentOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (task_struct, _) = find_struct(btf, "task_struct")?;
        Ok(Self {
            tgid: member_byte_offset(btf, &task_struct, "tgid")?,
            prio: member_byte_offset(btf, &task_struct, "prio")?,
            static_prio: member_byte_offset(btf, &task_struct, "static_prio")?,
            normal_prio: member_byte_offset(btf, &task_struct, "normal_prio")?,
            rt_priority: member_byte_offset(btf, &task_struct, "rt_priority")?,
            sched_class: member_byte_offset(btf, &task_struct, "sched_class")?,
            core_cookie: member_byte_offset(btf, &task_struct, "core_cookie").ok(),
            real_parent: member_byte_offset(btf, &task_struct, "real_parent")?,
            group_leader: member_byte_offset(btf, &task_struct, "group_leader")?,
            signal: member_byte_offset(btf, &task_struct, "signal")?,
            stack: member_byte_offset(btf, &task_struct, "stack")?,
            nvcsw: member_byte_offset(btf, &task_struct, "nvcsw")?,
            nivcsw: member_byte_offset(btf, &task_struct, "nivcsw")?,
        })
    }
}

/// Field offsets within `struct sched_ext_entity` (`include/linux/sched/ext.h`).
/// Offsets here are relative to the `sched_ext_entity` base; the full
/// offset within `task_struct` is
/// `TaskStructCoreOffsets::scx + <field>`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct SchedExtEntityOffsets {
    pub runnable_node: usize,
    pub runnable_at: usize,
    pub weight: usize,
    pub slice: usize,
    pub dsq_vtime: usize,
    pub dsq: usize,
    pub dsq_list: usize,
    pub flags: usize,
    pub dsq_flags: usize,
    pub sticky_cpu: usize,
    pub holding_cpu: usize,
}

impl SchedExtEntityOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (see, _) = find_struct(btf, "sched_ext_entity")?;
        Ok(Self {
            runnable_node: member_byte_offset(btf, &see, "runnable_node")?,
            runnable_at: member_byte_offset(btf, &see, "runnable_at")?,
            weight: member_byte_offset(btf, &see, "weight")?,
            slice: member_byte_offset(btf, &see, "slice")?,
            dsq_vtime: member_byte_offset(btf, &see, "dsq_vtime")?,
            dsq: member_byte_offset(btf, &see, "dsq")?,
            dsq_list: member_byte_offset(btf, &see, "dsq_list")?,
            flags: member_byte_offset(btf, &see, "flags")?,
            dsq_flags: member_byte_offset(btf, &see, "dsq_flags")?,
            sticky_cpu: member_byte_offset(btf, &see, "sticky_cpu")?,
            holding_cpu: member_byte_offset(btf, &see, "holding_cpu")?,
        })
    }
}

/// Field offsets within `struct scx_dsq_list_node`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxDsqListNodeOffsets {
    /// Offset of `node` (struct list_head). Fixed at 0 in current
    /// kernels; resolved via BTF for forward compat.
    pub node: usize,
    /// Offset of `flags` (u32). Tested against
    /// `SCX_DSQ_LNODE_ITER_CURSOR` to skip cursor entries.
    pub flags: usize,
}

impl ScxDsqListNodeOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (lnode, _) = find_struct(btf, "scx_dsq_list_node")?;
        Ok(Self {
            node: member_byte_offset(btf, &lnode, "node")?,
            flags: member_byte_offset(btf, &lnode, "flags")?,
        })
    }
}

/// Field offsets within `struct scx_dispatch_q`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxDispatchQOffsets {
    /// Offset of `list` (struct list_head). Head of the FIFO task list.
    pub list: usize,
    /// Offset of `nr` (u32). Number of tasks queued.
    pub nr: usize,
    /// Offset of `seq` (u32). BPF-iter sequence counter.
    pub seq: usize,
    /// Offset of `id` (u64). Synthetic for built-in DSQs;
    /// BPF-allocated for user DSQs.
    pub id: usize,
    /// Offset of `hash_node` (struct rhash_head) — used by the user
    /// DSQ rhashtable walker for container_of.
    pub hash_node: usize,
}

impl ScxDispatchQOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (dsq, _) = find_struct(btf, "scx_dispatch_q")?;
        Ok(Self {
            list: member_byte_offset(btf, &dsq, "list")?,
            nr: member_byte_offset(btf, &dsq, "nr")?,
            seq: member_byte_offset(btf, &dsq, "seq")?,
            id: member_byte_offset(btf, &dsq, "id")?,
            hash_node: member_byte_offset(btf, &dsq, "hash_node")?,
        })
    }
}

/// Field offsets within `struct scx_sched`. Walker reaches a
/// `scx_sched` KVA by dereferencing `*scx_root`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxSchedOffsets {
    /// Offset of `dsq_hash` (struct rhashtable). User-allocated DSQs.
    pub dsq_hash: usize,
    /// Offset of `pnode` (`struct scx_sched_pnode **`). Per-NUMA-node.
    pub pnode: usize,
    /// Offset of `pcpu` (`struct scx_sched_pcpu __percpu *`). Per-CPU.
    pub pcpu: usize,
    /// Offset of `aborting` (bool).
    pub aborting: usize,
    /// Offset of `bypass_depth` (s32).
    pub bypass_depth: usize,
    /// Offset of `exit_kind` (atomic_t). Read raw; the SCX_EXIT_*
    /// value lives in the atomic's `counter` field.
    pub exit_kind: usize,
}

impl ScxSchedOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (sched, _) = find_struct(btf, "scx_sched")?;
        Ok(Self {
            dsq_hash: member_byte_offset(btf, &sched, "dsq_hash")?,
            pnode: member_byte_offset(btf, &sched, "pnode")?,
            pcpu: member_byte_offset(btf, &sched, "pcpu")?,
            aborting: member_byte_offset(btf, &sched, "aborting")?,
            bypass_depth: member_byte_offset(btf, &sched, "bypass_depth")?,
            exit_kind: member_byte_offset(btf, &sched, "exit_kind")?,
        })
    }
}

/// Field offsets within `struct scx_sched_pnode`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxSchedPnodeOffsets {
    /// Offset of `global_dsq` (struct scx_dispatch_q). Per-NUMA-node
    /// global DSQ.
    pub global_dsq: usize,
}

impl ScxSchedPnodeOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (pnode, _) = find_struct(btf, "scx_sched_pnode")?;
        Ok(Self {
            global_dsq: member_byte_offset(btf, &pnode, "global_dsq")?,
        })
    }
}

/// Field offsets within `struct scx_sched_pcpu`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxSchedPcpuOffsets {
    /// Offset of `bypass_dsq` (struct scx_dispatch_q). Per-CPU bypass DSQ.
    pub bypass_dsq: usize,
}

impl ScxSchedPcpuOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (pcpu, _) = find_struct(btf, "scx_sched_pcpu")?;
        Ok(Self {
            bypass_dsq: member_byte_offset(btf, &pcpu, "bypass_dsq")?,
        })
    }
}

/// Field offsets within `struct rhashtable`, `struct bucket_table`,
/// and `struct rhash_head`. Bundled together since the user DSQ walk
/// needs all three to traverse a single rhashtable.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct RhashtableOffsets {
    /// `rhashtable.tbl` — `struct bucket_table __rcu *`.
    pub tbl: usize,
    /// `rhashtable.nelems` — `atomic_t`.
    pub nelems: usize,
    /// `bucket_table.size` — `unsigned int`.
    pub bucket_table_size: usize,
    /// `bucket_table.buckets` — flex array of
    /// `struct rhash_lock_head __rcu *`. After the bit-flag mask
    /// (RHT_PTR_LOCK_BIT) the entry points at the first chained
    /// `rhash_head`.
    pub bucket_table_buckets: usize,
    /// `rhash_head.next` — always 0 in current kernels but resolved
    /// via BTF.
    pub rhash_head_next: usize,
}

impl RhashtableOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (rht, _) = find_struct(btf, "rhashtable")?;
        let tbl = member_byte_offset(btf, &rht, "tbl")?;
        let nelems = member_byte_offset(btf, &rht, "nelems")?;

        let (btab, _) = find_struct(btf, "bucket_table")?;
        let bucket_table_size = member_byte_offset(btf, &btab, "size")?;
        let bucket_table_buckets = member_byte_offset(btf, &btab, "buckets")?;

        let (rhead, _) = find_struct(btf, "rhash_head")?;
        let rhash_head_next = member_byte_offset(btf, &rhead, "next")?;

        Ok(Self {
            tbl,
            nelems,
            bucket_table_size,
            bucket_table_buckets,
            rhash_head_next,
        })
    }
}

/// Field offsets within `struct signal_struct`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct SignalStructOffsets {
    pub nr_threads: usize,
    pub pids: usize,
    pub nvcsw: usize,
    pub nivcsw: usize,
}

impl SignalStructOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (signal, _) = find_struct(btf, "signal_struct")?;
        Ok(Self {
            nr_threads: member_byte_offset(btf, &signal, "nr_threads")?,
            pids: member_byte_offset(btf, &signal, "pids")?,
            nvcsw: member_byte_offset(btf, &signal, "nvcsw")?,
            nivcsw: member_byte_offset(btf, &signal, "nivcsw")?,
        })
    }
}

/// Field offsets within `struct pid` (and the size of the fixed
/// prefix before the `numbers[]` flex array).
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct PidStructOffsets {
    /// Offset of `numbers` (flex array of `struct upid`).
    pub numbers: usize,
    /// Size of the fixed prefix; equal to `numbers` (the flex array
    /// starts where the prefix ends). Tracked separately for
    /// self-documentation against future BTF-format changes that
    /// decouple the two.
    pub size: usize,
}

impl PidStructOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (pid, _) = find_struct(btf, "pid")?;
        let numbers = member_byte_offset(btf, &pid, "numbers")?;
        Ok(Self {
            numbers,
            size: numbers,
        })
    }
}

/// Field offsets within `struct upid` plus the struct's full size.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct UpidStructOffsets {
    /// Offset of `nr` (int).
    pub nr: usize,
    /// Total size of `struct upid` — 16 bytes on x86_64 / aarch64
    /// (the only architectures ktstr currently supports). Hardcoded
    /// because btf-rs doesn't expose `Btf::resolve_type_by_id` for
    /// raw struct sizes; cited to `include/linux/pid.h::struct upid`
    /// which has been unchanged since 2.6.x.
    pub size: usize,
}

impl UpidStructOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (upid, _) = find_struct(btf, "upid")?;
        Ok(Self {
            nr: member_byte_offset(btf, &upid, "nr")?,
            size: 16,
        })
    }
}

/// Byte offsets needed by the dual-snapshot freeze coordinator's
/// per-CPU `runnable_at` scanner.
///
/// Mirrors the kernel's `check_rq_for_timeouts`
/// (`kernel/sched/ext.c`) walk:
///
/// ```ignore
/// list_for_each_entry(p, &rq->scx.runnable_list, scx.runnable_node) {
///     unsigned long last_runnable = p->scx.runnable_at;
///     ...
/// }
/// ```
///
/// The host-side scanner walks the same list per CPU, follows each
/// `runnable_node` link back to its containing `task_struct` via
/// `container_of`, and reads `task_struct.scx.runnable_at` to
/// compare against the current `jiffies_64`. All offsets resolve
/// from a single `Btf` object loaded from vmlinux.
///
/// Path uses paired `member_byte_offset` calls: `rq.scx`
/// (already resolved by [`KernelOffsets`]) plus the four offsets
/// captured here. The two struct fields off `task_struct` are
/// resolved using the BTF's anonymous-struct-walking
/// `member_byte_offset`, so `task_struct.scx` works even if the
/// kernel later wraps the field in an anonymous union.
#[derive(Debug, Clone, Copy)]
pub struct RunnableScanOffsets {
    /// Offset of `runnable_list` (struct list_head) within
    /// `struct scx_rq`. Combined with [`KernelOffsets::rq_scx`] to
    /// reach the list head from a `struct rq` pointer:
    /// `rq + rq_scx + scx_rq_runnable_list`.
    pub scx_rq_runnable_list: usize,
    /// Offset of `scx` (struct sched_ext_entity) within
    /// `struct task_struct`. Used by the container_of step to
    /// translate a `runnable_node` list_head pointer back to the
    /// owning `task_struct`.
    pub task_struct_scx: usize,
    /// Offset of `runnable_node` (struct list_head) within
    /// `struct sched_ext_entity`. The full offset of
    /// `runnable_node` within `task_struct` is
    /// `task_struct_scx + sched_ext_entity_runnable_node` —
    /// subtract this from a `runnable_node` KVA to recover the
    /// `task_struct` KVA via container_of.
    pub sched_ext_entity_runnable_node: usize,
    /// Offset of `runnable_at` (unsigned long, jiffies) within
    /// `struct sched_ext_entity`. Combined with `task_struct_scx`
    /// to read the field off a `task_struct *`:
    /// `task + task_struct_scx + sched_ext_entity_runnable_at`.
    pub sched_ext_entity_runnable_at: usize,
}

impl RunnableScanOffsets {
    /// Resolve runnable_at scanner offsets from a pre-loaded BTF
    /// object. Returns Err on a kernel without sched_ext (the
    /// `sched_ext_entity` struct is missing) or one whose layout has
    /// dropped any of the four fields.
    ///
    /// Composed from [`ScxRqOffsets`], [`TaskStructCoreOffsets`], and
    /// [`SchedExtEntityOffsets`] so `task_struct.scx` and the scx_rq /
    /// sched_ext_entity field offsets resolve from a single source of
    /// truth shared with [`ScxWalkerOffsets`] and
    /// [`TaskEnrichmentOffsets`] (task #45).
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let scx_rq = ScxRqOffsets::from_btf(btf)?;
        let task_core = TaskStructCoreOffsets::from_btf(btf)?;
        let see = SchedExtEntityOffsets::from_btf(btf)?;

        Ok(Self {
            scx_rq_runnable_list: scx_rq.runnable_list,
            task_struct_scx: task_core.scx,
            sched_ext_entity_runnable_node: see.runnable_node,
            sched_ext_entity_runnable_at: see.runnable_at,
        })
    }
}

/// Stable indices of `kernel_cpustat::cpustat[NR_STATS]` from
/// `enum cpu_usage_stat` (`include/linux/kernel_stat.h`). The kernel
/// pins the order so external readers — `/proc/stat` formatting,
/// `account_user_time` / `account_system_index_time` accumulation,
/// every userspace tool that reads `kernel_cpustat` — depend on it.
/// Hard-code the indices the failure dump captures instead of
/// resolving them via BTF: BTF only encodes the array length, not the
/// enum-to-position mapping, so a BTF-driven read would have to
/// resolve the enum separately. The cited header values are the
/// authoritative source; mismatching kernels would be a UAPI break,
/// not a layout drift this code can adapt to.
pub const CPUTIME_USER: usize = 0;
/// Index of `cpustat[CPUTIME_NICE]` (CPU time spent on nice'd user
/// processes). See [`CPUTIME_USER`].
pub const CPUTIME_NICE: usize = 1;
/// Index of `cpustat[CPUTIME_SYSTEM]` (CPU time spent in kernel).
/// See [`CPUTIME_USER`].
pub const CPUTIME_SYSTEM: usize = 2;
/// Index of `cpustat[CPUTIME_SOFTIRQ]` (CPU time servicing softirqs).
/// See [`CPUTIME_USER`].
pub const CPUTIME_SOFTIRQ: usize = 3;
/// Index of `cpustat[CPUTIME_IRQ]` (CPU time servicing hardirqs).
/// See [`CPUTIME_USER`].
pub const CPUTIME_IRQ: usize = 4;
/// Index of `cpustat[CPUTIME_IDLE]` (CPU time spent idle).
/// See [`CPUTIME_USER`].
pub const CPUTIME_IDLE: usize = 5;
/// Index of `cpustat[CPUTIME_IOWAIT]` (CPU time waiting on
/// outstanding block IO). See [`CPUTIME_USER`].
pub const CPUTIME_IOWAIT: usize = 6;
/// Index of `cpustat[CPUTIME_STEAL]` (CPU time stolen by the
/// hypervisor — virt only). See [`CPUTIME_USER`].
pub const CPUTIME_STEAL: usize = 7;

/// Number of softirq vectors per `enum` in `include/linux/interrupt.h`
/// (HI/TIMER/NET_TX/NET_RX/BLOCK/IRQ_POLL/TASKLET/SCHED/HRTIMER/RCU,
/// in that order). The order is enum-stable, mirroring
/// [`CPUTIME_USER`]'s rationale: external consumers (`/proc/softirqs`
/// formatting, `softirq_to_name[]`) depend on the layout, so a
/// reordering would be a UAPI break and resolving each name via BTF
/// would buy nothing.
pub const NR_SOFTIRQS: usize = 10;

/// Names of every softirq vector, indexed by the enum order shared
/// with the kernel's `softirq_to_name[]` (kernel/softirq.c). Surfaced
/// in failure-dump JSON so a downstream consumer reading
/// `softirqs[i]` knows which vector each slot represents without
/// chasing the kernel header.
pub const SOFTIRQ_NAMES: [&str; NR_SOFTIRQS] = [
    "HI", "TIMER", "NET_TX", "NET_RX", "BLOCK", "IRQ_POLL", "TASKLET", "SCHED", "HRTIMER", "RCU",
];

/// Byte offsets used to read per-CPU CPU-time and softirq/IRQ
/// counters from guest memory.
///
/// Three structs participate:
///   - `struct kernel_cpustat` (`include/linux/kernel_stat.h`):
///     a per-CPU `u64 cpustat[NR_STATS]` table indexed by
///     `enum cpu_usage_stat`. Hand-rolled accumulators in the
///     kernel's CPU-time accounting (`account_idle_time`,
///     `account_user_time`, etc.) bump these in nanoseconds (or
///     jiffies pre-NO_HZ_FULL — the field is `u64 nsecs` regardless;
///     `cputime64_to_clock_t` does the conversion at `/proc/stat`
///     read).
///   - `struct kernel_stat` (`include/linux/kernel_stat.h`): a
///     per-CPU `unsigned long irqs_sum` plus
///     `unsigned int softirqs[NR_SOFTIRQS]` table (10 counters in
///     2026-04-30 mainline). `kstat_incr_softirqs_this_cpu` and
///     `kstat_incr_irq_this_cpu` are the producers.
///   - `struct tick_sched` (`kernel/time/tick-sched.h`): per-CPU
///     `iowait_sleeptime` (`ktime_t` aka `s64` ns) accumulated only
///     under NO_HZ when the CPU enters idle with `nr_iowait > 0`.
///
/// All three structs sit in `.data..percpu` symbols
/// (`kernel_cpustat`, `kstat`, `tick_cpu_sched`). Per-CPU symbols
/// carry section-relative offsets in vmlinux's symtab; the per-CPU
/// KVA for CPU `n` is `<symbol> + __per_cpu_offset[n]` —
/// [`super::symbols::KernelSymbols::cpu_time_symbols`] resolves the
/// symbols and the dump path adds `__per_cpu_offset[cpu]` per CPU.
///
/// Field-presence semantics: a kernel without sched_ext omits no
/// field captured here, but a kernel built without
/// `CONFIG_NO_HZ_COMMON` drops `tick_sched`. The offset resolver
/// reports `tick_sched_iowait_sleeptime` as `Some` only when the
/// type is present. Callers that observe `None` skip the
/// `iowait_sleeptime` capture and surface `nr_iowait` (an atomic
/// counter on `struct rq` that the existing scx walker already
/// reads) instead.
#[derive(Debug, Clone, Copy)]
pub struct CpuTimeOffsets {
    /// Offset of `cpustat[]` (the `u64[NR_STATS]` array) within
    /// `struct kernel_cpustat`. Always zero on every kernel since
    /// the introduction of the struct, but resolved via BTF rather
    /// than hard-coded so a future addition of a leading field
    /// surfaces here without silent miscalculation.
    pub kernel_cpustat_cpustat: usize,
    /// Offset of `irqs_sum` (`unsigned long`) within `struct kernel_stat`.
    pub kstat_irqs_sum: usize,
    /// Offset of `softirqs[]` (the `unsigned int[NR_SOFTIRQS]`
    /// array) within `struct kernel_stat`.
    pub kstat_softirqs: usize,
    /// Offset of `iowait_sleeptime` (`ktime_t` / `s64` ns) within
    /// `struct tick_sched`. `None` when the kernel was built
    /// without `CONFIG_NO_HZ_COMMON` (the type is absent from BTF).
    pub tick_sched_iowait_sleeptime: Option<usize>,
}

impl CpuTimeOffsets {
    /// Resolve CPU-time / softirq / IRQ offsets from a pre-loaded BTF
    /// object. Returns Err when `kernel_cpustat` or `kernel_stat` are
    /// missing — these are universal, so their absence indicates a
    /// stripped vmlinux. `tick_sched` is best-effort: a kernel
    /// without `CONFIG_NO_HZ_COMMON` has no such type, and the
    /// resolver returns `Ok` with `tick_sched_iowait_sleeptime: None`.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (kernel_cpustat, _) = find_struct(btf, "kernel_cpustat")?;
        let kernel_cpustat_cpustat = member_byte_offset(btf, &kernel_cpustat, "cpustat")?;

        let (kernel_stat, _) = find_struct(btf, "kernel_stat")?;
        let kstat_irqs_sum = member_byte_offset(btf, &kernel_stat, "irqs_sum")?;
        let kstat_softirqs = member_byte_offset(btf, &kernel_stat, "softirqs")?;

        // tick_sched is CONFIG_NO_HZ_COMMON-gated; report None
        // rather than Err so the caller can capture the rest of the
        // struct's fields without forcing every kernel to enable
        // dynticks for failure-dump support.
        let tick_sched_iowait_sleeptime = match find_struct(btf, "tick_sched") {
            Ok((tick_sched, _)) => member_byte_offset(btf, &tick_sched, "iowait_sleeptime").ok(),
            Err(_) => None,
        };

        Ok(Self {
            kernel_cpustat_cpustat,
            kstat_irqs_sum,
            kstat_softirqs,
            tick_sched_iowait_sleeptime,
        })
    }
}

/// Byte offsets for the per-task enrichment walker (`task_enrichment`
/// module). Resolved from BTF once at coordinator start; reused for
/// every task the dump path reaches.
///
/// Captures every field path the failure-dump enrichment surfaces:
/// task_struct's directly-accessed fields, plus the chained
/// signal_struct / pid / upid struct offsets needed to recover
/// pgid/sid via the kernel's standard
/// `signal->pids[PIDTYPE_*]->numbers[0].nr` traversal (see
/// `kernel/pid.c::pid_nr` for the canonical traversal pattern).
///
/// `TASK_COMM_LEN` is fixed at 16 by the kernel uapi
/// (`include/linux/sched.h::TASK_COMM_LEN`); the walker reads a
/// fixed-size 16-byte buffer at `task_struct_comm`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TaskEnrichmentOffsets {
    // -- struct task_struct fields --
    /// Offset of `comm` (char[TASK_COMM_LEN=16]).
    pub task_struct_comm: usize,
    /// Offset of `pid` (pid_t == int).
    pub task_struct_pid: usize,
    /// Offset of `tgid` (pid_t == int).
    pub task_struct_tgid: usize,
    /// Offset of `prio` (int). Effective scheduling priority,
    /// adjusted for PI boost.
    pub task_struct_prio: usize,
    /// Offset of `static_prio` (int). User-set priority before
    /// PI boost.
    pub task_struct_static_prio: usize,
    /// Offset of `normal_prio` (int). Normal priority for the
    /// task's scheduling class.
    pub task_struct_normal_prio: usize,
    /// Offset of `rt_priority` (unsigned int). Real-time priority
    /// (1-99) for SCHED_FIFO/RR; 0 for non-RT tasks.
    pub task_struct_rt_priority: usize,
    /// Offset of `sched_class` (`const struct sched_class *`).
    /// Compared against the cached sched_class symbol KVAs to
    /// decode to a name (CFS / RT / DL / IDLE / STOP / EXT) and
    /// to flag PI-boost-out-of-SCX.
    pub task_struct_sched_class: usize,
    /// Offset of `scx` (struct sched_ext_entity).
    pub task_struct_scx: usize,
    /// Offset of `core_cookie` (unsigned long). Core scheduling
    /// cookie (`CONFIG_SCHED_CORE`-gated; field is conditional).
    /// `None` when the kernel was built without core scheduling
    /// — the walker skips the read and surfaces `core_cookie` as
    /// `None` in the enrichment.
    pub task_struct_core_cookie: Option<usize>,
    /// Offset of `real_parent` (`struct task_struct __rcu *`).
    /// RCU-protected pointer to the real parent (process that
    /// fork()ed this one); the walker does a single deref to
    /// read the parent's pid + comm.
    pub task_struct_real_parent: usize,
    /// Offset of `group_leader` (`struct task_struct *`).
    /// Pointer to the thread group leader.
    pub task_struct_group_leader: usize,
    /// Offset of `signal` (`struct signal_struct *`). Shared
    /// signal_struct gives access to nr_threads + pgid/sid via
    /// the pids[] array.
    pub task_struct_signal: usize,
    /// Offset of `stack` (`void *`). Kernel stack base for the
    /// stack-trace lock-detection walker. Tasks share a kernel
    /// stack of `THREAD_SIZE`; a successful translate of `stack`
    /// + walking up `THREAD_SIZE` covers the active stack frames.
    pub task_struct_stack: usize,

    // -- struct sched_ext_entity fields (relative to scx base; the
    //    full offset inside task_struct is `task_struct_scx + ...`) --
    /// Offset of `weight` (u32) within `struct sched_ext_entity`.
    /// scx-domain CFS-equivalent weight; 100 default, scaled by
    /// scx_group_set_weight on cgroup writes.
    pub see_weight: usize,

    // -- struct signal_struct fields --
    /// Offset of `nr_threads` (int). Live thread count for the
    /// thread group.
    pub signal_struct_nr_threads: usize,
    /// Offset of `pids` (`struct pid *pids[PIDTYPE_MAX]`). The
    /// walker indexes this array by `PIDTYPE_*` constants
    /// (PID=0, TGID=1, PGID=2, SID=3) per
    /// `include/linux/pid_types.h::enum pid_type` to reach
    /// per-type pid pointers.
    pub signal_struct_pids: usize,
    /// Offset of `nvcsw` (unsigned long). Voluntary context
    /// switches accumulated for the thread group's dead threads;
    /// per-thread `task_struct_nvcsw` accumulates the live count.
    pub signal_struct_nvcsw: usize,
    /// Offset of `nivcsw` (unsigned long). Involuntary context
    /// switches; mirror of `nvcsw`.
    pub signal_struct_nivcsw: usize,

    // -- per-task voluntary/involuntary context-switch counters
    //    on task_struct itself (live thread count). The dump
    //    surfaces these directly from each task; the
    //    signal_struct totals are surfaced separately. --
    /// Offset of `nvcsw` (unsigned long) within `struct task_struct`.
    pub task_struct_nvcsw: usize,
    /// Offset of `nivcsw` (unsigned long) within `struct task_struct`.
    pub task_struct_nivcsw: usize,

    // -- struct pid fields --
    /// Offset of `numbers` (struct upid[]) flex array within
    /// `struct pid`. Index 0 is the global pid namespace's upid
    /// (the canonical, root-ns pid number).
    pub pid_numbers: usize,
    /// Total size of `struct pid`'s fixed prefix before the
    /// `numbers[]` flex array. Equal to `pid_numbers` — kept as
    /// a separate field for self-documentation against future
    /// kernel additions that move the flex array.
    pub pid_size: usize,

    // -- struct upid fields --
    /// Offset of `nr` (int) within `struct upid`. The canonical
    /// pid number for the namespace.
    pub upid_nr: usize,
    /// Total size of `struct upid` (8 + 8 = 16 bytes on x86_64
    /// per `include/linux/pid.h::struct upid`). The walker
    /// computes `pid_numbers + level * upid_size` to index the
    /// flex array; we always use level 0 (root ns) but capture
    /// the stride so future per-ns walking is straightforward.
    pub upid_size: usize,
}

#[allow(dead_code)] // wired into DumpContext + walk_task_enrichment;
                    // freeze coordinator passes None until #50 (rq->scx
                    // walker) lands a producer that builds
                    // TaskEnrichmentCapture.
impl TaskEnrichmentOffsets {
    /// Resolve all per-task / signal_struct / pid / upid offsets
    /// from a pre-loaded BTF object. Returns Err on a stripped
    /// vmlinux missing any required field. `core_cookie` is the
    /// only optional field — kernels built without
    /// CONFIG_SCHED_CORE drop it from `task_struct` BTF and the
    /// walker correspondingly skips that capture.
    ///
    /// Composed from per-struct sub-groups
    /// ([`TaskStructCoreOffsets`], [`TaskStructEnrichmentOffsets`],
    /// [`SchedExtEntityOffsets`], [`SignalStructOffsets`],
    /// [`PidStructOffsets`], [`UpidStructOffsets`]) so each kernel
    /// field is resolved exactly once across the codebase
    /// (deduplicated source of truth — task #45).
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let task_core = TaskStructCoreOffsets::from_btf(btf)?;
        let task_ext = TaskStructEnrichmentOffsets::from_btf(btf)?;
        let see = SchedExtEntityOffsets::from_btf(btf)?;
        let signal = SignalStructOffsets::from_btf(btf)?;
        let pid_offs = PidStructOffsets::from_btf(btf)?;
        let upid = UpidStructOffsets::from_btf(btf)?;

        Ok(Self {
            task_struct_comm: task_core.comm,
            task_struct_pid: task_core.pid,
            task_struct_tgid: task_ext.tgid,
            task_struct_prio: task_ext.prio,
            task_struct_static_prio: task_ext.static_prio,
            task_struct_normal_prio: task_ext.normal_prio,
            task_struct_rt_priority: task_ext.rt_priority,
            task_struct_sched_class: task_ext.sched_class,
            task_struct_scx: task_core.scx,
            task_struct_core_cookie: task_ext.core_cookie,
            task_struct_real_parent: task_ext.real_parent,
            task_struct_group_leader: task_ext.group_leader,
            task_struct_signal: task_ext.signal,
            task_struct_stack: task_ext.stack,
            see_weight: see.weight,
            signal_struct_nr_threads: signal.nr_threads,
            signal_struct_pids: signal.pids,
            signal_struct_nvcsw: signal.nvcsw,
            signal_struct_nivcsw: signal.nivcsw,
            task_struct_nvcsw: task_ext.nvcsw,
            task_struct_nivcsw: task_ext.nivcsw,
            pid_numbers: pid_offs.numbers,
            pid_size: pid_offs.size,
            upid_nr: upid.nr,
            upid_size: upid.size,
        })
    }
}

/// Stable indices into `signal_struct.pids[PIDTYPE_MAX]` for the four
/// pid types ktstr reads (per `include/linux/pid_types.h::enum pid_type`).
/// Re-exported here so the task_enrichment walker doesn't have to
/// duplicate the enum values inline.
#[allow(dead_code)]
pub mod pid_type {
    /// `PIDTYPE_PID` — the task's own pid number.
    pub const PID: usize = 0;
    /// `PIDTYPE_TGID` — thread-group leader pid.
    pub const TGID: usize = 1;
    /// `PIDTYPE_PGID` — process group id.
    pub const PGID: usize = 2;
    /// `PIDTYPE_SID` — session id.
    pub const SID: usize = 3;
}

/// Composite of per-struct offset sub-groups used by the host-side
/// scx walker (`scx_walker.rs`) to enumerate per-CPU `rq->scx` state
/// and per-DSQ depth/queue state.
///
/// Each kernel struct is resolved as an `Option<SubGroup>`: a
/// stripped vmlinux that drops one struct (e.g. a kernel built without
/// `CONFIG_NUMA` lacking `scx_sched_pnode`) doesn't blind the whole
/// walker — the walker's per-pass code (per-CPU local DSQ, per-CPU
/// bypass DSQ, per-node global DSQ, user dsq_hash, per-CPU
/// runnable_list) checks the relevant sub-group(s) and skips its pass
/// when missing. The `missing_groups()` helper surfaces the list of
/// absent sub-groups for the `scx_walker_unavailable` reason in the
/// failure dump.
///
/// Composes the per-struct sub-groups defined above:
/// [`RqStructOffsets`], [`ScxRqOffsets`], [`TaskStructCoreOffsets`],
/// [`SchedExtEntityOffsets`], [`ScxDsqListNodeOffsets`],
/// [`ScxDispatchQOffsets`], [`ScxSchedOffsets`],
/// [`ScxSchedPnodeOffsets`], [`ScxSchedPcpuOffsets`],
/// [`RhashtableOffsets`].
///
/// Kernel sources verified:
/// - `struct scx_rq`: kernel/sched/sched.h
/// - `struct scx_sched` / `struct scx_sched_pnode` /
///   `struct scx_sched_pcpu`: kernel/sched/ext_internal.h
/// - `struct scx_dispatch_q`: include/linux/sched/ext.h
/// - `struct scx_dsq_list_node`: include/linux/sched/ext.h
/// - `struct sched_ext_entity`: include/linux/sched/ext.h
/// - `struct rhashtable` / `struct bucket_table`: include/linux/rhashtable.h
#[derive(Debug, Clone)]
#[allow(dead_code)] // wired into ScxWalkerCapture; freeze coordinator
                    // populates the capture once the producer-side
                    // wiring lands.
pub struct ScxWalkerOffsets {
    /// `struct rq` field offsets (`scx`, `curr`). Required for every
    /// per-CPU read; `None` blinds the walker entirely.
    pub rq: Option<RqStructOffsets>,
    /// `struct scx_rq` field offsets. Required by the rq->scx walker
    /// (per-CPU runnable_list head, scalar state) and the per-CPU
    /// local DSQ walk (`scx_rq.local_dsq`).
    pub scx_rq: Option<ScxRqOffsets>,
    /// `struct task_struct` core offsets — `comm`, `pid`, `scx`.
    /// Required for both the runnable_list container_of walk and the
    /// DSQ list_head container_of walk. Shared single-source-of-truth
    /// with [`RunnableScanOffsets`] and [`TaskEnrichmentOffsets`]
    /// (task #45).
    pub task: Option<TaskStructCoreOffsets>,
    /// `struct sched_ext_entity` field offsets. Required for
    /// container_of math (`runnable_node`, `dsq_list`) and for
    /// scx-domain task fields (`weight`, `slice`, `dsq_vtime`,
    /// `flags`, etc.).
    pub see: Option<SchedExtEntityOffsets>,
    /// `struct scx_dsq_list_node` offsets. Required to detect and
    /// skip iterator cursor entries during DSQ walks.
    pub dsq_lnode: Option<ScxDsqListNodeOffsets>,
    /// `struct scx_dispatch_q` offsets. Required for every DSQ scalar
    /// read and the user-DSQ rhashtable container_of step.
    pub dsq: Option<ScxDispatchQOffsets>,
    /// `struct scx_sched` offsets. Required to read top-level
    /// scheduler scalar state (`aborting`, `bypass_depth`,
    /// `exit_kind`) and to reach `dsq_hash` / `pnode` / `pcpu`.
    pub sched: Option<ScxSchedOffsets>,
    /// `struct scx_sched_pnode` offsets. Required for the per-NUMA-
    /// node global DSQ pass; `None` blinds only that pass (kernels
    /// built without `CONFIG_NUMA` may lack the type).
    pub sched_pnode: Option<ScxSchedPnodeOffsets>,
    /// `struct scx_sched_pcpu` offsets. Required for the per-CPU
    /// bypass DSQ pass; `None` blinds only that pass.
    pub sched_pcpu: Option<ScxSchedPcpuOffsets>,
    /// `struct rhashtable` / `struct bucket_table` / `struct rhash_head`
    /// offsets. Required for the user DSQ hash walk; `None` blinds
    /// only that pass.
    pub rht: Option<RhashtableOffsets>,
}

#[allow(dead_code)] // wired into DumpContext::ScxWalkerCapture; the
                    // freeze coordinator passes None until the
                    // producer-side wiring (resolve offsets +
                    // build rq arrays) lands.
impl ScxWalkerOffsets {
    /// Resolve every scx walker sub-group from a pre-loaded BTF
    /// object. Each sub-group resolves independently — a missing
    /// kernel struct surfaces as the corresponding `Option` being
    /// `None`, NOT as an `Err`. Callers (the walker) check each
    /// sub-group before dereferencing and skip the relevant pass on
    /// `None`. The `missing_groups()` helper produces a list of
    /// absent sub-groups suitable for the failure dump's
    /// `scx_walker_unavailable` reason.
    ///
    /// This signature returns `Result` only because the BTF parse
    /// itself is fallible; per-sub-group BTF lookup failures are
    /// stored as `None`, not propagated.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        Ok(Self {
            rq: RqStructOffsets::from_btf(btf).ok(),
            scx_rq: ScxRqOffsets::from_btf(btf).ok(),
            task: TaskStructCoreOffsets::from_btf(btf).ok(),
            see: SchedExtEntityOffsets::from_btf(btf).ok(),
            dsq_lnode: ScxDsqListNodeOffsets::from_btf(btf).ok(),
            dsq: ScxDispatchQOffsets::from_btf(btf).ok(),
            sched: ScxSchedOffsets::from_btf(btf).ok(),
            sched_pnode: ScxSchedPnodeOffsets::from_btf(btf).ok(),
            sched_pcpu: ScxSchedPcpuOffsets::from_btf(btf).ok(),
            rht: RhashtableOffsets::from_btf(btf).ok(),
        })
    }

    /// Returns the names of sub-groups that failed to resolve. Empty
    /// when every kernel struct the walker touches was present in
    /// BTF. Surfaced in the failure dump's `scx_walker_unavailable`
    /// field so a partially-degraded walk announces which passes
    /// were skipped.
    pub fn missing_groups(&self) -> Vec<&'static str> {
        let mut missing = Vec::new();
        if self.rq.is_none() {
            missing.push("rq");
        }
        if self.scx_rq.is_none() {
            missing.push("scx_rq");
        }
        if self.task.is_none() {
            missing.push("task_struct");
        }
        if self.see.is_none() {
            missing.push("sched_ext_entity");
        }
        if self.dsq_lnode.is_none() {
            missing.push("scx_dsq_list_node");
        }
        if self.dsq.is_none() {
            missing.push("scx_dispatch_q");
        }
        if self.sched.is_none() {
            missing.push("scx_sched");
        }
        if self.sched_pnode.is_none() {
            missing.push("scx_sched_pnode");
        }
        if self.sched_pcpu.is_none() {
            missing.push("scx_sched_pcpu");
        }
        if self.rht.is_none() {
            missing.push("rhashtable/bucket_table/rhash_head");
        }
        missing
    }
}

/// `SCX_DSQ_LNODE_ITER_CURSOR` flag value tested against
/// `scx_dsq_list_node.flags` to skip iterator cursor entries during
/// a DSQ walk.
///
/// Pinned per `include/linux/sched/ext.h::SCX_DSQ_LNODE_ITER_CURSOR`
/// (= 1u32). Defining it here keeps the constant out of the walker
/// module's public surface — the walker imports this and applies the
/// mask at every list iteration.
#[allow(dead_code)]
pub const SCX_DSQ_LNODE_ITER_CURSOR: u32 = 1;

/// Bit mask the rhashtable bucket-pointer encoding uses to mark a
/// "lock" pointer (`BIT(0)`). Pointers stored in
/// `bucket_table.buckets[i]` have bit 0 set as a tagged-pointer
/// indicator; the host walker masks bit 0 off before chasing the
/// chain. See `include/linux/rhashtable.h::rht_ptr` for the kernel's
/// own bit-stripping helper.
#[allow(dead_code)]
pub const RHT_PTR_LOCK_BIT: u64 = 1;

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

    /// Resolve [`CpuTimeOffsets`] against the test vmlinux. Pins the
    /// offsets the per-CPU CPU-time / softirq / IRQ failure-dump
    /// capture path consumes:
    ///   - `kernel_cpustat::cpustat[]` is at offset 0 (single-field
    ///     struct).
    ///   - `kstat.irqs_sum` and `kstat.softirqs[]` are distinct.
    ///   - `tick_sched::iowait_sleeptime` is best-effort (Some only
    ///     under CONFIG_NO_HZ_COMMON).
    #[test]
    fn parse_cpu_time_offsets_from_vmlinux() {
        let path = match crate::monitor::find_test_vmlinux() {
            Some(p) => p,
            None => return,
        };
        let btf = match load_btf_from_path(&path) {
            Ok(b) => b,
            Err(e) => skip!("vmlinux BTF load failed: {e}"),
        };
        let offsets = match CpuTimeOffsets::from_btf(&btf) {
            Ok(o) => o,
            Err(e) => skip!("CpuTimeOffsets::from_btf failed: {e}"),
        };
        assert_eq!(
            offsets.kernel_cpustat_cpustat, 0,
            "kernel_cpustat::cpustat[] must live at offset 0 \
             (single-field struct in include/linux/kernel_stat.h)"
        );
        assert_ne!(
            offsets.kstat_irqs_sum, offsets.kstat_softirqs,
            "kstat irqs_sum and softirqs must be at distinct offsets"
        );
        // tick_sched is CONFIG_NO_HZ_COMMON-gated. None is a valid
        // outcome on dynticks-disabled kernels; just don't crash.
        if let Some(off) = offsets.tick_sched_iowait_sleeptime {
            assert!(
                off > 0,
                "tick_sched::iowait_sleeptime must be nonzero when present"
            );
        }
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
    // These tests exercise the `<path>.btf` sidecar pipeline:
    //   * pure helpers (path derivation, magic check, freshness)
    //     directly;
    //   * end-to-end `load_btf_from_path` behavior against a real-ELF
    //     fixture when one is available on the host;
    //   * the cache-root membership guard that suppresses sidecar
    //     reads/writes for vmlinux paths outside the cache, including
    //     symlink-resolution semantics and relative-path handling.

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

    /// Vmlinux staged inside a private cache root for sidecar tests.
    ///
    /// Field declaration order pins drop order — Rust drops struct
    /// fields top-to-bottom. `_cache_env` (the `KTSTR_CACHE_DIR`
    /// `EnvVarGuard`) is declared first so it drops first, restoring
    /// the env BEFORE `_root` drops and removes the temporary
    /// directory. Without that ordering, `KTSTR_CACHE_DIR` would
    /// transiently point at a deleted directory while the next test
    /// runs — a dangling-env-ref hazard.
    ///
    /// `entry_dir` and `vmlinux` are simple `PathBuf`s with no
    /// drop side effects, so their position only documents intent.
    struct CacheStagedVmlinux {
        _cache_env: crate::test_support::test_helpers::EnvVarGuard,
        entry_dir: std::path::PathBuf,
        vmlinux: std::path::PathBuf,
        _root: tempfile::TempDir,
    }

    /// Stage a vmlinux copy at `<cache_root>/<entry>/vmlinux` so the
    /// sidecar guard treats writes as in-cache, and point
    /// `KTSTR_CACHE_DIR` at the cache root for the returned value's
    /// lifetime. See [`CacheStagedVmlinux`] for drop semantics.
    fn stage_in_cache(src: &std::path::Path) -> CacheStagedVmlinux {
        let root = tempfile::TempDir::new().expect("cache-root tempdir");
        let entry_dir = root.path().join("kentry");
        std::fs::create_dir_all(&entry_dir).expect("create cache entry dir");
        let vmlinux = entry_dir.join("vmlinux");
        std::fs::copy(src, &vmlinux).expect("copy vmlinux into cache-staged dir");
        let _cache_env =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", root.path());
        CacheStagedVmlinux {
            _cache_env,
            entry_dir,
            vmlinux,
            _root: root,
        }
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

        // Stage the vmlinux inside a private KTSTR_CACHE_DIR so the
        // sidecar membership guard permits the write. lock_env held
        // for the test's lifetime — KTSTR_CACHE_DIR is process-wide.
        let _env = crate::test_support::test_helpers::lock_env();
        let staged = stage_in_cache(&path);
        let vmlinux = staged.vmlinux.as_path();
        let sidecar = btf_sidecar_path(vmlinux);
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
        let btf1 = load_btf_from_path(vmlinux).expect("first load must succeed");
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
            sidecar_fresh(&sidecar, vmlinux),
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
        let btf2 = load_btf_from_path(vmlinux).expect("second load must succeed");
        let _ = format!("{:?}", btf2.resolve_types_by_name("task_struct").is_ok());
        let sidecar_mtime_after = std::fs::metadata(&sidecar).unwrap().modified().unwrap();
        assert_eq!(
            sidecar_mtime_before, sidecar_mtime_after,
            "second load must hit sidecar cache — mtime bump proves a \
             redundant rewrite",
        );
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

        let _env = crate::test_support::test_helpers::lock_env();
        let staged = stage_in_cache(&path);
        let vmlinux = staged.vmlinux.as_path();
        let sidecar = btf_sidecar_path(vmlinux);

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
            !sidecar_fresh(&sidecar, vmlinux),
            "precondition: planted sidecar must be stale",
        );

        let btf = load_btf_from_path(vmlinux)
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
            sidecar_fresh(&sidecar, vmlinux),
            "sidecar must be fresh again after re-extraction",
        );
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

        let _env = crate::test_support::test_helpers::lock_env();
        let staged = stage_in_cache(&path);
        let vmlinux = staged.vmlinux.as_path();
        let sidecar = btf_sidecar_path(vmlinux);
        // Plant a sidecar that is newer than vmlinux but whose
        // contents do not carry the BTF magic.
        std::fs::write(&sidecar, b"not-btf-bytes").unwrap();
        assert!(
            sidecar_fresh(&sidecar, vmlinux),
            "precondition: planted sidecar must be mtime-fresh",
        );

        let btf = load_btf_from_path(vmlinux)
            .expect("load must recover when sidecar is fresh-but-corrupt");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        // Corrupt sidecar should have been overwritten.
        let sidecar_bytes = std::fs::read(&sidecar).unwrap();
        assert!(
            is_raw_btf(&sidecar_bytes),
            "corrupt sidecar must be overwritten on next load",
        );
    }

    /// When the sidecar would be written to a read-only directory,
    /// the load must still succeed — sidecar writes are
    /// best-effort and never surface as errors. Exercises the
    /// tracing::warn fallback in `write_btf_sidecar`'s error path
    /// while the path IS inside the cache root, so the
    /// membership-guard skip cannot be the reason no sidecar
    /// appears.
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

        let _env = crate::test_support::test_helpers::lock_env();
        let staged = stage_in_cache(&path);
        let vmlinux = staged.vmlinux.as_path();
        let entry_dir = staged.entry_dir.as_path();
        // Mark entry dir read-only after the vmlinux is in place so
        // the sidecar's tempfile+rename within `write_btf_sidecar`
        // fails on tempfile creation.
        std::fs::set_permissions(entry_dir, std::fs::Permissions::from_mode(0o555)).unwrap();

        // Load must succeed despite the sidecar write failing.
        let btf = load_btf_from_path(vmlinux)
            .expect("load must succeed even when sidecar dir is read-only");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        // Sidecar must not exist — write should have failed at the
        // best-effort layer, not at the membership guard.
        let sidecar = btf_sidecar_path(vmlinux);
        assert!(
            !sidecar.exists(),
            "sidecar must not exist after write to read-only dir",
        );

        // Restore permissions so the tempdir cleanup (TempDir drop)
        // can recurse into the entry dir.
        let _ = std::fs::set_permissions(entry_dir, std::fs::Permissions::from_mode(0o755));
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
        }
        // /sys/kernel/btf/vmlinux path: raw BTF on a read-only
        // filesystem. Any sidecar write would fail anyway, and the
        // sidecar path itself ("/sys/kernel/btf/vmlinux.btf") is
        // not writable — we cannot assert much here beyond "load
        // must succeed," which the pre-existing tests already
        // cover.
    }

    /// A vmlinux that lives outside the configured cache root must
    /// never have a sidecar written next to it. Models the
    /// kernel-source-tree pollution shape that motivated the guard:
    /// `KTSTR_CACHE_DIR` points at one tempdir, the vmlinux lives
    /// in a sibling tempdir (the "source tree"), and the load must
    /// produce parsed BTF without touching the source-tree
    /// directory.
    #[test]
    fn sidecar_skipped_when_path_outside_cache_root() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        // KTSTR_CACHE_DIR points at one tempdir.
        let cache_root = tempfile::TempDir::new().expect("cache root tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            cache_root.path(),
        );
        // vmlinux lives in a sibling tempdir — outside the cache
        // root, simulating a kernel source tree.
        let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
        let vmlinux = source_tree.path().join("vmlinux");
        std::fs::copy(&path, &vmlinux).expect("copy vmlinux into source-tree dir");

        let btf = load_btf_from_path(&vmlinux)
            .expect("load must succeed even when sidecar is suppressed");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        let sidecar = btf_sidecar_path(&vmlinux);
        assert!(
            !sidecar.exists(),
            "sidecar must not be written when vmlinux path is outside cache root, got {}",
            sidecar.display(),
        );
    }

    /// A vmlinux that lives inside the configured cache root must
    /// have its sidecar written. Sibling assertion to
    /// `sidecar_skipped_when_path_outside_cache_root`: the guard
    /// must not be a blanket suppression, only an out-of-cache
    /// suppression.
    #[test]
    fn sidecar_written_when_path_inside_cache_root() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        let staged = stage_in_cache(&path);
        let vmlinux = staged.vmlinux.as_path();

        let sidecar = btf_sidecar_path(vmlinux);
        assert!(
            !sidecar.exists(),
            "precondition: sidecar must not exist before the load — \
             a leftover from a prior test would falsely pass the post-load \
             existence check",
        );

        let btf = load_btf_from_path(vmlinux).expect("load must succeed inside cache root");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        assert!(
            sidecar.exists(),
            "sidecar must be written when vmlinux path is inside cache root, expected at {}",
            sidecar.display(),
        );
        let bytes = std::fs::read(&sidecar).unwrap();
        assert!(
            is_raw_btf(&bytes),
            "sidecar must contain raw BTF (0x9FEB magic) when written inside cache root",
        );
    }

    /// Cache root that cannot be resolved (every cascade variable
    /// removed) must produce `path_inside_cache_root == false` and
    /// suppress the sidecar. The load itself must still succeed —
    /// "no cache root" is not a failure mode for BTF resolution,
    /// just for sidecar caching.
    #[test]
    fn sidecar_skipped_when_cache_root_unresolvable() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        // Strip every variable in the resolution cascade so
        // resolve_cache_root_with_suffix has nothing to walk.
        let _no_ktstr = crate::test_support::test_helpers::EnvVarGuard::remove("KTSTR_CACHE_DIR");
        let _no_xdg = crate::test_support::test_helpers::EnvVarGuard::remove("XDG_CACHE_HOME");
        let _no_home = crate::test_support::test_helpers::EnvVarGuard::remove("HOME");

        let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
        let vmlinux = source_tree.path().join("vmlinux");
        std::fs::copy(&path, &vmlinux).expect("copy vmlinux");

        let btf = load_btf_from_path(&vmlinux)
            .expect("load must succeed when cache root is unresolvable");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        let sidecar = btf_sidecar_path(&vmlinux);
        assert!(
            !sidecar.exists(),
            "sidecar must not be written when cache root is unresolvable, got {}",
            sidecar.display(),
        );
    }

    /// Symlink E2E: real vmlinux LIVES in the cache, a symlink to
    /// it lives in a source tree. Loading via the symlink path
    /// must canonicalize through to the cache and write the
    /// sidecar NEXT TO THE REAL FILE — not next to the symlink.
    /// The sidecar derivation MUST track the same canonical path
    /// as the membership check.
    #[test]
    #[cfg(unix)]
    fn load_btf_symlink_into_cache_writes_sidecar_in_cache_only() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        let staged = stage_in_cache(&path);
        let real_vmlinux = staged.vmlinux.as_path();
        let real_sidecar = btf_sidecar_path(real_vmlinux);
        assert!(
            !real_sidecar.exists(),
            "precondition: real sidecar must not exist before the load",
        );

        // Symlink in a sibling tempdir (the "source tree") pointing
        // at the real cached vmlinux.
        let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
        let symlink_path = source_tree.path().join("vmlinux");
        std::os::unix::fs::symlink(real_vmlinux, &symlink_path)
            .expect("create symlink to real vmlinux");
        let lexical_sidecar = btf_sidecar_path(&symlink_path);

        // Load via the symlink path. Post-fix, canonicalize at the
        // top of load_btf_from_path resolves the symlink so the
        // sidecar writes to <cache>/kentry/vmlinux.btf next to the
        // real file. Pre-fix this flow missed the cache entirely:
        // lexical parent (<source-tree>) canonicalizes outside the
        // cache, the membership gate returns false, and the sidecar
        // is suppressed for what is, after symlink resolution, a
        // genuine cache entry.
        let btf = load_btf_from_path(&symlink_path)
            .expect("load via symlink must succeed and resolve the target");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        assert!(
            real_sidecar.exists(),
            "sidecar must land at the canonical path inside cache, expected {}",
            real_sidecar.display(),
        );
        assert!(
            !lexical_sidecar.exists(),
            "sidecar must NOT land next to the symlink in the source tree, \
             got pollution at {}",
            lexical_sidecar.display(),
        );
    }

    /// Symlink E2E inverse: real vmlinux lives in a source tree,
    /// symlink to it lives in the cache. Loading via the symlink
    /// path must canonicalize through to the source-tree real file
    /// — and the membership check on the canonical path returns
    /// false, so NO sidecar is written anywhere. The cache
    /// directory must remain free of sidecar files for symlinks
    /// pointing OUT.
    #[test]
    #[cfg(unix)]
    fn load_btf_symlink_out_of_cache_writes_no_sidecar() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        // Cache root with no real vmlinux inside it.
        let cache_root = tempfile::TempDir::new().expect("cache-root tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            cache_root.path(),
        );
        // Real vmlinux in source tree (outside cache).
        let source_tree = tempfile::TempDir::new().expect("source-tree tempdir");
        let real_vmlinux = source_tree.path().join("vmlinux");
        std::fs::copy(&path, &real_vmlinux).expect("copy vmlinux into source tree");
        // Symlink in cache pointing at the real source-tree vmlinux.
        let symlink_in_cache = cache_root.path().join("vmlinux");
        std::os::unix::fs::symlink(&real_vmlinux, &symlink_in_cache)
            .expect("create symlink to source-tree vmlinux");

        let btf = load_btf_from_path(&symlink_in_cache).expect("load via symlink must succeed");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        let real_sidecar = btf_sidecar_path(&real_vmlinux);
        let lexical_sidecar = btf_sidecar_path(&symlink_in_cache);
        assert!(
            !real_sidecar.exists(),
            "sidecar must not land in source tree (outside cache), got {}",
            real_sidecar.display(),
        );
        assert!(
            !lexical_sidecar.exists(),
            "sidecar must not land at the symlink path in cache either — \
             canonicalize-at-top resolves to the source-tree real file, \
             which is outside the cache",
        );
    }

    /// Relative path: pass a path that does not start with `/`,
    /// confirm no sidecar lands at either the lexical relative
    /// path's location or the absolute target.
    ///
    /// Production callers reach `load_btf_from_path` through several
    /// paths: `crate::vmm::find_vmlinux` (absolute paths derived from
    /// the kernel-cache entry or distro debug locations) and the `None`
    /// fallback in `crate::probe::btf::parse_btf_functions` /
    /// `crate::probe::btf::resolve_field_specs` (the absolute literal
    /// `/sys/kernel/btf/vmlinux`). All emit absolute paths. A
    /// relative-path invocation is unusual, and its semantics
    /// depend on the test process's CWD: if CWD is unrelated to
    /// the relative path's parent (the typical case during a test
    /// run), the initial `fs::read` fails and the function returns
    /// Err before reaching any sidecar branch. This pins:
    ///
    ///   * the function does not panic on a relative input;
    ///   * no sidecar is written at the lexical relative-path
    ///     location, so a CWD-relative pollution shape cannot leak
    ///     past the membership gate even when canonicalize would
    ///     otherwise have reached the cache;
    ///   * no sidecar is written at the absolute target either —
    ///     the read step never resolves to it.
    #[test]
    fn load_btf_relative_path_suppresses_sidecar() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        // Point KTSTR_CACHE_DIR somewhere isolated. The cache root
        // is irrelevant for the assertion — we just want the load
        // path to NOT be inside whatever it points at.
        let cache_root = tempfile::TempDir::new().expect("cache-root tempdir");
        let _cache_env = crate::test_support::test_helpers::EnvVarGuard::set(
            "KTSTR_CACHE_DIR",
            cache_root.path(),
        );
        // Stage a real vmlinux in a tempdir, then build a relative
        // path referring to it. The relative path is constructed
        // by stripping the leading `/` from the absolute path; from
        // the test process's CWD (the cargo workspace root), this
        // relative path will not resolve to a file, so the load's
        // initial `fs::read` step fails and the function returns
        // Err. The point of the test is the post-condition: NO
        // sidecar appears anywhere as a side effect.
        let outside = tempfile::TempDir::new().expect("outside tempdir");
        let abs_vmlinux = outside.path().join("vmlinux");
        std::fs::copy(&path, &abs_vmlinux).expect("copy vmlinux into outside dir");
        let rel_str = abs_vmlinux
            .to_str()
            .expect("test vmlinux path must be UTF-8")
            .strip_prefix('/')
            .expect("absolute path expected to start with /");
        let rel = std::path::Path::new(rel_str);
        assert!(
            !rel.is_absolute(),
            "precondition: constructed path must be relative, got {}",
            rel.display(),
        );

        let _ = load_btf_from_path(rel);
        let abs_sidecar = btf_sidecar_path(&abs_vmlinux);
        let rel_sidecar = btf_sidecar_path(rel);
        assert!(
            !abs_sidecar.exists(),
            "sidecar must not appear at the absolute target, got {}",
            abs_sidecar.display(),
        );
        assert!(
            !rel_sidecar.exists(),
            "sidecar must not appear at the relative path's lexical \
             location, got {}",
            rel_sidecar.display(),
        );
    }

    /// Empty `KTSTR_CACHE_DIR=""` falls through the cascade per
    /// `resolve_cache_root_with_suffix`. With the rest of the
    /// cascade pointed at an isolated tempdir, the membership
    /// check succeeds for paths inside the resolved root. Models
    /// the operator who clears KTSTR_CACHE_DIR expecting
    /// XDG/HOME to take over.
    #[test]
    fn load_btf_empty_ktstr_cache_dir_falls_through() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        let xdg = tempfile::TempDir::new().expect("xdg tempdir");
        let _g_ktstr = crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CACHE_DIR", "");
        let _g_xdg =
            crate::test_support::test_helpers::EnvVarGuard::set("XDG_CACHE_HOME", xdg.path());
        // Resolved root: <xdg>/ktstr/kernels.
        let resolved_root = xdg.path().join("ktstr").join("kernels");
        let entry = resolved_root.join("kentry");
        std::fs::create_dir_all(&entry).expect("create cache entry under XDG fallback");
        let vmlinux = entry.join("vmlinux");
        std::fs::copy(&path, &vmlinux).expect("copy vmlinux into XDG-derived cache");
        let sidecar = btf_sidecar_path(&vmlinux);
        assert!(
            !sidecar.exists(),
            "precondition: sidecar must not pre-exist",
        );

        let btf =
            load_btf_from_path(&vmlinux).expect("load must succeed inside XDG-derived cache root");
        let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());

        assert!(
            sidecar.exists(),
            "sidecar must be written even when cascade resolves via XDG_CACHE_HOME \
             (KTSTR_CACHE_DIR=\"\")",
        );
    }

    /// Mid-process `KTSTR_CACHE_DIR` change: a load that wrote a
    /// sidecar under cache_a must produce no sidecar under cache_b
    /// for the same vmlinux on the next call after the env points
    /// at cache_b. Pins that membership resolution does not stick
    /// to a memoized first-call answer.
    #[test]
    fn load_btf_fresh_resolution_per_call() {
        let Some(path) = crate::monitor::find_test_vmlinux() else {
            return;
        };
        if path.starts_with("/sys/") {
            return;
        }

        let _env = crate::test_support::test_helpers::lock_env();
        // Two cache roots; vmlinux always sits inside cache_a.
        let cache_a = tempfile::TempDir::new().expect("cache_a tempdir");
        let cache_b = tempfile::TempDir::new().expect("cache_b tempdir");
        let entry_a = cache_a.path().join("kentry");
        std::fs::create_dir_all(&entry_a).expect("create cache_a entry");
        let vmlinux = entry_a.join("vmlinux");
        std::fs::copy(&path, &vmlinux).expect("copy vmlinux into cache_a");
        let sidecar = btf_sidecar_path(&vmlinux);

        // First call: KTSTR_CACHE_DIR points at cache_a → in-cache,
        // sidecar written.
        {
            let _g = crate::test_support::test_helpers::EnvVarGuard::set(
                "KTSTR_CACHE_DIR",
                cache_a.path(),
            );
            assert!(
                !sidecar.exists(),
                "precondition: sidecar must not pre-exist"
            );
            let btf = load_btf_from_path(&vmlinux).expect("first load must succeed");
            let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());
            assert!(
                sidecar.exists(),
                "first load (KTSTR_CACHE_DIR=cache_a) must write sidecar",
            );
            // Wipe sidecar so the second call's outcome is unambiguous.
            std::fs::remove_file(&sidecar).expect("remove sidecar between calls");
        }

        // Second call: KTSTR_CACHE_DIR moved to cache_b. The vmlinux
        // is still under cache_a, so it is now outside the active
        // cache → no sidecar should be written. A memoized cache
        // root resolution would surface here as a stale `true` and
        // the sidecar would reappear.
        {
            let _g = crate::test_support::test_helpers::EnvVarGuard::set(
                "KTSTR_CACHE_DIR",
                cache_b.path(),
            );
            let btf = load_btf_from_path(&vmlinux).expect("second load must succeed");
            let _ = format!("{:?}", btf.resolve_types_by_name("task_struct").is_ok());
            assert!(
                !sidecar.exists(),
                "second load (KTSTR_CACHE_DIR=cache_b) must NOT write sidecar — \
                 the vmlinux is now outside the active cache root",
            );
        }
    }

    // ---- probe.bpf.o atomic-op verification -----------------------
    //
    // The probe BPF program publishes the error-exit latch via
    // `__sync_val_compare_and_swap(&ktstr_err_exit_detected, 0u, 1u)`.
    // Cross-core ordering on weakly-ordered architectures (aarch64)
    // depends on the BPF backend lowering this to a real
    // `BPF_STX | BPF_ATOMIC | BPF_W` instruction with `BPF_CMPXCHG`
    // in the imm field — NOT a plain store.
    //
    // A toolchain regression that silently degraded the cmpxchg to a
    // non-atomic store would leave the latch's publication
    // unsynchronized, causing the freeze coordinator on a different
    // core to miss the transition under TSO-violating reorder. This
    // test pins the BPF bytecode against that regression by parsing
    // the compiled `probe.o` and asserting at least one atomic op is
    // present in the `tp_btf/sched_ext_exit` program section.
    //
    // BPF instruction encoding (uapi/linux/bpf.h):
    //   - opcode byte: bits[2:0] = class (BPF_STX = 0x03),
    //     bits[4:3] = size (BPF_W = 0x00, BPF_DW = 0x18),
    //     bits[7:5] = mode (BPF_ATOMIC = 0xc0).
    //     STX | ATOMIC | W = 0xc3.
    //   - imm field (4 bytes, little-endian): atomic op type.
    //     BPF_CMPXCHG = 0xf1 (= 0xf0 | BPF_FETCH).
    #[test]
    fn probe_bpf_object_emits_atomic_for_err_exit_latch() {
        // probe.o is produced by build.rs at OUT_DIR/probe.o. A
        // missing or unparseable file is a HARD FAIL, not a skip —
        // the whole point of the test is catching silent
        // regressions in the BPF backend lowering, and a silent
        // skip when the artifact is gone defeats that. build.rs
        // produces probe.o on every cargo build of the lib, so a
        // missing artifact here means the build pipeline is
        // broken and the test should surface that loudly.
        //
        // Limitation: this test counts ANY BPF_CMPXCHG instruction
        // in the `tp_btf/sched_ext_exit` section, not specifically
        // the cmpxchg targeting `ktstr_err_exit_detected`. Today
        // the section contains exactly one
        // `__sync_val_compare_and_swap` call (against the latch),
        // so any cmpxchg present must be the latch's; if a future
        // change adds a second atomic in the same handler, the
        // assert still passes but stops being a tight check on
        // the latch specifically. A future refinement could parse
        // the ELF relocation entries to constrain by symbol name
        // (look for a relocation referencing
        // `ktstr_err_exit_detected` adjacent to the cmpxchg
        // instruction).
        let probe_obj_path = std::path::PathBuf::from(env!("OUT_DIR")).join("probe.o");
        let bytes = std::fs::read(&probe_obj_path).unwrap_or_else(|e| {
            panic!(
                "probe.o missing or unreadable at {}: {e}. \
                 build.rs failed to produce the BPF skeleton — fix the \
                 build pipeline before re-running this test.",
                probe_obj_path.display()
            )
        });
        let elf = goblin::elf::Elf::parse(&bytes).unwrap_or_else(|e| {
            panic!(
                "probe.o at {} is not a valid ELF: {e}. \
                 The BPF skeleton emitter changed format or the file is \
                 corrupted — re-run the build to regenerate.",
                probe_obj_path.display()
            )
        });
        // Locate the program section. libbpf names sections after
        // the SEC() macro argument; tp_btf programs land in
        // `tp_btf/sched_ext_exit`. Match exact name; a future
        // restructure that splits the program into a different
        // section would surface as a clear test failure with the
        // expected name.
        const TARGET_SECTION: &str = "tp_btf/sched_ext_exit";
        let mut found_section = false;
        let mut atomic_count: usize = 0;
        for sh in &elf.section_headers {
            let Some(name) = elf.shdr_strtab.get_at(sh.sh_name) else {
                continue;
            };
            if name != TARGET_SECTION {
                continue;
            }
            found_section = true;
            // BPF programs are SHT_PROGBITS sections of `n` 8-byte
            // instructions. Read the section bytes via offset/size.
            let off = sh.sh_offset as usize;
            let sz = sh.sh_size as usize;
            assert!(
                sz.is_multiple_of(8),
                "BPF section size {sz} must be a multiple of 8 (instruction width)"
            );
            let prog = &bytes[off..off + sz];
            // BPF_STX | BPF_ATOMIC | BPF_W = 0xc3
            // BPF_STX | BPF_ATOMIC | BPF_DW = 0xc3 | 0x18 = 0xdb
            // The latch is u32, so we expect 0xc3 specifically — but
            // accept either width to keep the test robust against
            // a future widening to u64.
            const STX_ATOMIC_W: u8 = 0xc3;
            const STX_ATOMIC_DW: u8 = 0xdb;
            // BPF_CMPXCHG = 0xf0 | BPF_FETCH(0x01) = 0xf1
            const BPF_CMPXCHG_IMM: i32 = 0xf1;
            for chunk in prog.chunks_exact(8) {
                let opcode = chunk[0];
                if opcode == STX_ATOMIC_W || opcode == STX_ATOMIC_DW {
                    let imm = i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                    if imm == BPF_CMPXCHG_IMM {
                        atomic_count += 1;
                    }
                }
            }
        }
        assert!(
            found_section,
            "probe.o is missing the expected `{TARGET_SECTION}` section — \
             SEC() macro changed?"
        );
        assert!(
            atomic_count >= 1,
            "probe.o `{TARGET_SECTION}` section has no BPF_STX|BPF_ATOMIC|cmpxchg \
             instruction — `__sync_val_compare_and_swap` was silently \
             lowered to a non-atomic store. Cross-core ordering on aarch64 \
             would be broken by this regression."
        );
    }

    /// `ScxWalkerOffsets::missing_groups` enumerates exactly the
    /// sub-groups that failed to resolve, naming each with the kernel
    /// struct name the freeze coordinator surfaces in the failure
    /// dump's `scx_walker_unavailable` field. Operators parsing the
    /// failure-dump JSON look for these exact names — drift in the
    /// `missing.push("...")` literals here breaks the human-visible
    /// diagnostic string AND the structural pattern matching in any
    /// downstream tooling that buckets walker availability by group.
    ///
    /// Pin every sub-group's missing-name string by constructing
    /// `ScxWalkerOffsets` instances with each `Option` field set to
    /// `None` in isolation, asserting `missing_groups()` returns
    /// exactly that one name. A regression that renamed any push
    /// literal trips here. Also pins the empty-vec contract for the
    /// fully-populated case (no groups missing) and the multi-missing
    /// case (every group missing — the diagnostic must list all 10).
    #[test]
    fn scx_walker_missing_groups_pins_every_group_name() {
        // Construct a ScxWalkerOffsets with every sub-group resolved
        // to `Some(...)` placeholder — used as the baseline; tests
        // override one field at a time to None to exercise each push
        // arm of `missing_groups()`.
        fn full() -> ScxWalkerOffsets {
            ScxWalkerOffsets {
                rq: Some(RqStructOffsets { scx: 0, curr: 0 }),
                scx_rq: Some(ScxRqOffsets {
                    local_dsq: 0,
                    runnable_list: 0,
                    nr_running: 0,
                    flags: 0,
                    cpu_released: 0,
                    ops_qseq: 0,
                    kick_sync: 0,
                    nr_immed: 0,
                    clock: 0,
                }),
                task: Some(TaskStructCoreOffsets {
                    comm: 0,
                    pid: 0,
                    scx: 0,
                }),
                see: Some(SchedExtEntityOffsets {
                    runnable_node: 0,
                    runnable_at: 0,
                    weight: 0,
                    slice: 0,
                    dsq_vtime: 0,
                    dsq: 0,
                    dsq_list: 0,
                    flags: 0,
                    dsq_flags: 0,
                    sticky_cpu: 0,
                    holding_cpu: 0,
                }),
                dsq_lnode: Some(ScxDsqListNodeOffsets { node: 0, flags: 0 }),
                dsq: Some(ScxDispatchQOffsets {
                    list: 0,
                    nr: 0,
                    seq: 0,
                    id: 0,
                    hash_node: 0,
                }),
                sched: Some(ScxSchedOffsets {
                    dsq_hash: 0,
                    pnode: 0,
                    pcpu: 0,
                    aborting: 0,
                    bypass_depth: 0,
                    exit_kind: 0,
                }),
                sched_pnode: Some(ScxSchedPnodeOffsets { global_dsq: 0 }),
                sched_pcpu: Some(ScxSchedPcpuOffsets { bypass_dsq: 0 }),
                rht: Some(RhashtableOffsets {
                    tbl: 0,
                    nelems: 0,
                    bucket_table_size: 0,
                    bucket_table_buckets: 0,
                    rhash_head_next: 0,
                }),
            }
        }

        // Fully-populated: no sub-group missing. Empty vec is the
        // "every walker pass has data" sentinel — the freeze
        // coordinator only writes a partial-degradation diagnostic
        // when this list is non-empty.
        let all = full();
        assert!(
            all.missing_groups().is_empty(),
            "fully-populated offsets must report no missing groups; got {:?}",
            all.missing_groups(),
        );

        // Pin each group's missing-name string in isolation: drop
        // exactly one field, expect exactly one entry whose string
        // matches the canonical name. The pairs below enumerate the
        // 10 sub-groups; a regression adding/removing/renaming a
        // push arm trips here.
        let cases: &[(
            fn(&mut ScxWalkerOffsets),
            &'static str,
        )] = &[
            ((|o: &mut ScxWalkerOffsets| o.rq = None) as fn(&mut ScxWalkerOffsets), "rq"),
            (|o: &mut ScxWalkerOffsets| o.scx_rq = None, "scx_rq"),
            (|o: &mut ScxWalkerOffsets| o.task = None, "task_struct"),
            (
                |o: &mut ScxWalkerOffsets| o.see = None,
                "sched_ext_entity",
            ),
            (
                |o: &mut ScxWalkerOffsets| o.dsq_lnode = None,
                "scx_dsq_list_node",
            ),
            (
                |o: &mut ScxWalkerOffsets| o.dsq = None,
                "scx_dispatch_q",
            ),
            (|o: &mut ScxWalkerOffsets| o.sched = None, "scx_sched"),
            (
                |o: &mut ScxWalkerOffsets| o.sched_pnode = None,
                "scx_sched_pnode",
            ),
            (
                |o: &mut ScxWalkerOffsets| o.sched_pcpu = None,
                "scx_sched_pcpu",
            ),
            (
                |o: &mut ScxWalkerOffsets| o.rht = None,
                "rhashtable/bucket_table/rhash_head",
            ),
        ];
        for (drop_fn, expected_name) in cases {
            let mut o = full();
            drop_fn(&mut o);
            let missing = o.missing_groups();
            assert_eq!(
                missing.len(),
                1,
                "exactly one group should be missing; expected {expected_name:?}, got {missing:?}",
            );
            assert_eq!(
                missing[0], *expected_name,
                "missing-group name string drifted: expected {expected_name:?}, got {:?}",
                missing[0],
            );
        }

        // Every group missing: the order of the names must match
        // the order of the `if self.<field>.is_none()` arms in
        // `missing_groups()` so a downstream consumer reading the
        // failure dump sees a stable, predictable sequence (rq,
        // scx_rq, task_struct, sched_ext_entity, scx_dsq_list_node,
        // scx_dispatch_q, scx_sched, scx_sched_pnode, scx_sched_pcpu,
        // rhashtable/bucket_table/rhash_head). A regression that
        // shuffled the arms would silently break that ordering.
        let empty = ScxWalkerOffsets {
            rq: None,
            scx_rq: None,
            task: None,
            see: None,
            dsq_lnode: None,
            dsq: None,
            sched: None,
            sched_pnode: None,
            sched_pcpu: None,
            rht: None,
        };
        let missing = empty.missing_groups();
        assert_eq!(
            missing,
            vec![
                "rq",
                "scx_rq",
                "task_struct",
                "sched_ext_entity",
                "scx_dsq_list_node",
                "scx_dispatch_q",
                "scx_sched",
                "scx_sched_pnode",
                "scx_sched_pcpu",
                "rhashtable/bucket_table/rhash_head",
            ],
            "all-missing order must match the if-chain order in `missing_groups()`",
        );
    }
}
