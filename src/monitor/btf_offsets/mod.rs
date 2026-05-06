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
//! ([`HtabOffsets`]), BPF local-storage iteration
//! ([`TaskStorageOffsets`]), BPF program enumeration
//! ([`BpfProgOffsets`]), and shared `struct idr` walking
//! ([`IdrOffsets`]).

use std::path::Path;

use anyhow::{Context, Result, bail};
use btf_rs::{Btf, Type};

mod local_storage;
pub use local_storage::TaskStorageOffsets;
use local_storage::resolve_task_storage_offsets;

mod ringbuf_stackmap;
pub use ringbuf_stackmap::{BpfRingbufOffsets, BpfStackmapOffsets};
use ringbuf_stackmap::{resolve_ringbuf_offsets, resolve_stackmap_offsets};

mod struct_ops;
pub use struct_ops::StructOpsOffsets;
use struct_ops::resolve_struct_ops_offsets;

mod htab;
pub use htab::HtabOffsets;
use htab::resolve_htab_offsets;

mod cpu_time;
pub use cpu_time::{
    CPUTIME_IDLE, CPUTIME_IOWAIT, CPUTIME_IRQ, CPUTIME_NICE, CPUTIME_SOFTIRQ, CPUTIME_STEAL,
    CPUTIME_SYSTEM, CPUTIME_USER, CpuTimeOffsets, NR_SOFTIRQS,
};
// SOFTIRQ_NAMES is referenced from `dump/mod.rs` doc comments via
// the qualified path; the renderer that materializes it is pending.
#[allow(unused_imports)]
pub use cpu_time::SOFTIRQ_NAMES;

mod numa;
// NUMA event capture is pending: the wire shape and BTF resolver
// are landed but the live walker that consumes these constants
// has not shipped. Re-exports are kept here so the walker can
// land without a follow-up `pub use` change.
#[allow(unused_imports)]
pub use numa::{
    NR_VM_NUMA_EVENT_ITEMS, NUMA_EVENT_NAMES, NUMA_FOREIGN, NUMA_HIT, NUMA_INTERLEAVE_HIT,
    NUMA_LOCAL, NUMA_MISS, NUMA_OTHER, NumaStatsOffsets,
};

mod sched_domain;
use sched_domain::resolve_sched_domain_offsets;
pub use sched_domain::{CPU_MAX_IDLE_TYPES, SchedDomainOffsets, SchedDomainStatsOffsets};

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
    /// Resolve `struct rq`, `struct scx_rq`, and `struct
    /// scx_dispatch_q` field offsets from a pre-loaded BTF object.
    ///
    /// Callers that already hold a parsed [`Btf`] (e.g. the freeze
    /// coordinator's monitor thread, which loads it once and threads
    /// it into both `KernelOffsets::from_btf` and
    /// [`BpfProgOffsets::from_btf`]) avoid a second
    /// [`load_btf_from_path`] call. See [`Self::from_vmlinux`] for
    /// the path-based wrapper that does the BTF load itself.
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (rq_struct, _) = find_struct(btf, "rq")?;
        let rq_nr_running = member_byte_offset(btf, &rq_struct, "nr_running")?;
        let rq_clock = member_byte_offset(btf, &rq_struct, "clock")?;
        let (rq_scx, scx_member) = member_byte_offset_with_member(btf, &rq_struct, "scx")?;

        // Resolve the type of rq.scx to get struct scx_rq.
        let scx_rq_struct =
            resolve_member_struct(btf, &scx_member).context("btf: resolve type of rq.scx")?;
        let scx_rq_nr_running = member_byte_offset(btf, &scx_rq_struct, "nr_running")?;
        let (scx_rq_local_dsq, local_dsq_member) =
            member_byte_offset_with_member(btf, &scx_rq_struct, "local_dsq")?;
        let scx_rq_flags = member_byte_offset(btf, &scx_rq_struct, "flags")?;

        // Resolve the type of scx_rq.local_dsq to get struct scx_dispatch_q.
        let dsq_struct = resolve_member_struct(btf, &local_dsq_member)
            .context("btf: resolve type of scx_rq.local_dsq")?;
        let dsq_nr = member_byte_offset(btf, &dsq_struct, "nr")?;

        let event_offsets = resolve_event_offsets(btf).ok();
        let schedstat_offsets = resolve_schedstat_offsets(btf).ok();
        let sched_domain_offsets = resolve_sched_domain_offsets(btf, &rq_struct).ok();
        let watchdog_offsets = resolve_watchdog_offsets(btf).ok();

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

    /// Parse BTF from a vmlinux ELF and resolve field offsets for
    /// `struct rq`, `struct scx_rq`, and `struct scx_dispatch_q`.
    /// Thin wrapper around [`Self::from_btf`] for callers that have
    /// only the path; see [`Self::from_btf`] for the
    /// already-parsed-Btf entry point.
    #[allow(dead_code)]
    pub fn from_vmlinux(path: &Path) -> Result<Self> {
        let btf =
            load_btf_from_path(path).with_context(|| format!("btf: open {}", path.display()))?;
        Self::from_btf(&btf)
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

/// Outcome of [`find_struct_or_fwd`]: either a full struct definition
/// (with members and `.size()`) or a forward declaration (`BTF_KIND_FWD`
/// with `kind_flag == 0`, struct flavour) that names the type without a
/// body.
///
/// BPF program BTFs emit `BTF_KIND_FWD` for any struct whose body is
/// not needed by the program — typically structs the program only
/// dereferences via library helpers, never accessing members directly.
/// The scheduler BTF for lavd is the canonical example: `struct sdt_data`
/// is referenced only through `lib/sdt_alloc.bpf.c` allocator helpers,
/// so the program BTF surfaces it as a forward declaration with no
/// struct body. Code that needs only the layout invariants (e.g. that
/// `sizeof(struct sdt_data) == 8` because the only non-flex-array
/// member is the 8-byte `union sdt_id`) can proceed by hardcoding the
/// kernel-source-vetted size; code that needs member offsets must
/// surface a clear failure instead of crashing on missing members.
pub(crate) enum StructOrFwd {
    /// Full `BTF_KIND_STRUCT` with members and resolvable size.
    Full(btf_rs::Struct),
    /// `BTF_KIND_FWD` (struct flavour) — name resolves but the body
    /// is not present in this BTF.
    Fwd,
}

/// Find a named struct in BTF, accepting either a full struct
/// definition or a `BTF_KIND_FWD` forward declaration of struct
/// flavour.
///
/// Mirrors [`find_struct`] but does not require a struct body. Use this
/// when the caller can tolerate a missing body (e.g. `sdt_data` whose
/// 8-byte size is fixed by `lib/sdt_task_defs.h`'s `union sdt_id`
/// header member with no other non-flex-array members) and wants to
/// distinguish "absent from BTF" from "present but no body".
///
/// A full struct definition takes precedence over a forward
/// declaration when both appear under the same name in BTF — the
/// caller almost always prefers concrete member access where it's
/// available.
pub(crate) fn find_struct_or_fwd(btf: &Btf, name: &str) -> Result<StructOrFwd> {
    let types = btf
        .resolve_types_by_name(name)
        .with_context(|| format!("btf: type '{name}' not found"))?;

    let mut saw_fwd_struct = false;
    for t in &types {
        match t {
            Type::Struct(s) => return Ok(StructOrFwd::Full(s.clone())),
            Type::Fwd(f) if f.is_struct() => saw_fwd_struct = true,
            _ => {}
        }
    }
    if saw_fwd_struct {
        return Ok(StructOrFwd::Fwd);
    }
    bail!("btf: '{name}' exists but is neither a struct nor a struct-flavour fwd")
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
pub(super) fn member_byte_offset_with_member(
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
pub(super) fn resolve_member_struct(btf: &Btf, member: &btf_rs::Member) -> Result<btf_rs::Struct> {
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
    /// Offset of `btf_vmlinux_value_type_id` (u32) within `struct
    /// bpf_map`. Populated by libbpf for `BPF_MAP_TYPE_STRUCT_OPS`
    /// maps with the kernel's `bpf_struct_ops_<name>` wrapper
    /// type id from vmlinux BTF (libbpf zeros `btf_value_type_id`
    /// for STRUCT_OPS — see
    /// `tools/lib/bpf/libbpf.c::bpf_object__create_maps`).
    /// Zero on every other map type. The dump path uses it to
    /// render the data payload of a STRUCT_OPS map when
    /// `btf_value_type_id` is absent, walking the wrapper's `data`
    /// member to reach the per-ops struct.
    pub map_btf_vmlinux_value_type_id: usize,
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
    /// Offsets for `bpf_local_storage_map` walking (TASK_STORAGE and
    /// the shape-identical INODE/SK/CGRP_STORAGE variants). None if
    /// BTF lacks any of `bpf_local_storage_map` /
    /// `bpf_local_storage_map_bucket` / `bpf_local_storage_elem` /
    /// `bpf_local_storage_data` / `bpf_local_storage`.
    pub task_storage_offsets: Option<TaskStorageOffsets>,
    /// Offsets for reading `BPF_MAP_TYPE_STRUCT_OPS` value bytes from
    /// guest memory. `None` if BTF lacks `bpf_struct_ops_map` or
    /// `bpf_struct_ops_value` (kernels built without struct_ops
    /// support).
    pub struct_ops_offsets: Option<StructOpsOffsets>,
    /// Offsets for reading `BPF_MAP_TYPE_RINGBUF` /
    /// `BPF_MAP_TYPE_USER_RINGBUF` consumer/producer positions. None
    /// if BTF lacks `bpf_ringbuf_map` or `bpf_ringbuf` (kernels older
    /// than v5.8 or built without the ringbuf map type).
    pub ringbuf_offsets: Option<BpfRingbufOffsets>,
    /// Offsets for reading `BPF_MAP_TYPE_STACK_TRACE` bucket array.
    /// None if BTF lacks `bpf_stack_map` / `stack_map_bucket`.
    pub stackmap_offsets: Option<BpfStackmapOffsets>,
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
        map_btf_vmlinux_value_type_id: 0,
        map_btf_key_type_id: 0,
        btf_data: 0,
        btf_data_size: 0,
        btf_base_btf: 0,
        htab_offsets: None,
        task_storage_offsets: None,
        struct_ops_offsets: None,
        ringbuf_offsets: None,
        stackmap_offsets: None,
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
        // `btf_vmlinux_value_type_id` is conditionally compiled in
        // (`CONFIG_BPF_JIT` gates the kernel struct_ops support that
        // populates it). Tolerate absence so non-struct_ops kernels
        // still resolve the rest of the offsets — the STRUCT_OPS arm
        // checks the offset and falls through to hex when zero.
        let map_btf_vmlinux_value_type_id =
            member_byte_offset(btf, &bpf_map, "btf_vmlinux_value_type_id").unwrap_or(0);
        let map_btf_key_type_id = member_byte_offset(btf, &bpf_map, "btf_key_type_id")?;

        let (btf_struct, _) = find_struct(btf, "btf")?;
        let btf_data = member_byte_offset(btf, &btf_struct, "data")?;
        let btf_data_size = member_byte_offset(btf, &btf_struct, "data_size")?;
        let btf_base_btf = member_byte_offset(btf, &btf_struct, "base_btf")?;

        let htab_offsets = resolve_htab_offsets(btf).ok();
        let task_storage_offsets = resolve_task_storage_offsets(btf).ok();
        let struct_ops_offsets = resolve_struct_ops_offsets(btf).ok();
        let ringbuf_offsets = resolve_ringbuf_offsets(btf).ok();
        let stackmap_offsets = resolve_stackmap_offsets(btf).ok();

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
            map_btf_vmlinux_value_type_id,
            map_btf_key_type_id,
            btf_data,
            btf_data_size,
            btf_base_btf,
            htab_offsets,
            task_storage_offsets,
            struct_ops_offsets,
            ringbuf_offsets,
            stackmap_offsets,
        })
    }
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
//      (deduplicated source of truth).
//   2. Higher-level structs that need only some groups can degrade
//      gracefully: a missing `scx_sched_pnode` group blinds the global
//      DSQ walk pass but leaves rq->scx + per-CPU local DSQ walks
//      working (graceful degradation).
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
    /// `scx_dump_state` ("ksync=%lu") on kernels that have it.
    /// Optional and version-renamed:
    ///
    /// - v6.12 → v6.18: the underlying field is named `pnt_seq`.
    ///   This struct stores its offset as `kick_sync` regardless,
    ///   since the data semantics (per-rq pick-next sequence
    ///   counter) are unchanged across the rename.
    /// - v6.19+: kernel rename to `kick_sync` lands (commit
    ///   "sched_ext: Rename pnt_seq to kick_sync"); BTF reports
    ///   the field under the new name and the dump uses
    ///   `ksync=…` for the rendering.
    /// - None: neither `kick_sync` nor `pnt_seq` resolved (the
    ///   kernel BTF was stripped, or sched_ext was not built in).
    ///   Consumers that render this field must skip the
    ///   `ksync=…` line when None.
    ///
    /// Resolution falls back from `kick_sync` to `pnt_seq` so a
    /// single `Option<usize>` covers the entire kernel range
    /// without leaking the rename into the consumer.
    pub kick_sync: Option<usize>,
    /// Offset of `nr_immed` (u32). NOT read by
    /// `scx_dump_state`; ktstr collects it for the host-side
    /// dump's ENQ_IMMED diagnosis path (kernel updates the
    /// counter in `do_enqueue_task` ENQ_IMMED branches and
    /// elsewhere in `kernel/sched/ext.c`).
    /// Optional and post-release: SCX_ENQ_IMMED is on a feature
    /// branch (`for-7.1`) and is absent on every release tag in
    /// our supported range (v6.12 → v7.0-rc5). Consumers that
    /// render this field elide the leg when None.
    pub nr_immed: Option<usize>,
    /// Offset of `clock` (u64). Per-rq scx clock; ktstr surfaces
    /// it for cross-CPU clock-skew analysis. NOT read by
    /// `scx_dump_state`.
    /// Optional and version-gated: `rq->scx.clock` was added by
    /// the `scx_bpf_now()` series in v6.14
    /// (commit 3a9910b5904d). v6.12 and v6.13 kernels lack the
    /// field — keeping `clock` mandatory on those versions
    /// would break BTF resolution of the entire ScxRqOffsets and
    /// every walker that gates on it. Consumers that surface the
    /// scx clock gate on Some; downstream RqScxState carries an
    /// `Option<u64>` so the JSON elides the field on absent
    /// kernels.
    pub clock: Option<usize>,
}

impl ScxRqOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (scx_rq, _) = find_struct(btf, "scx_rq")?;
        // kick_sync rename: try the v6.19+ name first, then the
        // legacy v6.12-v6.18 name. The Result-then-fallback
        // pattern keeps the storage Option<usize> so consumers
        // don't need to know which kernel they're on.
        let kick_sync = member_byte_offset(btf, &scx_rq, "kick_sync")
            .ok()
            .or_else(|| member_byte_offset(btf, &scx_rq, "pnt_seq").ok());
        Ok(Self {
            local_dsq: member_byte_offset(btf, &scx_rq, "local_dsq")?,
            runnable_list: member_byte_offset(btf, &scx_rq, "runnable_list")?,
            nr_running: member_byte_offset(btf, &scx_rq, "nr_running")?,
            flags: member_byte_offset(btf, &scx_rq, "flags")?,
            cpu_released: member_byte_offset(btf, &scx_rq, "cpu_released")?,
            ops_qseq: member_byte_offset(btf, &scx_rq, "ops_qseq")?,
            kick_sync,
            // nr_immed: feature-branch field, never on a release
            // tag in our supported range. `.ok()` makes the
            // member-lookup absence → None.
            nr_immed: member_byte_offset(btf, &scx_rq, "nr_immed").ok(),
            // clock: added in v6.14 (commit 3a9910b5904d). v6.12
            // and v6.13 lack the field. `.ok()` lets BTF
            // resolution succeed on those releases — consumers
            // that need the clock value gate on Some.
            clock: member_byte_offset(btf, &scx_rq, "clock").ok(),
        })
    }
}

/// Universal-subset field offsets within `struct task_struct`. The
/// fields here are read by every walker that touches a task — the
/// runnable scanner, the rq->scx walker, the DSQ walker, and the
/// task_enrichment walker. Resolved once and shared so
/// `task_struct.scx` etc. exist as a single source of truth.
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
    /// Offset of `tasks_node` (struct list_head). Links every
    /// scx-managed task into the kernel's global `scx_tasks`
    /// list (kernel/sched/ext.c:47). The host walker uses this
    /// to enumerate every task owned by an scx_sched, surviving
    /// the per-rq runnable_list drain that scx_bypass triggers
    /// during scheduler teardown (kernel/sched/ext.c:5341).
    pub tasks_node: usize,
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
            tasks_node: member_byte_offset(btf, &see, "tasks_node")?,
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
///
/// `scx_sched` itself does not exist on v6.12 / v6.13 kernels — the
/// containing struct landed with the per-sched-instance refactor that
/// introduced the `scx_root` indirection. Callers gate on
/// `Option<ScxSchedOffsets>` (`ScxWalkerOffsets::sched`) before
/// constructing this; once the struct exists, several internal fields
/// are also kernel-version-gated, so each is `Option<usize>`.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxSchedOffsets {
    /// Offset of `dsq_hash` (struct rhashtable). User-allocated DSQs.
    /// Present on every kernel that has `scx_sched` itself
    /// (v6.14+ release line).
    pub dsq_hash: usize,
    /// Offset of `pnode` (`struct scx_sched_pnode **`). Per-NUMA-node
    /// global DSQ array. Optional: `pnode` only exists on
    /// development branches that introduced the per-node split; no
    /// release tag in our supported range carries it.
    pub pnode: Option<usize>,
    /// Offset of `pcpu` (`struct scx_sched_pcpu __percpu *`). Per-CPU
    /// scx_sched data. Optional: added in v6.18+ alongside the
    /// per-CPU bypass DSQ split — earlier release lines that have
    /// `scx_sched` (v6.14 → v6.17) lack the field.
    pub pcpu: Option<usize>,
    /// Offset of `aborting` (bool). Optional: development-only field
    /// covering an in-flight scheduler abort path; absent on every
    /// release tag in our supported range.
    pub aborting: Option<usize>,
    /// Offset of `bypass_depth` (s32). Optional: development-only,
    /// see `aborting` above.
    pub bypass_depth: Option<usize>,
    /// Offset of `exit_kind` (atomic_t). Read raw; the SCX_EXIT_*
    /// value lives in the atomic's `counter` field. Present on every
    /// kernel that has `scx_sched` itself.
    pub exit_kind: usize,
}

impl ScxSchedOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (sched, _) = find_struct(btf, "scx_sched")?;
        Ok(Self {
            dsq_hash: member_byte_offset(btf, &sched, "dsq_hash")?,
            // Version-gated fields: collapse member-lookup absence
            // to None so a partial `scx_sched` layout still produces
            // an `Ok(ScxSchedOffsets)`. Consumers that read these
            // gate on Some.
            pnode: member_byte_offset(btf, &sched, "pnode").ok(),
            pcpu: member_byte_offset(btf, &sched, "pcpu").ok(),
            aborting: member_byte_offset(btf, &sched, "aborting").ok(),
            bypass_depth: member_byte_offset(btf, &sched, "bypass_depth").ok(),
            exit_kind: member_byte_offset(btf, &sched, "exit_kind")?,
        })
    }
}

/// Field offsets within `struct scx_sched_pnode`. The struct itself
/// is development-only (no released tag in our supported range), so
/// the parent `ScxWalkerOffsets::sched_pnode` is `Option<…>`. Within
/// the struct, the single `global_dsq` field is also dev-only.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxSchedPnodeOffsets {
    /// Offset of `global_dsq` (struct scx_dispatch_q). Per-NUMA-node
    /// global DSQ. Optional: development-only field; consumers that
    /// reach into the per-node global DSQ gate on Some.
    pub global_dsq: Option<usize>,
}

impl ScxSchedPnodeOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (pnode, _) = find_struct(btf, "scx_sched_pnode")?;
        Ok(Self {
            // global_dsq is dev-only — collapse absence to None so a
            // BTF that exposes a future `scx_sched_pnode` shape
            // without `global_dsq` still yields Ok.
            global_dsq: member_byte_offset(btf, &pnode, "global_dsq").ok(),
        })
    }
}

/// Field offsets within `struct scx_sched_pcpu`. The struct landed
/// in v6.18 alongside the per-CPU bypass split; the parent
/// `ScxWalkerOffsets::sched_pcpu` is `Option<…>` so kernels < v6.18
/// don't fail BTF resolution. Inside the struct, the single
/// `bypass_dsq` field is dev-only.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ScxSchedPcpuOffsets {
    /// Offset of `bypass_dsq` (struct scx_dispatch_q). Per-CPU
    /// bypass DSQ. Optional: development-only field; consumers that
    /// reach into the per-CPU bypass DSQ gate on Some.
    pub bypass_dsq: Option<usize>,
}

impl ScxSchedPcpuOffsets {
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let (pcpu, _) = find_struct(btf, "scx_sched_pcpu")?;
        Ok(Self {
            // bypass_dsq is dev-only — same .ok() rationale as
            // ScxSchedPnodeOffsets::global_dsq above.
            bypass_dsq: member_byte_offset(btf, &pcpu, "bypass_dsq").ok(),
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
/// global `runnable_at` scanner.
///
/// The scanner walks the kernel's global `scx_tasks` LIST_HEAD
/// (`kernel/sched/ext.c:47`) — every scx-managed task is linked into
/// it via `task_struct.scx.tasks_node`. Walking the global list (not
/// the per-rq `runnable_list`) keeps the scanner functional through
/// scheduler teardown: `scx_bypass`
/// (`kernel/sched/ext.c:5304-5404`) drains every per-rq
/// `runnable_list` while leaving `scx_tasks` intact, so a per-rq
/// walk would see an empty list at exactly the moment the dual
/// snapshot's early trigger needs to fire. For each task on the
/// list the scanner does a container_of via `task_struct.scx +
/// sched_ext_entity.tasks_node` to recover the `task_struct` KVA,
/// reads `task_struct.scx.runnable_at`, and compares against the
/// current `jiffies_64`. Cursor entries (stack-allocated
/// `sched_ext_entity` placeholders that `scx_task_iter_start`
/// inserts with `SCX_TASK_CURSOR` set) are filtered via
/// `sched_ext_entity.flags`. All offsets resolve from a single
/// `Btf` object loaded from vmlinux.
///
/// Path uses `member_byte_offset` calls on `task_struct` and
/// `sched_ext_entity`. The `task_struct.scx` field resolves using
/// the BTF's anonymous-struct-walking `member_byte_offset`, so the
/// path works even if the kernel later wraps the field in an
/// anonymous union.
#[derive(Debug, Clone, Copy)]
pub struct RunnableScanOffsets {
    /// Offset of `scx` (struct sched_ext_entity) within
    /// `struct task_struct`. Used by the container_of step to
    /// translate a `tasks_node` list_head pointer back to the
    /// owning `task_struct`.
    pub task_struct_scx: usize,
    /// Offset of `tasks_node` (struct list_head) within
    /// `struct sched_ext_entity`. The full offset of `tasks_node`
    /// within `task_struct` is
    /// `task_struct_scx + sched_ext_entity_tasks_node` — subtract
    /// this from a `tasks_node` KVA to recover the `task_struct`
    /// KVA via container_of. The kernel's global `scx_tasks`
    /// LIST_HEAD (`kernel/sched/ext.c:47`) links every scx-managed
    /// task via this field; walking it survives the per-rq
    /// `runnable_list` drain that `scx_bypass`
    /// (`kernel/sched/ext.c:5304-5404`) performs during scheduler
    /// teardown.
    pub sched_ext_entity_tasks_node: usize,
    /// Offset of `flags` (u32) within `struct sched_ext_entity`.
    /// Read off each `scx_tasks` list entry to skip cursor
    /// placeholders that `scx_task_iter_start`
    /// (`kernel/sched/ext.c:843-846`) inserts with `SCX_TASK_CURSOR`
    /// (`1 << 31`) set. Cursor entries are stack-allocated, not
    /// embedded in a `task_struct`, so container_of would synthesize
    /// a bogus task KVA without this skip.
    pub sched_ext_entity_flags: usize,
    /// Offset of `runnable_at` (unsigned long, jiffies) within
    /// `struct sched_ext_entity`. Combined with `task_struct_scx`
    /// to read the field off a `task_struct *`:
    /// `task + task_struct_scx + sched_ext_entity_runnable_at`.
    pub sched_ext_entity_runnable_at: usize,
    /// Offset of `runnable_node` (struct list_head) within
    /// `struct sched_ext_entity`. Per-rq runnable list entries link
    /// through this node, NOT `tasks_node`. Recovering the owning
    /// `task_struct` from a per-rq list entry requires
    /// `task_kva = node_kva - (task_struct_scx +
    /// sched_ext_entity_runnable_node)`. Used by the per-CPU
    /// `runnable_list` walker that mirrors the kernel's
    /// `check_rq_for_timeouts` (kernel/sched/ext.c).
    pub sched_ext_entity_runnable_node: usize,
    /// Offset of `scx` (struct scx_rq) within `struct rq`. Combined
    /// with `scx_rq_runnable_list` to address the per-CPU list head
    /// off a `struct rq *` pointer:
    /// `runnable_list_kva = rq_kva + rq_scx + scx_rq_runnable_list`.
    pub rq_scx: usize,
    /// Offset of `runnable_list` (struct list_head) within
    /// `struct scx_rq`. Head of the per-CPU runnable task list the
    /// kernel's own watchdog (`check_rq_for_timeouts`,
    /// `kernel/sched/ext.c`) walks. Tasks on this list carry live
    /// `runnable_at` stamps; the global `scx_tasks` list does not
    /// (its stamps are re-stamped on every enqueue and aren't
    /// definitive evidence of an aged stuck task).
    pub scx_rq_runnable_list: usize,
}

impl RunnableScanOffsets {
    /// Resolve runnable_at scanner offsets from a pre-loaded BTF
    /// object. Returns Err on a kernel without sched_ext (the
    /// `sched_ext_entity` struct is missing) or one whose layout has
    /// dropped any of the seven fields.
    ///
    /// Composed from [`TaskStructCoreOffsets`], [`SchedExtEntityOffsets`],
    /// [`RqStructOffsets`], and [`ScxRqOffsets`] so each kernel field
    /// resolves from a single source of truth shared with
    /// [`ScxWalkerOffsets`] and [`TaskEnrichmentOffsets`].
    pub fn from_btf(btf: &Btf) -> Result<Self> {
        let task_core = TaskStructCoreOffsets::from_btf(btf)?;
        let see = SchedExtEntityOffsets::from_btf(btf)?;
        let rq = RqStructOffsets::from_btf(btf)?;
        let scx_rq = ScxRqOffsets::from_btf(btf)?;

        Ok(Self {
            task_struct_scx: task_core.scx,
            sched_ext_entity_tasks_node: see.tasks_node,
            sched_ext_entity_flags: see.flags,
            sched_ext_entity_runnable_at: see.runnable_at,
            sched_ext_entity_runnable_node: see.runnable_node,
            rq_scx: rq.scx,
            scx_rq_runnable_list: scx_rq.runnable_list,
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
// freeze coordinator passes None until the
// rq->scx walker lands a producer that builds
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
    /// (deduplicated source of truth).
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
    /// with [`RunnableScanOffsets`] and [`TaskEnrichmentOffsets`].
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
mod tests;
