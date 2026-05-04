//! /proc parsers and tallying readers extracted from
//! `super::mod.rs`. Holds:
//! - `parse_psi` / `parse_centi_percent` and the `read_*psi_at`
//!   helpers that wrap them
//! - `read_sched_ext_sysfs_at` + `read_sysfs_u64`
//! - `parse_stat` / `parse_schedstat` / `parse_io` / `parse_status`
//!   plus their `read_*_at_with_tally` wrappers
//! - `parse_cgroup_v2` / `read_cgroup_at*`
//! - `parse_sched` and the `parsed_ns_from_dotted` half-millisecond
//!   recovery (with the `ParseDottedNs` discriminator)
//! - `parse_smaps_rollup` / `read_smaps_rollup_at_with_tally`
//! - `parse_cpu_stat` / `parse_kv_counters` / `parse_max_or_u64*` /
//!   `parse_floor_value` / `parse_cpu_max`
//! - `read_cgroup_stats_at` — opens the cgroup v2 files actually
//!   read by capture: `cpu.stat`, `cpu.max`, `cpu.weight`,
//!   `cpu.weight.nice`, `memory.current`, `memory.max`,
//!   `memory.high`, `memory.low`, `memory.min`, `memory.stat`,
//!   `memory.events`, `pids.current`, `pids.max`, plus the
//!   `<cgroup>.pressure` PSI files via `read_cgroup_psi_at`.
//!   `memory.swap.current`, `memory.peak`, `memory.zswap.current`,
//!   `pids.peak`, and `io.stat` are NOT read — the cgroup capture
//!   surface intentionally limits to the v2 fields the comparator
//!   schemas the rest of the pipeline.
//!
//! ## Visibility
//!
//! The parsers and `read_*` helpers are `pub(super)` so the capture
//! pipeline in `mod.rs` can call them without re-exporting through
//! the public API. The serialization surface
//! ([`CtprofSnapshot::load`](super::CtprofSnapshot::load) /
//! [`write`](super::CtprofSnapshot::write)) and the snapshot
//! constants stay `pub` on the parent because they ARE part of the
//! ktstr public API, consumed by `cargo ktstr ctprof compare` and
//! by the snapshot-loader crate consumers.

use super::*;
use std::collections::BTreeMap;
use std::path::Path;

/// Parse one PSI file's contents. The kernel emits one or two
/// lines (`some` then `full`), each formatted by `seq_printf` at
/// `kernel/sched/psi.c:1284`. Lines are tokenized by whitespace;
/// each token is `key=value`. Unknown keys are ignored so a
/// future kernel that adds a 4th avg or new field doesn't break
/// the parser. Missing fields default to 0 (matching the
/// absent-counter contract used elsewhere in this module).
pub(super) fn parse_psi(raw: &str) -> PsiResource {
    let mut out = PsiResource::default();
    for line in raw.lines() {
        let mut tokens = line.split_whitespace();
        let Some(prefix) = tokens.next() else {
            continue;
        };
        let half = match prefix {
            "some" => &mut out.some,
            "full" => &mut out.full,
            _ => continue,
        };
        for tok in tokens {
            let Some((key, value)) = tok.split_once('=') else {
                continue;
            };
            match key {
                "avg10" => half.avg10 = parse_centi_percent(value),
                "avg60" => half.avg60 = parse_centi_percent(value),
                "avg300" => half.avg300 = parse_centi_percent(value),
                "total" => half.total_usec = value.parse::<u64>().unwrap_or(0),
                _ => {}
            }
        }
    }
    out
}

/// Convert `"N.NN"` (kernel `%lu.%02lu` format from psi.c:1284)
/// to `N * 100 + NN` (centi-percent integer). On malformed input
/// returns 0, matching the absent-counter default contract.
/// Saturates at u16::MAX to guard against pathological input.
///
/// The kernel always emits a 2-digit zero-padded fraction
/// (`%02lu`), but a robust parser zero-pads its own input to
/// exactly 2 digits before combining: a stray `"1.5"` (one
/// fractional digit) must read as `150` (1.50%), not `105`
/// (1.05%); a stray `"1.501"` (three fractional digits) is
/// truncated to `1.50` rather than producing
/// `1*100 + 501 = 601`. Mirrors the
/// [`parsed_ns_from_dotted`] helper's zero-pad-to-six discipline.
pub(super) fn parse_centi_percent(s: &str) -> u16 {
    let (int_part, frac_part) = s.split_once('.').unwrap_or((s, ""));
    let Ok(int) = int_part.parse::<u32>() else {
        return 0;
    };
    let frac = if frac_part.is_empty() {
        0
    } else {
        // Zero-pad-to-2 then truncate-to-2: "5" → "50", "501" →
        // "50". Matches the kernel's `%02lu` format width
        // exactly so a parser-side roundtrip can never under- or
        // over-count the fractional weight.
        let padded: String = frac_part
            .chars()
            .chain(std::iter::repeat('0'))
            .take(2)
            .collect();
        padded.parse::<u32>().unwrap_or(0)
    };
    let combined = int.saturating_mul(100).saturating_add(frac);
    combined.try_into().unwrap_or(u16::MAX)
}

/// Read host-level PSI files (`<proc_root>/pressure/{cpu,memory,io,irq}`)
/// and populate a [`Psi`] bundle. Each file is read independently;
/// absent files (older kernels missing irq.pressure, or hosts
/// with CONFIG_PSI off) collapse to the all-zero default per the
/// absent-counter contract.
///
/// PSI readers (this fn, `read_cgroup_psi_at`) and the
/// `read_sched_ext_sysfs_at` reader deliberately omit the
/// `ParseTally` argument that the per-thread procfs readers
/// thread through. Their build-gate signal is presence of the
/// containing directory (`/proc/pressure/`,
/// `/sys/kernel/sched_ext/`): an absent directory means the
/// kernel feature is off, which is a host-property fact rather
/// than a per-tid parse failure, and the snapshot's all-zero
/// default already encodes the absence. Threading these
/// readers into the tally would multiply the failure tally by
/// the worker count without adding any operator-actionable
/// signal beyond what the absent fields already convey.
pub(super) fn read_host_psi_at(proc_root: &Path) -> Psi {
    let pressure_dir = proc_root.join("pressure");
    Psi {
        cpu: read_psi_file_at(&pressure_dir.join("cpu")),
        memory: read_psi_file_at(&pressure_dir.join("memory")),
        io: read_psi_file_at(&pressure_dir.join("io")),
        irq: read_psi_file_at(&pressure_dir.join("irq")),
    }
}

/// Read global sched_ext sysfs state from
/// `<sys_root>/kernel/sched_ext/`. Returns `None` when the
/// directory itself is absent (CONFIG_SCHED_CLASS_EXT=n
/// kernels never expose it). Per-file misses default the
/// affected field to zero / empty string per the
/// absent-counter contract — a future kernel that adds new
/// global attrs (and that we haven't surfaced as fields yet)
/// won't break the parser; old kernels missing one or more of
/// the existing five collapse cleanly.
pub(super) fn read_sched_ext_sysfs_at(sys_root: &Path) -> Option<SchedExtSysfs> {
    let dir = sys_root.join("kernel").join("sched_ext");
    // No `tally` arg: directory presence (Option<SchedExtSysfs>)
    // is THE not-built signal; per-attr misses collapse silently
    // per the absent-counter contract.
    if !dir.exists() {
        return None;
    }
    Some(SchedExtSysfs {
        state: fs::read_to_string(dir.join("state"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default(),
        switch_all: read_sysfs_u64(&dir.join("switch_all")),
        nr_rejected: read_sysfs_u64(&dir.join("nr_rejected")),
        hotplug_seq: read_sysfs_u64(&dir.join("hotplug_seq")),
        enable_seq: read_sysfs_u64(&dir.join("enable_seq")),
    })
}

/// Read a single-line u64 sysfs file. Trims trailing newline,
/// parses, defaults to 0 on read or parse failure (matches the
/// absent-counter contract).
pub(super) fn read_sysfs_u64(path: &Path) -> u64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Read per-cgroup PSI files (`<cgroup>/{cpu,memory,io,irq}.pressure`)
/// and populate a [`Psi`] bundle. The four files are exposed by
/// `kernel/cgroup/cgroup.c:5453-5482`; the per-cgroup interface
/// uses the `<resource>.pressure` filename pattern rather than
/// the host-level `pressure/<resource>` directory layout.
pub(super) fn read_cgroup_psi_at(cgroup_root: &Path, path: &str) -> Psi {
    let relative = path.strip_prefix('/').unwrap_or(path);
    let dir = if relative.is_empty() {
        cgroup_root.to_path_buf()
    } else {
        cgroup_root.join(relative)
    };
    Psi {
        cpu: read_psi_file_at(&dir.join("cpu.pressure")),
        memory: read_psi_file_at(&dir.join("memory.pressure")),
        io: read_psi_file_at(&dir.join("io.pressure")),
        irq: read_psi_file_at(&dir.join("irq.pressure")),
    }
}

/// Read one PSI file by path. Absent files or read errors
/// collapse to a default-zero [`PsiResource`].
pub(super) fn read_psi_file_at(path: &Path) -> PsiResource {
    fs::read_to_string(path)
        .ok()
        .as_deref()
        .map(parse_psi)
        .unwrap_or_default()
}

impl CtprofSnapshot {
    /// Load a snapshot from a zstd-compressed JSON file.
    ///
    /// Errors propagate via [`anyhow`] with the source path in the
    /// context chain so a malformed file surfaces an actionable
    /// message rather than a generic deserialize error. The loader
    /// does not validate that `threads` is non-empty — an empty
    /// snapshot is a legitimate edge case (host idle, capture
    /// filter excluded every thread) and the comparison engine
    /// handles it by emitting an empty diff.
    ///
    /// The decompression step is bounded by
    /// [`MAX_DECOMPRESSED_SNAPSHOT_BYTES`] — a payload that
    /// decompresses past that ceiling surfaces an error rather
    /// than allocating unbounded memory, guarding against a
    /// hostile zstd payload (zstd compresses pathologically well
    /// on repeated bytes).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let bytes = std::fs::read(path)
            .with_context(|| format!("read ctprof snapshot from {}", path.display()))?;
        let json = decompress_capped(&bytes, MAX_DECOMPRESSED_SNAPSHOT_BYTES)
            .with_context(|| format!("zstd decompress ctprof snapshot {}", path.display()))?;
        let snap: CtprofSnapshot = serde_json::from_slice(&json).with_context(|| {
            format!(
                "parse ctprof snapshot JSON from {} (did the capture format change?)",
                path.display(),
            )
        })?;
        Ok(snap)
    }

    /// Write a snapshot as zstd-compressed JSON.
    ///
    /// Used by the capture layer; exposed from this module so that
    /// both compare-side tests and the capture binary share one
    /// on-disk shape. Compression level `3` mirrors the ktstr
    /// remote-cache convention — adequate ratio at fast speed —
    /// and is not tunable because ctprof captures are small
    /// enough that further compression produces diminishing
    /// returns on I/O.
    pub fn write(&self, path: &std::path::Path) -> anyhow::Result<()> {
        use anyhow::Context;
        let json = serde_json::to_vec(self).context("serialize ctprof snapshot to JSON")?;
        let compressed =
            zstd::encode_all(json.as_slice(), 3).context("zstd compress ctprof snapshot")?;
        std::fs::write(path, compressed)
            .with_context(|| format!("write ctprof snapshot to {}", path.display()))?;
        Ok(())
    }
}

/// Decompress a zstd payload into a `Vec<u8>` capped at
/// `max_decompressed` bytes — bombing out with an error if the
/// payload would expand past the ceiling. Reads through
/// `Read::take(cap + 1)` so a payload that decompresses to
/// exactly `cap` bytes is accepted while one that produces
/// `cap + 1` bytes (or more) is rejected — the +1 sentinel
/// distinguishes "EOF coincided with the cap" from "more data
/// behind the cap".
pub(super) fn decompress_capped(bytes: &[u8], max_decompressed: u64) -> anyhow::Result<Vec<u8>> {
    use std::io::Read;
    let decoder = zstd::stream::read::Decoder::new(bytes)?;
    let mut out = Vec::new();
    decoder
        .take(max_decompressed.saturating_add(1))
        .read_to_end(&mut out)?;
    if out.len() as u64 > max_decompressed {
        anyhow::bail!(
            "zstd-decompressed payload exceeds the {}-byte cap (decompression-bomb guard)",
            max_decompressed,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------
// Capture layer: procfs readers + host walk.
// ---------------------------------------------------------------

/// Canonical file extension for a serialized snapshot.
///
/// `dead_code` allow: referenced from doc comments in
/// `monitor::debug_capture` and the `metric_types` overview. The
/// extension is hardcoded as the literal `"ctprof.zst"` at every
/// production write/load site (the CLI accepts any path the
/// operator supplies and the renderer reads via
/// [`CtprofSnapshot::load`]). Kept as a named constant so a future
/// caller that needs to construct paths from scratch has the
/// canonical token available without re-typing the literal.
#[allow(dead_code)]
pub const SNAPSHOT_EXTENSION: &str = "ctprof.zst";

/// Decompressed-size ceiling for [`CtprofSnapshot::load`].
/// Bounds the allocation a malicious or corrupted zstd payload
/// can force, since zstd compresses pathologically well on
/// repeated bytes (a few-KiB compressed blob can decompress to
/// gigabytes). 256 MiB covers any realistic production snapshot
/// (typical hosts run 1K-100K live threads) while bounding
/// worst-case allocation against hostile zstd payloads.
/// Public so a downstream consumer can size buffers against the
/// same ceiling without hardcoding the value.
pub const MAX_DECOMPRESSED_SNAPSHOT_BYTES: u64 = 256 * 1024 * 1024;

/// Default procfs root on Linux. The `_at` readers accept any
/// `&Path` so tests stage a synthetic tree under a tempdir; the
/// public readers delegate to those with this default.
pub const DEFAULT_PROC_ROOT: &str = "/proc";

/// Default cgroup v2 mount point.
pub const DEFAULT_CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// Default sysfs root. Tests pass a tempdir so they don't read
/// the live `/sys` tree (which would produce nondeterministic
/// `sched_ext` state depending on the host kernel config). The
/// public capture entry points pass this constant to read the
/// real sysfs tree at runtime.
pub const DEFAULT_SYS_ROOT: &str = "/sys";

pub(super) fn task_file(proc_root: &Path, tgid: i32, tid: i32, leaf: &str) -> PathBuf {
    proc_root
        .join(tgid.to_string())
        .join("task")
        .join(tid.to_string())
        .join(leaf)
}

pub(super) fn proc_file(proc_root: &Path, tgid: i32, leaf: &str) -> PathBuf {
    proc_root.join(tgid.to_string()).join(leaf)
}

/// Map a numeric scheduling policy (as it appears in
/// `/proc/<tgid>/task/<tid>/stat` field 41) to the canonical
/// kernel identifier string. Unknown integers render as
/// `"SCHED_UNKNOWN(<n>)"` rather than dropping the value so
/// diff output still surfaces a novel policy from a future
/// kernel.
pub(super) fn policy_name(policy: i32) -> String {
    match policy {
        libc::SCHED_OTHER => "SCHED_OTHER".to_string(),
        libc::SCHED_FIFO => "SCHED_FIFO".to_string(),
        libc::SCHED_RR => "SCHED_RR".to_string(),
        libc::SCHED_BATCH => "SCHED_BATCH".to_string(),
        libc::SCHED_IDLE => "SCHED_IDLE".to_string(),
        // `SCHED_DEADLINE` = 6, `SCHED_EXT` = 7 — neither is
        // exposed by the libc crate as of this writing; use the
        // kernel-canonical numeric codes.
        6 => "SCHED_DEADLINE".to_string(),
        7 => "SCHED_EXT".to_string(),
        other => format!("SCHED_UNKNOWN({other})"),
    }
}

/// Enumerate every numeric directory under the procfs root
/// (live tgids). Returns sorted ids so snapshot ordering is
/// deterministic. Empty vec on read failure.
pub(super) fn iter_tgids_at(proc_root: &Path) -> Vec<i32> {
    let Ok(entries) = fs::read_dir(proc_root) else {
        return Vec::new();
    };
    let mut tgids: Vec<i32> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()))
        .filter(|&p| p > 0)
        .collect();
    tgids.sort_unstable();
    tgids
}

/// Enumerate tids under `<proc_root>/<tgid>/task`. Empty vec on
/// read failure (process exited between enumeration and this
/// call).
pub(super) fn iter_task_ids_at(proc_root: &Path, tgid: i32) -> Vec<i32> {
    let path = proc_root.join(tgid.to_string()).join("task");
    let Ok(entries) = fs::read_dir(&path) else {
        return Vec::new();
    };
    let mut tids: Vec<i32> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().and_then(|s| s.parse::<i32>().ok()))
        .filter(|&t| t > 0)
        .collect();
    tids.sort_unstable();
    tids
}

/// Read `<proc_root>/<tgid>/comm` trimmed. `None` on read
/// failure or empty content.
pub(super) fn read_process_comm_at(proc_root: &Path, tgid: i32) -> Option<String> {
    let raw = fs::read_to_string(proc_file(proc_root, tgid, "comm")).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Read `<proc_root>/<tgid>/task/<tid>/comm` trimmed. `None`
/// on read failure or empty content.
pub(super) fn read_thread_comm_at(proc_root: &Path, tgid: i32, tid: i32) -> Option<String> {
    let raw = fs::read_to_string(task_file(proc_root, tgid, tid, "comm")).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Selected fields parsed out of `/proc/<tgid>/task/<tid>/stat`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct StatFields {
    pub(super) minflt: Option<u64>,
    pub(super) majflt: Option<u64>,
    pub(super) utime_clock_ticks: Option<u64>,
    pub(super) stime_clock_ticks: Option<u64>,
    /// Field 18: kernel-internal priority (signed, distinct
    /// from `nice`). `seq_put_decimal_ll(m, " ", priority)` at
    /// `fs/proc/array.c:602`; the value is the post-bias
    /// scheduler priority (`task_prio(task)`).
    pub(super) priority: Option<i32>,
    pub(super) nice: Option<i32>,
    pub(super) start_time_clock_ticks: Option<u64>,
    pub(super) processor: Option<i32>,
    /// Field 40: real-time priority. `seq_put_decimal_ull(m,
    /// " ", task->rt_priority)` at `fs/proc/array.c:637`.
    /// Stored as `u32` to match `unsigned int
    /// task_struct::rt_priority` from `include/linux/sched.h`;
    /// non-zero only when the task runs SCHED_FIFO / SCHED_RR.
    pub(super) rt_priority: Option<u32>,
    pub(super) policy: Option<i32>,
}

/// Pure parser for `/proc/<tgid>/task/<tid>/stat`. Per `proc(5)`,
/// field 2 (`comm`) is wrapped in parens and may contain
/// whitespace or `)`; every later field is indexed relative to
/// the LAST `)` in the line. Tail offsets (0-indexed from the
/// token past the final `)`):
///
/// | field | name                  | tail index |
/// |-------|-----------------------|------------|
/// | 10    | minflt                | 7          |
/// | 12    | majflt                | 9          |
/// | 14    | utime                 | 11         |
/// | 15    | stime                 | 12         |
/// | 18    | priority              | 15         |
/// | 19    | nice                  | 16         |
/// | 22    | starttime             | 19         |
/// | 39    | processor             | 36         |
/// | 40    | rt_priority           | 37         |
/// | 41    | policy                | 38         |
///
/// Field 42 (`delayacct_blkio_ticks`) is intentionally NOT
/// parsed — `blkio_delay_total_ns` from the taskstats genetlink
/// path supersedes it (ns precision vs USER_HZ ticks; both gated
/// by `CONFIG_TASK_DELAY_ACCT`, but the netlink path delivers
/// the same data without the procfs USER_HZ truncation).
///
/// Missing fields return `None` individually so a short line
/// (tid exited mid-read, stat truncated) degrades gracefully.
pub(super) fn parse_stat(raw: &str) -> StatFields {
    let Some(line) = raw.lines().next() else {
        return StatFields::default();
    };
    let Some(last_close) = line.rfind(')') else {
        return StatFields::default();
    };
    let Some(tail) = line.get(last_close + 1..) else {
        return StatFields::default();
    };
    let parts: Vec<&str> = tail.split_ascii_whitespace().collect();
    let get_u64 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<u64>().ok());
    let get_u32 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<u32>().ok());
    let get_i32 = |idx: usize| parts.get(idx).and_then(|s| s.parse::<i32>().ok());
    StatFields {
        minflt: get_u64(7),
        majflt: get_u64(9),
        utime_clock_ticks: get_u64(11),
        stime_clock_ticks: get_u64(12),
        priority: get_i32(15),
        nice: get_i32(16),
        start_time_clock_ticks: get_u64(19),
        processor: get_i32(36),
        rt_priority: get_u32(37),
        policy: get_i32(38),
    }
}

/// Read `<proc_root>/<tgid>/task/<tid>/stat` and parse fields.
/// Records a `"stat"` failure into `tally` on read error so the
/// per-snapshot [`CtprofParseSummary`] surfaces the dominant
/// procfs read-failure category. `tally: &mut None` skips the
/// recording (the synthetic-tree test pattern).
pub(super) fn read_stat_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> StatFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "stat")) {
        Ok(raw) => parse_stat(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("stat");
            }
            StatFields::default()
        }
    }
}

/// Parse the three leading u64 fields from a single-line
/// `/proc/<tgid>/task/<tid>/schedstat` — `(run_time_ns,
/// wait_time_ns, timeslices)`. Missing fields drop individually.
pub(super) fn parse_schedstat(raw: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    let Some(line) = raw.lines().next() else {
        return (None, None, None);
    };
    let mut parts = line.split_ascii_whitespace();
    let run = parts.next().and_then(|s| s.parse::<u64>().ok());
    let wait = parts.next().and_then(|s| s.parse::<u64>().ok());
    let slices = parts.next().and_then(|s| s.parse::<u64>().ok());
    (run, wait, slices)
}

/// Read `<proc_root>/<tgid>/task/<tid>/schedstat`. Three-tuple
/// of `Option<u64>` — kernel without `CONFIG_SCHEDSTATS` yields
/// all-`None`. Records a `"schedstat"` failure on read error
/// when a tally is supplied.
pub(super) fn read_schedstat_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> (Option<u64>, Option<u64>, Option<u64>) {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "schedstat")) {
        Ok(raw) => parse_schedstat(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("schedstat");
            }
            (None, None, None)
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct IoFields {
    pub(super) rchar: Option<u64>,
    pub(super) wchar: Option<u64>,
    pub(super) syscr: Option<u64>,
    pub(super) syscw: Option<u64>,
    pub(super) read_bytes: Option<u64>,
    pub(super) write_bytes: Option<u64>,
    pub(super) cancelled_write_bytes: Option<u64>,
}

/// Parse `/proc/<tgid>/task/<tid>/io` (line-oriented
/// `key: value` format).
pub(super) fn parse_io(raw: &str) -> IoFields {
    let mut out = IoFields::default();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let parsed = value.trim().parse::<u64>().ok();
        match key.trim() {
            "rchar" => out.rchar = parsed,
            "wchar" => out.wchar = parsed,
            "syscr" => out.syscr = parsed,
            "syscw" => out.syscw = parsed,
            "read_bytes" => out.read_bytes = parsed,
            "write_bytes" => out.write_bytes = parsed,
            "cancelled_write_bytes" => out.cancelled_write_bytes = parsed,
            _ => {}
        }
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/io` and parse fields.
/// Records an `"io"` failure into `tally` on read error (kernel
/// without `CONFIG_TASK_IO_ACCOUNTING` or per-tid race).
pub(super) fn read_io_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> IoFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "io")) {
        Ok(raw) => parse_io(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("io");
            }
            IoFields::default()
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct StatusFields {
    pub(super) voluntary_csw: Option<u64>,
    pub(super) nonvoluntary_csw: Option<u64>,
    /// First non-whitespace character of the `State:` line value.
    /// Real kernel chars are `R` / `S` / `D` / `T` / `t` / `X` /
    /// `Z` / `P` / `I` (see `fs/proc/array.c::task_state_array`).
    /// `None` when the line is absent or blank — the capture site
    /// collapses to `'~'` (via `default_state_char`) which sorts
    /// strictly after every real kernel char in lex order, so
    /// the [`crate::ctprof_compare::AggRule::ModeChar`]
    /// lex-smallest-wins tiebreak picks a real letter when one
    /// is present.
    pub(super) state: Option<char>,
    /// `Cpus_allowed_list:` as a parsed sorted vec. Kept separate
    /// from the `sched_getaffinity` reader because status-file
    /// reads attribute to the target task without a syscall
    /// round-trip — useful when the caller cannot hold a pid
    /// long enough for the syscall without a race.
    pub(super) cpus_allowed: Option<Vec<u32>>,
    /// `Threads:` value — `signal_struct->nr_threads` snapshot
    /// per `fs/proc/array.c:290`. Identical across every thread
    /// of the same tgid. The capture site dedups by populating
    /// [`ThreadState::nr_threads`] only on tid == tgid threads
    /// (see `capture_thread_at_with_tally`).
    pub(super) nr_threads: Option<u64>,
}

pub(super) fn parse_status(raw: &str) -> StatusFields {
    let mut out = StatusFields::default();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            // Kernel emits `State:\t<C> (<long>)` where <C> is the
            // single-letter code from `task_state_array`
            // (R/S/D/T/t/X/Z/P/I — nine codes, including the
            // off-by-default `P` parked state). First non-whitespace
            // char of the trimmed value is the letter;
            // `value.chars().next()` produces `None` only on a truly
            // empty line (which the split_once guards against
            // already).
            "State" => {
                out.state = value.chars().next();
            }
            "voluntary_ctxt_switches" => {
                out.voluntary_csw = value.parse::<u64>().ok();
            }
            "nonvoluntary_ctxt_switches" => {
                out.nonvoluntary_csw = value.parse::<u64>().ok();
            }
            "Cpus_allowed_list" => {
                out.cpus_allowed = crate::cpu_util::parse_cpu_list(value);
            }
            // `Threads:\t<num_threads>\n` per
            // `fs/proc/array.c:290`. Same value across every
            // thread of the same tgid; capture-side dedup picks
            // only the leader thread to avoid double-counting.
            "Threads" => {
                out.nr_threads = value.parse::<u64>().ok();
            }
            _ => {}
        }
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/status` and parse fields.
/// Records a `"status"` failure into `tally` on read error.
pub(super) fn read_status_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> StatusFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "status")) {
        Ok(raw) => parse_status(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("status");
            }
            StatusFields::default()
        }
    }
}

/// Read the cgroup v2 path from
/// `<proc_root>/<tgid>/task/<tid>/cgroup`. Format per
/// `cgroup(7)`: one line per hierarchy, shape
/// `hid:controllers:path`. The unified (v2) hierarchy is keyed
/// `0::<path>`; mixed-mode hosts expose legacy v1 hierarchies
/// alongside, which this reader skips. `None` on read failure
/// or when no v2 line is present. Test-only — production callers
/// pipe through [`read_cgroup_at_with_tally`] so per-tid failures
/// surface in `parse_summary`.
#[cfg(test)]
pub(super) fn read_cgroup_at(proc_root: &Path, tgid: i32, tid: i32) -> Option<String> {
    read_cgroup_at_with_tally(proc_root, tgid, tid, &mut None)
}

/// Records a `"cgroup"`
/// failure on read error (file absent — typical when the tid
/// exited mid-capture).
pub(super) fn read_cgroup_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> Option<String> {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "cgroup")) {
        Ok(raw) => parse_cgroup_v2(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("cgroup");
            }
            None
        }
    }
}

pub(super) fn parse_cgroup_v2(raw: &str) -> Option<String> {
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("0::") {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SchedFields {
    pub(super) nr_wakeups: Option<u64>,
    pub(super) nr_wakeups_local: Option<u64>,
    pub(super) nr_wakeups_remote: Option<u64>,
    pub(super) nr_wakeups_sync: Option<u64>,
    pub(super) nr_wakeups_migrate: Option<u64>,
    pub(super) nr_wakeups_affine: Option<u64>,
    pub(super) nr_wakeups_affine_attempts: Option<u64>,
    pub(super) nr_migrations: Option<u64>,
    pub(super) nr_forced_migrations: Option<u64>,
    pub(super) nr_failed_migrations_affine: Option<u64>,
    pub(super) nr_failed_migrations_running: Option<u64>,
    pub(super) nr_failed_migrations_hot: Option<u64>,
    pub(super) wait_sum: Option<u64>,
    pub(super) wait_count: Option<u64>,
    pub(super) wait_max: Option<u64>,
    pub(super) sleep_sum: Option<u64>,
    pub(super) sleep_max: Option<u64>,
    pub(super) block_sum: Option<u64>,
    pub(super) block_max: Option<u64>,
    pub(super) iowait_sum: Option<u64>,
    pub(super) iowait_count: Option<u64>,
    pub(super) exec_max: Option<u64>,
    pub(super) slice_max: Option<u64>,
    /// `core_forceidle_sum` from `/proc/<tid>/sched`, emitted via
    /// `PN_SCHEDSTAT(core_forceidle_sum)` at
    /// `kernel/sched/debug.c:1335`, build-gated on
    /// `CONFIG_SCHED_CORE`. Emission additionally lives inside
    /// the `if (schedstat_enabled())` block at
    /// `kernel/sched/debug.c:1285`, so on a host with schedstat
    /// off at runtime the line is absent and the parser arm
    /// never fires — leaving the field at `None`.
    /// Dotted ms.ns format like the other PN_SCHEDSTAT fields —
    /// reconstructed to full ns via [`parsed_ns_from_dotted`]. Counts
    /// time the task forced its SMT sibling idle for core-scheduling.
    /// `None` on kernels without `CONFIG_SCHED_CORE`, on hosts
    /// with schedstat disabled at runtime, or for tasks whose
    /// SMT cohort never accumulated forceidle.
    pub(super) core_forceidle_sum: Option<u64>,
    /// `se.slice` from `/proc/<tid>/sched`, emitted via
    /// `P(se.slice)` at `kernel/sched/debug.c:1364`. Plain
    /// `%lld` integer (NOT dotted ns; the `P` macro uses
    /// `%lld`, not `PN`'s `%lld.%06ld`). Per-thread
    /// `p->se.slice` in nanoseconds. For fair-class tasks
    /// (SCHED_NORMAL / SCHED_BATCH) it is the instantaneous
    /// slice CFS is currently running the task with; for
    /// SCHED_EXT tasks it reflects stale `p->se.slice` state
    /// because ext-class schedulers maintain slice in
    /// `p->scx.slice` and do not refresh `p->se.slice`. The
    /// kernel emits this line ONLY when `fair_policy(p->policy)`
    /// holds, which (per `kernel/sched/sched.h:194,203`) is
    /// true for SCHED_NORMAL, SCHED_BATCH, AND — under
    /// `CONFIG_SCHED_CLASS_EXT` — SCHED_EXT. `None` for
    /// SCHED_DEADLINE / SCHED_RR / SCHED_FIFO / SCHED_IDLE.
    pub(super) fair_slice_ns: Option<u64>,
    pub(super) ext_enabled: Option<bool>,
}

/// Outcome of [`parsed_ns_from_dotted`]. Distinguishes the two
/// failure modes the caller may want to treat separately:
/// [`Self::Negative`] (kernel emitted a value with a leading
/// `-`, observable on clock-skew / suspend-resume hosts) is
/// counted into [`CtprofParseSummary::negative_dotted_values`]
/// so an operator can see that the snapshot's schedstat values
/// are routinely negative-and-zeroed; [`Self::Malformed`]
/// (non-numeric, empty, overflow) is the every-other failure
/// mode and stays silent (the data source is ill-formed in a way
/// the operator can't act on).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParseDottedNs {
    /// Trimmed input started with `-` — the kernel's PN_SCHEDSTAT
    /// `%Ld.%06ld` format emitted a negative integer part. The
    /// `parse::<u64>()` rejection is by design (u64 cannot
    /// represent the sign) but the SIGNAL is meaningful: a
    /// negative schedstat field is rare and worth surfacing
    /// rather than silently zeroing.
    ///
    /// Note: `-0.000000` would also route here, but is
    /// unreachable from real kernel output — `SPLIT_NS(0)`
    /// yields `(0, 0)` which `%Ld.%06ld` formats as
    /// `0.000000` with no leading sign. The parser still
    /// classifies the unreachable shape as `Negative` rather
    /// than special-casing it; a fixture that synthesizes
    /// `-0.000000` directly will land in this variant.
    Negative,
    /// Otherwise unparseable: non-numeric integer or fractional
    /// part, empty input, or u64 overflow on the
    /// `ms * 1_000_000 + ns_remainder` reconstruction.
    Malformed,
}

/// Parse a `PN_SCHEDSTAT`-emitted dotted nanosecond value into
/// full ns. The kernel formats schedstat fractional fields via
/// `__PSN(S, F) SEQ_printf(m, "%-45s:%14Ld.%06ld\n", S, SPLIT_NS(F))`
/// where `SPLIT_NS(x) = (x / 1_000_000, x % 1_000_000)` — the
/// integer part is MILLISECONDS, the 6-digit fractional part is
/// the NANOSECOND remainder within a millisecond. Reconstructing
/// the original ns value is `ms * 1_000_000 + ns_remainder`.
///
/// Tolerates fractional widths other than 6 (some test fixtures
/// emit `5000.25` or `7.999`) by zero-padding the right side
/// before parsing — `.25` becomes `.250000` (=250_000 ns), `.999`
/// becomes `.999000` (=999_000 ns). Truncates fractional widths
/// >6 to the first 6 digits.
///
/// Returns [`Err(ParseDottedNs::Negative)`] when EITHER:
/// - the trimmed integer part starts with `-` (kernel emitted
///   `-5.000000` for a magnitude ≥ 1ms negative SPLIT_NS via
///   `%Ld`), OR
/// - the trimmed fractional part starts with `-` (kernel
///   emitted `0.-000500` for a sub-millisecond negative
///   SPLIT_NS — `%Ld` on the `(x / 1_000_000)` integer part
///   yields `0` with no sign for x in `(-1_000_000, 0)`, and
///   `%06ld` on the `(x % 1_000_000)` remainder yields the
///   `-`). The sub-millisecond shape is the COMMON case for
///   clock-skew bugs because most schedstat deltas land
///   sub-millisecond — missing it would defeat the
///   negative-detection contract on the bulk of real
///   negatives.
///
/// The caller records the bump in the per-snapshot
/// [`CtprofParseSummary::negative_dotted_values`] before
/// folding to zero. Returns [`Err(ParseDottedNs::Malformed)`]
/// for any other parse failure (non-numeric, empty, overflow);
/// the caller folds to zero silently per the best-effort capture
/// contract.
///
/// The bare-integer (no dot) branch parses the value as raw ns
/// — used for test fixtures and graceful degradation; the
/// kernel's PN_SCHEDSTAT format always emits the dotted form.
/// Same negative-vs-malformed split applies to the bare-integer
/// branch so a stray bare-integer negative is also tallied.
pub(super) fn parsed_ns_from_dotted(value: &str) -> Result<u64, ParseDottedNs> {
    if let Some((ms_str, ns_str)) = value.split_once('.') {
        let ms_trimmed = ms_str.trim();
        if ms_trimmed.starts_with('-') {
            return Err(ParseDottedNs::Negative);
        }
        // Sub-millisecond negative: kernel `%06ld` on a negative
        // remainder yields a leading `-` on the fractional side
        // even when the integer side is `0`. `0.-000500` is the
        // canonical shape for SPLIT_NS of a small (>-1ms)
        // negative — the integer-only check above misses it,
        // and it is the COMMON case for clock-skew bugs since
        // most schedstat deltas land sub-millisecond. Check the
        // fractional side BEFORE the chars().take(6) truncation
        // would otherwise swallow a sign-only fractional like
        // `-`.
        if ns_str.trim_start().starts_with('-') {
            return Err(ParseDottedNs::Negative);
        }
        let ms = ms_trimmed
            .parse::<u64>()
            .map_err(|_| ParseDottedNs::Malformed)?;
        let ns_part: String = ns_str.chars().take(6).collect();
        let padded = format!("{:0<6}", ns_part);
        let ns = padded
            .parse::<u64>()
            .map_err(|_| ParseDottedNs::Malformed)?;
        ms.checked_mul(1_000_000)
            .and_then(|x| x.checked_add(ns))
            .ok_or(ParseDottedNs::Malformed)
    } else {
        let trimmed = value.trim();
        if trimmed.starts_with('-') {
            return Err(ParseDottedNs::Negative);
        }
        trimmed.parse::<u64>().map_err(|_| ParseDottedNs::Malformed)
    }
}

/// Parse `/proc/<tgid>/task/<tid>/sched`. Requires
/// `CONFIG_SCHED_DEBUG`. Format is many lines of `key : value`
/// where the key is dot-delimited (`se.statistics.nr_wakeups`);
/// different kernel versions use `se.statistics.`, `stats.`,
/// or bare names. The reader matches on the LAST dot-delimited
/// segment to absorb that variation.
///
/// PN_SCHEDSTAT fields (`wait_sum`, `sum_sleep_runtime`,
/// `sum_block_runtime`, `iowait_sum`) emit a `ms.ns_remainder`
/// dotted format — reconstructed to full ns via
/// [`parsed_ns_from_dotted`]. P_SCHEDSTAT fields
/// (`wait_count`, `iowait_count`, `nr_wakeups*`,
/// `nr_migrations`) emit plain integers — parsed as `u64`.
///
/// `tally`, when supplied, records each negative dotted-ns parse
/// outcome via [`ParseTally::record_negative_dotted`] so the
/// per-snapshot summary surfaces the rate at which schedstat
/// fields were silently zeroed. `&mut None` skips the recording —
/// the synthetic-tree test path that doesn't carry a tally.
pub(super) fn parse_sched(raw: &str, tally: &mut Option<&mut ParseTally>) -> SchedFields {
    let mut out = SchedFields::default();
    let mut parse_dotted = |value: &str| -> Option<u64> {
        match parsed_ns_from_dotted(value) {
            Ok(v) => Some(v),
            Err(ParseDottedNs::Negative) => {
                if let Some(t) = tally.as_mut() {
                    t.record_negative_dotted();
                }
                None
            }
            Err(ParseDottedNs::Malformed) => None,
        }
    };
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        let parsed_u64 = || value.parse::<u64>().ok();
        // `ext.enabled` is the only key the kernel emits with a
        // literal dot in the variable name (every other dot is a
        // namespace prefix like `se.statistics.`). Match on the full
        // key BEFORE the rsplit-on-dot fallback so a future kernel
        // line ending in `.enabled` cannot collide.
        if key == "ext.enabled" {
            out.ext_enabled = value.parse::<u64>().ok().map(|n| n != 0);
            continue;
        }
        let short = key.rsplit('.').next().unwrap_or(key);
        match short {
            "nr_wakeups" => out.nr_wakeups = parsed_u64(),
            "nr_wakeups_local" => out.nr_wakeups_local = parsed_u64(),
            "nr_wakeups_remote" => out.nr_wakeups_remote = parsed_u64(),
            "nr_wakeups_sync" => out.nr_wakeups_sync = parsed_u64(),
            "nr_wakeups_migrate" => out.nr_wakeups_migrate = parsed_u64(),
            "nr_wakeups_affine" => out.nr_wakeups_affine = parsed_u64(),
            "nr_wakeups_affine_attempts" => out.nr_wakeups_affine_attempts = parsed_u64(),
            "nr_migrations" => out.nr_migrations = parsed_u64(),
            "nr_forced_migrations" => out.nr_forced_migrations = parsed_u64(),
            "nr_failed_migrations_affine" => out.nr_failed_migrations_affine = parsed_u64(),
            "nr_failed_migrations_running" => out.nr_failed_migrations_running = parsed_u64(),
            "nr_failed_migrations_hot" => out.nr_failed_migrations_hot = parsed_u64(),
            "wait_sum" => out.wait_sum = parse_dotted(value),
            "wait_count" => out.wait_count = parsed_u64(),
            "wait_max" => out.wait_max = parse_dotted(value),
            // Kernel emits `sum_sleep_runtime` (see
            // `kernel/sched/debug.c` -> `proc_sched_show_task`).
            // The raw value lands in `SchedFields::sleep_sum`; the
            // capture site at `capture_thread_at_with_tally`
            // subtracts `sum_block_runtime` to derive
            // `ThreadState::voluntary_sleep_ns` — the kernel
            // double-counts block under sum_sleep_runtime, so the
            // raw value is not surfaced in ThreadState. The kernel
            // does not emit a `sleep_count` counterpart;
            // `nr_wakeups` (matched above) covers the wake-side
            // event tally.
            "sum_sleep_runtime" => out.sleep_sum = parse_dotted(value),
            "sleep_max" => out.sleep_max = parse_dotted(value),
            // Kernel emits `sum_block_runtime`; the matching
            // ThreadState field is `block_sum` for symmetry with
            // the other `*_sum` fields. There is no `block_count`
            // counterpart from the kernel — the schedstat printout
            // pairs `wait_sum/wait_count` and `iowait_sum/iowait_count`
            // but `sum_block_runtime` has no per-event counter.
            "sum_block_runtime" => out.block_sum = parse_dotted(value),
            "block_max" => out.block_max = parse_dotted(value),
            "iowait_sum" => out.iowait_sum = parse_dotted(value),
            "iowait_count" => out.iowait_count = parsed_u64(),
            "exec_max" => out.exec_max = parse_dotted(value),
            "slice_max" => out.slice_max = parse_dotted(value),
            // PN_SCHEDSTAT dotted ns; CONFIG_SCHED_CORE-gated. Same
            // ms.ns reconstruction as wait_sum / block_sum.
            "core_forceidle_sum" => out.core_forceidle_sum = parse_dotted(value),
            // P plain integer in ns. The kernel emits this only
            // for fair-policy tasks (`fair_policy(p->policy)` at
            // debug.c:1363); for other policies the line is absent
            // and `parsed_u64()` collapses to None.
            "slice" => out.fair_slice_ns = parsed_u64(),
            _ => {}
        }
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/sched` and parse fields.
/// Records a `"sched"` failure into `tally` on read error, plus
/// per-line negative-dotted-value bumps via `parse_sched`.
pub(super) fn read_sched_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> SchedFields {
    match fs::read_to_string(task_file(proc_root, tgid, tid, "sched")) {
        Ok(raw) => parse_sched(&raw, tally),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("sched");
            }
            SchedFields::default()
        }
    }
}

/// Parse `/proc/<pid>/smaps_rollup` contents into a key→u64-kB
/// map. Format per `__show_smap()` at
/// `fs/proc/task_mmu.c:1330-1368`: each kv line is
/// `<Name>:<whitespace><u64><whitespace>kB`. The kernel ALSO
/// emits a `<addr_start>-<addr_end> ---p <off> XX:XX <inode> [rollup]`
/// header line (built by `show_vma_header_prefix()` then
/// `seq_puts(m, "[rollup]\n")` at `task_mmu.c:1500-1503`). That
/// header CONTAINS a `:` (the device-major:minor pair `XX:XX`),
/// so a naive `split_once(':')` would mis-extract a junk key
/// (the whitespace-laden address range + flags + offset prefix)
/// with value 0 (the minor-device integer parses as the first
/// whitespace token of the value side). Real smaps_rollup keys
/// are single-word identifiers (Rss, Pss, Pss_Dirty, etc.) that
/// never contain whitespace or `-`; the address-range header
/// always contains both. Reject any line whose pre-`:` segment
/// carries either character.
///
/// Lines whose value field doesn't parse as u64 are silently
/// dropped (best-effort, matching the absent-counter contract).
pub(super) fn parse_smaps_rollup(raw: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key_trimmed = key.trim();
        // Header-line guard: real smaps_rollup keys never
        // contain whitespace or `-`. Address-range headers
        // (`<addr_start>-<addr_end> ---p <off> XX:XX <inode>
        // [rollup]`) carry both.
        if key_trimmed.contains(char::is_whitespace) || key_trimmed.contains('-') {
            continue;
        }
        // Value field: whitespace-prefixed integer + " kB" suffix
        // (or rarely no suffix on a future kernel addition). The
        // first whitespace-token after trimming IS the integer;
        // dropping the unit suffix happens for free via
        // `split_ascii_whitespace().next()`.
        let Some(n_str) = value.split_ascii_whitespace().next() else {
            continue;
        };
        let Ok(n) = n_str.parse::<u64>() else {
            continue;
        };
        out.insert(key_trimmed.to_string(), n);
    }
    out
}

/// Read `<proc_root>/<tgid>/task/<tid>/smaps_rollup` for the
/// thread leader (tid == tgid) and parse it. Non-leader threads
/// short-circuit to an empty map: the underlying mm_struct is
/// shared per-tgid, so reading from any thread yields identical
/// values, and capturing once per tgid avoids redundant
/// per-thread work. Records a `"smaps_rollup"` failure into
/// `tally` on read error.
///
/// Permission semantics: `/proc/<pid>/smaps_rollup` requires
/// `CAP_SYS_PTRACE` for processes the caller doesn't own (PID 1
/// being the canonical example). Read failure is treated as
/// best-effort — empty map, tally bump, no panic. Older kernels
/// (pre-4.14) lack the file entirely; same handling.
pub(super) fn read_smaps_rollup_at_with_tally(
    proc_root: &Path,
    tgid: i32,
    tid: i32,
    tally: &mut Option<&mut ParseTally>,
) -> BTreeMap<String, u64> {
    if tid != tgid {
        // Leader-dedup: non-leader threads share the same
        // mm_struct, so the file would yield identical values.
        // Skip the read entirely.
        return BTreeMap::new();
    }
    match fs::read_to_string(task_file(proc_root, tgid, tid, "smaps_rollup")) {
        Ok(raw) => parse_smaps_rollup(&raw),
        Err(_) => {
            if let Some(t) = tally.as_mut() {
                t.record_failure("smaps_rollup");
            }
            BTreeMap::new()
        }
    }
}

/// Parse cgroup v2 `cpu.stat`. Format is lines of `key value`
/// (space-separated, not `key: value`).
pub(super) fn parse_cpu_stat(raw: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    let mut usage = None;
    let mut throttled = None;
    let mut throttled_usec = None;
    for line in raw.lines() {
        let mut parts = line.split_ascii_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next() else { continue };
        let parsed = value.parse::<u64>().ok();
        match key {
            "usage_usec" => usage = parsed,
            "nr_throttled" => throttled = parsed,
            "throttled_usec" => throttled_usec = parsed,
            _ => {}
        }
    }
    (usage, throttled, throttled_usec)
}

/// Parse a cgroup v2 key-value file (one `<key> <u64>` per
/// line). Used for `memory.stat` and `memory.events`. Lines
/// the parser cannot fully decompose into a key + u64 are
/// silently skipped — a future kernel that introduces non-u64
/// values won't break the parser, just elide the offending key.
pub(super) fn parse_kv_counters(raw: &str) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let mut parts = line.split_ascii_whitespace();
        let Some(key) = parts.next() else { continue };
        let Some(value) = parts.next() else { continue };
        let Ok(parsed) = value.parse::<u64>() else {
            continue;
        };
        out.insert(key.to_string(), parsed);
    }
    out
}

/// Parse a single-line LIMIT cgroup file (e.g. `memory.max`,
/// `memory.high`, `pids.max`). The literal token `max` means
/// "no limit" and yields `None`; a numeric value yields
/// `Some(u64)`. Whitespace-only or malformed input also yields
/// `None` (best-effort, matching the absent-counter contract).
///
/// Caller MUST NOT use this for FLOOR files (`memory.low`,
/// `memory.min`) — for floors, the literal token `max` means
/// "maximum protection", not "no floor", which is the semantic
/// opposite. Use [`parse_floor_value`] there instead.
pub(super) fn parse_max_or_u64(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed == "max" {
        return None;
    }
    trimmed.parse::<u64>().ok()
}

/// Parse a single-line FLOOR cgroup file (`memory.low`,
/// `memory.min`). The literal token `max` means
/// "maximum protection" — yields `Some(u64::MAX)` rather than
/// `None`, because FLOORS use `None` to mean "absent file"
/// only. A numeric value yields `Some(u64)`; whitespace-only or
/// malformed input yields `None` (absent-counter contract).
///
/// The semantic asymmetry vs. [`parse_max_or_u64`] is critical:
/// for limits, "max" is the absence of a cap (collapse to
/// `None`); for floors, "max" is a fully-protected floor (it
/// must NOT collapse to "no floor"). [`merge_min_option`] then
/// correctly picks `min(u64::MAX, 5G) = 5G` instead of None
/// when one contributor has full protection and another has a
/// concrete protection.
pub(super) fn parse_floor_value(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    if trimmed == "max" {
        return Some(u64::MAX);
    }
    trimmed.parse::<u64>().ok()
}

/// Parse `cpu.max` (one line, two whitespace-separated tokens:
/// `<quota|max> <period>`). Returns `(quota, period)` where
/// `quota` is `None` for the literal `max` token (no CFS
/// bandwidth cap) and `Some(usec)` otherwise; `period` defaults
/// to the kernel default of 100_000 µs when missing or
/// malformed.
pub(super) fn parse_cpu_max(raw: &str) -> (Option<u64>, u64) {
    let mut parts = raw.split_ascii_whitespace();
    let quota_token = parts.next();
    let period_token = parts.next();
    let quota = quota_token.and_then(parse_max_or_u64_str);
    let period = period_token
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(CPU_MAX_DEFAULT_PERIOD_US);
    (quota, period)
}

/// Helper for [`parse_cpu_max`]: route a single token through
/// the same `max`-vs-u64 disambiguation as [`parse_max_or_u64`]
/// without committing to a string-trimmed input shape.
pub(super) fn parse_max_or_u64_str(s: &str) -> Option<u64> {
    if s == "max" {
        return None;
    }
    s.parse::<u64>().ok()
}

/// Default CFS bandwidth period when `cpu.max` is absent or its
/// period token is unreadable. Matches the kernel default
/// returned by `default_bw_period_us()` at
/// `kernel/sched/sched.h:441`; child cgroups inherit this when
/// `cpu.max` is unset.
pub(super) const CPU_MAX_DEFAULT_PERIOD_US: u64 = 100_000;

/// Populate a [`CgroupStats`] by reading the cgroup v2 files
/// for `path` under `cgroup_root`. Missing files collapse to
/// the struct's `Default` (zero / `None` per field semantics) —
/// the root cgroup is missing most knob files, and child
/// cgroups on hosts without `pids` enabled in
/// `cgroup.subtree_control` are also expected to lack
/// `pids.{current,max}`.
pub(super) fn read_cgroup_stats_at(cgroup_root: &Path, path: &str) -> CgroupStats {
    let relative = path.strip_prefix('/').unwrap_or(path);
    let dir = if relative.is_empty() {
        cgroup_root.to_path_buf()
    } else {
        cgroup_root.join(relative)
    };

    let (usage, throttled, throttled_usec) = fs::read_to_string(dir.join("cpu.stat"))
        .ok()
        .as_deref()
        .map(parse_cpu_stat)
        .unwrap_or((None, None, None));
    let (max_quota_us, max_period_us) = fs::read_to_string(dir.join("cpu.max"))
        .ok()
        .as_deref()
        .map(parse_cpu_max)
        .unwrap_or((None, CPU_MAX_DEFAULT_PERIOD_US));
    let weight = fs::read_to_string(dir.join("cpu.weight"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let weight_nice = fs::read_to_string(dir.join("cpu.weight.nice"))
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok());

    let memory_current = fs::read_to_string(dir.join("memory.current"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let memory_max = fs::read_to_string(dir.join("memory.max"))
        .ok()
        .as_deref()
        .and_then(parse_max_or_u64);
    let memory_high = fs::read_to_string(dir.join("memory.high"))
        .ok()
        .as_deref()
        .and_then(parse_max_or_u64);
    let memory_low = fs::read_to_string(dir.join("memory.low"))
        .ok()
        .as_deref()
        .and_then(parse_floor_value);
    let memory_min = fs::read_to_string(dir.join("memory.min"))
        .ok()
        .as_deref()
        .and_then(parse_floor_value);
    let memory_stat = fs::read_to_string(dir.join("memory.stat"))
        .ok()
        .as_deref()
        .map(parse_kv_counters)
        .unwrap_or_default();
    let memory_events = fs::read_to_string(dir.join("memory.events"))
        .ok()
        .as_deref()
        .map(parse_kv_counters)
        .unwrap_or_default();

    let pids_current = fs::read_to_string(dir.join("pids.current"))
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let pids_max = fs::read_to_string(dir.join("pids.max"))
        .ok()
        .as_deref()
        .and_then(parse_max_or_u64);

    let psi = read_cgroup_psi_at(cgroup_root, path);

    CgroupStats {
        cpu: CgroupCpuStats {
            usage_usec: usage.unwrap_or(0),
            nr_throttled: throttled.unwrap_or(0),
            throttled_usec: throttled_usec.unwrap_or(0),
            max_quota_us,
            max_period_us,
            weight,
            weight_nice,
        },
        memory: CgroupMemoryStats {
            current: memory_current,
            max: memory_max,
            high: memory_high,
            low: memory_low,
            min: memory_min,
            stat: memory_stat,
            events: memory_events,
        },
        pids: CgroupPidsStats {
            current: pids_current,
            max: pids_max,
        },
        psi,
    }
}
