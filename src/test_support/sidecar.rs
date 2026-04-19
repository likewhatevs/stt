//! Per-run sidecar JSON — the durable record of a ktstr test outcome.
//!
//! Every test (pass, fail, or skip) writes a [`SidecarResult`] to a
//! JSON file under the run's sidecar directory; downstream analysis
//! (`cargo ktstr stats`, CI dashboards) aggregates those files to
//! compute pass/fail rates, verifier stats, callback profiles, and
//! KVM stats across gauntlet variants.
//!
//! Responsibilities owned by this module:
//! - [`SidecarResult`]: the on-disk schema. Fields are `serde`-tagged
//!   with `skip_serializing_if` / `default` so optional VM telemetry
//!   (monitor, KVM stats, BPF verifier stats) doesn't bloat sidecars
//!   that didn't run a VM.
//! - [`collect_sidecars`]: load every `*.ktstr.json` under a directory
//!   (one level of subdirectories for per-job gauntlet layouts).
//! - [`write_sidecar`] / [`write_skip_sidecar`]: serialize one run to
//!   disk; variant-hash the discriminating fields so gauntlet variants
//!   don't clobber each other.
//! - [`sidecar_dir`], [`runs_root`], [`newest_run_dir`]: resolve where
//!   sidecars live (env override, or `{target}/ktstr/{kernel}-{git}`).
//! - [`format_verifier_stats`], [`format_callback_profile`],
//!   [`format_kvm_stats`]: human-readable summaries from a
//!   `Vec<SidecarResult>` for CLI output.
//! - [`detect_kernel_version`]: read the kernel version from
//!   `KTSTR_KERNEL` cache metadata for sidecar-dir naming and the
//!   `kernel_version` field.

use std::path::PathBuf;

use anyhow::Context;

use crate::assert::{AssertResult, ScenarioStats};
use crate::monitor::MonitorSummary;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::entry::KtstrTestEntry;
use super::timefmt::{generate_run_id, now_iso8601};

/// Test result sidecar written to KTSTR_SIDECAR_DIR for post-run analysis.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SidecarResult {
    /// Fully qualified test name (matches `KtstrTestEntry::name`,
    /// the bare function name without the `ktstr/` nextest prefix).
    pub test_name: String,
    /// Rendered topology label (e.g. `1n2l4c1t`) for the variant this
    /// sidecar describes.
    pub topology: String,
    /// Scheduler name (matches `Scheduler::name`); `"eevdf"` for
    /// tests run without an scx scheduler.
    pub scheduler: String,
    /// Overall pass/fail verdict for this run.
    pub passed: bool,
    /// True when the test was skipped (e.g. topology mismatch,
    /// missing resource). A skipped test has `passed == true`
    /// (to keep the verdict gate simple) but downstream stats
    /// tooling must subtract `skipped` runs from "pass count" to
    /// avoid reporting non-executions as passes.
    pub skipped: bool,
    /// Aggregate per-cgroup statistics merged across every worker.
    pub stats: ScenarioStats,
    /// Monitor summary. `None` means the monitor loop did not run
    /// (host-only tests, early VM failure) or sample collection
    /// produced no valid data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor: Option<MonitorSummary>,
    /// Ordered stimulus events published by the guest step executor
    /// while the scenario ran.
    pub stimulus_events: Vec<StimulusEvent>,
    /// Work type label used for post-hoc filtering and A/B comparison
    /// (distinct from the `WorkType` enum — this is the text name).
    pub work_type: String,
    /// Scheduler flag names active for this gauntlet variant. Empty
    /// for the default (no-flags) profile. Participates in the
    /// sidecar variant-hash so flag-only variants don't clobber.
    pub active_flags: Vec<String>,
    /// Per-BPF-program verifier statistics captured from the VM's
    /// scheduler (when one was loaded). Absent when no scheduler
    /// programs were inspected; empty vec distinguishes "inspected,
    /// none matched" from absence.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub verifier_stats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    /// Aggregate per-vCPU KVM stats read after VM exit. `None` when
    /// the VM did not run (host-only tests) or KVM stats were
    /// unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kvm_stats: Option<crate::vmm::KvmStatsTotals>,
    /// Effective sysctls active during this test run, recorded as raw
    /// `sysctl.key=value` cmdline strings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sysctls: Vec<String>,
    /// Effective kernel command-line args active during this test run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub kargs: Vec<String>,
    /// Kernel version string (e.g. `"6.14.2"`). Populated from the VM's
    /// `/proc/version` or the cache entry metadata; `None` for host-only
    /// tests or when the version could not be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_version: Option<String>,
    /// ISO 8601 timestamp of when this test run started.
    pub timestamp: String,
    /// Unique identifier for the test run. Derived from the repo commit
    /// hash and a monotonic counter to distinguish runs within the same
    /// build.
    pub run_id: String,
}

/// Scan a directory for ktstr sidecar JSON files. Recurses one level
/// into subdirectories to handle per-job gauntlet layouts.
pub(crate) fn collect_sidecars(dir: &std::path::Path) -> Vec<SidecarResult> {
    let mut sidecars = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return sidecars,
    };
    let mut subdirs = Vec::new();
    let try_load = |path: &std::path::Path, out: &mut Vec<SidecarResult>| {
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            return;
        }
        if !path.to_str().is_some_and(|s| s.contains(".ktstr.")) {
            return;
        }
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => return,
        };
        match serde_json::from_str::<SidecarResult>(&data) {
            Ok(sc) => out.push(sc),
            Err(e) => eprintln!("ktstr: skipping {}: {e}", path.display()),
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            subdirs.push(path);
            continue;
        }
        try_load(&path, &mut sidecars);
    }
    for sub in subdirs {
        if let Ok(entries) = std::fs::read_dir(&sub) {
            for entry in entries.flatten() {
                try_load(&entry.path(), &mut sidecars);
            }
        }
    }
    sidecars
}

/// BPF verifier complexity limit (BPF_COMPLEXITY_LIMIT_INSNS).
const VERIFIER_INSN_LIMIT: u32 = 1_000_000;

/// Percentage of the verifier limit that triggers a warning.
const VERIFIER_WARN_PCT: f64 = 75.0;

/// Aggregate BPF verifier stats across sidecars into a summary table.
///
/// verified_insns is deterministic for a given binary, so per-program
/// values are deduplicated (max across observations). Flags programs
/// using >=75% of the 1M verifier complexity limit.
pub(crate) fn format_verifier_stats(sidecars: &[SidecarResult]) -> String {
    use std::collections::BTreeMap;

    let mut by_name: BTreeMap<&str, u32> = BTreeMap::new();
    for sc in sidecars {
        for info in &sc.verifier_stats {
            let entry = by_name.entry(&info.name).or_insert(0);
            *entry = (*entry).max(info.verified_insns);
        }
    }

    if by_name.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n=== BPF VERIFIER STATS ===\n\n");
    out.push_str(&format!(
        "  {:<24} {:>12} {:>8}\n",
        "program", "verified", "limit%"
    ));
    out.push_str(&format!("  {:-<24} {:-<12} {:-<8}\n", "", "", ""));

    let mut warnings = Vec::new();
    let mut total: u64 = 0;

    for (&name, &verified_insns) in &by_name {
        let pct = (verified_insns as f64 / VERIFIER_INSN_LIMIT as f64) * 100.0;
        let flag = if pct >= VERIFIER_WARN_PCT { " !" } else { "" };
        out.push_str(&format!(
            "  {:<24} {:>12} {:>7.1}%{flag}\n",
            name, verified_insns, pct,
        ));
        if pct >= VERIFIER_WARN_PCT {
            warnings.push(format!(
                "  {name}: {pct:.1}% of 1M limit ({verified_insns} verified insns)",
            ));
        }
        total += verified_insns as u64;
    }

    out.push_str(&format!("\n  total verified insns: {total}\n"));

    if !warnings.is_empty() {
        out.push_str("\nWARNING: programs near verifier complexity limit:\n");
        for w in &warnings {
            out.push_str(w);
            out.push('\n');
        }
    }

    out
}

/// Per-test BPF callback profile from monitor prog_stats_deltas.
///
/// Shows per-program invocation count, total CPU time, and average
/// nanoseconds per call. Each test's profile is printed independently.
pub(crate) fn format_callback_profile(sidecars: &[SidecarResult]) -> String {
    let mut out = String::new();

    for sc in sidecars {
        let deltas = match sc
            .monitor
            .as_ref()
            .and_then(|m| m.prog_stats_deltas.as_ref())
        {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        if out.is_empty() {
            out.push_str("\n=== BPF CALLBACK PROFILE ===\n");
        }
        out.push_str(&format!("\n  {} ({}):\n", sc.test_name, sc.topology));
        out.push_str(&format!(
            "    {:<24} {:>12} {:>14} {:>12}\n",
            "program", "cnt", "total_ns", "avg_ns"
        ));
        out.push_str(&format!(
            "    {:-<24} {:-<12} {:-<14} {:-<12}\n",
            "", "", "", ""
        ));
        for d in deltas {
            out.push_str(&format!(
                "    {:<24} {:>12} {:>14} {:>12.0}\n",
                d.name, d.cnt, d.nsecs, d.nsecs_per_call,
            ));
        }
    }

    out
}

/// Aggregate KVM stats across sidecars into a compact summary.
///
/// Averages each stat across all tests that returned `Some(KvmStatsTotals)`.
/// Tests without KVM stats (non-VM tests, old kernels) are excluded
/// from the denominator.
pub(crate) fn format_kvm_stats(sidecars: &[SidecarResult]) -> String {
    let with_stats: Vec<&crate::vmm::KvmStatsTotals> = sidecars
        .iter()
        .filter_map(|sc| sc.kvm_stats.as_ref())
        .collect();

    if with_stats.is_empty() {
        return String::new();
    }

    let n_vms = with_stats.len();

    // Compute cross-VM averages for each stat.
    let vm_avg = |name: &str| -> u64 {
        let sum: u64 = with_stats.iter().map(|d| d.avg(name)).sum();
        sum / n_vms as u64
    };

    let exits = vm_avg("exits");
    let halt = vm_avg("halt_exits");
    let halt_wait_ns = vm_avg("halt_wait_ns");
    let preempted = vm_avg("preemption_reported");
    let signal = vm_avg("signal_exits");
    let hypercalls = vm_avg("hypercalls");

    // Halt poll efficiency across all vCPUs and VMs.
    let total_poll_ok: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_successful_poll"))
        .sum();
    let total_poll_try: u64 = with_stats
        .iter()
        .map(|d| d.sum("halt_attempted_poll"))
        .sum();

    if exits == 0 {
        return String::new();
    }

    let halt_wait_ms = halt_wait_ns as f64 / 1_000_000.0;
    let poll_pct = if total_poll_try > 0 {
        (total_poll_ok as f64 / total_poll_try as f64) * 100.0
    } else {
        0.0
    };

    let mut out = format!("\n=== KVM STATS (avg across {n_vms} VMs) ===\n\n");
    out.push_str(&format!(
        "  exits/vcpu  {:>7}   halt/vcpu     {:>5}   halt_wait_ms {:>7.1}\n",
        exits, halt, halt_wait_ms,
    ));
    out.push_str(&format!(
        "  poll_ok%    {:>6.1}%   preempted/vcpu {:>4}   signal/vcpu  {:>7}\n",
        poll_pct, preempted, signal,
    ));
    if hypercalls > 0 {
        out.push_str(&format!("  hypercalls/vcpu {:>4}\n", hypercalls));
    }

    // Trust warnings.
    if preempted > 0 {
        let total: u64 = with_stats
            .iter()
            .map(|d| d.sum("preemption_reported"))
            .sum();
        out.push_str(&format!(
            "\n  WARNING: {total} host preemptions detected \
             -- timing results may be unreliable\n",
        ));
    }

    out
}

/// Resolve the sidecar output directory for the current test process.
///
/// Override: `KTSTR_SIDECAR_DIR` (used as-is when non-empty).
/// Default: `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{git_short}/`,
/// where `{kernel}` is the version detected from `KTSTR_KERNEL`'s
/// metadata (or `"unknown"` when no kernel is set / detection fails)
/// and `{git_short}` is the short commit hash baked in by `build.rs`.
pub(crate) fn sidecar_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KTSTR_SIDECAR_DIR")
        && !d.is_empty()
    {
        return PathBuf::from(d);
    }
    let kernel = detect_kernel_version().unwrap_or_else(|| "unknown".to_string());
    runs_root().join(format!("{kernel}-{}", crate::GIT_HASH))
}

/// Resolve the parent directory that holds all test-run subdirectories.
///
/// `{CARGO_TARGET_DIR or "target"}/ktstr/`. Used by `cargo ktstr stats`
/// to enumerate runs without needing to reconstruct a specific run key.
pub fn runs_root() -> PathBuf {
    let target = std::env::var("CARGO_TARGET_DIR")
        .ok()
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    target.join("ktstr")
}

/// Find the most recently modified run directory under [`runs_root`].
///
/// Used by bare `cargo ktstr stats` (no subcommand) when
/// `KTSTR_SIDECAR_DIR` isn't set: the stats command doesn't itself
/// run a kernel, so it can't reconstruct the `{kernel}-{git_short}` key
/// that the test process used. Picking the newest subdirectory by
/// mtime mirrors "show me the report from my last test run."
pub fn newest_run_dir() -> Option<PathBuf> {
    let root = runs_root();
    let entries = std::fs::read_dir(&root).ok()?;
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| e.path())
}

/// Detect kernel version from KTSTR_KERNEL env var or cache metadata.
pub(crate) fn detect_kernel_version() -> Option<String> {
    let kernel_dir = std::env::var("KTSTR_KERNEL").ok()?;
    let p = std::path::Path::new(&kernel_dir);
    let meta_path = p.join("metadata.json");
    if let Ok(data) = std::fs::read_to_string(&meta_path)
        && let Ok(meta) = serde_json::from_str::<crate::cache::KernelMetadata>(&data)
    {
        return meta.version;
    }
    let ver_path = p.join("include/config/kernel.release");
    if let Ok(v) = std::fs::read_to_string(ver_path) {
        let v = v.trim();
        if !v.is_empty() {
            return Some(v.to_string());
        }
    }
    None
}

/// Compute a stable 64-bit discriminator over the fields that
/// distinguish gauntlet variants of the same test. Used to suffix
/// the sidecar filename so concurrent variants do not clobber each
/// other's output.
///
/// Uses [`siphasher::sip::SipHasher13`] with zero keys for the same
/// stability reason as the initramfs cache keys — the discriminator
/// must be the same across Rust toolchain versions or downstream
/// tooling that groups variants by filename breaks.
fn sidecar_variant_hash(sidecar: &SidecarResult) -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(sidecar.topology.as_bytes());
    h.write(&[0]);
    h.write(sidecar.scheduler.as_bytes());
    h.write(&[0]);
    h.write(sidecar.work_type.as_bytes());
    h.write(&[0]);
    h.write(&[0xfe]);
    for f in &sidecar.active_flags {
        h.write(f.as_bytes());
        h.write(&[0]);
    }
    h.write(&[0xfd]);
    for s in &sidecar.sysctls {
        h.write(s.as_bytes());
        h.write(&[0]);
    }
    h.write(&[0xff]);
    for k in &sidecar.kargs {
        h.write(k.as_bytes());
        h.write(&[0]);
    }
    h.finish()
}

/// Materialize the entry-derived scheduler metadata that every
/// sidecar carries regardless of pass/fail/skip: formatted sysctl
/// lines, kargs, and the pretty scheduler name.
///
/// Both write paths (`write_sidecar` and `write_skip_sidecar`) need
/// this exact triple; keeping the derivation in one place means a
/// change to the sidecar schema (e.g. a new scheduler-level field)
/// shows up in all writers automatically.
fn scheduler_fingerprint(entry: &KtstrTestEntry) -> (String, Vec<String>, Vec<String>) {
    let sched_name = entry.scheduler.binary.display_name().to_string();
    let sysctls: Vec<String> = entry
        .scheduler
        .sysctls
        .iter()
        .map(|s| format!("sysctl.{}={}", s.key, s.value))
        .collect();
    let kargs: Vec<String> = entry
        .scheduler
        .kargs
        .iter()
        .map(|s| s.to_string())
        .collect();
    (sched_name, sysctls, kargs)
}

/// Compute the per-variant sidecar path and serialize + write the
/// result to disk.
///
/// Gauntlet variants of the same test differ by work_type, flags
/// (via scheduler args → sysctls/kargs), scheduler, and topology. A
/// filename of just `{test_name}.ktstr.json` causes variants to
/// overwrite each other, erasing all but the last-written result.
/// `sidecar_variant_hash` hashes the discriminating fields into a
/// short stable suffix so each variant gets its own sidecar file.
///
/// `label` is a caller-supplied noun for the context message ("skip
/// sidecar" / "sidecar") so the error chain points at the right call
/// site.
fn serialize_and_write_sidecar(sidecar: &SidecarResult, label: &str) -> anyhow::Result<()> {
    let dir = sidecar_dir();
    let variant_hash = sidecar_variant_hash(sidecar);
    let path = dir.join(format!(
        "{}-{:016x}.ktstr.json",
        sidecar.test_name, variant_hash
    ));
    let json = serde_json::to_string_pretty(sidecar)
        .with_context(|| format!("serialize {label} for '{}'", sidecar.test_name))?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create sidecar dir {}", dir.display()))?;
    std::fs::write(&path, json).with_context(|| format!("write {label} {}", path.display()))?;
    Ok(())
}

/// Emit a minimal sidecar for an early-skip path.
///
/// Stats tooling enumerates sidecars to compute pass/skip/fail
/// rates; when a test bails before `run_ktstr_test_inner` reaches
/// the VM-run site that calls [`write_sidecar`], the skip is
/// invisible to post-run analysis — it shows up as a missing
/// result rather than a recorded skip.
///
/// This helper writes a sidecar flagged `skipped: true, passed: true`
/// with empty VM telemetry (no monitor, no stimulus events, no
/// verifier stats, no kvm stats). Stats tooling that subtracts
/// skipped runs from the pass count treats the entry correctly.
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_skip_sidecar(
    entry: &KtstrTestEntry,
    active_flags: &[String],
) -> anyhow::Result<()> {
    let (scheduler, sysctls, kargs) = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: entry.topology.to_string(),
        scheduler,
        passed: true,
        skipped: true,
        stats: Default::default(),
        monitor: None,
        stimulus_events: Vec::new(),
        // Skip paths never ran a workload; work_type is "skipped"
        // so stats tooling that groups by work_type puts these in a
        // distinguishable bucket.
        work_type: "skipped".to_string(),
        active_flags: active_flags.to_vec(),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
    };
    serialize_and_write_sidecar(&sidecar, "skip sidecar")
}

/// Write a sidecar JSON file for post-run analysis.
///
/// Output goes to the current run's sidecar directory
/// (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{git_short}/`).
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_sidecar(
    entry: &KtstrTestEntry,
    vm_result: &vmm::VmResult,
    stimulus_events: &[StimulusEvent],
    verify_result: &AssertResult,
    work_type: &str,
    active_flags: &[String],
) -> anyhow::Result<()> {
    let (scheduler, sysctls, kargs) = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: entry.topology.to_string(),
        scheduler,
        passed: verify_result.passed,
        skipped: verify_result.is_skipped(),
        stats: verify_result.stats.clone(),
        monitor: vm_result.monitor.as_ref().map(|m| m.summary.clone()),
        stimulus_events: stimulus_events.to_vec(),
        work_type: work_type.to_string(),
        active_flags: active_flags.to_vec(),
        verifier_stats: vm_result.verifier_stats.clone(),
        kvm_stats: vm_result.kvm_stats.clone(),
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
    };
    serialize_and_write_sidecar(&sidecar, "sidecar")
}
