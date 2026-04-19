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
    ///
    /// `default` pairs with `skip_serializing_if` so a sidecar that
    /// never collected verifier stats (empty vec → omitted from
    /// JSON) round-trips cleanly. This is NOT a backward-compat shim
    /// for missing fields in older sidecars.
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
pub(crate) fn sidecar_variant_hash(sidecar: &SidecarResult) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::super::test_helpers::ENV_LOCK;
    use super::*;
    use crate::assert::{AssertResult, CgroupStats};
    use crate::scenario::Ctx;
    use anyhow::Result;

    #[test]
    fn sidecar_result_roundtrip() {
        let sc = SidecarResult {
            test_name: "my_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: true,
            skipped: false,
            stats: crate::assert::ScenarioStats {
                cgroups: vec![CgroupStats {
                    num_workers: 4,
                    num_cpus: 2,
                    avg_off_cpu_pct: 50.0,
                    min_off_cpu_pct: 40.0,
                    max_off_cpu_pct: 60.0,
                    spread: 20.0,
                    max_gap_ms: 100,
                    max_gap_cpu: 1,
                    total_migrations: 5,
                    ..Default::default()
                }],
                total_workers: 4,
                total_cpus: 2,
                total_migrations: 5,
                worst_spread: 20.0,
                worst_gap_ms: 100,
                worst_gap_cpu: 1,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                prog_stats_deltas: None,
                total_samples: 10,
                max_imbalance_ratio: 1.5,
                max_local_dsq_depth: 3,
                stall_detected: false,
                event_deltas: Some(crate::monitor::ScxEventDeltas {
                    total_fallback: 7,
                    fallback_rate: 0.5,
                    max_fallback_burst: 2,
                    total_dispatch_offline: 0,
                    total_dispatch_keep_last: 3,
                    keep_last_rate: 0.2,
                    total_enq_skip_exiting: 0,
                    total_enq_skip_migration_disabled: 0,
                    ..Default::default()
                }),
                schedstat_deltas: None,
                ..Default::default()
            }),
            stimulus_events: vec![crate::timeline::StimulusEvent {
                elapsed_ms: 500,
                label: "StepStart[0]".to_string(),
                op_kind: Some("SetCpuset".to_string()),
                detail: Some("4 cpus".to_string()),
                total_iterations: None,
            }],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string_pretty(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.test_name, "my_test");
        assert_eq!(loaded.topology, "1n2l4c2t");
        assert_eq!(loaded.scheduler, "scx_mitosis");
        assert!(loaded.passed);
        assert_eq!(loaded.stats.total_workers, 4);
        assert_eq!(loaded.stats.cgroups.len(), 1);
        assert_eq!(loaded.stats.cgroups[0].num_workers, 4);
        assert_eq!(loaded.stats.worst_spread, 20.0);
        let mon = loaded.monitor.unwrap();
        assert_eq!(mon.total_samples, 10);
        assert_eq!(mon.max_imbalance_ratio, 1.5);
        assert_eq!(mon.max_local_dsq_depth, 3);
        assert!(!mon.stall_detected);
        let deltas = mon.event_deltas.unwrap();
        assert_eq!(deltas.total_fallback, 7);
        assert_eq!(deltas.total_dispatch_keep_last, 3);
        assert_eq!(loaded.stimulus_events.len(), 1);
        assert_eq!(loaded.stimulus_events[0].label, "StepStart[0]");
    }

    #[test]
    fn sidecar_result_roundtrip_no_monitor() {
        let sc = SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: false,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.test_name, "eevdf_test");
        assert!(!loaded.passed);
        assert!(loaded.monitor.is_none());
        assert!(loaded.stimulus_events.is_empty());
        // monitor field should be absent from JSON when None
        assert!(!json.contains("\"monitor\""));
    }

    // -- collect_sidecars tests --

    #[test]
    fn collect_sidecars_empty_dir() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-empty-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_nonexistent_dir() {
        let results = collect_sidecars(std::path::Path::new("/nonexistent/path"));
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_reads_json() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-json-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let sc = SidecarResult {
            test_name: "test_x".to_string(),
            topology: "1n1l2c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(tmp.join("test_x.ktstr.json"), &json).unwrap();
        // Non-ktstr JSON should be ignored.
        std::fs::write(tmp.join("other.json"), r#"{"key":"val"}"#).unwrap();
        let results = collect_sidecars(&tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "test_x");
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_recurses_one_level() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-recurse-test");
        let sub = tmp.join("job-0");
        std::fs::create_dir_all(&sub).unwrap();
        let sc = SidecarResult {
            test_name: "nested_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: false,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(sub.join("nested_test.ktstr.json"), &json).unwrap();
        let results = collect_sidecars(&tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "nested_test");
        assert!(!results[0].passed);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_does_not_recurse_past_one_level() {
        // Companion to `collect_sidecars_recurses_one_level`: pin the
        // "exactly one level, no deeper" contract. A sidecar two
        // directories deep must be ignored. If a future change
        // switches collect_sidecars to a depth-unbounded walk, this
        // test catches the schema-scope regression before stats
        // tooling starts double-counting results from unrelated
        // sub-runs under the same `runs_root`.
        let tmp = std::env::temp_dir().join("ktstr-sidecars-depth-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let top_sub = tmp.join("job-0");
        let deep_sub = top_sub.join("replay-0");
        std::fs::create_dir_all(&deep_sub).unwrap();

        let sc = |name: &str| SidecarResult {
            test_name: name.to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        // One level: should be collected.
        std::fs::write(
            top_sub.join("top_level.ktstr.json"),
            serde_json::to_string(&sc("top_level")).unwrap(),
        )
        .unwrap();
        // Two levels: must NOT be collected.
        std::fs::write(
            deep_sub.join("deep_level.ktstr.json"),
            serde_json::to_string(&sc("deep_level")).unwrap(),
        )
        .unwrap();

        let results = collect_sidecars(&tmp);
        let names: Vec<&str> = results.iter().map(|r| r.test_name.as_str()).collect();
        assert_eq!(
            names,
            vec!["top_level"],
            "collect_sidecars must see only the one-level-deep sidecar, not the two-level one"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn collect_sidecars_skips_invalid_json() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-invalid-test");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("bad.ktstr.json"), "not json").unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn collect_sidecars_skips_non_ktstr_json() {
        let tmp = std::env::temp_dir().join("ktstr-sidecars-notktstr-test");
        std::fs::create_dir_all(&tmp).unwrap();
        // File ends in .json but does NOT contain ".ktstr." in the name
        std::fs::write(tmp.join("other.json"), r#"{"test":"val"}"#).unwrap();
        let results = collect_sidecars(&tmp);
        assert!(results.is_empty());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn sidecar_result_work_type_field() {
        let sc = SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "Bursty".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vec![],
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.work_type, "Bursty");
    }

    #[test]
    fn write_sidecar_defaults_to_target_dir_without_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let kernel_key = "KTSTR_KERNEL";
        let target_key = "CARGO_TARGET_DIR";
        let prev = std::env::var(key).ok();
        let prev_kernel = std::env::var(kernel_key).ok();
        let prev_target = std::env::var(target_key).ok();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe {
            std::env::remove_var(key);
            std::env::remove_var(kernel_key);
            std::env::remove_var(target_key);
        };

        let dir = sidecar_dir();
        let expected = format!("target/ktstr/unknown-{}", crate::GIT_HASH);
        assert_eq!(dir, PathBuf::from(&expected));

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_default_dir__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let verify_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin", &[]).unwrap();

        // Clean up written file.
        let path = dir.join("__sidecar_default_dir__.ktstr.json");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&dir);

        unsafe {
            match prev {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
            match prev_kernel {
                Some(v) => std::env::set_var(kernel_key, v),
                None => std::env::remove_var(kernel_key),
            }
            match prev_target {
                Some(v) => std::env::set_var(target_key, v),
                None => std::env::remove_var(target_key),
            }
        }
    }

    #[test]
    fn write_sidecar_writes_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-write-test");
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_write_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let verify_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &verify_result, "CpuSpin", &[]).unwrap();

        // Sidecar filename now includes a variant hash suffix so
        // gauntlet variants don't clobber each other. Find the file
        // by prefix match rather than exact path.
        let path: std::path::PathBuf = std::fs::read_dir(&tmp)
            .expect("sidecar dir was created")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("__sidecar_write_test__-") && n.ends_with(".ktstr.json"))
                    .unwrap_or(false)
            })
            .expect("sidecar file with variant suffix should be written");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__sidecar_write_test__");
        assert!(loaded.passed);
        assert!(!loaded.skipped, "pass result is not a skip");

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_active_flags() {
        // Regression for #34: two gauntlet variants differing ONLY in
        // active_flags must produce distinct sidecar filenames so
        // neither clobbers the other. This is the scenario the prior
        // fix (based on work_type/sysctls/kargs alone) missed.
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-flagvariant-test");
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__flagvariant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let ok = AssertResult::pass();
        let flags_a = vec!["llc".to_string()];
        let flags_b = vec!["llc".to_string(), "steal".to_string()];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_a).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_b).unwrap();

        let names: Vec<String> = std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("__flagvariant_test__-"))
            .collect();
        assert_eq!(
            names.len(),
            2,
            "two active_flags variants must produce two distinct files, got {names:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_work_types() {
        // Regression for #34: two gauntlet variants differing only in
        // work_type must produce distinct sidecar filenames so neither
        // clobbers the other.
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-variant-test");
        let _ = std::fs::remove_dir_all(&tmp);
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__variant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult {
            success: true,
            exit_code: 0,
            duration: std::time::Duration::from_secs(1),
            timed_out: false,
            output: String::new(),
            stderr: String::new(),
            monitor: None,
            shm_data: None,
            stimulus_events: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            crash_message: None,
        };
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "YieldHeavy", &[]).unwrap();

        let names: Vec<String> = std::fs::read_dir(&tmp)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("__variant_test__-"))
            .collect();
        assert_eq!(
            names.len(),
            2,
            "two work_type variants must produce two distinct files, got {names:?}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    /// Freeze the `sidecar_variant_hash` wire format to the exact 64-bit
    /// value produced for a representative populated SidecarResult.
    ///
    /// Sidecar filenames embed this hash as a hex suffix; gauntlet
    /// tooling groups variants by it. A silent change — e.g. bumping
    /// `siphasher`, switching keys, or reordering fields fed into the
    /// hasher — would let old-version tooling mis-group new-version
    /// sidecars and vice versa. Pinning the output against a
    /// pre-computed constant catches that drift before it ships.
    ///
    /// Only fields that feed the hash (topology, scheduler, work_type,
    /// active_flags, sysctls, kargs) are set to significant values;
    /// irrelevant fields are zero-ish defaults so this test stays
    /// robust against unrelated `SidecarResult` schema growth.
    #[test]
    fn sidecar_variant_hash_stability_populated() {
        use crate::assert::ScenarioStats;
        let sc = SidecarResult {
            test_name: String::new(),
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx-ktstr".to_string(),
            passed: false,
            skipped: false,
            stats: ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: "CpuSpin".to_string(),
            active_flags: vec!["llc".to_string(), "steal".to_string()],
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: vec!["sysctl.kernel.sched_cfs_bandwidth_slice_us=1000".to_string()],
            kargs: vec!["nosmt".to_string()],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        // If this assertion trips, the wire format changed. Bumping
        // the expected value is the wrong fix unless you also plan
        // for old sidecars to be regenerated — see the contract on
        // `sidecar_variant_hash`.
        assert_eq!(
            sidecar_variant_hash(&sc),
            0xcd4044360a818e72,
            "sidecar_variant_hash output drifted — regenerate expected only if \
             the wire format change is intentional and old sidecars are \
             disposable (which they are per ktstr's pre-1.0 stance)",
        );
    }

    /// Pair to [`sidecar_variant_hash_stability_populated`] covering
    /// the empty-collections path. If the inter-collection separator
    /// bytes (0xfe / 0xfd / 0xff) disappear or change, an empty-
    /// flags variant could collide with an empty-sysctls variant
    /// whose kargs start with bytes that happen to match the dropped
    /// separator. Pinning the empty-inputs hash catches separator
    /// regressions.
    #[test]
    fn sidecar_variant_hash_stability_empty_collections() {
        use crate::assert::ScenarioStats;
        let sc = SidecarResult {
            test_name: String::new(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            passed: false,
            skipped: false,
            stats: ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: String::new(),
            active_flags: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        };
        assert_eq!(sidecar_variant_hash(&sc), 0xe6b48e8fa3394bd8);
    }

    // -- format_verifier_stats tests --

    fn make_sidecar_with_vstats(
        vstats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    ) -> SidecarResult {
        SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "test".to_string(),
            passed: true,
            skipped: false,
            stats: Default::default(),
            monitor: None,
            stimulus_events: vec![],
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: vstats,
            kvm_stats: None,
            sysctls: vec![],
            kargs: vec![],
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
        }
    }

    #[test]
    fn format_verifier_stats_empty() {
        assert!(format_verifier_stats(&[]).is_empty());
    }

    #[test]
    fn format_verifier_stats_no_data() {
        let sc = make_sidecar_with_vstats(vec![]);
        assert!(format_verifier_stats(&[sc]).is_empty());
    }

    #[test]
    fn format_verifier_stats_table() {
        let sc = make_sidecar_with_vstats(vec![
            crate::monitor::bpf_prog::ProgVerifierStats {
                name: "dispatch".to_string(),
                verified_insns: 50000,
            },
            crate::monitor::bpf_prog::ProgVerifierStats {
                name: "enqueue".to_string(),
                verified_insns: 30000,
            },
        ]);
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("BPF VERIFIER STATS"));
        assert!(result.contains("dispatch"));
        assert!(result.contains("enqueue"));
        assert!(result.contains("50000"));
        assert!(result.contains("30000"));
        assert!(result.contains("total verified insns: 80000"));
        assert!(!result.contains("WARNING"));
    }

    #[test]
    fn format_verifier_stats_warning() {
        let sc = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "heavy".to_string(),
            verified_insns: 800000,
        }]);
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("WARNING"));
        assert!(result.contains("heavy"));
        assert!(result.contains("80.0%"));
    }

    #[test]
    fn sidecar_verifier_stats_serde_roundtrip() {
        let sc = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "init".to_string(),
            verified_insns: 5000,
        }]);
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("verifier_stats"));
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verifier_stats.len(), 1);
        assert_eq!(loaded.verifier_stats[0].name, "init");
        assert_eq!(loaded.verifier_stats[0].verified_insns, 5000);
    }

    #[test]
    fn sidecar_verifier_stats_empty_omitted() {
        let sc = make_sidecar_with_vstats(vec![]);
        let json = serde_json::to_string(&sc).unwrap();
        assert!(!json.contains("verifier_stats"));
    }

    #[test]
    fn format_verifier_stats_deduplicates() {
        let sc1 = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 50000,
        }]);
        let sc2 = make_sidecar_with_vstats(vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 50000,
        }]);
        let result = format_verifier_stats(&[sc1, sc2]);
        // Deduplicated: total should be 50000, not 100000.
        assert!(result.contains("total verified insns: 50000"));
    }

    // -- scheduler_fingerprint --

    #[test]
    fn scheduler_fingerprint_eevdf_empty_extras() {
        // Default scheduler (EEVDF) has no sysctls/kargs; fingerprint
        // returns the display name and two empty vecs.
        let entry = KtstrTestEntry {
            name: "eevdf_test",
            ..KtstrTestEntry::DEFAULT
        };
        let (name, sysctls, kargs) = scheduler_fingerprint(&entry);
        assert_eq!(name, "eevdf");
        assert!(sysctls.is_empty());
        assert!(kargs.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_formats_sysctls_with_prefix() {
        use super::super::entry::Sysctl;
        static SYSCTLS: &[Sysctl] = &[
            Sysctl::new("kernel.foo", "1"),
            Sysctl::new("kernel.bar", "yes"),
        ];
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").sysctls(SYSCTLS);
        let entry = KtstrTestEntry {
            name: "s_test",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (name, sysctls, kargs) = scheduler_fingerprint(&entry);
        assert_eq!(name, "eevdf");
        assert_eq!(
            sysctls,
            vec![
                "sysctl.kernel.foo=1".to_string(),
                "sysctl.kernel.bar=yes".to_string(),
            ]
        );
        assert!(kargs.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_forwards_kargs_verbatim() {
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").kargs(&["quiet", "splash"]);
        let entry = KtstrTestEntry {
            name: "s_test",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (_name, sysctls, kargs) = scheduler_fingerprint(&entry);
        assert_eq!(kargs, vec!["quiet".to_string(), "splash".to_string()]);
        assert!(sysctls.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_uses_display_name_for_discover() {
        use super::super::entry::SchedulerSpec;
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").binary(SchedulerSpec::Discover("scx_relaxed"));
        let entry = KtstrTestEntry {
            name: "rel_test",
            scheduler: &SCHED,
            ..KtstrTestEntry::DEFAULT
        };
        let (name, _, _) = scheduler_fingerprint(&entry);
        assert_eq!(name, "scx_relaxed");
    }

    // -- write_skip_sidecar --

    /// `write_skip_sidecar` is the path covered by the ResourceContention
    /// skip branch and any early-exit that bails before `run_ktstr_test_inner`
    /// reaches the VM-run call site. The sidecar must be flagged
    /// `skipped: true, passed: true` so stats tooling that subtracts
    /// skipped runs from pass counts sees a recorded skip instead of
    /// a missing file. This regression guards that contract against a
    /// future change that forgets the passed-true flag or drops skip
    /// sidecars entirely for non-VM early exits.
    #[test]
    fn write_skip_sidecar_records_passed_true_skipped_true() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-writes-test");
        let _ = std::fs::remove_dir_all(&tmp);
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, tmp.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skip_sidecar_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let active_flags: Vec<String> = vec!["llc".to_string()];
        write_skip_sidecar(&entry, &active_flags).expect("skip sidecar must write");

        let path: std::path::PathBuf = std::fs::read_dir(&tmp)
            .expect("skip sidecar dir was created")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.starts_with("__skip_sidecar_test__-") && n.ends_with(".ktstr.json")
                })
            })
            .expect("skip sidecar file with variant suffix should be written");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__skip_sidecar_test__");
        assert!(
            loaded.passed,
            "skip sidecar must set passed=true so the verdict gate does not flip fail",
        );
        assert!(
            loaded.skipped,
            "skip sidecar must set skipped=true so stats tooling excludes from pass count",
        );
        assert_eq!(
            loaded.work_type, "skipped",
            "skip path uses the 'skipped' work_type bucket so grouping keeps the skip distinguishable",
        );
        assert_eq!(loaded.active_flags, active_flags);

        let _ = std::fs::remove_dir_all(&tmp);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }

    /// When the sidecar directory cannot be created (path collision
    /// with a regular file), `write_skip_sidecar` must return `Err`
    /// rather than silently eating the failure. Stats tooling relies
    /// on the error chain to diagnose missing sidecars; a swallowed
    /// error would make skips invisible to post-run analysis.
    #[test]
    fn write_skip_sidecar_returns_err_when_dir_cannot_be_created() {
        let _guard = ENV_LOCK.lock().unwrap();
        let key = "KTSTR_SIDECAR_DIR";
        let prev = std::env::var(key).ok();

        // Create a regular file, then try to use it as the sidecar
        // directory. `create_dir_all` fails because the path exists
        // but is not a directory.
        let blocker = std::env::temp_dir().join("ktstr-sidecar-skip-blocker");
        let _ = std::fs::remove_file(&blocker);
        let _ = std::fs::remove_dir_all(&blocker);
        std::fs::write(&blocker, b"not a dir").unwrap();
        // SAFETY: test-only, single-threaded env mutation with save/restore.
        unsafe { std::env::set_var(key, blocker.to_str().unwrap()) };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skip_sidecar_err_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let result = write_skip_sidecar(&entry, &[]);
        assert!(
            result.is_err(),
            "skip sidecar write must return Err when the target is a regular file",
        );

        let _ = std::fs::remove_file(&blocker);
        match prev {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
    }
}
