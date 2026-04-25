//! Per-run sidecar JSON — the durable record of a ktstr test outcome.
//!
//! Every test (pass, fail, or skip) writes a [`SidecarResult`] to a
//! JSON file under the run's sidecar directory; downstream analysis
//! (`cargo ktstr stats`, CI dashboards) aggregates those files to
//! compute pass/fail rates, verifier stats, callback profiles, and
//! KVM stats across gauntlet variants.
//!
//! Responsibilities owned by this module:
//! - [`SidecarResult`]: the on-disk schema. Fields serialize as
//!   `null` / `[]` when empty — no `skip_serializing_if` or
//!   `serde(default)` — so serialize and deserialize are symmetric.
//!   A missing field in a parsed sidecar is a hard error (pre-1.0:
//!   old sidecar JSON is disposable; regenerate by re-running the
//!   test).
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
//!   `kernel_version` field, with fallback to
//!   `include/config/kernel.release` in the kernel source tree
//!   when the cache metadata is absent or does not carry a
//!   version (e.g. a raw source-tree path set in `KTSTR_KERNEL`
//!   rather than a cache key).

use std::path::PathBuf;

use anyhow::Context;

use crate::assert::{AssertResult, ScenarioStats};
use crate::monitor::MonitorSummary;
use crate::test_support::PayloadMetrics;
use crate::timeline::StimulusEvent;
use crate::vmm;

use super::entry::KtstrTestEntry;
use super::timefmt::{generate_run_id, now_iso8601, run_id_timestamp};

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
    /// Best-effort git commit of the scheduler binary used for this
    /// run. Currently ALWAYS `None` for every `SchedulerSpec`
    /// variant — no variant today has a reliable commit source.
    /// The field is reserved on the schema so stats tooling can
    /// enrich it once a reliable source exists (e.g. a
    /// `--version` probe or ELF-note read on the resolved
    /// scheduler binary). See
    /// [`crate::test_support::SchedulerSpec::scheduler_commit`]
    /// for the full per-variant rationale.
    ///
    /// Always emitted (`"scheduler_commit": null` on absence);
    /// required on deserialize — matches every other nullable on
    /// this struct.
    pub scheduler_commit: Option<String>,
    /// Binary payload name (matches `Payload::name` when
    /// `entry.payload` is set). `None` when the test declared no
    /// binary payload. Serialized as `"payload": null` in that case;
    /// required on deserialize — matches `host`'s symmetric pattern.
    pub payload: Option<String>,
    /// Per-payload extracted metrics collected from `ctx.payload(X).run()`
    /// / `.spawn().wait()` call sites during the test body.
    ///
    /// One [`PayloadMetrics`] per invocation, in the order the calls
    /// ran. Empty when no payload calls were made (scheduler-only
    /// tests, or a binary-only test where the body bailed before
    /// running the payload). Always emitted as `"metrics": []` in
    /// that case; required on deserialize.
    pub metrics: Vec<PayloadMetrics>,
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
    /// produced no valid data. Always emitted (`"monitor": null` on
    /// absence); required on deserialize.
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
    /// scheduler (when one was loaded). Empty when no scheduler
    /// programs were inspected. Always emitted as `"verifier_stats":
    /// []` in that case; required on deserialize.
    pub verifier_stats: Vec<crate::monitor::bpf_prog::ProgVerifierStats>,
    /// Aggregate per-vCPU KVM stats read after VM exit. `None` when
    /// the VM did not run (host-only tests) or KVM stats were
    /// unavailable. Always emitted as `"kvm_stats": null` on absence;
    /// required on deserialize.
    pub kvm_stats: Option<crate::vmm::KvmStatsTotals>,
    /// Effective sysctls active during this test run, recorded as raw
    /// `sysctl.key=value` cmdline strings. Always emitted as
    /// `"sysctls": []` when none; required on deserialize.
    pub sysctls: Vec<String>,
    /// Effective kernel command-line args active during this test run.
    /// Always emitted as `"kargs": []` when none; required on
    /// deserialize.
    pub kargs: Vec<String>,
    /// Kernel version of the VM under test (from cache metadata,
    /// e.g. `"6.14.2"`). Populated from the cache entry's
    /// `metadata.json` version field, with fallback to the kernel
    /// source tree's `include/config/kernel.release` when
    /// `KTSTR_KERNEL` points at a raw source path rather than a
    /// cache key; `None` for host-only tests or when neither
    /// source yields a version string. The host's running kernel
    /// release is carried separately in `host.kernel_release`.
    /// Always emitted (`"kernel_version": null` on absence);
    /// required on deserialize.
    pub kernel_version: Option<String>,
    /// ISO 8601 timestamp of when this test run started.
    pub timestamp: String,
    /// Unique identifier for the test run. Derived from the repo commit
    /// hash and a monotonic counter to distinguish runs within the same
    /// build.
    pub run_id: String,
    /// Host context — static-ish runtime state (CPU model,
    /// memory size, THP policy, kernel release, host cmdline,
    /// scheduler tunables). Populated by production sidecar
    /// writers; `None` on the test-fixture path.
    /// Deliberately excluded from the variant hash so
    /// gauntlet variants on different hosts collapse into the same
    /// hash bucket.
    ///
    /// No serde attributes: the field is always emitted
    /// (`"host": null` when `None`) and always required on
    /// deserialize. Every other `Option` and `Vec` field on this
    /// struct follows the same pattern — `serde(default)` and
    /// `skip_serializing_if` have been removed crate-wide so
    /// serialize and deserialize are symmetric for all sidecar
    /// fields. A missing field in a parsed sidecar is a hard error;
    /// pre-1.0, sidecar data is disposable, so regenerate by
    /// re-running the test rather than carrying a compat shim for
    /// older JSON.
    pub host: Option<crate::host_context::HostContext>,
}

#[cfg(test)]
impl SidecarResult {
    /// Populated [`SidecarResult`] for unit tests. Every field has a
    /// reasonable default so call sites only spell out what they want
    /// to vary via struct-update syntax:
    ///
    /// ```ignore
    /// let sc = SidecarResult {
    ///     test_name: "my_test".to_string(),
    ///     passed: false,
    ///     ..SidecarResult::test_fixture()
    /// };
    /// ```
    ///
    /// Defaults model a passing EEVDF run on a minimal `1n1l1c1t`
    /// topology with no payload and no VM telemetry: `test_name="t"`,
    /// `topology="1n1l1c1t"`, `scheduler="eevdf"`, `work_type="CpuSpin"`,
    /// `passed=true`, `skipped=false`, every [`Option`] `None`, every
    /// [`Vec`] empty, `stats` is `ScenarioStats::default()`, and both
    /// `timestamp`/`run_id` are empty strings.
    ///
    /// **Prefer this over local `base = || SidecarResult { ... }`
    /// closures.** A local closure duplicates the default set and
    /// drifts the moment [`SidecarResult`] grows a field; this fixture
    /// is the single place those defaults live.
    ///
    /// **Hash-stability tests must not rely on these defaults for
    /// hash-participating fields** (`topology`, `scheduler`, `payload`,
    /// `work_type`, `active_flags`, `sysctls`, `kargs`). Tests that pin
    /// a [`sidecar_variant_hash`] output against a literal constant
    /// must spell every hash-participating field out explicitly so a
    /// future change to these defaults cannot silently shift the
    /// pinned value.
    pub(crate) fn test_fixture() -> SidecarResult {
        SidecarResult {
            test_name: "t".to_string(),
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            scheduler_commit: None,
            payload: None,
            metrics: Vec::new(),
            passed: true,
            skipped: false,
            stats: crate::assert::ScenarioStats::default(),
            monitor: None,
            stimulus_events: Vec::new(),
            work_type: "CpuSpin".to_string(),
            active_flags: Vec::new(),
            verifier_stats: Vec::new(),
            kvm_stats: None,
            sysctls: Vec::new(),
            kargs: Vec::new(),
            kernel_version: None,
            timestamp: String::new(),
            run_id: String::new(),
            host: None,
        }
    }
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
            Err(e) => {
                // Enrich the diagnostic when a serde "missing field"
                // error mentions `host` (the most common miss after
                // the host-context landing) — point the operator at
                // the fix that pre-1.0 disposable-sidecar policy
                // calls for: re-run the test to regenerate the
                // sidecar under the current schema. Generic errors
                // fall through to the unadorned message so the
                // original serde line number / column remain visible.
                //
                // Matching on the Display text is deliberate: serde's
                // typed-error surface for `missing field "X"` is not
                // stable across serde_json versions, but the rendered
                // message is — a forward-compat regression-resilient
                // check costs one string search.
                let msg = e.to_string();
                let is_missing_host = msg.contains("missing field") && msg.contains("`host`");
                if is_missing_host {
                    eprintln!(
                        "ktstr_test: skipping {}: {e} — the `host` field was \
                         added to SidecarResult; pre-1.0 policy is \
                         disposable-sidecar: re-run the test to \
                         regenerate this file under the current schema \
                         (no migration shim exists)",
                        path.display(),
                    );
                } else {
                    eprintln!("ktstr_test: skipping {}: {e}", path.display());
                }
            }
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
/// Default: `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{timestamp}/`,
/// where `{kernel}` is the version detected from `KTSTR_KERNEL`'s
/// metadata (or `"unknown"` when no kernel is set / detection fails)
/// and `{timestamp}` is the compact `YYYYMMDDTHHMMSSZ` stamp captured
/// once per process by [`run_id_timestamp`]. Every sidecar written
/// from the same `cargo ktstr test` invocation lands in the same
/// directory; successive invocations get distinct directories so the
/// "runs ARE baselines" archival model retains all runs even when
/// the same kernel is re-tested.
pub(crate) fn sidecar_dir() -> PathBuf {
    if let Ok(d) = std::env::var("KTSTR_SIDECAR_DIR")
        && !d.is_empty()
    {
        return PathBuf::from(d);
    }
    let kernel = detect_kernel_version().unwrap_or_else(|| "unknown".to_string());
    runs_root().join(format!("{kernel}-{}", run_id_timestamp()))
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
/// run a kernel, so it can't reconstruct the `{kernel}-{timestamp}`
/// key that the test process used. Picking the newest subdirectory by
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

/// Detect the kernel version associated with the current test run.
///
/// Routes through [`crate::ktstr_kernel_env`] for the raw env value
/// and [`crate::kernel_path::KernelId`] for variant dispatch so the
/// three [`KernelId`] variants are honoured symmetrically:
///
/// - `KernelId::Path(dir)`: read `metadata.json` (cache entry
///   layout) or `include/config/kernel.release` (source tree
///   layout). Unchanged from the previous behaviour.
/// - `KernelId::Version(ver)`: the user asked for a specific
///   version — return it directly. No cache access needed; a
///   version string IS a version string.
/// - `KernelId::CacheKey(key)`: look up the cache entry and
///   return `entry.metadata.version`. The previous code path
///   silently treated the key as a directory name and read
///   `<cwd>/<key>/metadata.json`, which never matched — producing
///   `None` + `sidecar_dir()` using the `"unknown"` fallback even
///   though the cache metadata already carried the version.
///
/// Returns `None` when the env var is unset, or when the env
/// resolves to a variant whose underlying source doesn't yield a
/// version string (e.g. a Path whose metadata.json / kernel.release
/// are both absent, or a CacheKey with no cache hit).
pub(crate) fn detect_kernel_version() -> Option<String> {
    use crate::kernel_path::KernelId;
    let raw = crate::ktstr_kernel_env()?;
    match KernelId::parse(&raw) {
        KernelId::Path(_) => {
            let p = std::path::Path::new(&raw);
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
        KernelId::Version(ver) => Some(ver),
        KernelId::CacheKey(key) => {
            let cache = crate::cache::CacheDir::new().ok()?;
            let entry = cache.lookup(&key)?;
            entry.metadata.version
        }
    }
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
///
/// # Host-state collision caveat
///
/// The hash is over test-identity fields (topology, scheduler,
/// payload, work_type, flags, sysctls, kargs) — NOT over
/// [`HostContext`], and NOT over `scheduler_commit`. The
/// [`HostContext`] exclusion is pinned by
/// [`sidecar_variant_hash_excludes_host_context`]; the
/// `scheduler_commit` exclusion is deliberate for the same
/// cross-host grouping reason — a gauntlet rebuilt against a
/// different userspace scheduler commit (bumped ktstr checkout,
/// different CI runner, different developer machine) must still
/// bucket with the same-named variant so `compare_runs` can diff
/// two runs of the "same" test without the commit hash shattering
/// them into one-row-per-commit islands. Callers that want to
/// detect a commit drift between two runs inspect
/// `SidecarResult::scheduler_commit` directly; the filename stays
/// stable across commits by design.
///
/// The corollary of the HostContext exclusion: if the host's
/// observable state mutates mid-suite — NUMA hotplug, hugepage
/// reconfiguration, a `sysctl -w` from a parallel process — two
/// runs of the same test will produce the same sidecar filename
/// and the later write clobbers the earlier. ktstr treats host
/// state as stable-enough for a single suite run; callers
/// mutating host state during a run own the ordering themselves
/// (e.g. by writing to a different `KTSTR_SIDECAR_DIR` per host
/// snapshot).
pub(crate) fn sidecar_variant_hash(sidecar: &SidecarResult) -> u64 {
    use siphasher::sip::SipHasher13;
    use std::hash::Hasher;
    let mut h = SipHasher13::new_with_keys(0, 0);
    h.write(sidecar.topology.as_bytes());
    h.write(&[0]);
    h.write(sidecar.scheduler.as_bytes());
    h.write(&[0]);
    // Binary payload name — two tests that differ only in the
    // primary payload (e.g. scheduler=EEVDF + payload=FIO vs
    // scheduler=EEVDF + payload=STRESS_NG) must produce distinct
    // sidecar filenames. `None` emits a single separator byte so the
    // absent-payload variant doesn't collide with a payload name that
    // happens to hash-chain into the next field.
    h.write(&[0xfc]);
    if let Some(name) = &sidecar.payload {
        h.write(name.as_bytes());
    }
    h.write(&[0]);
    h.write(sidecar.work_type.as_bytes());
    h.write(&[0]);
    h.write(&[0xfe]);
    for f in &sidecar.active_flags {
        h.write(f.as_bytes());
        h.write(&[0]);
    }
    // Sysctls and kargs are canonicalized at hash time — NOT at
    // write time like `active_flags` — so the on-disk sidecar
    // preserves the scheduler-declared order (useful for humans
    // reading the JSON) while the filename suffix stays a pure
    // function of the SET, not the sequence. Sorting lexically
    // here means two schedulers that declare the same sysctls in
    // different source-code orders fold to the same filename,
    // matching the order-insensitivity contract documented on
    // `canonicalize_active_flags`. Two small `Vec<&str>` per
    // call — acceptable because `sidecar_variant_hash` runs
    // once per `write_sidecar`, not on a hot path.
    h.write(&[0xfd]);
    let mut sorted_sysctls: Vec<&str> = sidecar.sysctls.iter().map(String::as_str).collect();
    sorted_sysctls.sort_unstable();
    for s in &sorted_sysctls {
        h.write(s.as_bytes());
        h.write(&[0]);
    }
    h.write(&[0xff]);
    let mut sorted_kargs: Vec<&str> = sidecar.kargs.iter().map(String::as_str).collect();
    sorted_kargs.sort_unstable();
    for k in &sorted_kargs {
        h.write(k.as_bytes());
        h.write(&[0]);
    }
    h.finish()
}

/// Entry-derived scheduler metadata that every sidecar carries
/// regardless of pass/fail/skip.
///
/// Both write paths ([`write_sidecar`] and [`write_skip_sidecar`])
/// thread the same materialized fields through to their
/// `SidecarResult` constructors; keeping the derivation in a
/// named struct (rather than a 4-tuple) means a new
/// scheduler-level field shows up as a named field at both
/// writer sites and in every call-site binding, instead of as
/// an additional anonymous tuple slot that readers have to
/// remember the ordering of.
///
/// `pub(crate)` rather than `pub`: the intermediate struct is a
/// write-path detail, not a public API surface. No serde — this
/// is not a persisted shape, just a grouped return value.
///
/// Derives `Debug` for `assert_eq!` diagnostics, `Clone` so tests
/// can materialize a fixture once and reuse it across assertions,
/// and `PartialEq`/`Eq` so tests can compare whole fingerprints
/// in one statement rather than destructuring and asserting on
/// each field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SchedulerFingerprint {
    /// Pretty scheduler name (matches `SidecarResult::scheduler`),
    /// e.g. `"eevdf"` or a scheduler-kind payload's declared name.
    pub(crate) scheduler: String,
    /// Best-effort userspace scheduler commit; `None` for every
    /// current variant per
    /// [`crate::test_support::SchedulerSpec::scheduler_commit`].
    pub(crate) scheduler_commit: Option<String>,
    /// Formatted `sysctl.<key>=<value>` lines derived from the
    /// scheduler's declared `sysctls()`.
    pub(crate) sysctls: Vec<String>,
    /// Kernel command-line args declared by the scheduler,
    /// forwarded verbatim.
    pub(crate) kargs: Vec<String>,
}

/// Materialize the [`SchedulerFingerprint`] for a test entry.
///
/// A change to the sidecar schema (e.g. a new scheduler-level
/// field) extends this function + [`SchedulerFingerprint`] in
/// one place and every writer picks it up automatically.
fn scheduler_fingerprint(entry: &KtstrTestEntry) -> SchedulerFingerprint {
    let scheduler = entry.scheduler.scheduler_name().to_string();
    // `entry.scheduler` is a `&Payload` wrapper, not a `&Scheduler`
    // directly — routing through `scheduler_binary()` returns the
    // underlying `Option<&SchedulerSpec>` (None for binary-kind
    // payloads). Flatten with `and_then` so a binary-kind payload
    // naturally yields `None` without duplicating the
    // binary-vs-scheduler dispatch logic here.
    let scheduler_commit = entry
        .scheduler
        .scheduler_binary()
        .and_then(|s| s.scheduler_commit())
        .map(|s| s.to_string());
    let sysctls: Vec<String> = entry
        .scheduler
        .sysctls()
        .iter()
        .map(|s| format!("sysctl.{}={}", s.key, s.value))
        .collect();
    let kargs: Vec<String> = entry
        .scheduler
        .kargs()
        .iter()
        .map(|s| s.to_string())
        .collect();
    SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    }
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

/// Return `active_flags` sorted into canonical
/// [`crate::scenario::flags::ALL`] order. Both sidecar writers
/// pipe their caller-supplied flag slice through this helper so
/// the persisted ordering is a pure function of the flag SET,
/// not the order the caller happened to accumulate them in.
///
/// Why this matters: [`sidecar_variant_hash`] walks
/// `active_flags` in-order and folds each byte into a SipHasher
/// state (see sibling site that hashes `for f in
/// &sidecar.active_flags`). Two runs of the same semantic variant
/// that differ only in flag accumulation order — e.g. a gauntlet
/// path that inserts `llc` then `steal` versus one that inserts
/// `steal` then `llc` — would otherwise produce distinct hashes,
/// distinct sidecar filenames, and end up as two separate rows in
/// `compare_runs` even though they describe the same variant. By
/// canonicalizing at write time against the canonical
/// [`crate::scenario::flags::ALL`] positional ordering (shared
/// with `compute_flag_profiles` at scenario/mod.rs, which sorts
/// the same way), the on-disk representation is
/// order-insensitive by construction.
///
/// Flags not found in [`crate::scenario::flags::ALL`] are kept
/// and sorted to the end in lexical order. Sort key is composite:
/// positional for known flags (so the canonical ALL order leads),
/// then `&str` comparison as a tiebreaker. The lexical secondary
/// matters because two unknown flags both collide on the fallback
/// `usize::MAX` positional key — without the tiebreak, a caller
/// that supplies `["zzz_unknown", "aaa_unknown"]` versus the
/// reverse would share identical positional keys yet produce
/// different on-disk orderings under a stable sort, once again
/// breaking the "variant hash is a pure function of the flag
/// SET" invariant. The lexical secondary collapses them to one
/// canonical order so future or ad-hoc flag names are handled
/// without data loss AND without order sensitivity.
fn canonicalize_active_flags(flags: &[String]) -> Vec<String> {
    let mut v: Vec<String> = flags.to_vec();
    v.sort_by(|a, b| {
        let ka = crate::scenario::flags::ALL
            .iter()
            .position(|x| *x == a.as_str())
            .unwrap_or(usize::MAX);
        let kb = crate::scenario::flags::ALL
            .iter()
            .position(|x| *x == b.as_str())
            .unwrap_or(usize::MAX);
        ka.cmp(&kb).then_with(|| a.as_str().cmp(b.as_str()))
    });
    v
}

/// Emit a minimal sidecar for a PRE-VM-BOOT skip path.
///
/// Stats tooling enumerates sidecars to compute pass/skip/fail
/// rates; when a test bails before `run_ktstr_test_inner` reaches
/// the VM-run site that calls [`write_sidecar`], the skip is
/// invisible to post-run analysis — it shows up as a missing
/// result rather than a recorded skip.
///
/// This helper writes a sidecar flagged `skipped: true, passed: true`
/// with empty VM telemetry (no monitor, no stimulus events, no
/// verifier stats, no kvm stats, no payload metrics). Stats tooling
/// that subtracts skipped runs from the pass count treats the entry
/// correctly.
///
/// # Distinction from in-VM `AssertResult::skip` paths
///
/// There are TWO classes of skip, each with its own sidecar writer:
///
/// 1. **Pre-VM-boot skips** route through this helper
///    (`write_skip_sidecar`). Examples:
///    - `performance_mode` gated off via `KTSTR_NO_PERF_MODE`
///      (see `run_ktstr_test_inner`),
///    - `ResourceContention` at `builder.build()` or `vm.run()`
///      (topology-level unavailability — the VM never booted).
///
///    These paths write a MINIMAL sidecar: empty VM telemetry,
///    `work_type = "skipped"`, and `payload` pinned to the entry's
///    declared payload so stats can still attribute the skip to
///    the correct gauntlet variant. There is no VmResult to drain
///    because the VM didn't boot.
///
/// 2. **In-VM `AssertResult::skip` returns** — e.g. the
///    empty-cpuset skip in `scenario::run_scenario`
///    (`AssertResult::skip("not enough CPUs/LLCs")`), or the
///    `need >= 4 CPUs` checks in `scenario::dynamic::*` — route
///    through [`write_sidecar`] at `run_ktstr_test_inner`'s end.
///    The guest VM fully booted, ran through scenario setup,
///    discovered the topology couldn't accommodate the test, and
///    returned early. The resulting sidecar carries REAL VM
///    telemetry (monitor, kvm_stats, verifier_stats) alongside
///    `skipped: true` — not a blind spot, just a richer record
///    than what this helper emits.
///
/// The asymmetry is intentional: pre-VM-boot skips have no
/// telemetry to record, while in-VM skips do. Stats tooling that
/// wants to uniformly discount skipped runs filters on
/// [`SidecarResult::skipped == true`] regardless of which writer
/// produced the entry — both set the field identically.
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_skip_sidecar(
    entry: &KtstrTestEntry,
    active_flags: &[String],
) -> anyhow::Result<()> {
    let SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: entry.topology.to_string(),
        scheduler,
        scheduler_commit,
        // A skip never runs the payload. Still record the declared
        // payload name so stats tooling can attribute the skip to
        // the payload-gauntlet variant rather than losing the
        // association.
        payload: entry.payload.map(|p| p.name.to_string()),
        metrics: Vec::new(),
        passed: true,
        skipped: true,
        stats: Default::default(),
        monitor: None,
        stimulus_events: Vec::new(),
        // Skip paths never ran a workload; work_type is "skipped"
        // so stats tooling that groups by work_type puts these in a
        // distinguishable bucket.
        work_type: "skipped".to_string(),
        active_flags: canonicalize_active_flags(active_flags),
        verifier_stats: Vec::new(),
        kvm_stats: None,
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
        host: Some(crate::host_context::collect_host_context()),
    };
    serialize_and_write_sidecar(&sidecar, "skip sidecar")
}

/// Write a sidecar JSON file for post-run analysis.
///
/// Output goes to the current run's sidecar directory
/// (`KTSTR_SIDECAR_DIR` override, or
/// `{CARGO_TARGET_DIR or "target"}/ktstr/{kernel}-{timestamp}/`).
///
/// `payload_metrics` is the accumulated per-invocation output from
/// `ctx.payload(X).run()` / `.spawn().wait()` calls made in the
/// test body. Empty vec when the test body never called
/// `Ctx::payload` (scheduler-only tests, host-only probes).
///
/// Returns `Err` when the sidecar directory cannot be created, the
/// JSON cannot be serialized, or the file write fails. Callers that
/// ignore the Result accept the risk of stats-tooling blind spots on
/// this run.
pub(crate) fn write_sidecar(
    entry: &KtstrTestEntry,
    vm_result: &vmm::VmResult,
    stimulus_events: &[StimulusEvent],
    check_result: &AssertResult,
    work_type: &str,
    active_flags: &[String],
    payload_metrics: &[PayloadMetrics],
) -> anyhow::Result<()> {
    let SchedulerFingerprint {
        scheduler,
        scheduler_commit,
        sysctls,
        kargs,
    } = scheduler_fingerprint(entry);
    let sidecar = SidecarResult {
        test_name: entry.name.to_string(),
        topology: entry.topology.to_string(),
        scheduler,
        scheduler_commit,
        payload: entry.payload.map(|p| p.name.to_string()),
        metrics: payload_metrics.to_vec(),
        passed: check_result.passed,
        skipped: check_result.is_skipped(),
        stats: check_result.stats.clone(),
        monitor: vm_result.monitor.as_ref().map(|m| m.summary.clone()),
        stimulus_events: stimulus_events.to_vec(),
        work_type: work_type.to_string(),
        active_flags: canonicalize_active_flags(active_flags),
        verifier_stats: vm_result.verifier_stats.clone(),
        kvm_stats: vm_result.kvm_stats.clone(),
        sysctls,
        kargs,
        kernel_version: detect_kernel_version(),
        timestamp: now_iso8601(),
        run_id: generate_run_id(),
        host: Some(crate::host_context::collect_host_context()),
    };
    serialize_and_write_sidecar(&sidecar, "sidecar")
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::{EnvVarGuard, lock_env};
    use super::*;
    use crate::assert::{AssertResult, CgroupStats};
    use crate::scenario::Ctx;
    use anyhow::Result;

    /// Collect every sidecar file in `dir` whose name starts with
    /// `prefix` and ends with `.ktstr.json`. Returns paths in
    /// filesystem iteration order; non-UTF-8 filenames are skipped.
    ///
    /// Call sites that write a single sidecar take the first match
    /// via `.into_iter().next().expect(..)` (the variant-hash suffix
    /// is opaque to the test so prefix match is how the file is
    /// recovered); tests that assert on the number of gauntlet
    /// variants use `.len()`.
    ///
    /// **Prefer this over hand-rolling read_dir/filter_map in new
    /// write_sidecar tests** — the 7 pre-existing call sites were
    /// near-identical inline blocks; funneling new tests through
    /// this helper keeps the lookup contract in one place.
    ///
    /// The `.ktstr.json` suffix filter is an intentional tightening
    /// relative to two of the original inline patterns
    /// (`write_sidecar_variant_hash_distinguishes_active_flags` and
    /// `_work_types`), which filtered only by prefix. The write-side
    /// tests only ever produce `.ktstr.json` files in their temp
    /// dirs, so the tightening is safe and rules out future stray
    /// files (a `.json.tmp` atomic-write residue, for instance) from
    /// inflating the count assertions.
    fn find_sidecars_by_prefix(dir: &std::path::Path, prefix: &str) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir)
            .expect("sidecar dir must exist for lookup")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".ktstr.json"))
            })
            .collect()
    }

    /// Single-file variant of [`find_sidecars_by_prefix`] for tests
    /// that exercise one variant per run. Asserts exactly one match
    /// and returns the owned path.
    ///
    /// What the length assertion catches: a test producing MORE than
    /// one sidecar under the given prefix — typically a stray
    /// leftover from a prior run (if the temp-dir cleanup is stale),
    /// or a call-site bug that invokes the writer twice. A
    /// variant-hash collision on its own would overwrite the file
    /// in place (same hash → same filename → single file), so this
    /// assertion is NOT a collision detector; it's a
    /// "one-call-one-file" invariant for single-variant tests.
    /// Centralizes the pattern so the 5 single-variant writer tests
    /// share one length check + error message.
    fn find_single_sidecar_by_prefix(dir: &std::path::Path, prefix: &str) -> std::path::PathBuf {
        let paths = find_sidecars_by_prefix(dir, prefix);
        assert_eq!(
            paths.len(),
            1,
            "single-variant test must produce exactly one sidecar under \
             prefix {prefix:?}; got {paths:?}",
        );
        paths
            .into_iter()
            .next()
            .expect("length-1 vec yields Some on first next()")
    }

    // -- find_sidecars_by_prefix self-tests --
    //
    // Pin the helper's filter behavior so changes to its logic
    // surface as failures here rather than as behavior shifts in
    // call sites.

    /// The `.ktstr.json` suffix filter must exclude files that share
    /// the prefix but carry a different extension. Without the
    /// suffix check, an atomic-write residue (`.json.tmp`) or a
    /// non-ktstr `.json` written into the same directory would
    /// inflate the match count.
    #[test]
    fn find_sidecars_by_prefix_filters_suffix() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("foo-0001.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("foo-0002.ktstr.json.tmp"), b"{}").unwrap();
        std::fs::write(tmp.join("foo-0003.json"), b"{}").unwrap();
        std::fs::write(tmp.join("foo-0004.ktstr.txt"), b"{}").unwrap();
        let paths = find_sidecars_by_prefix(tmp, "foo-");
        assert_eq!(
            paths.len(),
            1,
            "only the .ktstr.json file must match, got {paths:?}",
        );
    }

    /// The prefix filter must reject filenames whose prefix does
    /// not match, so the count-based gauntlet-variant tests
    /// (`write_sidecar_variant_hash_distinguishes_*`) can coexist
    /// safely with sidecars from unrelated tests that happen to
    /// share a parent directory.
    #[test]
    fn find_sidecars_by_prefix_filters_prefix() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("foo-0001.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("bar-0002.ktstr.json"), b"{}").unwrap();
        std::fs::write(tmp.join("foobar-0003.ktstr.json"), b"{}").unwrap();
        let paths = find_sidecars_by_prefix(tmp, "foo-");
        assert_eq!(
            paths.len(),
            1,
            "only files starting with 'foo-' must match (not 'foobar-'), got {paths:?}",
        );
    }

    /// A directory that contains nothing matching the `prefix` +
    /// `.ktstr.json` contract must yield an empty `Vec`, not panic.
    /// Call sites that use `.into_iter().next().expect(..)` rely on
    /// this — an empty Vec lets them surface a descriptive "sidecar
    /// file ... should be written" error rather than an opaque
    /// helper-internal panic.
    #[test]
    fn find_sidecars_by_prefix_empty_when_no_match() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("bar-0001.ktstr.json"), b"{}").unwrap();
        let paths = find_sidecars_by_prefix(tmp, "foo-");
        assert!(
            paths.is_empty(),
            "no prefix match must yield empty Vec, got {paths:?}",
        );
    }

    // -- test_fixture self-tests --
    //
    // Guard the fixture's observable shape so call-site tests can rely
    // on these defaults without re-asserting them.

    /// Serializing the fixture and parsing the result back must
    /// succeed — proves every field is serde-compatible and no default
    /// produces a value that fails to round-trip (e.g. a NaN float or
    /// an invalid Option combination).
    #[test]
    fn test_fixture_round_trips_clean() {
        let sc = SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).expect("fixture must serialize");
        let _loaded: SidecarResult =
            serde_json::from_str(&json).expect("fixture JSON must parse back");
    }

    /// `passed=true, skipped=false` is the fixture's verdict default
    /// so tests that only care about the success path don't need to
    /// spell either field out. A silent flip of either bit would
    /// invert the meaning of every unmodified call-site test.
    #[test]
    fn test_fixture_is_pass_not_skip() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.passed, "fixture must default to passed=true");
        assert!(!sc.skipped, "fixture must default to skipped=false");
    }

    /// `host=None` is the fixture's host default so
    /// [`sidecar_variant_hash_excludes_host_context`] and every test
    /// that asserts the JSON does not carry a host key can rely on
    /// the default rather than spelling it out. Production writers
    /// populate host explicitly (see `write_sidecar` /
    /// `write_skip_sidecar`).
    #[test]
    fn test_fixture_host_is_none() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.host.is_none(), "fixture must default to host=None");
    }

    /// `payload=None, metrics=empty` is the fixture's default so
    /// tests that verify the serde always-emit contract
    /// (e.g. [`sidecar_payload_and_metrics_always_emit_when_empty`])
    /// can rely on these defaults rather than re-spelling them.
    #[test]
    fn test_fixture_payload_and_metrics_empty() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.payload.is_none(), "fixture must default to payload=None");
        assert!(
            sc.metrics.is_empty(),
            "fixture must default to metrics=empty"
        );
    }

    /// Summary guard on every empty-collection / None-Option /
    /// empty-String default. A silent flip of any of these defaults
    /// breaks every test that depends on "unset → serialized as
    /// null / []" via the symmetric always-emit contract — and
    /// there are many such tests across this file. One tripwire
    /// here catches the flip in one place rather than fanning out
    /// to per-default pins.
    ///
    /// Hash-participating string defaults (`test_name`,
    /// `topology`, `scheduler`, `work_type`) are intentionally NOT
    /// re-asserted here — their drift is caught by
    /// `test_fixture_variant_hash_is_stable` which pins the hash.
    #[test]
    fn test_fixture_all_collections_empty_by_default() {
        let sc = SidecarResult::test_fixture();
        assert!(sc.metrics.is_empty(), "metrics must default empty");
        assert!(
            sc.active_flags.is_empty(),
            "active_flags must default empty"
        );
        assert!(
            sc.stimulus_events.is_empty(),
            "stimulus_events must default empty"
        );
        assert!(
            sc.verifier_stats.is_empty(),
            "verifier_stats must default empty"
        );
        assert!(sc.sysctls.is_empty(), "sysctls must default empty");
        assert!(sc.kargs.is_empty(), "kargs must default empty");
        assert!(sc.payload.is_none(), "payload must default None");
        assert!(sc.monitor.is_none(), "monitor must default None");
        assert!(sc.kvm_stats.is_none(), "kvm_stats must default None");
        assert!(
            sc.kernel_version.is_none(),
            "kernel_version must default None"
        );
        assert!(sc.host.is_none(), "host must default None");
        assert!(
            sc.timestamp.is_empty(),
            "timestamp must default empty String"
        );
        assert!(sc.run_id.is_empty(), "run_id must default empty String");
        assert!(
            sc.stats.cgroups.is_empty(),
            "stats.cgroups must default empty (ScenarioStats::default)",
        );
        // Overlaps deliberately with `test_fixture_is_pass_not_skip`
        // so this single summary test is sufficient to catch a
        // verdict-default flip even if callers forget the other
        // self-test exists. Cheap belt + suspenders.
        assert!(sc.passed, "passed must default true");
        assert!(!sc.skipped, "skipped must default false");
    }

    /// Two fresh fixtures must hash to the same value and that value
    /// must match the pinned constant. Protects against a change to
    /// fixture defaults that would silently shift every call-site
    /// test that passes the fixture straight into
    /// [`sidecar_variant_hash`] (e.g. `sidecar_variant_hash_distinguishes_payload`'s
    /// `none` handle). If this constant needs to move, every such
    /// call site must be re-read to confirm the shift is intentional.
    #[test]
    fn test_fixture_variant_hash_is_stable() {
        let a = sidecar_variant_hash(&SidecarResult::test_fixture());
        let b = sidecar_variant_hash(&SidecarResult::test_fixture());
        assert_eq!(a, b, "two fresh fixtures must hash identically");
        assert_eq!(
            a, 0x55f6b9881e152f8c,
            "fixture hash drifted — update only if the fixture default \
             change is intentional; verify every call site that passes \
             the fixture straight into sidecar_variant_hash still expresses \
             the intent it had before",
        );
    }

    /// Full literal intentional: exercises every field through serde so
    /// a future addition is caught by a compile error here.
    #[test]
    fn sidecar_result_roundtrip() {
        let sc = SidecarResult {
            test_name: "my_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            scheduler_commit: Some("abc123".to_string()),
            payload: None,
            metrics: vec![],
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
            host: None,
        };
        let json = serde_json::to_string_pretty(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        // Exhaustive destructure — `SidecarResult` is `non_exhaustive`
        // only across crates, but in-crate destructure still requires
        // every field to appear by name. Adding a field to
        // `SidecarResult` without extending this pattern fails to
        // compile here, forcing the author to make an explicit
        // roundtrip-coverage decision at the same time they introduce
        // the field. See sibling
        // [`sidecar_payload_and_metrics_always_emit_when_empty`] for
        // the empty-collection variant of this pin.
        let SidecarResult {
            test_name,
            topology,
            scheduler,
            scheduler_commit,
            payload,
            metrics,
            passed,
            skipped,
            stats,
            monitor,
            stimulus_events,
            work_type,
            active_flags,
            verifier_stats,
            kvm_stats,
            sysctls,
            kargs,
            kernel_version,
            timestamp,
            run_id,
            host,
        } = loaded;
        // Hash-participating string fields round-trip verbatim.
        assert_eq!(test_name, "my_test");
        assert_eq!(topology, "1n2l4c2t");
        assert_eq!(scheduler, "scx_mitosis");
        assert_eq!(work_type, "CpuSpin");
        // Nullable string metadata fields.
        assert_eq!(scheduler_commit.as_deref(), Some("abc123"));
        assert_eq!(payload, None, "fixture declared no payload");
        assert_eq!(kvm_stats, None, "fixture declared no kvm_stats");
        assert_eq!(kernel_version, None, "fixture declared no kernel_version");
        assert_eq!(host, None, "fixture declared no host context");
        assert_eq!(timestamp, "", "fixture used empty-string timestamp");
        assert_eq!(run_id, "", "fixture used empty-string run_id");
        // Verdict bits — passed true + skipped false pinned.
        assert!(passed);
        assert!(!skipped, "fixture declared skipped=false");
        // Empty-Vec collections — regression guard against a serde
        // regression that dropped `[]` on round-trip.
        assert!(metrics.is_empty(), "fixture declared empty metrics");
        assert!(
            active_flags.is_empty(),
            "fixture declared empty active_flags",
        );
        assert!(
            verifier_stats.is_empty(),
            "fixture declared empty verifier_stats",
        );
        assert!(sysctls.is_empty(), "fixture declared empty sysctls");
        assert!(kargs.is_empty(), "fixture declared empty kargs");
        // Populated nested structs.
        assert_eq!(stats.total_workers, 4);
        assert_eq!(stats.cgroups.len(), 1);
        assert_eq!(stats.cgroups[0].num_workers, 4);
        assert_eq!(stats.worst_spread, 20.0);
        let mon = monitor.unwrap();
        assert_eq!(mon.total_samples, 10);
        assert_eq!(mon.max_imbalance_ratio, 1.5);
        assert_eq!(mon.max_local_dsq_depth, 3);
        assert!(!mon.stall_detected);
        let deltas = mon.event_deltas.unwrap();
        assert_eq!(deltas.total_fallback, 7);
        assert_eq!(deltas.total_dispatch_keep_last, 3);
        assert_eq!(stimulus_events.len(), 1);
        assert_eq!(stimulus_events[0].label, "StepStart[0]");
    }

    /// Exhaustive schema-audit gate for `SidecarResult`'s serde
    /// round-trip. Every field is populated with a value that is
    /// distinct from the `test_fixture` default AND every field is
    /// asserted individually after serialization + deserialization.
    /// A new field added to `SidecarResult` triggers failure at two
    /// independent sites for `SidecarResult` top-level fields; nested
    /// structs use `..Default::default()` and rely on their own
    /// per-type tests:
    /// 1. The construction literal below fails to compile (Rust
    ///    requires every field in a struct literal without
    ///    `..Default::default()`).
    /// 2. The per-field assertion block below misses the new field,
    ///    so the audit surfaces as a reviewer note.
    ///
    /// Nested struct literals inside the construction (e.g.
    /// `MonitorSummary`, `ScenarioStats`, `HostContext`,
    /// `PayloadMetrics`) use `..Default::default()` to remain
    /// resilient to unrelated nested-type growth — adding a field
    /// to one of those nested types does NOT trip this test. Fields
    /// of those nested types that should trigger a similar audit
    /// must grow their own all-fields round-trip test in their
    /// owning module (e.g.
    /// `host_context_populated_round_trips_via_json` for
    /// `HostContext`).
    ///
    /// Complements the structurally-populated
    /// [`sidecar_result_roundtrip`] which exercises nested-struct
    /// shapes but only asserts on a subset of fields. Leaving both
    /// is intentional: the structural test proves deep trees survive
    /// serde; this test proves every scalar and Option round-trips.
    ///
    /// Distinct non-default values used:
    /// - `test_name="audit"` (vs fixture `"t"`).
    /// - `topology="8n8l16c2t"` (vs fixture `"1n1l1c1t"`).
    /// - `scheduler="scx_audit"` (vs fixture `"eevdf"`).
    /// - `work_type="AuditWork"` (vs fixture `"CpuSpin"`).
    /// - `passed=false, skipped=true` (vs fixture `true`, `false`).
    /// - Non-empty collections for every `Vec<_>` field.
    /// - `Some(…)` for every `Option<_>` field.
    /// - Non-empty Strings for `timestamp`, `run_id`.
    #[test]
    fn sidecar_result_roundtrip_all_fields_round_trip() {
        use crate::assert::{CgroupStats, ScenarioStats};
        use crate::host_context::HostContext;
        use crate::monitor::MonitorSummary;
        use crate::monitor::bpf_prog::ProgVerifierStats;
        use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};
        use crate::timeline::StimulusEvent;

        let sc = SidecarResult {
            test_name: "audit".to_string(),
            topology: "8n8l16c2t".to_string(),
            scheduler: "scx_audit".to_string(),
            scheduler_commit: Some("deadbeef1234567890abcdef".to_string()),
            payload: Some("audit_payload".to_string()),
            metrics: vec![PayloadMetrics {
                payload_index: 0,
                metrics: vec![Metric {
                    name: "audit_metric".to_string(),
                    value: 42.0,
                    polarity: Polarity::HigherBetter,
                    unit: "audits".to_string(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
                }],
                exit_code: 7,
            }],
            passed: false,
            skipped: true,
            stats: ScenarioStats {
                cgroups: vec![CgroupStats {
                    num_workers: 3,
                    ..Default::default()
                }],
                total_workers: 3,
                ..Default::default()
            },
            monitor: Some(MonitorSummary {
                total_samples: 17,
                ..Default::default()
            }),
            stimulus_events: vec![StimulusEvent {
                elapsed_ms: 123,
                label: "audit_event".to_string(),
                op_kind: None,
                detail: None,
                total_iterations: None,
            }],
            work_type: "AuditWork".to_string(),
            active_flags: vec!["flag_a".to_string(), "flag_b".to_string()],
            verifier_stats: vec![ProgVerifierStats {
                name: "audit_prog".to_string(),
                verified_insns: 999,
            }],
            kvm_stats: Some(crate::vmm::KvmStatsTotals::default()),
            sysctls: vec!["sysctl.kernel.audit_sysctl=1".to_string()],
            kargs: vec!["audit_karg".to_string()],
            kernel_version: Some("6.99.0".to_string()),
            timestamp: "audit-timestamp".to_string(),
            run_id: "audit-run-id".to_string(),
            host: Some(HostContext {
                kernel_name: Some("AuditLinux".to_string()),
                ..Default::default()
            }),
        };

        let json = serde_json::to_string(&sc).expect("serialize");
        let loaded: SidecarResult = serde_json::from_str(&json).expect("deserialize");

        // Every field asserted, in struct-declaration order.
        assert_eq!(loaded.test_name, "audit");
        assert_eq!(loaded.topology, "8n8l16c2t");
        assert_eq!(loaded.scheduler, "scx_audit");
        assert_eq!(
            loaded.scheduler_commit.as_deref(),
            Some("deadbeef1234567890abcdef"),
            "scheduler_commit must round-trip the literal string \
             populated on the write side — not collapse to None via \
             a missing serde attribute or default fallback",
        );
        assert_eq!(loaded.payload.as_deref(), Some("audit_payload"));
        assert_eq!(loaded.metrics.len(), 1);
        assert_eq!(loaded.metrics[0].exit_code, 7);
        assert_eq!(loaded.metrics[0].metrics.len(), 1);
        assert_eq!(loaded.metrics[0].metrics[0].name, "audit_metric");
        assert_eq!(loaded.metrics[0].metrics[0].value, 42.0);
        assert!(!loaded.passed, "passed must survive as false");
        assert!(loaded.skipped, "skipped must survive as true");
        assert_eq!(loaded.stats.total_workers, 3);
        assert_eq!(loaded.stats.cgroups.len(), 1);
        assert_eq!(loaded.stats.cgroups[0].num_workers, 3);
        let mon = loaded.monitor.expect("monitor round-trips");
        assert_eq!(mon.total_samples, 17);
        assert_eq!(loaded.stimulus_events.len(), 1);
        assert_eq!(loaded.stimulus_events[0].label, "audit_event");
        assert_eq!(loaded.stimulus_events[0].elapsed_ms, 123);
        assert_eq!(loaded.work_type, "AuditWork");
        assert_eq!(loaded.active_flags, vec!["flag_a", "flag_b"]);
        assert_eq!(loaded.verifier_stats.len(), 1);
        assert_eq!(loaded.verifier_stats[0].name, "audit_prog");
        assert_eq!(loaded.verifier_stats[0].verified_insns, 999);
        assert!(
            loaded.kvm_stats.is_some(),
            "kvm_stats must round-trip as Some"
        );
        assert_eq!(loaded.sysctls, vec!["sysctl.kernel.audit_sysctl=1"]);
        assert_eq!(loaded.kargs, vec!["audit_karg"]);
        assert_eq!(loaded.kernel_version.as_deref(), Some("6.99.0"));
        assert_eq!(loaded.timestamp, "audit-timestamp");
        assert_eq!(loaded.run_id, "audit-run-id");
        let host = loaded.host.expect("host round-trips");
        assert_eq!(host.kernel_name.as_deref(), Some("AuditLinux"));
    }

    #[test]
    fn sidecar_result_roundtrip_no_monitor() {
        let sc = SidecarResult {
            test_name: "eevdf_test".to_string(),
            topology: "1n1l2c1t".to_string(),
            passed: false,
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.test_name, "eevdf_test");
        assert!(!loaded.passed);
        assert!(loaded.monitor.is_none());
        assert!(loaded.stimulus_events.is_empty());
        // `monitor` is emitted as `"monitor":null` when absent — the
        // sidecar schema is symmetric, with every `Option` field always
        // present on the wire and required on deserialize. Pinning the
        // emission pattern prevents a drift back to the old asymmetric
        // `skip_serializing_if` form that failed deserialize on a None-
        // produced sidecar.
        assert!(
            json.contains("\"monitor\":null"),
            "monitor=None must serialize as `\"monitor\":null`, not be omitted: {json}",
        );
    }

    /// Strict-schema rejection: a sidecar JSON that omits a required
    /// top-level field (here: `test_name`) must fail deserialization,
    /// not silently default to the empty string. The SidecarResult
    /// policy — serde(default) removed crate-wide so serialize and
    /// deserialize are symmetric — is stated in the module doc and
    /// on the `host` field; this test pins the policy by
    /// construction. A regression that reintroduces `#[serde(default)]`
    /// on any top-level SidecarResult field would cause the
    /// `from_str` below to succeed instead of error.
    ///
    /// `test_name` is the chosen field because it is a plain String
    /// and its absence produces a clean "missing field" error from
    /// serde without sibling-field interference. Other top-level
    /// fields (Vec, Option, nested struct) follow the same contract;
    /// picking one is sufficient to guard the policy.
    #[test]
    fn sidecar_result_missing_required_field_rejected_by_deserialize() {
        // Table-driven expansion covering every non-`Option` field of
        // `SidecarResult`. Each must fail deserialize when absent with
        // a missing-field error naming the removed key.
        //
        // **Why Option fields are excluded**: serde treats
        // `Option<T>` as tolerant-of-absence natively (no explicit
        // `#[serde(default)]` needed — it's a builtin rule), so
        // removing e.g. `payload: Option<String>` from the JSON
        // yields `None` on the parsed struct rather than a rejection.
        // The module doc at src/test_support/sidecar.rs promises
        // "required on deserialize" for Option fields, but that's
        // enforced at the writer (always-emitted) side, not the
        // parser side. The `serialize_always_emits_option_keys`
        // sibling tests pin the writer half; this test pins the
        // parser-side strictness for every non-Option field.
        //
        // Old single-field-sentinel form (checking only `test_name`)
        // would pass silently if e.g. a regression added
        // `#[serde(default)]` to `run_id` alone — this loop catches
        // that class of softening across every non-Option field.
        const REQUIRED_NON_OPTION_FIELDS: &[&str] = &[
            "test_name",
            "topology",
            "scheduler",
            "metrics",
            "passed",
            "skipped",
            "stats",
            "stimulus_events",
            "work_type",
            "active_flags",
            "verifier_stats",
            "sysctls",
            "kargs",
            "timestamp",
            "run_id",
        ];

        let fixture = SidecarResult::test_fixture();
        let full = match serde_json::to_value(&fixture).unwrap() {
            serde_json::Value::Object(m) => m,
            other => panic!("expected object, got {other:?}"),
        };

        for field in REQUIRED_NON_OPTION_FIELDS {
            let mut obj = full.clone();
            assert!(
                obj.remove(*field).is_some(),
                "SidecarResult test fixture must emit `{field}` for its \
                 rejection case to be meaningful — the required-fields \
                 list has drifted from the struct definition",
            );
            let json = serde_json::Value::Object(obj).to_string();
            let err = serde_json::from_str::<SidecarResult>(&json)
                .err()
                .unwrap_or_else(|| {
                    panic!(
                        "deserialize must reject SidecarResult with `{field}` removed, \
                     but succeeded — a regression may have added \
                     `#[serde(default)]` to this field",
                    )
                });
            let msg = format!("{err}");
            assert!(
                msg.contains(field),
                "missing-field error for `{field}` must name the field; got: {msg}",
            );
        }
    }

    // -- collect_sidecars tests --

    #[test]
    fn collect_sidecars_empty_dir() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let results = collect_sidecars(tmp_dir.path());
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_nonexistent_dir() {
        let results = collect_sidecars(std::path::Path::new("/nonexistent/path"));
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_reads_json() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let sc = SidecarResult {
            test_name: "test_x".to_string(),
            topology: "1n1l2c1t".to_string(),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(tmp.join("test_x.ktstr.json"), &json).unwrap();
        // Non-ktstr JSON should be ignored.
        std::fs::write(tmp.join("other.json"), r#"{"key":"val"}"#).unwrap();
        let results = collect_sidecars(tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "test_x");
    }

    #[test]
    fn collect_sidecars_recurses_one_level() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let sub = tmp.join("job-0");
        std::fs::create_dir_all(&sub).unwrap();
        let sc = SidecarResult {
            test_name: "nested_test".to_string(),
            topology: "1n2l4c2t".to_string(),
            scheduler: "scx_mitosis".to_string(),
            passed: false,
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        std::fs::write(sub.join("nested_test.ktstr.json"), &json).unwrap();
        let results = collect_sidecars(tmp);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].test_name, "nested_test");
        assert!(!results[0].passed);
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
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let top_sub = tmp.join("job-0");
        let deep_sub = top_sub.join("replay-0");
        std::fs::create_dir_all(&deep_sub).unwrap();

        let sc = |name: &str| SidecarResult {
            test_name: name.to_string(),
            ..SidecarResult::test_fixture()
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

        let results = collect_sidecars(tmp);
        let names: Vec<&str> = results.iter().map(|r| r.test_name.as_str()).collect();
        assert_eq!(
            names,
            vec!["top_level"],
            "collect_sidecars must see only the one-level-deep sidecar, not the two-level one"
        );
    }

    #[test]
    fn collect_sidecars_skips_invalid_json() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        std::fs::write(tmp.join("bad.ktstr.json"), "not json").unwrap();
        let results = collect_sidecars(tmp);
        assert!(results.is_empty());
    }

    #[test]
    fn collect_sidecars_skips_non_ktstr_json() {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        // File ends in .json but does NOT contain ".ktstr." in the name
        std::fs::write(tmp.join("other.json"), r#"{"test":"val"}"#).unwrap();
        let results = collect_sidecars(tmp);
        assert!(results.is_empty());
    }

    #[test]
    fn sidecar_result_work_type_field() {
        let sc = SidecarResult {
            work_type: "Bursty".to_string(),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.work_type, "Bursty");
    }

    #[test]
    fn write_sidecar_defaults_to_target_dir_without_env() {
        let _lock = lock_env();
        let _env_sidecar = EnvVarGuard::remove("KTSTR_SIDECAR_DIR");
        let _env_kernel = EnvVarGuard::remove("KTSTR_KERNEL");
        let _env_target = EnvVarGuard::remove("CARGO_TARGET_DIR");

        let dir = sidecar_dir();
        let expected = format!("target/ktstr/unknown-{}", run_id_timestamp());
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
        let vm_result = crate::vmm::VmResult::test_fixture();
        let check_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &check_result, "CpuSpin", &[], &[]).unwrap();

        // Clean up written files. The actual on-disk filename
        // embeds a variant-hash suffix (see
        // `serialize_and_write_sidecar`), so a fixed `test_name +
        // ".ktstr.json"` path never matches — use the
        // prefix-scan helper the sibling tests use. The parent
        // `dir` itself is shared with any other test that runs
        // without `KTSTR_SIDECAR_DIR` set, so leave it in place;
        // only this test's own files are removed.
        let paths = find_sidecars_by_prefix(&dir, "__sidecar_default_dir__-");
        // One call to `write_sidecar` above must produce exactly
        // one sidecar under this test's unique prefix. A count
        // above 1 exposes a variant-hash collision (two distinct
        // test_name + variant-hash pairs hashing to the same
        // filename suffix) or a stale file lingering from a
        // previous crashed run sharing this exact test_name — the
        // latter would hide a real collision today. Making the
        // check loud here (rather than silently wiping every
        // matching file) surfaces both regressions.
        assert_eq!(
            paths.len(),
            1,
            "single `write_sidecar` call against prefix \
             `__sidecar_default_dir__-` must produce exactly one \
             file; got {} ({paths:?}). If >1, either the variant \
             hash collided for this test's variant-field tuple or \
             a prior crashed run left a stale sidecar under the \
             same prefix — investigate before re-running the test.",
            paths.len(),
        );
        for p in paths {
            let _ = std::fs::remove_file(&p);
        }
    }

    #[test]
    fn write_sidecar_writes_file() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__sidecar_write_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let check_result = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &check_result, "CpuSpin", &[], &[]).unwrap();

        // Sidecar filename now includes a variant hash suffix so
        // gauntlet variants don't clobber each other. Use the
        // single-match helper, which also guards against stray
        // leftover files from prior runs or double-writer bugs.
        let path = find_single_sidecar_by_prefix(tmp, "__sidecar_write_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.test_name, "__sidecar_write_test__");
        assert!(loaded.passed);
        assert!(!loaded.skipped, "pass result is not a skip");
        // write_sidecar must populate the host-context snapshot so
        // downstream `stats compare --runs a b` can diff hosts.
        // Without this assertion, a regression that dropped the
        // `host: Some(collect_host_context())` builder line would
        // land silently. `kernel_name` is always `Some("Linux")`
        // on a running Linux process (uname syscall, no filesystem
        // dependency), matching the baseline asserted by
        // `host_context::tests::collect_host_context_returns_populated_struct_on_linux`.
        let host = loaded
            .host
            .as_ref()
            .expect("write_sidecar must populate host field from collect_host_context");
        assert_eq!(host.kernel_name.as_deref(), Some("Linux"));
        // Pair the uname check with a field that `HostContext::default()`
        // leaves None. A regression that swapped the full
        // `collect_host_context()` call for `HostContext { kernel_name:
        // Some("Linux".into()), ..Default::default() }` would pass the
        // uname assertion but drop every other captured field —
        // `kernel_cmdline` is present on every live Linux process
        // (/proc/cmdline is always readable; see host_context::tests:
        // collect_host_context_captures_cmdline_on_linux) so
        // `kernel_cmdline.is_some()` catches the default-substitution
        // regression.
        assert!(
            host.kernel_cmdline.is_some(),
            "write_sidecar must capture full HostContext, not Default::default() — \
             /proc/cmdline is always readable on Linux (see host_context tests)",
        );
        // Second Default-distinguishing field: `kernel_release` is
        // populated by the uname() syscall on any live Linux host
        // (filesystem-independent — no /proc/sys dependency), so a
        // `None` here would indicate the default-substitution
        // regression reached the uname path. Pairing cmdline
        // (filesystem-sourced) with kernel_release (syscall-sourced)
        // gives two independent capture paths, so a regression that
        // broke only one collection site is still caught.
        assert!(
            host.kernel_release.is_some(),
            "write_sidecar must capture kernel_release — uname() is \
             filesystem-independent; a None here means the default \
             substitution bypassed the full collect_host_context()",
        );
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_active_flags() {
        // Two gauntlet variants differing ONLY in active_flags must
        // produce distinct sidecar filenames so neither clobbers the
        // other. A hash of work_type/sysctls/kargs alone would miss
        // this difference.
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__flagvariant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        let flags_a = vec!["llc".to_string()];
        let flags_b = vec!["llc".to_string(), "steal".to_string()];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_a, &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &flags_b, &[]).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__flagvariant_test__-");
        assert_eq!(
            paths.len(),
            2,
            "two active_flags variants must produce two distinct files, got {paths:?}"
        );
    }

    /// Two `write_sidecar` calls differing ONLY in the ORDER their
    /// caller accumulated `active_flags` — same semantic variant,
    /// same flag SET — must produce identical sidecar filenames.
    /// Filenames are keyed on [`sidecar_variant_hash`], which walks
    /// `active_flags` in-order and folds each byte into the hash
    /// state. Without canonicalization at the write site, a caller
    /// that happened to collect `["steal", "llc"]` would hash to
    /// a different bucket than one that collected `["llc",
    /// "steal"]` for the same run — `stats compare` would then see
    /// two rows for one semantic variant and mark one as "new" or
    /// "removed" on a re-run that only changed flag accumulation
    /// order.
    ///
    /// This test pins the canonicalization done by
    /// `canonicalize_active_flags` (applied in both
    /// `write_sidecar` and `write_skip_sidecar`): two writes with
    /// reversed flag order collapse to a single file via normal
    /// overwrite. A regression that dropped the sort (reverting to
    /// `active_flags.to_vec()`) would make the second write land
    /// at a different hash → two files, caught here. Pair with
    /// `write_sidecar_variant_hash_distinguishes_active_flags`
    /// above, which pins the complementary property: different
    /// flag SETS must still hash distinctly.
    #[test]
    fn write_sidecar_variant_hash_is_order_invariant_for_active_flags() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__flagorder_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        // Same set of flags in reversed accumulation order. `llc` is
        // `ALL_DECLS[0].name` and `steal` is `ALL_DECLS[2].name`, so
        // the canonical order is ["llc","steal"] regardless of
        // which order the caller supplied them.
        let forward = vec!["llc".to_string(), "steal".to_string()];
        let reversed = vec!["steal".to_string(), "llc".to_string()];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &forward, &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &reversed, &[]).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__flagorder_test__-");
        assert_eq!(
            paths.len(),
            1,
            "reversed-order writes of the same flag SET must \
             collapse to a single canonical sidecar filename \
             (overwrite); got {paths:?}. If this fails with \
             `paths.len() == 2`, the write path has regressed to \
             hashing caller-order flags — re-sort via \
             `canonicalize_active_flags` in both write_sidecar \
             and write_skip_sidecar.",
        );

        // Defensive: the single surviving file must carry the
        // canonical order on disk, not whichever order the last
        // caller passed. Deserialize and check.
        let path = &paths[0];
        let data = std::fs::read_to_string(path).expect("read canonical sidecar");
        let loaded: SidecarResult =
            serde_json::from_str(&data).expect("deserialize canonical sidecar");
        assert_eq!(
            loaded.active_flags,
            vec!["llc".to_string(), "steal".to_string()],
            "on-disk active_flags must be sorted in \
             `scenario::flags::ALL` positional order; got: {:?}",
            loaded.active_flags,
        );
    }

    /// `sidecar_variant_hash` is order-insensitive for `sysctls`
    /// and `kargs` — same contract as `active_flags`, but
    /// canonicalized at hash time (local sort inside
    /// `sidecar_variant_hash`) rather than at write time. Pinning
    /// the invariant directly against the hash function catches a
    /// regression that drops the sort block (reverts to iterating
    /// `&sidecar.sysctls` / `&sidecar.kargs` in-order) even if all
    /// existing stability pins continue to pass — those pins use
    /// single-element collections where sorting is a no-op, so
    /// they cannot detect this regression by themselves.
    ///
    /// Calls the hash function directly rather than going through
    /// `write_sidecar` because the sysctls/kargs come from
    /// `entry.scheduler.sysctls()` / `kargs()` — static slices the
    /// caller cannot reorder. The only path for a reordered input
    /// is a direct `SidecarResult` construction with reordered
    /// fields, which this test exercises.
    #[test]
    fn sidecar_variant_hash_is_order_invariant_for_sysctls_and_kargs() {
        let forward = SidecarResult {
            sysctls: vec![
                "sysctl.a=1".to_string(),
                "sysctl.b=2".to_string(),
                "sysctl.c=3".to_string(),
            ],
            kargs: vec![
                "karg_alpha".to_string(),
                "karg_beta".to_string(),
                "karg_gamma".to_string(),
            ],
            ..SidecarResult::test_fixture()
        };
        let reversed = SidecarResult {
            sysctls: vec![
                "sysctl.c=3".to_string(),
                "sysctl.b=2".to_string(),
                "sysctl.a=1".to_string(),
            ],
            kargs: vec![
                "karg_gamma".to_string(),
                "karg_beta".to_string(),
                "karg_alpha".to_string(),
            ],
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&forward),
            sidecar_variant_hash(&reversed),
            "reversed-order sysctls/kargs must hash identically — \
             the hash sorts both collections lexically before \
             folding bytes in, matching the set-determines-hash \
             contract documented on `sidecar_variant_hash`. A \
             regression that dropped the sort block would produce \
             distinct hashes and duplicate sidecar files for the \
             same semantic variant.",
        );

        // Permutation check: a partial reorder (sysctls same,
        // kargs reversed) must also collapse. Guards against a
        // partial revert that drops the sort in only one of the
        // two collections.
        let partial = SidecarResult {
            sysctls: forward.sysctls.clone(),
            kargs: reversed.kargs.clone(),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&forward),
            sidecar_variant_hash(&partial),
            "kargs-only reversal must still hash identically — \
             partial revert (one of the two sorts dropped) must \
             fail this assertion. Got distinct hashes for: \
             sysctls={:?}, kargs={:?} vs sysctls={:?}, kargs={:?}",
            forward.sysctls,
            forward.kargs,
            partial.sysctls,
            partial.kargs,
        );
    }

    /// `write_skip_sidecar` sibling of
    /// `write_sidecar_variant_hash_is_order_invariant_for_active_flags`.
    /// The canonicalization path is applied at BOTH write sites
    /// (`write_sidecar` for run-to-completion results,
    /// `write_skip_sidecar` for pre-VM-boot skips), so both need
    /// order-invariance coverage — a partial revert that dropped
    /// `canonicalize_active_flags` in just the skip path would
    /// leave the run path covered by the sibling test yet leave
    /// skip-variant hashes order-sensitive, producing duplicate
    /// skip-sidecar files for the same semantic variant under
    /// `stats list` / `stats compare`.
    ///
    /// Pins the same two invariants as the sibling: (1) reversed
    /// flag-order inputs collapse to a single file via normal
    /// overwrite, (2) the surviving on-disk `active_flags` is in
    /// canonical `scenario::flags::ALL` order. Uses a distinct
    /// entry-name prefix (`__skipflagorder_test__`) so the
    /// `find_sidecars_by_prefix` scan doesn't overlap with the
    /// run-path test's fixtures.
    #[test]
    fn write_skip_sidecar_variant_hash_is_order_invariant_for_active_flags() {
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skipflagorder_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };

        // Same flag SET, reversed accumulation order. Mirrors the
        // `llc` / `steal` choice from the run-path sibling so the
        // canonical order (index 0, index 2 in `ALL_DECLS`) is
        // unambiguous.
        let forward = vec!["llc".to_string(), "steal".to_string()];
        let reversed = vec!["steal".to_string(), "llc".to_string()];
        write_skip_sidecar(&entry, &forward).unwrap();
        write_skip_sidecar(&entry, &reversed).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__skipflagorder_test__-");
        assert_eq!(
            paths.len(),
            1,
            "reversed-order skip-sidecar writes of the same flag \
             SET must collapse to a single canonical filename \
             (overwrite); got {paths:?}. If this fails with \
             `paths.len() == 2`, canonicalization was removed from \
             `write_skip_sidecar` even if the run-path test above \
             still passes — apply `canonicalize_active_flags` in \
             both write sites, not just one.",
        );

        let path = &paths[0];
        let data = std::fs::read_to_string(path).expect("read canonical skip sidecar");
        let loaded: SidecarResult =
            serde_json::from_str(&data).expect("deserialize canonical skip sidecar");
        assert_eq!(
            loaded.active_flags,
            vec!["llc".to_string(), "steal".to_string()],
            "on-disk active_flags of a skip sidecar must be sorted \
             in `scenario::flags::ALL` positional order; got: {:?}",
            loaded.active_flags,
        );
    }

    /// Directly exercises `canonicalize_active_flags` on a mixed
    /// input: known canonical flags AND ad-hoc unknown flags. The
    /// sibling
    /// `write_sidecar_variant_hash_is_order_invariant_for_active_flags`
    /// test pins the known-flag-only case through the full
    /// write-and-read round trip; this unit-level test pins the
    /// composite sort-key contract in isolation so a regression in
    /// the tiebreaker (e.g. dropping the secondary lexical
    /// comparator, reverting to `sort_by_key` with a bare
    /// positional key) fails here with a precise diagnostic,
    /// rather than going undetected until a user trips it with
    /// ad-hoc flags.
    ///
    /// Invariants pinned:
    /// 1. Known flags (members of `scenario::flags::ALL`) always
    ///    appear before unknown flags, regardless of input order.
    /// 2. Known flags are ordered by their position in ALL
    ///    (positional key as primary sort).
    /// 3. Unknown flags are ordered lexically among themselves
    ///    (secondary `&str` comparator). Without the secondary,
    ///    two unknown flags share `usize::MAX` as their positional
    ///    key and stable-sort preserves input order — so reversed
    ///    unknown-flag input would produce reversed output and
    ///    the variant hash would still depend on caller order.
    #[test]
    fn canonicalize_active_flags_orders_unknown_lexically_after_known() {
        // `llc` is `ALL[0]`, so it always wins against unknown
        // flags on the positional key. The two `*_unknown` flags
        // collide at `usize::MAX` and must then be ordered
        // lexically (`aaa_` < `zzz_`).
        let input = vec![
            "zzz_unknown".to_string(),
            "llc".to_string(),
            "aaa_unknown".to_string(),
        ];
        let got = canonicalize_active_flags(&input);
        assert_eq!(
            got,
            vec![
                "llc".to_string(),
                "aaa_unknown".to_string(),
                "zzz_unknown".to_string(),
            ],
            "known flags must sort first by ALL position, unknown \
             flags must sort lexically after; got: {got:?}",
        );

        // Invariance check: reversing the input must produce the
        // same output. Without the lexical secondary the two
        // unknowns would swap, breaking the set-determines-hash
        // property for any variant carrying ad-hoc flags.
        let reversed: Vec<String> = input.into_iter().rev().collect();
        let got_rev = canonicalize_active_flags(&reversed);
        assert_eq!(
            got_rev, got,
            "reversed input must canonicalize to the same output; \
             got: {got_rev:?}, expected: {got:?}",
        );
    }

    #[test]
    fn write_sidecar_variant_hash_distinguishes_work_types() {
        // Two gauntlet variants differing only in work_type must
        // produce distinct sidecar filenames so neither clobbers the
        // other.
        let _lock = lock_env();
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let tmp = tmp_dir.path();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__variant_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();
        write_sidecar(&entry, &vm_result, &[], &ok, "YieldHeavy", &[], &[]).unwrap();

        let paths = find_sidecars_by_prefix(tmp, "__variant_test__-");
        assert_eq!(
            paths.len(),
            2,
            "two work_type variants must produce two distinct files, got {paths:?}"
        );
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
    /// Every currently hash-participating field (topology, scheduler,
    /// payload, work_type, active_flags, sysctls, kargs) is set
    /// explicitly; non-participating fields come from
    /// [`SidecarResult::test_fixture`] so unrelated schema growth does
    /// not disturb the constant. If a future change adds a new
    /// hash-participating field to [`sidecar_variant_hash`], add it
    /// here too — otherwise this test silently degrades into a
    /// same-defaults check.
    #[test]
    fn sidecar_variant_hash_stability_populated() {
        // Every currently hash-participating field is spelled out
        // explicitly so a change to `test_fixture` defaults cannot
        // silently shift the pinned constant. If you add a new
        // hash-participating field to `sidecar_variant_hash`, add
        // it here and recompute the expected constant.
        let sc = SidecarResult {
            topology: "1n2l4c1t".to_string(),
            scheduler: "scx-ktstr".to_string(),
            payload: None,
            work_type: "CpuSpin".to_string(),
            active_flags: vec!["llc".to_string(), "steal".to_string()],
            sysctls: vec!["sysctl.kernel.sched_cfs_bandwidth_slice_us=1000".to_string()],
            kargs: vec!["nosmt".to_string()],
            ..SidecarResult::test_fixture()
        };
        // If this assertion trips, the wire format changed. Bumping
        // the expected value is the wrong fix unless you also plan
        // for old sidecars to be regenerated — see the contract on
        // `sidecar_variant_hash`.
        assert_eq!(
            sidecar_variant_hash(&sc),
            0xbc0f38005915a09f,
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
        // Every currently hash-participating field is spelled out
        // explicitly so a change to `test_fixture` defaults cannot
        // silently shift the pinned constant. If you add a new
        // hash-participating field to `sidecar_variant_hash`, add
        // it here and recompute the expected constant.
        let sc = SidecarResult {
            topology: "1n1l1c1t".to_string(),
            scheduler: "eevdf".to_string(),
            payload: None,
            work_type: String::new(),
            active_flags: Vec::new(),
            sysctls: Vec::new(),
            kargs: Vec::new(),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(sidecar_variant_hash(&sc), 0x1b61394511b42e01);
    }

    /// Two sidecars that differ only in `payload` must produce
    /// distinct variant hashes so gauntlet runs composing the same
    /// scheduler with different primary payloads (FIO vs STRESS_NG)
    /// don't clobber each other's files.
    #[test]
    fn sidecar_variant_hash_distinguishes_payload() {
        // `none` relies on [`SidecarResult::test_fixture`] defaulting
        // `payload` to `None`. If that default changes, the absent-vs-
        // present comparison below collapses — the assertion below
        // and this comment are intentionally load-bearing.
        let base = SidecarResult::test_fixture;
        let none = base();
        assert!(
            none.payload.is_none(),
            "fixture default for payload must remain None"
        );
        let fio = SidecarResult {
            payload: Some("fio".to_string()),
            ..base()
        };
        let stress = SidecarResult {
            payload: Some("stress-ng".to_string()),
            ..base()
        };
        let h_none = sidecar_variant_hash(&none);
        let h_fio = sidecar_variant_hash(&fio);
        let h_stress = sidecar_variant_hash(&stress);
        assert_ne!(
            h_none, h_fio,
            "absent vs present payload must hash differently",
        );
        assert_ne!(
            h_fio, h_stress,
            "different payload names must hash differently",
        );
    }

    // -- format_verifier_stats tests --

    #[test]
    fn format_verifier_stats_empty() {
        assert!(format_verifier_stats(&[]).is_empty());
    }

    #[test]
    fn format_verifier_stats_no_data() {
        let sc = SidecarResult::test_fixture();
        assert!(format_verifier_stats(&[sc]).is_empty());
    }

    #[test]
    fn format_verifier_stats_table() {
        let sc = SidecarResult {
            verifier_stats: vec![
                crate::monitor::bpf_prog::ProgVerifierStats {
                    name: "dispatch".to_string(),
                    verified_insns: 50000,
                },
                crate::monitor::bpf_prog::ProgVerifierStats {
                    name: "enqueue".to_string(),
                    verified_insns: 30000,
                },
            ],
            ..SidecarResult::test_fixture()
        };
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
        let sc = SidecarResult {
            verifier_stats: vec![crate::monitor::bpf_prog::ProgVerifierStats {
                name: "heavy".to_string(),
                verified_insns: 800000,
            }],
            ..SidecarResult::test_fixture()
        };
        let result = format_verifier_stats(&[sc]);
        assert!(result.contains("WARNING"));
        assert!(result.contains("heavy"));
        assert!(result.contains("80.0%"));
    }

    #[test]
    fn sidecar_verifier_stats_serde_roundtrip() {
        let sc = SidecarResult {
            verifier_stats: vec![crate::monitor::bpf_prog::ProgVerifierStats {
                name: "init".to_string(),
                verified_insns: 5000,
            }],
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("verifier_stats"));
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.verifier_stats.len(), 1);
        assert_eq!(loaded.verifier_stats[0].name, "init");
        assert_eq!(loaded.verifier_stats[0].verified_insns, 5000);
    }

    /// Every `Vec` field emits as `"x":[]` when empty rather than
    /// being omitted. Pin the always-emit contract so a regression
    /// that re-adds `skip_serializing_if` on `verifier_stats` is
    /// caught before it ships.
    #[test]
    fn sidecar_verifier_stats_empty_emits_as_empty_array() {
        let sc = SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).unwrap();
        assert!(
            json.contains("\"verifier_stats\":[]"),
            "empty verifier_stats must emit as `\"verifier_stats\":[]`: {json}",
        );
    }

    #[test]
    fn format_verifier_stats_deduplicates() {
        let vstats = vec![crate::monitor::bpf_prog::ProgVerifierStats {
            name: "dispatch".to_string(),
            verified_insns: 50000,
        }];
        let sc1 = SidecarResult {
            verifier_stats: vstats.clone(),
            ..SidecarResult::test_fixture()
        };
        let sc2 = SidecarResult {
            verifier_stats: vstats,
            ..SidecarResult::test_fixture()
        };
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
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: commit,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        assert_eq!(name, "eevdf");
        assert!(
            commit.is_none(),
            "Eevdf variant has no userspace binary; \
             scheduler_commit must be None. Got: {commit:?}",
        );
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
        static SCHED_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "s_test",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: _,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        assert_eq!(name, "s");
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
        static SCHED_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "s_test",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: _,
            scheduler_commit: _,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        assert_eq!(kargs, vec!["quiet".to_string(), "splash".to_string()]);
        assert!(sysctls.is_empty());
    }

    #[test]
    fn scheduler_fingerprint_uses_display_name_for_discover() {
        use super::super::entry::SchedulerSpec;
        static SCHED: super::super::entry::Scheduler =
            super::super::entry::Scheduler::new("s").binary(SchedulerSpec::Discover("scx_relaxed"));
        static SCHED_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::from_scheduler(&SCHED);
        let entry = KtstrTestEntry {
            name: "rel_test",
            scheduler: &SCHED_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: commit,
            sysctls: _,
            kargs: _,
        } = scheduler_fingerprint(&entry);
        assert_eq!(name, "s");
        assert!(
            commit.is_none(),
            "Discover variant currently returns None via \
             `SchedulerSpec::scheduler_commit` — \
             `resolve_scheduler`'s cascade does not guarantee a \
             fresh build, so there is no authoritative source for \
             the scheduler binary's commit and `scheduler_commit` \
             reports None honestly. Got: {commit:?}",
        );
    }

    /// `scheduler_fingerprint` on a binary-kind `Payload`
    /// (constructed via `Payload::binary`) must produce
    /// `commit: None`. The `and_then` chain in `scheduler_fingerprint`
    /// (`entry.scheduler.scheduler_binary().and_then(|s|
    /// s.scheduler_commit())`) relies on `Payload::scheduler_binary`
    /// returning `None` for `PayloadKind::Binary` to short-circuit
    /// the commit lookup — a regression that accidentally returned
    /// `Some(&some_default)` from `scheduler_binary` for
    /// binary-kind payloads would skip this short-circuit and
    /// populate `scheduler_commit` with a value that has nothing
    /// to do with a scheduler. This test pins that short-circuit
    /// end-to-end.
    ///
    /// Complements the `scheduler_commit_*` variant tests on
    /// `SchedulerSpec` itself (which cover the scheduler-kind
    /// branches) by exercising the binary-kind fallthrough that
    /// never touches `SchedulerSpec` at all.
    #[test]
    fn scheduler_fingerprint_binary_payload_has_no_commit() {
        static BINARY_PAYLOAD: super::super::payload::Payload =
            super::super::payload::Payload::binary("bin_test", "some_binary");
        let entry = KtstrTestEntry {
            name: "bin_test",
            scheduler: &BINARY_PAYLOAD,
            ..KtstrTestEntry::DEFAULT
        };
        let SchedulerFingerprint {
            scheduler: name,
            scheduler_commit: commit,
            sysctls,
            kargs,
        } = scheduler_fingerprint(&entry);
        // Per `Payload::scheduler_name`, binary-kind payloads
        // carry the intent-level label `"kernel_default"` — pinning
        // this alongside the None-commit keeps the binary-kind
        // contract visible in one place.
        assert_eq!(
            name, "kernel_default",
            "binary-kind payload must report the intent-level \
             scheduler label; got: {name:?}",
        );
        assert!(
            commit.is_none(),
            "binary-kind payload has no scheduler binary at all — \
             scheduler_commit must be None via the `and_then` \
             short-circuit on `scheduler_binary() == None`. Got: \
             {commit:?}",
        );
        assert!(
            sysctls.is_empty(),
            "binary-kind payload reports no sysctls; got: {sysctls:?}",
        );
        assert!(
            kargs.is_empty(),
            "binary-kind payload reports no kargs; got: {kargs:?}",
        );
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
        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-writes-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

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

        let path = find_single_sidecar_by_prefix(&tmp, "__skip_sidecar_test__-");
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
        // write_skip_sidecar shares the host-context capture with
        // write_sidecar (same `collect_host_context()` builder line)
        // so skip paths still give `stats compare --runs` a host
        // baseline. A regression that dropped the skip-path capture
        // would leave `host: None` in only the skip bucket, producing
        // silent per-run partial data.
        let host = loaded
            .host
            .as_ref()
            .expect("write_skip_sidecar must populate host field from collect_host_context");
        assert_eq!(host.kernel_name.as_deref(), Some("Linux"));
        // Pair the uname check with a Default-distinguishing field —
        // see `write_sidecar_writes_file` for the rationale. Keeps
        // both the happy-path writer and the skip-path writer guarded
        // against the same default-substitution regression.
        assert!(
            host.kernel_cmdline.is_some(),
            "write_skip_sidecar must capture full HostContext, not Default::default()",
        );
        // Syscall-sourced companion to the filesystem-sourced
        // `kernel_cmdline` check — see `write_sidecar_writes_file`
        // for the two-independent-paths rationale.
        assert!(
            host.kernel_release.is_some(),
            "write_skip_sidecar must capture kernel_release (syscall-sourced)",
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// When the sidecar directory cannot be created (path collision
    /// with a regular file), `write_skip_sidecar` must return `Err`
    /// rather than silently eating the failure. Stats tooling relies
    /// on the error chain to diagnose missing sidecars; a swallowed
    /// error would make skips invisible to post-run analysis.
    #[test]
    fn write_skip_sidecar_returns_err_when_dir_cannot_be_created() {
        let _lock = lock_env();

        // Create a regular file, then try to use it as the sidecar
        // directory. `create_dir_all` fails because the path exists
        // but is not a directory.
        let blocker = std::env::temp_dir().join("ktstr-sidecar-skip-blocker");
        let _ = std::fs::remove_file(&blocker);
        let _ = std::fs::remove_dir_all(&blocker);
        std::fs::write(&blocker, b"not a dir").unwrap();
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &blocker);

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
    }

    // -- sidecar payload + metrics fields --

    /// Empty `payload` / `metrics` serialize as `"payload":null` /
    /// `"metrics":[]` (always-emit symmetric with `host`) rather than
    /// being omitted. Pin the wire shape so a regression that re-adds
    /// `skip_serializing_if` on either field is caught before it
    /// ships, and verify the None/empty round-trip remains correct
    /// under the deserialize-requires contract.
    #[test]
    fn sidecar_payload_and_metrics_always_emit_when_empty() {
        let sc = SidecarResult::test_fixture();
        let json = serde_json::to_string(&sc).unwrap();
        assert!(
            json.contains("\"payload\":null"),
            "empty payload must emit as `\"payload\":null`: {json}",
        );
        assert!(
            json.contains("\"metrics\":[]"),
            "empty metrics must emit as `\"metrics\":[]`: {json}",
        );
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        // Exhaustive destructure so a new `Option<_>` / `Vec<_>`
        // field on `SidecarResult` that defaults to `None` / empty
        // forces this test to spell it out and make an
        // always-emit-vs-skip decision at the same time. See
        // [`sidecar_result_roundtrip`] for the same pattern on the
        // populated side — the two together pin the wire contract
        // at both extremes of the default distribution.
        let SidecarResult {
            test_name: _,
            topology: _,
            scheduler: _,
            scheduler_commit,
            payload,
            metrics,
            passed: _,
            skipped: _,
            stats: _,
            monitor,
            stimulus_events,
            work_type: _,
            active_flags,
            verifier_stats,
            kvm_stats,
            sysctls,
            kargs,
            kernel_version,
            timestamp: _,
            run_id: _,
            host,
        } = loaded;
        assert!(payload.is_none());
        assert!(metrics.is_empty());
        // The sibling-field defaults on the empty fixture — every
        // nullable must be None and every Vec empty, matching the
        // always-emit invariants that the JSON shape above pins.
        assert!(scheduler_commit.is_none());
        assert!(monitor.is_none());
        assert!(stimulus_events.is_empty());
        assert!(active_flags.is_empty());
        assert!(verifier_stats.is_empty());
        assert!(kvm_stats.is_none());
        assert!(sysctls.is_empty());
        assert!(kargs.is_empty());
        assert!(kernel_version.is_none());
        assert!(host.is_none());
    }

    /// Populated `payload` + `metrics` survive round-trip with the
    /// exact shape stats tooling will consume — one entry per
    /// `ctx.payload(X).run()` call, each carrying its exit code and
    /// any extracted metrics. Regression guard against a future
    /// schema shift that flattens metrics across payloads (which
    /// would lose the per-payload provenance the design requires).
    #[test]
    fn sidecar_payload_and_metrics_roundtrip_populated() {
        use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};
        let pm = PayloadMetrics {
            payload_index: 0,
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 5000.0,
                polarity: Polarity::HigherBetter,
                unit: "iops".to_string(),
                source: MetricSource::Json,
                stream: MetricStream::Stdout,
            }],
            exit_code: 0,
        };
        let sc = SidecarResult {
            test_name: "fio_run".to_string(),
            topology: "1n1l2c1t".to_string(),
            payload: Some("fio".to_string()),
            metrics: vec![pm],
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        assert!(json.contains("\"payload\":\"fio\""));
        assert!(json.contains("\"metrics\""));
        assert!(json.contains("\"iops\""));
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.payload.as_deref(), Some("fio"));
        assert_eq!(loaded.metrics.len(), 1);
        assert_eq!(loaded.metrics[0].exit_code, 0);
        assert_eq!(loaded.metrics[0].metrics.len(), 1);
        assert_eq!(loaded.metrics[0].metrics[0].name, "iops");
        assert_eq!(loaded.metrics[0].metrics[0].value, 5000.0);
        assert_eq!(
            loaded.metrics[0].metrics[0].stream,
            MetricStream::Stdout,
            "metric stream tag must round-trip through sidecar \
             serde; a regression that lost `stream` serialization \
             or deserialized it to a different variant would break \
             review-tooling's stdout-vs-stderr attribution",
        );
    }

    /// `write_sidecar` must populate `payload` from `entry.payload`
    /// so a test declaring a binary payload writes the payload name
    /// into the sidecar even when no payload-metrics have been
    /// threaded in yet. This pins the half-wired state the
    /// follow-up WOs will extend: stats tooling that already groups
    /// by payload name sees the grouping key on the sidecar
    /// immediately.
    #[test]
    fn write_sidecar_records_entry_payload_name() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};

        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-payload-name-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        static FIO: Payload = Payload {
            name: "fio",
            kind: PayloadKind::Binary("fio"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__payload_name_test__",
            func: dummy,
            auto_repro: false,
            payload: Some(&FIO),
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &[]).unwrap();

        let path = find_single_sidecar_by_prefix(&tmp, "__payload_name_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.payload.as_deref(), Some("fio"));
        assert!(
            loaded.metrics.is_empty(),
            "metrics stay empty until a Ctx-level accumulator lands",
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `write_sidecar` must forward the `payload_metrics` slice
    /// into `SidecarResult.metrics` unmodified — once the
    /// follow-up Ctx-accumulator WO lands, stats tooling will see
    /// every `ctx.payload(X).run()` invocation's output in order.
    #[test]
    fn write_sidecar_forwards_payload_metrics_slice() {
        use crate::test_support::{Metric, MetricSource, MetricStream, PayloadMetrics, Polarity};

        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-metrics-slice-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__metrics_slice_test__",
            func: dummy,
            auto_repro: false,
            ..KtstrTestEntry::DEFAULT
        };
        let vm_result = crate::vmm::VmResult::test_fixture();
        let ok = AssertResult::pass();
        let metrics = vec![
            PayloadMetrics {
                payload_index: 0,
                metrics: vec![Metric {
                    name: "iops".to_string(),
                    value: 1200.0,
                    polarity: Polarity::HigherBetter,
                    unit: "iops".to_string(),
                    source: MetricSource::Json,
                    stream: MetricStream::Stdout,
                }],
                exit_code: 0,
            },
            PayloadMetrics {
                payload_index: 1,
                metrics: vec![],
                exit_code: 2,
            },
        ];
        write_sidecar(&entry, &vm_result, &[], &ok, "CpuSpin", &[], &metrics).unwrap();

        let path = find_single_sidecar_by_prefix(&tmp, "__metrics_slice_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.metrics.len(), 2);
        assert_eq!(loaded.metrics[0].exit_code, 0);
        assert_eq!(loaded.metrics[0].metrics.len(), 1);
        assert_eq!(loaded.metrics[0].metrics[0].name, "iops");
        assert_eq!(loaded.metrics[1].exit_code, 2);
        assert!(loaded.metrics[1].metrics.is_empty());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `write_skip_sidecar` must also carry `entry.payload` through
    /// so a ResourceContention or early-skip on a payload-carrying
    /// test still records the payload name. Missing this would
    /// drop skipped runs out of payload-grouped stats.
    #[test]
    fn write_skip_sidecar_records_entry_payload_name() {
        use crate::test_support::{OutputFormat, Payload, PayloadKind};

        let _lock = lock_env();
        let tmp = std::env::temp_dir().join("ktstr-sidecar-skip-payload-test");
        let _ = std::fs::remove_dir_all(&tmp);
        let _env_sidecar = EnvVarGuard::set("KTSTR_SIDECAR_DIR", &tmp);

        static STRESS: Payload = Payload {
            name: "stress-ng",
            kind: PayloadKind::Binary("stress-ng"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
            include_files: &[],
            uses_parent_pgrp: false,
            known_flags: None,
        };

        fn dummy(_ctx: &Ctx) -> Result<AssertResult> {
            Ok(AssertResult::pass())
        }
        let entry = KtstrTestEntry {
            name: "__skip_payload_name_test__",
            func: dummy,
            auto_repro: false,
            payload: Some(&STRESS),
            ..KtstrTestEntry::DEFAULT
        };
        write_skip_sidecar(&entry, &[]).unwrap();

        let path = find_single_sidecar_by_prefix(&tmp, "__skip_payload_name_test__-");
        let data = std::fs::read_to_string(&path).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&data).unwrap();
        assert_eq!(loaded.payload.as_deref(), Some("stress-ng"));
        assert!(loaded.skipped);
        assert!(
            loaded.metrics.is_empty(),
            "skip path never accumulates metrics"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// `host` is deliberately excluded from `sidecar_variant_hash`:
    /// two gauntlet variants run on different hosts must collapse
    /// into the same hash bucket so downstream stats tooling groups
    /// them together. If a future change accidentally folds
    /// `HostContext` into the hash, this test catches it before
    /// the run-key split reaches on-disk sidecars.
    #[test]
    fn sidecar_variant_hash_excludes_host_context() {
        use crate::host_context::HostContext;
        let populated = HostContext {
            cpu_model: Some("Example CPU".to_string()),
            cpu_vendor: Some("GenuineExample".to_string()),
            total_memory_kb: Some(16_384_000),
            hugepages_total: Some(0),
            hugepages_free: Some(0),
            hugepages_size_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("[always] defer madvise never".to_string()),
            sched_tunables: None,
            online_cpus: Some(8),
            numa_nodes: Some(2),
            cpufreq_governor: std::collections::BTreeMap::new(),
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            arch: Some("x86_64".to_string()),
            kernel_cmdline: Some("preempt=lazy".to_string()),
            heap_state: None,
        };
        let without_host = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            ..SidecarResult::test_fixture()
        };
        let with_host = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            host: Some(populated),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&without_host),
            sidecar_variant_hash(&with_host),
            "host context must not influence variant hash",
        );
    }

    /// `scheduler_commit` is metadata, not a variant discriminator:
    /// two gauntlet runs differing only in the recorded scheduler
    /// commit (e.g. same variant re-run after a scheduler rebuild)
    /// must share one hash bucket so `stats compare` treats them as
    /// the same semantic variant. If a future change folds
    /// `scheduler_commit` into `sidecar_variant_hash`, this test
    /// catches it before the run-key split reaches on-disk sidecars
    /// and splits previously-comparable runs. Mirrors
    /// `sidecar_variant_hash_excludes_host_context`.
    #[test]
    fn sidecar_variant_hash_excludes_scheduler_commit() {
        let without_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            scheduler_commit: None,
            ..SidecarResult::test_fixture()
        };
        let with_commit = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            scheduler_commit: Some("0000000000000000000000000000000000000000".to_string()),
            ..SidecarResult::test_fixture()
        };
        assert_eq!(
            sidecar_variant_hash(&without_commit),
            sidecar_variant_hash(&with_commit),
            "scheduler_commit must not influence variant hash — \
             runs of the same semantic variant on different \
             scheduler-binary builds must remain comparable by \
             `stats compare`",
        );
    }

    /// A `SidecarResult` carrying a fully populated `HostContext`
    /// round-trips through serde_json without losing fields.
    /// Struct-level `PartialEq` on `HostContext` makes one
    /// `assert_eq!(host, ctx)` cover every field, so a future
    /// change that breaks composition between the outer
    /// `SidecarResult` and the embedded `HostContext` is caught at
    /// the seam without needing a per-field assertion.
    #[test]
    fn sidecar_result_roundtrip_with_populated_host_context() {
        use crate::host_context::HostContext;
        let mut tunables = std::collections::BTreeMap::new();
        tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
        let ctx = HostContext {
            cpu_model: Some("Example CPU".to_string()),
            cpu_vendor: Some("GenuineExample".to_string()),
            total_memory_kb: Some(16_384_000),
            hugepages_total: Some(4),
            hugepages_free: Some(2),
            hugepages_size_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("[always] defer madvise never".to_string()),
            sched_tunables: Some(tunables),
            online_cpus: Some(8),
            numa_nodes: Some(2),
            cpufreq_governor: std::collections::BTreeMap::new(),
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            arch: Some("x86_64".to_string()),
            kernel_cmdline: Some("preempt=lazy isolcpus=1-3".to_string()),
            heap_state: Some(crate::host_heap::HostHeapState::test_fixture()),
        };
        let sc = SidecarResult {
            topology: "1n1l2c1t".to_string(),
            host: Some(ctx.clone()),
            ..SidecarResult::test_fixture()
        };
        let json = serde_json::to_string(&sc).unwrap();
        let loaded: SidecarResult = serde_json::from_str(&json).unwrap();
        let host = loaded.host.expect("host must round-trip");
        assert_eq!(host, ctx);
    }

    /// Every sidecar produced within a single ktstr run records the
    /// SAME host context — all writers call
    /// [`crate::host_context::collect_host_context`], which
    /// memoises the static subset in a process-global `OnceLock`
    /// (`STATIC_HOST_INFO`) and re-reads the dynamic subset from
    /// the same `/proc` / `/sys` sources on every call. Runtime
    /// drift in the captured struct across sidecars in one run
    /// would mean one of two bad outcomes:
    ///   - a regression in the static memoisation (cache key / init
    ///     closure), producing per-call distinct values for fields
    ///     that cannot change across a process lifetime (uname,
    ///     CPU model, NUMA topology);
    ///   - a test concurrently mutating a dynamic field
    ///     (`thp_enabled`, `sched_tunables`, hugepage reservations)
    ///     while another test writes a sidecar, which would be a
    ///     test-isolation bug — every in-tree test treats host
    ///     tunables as read-only.
    ///
    /// This test runs a deterministic N-iteration loop (NOT a
    /// proptest-style property sampler — there is no input-space
    /// shrinker and no random seed; the same N calls with the same
    /// ordering produce the same comparisons every run) of
    /// back-to-back `collect_host_context()` calls simulating the
    /// per-test sidecar drumbeat of a gauntlet run. Every resulting
    /// `host` field must compare equal across all N samples. The
    /// sibling [`crate::host_context`] tests already pin
    /// `collect_host_context` internal stability; this test pins
    /// the SIDECAR surface so a regression that threaded a partial
    /// context through `write_sidecar` / `write_skip_sidecar`
    /// would fail here even if `collect_host_context` itself
    /// stayed stable.
    ///
    /// Bounded N=8: enough iterations to catch intermittent drift
    /// without bloating the test runtime — `collect_host_context`
    /// does ~20 sysfs/procfs reads per call, so the cost scales
    /// linearly and must stay modest.
    ///
    /// `#[cfg(target_os = "linux")]`: `collect_host_context` only
    /// reads meaningful data on Linux — on other hosts every field
    /// is `None` and the equality would trivially hold without
    /// exercising the contract.
    #[cfg(target_os = "linux")]
    #[test]
    fn sidecars_in_a_run_carry_identical_host_context() {
        const N: usize = 8;
        let samples: Vec<crate::host_context::HostContext> = (0..N)
            .map(|_| crate::host_context::collect_host_context())
            .collect();
        let first = samples
            .first()
            .expect("N > 0 samples must produce at least one host context");

        // Fields expected to stay STRICTLY equal — either memoised
        // in STATIC_HOST_INFO (uname, CPU, memory, topology) or
        // effectively reboot-static (kernel_cmdline). A regression
        // that broke the cache or mis-read /proc would diverge here.
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(
                s.kernel_name, first.kernel_name,
                "sidecar {i}: kernel_name drifted from first sample",
            );
            assert_eq!(
                s.kernel_release, first.kernel_release,
                "sidecar {i}: kernel_release drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.arch, first.arch,
                "sidecar {i}: arch drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.cpu_model, first.cpu_model,
                "sidecar {i}: cpu_model drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.cpu_vendor, first.cpu_vendor,
                "sidecar {i}: cpu_vendor drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.total_memory_kb, first.total_memory_kb,
                "sidecar {i}: total_memory_kb drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.hugepages_size_kb, first.hugepages_size_kb,
                "sidecar {i}: hugepages_size_kb drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.online_cpus, first.online_cpus,
                "sidecar {i}: online_cpus drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.numa_nodes, first.numa_nodes,
                "sidecar {i}: numa_nodes drifted — STATIC_HOST_INFO cache broken?",
            );
            assert_eq!(
                s.kernel_cmdline, first.kernel_cmdline,
                "sidecar {i}: kernel_cmdline drifted — only a reboot can change it",
            );
        }

        // Dynamic fields are allowed to vary in value under
        // concurrent sysctl/THP/hugepage twiddles (see the sibling
        // `collect_host_context_dynamic_subset_is_stable_across_calls`
        // test for the rationale), but the PRESENCE of each field
        // must stay consistent — a sidecar that suddenly loses the
        // THP row means the collector silently degraded, which
        // stats tooling would read as "no THP data on that host"
        // rather than the truth ("collector broke").
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(
                s.hugepages_total.is_some(),
                first.hugepages_total.is_some(),
                "sidecar {i}: hugepages_total presence flipped across sidecars",
            );
            assert_eq!(
                s.hugepages_free.is_some(),
                first.hugepages_free.is_some(),
                "sidecar {i}: hugepages_free presence flipped across sidecars",
            );
            assert_eq!(
                s.thp_enabled.is_some(),
                first.thp_enabled.is_some(),
                "sidecar {i}: thp_enabled presence flipped across sidecars",
            );
            assert_eq!(
                s.thp_defrag.is_some(),
                first.thp_defrag.is_some(),
                "sidecar {i}: thp_defrag presence flipped across sidecars",
            );
            assert_eq!(
                s.sched_tunables.is_some(),
                first.sched_tunables.is_some(),
                "sidecar {i}: sched_tunables presence flipped across sidecars",
            );
        }
    }
}
