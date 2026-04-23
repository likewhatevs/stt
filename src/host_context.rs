//! Host runtime state captured at sidecar-write time.
//!
//! [`HostContext`] is a snapshot of the host running the tool:
//! kernel release, CPU identity, memory size, hugepages config,
//! transparent-hugepage policy, kernel scheduler tunables, NUMA
//! node count, and kernel cmdline. Static fields (CPU identity,
//! total memory, hugepage size, NUMA count, uname triple) are
//! memoized in [`OnceLock`] across the process; dynamic fields
//! (sched tunables, hugepages totals, THP policy, cmdline) are
//! re-read on every call so run-time sysctl changes or hugepage
//! reservations between tests are not hidden by the cache.
//!
//! ## Static-cache staleness under hotplug
//!
//! The static-field cache pins the first snapshot it observes for
//! the life of the process. This is OUR invariant, not the
//! kernel's: `/proc/meminfo`'s `MemTotal`,
//! `/sys/devices/system/node/*`, and the `uname()` return all
//! update live when memory or NUMA hotplug fires, and a freshly-
//! started process would pick up the new values on its next
//! collect call. It is [`STATIC_HOST_INFO`]'s `OnceLock` that
//! binds a single read for the process lifetime — not any
//! kernel-side caching.
//!
//! So on a host where CPU / NUMA / memory hotplug fires between
//! two collect calls in the same process, `HostContext` continues
//! to report the pre-hotplug values — `total_memory_kb` stays at
//! the original snapshot, `numa_nodes` does not reflect an
//! added/removed node. `arch` is the only field genuinely immune
//! (a reboot is required to change architecture).
//!
//! Tests that need live-updated values must either (a) avoid
//! reading HostContext after the hotplug event, or (b) restart
//! the process to force a fresh `OnceLock` population. No
//! `reset` hook is exposed in production; the `#[cfg(test)]`-only
//! reset machinery is for unit tests, not runtime recapture.

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Host-level runtime state snapshot attached to each
/// [`SidecarResult`](crate::test_support::SidecarResult). Every
/// field is optional so a partial read (missing /proc entry,
/// permission denied, parse failure) still records the fields that
/// did succeed instead of dropping the whole snapshot.
///
/// # Constructing instances in tests
///
/// `HostContext` is `#[non_exhaustive]`: downstream consumers
/// cannot build one with a bare struct literal (`HostContext { ... }`)
/// and must combine [`Default`] with Rust's struct-update syntax so
/// a future field addition doesn't break the call site. The standard
/// idiom is:
///
/// ```
/// use ktstr::prelude::HostContext;
/// let ctx = HostContext {
///     cpu_model: Some("Test CPU".to_string()),
///     numa_nodes: Some(2),
///     ..HostContext::default()
/// };
/// ```
///
/// For tests that want a populated baseline (non-trivial defaults
/// for every field) instead of `Default`'s all-`None` minimum, use
/// [`HostContext::test_fixture`].
///
/// # Error-free deserialization under field drift
///
/// The `Deserialize` impl is derived WITHOUT
/// `#[serde(deny_unknown_fields)]`. An older binary reading a
/// sidecar written by a newer binary therefore silently ignores
/// any fields it does not recognize, and the downstream
/// `SidecarResult` parse succeeds with the older struct shape.
/// This is the intentional forward-compat contract: adding a new
/// `Option<T>` field to `HostContext` does NOT break consumers
/// built against a prior schema. Paired with the per-field
/// `#[serde(default)]` on every attribute, missing fields also
/// default cleanly — so a newer binary reading an older sidecar
/// that lacks a newly-added field gets `None` rather than a
/// deserialize error. Both directions of the version skew are
/// covered by this policy.
///
/// "Forward-compat" here means only that deserialization does
/// not error — it does NOT mean data is preserved across field
/// renames. If a field is renamed (e.g. `uname_sysname` →
/// `kernel_name`), a sidecar written under the old name
/// deserializes cleanly but the renamed field lands as `None` on
/// the new struct, because `#[serde(default)]` supplies the
/// absent-field default and there is no alias mapping. This is
/// by design: sidecar data is disposable (re-running the test
/// regenerates it with the current schema), so rename migrations
/// do not carry alias shims.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct HostContext {
    /// CPU model string — the `model name` line of `/proc/cpuinfo`.
    /// Single value (first processor entry) since heterogeneous
    /// CPU models on a single host are rare enough that the
    /// extra complexity is not worth carrying.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_model: Option<String>,
    /// CPU vendor ID — the `vendor_id` line of `/proc/cpuinfo`
    /// (e.g. `GenuineIntel`, `AuthenticAMD`). On ARM64,
    /// `/proc/cpuinfo` uses `CPU implementer` instead of
    /// `vendor_id`, so this field is `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_vendor: Option<String>,
    /// Total physical memory in KiB — `MemTotal:` from
    /// `/proc/meminfo`. Unit matches the file exactly so the sidecar
    /// reader does not need to guess the scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_memory_kb: Option<u64>,
    /// Configured huge pages — `HugePages_Total` from `/proc/meminfo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hugepages_total: Option<u64>,
    /// Free huge pages — `HugePages_Free` from `/proc/meminfo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hugepages_free: Option<u64>,
    /// Hugepage size in KiB — `Hugepagesize:` from `/proc/meminfo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hugepages_size_kb: Option<u64>,
    /// Active THP policy — content of
    /// `/sys/kernel/mm/transparent_hugepage/enabled` with the
    /// bracketed selection preserved verbatim (e.g.
    /// `"always [madvise] never"`). Trimmed of leading and
    /// trailing whitespace by `read_trimmed_sysfs`, so the trailing
    /// newline that sysfs appends does not appear in the captured
    /// value. Stored as-read rather than parsed because the bracket
    /// is the meaningful part and downstream tooling may want the
    /// full menu too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thp_enabled: Option<String>,
    /// Active THP defrag policy — content of
    /// `/sys/kernel/mm/transparent_hugepage/defrag`, bracket
    /// preserved. Trimmed of leading and trailing whitespace by
    /// `read_trimmed_sysfs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thp_defrag: Option<String>,
    /// `/proc/sys/kernel/sched_*` tunables. Keys are the leaf
    /// basename (e.g. `sched_migration_cost_ns`); values are the
    /// file content trimmed of leading and trailing whitespace
    /// (internal whitespace preserved — `read_trimmed_sysfs` uses
    /// `str::trim`, which only strips edges). Every current
    /// `sched_*` tunable is a scalar, but a future kernel that
    /// exposes a multi-line tunable would land here as a
    /// multi-line `String`. `None` when the `read_dir` of
    /// `/proc/sys/kernel` fails; empty map when the directory is
    /// readable but contains no entries starting with `sched_`
    /// (or all such entries fail the per-file read or trim to
    /// empty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sched_tunables: Option<BTreeMap<String, String>>,
    /// Count of NUMA nodes — derived from
    /// `HostTopology::from_sysfs` (the `cpu_to_node` map's distinct
    /// value count). `None` when the topology probe itself fails so
    /// "unknown" is distinguishable from a populated result. A probe
    /// that succeeds but reports no CPU→node entries defaults to
    /// `Some(1)` because every Linux system has at least one NUMA
    /// node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa_nodes: Option<usize>,
    /// Kernel name — `uname.sysname` (typically `"Linux"`).
    /// The nodename field is intentionally dropped; it's a local
    /// hostname and has no place in a published sidecar.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_name: Option<String>,
    /// Kernel release — `uname.release` (e.g. `"6.11.0-rc3"`).
    /// The full `/proc/version` banner is NOT captured because it
    /// embeds the build host + gcc version string, which is
    /// environment leakage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_release: Option<String>,
    /// Machine architecture — `uname.machine` (e.g. `"x86_64"`,
    /// `"aarch64"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    /// `/proc/cmdline` verbatim (trimmed of leading and trailing
    /// whitespace). Captures boot-time parameters that materially
    /// affect scheduler behavior — `preempt=`, `isolcpus=`,
    /// `nohz_full=`, `mitigations=`, hugepage reservations,
    /// `transparent_hugepage=`, and others. Stored as a single
    /// string because any split-into-pairs parser loses the
    /// quoted-value and flag-only variants the kernel accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cmdline: Option<String>,
}

impl HostContext {
    /// Populated [`HostContext`] for unit tests. Every field carries
    /// a reasonable non-trivial value so call sites only spell out
    /// what they want to vary via struct-update syntax:
    ///
    /// ```
    /// use ktstr::prelude::HostContext;
    /// let ctx = HostContext {
    ///     numa_nodes: Some(4),
    ///     ..HostContext::test_fixture()
    /// };
    /// ```
    ///
    /// Defaults model a plausible 2-node x86_64 Linux host: Intel
    /// CPU identity, 64 GiB memory, 2 NUMA nodes, default THP
    /// policies, a minimal `sched_*` tunable map, and a populated
    /// uname triple. Parity with
    /// [`SidecarResult::test_fixture`](crate::test_support::SidecarResult::test_fixture) —
    /// both fixtures exist so tests don't re-derive an "everything
    /// populated" baseline in every call site.
    ///
    /// **Prefer this over local "populated default" helpers.** A
    /// local closure duplicates the default set and drifts the
    /// moment [`HostContext`] grows a field; this fixture is the
    /// single place those defaults live.
    ///
    /// **Hash-stability or serialization-pin tests must not rely on
    /// these defaults.** Tests that fix a specific JSON/byte output
    /// should spell every participating field out explicitly so a
    /// future change to the fixture cannot silently shift the
    /// pinned value.
    pub fn test_fixture() -> HostContext {
        let mut sched_tunables = BTreeMap::new();
        sched_tunables.insert(
            "sched_migration_cost_ns".to_string(),
            "500000".to_string(),
        );
        sched_tunables.insert(
            "sched_latency_ns".to_string(),
            "24000000".to_string(),
        );
        HostContext {
            cpu_model: Some("Intel(R) Xeon(R) Test CPU".to_string()),
            cpu_vendor: Some("GenuineIntel".to_string()),
            total_memory_kb: Some(64 * 1024 * 1024),
            hugepages_total: Some(0),
            hugepages_free: Some(0),
            hugepages_size_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("always defer defer+madvise [madvise] never".to_string()),
            sched_tunables: Some(sched_tunables),
            numa_nodes: Some(2),
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.16.0-test".to_string()),
            arch: Some("x86_64".to_string()),
            cmdline: Some(
                "BOOT_IMAGE=/boot/vmlinuz-test root=/dev/sda1".to_string(),
            ),
        }
    }

    /// Render as a human-readable multi-line report. Each field
    /// occupies one line as `key: value`. Absent fields render as
    /// `(unknown)` rather than being dropped, so operators see
    /// which fields failed to populate. The `sched_tunables` map
    /// is expanded one entry per line under the parent key; an
    /// empty map renders as `(empty)` and a `None` map as
    /// `(unknown)`. The output ends with a newline.
    ///
    /// This output is for human inspection only. For programmatic
    /// access, parse the sidecar JSON directly or drive `serde_json`
    /// against the [`HostContext`] struct — the text format here is
    /// not a stable serialization contract and may be retuned for
    /// readability without notice.
    ///
    /// Naming: the name pair (`format_human` with no
    /// `format_machine`) is intentional rather than accidental
    /// asymmetry. The "machine" surface is serde JSON — callers
    /// that want a machine-readable rendering use
    /// `serde_json::to_string(ctx)` directly. A dedicated
    /// `format_machine` wrapper around that one line would add no
    /// value. `format_human` stays named as it is (not as
    /// `impl Display`) because it prints a multi-line block with
    /// its own newline, which clashes with `Display`'s implicit
    /// one-value-per-formatter convention; embedding this in
    /// `format!("{ctx}")` would surprise callers used to single-
    /// line Display output.
    pub fn format_human(&self) -> String {
        use std::fmt::Write;
        // Destructuring bind forces every field of HostContext to
        // appear by name here. Adding a new field to the struct
        // will fail compilation until this function handles it —
        // that is the intent, it prevents `show-host` from
        // silently dropping a freshly-captured dimension.
        let HostContext {
            cpu_model,
            cpu_vendor,
            total_memory_kb,
            hugepages_total,
            hugepages_free,
            hugepages_size_kb,
            thp_enabled,
            thp_defrag,
            sched_tunables,
            numa_nodes,
            kernel_name,
            kernel_release,
            arch,
            cmdline,
        } = self;
        fn row<T: std::fmt::Display>(out: &mut String, key: &str, value: Option<&T>) {
            match value {
                Some(v) => {
                    let _ = writeln!(out, "{key}: {v}");
                }
                None => {
                    let _ = writeln!(out, "{key}: (unknown)");
                }
            }
        }
        let mut out = String::new();
        row(&mut out, "kernel_name", kernel_name.as_ref());
        row(&mut out, "kernel_release", kernel_release.as_ref());
        row(&mut out, "arch", arch.as_ref());
        row(&mut out, "cpu_model", cpu_model.as_ref());
        row(&mut out, "cpu_vendor", cpu_vendor.as_ref());
        row(&mut out, "total_memory_kb", total_memory_kb.as_ref());
        row(&mut out, "hugepages_total", hugepages_total.as_ref());
        row(&mut out, "hugepages_free", hugepages_free.as_ref());
        row(&mut out, "hugepages_size_kb", hugepages_size_kb.as_ref());
        row(&mut out, "numa_nodes", numa_nodes.as_ref());
        row(&mut out, "thp_enabled", thp_enabled.as_ref());
        row(&mut out, "thp_defrag", thp_defrag.as_ref());
        row(&mut out, "cmdline", cmdline.as_ref());
        match sched_tunables {
            Some(map) if !map.is_empty() => {
                out.push_str("sched_tunables:\n");
                for (k, v) in map {
                    let _ = writeln!(&mut out, "  {k} = {v}");
                }
            }
            Some(_) => out.push_str("sched_tunables: (empty)\n"),
            None => out.push_str("sched_tunables: (unknown)\n"),
        }
        out
    }

    /// Render the differences between two host contexts as
    /// indented `key: before → after` lines. Fields that compare
    /// equal are omitted; an empty return value means the two
    /// contexts are field-for-field identical (including
    /// `sched_tunables`). `None` values render as `(unknown)` and
    /// map entries present in one side only render as `(absent)`
    /// so a `None → Some(..)` transition does not silently look
    /// the same as an unchanged absent field. When only one side
    /// has a `sched_tunables` map, the other side renders
    /// `(unknown)`; the Some side renders as `(empty)` for an
    /// empty map or `(N entries)` for a populated one so the
    /// cardinality of the new data is visible at a glance.
    pub fn diff(a: &HostContext, b: &HostContext) -> String {
        use std::collections::BTreeMap;
        use std::fmt::Write;
        // Symmetric destructuring bind of both sides: forces every
        // field to appear by name here, same reason as
        // `format_human` — a new HostContext field must be
        // explicitly classified as hash-participating, scalar, or
        // structured before diff will compile.
        let HostContext {
            cpu_model: a_cpu_model,
            cpu_vendor: a_cpu_vendor,
            total_memory_kb: a_total_memory_kb,
            hugepages_total: a_hugepages_total,
            hugepages_free: a_hugepages_free,
            hugepages_size_kb: a_hugepages_size_kb,
            thp_enabled: a_thp_enabled,
            thp_defrag: a_thp_defrag,
            sched_tunables: a_sched_tunables,
            numa_nodes: a_numa_nodes,
            kernel_name: a_kernel_name,
            kernel_release: a_kernel_release,
            arch: a_arch,
            cmdline: a_cmdline,
        } = a;
        let HostContext {
            cpu_model: b_cpu_model,
            cpu_vendor: b_cpu_vendor,
            total_memory_kb: b_total_memory_kb,
            hugepages_total: b_hugepages_total,
            hugepages_free: b_hugepages_free,
            hugepages_size_kb: b_hugepages_size_kb,
            thp_enabled: b_thp_enabled,
            thp_defrag: b_thp_defrag,
            sched_tunables: b_sched_tunables,
            numa_nodes: b_numa_nodes,
            kernel_name: b_kernel_name,
            kernel_release: b_kernel_release,
            arch: b_arch,
            cmdline: b_cmdline,
        } = b;
        fn fmt_opt<T: std::fmt::Display>(v: Option<&T>) -> String {
            match v {
                Some(v) => v.to_string(),
                None => "(unknown)".to_string(),
            }
        }
        fn row<T: std::fmt::Display + PartialEq>(
            out: &mut String,
            key: &str,
            a: Option<&T>,
            b: Option<&T>,
        ) {
            if a == b {
                return;
            }
            let _ = writeln!(out, "  {key}: {} → {}", fmt_opt(a), fmt_opt(b));
        }
        fn summarize_tunables(m: Option<&BTreeMap<String, String>>) -> String {
            match m {
                None => "(unknown)".to_string(),
                Some(map) if map.is_empty() => "(empty)".to_string(),
                Some(map) if map.len() == 1 => "(1 entry)".to_string(),
                Some(map) => format!("({} entries)", map.len()),
            }
        }
        let mut out = String::new();
        row(&mut out, "kernel_name", a_kernel_name.as_ref(), b_kernel_name.as_ref());
        row(&mut out, "kernel_release", a_kernel_release.as_ref(), b_kernel_release.as_ref());
        row(&mut out, "arch", a_arch.as_ref(), b_arch.as_ref());
        row(&mut out, "cpu_model", a_cpu_model.as_ref(), b_cpu_model.as_ref());
        row(&mut out, "cpu_vendor", a_cpu_vendor.as_ref(), b_cpu_vendor.as_ref());
        row(
            &mut out,
            "total_memory_kb",
            a_total_memory_kb.as_ref(),
            b_total_memory_kb.as_ref(),
        );
        row(
            &mut out,
            "hugepages_total",
            a_hugepages_total.as_ref(),
            b_hugepages_total.as_ref(),
        );
        row(
            &mut out,
            "hugepages_free",
            a_hugepages_free.as_ref(),
            b_hugepages_free.as_ref(),
        );
        row(
            &mut out,
            "hugepages_size_kb",
            a_hugepages_size_kb.as_ref(),
            b_hugepages_size_kb.as_ref(),
        );
        row(&mut out, "numa_nodes", a_numa_nodes.as_ref(), b_numa_nodes.as_ref());
        row(&mut out, "thp_enabled", a_thp_enabled.as_ref(), b_thp_enabled.as_ref());
        row(&mut out, "thp_defrag", a_thp_defrag.as_ref(), b_thp_defrag.as_ref());
        row(&mut out, "cmdline", a_cmdline.as_ref(), b_cmdline.as_ref());
        match (a_sched_tunables.as_ref(), b_sched_tunables.as_ref()) {
            (Some(am), Some(bm)) => {
                let mut keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
                keys.extend(am.keys().map(String::as_str));
                keys.extend(bm.keys().map(String::as_str));
                for k in keys {
                    let av = am.get(k);
                    let bv = bm.get(k);
                    if av != bv {
                        let _ = writeln!(
                            &mut out,
                            "  sched_tunables.{k}: {} → {}",
                            av.map(String::as_str).unwrap_or("(absent)"),
                            bv.map(String::as_str).unwrap_or("(absent)"),
                        );
                    }
                }
            }
            (am, bm) if am != bm => {
                let _ = writeln!(
                    &mut out,
                    "  sched_tunables: {} → {}",
                    summarize_tunables(am),
                    summarize_tunables(bm),
                );
            }
            _ => {}
        }
        out
    }
}

/// Static-fields cache. These values do not change for the lifetime
/// of the process (CPU identity, total installed memory, hugepage
/// size chosen at boot, NUMA count, uname triple), so walking
/// `/proc` and `/sys` for them once and reusing the result avoids
/// repeated syscalls on every sidecar write. Dynamic fields
/// (sched_tunables, hugepages_total, hugepages_free, thp_enabled,
/// thp_defrag, cmdline) are NOT cached — they can shift
/// between tests via sysctl, hugepage reservation, THP policy flip,
/// or live kexec, and a cached snapshot would hide that change.
#[derive(Clone)]
struct StaticHostInfo {
    cpu_model: Option<String>,
    cpu_vendor: Option<String>,
    total_memory_kb: Option<u64>,
    hugepages_size_kb: Option<u64>,
    numa_nodes: Option<usize>,
    kernel_name: Option<String>,
    kernel_release: Option<String>,
    arch: Option<String>,
}

static STATIC_HOST_INFO: OnceLock<StaticHostInfo> = OnceLock::new();

/// Test-only call counter for [`compute_static_host_info`]. Pinned
/// by `call_counts_*` tests to prove the OnceLock is exercised at
/// most once per process, independent of how many
/// `collect_host_context` calls happen. Production builds do not
/// carry the counter.
#[cfg(test)]
static STATIC_INIT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only call counter for [`read_meminfo`]. Pinned by
/// `call_counts_*` tests to prove the `/proc/meminfo` dedup holds
/// — exactly one read per `collect_host_context` call, not the
/// pre-dedup two reads on the cold path. Production builds do not
/// carry the counter.
#[cfg(test)]
static MEMINFO_READ_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Capture the host context. Static fields are collected once
/// and cached; dynamic fields are re-read on every call so
/// intra-run sysctl / hugepage / THP changes are reflected.
///
/// Every sub-read is fallible; individual failures leave the
/// corresponding field `None` and the rest of the context
/// proceeds. Even on a host where every `/proc` and `/sys` read
/// fails, the three uname-derived fields (`kernel_name`,
/// `kernel_release`, `arch`) still populate because they come from
/// the `uname()` syscall — filesystem-independent. An
/// otherwise-empty `HostContext` serializes to a near-empty JSON
/// object and distinguishes "collection attempted, nothing known"
/// from "collection not attempted" (represented at the enclosing
/// `Option<HostContext>` layer on
/// [`SidecarResult`](crate::test_support::SidecarResult)).
pub fn collect_host_context() -> HostContext {
    // Read `/proc/meminfo` exactly once per call and share the
    // parsed fields with `compute_static_host_info` (for `mem_total_kb`
    // / `hugepages_size_kb` on cold init) and with the per-call
    // hugepage counters. The prior formulation read `/proc/meminfo`
    // twice on the cold path — once here for the dynamic counters
    // and once inside the `OnceLock` init for the static fields —
    // which is wasted syscall + parse work.
    let meminfo = read_meminfo();
    let static_info = STATIC_HOST_INFO
        .get_or_init(|| compute_static_host_info(&meminfo))
        .clone();
    HostContext {
        cpu_model: static_info.cpu_model,
        cpu_vendor: static_info.cpu_vendor,
        total_memory_kb: static_info.total_memory_kb,
        hugepages_total: meminfo.hugepages_total,
        hugepages_free: meminfo.hugepages_free,
        hugepages_size_kb: static_info.hugepages_size_kb,
        thp_enabled: read_trimmed_sysfs("/sys/kernel/mm/transparent_hugepage/enabled"),
        thp_defrag: read_trimmed_sysfs("/sys/kernel/mm/transparent_hugepage/defrag"),
        sched_tunables: read_sched_tunables(),
        numa_nodes: static_info.numa_nodes,
        kernel_name: static_info.kernel_name,
        kernel_release: static_info.kernel_release,
        arch: static_info.arch,
        cmdline: read_trimmed_sysfs("/proc/cmdline"),
    }
}

/// Populate the static-fields cache on first access. Takes the
/// already-parsed `/proc/meminfo` from the caller so the cold path
/// does not re-read the file. Reads `/proc/cpuinfo` (CPU identity),
/// the host NUMA topology, and a single `uname()` call.
fn compute_static_host_info(meminfo: &MeminfoFields) -> StaticHostInfo {
    #[cfg(test)]
    STATIC_INIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (cpu_model, cpu_vendor) = read_cpuinfo_identity();
    // `uname(2)` is unit-tested only through
    // `collect_host_context_returns_populated_struct_on_linux`
    // (integration-style — runs the real syscall and asserts the
    // sysname field populates). No injection seam exists by design:
    // the only post-syscall logic here is `.to_str().ok().map(...)`,
    // which is three method calls on `rustix::system::UtsName`'s
    // already-null-terminated-`CStr` accessors. Extracting that into
    // a pure parser would test `CStr::to_str` — std's invariant, not
    // ours — and the real fragility (syscall return, encoding on
    // non-Linux hosts) is untestable without a kernel mock, which
    // is outside ktstr's scope. Marking this not-unit-tested by
    // design.
    let u = rustix::system::uname();
    StaticHostInfo {
        cpu_model,
        cpu_vendor,
        total_memory_kb: meminfo.mem_total_kb,
        hugepages_size_kb: meminfo.hugepages_size_kb,
        numa_nodes: count_numa_nodes_via_topology(),
        kernel_name: u.sysname().to_str().ok().map(|s| s.to_string()),
        kernel_release: u.release().to_str().ok().map(|s| s.to_string()),
        arch: u.machine().to_str().ok().map(|s| s.to_string()),
    }
}

/// Read `/proc/cpuinfo` and extract the first processor's
/// `vendor_id` and `model name` lines. Thin I/O wrapper; the
/// parsing logic lives in [`parse_cpuinfo_identity`] so it can
/// be unit-tested with synthetic fixtures.
fn read_cpuinfo_identity() -> (Option<String>, Option<String>) {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return (None, None);
    };
    parse_cpuinfo_identity(&text)
}

/// Pure parser split from `read_cpuinfo_identity` for unit
/// testability. Parses the first processor's `vendor_id` and
/// `model name` lines from `/proc/cpuinfo` content. Returning
/// after the first blank line (processor boundary) keeps the
/// scan O(one processor) on big machines where `/proc/cpuinfo`
/// can span many MiB.
fn parse_cpuinfo_identity(text: &str) -> (Option<String>, Option<String>) {
    let mut model: Option<String> = None;
    let mut vendor: Option<String> = None;
    for line in text.lines() {
        if line.is_empty() {
            // End of the first processor block — both fields we want
            // are per-processor and appear before the first blank
            // line.
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            if value.is_empty() {
                continue;
            }
            match key {
                "model name" if model.is_none() => model = Some(value.to_string()),
                "vendor_id" if vendor.is_none() => vendor = Some(value.to_string()),
                _ => {}
            }
        }
    }
    (model, vendor)
}

/// The `/proc/meminfo` fields the host-context snapshot consumes. A
/// purpose-built struct avoids the BTreeMap lookup/clone dance and
/// makes the set of captured fields explicit at the type level.
#[derive(Default)]
struct MeminfoFields {
    mem_total_kb: Option<u64>,
    hugepages_total: Option<u64>,
    hugepages_free: Option<u64>,
    hugepages_size_kb: Option<u64>,
}

/// Read `/proc/meminfo` and extract the four fields the host
/// context needs. Thin I/O wrapper; parsing lives in
/// [`parse_meminfo`] so it can be unit-tested with synthetic
/// fixtures.
fn read_meminfo() -> MeminfoFields {
    #[cfg(test)]
    MEMINFO_READ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return MeminfoFields::default();
    };
    parse_meminfo(&text)
}

/// Pure parser split from `read_meminfo` for unit testability.
/// Parses the four `/proc/meminfo` fields the host context needs
/// from already-read content. Lines without a numeric first token
/// are silently skipped so a kernel that introduces a new
/// non-numeric line (e.g. a future flags field) does not poison
/// the struct.
fn parse_meminfo(text: &str) -> MeminfoFields {
    let mut out = MeminfoFields::default();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let token = rest.split_whitespace().next().unwrap_or("");
        let Ok(n) = token.parse::<u64>() else {
            continue;
        };
        match key {
            "MemTotal" => out.mem_total_kb = Some(n),
            "HugePages_Total" => out.hugepages_total = Some(n),
            "HugePages_Free" => out.hugepages_free = Some(n),
            "Hugepagesize" => out.hugepages_size_kb = Some(n),
            _ => {}
        }
    }
    out
}

/// Read a sysfs leaf (or `/proc` pseudofile) and return its
/// trimmed content. Thin I/O wrapper; parsing lives in
/// [`parse_trimmed`] so it can be unit-tested with synthetic
/// fixtures. Returns `None` on any read error (ENOENT, EACCES,
/// EIO) so the caller records the field as absent without
/// treating it as a fatal context-collection failure.
fn read_trimmed_sysfs(path: impl AsRef<std::path::Path>) -> Option<String> {
    std::fs::read_to_string(path.as_ref())
        .ok()
        .and_then(|s| parse_trimmed(&s))
}

/// Pure parser split from `read_trimmed_sysfs` for unit
/// testability. Trims leading and trailing whitespace; returns
/// `None` when the result is empty — an empty cmdline or thp
/// file is not useful to record. Bracketed content inside the
/// value (e.g. `"always [madvise] never"` from THP) is preserved
/// verbatim because `str::trim` only affects the edges.
fn parse_trimmed(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Walk `/proc/sys/kernel` for entries whose name starts with
/// `sched_` and record each as `basename → content`. Skips any
/// entry that is not a regular file — directories, symlinks,
/// sockets, fifos, and block/char devices all fall through the
/// `file_type.is_file()` guard. The kernel exposes no non-file
/// `sched_*` entries today but guarding keeps behavior defined if
/// that changes. Also skips entries whose name is not valid UTF-8
/// and entries whose contents cannot be read or trim to empty.
///
/// Returns `None` only when the directory listing itself fails
/// (unreadable `/proc/sys/kernel`); an empty map is a valid result
/// — it means the directory was readable but had no entries
/// starting with `sched_`, or every such entry failed the
/// per-file read or trim to empty.
fn read_sched_tunables() -> Option<BTreeMap<String, String>> {
    read_sched_tunables_from(std::path::Path::new("/proc/sys/kernel"))
}

/// Path-parameterized walk used by [`read_sched_tunables`]. Seam for
/// unit tests that drive the walk with a tempdir full of `sched_*`
/// fixture files — everything the production caller does is mirrored
/// here except the hardcoded sysfs path, so a future test can
/// exercise the real walk + filter + read pipeline against a
/// controlled directory rather than against `/proc`.
fn read_sched_tunables_from(dir: &std::path::Path) -> Option<BTreeMap<String, String>> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut out = BTreeMap::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("sched_") {
            continue;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        if let Some(content) = read_trimmed_sysfs(&path) {
            out.insert(name.to_string(), content);
        }
    }
    Some(out)
}

/// Count distinct NUMA nodes reported by `HostTopology::from_sysfs`.
/// Reuses the existing topology probe rather than re-walking
/// `/sys/devices/system/node` so a single implementation owns the
/// "what counts as a NUMA node" decision. Returns `None` when the
/// topology probe fails. An empty `cpu_to_node` map maps to
/// `Some(1)` because every Linux system has at least one NUMA node
/// — returning zero would misrepresent the topology.
fn count_numa_nodes_via_topology() -> Option<usize> {
    let topo = crate::vmm::host_topology::HostTopology::from_sysfs().ok()?;
    Some(count_numa_nodes_in_topology(&topo))
}

/// Pure-function seam split from [`count_numa_nodes_via_topology`]:
/// given a [`HostTopology`](crate::vmm::host_topology::HostTopology),
/// return the number of distinct NUMA nodes it claims. An empty
/// `cpu_to_node` map maps to `1` because every Linux system has at
/// least one NUMA node — returning zero would misrepresent the
/// topology. Sparse / non-contiguous node IDs are counted correctly
/// because `BTreeSet::from_iter` deduplicates on insert.
///
/// Keeping the I/O (sysfs probe) separate from the pure counting
/// logic lets unit tests exercise the UMA-fallback branch and the
/// dedup path without standing up a real /sys layout. Also acts as
/// a partial injection seam for #22.
pub(crate) fn count_numa_nodes_in_topology(
    topo: &crate::vmm::host_topology::HostTopology,
) -> usize {
    topo.cpu_to_node
        .values()
        .copied()
        .collect::<std::collections::BTreeSet<usize>>()
        .len()
        .max(1)
}

// Most tests in this module are pure parsers / formatters / diff
// helpers that compile and pass on any target. The handful that
// actually read `/proc`, `/sys`, or assert `kernel_name == "Linux"`
// are individually gated with `#[cfg(target_os = "linux")]` at the
// test-fn level so non-Linux contributors still get coverage of the
// portable surface.
#[cfg(test)]
mod tests {
    use super::*;

    /// Host-context reads are host-dependent: we assert the
    /// collector returns SOMETHING, not specific values. On Linux
    /// CI the uname fields at least should populate.
    #[cfg(target_os = "linux")]
    #[test]
    fn collect_host_context_returns_populated_struct_on_linux() {
        let ctx = collect_host_context();
        // uname is always readable on Linux (it's a syscall, no
        // filesystem dependency), so these three must populate.
        assert_eq!(ctx.kernel_name.as_deref(), Some("Linux"));
        assert!(ctx.kernel_release.is_some(), "uname release present");
        assert!(ctx.arch.is_some(), "uname machine present");
    }

    /// `/proc/cmdline` is always readable on a running Linux system
    /// (the kernel exposes it unconditionally). The capture is
    /// verbatim — `read_trimmed_sysfs` trims leading/trailing
    /// whitespace and returns `None` only when the read fails or
    /// the file is empty after trim. No token filtering is applied.
    /// Because the cmdline is always present on Linux, this test
    /// asserts the field populates unconditionally; an if-let
    /// version of this check would pass vacuously on a kernel that
    /// accidentally dropped the capture.
    #[cfg(target_os = "linux")]
    #[test]
    fn collect_host_context_captures_cmdline_on_linux() {
        let ctx = collect_host_context();
        let cmdline = ctx
            .cmdline
            .as_deref()
            .expect("/proc/cmdline is always readable on a running Linux system");
        assert!(!cmdline.is_empty(), "populated cmdline must not be empty");
        assert_eq!(cmdline, cmdline.trim());
    }

    /// Stability regression — repeated calls return equal
    /// `HostContext` values. Proves stability across calls:
    /// static fields come from the cache, dynamic fields match
    /// between back-to-back reads on a quiescent host.
    #[cfg(target_os = "linux")]
    #[test]
    fn collect_host_context_is_stable_across_calls() {
        let a = collect_host_context();
        let b = collect_host_context();
        assert_eq!(a.kernel_name, b.kernel_name);
        assert_eq!(a.kernel_release, b.kernel_release);
        assert_eq!(a.arch, b.arch);
        assert_eq!(a.cpu_model, b.cpu_model);
        assert_eq!(a.cpu_vendor, b.cpu_vendor);
        assert_eq!(a.cmdline, b.cmdline);
    }

    /// Direct OnceLock caching test for `STATIC_HOST_INFO`. The
    /// sibling `collect_host_context_is_stable_across_calls` proves
    /// static fields match between calls but does not verify the
    /// cache mechanism itself — the two reads could both hit
    /// `compute_static_host_info` and still match on a quiescent
    /// host. This test pins the caching contract directly: after
    /// the first call populates `STATIC_HOST_INFO`, the stored
    /// reference survives the second call unchanged (same allocation
    /// address AND same field values), proving `get_or_init` hit the
    /// cached branch instead of re-running the init closure.
    ///
    /// Uses `OnceLock::get` (non-init probe) to observe cache state
    /// without touching it.
    ///
    /// Robust to test ordering: if another test populated
    /// `STATIC_HOST_INFO` first, `collect_host_context()` here hits
    /// the cache and the pointer comparison still passes because
    /// `OnceLock` permits no re-init.
    #[cfg(target_os = "linux")]
    #[test]
    fn static_host_info_is_cached_after_first_call() {
        let _ = collect_host_context();
        let first = STATIC_HOST_INFO
            .get()
            .expect("STATIC_HOST_INFO must be populated after collect_host_context");
        let first_ptr = first as *const StaticHostInfo;

        let _ = collect_host_context();
        let second = STATIC_HOST_INFO
            .get()
            .expect("STATIC_HOST_INFO must still be populated on second call");
        let second_ptr = second as *const StaticHostInfo;

        assert_eq!(
            first_ptr, second_ptr,
            "OnceLock must return the same allocation across calls — \
             a pointer mismatch means the cache re-initialized, \
             defeating the get_or_init contract",
        );
        // Cross-check field-level equality. Redundant with the pointer
        // check but serves as a second anchor so a future replacement
        // of `OnceLock` with something that clones on access still
        // fails loudly rather than silently weakening the cache.
        assert_eq!(first.cpu_model, second.cpu_model);
        assert_eq!(first.kernel_release, second.kernel_release);
        assert_eq!(first.total_memory_kb, second.total_memory_kb);
    }

    /// Host context round-trips through JSON — every field uses
    /// `#[serde(default, skip_serializing_if)]` so absent Options
    /// do not appear in the output and empty output parses back to
    /// `HostContext::default()`.
    #[test]
    fn host_context_empty_round_trips_via_json() {
        let empty = HostContext::default();
        let json = serde_json::to_string(&empty).expect("serialize empty");
        assert_eq!(json, "{}", "default host context must serialize to empty object");
        let decoded: HostContext =
            serde_json::from_str(&json).expect("deserialize empty");
        assert!(decoded.cpu_model.is_none());
        assert!(decoded.kernel_name.is_none());
        assert!(decoded.cmdline.is_none());
    }

    /// Populated host context round-trips — struct-level
    /// `PartialEq` makes one `assert_eq!(decoded, ctx)` cover every
    /// field. Any future field addition or serde-attr change that
    /// breaks the round-trip for any single field is caught without
    /// needing a per-field assertion.
    #[test]
    fn host_context_populated_round_trips_via_json() {
        let mut tunables = BTreeMap::new();
        tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
        let ctx = HostContext {
            cpu_model: Some("Example CPU".to_string()),
            cpu_vendor: Some("GenuineExample".to_string()),
            total_memory_kb: Some(16_384_000),
            hugepages_total: Some(0),
            hugepages_free: Some(0),
            hugepages_size_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("[always] defer defer+madvise madvise never".to_string()),
            sched_tunables: Some(tunables),
            numa_nodes: Some(2),
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            arch: Some("x86_64".to_string()),
            cmdline: Some("preempt=lazy transparent_hugepage=madvise".to_string()),
        };
        let json = serde_json::to_string(&ctx).expect("serialize");
        let decoded: HostContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, ctx);
    }

    /// Partial-None round-trip: mixed `Some`/`None` fields plus a
    /// `Some(BTreeMap)` that is intentionally empty. Covers the gap
    /// between the fully-None and fully-populated endpoints — a
    /// regression that drops a specific `Some` into `None` (or
    /// coerces `Some(empty map)` into `None` on deserialize) would
    /// pass both existing tests while breaking real sidecars where
    /// partial host-state captures are the norm (first `/proc`
    /// entry unreadable, sched_* dir readable but filtered to
    /// empty, etc.). Struct-level `PartialEq` catches the whole
    /// shape in one assertion.
    #[test]
    fn host_context_partial_none_round_trips_via_json() {
        let ctx = HostContext {
            // Identity captured on the production path.
            kernel_name: Some("Linux".to_string()),
            // Release read failed (e.g. uname syscall error on the
            // simulated failure path).
            kernel_release: None,
            arch: Some("x86_64".to_string()),
            // Map was captured but is empty — the `read_dir` of
            // /proc/sys/kernel succeeded, no entries matched the
            // `sched_*` filter (unusual but the code contract
            // explicitly distinguishes this from `None`).
            sched_tunables: Some(BTreeMap::new()),
            // Rest: None to exercise the omitted-key deserialize
            // path for every other Option field.
            cpu_model: None,
            cpu_vendor: None,
            total_memory_kb: None,
            hugepages_total: None,
            hugepages_free: None,
            hugepages_size_kb: None,
            thp_enabled: None,
            thp_defrag: None,
            numa_nodes: None,
            cmdline: None,
        };
        let json = serde_json::to_string(&ctx).expect("serialize");
        let decoded: HostContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded, ctx);
    }

    #[test]
    fn parse_cpuinfo_identity_happy_path() {
        let text = "\
processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 85
model name\t: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz
stepping\t: 4
";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert_eq!(
            model.as_deref(),
            Some("Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz"),
        );
        assert_eq!(vendor.as_deref(), Some("GenuineIntel"));
    }

    #[test]
    fn parse_cpuinfo_identity_empty_input() {
        let (model, vendor) = parse_cpuinfo_identity("");
        assert!(model.is_none());
        assert!(vendor.is_none());
    }

    #[test]
    fn parse_cpuinfo_identity_arm64_no_model_or_vendor() {
        // ARM64 /proc/cpuinfo has neither `model name` nor
        // `vendor_id` — it uses `CPU implementer`, `CPU part`, etc.
        let text = "\
processor\t: 0
BogoMIPS\t: 50.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x3
CPU part\t: 0xd0c
CPU revision\t: 1
";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert!(model.is_none(), "no 'model name' line on ARM64");
        assert!(vendor.is_none(), "no 'vendor_id' line on ARM64");
    }

    #[test]
    fn parse_cpuinfo_identity_malformed_lines_are_skipped() {
        // Lines without ':' are skipped; lines with empty value
        // after trim are skipped.
        let text = "\
nonsense line with no colon
vendor_id\t:
model name\t:    Actual Model Name
vendor_id\t: ActualVendor
";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert_eq!(model.as_deref(), Some("Actual Model Name"));
        assert_eq!(
            vendor.as_deref(),
            Some("ActualVendor"),
            "empty vendor line must not poison — next real value wins",
        );
    }

    #[test]
    fn parse_cpuinfo_identity_crlf_line_endings() {
        // `str::lines()` accepts both \n and \r\n — the \r in \r\n
        // is stripped by str::lines() itself; the trim handles any
        // residual whitespace.
        let text = "vendor_id\t: GenuineIntel\r\nmodel name\t: Some CPU\r\n";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert_eq!(model.as_deref(), Some("Some CPU"));
        assert_eq!(vendor.as_deref(), Some("GenuineIntel"));
    }

    #[test]
    fn parse_cpuinfo_identity_first_processor_only() {
        // Multi-processor /proc/cpuinfo — blank line separates
        // processor blocks. Only the first block's values must
        // surface; later blocks with different values are ignored.
        let text = "\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: First CPU

processor\t: 1
vendor_id\t: DifferentVendor
model name\t: Second CPU
";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert_eq!(model.as_deref(), Some("First CPU"));
        assert_eq!(vendor.as_deref(), Some("GenuineIntel"));
    }

    #[test]
    fn parse_meminfo_happy_path() {
        let text = "\
MemTotal:       16384000 kB
MemFree:         8000000 kB
HugePages_Total:      42
HugePages_Free:       40
Hugepagesize:       2048 kB
";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(16_384_000));
        assert_eq!(out.hugepages_total, Some(42));
        assert_eq!(out.hugepages_free, Some(40));
        assert_eq!(out.hugepages_size_kb, Some(2048));
    }

    #[test]
    fn parse_meminfo_empty_input() {
        let out = parse_meminfo("");
        assert!(out.mem_total_kb.is_none());
        assert!(out.hugepages_total.is_none());
        assert!(out.hugepages_free.is_none());
        assert!(out.hugepages_size_kb.is_none());
    }

    #[test]
    fn parse_meminfo_missing_fields_stay_none() {
        // Only MemTotal is present — the other three fields must
        // remain None so callers can distinguish "zero" from
        // "absent."
        let text = "MemTotal:       1024 kB\nMemFree:         512 kB\n";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(1024));
        assert!(out.hugepages_total.is_none());
        assert!(out.hugepages_free.is_none());
        assert!(out.hugepages_size_kb.is_none());
    }

    #[test]
    fn parse_meminfo_non_numeric_value_skipped() {
        // A future kernel flags-style line ("SomeFlags: abc def")
        // must not poison the struct — its non-numeric first token
        // causes the line to be skipped silently.
        let text = "\
MemTotal:       2048 kB
SomeFlags:      abc def ghi
Hugepagesize:      2048 kB
";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(2048));
        assert_eq!(out.hugepages_size_kb, Some(2048));
    }

    #[test]
    fn parse_meminfo_unknown_fields_tolerated() {
        // Unknown keys must be ignored without affecting known
        // fields — adding new /proc/meminfo lines upstream is a
        // no-op here.
        let text = "\
MemTotal:       100 kB
Unknown_Field:  999
HugePages_Total:   3
Another_Unknown: 77 kB
";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(100));
        assert_eq!(out.hugepages_total, Some(3));
        assert!(out.hugepages_free.is_none());
    }

    #[test]
    fn parse_meminfo_crlf_line_endings() {
        let text =
            "MemTotal:       512 kB\r\nHugePages_Total:    2\r\nHugepagesize:   2048 kB\r\n";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(512));
        assert_eq!(out.hugepages_total, Some(2));
        assert_eq!(out.hugepages_size_kb, Some(2048));
    }

    #[test]
    fn parse_cpuinfo_identity_duplicate_key_first_wins() {
        // Two `model name` / `vendor_id` lines in the first
        // processor block. The match guard is `if model.is_none()`,
        // so the first occurrence must win; the second is ignored.
        let text = "\
vendor_id\t: FirstVendor
model name\t: First Model
vendor_id\t: SecondVendor
model name\t: Second Model
";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert_eq!(model.as_deref(), Some("First Model"));
        assert_eq!(vendor.as_deref(), Some("FirstVendor"));
    }

    #[test]
    fn parse_cpuinfo_identity_value_with_internal_colon() {
        // `str::split_once(':')` splits on the first colon only,
        // so any ':' inside the value survives verbatim. Real
        // /proc/cpuinfo model names rarely contain ':' but the
        // parser must preserve them.
        let text = "model name\t: Intel(R): Xeon(R) CPU @ 2.00GHz\n";
        let (model, _vendor) = parse_cpuinfo_identity(text);
        assert_eq!(
            model.as_deref(),
            Some("Intel(R): Xeon(R) CPU @ 2.00GHz"),
            "internal ':' must be preserved in the value",
        );
    }

    #[test]
    fn parse_cpuinfo_identity_leading_blank_line() {
        // The loop breaks on the first empty line (processor-block
        // boundary). A leading blank line therefore terminates
        // before any field is read — result is (None, None).
        let text = "\nvendor_id\t: GenuineIntel\nmodel name\t: Some CPU\n";
        let (model, vendor) = parse_cpuinfo_identity(text);
        assert!(model.is_none(), "leading blank line must short-circuit");
        assert!(vendor.is_none(), "leading blank line must short-circuit");
    }

    #[test]
    fn parse_meminfo_duplicate_key_last_wins() {
        // Unlike parse_cpuinfo_identity, parse_meminfo's match
        // arms assign unconditionally — the last occurrence of a
        // key overrides earlier ones. Documented here so a future
        // change to this behavior (e.g. adding a first-wins guard)
        // is caught by this test.
        let text = "MemTotal:       100 kB\nMemTotal:       200 kB\n";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(200));
    }

    #[test]
    fn parse_meminfo_line_without_colon() {
        // Lines without ':' are skipped via `split_once(':')`
        // returning None. Real /proc/meminfo never emits such
        // lines but the parser must tolerate them without
        // dropping the surrounding valid content.
        let text = "\
garbage line without any colon
MemTotal:       100 kB
another garbage line
HugePages_Total:   3
";
        let out = parse_meminfo(text);
        assert_eq!(out.mem_total_kb, Some(100));
        assert_eq!(out.hugepages_total, Some(3));
    }

    #[test]
    fn parse_meminfo_empty_value_after_colon() {
        // A key with an empty value after the colon: rest is "",
        // split_whitespace().next() returns None, token becomes
        // the empty string, parse::<u64>() fails, the line is
        // skipped. The target field stays None so the absence is
        // visible to callers.
        let text = "MemTotal:\nHugePages_Total:  5\n";
        let out = parse_meminfo(text);
        assert!(
            out.mem_total_kb.is_none(),
            "empty value after ':' must leave the field None",
        );
        assert_eq!(
            out.hugepages_total,
            Some(5),
            "subsequent valid lines must still parse",
        );
    }

    #[test]
    fn parse_meminfo_negative_and_overflow_value_skipped() {
        // u64 parsing rejects both negative values and values
        // exceeding u64::MAX. Both failure modes must skip the
        // line silently; later valid lines still parse.
        let text = "\
MemTotal:       -1 kB
HugePages_Total:   99999999999999999999999
Hugepagesize:       2048 kB
";
        let out = parse_meminfo(text);
        assert!(
            out.mem_total_kb.is_none(),
            "negative value must fail u64 parse and skip",
        );
        assert!(
            out.hugepages_total.is_none(),
            "overflow value must fail u64 parse and skip",
        );
        assert_eq!(
            out.hugepages_size_kb,
            Some(2048),
            "later valid line must still parse",
        );
    }

    #[test]
    fn parse_trimmed_empty_is_none() {
        assert!(parse_trimmed("").is_none());
    }

    #[test]
    fn parse_trimmed_whitespace_only_is_none() {
        // Spaces, tabs, and newlines all count as whitespace for
        // `str::trim`; a file containing only those characters
        // carries no signal and must map to None.
        assert!(parse_trimmed("   \n\t  \r\n").is_none());
    }

    #[test]
    fn parse_trimmed_strips_trailing_newline() {
        // sysfs leaves typically end with a single trailing '\n';
        // the parser must strip it so downstream comparisons do
        // not carry stray whitespace.
        assert_eq!(parse_trimmed("content\n").as_deref(), Some("content"));
    }

    #[test]
    fn parse_trimmed_preserves_bracketed_thp() {
        // THP policy files read like `"always [madvise] never\n"`;
        // the bracket indicating the active selection must survive
        // the trim verbatim because `str::trim` only touches the
        // edges.
        assert_eq!(
            parse_trimmed("always [madvise] never\n").as_deref(),
            Some("always [madvise] never"),
        );
    }

    // -- format_human / diff --

    /// Snapshot-style pin of the label sequence `format_human`
    /// emits. The order is load-bearing — downstream diff tools and
    /// operator-eye scanning depend on a stable top-to-bottom field
    /// ordering (uname → CPU → memory → hugepages → NUMA → THP →
    /// cmdline → sched_tunables). A silent reorder from a future
    /// edit that shuffles the `row(...)` calls would slip past the
    /// existing `.contains(...)` checks, which are order-blind.
    /// This test fails the moment the sequence drifts; updating it
    /// forces the author to acknowledge the reorder and double-check
    /// that downstream consumers can absorb it.
    #[test]
    fn format_human_field_order_is_stable() {
        let out = HostContext::default().format_human();
        let labels: Vec<&str> = out
            .lines()
            .filter_map(|l| l.split(':').next())
            .filter(|s| !s.starts_with(' '))
            .collect();
        assert_eq!(
            labels,
            vec![
                "kernel_name",
                "kernel_release",
                "arch",
                "cpu_model",
                "cpu_vendor",
                "total_memory_kb",
                "hugepages_total",
                "hugepages_free",
                "hugepages_size_kb",
                "numa_nodes",
                "thp_enabled",
                "thp_defrag",
                "cmdline",
                "sched_tunables",
            ],
            "format_human field order drifted — if intentional, update \
             the expected vector and audit downstream diff/scan consumers",
        );
    }

    /// `format_human` on a default (all-`None`) context must
    /// render every field explicitly as `(unknown)` rather than
    /// dropping the field. Silently suppressing absent fields
    /// would hide collection failures from the operator running
    /// `cargo ktstr show-host` on a degraded host.
    #[test]
    fn format_human_default_renders_unknown_everywhere() {
        let out = HostContext::default().format_human();
        for key in [
            "kernel_name",
            "kernel_release",
            "arch",
            "cpu_model",
            "cpu_vendor",
            "total_memory_kb",
            "hugepages_total",
            "hugepages_free",
            "hugepages_size_kb",
            "numa_nodes",
            "thp_enabled",
            "thp_defrag",
            "cmdline",
            "sched_tunables",
        ] {
            assert!(
                out.contains(&format!("{key}: (unknown)")),
                "key '{key}' must render as (unknown) on a default context, got:\n{out}",
            );
        }
        assert!(
            out.ends_with('\n'),
            "format_human must end with a newline for direct print!() use",
        );
    }

    /// Populated fields render verbatim and `sched_tunables`
    /// expands per-entry under the parent key.
    #[test]
    fn format_human_populated_shows_values_and_tunables() {
        let mut tunables = BTreeMap::new();
        tunables.insert("sched_migration_cost_ns".to_string(), "500000".to_string());
        tunables.insert("sched_min_granularity_ns".to_string(), "750000".to_string());
        let ctx = HostContext {
            kernel_name: Some("Linux".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            arch: Some("x86_64".to_string()),
            cpu_model: Some("Example CPU".to_string()),
            total_memory_kb: Some(16_384_000),
            sched_tunables: Some(tunables),
            cmdline: Some("preempt=lazy".to_string()),
            ..HostContext::default()
        };
        let out = ctx.format_human();
        assert!(out.contains("kernel_name: Linux"), "{out}");
        assert!(out.contains("kernel_release: 6.11.0"), "{out}");
        assert!(out.contains("cpu_model: Example CPU"), "{out}");
        assert!(out.contains("total_memory_kb: 16384000"), "{out}");
        assert!(out.contains("cmdline: preempt=lazy"), "{out}");
        assert!(out.contains("sched_tunables:\n"), "{out}");
        assert!(out.contains("  sched_migration_cost_ns = 500000"), "{out}");
        assert!(out.contains("  sched_min_granularity_ns = 750000"), "{out}");
        // Non-populated fields still render as (unknown) — show-host
        // never silently hides a field.
        assert!(out.contains("cpu_vendor: (unknown)"), "{out}");
        assert!(
            out.ends_with('\n'),
            "format_human output must terminate with a newline so the \
             next line the operator sees sits on its own row: {out:?}",
        );
    }

    /// `sched_tunables: Some(empty)` must not render as the generic
    /// `(unknown)` — an empty map is a valid result (kernel with
    /// no `sched_*` entries readable) and is distinguishable from
    /// `None` (read_dir failure).
    #[test]
    fn format_human_sched_tunables_empty_vs_none() {
        let mut ctx = HostContext::default();
        ctx.sched_tunables = Some(BTreeMap::new());
        let out_empty = ctx.format_human();
        assert!(
            out_empty.contains("sched_tunables: (empty)"),
            "empty map must render distinctly from None: {out_empty}",
        );
        assert!(
            out_empty.ends_with('\n'),
            "format_human with empty tunables must still end with a \
             newline: {out_empty:?}",
        );
        ctx.sched_tunables = None;
        let out_none = ctx.format_human();
        assert!(
            out_none.contains("sched_tunables: (unknown)"),
            "None map must render as (unknown): {out_none}",
        );
        assert!(
            out_none.ends_with('\n'),
            "format_human with no tunables must still end with a \
             newline: {out_none:?}",
        );
    }

    /// Two identical contexts diff to an empty string. This is the
    /// signal `compare_runs` uses to print `host: identical
    /// between a and b` instead of an empty delta section.
    #[test]
    fn diff_identical_is_empty() {
        let ctx = HostContext {
            kernel_name: Some("Linux".to_string()),
            cpu_model: Some("Example CPU".to_string()),
            ..HostContext::default()
        };
        assert_eq!(HostContext::diff(&ctx, &ctx), "");
    }

    /// A single changed field produces a single `key: before →
    /// after` line; unchanged fields are omitted so the operator
    /// sees only what shifted.
    #[test]
    fn diff_single_field_surfaces_only_that_field() {
        let a = HostContext {
            cmdline: Some("preempt=lazy".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            ..HostContext::default()
        };
        let b = HostContext {
            cmdline: Some("preempt=full".to_string()),
            kernel_release: Some("6.11.0".to_string()),
            ..HostContext::default()
        };
        let out = HostContext::diff(&a, &b);
        assert!(
            out.contains("cmdline: preempt=lazy → preempt=full"),
            "cmdline change must appear: {out}",
        );
        assert!(
            !out.contains("kernel_release"),
            "unchanged kernel_release must not appear: {out}",
        );
    }

    /// `None → Some(..)` renders as `(unknown) → <value>` so a
    /// field that starts appearing in a newer run is not confused
    /// with a field that was already present.
    #[test]
    fn diff_none_to_some_shows_unknown_arrow() {
        let a = HostContext::default();
        let b = HostContext {
            kernel_name: Some("Linux".to_string()),
            ..HostContext::default()
        };
        let out = HostContext::diff(&a, &b);
        assert!(
            out.contains("kernel_name: (unknown) → Linux"),
            "(unknown) → Linux must appear: {out}",
        );
    }

    /// Per-key `sched_tunables` diff: identical keys are omitted,
    /// changed keys show old → new, and keys present on only one
    /// side render as `(absent)`.
    #[test]
    fn diff_sched_tunables_per_key() {
        let mut am = BTreeMap::new();
        am.insert("sched_a".to_string(), "1".to_string());
        am.insert("sched_b".to_string(), "old".to_string());
        let mut bm = BTreeMap::new();
        bm.insert("sched_a".to_string(), "1".to_string());
        bm.insert("sched_b".to_string(), "new".to_string());
        bm.insert("sched_c".to_string(), "3".to_string());
        let a = HostContext {
            sched_tunables: Some(am),
            ..HostContext::default()
        };
        let b = HostContext {
            sched_tunables: Some(bm),
            ..HostContext::default()
        };
        let out = HostContext::diff(&a, &b);
        assert!(
            !out.contains("sched_tunables.sched_a"),
            "unchanged sched_a must not appear: {out}",
        );
        assert!(
            out.contains("sched_tunables.sched_b: old → new"),
            "changed sched_b must appear: {out}",
        );
        assert!(
            out.contains("sched_tunables.sched_c: (absent) → 3"),
            "new key sched_c must appear as (absent) → 3: {out}",
        );
    }

    /// `None vs Some(map)` at the outer `sched_tunables` level
    /// still surfaces a line — otherwise a read_dir regression
    /// would silently suppress the tunables section in compare
    /// output. The Some side carries a cardinality sentinel so
    /// the reader knows how much new data appeared.
    #[test]
    fn diff_sched_tunables_none_vs_some() {
        let mut m = BTreeMap::new();
        m.insert("sched_x".to_string(), "1".to_string());
        let a = HostContext::default();
        let b = HostContext {
            sched_tunables: Some(m),
            ..HostContext::default()
        };
        let out = HostContext::diff(&a, &b);
        assert!(
            out.contains("sched_tunables: (unknown) → (1 entry)"),
            "None → Some(1 entry) must surface cardinality: {out}",
        );
    }

    /// A field that transitions from `Some(value)` → `None`
    /// (for example the kernel `cmdline` becoming unreadable in a
    /// later run) must surface as `<old> → (unknown)` so an
    /// operator running `stats compare` sees the disappearance
    /// explicitly.
    #[test]
    fn diff_some_to_none_shows_arrow_unknown() {
        let a = HostContext {
            kernel_release: Some("6.11.0".to_string()),
            ..HostContext::default()
        };
        let b = HostContext::default();
        let out = HostContext::diff(&a, &b);
        assert!(
            out.contains("kernel_release: 6.11.0 → (unknown)"),
            "Some → None must surface as <value> → (unknown): {out}",
        );
    }

    /// A per-key `sched_tunables` entry that exists in `a` but
    /// not in `b` renders as `<value> → (absent)`, the mirror of
    /// the `(absent) → <value>` case. Without this, a tunable
    /// that was being overridden in the older run and reverted to
    /// default in the newer run would silently disappear from the
    /// diff.
    #[test]
    fn diff_sched_tunables_key_removed() {
        let mut am = BTreeMap::new();
        am.insert("sched_a".to_string(), "1".to_string());
        am.insert("sched_b".to_string(), "2".to_string());
        let mut bm = BTreeMap::new();
        bm.insert("sched_a".to_string(), "1".to_string());
        let a = HostContext {
            sched_tunables: Some(am),
            ..HostContext::default()
        };
        let b = HostContext {
            sched_tunables: Some(bm),
            ..HostContext::default()
        };
        let out = HostContext::diff(&a, &b);
        assert!(
            !out.contains("sched_tunables.sched_a"),
            "unchanged sched_a must not appear: {out}",
        );
        assert!(
            out.contains("sched_tunables.sched_b: 2 → (absent)"),
            "removed sched_b must surface as <value> → (absent): {out}",
        );
    }

    // ------------------------------------------------------------
    // read_trimmed_sysfs — IO-wrapper edge cases. `parse_trimmed`
    // is tested separately; these tests exercise the `read_to_string
    // + parse_trimmed` chain end-to-end against real files via
    // `tempfile::NamedTempFile`.
    // ------------------------------------------------------------

    /// Nonexistent path → `None`. `read_to_string` returns `ENOENT`;
    /// `.ok()` converts to `None`; the `and_then` short-circuits.
    /// Guards against a regression that re-introduces `unwrap()`
    /// on the read result.
    ///
    /// The "nonexistent" path is constructed under a fresh
    /// `TempDir` (unique per invocation, auto-cleaned on drop)
    /// rather than a fixed name under `std::env::temp_dir()` —
    /// the latter would race with a concurrent run of the same
    /// test from a parallel test runner or cargo-watch session.
    #[test]
    fn read_trimmed_sysfs_missing_file_returns_none() {
        let scratch = tempfile::TempDir::new().expect("create scratch temp dir");
        let missing = scratch.path().join("nonexistent-target");
        assert!(read_trimmed_sysfs(&missing).is_none());
    }

    /// Whitespace-only file → `None`. `str::trim` leaves the empty
    /// string; `parse_trimmed` catches that and returns `None`.
    /// A kernel sysfs file that transiently reads as just `"\n"`
    /// must map to `None` rather than `Some("")`.
    #[test]
    fn read_trimmed_sysfs_whitespace_only_returns_none() {
        let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
        std::io::Write::write_all(&mut f, b"  \n\t \r\n  ").expect("write whitespace");
        assert!(read_trimmed_sysfs(f.path()).is_none());
    }

    /// Populated file → `Some(trimmed)`. Exercises the full IO +
    /// trim chain against a realistic sysfs shape (`value\n`).
    #[test]
    fn read_trimmed_sysfs_populated_file_returns_trimmed_content() {
        let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
        std::io::Write::write_all(&mut f, b"madvise\n").expect("write content");
        assert_eq!(read_trimmed_sysfs(f.path()).as_deref(), Some("madvise"));
    }

    /// Bracketed-selection THP shape round-trips through the IO
    /// wrapper. `parse_trimmed_preserves_bracketed_thp` already pins
    /// the pure trim-preservation; this test walks the whole IO +
    /// trim chain so a regression that double-trims or parses the
    /// brackets is caught at the wrapper boundary.
    #[test]
    fn read_trimmed_sysfs_preserves_thp_bracket_selection() {
        let mut f = tempfile::NamedTempFile::new().expect("create tempfile");
        std::io::Write::write_all(&mut f, b"always [madvise] never\n").expect("write");
        assert_eq!(
            read_trimmed_sysfs(f.path()).as_deref(),
            Some("always [madvise] never"),
        );
    }

    /// `read_sched_tunables_from` happy path: only regular files whose
    /// names start with `sched_` are included, non-prefix files are
    /// ignored, subdirectories are filtered by the `is_file` guard,
    /// and each value is trimmed by the existing `read_trimmed_sysfs`
    /// hop. Drives the path-parameterized seam against a controlled
    /// tempdir so the walk + filter + read pipeline is exercised end
    /// to end without touching `/proc`.
    #[test]
    fn read_sched_tunables_from_filters_and_trims() {
        let tmp = tempfile::TempDir::new().expect("create tempdir");
        let dir = tmp.path();
        std::fs::write(dir.join("sched_foo"), b"42\n").expect("write sched_foo");
        std::fs::write(dir.join("sched_bar"), b"1\n").expect("write sched_bar");
        // Non-`sched_` prefix — filtered out by the name check.
        std::fs::write(dir.join("not_sched_baz"), b"99\n").expect("write not_sched_baz");
        // Subdirectory whose name starts with `sched_` — filtered
        // out by the `is_file` guard.
        std::fs::create_dir(dir.join("sched_subdir")).expect("create sched_subdir");

        let out = read_sched_tunables_from(dir).expect("walk must succeed on readable dir");
        assert_eq!(out.len(), 2, "expected only two sched_* files, got {out:?}");
        assert_eq!(out.get("sched_foo").map(String::as_str), Some("42"));
        assert_eq!(out.get("sched_bar").map(String::as_str), Some("1"));
        assert!(
            !out.contains_key("not_sched_baz"),
            "non-sched_ prefix must be filtered out"
        );
        assert!(
            !out.contains_key("sched_subdir"),
            "subdirectories must be filtered by is_file"
        );
    }

    // ------------------------------------------------------------
    // count_numa_nodes_in_topology — UMA fallback + sparse / dense
    // dedup paths. Pure logic; the IO-reading wrapper
    // `count_numa_nodes_via_topology` is left untested here (that
    // was the tradeoff in the seam extraction — the IO path just
    // delegates to this helper after a sysfs probe).
    // ------------------------------------------------------------

    /// Empty `cpu_to_node` map → `1`. This is the UMA fallback
    /// branch: every Linux system has at least one NUMA node, so
    /// returning zero would misrepresent the topology. Guarded
    /// against a refactor that removes the `is_empty` check and
    /// lets `BTreeSet::len()` return 0.
    #[test]
    fn count_numa_nodes_in_topology_empty_returns_one() {
        let topo = crate::vmm::host_topology::HostTopology {
            llc_groups: Vec::new(),
            online_cpus: Vec::new(),
            cpu_to_node: std::collections::HashMap::new(),
        };
        assert_eq!(count_numa_nodes_in_topology(&topo), 1);
    }

    /// Single-node: every CPU maps to node 0. Dedup produces a
    /// set with one entry. Pinned separately from the empty-map
    /// case because the code path is different — `is_empty` is
    /// false here, so the `BTreeSet` branch runs and must still
    /// return 1.
    #[test]
    fn count_numa_nodes_in_topology_single_node() {
        let mut cpu_to_node = std::collections::HashMap::new();
        for cpu in 0..8 {
            cpu_to_node.insert(cpu, 0);
        }
        let topo = crate::vmm::host_topology::HostTopology {
            llc_groups: Vec::new(),
            online_cpus: (0..8).collect(),
            cpu_to_node,
        };
        assert_eq!(count_numa_nodes_in_topology(&topo), 1);
    }

    /// Two-node split (CPUs 0-3 → node 0, CPUs 4-7 → node 1).
    /// The common post-fix case a sidecar host-context snapshot
    /// needs to report correctly.
    #[test]
    fn count_numa_nodes_in_topology_two_nodes() {
        let mut cpu_to_node = std::collections::HashMap::new();
        for cpu in 0..4 {
            cpu_to_node.insert(cpu, 0);
        }
        for cpu in 4..8 {
            cpu_to_node.insert(cpu, 1);
        }
        let topo = crate::vmm::host_topology::HostTopology {
            llc_groups: Vec::new(),
            online_cpus: (0..8).collect(),
            cpu_to_node,
        };
        assert_eq!(count_numa_nodes_in_topology(&topo), 2);
    }

    /// Sparse node IDs — `{0, 2, 5}` with non-contiguous numbering
    /// (e.g. a CXL-host topology where some nodes are memory-only).
    /// `BTreeSet::from_iter` dedups on insert, so the count is the
    /// number of distinct IDs, NOT `max_id + 1`.
    #[test]
    fn count_numa_nodes_in_topology_sparse_ids() {
        let mut cpu_to_node = std::collections::HashMap::new();
        cpu_to_node.insert(0, 0);
        cpu_to_node.insert(1, 2);
        cpu_to_node.insert(2, 5);
        cpu_to_node.insert(3, 0); // duplicate of cpu 0's node
        let topo = crate::vmm::host_topology::HostTopology {
            llc_groups: Vec::new(),
            online_cpus: vec![0, 1, 2, 3],
            cpu_to_node,
        };
        assert_eq!(
            count_numa_nodes_in_topology(&topo),
            3,
            "sparse IDs {{0, 2, 5}} must count as 3, not max_id+1",
        );
    }

    /// Pin both caching invariants with a direct call-count probe:
    ///
    /// 1. `compute_static_host_info` runs at MOST once per process
    ///    — the `OnceLock::get_or_init` contract. Across N repeated
    ///    `collect_host_context()` calls, the delta must stay ≤ 1
    ///    (the first call from-cold executes the closure; every
    ///    subsequent call hits the cache).
    /// 2. `read_meminfo` runs EXACTLY N times across N calls — one
    ///    read per `collect_host_context` invocation, regardless of
    ///    cache state. The cold path no longer double-reads
    ///    meminfo (the dedup shares the parsed struct between the
    ///    init closure and the per-call path); this test pins the
    ///    dedup so a regression that re-adds a second read inside
    ///    `compute_static_host_info` trips the assertion.
    /// 3. Cold-init anchor: if `STATIC_HOST_INFO` was not yet
    ///    populated when the test started, exactly one
    ///    `compute_static_host_info` call must run during this test.
    ///
    /// Deltas (`load() - before-snapshot`) absorb pre-population
    /// from sibling tests: the test is robust to execution order.
    ///
    /// # Nextest subprocess-isolation assumption
    ///
    /// The before-snapshot / after-delta arithmetic assumes no
    /// **other** concurrent test inside the same process mutates
    /// the counters mid-run. ktstr's test suite is driven by
    /// `cargo nextest run`, which spawns a fresh subprocess per
    /// test by default — so each test sees a freshly-initialized
    /// process with its own counters, and the only writers to
    /// `STATIC_INIT_CALLS` / `MEMINFO_READ_CALLS` during this
    /// test's window are its own five `collect_host_context()`
    /// calls. Under `cargo test` (shared-process, thread-parallel)
    /// a sibling test calling `collect_host_context()` in parallel
    /// would skew the deltas. The CLAUDE.md rule "always use
    /// `cargo nextest run`, never `cargo test`" is what keeps this
    /// assumption load-bearing; a future migration away from
    /// nextest would need to re-assess this test's atomic-delta
    /// scheme (likely via per-test-thread counters or a mutex
    /// around the whole call window).
    #[cfg(target_os = "linux")]
    #[test]
    fn collect_host_context_call_counts_match_caching_invariants() {
        use std::sync::atomic::Ordering;
        const N: usize = 5;

        let static_was_populated_pre = STATIC_HOST_INFO.get().is_some();
        let init_before = STATIC_INIT_CALLS.load(Ordering::Relaxed);
        let meminfo_before = MEMINFO_READ_CALLS.load(Ordering::Relaxed);

        for _ in 0..N {
            let _ = collect_host_context();
        }

        let init_delta = STATIC_INIT_CALLS.load(Ordering::Relaxed) - init_before;
        let meminfo_delta = MEMINFO_READ_CALLS.load(Ordering::Relaxed) - meminfo_before;

        assert!(
            init_delta <= 1,
            "compute_static_host_info must run at most once across {N} collect_host_context calls, ran {init_delta}",
        );
        assert_eq!(
            meminfo_delta, N,
            "read_meminfo must run exactly {N} times across {N} collect_host_context calls, ran {meminfo_delta} — the dedup would regress if this trips",
        );

        if !static_was_populated_pre {
            assert_eq!(
                init_delta, 1,
                "cold-init anchor: compute_static_host_info must run exactly once on the populate path, not {init_delta}",
            );
        }

        assert!(
            STATIC_HOST_INFO.get().is_some(),
            "STATIC_HOST_INFO must be populated after at least one collect_host_context call",
        );
    }

    /// `count_numa_nodes_in_topology` counts the cardinality of
    /// distinct values in [`HostTopology::cpu_to_node`] — the
    /// "CPU-bearing nodes" count, and nothing else. Memory-only
    /// NUMA nodes (CXL / Intel Optane / persistent memory tiers)
    /// have no CPUs by definition and are structurally
    /// unrepresentable in the current [`HostTopology`]: the struct
    /// has no "all nodes" field populated from
    /// `/sys/devices/system/node/*` independently of the CPU
    /// mapping. From the counter's perspective a memory-only node
    /// and a non-existent node are indistinguishable — both are
    /// simply missing from `cpu_to_node`.
    ///
    /// **What this test pins is narrow**: the counter's only
    /// source is `cpu_to_node`. A regression that added a parallel
    /// source (e.g. an `all_nodes: Vec<u32>` field fed from
    /// `/sys/...`) and summed it into the count would inflate the
    /// "CPUs per node" denominator for every downstream consumer —
    /// cgroup cpuset assignments, scheduler placement, and the
    /// NUMA memory-policy validator in
    /// [`ops::validate_mempolicy_cpuset`] — all of which are
    /// CPU-keyed and would quietly break under an inflated count.
    /// The exclusion is therefore by construction (the parallel
    /// field doesn't exist), not by active filtering.
    ///
    /// Fixture: 4 CPUs mapped across nodes 0 and 1, so
    /// `cpu_to_node.values()` has 2 distinct entries. The assertion
    /// demands `count == 2`. A future impl that introduced a second
    /// source must either (a) audit all CPU-keyed consumers at the
    /// same time and update this doc to match, or (b) leave this
    /// counter cpu_to_node-driven and add a separate
    /// `count_all_nodes_including_memory_only` helper with its own
    /// coverage. The inline comment at the "absent node id" line
    /// carries the same contract for readers browsing the test
    /// body.
    #[test]
    fn count_numa_nodes_in_topology_excludes_memory_only_nodes() {
        let mut cpu_to_node = std::collections::HashMap::new();
        cpu_to_node.insert(0, 0);
        cpu_to_node.insert(1, 0);
        cpu_to_node.insert(2, 1);
        cpu_to_node.insert(3, 1);
        // Node id 2 intentionally absent from cpu_to_node — it is
        // the memory-only tier under test. The function has no
        // other channel to learn about node 2, so a future change
        // that adds awareness of memory-only nodes (via a separate
        // field) would need to opt-in explicitly — this test pins
        // the current silent-exclusion contract.
        let topo = crate::vmm::host_topology::HostTopology {
            llc_groups: Vec::new(),
            online_cpus: vec![0, 1, 2, 3],
            cpu_to_node,
        };
        assert_eq!(
            count_numa_nodes_in_topology(&topo),
            2,
            "memory-only nodes must not inflate the CPU-bearing node count",
        );
    }

}
