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

use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Host-level runtime state snapshot attached to each
/// [`SidecarResult`](crate::test_support::SidecarResult). Every
/// field is optional so a partial read (missing /proc entry,
/// permission denied, parse failure) still records the fields that
/// did succeed instead of dropping the whole snapshot.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    pub hugepagesize_kb: Option<u64>,
    /// Active THP policy — content of
    /// `/sys/kernel/mm/transparent_hugepage/enabled` with the
    /// bracketed selection preserved verbatim (e.g.
    /// `"always [madvise] never"`). Stored as-read rather than
    /// parsed because the bracket is the meaningful part and downstream
    /// tooling may want the full menu too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thp_enabled: Option<String>,
    /// Active THP defrag policy — content of
    /// `/sys/kernel/mm/transparent_hugepage/defrag`, bracket
    /// preserved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thp_defrag: Option<String>,
    /// `/proc/sys/kernel/sched_*` tunables. Keys are the leaf
    /// basename (e.g. `sched_migration_cost_ns`); values are the
    /// single-line content trimmed of leading and trailing
    /// whitespace. Empty map when the directory is unreadable; the
    /// containing Option is `None` only when the listing itself
    /// fails.
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
    pub uname_sysname: Option<String>,
    /// Kernel release — `uname.release` (e.g. `"6.11.0-rc3"`).
    /// The full `/proc/version` banner is NOT captured because it
    /// embeds the build host + gcc version string, which is
    /// environment leakage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uname_release: Option<String>,
    /// Machine architecture — `uname.machine` (e.g. `"x86_64"`,
    /// `"aarch64"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uname_machine: Option<String>,
    /// `/proc/cmdline` verbatim (trimmed of leading and trailing
    /// whitespace). Captures boot-time parameters that materially
    /// affect scheduler behavior — `preempt=`, `isolcpus=`,
    /// `nohz_full=`, `mitigations=`, hugepage reservations,
    /// `transparent_hugepage=`, and others. Stored as a single
    /// string because any split-into-pairs parser loses the
    /// quoted-value and flag-only variants the kernel accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_cmdline: Option<String>,
}

/// Static-fields cache. These values do not change for the lifetime
/// of the process (CPU identity, total installed memory, hugepage
/// size chosen at boot, NUMA count, uname triple), so walking
/// `/proc` and `/sys` for them once and reusing the result avoids
/// repeated syscalls on every sidecar write. Dynamic fields
/// (sched_tunables, hugepages_total, hugepages_free, thp_enabled,
/// thp_defrag, host_cmdline) are NOT cached — they can shift
/// between tests via sysctl, hugepage reservation, THP policy flip,
/// or live kexec, and a cached snapshot would hide that change.
#[derive(Clone)]
struct StaticHostInfo {
    cpu_model: Option<String>,
    cpu_vendor: Option<String>,
    total_memory_kb: Option<u64>,
    hugepagesize_kb: Option<u64>,
    numa_nodes: Option<usize>,
    uname_sysname: Option<String>,
    uname_release: Option<String>,
    uname_machine: Option<String>,
}

static STATIC_HOST_INFO: OnceLock<StaticHostInfo> = OnceLock::new();

/// Capture the host context. Static fields are collected once
/// and cached; dynamic fields are re-read on every call so
/// intra-run sysctl / hugepage / THP changes are reflected.
///
/// Every sub-read is fallible; individual failures leave the
/// corresponding field `None` and the rest of the context
/// proceeds. A host with `/proc` entirely unmounted (extreme test
/// fixture) returns a `HostContext` with every field `None` —
/// which serializes to an empty JSON object and distinguishes
/// "collection attempted, nothing known" from "collection not
/// attempted" (represented at the enclosing `Option<HostContext>`
/// layer on [`SidecarResult`](crate::test_support::SidecarResult)).
pub(crate) fn collect_host_context() -> HostContext {
    let static_info = STATIC_HOST_INFO.get_or_init(compute_static_host_info).clone();
    let meminfo = read_meminfo();
    HostContext {
        cpu_model: static_info.cpu_model,
        cpu_vendor: static_info.cpu_vendor,
        total_memory_kb: static_info.total_memory_kb,
        hugepages_total: meminfo.hugepages_total,
        hugepages_free: meminfo.hugepages_free,
        hugepagesize_kb: static_info.hugepagesize_kb,
        thp_enabled: read_trimmed_sysfs("/sys/kernel/mm/transparent_hugepage/enabled"),
        thp_defrag: read_trimmed_sysfs("/sys/kernel/mm/transparent_hugepage/defrag"),
        sched_tunables: read_sched_tunables(),
        numa_nodes: static_info.numa_nodes,
        uname_sysname: static_info.uname_sysname,
        uname_release: static_info.uname_release,
        uname_machine: static_info.uname_machine,
        host_cmdline: read_trimmed_sysfs("/proc/cmdline"),
    }
}

/// Populate the static-fields cache on first access. Reads
/// `/proc/cpuinfo` (CPU identity), `/proc/meminfo` (MemTotal +
/// Hugepagesize), the host NUMA topology, and a single `uname()`
/// call.
fn compute_static_host_info() -> StaticHostInfo {
    let (cpu_model, cpu_vendor) = read_cpuinfo_identity();
    let meminfo = read_meminfo();
    let u = rustix::system::uname();
    StaticHostInfo {
        cpu_model,
        cpu_vendor,
        total_memory_kb: meminfo.mem_total_kb,
        hugepagesize_kb: meminfo.hugepagesize_kb,
        numa_nodes: count_numa_nodes_via_topology(),
        uname_sysname: u.sysname().to_str().ok().map(|s| s.to_string()),
        uname_release: u.release().to_str().ok().map(|s| s.to_string()),
        uname_machine: u.machine().to_str().ok().map(|s| s.to_string()),
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
    hugepagesize_kb: Option<u64>,
}

/// Read `/proc/meminfo` and extract the four fields the host
/// context needs. Thin I/O wrapper; parsing lives in
/// [`parse_meminfo`] so it can be unit-tested with synthetic
/// fixtures.
fn read_meminfo() -> MeminfoFields {
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
            "Hugepagesize" => out.hugepagesize_kb = Some(n),
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
/// `sched_` and record each as `basename → content`. Skips
/// directories (the kernel exposes no sched_* directories today but
/// guarding keeps behavior defined if that changes) and entries
/// whose content is not readable as UTF-8.
///
/// Returns `None` only when the directory listing itself fails
/// (unreadable /proc); an empty map is a valid result (kernel
/// without any sched_* tunables — e.g. a very old or unusual
/// build).
fn read_sched_tunables() -> Option<BTreeMap<String, String>> {
    let entries = std::fs::read_dir("/proc/sys/kernel").ok()?;
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
    if topo.cpu_to_node.is_empty() {
        return Some(1);
    }
    let mut nodes: Vec<usize> = topo.cpu_to_node.values().copied().collect();
    nodes.sort_unstable();
    nodes.dedup();
    Some(nodes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host-context reads are host-dependent: we assert the
    /// collector returns SOMETHING, not specific values. On Linux
    /// CI the uname fields at least should populate.
    #[test]
    fn collect_host_context_returns_populated_struct_on_linux() {
        let ctx = collect_host_context();
        // uname is always readable on Linux (it's a syscall, no
        // filesystem dependency), so these three must populate.
        assert_eq!(ctx.uname_sysname.as_deref(), Some("Linux"));
        assert!(ctx.uname_release.is_some(), "uname release present");
        assert!(ctx.uname_machine.is_some(), "uname machine present");
    }

    /// `/proc/cmdline` is always readable on a running Linux system
    /// (the kernel exposes it unconditionally). The capture is
    /// verbatim — `read_trimmed_sysfs` trims leading/trailing
    /// whitespace and returns `None` only when the read fails or
    /// the file is empty after trim. No token filtering is applied.
    /// The assertion only requires that, when populated, the
    /// captured value is trimmed (no stray whitespace).
    #[test]
    fn collect_host_context_captures_host_cmdline_on_linux() {
        let ctx = collect_host_context();
        if let Some(cmdline) = ctx.host_cmdline.as_ref() {
            assert!(!cmdline.is_empty(), "populated cmdline must not be empty");
            assert_eq!(cmdline.as_str(), cmdline.trim());
        }
    }

    /// Stability regression — repeated calls return equal
    /// `HostContext` values. Proves stability across calls:
    /// static fields come from the cache, dynamic fields match
    /// between back-to-back reads on a quiescent host.
    #[test]
    fn collect_host_context_is_stable_across_calls() {
        let a = collect_host_context();
        let b = collect_host_context();
        assert_eq!(a.uname_sysname, b.uname_sysname);
        assert_eq!(a.uname_release, b.uname_release);
        assert_eq!(a.uname_machine, b.uname_machine);
        assert_eq!(a.cpu_model, b.cpu_model);
        assert_eq!(a.cpu_vendor, b.cpu_vendor);
        assert_eq!(a.host_cmdline, b.host_cmdline);
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
        assert!(decoded.uname_sysname.is_none());
        assert!(decoded.host_cmdline.is_none());
    }

    /// Populated host context round-trips — every field's
    /// serde_json shape is stable. Asserts ALL fields, not just a
    /// handful, so a future field addition or serde-attr change
    /// that breaks a subset of the shape is caught by this test.
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
            hugepagesize_kb: Some(2048),
            thp_enabled: Some("always [madvise] never".to_string()),
            thp_defrag: Some("[always] defer defer+madvise madvise never".to_string()),
            sched_tunables: Some(tunables.clone()),
            numa_nodes: Some(2),
            uname_sysname: Some("Linux".to_string()),
            uname_release: Some("6.11.0".to_string()),
            uname_machine: Some("x86_64".to_string()),
            host_cmdline: Some("preempt=lazy transparent_hugepage=madvise".to_string()),
        };
        let json = serde_json::to_string(&ctx).expect("serialize");
        let decoded: HostContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.cpu_model.as_deref(), Some("Example CPU"));
        assert_eq!(decoded.cpu_vendor.as_deref(), Some("GenuineExample"));
        assert_eq!(decoded.total_memory_kb, Some(16_384_000));
        assert_eq!(decoded.hugepages_total, Some(0));
        assert_eq!(decoded.hugepages_free, Some(0));
        assert_eq!(decoded.hugepagesize_kb, Some(2048));
        assert_eq!(decoded.thp_enabled.as_deref(), Some("always [madvise] never"));
        assert_eq!(
            decoded.thp_defrag.as_deref(),
            Some("[always] defer defer+madvise madvise never"),
        );
        assert_eq!(decoded.sched_tunables.as_ref().unwrap(), &tunables);
        assert_eq!(decoded.numa_nodes, Some(2));
        assert_eq!(decoded.uname_sysname.as_deref(), Some("Linux"));
        assert_eq!(decoded.uname_release.as_deref(), Some("6.11.0"));
        assert_eq!(decoded.uname_machine.as_deref(), Some("x86_64"));
        assert_eq!(
            decoded.host_cmdline.as_deref(),
            Some("preempt=lazy transparent_hugepage=madvise"),
        );
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
        assert_eq!(out.hugepagesize_kb, Some(2048));
    }

    #[test]
    fn parse_meminfo_empty_input() {
        let out = parse_meminfo("");
        assert!(out.mem_total_kb.is_none());
        assert!(out.hugepages_total.is_none());
        assert!(out.hugepages_free.is_none());
        assert!(out.hugepagesize_kb.is_none());
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
        assert!(out.hugepagesize_kb.is_none());
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
        assert_eq!(out.hugepagesize_kb, Some(2048));
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
        assert_eq!(out.hugepagesize_kb, Some(2048));
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
            out.hugepagesize_kb,
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

}
