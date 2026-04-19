//! CLI argument extraction for the ktstr dispatch path.
//!
//! The ktstr runtime hijacks its host binary's argv in two places — the
//! `#[ctor]` early-dispatch (host and guest) and nextest's `--exact`
//! invocation — so it needs a tiny, dependency-free parser that can
//! pick named values out of `std::env::args()` without getting in the
//! way of the harness's own flag handling.
//!
//! All helpers accept a `&[String]` slice and return either the first
//! matching value or `None`. They are intentionally lenient: they only
//! recognize the `--ktstr-*=VALUE` form (or, for `--ktstr-test-fn`,
//! also the space-separated form) and ignore unknown flags entirely.
//! That keeps the dispatch path inert for binaries that aren't built
//! against ktstr.
//!
//! [`resolve_cgroup_root`] is the one outlier: it sources the path from
//! the initramfs-mounted `/sched_args` file first, then falls back to
//! the process argv. Used only from guest-side dispatch to derive the
//! cgroup manager root for the running test.

/// Extract the test function name from `--ktstr-test-fn=NAME` or
/// `--ktstr-test-fn NAME` in the argument list.
pub(crate) fn extract_test_fn_arg(args: &[String]) -> Option<&str> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if let Some(val) = a.strip_prefix("--ktstr-test-fn=") {
            return Some(val);
        }
        if a == "--ktstr-test-fn" {
            return iter.next().map(|s| s.as_str());
        }
    }
    None
}

/// Extract `--ktstr-probe-stack=func1,func2,...` from the argument list.
pub(crate) fn extract_probe_stack_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-probe-stack=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--ktstr-topo=NnNlNcNt` (or legacy `NsNcNt`) from the argument list.
pub(crate) fn extract_topo_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-topo=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Extract `--ktstr-flags=borrow,rebal` from the argument list.
pub(crate) fn extract_flags_arg(args: &[String]) -> Option<Vec<String>> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-flags=")
            && !val.is_empty()
        {
            return Some(val.split(',').map(|s| s.to_string()).collect());
        }
    }
    None
}

/// Extract `--ktstr-work-type=NAME` from the argument list.
pub(crate) fn extract_work_type_arg(args: &[String]) -> Option<String> {
    for a in args {
        if let Some(val) = a.strip_prefix("--ktstr-work-type=")
            && !val.is_empty()
        {
            return Some(val.to_string());
        }
    }
    None
}

/// Derive the CgroupManager root path for guest-side dispatch.
///
/// Reads `/sched_args` to find `--cell-parent-cgroup <path>`. When
/// found, constructs `/sys/fs/cgroup{path}`. Falls back to
/// `/sys/fs/cgroup/ktstr` when the arg is absent.
pub(crate) fn resolve_cgroup_root(args: &[String]) -> String {
    // Check guest args for --cell-parent-cgroup (passed via sched_args
    // which are written to /sched_args in the initramfs).
    let sched_args = std::fs::read_to_string("/sched_args").unwrap_or_default();
    let parts: Vec<&str> = sched_args.split_whitespace().collect();
    for i in 0..parts.len() {
        if parts[i] == "--cell-parent-cgroup"
            && let Some(&path) = parts.get(i + 1)
        {
            return format!("/sys/fs/cgroup{path}");
        }
    }
    // Also check the process args in case --cell-parent-cgroup was
    // passed directly (e.g., via extra_sched_args on the test entry).
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == "--cell-parent-cgroup"
            && let Some(path) = iter.next()
        {
            return format!("/sys/fs/cgroup{path}");
        }
    }
    "/sys/fs/cgroup/ktstr".to_string()
}
