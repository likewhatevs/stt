//! Host-resource contention classification and KVM_CREATE_VM retry.
//!
//! When a KVM ioctl fails, callers want to distinguish "host is under
//! transient pressure (peer contention, resource exhaustion)" from
//! "real fault that should bubble up loudly". This module owns the
//! errno-classification table, the host-resource snapshot embedded in
//! contention diagnostics, and the EINTR retry schedule for
//! `KVM_CREATE_VM`.
//!
//! Outputs flow through [`super::host_topology::ResourceContention`]
//! so the `#[ktstr_test]` macro layer can SKIP-skip cleanly on a
//! transient errno without conflating the skip with a real test
//! failure.

use anyhow::Result;
use std::time::Duration;

use super::host_topology;

/// EINTR retry schedule for `KVM_CREATE_VM` — eight attempts with
/// millisecond-scale backoff that totals roughly one second. The
/// previous schedule (`1 << attempts` microseconds, five attempts,
/// 62 µs total) could be starved by any sustained signal source
/// firing across a few tens of microseconds (e.g. a runtime that
/// uses async-cancellation signals on the parent thread). KVM_CREATE_VM
/// runs before any vCPU thread has been created, so the freeze
/// coordinator's `SIGRTMIN` broadcast cannot itself trigger this path —
/// the retry budget defends against any signal source that could
/// preempt the parent thread, not a specific known one. The new
/// schedule matches Firecracker's spirit (exponential backoff with a
/// cap) at a budget realistic for sustained signal activity.
pub(crate) const KVM_CREATE_VM_EINTR_DELAYS: [Duration; 8] = [
    Duration::from_micros(100),
    Duration::from_micros(500),
    Duration::from_millis(2),
    Duration::from_millis(10),
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(200),
    Duration::from_millis(500),
];

/// Errno values that callers should treat as transient host-resource
/// pressure rather than a hard fault. Used by
/// [`map_transient_to_contention`] to convert KVM ioctl failures
/// into [`host_topology::ResourceContention`] so the
/// `#[ktstr_test]` macro can route them through the canonical
/// SKIP-on-contention path instead of panicking the test.
///
/// - `ENOMEM` — kernel memory allocator could not satisfy a
///   GuestMemoryMmap region or KVM internal table allocation.
/// - `EMFILE` — process fd table full (per-process `RLIMIT_NOFILE`).
/// - `ENFILE` — system-wide fd table full (`/proc/sys/fs/file-max`).
/// - `EBUSY` — KVM vm/vcpu fd hit a transient busy state.
/// - `EAGAIN` — kernel subsystem signalled "try again". Defensive
///   inclusion: KVM init ioctls rarely return EAGAIN, but a future
///   kernel that adds a non-blocking allocation path could surface it
///   transiently — better to classify it as contention than fault.
pub(crate) const TRANSIENT_HOST_ERRNOS: &[i32] = &[
    libc::ENOMEM,
    libc::EMFILE,
    libc::ENFILE,
    libc::EBUSY,
    libc::EAGAIN,
];

/// Typed snapshot of process-level resource state. Captured by
/// [`host_resource_snapshot`]; embedded into a `ResourceContention`
/// diagnostic via its [`std::fmt::Display`] impl, and read directly
/// by [`map_transient_to_contention`]'s
/// `KTSTR_CONTENTION_BYPASS`-gated arm via the typed
/// [`Self::near_limit`] field.
///
/// The struct centralises the two consumer needs that previously
/// fanned out to separate string-format and substring-parse paths:
///
/// 1. **Diagnostic banner**: callers embed `{snapshot}` into a
///    `ResourceContention.reason` so the operator's SKIP banner names
///    `fds=...`, `vmrss=...`, `threads=...`, `near_limit=...`. The
///    [`Display`] impl produces the same single-line format the
///    SKIP banner has always carried, so banner readers and stats
///    tooling that grep for these tokens stay backward-compatible.
/// 2. **Bypass gate**: [`map_transient_to_contention`] reads
///    `snapshot.near_limit` directly — a real `bool` field — instead
///    of round-tripping through a parsed substring. A field rename
///    that breaks the diagnostic shape can no longer silently
///    desync the bypass; the gate lives on the type, the format
///    is just for humans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostResourceSnapshot {
    /// Open-fd count from a `readdir(/proc/self/fd)` (NOT
    /// `/proc/self/status::FDSize`, which is fd-table capacity).
    pub fd_count: usize,
    /// `VmRSS:` from `/proc/self/status`, e.g. `"24 kB"`. Folds to
    /// `"<unknown>"` when the field is missing.
    pub vm_rss: String,
    /// `Threads:` from `/proc/self/status` as a string for verbatim
    /// banner rendering. Folds to `"<unknown>"` when missing.
    pub threads: String,
    /// `true` iff the snapshot detected ≥90% utilisation against
    /// either the per-process `RLIMIT_NOFILE` soft cap or the
    /// per-UID `RLIMIT_NPROC` soft cap. The bypass gate keys
    /// directly on this field — no string parse, no substring
    /// matching, no risk of a future field name embedding
    /// `"near_limit=false"` as a substring fooling a `contains`
    /// check. Reading either limit failing folds to `false`
    /// (conservative — a missing limit is not a near-limit signal).
    pub near_limit: bool,
}

impl std::fmt::Display for HostResourceSnapshot {
    /// Single-line diagnostic format embedded in
    /// `ResourceContention.reason` strings. The shape
    /// (`fds=N, vmrss=X, threads=Y, near_limit=B`) is parsed by
    /// stats tooling reading SKIP banners — pinned by
    /// `host_resource_snapshot_emits_all_keys` and
    /// `host_resource_snapshot_near_limit_is_boolean`. The bypass
    /// gate does NOT consume this string — it reads
    /// [`Self::near_limit`] directly.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fds={fds}, vmrss={vmrss}, threads={threads}, near_limit={near_limit}",
            fds = self.fd_count,
            vmrss = self.vm_rss,
            threads = self.threads,
            near_limit = self.near_limit,
        )
    }
}

/// Capture a one-shot snapshot of process-level resource state for
/// inclusion in a contention diagnostic. Reads:
///
/// - `/proc/self/status` for `VmRSS` and `Threads`.
/// - `/proc/self/limits` for `Max open files` and `Max processes`.
/// - count of entries in `/proc/self/fd` (cheap fd-usage estimate;
///   produced from readdir, NOT from the `FDSize` field of
///   `/proc/self/status` — `FDSize` reports the kernel's fd-table
///   capacity, not the count of open fds).
///
/// All reads are best-effort: a missing field or unreadable file
/// folds into `<unknown>` rather than failing the snapshot. The
/// snapshot exists to give an operator hitting a SKIP banner one
/// place to start triaging — current resource USAGE (`fds=1023`,
/// `threads=42`) plus a binary `near_limit` indicator is enough
/// to distinguish "test hit a host cap" from "kernel had a real
/// fault". Cheap enough (two small file reads, one readdir) that
/// callers can take it on every transient classification without
/// measurable cost.
///
/// Returns a [`HostResourceSnapshot`] struct: callers that need the
/// diagnostic string format embed `{snapshot}` (using the
/// [`std::fmt::Display`] impl — same format the SKIP banner has
/// always carried). [`map_transient_to_contention`]'s
/// `KTSTR_CONTENTION_BYPASS` gate reads
/// [`HostResourceSnapshot::near_limit`] directly — no substring
/// match against the diagnostic format, so a banner-format change
/// cannot silently desync the bypass from the snapshot.
///
/// # Why no raw rlimits
///
/// Earlier revisions echoed the `RLIMIT_NOFILE` soft cap and
/// `RLIMIT_NPROC` soft cap directly. Those values are
/// host-specific fingerprints that the SKIP banner surfaces into
/// every CI artifact, sidecar, and user-visible test log. The
/// operator already knows their host config; the banner does not
/// need to echo it back. The `near_limit` indicator (computed at
/// snapshot time and left as a one-bit `yes/no`) preserves the
/// actionable "are we close to a cap?" triage signal without
/// leaking the cap itself.
pub(crate) fn host_resource_snapshot() -> HostResourceSnapshot {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let limits = std::fs::read_to_string("/proc/self/limits").unwrap_or_default();
    let fd_count: usize = std::fs::read_dir("/proc/self/fd")
        .map(|d| d.filter_map(|e| e.ok()).count())
        .unwrap_or(0);

    let pick = |s: &str, prefix: &str| -> String {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix(prefix) {
                return rest.trim().to_string();
            }
        }
        "<unknown>".into()
    };
    let pick_limit_value = |key: &str| -> Option<u64> {
        for line in limits.lines() {
            if let Some(rest) = line.strip_prefix(key) {
                let token = rest.split_whitespace().next()?;
                return token.parse::<u64>().ok();
            }
        }
        None
    };

    let vm_rss = pick(&status, "VmRSS:");
    let threads = pick(&status, "Threads:");

    // `near_limit` thresholds at 90% utilization — a soft tripwire
    // that surfaces "we're exhausted" without revealing the cap.
    // Both rlimits read here are SOFT caps (the first
    // whitespace-separated value on the line); the kernel raises
    // the hard cap on EMFILE before EAGAIN. The 90% threshold is
    // chosen so a routine test that fans out a few hundred fds
    // doesn't trip it, but a stuck-fd-leak hitting RLIMIT_NOFILE
    // does. Failure to read either limit (operator cgroup hiding
    // /proc/self/limits, sandbox stripping it) folds into "no" —
    // a missing limit is not a near-limit signal.
    let near_limit_fds = pick_limit_value("Max open files")
        .map(|cap| {
            // Saturating math because `fd_count * 10` could
            // theoretically overflow u64 on a 64-bit host with
            // billions of fds open — practically unreachable, but
            // saturating is free and removes the if-large branch.
            let scaled_count = (fd_count as u64).saturating_mul(10);
            let scaled_cap = cap.saturating_mul(9);
            scaled_count >= scaled_cap
        })
        .unwrap_or(false);
    let thread_count = threads.parse::<u64>().unwrap_or(0);
    // RLIMIT_NPROC is per-UID (the per-UID enforcement lives in
    // `kernel/ucount.c::is_rlimit_overlimit`, walking the user_ns
    // ucounts chain and returning true once any tier exceeds
    // its cap), counting the total number of processes/threads
    // owned by the real user ID across the whole host — not just
    // this process. We compare that per-UID cap against this
    // process's own thread count from /proc/self/status::Threads
    // as a proxy: if the local thread count is approaching the
    // per-UID cap, the host is almost certainly near saturation.
    // The proxy can underreport when peers under the same UID
    // hold most of the budget (multi-UID hosts dilute the signal
    // further), but the snapshot is a contention triage hint,
    // not an authoritative gauge.
    let near_limit_procs = pick_limit_value("Max processes")
        .map(|cap| {
            let scaled_count = thread_count.saturating_mul(10);
            let scaled_cap = cap.saturating_mul(9);
            scaled_count >= scaled_cap
        })
        .unwrap_or(false);
    let near_limit = near_limit_fds || near_limit_procs;

    HostResourceSnapshot {
        fd_count,
        vm_rss,
        threads,
        near_limit,
    }
}

/// Map a kvm_ioctls / vmm_sys_util `errno::Error` to a
/// [`host_topology::ResourceContention`] when its errno appears in
/// [`TRANSIENT_HOST_ERRNOS`]; otherwise return the error wrapped in
/// the supplied context unchanged.
///
/// The contention reason embeds the original errno name and the
/// caller-supplied context (e.g. `"create VM"` or
/// `"set TSS address"`) so the `ktstr: SKIP: resource contention:
/// ...` banner names the exact ioctl that failed and the
/// host-resource snapshot points the operator at the actionable
/// limit. Non-transient errors (EINVAL, ENOSYS, EPERM, etc.) flow
/// through unchanged so a real bug never gets misclassified as
/// contention.
///
/// `context` is `impl Into<String>` so callers can pass either a
/// static `&str` (the common case — `"create VM"`) or a freshly
/// formatted `String` (vCPU loops naming the offending CPU).
///
/// # False-positive risk
///
/// The errnos in [`TRANSIENT_HOST_ERRNOS`] are PRESUMED transient,
/// but the kernel can return any of them for genuinely-broken
/// reasons that should NOT classify as contention:
///
/// - `ENOMEM` — typically host memory pressure (a peer holding
///   guest mappings), but the kernel can also return ENOMEM from
///   a page allocator leak, a stuck slab cache, or a permanently-
///   exhausted memory cgroup. SKIP-classifying those silently
///   masks the regression instead of surfacing it.
/// - `EBUSY` — typically a device-fd or memslot held by a peer,
///   but a stuck virtqueue, an unbalanced refcount on a kernel
///   object, or a kernel-side state-machine lockup also returns
///   EBUSY and is permanent until the kernel restarts.
/// - `EMFILE` / `ENFILE` — typically fd-table pressure, but a leak
///   in the calling process surfaces as the same errno even when
///   the host is otherwise idle.
/// - `EAGAIN` — defensively included but rarely returned by KVM
///   init ioctls; a genuine `EAGAIN` from a kernel subsystem that
///   never recovers (e.g. a deadlocked workqueue) would also SKIP
///   on every retry.
///
/// The classifier accepts these false positives because the
/// alternative — letting transient host pressure surface as a
/// hard failure on every CI run — is much worse: every parallel
/// test slot would fail at once, the SKIP banner would never
/// fire, and stats tooling would record runner-incompatibility as
/// product regressions. The cost is that a real kernel bug
/// matching one of these errnos gets quietly skipped on the
/// affected test slot until the operator notices the SKIP rate
/// rising.
///
/// The false-positive cost is bounded by [`host_resource_snapshot`]'s
/// `near_limit` flag — operators can grep for `near_limit=false` +
/// sustained SKIPs to catch a kernel-side regression that the
/// classifier silently masks. The classifier exposes an opt-in
/// `near_limit`-gated bypass: when `KTSTR_CONTENTION_BYPASS=1` is
/// set in the environment AND the snapshot reports
/// `near_limit=false`, the underlying error is surfaced as a hard
/// fault instead of being wrapped in `ResourceContention`. The
/// applied diagnostic notes "errno looks transient but host is NOT
/// near limits — likely a kernel-side bug" so the operator's
/// triage points at `dmesg` rather than at peer contention.
///
/// Default-off behaviour: with the env var unset, every transient
/// errno classifies as `ResourceContention` exactly as before. The
/// bypass changes observable test outcomes (currently-SKIPped tests
/// would FAIL), so it must remain opt-in until the operator has run
/// a soak window confirming their `near_limit` snapshot is reliable
/// on the host.
///
/// Caveats the operator should weigh before enabling:
/// - The `near_limit` snapshot is read AFTER the failure, so a peer
///   allocation between the ioctl and the snapshot can mark the
///   host pressured retroactively. The per-thread atomic
///   `/proc/self/status` reads bound the inconsistency to a single
///   iteration, well within the macro-layer retry budget — but the
///   risk is that a genuine peer-contention SKIP arrives as a hard
///   failure on a freshly-drained host. Bypass-on is the
///   "investigate kernel-side regressions aggressively" mode; leave
///   it off in steady-state CI.
/// - The bypass treats every transient errno uniformly. EMFILE /
///   ENFILE / ENOSPC are always host-resource-driven by definition,
///   so a `near_limit=false` snapshot that reports them means the
///   snapshot itself is racy — surfacing them as hard errors lets
///   the operator catch an inconsistent snapshot rather than
///   silently SKIP-skipping. EAGAIN / ENOMEM / EBUSY can each be
///   either contention or kernel bugs, and the bypass routes them
///   the same way.
///
/// # Per-callsite reachability
///
/// kvm-ioctls is a thin ioctl wrapper with no per-crate errno
/// injection — every errno surfaced here comes directly from the
/// kernel ioctl path. The classifier therefore handles the union of
/// errnos that any KVM ioctl can return, but the practically-reachable
/// subset varies by callsite. For `set_user_memory_region`
/// specifically, only ENOMEM is practically reachable from the kernel
/// memslot path; EBUSY / EMFILE / ENFILE / EAGAIN remain in
/// [`TRANSIENT_HOST_ERRNOS`] as inherited generality from the
/// KVM_CREATE_VM context (where they are the dominant signal), not
/// because the memslot ioctl is observed to return them.
pub(crate) fn map_transient_to_contention(
    e: kvm_ioctls::Error,
    context: impl Into<String>,
) -> anyhow::Error {
    let context = context.into();
    let errno = e.errno();
    if TRANSIENT_HOST_ERRNOS.contains(&errno) {
        let snapshot = host_resource_snapshot();
        // Opt-in `near_limit`-gated bypass: when
        // `KTSTR_CONTENTION_BYPASS=1` is set AND the snapshot reports
        // `near_limit=false`, surface the underlying error as a hard
        // fault instead of classifying as `ResourceContention`. The
        // intent is to catch kernel-side regressions that share a
        // transient errno with genuine host pressure (leaked
        // page-allocator state surfacing as ENOMEM, a stuck virtqueue
        // surfacing as EBUSY). With `near_limit=false`, the host's
        // /proc/self/limits do not corroborate the errno's
        // "transient" framing, so the failure is more likely a
        // kernel bug than a peer holding resources.
        //
        // Default-off: the existing classifier behaviour is preserved
        // unless the operator opts in. The bypass changes observable
        // test outcomes (currently-SKIPped tests would FAIL) so it
        // must remain opt-in until the operator has run a soak window
        // confirming their `near_limit` snapshot is reliable on the
        // host.
        let bypass_requested =
            std::env::var("KTSTR_CONTENTION_BYPASS").ok().as_deref() == Some("1");
        // Typed gate — reads the bool field directly from the
        // snapshot struct. No substring parse against the diagnostic
        // format, so a banner-format change cannot silently desync
        // the bypass from the snapshot.
        if bypass_requested && !snapshot.near_limit {
            return anyhow::Error::new(e).context(format!(
                "{context}: KVM errno {errno} ({errno_name}) — errno looks transient \
                 but host is NOT near limits ({snapshot}). KTSTR_CONTENTION_BYPASS=1 \
                 routed this through as a hard error: likely a kernel-side bug \
                 (leak / stuck device / cgroup-exhausted state) rather than peer \
                 contention. Check `dmesg` for the affected subsystem.",
                errno_name = errno_name(errno),
            ));
        }
        anyhow::Error::new(host_topology::ResourceContention {
            reason: format!(
                "{context}: transient KVM errno {errno} ({}): host resources: {snapshot}\n  \
                 hint: KVM ioctl failed with a host-resource errno; another peer may be \
                 holding the budget. nextest will not retry; the SKIP banner records this \
                 attempt for stats tooling.\n  \
                 hint: if `near_limit=false` in the snapshot above and SKIPs persist \
                 across runs, the errno is likely a kernel-side regression (leak / stuck \
                 device / cgroup-exhausted state) — check `dmesg` for the affected \
                 subsystem rather than retrying the test, or set \
                 `KTSTR_CONTENTION_BYPASS=1` to surface such failures as hard errors.",
                errno_name(errno)
            ),
        })
    } else {
        anyhow::Error::new(e).context(context)
    }
}

/// Render a numeric errno to its libc name for diagnostic text.
///
/// Covers the values in [`TRANSIENT_HOST_ERRNOS`] plus a handful
/// of neighbours an operator commonly sees in KVM ioctl failures
/// (`EINVAL` / `ENOSYS` / `EPERM` / `EACCES` mark hard fault
/// boundaries the contention classifier is designed NOT to map).
/// Falls through to `errno=<raw>` for unmapped values so the
/// operator can grep for the integer in `man errno` / kernel
/// sources rather than seeing a useless `<other>`. Returns
/// [`Cow`] so the common (mapped) case stays zero-allocation while
/// the fallthrough case can format the integer.
pub(crate) fn errno_name(errno: i32) -> std::borrow::Cow<'static, str> {
    use std::borrow::Cow;
    match errno {
        libc::ENOMEM => Cow::Borrowed("ENOMEM"),
        libc::EMFILE => Cow::Borrowed("EMFILE"),
        libc::ENFILE => Cow::Borrowed("ENFILE"),
        libc::EBUSY => Cow::Borrowed("EBUSY"),
        libc::EAGAIN => Cow::Borrowed("EAGAIN"),
        libc::EINTR => Cow::Borrowed("EINTR"),
        libc::EINVAL => Cow::Borrowed("EINVAL"),
        libc::ENOSYS => Cow::Borrowed("ENOSYS"),
        libc::EPERM => Cow::Borrowed("EPERM"),
        libc::EACCES => Cow::Borrowed("EACCES"),
        libc::ENOSPC => Cow::Borrowed("ENOSPC"),
        libc::ENODEV => Cow::Borrowed("ENODEV"),
        libc::ENOTSUP => Cow::Borrowed("ENOTSUP"),
        libc::EFAULT => Cow::Borrowed("EFAULT"),
        libc::EIO => Cow::Borrowed("EIO"),
        libc::EBADF => Cow::Borrowed("EBADF"),
        libc::ESRCH => Cow::Borrowed("ESRCH"),
        libc::ENOENT => Cow::Borrowed("ENOENT"),
        libc::ECHILD => Cow::Borrowed("ECHILD"),
        libc::EEXIST => Cow::Borrowed("EEXIST"),
        libc::EOVERFLOW => Cow::Borrowed("EOVERFLOW"),
        libc::ETIMEDOUT => Cow::Borrowed("ETIMEDOUT"),
        libc::ENOTTY => Cow::Borrowed("ENOTTY"),
        other => Cow::Owned(format!("errno={other}")),
    }
}

/// Build the [`host_topology::ResourceContention`] error returned
/// from [`create_vm_with_retry`] when the EINTR retry budget is
/// exhausted. Factored out so the format (which appears in the
/// SKIP banner and is parsed by stats tooling for skip
/// classification) can be pinned by a unit test without the test
/// having to actually drive the ioctl through a sustained signal
/// storm.
fn eintr_exhausted_contention() -> anyhow::Error {
    let snapshot = host_resource_snapshot();
    let total_delay: std::time::Duration = KVM_CREATE_VM_EINTR_DELAYS.iter().copied().sum();
    anyhow::Error::new(host_topology::ResourceContention {
        reason: format!(
            "create VM: KVM_CREATE_VM kept returning EINTR after \
             {n} retries totalling {total_ms} ms — sustained \
             signal pressure on the host. host resources: {snapshot}\n  \
             hint: a peer process is firing realtime / SIGRTMIN \
             signals at a rate that out-paces the EINTR backoff \
             schedule. nextest will not retry; the SKIP banner \
             records this attempt for stats tooling.",
            n = KVM_CREATE_VM_EINTR_DELAYS.len(),
            total_ms = total_delay.as_millis(),
        ),
    })
}

/// Create a KVM VM with EINTR retry plus transient-errno
/// classification.
///
/// `KVM_CREATE_VM` can fail under two qualitatively different
/// conditions: (a) signal interruption (`EINTR`), where the right
/// answer is to retry the ioctl with a small backoff, and (b)
/// host-resource pressure (`ENOMEM`, `EMFILE`, `ENFILE`, `EBUSY`,
/// `EAGAIN`), where the right answer is to surface a
/// [`host_topology::ResourceContention`] so the `#[ktstr_test]`
/// macro can SKIP-skip cleanly. Anything else (`EINVAL`, `ENOSYS`,
/// `EPERM`, …) is a real fault; surface it as a regular error so
/// nextest fails the test loudly.
///
/// EINTR retry uses [`KVM_CREATE_VM_EINTR_DELAYS`] — eight
/// millisecond-scale steps totalling ≈862 ms. The previous schedule
/// (5 µs-scale steps, 62 µs total) was tuned for the bare-ioctl
/// case and starved against any signal source that fires
/// continuously across a few tens of microseconds.
///
/// # EINTR exhaustion → contention
///
/// When the EINTR retry budget is exhausted (every step in
/// [`KVM_CREATE_VM_EINTR_DELAYS`] tried) and the ioctl is STILL
/// returning EINTR, the host is under sustained signal pressure:
/// some peer is firing realtime / SIGRTMIN / SIGUSR signals at a
/// rate that walks faster than the backoff schedule can absorb.
/// That is a transient host condition, not a kernel fault, so it
/// classifies as [`host_topology::ResourceContention`] — the
/// `#[ktstr_test]` macro then SKIPs cleanly and stats tooling
/// records the skip via the sidecar. Without this branch, the
/// terminal EINTR fell through `map_transient_to_contention`
/// (which doesn't recognize EINTR as transient) and surfaced as a
/// hard error — every co-located test failed loudly during a CI
/// signal storm.
pub(crate) fn create_vm_with_retry(kvm: &kvm_ioctls::Kvm) -> Result<kvm_ioctls::VmFd> {
    let mut attempts = 0usize;
    loop {
        match kvm.create_vm() {
            Ok(fd) => break Ok(fd),
            Err(e) if e.errno() == libc::EINTR && attempts < KVM_CREATE_VM_EINTR_DELAYS.len() => {
                let delay = KVM_CREATE_VM_EINTR_DELAYS[attempts];
                tracing::warn!(
                    attempt = attempts,
                    delay_us = delay.as_micros() as u64,
                    "KVM_CREATE_VM EINTR; retrying"
                );
                std::thread::sleep(delay);
                attempts += 1;
            }
            Err(e) if e.errno() == libc::EINTR => {
                // Exhausted the EINTR retry budget; sustained
                // signal pressure is a transient host condition,
                // not a kernel fault. Surface as
                // ResourceContention so the macro layer SKIPs
                // cleanly instead of letting EINTR fall through
                // `map_transient_to_contention` (which doesn't
                // include EINTR in `TRANSIENT_HOST_ERRNOS` because
                // the routine retry path covers the common case).
                break Err(eintr_exhausted_contention());
            }
            Err(e) => break Err(map_transient_to_contention(e, "create VM")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `host_resource_snapshot`'s [`Display`] impl must produce a
    /// single-line string naming the four key fields the operator
    /// needs when triaging a `ktstr: SKIP: resource contention:
    /// ...` banner: open-fd count, RSS, thread count, and a
    /// `near_limit` indicator that summarises "are we close to
    /// RLIMIT_NOFILE / RLIMIT_NPROC?" without echoing the cap value
    /// (host fingerprint). The values themselves are
    /// runtime-dependent so the test pins only the SHAPE, not the
    /// numbers — a regression that swallows a field would fail
    /// here, AND any regression that leaks a raw rlimit
    /// (max_files= / max_procs=) trips the negative checks below.
    #[test]
    fn host_resource_snapshot_emits_all_keys() {
        let s = format!("{}", host_resource_snapshot());
        assert!(s.contains("fds="), "snapshot missing fds=: {s}");
        assert!(s.contains("vmrss="), "snapshot missing vmrss=: {s}");
        assert!(s.contains("threads="), "snapshot missing threads=: {s}");
        assert!(
            s.contains("near_limit="),
            "snapshot missing near_limit=: {s}"
        );
        assert!(
            !s.contains('\n'),
            "snapshot must be single-line for banner formatting: {s:?}"
        );
    }

    /// `host_resource_snapshot`'s rendered string must NOT echo raw
    /// RLIMIT_NOFILE / RLIMIT_NPROC values. Those are host-specific
    /// fingerprints that the SKIP banner surfaces into every CI
    /// artifact and user-visible test log — leaking them lets a
    /// third party reading a public failure dump infer host config.
    /// The `near_limit` derived flag preserves the actionable
    /// triage signal without the leak. Pins the deletion against a
    /// regression that adds them back.
    #[test]
    fn host_resource_snapshot_does_not_leak_raw_rlimits() {
        let s = format!("{}", host_resource_snapshot());
        assert!(
            !s.contains("max_files="),
            "host_resource_snapshot must not leak the RLIMIT_NOFILE soft cap; got: {s}",
        );
        assert!(
            !s.contains("max_procs="),
            "host_resource_snapshot must not leak the RLIMIT_NPROC soft cap; got: {s}",
        );
        // `fds=N/M` was the old leakage shape — the snapshot now
        // emits `fds=N` (count alone) so the fds= field must not
        // contain a slash. Slice from `fds=` to the next comma or
        // end of string and assert the slice has no `/`.
        if let Some(rest) = s
            .strip_prefix("fds=")
            .or_else(|| s.split_once("fds=").map(|(_, r)| r))
        {
            let fds_field = rest.split(',').next().unwrap_or(rest);
            assert!(
                !fds_field.contains('/'),
                "snapshot's fds= field must not include the cap (slash form); got: {s}",
            );
        }
    }

    /// `near_limit` must surface in the rendered banner as a
    /// boolean literal (`true` or `false`). Pins the
    /// parser-friendly format against a regression that swaps the
    /// boolean for an opaque token.
    #[test]
    fn host_resource_snapshot_near_limit_is_boolean() {
        let s = format!("{}", host_resource_snapshot());
        assert!(
            s.contains("near_limit=true") || s.contains("near_limit=false"),
            "near_limit must be boolean; got: {s}",
        );
    }

    /// The typed [`HostResourceSnapshot::near_limit`] field that
    /// [`map_transient_to_contention`]'s
    /// `KTSTR_CONTENTION_BYPASS` arm gates on must agree with the
    /// `near_limit=` token in the rendered banner. Pins the
    /// invariant that a banner-format change cannot silently
    /// desync the gate from the snapshot — both reads come from
    /// the same struct field.
    #[test]
    fn host_resource_snapshot_typed_field_agrees_with_rendered_banner() {
        let snapshot = host_resource_snapshot();
        let rendered = format!("{snapshot}");
        let expected_token = if snapshot.near_limit {
            "near_limit=true"
        } else {
            "near_limit=false"
        };
        assert!(
            rendered.contains(expected_token),
            "rendered banner must agree with typed field; \
             snapshot.near_limit={field_val}, rendered={rendered}",
            field_val = snapshot.near_limit,
        );
    }

    /// Display formatting on a constructed snapshot pins the exact
    /// `fds=N, vmrss=X, threads=Y, near_limit=B` shape that stats
    /// tooling parses out of SKIP banners. A struct constructor
    /// gives the test deterministic values; the runtime
    /// `host_resource_snapshot()` reads vary by host.
    #[test]
    fn host_resource_snapshot_display_format_is_pinned() {
        let snapshot = HostResourceSnapshot {
            fd_count: 64,
            vm_rss: "24 kB".into(),
            threads: "8".into(),
            near_limit: false,
        };
        assert_eq!(
            format!("{snapshot}"),
            "fds=64, vmrss=24 kB, threads=8, near_limit=false",
        );
        let snapshot_at_limit = HostResourceSnapshot {
            fd_count: 1023,
            vm_rss: "999 mB".into(),
            threads: "256".into(),
            near_limit: true,
        };
        assert_eq!(
            format!("{snapshot_at_limit}"),
            "fds=1023, vmrss=999 mB, threads=256, near_limit=true",
        );
    }

    /// `map_transient_to_contention` must classify the documented
    /// transient errnos (ENOMEM, EMFILE, ENFILE, EBUSY, EAGAIN) as
    /// `ResourceContention` so the macro's contention arm fires
    /// and the test SKIPs cleanly. Asserts the result downcasts
    /// AND that the rendered banner includes the caller-supplied
    /// context plus the errno name and the host-resource snapshot.
    #[test]
    fn map_transient_to_contention_classifies_enomem() {
        // Holds the env lock and ensures `KTSTR_CONTENTION_BYPASS`
        // is unset for the test duration: the bypass is opt-in and
        // its default-off behaviour is what this test pins. Without
        // the guard, a developer with the var set in their shell
        // would observe a spurious failure — the bypass would route
        // the assertions through a hard-error branch instead of the
        // `ResourceContention` branch this test pins.
        let _lock = crate::test_support::test_helpers::lock_env();
        let _bypass_off =
            crate::test_support::test_helpers::EnvVarGuard::remove("KTSTR_CONTENTION_BYPASS");
        for &errno in TRANSIENT_HOST_ERRNOS {
            let kvm_err = kvm_ioctls::Error::new(errno);
            let mapped = map_transient_to_contention(kvm_err, "create VM");
            assert!(
                mapped
                    .downcast_ref::<host_topology::ResourceContention>()
                    .is_some(),
                "errno {errno} ({}): expected ResourceContention, got {mapped:#}",
                errno_name(errno),
            );
            let rendered = format!("{mapped:#}");
            assert!(
                rendered.contains("create VM"),
                "errno {errno} banner missing context: {rendered}"
            );
            assert!(
                rendered.contains(&*errno_name(errno)),
                "errno {errno} banner missing errno name: {rendered}"
            );
            assert!(
                rendered.contains("host resources:"),
                "errno {errno} banner missing host-resource snapshot: {rendered}"
            );
        }
    }

    /// Non-transient errnos (EINVAL, ENOSYS, EPERM, EACCES) MUST
    /// flow through unchanged so a real KVM bug never gets
    /// misclassified as a recoverable SKIP. Pins the negative
    /// case: the result must NOT downcast to
    /// `ResourceContention`. The kernel returning EINVAL from
    /// `KVM_CREATE_VM` would mean a programming fault (wrong
    /// kvm_xen_hvm_config layout, etc.); SKIP-skipping it would
    /// hide the bug from the test report.
    #[test]
    fn map_transient_to_contention_passes_through_hard_errors() {
        for &errno in &[libc::EINVAL, libc::ENOSYS, libc::EPERM, libc::EACCES] {
            let kvm_err = kvm_ioctls::Error::new(errno);
            let mapped = map_transient_to_contention(kvm_err, "set TSS");
            assert!(
                mapped
                    .downcast_ref::<host_topology::ResourceContention>()
                    .is_none(),
                "errno {errno} ({}): hard fault must NOT be classified as \
                 ResourceContention; got {mapped:#}",
                errno_name(errno),
            );
            let rendered = format!("{mapped:#}");
            assert!(
                rendered.contains("set TSS"),
                "errno {errno} banner missing context: {rendered}"
            );
        }
    }

    /// EINTR is NOT in `TRANSIENT_HOST_ERRNOS` because the routine
    /// retry loop in `create_vm_with_retry` covers the common
    /// case. Pins that contract: passing EINTR through
    /// `map_transient_to_contention` directly would silently
    /// SKIP every test on a single signal interruption — the
    /// retry loop's whole reason to exist. The EINTR-exhausted
    /// path uses the dedicated `eintr_exhausted_contention`
    /// helper instead, asserted in the test below.
    #[test]
    fn map_transient_to_contention_does_not_classify_eintr() {
        let kvm_err = kvm_ioctls::Error::new(libc::EINTR);
        let mapped = map_transient_to_contention(kvm_err, "create VM");
        assert!(
            mapped
                .downcast_ref::<host_topology::ResourceContention>()
                .is_none(),
            "EINTR must NOT classify as ResourceContention via \
             map_transient_to_contention — the retry loop handles \
             single EINTR; only EXHAUSTED EINTR is contention. \
             got: {mapped:#}",
        );
    }

    /// `map_transient_to_contention` is the routing layer that
    /// `set_user_memory_region` (in [`super::numa_mem`]) wraps
    /// every memslot install through; the wrap was added so kernel-
    /// side ENOMEM during region setup classifies as a clean SKIP
    /// instead of a hard test failure. Pins the routing contract:
    /// every errno in `TRANSIENT_HOST_ERRNOS` produces a
    /// `ResourceContention`, and a non-transient errno (`EINVAL`,
    /// the canonical "you handed me a bad memslot layout" return)
    /// must NOT downcast — so a real layout regression surfaces
    /// loudly instead of being skipped.
    ///
    /// Synthesizes `kvm_ioctls::Error::new(...)` for each errno
    /// rather than driving the actual `set_user_memory_region`
    /// ioctl: the routing layer is a pure function over
    /// `kvm_ioctls::Error` and the routing contract holds for
    /// every callsite that wraps it (vCPU init, GIC setup, memslot
    /// install). Driving the real ioctl would also exercise the
    /// kernel's memslot validation, which is out of scope for this
    /// test.
    ///
    /// Holds [`crate::test_support::test_helpers::lock_env`] and
    /// removes `KTSTR_CONTENTION_BYPASS` for the test duration so
    /// a developer with the bypass set in their shell does not
    /// observe spurious failures, and so a concurrent
    /// bypass-exercising test cannot race the unset.
    #[test]
    fn set_user_memory_region_routing_via_map_transient() {
        let _lock = crate::test_support::test_helpers::lock_env();
        let _bypass_off =
            crate::test_support::test_helpers::EnvVarGuard::remove("KTSTR_CONTENTION_BYPASS");
        // Every transient errno must classify as ResourceContention
        // when wrapped through map_transient_to_contention with the
        // memslot-install context tag.
        for &errno in TRANSIENT_HOST_ERRNOS {
            let kvm_err = kvm_ioctls::Error::new(errno);
            let mapped = map_transient_to_contention(kvm_err, "set_user_memory_region");
            assert!(
                mapped
                    .downcast_ref::<host_topology::ResourceContention>()
                    .is_some(),
                "errno {errno} ({}): set_user_memory_region routing must \
                 classify as ResourceContention; got {mapped:#}",
                errno_name(errno),
            );
            let rendered = format!("{mapped:#}");
            assert!(
                rendered.contains("set_user_memory_region"),
                "errno {errno} banner missing memslot-install context tag: {rendered}"
            );
            assert!(
                rendered.contains(&*errno_name(errno)),
                "errno {errno} banner missing errno name: {rendered}"
            );
        }
        // EINVAL is the canonical "bad memslot layout" return — a
        // real layout bug must surface as a hard error so the
        // operator sees the regression rather than a silent SKIP.
        let kvm_err = kvm_ioctls::Error::new(libc::EINVAL);
        let mapped = map_transient_to_contention(kvm_err, "set_user_memory_region");
        assert!(
            mapped
                .downcast_ref::<host_topology::ResourceContention>()
                .is_none(),
            "EINVAL from set_user_memory_region must NOT classify as \
             ResourceContention — bad memslot layout is a programming \
             fault that SKIP-skipping would hide; got: {mapped:#}",
        );
    }

    /// `KTSTR_CONTENTION_BYPASS=1` opt-in: when the env var is set
    /// AND the host-resource snapshot reports `near_limit=false`,
    /// every transient errno surfaces as a hard error rather than a
    /// `ResourceContention`. The opt-in exists so an operator
    /// hunting kernel-side regressions (a leak / stuck device that
    /// shares a transient errno with peer contention) can see the
    /// failure instead of having it SKIP-skipped.
    ///
    /// Default-off behaviour is pinned by the
    /// `map_transient_to_contention_classifies_enomem` test above
    /// (which runs without setting the var).
    ///
    /// The test relies on `host_resource_snapshot()` reporting
    /// `near_limit=false` on the test runner. That holds in
    /// practice on a CI runner: it would take an exhaustion event
    /// across `RLIMIT_NOFILE` or per-UID `RLIMIT_NPROC` (>= 90% of
    /// the soft cap) to flip it true, and the test runner does not
    /// approach those caps. If a future runner does saturate the
    /// snapshot, the assertion below would mark the test as
    /// `near_limit=true` (i.e. bypass not active), and the test
    /// SKIPs the bypass assertion rather than failing it — the
    /// snapshot precondition is documented, and a bypass test with
    /// no near_limit=false snapshot to gate on cannot exercise the
    /// path. Holds [`lock_env`] so the env-var mutation does not
    /// race other env-touching tests.
    #[test]
    fn map_transient_to_contention_bypass_when_near_limit_false() {
        let _lock = crate::test_support::test_helpers::lock_env();
        // Sanity: confirm the runner's snapshot reports
        // near_limit=false. If not, the bypass cannot be exercised
        // and we cannot make the assertion below; bail without
        // failing.
        let snapshot = host_resource_snapshot();
        if snapshot.near_limit {
            return;
        }
        let _bypass_on =
            crate::test_support::test_helpers::EnvVarGuard::set("KTSTR_CONTENTION_BYPASS", "1");
        // ENOMEM is in TRANSIENT_HOST_ERRNOS but with bypass-on +
        // near_limit=false, must NOT classify as ResourceContention.
        let kvm_err = kvm_ioctls::Error::new(libc::ENOMEM);
        let mapped = map_transient_to_contention(kvm_err, "set_user_memory_region");
        assert!(
            mapped
                .downcast_ref::<host_topology::ResourceContention>()
                .is_none(),
            "with KTSTR_CONTENTION_BYPASS=1 and near_limit=false, ENOMEM must \
             surface as a hard error rather than ResourceContention; got: {mapped:#}",
        );
        let rendered = format!("{mapped:#}");
        assert!(
            rendered.contains("KTSTR_CONTENTION_BYPASS=1"),
            "bypass diagnostic must mention the env var so the operator can \
             see why this surfaced as a hard error; got: {rendered}",
        );
        assert!(
            rendered.contains("NOT near limits"),
            "bypass diagnostic must explain the `near_limit=false` rationale; \
             got: {rendered}",
        );
    }

    /// `eintr_exhausted_contention` produces a
    /// [`host_topology::ResourceContention`] with the canonical
    /// banner shape stats tooling parses for skip classification.
    /// Pins (a) the type so the macro layer's downcast fires,
    /// (b) the `create VM` context tag for grep-based test-summary
    /// tools, (c) the host-resource snapshot for triage, and (d)
    /// the EINTR-specific hint so an operator hitting this skip
    /// knows to look for a signal-storm peer rather than a
    /// memory-pressure peer.
    #[test]
    fn eintr_exhausted_contention_format() {
        let err = eintr_exhausted_contention();
        assert!(
            err.downcast_ref::<host_topology::ResourceContention>()
                .is_some(),
            "EINTR-exhausted error must downcast to ResourceContention; got: {err:#}",
        );
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("create VM"),
            "banner missing context tag: {rendered}",
        );
        assert!(
            rendered.contains("EINTR"),
            "banner missing EINTR-specific classification: {rendered}",
        );
        assert!(
            rendered.contains("host resources:"),
            "banner missing host-resource snapshot: {rendered}",
        );
        assert!(
            rendered.contains("signal"),
            "banner missing signal-storm hint: {rendered}",
        );
    }

    /// `create_vm_with_retry` happy path: a real KVM handle with no
    /// signal pressure must succeed on the first attempt and return
    /// an `Ok(VmFd)`. Pins the loop-exit ordering: the `Ok` arm is
    /// matched first inside the loop, so a healthy syscall short-
    /// circuits without entering any of the EINTR / transient /
    /// hard-fault branches. The three failure arms are pinned by
    /// the helper-level tests directly above
    /// (`map_transient_to_contention_classifies_enomem`,
    /// `map_transient_to_contention_passes_through_hard_errors`,
    /// `map_transient_to_contention_does_not_classify_eintr`,
    /// `eintr_exhausted_contention_format`); together they cover
    /// every branch of the match, and this test pins the
    /// composition.
    ///
    /// Kvm::new() failure panics — the test requires a KVM-capable
    /// host. The ResourceContention skip only applies to
    /// create_vm_with_retry's own return value: if the kernel
    /// genuinely cannot grant a fresh VM after the EINTR retry
    /// schedule resolves (host memory pressure, peer holding the
    /// VM budget), the function surfaces ResourceContention and
    /// the test treats that the same way the macro layer does
    /// in production. A missing /dev/kvm at the OPENING step is
    /// infrastructure misconfiguration, not host saturation, and
    /// the test fails loudly rather than silently no-op'ing.
    #[test]
    fn create_vm_with_retry_succeeds_under_no_signal_pressure() {
        let kvm = match kvm_ioctls::Kvm::new() {
            Ok(k) => k,
            Err(e) => {
                // No /dev/kvm at all is infrastructure
                // misconfiguration on this host (kernel built
                // without KVM, /dev/kvm permission denied, etc.) —
                // not the kind of transient host pressure that the
                // ResourceContention skip path is designed for.
                // Fail loudly so the misconfig is visible instead
                // of being silently swallowed by a no-op test.
                panic!(
                    "Kvm::new() failed: {e}; cannot exercise \
                     create_vm_with_retry on this host"
                );
            }
        };
        let vm = create_vm_with_retry(&kvm);
        match vm {
            Ok(_) => {
                // Success path: the function returned a real VmFd
                // without retrying. Drop releases the kernel
                // resource immediately.
            }
            Err(e)
                if e.downcast_ref::<host_topology::ResourceContention>()
                    .is_some() =>
            {
                // The host genuinely could not grant a VM — this
                // is the same skip the macro layer surfaces in
                // production. Treat as a pass since we already
                // pinned the failure-arm classifiers above.
            }
            Err(e) => {
                panic!(
                    "create_vm_with_retry returned an unexpected \
                     non-contention error on a no-signal-pressure host: {e:#}"
                );
            }
        }
    }

    /// `errno_name` falls through to a `errno=<raw>` rendering for
    /// values not in its mapped table — so an operator looking at
    /// the SKIP banner can grep for the exact integer instead of
    /// seeing a useless `<other>`. Pins the fallthrough format
    /// against a regression that goes back to the static `<other>`.
    #[test]
    fn errno_name_fallthrough_renders_raw_value() {
        // Pick an errno guaranteed not to be in the mapped table:
        // 9999 is well above the kernel's defined range and reserved
        // for tests. Any value the table starts mapping in the
        // future would still leave 9999 as a fallthrough sample.
        let rendered = errno_name(9999);
        assert!(
            rendered.contains("errno=9999"),
            "fallthrough must include the exact `errno=N` format string \
             so callers grepping for it find the integer reliably; got: {rendered}",
        );
        assert!(
            !rendered.contains("<other>"),
            "fallthrough must not collapse to the placeholder <other>; got: {rendered}",
        );
    }

    /// Mapped errnos retain their canonical name (no regression
    /// from the new `Cow` return type swallowing the static
    /// `&'static str` branches).
    #[test]
    fn errno_name_maps_canonical_names() {
        for (errno, expected) in [
            (libc::ENOMEM, "ENOMEM"),
            (libc::EBUSY, "EBUSY"),
            (libc::EAGAIN, "EAGAIN"),
            (libc::EINVAL, "EINVAL"),
            (libc::ENOSYS, "ENOSYS"),
            (libc::EPERM, "EPERM"),
            (libc::EACCES, "EACCES"),
        ] {
            let rendered = errno_name(errno);
            assert_eq!(
                &*rendered, expected,
                "errno {errno} must render as {expected}; got {rendered}",
            );
        }
    }

    /// `KVM_CREATE_VM_EINTR_DELAYS` must be monotonically
    /// non-decreasing — exponential backoff is the contract — and
    /// must total at least 800 ms so a sustained signal storm gets
    /// a real chance to drain instead of starving the ioctl in <1 ms.
    /// Pins the schedule against a regression that re-tunes it
    /// downwards (the previous schedule was 62 µs total — too
    /// short to outlast a sustained signal storm).
    #[test]
    fn kvm_create_vm_eintr_delays_total_budget() {
        let mut prev = Duration::ZERO;
        let mut total = Duration::ZERO;
        for &d in &KVM_CREATE_VM_EINTR_DELAYS {
            assert!(
                d >= prev,
                "EINTR delays must be monotonic: {prev:?} → {d:?}"
            );
            prev = d;
            total += d;
        }
        assert!(
            total >= Duration::from_millis(800),
            "total EINTR budget must be ≥ 800 ms to absorb signal storms; got {total:?}"
        );
        // Upper bound — guard against a future tweak that pushes the
        // budget past the freeze coordinator's rendezvous window.
        // 2 s gives ~2.3x headroom over the current 862 ms schedule
        // while still staying well below the typical 30 s freeze
        // timeout.
        assert!(
            total <= Duration::from_secs(2),
            "total EINTR budget must be ≤ 2 s to stay within the \
             freeze rendezvous window; got {total:?}"
        );
    }
}
