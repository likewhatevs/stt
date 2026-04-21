//! Runtime builder for launching a [`Payload`] from a test body.
//!
//! `ctx.payload(&X)` returns a [`PayloadRun`] whose chainable
//! methods configure args, checks, and cgroup placement before the
//! terminal `.run()` (foreground) or `.spawn()` (background)
//! executes the binary inside the guest VM.
//!
//! `.run()` blocks until the child exits and returns
//! `Result<(AssertResult, PayloadMetrics)>`. The builder is a pure
//! guest-side std::process::Child wrapper — no cross-VM proxy.
//!
//! `PayloadKind::Scheduler` payloads are rejected at `.run()`:
//! schedulers are launched by the framework at test start, not by
//! test-body invocation. Only `PayloadKind::Binary` payloads are
//! runnable via this builder.
//!
//! Args composition:
//! 1. `payload.default_args` unless `.clear_args()` was called.
//! 2. Plus any runtime `.arg(...)` / `.args(...)` appends.
//!
//! Checks composition is identical in shape.
//!
//! # Stdout-only metric extraction
//!
//! The extraction pipeline consumes **stdout only**. Stderr is
//! captured and forwarded verbatim into the exit-code-mismatch
//! detail produced by [`Check::ExitCodeEq`] (see the
//! `format_exit_mismatch` path) but is NEVER fed to
//! [`crate::test_support::extract_metrics`] — neither the
//! `OutputFormat::Json` walker nor the `OutputFormat::LlmExtract`
//! prompt sees a stderr byte. Payloads that emit their structured
//! output on stderr (e.g. schbench's default percentile tables via
//! `show_latencies` → `fprintf(stderr, ...)`) therefore hand the
//! extractor an empty string and produce zero metrics. Redirect the
//! payload's output to stdout at the invocation site (schbench:
//! `--json -`) or declare an `OutputFormat::ExitCode` fixture for
//! stderr-only binaries.

use std::borrow::Cow;
use std::ffi::CString;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

use crate::assert::{AssertDetail, AssertResult, DetailKind};
use crate::scenario::Ctx;
use crate::test_support::{Check, Metric, Payload, PayloadKind, PayloadMetrics, extract_metrics};

/// Builder returned by [`Ctx::payload`](crate::scenario::Ctx).
///
/// Configure the run via chainable methods, then invoke `.run()`
/// (foreground, blocking) or `.spawn()` (background) to execute the
/// payload's binary inside the guest VM and receive the extracted
/// [`PayloadMetrics`] plus an [`AssertResult`] for any declared
/// [`Check`]s.
pub struct PayloadRun<'a> {
    ctx: &'a Ctx<'a>,
    payload: &'static Payload,
    /// Effective argv. Initialized to `payload.default_args` on
    /// construction; `.arg`/`.args` append, `.clear_args` truncates.
    args: Vec<String>,
    /// Effective check list. Initialized to `payload.default_checks`;
    /// `.check` appends, `.clear_checks` truncates.
    checks: Vec<Check>,
    /// User-supplied relative cgroup name (from [`in_cgroup`]). The
    /// absolute path is resolved + validated at `.run()`/`.spawn()`.
    /// [`Cow`] keeps static-name callers zero-alloc while still
    /// accepting owned Strings from dynamic call sites.
    cgroup: Option<Cow<'static, str>>,
}

impl std::fmt::Debug for PayloadRun<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PayloadRun")
            .field("payload", &self.payload.name)
            .field("args_len", &self.args.len())
            .field("checks_len", &self.checks.len())
            .field("cgroup", &self.cgroup)
            .finish()
    }
}

impl<'a> PayloadRun<'a> {
    pub(crate) fn new(ctx: &'a Ctx<'a>, payload: &'static Payload) -> Self {
        let args = payload.default_args.iter().map(|s| s.to_string()).collect();
        let checks = payload.default_checks.to_vec();
        Self {
            ctx,
            payload,
            args,
            checks,
            cgroup: None,
        }
    }

    /// Append one CLI argument to the effective argv.
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Append multiple CLI arguments to the effective argv.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Wipe ALL args (both `payload.default_args` and any prior
    /// `.arg()` calls). Subsequent `.arg()` calls start from empty.
    pub fn clear_args(mut self) -> Self {
        self.args.clear();
        self
    }

    /// Append a [`Check`] to the effective check list.
    pub fn check(mut self, c: Check) -> Self {
        self.checks.push(c);
        self
    }

    /// Wipe ALL checks (both `payload.default_checks` and any prior
    /// `.check()` calls).
    pub fn clear_checks(mut self) -> Self {
        self.checks.clear();
        self
    }

    /// Place the spawned child in the named cgroup (a plain name,
    /// resolved relative to `ctx.cgroups.parent_path()`). When
    /// omitted, the child inherits the spawning process's cgroup.
    ///
    /// Accepts `&'static str` (zero-alloc, the common case of a
    /// const cgroup name) or any owned string type via [`Cow`]'s
    /// `From` impls.
    ///
    /// The name is validated at `.run()`/`.spawn()` — leading `/`
    /// is stripped, `..` and NUL bytes are rejected.
    pub fn in_cgroup(mut self, name: impl Into<Cow<'static, str>>) -> Self {
        self.cgroup = Some(name.into());
        self
    }

    /// Blocking foreground run. Spawns the payload binary, waits
    /// for it to exit, extracts metrics from stdout per the
    /// payload's [`OutputFormat`], and evaluates declared [`Check`]s
    /// into an [`AssertResult`].
    ///
    /// Metrics are also recorded to the per-test sidecar via the
    /// SHM ring; the returned tuple is a convenience view of the
    /// same values.
    ///
    /// Returns `Err` when the payload is not
    /// [`PayloadKind::Binary`] (schedulers are framework-launched,
    /// not test-body-launched), when the cgroup name fails
    /// validation, or when the spawn itself fails.
    pub fn run(self) -> Result<(AssertResult, PayloadMetrics)> {
        let binary = payload_binary(self.payload)?;
        let cgroup_path = resolve_cgroup_path(self.ctx, self.cgroup.as_deref())?;
        let output = spawn_and_wait(binary, &self.args, cgroup_path.as_deref())
            .with_context(|| format!("spawn payload '{}'", self.payload.name))?;
        Ok(evaluate(self.payload, &self.checks, output))
    }

    /// Spawn the payload binary in the background and return a
    /// [`PayloadHandle`] the caller can `.wait()`, `.kill()`, or
    /// `.try_wait()` on.
    ///
    /// The child runs in the guest's process namespace (all ktstr
    /// tests execute inside the VM); `PayloadHandle` is a thin
    /// wrapper over [`std::process::Child`]. No cross-VM proxy.
    ///
    /// Dropping the handle without first calling one of the waiters
    /// emits a stderr warning and SIGKILLs the child — leaked
    /// handles would lose metrics and potentially outlive the test.
    ///
    /// Returns `Err` when the payload is not
    /// [`PayloadKind::Binary`] or when the spawn itself fails.
    pub fn spawn(self) -> Result<PayloadHandle> {
        let binary = payload_binary(self.payload)?;
        let cgroup_path = resolve_cgroup_path(self.ctx, self.cgroup.as_deref())?;
        let child = spawn_child(binary, &self.args, cgroup_path.as_deref())
            .with_context(|| format!("spawn payload '{}'", self.payload.name))?;
        Ok(PayloadHandle {
            child: Some(child),
            payload: self.payload,
            checks: self.checks,
        })
    }
}

/// Unwrap [`PayloadKind::Binary`] to its binary name, erroring when
/// a scheduler-kind payload is passed.
fn payload_binary(payload: &Payload) -> Result<&'static str> {
    match payload.kind {
        PayloadKind::Binary(name) => Ok(name),
        PayloadKind::Scheduler(_) => anyhow::bail!(
            "ctx.payload({}) called on a scheduler-kind payload; \
             schedulers are launched by the test framework, not from \
             the test body. Use ctx.payload(&BINARY_PAYLOAD) instead.",
            payload.name,
        ),
    }
}

/// Common post-exit pipeline: extract metrics, resolve polarities,
/// evaluate checks. Shared between foreground `.run()` and
/// background handle `wait`/`kill` paths. The `PayloadMetrics` is
/// serialized to the guest-to-host SHM ring here — once per
/// invocation — so the host can reconstruct per-call provenance in
/// the sidecar without any Ctx-side accumulator.
fn evaluate(
    payload: &Payload,
    checks: &[Check],
    output: SpawnOutput,
) -> (AssertResult, PayloadMetrics) {
    let mut metrics = extract_metrics(&output.stdout, &payload.output);
    resolve_polarities(&mut metrics, payload);

    let payload_metrics = PayloadMetrics {
        metrics,
        exit_code: output.exit_code,
    };

    emit_payload_metrics_to_shm(&payload_metrics);

    let result = evaluate_checks(checks, &payload_metrics, &output.stderr);
    (result, payload_metrics)
}

/// Serialize a [`PayloadMetrics`] to JSON and emit it on the
/// guest-to-host SHM ring under
/// [`MSG_TYPE_PAYLOAD_METRICS`](crate::vmm::shm_ring::MSG_TYPE_PAYLOAD_METRICS).
///
/// The `serde_json::to_vec` call is infallible in practice for
/// `PayloadMetrics`: every field is an owned, serde-trivial value
/// (`Vec<Metric>` of `{ name: String, value: f64, polarity, unit:
/// String, source }` plus an `i32` exit code). None of these can
/// fail serialization for any inhabited `PayloadMetrics` value —
/// the Err arm exists only to satisfy `serde_json::to_vec`'s
/// `Result` signature. The defensive `eprintln!` guards against a
/// future struct change that introduces a fallible field (e.g. a
/// `#[serde(with = "...")]` custom serializer) rather than any
/// currently-reachable failure path.
///
/// A full SHM ring is handled silently by `write_msg` itself —
/// the writer drops the payload when no ring space is left. This
/// function does not re-handle ring pressure; it only handles the
/// serialize step.
fn emit_payload_metrics_to_shm(pm: &PayloadMetrics) {
    match serde_json::to_vec(pm) {
        Ok(bytes) => {
            crate::vmm::shm_ring::write_msg(crate::vmm::shm_ring::MSG_TYPE_PAYLOAD_METRICS, &bytes)
        }
        Err(e) => eprintln!("ktstr: serialize PayloadMetrics for SHM emit: {e}"),
    }
}

// ---------------------------------------------------------------------------
// PayloadHandle — background spawn result
// ---------------------------------------------------------------------------

/// Handle to a background payload spawned via
/// [`PayloadRun::spawn`]. Wraps a guest-local
/// [`std::process::Child`]; `wait` / `kill` both consume the handle
/// and return the collected metrics + assertion verdict.
///
/// Drop behavior: if the handle is dropped without `wait`/`kill`,
/// the child is SIGKILLed to prevent runaway processes from
/// outliving the test, and a stderr warning is emitted so the test
/// author sees the implicit drop.
///
/// When multiple handles are active, sidecar entries appear in
/// finalization order (the order `.wait()`/`.kill()` are called),
/// not spawn order.
#[must_use = "dropping a PayloadHandle kills the child process; call .wait() or .kill() explicitly"]
pub struct PayloadHandle {
    /// Live child process. Wrapped in `Option` so consumers can
    /// take ownership in `wait`/`kill` without making the drop-guard
    /// reach into a `None`.
    child: Option<std::process::Child>,
    payload: &'static Payload,
    checks: Vec<Check>,
}

impl std::fmt::Debug for PayloadHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Payload's manual Debug renders identity fields; the inner
        // Child is omitted (not Debug-rendering-friendly and carries
        // OS handles) — a one-line summary is enough for panics /
        // test output.
        f.debug_struct("PayloadHandle")
            .field("payload", &self.payload.name)
            .field("child_alive", &self.child.is_some())
            .field("checks_len", &self.checks.len())
            .finish()
    }
}

impl PayloadHandle {
    /// Name of the [`Payload`] this handle was spawned from — i.e.
    /// the identity key used by step-level ops to address a running
    /// payload. Step-local ops ([`Op::WaitPayload`](crate::scenario::ops::Op::WaitPayload),
    /// [`Op::KillPayload`](crate::scenario::ops::Op::KillPayload))
    /// match handles by this name.
    pub fn payload_name(&self) -> &'static str {
        self.payload.name
    }

    /// Block until the child exits naturally, then extract metrics
    /// and evaluate checks, matching the foreground `.run()` return
    /// shape.
    ///
    /// Metrics are also recorded to the per-test sidecar via the
    /// SHM ring; the returned tuple is a convenience view of the
    /// same values.
    pub fn wait(mut self) -> Result<(AssertResult, PayloadMetrics)> {
        let child = self
            .child
            .take()
            .ok_or_else(|| already_consumed(self.payload))?;
        let output = wait_and_capture(child)
            .with_context(|| format!("wait payload '{}'", self.payload.name))?;
        Ok(evaluate(self.payload, &self.checks, output))
    }

    /// SIGKILL the child, reap it, and return whatever stdout was
    /// captured along with the process exit code. Suitable for
    /// time-boxed background loads.
    ///
    /// Metrics are also recorded to the per-test sidecar via the
    /// SHM ring; the returned tuple is a convenience view of the
    /// same values.
    pub fn kill(mut self) -> Result<(AssertResult, PayloadMetrics)> {
        let mut child = self
            .child
            .take()
            .ok_or_else(|| already_consumed(self.payload))?;
        let _ = child.kill();
        let output = wait_and_capture(child)
            .with_context(|| format!("reap killed payload '{}'", self.payload.name))?;
        Ok(evaluate(self.payload, &self.checks, output))
    }

    /// Non-blocking check for exit without consuming the handle.
    /// Returns `Ok(Some((result, metrics)))` once the child has
    /// exited and output is drained; `Ok(None)` while still
    /// running. The handle remains live on `Ok(None)`.
    ///
    /// On the terminal `Ok(Some(..))` return, metrics are also
    /// recorded to the per-test sidecar via the SHM ring; the
    /// returned tuple is a convenience view of the same values.
    pub fn try_wait(&mut self) -> Result<Option<(AssertResult, PayloadMetrics)>> {
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| already_consumed(self.payload))?;
        match child.try_wait()? {
            Some(_status) => {
                // `child` was Some above; the earlier branch didn't
                // `take()` it, so this unwrap is guaranteed to hold.
                let child = self
                    .child
                    .take()
                    .expect("child still present on terminal branch");
                let output = wait_and_capture(child)
                    .with_context(|| format!("reap payload '{}'", self.payload.name))?;
                Ok(Some(evaluate(self.payload, &self.checks, output)))
            }
            None => Ok(None),
        }
    }
}

/// Error value produced when `wait`/`kill`/`try_wait` is called on a
/// handle whose child has already been taken by a prior call. The
/// payload name anchors the error to a specific handle so the
/// test log points directly at the misuse site.
fn already_consumed(payload: &'static Payload) -> anyhow::Error {
    anyhow!(
        "PayloadHandle for '{}' already consumed by a prior \
         wait/kill/try_wait call; each handle can only produce \
         one (AssertResult, PayloadMetrics) pair",
        payload.name,
    )
}

impl Drop for PayloadHandle {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            eprintln!(
                "ktstr: PayloadHandle for '{}' dropped without wait/kill — \
                 child SIGKILLed, metrics not recorded.",
                self.payload.name,
            );
        }
    }
}

/// Resolve each extracted metric's polarity + unit against the
/// payload's declared `metrics` hints.
///
/// Unhinted metrics keep [`Polarity::Unknown`] and empty unit.
///
/// Complexity: O(N + M) — build a `HashMap<&str, &MetricHint>` from
/// the hint slice once, then look up each metric by name in O(1).
/// The prior linear-scan implementation was O(N × M) where N is
/// extracted metrics and M is declared hints; fio JSON with
/// thousands of leaves + a dozen hints was the hottest path this
/// module sees per payload run.
fn resolve_polarities(metrics: &mut [Metric], payload: &Payload) {
    if payload.metrics.is_empty() || metrics.is_empty() {
        return;
    }
    let hints: std::collections::HashMap<&str, &crate::test_support::MetricHint> =
        payload.metrics.iter().map(|h| (h.name, h)).collect();
    for metric in metrics {
        if let Some(hint) = hints.get(metric.name.as_str()) {
            metric.polarity = hint.polarity;
            metric.unit = hint.unit.to_string();
        }
    }
}

/// Evaluate [`Check`]s against a [`PayloadMetrics`] and fold the
/// verdict into an [`AssertResult`].
///
/// Evaluation order:
/// 1. [`Check::ExitCodeEq`] pre-pass — evaluated FIRST so a
///    misconfigured binary fails with an actionable exit-code error
///    rather than "metric X not found".
/// 2. Metric-path checks ([`Check::Min`], [`Check::Max`],
///    [`Check::Range`], [`Check::Exists`]).
///
/// `stderr` is folded into the exit-code-mismatch detail when
/// present — when a binary fails with "expected 0 got 1", the
/// captured stderr almost always explains why, and forcing the test
/// author to go hunt it down defeats actionable diagnostics.
///
/// Missing metrics fail loudly — a `Min` / `Max` / `Range` / `Exists`
/// check against an absent metric reports a "not found" detail
/// instead of silently passing.
fn evaluate_checks(checks: &[Check], pm: &PayloadMetrics, stderr: &str) -> AssertResult {
    let mut result = AssertResult::pass();
    // Pre-pass: exit-code checks first.
    for check in checks {
        if let Check::ExitCodeEq(expected) = check
            && pm.exit_code != *expected
        {
            result.merge(fail_result(AssertDetail {
                kind: DetailKind::Other,
                message: format_exit_mismatch(pm.exit_code, *expected, stderr),
            }));
            // Short-circuit metric checks: a bad exit probably means
            // the metric extraction found nothing useful.
            return result;
        }
    }
    // Metric-path pass.
    for check in checks {
        let detail = match check {
            Check::Min { metric, value } => pm.get(metric).map_or_else(
                || Some(missing_metric(metric)),
                |actual| {
                    (actual < *value).then(|| AssertDetail {
                        kind: DetailKind::Other,
                        message: format!("metric '{metric}' = {actual} below minimum {value}"),
                    })
                },
            ),
            Check::Max { metric, value } => pm.get(metric).map_or_else(
                || Some(missing_metric(metric)),
                |actual| {
                    (actual > *value).then(|| AssertDetail {
                        kind: DetailKind::Other,
                        message: format!("metric '{metric}' = {actual} exceeds maximum {value}"),
                    })
                },
            ),
            Check::Range { metric, lo, hi } => pm.get(metric).map_or_else(
                || Some(missing_metric(metric)),
                |actual| {
                    ((actual < *lo) || (actual > *hi)).then(|| AssertDetail {
                        kind: DetailKind::Other,
                        message: format!("metric '{metric}' = {actual} outside [{lo}, {hi}]"),
                    })
                },
            ),
            Check::Exists(metric) => pm.get(metric).is_none().then(|| missing_metric(metric)),
            Check::ExitCodeEq(_) => None, // already evaluated in pre-pass
        };
        if let Some(d) = detail {
            result.merge(fail_result(d));
        }
    }
    result
}

/// Build a failing [`AssertResult`] from a single [`AssertDetail`].
/// [`AssertResult`] itself has no `fail()` constructor; this helper
/// centralizes the struct-literal shape so callers don't each need
/// to reach for stats defaults.
fn fail_result(detail: AssertDetail) -> AssertResult {
    AssertResult {
        passed: false,
        skipped: false,
        details: vec![detail],
        stats: Default::default(),
    }
}

fn missing_metric(metric: &str) -> AssertDetail {
    AssertDetail {
        kind: DetailKind::Other,
        message: format!("metric '{metric}' not found in payload output"),
    }
}

/// Render an exit-code mismatch with a trailing stderr tail when
/// non-empty. Long stderr is tail-truncated (last 1 KiB) — the end
/// of a failing process usually carries the actionable error.
const STDERR_TAIL_BYTES: usize = 1024;

fn format_exit_mismatch(actual: i32, expected: i32, stderr: &str) -> String {
    let trimmed = stderr.trim_end();
    if trimmed.is_empty() {
        return format!("payload exited with code {actual}, expected {expected}");
    }
    let tail = stderr_tail(trimmed, STDERR_TAIL_BYTES);
    format!("payload exited with code {actual}, expected {expected}; stderr:\n{tail}")
}

/// Return the final `max_bytes` of `s`, snapped forward to a char
/// boundary so slicing never panics on multi-byte input. Emits a
/// leading `...` when truncation actually happens.
fn stderr_tail(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut start = s.len() - max_bytes;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("...{}", &s[start..])
}

/// Captured output from a payload process invocation. `stderr`
/// is kept so the evaluator can surface it on non-zero exit — the
/// extracted metrics alone don't explain why a binary failed.
struct SpawnOutput {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

/// Resolve the user-supplied cgroup name to an absolute path
/// under `ctx.cgroups.parent_path()`, validating BEFORE fork so a
/// bad name produces a clear error rather than a `pre_exec` failure
/// that surfaces as an `io::Error` after the child is already spawning.
///
/// Rules:
/// - `None` → child inherits caller's cgroup (returns `Ok(None)`).
/// - A leading `/` is tolerated and stripped so `"/workload"` and
///   `"workload"` behave identically.
/// - NUL bytes are rejected (would break `CString` used by the
///   child's `pre_exec` write).
/// - Any `..` component is rejected to prevent the name from
///   escaping the parent cgroup.
/// - Empty names (or names that strip to empty) are rejected so a
///   typo doesn't silently target the parent cgroup itself.
fn resolve_cgroup_path(ctx: &Ctx<'_>, name: Option<&str>) -> Result<Option<PathBuf>> {
    let Some(name) = name else {
        return Ok(None);
    };
    if name.as_bytes().contains(&0) {
        return Err(anyhow!("cgroup name '{name}' contains a NUL byte"));
    }
    let trimmed = name.trim_start_matches('/');
    if trimmed.is_empty() {
        return Err(anyhow!(
            "cgroup name '{name}' is empty or resolves to the parent cgroup"
        ));
    }
    let relative = std::path::Path::new(trimmed);
    if relative
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(anyhow!(
            "cgroup name '{name}' contains '..'; paths must stay within the \
             test's cgroup parent"
        ));
    }
    Ok(Some(ctx.cgroups.parent_path().join(relative)))
}

/// Build a [`Command`] with args, piped stdout/stderr, and an
/// optional `pre_exec` hook that writes the child's PID into the
/// specified cgroup's `cgroup.procs` before `execve`.
///
/// Returns `Err` if the cgroup path cannot be converted to a
/// NUL-terminated C string — `resolve_cgroup_path` already rejects
/// interior NULs, but a `PathBuf` built from `OsStr` can still
/// carry one on exotic platforms, so we check explicitly.
fn build_command(
    binary: &str,
    args: &[String],
    cgroup_path: Option<&std::path::Path>,
) -> Result<std::process::Command> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};

    let mut cmd = Command::new(binary);
    cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped());

    if let Some(cg) = cgroup_path {
        // Precompute the full `.../cgroup.procs` CString on the
        // PARENT side so the pre_exec closure never allocates.
        // Between `fork(2)` and `execve(2)` a multithreaded parent's
        // child can deadlock on any malloc-holding mutex, so no
        // allocation may happen in the closure body. See
        // signal-safety(7).
        let procs_path = cg.join("cgroup.procs");
        let cstr = CString::new(procs_path.as_os_str().as_bytes()).with_context(|| {
            format!(
                "cgroup path {} contains an interior NUL byte",
                procs_path.display(),
            )
        })?;
        unsafe {
            cmd.pre_exec(move || write_pid_to_cgroup(&cstr));
        }
    }
    Ok(cmd)
}

/// Async-signal-safe body of the cgroup-placement `pre_exec` hook:
/// open `cgroup.procs` for write-only/append, render `getpid()` to
/// a stack buffer with no allocation, write it, close the fd.
///
/// # Safety
///
/// Runs between fork and execve. Only async-signal-safe operations
/// are permitted — no `malloc`, no `std::fs`, no `libc::printf`
/// family, no locks (including the jemalloc arena). This function
/// uses only `open`/`write`/`close`/`getpid` (all AS-safe per
/// POSIX.1-2017, 2.4.3) and stack-buffer integer formatting.
///
/// Errors are mapped to `io::Error::from_raw_os_error` so the
/// parent `spawn()` returns an actionable errno rather than the
/// child silently racing through the cgroup-placement step.
fn write_pid_to_cgroup(procs_path: &CString) -> std::io::Result<()> {
    // getpid() is AS-safe. Stack-render the i32 with no alloc —
    // 12 bytes cover i32::MIN's sign + 10 digits + a trailing LF
    // that some cgroup writers expect.
    let pid = unsafe { libc::getpid() };
    let mut buf = [0u8; 12];
    let len = render_pid(pid, &mut buf);

    // O_WRONLY | O_CLOEXEC — the fd must not leak across the
    // upcoming execve(2) in case the binary later opens high-FD
    // numbers.
    let fd = unsafe { libc::open(procs_path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fd` is a valid open file descriptor until we close
    // it below; `buf[..len]` is a live stack buffer of known size.
    let written = unsafe { libc::write(fd, buf.as_ptr() as *const libc::c_void, len) };
    let write_err = if written < 0 {
        Some(std::io::Error::last_os_error())
    } else {
        None
    };
    // Close unconditionally; preserve the write error if one
    // occurred so the parent sees the underlying failure.
    unsafe {
        libc::close(fd);
    }
    if let Some(e) = write_err {
        return Err(e);
    }
    Ok(())
}

/// Render a PID (signed 32-bit) into `buf`, returning the number
/// of bytes written. A trailing LF is appended. No allocation —
/// safe to call between fork and execve.
///
/// `buf` must be at least 12 bytes (worst case: sign + 10 digits +
/// LF).
fn render_pid(pid: libc::pid_t, buf: &mut [u8]) -> usize {
    debug_assert!(buf.len() >= 12);
    // PIDs on Linux are non-negative, but handle the signed range
    // correctly via i64 to avoid a panic on i32::MIN negation.
    let mut n = i64::from(pid);
    let negative = n < 0;
    if negative {
        n = -n;
    }
    // Write digits in reverse, then reverse in place.
    let mut tmp = [0u8; 11];
    let mut i = 0;
    if n == 0 {
        tmp[0] = b'0';
        i = 1;
    } else {
        while n > 0 {
            tmp[i] = b'0' + (n % 10) as u8;
            n /= 10;
            i += 1;
        }
    }
    let mut out = 0;
    if negative {
        buf[out] = b'-';
        out += 1;
    }
    for d in tmp[..i].iter().rev() {
        buf[out] = *d;
        out += 1;
    }
    buf[out] = b'\n';
    out += 1;
    out
}

/// Foreground path: spawn + wait + capture. Used by `.run()`.
fn spawn_and_wait(
    binary: &str,
    args: &[String],
    cgroup_path: Option<&std::path::Path>,
) -> Result<SpawnOutput> {
    let output = build_command(binary, args, cgroup_path)?
        .output()
        .with_context(|| format!("spawn '{binary}'"))?;
    Ok(SpawnOutput {
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        exit_code: output.status.code().unwrap_or(-1),
    })
}

/// Background path: spawn without waiting. Returns the live
/// [`Child`] for [`PayloadHandle`] to manage.
fn spawn_child(
    binary: &str,
    args: &[String],
    cgroup_path: Option<&std::path::Path>,
) -> Result<std::process::Child> {
    build_command(binary, args, cgroup_path)?
        .spawn()
        .with_context(|| format!("spawn '{binary}'"))
}

/// Reap a (possibly already-killed) [`Child`]: wait for it to
/// exit, drain stdout + stderr, return the captured output.
///
/// Sequential stdout-then-stderr reads deadlock when the child
/// fills one pipe buffer (typically 64KiB) while the other is
/// unread — the child blocks on write, the parent blocks on read
/// of the empty pipe. Drain both pipes concurrently via helper
/// threads, mirroring what `std::process::Command::output` does
/// for the foreground path.
fn wait_and_capture(mut child: std::process::Child) -> Result<SpawnOutput> {
    use std::io::Read;
    let stdout_handle = child.stdout.take().map(|mut out| {
        std::thread::spawn(move || -> std::io::Result<String> {
            let mut buf = String::new();
            out.read_to_string(&mut buf)?;
            Ok(buf)
        })
    });
    let stderr_handle = child.stderr.take().map(|mut err| {
        std::thread::spawn(move || -> std::io::Result<String> {
            let mut buf = String::new();
            err.read_to_string(&mut buf)?;
            Ok(buf)
        })
    });
    let status = child.wait().with_context(|| "wait child")?;
    let stdout = match stdout_handle {
        Some(h) => h
            .join()
            .map_err(|_| anyhow!("stdout reader thread panicked"))?
            .with_context(|| "read child stdout")?,
        None => String::new(),
    };
    let stderr = match stderr_handle {
        Some(h) => h
            .join()
            .map_err(|_| anyhow!("stderr reader thread panicked"))?
            .with_context(|| "read child stderr")?,
        None => String::new(),
    };
    Ok(SpawnOutput {
        stdout,
        stderr,
        exit_code: status.code().unwrap_or(-1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgroup::CgroupManager;
    use crate::test_support::{MetricSource, OutputFormat, Polarity, Scheduler};
    use crate::topology::TestTopology;

    // Minimal Ctx builder fixture for tests — no VM boot.
    fn make_ctx<'a>(
        cgroups: &'a CgroupManager,
        topo: &'a TestTopology,
    ) -> crate::scenario::Ctx<'a> {
        crate::scenario::Ctx::builder(cgroups, topo).build()
    }

    const FIO_BINARY: Payload = Payload {
        name: "fio",
        kind: PayloadKind::Binary("fio"),
        output: OutputFormat::Json,
        default_args: &["--output-format=json"],
        default_checks: &[],
        metrics: &[],
    };

    const EEVDF_SCHED_PAYLOAD: Payload = Payload {
        name: "eevdf",
        kind: PayloadKind::Scheduler(&Scheduler::EEVDF),
        output: OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
    };

    #[test]
    fn builder_inherits_default_args() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY);
        assert_eq!(run.args, vec!["--output-format=json"]);
    }

    #[test]
    fn arg_appends() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY)
            .arg("--runtime=30")
            .arg("job.fio");
        assert_eq!(
            run.args,
            vec!["--output-format=json", "--runtime=30", "job.fio"]
        );
    }

    #[test]
    fn clear_args_wipes_defaults() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY)
            .clear_args()
            .arg("--custom");
        assert_eq!(run.args, vec!["--custom"]);
    }

    #[test]
    fn args_method_bulk_appends() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY).args(["--a", "--b", "--c"]);
        assert_eq!(run.args, vec!["--output-format=json", "--a", "--b", "--c"]);
    }

    #[test]
    fn check_and_clear_checks() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY)
            .check(Check::min("iops", 1000.0))
            .check(Check::max("latency", 500.0));
        assert_eq!(run.checks.len(), 2);
        let cleared = PayloadRun::new(&ctx, &FIO_BINARY)
            .clear_checks()
            .check(Check::exit_code_eq(0));
        assert_eq!(cleared.checks.len(), 1);
    }

    #[test]
    fn in_cgroup_stores_name() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY).in_cgroup("fio_cg");
        assert_eq!(run.cgroup.as_deref(), Some("fio_cg"));
    }

    #[test]
    fn resolve_cgroup_path_strips_leading_slash_and_joins() {
        let cgroups = CgroupManager::new("/sys/fs/cgroup/test-parent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        // Leading "/" tolerated, joined under parent.
        let resolved = resolve_cgroup_path(&ctx, Some("/workload"))
            .expect("valid cgroup name")
            .expect("Some(path)");
        assert_eq!(
            resolved,
            std::path::PathBuf::from("/sys/fs/cgroup/test-parent/workload")
        );
        // Same name without leading slash produces the same path.
        let plain = resolve_cgroup_path(&ctx, Some("workload"))
            .expect("valid")
            .expect("Some");
        assert_eq!(resolved, plain);
    }

    #[test]
    fn resolve_cgroup_path_rejects_parent_dir() {
        let cgroups = CgroupManager::new("/sys/fs/cgroup/test-parent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let err = resolve_cgroup_path(&ctx, Some("../escape")).expect_err("'..' must be rejected");
        assert!(format!("{err:#}").contains(".."), "err: {err:#}");
    }

    #[test]
    fn resolve_cgroup_path_rejects_nul_byte() {
        let cgroups = CgroupManager::new("/sys/fs/cgroup/test-parent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let err = resolve_cgroup_path(&ctx, Some("bad\0name")).expect_err("NUL must be rejected");
        assert!(format!("{err:#}").contains("NUL"), "err: {err:#}");
    }

    #[test]
    fn resolve_cgroup_path_rejects_empty_after_strip() {
        let cgroups = CgroupManager::new("/sys/fs/cgroup/test-parent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        // "/" strips to empty — reject so we don't silently target
        // the parent cgroup itself.
        let err = resolve_cgroup_path(&ctx, Some("/")).expect_err("slash-only must be rejected");
        assert!(format!("{err:#}").contains("empty"), "err: {err:#}");
        let err = resolve_cgroup_path(&ctx, Some("")).expect_err("empty must be rejected");
        assert!(format!("{err:#}").contains("empty"), "err: {err:#}");
    }

    #[test]
    fn resolve_cgroup_path_none_passes_through() {
        let cgroups = CgroupManager::new("/sys/fs/cgroup/test-parent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        assert!(resolve_cgroup_path(&ctx, None).unwrap().is_none());
    }

    #[test]
    fn run_rejects_scheduler_kind() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &EEVDF_SCHED_PAYLOAD);
        let err = run.run().unwrap_err();
        assert!(
            format!("{err:#}").contains("scheduler-kind"),
            "err: {err:#}"
        );
    }

    #[test]
    fn evaluate_checks_passes_when_no_checks() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 0,
        };
        let r = evaluate_checks(&[], &pm, "");
        assert!(r.passed);
    }

    #[test]
    fn evaluate_checks_exit_code_mismatch_fails_fast() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 42,
        };
        let checks = [Check::exit_code_eq(0), Check::min("iops", 100.0)];
        let r = evaluate_checks(&checks, &pm, "");
        assert!(!r.passed);
        // exit-code failure short-circuits — only one detail, not
        // a "missing metric" detail from the min check.
        assert_eq!(r.details.len(), 1);
        assert!(
            r.details[0].message.contains("exited with code 42"),
            "details: {:?}",
            r.details
        );
    }

    #[test]
    fn evaluate_checks_exit_code_mismatch_surfaces_stderr() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 1,
        };
        let r = evaluate_checks(&[Check::exit_code_eq(0)], &pm, "fatal: config missing\n");
        assert!(!r.passed);
        assert!(
            r.details[0].message.contains("fatal: config missing"),
            "stderr tail must appear in detail: {:?}",
            r.details,
        );
        assert!(
            r.details[0].message.contains("stderr:"),
            "detail must label the stderr block: {:?}",
            r.details,
        );
    }

    #[test]
    fn evaluate_checks_exit_code_mismatch_without_stderr_stays_terse() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 1,
        };
        let r = evaluate_checks(&[Check::exit_code_eq(0)], &pm, "");
        assert!(!r.passed);
        // Empty stderr → no "stderr:" prefix in the detail.
        assert!(
            !r.details[0].message.contains("stderr:"),
            "empty stderr must not produce a stderr: block: {:?}",
            r.details,
        );
    }

    /// Signal-terminated payloads report `exit_code = -1` because
    /// `std::process::ExitStatus::code()` returns `None` on
    /// signal death and the spawn layer maps that to `-1` (see
    /// `spawn_foreground`). A user who expects the signal-death
    /// case can assert `Check::exit_code_eq(-1)`, and the pre-pass
    /// comparison must pass under exact `i32` equality.
    #[test]
    fn evaluate_checks_exit_code_eq_negative_one_matches_signal_death() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: -1,
        };
        let r = evaluate_checks(&[Check::exit_code_eq(-1)], &pm, "");
        assert!(
            r.passed,
            "exit_code_eq(-1) must pass when exit_code == -1: {:?}",
            r.details,
        );
    }

    /// Symmetric negative case: `Check::exit_code_eq(-1)` against a
    /// CLEAN exit (`exit_code == 0`) must fail and surface the
    /// mismatch with both integers printed so the user sees what
    /// they asked for vs what happened.
    #[test]
    fn evaluate_checks_exit_code_eq_negative_one_fails_on_clean_exit() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::exit_code_eq(-1)], &pm, "");
        assert!(!r.passed);
        let msg = &r.details[0].message;
        assert!(
            msg.contains("exited with code 0"),
            "mismatch detail must cite the actual exit code, got: {msg}"
        );
        assert!(
            msg.contains("-1"),
            "mismatch detail must cite the expected exit code, got: {msg}"
        );
    }

    /// `Check::Range { lo: 100, hi: 50 }` — reversed bounds that make
    /// `lo > hi` — currently fails EVERY finite metric value because
    /// the evaluator tests `(actual < lo) || (actual > hi)` and both
    /// halves are always true for an empty interval. No up-front
    /// validation rejects the reversed construction. Pin the current
    /// behavior so a future constructor-level validation (which
    /// SHOULD exist, since a reversed range is almost certainly a
    /// user error) flips this assertion visibly instead of silently
    /// changing the failure mode.
    #[test]
    fn evaluate_checks_range_reversed_bounds_fails_every_finite_value() {
        use crate::test_support::{Metric, MetricSource, Polarity};
        let reversed = Check::range("iops", 100.0, 50.0);
        for actual in &[0.0, 50.0, 75.0, 100.0, 200.0, -1000.0, 1e9] {
            let pm = PayloadMetrics {
                metrics: vec![Metric {
                    name: "iops".to_string(),
                    value: *actual,
                    polarity: Polarity::HigherBetter,
                    unit: String::new(),
                    source: MetricSource::Json,
                }],
                exit_code: 0,
            };
            let r = evaluate_checks(&[reversed], &pm, "");
            assert!(
                !r.passed,
                "reversed range must fail for value {actual}: {:?}",
                r.details,
            );
            assert!(
                r.details[0].message.contains("outside [100, 50]"),
                "detail must cite the (reversed) declared bounds verbatim, got: {:?}",
                r.details,
            );
        }
    }

    #[test]
    fn stderr_tail_truncates_long_input() {
        // Build >STDERR_TAIL_BYTES of ASCII so char-boundary logic
        // is a no-op and the tail size is deterministic.
        let long: String = "A".repeat(STDERR_TAIL_BYTES + 500);
        let tail = stderr_tail(&long, STDERR_TAIL_BYTES);
        assert!(tail.starts_with("..."));
        // Leading "..." + exactly STDERR_TAIL_BYTES of suffix.
        assert_eq!(tail.len(), STDERR_TAIL_BYTES + 3);
    }

    #[test]
    fn stderr_tail_preserves_short_input() {
        let tail = stderr_tail("short error", STDERR_TAIL_BYTES);
        assert_eq!(tail, "short error");
    }

    /// When `s.len() - max_bytes` lands inside a multi-byte UTF-8
    /// code unit, `stderr_tail` snaps the start index forward to the
    /// next char boundary so the slice operation never panics. This
    /// test uses a 2-byte UTF-8 character ("é") placed at the exact
    /// boundary so a naive `&s[start..]` would slice mid-codepoint.
    #[test]
    fn stderr_tail_snaps_forward_across_multibyte_char_boundary() {
        // "A"*10 + "é" + "B"*10 → 22 bytes total, len 22, "é" = 2 bytes.
        // With max_bytes = 11, start = 22 - 11 = 11. The byte at 11 is
        // the second byte of "é" (non-boundary). The snap-forward
        // advances start to 12, yielding the trailing "B"*10 + preamble.
        let mut s = String::from("A").repeat(10);
        s.push('é');
        s.push_str(&"B".repeat(10));
        let tail = stderr_tail(&s, 11);
        assert!(tail.starts_with("..."));
        // The multi-byte char must have been skipped (advanced off its
        // interior), so the tail begins with ASCII "B"s.
        assert!(
            tail[3..].starts_with('B'),
            "expected snap-forward past 'é', got: {tail:?}"
        );
    }

    /// When the whole multi-byte character sits at the snap-forward
    /// boundary (start lands exactly on its first byte), the
    /// character is preserved intact — no off-by-one that drops its
    /// first byte.
    #[test]
    fn stderr_tail_preserves_multibyte_char_at_exact_boundary() {
        // Build a string so the multi-byte char starts exactly at the
        // snap-forward position. ASCII x10 + "é" (2B) + ASCII x10
        // = 22B. max_bytes = 12 → start = 22-12 = 10, which IS "é"'s
        // first byte (a boundary). No snap happens; "é" is included.
        let mut s = String::from("A").repeat(10);
        s.push('é');
        s.push_str(&"B".repeat(10));
        let tail = stderr_tail(&s, 12);
        assert!(tail.starts_with("..."));
        assert!(
            tail.contains('é'),
            "boundary-aligned multibyte char must survive the tail, got: {tail:?}"
        );
    }

    /// `stderr_tail` is valid UTF-8 regardless of where the
    /// multi-byte character falls. Property-style sanity check
    /// constructing every single-byte offset within a surrounding
    /// multi-byte character and verifying `str::from_utf8` round-trips.
    #[test]
    fn stderr_tail_output_is_always_valid_utf8() {
        // Chinese "好" = 3 bytes (E5 A5 BD); pin it mid-string.
        let s = "xxxxxxxxxx好yyyyyyyyyy"; // 10 + 3 + 10 = 23 bytes
        for max in 1..=s.len() {
            let tail = stderr_tail(s, max);
            // Assertion: `tail` is already a `String`, so it is
            // by construction valid UTF-8; re-rendering it proves
            // no byte-level corruption leaked into the String.
            let _ = tail.as_str();
        }
    }

    #[test]
    fn evaluate_checks_missing_metric_fails_loudly() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 0,
        };
        let checks = [Check::min("iops", 100.0)];
        let r = evaluate_checks(&checks, &pm, "");
        assert!(!r.passed);
        assert!(
            r.details[0].message.contains("not found"),
            "details: {:?}",
            r.details
        );
    }

    #[test]
    fn evaluate_checks_min_below_threshold_fails() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 50.0,
                polarity: Polarity::HigherBetter,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::min("iops", 100.0)], &pm, "");
        assert!(!r.passed);
        assert!(r.details[0].message.contains("below minimum"));
    }

    #[test]
    fn evaluate_checks_max_above_threshold_fails() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "lat".to_string(),
                value: 1000.0,
                polarity: Polarity::LowerBetter,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::max("lat", 500.0)], &pm, "");
        assert!(!r.passed);
        assert!(r.details[0].message.contains("exceeds maximum"));
    }

    #[test]
    fn evaluate_checks_range_out_of_bounds_fails() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "cpu".to_string(),
                value: 150.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::range("cpu", 0.0, 100.0)], &pm, "");
        assert!(!r.passed);
        assert!(r.details[0].message.contains("outside"));
    }

    #[test]
    fn evaluate_checks_exists_missing_fails() {
        let pm = PayloadMetrics {
            metrics: vec![],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::exists("thing")], &pm, "");
        assert!(!r.passed);
    }

    #[test]
    fn evaluate_checks_all_pass_returns_pass() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 5000.0,
                polarity: Polarity::HigherBetter,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(
            &[
                Check::exit_code_eq(0),
                Check::min("iops", 1000.0),
                Check::exists("iops"),
            ],
            &pm,
            "",
        );
        assert!(r.passed);
    }

    /// Multiple checks on the same metric all fire — the evaluator
    /// does not dedup by metric name. Two `Min`s on the same path
    /// either both pass (value >= max threshold) or both fail
    /// (value < one of the thresholds, depending on which is more
    /// restrictive). This test uses a pair where the metric value
    /// (100) is below the second threshold (200) but above the
    /// first (50). The second failure must appear in the details
    /// list — the evaluator must not short-circuit after the first
    /// matching metric check.
    #[test]
    fn evaluate_checks_duplicate_min_on_same_metric_both_evaluated() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 100.0,
                polarity: Polarity::HigherBetter,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(
            &[Check::min("iops", 50.0), Check::min("iops", 200.0)],
            &pm,
            "",
        );
        assert!(!r.passed, "second min must fail");
        assert_eq!(r.details.len(), 1, "only the failing check emits a detail");
        // The passing check produces no detail; only the failing one
        // shows up. The message must reference the 200 threshold.
        assert!(
            r.details[0].message.contains("below minimum 200"),
            "failing check must cite its threshold: {:?}",
            r.details,
        );
    }

    /// Two conflicting checks on the same metric (Min 100 and Max 50)
    /// produce TWO failures in the details list — not one collapsed
    /// failure. Pins the "each check evaluated independently"
    /// invariant so a future optimization doesn't accidentally merge
    /// / dedup.
    #[test]
    fn evaluate_checks_conflicting_checks_on_same_metric_both_report() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "iops".to_string(),
                value: 75.0,
                polarity: Polarity::HigherBetter,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(
            &[
                Check::min("iops", 100.0), // 75 < 100: fail
                Check::max("iops", 50.0),  // 75 > 50: fail
            ],
            &pm,
            "",
        );
        assert!(!r.passed);
        assert_eq!(
            r.details.len(),
            2,
            "both conflicting checks must each emit a detail: {:?}",
            r.details,
        );
    }

    /// `Check::Exists` with a zero-value metric passes. The check is
    /// presence-only — a metric of 0.0 is still present in the
    /// PayloadMetrics map and `pm.get(name).is_some()` returns true.
    /// A naive `pm.get(name).filter(|v| *v != 0.0)` would spuriously
    /// fail here; pin the "exists is sign-agnostic and zero-
    /// friendly" invariant.
    #[test]
    fn evaluate_checks_exists_passes_for_zero_value_metric() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "errors".to_string(),
                value: 0.0,
                polarity: Polarity::LowerBetter,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::exists("errors")], &pm, "");
        assert!(
            r.passed,
            "exists('errors') must pass when metric is 0.0: {:?}",
            r.details,
        );
    }

    /// Negative zero (`-0.0`) also counts as present for
    /// `Check::Exists`. Paranoid pin because f64 `-0.0` surprises
    /// some pattern-matching code (`0.0 == -0.0` but they differ
    /// under `f64::to_bits`).
    #[test]
    fn evaluate_checks_exists_passes_for_negative_zero() {
        let pm = PayloadMetrics {
            metrics: vec![Metric {
                name: "drift".to_string(),
                value: -0.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            }],
            exit_code: 0,
        };
        let r = evaluate_checks(&[Check::exists("drift")], &pm, "");
        assert!(r.passed);
    }

    /// `PayloadRun`'s custom `Debug` impl renders the stable
    /// identity fields — payload name, args/checks lengths, and
    /// cgroup placement — without dumping the `Ctx` pointer. Pins
    /// the output shape so a future rename can't silently drop a
    /// field that debug-printing consumers rely on.
    #[test]
    fn payload_run_debug_renders_identity_fields() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &TRUE_BIN)
            .arg("--foo")
            .arg("--bar")
            .check(Check::exit_code_eq(0))
            .in_cgroup("workers");
        let s = format!("{run:?}");
        assert!(s.contains("PayloadRun"), "prefix: {s}");
        assert!(s.contains("payload:"), "payload field: {s}");
        assert!(s.contains("true_bin"), "payload name: {s}");
        assert!(s.contains("args_len"), "args_len field: {s}");
        assert!(s.contains("checks_len"), "checks_len field: {s}");
        assert!(s.contains("cgroup:"), "cgroup field: {s}");
        // Values: 2 args added (on top of 0 default) + 1 check.
        assert!(s.contains("args_len: 2"), "computed args_len: {s}");
        assert!(s.contains("checks_len: 1"), "computed checks_len: {s}");
        // cgroup is Some("workers"); the debug form of Cow<str>
        // renders as "workers" inside Some(..).
        assert!(s.contains("workers"), "cgroup value: {s}");
        // Must NOT leak the Ctx pointer (no raw-address tokens).
        assert!(
            !s.contains("Ctx {"),
            "Ctx should not appear in PayloadRun Debug: {s}"
        );
    }

    /// Default `PayloadRun` (no args, no checks, no cgroup)
    /// renders sensible zeroes.
    #[test]
    fn payload_run_debug_renders_defaults() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &TRUE_BIN);
        let s = format!("{run:?}");
        assert!(s.contains("args_len: 0"), "default args_len: {s}");
        assert!(s.contains("checks_len: 0"), "default checks_len: {s}");
        assert!(s.contains("cgroup: None"), "default cgroup: {s}");
    }

    #[test]
    fn resolve_polarities_applies_hints() {
        let mut metrics = vec![Metric {
            name: "iops".to_string(),
            value: 100.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        }];
        const HINTED: Payload = Payload {
            name: "p",
            kind: PayloadKind::Binary("p"),
            output: OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[crate::test_support::MetricHint {
                name: "iops",
                polarity: Polarity::HigherBetter,
                unit: "iops",
            }],
        };
        resolve_polarities(&mut metrics, &HINTED);
        assert_eq!(metrics[0].polarity, Polarity::HigherBetter);
        assert_eq!(metrics[0].unit, "iops");
    }

    // -- PayloadHandle + .spawn() tests --

    const TRUE_BIN: Payload = Payload {
        name: "true_bin",
        kind: PayloadKind::Binary("/bin/true"),
        output: crate::test_support::OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
    };

    const FALSE_BIN: Payload = Payload {
        name: "false_bin",
        kind: PayloadKind::Binary("/bin/false"),
        output: crate::test_support::OutputFormat::ExitCode,
        default_args: &[],
        default_checks: &[],
        metrics: &[],
    };

    #[test]
    fn spawn_rejects_scheduler_kind() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &EEVDF_SCHED_PAYLOAD);
        let err = run.spawn().unwrap_err();
        assert!(
            format!("{err:#}").contains("scheduler-kind"),
            "err: {err:#}"
        );
    }

    #[test]
    fn spawn_then_wait_returns_result_and_metrics() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let handle = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true");
        let (result, metrics) = handle.wait().expect("wait");
        assert!(result.passed);
        assert_eq!(metrics.exit_code, 0);
    }

    #[test]
    fn spawn_then_kill_returns_collected_output() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        // /bin/sleep runs for a while; .kill() terminates it.
        const SLEEPER: Payload = Payload {
            name: "sleeper",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: crate::test_support::OutputFormat::ExitCode,
            default_args: &["60"],
            default_checks: &[],
            metrics: &[],
        };
        let handle = PayloadRun::new(&ctx, &SLEEPER)
            .spawn()
            .expect("spawn sleep");
        let (_result, metrics) = handle.kill().expect("kill+collect");
        // Killed process produces a non-zero exit (SIGKILL -> None
        // status code, wait_and_capture maps to -1).
        assert_ne!(metrics.exit_code, 0);
    }

    #[test]
    fn spawn_try_wait_returns_none_while_running() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        const SLEEPER: Payload = Payload {
            name: "sleeper3",
            kind: PayloadKind::Binary("/bin/sleep"),
            output: crate::test_support::OutputFormat::ExitCode,
            default_args: &["60"],
            default_checks: &[],
            metrics: &[],
        };
        let mut handle = PayloadRun::new(&ctx, &SLEEPER)
            .spawn()
            .expect("spawn sleep");
        // Not yet exited.
        assert!(handle.try_wait().expect("try_wait").is_none());
        // Cleanup — kill so Drop warning doesn't fire.
        let _ = handle.kill();
    }

    #[test]
    fn spawn_try_wait_returns_some_after_exit() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let mut handle = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true");
        // /bin/true exits quickly. Poll a few times.
        let mut result = None;
        for _ in 0..100 {
            if let Some(r) = handle.try_wait().expect("try_wait") {
                result = Some(r);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let (r, metrics) = result.expect("try_wait eventually returns Some");
        assert!(r.passed);
        assert_eq!(metrics.exit_code, 0);
    }

    #[test]
    fn spawn_false_binary_produces_failing_exit_code() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let handle = PayloadRun::new(&ctx, &FALSE_BIN)
            .spawn()
            .expect("spawn /bin/false");
        let (_result, metrics) = handle.wait().expect("wait");
        assert_ne!(metrics.exit_code, 0);
    }

    #[test]
    fn resolve_polarities_leaves_unhinted_alone() {
        let mut metrics = vec![Metric {
            name: "no_hint".to_string(),
            value: 1.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        }];
        resolve_polarities(&mut metrics, &FIO_BINARY);
        assert_eq!(metrics[0].polarity, Polarity::Unknown);
        assert_eq!(metrics[0].unit, "");
    }

    // -- Builder-composition + evaluator-coverage regression tests --

    #[test]
    fn evaluate_checks_three_failing_checks_produce_three_details() {
        // Exit-code check passes (0 == 0), so pre-pass does not
        // short-circuit; all three metric checks fail and each must
        // contribute its own AssertDetail — regression guard
        // against detail dedup/overwrite bugs.
        let pm = PayloadMetrics {
            metrics: vec![
                Metric {
                    name: "iops".to_string(),
                    value: 10.0,
                    polarity: Polarity::HigherBetter,
                    unit: String::new(),
                    source: MetricSource::Json,
                },
                Metric {
                    name: "lat".to_string(),
                    value: 900.0,
                    polarity: Polarity::LowerBetter,
                    unit: String::new(),
                    source: MetricSource::Json,
                },
                Metric {
                    name: "cpu".to_string(),
                    value: 200.0,
                    polarity: Polarity::Unknown,
                    unit: String::new(),
                    source: MetricSource::Json,
                },
            ],
            exit_code: 0,
        };
        let checks = [
            Check::exit_code_eq(0),
            Check::min("iops", 1000.0),
            Check::max("lat", 100.0),
            Check::range("cpu", 0.0, 100.0),
        ];
        let r = evaluate_checks(&checks, &pm, "");
        assert!(!r.passed);
        assert_eq!(
            r.details.len(),
            3,
            "expected one detail per failed metric check, got: {:?}",
            r.details,
        );
        // Each check's message must surface — not an aggregate or
        // a deduped first-only line.
        assert!(r.details.iter().any(|d| d.message.contains("iops")));
        assert!(r.details.iter().any(|d| d.message.contains("lat")));
        assert!(r.details.iter().any(|d| d.message.contains("cpu")));
    }

    #[test]
    fn arg_then_clear_args_then_arg_yields_only_the_final_arg() {
        // clear_args() wipes EVERYTHING — the default_args AND any
        // previously-appended .arg(...) — and subsequent .arg(...)
        // calls start from empty. Regression guard for the
        // "clear_args truncates, arg appends" contract.
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY)
            .arg("--x")
            .clear_args()
            .arg("--y");
        assert_eq!(run.args, vec!["--y"]);
    }

    #[test]
    fn default_checks_are_inherited_by_new_builder() {
        // Payload.default_checks are the starting check list: they
        // MUST be present on a fresh PayloadRun before any runtime
        // .check() calls. `.check` appends on top, `.clear_checks`
        // wipes them.
        const CHECKED: Payload = Payload {
            name: "checked",
            kind: PayloadKind::Binary("checked"),
            output: OutputFormat::ExitCode,
            default_args: &[],
            default_checks: &[Check::exit_code_eq(0), Check::min("iops", 500.0)],
            metrics: &[],
        };
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);

        // Fresh builder inherits both default checks in order.
        let fresh = PayloadRun::new(&ctx, &CHECKED);
        assert_eq!(fresh.checks.len(), 2);
        assert!(matches!(fresh.checks[0], Check::ExitCodeEq(0)));
        assert!(matches!(
            fresh.checks[1],
            Check::Min { value, .. } if value == 500.0,
        ));

        // Appending preserves defaults and adds on top.
        let appended = PayloadRun::new(&ctx, &CHECKED).check(Check::exists("latency"));
        assert_eq!(appended.checks.len(), 3);

        // Clearing wipes defaults too.
        let cleared = PayloadRun::new(&ctx, &CHECKED).clear_checks();
        assert!(cleared.checks.is_empty());
    }

    // -- render_pid (async-signal-safe helper) --

    #[test]
    fn render_pid_zero() {
        let mut buf = [0u8; 12];
        let n = render_pid(0, &mut buf);
        assert_eq!(&buf[..n], b"0\n");
    }

    #[test]
    fn render_pid_typical_linux_pid() {
        let mut buf = [0u8; 12];
        let n = render_pid(12345, &mut buf);
        assert_eq!(&buf[..n], b"12345\n");
    }

    #[test]
    fn render_pid_i32_max() {
        let mut buf = [0u8; 12];
        let n = render_pid(i32::MAX, &mut buf);
        assert_eq!(&buf[..n], b"2147483647\n");
    }

    #[test]
    fn render_pid_i32_min_no_panic() {
        // i32::MIN cannot be negated within i32 — the helper must
        // promote to i64 to handle this without panicking. Linux
        // does not emit negative PIDs, but the helper is defensive.
        let mut buf = [0u8; 12];
        let n = render_pid(i32::MIN, &mut buf);
        assert_eq!(&buf[..n], b"-2147483648\n");
    }

    #[test]
    fn in_cgroup_accepts_static_str_zero_alloc() {
        // Static &'static str goes in as Cow::Borrowed; no heap
        // allocation happens for the common case of a const cgroup
        // name. Regression guard for the Cow<'static, str> API shape.
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let run = PayloadRun::new(&ctx, &FIO_BINARY).in_cgroup("workload");
        match &run.cgroup {
            Some(Cow::Borrowed(s)) => assert_eq!(*s, "workload"),
            other => panic!("expected Cow::Borrowed for &'static str input, got {other:?}"),
        }
    }

    #[test]
    fn in_cgroup_accepts_owned_string() {
        // Owned String goes in as Cow::Owned; the builder must not
        // require the caller to convert themselves.
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let name = String::from("dynamic");
        let run = PayloadRun::new(&ctx, &FIO_BINARY).in_cgroup(name);
        match &run.cgroup {
            Some(Cow::Owned(s)) => assert_eq!(s, "dynamic"),
            other => panic!("expected Cow::Owned for String input, got {other:?}"),
        }
    }

    /// Host-side decode of a guest-emitted `PayloadMetrics` JSON
    /// body must round-trip exactly — the SHM transport only carries
    /// bytes, and a schema drift between emit-side (serde_json on a
    /// `PayloadMetrics`) and drain-side (serde_json::from_slice) would
    /// silently drop metrics from the sidecar.
    #[test]
    fn payload_metrics_shm_payload_json_round_trip() {
        let emit = PayloadMetrics {
            metrics: vec![
                Metric {
                    name: "jobs.0.read.iops".to_string(),
                    value: 12345.0,
                    polarity: Polarity::HigherBetter,
                    unit: "iops".to_string(),
                    source: MetricSource::Json,
                },
                Metric {
                    name: "lat_ns".to_string(),
                    value: 500.0,
                    polarity: Polarity::LowerBetter,
                    unit: "ns".to_string(),
                    source: MetricSource::LlmExtract,
                },
            ],
            exit_code: 0,
        };
        let bytes = serde_json::to_vec(&emit).expect("serialize PayloadMetrics");
        let decoded: PayloadMetrics =
            serde_json::from_slice(&bytes).expect("decode PayloadMetrics from JSON bytes");
        assert_eq!(decoded.exit_code, emit.exit_code);
        assert_eq!(decoded.metrics.len(), emit.metrics.len());
        for (a, b) in decoded.metrics.iter().zip(emit.metrics.iter()) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.value, b.value);
            assert_eq!(a.polarity, b.polarity);
            assert_eq!(a.unit, b.unit);
            assert_eq!(a.source, b.source);
        }
    }

    /// Hinted metrics pick up polarity + unit from the payload's
    /// declared MetricHints regardless of declaration order. Also
    /// pins that resolve_polarities leaves unhinted metrics at
    /// Polarity::Unknown / empty unit — the non-over-applying
    /// invariant the prior linear scan had.
    #[test]
    fn resolve_polarities_applies_hints_by_name_lookup() {
        use crate::test_support::{Metric, MetricHint, MetricSource, Polarity};
        static PAYLOAD: crate::test_support::Payload = crate::test_support::Payload {
            name: "hinted",
            kind: crate::test_support::PayloadKind::Binary("x"),
            output: crate::test_support::OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            // Out-of-order with the metric slice below so a naive
            // position-based lookup would miss.
            metrics: &[
                MetricHint {
                    name: "lat_ns",
                    polarity: Polarity::LowerBetter,
                    unit: "ns",
                },
                MetricHint {
                    name: "iops",
                    polarity: Polarity::HigherBetter,
                    unit: "iops",
                },
            ],
        };
        let mut ms = vec![
            Metric {
                name: "iops".into(),
                value: 1.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            },
            Metric {
                name: "unhinted".into(),
                value: 2.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            },
            Metric {
                name: "lat_ns".into(),
                value: 3.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            },
        ];
        resolve_polarities(&mut ms, &PAYLOAD);
        // iops hinted → HigherBetter / "iops".
        assert_eq!(ms[0].polarity, Polarity::HigherBetter);
        assert_eq!(ms[0].unit, "iops");
        // unhinted stays Unknown + empty.
        assert_eq!(ms[1].polarity, Polarity::Unknown);
        assert_eq!(ms[1].unit, "");
        // lat_ns hinted → LowerBetter / "ns".
        assert_eq!(ms[2].polarity, Polarity::LowerBetter);
        assert_eq!(ms[2].unit, "ns");
    }

    /// Empty hints or empty metrics are a fast-path — the HashMap
    /// build is skipped entirely. Pins the no-op invariant so a
    /// regression can't accidentally materialize an empty map for
    /// zero metrics on every hot-path call.
    /// When the payload declares two MetricHints with the same
    /// name, the HashMap build keeps the LAST insertion. The test
    /// pins that behavior so a future switch to a multimap or to
    /// first-wins semantics surfaces here. First-wins would be
    /// surprising: users who copy-paste a hint to tweak it expect
    /// the new value.
    #[test]
    fn resolve_polarities_duplicate_hint_names_last_wins() {
        use crate::test_support::{Metric, MetricHint, MetricSource, Polarity};
        static PAYLOAD: crate::test_support::Payload = crate::test_support::Payload {
            name: "dup_hints",
            kind: crate::test_support::PayloadKind::Binary("x"),
            output: crate::test_support::OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[
                MetricHint {
                    name: "iops",
                    polarity: Polarity::HigherBetter,
                    unit: "iops",
                },
                MetricHint {
                    name: "iops",
                    polarity: Polarity::LowerBetter,
                    unit: "overridden",
                },
            ],
        };
        let mut ms = vec![Metric {
            name: "iops".into(),
            value: 1.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        }];
        resolve_polarities(&mut ms, &PAYLOAD);
        // Second declaration wins (HashMap last-insertion semantics).
        assert_eq!(ms[0].polarity, Polarity::LowerBetter);
        assert_eq!(ms[0].unit, "overridden");
    }

    /// When the metric slice has duplicate names (e.g. a payload
    /// emitting the same dotted path twice in one run), the hint
    /// is applied to each occurrence. Each is a distinct Metric
    /// value in the sidecar; both must carry the hinted polarity +
    /// unit so downstream regression reports are consistent.
    #[test]
    fn resolve_polarities_duplicate_metric_names_each_gets_hint() {
        use crate::test_support::{Metric, MetricHint, MetricSource, Polarity};
        static PAYLOAD: crate::test_support::Payload = crate::test_support::Payload {
            name: "dup_metrics",
            kind: crate::test_support::PayloadKind::Binary("x"),
            output: crate::test_support::OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[MetricHint {
                name: "iops",
                polarity: Polarity::HigherBetter,
                unit: "iops",
            }],
        };
        let mut ms = vec![
            Metric {
                name: "iops".into(),
                value: 1.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            },
            Metric {
                name: "iops".into(),
                value: 2.0,
                polarity: Polarity::Unknown,
                unit: String::new(),
                source: MetricSource::Json,
            },
        ];
        resolve_polarities(&mut ms, &PAYLOAD);
        for m in &ms {
            assert_eq!(m.polarity, Polarity::HigherBetter);
            assert_eq!(m.unit, "iops");
        }
    }

    #[test]
    fn resolve_polarities_empty_inputs_are_noop() {
        use crate::test_support::{Metric, MetricSource, Polarity};
        static NO_HINTS: crate::test_support::Payload = crate::test_support::Payload {
            name: "no_hints",
            kind: crate::test_support::PayloadKind::Binary("x"),
            output: crate::test_support::OutputFormat::Json,
            default_args: &[],
            default_checks: &[],
            metrics: &[],
        };
        let mut ms = vec![Metric {
            name: "anything".into(),
            value: 1.0,
            polarity: Polarity::Unknown,
            unit: String::new(),
            source: MetricSource::Json,
        }];
        resolve_polarities(&mut ms, &NO_HINTS);
        assert_eq!(ms[0].polarity, Polarity::Unknown);
        assert_eq!(ms[0].unit, "");

        // Empty metrics list — also a fast-path no-op, just pin it
        // doesn't panic / over-allocate.
        let mut empty: Vec<Metric> = vec![];
        resolve_polarities(&mut empty, &NO_HINTS);
        assert!(empty.is_empty());
    }

    #[test]
    fn emit_payload_metrics_to_shm_no_panic_without_shm() {
        let pm = PayloadMetrics {
            metrics: Vec::new(),
            exit_code: 0,
        };
        emit_payload_metrics_to_shm(&pm);
    }

    // -- PayloadHandle double-consume returns Err, not panic --

    /// After `try_wait()` returns `Ok(Some(..))` (terminal branch
    /// that takes the child), a subsequent `try_wait()` on the same
    /// handle must return `Err` instead of panicking. Previously
    /// the implementation unwrapped `self.child.as_mut()` with a
    /// panicking `.expect(...)`.
    #[test]
    fn try_wait_after_terminal_returns_err() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let mut handle = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true");
        // First terminal: /bin/true exits immediately; spin until
        // try_wait returns Some.
        for _ in 0..100 {
            if handle.try_wait().expect("try_wait").is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Second call must not panic — expect Err describing the
        // consumed state.
        let err = handle
            .try_wait()
            .expect_err("second try_wait on consumed handle must be Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already consumed") && msg.contains("true_bin"),
            "error must name the handle + misuse, got: {msg}"
        );
    }

    /// Calling `wait()` after `try_wait()` has consumed the child
    /// must Err rather than panic. Test pairs with
    /// `try_wait_after_terminal_returns_err`: same state, different
    /// terminal method.
    #[test]
    fn wait_after_try_wait_consumed_returns_err() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let mut handle = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true");
        for _ in 0..100 {
            if handle.try_wait().expect("try_wait").is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Child is now taken; wait() must return Err, not panic.
        let err = handle
            .wait()
            .expect_err("wait() on consumed handle must return Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already consumed") && msg.contains("true_bin"),
            "error must name the handle + misuse, got: {msg}"
        );
    }

    /// Calling `kill()` after `try_wait()` has consumed the child
    /// must Err rather than panic.
    #[test]
    fn kill_after_try_wait_consumed_returns_err() {
        let cgroups = CgroupManager::new("/nonexistent");
        let topo = TestTopology::synthetic(4, 1);
        let ctx = make_ctx(&cgroups, &topo);
        let mut handle = PayloadRun::new(&ctx, &TRUE_BIN)
            .spawn()
            .expect("spawn /bin/true");
        for _ in 0..100 {
            if handle.try_wait().expect("try_wait").is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let err = handle
            .kill()
            .expect_err("kill() on consumed handle must return Err");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("already consumed") && msg.contains("true_bin"),
            "error must name the handle + misuse, got: {msg}"
        );
    }
}
